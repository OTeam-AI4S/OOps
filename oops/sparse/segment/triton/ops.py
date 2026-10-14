from __future__ import annotations

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.core._math import _next_power_of_two
from oops.runtime import (
    CUDA_MAX_GRID_X,
    CUDA_MAX_GRID_YZ,
    HALF_DTYPES,
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
)
from oops.runtime import needs_grad as tensors_need_grad

from ...csr import canonical_segment_ids
from ...csr.triton import kernel as csr_kernel
from ...csr.triton.backend import (
    finish_long_reduction,
    flat_features,
    fused_reduction_csr,
    prepare_long_reduction,
    reduction_part_count,
    resolve_execution_plan,
    rows_per_edge,
    schedule_supports,
    sum_csr,
    supports_reduction_layout,
)
from ...csr.triton.schedule import LONG_BLOCK_EDGES, LONG_ROW_EDGES, SegmentPlan, adaptive_reduction_geometry
from . import kernel
from .profiles import SEGMENT_MAX_SCHEDULE, SEGMENT_SOFTMAX_ADAPTIVE, SEGMENT_SOFTMAX_SCHEDULE, SEGMENT_SUM_SCHEDULE
from .softmax_schedule import adaptive_geometry


def _plain_segment_call(values: Tensor, row_offsets: Tensor) -> bool:
    operands = (values, row_offsets)
    return (
        not is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
    )


def supports_segment_sum(values: Tensor, row_offsets: Tensor, out_segments: int) -> bool:

    return supports_reduction_layout(values, row_offsets, out_segments) and schedule_supports(
        SEGMENT_SUM_SCHEDULE, values, out_segments
    )


def triton_segment_sum(
    values: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    # Keep schema SymInt extents symbolic; int() would specialize them.
    if tensors_need_grad(values):
        if _plain_segment_call(values, row_offsets):
            return _SegmentSum.apply(values, row_offsets, segment_ids, out_segments, inferred_segments)
        return _segment_sum_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    compiling = is_compiling()
    value_type = type(values)
    external_fake = (
        not compiling and value_type is not Tensor and value_type is not torch.nn.Parameter and is_external_fake(values)
    )
    if compiling or external_fake:
        return _segment_sum_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    return sum_csr(values, row_offsets, out_segments, inferred_segments)


@torch.library.custom_op("ooperators::_segment_sum_csr_forward", mutates_args=(), device_types="cuda")
def _segment_sum_csr_forward(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del segment_ids
    plan = resolve_execution_plan(SEGMENT_SUM_SCHEDULE, values, out_segments)
    return sum_csr(values, row_offsets, out_segments, inferred_segments, plan=plan)


@torch.library.register_fake(_segment_sum_csr_forward)
def _segment_sum_csr_forward_fake(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del row_offsets, segment_ids, inferred_segments
    return torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)


def _segment_sum_csr_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor | None, int, int],
    output: Tensor,
) -> None:
    values, row_offsets, segment_ids, _out_segments, inferred_segments = inputs
    del output
    ctx.save_for_backward(row_offsets, segment_ids)
    ctx.num_edges = values.shape[0]
    ctx.inferred_segments = inferred_segments


def _segment_sum_csr_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None, None]:
    row_offsets, cached_segment_ids = ctx.saved_tensors
    if not torch.is_grad_enabled():
        return (
            _run_segment_backward(
                grad_out, grad_out, grad_out, row_offsets, ctx.num_edges, ctx.inferred_segments, "sum"
            ),
            None,
            None,
            None,
            None,
        )
    segment_ids = canonical_segment_ids(row_offsets, ctx.num_edges, cached_segment_ids)
    grad_values = rows_per_edge(grad_out, segment_ids, ctx.inferred_segments)
    return grad_values, None, None, None, None


torch.library.register_autograd(
    _segment_sum_csr_forward,
    _segment_sum_csr_forward_autograd_backward,
    setup_context=_segment_sum_csr_forward_setup_context,
)


class _SegmentSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_offsets, segment_ids, out_segments, inferred_segments):
        output = sum_csr(values, row_offsets, out_segments, inferred_segments)
        _segment_sum_csr_forward_setup_context(
            ctx, (values, row_offsets, segment_ids, out_segments, inferred_segments), output
        )
        return output

    backward = staticmethod(_segment_sum_csr_forward_autograd_backward)


def supports_segment_max(values: Tensor, row_offsets: Tensor, out_segments: int) -> bool:

    return supports_reduction_layout(values, row_offsets, out_segments) and schedule_supports(
        SEGMENT_MAX_SCHEDULE, values, out_segments
    )


def triton_segment_max(
    values: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    if tensors_need_grad(values):
        if _plain_segment_call(values, row_offsets):
            return _SegmentMax.apply(values, row_offsets, segment_ids, out_segments, inferred_segments)
        return _segment_max_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    compiling = is_compiling()
    value_type = type(values)
    external_fake = (
        not compiling and value_type is not Tensor and value_type is not torch.nn.Parameter and is_external_fake(values)
    )
    if compiling or external_fake:
        return _segment_max_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    return _launch_segment_max(values, row_offsets, out_segments, inferred_segments)


@torch.library.custom_op("ooperators::_segment_max_csr_forward", mutates_args=(), device_types="cuda")
def _segment_max_csr_forward(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del segment_ids
    plan = resolve_execution_plan(SEGMENT_MAX_SCHEDULE, values, out_segments)
    return _launch_segment_max(values, row_offsets, out_segments, inferred_segments, plan=plan)


def _launch_segment_max(
    values: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    *,
    plan: SegmentPlan | None = None,
) -> Tensor:
    fused = fused_reduction_csr(values, row_offsets, out_segments, inferred_segments, "max")
    if fused is not None:
        return fused
    with cuda_device_context(values.device):
        values = values.resolve_neg()
        row_offsets = row_offsets.resolve_neg()
        features = flat_features(values)
        if plan is None:
            config = SEGMENT_MAX_SCHEDULE.config
            grid = (
                out_segments,
                (features + config.block_features - 1) // config.block_features,
            )
            block_features = min(config.block_features, _next_power_of_two(features))
            num_warps = config.num_warps
        else:
            grid = plan.grid
            block_features = plan.block_features
            num_warps = plan.num_warps
        out = torch.empty((out_segments, features), device=values.device, dtype=values.dtype)
        parts = reduction_part_count(values, values.shape[0], inferred_segments, out_segments, block_features)
        accumulation_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        partial = (
            torch.empty((out_segments * parts, features), device=values.device, dtype=accumulation_dtype)
            if parts > 1
            else out
        )
        launch_grid = (out_segments * parts, grid[1])
        adaptive = (
            adaptive_reduction_geometry(values.shape[0], inferred_segments, out_segments, block_features)
            if parts == 1
            else None
        )
        if adaptive is not None:
            prepared = prepare_long_reduction(values, out_segments, inferred_segments, block_features)
            if prepared is not None:
                descriptors, long_partial, buckets, long_parts = prepared
                kernel._segment_max_forward_adaptive.run(
                    values,
                    row_offsets,
                    out,
                    inferred_segments,
                    out_segments,
                    values.shape[0],
                    features,
                    values.dtype == torch.float64,
                    block_features,
                    SEGMENT_MAX_SCHEDULE.config.max_block_edges,
                    min(8, block_features),
                    LONG_BLOCK_EDGES,
                    LONG_ROW_EDGES,
                    skip_long_rows=True,
                    long_descriptors_ptr=descriptors,
                    num_warps=num_warps,
                    grid=(out_segments + buckets, grid[1]),
                    warmup=False,
                )
                finish_long_reduction(
                    values, descriptors, long_partial, out, buckets, long_parts, block_features, num_warps, "max"
                )
                return out.reshape(out_segments, *values.shape[1:])
            programs, long_features = adaptive
            kernel._segment_max_forward_adaptive.run(
                values,
                row_offsets,
                out,
                inferred_segments,
                out_segments,
                values.shape[0],
                features,
                values.dtype == torch.float64,
                block_features,
                SEGMENT_MAX_SCHEDULE.config.max_block_edges,
                long_features,
                LONG_BLOCK_EDGES,
                LONG_ROW_EDGES,
                num_warps=num_warps,
                grid=(programs, grid[1]),
                warmup=False,
            )
            return out.reshape(out_segments, *values.shape[1:])
        kernel._segment_max_csr_forward.run(
            values,
            row_offsets,
            partial,
            inferred_segments,
            values.shape[0],
            features,
            values.dtype == torch.float64,
            block_features,
            SEGMENT_MAX_SCHEDULE.config.max_block_edges,
            parts,
            num_warps=num_warps,
            grid=launch_grid,
            warmup=False,
        )
        if parts > 1:
            csr_kernel.merge_partials.run(
                partial,
                out,
                features,
                "max",
                parts,
                block_features,
                num_warps=num_warps,
                grid=grid,
                warmup=False,
            )
        return out.reshape(out_segments, *values.shape[1:])


@torch.library.register_fake(_segment_max_csr_forward)
def _segment_max_csr_forward_fake(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del row_offsets, segment_ids, inferred_segments
    return torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)


def _segment_max_csr_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor | None, int, int],
    output: Tensor,
) -> None:
    values, row_offsets, segment_ids, _out_segments, inferred_segments = inputs
    ctx.save_for_backward(values, row_offsets, segment_ids, output)
    ctx.inferred_segments = inferred_segments


def _segment_max_csr_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None, None]:

    values, row_offsets, cached_segment_ids, maxima = ctx.saved_tensors
    if not torch.is_grad_enabled():
        return (
            _run_segment_backward(grad_out, values, maxima, row_offsets, values.shape[0], ctx.inferred_segments, "max"),
            None,
            None,
            None,
            None,
        )
    segment_ids = canonical_segment_ids(row_offsets, values.shape[0], cached_segment_ids)
    attained = values == rows_per_edge(maxima, segment_ids, ctx.inferred_segments)
    count_dtype = torch.float32 if values.dtype in HALF_DTYPES else values.dtype
    attaining = _segment_sum_csr_forward(
        attained.to(count_dtype),
        row_offsets,
        segment_ids,
        ctx.inferred_segments,
        ctx.inferred_segments,
    )
    # Zero tie counts occur only in unselected branches.
    divisor = rows_per_edge(attaining, segment_ids, ctx.inferred_segments).clamp(min=1)
    upstream = rows_per_edge(grad_out, segment_ids, ctx.inferred_segments)
    grad_values = torch.where(attained, upstream / divisor, torch.zeros_like(upstream))
    return grad_values.to(values.dtype), None, None, None, None


torch.library.register_autograd(
    _segment_max_csr_forward,
    _segment_max_csr_forward_autograd_backward,
    setup_context=_segment_max_csr_forward_setup_context,
)


class _SegmentMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_offsets, segment_ids, out_segments, inferred_segments):
        output = _launch_segment_max(values, row_offsets, out_segments, inferred_segments)
        _segment_max_csr_forward_setup_context(
            ctx, (values, row_offsets, segment_ids, out_segments, inferred_segments), output
        )
        return output

    backward = staticmethod(_segment_max_csr_forward_autograd_backward)


def supports_segment_softmax(logits: Tensor, row_offsets: Tensor, inferred_segments: int) -> bool:

    return supports_reduction_layout(logits, row_offsets, inferred_segments) and schedule_supports(
        SEGMENT_SOFTMAX_SCHEDULE, logits, inferred_segments
    )


def triton_segment_softmax(
    logits: Tensor,
    row_offsets: Tensor,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    if tensors_need_grad(logits):
        if _plain_segment_call(logits, row_offsets):
            return _SegmentSoftmax.apply(logits, row_offsets, segment_ids, inferred_segments)
        return _segment_softmax_csr_forward(logits, row_offsets, segment_ids, inferred_segments)
    compiling = is_compiling()
    logit_type = type(logits)
    external_fake = (
        not compiling and logit_type is not Tensor and logit_type is not torch.nn.Parameter and is_external_fake(logits)
    )
    if compiling or external_fake:
        return _segment_softmax_csr_forward(logits, row_offsets, segment_ids, inferred_segments)
    return _launch_segment_softmax(logits, row_offsets, inferred_segments)


@torch.library.custom_op("ooperators::_segment_softmax_csr_forward", mutates_args=(), device_types="cuda")
def _segment_softmax_csr_forward(
    logits: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
) -> Tensor:
    del segment_ids
    plan = resolve_execution_plan(SEGMENT_SOFTMAX_SCHEDULE, logits, inferred_segments)
    return _launch_segment_softmax(logits, row_offsets, inferred_segments, plan=plan)


def _launch_long_softmax(
    values: Tensor,
    grad_out: Tensor,
    out: Tensor,
    row_offsets: Tensor,
    segments: int,
    block_features: int,
    num_warps: int,
    backward: bool,
) -> bool:
    values = values.resolve_neg().contiguous()
    grad_out = grad_out.resolve_neg().contiguous()
    row_offsets = row_offsets.resolve_neg().contiguous()
    features = flat_features(values)
    config = SEGMENT_SOFTMAX_ADAPTIVE
    buckets = values.shape[0] // config.long_row_edges
    long_features = min(config.block_features, block_features)
    feature_tiles = (features + long_features - 1) // long_features
    parts = 128
    itemsize = 8 if values.dtype == torch.float64 else 4
    statistics = 1 if backward else 2
    while parts > 1 and (
        buckets * parts > CUDA_MAX_GRID_X
        or buckets * parts * features * itemsize * statistics + buckets * 16 > 16 << 20
    ):
        parts //= 2
    if parts == 1 or feature_tiles > CUDA_MAX_GRID_YZ:
        return False
    partial = torch.empty(
        (buckets * parts * statistics, features),
        device=values.device,
        dtype=torch.float64 if values.dtype == torch.float64 else torch.float32,
    )
    descriptors = torch.empty((buckets, 2), device=values.device, dtype=torch.int64)
    kernel._segment_softmax_adaptive.run(
        values,
        grad_out,
        out,
        row_offsets,
        values.shape[0],
        segments,
        features,
        values.dtype == torch.float64,
        backward,
        block_features,
        SEGMENT_SOFTMAX_SCHEDULE.config.max_block_edges,
        long_features,
        config.block_edges,
        config.long_row_edges,
        skip_long_rows=True,
        long_descriptors_ptr=descriptors,
        pack_short_rows=64 < features <= 128,
        num_warps=num_warps,
        enable_fp_fusion=not backward,
        grid=(segments + buckets, (features + block_features - 1) // block_features),
        warmup=False,
    )
    short_layout = values.shape[0] <= segments * 32
    grid = (min(parts, 32) if short_layout and features > 8 else parts, feature_tiles)
    args = (values, grad_out, descriptors, partial)
    common = (
        values.shape[0],
        segments,
        features,
        values.dtype == torch.float64,
        backward,
        parts,
        long_features,
        config.block_edges,
        config.long_row_edges,
    )
    kernel._segment_softmax_long_statistics.run(
        *args,
        *common,
        num_warps=num_warps,
        enable_fp_fusion=not backward,
        grid=grid,
        warmup=False,
    )
    kernel._segment_softmax_long_finish.run(
        *args,
        out,
        *common,
        num_warps=num_warps,
        enable_fp_fusion=not backward,
        grid=grid,
        warmup=False,
    )
    return True


def _launch_segment_softmax(
    logits: Tensor,
    row_offsets: Tensor,
    inferred_segments: int,
    *,
    plan: SegmentPlan | None = None,
) -> Tensor:
    with cuda_device_context(logits.device):
        logits = logits.resolve_neg()
        row_offsets = row_offsets.resolve_neg()
        features = flat_features(logits)
        if plan is None:
            config = SEGMENT_SOFTMAX_SCHEDULE.config
            grid = (
                inferred_segments,
                (features + config.block_features - 1) // config.block_features,
            )
            block_features = min(config.block_features, _next_power_of_two(features))
            num_warps = config.num_warps
        else:
            grid = plan.grid
            block_features = plan.block_features
            num_warps = plan.num_warps
        out = torch.empty_like(logits.reshape(logits.shape[0], features))
        parts = reduction_part_count(logits, logits.shape[0], inferred_segments, inferred_segments, block_features)
        adaptive = (
            adaptive_geometry(SEGMENT_SOFTMAX_ADAPTIVE, logits.shape[0], inferred_segments, features, block_features)
            if parts == 1
            else None
        )
        if (
            parts == 1
            and logits.shape[0] >= SEGMENT_SOFTMAX_ADAPTIVE.long_row_edges
            and inferred_segments > 0
            and _launch_long_softmax(
                logits, logits, out, row_offsets, inferred_segments, block_features, num_warps, False
            )
        ):
            return out.reshape_as(logits)
        if adaptive is not None:
            adaptive_work, long_features = adaptive
            kernel._segment_softmax_adaptive.run(
                logits,
                logits,
                out,
                row_offsets,
                logits.shape[0],
                inferred_segments,
                features,
                logits.dtype == torch.float64,
                False,
                block_features,
                SEGMENT_SOFTMAX_SCHEDULE.config.max_block_edges,
                long_features,
                SEGMENT_SOFTMAX_ADAPTIVE.block_edges,
                SEGMENT_SOFTMAX_ADAPTIVE.long_row_edges,
                num_warps=num_warps,
                grid=(adaptive_work, grid[1]),
                warmup=False,
            )
            return out.reshape_as(logits)
        accumulation_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        partial = (
            torch.empty((2, inferred_segments * parts, features), device=logits.device, dtype=accumulation_dtype)
            if parts > 1
            else out
        )
        launch_grid = (inferred_segments * parts, grid[1])
        block_edges = SEGMENT_SOFTMAX_SCHEDULE.config.max_block_edges
        kernel._segment_softmax_csr_forward.run(
            logits,
            row_offsets,
            out,
            partial,
            inferred_segments,
            logits.shape[0],
            features,
            logits.dtype == torch.float64,
            block_features,
            block_edges,
            parts,
            num_warps=num_warps,
            grid=launch_grid,
            warmup=False,
        )
        if parts > 1:
            kernel._segment_finish_slices.run(
                logits,
                out,
                out,
                row_offsets,
                partial,
                out,
                logits.shape[0],
                inferred_segments,
                features,
                "softmax_forward",
                parts,
                block_features,
                block_edges,
                num_warps=num_warps,
                grid=launch_grid,
                warmup=False,
            )
        return out.reshape_as(logits)


@torch.library.register_fake(_segment_softmax_csr_forward)
def _segment_softmax_csr_forward_fake(
    logits: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
) -> Tensor:
    del row_offsets, segment_ids, inferred_segments
    return torch.empty_like(logits)


def _segment_softmax_csr_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor | None, int],
    output: Tensor,
) -> None:
    _logits, row_offsets, segment_ids, inferred_segments = inputs
    ctx.save_for_backward(output, row_offsets, segment_ids)
    ctx.inferred_segments = inferred_segments


def _segment_softmax_csr_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None]:
    r"""The VJP is ``y * (g - sum_segment(g * y))``."""

    probabilities, row_offsets, cached_segment_ids = ctx.saved_tensors
    if not torch.is_grad_enabled():
        return (
            _run_segment_backward(
                grad_out,
                probabilities,
                probabilities,
                row_offsets,
                probabilities.shape[0],
                ctx.inferred_segments,
                "softmax",
            ),
            None,
            None,
            None,
        )
    segment_ids = canonical_segment_ids(row_offsets, probabilities.shape[0], cached_segment_ids)
    accumulate_dtype = torch.float32 if probabilities.dtype in HALF_DTYPES else probabilities.dtype
    probs = probabilities.to(accumulate_dtype)
    upstream = grad_out.to(accumulate_dtype)
    row_totals = _segment_sum_csr_forward(
        (probs * upstream).contiguous(),
        row_offsets,
        segment_ids,
        ctx.inferred_segments,
        ctx.inferred_segments,
    )
    grad_logits = probs * (upstream - rows_per_edge(row_totals, segment_ids, ctx.inferred_segments))
    return grad_logits.to(probabilities.dtype), None, None, None


torch.library.register_autograd(
    _segment_softmax_csr_forward,
    _segment_softmax_csr_forward_autograd_backward,
    setup_context=_segment_softmax_csr_forward_setup_context,
)


class _SegmentSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, row_offsets, segment_ids, inferred_segments):
        output = _launch_segment_softmax(logits, row_offsets, inferred_segments)
        _segment_softmax_csr_forward_setup_context(ctx, (logits, row_offsets, segment_ids, inferred_segments), output)
        return output

    backward = staticmethod(_segment_softmax_csr_forward_autograd_backward)


@torch.library.custom_op("ooperators::_segment_csr_backward", mutates_args=(), device_types="cuda")
def _segment_csr_backward(
    grad_out: Tensor,
    values: Tensor,
    output: Tensor,
    row_offsets: Tensor,
    num_edges: int,
    inferred_segments: int,
    reduction: str,
) -> Tensor:
    return _launch_segment_backward(grad_out, values, output, row_offsets, num_edges, inferred_segments, reduction)


def _run_segment_backward(
    grad_out: Tensor,
    values: Tensor,
    output: Tensor,
    row_offsets: Tensor,
    num_edges: int,
    inferred_segments: int,
    reduction: str,
) -> Tensor:
    grad_type = type(grad_out)
    opaque = (
        is_compiling()
        or functorch_transforms_active()
        or (grad_type is not Tensor and grad_type is not torch.nn.Parameter and is_external_fake(grad_out))
    )
    operation = _segment_csr_backward if opaque else _launch_segment_backward
    return operation(grad_out, values, output, row_offsets, num_edges, inferred_segments, reduction)


def _launch_long_max_backward(
    values: Tensor,
    maxima: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    result: Tensor,
    segments: int,
    plan: SegmentPlan,
) -> bool:
    values = values.resolve_neg().contiguous()
    maxima = maxima.resolve_neg().contiguous()
    grad_out = grad_out.resolve_neg().contiguous()
    row_offsets = row_offsets.resolve_neg().contiguous()
    features = flat_features(values)
    buckets = values.shape[0] // LONG_ROW_EDGES
    long_features = min(8, plan.block_features)
    feature_tiles = (features + long_features - 1) // long_features
    parts = 128
    itemsize = 8 if values.dtype == torch.float64 else 4
    while parts > 1 and (buckets * parts * features * itemsize + buckets * 24 > 16 << 20):
        parts //= 2
    if parts == 1 or feature_tiles > CUDA_MAX_GRID_YZ or segments + buckets > CUDA_MAX_GRID_X:
        return False
    descriptors = torch.empty((buckets, 3), device=values.device, dtype=torch.int64)
    partial = torch.empty(
        (buckets * parts, features),
        device=values.device,
        dtype=torch.float64 if values.dtype == torch.float64 else torch.float32,
    )
    kernel._segment_max_backward_adaptive.run(
        values,
        maxima,
        grad_out,
        row_offsets,
        result,
        values.shape[0],
        segments,
        features,
        values.dtype == torch.float64,
        plan.block_features,
        plan.max_block_edges,
        long_features,
        LONG_BLOCK_EDGES,
        LONG_ROW_EDGES,
        skip_long_rows=True,
        long_descriptors_ptr=descriptors,
        num_warps=plan.num_warps,
        enable_fp_fusion=False,
        grid=(segments + buckets, plan.grid[1]),
        warmup=False,
    )
    grid = (parts, feature_tiles)
    kernel._segment_max_long_statistics.run(
        values,
        maxima,
        descriptors,
        partial,
        values.shape[0],
        features,
        values.dtype == torch.float64,
        parts,
        long_features,
        LONG_BLOCK_EDGES,
        LONG_ROW_EDGES,
        num_warps=plan.num_warps,
        enable_fp_fusion=False,
        grid=grid,
        warmup=False,
    )
    kernel._segment_max_long_finish.run(
        values,
        maxima,
        grad_out,
        descriptors,
        partial,
        result,
        values.shape[0],
        features,
        parts,
        long_features,
        LONG_BLOCK_EDGES,
        LONG_ROW_EDGES,
        num_warps=plan.num_warps,
        enable_fp_fusion=False,
        grid=grid,
        warmup=False,
    )
    return True


def _launch_segment_backward(
    grad_out: Tensor,
    values: Tensor,
    output: Tensor,
    row_offsets: Tensor,
    num_edges: int,
    inferred_segments: int,
    reduction: str,
) -> Tensor:

    with cuda_device_context(values.device):
        grad_out = grad_out.resolve_neg().contiguous()
        if reduction != "sum":
            values = values.resolve_neg().contiguous()
            output = output.resolve_neg().contiguous()
        row_offsets = row_offsets.resolve_neg().contiguous()
        if inferred_segments == 0:
            return torch.zeros((num_edges, *values.shape[1:]), device=values.device, dtype=values.dtype)
        features = flat_features(values)
        grad_values = torch.empty((num_edges, *values.shape[1:]), device=values.device, dtype=values.dtype)
        if reduction == "sum" and num_edges >= LONG_ROW_EDGES and 16 <= features <= 512:
            flat_programs = (inferred_segments * features + 511) // 512
            long_parts = 128
            programs = flat_programs + (num_edges // LONG_ROW_EDGES) * long_parts
            if programs <= CUDA_MAX_GRID_X:
                kernel._sum_backward_flat.run(
                    grad_out,
                    row_offsets,
                    grad_values,
                    num_edges,
                    inferred_segments,
                    features,
                    512,
                    long_parts,
                    32,
                    LONG_ROW_EDGES,
                    num_warps=4,
                    grid=(programs,),
                    warmup=False,
                )
                return grad_values
        plan = resolve_execution_plan(SEGMENT_SUM_SCHEDULE, values, inferred_segments)
        parts = reduction_part_count(values, num_edges, inferred_segments, inferred_segments, plan.block_features)
        adaptive_reduction = (
            adaptive_reduction_geometry(num_edges, inferred_segments, inferred_segments, plan.block_features)
            if reduction != "softmax" and parts == 1
            else None
        )
        if adaptive_reduction is not None:
            if reduction == "max" and _launch_long_max_backward(
                values, output, grad_out, row_offsets, grad_values, inferred_segments, plan
            ):
                return grad_values
            programs, long_features = adaptive_reduction
            long_parts = 1
            long_edges = LONG_BLOCK_EDGES
            if reduction == "sum":
                long_parts = min(128, max(8, plan.block_features * 2))
                programs = inferred_segments + (num_edges // LONG_ROW_EDGES) * long_parts
                if programs <= CUDA_MAX_GRID_X:
                    long_features = plan.block_features
                    long_edges = 32
                else:
                    programs, long_features = adaptive_reduction
                    long_parts = 1
            extra_operands = (values, output) if reduction == "max" else ()
            entry = (
                kernel._segment_max_backward_adaptive if reduction == "max" else kernel._segment_sum_backward_adaptive
            )
            entry.run(
                *extra_operands,
                grad_out,
                row_offsets,
                grad_values,
                num_edges,
                inferred_segments,
                features,
                *((values.dtype == torch.float64,) if reduction == "max" else ()),
                plan.block_features,
                plan.max_block_edges,
                long_features,
                long_edges,
                LONG_ROW_EDGES,
                **({"long_parts": long_parts} if reduction == "sum" else {}),
                num_warps=plan.num_warps,
                enable_fp_fusion=False,
                grid=(programs, plan.grid[1]),
                warmup=False,
            )
            return grad_values
        adaptive = (
            adaptive_geometry(SEGMENT_SOFTMAX_ADAPTIVE, num_edges, inferred_segments, features, plan.block_features)
            if reduction == "softmax" and parts == 1
            else None
        )
        if (
            reduction == "softmax"
            and parts == 1
            and num_edges >= SEGMENT_SOFTMAX_ADAPTIVE.long_row_edges
            and _launch_long_softmax(
                values, grad_out, grad_values, row_offsets, inferred_segments, plan.block_features, plan.num_warps, True
            )
        ):
            return grad_values
        if adaptive is not None:
            adaptive_work, long_features = adaptive
            kernel._segment_softmax_adaptive.run(
                values,
                grad_out,
                grad_values,
                row_offsets,
                num_edges,
                inferred_segments,
                features,
                values.dtype == torch.float64,
                True,
                plan.block_features,
                plan.max_block_edges,
                long_features,
                SEGMENT_SOFTMAX_ADAPTIVE.block_edges,
                SEGMENT_SOFTMAX_ADAPTIVE.long_row_edges,
                num_warps=plan.num_warps,
                enable_fp_fusion=False,
                grid=(adaptive_work, plan.grid[1]),
                warmup=False,
            )
            return grad_values
        accumulation_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        partial = (
            torch.empty((inferred_segments * parts, features), device=values.device, dtype=accumulation_dtype)
            if parts > 1 and reduction != "sum"
            else grad_values
        )
        launch_grid = (inferred_segments * parts, plan.grid[1])
        block_edges = plan.max_block_edges
        # Separate product rounding is part of the VJP contract.
        kernel._segment_csr_backward.run(
            values,
            output,
            grad_out,
            row_offsets,
            grad_values,
            partial,
            num_edges,
            inferred_segments,
            features,
            values.dtype == torch.float64,
            reduction,
            plan.block_features,
            block_edges,
            parts,
            num_warps=plan.num_warps,
            enable_fp_fusion=False,
            grid=launch_grid,
            warmup=False,
        )
        if parts > 1 and reduction != "sum":
            kernel._segment_finish_slices.run(
                values,
                output,
                grad_out,
                row_offsets,
                partial,
                grad_values,
                num_edges,
                inferred_segments,
                features,
                reduction,
                parts,
                plan.block_features,
                block_edges,
                num_warps=plan.num_warps,
                enable_fp_fusion=False,
                grid=launch_grid,
                warmup=False,
            )
        return grad_values


@torch.library.register_fake(_segment_csr_backward)
def _segment_csr_backward_fake(
    grad_out: Tensor,
    values: Tensor,
    output: Tensor,
    row_offsets: Tensor,
    num_edges: int,
    inferred_segments: int,
    reduction: str,
) -> Tensor:
    del grad_out, output, row_offsets, inferred_segments, reduction
    return torch.empty((num_edges, *values.shape[1:]), device=values.device, dtype=values.dtype)
