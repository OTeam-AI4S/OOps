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
from oops.runtime import tensors_share_device

from ...csr import canonical_segment_ids
from ...csr.triton import kernel as csr_kernel
from ...csr.triton.backend import (
    finish_long_reduction,
    flat_features,
    prepare_long_reduction,
    reduction_part_count,
    resolve_execution_plan,
    rows_per_edge,
    schedule_supports,
    supports_reduction_layout,
)
from ...csr.triton.schedule import (
    LONG_BLOCK_EDGES,
    LONG_ROW_EDGES,
    SegmentPlan,
    adaptive_reduction_geometry,
    segment_block_edges,
)
from . import kernel
from .profiles import CSR_WEIGHTED_SUM_SCHEDULE


def _plain_weighted_sum_call(values: Tensor, weights: Tensor, row_offsets: Tensor) -> bool:
    operands = (values, weights, row_offsets)
    return (
        not is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
    )


def _inner_size(values: Tensor) -> int:
    inner = 1
    for dim in values.shape[2:]:
        inner *= dim
    return inner


def supports_csr_weighted_sum(values: Tensor, weights: Tensor, row_offsets: Tensor, out_segments: int) -> bool:

    return (
        supports_reduction_layout(values, row_offsets, out_segments)
        and schedule_supports(CSR_WEIGHTED_SUM_SCHEDULE, values, out_segments)
        and weights.is_cuda
        and tensors_share_device(values, weights, row_offsets)
        and weights.dtype == values.dtype
        and weights.is_contiguous()
        and _weight_kind(values, weights) is not None
    )


def triton_csr_weighted_sum(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    if tensors_need_grad(values, weights):
        if _plain_weighted_sum_call(values, weights, row_offsets):
            return _CSRWeightedSum.apply(values, weights, row_offsets, segment_ids, out_segments, inferred_segments)
        return _csr_weighted_sum_forward(values, weights, row_offsets, segment_ids, out_segments, inferred_segments)
    compiling = is_compiling()
    value_type = type(values)
    external_fake = (
        not compiling and value_type is not Tensor and value_type is not torch.nn.Parameter and is_external_fake(values)
    )
    if compiling or external_fake:
        return _csr_weighted_sum_forward(values, weights, row_offsets, segment_ids, out_segments, inferred_segments)
    return _launch_csr_weighted_sum(values, weights, row_offsets, out_segments, inferred_segments)


@torch.library.custom_op("ooperators::_csr_weighted_sum_forward", mutates_args=(), device_types="cuda")
def _csr_weighted_sum_forward(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del segment_ids
    plan = resolve_execution_plan(CSR_WEIGHTED_SUM_SCHEDULE, values, out_segments)
    return _launch_csr_weighted_sum(
        values,
        weights,
        row_offsets,
        out_segments,
        inferred_segments,
        plan=plan,
    )


def _launch_fused_weighted_sum(
    values: Tensor, weights: Tensor, row_offsets: Tensor, out_segments: int, inferred_segments: int
) -> Tensor | None:
    features = flat_features(values)
    num_edges = values.shape[0]
    dtype = values.dtype
    device = values.device
    use_float64 = dtype is torch.float64
    buckets = num_edges // LONG_ROW_EDGES
    parts = 16
    itemsize = 8 if use_float64 else 4
    tiles = (features + 7) // 8
    while parts > 1 and buckets * (parts * features * itemsize + tiles * 4) > 16 << 20:
        parts //= 2
    programs = (out_segments * features + 511) // 512 + buckets * parts * tiles
    if buckets == 0 or parts == 1 or programs > CUDA_MAX_GRID_X or inferred_segments == 0:
        return None
    values = values.resolve_neg()
    row_offsets = row_offsets.resolve_neg()
    with cuda_device_context(device):
        out = torch.empty((out_segments, *values.shape[1:]), device=device, dtype=dtype)
        partial = torch.empty(
            (buckets * parts, features),
            device=device,
            dtype=torch.float64 if use_float64 else torch.float32,
        )
        counters = torch.zeros((buckets, tiles), device=device, dtype=torch.int32)
        weights = weights.resolve_neg()
        weight_groups = weights.numel() // num_edges
        kernel._csr_weighted_sum_forward_fused.run(
            values,
            row_offsets,
            out,
            partial,
            counters,
            inferred_segments,
            out_segments,
            num_edges,
            features,
            use_float64,
            buckets,
            parts,
            8,
            LONG_BLOCK_EDGES,
            512,
            LONG_ROW_EDGES,
            grid=(programs,),
            warmup=False,
            num_warps=4,
            weights_ptr=weights,
            weight_groups=weight_groups,
            weight_features=features // weight_groups,
        )
        return out


def _launch_csr_weighted_sum(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    *,
    plan: SegmentPlan | None = None,
) -> Tensor:
    fused = _launch_fused_weighted_sum(values, weights, row_offsets, out_segments, inferred_segments)
    if fused is not None:
        return fused
    with cuda_device_context(values.device):
        values = values.resolve_neg()
        weights = weights.resolve_neg()
        row_offsets = row_offsets.resolve_neg()
        features = flat_features(values)
        if plan is None:
            config = CSR_WEIGHTED_SUM_SCHEDULE.config
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
        kind = _weight_kind(values, weights)
        if kind is None:
            raise ValueError("unsupported sparse weighted sum layout")

        inner_size = 1
        weight_group_dim = 1
        weight_e_stride = 1
        weight_group_stride = 1
        if kind == 1:
            inner_size = _inner_size(values)
            weight_group_dim = int(weights.shape[1])
            weight_e_stride = int(weights.stride(0))
            weight_group_stride = int(weights.stride(1))

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
                kernel._csr_weighted_sum_forward_adaptive[(out_segments + buckets, grid[1])](
                    values,
                    weights,
                    row_offsets,
                    out,
                    inferred_segments,
                    out_segments,
                    values.shape[0],
                    features,
                    inner_size,
                    weight_group_dim,
                    weight_e_stride,
                    weight_group_stride,
                    kind,
                    values.dtype == torch.float64,
                    block_features,
                    segment_block_edges(
                        values.shape[0],
                        inferred_segments,
                        CSR_WEIGHTED_SUM_SCHEDULE.config.max_block_edges,
                        serial_short_rows=True,
                    ),
                    min(8, block_features),
                    LONG_BLOCK_EDGES,
                    LONG_ROW_EDGES,
                    skip_long_rows=True,
                    long_descriptors_ptr=descriptors,
                    counter_tiles=(features + min(8, block_features) - 1) // min(8, block_features),
                    num_warps=num_warps,
                )
                finish_long_reduction(
                    values,
                    descriptors,
                    long_partial,
                    out,
                    buckets,
                    long_parts,
                    block_features,
                    num_warps,
                    "sum",
                    weights,
                )
                return out.reshape(out_segments, *values.shape[1:])
            programs, long_features = adaptive
            kernel._csr_weighted_sum_forward_adaptive[(programs, grid[1])](
                values,
                weights,
                row_offsets,
                out,
                inferred_segments,
                out_segments,
                values.shape[0],
                features,
                inner_size,
                weight_group_dim,
                weight_e_stride,
                weight_group_stride,
                kind,
                values.dtype == torch.float64,
                block_features,
                segment_block_edges(
                    values.shape[0],
                    inferred_segments,
                    CSR_WEIGHTED_SUM_SCHEDULE.config.max_block_edges,
                    serial_short_rows=True,
                ),
                long_features,
                LONG_BLOCK_EDGES,
                LONG_ROW_EDGES,
                num_warps=num_warps,
            )
            return out.reshape(out_segments, *values.shape[1:])
        kernel._csr_weighted_sum_forward[launch_grid](
            values,
            weights,
            row_offsets,
            partial,
            inferred_segments,
            values.shape[0],
            features,
            inner_size,
            weight_group_dim,
            weight_e_stride,
            weight_group_stride,
            kind,
            values.dtype == torch.float64,
            block_features,
            segment_block_edges(
                values.shape[0],
                inferred_segments,
                CSR_WEIGHTED_SUM_SCHEDULE.config.max_block_edges,
                serial_short_rows=True,
            ),
            parts,
            num_warps=num_warps,
        )
        if parts > 1:
            csr_kernel.merge_partials[grid](partial, out, features, "sum", parts, block_features, num_warps=num_warps)
        return out.reshape(out_segments, *values.shape[1:])


@torch.library.register_fake(_csr_weighted_sum_forward)
def _csr_weighted_sum_forward_fake(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del weights, row_offsets, segment_ids, inferred_segments
    return torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)


def _csr_weighted_sum_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor | None, int, int],
    output: Tensor,
) -> None:
    values, weights, row_offsets, segment_ids, _out_segments, inferred_segments = inputs
    del output
    ctx.save_for_backward(values, weights, row_offsets, segment_ids)
    ctx.inferred_segments = inferred_segments


def _csr_weighted_sum_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
) -> tuple[Tensor | None, Tensor | None, None, None, None, None]:

    values, weights, row_offsets, cached_segment_ids = ctx.saved_tensors
    need_grad_values, need_grad_weights = ctx.needs_input_grad[:2]
    if not torch.is_grad_enabled():
        grad_values, grad_weights = _run_weighted_sum_backward(
            values,
            weights,
            grad_out,
            row_offsets,
            cached_segment_ids,
            ctx.inferred_segments,
            need_grad_values,
            need_grad_weights,
        )
        return grad_values, grad_weights, None, None, None, None
    segment_ids = canonical_segment_ids(row_offsets, values.shape[0], cached_segment_ids)
    upstream = rows_per_edge(grad_out, segment_ids, ctx.inferred_segments)
    work_dtype = torch.float32 if values.dtype in HALF_DTYPES else values.dtype
    upstream = upstream.to(work_dtype)
    covered = segment_ids < ctx.inferred_segments
    grad_values = None
    grad_weights = None
    if need_grad_values:
        weight_mask = covered.reshape(values.shape[0], *((1,) * (weights.dim() - 1)))
        work_weights = torch.where(weight_mask, weights.to(work_dtype), 0)
        grad_values = (upstream * _weights_broadcast_over(work_weights, values)).to(values.dtype)
    if need_grad_weights:
        value_mask = covered.reshape(values.shape[0], *((1,) * (values.dim() - 1)))
        work_values = torch.where(value_mask, values.to(work_dtype), 0)
        grad_weights = _sum_to_weight_layout(upstream * work_values, weights).to(weights.dtype)
    return grad_values, grad_weights, None, None, None, None


torch.library.register_autograd(
    _csr_weighted_sum_forward,
    _csr_weighted_sum_forward_autograd_backward,
    setup_context=_csr_weighted_sum_forward_setup_context,
)


class _CSRWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weights, row_offsets, segment_ids, out_segments, inferred_segments):
        output = _launch_csr_weighted_sum(values, weights, row_offsets, out_segments, inferred_segments)
        _csr_weighted_sum_forward_setup_context(
            ctx, (values, weights, row_offsets, segment_ids, out_segments, inferred_segments), output
        )
        return output

    backward = staticmethod(_csr_weighted_sum_forward_autograd_backward)


@torch.library.custom_op(
    "ooperators::_csr_weighted_sum_backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor values, Tensor weights, Tensor grad_out, Tensor row_offsets, Tensor? segment_ids, "
    "SymInt inferred_segments, bool need_grad_values, bool need_grad_weights) -> (Tensor?, Tensor?)",
)
def _csr_weighted_sum_backward(
    values: Tensor,
    weights: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
    need_grad_values: bool,
    need_grad_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _launch_weighted_sum_backward(
        values, weights, grad_out, row_offsets, segment_ids, inferred_segments, need_grad_values, need_grad_weights
    )


def _run_weighted_sum_backward(
    values: Tensor,
    weights: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
    need_grad_values: bool,
    need_grad_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    grad_type = type(grad_out)
    opaque = (
        is_compiling()
        or functorch_transforms_active()
        or (grad_type is not Tensor and grad_type is not torch.nn.Parameter and is_external_fake(grad_out))
    )
    operation = _csr_weighted_sum_backward if opaque else _launch_weighted_sum_backward
    return operation(
        values, weights, grad_out, row_offsets, segment_ids, inferred_segments, need_grad_values, need_grad_weights
    )


def _launch_weighted_sum_backward(
    values: Tensor,
    weights: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
    need_grad_values: bool,
    need_grad_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:

    with cuda_device_context(values.device):
        values = values.resolve_neg().contiguous()
        weights = None if weights is None else weights.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        row_offsets = row_offsets.resolve_neg().contiguous()
        segment_ids = None if segment_ids is None else segment_ids.resolve_neg()
        grad_out = grad_out.contiguous()
        groups = weights.numel() // values.shape[0]
        group_features = flat_features(values) // groups
        block_features = min(256, _next_power_of_two(group_features))
        block_groups = max(1, 512 // block_features)
        grad_values = torch.empty_like(values) if need_grad_values else None
        grad_weights = torch.empty_like(weights) if need_grad_weights else None
        if 16 <= flat_features(values) <= 128:
            edge_groups = min(_next_power_of_two(groups), max(1, 64 // block_features))
            edge_rows = min(32, max(1, 2048 // (edge_groups * block_features)))
            edge_grid = ((values.shape[0] + edge_rows - 1) // edge_rows, (groups + edge_groups - 1) // edge_groups)
            if groups % edge_groups == 0 and edge_grid[0] <= CUDA_MAX_GRID_X and edge_grid[1] <= CUDA_MAX_GRID_YZ:
                kernel._csr_weighted_backward_edges[edge_grid](
                    values,
                    weights,
                    grad_out,
                    row_offsets,
                    row_offsets if segment_ids is None else segment_ids,
                    values if grad_values is None else grad_values,
                    weights if grad_weights is None else grad_weights,
                    values.shape[0],
                    inferred_segments,
                    groups,
                    group_features,
                    segment_ids is not None,
                    1 if segment_ids is None else segment_ids.stride(0),
                    need_grad_values,
                    need_grad_weights,
                    values.dtype == torch.float64,
                    edge_rows,
                    edge_groups,
                    block_features,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                return grad_values, grad_weights
        grid = (min(CUDA_MAX_GRID_X, (values.shape[0] * groups + block_groups - 1) // block_groups),)
        # Separate product rounding is part of the VJP contract.
        kernel._csr_weighted_sum_backward[grid](
            values,
            weights,
            grad_out,
            row_offsets,
            row_offsets if segment_ids is None else segment_ids,
            values if grad_values is None else grad_values,
            weights if grad_weights is None else grad_weights,
            values.shape[0],
            inferred_segments,
            groups,
            group_features,
            segment_ids is not None,
            1 if segment_ids is None else segment_ids.stride(0),
            need_grad_values,
            need_grad_weights,
            values.dtype == torch.float64,
            block_groups,
            block_features,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return grad_values, grad_weights


@torch.library.register_fake(_csr_weighted_sum_backward)
def _csr_weighted_sum_backward_fake(
    values: Tensor,
    weights: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
    need_grad_values: bool,
    need_grad_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del grad_out, row_offsets, segment_ids, inferred_segments
    return (
        torch.empty_like(values) if need_grad_values else None,
        torch.empty_like(weights) if need_grad_weights else None,
    )


def _weights_broadcast_over(weights: Tensor, values: Tensor) -> Tensor:

    if weights.dim() == values.dim():
        return weights
    return weights.reshape(*weights.shape, *((1,) * (values.dim() - weights.dim())))


def _sum_to_weight_layout(product: Tensor, weights: Tensor) -> Tensor:

    reduced = product
    if product.dim() > weights.dim():
        reduced = reduced.sum(dim=tuple(range(weights.dim(), product.dim())))
    spread = tuple(axis for axis in range(1, weights.dim()) if weights.shape[axis] != reduced.shape[axis])
    if spread:
        reduced = reduced.sum(dim=spread, keepdim=True)
    return reduced


def _weight_kind(values: Tensor, weights: Tensor) -> int | None:
    if weights.dim() < 1 or weights.shape[0] != values.shape[0]:
        return None
    if weights.dim() == 1:
        return 0
    if tuple(weights.shape) == tuple(values.shape):
        return 2
    if values.dim() >= 2 and weights.dim() == 2 and int(weights.shape[1]) in (1, int(values.shape[1])):
        return 1
    return None
