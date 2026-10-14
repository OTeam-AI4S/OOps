from __future__ import annotations

import torch
import triton
from torch import Tensor

from oops.core import CoreLaunchContext, CoreTileConfig, TargetDescriptor, select_core_tile, select_launch_candidates
from oops.normalization.layer_norm.triton.schedule import LayerNormPlan
from oops.runtime import (
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps_if_needed,
    cuda_device_context,
    is_compiling,
    shared_memory_budget,
    tensors_share_device,
)

from . import kernel
from .profiles import DUAL_PROJECTION_FALLBACK, DUAL_PROJECTION_TARGETS, TAIL_FALLBACK, TAIL_TARGETS
from .schedule import (
    DualProjectionPlan,
    DualProjectionProblem,
    PairTailPlan,
    PairTailProblem,
    make_dual_projection_plan,
    make_pair_tail_plan,
)

_FAKE_SHARED_MEMORY_BUDGET = 47 * 1024


def triton_outer_product_count_backward(
    gradient: Tensor,
    numerator: Tensor,
    count: Tensor,
    bias: Tensor | None,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> Tensor:
    with cuda_device_context(gradient.device):
        gradient = gradient.resolve_neg()
        numerator = numerator.resolve_neg().contiguous()
        count = count.resolve_neg().contiguous()
        bias = None if bias is None else bias.resolve_neg().contiguous()
        output = torch.empty_like(count)
        channels = gradient.shape[-1]
        rows = count.numel()
        pairs_per_batch = rows // gradient.shape[0]
        kernel._outer_product_count_backward.run(
            gradient,
            numerator,
            count,
            numerator if bias is None else bias,
            output,
            rows,
            pairs_per_batch,
            gradient.stride(0),
            channels,
            eps,
            divide_before,
            clamp,
            bias is not None,
            16,
            triton.next_power_of_2(min(channels, 256)),
            grid=((rows + 15) // 16,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output


def supports_msa_dual_projection(
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    *,
    portable: bool = False,
) -> bool:

    hidden = left_weight.shape[0] if left_weight.dim() == 2 else 0
    channels = msa.shape[-1] if msa.dim() == 4 else 0
    rows = msa.shape[0] * msa.shape[1] * msa.shape[2] if msa.dim() == 4 else 0
    projection_dtype = left_weight.dtype
    projection_biases = tuple(bias for bias in (left_bias, right_bias) if bias is not None)
    return (
        msa.is_cuda
        and norm_weight.is_cuda
        and norm_bias.is_cuda
        and left_weight.is_cuda
        and right_weight.is_cuda
        and norm_weight.device == msa.device
        and norm_bias.device == msa.device
        and left_weight.device == msa.device
        and right_weight.device == msa.device
        and msa.dim() == 4
        and msa.stride(-1) == 1
        and norm_weight.is_contiguous()
        and norm_bias.is_contiguous()
        and left_weight.is_contiguous()
        and right_weight.is_contiguous()
        and all(bias.is_cuda and bias.device == msa.device and bias.is_contiguous() for bias in projection_biases)
        and msa.dtype in STANDARD_FLOAT_DTYPES
        and norm_weight.dtype in (msa.dtype, torch.float32)
        and norm_bias.dtype in (msa.dtype, torch.float32)
        and projection_dtype in STANDARD_FLOAT_DTYPES
        and right_weight.dtype == projection_dtype
        and all(bias.dtype == projection_dtype for bias in projection_biases)
        and channels > 0
        and hidden > 0
        and norm_weight.shape == (channels,)
        and norm_bias.shape == (channels,)
        and left_weight.shape == (hidden, channels)
        and right_weight.shape == (hidden, channels)
        and all(bias.shape == (hidden,) for bias in projection_biases)
        and (
            is_compiling()
            or isinstance(rows, torch.SymInt)
            or rows == 0
            or _dual_projection_domain_supported(
                rows,
                channels,
                hidden,
                projection_dtype,
                msa.device,
                budget=_FAKE_SHARED_MEMORY_BUDGET if portable else None,
            )
        )
    )


def _dual_projection_config(
    rows: int,
    channels: int,
    hidden: int,
    dtype: torch.dtype,
    *,
    target: TargetDescriptor | None = None,
) -> CoreTileConfig:
    return select_core_tile(
        CoreLaunchContext(
            num_rows=rows,
            channels=hidden,
            reduction=channels,
            dtype=dtype,
        ),
        DUAL_PROJECTION_FALLBACK,
        target=target,
        target_deltas=DUAL_PROJECTION_TARGETS,
    )


def _dual_projection_configs(
    rows: int,
    channels: int,
    hidden: int,
    dtype: torch.dtype,
) -> tuple[CoreTileConfig, ...]:
    return select_launch_candidates(
        CoreLaunchContext(
            num_rows=rows,
            channels=hidden,
            reduction=channels,
            dtype=dtype,
        ),
        DUAL_PROJECTION_FALLBACK,
        target_deltas=DUAL_PROJECTION_TARGETS,
    )


def _dual_projection_domain_supported(
    rows: int,
    channels: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    budget: int | None = None,
) -> bool:

    problem = DualProjectionProblem(rows, channels, hidden, dtype)
    budget = shared_memory_budget(device) if budget is None else budget
    return all(
        make_dual_projection_plan(problem, config, budget) is not None
        for config in _dual_projection_configs(rows, channels, hidden, dtype)
    )


def _resolve_dual_projection_plan(
    rows: int,
    channels: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    target: TargetDescriptor | None = None,
) -> DualProjectionPlan | None:

    problem = DualProjectionProblem(rows, channels, hidden, dtype)
    return make_dual_projection_plan(
        problem,
        _dual_projection_config(rows, channels, hidden, dtype, target=target),
        shared_memory_budget(device),
    )


def triton_msa_dual_projection(
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    eps: float,
    input_precision: str,
    *,
    plan: DualProjectionPlan | None = None,
    save_stats: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:

    with cuda_device_context(msa.device):
        msa = msa.resolve_neg().contiguous()
        norm_weight = norm_weight.resolve_neg().contiguous()
        norm_bias = norm_bias.resolve_neg().contiguous()
        left_weight = left_weight.resolve_neg().contiguous()
        right_weight = right_weight.resolve_neg().contiguous()
        left_bias = None if left_bias is None else left_bias.resolve_neg().contiguous()
        right_bias = None if right_bias is None else right_bias.resolve_neg().contiguous()
        batch, msa_depth, tokens, channels = msa.shape
        hidden = left_weight.shape[0]
        rows = batch * msa_depth * tokens
        projection_dtype = left_weight.dtype
        left = torch.empty((batch, msa_depth, tokens, hidden), device=msa.device, dtype=projection_dtype)
        right = torch.empty_like(left)
        normalized = torch.empty(msa.shape, device=msa.device, dtype=projection_dtype) if save_stats else None
        mean = torch.empty(rows, device=msa.device, dtype=torch.float32) if save_stats else None
        rstd = torch.empty_like(mean) if mean is not None else None
        if rows == 0:
            return left, right, normalized, mean, rstd
        if plan is None:
            plan = _resolve_dual_projection_plan(
                rows,
                channels,
                hidden,
                projection_dtype,
                msa.device,
                target=cuda_device_caps_if_needed(msa.device, bool(DUAL_PROJECTION_TARGETS)),
            )
        if plan is None:
            raise ValueError("outer_product_mean dual projection tile or launch grid is unsupported")
        has_left_bias = left_bias is not None
        has_right_bias = right_bias is not None
        left_bias_pointer = left_weight if left_bias is None else left_bias
        right_bias_pointer = right_weight if right_bias is None else right_bias
        kernel._msa_dual_projection_forward[plan.grid](
            msa,
            norm_weight,
            norm_bias,
            left_weight,
            right_weight,
            left_bias_pointer,
            right_bias_pointer,
            left.reshape(rows, hidden),
            right.reshape(rows, hidden),
            msa if normalized is None else normalized,
            msa if mean is None else mean,
            msa if rstd is None else rstd,
            save_stats,
            rows,
            channels,
            hidden,
            msa_depth,
            tokens,
            msa.stride(0),
            msa.stride(1),
            msa.stride(2),
            float(eps),
            plan.block_rows,
            plan.block_reduction,
            plan.block_hidden,
            has_left_bias,
            has_right_bias,
            input_precision=input_precision,
            contiguous_rows=msa.is_contiguous(),
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
        )
        return left, right, normalized, mean, rstd


def _select_pair_tail_config(
    total_pairs: int,
    channels: int,
    hidden: int,
    tokens: int,
    dtype: torch.dtype,
    *,
    target: TargetDescriptor | None = None,
) -> CoreTileConfig:

    return select_core_tile(
        CoreLaunchContext(
            num_rows=total_pairs,
            channels=channels,
            reduction=hidden,
            tokens=tokens,
            dtype=dtype,
        ),
        TAIL_FALLBACK,
        target=target,
        target_deltas=TAIL_TARGETS,
    )


def _resolve_pair_tail_plan(
    pair_rows: int,
    tokens: int,
    channels: int,
    hidden: int,
    dtype: torch.dtype,
    shared_memory_bytes: int,
    *,
    target: TargetDescriptor | None = None,
) -> PairTailPlan | None:
    return make_pair_tail_plan(
        PairTailProblem(pair_rows, tokens, channels, hidden, dtype),
        _select_pair_tail_config(
            pair_rows * tokens,
            channels,
            hidden,
            tokens,
            dtype,
            target=target,
        ),
        shared_memory_bytes,
    )


def triton_project_pair_outer_mean(
    outer: Tensor,
    count: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out: Tensor,
    *,
    numerator: Tensor | None,
    input_precision: str,
    divide_before_projection: bool,
    denominator_eps: float,
    clamp_denominator: bool,
    target: TargetDescriptor | None,
) -> None:

    if target is None:
        target = cuda_device_caps_if_needed(outer.device, bool(TAIL_TARGETS))
    outer = outer.resolve_neg().contiguous()
    count = count.resolve_neg().contiguous()
    weight = weight.resolve_neg().contiguous()
    bias = None if bias is None else bias.resolve_neg().contiguous()
    batch, tokens, hidden_left, _token_columns, hidden_right = outer.shape
    if hidden_left != hidden_right:
        raise ValueError("outer_product_mean pair rows must carry equal hidden axes")
    hidden = hidden_left * hidden_right
    channels = weight.shape[0]
    total_pairs = batch * tokens * tokens
    pair_rows = batch * tokens
    plan = _resolve_pair_tail_plan(
        pair_rows,
        tokens,
        channels,
        hidden,
        outer.dtype,
        shared_memory_budget(outer.device),
        target=target,
    )
    if plan is None:
        raise ValueError("outer_product_mean pair-tail launch grid is unsupported")
    bias_kind = 0 if bias is None else 1
    bias_tensor = weight if bias is None else bias
    kernel._project_outer_mean_forward[plan.grid](
        outer,
        count,
        weight,
        bias_tensor,
        out,
        out if numerator is None else numerator,
        numerator is not None,
        total_pairs,
        tokens,
        hidden,
        hidden_left,
        channels,
        float(denominator_eps),
        bool(divide_before_projection),
        bool(clamp_denominator),
        bias_kind,
        plan.block_rows,
        plan.block_channels,
        plan.block_hidden,
        plan.row_aligned,
        input_precision=input_precision,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )


def triton_msa_projection_normalization_backward(
    gradient: Tensor,
    msa: Tensor,
    weight: Tensor,
    eps: float,
    mean: Tensor,
    rstd: Tensor,
    destination: Tensor | None,
    need_weight: bool,
    need_bias: bool,
    plan: LayerNormPlan,
) -> tuple[Tensor | None, Tensor | None]:
    gradient = gradient.resolve_neg().contiguous()
    msa = msa.resolve_neg().contiguous()
    weight = weight.resolve_neg().contiguous()
    mean = mean.resolve_neg().contiguous()
    rstd = rstd.resolve_neg().contiguous()
    rows, channels = msa.shape
    partial = torch.empty(
        (int(need_weight) + int(need_bias), plan.grid[0], channels), device=msa.device, dtype=torch.float32
    )
    kernel._msa_projection_normalization_backward[plan.grid](
        gradient,
        msa,
        weight,
        mean,
        rstd,
        msa if destination is None else destination,
        partial,
        rows,
        channels,
        eps,
        destination is not None,
        need_weight,
        need_bias,
        plan.block_rows,
        plan.block_columns,
        num_warps=max(plan.num_warps, 8) if plan.block_columns == 8192 else plan.num_warps,
        enable_fp_fusion=False,
    )
    grad_weight = torch.empty(channels, device=msa.device, dtype=torch.float32) if need_weight else None
    grad_bias = torch.empty(channels, device=msa.device, dtype=torch.float32) if need_bias else None
    if need_weight or need_bias:
        kernel._msa_projection_normalization_affine_reduce.run(
            partial,
            msa if grad_weight is None else grad_weight,
            msa if grad_bias is None else grad_bias,
            plan.grid[0],
            channels,
            need_weight,
            need_bias,
            64,
            16,
            grid=((channels + 15) // 16,),
            warmup=False,
            num_warps=2,
        )
    return grad_weight, grad_bias


def supports_segmented_outer_product_mean(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    msa_outer_offsets: Tensor,
    msa_packed_offsets: Tensor,
    pair_offsets: Tensor,
) -> bool:

    metadata = (msa_outer_offsets, msa_packed_offsets, pair_offsets)
    supported = (
        left.is_cuda
        and tensors_share_device(left, right, mask, weight, bias, *metadata)
        and left.dim() == 2
        and right.shape == left.shape
        and mask.dim() == 1
        and mask.shape[0] == left.shape[0]
        and weight.dim() == 2
        and weight.shape[1] == left.shape[1] * left.shape[1]
        and (bias is None or bias.shape == (weight.shape[0],))
        and left.dtype in STANDARD_FLOAT_DTYPES
        and right.dtype == left.dtype
        and weight.dtype in STANDARD_FLOAT_DTYPES
        and (bias is None or bias.dtype == weight.dtype)
        and left.is_contiguous()
        and right.is_contiguous()
        and mask.is_contiguous()
        and weight.is_contiguous()
        and (bias is None or bias.is_contiguous())
        and left.shape[1] > 0
        and weight.shape[0] > 0
        and all(
            value.is_cuda and value.dtype is torch.int64 and value.dim() == 1 and value.is_contiguous()
            for value in metadata
        )
        and msa_outer_offsets.shape == msa_packed_offsets.shape == pair_offsets.shape
        and msa_outer_offsets.shape[0] > 1
    )
    return supported
