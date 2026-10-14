# OOps
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OOps.

# OOps is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OOps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://oops.valtech.ai/about/license-faq/>.

from __future__ import annotations

import torch
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true

from oops.runtime import (
    FLOAT32_DOT_TF32X3,
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps,
    cuda_device_context,
    is_compiling,
    is_external_fake,
    matmul_precision,
    require_triton_dot_precision,
    tensors_share_device,
)
from oops.runtime.cuda import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ

from ...flash import degenerate_shape, pair_mask_kind
from ...flash.triton import (
    AttentionAdmissionKey,
    AttentionLaunchContext,
    AttentionPhase,
    AttentionProblem,
    AttentionTensorLayout,
    FlashAttentionBackend,
    attention_admission_key,
    attention_bias_strides,
    attention_grids_launchable,
    attention_shared_bias_strides,
    attention_tensor_strides,
    optional_tensor,
    pair_mask_strides,
    resolve_scale,
)
from ...flash.triton.generic import _resolve_attention_operand
from ...flash.triton.schedule import SegmentedAttentionPlan, SegmentedAttentionProblem
from . import kernel
from .profiles import SCHEDULES, SEGMENTED_SCHEDULES

_PAIR_BIAS_FLOAT32_DOT = FLOAT32_DOT_TF32X3


def _pair_bias_attention_grids_launchable(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
) -> bool:
    if is_compiling():
        torch._dynamo.mark_static(query, -1)
    return attention_grids_launchable(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        mask,
        pair_bias,
        maximum_grid_x=CUDA_MAX_GRID_X,
        maximum_grid_yz=CUDA_MAX_GRID_YZ,
    )


# The sample axis S is the shared kernel group axis, not its batch axis.


def _bind_pair_bias_problem(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    *,
    bind_launch_strides: bool,
) -> AttentionProblem:
    batch_size, num_samples, num_heads, query_length, head_size = degenerate_shape(query, 5)
    key_length = key.shape[-2]
    tensor_layout = AttentionTensorLayout.HEAD_TOKEN
    mask_kind = pair_mask_kind(query, mask)
    bias_groups = 1 if pair_bias is None else pair_bias.shape[-4]
    bias_strides = attention_bias_strides(pair_bias) if bind_launch_strides else (0, 0, 0, 0, 0)
    if bind_launch_strides and statically_known_true(bias_groups == 1):
        batch_stride, _group_stride, head_stride, query_stride, key_stride = bias_strides
        bias_strides = batch_stride, 0, head_stride, query_stride, key_stride
    return AttentionProblem(
        batch_size=batch_size,
        num_groups=num_samples,
        context=AttentionLaunchContext(
            num_heads=num_heads,
            query_length=query_length,
            key_length=key_length,
            head_size=head_size,
            mask_kind=mask_kind,
            element_size=query.element_size(),
        ),
        tensor_layout=tensor_layout,
        lse_shape=tuple(query.shape[:-1]),
        query_strides=(attention_tensor_strides(tensor_layout, query) if bind_launch_strides else (0, 0, 0, 0, 0)),
        key_strides=attention_tensor_strides(tensor_layout, key) if bind_launch_strides else (0, 0, 0, 0, 0),
        value_strides=(attention_tensor_strides(tensor_layout, value) if bind_launch_strides else (0, 0, 0, 0, 0)),
        mask_strides=pair_mask_strides(mask_kind, mask) if bind_launch_strides else (0, 0, 0, 0),
        bias_strides=bias_strides,
        shared_bias_strides=(
            attention_shared_bias_strides(pair_bias) if bind_launch_strides and pair_bias is not None else (0, 0, 0, 0)
        ),
        bias_groups=bias_groups,
    )


_PAIR_BIAS_BACKEND = FlashAttentionBackend(
    operator="pair_bias_attention",
    float32_dot=_PAIR_BIAS_FLOAT32_DOT,
    bind_problem=_bind_pair_bias_problem,
    schedules=SCHEDULES,
    contiguous_fake_bias_gradient=True,
)


_SEGMENTED_PAIR_BIAS_FLOAT32_DOT = FLOAT32_DOT_TF32X3


_SEGMENTED_PAIR_BIAS_EAGER_ADMISSION: tuple[AttentionAdmissionKey, int, int, int, int] | None = None


def _segmented_problem(
    query: Tensor,
    mask_kind: int,
    query_sample_offsets: Tensor,
    precision: str,
) -> SegmentedAttentionProblem:
    return SegmentedAttentionProblem(
        num_samples=query_sample_offsets.shape[0] - 1,
        total_positions=query.shape[0] * query.shape[1],
        num_heads=query.shape[2],
        head_size=query.shape[3],
        dtype=query.dtype,
        element_size=query.element_size(),
        mask_kind=mask_kind,
        input_precision=precision,
    )


def _supports_bias_dtype(query: Tensor, pair_bias: Tensor | None) -> bool:
    if pair_bias is None:
        return True
    autocast_float32_bias = (
        query.dtype in HALF_DTYPES
        and pair_bias.dtype is torch.float32
        and pair_bias.device == query.device
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == query.dtype
    )
    return pair_bias.dtype == query.dtype or autocast_float32_bias


def supports_segmented_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    *,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
) -> bool:

    global _SEGMENTED_PAIR_BIAS_EAGER_ADMISSION

    offsets = (query_sample_offsets, key_sample_offsets, mask_sample_offsets, bias_sample_offsets)
    supported = (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and tensors_share_device(query, key, value, mask, pair_bias, *offsets)
        and query.dim() == key.dim() == value.dim() == 4
        and query.shape[1:] == key.shape[1:] == value.shape[1:]
        and query.dtype in STANDARD_FLOAT_DTYPES
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and query.shape[0] > 0
        and key.shape[0] > 0
        and query.shape[1] > 0
        and query.shape[2] > 0
        and 0 < query.shape[3] <= 256
        and all(
            offset.is_cuda and offset.dtype is torch.int64 and offset.dim() == 1 and offset.is_contiguous()
            for offset in offsets
        )
        and query_sample_offsets.shape
        == key_sample_offsets.shape
        == mask_sample_offsets.shape
        == bias_sample_offsets.shape
        and query_sample_offsets.shape[0] > 1
        and (
            mask is None
            or (mask.dtype is torch.bool and mask.dim() == 2 and mask.shape[1] == mask_groups and mask.is_contiguous())
        )
        and (
            pair_bias is None
            or (
                pair_bias.dim() == 3
                and pair_bias.shape[1] == bias_groups
                and pair_bias.shape[2] == bias_heads
                and pair_bias.is_contiguous()
                and _supports_bias_dtype(query, pair_bias)
            )
        )
    )
    if not supported:
        return False
    precision = matmul_precision(query.dtype, _SEGMENTED_PAIR_BIAS_FLOAT32_DOT)
    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(query)
    admission_key = None
    if not compiling and not external_fake:
        tensor_key = attention_admission_key(
            (query, key, value, mask, pair_bias, *offsets),
            precision=precision,
        )
        admission_key = tensor_key, mask_kind, mask_groups, bias_groups, bias_heads
        if admission_key == _SEGMENTED_PAIR_BIAS_EAGER_ADMISSION:
            return True
    problem = _segmented_problem(query, mask_kind, query_sample_offsets, precision)
    phases: tuple[AttentionPhase, ...] = (
        AttentionPhase.FORWARD,
        AttentionPhase.BACKWARD_PREPROCESS,
        AttentionPhase.BACKWARD_QUERY,
        AttentionPhase.BACKWARD_KEY_VALUE,
    )
    if pair_bias is not None:
        phases += (AttentionPhase.GRAD_BIAS,)
    supported = all(SEGMENTED_SCHEDULES.supports(problem, phase) for phase in phases)
    if supported and admission_key is not None:
        _SEGMENTED_PAIR_BIAS_EAGER_ADMISSION = admission_key
    return supported


def _resolve_segmented_plan(
    problem: SegmentedAttentionProblem,
    phase: AttentionPhase,
    device: torch.device,
) -> SegmentedAttentionPlan:
    return SEGMENTED_SCHEDULES.resolve(problem, phase, target=cuda_device_caps(device))


def _run_segmented_pair_bias_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    *,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
    write_lse: bool = True,
) -> tuple[Tensor, Tensor | None]:
    with cuda_device_context(query.device):
        original_query, original_key = query, key
        query = _resolve_attention_operand(query)
        key = query if key is original_query else _resolve_attention_operand(key)
        value = (
            query if value is original_query else key if value is original_key else _resolve_attention_operand(value)
        )
        mask = None if mask is None else _resolve_attention_operand(mask)
        pair_bias = None if pair_bias is None else _resolve_attention_operand(pair_bias)
        query_sample_offsets = _resolve_attention_operand(query_sample_offsets).contiguous()
        key_sample_offsets = _resolve_attention_operand(key_sample_offsets).contiguous()
        mask_sample_offsets = _resolve_attention_operand(mask_sample_offsets).contiguous()
        bias_sample_offsets = _resolve_attention_operand(bias_sample_offsets).contiguous()
        require_triton_dot_precision(
            "pair_bias_attention",
            query.dtype,
            _SEGMENTED_PAIR_BIAS_FLOAT32_DOT,
            precision,
        )
        problem = _segmented_problem(query, mask_kind, query_sample_offsets, precision)
        plan = _resolve_segmented_plan(problem, AttentionPhase.FORWARD, query.device)
        tile = plan.tile
        block_key = tile.block_key
        assert block_key is not None
        # Backward centering reads the normalized output before storage-dtype rounding.
        output = torch.empty_like(query, dtype=torch.float32 if write_lse else query.dtype)
        lse = torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32) if write_lse else None
        kernel._segmented_pair_bias_attention_forward[plan.grid](
            query,
            key,
            value,
            optional_tensor(mask, query),
            optional_tensor(pair_bias, query),
            output,
            optional_tensor(lse, query),
            query_sample_offsets,
            key_sample_offsets,
            mask_sample_offsets,
            bias_sample_offsets,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *((0, 0) if mask is None else mask.stride()),
            *((0, 0, 0) if pair_bias is None else pair_bias.stride()),
            *output.stride(),
            *((0, 0, 0) if lse is None else lse.stride()),
            problem.num_samples,
            query.shape[1],
            problem.num_heads,
            mask_groups,
            bias_groups,
            bias_heads,
            problem.head_size,
            plan.grid[0],
            resolve_scale(problem.head_size, scale),
            mask_kind,
            pair_bias is not None,
            write_lse,
            tile.block_query,
            block_key,
            tile.block_channel,
            input_precision=precision,
            num_warps=tile.num_warps,
            num_stages=tile.num_stages,
        )
        return output, lse


def _run_segmented_pair_bias_attention_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    output: Tensor,
    lse: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    *,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    with cuda_device_context(query.device):
        original_query, original_key = query, key
        query = _resolve_attention_operand(query)
        key = query if key is original_query else _resolve_attention_operand(key)
        needs_delta = need_grad_query or need_grad_key or need_grad_bias
        if needs_delta:
            value = (
                query
                if value is original_query
                else key if value is original_key else _resolve_attention_operand(value)
            )
            output = _resolve_attention_operand(output)
        mask = None if mask is None else _resolve_attention_operand(mask)
        pair_bias = None if pair_bias is None else _resolve_attention_operand(pair_bias)
        lse = _resolve_attention_operand(lse)
        grad_output = _resolve_attention_operand(grad_output.to(query.dtype))
        query_sample_offsets = _resolve_attention_operand(query_sample_offsets).contiguous()
        key_sample_offsets = _resolve_attention_operand(key_sample_offsets).contiguous()
        mask_sample_offsets = _resolve_attention_operand(mask_sample_offsets).contiguous()
        bias_sample_offsets = _resolve_attention_operand(bias_sample_offsets).contiguous()
        problem = _segmented_problem(query, mask_kind, query_sample_offsets, precision)
        effective_scale = resolve_scale(problem.head_size, scale)
        keep_mask = optional_tensor(mask, query)
        kernel_bias = optional_tensor(pair_bias, query)
        needs_delta = need_grad_query or need_grad_key or need_grad_bias
        delta: Tensor | None = None
        if needs_delta:
            preprocess_plan = _resolve_segmented_plan(problem, AttentionPhase.BACKWARD_PREPROCESS, query.device)
            preprocess_tile = preprocess_plan.tile
            delta = torch.empty_like(lse)
            kernel._segmented_pair_bias_attention_backward_preprocess[preprocess_plan.grid](
                output,
                grad_output,
                delta,
                query_sample_offsets,
                *output.stride(),
                *grad_output.stride(),
                *delta.stride(),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                problem.head_size,
                preprocess_plan.grid[0],
                preprocess_tile.block_query,
                preprocess_tile.block_channel,
                num_warps=preprocess_tile.num_warps,
            )

        grad_bias = torch.empty_like(pair_bias) if pair_bias is not None and need_grad_bias else None
        full_bias_gradient = grad_bias is not None and bias_groups == query.shape[1] and bias_heads == query.shape[2]
        grad_query: Tensor | None = None
        if need_grad_query or full_bias_gradient:
            assert delta is not None
            query_plan = _resolve_segmented_plan(problem, AttentionPhase.BACKWARD_QUERY, query.device)
            query_tile = query_plan.tile
            query_block_key = query_tile.block_key
            assert query_block_key is not None
            grad_query = torch.empty_like(query) if need_grad_query else None
            kernel._segmented_pair_bias_attention_backward_query[query_plan.grid](
                query,
                key,
                value,
                keep_mask,
                kernel_bias,
                lse,
                delta,
                grad_output,
                optional_tensor(grad_query, query),
                optional_tensor(grad_bias, query),
                query_sample_offsets,
                key_sample_offsets,
                mask_sample_offsets,
                bias_sample_offsets,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *((0, 0) if mask is None else mask.stride()),
                *((0, 0, 0) if pair_bias is None else pair_bias.stride()),
                *lse.stride(),
                *delta.stride(),
                *grad_output.stride(),
                *((0, 0, 0, 0) if grad_query is None else grad_query.stride()),
                *((0, 0, 0) if grad_bias is None else grad_bias.stride()),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                mask_groups,
                bias_groups,
                bias_heads,
                problem.head_size,
                query_plan.grid[0],
                effective_scale,
                mask_kind,
                pair_bias is not None,
                need_grad_query,
                full_bias_gradient,
                query_tile.block_query,
                query_block_key,
                query_tile.block_channel,
                input_precision=precision,
                num_warps=query_tile.num_warps,
                num_stages=query_tile.num_stages,
            )

        grad_key: Tensor | None = None
        grad_value: Tensor | None = None
        if need_grad_key or need_grad_value:
            key_value_plan = _resolve_segmented_plan(problem, AttentionPhase.BACKWARD_KEY_VALUE, query.device)
            key_value_tile = key_value_plan.tile
            key_value_block_key = key_value_tile.block_key
            assert key_value_block_key is not None
            grad_key = torch.empty_like(key) if need_grad_key else None
            grad_value = torch.empty_like(value) if need_grad_value else None
            kernel._segmented_pair_bias_attention_backward_key_value[key_value_plan.grid](
                query,
                key,
                value,
                keep_mask,
                kernel_bias,
                lse,
                optional_tensor(delta, lse),
                grad_output,
                optional_tensor(grad_key, key),
                optional_tensor(grad_value, value),
                query_sample_offsets,
                key_sample_offsets,
                mask_sample_offsets,
                bias_sample_offsets,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *((0, 0) if mask is None else mask.stride()),
                *((0, 0, 0) if pair_bias is None else pair_bias.stride()),
                *lse.stride(),
                *((0, 0, 0) if delta is None else delta.stride()),
                *grad_output.stride(),
                *((0, 0, 0, 0) if grad_key is None else grad_key.stride()),
                *((0, 0, 0, 0) if grad_value is None else grad_value.stride()),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                mask_groups,
                bias_groups,
                bias_heads,
                problem.head_size,
                key_value_plan.grid[0],
                effective_scale,
                mask_kind,
                pair_bias is not None,
                need_grad_key,
                need_grad_value,
                key_value_tile.block_query,
                key_value_block_key,
                key_value_tile.block_channel,
                input_precision=precision,
                num_warps=key_value_tile.num_warps,
                num_stages=key_value_tile.num_stages,
            )

        if grad_bias is not None and not full_bias_gradient:
            assert delta is not None
            assert pair_bias is not None
            grad_bias_plan = _resolve_segmented_plan(problem, AttentionPhase.GRAD_BIAS, query.device)
            grad_bias_tile = grad_bias_plan.tile
            grad_bias_block_key = grad_bias_tile.block_key
            assert grad_bias_block_key is not None
            kernel._segmented_pair_bias_attention_grad_bias[grad_bias_plan.grid](
                query,
                key,
                value,
                keep_mask,
                pair_bias,
                lse,
                delta,
                grad_output,
                grad_bias,
                query_sample_offsets,
                key_sample_offsets,
                mask_sample_offsets,
                bias_sample_offsets,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *((0, 0) if mask is None else mask.stride()),
                *pair_bias.stride(),
                *lse.stride(),
                *delta.stride(),
                *grad_output.stride(),
                *grad_bias.stride(),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                mask_groups,
                bias_groups,
                bias_heads,
                problem.head_size,
                grad_bias_plan.grid[0],
                effective_scale,
                mask_kind,
                bias_groups == 1,
                bias_heads == 1,
                grad_bias_tile.block_query,
                grad_bias_block_key,
                grad_bias_tile.block_channel,
                input_precision=precision,
                num_warps=grad_bias_tile.num_warps,
                num_stages=grad_bias_tile.num_stages,
            )

        return (
            grad_query if need_grad_query else None,
            grad_key if need_grad_key else None,
            grad_value if need_grad_value else None,
            grad_bias,
        )
