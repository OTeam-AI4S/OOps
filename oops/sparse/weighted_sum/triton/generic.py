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

from oops.core._math import _next_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ, cuda_device_context, tensors_share_device

from ...csr.triton import kernel as csr_kernel
from ...csr.triton.backend import (
    finish_long_reduction,
    flat_features,
    prepare_long_reduction,
    reduction_part_count,
    resolve_execution_plan,
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


def _execute_weighted_sum_forward(
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
