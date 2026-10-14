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

from functools import lru_cache

import torch
from torch import Tensor

from oops.core._math import _next_power_of_two
from oops.runtime import (
    CUDA_MAX_GRID_X,
    CUDA_MAX_GRID_YZ,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps_if_needed,
    cuda_device_context,
    is_compiling,
    tensors_share_device,
)

from . import kernel
from .schedule import (
    CSR_SUM_SCHEDULE,
    LONG_BLOCK_EDGES,
    LONG_ROW_EDGES,
    SegmentPlan,
    SegmentProblem,
    SegmentSchedule,
    adaptive_reduction_geometry,
)

_SUPPORTED_DTYPES = (*STANDARD_FLOAT_DTYPES, torch.float64)


def supports_reduction_layout(values: Tensor, row_offsets: Tensor, out_segments: int) -> bool:
    return (
        values.is_cuda
        and row_offsets.is_cuda
        and tensors_share_device(values, row_offsets)
        and values.is_floating_point()
        and values.dtype in _SUPPORTED_DTYPES
        and row_offsets.dtype in (torch.int32, torch.int64)
        and values.dim() >= 1
        and values.is_contiguous()
        and row_offsets.dim() == 1
        and row_offsets.is_contiguous()
        and out_segments > 0
        and out_segments <= CUDA_MAX_GRID_X
        and values.shape[0] > 0
        and flat_features(values) > 0
    )


def schedule_supports(schedule: SegmentSchedule, values: Tensor, out_segments: int) -> bool:
    features = flat_features(values)
    return (out_segments <= CUDA_MAX_GRID_X) & (features <= CUDA_MAX_GRID_YZ * schedule.config.block_features)


@lru_cache(maxsize=1024)
def _cached_segment_execution_plan(
    schedule: SegmentSchedule,
    out_segments: int,
    features: int,
) -> SegmentPlan:

    return schedule.resolve(SegmentProblem(out_segments=out_segments, features=features))


def resolve_execution_plan(
    schedule: SegmentSchedule,
    values: Tensor,
    out_segments: int,
) -> SegmentPlan:
    features = flat_features(values)
    if is_compiling():
        return schedule.resolve(SegmentProblem(out_segments=out_segments, features=features))
    return _cached_segment_execution_plan(schedule, out_segments, features)


def reduction_part_count(
    values: Tensor,
    num_edges: int,
    inferred_segments: int,
    out_segments: int,
    block_features: int,
) -> int:

    if inferred_segments == 0:
        return 1
    average = num_edges // inferred_segments
    if average < 1024:
        return 1
    target = cuda_device_caps_if_needed(values.device, True)
    multiprocessors = None if target is None else target.resources.multiprocessor_count
    if multiprocessors is None:
        return 1
    parts = min(16, 1 << ((average // 256).bit_length() - 1))
    features = flat_features(values)
    programs = out_segments * ((features + block_features - 1) // block_features)
    part_bytes = 2 * out_segments * features * (8 if values.dtype == torch.float64 else 4)
    while parts > 1 and (
        programs * (parts // 2) >= multiprocessors * 4
        or out_segments * parts > CUDA_MAX_GRID_X
        or parts * part_bytes > 16 << 20
    ):
        parts //= 2
    return parts


def rows_per_edge(per_segment: Tensor, segment_ids: Tensor, inferred_segments: int) -> Tensor:
    r"""Uncovered edges and trailing empty segments contribute zero."""

    covered = per_segment.narrow(0, 0, inferred_segments)
    padded = torch.cat((covered, covered.new_zeros((1, *covered.shape[1:]))))
    return padded.index_select(0, segment_ids)


def flat_features(values: Tensor) -> int:
    features = 1
    for dim in values.shape[1:]:
        features *= dim
    return features


def prepare_long_reduction(
    values: Tensor, out_segments: int, inferred_segments: int, block_features: int
) -> tuple[Tensor, Tensor, int, int] | None:
    if values.shape[0] < LONG_ROW_EDGES or inferred_segments == 0:
        return None
    features = flat_features(values)
    buckets = values.shape[0] // LONG_ROW_EDGES
    parts = 128
    itemsize = 8 if values.dtype == torch.float64 else 4
    tiles = (features + min(8, block_features) - 1) // min(8, block_features)
    while parts > 1 and (buckets * parts * features * itemsize + buckets * 24 + buckets * tiles * 4 > 16 << 20):
        parts //= 2
    if (
        parts == 1
        or out_segments + buckets > CUDA_MAX_GRID_X
        or (features + min(8, block_features) - 1) // min(8, block_features) > CUDA_MAX_GRID_YZ
    ):
        return None
    descriptors = torch.empty((buckets * 3 + (buckets * tiles + 1) // 2,), device=values.device, dtype=torch.int64)
    partial = torch.empty(
        (buckets * parts, features),
        device=values.device,
        dtype=torch.float64 if values.dtype == torch.float64 else torch.float32,
    )
    return descriptors, partial, buckets, parts


def finish_long_reduction(
    values: Tensor,
    descriptors: Tensor,
    partial: Tensor,
    out: Tensor,
    buckets: int,
    parts: int,
    block_features: int,
    num_warps: int,
    reduction: str,
    weights: Tensor | None = None,
) -> None:
    values = values.resolve_neg()
    descriptors = descriptors.resolve_neg()
    partial = partial.resolve_neg()
    weights = None if weights is None else weights.resolve_neg()
    features = flat_features(values)
    long_features = min(8, block_features)
    if reduction == "sum":
        tiles = (features + long_features - 1) // long_features
        kernel.long_sum_finish.run(
            values,
            descriptors,
            partial,
            out,
            values.shape[0],
            features,
            values.dtype == torch.float64,
            buckets,
            parts,
            long_features,
            LONG_BLOCK_EDGES,
            grid=(parts, tiles),
            warmup=False,
            num_warps=num_warps,
            weights_ptr=values if weights is None else weights,
            weight_groups=0 if weights is None else weights.numel() // values.shape[0],
            weight_features=1 if weights is None else features // (weights.numel() // values.shape[0]),
        )
        return
    kernel.long_reduction_partials.run(
        values,
        descriptors,
        partial,
        values.shape[0],
        features,
        values.dtype == torch.float64,
        reduction,
        parts,
        long_features,
        LONG_BLOCK_EDGES,
        LONG_ROW_EDGES,
        weights_ptr=values if weights is None else weights,
        weight_groups=0 if weights is None else weights.numel() // values.shape[0],
        weight_features=1 if weights is None else features // (weights.numel() // values.shape[0]),
        num_warps=num_warps,
        grid=(parts, (features + long_features - 1) // long_features),
        warmup=False,
    )
    kernel.long_reduction_merge.run(
        descriptors,
        partial,
        out,
        buckets,
        features,
        reduction,
        parts,
        block_features,
        num_warps=num_warps,
        grid=(min(buckets, 128), (features + block_features - 1) // block_features),
        warmup=False,
    )


def fused_reduction_csr(
    values: Tensor, row_offsets: Tensor, out_segments: int, inferred_segments: int, reduction: str = "sum"
) -> Tensor | None:
    features = flat_features(values)
    buckets = values.shape[0] // LONG_ROW_EDGES
    parts = 16
    if reduction == "max" and buckets:
        target = cuda_device_caps_if_needed(values.device, True)
        multiprocessors = None if target is None else target.resources.multiprocessor_count
        desired = (
            128
            if multiprocessors is None
            else max(
                2, (multiprocessors * 4 + buckets * ((features + 7) // 8) - 1) // (buckets * ((features + 7) // 8))
            )
        )
        parts = min(128, _next_power_of_two(desired))
    itemsize = 8 if values.dtype == torch.float64 else 4
    tiles = (features + 7) // 8
    while parts > 1 and buckets * (parts * features * itemsize + tiles * 4) > 16 << 20:
        parts //= 2
    programs = (out_segments * features + 511) // 512 + buckets * parts * tiles
    if buckets == 0 or parts == 1 or programs > CUDA_MAX_GRID_X or inferred_segments == 0:
        return None
    values = values.resolve_neg()
    row_offsets = row_offsets.resolve_neg()
    with cuda_device_context(values.device):
        out = torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)
        partial = torch.empty(
            (buckets * parts, features),
            device=values.device,
            dtype=torch.float64 if values.dtype == torch.float64 else torch.float32,
        )
        counters = torch.zeros((buckets, tiles), device=values.device, dtype=torch.int32)
        kernel.reduction_forward_fused.run(
            values,
            row_offsets,
            out,
            partial,
            counters,
            inferred_segments,
            out_segments,
            values.shape[0],
            features,
            values.dtype == torch.float64,
            buckets,
            parts,
            8,
            LONG_BLOCK_EDGES,
            512,
            LONG_ROW_EDGES,
            grid=(programs,),
            warmup=False,
            num_warps=4,
            reduction=reduction,
        )
        return out


def sum_csr(
    values: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    *,
    plan: SegmentPlan | None = None,
) -> Tensor:
    # out_segments includes trailing empty rows; inferred_segments counts offset rows.
    fused = fused_reduction_csr(values, row_offsets, out_segments, inferred_segments)
    if fused is not None:
        return fused
    with cuda_device_context(values.device):
        values = values.resolve_neg()
        row_offsets = row_offsets.resolve_neg()
        features = flat_features(values)
        if plan is None:
            config = CSR_SUM_SCHEDULE.config
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
        out = torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)
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
                flat_programs = (out_segments * features + 511) // 512 + buckets
                if (
                    16 <= features <= 512
                    and values.shape[0] <= inferred_segments * 32
                    and flat_programs <= CUDA_MAX_GRID_X
                ):
                    kernel.sum_forward_owned.run(
                        values,
                        row_offsets,
                        out,
                        descriptors,
                        inferred_segments,
                        out_segments,
                        values.shape[0],
                        features,
                        values.dtype == torch.float64,
                        512,
                        LONG_ROW_EDGES,
                        counter_tiles=(features + min(8, block_features) - 1) // min(8, block_features),
                        num_warps=4,
                        grid=((out_segments * features + 511) // 512 + buckets,),
                        warmup=False,
                    )
                else:
                    kernel.sum_forward_adaptive.run(
                        values,
                        row_offsets,
                        out,
                        inferred_segments,
                        out_segments,
                        values.shape[0],
                        features,
                        values.dtype == torch.float64,
                        block_features,
                        CSR_SUM_SCHEDULE.config.max_block_edges,
                        min(8, block_features),
                        LONG_BLOCK_EDGES,
                        LONG_ROW_EDGES,
                        skip_long_rows=True,
                        long_descriptors_ptr=descriptors,
                        counter_tiles=(features + min(8, block_features) - 1) // min(8, block_features),
                        num_warps=num_warps,
                        grid=(out_segments + buckets, grid[1]),
                        warmup=False,
                    )
                finish_long_reduction(
                    values, descriptors, long_partial, out, buckets, long_parts, block_features, num_warps, "sum"
                )
                return out
            programs, long_features = adaptive
            kernel.sum_forward_adaptive.run(
                values,
                row_offsets,
                out,
                inferred_segments,
                out_segments,
                values.shape[0],
                features,
                values.dtype == torch.float64,
                block_features,
                CSR_SUM_SCHEDULE.config.max_block_edges,
                long_features,
                LONG_BLOCK_EDGES,
                LONG_ROW_EDGES,
                num_warps=num_warps,
                grid=(programs, grid[1]),
                warmup=False,
            )
            return out
        kernel.sum_forward.run(
            values,
            row_offsets,
            partial,
            inferred_segments,
            values.shape[0],
            features,
            values.dtype == torch.float64,
            block_features,
            CSR_SUM_SCHEDULE.config.max_block_edges,
            parts,
            num_warps=num_warps,
            grid=launch_grid,
            warmup=False,
        )
        if parts > 1:
            kernel.merge_partials.run(
                partial, out, features, "sum", parts, block_features, num_warps=num_warps, grid=grid, warmup=False
            )
        return out
