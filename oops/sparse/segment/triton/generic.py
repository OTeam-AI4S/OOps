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
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ, cuda_device_context

from ...csr.triton import kernel as csr_kernel
from ...csr.triton.backend import (
    finish_long_reduction,
    flat_features,
    fused_reduction_csr,
    prepare_long_reduction,
    reduction_part_count,
    resolve_execution_plan,
    schedule_supports,
    sum_csr,
    supports_reduction_layout,
)
from ...csr.triton.schedule import LONG_BLOCK_EDGES, LONG_ROW_EDGES, SegmentPlan, adaptive_reduction_geometry
from . import kernel
from .profiles import SEGMENT_MAX_SCHEDULE, SEGMENT_SOFTMAX_ADAPTIVE, SEGMENT_SOFTMAX_SCHEDULE, SEGMENT_SUM_SCHEDULE
from .softmax_schedule import adaptive_geometry


def supports_segment_sum(values: Tensor, row_offsets: Tensor, out_segments: int) -> bool:

    return supports_reduction_layout(values, row_offsets, out_segments) and schedule_supports(
        SEGMENT_SUM_SCHEDULE, values, out_segments
    )


def _execute_segment_sum_forward(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del segment_ids
    plan = resolve_execution_plan(SEGMENT_SUM_SCHEDULE, values, out_segments)
    return sum_csr(values, row_offsets, out_segments, inferred_segments, plan=plan)


def supports_segment_max(values: Tensor, row_offsets: Tensor, out_segments: int) -> bool:

    return supports_reduction_layout(values, row_offsets, out_segments) and schedule_supports(
        SEGMENT_MAX_SCHEDULE, values, out_segments
    )


def _execute_segment_max_forward(
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


def supports_segment_softmax(logits: Tensor, row_offsets: Tensor, inferred_segments: int) -> bool:

    return supports_reduction_layout(logits, row_offsets, inferred_segments) and schedule_supports(
        SEGMENT_SOFTMAX_SCHEDULE, logits, inferred_segments
    )


def _execute_segment_softmax_forward(
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
