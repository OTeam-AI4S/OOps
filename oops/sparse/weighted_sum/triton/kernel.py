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

import triton
import triton.language as tl

from ...csr.triton.kernel import long_start_bucket


@triton.jit
def _bounded_edge(edge, num_edges):

    return tl.minimum(tl.maximum(edge.to(tl.int64), 0), num_edges)


@triton.jit
def _csr_weighted_sum_forward(
    values_ptr,
    weights_ptr,
    row_offsets_ptr,
    out_ptr,
    inferred_segments,
    num_edges,
    features,
    inner_size: tl.constexpr,
    weight_group_dim: tl.constexpr,
    weight_e_stride: tl.constexpr,
    weight_group_stride: tl.constexpr,
    weight_kind: tl.constexpr,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    segment_parts: tl.constexpr,
) -> None:
    slot = tl.program_id(0).to(tl.int64)
    segment = slot // segment_parts
    part = slot % segment_parts
    columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    start = _bounded_edge(tl.load(row_offsets_ptr + segment, mask=segment < inferred_segments, other=0), num_edges)
    stop = _bounded_edge(tl.load(row_offsets_ptr + segment + 1, mask=segment < inferred_segments, other=0), num_edges)
    length = tl.maximum(stop - start, 0)
    stop = start + length * (part + 1) // segment_parts
    start += length * part // segment_parts
    accumulator = tl.full((block_edges, block_features), 0, tl.float64 if use_float64 else tl.float32)
    for begin in range(start, stop, block_edges):
        edges = begin + rows
        live = (edges < stop)[:, None] & (columns < features)[None, :]
        values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
        if weight_kind == 0:
            weights = tl.load(weights_ptr + edges, mask=edges < stop, other=0)[:, None]
        elif weight_kind == 1:
            weight_group_indices = tl.where(weight_group_dim == 1, 0, columns // inner_size)
            weights = tl.load(
                weights_ptr + edges[:, None] * weight_e_stride + weight_group_indices[None, :] * weight_group_stride,
                mask=live,
                other=0,
            )
        else:
            weights = tl.load(weights_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
        accumulator += values.to(accumulator.dtype) * weights.to(accumulator.dtype)
    tl.store(out_ptr + slot * features + columns, tl.sum(accumulator, 0), mask=columns < features)


@triton.jit
def _csr_weighted_sum_forward_adaptive(
    values_ptr,
    weights_ptr,
    row_offsets_ptr,
    out_ptr,
    inferred_segments,
    out_segments,
    num_edges,
    features,
    inner_size: tl.constexpr,
    weight_group_dim: tl.constexpr,
    weight_e_stride: tl.constexpr,
    weight_group_stride: tl.constexpr,
    weight_kind: tl.constexpr,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    long_features: tl.constexpr,
    long_edges: tl.constexpr,
    threshold: tl.constexpr,
    skip_long_rows: tl.constexpr = False,
    long_descriptors_ptr=None,
    counter_tiles: tl.constexpr = 0,
):
    work = tl.program_id(0).to(tl.int64)
    if work < out_segments:
        start = _bounded_edge(tl.load(row_offsets_ptr + work, mask=work < inferred_segments, other=0), num_edges)
        stop = _bounded_edge(tl.load(row_offsets_ptr + work + 1, mask=work < inferred_segments, other=0), num_edges)
        if stop - start < threshold:
            _csr_weighted_sum_forward(
                values_ptr,
                weights_ptr,
                row_offsets_ptr,
                out_ptr,
                inferred_segments,
                num_edges,
                features,
                inner_size,
                weight_group_dim,
                weight_e_stride,
                weight_group_stride,
                weight_kind,
                use_float64,
                block_features,
                block_edges,
                1,
            )
    elif skip_long_rows:
        if tl.program_id(1) == 0:
            bucket = work - out_segments
            segment, start, stop, active = long_start_bucket(
                row_offsets_ptr, num_edges, inferred_segments, bucket, threshold
            )
            tl.store(long_descriptors_ptr + bucket * 3, segment)
            tl.store(long_descriptors_ptr + bucket * 3 + 1, start)
            tl.store(long_descriptors_ptr + bucket * 3 + 2, tl.where(active, stop, start))
            if counter_tiles:
                counters = (long_descriptors_ptr + (num_edges // threshold) * 3).to(tl.pointer_type(tl.int32))
                lanes = tl.arange(0, 32)
                for base in range(0, counter_tiles, 32):
                    column = base + lanes
                    tl.store(counters + bucket * counter_tiles + column, 0, mask=column < counter_tiles)
    else:
        tiles: tl.constexpr = block_features // long_features
        extra = work - out_segments
        feature_start = tl.program_id(1) * block_features + (extra % tiles) * long_features
        segment, start, stop, active = long_start_bucket(
            row_offsets_ptr, num_edges, inferred_segments, extra // tiles, threshold
        )
        if active & (feature_start < features):
            columns = feature_start + tl.arange(0, long_features)
            rows = tl.arange(0, long_edges).to(tl.int64)
            accumulator = tl.full((long_edges, long_features), 0, tl.float64 if use_float64 else tl.float32)
            for begin in range(start, stop, long_edges):
                edges = begin + rows
                live = (edges < stop)[:, None] & (columns < features)[None, :]
                values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                if weight_kind == 0:
                    weights = tl.load(weights_ptr + edges, mask=edges < stop, other=0)[:, None]
                elif weight_kind == 1:
                    weight_group_indices = tl.where(weight_group_dim == 1, 0, columns // inner_size)
                    weights = tl.load(
                        weights_ptr
                        + edges[:, None] * weight_e_stride
                        + weight_group_indices[None, :] * weight_group_stride,
                        mask=live,
                        other=0,
                    )
                else:
                    weights = tl.load(weights_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                accumulator += values.to(accumulator.dtype) * weights.to(accumulator.dtype)
            tl.store(out_ptr + segment * features + columns, tl.sum(accumulator, 0), mask=columns < features)


@triton.jit
def _csr_weighted_sum_backward(
    values_ptr,
    weights_ptr,
    grad_out_ptr,
    offsets_ptr,
    segment_ids_ptr,
    grad_values_ptr,
    grad_weights_ptr,
    num_edges,
    num_segments,
    groups_per_edge: tl.constexpr,
    group_features: tl.constexpr,
    has_segment_ids: tl.constexpr,
    segment_id_stride: tl.constexpr,
    need_grad_values: tl.constexpr,
    need_grad_weights: tl.constexpr,
    use_float64: tl.constexpr,
    block_groups: tl.constexpr,
    block_features: tl.constexpr,
) -> None:
    group_count = tl.cast(num_edges, tl.int64) * groups_per_edge
    columns = tl.arange(0, block_features)
    first_group = tl.program_id(0).to(tl.int64) * block_groups
    group_stride = tl.num_programs(0).to(tl.int64) * block_groups
    for group_begin in range(first_group, group_count, group_stride):
        groups = group_begin + tl.arange(0, block_groups)
        edges = groups // groups_per_edge
        weight_group_indices = groups % groups_per_edge
        valid_groups = groups < group_count
        if has_segment_ids:
            segments = tl.load(segment_ids_ptr + edges * segment_id_stride, mask=valid_groups, other=num_segments).to(
                tl.int64
            )
        else:
            low = tl.zeros((block_groups,), tl.int64)
            high = tl.full((block_groups,), num_segments, tl.int64)
            while tl.sum((low < high).to(tl.int32), 0) != 0:
                middle = (low + high) // 2
                boundary = tl.load(offsets_ptr + middle + 1, mask=low < high, other=0).to(tl.int64)
                advance = (low < high) & (edges >= boundary)
                high = tl.where((low < high) & ~advance, middle, high)
                low = tl.where(advance, middle + 1, low)
            segments = low
        first_edge = tl.load(offsets_ptr).to(tl.int64)
        covered = valid_groups & (edges >= first_edge) & (segments >= 0) & (segments < num_segments)
        accumulator = tl.full((block_groups, block_features), 0, tl.float64 if use_float64 else tl.float32)
        if need_grad_values:
            weights = tl.load(weights_ptr + groups, mask=covered, other=0).to(accumulator.dtype)
        for begin in range(0, group_features, block_features):
            features = begin + columns
            live = valid_groups[:, None] & (features < group_features)[None, :]
            value_offsets = groups[:, None] * group_features + features[None, :]
            upstream_offsets = (
                segments[:, None] * groups_per_edge + weight_group_indices[:, None]
            ) * group_features + features[None, :]
            upstream = tl.load(
                grad_out_ptr + upstream_offsets, mask=covered[:, None] & (features < group_features)[None, :], other=0
            ).to(accumulator.dtype)
            if need_grad_values:
                tl.store(grad_values_ptr + value_offsets, upstream * weights[:, None], mask=live)
            if need_grad_weights:
                values = tl.load(
                    values_ptr + value_offsets,
                    mask=covered[:, None] & (features < group_features)[None, :],
                    other=0,
                ).to(accumulator.dtype)
                accumulator += upstream * values
        if need_grad_weights:
            tl.store(grad_weights_ptr + groups, tl.sum(accumulator, 1), mask=valid_groups)


@triton.jit
def _csr_weighted_backward_edges(
    values_ptr,
    weights_ptr,
    upstream_ptr,
    offsets_ptr,
    ids_ptr,
    grad_values_ptr,
    grad_weights_ptr,
    num_edges,
    num_segments,
    groups_per_edge: tl.constexpr,
    group_features: tl.constexpr,
    has_ids: tl.constexpr,
    id_stride: tl.constexpr,
    need_values: tl.constexpr,
    need_weights: tl.constexpr,
    use_float64: tl.constexpr,
    block_edges: tl.constexpr,
    block_groups: tl.constexpr,
    block_features: tl.constexpr,
):
    edges = tl.program_id(0).to(tl.int64) * block_edges + tl.arange(0, block_edges).to(tl.int64)
    valid = edges < num_edges
    if has_ids:
        segment = tl.load(ids_ptr + edges * id_stride, mask=valid, other=num_segments).to(tl.int64)
    else:
        low = tl.zeros((block_edges,), tl.int64)
        high = tl.full((block_edges,), num_segments, tl.int64)
        while tl.sum((low < high).to(tl.int32), 0) != 0:
            middle = (low + high) // 2
            boundary = tl.load(offsets_ptr + middle + 1, mask=low < high, other=0).to(tl.int64)
            advance = (low < high) & (edges >= boundary)
            high = tl.where((low < high) & ~advance, middle, high)
            low = tl.where(advance, middle + 1, low)
        segment = low
    first_edge = tl.load(offsets_ptr).to(tl.int64)
    covered = valid & (edges >= first_edge) & (segment >= 0) & (segment < num_segments)
    groups = tl.program_id(1) * block_groups + tl.arange(0, block_groups)
    columns = tl.arange(0, block_features)
    live = valid[:, None, None] & (groups < groups_per_edge)[None, :, None] & (columns < group_features)[None, None, :]
    value_index = (edges[:, None, None] * groups_per_edge + groups[None, :, None]) * group_features + columns[
        None, None, :
    ]
    grad_index = (segment[:, None, None] * groups_per_edge + groups[None, :, None]) * group_features + columns[
        None, None, :
    ]
    upstream = tl.load(
        upstream_ptr + grad_index,
        mask=covered[:, None, None]
        & (groups < groups_per_edge)[None, :, None]
        & (columns < group_features)[None, None, :],
        other=0,
    ).to(tl.float64 if use_float64 else tl.float32)
    if need_values:
        weight = tl.load(
            weights_ptr + edges[:, None] * groups_per_edge + groups[None, :],
            mask=covered[:, None] & (groups < groups_per_edge)[None, :],
            other=0,
        ).to(upstream.dtype)
        tl.store(grad_values_ptr + value_index, upstream * weight[:, :, None], mask=live)
    if need_weights:
        value = tl.load(
            values_ptr + value_index,
            mask=covered[:, None, None]
            & (groups < groups_per_edge)[None, :, None]
            & (columns < group_features)[None, None, :],
            other=0,
        ).to(upstream.dtype)
        result = tl.sum(upstream * value, 2)
        tl.store(
            grad_weights_ptr + edges[:, None] * groups_per_edge + groups[None, :],
            result,
            mask=valid[:, None] & (groups < groups_per_edge)[None, :],
        )


@triton.jit
def _csr_weighted_sum_forward_middle(
    values_ptr,
    out_ptr,
    start,
    stop,
    segment,
    features,
    owned_begin,
    owned_end,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    weights_ptr=None,
    weight_groups: tl.constexpr = 0,
    weight_features: tl.constexpr = 1,
):
    first = tl.maximum(0, owned_begin - segment * features) // block_features * block_features
    last = tl.minimum(features, owned_end - segment * features)
    for column_begin in range(first, last, block_features):
        columns = column_begin + tl.arange(0, block_features)
        owned = (
            (segment * features + columns >= owned_begin)
            & (segment * features + columns < owned_end)
            & (columns < features)
        )
        rows = tl.arange(0, block_edges).to(tl.int64)
        initial: tl.constexpr = 0
        accumulator = tl.full((block_edges, block_features), initial, tl.float64 if use_float64 else tl.float32)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & owned[None, :]
            values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=initial).to(
                accumulator.dtype
            )
            if weight_groups:
                weights = tl.load(
                    weights_ptr + edges[:, None] * weight_groups + (columns // weight_features)[None, :],
                    mask=live,
                    other=0,
                )
                values *= weights.to(accumulator.dtype)
            accumulator += values
        total = tl.sum(accumulator, 0)
        tl.store(out_ptr + segment * features + columns, total, mask=owned)


@triton.jit
def _csr_weighted_sum_forward_fused(
    values_ptr,
    offsets_ptr,
    out_ptr,
    partial_ptr,
    counter_ptr,
    inferred_segments: tl.constexpr,
    out_segments: tl.constexpr,
    num_edges: tl.constexpr,
    features: tl.constexpr,
    use_float64: tl.constexpr,
    buckets: tl.constexpr,
    parts: tl.constexpr,
    long_features: tl.constexpr,
    block_edges: tl.constexpr,
    block: tl.constexpr,
    threshold: tl.constexpr,
    weights_ptr=None,
    weight_groups: tl.constexpr = 0,
    weight_features: tl.constexpr = 1,
):
    work = tl.program_id(0)
    flat_programs: tl.constexpr = triton.cdiv(out_segments * features, block)
    feature_tiles: tl.constexpr = triton.cdiv(features, long_features)
    if work < flat_programs:
        indices = work * block + tl.arange(0, block)
        if out_segments * features >= 2**31:
            indices = indices.to(tl.int64)
        segment = indices // features
        column = indices % features
        live = indices < out_segments * features
        start = _bounded_edge(
            tl.load(offsets_ptr + segment, mask=live & (segment < inferred_segments), other=0), num_edges
        )
        stop = _bounded_edge(
            tl.load(offsets_ptr + segment + 1, mask=live & (segment < inferred_segments), other=0), num_edges
        )
        length = tl.maximum(stop - start, 0)
        short = length <= 32
        middle = (length > 32) & (length < threshold)
        length = tl.where(short, length, 0)
        initial: tl.constexpr = 0
        acc = tl.full((block,), initial, tl.float64 if use_float64 else tl.float32)
        for row in range(0, tl.max(length, 0)):
            edge = start + row
            mask = live & (row < length)
            value = tl.load(values_ptr + edge * features + column, mask=mask, other=initial).to(acc.dtype)
            if weight_groups:
                weight = tl.load(weights_ptr + edge * weight_groups + column // weight_features, mask=mask, other=0)
                value *= weight.to(acc.dtype)
            acc += value
        tl.store(out_ptr + indices, acc, mask=live & short)
        middle_segment = tl.min(tl.where(live & middle, segment, out_segments), 0)
        while middle_segment < out_segments:
            middle_start = _bounded_edge(
                tl.load(offsets_ptr + middle_segment, mask=middle_segment < inferred_segments, other=0), num_edges
            )
            middle_stop = _bounded_edge(
                tl.load(offsets_ptr + middle_segment + 1, mask=middle_segment < inferred_segments, other=0),
                num_edges,
            )
            _csr_weighted_sum_forward_middle(
                values_ptr,
                out_ptr,
                middle_start,
                middle_stop,
                middle_segment,
                features,
                work * block,
                work * block + block,
                use_float64,
                64,
                32,
                weights_ptr,
                weight_groups,
                weight_features,
            )
            middle_segment = tl.min(tl.where(live & middle & (segment > middle_segment), segment, out_segments), 0)

    else:
        extra = work - flat_programs
        bucket = extra // (parts * feature_tiles)
        part, feature_tile = extra // feature_tiles % parts, extra % feature_tiles
        columns = feature_tile * long_features + tl.arange(0, long_features)
        rows = tl.arange(0, block_edges).to(tl.int64)
        if bucket < buckets:
            segment, start, stop, active = long_start_bucket(
                offsets_ptr, num_edges, inferred_segments, bucket, threshold
            )
            if active:
                length = stop - start
                part_stop = start + length * (part + 1) // parts
                part_start = start + length * part // parts
                accumulator = tl.zeros((block_edges, long_features), tl.float64 if use_float64 else tl.float32)
                for begin in range(part_start, part_stop, block_edges):
                    edges = begin + rows
                    values = tl.load(
                        values_ptr + edges[:, None] * features + columns[None, :],
                        mask=(edges < part_stop)[:, None] & (columns < features)[None, :],
                        other=0,
                    )
                    contribution = values.to(accumulator.dtype)
                    if weight_groups:
                        weight = tl.load(
                            weights_ptr + edges[:, None] * weight_groups + (columns // weight_features)[None, :],
                            mask=(edges < part_stop)[:, None] & (columns < features)[None, :],
                            other=0,
                        )
                        contribution *= weight.to(accumulator.dtype)
                    accumulator += contribution
                slot = (bucket * parts + part).to(tl.int64)
                partial = tl.sum(accumulator, 0)
                tl.store(
                    partial_ptr + slot * features + columns, partial, mask=columns < features, cache_modifier=".cg"
                )
                # Publish every lane's partial before signaling block completion.
                tl.debug_barrier()
                ticket = tl.atomic_add(
                    counter_ptr + bucket * feature_tiles + feature_tile, 1, sem="acq_rel", scope="gpu"
                )
                tl.debug_barrier()
                if ticket == parts - 1:
                    part_rows = bucket * parts + tl.arange(0, parts)
                    values = tl.load(
                        partial_ptr + part_rows[:, None] * features + columns[None, :],
                        mask=(columns < features)[None, :],
                        other=0,
                        cache_modifier=".cg",
                    )
                    total = tl.sum(values, 0)
                    tl.store(out_ptr + segment * features + columns, total, mask=columns < features)
