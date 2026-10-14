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

_UINT64_MAX = tl.constexpr(0xFFFFFFFFFFFFFFFF)
_UINT32_MASK = tl.constexpr(0xFFFFFFFF)
_FLOAT32_MAX = tl.constexpr(3.4028234663852886e38)


@triton.jit
def _finite_squared_distance(dx, dy, dz):
    distance_sq = dx * dx + dy * dy + dz * dz
    return tl.minimum(tl.where(distance_sq == distance_sq, distance_sq, _FLOAT32_MAX), _FLOAT32_MAX)


@triton.jit
def _segmented_masked_knn_forward(
    coordinates_ptr,
    mask_ptr,
    offsets_ptr,
    batch_indices_ptr,
    distances_ptr,
    local_indices_ptr,
    valid_ptr,
    eps,
    NUM_NEIGHBORS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
):
    query = tl.program_id(0).to(tl.int64)
    sample = tl.load(batch_indices_ptr + query).to(tl.int64)
    sample_start = tl.load(offsets_ptr + sample).to(tl.int64)
    sample_stop = tl.load(offsets_ptr + sample + 1).to(tl.int64)
    query_valid = tl.load(mask_ptr + query) != 0
    scan_stop = tl.where(query_valid, sample_stop, sample_start)

    query_x = tl.load(coordinates_ptr + query * 3, mask=query_valid, other=0.0).to(tl.float32)
    query_y = tl.load(coordinates_ptr + query * 3 + 1, mask=query_valid, other=0.0).to(tl.float32)
    query_z = tl.load(coordinates_ptr + query * 3 + 2, mask=query_valid, other=0.0).to(tl.float32)

    best_scores = tl.zeros((TOPK,), dtype=tl.uint64)
    maximum_distance_sq = 0.0
    has_valid = False
    key_lanes = tl.arange(0, BLOCK_KEY)
    for key_begin in range(sample_start, scan_stop, BLOCK_KEY):
        keys = key_begin + key_lanes
        in_range = keys < sample_stop
        candidate_valid = in_range & query_valid
        candidate_valid &= tl.load(mask_ptr + keys, mask=in_range, other=0) != 0

        key_x = tl.load(coordinates_ptr + keys * 3, mask=candidate_valid, other=0.0).to(tl.float32)
        key_y = tl.load(coordinates_ptr + keys * 3 + 1, mask=candidate_valid, other=0.0).to(tl.float32)
        key_z = tl.load(coordinates_ptr + keys * 3 + 2, mask=candidate_valid, other=0.0).to(tl.float32)
        distance_sq = _finite_squared_distance(query_x - key_x, query_y - key_y, query_z - key_z)

        maximum_distance_sq = tl.maximum(
            maximum_distance_sq,
            tl.max(tl.where(candidate_valid, distance_sq, 0.0), axis=0),
        )
        has_valid |= tl.max(candidate_valid.to(tl.int32), axis=0) != 0

        distance_bits = distance_sq.to(tl.uint32, bitcast=True).to(tl.uint64)
        local_keys = (keys - sample_start).to(tl.uint64)
        composite = (distance_bits << 32) | local_keys
        composite = tl.where(candidate_valid, composite, _UINT64_MAX)
        candidate_scores = _UINT64_MAX - composite
        if TOPK == 1:
            best_scores = tl.maximum(best_scores, tl.max(candidate_scores, axis=0))
        else:
            candidate_scores = tl.topk(candidate_scores, TOPK)
            best_scores = tl.topk(tl.cat(best_scores, candidate_scores), TOPK)

    output_lanes = tl.arange(0, TOPK)
    selected_scores = best_scores
    selected_valid = (output_lanes < NUM_NEIGHBORS) & (selected_scores != 0)
    selected_composite = _UINT64_MAX - selected_scores
    selected_local = (selected_composite & _UINT32_MASK).to(tl.int64)
    selected_local = tl.where(selected_valid, selected_local, 0)
    selected_keys = sample_start + selected_local

    key_x = tl.load(coordinates_ptr + selected_keys * 3, mask=selected_valid, other=0.0).to(tl.float32)
    key_y = tl.load(coordinates_ptr + selected_keys * 3 + 1, mask=selected_valid, other=0.0).to(tl.float32)
    key_z = tl.load(coordinates_ptr + selected_keys * 3 + 2, mask=selected_valid, other=0.0).to(tl.float32)
    selected_distance_sq = _finite_squared_distance(query_x - key_x, query_y - key_y, query_z - key_z)
    selected_distances = tl.sqrt(selected_distance_sq + eps)
    filler = tl.where(has_valid, tl.sqrt(maximum_distance_sq + eps) + 1.0, 1.0)

    output_offsets = query * NUM_NEIGHBORS + output_lanes
    output_mask = output_lanes < NUM_NEIGHBORS
    tl.store(distances_ptr + output_offsets, tl.where(selected_valid, selected_distances, filler), mask=output_mask)
    tl.store(local_indices_ptr + output_offsets, selected_local, mask=output_mask)
    tl.store(valid_ptr + output_offsets, selected_valid, mask=output_mask)


@triton.jit
def _distance_gradient_scale(
    coordinates_ptr,
    grad_distances_ptr,
    query,
    target,
    edge_slot,
    eps,
    edge_mask,
):

    pointer_broadcast = tl.zeros(edge_mask.shape, dtype=tl.int64)
    query_base = query * 3 + pointer_broadcast
    target_base = target * 3 + pointer_broadcast
    delta_x = tl.load(coordinates_ptr + query_base, mask=edge_mask, other=0.0).to(tl.float32) - tl.load(
        coordinates_ptr + target_base, mask=edge_mask, other=0.0
    ).to(tl.float32)
    delta_y = tl.load(coordinates_ptr + query_base + 1, mask=edge_mask, other=0.0).to(tl.float32) - tl.load(
        coordinates_ptr + target_base + 1, mask=edge_mask, other=0.0
    ).to(tl.float32)
    delta_z = tl.load(coordinates_ptr + query_base + 2, mask=edge_mask, other=0.0).to(tl.float32) - tl.load(
        coordinates_ptr + target_base + 2, mask=edge_mask, other=0.0
    ).to(tl.float32)
    raw_distance_sq = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
    finite_distance = (raw_distance_sq == raw_distance_sq) & (raw_distance_sq < _FLOAT32_MAX)
    distance_sq = tl.minimum(tl.where(finite_distance, raw_distance_sq, _FLOAT32_MAX), _FLOAT32_MAX)
    distance = tl.sqrt(distance_sq + eps)
    inverse_distance = tl.where(finite_distance & (distance > 0.0), 1.0 / distance, 0.0)
    scale = tl.load(grad_distances_ptr + edge_slot, mask=edge_mask, other=0.0).to(tl.float32)
    scale *= inverse_distance
    return delta_x * scale, delta_y * scale, delta_z * scale


@triton.jit
def _segmented_masked_knn_atomic_backward(
    grad_distances_ptr,
    coordinates_ptr,
    offsets_ptr,
    batch_indices_ptr,
    local_indices_ptr,
    valid_ptr,
    grad_coordinates_ptr,
    num_edges,
    eps,
    NUM_NEIGHBORS: tl.constexpr,
    BLOCK_EDGES: tl.constexpr,
):

    # Widen before multiplying: admitted edge indices may exceed int32.
    edge_slot = tl.program_id(0).to(tl.int64) * BLOCK_EDGES + tl.arange(0, BLOCK_EDGES)
    in_bounds = edge_slot < num_edges
    edge_valid = in_bounds & tl.load(valid_ptr + edge_slot, mask=in_bounds, other=False)
    query = edge_slot // NUM_NEIGHBORS
    sample = tl.load(batch_indices_ptr + query, mask=edge_valid, other=0).to(tl.int64)
    sample_start = tl.load(offsets_ptr + sample, mask=edge_valid, other=0).to(tl.int64)
    local_target = tl.load(local_indices_ptr + edge_slot, mask=edge_valid, other=0).to(tl.int64)
    target = sample_start + local_target
    grad_x, grad_y, grad_z = _distance_gradient_scale(
        coordinates_ptr,
        grad_distances_ptr,
        query,
        target,
        edge_slot,
        eps,
        edge_valid,
    )
    query_base = query * 3
    target_base = target * 3
    tl.atomic_add(grad_coordinates_ptr + query_base, grad_x, mask=edge_valid, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + query_base + 1, grad_y, mask=edge_valid, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + query_base + 2, grad_z, mask=edge_valid, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + target_base, -grad_x, mask=edge_valid, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + target_base + 1, -grad_y, mask=edge_valid, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + target_base + 2, -grad_z, mask=edge_valid, sem="relaxed")


@triton.jit
def _segmented_masked_knn_deterministic_backward(
    grad_distances_ptr,
    coordinates_ptr,
    offsets_ptr,
    batch_indices_ptr,
    local_indices_ptr,
    valid_ptr,
    target_order_ptr,
    target_boundaries_ptr,
    grad_coordinates_ptr,
    num_points,
    eps,
    NUM_NEIGHBORS: tl.constexpr,
    BLOCK_QUERY_EDGES: tl.constexpr,
    BLOCK_TARGET_EDGES: tl.constexpr,
):

    point = tl.program_id(0).to(tl.int64)
    point_valid = point < num_points

    query_lane = tl.arange(0, BLOCK_QUERY_EDGES)
    query_edge_valid = point_valid & (query_lane < NUM_NEIGHBORS)
    query_edge_slot = point * NUM_NEIGHBORS + query_lane
    query_edge_valid &= tl.load(valid_ptr + query_edge_slot, mask=query_edge_valid, other=False)
    sample = tl.load(batch_indices_ptr + point, mask=point_valid, other=0).to(tl.int64)
    sample_start = tl.load(offsets_ptr + sample, mask=point_valid, other=0).to(tl.int64)
    local_target = tl.load(local_indices_ptr + query_edge_slot, mask=query_edge_valid, other=0).to(tl.int64)
    query_target = sample_start + local_target
    query_x, query_y, query_z = _distance_gradient_scale(
        coordinates_ptr,
        grad_distances_ptr,
        point,
        query_target,
        query_edge_slot,
        eps,
        query_edge_valid,
    )
    grad_x = tl.sum(tl.where(query_edge_valid, query_x, 0.0), axis=0)
    grad_y = tl.sum(tl.where(query_edge_valid, query_y, 0.0), axis=0)
    grad_z = tl.sum(tl.where(query_edge_valid, query_z, 0.0), axis=0)

    target_begin = tl.load(target_boundaries_ptr + point, mask=point_valid, other=0).to(tl.int64)
    target_end = tl.load(target_boundaries_ptr + point + 1, mask=point_valid, other=0).to(tl.int64)
    target_lane = tl.arange(0, BLOCK_TARGET_EDGES)
    for target_start in tl.range(target_begin, target_end, BLOCK_TARGET_EDGES):
        target_position = target_start + target_lane
        target_edge_valid = point_valid & (target_position < target_end)
        target_edge_slot = tl.load(
            target_order_ptr + target_position,
            mask=target_edge_valid,
            other=0,
        ).to(tl.int64)
        target_query = target_edge_slot // NUM_NEIGHBORS
        target_x, target_y, target_z = _distance_gradient_scale(
            coordinates_ptr,
            grad_distances_ptr,
            target_query,
            point,
            target_edge_slot,
            eps,
            target_edge_valid,
        )
        grad_x -= tl.sum(tl.where(target_edge_valid, target_x, 0.0), axis=0)
        grad_y -= tl.sum(tl.where(target_edge_valid, target_y, 0.0), axis=0)
        grad_z -= tl.sum(tl.where(target_edge_valid, target_z, 0.0), axis=0)

    point_base = point * 3
    tl.store(grad_coordinates_ptr + point_base, grad_x, mask=point_valid)
    tl.store(grad_coordinates_ptr + point_base + 1, grad_y, mask=point_valid)
    tl.store(grad_coordinates_ptr + point_base + 2, grad_z, mask=point_valid)
