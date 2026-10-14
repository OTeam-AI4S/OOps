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


@triton.jit
def _bounded_atom(atom, atoms, wide_offsets: tl.constexpr):

    bounded = tl.minimum(tl.maximum(atom, 0), atoms - 1)
    if wide_offsets:
        bounded = bounded.to(tl.int64)
    return bounded


@triton.jit
def _smooth_lddt_score(distance_difference):
    r"""Return the score and distance-error derivative.

    s(e) = mean_t sigmoid(t - e); s'(e) = -mean_t sigma_t * (1 - sigma_t).
    """

    first = tl.sigmoid(0.5 - distance_difference)
    second = tl.sigmoid(1.0 - distance_difference)
    third = tl.sigmoid(2.0 - distance_difference)
    fourth = tl.sigmoid(4.0 - distance_difference)
    score = (first + second + third + fourth) * 0.25
    slope = -0.25 * (first * (1.0 - first) + second * (1.0 - second) + third * (1.0 - third) + fourth * (1.0 - fourth))
    return score, slope


@triton.jit
def _smooth_lddt_direction(delta, distance):
    r"""Return delta / distance, with zero derivative at a coincident pair."""

    return delta * tl.where(distance > 0.0, 1.0 / distance, 0.0)


@triton.jit
def _smooth_lddt_sign(value):

    return tl.where(value > 0.0, 1.0, tl.where(value < 0.0, -1.0, 0.0))


@triton.jit
def _smooth_lddt_pair_distances(
    coordinates_ptr,
    pair_indices_ptr,
    out_ptr,
    total_pairs,
    pairs,
    atoms,
    unbatched: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
) -> None:
    row_pid = tl.program_id(0)
    if wide_offsets:
        row_pid = row_pid.to(tl.int64)
    row_offsets = row_pid * block_pairs + tl.arange(0, block_pairs)
    row_mask = row_offsets < total_pairs
    batch = row_offsets // pairs
    pair = row_offsets - batch * pairs
    first_atom_index = _bounded_atom(
        tl.load(pair_indices_ptr + pair * 2, mask=row_mask, other=0),
        atoms,
        wide_offsets,
    )
    second_atom_index = _bounded_atom(
        tl.load(pair_indices_ptr + pair * 2 + 1, mask=row_mask, other=0),
        atoms,
        wide_offsets,
    )
    if unbatched:
        coordinate_base = 0
    else:
        coordinate_base = batch * atoms * 3
    first_atom_x = tl.load(coordinates_ptr + coordinate_base + first_atom_index * 3, mask=row_mask, other=0.0).to(
        tl.float32
    )
    first_atom_y = tl.load(coordinates_ptr + coordinate_base + first_atom_index * 3 + 1, mask=row_mask, other=0.0).to(
        tl.float32
    )
    first_atom_z = tl.load(coordinates_ptr + coordinate_base + first_atom_index * 3 + 2, mask=row_mask, other=0.0).to(
        tl.float32
    )
    second_atom_x = tl.load(coordinates_ptr + coordinate_base + second_atom_index * 3, mask=row_mask, other=0.0).to(
        tl.float32
    )
    second_atom_y = tl.load(coordinates_ptr + coordinate_base + second_atom_index * 3 + 1, mask=row_mask, other=0.0).to(
        tl.float32
    )
    second_atom_z = tl.load(coordinates_ptr + coordinate_base + second_atom_index * 3 + 2, mask=row_mask, other=0.0).to(
        tl.float32
    )
    dx = first_atom_x - second_atom_x
    dy = first_atom_y - second_atom_y
    dz = first_atom_z - second_atom_z
    tl.store(out_ptr + row_offsets, tl.sqrt(dx * dx + dy * dy + dz * dz), mask=row_mask)


@triton.jit
def _smooth_lddt_pairs_atomic_backward(
    coordinates_ptr,
    pair_indices_ptr,
    grad_out_ptr,
    true_distances_ptr,
    grad_coordinates_ptr,
    grad_true_ptr,
    total_pairs,
    pairs,
    atoms,
    unbatched: tl.constexpr,
    wide_offsets: tl.constexpr,
    loss_gradient: tl.constexpr,
    true_batch_stride: tl.constexpr,
    need_true: tl.constexpr,
    block_pairs: tl.constexpr,
):
    index_dtype = tl.int64 if wide_offsets else tl.int32
    offset = tl.program_id(0).to(index_dtype) * block_pairs + tl.arange(0, block_pairs)
    active = offset < total_pairs
    batch = offset // pairs
    pair = offset % pairs
    first = _bounded_atom(tl.load(pair_indices_ptr + pair * 2, mask=active, other=0), atoms, wide_offsets)
    second = _bounded_atom(tl.load(pair_indices_ptr + pair * 2 + 1, mask=active, other=0), atoms, wide_offsets)
    base = 0 if unbatched else batch * atoms * 3
    first_base = base + first * 3
    second_base = base + second * 3
    dx = tl.load(coordinates_ptr + first_base, mask=active, other=0).to(tl.float32) - tl.load(
        coordinates_ptr + second_base, mask=active, other=0
    ).to(tl.float32)
    dy = tl.load(coordinates_ptr + first_base + 1, mask=active, other=0).to(tl.float32) - tl.load(
        coordinates_ptr + second_base + 1, mask=active, other=0
    ).to(tl.float32)
    dz = tl.load(coordinates_ptr + first_base + 2, mask=active, other=0).to(tl.float32) - tl.load(
        coordinates_ptr + second_base + 2, mask=active, other=0
    ).to(tl.float32)
    distance = tl.sqrt(dx * dx + dy * dy + dz * dz)
    if loss_gradient:
        truth_offset = batch * true_batch_stride + pair
        truth = tl.load(true_distances_ptr + truth_offset, mask=active, other=0).to(tl.float32)
        error = distance - truth
        _, slope = _smooth_lddt_score(tl.abs(error))
        gradient = -tl.load(grad_out_ptr).to(tl.float32) * slope * _smooth_lddt_sign(error) / total_pairs
        if need_true:
            if true_batch_stride == 0:
                tl.atomic_add(grad_true_ptr + pair, -gradient, mask=active, sem="relaxed")
            else:
                tl.store(grad_true_ptr + truth_offset, -gradient, mask=active)
    else:
        gradient = tl.load(grad_out_ptr + offset, mask=active, other=0).to(tl.float32)
    gx = gradient * _smooth_lddt_direction(dx, distance)
    gy = gradient * _smooth_lddt_direction(dy, distance)
    gz = gradient * _smooth_lddt_direction(dz, distance)
    tl.atomic_add(grad_coordinates_ptr + first_base, gx, mask=active, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + first_base + 1, gy, mask=active, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + first_base + 2, gz, mask=active, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + second_base, -gx, mask=active, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + second_base + 1, -gy, mask=active, sem="relaxed")
    tl.atomic_add(grad_coordinates_ptr + second_base + 2, -gz, mask=active, sem="relaxed")


@triton.jit
def _smooth_lddt_pair_distances_backward_rows(
    coordinates_ptr,
    pair_indices_ptr,
    order_ptr,
    boundaries_ptr,
    grad_out_ptr,
    grad_coordinates_ptr,
    pairs,
    atoms,
    unbatched: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
    block_rows: tl.constexpr,
    row_base,
) -> None:
    row = row_base + tl.arange(0, block_rows)
    row_mask = row < atoms
    batch = tl.program_id(1)
    if wide_offsets:
        row = row.to(tl.int64)
        batch = batch.to(tl.int64)
    if unbatched:
        coordinate_base = 0
    else:
        coordinate_base = batch * atoms * 3
    # Each endpoint contributes (this - other) / distance to its coordinate VJP.
    row_x = tl.load(coordinates_ptr + coordinate_base + row * 3, mask=row_mask, other=0.0).to(tl.float32)
    row_y = tl.load(coordinates_ptr + coordinate_base + row * 3 + 1, mask=row_mask, other=0.0).to(tl.float32)
    row_z = tl.load(coordinates_ptr + coordinate_base + row * 3 + 2, mask=row_mask, other=0.0).to(tl.float32)
    start = tl.load(boundaries_ptr + row, mask=row_mask, other=0)
    stop = tl.load(boundaries_ptr + row + 1, mask=row_mask, other=0)
    if wide_offsets:
        start = start.to(tl.int64)
        stop = stop.to(tl.int64)

    grad_x = tl.zeros((block_rows, block_pairs), dtype=tl.float32)
    grad_y = tl.zeros((block_rows, block_pairs), dtype=tl.float32)
    grad_z = tl.zeros((block_rows, block_pairs), dtype=tl.float32)
    for begin in range(0, tl.max(stop - start, axis=0), block_pairs):
        slot = start[:, None] + begin + tl.arange(0, block_pairs)[None, :]
        slot_mask = row_mask[:, None] & (slot < stop[:, None])
        entry = tl.load(order_ptr + slot, mask=slot_mask, other=0)
        if wide_offsets:
            entry = entry.to(tl.int64)
        pair = entry // 2
        partner = _bounded_atom(
            tl.load(pair_indices_ptr + (entry ^ 1), mask=slot_mask, other=0),
            atoms,
            wide_offsets,
        )
        other_x = tl.load(coordinates_ptr + coordinate_base + partner * 3, mask=slot_mask, other=0.0).to(tl.float32)
        other_y = tl.load(coordinates_ptr + coordinate_base + partner * 3 + 1, mask=slot_mask, other=0.0).to(tl.float32)
        other_z = tl.load(coordinates_ptr + coordinate_base + partner * 3 + 2, mask=slot_mask, other=0.0).to(tl.float32)
        dx = row_x[:, None] - other_x
        dy = row_y[:, None] - other_y
        dz = row_z[:, None] - other_z
        distance = tl.sqrt(dx * dx + dy * dy + dz * dz)
        upstream = tl.load(grad_out_ptr + batch * pairs + pair, mask=slot_mask, other=0.0).to(tl.float32)
        grad_x += tl.where(slot_mask, upstream * _smooth_lddt_direction(dx, distance), 0.0)
        grad_y += tl.where(slot_mask, upstream * _smooth_lddt_direction(dy, distance), 0.0)
        grad_z += tl.where(slot_mask, upstream * _smooth_lddt_direction(dz, distance), 0.0)

    tl.store(grad_coordinates_ptr + coordinate_base + row * 3, tl.sum(grad_x, axis=1), mask=row_mask)
    tl.store(grad_coordinates_ptr + coordinate_base + row * 3 + 1, tl.sum(grad_y, axis=1), mask=row_mask)
    tl.store(grad_coordinates_ptr + coordinate_base + row * 3 + 2, tl.sum(grad_z, axis=1), mask=row_mask)


@triton.jit
def _smooth_lddt_pair_distances_backward(
    coordinates_ptr,
    pair_indices_ptr,
    order_ptr,
    boundaries_ptr,
    grad_out_ptr,
    grad_coordinates_ptr,
    pairs,
    atoms,
    unbatched: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    base = tl.program_id(0).to(tl.int64) * block_rows
    if block_rows == 1:
        _smooth_lddt_pair_distances_backward_rows(
            coordinates_ptr,
            pair_indices_ptr,
            order_ptr,
            boundaries_ptr,
            grad_out_ptr,
            grad_coordinates_ptr,
            pairs,
            atoms,
            unbatched,
            wide_offsets,
            block_pairs,
            1,
            base,
        )
    else:
        rows = base + tl.arange(0, block_rows)
        lo = tl.load(boundaries_ptr + rows, mask=rows < atoms, other=0)
        hi = tl.load(boundaries_ptr + rows + 1, mask=rows < atoms, other=0)
        if tl.max(hi - lo, axis=0) > block_pairs:
            for row in range(0, block_rows):
                _smooth_lddt_pair_distances_backward_rows(
                    coordinates_ptr,
                    pair_indices_ptr,
                    order_ptr,
                    boundaries_ptr,
                    grad_out_ptr,
                    grad_coordinates_ptr,
                    pairs,
                    atoms,
                    unbatched,
                    wide_offsets,
                    128,
                    1,
                    base + row,
                )
        else:
            _smooth_lddt_pair_distances_backward_rows(
                coordinates_ptr,
                pair_indices_ptr,
                order_ptr,
                boundaries_ptr,
                grad_out_ptr,
                grad_coordinates_ptr,
                pairs,
                atoms,
                unbatched,
                wide_offsets,
                block_pairs,
                block_rows,
                base,
            )


@triton.jit
def _smooth_lddt_dense_accumulate(
    predicted_ptr,
    true_ptr,
    is_nucleotide_ptr,
    coordinates_mask_ptr,
    accum_ptr,
    pairs_per_batch,
    atoms,
    nucleic_acid_cutoff: tl.constexpr,
    other_cutoff: tl.constexpr,
    block_pairs: tl.constexpr,
) -> None:
    pair_offsets = tl.program_id(0).to(tl.int64) * block_pairs + tl.arange(0, block_pairs)
    batch = tl.program_id(1).to(tl.int64)
    pair_mask = pair_offsets < pairs_per_batch
    row = pair_offsets // atoms
    column = pair_offsets - row * atoms
    coordinate_base = batch * atoms * 3

    predicted_row_x = tl.load(predicted_ptr + coordinate_base + row * 3, mask=pair_mask, other=0.0).to(tl.float32)
    predicted_row_y = tl.load(predicted_ptr + coordinate_base + row * 3 + 1, mask=pair_mask, other=0.0).to(tl.float32)
    predicted_row_z = tl.load(predicted_ptr + coordinate_base + row * 3 + 2, mask=pair_mask, other=0.0).to(tl.float32)
    predicted_column_x = tl.load(predicted_ptr + coordinate_base + column * 3, mask=pair_mask, other=0.0).to(tl.float32)
    predicted_column_y = tl.load(predicted_ptr + coordinate_base + column * 3 + 1, mask=pair_mask, other=0.0).to(
        tl.float32
    )
    predicted_column_z = tl.load(predicted_ptr + coordinate_base + column * 3 + 2, mask=pair_mask, other=0.0).to(
        tl.float32
    )

    true_row_x = tl.load(true_ptr + coordinate_base + row * 3, mask=pair_mask, other=0.0).to(tl.float32)
    true_row_y = tl.load(true_ptr + coordinate_base + row * 3 + 1, mask=pair_mask, other=0.0).to(tl.float32)
    true_row_z = tl.load(true_ptr + coordinate_base + row * 3 + 2, mask=pair_mask, other=0.0).to(tl.float32)
    true_column_x = tl.load(true_ptr + coordinate_base + column * 3, mask=pair_mask, other=0.0).to(tl.float32)
    true_column_y = tl.load(true_ptr + coordinate_base + column * 3 + 1, mask=pair_mask, other=0.0).to(tl.float32)
    true_column_z = tl.load(true_ptr + coordinate_base + column * 3 + 2, mask=pair_mask, other=0.0).to(tl.float32)

    predicted_delta_x = predicted_row_x - predicted_column_x
    predicted_delta_y = predicted_row_y - predicted_column_y
    predicted_delta_z = predicted_row_z - predicted_column_z
    true_dx = true_row_x - true_column_x
    true_dy = true_row_y - true_column_y
    true_dz = true_row_z - true_column_z
    predicted_distance = tl.sqrt(
        predicted_delta_x * predicted_delta_x
        + predicted_delta_y * predicted_delta_y
        + predicted_delta_z * predicted_delta_z
    )
    true_distance = tl.sqrt(true_dx * true_dx + true_dy * true_dy + true_dz * true_dz)

    mask_base = batch * atoms
    row_coordinates_weight = tl.load(coordinates_mask_ptr + mask_base + row, mask=pair_mask, other=0.0).to(tl.float32)
    column_coordinates_weight = tl.load(coordinates_mask_ptr + mask_base + column, mask=pair_mask, other=0.0).to(
        tl.float32
    )
    is_nucleotide = tl.load(is_nucleotide_ptr + mask_base + row, mask=pair_mask, other=0.0).to(tl.float32)
    cutoff = tl.where(is_nucleotide != 0.0, nucleic_acid_cutoff, other_cutoff)
    valid = (
        row_coordinates_weight
        * column_coordinates_weight
        * tl.where(row != column, 1.0, 0.0)
        * tl.where(true_distance < cutoff, 1.0, 0.0)
    )

    distance_difference = tl.abs(true_distance - predicted_distance)
    score = (
        tl.sigmoid(0.5 - distance_difference)
        + tl.sigmoid(1.0 - distance_difference)
        + tl.sigmoid(2.0 - distance_difference)
        + tl.sigmoid(4.0 - distance_difference)
    ) * 0.25
    numerator = tl.sum(tl.where(pair_mask, score * valid, 0.0), axis=0)
    denominator = tl.sum(tl.where(pair_mask, valid, 0.0), axis=0)
    tl.atomic_add(accum_ptr + batch * 2, numerator, sem="relaxed")
    tl.atomic_add(accum_ptr + batch * 2 + 1, denominator, sem="relaxed")


@triton.jit
def _smooth_lddt_dense_backward(
    predicted_ptr,
    true_ptr,
    is_nucleotide_ptr,
    coordinates_mask_ptr,
    accum_ptr,
    grad_out_ptr,
    grad_predicted_ptr,
    grad_true_ptr,
    grad_coordinates_mask_ptr,
    atoms,
    batch,
    nucleic_acid_cutoff: tl.constexpr,
    other_cutoff: tl.constexpr,
    need_grad_predicted: tl.constexpr,
    need_grad_true: tl.constexpr,
    need_grad_coordinates_mask: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    sample = tl.program_id(1).to(tl.int64)
    coordinate_base = sample * atoms * 3
    mask_base = sample * atoms

    # clamp_min(..., 1) propagates the denominator gradient at equality.
    numerator = tl.load(accum_ptr + sample * 2).to(tl.float32)
    denominator = tl.load(accum_ptr + sample * 2 + 1).to(tl.float32)
    clamped = tl.maximum(denominator, 1.0)
    upstream = tl.load(grad_out_ptr).to(tl.float32)
    grad_numerator = -upstream / (batch * clamped)
    grad_denominator = tl.where(denominator >= 1.0, upstream * numerator / (batch * clamped * clamped), 0.0)

    predicted_row_x = tl.load(predicted_ptr + coordinate_base + row * 3).to(tl.float32)
    predicted_row_y = tl.load(predicted_ptr + coordinate_base + row * 3 + 1).to(tl.float32)
    predicted_row_z = tl.load(predicted_ptr + coordinate_base + row * 3 + 2).to(tl.float32)
    true_row_x = tl.load(true_ptr + coordinate_base + row * 3).to(tl.float32)
    true_row_y = tl.load(true_ptr + coordinate_base + row * 3 + 1).to(tl.float32)
    true_row_z = tl.load(true_ptr + coordinate_base + row * 3 + 2).to(tl.float32)
    row_coordinates_mask = tl.load(coordinates_mask_ptr + mask_base + row).to(tl.float32)
    row_is_nucleotide = tl.load(is_nucleotide_ptr + mask_base + row).to(tl.float32)
    row_cutoff = tl.where(row_is_nucleotide != 0.0, nucleic_acid_cutoff, other_cutoff)

    if need_grad_predicted:
        grad_predicted_x = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_predicted_y = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_predicted_z = tl.zeros((block_atoms,), dtype=tl.float32)
    if need_grad_true:
        grad_true_x = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_true_y = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_true_z = tl.zeros((block_atoms,), dtype=tl.float32)
    if need_grad_coordinates_mask:
        grad_mask = tl.zeros((block_atoms,), dtype=tl.float32)

    for start in range(0, atoms, block_atoms):
        column = tl.arange(0, block_atoms).to(tl.int64) + start
        column_mask = column < atoms
        predicted_column_x = tl.load(predicted_ptr + coordinate_base + column * 3, mask=column_mask, other=0.0).to(
            tl.float32
        )
        predicted_column_y = tl.load(predicted_ptr + coordinate_base + column * 3 + 1, mask=column_mask, other=0.0).to(
            tl.float32
        )
        predicted_column_z = tl.load(predicted_ptr + coordinate_base + column * 3 + 2, mask=column_mask, other=0.0).to(
            tl.float32
        )
        true_column_x = tl.load(true_ptr + coordinate_base + column * 3, mask=column_mask, other=0.0).to(tl.float32)
        true_column_y = tl.load(true_ptr + coordinate_base + column * 3 + 1, mask=column_mask, other=0.0).to(tl.float32)
        true_column_z = tl.load(true_ptr + coordinate_base + column * 3 + 2, mask=column_mask, other=0.0).to(tl.float32)
        column_coordinates_mask = tl.load(coordinates_mask_ptr + mask_base + column, mask=column_mask, other=0.0).to(
            tl.float32
        )
        column_is_nucleotide = tl.load(is_nucleotide_ptr + mask_base + column, mask=column_mask, other=0.0).to(
            tl.float32
        )

        predicted_delta_x = predicted_row_x - predicted_column_x
        predicted_delta_y = predicted_row_y - predicted_column_y
        predicted_delta_z = predicted_row_z - predicted_column_z
        true_dx = true_row_x - true_column_x
        true_dy = true_row_y - true_column_y
        true_dz = true_row_z - true_column_z
        predicted_distance = tl.sqrt(
            predicted_delta_x * predicted_delta_x
            + predicted_delta_y * predicted_delta_y
            + predicted_delta_z * predicted_delta_z
        )
        true_distance = tl.sqrt(true_dx * true_dx + true_dy * true_dy + true_dz * true_dz)

        # Cutoffs use the row atom, so pair weights can be asymmetric.
        column_cutoff = tl.where(column_is_nucleotide != 0.0, nucleic_acid_cutoff, other_cutoff)
        off_diagonal = tl.where(column != row, 1.0, 0.0) * tl.where(column_mask, 1.0, 0.0)
        selects_forward = off_diagonal * tl.where(true_distance < row_cutoff, 1.0, 0.0)
        selects_reverse = off_diagonal * tl.where(true_distance < column_cutoff, 1.0, 0.0)
        pair_weight = row_coordinates_mask * column_coordinates_mask * (selects_forward + selects_reverse)

        signed_error = true_distance - predicted_distance
        score, slope = _smooth_lddt_score(tl.abs(signed_error))
        error_sign = _smooth_lddt_sign(signed_error)
        # d|true - predicted|/d predicted = -sign; d/d true = +sign.
        if need_grad_predicted:
            grad_predicted_distance = -(grad_numerator * pair_weight) * slope * error_sign
            grad_predicted_x += grad_predicted_distance * _smooth_lddt_direction(predicted_delta_x, predicted_distance)
            grad_predicted_y += grad_predicted_distance * _smooth_lddt_direction(predicted_delta_y, predicted_distance)
            grad_predicted_z += grad_predicted_distance * _smooth_lddt_direction(predicted_delta_z, predicted_distance)
        if need_grad_true:
            grad_true_distance = (grad_numerator * pair_weight) * slope * error_sign
            grad_true_x += grad_true_distance * _smooth_lddt_direction(true_dx, true_distance)
            grad_true_y += grad_true_distance * _smooth_lddt_direction(true_dy, true_distance)
            grad_true_z += grad_true_distance * _smooth_lddt_direction(true_dz, true_distance)
        # Mask VJP includes both numerator and denominator terms.
        if need_grad_coordinates_mask:
            grad_mask += (
                (grad_numerator * score + grad_denominator)
                * column_coordinates_mask
                * (selects_forward + selects_reverse)
            )

    if need_grad_predicted:
        tl.store(grad_predicted_ptr + coordinate_base + row * 3, tl.sum(grad_predicted_x, axis=0))
        tl.store(grad_predicted_ptr + coordinate_base + row * 3 + 1, tl.sum(grad_predicted_y, axis=0))
        tl.store(grad_predicted_ptr + coordinate_base + row * 3 + 2, tl.sum(grad_predicted_z, axis=0))
    if need_grad_true:
        tl.store(grad_true_ptr + coordinate_base + row * 3, tl.sum(grad_true_x, axis=0))
        tl.store(grad_true_ptr + coordinate_base + row * 3 + 1, tl.sum(grad_true_y, axis=0))
        tl.store(grad_true_ptr + coordinate_base + row * 3 + 2, tl.sum(grad_true_z, axis=0))
    if need_grad_coordinates_mask:
        tl.store(grad_coordinates_mask_ptr + mask_base + row, tl.sum(grad_mask, axis=0))


@triton.jit
def _smooth_lddt_mask_accumulate(
    predicted_ptr,
    true_ptr,
    mask_ptr,
    accum_ptr,
    total_pairs,
    pairs_per_batch,
    atoms,
    truth_batched: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
) -> None:
    pair_pid = tl.program_id(0)
    if wide_offsets:
        pair_pid = pair_pid.to(tl.int64)
    pair_offsets = pair_pid * block_pairs + tl.arange(0, block_pairs)
    row_mask = pair_offsets < total_pairs
    batch = pair_offsets // pairs_per_batch
    pair = pair_offsets - batch * pairs_per_batch
    row = pair // atoms
    column = pair - row * atoms
    mask_value = tl.load(mask_ptr + pair, mask=row_mask, other=0) != 0
    active = row_mask & mask_value
    predicted_base = batch * atoms * 3
    if truth_batched:
        true_base = batch * atoms * 3
    else:
        true_base = 0

    predicted_row_x = tl.load(predicted_ptr + predicted_base + row * 3, mask=active, other=0.0).to(tl.float32)
    predicted_row_y = tl.load(predicted_ptr + predicted_base + row * 3 + 1, mask=active, other=0.0).to(tl.float32)
    predicted_row_z = tl.load(predicted_ptr + predicted_base + row * 3 + 2, mask=active, other=0.0).to(tl.float32)
    predicted_column_x = tl.load(predicted_ptr + predicted_base + column * 3, mask=active, other=0.0).to(tl.float32)
    predicted_column_y = tl.load(predicted_ptr + predicted_base + column * 3 + 1, mask=active, other=0.0).to(tl.float32)
    predicted_column_z = tl.load(predicted_ptr + predicted_base + column * 3 + 2, mask=active, other=0.0).to(tl.float32)

    true_row_x = tl.load(true_ptr + true_base + row * 3, mask=active, other=0.0).to(tl.float32)
    true_row_y = tl.load(true_ptr + true_base + row * 3 + 1, mask=active, other=0.0).to(tl.float32)
    true_row_z = tl.load(true_ptr + true_base + row * 3 + 2, mask=active, other=0.0).to(tl.float32)
    true_column_x = tl.load(true_ptr + true_base + column * 3, mask=active, other=0.0).to(tl.float32)
    true_column_y = tl.load(true_ptr + true_base + column * 3 + 1, mask=active, other=0.0).to(tl.float32)
    true_column_z = tl.load(true_ptr + true_base + column * 3 + 2, mask=active, other=0.0).to(tl.float32)

    predicted_delta_x = predicted_row_x - predicted_column_x
    predicted_delta_y = predicted_row_y - predicted_column_y
    predicted_delta_z = predicted_row_z - predicted_column_z
    true_dx = true_row_x - true_column_x
    true_dy = true_row_y - true_column_y
    true_dz = true_row_z - true_column_z
    predicted_distance = tl.sqrt(
        predicted_delta_x * predicted_delta_x
        + predicted_delta_y * predicted_delta_y
        + predicted_delta_z * predicted_delta_z
    )
    true_distance = tl.sqrt(true_dx * true_dx + true_dy * true_dy + true_dz * true_dz)
    distance_difference = tl.abs(true_distance - predicted_distance)
    score = (
        tl.sigmoid(0.5 - distance_difference)
        + tl.sigmoid(1.0 - distance_difference)
        + tl.sigmoid(2.0 - distance_difference)
        + tl.sigmoid(4.0 - distance_difference)
    ) * 0.25
    numerator = tl.sum(tl.where(active, score, 0.0), axis=0)
    denominator = tl.sum(tl.where(active, 1.0, 0.0), axis=0)
    tl.atomic_add(accum_ptr, numerator, sem="relaxed")
    tl.atomic_add(accum_ptr + 1, denominator, sem="relaxed")


@triton.jit
def _smooth_lddt_mask_backward(
    predicted_ptr,
    true_ptr,
    mask_ptr,
    mask_transposed_ptr,
    accum_ptr,
    grad_out_ptr,
    grad_predicted_ptr,
    grad_true_ptr,
    atoms,
    truth_batched: tl.constexpr,
    need_grad_predicted: tl.constexpr,
    need_grad_true: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    sample = tl.program_id(1).to(tl.int64)
    predicted_base = sample * atoms * 3
    if truth_batched:
        true_base = sample * atoms * 3
    else:
        true_base = 0

    denominator = tl.maximum(tl.load(accum_ptr + 1).to(tl.float32), 1.0)
    grad_numerator = -tl.load(grad_out_ptr).to(tl.float32) / denominator

    predicted_row_x = tl.load(predicted_ptr + predicted_base + row * 3).to(tl.float32)
    predicted_row_y = tl.load(predicted_ptr + predicted_base + row * 3 + 1).to(tl.float32)
    predicted_row_z = tl.load(predicted_ptr + predicted_base + row * 3 + 2).to(tl.float32)
    true_row_x = tl.load(true_ptr + true_base + row * 3).to(tl.float32)
    true_row_y = tl.load(true_ptr + true_base + row * 3 + 1).to(tl.float32)
    true_row_z = tl.load(true_ptr + true_base + row * 3 + 2).to(tl.float32)

    if need_grad_predicted:
        grad_predicted_x = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_predicted_y = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_predicted_z = tl.zeros((block_atoms,), dtype=tl.float32)
    if need_grad_true:
        grad_true_x = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_true_y = tl.zeros((block_atoms,), dtype=tl.float32)
        grad_true_z = tl.zeros((block_atoms,), dtype=tl.float32)

    for start in range(0, atoms, block_atoms):
        column = tl.arange(0, block_atoms).to(tl.int64) + start
        column_mask = column < atoms
        forward_selected = tl.load(mask_ptr + row * atoms + column, mask=column_mask, other=0) != 0
        reverse_selected = tl.load(mask_transposed_ptr + row * atoms + column, mask=column_mask, other=0) != 0
        pair_weight = tl.where(forward_selected, 1.0, 0.0) + tl.where(reverse_selected, 1.0, 0.0)
        pair_weight = pair_weight * tl.where(column_mask, 1.0, 0.0)

        predicted_column_x = tl.load(predicted_ptr + predicted_base + column * 3, mask=column_mask, other=0.0).to(
            tl.float32
        )
        predicted_column_y = tl.load(predicted_ptr + predicted_base + column * 3 + 1, mask=column_mask, other=0.0).to(
            tl.float32
        )
        predicted_column_z = tl.load(predicted_ptr + predicted_base + column * 3 + 2, mask=column_mask, other=0.0).to(
            tl.float32
        )
        true_column_x = tl.load(true_ptr + true_base + column * 3, mask=column_mask, other=0.0).to(tl.float32)
        true_column_y = tl.load(true_ptr + true_base + column * 3 + 1, mask=column_mask, other=0.0).to(tl.float32)
        true_column_z = tl.load(true_ptr + true_base + column * 3 + 2, mask=column_mask, other=0.0).to(tl.float32)

        predicted_delta_x = predicted_row_x - predicted_column_x
        predicted_delta_y = predicted_row_y - predicted_column_y
        predicted_delta_z = predicted_row_z - predicted_column_z
        true_dx = true_row_x - true_column_x
        true_dy = true_row_y - true_column_y
        true_dz = true_row_z - true_column_z
        predicted_distance = tl.sqrt(
            predicted_delta_x * predicted_delta_x
            + predicted_delta_y * predicted_delta_y
            + predicted_delta_z * predicted_delta_z
        )
        true_distance = tl.sqrt(true_dx * true_dx + true_dy * true_dy + true_dz * true_dz)

        signed_error = true_distance - predicted_distance
        _score, slope = _smooth_lddt_score(tl.abs(signed_error))
        error_sign = _smooth_lddt_sign(signed_error)
        if need_grad_predicted:
            grad_predicted_distance = -(grad_numerator * pair_weight) * slope * error_sign
            grad_predicted_x += grad_predicted_distance * _smooth_lddt_direction(predicted_delta_x, predicted_distance)
            grad_predicted_y += grad_predicted_distance * _smooth_lddt_direction(predicted_delta_y, predicted_distance)
            grad_predicted_z += grad_predicted_distance * _smooth_lddt_direction(predicted_delta_z, predicted_distance)
        if need_grad_true:
            grad_true_distance = (grad_numerator * pair_weight) * slope * error_sign
            grad_true_x += grad_true_distance * _smooth_lddt_direction(true_dx, true_distance)
            grad_true_y += grad_true_distance * _smooth_lddt_direction(true_dy, true_distance)
            grad_true_z += grad_true_distance * _smooth_lddt_direction(true_dz, true_distance)

    if need_grad_predicted:
        tl.store(grad_predicted_ptr + predicted_base + row * 3, tl.sum(grad_predicted_x, axis=0))
        tl.store(grad_predicted_ptr + predicted_base + row * 3 + 1, tl.sum(grad_predicted_y, axis=0))
        tl.store(grad_predicted_ptr + predicted_base + row * 3 + 2, tl.sum(grad_predicted_z, axis=0))
    if need_grad_true:
        tl.store(grad_true_ptr + sample * atoms * 3 + row * 3, tl.sum(grad_true_x, axis=0))
        tl.store(
            grad_true_ptr + sample * atoms * 3 + row * 3 + 1,
            tl.sum(grad_true_y, axis=0),
        )
        tl.store(
            grad_true_ptr + sample * atoms * 3 + row * 3 + 2,
            tl.sum(grad_true_z, axis=0),
        )


@triton.jit
def _smooth_lddt_sparse_accumulate(
    predicted_ptr,
    pair_indices_ptr,
    true_distance_ptr,
    accum_ptr,
    total_pairs,
    pairs,
    atoms,
    true_batch_stride,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
) -> None:
    row_pid = tl.program_id(0)
    if wide_offsets:
        row_pid = row_pid.to(tl.int64)
    row_offsets = row_pid * block_pairs + tl.arange(0, block_pairs)
    row_mask = row_offsets < total_pairs
    batch = row_offsets // pairs
    pair = row_offsets - batch * pairs
    first_atom_index = _bounded_atom(
        tl.load(pair_indices_ptr + pair * 2, mask=row_mask, other=0),
        atoms,
        wide_offsets,
    )
    second_atom_index = _bounded_atom(
        tl.load(pair_indices_ptr + pair * 2 + 1, mask=row_mask, other=0),
        atoms,
        wide_offsets,
    )
    coordinate_base = batch * atoms * 3

    predicted_first_atom_x = tl.load(
        predicted_ptr + coordinate_base + first_atom_index * 3, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_first_atom_y = tl.load(
        predicted_ptr + coordinate_base + first_atom_index * 3 + 1, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_first_atom_z = tl.load(
        predicted_ptr + coordinate_base + first_atom_index * 3 + 2, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_second_atom_x = tl.load(
        predicted_ptr + coordinate_base + second_atom_index * 3, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_second_atom_y = tl.load(
        predicted_ptr + coordinate_base + second_atom_index * 3 + 1, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_second_atom_z = tl.load(
        predicted_ptr + coordinate_base + second_atom_index * 3 + 2, mask=row_mask, other=0.0
    ).to(tl.float32)

    dx = predicted_first_atom_x - predicted_second_atom_x
    dy = predicted_first_atom_y - predicted_second_atom_y
    dz = predicted_first_atom_z - predicted_second_atom_z
    predicted_distance = tl.sqrt(dx * dx + dy * dy + dz * dz)
    true_distance = tl.load(true_distance_ptr + batch * true_batch_stride + pair, mask=row_mask, other=0.0).to(
        tl.float32
    )
    distance_difference = tl.abs(predicted_distance - true_distance)
    score = (
        tl.sigmoid(0.5 - distance_difference)
        + tl.sigmoid(1.0 - distance_difference)
        + tl.sigmoid(2.0 - distance_difference)
        + tl.sigmoid(4.0 - distance_difference)
    ) * 0.25
    tl.atomic_add(accum_ptr, tl.sum(tl.where(row_mask, score, 0.0), axis=0), sem="relaxed")


@triton.jit
def _smooth_lddt_sparse_truth_backward(
    predicted_ptr,
    pair_indices_ptr,
    true_distance_ptr,
    grad_out_ptr,
    grad_true_distance_ptr,
    total_pairs,
    pairs,
    atoms,
    true_batch_stride,
    shared_truth: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
) -> None:

    pair_pid = tl.program_id(0)
    if wide_offsets:
        pair_pid = pair_pid.to(tl.int64)
    row_offsets = pair_pid * block_pairs + tl.arange(0, block_pairs)
    row_mask = row_offsets < total_pairs
    batch = row_offsets // pairs
    pair = row_offsets - batch * pairs
    first_atom_index = _bounded_atom(
        tl.load(pair_indices_ptr + pair * 2, mask=row_mask, other=0),
        atoms,
        wide_offsets,
    )
    second_atom_index = _bounded_atom(
        tl.load(pair_indices_ptr + pair * 2 + 1, mask=row_mask, other=0),
        atoms,
        wide_offsets,
    )
    coordinate_base = batch * atoms * 3

    predicted_first_atom_x = tl.load(
        predicted_ptr + coordinate_base + first_atom_index * 3, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_first_atom_y = tl.load(
        predicted_ptr + coordinate_base + first_atom_index * 3 + 1, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_first_atom_z = tl.load(
        predicted_ptr + coordinate_base + first_atom_index * 3 + 2, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_second_atom_x = tl.load(
        predicted_ptr + coordinate_base + second_atom_index * 3, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_second_atom_y = tl.load(
        predicted_ptr + coordinate_base + second_atom_index * 3 + 1, mask=row_mask, other=0.0
    ).to(tl.float32)
    predicted_second_atom_z = tl.load(
        predicted_ptr + coordinate_base + second_atom_index * 3 + 2, mask=row_mask, other=0.0
    ).to(tl.float32)
    dx = predicted_first_atom_x - predicted_second_atom_x
    dy = predicted_first_atom_y - predicted_second_atom_y
    dz = predicted_first_atom_z - predicted_second_atom_z
    predicted_distance = tl.sqrt(dx * dx + dy * dy + dz * dz)
    true_distance = tl.load(
        true_distance_ptr + batch * true_batch_stride + pair,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    signed_error = predicted_distance - true_distance
    _score, slope = _smooth_lddt_score(tl.abs(signed_error))
    upstream = -tl.load(grad_out_ptr).to(tl.float32) / total_pairs
    grad_predicted_distance = upstream * slope * _smooth_lddt_sign(signed_error)
    grad_true_distance = -grad_predicted_distance
    output_offsets = batch * true_batch_stride + pair
    if shared_truth:
        tl.atomic_add(
            grad_true_distance_ptr + output_offsets,
            grad_true_distance,
            mask=row_mask,
            sem="relaxed",
        )
    else:
        tl.store(grad_true_distance_ptr + output_offsets, grad_true_distance, mask=row_mask)


@triton.jit
def _smooth_lddt_sparse_backward_rows(
    predicted_ptr,
    pair_indices_ptr,
    order_ptr,
    boundaries_ptr,
    true_distance_ptr,
    grad_out_ptr,
    grad_predicted_ptr,
    grad_true_distance_ptr,
    total_pairs,
    atoms,
    true_batch_stride,
    write_grad_true: tl.constexpr,
    shared_truth: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
    block_rows: tl.constexpr,
    row_base,
) -> None:
    row = row_base + tl.arange(0, block_rows)
    row_mask = row < atoms
    batch = tl.program_id(1).to(tl.int64)
    coordinate_base = batch * atoms * 3
    row_x = tl.load(predicted_ptr + coordinate_base + row * 3, mask=row_mask, other=0.0).to(tl.float32)
    row_y = tl.load(predicted_ptr + coordinate_base + row * 3 + 1, mask=row_mask, other=0.0).to(tl.float32)
    row_z = tl.load(predicted_ptr + coordinate_base + row * 3 + 2, mask=row_mask, other=0.0).to(tl.float32)
    start = tl.load(boundaries_ptr + row, mask=row_mask, other=0)
    stop = tl.load(boundaries_ptr + row + 1, mask=row_mask, other=0)
    if wide_offsets:
        start = start.to(tl.int64)
        stop = stop.to(tl.int64)
    upstream = -tl.load(grad_out_ptr).to(tl.float32) / total_pairs

    grad_x = tl.zeros((block_rows, block_pairs), dtype=tl.float32)
    grad_y = tl.zeros((block_rows, block_pairs), dtype=tl.float32)
    grad_z = tl.zeros((block_rows, block_pairs), dtype=tl.float32)
    for begin in range(0, tl.max(stop - start, axis=0), block_pairs):
        slot = start[:, None] + begin + tl.arange(0, block_pairs)[None, :]
        slot_mask = row_mask[:, None] & (slot < stop[:, None])
        entry = tl.load(order_ptr + slot, mask=slot_mask, other=0)
        if wide_offsets:
            entry = entry.to(tl.int64)
        pair = entry // 2
        # Widen atom indices before stride multiplication.
        partner = _bounded_atom(
            tl.load(pair_indices_ptr + (entry ^ 1), mask=slot_mask, other=0),
            atoms,
            True,
        )
        other_x = tl.load(predicted_ptr + coordinate_base + partner * 3, mask=slot_mask, other=0.0).to(tl.float32)
        other_y = tl.load(predicted_ptr + coordinate_base + partner * 3 + 1, mask=slot_mask, other=0.0).to(tl.float32)
        other_z = tl.load(predicted_ptr + coordinate_base + partner * 3 + 2, mask=slot_mask, other=0.0).to(tl.float32)
        dx = row_x[:, None] - other_x
        dy = row_y[:, None] - other_y
        dz = row_z[:, None] - other_z
        predicted_distance = tl.sqrt(dx * dx + dy * dy + dz * dz)
        true_distance = tl.load(
            true_distance_ptr + batch * true_batch_stride + pair,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)

        # This error is predicted - true, so its coordinate derivative uses +sign.
        signed_error = predicted_distance - true_distance
        _score, slope = _smooth_lddt_score(tl.abs(signed_error))
        # Masked lanes contribute zero even when inactive coordinates are non-finite.
        grad_predicted_distance = tl.where(slot_mask, upstream * slope * _smooth_lddt_sign(signed_error), 0.0)
        grad_x += tl.where(slot_mask, grad_predicted_distance * _smooth_lddt_direction(dx, predicted_distance), 0.0)
        grad_y += tl.where(slot_mask, grad_predicted_distance * _smooth_lddt_direction(dy, predicted_distance), 0.0)
        grad_z += tl.where(slot_mask, grad_predicted_distance * _smooth_lddt_direction(dz, predicted_distance), 0.0)
        # Pair distances are written at the first endpoint; stride zero shares truth across samples.
        if write_grad_true:
            grad_true_mask = slot_mask & ((entry & 1) == 0)
            grad_true_offset = batch * true_batch_stride + pair
            if shared_truth:
                tl.atomic_add(
                    grad_true_distance_ptr + grad_true_offset,
                    -grad_predicted_distance,
                    mask=grad_true_mask,
                    sem="relaxed",
                )
            else:
                tl.store(
                    grad_true_distance_ptr + grad_true_offset,
                    -grad_predicted_distance,
                    mask=grad_true_mask,
                )

    tl.store(grad_predicted_ptr + coordinate_base + row * 3, tl.sum(grad_x, axis=1), mask=row_mask)
    tl.store(grad_predicted_ptr + coordinate_base + row * 3 + 1, tl.sum(grad_y, axis=1), mask=row_mask)
    tl.store(grad_predicted_ptr + coordinate_base + row * 3 + 2, tl.sum(grad_z, axis=1), mask=row_mask)


@triton.jit
def _smooth_lddt_sparse_backward(
    predicted_ptr,
    pair_indices_ptr,
    order_ptr,
    boundaries_ptr,
    true_distance_ptr,
    grad_out_ptr,
    grad_predicted_ptr,
    grad_true_distance_ptr,
    total_pairs,
    atoms,
    true_batch_stride,
    write_grad_true: tl.constexpr,
    shared_truth: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_pairs: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    base = tl.program_id(0).to(tl.int64) * block_rows
    if block_rows == 1:
        _smooth_lddt_sparse_backward_rows(
            predicted_ptr,
            pair_indices_ptr,
            order_ptr,
            boundaries_ptr,
            true_distance_ptr,
            grad_out_ptr,
            grad_predicted_ptr,
            grad_true_distance_ptr,
            total_pairs,
            atoms,
            true_batch_stride,
            write_grad_true,
            shared_truth,
            wide_offsets,
            block_pairs,
            1,
            base,
        )
    else:
        rows = base + tl.arange(0, block_rows)
        lo = tl.load(boundaries_ptr + rows, mask=rows < atoms, other=0)
        hi = tl.load(boundaries_ptr + rows + 1, mask=rows < atoms, other=0)
        if tl.max(hi - lo, axis=0) > block_pairs:
            for row in range(0, block_rows):
                _smooth_lddt_sparse_backward_rows(
                    predicted_ptr,
                    pair_indices_ptr,
                    order_ptr,
                    boundaries_ptr,
                    true_distance_ptr,
                    grad_out_ptr,
                    grad_predicted_ptr,
                    grad_true_distance_ptr,
                    total_pairs,
                    atoms,
                    true_batch_stride,
                    write_grad_true,
                    shared_truth,
                    wide_offsets,
                    128,
                    1,
                    base + row,
                )
        else:
            _smooth_lddt_sparse_backward_rows(
                predicted_ptr,
                pair_indices_ptr,
                order_ptr,
                boundaries_ptr,
                true_distance_ptr,
                grad_out_ptr,
                grad_predicted_ptr,
                grad_true_distance_ptr,
                total_pairs,
                atoms,
                true_batch_stride,
                write_grad_true,
                shared_truth,
                wide_offsets,
                block_pairs,
                block_rows,
                base,
            )


@triton.jit
def _smooth_lddt_dense_finalize_tiles(
    accum_ptr,
    partials_ptr,
    out_ptr,
    batch,
    single_partial: tl.constexpr,
    block_items: tl.constexpr,
) -> None:
    tile = tl.program_id(0).to(tl.int64)
    batch_offsets = tile * block_items + tl.arange(0, block_items)
    mask = batch_offsets < batch
    numerator = tl.load(accum_ptr + batch_offsets * 2, mask=mask, other=0.0).to(tl.float32)
    denominator = tl.maximum(
        tl.load(accum_ptr + batch_offsets * 2 + 1, mask=mask, other=1.0).to(tl.float32),
        1.0,
    )
    score = numerator / denominator
    total = tl.sum(tl.where(mask, score, 0.0), axis=0)
    if single_partial:
        tl.store(out_ptr, 1.0 - total / batch)
    else:
        tl.store(partials_ptr + tile, total)


@triton.jit
def _smooth_lddt_dense_finalize_partials(
    partials_ptr,
    out_ptr,
    batch,
    num_partials,
    block_partials: tl.constexpr,
) -> None:
    offsets = tl.arange(0, block_partials)
    partials = tl.load(partials_ptr + offsets, mask=offsets < num_partials, other=0.0).to(tl.float32)
    mean_score = tl.sum(partials, axis=0) / batch
    tl.store(out_ptr, 1.0 - mean_score)


@triton.jit
def _smooth_lddt_single_finalize(accum_ptr, out_ptr) -> None:
    numerator = tl.load(accum_ptr).to(tl.float32)
    denominator = tl.maximum(tl.load(accum_ptr + 1).to(tl.float32), 1.0)
    tl.store(out_ptr, 1.0 - numerator / denominator)


@triton.jit
def _smooth_lddt_sparse_finalize(accum_ptr, out_ptr, total_pairs) -> None:
    score = tl.load(accum_ptr).to(tl.float32) / total_pairs
    tl.store(out_ptr, 1.0 - score)
