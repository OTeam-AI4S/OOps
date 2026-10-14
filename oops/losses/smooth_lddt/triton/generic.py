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
from torch.utils.weak import WeakIdKeyDictionary

from oops.runtime import STANDARD_FLOAT_DTYPES, cuda_device_caps_if_needed, cuda_device_context, tensors_share_device

from ...determinism import require_nondeterminism_allowed
from . import kernel
from .profiles import (
    SMOOTH_LDDT_DENSE,
    SMOOTH_LDDT_MASK,
    SMOOTH_LDDT_PAIR_DISTANCE,
    SMOOTH_LDDT_SPARSE,
    SMOOTH_LDDT_SPARSE_COORDS,
    dense_target_changes_plan,
    mask_forward_target_changes_plan,
    resolve_dense_plan,
    resolve_mask_forward,
)
from .schedule import (
    SmoothLddtDensePlan,
    SmoothLddtDenseProblem,
    SmoothLddtMaskPlan,
    SmoothLddtMaskProblem,
    SmoothLddtPairDistancePlan,
    SmoothLddtPairDistanceProblem,
    SmoothLddtSparseCoordsProblem,
    SmoothLddtSparsePlan,
    SmoothLddtSparseProblem,
)

_SUPPORTED_FLOAT_DTYPES = STANDARD_FLOAT_DTYPES

_SUPPORTED_INDICATOR_DTYPES = (torch.bool, *_SUPPORTED_FLOAT_DTYPES)


def supports_smooth_lddt_loss_dense(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    is_nucleotide: Tensor,
    coordinates_mask: Tensor,
    true_distances: Tensor | None,
) -> bool:

    layout_supported = (
        true_distances is None
        and predicted_coordinates.is_cuda
        and true_coordinates.is_cuda
        and is_nucleotide.is_cuda
        and coordinates_mask.is_cuda
        and tensors_share_device(predicted_coordinates, true_coordinates, is_nucleotide, coordinates_mask)
        and predicted_coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and true_coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and is_nucleotide.dtype in _SUPPORTED_INDICATOR_DTYPES
        and coordinates_mask.dtype in _SUPPORTED_INDICATOR_DTYPES
        and predicted_coordinates.dim() == 3
        and true_coordinates.shape == predicted_coordinates.shape
        and predicted_coordinates.shape[-1] == 3
        and is_nucleotide.shape == predicted_coordinates.shape[:2]
        and coordinates_mask.shape == predicted_coordinates.shape[:2]
        and predicted_coordinates.is_contiguous()
        and true_coordinates.is_contiguous()
        and is_nucleotide.is_contiguous()
        and coordinates_mask.is_contiguous()
        and predicted_coordinates.shape[0] > 0
        and predicted_coordinates.shape[1] > 0
    )
    return layout_supported and SMOOTH_LDDT_DENSE.supports(
        SmoothLddtDenseProblem(predicted_coordinates.shape[0], predicted_coordinates.shape[1])
    )


def supports_smooth_lddt_loss_sparse(
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
) -> bool:

    layout_supported = (
        predicted_coordinates.is_cuda
        and pair_indices.is_cuda
        and true_pair_distances.is_cuda
        and tensors_share_device(predicted_coordinates, pair_indices, true_pair_distances)
        and predicted_coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and true_pair_distances.dtype in _SUPPORTED_FLOAT_DTYPES
        and pair_indices.dtype in (torch.int32, torch.int64)
        and predicted_coordinates.dim() == 3
        and predicted_coordinates.shape[-1] == 3
        and pair_indices.dim() == 1
        and pair_indices.numel() % 2 == 0
        and true_pair_distances.shape
        in (
            (pair_indices.numel() // 2,),
            (predicted_coordinates.shape[0], pair_indices.numel() // 2),
        )
        and predicted_coordinates.is_contiguous()
        and pair_indices.is_contiguous()
        and true_pair_distances.is_contiguous()
        and predicted_coordinates.shape[0] > 0
        and predicted_coordinates.shape[1] > 0
    )
    return layout_supported and SMOOTH_LDDT_SPARSE.supports(
        SmoothLddtSparseProblem(
            predicted_coordinates.shape[0],
            predicted_coordinates.shape[1],
            pair_indices.numel() // 2,
            true_pair_distances.dim() == 2,
        )
    )


def supports_smooth_lddt_loss_sparse_coordinates(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    pair_indices: Tensor,
) -> bool:

    layout_supported = (
        _pair_distance_layout_supported(true_coordinates, pair_indices)
        and predicted_coordinates.is_cuda
        and tensors_share_device(predicted_coordinates, true_coordinates, pair_indices)
        and predicted_coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and true_coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and predicted_coordinates.dim() == 3
        and predicted_coordinates.shape[-1] == 3
        and predicted_coordinates.is_contiguous()
        and predicted_coordinates.shape[0] > 0
        and predicted_coordinates.shape[1] > 0
        and true_coordinates.shape[-2] == predicted_coordinates.shape[1]
        and (true_coordinates.dim() == 2 or true_coordinates.shape[0] == predicted_coordinates.shape[0])
    )
    return layout_supported and SMOOTH_LDDT_SPARSE_COORDS.supports(
        SmoothLddtSparseCoordsProblem(
            predicted_coordinates.shape[0],
            predicted_coordinates.shape[1],
            pair_indices.numel() // 2,
            true_coordinates.dim() == 3,
        )
    )


def supports_smooth_lddt_loss_mask(predicted_coordinates: Tensor, true_coordinates: Tensor, lddt_mask: Tensor) -> bool:

    layout_supported = (
        predicted_coordinates.is_cuda
        and true_coordinates.is_cuda
        and lddt_mask.is_cuda
        and tensors_share_device(predicted_coordinates, true_coordinates, lddt_mask)
        and predicted_coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and true_coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and lddt_mask.dtype == torch.bool
        and predicted_coordinates.dim() == 3
        and true_coordinates.dim() in (2, 3)
        and predicted_coordinates.shape[-1] == 3
        and true_coordinates.shape[-1] == 3
        and true_coordinates.shape[-2] == predicted_coordinates.shape[1]
        and lddt_mask.shape == predicted_coordinates.shape[1:2] * 2
        and (true_coordinates.dim() == 2 or true_coordinates.shape[0] == predicted_coordinates.shape[0])
        and predicted_coordinates.is_contiguous()
        and true_coordinates.is_contiguous()
        and lddt_mask.is_contiguous()
        and predicted_coordinates.shape[0] > 0
        and predicted_coordinates.shape[1] > 0
    )
    return layout_supported and SMOOTH_LDDT_MASK.supports(
        SmoothLddtMaskProblem(
            predicted_coordinates.shape[0],
            predicted_coordinates.shape[1],
            true_coordinates.dim() == 3,
        )
    )


def supports_smooth_lddt_pair_distances(coordinates: Tensor, pair_indices: Tensor) -> bool:

    layout_supported = _pair_distance_layout_supported(coordinates, pair_indices)
    if not layout_supported:
        return False
    batch = 1 if coordinates.dim() == 2 else coordinates.shape[0]
    return SMOOTH_LDDT_PAIR_DISTANCE.supports(
        SmoothLddtPairDistanceProblem(
            batch,
            coordinates.shape[-2],
            pair_indices.numel() // 2,
            coordinates.dim() == 2,
        )
    )


def _pair_distance_layout_supported(coordinates: Tensor, pair_indices: Tensor) -> bool:
    return (
        coordinates.is_cuda
        and pair_indices.is_cuda
        and tensors_share_device(coordinates, pair_indices)
        and coordinates.dtype in _SUPPORTED_FLOAT_DTYPES
        and pair_indices.dtype in (torch.int32, torch.int64)
        and coordinates.dim() in (2, 3)
        and coordinates.shape[-1] == 3
        and pair_indices.dim() == 1
        and pair_indices.numel() % 2 == 0
        and coordinates.is_contiguous()
        and pair_indices.is_contiguous()
        and coordinates.shape[-2] > 0
        and (coordinates.dim() == 2 or coordinates.shape[0] > 0)
    )


def _run_smooth_lddt_loss_dense(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    is_nucleotide: Tensor,
    coordinates_mask: Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
) -> tuple[Tensor, Tensor]:
    with cuda_device_context(predicted_coordinates.device):
        predicted_coordinates = predicted_coordinates.resolve_neg()
        true_coordinates = true_coordinates.resolve_neg()
        is_nucleotide = is_nucleotide.resolve_neg()
        coordinates_mask = coordinates_mask.resolve_neg()
        problem = SmoothLddtDenseProblem(
            predicted_coordinates.shape[0], predicted_coordinates.shape[1], predicted_coordinates.dtype
        )
        target = cuda_device_caps_if_needed(predicted_coordinates.device, dense_target_changes_plan(problem))
        plan = resolve_dense_plan(problem, target=target)
        if plan.accumulate_grid[0] > 1:
            require_nondeterminism_allowed("smooth_lddt_loss_dense")
        batch = int(plan.problem.batch)
        atoms = int(plan.problem.atoms)
        accum = _zero_accumulator((batch, 2), predicted_coordinates.device)
        out = torch.empty((), device=predicted_coordinates.device, dtype=torch.float32)
        kernel._smooth_lddt_dense_accumulate[plan.accumulate_grid](
            predicted_coordinates,
            true_coordinates,
            is_nucleotide,
            coordinates_mask,
            accum,
            plan.pairs_per_batch,
            atoms,
            float(nucleic_acid_cutoff),
            float(other_cutoff),
            plan.config.block_pairs,
            num_warps=plan.config.num_warps,
        )
        single_partial = plan.finalize_plan.num_partials == 1
        partials = (
            out
            if single_partial
            else torch.empty(
                (plan.finalize_plan.num_partials,),
                device=predicted_coordinates.device,
                dtype=torch.float32,
            )
        )
        kernel._smooth_lddt_dense_finalize_tiles[plan.finalize_plan.tile_grid](
            accum,
            partials,
            out,
            batch,
            single_partial,
            plan.finalize_plan.block_items,
            num_warps=plan.finalize_tile_num_warps,
        )
        if not single_partial:
            kernel._smooth_lddt_dense_finalize_partials[plan.finalize_plan.finalize_grid](
                partials,
                out,
                batch,
                plan.finalize_plan.num_partials,
                plan.finalize_plan.block_partials,
                num_warps=plan.finalize_partials_num_warps,
            )
        return out, accum


def _run_smooth_lddt_loss_dense_backward(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    is_nucleotide: Tensor,
    coordinates_mask: Tensor,
    accum: Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
    need_grad_predicted: bool,
    need_grad_true: bool,
    need_grad_coordinates_mask: bool,
    plan: SmoothLddtDensePlan | None = None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    with cuda_device_context(predicted_coordinates.device):
        predicted_coordinates = predicted_coordinates.resolve_neg().contiguous()
        true_coordinates = true_coordinates.resolve_neg().contiguous()
        is_nucleotide = is_nucleotide.resolve_neg().contiguous()
        coordinates_mask = coordinates_mask.resolve_neg().contiguous()
        accum = accum.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        if not need_grad_predicted and not need_grad_true and not need_grad_coordinates_mask:
            return None, None, None
        if plan is None:
            problem = SmoothLddtDenseProblem(
                predicted_coordinates.shape[0], predicted_coordinates.shape[1], predicted_coordinates.dtype
            )
            target = cuda_device_caps_if_needed(predicted_coordinates.device, dense_target_changes_plan(problem))
            plan = resolve_dense_plan(problem, target=target)
        batch = int(plan.problem.batch)
        atoms = int(plan.problem.atoms)
        grad_predicted = torch.empty_like(predicted_coordinates) if need_grad_predicted else None
        grad_true = torch.empty_like(true_coordinates) if need_grad_true else None
        grad_coordinates_mask = torch.empty_like(coordinates_mask) if need_grad_coordinates_mask else None
        kernel._smooth_lddt_dense_backward[plan.backward_grid](
            predicted_coordinates,
            true_coordinates,
            is_nucleotide,
            coordinates_mask,
            accum,
            grad_out,
            predicted_coordinates if grad_predicted is None else grad_predicted,
            true_coordinates if grad_true is None else grad_true,
            coordinates_mask if grad_coordinates_mask is None else grad_coordinates_mask,
            atoms,
            batch,
            float(nucleic_acid_cutoff),
            float(other_cutoff),
            need_grad_predicted,
            need_grad_true,
            need_grad_coordinates_mask,
            plan.config.row_block_atoms,
            num_warps=plan.config.num_warps,
        )
        return grad_predicted, grad_true, grad_coordinates_mask


def _run_smooth_lddt_loss_mask_forward(
    predicted_coordinates: Tensor, true_coordinates: Tensor, lddt_mask: Tensor
) -> tuple[Tensor, Tensor]:
    with cuda_device_context(predicted_coordinates.device):
        predicted_coordinates = predicted_coordinates.resolve_neg()
        true_coordinates = true_coordinates.resolve_neg()
        lddt_mask = lddt_mask.resolve_neg()
        problem = SmoothLddtMaskProblem(
            predicted_coordinates.shape[0],
            predicted_coordinates.shape[1],
            true_coordinates.dim() == 3,
            predicted_coordinates.dtype,
        )
        target = cuda_device_caps_if_needed(predicted_coordinates.device, mask_forward_target_changes_plan(problem))
        plan = resolve_mask_forward(problem, target=target)
        if plan.forward_grid[0] > 1:
            require_nondeterminism_allowed("smooth_lddt_loss_sparse")
        atoms = int(plan.problem.atoms)
        accum = _zero_accumulator((1, 2), predicted_coordinates.device)
        out = torch.empty((), device=predicted_coordinates.device, dtype=torch.float32)
        kernel._smooth_lddt_mask_accumulate[plan.forward_grid](
            predicted_coordinates,
            true_coordinates,
            lddt_mask,
            accum,
            plan.total_pairs,
            plan.pairs_per_batch,
            atoms,
            plan.problem.truth_batched,
            plan.forward_wide_offsets,
            plan.config.block_pairs,
            num_warps=plan.config.num_warps,
        )
        kernel._smooth_lddt_single_finalize[plan.finalize_grid](accum, out)
        return out, accum


def _run_smooth_lddt_loss_mask_backward(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    lddt_mask: Tensor,
    accum: Tensor,
    need_grad_predicted: bool,
    need_grad_true: bool,
    plan: SmoothLddtMaskPlan | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    with cuda_device_context(predicted_coordinates.device):
        predicted_coordinates = predicted_coordinates.resolve_neg().contiguous()
        true_coordinates = true_coordinates.resolve_neg().contiguous()
        lddt_mask = lddt_mask.resolve_neg().contiguous()
        accum = accum.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        if not need_grad_predicted and not need_grad_true:
            return None, None
        if plan is None:
            plan = SMOOTH_LDDT_MASK.resolve(
                SmoothLddtMaskProblem(
                    predicted_coordinates.shape[0],
                    predicted_coordinates.shape[1],
                    true_coordinates.dim() == 3,
                )
            )
        samples = int(plan.problem.batch)
        atoms = int(plan.problem.atoms)
        grad_predicted = torch.empty_like(predicted_coordinates) if need_grad_predicted else None
        grad_true_rows = None
        if need_grad_true:
            grad_true_rows = torch.empty(
                (samples, atoms, 3),
                device=true_coordinates.device,
                dtype=true_coordinates.dtype if plan.problem.truth_batched else torch.float32,
            )
        kernel._smooth_lddt_mask_backward[plan.backward_grid](
            predicted_coordinates,
            true_coordinates,
            lddt_mask,
            lddt_mask.t().contiguous(),
            accum,
            grad_out,
            predicted_coordinates if grad_predicted is None else grad_predicted,
            true_coordinates if grad_true_rows is None else grad_true_rows,
            atoms,
            plan.problem.truth_batched,
            need_grad_predicted,
            need_grad_true,
            plan.config.row_block_atoms,
            num_warps=plan.config.num_warps,
        )
        grad_true = None
        if grad_true_rows is not None:
            grad_true = (
                grad_true_rows if plan.problem.truth_batched else grad_true_rows.sum(0).to(dtype=true_coordinates.dtype)
            )
        return grad_predicted, grad_true


def _run_smooth_lddt_loss_sparse_backward(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    true_coordinates: Tensor | None,
    need_predicted: bool,
    need_truth: bool,
    sparse_plan: SmoothLddtSparsePlan | None = None,
    pair_plan: SmoothLddtPairDistancePlan | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    if not need_predicted and not need_truth:
        return None, None
    if sparse_plan is None and true_coordinates is None:
        sparse_plan = SMOOTH_LDDT_SPARSE.resolve(
            _sparse_loss_problem(predicted_coordinates, pair_indices, true_pair_distances)
        )
    elif sparse_plan is None:
        assert true_coordinates is not None
        combined = SMOOTH_LDDT_SPARSE_COORDS.resolve(
            SmoothLddtSparseCoordsProblem(
                predicted_coordinates.shape[0],
                predicted_coordinates.shape[1],
                pair_indices.numel() // 2,
                true_coordinates.dim() == 3,
            )
        )
        sparse_plan = combined.sparse_loss_plan
        pair_plan = combined.pair_distance_plan
    assert sparse_plan is not None
    if not need_truth:
        return (
            _run_sparse_loss_backward_predicted(
                grad_out,
                predicted_coordinates,
                pair_indices,
                true_pair_distances,
                sparse_plan,
            ),
            None,
        )
    if true_pair_distances.dim() == 1 and predicted_coordinates.shape[0] > 1:
        require_nondeterminism_allowed("smooth_lddt_loss_sparse backward")
    inversion = None
    if (
        torch.are_deterministic_algorithms_enabled()
        and need_predicted
        and true_coordinates is not None
        and pair_plan is not None
        and sparse_plan.inversion_dtype == pair_plan.inversion_dtype
    ):
        inversion = _pair_list_inversion(pair_indices, int(sparse_plan.problem.atoms), sparse_plan.inversion_dtype)
    if need_predicted:
        grad_predicted, grad_true_pair_distances = _run_sparse_loss_backward(
            grad_out,
            predicted_coordinates,
            pair_indices,
            true_pair_distances,
            sparse_plan,
            inversion,
        )
    else:
        grad_predicted = None
        grad_true_pair_distances = _run_sparse_loss_backward_truth(
            grad_out,
            predicted_coordinates,
            pair_indices,
            true_pair_distances,
            sparse_plan,
        )
    if true_coordinates is None:
        return grad_predicted, grad_true_pair_distances
    assert pair_plan is not None
    return (
        grad_predicted,
        _run_pair_distances_backward(
            grad_true_pair_distances,
            true_coordinates,
            pair_indices,
            pair_plan,
            inversion,
        ),
    )


def _run_smooth_lddt_loss_sparse_coordinates_forward(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    pair_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    plan = SMOOTH_LDDT_SPARSE_COORDS.resolve(
        SmoothLddtSparseCoordsProblem(
            predicted_coordinates.shape[0],
            predicted_coordinates.shape[1],
            pair_indices.numel() // 2,
            true_coordinates.dim() == 3,
        )
    )
    true_pair_distances = _run_pair_distances(true_coordinates, pair_indices, plan.pair_distance_plan)
    return (
        _run_sparse_loss(predicted_coordinates, pair_indices, true_pair_distances, plan.sparse_loss_plan),
        true_pair_distances,
    )


def _zero_accumulator(shape: tuple[int, ...], device: torch.device) -> Tensor:

    return torch.zeros(shape, device=device, dtype=torch.float32)


def _pair_distance_problem(coordinates: Tensor, pair_indices: Tensor) -> SmoothLddtPairDistanceProblem:
    return SmoothLddtPairDistanceProblem(
        1 if coordinates.dim() == 2 else coordinates.shape[0],
        coordinates.shape[-2],
        pair_indices.numel() // 2,
        coordinates.dim() == 2,
    )


def _run_pair_distances(
    coordinates: Tensor,
    pair_indices: Tensor,
    plan: SmoothLddtPairDistancePlan | None = None,
) -> Tensor:
    with cuda_device_context(coordinates.device):
        coordinates = coordinates.resolve_neg()
        pair_indices = pair_indices.resolve_neg()
        plan = (
            SMOOTH_LDDT_PAIR_DISTANCE.resolve(_pair_distance_problem(coordinates, pair_indices))
            if plan is None
            else plan
        )
        if plan.forward_grid is None:
            raise ValueError("pair-distance launcher requires a non-empty plan")
        atoms = int(plan.problem.atoms)
        pairs = int(plan.problem.pairs)
        out = torch.empty(plan.output_shape, device=coordinates.device, dtype=torch.float32)
        kernel._smooth_lddt_pair_distances[plan.forward_grid](
            coordinates,
            pair_indices,
            out,
            plan.total_pairs,
            pairs,
            atoms,
            plan.problem.unbatched,
            plan.forward_wide_offsets,
            plan.config.forward_block_pairs,
            num_warps=plan.config.num_warps,
        )
        return out


def _run_pair_distances_backward(
    grad_out: Tensor,
    coordinates: Tensor,
    pair_indices: Tensor,
    plan: SmoothLddtPairDistancePlan | None = None,
    inversion: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    with cuda_device_context(coordinates.device):
        coordinates = coordinates.resolve_neg().contiguous()
        pair_indices = pair_indices.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        plan = (
            SMOOTH_LDDT_PAIR_DISTANCE.resolve(_pair_distance_problem(coordinates, pair_indices))
            if plan is None
            else plan
        )
        if plan.backward_grid is None:
            raise ValueError("pair-distance backward launcher requires a non-empty plan")
        if not torch.are_deterministic_algorithms_enabled():
            grad_coordinates = torch.zeros_like(coordinates, dtype=torch.float32)
            kernel._smooth_lddt_pairs_atomic_backward[plan.forward_grid](
                coordinates,
                pair_indices,
                grad_out,
                coordinates,
                grad_coordinates,
                grad_coordinates,
                plan.total_pairs,
                int(plan.problem.pairs),
                int(plan.problem.atoms),
                plan.problem.unbatched,
                plan.forward_wide_offsets,
                False,
                0,
                False,
                plan.config.forward_block_pairs,
                num_warps=plan.config.num_warps,
            )
            return grad_coordinates.to(coordinates.dtype)
        atoms = int(plan.problem.atoms)
        pairs = int(plan.problem.pairs)
        order, boundaries = (
            _pair_list_inversion(pair_indices, atoms, plan.inversion_dtype) if inversion is None else inversion
        )
        grad_coordinates = torch.empty_like(coordinates)
        kernel._smooth_lddt_pair_distances_backward[plan.backward_grid](
            coordinates,
            pair_indices,
            order,
            boundaries,
            grad_out,
            grad_coordinates,
            pairs,
            atoms,
            plan.problem.unbatched,
            plan.backward_wide_offsets,
            32 if plan.backward_block_rows > 1 else plan.config.backward_block_pairs,
            plan.backward_block_rows,
            num_warps=plan.config.num_warps,
        )
        return grad_coordinates


def _sparse_loss_problem(
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
) -> SmoothLddtSparseProblem:
    return SmoothLddtSparseProblem(
        predicted_coordinates.shape[0],
        predicted_coordinates.shape[1],
        pair_indices.numel() // 2,
        true_pair_distances.dim() == 2,
    )


def _run_sparse_loss(
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    plan: SmoothLddtSparsePlan | None = None,
) -> Tensor:
    with cuda_device_context(predicted_coordinates.device):
        predicted_coordinates = predicted_coordinates.resolve_neg()
        pair_indices = pair_indices.resolve_neg()
        true_pair_distances = true_pair_distances.resolve_neg()
        plan = (
            SMOOTH_LDDT_SPARSE.resolve(_sparse_loss_problem(predicted_coordinates, pair_indices, true_pair_distances))
            if plan is None
            else plan
        )
        if plan.forward_grid is None:
            raise ValueError("sparse smooth-LDDT launcher requires a non-empty plan")
        if plan.forward_grid[0] > 1:
            require_nondeterminism_allowed("smooth_lddt_loss_sparse")
        pairs = int(plan.problem.pairs)
        accum = _zero_accumulator((1,), predicted_coordinates.device)
        out = torch.empty((), device=predicted_coordinates.device, dtype=torch.float32)
        kernel._smooth_lddt_sparse_accumulate[plan.forward_grid](
            predicted_coordinates,
            pair_indices,
            true_pair_distances,
            accum,
            plan.total_pairs,
            pairs,
            plan.problem.atoms,
            plan.true_batch_stride,
            plan.forward_wide_offsets,
            plan.config.forward_block_pairs,
            num_warps=plan.config.num_warps,
        )
        if plan.finalize_grid is None:
            raise ValueError("sparse smooth-LDDT finalizer requires a non-empty plan")
        kernel._smooth_lddt_sparse_finalize[plan.finalize_grid](accum, out, plan.total_pairs)
        return out


def _run_sparse_loss_backward(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    plan: SmoothLddtSparsePlan | None = None,
    inversion: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    # Shared truth-distance gradients accumulate contributions from each sample.
    shared_truth = true_pair_distances.dim() == 1
    grad_true_accum = (
        torch.zeros_like(true_pair_distances, dtype=torch.float32)
        if shared_truth
        else torch.empty_like(true_pair_distances)
    )
    grad_predicted = _launch_sparse_loss_backward(
        grad_out,
        predicted_coordinates,
        pair_indices,
        true_pair_distances,
        grad_true_accum,
        plan,
        inversion,
    )
    return grad_predicted, grad_true_accum.to(dtype=true_pair_distances.dtype)


def _run_sparse_loss_backward_predicted(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    plan: SmoothLddtSparsePlan | None = None,
) -> Tensor:
    return _launch_sparse_loss_backward(grad_out, predicted_coordinates, pair_indices, true_pair_distances, None, plan)


def _run_sparse_loss_backward_truth(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    plan: SmoothLddtSparsePlan | None = None,
) -> Tensor:

    with cuda_device_context(predicted_coordinates.device):
        predicted_coordinates = predicted_coordinates.resolve_neg().contiguous()
        pair_indices = pair_indices.resolve_neg().contiguous()
        true_pair_distances = true_pair_distances.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        plan = (
            SMOOTH_LDDT_SPARSE.resolve(_sparse_loss_problem(predicted_coordinates, pair_indices, true_pair_distances))
            if plan is None
            else plan
        )
        if plan.forward_grid is None:
            raise ValueError("sparse smooth-LDDT truth backward requires a non-empty plan")
        shared_truth = true_pair_distances.dim() == 1
        grad_true = (
            torch.zeros_like(true_pair_distances, dtype=torch.float32)
            if shared_truth
            else torch.empty_like(true_pair_distances)
        )
        kernel._smooth_lddt_sparse_truth_backward[plan.forward_grid](
            predicted_coordinates,
            pair_indices,
            true_pair_distances,
            grad_out,
            grad_true,
            plan.total_pairs,
            int(plan.problem.pairs),
            int(plan.problem.atoms),
            plan.true_batch_stride,
            shared_truth,
            plan.forward_wide_offsets,
            plan.config.forward_block_pairs,
            num_warps=plan.config.num_warps,
        )
        return grad_true.to(dtype=true_pair_distances.dtype)


def _launch_sparse_loss_backward(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    grad_true_pair_distances: Tensor | None,
    plan: SmoothLddtSparsePlan | None,
    inversion: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    with cuda_device_context(predicted_coordinates.device):
        predicted_coordinates = predicted_coordinates.resolve_neg().contiguous()
        pair_indices = pair_indices.resolve_neg().contiguous()
        true_pair_distances = true_pair_distances.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        plan = (
            SMOOTH_LDDT_SPARSE.resolve(_sparse_loss_problem(predicted_coordinates, pair_indices, true_pair_distances))
            if plan is None
            else plan
        )
        if plan.backward_grid is None:
            raise ValueError("sparse smooth-LDDT backward launcher requires a non-empty plan")
        if not torch.are_deterministic_algorithms_enabled():
            grad_predicted = torch.zeros_like(predicted_coordinates, dtype=torch.float32)
            grad_true_ptr = grad_predicted if grad_true_pair_distances is None else grad_true_pair_distances
            kernel._smooth_lddt_pairs_atomic_backward[plan.forward_grid](
                predicted_coordinates,
                pair_indices,
                grad_out,
                true_pair_distances,
                grad_predicted,
                grad_true_ptr,
                plan.total_pairs,
                int(plan.problem.pairs),
                int(plan.problem.atoms),
                False,
                plan.forward_wide_offsets,
                True,
                plan.true_batch_stride,
                grad_true_pair_distances is not None,
                plan.config.forward_block_pairs,
                num_warps=plan.config.num_warps,
            )
            return grad_predicted.to(predicted_coordinates.dtype)
        atoms = int(plan.problem.atoms)
        order, boundaries = (
            _pair_list_inversion(pair_indices, atoms, plan.inversion_dtype) if inversion is None else inversion
        )
        grad_predicted = torch.empty_like(predicted_coordinates)
        # This pointer is inert when its constexpr demand flag is false.
        grad_true_ptr = grad_predicted if grad_true_pair_distances is None else grad_true_pair_distances
        kernel._smooth_lddt_sparse_backward[plan.backward_grid](
            predicted_coordinates,
            pair_indices,
            order,
            boundaries,
            true_pair_distances,
            grad_out,
            grad_predicted,
            grad_true_ptr,
            plan.total_pairs,
            atoms,
            plan.true_batch_stride,
            grad_true_pair_distances is not None,
            plan.true_batch_stride == 0,
            plan.backward_order_wide,
            32 if plan.backward_block_rows > 1 else plan.config.backward_block_pairs,
            plan.backward_block_rows,
            num_warps=plan.config.num_warps,
        )
        return grad_predicted


_PAIR_LIST_INVERSIONS: WeakIdKeyDictionary = WeakIdKeyDictionary()


def _pair_list_inversion(
    pair_indices: Tensor,
    atoms: int,
    index_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    r"""Return stable current-pair end order and atom-row boundaries."""

    with torch.cuda.device(pair_indices.device):
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing or torch.is_inference(pair_indices):
        return _invert_pair_list(pair_indices, atoms, index_dtype)
    version = pair_indices._version
    stream = torch.cuda.current_stream(pair_indices.device).cuda_stream
    cached = _PAIR_LIST_INVERSIONS.get(pair_indices)
    if (
        cached is not None
        and cached[0] == version
        and cached[1] == atoms
        and cached[2] == index_dtype
        and cached[3] == stream
        and torch.equal(pair_indices, cached[4])
    ):
        return cached[5], cached[6]
    order, boundaries = _invert_pair_list(pair_indices, atoms, index_dtype)
    with torch.inference_mode(False):
        snapshot = pair_indices.clone()
    _PAIR_LIST_INVERSIONS[pair_indices] = (version, atoms, index_dtype, stream, snapshot, order, boundaries)
    return order, boundaries


def _invert_pair_list(
    pair_indices: Tensor,
    atoms: int,
    index_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    r"""Return pair-end order and atom-row boundaries, stably sorted by atom."""

    ends = pair_indices.clamp(0, atoms - 1).to(index_dtype)
    sorted_ends, order = torch.sort(ends, stable=True)
    queries = torch.arange(atoms + 1, device=pair_indices.device, dtype=index_dtype)
    return order.to(index_dtype), torch.searchsorted(sorted_ends, queries).to(index_dtype)
