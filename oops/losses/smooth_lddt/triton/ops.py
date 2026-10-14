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

from oops.runtime import cuda_device_caps_if_needed, functorch_transforms_active, is_compiling, is_external_fake
from oops.runtime import needs_grad as tensors_need_grad

from .generic import (
    _pair_distance_problem,
    _run_pair_distances,
    _run_pair_distances_backward,
    _run_smooth_lddt_loss_dense,
    _run_smooth_lddt_loss_dense_backward,
    _run_smooth_lddt_loss_mask_backward,
    _run_smooth_lddt_loss_mask_forward,
    _run_smooth_lddt_loss_sparse_backward,
    _run_smooth_lddt_loss_sparse_coordinates_forward,
    _run_sparse_loss,
    _sparse_loss_problem,
)
from .profiles import (
    SMOOTH_LDDT_MASK,
    SMOOTH_LDDT_PAIR_DISTANCE,
    SMOOTH_LDDT_SPARSE,
    dense_target_changes_plan,
    resolve_dense_plan,
)
from .schedule import SmoothLddtDenseProblem, SmoothLddtMaskProblem


def triton_smooth_lddt_loss_dense(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    is_nucleotide: Tensor,
    coordinates_mask: Tensor,
    *,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
) -> Tensor:

    if (
        tensors_need_grad(predicted_coordinates, true_coordinates, coordinates_mask)
        or is_compiling()
        or is_external_fake(predicted_coordinates)
    ):
        loss, _accum = _smooth_lddt_loss_dense_forward(
            predicted_coordinates,
            true_coordinates,
            is_nucleotide,
            coordinates_mask,
            float(nucleic_acid_cutoff),
            float(other_cutoff),
        )
    else:
        loss, _accum = _run_smooth_lddt_loss_dense(
            predicted_coordinates,
            true_coordinates,
            is_nucleotide,
            coordinates_mask,
            float(nucleic_acid_cutoff),
            float(other_cutoff),
        )
    return loss


def triton_smooth_lddt_pair_distances(coordinates: Tensor, pair_indices: Tensor) -> Tensor:

    if tensors_need_grad(coordinates) or is_compiling() or is_external_fake(coordinates):
        return _smooth_lddt_pair_distances_forward(coordinates, pair_indices)
    return _run_pair_distances(coordinates, pair_indices)


def triton_smooth_lddt_loss_mask(predicted_coordinates: Tensor, true_coordinates: Tensor, lddt_mask: Tensor) -> Tensor:

    loss, _accum = _smooth_lddt_loss_mask_forward(predicted_coordinates, true_coordinates, lddt_mask)
    return loss


def triton_smooth_lddt_loss_sparse(
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
) -> Tensor:

    if (
        tensors_need_grad(predicted_coordinates, true_pair_distances)
        or is_compiling()
        or is_external_fake(predicted_coordinates)
    ):
        return _smooth_lddt_loss_sparse_forward(predicted_coordinates, pair_indices, true_pair_distances)
    return _run_sparse_loss(predicted_coordinates, pair_indices, true_pair_distances)


def triton_smooth_lddt_loss_sparse_coordinates(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    pair_indices: Tensor,
) -> Tensor:

    loss, _true_pair_distances = _smooth_lddt_loss_sparse_coordinates_forward(
        predicted_coordinates, true_coordinates, pair_indices
    )
    return loss


@torch.library.custom_op("ooperators::_smooth_lddt_loss_dense_forward", mutates_args=(), device_types="cuda")
def _smooth_lddt_loss_dense_forward(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    is_nucleotide: Tensor,
    coordinates_mask: Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
) -> tuple[Tensor, Tensor]:
    return _run_smooth_lddt_loss_dense(
        predicted_coordinates,
        true_coordinates,
        is_nucleotide,
        coordinates_mask,
        nucleic_acid_cutoff,
        other_cutoff,
    )


@torch.library.register_fake(_smooth_lddt_loss_dense_forward)
def _smooth_lddt_loss_dense_forward_fake(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    is_nucleotide: Tensor,
    coordinates_mask: Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
) -> tuple[Tensor, Tensor]:
    del true_coordinates, is_nucleotide, coordinates_mask, nucleic_acid_cutoff, other_cutoff
    return (
        torch.empty((), device=predicted_coordinates.device, dtype=torch.float32),
        torch.empty((predicted_coordinates.shape[0], 2), device=predicted_coordinates.device, dtype=torch.float32),
    )


_DENSE_LOSS_BACKWARD_SCHEMA = (
    "(Tensor grad_out, Tensor predicted_coordinates, Tensor true_coordinates, Tensor is_nucleotide, "
    "Tensor coordinates_mask, Tensor accum, float nucleic_acid_cutoff, float other_cutoff, "
    "bool need_grad_predicted, bool need_grad_true, bool need_grad_coordinates_mask) "
    "-> (Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_smooth_lddt_loss_dense_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_DENSE_LOSS_BACKWARD_SCHEMA,
)
def _smooth_lddt_loss_dense_backward(
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
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    return _run_smooth_lddt_loss_dense_backward(
        grad_out,
        predicted_coordinates,
        true_coordinates,
        is_nucleotide,
        coordinates_mask,
        accum,
        nucleic_acid_cutoff,
        other_cutoff,
        need_grad_predicted,
        need_grad_true,
        need_grad_coordinates_mask,
    )


@torch.library.register_fake(_smooth_lddt_loss_dense_backward)
def _smooth_lddt_loss_dense_backward_fake(
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
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    del grad_out, is_nucleotide, accum, nucleic_acid_cutoff, other_cutoff
    return (
        torch.empty_like(predicted_coordinates) if need_grad_predicted else None,
        torch.empty_like(true_coordinates) if need_grad_true else None,
        torch.empty_like(coordinates_mask) if need_grad_coordinates_mask else None,
    )


def _smooth_lddt_loss_dense_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, float, float],
    output: tuple[Tensor, Tensor],
) -> None:
    (
        predicted_coordinates,
        true_coordinates,
        is_nucleotide,
        coordinates_mask,
        nucleic_acid_cutoff,
        other_cutoff,
    ) = inputs
    _loss, accum = output
    ctx.mark_non_differentiable(accum)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(predicted_coordinates, true_coordinates, is_nucleotide, coordinates_mask, accum)
    ctx.opaque_backward = is_compiling() or is_external_fake(predicted_coordinates) or functorch_transforms_active()
    if not ctx.opaque_backward:
        problem = SmoothLddtDenseProblem(
            predicted_coordinates.shape[0], predicted_coordinates.shape[1], predicted_coordinates.dtype
        )
        target = cuda_device_caps_if_needed(predicted_coordinates.device, dense_target_changes_plan(problem))
        ctx.backward_plan = resolve_dense_plan(problem, target=target)
    ctx.nucleic_acid_cutoff = nucleic_acid_cutoff
    ctx.other_cutoff = other_cutoff


def _smooth_lddt_loss_dense_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
    grad_accum: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, None, Tensor | None, None, None]:
    del grad_accum
    if grad_out is None:
        return None, None, None, None, None, None
    predicted_coordinates, true_coordinates, is_nucleotide, coordinates_mask, accum = ctx.saved_tensors
    need_grad_predicted, need_grad_true, _need_grad_is_nucleotide, need_grad_coordinates_mask = ctx.needs_input_grad[:4]
    if ctx.opaque_backward:
        grad_predicted, grad_true, grad_coordinates_mask = _smooth_lddt_loss_dense_backward(
            grad_out,
            predicted_coordinates,
            true_coordinates,
            is_nucleotide,
            coordinates_mask,
            accum,
            ctx.nucleic_acid_cutoff,
            ctx.other_cutoff,
            need_grad_predicted,
            need_grad_true,
            need_grad_coordinates_mask,
        )
    else:
        grad_predicted, grad_true, grad_coordinates_mask = _run_smooth_lddt_loss_dense_backward(
            grad_out,
            predicted_coordinates,
            true_coordinates,
            is_nucleotide,
            coordinates_mask,
            accum,
            ctx.nucleic_acid_cutoff,
            ctx.other_cutoff,
            need_grad_predicted,
            need_grad_true,
            need_grad_coordinates_mask,
            ctx.backward_plan,
        )
    return grad_predicted, grad_true, None, grad_coordinates_mask, None, None


torch.library.register_autograd(
    _smooth_lddt_loss_dense_forward,
    _smooth_lddt_loss_dense_forward_autograd_backward,
    setup_context=_smooth_lddt_loss_dense_forward_setup_context,
)


@torch.library.custom_op("ooperators::_smooth_lddt_pair_distances_forward", mutates_args=(), device_types="cuda")
def _smooth_lddt_pair_distances_forward(coordinates: Tensor, pair_indices: Tensor) -> Tensor:
    return _run_pair_distances(coordinates, pair_indices)


@torch.library.register_fake(_smooth_lddt_pair_distances_forward)
def _smooth_lddt_pair_distances_forward_fake(coordinates: Tensor, pair_indices: Tensor) -> Tensor:
    pairs = pair_indices.numel() // 2
    shape = (pairs,) if coordinates.dim() == 2 else (coordinates.shape[0], pairs)
    return torch.empty(shape, device=coordinates.device, dtype=torch.float32)


@torch.library.custom_op("ooperators::_smooth_lddt_pair_distances_backward", mutates_args=(), device_types="cuda")
def _smooth_lddt_pair_distances_backward(grad_out: Tensor, coordinates: Tensor, pair_indices: Tensor) -> Tensor:
    return _run_pair_distances_backward(grad_out, coordinates, pair_indices)


@torch.library.register_fake(_smooth_lddt_pair_distances_backward)
def _smooth_lddt_pair_distances_backward_fake(grad_out: Tensor, coordinates: Tensor, pair_indices: Tensor) -> Tensor:
    del grad_out, pair_indices
    return torch.empty_like(coordinates)


def _smooth_lddt_pair_distances_forward_setup_context(ctx, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
    del output
    coordinates, pair_indices = inputs
    ctx.save_for_backward(coordinates, pair_indices)
    ctx.direct_backward = not is_compiling() and not is_external_fake(coordinates) and not functorch_transforms_active()
    if ctx.direct_backward:
        ctx.backward_plan = SMOOTH_LDDT_PAIR_DISTANCE.resolve(_pair_distance_problem(coordinates, pair_indices))


def _smooth_lddt_pair_distances_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None]:
    coordinates, pair_indices = ctx.saved_tensors
    if ctx.direct_backward:
        grad_coordinates = _run_pair_distances_backward(grad_out, coordinates, pair_indices, ctx.backward_plan)
    else:
        grad_coordinates = _smooth_lddt_pair_distances_backward(grad_out, coordinates, pair_indices)
    return grad_coordinates, None


torch.library.register_autograd(
    _smooth_lddt_pair_distances_forward,
    _smooth_lddt_pair_distances_forward_autograd_backward,
    setup_context=_smooth_lddt_pair_distances_forward_setup_context,
)


@torch.library.custom_op("ooperators::_smooth_lddt_loss_mask_forward", mutates_args=(), device_types="cuda")
def _smooth_lddt_loss_mask_forward(
    predicted_coordinates: Tensor, true_coordinates: Tensor, lddt_mask: Tensor
) -> tuple[Tensor, Tensor]:
    return _run_smooth_lddt_loss_mask_forward(predicted_coordinates, true_coordinates, lddt_mask)


@torch.library.register_fake(_smooth_lddt_loss_mask_forward)
def _smooth_lddt_loss_mask_forward_fake(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    lddt_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    del true_coordinates, lddt_mask
    return (
        torch.empty((), device=predicted_coordinates.device, dtype=torch.float32),
        torch.empty((1, 2), device=predicted_coordinates.device, dtype=torch.float32),
    )


_MASK_LOSS_BACKWARD_SCHEMA = (
    "(Tensor grad_out, Tensor predicted_coordinates, Tensor true_coordinates, Tensor lddt_mask, Tensor accum, "
    "bool need_grad_predicted, bool need_grad_true) -> (Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_smooth_lddt_loss_mask_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_MASK_LOSS_BACKWARD_SCHEMA,
)
def _smooth_lddt_loss_mask_backward(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    lddt_mask: Tensor,
    accum: Tensor,
    need_grad_predicted: bool,
    need_grad_true: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _run_smooth_lddt_loss_mask_backward(
        grad_out,
        predicted_coordinates,
        true_coordinates,
        lddt_mask,
        accum,
        need_grad_predicted,
        need_grad_true,
    )


@torch.library.register_fake(_smooth_lddt_loss_mask_backward)
def _smooth_lddt_loss_mask_backward_fake(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    lddt_mask: Tensor,
    accum: Tensor,
    need_grad_predicted: bool,
    need_grad_true: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del grad_out, lddt_mask, accum
    return (
        torch.empty_like(predicted_coordinates) if need_grad_predicted else None,
        torch.empty_like(true_coordinates) if need_grad_true else None,
    )


def _smooth_lddt_loss_mask_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor],
) -> None:
    predicted_coordinates, true_coordinates, lddt_mask = inputs
    _loss, accum = output
    ctx.mark_non_differentiable(accum)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(predicted_coordinates, true_coordinates, lddt_mask, accum)
    ctx.direct_backward = (
        not is_compiling() and not is_external_fake(predicted_coordinates) and not functorch_transforms_active()
    )
    if ctx.direct_backward:
        ctx.backward_plan = SMOOTH_LDDT_MASK.resolve(
            SmoothLddtMaskProblem(
                predicted_coordinates.shape[0],
                predicted_coordinates.shape[1],
                true_coordinates.dim() == 3,
            )
        )


def _smooth_lddt_loss_mask_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
    grad_accum: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, None]:
    del grad_accum
    if grad_out is None:
        return None, None, None
    predicted_coordinates, true_coordinates, lddt_mask, accum = ctx.saved_tensors
    need_grad_predicted, need_grad_true = ctx.needs_input_grad[:2]
    if ctx.direct_backward:
        grad_predicted, grad_true = _run_smooth_lddt_loss_mask_backward(
            grad_out,
            predicted_coordinates,
            true_coordinates,
            lddt_mask,
            accum,
            need_grad_predicted,
            need_grad_true,
            ctx.backward_plan,
        )
    else:
        grad_predicted, grad_true = _smooth_lddt_loss_mask_backward(
            grad_out,
            predicted_coordinates,
            true_coordinates,
            lddt_mask,
            accum,
            need_grad_predicted,
            need_grad_true,
        )
    return grad_predicted, grad_true, None


torch.library.register_autograd(
    _smooth_lddt_loss_mask_forward,
    _smooth_lddt_loss_mask_forward_autograd_backward,
    setup_context=_smooth_lddt_loss_mask_forward_setup_context,
)


@torch.library.custom_op("ooperators::_smooth_lddt_loss_sparse_forward", mutates_args=(), device_types="cuda")
def _smooth_lddt_loss_sparse_forward(
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
) -> Tensor:
    return _run_sparse_loss(predicted_coordinates, pair_indices, true_pair_distances)


@torch.library.register_fake(_smooth_lddt_loss_sparse_forward)
def _smooth_lddt_loss_sparse_forward_fake(
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
) -> Tensor:
    del pair_indices, true_pair_distances
    return torch.empty((), device=predicted_coordinates.device, dtype=torch.float32)


_SPARSE_LOSS_BACKWARD_SCHEMA = (
    "(Tensor grad_out, Tensor predicted_coordinates, Tensor pair_indices, Tensor true_pair_distances, "
    "Tensor? true_coordinates, bool need_predicted, bool need_truth) -> (Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_smooth_lddt_loss_sparse_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_SPARSE_LOSS_BACKWARD_SCHEMA,
)
def _smooth_lddt_loss_sparse_backward(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    true_coordinates: Tensor | None,
    need_predicted: bool,
    need_truth: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _run_smooth_lddt_loss_sparse_backward(
        grad_out,
        predicted_coordinates,
        pair_indices,
        true_pair_distances,
        true_coordinates,
        need_predicted,
        need_truth,
    )


@torch.library.register_fake(_smooth_lddt_loss_sparse_backward)
def _smooth_lddt_loss_sparse_backward_fake(
    grad_out: Tensor,
    predicted_coordinates: Tensor,
    pair_indices: Tensor,
    true_pair_distances: Tensor,
    true_coordinates: Tensor | None,
    need_predicted: bool,
    need_truth: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del grad_out, pair_indices
    grad_truth = (
        torch.empty_like(true_pair_distances if true_coordinates is None else true_coordinates) if need_truth else None
    )
    return torch.empty_like(predicted_coordinates) if need_predicted else None, grad_truth


def _smooth_lddt_loss_sparse_forward_setup_context(ctx, inputs: tuple[Tensor, Tensor, Tensor], output: Tensor) -> None:
    del output
    predicted_coordinates, pair_indices, true_pair_distances = inputs
    ctx.save_for_backward(predicted_coordinates, pair_indices, true_pair_distances)
    ctx.direct_backward = (
        not is_compiling() and not is_external_fake(predicted_coordinates) and not functorch_transforms_active()
    )
    if ctx.direct_backward:
        ctx.backward_plan = SMOOTH_LDDT_SPARSE.resolve(
            _sparse_loss_problem(predicted_coordinates, pair_indices, true_pair_distances)
        )


def _smooth_lddt_loss_sparse_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
) -> tuple[Tensor | None, None, Tensor | None]:
    predicted_coordinates, pair_indices, true_pair_distances = ctx.saved_tensors
    if ctx.direct_backward:
        grad_predicted, grad_true_pair_distances = _run_smooth_lddt_loss_sparse_backward(
            grad_out,
            predicted_coordinates,
            pair_indices,
            true_pair_distances,
            None,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[2],
            ctx.backward_plan,
        )
    else:
        grad_predicted, grad_true_pair_distances = _smooth_lddt_loss_sparse_backward(
            grad_out,
            predicted_coordinates,
            pair_indices,
            true_pair_distances,
            None,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[2],
        )
    return grad_predicted, None, grad_true_pair_distances


torch.library.register_autograd(
    _smooth_lddt_loss_sparse_forward,
    _smooth_lddt_loss_sparse_forward_autograd_backward,
    setup_context=_smooth_lddt_loss_sparse_forward_setup_context,
)


@torch.library.custom_op(
    "ooperators::_smooth_lddt_loss_sparse_coordinates_forward", mutates_args=(), device_types="cuda"
)
def _smooth_lddt_loss_sparse_coordinates_forward(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    pair_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    return _run_smooth_lddt_loss_sparse_coordinates_forward(predicted_coordinates, true_coordinates, pair_indices)


@torch.library.register_fake(_smooth_lddt_loss_sparse_coordinates_forward)
def _smooth_lddt_loss_sparse_coordinates_forward_fake(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    pair_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    pairs = pair_indices.numel() // 2
    shape = (pairs,) if true_coordinates.dim() == 2 else (true_coordinates.shape[0], pairs)
    return (
        torch.empty((), device=predicted_coordinates.device, dtype=torch.float32),
        torch.empty(shape, device=true_coordinates.device, dtype=torch.float32),
    )


def _smooth_lddt_loss_sparse_coordinates_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor],
) -> None:
    predicted_coordinates, true_coordinates, pair_indices = inputs
    _loss, true_pair_distances = output
    ctx.mark_non_differentiable(true_pair_distances)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(predicted_coordinates, true_coordinates, pair_indices, true_pair_distances)


def _smooth_lddt_loss_sparse_coordinates_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
    grad_true_pair_distances: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, None]:
    del grad_true_pair_distances
    if grad_out is None:
        return None, None, None
    predicted_coordinates, true_coordinates, pair_indices, true_pair_distances = ctx.saved_tensors
    grad_predicted, grad_true = _smooth_lddt_loss_sparse_backward(
        grad_out,
        predicted_coordinates,
        pair_indices,
        true_pair_distances,
        true_coordinates,
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[1],
    )
    return grad_predicted, grad_true, None


torch.library.register_autograd(
    _smooth_lddt_loss_sparse_coordinates_forward,
    _smooth_lddt_loss_sparse_coordinates_forward_autograd_backward,
    setup_context=_smooth_lddt_loss_sparse_coordinates_forward_setup_context,
)
