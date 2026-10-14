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

from oops.runtime import is_compiling, is_external_fake, needs_grad

from .generic import _run_segmented_masked_knn, _run_segmented_masked_knn_backward


def segmented_masked_knn(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(coordinates)
    if needs_grad(coordinates) or compiling or external_fake:
        return _segmented_masked_knn_forward(
            coordinates,
            mask,
            offsets,
            batch_indices,
            num_neighbors,
            eps,
        )
    return _run_segmented_masked_knn(
        coordinates,
        mask,
        offsets,
        batch_indices,
        num_neighbors,
        eps,
    )


@torch.library.custom_op("ooperators::_segmented_masked_knn_forward", mutates_args=(), device_types="cuda")
def _segmented_masked_knn_forward(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return _run_segmented_masked_knn(
        coordinates,
        mask,
        offsets,
        batch_indices,
        num_neighbors,
        eps,
    )


@torch.library.register_fake(_segmented_masked_knn_forward)
def _segmented_masked_knn_forward_fake(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    del mask, offsets, batch_indices, eps
    shape = (coordinates.shape[0], num_neighbors)
    return (
        torch.empty(shape, device=coordinates.device, dtype=coordinates.dtype),
        torch.empty(shape, device=coordinates.device, dtype=torch.long),
        torch.empty(shape, device=coordinates.device, dtype=torch.bool),
    )


def _segmented_masked_knn_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, int, float],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    coordinates, _mask, offsets, batch_indices, num_neighbors, eps = inputs
    _distances, local_indices, valid = output
    ctx.mark_non_differentiable(local_indices, valid)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(coordinates, offsets, batch_indices, local_indices, valid)
    ctx.num_neighbors = num_neighbors
    ctx.eps = eps


def _segmented_masked_knn_forward_autograd_backward(
    ctx,
    grad_distances: Tensor | None,
    grad_local_indices: Tensor | None,
    grad_valid: Tensor | None,
) -> tuple[Tensor | None, None, None, None, None, None]:
    del grad_local_indices, grad_valid
    grad_coordinates = None
    if ctx.needs_input_grad[0] and grad_distances is not None:
        coordinates, offsets, batch_indices, local_indices, valid = ctx.saved_tensors
        grad_coordinates = _segmented_masked_knn_backward(
            grad_distances,
            coordinates,
            offsets,
            batch_indices,
            local_indices,
            valid,
            ctx.num_neighbors,
            ctx.eps,
        )
    return grad_coordinates, None, None, None, None, None


torch.library.register_autograd(
    _segmented_masked_knn_forward,
    _segmented_masked_knn_forward_autograd_backward,
    setup_context=_segmented_masked_knn_forward_setup_context,
)


@torch.library.custom_op(
    "ooperators::_segmented_masked_knn_backward",
    mutates_args=(),
    device_types="cuda",
)
def _segmented_masked_knn_backward(
    grad_distances: Tensor,
    coordinates: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    local_indices: Tensor,
    valid: Tensor,
    num_neighbors: int,
    eps: float,
) -> Tensor:
    return _run_segmented_masked_knn_backward(
        grad_distances, coordinates, offsets, batch_indices, local_indices, valid, num_neighbors, eps
    )


@torch.library.register_fake(_segmented_masked_knn_backward)
def _segmented_masked_knn_backward_fake(
    grad_distances: Tensor,
    coordinates: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    local_indices: Tensor,
    valid: Tensor,
    num_neighbors: int,
    eps: float,
) -> Tensor:
    del grad_distances, offsets, batch_indices, local_indices, valid, num_neighbors, eps
    return torch.empty_like(coordinates)
