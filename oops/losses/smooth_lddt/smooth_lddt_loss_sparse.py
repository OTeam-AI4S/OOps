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

from oops.runtime import empty_with_dependencies, require_no_forward_mode, require_same_device, unsupported

from .pairs import LDDTPairIndices
from .pairs import flat_pair_indices as _flat_pair_indices
from .smooth_lddt_pair_indices import smooth_lddt_pair_indices
from .triton import (
    supports_smooth_lddt_loss_mask,
    supports_smooth_lddt_loss_sparse,
    supports_smooth_lddt_loss_sparse_coordinates,
    triton_smooth_lddt_loss_mask,
    triton_smooth_lddt_loss_sparse,
    triton_smooth_lddt_loss_sparse_coordinates,
)


def smooth_lddt_loss_sparse(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor | None = None,
    lddt_mask: Tensor | None = None,
    *,
    pair_indices: LDDTPairIndices | None = None,
    true_pair_distances: Tensor | None = None,
) -> Tensor:
    r"""Compute smooth-LDDT loss for selected atom pairs.

    Args:
        predicted_coordinates: Contiguous CUDA coordinates of shape ``(B, A, 3)``
            and dtype float16, bfloat16, or float32.
        true_coordinates: Contiguous truth of shape ``(A, 3)`` shared across samples
            or ``(B, A, 3)``, on the prediction device. Its supported storage dtype
            is independent of the prediction dtype. Mutually exclusive with
            ``true_pair_distances``.
        lddt_mask: Boolean square ``(A, A)`` pair mask on the prediction device.
            Mutually exclusive with ``pair_indices``.
        pair_indices: Int32 or int64 pair indices on the prediction device, as a
            flat interleaved tensor, ``(P, 2)`` tensor, or tuple of two ``(P,)``
            tensors. Use instead of ``lddt_mask``; indices must lie in ``[0, A)``.
        true_pair_distances: Contiguous float16/bfloat16/float32 truth distances of
            shape ``(P,)`` shared across samples or ``(B, P)``, on the prediction
            device. Use instead of ``true_coordinates``.

    Returns:
        A scalar float32 loss: one minus the mean of
        ``sigmoid(t - abs(d_pred - d_true))`` over thresholds ``t = 0.5, 1, 2, 4``,
        samples, and selected pairs. No selected pairs give loss one and zero
        gradients. Explicit pair lists retain their order, duplicates, and diagonal
        entries; this function applies no distance cutoff or diagonal filtering.

    Raises:
        TypeError: If exactly one pair representation and exactly one truth
            representation are not supplied.
        ValueError: If devices differ or the pair-index layout is malformed.
        OOpsUnsupportedError: If inputs are outside the supported dtype, shape,
            layout, or device domain, forward-mode AD is requested, or strict
            determinism is unsupported for the supplied shape and requested gradients.

    Examples:
        >>> import torch
        >>> from oops import smooth_lddt_loss_sparse
        >>> if torch.cuda.is_available():
        ...     prediction = torch.zeros(1, 2, 3, device="cuda")
        ...     truth = torch.zeros(2, 3, device="cuda")
        ...     pairs = torch.tensor([0, 1], device="cuda")
        ...     loss = smooth_lddt_loss_sparse(prediction, truth, pair_indices=pairs)
        ...     assert loss.shape == () and loss.dtype == torch.float32
    """
    require_no_forward_mode("smooth_lddt_loss_sparse", predicted_coordinates, true_coordinates, true_pair_distances)

    if (pair_indices is None) == (lddt_mask is None):
        raise TypeError("smooth_lddt_loss_sparse requires exactly one of lddt_mask and pair_indices")
    if (true_coordinates is None) == (true_pair_distances is None):
        raise TypeError("smooth_lddt_loss_sparse requires exactly one of true_coordinates and true_pair_distances")

    pair_tensors = pair_indices if isinstance(pair_indices, tuple) else (pair_indices,)
    require_same_device(
        "smooth_lddt_loss_sparse",
        predicted_coordinates,
        true_coordinates,
        lddt_mask,
        true_pair_distances,
        *pair_tensors,
    )

    if pair_indices is None:
        assert lddt_mask is not None
        if true_pair_distances is None:
            assert true_coordinates is not None
            if not supports_smooth_lddt_loss_mask(predicted_coordinates, true_coordinates, lddt_mask):
                raise unsupported(
                    "smooth_lddt_loss_sparse",
                    "requires contiguous CUDA float16/bfloat16/float32 coordinates within CUDA launch limits with "
                    "a contiguous boolean (N_atom, N_atom) pair mask",
                    dtype=predicted_coordinates.dtype,
                    device=predicted_coordinates.device,
                    predicted_coordinates_shape=tuple(predicted_coordinates.shape),
                    true_coordinates_shape=tuple(true_coordinates.shape),
                    lddt_mask_dtype=lddt_mask.dtype,
                    lddt_mask_shape=tuple(lddt_mask.shape),
                )
            return triton_smooth_lddt_loss_mask(predicted_coordinates, true_coordinates, lddt_mask)
        pair_indices = smooth_lddt_pair_indices(lddt_mask)

    flat_pair_indices = _flat_pair_indices(pair_indices)
    if true_pair_distances is None:
        assert true_coordinates is not None
        if not supports_smooth_lddt_loss_sparse_coordinates(predicted_coordinates, true_coordinates, flat_pair_indices):
            raise unsupported(
                "smooth_lddt_loss_sparse",
                "requires contiguous CUDA float16/bfloat16/float32 (N_sample, N_atom, 3) predictions within CUDA "
                "launch limits and float16/bfloat16/float32 ground-truth coordinates with a contiguous 1-D "
                "int32/int64 pair index tensor",
                dtype=predicted_coordinates.dtype,
                device=predicted_coordinates.device,
                predicted_coordinates_shape=tuple(predicted_coordinates.shape),
                true_coordinates_shape=tuple(true_coordinates.shape),
                pair_indices_dtype=flat_pair_indices.dtype,
                pair_indices_numel=flat_pair_indices.numel(),
            )
        if flat_pair_indices.numel() == 0:
            return _empty_loss(predicted_coordinates, true_coordinates)
        return triton_smooth_lddt_loss_sparse_coordinates(predicted_coordinates, true_coordinates, flat_pair_indices)

    if not supports_smooth_lddt_loss_sparse(predicted_coordinates, flat_pair_indices, true_pair_distances):
        raise unsupported(
            "smooth_lddt_loss_sparse",
            "requires contiguous CUDA float16/bfloat16/float32 (N_sample, N_atom, 3) predictions within CUDA launch "
            "limits with a contiguous 1-D int32/int64 pair index tensor and float16/bfloat16/float32 pair "
            "distances",
            dtype=predicted_coordinates.dtype,
            device=predicted_coordinates.device,
            predicted_coordinates_shape=tuple(predicted_coordinates.shape),
            pair_indices_dtype=flat_pair_indices.dtype,
            pair_indices_numel=flat_pair_indices.numel(),
            true_distances_shape=tuple(true_pair_distances.shape),
        )
    if flat_pair_indices.numel() == 0:
        return _empty_loss(predicted_coordinates, true_pair_distances)
    return triton_smooth_lddt_loss_sparse(predicted_coordinates, flat_pair_indices, true_pair_distances)


def _empty_loss(predicted_coordinates: Tensor, truth: Tensor) -> Tensor:

    return empty_with_dependencies(
        torch.ones((), device=predicted_coordinates.device, dtype=torch.float32),
        predicted_coordinates,
        truth,
    )
