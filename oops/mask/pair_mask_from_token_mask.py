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
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import unsupported

from .triton import packed_pair_mask


def pair_mask_from_token_mask(token_mask: Tensor | NestedTensor) -> Tensor | NestedTensor:
    r"""Form pair validity by taking the outer conjunction of token validity.

    Args:
        token_mask: Dense validity values shaped ``(..., N)``, or a
            batch-first NestedTensor with canonical ``(N_i,)`` elements
            in row-major packed order. Nonzero values count as true.

    Returns:
        A bool tensor shaped ``(..., N, N)``, or a NestedTensor containing
        explicit packed ``(N_i, N_i)`` elements. The result equals
        ``token_mask.bool()[..., :, None] & token_mask.bool()[..., None, :]``.

    Raises:
        ValueError: The dense rank or nested logical topology is malformed.
        OOpsUnsupportedError: Nested storage is not supported CUDA bool,
            integer, or floating-point storage, its launch extent is
            unsupported, or the wrapper batch is empty.

    Notes:
        Dense inputs accept any device and dtype supported by conversion to
        bool. Nested inputs require CUDA storage and preserve sample topology.
        Empty token elements produce empty pair elements.
        Both representations support full-graph compilation. Outputs are
        non-differentiable.

    Examples:
        >>> import torch
        >>> from oops import pair_mask_from_token_mask
        >>> mask = torch.tensor([True, False])
        >>> pair_mask_from_token_mask(mask).tolist()
        [[True, False], [False, False]]
    """

    if not isinstance(token_mask, NestedTensor):
        if token_mask.dim() < 1:
            raise ValueError("pair_mask_from_token_mask token_mask must have at least one dimension")
        dense_mask = token_mask.bool()
        return dense_mask[..., :, None] & dense_mask[..., None, :]
    if (
        not token_mask.batch_first
        or token_mask.dim() != 2
        or token_mask.ragged_dims != (0,)
        or token_mask.packed_dim_order != (0,)
    ):
        raise ValueError("pair_mask_from_token_mask nested elements require canonical (N_i,) packing")
    if len(token_mask) == 0:
        raise unsupported(
            "pair_mask_from_token_mask",
            "does not serve an empty NestedTensor wrapper batch without retained element rank",
        )
    packed_values = token_mask.concat
    if packed_values.is_complex():
        raise unsupported(
            "pair_mask_from_token_mask",
            "requires bool, integer, or floating-point packed values",
            dtype=packed_values.dtype,
        )
    token_values = packed_values.bool()
    if token_values.dim() != 1 or not token_values.is_contiguous():
        raise unsupported(
            "pair_mask_from_token_mask",
            "requires contiguous one-dimensional packed token values",
            packed_shape=tuple(token_values.shape),
        )
    if not token_values.is_cuda:
        raise unsupported("pair_mask_from_token_mask", "requires CUDA packed values", device=token_values.device)

    offsets = token_mask.ragged_level_offsets(0, dtype=torch.int64)
    output_values = packed_pair_mask(token_values, offsets)
    return token_mask.packed_with_square_lengths(output_values, offsets[1:] - offsets[:-1])
