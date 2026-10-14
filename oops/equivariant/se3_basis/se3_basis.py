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

from oops.runtime import is_compiling, is_external_fake, require_no_forward_mode, unsupported

from . import triton as _triton_equivariant


def se3_basis(
    relative_positions: Tensor,
    max_degree: int,
    *,
    use_pad_trick: bool = False,
    fully_fused: bool = False,
) -> dict[str, Tensor]:
    r"""Build RFdiffusion-compatible degree-one SE(3) basis tensors.

    Args:
        relative_positions: Contiguous, non-empty CUDA float16, bfloat16, or
            float32 edge-relative positions shaped ``(E, 3)``.
        max_degree: Maximum input and output representation degree. Only one is
            supported.
        use_pad_trick: Add one zero-filled column to pairwise and ``out*_fused`` bases.
        fully_fused: Include the additional ``fully_fused`` basis.

    Returns:
        A dictionary of tensors on the input device and in its dtype. Every shape
        begins with ``E``. Pairwise keys ``"0,0"``, ``"0,1"``, ``"1,0"``, and
        ``"1,1"`` have trailing shapes ``(1, 1, 1)``, ``(1, 1, 3)``, ``(3, 1, 1)``,
        and ``(3, 3, 3)``. Additional keys ``out0_fused``, ``out1_fused``, ``in0_fused``,
        and ``in1_fused`` have trailing shapes ``(4, 2, 1)``, ``(4, 4, 3)``,
        ``(1, 2, 4)``, and ``(3, 4, 4)``. Requested ``fully_fused`` has trailing
        shape ``(4, 6, 4)``. Padding increases the last dimension of pairwise and
        ``out*_fused`` tensors by one; ``in*_fused`` and ``fully_fused`` are unchanged.

    Raises:
        ValueError: Relative positions are not shaped ``(E, 3)``.
        TypeError: Relative positions are not floating point.
        OOpsUnsupportedError: Degree, device, dtype, layout, tensor extent, or
            forward-mode differentiation is unsupported.

    Notes:
        Basis construction uses integral-normalized real spherical harmonics and
        the RFdiffusion Clebsch-Gordan convention. Relative positions are normalized
        as ``v / max(norm(v), 1e-12)``.

    Examples:
        >>> import torch
        >>> from oops import se3_basis
        >>> if torch.cuda.is_available():
        ...     positions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
        ...     bases = se3_basis(positions, 1, fully_fused=True)
        ...     assert bases["1,1"].shape == (1, 3, 3, 3)
        ...     assert bases["fully_fused"].shape == (1, 4, 6, 4)
    """

    require_no_forward_mode("se3_basis", relative_positions)
    _check_inputs(relative_positions, max_degree)
    compiling = is_compiling()
    position_type = type(relative_positions)
    external_fake = (
        not compiling
        and position_type is not Tensor
        and position_type is not torch.nn.Parameter
        and is_external_fake(relative_positions)
    )
    if not _triton_equivariant.supports_se3_basis_l1(
        relative_positions,
        max_degree,
        use_pad_trick=use_pad_trick,
        fully_fused=fully_fused,
        cache=not external_fake,
    ):
        raise unsupported(
            "se3_basis",
            "requires max_degree 1 over a contiguous non-empty CUDA float16/bfloat16/float32 "
            "(num_edges, 3) tensor whose launch grid fits CUDA limits",
            max_degree=max_degree,
            dtype=relative_positions.dtype,
            device=relative_positions.device,
            shape=tuple(relative_positions.shape),
            contiguous=relative_positions.is_contiguous(),
        )
    return _triton_equivariant.triton_se3_basis_l1(
        relative_positions,
        use_pad_trick=use_pad_trick,
        fully_fused=fully_fused,
        compiling=compiling,
        external_fake=external_fake,
    )


def _check_inputs(relative_positions: Tensor, max_degree: int) -> None:
    if relative_positions.shape[-1:] != (3,):
        raise ValueError(f"relative_positions must have shape (num_edges, 3), got {tuple(relative_positions.shape)}")
    if relative_positions.ndim != 2:
        raise ValueError(f"relative_positions must have shape (num_edges, 3), got {tuple(relative_positions.shape)}")
    if not relative_positions.is_floating_point():
        raise TypeError(f"relative_positions must be floating point, got {relative_positions.dtype}")
    if max_degree != 1:
        raise unsupported("se3_basis", "serves max_degree == 1 only", max_degree=max_degree)
