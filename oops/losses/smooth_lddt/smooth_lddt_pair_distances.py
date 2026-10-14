from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import require_no_forward_mode, require_same_device, unsupported

from .pairs import LDDTPairIndices
from .pairs import flat_pair_indices as _flat_pair_indices
from .triton import supports_smooth_lddt_pair_distances, triton_smooth_lddt_pair_distances


def smooth_lddt_pair_distances(coordinates: Tensor, pair_indices: LDDTPairIndices) -> Tensor:
    r"""Compute pair distances reusable by the sparse smooth-LDDT loss.

    Args:
        coordinates: Contiguous CUDA float16/bfloat16/float32 coordinates of shape
            ``(A, 3)`` or ``(B, A, 3)``.
        pair_indices: Int32 or int64 pair indices on the coordinate device, as a
            flat interleaved tensor, ``(P, 2)`` tensor, or tuple of two ``(P,)``
            tensors. Index values must lie in ``[0, A)``.

    Returns:
        Float32 Euclidean distances of shape ``(P,)`` or ``(B, P)``, on the
        coordinate device, in the supplied pair order. An empty list gives a
        corresponding empty output and zero coordinate gradients.

    Raises:
        ValueError: If devices differ or the pair-index layout is malformed.
        OOpsUnsupportedError: If inputs are outside the supported dtype, shape,
            layout, or device domain, or forward-mode AD is requested.

    Notes:
        Coordinates are widened before subtraction and norm evaluation. CUDA
        autocast does not narrow inputs or change the float32 output dtype.
        First-order reverse-mode gradients return in the coordinate storage dtype;
        coincident pairs have zero gradients. Forward-mode and higher-order AD are
        unsupported. Both forward and backward support strict deterministic mode.
        Index values are not validated; out-of-range indices are outside the numerical
        contract.

    Examples:
        >>> import torch
        >>> from oops import smooth_lddt_pair_distances
        >>> if torch.cuda.is_available():
        ...     coordinates = torch.zeros(2, 3, device="cuda")
        ...     pairs = torch.tensor([0, 1], device="cuda")
        ...     distances = smooth_lddt_pair_distances(coordinates, pairs)
        ...     assert distances.shape == (1,) and distances.dtype == torch.float32
    """
    require_no_forward_mode("smooth_lddt_pair_distances", coordinates)
    return _pair_distances(coordinates, pair_indices)


def _pair_distances(coordinates: Tensor, pair_indices: LDDTPairIndices) -> Tensor:
    if isinstance(pair_indices, tuple):
        require_same_device("smooth_lddt_pair_distances", coordinates, *pair_indices)
    else:
        require_same_device("smooth_lddt_pair_distances", coordinates, pair_indices)
    flat_pair_indices = _flat_pair_indices(pair_indices)
    if not supports_smooth_lddt_pair_distances(coordinates, flat_pair_indices):
        raise unsupported(
            "smooth_lddt_pair_distances",
            "requires contiguous CUDA float16/bfloat16/float32 coordinates within CUDA launch limits with a "
            "contiguous 1-D int32/int64 pair index tensor",
            dtype=coordinates.dtype,
            device=coordinates.device,
            coordinates_shape=tuple(coordinates.shape),
            pair_indices_dtype=flat_pair_indices.dtype,
            pair_indices_numel=flat_pair_indices.numel(),
        )
    if flat_pair_indices.numel() == 0:
        return coordinates[..., 0, :0].to(dtype=torch.float32)
    return triton_smooth_lddt_pair_distances(coordinates, flat_pair_indices)
