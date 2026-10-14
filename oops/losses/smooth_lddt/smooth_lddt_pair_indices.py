from __future__ import annotations

import torch
from torch import Tensor


def smooth_lddt_pair_indices(lddt_mask: Tensor) -> Tensor:
    r"""Convert a boolean pair mask into flat interleaved atom indices.

    Args:
        lddt_mask: Square boolean tensor of shape ``(A, A)``, on CPU or any device
            supported by ``torch.nonzero``.

    Returns:
        An int64 tensor of shape ``(2 * P,)`` on the mask device, storing
        ``i0, j0, i1, j1, ...`` for true entries in row-major order. An all-false
        mask gives an empty tensor. Diagonal entries are retained if selected.

    Raises:
        ValueError: If the mask is not a square two-dimensional boolean tensor.

    Notes:
        Precompute indices when reusing a mask in ``smooth_lddt_loss_sparse``.
        The number of pairs depends on mask values; construction may synchronize
        CUDA and follows the dynamic-output limitations of ``torch.nonzero``.
        Boolean masks and integer indices are not differentiable.

    Examples:
        >>> import torch
        >>> from oops import smooth_lddt_pair_indices
        >>> mask = torch.tensor([[False, True], [True, False]])
        >>> smooth_lddt_pair_indices(mask).tolist()
        [0, 1, 1, 0]
    """
    if lddt_mask.dtype != torch.bool or lddt_mask.ndim != 2 or lddt_mask.shape[0] != lddt_mask.shape[1]:
        raise ValueError(
            "lddt_mask must be a square boolean pair mask, "
            f"got dtype {lddt_mask.dtype} and shape {tuple(lddt_mask.shape)}"
        )
    return torch.nonzero(lddt_mask, as_tuple=False).reshape(-1)
