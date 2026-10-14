from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import unsupported

from .triton import packed_msa_pair_mask


def pair_mask_from_msa_mask(msa_mask: Tensor | NestedTensor) -> Tensor | NestedTensor:
    r"""Reduce MSA validity and form the outer token conjunction.

    Args:
        msa_mask: Dense validity values shaped ``(..., M, N)``, or a
            batch-first NestedTensor with canonical ``(M_i, N_i)`` elements
            in row-major packed order. Nonzero values count as true.

    Returns:
        A bool tensor shaped ``(..., N, N)``, or a NestedTensor containing
        explicit packed ``(N_i, N_i)`` elements. The result equals
        ``pair_mask_from_token_mask(token_mask_from_msa_mask(msa_mask))``.

    Raises:
        ValueError: The dense rank or nested logical topology is malformed.
        OOpsUnsupportedError: Nested storage is not supported CUDA bool,
            integer, or floating-point storage, its launch extent is
            unsupported, or the wrapper batch is empty.

    Notes:
        Dense inputs accept any device and dtype supported by conversion to
        bool. Nested inputs require CUDA storage and preserve sample topology.
        Empty MSA axes produce false output; empty token axes remain empty.
        Both representations support full-graph compilation. Outputs are
        non-differentiable.

    Examples:
        >>> import torch
        >>> from oops import pair_mask_from_msa_mask
        >>> mask = torch.tensor([[1, 0], [0, 0]])
        >>> pair_mask_from_msa_mask(mask).tolist()
        [[True, False], [False, False]]
    """

    if not isinstance(msa_mask, NestedTensor):
        if msa_mask.dim() < 2:
            raise ValueError("pair_mask_from_msa_mask msa_mask must have at least two dimensions")
        token_mask = msa_mask.bool().any(dim=-2)
        return token_mask[..., :, None] & token_mask[..., None, :]
    if (
        not msa_mask.batch_first
        or msa_mask.dim() != 3
        or msa_mask.ragged_dims != (0, 1)
        or msa_mask.packed_dim_order != (0, 1)
    ):
        raise ValueError("pair_mask_from_msa_mask nested elements require canonical (M_i, N_i) packing")
    if len(msa_mask) == 0:
        raise unsupported(
            "pair_mask_from_msa_mask",
            "does not serve an empty NestedTensor wrapper batch without retained element rank",
        )
    packed_values = msa_mask.concat
    if packed_values.is_complex():
        raise unsupported(
            "pair_mask_from_msa_mask",
            "requires bool, integer, or floating-point packed values",
            dtype=packed_values.dtype,
        )
    values = packed_values.bool()
    if values.dim() != 1 or not values.is_contiguous():
        raise unsupported(
            "pair_mask_from_msa_mask",
            "requires contiguous one-dimensional packed MSA values",
            packed_shape=tuple(values.shape),
        )
    if not values.is_cuda:
        raise unsupported("pair_mask_from_msa_mask", "requires CUDA packed values", device=values.device)

    sizes = msa_mask.element_sizes()
    output_values = packed_msa_pair_mask(values, sizes, msa_mask.packed_offsets(dtype=torch.int64))
    return msa_mask.packed_with_square_lengths(output_values, sizes[:, 1])
