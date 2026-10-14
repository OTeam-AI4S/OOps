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

from torch import Tensor

from oops.runtime import require_no_forward_mode, require_same_device, unsupported

from .triton import supports_distogram_cross_entropy, triton_distogram_cross_entropy


def distogram_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    token_mask: Tensor | None = None,
    pair_mask: Tensor | None = None,
    eps: float = 1e-5,
    reduction: str | None = "mean",
) -> Tensor:
    r"""Reduce token-masked distogram cross entropy over off-diagonal pairs.

    Args:
        logits: CUDA float16, bfloat16, or float32 tensor ``(B, N, N, C)`` with
            ``1 <= C <= 256`` bins.
        labels: Integer bin indices ``(B, N, N)`` on the logits device, with values in ``[0, C)``.
        token_mask: Required boolean or floating tensor ``(B, N)``. Pair weights are the
            product of the two token weights, with the diagonal excluded.
        pair_mask: Must be None; precomputed pair-mask input is unsupported.
        eps: Stabilizer added to each batch element's pair-weight denominator.
        reduction: ``"mean"`` or ``"sum"`` reduce the per-batch losses; None returns them.

    Returns:
        A scalar for mean/sum reduction, or ``(B,)`` losses for None.

    Raises:
        ValueError: Devices disagree, reduction is invalid, or both mask forms are supplied.
        OOpsUnsupportedError: The token mask is absent, a pair mask or probability labels
            are supplied, the device, dtype, shape, or layout is unsupported, or a forward-mode tangent is present.

    Notes:
        Floating token weights contribute to both the weighted losses and their
        normalizing denominator.

    Examples:
        >>> import torch
        >>> from oops import distogram_cross_entropy
        >>> if torch.cuda.is_available():
        ...     logits = torch.zeros(1, 3, 3, 4, device="cuda")
        ...     labels = torch.zeros(1, 3, 3, dtype=torch.long, device="cuda")
        ...     token_mask = torch.ones(1, 3, device="cuda")
        ...     loss = distogram_cross_entropy(logits, labels, token_mask=token_mask)
        ...     assert loss.shape == torch.Size([]) and loss.dtype == torch.float32
    """

    require_no_forward_mode("distogram_cross_entropy", logits, token_mask)
    require_same_device("distogram_cross_entropy", logits, labels, token_mask, pair_mask)

    if pair_mask is not None and token_mask is not None:
        raise ValueError("pass either token_mask or pair_mask, not both")
    if reduction not in ("mean", "sum", None):
        raise ValueError("reduction must be 'mean', 'sum', or None")

    if token_mask is None:
        raise unsupported(
            "distogram_cross_entropy",
            "requires a token mask of shape (B, N); the kernel builds the pair mask itself",
            pair_mask_shape=None if pair_mask is None else tuple(pair_mask.shape),
        )
    if not supports_distogram_cross_entropy(logits, labels, token_mask, reduction):
        raise unsupported(
            "distogram_cross_entropy",
            "requires CUDA float16/bfloat16/float32 (B, N, N, bins) logits with integer labels, "
            "a (B, N) token mask, at most 256 bins, and launchable row and batch grids",
            logits_dtype=logits.dtype,
            labels_dtype=labels.dtype,
            token_mask_dtype=token_mask.dtype,
            device=logits.device,
            logits_shape=tuple(logits.shape),
            labels_shape=tuple(labels.shape),
            token_mask_shape=tuple(token_mask.shape),
        )
    return triton_distogram_cross_entropy(logits, labels, token_mask, eps=eps, reduction=reduction)
