from __future__ import annotations

from torch import Tensor

MASK_NONE = 0
MASK_KEY = 1
MASK_PAIR = 2


def pair_mask_kind(query: Tensor, mask: Tensor | None) -> int:

    if mask is None:
        return MASK_NONE
    if mask.dim() == query.dim() - 2:
        return MASK_KEY
    return MASK_PAIR if mask.dim() == query.dim() - 1 else MASK_NONE
