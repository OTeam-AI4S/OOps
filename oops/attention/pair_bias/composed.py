from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def native_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    scale: float | None,
) -> Tensor | None:
    if bias is None:
        return None
    additive = bias
    if mask is not None:
        keep = mask.unsqueeze(-3) if mask.dim() == query.dim() - 1 else mask.unsqueeze(-2).unsqueeze(-3)
        additive = additive.masked_fill(~keep, float("-inf"))
    if additive.shape[:-3] != query.shape[:-3]:
        additive = additive.expand(*query.shape[:-3], *additive.shape[-3:])
    additive = additive.flatten(0, -4)
    autocast = torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled("cuda") else nullcontext()
    with autocast, sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = F.scaled_dot_product_attention(
            query.flatten(0, -4),
            key.flatten(0, -4),
            value.flatten(0, -4),
            attn_mask=additive,
            scale=scale,
        )
    return output if output.shape == query.shape else output.reshape_as(query)
