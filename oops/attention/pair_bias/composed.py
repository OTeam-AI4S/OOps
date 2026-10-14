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
