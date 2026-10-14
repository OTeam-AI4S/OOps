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

"""Independent PyTorch pair-bias attention equation shared by attention tests."""

from __future__ import annotations

import torch


def pair_bias_attention_reference(query, key, value, mask=None, pair_bias=None, scale=None):
    with torch.autocast("cuda", enabled=False):
        scale = query.shape[-1] ** -0.5 if scale is None else scale
        scores = (query.float() @ key.float().transpose(-1, -2)) * scale
        if pair_bias is not None:
            scores = scores + pair_bias.float()
        keep = None
        if mask is not None:
            keep = mask.bool()
            if keep.dim() == query.dim() - 2:
                keep = keep.unsqueeze(-2).unsqueeze(-2)
            else:
                while keep.dim() < scores.dim():
                    keep = keep.unsqueeze(-3)
            scores = scores.masked_fill(~keep, float("-inf"))
            scores = torch.where(keep.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
        probabilities = torch.softmax(scores, dim=-1)
        if keep is not None:
            probabilities = probabilities.masked_fill(~keep, 0.0)
        return probabilities @ value.float()
