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

r"""
Reference implementation for outer-product mean.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from oops.runtime import HALF_DTYPES


def outer_product_mean_reference(
    msa: Tensor,
    msa_mask: Tensor,
    *,
    msa_norm_weight: Tensor,
    msa_norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    output_weight: Tensor,
    left_bias: Tensor | None = None,
    right_bias: Tensor | None = None,
    output_bias: Tensor | None = None,
    outer_dtype: torch.dtype | None = None,
    divide_before_projection: bool = True,
    denominator_eps: float = 0.0,
    clamp_denominator: bool = False,
    msa_norm_eps: float = 1e-5,
) -> Tensor:
    r"""
    Return the direct PyTorch outer-product mean used as a correctness oracle.
    """

    activation_dtype = left_weight.dtype
    if msa.is_cuda and torch.is_autocast_enabled("cuda"):
        autocast_dtype = torch.get_autocast_dtype("cuda")
        if autocast_dtype in HALF_DTYPES:
            activation_dtype = autocast_dtype
    mask = msa_mask.to(activation_dtype)
    hidden = msa.double()
    mean = hidden.mean(dim=-1, keepdim=True)
    centered = hidden - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = (
        msa_norm_weight.double() * torch.rsqrt(variance + msa_norm_eps) * centered + msa_norm_bias.double()
    ).to(activation_dtype)
    with torch.autocast("cuda", enabled=False):
        left = F.linear(
            normalized,
            left_weight.to(activation_dtype),
            None if left_bias is None else left_bias.to(activation_dtype),
        ).to(activation_dtype)
        right = F.linear(
            normalized,
            right_weight.to(activation_dtype),
            None if right_bias is None else right_bias.to(activation_dtype),
        ).to(activation_dtype)
    left = left * mask.unsqueeze(-1)
    right = right * mask.unsqueeze(-1)
    if outer_dtype is not None:
        left = left.to(outer_dtype)
        right = right.to(outer_dtype)
    accumulation_dtype = torch.float64 if activation_dtype is torch.float64 else torch.float32
    count_mask = msa_mask.to(accumulation_dtype)
    with torch.autocast("cuda", enabled=False):
        count = torch.einsum("bsi,bsj->bij", count_mask, count_mask).unsqueeze(-1)
    with torch.autocast("cuda", enabled=False):
        outer = torch.einsum("bsic,bsjd->bijcd", left, right).reshape(
            msa.shape[0],
            msa.shape[2],
            msa.shape[2],
            left.shape[-1] * right.shape[-1],
        )
        output = F.linear(outer.to(activation_dtype), output_weight.to(activation_dtype), None)
    output_float = output.to(accumulation_dtype)
    bias_float = None if output_bias is None else output_bias.to(activation_dtype).to(accumulation_dtype)
    if divide_before_projection:
        output_float = output_float / count.clamp_min(1)
        if bias_float is not None:
            output_float = output_float + bias_float
    else:
        if bias_float is not None:
            output_float = output_float + bias_float
        denominator = count + denominator_eps
        if clamp_denominator:
            denominator = denominator.clamp_min(1)
        output_float = output_float / denominator
    return output_float.to(activation_dtype)
