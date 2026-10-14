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


def _validate_token_offsets(offsets: Tensor, total: int) -> list[int]:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError("pair_mask_from_token_mask offsets must be one-dimensional CPU int64")
    if offsets.numel() < 2:
        raise ValueError("pair_mask_from_token_mask offsets must describe a non-empty wrapper batch")
    boundaries = offsets.tolist()
    if boundaries[0] != 0 or any(right < left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("pair_mask_from_token_mask offsets must be monotonic and start at zero")
    if boundaries[-1] != total:
        raise ValueError("pair_mask_from_token_mask offsets do not cover packed token values")
    return boundaries


def _validate_msa_layout(
    element_sizes: Tensor, input_offsets: Tensor, input_total: int
) -> tuple[list[list[int]], list[int]]:
    if (
        element_sizes.device.type != "cpu"
        or element_sizes.dtype is not torch.int64
        or element_sizes.dim() != 2
        or element_sizes.shape[1] != 2
    ):
        raise ValueError("token_mask_from_msa_mask element_sizes must be (batch, 2) CPU int64")
    samples = element_sizes.shape[0]
    if samples == 0:
        raise ValueError("token_mask_from_msa_mask requires a non-empty wrapper batch")
    sizes = element_sizes.tolist()
    if any(rows < 0 or columns < 0 for rows, columns in sizes):
        raise ValueError("token_mask_from_msa_mask element sizes must be non-negative")
    if input_offsets.device.type != "cpu" or input_offsets.dtype is not torch.int64:
        raise ValueError("token_mask_from_msa_mask offsets must be CPU int64")
    if input_offsets.dim() != 1 or input_offsets.shape[0] != samples + 1:
        raise ValueError("token_mask_from_msa_mask offsets must contain one boundary per sample")
    offsets = input_offsets.tolist()
    if offsets[0] != 0 or any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("token_mask_from_msa_mask offsets must be monotonic and start at zero")
    if any(rows * columns > (1 << 63) - 1 for rows, columns in sizes):
        raise ValueError("token_mask_from_msa_mask element area overflows int64 packed storage")
    if (
        any(right - left != rows * columns for (rows, columns), left, right in zip(sizes, offsets, offsets[1:]))
        or offsets[-1] != input_total
    ):
        raise ValueError("token_mask_from_msa_mask offsets do not match rectangular element sizes")
    return sizes, offsets
