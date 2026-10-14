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


def _check_gate_output(attention: Tensor, single: Tensor, gate_weight: Tensor, output_weight: Tensor) -> None:
    if attention.dim() not in (3, 4, 5):
        raise ValueError(
            "attention must have layout (num_heads, query_length, head_size) under optional batch and sample axes"
        )
    if single.dim() != attention.dim() - 1:
        raise ValueError("single must have layout (..., query_length, single_channels)")
    if gate_weight.dim() != 2 or output_weight.dim() != 2:
        raise ValueError("gate_weight and output_weight must be rank-2 matrices")

    *prefix, num_heads, query_length, head_size = attention.shape
    expected_single_prefix = (*prefix, query_length)
    gate_channels = num_heads * head_size
    if tuple(single.shape[:-1]) != expected_single_prefix:
        raise ValueError("single must share the attention leading axes and its query_length")
    if gate_weight.shape != (gate_channels, single.shape[-1]):
        raise ValueError("gate_weight must have shape (num_heads * head_size, single_channels)")
    if output_weight.shape[1] != gate_channels:
        raise ValueError("output_weight input channels must equal num_heads * head_size")
