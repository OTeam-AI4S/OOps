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

SUPPORTED_LAYOUT = (
    "requires contiguous CUDA float16/bfloat16/float32/float64 values with a non-empty "
    "edge axis, output-segment and flattened feature grids within CUDA launch limits, and a contiguous CUDA "
    "int32/int64 row-offset vector"
)


def num_edges(values: Tensor, name: str) -> int:
    if values.dim() < 1:
        raise ValueError(f"{name} must have shape (E, ...)")
    return values.shape[0]


def layout_context(values: Tensor, offsets: Tensor) -> dict[str, object]:
    return {
        "dtype": values.dtype,
        "device": values.device,
        "shape": tuple(values.shape),
        "contiguous": values.is_contiguous(),
        "row_offsets_dtype": offsets.dtype,
    }
