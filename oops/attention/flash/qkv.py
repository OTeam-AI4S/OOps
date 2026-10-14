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


def degenerate_shape(tensor: Tensor, rank: int) -> tuple[int, ...]:

    return (1,) * (rank - tensor.dim()) + tuple(tensor.shape)


def degenerate_strides(tensor: Tensor, rank: int) -> tuple[int, ...]:
    r"""Omitted leading axes have extent one and stride zero."""

    return (0,) * (rank - tensor.dim()) + tuple(tensor.stride())


def check_head_major_qkv(query: Tensor, key: Tensor, value: Tensor) -> None:

    if query.dim() < 3:
        raise ValueError("query must have layout (num_heads, query_length, head_size) under an optional prefix")
    if key.dim() != query.dim() or value.dim() != query.dim():
        raise ValueError("query, key, and value must have the same rank")
    if key.shape[:-2] != query.shape[:-2] or value.shape[:-2] != key.shape[:-2]:
        raise ValueError("query, key, and value must share prefix and num_heads dimensions")
    if key.shape[-1] != query.shape[-1]:
        raise ValueError("query and key head_size dimensions must match")
    if value.shape[-1] != query.shape[-1]:
        raise ValueError("query and value head_size dimensions must match")
    if value.shape[-2] != key.shape[-2]:
        raise ValueError("key and value key_len dimensions must match")


def optional_unit_batch(tensor: Tensor | None) -> Tensor | None:

    return None if tensor is None else tensor.unsqueeze(0)
