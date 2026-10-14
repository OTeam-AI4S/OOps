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

from .mask import (
    MASK_KEY,
    MASK_NONE,
    MASK_PAIR,
    pair_mask_kind,
)
from .qkv import (
    check_head_major_qkv,
    degenerate_shape,
    degenerate_strides,
    optional_unit_batch,
)
from .spec import DenseAttentionSpec, run_dense_attention

__all__ = [
    "DenseAttentionSpec",
    "MASK_KEY",
    "MASK_NONE",
    "MASK_PAIR",
    "check_head_major_qkv",
    "degenerate_shape",
    "degenerate_strides",
    "optional_unit_batch",
    "pair_mask_kind",
    "run_dense_attention",
]
