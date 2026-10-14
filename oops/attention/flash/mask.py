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

MASK_NONE = 0
MASK_KEY = 1
MASK_PAIR = 2


def pair_mask_kind(query: Tensor, mask: Tensor | None) -> int:

    if mask is None:
        return MASK_NONE
    if mask.dim() == query.dim() - 2:
        return MASK_KEY
    return MASK_PAIR if mask.dim() == query.dim() - 1 else MASK_NONE
