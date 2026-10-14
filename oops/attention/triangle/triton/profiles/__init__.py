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

from oops.core import TargetDescriptor

from ..schedule import TileTable, select_tile_table
from . import generic as _generic
from . import sm80 as _sm80

TARGETS = _sm80.TARGETS


def tile_table(target: TargetDescriptor | None) -> TileTable:

    return select_tile_table(_generic.TABLE, TARGETS, target)


__all__ = ["TARGETS", "tile_table"]
