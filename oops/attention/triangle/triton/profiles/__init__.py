from __future__ import annotations

from oops.core import TargetDescriptor

from ..schedule import TileTable, select_tile_table
from . import generic as _generic
from . import sm80 as _sm80

TARGETS = _sm80.TARGETS


def tile_table(target: TargetDescriptor | None) -> TileTable:

    return select_tile_table(_generic.TABLE, TARGETS, target)


__all__ = ["TARGETS", "tile_table"]
