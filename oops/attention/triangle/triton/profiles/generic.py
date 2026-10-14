from __future__ import annotations

import torch

from oops.runtime import HALF_DTYPES

from ..schedule import (
    AttentionTile,
    PhaseTile,
    RowTile,
    SharedBiasSplit,
    TileKey,
    TileTable,
    TriangleAttentionPhase,
)

_DTYPES = (*HALF_DTYPES, torch.float32)
_WIDTHS = (16, 32, 64, 128, 256)


def _staged(tile: AttentionTile, block_channel: int, phase: TriangleAttentionPhase) -> AttentionTile:
    stages = 3 if block_channel <= 64 else 2 if block_channel <= 128 else 1
    warps = max(tile.num_warps, 4) if phase is TriangleAttentionPhase.BACKWARD_KEY_VALUE else tile.num_warps
    return AttentionTile(tile.block_query, tile.block_key, warps, stages)


_TILES = {
    TriangleAttentionPhase.FORWARD: AttentionTile(128, 32, 4, 3),
    TriangleAttentionPhase.BACKWARD_QUERY: AttentionTile(128, 32, 4, 3),
    TriangleAttentionPhase.BACKWARD_KEY_VALUE: AttentionTile(32, 64, 2, 3),
    TriangleAttentionPhase.BACKWARD_SHARED_BIAS: AttentionTile(64, 64, 4, 3),
}
_PREPROCESS = RowTile(128, 4)

_DEFAULTS: dict[TileKey, PhaseTile] = {
    (phase, dtype, width): _staged(tile, width, phase)
    for phase, tile in _TILES.items()
    for dtype in _DTYPES
    for width in _WIDTHS
}
_DEFAULTS.update(
    {(TriangleAttentionPhase.BACKWARD_PREPROCESS, dtype, width): _PREPROCESS for dtype in _DTYPES for width in _WIDTHS}
)

_SHARED_BIAS_SPLIT = SharedBiasSplit(saturating_tiles=256, max_parts=16, min_rows_per_part=4, workspace_bytes=16 << 20)

TABLE = TileTable(defaults=_DEFAULTS, shared_bias_split=_SHARED_BIAS_SPLIT)
