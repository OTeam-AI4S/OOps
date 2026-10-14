from __future__ import annotations

from dataclasses import replace

import torch

from oops.core import TargetProfile, TargetSelector

from ..schedule import AttentionTile, PhaseTile, RowTile, TileKey, TriangleAttentionPhase
from . import generic

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

_DENSE = {
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float32, 64): AttentionTile(16, 64, 2, 3),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float32, 128): AttentionTile(16, 64, 2, 2),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.bfloat16, 128): AttentionTile(16, 64, 2, 2, head_alignment=32),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float32, 32): AttentionTile(32, 64, 4, 3),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float32, 64): AttentionTile(32, 64, 4, 3),
    (TriangleAttentionPhase.FORWARD, torch.bfloat16, 32): AttentionTile(128, 64, 4, 3),
    (TriangleAttentionPhase.FORWARD, torch.bfloat16, 64): AttentionTile(64, 64, 4, 3),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.bfloat16, 64): AttentionTile(64, 32, 2, 2),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.bfloat16, 32): AttentionTile(128, 64, 8, 3),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.bfloat16, 64): AttentionTile(128, 64, 8, 3),
    (TriangleAttentionPhase.FORWARD, torch.float16, 32): AttentionTile(128, 64, 4, 3),
    (TriangleAttentionPhase.FORWARD, torch.float16, 64): AttentionTile(128, 64, 4, 3),
    (TriangleAttentionPhase.BACKWARD_QUERY, torch.float16, 32): AttentionTile(128, 32, 2, 2),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float16, 64): AttentionTile(64, 64, 4, 3),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float16, 32): AttentionTile(32, 128, 4, 3),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float16, 64): AttentionTile(128, 64, 8, 3),
}
_RAGGED: dict[TileKey, PhaseTile] = {
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float32, 128): AttentionTile(16, 64, 2, 2),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.bfloat16, 128): AttentionTile(16, 64, 2, 2, head_alignment=32),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float32, 128): AttentionTile(16, 32, 4, 2),
    (TriangleAttentionPhase.FORWARD, torch.bfloat16, 32): AttentionTile(64, 32, 2, 3, 4),
    (TriangleAttentionPhase.FORWARD, torch.bfloat16, 64): AttentionTile(64, 64, 4, 3, 2),
    (TriangleAttentionPhase.BACKWARD_PREPROCESS, torch.bfloat16, 32): RowTile(128, 4, 8),
    (TriangleAttentionPhase.BACKWARD_PREPROCESS, torch.bfloat16, 64): RowTile(64, 2, 8),
    (TriangleAttentionPhase.BACKWARD_QUERY, torch.bfloat16, 32): AttentionTile(128, 32, 2, 3, 8),
    (TriangleAttentionPhase.BACKWARD_QUERY, torch.bfloat16, 64): AttentionTile(32, 64, 2, 2, 4),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.bfloat16, 32): AttentionTile(32, 64, 2, 2, 4),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.bfloat16, 64): AttentionTile(32, 32, 2, 2, 4),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.bfloat16, 32): AttentionTile(32, 64, 4, 3, 2),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.bfloat16, 64): AttentionTile(32, 64, 4, 3, 2),
    (TriangleAttentionPhase.FORWARD, torch.float16, 32): AttentionTile(64, 32, 2, 3, 4),
    (TriangleAttentionPhase.FORWARD, torch.float16, 64): AttentionTile(64, 64, 4, 3, 4),
    (TriangleAttentionPhase.BACKWARD_PREPROCESS, torch.float16, 32): RowTile(128, 4, 8),
    (TriangleAttentionPhase.BACKWARD_PREPROCESS, torch.float16, 64): RowTile(128, 8, 4),
    (TriangleAttentionPhase.BACKWARD_QUERY, torch.float16, 32): AttentionTile(128, 32, 2, 3, 4),
    (TriangleAttentionPhase.BACKWARD_QUERY, torch.float16, 64): AttentionTile(32, 64, 2, 2, 4),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float16, 32): AttentionTile(32, 64, 2, 3, 8),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float16, 64): AttentionTile(32, 32, 2, 3, 4),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float16, 32): AttentionTile(32, 64, 4, 3, 2),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float16, 64): AttentionTile(32, 64, 4, 3, 2),
    (TriangleAttentionPhase.FORWARD, torch.float32, 32): AttentionTile(128, 32, 4, 3, 2),
    (TriangleAttentionPhase.FORWARD, torch.float32, 64): AttentionTile(64, 32, 4, 3, 8),
    (TriangleAttentionPhase.BACKWARD_PREPROCESS, torch.float32, 32): RowTile(128, 4, 8),
    (TriangleAttentionPhase.BACKWARD_PREPROCESS, torch.float32, 64): RowTile(64, 4, 4),
    (TriangleAttentionPhase.BACKWARD_QUERY, torch.float32, 32): AttentionTile(128, 32, 4, 1, 2),
    (TriangleAttentionPhase.BACKWARD_QUERY, torch.float32, 64): AttentionTile(32, 64, 4, 1, 4),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float32, 32): AttentionTile(32, 32, 2, 1, 4),
    (TriangleAttentionPhase.BACKWARD_KEY_VALUE, torch.float32, 64): AttentionTile(32, 32, 4, 1, 8),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float32, 32): AttentionTile(64, 32, 4, 2, 2),
    (TriangleAttentionPhase.BACKWARD_SHARED_BIAS, torch.float32, 64): AttentionTile(32, 64, 4, 2, 2),
}

TABLE = replace(generic.TABLE, dense=_DENSE, ragged=_RAGGED)

TARGETS = (TargetProfile(_SM80, TABLE),)
