from __future__ import annotations

from dataclasses import replace

import torch

from oops.core import TargetProfile, TargetSelector

from ..schedule import GemmTile, LayerNormBackwardTile, TileTable, TriangleMultiplicationPhase
from . import generic

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

TABLE = TileTable(
    gemm_defaults=generic.TABLE.gemm_defaults,
    layer_norm_backward_defaults=generic.TABLE.layer_norm_backward_defaults,
    segmented_defaults={
        key: replace(tile, programs_per_multiprocessor=2) for key, tile in generic.TABLE.segmented_defaults.items()
    },
    layer_norm_forward=generic.TABLE.layer_norm_forward,
    column_sum=generic.TABLE.column_sum,
    gemm={
        (TriangleMultiplicationPhase.PROJECTION_FORWARD, torch.float32, 128, 128): GemmTile(32, 32, 32, 2, 2),
        (TriangleMultiplicationPhase.PROJECTION_BACKWARD, torch.float32, 128, 128): GemmTile(32, 32, 32, 2, 2),
        (TriangleMultiplicationPhase.OUTPUT_FORWARD, torch.float32, 128, 128): GemmTile(32, 32, 32, 2, 2),
        (TriangleMultiplicationPhase.OUTPUT_BACKWARD, torch.float32, 128, 128): GemmTile(32, 32, 32, 2, 2),
        (TriangleMultiplicationPhase.PROJECTION_FORWARD, torch.bfloat16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.PROJECTION_FORWARD, torch.bfloat16, 128, 128): GemmTile(128, 64, 32, 4, 3),
        (TriangleMultiplicationPhase.PROJECTION_FORWARD, torch.bfloat16, 256, 256): GemmTile(128, 64, 32, 4, 3),
        (TriangleMultiplicationPhase.PROJECTION_BACKWARD, torch.bfloat16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.PROJECTION_BACKWARD, torch.bfloat16, 128, 128): GemmTile(64, 128, 32, 4, 3),
        (TriangleMultiplicationPhase.PROJECTION_BACKWARD, torch.bfloat16, 256, 256): GemmTile(64, 128, 64, 4, 3),
        (TriangleMultiplicationPhase.OUTPUT_FORWARD, torch.bfloat16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.OUTPUT_FORWARD, torch.bfloat16, 128, 128): GemmTile(64, 128, 64, 8, 3),
        (TriangleMultiplicationPhase.OUTPUT_FORWARD, torch.bfloat16, 256, 256): GemmTile(64, 128, 32, 4, 3),
        (TriangleMultiplicationPhase.OUTPUT_BACKWARD, torch.bfloat16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.OUTPUT_BACKWARD, torch.bfloat16, 128, 128): GemmTile(64, 128, 64, 8, 3),
        (TriangleMultiplicationPhase.OUTPUT_BACKWARD, torch.bfloat16, 256, 256): GemmTile(64, 128, 64, 8, 3),
        (TriangleMultiplicationPhase.PROJECTION_FORWARD, torch.float16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.PROJECTION_FORWARD, torch.float16, 128, 128): GemmTile(128, 64, 32, 4, 3),
        (TriangleMultiplicationPhase.PROJECTION_FORWARD, torch.float16, 256, 256): GemmTile(128, 64, 32, 4, 3),
        (TriangleMultiplicationPhase.PROJECTION_BACKWARD, torch.float16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.PROJECTION_BACKWARD, torch.float16, 128, 128): GemmTile(64, 128, 64, 4, 3),
        (TriangleMultiplicationPhase.PROJECTION_BACKWARD, torch.float16, 256, 256): GemmTile(64, 128, 32, 4, 3),
        (TriangleMultiplicationPhase.OUTPUT_FORWARD, torch.float16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.OUTPUT_FORWARD, torch.float16, 128, 128): GemmTile(128, 32, 32, 4, 3),
        (TriangleMultiplicationPhase.OUTPUT_FORWARD, torch.float16, 256, 256): GemmTile(64, 128, 32, 4, 3),
        (TriangleMultiplicationPhase.OUTPUT_BACKWARD, torch.float16, 64, 64): GemmTile(64, 64, 64, 4, 1),
        (TriangleMultiplicationPhase.OUTPUT_BACKWARD, torch.float16, 256, 256): GemmTile(64, 128, 64, 8, 3),
    },
    layer_norm_backward={
        (torch.bfloat16, 64, 64): LayerNormBackwardTile(64, 64, 2, 4, 1),
        (torch.bfloat16, 128, 128): LayerNormBackwardTile(64, 128, 4, 4, 1),
        (torch.bfloat16, 256, 256): LayerNormBackwardTile(32, 64, 2, 4, 2),
        (torch.float16, 64, 64): LayerNormBackwardTile(64, 64, 2, 4, 1),
        (torch.float16, 128, 128): LayerNormBackwardTile(32, 128, 4, 4, 1),
        (torch.float16, 256, 256): LayerNormBackwardTile(32, 64, 2, 4, 2),
    },
)

TARGETS = (TargetProfile(_SM80, TABLE),)
