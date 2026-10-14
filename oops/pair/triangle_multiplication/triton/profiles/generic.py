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

import torch

from oops.runtime import HALF_DTYPES

from ..schedule import (
    GEMM_PHASES,
    SEGMENTED_PHASES,
    ColumnSumTile,
    GemmTile,
    LayerNormBackwardTile,
    LayerNormForwardTile,
    SegmentedGemmTile,
    TileTable,
)

_HALF_GEMM = GemmTile(64, 64, 32, 4, 3)
_FLOAT32_GEMM = GemmTile(32, 32, 32, 4, 2)
_HALF_LAYER_NORM_BACKWARD = LayerNormBackwardTile(32, 32, 2, 4, 1)
_FLOAT32_LAYER_NORM_BACKWARD = LayerNormBackwardTile(16, 32, 2, 4, 1)
_SEGMENTED = SegmentedGemmTile(64, 64, 32, 2, 4, 3)

TABLE = TileTable(
    gemm_defaults={
        **{(phase, dtype): _HALF_GEMM for phase in GEMM_PHASES for dtype in HALF_DTYPES},
        **{(phase, torch.float32): _FLOAT32_GEMM for phase in GEMM_PHASES},
    },
    layer_norm_backward_defaults={
        **{dtype: _HALF_LAYER_NORM_BACKWARD for dtype in HALF_DTYPES},
        torch.float32: _FLOAT32_LAYER_NORM_BACKWARD,
    },
    segmented_defaults={
        (phase, dtype): _SEGMENTED for phase in SEGMENTED_PHASES for dtype in (*HALF_DTYPES, torch.float32)
    },
    layer_norm_forward=LayerNormForwardTile(max_tile_values=16384, min_rows=8, max_rows=64, num_warps=4),
    column_sum=ColumnSumTile(block_rows=256, block_columns=64, max_partitions=256, num_warps=8),
)
