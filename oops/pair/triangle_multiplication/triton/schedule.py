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

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

import torch

from oops.core import TargetDescriptor, TargetProfile, select_target_profile, validate_positive_parameters
from oops.core._math import _next_power_of_two
from oops.runtime import dot_tile_shared_bytes, halved_dot_blocks

MAX_FUSED_WIDTH = 1024


class TriangleMultiplicationPhase(Enum):

    PROJECTION_FORWARD = "projection_forward"
    PROJECTION_BACKWARD = "projection_backward"
    OUTPUT_FORWARD = "output_forward"
    OUTPUT_BACKWARD = "output_backward"
    LAYER_NORM_BACKWARD = "layer_norm_backward"
    CONTRACTION_FORWARD = "contraction_forward"
    CONTRACTION_BACKWARD = "contraction_backward"


GEMM_PHASES = (
    TriangleMultiplicationPhase.PROJECTION_FORWARD,
    TriangleMultiplicationPhase.PROJECTION_BACKWARD,
    TriangleMultiplicationPhase.OUTPUT_FORWARD,
    TriangleMultiplicationPhase.OUTPUT_BACKWARD,
)


@dataclass(frozen=True, slots=True)
class GemmTile:

    block_rows: int
    block_channels: int
    block_reduction: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        validate_positive_parameters(num_warps=self.num_warps, num_stages=self.num_stages)
        _require_dot_extent("block_rows", self.block_rows)
        _require_dot_extent("block_channels", self.block_channels)
        _require_dot_extent("block_reduction", self.block_reduction)

    def fit(self, element_size: int, shared_memory_bytes: int) -> GemmTile | None:

        for block_rows in halved_dot_blocks(self.block_rows):
            staged = dot_tile_shared_bytes(
                block_rows,
                self.block_reduction,
                self.block_channels,
                element_size,
                2,
                buffers=self.num_stages,
            )
            if staged <= shared_memory_bytes:
                return self if block_rows == self.block_rows else replace(self, block_rows=block_rows)
        return None


@dataclass(frozen=True, slots=True)
class LayerNormBackwardTile:

    block_rows: int
    block_channels: int
    programs_per_multiprocessor: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        validate_positive_parameters(
            programs_per_multiprocessor=self.programs_per_multiprocessor,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        _require_dot_extent("block_rows", self.block_rows)
        _require_dot_extent("block_channels", self.block_channels)

    def fit(self, block_hidden: int, element_size: int, shared_memory_bytes: int) -> LayerNormBackwardTile | None:

        for block_rows in halved_dot_blocks(self.block_rows):
            staged = (
                element_size * self.num_stages * (block_rows * self.block_channels + self.block_channels * block_hidden)
            )
            if staged <= shared_memory_bytes:
                return self if block_rows == self.block_rows else replace(self, block_rows=block_rows)
        return None


@dataclass(frozen=True, slots=True)
class SegmentedGemmTile:

    block_rows: int
    block_columns: int
    block_reduction: int
    programs_per_multiprocessor: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        validate_positive_parameters(
            programs_per_multiprocessor=self.programs_per_multiprocessor,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        _require_dot_extent("block_rows", self.block_rows)
        _require_dot_extent("block_columns", self.block_columns)
        _require_dot_extent("block_reduction", self.block_reduction)

    def fit(self, element_size: int, shared_memory_bytes: int) -> SegmentedGemmTile | None:

        for block_rows in halved_dot_blocks(self.block_rows):
            staged = dot_tile_shared_bytes(
                block_rows,
                self.block_reduction,
                self.block_columns,
                element_size,
                1,
                buffers=self.num_stages,
            )
            if staged <= shared_memory_bytes:
                return self if block_rows == self.block_rows else replace(self, block_rows=block_rows)
        return None


SEGMENTED_PHASES = (TriangleMultiplicationPhase.CONTRACTION_FORWARD, TriangleMultiplicationPhase.CONTRACTION_BACKWARD)


@dataclass(frozen=True, slots=True)
class LayerNormForwardTile:

    max_tile_values: int
    min_rows: int
    max_rows: int
    num_warps: int

    def __post_init__(self) -> None:
        validate_positive_parameters(
            max_tile_values=self.max_tile_values,
            min_rows=self.min_rows,
            max_rows=self.max_rows,
            num_warps=self.num_warps,
        )

    def block_rows(self, hidden_size: int) -> int:
        return max(self.min_rows, min(self.max_rows, self.max_tile_values // full_width(hidden_size)))


@dataclass(frozen=True, slots=True)
class ColumnSumTile:

    block_rows: int
    block_columns: int
    max_partitions: int
    num_warps: int

    def __post_init__(self) -> None:
        validate_positive_parameters(
            block_rows=self.block_rows,
            block_columns=self.block_columns,
            max_partitions=self.max_partitions,
            num_warps=self.num_warps,
        )


GemmKey = tuple[TriangleMultiplicationPhase, torch.dtype, int, int]


@dataclass(frozen=True, slots=True)
class TileTable:

    gemm_defaults: Mapping[tuple[TriangleMultiplicationPhase, torch.dtype], GemmTile]
    layer_norm_backward_defaults: Mapping[torch.dtype, LayerNormBackwardTile]
    segmented_defaults: Mapping[tuple[TriangleMultiplicationPhase, torch.dtype], SegmentedGemmTile]
    layer_norm_forward: LayerNormForwardTile
    column_sum: ColumnSumTile
    gemm: Mapping[GemmKey, GemmTile] = field(default_factory=dict)
    layer_norm_backward: Mapping[tuple[torch.dtype, int, int], LayerNormBackwardTile] = field(default_factory=dict)

    def gemm_tile(
        self,
        phase: TriangleMultiplicationPhase,
        dtype: torch.dtype,
        pair_size: int,
        hidden_size: int,
        shared_memory_bytes: int,
    ) -> GemmTile | None:
        key = (phase, dtype, full_width(pair_size), full_width(hidden_size))
        tile = self.gemm.get(key) or self.gemm_defaults[(phase, dtype)]
        return tile.fit(dtype.itemsize, shared_memory_bytes)

    def layer_norm_backward_tile(
        self,
        dtype: torch.dtype,
        pair_size: int,
        hidden_size: int,
        shared_memory_bytes: int,
    ) -> LayerNormBackwardTile | None:
        key = (dtype, full_width(pair_size), full_width(hidden_size))
        tile = self.layer_norm_backward.get(key) or self.layer_norm_backward_defaults[dtype]
        return tile.fit(full_width(hidden_size), dtype.itemsize, shared_memory_bytes)

    def segmented_tile(
        self,
        phase: TriangleMultiplicationPhase,
        dtype: torch.dtype,
        shared_memory_bytes: int,
    ) -> SegmentedGemmTile | None:
        return self.segmented_defaults[(phase, dtype)].fit(dtype.itemsize, shared_memory_bytes)


def select_tile_table(
    generic: TileTable,
    targets: tuple[TargetProfile[TileTable], ...],
    target: TargetDescriptor | None,
) -> TileTable:

    return select_target_profile(generic, target=target, profiles=targets)


def full_width(channels: int) -> int:

    return max(16, _next_power_of_two(channels))


def _require_dot_extent(name: str, value: int) -> None:
    if value < 16 or value & (value - 1):
        raise ValueError(f"{name} must be a power of two of at least 16 for tl.dot; got {value}")
