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
from oops.runtime import CUDA_MAX_GRID_X, MIN_DOT_BLOCK, dot_tile_shared_bytes, halved_dot_blocks

MAX_HEAD_SIZE = 256
_OPERAND_BUDGET_BYTES = 16384


class TriangleAttentionPhase(Enum):

    FORWARD = "forward"
    INFERENCE = "inference"
    BACKWARD_PREPROCESS = "backward_preprocess"
    BACKWARD_QUERY = "backward_query"
    BACKWARD_KEY_VALUE = "backward_key_value"
    BACKWARD_SHARED_BIAS = "backward_shared_bias"


@dataclass(frozen=True, slots=True)
class AttentionTile:

    block_query: int
    block_key: int
    num_warps: int
    num_stages: int
    programs_per_multiprocessor: int = 1
    head_alignment: int = 1

    def __post_init__(self) -> None:
        validate_positive_parameters(
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            programs_per_multiprocessor=self.programs_per_multiprocessor,
            head_alignment=self.head_alignment,
        )
        _require_dot_extent("block_query", self.block_query)
        _require_dot_extent("block_key", self.block_key)

    def fit(self, block_channel: int, element_size: int, shared_memory_bytes: int) -> AttentionTile | None:

        bound = _operand_rows(block_channel, element_size)
        block_key = min(self.block_key, bound)
        for block_query in halved_dot_blocks(min(self.block_query, bound)):
            staged = _staged_bytes(block_query, block_key, block_channel, element_size, self.num_stages)
            if staged <= shared_memory_bytes:
                return (
                    self
                    if block_query == self.block_query and block_key == self.block_key
                    else replace(self, block_query=block_query, block_key=block_key)
                )
        return None


@dataclass(frozen=True, slots=True)
class RowTile:

    block_rows: int
    num_warps: int
    programs_per_multiprocessor: int = 1

    def __post_init__(self) -> None:
        validate_positive_parameters(
            num_warps=self.num_warps,
            programs_per_multiprocessor=self.programs_per_multiprocessor,
        )
        if self.block_rows < MIN_DOT_BLOCK or self.block_rows & (self.block_rows - 1):
            raise ValueError(f"block_rows must be a power of two of at least {MIN_DOT_BLOCK}; got {self.block_rows}")

    def fit(self, block_channel: int, element_size: int) -> RowTile:

        block_rows = min(self.block_rows, _operand_rows(block_channel, element_size))
        return self if block_rows == self.block_rows else replace(self, block_rows=block_rows)


@dataclass(frozen=True, slots=True)
class SharedBiasSplit:

    saturating_tiles: int
    max_parts: int
    min_rows_per_part: int
    workspace_bytes: int

    def __post_init__(self) -> None:
        validate_positive_parameters(
            saturating_tiles=self.saturating_tiles,
            max_parts=self.max_parts,
            min_rows_per_part=self.min_rows_per_part,
            workspace_bytes=self.workspace_bytes,
        )
        if 2 * self.saturating_tiles > CUDA_MAX_GRID_X:
            raise ValueError(
                f"a split grid of {2 * self.saturating_tiles} programs exceeds CUDA's first grid dimension"
            )

    def parts(self, score_tiles: int, rows: int, bias_numel: int, bias_element_size: int) -> int:

        parts = 1
        while (
            score_tiles > 0
            and parts < self.max_parts
            and score_tiles * parts < self.saturating_tiles
            and self.min_rows_per_part * 2 * parts <= rows
            and self.live_bytes(2 * parts, bias_numel, bias_element_size) <= self.workspace_bytes
        ):
            parts *= 2
        return parts

    @staticmethod
    def live_bytes(parts: int, bias_numel: int, bias_element_size: int) -> int:

        cast = 0 if bias_element_size == 4 else bias_element_size
        return ((parts + 1) * 4 + cast) * bias_numel


TileKey = tuple[TriangleAttentionPhase, torch.dtype, int]
PhaseTile = AttentionTile | RowTile


@dataclass(frozen=True, slots=True)
class TileTable:

    defaults: Mapping[TileKey, PhaseTile]
    dense: Mapping[TileKey, PhaseTile] = field(default_factory=dict)
    ragged: Mapping[TileKey, PhaseTile] = field(default_factory=dict)
    shared_bias_split: SharedBiasSplit | None = None

    def __post_init__(self) -> None:
        for entries in (self.defaults, self.dense, self.ragged):
            for (phase, _dtype, _block_channel), tile in entries.items():
                if phase is TriangleAttentionPhase.INFERENCE:
                    raise ValueError(
                        "triangle attention INFERENCE launches with the FORWARD tile and takes no entry of its own"
                    )
                expected = RowTile if phase is TriangleAttentionPhase.BACKWARD_PREPROCESS else AttentionTile
                if not isinstance(tile, expected):
                    raise TypeError(f"triangle attention {phase.value} takes a {expected.__name__}; got {tile!r}")

    def tile(
        self,
        phase: TriangleAttentionPhase,
        dtype: torch.dtype,
        head_size: int,
        shared_memory_bytes: int,
        *,
        ragged: bool,
    ) -> AttentionTile | None:

        if phase is TriangleAttentionPhase.BACKWARD_PREPROCESS:
            raise ValueError("the triangle attention backward preprocess launches a row tile; use row_tile()")
        if phase is TriangleAttentionPhase.INFERENCE:
            phase = TriangleAttentionPhase.FORWARD
        block_channel = full_width(head_size)
        tile = self._entry(phase, dtype, block_channel, ragged)
        assert isinstance(tile, AttentionTile)
        if head_size % tile.head_alignment:
            tile = self.defaults[(phase, dtype, block_channel)]
            assert isinstance(tile, AttentionTile)
        return tile.fit(block_channel, dtype.itemsize, shared_memory_bytes)

    def row_tile(self, dtype: torch.dtype, head_size: int, *, ragged: bool) -> RowTile:

        block_channel = full_width(head_size)
        tile = self._entry(TriangleAttentionPhase.BACKWARD_PREPROCESS, dtype, block_channel, ragged)
        assert isinstance(tile, RowTile)
        return tile.fit(block_channel, dtype.itemsize)

    def _entry(self, phase: TriangleAttentionPhase, dtype: torch.dtype, block_channel: int, ragged: bool) -> PhaseTile:
        key = (phase, dtype, block_channel)
        return (self.ragged if ragged else self.dense).get(key) or self.defaults[key]


def select_tile_table(
    generic: TileTable,
    targets: tuple[TargetProfile[TileTable], ...],
    target: TargetDescriptor | None,
) -> TileTable:

    return select_target_profile(generic, target=target, profiles=targets)


def full_width(channels: int) -> int:

    return max(16, _next_power_of_two(channels))


def _operand_rows(block_channel: int, element_size: int) -> int:

    bound = max(MIN_DOT_BLOCK, _OPERAND_BUDGET_BYTES // (block_channel * element_size))
    return 1 << (bound.bit_length() - 1)


def _staged_bytes(
    block_query: int,
    block_key: int,
    block_channel: int,
    element_size: int,
    num_stages: int,
) -> int:

    scores = dot_tile_shared_bytes(block_query, block_channel, block_key, element_size, 1, buffers=num_stages)
    values = dot_tile_shared_bytes(block_query, block_key, block_channel, element_size, 1, buffers=num_stages)
    return max(scores, values)


def _require_dot_extent(name: str, value: int) -> None:
    if value < 16 or value & (value - 1):
        raise ValueError(f"{name} must be a power of two of at least 16 for tl.dot; got {value}")
