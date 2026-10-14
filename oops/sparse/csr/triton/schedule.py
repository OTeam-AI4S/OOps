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

from dataclasses import dataclass
from functools import lru_cache

import torch

from oops.core import validate_positive_parameters
from oops.core._math import _ceil_div, _next_power_of_two, _require_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ
from oops.runtime.dispatch import _require_num_warps


@dataclass(frozen=True, slots=True)
class SegmentProblem:
    out_segments: int
    features: int

    def __post_init__(self) -> None:
        validate_positive_parameters(out_segments=self.out_segments, features=self.features)


@dataclass(frozen=True, slots=True)
class SegmentConfig:
    block_features: int = 64
    num_warps: int = 4
    max_block_edges: int = 32

    def __post_init__(self) -> None:
        _require_power_of_two("block_features", self.block_features)
        _require_num_warps("num_warps", self.num_warps)
        _require_power_of_two("max_block_edges", self.max_block_edges)


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    grid: tuple[int, int]
    block_features: int
    num_warps: int
    max_block_edges: int = 32

    def grid_fits(self) -> bool:
        return (
            (0 < self.grid[0])
            & (self.grid[0] <= CUDA_MAX_GRID_X)
            & (0 < self.grid[1])
            & (self.grid[1] <= CUDA_MAX_GRID_YZ)
        )


@dataclass(frozen=True, slots=True)
class SegmentSchedule:
    config: SegmentConfig

    def resolve(self, problem: SegmentProblem) -> SegmentPlan:
        if torch.compiler.is_compiling():
            return _resolve_segment_plan(problem, self.config)
        return _cached_segment_plan(problem, self.config)

    def supports(self, problem: SegmentProblem) -> bool:
        return (problem.out_segments <= CUDA_MAX_GRID_X) & (
            problem.features <= CUDA_MAX_GRID_YZ * self.config.block_features
        )


def _resolve_segment_plan(problem: SegmentProblem, config: SegmentConfig) -> SegmentPlan:
    plan = _make_segment_plan(problem, config)
    if not plan.grid_fits():
        raise ValueError("sparse segment operation has no launchable plan")
    return plan


@lru_cache(maxsize=1024)
def _cached_segment_plan(problem: SegmentProblem, config: SegmentConfig) -> SegmentPlan:

    return _resolve_segment_plan(problem, config)


def _make_segment_plan(problem: SegmentProblem, config: SegmentConfig) -> SegmentPlan:
    return SegmentPlan(
        grid=(problem.out_segments, _ceil_div(problem.features, config.block_features)),
        block_features=min(config.block_features, _next_power_of_two(problem.features)),
        num_warps=config.num_warps,
        max_block_edges=config.max_block_edges,
    )


def segment_block_edges(
    num_edges: int,
    num_segments: int,
    maximum: int,
    *,
    serial_short_rows: bool = False,
) -> int:

    rows = max(1, num_segments)
    average = max(1, (num_edges + rows - 1) // rows)
    if serial_short_rows and average <= 32:
        return 1
    return min(maximum, _next_power_of_two(average))


CSR_SUM_SCHEDULE = SegmentSchedule(SegmentConfig())

LONG_ROW_EDGES = 8192
LONG_BLOCK_FEATURES = 8
LONG_BLOCK_EDGES = 256


@lru_cache(maxsize=1024)
def adaptive_reduction_geometry(
    edges: int, inferred_segments: int, out_segments: int, block_features: int
) -> tuple[int, int] | None:
    if inferred_segments == 0 or edges < LONG_ROW_EDGES:
        return None
    long_features = min(LONG_BLOCK_FEATURES, block_features)
    programs = out_segments + (edges // LONG_ROW_EDGES) * (block_features // long_features)
    return (programs, long_features) if programs <= CUDA_MAX_GRID_X else None
