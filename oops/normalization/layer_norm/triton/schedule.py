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

from oops.core import (
    LaunchProfile,
    LaunchProfileDelta,
    TargetDescriptor,
    TargetProfile,
    select_launch_candidates,
    select_launch_config,
    select_target_launch_profile,
    validate_positive_parameters,
)
from oops.core._math import _ceil_div, _next_power_of_two
from oops.runtime import CUDA_MAX_GRID_X

MAX_LAYER_NORM_WIDTH = 8192


@dataclass(frozen=True, slots=True)
class ForwardWidthOverride:

    block_columns: int
    block_rows: int
    half_num_warps: int
    float32_num_warps: int


@dataclass(frozen=True, slots=True)
class BackwardWidthBand:

    max_block_columns: int | None
    block_rows: int


@dataclass(frozen=True, slots=True)
class LayerNormLaunchPolicy:

    forward_tile_elements: int
    forward_max_block_rows: int
    forward_min_programs: int
    forward_warp_elements: int
    forward_max_num_warps: int
    forward_width_overrides: tuple[ForwardWidthOverride, ...]
    backward_width_bands: tuple[BackwardWidthBand, ...]
    backward_min_programs: int
    backward_warp_columns: int
    backward_max_num_warps: int


@dataclass(frozen=True, slots=True)
class LayerNormTileConfig:

    block_rows: int
    num_warps: int


def resolve_forward_tile(
    policy: LayerNormLaunchPolicy,
    dtype: torch.dtype,
    num_rows: int,
    block_columns: int,
) -> LayerNormTileConfig:

    for override in policy.forward_width_overrides:
        if block_columns == override.block_columns:
            num_warps = override.float32_num_warps if dtype is torch.float32 else override.half_num_warps
            return LayerNormTileConfig(override.block_rows, num_warps)

    block_rows = min(
        max(policy.forward_tile_elements // block_columns, 1),
        policy.forward_max_block_rows,
    )
    block_rows = min(block_rows, _row_occupancy_cap(num_rows, policy.forward_min_programs))
    num_warps = min(
        max((block_rows * block_columns) // policy.forward_warp_elements, 1),
        policy.forward_max_num_warps,
    )
    return LayerNormTileConfig(block_rows, num_warps)


def forward_grid_block_rows(policy: LayerNormLaunchPolicy, block_columns: int) -> int:

    for override in policy.forward_width_overrides:
        if block_columns == override.block_columns:
            return override.block_rows
    return min(
        max(policy.forward_tile_elements // block_columns, 1),
        policy.forward_max_block_rows,
    )


def resolve_backward_tile(
    policy: LayerNormLaunchPolicy,
    num_rows: int,
    block_columns: int,
) -> LayerNormTileConfig:

    block_rows = backward_grid_block_rows(policy, block_columns)
    block_rows = min(block_rows, _row_occupancy_cap(num_rows, policy.backward_min_programs))
    num_warps = min(
        max(block_columns // policy.backward_warp_columns, block_rows * block_columns // 2048, 1),
        policy.backward_max_num_warps,
    )
    return LayerNormTileConfig(block_rows, num_warps)


def backward_grid_block_rows(policy: LayerNormLaunchPolicy, block_columns: int) -> int:

    for band in policy.backward_width_bands:
        if band.max_block_columns is None or block_columns <= band.max_block_columns:
            return band.block_rows
    raise ValueError("backward_width_bands must end in an unbounded band")


def _row_occupancy_cap(num_rows: int, min_programs: int) -> int:
    cap = max(1, num_rows // min_programs)
    return 1 << (cap.bit_length() - 1)


PORTABLE_POLICY = LayerNormLaunchPolicy(
    forward_tile_elements=2048,
    forward_max_block_rows=64,
    forward_min_programs=256,
    forward_warp_elements=2048,
    forward_max_num_warps=4,
    forward_width_overrides=(
        ForwardWidthOverride(
            block_columns=128,
            block_rows=4,
            half_num_warps=1,
            float32_num_warps=1,
        ),
    ),
    backward_width_bands=(
        BackwardWidthBand(max_block_columns=128, block_rows=16),
        BackwardWidthBand(max_block_columns=2048, block_rows=8),
        BackwardWidthBand(max_block_columns=None, block_rows=4),
    ),
    backward_min_programs=256,
    backward_warp_columns=512,
    backward_max_num_warps=4,
)


@dataclass(frozen=True, slots=True)
class LayerNormProblem:
    num_rows: int
    num_columns: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        validate_positive_parameters(num_rows=self.num_rows, num_columns=self.num_columns)
        if self.num_columns > MAX_LAYER_NORM_WIDTH:
            raise ValueError(f"num_columns exceeds LayerNorm's {MAX_LAYER_NORM_WIDTH}-column domain")

    @property
    def block_columns(self) -> int:
        return _next_power_of_two(self.num_columns)

    @property
    def is_cacheable(self) -> bool:

        return type(self.num_rows) is int and type(self.num_columns) is int


@dataclass(frozen=True, slots=True)
class LayerNormPhaseProfile:
    generic: LaunchProfile[int, LayerNormLaunchPolicy]
    target_deltas: tuple[TargetProfile[LaunchProfileDelta[int, LayerNormLaunchPolicy]], ...] = ()

    def select(self, problem: LayerNormProblem, target: TargetDescriptor | None) -> LayerNormLaunchPolicy:
        profile = select_target_launch_profile(self.generic, target=target, deltas=self.target_deltas)
        return select_launch_config(problem.block_columns, profile)

    def candidates(self, problem: LayerNormProblem) -> tuple[LayerNormLaunchPolicy, ...]:
        return select_launch_candidates(
            problem.block_columns,
            self.generic,
            target_deltas=self.target_deltas,
        )


@dataclass(frozen=True, slots=True)
class LayerNormPlan:
    grid: tuple[int]
    block_rows: int
    block_columns: int
    num_warps: int

    def grid_fits(self) -> bool:
        return (0 < self.grid[0]) & (self.grid[0] <= CUDA_MAX_GRID_X)


@dataclass(frozen=True, slots=True, eq=False)
class LayerNormScheduleCatalog:
    forward: LayerNormPhaseProfile
    backward: LayerNormPhaseProfile

    def resolve_forward(
        self,
        problem: LayerNormProblem,
        *,
        target: TargetDescriptor | None = None,
        cache: bool = True,
    ) -> LayerNormPlan:
        if cache and problem.is_cacheable and not torch.compiler.is_compiling():
            return _cached_resolve_forward(self, problem, target)
        return _resolve_forward(self, problem, target)

    def resolve_backward(
        self,
        problem: LayerNormProblem,
        *,
        target: TargetDescriptor | None = None,
        cache: bool = True,
    ) -> LayerNormPlan:
        if cache and problem.is_cacheable and not torch.compiler.is_compiling():
            return _cached_resolve_backward(self, problem, target)
        return _resolve_backward(self, problem, target)

    def supports(self, problem: LayerNormProblem, *, cache: bool = True) -> bool:
        if cache and problem.is_cacheable and not torch.compiler.is_compiling():
            return _cached_supports(self, problem)
        return _supports(self, problem)

    def supports_forward(self, problem: LayerNormProblem, *, cache: bool = True) -> bool:
        if cache and problem.is_cacheable and not torch.compiler.is_compiling():
            return _cached_supports_forward(self, problem)
        return _supports_forward(self, problem)

    def supports_backward(self, problem: LayerNormProblem, *, cache: bool = True) -> bool:
        if cache and problem.is_cacheable and not torch.compiler.is_compiling():
            return _cached_supports_backward(self, problem)
        return _supports_backward(self, problem)


def _resolve_forward(
    catalog: LayerNormScheduleCatalog,
    problem: LayerNormProblem,
    target: TargetDescriptor | None,
) -> LayerNormPlan:
    plan = _make_forward_plan(problem, catalog.forward.select(problem, target))
    if not plan.grid_fits():
        raise ValueError("LayerNorm has no launchable forward plan")
    return plan


@lru_cache(maxsize=1024)
def _cached_resolve_forward(
    catalog: LayerNormScheduleCatalog,
    problem: LayerNormProblem,
    target: TargetDescriptor | None,
) -> LayerNormPlan:
    return _resolve_forward(catalog, problem, target)


def _resolve_backward(
    catalog: LayerNormScheduleCatalog,
    problem: LayerNormProblem,
    target: TargetDescriptor | None,
) -> LayerNormPlan:
    plan = _make_backward_plan(problem, catalog.backward.select(problem, target))
    if not plan.grid_fits():
        raise ValueError("LayerNorm has no launchable backward plan")
    return plan


@lru_cache(maxsize=1024)
def _cached_resolve_backward(
    catalog: LayerNormScheduleCatalog,
    problem: LayerNormProblem,
    target: TargetDescriptor | None,
) -> LayerNormPlan:
    return _resolve_backward(catalog, problem, target)


def _supports(catalog: LayerNormScheduleCatalog, problem: LayerNormProblem) -> bool:
    return _supports_forward(catalog, problem) and _supports_backward(catalog, problem)


@lru_cache(maxsize=1024)
def _cached_supports(catalog: LayerNormScheduleCatalog, problem: LayerNormProblem) -> bool:
    return _supports(catalog, problem)


def _supports_forward(catalog: LayerNormScheduleCatalog, problem: LayerNormProblem) -> bool:
    supported = True
    for policy in catalog.forward.candidates(problem):
        block_rows = forward_grid_block_rows(policy, problem.block_columns)
        grid = _ceil_div(problem.num_rows, block_rows)
        supported = supported & (0 < grid) & (grid <= CUDA_MAX_GRID_X)
    return supported


@lru_cache(maxsize=1024)
def _cached_supports_forward(catalog: LayerNormScheduleCatalog, problem: LayerNormProblem) -> bool:
    return _supports_forward(catalog, problem)


def _supports_backward(catalog: LayerNormScheduleCatalog, problem: LayerNormProblem) -> bool:
    supported = True
    for policy in catalog.backward.candidates(problem):
        block_rows = backward_grid_block_rows(policy, problem.block_columns)
        grid = _ceil_div(problem.num_rows, block_rows)
        supported = supported & (0 < grid) & (grid <= CUDA_MAX_GRID_X)
    return supported


@lru_cache(maxsize=1024)
def _cached_supports_backward(catalog: LayerNormScheduleCatalog, problem: LayerNormProblem) -> bool:
    return _supports_backward(catalog, problem)


def _make_forward_plan(problem: LayerNormProblem, policy: LayerNormLaunchPolicy) -> LayerNormPlan:
    config = resolve_forward_tile(
        policy,
        problem.dtype,
        problem.num_rows,
        problem.block_columns,
    )
    return _make_plan(problem, config.block_rows, config.num_warps)


def _make_backward_plan(problem: LayerNormProblem, policy: LayerNormLaunchPolicy) -> LayerNormPlan:
    config = resolve_backward_tile(policy, problem.num_rows, problem.block_columns)
    return _make_plan(problem, config.block_rows, config.num_warps)


def _make_plan(problem: LayerNormProblem, block_rows: int, num_warps: int) -> LayerNormPlan:
    return LayerNormPlan(
        grid=(_ceil_div(problem.num_rows, block_rows),),
        block_rows=block_rows,
        block_columns=problem.block_columns,
        num_warps=num_warps,
    )
