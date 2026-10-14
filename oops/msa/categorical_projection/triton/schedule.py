from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from oops.core import validate_positive_parameters
from oops.core._math import _ceil_div, _next_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ


@dataclass(frozen=True, slots=True)
class CategoricalProjectionProblem:
    rows: int
    channels: int
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(rows=self.rows, channels=self.channels)


@dataclass(frozen=True, slots=True)
class CategoricalProjectionConfig:
    block_rows: int = 128
    block_channels: int = 64
    num_warps: int = 4

    def __post_init__(self) -> None:
        validate_positive_parameters(
            block_rows=self.block_rows,
            block_channels=self.block_channels,
            num_warps=self.num_warps,
        )


@dataclass(frozen=True, slots=True)
class CategoricalProjectionPlan:
    grid: tuple[int, int]
    block_rows: int
    block_channels: int
    num_warps: int

    def grid_fits(self) -> bool:
        return (
            (0 < self.grid[0])
            & (self.grid[0] <= CUDA_MAX_GRID_X)
            & (0 < self.grid[1])
            & (self.grid[1] <= CUDA_MAX_GRID_YZ)
        )


@dataclass(frozen=True, slots=True)
class CategoricalProjectionSchedule:
    config: CategoricalProjectionConfig

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: CategoricalProjectionProblem) -> CategoricalProjectionPlan:
        plan = _make_plan(problem, self.config)
        if not plan.grid_fits():
            raise ValueError("categorical MSA projection has no launchable plan")
        return plan

    def supports(self, problem: CategoricalProjectionProblem) -> bool:
        return (problem.rows <= CUDA_MAX_GRID_X * self.config.block_rows) & (
            problem.channels <= CUDA_MAX_GRID_YZ * self.config.block_channels
        )


def _make_plan(
    problem: CategoricalProjectionProblem,
    config: CategoricalProjectionConfig,
) -> CategoricalProjectionPlan:
    block_channels = min(config.block_channels, _next_power_of_two(problem.channels))
    return CategoricalProjectionPlan(
        grid=(
            _ceil_div(problem.rows, config.block_rows),
            _ceil_div(problem.channels, block_channels),
        ),
        block_rows=config.block_rows,
        block_channels=block_channels,
        num_warps=config.num_warps,
    )


@dataclass(frozen=True, slots=True)
class CategoricalProjectionRule:
    config: CategoricalProjectionSchedule
    min_rows: int = 0
    max_channels: int | None = None
    max_rows: int | None = None
    dtypes: tuple[torch.dtype, ...] | None = None

    def can_implement(self, problem: CategoricalProjectionProblem) -> bool:
        return self.config.supports(problem) and (self.dtypes is None or problem.dtype in self.dtypes)

    def is_preferred(self, problem: CategoricalProjectionProblem) -> bool:
        return (
            self.can_implement(problem)
            and problem.rows >= self.min_rows
            and (self.max_rows is None or problem.rows <= self.max_rows)
            and (self.max_channels is None or problem.channels <= self.max_channels)
        )
