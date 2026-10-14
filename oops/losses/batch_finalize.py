from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from oops.core import validate_positive_parameters
from oops.core._math import _ceil_div, _next_power_of_two
from oops.runtime import CUDA_MAX_GRID_X

_MAX_FINAL_PARTIALS = 256


@dataclass(frozen=True, slots=True)
class BatchFinalizeProblem:
    length: int
    scalar_output: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(length=self.length)


@dataclass(frozen=True, slots=True)
class BatchFinalizeConfig:
    max_block_items: int = 256

    def __post_init__(self) -> None:
        validate_positive_parameters(max_block_items=self.max_block_items)
        if self.max_block_items & (self.max_block_items - 1):
            raise ValueError("max_block_items must be a power of two")


@dataclass(frozen=True, slots=True)
class BatchFinalizePlan:
    tile_grid: tuple[int]
    finalize_grid: tuple[int]
    block_items: int
    num_partials: int
    block_partials: int
    scalar_output: bool
    tile_num_warps: int
    partials_num_warps: int

    def grid_fits(self) -> bool:
        return (
            (0 < self.tile_grid[0])
            & (self.tile_grid[0] <= CUDA_MAX_GRID_X)
            & (self.finalize_grid == (1,))
            & (self.num_partials <= _MAX_FINAL_PARTIALS)
        )


@dataclass(frozen=True, slots=True)
class BatchFinalizeSchedule:
    config: BatchFinalizeConfig

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: BatchFinalizeProblem) -> BatchFinalizePlan:
        plan = _make_batch_finalize_plan(problem, self.config)
        if not plan.grid_fits():
            raise ValueError("loss batch finalization has no launchable plan")
        return plan

    def supports(self, problem: BatchFinalizeProblem) -> bool:
        max_partials = min(CUDA_MAX_GRID_X, _MAX_FINAL_PARTIALS)
        return problem.length <= max_partials * self.config.max_block_items


def _make_batch_finalize_plan(
    problem: BatchFinalizeProblem,
    config: BatchFinalizeConfig,
) -> BatchFinalizePlan:
    block_items = min(config.max_block_items, _next_power_of_two(problem.length))
    num_partials = _ceil_div(problem.length, block_items)
    return BatchFinalizePlan(
        tile_grid=(num_partials,),
        finalize_grid=(1,),
        block_items=block_items,
        num_partials=num_partials,
        block_partials=_next_power_of_two(num_partials),
        scalar_output=problem.scalar_output,
        tile_num_warps=_num_warps(block_items),
        partials_num_warps=_num_warps(_next_power_of_two(num_partials)),
    )


def _num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 1
