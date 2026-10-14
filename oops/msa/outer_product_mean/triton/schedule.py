from __future__ import annotations

from dataclasses import dataclass

import torch

from oops.core import CoreTileConfig, validate_positive_parameters
from oops.core._math import _ceil_div, _next_power_of_two
from oops.runtime import MIN_DOT_REDUCTION, halved_dot_blocks, narrowed_dot_tile
from oops.runtime.cuda import _grid_2d_fits, _grid_3d_fits


@dataclass(frozen=True, slots=True)
class DualProjectionProblem:
    rows: int
    channels: int
    hidden: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        validate_positive_parameters(rows=self.rows, channels=self.channels, hidden=self.hidden)


@dataclass(frozen=True, slots=True)
class DualProjectionPlan:
    grid: tuple[int, int]
    block_rows: int
    block_reduction: int
    block_hidden: int
    num_warps: int
    num_stages: int

    def grid_fits(self) -> bool:
        return _grid_2d_fits(self.grid)


def make_dual_projection_plan(
    problem: DualProjectionProblem,
    config: CoreTileConfig,
    shared_memory_bytes: int,
) -> DualProjectionPlan | None:
    configured_reduction = config.block_reduction
    if configured_reduction is None or not _core_config_is_legal(
        config,
        (config.block_rows, configured_reduction),
    ):
        return None
    block_hidden = _next_power_of_two(problem.hidden)
    block_reduction = max(MIN_DOT_REDUCTION, min(_next_power_of_two(problem.channels), configured_reduction))
    tile = narrowed_dot_tile(
        ((candidate, block_hidden) for candidate in halved_dot_blocks(config.block_rows)),
        block_reduction,
        problem.dtype.itemsize,
        shared_memory_bytes,
        buffers=config.num_stages,
    )
    if tile is None:
        return None
    plan = DualProjectionPlan(
        grid=(_ceil_div(problem.rows, tile[0]), 1),
        block_rows=tile[0],
        block_reduction=block_reduction,
        block_hidden=block_hidden,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return plan if plan.grid_fits() else None


@dataclass(frozen=True, slots=True)
class PairTailProblem:
    pair_rows: int
    tokens: int
    channels: int
    hidden: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        validate_positive_parameters(
            pair_rows=self.pair_rows,
            tokens=self.tokens,
            channels=self.channels,
            hidden=self.hidden,
        )

    @property
    def total_pairs(self) -> int:
        return self.pair_rows * self.tokens


@dataclass(frozen=True, slots=True)
class PairTailPlan:
    grid: tuple[int, int, int]
    row_aligned: bool
    block_rows: int
    block_channels: int
    block_hidden: int
    num_warps: int
    num_stages: int

    def grid_fits(self) -> bool:
        return _grid_3d_fits(self.grid)


def make_pair_tail_plan(
    problem: PairTailProblem,
    config: CoreTileConfig,
    shared_memory_bytes: int,
) -> PairTailPlan | None:

    block_channels = config.block_channels
    block_hidden = config.block_reduction
    if (
        block_channels is None
        or block_hidden is None
        or not _core_config_is_legal(
            config,
            (config.block_rows, block_channels, block_hidden),
        )
    ):
        return None
    tile = narrowed_dot_tile(
        ((rows, block_channels) for rows in halved_dot_blocks(config.block_rows)),
        block_hidden,
        problem.dtype.itemsize,
        shared_memory_bytes,
        operands=1,
        buffers=config.num_stages,
    )
    if tile is None:
        return None
    block_rows = tile[0]
    row_grid = (
        _ceil_div(problem.tokens, block_rows),
        problem.pair_rows,
        _ceil_div(problem.channels, block_channels),
    )
    row_aligned = _row_aligned_grid_is_preferred(problem.pair_rows, problem.tokens, block_rows, row_grid)
    plan = PairTailPlan(
        grid=(
            row_grid
            if row_aligned
            else (
                _ceil_div(problem.total_pairs, block_rows),
                _ceil_div(problem.channels, block_channels),
                1,
            )
        ),
        row_aligned=row_aligned,
        block_rows=block_rows,
        block_channels=block_channels,
        block_hidden=block_hidden,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return plan if plan.grid_fits() else None


def _row_aligned_grid_is_preferred(
    pair_rows: int,
    tokens: int,
    block_rows: int,
    grid: tuple[int, int, int],
) -> bool:
    if not _grid_3d_fits(grid):
        return False
    row_programs = grid[0] * pair_rows
    flat_programs = _ceil_div(pair_rows * tokens, block_rows)
    return row_programs * 8 <= flat_programs * 9


def _core_config_is_legal(config: CoreTileConfig, required_blocks: tuple[int, ...]) -> bool:
    return all(_is_power_of_two(value) for value in required_blocks) and config.num_warps in (1, 2, 4, 8)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0
