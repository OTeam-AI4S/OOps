from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import torch

from oops.core import validate_positive_parameters
from oops.core._math import _ceil_div, _next_power_of_two, _require_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ, INT32_MAX
from oops.runtime.cuda import _grid_1d_fits, _grid_2d_fits

from ...batch_finalize import BatchFinalizePlan, BatchFinalizeProblem, BatchFinalizeSchedule

_FLOAT32_BYTES = 4
_MIN_ROW_BLOCKS_PER_PART = 4
_DISTOGRAM_REDUCE_BLOCK_ITEMS = 256


@dataclass(frozen=True, slots=True)
class CrossEntropyProblem:
    rows: int
    num_classes: int
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(rows=self.rows, num_classes=self.num_classes)


@dataclass(frozen=True, slots=True)
class CrossEntropyConfig:
    block_rows: int = 32
    max_num_classes: int = 256

    def __post_init__(self) -> None:
        _require_power_of_two("cross-entropy block_rows", self.block_rows)
        validate_positive_parameters(max_num_classes=self.max_num_classes)


@dataclass(frozen=True, slots=True)
class CrossEntropyPlan:
    grid: tuple[int]
    block_rows: int
    block_classes: int
    num_warps: int

    def grid_fits(self) -> bool:
        return (0 < self.grid[0]) & (self.grid[0] <= CUDA_MAX_GRID_X)


@dataclass(frozen=True, slots=True)
class CrossEntropySchedule:
    config: CrossEntropyConfig

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: CrossEntropyProblem) -> CrossEntropyPlan:
        plan = _make_cross_entropy_plan(problem, self.config)
        if problem.num_classes > self.config.max_num_classes or not plan.grid_fits():
            raise ValueError("cross entropy has no launchable plan")
        return plan

    def supports(self, problem: CrossEntropyProblem) -> bool:
        return (problem.num_classes <= self.config.max_num_classes) & (
            problem.rows <= CUDA_MAX_GRID_X * self.config.block_rows
        )


def _make_cross_entropy_plan(
    problem: CrossEntropyProblem,
    config: CrossEntropyConfig,
) -> CrossEntropyPlan:
    block_classes = _next_power_of_two(problem.num_classes)
    return CrossEntropyPlan(
        grid=(_ceil_div(problem.rows, config.block_rows),),
        block_rows=config.block_rows,
        block_classes=block_classes,
        num_warps=_num_warps(config.block_rows * block_classes),
    )


@dataclass(frozen=True, slots=True)
class DistogramCrossEntropyProblem:
    batch: int
    num_tokens: int
    num_classes: int
    scalar_output: bool
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(
            batch=self.batch,
            num_tokens=self.num_tokens,
            num_classes=self.num_classes,
        )

    @property
    def rows_per_batch(self) -> int:
        return self.num_tokens * self.num_tokens


@dataclass(frozen=True, slots=True)
class DistogramCrossEntropyConfig:
    block_rows: int = 128
    max_block_tokens: int = 32
    max_num_classes: int = 256
    max_deterministic_row_parts: int = 256
    deterministic_workspace_bytes: int = 16 << 20
    max_block_elements: int | None = None
    accumulate_num_warps: int | None = None

    def __post_init__(self) -> None:
        _require_power_of_two("distogram block_rows", self.block_rows)
        if self.accumulate_num_warps not in (None, 1, 2, 4, 8):
            raise ValueError("distogram accumulate_num_warps must be one of 1, 2, 4, or 8")
        _require_power_of_two("distogram max_block_tokens", self.max_block_tokens)
        if self.max_block_elements is not None:
            _require_power_of_two("distogram max_block_elements", self.max_block_elements)
        _require_power_of_two("distogram max_deterministic_row_parts", self.max_deterministic_row_parts)
        validate_positive_parameters(
            max_num_classes=self.max_num_classes,
            deterministic_workspace_bytes=self.deterministic_workspace_bytes,
        )


@dataclass(frozen=True, slots=True)
class DistogramCrossEntropyPlan:
    accumulate_grid: tuple[int, int]
    token_mask_backward_grid: tuple[int, int]
    block_rows: int
    block_tokens: int
    block_classes: int
    accumulate_num_warps: int
    token_mask_num_warps: int
    wide_offsets: bool
    finalize_plan: BatchFinalizePlan
    deterministic_row_parts: int
    deterministic_accumulate_grid: tuple[int, int]
    deterministic_reduce_grid: tuple[int]
    deterministic_reduce_block_items: int
    deterministic_reduce_num_warps: int

    def grid_fits(self) -> bool:
        return (
            (0 < self.accumulate_grid[0])
            & (self.accumulate_grid[0] <= CUDA_MAX_GRID_X)
            & (0 < self.accumulate_grid[1])
            & (self.accumulate_grid[1] <= CUDA_MAX_GRID_YZ)
            & (0 < self.token_mask_backward_grid[0])
            & (self.token_mask_backward_grid[0] <= CUDA_MAX_GRID_X)
            & (0 < self.token_mask_backward_grid[1])
            & (self.token_mask_backward_grid[1] <= CUDA_MAX_GRID_YZ)
            & _grid_2d_fits(self.deterministic_accumulate_grid)
            & _grid_1d_fits(self.deterministic_reduce_grid)
            & self.finalize_plan.grid_fits()
        )


@dataclass(frozen=True, slots=True)
class DistogramCrossEntropySchedule:
    config: DistogramCrossEntropyConfig
    finalize_schedule: BatchFinalizeSchedule

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: DistogramCrossEntropyProblem, *, backward: bool = False) -> DistogramCrossEntropyPlan:
        plan = _make_distogram_cross_entropy_plan(problem, self.config, self.finalize_schedule)
        if problem.num_classes > self.config.max_num_classes or not plan.grid_fits():
            raise ValueError("distogram cross entropy has no launchable plan")
        return _bounded_class_reduction_plan(problem, plan, backward=backward)

    def supports(self, problem: DistogramCrossEntropyProblem) -> bool:
        supported = (
            (problem.num_classes <= self.config.max_num_classes)
            & (problem.rows_per_batch <= CUDA_MAX_GRID_X * self.config.block_rows)
            & (problem.batch <= CUDA_MAX_GRID_YZ)
            & (problem.num_tokens <= CUDA_MAX_GRID_X)
        )
        return supported & self.finalize_schedule.supports(BatchFinalizeProblem(problem.batch, problem.scalar_output))


def _bounded_class_reduction_plan(
    problem: DistogramCrossEntropyProblem, plan: DistogramCrossEntropyPlan, *, backward: bool = False
) -> DistogramCrossEntropyPlan:
    half_wide_backward = (
        backward and problem.dtype in (torch.float16, torch.bfloat16) and 256 <= plan.block_classes <= 2048
    )
    if not half_wide_backward and (problem.dtype is not torch.float32 or plan.block_classes < 128):
        return plan
    rows = min(plan.block_rows, 2048 // plan.block_classes)
    grid = ((problem.rows_per_batch + rows - 1) // rows, problem.batch)
    candidate = replace(plan, block_rows=rows, accumulate_grid=grid, accumulate_num_warps=4)
    return candidate if candidate.grid_fits() else plan


def _make_distogram_cross_entropy_plan(
    problem: DistogramCrossEntropyProblem,
    config: DistogramCrossEntropyConfig,
    finalize_schedule: BatchFinalizeSchedule,
) -> DistogramCrossEntropyPlan:
    block_classes = _next_power_of_two(problem.num_classes)
    block_rows = config.block_rows
    block_tokens = min(config.max_block_tokens, _next_power_of_two(problem.num_tokens))
    if config.max_block_elements is not None:
        block_rows = min(block_rows, max(1, config.max_block_elements // block_classes))
        block_tokens = min(block_tokens, max(1, config.max_block_elements // (2 * block_classes)))
        if _ceil_div(problem.rows_per_batch, block_rows) > CUDA_MAX_GRID_X:
            block_rows = config.block_rows
    reduce_block_items = min(_DISTOGRAM_REDUCE_BLOCK_ITEMS, _next_power_of_two(problem.batch * 2))
    finalize_plan = finalize_schedule.resolve(BatchFinalizeProblem(problem.batch, problem.scalar_output))
    row_blocks = _ceil_div(problem.rows_per_batch, block_rows)
    row_parts = _distogram_row_parts(row_blocks, problem.batch, config)
    accumulator_items = problem.batch * 2
    return DistogramCrossEntropyPlan(
        accumulate_grid=(row_blocks, problem.batch),
        token_mask_backward_grid=(problem.num_tokens, problem.batch),
        block_rows=block_rows,
        block_tokens=block_tokens,
        block_classes=block_classes,
        accumulate_num_warps=config.accumulate_num_warps or _num_warps(block_rows * block_classes),
        token_mask_num_warps=_num_warps(block_tokens * block_classes),
        wide_offsets=_distogram_needs_wide_offsets(problem, block_rows, block_classes),
        finalize_plan=finalize_plan,
        deterministic_row_parts=row_parts,
        deterministic_accumulate_grid=(row_parts, problem.batch),
        deterministic_reduce_grid=(_ceil_div(accumulator_items, reduce_block_items),),
        deterministic_reduce_block_items=reduce_block_items,
        deterministic_reduce_num_warps=_num_warps(row_parts * reduce_block_items),
    )


def _distogram_row_parts(row_blocks: int, batch: int, config: DistogramCrossEntropyConfig) -> int:

    parts = config.max_deterministic_row_parts
    partial_bytes = batch * 2 * _FLOAT32_BYTES
    while parts > 1 and (
        parts * _MIN_ROW_BLOCKS_PER_PART > row_blocks or parts * partial_bytes > config.deterministic_workspace_bytes
    ):
        parts //= 2
    return parts


def _distogram_needs_wide_offsets(
    problem: DistogramCrossEntropyProblem,
    block_rows: int,
    block_classes: int,
) -> bool:

    # Masked row padding is included in the address-width bound.
    padding_rows = block_rows - 1
    row_base = problem.batch * problem.rows_per_batch - 1 + padding_rows
    logits_offset = (
        problem.batch * problem.rows_per_batch * problem.num_classes
        - 1
        + padding_rows * problem.num_classes
        + (block_classes - problem.num_classes)
    )
    token_mask_offset = problem.batch * problem.num_tokens - 1 + padding_rows
    return max(row_base, logits_offset, token_mask_offset, problem.batch * 2 - 1) > INT32_MAX


def _num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 1


@dataclass(frozen=True, slots=True)
class CrossEntropyRule:
    config: CrossEntropySchedule
    min_rows: int = 0
    min_classes: int = 0
    max_classes: int | None = None
    dtypes: tuple[torch.dtype, ...] | None = None

    def can_implement(self, problem: CrossEntropyProblem) -> bool:
        return self.config.supports(problem) and (self.dtypes is None or problem.dtype in self.dtypes)

    def is_preferred(self, problem: CrossEntropyProblem) -> bool:
        return (
            self.can_implement(problem)
            and problem.rows >= self.min_rows
            and problem.num_classes >= self.min_classes
            and (self.max_classes is None or problem.num_classes <= self.max_classes)
        )


@dataclass(frozen=True, slots=True)
class DistogramCrossEntropyRule:
    config: DistogramCrossEntropySchedule
    min_tokens: int = 0
    min_classes: int = 0
    max_classes: int | None = None
    dtypes: tuple[torch.dtype, ...] | None = None

    def can_implement(self, problem: DistogramCrossEntropyProblem) -> bool:
        return self.config.supports(problem) and (self.dtypes is None or problem.dtype in self.dtypes)

    def is_preferred(self, problem: DistogramCrossEntropyProblem) -> bool:
        return (
            self.can_implement(problem)
            and problem.num_tokens >= self.min_tokens
            and problem.num_classes >= self.min_classes
            and (self.max_classes is None or problem.num_classes <= self.max_classes)
        )
