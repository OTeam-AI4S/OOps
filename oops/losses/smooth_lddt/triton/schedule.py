from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from oops.core import validate_positive_parameters
from oops.core._math import _ceil_div, _require_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ, INT32_MAX
from oops.runtime.cuda import _grid_1d_fits, _grid_2d_fits

from ...batch_finalize import BatchFinalizePlan, BatchFinalizeProblem, BatchFinalizeSchedule


@dataclass(frozen=True, slots=True)
class SmoothLddtDenseProblem:
    batch: int
    atoms: int
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(batch=self.batch, atoms=self.atoms)


@dataclass(frozen=True, slots=True)
class SmoothLddtDenseConfig:
    block_pairs: int = 128
    row_block_atoms: int = 128
    num_warps: int = 4

    def __post_init__(self) -> None:
        _require_power_of_two("smooth-LDDT dense block_pairs", self.block_pairs)
        _require_power_of_two("smooth-LDDT dense row_block_atoms", self.row_block_atoms)
        _require_num_warps("smooth-LDDT dense", self.num_warps)


@dataclass(frozen=True, slots=True)
class SmoothLddtDensePlan:
    problem: SmoothLddtDenseProblem
    config: SmoothLddtDenseConfig
    pairs_per_batch: int
    accumulate_grid: tuple[int, int]
    backward_grid: tuple[int, int]
    finalize_plan: BatchFinalizePlan
    finalize_tile_num_warps: int
    finalize_partials_num_warps: int

    def grid_fits(self) -> bool:
        return _grid_2d_fits(self.accumulate_grid) & _grid_2d_fits(self.backward_grid) & self.finalize_plan.grid_fits()


@dataclass(frozen=True, slots=True)
class SmoothLddtDenseSchedule:
    config: SmoothLddtDenseConfig
    finalize_schedule: BatchFinalizeSchedule

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: SmoothLddtDenseProblem) -> SmoothLddtDensePlan:
        plan = _make_smooth_lddt_dense_plan(problem, self.config, self.finalize_schedule)
        if not plan.grid_fits():
            raise ValueError("smooth-LDDT dense has no launchable plan")
        return plan

    def supports(self, problem: SmoothLddtDenseProblem) -> bool:
        return (
            (problem.atoms * problem.atoms <= CUDA_MAX_GRID_X * self.config.block_pairs)
            & (problem.atoms <= CUDA_MAX_GRID_X)
            & (problem.batch <= CUDA_MAX_GRID_YZ)
            & self.finalize_schedule.supports(BatchFinalizeProblem(problem.batch, True))
        )


def _make_smooth_lddt_dense_plan(
    problem: SmoothLddtDenseProblem,
    config: SmoothLddtDenseConfig,
    finalize_schedule: BatchFinalizeSchedule,
) -> SmoothLddtDensePlan:
    pairs_per_batch = problem.atoms * problem.atoms
    finalize_plan = finalize_schedule.resolve(BatchFinalizeProblem(problem.batch, True))
    return SmoothLddtDensePlan(
        problem=problem,
        config=config,
        pairs_per_batch=pairs_per_batch,
        accumulate_grid=(_ceil_div(pairs_per_batch, config.block_pairs), problem.batch),
        backward_grid=(problem.atoms, problem.batch),
        finalize_plan=finalize_plan,
        finalize_tile_num_warps=_smooth_num_warps(finalize_plan.block_items),
        finalize_partials_num_warps=_smooth_num_warps(finalize_plan.block_partials),
    )


@dataclass(frozen=True, slots=True)
class SmoothLddtMaskProblem:
    batch: int
    atoms: int
    truth_batched: bool
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(batch=self.batch, atoms=self.atoms)


@dataclass(frozen=True, slots=True)
class SmoothLddtMaskConfig:
    block_pairs: int = 256
    row_block_atoms: int = 128
    num_warps: int = 4

    def __post_init__(self) -> None:
        _require_power_of_two("smooth-LDDT mask block_pairs", self.block_pairs)
        _require_power_of_two("smooth-LDDT mask row_block_atoms", self.row_block_atoms)
        _require_num_warps("smooth-LDDT mask", self.num_warps)


@dataclass(frozen=True, slots=True)
class SmoothLddtMaskPlan:
    problem: SmoothLddtMaskProblem
    config: SmoothLddtMaskConfig
    pairs_per_batch: int
    total_pairs: int
    forward_grid: tuple[int]
    finalize_grid: tuple[int]
    backward_grid: tuple[int, int]
    forward_wide_offsets: bool

    def grid_fits(self) -> bool:
        return _grid_1d_fits(self.forward_grid) & (self.finalize_grid == (1,)) & _grid_2d_fits(self.backward_grid)


@dataclass(frozen=True, slots=True)
class SmoothLddtMaskSchedule:
    config: SmoothLddtMaskConfig

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: SmoothLddtMaskProblem) -> SmoothLddtMaskPlan:
        plan = _make_smooth_lddt_mask_plan(problem, self.config)
        if not plan.grid_fits():
            raise ValueError("smooth-LDDT mask has no launchable plan")
        return plan

    def supports(self, problem: SmoothLddtMaskProblem) -> bool:
        return (
            (problem.batch * problem.atoms * problem.atoms <= CUDA_MAX_GRID_X * self.config.block_pairs)
            & (problem.atoms <= CUDA_MAX_GRID_X)
            & (problem.batch <= CUDA_MAX_GRID_YZ)
        )


def _make_smooth_lddt_mask_plan(
    problem: SmoothLddtMaskProblem,
    config: SmoothLddtMaskConfig,
) -> SmoothLddtMaskPlan:
    pairs_per_batch = problem.atoms * problem.atoms
    total_pairs = problem.batch * pairs_per_batch
    coordinate_padding = (config.block_pairs - 1) * problem.atoms * 3
    true_numel = (problem.batch if problem.truth_batched else 1) * problem.atoms * 3
    return SmoothLddtMaskPlan(
        problem=problem,
        config=config,
        pairs_per_batch=pairs_per_batch,
        total_pairs=total_pairs,
        forward_grid=(_ceil_div(total_pairs, config.block_pairs),),
        finalize_grid=(1,),
        backward_grid=(problem.atoms, problem.batch),
        forward_wide_offsets=_needs_wide_offsets(
            _padded_program_offset_upper_bound(total_pairs, config.block_pairs),
            pairs_per_batch - 1,
            problem.batch * problem.atoms * 3 - 1 + coordinate_padding,
            true_numel - 1 + (coordinate_padding if problem.truth_batched else 0),
        ),
    )


@dataclass(frozen=True, slots=True)
class SmoothLddtPairDistanceProblem:
    batch: int
    atoms: int
    pairs: int
    unbatched: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(batch=self.batch, atoms=self.atoms)
        if self.pairs < 0:
            raise ValueError("smooth-LDDT pair count must be non-negative")
        if self.unbatched and self.batch != 1:
            raise ValueError("an unbatched pair-distance problem must have batch 1")


@dataclass(frozen=True, slots=True)
class SmoothLddtPairDistanceConfig:
    forward_block_pairs: int = 512
    backward_block_pairs: int = 128
    num_warps: int = 4

    def __post_init__(self) -> None:
        _require_power_of_two("smooth-LDDT pair-distance forward_block_pairs", self.forward_block_pairs)
        _require_power_of_two("smooth-LDDT pair-distance backward_block_pairs", self.backward_block_pairs)
        _require_num_warps("smooth-LDDT pair-distance", self.num_warps)


@dataclass(frozen=True, slots=True)
class SmoothLddtPairDistancePlan:
    problem: SmoothLddtPairDistanceProblem
    config: SmoothLddtPairDistanceConfig
    total_pairs: int
    output_shape: tuple[int, ...]
    forward_grid: tuple[int] | None
    backward_grid: tuple[int, int] | None
    forward_wide_offsets: bool
    backward_wide_offsets: bool
    inversion_dtype: torch.dtype
    backward_block_rows: int = 1

    def grid_fits(self) -> bool:
        if self.forward_grid is None or self.backward_grid is None:
            return self.forward_grid is self.backward_grid is None
        return _grid_1d_fits(self.forward_grid) & _grid_2d_fits(self.backward_grid)


@dataclass(frozen=True, slots=True)
class SmoothLddtPairDistanceSchedule:
    config: SmoothLddtPairDistanceConfig

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: SmoothLddtPairDistanceProblem) -> SmoothLddtPairDistancePlan:
        plan = _make_smooth_lddt_pair_distance_plan(problem, self.config)
        if not plan.grid_fits():
            raise ValueError("smooth-LDDT pair distance has no launchable plan")
        return plan

    def supports(self, problem: SmoothLddtPairDistanceProblem) -> bool:
        if problem.pairs == 0:
            return True
        return (
            (problem.batch * problem.pairs <= CUDA_MAX_GRID_X * self.config.forward_block_pairs)
            & (problem.atoms <= CUDA_MAX_GRID_X)
            & (problem.batch <= CUDA_MAX_GRID_YZ)
        )


def _make_smooth_lddt_pair_distance_plan(
    problem: SmoothLddtPairDistanceProblem,
    config: SmoothLddtPairDistanceConfig,
) -> SmoothLddtPairDistancePlan:
    total_pairs = problem.batch * problem.pairs
    backward_rows = 8 if problem.atoms >= 1024 and 2 * problem.pairs <= 32 * problem.atoms else 1
    coordinates_numel = problem.batch * problem.atoms * 3
    output_shape = (problem.pairs,) if problem.unbatched else (problem.batch, problem.pairs)
    if problem.pairs == 0:
        return SmoothLddtPairDistancePlan(
            problem,
            config,
            total_pairs,
            output_shape,
            None,
            None,
            False,
            False,
            _inversion_dtype(problem.atoms, problem.pairs),
        )
    coordinate_padding = 0 if problem.unbatched else (config.forward_block_pairs - 1) * problem.atoms * 3
    backward_padding = config.backward_block_pairs - 2
    return SmoothLddtPairDistancePlan(
        problem=problem,
        config=config,
        total_pairs=total_pairs,
        output_shape=output_shape,
        forward_grid=(_ceil_div(total_pairs, config.forward_block_pairs),),
        backward_grid=(_ceil_div(problem.atoms, backward_rows), problem.batch),
        forward_wide_offsets=_needs_wide_offsets(
            _padded_program_offset_upper_bound(total_pairs, config.forward_block_pairs),
            2 * problem.pairs - 1,
            coordinates_numel - 1 + coordinate_padding,
        ),
        backward_wide_offsets=_needs_wide_offsets(
            coordinates_numel - 1,
            2 * problem.pairs - 1,
            2 * problem.pairs + backward_padding,
            problem.atoms,
            total_pairs - 1,
            coordinates_numel - 1,
        ),
        inversion_dtype=_inversion_dtype(problem.atoms, problem.pairs),
        backward_block_rows=backward_rows,
    )


@dataclass(frozen=True, slots=True)
class SmoothLddtSparseProblem:
    batch: int
    atoms: int
    pairs: int
    truth_batched: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(batch=self.batch, atoms=self.atoms)
        if self.pairs < 0:
            raise ValueError("smooth-LDDT sparse pair count must be non-negative")


@dataclass(frozen=True, slots=True)
class SmoothLddtSparseConfig:
    forward_block_pairs: int = 64
    backward_block_pairs: int = 128
    num_warps: int = 4

    def __post_init__(self) -> None:
        _require_power_of_two("smooth-LDDT sparse forward_block_pairs", self.forward_block_pairs)
        _require_power_of_two("smooth-LDDT sparse backward_block_pairs", self.backward_block_pairs)
        _require_num_warps("smooth-LDDT sparse", self.num_warps)


@dataclass(frozen=True, slots=True)
class SmoothLddtSparsePlan:
    problem: SmoothLddtSparseProblem
    config: SmoothLddtSparseConfig
    total_pairs: int
    true_batch_stride: int
    forward_grid: tuple[int] | None
    finalize_grid: tuple[int] | None
    backward_grid: tuple[int, int] | None
    forward_wide_offsets: bool
    backward_order_wide: bool
    inversion_dtype: torch.dtype
    backward_block_rows: int = 1

    def grid_fits(self) -> bool:
        if self.forward_grid is None or self.finalize_grid is None or self.backward_grid is None:
            return self.forward_grid is self.finalize_grid is self.backward_grid is None
        return _grid_1d_fits(self.forward_grid) & (self.finalize_grid == (1,)) & _grid_2d_fits(self.backward_grid)


@dataclass(frozen=True, slots=True)
class SmoothLddtSparseSchedule:
    config: SmoothLddtSparseConfig

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: SmoothLddtSparseProblem) -> SmoothLddtSparsePlan:
        plan = _make_smooth_lddt_sparse_plan(problem, self.config)
        if not plan.grid_fits():
            raise ValueError("smooth-LDDT sparse has no launchable plan")
        return plan

    def supports(self, problem: SmoothLddtSparseProblem) -> bool:
        if problem.pairs == 0:
            return True
        return (
            (problem.batch * problem.pairs <= CUDA_MAX_GRID_X * self.config.forward_block_pairs)
            & (problem.atoms <= CUDA_MAX_GRID_X)
            & (problem.batch <= CUDA_MAX_GRID_YZ)
        )


def _make_smooth_lddt_sparse_plan(
    problem: SmoothLddtSparseProblem,
    config: SmoothLddtSparseConfig,
) -> SmoothLddtSparsePlan:
    total_pairs = problem.batch * problem.pairs
    backward_rows = 8 if problem.atoms >= 1024 and 2 * problem.pairs <= 32 * problem.atoms else 1
    true_batch_stride = problem.pairs if problem.truth_batched else 0
    if problem.pairs == 0:
        return SmoothLddtSparsePlan(
            problem,
            config,
            total_pairs,
            true_batch_stride,
            None,
            None,
            None,
            False,
            False,
            _inversion_dtype(problem.atoms, problem.pairs),
        )
    coordinate_padding = (config.forward_block_pairs - 1) * problem.atoms * 3
    padded_pair = _padded_program_offset_upper_bound(total_pairs, config.forward_block_pairs)
    return SmoothLddtSparsePlan(
        problem=problem,
        config=config,
        total_pairs=total_pairs,
        true_batch_stride=true_batch_stride,
        forward_grid=(_ceil_div(total_pairs, config.forward_block_pairs),),
        finalize_grid=(1,),
        backward_grid=(_ceil_div(problem.atoms, backward_rows), problem.batch),
        forward_wide_offsets=_needs_wide_offsets(
            padded_pair,
            2 * problem.pairs - 1,
            problem.batch * problem.atoms * 3 - 1 + coordinate_padding,
            padded_pair if problem.truth_batched else problem.pairs - 1,
        ),
        backward_order_wide=(2 * problem.pairs + config.backward_block_pairs - 2 > INT32_MAX),
        inversion_dtype=_inversion_dtype(problem.atoms, problem.pairs),
        backward_block_rows=backward_rows,
    )


@dataclass(frozen=True, slots=True)
class SmoothLddtSparseCoordsProblem:
    batch: int
    atoms: int
    pairs: int
    truth_batched: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(batch=self.batch, atoms=self.atoms)
        if self.pairs < 0:
            raise ValueError("smooth-LDDT sparse-coordinate pair count must be non-negative")


@dataclass(frozen=True, slots=True)
class SmoothLddtSparseCoordsPlan:
    pair_distance_plan: SmoothLddtPairDistancePlan
    sparse_loss_plan: SmoothLddtSparsePlan


@dataclass(frozen=True, slots=True)
class SmoothLddtSparseCoordsSchedule:
    pair_distance_schedule: SmoothLddtPairDistanceSchedule
    sparse_loss_schedule: SmoothLddtSparseSchedule

    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: SmoothLddtSparseCoordsProblem) -> SmoothLddtSparseCoordsPlan:
        pair_problem = SmoothLddtPairDistanceProblem(
            problem.batch if problem.truth_batched else 1,
            problem.atoms,
            problem.pairs,
            not problem.truth_batched,
        )
        sparse_problem = SmoothLddtSparseProblem(
            problem.batch,
            problem.atoms,
            problem.pairs,
            problem.truth_batched,
        )
        return SmoothLddtSparseCoordsPlan(
            self.pair_distance_schedule.resolve(pair_problem),
            self.sparse_loss_schedule.resolve(sparse_problem),
        )

    def supports(self, problem: SmoothLddtSparseCoordsProblem) -> bool:
        pair_problem = SmoothLddtPairDistanceProblem(
            problem.batch if problem.truth_batched else 1,
            problem.atoms,
            problem.pairs,
            not problem.truth_batched,
        )
        sparse_problem = SmoothLddtSparseProblem(
            problem.batch,
            problem.atoms,
            problem.pairs,
            problem.truth_batched,
        )
        return self.pair_distance_schedule.supports(pair_problem) & self.sparse_loss_schedule.supports(sparse_problem)


def _padded_program_offset_upper_bound(items: int, block_size: int) -> int:
    return items + block_size - 2


def _needs_wide_offsets(*max_offsets: int) -> bool:
    return max(max_offsets) > INT32_MAX


def _inversion_dtype(atoms: int, pairs: int) -> torch.dtype:
    return torch.int32 if atoms <= INT32_MAX and 2 * pairs <= INT32_MAX else torch.int64


def _smooth_num_warps(block_size: int) -> int:
    if block_size >= 256:
        return 4
    if block_size >= 64:
        return 2
    return 1


def _require_num_warps(name: str, value: int) -> None:
    if value not in (1, 2, 4, 8):
        raise ValueError(f"{name} num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class SmoothLddtDenseRule:
    config: SmoothLddtDenseSchedule
    min_atoms: int = 0
    dtypes: tuple[torch.dtype, ...] | None = None

    def can_implement(self, problem: SmoothLddtDenseProblem) -> bool:
        return self.config.supports(problem) and (self.dtypes is None or problem.dtype in self.dtypes)

    def is_preferred(self, problem: SmoothLddtDenseProblem) -> bool:
        return self.can_implement(problem) and problem.atoms >= self.min_atoms


@dataclass(frozen=True, slots=True)
class SmoothLddtMaskRule:
    config: SmoothLddtMaskSchedule
    min_atoms: int = 0
    dtypes: tuple[torch.dtype, ...] | None = None

    def can_implement(self, problem: SmoothLddtMaskProblem) -> bool:
        return self.config.supports(problem) and (self.dtypes is None or problem.dtype in self.dtypes)

    def is_preferred(self, problem: SmoothLddtMaskProblem) -> bool:
        return self.can_implement(problem) and problem.atoms >= self.min_atoms
