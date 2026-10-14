from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch

from oops.core import (
    LaunchProfile,
    LaunchProfileDelta,
    TargetDescriptor,
    TargetProfile,
    select_launch_candidates,
    select_launch_config,
    select_target_launch_profile,
    target_delta_changes_config,
    validate_positive_parameters,
)
from oops.core._math import _ceil_div, _next_power_of_two, _require_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ, HALF_DTYPES, INT32_MAX
from oops.runtime.cuda import _grid_1d_fits, _grid_2d_fits, _optional_grid_1d_fits, _optional_grid_2d_fits

AtomToTokenReductionKind = Literal["regular", "interleaved", "sorted"]

_MAX_FP32_WORKSPACE_ELEMENTS = ((1 << 63) - 1) // 4

_MASK_PLAN_KIND = {
    "regular": 0,
    "interleaved": 1,
    "sorted": 2,
}


@dataclass(frozen=True, slots=True)
class AtomToTokenReductionProblem:
    batch: int
    atoms: int
    num_tokens: int
    channels: int
    dtype: torch.dtype
    kind: AtomToTokenReductionKind
    width: int
    has_atom_mask: bool
    mask_batch_repeat: int
    mask_vjp: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(
            batch=self.batch,
            atoms=self.atoms,
            num_tokens=self.num_tokens,
            channels=self.channels,
            width=self.width,
            mask_batch_repeat=self.mask_batch_repeat,
        )
        if self.kind not in _MASK_PLAN_KIND:
            raise ValueError(f"unknown atom-to-token mean kind: {self.kind}")
        if self.batch % self.mask_batch_repeat:
            raise ValueError("mask_batch_repeat must divide the flattened batch")
        if self.mask_vjp and not self.has_atom_mask:
            raise ValueError("mask_vjp requires an atom mask")


@dataclass(frozen=True, slots=True)
class AtomToTokenReductionConfig:
    block_channels: int = 64
    regular_max_width: int = 64
    interleaved_max_width: int = 64
    sorted_max_width: int = 64
    num_warps: int = 2

    def __post_init__(self) -> None:
        _require_power_of_two("atom-to-token block_channels", self.block_channels)
        validate_positive_parameters(
            regular_max_width=self.regular_max_width,
            interleaved_max_width=self.interleaved_max_width,
            sorted_max_width=self.sorted_max_width,
        )
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("atom-to-token num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class AtomToTokenMaskVjpPlan:
    state_grid: tuple[int, int]
    reduce_grid: tuple[int]
    numerator_shape: tuple[int, int, int]
    denominator_shape: tuple[int, int]

    def grid_fits(self) -> bool:
        return _grid_2d_fits(self.state_grid) & _grid_1d_fits(self.reduce_grid)


@dataclass(frozen=True, slots=True)
class AtomToTokenReductionPlan:
    problem: AtomToTokenReductionProblem
    segment_grid: tuple[int, int]
    flat_output_shape: tuple[int, int, int]
    flat_atom_shape: tuple[int, int, int]
    block_channels: int
    block_atoms: int
    num_warps: int
    max_iters: int
    accumulate_fp32: bool
    mask_plan_kind: int
    mask_vjp_plan: AtomToTokenMaskVjpPlan | None

    def grid_fits(self) -> bool:
        if not _grid_2d_fits(self.segment_grid):
            return False
        return self.mask_vjp_plan is None or self.mask_vjp_plan.grid_fits()


@dataclass(frozen=True, slots=True)
class AtomToTokenReductionSchedule:
    config: AtomToTokenReductionConfig

    def resolve(self, problem: AtomToTokenReductionProblem) -> AtomToTokenReductionPlan:
        if not torch.compiler.is_compiling():
            return _cached_atom_to_token_reduce_plan(self, problem)
        return _resolve_atom_to_token_reduce_plan(self, problem)

    def supports(self, problem: AtomToTokenReductionProblem) -> bool:
        supported = (
            _domain_fits(problem, self.config)
            & (problem.batch * problem.num_tokens <= CUDA_MAX_GRID_X)
            & (problem.channels <= CUDA_MAX_GRID_YZ * self.config.block_channels)
        )
        if problem.mask_vjp:
            supported = supported & (problem.batch * problem.atoms <= CUDA_MAX_GRID_X)
        return supported

    def supports_layout(
        self,
        kind: AtomToTokenReductionKind,
        *,
        atoms: int,
        num_tokens: int,
        width: int,
    ) -> bool:

        return _layout_fits(kind, atoms=atoms, num_tokens=num_tokens, width=width, config=self.config)


@dataclass(frozen=True, slots=True)
class BatchedAtomToTokenReductionProblem:
    r"""``mapping_batch_repeat`` is S when the mapping omits the final sample axis, otherwise one."""

    batch: int
    mapping_batch: int
    mapping_batch_repeat: int
    atoms: int
    num_tokens: int
    channels: int
    dtype: torch.dtype
    has_atom_mask: bool
    mask_batch_repeat: int
    mask_vjp: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(
            batch=self.batch,
            mapping_batch=self.mapping_batch,
            mapping_batch_repeat=self.mapping_batch_repeat,
            atoms=self.atoms,
            num_tokens=self.num_tokens,
            channels=self.channels,
            mask_batch_repeat=self.mask_batch_repeat,
        )
        if self.mapping_batch * self.mapping_batch_repeat != self.batch:
            raise ValueError("mapping_batch_repeat must expand mapping_batch exactly to batch")
        if self.batch % self.mask_batch_repeat:
            raise ValueError("mask_batch_repeat must divide the flattened batch")
        if self.mask_vjp and not self.has_atom_mask:
            raise ValueError("batched atom-to-token mask_vjp requires an atom mask")


@dataclass(frozen=True, slots=True)
class BatchedAtomToTokenReductionConfig:
    block_channels: int = 64
    block_atoms: int = 64
    mask_vjp_block_atoms: int = 16
    classify_block_atoms: int = 256
    num_warps: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("batched atom-to-token block_channels", self.block_channels),
            ("batched atom-to-token block_atoms", self.block_atoms),
            ("batched atom-to-token mask_vjp_block_atoms", self.mask_vjp_block_atoms),
            ("batched atom-to-token classify_block_atoms", self.classify_block_atoms),
        ):
            _require_power_of_two(name, value)
        if max(self.block_atoms, self.mask_vjp_block_atoms) * self.block_channels > 1 << 20:
            raise ValueError("batched atom-to-token tile exceeds Triton's block-tensor element limit")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("batched atom-to-token num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class BatchedAtomToTokenReductionPlan:
    problem: BatchedAtomToTokenReductionProblem
    classify_grid: tuple[int]
    segment_grid: tuple[int, int]
    mask_vjp_grid: tuple[int]
    flat_atom_shape: tuple[int, int, int]
    flat_output_shape: tuple[int, int, int]
    mapping_flag_shape: tuple[int]
    numerator_shape: tuple[int, int, int] | None
    denominator_shape: tuple[int, int] | None
    block_channels: int
    block_atoms: int
    mask_vjp_block_atoms: int
    classify_block_atoms: int
    num_warps: int

    def grid_fits(self) -> bool:
        return (
            _grid_1d_fits(self.classify_grid)
            & _grid_2d_fits(self.segment_grid)
            & _optional_grid_1d_fits(self.mask_vjp_grid)
        )


@dataclass(frozen=True, slots=True)
class BatchedAtomToTokenReductionSchedule:

    generic: LaunchProfile[BatchedAtomToTokenReductionProblem, BatchedAtomToTokenReductionConfig]
    target_deltas: tuple[
        TargetProfile[LaunchProfileDelta[BatchedAtomToTokenReductionProblem, BatchedAtomToTokenReductionConfig]],
        ...,
    ] = ()

    def resolve(
        self,
        problem: BatchedAtomToTokenReductionProblem,
        *,
        target: TargetDescriptor | None = None,
    ) -> BatchedAtomToTokenReductionPlan:
        profile = select_target_launch_profile(self.generic, target=target, deltas=self.target_deltas)
        config = select_launch_config(problem, profile)
        plan = _make_batched_atom_to_token_reduce_plan(problem, config)
        if problem.atoms > INT32_MAX or not plan.grid_fits():
            raise ValueError("batched atom-to-token mean has no launchable plan")
        return plan

    def supports(self, problem: BatchedAtomToTokenReductionProblem) -> bool:
        return (problem.atoms <= INT32_MAX) & all(
            _make_batched_atom_to_token_reduce_plan(problem, config).grid_fits()
            for config in select_launch_candidates(
                problem,
                self.generic,
                target_deltas=self.target_deltas,
            )
        )

    def target_changes_plan(self, problem: BatchedAtomToTokenReductionProblem) -> bool:
        return target_delta_changes_config(problem, self.generic, self.target_deltas)


@dataclass(frozen=True, slots=True)
class NestedAtomToTokenReductionProblem:

    num_samples: int
    total_atoms: int
    total_tokens: int
    sample_size: int
    channels: int
    dtype: torch.dtype
    has_atom_mask: bool
    mask_vjp: bool
    shared_mask_vjp: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(
            num_samples=self.num_samples,
            sample_size=self.sample_size,
            channels=self.channels,
        )
        if self.total_atoms < 0:
            raise ValueError("nested atom-to-token total_atoms must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("nested atom-to-token total_tokens must be non-negative")
        if self.mask_vjp and not self.has_atom_mask:
            raise ValueError("nested atom-to-token mask_vjp requires an atom mask")
        if self.shared_mask_vjp and not self.mask_vjp:
            raise ValueError("nested atom-to-token shared_mask_vjp requires mask_vjp")


@dataclass(frozen=True, slots=True)
class NestedAtomToTokenReductionConfig:

    block_channels: int = 64
    block_atoms: int = 64
    classify_block_atoms: int = 256
    finalize_block_samples: int = 128
    mask_reduce_block_samples: int = 128
    num_warps: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("nested atom-to-token block_channels", self.block_channels),
            ("nested atom-to-token block_atoms", self.block_atoms),
            ("nested atom-to-token classify_block_atoms", self.classify_block_atoms),
            (
                "nested atom-to-token finalize_block_samples",
                self.finalize_block_samples,
            ),
            (
                "nested atom-to-token mask_reduce_block_samples",
                self.mask_reduce_block_samples,
            ),
        ):
            _require_power_of_two(name, value)
        if self.block_atoms * self.block_channels > 1 << 20:
            raise ValueError("nested atom-to-token reduction tile exceeds Triton's block-tensor element limit")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("nested atom-to-token num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class NestedAtomToTokenReductionPlan:

    problem: NestedAtomToTokenReductionProblem
    classify_grid: tuple[int]
    finalize_grid: tuple[int]
    descriptor_grid: tuple[int]
    segment_grid: tuple[int, int]
    mask_vjp_grid: tuple[int]
    mask_reduce_grid: tuple[int]
    block_channels: int
    block_atoms: int
    classify_block_atoms: int
    finalize_block_samples: int
    mask_reduce_block_samples: int
    num_warps: int
    token_count_shape: tuple[int]
    sample_flag_shape: tuple[int]
    mask_partial_shape: tuple[int, int] | None

    def grid_fits(self) -> bool:
        one_dimensional = (
            self.classify_grid,
            self.finalize_grid,
            self.descriptor_grid,
            self.mask_vjp_grid,
            self.mask_reduce_grid,
        )
        if not all(_optional_grid_1d_fits(grid) for grid in one_dimensional):
            return False
        if not _optional_grid_2d_fits(self.segment_grid):
            return False
        return (
            self.mask_partial_shape is None
            or self.mask_partial_shape[0] * self.mask_partial_shape[1] <= _MAX_FP32_WORKSPACE_ELEMENTS
        )


@dataclass(frozen=True, slots=True)
class NestedAtomToTokenReductionSchedule:

    generic: LaunchProfile[NestedAtomToTokenReductionProblem, NestedAtomToTokenReductionConfig]
    target_deltas: tuple[
        TargetProfile[LaunchProfileDelta[NestedAtomToTokenReductionProblem, NestedAtomToTokenReductionConfig]],
        ...,
    ] = ()

    def resolve(
        self,
        problem: NestedAtomToTokenReductionProblem,
        *,
        target: TargetDescriptor | None = None,
    ) -> NestedAtomToTokenReductionPlan:
        profile = select_target_launch_profile(self.generic, target=target, deltas=self.target_deltas)
        config = select_launch_config(problem, profile)
        plan = _make_nested_atom_to_token_reduce_plan(problem, config)
        if not plan.grid_fits():
            raise ValueError("nested atom-to-token mean has no launchable plan")
        return plan

    def supports(self, problem: NestedAtomToTokenReductionProblem) -> bool:
        return (
            (problem.total_atoms <= INT32_MAX)
            & (problem.total_tokens <= INT32_MAX)
            & all(
                _make_nested_atom_to_token_reduce_plan(problem, config).grid_fits()
                for config in select_launch_candidates(
                    problem,
                    self.generic,
                    target_deltas=self.target_deltas,
                )
            )
        )

    def target_changes_plan(self, problem: NestedAtomToTokenReductionProblem) -> bool:
        return target_delta_changes_config(problem, self.generic, self.target_deltas)


def _resolve_atom_to_token_reduce_plan(
    schedule: AtomToTokenReductionSchedule,
    problem: AtomToTokenReductionProblem,
) -> AtomToTokenReductionPlan:
    config = schedule.config
    if not _domain_fits(problem, config):
        raise ValueError("atom-to-token mean has no launchable plan")
    plan = _make_plan(problem, config)
    if not plan.grid_fits():
        raise ValueError("atom-to-token mean has no launchable plan")
    return plan


@lru_cache(maxsize=1024)
def _cached_atom_to_token_reduce_plan(
    schedule: AtomToTokenReductionSchedule,
    problem: AtomToTokenReductionProblem,
) -> AtomToTokenReductionPlan:
    return _resolve_atom_to_token_reduce_plan(schedule, problem)


def _make_plan(
    problem: AtomToTokenReductionProblem,
    config: AtomToTokenReductionConfig,
) -> AtomToTokenReductionPlan:
    segment_grid = (
        problem.batch * problem.num_tokens,
        _ceil_div(problem.channels, config.block_channels),
    )
    mask_vjp_plan = (
        AtomToTokenMaskVjpPlan(
            state_grid=segment_grid,
            reduce_grid=(problem.batch * problem.atoms,),
            numerator_shape=(problem.batch, problem.num_tokens, problem.channels),
            denominator_shape=(problem.batch, problem.num_tokens),
        )
        if problem.mask_vjp
        else None
    )
    return AtomToTokenReductionPlan(
        problem=problem,
        segment_grid=segment_grid,
        flat_output_shape=(problem.batch, problem.num_tokens, problem.channels),
        flat_atom_shape=(problem.batch, problem.atoms, problem.channels),
        block_channels=_launch_channel_block(problem.channels, config.block_channels),
        block_atoms=_next_power_of_two(problem.width),
        num_warps=config.num_warps,
        max_iters=max(1, problem.atoms.bit_length()),
        accumulate_fp32=problem.dtype in HALF_DTYPES,
        mask_plan_kind=_MASK_PLAN_KIND[problem.kind],
        mask_vjp_plan=mask_vjp_plan,
    )


def _make_nested_atom_to_token_reduce_plan(
    problem: NestedAtomToTokenReductionProblem,
    config: NestedAtomToTokenReductionConfig,
) -> NestedAtomToTokenReductionPlan:
    mask_vjp_grid = (problem.total_tokens * problem.sample_size,) if problem.mask_vjp else (0,)
    mask_reduce_grid = (problem.total_atoms,) if problem.shared_mask_vjp else (0,)
    mask_partial_shape = (problem.total_atoms, problem.sample_size) if problem.shared_mask_vjp else None
    return NestedAtomToTokenReductionPlan(
        problem=problem,
        classify_grid=(_ceil_div(problem.total_atoms, config.classify_block_atoms),),
        finalize_grid=(_ceil_div(problem.num_samples, _ceil_div(problem.num_samples, CUDA_MAX_GRID_X)),),
        descriptor_grid=(problem.total_tokens,),
        segment_grid=(
            problem.total_tokens * problem.sample_size,
            _ceil_div(problem.channels, config.block_channels),
        ),
        mask_vjp_grid=mask_vjp_grid,
        mask_reduce_grid=mask_reduce_grid,
        block_channels=_launch_channel_block(problem.channels, config.block_channels),
        block_atoms=config.block_atoms,
        classify_block_atoms=config.classify_block_atoms,
        finalize_block_samples=config.finalize_block_samples,
        mask_reduce_block_samples=config.mask_reduce_block_samples,
        num_warps=config.num_warps,
        token_count_shape=(problem.total_tokens,),
        sample_flag_shape=(problem.num_samples,),
        mask_partial_shape=mask_partial_shape,
    )


def _make_batched_atom_to_token_reduce_plan(
    problem: BatchedAtomToTokenReductionProblem,
    config: BatchedAtomToTokenReductionConfig,
) -> BatchedAtomToTokenReductionPlan:
    mask_vjp_grid = (
        (_ceil_div(problem.batch * problem.atoms, config.mask_vjp_block_atoms),) if problem.mask_vjp else (0,)
    )
    return BatchedAtomToTokenReductionPlan(
        problem=problem,
        classify_grid=(_ceil_div(problem.mapping_batch * problem.atoms, config.classify_block_atoms),),
        segment_grid=(
            problem.batch * problem.num_tokens,
            _ceil_div(problem.channels, config.block_channels),
        ),
        mask_vjp_grid=mask_vjp_grid,
        flat_atom_shape=(problem.batch, problem.atoms, problem.channels),
        flat_output_shape=(problem.batch, problem.num_tokens, problem.channels),
        mapping_flag_shape=(problem.mapping_batch,),
        numerator_shape=((problem.batch, problem.num_tokens, problem.channels) if problem.mask_vjp else None),
        denominator_shape=((problem.batch, problem.num_tokens) if problem.mask_vjp else None),
        block_channels=_launch_channel_block(problem.channels, config.block_channels),
        block_atoms=min(config.block_atoms, _next_power_of_two(_ceil_div(problem.atoms, problem.num_tokens))),
        mask_vjp_block_atoms=config.mask_vjp_block_atoms,
        classify_block_atoms=config.classify_block_atoms,
        num_warps=config.num_warps,
    )


def _domain_fits(problem: AtomToTokenReductionProblem, config: AtomToTokenReductionConfig) -> bool:
    return _layout_fits(
        problem.kind,
        atoms=problem.atoms,
        num_tokens=problem.num_tokens,
        width=problem.width,
        config=config,
    )


def _layout_fits(
    kind: AtomToTokenReductionKind,
    *,
    atoms: int,
    num_tokens: int,
    width: int,
    config: AtomToTokenReductionConfig,
) -> bool:

    regular_shape = atoms == num_tokens * width
    if kind == "regular":
        return regular_shape & (width <= config.regular_max_width)
    if kind == "interleaved":
        return regular_shape & (width <= config.interleaved_max_width)
    return width <= config.sorted_max_width


def _launch_channel_block(channels: int, capacity: int) -> int:
    if torch.compiler.is_compiling() or isinstance(channels, torch.SymInt):
        return capacity
    return min(capacity, _next_power_of_two(channels))
