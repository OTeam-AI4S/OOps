from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
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
    target_delta_changes_config,
    validate_positive_parameters,
)
from oops.core._math import _ceil_div, _next_power_of_two, _require_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, HALF_DTYPES, INT32_MAX
from oops.runtime.cuda import _grid_1d_fits, _grid_2d_fits, _optional_grid_2d_fits


class WindowBackwardKind(IntEnum):
    ATOMIC = 0
    REGULAR = 1


@dataclass(frozen=True, slots=True)
class TokenPairWindowProblem:
    batch: int
    tokens: int
    atoms: int
    channels: int
    n_queries: int
    n_keys: int
    atoms_per_token: int | None
    dtype: torch.dtype

    def __post_init__(self) -> None:
        validate_positive_parameters(
            batch=self.batch,
            tokens=self.tokens,
            atoms=self.atoms,
            channels=self.channels,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
        )
        if self.n_keys < self.n_queries:
            raise ValueError("atom-window n_keys must be at least n_queries")
        if self.n_queries % 2 or self.n_keys % 2:
            raise ValueError("atom-window sizes must be even")
        if self.atoms_per_token is not None:
            if self.atoms_per_token <= 0:
                raise ValueError("atom-window atoms_per_token must be positive")
            if self.atoms != self.tokens * self.atoms_per_token:
                raise ValueError("regular atom-window atoms must equal tokens * atoms_per_token")


@dataclass(frozen=True, slots=True)
class TokenPairWindowConfig:

    block_elements: int = 1024
    num_warps: int = 4
    atomic_block_elements: int = 1024
    atomic_num_warps: int = 4
    atomic_block_points: int = 4
    regular_block_channels: int = 64
    regular_min_atoms_per_token: int = 4
    regular_max_key_slots: int = 64
    regular_max_tile_elements: int = 1 << 20
    regular_num_warps: int = 1

    def __post_init__(self) -> None:
        _require_power_of_two("atom-window block_elements", self.block_elements)
        _require_power_of_two("atom-window atomic_block_elements", self.atomic_block_elements)
        _require_power_of_two("atom-window atomic_block_points", self.atomic_block_points)
        _require_power_of_two("atom-window regular_block_channels", self.regular_block_channels)
        validate_positive_parameters(
            regular_min_atoms_per_token=self.regular_min_atoms_per_token,
            regular_max_key_slots=self.regular_max_key_slots,
            regular_max_tile_elements=self.regular_max_tile_elements,
        )
        for name, value in (
            ("num_warps", self.num_warps),
            ("atomic_num_warps", self.atomic_num_warps),
            ("regular_num_warps", self.regular_num_warps),
        ):
            if value not in (1, 2, 4, 8):
                raise ValueError(f"atom-window {name} must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class TokenPairWindowForwardPlan:
    grid: tuple[int]
    flat_output_shape: tuple[int, int, int, int, int]
    total_out: int
    n_blocks: int
    pad_left: int
    block_elements: int
    num_warps: int

    def grid_fits(self) -> bool:
        return _grid_1d_fits(self.grid)


@dataclass(frozen=True, slots=True)
class TokenPairWindowAtomicBackwardPlan:
    kind: WindowBackwardKind
    grid: tuple[int]
    block_elements: int
    num_warps: int
    block_points: int = 0
    block_channels: int = 0

    def grid_fits(self) -> bool:
        return self.kind is WindowBackwardKind.ATOMIC and _grid_1d_fits(self.grid)


@dataclass(frozen=True, slots=True)
class TokenPairWindowRegularBackwardPlan:
    kind: WindowBackwardKind
    zero_grid: tuple[int, int]
    reduce_grid: tuple[int, int] | None
    max_key_slots: int
    block_channels: int
    block_atoms: int
    wide_offsets: bool
    num_warps: int

    def grid_fits(self) -> bool:
        return (
            (self.kind is WindowBackwardKind.REGULAR)
            & _grid_2d_fits(self.zero_grid)
            & (self.reduce_grid is None or _grid_2d_fits(self.reduce_grid))
        )


@dataclass(frozen=True, slots=True)
class TokenPairWindowPlan:
    problem: TokenPairWindowProblem
    config: TokenPairWindowConfig
    forward: TokenPairWindowForwardPlan
    backward: TokenPairWindowAtomicBackwardPlan | TokenPairWindowRegularBackwardPlan
    accumulation_dtype: torch.dtype

    def grid_fits(self) -> bool:
        return self.forward.grid_fits() & self.backward.grid_fits()


@dataclass(frozen=True, slots=True)
class TokenPairWindowSchedule:
    config: TokenPairWindowConfig

    def resolve(self, problem: TokenPairWindowProblem) -> TokenPairWindowPlan:
        if not torch.compiler.is_compiling():
            return _cached_token_pair_window_plan(self, problem)
        return _resolve_token_pair_window_plan(self, problem)

    def supports(self, problem: TokenPairWindowProblem) -> bool:
        n_blocks = _ceil_div(problem.atoms, problem.n_queries)
        total_out = problem.batch * n_blocks * problem.n_queries * problem.n_keys * problem.channels
        block = min(self.config.block_elements, self.config.atomic_block_elements)
        return total_out <= CUDA_MAX_GRID_X * block


@dataclass(frozen=True, slots=True)
class NestedTokenPairWindowProblem:

    num_samples: int
    total_pair_positions: int
    total_atoms: int
    total_blocks: int
    channels: int
    n_queries: int
    n_keys: int
    dtype: torch.dtype
    has_num_valid_atoms: bool

    def __post_init__(self) -> None:
        validate_positive_parameters(
            num_samples=self.num_samples,
            channels=self.channels,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
        )
        if self.total_pair_positions < 0:
            raise ValueError("nested atom-window total_pair_positions must be non-negative")
        if self.total_atoms < 0:
            raise ValueError("nested atom-window total_atoms must be non-negative")
        if self.total_blocks < 0:
            raise ValueError("nested atom-window total_blocks must be non-negative")
        if self.n_keys < self.n_queries:
            raise ValueError("nested atom-window n_keys must be at least n_queries")
        if self.n_queries % 2 or self.n_keys % 2:
            raise ValueError("nested atom-window sizes must be even")


@dataclass(frozen=True, slots=True)
class NestedTokenPairWindowConfig:

    block_elements: int = 1024
    num_warps: int = 4

    def __post_init__(self) -> None:
        _require_power_of_two("nested atom-window block_elements", self.block_elements)
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("nested atom-window num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class NestedTokenPairWindowRule:

    config: NestedTokenPairWindowConfig
    dtypes: tuple[torch.dtype, ...]
    channels: int
    n_queries: int
    n_keys: int
    has_num_valid_atoms: bool

    def can_implement(self, problem: NestedTokenPairWindowProblem) -> bool:
        return (
            problem.dtype in self.dtypes
            and problem.channels == self.channels
            and problem.n_queries == self.n_queries
            and problem.n_keys == self.n_keys
            and problem.has_num_valid_atoms is self.has_num_valid_atoms
        )

    def is_preferred(self, problem: NestedTokenPairWindowProblem) -> bool:
        return self.can_implement(problem)


@dataclass(frozen=True, slots=True)
class NestedTokenPairWindowPlan:

    problem: NestedTokenPairWindowProblem
    grid: tuple[int, int]
    block_elements: int
    num_warps: int
    accumulation_dtype: torch.dtype

    def grid_fits(self) -> bool:
        return _optional_grid_2d_fits(self.grid)


@dataclass(frozen=True, slots=True)
class NestedTokenPairWindowSchedule:

    generic: LaunchProfile[NestedTokenPairWindowProblem, NestedTokenPairWindowConfig]
    target_deltas: tuple[
        TargetProfile[LaunchProfileDelta[NestedTokenPairWindowProblem, NestedTokenPairWindowConfig]],
        ...,
    ] = ()

    def resolve(
        self,
        problem: NestedTokenPairWindowProblem,
        *,
        target: TargetDescriptor | None = None,
    ) -> NestedTokenPairWindowPlan:
        profile = select_target_launch_profile(self.generic, target=target, deltas=self.target_deltas)
        config = select_launch_config(problem, profile)
        plan = _make_nested_token_pair_window_plan(problem, config)
        if not plan.grid_fits():
            raise ValueError("nested token-pair atom window has no launchable plan")
        return plan

    def supports(self, problem: NestedTokenPairWindowProblem) -> bool:
        return all(
            _make_nested_token_pair_window_plan(problem, config).grid_fits()
            for config in select_launch_candidates(
                problem,
                self.generic,
                target_deltas=self.target_deltas,
            )
        )

    def target_changes_plan(self, problem: NestedTokenPairWindowProblem) -> bool:
        return target_delta_changes_config(problem, self.generic, self.target_deltas)


def _resolve_token_pair_window_plan(
    schedule: TokenPairWindowSchedule,
    problem: TokenPairWindowProblem,
) -> TokenPairWindowPlan:
    plan = _make_token_pair_window_plan(problem, schedule.config)
    if not plan.grid_fits():
        raise ValueError("token-pair atom window has no launchable plan")
    return plan


@lru_cache(maxsize=1024)
def _cached_token_pair_window_plan(
    schedule: TokenPairWindowSchedule,
    problem: TokenPairWindowProblem,
) -> TokenPairWindowPlan:
    return _resolve_token_pair_window_plan(schedule, problem)


def _make_token_pair_window_plan(
    problem: TokenPairWindowProblem,
    config: TokenPairWindowConfig,
) -> TokenPairWindowPlan:
    n_blocks = _ceil_div(problem.atoms, problem.n_queries)
    total_out = problem.batch * n_blocks * problem.n_queries * problem.n_keys * problem.channels
    forward = TokenPairWindowForwardPlan(
        grid=(_ceil_div(total_out, config.block_elements),),
        flat_output_shape=(
            problem.batch,
            n_blocks,
            problem.n_queries,
            problem.n_keys,
            problem.channels,
        ),
        total_out=total_out,
        n_blocks=n_blocks,
        pad_left=(problem.n_keys - problem.n_queries) // 2,
        block_elements=config.block_elements,
        num_warps=config.num_warps,
    )
    backward = _make_token_pair_window_backward(problem, config, forward)
    accumulation_dtype = torch.float32 if problem.dtype in HALF_DTYPES else problem.dtype
    return TokenPairWindowPlan(problem, config, forward, backward, accumulation_dtype)


def _make_token_pair_window_backward(
    problem: TokenPairWindowProblem,
    config: TokenPairWindowConfig,
    forward: TokenPairWindowForwardPlan,
) -> TokenPairWindowAtomicBackwardPlan | TokenPairWindowRegularBackwardPlan:
    atoms_per_token = problem.atoms_per_token
    if atoms_per_token is not None and atoms_per_token >= config.regular_min_atoms_per_token:
        max_key_slots = _regular_reduce_key_slots(
            problem.tokens,
            atoms_per_token,
            problem.n_queries,
            problem.n_keys,
        )
        channel_grid = _ceil_div(problem.channels, config.regular_block_channels)
        block_atoms = _next_power_of_two(atoms_per_token)
        zero_grid = (problem.batch * problem.tokens, channel_grid)
        reduce_grid = None if max_key_slots == 0 else (problem.batch * problem.tokens * max_key_slots, channel_grid)
        regular_candidate = (
            (problem.atoms % problem.n_queries == 0)
            & (max_key_slots <= config.regular_max_key_slots)
            & (block_atoms * config.regular_block_channels <= config.regular_max_tile_elements)
            & _grid_2d_fits(zero_grid)
            & (reduce_grid is None or _grid_2d_fits(reduce_grid))
        )
        if regular_candidate:
            return TokenPairWindowRegularBackwardPlan(
                kind=WindowBackwardKind.REGULAR,
                zero_grid=zero_grid,
                reduce_grid=reduce_grid,
                max_key_slots=max_key_slots,
                block_channels=(
                    config.regular_block_channels
                    if torch.compiler.is_compiling() or isinstance(problem.channels, torch.SymInt)
                    else min(config.regular_block_channels, _next_power_of_two(problem.channels))
                ),
                block_atoms=block_atoms,
                wide_offsets=_regular_backward_needs_wide_offsets(
                    problem,
                    forward,
                    config,
                    atoms_per_token,
                    max_key_slots,
                    block_atoms,
                ),
                num_warps=config.regular_num_warps,
            )
    grid = (_ceil_div(forward.total_out, config.atomic_block_elements),)
    block_points = block_channels = 0
    if not torch.compiler.is_compiling() and problem.dtype is torch.float32 and 32 <= problem.channels <= 128:
        grouped_grid = (_ceil_div(forward.total_out // problem.channels, config.atomic_block_points),)
        if _grid_1d_fits(grouped_grid):
            grid = grouped_grid
            block_points = config.atomic_block_points
            block_channels = _next_power_of_two(problem.channels)
    return TokenPairWindowAtomicBackwardPlan(
        kind=WindowBackwardKind.ATOMIC,
        grid=grid,
        block_elements=config.atomic_block_elements,
        num_warps=config.atomic_num_warps,
        block_points=block_points,
        block_channels=block_channels,
    )


def _regular_reduce_key_slots(
    tokens: int,
    atoms_per_token: int,
    n_queries: int,
    n_keys: int,
) -> int:
    span_blocks = (atoms_per_token + 2 * n_queries - 2) // n_queries
    key_atoms = n_keys + (span_blocks - 1) * n_queries
    return min(tokens - 1, _ceil_div(key_atoms, atoms_per_token) + 2)


def _regular_backward_needs_wide_offsets(
    problem: TokenPairWindowProblem,
    forward: TokenPairWindowForwardPlan,
    config: TokenPairWindowConfig,
    atoms_per_token: int,
    max_key_slots: int,
    block_atoms: int,
) -> bool:
    grad_out_upper = (
        forward.total_out
        - 1
        + (max_key_slots * atoms_per_token + block_atoms - 1) * problem.channels
        + config.regular_block_channels
        - 1
    )
    grad_token_pair_upper = (
        problem.batch * problem.tokens * problem.tokens * problem.channels
        - 1
        + max(max_key_slots - 1, 0) * problem.channels
        + config.regular_block_channels
        - 1
    )
    return max(grad_out_upper, grad_token_pair_upper) > INT32_MAX


def _make_nested_token_pair_window_plan(
    problem: NestedTokenPairWindowProblem,
    config: NestedTokenPairWindowConfig,
) -> NestedTokenPairWindowPlan:
    block_width = problem.n_queries * problem.n_keys * problem.channels
    grid = (problem.total_blocks, _ceil_div(block_width, config.block_elements))
    accumulation_dtype = torch.float32 if problem.dtype in HALF_DTYPES else problem.dtype
    return NestedTokenPairWindowPlan(
        problem=problem,
        grid=grid,
        block_elements=config.block_elements,
        num_warps=config.num_warps,
        accumulation_dtype=accumulation_dtype,
    )
