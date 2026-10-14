from __future__ import annotations

from dataclasses import dataclass, replace

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
)
from oops.core._math import _next_power_of_two
from oops.runtime import CUDA_MAX_GRID_X, STANDARD_FLOAT_DTYPES


@dataclass(frozen=True, slots=True)
class MaskedKnnProblem:
    total_points: int
    num_samples: int
    num_neighbors: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if torch.compiler.is_compiling():
            torch._check(self.total_points >= 0)
            torch._check(self.num_samples > 0)
        else:
            if self.total_points < 0:
                raise ValueError("masked kNN total_points must be non-negative")
            if self.num_samples <= 0:
                raise ValueError("masked kNN num_samples must be positive")
        if self.num_neighbors < 1 or self.num_neighbors > 128:
            raise ValueError("masked kNN num_neighbors must lie in [1, 128]")
        if self.dtype not in STANDARD_FLOAT_DTYPES:
            raise ValueError("masked kNN dtype must be float16, bfloat16, or float32")

    @property
    def topk(self) -> int:
        return _next_power_of_two(self.num_neighbors)


@dataclass(frozen=True, slots=True)
class MaskedKnnConfig:

    block_key: int
    num_warps: int

    def __post_init__(self) -> None:
        if self.block_key <= 0 or self.block_key & (self.block_key - 1):
            raise ValueError("masked kNN block_key must be a positive power of two")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("masked kNN num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class MaskedKnnBackwardConfig:

    atomic_block_edges: int
    deterministic_target_block: int
    num_warps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("atomic_block_edges", self.atomic_block_edges),
            ("deterministic_target_block", self.deterministic_target_block),
        ):
            if value <= 0 or value & (value - 1):
                raise ValueError(f"masked kNN {name} must be a positive power of two")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("masked kNN backward num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class MaskedKnnRule:
    config: MaskedKnnConfig
    max_topk: int
    dtypes: tuple[torch.dtype, ...] | None = None
    min_topk: int = 1
    min_points_per_sample: int = 0
    max_total_points: int | None = None

    def __post_init__(self) -> None:
        if self.max_topk <= 0 or self.max_topk & (self.max_topk - 1) or self.max_topk > 128:
            raise ValueError("masked kNN max_topk must be a power of two no larger than 128")
        if self.min_topk < 1 or self.min_topk & (self.min_topk - 1) or self.min_topk > self.max_topk:
            raise ValueError("masked kNN min_topk must be a power of two no larger than max_topk")
        if self.min_points_per_sample < 0:
            raise ValueError("masked kNN min_points_per_sample must be non-negative")
        if self.max_total_points is not None and self.max_total_points < 1:
            raise ValueError("masked kNN max_total_points must be positive or None")
        if self.max_topk > self.config.block_key:
            raise ValueError("masked kNN rule exceeds its config's block_key")
        if self.dtypes is not None and not self.dtypes:
            raise ValueError("masked kNN rule dtypes must be None or non-empty")

    def can_implement(self, problem: MaskedKnnProblem) -> bool:
        return problem.topk <= self.config.block_key and (self.dtypes is None or problem.dtype in self.dtypes)

    def is_preferred(self, problem: MaskedKnnProblem) -> bool:
        return (
            self.can_implement(problem)
            and self.min_topk <= problem.topk <= self.max_topk
            and problem.total_points >= problem.num_samples * self.min_points_per_sample
            and (self.max_total_points is None or problem.total_points <= self.max_total_points)
        )


@dataclass(frozen=True, slots=True)
class MaskedKnnPlan:
    problem: MaskedKnnProblem
    config: MaskedKnnConfig
    grid: tuple[int] | None
    topk: int

    def grid_fits(self) -> bool:
        return self.grid is None or (0 < self.grid[0] <= CUDA_MAX_GRID_X)


@dataclass(frozen=True, slots=True)
class MaskedKnnBackwardPlan:
    problem: MaskedKnnProblem
    config: MaskedKnnBackwardConfig
    atomic_grid: tuple[int] | None
    deterministic_grid: tuple[int] | None
    topk: int

    def grid_fits(self) -> bool:
        return all(
            grid is None or (0 < grid[0] <= CUDA_MAX_GRID_X) for grid in (self.atomic_grid, self.deterministic_grid)
        )


MaskedKnnDeltas = tuple[TargetProfile[LaunchProfileDelta[MaskedKnnProblem, MaskedKnnConfig]], ...]
MaskedKnnBackwardDeltas = tuple[TargetProfile[LaunchProfileDelta[MaskedKnnProblem, MaskedKnnBackwardConfig]], ...]


@dataclass(frozen=True, slots=True, eq=False)
class MaskedKnnSchedule:

    forward: LaunchProfile[MaskedKnnProblem, MaskedKnnConfig]
    backward: LaunchProfile[MaskedKnnProblem, MaskedKnnBackwardConfig]
    forward_deltas: MaskedKnnDeltas = ()
    backward_deltas: MaskedKnnBackwardDeltas = ()

    def resolve(
        self,
        problem: MaskedKnnProblem,
        *,
        target: TargetDescriptor | None,
    ) -> MaskedKnnPlan:
        profile = select_target_launch_profile(self.forward, target=target, deltas=self.forward_deltas)
        plan = _make_plan(problem, select_launch_config(problem, profile))
        if not plan.grid_fits():
            raise ValueError("masked kNN has no launchable forward plan")
        return plan

    def resolve_backward(
        self,
        problem: MaskedKnnProblem,
        *,
        target: TargetDescriptor | None,
    ) -> MaskedKnnBackwardPlan:
        profile = select_target_launch_profile(self.backward, target=target, deltas=self.backward_deltas)
        plan = _make_backward_plan(problem, select_launch_config(problem, profile))
        if not plan.grid_fits():
            raise ValueError("masked kNN has no launchable backward plan")
        return plan

    def supports(self, problem: MaskedKnnProblem) -> bool:
        total_edges = problem.total_points * problem.num_neighbors
        forward = all(
            problem.topk <= config.block_key
            for config in select_launch_candidates(problem, self.forward, target_deltas=self.forward_deltas)
        )
        backward = all(
            total_edges <= CUDA_MAX_GRID_X * config.atomic_block_edges
            for config in select_launch_candidates(problem, self.backward, target_deltas=self.backward_deltas)
        )
        return (problem.total_points <= CUDA_MAX_GRID_X) & forward & backward

    def target_changes_plan(self, problem: MaskedKnnProblem) -> bool:
        return target_delta_changes_config(problem, self.forward, self.forward_deltas)

    def backward_target_changes_plan(self, problem: MaskedKnnProblem) -> bool:
        return target_delta_changes_config(problem, self.backward, self.backward_deltas)


def _make_plan(problem: MaskedKnnProblem, config: MaskedKnnConfig) -> MaskedKnnPlan:
    if not torch.compiler.is_compiling() and not isinstance(problem.total_points, torch.SymInt):
        block_key = max(problem.topk, min(config.block_key, _next_power_of_two(max(problem.total_points, 1))))
        if block_key != config.block_key:
            config = replace(config, block_key=block_key)
    return MaskedKnnPlan(
        problem=problem,
        config=config,
        grid=None if problem.total_points == 0 else (problem.total_points,),
        topk=problem.topk,
    )


def _make_backward_plan(problem: MaskedKnnProblem, config: MaskedKnnBackwardConfig) -> MaskedKnnBackwardPlan:
    total_edges = problem.total_points * problem.num_neighbors
    return MaskedKnnBackwardPlan(
        problem=problem,
        config=config,
        atomic_grid=(
            None if total_edges == 0 else ((total_edges + config.atomic_block_edges - 1) // config.atomic_block_edges,)
        ),
        deterministic_grid=None if problem.total_points == 0 else (problem.total_points,),
        topk=problem.topk,
    )
