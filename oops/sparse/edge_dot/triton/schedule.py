from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from oops.core import validate_positive_parameters
from oops.core._math import _ceil_div, _next_power_of_two
from oops.runtime import CUDA_MAX_GRID_X


@dataclass(frozen=True, slots=True)
class EdgeDotProblem:
    edges: int
    nodes: int
    groups: int
    channels: int

    def __post_init__(self) -> None:
        validate_positive_parameters(edges=self.edges, groups=self.groups, channels=self.channels)
        if self.nodes < 0:
            raise ValueError(f"nodes must be non-negative; got {self.nodes}")
        if self.channels > 256:
            raise ValueError("edge-dot channels exceed the 256-channel domain")


@dataclass(frozen=True, slots=True)
class EdgeDotPlan:
    problem: EdgeDotProblem
    grid: tuple[int]
    block_outputs: int
    block_channels: int
    num_warps: int

    def grid_fits(self) -> bool:
        return (0 < self.grid[0]) & (self.grid[0] <= CUDA_MAX_GRID_X)


@dataclass(frozen=True, slots=True)
class EdgeDotSchedule:
    @lru_cache(maxsize=1024)  # noqa: B019 - module-owned immutable schedule
    def resolve(self, problem: EdgeDotProblem) -> EdgeDotPlan:
        plan = _make_edge_dot_plan(problem)
        if not plan.grid_fits():
            raise ValueError("edge-dot has no launchable plan")
        return plan

    def supports(self, problem: EdgeDotProblem) -> bool:
        outputs = problem.edges * problem.groups
        block_outputs = _edge_dot_block_outputs(problem.channels)
        return outputs <= CUDA_MAX_GRID_X * block_outputs


def _make_edge_dot_plan(problem: EdgeDotProblem) -> EdgeDotPlan:
    block_channels = _next_power_of_two(problem.channels)
    block_outputs = _edge_dot_block_outputs(block_channels)
    return EdgeDotPlan(
        problem=problem,
        grid=(_ceil_div(problem.edges * problem.groups, block_outputs),),
        block_outputs=block_outputs,
        block_channels=block_channels,
        num_warps=min(max(block_channels // 64, 1), 4),
    )


def _edge_dot_block_outputs(channels: int) -> int:
    if not isinstance(channels, torch.SymInt):
        return 128 if channels <= 16 else 64 if channels <= 64 else 32 if channels <= 128 else 16
    return (
        16
        + torch.sym_ite(channels <= 128, 16, 0)
        + torch.sym_ite(channels <= 64, 32, 0)
        + torch.sym_ite(channels <= 16, 64, 0)
    )
