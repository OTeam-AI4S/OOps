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
from typing import Generic, TypeVar

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_false

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
from oops.runtime import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ, dot_tile_shared_bytes

_ProblemT = TypeVar("_ProblemT")
_ConfigT = TypeVar("_ConfigT")


@dataclass(frozen=True, slots=True)
class GateMultiplyProblem:

    rows: int
    heads: int
    head_size: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        validate_positive_parameters(heads=self.heads, head_size=self.head_size)

    @property
    def gate_channels(self) -> int:
        return self.heads * self.head_size


@dataclass(frozen=True, slots=True)
class GateProjectionProblem:

    rows: int
    single_channels: int
    heads: int
    head_size: int
    dtype: torch.dtype
    input_precision: str

    def __post_init__(self) -> None:
        validate_positive_parameters(single_channels=self.single_channels, heads=self.heads, head_size=self.head_size)

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.dtype

    @property
    def gate_channels(self) -> int:
        return self.heads * self.head_size

    @property
    def multiply_accumulates(self) -> int:

        return self.rows * self.single_channels * self.gate_channels

    @property
    def intensity(self) -> float:

        return self.single_channels * self.gate_channels / (self.single_channels + self.gate_channels)


@dataclass(frozen=True, slots=True)
class GateMultiplyConfig:

    block_rows: int
    block_columns: int
    num_warps: int

    def __post_init__(self) -> None:
        validate_positive_parameters(num_warps=self.num_warps)
        _require_power_of_two("gate block_rows", self.block_rows)
        _require_power_of_two("gate block_columns", self.block_columns)

    def blocks(self, gate_channels: int) -> tuple[int, int]:
        columns = min(self.block_columns, _next_power_of_two(max(gate_channels, 1)))
        return self.block_rows * (self.block_columns // columns), columns

    def grid(self, rows: int, gate_channels: int) -> tuple[int, int]:
        block_rows, block_columns = self.blocks(gate_channels)
        return _ceil_div(rows, block_rows), _ceil_div(gate_channels, block_columns)


@dataclass(frozen=True, slots=True)
class GateProjectionConfig:

    block_rows: int = 64
    block_gate: int = 32
    block_reduction: int = 64
    num_warps: int = 4
    num_stages: int = 3

    def __post_init__(self) -> None:
        validate_positive_parameters(num_warps=self.num_warps, num_stages=self.num_stages)
        for name in ("block_rows", "block_gate", "block_reduction"):
            value = getattr(self, name)
            _require_power_of_two(f"gate {name}", value)
            if value < 16:
                raise ValueError(f"gate projection {name} must be at least 16 for tl.dot; got {value}")

    def staged_bytes(self, element_size: int) -> int:

        return dot_tile_shared_bytes(
            self.block_rows, self.block_reduction, self.block_gate, element_size, 1, buffers=self.num_stages
        )

    def grid(self, rows: int, gate_channels: int) -> tuple[int, int]:
        return _ceil_div(rows, self.block_rows), _ceil_div(gate_channels, self.block_gate)


@dataclass(frozen=True, slots=True)
class GateProjectionRule:

    config: GateProjectionConfig
    dtypes: tuple[torch.dtype, ...] | None = None
    min_values: int = 0

    def can_implement(self, problem: GateProjectionProblem) -> bool:
        return self.dtypes is None or problem.dtype in self.dtypes

    def is_preferred(self, problem: GateProjectionProblem) -> bool:
        return self.can_implement(problem) and problem.rows * problem.gate_channels >= self.min_values


@dataclass(frozen=True, slots=True)
class GateNativeLinearRule:

    dtypes: tuple[torch.dtype, ...]
    input_precisions: tuple[str, ...] | None = None
    min_intensity: float = 0.0
    min_multiply_accumulates: int = 0

    def applies(self, problem: GateProjectionProblem) -> bool:
        return (
            problem.dtype in self.dtypes
            and (self.input_precisions is None or problem.input_precision in self.input_precisions)
            and problem.intensity >= self.min_intensity
            and problem.multiply_accumulates >= self.min_multiply_accumulates
        )


@dataclass(frozen=True, slots=True)
class GateMultiplyRule:

    config: GateMultiplyConfig
    dtypes: tuple[torch.dtype, ...] | None = None
    min_values: int = 0

    def can_implement(self, problem: GateMultiplyProblem) -> bool:
        return self.dtypes is None or problem.dtype in self.dtypes

    def is_preferred(self, problem: GateMultiplyProblem) -> bool:
        return self.can_implement(problem) and problem.rows * problem.gate_channels >= self.min_values


@dataclass(frozen=True, slots=True)
class GatePhaseProfile(Generic[_ProblemT, _ConfigT]):
    generic: LaunchProfile[_ProblemT, _ConfigT]
    target_deltas: tuple[TargetProfile[LaunchProfileDelta[_ProblemT, _ConfigT]], ...] = ()

    def select(self, problem: _ProblemT, target: TargetDescriptor | None) -> _ConfigT:
        profile = select_target_launch_profile(self.generic, target=target, deltas=self.target_deltas)
        return select_launch_config(problem, profile)

    def candidates(self, problem: _ProblemT) -> tuple[_ConfigT, ...]:
        return select_launch_candidates(problem, self.generic, target_deltas=self.target_deltas)

    def target_changes_config(self, problem: _ProblemT) -> bool:
        return target_delta_changes_config(problem, self.generic, self.target_deltas)


@dataclass(frozen=True, slots=True)
class GateAttentionOutputScheduleCatalog:
    projection: GatePhaseProfile[GateProjectionProblem, GateProjectionConfig]
    native_linear: tuple[GateNativeLinearRule, ...]
    multiply_forward: GatePhaseProfile[GateMultiplyProblem, GateMultiplyConfig]
    multiply_backward: GatePhaseProfile[GateMultiplyProblem, GateMultiplyConfig]

    def fuses(self, problem: GateProjectionProblem) -> bool:

        return not any(rule.applies(problem) for rule in self.native_linear)

    def projection_admitted(self, problem: GateProjectionProblem, shared_memory_bytes: int) -> bool:

        configs = self.projection.candidates(problem)
        if any(config.staged_bytes(problem.dtype.itemsize) > shared_memory_bytes for config in configs):
            return False
        multiply = GateMultiplyProblem(problem.rows, problem.heads, problem.head_size, problem.dtype)
        grids = (
            *(config.grid(problem.rows, problem.gate_channels) for config in configs),
            *(
                tile.grid(multiply.rows, multiply.gate_channels)
                for profile in (self.multiply_forward, self.multiply_backward)
                for tile in profile.candidates(multiply)
            ),
        )
        launchable = True
        for rows, columns in grids:
            launchable = launchable & (rows <= CUDA_MAX_GRID_X) & (columns <= CUDA_MAX_GRID_YZ)
        if launchable is True or launchable is False:
            return launchable
        if statically_known_false(launchable):
            return False
        torch._check(launchable)
        return True
