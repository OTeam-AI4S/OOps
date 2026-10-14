from __future__ import annotations

from dataclasses import dataclass

from oops.core import (
    LaunchProfile,
    LaunchProfileDelta,
    TargetDescriptor,
    TargetProfile,
    select_launch_config,
    select_target_launch_profile,
    target_delta_changes_config,
)
from oops.runtime import CUDA_MAX_GRID_X

from .._abi import PACKED_MASK_BLOCK_MSA, PACKED_MASK_BLOCK_TOKENS, PACKED_PAIR_MASK_BLOCK_TOKENS


@dataclass(frozen=True, slots=True)
class PackedPairMaskProblem:

    samples: int
    total_tokens: int
    total_pairs: int
    total_tiles: int

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("packed pair mask requires a non-empty wrapper batch")
        if min(self.total_tokens, self.total_pairs, self.total_tiles) < 0:
            raise ValueError("packed pair mask extents must be non-negative")


@dataclass(frozen=True, slots=True)
class PackedPairMaskConfig:

    block_tokens: int = PACKED_PAIR_MASK_BLOCK_TOKENS
    num_warps: int = 4

    def __post_init__(self) -> None:
        if self.block_tokens != PACKED_PAIR_MASK_BLOCK_TOKENS:
            raise ValueError(f"packed pair mask metadata ABI requires block_tokens={PACKED_PAIR_MASK_BLOCK_TOKENS}")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("packed pair mask num_warps must be 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class PackedPairMaskPlan:
    grid: tuple[int]
    config: PackedPairMaskConfig

    def grid_fits(self) -> bool:
        return 0 <= self.grid[0] <= CUDA_MAX_GRID_X

    @property
    def block_tokens(self) -> int:
        return self.config.block_tokens

    @property
    def num_warps(self) -> int:
        return self.config.num_warps


@dataclass(frozen=True, slots=True, eq=False)
class PackedPairMaskSchedule:

    generic: LaunchProfile[PackedPairMaskProblem, PackedPairMaskConfig]
    target_deltas: tuple[TargetProfile[LaunchProfileDelta[PackedPairMaskProblem, PackedPairMaskConfig]], ...] = ()

    def resolve(
        self,
        problem: PackedPairMaskProblem,
        *,
        target: TargetDescriptor | None,
    ) -> PackedPairMaskPlan:
        profile = select_target_launch_profile(self.generic, target=target, deltas=self.target_deltas)
        plan = PackedPairMaskPlan((problem.total_tiles,), select_launch_config(problem, profile))
        if not plan.grid_fits():
            raise ValueError("packed pair mask has no launchable plan")
        return plan

    def target_changes_plan(self, problem: PackedPairMaskProblem) -> bool:
        return target_delta_changes_config(problem, self.generic, self.target_deltas)


@dataclass(frozen=True, slots=True)
class PackedMsaAnyProblem:

    samples: int
    total_cells: int
    total_tokens: int
    num_input_tiles: int
    num_output_tiles: int

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("packed MSA any requires a non-empty wrapper batch")
        if min(self.total_cells, self.total_tokens, self.num_input_tiles, self.num_output_tiles) < 0:
            raise ValueError("packed MSA any extents must be non-negative")


@dataclass(frozen=True, slots=True)
class PackedMsaAnyConfig:

    block_msa: int = PACKED_MASK_BLOCK_MSA
    block_tokens: int = PACKED_MASK_BLOCK_TOKENS
    partial_num_warps: int = 4
    output_num_warps: int = 4

    def __post_init__(self) -> None:
        if self.block_msa != PACKED_MASK_BLOCK_MSA or self.block_tokens != PACKED_MASK_BLOCK_TOKENS:
            raise ValueError(
                "packed MSA any metadata ABI requires " f"{PACKED_MASK_BLOCK_MSA}x{PACKED_MASK_BLOCK_TOKENS} tiles"
            )
        if self.partial_num_warps not in (1, 2, 4, 8) or self.output_num_warps not in (1, 2, 4, 8):
            raise ValueError("packed MSA any num_warps must be 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class PackedMsaAnyPlan:
    partial_grid: tuple[int]
    output_grid: tuple[int]
    config: PackedMsaAnyConfig

    def grid_fits(self) -> bool:
        return all(0 <= grid[0] <= CUDA_MAX_GRID_X for grid in (self.partial_grid, self.output_grid))


@dataclass(frozen=True, slots=True, eq=False)
class PackedMsaAnySchedule:

    generic: LaunchProfile[PackedMsaAnyProblem, PackedMsaAnyConfig]
    target_deltas: tuple[TargetProfile[LaunchProfileDelta[PackedMsaAnyProblem, PackedMsaAnyConfig]], ...] = ()

    def resolve(
        self,
        problem: PackedMsaAnyProblem,
        *,
        target: TargetDescriptor | None,
    ) -> PackedMsaAnyPlan:
        profile = select_target_launch_profile(self.generic, target=target, deltas=self.target_deltas)
        config = select_launch_config(problem, profile)
        plan = PackedMsaAnyPlan((problem.num_input_tiles,), (problem.num_output_tiles,), config)
        if not plan.grid_fits():
            raise ValueError("packed MSA any has no launchable plan")
        return plan

    def target_changes_plan(self, problem: PackedMsaAnyProblem) -> bool:
        return target_delta_changes_config(problem, self.generic, self.target_deltas)
