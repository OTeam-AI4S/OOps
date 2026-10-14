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
from enum import Enum
from functools import lru_cache
from typing import TypeAlias

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
from oops.core._math import _ceil_div
from oops.core.launch import apply_launch_delta
from oops.runtime import CUDA_MAX_GRID_X, FLOAT32_DOT_TF32X3

from .problem import MIN_BLOCK_CHANNEL, AttentionLaunchContext, AttentionProblem


def fit_block(default_block: int, block_channel: int, element_size: int, budget: int) -> int:
    return min(default_block, max(MIN_BLOCK_CHANNEL, budget // (block_channel * element_size)))


@dataclass(frozen=True, slots=True)
class AttentionStageRule:

    max_block_channel: int
    num_stages: int

    def __post_init__(self) -> None:
        validate_positive_parameters(
            max_block_channel=self.max_block_channel,
            num_stages=self.num_stages,
        )


@dataclass(frozen=True, slots=True)
class AttentionStagePolicy:

    rules: tuple[AttentionStageRule, ...]
    fallback: int

    def __post_init__(self) -> None:
        validate_positive_parameters(fallback=self.fallback)
        boundaries = tuple(rule.max_block_channel for rule in self.rules)
        if boundaries != tuple(sorted(set(boundaries))):
            raise ValueError("attention stage boundaries must be unique and strictly increasing")

    def resolve(self, block_channel: int) -> int:
        for rule in self.rules:
            if block_channel <= rule.max_block_channel:
                return rule.num_stages
        return self.fallback


@dataclass(frozen=True, slots=True)
class AttentionTileConfig:

    name: str
    block_query: int
    block_key: int | None
    num_warps: int
    stage_policy: AttentionStagePolicy
    query_budget_bytes: int | None = None
    key_budget_bytes: int | None = None
    group_parts: int = 1
    group_workspace_bytes: int | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(
            block_query=self.block_query,
            block_key=self.block_key,
            num_warps=self.num_warps,
            query_budget_bytes=self.query_budget_bytes,
            key_budget_bytes=self.key_budget_bytes,
            group_parts=self.group_parts,
            group_workspace_bytes=self.group_workspace_bytes,
        )

    def resolve(self, context: AttentionLaunchContext) -> ResolvedAttentionTile:

        block_channel = context.block_channel
        element_size = context.element_size
        block_query = (
            fit_block(self.block_query, block_channel, element_size, self.query_budget_bytes)
            if self.query_budget_bytes is not None
            else self.block_query
        )
        block_key = None
        if self.block_key is not None:
            block_key = (
                fit_block(self.block_key, block_channel, element_size, self.key_budget_bytes)
                if self.key_budget_bytes is not None
                else self.block_key
            )
        return ResolvedAttentionTile(
            name=self.name,
            block_query=block_query,
            block_key=block_key,
            block_channel=block_channel,
            num_warps=self.num_warps,
            num_stages=self.stage_policy.resolve(block_channel),
            group_parts=self.group_parts,
            group_workspace_bytes=self.group_workspace_bytes,
        )


@dataclass(frozen=True, slots=True)
class ResolvedAttentionTile:

    name: str
    block_query: int
    block_key: int | None
    block_channel: int
    num_warps: int
    num_stages: int
    group_parts: int = 1
    group_workspace_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class AttentionLaunchRule:

    config: AttentionTileConfig
    min_query_length: int = 0
    max_query_length: int | None = None
    min_score_elements: int = 0
    min_heads: int = 0
    max_head_size: int | None = None
    min_block_channel: int = 0
    max_block_channel: int | None = None
    element_sizes: tuple[int, ...] | None = None
    mask_kinds: tuple[int, ...] | None = None
    has_mask: bool | None = None

    def is_preferred(self, context: AttentionLaunchContext) -> bool:

        if not self.can_implement(context):
            return False
        preferred = (
            (context.query_length >= self.min_query_length)
            & (context.score_elements >= self.min_score_elements)
            & (context.num_heads >= self.min_heads)
        )
        if self.max_query_length is not None:
            preferred = preferred & (context.query_length <= self.max_query_length)
        return preferred

    def can_implement(self, context: AttentionLaunchContext) -> bool:

        if self.max_head_size is not None and context.head_size > self.max_head_size:
            return False
        if context.block_channel < self.min_block_channel:
            return False
        if self.max_block_channel is not None and context.block_channel > self.max_block_channel:
            return False
        if self.element_sizes is not None and context.element_size not in self.element_sizes:
            return False
        if self.mask_kinds is not None and context.mask_kind not in self.mask_kinds:
            return False
        return self.has_mask is None or context.has_mask is self.has_mask


_Profile: TypeAlias = LaunchProfile[AttentionLaunchContext, AttentionTileConfig]
_TargetDelta: TypeAlias = TargetProfile[LaunchProfileDelta[AttentionLaunchContext, AttentionTileConfig]]


class AttentionPhase(Enum):

    FORWARD = "forward"
    BACKWARD_PREPROCESS = "backward_preprocess"
    BACKWARD_QUERY = "backward_query"
    BACKWARD_KEY_VALUE = "backward_key_value"
    GRAD_BIAS = "grad_bias"


@dataclass(frozen=True, slots=True)
class AttentionPhaseProfile:

    generic: _Profile
    target_deltas: tuple[_TargetDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class AttentionLaunchPlan:

    phase: AttentionPhase
    work_shape: tuple[int, ...]
    block_shape: tuple[int, ...]
    tile: ResolvedAttentionTile

    @property
    def grid(self) -> tuple[int, ...]:

        return tuple(_ceil_div(extent, block) for extent, block in zip(self.work_shape, self.block_shape, strict=True))

    def grid_fits(self, maximum_grid_x: int, maximum_grid_yz: int) -> bool:

        fits = True
        for axis, (extent, block) in enumerate(zip(self.work_shape, self.block_shape, strict=True)):
            bound = maximum_grid_x if axis == 0 else maximum_grid_yz
            fits = fits & (extent > 0) & (extent <= block * bound)
        return fits


@dataclass(frozen=True, slots=True, eq=False)
class AttentionScheduleCatalog:

    forward: AttentionPhaseProfile
    backward_preprocess: AttentionPhaseProfile
    backward_query: AttentionPhaseProfile
    backward_query_tf32x3: AttentionPhaseProfile
    backward_key_value: AttentionPhaseProfile
    backward_key_value_tf32x3: AttentionPhaseProfile
    grad_bias: AttentionPhaseProfile

    def resolve(
        self,
        problem: AttentionProblem,
        phase: AttentionPhase,
        *,
        precision: str,
        target: TargetDescriptor | None = None,
    ) -> AttentionLaunchPlan:

        if not torch.compiler.is_compiling():
            return _cached_resolve(
                self,
                problem.context,
                problem.batch_size,
                problem.num_groups,
                phase,
                precision,
                target,
            )
        return _resolve(
            self,
            problem.context,
            problem.batch_size,
            problem.num_groups,
            phase,
            precision,
            target,
        )

    def grids_fit(
        self,
        problem: AttentionProblem,
        phase: AttentionPhase,
        *,
        precision: str,
        maximum_grid_x: int,
        maximum_grid_yz: int,
        cache: bool = True,
    ) -> bool:

        if cache and not torch.compiler.is_compiling():
            return _cached_grids_fit(
                self,
                problem.context,
                problem.batch_size,
                problem.num_groups,
                phase,
                precision,
                maximum_grid_x,
                maximum_grid_yz,
            )
        return _grids_fit(
            self,
            problem.context,
            problem.batch_size,
            problem.num_groups,
            phase,
            precision,
            maximum_grid_x,
            maximum_grid_yz,
        )

    def target_changes_plan(
        self,
        problem: AttentionProblem,
        phases: tuple[AttentionPhase, ...],
        *,
        precision: str,
    ) -> bool:

        if not torch.compiler.is_compiling():
            return _cached_target_changes_plan(self, problem.context, phases, precision)
        return _target_changes_plan(self, problem.context, phases, precision)

    def _profile(self, phase: AttentionPhase, precision: str) -> AttentionPhaseProfile:
        if phase is AttentionPhase.FORWARD:
            return self.forward
        if phase is AttentionPhase.BACKWARD_PREPROCESS:
            return self.backward_preprocess
        if phase is AttentionPhase.BACKWARD_QUERY:
            return self.backward_query_tf32x3 if precision == FLOAT32_DOT_TF32X3 else self.backward_query
        if phase is AttentionPhase.BACKWARD_KEY_VALUE:
            return self.backward_key_value_tf32x3 if precision == FLOAT32_DOT_TF32X3 else self.backward_key_value
        if phase is AttentionPhase.GRAD_BIAS:
            return self.grad_bias
        raise ValueError(f"unknown attention phase: {phase}")


def _resolve(
    catalog: AttentionScheduleCatalog,
    context: AttentionLaunchContext,
    batch_size: int,
    num_groups: int,
    phase: AttentionPhase,
    precision: str,
    target: TargetDescriptor | None,
) -> AttentionLaunchPlan:
    profile = catalog._profile(phase, precision)
    selected = select_target_launch_profile(
        profile.generic,
        target=target,
        deltas=profile.target_deltas,
    )
    tile = select_launch_config(context, selected).resolve(context)
    return _make_plan(context, batch_size, num_groups, phase, tile)


@lru_cache(maxsize=4096)
def _cached_resolve(
    catalog: AttentionScheduleCatalog,
    context: AttentionLaunchContext,
    batch_size: int,
    num_groups: int,
    phase: AttentionPhase,
    precision: str,
    target: TargetDescriptor | None,
) -> AttentionLaunchPlan:
    return _resolve(catalog, context, batch_size, num_groups, phase, precision, target)


def _symbolic_not(value: bool) -> bool:
    r"""Do not evaluate symbolic predicates as Python booleans."""

    if value is True:
        return False
    if value is False:
        return True
    return torch.sym_not(value)


def _grids_fit(
    catalog: AttentionScheduleCatalog,
    context: AttentionLaunchContext,
    batch_size: int,
    num_groups: int,
    phase: AttentionPhase,
    precision: str,
    maximum_grid_x: int,
    maximum_grid_yz: int,
) -> bool:
    profile = catalog._profile(phase, precision)
    profiles = (
        profile.generic,
        *(apply_launch_delta(profile.generic, delta.value) for delta in profile.target_deltas),
    )
    legal = True
    for candidate in profiles:
        default_plan = _make_plan(context, batch_size, num_groups, phase, candidate.default.resolve(context))
        rule_plans = tuple(
            (rule, _make_plan(context, batch_size, num_groups, phase, rule.config.resolve(context)))
            for rule in candidate.rules
        )
        if all(plan.block_shape == default_plan.block_shape for _rule, plan in rule_plans):
            legal = legal & default_plan.grid_fits(maximum_grid_x, maximum_grid_yz)
            continue
        unselected = True
        for rule, plan in rule_plans:
            preferred = rule.is_preferred(context)
            active = unselected & preferred
            legal = legal & (_symbolic_not(active) | plan.grid_fits(maximum_grid_x, maximum_grid_yz))
            unselected = unselected & _symbolic_not(preferred)
        legal = legal & (_symbolic_not(unselected) | default_plan.grid_fits(maximum_grid_x, maximum_grid_yz))
    return legal


@lru_cache(maxsize=4096)
def _cached_grids_fit(
    catalog: AttentionScheduleCatalog,
    context: AttentionLaunchContext,
    batch_size: int,
    num_groups: int,
    phase: AttentionPhase,
    precision: str,
    maximum_grid_x: int,
    maximum_grid_yz: int,
) -> bool:
    return _grids_fit(
        catalog,
        context,
        batch_size,
        num_groups,
        phase,
        precision,
        maximum_grid_x,
        maximum_grid_yz,
    )


def _target_changes_plan(
    catalog: AttentionScheduleCatalog,
    context: AttentionLaunchContext,
    phases: tuple[AttentionPhase, ...],
    precision: str,
) -> bool:
    return any(
        target_delta_changes_config(
            context,
            phase_profile.generic,
            phase_profile.target_deltas,
        )
        for phase_profile in (catalog._profile(phase, precision) for phase in phases)
    )


@lru_cache(maxsize=4096)
def _cached_target_changes_plan(
    catalog: AttentionScheduleCatalog,
    context: AttentionLaunchContext,
    phases: tuple[AttentionPhase, ...],
    precision: str,
) -> bool:
    return _target_changes_plan(catalog, context, phases, precision)


def _make_plan(
    context: AttentionLaunchContext,
    batch_size: int,
    num_groups: int,
    phase: AttentionPhase,
    tile: ResolvedAttentionTile,
) -> AttentionLaunchPlan:
    work_shape: tuple[int, ...]
    block_shape: tuple[int, ...]
    if phase is AttentionPhase.BACKWARD_KEY_VALUE:
        assert tile.block_key is not None
        work_shape = (batch_size * num_groups * context.num_heads, context.key_length)
        block_shape = (1, tile.block_key)
    elif phase is AttentionPhase.GRAD_BIAS:
        assert tile.block_key is not None
        work_shape = (batch_size * context.num_heads, context.query_length, context.key_length)
        block_shape = (1, tile.block_query, tile.block_key)
    else:
        work_shape = (batch_size * num_groups * context.num_heads, context.query_length)
        block_shape = (1, tile.block_query)
    return AttentionLaunchPlan(phase=phase, work_shape=work_shape, block_shape=block_shape, tile=tile)


@dataclass(frozen=True, slots=True)
class SegmentedAttentionProblem:

    num_samples: int
    total_positions: int
    num_heads: int
    head_size: int
    dtype: torch.dtype
    element_size: int
    mask_kind: int
    input_precision: str

    def __post_init__(self) -> None:
        validate_positive_parameters(
            num_samples=self.num_samples,
            total_positions=self.total_positions,
            num_heads=self.num_heads,
            head_size=self.head_size,
            element_size=self.element_size,
        )

    @property
    def launch_context(self) -> AttentionLaunchContext:

        return AttentionLaunchContext(
            num_heads=self.num_heads,
            query_length=1,
            key_length=1,
            head_size=self.head_size,
            mask_kind=self.mask_kind,
            element_size=self.element_size,
        )


@dataclass(frozen=True, slots=True)
class SegmentedAttentionConfig:

    tile: AttentionTileConfig
    programs_per_multiprocessor: int

    def __post_init__(self) -> None:
        validate_positive_parameters(programs_per_multiprocessor=self.programs_per_multiprocessor)


@dataclass(frozen=True, slots=True)
class SegmentedAttentionRule:

    config: SegmentedAttentionConfig
    dtypes: tuple[torch.dtype, ...]

    def can_implement(self, problem: SegmentedAttentionProblem) -> bool:
        return problem.dtype in self.dtypes

    def is_preferred(self, problem: SegmentedAttentionProblem) -> bool:
        return self.can_implement(problem)


@dataclass(frozen=True, slots=True)
class SegmentedAttentionPlan:

    phase: AttentionPhase
    grid: tuple[int]
    tile: ResolvedAttentionTile

    def grid_fits(self) -> bool:
        return 0 < self.grid[0] <= CUDA_MAX_GRID_X


_SegmentedProfile: TypeAlias = LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig]
_SegmentedTargetDelta: TypeAlias = TargetProfile[
    LaunchProfileDelta[SegmentedAttentionProblem, SegmentedAttentionConfig]
]


@dataclass(frozen=True, slots=True)
class SegmentedAttentionPhaseProfile:

    generic: _SegmentedProfile
    target_deltas: tuple[_SegmentedTargetDelta, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class SegmentedAttentionScheduleCatalog:

    forward: SegmentedAttentionPhaseProfile
    backward_preprocess: SegmentedAttentionPhaseProfile
    backward_query: SegmentedAttentionPhaseProfile
    backward_key_value: SegmentedAttentionPhaseProfile
    backward_key_value_tf32x3: SegmentedAttentionPhaseProfile
    grad_bias: SegmentedAttentionPhaseProfile

    def resolve(
        self,
        problem: SegmentedAttentionProblem,
        phase: AttentionPhase,
        *,
        target: TargetDescriptor | None,
        portable: bool = False,
    ) -> SegmentedAttentionPlan:
        profile = self._profile(phase, problem.input_precision)
        selected = select_target_launch_profile(
            profile.generic,
            target=None if portable else target,
            deltas=profile.target_deltas,
        )
        config = select_launch_config(problem, selected)
        if portable:
            num_programs = 1
        else:
            if target is None or target.resources.multiprocessor_count is None:
                raise ValueError("segmented attention requires a target multiprocessor count")
            num_programs = target.resources.multiprocessor_count * config.programs_per_multiprocessor
        plan = SegmentedAttentionPlan(
            phase=phase,
            grid=(num_programs,),
            tile=config.tile.resolve(problem.launch_context),
        )
        if not plan.grid_fits():
            raise ValueError(f"segmented attention has no launchable {phase.value} plan")
        return plan

    def supports(self, problem: SegmentedAttentionProblem, phase: AttentionPhase) -> bool:
        profile = self._profile(phase, problem.input_precision)
        return all(
            SegmentedAttentionPlan(
                phase=phase,
                grid=(1,),
                tile=config.tile.resolve(problem.launch_context),
            ).grid_fits()
            for config in select_launch_candidates(
                problem,
                profile.generic,
                target_deltas=profile.target_deltas,
            )
        )

    def target_changes_plan(
        self,
        problem: SegmentedAttentionProblem,
        phases: tuple[AttentionPhase, ...],
    ) -> bool:
        return any(
            target_delta_changes_config(problem, profile.generic, profile.target_deltas)
            for profile in (self._profile(phase, problem.input_precision) for phase in phases)
        )

    def _profile(self, phase: AttentionPhase, precision: str) -> SegmentedAttentionPhaseProfile:
        if phase is AttentionPhase.FORWARD:
            return self.forward
        if phase is AttentionPhase.BACKWARD_PREPROCESS:
            return self.backward_preprocess
        if phase is AttentionPhase.BACKWARD_QUERY:
            return self.backward_query
        if phase is AttentionPhase.BACKWARD_KEY_VALUE:
            return self.backward_key_value_tf32x3 if precision == FLOAT32_DOT_TF32X3 else self.backward_key_value
        if phase is AttentionPhase.GRAD_BIAS:
            return self.grad_bias
        raise ValueError(f"unknown segmented attention phase: {phase}")
