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

r"""Shared launch configuration types and selectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

import torch

_ContextT_contra = TypeVar("_ContextT_contra", contravariant=True)
_ConfigT_co = TypeVar("_ConfigT_co", covariant=True)
_ContextT = TypeVar("_ContextT")
_ConfigT = TypeVar("_ConfigT")
_ValueT = TypeVar("_ValueT")


class _LaunchRule(Protocol[_ContextT_contra, _ConfigT_co]):
    @property
    def config(self) -> _ConfigT_co: ...

    def can_implement(self, context: _ContextT_contra) -> bool: ...

    def is_preferred(self, context: _ContextT_contra) -> bool: ...


@dataclass(frozen=True, slots=True)
class TargetResources:
    r"""Optional launch-resource limits reported by a device adapter."""

    shared_memory_per_block: int | None = None
    shared_memory_per_multiprocessor: int | None = None
    registers_per_block: int | None = None
    registers_per_multiprocessor: int | None = None
    max_threads_per_block: int | None = None
    max_threads_per_multiprocessor: int | None = None
    warp_size: int | None = None
    multiprocessor_count: int | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(
            shared_memory_per_block=self.shared_memory_per_block,
            shared_memory_per_multiprocessor=self.shared_memory_per_multiprocessor,
            registers_per_block=self.registers_per_block,
            registers_per_multiprocessor=self.registers_per_multiprocessor,
            max_threads_per_block=self.max_threads_per_block,
            max_threads_per_multiprocessor=self.max_threads_per_multiprocessor,
            warp_size=self.warp_size,
            multiprocessor_count=self.multiprocessor_count,
        )


@dataclass(frozen=True, slots=True)
class TargetDescriptor:
    r"""Hardware/compiler target with architecture, optional variant, features, and resources."""

    backend: str
    architecture: str
    variant: str | None
    features: frozenset[str]
    resources: TargetResources
    _profile_hash: int | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be non-empty")
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError("architecture must be non-empty")
        if self.variant is not None and (not isinstance(self.variant, str) or not self.variant):
            raise ValueError("variant must be non-empty when specified")
        if isinstance(self.features, str):
            raise TypeError("features must be an iterable of feature names, not a string")
        object.__setattr__(self, "features", frozenset(self.features))
        if any(not isinstance(feature, str) or not feature for feature in self.features):
            raise ValueError("features must contain only non-empty names")
        if not isinstance(self.resources, TargetResources):
            raise TypeError("resources must be TargetResources")

    def __hash__(self) -> int:
        # Hash lazily: Dynamo cannot trace tuple hashing during construction.
        cached = self._profile_hash
        if cached is None:
            cached = hash((self.backend, self.architecture, self.variant, self.features))
            object.__setattr__(self, "_profile_hash", cached)
        return cached


@dataclass(frozen=True, slots=True)
class TargetSelector:
    r"""Declarative identity and feature predicate for one target profile."""

    backend: str
    architectures: tuple[str, ...]
    variants: tuple[str | None, ...] | None = None
    required_features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be non-empty")
        if isinstance(self.architectures, str):
            raise TypeError("architectures must be an iterable of architecture names, not a string")
        object.__setattr__(self, "architectures", tuple(self.architectures))
        if not self.architectures or any(
            not isinstance(architecture, str) or not architecture for architecture in self.architectures
        ):
            raise ValueError("architectures must contain at least one non-empty name")
        if self.variants is not None:
            if isinstance(self.variants, str):
                raise TypeError("variants must be an iterable of variants, not a string")
            object.__setattr__(self, "variants", tuple(self.variants))
        if self.variants is not None and (
            not self.variants
            or any(variant is not None and (not isinstance(variant, str) or not variant) for variant in self.variants)
        ):
            raise ValueError("variants must be None or contain at least one valid variant")
        if isinstance(self.required_features, str):
            raise TypeError("required_features must be an iterable of feature names, not a string")
        object.__setattr__(self, "required_features", frozenset(self.required_features))
        if any(not isinstance(feature, str) or not feature for feature in self.required_features):
            raise ValueError("required_features must contain only non-empty names")

    def matches(self, target: TargetDescriptor) -> bool:
        if target.backend != self.backend or target.architecture not in self.architectures:
            return False
        if self.variants is not None and target.variant not in self.variants:
            return False
        return self.required_features.issubset(target.features)


@dataclass(frozen=True, slots=True)
class LaunchProfile(Generic[_ContextT, _ConfigT]):
    r"""One deterministic default and its ordered shape-specific overrides."""

    default: _ConfigT
    rules: tuple[_LaunchRule[_ContextT, _ConfigT], ...] = ()


@dataclass(frozen=True, slots=True)
class LaunchProfileDelta(Generic[_ContextT, _ConfigT]):
    r"""Target-specific overrides layered ahead of a generic launch profile.

    Target rules have first-match precedence over generic rules. ``default`` replaces only
    the generic default; it never discards generic rules.
    """

    rules: tuple[_LaunchRule[_ContextT, _ConfigT], ...] = ()
    default: _ConfigT | None = None


@dataclass(frozen=True, slots=True)
class TargetProfile(Generic[_ValueT]):
    r"""One profiled value guarded by an explicit backend-neutral target selector."""

    selector: TargetSelector
    value: _ValueT

    def matches(self, target: TargetDescriptor) -> bool:
        return self.selector.matches(target)


@dataclass(frozen=True, slots=True)
class CoreLaunchContext:
    r"""
    Shape features used to select a contraction/reduction launch configuration.
    """

    num_rows: int
    channels: int
    reduction: int | None = None
    tokens: int | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        validate_positive_parameters(
            num_rows=self.num_rows,
            channels=self.channels,
            reduction=self.reduction,
            tokens=self.tokens,
        )

    @property
    def elements(self) -> int:
        r"""
        Number of row-channel output elements touched by the launch.
        """

        return self.num_rows * self.channels

    @property
    def contraction_work(self) -> int:
        r"""
        Approximate scalar multiply-add work for row/channel/reduction kernels.
        """

        reduction = 1 if self.reduction is None else self.reduction
        return self.elements * reduction


@dataclass(frozen=True, slots=True)
class CoreTileConfig:
    r"""
    Compile-time tile and pipeline parameters for one core kernel launch.
    """

    name: str
    block_rows: int
    block_channels: int | None = None
    block_reduction: int | None = None
    block_tokens: int | None = None
    num_warps: int = 4
    num_stages: int = 3

    def __post_init__(self) -> None:
        validate_positive_parameters(
            block_rows=self.block_rows,
            block_channels=self.block_channels,
            block_reduction=self.block_reduction,
            block_tokens=self.block_tokens,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )


@dataclass(frozen=True, slots=True)
class CoreLaunchRule:
    r"""
    Shape predicate for choosing a non-default core tile.
    """

    config: CoreTileConfig
    min_rows: int = 0
    max_rows: int | None = None
    min_channels: int = 0
    max_channels: int | None = None
    min_reduction: int = 0
    max_reduction: int | None = None
    min_tokens: int = 0
    max_tokens: int | None = None
    min_elements: int = 0
    min_contraction_work: int = 0
    dtypes: tuple[torch.dtype, ...] | None = None

    def is_preferred(self, context: CoreLaunchContext) -> bool:
        r"""
        Return whether this legal candidate is preferred for ``context``.
        """

        if not self.can_implement(context):
            return False
        if context.num_rows < self.min_rows:
            return False
        if self.max_rows is not None and context.num_rows > self.max_rows:
            return False
        if not _optional_range_matches(context.tokens, self.min_tokens, self.max_tokens):
            return False
        if context.elements < self.min_elements:
            return False
        if context.contraction_work < self.min_contraction_work:
            return False
        return True

    def can_implement(self, context: CoreLaunchContext) -> bool:
        r"""Check channel, reduction, and dtype compatibility, independent of preference thresholds."""

        if context.channels < self.min_channels:
            return False
        if self.max_channels is not None and context.channels > self.max_channels:
            return False
        if not _optional_range_matches(context.reduction, self.min_reduction, self.max_reduction):
            return False
        return self.dtypes is None or context.dtype in self.dtypes


def select_core_tile(
    context: CoreLaunchContext,
    profile: LaunchProfile[CoreLaunchContext, CoreTileConfig],
    *,
    target: TargetDescriptor | None = None,
    target_deltas: tuple[TargetProfile[LaunchProfileDelta[CoreLaunchContext, CoreTileConfig]], ...] = (),
) -> CoreTileConfig:
    r"""Select a core tile from ordered shape rules."""

    resolved = select_target_launch_profile(profile, target=target, deltas=target_deltas)
    return select_launch_config(context, resolved)


def select_target_profile(
    fallback: _ValueT,
    *,
    target: TargetDescriptor | None = None,
    profiles: tuple[TargetProfile[_ValueT], ...] = (),
) -> _ValueT:
    r"""Return the first value matching ``target``, or ``fallback``."""

    if target is not None:
        for profile in profiles:
            if profile.matches(target):
                return profile.value
    return fallback


def apply_launch_delta(
    profile: LaunchProfile[_ContextT, _ConfigT],
    delta: LaunchProfileDelta[_ContextT, _ConfigT],
) -> LaunchProfile[_ContextT, _ConfigT]:
    r"""Overlay one target delta without dropping target-independent rules."""

    default = profile.default if delta.default is None else delta.default
    return LaunchProfile(default, (*delta.rules, *profile.rules))


def select_target_launch_profile(
    profile: LaunchProfile[_ContextT, _ConfigT],
    *,
    target: TargetDescriptor | None = None,
    deltas: tuple[TargetProfile[LaunchProfileDelta[_ContextT, _ConfigT]], ...] = (),
) -> LaunchProfile[_ContextT, _ConfigT]:
    r"""Apply the first target delta matching ``target``, or return ``profile``."""

    if target is not None:
        for target_delta in deltas:
            if target_delta.matches(target):
                return apply_launch_delta(profile, target_delta.value)
    return profile


def target_delta_changes_config(
    context: _ContextT,
    fallback: LaunchProfile[_ContextT, _ConfigT],
    deltas: tuple[TargetProfile[LaunchProfileDelta[_ContextT, _ConfigT]], ...],
) -> bool:
    r"""Return whether any target delta selects a different configuration object."""

    if not deltas:
        return False
    generic = select_launch_config(context, fallback)
    return any(
        select_launch_config(context, apply_launch_delta(fallback, target_delta.value)) is not generic
        for target_delta in deltas
    )


def select_launch_config(
    context: _ContextT,
    profile: LaunchProfile[_ContextT, _ConfigT],
) -> _ConfigT:
    r"""
    Return the first matching launch configuration, or ``default``.
    """

    for rule in profile.rules:
        if rule.is_preferred(context):
            return rule.config
    return profile.default


def select_launch_candidates(
    context: _ContextT,
    profile: LaunchProfile[_ContextT, _ConfigT],
    *,
    target_deltas: tuple[TargetProfile[LaunchProfileDelta[_ContextT, _ConfigT]], ...] = (),
) -> tuple[_ConfigT, ...]:
    r"""Return distinct compatible defaults and rule configurations, ignoring preference thresholds."""

    if not profile.rules and not target_deltas:
        return (profile.default,)

    profiles = (
        *(apply_launch_delta(profile, target_delta.value) for target_delta in target_deltas),
        profile,
    )
    selected = tuple(
        config
        for candidate in profiles
        for config in (
            candidate.default,
            *(rule.config for rule in candidate.rules if rule.can_implement(context)),
        )
    )
    unique: list[_ConfigT] = []
    for config in selected:
        if not any(config is candidate for candidate in unique):
            unique.append(config)
    return tuple(unique)


def _optional_range_matches(value: int | None, minimum: int, maximum: int | None) -> bool:
    r"""Match a feature against bounds; an absent feature only matches an unconstrained range."""

    if value is None:
        return minimum == 0 and maximum is None
    if value < minimum:
        return False
    return maximum is None or value <= maximum


def validate_positive_parameters(**parameters: int | None) -> None:
    for name, value in parameters.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive; got {value}")
