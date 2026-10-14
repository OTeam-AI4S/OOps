from __future__ import annotations

from functools import lru_cache

from oops.core import TargetDescriptor, select_launch_config, select_target_launch_profile, target_delta_changes_config

from ..schedule import CrossEntropyPlan, CrossEntropyProblem, DistogramCrossEntropyPlan, DistogramCrossEntropyProblem
from .generic import (
    CROSS_ENTROPY,
    CROSS_ENTROPY_FORWARD_FALLBACK,
    DISTOGRAM_CROSS_ENTROPY,
    DISTOGRAM_FINALIZE,
    DISTOGRAM_FORWARD_FALLBACK,
)
from .sm80 import CROSS_ENTROPY_FORWARD_TARGETS, DISTOGRAM_BACKWARD_TARGETS, DISTOGRAM_FORWARD_TARGETS

__all__ = [
    "CROSS_ENTROPY",
    "DISTOGRAM_CROSS_ENTROPY",
    "DISTOGRAM_FINALIZE",
]


@lru_cache(maxsize=1024)
def cross_entropy_forward_target_changes_plan(problem: CrossEntropyProblem) -> bool:
    return target_delta_changes_config(problem, CROSS_ENTROPY_FORWARD_FALLBACK, CROSS_ENTROPY_FORWARD_TARGETS)


@lru_cache(maxsize=1024)
def resolve_cross_entropy_forward(problem: CrossEntropyProblem, *, target: TargetDescriptor | None) -> CrossEntropyPlan:
    profile = select_target_launch_profile(
        CROSS_ENTROPY_FORWARD_FALLBACK, target=target, deltas=CROSS_ENTROPY_FORWARD_TARGETS
    )
    return select_launch_config(problem, profile).resolve(problem)


@lru_cache(maxsize=1024)
def distogram_forward_target_changes_plan(problem: DistogramCrossEntropyProblem) -> bool:
    return target_delta_changes_config(problem, DISTOGRAM_FORWARD_FALLBACK, DISTOGRAM_FORWARD_TARGETS)


@lru_cache(maxsize=1024)
def resolve_distogram_forward(
    problem: DistogramCrossEntropyProblem, *, target: TargetDescriptor | None
) -> DistogramCrossEntropyPlan:
    profile = select_target_launch_profile(DISTOGRAM_FORWARD_FALLBACK, target=target, deltas=DISTOGRAM_FORWARD_TARGETS)
    return select_launch_config(problem, profile).resolve(problem)


@lru_cache(maxsize=1024)
def distogram_backward_target_changes_plan(problem: DistogramCrossEntropyProblem) -> bool:
    return target_delta_changes_config(problem, DISTOGRAM_FORWARD_FALLBACK, DISTOGRAM_BACKWARD_TARGETS)


@lru_cache(maxsize=1024)
def resolve_distogram_backward(
    problem: DistogramCrossEntropyProblem, *, target: TargetDescriptor | None
) -> DistogramCrossEntropyPlan:
    profile = select_target_launch_profile(DISTOGRAM_FORWARD_FALLBACK, target=target, deltas=DISTOGRAM_BACKWARD_TARGETS)
    return select_launch_config(problem, profile).resolve(problem, backward=True)
