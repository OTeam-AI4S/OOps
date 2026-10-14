from __future__ import annotations

from functools import lru_cache

from oops.core import TargetDescriptor, select_launch_config, select_target_launch_profile, target_delta_changes_config

from ..schedule import SmoothLddtDensePlan, SmoothLddtDenseProblem, SmoothLddtMaskPlan, SmoothLddtMaskProblem
from .generic import (
    DENSE_FALLBACK,
    MASK_FORWARD_FALLBACK,
    SMOOTH_LDDT_DENSE,
    SMOOTH_LDDT_DENSE_FINALIZE,
    SMOOTH_LDDT_MASK,
    SMOOTH_LDDT_PAIR_DISTANCE,
    SMOOTH_LDDT_SPARSE,
    SMOOTH_LDDT_SPARSE_COORDS,
)
from .sm80 import DENSE_TARGETS, MASK_FORWARD_TARGETS

__all__ = [
    "SMOOTH_LDDT_DENSE",
    "SMOOTH_LDDT_DENSE_FINALIZE",
    "SMOOTH_LDDT_MASK",
    "SMOOTH_LDDT_PAIR_DISTANCE",
    "SMOOTH_LDDT_SPARSE",
    "SMOOTH_LDDT_SPARSE_COORDS",
]


@lru_cache(maxsize=1024)
def dense_target_changes_plan(problem: SmoothLddtDenseProblem) -> bool:
    return target_delta_changes_config(problem, DENSE_FALLBACK, DENSE_TARGETS)


@lru_cache(maxsize=1024)
def resolve_dense_plan(problem: SmoothLddtDenseProblem, *, target: TargetDescriptor | None) -> SmoothLddtDensePlan:
    profile = select_target_launch_profile(DENSE_FALLBACK, target=target, deltas=DENSE_TARGETS)
    return select_launch_config(problem, profile).resolve(problem)


@lru_cache(maxsize=1024)
def mask_forward_target_changes_plan(problem: SmoothLddtMaskProblem) -> bool:
    return target_delta_changes_config(problem, MASK_FORWARD_FALLBACK, MASK_FORWARD_TARGETS)


@lru_cache(maxsize=1024)
def resolve_mask_forward(problem: SmoothLddtMaskProblem, *, target: TargetDescriptor | None) -> SmoothLddtMaskPlan:
    profile = select_target_launch_profile(MASK_FORWARD_FALLBACK, target=target, deltas=MASK_FORWARD_TARGETS)
    return select_launch_config(problem, profile).resolve(problem)
