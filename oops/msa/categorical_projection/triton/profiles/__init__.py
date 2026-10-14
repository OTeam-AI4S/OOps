from functools import lru_cache

from oops.core import TargetDescriptor, select_launch_config, select_target_launch_profile, target_delta_changes_config

from ..schedule import CategoricalProjectionPlan, CategoricalProjectionProblem
from .generic import FALLBACK, SCHEDULE
from .sm80 import FORWARD_TARGETS

__all__ = ["SCHEDULE"]


@lru_cache(maxsize=1024)
def target_changes_plan(problem: CategoricalProjectionProblem) -> bool:
    return target_delta_changes_config(problem, FALLBACK, FORWARD_TARGETS)


@lru_cache(maxsize=1024)
def resolve_plan(
    problem: CategoricalProjectionProblem, *, target: TargetDescriptor | None
) -> CategoricalProjectionPlan:
    profile = select_target_launch_profile(FALLBACK, target=target, deltas=FORWARD_TARGETS)
    return select_launch_config(problem, profile).resolve(problem)
