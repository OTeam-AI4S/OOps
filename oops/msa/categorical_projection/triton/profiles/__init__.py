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
