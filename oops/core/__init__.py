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

r"""
Core launch abstractions for contraction and reduction kernels.
"""

from .launch import (
    CoreLaunchContext,
    CoreLaunchRule,
    CoreTileConfig,
    LaunchProfile,
    LaunchProfileDelta,
    TargetDescriptor,
    TargetProfile,
    TargetResources,
    TargetSelector,
    apply_launch_delta,
    select_core_tile,
    select_launch_candidates,
    select_launch_config,
    select_target_launch_profile,
    select_target_profile,
    target_delta_changes_config,
    validate_positive_parameters,
)

__all__ = [
    "CoreLaunchContext",
    "CoreLaunchRule",
    "CoreTileConfig",
    "LaunchProfile",
    "LaunchProfileDelta",
    "TargetDescriptor",
    "TargetProfile",
    "TargetResources",
    "TargetSelector",
    "apply_launch_delta",
    "select_core_tile",
    "select_launch_candidates",
    "select_launch_config",
    "select_target_launch_profile",
    "select_target_profile",
    "target_delta_changes_config",
    "validate_positive_parameters",
]
