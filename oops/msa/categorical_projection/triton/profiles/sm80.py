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

import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import (
    CategoricalProjectionConfig,
    CategoricalProjectionProblem,
    CategoricalProjectionRule,
    CategoricalProjectionSchedule,
)

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[CategoricalProjectionProblem, CategoricalProjectionSchedule](
            rules=(
                CategoricalProjectionRule(
                    CategoricalProjectionSchedule(
                        CategoricalProjectionConfig(block_rows=64, block_channels=32, num_warps=4)
                    ),
                    min_rows=1 << 16,
                    max_channels=128,
                    dtypes=(torch.float32, torch.bfloat16),
                ),
                CategoricalProjectionRule(
                    CategoricalProjectionSchedule(
                        CategoricalProjectionConfig(block_rows=16, block_channels=64, num_warps=4)
                    ),
                    max_rows=1024,
                    max_channels=64,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
