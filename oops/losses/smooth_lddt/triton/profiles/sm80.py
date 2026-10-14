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
    SmoothLddtDenseConfig,
    SmoothLddtDenseProblem,
    SmoothLddtDenseRule,
    SmoothLddtDenseSchedule,
    SmoothLddtMaskConfig,
    SmoothLddtMaskProblem,
    SmoothLddtMaskRule,
    SmoothLddtMaskSchedule,
)
from .generic import SMOOTH_LDDT_DENSE_FINALIZE

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
DENSE_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[SmoothLddtDenseProblem, SmoothLddtDenseSchedule](
            rules=(
                SmoothLddtDenseRule(
                    SmoothLddtDenseSchedule(
                        SmoothLddtDenseConfig(block_pairs=256, row_block_atoms=256, num_warps=8),
                        SMOOTH_LDDT_DENSE_FINALIZE,
                    ),
                    min_atoms=512,
                    dtypes=(torch.float32, torch.bfloat16),
                ),
            )
        ),
    ),
)
MASK_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[SmoothLddtMaskProblem, SmoothLddtMaskSchedule](
            rules=(
                SmoothLddtMaskRule(
                    SmoothLddtMaskSchedule(SmoothLddtMaskConfig(block_pairs=512, num_warps=8)),
                    min_atoms=1024,
                    dtypes=(torch.float32, torch.bfloat16),
                ),
            )
        ),
    ),
)
