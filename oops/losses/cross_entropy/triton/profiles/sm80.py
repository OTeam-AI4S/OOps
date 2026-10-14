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
    CrossEntropyConfig,
    CrossEntropyProblem,
    CrossEntropyRule,
    CrossEntropySchedule,
    DistogramCrossEntropyConfig,
    DistogramCrossEntropyProblem,
    DistogramCrossEntropyRule,
    DistogramCrossEntropySchedule,
)
from .generic import DISTOGRAM_FINALIZE

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
CROSS_ENTROPY_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[CrossEntropyProblem, CrossEntropySchedule](
            rules=(
                CrossEntropyRule(
                    CrossEntropySchedule(CrossEntropyConfig(block_rows=64)),
                    min_rows=1 << 20,
                    min_classes=128,
                    max_classes=128,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
DISTOGRAM_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[DistogramCrossEntropyProblem, DistogramCrossEntropySchedule](
            rules=(
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=128, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=64,
                    max_classes=64,
                    dtypes=(torch.float32,),
                ),
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=128, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=128,
                    max_classes=128,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
DISTOGRAM_BACKWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[DistogramCrossEntropyProblem, DistogramCrossEntropySchedule](
            rules=(
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=32, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=64,
                    max_classes=64,
                    dtypes=(torch.float32,),
                ),
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=32, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=128,
                    max_classes=128,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
