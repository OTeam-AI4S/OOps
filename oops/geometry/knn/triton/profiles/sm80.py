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

from ..schedule import MaskedKnnConfig, MaskedKnnProblem, MaskedKnnRule

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

_MASKED_KNN_FORWARD = LaunchProfileDelta[MaskedKnnProblem, MaskedKnnConfig](
    rules=(
        MaskedKnnRule(
            config=MaskedKnnConfig(block_key=512, num_warps=4),
            min_topk=128,
            max_topk=128,
            dtypes=(torch.float32, torch.bfloat16),
            min_points_per_sample=512,
            max_total_points=8192,
        ),
    )
)

MASKED_KNN_TARGETS = (TargetProfile(_SM80, _MASKED_KNN_FORWARD),)
MASKED_KNN_BACKWARD_TARGETS = ()
