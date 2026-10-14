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

from dataclasses import replace

import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import NestedTokenPairWindowConfig, NestedTokenPairWindowProblem, NestedTokenPairWindowRule
from .generic import NESTED_TOKEN_PAIR_WINDOW_BACKWARD_FALLBACK

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

_NESTED_TOKEN_PAIR_WINDOW_BACKWARD = LaunchProfileDelta[NestedTokenPairWindowProblem, NestedTokenPairWindowConfig](
    rules=(
        NestedTokenPairWindowRule(
            config=replace(NESTED_TOKEN_PAIR_WINDOW_BACKWARD_FALLBACK.default, block_elements=512, num_warps=8),
            dtypes=(torch.bfloat16,),
            channels=64,
            n_queries=32,
            n_keys=128,
            has_num_valid_atoms=False,
        ),
    )
)

NESTED_TOKEN_PAIR_WINDOW_BACKWARD_TARGETS = (TargetProfile(_SM80, _NESTED_TOKEN_PAIR_WINDOW_BACKWARD),)
