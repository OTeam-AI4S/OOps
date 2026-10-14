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
from oops.runtime import HALF_DTYPES

from ..schedule import (
    GateMultiplyConfig,
    GateMultiplyProblem,
    GateMultiplyRule,
    GateProjectionConfig,
    GateProjectionProblem,
    GateProjectionRule,
)
from . import generic

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

PROJECTION_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[GateProjectionProblem, GateProjectionConfig](
            rules=(
                GateProjectionRule(
                    replace(generic.PROJECTION_FALLBACK.default, block_gate=64), HALF_DTYPES, min_values=1 << 19
                ),
                GateProjectionRule(
                    replace(generic.PROJECTION_FALLBACK.default, block_gate=128, block_reduction=32),
                    (torch.float32,),
                    min_values=1 << 21,
                ),
            )
        ),
    ),
)
MULTIPLY_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[GateMultiplyProblem, GateMultiplyConfig](
            rules=(GateMultiplyRule(replace(generic.MULTIPLY_FORWARD_FALLBACK.default, block_rows=16), HALF_DTYPES),)
        ),
    ),
)
MULTIPLY_BACKWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[GateMultiplyProblem, GateMultiplyConfig](
            rules=(GateMultiplyRule(replace(generic.MULTIPLY_BACKWARD_FALLBACK.default, block_rows=16), HALF_DTYPES),)
        ),
    ),
)
