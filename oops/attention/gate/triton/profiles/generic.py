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

from oops.core import LaunchProfile
from oops.runtime import HALF_DTYPES

from ..schedule import (
    GateMultiplyConfig,
    GateMultiplyProblem,
    GateNativeLinearRule,
    GateProjectionConfig,
    GateProjectionProblem,
)

PROJECTION_FALLBACK = LaunchProfile[GateProjectionProblem, GateProjectionConfig](GateProjectionConfig(64, 32, 64, 4, 3))
NATIVE_LINEAR = (
    GateNativeLinearRule(HALF_DTYPES, min_intensity=150, min_multiply_accumulates=1 << 30),
    GateNativeLinearRule((torch.float32,), ("tf32",), min_intensity=150, min_multiply_accumulates=1 << 28),
)
MULTIPLY_FORWARD_FALLBACK = LaunchProfile[GateMultiplyProblem, GateMultiplyConfig](GateMultiplyConfig(4, 64, 4))
MULTIPLY_BACKWARD_FALLBACK = LaunchProfile[GateMultiplyProblem, GateMultiplyConfig](GateMultiplyConfig(4, 64, 4))
