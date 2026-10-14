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

from ..schedule import GateAttentionOutputScheduleCatalog, GatePhaseProfile
from . import generic as _generic
from . import sm80 as _sm80

SCHEDULES = GateAttentionOutputScheduleCatalog(
    projection=GatePhaseProfile(_generic.PROJECTION_FALLBACK, _sm80.PROJECTION_TARGETS),
    native_linear=_generic.NATIVE_LINEAR,
    multiply_forward=GatePhaseProfile(_generic.MULTIPLY_FORWARD_FALLBACK, _sm80.MULTIPLY_FORWARD_TARGETS),
    multiply_backward=GatePhaseProfile(_generic.MULTIPLY_BACKWARD_FALLBACK, _sm80.MULTIPLY_BACKWARD_TARGETS),
)

__all__ = ["SCHEDULES"]
