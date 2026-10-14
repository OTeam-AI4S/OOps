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

from ..schedule import NestedTokenPairWindowSchedule
from . import generic as _generic
from . import sm80 as _sm80
from .generic import TOKEN_PAIR_WINDOW

NESTED_TOKEN_PAIR_WINDOW_FORWARD = NestedTokenPairWindowSchedule(_generic.NESTED_TOKEN_PAIR_WINDOW_FORWARD_FALLBACK)
NESTED_TOKEN_PAIR_WINDOW_BACKWARD = NestedTokenPairWindowSchedule(
    _generic.NESTED_TOKEN_PAIR_WINDOW_BACKWARD_FALLBACK,
    _sm80.NESTED_TOKEN_PAIR_WINDOW_BACKWARD_TARGETS,
)

__all__ = [
    "NESTED_TOKEN_PAIR_WINDOW_BACKWARD",
    "NESTED_TOKEN_PAIR_WINDOW_FORWARD",
    "TOKEN_PAIR_WINDOW",
]
