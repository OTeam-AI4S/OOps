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

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import pytest
import torch


class Tolerance(TypedDict):
    atol: float
    rtol: float


@pytest.fixture
def tolerance() -> Callable[[torch.dtype], Tolerance]:
    def get(dtype: torch.dtype) -> Tolerance:
        if dtype in (torch.float16, torch.bfloat16):
            return {"atol": 6e-2, "rtol": 6e-2}
        if dtype is torch.float32:
            return {"atol": 2e-5, "rtol": 2e-5}
        return {"atol": 1e-12, "rtol": 1e-12}

    return get
