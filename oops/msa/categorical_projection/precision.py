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

import torch

from oops.runtime import (
    captured_native_matmul_precision_state,
    native_matmul_domain,
    require_native_matmul_precision_state,
)


def resolve_categorical_msa_projection_native_matmul_state(dtype: torch.dtype) -> int:
    return captured_native_matmul_precision_state(native_matmul_domain(dtype))


def require_categorical_msa_projection_native_matmul_state(expected: int) -> None:
    require_native_matmul_precision_state("categorical_msa_projection", expected)
