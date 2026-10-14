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
from torch import Tensor
from torch.nn import functional as F

from oops.runtime import forward_mode_active, functorch_transforms_active

from .triton import project_gate, projection_operands


def _gate_attention_values(
    attention: Tensor,
    single: Tensor,
    gate_weight: Tensor,
    output_weight: Tensor,
) -> Tensor:
    if (
        attention.is_cuda
        and not functorch_transforms_active()
        and not forward_mode_active(attention, single, gate_weight)
    ):
        operands = projection_operands(attention, single, gate_weight)
        if operands is not None:
            return F.linear(project_gate(attention, *operands), output_weight)
    gate = torch.sigmoid(F.linear(single, gate_weight))
    if attention.is_contiguous():
        gated = gate * attention.flatten(-2)
    else:
        gated = (gate.unflatten(-1, attention.shape[-2:]) * attention).flatten(-2)
    del gate
    return F.linear(gated, output_weight)
