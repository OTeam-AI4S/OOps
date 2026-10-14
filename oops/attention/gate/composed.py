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
