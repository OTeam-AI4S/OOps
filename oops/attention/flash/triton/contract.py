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

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
from torch import Tensor

from .problem import AttentionProblem
from .schedule import AttentionLaunchPlan, AttentionScheduleCatalog


class ProblemBinder(Protocol):
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        bias: Tensor | None,
        /,
        *,
        bind_launch_strides: bool,
    ) -> AttentionProblem: ...


TensorMetadata: TypeAlias = tuple[
    torch.device,
    torch.dtype,
    torch.layout,
    tuple[int, ...],
    tuple[int, ...],
]


AttentionAdmissionKey: TypeAlias = tuple[
    tuple[TensorMetadata | None, ...],
    bool,
    torch.dtype | None,
    str,
]


@dataclass(frozen=True, slots=True)
class FlashAttentionBackend:

    operator: str
    float32_dot: str
    bind_problem: ProblemBinder
    schedules: AttentionScheduleCatalog
    contiguous_fake_bias_gradient: bool = False


@dataclass(frozen=True, slots=True)
class PreparedAttention:

    problem: AttentionProblem
    forward_plan: AttentionLaunchPlan
    precision: str


def attention_admission_key(
    tensors: tuple[Tensor | None, ...],
    *,
    precision: str,
) -> AttentionAdmissionKey:

    metadata = tuple(
        (
            None
            if tensor is None
            else (
                tensor.device,
                tensor.dtype,
                tensor.layout,
                tensor.shape,
                tensor.stride(),
            )
        )
        for tensor in tensors
    )
    autocast_enabled = torch.is_autocast_enabled("cuda")
    autocast_dtype = torch.get_autocast_dtype("cuda") if autocast_enabled else None
    return metadata, autocast_enabled, autocast_dtype, precision
