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
from dataclasses import dataclass

from torch import Tensor

from oops.runtime import require_same_device

from .qkv import optional_unit_batch

DenseAttentionRun = Callable[[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, float | None], Tensor]
QkvCheck = Callable[[Tensor, Tensor, Tensor], None]


@dataclass(frozen=True)
class DenseAttentionSpec:

    name: str
    batchless_rank: int
    maximum_rank: int
    check_qkv: QkvCheck
    run: DenseAttentionRun

    @property
    def batched_rank(self) -> int:

        return self.batchless_rank + 1

    @property
    def accepted_ranks(self) -> tuple[int, ...]:

        return tuple(range(self.batchless_rank, self.maximum_rank + 1))


def run_dense_attention(
    spec: DenseAttentionSpec,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    *,
    scale: float | None,
) -> Tensor:

    spec.check_qkv(query, key, value)
    require_same_device(spec.name, query, key, value, mask, bias)
    rank = query.dim()
    if rank == spec.batchless_rank:
        batched_query = query.unsqueeze(0)
        batched_key = batched_query if key is query else key.unsqueeze(0)
        if value is query:
            batched_value = batched_query
        elif value is key:
            batched_value = batched_key
        else:
            batched_value = value.unsqueeze(0)
        output = run_dense_attention(
            spec,
            batched_query,
            batched_key,
            batched_value,
            optional_unit_batch(mask),
            optional_unit_batch(bias),
            scale=scale,
        )
        return output.squeeze(0)
    if not spec.batched_rank <= rank <= spec.maximum_rank:
        accepted = " or ".join(str(accepted_rank) for accepted_rank in spec.accepted_ranks)
        raise ValueError(f"{spec.name} query must have rank {accepted}, got {rank}")
    return spec.run(query, key, value, mask, bias, scale)
