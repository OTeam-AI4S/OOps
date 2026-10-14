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


def stable_destination_ordering(dst_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:

    destinations = dst_index.to(torch.long).clamp_max(num_nodes - 1).clamp_min(0)
    sorted_destinations, order = torch.sort(destinations, stable=True)
    boundaries = torch.searchsorted(
        sorted_destinations,
        torch.arange(num_nodes + 1, device=dst_index.device, dtype=destinations.dtype),
    )
    return order, boundaries
