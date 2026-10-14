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
