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


@pytest.fixture
def csr_reduce() -> Callable[..., torch.Tensor]:
    def reduce(
        values: torch.Tensor,
        offsets: torch.Tensor,
        segments: int,
        reduction: Callable[[torch.Tensor], torch.Tensor],
        *,
        fill: float = 0.0,
    ) -> torch.Tensor:
        bounds = offsets.cpu().tolist()
        output = values.new_full((segments, *values.shape[1:]), fill)
        for segment, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
            if start != stop:
                output[segment] = reduction(values[start:stop])
        return output

    return reduce
