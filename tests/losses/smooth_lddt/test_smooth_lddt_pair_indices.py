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

import pytest
import torch

from oops import smooth_lddt_pair_indices


def _mask(pattern: str, size: int, device: str) -> torch.Tensor:
    if pattern == "directed":
        return torch.tensor([[False, True, False], [True, False, True], [False, False, False]], device=device)
    if pattern == "empty":
        return torch.empty(0, 0, dtype=torch.bool, device=device)
    if pattern in ("all-false", "all-true"):
        return torch.full((size, size), pattern == "all-true", dtype=torch.bool, device=device)
    diagonal = torch.eye(size, dtype=torch.bool, device=device)
    return diagonal | diagonal.roll(1, dims=1) if pattern == "band" else diagonal


@pytest.mark.usefixtures("compiler_cache")
class TestSmoothLDDTPairIndices:

    @pytest.mark.parametrize(
        ("pattern", "device", "compiled"),
        [
            ("directed", "cpu", False),
            ("empty", "cpu", False),
            ("empty", "cuda", False),
            ("all-false", "cuda", False),
            ("all-true", "cpu", False),
            ("diagonal", "cuda", False),
            ("band", "cpu", True),
        ],
    )
    def test_forward(self, pattern: str, device: str, compiled: bool) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        operation = (
            torch.compile(smooth_lddt_pair_indices, fullgraph=True, dynamic=True)
            if compiled
            else smooth_lddt_pair_indices
        )
        for size in ((3, 5) if compiled else (3,)):
            mask = _mask(pattern, size, device)
            actual = operation(mask)
            expected = torch.nonzero(mask, as_tuple=False).reshape(-1)
            assert actual.device == mask.device
            assert actual.dtype is torch.long
            assert not actual.requires_grad
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("fault", ["rank", "nonsquare", "nonboolean"])
    def test_contract(self, fault: str) -> None:
        mask = (
            torch.ones(4, dtype=torch.bool)
            if fault == "rank"
            else torch.ones(2, 3, dtype=torch.bool) if fault == "nonsquare" else torch.ones(2, 2)
        )
        with pytest.raises(ValueError, match="square boolean pair mask"):
            smooth_lddt_pair_indices(mask)
