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
from danling.tensors import NestedTensor

from oops import OOpsUnsupportedError, pair_mask_from_token_mask


def _reference(mask):
    token = mask.bool()
    return token[..., :, None] & token[..., None, :]


class TestPairMaskFromTokenMask:

    @pytest.mark.parametrize(
        ("kind", "dtype", "mode"),
        [
            pytest.param("nested", torch.float32, "eager", marks=pytest.mark.cuda_device(1), id="secondary-cuda"),
            ("dense", torch.bool, "eager"),
            ("dense", torch.int64, "eager"),
            ("dense", torch.float32, "eager"),
            ("nested", torch.bool, "compile"),
            ("nested", torch.float32, "eager"),
            ("empty", torch.bool, "eager"),
        ],
    )
    def test_forward(self, request, kind, dtype, mode):
        if kind == "dense":
            mask = torch.tensor([[1, 0, 2], [0, -1, 3]], dtype=dtype)
            actual, expected = pair_mask_from_token_mask(mask), _reference(mask)
            assert actual.dtype is torch.bool
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            return
        device = request.getfixturevalue("cuda")
        operation = (
            torch.compile(pair_mask_from_token_mask, fullgraph=True, dynamic=True)
            if mode == "compile"
            else pair_mask_from_token_mask
        )
        initial = (
            (torch.empty(0), torch.empty(0)) if kind == "empty" else (torch.tensor([1, 0]), torch.tensor([-2, 0, 3]))
        )
        for parts in (initial, (torch.tensor([0]), torch.tensor([1, 0, 1, 1, 0]))) if mode == "compile" else (initial,):
            parts = tuple(part.to(device=device, dtype=dtype) for part in parts)
            mask = NestedTensor(parts, ragged_dims=(0,))
            expected_parts = tuple(_reference(part) for part in parts)
            expected = torch.cat([part.flatten() for part in expected_parts])
            actual = operation(mask)
            assert isinstance(actual, NestedTensor)
            assert actual.dtype is torch.bool
            torch.testing.assert_close(actual.concat, expected, atol=0, rtol=0)
            assert actual.element_sizes().tolist() == [list(part.shape) for part in expected_parts]

    @pytest.mark.parametrize("case", ["rank", "cpu_nested", "complex", "nested_rank"])
    def test_contract(self, case):
        if case == "rank":
            with pytest.raises(ValueError):
                pair_mask_from_token_mask(torch.tensor(True))
            return
        shape = (2,)
        if case == "nested_rank":
            shape = (*shape, 1)
        dtype = torch.complex64 if case == "complex" else torch.bool
        mask = NestedTensor([torch.ones(shape, dtype=dtype)], ragged_dims=(0,))
        error = ValueError if case == "nested_rank" else OOpsUnsupportedError
        with pytest.raises(error):
            pair_mask_from_token_mask(mask)
