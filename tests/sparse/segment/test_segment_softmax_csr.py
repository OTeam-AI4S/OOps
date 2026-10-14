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

from contextlib import nullcontext

import pytest
import torch
from torch.autograd import forward_ad

from oops import CSRGraph, OOpsUnsupportedError, segment_softmax_csr


def _case(cuda, dtype, lengths, tail, values_kind="finite"):
    torch.manual_seed(6)
    values = torch.randn(sum(lengths), *tail, device=cuda, dtype=dtype)
    if values_kind == "masked_prefix":
        values[: values.shape[0] // 2] = -torch.inf
    elif values_kind == "all_masked":
        values.fill_(-torch.inf)
    elif values_kind == "positive_infinity":
        values[-1] = torch.inf
    elif values_kind == "nan":
        values[-1] = torch.nan
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=cuda)
    return (values, offsets)


def _reference(values, lengths):
    return torch.cat([part.softmax(0) for part in values.split(lengths)])


class TestSegmentSoftmaxCSR:

    @pytest.mark.parametrize(
        ("dtype", "lengths", "tail", "metadata", "mode", "values_kind"),
        [
            (torch.float16, (0, 1, 17, 129, 2049), (3,), "offsets", "eager", "finite"),
            (torch.bfloat16, (0, 8193, 4099), (65,), "offsets", "eager", "finite"),
            (torch.float32, (0, 3, 0, 8), (2, 3), "offsets", "compile", "finite"),
            (torch.float64, (0, 2, 5), (3,), "segments", "eager", "finite"),
            (torch.float32, (2, 0, 3), (4,), "graph", "eager", "finite"),
            (torch.float32, (8193, *(1,) * 511), (3,), "offsets", "eager", "finite"),
            (torch.float32, (0, 17, 8193), (17,), "offsets", "negative", "finite"),
            (torch.float32, (65,), (2,), "offsets", "eager", "masked_prefix"),
            (torch.bfloat16, (4096,), (2,), "offsets", "eager", "masked_prefix"),
            (torch.float64, (4096,), (2,), "offsets", "eager", "masked_prefix"),
            (torch.float32, (33,), (2,), "offsets", "eager", "all_masked"),
            (torch.float32, (65,), (2,), "offsets", "eager", "positive_infinity"),
            (torch.float32, (65,), (2,), "offsets", "eager", "nan"),
            (torch.float32, (8193, *(1,) * 511), (1,), "offsets", "compile", "masked_prefix"),
            (torch.float64, (8193, *(1,) * 31), (1,), "offsets", "eager", "finite"),
            (torch.float32, (8193, *(1,) * 511), (1,), "offsets", "eager", "all_masked"),
        ],
    )
    def test_forward(self, cuda, dtype, lengths, tail, metadata, mode, values_kind, tolerance):
        operation = (
            torch.compile(segment_softmax_csr, fullgraph=True, dynamic=True)
            if mode == "compile"
            else segment_softmax_csr
        )
        for current in (lengths, (1, 0, 9)) if mode == "compile" else (lengths,):
            lengths = current
            values, offsets = _case(cuda, dtype, lengths, tail, values_kind)
            if mode == "negative":
                values = torch._neg_view(values)
            segment_ids = torch.repeat_interleave(
                torch.arange(len(lengths), device=cuda), torch.tensor(lengths, device=cuda)
            )
            descriptor = CSRGraph.from_row_offsets(offsets, segment_ids=segment_ids) if metadata == "graph" else offsets
            kwargs = {"segment_ids": segment_ids.int()} if metadata == "segments" else {}
            actual = operation(values, descriptor, **kwargs)
            expected = _reference(values.double(), lengths)
            torch.testing.assert_close(actual.double(), expected, equal_nan=True, **tolerance(dtype))
            repeated = operation(values, descriptor, **kwargs)
            torch.testing.assert_close(actual, repeated, equal_nan=True, atol=0, rtol=0)

    @pytest.mark.parametrize(
        ("dtype", "lengths", "tail", "mode", "values_kind"),
        [
            (torch.float16, (0, 1, 17, 129, 2049), (3,), "eager", "finite"),
            (torch.bfloat16, (0, 8193, 4099), (65,), "eager", "finite"),
            (torch.float32, (0, 3, 0, 8), (2, 3), "compile", "finite"),
            (torch.float32, (8193, *(1,) * 511), (3,), "eager", "finite"),
            (torch.float64, (1, 6, 10), (3,), "eager", "finite"),
            (torch.float32, (0, 17, 8193), (17,), "negative", "finite"),
            (torch.float32, (0, 17, 8193), (17,), "saved-hook", "finite"),
            (torch.float32, (0, 17, 8193), (1,), "saved-strided", "finite"),
            (torch.float32, (0, 17, 8193), (17,), "saved-strided", "finite"),
            (torch.float64, (1, 6, 10), (3,), "higher_order", "finite"),
            (torch.float32, (65,), (2,), "compile", "masked_prefix"),
            (torch.bfloat16, (4096,), (2,), "eager", "masked_prefix"),
            (torch.float64, (4096,), (2,), "eager", "masked_prefix"),
            (torch.float32, (8193, *(1,) * 511), (1,), "compile", "masked_prefix"),
            (torch.bfloat16, (8193, *(1,) * 511), (1,), "eager", "finite"),
            pytest.param(
                torch.float16,
                (0, 1, 17, 129, 2049),
                (3,),
                "eager",
                "finite",
                marks=pytest.mark.cuda_device(1),
                id="secondary_cuda",
            ),
        ],
    )
    def test_backward(self, cuda, dtype, lengths, tail, mode, values_kind, tolerance, saved_strided_hook):
        values, offsets = _case(cuda, dtype, lengths, tail, values_kind)
        if mode == "negative":
            values = torch._neg_view(values)
        values.requires_grad_()
        reference_values = values.detach().double().requires_grad_()
        operation = (
            torch.compile(segment_softmax_csr, fullgraph=True, dynamic=True)
            if mode == "compile"
            else segment_softmax_csr
        )
        hooks = (
            saved_strided_hook()
            if mode == "saved-strided"
            else (
                torch.autograd.graph.saved_tensors_hooks(
                    lambda tensor: -tensor if tensor.is_floating_point() else tensor,
                    lambda tensor: torch._neg_view(tensor) if tensor.is_floating_point() else tensor,
                )
                if mode == "saved-hook"
                else nullcontext()
            )
        )
        with hooks:
            output = operation(values, offsets)
        reference = _reference(reference_values, lengths)
        upstream = torch.randn_like(output)
        if mode in ("negative", "saved-hook"):
            upstream = torch._neg_view(upstream)
        higher = mode == "higher_order"
        actual = torch.autograd.grad(output, values, upstream, create_graph=higher)[0]
        expected = torch.autograd.grad(reference, reference_values, upstream.double(), create_graph=higher)[0]
        torch.testing.assert_close(actual.double(), expected, equal_nan=True, **tolerance(dtype))
        if mode == "higher_order":
            direction = torch.randn_like(values)
            actual = torch.autograd.grad(actual, values, direction)[0]
            expected = torch.autograd.grad(expected, reference_values, direction.double())[0]
            torch.testing.assert_close(actual.double(), expected, equal_nan=True, **tolerance(dtype))

    @pytest.mark.parametrize("case", ["cpu", "integer", "empty", "forward_ad"])
    def test_contract(self, request, case):
        device = "cpu" if case in ("cpu", "forward_ad") else request.getfixturevalue("cuda")
        values = torch.ones(
            0 if case == "empty" else 4, 2, device=device, dtype=torch.int64 if case == "integer" else torch.float32
        )
        offsets = torch.tensor([0, values.shape[0]], device=device)
        if case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(values, torch.ones_like(values))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    segment_softmax_csr(dual, offsets)
        else:
            with pytest.raises(ValueError if case == "integer" else OOpsUnsupportedError):
                segment_softmax_csr(values, offsets)
