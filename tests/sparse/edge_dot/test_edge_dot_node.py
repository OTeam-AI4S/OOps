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
from torch.autograd import forward_ad

from oops import OOpsUnsupportedError, edge_dot_node


def _case(cuda, dtype, edges=4096):
    torch.manual_seed(5)
    edge = torch.randn(edges, 2, 16, device=cuda, dtype=dtype)
    node = torch.randn(64, 2, 16, device=cuda, dtype=dtype)
    index = torch.arange(edges, device=cuda) % 64
    return edge, node, index


def _reference(edge, node, index):
    return (edge * node.index_select(0, index.long())).sum(-1)


class TestEdgeDotNode:

    @pytest.mark.parametrize(
        ("dtype", "mode"),
        [
            (torch.float16, "eager"),
            (torch.bfloat16, "eager"),
            (torch.float32, "compile"),
            (torch.float32, "static-compile"),
            (torch.float32, "negative"),
            (torch.float64, "eager"),
        ],
    )
    def test_forward(self, cuda, dtype, mode, tolerance):
        operation = (
            torch.compile(edge_dot_node, fullgraph=True, dynamic=mode == "compile")
            if mode in ("compile", "static-compile")
            else edge_dot_node
        )
        for edges in (17, 33) if mode == "compile" else (17 if mode == "static-compile" else 4096,):
            edge, node, index = _case(cuda, dtype, edges)
            if mode == "negative":
                edge, node = torch._neg_view(edge), torch._neg_view(node)
            actual = operation(edge, node, index.int() if dtype == torch.float16 else index)
            expected = _reference(edge.double(), node.double(), index)
            torch.testing.assert_close(actual.double(), expected, **tolerance(dtype))

    @pytest.mark.parametrize(
        ("dtype", "mode"),
        [
            (torch.float16, "eager"),
            (torch.bfloat16, "eager"),
            (torch.float32, "deterministic"),
            (torch.float32, "skew"),
            (torch.bfloat16, "mixed"),
            (torch.float32, "saved-strided"),
            (torch.float32, "negative"),
            (torch.float64, "compile"),
            pytest.param(torch.float16, "eager", marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
        ],
    )
    def test_backward(self, cuda, dtype, mode, tolerance, saved_strided_hook):
        edge, node, index = _case(cuda, dtype)
        if mode == "skew":
            index.zero_()
        elif mode == "mixed":
            index.remainder_(2)
        if mode == "negative":
            edge, node = torch._neg_view(edge), torch._neg_view(node)
        inputs = (edge.requires_grad_(), node.requires_grad_())
        references = tuple(value.detach().double().requires_grad_() for value in inputs)
        upstream = torch.randn(edge.shape[:-1], device=cuda, dtype=dtype)
        if mode == "negative":
            upstream = torch._neg_view(upstream)
        expected = _reference(*references, index)
        expected_gradients = torch.autograd.grad(expected, references, upstream.double())
        previous, previous_warn = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        try:
            torch.use_deterministic_algorithms(mode == "deterministic")
            operation = (
                torch.compile(edge_dot_node, fullgraph=True, dynamic=True) if mode == "compile" else edge_dot_node
            )
            if mode == "saved-strided":
                with saved_strided_hook():
                    output = operation(*inputs, index)
            else:
                output = operation(*inputs, index)
            gradients = torch.autograd.grad(output, inputs, upstream, retain_graph=mode == "deterministic")
            if mode == "deterministic":
                repeated = torch.autograd.grad(output, inputs, upstream)
                for first, second in zip(gradients, repeated, strict=True):
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
        for result, reference in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(result.double(), reference, **tolerance(dtype))
        if dtype in (torch.float16, torch.bfloat16):
            relative = (gradients[1].double() - expected_gradients[1]).norm() / expected_gradients[1].norm().clamp_min(
                1
            )
            assert relative < (2e-2 if dtype == torch.bfloat16 else 5e-3)

    @pytest.mark.parametrize("case", ["cpu", "feature_shape", "index_shape", "index_dtype", "forward_ad"])
    def test_contract(self, case):
        edge, node = torch.randn(8, 2, 4), torch.randn(4, 2, 3 if case == "feature_shape" else 4)
        index = torch.zeros(
            7 if case == "index_shape" else 8, dtype=torch.float32 if case == "index_dtype" else torch.int64
        )
        if case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(edge, torch.ones_like(edge))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    edge_dot_node(dual, node, index)
        else:
            with pytest.raises(OOpsUnsupportedError if case == "cpu" else ValueError):
                edge_dot_node(edge, node, index)
