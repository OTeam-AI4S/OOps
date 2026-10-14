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
from danling.tensors import NestedTensor
from torch.autograd import forward_ad
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oops import OOpsUnsupportedError, gated_pair_bias_attention
from tests.references.pair_bias_attention import pair_bias_attention_reference as _attention_reference

_TOLERANCE = {
    torch.float32: (5e-3, 5e-3),
    torch.float16: (5e-2, 5e-2),
    torch.bfloat16: (5e-2, 5e-2),
}


def _reference(single, query, key, value, mask, pair_bias, gate_weight, output_weight):
    attention = _attention_reference(query, key, value, mask, pair_bias).to(query.dtype)
    attention = attention.transpose(-3, -2).flatten(-2)
    return F.linear(torch.sigmoid(F.linear(single, gate_weight)) * attention, output_weight)


def _nested_gated_case(
    dtype,
    device,
    *,
    lengths=(3, 5),
    samples=2,
    heads=2,
    head_size=16,
    single_channels=19,
    output_channels=23,
    requires_grad=True,
    implicit_group=False,
):
    def packed(template):
        values = torch.randn(
            template.concat.shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        return template.packed_like(values), values

    qkv_template = (
        NestedTensor(
            [torch.empty(heads, length, head_size) for length in lengths],
            ragged_dims=(1,),
        )
        if implicit_group
        else NestedTensor(
            [torch.empty(samples, heads, length, head_size) for length in lengths],
            ragged_dims=(2,),
        )
    )
    query, query_values = packed(qkv_template)
    key, key_values = packed(qkv_template)
    value, value_values = packed(qkv_template)
    single_template = (
        NestedTensor([torch.empty(length, single_channels) for length in lengths], ragged_dims=(0,))
        if implicit_group
        else NestedTensor(
            [torch.empty(samples, length, single_channels) for length in lengths],
            ragged_dims=(1,),
        )
    )
    single, single_values = packed(single_template)
    bias_template = (
        NestedTensor([torch.empty(1, length, length) for length in lengths], ragged_dims=(1, 2))
        if implicit_group
        else NestedTensor(
            [torch.empty(1, 1, length, length) for length in lengths],
            ragged_dims=(2, 3),
        )
    )
    bias, bias_values = packed(bias_template)
    mask = (
        NestedTensor(
            [torch.ones(length, length, device=device, dtype=torch.bool) for length in lengths],
            ragged_dims=(0, 1),
        )
        if implicit_group
        else NestedTensor(
            [torch.ones(1, length, length, device=device, dtype=torch.bool) for length in lengths],
            ragged_dims=(1, 2),
        )
    )
    gate_weight = torch.randn(
        heads * head_size,
        single_channels,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    output_weight = torch.randn(
        output_channels,
        heads * head_size,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return (
        single,
        query,
        key,
        value,
        mask,
        bias,
        gate_weight,
        output_weight,
        (single_values, query_values, key_values, value_values, bias_values, gate_weight, output_weight),
    )


def _independent_nested_gated(single, query, key, value, mask, bias, gate_weight, output_weight):
    return tuple(
        _reference(s, q, k, v, m, b, gate_weight, output_weight)
        for s, q, k, v, m, b in zip(
            torch.unbind(single),
            torch.unbind(query),
            torch.unbind(key),
            torch.unbind(value),
            torch.unbind(mask),
            torch.unbind(bias),
        )
    )


def _flatten_nested_gated(elements):
    return torch.cat(tuple(element.movedim(-2, 0) for element in elements))


def _case(cuda, dtype, layout, *, active=(), master=False):
    case = _nested_gated_case(
        dtype,
        cuda,
        lengths=(
            (128,)
            if layout == "dense_wide"
            else (
                (3, 5)
                if layout not in ("dense", "negative", "dense_broadcast", "empty_single", "empty_output")
                else (9,)
            )
        ),
        implicit_group=layout == "nested_implicit",
        head_size=128 if layout == "dense_wide" else 16,
        requires_grad=False,
        single_channels=0 if layout == "empty_single" else 19,
        output_channels=0 if layout == "empty_output" else 23,
    )
    single, q, k, v, mask, bias, gate, weight, raw = case
    values = tuple(
        (t.float() if master and index in (4, 5, 6) else t).detach().requires_grad_(index in active)
        for index, t in enumerate(raw)
    )
    if layout == "negative":
        values = tuple(torch._neg_view(value) for value in values)
    single = single.packed_like(values[0])
    q, k, v = (wrapper.packed_like(tensor) for wrapper, tensor in zip((q, k, v), values[1:4], strict=True))
    bias = bias.packed_like(values[4])
    gate, weight = values[5:]
    if layout in ("dense", "negative", "dense_wide", "dense_broadcast", "empty_single", "empty_output"):
        single, q, k, v, mask, bias = (torch.unbind(t)[0] for t in (single, q, k, v, mask, bias))
    if layout == "dense_wide":
        values = tuple(
            tensor.contiguous().detach().requires_grad_(index in active)
            for index, tensor in enumerate((single, q, k, v, bias, gate, weight))
        )
        single, q, k, v, bias, gate, weight = values
    if layout == "dense_broadcast":
        bias = bias[..., :1]
        mask = None
    return (single, q, k, v, mask, bias, gate, weight), tuple(values[index] for index in active)


def _reference_output(inputs):
    return (
        _flatten_nested_gated(_independent_nested_gated(*inputs))
        if isinstance(inputs[0], NestedTensor)
        else _reference(*inputs)
    )


@pytest.mark.usefixtures("compiler_cache")
class TestGatedPairBiasAttention:

    @pytest.mark.parametrize(
        "dtype,layout,compiled,master",
        [
            (torch.float32, "dense", False, False),
            (torch.float32, "dense_broadcast", False, False),
            (torch.float32, "empty_single", False, False),
            (torch.float32, "empty_output", False, False),
            (torch.float32, "negative", False, False),
            (torch.float16, "nested", False, False),
            (torch.bfloat16, "nested_implicit", True, False),
            (torch.bfloat16, "dense", True, True),
        ],
    )
    def test_forward(self, cuda, dtype, layout, compiled, master):
        inputs, _ = _case(cuda, dtype, layout, master=master)

        def run(*values):
            out = gated_pair_bias_attention(*values)
            return out.concat if isinstance(out, NestedTensor) else out

        execute = (
            torch.compile(run, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
            if compiled
            else run
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
            actual = execute(*inputs)
            reference_inputs = (
                tuple(value.double() if value is not None and value.is_floating_point() else value for value in inputs)
                if layout in ("negative", "dense_wide", "dense_broadcast")
                else inputs
            )
            expected = _reference_output(reference_inputs).to(actual.dtype)
        torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])

    @pytest.mark.parametrize(
        "dtype,layout,active,mode,master",
        [
            (torch.float32, "dense", tuple(range(7)), "eager", False),
            (torch.float32, "dense_broadcast", tuple(range(7)), "eager", False),
            (torch.float32, "empty_single", tuple(range(7)), "eager", False),
            (torch.float32, "empty_output", tuple(range(7)), "eager", False),
            (torch.float32, "negative", tuple(range(7)), "eager", False),
            (torch.float32, "dense_wide", tuple(range(7)), "eager", False),
            (torch.bfloat16, "nested", tuple(range(7)), "compile", True),
            (torch.float16, "nested_implicit", tuple(range(7)), "checkpoint", False),
            (torch.float32, "dense", (0,), "eager", False),
            (torch.bfloat16, "nested", (5,), "eager", False),
            (torch.float32, "dense", (6,), "eager", False),
            (torch.bfloat16, "nested_implicit", (1,), "eager", False),
            (torch.bfloat16, "nested_implicit", tuple(range(7)), "saved", False),
            pytest.param(
                torch.float32,
                "dense",
                tuple(range(7)),
                "eager",
                False,
                marks=pytest.mark.cuda_device(1),
                id="secondary_cuda",
            ),
        ],
    )
    def test_backward(self, cuda, dtype, layout, active, mode, master, saved_strided_hook):
        inputs, leaves = _case(cuda, dtype, layout, active=active, master=master)

        def run(*values):
            out = gated_pair_bias_attention(*values)
            return out.concat if isinstance(out, NestedTensor) else out

        execute = (
            torch.compile(run, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
            if mode == "compile"
            else run
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
            with saved_strided_hook() if mode == "saved" else nullcontext():
                actual = checkpoint(execute, *inputs, use_reentrant=False) if mode == "checkpoint" else execute(*inputs)
            reference_inputs = (
                tuple(value.double() if value is not None and value.is_floating_point() else value for value in inputs)
                if layout in ("negative", "dense_wide", "dense_broadcast")
                else inputs
            )
            expected = _reference_output(reference_inputs).to(actual.dtype)
        torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
        grad = torch.randn_like(actual)
        if layout == "negative":
            grad = torch._neg_view(grad)
        actual_gradients = torch.autograd.grad(actual, leaves, grad)
        for got, want, leaf in zip(actual_gradients, torch.autograd.grad(expected, leaves, grad), leaves, strict=True):
            assert got.dtype == leaf.dtype and got.shape == leaf.shape
            torch.testing.assert_close(got, want, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])

    @pytest.mark.parametrize("scenario", ["gate_jvp", "attention_jvp", "masked", "missing_weight", "shape"])
    def test_contract(self, cuda, scenario):
        inputs, _ = _case(cuda, torch.float32, "dense")
        single, q, k, v, mask, bias, gate, weight = inputs
        if scenario == "gate_jvp":
            primals = (single, gate, weight)
            tangents = tuple(torch.randn_like(t) for t in primals)
            with forward_ad.dual_level():
                s, g, w = tuple(forward_ad.make_dual(p, t) for p, t in zip(primals, tangents, strict=True))
                actual = forward_ad.unpack_dual(gated_pair_bias_attention(s, q, k, v, mask, bias, g, w))
                expected = forward_ad.unpack_dual(_reference(s, q, k, v, mask, bias, g, w))
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, want, atol=5e-3, rtol=5e-3)
            return
        if scenario == "attention_jvp":
            with forward_ad.dual_level():
                q = forward_ad.make_dual(q, torch.ones_like(q))
                with pytest.raises(OOpsUnsupportedError):
                    gated_pair_bias_attention(single, q, k, v, mask, bias, gate, weight)
            return
        if scenario == "masked":
            mask = torch.zeros_like(mask)
            actual = gated_pair_bias_attention(single, q, k, v, mask, bias, gate, weight)
            torch.testing.assert_close(
                actual, _reference(single, q, k, v, mask, bias, gate, weight), atol=5e-3, rtol=5e-3
            )
            return
        if scenario == "missing_weight":
            gate = None
        else:
            weight = weight[:, :-1]
        with pytest.raises(ValueError):
            gated_pair_bias_attention(single, q, k, v, mask, bias, gate, weight)
