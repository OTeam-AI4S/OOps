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
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oops.attention.gate import gate_attention_output

_TOLERANCE = {
    torch.float64: (1e-12, 1e-12),
    torch.float32: (2e-3, 2e-3),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (3e-2, 3e-2),
}


def _reference(attention, single, gate_weight, output_weight):
    *prefix, heads, length, head_size = attention.shape
    attention = attention.transpose(-3, -2).reshape(*prefix, length, heads * head_size)
    return F.linear(torch.sigmoid(F.linear(single, gate_weight)) * attention, output_weight)


def _nested_case(
    device: torch.device | str,
    dtype: torch.dtype,
    *,
    explicit_sample: bool,
    lengths: tuple[int, ...] = (2, 4),
    requires_grad: bool,
) -> tuple[NestedTensor, NestedTensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    if explicit_sample:
        attention_template = NestedTensor(
            [torch.empty(3, 2, length, 4) for length in lengths],
            ragged_dims=(2,),
        )
        single_template = NestedTensor(
            [torch.empty(3, length, 5) for length in lengths],
            ragged_dims=(1,),
        )
    else:
        attention_template = NestedTensor(
            [torch.empty(2, length, 4) for length in lengths],
            ragged_dims=(1,),
        )
        single_template = NestedTensor(
            [torch.empty(length, 5) for length in lengths],
            ragged_dims=(0,),
        )
    attention_values = torch.randn(
        attention_template.concat.shape,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    single_values = torch.randn(
        single_template.concat.shape,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    gate_weight = torch.randn(8, 5, device=device, dtype=dtype, requires_grad=requires_grad)
    output_weight = torch.randn(7, 8, device=device, dtype=dtype, requires_grad=requires_grad)
    return (
        attention_template.packed_like(attention_values),
        single_template.packed_like(single_values),
        gate_weight,
        output_weight,
        (attention_values, single_values, gate_weight, output_weight),
    )


def _nested_reference(
    attention: NestedTensor,
    single: NestedTensor,
    gate_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    outputs = tuple(
        _reference(attention_element, single_element, gate_weight, output_weight)
        for attention_element, single_element in zip(torch.unbind(attention), torch.unbind(single), strict=True)
    )
    return torch.cat(tuple(output.movedim(-2, 0) for output in outputs))


def _case(device, dtype, layout, rows=37, *, master=False, active=()):
    torch.manual_seed(71)
    if layout.startswith("nested"):
        attention, single, gate, weight, raw = _nested_case(
            device, dtype, explicit_sample=layout == "nested_sampled", lengths=(3, 5), requires_grad=False
        )
        av, sv = raw[:2]
    else:
        av = torch.randn(2, 4, rows, 16, device=device, dtype=dtype)
        sv = torch.randn(2, rows, 48, device=device, dtype=dtype)
        gate = torch.randn(64, 48, device=device, dtype=dtype) / 48**0.5
        weight = torch.randn(31, 64, device=device, dtype=dtype) / 8
        if layout == "sliced":
            av = torch.stack((av, av), -1)[..., 0]
        if layout == "token":
            av = av.transpose(-3, -2).contiguous().transpose(-3, -2)
        if layout == "negative":
            av, sv, gate, weight = (torch._neg_view(tensor) for tensor in (av, sv, gate, weight))
    if master:
        gate, weight = gate.float(), weight.float()
    operands = tuple(t.detach().requires_grad_(index in active) for index, t in enumerate((av, sv, gate, weight)))
    if layout.startswith("nested"):
        attention, single = attention.packed_like(operands[0]), single.packed_like(operands[1])
    else:
        attention, single = operands[:2]
    return (attention, single, *operands[2:]), operands


def _expected(inputs):
    return _nested_reference(*inputs) if isinstance(inputs[0], NestedTensor) else _reference(*inputs)


@pytest.mark.usefixtures("compiler_cache")
class TestGateAttentionOutput:

    @pytest.mark.parametrize(
        "dtype,layout,mode,master,rows",
        [
            (torch.float64, "cpu", "eager", False, (7,)),
            (torch.float32, "head", "eager", False, (37,)),
            pytest.param(torch.float32, "negative", "eager", False, (37,), id="negative_view"),
            (torch.float16, "sliced", "compile", False, (33, 57)),
            (torch.bfloat16, "token", "eager", False, (4489,)),
            (torch.float32, "nested", "compile", False, (37,)),
            (torch.bfloat16, "nested_sampled", "eager", True, (37,)),
        ],
    )
    def test_forward(self, request, dtype, layout, mode, master, rows):
        device = "cpu" if layout == "cpu" else request.getfixturevalue("cuda")

        def run(*args):
            result = gate_attention_output(*args)
            return result.concat if isinstance(result, NestedTensor) else result

        execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run
        for length in rows:
            inputs, _ = _case(device, dtype, layout, length, master=master)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
                actual = execute(*inputs)
                expected = (
                    _expected(tuple(tensor.double() for tensor in inputs))
                    if layout == "negative"
                    else _expected(inputs)
                )
            torch.testing.assert_close(
                actual.double() if layout == "negative" else actual,
                expected,
                atol=_TOLERANCE[actual.dtype][0],
                rtol=_TOLERANCE[actual.dtype][1],
            )

    @pytest.mark.parametrize(
        "dtype,layout,active,mode,master",
        [
            (torch.float32, "head", (0, 1, 2, 3), "saved", False),
            (torch.float16, "sliced", (0, 1, 2, 3), "saved", False),
            (torch.float32, "head", (0,), "eager", False),
            (torch.bfloat16, "sliced", (1,), "compile", False),
            (torch.float32, "token", (2,), "eager", False),
            (torch.float16, "head", (3,), "eager", False),
            (torch.bfloat16, "nested", (0, 1, 2, 3), "checkpoint", False),
            (torch.bfloat16, "nested_sampled", (0, 1, 2, 3), "compile", True),
            (torch.float32, "head", (0, 1, 2, 3), "eager", True),
            pytest.param(torch.float32, "negative", (0, 1, 2, 3), "eager", False, id="negative_view"),
            pytest.param(
                torch.float32, "head", (0,), "eager", False, marks=pytest.mark.cuda_device(1), id="secondary_cuda"
            ),
        ],
    )
    def test_backward(self, cuda, dtype, layout, active, mode, master, saved_strided_hook):
        inputs, operands = _case(cuda, dtype, layout, master=master, active=active)
        leaves = tuple(operands[index] for index in active)
        reference_inputs, reference_leaves = inputs, leaves
        if layout == "negative":
            reference_inputs = tuple(tensor.detach().double().requires_grad_(tensor.requires_grad) for tensor in inputs)
            reference_leaves = tuple(reference_inputs[index] for index in active)

        def run(*args):
            out = gate_attention_output(*args)
            return out.concat if isinstance(out, NestedTensor) else out

        execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
            if mode == "saved":
                with saved_strided_hook():
                    actual = execute(*inputs)
            else:
                actual = checkpoint(execute, *inputs, use_reentrant=False) if mode == "checkpoint" else execute(*inputs)
            expected = _expected(reference_inputs)
        torch.testing.assert_close(
            actual.double() if layout == "negative" else actual,
            expected,
            atol=_TOLERANCE[actual.dtype][0],
            rtol=_TOLERANCE[actual.dtype][1],
        )
        grad = torch.randn_like(actual)
        if layout == "negative":
            grad = torch._neg_view(grad)
        for got, want, leaf in zip(
            torch.autograd.grad(actual, leaves, grad),
            torch.autograd.grad(expected, reference_leaves, grad.to(expected.dtype)),
            leaves,
            strict=True,
        ):
            assert got.shape == leaf.shape and got.dtype == leaf.dtype
            torch.testing.assert_close(
                got.double() if layout == "negative" else got,
                want,
                atol=_TOLERANCE[actual.dtype][0],
                rtol=_TOLERANCE[actual.dtype][1],
            )

    @pytest.mark.parametrize(
        "scenario", ["jvp", "higher_order", "rank", "shape", "mixed_nested", "nested_lengths", "live_weights"]
    )
    def test_contract(self, cuda, scenario):
        inputs, operands = _case(
            cuda, torch.float32, "head", 7, active=(0, 1, 2, 3) if scenario == "higher_order" else ()
        )
        if scenario == "nested_lengths":
            attention, _, gate, weight, _ = _nested_case(
                cuda, torch.float32, explicit_sample=False, lengths=(3, 5), requires_grad=False
            )
            _, single, _, _, _ = _nested_case(
                cuda, torch.float32, explicit_sample=False, lengths=(4, 4), requires_grad=False
            )
            with pytest.raises(ValueError, match="gate_attention_output"):
                gate_attention_output(attention, single, gate, weight)
            return
        if scenario == "jvp":
            tangents = tuple(torch.randn_like(t) for t in operands)
            actual = torch.func.jvp(gate_attention_output, operands, tangents)
            expected = torch.func.jvp(_reference, operands, tangents)
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)
            return
        if scenario == "higher_order":
            actual = gate_attention_output(*inputs)
            expected = _reference(*inputs)
            grad = torch.randn_like(actual)
            first = torch.autograd.grad(actual, operands, grad, create_graph=True)
            reference = torch.autograd.grad(expected, operands, grad, create_graph=True)
            second = torch.autograd.grad(sum(t.square().sum() for t in first), operands)
            target = torch.autograd.grad(sum(t.square().sum() for t in reference), operands)
            for got, want in zip(first + second, reference + target, strict=True):
                torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)
            return
        if scenario == "live_weights":
            execute = torch.compile(gate_attention_output, fullgraph=True, dynamic=True)
            with torch.no_grad():
                for shift in (0.01, -0.02):
                    inputs[2].add_(shift)
                    torch.testing.assert_close(execute(*inputs), _reference(*inputs), atol=2e-3, rtol=2e-3)
            return
        attention, single, gate, weight = inputs
        if scenario == "rank":
            attention = attention.flatten()
        elif scenario == "shape":
            gate = gate[:, :-1]
        else:
            single = NestedTensor([torch.randn(7, 48, device=cuda)], ragged_dims=(0,))
        with pytest.raises(ValueError):
            gate_attention_output(attention, single, gate, weight)
