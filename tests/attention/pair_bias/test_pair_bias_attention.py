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
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.autograd import forward_ad
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils.checkpoint import checkpoint

from oops import OOpsUnsupportedError, pair_bias_attention
from tests.references.pair_bias_attention import pair_bias_attention_reference as _reference

_TOLERANCE = {
    torch.float32: (5e-3, 5e-3),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (5e-2, 5e-2),
}


def _inputs(dtype, device, *, prefix=(2,), requires_grad=True, shape=(4, 32, 32), key_length=None):
    torch.manual_seed(0)
    heads, query_length, head_size = shape
    key_length = query_length if key_length is None else key_length
    query = torch.randn(
        (*prefix, heads, query_length, head_size), device=device, dtype=dtype, requires_grad=requires_grad
    )
    key, value = (
        torch.randn((*prefix, heads, key_length, head_size), device=device, dtype=dtype, requires_grad=requires_grad)
        for _ in range(2)
    )
    return query, key, value


def _nested_inputs(
    dtype,
    device,
    *,
    query_lengths=(3, 5),
    key_lengths=(4, 2),
    samples=2,
    heads=2,
    head_size=16,
    mask_kind="pair",
    mask_shared=True,
    bias_group_shared=True,
    bias_head_shared=True,
    with_bias=True,
    requires_grad=True,
    implicit_group=False,
):
    def qkv(lengths):
        if implicit_group:
            template = NestedTensor(
                [torch.empty(heads, length, head_size) for length in lengths],
                ragged_dims=(1,),
            )
        else:
            template = NestedTensor(
                [torch.empty(samples, heads, length, head_size) for length in lengths],
                ragged_dims=(2,),
            )
        values = torch.randn(
            template.concat.shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        return template.packed_like(values), values

    query, query_values = qkv(query_lengths)
    key, key_values = qkv(key_lengths)
    value, value_values = qkv(key_lengths)
    mask_groups = 1 if mask_shared else samples
    if mask_kind == "none":
        mask = None
    elif mask_kind == "key":
        mask_template = (
            NestedTensor([torch.empty(length, dtype=torch.bool) for length in key_lengths], ragged_dims=(0,))
            if implicit_group
            else NestedTensor(
                [torch.empty(mask_groups, length, dtype=torch.bool) for length in key_lengths],
                ragged_dims=(1,),
            )
        )
        mask = mask_template.packed_like(torch.ones(mask_template.concat.shape, device=device, dtype=torch.bool))
    elif mask_kind == "pair":
        if implicit_group:
            mask_template = NestedTensor(
                [
                    torch.empty(query_length, key_length, dtype=torch.bool)
                    for query_length, key_length in zip(query_lengths, key_lengths)
                ],
                ragged_dims=(0, 1),
            )
        else:
            mask_template = NestedTensor(
                [
                    torch.empty(mask_groups, query_length, key_length, dtype=torch.bool)
                    for query_length, key_length in zip(query_lengths, key_lengths)
                ],
                ragged_dims=(1, 2),
            )
        mask = mask_template.packed_like(torch.ones(mask_template.concat.shape, device=device, dtype=torch.bool))
    else:
        raise ValueError(f"unknown mask kind {mask_kind}")

    bias_values = None
    if not with_bias:
        bias = None
    else:
        bias_groups = 1 if bias_group_shared else samples
        bias_heads = 1 if bias_head_shared else heads
        if implicit_group:
            bias_template = NestedTensor(
                [
                    torch.empty(bias_heads, query_length, key_length)
                    for query_length, key_length in zip(query_lengths, key_lengths)
                ],
                ragged_dims=(1, 2),
            )
        else:
            bias_template = NestedTensor(
                [
                    torch.empty(bias_groups, bias_heads, query_length, key_length)
                    for query_length, key_length in zip(query_lengths, key_lengths)
                ],
                ragged_dims=(2, 3),
            )
        bias_values = torch.randn(
            bias_template.concat.shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        bias = bias_template.packed_like(bias_values)
    return query, key, value, mask, bias, (query_values, key_values, value_values), bias_values


def _independent_nested_pair_bias(query, key, value, mask, bias, *, scale=None):
    query_parts = torch.unbind(query)
    key_parts = torch.unbind(key)
    value_parts = torch.unbind(value)
    mask_parts = (None,) * len(query_parts) if mask is None else torch.unbind(mask)
    bias_parts = (None,) * len(query_parts) if bias is None else torch.unbind(bias)
    return tuple(
        _reference(q, k, v, m, b, scale=scale).to(q.dtype)
        for q, k, v, m, b in zip(query_parts, key_parts, value_parts, mask_parts, bias_parts)
    )


def _flatten_nested_pair_bias(elements):
    return torch.cat(tuple(element.movedim(-2, 0) for element in elements))


def _case(cuda, dtype, layout, mask_kind, bias_kind, *, active=(), lengths=(33, 35), head_size=24):
    if layout.startswith("nested"):
        q, k, v, mask, bias, raw, bv = _nested_inputs(
            dtype,
            cuda,
            query_lengths=(3, 5),
            key_lengths=(4, 2),
            mask_kind=mask_kind,
            with_bias=bias_kind != "none",
            bias_group_shared=bias_kind == "shared",
            implicit_group=layout == "nested_implicit",
            requires_grad=False,
        )
    else:
        q, k, v = _inputs(
            dtype,
            cuda,
            prefix=(2, 3) if layout == "sampled" else (2,),
            requires_grad=False,
            shape=(2, lengths[0], head_size),
            key_length=lengths[1],
        )
        if layout == "sliced":
            q, k, v = (torch.stack((t, t), -1)[..., 0] for t in (q, k, v))
        prefix = q.shape[:-3]
        rows, columns = q.shape[-2], k.shape[-2]
        shape = (*prefix, columns) if mask_kind == "key" else (*prefix, rows, columns)
        mask = None if mask_kind == "none" else torch.rand(shape, device=cuda) > 0.2
        if mask is not None:
            mask[..., 0] = True
        bias_prefix = (*prefix[:-1], 1) if bias_kind == "shared" and len(prefix) > 1 else prefix
        bias = None if bias_kind == "none" else torch.randn((*bias_prefix, 2, rows, columns), device=cuda, dtype=dtype)
        raw, bv = (q, k, v), bias
    names = ("query", "key", "value", "bias")
    tensors = tuple(
        None if t is None else t.detach().requires_grad_(name in active)
        for name, t in zip(names, (*raw, bv), strict=True)
    )
    if layout == "negative":
        tensors = tuple(None if tensor is None else torch._neg_view(tensor) for tensor in tensors)
    if layout.startswith("nested"):
        q, k, v = (wrapper.packed_like(tensor) for wrapper, tensor in zip((q, k, v), tensors[:3], strict=True))
        bias = None if bias is None else bias.packed_like(tensors[3])
    else:
        q, k, v, bias = tensors
    leaves = tuple(t for t in tensors if t is not None and t.requires_grad)
    return (q, k, v, mask, bias), leaves


def _reference_output(inputs, scale):
    q, k, v, mask, bias = inputs
    if isinstance(q, NestedTensor):
        return _flatten_nested_pair_bias(_independent_nested_pair_bias(q, k, v, mask, bias, scale=scale))
    return _reference(q, k, v, mask, bias, scale=scale).to(q.dtype)


@pytest.mark.usefixtures("compiler_cache")
class TestPairBiasAttention:

    @pytest.mark.parametrize(
        "dtype,layout,mask_kind,bias_kind,mode,lengths",
        [
            (torch.float32, "dense", "pair", "none", "eager", (33, 35)),
            (torch.float32, "dense", "none", "none", "eager", (1, 31)),
            (torch.float32, "negative", "pair", "full", "eager", (33, 35)),
            (torch.float16, "sliced", "key", "full", "eager", (33, 35)),
            (torch.bfloat16, "sampled", "pair", "shared", "compile", (31, 129)),
            (torch.bfloat16, "nested", "pair", "shared", "eager", (3, 5)),
            (torch.float32, "nested_implicit", "key", "full", "compile", (3, 5)),
        ],
    )
    def test_forward(self, cuda, dtype, layout, mask_kind, bias_kind, mode, lengths):

        def run(*inputs):
            result = pair_bias_attention(*inputs, scale=0.25)
            return result.concat if isinstance(result, NestedTensor) else result

        execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run
        for sizes in (
            (lengths, (lengths[0] + 2, lengths[1] + 4))
            if mode == "compile" and not layout.startswith("nested")
            else (lengths,)
        ):
            inputs, _ = _case(cuda, dtype, layout, mask_kind, bias_kind, lengths=sizes)
            actual = execute(*inputs)
            expected = _reference_output(inputs, 0.25)
            torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])

    @pytest.mark.parametrize(
        "dtype,layout,mask_kind,bias_kind,active,mode,shape",
        [
            (torch.float32, "dense", "pair", "full", ("query", "key", "value", "bias"), "eager", (33, 129, 24)),
            (torch.float32, "dense", "pair", "none", ("query", "key", "value"), "eager", (17, 19, 16)),
            (torch.float32, "dense", "pair", "full", ("query", "key", "value", "bias"), "cpu_saved", (17, 19, 32)),
            (torch.float32, "dense", "pair", "full", ("query", "key", "value", "bias"), "strided_saved", (17, 19, 32)),
            (torch.float32, "negative", "pair", "full", ("query", "key", "value", "bias"), "eager", (33, 35, 24)),
            (torch.float16, "sliced", "key", "shared", ("value",), "compile", (33, 129, 24)),
            (torch.bfloat16, "sampled", "pair", "full", ("query", "value"), "eager", (33, 129, 24)),
            (torch.float32, "dense", "pair", "shared", ("bias",), "eager", (33, 129, 24)),
            (torch.bfloat16, "dense", "none", "none", ("key",), "eager", (33, 129, 24)),
            (
                torch.bfloat16,
                "nested",
                "pair",
                "shared",
                ("query", "key", "value", "bias"),
                "checkpoint",
                (33, 129, 24),
            ),
            (torch.float32, "nested_implicit", "key", "full", ("bias",), "compile", (33, 129, 24)),
            (torch.float32, "dense", "pair", "shared", ("query", "key", "value", "bias"), "eager", (128, 128, 128)),
            pytest.param(
                torch.float32,
                "dense",
                "pair",
                "full",
                ("query", "key", "value", "bias"),
                "eager",
                (33, 129, 24),
                marks=pytest.mark.cuda_device(1),
                id="secondary_cuda",
            ),
        ],
    )
    def test_backward(self, cuda, dtype, layout, mask_kind, bias_kind, active, mode, shape):
        inputs, leaves = _case(
            cuda, dtype, layout, mask_kind, bias_kind, active=active, lengths=shape[:2], head_size=shape[2]
        )

        def run(*values):
            result = pair_bias_attention(*values, scale=0.25)
            return result.concat if isinstance(result, NestedTensor) else result

        execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run

        def unpack(tensor):
            if tensor.is_floating_point() and tensor.ndim >= 3:
                expanded = tensor.new_empty((*tensor.shape[:-1], tensor.shape[-1] + 4))
                unpacked = expanded[..., : tensor.shape[-1]]
                unpacked.copy_(tensor)
                return unpacked
            return tensor

        saved = (
            torch.autograd.graph.save_on_cpu()
            if mode == "cpu_saved"
            else (
                torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, unpack)
                if mode == "strided_saved"
                else nullcontext()
            )
        )
        with saved:
            actual = checkpoint(execute, *inputs, use_reentrant=False) if mode == "checkpoint" else execute(*inputs)
        expected = _reference_output(inputs, 0.25)
        torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
        upstream = torch.randn_like(actual)
        if layout == "negative":
            upstream = torch._neg_view(upstream)
        for got, want, leaf in zip(
            torch.autograd.grad(actual, leaves, upstream),
            torch.autograd.grad(expected, leaves, upstream),
            leaves,
            strict=True,
        ):
            assert got.shape == leaf.shape and got.dtype == leaf.dtype
            torch.testing.assert_close(got, want, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])

    @pytest.mark.parametrize(
        "scenario",
        [
            "masked",
            "alias",
            "determinism",
            "precision",
            "autocast_bias",
            "fake",
            "cpu",
            "shape",
            "mask_head",
            "forward_ad",
            "mixed_nested",
        ],
    )
    def test_contract(self, request, scenario):
        if scenario == "fake":
            shape_env = ShapeEnv()
            query_length = shape_env.create_unbacked_symint()
            key_length = shape_env.create_unbacked_symint()
            torch._check(query_length >= 2)
            torch._check(key_length >= 2)
            with FakeTensorMode(shape_env=shape_env):
                q, k, v = _inputs(
                    torch.float32,
                    "cuda",
                    requires_grad=False,
                    shape=(2, query_length, 24),
                    key_length=key_length,
                )
                mask = torch.empty((*q.shape[:-3], query_length, key_length), device=q.device, dtype=torch.bool)
                bias = torch.empty((*q.shape[:-1], key_length), device=q.device, dtype=q.dtype)
                output = pair_bias_attention(q, k, v, mask, bias)
            assert is_fake(output) and output.shape == q.shape and output.dtype == q.dtype
            return
        device = "cpu" if scenario == "cpu" else request.getfixturevalue("cuda")
        dtype = torch.bfloat16 if scenario == "autocast_bias" else torch.float32
        inputs, leaves = _case(
            device,
            dtype,
            "dense",
            "pair",
            "shared",
            active=("query", "key", "value", "bias"),
            lengths=(128, 128) if scenario == "precision" else (33, 33),
            head_size=128 if scenario == "precision" else 24,
        )
        q, k, v, mask, bias = inputs
        if scenario == "precision":
            previous = torch.backends.cuda.matmul.fp32_precision
            try:
                torch.backends.cuda.matmul.fp32_precision = "ieee"
                actual = pair_bias_attention(q, k, v, mask, bias)
                upstream = torch.randn_like(actual)
                captured = torch.autograd.grad(actual, leaves, upstream, retain_graph=True)
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                changed = torch.autograd.grad(actual, leaves, upstream)
                for got, want in zip(changed, captured, strict=True):
                    torch.testing.assert_close(got, want)
            finally:
                torch.backends.cuda.matmul.fp32_precision = previous
            return
        if scenario in ("masked", "alias", "autocast_bias", "determinism"):
            if scenario == "masked":
                mask.zero_()
            if scenario == "alias":
                k = v = q
                leaves = (q, bias)
            if scenario == "autocast_bias":
                bias = bias.detach().float().requires_grad_()
                leaves = (q, k, v, bias)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scenario == "autocast_bias"):
                expected = _reference(q, k, v, mask, bias).to(dtype)
                actual = pair_bias_attention(q, k, v, mask, bias)
            upstream = torch.randn_like(actual)
            references = torch.autograd.grad(expected, leaves, upstream)
            previous = torch.are_deterministic_algorithms_enabled()
            warning = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                if scenario == "determinism":
                    torch.use_deterministic_algorithms(True)
                gradients = torch.autograd.grad(actual, leaves, upstream, retain_graph=scenario == "determinism")
                if scenario == "determinism":
                    for first, second in zip(gradients, torch.autograd.grad(actual, leaves, upstream), strict=True):
                        torch.testing.assert_close(first, second, atol=0, rtol=0)
            finally:
                torch.use_deterministic_algorithms(previous, warn_only=warning)
            torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
            for got, want in zip(gradients, references, strict=True):
                torch.testing.assert_close(got, want, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
            return
        if scenario == "forward_ad":
            with forward_ad.dual_level():
                q = forward_ad.make_dual(q, torch.ones_like(q))
                with pytest.raises(OOpsUnsupportedError):
                    pair_bias_attention(q, k, v, mask, bias)
            return
        if scenario == "shape":
            v = v[..., :-1]
        if scenario == "mask_head":
            mask = mask[:, None].expand(-1, 2, -1, -1)
        if scenario == "mixed_nested":
            q = NestedTensor([q[0]], ragged_dims=(1,))
        with pytest.raises(ValueError if scenario in ("shape", "mixed_nested") else OOpsUnsupportedError):
            pair_bias_attention(q, k, v, mask, bias)
