from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from danling.tensors import NestedTensor
from torch.utils.checkpoint import checkpoint

from oops import AtomTokenLayout, OOpsUnsupportedError, atom_to_token_sum

_TOLERANCE = {
    torch.float64: (1e-7, 1e-7),
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}


def _reference(values, mapping, tokens, mask, offset):
    weights = torch.ones(values.shape[:-1], device=values.device, dtype=values.dtype) if mask is None else mask
    assignment = mapping.is_floating_point()
    if assignment:
        weighted = mapping * weights.unsqueeze(-1)
        numerator = weighted.transpose(-1, -2) @ values
    else:
        numerator = torch.zeros(tokens, *values.shape[1:], device=values.device, dtype=values.dtype).index_add(
            0, mapping, values * weights.unsqueeze(-1)
        )
    return numerator


def _case(kind, dtype, device, training, variant=0, masked=True):
    torch.manual_seed(8421 + variant)
    nested = kind in ("nested", "sampled", "empty")
    samples, channels = (3 if kind == "sampled" else 1), 17
    if nested:
        lengths, tokens = (
            ((0, 0), (0, 3)) if kind == "empty" else (((6, 9), (3, 5)) if not variant else ((8, 7), (4, 6)))
        )
    else:
        size, count = (
            (16384, 1024) if kind == "long_shared" else ((32773, 1025) if kind == "long" else (132 + 44 * variant, 11))
        )
        lengths, tokens = (size, size), (count, count)
        samples = 3
    maps = []
    for length, count in zip(lengths, tokens, strict=True):
        index = torch.arange(length, device=device) % max(count, 1)
        if kind in ("regular", "layout", "shared", "long_shared", "broadcast"):
            index = torch.arange(length, device=device) // max(length // count, 1)
        elif kind not in ("interleaved", "empty"):
            index = index[torch.randperm(length, device=device)]
        maps.append(index)
    shape = (
        ((sum(lengths), samples, channels) if kind == "sampled" else (sum(lengths), channels))
        if nested
        else (2, lengths[0], samples, channels)
    )
    values = torch.randn(shape, device=device, dtype=dtype, requires_grad=training)
    weights = (
        (
            0.5
            + torch.rand(shape[:-1], device=device, dtype=torch.float64 if dtype == torch.float64 else torch.float32)
        ).requires_grad_(training)
        if masked
        else None
    )
    if kind == "broadcast" and weights is not None:
        weights = weights[:, :1].expand_as(weights)
    if kind == "dense" and weights is not None:
        with torch.no_grad():
            weights[..., 0] = 0
    refs = (
        values.detach().double().requires_grad_(training),
        None if weights is None else weights.detach().double().requires_grad_(training),
    )
    if nested:
        atom_template = NestedTensor(
            [torch.empty(samples, n, channels) if kind == "sampled" else torch.empty(n, channels) for n in lengths],
            ragged_dims=(1,) if kind == "sampled" else (0,),
        )
        atom = atom_template.packed_like(values)
        mapping = NestedTensor(maps, ragged_dims=(0,))
        reference = NestedTensor(
            [torch.empty(samples, n, 1) if kind == "sampled" else torch.empty(n, 1) for n in tokens],
            ragged_dims=(1,) if kind == "sampled" else (0,),
        )
        mask = (
            None
            if weights is None
            else NestedTensor(
                [torch.empty(samples, n) if kind == "sampled" else torch.empty(n) for n in lengths],
                ragged_dims=(1,) if kind == "sampled" else (0,),
            ).packed_like(weights)
        )
    else:
        atom = values.transpose(1, 2).contiguous()
        mapping = torch.stack(maps)
        if kind in ("shared", "long_shared"):
            mapping = maps[0]
        elif kind in ("layout", "sorted", "sorted_view", "broadcast"):
            mapping = AtomTokenLayout.from_atom_to_token(
                maps[0], num_tokens=tokens[0], sort=kind in ("sorted", "sorted_view")
            )
        elif kind == "assignment":
            mapping = torch.rand(2, samples, lengths[0], tokens[0], device=device, dtype=dtype)
        if kind == "sorted_view":
            sorted_layout = mapping.sorted_layout
            offsets = sorted_layout.token_offsets
            storage = torch.empty(offsets.numel() * 2, device=device, dtype=offsets.dtype)
            storage[::2].copy_(offsets)
            mapping = replace(mapping, sorted_layout=replace(sorted_layout, token_offsets=storage[::2]))
        reference = None
        mask = None if weights is None else weights.transpose(1, 2).contiguous()

    def expected(offset):
        value_parts = refs[0].split(lengths) if nested else refs[0].unbind(0)
        mask_parts = (
            (None,) * len(lengths) if refs[1] is None else (refs[1].split(lengths) if nested else refs[1].unbind(0))
        )
        if kind == "assignment":
            return torch.stack(
                [
                    _reference(
                        v.transpose(0, 1), matrix.double(), count, None if m is None else m.transpose(0, 1), offset
                    )
                    for v, m, matrix, count in zip(value_parts, mask_parts, mapping, tokens, strict=True)
                ]
            )
        indices = (
            mapping.concat.split(lengths)
            if nested
            else (
                (mapping,) * len(lengths)
                if kind in ("shared", "long_shared")
                else (
                    (maps[0],) * len(lengths)
                    if kind in ("sorted", "sorted_view")
                    else (maps if kind in ("layout", "assignment", "broadcast") else mapping.unbind(0))
                )
            )
        )
        result = [
            _reference(v, index, count, m, offset)
            for v, m, index, count in zip(value_parts, mask_parts, indices, tokens, strict=True)
        ]
        return torch.cat(result) if nested else torch.stack(result).transpose(1, 2)

    return (
        (atom, mapping, mask, reference, tokens[0]),
        tuple(x for x in (values, weights) if x is not None),
        tuple(x for x in refs if x is not None),
        expected,
    )


def _saved_view(tensor):
    return tensor.repeat_interleave(2, dim=-1)[..., ::2]


def _negative_operand(value):
    if value is None:
        return None
    if isinstance(value, NestedTensor):
        return value.packed_like(torch._neg_view(-value.concat))
    return torch._neg_view(-value)


def _negative_inputs(inputs):
    atom, mapping, mask, reference, tokens = inputs
    return _negative_operand(atom), mapping, _negative_operand(mask), reference, tokens


def _run(atom, mapping, mask, reference, tokens, offset=0.0):
    if isinstance(atom, NestedTensor):
        return atom_to_token_sum(atom, mapping, token_reference=reference, atom_mask=mask).concat
    return atom_to_token_sum(atom, mapping, num_tokens=tokens, atom_mask=mask)


@pytest.mark.usefixtures("compiler_cache")
class TestAtomToTokenSum:

    @pytest.mark.parametrize(
        ("kind", "dtype", "mode", "masked"),
        [
            ("regular", torch.float32, "graph", False),
            ("shared", torch.float32, "graph", True),
            ("long_shared", torch.float32, "graph", True),
            ("interleaved", torch.float16, "eager", True),
            ("layout", torch.float32, "negative", True),
            ("broadcast", torch.float32, "negative", True),
            ("dense", torch.bfloat16, "negative", True),
            ("sampled", torch.float16, "negative", True),
            ("layout", torch.bfloat16, "eager", True),
            ("assignment", torch.float32, "eager", True),
            ("assignment", torch.float64, "cpu", True),
            ("dense", torch.float32, "compile", True),
            ("nested", torch.bfloat16, "compile", True),
            ("sampled", torch.float16, "eager", True),
            ("empty", torch.float32, "eager", True),
            ("long", torch.float32, "graph", True),
        ],
    )
    def test_forward(self, request, kind, dtype, mode, masked):
        device = "cpu" if mode == "cpu" else request.getfixturevalue("cuda")
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        for variant in (0, 1) if mode == "compile" else (0,):
            inputs, _, _, reference = _case(kind, dtype, device, False, variant, masked)
            if mode == "negative":
                inputs = _negative_inputs(inputs)
            if mode == "graph":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        operation(*inputs, 1e-6)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = operation(*inputs, 1e-6)
                inputs[1].copy_(inputs[1].roll(13, -1))
                graph.replay()
            else:
                output = operation(*inputs, 1e-6)
            expected = reference(1e-6)
            atol, rtol = _TOLERANCE[dtype]
            torch.testing.assert_close(output.double(), expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("kind", "dtype", "mode", "wanted"),
        [
            pytest.param(
                "layout", torch.float32, "eager", "all", marks=pytest.mark.cuda_device(1), id="secondary-cuda"
            ),
            ("regular", torch.float32, "compile", "all"),
            ("layout", torch.float32, "negative", "all"),
            ("broadcast", torch.float32, "negative", "all"),
            ("sorted", torch.float32, "saved", "all"),
            ("dense", torch.bfloat16, "negative", "all"),
            ("sampled", torch.float16, "negative", "all"),
            ("layout", torch.float32, "eager", "all"),
            ("layout", torch.bfloat16, "eager", "all"),
            ("layout", torch.float16, "eager", "all"),
            ("dense", torch.bfloat16, "eager", "mask"),
            ("sorted", torch.bfloat16, "eager", "all"),
            ("layout", torch.float32, "compile", "all"),
            ("sorted_view", torch.float32, "eager", "all"),
            ("sorted_view", torch.bfloat16, "compile", "all"),
            ("assignment", torch.float32, "eager", "all"),
            ("assignment", torch.float64, "cpu", "all"),
            ("nested", torch.float32, "compile", "all"),
            ("sampled", torch.bfloat16, "checkpoint", "all"),
            ("empty", torch.float32, "eager", "all"),
            ("long", torch.float32, "eager", "all"),
            ("dense", torch.float32, "graph_write", "all"),
            ("shared", torch.float32, "graph_write", "all"),
            ("shared", torch.float32, "graph_write_compile", "all"),
            ("long_shared", torch.float32, "graph_write_compile", "all"),
        ],
    )
    def test_backward(self, request, kind, dtype, mode, wanted):
        device = "cpu" if mode == "cpu" else request.getfixturevalue("cuda")
        inputs, leaves, references, reference = _case(kind, dtype, device, True)
        if mode == "negative":
            inputs = _negative_inputs(inputs)
        operation = torch.compile(_run, fullgraph=True) if mode in ("compile", "graph_write_compile") else _run
        selected = (1,) if wanted == "mask" else tuple(range(len(leaves)))
        writer_mode = mode in ("graph_write", "graph_write_compile")
        if writer_mode:
            replacement = inputs[1].clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                inputs[1].copy_(replacement)
            stream.synchronize()
            torch.cuda.current_stream().wait_stream(stream)
            warm = operation(*inputs, 1e-6)
            torch.autograd.grad(warm, tuple(leaves[i] for i in selected), torch.ones_like(warm))
        for iteration in range(2 if writer_mode else 1):
            if writer_mode:
                replacement.copy_(inputs[1].roll(13 + iteration, -1))
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                torch.cuda.current_stream().wait_stream(stream)
            hooks = (
                torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor.detach(), _saved_view)
                if mode == "saved"
                else nullcontext()
            )
            with hooks:
                output = (
                    checkpoint(operation, *inputs, 1e-6, use_reentrant=False)
                    if mode == "checkpoint"
                    else operation(*inputs, 1e-6)
                )
            expected = reference(1e-6)
            atol, rtol = _TOLERANCE[dtype]
            torch.testing.assert_close(output.double(), expected, atol=atol, rtol=rtol)
            upstream = torch.randn_like(output)
            if mode == "negative":
                upstream = torch._neg_view(-upstream)
            actual_grads = torch.autograd.grad(output, tuple(leaves[i] for i in selected), upstream)
            expected_grads = torch.autograd.grad(expected, tuple(references[i] for i in selected), upstream.double())
            for actual, expected in zip(actual_grads, expected_grads, strict=True):
                torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("case", ["missing_tokens", "index_dtype", "mask_shape", "empty_sample_axis"])
    def test_contract(self, case):
        atom = torch.ones(2, 3, 6, 4)
        mapping = torch.arange(6).repeat(2, 1) % 3
        kwargs = {"num_tokens": 3}
        if case == "missing_tokens":
            kwargs = {}
        elif case == "index_dtype":
            mapping = mapping.float()
        elif case == "mask_shape":
            kwargs["atom_mask"] = torch.ones(2, 3, 7)
        else:
            atom = torch.ones(2, 0, 6, 4)
        with pytest.raises((ValueError, TypeError, OOpsUnsupportedError)):
            atom_to_token_sum(atom, mapping, **kwargs)
