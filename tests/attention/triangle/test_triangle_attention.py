from __future__ import annotations

import gc

import pytest
import torch
from danling.tensors import NestedTensor
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.autograd import forward_ad
from torch.utils.checkpoint import checkpoint

from oops import OOpsUnsupportedError, triangle_attention

_TOLERANCE = {
    torch.float32: (5e-3, 5e-3),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (5e-2, 5e-2),
}


def _reference(query, key, value, mask=None, bias=None, scale=None, *, compute_dtype=torch.float32):
    with torch.autocast("cuda", enabled=False):
        scale = query.shape[-1] ** -0.5 if scale is None else scale
        query = query.to(compute_dtype).transpose(-2, -3)
        key = key.to(compute_dtype).transpose(-2, -3)
        value = value.to(compute_dtype).transpose(-2, -3)
        scores = (query @ key.transpose(-1, -2)) * scale
        if bias is not None:
            scores = scores + bias.to(compute_dtype)
        keep = None if mask is None else mask.bool()[..., :, None, None, :]
        if keep is not None:
            scores = scores.masked_fill(~keep, float("-inf"))
            scores = torch.where(keep.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
        probabilities = torch.softmax(scores, dim=-1)
        if keep is not None:
            probabilities = probabilities.masked_fill(~keep, 0.0)
        return (probabilities @ value).transpose(-2, -3)


def _inputs(dtype, device, *, prefix=(2,), shape=(4, 32, 4, 32)):
    torch.manual_seed(0)
    return tuple(torch.randn((*prefix, *shape), device=device, dtype=dtype, requires_grad=True) for _ in range(3))


def _nested_case(
    dtype,
    device,
    *,
    lengths=(3, 5),
    heads=2,
    head_size=16,
    mask=True,
    bias_kind="shared",
    requires_grad=True,
    transposed_pair_axes=False,
):
    qkv_template = NestedTensor(
        [torch.empty(length, length, heads, head_size) for length in lengths],
        ragged_dims=(0, 1),
    )
    qkv_values = tuple(
        torch.randn(qkv_template.concat.shape, device=device, dtype=dtype, requires_grad=requires_grad)
        for _ in range(3)
    )
    query, key, value = (qkv_template.packed_like(values) for values in qkv_values)

    if mask:
        mask_template = NestedTensor(
            [torch.empty(length, length, dtype=torch.bool) for length in lengths],
            ragged_dims=(0, 1),
        )
        mask_values = torch.rand(mask_template.concat.shape, device=device) > 0.2
        offset = 0
        for length in lengths:
            mask_values[offset] = True
            offset += length * length
        nested_mask = mask_template.packed_like(mask_values)
    else:
        nested_mask = None

    bias_values = None
    if bias_kind == "none":
        triangle_bias = None
    elif bias_kind == "shared":
        bias_template = NestedTensor(
            [torch.empty(1, heads, length, length) for length in lengths],
            ragged_dims=(2, 3),
        )
        bias_values = torch.randn(
            bias_template.concat.shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        triangle_bias = bias_template.packed_like(bias_values)
    elif bias_kind in {"full", "mixed"}:
        group_sizes = lengths if bias_kind == "full" else (1, *lengths[1:])
        bias_template = NestedTensor(
            [torch.empty(groups, heads, length, length) for groups, length in zip(group_sizes, lengths)],
            ragged_dims=(0, 2, 3),
        )
        bias_values = torch.randn(
            bias_template.concat.shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        triangle_bias = bias_template.packed_like(bias_values)
    else:
        raise ValueError(f"unknown bias kind: {bias_kind}")
    if transposed_pair_axes:
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        if nested_mask is not None:
            nested_mask = nested_mask.transpose(1, 2)
        if triangle_bias is not None:
            triangle_bias = triangle_bias.transpose(-2, -1)
    return query, key, value, nested_mask, triangle_bias, qkv_values, bias_values


def _independent_nested_attention(query, key, value, mask, triangle_bias, *, scale=None):
    query_parts = torch.unbind(query)
    key_parts = torch.unbind(key)
    value_parts = torch.unbind(value)
    mask_parts = (None,) * len(query_parts) if mask is None else torch.unbind(mask)
    bias_parts = (None,) * len(query_parts) if triangle_bias is None else torch.unbind(triangle_bias)
    return tuple(
        _reference(q, k, v, m, b, scale=scale).to(q.dtype)
        for q, k, v, m, b in zip(query_parts, key_parts, value_parts, mask_parts, bias_parts)
    )


def _flatten_nested_attention(elements, *, transposed_pair_axes=False):
    return torch.cat(
        tuple(
            (element.transpose(0, 1) if transposed_pair_axes else element).reshape(-1, *element.shape[-2:])
            for element in elements
        )
    )


def _case(cuda, dtype, layout, bias_kind, *, active=(), columns=33, head_size=None, lengths=(3, 5)):
    if head_size is None:
        head_size = 16 if layout.startswith("nested") else 24
    if layout.startswith("nested"):
        q, k, v, mask, bias, raw, bv = _nested_case(
            dtype,
            cuda,
            lengths=lengths,
            head_size=head_size,
            bias_kind=bias_kind,
            requires_grad=False,
            transposed_pair_axes=layout == "nested_swapped",
        )
    else:
        prefix = (2, 3) if layout == "prefixed" else (2,)
        q, k, v = _inputs(dtype, cuda, prefix=prefix, shape=(4, columns, 2, head_size))
        if layout == "swapped":
            q, k, v = (t.transpose(-4, -3).contiguous().transpose(-4, -3) for t in (q, k, v))
        mask = torch.rand(*prefix, 4, columns, device=cuda) > 0.2
        mask[..., 0] = True
        bias = (
            None
            if bias_kind == "none"
            else torch.randn(*prefix, 1 if bias_kind == "shared" else 4, 2, columns, columns, device=cuda, dtype=dtype)
        )
        raw, bv = (q, k, v), bias
    tensors = tuple(
        None if tensor is None else tensor.detach().requires_grad_(name in active)
        for name, tensor in zip(("query", "key", "value", "bias"), (*raw, bv), strict=True)
    )
    if layout in ("negative", "nested_negative"):
        tensors = tuple(None if tensor is None else torch._neg_view(tensor) for tensor in tensors)
    if layout.startswith("nested"):
        q, k, v = (wrapper.packed_like(tensor) for wrapper, tensor in zip((q, k, v), tensors[:3], strict=True))
        bias = None if bias is None else bias.packed_like(tensors[3])
    else:
        q, k, v, bias = tensors
    return (q, k, v, mask, bias), tuple(t for t in tensors if t is not None and t.requires_grad)


def _reference_output(inputs, layout):
    q, k, v, mask, bias = inputs
    if isinstance(q, NestedTensor):
        return _flatten_nested_attention(
            _independent_nested_attention(q, k, v, mask, bias), transposed_pair_axes=layout == "nested_swapped"
        )
    return _reference(q, k, v, mask, bias).to(q.dtype)


@pytest.mark.usefixtures("compiler_cache")
class TestTriangleAttention:

    @pytest.mark.parametrize(
        "dtype,layout,bias_kind,mode,precision",
        [
            (torch.float32, "dense", "none", "eager", "ieee"),
            (torch.float32, "negative", "full", "eager", "ieee"),
            (torch.bfloat16, "nested_negative", "mixed", "eager", "ieee"),
            (torch.float16, "swapped", "shared", "eager", "ieee"),
            (torch.bfloat16, "prefixed", "full", "compile", "ieee"),
            (torch.float32, "dense", "full", "eager", "tf32"),
            (torch.bfloat16, "nested", "mixed", "eager", "ieee"),
            (torch.float32, "nested_swapped", "shared", "compile", "ieee"),
            (torch.float32, "nested", "shared", "layouts", "ieee"),
        ],
    )
    def test_forward(self, cuda, dtype, layout, bias_kind, mode, precision):
        previous = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = precision

            def run(*inputs):
                result = triangle_attention(*inputs)
                return result.concat if isinstance(result, NestedTensor) else result

            execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run
            for columns in (33, 41) if mode == "compile" and not layout.startswith("nested") else (33,):
                for lengths in ((3, 5), (5, 3), (3, 5)) if mode == "layouts" else ((3, 5),):
                    inputs, _ = _case(cuda, dtype, layout, bias_kind, columns=columns, lengths=lengths)
                    actual = execute(*inputs)
                    expected = _reference_output(inputs, layout)
                    torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
        finally:
            torch.backends.cuda.matmul.fp32_precision = previous

    @pytest.mark.parametrize(
        "dtype,layout,bias_kind,active,mode,head_size",
        [
            (torch.float32, "dense", "shared", ("query", "key", "value", "bias"), "wide_odd", 129),
            (torch.float32, "dense", "shared", ("query", "key", "value", "bias"), "wide_odd_negative", 129),
            (torch.float32, "dense", "none", ("query", "key", "value"), "wide_odd", 129),
            (torch.float32, "dense", "full", ("query",), "wide_odd", 137),
            (torch.float32, "dense", "shared", ("key",), "wide_odd", 129),
            (torch.float32, "swapped", "full", ("bias",), "wide_odd", 137),
            (torch.float32, "prefixed", "shared", ("value",), "wide_odd", 255),
            (torch.float32, "nested", "shared", ("query", "key", "value", "bias"), "wide_odd", 129),
            (torch.float32, "nested", "shared", ("query", "key", "value", "bias"), "wide_odd_ragged", 129),
            (torch.float32, "nested_swapped", "mixed", ("query", "key", "value", "bias"), "wide_odd", 137),
            (torch.float32, "nested", "full", ("query", "key", "value", "bias"), "wide_odd_deterministic", 129),
            (torch.float32, "dense", "shared", ("query", "key", "value", "bias"), "wide_odd_deterministic", 129),
            (torch.float32, "dense", "shared", ("query", "key", "value", "bias"), "saved", 32),
            (torch.bfloat16, "swapped", "full", ("query", "key", "value", "bias"), "saved", 32),
            (torch.float32, "negative", "shared", ("query", "bias"), "saved_negative", 32),
            (torch.float32, "dense", "full", ("query", "key", "value", "bias"), "eager", 24),
            (torch.float32, "dense", "shared", ("query", "key", "value", "bias"), "sum", 24),
            (torch.float32, "nested", "shared", ("query", "key", "value", "bias"), "sum", 16),
            (torch.float32, "negative", "full", ("query", "key", "value", "bias"), "eager", 24),
            (torch.bfloat16, "nested_negative", "mixed", ("query", "key", "value", "bias"), "eager", 16),
            (torch.float16, "swapped", "shared", ("query",), "eager", 24),
            (torch.bfloat16, "prefixed", "none", ("key",), "compile", 24),
            (torch.float32, "nested", "shared", ("value",), "eager", 16),
            (torch.bfloat16, "nested_swapped", "shared", ("bias",), "eager", 16),
            (torch.bfloat16, "nested", "mixed", ("query", "key", "value", "bias"), "checkpoint", 16),
            (torch.float32, "dense", "shared", ("bias",), "compile", 128),
            (torch.float32, "nested", "shared", ("query", "key", "value", "bias"), "eager", 129),
            (torch.float32, "nested", "shared", ("query", "key", "value", "bias"), "compile", 128),
            (torch.bfloat16, "nested", "shared", ("bias",), "eager", 129),
            (torch.float16, "dense", "shared", ("bias",), "eager", 129),
            pytest.param(
                torch.float16,
                "swapped",
                "shared",
                ("query",),
                "eager",
                24,
                marks=pytest.mark.cuda_device(1),
                id="secondary_cuda",
            ),
        ],
    )
    def test_backward(self, cuda, dtype, layout, bias_kind, active, mode, head_size, saved_strided_hook):
        if mode.startswith("wide_odd"):
            deterministic = mode.endswith("_deterministic")
            torch.manual_seed(117)
            if layout.startswith("nested"):
                inputs, leaves = _case(
                    cuda,
                    torch.float32,
                    layout,
                    bias_kind,
                    active=active,
                    head_size=head_size,
                    lengths=(65, 66) * 16 if mode == "wide_odd_ragged" else (17, 19),
                )
            else:
                prefix = (2, 3) if layout == "prefixed" else (2,)
                q, k, v = _inputs(torch.float32, cuda, prefix=prefix, shape=(17, 17, 2, head_size))
                if layout == "swapped":
                    q, k, v = (x.transpose(-4, -3).contiguous().transpose(-4, -3) for x in (q, k, v))
                mask = torch.rand(*prefix, 17, 17, device=cuda) > 0.2
                mask[..., 0, :] = False
                bias = (
                    None
                    if bias_kind == "none"
                    else torch.randn(*prefix, 1 if bias_kind == "shared" else 17, 2, 17, 17, device=cuda)
                )
                values = tuple(
                    None if x is None else x.detach().requires_grad_(name in active)
                    for name, x in zip(("query", "key", "value", "bias"), (q, k, v, bias), strict=True)
                )
                if mode.endswith("_negative"):
                    values = tuple(None if x is None else torch._neg_view(x) for x in values)
                inputs = (*values[:3], mask, values[3])
                leaves = tuple(x for x in values if x is not None and x.requires_grad)
            q, k, v, mask, bias = inputs
            if isinstance(q, NestedTensor):
                parts = tuple(
                    _reference(*xs, compute_dtype=torch.float64)
                    for xs in zip(*(torch.unbind(x) for x in inputs), strict=True)
                )
                expected = _flatten_nested_attention(parts, transposed_pair_axes=layout == "nested_swapped")
            else:
                expected = _reference(*inputs, compute_dtype=torch.float64)
            previous_determinism = torch.are_deterministic_algorithms_enabled()
            previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            previous_precision = torch.backends.cuda.matmul.fp32_precision
            try:
                torch.use_deterministic_algorithms(deterministic)
                actual = triangle_attention(*inputs)
                actual = actual.concat if isinstance(actual, NestedTensor) else actual
                upstream = torch.randn_like(actual)
                if mode.endswith("_negative"):
                    upstream = torch._neg_view(upstream)
                # The reverse pass must retain the forward precision after a scoped change.
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                got = torch.autograd.grad(actual, leaves, upstream)
                want = torch.autograd.grad(expected, leaves, upstream.double())
            finally:
                torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)
                torch.backends.cuda.matmul.fp32_precision = previous_precision
            torch.testing.assert_close(actual.double(), expected, atol=5e-3, rtol=5e-3)
            for actual_grad, expected_grad, leaf in zip(got, want, leaves, strict=True):
                assert actual_grad.dtype == leaf.dtype and actual_grad.shape == leaf.shape
                torch.testing.assert_close(actual_grad.double(), expected_grad.double(), atol=5e-3, rtol=5e-3)
            return
        inputs, leaves = _case(cuda, dtype, layout, bias_kind, active=active, head_size=head_size)

        def run(*inputs):
            result = triangle_attention(*inputs)
            return result.concat if isinstance(result, NestedTensor) else result

        execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run
        if mode.startswith("saved"):
            with saved_strided_hook():
                actual = execute(*inputs)
        else:
            actual = checkpoint(execute, *inputs, use_reentrant=False) if mode == "checkpoint" else execute(*inputs)
        expected = _reference_output(inputs, layout)
        if mode == "sum":
            torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
            actual.sum().backward()
            gradients = tuple(leaf.grad.detach().clone() for leaf in leaves)
            for leaf in leaves:
                leaf.grad = None
            expected.sum().backward()
            for got, leaf in zip(gradients, leaves, strict=True):
                assert got.dtype == leaf.dtype and got.shape == leaf.shape
                torch.testing.assert_close(got, leaf.grad, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
            return
        upstream = torch.randn_like(actual)
        if mode.endswith("_negative"):
            upstream = torch._neg_view(upstream)
        torch.testing.assert_close(actual, expected, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])
        for got, want, leaf in zip(
            torch.autograd.grad(actual, leaves, upstream),
            torch.autograd.grad(expected, leaves, upstream),
            leaves,
            strict=True,
        ):
            assert got.dtype == leaf.dtype and got.shape == leaf.shape
            torch.testing.assert_close(got, want, atol=_TOLERANCE[dtype][0], rtol=_TOLERANCE[dtype][1])

    @pytest.mark.parametrize(
        "scenario",
        [
            "nested_graph_cold_24_all",
            "nested_graph_warm_24_all",
            "nested_graph_cold_129_all",
            "nested_graph_warm_129_all",
            "nested_graph_cold_129_bias",
            "nested_graph_warm_129_bias",
            "masked",
            "masked_nonfinite",
            "alias",
            "bias_orientation",
            "fake",
            "cpu",
            "shape",
            "dtype",
            "forward_ad",
            "nested_shape",
            "cuda_graph",
        ],
    )
    def test_contract(self, request, scenario):
        if scenario == "masked_nonfinite":
            cuda = request.getfixturevalue("cuda")
            q, k, v = (torch.full((1, 17, 17, 2, 129), float("inf"), device=cuda, requires_grad=True) for _ in range(3))
            mask = torch.zeros(1, 17, 17, device=cuda, dtype=torch.bool)
            bias = torch.full((1, 1, 2, 17, 17), float("nan"), device=cuda, requires_grad=True)
            output = triangle_attention(q, k, v, mask, bias)
            torch.testing.assert_close(output, torch.zeros_like(output))
            for gradient, leaf in zip(torch.autograd.grad(output.sum(), (q, k, v, bias)), (q, k, v, bias)):
                torch.testing.assert_close(gradient, torch.zeros_like(leaf))
            return
        if scenario.startswith("nested_graph_"):
            _, _, warmup, width, demand = scenario.split("_")
            warm, head_size = warmup == "warm", int(width)
            active = ("bias",) if demand == "bias" else ("query", "key", "value", "bias")
            cuda = request.getfixturevalue("cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                other, other_leaves = _case(
                    cuda,
                    torch.float32,
                    "nested",
                    "shared",
                    active=active,
                    lengths=(19, 17),
                    head_size=head_size,
                )
                inputs, leaves = _case(
                    cuda,
                    torch.float32,
                    "nested",
                    "shared",
                    active=active,
                    lengths=(17, 19),
                    head_size=head_size,
                )
                upstream = torch.randn_like(inputs[0].concat)

            def call(values, active):
                output = triangle_attention(*values).concat
                return output, *torch.autograd.grad(output, active, upstream)

            with torch.cuda.stream(stream):
                for _ in range(3):
                    call(other, other_leaves)
                if warm:
                    call(inputs, leaves)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = call(inputs, leaves)
                with torch.no_grad():
                    for lengths in ((11, 7), (23, 13)):
                        other_inputs, _ = _case(cuda, torch.float32, "nested", "none", lengths=lengths, head_size=24)
                        triangle_attention(*other_inputs)
                        del other_inputs
                    gc.collect()
                    leaves[0].add_(0.1)
                graph.replay()
                expected = _reference_output(inputs, "nested")
                expected = expected, *torch.autograd.grad(expected, leaves, upstream)
            stream.synchronize()
            for actual, reference in zip(captured, expected, strict=True):
                torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-3)
            return
        device = "cpu" if scenario == "cpu" else request.getfixturevalue("cuda")
        layout = "nested" if scenario in ("masked", "alias", "bias_orientation") else "dense"
        inputs, leaves = _case(
            device,
            torch.float32,
            layout,
            "shared",
            active=("query", "key", "value", "bias"),
        )
        q, k, v, mask, bias = inputs
        if scenario in ("masked", "alias", "bias_orientation"):
            if scenario == "masked":
                mask = mask.packed_like(torch.zeros_like(mask.concat))
            if scenario == "alias":
                k = v = q
                leaves = (leaves[0], leaves[-1])
            if scenario == "bias_orientation":
                bias = bias.transpose(-2, -1)
            actual = triangle_attention(q, k, v, mask, bias).concat
            expected = _reference_output((q, k, v, mask, bias), layout)
            grad = torch.randn_like(actual)
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
            for got, want in zip(
                torch.autograd.grad(actual, leaves, grad), torch.autograd.grad(expected, leaves, grad), strict=True
            ):
                torch.testing.assert_close(got, want, atol=5e-3, rtol=5e-3)
            return
        if scenario == "cuda_graph":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(stream):
                for _ in range(3):
                    triangle_attention(*inputs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = triangle_attention(*inputs)
                for columns in (17, 41):
                    other, _ = _case(device, torch.float32, "dense", "none", columns=columns)
                    triangle_attention(*other)
                    del other
                gc.collect()
                q.add_(0.1)
                graph.replay()
                expected = _reference_output(inputs, layout)
            stream.synchronize()
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
            return
        if scenario == "fake":
            with FakeTensorMode(allow_non_fake_inputs=True):
                fake_inputs, _ = _case(device, torch.float32, "dense", "shared", columns=37)
                output = triangle_attention(*fake_inputs)
            assert is_fake(output) and output.shape == fake_inputs[0].shape and output.dtype == q.dtype
            actual_inputs, _ = _case(device, torch.float32, "dense", "shared", columns=37)
            actual = triangle_attention(*actual_inputs)
            expected = _reference_output(actual_inputs, "dense")
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
            return
        if scenario == "forward_ad":
            with forward_ad.dual_level():
                q = forward_ad.make_dual(q, torch.ones_like(q))
                with pytest.raises(OOpsUnsupportedError):
                    triangle_attention(q, k, v, mask, bias)
            return
        if scenario == "shape":
            v = v[..., :-1]
        if scenario == "dtype":
            q, k, v = (t.double() for t in (q, k, v))
        if scenario == "nested_shape":
            q = NestedTensor([torch.randn(2, 3, 2, 24, device=device)], ragged_dims=(0, 1))
            k = v = q
            mask = bias = None
        with pytest.raises(ValueError if scenario in ("shape", "nested_shape") else OOpsUnsupportedError):
            triangle_attention(q, k, v, mask, bias)
