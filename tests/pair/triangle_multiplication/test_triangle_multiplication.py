from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch
from danling.tensors import NestedTensor
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.autograd import forward_ad
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oops import OOpsUnsupportedError, triangle_multiplication

_TOLERANCE = {
    torch.float32: (5e-3, 5e-3),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (5e-2, 5e-2),
}


def _permute(tensor, order):
    offset = tensor.dim() - len(order)
    return tensor.permute(*range(offset), *(offset + axis for axis in order))


def _reference(pair, mask, weights, direction, input_eps=1e-5, output_eps=1e-5, cast=torch.Tensor.float):
    (
        input_norm_weight,
        input_norm_bias,
        left_value_weight,
        left_gate_weight,
        right_value_weight,
        right_gate_weight,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        output_gate_weight,
    ) = weights
    pair = cast(pair)
    normalized = F.layer_norm(pair, (pair.shape[-1],), cast(input_norm_weight), cast(input_norm_bias), input_eps)
    pair_mask = 1.0 if mask is None else mask.to(normalized.dtype).unsqueeze(-1)
    left = (
        pair_mask
        * torch.sigmoid(F.linear(normalized, cast(left_gate_weight)))
        * F.linear(normalized, cast(left_value_weight))
    )
    right = (
        pair_mask
        * torch.sigmoid(F.linear(normalized, cast(right_gate_weight)))
        * F.linear(normalized, cast(right_value_weight))
    )
    if direction == "outgoing":
        left, right = _permute(left, (2, 0, 1)), _permute(right, (2, 1, 0))
    else:
        left, right = _permute(left, (2, 1, 0)), _permute(right, (2, 0, 1))
    update = _permute(left @ right, (1, 2, 0))
    update = F.layer_norm(update, (update.shape[-1],), cast(output_norm_weight), cast(output_norm_bias), output_eps)
    return F.linear(update, cast(output_weight)) * torch.sigmoid(F.linear(normalized, cast(output_gate_weight)))


def _case(dtype, *, channels=32, hidden=32, tokens=16, batchless=False, device="cuda"):
    torch.manual_seed(0)
    prefix = () if batchless else (2,)

    def tensor(*shape, scale=1.0):
        value = torch.randn(*shape, device=device, dtype=dtype)
        value.mul_(scale)
        return value.requires_grad_()

    pair = tensor(*prefix, tokens, tokens, channels)
    weights = (
        tensor(channels),
        tensor(channels),
        tensor(hidden, channels, scale=channels**-0.5),
        tensor(hidden, channels, scale=channels**-0.5),
        tensor(hidden, channels, scale=channels**-0.5),
        tensor(hidden, channels, scale=channels**-0.5),
        tensor(hidden),
        tensor(hidden),
        tensor(channels, hidden, scale=hidden**-0.5),
        tensor(channels, channels, scale=channels**-0.5),
    )
    mask = torch.rand(*prefix, tokens, tokens, device=device) > 0.2
    return pair, mask, weights


def _call(pair, mask, weights, direction, *, biases=(None,) * 6, input_eps=1e-5, output_eps=1e-5):
    return triangle_multiplication(
        pair,
        mask,
        direction=direction,
        input_norm_weight=weights[0],
        input_norm_bias=weights[1],
        left_value_weight=weights[2],
        left_gate_weight=weights[3],
        right_value_weight=weights[4],
        right_gate_weight=weights[5],
        output_norm_weight=weights[6],
        output_norm_bias=weights[7],
        output_weight=weights[8],
        output_gate_weight=weights[9],
        left_value_bias=biases[0],
        left_gate_bias=biases[1],
        right_value_bias=biases[2],
        right_gate_bias=biases[3],
        output_bias=biases[4],
        output_gate_bias=biases[5],
        input_norm_eps=input_eps,
        output_norm_eps=output_eps,
    )


def _negative_operand(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, NestedTensor):
        return tensor.packed_like(torch._neg_view(-tensor.concat))
    return torch._neg_view(-tensor)


_BIAS_NAMES = (
    "left_value_bias",
    "left_gate_bias",
    "right_value_bias",
    "right_gate_bias",
    "output_bias",
    "output_gate_bias",
)

_PROJECTION_INDICES = (2, 3, 4, 5, 8, 9)


def _biases(weights, *, requires_grad=True):
    return tuple(
        (0.2 * torch.randn(weight.shape[0], device=weight.device, dtype=weight.dtype)).requires_grad_(requires_grad)
        for weight in (weights[index] for index in _PROJECTION_INDICES)
    )


def _biased_reference(pair, mask, weights, biases, direction, *, input_eps=1e-5, output_eps=1e-5):
    """Independent FP32 formula; every projection has its own optional additive bias."""
    (
        input_norm_weight,
        input_norm_bias,
        left_value_weight,
        left_gate_weight,
        right_value_weight,
        right_gate_weight,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        output_gate_weight,
    ) = (weight.float() for weight in weights)
    left_value_bias, left_gate_bias, right_value_bias, right_gate_bias, output_bias, output_gate_bias = (
        None if bias is None else bias.float() for bias in biases
    )
    normalized = F.layer_norm(pair.float(), (pair.shape[-1],), input_norm_weight, input_norm_bias, input_eps)
    left = F.linear(normalized, left_value_weight, left_value_bias) * torch.sigmoid(
        F.linear(normalized, left_gate_weight, left_gate_bias)
    )
    right = F.linear(normalized, right_value_weight, right_value_bias) * torch.sigmoid(
        F.linear(normalized, right_gate_weight, right_gate_bias)
    )
    if mask is not None:
        left = left * mask.float().unsqueeze(-1)
        right = right * mask.float().unsqueeze(-1)
    left = left.movedim(-1, -3)
    right = right.movedim(-1, -3)
    combined = left @ right.transpose(-1, -2) if direction == "outgoing" else left.transpose(-1, -2) @ right
    combined = combined.movedim(-3, -1)
    normalized_update = F.layer_norm(combined, (combined.shape[-1],), output_norm_weight, output_norm_bias, output_eps)
    return F.linear(normalized_update, output_weight, output_bias) * torch.sigmoid(
        F.linear(normalized, output_gate_weight, output_gate_bias)
    )


def _cancelling_cotangent(shape, *, dtype, device):
    channels = shape[-1]
    rows = torch.Size(shape[:-1]).numel()
    row = torch.arange(rows, device=device).unsqueeze(1)
    channel = torch.arange(channels, device=device).unsqueeze(0)
    pattern = torch.tensor((128.0, 1 / 16, -128.0), device=device, dtype=dtype)
    # Cancellation triplets retain a small tail in float64.
    values = pattern[(row + channel) % 3]
    values = values * torch.where(channel % 2 == 0, 1, -1)
    return values.reshape(shape)


def _expected_bias_gradient(cotangent, scale):
    # Round local gradients to the storage dtype before float64 reduction.
    local_gradient = (cotangent.float() * scale).to(cotangent.dtype)
    return local_gradient.reshape(-1, cotangent.shape[-1]).double().sum(0).to(cotangent.dtype)


_WEIGHT_NAMES = (
    "input_norm_weight",
    "input_norm_bias",
    "left_value_weight",
    "left_gate_weight",
    "right_value_weight",
    "right_gate_weight",
    "output_norm_weight",
    "output_norm_bias",
    "output_weight",
    "output_gate_weight",
)


@dataclass(frozen=True, slots=True)
class Case:
    dtype: torch.dtype = torch.float32
    channels: int = 17
    hidden: int = 24
    lengths: tuple[int, ...] = (7,)
    layout: str = "dense"
    direction: str = "outgoing"
    active: tuple[str, ...] | str = ()
    mode: str = "eager"
    bias: bool = True
    master: bool = False


def _public_case(cuda, dtype, channels, hidden, lengths, layout, *, master=False, bias=True, active=()):
    source_dtype = torch.float32 if master else dtype
    pair, mask, weights = _case(source_dtype, channels=channels, hidden=hidden, tokens=lengths[0], device=cuda)
    pair = pair.detach().to(dtype)
    mask = mask.float()
    if layout.startswith("nested"):
        prefix = (2,) if layout == "nested_sampled" else ()
        pair_template = NestedTensor(
            [torch.empty(*prefix, n, n, channels) for n in lengths], ragged_dims=(len(prefix), len(prefix) + 1)
        )
        mask_template = NestedTensor(
            [torch.empty(*prefix, n, n) for n in lengths], ragged_dims=(len(prefix), len(prefix) + 1)
        )
        pair = torch.randn(pair_template.concat.shape, device=cuda, dtype=dtype)
        mask = torch.rand(mask_template.concat.shape, device=cuda)
    elif layout == "strided":
        pair = torch.stack((pair, pair), dim=-1)[..., 0]
    elif layout == "sampled":
        pair = pair[:, None].expand(-1, 3, -1, -1, -1).contiguous()
        mask = mask[:, None, :1, :]
    biases = _biases(weights, requires_grad=False) if bias else (None,) * 6
    names = ("pair", *_WEIGHT_NAMES, *_BIAS_NAMES, "pair_mask")
    tensors = (pair, *weights, *biases, mask)
    operands = {
        name: value.detach().requires_grad_(active == "all" or name in active)
        for name, value in zip(names, tensors, strict=True)
        if value is not None
    }
    pair, mask = operands["pair"], operands["pair_mask"]
    weights = tuple(operands[name] for name in _WEIGHT_NAMES)
    biases = tuple(operands.get(name) for name in _BIAS_NAMES)
    if layout.startswith("nested"):
        pair, mask = pair_template.packed_like(pair), mask_template.packed_like(mask)
    leaves = tuple(tensor for tensor in operands.values() if tensor.requires_grad)
    return pair, mask, weights, biases, operands, leaves


def _reference_output(pair, mask, weights, biases, direction):
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else pair.dtype
    weights = tuple(
        weight.to(dtype) if index in _PROJECTION_INDICES else weight for index, weight in enumerate(weights)
    )
    biases = tuple(None if bias is None else bias.to(dtype) for bias in biases)
    with torch.autocast("cuda", enabled=False):
        if isinstance(pair, NestedTensor):
            outputs = tuple(
                _biased_reference(p.to(dtype), m.to(dtype), weights, biases, direction).to(dtype)
                for p, m in zip(torch.unbind(pair), torch.unbind(mask), strict=True)
            )
            return torch.cat(tuple(output.movedim((-3, -2), (0, 1)).flatten(0, 1) for output in outputs)).reshape_as(
                pair.concat
            )
        return _biased_reference(pair.to(dtype), mask.to(dtype), weights, biases, direction).to(dtype)


@pytest.mark.usefixtures("compiler_cache")
class TestTriangleMultiplication:

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(Case(), marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
            Case(channels=8, hidden=12, lengths=(9,), bias=False),
            Case(mode="negative"),
            Case(dtype=torch.bfloat16, lengths=(3, 5), layout="nested", mode="negative"),
            Case(dtype=torch.float16, channels=65, hidden=96, lengths=(17,), layout="strided", direction="incoming"),
            Case(dtype=torch.bfloat16, channels=128, hidden=128, lengths=(19,), layout="sampled", bias=False),
            Case(
                dtype=torch.bfloat16,
                channels=32,
                hidden=48,
                lengths=(3, 5),
                layout="nested",
                direction="incoming",
                mode="compile",
            ),
            Case(channels=8, hidden=8, lengths=(3, 3), layout="nested", bias=False),
            Case(dtype=torch.bfloat16, channels=384, hidden=384, lengths=(193, 197), mode="compile", bias=False),
            Case(dtype=torch.bfloat16, channels=384, hidden=384, lengths=(193,), direction="incoming", master=True),
        ],
    )
    def test_forward(self, cuda, case: Case):
        dtype = case.dtype
        channels = case.channels
        hidden = case.hidden
        lengths = case.lengths
        layout = case.layout
        direction = case.direction
        mode = case.mode
        bias = case.bias
        master = case.master

        def run(pair, mask, weights, biases):
            if mode == "negative":
                pair, mask = _negative_operand(pair), _negative_operand(mask)
                weights = tuple(_negative_operand(weight) for weight in weights)
                biases = tuple(_negative_operand(bias) for bias in biases)
            result = _call(pair, mask, weights, direction, biases=biases)
            return result.concat if isinstance(result, NestedTensor) else result

        execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run
        shapes = (lengths,) if layout.startswith("nested") else tuple((length,) for length in lengths)
        for current in shapes:
            pair, mask, weights, biases, _, _ = _public_case(
                cuda, dtype, channels, hidden, current, layout, master=master, bias=bias
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16) if master else nullcontext():
                actual = execute(pair, mask, weights, biases)
                expected = _reference_output(pair, mask, weights, biases, direction)
            torch.testing.assert_close(
                actual, expected, atol=(_TOLERANCE[actual.dtype])[0], rtol=(_TOLERANCE[actual.dtype])[1]
            )

    @pytest.mark.parametrize(
        "case",
        [
            Case(active=("pair",)),
            Case(active="all", mode="negative"),
            Case(dtype=torch.bfloat16, lengths=(3, 5), layout="nested", active="all", mode="negative"),
            Case(active="all", mode="saved"),
            Case(dtype=torch.bfloat16, direction="incoming", active=("input_norm_weight", "input_norm_bias")),
            Case(active=("left_value_weight",), mode="compile"),
            Case(dtype=torch.bfloat16, direction="incoming", active=("left_gate_weight",)),
            Case(dtype=torch.float16, direction="incoming", active=("right_value_weight",)),
            Case(active=("right_gate_weight",)),
            Case(dtype=torch.bfloat16, direction="incoming", active=("output_norm_bias",)),
            Case(direction="incoming", active=("output_weight",)),
            Case(dtype=torch.float16, active=("output_gate_weight",)),
            Case(direction="incoming", active=("left_gate_weight", "right_value_bias", "output_gate_weight")),
            Case(active=("pair_mask",)),
            Case(
                dtype=torch.bfloat16,
                channels=32,
                hidden=48,
                lengths=(3, 7),
                layout="nested_sampled",
                direction="incoming",
                active="all",
                mode="checkpoint",
                master=True,
            ),
            Case(dtype=torch.float16, channels=129, hidden=192, lengths=(17,), layout="strided", active="all"),
            Case(dtype=torch.bfloat16, channels=384, hidden=384, lengths=(193,), active="all"),
            Case(
                dtype=torch.bfloat16,
                channels=384,
                hidden=384,
                lengths=(193,),
                direction="incoming",
                active="all",
                master=True,
            ),
            Case(channels=257, hidden=513, lengths=(9,), direction="incoming", active="all"),
            Case(dtype=torch.float16, channels=1024, hidden=1024, lengths=(5,), active="all"),
            Case(channels=8, hidden=1024, lengths=(5,), direction="incoming", active="all"),
            Case(dtype=torch.bfloat16, channels=1024, hidden=8, lengths=(5,), active="all", master=True),
        ],
    )
    def test_backward(self, cuda, case: Case, saved_strided_hook):
        dtype = case.dtype
        channels = case.channels
        hidden = case.hidden
        lengths = case.lengths
        layout = case.layout
        direction = case.direction
        active = case.active
        mode = case.mode
        master = case.master
        pair, mask, weights, biases, _, leaves = _public_case(
            cuda, dtype, channels, hidden, lengths, layout, master=master, active=active
        )

        def run(pair, mask, weights, biases):
            if mode == "negative":
                pair, mask = _negative_operand(pair), _negative_operand(mask)
                weights = tuple(_negative_operand(weight) for weight in weights)
                biases = tuple(_negative_operand(bias) for bias in biases)
            output = _call(pair, mask, weights, direction, biases=biases)
            return output.concat if isinstance(output, NestedTensor) else output

        execute = torch.compile(run, fullgraph=True, dynamic=True) if mode == "compile" else run
        with torch.autocast("cuda", dtype=torch.bfloat16) if master else nullcontext():
            hooks = saved_strided_hook() if mode == "saved" else nullcontext()
            with hooks:
                actual = (
                    checkpoint(execute, pair, mask, weights, biases, use_reentrant=False)
                    if mode == "checkpoint"
                    else execute(pair, mask, weights, biases)
                )
            expected = _reference_output(pair, mask, weights, biases, direction)
        torch.testing.assert_close(
            actual, expected, atol=(_TOLERANCE[actual.dtype])[0], rtol=(_TOLERANCE[actual.dtype])[1]
        )
        upstream = torch.randn_like(actual)
        if mode == "negative":
            upstream = torch._neg_view(-upstream)
        for got, want, operand in zip(
            torch.autograd.grad(actual, leaves, upstream),
            torch.autograd.grad(expected, leaves, upstream),
            leaves,
            strict=True,
        ):
            assert got.shape == operand.shape and got.dtype == operand.dtype
            torch.testing.assert_close(
                (got).float(),
                (want).float(),
                atol=(_TOLERANCE[actual.dtype][1]) * (want).float().abs().max().clamp_min(1).item(),
                rtol=0,
            )

    @pytest.mark.parametrize(
        "scenario,dtype,direction,layout",
        [
            ("graph", torch.float32, "outgoing", "nested"),
            ("graph", torch.float16, "incoming", "nested_sampled"),
            ("graph", torch.bfloat16, "outgoing", "nested"),
            ("fake", torch.bfloat16, "incoming", "dense"),
            ("cancellation", torch.bfloat16, "outgoing", "dense"),
            ("direction", torch.float32, "invalid", "dense"),
            ("channels", torch.float32, "outgoing", "dense"),
            ("mask", torch.float32, "outgoing", "dense"),
            ("forward_ad", torch.float32, "outgoing", "dense"),
            ("nested_shape", torch.float32, "outgoing", "nested"),
            ("nested_dynamic", torch.float32, "outgoing", "nested"),
        ],
    )
    def test_contract(self, cuda, scenario, dtype, direction, layout):
        if scenario == "nested_dynamic":
            _, _, weights = _case(dtype, channels=8, hidden=12, tokens=3, device=cuda)
            template = NestedTensor([torch.empty(1, 1, 8), torch.empty(1, 1, 8)], ragged_dims=(0, 1))

            def run_dynamic(values, rows, columns):
                selected = torch.arange(values.shape[0], device=cuda) < (rows * columns).sum().to(cuda)
                pair = template.packed_with_rectangular_lengths(values[selected], rows, columns)
                return _call(pair, None, weights, direction).concat

            execute = torch.compile(run_dynamic, fullgraph=True, dynamic=True)
            for lengths in ((2, 3), (3, 1)):
                values = torch.randn(25, 8, device=cuda, requires_grad=True)
                rows = torch.tensor(lengths)
                actual = execute(values, rows, rows.clone())
                parts = values[: sum(length * length for length in lengths)].split(tuple(n * n for n in lengths))
                expected = torch.cat(
                    tuple(
                        _reference(p.reshape(n, n, 8), None, weights, direction).flatten(0, 1)
                        for p, n in zip(parts, lengths, strict=True)
                    )
                )
                upstream = torch.randn_like(actual)
                torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
                for got, want in zip(
                    torch.autograd.grad(actual, (values, *weights), upstream),
                    torch.autograd.grad(expected, (values, *weights), upstream),
                    strict=True,
                ):
                    torch.testing.assert_close(got, want, atol=5e-3, rtol=5e-3)
            with pytest.raises(ValueError, match="square"):
                execute(values, torch.tensor([2, 3]), torch.tensor([3, 2]))
            return
        if scenario == "fake":
            with torch.no_grad(), FakeTensorMode():
                pair, mask, weights = _case(dtype, channels=384, hidden=384, tokens=193, batchless=True, device=cuda)
                output = _call(pair, mask, weights, direction, biases=_biases(weights, requires_grad=False))
            assert is_fake(output) and output.shape == pair.shape and output.dtype == dtype
            return
        if scenario == "cancellation":
            pair, mask, weights = _case(dtype, channels=128, hidden=128, tokens=257, batchless=True, device=cuda)
            weights = tuple(
                torch.zeros_like(weight) if index in (8, 9) else weight.detach() for index, weight in enumerate(weights)
            )
            bias = torch.full((128,), 0.5, device=cuda, dtype=dtype, requires_grad=True)
            output = _call(pair.detach(), mask, weights, direction, biases=(None, None, None, None, bias, None))
            upstream = _cancelling_cotangent(output.shape, dtype=dtype, device=cuda)
            actual = torch.autograd.grad(output, bias, upstream)[0]
            torch.testing.assert_close(
                actual, _expected_bias_gradient(upstream, 0.5), atol=(_TOLERANCE[dtype])[0], rtol=(_TOLERANCE[dtype])[1]
            )
            return
        if scenario == "graph":
            for lengths in ((5, 9), (9, 5)):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    pair, mask, weights, biases, operands, leaves = _public_case(
                        cuda, dtype, 8, 12, lengths, layout, active="all"
                    )

                def run():
                    p = pair.packed_like(operands["pair"])
                    m = mask.packed_like(operands["pair_mask"])
                    output = _call(p, m, weights, direction, biases=biases).concat
                    return (output, *torch.autograd.grad(output, leaves, torch.ones_like(output)))

                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run()
                for offset in (0.1, -0.2):
                    with torch.no_grad():
                        operands["pair"].add_(offset)
                    graph.replay()
                    torch.cuda.synchronize()
                    with torch.cuda.stream(stream):
                        expected = _reference_output(pair, mask, weights, biases, direction)
                        references = torch.autograd.grad(expected, leaves, torch.ones_like(expected))
                    stream.synchronize()
                    torch.testing.assert_close(
                        captured[0], expected, atol=(_TOLERANCE[dtype])[0], rtol=(_TOLERANCE[dtype])[1]
                    )
                    for got, want in zip(captured[1:], references, strict=True):
                        torch.testing.assert_close(
                            (got).float(),
                            (want).float(),
                            atol=(_TOLERANCE[dtype][1]) * (want).float().abs().max().clamp_min(1).item(),
                            rtol=0,
                        )
            return
        pair, mask, weights = _case(torch.float32, channels=8, hidden=12, tokens=4, device=cuda)
        if scenario == "forward_ad":
            with forward_ad.dual_level():
                pair = forward_ad.make_dual(pair, torch.ones_like(pair))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode"):
                    _call(pair, mask, weights, direction)
            return
        if scenario == "channels":
            pair, mask, weights = _case(torch.float32, channels=1025, hidden=12, tokens=1, device=cuda)
            error = OOpsUnsupportedError
        elif scenario == "mask":
            mask = torch.ones(3, 4, 4, device=cuda, dtype=torch.bool)
            error = ValueError
        elif scenario == "nested_shape":
            pair = NestedTensor([torch.randn(2, 3, 8, device=cuda)], ragged_dims=(0, 1))
            mask = None
            error = ValueError
        else:
            error = ValueError
        with pytest.raises(error):
            _call(pair, mask, weights, direction)
