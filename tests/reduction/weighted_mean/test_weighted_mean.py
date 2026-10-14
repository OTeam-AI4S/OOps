from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor

from oops import OOpsUnsupportedError
from oops.reduction.weighted_mean import weighted_mean


def _reference(values, weights):
    x, w = torch.broadcast_tensors(values.double(), weights.double())
    x = torch.where((w == 0) & ~torch.isfinite(x), 0.0, x)
    denominator = w.sum(-1)
    present = denominator > 0
    mean = (x * w).sum(-1) / torch.where(present, denominator, 1.0)
    return torch.where(present, mean, float("nan"))


def _case(device, dtype, case):
    torch.manual_seed(841)
    if case == "half":
        values = torch.full((4097,), 65504.0, device=device, dtype=dtype)
        weights = torch.full_like(values, 0.0019521713256835938)
        values[0], weights[0] = 0.0, 65504.0
    elif case == "sparse":
        values = torch.zeros(2**26, device=device, dtype=dtype)
        weights = torch.zeros_like(values, dtype=torch.bool)
        values[47], weights[47] = 3.0, True
    elif case == "zero":
        values = torch.tensor([[20.0, 80.0, float("nan")], [2.0, 3.0, float("inf")]], device=device, dtype=dtype)
        weights = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    elif case == "broadcast":
        values = torch.tensor([[-65504.0, 65504.0]], device=device, dtype=dtype)
        weights = torch.tensor(1.0, device=device, dtype=dtype)
    elif case == "empty":
        values = torch.empty((2, 0), device=device, dtype=dtype)
        weights = torch.ones((1, 0), device=device, dtype=dtype)
    elif case == "absent":
        values = torch.randn((2, 7), device=device, dtype=dtype)
        weights = torch.zeros((1, 7), device=device, dtype=dtype)
    else:
        values = torch.randn((2, 13, 3), device=device, dtype=dtype).transpose(1, 2)
        weights = torch.rand((1, 1, 13), device=device, dtype=dtype)
    return values, weights


@pytest.mark.usefixtures("compiler_cache")
class TestWeightedMean:
    @pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
    @pytest.mark.parametrize(
        "dtype,case",
        [
            (torch.float16, "regular"),
            (torch.bfloat16, "regular"),
            (torch.float32, "regular"),
            (torch.float64, "regular"),
            (torch.float16, "half"),
            (torch.float16, "sparse"),
            (torch.float32, "zero"),
            (torch.float64, "empty"),
        ],
    )
    def test_forward(self, device, dtype, case):
        values, weights = _case(device, dtype, case)
        expected = _reference(values, weights).to(dtype)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            actual = weighted_mean(values, weights)
        torch.testing.assert_close(actual, expected, equal_nan=True)

    @pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
    @pytest.mark.parametrize(
        "dtype,case",
        [
            (torch.float16, "regular"),
            (torch.bfloat16, "regular"),
            (torch.float32, "regular"),
            (torch.float64, "regular"),
            (torch.float16, "half"),
            (torch.float16, "sparse"),
            (torch.float16, "broadcast"),
            (torch.float32, "zero"),
            (torch.float64, "empty"),
            (torch.float16, "absent"),
        ],
    )
    def test_backward(self, device, dtype, case, saved_strided_hook):
        values, weights = _case(device, dtype, case)
        if case == "sparse":
            weights = weights.to(dtype)
        if case == "regular":
            values, weights = torch._neg_view(-values), torch._neg_view(-weights)
        values.requires_grad_()
        weights.requires_grad_()
        with saved_strided_hook():
            output = weighted_mean(values, weights)
        cotangent = torch.full(
            (*output.shape, 2),
            4096.0 if case == "half" else 65504.0 if case == "broadcast" else 1.0,
            device=device,
            dtype=dtype,
        )[..., 0]
        x, w = values.detach().double().requires_grad_(), weights.detach().double().requires_grad_()
        reference = _reference(x, w)
        expected = torch.autograd.grad(reference, (x, w), cotangent.double())
        actual = torch.autograd.grad(output, (values, weights), cotangent)
        for grad, target, source in zip(actual, expected, (values, weights)):
            torch.testing.assert_close(grad, target.to(source.dtype), equal_nan=True)

    @pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
    @pytest.mark.parametrize(
        "nested,compiled",
        [(False, False), (False, True), (True, False), (True, True), ("samples", True), ("prefix", False)],
    )
    def test_contract(self, device, nested, compiled):
        if nested == "samples":
            shapes, ragged_dims = ((2, 3), (0, 5), (3, 0), (1, 4)), (0, 1)
        elif nested == "prefix":
            shapes, ragged_dims = ((2, 3), (2, 0), (2, 7)), (1,)
        else:
            shapes, ragged_dims = ((3,), (0,), (7,)), (0,)
        template = NestedTensor([torch.empty(shape) for shape in shapes], ragged_dims=ragged_dims).to(device)
        values = torch.arange(template.concat.numel(), device=device, dtype=torch.float64).reshape(
            template.concat.shape
        )
        weights = (values + 1).to(torch.float32)
        values.requires_grad_()
        weights.requires_grad_()

        def run(x, w, selected):
            x, w = x[selected], w[selected]
            if nested:
                x = template.packed_with_static_tail(x)
                w = template.packed_with_static_tail(w)
            return weighted_mean(x, w)

        function = torch.compile(run, fullgraph=True, dynamic=True) if compiled else run
        selections = (10,) if nested else (10, 0, 4)
        for length in selections:
            selected = torch.arange(values.shape[0], device=device) < length
            output = function(values, weights, selected)
            if nested:
                elements = tuple(
                    _reference(x, w)
                    for x, w in zip(
                        torch.unbind(template.packed_like(values.detach())),
                        torch.unbind(template.packed_like(weights.detach())),
                    )
                )
                expected = torch.cat(elements) if nested == "samples" else torch.stack(elements)
            else:
                expected = _reference(values.detach()[selected], weights.detach()[selected])
            actual = output.concat if isinstance(output, NestedTensor) else output
            torch.testing.assert_close(actual, expected, equal_nan=True)
            if nested == "samples":
                assert isinstance(output, NestedTensor)
                assert output.ragged_dims == (0,)
                for element, target in zip(torch.unbind(output), elements):
                    torch.testing.assert_close(element, target, equal_nan=True)
            rx = values.detach().double().requires_grad_()
            rw = weights.detach().double().requires_grad_()
            if nested:
                references = tuple(
                    _reference(x, w)
                    for x, w in zip(
                        torch.unbind(template.packed_like(rx)),
                        torch.unbind(template.packed_like(rw)),
                    )
                )
                reference = torch.cat(references) if nested == "samples" else torch.stack(references)
            else:
                reference = _reference(rx[selected], rw[selected])
            expected_grads = torch.autograd.grad(reference, (rx, rw), torch.ones_like(reference))
            grads = torch.autograd.grad(actual, (values, weights), torch.ones_like(actual))
            for grad, expected_grad in zip(grads, expected_grads):
                torch.testing.assert_close(grad, expected_grad.to(grad.dtype))
            if length == 0:
                for grad in grads:
                    torch.testing.assert_close(grad, torch.zeros_like(grad))
        if not nested and not compiled:
            with pytest.raises(OOpsUnsupportedError, match="floating observations"):
                weighted_mean(values.long(), weights)
            with pytest.raises(ValueError, match="observation axis"):
                weighted_mean(values[0], weights[0])
