from __future__ import annotations

import decimal

import pytest
import torch
from danling.tensors import NestedTensor

from oops import OOpsUnsupportedError
from oops.geometry.norm import smoothed_vector_norm

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _reference(vectors, upstream, eps):
    with decimal.localcontext() as context:
        context.prec = 96
        epsilon = decimal.Decimal.from_float(eps)
        rows = vectors.detach().reshape(-1, vectors.shape[-1]).cpu().tolist()
        cotangents = upstream.detach().reshape(-1).cpu().tolist()
        outputs, gradients = [], []
        for row, cotangent in zip(rows, cotangents, strict=True):
            values = [decimal.Decimal.from_float(value) for value in row]
            norm = (sum((value * value for value in values), epsilon)).sqrt()
            outputs.append(float(norm))
            gradient = [decimal.Decimal.from_float(cotangent) * value / norm if norm else 0 for value in values]
            gradients.append([float(value) for value in gradient])
    return (
        torch.tensor(outputs, device=vectors.device, dtype=vectors.dtype).reshape(vectors.shape[:-1]),
        torch.tensor(gradients, device=vectors.device, dtype=vectors.dtype).reshape(vectors.shape),
    )


def _nested_run(vectors, eps, keepdim):
    return smoothed_vector_norm(vectors, eps=eps, keepdim=keepdim).concat


def _dynamic_run(vectors, selection):
    return smoothed_vector_norm(vectors[selection], eps=0.125)


@pytest.mark.usefixtures("compiler_cache")
class TestSmoothedVectorNorm:

    @pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("kind", ["strided", "nested", "empty-vector", "empty-rows"])
    def test_forward(self, device, dtype, kind):
        torch.manual_seed(18)
        shape = (3, 0) if kind == "empty-vector" else (0, 3) if kind == "empty-rows" else (6, 3)
        vectors = torch.randn((*shape[:-1], shape[-1] * 2), device=device, dtype=dtype)[..., ::2]
        eps, keepdim = 0.25, kind == "nested"
        input = (
            NestedTensor([vectors[:2], vectors[2:2], vectors[2:]], ragged_dims=(0,)) if kind == "nested" else vectors
        )
        actual = smoothed_vector_norm(input, eps=eps, keepdim=keepdim)
        if kind == "nested":
            assert isinstance(actual, NestedTensor)
            assert actual.ragged_dims == input.ragged_dims
            torch.testing.assert_close(actual.element_sizes(), torch.tensor([[2, 1], [0, 1], [4, 1]]))
        actual = actual.concat if isinstance(actual, NestedTensor) else actual
        expected = (vectors.double().square().sum(-1, keepdim=keepdim) + eps).sqrt().to(dtype)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
    @pytest.mark.parametrize(
        ("dtype", "values", "cotangent", "eps"),
        [
            (torch.float16, [2.0**-24, 0, 0], 2.0**-10, 0),
            (torch.float32, [1e20, 1e20, 0], 2e20, 0),
            (torch.float32, [0.6984478235244751], torch.finfo(torch.float32).max, 0),
            (torch.float32, [2.0**-149, 0], 0.125, 0),
            (torch.float32, [1e30, 1e-30], 1e30, 0),
            (torch.bfloat16, [1e30, 1e-30], 1e30, 0),
            (torch.float64, [1e308, 1e-308], 1e308, 0),
            (torch.float64, [1.7976931348623157e308] * 2, 1e308, 0),
            (torch.float64, [2.0**-1074, 0], 0.125, 0),
            (torch.float64, [1e-308, 0], 1e308, 1e308),
            (torch.float64, [0, 0], 1, 0),
        ],
    )
    def test_backward(self, device, dtype, values, cotangent, eps):
        vectors = torch.tensor(values, dtype=dtype, device=device).expand(2, -1).clone().requires_grad_()
        upstream = torch.tensor(cotangent, dtype=dtype, device=device).expand(2)
        expected, expected_gradient = _reference(vectors, upstream, eps)
        with torch.autocast(device.type, enabled=dtype is not torch.float64):
            actual = smoothed_vector_norm(vectors, eps=eps)
        gradient = torch.autograd.grad(actual, vectors, upstream)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=2 * torch.finfo(dtype).eps)
        torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=2 * torch.finfo(dtype).eps)

    @pytest.mark.parametrize(
        "case",
        [
            "dynamic-cpu",
            "dynamic-cuda",
            "nested-compile",
            "nested-double-compile",
            "nested-prefix-compile",
            "empty-vjp",
            "invalid-eps",
            "rank",
            "dtype",
            "forward-ad",
        ],
    )
    def test_contract(self, case, request):
        if case == "nested-prefix-compile":
            device = request.getfixturevalue("cuda")
            operation = torch.compile(smoothed_vector_norm, fullgraph=True, dynamic=True)
            for keepdim in (False, True):
                for prefix, lengths in [(2, (2, 0, 3)), (0, (1, 0, 2)), (3, (0, 0, 0)), (2, (1, 2, 1))]:
                    template = NestedTensor([torch.empty(prefix, length, 3) for length in lengths], ragged_dims=(1,))
                    values = torch.randn(sum(lengths), prefix, 3, device=device, requires_grad=True)
                    vectors = template.packed_like(values)
                    result = operation(vectors, eps=0.5, keepdim=keepdim)
                    assert isinstance(result, NestedTensor)
                    assert result.ragged_dims == vectors.ragged_dims
                    expected = (values.double().square().sum(-1, keepdim=keepdim) + 0.5).sqrt().float()
                    upstream = torch.randn((), device=device).expand_as(expected)
                    expected_gradient = torch.autograd.grad(expected, values, upstream)[0]
                    gradient = torch.autograd.grad(result.concat, values, upstream)[0]
                    torch.testing.assert_close(result.concat, expected)
                    torch.testing.assert_close(gradient, expected_gradient)
                    for index in range(len(lengths)):
                        expected_element = (
                            (vectors[index].double().square().sum(-1, keepdim=keepdim) + 0.5).sqrt().float()
                        )
                        torch.testing.assert_close(result[index], expected_element)
        elif case.startswith("dynamic"):
            device = request.getfixturevalue("cuda") if case.endswith("cuda") else torch.device("cpu")
            operation = torch.compile(_dynamic_run, fullgraph=True, dynamic=True)
            for count in (2, 0, 4):
                vectors = torch.randn(4, 3, device=device, requires_grad=True)
                mask = torch.arange(4, device=device) < count
                actual = operation(vectors, mask)
                expected = (vectors[mask].double().square().sum(-1) + 0.125).sqrt().to(vectors.dtype)
                expected_gradient = torch.autograd.grad(expected, vectors, torch.ones_like(expected))[0]
                gradient = torch.autograd.grad(actual, vectors, torch.ones_like(actual))[0]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(gradient, expected_gradient)
        elif case in ("nested-compile", "nested-double-compile"):
            device = request.getfixturevalue("cuda")
            values = torch.randn(5, 3, device=device, requires_grad=True)
            if case == "nested-double-compile":
                template = NestedTensor(
                    [torch.empty(2, 1, 3), torch.empty(1, 0, 3), torch.empty(1, 3, 3)], ragged_dims=(0, 1)
                )
                vectors = template.packed_like(values)
                operation = torch.compile(smoothed_vector_norm, fullgraph=True, dynamic=True)
                result = operation(vectors, eps=0.5)
                assert isinstance(result, NestedTensor)
                assert result.ragged_dims == (0, 1)
                torch.testing.assert_close(result.element_sizes(), torch.tensor([[2, 1], [1, 0], [1, 3]]))
                actual = result.concat
            else:
                vectors = NestedTensor([values[:2], values[2:]], ragged_dims=(0,))
                operation = torch.compile(_nested_run, fullgraph=True, dynamic=True)
                actual = operation(vectors, 0.5, False)
            expected = (vectors.concat.double().square().sum(-1) + 0.5).sqrt().float()
            expected_gradient = torch.autograd.grad(expected, values, torch.ones_like(expected))[0]
            gradient = torch.autograd.grad(actual, values, torch.ones_like(actual))[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(gradient, expected_gradient)
        elif case == "empty-vjp":
            values = torch.empty(2, 0, requires_grad=True)
            output = smoothed_vector_norm(values, eps=0, keepdim=True)
            gradient = torch.autograd.grad(output.sum(), values)[0]
            torch.testing.assert_close(output, torch.zeros(2, 1))
            assert gradient.shape == values.shape
        elif case == "forward-ad":
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.ones(3), torch.ones(3))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode"):
                    smoothed_vector_norm(dual)
        else:
            input = (
                torch.tensor(1.0)
                if case == "rank"
                else torch.ones(3, dtype=torch.long if case == "dtype" else torch.float32)
            )
            context = pytest.raises(OOpsUnsupportedError if case == "dtype" else ValueError)
            with context:
                smoothed_vector_norm(input, eps=-1 if case == "invalid-eps" else 0)
