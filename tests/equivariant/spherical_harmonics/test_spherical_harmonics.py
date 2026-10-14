from __future__ import annotations

import math

import pytest
import torch
from torch.autograd import forward_ad

from oops import OOpsUnsupportedError, spherical_harmonics


def _reference(
    vectors: torch.Tensor,
    lmax: int,
    *,
    normalize: bool = True,
    normalization: str = "integral",
    eps: float = 1e-12,
) -> torch.Tensor:
    x, y, z = vectors.unbind(-1)
    if normalize:
        inverse_norm = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(eps).reciprocal()
        x, y, z = x * inverse_norm, y * inverse_norm, z * inverse_norm

    scale = [1.0, 1.0, 1.0]
    if normalization == "component":
        scale = [math.sqrt(4.0 * math.pi)] * 3
    elif normalization == "norm":
        scale = [math.sqrt(4.0 * math.pi / (2 * degree + 1)) for degree in range(3)]

    values = [torch.full_like(x, 0.28209479177387814 * scale[0])]
    if lmax >= 1:
        values += [0.4886025119029199 * scale[1] * component for component in (x, y, z)]
    if lmax >= 2:
        values += [
            1.0925484305920792 * scale[2] * x * z,
            1.0925484305920792 * scale[2] * x * y,
            0.31539156525252005 * scale[2] * (2.0 * y * y - x * x - z * z),
            1.0925484305920792 * scale[2] * y * z,
            0.5462742152960396 * scale[2] * (z * z - x * x),
        ]
    return torch.stack(values, -1)


def _run(vectors, lmax, normalize, normalization, eps, return_list):
    output = spherical_harmonics(
        vectors, lmax, normalize=normalize, normalization=normalization, eps=eps, return_list=return_list
    )
    return torch.cat(output, -1) if return_list else output


class TestSphericalHarmonics:

    @pytest.mark.parametrize(
        ("dtype", "lmax", "normalization", "return_list", "mode"),
        [
            (torch.float32, 2, "integral", False, "compile"),
            (torch.float16, 2, "component", False, "eager"),
            (torch.bfloat16, 1, "norm", True, "eager"),
            (torch.float32, 2, "integral", True, "zero"),
            (torch.float32, 0, "integral", False, "eager"),
            (torch.float32, 2, "integral", False, "negative"),
        ],
    )
    def test_forward(self, cuda, dtype, lmax, normalization, return_list, mode, logical_negative_view):
        torch.manual_seed(731)
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        for elements in (31, 37) if mode == "compile" else (17,):
            vectors = (
                torch.zeros(elements, 3, device=cuda, dtype=dtype)
                if mode == "zero"
                else torch.randn(elements, 3, device=cuda, dtype=dtype)
            )
            if mode == "negative":
                vectors = logical_negative_view(vectors)
            actual = operation(vectors, lmax, True, normalization, 1e-12, return_list)
            expected = _reference(vectors.double(), lmax, normalization=normalization)
            tolerance = 8e-3 if dtype == torch.bfloat16 else (2e-3 if dtype == torch.float16 else 1e-6)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize(
        ("lmax", "normalize", "normalization", "eps", "return_list", "mode"),
        [
            (0, True, "integral", 1e-12, False, "eager"),
            (1, True, "integral", 1e-12, False, "eager"),
            (2, True, "component", 0.25, True, "compile"),
            (2, False, "component", 0.25, False, "eager"),
            (2, False, "norm", 1e-4, False, "eager"),
            (2, True, "norm", 0.25, False, "eager"),
            pytest.param(
                0, True, "integral", 1e-12, False, "eager", marks=pytest.mark.cuda_device(1), id="secondary_cuda"
            ),
            (2, True, "component", 0.25, False, "negative"),
        ],
    )
    def test_backward(self, cuda, lmax, normalize, normalization, eps, return_list, mode, logical_negative_view):
        vectors = torch.tensor(
            [[0.01, -0.02, 0.03], [1.0, 2.0, -1.0], [-0.5, 0.25, 0.75]], device=cuda, requires_grad=True
        )
        if mode == "negative":
            vectors = logical_negative_view(vectors)
        reference = vectors.detach().double().requires_grad_()
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        actual = operation(vectors, lmax, normalize, normalization, eps, return_list)
        expected = _reference(reference, lmax, normalize=normalize, normalization=normalization, eps=eps)
        upstream = torch.randn_like(actual)
        gradient = torch.autograd.grad(actual, vectors, upstream)[0]
        expected_gradient = (
            torch.autograd.grad(expected, reference, upstream.double())[0]
            if expected.requires_grad
            else torch.zeros_like(reference)
        )
        torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(gradient.double(), expected_gradient, atol=2e-5, rtol=2e-5)

    @pytest.mark.parametrize("case", ["cpu", "shape", "degree", "normalization", "dtype", "forward_ad"])
    def test_contract(self, case):
        vectors = torch.randn(2, 4 if case == "shape" else 3)
        if case == "dtype":
            vectors = vectors.long()
        if case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(vectors, torch.ones_like(vectors))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    spherical_harmonics(dual, 1)
        else:
            error = (
                OOpsUnsupportedError
                if case == "cpu"
                else (NotImplementedError if case == "degree" else (TypeError if case == "dtype" else ValueError))
            )
            with pytest.raises(error):
                spherical_harmonics(
                    vectors,
                    3 if case == "degree" else 1,
                    normalization="bad" if case == "normalization" else "integral",
                )
