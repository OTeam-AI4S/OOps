from __future__ import annotations

import math

import pytest
import torch
from torch.autograd import forward_ad

from oops import OOpsUnsupportedError, se3_basis


def _edges(*, seed: int, edges: int) -> torch.Tensor:
    # Keep finite differences away from the normalization clamp.
    generator = torch.Generator(device="cuda").manual_seed(seed)
    directions = torch.randn(edges, 3, device="cuda", generator=generator)
    lengths = 0.5 + 1.5 * torch.rand(edges, 1, device="cuda", generator=generator)
    return directions / directions.norm(dim=-1, keepdim=True) * lengths


def _reference_pairwise_basis(relative_positions: torch.Tensor, max_degree: int) -> dict[str, torch.Tensor]:
    unit = relative_positions / torch.linalg.vector_norm(relative_positions, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = unit.unbind(-1)
    harmonics = (
        torch.full_like(x[:, None], 0.28209479177387814),
        0.4886025119029199 * unit,
        torch.stack(
            (
                1.0925484305920792 * x * z,
                1.0925484305920792 * x * y,
                0.31539156525252005 * (2 * y * y - x * x - z * z),
                1.0925484305920792 * y * z,
                0.5462742152960396 * (z * z - x * x),
            ),
            dim=-1,
        ),
    )
    basis = {}
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            blocks = []
            for harmonic_degree in range(abs(d_in - d_out), d_in + d_out + 1):
                blocks.append(
                    torch.einsum(
                        "nf,klf->nlk",
                        harmonics[harmonic_degree],
                        _reference_clebsch_gordon(
                            harmonic_degree, d_in, d_out, relative_positions.device, relative_positions.dtype
                        ),
                    )
                )
            basis[f"{d_in},{d_out}"] = torch.stack(blocks, dim=2)
    return basis


def _reference_packed_basis(
    relative_positions: torch.Tensor,
    *,
    use_pad_trick: bool,
    fully_fused: bool,
) -> dict[str, torch.Tensor]:
    unpadded = _reference_pairwise_basis(relative_positions, 1)
    edges = relative_positions.shape[0]
    pad = int(use_pad_trick)
    pairwise: dict[str, torch.Tensor] = {}
    for key, value in unpadded.items():
        padded = relative_positions.new_zeros((*value.shape[:-1], value.shape[-1] + pad))
        padded[..., : value.shape[-1]] = value
        pairwise[key] = padded

    out0 = relative_positions.new_zeros((edges, 4, 2, 1 + pad))
    out0[:, 0:1, 0:1, :1] = unpadded["0,0"]
    out0[:, 1:4, 1:2, :1] = unpadded["1,0"]

    out1 = relative_positions.new_zeros((edges, 4, 4, 3 + pad))
    out1[:, 0:1, 0:1, :3] = unpadded["0,1"]
    out1[:, 1:4, 1:4, :3] = unpadded["1,1"]

    in0 = relative_positions.new_zeros((edges, 1, 2, 4))
    in0[:, :, 0:1, 0:1] = unpadded["0,0"]
    in0[:, :, 1:2, 1:4] = unpadded["0,1"]

    in1 = relative_positions.new_zeros((edges, 3, 4, 4))
    in1[:, :, 0:1, 0:1] = unpadded["1,0"]
    in1[:, :, 1:4, 1:4] = unpadded["1,1"]

    result = {
        **pairwise,
        "out0_fused": out0,
        "out1_fused": out1,
        "in0_fused": in0,
        "in1_fused": in1,
    }
    if fully_fused:
        fused = relative_positions.new_zeros((edges, 4, 6, 4))
        fused[:, 0:1, 0:1, 0:1] = unpadded["0,0"]
        fused[:, 1:4, 1:2, 0:1] = unpadded["1,0"]
        fused[:, 0:1, 2:3, 1:4] = unpadded["0,1"]
        fused[:, 1:4, 3:6, 1:4] = unpadded["1,1"]
        result["fully_fused"] = fused
    return result


def _reference_clebsch_gordon(
    harmonic_degree: int,
    d_in: int,
    d_out: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    inv_sqrt6 = 1.0 / math.sqrt(6.0)
    inv_sqrt10 = 1.0 / math.sqrt(10.0)
    inv_sqrt30 = 1.0 / math.sqrt(30.0)

    if (harmonic_degree, d_in, d_out) == (0, 0, 0):
        return torch.ones(1, 1, 1, device=device, dtype=dtype)
    if (harmonic_degree, d_in, d_out) == (1, 0, 1):
        return torch.eye(3, device=device, dtype=dtype).reshape(3, 1, 3) * inv_sqrt3
    if (harmonic_degree, d_in, d_out) == (1, 1, 0):
        return torch.eye(3, device=device, dtype=dtype).reshape(1, 3, 3) * inv_sqrt3
    if (harmonic_degree, d_in, d_out) == (0, 1, 1):
        return torch.eye(3, device=device, dtype=dtype).reshape(3, 3, 1) * inv_sqrt3
    if (harmonic_degree, d_in, d_out) == (1, 1, 1):
        return torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, inv_sqrt6], [0.0, -inv_sqrt6, 0.0]],
                [[0.0, 0.0, -inv_sqrt6], [0.0, 0.0, 0.0], [inv_sqrt6, 0.0, 0.0]],
                [[0.0, inv_sqrt6, 0.0], [-inv_sqrt6, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            device=device,
            dtype=dtype,
        ).permute(2, 1, 0)
    if (harmonic_degree, d_in, d_out) == (2, 1, 1):
        return torch.tensor(
            [
                [[0.0, 0.0, inv_sqrt10], [0.0, 0.0, 0.0], [inv_sqrt10, 0.0, 0.0]],
                [[0.0, inv_sqrt10, 0.0], [inv_sqrt10, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[-inv_sqrt30, 0.0, 0.0], [0.0, 2.0 * inv_sqrt30, 0.0], [0.0, 0.0, -inv_sqrt30]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, inv_sqrt10], [0.0, inv_sqrt10, 0.0]],
                [[-inv_sqrt10, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, inv_sqrt10]],
            ],
            device=device,
            dtype=dtype,
        ).permute(2, 1, 0)
    raise AssertionError(f"unexpected reference block {(harmonic_degree, d_in, d_out)}")


def _run(vectors, pad, fused):
    return se3_basis(vectors, 1, use_pad_trick=pad, fully_fused=fused)


class TestSE3Basis:

    @pytest.mark.parametrize(
        ("dtype", "pad", "fused", "mode"),
        [
            (torch.float32, False, False, "eager"),
            (torch.float16, True, False, "eager"),
            (torch.bfloat16, False, True, "eager"),
            (torch.float32, True, True, "compile"),
            (torch.float32, True, True, "negative"),
        ],
    )
    def test_forward(self, cuda, dtype, pad, fused, mode, logical_negative_view):
        del cuda
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        for edges in (31, 33) if mode == "compile" else (17,):
            vectors = _edges(seed=107, edges=edges).to(dtype)
            if mode == "negative":
                vectors = logical_negative_view(vectors)
            actual = operation(vectors, pad, fused)
            expected = _reference_packed_basis(vectors.double(), use_pad_trick=pad, fully_fused=fused)
            assert actual.keys() == expected.keys()
            tolerance = 2e-5 if dtype == torch.float32 else 2e-3
            for key in expected:
                assert actual[key].dtype == dtype
                torch.testing.assert_close(actual[key].double(), expected[key], atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize(
        ("dtype", "pad", "fused", "mode"),
        [
            (torch.float32, False, False, "all"),
            (torch.float16, True, False, "all"),
            (torch.bfloat16, False, False, "subset"),
            (torch.float32, True, True, "compile"),
            (torch.float32, True, True, "padding"),
            (torch.float32, False, False, "zero"),
            pytest.param(torch.float32, False, False, "all", marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
            (torch.float32, True, True, "negative"),
        ],
    )
    def test_backward(self, cuda, dtype, pad, fused, mode, logical_negative_view):
        vectors = _edges(seed=217, edges=17).to(dtype)
        if mode == "zero":
            vectors[0].zero_()
        vectors.requires_grad_()
        if mode == "negative":
            vectors = logical_negative_view(vectors)
        reference = vectors.detach().double().requires_grad_()
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        actual = operation(vectors, pad, fused)
        expected = _reference_packed_basis(reference, use_pad_trick=pad, fully_fused=fused)
        keys = ("0,1",) if mode == "subset" else (("1,1",) if mode == "padding" else tuple(expected))
        weights = {key: torch.randn_like(actual[key]) for key in keys}
        if mode == "padding":
            weights["1,1"].zero_()
            weights["1,1"][..., -1] = 1
        gradient = torch.autograd.grad(sum((actual[key] * weights[key]).sum() for key in keys), vectors)[0]
        expected_gradient = torch.autograd.grad(
            sum((expected[key] * weights[key].double()).sum() for key in keys), reference
        )[0]
        tolerance = 8e-2 if dtype == torch.bfloat16 else (2e-2 if dtype == torch.float16 else 2e-4)
        assert gradient.dtype == dtype
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient.double(), expected_gradient, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("case", ["cpu", "shape", "dtype", "degree", "forward_ad"])
    def test_contract(self, case):
        vectors = torch.randn(2, 4 if case == "shape" else 3)
        if case == "dtype":
            vectors = vectors.long()
        if case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(vectors, torch.ones_like(vectors))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    se3_basis(dual, 1)
        else:
            error = (
                OOpsUnsupportedError
                if case == "cpu"
                else (ValueError if case == "shape" else (TypeError if case == "dtype" else NotImplementedError))
            )
            with pytest.raises(error):
                se3_basis(vectors, 0 if case == "degree" else 1)
