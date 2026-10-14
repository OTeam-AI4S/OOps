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

"""Categorical projection values and VJPs against a PyTorch one-hot linear layer."""

from __future__ import annotations

import subprocess
import sys
from contextlib import nullcontext

import pytest
import torch
from danling.tensors import NestedTensor
from torch.autograd import forward_ad
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oops import OOpsUnsupportedError, categorical_msa_projection

_TOLERANCE = {torch.float32: (2e-5, 2e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


def _case(device, dtype, layout, wanted=(), variant=0, geometry=None, nonfinite=False):
    channels, vocabulary = (17, 23) if not variant else (65, 31)
    shapes = ((2, 3), (4, 5), (0, 2)) if not variant else ((3, 4), (5, 7), (2, 0))
    if layout == "empty":
        shapes, channels = ((0, 3), (2, 0), (1, 4)), 0
    packed = layout in ("nested", "empty")
    shape = (sum(m * n for m, n in shapes),) if packed else (2, 5 + variant, 7)
    if geometry is not None:
        shape, channels, vocabulary = geometry
        if packed:
            shapes = shape
            shape = (sum(m * n for m, n in shapes),)
    tokens = (
        torch.randint(vocabulary, shape, device=device)
        if geometry is None
        else torch.arange(shape[0], device=device).remainder(vocabulary)
    )
    if nonfinite:
        tokens.zero_()
    values = torch.randn((*shape, 2), device=device, dtype=dtype, requires_grad="continuous" in wanted)
    weight = (torch.randn(channels, vocabulary + 2, device=device, dtype=dtype) * 0.1).requires_grad_(
        "weight" in wanted
    )
    continuous = values
    if layout == "negative":
        continuous = torch._neg_view(-continuous)
        weight = torch._neg_view(-weight)
    if packed:
        token_template = NestedTensor([torch.empty(m, n, dtype=torch.long) for m, n in shapes], ragged_dims=(0, 1))
        template = NestedTensor([torch.empty(m, n, 2) for m, n in shapes], ragged_dims=(0, 1))
        tokens = token_template.packed_like(tokens)
        continuous = template.packed_like(values)
    operands = tuple(t for t in (values, weight) if t.requires_grad)
    return tokens, continuous, weight, operands


def _values(value):
    return value.concat if isinstance(value, NestedTensor) else value


def _reference(tokens, continuous, weight, embedding=False):
    if embedding:
        return F.embedding(_values(tokens), weight[:, :-2].transpose(0, 1)) + F.linear(
            _values(continuous), weight[:, -2:]
        )
    one_hot = F.one_hot(_values(tokens), weight.shape[-1] - 2).to(continuous.dtype)
    return F.linear(torch.cat((one_hot, _values(continuous)), dim=-1), weight)


def _run(tokens, continuous, weight):
    return _values(categorical_msa_projection(tokens, continuous, weight))


def _autocast(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


@pytest.mark.usefixtures("compiler_cache")
class TestCategoricalMSAProjection:

    @pytest.mark.parametrize(
        ("layout", "dtype", "cast", "mode", "precision"),
        [
            ("dense", torch.float32, None, "eager", "ieee"),
            ("dense", torch.float32, None, "compile", "tf32"),
            ("dense", torch.float16, None, "eager", "ieee"),
            ("dense", torch.bfloat16, None, "vmap", "ieee"),
            ("nested", torch.float32, torch.bfloat16, "compile", "ieee"),
            ("nested", torch.bfloat16, None, "eager", "ieee"),
            ("empty", torch.float32, None, "eager", "ieee"),
        ],
    )
    def test_forward(self, cuda, layout, dtype, cast, mode, precision):
        torch.manual_seed(71)
        torch.backends.cuda.matmul.fp32_precision = precision
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        for variant in (0, 1) if mode == "compile" else (0,):
            tokens, continuous, weight, _ = _case(cuda, dtype, layout, variant=variant)
            with torch.no_grad(), _autocast(cast):
                actual = (
                    torch.vmap(_run, in_dims=(0, 0, None))(tokens, continuous, weight)
                    if mode == "vmap"
                    else operation(tokens, continuous, weight)
                )
                expected = _reference(tokens, continuous, weight)
            atol, rtol = (5e-3, 5e-3) if precision == "tf32" else _TOLERANCE[cast or dtype]
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            if layout in ("nested", "empty"):
                with _autocast(cast):
                    output = categorical_msa_projection(tokens, continuous, weight)
                assert isinstance(output, NestedTensor)
                assert [tuple(x.shape) for x in torch.unbind(output)] == [
                    (*x.shape[:-1], weight.shape[0]) for x in torch.unbind(continuous)
                ]

    @pytest.mark.parametrize(
        ("layout", "dtype", "cast", "mode", "wanted", "geometry", "upstream_kind"),
        [
            pytest.param(
                "dense",
                torch.float32,
                None,
                "eager",
                ("continuous", "weight"),
                None,
                "random",
                marks=pytest.mark.cuda_device(1),
                id="secondary-cuda",
            ),
            ("negative", torch.float32, None, "eager", ("continuous", "weight"), None, "random"),
            ("dense", torch.float32, None, "eager", ("continuous", "weight"), None, "random"),
            ("dense", torch.float32, None, "eager", ("continuous", "weight"), ((6,), 4, 1), "random"),
            ("dense", torch.float32, None, "saved-strided", ("continuous", "weight"), None, "random"),
            ("dense", torch.float16, None, "compile", ("continuous", "weight"), None, "random"),
            ("dense", torch.float32, torch.bfloat16, "compile", ("continuous", "weight"), None, "random"),
            ("nested", torch.float32, None, "eager", ("continuous", "weight"), None, "random"),
            ("nested", torch.bfloat16, None, "compile", ("continuous",), None, "random"),
            ("nested", torch.bfloat16, None, "checkpoint", ("weight",), None, "random"),
            ("empty", torch.float32, None, "eager", ("continuous", "weight"), None, "random"),
            ("dense", torch.float32, None, "eager", ("continuous", "weight"), ((561,), 33, 32), "strided"),
            (
                "nested",
                torch.bfloat16,
                None,
                "compile",
                ("continuous", "weight"),
                (((3, 11), (16, 33)), 33, 32),
                "strided",
            ),
            ("dense", torch.float16, None, "compile", ("continuous", "weight"), ((127,), 65, 23), "strided"),
            ("dense", torch.float32, None, "eager", ("continuous", "weight"), ((1057,), 129, 151), "strided"),
            ("dense", torch.float32, None, "eager", ("weight",), ((13,), 17, 3), "nonfinite"),
        ],
    )
    def test_backward(self, cuda, layout, dtype, cast, mode, wanted, geometry, upstream_kind, saved_strided_hook):
        torch.manual_seed(72)
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        for variant in (0, 1) if mode == "compile" and geometry is None else (0,):
            tokens, continuous, weight, operands = _case(
                cuda, dtype, layout, wanted, variant=variant, geometry=geometry, nonfinite=upstream_kind == "nonfinite"
            )
            with _autocast(cast):
                if mode == "saved-strided":
                    with saved_strided_hook():
                        actual = operation(tokens, continuous, weight)
                else:
                    actual = (
                        checkpoint(operation, tokens, continuous, weight, use_reentrant=False)
                        if mode == "checkpoint"
                        else operation(tokens, continuous, weight)
                    )
                expected = _reference(tokens, continuous, weight, embedding=upstream_kind == "nonfinite")
            if upstream_kind == "strided":
                upstream = torch.randn(actual.shape[1], actual.shape[0], device=cuda, dtype=dtype).transpose(0, 1)
            else:
                upstream = torch.randn_like(actual)
                if upstream_kind == "nonfinite":
                    upstream[0, 0] = float("nan")
                    upstream[0, 1] = float("inf")
            with _autocast(cast):
                gradients = torch.autograd.grad(actual, operands, upstream)
                expected_gradients = torch.autograd.grad(expected, operands, upstream)
            if upstream_kind == "strided":
                x, w, g = _values(continuous).detach().double(), weight.detach().double(), upstream.double()
                ids = _values(tokens)
                vocabulary = w.shape[1] - 2
                expected = (
                    w[:, :vocabulary].transpose(0, 1)[ids]
                    + x[:, :1] * w[:, vocabulary]
                    + x[:, 1:] * w[:, vocabulary + 1]
                )
                grad_continuous = g @ w[:, vocabulary:]
                grad_weight = torch.zeros_like(w)
                grad_weight[:, :vocabulary].transpose(0, 1).index_add_(0, ids, g)
                grad_weight[:, vocabulary:] = g.transpose(0, 1) @ x
                expected_gradients = (grad_continuous, grad_weight)
            atol, rtol = _TOLERANCE[cast or dtype]
            torch.testing.assert_close(
                actual.double() if upstream_kind == "strided" else actual, expected, atol=atol, rtol=rtol
            )
            for gradient, reference, operand in zip(gradients, expected_gradients, operands, strict=True):
                assert gradient.shape == operand.shape and gradient.dtype is operand.dtype
                if upstream_kind == "nonfinite":
                    torch.testing.assert_close(gradient, reference, atol=atol, rtol=rtol, equal_nan=True)
                    continue
                converted = gradient.double() if upstream_kind == "strided" else gradient.float()
                reference = reference.double() if upstream_kind == "strided" else reference.float()
                relative = (converted - reference).norm() / reference.norm().clamp_min(1)
                torch.testing.assert_close(
                    relative,
                    torch.zeros_like(relative),
                    atol=5e-2 if cast is not None else (2e-5 if dtype is torch.float32 else 3e-2),
                    rtol=0,
                )

    @pytest.mark.parametrize(
        "case", ["cpu", "index-negative", "index-upper", "forward_ad", "topology", "mixed-layout", "precision-change"]
    )
    def test_contract(self, request, case):
        device = "cpu" if case in ("cpu", "topology", "mixed-layout") else request.getfixturevalue("cuda")
        if case.startswith("index"):
            index = -1 if case == "index-negative" else 7
            script = (
                "import torch\n"
                "from oops import categorical_msa_projection\n"
                f'tokens = torch.tensor([[0, {index}]], device="cuda")\n'
                'continuous = torch.randn(1, 2, 2, device="cuda")\n'
                'weight = torch.randn(4, 9, device="cuda")\n'
                "categorical_msa_projection(tokens, continuous, weight)\n"
                "torch.cuda.synchronize()\n"
            )
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
            assert result.returncode != 0
            assert "categorical_msa_projection token index out of range" in result.stderr
            return
        if case in ("topology", "mixed-layout"):
            tokens = NestedTensor([torch.zeros(2, 3, dtype=torch.long)], ragged_dims=(0, 1))
            continuous = (
                NestedTensor([torch.empty(3, 2, 2)], ragged_dims=(0, 1)) if case == "topology" else torch.empty(2, 3, 2)
            )
            with pytest.raises(ValueError):
                categorical_msa_projection(tokens, continuous, torch.empty(4, 9))
            return
        tokens, continuous, weight, _ = _case(device, torch.float32, "dense", ("weight",))
        if case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(continuous, torch.ones_like(continuous))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    categorical_msa_projection(tokens, dual, weight)
        elif case == "precision-change":
            output = categorical_msa_projection(tokens, continuous, weight)
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            with pytest.raises(OOpsUnsupportedError, match="remain unchanged"):
                torch.autograd.grad(output.sum(), weight)
        else:
            with pytest.raises(OOpsUnsupportedError):
                categorical_msa_projection(tokens, continuous, weight)
