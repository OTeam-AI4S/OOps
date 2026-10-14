"""Public outer-product-mean behavior across layouts and precision modes."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from danling.tensors import NestedTensor
from torch._dynamo.decorators import mark_unbacked
from torch.autograd import forward_ad
from torch.utils.checkpoint import checkpoint

from oops import OOpsUnsupportedError, outer_product_mean
from tests.references.outer_product_mean import outer_product_mean_reference

_TOLERANCE = {
    torch.float32: {"atol": 3e-3, "rtol": 3e-3},
    torch.float16: {"atol": 3e-2, "rtol": 3e-2},
    torch.bfloat16: {"atol": 8e-2, "rtol": 5e-2},
}
_GRAD_ERROR = {torch.float32: 8e-3, torch.float16: 5e-2, torch.bfloat16: 5e-2}


@dataclass(frozen=True)
class Case:
    dtype: torch.dtype = torch.float32
    layout: str = "dense"
    outer_dtype: torch.dtype | None = None
    mode: str = "eager"
    mask_kind: str = "bool"
    divide_before: bool = True
    fp32_policy: str = "ieee"
    biases: bool = True
    clamp: bool = False
    wanted: str | tuple[str, ...] = "all"
    input_offset: float = 0.0
    input_scale: float = 1.0
    negative: bool = False


def _inputs(device, dtype, layout, mask_kind, wanted=(), variant=0, *, input_offset=0.0, input_scale=1.0):
    channels, hidden, output_channels = (
        (6, 1, 5)
        if layout in ("tiny", "selected")
        else (
            (128, 32, 256)
            if layout == "strided"
            else (64, 32, 128) if layout in ("wide", "nested_wide") else (16, 8, 12)
        )
    )
    if layout == "nested_empty":
        shapes = (((3, 4), (2, 2)), ((0, 4), (3, 0)), ((0, 2), (3, 3)))[variant]
    elif layout == "nested_wide":
        shapes = ((256 + variant, 64 + variant),)
    elif layout == "nested":
        shapes = ((7 + variant, 5 + variant), (11 + variant, 9 + variant), (2 + variant, 3 + variant))
    else:
        shapes = None
    if shapes:
        shape = (sum(m * n for m, n in shapes), channels)
    elif layout == "dense_empty":
        shape = (*((1, 3, 4), (1, 0, 4), (1, 2, 0), (0, 2, 4), (1, 5, 3))[variant], channels)
    elif layout == "selected":
        shape = (1, 3, 4, channels)
    elif layout == "tiny":
        shape = (1, 3 + 2 * variant, 4 + variant, channels)
    elif layout == "strided":
        shape = (2, 5 + 2 * variant, 4 + variant, channels)
    else:
        shape = (2, 17 + 2 * variant, 35 + 2 * variant, channels)
    values = torch.randn(shape, device=device, dtype=dtype)
    if input_offset or input_scale != 1.0:
        values = values * input_scale + input_offset
    if shapes:
        template = NestedTensor([torch.empty(m, n, channels) for m, n in shapes], ragged_dims=(0, 1))
        msa = template.packed_like(values)
    else:
        msa = values
    mask_values = None
    if mask_kind != "none":
        mask_values = torch.rand(shape[:-1], device=device)
        if mask_kind == "bool":
            mask_values = mask_values > 0.25
        elif mask_kind == "zero":
            mask_values.zero_()
        else:
            mask_values.add_(0.1)
        if layout == "strided":
            mask_values[0, 2] = 0
            mask_values[1] = 0
    if shapes and mask_values is not None:
        mask_template = NestedTensor([torch.empty(m, n) for m, n in shapes], ragged_dims=(0, 1))
        mask = mask_template.packed_like(mask_values)
    else:
        mask = mask_values
    weights = {
        "msa_norm_weight": torch.randn(channels, device=device),
        "msa_norm_bias": torch.randn(channels, device=device),
        "left_weight": torch.randn(hidden, channels, device=device) / channels**0.5,
        "right_weight": torch.randn(hidden, channels, device=device) / channels**0.5,
        "output_weight": torch.randn(output_channels, hidden * hidden, device=device) / hidden,
        "left_bias": torch.randn(hidden, device=device),
        "right_bias": torch.randn(hidden, device=device),
        "output_bias": torch.randn(output_channels, device=device),
    }
    named = {"msa": values, **weights}
    if mask_values is not None and mask_values.is_floating_point():
        named["mask"] = mask_values
    for name, tensor in named.items():
        tensor.requires_grad_(wanted == "all" or name in wanted)
    if layout == "nested_empty":
        mark_unbacked(values, 0, min=0, shape_id="opm_packed_rows")
        if mask_values is not None:
            mark_unbacked(mask_values, 0, min=0, shape_id="opm_packed_rows")
    if shapes:
        msa = template.packed_like(values)
        if mask_values is not None:
            mask = mask_template.packed_like(mask_values)
    elif layout == "strided":
        msa = values.transpose(1, 2)
        if mask_values is not None:
            mask = mask_values.transpose(1, 2)
    return msa, mask, weights, tuple(t for t in named.values() if t.requires_grad)


def _reference(msa, mask, weights, options, high_precision=False):
    if isinstance(msa, NestedTensor):
        masks = (None,) * len(msa) if mask is None else torch.unbind(mask)
        outputs = []
        for value, keep in zip(torch.unbind(msa), masks, strict=True):
            if keep is None:
                keep = value.new_ones(value.shape[:-1])
            result = _reference(value.unsqueeze(0), keep.unsqueeze(0), weights, options, high_precision)
            outputs.append(result.flatten(0, 2))
        return torch.cat(outputs)
    if mask is None:
        mask = msa.new_ones(msa.shape[:-1])
    if high_precision:
        msa, mask = msa.double(), mask.double()
        weights = {name: None if value is None else value.double() for name, value in weights.items()}
    return outer_product_mean_reference(msa, mask, **weights, **options)


def _values(value):
    return value.concat if isinstance(value, NestedTensor) else value


def _run(msa, mask, weights, options):
    return _values(outer_product_mean(msa, mask, **weights, **options))


def _run_selected(msa, mask, keep, weights, options):
    indices = keep.nonzero().flatten()
    return _run(msa[:, indices], mask[:, indices], weights, options)


def _variants(case):
    if case.layout == "dense_empty":
        return range(5)
    if case.layout in ("selected", "nested_empty"):
        return range(3)
    return (0, 1) if case.mode in ("compile", "compile_return") else (0,)


def _selection(msa, variant):
    return torch.tensor(((True, False, True), (False, False, False), (True, True, True))[variant], device=msa.device)


def _precision(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype in (torch.float16, torch.bfloat16) else nullcontext()


@pytest.mark.usefixtures("compiler_cache")
class TestOuterProductMean:

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(Case(), id="dense-fp32"),
            pytest.param(Case(layout="dense_empty", mode="compile", mask_kind="float"), id="dense-empty-compile"),
            pytest.param(Case(layout="selected", mode="compile", mask_kind="float"), id="unbacked-depth-compile"),
            pytest.param(Case(layout="nested_empty", mode="compile", mask_kind="float"), id="nested-empty-compile"),
            pytest.param(Case(layout="tiny"), id="tiny-fp32"),
            pytest.param(Case(layout="tiny", input_offset=100.0, input_scale=0.01), id="tiny-low-variance-fp32"),
            pytest.param(
                Case(layout="tiny", input_offset=10000.0, input_scale=0.03125, mask_kind="float"),
                id="tiny-large-offset-fp32",
            ),
            pytest.param(
                Case(layout="strided", mode="compile", input_offset=100.0, input_scale=0.01),
                id="token-major-low-variance-compile",
            ),
            pytest.param(Case(layout="nested", input_offset=100.0, input_scale=0.01), id="nested-low-variance-fp32"),
            pytest.param(Case(layout="tiny", dtype=torch.bfloat16, mode="compile"), id="tiny-bf16-compile"),
            pytest.param(
                Case(layout="nested_wide", mode="compile", mask_kind="float"), id="single-nested-fp32-compile"
            ),
            pytest.param(Case(layout="strided"), id="token-major-fp32"),
            pytest.param(Case(layout="strided", dtype=torch.float16, mask_kind="none"), id="token-major-fp16"),
            pytest.param(
                Case(layout="strided", dtype=torch.bfloat16, mask_kind="zero", clamp=True),
                id="token-major-bf16-empty",
            ),
            pytest.param(Case(mode="compile", mask_kind="float", fp32_policy="tf32"), id="dense-tf32-compile"),
            pytest.param(
                Case(
                    dtype=torch.float16, outer_dtype=torch.float32, mask_kind="none", divide_before=False, biases=False
                ),
                id="fp16-no-bias",
            ),
            pytest.param(
                Case(
                    layout="wide",
                    dtype=torch.bfloat16,
                    outer_dtype=torch.float32,
                    mask_kind="float",
                    divide_before=False,
                ),
                id="wide-bf16-fp32-outer",
            ),
            pytest.param(Case(outer_dtype=torch.bfloat16, mode="compile"), id="bf16-outer-compile"),
            pytest.param(Case(layout="nested", mask_kind="none"), id="nested-fp32"),
            pytest.param(
                Case(layout="nested", dtype=torch.bfloat16, mode="compile", divide_before=False),
                id="nested-bf16-compile",
            ),
            pytest.param(
                Case(layout="nested", dtype=torch.float16, mask_kind="zero", divide_before=False, clamp=True),
                id="nested-zero-clamped",
            ),
            pytest.param(Case(negative=True), id="logical-negative"),
        ],
    )
    def test_forward(self, cuda: torch.device, case: Case, logical_negative_view):
        torch.manual_seed(903)
        torch.backends.cuda.matmul.fp32_precision = case.fp32_policy
        options: dict[str, Any] = {
            "outer_dtype": case.outer_dtype,
            "divide_before_projection": case.divide_before,
            "denominator_eps": 1e-3,
            "clamp_denominator": case.clamp,
        }
        operation: Any = (
            torch.compile(
                _run_selected if case.layout == "selected" else outer_product_mean, fullgraph=True, dynamic=True
            )
            if case.mode == "compile"
            else outer_product_mean
        )
        for variant in _variants(case):
            msa, mask, weights, _ = _inputs(
                cuda,
                case.dtype,
                case.layout,
                case.mask_kind,
                variant=variant,
                input_offset=case.input_offset,
                input_scale=case.input_scale,
            )
            if case.negative:
                msa = logical_negative_view(msa)
                if mask is not None and mask.dtype is not torch.bool:
                    mask = logical_negative_view(mask)
                weights = {name: logical_negative_view(value) for name, value in weights.items()}
            if not case.biases:
                weights.update(left_bias=None, right_bias=None, output_bias=None)
            with torch.no_grad(), _precision(case.dtype):
                if case.layout == "selected":
                    keep = _selection(msa, variant)
                    actual = operation(msa, mask, keep, weights, options)
                    indices = keep.nonzero().flatten()
                    reference_msa, reference_mask = msa[:, indices], mask[:, indices]
                else:
                    actual = operation(msa, mask, **weights, **options)
                    reference_msa, reference_mask = msa, mask
                expected = _reference(
                    reference_msa,
                    reference_mask,
                    weights,
                    options,
                    high_precision=case.dtype is torch.float32 and case.outer_dtype is None,
                )
            effective_dtype = max(
                (case.dtype, case.outer_dtype or case.dtype), key=lambda value: torch.finfo(value).eps
            )
            compared = _values(actual).to(expected.dtype)
            if case.fp32_policy == "tf32":
                scale = expected.abs().amax().clamp_min(1)
                torch.testing.assert_close(compared / scale, expected / scale, atol=3e-3, rtol=0)
                relative_error = (compared - expected).norm() / expected.norm().clamp_min(1)
                torch.testing.assert_close(relative_error, torch.zeros_like(relative_error), atol=3e-3, rtol=0)
            else:
                torch.testing.assert_close(compared, expected, **_TOLERANCE[effective_dtype])
            if case.layout in ("nested", "nested_wide", "nested_empty"):
                assert isinstance(actual, NestedTensor)
                assert [tuple(x.shape) for x in torch.unbind(actual)] == [
                    (x.shape[1], x.shape[1], weights["output_weight"].shape[0]) for x in torch.unbind(msa)
                ]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(Case(), marks=pytest.mark.cuda_device(1), id="secondary-cuda"),
            pytest.param(Case(), id="dense-fp32"),
            pytest.param(Case(layout="tiny"), id="tiny-fp32"),
            pytest.param(Case(layout="tiny", input_offset=100.0, input_scale=0.01), id="tiny-low-variance-fp32"),
            pytest.param(
                Case(layout="tiny", input_offset=10000.0, input_scale=0.03125, mask_kind="float"),
                id="tiny-large-offset-fp32",
            ),
            pytest.param(
                Case(layout="strided", mode="compile", input_offset=100.0, input_scale=0.01),
                id="token-major-low-variance-compile",
            ),
            pytest.param(Case(layout="nested", input_offset=100.0, input_scale=0.01), id="nested-low-variance-fp32"),
            pytest.param(Case(layout="tiny", mode="compile"), id="tiny-fp32-compile"),
            pytest.param(Case(layout="nested_wide", mode="compile"), id="single-nested-fp32-compile"),
            pytest.param(Case(layout="strided"), id="token-major-fp32"),
            pytest.param(Case(layout="strided", dtype=torch.float16), id="token-major-fp16"),
            pytest.param(Case(layout="strided", dtype=torch.bfloat16), id="token-major-bf16"),
            pytest.param(Case(layout="strided", mode="compile"), id="token-major-fp32-compile"),
            pytest.param(Case(mode="compile", wanted=("output_weight",), fp32_policy="tf32"), id="tf32-output-weight"),
            pytest.param(
                Case(layout="wide", dtype=torch.bfloat16, outer_dtype=torch.float32, divide_before=False),
                id="wide-bf16",
            ),
            pytest.param(
                Case(dtype=torch.float16, outer_dtype=torch.float32, mode="checkpoint", wanted=("msa", "left_weight")),
                id="fp16-checkpoint",
            ),
            pytest.param(
                Case(outer_dtype=torch.bfloat16, mode="compile", wanted=("output_weight",)),
                id="bf16-outer-output-weight",
            ),
            pytest.param(Case(layout="nested", fp32_policy="tf32"), id="nested-tf32"),
            pytest.param(
                Case(
                    layout="nested",
                    dtype=torch.bfloat16,
                    mode="compile",
                    wanted=("mask", "output_weight"),
                    divide_before=False,
                ),
                id="nested-mask-compile",
            ),
            pytest.param(
                Case(
                    layout="nested",
                    dtype=torch.float16,
                    outer_dtype=torch.float32,
                    mode="checkpoint",
                    wanted=("msa", "msa_norm_weight"),
                ),
                id="nested-fp16-checkpoint",
            ),
            pytest.param(
                Case(layout="nested", divide_before=False, clamp=True, wanted=("mask", "output_weight")),
                id="nested-clamped-mask-gradient",
            ),
            pytest.param(Case(layout="dense_empty", mode="compile"), id="dense-empty-compile"),
            pytest.param(Case(layout="selected", mode="compile"), id="unbacked-depth-compile"),
            pytest.param(Case(layout="nested_empty", mode="compile"), id="nested-empty-compile"),
            pytest.param(Case(layout="nested_empty", mode="compile_return"), id="nested-return-unbacked-compile"),
            pytest.param(
                Case(layout="nested_empty", mode="compile", divide_before=False), id="nested-empty-post-divide"
            ),
            pytest.param(
                Case(layout="nested_empty", mode="compile", divide_before=False, clamp=True), id="nested-empty-clamped"
            ),
            pytest.param(Case(layout="nested", negative=True), id="nested-logical-negative"),
        ],
    )
    def test_backward(self, cuda: torch.device, case: Case, logical_negative_view):
        torch.manual_seed(917)
        torch.backends.cuda.matmul.fp32_precision = case.fp32_policy
        options: dict[str, Any] = {
            "outer_dtype": case.outer_dtype,
            "divide_before_projection": case.divide_before,
            "denominator_eps": 1e-3,
            "clamp_denominator": case.clamp,
        }
        operation: Any = (
            torch.compile(
                (
                    outer_product_mean
                    if case.mode == "compile_return"
                    else _run_selected if case.layout == "selected" else _run
                ),
                fullgraph=True,
                dynamic=True,
            )
            if case.mode in ("compile", "compile_return")
            else _run
        )
        for variant in _variants(case):
            msa, mask, weights, operands = _inputs(
                cuda,
                case.dtype,
                case.layout,
                "float",
                case.wanted,
                variant=variant,
                input_offset=case.input_offset,
                input_scale=case.input_scale,
            )
            if case.negative:
                msa = logical_negative_view(msa)
                if mask is not None and mask.dtype is not torch.bool:
                    mask = logical_negative_view(mask)
                weights = {name: logical_negative_view(value) for name, value in weights.items()}
            with _precision(case.dtype):
                if case.layout == "selected":
                    keep = _selection(msa, variant)
                    actual = operation(msa, mask, keep, weights, options)
                    indices = keep.nonzero().flatten()
                    reference_msa, reference_mask = msa[:, indices], mask[:, indices]
                elif case.mode == "compile_return":
                    reference_msa, reference_mask = msa, mask
                    actual = _values(operation(msa, mask, **weights, **options))
                else:
                    reference_msa, reference_mask = msa, mask
                    actual = (
                        checkpoint(operation, msa, mask, weights, options, use_reentrant=False)
                        if case.mode == "checkpoint"
                        else operation(msa, mask, weights, options)
                    )
                expected = _reference(
                    reference_msa,
                    reference_mask,
                    weights,
                    options,
                    high_precision=case.dtype is torch.float32 and case.outer_dtype is None,
                )
            effective_dtype = max(
                (case.dtype, case.outer_dtype or case.dtype), key=lambda value: torch.finfo(value).eps
            )
            compared = actual.to(expected.dtype)
            if case.fp32_policy == "tf32":
                scale = expected.abs().amax().clamp_min(1)
                torch.testing.assert_close(compared / scale, expected / scale, atol=3e-3, rtol=0)
                relative_error = (compared - expected).norm() / expected.norm().clamp_min(1)
                torch.testing.assert_close(relative_error, torch.zeros_like(relative_error), atol=3e-3, rtol=0)
            else:
                torch.testing.assert_close(compared, expected, **_TOLERANCE[effective_dtype])
            upstream = torch.randn((*actual.shape[:-1], actual.shape[-1] * 2), device=cuda, dtype=actual.dtype)[
                ..., ::2
            ]
            if case.mode == "compile_return":
                upstream = actual.new_ones(()).expand_as(actual)
            if case.negative:
                upstream = logical_negative_view(upstream)
            expected_gradients = torch.autograd.grad(expected, operands, upstream, retain_graph=True)
            actual_gradients = torch.autograd.grad(actual, operands, upstream)
            for actual_gradient, expected_gradient, operand in zip(
                actual_gradients, expected_gradients, operands, strict=True
            ):
                assert actual_gradient.shape == operand.shape and actual_gradient.dtype == operand.dtype
                difference = (actual_gradient.double() - expected_gradient.double()).norm()
                relative_error = difference / expected_gradient.double().norm().clamp_min(1)
                torch.testing.assert_close(
                    relative_error, torch.zeros_like(relative_error), atol=_GRAD_ERROR[effective_dtype], rtol=0
                )

    @pytest.mark.parametrize(
        "case, dtype, layout",
        [
            ("cpu", torch.float32, "dense"),
            ("mask_shape", torch.float32, "dense"),
            ("weight_shape", torch.float32, "dense"),
            ("forward_ad", torch.float32, "dense"),
            ("fake", torch.float32, "dense"),
            ("fake", torch.float32, "strided"),
            ("fake", torch.float16, "strided"),
            ("fake", torch.bfloat16, "strided"),
            ("precision_change", torch.float32, "dense"),
        ],
    )
    def test_contract(self, case, dtype, layout):
        if case in ("fake", "precision_change") and not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        device = torch.device("cuda" if case in ("fake", "precision_change") else "cpu")
        msa, mask, weights, _ = _inputs(device, dtype, layout, "bool")
        if case == "fake":
            from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

            with FakeTensorMode(), _precision(dtype):
                msa, mask, weights, operands = _inputs(device, dtype, layout, "bool", wanted=("msa",))
                output = outer_product_mean(msa, mask, **weights)
                gradients = torch.autograd.grad(output, operands, torch.empty_like(output))
            assert isinstance(output, FakeTensor)
            expected_shape = (*msa.shape[:-3], msa.shape[-2], msa.shape[-2], weights["output_weight"].shape[0])
            assert output.shape == expected_shape and output.dtype is dtype
            for gradient, operand in zip(gradients, operands, strict=True):
                assert isinstance(gradient, FakeTensor)
                assert gradient.shape == operand.shape and gradient.dtype is operand.dtype
        elif case == "precision_change":
            weights["output_weight"].requires_grad_()
            output = outer_product_mean(msa, mask, **weights)
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            with pytest.raises(OOpsUnsupportedError, match="remain unchanged"):
                torch.autograd.grad(output.sum(), weights["output_weight"])
        elif case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(msa, torch.ones_like(msa))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    outer_product_mean(dual, mask, **weights)
        else:
            if case == "mask_shape":
                mask = mask[..., :-1]
            elif case == "weight_shape":
                weights["left_weight"] = weights["left_weight"][:, :-1]
            error = OOpsUnsupportedError if case == "cpu" else ValueError
            with pytest.raises(error):
                outer_product_mean(msa, mask, **weights)
