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

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.normalization.layer_norm.triton import resolve_layer_norm_backward_plan
from oops.runtime import (
    FLOAT32_DOT_TF32X3,
    NativeMatmulDomain,
    captured_native_matmul_precision_state,
    cuda_device_context,
    float32_accumulating_matmul,
    functorch_transforms_active,
    is_compiling,
    native_matmul_domain,
    needs_grad,
    require_native_matmul_precision_state,
    triton_dot_precision,
    unsupported,
)

from .triton.generic import (
    supports_msa_dual_projection,
    triton_msa_dual_projection,
    triton_msa_projection_normalization_backward,
    triton_outer_product_count_backward,
    triton_project_pair_outer_mean,
)


def _plain(*operands):
    return (
        not is_compiling()
        and not is_in_torch_dispatch_mode()
        and not functorch_transforms_active()
        and all(value is None or type(value) in (Tensor, torch.nn.Parameter) for value in operands)
    )


@torch.compiler.assume_constant_result
def _native_matmul_state(*dtypes):
    domains = int(NativeMatmulDomain.FP32)
    for dtype in dtypes:
        domains |= int(native_matmul_domain(dtype))
    return captured_native_matmul_precision_state(domains)


def _execute_projection(
    msa,
    norm_weight,
    norm_bias,
    left_weight,
    right_weight,
    left_bias,
    right_bias,
    eps,
    input_precision,
    state,
    save_stats,
):
    require_native_matmul_precision_state("outer_product_mean", state)
    return triton_msa_dual_projection(
        msa,
        norm_weight,
        norm_bias,
        left_weight,
        right_weight,
        left_bias,
        right_bias,
        eps,
        input_precision,
        save_stats=save_stats,
    )


def _execute_projection_backward(
    grad_left,
    grad_right,
    msa,
    norm_weight,
    norm_bias,
    left_weight,
    right_weight,
    left_bias,
    right_bias,
    normalized,
    mean,
    rstd,
    eps,
    input_precision,
    state,
    need_msa,
    need_norm_weight,
    need_norm_bias,
    need_left_weight,
    need_right_weight,
    need_left_bias,
    need_right_bias,
):
    with cuda_device_context(msa.device):
        require_native_matmul_precision_state("outer_product_mean", state)
        parameters = (msa, norm_weight, norm_bias, left_weight, right_weight, left_bias, right_bias)
        needed = (
            need_msa,
            need_norm_weight,
            need_norm_bias,
            need_left_weight,
            need_right_weight,
            need_left_bias,
            need_right_bias,
        )
        rows = msa.numel() // msa.shape[-1]
        if rows == 0:
            return tuple(
                torch.zeros_like(value) if need and value is not None else None
                for value, need in zip(parameters, needed)
            )
        hidden, channels = left_weight.shape
        normalized = normalized.resolve_neg().contiguous().reshape(rows, channels)
        left = (
            torch.zeros((rows, hidden), device=msa.device, dtype=left_weight.dtype)
            if grad_left is None
            else grad_left.reshape(rows, hidden)
        )
        right = torch.zeros_like(left) if grad_right is None else grad_right.reshape(rows, hidden)
        grad_left_weight = float32_accumulating_matmul(left.transpose(0, 1), normalized) if need_left_weight else None
        grad_right_weight = (
            float32_accumulating_matmul(right.transpose(0, 1), normalized) if need_right_weight else None
        )
        grad_left_bias = left.sum(0, dtype=torch.float32) if need_left_bias else None
        grad_right_bias = right.sum(0, dtype=torch.float32) if need_right_bias else None
        grad_msa = grad_norm_weight = grad_norm_bias = None
        if need_msa or need_norm_weight or need_norm_bias:
            grad_affine = float32_accumulating_matmul(left, left_weight) + float32_accumulating_matmul(
                right, right_weight
            )
            msa_rows = msa.resolve_neg().contiguous().reshape(rows, channels)
            destination = torch.empty_like(msa_rows) if need_msa else None
            plan = resolve_layer_norm_backward_plan(msa_rows)
            grad_norm_weight, grad_norm_bias = triton_msa_projection_normalization_backward(
                grad_affine,
                msa_rows,
                norm_weight,
                eps,
                mean.resolve_neg().contiguous(),
                rstd.resolve_neg().contiguous(),
                destination,
                need_norm_weight,
                need_norm_bias,
                plan,
            )
            grad_msa = None if destination is None else destination.reshape(msa.shape)
        gradients = (
            grad_msa,
            grad_norm_weight,
            grad_norm_bias,
            grad_left_weight,
            grad_right_weight,
            grad_left_bias,
            grad_right_bias,
        )
        return tuple(
            None if gradient is None else gradient.contiguous().to(parameter.dtype)
            for gradient, parameter in zip(gradients, parameters)
        )


def _execute_contraction(
    left, right, count, weight, bias, input_precision, state, divide_before, eps, clamp, save_projection
):
    with cuda_device_context(left.device), torch.autocast("cuda", enabled=False):
        require_native_matmul_precision_state("outer_product_mean", state)
        outer = torch.einsum("bsih,bsjk->bihjk", left, right).contiguous()
        output = torch.empty(
            (left.shape[0], left.shape[2], right.shape[2], weight.shape[0]), device=left.device, dtype=weight.dtype
        )
        numerator = torch.empty_like(output) if save_projection else None
        if output.numel():
            triton_project_pair_outer_mean(
                outer,
                count.squeeze(-1),
                weight,
                bias,
                output,
                numerator=numerator,
                input_precision=input_precision,
                divide_before_projection=divide_before,
                denominator_eps=eps,
                clamp_denominator=clamp,
                target=None,
            )
        return output, outer, numerator


def _execute_contraction_backward(
    gradient,
    left,
    right,
    count,
    weight,
    bias,
    outer,
    numerator,
    input_precision,
    state,
    divide_before,
    eps,
    clamp,
    need_left,
    need_right,
    need_count,
    need_weight,
    need_bias,
):
    with cuda_device_context(left.device), torch.autocast("cuda", enabled=False):
        require_native_matmul_precision_state("outer_product_mean", state)
        gradient = gradient.resolve_neg().contiguous()
        outer = outer.resolve_neg().contiguous()
        count = count.resolve_neg().contiguous()
        denominator = count.float().clamp_min(1) if divide_before else count.float() + eps
        if not divide_before and clamp:
            denominator = denominator.clamp_min(1)
        scaled = gradient.float() / denominator
        grad_projected = scaled.to(weight.dtype)
        grad_weight = None
        if need_weight:
            pair_outer = outer.permute(0, 1, 3, 2, 4).contiguous().flatten(-2).to(weight.dtype)
            grad_weight = float32_accumulating_matmul(
                grad_projected.reshape(-1, weight.shape[0]).transpose(0, 1), pair_outer.reshape(-1, weight.shape[1])
            )
        grad_bias = (gradient.float() if divide_before else scaled).sum((0, 1, 2)) if need_bias else None
        grad_count = None
        if need_count:
            grad_count = (
                triton_outer_product_count_backward(gradient, numerator, count, bias, divide_before, eps, clamp)
                if count.numel()
                else torch.zeros_like(count)
            )
        grad_left = grad_right = None
        if need_left or need_right:
            grad_outer = (
                torch.matmul(grad_projected, weight)
                .to(left.dtype)
                .reshape(left.shape[0], left.shape[2], right.shape[2], left.shape[-1], right.shape[-1])
            )
            if need_left:
                grad_left = torch.einsum("bijhk,bsjk->bsih", grad_outer, right)
            if need_right:
                grad_right = torch.einsum("bijhk,bsih->bsjk", grad_outer, left)
        gradients = (grad_left, grad_right, grad_count, grad_weight, grad_bias)
        parameters = (left, right, count, weight, bias)
        return (
            *tuple(
                None if gradient is None else gradient.contiguous().to(parameter.dtype)
                for gradient, parameter in zip(gradients[:3], parameters[:3])
            ),
            grad_weight,
            grad_bias,
        )


class _MSAPairCount(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(mask):
        with torch.autocast("cuda", enabled=False):
            return torch.einsum("bsi,bsj->bij", mask, mask).unsqueeze(-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, gradient):
        (mask,) = ctx.saved_tensors
        with torch.autocast("cuda", enabled=False):
            gradient = gradient.squeeze(-1)
            return (torch.einsum("bsj,bij->bsi", mask, gradient) + torch.einsum("bsi,bij->bsj", mask, gradient),)


@torch.library.custom_op(
    "ooperators::_msa_dual_projection_forward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor msa, Tensor norm_weight, Tensor norm_bias, Tensor left_weight, Tensor "
        "right_weight, Tensor? left_bias, Tensor? right_bias, float eps, str input_precision, "
        "int state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)
def _projection_forward(
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    eps: float,
    input_precision: str,
    state: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _execute_projection(
        msa, norm_weight, norm_bias, left_weight, right_weight, left_bias, right_bias, eps, input_precision, state, True
    )


@_projection_forward.register_fake
def _projection_forward_fake(
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    eps: float,
    input_precision: str,
    state: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    shape = (*msa.shape[:-1], left_weight.shape[0])
    left = torch.empty(shape, device=msa.device, dtype=left_weight.dtype)
    normalized = torch.empty(msa.shape, device=msa.device, dtype=left_weight.dtype)
    mean = torch.empty(msa.numel() // msa.shape[-1], device=msa.device, dtype=torch.float32)
    return left, torch.empty_like(left), normalized, mean, torch.empty_like(mean)


@torch.library.custom_op(
    "ooperators::_msa_dual_projection_inference",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor msa, Tensor norm_weight, Tensor norm_bias, Tensor left_weight, Tensor "
        "right_weight, Tensor? left_bias, Tensor? right_bias, float eps, str input_precision, "
        "int state) -> (Tensor, Tensor)"
    ),
)
def _projection_inference(
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    eps: float,
    input_precision: str,
    state: int,
) -> tuple[Tensor, Tensor]:
    return _execute_projection(
        msa,
        norm_weight,
        norm_bias,
        left_weight,
        right_weight,
        left_bias,
        right_bias,
        eps,
        input_precision,
        state,
        False,
    )[:2]


@_projection_inference.register_fake
def _projection_inference_fake(
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    eps: float,
    input_precision: str,
    state: int,
) -> tuple[Tensor, Tensor]:
    return _projection_forward_fake(
        msa, norm_weight, norm_bias, left_weight, right_weight, left_bias, right_bias, eps, input_precision, state
    )[:2]


@torch.library.custom_op(
    "ooperators::_msa_dual_projection_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor? grad_left, Tensor? grad_right, Tensor msa, Tensor norm_weight, Tensor "
        "norm_bias, Tensor left_weight, Tensor right_weight, Tensor? left_bias, Tensor? "
        "right_bias, Tensor normalized, Tensor mean, Tensor rstd, float eps, str "
        "input_precision, int state, bool need_msa, bool need_norm_weight, bool "
        "need_norm_bias, bool need_left_weight, bool need_right_weight, bool need_left_bias, "
        "bool need_right_bias) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, "
        "Tensor?)"
    ),
)
def _projection_backward(
    grad_left: Tensor | None,
    grad_right: Tensor | None,
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    normalized: Tensor,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    input_precision: str,
    state: int,
    need_msa: bool,
    need_norm_weight: bool,
    need_norm_bias: bool,
    need_left_weight: bool,
    need_right_weight: bool,
    need_left_bias: bool,
    need_right_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return _execute_projection_backward(
        grad_left,
        grad_right,
        msa,
        norm_weight,
        norm_bias,
        left_weight,
        right_weight,
        left_bias,
        right_bias,
        normalized,
        mean,
        rstd,
        eps,
        input_precision,
        state,
        need_msa,
        need_norm_weight,
        need_norm_bias,
        need_left_weight,
        need_right_weight,
        need_left_bias,
        need_right_bias,
    )


@_projection_backward.register_fake
def _projection_backward_fake(
    grad_left: Tensor | None,
    grad_right: Tensor | None,
    msa: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    normalized: Tensor,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    input_precision: str,
    state: int,
    need_msa: bool,
    need_norm_weight: bool,
    need_norm_bias: bool,
    need_left_weight: bool,
    need_right_weight: bool,
    need_left_bias: bool,
    need_right_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return (
        torch.empty(msa.shape, device=msa.device, dtype=msa.dtype) if need_msa else None,
        (
            torch.empty(norm_weight.shape, device=norm_weight.device, dtype=norm_weight.dtype)
            if need_norm_weight
            else None
        ),
        torch.empty(norm_bias.shape, device=norm_bias.device, dtype=norm_bias.dtype) if need_norm_bias else None,
        (
            torch.empty(left_weight.shape, device=left_weight.device, dtype=left_weight.dtype)
            if need_left_weight
            else None
        ),
        (
            torch.empty(right_weight.shape, device=right_weight.device, dtype=right_weight.dtype)
            if need_right_weight
            else None
        ),
        (
            torch.empty(left_bias.shape, device=left_bias.device, dtype=left_bias.dtype)
            if need_left_bias and left_bias is not None
            else None
        ),
        (
            torch.empty(right_bias.shape, device=right_bias.device, dtype=right_bias.dtype)
            if need_right_bias and right_bias is not None
            else None
        ),
    )


def _projection_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:7], *output[2:])
    ctx.constants = inputs[7:]
    ctx.demand = ctx.needs_input_grad[:7]
    ctx.mark_non_differentiable(*output[2:])
    ctx.set_materialize_grads(False)


def _projection_autograd_backward(ctx, grad_left, grad_right, *auxiliary):
    arguments = (grad_left, grad_right, *ctx.saved_tensors, *ctx.constants, *ctx.demand)
    gradients = (
        _execute_projection_backward(*arguments)
        if getattr(ctx, "direct", False) and not torch.is_grad_enabled()
        else _projection_backward(*arguments)
    )
    return (*gradients, *((None,) * 3))


torch.library.register_autograd(_projection_forward, _projection_autograd_backward, setup_context=_projection_setup)


class _SavedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = _execute_projection(*inputs, True)
        ctx.save_for_backward(*inputs[:7], *output[2:])
        ctx.constants = inputs[7:]
        ctx.demand = ctx.needs_input_grad[:7]
        ctx.set_materialize_grads(False)
        ctx.direct = True
        return output[:2]

    @staticmethod
    def backward(ctx, grad_left, grad_right):
        return _projection_autograd_backward(ctx, grad_left, grad_right)


def _run_projection(
    msa, norm_weight, norm_bias, left_weight, right_weight, left_bias, right_bias, eps, input_precision, state
):
    arguments = (
        msa,
        norm_weight,
        norm_bias,
        left_weight,
        right_weight,
        left_bias,
        right_bias,
        eps,
        input_precision,
        state,
    )
    if needs_grad(*arguments[:7]):
        if _plain(*arguments[:7]):
            return _SavedProjection.apply(*arguments)
        return _projection_forward(*arguments)[:2]
    if _plain(*arguments[:7]):
        return _execute_projection(*arguments, False)[:2]
    return _projection_inference(*arguments)


@torch.library.custom_op(
    "ooperators::_msa_contraction_forward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor left, Tensor right, Tensor count, Tensor weight, Tensor? bias, str "
        "input_precision, int state, bool divide_before, float eps, bool clamp) -> (Tensor, "
        "Tensor, Tensor)"
    ),
)
def _contraction_forward(
    left: Tensor,
    right: Tensor,
    count: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return _execute_contraction(
        left, right, count, weight, bias, input_precision, state, divide_before, eps, clamp, True
    )


@_contraction_forward.register_fake
def _contraction_forward_fake(
    left: Tensor,
    right: Tensor,
    count: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    output = torch.empty(
        (left.shape[0], left.shape[2], right.shape[2], weight.shape[0]), device=left.device, dtype=weight.dtype
    )
    outer = torch.empty(
        (left.shape[0], left.shape[2], left.shape[-1], right.shape[2], right.shape[-1]),
        device=left.device,
        dtype=left.dtype,
    )
    return output, outer, torch.empty_like(output)


@torch.library.custom_op(
    "ooperators::_msa_contraction_inference",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor left, Tensor right, Tensor count, Tensor weight, Tensor? bias, str "
        "input_precision, int state, bool divide_before, float eps, bool clamp) -> Tensor"
    ),
)
def _contraction_inference(
    left: Tensor,
    right: Tensor,
    count: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> Tensor:
    return _execute_contraction(
        left, right, count, weight, bias, input_precision, state, divide_before, eps, clamp, False
    )[0]


@_contraction_inference.register_fake
def _contraction_inference_fake(
    left: Tensor,
    right: Tensor,
    count: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> Tensor:
    return _contraction_forward_fake(
        left, right, count, weight, bias, input_precision, state, divide_before, eps, clamp
    )[0]


@torch.library.custom_op(
    "ooperators::_msa_contraction_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor gradient, Tensor left, Tensor right, Tensor count, Tensor weight, Tensor? "
        "bias, Tensor outer, Tensor numerator, str input_precision, int state, bool "
        "divide_before, float eps, bool clamp, bool need_left, bool need_right, bool "
        "need_count, bool need_weight, bool need_bias) -> (Tensor?, Tensor?, Tensor?, "
        "Tensor?, Tensor?)"
    ),
)
def _contraction_backward(
    gradient: Tensor,
    left: Tensor,
    right: Tensor,
    count: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    outer: Tensor,
    numerator: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
    need_left: bool,
    need_right: bool,
    need_count: bool,
    need_weight: bool,
    need_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    gradients = _execute_contraction_backward(
        gradient,
        left,
        right,
        count,
        weight,
        bias,
        outer,
        numerator,
        input_precision,
        state,
        divide_before,
        eps,
        clamp,
        need_left,
        need_right,
        need_count,
        need_weight,
        need_bias,
    )
    grad_left, grad_right, grad_count, grad_weight, grad_bias = gradients
    return (
        None if grad_left is None else grad_left.to(left.dtype),
        None if grad_right is None else grad_right.to(right.dtype),
        None if grad_count is None else grad_count.to(count.dtype),
        None if grad_weight is None else grad_weight.to(weight.dtype),
        None if grad_bias is None or bias is None else grad_bias.to(bias.dtype),
    )


@_contraction_backward.register_fake
def _contraction_backward_fake(
    gradient: Tensor,
    left: Tensor,
    right: Tensor,
    count: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    outer: Tensor,
    numerator: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
    need_left: bool,
    need_right: bool,
    need_count: bool,
    need_weight: bool,
    need_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return (
        torch.empty(left.shape, device=left.device, dtype=left.dtype) if need_left else None,
        torch.empty(right.shape, device=right.device, dtype=right.dtype) if need_right else None,
        torch.empty(count.shape, device=count.device, dtype=count.dtype) if need_count else None,
        torch.empty(weight.shape, device=weight.device, dtype=weight.dtype) if need_weight else None,
        torch.empty(bias.shape, device=bias.device, dtype=bias.dtype) if need_bias and bias is not None else None,
    )


def _contraction_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:5], *output[1:])
    ctx.constants = inputs[5:]
    ctx.demand = ctx.needs_input_grad[:5]
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)


def _contraction_autograd_backward(ctx, gradient, *auxiliary):
    saved = ctx.saved_tensors
    arguments = (gradient, *saved, *ctx.constants, *ctx.demand)
    gradients = (
        _execute_contraction_backward(*arguments)
        if getattr(ctx, "direct", False) and not torch.is_grad_enabled()
        else _contraction_backward(*arguments)
    )
    gradients = tuple(
        None if gradient is None else gradient.to(parameter.dtype) for gradient, parameter in zip(gradients, saved[:5])
    )
    return (*gradients, *((None,) * 5))


torch.library.register_autograd(_contraction_forward, _contraction_autograd_backward, setup_context=_contraction_setup)


class _SavedContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = _execute_contraction(*inputs, True)
        ctx.save_for_backward(*inputs[:5], *output[1:])
        ctx.constants = inputs[5:]
        ctx.demand = ctx.needs_input_grad[:5]
        ctx.set_materialize_grads(False)
        ctx.direct = True
        return output[0]

    @staticmethod
    def backward(ctx, gradient):
        return _contraction_autograd_backward(ctx, gradient)


def _run_contraction(left, right, count, weight, bias, input_precision, state, divide_before, eps, clamp):
    arguments = (left, right, count, weight, bias, input_precision, state, divide_before, eps, clamp)
    if needs_grad(*arguments[:5]):
        if _plain(*arguments[:5]):
            return _SavedContraction.apply(*arguments)
        return _contraction_forward(*arguments)[0]
    if _plain(*arguments[:5]):
        return _execute_contraction(*arguments, False)[0]
    return _contraction_inference(*arguments)


def _packed_msa_dual_projection(
    msa,
    norm_weight,
    norm_bias,
    left_weight,
    right_weight,
    left_bias,
    right_bias,
    eps,
    input_precision,
    native_matmul_state,
):
    left, right = _run_projection(
        msa.reshape(1, -1, 1, msa.shape[-1]),
        norm_weight,
        norm_bias,
        left_weight,
        right_weight,
        left_bias,
        right_bias,
        eps,
        input_precision,
        native_matmul_state,
    )
    return left.reshape(msa.shape[0], left_weight.shape[0]), right.reshape(msa.shape[0], right_weight.shape[0])


def _dispatch_outer_product_mean(
    msa,
    msa_mask,
    msa_norm_weight,
    msa_norm_bias,
    left_weight,
    right_weight,
    output_weight,
    left_bias,
    right_bias,
    output_bias,
    *,
    msa_norm_eps,
    denominator_eps,
    divide_before_projection,
    clamp_denominator,
    outer_dtype,
):
    if not supports_msa_dual_projection(
        msa, msa_norm_weight, msa_norm_bias, left_weight, right_weight, left_bias, right_bias
    ):
        raise unsupported(
            "outer_product_mean",
            "requires channel-contiguous CUDA msa and contiguous affine operands with a launchable fused tile",
            msa_dtype=msa.dtype,
            projection_dtype=left_weight.dtype,
            msa_shape=tuple(msa.shape),
            hidden=left_weight.shape[0],
        )
    dtype = left_weight.dtype if outer_dtype is None else outer_dtype
    state = _native_matmul_state(dtype, left_weight.dtype, output_weight.dtype)
    precision = triton_dot_precision(left_weight.dtype, FLOAT32_DOT_TF32X3, state)
    if msa_mask is None:
        msa_mask = msa.new_ones(msa.shape[:-1])
    left, right = _run_projection(
        msa,
        msa_norm_weight,
        msa_norm_bias,
        left_weight,
        right_weight,
        left_bias,
        right_bias,
        float(msa_norm_eps),
        precision,
        state,
    )
    projection_mask = msa_mask.to(left_weight.dtype).unsqueeze(-1)
    left = (left * projection_mask).to(dtype)
    right = (right * projection_mask).to(dtype)
    count = _MSAPairCount.apply(msa_mask.float())
    precision = triton_dot_precision(output_weight.dtype, FLOAT32_DOT_TF32X3, state)
    return _run_contraction(
        left,
        right,
        count,
        output_weight,
        output_bias,
        precision,
        state,
        divide_before_projection,
        float(denominator_eps),
        clamp_denominator,
    )
