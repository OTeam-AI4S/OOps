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

from oops.runtime import functorch_transforms_active, needs_grad

from .generic import (
    _dispatch_layer_norm_backward,
    _matrix_shape,
    _native_layer_norm,
    _native_startup_supported,
    _row_addressing,
    _run_layer_norm_backward,
    _run_layer_norm_inference,
    _validate_layer_norm_backward_input,
    layer_norm_forward_with_stats,
    resolve_layer_norm_backward_plan,
)
from .schedule import LayerNormPlan


def triton_layer_norm(input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:

    operands = tuple(tensor for tensor in (input, weight, bias) if tensor is not None)
    plain = (
        not torch.compiler.is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not functorch_transforms_active()
        and not is_in_torch_dispatch_mode()
    )
    if (
        plain
        and all(tensor.dtype is input.dtype for tensor in operands)
        and (not needs_grad(*operands) or not torch._C._autograd._top_saved_tensors_default_hooks(True))
        and _native_startup_supported(input)
    ):
        return _native_layer_norm(input, weight, bias, eps)
    input, _ = _row_addressing(input)
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if plain:
        if needs_grad(*operands):
            return _EagerLayerNorm.apply(input, weight, bias, eps)[0]
        return _run_layer_norm_inference(input, weight, bias, eps)
    if needs_grad(*operands):
        return _layer_norm_forward(input, weight, bias, eps)[0]
    return _layer_norm_inference(input, weight, bias, eps)


def triton_layer_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    need_grad_input: bool,
    need_grad_weight: bool,
    need_grad_bias: bool,
    *,
    plan: LayerNormPlan | None = None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:

    if plan is None:
        return _layer_norm_backward(
            grad_output,
            input,
            weight,
            bias,
            mean,
            rstd,
            eps,
            input.dtype,
            need_grad_input,
            need_grad_weight,
            need_grad_bias,
        )
    return _run_layer_norm_backward(
        grad_output,
        input,
        weight,
        bias,
        mean,
        rstd,
        eps,
        need_grad_input,
        need_grad_weight,
        need_grad_bias,
        plan=plan,
    )


@torch.library.custom_op("ooperators::_layer_norm_forward", mutates_args=(), device_types="cuda")
def _layer_norm_forward(
    input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    return layer_norm_forward_with_stats(input, weight, bias, eps)


@torch.library.custom_op("ooperators::_layer_norm_inference", mutates_args=(), device_types="cuda")
def _layer_norm_inference(input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    return _run_layer_norm_inference(input, weight, bias, eps)


_LAYER_NORM_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor rstd, float eps, "
    "ScalarType input_dtype, bool need_grad_input, bool need_grad_weight, bool need_grad_bias) "
    "-> (Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_layer_norm_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_LAYER_NORM_BACKWARD_SCHEMA,
)
def _layer_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    input_dtype: torch.dtype,
    need_grad_input: bool,
    need_grad_weight: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    return _dispatch_layer_norm_backward(
        grad_output,
        input,
        weight,
        bias,
        mean,
        rstd,
        eps,
        input_dtype,
        need_grad_input,
        need_grad_weight,
        need_grad_bias,
    )


@torch.library.register_fake(_layer_norm_forward)
def _layer_norm_forward_fake(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    num_rows, _ = _matrix_shape(input)
    return (
        torch.empty(input.shape, device=input.device, dtype=input.dtype),
        torch.empty((num_rows,), device=input.device, dtype=torch.float32),
        torch.empty((num_rows,), device=input.device, dtype=torch.float32),
    )


@torch.library.register_fake(_layer_norm_inference)
def _layer_norm_inference_fake(input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    return torch.empty(input.shape, device=input.device, dtype=input.dtype)


@torch.library.register_fake(_layer_norm_backward)
def _layer_norm_backward_fake(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    input_dtype: torch.dtype,
    need_grad_input: bool,
    need_grad_weight: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    _validate_layer_norm_backward_input(input, input_dtype, need_grad_input, need_grad_weight)
    del grad_output, mean, rstd, eps
    return (
        torch.empty(input.shape, device=input.device, dtype=input.dtype) if need_grad_input else None,
        torch.empty_like(weight) if need_grad_weight and weight is not None else None,
        torch.empty_like(bias) if need_grad_bias and bias is not None else None,
    )


def _setup_context(
    ctx, inputs: tuple[Tensor, Tensor | None, Tensor | None, float], output: tuple[Tensor, Tensor, Tensor]
) -> None:
    input, weight, bias, eps = inputs
    _output, mean, rstd = output
    ctx.mark_non_differentiable(mean, rstd)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(input, weight, bias, mean, rstd)
    ctx.eps = eps
    ctx.input_dtype = input.dtype
    ctx.need_grad_input, ctx.need_grad_weight, ctx.need_grad_bias = ctx.needs_input_grad[:3]


def _backward(ctx, grad_output: Tensor, _grad_mean: Tensor | None, _grad_rstd: Tensor | None):
    input, weight, bias, mean, rstd = ctx.saved_tensors
    grad_input, grad_weight, grad_bias = _layer_norm_backward(
        grad_output.contiguous(),
        input,
        weight,
        bias,
        mean,
        rstd,
        ctx.eps,
        ctx.input_dtype,
        ctx.need_grad_input,
        ctx.need_grad_weight,
        ctx.need_grad_bias,
    )
    return grad_input, grad_weight, grad_bias, None


torch.library.register_autograd(_layer_norm_forward, _backward, setup_context=_setup_context)


class _EagerLayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        output = layer_norm_forward_with_stats(input, weight, bias, eps)
        _setup_context(ctx, (input, weight, bias, eps), output)
        ctx.plan = resolve_layer_norm_backward_plan(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor, _grad_mean: Tensor | None, _grad_rstd: Tensor | None):
        if torch.is_grad_enabled():
            return _backward(ctx, grad_output, _grad_mean, _grad_rstd)
        input, weight, bias, mean, rstd = ctx.saved_tensors
        grads = _run_layer_norm_backward(
            grad_output,
            input,
            weight,
            bias,
            mean,
            rstd,
            ctx.eps,
            ctx.need_grad_input,
            ctx.need_grad_weight,
            ctx.need_grad_bias,
            plan=ctx.plan,
        )
        return (*grads, None)
