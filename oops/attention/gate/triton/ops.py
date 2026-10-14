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

from oops.runtime import is_compiling, require_triton_dot_precision

from .generic import (
    _FLOAT32_DOT,
    _OPERATOR,
    _attention_gate_multiply_backward_impl,
    _gated_shape,
    _native_logits,
    _projection_backward_impl,
    _projection_forward_impl,
    _projection_inference_impl,
    _projection_linear_gradients,
)
from .schedule import GateProjectionProblem


class _Relabel(torch.autograd.Function):
    r"""Forward uses ``values``; derivatives follow ``source``."""

    @staticmethod
    def forward(ctx, values: Tensor, source: Tensor) -> Tensor:
        del ctx, source
        return values.detach()

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[None, Tensor]:
        del ctx
        return None, grad


def _multiply_backward_graph(
    grad: Tensor, attention: Tensor, logits: Tensor, need_attention: bool, need_logits: bool
) -> tuple[Tensor | None, Tensor | None]:

    grad_attention, grad_logits = _attention_gate_multiply_backward_impl(
        grad, attention, logits, need_attention, need_logits
    )
    gate = torch.sigmoid(logits)
    if grad_attention is not None:
        composed_attention = (grad * gate).reshape(attention.shape)
        grad_attention = _Relabel.apply(grad_attention, composed_attention)
    if grad_logits is not None:
        composed_logits = torch.ops.aten.sigmoid_backward(grad * attention.reshape(logits.shape), gate)
        grad_logits = _Relabel.apply(grad_logits, composed_logits)
    return grad_attention, grad_logits


def _projection_backward_graph(
    grad: Tensor,
    attention: Tensor,
    single: Tensor,
    weight: Tensor,
    logits: Tensor,
    need_attention: bool,
    need_single: bool,
    need_weight: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:

    if need_single or need_weight:
        logits = _Relabel.apply(logits, _native_logits(single, weight))
    grad_attention, grad_logits = _multiply_backward_graph(
        grad, attention, logits, need_attention, need_single or need_weight
    )
    if grad_logits is None:
        return grad_attention, None, None
    grad_single, grad_weight = _projection_linear_gradients(grad_logits, single, weight, need_single, need_weight)
    return grad_attention if need_attention else None, grad_single, grad_weight


@torch.library.custom_op(
    "ooperators::_attention_gate_projection_forward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor attention, Tensor single, Tensor weight, str precision) -> (Tensor, Tensor)",
)
def _attention_gate_projection_forward(
    attention: Tensor, single: Tensor, weight: Tensor, precision: str, problem: GateProjectionProblem | None = None
) -> tuple[Tensor, Tensor]:
    require_triton_dot_precision(_OPERATOR, single.dtype, _FLOAT32_DOT, precision)
    return _projection_forward_impl(attention, single, weight, precision)


@_attention_gate_projection_forward.register_fake
def _attention_gate_projection_forward_fake(
    attention: Tensor, single: Tensor, weight: Tensor, precision: str, problem: GateProjectionProblem | None = None
) -> tuple[Tensor, Tensor]:
    del attention, precision
    output = single.new_empty(_gated_shape(single, weight))
    return output, torch.empty_like(output)


@torch.library.custom_op(
    "ooperators::_attention_gate_projection_inference",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor attention, Tensor single, Tensor weight, str precision) -> Tensor",
)
def _attention_gate_projection_inference(attention: Tensor, single: Tensor, weight: Tensor, precision: str) -> Tensor:
    require_triton_dot_precision(_OPERATOR, single.dtype, _FLOAT32_DOT, precision)
    return _projection_inference_impl(attention, single, weight, precision)


@_attention_gate_projection_inference.register_fake
def _attention_gate_projection_inference_fake(
    attention: Tensor, single: Tensor, weight: Tensor, precision: str
) -> Tensor:
    del attention, precision
    return single.new_empty(_gated_shape(single, weight))


@torch.library.custom_op(
    "ooperators::_attention_gate_projection_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor grad, Tensor attention, Tensor single, Tensor weight, Tensor logits, bool need_attention, "
        "bool need_single, bool need_weight) -> (Tensor?, Tensor?, Tensor?)"
    ),
)
def _attention_gate_projection_backward(
    grad: Tensor,
    attention: Tensor,
    single: Tensor,
    weight: Tensor,
    logits: Tensor,
    need_attention: bool,
    need_single: bool,
    need_weight: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    return _projection_backward_impl(grad, attention, single, weight, logits, need_attention, need_single, need_weight)


@_attention_gate_projection_backward.register_fake
def _attention_gate_projection_backward_fake(
    grad: Tensor,
    attention: Tensor,
    single: Tensor,
    weight: Tensor,
    logits: Tensor,
    need_attention: bool,
    need_single: bool,
    need_weight: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    del grad, logits
    return (
        torch.empty(attention.shape, dtype=attention.dtype, device=attention.device) if need_attention else None,
        torch.empty_like(single) if need_single else None,
        torch.empty_like(weight) if need_weight else None,
    )


def _projection_setup_context(ctx, inputs, output) -> None:
    attention, single, weight, _precision = inputs
    _gated, logits = output
    _save_projection_context(ctx, attention, single, weight, logits)
    ctx.mark_non_differentiable(logits)


def _save_projection_context(ctx, attention: Tensor, single: Tensor, weight: Tensor, logits: Tensor) -> None:
    if not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2]:
        attention = logits.unflatten(-1, attention.shape[-2:])
        single = weight = logits
    ctx.save_for_backward(attention, single, weight, logits)


def _projection_autograd_backward(ctx, grad: Tensor, _grad_logits: Tensor | None):
    attention, single, weight, logits = ctx.saved_tensors
    needs = tuple(ctx.needs_input_grad[:3])
    if torch.is_grad_enabled():
        return (*_projection_backward_graph(grad, attention, single, weight, logits, *needs), None)
    return (*_attention_gate_projection_backward(grad, attention, single, weight, logits, *needs), None)


torch.library.register_autograd(
    _attention_gate_projection_forward, _projection_autograd_backward, setup_context=_projection_setup_context
)


class _GateProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, attention: Tensor, single: Tensor, weight: Tensor, precision: str, problem: GateProjectionProblem
    ) -> Tensor:
        output, logits = _projection_forward_impl(attention, single, weight, precision, problem)
        _save_projection_context(ctx, attention, single, weight, logits)
        return output

    @staticmethod
    def backward(ctx, grad: Tensor):
        attention, single, weight, logits = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        needs = tuple(ctx.needs_input_grad[:3])
        if torch.is_grad_enabled():
            return (*_projection_backward_graph(grad, attention, single, weight, logits, *needs), None, None)
        return (*_projection_backward_impl(grad, attention, single, weight, logits, *needs), None, None)


def project_gate(attention: Tensor, single: Tensor, weight: Tensor, problem: GateProjectionProblem) -> Tensor:
    precision = problem.input_precision
    needs_grad = torch.is_grad_enabled() and (attention.requires_grad or single.requires_grad or weight.requires_grad)
    if is_compiling():
        if needs_grad:
            return _attention_gate_projection_forward(attention, single, weight, precision)[0]
        return _attention_gate_projection_inference(attention, single, weight, precision)
    if needs_grad:
        return _GateProjection.apply(attention, single, weight, precision, problem)
    return _projection_inference_impl(attention, single, weight, precision, problem)
