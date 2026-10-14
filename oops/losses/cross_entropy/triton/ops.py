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

from oops.runtime import HALF_DTYPES, functorch_transforms_active, is_compiling, is_external_fake
from oops.runtime import needs_grad as tensors_need_grad

from .generic import (
    _run_cross_entropy,
    _run_cross_entropy_backward,
    _run_distogram_cross_entropy,
    _run_distogram_cross_entropy_backward,
)
from .profiles import CROSS_ENTROPY
from .schedule import CrossEntropyProblem


def _autocast_loss_output_is_float32(logits: Tensor) -> bool:

    return logits.is_cuda and logits.dtype in HALF_DTYPES and torch.is_autocast_enabled("cuda")


def triton_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:

    output_shape = labels.shape
    logits = logits.contiguous()
    labels = labels.contiguous()
    output_float32 = _autocast_loss_output_is_float32(logits)
    if tensors_need_grad(logits) or is_compiling() or is_external_fake(logits):
        output = _cross_entropy_forward(logits, labels, output_float32)
    else:
        output = _run_cross_entropy(logits, labels, output_float32)
    return output.reshape(output_shape)


def triton_distogram_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    *,
    eps: float = 1e-5,
    reduction: str | None = "mean",
) -> Tensor:

    logits = logits.contiguous()
    labels = labels.contiguous()
    token_mask = token_mask.contiguous()
    reduction_mode = _reduction_mode(reduction)
    output_float32 = _autocast_loss_output_is_float32(logits)
    if tensors_need_grad(logits, token_mask) or is_compiling() or is_external_fake(logits):
        loss, _denominator, _accum, _ce = _distogram_cross_entropy_forward(
            logits,
            labels,
            token_mask,
            float(eps),
            reduction_mode,
            output_float32,
        )
    else:
        loss, _denominator, _accum, _ce = _run_distogram_cross_entropy(
            logits,
            labels,
            token_mask,
            float(eps),
            reduction_mode,
            output_float32,
        )
    return loss


@torch.library.custom_op("ooperators::_cross_entropy_forward", mutates_args=(), device_types="cuda")
def _cross_entropy_forward(logits: Tensor, labels: Tensor, output_float32: bool) -> Tensor:
    return _run_cross_entropy(logits, labels, output_float32)


@torch.library.register_fake(_cross_entropy_forward)
def _cross_entropy_forward_fake(logits: Tensor, labels: Tensor, output_float32: bool) -> Tensor:
    out_dtype = torch.float32 if output_float32 else logits.dtype
    return torch.empty((labels.numel(),), device=logits.device, dtype=out_dtype)


@torch.library.custom_op("ooperators::_cross_entropy_backward", mutates_args=(), device_types="cuda")
def _cross_entropy_backward(grad_out: Tensor, logits: Tensor, labels: Tensor) -> Tensor:
    return _run_cross_entropy_backward(grad_out, logits, labels)


@torch.library.register_fake(_cross_entropy_backward)
def _cross_entropy_backward_fake(grad_out: Tensor, logits: Tensor, labels: Tensor) -> Tensor:
    del grad_out, labels
    return torch.empty_like(logits)


def _cross_entropy_forward_setup_context(ctx, inputs: tuple[Tensor, Tensor, bool], output: Tensor) -> None:
    del output
    logits, labels, _output_float32 = inputs
    ctx.save_for_backward(logits, labels)
    ctx.direct_backward = not is_compiling() and not is_external_fake(logits) and not functorch_transforms_active()
    if ctx.direct_backward:
        ctx.backward_plan = CROSS_ENTROPY.resolve(CrossEntropyProblem(labels.numel(), logits.shape[-1]))


def _cross_entropy_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None]:
    logits, labels = ctx.saved_tensors
    if ctx.direct_backward:
        grad_logits = _run_cross_entropy_backward(grad_out, logits, labels, ctx.backward_plan)
    else:
        grad_logits = _cross_entropy_backward(grad_out, logits, labels)
    return grad_logits, None, None


torch.library.register_autograd(
    _cross_entropy_forward,
    _cross_entropy_forward_autograd_backward,
    setup_context=_cross_entropy_forward_setup_context,
)


@torch.library.custom_op("ooperators::_distogram_cross_entropy_forward", mutates_args=(), device_types="cuda")
def _distogram_cross_entropy_forward(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    eps: float,
    reduction_mode: int,
    output_float32: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _run_distogram_cross_entropy(logits, labels, token_mask, eps, reduction_mode, output_float32)


@torch.library.register_fake(_distogram_cross_entropy_forward)
def _distogram_cross_entropy_forward_fake(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    eps: float,
    reduction_mode: int,
    output_float32: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del labels, token_mask, eps
    batch = logits.shape[0]
    # Modes: 0 per batch, 1 mean, 2 sum.
    loss_shape = (batch,) if reduction_mode == 0 else ()
    loss_dtype = torch.float32 if output_float32 else logits.dtype
    return (
        torch.empty(loss_shape, device=logits.device, dtype=loss_dtype),
        torch.empty((batch,), device=logits.device, dtype=torch.float32),
        torch.empty((batch, 2), device=logits.device, dtype=torch.float32),
        torch.empty(
            (logits.shape[0], logits.shape[1] * logits.shape[2]),
            device=logits.device,
            dtype=torch.float32,
        ),
    )


@torch.library.custom_op(
    "ooperators::_distogram_cross_entropy_backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor grad_out, Tensor logits, Tensor labels, Tensor token_mask, Tensor denominator, Tensor accum, "
    "Tensor ce, int reduction_mode, bool need_logits, bool need_token_mask) -> (Tensor?, Tensor?)",
)
def _distogram_cross_entropy_backward(
    grad_out: Tensor,
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    denominator: Tensor,
    accum: Tensor,
    ce: Tensor,
    reduction_mode: int,
    need_logits: bool,
    need_token_mask: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _run_distogram_cross_entropy_backward(
        grad_out, logits, labels, token_mask, denominator, accum, ce, reduction_mode, need_logits, need_token_mask
    )


@torch.library.register_fake(_distogram_cross_entropy_backward)
def _distogram_cross_entropy_backward_fake(
    grad_out: Tensor,
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    denominator: Tensor,
    accum: Tensor,
    ce: Tensor,
    reduction_mode: int,
    need_logits: bool,
    need_token_mask: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del grad_out, labels, denominator, accum, ce, reduction_mode
    return torch.empty_like(logits) if need_logits else None, torch.empty_like(token_mask) if need_token_mask else None


def _distogram_cross_entropy_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor, float, int, bool],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    logits, labels, token_mask, _eps, reduction_mode, _output_float32 = inputs
    _loss, denominator, accum, ce = output
    ctx.mark_non_differentiable(denominator, accum, ce)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(logits, labels, token_mask, denominator, accum, ce)
    ctx.reduction_mode = reduction_mode


def _distogram_cross_entropy_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
    grad_denominator: Tensor | None,
    grad_accum: Tensor | None,
    grad_ce: Tensor | None,
) -> tuple[Tensor | None, None, Tensor | None, None, None, None]:
    del grad_denominator, grad_accum, grad_ce
    logits, labels, token_mask, denominator, accum, ce = ctx.saved_tensors
    grad_logits, grad_token_mask = _distogram_cross_entropy_backward(
        grad_out,
        logits,
        labels,
        token_mask,
        denominator,
        accum,
        ce,
        ctx.reduction_mode,
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[2],
    )
    return grad_logits, None, grad_token_mask, None, None, None


torch.library.register_autograd(
    _distogram_cross_entropy_forward,
    _distogram_cross_entropy_forward_autograd_backward,
    setup_context=_distogram_cross_entropy_forward_setup_context,
)


def _reduction_mode(reduction: str | None) -> int:
    if reduction is None:
        return 0
    if reduction == "mean":
        return 1
    if reduction == "sum":
        return 2
    raise ValueError("reduction must be 'mean', 'sum', or None")
