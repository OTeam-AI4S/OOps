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

from dataclasses import replace

import torch
from torch import Tensor

from oops.runtime import is_compiling, is_external_fake, matmul_precision
from oops.runtime import needs_grad as tensors_need_grad

from .contract import (
    FlashAttentionBackend,
    PreparedAttention,
)
from .generic import (
    run_attention_forward,
)


def dispatch_attention(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    scale: float | None,
    inference_op,
    bias_forward_op,
    forward_op,
    prepared: PreparedAttention | None = None,
) -> Tensor:

    if mask is not None:
        previous_strides = mask.stride()
        mask = mask.to(dtype=torch.int16, memory_format=torch.contiguous_format)
        if prepared is not None and mask.stride() != previous_strides:
            prepared = replace(
                prepared,
                problem=backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=True),
            )
    precision = prepared.precision if prepared is not None else matmul_precision(query.dtype, backend.float32_dot)
    if prepared is None:
        compiling = is_compiling()
        external_fake = False if compiling else is_external_fake(query)
    else:
        compiling = external_fake = False
    if not tensors_need_grad(query, key, value, bias):
        if not compiling and not external_fake:
            return run_attention_forward(
                backend,
                query,
                key,
                value,
                mask,
                bias,
                scale=scale,
                precision=precision,
                write_lse=False,
                prepared=prepared,
            )[0]
        return inference_op(query, key, value, mask, bias, scale, precision)
    if bias is not None:
        output, _lse = bias_forward_op(query, key, value, mask, bias, scale, precision)
    else:
        output, _lse = forward_op(query, key, value, mask, scale, precision)
    return output.to(query.dtype)


def fake_attention_forward(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
) -> tuple[Tensor, Tensor]:

    problem = backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=False)
    return (
        torch.empty_like(query, dtype=torch.float32),
        torch.empty(problem.lse_shape, device=query.device, dtype=torch.float32),
    )


def fake_attention_backward(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Tensor | None,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:

    grad_bias = None
    if bias is not None and need_grad_bias:
        memory_format = torch.contiguous_format if backend.contiguous_fake_bias_gradient else torch.preserve_format
        grad_bias = torch.empty_like(bias, memory_format=memory_format)
    return (
        torch.empty_like(query) if need_grad_query else None,
        torch.empty_like(key) if need_grad_key else None,
        torch.empty_like(value) if need_grad_value else None,
        grad_bias,
    )


def setup_attention_bias_context(ctx, inputs, output) -> None:

    query, key, value, mask, bias, scale, precision = inputs
    attention_output, lse = output
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[4]
    ctx.scale = scale
    ctx.precision = precision
    ctx.mark_non_differentiable(lse)
    ctx.set_materialize_grads(False)
    # A dV-only backward never dereferences the V/output placeholder slots.
    ctx.save_for_backward(
        query, key, value if needs_delta else key, mask, bias, attention_output if needs_delta else query, lse
    )


def setup_attention_context(ctx, inputs, output) -> None:

    query, key, value, mask, scale, precision = inputs
    attention_output, lse = output
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
    ctx.scale = scale
    ctx.precision = precision
    ctx.mark_non_differentiable(lse)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        query, key, value if needs_delta else key, mask, attention_output if needs_delta else query, lse
    )


def make_attention_bias_backward(backward_op):

    def backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
        query, key, value, mask, bias, output, lse = ctx.saved_tensors
        need_grad_query, need_grad_key, need_grad_value = ctx.needs_input_grad[:3]
        grad_query, grad_key, grad_value, grad_bias = backward_op(
            grad_output,
            query,
            key,
            value,
            mask,
            bias,
            output,
            lse,
            ctx.scale,
            ctx.precision,
            need_grad_query,
            need_grad_key,
            need_grad_value,
            ctx.needs_input_grad[4],
        )
        return grad_query, grad_key, grad_value, None, grad_bias, None, None

    return backward


def make_attention_backward(backward_op):

    def backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
        query, key, value, mask, output, lse = ctx.saved_tensors
        need_grad_query, need_grad_key, need_grad_value = ctx.needs_input_grad[:3]
        grad_query, grad_key, grad_value = backward_op(
            grad_output,
            query,
            key,
            value,
            mask,
            output,
            lse,
            ctx.scale,
            ctx.precision,
            need_grad_query,
            need_grad_key,
            need_grad_value,
        )
        return grad_query, grad_key, grad_value, None, None, None

    return backward
