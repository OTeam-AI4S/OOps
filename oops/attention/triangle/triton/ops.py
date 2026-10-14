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

from collections.abc import Callable
from functools import lru_cache
from math import prod

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import forward_mode_active, functorch_transforms_active, is_compiling, matmul_precision

from .generic import (
    _FLOAT32_DOT,
    _run_backward,
    _run_forward,
    _run_inference,
)

_PLAIN_TENSOR_TYPES = (Tensor, torch.nn.Parameter)


def uniform_sample_table(
    num_samples: int,
    groups: int,
    cols: int,
    num_heads: int,
    bias_groups: int,
    device: torch.device,
) -> Tensor:
    r"""Table columns: cell_start, cols, groups, bias_start, bias_groups,
    bias_group_stride, bias_head_stride.
    """

    block = cols * cols
    if (
        is_compiling()
        or is_in_torch_dispatch_mode()
        or functorch_transforms_active()
        or device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        sample = torch.arange(num_samples, device=device, dtype=torch.int64)
        columns = (
            sample * (groups * cols),
            sample * 0 + cols,
            sample * 0 + groups,
            sample * (bias_groups * num_heads * block),
            sample * 0 + max(bias_groups, 1),
            sample * 0 + num_heads * block,
            sample * 0 + block,
        )
        return torch.stack(columns, dim=1)
    return _cached_uniform_sample_table(num_samples, groups, cols, num_heads, bias_groups, device)


@lru_cache(maxsize=256)
def _cached_uniform_sample_table(
    num_samples: int, groups: int, cols: int, num_heads: int, bias_groups: int, device: torch.device
) -> Tensor:
    block = cols * cols
    rows = [
        (
            sample * groups * cols,
            cols,
            groups,
            sample * bias_groups * num_heads * block,
            max(bias_groups, 1),
            num_heads * block,
            block,
        )
        for sample in range(num_samples)
    ]
    return torch.tensor(rows, dtype=torch.int64, device=device)


def triton_triangle_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    *,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    semantic: bool,
) -> Tensor:

    precision = matmul_precision(query.dtype, _FLOAT32_DOT)
    operands = tuple(operand for operand in (query, key, value, bias) if operand is not None)
    needs_grad = torch.is_grad_enabled() and any(operand.requires_grad for operand in operands)
    lean = (
        not semantic
        and all(type(operand) in _PLAIN_TENSOR_TYPES for operand in (*operands, sample_table))
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
        and not forward_mode_active(*operands)
    )
    arguments = (
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
    )
    if lean:
        if needs_grad:
            output, _lse = _TriangleAttention.apply(*arguments)
            return output.to(query.dtype)
        return _run_inference(*arguments)
    if needs_grad:
        output, _lse = _triangle_attention_forward(*arguments)
        return output.to(query.dtype)
    return _triangle_attention_inference(*arguments)


_FORWARD_SCHEMA = (
    "(Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor? bias, Tensor sample_table, "
    "SymInt uniform_groups, SymInt uniform_cols, bool bias_shared_only, bool qkv_transposed, "
    "bool mask_transposed, float? scale, str precision) -> (Tensor, Tensor)"
)


_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor? bias, "
    "Tensor output, Tensor lse, Tensor sample_table, SymInt uniform_groups, SymInt uniform_cols, "
    "bool bias_shared_only, bool qkv_transposed, bool mask_transposed, "
    "float? scale, str precision, bool need_grad_query, bool need_grad_key, bool need_grad_value, "
    "bool need_grad_bias) -> (Tensor?, Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_triangle_attention_forward", mutates_args=(), device_types="cuda", schema=_FORWARD_SCHEMA
)
def _triangle_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    return _run_forward(
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
    )


@torch.library.register_fake(_triangle_attention_forward)
def _triangle_attention_forward_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    output = torch.empty_like(query, dtype=torch.float32)
    lse = torch.empty((prod(query.shape[:-2]), query.shape[-2]), device=query.device, dtype=torch.float32)
    return output, lse


_INFERENCE_SCHEMA = (
    "(Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor? bias, Tensor sample_table, "
    "SymInt uniform_groups, SymInt uniform_cols, bool bias_shared_only, bool qkv_transposed, "
    "bool mask_transposed, float? scale, str precision) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_triangle_attention_inference", mutates_args=(), device_types="cuda", schema=_INFERENCE_SCHEMA
)
def _triangle_attention_inference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
) -> Tensor:
    return _run_inference(
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
    )


@torch.library.register_fake(_triangle_attention_inference)
def _triangle_attention_inference_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
) -> Tensor:
    return torch.empty_like(query)


@torch.library.custom_op(
    "ooperators::_triangle_attention_backward", mutates_args=(), device_types="cuda", schema=_BACKWARD_SCHEMA
)
def _triangle_attention_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    lse: Tensor,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return _run_backward(
        grad_output,
        query,
        key,
        value,
        mask,
        bias,
        output,
        lse,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
        need_grad_query,
        need_grad_key,
        need_grad_value,
        need_grad_bias,
    )


@torch.library.register_fake(_triangle_attention_backward)
def _triangle_attention_backward_fake(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    lse: Tensor,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return (
        torch.empty_like(query) if need_grad_query else None,
        torch.empty_like(key) if need_grad_key else None,
        torch.empty_like(value) if need_grad_value else None,
        torch.empty_like(bias) if need_grad_bias and bias is not None else None,
    )


def _setup_context(ctx, inputs, output) -> None:
    query, key, value, mask, bias, sample_table, *scalars = inputs
    attention_output, lse = output
    # dV needs no delta; dQ, dK, and dBias do.
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[4]
    ctx.scalars = scalars
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(lse)
    ctx.save_for_backward(query, key, value, mask, bias, attention_output if needs_delta else query, lse, sample_table)


def _backward_from_context(ctx, grad_output: Tensor, grad_lse: Tensor | None, *, semantic: bool):
    del grad_lse
    query, key, value, mask, bias, output, lse, sample_table = ctx.saved_tensors
    ctx.maybe_clear_saved_tensors()
    backward: Callable = _triangle_attention_backward if semantic else _run_backward
    gradients = backward(
        grad_output,
        query,
        key,
        value,
        mask,
        bias,
        output,
        lse,
        sample_table,
        *ctx.scalars,
        need_grad_query=ctx.needs_input_grad[0],
        need_grad_key=ctx.needs_input_grad[1],
        need_grad_value=ctx.needs_input_grad[2],
        need_grad_bias=ctx.needs_input_grad[4],
    )
    grad_query, grad_key, grad_value, grad_bias = gradients
    return grad_query, grad_key, grad_value, None, grad_bias, *((None,) * 8)


def _autograd_backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
    return _backward_from_context(ctx, grad_output, grad_lse, semantic=True)


def _plain_autograd_backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
    return _backward_from_context(ctx, grad_output, grad_lse, semantic=torch.is_grad_enabled())


class _TriangleAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = _run_forward(*inputs)
        _setup_context(ctx, inputs, output)
        return output

    backward = staticmethod(_plain_autograd_backward)


torch.library.register_autograd(_triangle_attention_forward, _autograd_backward, setup_context=_setup_context)
