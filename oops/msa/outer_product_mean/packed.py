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

from oops.runtime import needs_grad

from .composed import _plain
from .triton.packed_generic import _execute_backward, _execute_forward


@torch.library.custom_op(
    "ooperators::_segmented_outer_product_mean_forward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor left, Tensor right, Tensor mask, Tensor weight, Tensor? bias, Tensor "
        "pair_offsets_cpu, Tensor msa_sizes_cpu, str input_precision, int state, bool "
        "divide_before, float eps, bool clamp) -> (Tensor, Tensor, Tensor, Tensor)"
    ),
)
def _forward(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _execute_forward(
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        input_precision,
        state,
        divide_before,
        eps,
        clamp,
        True,
    )


@_forward.register_fake
def _forward_fake(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pairs = torch.library.get_ctx().new_dynamic_size(min=0)
    output = torch.empty((pairs, weight.shape[0]), device=left.device, dtype=weight.dtype)
    return (
        output,
        left.new_empty((pairs, left.shape[1] ** 2)),
        torch.empty_like(output),
        torch.empty((pairs, 1), device=left.device, dtype=torch.float32),
    )


@torch.library.custom_op(
    "ooperators::_segmented_outer_product_mean_inference",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor left, Tensor right, Tensor mask, Tensor weight, Tensor? bias, Tensor "
        "pair_offsets_cpu, Tensor msa_sizes_cpu, str input_precision, int state, bool "
        "divide_before, float eps, bool clamp) -> Tensor"
    ),
)
def _inference(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> Tensor:
    return _execute_forward(
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        input_precision,
        state,
        divide_before,
        eps,
        clamp,
        False,
    )[0]


@_inference.register_fake
def _inference_fake(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> Tensor:
    pairs = torch.library.get_ctx().new_dynamic_size(min=0)
    return torch.empty((pairs, weight.shape[0]), device=left.device, dtype=weight.dtype)


@torch.library.custom_op(
    "ooperators::_segmented_outer_product_mean_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor gradient, Tensor left, Tensor right, Tensor mask, Tensor weight, Tensor? "
        "bias, Tensor pair_offsets_cpu, Tensor msa_sizes_cpu, Tensor outer, Tensor numerator, "
        "Tensor counts, str input_precision, int state, bool divide_before, float eps, bool "
        "clamp, bool need_left, bool need_right, bool need_mask, bool need_weight, bool "
        "need_bias) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
    ),
)
def _backward(
    gradient: Tensor,
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    outer: Tensor,
    numerator: Tensor,
    counts: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
    need_left: bool,
    need_right: bool,
    need_mask: bool,
    need_weight: bool,
    need_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return _execute_backward(
        gradient,
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        outer,
        numerator,
        counts,
        input_precision,
        state,
        divide_before,
        eps,
        clamp,
        need_left,
        need_right,
        need_mask,
        need_weight,
        need_bias,
    )


@_backward.register_fake
def _backward_fake(
    gradient: Tensor,
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    outer: Tensor,
    numerator: Tensor,
    counts: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
    need_left: bool,
    need_right: bool,
    need_mask: bool,
    need_weight: bool,
    need_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return (
        torch.empty(left.shape, device=left.device, dtype=left.dtype) if need_left else None,
        torch.empty(right.shape, device=right.device, dtype=right.dtype) if need_right else None,
        torch.empty(mask.shape, device=mask.device, dtype=mask.dtype) if need_mask else None,
        torch.empty(weight.shape, device=weight.device, dtype=weight.dtype) if need_weight else None,
        torch.empty(bias.shape, device=bias.device, dtype=bias.dtype) if need_bias and bias is not None else None,
    )


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:7], *output[1:])
    ctx.constants = inputs[7:]
    ctx.demand = ctx.needs_input_grad[:5]
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)


def _autograd_backward(ctx, gradient, *auxiliary):
    arguments = (gradient, *ctx.saved_tensors, *ctx.constants, *ctx.demand)
    grads = (
        _execute_backward(*arguments)
        if getattr(ctx, "direct", False) and not torch.is_grad_enabled()
        else _backward(*arguments)
    )
    return (*grads, *((None,) * 7))


torch.library.register_autograd(_forward, _autograd_backward, setup_context=_setup_context)


class _SavedContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = _execute_forward(*inputs, True)
        ctx.save_for_backward(*inputs[:7], *output[1:])
        ctx.constants = inputs[7:]
        ctx.demand = ctx.needs_input_grad[:5]
        ctx.set_materialize_grads(False)
        ctx.direct = True
        return output[0]

    @staticmethod
    def backward(ctx, gradient):
        return _autograd_backward(ctx, gradient)


def segmented_outer_product_mean(
    left,
    right,
    mask,
    weight,
    bias,
    msa_outer_offsets,
    msa_packed_offsets,
    pair_offsets,
    pair_offsets_cpu,
    msa_sizes_cpu,
    input_precision,
    projection_precision,
    count_precision,
    native_matmul_state,
    divide_before_projection,
    denominator_eps,
    clamp_denominator,
):
    arguments = (
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        projection_precision,
        native_matmul_state,
        divide_before_projection,
        denominator_eps,
        clamp_denominator,
    )
    if needs_grad(left, right, mask, weight, bias):
        if _plain(*arguments[:7]):
            return _SavedContraction.apply(*arguments)
        return _forward(*arguments)[0]
    if _plain(*arguments[:7]):
        return _execute_forward(*arguments, False)[0]
    return _inference(*arguments)
