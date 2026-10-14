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

from oops.runtime import HALF_DTYPES, functorch_transforms_active, is_compiling, is_external_fake
from oops.runtime import needs_grad as tensors_need_grad

from ...csr import canonical_segment_ids
from ...csr.triton.backend import rows_per_edge
from .generic import _execute_weighted_sum_forward, _launch_csr_weighted_sum, _launch_weighted_sum_backward


def _plain_weighted_sum_call(values: Tensor, weights: Tensor, row_offsets: Tensor) -> bool:
    operands = (values, weights, row_offsets)
    return (
        not is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
    )


def triton_csr_weighted_sum(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    if tensors_need_grad(values, weights):
        if _plain_weighted_sum_call(values, weights, row_offsets):
            return _CSRWeightedSum.apply(values, weights, row_offsets, segment_ids, out_segments, inferred_segments)
        return _csr_weighted_sum_forward(values, weights, row_offsets, segment_ids, out_segments, inferred_segments)
    compiling = is_compiling()
    value_type = type(values)
    external_fake = (
        not compiling and value_type is not Tensor and value_type is not torch.nn.Parameter and is_external_fake(values)
    )
    if compiling or external_fake:
        return _csr_weighted_sum_forward(values, weights, row_offsets, segment_ids, out_segments, inferred_segments)
    return _launch_csr_weighted_sum(values, weights, row_offsets, out_segments, inferred_segments)


@torch.library.custom_op("ooperators::_csr_weighted_sum_forward", mutates_args=(), device_types="cuda")
def _csr_weighted_sum_forward(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    return _execute_weighted_sum_forward(values, weights, row_offsets, segment_ids, out_segments, inferred_segments)


@torch.library.register_fake(_csr_weighted_sum_forward)
def _csr_weighted_sum_forward_fake(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del weights, row_offsets, segment_ids, inferred_segments
    return torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)


def _csr_weighted_sum_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor | None, int, int],
    output: Tensor,
) -> None:
    values, weights, row_offsets, segment_ids, _out_segments, inferred_segments = inputs
    del output
    ctx.save_for_backward(values, weights, row_offsets, segment_ids)
    ctx.inferred_segments = inferred_segments


def _csr_weighted_sum_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
) -> tuple[Tensor | None, Tensor | None, None, None, None, None]:

    values, weights, row_offsets, cached_segment_ids = ctx.saved_tensors
    need_grad_values, need_grad_weights = ctx.needs_input_grad[:2]
    if not torch.is_grad_enabled():
        grad_values, grad_weights = _run_weighted_sum_backward(
            values,
            weights,
            grad_out,
            row_offsets,
            cached_segment_ids,
            ctx.inferred_segments,
            need_grad_values,
            need_grad_weights,
        )
        return grad_values, grad_weights, None, None, None, None
    segment_ids = canonical_segment_ids(row_offsets, values.shape[0], cached_segment_ids)
    upstream = rows_per_edge(grad_out, segment_ids, ctx.inferred_segments)
    work_dtype = torch.float32 if values.dtype in HALF_DTYPES else values.dtype
    upstream = upstream.to(work_dtype)
    covered = segment_ids < ctx.inferred_segments
    grad_values = None
    grad_weights = None
    if need_grad_values:
        weight_mask = covered.reshape(values.shape[0], *((1,) * (weights.dim() - 1)))
        work_weights = torch.where(weight_mask, weights.to(work_dtype), 0)
        grad_values = (upstream * _weights_broadcast_over(work_weights, values)).to(values.dtype)
    if need_grad_weights:
        value_mask = covered.reshape(values.shape[0], *((1,) * (values.dim() - 1)))
        work_values = torch.where(value_mask, values.to(work_dtype), 0)
        grad_weights = _sum_to_weight_layout(upstream * work_values, weights).to(weights.dtype)
    return grad_values, grad_weights, None, None, None, None


torch.library.register_autograd(
    _csr_weighted_sum_forward,
    _csr_weighted_sum_forward_autograd_backward,
    setup_context=_csr_weighted_sum_forward_setup_context,
)


class _CSRWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weights, row_offsets, segment_ids, out_segments, inferred_segments):
        output = _launch_csr_weighted_sum(values, weights, row_offsets, out_segments, inferred_segments)
        _csr_weighted_sum_forward_setup_context(
            ctx, (values, weights, row_offsets, segment_ids, out_segments, inferred_segments), output
        )
        return output

    backward = staticmethod(_csr_weighted_sum_forward_autograd_backward)


@torch.library.custom_op(
    "ooperators::_csr_weighted_sum_backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor values, Tensor weights, Tensor grad_out, Tensor row_offsets, Tensor? segment_ids, "
    "SymInt inferred_segments, bool need_grad_values, bool need_grad_weights) -> (Tensor?, Tensor?)",
)
def _csr_weighted_sum_backward(
    values: Tensor,
    weights: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
    need_grad_values: bool,
    need_grad_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _launch_weighted_sum_backward(
        values, weights, grad_out, row_offsets, segment_ids, inferred_segments, need_grad_values, need_grad_weights
    )


def _run_weighted_sum_backward(
    values: Tensor,
    weights: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
    need_grad_values: bool,
    need_grad_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    grad_type = type(grad_out)
    opaque = (
        is_compiling()
        or functorch_transforms_active()
        or (grad_type is not Tensor and grad_type is not torch.nn.Parameter and is_external_fake(grad_out))
    )
    operation = _csr_weighted_sum_backward if opaque else _launch_weighted_sum_backward
    return operation(
        values, weights, grad_out, row_offsets, segment_ids, inferred_segments, need_grad_values, need_grad_weights
    )


@torch.library.register_fake(_csr_weighted_sum_backward)
def _csr_weighted_sum_backward_fake(
    values: Tensor,
    weights: Tensor,
    grad_out: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
    need_grad_values: bool,
    need_grad_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del grad_out, row_offsets, segment_ids, inferred_segments
    return (
        torch.empty_like(values) if need_grad_values else None,
        torch.empty_like(weights) if need_grad_weights else None,
    )


def _weights_broadcast_over(weights: Tensor, values: Tensor) -> Tensor:

    if weights.dim() == values.dim():
        return weights
    return weights.reshape(*weights.shape, *((1,) * (values.dim() - weights.dim())))


def _sum_to_weight_layout(product: Tensor, weights: Tensor) -> Tensor:

    reduced = product
    if product.dim() > weights.dim():
        reduced = reduced.sum(dim=tuple(range(weights.dim(), product.dim())))
    spread = tuple(axis for axis in range(1, weights.dim()) if weights.shape[axis] != reduced.shape[axis])
    if spread:
        reduced = reduced.sum(dim=spread, keepdim=True)
    return reduced
