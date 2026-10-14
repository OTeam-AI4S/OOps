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
from ...csr.triton.backend import rows_per_edge, sum_csr
from .generic import (
    _execute_segment_max_forward,
    _execute_segment_softmax_forward,
    _execute_segment_sum_forward,
    _launch_segment_backward,
    _launch_segment_max,
    _launch_segment_softmax,
)


def _plain_segment_call(values: Tensor, row_offsets: Tensor) -> bool:
    operands = (values, row_offsets)
    return (
        not is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
    )


def triton_segment_sum(
    values: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    # Keep schema SymInt extents symbolic; int() would specialize them.
    if tensors_need_grad(values):
        if _plain_segment_call(values, row_offsets):
            return _SegmentSum.apply(values, row_offsets, segment_ids, out_segments, inferred_segments)
        return _segment_sum_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    compiling = is_compiling()
    value_type = type(values)
    external_fake = (
        not compiling and value_type is not Tensor and value_type is not torch.nn.Parameter and is_external_fake(values)
    )
    if compiling or external_fake:
        return _segment_sum_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    return sum_csr(values, row_offsets, out_segments, inferred_segments)


@torch.library.custom_op("ooperators::_segment_sum_csr_forward", mutates_args=(), device_types="cuda")
def _segment_sum_csr_forward(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    return _execute_segment_sum_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)


@torch.library.register_fake(_segment_sum_csr_forward)
def _segment_sum_csr_forward_fake(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del row_offsets, segment_ids, inferred_segments
    return torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)


def _segment_sum_csr_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor | None, int, int],
    output: Tensor,
) -> None:
    values, row_offsets, segment_ids, _out_segments, inferred_segments = inputs
    del output
    ctx.save_for_backward(row_offsets, segment_ids)
    ctx.num_edges = values.shape[0]
    ctx.inferred_segments = inferred_segments


def _segment_sum_csr_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None, None]:
    row_offsets, cached_segment_ids = ctx.saved_tensors
    if not torch.is_grad_enabled():
        return (
            _run_segment_backward(
                grad_out, grad_out, grad_out, row_offsets, ctx.num_edges, ctx.inferred_segments, "sum"
            ),
            None,
            None,
            None,
            None,
        )
    segment_ids = canonical_segment_ids(row_offsets, ctx.num_edges, cached_segment_ids)
    grad_values = rows_per_edge(grad_out, segment_ids, ctx.inferred_segments)
    return grad_values, None, None, None, None


torch.library.register_autograd(
    _segment_sum_csr_forward,
    _segment_sum_csr_forward_autograd_backward,
    setup_context=_segment_sum_csr_forward_setup_context,
)


class _SegmentSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_offsets, segment_ids, out_segments, inferred_segments):
        output = sum_csr(values, row_offsets, out_segments, inferred_segments)
        _segment_sum_csr_forward_setup_context(
            ctx, (values, row_offsets, segment_ids, out_segments, inferred_segments), output
        )
        return output

    backward = staticmethod(_segment_sum_csr_forward_autograd_backward)


def triton_segment_max(
    values: Tensor,
    row_offsets: Tensor,
    out_segments: int,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    if tensors_need_grad(values):
        if _plain_segment_call(values, row_offsets):
            return _SegmentMax.apply(values, row_offsets, segment_ids, out_segments, inferred_segments)
        return _segment_max_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    compiling = is_compiling()
    value_type = type(values)
    external_fake = (
        not compiling and value_type is not Tensor and value_type is not torch.nn.Parameter and is_external_fake(values)
    )
    if compiling or external_fake:
        return _segment_max_csr_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)
    return _launch_segment_max(values, row_offsets, out_segments, inferred_segments)


@torch.library.custom_op("ooperators::_segment_max_csr_forward", mutates_args=(), device_types="cuda")
def _segment_max_csr_forward(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    return _execute_segment_max_forward(values, row_offsets, segment_ids, out_segments, inferred_segments)


@torch.library.register_fake(_segment_max_csr_forward)
def _segment_max_csr_forward_fake(
    values: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    out_segments: int,
    inferred_segments: int,
) -> Tensor:
    del row_offsets, segment_ids, inferred_segments
    return torch.empty((out_segments, *values.shape[1:]), device=values.device, dtype=values.dtype)


def _segment_max_csr_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor | None, int, int],
    output: Tensor,
) -> None:
    values, row_offsets, segment_ids, _out_segments, inferred_segments = inputs
    ctx.save_for_backward(values, row_offsets, segment_ids, output)
    ctx.inferred_segments = inferred_segments


def _segment_max_csr_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None, None]:

    values, row_offsets, cached_segment_ids, maxima = ctx.saved_tensors
    if not torch.is_grad_enabled():
        return (
            _run_segment_backward(grad_out, values, maxima, row_offsets, values.shape[0], ctx.inferred_segments, "max"),
            None,
            None,
            None,
            None,
        )
    segment_ids = canonical_segment_ids(row_offsets, values.shape[0], cached_segment_ids)
    attained = values == rows_per_edge(maxima, segment_ids, ctx.inferred_segments)
    count_dtype = torch.float32 if values.dtype in HALF_DTYPES else values.dtype
    attaining = _segment_sum_csr_forward(
        attained.to(count_dtype),
        row_offsets,
        segment_ids,
        ctx.inferred_segments,
        ctx.inferred_segments,
    )
    # Zero tie counts occur only in unselected branches.
    divisor = rows_per_edge(attaining, segment_ids, ctx.inferred_segments).clamp(min=1)
    upstream = rows_per_edge(grad_out, segment_ids, ctx.inferred_segments)
    grad_values = torch.where(attained, upstream / divisor, torch.zeros_like(upstream))
    return grad_values.to(values.dtype), None, None, None, None


torch.library.register_autograd(
    _segment_max_csr_forward,
    _segment_max_csr_forward_autograd_backward,
    setup_context=_segment_max_csr_forward_setup_context,
)


class _SegmentMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_offsets, segment_ids, out_segments, inferred_segments):
        output = _launch_segment_max(values, row_offsets, out_segments, inferred_segments)
        _segment_max_csr_forward_setup_context(
            ctx, (values, row_offsets, segment_ids, out_segments, inferred_segments), output
        )
        return output

    backward = staticmethod(_segment_max_csr_forward_autograd_backward)


def triton_segment_softmax(
    logits: Tensor,
    row_offsets: Tensor,
    inferred_segments: int,
    segment_ids: Tensor | None = None,
) -> Tensor:

    if tensors_need_grad(logits):
        if _plain_segment_call(logits, row_offsets):
            return _SegmentSoftmax.apply(logits, row_offsets, segment_ids, inferred_segments)
        return _segment_softmax_csr_forward(logits, row_offsets, segment_ids, inferred_segments)
    compiling = is_compiling()
    logit_type = type(logits)
    external_fake = (
        not compiling and logit_type is not Tensor and logit_type is not torch.nn.Parameter and is_external_fake(logits)
    )
    if compiling or external_fake:
        return _segment_softmax_csr_forward(logits, row_offsets, segment_ids, inferred_segments)
    return _launch_segment_softmax(logits, row_offsets, inferred_segments)


@torch.library.custom_op("ooperators::_segment_softmax_csr_forward", mutates_args=(), device_types="cuda")
def _segment_softmax_csr_forward(
    logits: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
) -> Tensor:
    return _execute_segment_softmax_forward(logits, row_offsets, segment_ids, inferred_segments)


@torch.library.register_fake(_segment_softmax_csr_forward)
def _segment_softmax_csr_forward_fake(
    logits: Tensor,
    row_offsets: Tensor,
    segment_ids: Tensor | None,
    inferred_segments: int,
) -> Tensor:
    del row_offsets, segment_ids, inferred_segments
    return torch.empty_like(logits)


def _segment_softmax_csr_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor | None, int],
    output: Tensor,
) -> None:
    _logits, row_offsets, segment_ids, inferred_segments = inputs
    ctx.save_for_backward(output, row_offsets, segment_ids)
    ctx.inferred_segments = inferred_segments


def _segment_softmax_csr_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None]:
    r"""The VJP is ``y * (g - sum_segment(g * y))``."""

    probabilities, row_offsets, cached_segment_ids = ctx.saved_tensors
    if not torch.is_grad_enabled():
        return (
            _run_segment_backward(
                grad_out,
                probabilities,
                probabilities,
                row_offsets,
                probabilities.shape[0],
                ctx.inferred_segments,
                "softmax",
            ),
            None,
            None,
            None,
        )
    segment_ids = canonical_segment_ids(row_offsets, probabilities.shape[0], cached_segment_ids)
    accumulate_dtype = torch.float32 if probabilities.dtype in HALF_DTYPES else probabilities.dtype
    probs = probabilities.to(accumulate_dtype)
    upstream = grad_out.to(accumulate_dtype)
    row_totals = _segment_sum_csr_forward(
        (probs * upstream).contiguous(),
        row_offsets,
        segment_ids,
        ctx.inferred_segments,
        ctx.inferred_segments,
    )
    grad_logits = probs * (upstream - rows_per_edge(row_totals, segment_ids, ctx.inferred_segments))
    return grad_logits.to(probabilities.dtype), None, None, None


torch.library.register_autograd(
    _segment_softmax_csr_forward,
    _segment_softmax_csr_forward_autograd_backward,
    setup_context=_segment_softmax_csr_forward_setup_context,
)


class _SegmentSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, row_offsets, segment_ids, inferred_segments):
        output = _launch_segment_softmax(logits, row_offsets, inferred_segments)
        _segment_softmax_csr_forward_setup_context(ctx, (logits, row_offsets, segment_ids, inferred_segments), output)
        return output

    backward = staticmethod(_segment_softmax_csr_forward_autograd_backward)


@torch.library.custom_op("ooperators::_segment_csr_backward", mutates_args=(), device_types="cuda")
def _segment_csr_backward(
    grad_out: Tensor,
    values: Tensor,
    output: Tensor,
    row_offsets: Tensor,
    num_edges: int,
    inferred_segments: int,
    reduction: str,
) -> Tensor:
    return _launch_segment_backward(grad_out, values, output, row_offsets, num_edges, inferred_segments, reduction)


def _run_segment_backward(
    grad_out: Tensor,
    values: Tensor,
    output: Tensor,
    row_offsets: Tensor,
    num_edges: int,
    inferred_segments: int,
    reduction: str,
) -> Tensor:
    grad_type = type(grad_out)
    opaque = (
        is_compiling()
        or functorch_transforms_active()
        or (grad_type is not Tensor and grad_type is not torch.nn.Parameter and is_external_fake(grad_out))
    )
    operation = _segment_csr_backward if opaque else _launch_segment_backward
    return operation(grad_out, values, output, row_offsets, num_edges, inferred_segments, reduction)


@torch.library.register_fake(_segment_csr_backward)
def _segment_csr_backward_fake(
    grad_out: Tensor,
    values: Tensor,
    output: Tensor,
    row_offsets: Tensor,
    num_edges: int,
    inferred_segments: int,
    reduction: str,
) -> Tensor:
    del grad_out, output, row_offsets, inferred_segments, reduction
    return torch.empty((num_edges, *values.shape[1:]), device=values.device, dtype=values.dtype)
