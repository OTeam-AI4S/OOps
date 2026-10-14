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
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import unsupported

from .triton.generic import _output_dtype
from .triton.nested_generic import _run_nested_backward, _run_nested_forward


def nested_weighted_mean(values: Tensor | NestedTensor, weights: Tensor | NestedTensor) -> Tensor | NestedTensor:
    nested = tuple(operand for operand in (values, weights) if isinstance(operand, NestedTensor))
    template = nested[0]
    for operand in nested[1:]:
        if len(operand.ragged_dims) > len(template.ragged_dims):
            template = operand
    rank = max(len(operand.packed_dim_order) for operand in nested)
    hierarchical = len(template.ragged_dims) == 2
    for operand in nested:
        final = len(operand.packed_dim_order) - 1
        if (
            not operand.batch_first
            or operand.ragged_dims not in ((final,), (0, final))
            or operand.packed_dim_order
            != ((final, *range(final)) if len(operand.ragged_dims) == 1 else (0, final, *range(1, final)))
            or (len(operand.ragged_dims) == 2 and len(operand.packed_dim_order) != rank)
        ):
            raise unsupported(
                "weighted_mean",
                "nested operands require a final ragged observation axis and optional leading ragged axis",
            )
    for operand in (values, weights):
        if not isinstance(operand, NestedTensor) and operand.ndim > rank + 1:
            raise ValueError("weighted_mean dense operands cannot add axes before the nested batch")
    vx = values.concat if isinstance(values, NestedTensor) else values
    wx = weights.concat if isinstance(weights, NestedTensor) else weights
    vo = values.packed_offsets() if isinstance(values, NestedTensor) else None
    wo = weights.packed_offsets() if isinstance(weights, NestedTensor) else None
    vs = values.element_sizes() if isinstance(values, NestedTensor) else None
    ws = weights.element_sizes() if isinstance(weights, NestedTensor) else None
    vp = list(values.packed_dim_order) if isinstance(values, NestedTensor) else []
    wp = list(weights.packed_dim_order) if isinstance(weights, NestedTensor) else []
    rows = template.ragged_level_offsets(-1) if hierarchical else None
    prefixes = []
    for operand, packed in ((values, vx), (weights, wx)):
        if isinstance(operand, NestedTensor):
            prefix = tuple(packed.shape[1:])
            if hierarchical and len(operand.ragged_dims) == 1 and len(operand.packed_dim_order) == rank:
                prefix = prefix[1:]
        else:
            prefix = tuple(operand.shape[-rank:-1]) if rank > 1 else ()
            if hierarchical and operand.ndim >= rank:
                prefix = prefix[1:]
        prefixes.append(prefix)
    output_prefix = list(torch.broadcast_shapes(*prefixes))
    output = _nested_forward(vx, wx, vo, wo, vs, ws, vp, wp, rows, output_prefix, rank)[0]
    return template.packed_with_lengths(output, template.element_sizes()[:, 0]) if hierarchical else output


@torch.library.custom_op("ooperators::_weighted_mean_nested_forward", mutates_args=())
def _nested_forward(
    values: Tensor,
    weights: Tensor,
    value_offsets: Tensor | None,
    weight_offsets: Tensor | None,
    value_sizes: Tensor | None,
    weight_sizes: Tensor | None,
    value_order: list[int],
    weight_order: list[int],
    row_offsets: Tensor | None,
    prefix: list[int],
    rank: int,
) -> tuple[Tensor, Tensor]:
    return _run_nested_forward(
        values,
        weights,
        value_offsets,
        weight_offsets,
        value_sizes,
        weight_sizes,
        value_order,
        weight_order,
        row_offsets,
        prefix,
        rank,
    )


@_nested_forward.register_fake
def _nested_fake(
    values: Tensor,
    weights: Tensor,
    value_offsets: Tensor | None,
    weight_offsets: Tensor | None,
    value_sizes: Tensor | None,
    weight_sizes: Tensor | None,
    value_order: list[int],
    weight_order: list[int],
    row_offsets: Tensor | None,
    prefix: list[int],
    rank: int,
) -> tuple[Tensor, Tensor]:
    del value_sizes, weight_sizes, value_order, weight_order, rank
    offsets = row_offsets if row_offsets is not None else value_offsets if value_offsets is not None else weight_offsets
    assert offsets is not None
    rows = offsets.numel() - 1
    return values.new_empty((rows, *prefix), dtype=_output_dtype(values, weights)), values.new_empty(
        (rows, *prefix, 4), dtype=torch.float64
    )


@torch.library.custom_op(
    "ooperators::_weighted_mean_nested_backward",
    mutates_args=(),
    schema="(Tensor values, Tensor weights, Tensor stats, Tensor grad, Tensor? value_offsets, "
    "Tensor? weight_offsets, Tensor? value_sizes, Tensor? weight_sizes, int[] value_order, int[] weight_order, "
    "Tensor? row_offsets, SymInt rank, bool need_values, bool need_weights) -> (Tensor?, Tensor?)",
)
def _nested_backward(
    values: Tensor,
    weights: Tensor,
    stats: Tensor,
    grad: Tensor,
    value_offsets: Tensor | None,
    weight_offsets: Tensor | None,
    value_sizes: Tensor | None,
    weight_sizes: Tensor | None,
    value_order: list[int],
    weight_order: list[int],
    row_offsets: Tensor | None,
    rank: int,
    need_values: bool,
    need_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _run_nested_backward(
        values,
        weights,
        stats,
        grad,
        value_offsets,
        weight_offsets,
        value_sizes,
        weight_sizes,
        value_order,
        weight_order,
        row_offsets,
        rank,
        need_values,
        need_weights,
    )


@_nested_backward.register_fake
def _nested_backward_fake(
    values: Tensor,
    weights: Tensor,
    stats: Tensor,
    grad: Tensor,
    value_offsets: Tensor | None,
    weight_offsets: Tensor | None,
    value_sizes: Tensor | None,
    weight_sizes: Tensor | None,
    value_order: list[int],
    weight_order: list[int],
    row_offsets: Tensor | None,
    rank: int,
    need_values: bool,
    need_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del (
        stats,
        grad,
        value_offsets,
        weight_offsets,
        value_sizes,
        weight_sizes,
        value_order,
        weight_order,
        row_offsets,
        rank,
    )
    return torch.empty_like(values) if need_values else None, torch.empty_like(weights) if need_weights else None


def _setup_context(ctx, inputs, output) -> None:
    values, weights, vo, wo, vs, ws, vp, wp, rows, _, rank = inputs
    _, stats = output
    ctx.save_for_backward(values, weights, stats, vo, wo, vs, ws, rows)
    ctx.value_order, ctx.weight_order, ctx.rank = vp, wp, rank
    ctx.mark_non_differentiable(stats)


def _autograd_backward(ctx, grad: Tensor, grad_stats: Tensor | None):
    del grad_stats
    if torch.is_grad_enabled():
        raise unsupported("weighted_mean", "does not support higher-order AD")
    values, weights, stats, vo, wo, vs, ws, rows = ctx.saved_tensors
    gx, gw = _nested_backward(
        values,
        weights,
        stats,
        grad,
        vo,
        wo,
        vs,
        ws,
        ctx.value_order,
        ctx.weight_order,
        rows,
        ctx.rank,
        *ctx.needs_input_grad[:2],
    )
    return gx, gw, None, None, None, None, None, None, None, None, None


torch.library.register_autograd(_nested_forward, _autograd_backward, setup_context=_setup_context)
