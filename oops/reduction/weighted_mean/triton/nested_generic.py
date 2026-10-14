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

from .._native import native_forward
from .generic import _cuda_forward, _output_dtype, expanded_backward


def _samples(
    values: Tensor, offsets: Tensor | None, sizes: Tensor | None, order: list[int], rank: int, batch: int
) -> tuple[Tensor, ...]:
    if offsets is not None:
        boundaries = offsets.tolist()
        if order[0] == len(order) - 1:
            return tuple(values[start:end].movedim(0, -1) for start, end in zip(boundaries, boundaries[1:]))
        assert sizes is not None
        shapes = sizes.tolist()
        inverse = tuple(order.index(dim) for dim in range(len(order)))
        return tuple(
            values[start:end].reshape(*(shape[dim] for dim in order)).permute(inverse)
            for start, end, shape in zip(boundaries, boundaries[1:], shapes)
        )
    if values.ndim == rank + 1:
        if values.shape[0] not in (1, batch):
            raise ValueError("weighted_mean dense and nested batch dimensions disagree")
        return tuple(values[0 if values.shape[0] == 1 else i] for i in range(batch))
    return (values,) * batch


def _run_nested_forward(
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
    offsets = value_offsets if value_offsets is not None else weight_offsets
    assert offsets is not None
    batch = offsets.numel() - 1
    if (
        row_offsets is None
        and value_offsets is not None
        and weight_offsets is not None
        and not torch.equal(value_offsets, weight_offsets)
    ):
        raise ValueError("weighted_mean nested observation lengths disagree")
    if row_offsets is not None and value_sizes is not None and weight_sizes is not None:
        if value_sizes.shape[0] != weight_sizes.shape[0] or not torch.equal(value_sizes[:, -1], weight_sizes[:, -1]):
            raise ValueError("weighted_mean nested observation lengths disagree")
        if (
            row_offsets is not None
            and len(value_order) > 1
            and len(weight_order) > 1
            and value_order[0] == 0
            and weight_order[0] == 0
            and not torch.equal(value_sizes[:, 0], weight_sizes[:, 0])
        ):
            raise ValueError("weighted_mean nested sample lengths disagree")
    xs = _samples(values, value_offsets, value_sizes, value_order, rank, batch)
    ws = _samples(weights, weight_offsets, weight_sizes, weight_order, rank, batch)
    outputs, states = [], []
    source_sizes = (
        value_sizes if value_sizes is not None and len(value_order) > 1 and value_order[0] == 0 else weight_sizes
    )
    for i, (x, w) in enumerate(zip(xs, ws)):
        x, w = torch.broadcast_tensors(x, w)
        if row_offsets is not None:
            assert source_sizes is not None
            if x.shape[0] != int(source_sizes[i, 0]):
                raise ValueError("weighted_mean broadcasting cannot expand nested sample lengths")
        output, state = _cuda_forward(x, w) if x.is_cuda else native_forward(x, w)
        outputs.append(output)
        states.append(state)
    if not outputs:
        return values.new_empty((0, *prefix), dtype=_output_dtype(values, weights)), values.new_empty(
            (0, *prefix, 4), dtype=torch.float64
        )
    if row_offsets is not None:
        return torch.cat(outputs), torch.cat(states)
    return torch.stack(outputs), torch.stack(states)


def _collect(grads: list[Tensor], operand: Tensor, offsets: Tensor | None, order: list[int], rank: int) -> Tensor:
    if offsets is not None:
        result = torch.cat(tuple(grad.permute(order).flatten(0, len(order) - operand.ndim) for grad in grads), 0)
    elif operand.ndim == rank + 1:
        result = torch.stack(grads).sum_to_size(operand.shape)
    else:
        result = torch.stack(grads).sum(0)
    return result.to(operand.dtype)


def _run_nested_backward(
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
    offsets = value_offsets if value_offsets is not None else weight_offsets
    assert offsets is not None
    batch = offsets.numel() - 1
    uniform_weights = weight_offsets is None and (weights.ndim == 0 or weights.shape[-1] == 1)
    xs = _samples(values, value_offsets, value_sizes, value_order, rank, batch)
    ws = _samples(weights, weight_offsets, weight_sizes, weight_order, rank, batch)
    source_sizes = (
        value_sizes if value_sizes is not None and len(value_order) > 1 and value_order[0] == 0 else weight_sizes
    )
    value_grads, weight_grads = [], []
    row_start = 0
    for i, (x, w) in enumerate(zip(xs, ws)):
        bx, bw = torch.broadcast_tensors(x, w)
        if row_offsets is not None:
            assert source_sizes is not None
            row_end = row_start + int(source_sizes[i, 0])
            state, cotangent = stats[row_start:row_end], grad[row_start:row_end]
            row_start = row_end
        else:
            state, cotangent = stats[i], grad[i]
        gx, gw = expanded_backward(
            bx,
            bw,
            state,
            cotangent,
            need_values,
            need_weights and not uniform_weights,
            values.dtype if value_offsets is not None and bx.shape == x.shape else torch.float64,
            weights.dtype if weight_offsets is not None and bw.shape == w.shape else torch.float64,
        )
        if gx is not None:
            value_grads.append(gx.sum_to_size(x.shape))
        if gw is not None:
            weight_grads.append(gw.sum_to_size(w.shape))
    return (
        (
            _collect(value_grads, values, value_offsets, value_order, rank)
            if need_values and batch
            else torch.zeros_like(values) if need_values else None
        ),
        (
            _collect(weight_grads, weights, weight_offsets, weight_order, rank)
            if need_weights and batch and not uniform_weights
            else torch.zeros_like(weights) if need_weights else None
        ),
    )
