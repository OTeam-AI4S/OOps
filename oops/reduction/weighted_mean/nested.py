from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import unsupported

from ._native import native_forward
from .triton.ops import _cuda_forward, _output_dtype, expanded_backward


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


def _collect(grads: list[Tensor], operand: Tensor, offsets: Tensor | None, order: list[int], rank: int) -> Tensor:
    if offsets is not None:
        result = torch.cat(tuple(grad.permute(order).flatten(0, len(order) - operand.ndim) for grad in grads), 0)
    elif operand.ndim == rank + 1:
        result = torch.stack(grads).sum_to_size(operand.shape)
    else:
        result = torch.stack(grads).sum(0)
    return result.to(operand.dtype)


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
