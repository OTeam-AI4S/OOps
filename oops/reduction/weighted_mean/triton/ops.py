from __future__ import annotations

from math import prod

import torch
import triton
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import cuda_device_context, functorch_transforms_active, unsupported

from .._native import native_backward, native_forward
from . import kernel


def _output_dtype(values: Tensor, weights: Tensor) -> torch.dtype:
    return torch.promote_types(values.dtype, weights.dtype) if weights.is_floating_point() else values.dtype


def _cuda_forward(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    values, weights = values.resolve_neg(), weights.resolve_neg()
    shape, n = tuple(values.shape[:-1]), values.shape[-1]
    rows = prod(shape)
    output = torch.empty(shape, device=values.device, dtype=_output_dtype(values, weights))
    stats = torch.empty((*shape, 4), device=values.device, dtype=torch.float64)
    if rows == 0:
        return output, stats
    if n == 0:
        output.fill_(float("nan"))
        stats.zero_()
        return output, stats
    block = min(triton.next_power_of_2(n), 4096)
    parts = triton.cdiv(n, block)
    partial = torch.empty((rows, parts, 4), device=values.device, dtype=torch.float64) if parts > 1 else stats
    scaled = values.dtype == torch.float64 or weights.dtype == torch.float64
    with cuda_device_context(values.device):
        kernel._partial[(rows * parts,)](
            values,
            weights,
            partial,
            output,
            stats,
            n,
            shape,
            values.stride(),
            weights.stride(),
            parts,
            scaled,
            block,
            num_warps=8 if block >= 1024 else 4,
        )
        while parts > 1:
            merge_block = min(triton.next_power_of_2(parts), 512)
            groups = triton.cdiv(parts, merge_block)
            next_partial = (
                torch.empty((rows, groups, 4), device=values.device, dtype=torch.float64) if groups > 1 else stats
            )
            kernel._finish[(rows * groups,)](
                partial,
                next_partial,
                output,
                stats,
                parts,
                groups,
                merge_block,
                num_warps=4,
            )
            partial, parts = next_partial, groups
    return output, stats


@torch.library.custom_op("ooperators::_weighted_mean_forward", mutates_args=())
def weighted_mean_forward(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    return _execute_forward(values, weights)


def _execute_forward(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    values, weights = torch.broadcast_tensors(values, weights)
    return _cuda_forward(values, weights) if values.is_cuda else native_forward(values, weights)


@weighted_mean_forward.register_fake
def _forward_fake(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    shape = torch.broadcast_shapes(values.shape, weights.shape)[:-1]
    return (
        values.new_empty(shape, dtype=_output_dtype(values, weights)),
        values.new_empty((*shape, 4), dtype=torch.float64),
    )


@torch.library.custom_op(
    "ooperators::_weighted_mean_backward",
    mutates_args=(),
    schema="(Tensor values, Tensor weights, Tensor stats, Tensor grad, bool need_values, bool need_weights) "
    "-> (Tensor?, Tensor?)",
)
def _backward_op(
    values: Tensor,
    weights: Tensor,
    stats: Tensor,
    grad: Tensor,
    need_values: bool,
    need_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _run_backward(values, weights, stats, grad, need_values, need_weights)


def _run_backward(
    values: Tensor,
    weights: Tensor,
    stats: Tensor,
    grad: Tensor,
    need_values: bool,
    need_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    original_values, original_weights = values, weights
    uniform_weights = weights.ndim == 0 or weights.shape[-1] == 1
    values, weights = torch.broadcast_tensors(values, weights)
    gx, gw = expanded_backward(
        values,
        weights,
        stats,
        grad,
        need_values,
        need_weights and not uniform_weights,
        original_values.dtype if values.shape == original_values.shape else torch.float64,
        original_weights.dtype if weights.shape == original_weights.shape else torch.float64,
    )
    return (
        gx.sum_to_size(original_values.shape).to(original_values.dtype) if gx is not None else None,
        (
            torch.zeros_like(original_weights)
            if need_weights and uniform_weights
            else gw.sum_to_size(original_weights.shape).to(original_weights.dtype) if gw is not None else None
        ),
    )


def expanded_backward(
    values: Tensor,
    weights: Tensor,
    stats: Tensor,
    grad: Tensor,
    need_values: bool,
    need_weights: bool,
    value_grad_dtype: torch.dtype = torch.float64,
    weight_grad_dtype: torch.dtype = torch.float64,
) -> tuple[Tensor | None, Tensor | None]:
    if not values.is_cuda:
        gx, gw = native_backward(values, weights, stats, grad)
        return gx if need_values else None, gw if need_weights else None
    stats = stats.contiguous()
    gx = (
        torch.empty_like(values, dtype=value_grad_dtype, memory_format=torch.contiguous_format) if need_values else None
    )
    gw = (
        torch.empty_like(weights, dtype=weight_grad_dtype, memory_format=torch.contiguous_format)
        if need_weights
        else None
    )
    values, weights, grad = values.resolve_neg(), weights.resolve_neg(), grad.resolve_neg()
    shape, n = tuple(values.shape[:-1]), values.shape[-1]
    rows = prod(shape)
    if rows and n:
        block = min(triton.next_power_of_2(n), 1024)
        with cuda_device_context(values.device):
            kernel._backward[(rows * triton.cdiv(n, block),)](
                values,
                weights,
                stats,
                grad,
                gx,
                gw,
                n,
                shape,
                values.stride(),
                weights.stride(),
                grad.stride(),
                values.dtype == torch.float64 or weights.dtype == torch.float64,
                need_values,
                need_weights,
                block,
                num_warps=4,
            )
    return gx, gw


@_backward_op.register_fake
def _backward_fake(
    values: Tensor,
    weights: Tensor,
    stats: Tensor,
    grad: Tensor,
    need_values: bool,
    need_weights: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del stats, grad
    return (
        torch.empty_like(values, memory_format=torch.contiguous_format) if need_values else None,
        torch.empty_like(weights, memory_format=torch.contiguous_format) if need_weights else None,
    )


def _setup_context(ctx, inputs: tuple[Tensor, Tensor], output: tuple[Tensor, Tensor]) -> None:
    values, weights = inputs
    _, stats = output
    ctx.save_for_backward(values, weights, stats)
    ctx.mark_non_differentiable(stats)
    ctx.plain = False


def _autograd_backward(ctx, grad: Tensor, grad_stats: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
    del grad_stats
    if torch.is_grad_enabled():
        raise unsupported("weighted_mean", "does not support higher-order AD")
    values, weights, stats = ctx.saved_tensors
    backward = _run_backward if ctx.plain else _backward_op
    return backward(values, weights, stats, grad, ctx.needs_input_grad[0], ctx.needs_input_grad[1])


torch.library.register_autograd(weighted_mean_forward, _autograd_backward, setup_context=_setup_context)


class _WeightedMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weights):
        output = _execute_forward(values, weights)
        _setup_context(ctx, (values, weights), output)
        ctx.plain = True
        return output

    backward = staticmethod(_autograd_backward)


def run_weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    operands = (values, weights)
    if (
        not torch.compiler.is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
    ):
        return _WeightedMean.apply(values, weights)[0]
    return weighted_mean_forward(values, weights)[0]
