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

from math import prod

import torch
import triton
from torch import Tensor

from oops.runtime import cuda_device_context

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


def _execute_forward(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    values, weights = torch.broadcast_tensors(values, weights)
    return _cuda_forward(values, weights) if values.is_cuda else native_forward(values, weights)


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
