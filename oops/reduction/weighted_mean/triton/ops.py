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

from oops.runtime import functorch_transforms_active, unsupported

from .generic import _execute_forward, _output_dtype, _run_backward


@torch.library.custom_op("ooperators::_weighted_mean_forward", mutates_args=())
def weighted_mean_forward(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    return _execute_forward(values, weights)


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
