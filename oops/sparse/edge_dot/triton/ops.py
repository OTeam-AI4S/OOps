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

from oops.runtime import functorch_transforms_active, is_compiling, is_external_fake
from oops.runtime import needs_grad as tensors_need_grad

from .generic import _bind_edge_dot_problem, _launch_edge_dot_node, _launch_edge_dot_node_backward
from .profiles import EDGE_DOT_SCHEDULE
from .schedule import EdgeDotPlan


def triton_edge_dot_node(
    edge_values: Tensor,
    node_values: Tensor,
    dst_index: Tensor,
    *,
    plan: EdgeDotPlan | None = None,
) -> Tensor:

    if tensors_need_grad(edge_values, node_values):
        if _plain_edge_dot_call(edge_values, node_values, dst_index):
            return _EagerEdgeDotNode.apply(edge_values, node_values, dst_index, plan)
        return _edge_dot_node_forward(edge_values, node_values, dst_index)
    if plan is not None:
        return _launch_edge_dot_node(edge_values, node_values, dst_index, plan=plan)
    compiling = is_compiling()
    edge_type = type(edge_values)
    external_fake = (
        not compiling
        and edge_type is not Tensor
        and edge_type is not torch.nn.Parameter
        and is_external_fake(edge_values)
    )
    if compiling or external_fake:
        return _edge_dot_node_forward(edge_values, node_values, dst_index)
    return _launch_edge_dot_node(edge_values, node_values, dst_index, plan=plan)


@torch.library.custom_op("ooperators::_edge_dot_node_forward", mutates_args=(), device_types="cuda")
def _edge_dot_node_forward(edge_values: Tensor, node_values: Tensor, dst_index: Tensor) -> Tensor:
    return _launch_edge_dot_node(edge_values, node_values, dst_index)


@torch.library.register_fake(_edge_dot_node_forward)
def _edge_dot_node_forward_fake(edge_values: Tensor, node_values: Tensor, dst_index: Tensor) -> Tensor:
    del node_values, dst_index
    return edge_values.new_empty(edge_values.shape[:-1])


_EDGE_DOT_NODE_BACKWARD_SCHEMA = (
    "(Tensor grad_out, Tensor edge_values, Tensor node_values, Tensor dst_index, "
    "bool need_grad_edge_values, bool need_grad_node_values) -> (Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_edge_dot_node_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_EDGE_DOT_NODE_BACKWARD_SCHEMA,
)
def _edge_dot_node_backward(
    grad_out: Tensor,
    edge_values: Tensor,
    node_values: Tensor,
    dst_index: Tensor,
    need_grad_edge_values: bool,
    need_grad_node_values: bool,
) -> tuple[Tensor | None, Tensor | None]:
    return _launch_edge_dot_node_backward(
        grad_out, edge_values, node_values, dst_index, need_grad_edge_values, need_grad_node_values
    )


@torch.library.register_fake(_edge_dot_node_backward)
def _edge_dot_node_backward_fake(
    grad_out: Tensor,
    edge_values: Tensor,
    node_values: Tensor,
    dst_index: Tensor,
    need_grad_edge_values: bool,
    need_grad_node_values: bool,
) -> tuple[Tensor | None, Tensor | None]:
    del grad_out, dst_index
    return (
        torch.empty_like(edge_values) if need_grad_edge_values else None,
        torch.empty_like(node_values) if need_grad_node_values else None,
    )


def _edge_dot_node_forward_setup_context(ctx, inputs: tuple[Tensor, Tensor, Tensor], output: Tensor) -> None:
    edge_values, node_values, dst_index = inputs
    del output
    ctx.save_for_backward(edge_values, node_values, dst_index)


def _edge_dot_node_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None, None]:
    edge_values, node_values, dst_index = ctx.saved_tensors
    need_grad_edge_values, need_grad_node_values = ctx.needs_input_grad[:2]
    grad_edge_values, grad_node_values = _edge_dot_node_backward(
        grad_out,
        edge_values,
        node_values,
        dst_index,
        need_grad_edge_values,
        need_grad_node_values,
    )
    return grad_edge_values, grad_node_values, None


torch.library.register_autograd(
    _edge_dot_node_forward,
    _edge_dot_node_forward_autograd_backward,
    setup_context=_edge_dot_node_forward_setup_context,
)


def _plain_edge_dot_call(*operands: Tensor) -> bool:
    return (
        not is_compiling()
        and all(type(value) in (Tensor, torch.nn.Parameter) for value in operands)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
    )


class _EagerEdgeDotNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_values, node_values, dst_index, plan):
        if plan is None:
            plan = EDGE_DOT_SCHEDULE.resolve(_bind_edge_dot_problem(edge_values, node_values))
        output = _launch_edge_dot_node(edge_values, node_values, dst_index, plan=plan)
        _edge_dot_node_forward_setup_context(ctx, (edge_values, node_values, dst_index), output)
        ctx.plan = plan
        return output

    @staticmethod
    def backward(ctx, grad_out):
        edge_values, node_values, dst_index = ctx.saved_tensors
        opaque = torch.is_grad_enabled() or not _plain_edge_dot_call(grad_out, edge_values, node_values, dst_index)
        operation = _edge_dot_node_backward if opaque else _launch_edge_dot_node_backward
        kwargs = {} if operation is _edge_dot_node_backward else {"plan": ctx.plan}
        grad_edge, grad_node = operation(
            grad_out, edge_values, node_values, dst_index, *ctx.needs_input_grad[:2], **kwargs
        )
        return grad_edge, grad_node, None, None
