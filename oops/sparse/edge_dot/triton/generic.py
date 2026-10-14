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

import warnings
from functools import lru_cache

import torch
from torch import Tensor

from oops.runtime import HALF_DTYPES, STANDARD_FLOAT_DTYPES, cuda_device_context, tensors_share_device, unsupported

from ...csr.ordering import stable_destination_ordering
from ...csr.triton.backend import sum_csr
from ...csr.triton.schedule import CSR_SUM_SCHEDULE, SegmentPlan, SegmentProblem
from . import kernel
from .profiles import EDGE_DOT_SCHEDULE
from .schedule import EdgeDotPlan, EdgeDotProblem

_SUPPORTED_DTYPES = (*STANDARD_FLOAT_DTYPES, torch.float64)

_EDGE_DOT_NODE_MAX_CHANNELS = 256

_DETERMINISTIC_EDGE_DOT_NODE_VJP_DOMAIN = (
    "requires the stable node-reduction grid to fit CUDA launch limits. Turn off determinism "
    "for this backward or use 'warn_only=True' to fall back to relaxed floating atomics."
)


def _edge_dot_groups(edge_values: Tensor) -> int:
    groups = 1
    for dim in edge_values.shape[1:-1]:
        groups *= dim
    return groups


def supports_edge_dot_node(edge_values: Tensor, node_values: Tensor, dst_index: Tensor) -> bool:

    supported = (
        edge_values.is_cuda
        and node_values.is_cuda
        and dst_index.is_cuda
        and tensors_share_device(edge_values, node_values, dst_index)
        and edge_values.is_floating_point()
        and edge_values.dtype in _SUPPORTED_DTYPES
        and node_values.dtype == edge_values.dtype
        and edge_values.is_contiguous()
        and node_values.is_contiguous()
        and dst_index.is_contiguous()
        and dst_index.dtype in (torch.int32, torch.int64)
        and edge_values.dim() >= 2
        and node_values.dim() == edge_values.dim()
        and tuple(node_values.shape[1:]) == tuple(edge_values.shape[1:])
        and dst_index.dim() == 1
        and dst_index.shape[0] == edge_values.shape[0]
        and edge_values.shape[0] > 0
        and edge_values.shape[-1] > 0
        and _edge_dot_groups(edge_values) > 0
    )
    if not supported or edge_values.shape[-1] > _EDGE_DOT_NODE_MAX_CHANNELS:
        return False
    return EDGE_DOT_SCHEDULE.supports(_bind_edge_dot_problem(edge_values, node_values))


def prepare_edge_dot_node(
    edge_values: Tensor,
    node_values: Tensor,
    dst_index: Tensor,
) -> EdgeDotPlan | None:
    r"""Resolve the CUDA layout after public feature/index validation."""

    groups = _edge_dot_groups(edge_values)
    supported = (
        edge_values.is_cuda
        and edge_values.dtype in _SUPPORTED_DTYPES
        and node_values.dtype == edge_values.dtype
        and edge_values.is_contiguous()
        and node_values.is_contiguous()
        and dst_index.is_contiguous()
        and edge_values.shape[0] > 0
        and 0 < edge_values.shape[-1] <= _EDGE_DOT_NODE_MAX_CHANNELS
        and groups > 0
    )
    if not supported:
        return None
    try:
        return _prepared_edge_dot_plan(
            int(edge_values.shape[0]), int(node_values.shape[0]), int(groups), int(edge_values.shape[-1])
        )
    except ValueError:
        return None


@lru_cache(maxsize=1024)
def _prepared_edge_dot_plan(edges: int, nodes: int, groups: int, channels: int) -> EdgeDotPlan:

    return EDGE_DOT_SCHEDULE.resolve(EdgeDotProblem(edges=edges, nodes=nodes, groups=groups, channels=channels))


def _bind_edge_dot_problem(edge_values: Tensor, node_values: Tensor) -> EdgeDotProblem:
    return EdgeDotProblem(
        edges=edge_values.shape[0],
        nodes=node_values.shape[0],
        groups=_edge_dot_groups(edge_values),
        channels=edge_values.shape[-1],
    )


def _launch_edge_dot_node(
    edge_values: Tensor,
    node_values: Tensor,
    dst_index: Tensor,
    *,
    plan: EdgeDotPlan | None = None,
) -> Tensor:
    with cuda_device_context(edge_values.device):
        edge_values = edge_values.resolve_neg().contiguous()
        node_values = node_values.resolve_neg().contiguous()
        dst_index = dst_index.resolve_neg().contiguous()
        if plan is None:
            plan = EDGE_DOT_SCHEDULE.resolve(_bind_edge_dot_problem(edge_values, node_values))
        problem = plan.problem
        outputs = problem.edges * problem.groups
        out = edge_values.new_empty(edge_values.shape[:-1])
        kernel._edge_dot_node_forward.run(
            edge_values,
            node_values,
            dst_index,
            out,
            outputs,
            problem.nodes,
            problem.groups,
            problem.channels,
            edge_values.dtype == torch.float64,
            plan.block_outputs,
            plan.block_channels,
            num_warps=plan.num_warps,
            grid=plan.grid,
            warmup=False,
        )
        return out


def _launch_edge_dot_node_backward(
    grad_out: Tensor,
    edge_values: Tensor,
    node_values: Tensor,
    dst_index: Tensor,
    need_grad_edge_values: bool,
    need_grad_node_values: bool,
    *,
    plan: EdgeDotPlan | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    with cuda_device_context(edge_values.device):
        edge_values = edge_values.resolve_neg().contiguous()
        node_values = node_values.resolve_neg().contiguous()
        dst_index = dst_index.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg()
        if plan is None:
            plan = EDGE_DOT_SCHEDULE.resolve(_bind_edge_dot_problem(edge_values, node_values))
        problem = plan.problem
        num_nodes = problem.nodes
        if not need_grad_edge_values and not need_grad_node_values:
            return None, None
        if num_nodes == 0:
            return (
                torch.zeros_like(edge_values) if need_grad_edge_values else None,
                torch.zeros_like(node_values) if need_grad_node_values else None,
            )

        deterministic_node_vjp = need_grad_node_values and torch.are_deterministic_algorithms_enabled()
        deterministic_plan: SegmentPlan | None = None
        if deterministic_node_vjp:
            try:
                deterministic_plan = CSR_SUM_SCHEDULE.resolve(
                    SegmentProblem(
                        out_segments=num_nodes,
                        features=problem.groups * problem.channels,
                    )
                )
            except ValueError as error:
                rejection = unsupported(
                    "edge_dot_node backward",
                    _DETERMINISTIC_EDGE_DOT_NODE_VJP_DOMAIN,
                    edge_values_shape=tuple(edge_values.shape),
                    node_values_shape=tuple(node_values.shape),
                )
                if torch.is_deterministic_algorithms_warn_only_enabled():
                    warnings.warn(str(rejection), UserWarning, stacklevel=3)
                    deterministic_node_vjp = False
                else:
                    raise rejection from error

        grad_edge_values = torch.empty_like(edge_values) if need_grad_edge_values else None
        atomic_node_vjp = need_grad_node_values and not deterministic_node_vjp
        node_accumulate_fp32 = node_values.dtype in HALF_DTYPES
        grad_node_accum = None
        if atomic_node_vjp:
            accumulation_dtype = torch.float32 if node_accumulate_fp32 else node_values.dtype
            grad_node_accum = torch.zeros_like(node_values, dtype=accumulation_dtype)

        contiguous_grad_out = grad_out.contiguous()
        if need_grad_edge_values or atomic_node_vjp:
            groups = problem.groups
            channels = problem.channels
            # Disabled pointer arguments still require a tensor placeholder.
            grad_edge_ptr = edge_values if grad_edge_values is None else grad_edge_values
            grad_node_ptr = node_values if grad_node_accum is None else grad_node_accum
            backward_warps = plan.num_warps
            live_tiles = max(2, groups - 1) if plan.block_outputs % groups else 2
            register_cells = plan.block_outputs * plan.block_channels * live_tiles
            aggregate_local_nodes = (
                atomic_node_vjp
                and edge_values.dtype != torch.float64
                and problem.edges * groups >= 4096
                and plan.block_outputs >= 2 * groups
                and (plan.block_outputs % groups == 0 or groups <= 8)
                and register_cells <= 8192
            )
            if aggregate_local_nodes and plan.block_outputs * plan.block_channels >= 2048:
                backward_warps = max(backward_warps, 2)
            kernel._edge_dot_node_backward.run(
                contiguous_grad_out,
                edge_values,
                node_values,
                dst_index,
                grad_edge_ptr,
                grad_node_ptr,
                problem.edges * groups,
                num_nodes,
                groups,
                channels,
                need_grad_edge_values,
                atomic_node_vjp,
                node_accumulate_fp32,
                plan.block_outputs,
                plan.block_channels,
                aggregate_local_nodes,
                num_warps=backward_warps,
                grid=plan.grid,
                warmup=False,
            )
        grad_node_values = None
        if deterministic_node_vjp:
            assert deterministic_plan is not None
            grad_node_values = _deterministic_node_values_vjp(
                contiguous_grad_out,
                edge_values,
                node_values,
                dst_index,
                deterministic_plan,
            )
        elif grad_node_accum is not None:
            grad_node_values = grad_node_accum.to(node_values.dtype) if node_accumulate_fp32 else grad_node_accum
        return grad_edge_values, grad_node_values


def _deterministic_node_values_vjp(
    grad_out: Tensor,
    edge_values: Tensor,
    node_values: Tensor,
    dst_index: Tensor,
    plan: SegmentPlan,
) -> Tensor:

    num_nodes = int(node_values.shape[0])
    order, boundaries = stable_destination_ordering(dst_index, num_nodes)
    accumulation_dtype = torch.float32 if edge_values.dtype in HALF_DTYPES else edge_values.dtype
    sorted_edges = edge_values.index_select(0, order)
    sorted_grad = grad_out.index_select(0, order).to(accumulation_dtype)
    contributions = sorted_grad.unsqueeze(-1) * sorted_edges
    reduced = sum_csr(contributions, boundaries, num_nodes, num_nodes, plan=plan)
    return reduced.to(node_values.dtype) if reduced.dtype != node_values.dtype else reduced
