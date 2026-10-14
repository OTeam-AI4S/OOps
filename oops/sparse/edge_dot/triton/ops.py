from __future__ import annotations

import warnings
from functools import lru_cache

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import (
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
)
from oops.runtime import needs_grad as tensors_need_grad
from oops.runtime import tensors_share_device, unsupported

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
