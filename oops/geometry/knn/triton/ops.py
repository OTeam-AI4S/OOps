from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import (
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps_if_needed,
    cuda_device_context,
    is_compiling,
    is_external_fake,
    needs_grad,
    tensors_share_device,
)

from . import kernel
from .profiles import MASKED_KNN
from .schedule import MaskedKnnBackwardPlan, MaskedKnnPlan, MaskedKnnProblem

_SUPPORTED_MASK_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    *STANDARD_FLOAT_DTYPES,
)


def _problem(coordinates: Tensor, offsets: Tensor, num_neighbors: int) -> MaskedKnnProblem:
    return MaskedKnnProblem(
        total_points=coordinates.shape[0],
        num_samples=offsets.shape[0] - 1,
        num_neighbors=num_neighbors,
        dtype=coordinates.dtype,
    )


def supports_segmented_masked_knn(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
) -> bool:
    supported = (
        coordinates.is_cuda
        and mask.is_cuda
        and offsets.is_cuda
        and batch_indices.is_cuda
        and tensors_share_device(coordinates, mask, offsets, batch_indices)
        and coordinates.dtype in STANDARD_FLOAT_DTYPES
        and mask.dtype in _SUPPORTED_MASK_DTYPES
        and offsets.dtype is torch.int64
        and batch_indices.dtype in (torch.int32, torch.int64)
        and coordinates.dim() == 2
        and coordinates.shape[1] == 3
        and mask.dim() == 1
        and offsets.dim() == 1
        and offsets.shape[0] > 1
        and batch_indices.dim() == 1
        and coordinates.is_contiguous()
        and mask.is_contiguous()
        and offsets.is_contiguous()
        and batch_indices.is_contiguous()
    )
    if not supported:
        return False
    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(coordinates)
    if compiling or external_fake:
        torch._check(mask.shape[0] == coordinates.shape[0])
        torch._check(batch_indices.shape[0] == coordinates.shape[0])
    elif mask.shape[0] != coordinates.shape[0] or batch_indices.shape[0] != coordinates.shape[0]:
        return False
    return MASKED_KNN.supports(_problem(coordinates, offsets, num_neighbors))


def segmented_masked_knn(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(coordinates)
    if needs_grad(coordinates) or compiling or external_fake:
        return _segmented_masked_knn_forward(
            coordinates,
            mask,
            offsets,
            batch_indices,
            num_neighbors,
            eps,
        )
    return _run_segmented_masked_knn(
        coordinates,
        mask,
        offsets,
        batch_indices,
        num_neighbors,
        eps,
    )


@torch.library.custom_op("ooperators::_segmented_masked_knn_forward", mutates_args=(), device_types="cuda")
def _segmented_masked_knn_forward(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return _run_segmented_masked_knn(
        coordinates,
        mask,
        offsets,
        batch_indices,
        num_neighbors,
        eps,
    )


@torch.library.register_fake(_segmented_masked_knn_forward)
def _segmented_masked_knn_forward_fake(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    del mask, offsets, batch_indices, eps
    shape = (coordinates.shape[0], num_neighbors)
    return (
        torch.empty(shape, device=coordinates.device, dtype=coordinates.dtype),
        torch.empty(shape, device=coordinates.device, dtype=torch.long),
        torch.empty(shape, device=coordinates.device, dtype=torch.bool),
    )


def _segmented_masked_knn_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, int, float],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    coordinates, _mask, offsets, batch_indices, num_neighbors, eps = inputs
    _distances, local_indices, valid = output
    ctx.mark_non_differentiable(local_indices, valid)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(coordinates, offsets, batch_indices, local_indices, valid)
    ctx.num_neighbors = num_neighbors
    ctx.eps = eps


def _segmented_masked_knn_forward_autograd_backward(
    ctx,
    grad_distances: Tensor | None,
    grad_local_indices: Tensor | None,
    grad_valid: Tensor | None,
) -> tuple[Tensor | None, None, None, None, None, None]:
    del grad_local_indices, grad_valid
    grad_coordinates = None
    if ctx.needs_input_grad[0] and grad_distances is not None:
        coordinates, offsets, batch_indices, local_indices, valid = ctx.saved_tensors
        grad_coordinates = _segmented_masked_knn_backward(
            grad_distances,
            coordinates,
            offsets,
            batch_indices,
            local_indices,
            valid,
            ctx.num_neighbors,
            ctx.eps,
        )
    return grad_coordinates, None, None, None, None, None


torch.library.register_autograd(
    _segmented_masked_knn_forward,
    _segmented_masked_knn_forward_autograd_backward,
    setup_context=_segmented_masked_knn_forward_setup_context,
)


def _run_segmented_masked_knn(
    coordinates: Tensor,
    mask: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    num_neighbors: int,
    eps: float,
    plan: MaskedKnnPlan | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    with cuda_device_context(coordinates.device):
        coordinates = coordinates.resolve_neg().contiguous()
        mask = mask.resolve_neg().contiguous()
        offsets = offsets.resolve_neg().contiguous()
        batch_indices = batch_indices.resolve_neg().contiguous()
        problem = _problem(coordinates, offsets, num_neighbors)
        if plan is None:
            target = cuda_device_caps_if_needed(
                coordinates.device,
                MASKED_KNN.target_changes_plan(problem),
            )
            plan = MASKED_KNN.resolve(problem, target=target)
        output_shape = (problem.total_points, problem.num_neighbors)
        distances = torch.empty(output_shape, device=coordinates.device, dtype=coordinates.dtype)
        local_indices = torch.empty(output_shape, device=coordinates.device, dtype=torch.long)
        valid = torch.empty(output_shape, device=coordinates.device, dtype=torch.bool)
        if plan.grid is None:
            return distances, local_indices, valid
        kernel._segmented_masked_knn_forward[plan.grid](
            coordinates,
            mask,
            offsets,
            batch_indices,
            distances,
            local_indices,
            valid,
            eps,
            NUM_NEIGHBORS=problem.num_neighbors,
            TOPK=plan.topk,
            BLOCK_KEY=plan.config.block_key,
            num_warps=plan.config.num_warps,
        )
        return distances, local_indices, valid


def _target_inversion(
    offsets: Tensor,
    batch_indices: Tensor,
    local_indices: Tensor,
    valid: Tensor,
    num_points: int,
) -> tuple[Tensor, Tensor]:

    sample_starts = offsets.index_select(0, batch_indices)
    target = sample_starts.unsqueeze(1) + local_indices
    target = torch.where(valid, target, torch.full_like(target, num_points)).reshape(-1)
    sorted_target, target_order = torch.sort(target, stable=True)
    target_boundary_keys = torch.arange(num_points + 1, device=target.device, dtype=target.dtype)
    target_boundaries = torch.searchsorted(sorted_target, target_boundary_keys)
    return target_order.contiguous(), target_boundaries.contiguous()


def _atomic_backward(
    grad_distances: Tensor,
    coordinates: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    local_indices: Tensor,
    valid: Tensor,
    num_neighbors: int,
    eps: float,
    plan: MaskedKnnBackwardPlan,
) -> Tensor:
    num_points = coordinates.shape[0]
    num_edges = num_points * num_neighbors
    grad_accum = torch.zeros(coordinates.shape, dtype=torch.float32, device=coordinates.device)
    assert plan.atomic_grid is not None
    kernel._segmented_masked_knn_atomic_backward[plan.atomic_grid](
        grad_distances,
        coordinates,
        offsets,
        batch_indices,
        local_indices,
        valid,
        grad_accum,
        num_edges,
        eps,
        NUM_NEIGHBORS=num_neighbors,
        BLOCK_EDGES=plan.config.atomic_block_edges,
        num_warps=plan.config.num_warps,
    )
    return grad_accum.to(dtype=coordinates.dtype)


def _deterministic_backward(
    grad_distances: Tensor,
    coordinates: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    local_indices: Tensor,
    valid: Tensor,
    num_neighbors: int,
    eps: float,
    plan: MaskedKnnBackwardPlan,
) -> Tensor:
    num_points = coordinates.shape[0]
    target_order, target_boundaries = _target_inversion(
        offsets,
        batch_indices,
        local_indices,
        valid,
        num_points,
    )
    grad_accum = torch.empty(coordinates.shape, dtype=torch.float32, device=coordinates.device)
    assert plan.deterministic_grid is not None
    kernel._segmented_masked_knn_deterministic_backward[plan.deterministic_grid](
        grad_distances,
        coordinates,
        offsets,
        batch_indices,
        local_indices,
        valid,
        target_order,
        target_boundaries,
        grad_accum,
        num_points,
        eps,
        NUM_NEIGHBORS=num_neighbors,
        BLOCK_QUERY_EDGES=plan.topk,
        BLOCK_TARGET_EDGES=plan.config.deterministic_target_block,
        num_warps=plan.config.num_warps,
    )
    return grad_accum.to(dtype=coordinates.dtype)


@torch.library.custom_op(
    "ooperators::_segmented_masked_knn_backward",
    mutates_args=(),
    device_types="cuda",
)
def _segmented_masked_knn_backward(
    grad_distances: Tensor,
    coordinates: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    local_indices: Tensor,
    valid: Tensor,
    num_neighbors: int,
    eps: float,
) -> Tensor:

    with cuda_device_context(coordinates.device):
        coordinates = coordinates.resolve_neg().contiguous()
        offsets = offsets.resolve_neg().contiguous()
        batch_indices = batch_indices.resolve_neg().contiguous()
        local_indices = local_indices.resolve_neg().contiguous()
        valid = valid.resolve_neg().contiguous()
        grad_distances = grad_distances.resolve_neg()
        if coordinates.shape[0] == 0:
            return torch.empty_like(coordinates)
        contiguous_grad = grad_distances.contiguous()
        problem = MaskedKnnProblem(
            total_points=coordinates.shape[0],
            num_samples=offsets.shape[0] - 1,
            num_neighbors=num_neighbors,
            dtype=coordinates.dtype,
        )
        target = cuda_device_caps_if_needed(
            coordinates.device,
            MASKED_KNN.backward_target_changes_plan(problem),
        )
        plan = MASKED_KNN.resolve_backward(problem, target=target)
        if torch.are_deterministic_algorithms_enabled():
            return _deterministic_backward(
                contiguous_grad,
                coordinates,
                offsets,
                batch_indices,
                local_indices,
                valid,
                num_neighbors,
                eps,
                plan,
            )
        return _atomic_backward(
            contiguous_grad,
            coordinates,
            offsets,
            batch_indices,
            local_indices,
            valid,
            num_neighbors,
            eps,
            plan,
        )


@torch.library.register_fake(_segmented_masked_knn_backward)
def _segmented_masked_knn_backward_fake(
    grad_distances: Tensor,
    coordinates: Tensor,
    offsets: Tensor,
    batch_indices: Tensor,
    local_indices: Tensor,
    valid: Tensor,
    num_neighbors: int,
    eps: float,
) -> Tensor:
    del grad_distances, offsets, batch_indices, local_indices, valid, num_neighbors, eps
    return torch.empty_like(coordinates)
