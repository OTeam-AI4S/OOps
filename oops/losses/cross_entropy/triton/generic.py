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

from oops.runtime import STANDARD_FLOAT_DTYPES, cuda_device_caps_if_needed, cuda_device_context, tensors_share_device

from . import kernel
from .profiles import (
    CROSS_ENTROPY,
    DISTOGRAM_CROSS_ENTROPY,
    cross_entropy_forward_target_changes_plan,
    distogram_backward_target_changes_plan,
    distogram_forward_target_changes_plan,
    resolve_cross_entropy_forward,
    resolve_distogram_backward,
    resolve_distogram_forward,
)
from .schedule import CrossEntropyPlan, CrossEntropyProblem, DistogramCrossEntropyPlan, DistogramCrossEntropyProblem

_SUPPORTED_LOGIT_DTYPES = STANDARD_FLOAT_DTYPES

_SUPPORTED_LABEL_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)

_SUPPORTED_MASK_DTYPES = (torch.bool, *STANDARD_FLOAT_DTYPES)


def supports_cross_entropy(logits: Tensor, labels: Tensor) -> bool:

    supported = (
        logits.is_cuda
        and labels.is_cuda
        and tensors_share_device(logits, labels)
        and logits.ndim > 0
        and labels.shape == logits.shape[:-1]
        and logits.dtype in _SUPPORTED_LOGIT_DTYPES
        and labels.dtype in _SUPPORTED_LABEL_DTYPES
        and logits.shape[-1] > 0
        and logits.numel() > 0
    )
    return supported and CROSS_ENTROPY.supports(CrossEntropyProblem(labels.numel(), logits.shape[-1]))


def supports_distogram_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    reduction: str | None,
) -> bool:

    supported = (
        logits.is_cuda
        and labels.is_cuda
        and token_mask.is_cuda
        and tensors_share_device(logits, labels, token_mask)
        and logits.ndim == 4
        and labels.shape == logits.shape[:-1]
        and token_mask.shape == (logits.shape[0], logits.shape[1])
        and logits.shape[1] == logits.shape[2]
        and logits.dtype in _SUPPORTED_LOGIT_DTYPES
        and labels.dtype in _SUPPORTED_LABEL_DTYPES
        and token_mask.dtype in _SUPPORTED_MASK_DTYPES
        and logits.shape[-1] > 0
        and logits.numel() > 0
        and reduction in ("mean", "sum", None)
    )
    return supported and DISTOGRAM_CROSS_ENTROPY.supports(
        DistogramCrossEntropyProblem(
            logits.shape[0],
            logits.shape[1],
            logits.shape[-1],
            scalar_output=reduction is not None,
        )
    )


def _run_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    output_float32: bool,
    plan: CrossEntropyPlan | None = None,
) -> Tensor:
    with cuda_device_context(logits.device):
        logits = logits.resolve_neg()
        labels = labels.resolve_neg()
        problem = CrossEntropyProblem(labels.numel(), logits.shape[-1], logits.dtype)
        if plan is None:
            target = cuda_device_caps_if_needed(logits.device, cross_entropy_forward_target_changes_plan(problem))
            plan = resolve_cross_entropy_forward(problem, target=target)
        rows = problem.rows
        num_classes = problem.num_classes
        out_dtype = torch.float32 if output_float32 else logits.dtype
        out = torch.empty((rows,), device=logits.device, dtype=out_dtype)
        kernel._cross_entropy_forward[plan.grid](
            logits,
            labels,
            out,
            rows,
            num_classes,
            plan.block_rows,
            plan.block_classes,
            num_warps=plan.num_warps,
        )
        return out


def _run_cross_entropy_backward(
    grad_out: Tensor,
    logits: Tensor,
    labels: Tensor,
    plan: CrossEntropyPlan | None = None,
) -> Tensor:
    with cuda_device_context(logits.device):
        logits = logits.resolve_neg().contiguous()
        labels = labels.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        problem = CrossEntropyProblem(labels.numel(), logits.shape[-1])
        plan = CROSS_ENTROPY.resolve(problem) if plan is None else plan
        rows = problem.rows
        num_classes = problem.num_classes
        grad_logits = torch.empty_like(logits)
        kernel._cross_entropy_backward[plan.grid](
            logits,
            labels,
            grad_out,
            grad_logits,
            rows,
            num_classes,
            plan.block_rows,
            plan.block_classes,
            num_warps=plan.num_warps,
        )
        return grad_logits


def _run_distogram_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    eps: float,
    reduction_mode: int,
    output_float32: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with cuda_device_context(logits.device):
        logits = logits.resolve_neg()
        labels = labels.resolve_neg()
        token_mask = token_mask.resolve_neg()
        problem = _distogram_problem(logits, reduction_mode)
        problem = DistogramCrossEntropyProblem(
            problem.batch, problem.num_tokens, problem.num_classes, problem.scalar_output, logits.dtype
        )
        target = cuda_device_caps_if_needed(logits.device, distogram_forward_target_changes_plan(problem))
        plan = resolve_distogram_forward(problem, target=target)
        batch = problem.batch
        num_tokens = problem.num_tokens
        num_classes = problem.num_classes
        deterministic = torch.are_deterministic_algorithms_enabled()
        accum = (
            torch.empty((batch, 2), device=logits.device, dtype=torch.float32)
            if deterministic
            else torch.zeros((batch, 2), device=logits.device, dtype=torch.float32)
        )
        denominator = torch.empty((batch,), device=logits.device, dtype=torch.float32)
        out_dtype = torch.float32 if output_float32 else logits.dtype
        out = (
            torch.empty((batch,), device=logits.device, dtype=out_dtype)
            if reduction_mode == 0
            else torch.empty((), device=logits.device, dtype=out_dtype)
        )
        finalize_plan = plan.finalize_plan
        rows_per_batch = problem.rows_per_batch
        ce = torch.empty((problem.batch, problem.rows_per_batch), device=logits.device, dtype=torch.float32)
        if deterministic:
            _accumulate_distogram_row_partials(plan, problem, logits, labels, token_mask, accum, ce)
        else:
            kernel._distogram_forward_accumulate[plan.accumulate_grid](
                logits,
                labels,
                token_mask,
                accum,
                rows_per_batch,
                num_tokens,
                num_classes,
                plan.wide_offsets,
                plan.block_rows,
                plan.block_classes,
                ce_ptr=ce,
                num_warps=plan.accumulate_num_warps,
            )
        single_partial = finalize_plan.scalar_output and finalize_plan.num_partials == 1
        partials = (
            torch.empty((finalize_plan.num_partials,), device=logits.device, dtype=torch.float32)
            if finalize_plan.scalar_output and not single_partial
            else out
        )
        kernel._distogram_finalize_tiles[finalize_plan.tile_grid](
            accum,
            denominator,
            out,
            partials,
            batch,
            eps,
            reduction_mode,
            single_partial,
            finalize_plan.block_items,
            num_warps=finalize_plan.tile_num_warps,
        )
        if finalize_plan.scalar_output and not single_partial:
            kernel._distogram_finalize_partials[finalize_plan.finalize_grid](
                partials,
                out,
                batch,
                finalize_plan.num_partials,
                reduction_mode,
                finalize_plan.block_partials,
                num_warps=finalize_plan.partials_num_warps,
            )
        # Saved numerator matches the forward loss used by the mask VJP.
        return out, denominator, accum, ce


def _accumulate_distogram_row_partials(
    plan: DistogramCrossEntropyPlan,
    problem: DistogramCrossEntropyProblem,
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    accum: Tensor,
    ce: Tensor,
) -> None:

    row_parts = plan.deterministic_row_parts
    single_part = row_parts == 1
    accumulator_items = accum.numel()
    partials = (
        accum if single_part else torch.empty((row_parts, *accum.shape), device=accum.device, dtype=torch.float32)
    )
    kernel._distogram_forward_partials[plan.deterministic_accumulate_grid](
        logits,
        labels,
        token_mask,
        partials,
        plan.accumulate_grid[0],
        0 if single_part else accumulator_items,
        problem.rows_per_batch,
        problem.num_tokens,
        problem.num_classes,
        row_parts,
        plan.wide_offsets,
        plan.block_rows,
        plan.block_classes,
        ce_ptr=ce,
        num_warps=plan.accumulate_num_warps,
    )
    if single_part:
        return
    kernel._reduce_distogram_partials[plan.deterministic_reduce_grid](
        partials,
        accum,
        accumulator_items,
        row_parts,
        plan.deterministic_reduce_block_items,
        num_warps=plan.deterministic_reduce_num_warps,
    )


def _run_distogram_cross_entropy_backward(
    grad_out: Tensor,
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    denominator: Tensor,
    accum: Tensor,
    ce: Tensor,
    reduction_mode: int,
    need_logits: bool,
    need_token_mask: bool,
) -> tuple[Tensor | None, Tensor | None]:
    with cuda_device_context(logits.device):
        logits = logits.resolve_neg().contiguous()
        labels = labels.resolve_neg().contiguous()
        token_mask = token_mask.resolve_neg().contiguous()
        denominator = denominator.resolve_neg().contiguous()
        accum = accum.resolve_neg().contiguous()
        ce = ce.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg().contiguous()
        problem = _distogram_problem(logits, reduction_mode)
        target = cuda_device_caps_if_needed(logits.device, distogram_backward_target_changes_plan(problem))
        plan = resolve_distogram_backward(problem, target=target)
        grad_logits = torch.empty_like(logits) if need_logits else None
        grad_token_mask = torch.empty_like(token_mask) if need_token_mask else None
        if grad_logits is not None:
            kernel._distogram_backward[plan.accumulate_grid](
                logits,
                labels,
                token_mask,
                denominator,
                grad_out,
                grad_logits,
                problem.rows_per_batch,
                problem.num_tokens,
                problem.num_classes,
                problem.batch,
                reduction_mode,
                plan.wide_offsets,
                plan.block_rows,
                plan.block_classes,
                num_warps=plan.accumulate_num_warps,
            )
        if grad_token_mask is not None:
            kernel._distogram_token_mask_backward[plan.token_mask_backward_grid](
                ce,
                token_mask,
                denominator,
                accum,
                grad_out,
                grad_token_mask,
                problem.rows_per_batch,
                problem.num_tokens,
                problem.batch,
                reduction_mode,
                plan.wide_offsets,
                256,
                num_warps=plan.token_mask_num_warps,
            )
        return grad_logits, grad_token_mask


def _distogram_problem(logits: Tensor, reduction_mode: int) -> DistogramCrossEntropyProblem:
    return DistogramCrossEntropyProblem(
        batch=logits.shape[0],
        num_tokens=logits.shape[1],
        num_classes=logits.shape[-1],
        scalar_output=reduction_mode != 0,
        dtype=logits.dtype,
    )
