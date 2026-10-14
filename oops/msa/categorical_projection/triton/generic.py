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
import triton
from torch import Tensor

from oops.runtime import STANDARD_FLOAT_DTYPES, cuda_device_caps_if_needed, cuda_device_context

from . import kernel
from .profiles import SCHEDULE, resolve_plan, target_changes_plan
from .schedule import CategoricalProjectionPlan, CategoricalProjectionProblem


def supports_categorical_msa_projection(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
) -> bool:

    rows = tokens.numel()
    channels = weight.shape[0]
    supported = (
        tokens.dim() >= 1
        and continuous.shape == (*tokens.shape, 2)
        and weight.dim() == 2
        and weight.shape[1] > 2
        and rows > 0
        and channels > 0
        and tokens.dtype is torch.int64
        and continuous.dtype in STANDARD_FLOAT_DTYPES
        and weight.dtype is continuous.dtype
        and tokens.is_cuda
        and continuous.is_cuda
        and weight.is_cuda
        and tokens.device == continuous.device == weight.device
        and tokens.is_contiguous()
        and continuous.is_contiguous()
        and weight.is_contiguous()
    )
    return supported and SCHEDULE.supports(CategoricalProjectionProblem(rows, channels))


def supports_prepared_categorical_msa_projection(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
) -> bool:

    return _prepared_categorical_msa_projection_problem(tokens, continuous, weight) is not None


def prepare_eager_categorical_msa_projection(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
) -> CategoricalProjectionPlan | None:

    problem = _prepared_categorical_msa_projection_problem(tokens, continuous, weight)
    if problem is None:
        return None
    target = cuda_device_caps_if_needed(tokens.device, target_changes_plan(problem))
    return resolve_plan(problem, target=target)


def _prepared_categorical_msa_projection_problem(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
) -> CategoricalProjectionProblem | None:
    layout_supported = (
        tokens.is_cuda and tokens.is_contiguous() and continuous.is_contiguous() and weight.is_contiguous()
    )
    if not layout_supported:
        return None
    problem = CategoricalProjectionProblem(tokens.numel(), weight.shape[0], continuous.dtype)
    return problem if SCHEDULE.supports(problem) else None


def _run_categorical_msa_projection(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
    plan: CategoricalProjectionPlan | None = None,
) -> Tensor:
    with cuda_device_context(tokens.device):
        continuous = continuous.resolve_neg() if continuous.is_neg() else continuous
        weight = weight.resolve_neg() if weight.is_neg() else weight
        rows = tokens.numel()
        channels = weight.shape[0]
        vocabulary_size = weight.shape[1] - 2
        if plan is None:
            problem = CategoricalProjectionProblem(rows, channels, continuous.dtype)
            target = cuda_device_caps_if_needed(tokens.device, target_changes_plan(problem))
            plan = resolve_plan(problem, target=target)
        output = torch.empty((*tokens.shape, channels), device=continuous.device, dtype=continuous.dtype)
        kernel._categorical_msa_projection[plan.grid](
            tokens,
            continuous,
            weight,
            output,
            rows,
            vocabulary_size,
            channels,
            plan.block_rows,
            plan.block_channels,
            num_warps=plan.num_warps,
        )
        return output


def _categorical_msa_projection_backward_impl(
    grad_output: Tensor,
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
    need_continuous: bool,
    need_weight: bool,
) -> tuple[Tensor | None, Tensor | None]:
    with cuda_device_context(tokens.device):
        vocabulary_size = weight.shape[1] - 2
        rows = tokens.numel()
        channels = weight.shape[0]
        flat_tokens = tokens.reshape(rows)
        flat_continuous = continuous.reshape(rows, 2)
        flat_grad = grad_output.reshape(rows, channels)

        # Backward differentiates prepared operands; ambient autocast must not recast them.
        with torch.autocast(device_type="cuda", enabled=False):
            grad_continuous = None
            if need_continuous:
                grad_continuous = (flat_grad @ weight[:, vocabulary_size:]).reshape_as(continuous)

            grad_weight = None
            if need_weight:
                if rows <= 1024 and channels <= 128 and vocabulary_size <= 128:
                    flat_tokens = flat_tokens.resolve_neg().contiguous()
                    flat_grad = (flat_grad.resolve_neg() if flat_grad.is_neg() else flat_grad).contiguous()
                    flat_continuous = (
                        flat_continuous.resolve_neg() if flat_continuous.is_neg() else flat_continuous
                    ).contiguous()
                    grad_weight = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
                    kernel._categorical_msa_projection_weight_backward[(weight.shape[1], triton.cdiv(channels, 8))](
                        flat_grad,
                        flat_tokens,
                        flat_continuous,
                        grad_weight,
                        rows,
                        channels,
                        weight.shape[1],
                        flat_grad.stride(0),
                        flat_grad.stride(1),
                        triton.next_power_of_2(rows),
                        8,
                        num_warps=4,
                    )
                else:
                    categorical = torch.ops.aten.embedding_dense_backward.default(
                        flat_grad,
                        flat_tokens,
                        vocabulary_size,
                        -1,
                        False,
                    )
                    projected = flat_grad.transpose(0, 1) @ flat_continuous
                    grad_weight = torch.cat((categorical.transpose(0, 1), projected), dim=-1)

        return grad_continuous, grad_weight
