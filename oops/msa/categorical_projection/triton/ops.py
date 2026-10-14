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

from oops.runtime import autograd_function_active, functorch_transforms_active, is_compiling, is_external_fake
from oops.runtime import needs_grad as tensors_need_grad

from ..precision import require_categorical_msa_projection_native_matmul_state
from .generic import (
    _categorical_msa_projection_backward_impl,
    _run_categorical_msa_projection,
    prepare_eager_categorical_msa_projection,
)
from .schedule import CategoricalProjectionPlan

_CATEGORICAL_MSA_PROJECTION_SCHEMA = (
    "(Tensor tokens, Tensor continuous, Tensor weight, int native_matmul_state, "
    "bool guard_native_matmul_state) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_categorical_msa_projection_forward",
    mutates_args=(),
    device_types="cuda",
    schema=_CATEGORICAL_MSA_PROJECTION_SCHEMA,
)
def _categorical_msa_projection_forward(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    if guard_native_matmul_state:
        require_categorical_msa_projection_native_matmul_state(native_matmul_state)
    return _run_categorical_msa_projection(tokens, continuous, weight)


@torch.library.register_fake(_categorical_msa_projection_forward)
def _categorical_msa_projection_forward_fake(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    del native_matmul_state, guard_native_matmul_state
    return torch.empty((*tokens.shape, weight.shape[0]), device=continuous.device, dtype=continuous.dtype)


def _categorical_msa_projection_forward_setup_context(ctx, inputs, output) -> None:
    del output
    tokens, continuous, weight, native_matmul_state, guard_native_matmul_state = inputs
    ctx.save_for_backward(tokens, continuous, weight)
    ctx.native_matmul_state = native_matmul_state
    ctx.compiled_forward = guard_native_matmul_state
    ctx.set_materialize_grads(False)


def _categorical_msa_projection_forward_autograd_backward(ctx, grad_output: Tensor | None):
    if grad_output is None:
        return None, None, None, None, None
    tokens, continuous, weight = ctx.saved_tensors
    need_continuous = ctx.needs_input_grad[1]
    need_weight = ctx.needs_input_grad[2]
    if ctx.compiled_forward:
        grad_continuous, grad_weight = _categorical_msa_projection_backward(
            grad_output,
            tokens,
            continuous,
            weight,
            need_continuous,
            need_weight,
            ctx.native_matmul_state,
        )
    else:
        require_categorical_msa_projection_native_matmul_state(ctx.native_matmul_state)
        grad_continuous, grad_weight = _categorical_msa_projection_backward_impl(
            grad_output,
            tokens,
            continuous,
            weight,
            need_continuous,
            need_weight,
        )
    return None, grad_continuous, grad_weight, None, None


_CATEGORICAL_MSA_PROJECTION_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor tokens, Tensor continuous, Tensor weight,"
    " bool need_continuous, bool need_weight, int native_matmul_state) -> (Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_categorical_msa_projection_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_CATEGORICAL_MSA_PROJECTION_BACKWARD_SCHEMA,
)
def _categorical_msa_projection_backward(
    grad_output: Tensor,
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
    need_continuous: bool,
    need_weight: bool,
    native_matmul_state: int,
) -> tuple[Tensor | None, Tensor | None]:
    require_categorical_msa_projection_native_matmul_state(native_matmul_state)
    return _categorical_msa_projection_backward_impl(
        grad_output,
        tokens,
        continuous,
        weight,
        need_continuous,
        need_weight,
    )


@torch.library.register_fake(_categorical_msa_projection_backward)
def _categorical_msa_projection_backward_fake(
    grad_output: Tensor,
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
    need_continuous: bool,
    need_weight: bool,
    native_matmul_state: int,
) -> tuple[Tensor | None, Tensor | None]:
    del grad_output, tokens, native_matmul_state
    return (
        torch.empty_like(continuous) if need_continuous else None,
        torch.empty_like(weight) if need_weight else None,
    )


torch.library.register_autograd(
    "ooperators::_categorical_msa_projection_forward",
    _categorical_msa_projection_forward_autograd_backward,
    setup_context=_categorical_msa_projection_forward_setup_context,
)


class _FunctionalCategoricalMsaProjection(torch.autograd.Function):

    @staticmethod
    def forward(
        tokens: Tensor,
        continuous: Tensor,
        weight: Tensor,
        native_matmul_state: int,
        plan: CategoricalProjectionPlan,
    ) -> Tensor:
        return _run_categorical_msa_projection(tokens, continuous, weight, plan)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        tokens, continuous, weight, native_matmul_state, _plan = inputs
        del output
        ctx.save_for_backward(tokens, continuous, weight)
        ctx.native_matmul_state = native_matmul_state
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output: Tensor | None):
        if grad_output is None:
            return None, None, None, None, None
        tokens, continuous, weight = ctx.saved_tensors
        require_categorical_msa_projection_native_matmul_state(ctx.native_matmul_state)
        grad_continuous, grad_weight = _categorical_msa_projection_backward_impl(
            grad_output,
            tokens,
            continuous,
            weight,
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        return None, grad_continuous, grad_weight, None, None


class _EagerCategoricalMsaProjection(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        tokens: Tensor,
        continuous: Tensor,
        weight: Tensor,
        native_matmul_state: int,
        plan: CategoricalProjectionPlan,
    ) -> Tensor:
        output = _run_categorical_msa_projection(tokens, continuous, weight, plan)
        ctx.save_for_backward(tokens, continuous, weight)
        ctx.native_matmul_state = native_matmul_state
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor | None):
        if grad_output is None:
            return None, None, None, None, None
        tokens, continuous, weight = ctx.saved_tensors
        require_categorical_msa_projection_native_matmul_state(ctx.native_matmul_state)
        grad_continuous, grad_weight = _categorical_msa_projection_backward_impl(
            grad_output,
            tokens,
            continuous,
            weight,
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        return None, grad_continuous, grad_weight, None, None


def triton_categorical_msa_projection(
    tokens: Tensor,
    continuous: Tensor,
    weight: Tensor,
    native_matmul_state: int,
    plan: CategoricalProjectionPlan | None = None,
) -> Tensor:

    if plan is None:
        compiling = is_compiling()
        fake = False if compiling else is_external_fake(tokens)
        if compiling or fake:
            return _categorical_msa_projection_forward(tokens, continuous, weight, native_matmul_state, compiling)
        plan = prepare_eager_categorical_msa_projection(tokens, continuous, weight)
        if plan is None:
            raise RuntimeError("categorical projection execution plan disappeared after admission")
    transforms_active = functorch_transforms_active()
    if transforms_active:
        if autograd_function_active(continuous, weight, transforms_active=True):
            return _FunctionalCategoricalMsaProjection.apply(tokens, continuous, weight, native_matmul_state, plan)
        return _categorical_msa_projection_forward(tokens, continuous, weight, native_matmul_state, False)
    if tensors_need_grad(continuous, weight):
        return _EagerCategoricalMsaProjection.apply(tokens, continuous, weight, native_matmul_state, plan)
    return _run_categorical_msa_projection(tokens, continuous, weight, plan)
