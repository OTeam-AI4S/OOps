from __future__ import annotations

import torch
import triton
from torch import Tensor

from oops.runtime import (
    STANDARD_FLOAT_DTYPES,
    autograd_function_active,
    cuda_device_caps_if_needed,
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
)
from oops.runtime import needs_grad as tensors_need_grad

from ..precision import require_categorical_msa_projection_native_matmul_state
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
