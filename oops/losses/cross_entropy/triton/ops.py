from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import (
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps_if_needed,
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
)
from oops.runtime import needs_grad as tensors_need_grad
from oops.runtime import tensors_share_device

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


def _autocast_loss_output_is_float32(logits: Tensor) -> bool:

    return logits.is_cuda and logits.dtype in HALF_DTYPES and torch.is_autocast_enabled("cuda")


def triton_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:

    output_shape = labels.shape
    logits = logits.contiguous()
    labels = labels.contiguous()
    output_float32 = _autocast_loss_output_is_float32(logits)
    if tensors_need_grad(logits) or is_compiling() or is_external_fake(logits):
        output = _cross_entropy_forward(logits, labels, output_float32)
    else:
        output = _run_cross_entropy(logits, labels, output_float32)
    return output.reshape(output_shape)


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


def triton_distogram_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    *,
    eps: float = 1e-5,
    reduction: str | None = "mean",
) -> Tensor:

    logits = logits.contiguous()
    labels = labels.contiguous()
    token_mask = token_mask.contiguous()
    reduction_mode = _reduction_mode(reduction)
    output_float32 = _autocast_loss_output_is_float32(logits)
    if tensors_need_grad(logits, token_mask) or is_compiling() or is_external_fake(logits):
        loss, _denominator, _accum, _ce = _distogram_cross_entropy_forward(
            logits,
            labels,
            token_mask,
            float(eps),
            reduction_mode,
            output_float32,
        )
    else:
        loss, _denominator, _accum, _ce = _run_distogram_cross_entropy(
            logits,
            labels,
            token_mask,
            float(eps),
            reduction_mode,
            output_float32,
        )
    return loss


@torch.library.custom_op("ooperators::_cross_entropy_forward", mutates_args=(), device_types="cuda")
def _cross_entropy_forward(logits: Tensor, labels: Tensor, output_float32: bool) -> Tensor:
    return _run_cross_entropy(logits, labels, output_float32)


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


@torch.library.register_fake(_cross_entropy_forward)
def _cross_entropy_forward_fake(logits: Tensor, labels: Tensor, output_float32: bool) -> Tensor:
    out_dtype = torch.float32 if output_float32 else logits.dtype
    return torch.empty((labels.numel(),), device=logits.device, dtype=out_dtype)


@torch.library.custom_op("ooperators::_cross_entropy_backward", mutates_args=(), device_types="cuda")
def _cross_entropy_backward(grad_out: Tensor, logits: Tensor, labels: Tensor) -> Tensor:
    return _run_cross_entropy_backward(grad_out, logits, labels)


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


@torch.library.register_fake(_cross_entropy_backward)
def _cross_entropy_backward_fake(grad_out: Tensor, logits: Tensor, labels: Tensor) -> Tensor:
    del grad_out, labels
    return torch.empty_like(logits)


def _cross_entropy_forward_setup_context(ctx, inputs: tuple[Tensor, Tensor, bool], output: Tensor) -> None:
    del output
    logits, labels, _output_float32 = inputs
    ctx.save_for_backward(logits, labels)
    ctx.direct_backward = not is_compiling() and not is_external_fake(logits) and not functorch_transforms_active()
    if ctx.direct_backward:
        ctx.backward_plan = CROSS_ENTROPY.resolve(CrossEntropyProblem(labels.numel(), logits.shape[-1]))


def _cross_entropy_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None]:
    logits, labels = ctx.saved_tensors
    if ctx.direct_backward:
        grad_logits = _run_cross_entropy_backward(grad_out, logits, labels, ctx.backward_plan)
    else:
        grad_logits = _cross_entropy_backward(grad_out, logits, labels)
    return grad_logits, None, None


torch.library.register_autograd(
    _cross_entropy_forward,
    _cross_entropy_forward_autograd_backward,
    setup_context=_cross_entropy_forward_setup_context,
)


@torch.library.custom_op("ooperators::_distogram_cross_entropy_forward", mutates_args=(), device_types="cuda")
def _distogram_cross_entropy_forward(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    eps: float,
    reduction_mode: int,
    output_float32: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _run_distogram_cross_entropy(logits, labels, token_mask, eps, reduction_mode, output_float32)


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


@torch.library.register_fake(_distogram_cross_entropy_forward)
def _distogram_cross_entropy_forward_fake(
    logits: Tensor,
    labels: Tensor,
    token_mask: Tensor,
    eps: float,
    reduction_mode: int,
    output_float32: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del labels, token_mask, eps
    batch = logits.shape[0]
    # Modes: 0 per batch, 1 mean, 2 sum.
    loss_shape = (batch,) if reduction_mode == 0 else ()
    loss_dtype = torch.float32 if output_float32 else logits.dtype
    return (
        torch.empty(loss_shape, device=logits.device, dtype=loss_dtype),
        torch.empty((batch,), device=logits.device, dtype=torch.float32),
        torch.empty((batch, 2), device=logits.device, dtype=torch.float32),
        torch.empty(
            (logits.shape[0], logits.shape[1] * logits.shape[2]),
            device=logits.device,
            dtype=torch.float32,
        ),
    )


@torch.library.custom_op(
    "ooperators::_distogram_cross_entropy_backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor grad_out, Tensor logits, Tensor labels, Tensor token_mask, Tensor denominator, Tensor accum, "
    "Tensor ce, int reduction_mode, bool need_logits, bool need_token_mask) -> (Tensor?, Tensor?)",
)
def _distogram_cross_entropy_backward(
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


@torch.library.register_fake(_distogram_cross_entropy_backward)
def _distogram_cross_entropy_backward_fake(
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
    del grad_out, labels, denominator, accum, ce, reduction_mode
    return torch.empty_like(logits) if need_logits else None, torch.empty_like(token_mask) if need_token_mask else None


def _distogram_cross_entropy_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, Tensor, float, int, bool],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    logits, labels, token_mask, _eps, reduction_mode, _output_float32 = inputs
    _loss, denominator, accum, ce = output
    ctx.mark_non_differentiable(denominator, accum, ce)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(logits, labels, token_mask, denominator, accum, ce)
    ctx.reduction_mode = reduction_mode


def _distogram_cross_entropy_forward_autograd_backward(
    ctx,
    grad_out: Tensor,
    grad_denominator: Tensor | None,
    grad_accum: Tensor | None,
    grad_ce: Tensor | None,
) -> tuple[Tensor | None, None, Tensor | None, None, None, None]:
    del grad_denominator, grad_accum, grad_ce
    logits, labels, token_mask, denominator, accum, ce = ctx.saved_tensors
    grad_logits, grad_token_mask = _distogram_cross_entropy_backward(
        grad_out,
        logits,
        labels,
        token_mask,
        denominator,
        accum,
        ce,
        ctx.reduction_mode,
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[2],
    )
    return grad_logits, None, grad_token_mask, None, None, None


torch.library.register_autograd(
    _distogram_cross_entropy_forward,
    _distogram_cross_entropy_forward_autograd_backward,
    setup_context=_distogram_cross_entropy_forward_setup_context,
)


def _distogram_problem(logits: Tensor, reduction_mode: int) -> DistogramCrossEntropyProblem:
    return DistogramCrossEntropyProblem(
        batch=logits.shape[0],
        num_tokens=logits.shape[1],
        num_classes=logits.shape[-1],
        scalar_output=reduction_mode != 0,
        dtype=logits.dtype,
    )


def _reduction_mode(reduction: str | None) -> int:
    if reduction is None:
        return 0
    if reduction == "mean":
        return 1
    if reduction == "sum":
        return 2
    raise ValueError("reduction must be 'mean', 'sum', or None")
