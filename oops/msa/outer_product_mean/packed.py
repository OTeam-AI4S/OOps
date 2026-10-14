from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import cuda_device_context, needs_grad, require_native_matmul_precision_state

from .composed import _execute_contraction, _execute_contraction_backward, _plain


def _topology(left, mask, sizes, offsets):
    if sizes.device.type != "cpu" or sizes.dtype is not torch.int64 or sizes.dim() != 2 or sizes.shape[1] != 3:
        raise ValueError("outer_product_mean requires CPU int64 MSA element sizes")
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError("outer_product_mean requires CPU int64 pair offsets")
    topology = sizes.tolist()
    pairs = offsets.tolist()
    expected = [0]
    for depth, tokens, channels in topology:
        if depth < 0 or tokens < 0 or channels <= 0:
            raise ValueError("outer_product_mean MSA element sizes disagree with the public domain")
        expected.append(expected[-1] + tokens * tokens)
    if pairs != expected or sum(m * n for m, n, _ in topology) != left.shape[0] or mask.shape[0] != left.shape[0]:
        raise ValueError("outer_product_mean packed rows disagree with element sizes and pair offsets")
    return topology, expected[-1]


def _execute_forward(
    left,
    right,
    mask,
    weight,
    bias,
    pair_offsets_cpu,
    msa_sizes_cpu,
    input_precision,
    state,
    divide_before,
    eps,
    clamp,
    save_state,
):
    with cuda_device_context(left.device), torch.autocast("cuda", enabled=False):
        require_native_matmul_precision_state("outer_product_mean", state)
        topology, pairs = _topology(left, mask, msa_sizes_cpu, pair_offsets_cpu)
        output = torch.empty((pairs, weight.shape[0]), device=left.device, dtype=weight.dtype)
        outer = torch.empty((pairs, left.shape[1] ** 2), device=left.device, dtype=left.dtype) if save_state else None
        numerator = torch.empty_like(output) if save_state else None
        counts = torch.empty((pairs, 1), device=left.device, dtype=torch.float32) if save_state else None
        row_start = pair_start = 0
        for depth, tokens, _channels in topology:
            row_stop, pair_stop = row_start + depth * tokens, pair_start + tokens * tokens
            left_sample = left[row_start:row_stop].reshape(1, depth, tokens, left.shape[1])
            right_sample = right[row_start:row_stop].reshape_as(left_sample)
            keep = mask[row_start:row_stop].float().reshape(1, depth, tokens)
            count = torch.einsum("bsi,bsj->bij", keep, keep).unsqueeze(-1)
            result, item_outer, item_numerator = _execute_contraction(
                left_sample,
                right_sample,
                count,
                weight,
                bias,
                input_precision,
                state,
                divide_before,
                eps,
                clamp,
                save_state,
            )
            output[pair_start:pair_stop].copy_(result.reshape(-1, weight.shape[0]))
            if save_state:
                outer[pair_start:pair_stop].copy_(item_outer.reshape(-1, left.shape[1] ** 2))
                numerator[pair_start:pair_stop].copy_(item_numerator.reshape(-1, weight.shape[0]))
                counts[pair_start:pair_stop].copy_(count.reshape(-1, 1))
            row_start, pair_start = row_stop, pair_stop
        return output, outer, numerator, counts


def _execute_backward(
    gradient,
    left,
    right,
    mask,
    weight,
    bias,
    pair_offsets_cpu,
    msa_sizes_cpu,
    outer,
    numerator,
    counts,
    input_precision,
    state,
    divide_before,
    eps,
    clamp,
    need_left,
    need_right,
    need_mask,
    need_weight,
    need_bias,
):
    with cuda_device_context(left.device), torch.autocast("cuda", enabled=False):
        require_native_matmul_precision_state("outer_product_mean", state)
        topology, _ = _topology(left, mask, msa_sizes_cpu, pair_offsets_cpu)
        grad_left = torch.empty_like(left) if need_left else None
        grad_right = torch.empty_like(right) if need_right else None
        grad_mask = torch.empty_like(mask) if need_mask else None
        grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if need_weight else None
        grad_bias = (
            torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if need_bias and bias is not None else None
        )
        row_start = pair_start = 0
        for depth, tokens, _channels in topology:
            row_stop, pair_stop = row_start + depth * tokens, pair_start + tokens * tokens
            left_sample = left[row_start:row_stop].reshape(1, depth, tokens, left.shape[1])
            right_sample = right[row_start:row_stop].reshape_as(left_sample)
            item_outer = outer[pair_start:pair_stop].reshape(1, tokens, left.shape[1], tokens, left.shape[1])
            item_numerator = numerator[pair_start:pair_stop].reshape(1, tokens, tokens, weight.shape[0])
            count = counts[pair_start:pair_stop].reshape(1, tokens, tokens, 1)
            grads = _execute_contraction_backward(
                gradient[pair_start:pair_stop].reshape_as(item_numerator),
                left_sample,
                right_sample,
                count,
                weight,
                bias,
                item_outer,
                item_numerator,
                input_precision,
                state,
                divide_before,
                eps,
                clamp,
                need_left,
                need_right,
                need_mask,
                need_weight,
                need_bias,
            )
            if need_left:
                grad_left[row_start:row_stop].copy_(grads[0].reshape(-1, left.shape[1]))
            if need_right:
                grad_right[row_start:row_stop].copy_(grads[1].reshape(-1, right.shape[1]))
            if need_mask:
                keep = mask[row_start:row_stop].float().reshape(1, depth, tokens)
                incoming = grads[2].squeeze(-1)
                result = torch.einsum("bsj,bij->bsi", keep, incoming) + torch.einsum("bsi,bij->bsj", keep, incoming)
                grad_mask[row_start:row_stop].copy_(result.reshape(-1))
            if need_weight:
                grad_weight.add_(grads[3])
            if grad_bias is not None:
                grad_bias.add_(grads[4])
            row_start, pair_start = row_stop, pair_stop
        return (
            grad_left,
            grad_right,
            grad_mask,
            None if grad_weight is None else grad_weight.to(weight.dtype),
            None if grad_bias is None else grad_bias.to(bias.dtype),
        )


@torch.library.custom_op(
    "ooperators::_segmented_outer_product_mean_forward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor left, Tensor right, Tensor mask, Tensor weight, Tensor? bias, Tensor "
        "pair_offsets_cpu, Tensor msa_sizes_cpu, str input_precision, int state, bool "
        "divide_before, float eps, bool clamp) -> (Tensor, Tensor, Tensor, Tensor)"
    ),
)
def _forward(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _execute_forward(
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        input_precision,
        state,
        divide_before,
        eps,
        clamp,
        True,
    )


@_forward.register_fake
def _forward_fake(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pairs = torch.library.get_ctx().new_dynamic_size(min=0)
    output = torch.empty((pairs, weight.shape[0]), device=left.device, dtype=weight.dtype)
    return (
        output,
        left.new_empty((pairs, left.shape[1] ** 2)),
        torch.empty_like(output),
        torch.empty((pairs, 1), device=left.device, dtype=torch.float32),
    )


@torch.library.custom_op(
    "ooperators::_segmented_outer_product_mean_inference",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor left, Tensor right, Tensor mask, Tensor weight, Tensor? bias, Tensor "
        "pair_offsets_cpu, Tensor msa_sizes_cpu, str input_precision, int state, bool "
        "divide_before, float eps, bool clamp) -> Tensor"
    ),
)
def _inference(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> Tensor:
    return _execute_forward(
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        input_precision,
        state,
        divide_before,
        eps,
        clamp,
        False,
    )[0]


@_inference.register_fake
def _inference_fake(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
) -> Tensor:
    pairs = torch.library.get_ctx().new_dynamic_size(min=0)
    return torch.empty((pairs, weight.shape[0]), device=left.device, dtype=weight.dtype)


@torch.library.custom_op(
    "ooperators::_segmented_outer_product_mean_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        "(Tensor gradient, Tensor left, Tensor right, Tensor mask, Tensor weight, Tensor? "
        "bias, Tensor pair_offsets_cpu, Tensor msa_sizes_cpu, Tensor outer, Tensor numerator, "
        "Tensor counts, str input_precision, int state, bool divide_before, float eps, bool "
        "clamp, bool need_left, bool need_right, bool need_mask, bool need_weight, bool "
        "need_bias) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
    ),
)
def _backward(
    gradient: Tensor,
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    outer: Tensor,
    numerator: Tensor,
    counts: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
    need_left: bool,
    need_right: bool,
    need_mask: bool,
    need_weight: bool,
    need_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return _execute_backward(
        gradient,
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        outer,
        numerator,
        counts,
        input_precision,
        state,
        divide_before,
        eps,
        clamp,
        need_left,
        need_right,
        need_mask,
        need_weight,
        need_bias,
    )


@_backward.register_fake
def _backward_fake(
    gradient: Tensor,
    left: Tensor,
    right: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    pair_offsets_cpu: Tensor,
    msa_sizes_cpu: Tensor,
    outer: Tensor,
    numerator: Tensor,
    counts: Tensor,
    input_precision: str,
    state: int,
    divide_before: bool,
    eps: float,
    clamp: bool,
    need_left: bool,
    need_right: bool,
    need_mask: bool,
    need_weight: bool,
    need_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return (
        torch.empty(left.shape, device=left.device, dtype=left.dtype) if need_left else None,
        torch.empty(right.shape, device=right.device, dtype=right.dtype) if need_right else None,
        torch.empty(mask.shape, device=mask.device, dtype=mask.dtype) if need_mask else None,
        torch.empty(weight.shape, device=weight.device, dtype=weight.dtype) if need_weight else None,
        torch.empty(bias.shape, device=bias.device, dtype=bias.dtype) if need_bias and bias is not None else None,
    )


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:7], *output[1:])
    ctx.constants = inputs[7:]
    ctx.demand = ctx.needs_input_grad[:5]
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)


def _autograd_backward(ctx, gradient, *auxiliary):
    arguments = (gradient, *ctx.saved_tensors, *ctx.constants, *ctx.demand)
    grads = (
        _execute_backward(*arguments)
        if getattr(ctx, "direct", False) and not torch.is_grad_enabled()
        else _backward(*arguments)
    )
    return (*grads, *((None,) * 7))


torch.library.register_autograd(_forward, _autograd_backward, setup_context=_setup_context)


class _SavedContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = _execute_forward(*inputs, True)
        ctx.save_for_backward(*inputs[:7], *output[1:])
        ctx.constants = inputs[7:]
        ctx.demand = ctx.needs_input_grad[:5]
        ctx.set_materialize_grads(False)
        ctx.direct = True
        return output[0]

    @staticmethod
    def backward(ctx, gradient):
        return _autograd_backward(ctx, gradient)


def segmented_outer_product_mean(
    left,
    right,
    mask,
    weight,
    bias,
    msa_outer_offsets,
    msa_packed_offsets,
    pair_offsets,
    pair_offsets_cpu,
    msa_sizes_cpu,
    input_precision,
    projection_precision,
    count_precision,
    native_matmul_state,
    divide_before_projection,
    denominator_eps,
    clamp_denominator,
):
    arguments = (
        left,
        right,
        mask,
        weight,
        bias,
        pair_offsets_cpu,
        msa_sizes_cpu,
        projection_precision,
        native_matmul_state,
        divide_before_projection,
        denominator_eps,
        clamp_denominator,
    )
    if needs_grad(left, right, mask, weight, bias):
        if _plain(*arguments[:7]):
            return _SavedContraction.apply(*arguments)
        return _forward(*arguments)[0]
    if _plain(*arguments[:7]):
        return _execute_forward(*arguments, False)[0]
    return _inference(*arguments)
