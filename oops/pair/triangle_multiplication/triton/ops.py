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

from typing import cast

import torch
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import guard_or_true
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.normalization.layer_norm.triton import supports_layer_norm
from oops.runtime import (
    FLOAT32_DOT_TF32X3,
    STANDARD_FLOAT_DTYPES,
    captured_native_matmul_precision_state,
    forward_mode_active,
    functorch_transforms_active,
    native_matmul_domain,
    triton_dot_precision,
)

from ..parameters import TriangleMultiplicationBiases, TriangleMultiplicationWeights
from .generic import (
    _contraction_backward,
    _contraction_forward,
    _epilogue_backward,
    _epilogue_forward,
    _inference,
    _input_layer_norm,
    _phases_fit,
    _projection_backward,
    _projection_forward,
)
from .schedule import MAX_FUSED_WIDTH

_FLOAT32_DOT = FLOAT32_DOT_TF32X3


_PLAIN_TENSOR_TYPES = (Tensor, torch.nn.Parameter)


def resolve_triangle_multiplication_precisions(dtype: torch.dtype) -> tuple[str, int]:

    native_matmul_state = captured_native_matmul_precision_state(native_matmul_domain(dtype))
    return triton_dot_precision(dtype, _FLOAT32_DOT, native_matmul_state), native_matmul_state


def supports_triangle_multiplication(
    pair: Tensor,
    mask: Tensor | None,
    sample_row_offsets: Tensor | None,
    weights: TriangleMultiplicationWeights,
) -> bool:

    pair_size = pair.shape[-1]
    hidden_size = weights.left_value_weight.shape[0]
    affine = (weights.output_norm_weight, weights.output_norm_bias)
    return (
        pair.is_cuda
        and pair.dim() == 3
        and pair.dtype in STANDARD_FLOAT_DTYPES
        and guard_or_true(pair.shape[0] > 0)
        and guard_or_true(pair.shape[1] > 0)
        and 0 < pair_size <= MAX_FUSED_WIDTH
        and 0 < hidden_size <= MAX_FUSED_WIDTH
        and supports_layer_norm(pair, pair_size, weights.input_norm_weight, weights.input_norm_bias)
        and all(weight.device == pair.device and weight.is_contiguous() for weight in weights)
        and all(weight.dtype == pair.dtype for weight in weights.projections)
        and affine[0].dtype == affine[1].dtype
        and affine[0].dtype in (pair.dtype, torch.float32)
        and (mask is None or (mask.device == pair.device and mask.dim() == 2))
        and (
            sample_row_offsets is None
            or (
                sample_row_offsets.device == pair.device
                and sample_row_offsets.dtype is torch.int64
                and sample_row_offsets.dim() == 1
                and sample_row_offsets.shape[0] > 1
                and sample_row_offsets.is_contiguous()
            )
        )
        and _phases_fit(pair.dtype, pair_size, hidden_size, pair.device, groups=pair.shape[0], positions=pair.shape[1])
    )


def triton_triangle_multiplication(
    pair: Tensor,
    mask: Tensor | None,
    sample_row_offsets: Tensor | None,
    direction: str,
    weights: TriangleMultiplicationWeights,
    biases: TriangleMultiplicationBiases,
    num_tokens: int,
    input_norm_eps: float,
    output_norm_eps: float,
    precision: str,
    native_matmul_state: int,
    *,
    semantic: bool,
) -> Tensor:

    pair = pair.contiguous()
    operands = (pair, mask, *weights, *biases.present)
    eager = (
        not semantic
        and all(type(operand) in _PLAIN_TENSOR_TYPES for operand in operands if operand is not None)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(tuple(operand for operand in operands if operand is not None))
        and not functorch_transforms_active()
        and not forward_mode_active(*operands)
    )
    lean = eager and not (
        torch.is_grad_enabled() and any(operand.requires_grad for operand in operands if operand is not None)
    )
    normalized_pair = _input_layer_norm(
        pair, weights.input_norm_weight, weights.input_norm_bias, input_norm_eps, direct=lean
    )
    # Round normalization output to the prepared projection dtype before the GEMM.
    if normalized_pair.dtype != pair.dtype:
        normalized_pair = normalized_pair.to(pair.dtype)
    mask = None if mask is None else mask.to(pair.dtype)
    arguments = (
        normalized_pair,
        weights.left_value_weight,
        weights.left_gate_weight,
        weights.right_value_weight,
        weights.right_gate_weight,
        mask,
        weights.output_norm_weight,
        weights.output_norm_bias,
        weights.output_weight,
        weights.output_gate_weight,
        *biases,
        sample_row_offsets,
        direction,
        num_tokens,
        output_norm_eps,
        precision,
        native_matmul_state,
    )
    training = torch.is_grad_enabled() and any(
        operand.requires_grad for operand in arguments[:16] if operand is not None
    )
    if not training:
        if eager:
            return cast(Tensor, _inference(*arguments, guard_native_matmul_state=False))
        return _triangle_multiplication_inference(*arguments, semantic)
    projection_arguments = (
        normalized_pair,
        *arguments[1:6],
        *arguments[10:14],
        precision,
        native_matmul_state,
    )
    if eager:
        factors = _EagerProjection.apply(*projection_arguments)
    else:
        factors = _triangle_projection_forward(*projection_arguments, semantic)
    contraction_arguments = (factors, sample_row_offsets, direction, num_tokens, precision, native_matmul_state)
    if eager:
        combined = _EagerContraction.apply(*contraction_arguments)
    else:
        combined = _triangle_contraction_forward(*contraction_arguments, semantic)
    epilogue_arguments = (
        combined,
        normalized_pair,
        weights.output_norm_weight,
        weights.output_norm_bias,
        weights.output_weight,
        weights.output_gate_weight,
        biases.output_bias,
        biases.output_gate_bias,
        output_norm_eps,
        precision,
        native_matmul_state,
    )
    if eager:
        return _EagerEpilogue.apply(*epilogue_arguments)[0]
    return _triangle_epilogue_forward(*epilogue_arguments, semantic)[0]


_TENSOR_ARGUMENTS = (
    "Tensor normalized_pair, Tensor left_value_weight, Tensor left_gate_weight, "
    "Tensor right_value_weight, Tensor right_gate_weight, Tensor? mask, "
    "Tensor output_norm_weight, Tensor output_norm_bias, Tensor output_weight, Tensor output_gate_weight, "
    "Tensor? left_value_bias, Tensor? left_gate_bias, Tensor? right_value_bias, Tensor? right_gate_bias, "
    "Tensor? output_bias, Tensor? output_gate_bias, Tensor? sample_row_offsets"
)


_INFERENCE_SCHEMA = (
    f"({_TENSOR_ARGUMENTS}, str direction, SymInt num_tokens, float eps, str precision, "
    "int native_matmul_state, bool guard_native_matmul_state) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_triangle_multiplication_inference", mutates_args=(), device_types="cuda", schema=_INFERENCE_SCHEMA
)
def _triangle_multiplication_inference(
    normalized_pair: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    mask: Tensor | None,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    left_value_bias: Tensor | None,
    left_gate_bias: Tensor | None,
    right_value_bias: Tensor | None,
    right_gate_bias: Tensor | None,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    eps: float,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    return _inference(
        normalized_pair,
        left_value_weight,
        left_gate_weight,
        right_value_weight,
        right_gate_weight,
        mask,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        output_gate_weight,
        left_value_bias,
        left_gate_bias,
        right_value_bias,
        right_gate_bias,
        output_bias,
        output_gate_bias,
        sample_row_offsets,
        direction,
        num_tokens,
        eps,
        precision,
        native_matmul_state,
        guard_native_matmul_state,
    )


@torch.library.register_fake(_triangle_multiplication_inference)
def _triangle_multiplication_inference_fake(
    normalized_pair: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    mask: Tensor | None,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    left_value_bias: Tensor | None,
    left_gate_bias: Tensor | None,
    right_value_bias: Tensor | None,
    right_gate_bias: Tensor | None,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    eps: float,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    return torch.empty_like(normalized_pair)


_TRIANGLE_PROJECTION_FORWARD_SCHEMA = (
    "(Tensor normalized_pair, Tensor left_value_weight, Tensor left_gate_weight, Tensor "
    "right_value_weight, Tensor right_gate_weight, Tensor? mask, Tensor? left_value_bias, Tensor? "
    "left_gate_bias, Tensor? right_value_bias, Tensor? right_gate_bias, str precision, int "
    "native_matmul_state, bool guard_native_matmul_state) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_triangle_projection_forward",
    mutates_args=(),
    device_types="cuda",
    schema=_TRIANGLE_PROJECTION_FORWARD_SCHEMA,
)
def _triangle_projection_forward(
    normalized_pair: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    mask: Tensor | None,
    left_value_bias: Tensor | None,
    left_gate_bias: Tensor | None,
    right_value_bias: Tensor | None,
    right_gate_bias: Tensor | None,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    return _projection_forward(
        normalized_pair,
        left_value_weight,
        left_gate_weight,
        right_value_weight,
        right_gate_weight,
        mask,
        left_value_bias,
        left_gate_bias,
        right_value_bias,
        right_gate_bias,
        precision,
        native_matmul_state,
        guard_native_matmul_state,
    )


@torch.library.register_fake(_triangle_projection_forward)
def _triangle_projection_forward_fake(
    normalized_pair: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    mask: Tensor | None,
    left_value_bias: Tensor | None,
    left_gate_bias: Tensor | None,
    right_value_bias: Tensor | None,
    right_gate_bias: Tensor | None,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    num_groups, num_positions, _ = normalized_pair.shape
    hidden_size = left_value_weight.shape[0]
    return normalized_pair.new_empty((2, num_groups, hidden_size, num_positions))


_TRIANGLE_PROJECTION_BACKWARD_SCHEMA = (
    "(Tensor grad_factors, Tensor normalized_pair, Tensor left_value_weight, Tensor left_gate_weight, "
    "Tensor right_value_weight, Tensor right_gate_weight, Tensor? mask, Tensor? left_value_bias, Tensor? "
    "left_gate_bias, Tensor? right_value_bias, Tensor? right_gate_bias, str precision, int "
    "native_matmul_state, bool need_grad_pair, bool need_grad_left_value_weight, bool "
    "need_grad_left_gate_weight, bool need_grad_right_value_weight, bool need_grad_right_gate_weight, "
    "bool need_grad_mask, bool need_grad_left_value_bias, bool need_grad_left_gate_bias, bool "
    "need_grad_right_value_bias, bool need_grad_right_gate_bias) -> (Tensor?, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_triangle_projection_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_TRIANGLE_PROJECTION_BACKWARD_SCHEMA,
)
def _triangle_projection_backward(
    grad_factors: Tensor,
    normalized_pair: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    mask: Tensor | None,
    left_value_bias: Tensor | None,
    left_gate_bias: Tensor | None,
    right_value_bias: Tensor | None,
    right_gate_bias: Tensor | None,
    precision: str,
    native_matmul_state: int,
    need_grad_pair: bool,
    need_grad_left_value_weight: bool,
    need_grad_left_gate_weight: bool,
    need_grad_right_value_weight: bool,
    need_grad_right_gate_weight: bool,
    need_grad_mask: bool,
    need_grad_left_value_bias: bool,
    need_grad_left_gate_bias: bool,
    need_grad_right_value_bias: bool,
    need_grad_right_gate_bias: bool,
) -> tuple[Tensor | None, ...]:
    return _projection_backward(
        grad_factors,
        normalized_pair,
        left_value_weight,
        left_gate_weight,
        right_value_weight,
        right_gate_weight,
        mask,
        left_value_bias,
        left_gate_bias,
        right_value_bias,
        right_gate_bias,
        precision,
        native_matmul_state,
        (
            need_grad_pair,
            need_grad_left_value_weight,
            need_grad_left_gate_weight,
            need_grad_right_value_weight,
            need_grad_right_gate_weight,
            need_grad_mask,
            need_grad_left_value_bias,
            need_grad_left_gate_bias,
            need_grad_right_value_bias,
            need_grad_right_gate_bias,
        ),
    )


@torch.library.register_fake(_triangle_projection_backward)
def _triangle_projection_backward_fake(
    grad_factors: Tensor,
    normalized_pair: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    mask: Tensor | None,
    left_value_bias: Tensor | None,
    left_gate_bias: Tensor | None,
    right_value_bias: Tensor | None,
    right_gate_bias: Tensor | None,
    precision: str,
    native_matmul_state: int,
    need_grad_pair: bool,
    need_grad_left_value_weight: bool,
    need_grad_left_gate_weight: bool,
    need_grad_right_value_weight: bool,
    need_grad_right_gate_weight: bool,
    need_grad_mask: bool,
    need_grad_left_value_bias: bool,
    need_grad_left_gate_bias: bool,
    need_grad_right_value_bias: bool,
    need_grad_right_gate_bias: bool,
) -> tuple[Tensor | None, ...]:
    tensors = (
        normalized_pair,
        left_value_weight,
        left_gate_weight,
        right_value_weight,
        right_gate_weight,
        mask,
        left_value_bias,
        left_gate_bias,
        right_value_bias,
        right_gate_bias,
    )
    needs = (
        need_grad_pair,
        need_grad_left_value_weight,
        need_grad_left_gate_weight,
        need_grad_right_value_weight,
        need_grad_right_gate_weight,
        need_grad_mask,
        need_grad_left_value_bias,
        need_grad_left_gate_bias,
        need_grad_right_value_bias,
        need_grad_right_gate_bias,
    )
    return tuple(
        torch.empty_like(tensor) if needed and tensor is not None else None
        for tensor, needed in zip(tensors, needs, strict=True)
    )


_TRIANGLE_CONTRACTION_FORWARD_SCHEMA = (
    "(Tensor factors, Tensor? sample_row_offsets, str direction, SymInt num_tokens, str precision, int "
    "native_matmul_state, bool guard_native_matmul_state) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_triangle_contraction_forward",
    mutates_args=(),
    device_types="cuda",
    schema=_TRIANGLE_CONTRACTION_FORWARD_SCHEMA,
)
def _triangle_contraction_forward(
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    return _contraction_forward(
        factors,
        sample_row_offsets,
        direction,
        num_tokens,
        precision,
        native_matmul_state,
        guard_native_matmul_state,
    )


@torch.library.register_fake(_triangle_contraction_forward)
def _triangle_contraction_forward_fake(
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    _, num_groups, hidden_size, num_positions = factors.shape
    return factors.new_empty((num_groups, hidden_size, num_positions))


_TRIANGLE_CONTRACTION_BACKWARD_SCHEMA = (
    "(Tensor grad_combined, Tensor factors, Tensor? sample_row_offsets, str direction, SymInt "
    "num_tokens, str precision, int native_matmul_state) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_triangle_contraction_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_TRIANGLE_CONTRACTION_BACKWARD_SCHEMA,
)
def _triangle_contraction_backward(
    grad_combined: Tensor,
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
) -> Tensor:
    return _contraction_backward(
        grad_combined,
        factors,
        sample_row_offsets,
        direction,
        num_tokens,
        precision,
        native_matmul_state,
    )


@torch.library.register_fake(_triangle_contraction_backward)
def _triangle_contraction_backward_fake(
    grad_combined: Tensor,
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
) -> Tensor:
    return torch.empty_like(factors)


_TRIANGLE_EPILOGUE_FORWARD_SCHEMA = (
    "(Tensor combined, Tensor normalized_pair, Tensor output_norm_weight, Tensor output_norm_bias, "
    "Tensor output_weight, Tensor output_gate_weight, Tensor? output_bias, Tensor? output_gate_bias, "
    "float eps, str precision, int native_matmul_state, bool guard_native_matmul_state) -> (Tensor, "
    "Tensor, Tensor, Tensor)"
)


@torch.library.custom_op(
    "ooperators::_triangle_epilogue_forward",
    mutates_args=(),
    device_types="cuda",
    schema=_TRIANGLE_EPILOGUE_FORWARD_SCHEMA,
)
def _triangle_epilogue_forward(
    combined: Tensor,
    normalized_pair: Tensor,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    eps: float,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> tuple[Tensor, ...]:
    return _epilogue_forward(
        combined,
        normalized_pair,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        output_gate_weight,
        output_bias,
        output_gate_bias,
        eps,
        precision,
        native_matmul_state,
        guard_native_matmul_state,
    )


@torch.library.register_fake(_triangle_epilogue_forward)
def _triangle_epilogue_forward_fake(
    combined: Tensor,
    normalized_pair: Tensor,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    eps: float,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> tuple[Tensor, ...]:
    num_groups, hidden_size, num_positions = combined.shape
    return (
        normalized_pair.new_empty(normalized_pair.shape),
        combined.new_empty((num_groups, hidden_size, num_positions)),
        torch.empty((num_groups * num_positions,), device=combined.device, dtype=torch.float32),
        torch.empty((num_groups * num_positions,), device=combined.device, dtype=torch.float32),
    )


_TRIANGLE_EPILOGUE_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor combined, Tensor normalized_pair, Tensor output_norm_weight, Tensor "
    "output_norm_bias, Tensor output_weight, Tensor output_gate_weight, Tensor? output_bias, Tensor? "
    "output_gate_bias, Tensor normalized, Tensor mean, Tensor rstd, str precision, int "
    "native_matmul_state, bool need_grad_combined, bool need_grad_pair, bool "
    "need_grad_output_norm_weight, bool need_grad_output_norm_bias, bool need_grad_output_weight, bool "
    "need_grad_output_gate_weight, bool need_grad_output_bias, bool need_grad_output_gate_bias) -> "
    "(Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_triangle_epilogue_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_TRIANGLE_EPILOGUE_BACKWARD_SCHEMA,
)
def _triangle_epilogue_backward(
    grad_output: Tensor,
    combined: Tensor,
    normalized_pair: Tensor,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    normalized: Tensor,
    mean: Tensor,
    rstd: Tensor,
    precision: str,
    native_matmul_state: int,
    need_grad_combined: bool,
    need_grad_pair: bool,
    need_grad_output_norm_weight: bool,
    need_grad_output_norm_bias: bool,
    need_grad_output_weight: bool,
    need_grad_output_gate_weight: bool,
    need_grad_output_bias: bool,
    need_grad_output_gate_bias: bool,
) -> tuple[Tensor | None, ...]:
    return _epilogue_backward(
        grad_output,
        combined,
        normalized_pair,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        output_gate_weight,
        output_bias,
        output_gate_bias,
        normalized,
        mean,
        rstd,
        precision,
        native_matmul_state,
        (
            need_grad_combined,
            need_grad_pair,
            need_grad_output_norm_weight,
            need_grad_output_norm_bias,
            need_grad_output_weight,
            need_grad_output_gate_weight,
            need_grad_output_bias,
            need_grad_output_gate_bias,
        ),
    )


@torch.library.register_fake(_triangle_epilogue_backward)
def _triangle_epilogue_backward_fake(
    grad_output: Tensor,
    combined: Tensor,
    normalized_pair: Tensor,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    normalized: Tensor,
    mean: Tensor,
    rstd: Tensor,
    precision: str,
    native_matmul_state: int,
    need_grad_combined: bool,
    need_grad_pair: bool,
    need_grad_output_norm_weight: bool,
    need_grad_output_norm_bias: bool,
    need_grad_output_weight: bool,
    need_grad_output_gate_weight: bool,
    need_grad_output_bias: bool,
    need_grad_output_gate_bias: bool,
) -> tuple[Tensor | None, ...]:
    tensors = (
        combined,
        normalized_pair,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        output_gate_weight,
        output_bias,
        output_gate_bias,
    )
    needs = (
        need_grad_combined,
        need_grad_pair,
        need_grad_output_norm_weight,
        need_grad_output_norm_bias,
        need_grad_output_weight,
        need_grad_output_gate_weight,
        need_grad_output_bias,
        need_grad_output_gate_bias,
    )
    return tuple(
        torch.empty_like(tensor) if needed and tensor is not None else None
        for tensor, needed in zip(tensors, needs, strict=True)
    )


def _setup_projection_context(ctx, inputs, output) -> None:
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*inputs[:10])
    ctx.precision, ctx.native_matmul_state = inputs[10:12]


def _projection_autograd_backward(ctx, grad_factors: Tensor):
    gradients = _triangle_projection_backward(
        grad_factors.contiguous(),
        *ctx.saved_tensors,
        ctx.precision,
        ctx.native_matmul_state,
        *ctx.needs_input_grad[:10],
    )
    return (*gradients, None, None, None)


class _EagerProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *arguments):
        output = _projection_forward(*arguments, guard_native_matmul_state=False)
        _setup_projection_context(ctx, (*arguments, False), output)
        return output

    @staticmethod
    def backward(ctx, grad_factors):
        if torch.is_grad_enabled():
            return _projection_autograd_backward(ctx, grad_factors)[:-1]
        gradients = _projection_backward(
            grad_factors.contiguous(),
            *ctx.saved_tensors,
            ctx.precision,
            ctx.native_matmul_state,
            ctx.needs_input_grad[:10],
        )
        return (*gradients, None, None)


def _setup_contraction_context(ctx, inputs, output) -> None:
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*inputs[:2])
    ctx.direction, ctx.num_tokens, ctx.precision, ctx.native_matmul_state = inputs[2:6]


def _contraction_autograd_backward(ctx, grad_combined: Tensor):
    grad_factors = _triangle_contraction_backward(
        grad_combined.contiguous(),
        *ctx.saved_tensors,
        ctx.direction,
        ctx.num_tokens,
        ctx.precision,
        ctx.native_matmul_state,
    )
    return grad_factors, None, None, None, None, None, None


class _EagerContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *arguments):
        output = _contraction_forward(*arguments, guard_native_matmul_state=False)
        _setup_contraction_context(ctx, (*arguments, False), output)
        return output

    @staticmethod
    def backward(ctx, grad_combined):
        if torch.is_grad_enabled():
            return _contraction_autograd_backward(ctx, grad_combined)[:-1]
        grad_factors = _contraction_backward(
            grad_combined.contiguous(),
            *ctx.saved_tensors,
            ctx.direction,
            ctx.num_tokens,
            ctx.precision,
            ctx.native_matmul_state,
        )
        return grad_factors, None, None, None, None, None


torch.library.register_autograd(
    _triangle_projection_forward, _projection_autograd_backward, setup_context=_setup_projection_context
)


torch.library.register_autograd(
    _triangle_contraction_forward, _contraction_autograd_backward, setup_context=_setup_contraction_context
)


def _setup_epilogue_context(ctx, inputs, output) -> None:
    tensors = inputs[:8]
    _, precision, native_matmul_state = inputs[8:11]
    state = output[1:]
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*state)
    ctx.save_for_backward(*tensors, *state)
    ctx.precision = precision
    ctx.native_matmul_state = native_matmul_state


def _epilogue_autograd_backward(ctx, grad_output: Tensor, *grad_state):
    gradients = _triangle_epilogue_backward(
        grad_output.contiguous(),
        *ctx.saved_tensors,
        ctx.precision,
        ctx.native_matmul_state,
        *ctx.needs_input_grad[:8],
    )
    return (*gradients, *((None,) * 4))


class _EagerEpilogue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *arguments):
        output = _epilogue_forward(*arguments, guard_native_matmul_state=False)
        _setup_epilogue_context(ctx, (*arguments, False), output)
        return output

    @staticmethod
    def backward(ctx, grad_output, *grad_state):
        if torch.is_grad_enabled():
            return _epilogue_autograd_backward(ctx, grad_output, *grad_state)[:-1]
        gradients = _epilogue_backward(
            grad_output.contiguous(),
            *ctx.saved_tensors,
            ctx.precision,
            ctx.native_matmul_state,
            ctx.needs_input_grad[:8],
        )
        return (*gradients, *((None,) * 3))


torch.library.register_autograd(
    _triangle_epilogue_forward, _epilogue_autograd_backward, setup_context=_setup_epilogue_context
)
