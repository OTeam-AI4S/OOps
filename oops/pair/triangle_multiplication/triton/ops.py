from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import guard_or_true, statically_known_false
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.core import TargetDescriptor
from oops.core._math import _ceil_div
from oops.normalization.layer_norm.layer_norm import layer_norm
from oops.normalization.layer_norm.triton import supports_layer_norm
from oops.normalization.layer_norm.triton.backend import layer_norm_forward_into
from oops.runtime import (
    CUDA_MAX_GRID_X,
    FLOAT32_DOT_TF32X3,
    STANDARD_FLOAT_DTYPES,
    captured_native_matmul_precision_state,
    cuda_device_caps,
    cuda_device_context,
    forward_mode_active,
    functorch_transforms_active,
    native_matmul_domain,
    native_matmul_fp16_accumulation_enabled,
    require_native_matmul_precision_state,
    shared_memory_budget,
    triton_dot_precision,
)
from oops.runtime.dispatch import _kernel_operand

from ..parameters import TriangleMultiplicationBiases, TriangleMultiplicationWeights
from . import kernel
from .profiles import tile_table
from .schedule import (
    GEMM_PHASES,
    MAX_FUSED_WIDTH,
    SEGMENTED_PHASES,
    GemmTile,
    TileTable,
    TriangleMultiplicationPhase,
    full_width,
)

_OPERATOR = "triangle_multiplication"
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


def _phases_fit(
    dtype: torch.dtype,
    pair_size: int,
    hidden_size: int,
    device: torch.device,
    *,
    groups: int,
    positions: int,
) -> bool:

    target = cuda_device_caps(device)
    table = tile_table(target)
    budget = shared_memory_budget(target)
    gemm_tiles = {phase: table.gemm_tile(phase, dtype, pair_size, hidden_size, budget) for phase in GEMM_PHASES}
    if (
        any(tile is None for tile in gemm_tiles.values())
        or any(table.segmented_tile(phase, dtype, budget) is None for phase in SEGMENTED_PHASES)
        or table.layer_norm_backward_tile(dtype, pair_size, hidden_size, budget) is None
    ):
        return False
    launchable = groups * _ceil_div(positions, table.layer_norm_forward.block_rows(hidden_size)) <= CUDA_MAX_GRID_X
    for phase, tile in gemm_tiles.items():
        assert tile is not None
        if phase in (TriangleMultiplicationPhase.PROJECTION_FORWARD, TriangleMultiplicationPhase.PROJECTION_BACKWARD):
            channel_programs = 2 * _ceil_div(hidden_size, tile.block_channels)
        else:
            channel_programs = _ceil_div(pair_size, tile.block_channels)
        launchable = launchable & (groups * _ceil_div(positions, tile.block_rows) * channel_programs <= CUDA_MAX_GRID_X)
    if launchable is True or launchable is False:
        return launchable
    if statically_known_false(launchable):
        return False
    torch._check(launchable)
    return True


def _gemm_tile(
    table: TileTable,
    phase: TriangleMultiplicationPhase,
    dtype: torch.dtype,
    pair_size: int,
    hidden_size: int,
    shared_memory_bytes: int,
) -> GemmTile:

    tile = table.gemm_tile(phase, dtype, pair_size, hidden_size, shared_memory_bytes)
    if tile is None:
        raise RuntimeError(f"triangle multiplication {phase.value} tile disappeared after admission")
    return tile


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


def _input_layer_norm(pair: Tensor, weight: Tensor, bias: Tensor, eps: float, *, direct: bool) -> Tensor:

    if direct and type(eps) in (float, int):
        output = torch.empty(pair.shape, device=pair.device, dtype=pair.dtype)
        layer_norm_forward_into(pair, weight, bias, output, float(eps))
        return output
    return layer_norm(pair, pair.shape[-1], weight, bias, eps)


def _inference(
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

    with cuda_device_context(normalized_pair.device):
        normalized_pair = _kernel_operand(normalized_pair)
        left_value_weight = _kernel_operand(left_value_weight)
        left_gate_weight = _kernel_operand(left_gate_weight)
        right_value_weight = _kernel_operand(right_value_weight)
        right_gate_weight = _kernel_operand(right_gate_weight)
        output_norm_weight = _kernel_operand(output_norm_weight)
        output_norm_bias = _kernel_operand(output_norm_bias)
        output_weight = _kernel_operand(output_weight)
        output_gate_weight = _kernel_operand(output_gate_weight)
        if mask is not None:
            mask = _kernel_operand(mask)
        if left_value_bias is not None:
            left_value_bias = _kernel_operand(left_value_bias)
        if left_gate_bias is not None:
            left_gate_bias = _kernel_operand(left_gate_bias)
        if right_value_bias is not None:
            right_value_bias = _kernel_operand(right_value_bias)
        if right_gate_bias is not None:
            right_gate_bias = _kernel_operand(right_gate_bias)
        if output_bias is not None:
            output_bias = _kernel_operand(output_bias)
        if output_gate_bias is not None:
            output_gate_bias = _kernel_operand(output_gate_bias)
        if guard_native_matmul_state:
            require_native_matmul_precision_state(_OPERATOR, native_matmul_state)
        target = cuda_device_caps(normalized_pair.device)
        table = tile_table(target)
        budget = shared_memory_budget(target)
        projection_weights = (left_value_weight, left_gate_weight, right_value_weight, right_gate_weight)
        projection_biases = (left_value_bias, left_gate_bias, right_value_bias, right_gate_bias)
        factors = _gated_projection(
            normalized_pair,
            mask,
            projection_weights,
            projection_biases,
            table,
            budget,
            precision,
        )
        combined = _combine(
            factors, sample_row_offsets, direction, num_tokens, precision, native_matmul_state, table, target
        )
        del factors
        normalized, mean, rstd = _channel_major_layer_norm(
            combined,
            output_norm_weight,
            output_norm_bias,
            eps,
            table,
            write_statistics=False,
        )
        output = _gated_output(
            normalized,
            normalized_pair,
            output_weight,
            output_gate_weight,
            output_bias,
            output_gate_bias,
            table,
            budget,
            precision,
        )
        return output


def _projection_forward(
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
    with cuda_device_context(normalized_pair.device):
        normalized_pair = _kernel_operand(normalized_pair)
        projection_weights = (
            _kernel_operand(left_value_weight),
            _kernel_operand(left_gate_weight),
            _kernel_operand(right_value_weight),
            _kernel_operand(right_gate_weight),
        )
        projection_biases = (
            None if left_value_bias is None else _kernel_operand(left_value_bias),
            None if left_gate_bias is None else _kernel_operand(left_gate_bias),
            None if right_value_bias is None else _kernel_operand(right_value_bias),
            None if right_gate_bias is None else _kernel_operand(right_gate_bias),
        )
        if mask is not None:
            mask = _kernel_operand(mask)
        if guard_native_matmul_state:
            require_native_matmul_precision_state(_OPERATOR, native_matmul_state)
        target = cuda_device_caps(normalized_pair.device)
        table = tile_table(target)
        factors = _gated_projection(
            normalized_pair,
            mask,
            projection_weights,
            projection_biases,
            table,
            shared_memory_budget(target),
            precision,
        )
        return factors


def _contraction_forward(
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
    guard_native_matmul_state: bool,
) -> Tensor:
    with cuda_device_context(factors.device):
        factors = _kernel_operand(factors)
        if guard_native_matmul_state:
            require_native_matmul_precision_state(_OPERATOR, native_matmul_state)
        target = cuda_device_caps(factors.device)
        return _combine(
            factors,
            sample_row_offsets,
            direction,
            num_tokens,
            precision,
            native_matmul_state,
            tile_table(target),
            target,
        )


def _contraction_backward(
    grad_combined: Tensor,
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
) -> Tensor:
    with cuda_device_context(factors.device):
        grad_combined = _kernel_operand(grad_combined)
        factors = _kernel_operand(factors)
        require_native_matmul_precision_state(_OPERATOR, native_matmul_state)
        target = cuda_device_caps(factors.device)
        return _combine_backward(
            grad_combined,
            factors,
            sample_row_offsets,
            direction,
            num_tokens,
            precision,
            native_matmul_state,
            tile_table(target),
            target,
            need_left=True,
            need_right=True,
        )


def _epilogue_forward(
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with cuda_device_context(normalized_pair.device):
        combined = _kernel_operand(combined)
        normalized_pair = _kernel_operand(normalized_pair)
        output_norm_weight = _kernel_operand(output_norm_weight)
        output_norm_bias = _kernel_operand(output_norm_bias)
        output_weight = _kernel_operand(output_weight)
        output_gate_weight = _kernel_operand(output_gate_weight)
        if output_bias is not None:
            output_bias = _kernel_operand(output_bias)
        if output_gate_bias is not None:
            output_gate_bias = _kernel_operand(output_gate_bias)
        if guard_native_matmul_state:
            require_native_matmul_precision_state(_OPERATOR, native_matmul_state)
        target = cuda_device_caps(normalized_pair.device)
        table = tile_table(target)
        normalized, mean, rstd = _channel_major_layer_norm(
            combined, output_norm_weight, output_norm_bias, eps, table, write_statistics=True
        )
        output = _gated_output(
            normalized,
            normalized_pair,
            output_weight,
            output_gate_weight,
            output_bias,
            output_gate_bias,
            table,
            shared_memory_budget(target),
            precision,
        )
        return output, normalized, cast(Tensor, mean), cast(Tensor, rstd)


def _gated_projection(
    normalized_pair: Tensor,
    mask: Tensor | None,
    weights: tuple[Tensor, Tensor, Tensor, Tensor],
    biases: tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None],
    table: TileTable,
    shared_memory_bytes: int,
    precision: str,
    *,
    need_left: bool = True,
    need_right: bool = True,
) -> Tensor:

    num_groups, num_positions, pair_size = normalized_pair.shape
    hidden_size = weights[0].shape[0]
    tile = _gemm_tile(
        table,
        TriangleMultiplicationPhase.PROJECTION_FORWARD,
        normalized_pair.dtype,
        pair_size,
        hidden_size,
        shared_memory_bytes,
    )
    row_blocks = _ceil_div(num_positions, tile.block_rows)
    channel_blocks = _ceil_div(hidden_size, tile.block_channels)
    factor_count = int(need_left) + int(need_right)
    factors = torch.empty(
        (factor_count, num_groups, hidden_size, num_positions),
        device=normalized_pair.device,
        dtype=normalized_pair.dtype,
    )
    kernel._gated_projection_forward.run(
        normalized_pair,
        *weights,
        *(_pointer(bias, normalized_pair) for bias in biases),
        _pointer(mask, normalized_pair),
        factors,
        num_groups,
        num_positions,
        pair_size,
        hidden_size,
        row_blocks,
        channel_blocks,
        mask is not None,
        need_left,
        need_right,
        *(bias is not None for bias in biases),
        tile.block_rows,
        tile.block_channels,
        tile.block_reduction,
        input_precision=precision,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        grid=_grid(num_groups * row_blocks * factor_count * channel_blocks),
        warmup=False,
    )
    return factors


def _channel_major_layer_norm(
    combined: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    table: TileTable,
    *,
    write_statistics: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None]:

    num_groups, hidden_size, num_positions = combined.shape
    normalized = torch.empty_like(combined)
    mean = (
        torch.empty((num_groups * num_positions,), device=combined.device, dtype=torch.float32)
        if write_statistics
        else None
    )
    rstd = torch.empty_like(mean) if mean is not None else None
    tile = table.layer_norm_forward
    block_rows = tile.block_rows(hidden_size)
    row_blocks = _ceil_div(num_positions, block_rows)
    kernel._channel_major_layer_norm.run(
        combined,
        weight,
        bias,
        normalized,
        _pointer(mean, combined),
        _pointer(rstd, combined),
        num_positions,
        hidden_size,
        row_blocks,
        eps,
        write_statistics,
        True,
        block_rows,
        full_width(hidden_size),
        num_warps=tile.num_warps,
        grid=_grid(num_groups * row_blocks),
        warmup=False,
    )
    return normalized, mean, rstd


def _gated_output(
    normalized: Tensor,
    normalized_pair: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    table: TileTable,
    shared_memory_bytes: int,
    precision: str,
) -> Tensor:

    num_groups, hidden_size, num_positions = normalized.shape
    pair_size = normalized_pair.shape[-1]
    tile = _gemm_tile(
        table,
        TriangleMultiplicationPhase.OUTPUT_FORWARD,
        normalized.dtype,
        pair_size,
        hidden_size,
        shared_memory_bytes,
    )
    row_blocks = _ceil_div(num_positions, tile.block_rows)
    channel_blocks = _ceil_div(pair_size, tile.block_channels)
    output = torch.empty((num_groups, num_positions, pair_size), device=normalized.device, dtype=normalized.dtype)
    kernel._gated_output_forward.run(
        normalized,
        normalized_pair,
        output_weight,
        output_gate_weight,
        _pointer(output_bias, normalized_pair),
        _pointer(output_gate_bias, normalized_pair),
        output,
        num_positions,
        pair_size,
        hidden_size,
        row_blocks,
        channel_blocks,
        output_bias is not None,
        output_gate_bias is not None,
        tile.block_rows,
        tile.block_channels,
        tile.block_reduction,
        input_precision=precision,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        grid=_grid(num_groups * row_blocks * channel_blocks),
        warmup=False,
    )
    return output


def _combine(
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
    table: TileTable,
    target: TargetDescriptor,
) -> Tensor:

    left, right = factors
    if sample_row_offsets is None:
        num_groups, hidden_size, num_positions = left.shape
        square = (num_groups * hidden_size, num_tokens, num_tokens)
        left, right = left.reshape(square), right.reshape(square)
        combined = torch.empty(square, device=left.device, dtype=left.dtype)
        # The prepared factor dtype governs precision; ambient autocast must not recast it.
        with torch.autocast("cuda", enabled=False):
            if direction == "outgoing":
                torch.bmm(left, right.transpose(-1, -2), out=combined)
            else:
                torch.bmm(left.transpose(-1, -2), right, out=combined)
        return combined.reshape(num_groups, hidden_size, num_positions)
    combined = torch.empty_like(left)
    _segmented_grouped_gemm(
        left,
        right,
        sample_row_offsets,
        combined,
        phase=TriangleMultiplicationPhase.CONTRACTION_FORWARD,
        transpose_left=direction == "incoming",
        transpose_right=direction == "outgoing",
        precision=precision,
        native_matmul_state=native_matmul_state,
        table=table,
        target=target,
    )
    return combined


def _combine_backward(
    grad_combined: Tensor,
    factors: Tensor,
    sample_row_offsets: Tensor | None,
    direction: str,
    num_tokens: int,
    precision: str,
    native_matmul_state: int,
    table: TileTable,
    target: TargetDescriptor,
    *,
    need_left: bool,
    need_right: bool,
) -> Tensor:

    left = factors[0] if need_right else None
    right = factors[-1] if need_left else None
    grad_factors = torch.empty_like(factors)
    grad_left = grad_factors[0] if need_left else None
    grad_right = grad_factors[-1] if need_right else None
    if sample_row_offsets is None:
        num_groups, hidden_size, num_positions = grad_combined.shape
        square = (num_groups * hidden_size, num_tokens, num_tokens)
        grad = grad_combined.reshape(square)
        left = left.reshape(square) if left is not None else None
        right = right.reshape(square) if right is not None else None
        with torch.autocast("cuda", enabled=False):
            if direction == "outgoing":
                if grad_left is not None:
                    assert right is not None
                    torch.bmm(grad, right, out=grad_left.view(square))
                if grad_right is not None:
                    assert left is not None
                    torch.bmm(grad.transpose(-1, -2), left, out=grad_right.view(square))
            else:
                if grad_left is not None:
                    assert right is not None
                    torch.bmm(right, grad.transpose(-1, -2), out=grad_left.view(square))
                if grad_right is not None:
                    assert left is not None
                    torch.bmm(left, grad, out=grad_right.view(square))
        return grad_factors
    if direction == "outgoing":
        operands = ((grad_combined, right, grad_left, False, False), (grad_combined, left, grad_right, True, False))
    else:
        operands = ((right, grad_combined, grad_left, False, True), (left, grad_combined, grad_right, False, False))
    for lhs, rhs, output, transpose_left, transpose_right in operands:
        if output is None:
            continue
        assert lhs is not None and rhs is not None
        _segmented_grouped_gemm(
            lhs,
            rhs,
            sample_row_offsets,
            output,
            phase=TriangleMultiplicationPhase.CONTRACTION_BACKWARD,
            transpose_left=transpose_left,
            transpose_right=transpose_right,
            precision=precision,
            native_matmul_state=native_matmul_state,
            table=table,
            target=target,
        )
    return grad_factors


def _segmented_grouped_gemm(
    left: Tensor,
    right: Tensor,
    sample_row_offsets: Tensor,
    output: Tensor,
    *,
    phase: TriangleMultiplicationPhase,
    transpose_left: bool = False,
    transpose_right: bool = False,
    precision: str,
    native_matmul_state: int,
    table: TileTable,
    target: TargetDescriptor,
) -> None:

    sample_row_offsets = _kernel_operand(sample_row_offsets)
    hidden_size, total_positions = left.shape[-2], left.shape[-1]
    tile = table.segmented_tile(phase, left.dtype, shared_memory_budget(target))
    if tile is None:
        raise RuntimeError(f"triangle multiplication {phase.value} tile disappeared after admission")
    num_programs = _persistent_programs(target, tile.programs_per_multiprocessor)
    fp16_accumulation = left.dtype is torch.float16 and native_matmul_fp16_accumulation_enabled(native_matmul_state)
    kernel._segmented_grouped_gemm.run(
        left,
        right,
        sample_row_offsets,
        output,
        sample_row_offsets.shape[0] - 1,
        total_positions,
        hidden_size,
        num_programs,
        transpose_left,
        transpose_right,
        tile.block_rows,
        tile.block_columns,
        tile.block_reduction,
        input_precision=precision,
        fp16_accumulation=fp16_accumulation,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        grid=(num_programs,),
        warmup=False,
    )


def _projection_backward(
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
    needs: tuple[bool, ...],
) -> tuple[Tensor | None, ...]:
    need_pair, need_mask = needs[0], needs[5]
    weight_demands = needs[1:5]
    bias_demands = needs[6:10]
    slot_demands = tuple(need_pair or weight or bias for weight, bias in zip(weight_demands, bias_demands, strict=True))
    need_left_factor = need_mask or any(slot_demands[:2])
    need_right_factor = need_mask or any(slot_demands[2:])
    grad_factors = _kernel_operand(grad_factors)
    normalized_pair = _kernel_operand(normalized_pair)
    weights = (
        _kernel_operand(left_value_weight),
        _kernel_operand(left_gate_weight),
        _kernel_operand(right_value_weight),
        _kernel_operand(right_gate_weight),
    )
    biases = (
        None if left_value_bias is None else _kernel_operand(left_value_bias),
        None if left_gate_bias is None else _kernel_operand(left_gate_bias),
        None if right_value_bias is None else _kernel_operand(right_value_bias),
        None if right_gate_bias is None else _kernel_operand(right_gate_bias),
    )
    mask = None if mask is None else _kernel_operand(mask)
    require_native_matmul_precision_state(_OPERATOR, native_matmul_state)
    target = cuda_device_caps(normalized_pair.device)
    table = tile_table(target)
    budget = shared_memory_budget(target)
    num_groups, num_positions, pair_size = normalized_pair.shape
    hidden_size = left_value_weight.shape[0]
    widths = (hidden_size,) * 4
    starts_list = []
    packed_width = 0
    for needed, width in zip(slot_demands, widths, strict=True):
        starts_list.append(packed_width if needed else -1)
        if needed:
            packed_width += width
    starts = tuple(starts_list)
    grad_packed = torch.empty(
        (num_groups, num_positions, packed_width), device=normalized_pair.device, dtype=normalized_pair.dtype
    )
    grad_mask = _gated_projection_backward(
        grad_factors,
        normalized_pair,
        mask,
        weights,
        biases,
        grad_packed,
        table,
        budget,
        precision,
        need_grad_mask=need_mask,
        need_left=need_left_factor,
        need_right=need_right_factor,
        slot_demands=slot_demands,
    )
    del grad_factors
    packed_rows = grad_packed.reshape(num_groups * num_positions, packed_width)
    pair_rows = normalized_pair.reshape(-1, pair_size)
    grad_pair = None
    if need_pair:
        with torch.autocast("cuda", enabled=False):
            grad_pair = (packed_rows @ torch.cat(weights)).reshape(normalized_pair.shape)
    weight_gradients: list[Tensor | None] = [None] * 4
    for indices, first, last in _demanded_slot_runs(weight_demands, starts, widths):
        with torch.autocast("cuda", enabled=False):
            grad_weights = packed_rows[:, first:last].transpose(0, 1) @ pair_rows
        for index in indices:
            start = starts[index] - first
            weight_gradients[index] = grad_weights[start : start + widths[index]].clone()
        del grad_weights
    bias_gradients: list[Tensor | None] = [None] * 4
    for indices, first, last in _demanded_slot_runs(bias_demands, starts, widths):
        sums = _column_sums(packed_rows[:, first:last], table)
        for index in indices:
            bias = biases[index]
            assert bias is not None
            start = starts[index] - first
            bias_gradients[index] = sums[start : start + widths[index]].to(bias.dtype, copy=True)
    gradients = (grad_pair, *weight_gradients, grad_mask if mask is not None else None, *bias_gradients)
    return tuple(gradient if needed else None for gradient, needed in zip(gradients, needs, strict=True))


def _epilogue_backward(
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
    needs: tuple[bool, ...],
) -> tuple[Tensor | None, ...]:
    (
        need_combined,
        need_pair,
        need_norm_weight,
        need_norm_bias,
        need_output_weight,
        need_gate_weight,
        need_output_bias,
        need_gate_bias,
    ) = needs
    need_grad_value = need_combined or need_norm_weight or need_norm_bias or need_output_weight or need_output_bias
    need_grad_gate = need_pair or need_gate_weight or need_gate_bias
    grad_output = _kernel_operand(grad_output)
    combined = _kernel_operand(combined)
    normalized_pair = _kernel_operand(normalized_pair)
    output_norm_weight = _kernel_operand(output_norm_weight)
    output_weight = _kernel_operand(output_weight)
    output_gate_weight = _kernel_operand(output_gate_weight)
    if output_bias is not None:
        output_bias = _kernel_operand(output_bias)
    if output_gate_bias is not None:
        output_gate_bias = _kernel_operand(output_gate_bias)
    normalized = _kernel_operand(normalized)
    mean = _kernel_operand(mean)
    rstd = _kernel_operand(rstd)
    require_native_matmul_precision_state(_OPERATOR, native_matmul_state)
    target = cuda_device_caps(normalized_pair.device)
    table = tile_table(target)
    num_groups, num_positions, pair_size = normalized_pair.shape
    grad_gate = torch.empty(
        (num_groups, num_positions, pair_size if need_grad_gate else 0),
        device=normalized_pair.device,
        dtype=normalized_pair.dtype,
    )
    grad_value = _gated_output_backward(
        grad_output,
        normalized,
        normalized_pair,
        output_weight,
        output_gate_weight,
        output_bias,
        output_gate_bias,
        grad_gate,
        table,
        shared_memory_budget(target),
        precision,
        need_grad_value=need_grad_value,
        need_grad_gate=need_grad_gate,
    )
    grad_output_weight = None
    if need_output_weight:
        assert grad_value is not None
        with torch.autocast("cuda", enabled=False):
            per_group = torch.bmm(grad_value.transpose(1, 2), normalized.transpose(1, 2))
        grad_output_weight = per_group.float().sum(0).to(output_weight.dtype)
    grad_combined = grad_norm_weight = grad_norm_bias = None
    if need_combined or need_norm_weight or need_norm_bias:
        assert grad_value is not None
        grad_combined, grad_norm_weight, grad_norm_bias = _layer_norm_transpose_backward(
            grad_value,
            combined,
            output_norm_weight,
            mean,
            rstd,
            output_weight,
            table,
            target,
            precision,
            need_grad_combined=need_combined,
            need_norm_weight=need_norm_weight,
            need_norm_bias=need_norm_bias,
        )
    grad_output_bias = None
    if need_output_bias and output_bias is not None:
        assert grad_value is not None
        grad_output_bias = _column_sums(grad_value, table).to(output_bias.dtype)
    del grad_value
    grad_pair = grad_gate_weight = grad_gate_bias = None
    if need_grad_gate:
        gate_rows = grad_gate.reshape(-1, pair_size)
        with torch.autocast("cuda", enabled=False):
            if need_pair:
                grad_pair = (gate_rows @ output_gate_weight).reshape(normalized_pair.shape)
            if need_gate_weight:
                grad_gate_weight = gate_rows.transpose(0, 1) @ normalized_pair.reshape(-1, pair_size)
        if need_gate_bias and output_gate_bias is not None:
            grad_gate_bias = _column_sums(gate_rows, table).to(output_gate_bias.dtype)
    gradients = (
        grad_combined,
        grad_pair,
        grad_norm_weight,
        grad_norm_bias,
        grad_output_weight,
        grad_gate_weight,
        grad_output_bias,
        grad_gate_bias,
    )
    return tuple(gradient if needed else None for gradient, needed in zip(gradients, needs, strict=True))


def _demanded_slot_runs(
    demanded: tuple[bool, ...], starts: tuple[int, ...], widths: tuple[int, ...]
) -> list[tuple[list[int], int, int]]:

    runs: list[tuple[list[int], int, int]] = []
    for index, needed in enumerate(demanded):
        if not needed:
            continue
        start, stop = starts[index], starts[index] + widths[index]
        if runs and runs[-1][2] == start:
            indices, first, _ = runs[-1]
            indices.append(index)
            runs[-1] = (indices, first, stop)
        else:
            runs.append(([index], start, stop))
    return runs


def _gated_output_backward(
    grad_output: Tensor,
    normalized: Tensor,
    normalized_pair: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    output_bias: Tensor | None,
    output_gate_bias: Tensor | None,
    grad_packed: Tensor,
    table: TileTable,
    shared_memory_bytes: int,
    precision: str,
    *,
    need_grad_value: bool,
    need_grad_gate: bool,
) -> Tensor | None:
    num_groups, num_positions, pair_size = normalized_pair.shape
    hidden_size = output_weight.shape[1]
    tile = _gemm_tile(
        table,
        TriangleMultiplicationPhase.OUTPUT_BACKWARD,
        normalized_pair.dtype,
        pair_size,
        hidden_size,
        shared_memory_bytes,
    )
    row_blocks = _ceil_div(num_positions, tile.block_rows)
    channel_blocks = _ceil_div(pair_size, tile.block_channels)
    grad_value = (
        torch.empty((num_groups, num_positions, pair_size), device=normalized_pair.device, dtype=normalized_pair.dtype)
        if need_grad_value
        else None
    )
    kernel._gated_output_backward.run(
        grad_output,
        normalized,
        normalized_pair,
        output_weight,
        output_gate_weight,
        _pointer(output_bias, normalized_pair),
        _pointer(output_gate_bias, normalized_pair),
        _pointer(grad_value, normalized_pair),
        grad_packed,
        num_positions,
        pair_size,
        hidden_size,
        grad_packed.shape[-1],
        row_blocks,
        channel_blocks,
        output_bias is not None,
        output_gate_bias is not None,
        need_grad_value,
        need_grad_gate,
        tile.block_rows,
        tile.block_channels,
        tile.block_reduction,
        input_precision=precision,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        grid=_grid(num_groups * row_blocks * channel_blocks),
        warmup=False,
    )
    return grad_value


def _layer_norm_transpose_backward(
    grad_value: Tensor,
    combined: Tensor,
    output_norm_weight: Tensor,
    mean: Tensor,
    rstd: Tensor,
    output_weight: Tensor,
    table: TileTable,
    target: TargetDescriptor,
    precision: str,
    *,
    need_grad_combined: bool,
    need_norm_weight: bool,
    need_norm_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:

    num_groups, num_positions, pair_size = grad_value.shape
    hidden_size = output_norm_weight.shape[0]
    tile = table.layer_norm_backward_tile(combined.dtype, pair_size, hidden_size, shared_memory_budget(target))
    if tile is None:
        raise RuntimeError("triangle multiplication layer_norm_backward tile disappeared after admission")
    row_blocks = _ceil_div(num_positions, tile.block_rows)
    num_programs = min(_persistent_programs(target, tile.programs_per_multiprocessor), num_groups * row_blocks)
    grad_combined = torch.empty_like(combined) if need_grad_combined else None
    grad_weight_partial = (
        torch.empty((num_programs, hidden_size), device=combined.device, dtype=torch.float32)
        if need_norm_weight
        else None
    )
    grad_bias_partial = (
        torch.empty((num_programs, hidden_size), device=combined.device, dtype=torch.float32)
        if need_norm_bias
        else None
    )
    kernel._layer_norm_transpose_backward.run(
        grad_value,
        combined,
        output_norm_weight,
        mean,
        rstd,
        output_weight,
        _pointer(grad_combined, combined),
        _pointer(grad_weight_partial, combined),
        _pointer(grad_bias_partial, combined),
        num_groups,
        num_positions,
        pair_size,
        hidden_size,
        row_blocks,
        num_programs,
        need_grad_combined,
        need_norm_weight,
        need_norm_bias,
        tile.block_rows,
        tile.block_channels,
        full_width(hidden_size),
        input_precision=precision,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        grid=(num_programs,),
        warmup=False,
    )
    return (
        grad_combined,
        grad_weight_partial.sum(0).to(output_norm_weight.dtype) if grad_weight_partial is not None else None,
        grad_bias_partial.sum(0).to(output_norm_weight.dtype) if grad_bias_partial is not None else None,
    )


def _gated_projection_backward(
    grad_factors: Tensor,
    normalized_pair: Tensor,
    mask: Tensor | None,
    weights: tuple[Tensor, Tensor, Tensor, Tensor],
    biases: tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None],
    grad_packed: Tensor,
    table: TileTable,
    shared_memory_bytes: int,
    precision: str,
    *,
    need_grad_mask: bool,
    need_left: bool,
    need_right: bool,
    slot_demands: tuple[bool, ...],
) -> Tensor | None:
    num_groups, num_positions, pair_size = normalized_pair.shape
    hidden_size = weights[0].shape[0]
    tile = _gemm_tile(
        table,
        TriangleMultiplicationPhase.PROJECTION_BACKWARD,
        normalized_pair.dtype,
        pair_size,
        hidden_size,
        shared_memory_bytes,
    )
    row_blocks = _ceil_div(num_positions, tile.block_rows)
    channel_blocks = _ceil_div(hidden_size, tile.block_channels)
    factor_count = int(need_left) + int(need_right)
    grad_mask_partial = (
        torch.empty(
            (factor_count * channel_blocks, num_groups, num_positions),
            device=normalized_pair.device,
            dtype=torch.float32,
        )
        if need_grad_mask
        else None
    )
    kernel._gated_projection_backward.run(
        normalized_pair,
        *weights,
        *(_pointer(bias, normalized_pair) for bias in biases),
        _pointer(mask, normalized_pair),
        grad_factors,
        grad_packed,
        _pointer(grad_mask_partial, normalized_pair),
        num_groups,
        num_positions,
        pair_size,
        hidden_size,
        grad_packed.shape[-1],
        row_blocks,
        channel_blocks,
        mask is not None,
        need_grad_mask,
        need_left,
        need_right,
        *slot_demands,
        *(bias is not None for bias in biases),
        tile.block_rows,
        tile.block_channels,
        tile.block_reduction,
        input_precision=precision,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        grid=_grid(num_groups * row_blocks * factor_count * channel_blocks),
        warmup=False,
    )
    if grad_mask_partial is None or mask is None:
        return None
    return grad_mask_partial.sum(0).to(mask.dtype)


def _column_sums(matrix: Tensor, table: TileTable) -> Tensor:

    matrix = matrix.reshape(-1, matrix.shape[-1]) if matrix.dim() != 2 else matrix
    rows, width = matrix.shape
    tile = table.column_sum
    partitions = max(1, min(_ceil_div(rows, tile.block_rows), tile.max_partitions))
    partials = torch.empty((partitions, width), device=matrix.device, dtype=torch.float32)
    kernel._column_sum_partial.run(
        matrix,
        partials,
        rows,
        width,
        matrix.stride(0),
        partitions,
        tile.block_rows,
        tile.block_columns,
        num_warps=tile.num_warps,
        grid=(_ceil_div(width, tile.block_columns), partitions),
        warmup=False,
    )
    return partials.sum(0)


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


def _pointer(tensor: Tensor | None, placeholder: Tensor) -> Tensor:

    return placeholder if tensor is None else tensor


def _persistent_programs(target: TargetDescriptor, programs_per_multiprocessor: int) -> int:
    multiprocessors = target.resources.multiprocessor_count
    if multiprocessors is None:
        raise RuntimeError("triangle multiplication persistent kernels need the target multiprocessor count")
    return multiprocessors * programs_per_multiprocessor


def _grid(programs: int) -> tuple[int]:
    if programs > CUDA_MAX_GRID_X:
        raise RuntimeError(f"triangle multiplication launch of {programs} programs escaped grid admission")
    return (programs,)
