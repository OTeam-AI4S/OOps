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
from torch.fx.experimental.symbolic_shapes import statically_known_false

from oops.core import TargetDescriptor
from oops.core._math import _ceil_div
from oops.normalization.layer_norm.layer_norm import layer_norm
from oops.normalization.layer_norm.triton.backend import layer_norm_forward_into
from oops.runtime import (
    CUDA_MAX_GRID_X,
    cuda_device_caps,
    cuda_device_context,
    native_matmul_fp16_accumulation_enabled,
    require_native_matmul_precision_state,
    shared_memory_budget,
)
from oops.runtime.dispatch import _kernel_operand

from . import kernel
from .profiles import tile_table
from .schedule import GEMM_PHASES, SEGMENTED_PHASES, GemmTile, TileTable, TriangleMultiplicationPhase, full_width

_OPERATOR = "triangle_multiplication"


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
