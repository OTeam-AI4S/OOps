from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _rows_dot_weight(
    activation_ptr,
    row_stride,
    reduction_stride,
    weight_ptr,
    rows,
    row_mask,
    channels,
    channel_mask,
    reduction,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
):

    accumulator = tl.zeros((rows.shape[0], channels.shape[0]), tl.float32)
    for start in range(0, tl.cdiv(reduction, block_reduction)):
        k = start * block_reduction + tl.arange(0, block_reduction)
        k_mask = k < reduction
        activation = tl.load(
            activation_ptr + rows[:, None] * row_stride + k[None, :].to(tl.int64) * reduction_stride,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + channels[None, :] * reduction + k[:, None],
            mask=channel_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator = tl.dot(activation, weight, accumulator, input_precision=input_precision)
    return accumulator


@triton.jit
def _rows_dot_dual_weight(
    activation_ptr,
    row_stride,
    reduction_stride,
    value_weight_ptr,
    gate_weight_ptr,
    rows,
    row_mask,
    channels,
    channel_mask,
    reduction,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
):
    value = tl.zeros((rows.shape[0], channels.shape[0]), tl.float32)
    gate = tl.zeros((rows.shape[0], channels.shape[0]), tl.float32)
    for start in range(0, tl.cdiv(reduction, block_reduction)):
        k = start * block_reduction + tl.arange(0, block_reduction)
        k_mask = k < reduction
        activation = tl.load(
            activation_ptr + rows[:, None] * row_stride + k[None, :].to(tl.int64) * reduction_stride,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        value_weight = tl.load(
            value_weight_ptr + channels[None, :] * reduction + k[:, None],
            mask=channel_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        gate_weight = tl.load(
            gate_weight_ptr + channels[None, :] * reduction + k[:, None],
            mask=channel_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        value = tl.dot(activation, value_weight, value, input_precision=input_precision)
        gate = tl.dot(activation, gate_weight, gate, input_precision=input_precision)
    return value, gate


@triton.jit
def _add_bias(accumulator, bias_ptr, channels, channel_mask, has_bias: tl.constexpr):
    if has_bias:
        accumulator += tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    return accumulator


@triton.jit
def _gated_projection_forward(
    pair_ptr,
    left_value_weight_ptr,
    left_gate_weight_ptr,
    right_value_weight_ptr,
    right_gate_weight_ptr,
    left_value_bias_ptr,
    left_gate_bias_ptr,
    right_value_bias_ptr,
    right_gate_bias_ptr,
    mask_ptr,
    factors_ptr,
    num_groups,
    num_positions,
    pair_size,
    hidden_size,
    row_blocks,
    channel_blocks,
    has_mask: tl.constexpr,
    need_left: tl.constexpr,
    need_right: tl.constexpr,
    has_left_value_bias: tl.constexpr,
    has_left_gate_bias: tl.constexpr,
    has_right_value_bias: tl.constexpr,
    has_right_gate_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    factor_count: tl.constexpr = (1 if need_left else 0) + (1 if need_right else 0)
    combo = tl.program_id(0) % (factor_count * channel_blocks)
    block = tl.program_id(0) // (factor_count * channel_blocks)
    physical_factor = combo // channel_blocks
    factor = physical_factor if need_left else 1
    group = block // row_blocks
    rows = (block % row_blocks).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    channels = (combo % channel_blocks) * block_channels + tl.arange(0, block_channels)
    row_mask = rows < num_positions
    channel_mask = channels < hidden_size
    pair_rows = pair_ptr + group.to(tl.int64) * num_positions * pair_size
    if factor == 0:
        value_weight_ptr = left_value_weight_ptr
        gate_weight_ptr = left_gate_weight_ptr
    else:
        value_weight_ptr = right_value_weight_ptr
        gate_weight_ptr = right_gate_weight_ptr
    value, gate = _rows_dot_dual_weight(
        pair_rows,
        pair_size,
        1,
        value_weight_ptr,
        gate_weight_ptr,
        rows,
        row_mask,
        channels,
        channel_mask,
        pair_size,
        block_reduction,
        input_precision,
    )
    if factor == 0:
        value = _add_bias(value, left_value_bias_ptr, channels, channel_mask, has_left_value_bias)
        gate = _add_bias(gate, left_gate_bias_ptr, channels, channel_mask, has_left_gate_bias)
    else:
        value = _add_bias(value, right_value_bias_ptr, channels, channel_mask, has_right_value_bias)
        gate = _add_bias(gate, right_gate_bias_ptr, channels, channel_mask, has_right_gate_bias)
    gate_sigmoid = tl.sigmoid(gate)
    result = value * gate_sigmoid
    if has_mask:
        keep = tl.load(mask_ptr + group.to(tl.int64) * num_positions + rows, mask=row_mask, other=0.0)
        result *= keep.to(tl.float32)[:, None]
    factor_rows = factors_ptr + (physical_factor * num_groups + group).to(tl.int64) * hidden_size * num_positions
    tl.store(
        factor_rows + channels[:, None].to(tl.int64) * num_positions + rows[None, :],
        tl.trans(result),
        mask=channel_mask[:, None] & row_mask[None, :],
    )


@triton.jit
def _gated_projection_backward(
    pair_ptr,
    left_value_weight_ptr,
    left_gate_weight_ptr,
    right_value_weight_ptr,
    right_gate_weight_ptr,
    left_value_bias_ptr,
    left_gate_bias_ptr,
    right_value_bias_ptr,
    right_gate_bias_ptr,
    mask_ptr,
    grad_factors_ptr,
    grad_packed_ptr,
    grad_mask_partial_ptr,
    num_groups,
    num_positions,
    pair_size,
    hidden_size,
    packed_width,
    row_blocks,
    channel_blocks,
    has_mask: tl.constexpr,
    need_grad_mask: tl.constexpr,
    need_left: tl.constexpr,
    need_right: tl.constexpr,
    has_left_value_slot: tl.constexpr,
    has_left_gate_slot: tl.constexpr,
    has_right_value_slot: tl.constexpr,
    has_right_gate_slot: tl.constexpr,
    has_left_value_bias: tl.constexpr,
    has_left_gate_bias: tl.constexpr,
    has_right_value_bias: tl.constexpr,
    has_right_gate_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    factor_count: tl.constexpr = (1 if need_left else 0) + (1 if need_right else 0)
    left_gate_slot: tl.constexpr = 1 if has_left_value_slot else 0
    right_value_slot: tl.constexpr = left_gate_slot + (1 if has_left_gate_slot else 0)
    right_gate_slot: tl.constexpr = right_value_slot + (1 if has_right_value_slot else 0)
    combo = tl.program_id(0) % (factor_count * channel_blocks)
    block = tl.program_id(0) // (factor_count * channel_blocks)
    physical_factor = combo // channel_blocks
    factor = physical_factor if need_left else 1
    group = block // row_blocks
    rows = (block % row_blocks).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    channels = (combo % channel_blocks) * block_channels + tl.arange(0, block_channels)
    row_mask = rows < num_positions
    channel_mask = channels < hidden_size
    tile_mask = row_mask[:, None] & channel_mask[None, :]
    pair_rows = pair_ptr + group.to(tl.int64) * num_positions * pair_size
    if factor == 0:
        value_weight_ptr = left_value_weight_ptr
        gate_weight_ptr = left_gate_weight_ptr
    else:
        value_weight_ptr = right_value_weight_ptr
        gate_weight_ptr = right_gate_weight_ptr
    if need_grad_mask or (has_left_gate_slot and has_right_gate_slot):
        value, gate = _rows_dot_dual_weight(
            pair_rows,
            pair_size,
            1,
            value_weight_ptr,
            gate_weight_ptr,
            rows,
            row_mask,
            channels,
            channel_mask,
            pair_size,
            block_reduction,
            input_precision,
        )
    else:
        value = tl.zeros((block_rows, block_channels), tl.float32)
        if factor == 0:
            if has_left_gate_slot:
                value = _rows_dot_weight(
                    pair_rows,
                    pair_size,
                    1,
                    value_weight_ptr,
                    rows,
                    row_mask,
                    channels,
                    channel_mask,
                    pair_size,
                    block_reduction,
                    input_precision,
                )
        else:
            if has_right_gate_slot:
                value = _rows_dot_weight(
                    pair_rows,
                    pair_size,
                    1,
                    value_weight_ptr,
                    rows,
                    row_mask,
                    channels,
                    channel_mask,
                    pair_size,
                    block_reduction,
                    input_precision,
                )
        gate = _rows_dot_weight(
            pair_rows,
            pair_size,
            1,
            gate_weight_ptr,
            rows,
            row_mask,
            channels,
            channel_mask,
            pair_size,
            block_reduction,
            input_precision,
        )
    if factor == 0:
        value = _add_bias(value, left_value_bias_ptr, channels, channel_mask, has_left_value_bias)
        gate = _add_bias(gate, left_gate_bias_ptr, channels, channel_mask, has_left_gate_bias)
    else:
        value = _add_bias(value, right_value_bias_ptr, channels, channel_mask, has_right_value_bias)
        gate = _add_bias(gate, right_gate_bias_ptr, channels, channel_mask, has_right_gate_bias)
    gate_sigmoid = tl.sigmoid(gate)
    grad_rows = grad_factors_ptr + (factor * num_groups + group).to(tl.int64) * hidden_size * num_positions
    grad_factor = tl.trans(
        tl.load(
            grad_rows + channels[:, None].to(tl.int64) * num_positions + rows[None, :],
            mask=channel_mask[:, None] & row_mask[None, :],
            other=0.0,
        )
    ).to(tl.float32)
    if need_grad_mask:
        tl.store(
            grad_mask_partial_ptr + (combo * num_groups + group).to(tl.int64) * num_positions + rows,
            tl.sum(grad_factor * value * gate_sigmoid, axis=1),
            mask=row_mask,
        )
    if has_mask:
        keep = tl.load(mask_ptr + group.to(tl.int64) * num_positions + rows, mask=row_mask, other=0.0)
        grad_factor *= keep.to(tl.float32)[:, None]
    packed_rows = grad_packed_ptr + (group.to(tl.int64) * num_positions + rows)[:, None] * packed_width
    if has_left_value_slot and has_left_gate_slot and has_right_value_slot and has_right_gate_slot:
        columns = (2 * factor * hidden_size + channels)[None, :]
        tl.store(packed_rows + columns, grad_factor * gate_sigmoid, mask=tile_mask)
        tl.store(
            packed_rows + columns + hidden_size,
            grad_factor * value * gate_sigmoid * (1.0 - gate_sigmoid),
            mask=tile_mask,
        )
    elif factor == 0:
        if has_left_value_slot:
            tl.store(packed_rows + channels[None, :], grad_factor * gate_sigmoid, mask=tile_mask)
        if has_left_gate_slot:
            tl.store(
                packed_rows + left_gate_slot * hidden_size + channels[None, :],
                grad_factor * value * gate_sigmoid * (1.0 - gate_sigmoid),
                mask=tile_mask,
            )
    else:
        if has_right_value_slot:
            tl.store(
                packed_rows + right_value_slot * hidden_size + channels[None, :],
                grad_factor * gate_sigmoid,
                mask=tile_mask,
            )
        if has_right_gate_slot:
            tl.store(
                packed_rows + right_gate_slot * hidden_size + channels[None, :],
                grad_factor * value * gate_sigmoid * (1.0 - gate_sigmoid),
                mask=tile_mask,
            )


@triton.jit
def _channel_major_layer_norm(
    input_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    mean_ptr,
    rstd_ptr,
    num_positions,
    hidden_size,
    row_blocks,
    eps,
    write_statistics: tl.constexpr,
    write_normalized: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    group = tl.program_id(0) // row_blocks
    rows = (tl.program_id(0) % row_blocks).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    channels = tl.arange(0, block_channels)
    row_mask = rows < num_positions
    channel_mask = channels < hidden_size
    valid = channel_mask[:, None] & row_mask[None, :]
    offsets = group.to(tl.int64) * hidden_size * num_positions + channels[:, None].to(tl.int64) * num_positions
    offsets += rows[None, :]
    value = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(value, axis=0) / hidden_size
    centered = tl.where(valid, value - mean[None, :], 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / hidden_size + eps)
    if write_normalized:
        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
        tl.store(normalized_ptr + offsets, centered * rstd[None, :] * weight[:, None] + bias[:, None], mask=valid)
    if write_statistics:
        tl.store(mean_ptr + group.to(tl.int64) * num_positions + rows, mean, mask=row_mask)
        tl.store(rstd_ptr + group.to(tl.int64) * num_positions + rows, rstd, mask=row_mask)


@triton.jit
def _gated_output_forward(
    normalized_ptr,
    pair_ptr,
    output_weight_ptr,
    output_gate_weight_ptr,
    output_bias_ptr,
    output_gate_bias_ptr,
    output_ptr,
    num_positions,
    pair_size,
    hidden_size,
    row_blocks,
    channel_blocks,
    has_output_bias: tl.constexpr,
    has_output_gate_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    block = tl.program_id(0) // channel_blocks
    group = block // row_blocks
    rows = (block % row_blocks).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    channels = (tl.program_id(0) % channel_blocks) * block_channels + tl.arange(0, block_channels)
    row_mask = rows < num_positions
    channel_mask = channels < pair_size
    normalized_rows = normalized_ptr + group.to(tl.int64) * hidden_size * num_positions
    pair_rows = pair_ptr + group.to(tl.int64) * num_positions * pair_size
    value = _rows_dot_weight(
        normalized_rows,
        1,
        num_positions,
        output_weight_ptr,
        rows,
        row_mask,
        channels,
        channel_mask,
        hidden_size,
        block_reduction,
        input_precision,
    )
    gate = _rows_dot_weight(
        pair_rows,
        pair_size,
        1,
        output_gate_weight_ptr,
        rows,
        row_mask,
        channels,
        channel_mask,
        pair_size,
        block_reduction,
        input_precision,
    )
    value = _add_bias(value, output_bias_ptr, channels, channel_mask, has_output_bias)
    gate = _add_bias(gate, output_gate_bias_ptr, channels, channel_mask, has_output_gate_bias)
    gate_sigmoid = tl.sigmoid(gate)
    tl.store(
        output_ptr + (group.to(tl.int64) * num_positions + rows)[:, None] * pair_size + channels[None, :],
        value * gate_sigmoid,
        mask=row_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _gated_output_backward(
    grad_output_ptr,
    normalized_ptr,
    pair_ptr,
    output_weight_ptr,
    output_gate_weight_ptr,
    output_bias_ptr,
    output_gate_bias_ptr,
    grad_value_ptr,
    grad_packed_ptr,
    num_positions,
    pair_size,
    hidden_size,
    packed_width,
    row_blocks,
    channel_blocks,
    has_output_bias: tl.constexpr,
    has_output_gate_bias: tl.constexpr,
    need_grad_value: tl.constexpr,
    need_grad_gate: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    block = tl.program_id(0) // channel_blocks
    group = block // row_blocks
    rows = (block % row_blocks).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    channels = (tl.program_id(0) % channel_blocks) * block_channels + tl.arange(0, block_channels)
    row_mask = rows < num_positions
    channel_mask = channels < pair_size
    tile_mask = row_mask[:, None] & channel_mask[None, :]
    normalized_rows = normalized_ptr + group.to(tl.int64) * hidden_size * num_positions
    pair_rows = pair_ptr + group.to(tl.int64) * num_positions * pair_size
    if need_grad_gate:
        value = _rows_dot_weight(
            normalized_rows,
            1,
            num_positions,
            output_weight_ptr,
            rows,
            row_mask,
            channels,
            channel_mask,
            hidden_size,
            block_reduction,
            input_precision,
        )
    gate = _rows_dot_weight(
        pair_rows,
        pair_size,
        1,
        output_gate_weight_ptr,
        rows,
        row_mask,
        channels,
        channel_mask,
        pair_size,
        block_reduction,
        input_precision,
    )
    if need_grad_gate:
        value = _add_bias(value, output_bias_ptr, channels, channel_mask, has_output_bias)
    gate = _add_bias(gate, output_gate_bias_ptr, channels, channel_mask, has_output_gate_bias)
    gate_sigmoid = tl.sigmoid(gate)
    positions = (group.to(tl.int64) * num_positions + rows)[:, None]
    grad_output = tl.load(grad_output_ptr + positions * pair_size + channels[None, :], mask=tile_mask, other=0.0).to(
        tl.float32
    )
    if need_grad_value:
        tl.store(grad_value_ptr + positions * pair_size + channels[None, :], grad_output * gate_sigmoid, mask=tile_mask)
    if need_grad_gate:
        tl.store(
            grad_packed_ptr + positions * packed_width + (packed_width - pair_size + channels)[None, :],
            grad_output * value * gate_sigmoid * (1.0 - gate_sigmoid),
            mask=tile_mask,
        )


@triton.jit
def _layer_norm_transpose_backward(
    grad_value_ptr,
    combined_ptr,
    norm_weight_ptr,
    mean_ptr,
    rstd_ptr,
    output_weight_ptr,
    grad_combined_ptr,
    grad_norm_weight_partial_ptr,
    grad_norm_bias_partial_ptr,
    num_groups,
    num_positions,
    pair_size,
    hidden_size,
    row_blocks,
    num_programs,
    need_grad_combined: tl.constexpr,
    need_norm_weight: tl.constexpr,
    need_norm_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
    block_hidden: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    hidden = tl.arange(0, block_hidden)
    hidden_mask = hidden < hidden_size
    if need_grad_combined:
        norm_weight = tl.load(norm_weight_ptr + hidden, mask=hidden_mask, other=1.0).to(tl.float32)
    if need_norm_weight:
        grad_norm_weight = tl.zeros((block_hidden,), tl.float32)
    if need_norm_bias:
        grad_norm_bias = tl.zeros((block_hidden,), tl.float32)
    for block in range(program, num_groups * row_blocks, num_programs):
        group = block // row_blocks
        rows = (block % row_blocks).to(tl.int64) * block_rows + tl.arange(0, block_rows)
        row_mask = rows < num_positions
        positions = group.to(tl.int64) * num_positions + rows
        grad_normalized = tl.zeros((block_rows, block_hidden), tl.float32)
        for start in range(0, tl.cdiv(pair_size, block_channels)):
            channels = start * block_channels + tl.arange(0, block_channels)
            channel_mask = channels < pair_size
            grad_value = tl.load(
                grad_value_ptr + positions[:, None] * pair_size + channels[None, :],
                mask=row_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            output_weight = tl.load(
                output_weight_ptr + channels[:, None] * hidden_size + hidden[None, :],
                mask=channel_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            )
            grad_normalized = tl.dot(grad_value, output_weight, grad_normalized, input_precision=input_precision)
        offsets = group.to(tl.int64) * hidden_size * num_positions + hidden[None, :].to(tl.int64) * num_positions
        offsets += rows[:, None]
        valid = row_mask[:, None] & hidden_mask[None, :]
        if need_grad_combined or need_norm_weight:
            combined = tl.load(combined_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + positions, mask=row_mask, other=0.0)
            rstd = tl.load(rstd_ptr + positions, mask=row_mask, other=0.0)
            normalized = tl.where(hidden_mask[None, :], (combined - mean[:, None]) * rstd[:, None], 0.0)
        if need_grad_combined:
            grad_affine = grad_normalized * norm_weight[None, :]
            grad_mean = tl.sum(grad_affine, axis=1) / hidden_size
            grad_normalized_mean = tl.sum(grad_affine * normalized, axis=1) / hidden_size
            grad_combined = (grad_affine - grad_mean[:, None] - normalized * grad_normalized_mean[:, None]) * rstd[
                :, None
            ]
            tl.store(grad_combined_ptr + offsets, grad_combined, mask=valid)
        if need_norm_weight:
            grad_norm_weight += tl.sum(grad_normalized * normalized, axis=0)
        if need_norm_bias:
            grad_norm_bias += tl.sum(grad_normalized, axis=0)
    partial = program.to(tl.int64) * hidden_size + hidden
    if need_norm_weight:
        tl.store(grad_norm_weight_partial_ptr + partial, grad_norm_weight, mask=hidden_mask)
    if need_norm_bias:
        tl.store(grad_norm_bias_partial_ptr + partial, grad_norm_bias, mask=hidden_mask)


@triton.jit
def _segmented_grouped_gemm(
    left_ptr,
    right_ptr,
    sample_row_offsets_ptr,
    output_ptr,
    num_samples,
    total_positions,
    hidden_size,
    num_programs,
    transpose_left: tl.constexpr,
    transpose_right: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
    fp16_accumulation: tl.constexpr,
) -> None:

    tile_index = tl.program_id(0).to(tl.int64)
    last_problem_end = tile_index * 0
    cell_start = tile_index * 0
    total_positions = total_positions.to(tl.int64)
    hidden_size = hidden_size.to(tl.int64)

    for sample in range(0, num_samples):
        row_start = tl.load(sample_row_offsets_ptr + sample).to(tl.int64)
        row_end = tl.load(sample_row_offsets_ptr + sample + 1).to(tl.int64)
        size = row_end - row_start
        num_row_tiles = tl.cdiv(size, block_rows)
        num_column_tiles = tl.cdiv(size, block_columns)
        tiles_per_matrix = num_row_tiles * num_column_tiles
        problem_end = last_problem_end + hidden_size * tiles_per_matrix

        while tile_index >= last_problem_end and tile_index < problem_end:
            local_tile = tile_index - last_problem_end
            channel = local_tile // tiles_per_matrix
            matrix_tile = local_tile - channel * tiles_per_matrix
            row_tile = matrix_tile // num_column_tiles
            column_tile = matrix_tile - row_tile * num_column_tiles
            rows = row_tile * block_rows + tl.arange(0, block_rows).to(tl.int64)
            columns = column_tile * block_columns + tl.arange(0, block_columns).to(tl.int64)
            row_mask = rows < size
            column_mask = columns < size
            if fp16_accumulation:
                accumulator = tl.zeros((block_rows, block_columns), dtype=tl.float16)
            else:
                accumulator = tl.zeros((block_rows, block_columns), dtype=tl.float32)
            channel_base = channel * total_positions + cell_start
            for reduction_start in range(0, size, block_reduction):
                reduction = reduction_start + tl.arange(0, block_reduction).to(tl.int64)
                reduction_mask = reduction < size
                if transpose_left:
                    left_offsets = reduction[None, :] * size + rows[:, None]
                else:
                    left_offsets = rows[:, None] * size + reduction[None, :]
                if transpose_right:
                    right_offsets = columns[None, :] * size + reduction[:, None]
                else:
                    right_offsets = reduction[:, None] * size + columns[None, :]
                left = tl.load(
                    left_ptr + channel_base + left_offsets,
                    mask=row_mask[:, None] & reduction_mask[None, :],
                    other=0.0,
                )
                right = tl.load(
                    right_ptr + channel_base + right_offsets,
                    mask=reduction_mask[:, None] & column_mask[None, :],
                    other=0.0,
                )
                if fp16_accumulation:
                    accumulator += tl.dot(left, right, input_precision=input_precision, out_dtype=tl.float16)
                else:
                    accumulator += tl.dot(left, right, input_precision=input_precision)
            tl.store(
                output_ptr + channel_base + rows[:, None] * size + columns[None, :],
                accumulator,
                mask=row_mask[:, None] & column_mask[None, :],
            )
            tile_index += num_programs

        last_problem_end = problem_end
        cell_start += size * size


@triton.jit
def _column_sum_partial(
    input_ptr,
    partials_ptr,
    num_rows,
    width,
    row_stride,
    num_partials,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
) -> None:

    partition = tl.program_id(1)
    columns = tl.program_id(0) * block_columns + tl.arange(0, block_columns)
    column_mask = columns < width
    partial = tl.zeros((block_columns,), tl.float32)
    for block in range(partition, tl.cdiv(num_rows, block_rows), num_partials):
        rows = tl.cast(block, tl.int64) * block_rows + tl.arange(0, block_rows)
        values = tl.load(
            input_ptr + rows[:, None] * row_stride + columns[None, :],
            mask=(rows[:, None] < num_rows) & column_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partial += tl.sum(values, axis=0)
    tl.store(partials_ptr + partition.to(tl.int64) * width + columns, partial, mask=column_mask)
