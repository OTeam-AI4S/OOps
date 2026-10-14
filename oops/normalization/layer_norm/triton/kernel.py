from __future__ import annotations

import triton
import triton.language as tl

from ...variance import centered_variance


@triton.jit
def _row_offsets(
    rows,
    columns,
    num_columns: tl.constexpr,
    strided_rows: tl.constexpr,
    inner_rows,
    inner_stride,
    outer_stride,
):
    rows = rows.to(tl.int64)
    if strided_rows:
        starts = (rows // inner_rows) * outer_stride + (rows % inner_rows) * inner_stride
    else:
        starts = rows * num_columns
    return starts[:, None] + columns[None, :]


@triton.jit
def _layer_norm_forward(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    num_rows,
    num_columns: tl.constexpr,
    strided_rows: tl.constexpr,
    inner_rows,
    inner_stride,
    outer_stride,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    write_stats: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
) -> None:
    block = tl.program_id(0).to(tl.int64)
    rows = block * block_rows + tl.arange(0, block_rows)
    columns = tl.arange(0, block_columns)
    row_mask = rows < num_rows
    column_mask = columns < num_columns
    mask = row_mask[:, None] & column_mask[None, :]
    # Widen before rows * num_columns can exceed int32.
    offsets = rows[:, None].to(tl.int64) * num_columns + columns[None, :]
    input_offsets = _row_offsets(rows, columns, num_columns, strided_rows, inner_rows, inner_stride, outer_stride)

    hidden = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(hidden, axis=1) / num_columns
    centered = tl.where(mask, hidden - mean[:, None], 0.0)
    variance = centered_variance(centered, num_columns)
    rstd = tl.rsqrt(variance + eps)

    output = centered * rstd[:, None]
    if has_weight:
        weight = tl.load(weight_ptr + columns, mask=column_mask, other=1.0).to(tl.float32)
        output = output * weight[None, :]
    if has_bias:
        bias = tl.load(bias_ptr + columns, mask=column_mask, other=0.0).to(tl.float32)
        output = output + bias[None, :]

    tl.store(output_ptr + offsets, output, mask=mask)
    if write_stats:
        tl.store(mean_ptr + rows, mean, mask=row_mask)
        tl.store(rstd_ptr + rows, rstd, mask=row_mask)


@triton.jit
def _layer_norm_backward(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    grad_input_ptr,
    grad_affine_partial_ptr,
    num_rows,
    num_columns: tl.constexpr,
    strided_rows: tl.constexpr,
    inner_rows,
    inner_stride,
    outer_stride,
    has_weight: tl.constexpr,
    need_grad_input: tl.constexpr,
    need_grad_weight: tl.constexpr,
    need_grad_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
) -> None:
    block = tl.program_id(0).to(tl.int64)
    rows = block * block_rows + tl.arange(0, block_rows)
    columns = tl.arange(0, block_columns)
    mask = (rows[:, None] < num_rows) & (columns[None, :] < num_columns)
    offsets = rows[:, None].to(tl.int64) * num_columns + columns[None, :]
    input_offsets = _row_offsets(rows, columns, num_columns, strided_rows, inner_rows, inner_stride, outer_stride)

    partial_offsets = block.to(tl.int64) * num_columns + columns
    column_mask = columns < num_columns
    if need_grad_input or need_grad_weight or need_grad_bias:
        grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if need_grad_input or need_grad_weight:
            hidden = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + rows, mask=rows < num_rows, other=0.0).to(tl.float32)[:, None]
            rstd = tl.load(rstd_ptr + rows, mask=rows < num_rows, other=0.0).to(tl.float32)[:, None]
            centered = tl.where(mask, hidden - mean, 0.0)
            normalized = centered * rstd

            if need_grad_input:
                if has_weight:
                    weight = tl.load(weight_ptr + columns, mask=column_mask, other=1.0).to(tl.float32)
                    grad_normalized = grad_output * weight[None, :]
                else:
                    grad_normalized = grad_output

                grad_mean = tl.sum(grad_normalized, axis=1)[:, None] / num_columns
                grad_normalized_mean = tl.sum(grad_normalized * normalized, axis=1)[:, None] / num_columns
                grad_input = (grad_normalized - grad_mean - normalized * grad_normalized_mean) * rstd
                tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

            if need_grad_weight:
                grad_weight = tl.sum(grad_output * normalized, axis=0)
                tl.store(grad_affine_partial_ptr + partial_offsets, grad_weight, mask=column_mask)
        if need_grad_bias:
            grad_bias = tl.sum(grad_output, axis=0)
            tl.store(
                grad_affine_partial_ptr
                + partial_offsets
                + need_grad_weight * tl.cdiv(num_rows, block_rows).to(tl.int64) * num_columns,
                grad_bias,
                mask=column_mask,
            )


@triton.jit
def _layer_norm_backward_wide(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    grad_input_ptr,
    grad_affine_partial_ptr,
    num_rows,
    num_columns: tl.constexpr,
    strided_rows: tl.constexpr,
    inner_rows,
    inner_stride,
    outer_stride,
    has_weight: tl.constexpr,
    need_grad_input: tl.constexpr,
    need_grad_weight: tl.constexpr,
    need_grad_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
) -> None:
    block = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, block_columns)
    column_mask = columns < num_columns
    if need_grad_weight:
        weight_partial = tl.zeros((block_columns,), tl.float32)
    if need_grad_bias:
        bias_partial = tl.zeros((block_columns,), tl.float32)
    for offset in range(block_rows):
        row = block * block_rows + offset
        mask = (row < num_rows) & column_mask
        offsets = row * num_columns + columns
        if strided_rows:
            input_start = (row // inner_rows) * outer_stride + (row % inner_rows) * inner_stride
        else:
            input_start = row * num_columns
        grad = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        hidden = tl.load(input_ptr + input_start + columns, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row, mask=row < num_rows, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row, mask=row < num_rows, other=0.0).to(tl.float32)
        centered = tl.where(mask, hidden - mean, 0.0)
        normalized = centered * rstd
        if need_grad_input:
            if has_weight:
                weight = tl.load(weight_ptr + columns, mask=column_mask, other=1.0).to(tl.float32)
                grad_normalized = grad * weight
            else:
                grad_normalized = grad
            grad_mean = tl.sum(grad_normalized, 0) / num_columns
            grad_normalized_mean = tl.sum(grad_normalized * normalized, 0) / num_columns
            value = (grad_normalized - grad_mean - normalized * grad_normalized_mean) * rstd
            tl.store(grad_input_ptr + offsets, value, mask=mask)
        if need_grad_weight:
            weight_partial += grad * normalized
        if need_grad_bias:
            bias_partial += grad
    partial_offsets = block * num_columns + columns
    if need_grad_weight:
        tl.store(grad_affine_partial_ptr + partial_offsets, weight_partial, mask=column_mask)
    if need_grad_bias:
        tl.store(
            grad_affine_partial_ptr
            + partial_offsets
            + need_grad_weight * tl.cdiv(num_rows, block_rows).to(tl.int64) * num_columns,
            bias_partial,
            mask=column_mask,
        )


@triton.jit
def _layer_norm_affine_backward_reduce(
    affine_partials_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    num_blocks,
    num_columns: tl.constexpr,
    need_weight: tl.constexpr,
    need_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
):
    columns = tl.program_id(0) * block_columns + tl.arange(0, block_columns)
    rows = tl.arange(0, block_rows)
    if need_weight:
        weight_accumulator = tl.zeros((block_rows, block_columns), tl.float32)
    if need_bias:
        bias_accumulator = tl.zeros((block_rows, block_columns), tl.float32)
    for begin in range(0, num_blocks, block_rows):
        partial_rows = begin + rows
        offsets = partial_rows[:, None] * num_columns + columns[None, :]
        mask = (partial_rows < num_blocks)[:, None] & (columns < num_columns)[None, :]
        if need_weight:
            weight_accumulator += tl.load(affine_partials_ptr + offsets, mask=mask, other=0)
        if need_bias:
            bias_accumulator += tl.load(
                affine_partials_ptr + offsets + need_weight * num_blocks.to(tl.int64) * num_columns, mask=mask, other=0
            )
    if need_weight:
        tl.store(grad_weight_ptr + columns, tl.sum(weight_accumulator, 0), mask=columns < num_columns)
    if need_bias:
        tl.store(grad_bias_ptr + columns, tl.sum(bias_accumulator, 0), mask=columns < num_columns)
