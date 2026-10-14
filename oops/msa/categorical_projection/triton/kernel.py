from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(debug=True)
def _categorical_msa_projection(
    token_ptr,
    continuous_ptr,
    weight_ptr,
    output_ptr,
    rows,
    vocabulary_size,
    channels,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    row_offsets = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows).to(tl.int64)
    channel_offsets = tl.program_id(1).to(tl.int64) * block_channels + tl.arange(0, block_channels).to(tl.int64)
    row_mask = row_offsets < rows
    channel_mask = channel_offsets < channels
    mask = row_mask[:, None] & channel_mask[None, :]

    token = tl.load(token_ptr + row_offsets, mask=row_mask, other=0).to(tl.int64)
    token_valid = (token >= 0) & (token < vocabulary_size)
    assert_condition = token_valid | ~row_mask | (tl.program_id(1) != 0)
    tl.device_assert(assert_condition, "categorical_msa_projection token index out of range")
    packed_size = vocabulary_size.to(tl.int64) + 2
    weight_row = weight_ptr + channel_offsets * packed_size
    categorical = tl.load(weight_row[None, :] + token[:, None], mask=mask & token_valid[:, None], other=0.0).to(
        tl.float32
    )

    first = tl.load(continuous_ptr + row_offsets * 2, mask=row_mask, other=0.0).to(tl.float32)
    second = tl.load(continuous_ptr + row_offsets * 2 + 1, mask=row_mask, other=0.0).to(tl.float32)
    if weight_ptr.dtype.element_ty == tl.float32 and block_channels <= 32:
        first_weight = tl.load(weight_row[None, :] + vocabulary_size, mask=mask, other=0.0).to(tl.float32)
        second_weight = tl.load(weight_row[None, :] + vocabulary_size + 1, mask=mask, other=0.0).to(tl.float32)
    else:
        first_weight = tl.load(weight_row + vocabulary_size, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
        second_weight = tl.load(weight_row + vocabulary_size + 1, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    output = categorical + first[:, None] * first_weight + second[:, None] * second_weight

    tl.store(output_ptr + row_offsets[:, None] * channels + channel_offsets[None, :], output, mask=mask)


@triton.jit
def _categorical_msa_projection_weight_backward(
    grad_ptr,
    token_ptr,
    continuous_ptr,
    output_ptr,
    rows: tl.constexpr,
    channels: tl.constexpr,
    features: tl.constexpr,
    grad_stride_row: tl.constexpr,
    grad_stride_channel: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    feature = tl.program_id(0)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    row = tl.arange(0, block_rows)
    vocabulary = features - 2
    if feature < vocabulary:
        token = tl.load(token_ptr + row, row < rows, other=-1)
        coefficient = (token == feature).to(tl.float32)
    else:
        coefficient = tl.load(continuous_ptr + row * 2 + feature - vocabulary, row < rows, other=0).to(tl.float32)
    grad = tl.load(
        grad_ptr + row[:, None] * grad_stride_row + channel[None, :] * grad_stride_channel,
        (row[:, None] < rows) & (channel[None, :] < channels),
        other=0,
    ).to(tl.float32)
    if feature < vocabulary:
        contribution = tl.where(coefficient[:, None] != 0, grad, 0)
    else:
        contribution = grad * coefficient[:, None]
    summed = tl.sum(contribution, axis=0)
    tl.store(output_ptr + channel * features + feature, summed, channel < channels)
