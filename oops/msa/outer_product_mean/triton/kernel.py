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

import triton
import triton.language as tl


@triton.jit
def _tf32_operand(value, precision: tl.constexpr):
    if precision == "tf32":
        return tl.inline_asm_elementwise(
            "cvt.rna.tf32.f32 $0, $1;", "=f,f", [value], dtype=tl.float32, is_pure=True, pack=1
        )
    return value


@triton.jit
def _outer_product_count_backward(
    grad_ptr,
    numerator_ptr,
    count_ptr,
    bias_ptr,
    out_ptr,
    rows,
    pairs_per_batch,
    grad_batch_stride,
    channels: tl.constexpr,
    eps: tl.constexpr,
    divide_before: tl.constexpr,
    clamp: tl.constexpr,
    has_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows).to(tl.int64)
    batch = row // pairs_per_batch
    local_row = row % pairs_per_batch
    total = tl.zeros((block_rows,), tl.float32)
    for start in range(0, channels, block_channels):
        channel = start + tl.arange(0, block_channels)
        valid = (row[:, None] < rows) & (channel[None, :] < channels)
        grad = tl.load(
            grad_ptr + batch[:, None] * grad_batch_stride + local_row[:, None] * channels + channel[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        numerator = tl.load(
            numerator_ptr + row[:, None] * channels + channel[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        if has_bias and not divide_before:
            bias = tl.load(bias_ptr + channel, mask=channel < channels, other=0.0).to(tl.float32)
            numerator += bias[None, :]
        total += tl.sum(grad * numerator, axis=1)
    count = tl.load(count_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    denominator = tl.maximum(count, 1.0) if divide_before else count + eps
    if clamp:
        denominator = tl.maximum(denominator, 1.0)
    result = -total / (denominator * denominator)
    if divide_before:
        result = tl.where(count >= 1.0, result, 0.0)
    elif clamp:
        result = tl.where(count + eps >= 1.0, result, 0.0)
    tl.store(out_ptr + row, result, mask=row < rows)


@triton.jit
def _msa_dual_projection_forward(
    msa_ptr,
    norm_w_ptr,
    norm_b_ptr,
    left_w_ptr,
    right_w_ptr,
    left_b_ptr,
    right_b_ptr,
    left_out_ptr,
    right_out_ptr,
    normalized_ptr,
    mean_ptr,
    rstd_ptr,
    save_stats: tl.constexpr,
    rows,
    K,
    H,
    msa_depth,
    tokens,
    stride_batch,
    stride_depth,
    stride_token,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
    block_k: tl.constexpr,
    block_h: tl.constexpr,
    has_left_bias: tl.constexpr,
    has_right_bias: tl.constexpr,
    input_precision: tl.constexpr,
    contiguous_rows: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows).to(tl.int64)
    hidden = tl.program_id(1).to(tl.int64) * block_h + tl.arange(0, block_h).to(tl.int64)
    row_mask = row < rows
    hidden_mask = hidden < H
    if contiguous_rows:
        row_offset = row * K
    else:
        sequence_row = row // tokens
        row_offset = (
            sequence_row // msa_depth * stride_batch
            + sequence_row % msa_depth * stride_depth
            + row % tokens * stride_token
        )
    row_base = msa_ptr + row_offset[:, None]

    origin = tl.load(msa_ptr + row_offset, mask=row_mask, other=0.0).to(tl.float32)
    total = tl.zeros((block_rows,), dtype=tl.float32)
    for k0 in range(0, K, block_k):
        k = k0 + tl.arange(0, block_k)
        k_mask = k < K
        # Masked lanes need another select after centering: zero minus the mean is not zero.
        x_float = tl.load(row_base + k[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        shifted = tl.where(k_mask[None, :], x_float - origin[:, None], 0.0)
        total += tl.sum(shifted, axis=1)
    mean = total / K
    sum_squares = tl.zeros((block_rows,), dtype=tl.float32)
    for k0 in range(0, K, block_k):
        k = k0 + tl.arange(0, block_k)
        k_mask = k < K
        x_float = tl.load(row_base + k[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        centered = tl.where(k_mask[None, :], (x_float - origin[:, None]) - mean[:, None], 0.0)
        sum_squares += tl.sum(centered * centered, axis=1)
    variance = sum_squares / K
    rstd = tl.rsqrt(variance + eps)
    if save_stats:
        tl.store(mean_ptr + row, mean, mask=row_mask)
        tl.store(rstd_ptr + row, rstd, mask=row_mask)

    left = tl.zeros((block_rows, block_h), dtype=tl.float32)
    right = tl.zeros((block_rows, block_h), dtype=tl.float32)
    for k0 in range(0, K, block_k):
        k = k0 + tl.arange(0, block_k)
        k_mask = k < K
        x = tl.load(row_base + k[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        centered = tl.where(k_mask[None, :], (x.to(tl.float32) - origin[:, None]) - mean[:, None], 0.0)
        norm_w = tl.load(norm_w_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        norm_b = tl.load(norm_b_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        left_w = tl.load(
            left_w_ptr + hidden[:, None] * K + k[None, :],
            mask=hidden_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        right_w = tl.load(
            right_w_ptr + hidden[:, None] * K + k[None, :],
            mask=hidden_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        x_norm = (centered * rstd[:, None] * norm_w[None, :] + norm_b[None, :]).to(left_w.dtype)
        if save_stats:
            tl.store(normalized_ptr + row[:, None] * K + k[None, :], x_norm, mask=row_mask[:, None] & k_mask[None, :])
        dot_norm = _tf32_operand(x_norm, input_precision)
        left += tl.dot(dot_norm, tl.trans(_tf32_operand(left_w, input_precision)), input_precision=input_precision)
        right += tl.dot(dot_norm, tl.trans(_tf32_operand(right_w, input_precision)), input_precision=input_precision)

    if has_left_bias:
        left_bias = tl.load(left_b_ptr + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
        left += left_bias[None, :]
    if has_right_bias:
        right_bias = tl.load(right_b_ptr + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
        right += right_bias[None, :]

    out_offsets = row[:, None].to(tl.int64) * H + hidden[None, :]
    tl.store(left_out_ptr + out_offsets, left, mask=row_mask[:, None] & hidden_mask[None, :])
    tl.store(right_out_ptr + out_offsets, right, mask=row_mask[:, None] & hidden_mask[None, :])


@triton.jit
def _project_outer_mean_forward(
    outer_ptr,
    count_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    numerator_ptr,
    save_projection: tl.constexpr,
    total_pairs,
    tokens,
    hidden: tl.constexpr,
    hidden_side: tl.constexpr,
    channels: tl.constexpr,
    denominator_eps: tl.constexpr,
    divide_before_projection: tl.constexpr,
    clamp_denominator: tl.constexpr,
    bias_kind: tl.constexpr,
    block_pairs: tl.constexpr,
    block_channels: tl.constexpr,
    block_hidden: tl.constexpr,
    row_aligned: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    lane = tl.arange(0, block_pairs).to(tl.int64)
    if row_aligned:
        row_j = tl.program_id(0).to(tl.int64) * block_pairs + lane
        pair_row = tl.program_id(1).to(tl.int64)
        pair_offsets = pair_row * tokens + row_j
        channel_program = tl.program_id(2)
        pair_mask = row_j < tokens
        batch = pair_row // tokens
        local_i = pair_row - batch * tokens
        native_pair_base = pair_row * tokens * hidden + row_j * hidden_side
    else:
        pair_offsets = tl.program_id(0).to(tl.int64) * block_pairs + lane
        channel_program = tl.program_id(1)
        pair_mask = pair_offsets < total_pairs
        batch = pair_offsets // tokens // tokens
        pair_in_batch = pair_offsets - batch * tokens * tokens
        local_i = pair_in_batch // tokens
        row_j = pair_in_batch - local_i * tokens
        native_pair_base = (batch * tokens + local_i).to(tl.int64) * tokens * hidden + row_j.to(tl.int64) * hidden_side
    channel_offsets = channel_program.to(tl.int64) * block_channels + tl.arange(0, block_channels).to(tl.int64)
    channel_mask = channel_offsets < channels

    row_i = local_i
    global_pair = batch.to(tl.int64) * tokens * tokens + row_i.to(tl.int64) * tokens + row_j

    acc = tl.zeros((block_pairs, block_channels), tl.float32)
    for hidden_start in range(0, hidden, block_hidden):
        hidden_offsets = hidden_start + tl.arange(0, block_hidden)
        hidden_mask = hidden_offsets < hidden
        hidden_left = hidden_offsets // hidden_side
        hidden_right = hidden_offsets - hidden_left * hidden_side
        outer_offsets = (
            native_pair_base[:, None] + hidden_left[None, :].to(tl.int64) * tokens * hidden_side + hidden_right[None, :]
        )
        weight_offsets = channel_offsets[None, :].to(tl.int64) * hidden + hidden_offsets[:, None]
        outer = tl.load(outer_ptr + outer_offsets, mask=pair_mask[:, None] & hidden_mask[None, :], other=0.0)
        weight = tl.load(weight_ptr + weight_offsets, mask=hidden_mask[:, None] & channel_mask[None, :], other=0.0)
        acc += tl.dot(
            _tf32_operand(outer.to(weight.dtype), input_precision),
            _tf32_operand(weight, input_precision),
            out_dtype=tl.float32,
            input_precision=input_precision,
        )

    # Round the projection before its epilogue, as at the public activation boundary.
    acc = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    if save_projection:
        tl.store(
            numerator_ptr + global_pair[:, None] * channels + channel_offsets[None, :],
            acc,
            mask=pair_mask[:, None] & channel_mask[None, :],
        )
    count = tl.load(count_ptr + global_pair, mask=pair_mask, other=0.0).to(tl.float32)
    if divide_before_projection:
        acc = acc / tl.maximum(count[:, None], 1.0)
        if bias_kind == 1:
            bias = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]
    else:
        if bias_kind == 1:
            bias = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]
        denominator = count + denominator_eps
        if clamp_denominator:
            denominator = tl.maximum(denominator, 1.0)
        acc = acc / denominator[:, None]

    out_offsets = global_pair[:, None].to(tl.int64) * channels + channel_offsets[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=pair_mask[:, None] & channel_mask[None, :])


@triton.jit
def _msa_projection_normalization_backward(
    gradient_ptr,
    msa_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    grad_msa_ptr,
    partial_ptr,
    rows,
    channels: tl.constexpr,
    eps: tl.constexpr,
    need_msa: tl.constexpr,
    need_weight: tl.constexpr,
    need_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    row = block * block_rows + tl.arange(0, block_rows)
    column = tl.arange(0, block_columns)
    mask = (row[:, None] < rows) & (column[None, :] < channels)
    gradient = tl.load(gradient_ptr + row[:, None] * channels + column[None, :], mask=mask, other=0).to(tl.float32)
    if need_msa or need_weight:
        values = tl.load(msa_ptr + row[:, None] * channels + column[None, :], mask=mask, other=0).to(tl.float32)
        origin = tl.load(msa_ptr + row * channels, mask=row < rows, other=0).to(tl.float32)
        shifted = tl.where(mask, values - origin[:, None], 0.0)
        mean = tl.load(mean_ptr + row, mask=row < rows, other=0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row, mask=row < rows, other=0).to(tl.float32)
        centered = tl.where(mask, shifted - mean[:, None], 0.0)
        normalized = centered * rstd[:, None]
        if need_msa:
            weight = tl.load(weight_ptr + column, mask=column < channels, other=0).to(tl.float32)
            scaled = gradient * weight[None, :]
            mean_gradient = tl.sum(scaled, axis=1) / channels
            mean_normalized = tl.sum(scaled * normalized, axis=1) / channels
            result = (scaled - mean_gradient[:, None] - normalized * mean_normalized[:, None]) * rstd[:, None]
            tl.store(grad_msa_ptr + row[:, None] * channels + column[None, :], result, mask=mask)
        if need_weight:
            partial_weight = tl.sum(gradient * normalized, axis=0)
            tl.store(partial_ptr + block * channels + column, partial_weight, mask=column < channels)
    if need_bias:
        partial_bias = tl.sum(gradient, axis=0)
        offset = need_weight * tl.cdiv(tl.cast(rows, tl.int64), block_rows) * channels
        tl.store(partial_ptr + offset + block * channels + column, partial_bias, mask=column < channels)


@triton.jit
def _msa_projection_normalization_affine_reduce(
    partial_ptr,
    weight_ptr,
    bias_ptr,
    blocks,
    channels: tl.constexpr,
    need_weight: tl.constexpr,
    need_bias: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
):
    row = tl.arange(0, block_rows).to(tl.int64)
    column = tl.program_id(0) * block_columns + tl.arange(0, block_columns)
    if need_weight:
        weight_accumulator = tl.zeros((block_rows, block_columns), tl.float32)
    if need_bias:
        bias_accumulator = tl.zeros((block_rows, block_columns), tl.float32)
    for begin in range(0, blocks, block_rows):
        partial_row = begin + row
        offsets = partial_row[:, None] * channels + column[None, :]
        mask = (partial_row[:, None] < blocks) & (column[None, :] < channels)
        if need_weight:
            weight_accumulator += tl.load(partial_ptr + offsets, mask=mask, other=0)
        if need_bias:
            bias_accumulator += tl.load(
                partial_ptr + offsets + need_weight * tl.cast(blocks, tl.int64) * channels, mask=mask, other=0
            )
    if need_weight:
        tl.store(weight_ptr + column, tl.sum(weight_accumulator, axis=0), mask=column < channels)
    if need_bias:
        tl.store(bias_ptr + column, tl.sum(bias_accumulator, axis=0), mask=column < channels)
