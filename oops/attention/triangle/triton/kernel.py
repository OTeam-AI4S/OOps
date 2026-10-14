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

from ...flash.triton.kernel import LOG2E, MASKED_LOGIT, online_softmax_step, softmax_backward_dscores

SAMPLE_TABLE_WIDTH = 7
ROW_ALIGNMENT = 16
_TABLE_WIDTH = tl.constexpr(SAMPLE_TABLE_WIDTH)
_ROW_ALIGNMENT = tl.constexpr(ROW_ALIGNMENT)


@triton.jit
def _extent(row_ptr, column: tl.constexpr, aligned: tl.constexpr):

    value = tl.load(row_ptr + column)
    if aligned:
        return tl.multiple_of(value, _ROW_ALIGNMENT)
    return value


@triton.jit
def _sample(sample_table_ptr, sample, aligned: tl.constexpr):
    row = sample_table_ptr + sample * _TABLE_WIDTH
    cell_start = _extent(row, 0, aligned)
    cols = _extent(row, 1, aligned)
    groups = tl.load(row + 2)
    bias_start = _extent(row, 3, aligned)
    bias_groups = tl.load(row + 4)
    bias_group_stride = _extent(row, 5, aligned)
    bias_head_stride = _extent(row, 6, aligned)
    return cell_start, cols, groups, bias_start, bias_groups, bias_group_stride, bias_head_stride


@triton.jit
def _in_range(index, extent, even: tl.constexpr):

    if even:
        return tl.full(index.shape, 1, tl.int1)
    else:
        return index < extent


@triton.jit
def _first_cell(cell_start, group, cols, transposed: tl.constexpr):

    if transposed:
        return cell_start + group
    return cell_start + group * cols


@triton.jit
def _token_stride(groups, transposed: tl.constexpr):

    if transposed:
        return groups
    else:
        return tl.full((), 1, tl.int64)


@triton.jit
def _row_pointers(base_ptr, first_cell, tokens, token_stride, head, num_heads, head_size: tl.constexpr):

    cells = first_cell + tokens.to(tl.int64) * token_stride
    return base_ptr + (cells * num_heads + head) * head_size


@triton.jit
def _lse_pointers(lse_ptr, first_cell, tokens, token_stride, head, num_heads):

    return lse_ptr + (first_cell + tokens.to(tl.int64) * token_stride) * num_heads + head


@triton.jit
def _tile(row_pointers, lanes):

    return row_pointers[:, None] + lanes[None, :]


@triton.jit
def _bias_pointers(
    bias_ptr,
    bias_start,
    bias_group,
    bias_group_stride,
    head,
    bias_head_stride,
    queries,
    keys,
    cols,
):

    block = bias_ptr + bias_start + bias_group * bias_group_stride + head * bias_head_stride
    return (block + queries.to(tl.int64) * cols)[:, None] + keys[None, :]


@triton.jit
def _active_keys(mask_pointers, key_mask, has_mask: tl.constexpr):
    if has_mask:
        return key_mask & tl.load(mask_pointers, mask=key_mask, other=0).to(tl.int1)
    return key_mask


@triton.jit
def _masked_scores(
    scores,
    bias_pointers,
    mask_pointers,
    query_mask,
    key_mask,
    scale,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
):

    if has_bias:
        bias = tl.load(bias_pointers, mask=query_mask[:, None] & key_mask[None, :], other=0.0)
        scores = tl.fma(scores, scale, bias.to(tl.float32))
    else:
        scores = scores * scale
    if has_mask:
        keep = tl.load(mask_pointers, mask=key_mask, other=0).to(tl.int1)
        scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
    return tl.where(key_mask[None, :], scores, MASKED_LOGIT)


@triton.jit
def _attend(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    output_ptr,
    lse_ptr,
    sample_table_ptr,
    num_samples,
    num_heads,
    num_programs,
    scale,
    head_size: tl.constexpr,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    qkv_transposed: tl.constexpr,
    mask_transposed: tl.constexpr,
    aligned: tl.constexpr,
    even: tl.constexpr,
    write_lse: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    tile = tl.program_id(0).to(tl.int64)
    tiles_end = tile * 0
    query_lanes = tl.arange(0, block_query)
    key_lanes = tl.arange(0, block_key)
    channels = tl.arange(0, block_channel)
    channel_mask = _in_range(channels, head_size, block_channel == head_size)

    for sample in range(0, num_samples):
        cell_start, cols, groups, bias_start, bias_groups, bias_group_stride, bias_head_stride = _sample(
            sample_table_ptr, sample, aligned
        )
        query_blocks = tl.cdiv(cols, block_query)
        tiles_begin = tiles_end
        tiles_end += groups * num_heads * query_blocks
        cols32 = cols.to(tl.int32)
        token_stride = _token_stride(groups, qkv_transposed)
        key_advance = block_key * token_stride * num_heads * head_size
        mask_stride = _token_stride(groups, mask_transposed)

        while tile >= tiles_begin and tile < tiles_end:
            local = tile - tiles_begin
            query_block = local % query_blocks
            group_head = local // query_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            bias_group = tl.where(bias_groups == 1, 0, group)
            queries = query_block.to(tl.int32) * block_query + query_lanes
            query_mask = _in_range(queries, cols32, even)
            first_cell = _first_cell(cell_start, group, cols, qkv_transposed)
            query_rows = _row_pointers(query_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
            query = tl.load(_tile(query_rows, channels), mask=query_mask[:, None] & channel_mask[None, :], other=0.0)
            key_rows = _row_pointers(key_ptr, first_cell, key_lanes, token_stride, head, num_heads, head_size)
            value_rows = _row_pointers(value_ptr, first_cell, key_lanes, token_stride, head, num_heads, head_size)
            mask_row = mask_ptr + _first_cell(cell_start, group, cols, mask_transposed)
            max_score = tl.full((block_query,), MASKED_LOGIT, tl.float32)
            normalizer = tl.zeros((block_query,), tl.float32)
            accumulator = tl.zeros((block_query, block_channel), tl.float32)

            for key_start in range(0, cols32, block_key):
                key_start = tl.multiple_of(key_start, block_key)
                keys = key_start + key_lanes
                key_mask = _in_range(keys, cols32, even)
                active_keys = _active_keys(mask_row + keys * mask_stride, key_mask, has_mask)
                kv_mask = active_keys[:, None] & channel_mask[None, :]
                key = tl.load(_tile(key_rows, channels), mask=kv_mask, other=0.0)
                value = tl.load(_tile(value_rows, channels), mask=kv_mask, other=0.0)
                scores = _masked_scores(
                    tl.dot(query, tl.trans(key), input_precision=input_precision),
                    _bias_pointers(
                        bias_ptr,
                        bias_start,
                        bias_group,
                        bias_group_stride,
                        head,
                        bias_head_stride,
                        queries,
                        keys,
                        cols,
                    ),
                    mask_row + keys * mask_stride,
                    query_mask,
                    key_mask,
                    scale,
                    has_mask,
                    has_bias,
                )
                accumulator, normalizer, max_score = online_softmax_step(
                    accumulator, normalizer, max_score, scores, value, input_precision
                )
                key_rows += key_advance
                value_rows += key_advance

            # A fully masked row divides by one and returns zero.
            safe_normalizer = tl.where(normalizer == 0.0, 1.0, normalizer)
            output_rows = _row_pointers(output_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
            tl.store(
                _tile(output_rows, channels),
                accumulator / safe_normalizer[:, None],
                mask=query_mask[:, None] & channel_mask[None, :],
            )
            if write_lse:
                lse = tl.where(normalizer == 0.0, 0.0, max_score + tl.log(normalizer))
                lse_rows = _lse_pointers(lse_ptr, first_cell, queries, token_stride, head, num_heads)
                tl.store(lse_rows, lse, mask=query_mask)
            tile += num_programs


@triton.jit
def _forward(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    output_ptr,
    lse_ptr,
    sample_table_ptr,
    num_samples,
    num_heads,
    num_programs,
    scale,
    head_size: tl.constexpr,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    qkv_transposed: tl.constexpr,
    mask_transposed: tl.constexpr,
    aligned: tl.constexpr,
    even: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    _attend(
        query_ptr,
        key_ptr,
        value_ptr,
        mask_ptr,
        bias_ptr,
        output_ptr,
        lse_ptr,
        sample_table_ptr,
        num_samples,
        num_heads,
        num_programs,
        scale,
        head_size,
        has_mask,
        has_bias,
        qkv_transposed,
        mask_transposed,
        aligned,
        even,
        True,
        block_query,
        block_key,
        block_channel,
        input_precision,
    )


@triton.jit
def _inference(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    output_ptr,
    sample_table_ptr,
    num_samples,
    num_heads,
    num_programs,
    scale,
    head_size: tl.constexpr,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    qkv_transposed: tl.constexpr,
    mask_transposed: tl.constexpr,
    aligned: tl.constexpr,
    even: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    _attend(
        query_ptr,
        key_ptr,
        value_ptr,
        mask_ptr,
        bias_ptr,
        output_ptr,
        output_ptr,
        sample_table_ptr,
        num_samples,
        num_heads,
        num_programs,
        scale,
        head_size,
        has_mask,
        has_bias,
        qkv_transposed,
        mask_transposed,
        aligned,
        even,
        False,
        block_query,
        block_key,
        block_channel,
        input_precision,
    )


@triton.jit
def _backward_preprocess(
    output_ptr,
    grad_output_ptr,
    delta_ptr,
    sample_table_ptr,
    num_samples,
    num_heads,
    num_programs,
    head_size: tl.constexpr,
    qkv_transposed: tl.constexpr,
    aligned: tl.constexpr,
    even: tl.constexpr,
    block_query: tl.constexpr,
    block_channel: tl.constexpr,
) -> None:

    tile = tl.program_id(0).to(tl.int64)
    tiles_end = tile * 0
    query_lanes = tl.arange(0, block_query)
    channels = tl.arange(0, block_channel)
    channel_mask = _in_range(channels, head_size, block_channel == head_size)

    for sample in range(0, num_samples):
        cell_start, cols, groups, _bias_start, _bias_groups, _bias_group_stride, _bias_head_stride = _sample(
            sample_table_ptr, sample, aligned
        )
        query_blocks = tl.cdiv(cols, block_query)
        tiles_begin = tiles_end
        tiles_end += groups * num_heads * query_blocks
        cols32 = cols.to(tl.int32)
        token_stride = _token_stride(groups, qkv_transposed)

        while tile >= tiles_begin and tile < tiles_end:
            local = tile - tiles_begin
            query_block = local % query_blocks
            group_head = local // query_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            queries = query_block.to(tl.int32) * block_query + query_lanes
            query_mask = _in_range(queries, cols32, even)
            load_mask = query_mask[:, None] & channel_mask[None, :]
            first_cell = _first_cell(cell_start, group, cols, qkv_transposed)
            output_rows = _row_pointers(output_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
            grad_rows = _row_pointers(grad_output_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
            output = tl.load(_tile(output_rows, channels), mask=load_mask, other=0.0)
            grad_output = tl.load(_tile(grad_rows, channels), mask=load_mask, other=0.0)
            delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), axis=1)
            delta_rows = _lse_pointers(delta_ptr, first_cell, queries, token_stride, head, num_heads)
            tl.store(delta_rows, delta, mask=query_mask)
            tile += num_programs


@triton.jit
def _backward_query(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    lse_ptr,
    output_ptr,
    delta_ptr,
    grad_output_ptr,
    grad_query_ptr,
    grad_bias_ptr,
    sample_table_ptr,
    num_samples,
    num_heads,
    num_programs,
    scale,
    head_size: tl.constexpr,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    qkv_transposed: tl.constexpr,
    mask_transposed: tl.constexpr,
    aligned: tl.constexpr,
    even: tl.constexpr,
    need_grad_query: tl.constexpr,
    need_grad_full_bias: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    tile = tl.program_id(0).to(tl.int64)
    tiles_end = tile * 0
    query_lanes = tl.arange(0, block_query)
    key_lanes = tl.arange(0, block_key)
    channels = tl.arange(0, block_channel)
    channel_mask = _in_range(channels, head_size, block_channel == head_size)

    for sample in range(0, num_samples):
        cell_start, cols, groups, bias_start, bias_groups, bias_group_stride, bias_head_stride = _sample(
            sample_table_ptr, sample, aligned
        )
        query_blocks = tl.cdiv(cols, block_query)
        tiles_begin = tiles_end
        tiles_end += groups * num_heads * query_blocks
        cols32 = cols.to(tl.int32)
        token_stride = _token_stride(groups, qkv_transposed)
        key_advance = block_key * token_stride * num_heads * head_size
        mask_stride = _token_stride(groups, mask_transposed)

        while tile >= tiles_begin and tile < tiles_end:
            local = tile - tiles_begin
            query_block = local % query_blocks
            group_head = local // query_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            bias_group = tl.where(bias_groups == 1, 0, group)
            queries = query_block.to(tl.int32) * block_query + query_lanes
            query_mask = _in_range(queries, cols32, even)
            load_mask = query_mask[:, None] & channel_mask[None, :]
            first_cell = _first_cell(cell_start, group, cols, qkv_transposed)
            query_rows = _row_pointers(query_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
            grad_rows = _row_pointers(grad_output_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
            query = tl.load(_tile(query_rows, channels), mask=load_mask, other=0.0)
            grad_output = tl.load(_tile(grad_rows, channels), mask=load_mask, other=0.0)
            lse_rows = _lse_pointers(lse_ptr, first_cell, queries, token_stride, head, num_heads)
            delta_rows = _lse_pointers(delta_ptr, first_cell, queries, token_stride, head, num_heads)
            lse = tl.load(lse_rows, mask=query_mask, other=0.0)
            output_rows = _row_pointers(output_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
            output = tl.load(_tile(output_rows, channels), mask=load_mask, other=0.0)
            delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), axis=1)
            tl.store(delta_rows, delta, mask=query_mask)
            key_rows = _row_pointers(key_ptr, first_cell, key_lanes, token_stride, head, num_heads, head_size)
            value_rows = _row_pointers(value_ptr, first_cell, key_lanes, token_stride, head, num_heads, head_size)
            mask_row = mask_ptr + _first_cell(cell_start, group, cols, mask_transposed)
            grad_query = tl.zeros((block_query, block_channel), tl.float32)

            for key_start in range(0, cols32, block_key):
                key_start = tl.multiple_of(key_start, block_key)
                keys = key_start + key_lanes
                key_mask = _in_range(keys, cols32, even)
                active_keys = _active_keys(mask_row + keys * mask_stride, key_mask, has_mask)
                kv_mask = active_keys[:, None] & channel_mask[None, :]
                key = tl.load(_tile(key_rows, channels), mask=kv_mask, other=0.0)
                value = tl.load(_tile(value_rows, channels), mask=kv_mask, other=0.0)
                bias_pointers = _bias_pointers(
                    bias_ptr,
                    bias_start,
                    bias_group,
                    bias_group_stride,
                    head,
                    bias_head_stride,
                    queries,
                    keys,
                    cols,
                )
                scores = _masked_scores(
                    tl.dot(query, tl.trans(key), input_precision=input_precision),
                    bias_pointers,
                    mask_row + keys * mask_stride,
                    query_mask,
                    key_mask,
                    scale,
                    has_mask,
                    has_bias,
                )
                _probabilities, grad_scores = softmax_backward_dscores(
                    scores, lse, grad_output, value, delta, input_precision
                )
                grad_scores = tl.where(active_keys[None, :], grad_scores, 0.0)
                if need_grad_query:
                    grad_query = tl.dot(grad_scores.to(key.dtype), key, grad_query, input_precision=input_precision)
                if need_grad_full_bias:
                    tl.store(
                        _bias_pointers(
                            grad_bias_ptr,
                            bias_start,
                            bias_group,
                            bias_group_stride,
                            head,
                            bias_head_stride,
                            queries,
                            keys,
                            cols,
                        ),
                        grad_scores,
                        mask=query_mask[:, None] & key_mask[None, :] & (bias_groups != 1),
                    )
                key_rows += key_advance
                value_rows += key_advance

            if need_grad_query:
                grad_query_rows = _row_pointers(
                    grad_query_ptr, first_cell, queries, token_stride, head, num_heads, head_size
                )
                tl.store(_tile(grad_query_rows, channels), grad_query * scale, mask=load_mask)
            tile += num_programs


@triton.jit
def _backward_key_value(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    grad_output_ptr,
    grad_key_ptr,
    grad_value_ptr,
    sample_table_ptr,
    num_samples,
    num_heads,
    num_programs,
    scale,
    head_size: tl.constexpr,
    has_mask: tl.constexpr,
    has_bias: tl.constexpr,
    qkv_transposed: tl.constexpr,
    mask_transposed: tl.constexpr,
    aligned: tl.constexpr,
    even: tl.constexpr,
    need_grad_key: tl.constexpr,
    need_grad_value: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    tile = tl.program_id(0).to(tl.int64)
    tiles_end = tile * 0
    query_lanes = tl.arange(0, block_query)
    key_lanes = tl.arange(0, block_key)
    channels = tl.arange(0, block_channel)
    channel_mask = _in_range(channels, head_size, block_channel == head_size)

    for sample in range(0, num_samples):
        cell_start, cols, groups, bias_start, bias_groups, bias_group_stride, bias_head_stride = _sample(
            sample_table_ptr, sample, aligned
        )
        key_blocks = tl.cdiv(cols, block_key)
        tiles_begin = tiles_end
        tiles_end += groups * num_heads * key_blocks
        cols32 = cols.to(tl.int32)
        token_stride = _token_stride(groups, qkv_transposed)
        query_advance = block_query * token_stride * num_heads * head_size
        statistic_advance = block_query * token_stride * num_heads
        mask_stride = _token_stride(groups, mask_transposed)

        while tile >= tiles_begin and tile < tiles_end:
            local = tile - tiles_begin
            key_block = local % key_blocks
            group_head = local // key_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            bias_group = tl.where(bias_groups == 1, 0, group)
            keys = key_block.to(tl.int32) * block_key + key_lanes
            key_mask = _in_range(keys, cols32, even)
            kv_mask = key_mask[:, None] & channel_mask[None, :]
            first_cell = _first_cell(cell_start, group, cols, qkv_transposed)
            key_rows = _row_pointers(key_ptr, first_cell, keys, token_stride, head, num_heads, head_size)
            value_rows = _row_pointers(value_ptr, first_cell, keys, token_stride, head, num_heads, head_size)
            mask_pointers = mask_ptr + _first_cell(cell_start, group, cols, mask_transposed) + keys * mask_stride
            active_keys = _active_keys(mask_pointers, key_mask, has_mask)
            kv_load_mask = active_keys[:, None] & channel_mask[None, :]
            key = tl.load(_tile(key_rows, channels), mask=kv_load_mask, other=0.0)
            value = tl.load(_tile(value_rows, channels), mask=kv_load_mask, other=0.0)
            query_rows = _row_pointers(query_ptr, first_cell, query_lanes, token_stride, head, num_heads, head_size)
            grad_rows = _row_pointers(
                grad_output_ptr, first_cell, query_lanes, token_stride, head, num_heads, head_size
            )
            lse_rows = _lse_pointers(lse_ptr, first_cell, query_lanes, token_stride, head, num_heads)
            delta_rows = _lse_pointers(delta_ptr, first_cell, query_lanes, token_stride, head, num_heads)
            grad_key = tl.zeros((block_key, block_channel), tl.float32)
            grad_value = tl.zeros((block_key, block_channel), tl.float32)

            for query_start in range(0, cols32, block_query):
                query_start = tl.multiple_of(query_start, block_query)
                queries = query_start + query_lanes
                query_mask = _in_range(queries, cols32, even)
                load_mask = query_mask[:, None] & channel_mask[None, :]
                query = tl.load(_tile(query_rows, channels), mask=load_mask, other=0.0)
                grad_output = tl.load(_tile(grad_rows, channels), mask=load_mask, other=0.0)
                lse = tl.load(lse_rows, mask=query_mask, other=0.0)
                delta = tl.load(delta_rows, mask=query_mask, other=0.0)
                scores = _masked_scores(
                    tl.dot(query, tl.trans(key), input_precision=input_precision),
                    _bias_pointers(
                        bias_ptr,
                        bias_start,
                        bias_group,
                        bias_group_stride,
                        head,
                        bias_head_stride,
                        queries,
                        keys,
                        cols,
                    ),
                    mask_pointers,
                    query_mask,
                    key_mask,
                    scale,
                    has_mask,
                    has_bias,
                )
                if need_grad_key:
                    probabilities, grad_scores = softmax_backward_dscores(
                        scores, lse, grad_output, value, delta, input_precision
                    )
                    grad_scores = tl.where(active_keys[None, :], grad_scores, 0.0)
                    grad_key = tl.dot(
                        tl.trans(grad_scores.to(query.dtype)), query, grad_key, input_precision=input_precision
                    )
                else:
                    probabilities = tl.exp2(scores * LOG2E - (lse * LOG2E)[:, None])
                probabilities = tl.where(active_keys[None, :], probabilities, 0.0)
                if need_grad_value:
                    grad_value = tl.dot(
                        tl.trans(probabilities.to(grad_output.dtype)),
                        grad_output,
                        grad_value,
                        input_precision=input_precision,
                    )
                query_rows += query_advance
                grad_rows += query_advance
                lse_rows += statistic_advance
                delta_rows += statistic_advance

            if need_grad_key:
                grad_key_rows = _row_pointers(grad_key_ptr, first_cell, keys, token_stride, head, num_heads, head_size)
                tl.store(
                    _tile(grad_key_rows, channels),
                    tl.where(active_keys[:, None], grad_key * scale, 0.0),
                    mask=kv_mask,
                )
            if need_grad_value:
                grad_value_rows = _row_pointers(
                    grad_value_ptr, first_cell, keys, token_stride, head, num_heads, head_size
                )
                tl.store(
                    _tile(grad_value_rows, channels),
                    tl.where(active_keys[:, None], grad_value, 0.0),
                    mask=kv_mask,
                )
            tile += num_programs


@triton.jit
def _backward_shared_bias(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    grad_output_ptr,
    grad_bias_ptr,
    sample_table_ptr,
    num_samples,
    num_heads,
    num_programs,
    scale,
    group_parts,
    partial_stride,
    head_size: tl.constexpr,
    has_mask: tl.constexpr,
    qkv_transposed: tl.constexpr,
    mask_transposed: tl.constexpr,
    aligned: tl.constexpr,
    even: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    tile = tl.program_id(0).to(tl.int64)
    tiles_end = tile * 0
    query_lanes = tl.arange(0, block_query)
    key_lanes = tl.arange(0, block_key)
    channels = tl.arange(0, block_channel)
    channel_mask = _in_range(channels, head_size, block_channel == head_size)

    for sample in range(0, num_samples):
        cell_start, cols, groups, bias_start, bias_groups, bias_group_stride, bias_head_stride = _sample(
            sample_table_ptr, sample, aligned
        )
        query_blocks = tl.cdiv(cols, block_query)
        key_blocks = tl.cdiv(cols, block_key)
        tiles_begin = tiles_end
        tiles_end += tl.where(bias_groups == 1, group_parts * num_heads * query_blocks * key_blocks, 0)
        cols32 = cols.to(tl.int32)
        token_stride = _token_stride(groups, qkv_transposed)
        mask_stride = _token_stride(groups, mask_transposed)

        while tile >= tiles_begin and tile < tiles_end:
            local = tile - tiles_begin
            key_block = local % key_blocks
            local = local // key_blocks
            query_block = local % query_blocks
            local = local // query_blocks
            head = local % num_heads
            part = local // num_heads
            queries = query_block.to(tl.int32) * block_query + query_lanes
            keys = key_block.to(tl.int32) * block_key + key_lanes
            query_mask = _in_range(queries, cols32, even)
            key_mask = _in_range(keys, cols32, even)
            load_mask = query_mask[:, None] & channel_mask[None, :]
            score_mask = query_mask[:, None] & key_mask[None, :]
            bias_pointers = _bias_pointers(
                bias_ptr, bias_start, 0, bias_group_stride, head, bias_head_stride, queries, keys, cols
            )
            bias = tl.load(bias_pointers, mask=score_mask, other=0.0).to(tl.float32)
            grad_bias = tl.zeros((block_query, block_key), tl.float32)

            for group in range(part * groups // group_parts, (part + 1) * groups // group_parts):
                first_cell = _first_cell(cell_start, group, cols, qkv_transposed)
                mask_pointers = mask_ptr + _first_cell(cell_start, group, cols, mask_transposed) + keys * mask_stride
                active_keys = _active_keys(mask_pointers, key_mask, has_mask)
                kv_load_mask = active_keys[:, None] & channel_mask[None, :]
                query_rows = _row_pointers(query_ptr, first_cell, queries, token_stride, head, num_heads, head_size)
                key_rows = _row_pointers(key_ptr, first_cell, keys, token_stride, head, num_heads, head_size)
                value_rows = _row_pointers(value_ptr, first_cell, keys, token_stride, head, num_heads, head_size)
                grad_rows = _row_pointers(
                    grad_output_ptr, first_cell, queries, token_stride, head, num_heads, head_size
                )
                element_bytes: tl.constexpr = 4 if query_ptr.dtype.element_ty == tl.float32 else 2
                operand_bytes: tl.constexpr = 2 * (block_query + block_key) * block_channel * element_bytes
                stream_compensated: tl.constexpr = (
                    query_ptr.dtype.element_ty == tl.float32 and input_precision == "tf32x3" and operand_bytes >= 65536
                )
                stream_native: tl.constexpr = query_ptr.dtype.element_ty != tl.float32 and (
                    operand_bytes >= 98304 or block_channel >= 256
                )
                if stream_compensated or stream_native:
                    dot_scores = tl.zeros((block_query, block_key), tl.float32)
                    grad_probabilities = tl.zeros((block_query, block_key), tl.float32)
                    for channel_start in tl.range(0, head_size, 32, loop_unroll_factor=1):
                        tile_channels = channel_start + tl.arange(0, 32)
                        tile_channel_mask = tile_channels < head_size
                        q_chunk = tl.load(
                            _tile(query_rows, tile_channels),
                            mask=query_mask[:, None] & tile_channel_mask[None, :],
                            other=0.0,
                        )
                        k_chunk = tl.load(
                            _tile(key_rows, tile_channels),
                            mask=active_keys[:, None] & tile_channel_mask[None, :],
                            other=0.0,
                        )
                        dot_scores += tl.dot(q_chunk, tl.trans(k_chunk), input_precision=input_precision)
                        do_chunk = tl.load(
                            _tile(grad_rows, tile_channels),
                            mask=query_mask[:, None] & tile_channel_mask[None, :],
                            other=0.0,
                        )
                        v_chunk = tl.load(
                            _tile(value_rows, tile_channels),
                            mask=active_keys[:, None] & tile_channel_mask[None, :],
                            other=0.0,
                        )
                        grad_probabilities += tl.dot(do_chunk, tl.trans(v_chunk), input_precision=input_precision)
                else:
                    query = tl.load(_tile(query_rows, channels), mask=load_mask, other=0.0)
                    key = tl.load(_tile(key_rows, channels), mask=kv_load_mask, other=0.0)
                    value = tl.load(_tile(value_rows, channels), mask=kv_load_mask, other=0.0)
                    grad_output = tl.load(_tile(grad_rows, channels), mask=load_mask, other=0.0)
                    dot_scores = tl.dot(query, tl.trans(key), input_precision=input_precision)
                    grad_probabilities = tl.dot(grad_output, tl.trans(value), input_precision=input_precision)
                lse_rows = _lse_pointers(lse_ptr, first_cell, queries, token_stride, head, num_heads)
                delta_rows = _lse_pointers(delta_ptr, first_cell, queries, token_stride, head, num_heads)
                lse = tl.load(lse_rows, mask=query_mask, other=0.0)
                delta = tl.load(delta_rows, mask=query_mask, other=0.0)
                scores = tl.fma(dot_scores, scale, bias)
                if has_mask:
                    scores = tl.where(active_keys[None, :], scores, MASKED_LOGIT)
                scores = tl.where(key_mask[None, :], scores, MASKED_LOGIT)
                probabilities = tl.exp2(scores * LOG2E - (lse * LOG2E)[:, None])
                grad_scores = probabilities * (grad_probabilities.to(tl.float32) - delta[:, None])
                grad_scores = tl.where(active_keys[None, :], grad_scores, 0.0)
                grad_bias += tl.where(query_mask[:, None], grad_scores, 0.0)

            tl.store(
                _bias_pointers(
                    grad_bias_ptr + part * partial_stride,
                    bias_start,
                    0,
                    bias_group_stride,
                    head,
                    bias_head_stride,
                    queries,
                    keys,
                    cols,
                ),
                grad_bias,
                mask=score_mask,
            )
            tile += num_programs
