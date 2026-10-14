from __future__ import annotations

import triton
import triton.language as tl

from ...flash import MASK_KEY, MASK_PAIR
from ...flash.triton.kernel import MASKED_LOGIT, online_softmax_step, softmax_backward_dscores

_MASK_KEY = tl.constexpr(MASK_KEY)
_MASK_PAIR = tl.constexpr(MASK_PAIR)


@triton.jit
def _segmented_pair_bias_attention_forward(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    output_ptr,
    lse_ptr,
    query_sample_offsets_ptr,
    key_sample_offsets_ptr,
    mask_sample_offsets_ptr,
    bias_sample_offsets_ptr,
    query_token_stride,
    query_group_stride,
    query_head_stride,
    query_channel_stride,
    key_token_stride,
    key_group_stride,
    key_head_stride,
    key_channel_stride,
    value_token_stride,
    value_group_stride,
    value_head_stride,
    value_channel_stride,
    mask_position_stride,
    mask_group_stride,
    bias_position_stride,
    bias_group_stride,
    bias_head_stride,
    output_token_stride,
    output_group_stride,
    output_head_stride,
    output_channel_stride,
    lse_token_stride,
    lse_group_stride,
    lse_head_stride,
    num_samples,
    num_groups,
    num_heads,
    mask_groups,
    bias_groups,
    bias_heads,
    head_size: tl.constexpr,
    num_programs: tl.constexpr,
    scale,
    mask_kind: tl.constexpr,
    has_bias: tl.constexpr,
    write_lse: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    tile_index = tl.program_id(0).to(tl.int64)
    last_problem_end = tile_index * 0
    query_lanes = tl.arange(0, block_query).to(tl.int64)
    key_lanes = tl.arange(0, block_key).to(tl.int64)
    channels = tl.arange(0, block_channel).to(tl.int64)
    channel_mask = channels < head_size

    for sample_index in range(0, num_samples):
        sample = sample_index
        query_start = tl.load(query_sample_offsets_ptr + sample).to(tl.int64)
        query_end = tl.load(query_sample_offsets_ptr + sample + 1).to(tl.int64)
        key_start = tl.load(key_sample_offsets_ptr + sample).to(tl.int64)
        key_end = tl.load(key_sample_offsets_ptr + sample + 1).to(tl.int64)
        query_length = query_end - query_start
        key_length = key_end - key_start
        mask_start = tl.load(mask_sample_offsets_ptr + sample).to(tl.int64)
        bias_start = tl.load(bias_sample_offsets_ptr + sample).to(tl.int64)
        num_query_blocks = tl.cdiv(query_length, block_query)
        problem_end = last_problem_end + num_groups * num_heads * num_query_blocks

        while tile_index >= last_problem_end and tile_index < problem_end:
            local_tile = tile_index - last_problem_end
            query_block = local_tile % num_query_blocks
            group_head = local_tile // num_query_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            mask_group = tl.where(mask_groups == 1, 0, group)
            bias_group = tl.where(bias_groups == 1, 0, group)
            bias_head = tl.where(bias_heads == 1, 0, head)

            queries = query_block * block_query + query_lanes
            query_mask = queries < query_length
            query = tl.load(
                query_ptr
                + (query_start + queries[:, None]) * query_token_stride
                + group * query_group_stride
                + head * query_head_stride
                + channels[None, :] * query_channel_stride,
                mask=query_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            max_score = tl.full((block_query,), MASKED_LOGIT, tl.float32)
            normalizer = tl.zeros((block_query,), tl.float32)
            accumulator = tl.zeros((block_query, block_channel), tl.float32)

            for key_block_start in range(0, key_length, block_key):
                key_block_start = tl.multiple_of(key_block_start, block_key)
                keys = key_block_start + key_lanes
                key_mask = keys < key_length
                key = tl.load(
                    key_ptr
                    + (key_start + keys[:, None]) * key_token_stride
                    + group * key_group_stride
                    + head * key_head_stride
                    + channels[None, :] * key_channel_stride,
                    mask=key_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                value = tl.load(
                    value_ptr
                    + (key_start + keys[:, None]) * value_token_stride
                    + group * value_group_stride
                    + head * value_head_stride
                    + channels[None, :] * value_channel_stride,
                    mask=key_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(query, tl.trans(key), input_precision=input_precision) * scale
                if has_bias:
                    bias_positions = bias_start + queries[:, None] * key_length + keys[None, :]
                    bias = tl.load(
                        bias_ptr
                        + bias_positions * bias_position_stride
                        + bias_group * bias_group_stride
                        + bias_head * bias_head_stride,
                        mask=query_mask[:, None] & key_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    scores += bias
                if mask_kind == _MASK_KEY:
                    keep = tl.load(
                        mask_ptr + (mask_start + keys) * mask_position_stride + mask_group * mask_group_stride,
                        mask=key_mask,
                        other=0,
                    ).to(tl.int1)
                    scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
                elif mask_kind == _MASK_PAIR:
                    mask_positions = mask_start + queries[:, None] * key_length + keys[None, :]
                    keep = tl.load(
                        mask_ptr + mask_positions * mask_position_stride + mask_group * mask_group_stride,
                        mask=query_mask[:, None] & key_mask[None, :],
                        other=0,
                    ).to(tl.int1)
                    scores = tl.where(keep, scores, MASKED_LOGIT)
                scores = tl.where(key_mask[None, :], scores, MASKED_LOGIT)
                accumulator, normalizer, max_score = online_softmax_step(
                    accumulator,
                    normalizer,
                    max_score,
                    scores,
                    value,
                    input_precision,
                )

            safe_normalizer = tl.where(normalizer == 0.0, 1.0, normalizer)
            output = accumulator / safe_normalizer[:, None]
            tl.store(
                output_ptr
                + (query_start + queries[:, None]) * output_token_stride
                + group * output_group_stride
                + head * output_head_stride
                + channels[None, :] * output_channel_stride,
                output,
                mask=query_mask[:, None] & channel_mask[None, :],
            )
            if write_lse:
                lse_value = tl.where(normalizer == 0.0, 0.0, max_score + tl.log(normalizer))
                tl.store(
                    lse_ptr
                    + (query_start + queries) * lse_token_stride
                    + group * lse_group_stride
                    + head * lse_head_stride,
                    lse_value,
                    mask=query_mask,
                )
            tile_index += num_programs

        last_problem_end = problem_end


@triton.jit
def _segmented_pair_bias_attention_backward_preprocess(
    output_ptr,
    grad_output_ptr,
    delta_ptr,
    query_sample_offsets_ptr,
    output_token_stride,
    output_group_stride,
    output_head_stride,
    output_channel_stride,
    grad_output_token_stride,
    grad_output_group_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    delta_token_stride,
    delta_group_stride,
    delta_head_stride,
    num_samples,
    num_groups,
    num_heads,
    head_size: tl.constexpr,
    num_programs: tl.constexpr,
    block_query: tl.constexpr,
    block_channel: tl.constexpr,
) -> None:
    tile_index = tl.program_id(0).to(tl.int64)
    last_problem_end = tile_index * 0
    query_lanes = tl.arange(0, block_query).to(tl.int64)
    channels = tl.arange(0, block_channel).to(tl.int64)
    channel_mask = channels < head_size

    for sample_index in range(0, num_samples):
        sample = sample_index
        query_start = tl.load(query_sample_offsets_ptr + sample).to(tl.int64)
        query_end = tl.load(query_sample_offsets_ptr + sample + 1).to(tl.int64)
        query_length = query_end - query_start
        num_query_blocks = tl.cdiv(query_length, block_query)
        problem_end = last_problem_end + num_groups * num_heads * num_query_blocks

        while tile_index >= last_problem_end and tile_index < problem_end:
            local_tile = tile_index - last_problem_end
            query_block = local_tile % num_query_blocks
            group_head = local_tile // num_query_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            queries = query_block * block_query + query_lanes
            query_mask = queries < query_length
            output = tl.load(
                output_ptr
                + (query_start + queries[:, None]) * output_token_stride
                + group * output_group_stride
                + head * output_head_stride
                + channels[None, :] * output_channel_stride,
                mask=query_mask[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_output_ptr
                + (query_start + queries[:, None]) * grad_output_token_stride
                + group * grad_output_group_stride
                + head * grad_output_head_stride
                + channels[None, :] * grad_output_channel_stride,
                mask=query_mask[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            delta = tl.sum(output * grad_output, axis=1)
            tl.store(
                delta_ptr
                + (query_start + queries) * delta_token_stride
                + group * delta_group_stride
                + head * delta_head_stride,
                delta,
                mask=query_mask,
            )
            tile_index += num_programs

        last_problem_end = problem_end


@triton.jit
def _segmented_pair_bias_attention_backward_query(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    grad_output_ptr,
    grad_query_ptr,
    grad_bias_ptr,
    query_sample_offsets_ptr,
    key_sample_offsets_ptr,
    mask_sample_offsets_ptr,
    bias_sample_offsets_ptr,
    query_token_stride,
    query_group_stride,
    query_head_stride,
    query_channel_stride,
    key_token_stride,
    key_group_stride,
    key_head_stride,
    key_channel_stride,
    value_token_stride,
    value_group_stride,
    value_head_stride,
    value_channel_stride,
    mask_position_stride,
    mask_group_stride,
    bias_position_stride,
    bias_group_stride,
    bias_head_stride,
    lse_token_stride,
    lse_group_stride,
    lse_head_stride,
    delta_token_stride,
    delta_group_stride,
    delta_head_stride,
    grad_output_token_stride,
    grad_output_group_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    grad_query_token_stride,
    grad_query_group_stride,
    grad_query_head_stride,
    grad_query_channel_stride,
    grad_bias_position_stride,
    grad_bias_group_stride,
    grad_bias_head_stride,
    num_samples,
    num_groups,
    num_heads,
    mask_groups,
    bias_groups,
    bias_heads,
    head_size: tl.constexpr,
    num_programs: tl.constexpr,
    scale,
    mask_kind: tl.constexpr,
    has_bias: tl.constexpr,
    need_grad_query: tl.constexpr,
    need_grad_bias: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    tile_index = tl.program_id(0).to(tl.int64)
    last_problem_end = tile_index * 0
    query_lanes = tl.arange(0, block_query).to(tl.int64)
    key_lanes = tl.arange(0, block_key).to(tl.int64)
    channels = tl.arange(0, block_channel).to(tl.int64)
    channel_mask = channels < head_size

    for sample_index in range(0, num_samples):
        sample = sample_index
        query_start = tl.load(query_sample_offsets_ptr + sample).to(tl.int64)
        query_end = tl.load(query_sample_offsets_ptr + sample + 1).to(tl.int64)
        key_start = tl.load(key_sample_offsets_ptr + sample).to(tl.int64)
        key_end = tl.load(key_sample_offsets_ptr + sample + 1).to(tl.int64)
        query_length = query_end - query_start
        key_length = key_end - key_start
        mask_start = tl.load(mask_sample_offsets_ptr + sample).to(tl.int64)
        bias_start = tl.load(bias_sample_offsets_ptr + sample).to(tl.int64)
        num_query_blocks = tl.cdiv(query_length, block_query)
        problem_end = last_problem_end + num_groups * num_heads * num_query_blocks

        while tile_index >= last_problem_end and tile_index < problem_end:
            local_tile = tile_index - last_problem_end
            query_block = local_tile % num_query_blocks
            group_head = local_tile // num_query_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            mask_group = tl.where(mask_groups == 1, 0, group)
            bias_group = tl.where(bias_groups == 1, 0, group)
            bias_head = tl.where(bias_heads == 1, 0, head)
            queries = query_block * block_query + query_lanes
            query_mask = queries < query_length
            query = tl.load(
                query_ptr
                + (query_start + queries[:, None]) * query_token_stride
                + group * query_group_stride
                + head * query_head_stride
                + channels[None, :] * query_channel_stride,
                mask=query_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            grad_output = tl.load(
                grad_output_ptr
                + (query_start + queries[:, None]) * grad_output_token_stride
                + group * grad_output_group_stride
                + head * grad_output_head_stride
                + channels[None, :] * grad_output_channel_stride,
                mask=query_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr
                + (query_start + queries) * lse_token_stride
                + group * lse_group_stride
                + head * lse_head_stride,
                mask=query_mask,
                other=0.0,
            ).to(tl.float32)
            delta = tl.load(
                delta_ptr
                + (query_start + queries) * delta_token_stride
                + group * delta_group_stride
                + head * delta_head_stride,
                mask=query_mask,
                other=0.0,
            ).to(tl.float32)
            grad_query = tl.zeros((block_query, block_channel), tl.float32)

            for key_block_start in range(0, key_length, block_key):
                key_block_start = tl.multiple_of(key_block_start, block_key)
                keys = key_block_start + key_lanes
                key_mask = keys < key_length
                key = tl.load(
                    key_ptr
                    + (key_start + keys[:, None]) * key_token_stride
                    + group * key_group_stride
                    + head * key_head_stride
                    + channels[None, :] * key_channel_stride,
                    mask=key_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                value = tl.load(
                    value_ptr
                    + (key_start + keys[:, None]) * value_token_stride
                    + group * value_group_stride
                    + head * value_head_stride
                    + channels[None, :] * value_channel_stride,
                    mask=key_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(query, tl.trans(key), input_precision=input_precision) * scale
                bias_positions = bias_start + queries[:, None] * key_length + keys[None, :]
                if has_bias:
                    bias = tl.load(
                        bias_ptr
                        + bias_positions * bias_position_stride
                        + bias_group * bias_group_stride
                        + bias_head * bias_head_stride,
                        mask=query_mask[:, None] & key_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    scores += bias
                if mask_kind == _MASK_KEY:
                    keep = tl.load(
                        mask_ptr + (mask_start + keys) * mask_position_stride + mask_group * mask_group_stride,
                        mask=key_mask,
                        other=0,
                    ).to(tl.int1)
                    scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
                elif mask_kind == _MASK_PAIR:
                    mask_positions = mask_start + queries[:, None] * key_length + keys[None, :]
                    keep = tl.load(
                        mask_ptr + mask_positions * mask_position_stride + mask_group * mask_group_stride,
                        mask=query_mask[:, None] & key_mask[None, :],
                        other=0,
                    ).to(tl.int1)
                    scores = tl.where(keep, scores, MASKED_LOGIT)
                scores = tl.where(key_mask[None, :], scores, MASKED_LOGIT)
                _probabilities, grad_scores = softmax_backward_dscores(
                    scores,
                    lse,
                    grad_output,
                    value,
                    delta,
                    input_precision,
                )
                if need_grad_query:
                    grad_query += (
                        tl.dot(
                            grad_scores.to(key.dtype),
                            key,
                            input_precision=input_precision,
                        )
                        * scale
                    )
                if need_grad_bias:
                    tl.store(
                        grad_bias_ptr
                        + bias_positions * grad_bias_position_stride
                        + bias_group * grad_bias_group_stride
                        + bias_head * grad_bias_head_stride,
                        grad_scores,
                        mask=query_mask[:, None] & key_mask[None, :],
                    )

            if need_grad_query:
                tl.store(
                    grad_query_ptr
                    + (query_start + queries[:, None]) * grad_query_token_stride
                    + group * grad_query_group_stride
                    + head * grad_query_head_stride
                    + channels[None, :] * grad_query_channel_stride,
                    grad_query,
                    mask=query_mask[:, None] & channel_mask[None, :],
                )
            tile_index += num_programs

        last_problem_end = problem_end


@triton.jit
def _segmented_pair_bias_attention_backward_key_value(
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
    query_sample_offsets_ptr,
    key_sample_offsets_ptr,
    mask_sample_offsets_ptr,
    bias_sample_offsets_ptr,
    query_token_stride,
    query_group_stride,
    query_head_stride,
    query_channel_stride,
    key_token_stride,
    key_group_stride,
    key_head_stride,
    key_channel_stride,
    value_token_stride,
    value_group_stride,
    value_head_stride,
    value_channel_stride,
    mask_position_stride,
    mask_group_stride,
    bias_position_stride,
    bias_group_stride,
    bias_head_stride,
    lse_token_stride,
    lse_group_stride,
    lse_head_stride,
    delta_token_stride,
    delta_group_stride,
    delta_head_stride,
    grad_output_token_stride,
    grad_output_group_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    grad_key_token_stride,
    grad_key_group_stride,
    grad_key_head_stride,
    grad_key_channel_stride,
    grad_value_token_stride,
    grad_value_group_stride,
    grad_value_head_stride,
    grad_value_channel_stride,
    num_samples,
    num_groups,
    num_heads,
    mask_groups,
    bias_groups,
    bias_heads,
    head_size: tl.constexpr,
    num_programs: tl.constexpr,
    scale,
    mask_kind: tl.constexpr,
    has_bias: tl.constexpr,
    need_grad_key: tl.constexpr,
    need_grad_value: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    tile_index = tl.program_id(0).to(tl.int64)
    last_problem_end = tile_index * 0
    query_lanes = tl.arange(0, block_query).to(tl.int64)
    key_lanes = tl.arange(0, block_key).to(tl.int64)
    channels = tl.arange(0, block_channel).to(tl.int64)
    channel_mask = channels < head_size

    for sample_index in range(0, num_samples):
        sample = sample_index
        query_start = tl.load(query_sample_offsets_ptr + sample).to(tl.int64)
        query_end = tl.load(query_sample_offsets_ptr + sample + 1).to(tl.int64)
        key_start = tl.load(key_sample_offsets_ptr + sample).to(tl.int64)
        key_end = tl.load(key_sample_offsets_ptr + sample + 1).to(tl.int64)
        query_length = query_end - query_start
        key_length = key_end - key_start
        mask_start = tl.load(mask_sample_offsets_ptr + sample).to(tl.int64)
        bias_start = tl.load(bias_sample_offsets_ptr + sample).to(tl.int64)
        num_key_blocks = tl.cdiv(key_length, block_key)
        problem_end = last_problem_end + num_groups * num_heads * num_key_blocks

        while tile_index >= last_problem_end and tile_index < problem_end:
            local_tile = tile_index - last_problem_end
            key_block = local_tile % num_key_blocks
            group_head = local_tile // num_key_blocks
            head = group_head % num_heads
            group = group_head // num_heads
            mask_group = tl.where(mask_groups == 1, 0, group)
            bias_group = tl.where(bias_groups == 1, 0, group)
            bias_head = tl.where(bias_heads == 1, 0, head)
            keys = key_block * block_key + key_lanes
            key_mask = keys < key_length
            key = tl.load(
                key_ptr
                + (key_start + keys[:, None]) * key_token_stride
                + group * key_group_stride
                + head * key_head_stride
                + channels[None, :] * key_channel_stride,
                mask=key_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            if need_grad_key:
                value = tl.load(
                    value_ptr
                    + (key_start + keys[:, None]) * value_token_stride
                    + group * value_group_stride
                    + head * value_head_stride
                    + channels[None, :] * value_channel_stride,
                    mask=key_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
            grad_key = tl.zeros((block_key, block_channel), tl.float32)
            grad_value = tl.zeros((block_key, block_channel), tl.float32)

            for query_block_start in range(0, query_length, block_query):
                query_block_start = tl.multiple_of(query_block_start, block_query)
                queries = query_block_start + query_lanes
                query_mask = queries < query_length
                query = tl.load(
                    query_ptr
                    + (query_start + queries[:, None]) * query_token_stride
                    + group * query_group_stride
                    + head * query_head_stride
                    + channels[None, :] * query_channel_stride,
                    mask=query_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                grad_output = tl.load(
                    grad_output_ptr
                    + (query_start + queries[:, None]) * grad_output_token_stride
                    + group * grad_output_group_stride
                    + head * grad_output_head_stride
                    + channels[None, :] * grad_output_channel_stride,
                    mask=query_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                lse = tl.load(
                    lse_ptr
                    + (query_start + queries) * lse_token_stride
                    + group * lse_group_stride
                    + head * lse_head_stride,
                    mask=query_mask,
                    other=0.0,
                ).to(tl.float32)
                scores = tl.dot(query, tl.trans(key), input_precision=input_precision) * scale
                if has_bias:
                    bias_positions = bias_start + queries[:, None] * key_length + keys[None, :]
                    bias = tl.load(
                        bias_ptr
                        + bias_positions * bias_position_stride
                        + bias_group * bias_group_stride
                        + bias_head * bias_head_stride,
                        mask=query_mask[:, None] & key_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    scores += bias
                if mask_kind == _MASK_KEY:
                    keep = tl.load(
                        mask_ptr + (mask_start + keys) * mask_position_stride + mask_group * mask_group_stride,
                        mask=key_mask,
                        other=0,
                    ).to(tl.int1)
                    scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
                elif mask_kind == _MASK_PAIR:
                    mask_positions = mask_start + queries[:, None] * key_length + keys[None, :]
                    keep = tl.load(
                        mask_ptr + mask_positions * mask_position_stride + mask_group * mask_group_stride,
                        mask=query_mask[:, None] & key_mask[None, :],
                        other=0,
                    ).to(tl.int1)
                    scores = tl.where(keep, scores, MASKED_LOGIT)
                scores = tl.where(key_mask[None, :], scores, MASKED_LOGIT)
                if need_grad_key:
                    delta = tl.load(
                        delta_ptr
                        + (query_start + queries) * delta_token_stride
                        + group * delta_group_stride
                        + head * delta_head_stride,
                        mask=query_mask,
                        other=0.0,
                    ).to(tl.float32)
                    probabilities, grad_scores = softmax_backward_dscores(
                        scores,
                        lse,
                        grad_output,
                        value,
                        delta,
                        input_precision,
                    )
                    grad_key += (
                        tl.dot(
                            tl.trans(grad_scores.to(query.dtype)),
                            query,
                            input_precision=input_precision,
                        )
                        * scale
                    )
                else:
                    probabilities = tl.exp(scores - lse[:, None])
                if need_grad_value:
                    grad_value += tl.dot(
                        tl.trans(probabilities.to(grad_output.dtype)),
                        grad_output,
                        input_precision=input_precision,
                    )

            if need_grad_key:
                tl.store(
                    grad_key_ptr
                    + (key_start + keys[:, None]) * grad_key_token_stride
                    + group * grad_key_group_stride
                    + head * grad_key_head_stride
                    + channels[None, :] * grad_key_channel_stride,
                    grad_key,
                    mask=key_mask[:, None] & channel_mask[None, :],
                )
            if need_grad_value:
                tl.store(
                    grad_value_ptr
                    + (key_start + keys[:, None]) * grad_value_token_stride
                    + group * grad_value_group_stride
                    + head * grad_value_head_stride
                    + channels[None, :] * grad_value_channel_stride,
                    grad_value,
                    mask=key_mask[:, None] & channel_mask[None, :],
                )
            tile_index += num_programs

        last_problem_end = problem_end


@triton.jit
def _pair_bias_grad_scores_for_group_head(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    grad_output_ptr,
    query_start,
    key_start,
    mask_start,
    bias_start,
    group,
    head,
    bias_group,
    bias_head,
    query_length,
    key_length,
    queries,
    keys,
    channels,
    query_mask,
    key_mask,
    channel_mask,
    query_token_stride,
    query_group_stride,
    query_head_stride,
    query_channel_stride,
    key_token_stride,
    key_group_stride,
    key_head_stride,
    key_channel_stride,
    value_token_stride,
    value_group_stride,
    value_head_stride,
    value_channel_stride,
    mask_position_stride,
    mask_group_stride,
    bias_position_stride,
    bias_group_stride,
    bias_head_stride,
    lse_token_stride,
    lse_group_stride,
    lse_head_stride,
    delta_token_stride,
    delta_group_stride,
    delta_head_stride,
    grad_output_token_stride,
    grad_output_group_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    mask_group,
    scale,
    mask_kind: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
):
    query = tl.load(
        query_ptr
        + (query_start + queries[:, None]) * query_token_stride
        + group * query_group_stride
        + head * query_head_stride
        + channels[None, :] * query_channel_stride,
        mask=query_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    key = tl.load(
        key_ptr
        + (key_start + keys[:, None]) * key_token_stride
        + group * key_group_stride
        + head * key_head_stride
        + channels[None, :] * key_channel_stride,
        mask=key_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    value = tl.load(
        value_ptr
        + (key_start + keys[:, None]) * value_token_stride
        + group * value_group_stride
        + head * value_head_stride
        + channels[None, :] * value_channel_stride,
        mask=key_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    bias_positions = bias_start + queries[:, None] * key_length + keys[None, :]
    bias = tl.load(
        bias_ptr
        + bias_positions * bias_position_stride
        + bias_group * bias_group_stride
        + bias_head * bias_head_stride,
        mask=query_mask[:, None] & key_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.dot(query, tl.trans(key), input_precision=input_precision) * scale + bias
    if mask_kind == _MASK_KEY:
        keep = tl.load(
            mask_ptr + (mask_start + keys) * mask_position_stride + mask_group * mask_group_stride,
            mask=key_mask,
            other=0,
        ).to(tl.int1)
        scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
    elif mask_kind == _MASK_PAIR:
        mask_positions = mask_start + queries[:, None] * key_length + keys[None, :]
        keep = tl.load(
            mask_ptr + mask_positions * mask_position_stride + mask_group * mask_group_stride,
            mask=query_mask[:, None] & key_mask[None, :],
            other=0,
        ).to(tl.int1)
        scores = tl.where(keep, scores, MASKED_LOGIT)
    scores = tl.where(key_mask[None, :], scores, MASKED_LOGIT)
    lse = tl.load(
        lse_ptr + (query_start + queries) * lse_token_stride + group * lse_group_stride + head * lse_head_stride,
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)
    delta = tl.load(
        delta_ptr
        + (query_start + queries) * delta_token_stride
        + group * delta_group_stride
        + head * delta_head_stride,
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr
        + (query_start + queries[:, None]) * grad_output_token_stride
        + group * grad_output_group_stride
        + head * grad_output_head_stride
        + channels[None, :] * grad_output_channel_stride,
        mask=query_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    _probabilities, grad_scores = softmax_backward_dscores(
        scores,
        lse,
        grad_output,
        value,
        delta,
        input_precision,
    )
    return tl.where(query_mask[:, None] & key_mask[None, :], grad_scores, 0.0)


@triton.jit
def _segmented_pair_bias_attention_grad_bias(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    grad_output_ptr,
    grad_bias_ptr,
    query_sample_offsets_ptr,
    key_sample_offsets_ptr,
    mask_sample_offsets_ptr,
    bias_sample_offsets_ptr,
    query_token_stride,
    query_group_stride,
    query_head_stride,
    query_channel_stride,
    key_token_stride,
    key_group_stride,
    key_head_stride,
    key_channel_stride,
    value_token_stride,
    value_group_stride,
    value_head_stride,
    value_channel_stride,
    mask_position_stride,
    mask_group_stride,
    bias_position_stride,
    bias_group_stride,
    bias_head_stride,
    lse_token_stride,
    lse_group_stride,
    lse_head_stride,
    delta_token_stride,
    delta_group_stride,
    delta_head_stride,
    grad_output_token_stride,
    grad_output_group_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    grad_bias_position_stride,
    grad_bias_group_stride,
    grad_bias_head_stride,
    num_samples,
    num_groups,
    num_heads,
    mask_groups,
    bias_groups,
    bias_heads,
    head_size: tl.constexpr,
    num_programs: tl.constexpr,
    scale,
    mask_kind: tl.constexpr,
    shared_bias_group: tl.constexpr,
    shared_bias_head: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    tile_index = tl.program_id(0).to(tl.int64)
    last_problem_end = tile_index * 0
    query_lanes = tl.arange(0, block_query).to(tl.int64)
    key_lanes = tl.arange(0, block_key).to(tl.int64)
    channels = tl.arange(0, block_channel).to(tl.int64)
    channel_mask = channels < head_size

    for sample_index in range(0, num_samples):
        sample = sample_index
        query_start = tl.load(query_sample_offsets_ptr + sample).to(tl.int64)
        query_end = tl.load(query_sample_offsets_ptr + sample + 1).to(tl.int64)
        key_start = tl.load(key_sample_offsets_ptr + sample).to(tl.int64)
        key_end = tl.load(key_sample_offsets_ptr + sample + 1).to(tl.int64)
        query_length = query_end - query_start
        key_length = key_end - key_start
        mask_start = tl.load(mask_sample_offsets_ptr + sample).to(tl.int64)
        bias_start = tl.load(bias_sample_offsets_ptr + sample).to(tl.int64)
        num_query_blocks = tl.cdiv(query_length, block_query)
        num_key_blocks = tl.cdiv(key_length, block_key)
        problem_end = last_problem_end + bias_groups * bias_heads * num_query_blocks * num_key_blocks

        while tile_index >= last_problem_end and tile_index < problem_end:
            local_tile = tile_index - last_problem_end
            key_block = local_tile % num_key_blocks
            query_head = local_tile // num_key_blocks
            query_block = query_head % num_query_blocks
            bias_group_head = query_head // num_query_blocks
            bias_head = bias_group_head % bias_heads
            bias_group = bias_group_head // bias_heads
            queries = query_block * block_query + query_lanes
            keys = key_block * block_key + key_lanes
            query_mask = queries < query_length
            key_mask = keys < key_length
            grad_bias = tl.zeros((block_query, block_key), tl.float32)

            if shared_bias_group:
                for group_index in range(0, num_groups):
                    group = group_index
                    mask_group = tl.where(mask_groups == 1, 0, group)
                    if shared_bias_head:
                        for head_index in range(0, num_heads):
                            grad_bias += _pair_bias_grad_scores_for_group_head(
                                query_ptr,
                                key_ptr,
                                value_ptr,
                                mask_ptr,
                                bias_ptr,
                                lse_ptr,
                                delta_ptr,
                                grad_output_ptr,
                                query_start,
                                key_start,
                                mask_start,
                                bias_start,
                                group,
                                head_index,
                                bias_group,
                                bias_head,
                                query_length,
                                key_length,
                                queries,
                                keys,
                                channels,
                                query_mask,
                                key_mask,
                                channel_mask,
                                query_token_stride,
                                query_group_stride,
                                query_head_stride,
                                query_channel_stride,
                                key_token_stride,
                                key_group_stride,
                                key_head_stride,
                                key_channel_stride,
                                value_token_stride,
                                value_group_stride,
                                value_head_stride,
                                value_channel_stride,
                                mask_position_stride,
                                mask_group_stride,
                                bias_position_stride,
                                bias_group_stride,
                                bias_head_stride,
                                lse_token_stride,
                                lse_group_stride,
                                lse_head_stride,
                                delta_token_stride,
                                delta_group_stride,
                                delta_head_stride,
                                grad_output_token_stride,
                                grad_output_group_stride,
                                grad_output_head_stride,
                                grad_output_channel_stride,
                                mask_group,
                                scale,
                                mask_kind,
                                block_query,
                                block_key,
                                block_channel,
                                input_precision,
                            )
                    else:
                        grad_bias += _pair_bias_grad_scores_for_group_head(
                            query_ptr,
                            key_ptr,
                            value_ptr,
                            mask_ptr,
                            bias_ptr,
                            lse_ptr,
                            delta_ptr,
                            grad_output_ptr,
                            query_start,
                            key_start,
                            mask_start,
                            bias_start,
                            group,
                            bias_head,
                            bias_group,
                            bias_head,
                            query_length,
                            key_length,
                            queries,
                            keys,
                            channels,
                            query_mask,
                            key_mask,
                            channel_mask,
                            query_token_stride,
                            query_group_stride,
                            query_head_stride,
                            query_channel_stride,
                            key_token_stride,
                            key_group_stride,
                            key_head_stride,
                            key_channel_stride,
                            value_token_stride,
                            value_group_stride,
                            value_head_stride,
                            value_channel_stride,
                            mask_position_stride,
                            mask_group_stride,
                            bias_position_stride,
                            bias_group_stride,
                            bias_head_stride,
                            lse_token_stride,
                            lse_group_stride,
                            lse_head_stride,
                            delta_token_stride,
                            delta_group_stride,
                            delta_head_stride,
                            grad_output_token_stride,
                            grad_output_group_stride,
                            grad_output_head_stride,
                            grad_output_channel_stride,
                            mask_group,
                            scale,
                            mask_kind,
                            block_query,
                            block_key,
                            block_channel,
                            input_precision,
                        )
            elif shared_bias_head:
                group = bias_group
                mask_group = tl.where(mask_groups == 1, 0, group)
                for head_index in range(0, num_heads):
                    grad_bias += _pair_bias_grad_scores_for_group_head(
                        query_ptr,
                        key_ptr,
                        value_ptr,
                        mask_ptr,
                        bias_ptr,
                        lse_ptr,
                        delta_ptr,
                        grad_output_ptr,
                        query_start,
                        key_start,
                        mask_start,
                        bias_start,
                        group,
                        head_index,
                        bias_group,
                        bias_head,
                        query_length,
                        key_length,
                        queries,
                        keys,
                        channels,
                        query_mask,
                        key_mask,
                        channel_mask,
                        query_token_stride,
                        query_group_stride,
                        query_head_stride,
                        query_channel_stride,
                        key_token_stride,
                        key_group_stride,
                        key_head_stride,
                        key_channel_stride,
                        value_token_stride,
                        value_group_stride,
                        value_head_stride,
                        value_channel_stride,
                        mask_position_stride,
                        mask_group_stride,
                        bias_position_stride,
                        bias_group_stride,
                        bias_head_stride,
                        lse_token_stride,
                        lse_group_stride,
                        lse_head_stride,
                        delta_token_stride,
                        delta_group_stride,
                        delta_head_stride,
                        grad_output_token_stride,
                        grad_output_group_stride,
                        grad_output_head_stride,
                        grad_output_channel_stride,
                        mask_group,
                        scale,
                        mask_kind,
                        block_query,
                        block_key,
                        block_channel,
                        input_precision,
                    )

            bias_positions = bias_start + queries[:, None] * key_length + keys[None, :]
            tl.store(
                grad_bias_ptr
                + bias_positions * grad_bias_position_stride
                + bias_group * grad_bias_group_stride
                + bias_head * grad_bias_head_stride,
                grad_bias,
                mask=query_mask[:, None] & key_mask[None, :],
            )
            tile_index += num_programs

        last_problem_end = problem_end
