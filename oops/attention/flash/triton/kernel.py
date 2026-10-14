from __future__ import annotations

import triton
import triton.language as tl

from ..mask import MASK_KEY, MASK_PAIR

MASKED_LOGIT = tl.constexpr(-float("inf"))

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def online_softmax_step(accumulator, normalizer, max_score, scores, value, input_precision: tl.constexpr):
    r"""Running maxima and LSE use natural-log units; exponentials use base two."""
    block_max = tl.max(scores, axis=1)
    new_max = tl.maximum(max_score, block_max)
    # Clamp only the exponential reference to avoid -inf - -inf on fully masked rows.
    safe_max_log2 = tl.where(new_max == float("-inf"), 0.0, new_max) * LOG2E
    old_scale = tl.exp2(max_score * LOG2E - safe_max_log2)
    probabilities = tl.exp2(scores * LOG2E - safe_max_log2[:, None])
    accumulator = accumulator * old_scale[:, None] + tl.dot(
        probabilities.to(value.dtype),
        value,
        input_precision=input_precision,
    )
    normalizer = normalizer * old_scale + tl.sum(probabilities, axis=1)
    return accumulator, normalizer, new_max


@triton.jit
def softmax_backward_dscores(scores, lse, grad_output, value, delta, input_precision: tl.constexpr):
    r"""``P = exp(scores - LSE)``; ``dScores = P * (dP - delta)``."""
    probabilities = tl.exp2(scores * LOG2E - (lse * LOG2E)[:, None])
    grad_probabilities = tl.dot(
        grad_output,
        tl.trans(value),
        input_precision=input_precision,
    ).to(tl.float32)
    grad_scores = probabilities * (grad_probabilities - delta[:, None])
    return probabilities, grad_scores


_MASK_KEY = tl.constexpr(MASK_KEY)
_MASK_PAIR = tl.constexpr(MASK_PAIR)


@triton.jit
def _load_tile(pointers, mask, other, even: tl.constexpr):
    if even:
        return tl.load(pointers)
    return tl.load(pointers, mask=mask, other=other)


# Widen roots before stride multiplication; widening a wrapped product is too late.


@triton.jit
def _flash_attention_forward(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    output_ptr,
    lse_ptr,
    query_batch_stride,
    query_group_stride,
    query_token_stride,
    query_head_stride,
    query_channel_stride,
    key_batch_stride,
    key_group_stride,
    key_token_stride,
    key_head_stride,
    key_channel_stride,
    value_batch_stride,
    value_group_stride,
    value_token_stride,
    value_head_stride,
    value_channel_stride,
    mask_batch_stride,
    mask_group_stride,
    mask_query_stride,
    mask_key_stride,
    bias_batch_stride,
    bias_group_stride,
    bias_head_stride,
    bias_query_stride,
    bias_key_stride,
    output_batch_stride,
    output_group_stride,
    output_token_stride,
    output_head_stride,
    output_channel_stride,
    lse_batch_stride,
    lse_group_stride,
    lse_head_stride,
    lse_token_stride,
    num_groups,
    query_length,
    key_length,
    num_heads,
    head_size: tl.constexpr,
    bias_groups,
    scale,
    mask_kind: tl.constexpr,
    has_bias: tl.constexpr,
    write_lse: tl.constexpr,
    even_query: tl.constexpr,
    even_key: tl.constexpr,
    even_head_size: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:
    batch_group_head = tl.program_id(0).to(tl.int64)
    query_block = tl.program_id(1)
    batch = batch_group_head // (num_groups * num_heads)
    group_head = batch_group_head - batch * num_groups * num_heads
    group = group_head // num_heads
    head = group_head - group * num_heads
    bias_group = tl.where(bias_groups == 1, 0, group)

    # Slice bases use int64; tile-local offsets must fit int32.
    query_offsets = query_block * block_query + tl.arange(0, block_query)
    key_offsets = tl.arange(0, block_key)
    channel_offsets = tl.arange(0, block_channel)
    query_mask = query_offsets < query_length
    channel_mask = channel_offsets < head_size

    q_base = (
        query_ptr
        + (batch * query_batch_stride + group * query_group_stride + head * query_head_stride)
        + query_offsets[:, None] * query_token_stride
        + channel_offsets[None, :] * query_channel_stride
    )
    if even_query & even_head_size:
        query = tl.load(q_base)
    elif even_head_size:
        query = tl.load(q_base, mask=query_mask[:, None], other=0.0)
    elif even_query:
        query = tl.load(q_base, mask=channel_mask[None, :], other=0.0)
    else:
        query = tl.load(q_base, mask=query_mask[:, None] & channel_mask[None, :], other=0.0)

    k_base = (
        key_ptr
        + (batch * key_batch_stride + group * key_group_stride + head * key_head_stride)
        + key_offsets[:, None] * key_token_stride
        + channel_offsets[None, :] * key_channel_stride
    )
    v_base = (
        value_ptr
        + (batch * value_batch_stride + group * value_group_stride + head * value_head_stride)
        + key_offsets[:, None] * value_token_stride
        + channel_offsets[None, :] * value_channel_stride
    )
    b_base = (
        bias_ptr
        + (batch * bias_batch_stride + bias_group * bias_group_stride + head * bias_head_stride)
        + query_offsets[:, None] * bias_query_stride
        + key_offsets[None, :] * bias_key_stride
    )
    m_base = mask_ptr + (batch * mask_batch_stride + group * mask_group_stride)

    max_score = tl.full((block_query,), MASKED_LOGIT, tl.float32)
    normalizer = tl.zeros((block_query,), tl.float32)
    accumulator = tl.zeros((block_query, block_channel), tl.float32)

    for key_start in range(0, key_length, block_key):
        key_start = tl.multiple_of(key_start, block_key)
        keys = key_start + key_offsets
        key_mask = keys < key_length
        k_ptrs = k_base + key_start * key_token_stride
        v_ptrs = v_base + key_start * value_token_stride
        if even_key & even_head_size:
            key = tl.load(k_ptrs)
            value = tl.load(v_ptrs)
        elif even_head_size:
            key = tl.load(k_ptrs, mask=key_mask[:, None], other=0.0)
            value = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        elif even_key:
            key = tl.load(k_ptrs, mask=channel_mask[None, :], other=0.0)
            value = tl.load(v_ptrs, mask=channel_mask[None, :], other=0.0)
        else:
            kv_mask = key_mask[:, None] & channel_mask[None, :]
            key = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            value = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        scores = tl.dot(query, tl.trans(key), input_precision=input_precision)
        if has_bias:
            b_ptrs = b_base + key_start * bias_key_stride
            if even_query & even_key:
                bias = tl.load(b_ptrs)
            else:
                bias = tl.load(b_ptrs, mask=query_mask[:, None] & key_mask[None, :], other=0.0)
            scores = tl.fma(scores, scale, bias.to(tl.float32))
        else:
            scores = scores * scale

        if mask_kind == _MASK_KEY:
            if even_key:
                keep = tl.load(m_base + keys * mask_key_stride).to(tl.int1)
            else:
                keep = tl.load(m_base + keys * mask_key_stride, mask=key_mask, other=0).to(tl.int1)
            scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
        elif mask_kind == _MASK_PAIR:
            keep = tl.load(
                m_base + query_offsets[:, None] * mask_query_stride + keys[None, :] * mask_key_stride,
                mask=query_mask[:, None] & key_mask[None, :],
                other=0,
            ).to(tl.int1)
            scores = tl.where(keep, scores, MASKED_LOGIT)

        # Out-of-range keys must not enter softmax; query tails are masked at stores.
        if not even_key:
            scores = tl.where(key_mask[None, :], scores, MASKED_LOGIT)
        accumulator, normalizer, max_score = online_softmax_step(
            accumulator, normalizer, max_score, scores, value, input_precision
        )

    # A fully masked row divides by one and returns zero.
    safe_normalizer = tl.where(normalizer == 0.0, 1.0, normalizer)
    output = accumulator / safe_normalizer[:, None]
    o_base = (
        output_ptr
        + (batch * output_batch_stride + group * output_group_stride + head * output_head_stride)
        + query_offsets[:, None] * output_token_stride
        + channel_offsets[None, :] * output_channel_stride
    )
    if even_query & even_head_size:
        tl.store(o_base, output)
    elif even_head_size:
        tl.store(o_base, output, mask=query_mask[:, None])
    elif even_query:
        tl.store(o_base, output, mask=channel_mask[None, :])
    else:
        tl.store(o_base, output, mask=query_mask[:, None] & channel_mask[None, :])
    if write_lse:
        lse_base = (
            lse_ptr
            + (batch * lse_batch_stride + group * lse_group_stride + head * lse_head_stride)
            + query_offsets * lse_token_stride
        )
        # Finite LSE makes the masked backward exponential zero rather than NaN.
        lse_value = tl.where(normalizer == 0.0, 0.0, max_score + tl.log(normalizer))
        if even_query:
            tl.store(lse_base, lse_value)
        else:
            tl.store(lse_base, lse_value, mask=query_mask)


@triton.jit
def _flash_attention_backward_preprocess(
    output_ptr,
    grad_output_ptr,
    delta_ptr,
    output_batch_stride,
    output_group_stride,
    output_token_stride,
    output_head_stride,
    output_channel_stride,
    grad_output_batch_stride,
    grad_output_group_stride,
    grad_output_token_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    delta_batch_stride,
    delta_group_stride,
    delta_head_stride,
    delta_token_stride,
    num_groups,
    query_length,
    num_heads,
    head_size: tl.constexpr,
    even_query: tl.constexpr,
    even_head_size: tl.constexpr,
    block_query: tl.constexpr,
    block_channel: tl.constexpr,
) -> None:

    batch_group_head = tl.program_id(0).to(tl.int64)
    query_block = tl.program_id(1).to(tl.int64)
    batch = batch_group_head // (num_groups * num_heads)
    group_head = batch_group_head - batch * num_groups * num_heads
    group = group_head // num_heads
    head = group_head - group * num_heads

    query_offsets = query_block * block_query + tl.arange(0, block_query).to(tl.int64)
    channel_offsets = tl.arange(0, block_channel).to(tl.int64)
    o_base = (
        output_ptr
        + batch * output_batch_stride
        + group * output_group_stride
        + head * output_head_stride
        + query_offsets[:, None] * output_token_stride
        + channel_offsets[None, :] * output_channel_stride
    )
    do_base = (
        grad_output_ptr
        + batch * grad_output_batch_stride
        + group * grad_output_group_stride
        + head * grad_output_head_stride
        + query_offsets[:, None] * grad_output_token_stride
        + channel_offsets[None, :] * grad_output_channel_stride
    )
    if even_query & even_head_size:
        output = tl.load(o_base).to(tl.float32)
        grad_output = tl.load(do_base).to(tl.float32)
    elif even_head_size:
        output = tl.load(o_base, mask=query_offsets[:, None] < query_length, other=0.0).to(tl.float32)
        grad_output = tl.load(do_base, mask=query_offsets[:, None] < query_length, other=0.0).to(tl.float32)
    else:
        load_mask = (query_offsets[:, None] < query_length) & (channel_offsets[None, :] < head_size)
        output = tl.load(o_base, mask=load_mask, other=0.0).to(tl.float32)
        grad_output = tl.load(do_base, mask=load_mask, other=0.0).to(tl.float32)
    delta = tl.sum(output * grad_output, axis=1)
    delta_base = (
        delta_ptr
        + batch * delta_batch_stride
        + group * delta_group_stride
        + head * delta_head_stride
        + query_offsets * delta_token_stride
    )
    if even_query:
        tl.store(delta_base, delta)
    else:
        tl.store(delta_base, delta, mask=query_offsets < query_length)


@triton.jit
def _flash_attention_backward_query(
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
    query_batch_stride,
    query_group_stride,
    query_token_stride,
    query_head_stride,
    query_channel_stride,
    key_batch_stride,
    key_group_stride,
    key_token_stride,
    key_head_stride,
    key_channel_stride,
    value_batch_stride,
    value_group_stride,
    value_token_stride,
    value_head_stride,
    value_channel_stride,
    mask_batch_stride,
    mask_group_stride,
    mask_query_stride,
    mask_key_stride,
    bias_batch_stride,
    bias_group_stride,
    bias_head_stride,
    bias_query_stride,
    bias_key_stride,
    lse_batch_stride,
    lse_group_stride,
    lse_head_stride,
    lse_token_stride,
    grad_output_batch_stride,
    grad_output_group_stride,
    grad_output_token_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    grad_query_batch_stride,
    grad_query_group_stride,
    grad_query_token_stride,
    grad_query_head_stride,
    grad_query_channel_stride,
    grad_bias_batch_stride,
    grad_bias_group_stride,
    grad_bias_head_stride,
    grad_bias_query_stride,
    grad_bias_key_stride,
    num_groups,
    query_length,
    key_length,
    num_heads,
    head_size: tl.constexpr,
    bias_groups,
    scale,
    mask_kind: tl.constexpr,
    has_bias: tl.constexpr,
    need_grad_query: tl.constexpr,
    need_grad_bias: tl.constexpr,
    even_query: tl.constexpr,
    even_key: tl.constexpr,
    even_head_size: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    batch_group_head = tl.program_id(0).to(tl.int64)
    query_block = tl.program_id(1)
    batch = batch_group_head // (num_groups * num_heads)
    group_head = batch_group_head - batch * num_groups * num_heads
    group = group_head // num_heads
    head = group_head - group * num_heads
    bias_group = tl.where(bias_groups == 1, 0, group)
    even_tile: tl.constexpr = even_query & even_head_size

    query_offsets = query_block * block_query + tl.arange(0, block_query)
    key_offsets = tl.arange(0, block_key)
    channel_offsets = tl.arange(0, block_channel)
    query_mask = query_offsets < query_length
    channel_mask = channel_offsets < head_size
    q_mask = query_mask[:, None] & channel_mask[None, :]

    query = _load_tile(
        query_ptr
        + (batch * query_batch_stride + group * query_group_stride + head * query_head_stride)
        + query_offsets[:, None] * query_token_stride
        + channel_offsets[None, :] * query_channel_stride,
        q_mask,
        0.0,
        even_tile,
    )
    grad_output = _load_tile(
        grad_output_ptr
        + (batch * grad_output_batch_stride + group * grad_output_group_stride + head * grad_output_head_stride)
        + query_offsets[:, None] * grad_output_token_stride
        + channel_offsets[None, :] * grad_output_channel_stride,
        q_mask,
        0.0,
        even_tile,
    )
    row_base = (batch * lse_batch_stride + group * lse_group_stride + head * lse_head_stride) + (
        query_offsets * lse_token_stride
    )
    delta = _load_tile(delta_ptr + row_base, query_mask, 0.0, even_query).to(tl.float32)
    lse_log2 = _load_tile(lse_ptr + row_base, query_mask, 0.0, even_query).to(tl.float32) * LOG2E
    if need_grad_query:
        grad_query = tl.zeros((block_query, block_channel), tl.float32)

    k_base = (
        key_ptr
        + (batch * key_batch_stride + group * key_group_stride + head * key_head_stride)
        + key_offsets[:, None] * key_token_stride
        + channel_offsets[None, :] * key_channel_stride
    )
    v_base = (
        value_ptr
        + (batch * value_batch_stride + group * value_group_stride + head * value_head_stride)
        + key_offsets[:, None] * value_token_stride
        + channel_offsets[None, :] * value_channel_stride
    )
    b_base = (
        bias_ptr
        + (batch * bias_batch_stride + bias_group * bias_group_stride + head * bias_head_stride)
        + query_offsets[:, None] * bias_query_stride
        + key_offsets[None, :] * bias_key_stride
    )
    m_base = mask_ptr + (batch * mask_batch_stride + group * mask_group_stride)
    db_base = (
        grad_bias_ptr
        + (batch * grad_bias_batch_stride + bias_group * grad_bias_group_stride + head * grad_bias_head_stride)
        + query_offsets[:, None] * grad_bias_query_stride
        + key_offsets[None, :] * grad_bias_key_stride
    )

    for key_start in range(0, key_length, block_key):
        key_start = tl.multiple_of(key_start, block_key)
        keys = key_start + key_offsets
        key_mask = keys < key_length
        kv_mask = key_mask[:, None] & channel_mask[None, :]
        key = _load_tile(k_base + key_start * key_token_stride, kv_mask, 0.0, even_key & even_head_size)
        value = _load_tile(v_base + key_start * value_token_stride, kv_mask, 0.0, even_key & even_head_size)
        scores = tl.dot(query, tl.trans(key), input_precision=input_precision)
        if has_bias:
            bias = _load_tile(
                b_base + key_start * bias_key_stride,
                query_mask[:, None] & key_mask[None, :],
                MASKED_LOGIT,
                even_query & even_key,
            )
            scores = tl.fma(scores, scale, bias.to(tl.float32))
        else:
            scores = scores * scale
        if mask_kind == _MASK_KEY:
            keep = _load_tile(m_base + keys * mask_key_stride, key_mask, 0, even_key).to(tl.int1)
            scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
        elif mask_kind == _MASK_PAIR:
            keep = tl.load(
                m_base + query_offsets[:, None] * mask_query_stride + keys[None, :] * mask_key_stride,
                mask=query_mask[:, None] & key_mask[None, :],
                other=0,
            ).to(tl.int1)
            scores = tl.where(keep, scores, MASKED_LOGIT)
        if not even_key:
            scores = tl.where(key_mask[None, :], scores, MASKED_LOGIT)
        probabilities = tl.exp2(tl.fma(scores, LOG2E, -lse_log2[:, None]))
        grad_probabilities = tl.dot(grad_output, tl.trans(value), input_precision=input_precision)
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        if need_grad_query:
            grad_query = tl.dot(grad_scores.to(key.dtype), key, grad_query, input_precision=input_precision)
        if need_grad_bias:
            tl.store(
                db_base + key_start * grad_bias_key_stride, grad_scores, mask=query_mask[:, None] & key_mask[None, :]
            )

    if need_grad_query:
        tl.store(
            grad_query_ptr
            + (batch * grad_query_batch_stride + group * grad_query_group_stride + head * grad_query_head_stride)
            + query_offsets[:, None] * grad_query_token_stride
            + channel_offsets[None, :] * grad_query_channel_stride,
            grad_query * scale,
            mask=q_mask,
        )


@triton.jit
def _flash_attention_backward_key_value(
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
    query_batch_stride,
    query_group_stride,
    query_token_stride,
    query_head_stride,
    query_channel_stride,
    key_batch_stride,
    key_group_stride,
    key_token_stride,
    key_head_stride,
    key_channel_stride,
    value_batch_stride,
    value_group_stride,
    value_token_stride,
    value_head_stride,
    value_channel_stride,
    mask_batch_stride,
    mask_group_stride,
    mask_query_stride,
    mask_key_stride,
    bias_batch_stride,
    bias_group_stride,
    bias_head_stride,
    bias_query_stride,
    bias_key_stride,
    lse_batch_stride,
    lse_group_stride,
    lse_head_stride,
    lse_token_stride,
    grad_output_batch_stride,
    grad_output_group_stride,
    grad_output_token_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    grad_key_batch_stride,
    grad_key_group_stride,
    grad_key_token_stride,
    grad_key_head_stride,
    grad_key_channel_stride,
    grad_value_batch_stride,
    grad_value_group_stride,
    grad_value_token_stride,
    grad_value_head_stride,
    grad_value_channel_stride,
    num_groups,
    query_length,
    key_length,
    num_heads,
    head_size: tl.constexpr,
    bias_groups,
    scale,
    mask_kind: tl.constexpr,
    has_bias: tl.constexpr,
    bias_transposed: tl.constexpr,
    need_grad_key: tl.constexpr,
    need_grad_value: tl.constexpr,
    even_query: tl.constexpr,
    even_key: tl.constexpr,
    even_head_size: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
) -> None:

    batch_group_head = tl.program_id(0).to(tl.int64)
    key_block = tl.program_id(1)
    batch = batch_group_head // (num_groups * num_heads)
    group_head = batch_group_head - batch * num_groups * num_heads
    group = group_head // num_heads
    head = group_head - group * num_heads
    bias_group = tl.where(bias_groups == 1, 0, group)
    even_tile: tl.constexpr = even_query & even_head_size

    query_offsets = tl.arange(0, block_query)
    key_offsets = key_block * block_key + tl.arange(0, block_key)
    channel_offsets = tl.arange(0, block_channel)
    key_mask = key_offsets < key_length
    channel_mask = channel_offsets < head_size
    kv_mask = key_mask[:, None] & channel_mask[None, :]

    key = _load_tile(
        key_ptr
        + (batch * key_batch_stride + group * key_group_stride + head * key_head_stride)
        + key_offsets[:, None] * key_token_stride
        + channel_offsets[None, :] * key_channel_stride,
        kv_mask,
        0.0,
        even_key & even_head_size,
    )
    if need_grad_key:
        value = _load_tile(
            value_ptr
            + (batch * value_batch_stride + group * value_group_stride + head * value_head_stride)
            + key_offsets[:, None] * value_token_stride
            + channel_offsets[None, :] * value_channel_stride,
            kv_mask,
            0.0,
            even_key & even_head_size,
        )
        grad_key = tl.zeros((block_key, block_channel), tl.float32)
    if need_grad_value:
        grad_value = tl.zeros((block_key, block_channel), tl.float32)
    m_base = mask_ptr + (batch * mask_batch_stride + group * mask_group_stride)
    if mask_kind == _MASK_KEY:
        keep = _load_tile(m_base + key_offsets * mask_key_stride, key_mask, 0, even_key).to(tl.int1)

    q_base = (
        query_ptr
        + (batch * query_batch_stride + group * query_group_stride + head * query_head_stride)
        + query_offsets[:, None] * query_token_stride
        + channel_offsets[None, :] * query_channel_stride
    )
    do_base = (
        grad_output_ptr
        + (batch * grad_output_batch_stride + group * grad_output_group_stride + head * grad_output_head_stride)
        + query_offsets[:, None] * grad_output_token_stride
        + channel_offsets[None, :] * grad_output_channel_stride
    )
    row_base = (batch * lse_batch_stride + group * lse_group_stride + head * lse_head_stride) + (
        query_offsets * lse_token_stride
    )
    b_group = bias_ptr + (batch * bias_batch_stride + bias_group * bias_group_stride + head * bias_head_stride)
    if bias_transposed:
        b_base = b_group + key_offsets[:, None] * bias_key_stride + query_offsets[None, :] * bias_query_stride
    else:
        b_base = b_group + query_offsets[:, None] * bias_query_stride + key_offsets[None, :] * bias_key_stride

    for query_start in range(0, query_length, block_query):
        query_start = tl.multiple_of(query_start, block_query)
        queries = query_start + query_offsets
        query_mask = queries < query_length
        q_mask = query_mask[:, None] & channel_mask[None, :]
        query = _load_tile(q_base + query_start * query_token_stride, q_mask, 0.0, even_tile)
        grad_output = _load_tile(do_base + query_start * grad_output_token_stride, q_mask, 0.0, even_tile)
        rows = row_base + query_start * lse_token_stride
        if need_grad_key:
            delta = _load_tile(delta_ptr + rows, query_mask, 0.0, even_query).to(tl.float32)
        lse_log2 = _load_tile(lse_ptr + rows, query_mask, 0.0, even_query).to(tl.float32) * LOG2E

        scores_t = tl.dot(key, tl.trans(query), input_precision=input_precision)
        if has_bias & bias_transposed:
            bias_t = _load_tile(
                b_base + query_start * bias_query_stride,
                key_mask[:, None] & query_mask[None, :],
                MASKED_LOGIT,
                even_query & even_key,
            )
            scores_t = tl.fma(scores_t, scale, bias_t.to(tl.float32))
        elif has_bias:
            bias = _load_tile(
                b_base + query_start * bias_query_stride,
                query_mask[:, None] & key_mask[None, :],
                MASKED_LOGIT,
                even_query & even_key,
            )
            scores_t = tl.fma(scores_t, scale, tl.trans(bias).to(tl.float32))
        else:
            scores_t = scores_t * scale
        if mask_kind == _MASK_KEY:
            scores_t = tl.where(keep[:, None], scores_t, MASKED_LOGIT)
        elif mask_kind == _MASK_PAIR:
            keep_pair = tl.load(
                m_base + queries[:, None] * mask_query_stride + key_offsets[None, :] * mask_key_stride,
                mask=query_mask[:, None] & key_mask[None, :],
                other=0,
            ).to(tl.int1)
            scores_t = tl.where(tl.trans(keep_pair), scores_t, MASKED_LOGIT)
        if not even_key:
            scores_t = tl.where(key_mask[:, None], scores_t, MASKED_LOGIT)
        probabilities_t = tl.exp2(tl.fma(scores_t, LOG2E, -lse_log2[None, :]))
        if need_grad_key:
            grad_probabilities_t = tl.dot(value, tl.trans(grad_output), input_precision=input_precision)
            grad_scores_t = probabilities_t * (grad_probabilities_t - delta[None, :])
        if need_grad_value:
            grad_value = tl.dot(
                probabilities_t.to(grad_output.dtype), grad_output, grad_value, input_precision=input_precision
            )
        if need_grad_key:
            grad_key = tl.dot(grad_scores_t.to(query.dtype), query, grad_key, input_precision=input_precision)

    if need_grad_key:
        tl.store(
            grad_key_ptr
            + (batch * grad_key_batch_stride + group * grad_key_group_stride + head * grad_key_head_stride)
            + key_offsets[:, None] * grad_key_token_stride
            + channel_offsets[None, :] * grad_key_channel_stride,
            grad_key * scale,
            mask=kv_mask,
        )
    if need_grad_value:
        tl.store(
            grad_value_ptr
            + (batch * grad_value_batch_stride + group * grad_value_group_stride + head * grad_value_head_stride)
            + key_offsets[:, None] * grad_value_token_stride
            + channel_offsets[None, :] * grad_value_channel_stride,
            grad_value,
            mask=kv_mask,
        )


@triton.jit
def _grad_bias_half(
    query,
    grad_output,
    delta,
    lse_log2,
    k_ptrs,
    v_ptrs,
    m_ptrs,
    pair_m_ptrs,
    kv_mask,
    k_mask,
    q_mask,
    bias_tile,
    scale,
    accumulator,
    mask_kind: tl.constexpr,
    even_query: tl.constexpr,
    even_key: tl.constexpr,
    even_head_size: tl.constexpr,
    input_precision: tl.constexpr,
):

    key = _load_tile(k_ptrs, kv_mask, 0.0, even_key & even_head_size)
    value = _load_tile(v_ptrs, kv_mask, 0.0, even_key & even_head_size)
    scores = tl.fma(tl.dot(query, tl.trans(key), input_precision=input_precision), scale, bias_tile)
    if mask_kind == _MASK_KEY:
        keep = _load_tile(m_ptrs, k_mask, 0, even_key).to(tl.int1)
        scores = tl.where(keep[None, :], scores, MASKED_LOGIT)
    elif mask_kind == _MASK_PAIR:
        keep = tl.load(pair_m_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0).to(tl.int1)
        scores = tl.where(keep, scores, MASKED_LOGIT)
    if not even_key:
        scores = tl.where(k_mask[None, :], scores, MASKED_LOGIT)
    probabilities = tl.exp2(tl.fma(scores, LOG2E, -lse_log2[:, None]))
    grad_probabilities = tl.dot(grad_output, tl.trans(value), input_precision=input_precision)
    grad_scores = probabilities * (grad_probabilities - delta[:, None])
    if not even_query:
        grad_scores = tl.where(q_mask[:, None], grad_scores, 0.0)
    return accumulator + grad_scores


@triton.jit
def _flash_attention_grad_bias(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    grad_output_ptr,
    grad_bias_ptr,
    query_batch_stride,
    query_group_stride,
    query_token_stride,
    query_head_stride,
    query_channel_stride,
    key_batch_stride,
    key_group_stride,
    key_token_stride,
    key_head_stride,
    key_channel_stride,
    value_batch_stride,
    value_group_stride,
    value_token_stride,
    value_head_stride,
    value_channel_stride,
    mask_batch_stride,
    mask_group_stride,
    mask_query_stride,
    mask_key_stride,
    bias_batch_stride,
    bias_head_stride,
    bias_query_stride,
    bias_key_stride,
    lse_batch_stride,
    lse_group_stride,
    lse_head_stride,
    lse_token_stride,
    grad_output_batch_stride,
    grad_output_group_stride,
    grad_output_token_stride,
    grad_output_head_stride,
    grad_output_channel_stride,
    grad_bias_batch_stride,
    grad_bias_head_stride,
    grad_bias_query_stride,
    grad_bias_key_stride,
    num_groups,
    query_length,
    key_length,
    num_heads,
    head_size: tl.constexpr,
    scale,
    mask_kind: tl.constexpr,
    even_query: tl.constexpr,
    even_key: tl.constexpr,
    even_head_size: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
    block_channel: tl.constexpr,
    input_precision: tl.constexpr,
    group_parts: tl.constexpr = 1,
    partial_stride: tl.constexpr = 0,
) -> None:

    part = tl.program_id(0).to(tl.int64) % group_parts
    batch_head = tl.program_id(0).to(tl.int64) // group_parts
    grad_bias_ptr += part * partial_stride
    q_block = tl.program_id(1)
    k_block = tl.program_id(2)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    split: tl.constexpr = block_key >= 128
    half_key: tl.constexpr = block_key // 2 if split else block_key
    even_tile: tl.constexpr = even_query & even_head_size

    q_offsets = q_block * block_query + tl.arange(0, block_query)
    k_offsets = k_block * block_key + tl.arange(0, half_key)
    channel_offsets = tl.arange(0, block_channel)
    q_mask = q_offsets < query_length
    k_mask0 = k_offsets < key_length
    k_mask1 = k_offsets + half_key < key_length
    channel_mask = channel_offsets < head_size
    q_load_mask = q_mask[:, None] & channel_mask[None, :]
    kv_mask0 = k_mask0[:, None] & channel_mask[None, :]
    kv_mask1 = k_mask1[:, None] & channel_mask[None, :]

    b_base = (
        bias_ptr
        + (batch * bias_batch_stride + head * bias_head_stride)
        + q_offsets[:, None] * bias_query_stride
        + k_offsets[None, :] * bias_key_stride
    )
    bias0 = _load_tile(b_base, q_mask[:, None] & k_mask0[None, :], 0.0, even_query & even_key).to(tl.float32)
    if split:
        bias1 = _load_tile(
            b_base + half_key * bias_key_stride, q_mask[:, None] & k_mask1[None, :], 0.0, even_query & even_key
        ).to(tl.float32)

    q_base = (
        query_ptr
        + (batch * query_batch_stride + head * query_head_stride)
        + q_offsets[:, None] * query_token_stride
        + channel_offsets[None, :] * query_channel_stride
    )
    k_base = (
        key_ptr
        + (batch * key_batch_stride + head * key_head_stride)
        + k_offsets[:, None] * key_token_stride
        + channel_offsets[None, :] * key_channel_stride
    )
    v_base = (
        value_ptr
        + (batch * value_batch_stride + head * value_head_stride)
        + k_offsets[:, None] * value_token_stride
        + channel_offsets[None, :] * value_channel_stride
    )
    do_base = (
        grad_output_ptr
        + (batch * grad_output_batch_stride + head * grad_output_head_stride)
        + q_offsets[:, None] * grad_output_token_stride
        + channel_offsets[None, :] * grad_output_channel_stride
    )
    row_base = (batch * lse_batch_stride + head * lse_head_stride) + q_offsets * lse_token_stride
    m_base = mask_ptr + batch * mask_batch_stride + k_offsets * mask_key_stride
    pair_m_base = (
        mask_ptr
        + batch * mask_batch_stride
        + q_offsets[:, None] * mask_query_stride
        + k_offsets[None, :] * mask_key_stride
    )

    accumulator0 = tl.zeros((block_query, half_key), tl.float32)
    if split:
        accumulator1 = tl.zeros((block_query, half_key), tl.float32)
    for group_index in range(part * num_groups // group_parts, (part + 1) * num_groups // group_parts):
        group = group_index.to(tl.int64)  # type: ignore[attr-defined]
        query = _load_tile(q_base + group * query_group_stride, q_load_mask, 0.0, even_tile)
        grad_output = _load_tile(do_base + group * grad_output_group_stride, q_load_mask, 0.0, even_tile)
        rows = row_base + group * lse_group_stride
        lse_log2 = _load_tile(lse_ptr + rows, q_mask, 0.0, even_query).to(tl.float32) * LOG2E
        delta = _load_tile(delta_ptr + rows, q_mask, 0.0, even_query).to(tl.float32)
        k_ptrs = k_base + group * key_group_stride
        v_ptrs = v_base + group * value_group_stride
        m_ptrs = m_base + group * mask_group_stride
        pair_m_ptrs = pair_m_base + group * mask_group_stride
        accumulator0 = _grad_bias_half(
            query,
            grad_output,
            delta,
            lse_log2,
            k_ptrs,
            v_ptrs,
            m_ptrs,
            pair_m_ptrs,
            kv_mask0,
            k_mask0,
            q_mask,
            bias0,
            scale,
            accumulator0,
            mask_kind,
            even_query,
            even_key,
            even_head_size,
            input_precision,
        )
        if split:
            accumulator1 = _grad_bias_half(
                query,
                grad_output,
                delta,
                lse_log2,
                k_ptrs + half_key * key_token_stride,
                v_ptrs + half_key * value_token_stride,
                m_ptrs + half_key * mask_key_stride,
                pair_m_ptrs + half_key * mask_key_stride,
                kv_mask1,
                k_mask1,
                q_mask,
                bias1,
                scale,
                accumulator1,
                mask_kind,
                even_query,
                even_key,
                even_head_size,
                input_precision,
            )

    db_base = (
        grad_bias_ptr
        + (batch * grad_bias_batch_stride + head * grad_bias_head_stride)
        + q_offsets[:, None] * grad_bias_query_stride
        + k_offsets[None, :] * grad_bias_key_stride
    )
    tl.store(db_base, accumulator0, mask=q_mask[:, None] & k_mask0[None, :])
    if split:
        tl.store(db_base + half_key * grad_bias_key_stride, accumulator1, mask=q_mask[:, None] & k_mask1[None, :])


@triton.jit
def _reduce_attention_bias_parts(
    partials_ptr,
    output_ptr,
    size,
    group_parts: tl.constexpr,
    block_size: tl.constexpr,
) -> None:

    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    parts = tl.arange(0, group_parts).to(tl.int64)
    partials = tl.load(partials_ptr + parts[:, None] * size + offsets[None, :], offsets[None, :] < size, 0.0)
    tl.store(output_ptr + offsets, tl.sum(partials, axis=0), offsets < size)
