from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _bounded_class(label, num_classes):

    return tl.minimum(tl.maximum(label.to(tl.int64), 0), num_classes - 1)


@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    labels_ptr,
    out_ptr,
    rows,
    num_classes,
    block_rows: tl.constexpr,
    block_classes: tl.constexpr,
) -> None:
    row_offsets = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    classes = tl.arange(0, block_classes)
    row_mask = row_offsets < rows
    class_mask = classes < num_classes
    logits = tl.load(
        logits_ptr + row_offsets[:, None] * num_classes + classes[None, :],
        mask=row_mask[:, None] & class_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=1)[:, None]
    shifted = logits - row_max
    logsumexp = tl.log(tl.sum(tl.exp(shifted), axis=1)) + tl.reshape(row_max, (block_rows,))
    label = _bounded_class(tl.load(labels_ptr + row_offsets, mask=row_mask, other=0), num_classes)
    target = tl.load(logits_ptr + row_offsets * num_classes + label, mask=row_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_offsets, logsumexp - target, mask=row_mask)


@triton.jit
def _cross_entropy_backward(
    logits_ptr,
    labels_ptr,
    grad_out_ptr,
    grad_logits_ptr,
    rows,
    num_classes,
    block_rows: tl.constexpr,
    block_classes: tl.constexpr,
) -> None:
    row_offsets = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    classes = tl.arange(0, block_classes)
    row_mask = row_offsets < rows
    class_mask = classes < num_classes
    logits = tl.load(
        logits_ptr + row_offsets[:, None] * num_classes + classes[None, :],
        mask=row_mask[:, None] & class_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=1)[:, None]
    weights = tl.exp(logits - row_max)
    weights = weights / tl.sum(weights, axis=1)[:, None]
    label = _bounded_class(tl.load(labels_ptr + row_offsets, mask=row_mask, other=0), num_classes)
    grad_out = tl.load(grad_out_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    grad = (weights - tl.where(classes[None, :] == label[:, None], 1.0, 0.0)) * grad_out[:, None]
    tl.store(
        grad_logits_ptr + row_offsets[:, None] * num_classes + classes[None, :],
        grad,
        mask=row_mask[:, None] & class_mask[None, :],
    )


@triton.jit
def _distogram_forward_accumulate(
    logits_ptr,
    labels_ptr,
    token_mask_ptr,
    accum_ptr,
    rows_per_batch,
    num_tokens,
    num_classes,
    wide_offsets: tl.constexpr,
    block_rows: tl.constexpr,
    block_classes: tl.constexpr,
    ce_ptr,
) -> None:
    row_pid = tl.program_id(0)
    batch = tl.program_id(1)
    if wide_offsets:
        row_pid = row_pid.to(tl.int64)
        batch = batch.to(tl.int64)
    row_offsets = row_pid * block_rows + tl.arange(0, block_rows)
    classes = tl.arange(0, block_classes)
    row_mask = row_offsets < rows_per_batch
    class_mask = classes < num_classes
    query_offsets = row_offsets // num_tokens
    key_offsets = row_offsets - query_offsets * num_tokens
    token_mask_base = batch * num_tokens
    query_mask = tl.load(token_mask_ptr + token_mask_base + query_offsets, mask=row_mask, other=0.0).to(tl.float32)
    key_mask = tl.load(token_mask_ptr + token_mask_base + key_offsets, mask=row_mask, other=0.0).to(tl.float32)
    pair_weight = query_mask * key_mask * tl.where(query_offsets != key_offsets, 1.0, 0.0)

    row_base = batch * rows_per_batch + row_offsets
    logits = tl.load(
        logits_ptr + row_base[:, None] * num_classes + classes[None, :],
        mask=row_mask[:, None] & class_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    weights = tl.exp(logits - safe_max[:, None])
    logsumexp = tl.log(tl.sum(weights, axis=1)) + safe_max
    label = _bounded_class(tl.load(labels_ptr + row_base, mask=row_mask, other=0), num_classes).to(tl.int64)
    target = tl.load(logits_ptr + row_base * num_classes + label, mask=row_mask, other=0.0).to(tl.float32)
    ce = tl.where(row_mask, logsumexp - target, 0.0)
    tl.store(ce_ptr + row_base, ce, mask=row_mask)
    numerator = tl.sum(ce * pair_weight, axis=0)
    denominator = tl.sum(pair_weight, axis=0)
    tl.atomic_add(accum_ptr + batch * 2, numerator, sem="relaxed")
    tl.atomic_add(accum_ptr + batch * 2 + 1, denominator, sem="relaxed")


@triton.jit
def _distogram_forward_partials(
    logits_ptr,
    labels_ptr,
    token_mask_ptr,
    partials_ptr,
    row_blocks,
    partial_stride,
    rows_per_batch,
    num_tokens,
    num_classes,
    row_parts: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_rows: tl.constexpr,
    block_classes: tl.constexpr,
    ce_ptr,
) -> None:
    r"""Write disjoint row-block partials.

    Part p owns [p * row_blocks // row_parts, (p + 1) * row_blocks // row_parts).
    partial_stride=0 writes directly into the accumulator.
    """

    part = tl.program_id(0)
    batch = tl.program_id(1)
    if wide_offsets:
        part = part.to(tl.int64)
        batch = batch.to(tl.int64)
    classes = tl.arange(0, block_classes)
    class_mask = classes < num_classes
    token_mask_base = batch * num_tokens
    numerator = tl.zeros((), dtype=tl.float32)
    denominator = tl.zeros((), dtype=tl.float32)

    for row_block in range(part * row_blocks // row_parts, (part + 1) * row_blocks // row_parts):
        row_offsets = row_block * block_rows + tl.arange(0, block_rows)
        row_mask = row_offsets < rows_per_batch
        query_offsets = row_offsets // num_tokens
        key_offsets = row_offsets - query_offsets * num_tokens
        query_mask = tl.load(token_mask_ptr + token_mask_base + query_offsets, mask=row_mask, other=0.0).to(tl.float32)
        key_mask = tl.load(token_mask_ptr + token_mask_base + key_offsets, mask=row_mask, other=0.0).to(tl.float32)
        pair_weight = query_mask * key_mask * tl.where(query_offsets != key_offsets, 1.0, 0.0)

        row_base = batch * rows_per_batch + row_offsets
        logits = tl.load(
            logits_ptr + row_base[:, None] * num_classes + classes[None, :],
            mask=row_mask[:, None] & class_mask[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        row_max = tl.max(logits, axis=1)
        safe_max = tl.where(row_mask, row_max, 0.0)
        weights = tl.exp(logits - safe_max[:, None])
        logsumexp = tl.log(tl.sum(weights, axis=1)) + safe_max
        label = _bounded_class(tl.load(labels_ptr + row_base, mask=row_mask, other=0), num_classes).to(tl.int64)
        target = tl.load(logits_ptr + row_base * num_classes + label, mask=row_mask, other=0.0).to(tl.float32)
        ce = tl.where(row_mask, logsumexp - target, 0.0)
        tl.store(ce_ptr + row_base, ce, mask=row_mask)
        numerator += tl.sum(ce * pair_weight, axis=0)
        denominator += tl.sum(pair_weight, axis=0)

    partial_base = partials_ptr + part * partial_stride + batch * 2
    tl.store(partial_base, numerator)
    tl.store(partial_base + 1, denominator)


@triton.jit
def _reduce_distogram_partials(
    partials_ptr,
    accum_ptr,
    items,
    row_parts: tl.constexpr,
    block_items: tl.constexpr,
) -> None:
    r"""items is the flattened (batch, 2) extent and the stride between partials."""

    offsets = tl.program_id(0).to(tl.int64) * block_items + tl.arange(0, block_items)
    parts = tl.arange(0, row_parts).to(tl.int64)
    item_mask = offsets < items
    partials = tl.load(
        partials_ptr + parts[:, None] * items + offsets[None, :],
        mask=item_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(accum_ptr + offsets, tl.sum(partials, axis=0), mask=item_mask)


@triton.jit
def _distogram_finalize_tiles(
    accum_ptr,
    denominator_ptr,
    out_ptr,
    partials_ptr,
    batch,
    eps,
    reduction_mode: tl.constexpr,
    single_partial: tl.constexpr,
    block_items: tl.constexpr,
) -> None:
    tile = tl.program_id(0).to(tl.int64)
    batch_offsets = tile * block_items + tl.arange(0, block_items)
    mask = batch_offsets < batch
    numerator = tl.load(accum_ptr + batch_offsets * 2, mask=mask, other=0.0).to(tl.float32)
    denominator = tl.load(accum_ptr + batch_offsets * 2 + 1, mask=mask, other=0.0).to(tl.float32) + eps
    loss = numerator / denominator
    tl.store(denominator_ptr + batch_offsets, denominator, mask=mask)
    if reduction_mode == 0:
        tl.store(out_ptr + batch_offsets, loss, mask=mask)
    else:
        total = tl.sum(tl.where(mask, loss, 0.0), axis=0)
        if single_partial:
            if reduction_mode == 1:
                total = total / batch
            tl.store(out_ptr, total)
        else:
            tl.store(partials_ptr + tile, total)


@triton.jit
def _distogram_finalize_partials(
    partials_ptr,
    out_ptr,
    batch,
    num_partials,
    reduction_mode: tl.constexpr,
    block_partials: tl.constexpr,
) -> None:
    offsets = tl.arange(0, block_partials)
    partials = tl.load(partials_ptr + offsets, mask=offsets < num_partials, other=0.0).to(tl.float32)
    total = tl.sum(partials, axis=0)
    if reduction_mode == 1:
        total = total / batch
    tl.store(out_ptr, total)


@triton.jit
def _distogram_backward(
    logits_ptr,
    labels_ptr,
    token_mask_ptr,
    denominator_ptr,
    grad_out_ptr,
    grad_logits_ptr,
    rows_per_batch,
    num_tokens,
    num_classes,
    batch_size,
    reduction_mode: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_rows: tl.constexpr,
    block_classes: tl.constexpr,
) -> None:
    row_pid = tl.program_id(0)
    batch = tl.program_id(1)
    if wide_offsets:
        row_pid = row_pid.to(tl.int64)
        batch = batch.to(tl.int64)
    row_offsets = row_pid * block_rows + tl.arange(0, block_rows)
    classes = tl.arange(0, block_classes)
    row_mask = row_offsets < rows_per_batch
    class_mask = classes < num_classes
    query_offsets = row_offsets // num_tokens
    key_offsets = row_offsets - query_offsets * num_tokens
    token_mask_base = batch * num_tokens
    query_mask = tl.load(token_mask_ptr + token_mask_base + query_offsets, mask=row_mask, other=0.0).to(tl.float32)
    key_mask = tl.load(token_mask_ptr + token_mask_base + key_offsets, mask=row_mask, other=0.0).to(tl.float32)
    pair_weight = query_mask * key_mask * tl.where(query_offsets != key_offsets, 1.0, 0.0)

    row_base = batch * rows_per_batch + row_offsets
    logits = tl.load(
        logits_ptr + row_base[:, None] * num_classes + classes[None, :],
        mask=row_mask[:, None] & class_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=1)[:, None]
    weights = tl.exp(logits - row_max)
    exponential_sum = tl.sum(weights, axis=1)
    weights = weights / exponential_sum[:, None]
    label = _bounded_class(tl.load(labels_ptr + row_base, mask=row_mask, other=0), num_classes).to(tl.int64)
    denominator = tl.load(denominator_ptr + batch).to(tl.float32)
    if reduction_mode == 0:
        upstream = tl.load(grad_out_ptr + batch).to(tl.float32)
    else:
        upstream = tl.load(grad_out_ptr).to(tl.float32)
        if reduction_mode == 1:
            upstream = upstream / batch_size
    scale = pair_weight * upstream / denominator
    grad = (weights - tl.where(classes[None, :] == label[:, None], 1.0, 0.0)) * scale[:, None]
    tl.store(
        grad_logits_ptr + row_base[:, None] * num_classes + classes[None, :],
        grad,
        mask=row_mask[:, None] & class_mask[None, :],
    )


@triton.jit
def _distogram_token_mask_backward(
    ce_ptr,
    token_mask_ptr,
    denominator_ptr,
    accum_ptr,
    grad_out_ptr,
    grad_token_mask_ptr,
    rows_per_batch,
    num_tokens,
    batch_size,
    reduction_mode: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_tokens: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    batch = tl.program_id(1)
    if wide_offsets:
        token = token.to(tl.int64)
        batch = batch.to(tl.int64)

    mask_base = batch * num_tokens
    numerator = tl.load(accum_ptr + batch * 2).to(tl.float32)
    denominator = tl.load(denominator_ptr + batch).to(tl.float32)
    batch_loss = numerator / denominator
    derivative = tl.zeros((), dtype=tl.float32)

    for start in tl.range(0, num_tokens, block_tokens):
        partner = start + tl.arange(0, block_tokens)
        partner_mask = (partner < num_tokens) & (partner != token)
        partner_weight = tl.load(
            token_mask_ptr + mask_base + partner,
            mask=partner_mask,
            other=0.0,
        ).to(tl.float32)

        forward_row = batch * rows_per_batch + token * num_tokens + partner
        reverse_row = batch * rows_per_batch + partner * num_tokens + token
        forward_ce = tl.load(ce_ptr + forward_row, mask=partner_mask, other=0.0)
        reverse_ce = tl.load(ce_ptr + reverse_row, mask=partner_mask, other=0.0)

        derivative += tl.sum(partner_weight * (forward_ce + reverse_ce - 2.0 * batch_loss), axis=0)

    if reduction_mode == 0:
        upstream = tl.load(grad_out_ptr + batch).to(tl.float32)
    else:
        upstream = tl.load(grad_out_ptr).to(tl.float32)
        if reduction_mode == 1:
            upstream = upstream / batch_size
    tl.store(grad_token_mask_ptr + mask_base + token, derivative * upstream / denominator)
