from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _bounded_edge(edge, num_edges):

    return tl.minimum(tl.maximum(edge.to(tl.int64), 0), num_edges)


@triton.jit
def sum_forward(
    values_ptr,
    row_offsets_ptr,
    out_ptr,
    inferred_segments,
    num_edges,
    features,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    segment_parts: tl.constexpr,
) -> None:
    slot = tl.program_id(0).to(tl.int64)
    segment = slot // segment_parts
    part = slot % segment_parts
    columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    start = _bounded_edge(tl.load(row_offsets_ptr + segment, mask=segment < inferred_segments, other=0), num_edges)
    stop = _bounded_edge(tl.load(row_offsets_ptr + segment + 1, mask=segment < inferred_segments, other=0), num_edges)
    length = tl.maximum(stop - start, 0)
    stop = start + length * (part + 1) // segment_parts
    start += length * part // segment_parts
    if stop - start <= 32:
        short_accumulator = tl.full((block_features,), 0, tl.float64 if use_float64 else tl.float32)
        for edge in range(start, stop):
            short_values = tl.load(values_ptr + edge * features + columns, mask=columns < features, other=0)
            short_accumulator += short_values.to(short_accumulator.dtype)
        result = short_accumulator
    else:
        accumulator = tl.full((block_edges, block_features), 0, tl.float64 if use_float64 else tl.float32)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            values = tl.load(
                values_ptr + edges[:, None] * features + columns[None, :],
                mask=(edges < stop)[:, None] & (columns < features)[None, :],
                other=0,
            )
            accumulator += values.to(accumulator.dtype)
        result = tl.sum(accumulator, 0)
    tl.store(out_ptr + slot * features + columns, result, mask=columns < features)


@triton.jit
def merge_partials(
    partial_ptr,
    output_ptr,
    features,
    reduction: tl.constexpr,
    segment_parts: tl.constexpr,
    block_features: tl.constexpr,
):
    segment = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
    parts = tl.arange(0, segment_parts)
    partial = tl.load(
        partial_ptr + (segment * segment_parts + parts[:, None]) * features + columns[None, :],
        mask=(columns < features)[None, :],
        other=0,
    )
    if reduction == "max":
        value = tl.max(partial, 0)
    else:
        value = tl.sum(partial, 0)
    tl.store(output_ptr + segment * features + columns, value, mask=columns < features)


@triton.jit
def long_start_bucket(offsets_ptr, num_edges, num_segments, bucket, threshold: tl.constexpr):
    # A qualifying row belongs to the bucket floor(start / threshold).
    edge = (bucket + 1) * threshold - 1
    low = tl.full((), 0, tl.int64)
    high = tl.cast(num_segments, tl.int64)
    while low + 1 < high:
        middle = (low + high) // 2
        offset = _bounded_edge(tl.load(offsets_ptr + middle), num_edges)
        take_right = offset <= edge
        low = tl.where(take_right, middle, low)
        high = tl.where(take_right, high, middle)
    start = _bounded_edge(tl.load(offsets_ptr + low), num_edges)
    stop = _bounded_edge(tl.load(offsets_ptr + low + 1), num_edges)
    active = (stop - start >= threshold) & (start // threshold == bucket)
    return low, start, stop, active


@triton.jit
def sum_forward_adaptive(
    values_ptr,
    row_offsets_ptr,
    out_ptr,
    inferred_segments,
    out_segments,
    num_edges,
    features,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    long_features: tl.constexpr,
    long_edges: tl.constexpr,
    threshold: tl.constexpr,
    skip_long_rows: tl.constexpr = False,
    long_descriptors_ptr=None,
    counter_tiles: tl.constexpr = 0,
):
    work = tl.program_id(0).to(tl.int64)
    if work < out_segments:
        start = _bounded_edge(tl.load(row_offsets_ptr + work, mask=work < inferred_segments, other=0), num_edges)
        stop = _bounded_edge(tl.load(row_offsets_ptr + work + 1, mask=work < inferred_segments, other=0), num_edges)
        if stop - start < threshold:
            sum_forward(
                values_ptr,
                row_offsets_ptr,
                out_ptr,
                inferred_segments,
                num_edges,
                features,
                use_float64,
                block_features,
                block_edges,
                1,
            )
    elif skip_long_rows:
        if tl.program_id(1) == 0:
            bucket = work - out_segments
            segment, start, stop, active = long_start_bucket(
                row_offsets_ptr, num_edges, inferred_segments, bucket, threshold
            )
            tl.store(long_descriptors_ptr + bucket * 3, segment)
            tl.store(long_descriptors_ptr + bucket * 3 + 1, start)
            tl.store(long_descriptors_ptr + bucket * 3 + 2, tl.where(active, stop, start))
            if counter_tiles:
                counters = (long_descriptors_ptr + (num_edges // threshold) * 3).to(tl.pointer_type(tl.int32))
                lanes = tl.arange(0, 32)
                for base in range(0, counter_tiles, 32):
                    column = base + lanes
                    tl.store(counters + bucket * counter_tiles + column, 0, mask=column < counter_tiles)
    else:
        tiles: tl.constexpr = block_features // long_features
        extra = work - out_segments
        bucket = extra // tiles
        feature_start = tl.program_id(1) * block_features + (extra % tiles) * long_features
        segment, start, stop, active = long_start_bucket(
            row_offsets_ptr, num_edges, inferred_segments, bucket, threshold
        )
        if active & (feature_start < features):
            columns = feature_start + tl.arange(0, long_features)
            rows = tl.arange(0, long_edges).to(tl.int64)
            accumulator = tl.full((long_edges, long_features), 0, tl.float64 if use_float64 else tl.float32)
            for begin in range(start, stop, long_edges):
                edges = begin + rows
                values = tl.load(
                    values_ptr + edges[:, None] * features + columns[None, :],
                    mask=(edges < stop)[:, None] & (columns < features)[None, :],
                    other=0,
                )
                accumulator += values.to(accumulator.dtype)
            result = tl.sum(accumulator, 0)
            tl.store(out_ptr + segment * features + columns, result, mask=columns < features)


@triton.jit
def long_reduction_partials(
    values_ptr,
    descriptors_ptr,
    partial_ptr,
    num_edges,
    features,
    use_float64: tl.constexpr,
    reduction: tl.constexpr,
    parts: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    threshold: tl.constexpr,
    weights_ptr=None,
    weight_groups: tl.constexpr = 0,
    weight_features: tl.constexpr = 1,
):
    for slot in range(tl.program_id(0).to(tl.int64), (num_edges // threshold) * parts, tl.num_programs(0)):
        bucket, part = slot // parts, slot % parts
        start = tl.load(descriptors_ptr + bucket * 3 + 1)
        stop = tl.load(descriptors_ptr + bucket * 3 + 2)
        if start < stop:
            length = stop - start
            stop = start + length * (part + 1) // parts
            start += length * part // parts
            columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
            rows = tl.arange(0, block_edges).to(tl.int64)
            initial: tl.constexpr = -float("inf") if reduction == "max" else 0
            accumulator = tl.full((block_edges, block_features), initial, tl.float64 if use_float64 else tl.float32)
            for begin in range(start, stop, block_edges):
                edges = begin + rows
                values = tl.load(
                    values_ptr + edges[:, None] * features + columns[None, :],
                    mask=(edges < stop)[:, None] & (columns < features)[None, :],
                    other=initial,
                )
                if reduction == "max":
                    accumulator = tl.maximum(accumulator, values.to(accumulator.dtype))
                else:
                    contribution = values.to(accumulator.dtype)
                    if weight_groups:
                        weights = tl.load(
                            weights_ptr + edges[:, None] * weight_groups + (columns // weight_features)[None, :],
                            mask=(edges < stop)[:, None] & (columns < features)[None, :],
                            other=0,
                        )
                        contribution *= weights.to(accumulator.dtype)
                    accumulator += contribution
            if reduction == "max":
                result = tl.max(accumulator, 0)
            else:
                result = tl.sum(accumulator, 0)
            tl.store(partial_ptr + slot * features + columns, result, mask=columns < features)


@triton.jit
def long_reduction_merge(
    descriptors_ptr,
    partial_ptr,
    out_ptr,
    buckets,
    features,
    reduction: tl.constexpr,
    parts: tl.constexpr,
    block_features: tl.constexpr,
):
    for bucket in range(tl.program_id(0).to(tl.int64), buckets, tl.num_programs(0)):
        segment = tl.load(descriptors_ptr + bucket * 3)
        start = tl.load(descriptors_ptr + bucket * 3 + 1)
        stop = tl.load(descriptors_ptr + bucket * 3 + 2)
        if start < stop:
            columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
            part_offsets = bucket * parts + tl.arange(0, parts)
            initial: tl.constexpr = -float("inf") if reduction == "max" else 0
            values = tl.load(
                partial_ptr + part_offsets[:, None] * features + columns[None, :],
                mask=(columns < features)[None, :],
                other=initial,
            )
            if reduction == "max":
                result = tl.max(values, 0)
            else:
                result = tl.sum(values, 0)
            tl.store(out_ptr + segment * features + columns, result, mask=columns < features)


@triton.jit
def sum_forward_owned(
    values_ptr,
    offsets_ptr,
    out_ptr,
    descriptors_ptr,
    inferred_segments: tl.constexpr,
    out_segments: tl.constexpr,
    num_edges: tl.constexpr,
    features: tl.constexpr,
    use_float64: tl.constexpr,
    block: tl.constexpr,
    threshold: tl.constexpr,
    counter_tiles: tl.constexpr = 0,
):
    work = tl.program_id(0)
    if out_segments * features >= 2**31:
        work = work.to(tl.int64)
    flat_programs: tl.constexpr = triton.cdiv(out_segments * features, block)
    if work < flat_programs:
        indices = work * block + tl.arange(0, block)
        segment = indices // features
        column = indices % features
        live = indices < out_segments * features
        start = _bounded_edge(
            tl.load(offsets_ptr + segment, mask=live & (segment < inferred_segments), other=0), num_edges
        )
        stop = _bounded_edge(
            tl.load(offsets_ptr + segment + 1, mask=live & (segment < inferred_segments), other=0), num_edges
        )
        length = tl.maximum(stop - start, 0)
        short = length <= 32
        middle = (length > 32) & (length < threshold)
        length = tl.where(short, length, 0)
        initial: tl.constexpr = 0
        acc = tl.full((block,), initial, tl.float64 if use_float64 else tl.float32)
        for row in range(0, tl.max(length, 0)):
            edge = start + row
            mask = live & (row < length)
            value = tl.load(values_ptr + edge * features + column, mask=mask, other=initial).to(acc.dtype)
            acc += value
        tl.store(out_ptr + indices, acc, mask=live & short)
        middle_segment = tl.min(tl.where(live & middle, segment, out_segments), 0)
        while middle_segment < out_segments:
            middle_start = _bounded_edge(
                tl.load(offsets_ptr + middle_segment, mask=middle_segment < inferred_segments, other=0), num_edges
            )
            middle_stop = _bounded_edge(
                tl.load(offsets_ptr + middle_segment + 1, mask=middle_segment < inferred_segments, other=0),
                num_edges,
            )
            reduction_forward_middle(
                values_ptr,
                out_ptr,
                middle_start,
                middle_stop,
                middle_segment,
                features,
                work * block,
                work * block + block,
                use_float64,
                64,
                32,
            )
            middle_segment = tl.min(tl.where(live & middle & (segment > middle_segment), segment, out_segments), 0)

    else:
        bucket = work - flat_programs
        long_segment, long_start, long_stop, long_active = long_start_bucket(
            offsets_ptr, num_edges, inferred_segments, bucket, threshold
        )
        tl.store(descriptors_ptr + bucket * 3, long_segment)
        tl.store(descriptors_ptr + bucket * 3 + 1, long_start)
        tl.store(descriptors_ptr + bucket * 3 + 2, tl.where(long_active, long_stop, long_start))
        if counter_tiles:
            counters = (descriptors_ptr + (num_edges // threshold) * 3).to(tl.pointer_type(tl.int32))
            lanes = tl.arange(0, 32)
            for base in range(0, counter_tiles, 32):
                column = base + lanes
                tl.store(counters + bucket * counter_tiles + column, 0, mask=column < counter_tiles)


@triton.jit
def reduction_forward_middle(
    values_ptr,
    out_ptr,
    start,
    stop,
    segment,
    features,
    owned_begin,
    owned_end,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    reduction: tl.constexpr = "sum",
):
    first = tl.maximum(0, owned_begin - segment * features) // block_features * block_features
    last = tl.minimum(features, owned_end - segment * features)
    for column_begin in range(first, last, block_features):
        columns = column_begin + tl.arange(0, block_features)
        owned = (
            (segment * features + columns >= owned_begin)
            & (segment * features + columns < owned_end)
            & (columns < features)
        )
        rows = tl.arange(0, block_edges).to(tl.int64)
        initial: tl.constexpr = -float("inf") if reduction == "max" else 0
        accumulator = tl.full((block_edges, block_features), initial, tl.float64 if use_float64 else tl.float32)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & owned[None, :]
            values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=initial).to(
                accumulator.dtype
            )
            if reduction == "max":
                accumulator = tl.maximum(accumulator, values)
            else:
                accumulator += values
        total = tl.max(accumulator, 0) if reduction == "max" else tl.sum(accumulator, 0)
        tl.store(out_ptr + segment * features + columns, total, mask=owned)


@triton.jit
def long_sum_finish(
    values_ptr,
    descriptors_ptr,
    partial_ptr,
    out_ptr,
    num_edges: tl.constexpr,
    features: tl.constexpr,
    use_float64: tl.constexpr,
    buckets: tl.constexpr,
    parts: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    weights_ptr=None,
    weight_groups: tl.constexpr = 0,
    weight_features: tl.constexpr = 1,
):
    part = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile * block_features + tl.arange(0, block_features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    tiles: tl.constexpr = triton.cdiv(features, block_features)
    counter_ptr = (descriptors_ptr + buckets * 3).to(tl.pointer_type(tl.int32))
    for bucket in range(buckets):
        segment = tl.load(descriptors_ptr + bucket * 3)
        start = tl.load(descriptors_ptr + bucket * 3 + 1)
        stop = tl.load(descriptors_ptr + bucket * 3 + 2)
        if start < stop:
            length = stop - start
            part_stop = start + length * (part + 1) // parts
            part_start = start + length * part // parts
            accumulator = tl.zeros((block_edges, block_features), tl.float64 if use_float64 else tl.float32)
            for begin in range(part_start, part_stop, block_edges):
                edges = begin + rows
                value = tl.load(
                    values_ptr + edges[:, None] * features + columns[None, :],
                    mask=(edges < part_stop)[:, None] & (columns < features)[None, :],
                    other=0,
                )
                contribution = value.to(accumulator.dtype)
                if weight_groups:
                    weights = tl.load(
                        weights_ptr + edges[:, None] * weight_groups + (columns // weight_features)[None, :],
                        mask=(edges < part_stop)[:, None] & (columns < features)[None, :],
                        other=0,
                    )
                    contribution *= weights.to(accumulator.dtype)
                accumulator += contribution
            partial = tl.sum(accumulator, 0)
            slot = (bucket * parts + part).to(tl.int64)
            tl.store(partial_ptr + slot * features + columns, partial, mask=columns < features, cache_modifier=".cg")
            # Publish every lane's partial before signaling block completion.
            tl.debug_barrier()
            ticket = tl.atomic_add(counter_ptr + bucket * tiles + tile, 1, sem="acq_rel", scope="gpu")
            tl.debug_barrier()
            if ticket == parts - 1:
                part_rows = bucket * parts + tl.arange(0, parts)
                value = tl.load(
                    partial_ptr + part_rows[:, None] * features + columns[None, :],
                    mask=(columns < features)[None, :],
                    other=0,
                    cache_modifier=".cg",
                )
                total = tl.sum(value, 0)
                tl.store(out_ptr + segment * features + columns, total, mask=columns < features)


@triton.jit
def reduction_forward_fused(
    values_ptr,
    offsets_ptr,
    out_ptr,
    partial_ptr,
    counter_ptr,
    inferred_segments: tl.constexpr,
    out_segments: tl.constexpr,
    num_edges: tl.constexpr,
    features: tl.constexpr,
    use_float64: tl.constexpr,
    buckets: tl.constexpr,
    parts: tl.constexpr,
    long_features: tl.constexpr,
    block_edges: tl.constexpr,
    block: tl.constexpr,
    threshold: tl.constexpr,
    reduction: tl.constexpr = "sum",
):
    initial: tl.constexpr = -float("inf") if reduction == "max" else 0
    work = tl.program_id(0)
    if out_segments * features >= 2**31:
        work = work.to(tl.int64)
    flat_programs: tl.constexpr = triton.cdiv(out_segments * features, block)
    feature_tiles: tl.constexpr = triton.cdiv(features, long_features)
    if work < flat_programs:
        indices = work * block + tl.arange(0, block)
        segment = indices // features
        column = indices % features
        live = indices < out_segments * features
        start = _bounded_edge(
            tl.load(offsets_ptr + segment, mask=live & (segment < inferred_segments), other=0), num_edges
        )
        stop = _bounded_edge(
            tl.load(offsets_ptr + segment + 1, mask=live & (segment < inferred_segments), other=0), num_edges
        )
        length = tl.maximum(stop - start, 0)
        short = length <= 32
        middle = (length > 32) & (length < threshold)
        length = tl.where(short, length, 0)
        acc = tl.full((block,), initial, tl.float64 if use_float64 else tl.float32)
        for row in range(0, tl.max(length, 0)):
            edge = start + row
            mask = live & (row < length)
            value = tl.load(values_ptr + edge * features + column, mask=mask, other=initial).to(acc.dtype)
            if reduction == "max":
                acc = tl.maximum(acc, value)
            else:
                acc += value
        tl.store(out_ptr + indices, acc, mask=live & short)
        middle_segment = tl.min(tl.where(live & middle, segment, out_segments), 0)
        while middle_segment < out_segments:
            middle_start = _bounded_edge(
                tl.load(offsets_ptr + middle_segment, mask=middle_segment < inferred_segments, other=0), num_edges
            )
            middle_stop = _bounded_edge(
                tl.load(offsets_ptr + middle_segment + 1, mask=middle_segment < inferred_segments, other=0),
                num_edges,
            )
            reduction_forward_middle(
                values_ptr,
                out_ptr,
                middle_start,
                middle_stop,
                middle_segment,
                features,
                work * block,
                work * block + block,
                use_float64,
                64,
                32,
                reduction,
            )
            middle_segment = tl.min(tl.where(live & middle & (segment > middle_segment), segment, out_segments), 0)

    else:
        extra = work - flat_programs
        bucket = extra // (parts * feature_tiles)
        part, feature_tile = extra // feature_tiles % parts, extra % feature_tiles
        columns = feature_tile * long_features + tl.arange(0, long_features)
        rows = tl.arange(0, block_edges).to(tl.int64)
        if bucket < buckets:
            segment, start, stop, active = long_start_bucket(
                offsets_ptr, num_edges, inferred_segments, bucket, threshold
            )
            if active:
                length = stop - start
                part_stop = start + length * (part + 1) // parts
                part_start = start + length * part // parts
                accumulator = tl.full((block_edges, long_features), initial, tl.float64 if use_float64 else tl.float32)
                for begin in range(part_start, part_stop, block_edges):
                    edges = begin + rows
                    values = tl.load(
                        values_ptr + edges[:, None] * features + columns[None, :],
                        mask=(edges < part_stop)[:, None] & (columns < features)[None, :],
                        other=initial,
                    )
                    if reduction == "max":
                        accumulator = tl.maximum(accumulator, values.to(accumulator.dtype))
                    else:
                        accumulator += values.to(accumulator.dtype)
                slot = (bucket * parts + part).to(tl.int64)
                partial = tl.max(accumulator, 0) if reduction == "max" else tl.sum(accumulator, 0)
                tl.store(
                    partial_ptr + slot * features + columns, partial, mask=columns < features, cache_modifier=".cg"
                )
                # Publish every lane's partial before signaling block completion.
                tl.debug_barrier()
                ticket = tl.atomic_add(
                    counter_ptr + bucket * feature_tiles + feature_tile, 1, sem="acq_rel", scope="gpu"
                )
                tl.debug_barrier()
                if ticket == parts - 1:
                    part_rows = bucket * parts + tl.arange(0, parts)
                    values = tl.load(
                        partial_ptr + part_rows[:, None] * features + columns[None, :],
                        mask=(columns < features)[None, :],
                        other=initial,
                        cache_modifier=".cg",
                    )
                    total = tl.max(values, 0) if reduction == "max" else tl.sum(values, 0)
                    tl.store(out_ptr + segment * features + columns, total, mask=columns < features)
