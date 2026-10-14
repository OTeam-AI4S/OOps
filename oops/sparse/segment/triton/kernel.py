from __future__ import annotations

import triton
import triton.language as tl

from ...csr.triton.kernel import long_start_bucket


@triton.jit
def _bounded_edge(edge, num_edges):

    return tl.minimum(tl.maximum(edge.to(tl.int64), 0), num_edges)


@triton.jit
def _clear_uncovered_edges(
    offsets_ptr,
    output_ptr,
    num_edges,
    num_segments,
    features,
    segment,
    part,
    segment_parts: tl.constexpr,
    columns,
    rows,
    block_edges: tl.constexpr,
):
    if segment == 0 and part == 0:
        first = _bounded_edge(tl.load(offsets_ptr), num_edges)
        for begin in range(0, first, block_edges):
            edges = begin + rows
            tl.store(
                output_ptr + edges[:, None] * features + columns[None, :],
                0,
                mask=(edges < first)[:, None] & (columns < features)[None, :],
            )
    if segment == num_segments - 1 and part == segment_parts - 1:
        last = _bounded_edge(tl.load(offsets_ptr + num_segments), num_edges)
        for begin in range(last, num_edges, block_edges):
            edges = begin + rows
            tl.store(
                output_ptr + edges[:, None] * features + columns[None, :],
                0,
                mask=(edges < num_edges)[:, None] & (columns < features)[None, :],
            )


@triton.jit
def _segment_max_csr_forward(
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
        short_accumulator = tl.full((block_features,), -float("inf"), tl.float64 if use_float64 else tl.float32)
        for edge in range(start, stop):
            short_values = tl.load(values_ptr + edge * features + columns, mask=columns < features, other=-float("inf"))
            short_accumulator = tl.maximum(short_accumulator, short_values.to(short_accumulator.dtype))
        result = short_accumulator
    else:
        accumulator = tl.full((block_edges, block_features), -float("inf"), tl.float64 if use_float64 else tl.float32)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            values = tl.load(
                values_ptr + edges[:, None] * features + columns[None, :],
                mask=(edges < stop)[:, None] & (columns < features)[None, :],
                other=-float("inf"),
            )
            accumulator = tl.maximum(accumulator, values.to(accumulator.dtype))
        result = tl.max(accumulator, 0)
    tl.store(out_ptr + slot * features + columns, result, mask=columns < features)


@triton.jit
def _segment_max_forward_adaptive(
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
):
    work = tl.program_id(0).to(tl.int64)
    if work < out_segments:
        start = _bounded_edge(tl.load(row_offsets_ptr + work, mask=work < inferred_segments, other=0), num_edges)
        stop = _bounded_edge(tl.load(row_offsets_ptr + work + 1, mask=work < inferred_segments, other=0), num_edges)
        if stop - start < threshold:
            _segment_max_csr_forward(
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
    else:
        tiles: tl.constexpr = block_features // long_features
        extra = work - out_segments
        feature_start = tl.program_id(1) * block_features + (extra % tiles) * long_features
        segment, start, stop, active = long_start_bucket(
            row_offsets_ptr, num_edges, inferred_segments, extra // tiles, threshold
        )
        if active & (feature_start < features):
            columns = feature_start + tl.arange(0, long_features)
            rows = tl.arange(0, long_edges).to(tl.int64)
            accumulator = tl.full((long_edges, long_features), -float("inf"), tl.float64 if use_float64 else tl.float32)
            for begin in range(start, stop, long_edges):
                edges = begin + rows
                values = tl.load(
                    values_ptr + edges[:, None] * features + columns[None, :],
                    mask=(edges < stop)[:, None] & (columns < features)[None, :],
                    other=-float("inf"),
                )
                accumulator = tl.maximum(accumulator, values.to(accumulator.dtype))
            tl.store(out_ptr + segment * features + columns, tl.max(accumulator, 0), mask=columns < features)


@triton.jit
def _segment_sum_backward_adaptive(
    upstream_ptr,
    offsets_ptr,
    result_ptr,
    num_edges,
    num_segments,
    features,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    long_features: tl.constexpr,
    long_edges: tl.constexpr,
    threshold: tl.constexpr,
    long_parts: tl.constexpr = 1,
):
    work = tl.program_id(0).to(tl.int64)
    if work < num_segments:
        columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
        rows = tl.arange(0, block_edges).to(tl.int64)
        start = _bounded_edge(tl.load(offsets_ptr + work), num_edges)
        stop = _bounded_edge(tl.load(offsets_ptr + work + 1), num_edges)
        if stop - start < threshold:
            upstream = tl.load(upstream_ptr + work * features + columns, mask=columns < features, other=0)
            for begin in range(start, stop, block_edges):
                edges = begin + rows
                tl.store(
                    result_ptr + edges[:, None] * features + columns[None, :],
                    upstream[None, :],
                    mask=(edges < stop)[:, None] & (columns < features)[None, :],
                )
        _clear_uncovered_edges(
            offsets_ptr, result_ptr, num_edges, num_segments, features, work, 0, 1, columns, rows, block_edges
        )
    else:
        tiles: tl.constexpr = block_features // long_features
        extra = work - num_segments
        feature_start = tl.program_id(1) * block_features + (extra % tiles) * long_features
        segment, start, stop, active = long_start_bucket(
            offsets_ptr, num_edges, num_segments, extra // (tiles * long_parts), threshold
        )
        if active & (feature_start < features):
            part = (extra // tiles) % long_parts
            length = stop - start
            stop = start + length * (part + 1) // long_parts
            start += length * part // long_parts
            columns = feature_start + tl.arange(0, long_features)
            rows = tl.arange(0, long_edges).to(tl.int64)
            upstream = tl.load(upstream_ptr + segment * features + columns, mask=columns < features, other=0)
            for begin in range(start, stop, long_edges):
                edges = begin + rows
                tl.store(
                    result_ptr + edges[:, None] * features + columns[None, :],
                    upstream[None, :],
                    mask=(edges < stop)[:, None] & (columns < features)[None, :],
                )


@triton.jit
def _max_backward_body(
    values_ptr,
    maximum_ptr,
    upstream_ptr,
    result_ptr,
    segment,
    start,
    stop,
    features,
    feature_start,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
):
    columns = feature_start + tl.arange(0, block_features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    maximum = tl.load(maximum_ptr + segment * features + columns, mask=columns < features, other=0)
    accumulator = tl.full((block_edges, block_features), 0, tl.float64 if use_float64 else tl.float32)
    for begin in range(start, stop, block_edges):
        edges = begin + rows
        live = (edges < stop)[:, None] & (columns < features)[None, :]
        values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
        accumulator += (live & (values == maximum[None, :])).to(accumulator.dtype)
    total = tl.sum(accumulator, 0)
    upstream = tl.load(upstream_ptr + segment * features + columns, mask=columns < features, other=0)
    for begin in range(start, stop, block_edges):
        edges = begin + rows
        live = (edges < stop)[:, None] & (columns < features)[None, :]
        values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
        gradient = tl.where(values == maximum[None, :], upstream[None, :] / tl.maximum(total[None, :], 1), 0)
        tl.store(result_ptr + edges[:, None] * features + columns[None, :], gradient, mask=live)


@triton.jit
def _segment_max_backward_adaptive(
    values_ptr,
    maximum_ptr,
    upstream_ptr,
    offsets_ptr,
    result_ptr,
    num_edges,
    num_segments,
    features,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    long_features: tl.constexpr,
    long_edges: tl.constexpr,
    threshold: tl.constexpr,
    skip_long_rows: tl.constexpr = False,
    long_descriptors_ptr=None,
):
    work = tl.program_id(0).to(tl.int64)
    if work < num_segments:
        start = _bounded_edge(tl.load(offsets_ptr + work), num_edges)
        stop = _bounded_edge(tl.load(offsets_ptr + work + 1), num_edges)
        if stop - start < threshold:
            _max_backward_body(
                values_ptr,
                maximum_ptr,
                upstream_ptr,
                result_ptr,
                work,
                start,
                stop,
                features,
                tl.program_id(1) * block_features,
                use_float64,
                block_features,
                block_edges,
            )
        columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
        rows = tl.arange(0, block_edges).to(tl.int64)
        _clear_uncovered_edges(
            offsets_ptr, result_ptr, num_edges, num_segments, features, work, 0, 1, columns, rows, block_edges
        )
    elif skip_long_rows:
        if tl.program_id(1) == 0:
            bucket = work - num_segments
            segment, start, stop, active = long_start_bucket(offsets_ptr, num_edges, num_segments, bucket, threshold)
            tl.store(long_descriptors_ptr + bucket * 3, segment)
            tl.store(long_descriptors_ptr + bucket * 3 + 1, start)
            tl.store(long_descriptors_ptr + bucket * 3 + 2, tl.where(active, stop, start))
    else:
        tiles: tl.constexpr = block_features // long_features
        extra = work - num_segments
        feature_start = tl.program_id(1) * block_features + (extra % tiles) * long_features
        segment, start, stop, active = long_start_bucket(
            offsets_ptr, num_edges, num_segments, extra // tiles, threshold
        )
        if active & (feature_start < features):
            _max_backward_body(
                values_ptr,
                maximum_ptr,
                upstream_ptr,
                result_ptr,
                segment,
                start,
                stop,
                features,
                feature_start,
                use_float64,
                long_features,
                long_edges,
            )


@triton.jit
def _softmax_online_statistics(
    logits_ptr,
    start,
    stop,
    features,
    columns,
    rows,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
):
    maximum = tl.full((block_features,), -float("inf"), tl.float64 if use_float64 else tl.float32)
    denominator = tl.full((block_features,), 0, tl.float64 if use_float64 else tl.float32)
    for begin in range(start, stop, block_edges):
        edges = begin + rows
        live = (edges < stop)[:, None] & (columns < features)[None, :]
        logits = tl.load(logits_ptr + edges[:, None] * features + columns[None, :], mask=live, other=-float("inf"))
        logits = logits.to(maximum.dtype)
        next_max = tl.maximum(maximum, tl.max(logits, 0))
        safe_max = tl.where(next_max == -float("inf"), 0, next_max)
        scale = tl.exp(maximum - safe_max)
        terms = tl.where(live, tl.exp(logits - safe_max[None, :]), 0)
        denominator = denominator * scale + tl.sum(terms, 0)
        maximum = next_max
    return maximum, denominator


@triton.jit
def _segment_softmax_csr_forward(
    logits_ptr,
    row_offsets_ptr,
    out_ptr,
    partial_ptr,
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
    start = _bounded_edge(tl.load(row_offsets_ptr + segment), num_edges)
    stop = _bounded_edge(tl.load(row_offsets_ptr + segment + 1), num_edges)
    length = tl.maximum(stop - start, 0)
    stop = start + length * (part + 1) // segment_parts
    start += length * part // segment_parts
    maximum, denominator = _softmax_online_statistics(
        logits_ptr, start, stop, features, columns, rows, use_float64, block_features, block_edges
    )
    if segment_parts > 1:
        partial_base = slot * features + columns
        tl.store(partial_ptr + partial_base, maximum, mask=columns < features)
        tl.store(
            partial_ptr + inferred_segments * segment_parts * features + partial_base,
            denominator,
            mask=columns < features,
        )
    else:
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & (columns < features)[None, :]
            logits = tl.load(logits_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
            result = tl.exp(logits.to(maximum.dtype) - maximum[None, :]) / denominator[None, :]
            tl.store(out_ptr + edges[:, None] * features + columns[None, :], result, mask=live)

    if segment_parts == 1:
        _clear_uncovered_edges(
            row_offsets_ptr,
            out_ptr,
            num_edges,
            inferred_segments,
            features,
            segment,
            part,
            segment_parts,
            columns,
            rows,
            block_edges,
        )


@triton.jit
def _segment_csr_backward(
    values_ptr,
    output_ptr,
    grad_out_ptr,
    row_offsets_ptr,
    grad_values_ptr,
    partial_ptr,
    num_edges,
    num_segments,
    features,
    use_float64: tl.constexpr,
    reduction: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    segment_parts: tl.constexpr,
) -> None:
    slot = tl.program_id(0).to(tl.int64)
    segment = slot // segment_parts
    part = slot % segment_parts
    columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    start = _bounded_edge(tl.load(row_offsets_ptr + segment), num_edges)
    stop = _bounded_edge(tl.load(row_offsets_ptr + segment + 1), num_edges)
    length = tl.maximum(stop - start, 0)
    stop = start + length * (part + 1) // segment_parts
    start += length * part // segment_parts
    if reduction == "max":
        maximum = tl.load(output_ptr + segment * features + columns, mask=columns < features, other=0)
    if reduction != "sum":
        accumulator = tl.full((block_edges, block_features), 0, tl.float64 if use_float64 else tl.float32)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & (columns < features)[None, :]
            values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
            if reduction == "max":
                accumulator += (live & (values == maximum[None, :])).to(accumulator.dtype)
            else:
                upstream = tl.load(grad_out_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                accumulator += values.to(accumulator.dtype) * upstream.to(accumulator.dtype)
        total = tl.sum(accumulator, 0)
    if segment_parts > 1 and reduction != "sum":
        tl.store(partial_ptr + slot * features + columns, total, mask=columns < features)
    else:
        if reduction != "softmax":
            upstream = tl.load(grad_out_ptr + segment * features + columns, mask=columns < features, other=0)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & (columns < features)[None, :]
            if reduction == "sum":
                gradient = tl.broadcast_to(upstream[None, :], (block_edges, block_features))
            else:
                values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                if reduction == "max":
                    gradient = tl.where(
                        values == maximum[None, :], upstream[None, :] / tl.maximum(total[None, :], 1), 0
                    )
                else:
                    upstream = tl.load(grad_out_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                    gradient = values.to(total.dtype) * (upstream.to(total.dtype) - total[None, :])
            tl.store(grad_values_ptr + edges[:, None] * features + columns[None, :], gradient, mask=live)

    if segment_parts == 1 or reduction == "sum":
        _clear_uncovered_edges(
            row_offsets_ptr,
            grad_values_ptr,
            num_edges,
            num_segments,
            features,
            segment,
            part,
            segment_parts,
            columns,
            rows,
            block_edges,
        )


@triton.jit
def _segment_finish_slices(
    values_ptr,
    output_ptr,
    grad_out_ptr,
    offsets_ptr,
    partial_ptr,
    result_ptr,
    num_edges,
    num_segments,
    features,
    reduction: tl.constexpr,
    segment_parts: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
):
    slot = tl.program_id(0).to(tl.int64)
    segment = slot // segment_parts
    part = slot % segment_parts
    columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
    parts = tl.arange(0, segment_parts)
    partial_offsets = (segment * segment_parts + parts[:, None]) * features + columns[None, :]
    partial = tl.load(partial_ptr + partial_offsets, mask=(columns < features)[None, :], other=0)
    if reduction == "softmax_forward":
        maximum = tl.max(partial, 0)
        safe_maximum = tl.where(maximum == -float("inf"), 0, maximum)
        denominator = tl.load(
            partial_ptr + num_segments * segment_parts * features + partial_offsets,
            mask=(columns < features)[None, :],
            other=0,
        )
        total = tl.sum(denominator * tl.exp(partial - safe_maximum[None, :]), 0)
    else:
        total = tl.sum(partial, 0)
    if reduction == "max":
        maximum = tl.load(output_ptr + segment * features + columns, mask=columns < features, other=0)
        upstream = tl.load(grad_out_ptr + segment * features + columns, mask=columns < features, other=0)
    start = _bounded_edge(tl.load(offsets_ptr + segment), num_edges)
    stop = _bounded_edge(tl.load(offsets_ptr + segment + 1), num_edges)
    length = tl.maximum(stop - start, 0)
    stop = start + length * (part + 1) // segment_parts
    start += length * part // segment_parts
    rows = tl.arange(0, block_edges).to(tl.int64)
    for begin in range(start, stop, block_edges):
        edges = begin + rows
        live = (edges < stop)[:, None] & (columns < features)[None, :]
        offsets = edges[:, None] * features + columns[None, :]
        values = tl.load(values_ptr + offsets, mask=live, other=0)
        if reduction == "softmax_forward":
            result = tl.exp(values.to(total.dtype) - maximum[None, :]) / total[None, :]
        elif reduction == "max":
            result = tl.where(values == maximum[None, :], upstream[None, :] / tl.maximum(total[None, :], 1), 0)
        else:
            upstream = tl.load(grad_out_ptr + offsets, mask=live, other=0)
            result = values.to(total.dtype) * (upstream.to(total.dtype) - total[None, :])
        tl.store(result_ptr + offsets, result, mask=live)

    _clear_uncovered_edges(
        offsets_ptr,
        result_ptr,
        num_edges,
        num_segments,
        features,
        segment,
        part,
        segment_parts,
        columns,
        rows,
        block_edges,
    )


@triton.jit
def _softmax_feature_forward_body(
    logits_ptr,
    out_ptr,
    start,
    stop,
    features,
    feature_start,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
):
    columns = feature_start + tl.arange(0, block_features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    if stop - start <= block_edges:
        edges = start + rows
        live = (edges < stop)[:, None] & (columns < features)[None, :]
        short_values = tl.load(
            logits_ptr + edges[:, None] * features + columns[None, :], mask=live, other=-float("inf")
        )
        short_values = short_values.to(tl.float64 if use_float64 else tl.float32)
        short_maximum = tl.max(short_values, 0)
        short_terms = tl.exp(short_values - short_maximum[None, :])
        short_denominator = tl.sum(short_terms, 0)
        short_result = short_terms / short_denominator[None, :]
        tl.store(out_ptr + edges[:, None] * features + columns[None, :], short_result, mask=live)
    else:
        maximum, denominator = _softmax_online_statistics(
            logits_ptr, start, stop, features, columns, rows, use_float64, block_features, block_edges
        )
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & (columns < features)[None, :]
            values = tl.load(logits_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
            result = tl.exp(values.to(maximum.dtype) - maximum[None, :]) / denominator[None, :]
            tl.store(out_ptr + edges[:, None] * features + columns[None, :], result, mask=live)


@triton.jit
def _softmax_feature_backward_body(
    values_ptr,
    grad_out_ptr,
    grad_values_ptr,
    start,
    stop,
    features,
    feature_start,
    use_float64: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
):
    columns = feature_start + tl.arange(0, block_features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    if stop - start <= block_edges:
        edges = start + rows
        live = (edges < stop)[:, None] & (columns < features)[None, :]
        short_values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
        short_upstream = tl.load(grad_out_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
        short_values = short_values.to(tl.float64 if use_float64 else tl.float32)
        short_upstream = short_upstream.to(short_values.dtype)
        short_total = tl.sum(short_values * short_upstream, 0)
        short_gradient = short_values * (short_upstream - short_total[None, :])
        tl.store(grad_values_ptr + edges[:, None] * features + columns[None, :], short_gradient, mask=live)
    else:
        accumulator = tl.full((block_edges, block_features), 0, tl.float64 if use_float64 else tl.float32)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & (columns < features)[None, :]
            values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
            upstream = tl.load(grad_out_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
            accumulator += values.to(accumulator.dtype) * upstream.to(accumulator.dtype)
        total = tl.sum(accumulator, 0)
        for begin in range(start, stop, block_edges):
            edges = begin + rows
            live = (edges < stop)[:, None] & (columns < features)[None, :]
            values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
            upstream = tl.load(grad_out_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
            gradient = values.to(total.dtype) * (upstream.to(total.dtype) - total[None, :])
            tl.store(grad_values_ptr + edges[:, None] * features + columns[None, :], gradient, mask=live)


@triton.jit
def _segment_softmax_adaptive(
    values_ptr,
    grad_out_ptr,
    out_ptr,
    row_offsets_ptr,
    num_edges,
    segments,
    features,
    use_float64: tl.constexpr,
    backward: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    long_features: tl.constexpr,
    long_edges: tl.constexpr,
    threshold: tl.constexpr,
    skip_long_rows: tl.constexpr = False,
    long_descriptors_ptr=None,
    pack_short_rows: tl.constexpr = False,
):
    slot = tl.program_id(0).to(tl.int64)
    if slot < segments:
        segment = slot
        feature_start = tl.program_id(1) * block_features
        start = _bounded_edge(tl.load(row_offsets_ptr + segment), num_edges)
        stop = _bounded_edge(tl.load(row_offsets_ptr + segment + 1), num_edges)
        if pack_short_rows and stop - start <= block_edges:
            if tl.program_id(1) == 0:
                if backward:
                    _softmax_feature_backward_body(
                        values_ptr, grad_out_ptr, out_ptr, start, stop, features, 0, use_float64, 128, block_edges
                    )
                else:
                    _softmax_feature_forward_body(
                        values_ptr, out_ptr, start, stop, features, 0, use_float64, 128, block_edges
                    )
        elif stop - start < threshold:
            if backward:
                _softmax_feature_backward_body(
                    values_ptr,
                    grad_out_ptr,
                    out_ptr,
                    start,
                    stop,
                    features,
                    feature_start,
                    use_float64,
                    block_features,
                    block_edges,
                )
            else:
                _softmax_feature_forward_body(
                    values_ptr, out_ptr, start, stop, features, feature_start, use_float64, block_features, block_edges
                )
        columns = feature_start + tl.arange(0, block_features)
        rows = tl.arange(0, block_edges).to(tl.int64)
        _clear_uncovered_edges(
            row_offsets_ptr, out_ptr, num_edges, segments, features, segment, 0, 1, columns, rows, block_edges
        )
    elif skip_long_rows:
        if tl.program_id(1) == 0:
            bucket = slot - segments
            _segment, start, stop, active = long_start_bucket(row_offsets_ptr, num_edges, segments, bucket, threshold)
            tl.store(long_descriptors_ptr + bucket * 2, start)
            tl.store(long_descriptors_ptr + bucket * 2 + 1, tl.where(active, stop, start))
    else:
        extra = slot - segments
        long_tiles: tl.constexpr = block_features // long_features
        bucket = extra // long_tiles
        feature_start = tl.program_id(1) * block_features + (extra % long_tiles).to(tl.int32) * long_features
        segment, start, stop, active = long_start_bucket(row_offsets_ptr, num_edges, segments, bucket, threshold)
        if active & (feature_start < features):
            if backward:
                _softmax_feature_backward_body(
                    values_ptr,
                    grad_out_ptr,
                    out_ptr,
                    start,
                    stop,
                    features,
                    feature_start,
                    use_float64,
                    long_features,
                    long_edges,
                )
            else:
                _softmax_feature_forward_body(
                    values_ptr, out_ptr, start, stop, features, feature_start, use_float64, long_features, long_edges
                )


@triton.jit
def _segment_softmax_long_statistics(
    values_ptr,
    grad_out_ptr,
    long_descriptors_ptr,
    partial_ptr,
    num_edges,
    segments,
    features,
    use_float64: tl.constexpr,
    backward: tl.constexpr,
    parts: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    threshold: tl.constexpr,
):
    for bucket in range(num_edges // threshold):
        start = tl.load(long_descriptors_ptr + bucket * 2)
        stop = tl.load(long_descriptors_ptr + bucket * 2 + 1)
        if start < stop:
            row_start, row_stop = start, stop
            for part in range(tl.program_id(0), parts, tl.num_programs(0)):
                slot = bucket * parts + part
                length = tl.maximum(row_stop - row_start, 0)
                stop = row_start + length * (part + 1) // parts
                start = row_start + length * part // parts
                columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
                rows = tl.arange(0, block_edges).to(tl.int64)
                if backward:
                    accumulator = tl.full((block_edges, block_features), 0, tl.float64 if use_float64 else tl.float32)
                    for begin in range(start, stop, block_edges):
                        edges = begin + rows
                        live = (edges < stop)[:, None] & (columns < features)[None, :]
                        values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                        upstream = tl.load(
                            grad_out_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0
                        )
                        accumulator += values.to(accumulator.dtype) * upstream.to(accumulator.dtype)
                    tl.store(partial_ptr + slot * features + columns, tl.sum(accumulator, 0), mask=(columns < features))
                else:
                    maximum, denominator = _softmax_online_statistics(
                        values_ptr, start, stop, features, columns, rows, use_float64, block_features, block_edges
                    )
                    tl.store(partial_ptr + (slot * 2) * features + columns, maximum, mask=(columns < features))
                    tl.store(partial_ptr + (slot * 2 + 1) * features + columns, denominator, mask=(columns < features))


@triton.jit
def _segment_softmax_long_finish(
    values_ptr,
    grad_out_ptr,
    long_descriptors_ptr,
    partial_ptr,
    out_ptr,
    num_edges,
    segments,
    features,
    use_float64: tl.constexpr,
    backward: tl.constexpr,
    parts: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    threshold: tl.constexpr,
):
    for bucket in range(num_edges // threshold):
        start = tl.load(long_descriptors_ptr + bucket * 2)
        stop = tl.load(long_descriptors_ptr + bucket * 2 + 1)
        if start < stop:
            row_start, row_stop = start, stop
            for part in range(tl.program_id(0), parts, tl.num_programs(0)):
                length = tl.maximum(row_stop - row_start, 0)
                stop = row_start + length * (part + 1) // parts
                start = row_start + length * part // parts
                columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
                part_offsets = bucket * parts + tl.arange(0, parts)
                if backward:
                    statistics = tl.load(
                        partial_ptr + part_offsets[:, None] * features + columns[None, :],
                        mask=(columns < features)[None, :],
                        other=0,
                    )
                    total = tl.sum(statistics, 0)
                else:
                    maxima = tl.load(
                        partial_ptr + (part_offsets[:, None] * 2) * features + columns[None, :],
                        mask=(columns < features)[None, :],
                        other=-float("inf"),
                    )
                    denominators = tl.load(
                        partial_ptr + (part_offsets[:, None] * 2 + 1) * features + columns[None, :],
                        mask=(columns < features)[None, :],
                        other=0,
                    )
                    maximum = tl.max(maxima, 0)
                    safe_maximum = tl.where(maximum == -float("inf"), 0, maximum)
                    total = tl.sum(denominators * tl.exp(maxima - safe_maximum[None, :]), 0)
                rows = tl.arange(0, block_edges).to(tl.int64)
                for begin in range(start, stop, block_edges):
                    edges = begin + rows
                    live = (edges < stop)[:, None] & (columns < features)[None, :]
                    values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                    values = values.to(tl.float64 if use_float64 else tl.float32)
                    if backward:
                        upstream = tl.load(
                            grad_out_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0
                        )
                        result = values * (upstream.to(values.dtype) - total[None, :])
                    else:
                        result = tl.exp(values - maximum[None, :]) / total[None, :]
                    tl.store(out_ptr + edges[:, None] * features + columns[None, :], result, mask=live)


@triton.jit
def _segment_max_long_statistics(
    values_ptr,
    maxima_ptr,
    descriptors_ptr,
    partial_ptr,
    num_edges,
    features,
    use_float64: tl.constexpr,
    parts: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    threshold: tl.constexpr,
):
    for slot in range(tl.program_id(0).to(tl.int64), (num_edges // threshold) * parts, tl.num_programs(0)):
        bucket, part = slot // parts, slot % parts
        segment = tl.load(descriptors_ptr + bucket * 3)
        start = tl.load(descriptors_ptr + bucket * 3 + 1)
        stop = tl.load(descriptors_ptr + bucket * 3 + 2)
        if start < stop:
            length = stop - start
            stop = start + length * (part + 1) // parts
            start += length * part // parts
            columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
            rows = tl.arange(0, block_edges).to(tl.int64)
            maximum = tl.load(maxima_ptr + segment * features + columns, mask=columns < features, other=0)
            count = tl.full((block_edges, block_features), 0, tl.float64 if use_float64 else tl.float32)
            for begin in range(start, stop, block_edges):
                edges = begin + rows
                live = (edges < stop)[:, None] & (columns < features)[None, :]
                values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                count += (live & (values == maximum[None, :])).to(count.dtype)
            tl.store(partial_ptr + slot * features + columns, tl.sum(count, 0), mask=columns < features)


@triton.jit
def _segment_max_long_finish(
    values_ptr,
    maxima_ptr,
    upstream_ptr,
    descriptors_ptr,
    partial_ptr,
    result_ptr,
    num_edges,
    features,
    parts: tl.constexpr,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
    threshold: tl.constexpr,
):
    for slot in range(tl.program_id(0).to(tl.int64), (num_edges // threshold) * parts, tl.num_programs(0)):
        bucket, part = slot // parts, slot % parts
        segment = tl.load(descriptors_ptr + bucket * 3)
        start = tl.load(descriptors_ptr + bucket * 3 + 1)
        stop = tl.load(descriptors_ptr + bucket * 3 + 2)
        if start < stop:
            length = stop - start
            stop = start + length * (part + 1) // parts
            start += length * part // parts
            columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
            part_offsets = bucket * parts + tl.arange(0, parts)
            counts = tl.load(
                partial_ptr + part_offsets[:, None] * features + columns[None, :],
                mask=(columns < features)[None, :],
                other=0,
            )
            count = tl.maximum(tl.sum(counts, 0), 1)
            maximum = tl.load(maxima_ptr + segment * features + columns, mask=columns < features, other=0)
            upstream = tl.load(upstream_ptr + segment * features + columns, mask=columns < features, other=0)
            rows = tl.arange(0, block_edges).to(tl.int64)
            for begin in range(start, stop, block_edges):
                edges = begin + rows
                live = (edges < stop)[:, None] & (columns < features)[None, :]
                values = tl.load(values_ptr + edges[:, None] * features + columns[None, :], mask=live, other=0)
                gradient = tl.where(values == maximum[None, :], upstream[None, :] / count[None, :], 0)
                tl.store(result_ptr + edges[:, None] * features + columns[None, :], gradient, mask=live)


@triton.jit
def _segment_sum_backward_edges(
    grad_out_ptr,
    row_offsets_ptr,
    result_ptr,
    num_edges,
    num_segments,
    features,
    block_rows: tl.constexpr,
    block_features: tl.constexpr,
    search_steps: tl.constexpr,
):
    edges = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    columns = tl.program_id(1) * block_features + tl.arange(0, block_features)
    low = tl.full((block_rows,), 0, tl.int64)
    high = tl.full((block_rows,), num_segments, tl.int64)
    for _ in range(search_steps):
        middle = (low + high) // 2
        stop = _bounded_edge(
            tl.load(row_offsets_ptr + middle + 1, mask=middle < num_segments, other=num_edges), num_edges
        )
        take_right = (low < high) & (stop <= edges)
        high = tl.where((low < high) & ~take_right, middle, high)
        low = tl.where(take_right, middle + 1, low)
    first = _bounded_edge(tl.load(row_offsets_ptr), num_edges)
    covered = (edges < num_edges) & (edges >= first) & (low < num_segments)
    gradients = tl.load(
        grad_out_ptr + low[:, None] * features + columns[None, :],
        mask=covered[:, None] & (columns < features)[None, :],
        other=0,
    )
    tl.store(
        result_ptr + edges[:, None] * features + columns[None, :],
        gradients,
        mask=(edges < num_edges)[:, None] & (columns < features)[None, :],
    )


@triton.jit
def _sum_backward_flat(
    grad_ptr,
    offsets_ptr,
    out_ptr,
    num_edges: tl.constexpr,
    segments: tl.constexpr,
    features: tl.constexpr,
    block: tl.constexpr,
    long_parts: tl.constexpr,
    block_edges: tl.constexpr,
    threshold: tl.constexpr,
):
    work = tl.program_id(0)
    flats: tl.constexpr = triton.cdiv(segments * features, block)
    if work < flats:
        index = work * block + tl.arange(0, block)
        if segments * features >= 2**31:
            index = index.to(tl.int64)
        segment = index // features
        column = index % features
        valid = index < segments * features
        start = _bounded_edge(tl.load(offsets_ptr + segment, mask=valid, other=0), num_edges)
        stop = _bounded_edge(tl.load(offsets_ptr + segment + 1, mask=valid, other=0), num_edges)
        length = tl.maximum(stop - start, 0)
        middle = (length > 32) & (length < threshold)
        length = tl.where(length <= 32, length, 0)
        upstream = tl.load(grad_ptr + index, mask=valid, other=0)
        for row in range(tl.max(length, 0)):
            tl.store(out_ptr + (start + row) * features + column, upstream, mask=valid & (row < length))
        middle_segment = tl.min(tl.where(valid & middle, segment, segments), 0)
        while middle_segment < segments:
            middle_start = _bounded_edge(tl.load(offsets_ptr + middle_segment), num_edges)
            middle_stop = _bounded_edge(tl.load(offsets_ptr + middle_segment + 1), num_edges)
            _sum_middle_backward(
                grad_ptr,
                out_ptr,
                middle_start,
                middle_stop,
                middle_segment,
                features,
                work * block,
                work * block + block,
                64,
                32,
            )
            middle_segment = tl.min(tl.where(valid & middle & (segment > middle_segment), segment, segments), 0)

    else:
        extra = work - flats
        bucket = extra // long_parts
        part = extra % long_parts
        long_segment, long_start, long_stop, active = long_start_bucket(
            offsets_ptr, num_edges, segments, bucket, threshold
        )
        if active:
            length = long_stop - long_start
            end = long_start + length * (part + 1) // long_parts
            begin = long_start + length * part // long_parts
            columns = tl.arange(0, block)
            upstream = tl.load(grad_ptr + long_segment * features + columns, mask=columns < features, other=0)
            rows = tl.arange(0, block_edges).to(tl.int64)
            for first in range(begin, end, block_edges):
                edge = first + rows
                tl.store(
                    out_ptr + edge[:, None] * features + columns[None, :],
                    upstream[None, :],
                    mask=(edge < end)[:, None] & (columns < features)[None, :],
                )
    if work == 0:
        columns = tl.arange(0, block)
        prefix = _bounded_edge(tl.load(offsets_ptr), num_edges)
        suffix = _bounded_edge(tl.load(offsets_ptr + segments), num_edges)
        rows = tl.arange(0, block_edges).to(tl.int64)
        for begin in range(0, prefix, block_edges):
            edge = begin + rows
            tl.store(
                out_ptr + edge[:, None] * features + columns[None, :],
                0,
                mask=(edge < prefix)[:, None] & (columns < features)[None, :],
            )
        for begin in range(suffix, num_edges, block_edges):
            edge = begin + rows
            tl.store(
                out_ptr + edge[:, None] * features + columns[None, :],
                0,
                mask=(edge < num_edges)[:, None] & (columns < features)[None, :],
            )


@triton.jit
def _sum_middle_backward(
    grad_ptr,
    out_ptr,
    start,
    stop,
    segment,
    features,
    owned_begin,
    owned_end,
    block_features: tl.constexpr,
    block_edges: tl.constexpr,
):
    first = tl.maximum(0, owned_begin - segment * features) // block_features * block_features
    last = tl.minimum(features, owned_end - segment * features)
    rows = tl.arange(0, block_edges).to(tl.int64)
    for column_begin in range(first, last, block_features):
        columns = column_begin + tl.arange(0, block_features)
        owned = (
            (segment * features + columns >= owned_begin)
            & (segment * features + columns < owned_end)
            & (columns < features)
        )
        upstream = tl.load(grad_ptr + segment * features + columns, mask=owned, other=0)
        for first_edge in range(start, stop, block_edges):
            edges = first_edge + rows
            tl.store(
                out_ptr + edges[:, None] * features + columns[None, :],
                upstream[None, :],
                mask=(edges < stop)[:, None] & owned[None, :],
            )
