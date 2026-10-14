import triton as tr
import triton.language as tl


@tr.jit
def _row_offsets(rows, tokens, groups, stride_batch, stride_group, stride_token):
    token = rows % tokens
    sample = rows // tokens
    return (sample // groups) * stride_batch + (sample % groups) * stride_group + token * stride_token


@tr.jit
def _column_offsets(columns, stride_head, stride_channel, D: tl.constexpr):
    head = columns // D
    return head * stride_head + (columns - head * D) * stride_channel


@tr.jit
def _gate_forward(
    attention_ptr,
    logits_ptr,
    output_ptr,
    num_rows,
    tokens,
    groups,
    stride_batch,
    stride_group,
    stride_token,
    stride_head,
    stride_channel,
    H: tl.constexpr,
    D: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    columns = tl.program_id(1) * block_columns + tl.arange(0, block_columns)
    active = (rows < num_rows)[:, None] & (columns < H * D)[None, :]
    offset = (
        _row_offsets(rows, tokens, groups, stride_batch, stride_group, stride_token)[:, None]
        + _column_offsets(columns, stride_head, stride_channel, D)[None, :]
    )
    index = rows[:, None] * (H * D) + columns[None, :]
    attention = tl.load(attention_ptr + offset, active, 0).to(tl.float32)
    logits = tl.load(logits_ptr + index, active, 0).to(tl.float32)
    # Round sigmoid before multiplication to match its storage boundary.
    gate = tl.sigmoid(logits).to(logits_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + index, attention * gate, active)


@tr.jit
def _gate_backward(
    attention_ptr,
    logits_ptr,
    grad_output_ptr,
    grad_attention_ptr,
    grad_logits_ptr,
    num_rows,
    tokens,
    groups,
    stride_batch,
    stride_group,
    stride_token,
    stride_head,
    stride_channel,
    H: tl.constexpr,
    D: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
    need_attention: tl.constexpr,
    need_logits: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    columns = tl.program_id(1) * block_columns + tl.arange(0, block_columns)
    active = (rows < num_rows)[:, None] & (columns < H * D)[None, :]
    offset = (
        _row_offsets(rows, tokens, groups, stride_batch, stride_group, stride_token)[:, None]
        + _column_offsets(columns, stride_head, stride_channel, D)[None, :]
    )
    index = rows[:, None] * (H * D) + columns[None, :]
    logits = tl.load(logits_ptr + index, active, 0).to(tl.float32)
    grad = tl.load(grad_output_ptr + index, active, 0).to(tl.float32)
    gate = tl.sigmoid(logits).to(logits_ptr.dtype.element_ty).to(tl.float32)
    if need_attention:
        tl.store(grad_attention_ptr + index, grad * gate, active)
    if need_logits:
        attention = tl.load(attention_ptr + offset, active, 0).to(tl.float32)
        grad_gate = (grad * attention).to(logits_ptr.dtype.element_ty).to(tl.float32)
        # Low-precision sigmoid backward rounds each scalar operation.
        complement = (1.0 - gate).to(logits_ptr.dtype.element_ty).to(tl.float32)
        intermediate = (grad_gate * complement).to(logits_ptr.dtype.element_ty).to(tl.float32)
        tl.store(grad_logits_ptr + index, intermediate * gate, active)


@tr.jit
def _gate_projection_body(
    single_ptr,
    gate_weight_ptr,
    attention_ptr,
    output_ptr,
    logits_ptr,
    num_rows,
    single_channels,
    stride_single_row,
    stride_single_channel,
    stride_weight_row,
    stride_weight_channel,
    tokens,
    groups,
    stride_batch,
    stride_group,
    stride_token,
    stride_head,
    stride_channel,
    H: tl.constexpr,
    D: tl.constexpr,
    block_rows: tl.constexpr,
    block_gate: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
    store_logits: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    columns = tl.program_id(1) * block_gate + tl.arange(0, block_gate)
    reduction = tl.arange(0, block_reduction)
    row_mask = rows < num_rows
    column_mask = columns < H * D
    accumulator = tl.zeros((block_rows, block_gate), tl.float32)
    for block in range(tl.cdiv(single_channels, block_reduction)):
        channels = block * block_reduction + reduction
        channel_mask = channels < single_channels
        single = tl.load(
            single_ptr + rows[:, None] * stride_single_row + channels[None, :] * stride_single_channel,
            row_mask[:, None] & channel_mask[None, :],
            0,
        ).to(output_ptr.dtype.element_ty)
        weight = tl.load(
            gate_weight_ptr + columns[None, :] * stride_weight_row + channels[:, None] * stride_weight_channel,
            column_mask[None, :] & channel_mask[:, None],
            0,
        ).to(output_ptr.dtype.element_ty)
        accumulator = tl.dot(single, weight, accumulator, input_precision=input_precision)
    logits = accumulator.to(output_ptr.dtype.element_ty)
    index = rows[:, None] * (H * D) + columns[None, :]
    active = row_mask[:, None] & column_mask[None, :]
    if store_logits:
        tl.store(logits_ptr + index, logits, active)
    gate = tl.sigmoid(logits.to(tl.float32)).to(output_ptr.dtype.element_ty).to(tl.float32)
    offset = (
        _row_offsets(rows, tokens, groups, stride_batch, stride_group, stride_token)[:, None]
        + _column_offsets(columns, stride_head, stride_channel, D)[None, :]
    )
    attention = tl.load(attention_ptr + offset, active, 0).to(tl.float32)
    tl.store(output_ptr + index, gate * attention, active)


@tr.jit
def _gate_projection_forward(
    single_ptr,
    gate_weight_ptr,
    attention_ptr,
    output_ptr,
    logits_ptr,
    num_rows,
    single_channels,
    stride_single_row,
    stride_single_channel,
    stride_weight_row,
    stride_weight_channel,
    tokens,
    groups,
    stride_batch,
    stride_group,
    stride_token,
    stride_head,
    stride_channel,
    H: tl.constexpr,
    D: tl.constexpr,
    block_rows: tl.constexpr,
    block_gate: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
):
    _gate_projection_body(
        single_ptr,
        gate_weight_ptr,
        attention_ptr,
        output_ptr,
        logits_ptr,
        num_rows,
        single_channels,
        stride_single_row,
        stride_single_channel,
        stride_weight_row,
        stride_weight_channel,
        tokens,
        groups,
        stride_batch,
        stride_group,
        stride_token,
        stride_head,
        stride_channel,
        H,
        D,
        block_rows,
        block_gate,
        block_reduction,
        input_precision,
        True,
    )


@tr.jit
def _gate_projection_inference(
    single_ptr,
    gate_weight_ptr,
    attention_ptr,
    output_ptr,
    num_rows,
    single_channels,
    stride_single_row,
    stride_single_channel,
    stride_weight_row,
    stride_weight_channel,
    tokens,
    groups,
    stride_batch,
    stride_group,
    stride_token,
    stride_head,
    stride_channel,
    H: tl.constexpr,
    D: tl.constexpr,
    block_rows: tl.constexpr,
    block_gate: tl.constexpr,
    block_reduction: tl.constexpr,
    input_precision: tl.constexpr,
):
    _gate_projection_body(
        single_ptr,
        gate_weight_ptr,
        attention_ptr,
        output_ptr,
        output_ptr,
        num_rows,
        single_channels,
        stride_single_row,
        stride_single_channel,
        stride_weight_row,
        stride_weight_channel,
        tokens,
        groups,
        stride_batch,
        stride_group,
        stride_token,
        stride_head,
        stride_channel,
        H,
        D,
        block_rows,
        block_gate,
        block_reduction,
        input_precision,
        False,
    )
