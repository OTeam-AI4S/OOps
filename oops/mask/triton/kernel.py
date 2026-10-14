from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _msa_member(
    msa_ptr,
    out_ptr,
    partial_ptr,
    begin: tl.constexpr,
    out_begin: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    block_rows: tl.constexpr,
    parts: tl.constexpr,
    tile_begin: tl.constexpr,
    partial_begin: tl.constexpr,
    block_tokens: tl.constexpr,
    wide: tl.constexpr,
):
    slot = tl.program_id(0) - tile_begin
    tiles: tl.constexpr = (n + block_tokens - 1) // block_tokens
    part = slot // tiles
    row = part * block_rows + tl.arange(0, block_rows)
    col = (slot % tiles) * block_tokens + tl.arange(0, block_tokens)
    if wide:
        row, col = row.to(tl.int64), col.to(tl.int64)
    values = tl.load(
        msa_ptr + begin + row[:, None] * n + col[None, :], mask=(row < m)[:, None] & (col < n)[None, :], other=0
    )
    result = tl.max(values.to(tl.int32), 0) != 0
    if parts == 1:
        tl.store(out_ptr + out_begin + col, result, mask=col < n)
    else:
        tl.store(partial_ptr + partial_begin + part * n + col, result, mask=col < n)


@triton.jit
def _msa_any_multi(msa_ptr, out_ptr, partial_ptr, layout: tl.constexpr, block_tokens: tl.constexpr, wide: tl.constexpr):
    program = tl.program_id(0)
    for sample in tl.static_range(len(layout)):
        if (program >= layout[sample][6]) & (
            program < layout[sample][6] + layout[sample][5] * ((layout[sample][3] + block_tokens - 1) // block_tokens)
        ):
            _msa_member(msa_ptr, out_ptr, partial_ptr, *layout[sample], block_tokens, wide)


@triton.jit
def _msa_merge_member(
    partial_ptr,
    out_ptr,
    partial_begin: tl.constexpr,
    out_begin: tl.constexpr,
    n: tl.constexpr,
    parts: tl.constexpr,
    tile_begin: tl.constexpr,
    block_tokens: tl.constexpr,
):
    col = (tl.program_id(0) - tile_begin) * block_tokens + tl.arange(0, block_tokens)
    part = tl.arange(0, triton.next_power_of_2(parts))
    values = tl.load(
        partial_ptr + partial_begin + part[:, None] * n + col[None, :],
        mask=(part < parts)[:, None] & (col < n)[None, :],
        other=0,
    )
    tl.store(out_ptr + out_begin + col, tl.max(values.to(tl.int32), 0) != 0, mask=col < n)


@triton.jit
def _msa_merge_multi(partial_ptr, out_ptr, layout: tl.constexpr, block_tokens: tl.constexpr):
    program = tl.program_id(0)
    for sample in tl.static_range(len(layout)):
        if (program >= layout[sample][4]) & (
            program < layout[sample][4] + ((layout[sample][2] + block_tokens - 1) // block_tokens)
        ):
            _msa_merge_member(partial_ptr, out_ptr, *layout[sample], block_tokens)


@triton.jit
def _pair_member(
    token_ptr,
    out_ptr,
    begin: tl.constexpr,
    out_begin: tl.constexpr,
    n: tl.constexpr,
    tile_begin: tl.constexpr,
    block: tl.constexpr,
    wide: tl.constexpr,
):
    index = (tl.program_id(0) - tile_begin) * block + tl.arange(0, block)
    if wide:
        index = index.to(tl.int64)
    live = index < n * n
    a = tl.load(token_ptr + begin + index // n, mask=live, other=0) != 0
    b = tl.load(token_ptr + begin + index % n, mask=live, other=0) != 0
    tl.store(out_ptr + out_begin + index, a & b, mask=live)


@triton.jit
def _pair_mask_multi(token_ptr, out_ptr, layout: tl.constexpr, block: tl.constexpr, wide: tl.constexpr):
    program = tl.program_id(0)
    for sample in tl.static_range(len(layout)):
        if (program >= layout[sample][3]) & (
            program < layout[sample][3] + ((layout[sample][2] * layout[sample][2] + block - 1) // block)
        ):
            _pair_member(token_ptr, out_ptr, *layout[sample], block, wide)
