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
def _bounded_token(token, tokens):

    return tl.minimum(tl.maximum(token.to(tl.int64), 0), tokens - 1)


@triton.jit
def _token_pair_to_atom_window_gather_forward(
    token_pair_ptr,
    atom_to_token_ptr,
    out_ptr,
    total_out: tl.constexpr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    n_blocks: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    pad_left: tl.constexpr,
    block_elements: tl.constexpr,
    use_int64: tl.constexpr = True,
) -> None:
    index_dtype = tl.int64 if use_int64 else tl.int32
    offsets = tl.program_id(0).to(index_dtype) * block_elements + tl.arange(0, block_elements)
    mask = offsets < total_out

    c = offsets % channels
    h = (offsets // channels) % n_keys
    row_width = tl.full((), channels, index_dtype) * n_keys
    block_width = row_width * n_queries
    q = (offsets // row_width) % n_queries
    block = (offsets // block_width) % n_blocks
    batch = offsets // (block_width * n_blocks)

    q_atom = block * n_queries + q
    k_atom = block * n_queries + h - pad_left
    q_index = tl.load(atom_to_token_ptr + q_atom, mask=mask & (q_atom < atoms), other=0)
    k_index = tl.load(
        atom_to_token_ptr + k_atom,
        mask=mask & (k_atom >= 0) & (k_atom < atoms),
        other=0,
    )
    q_token = _bounded_token(q_index, tokens).to(index_dtype)
    k_token = _bounded_token(k_index, tokens).to(index_dtype)

    token_pair_offsets = (batch * tokens + q_token) * tokens * channels + k_token * channels + c
    values = tl.load(token_pair_ptr + token_pair_offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _token_pair_to_atom_window_gather_backward(
    grad_out_ptr,
    atom_to_token_ptr,
    grad_token_pair_ptr,
    total_out: tl.constexpr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    n_blocks: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    pad_left: tl.constexpr,
    block_elements: tl.constexpr,
    use_int64: tl.constexpr = True,
) -> None:
    index_dtype = tl.int64 if use_int64 else tl.int32
    offsets = tl.program_id(0).to(index_dtype) * block_elements + tl.arange(0, block_elements)
    mask = offsets < total_out

    c = offsets % channels
    h = (offsets // channels) % n_keys
    row_width = tl.full((), channels, index_dtype) * n_keys
    block_width = row_width * n_queries
    q = (offsets // row_width) % n_queries
    block = (offsets // block_width) % n_blocks
    batch = offsets // (block_width * n_blocks)

    q_atom = block * n_queries + q
    k_atom = block * n_queries + h - pad_left
    q_index = tl.load(atom_to_token_ptr + q_atom, mask=mask & (q_atom < atoms), other=0)
    k_index = tl.load(
        atom_to_token_ptr + k_atom,
        mask=mask & (k_atom >= 0) & (k_atom < atoms),
        other=0,
    )
    q_token = _bounded_token(q_index, tokens).to(index_dtype)
    k_token = _bounded_token(k_index, tokens).to(index_dtype)

    values = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    token_pair_offsets = (batch * tokens + q_token) * tokens * channels + k_token * channels + c
    tl.atomic_add(grad_token_pair_ptr + token_pair_offsets, values, sem="relaxed", mask=mask)


@triton.jit
def _segmented_add(left_head, left_value, right_head, right_value):
    return left_head | right_head, tl.where(right_head, right_value, left_value + right_value)


@triton.jit
def _token_pair_to_atom_window_grouped_backward(
    grad_out_ptr,
    atom_to_token_ptr,
    grad_token_pair_ptr,
    total_points: tl.constexpr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    n_blocks: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    pad_left: tl.constexpr,
    block_points: tl.constexpr,
    block_channels: tl.constexpr,
    use_int64: tl.constexpr,
):
    index_dtype = tl.int64 if use_int64 else tl.int32
    lane = tl.arange(0, block_points)
    point = tl.program_id(0).to(index_dtype) * block_points + lane
    active = point < total_points
    h = point % n_keys
    q = (point // n_keys) % n_queries
    block = (point // (n_keys * n_queries)) % n_blocks
    batch = point // (n_keys * n_queries * n_blocks)
    q_atom = block * n_queries + q
    k_atom = block * n_queries + h - pad_left
    q_index = tl.load(atom_to_token_ptr + q_atom, mask=active & (q_atom < atoms), other=0)
    k_index = tl.load(atom_to_token_ptr + k_atom, mask=active & (k_atom >= 0) & (k_atom < atoms), other=0)
    q_token = _bounded_token(q_index, tokens).to(index_dtype)
    k_token = _bounded_token(k_index, tokens).to(index_dtype)
    target = (batch * tokens + q_token) * tokens + k_token
    label = tl.where(active, target, -1)
    previous = tl.gather(label, tl.maximum(lane - 1, 0), axis=0)
    following = tl.gather(label, tl.minimum(lane + 1, block_points - 1), axis=0)
    head = (lane == 0) | (label != previous)
    tail = (lane == block_points - 1) | (label != following)
    c = tl.arange(0, block_channels)
    mask = active[:, None] & (c < channels)[None, :]
    values = tl.load(grad_out_ptr + point[:, None] * channels + c[None, :], mask=mask, other=0).to(tl.float32)
    flags = tl.broadcast_to(head[:, None], (block_points, block_channels))
    _, sums = tl.associative_scan((flags, values), axis=0, combine_fn=_segmented_add)
    tl.atomic_add(
        grad_token_pair_ptr + target[:, None] * channels + c[None, :],
        sums,
        mask=mask & tail[:, None],
        sem="relaxed",
    )


@triton.jit
def _nested_token_pair_to_atom_window_gather_forward(
    token_pair_ptr,
    atom_to_token_ptr,
    token_offsets_ptr,
    atom_offsets_ptr,
    metadata_ptr,
    block_to_sample_ptr,
    out_ptr,
    num_samples,
    channels: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    pad_left: tl.constexpr,
    has_num_valid_atoms: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:

    output_block = tl.program_id(0).to(tl.int64)
    within = tl.program_id(1).to(tl.int64) * block_elements + tl.arange(0, block_elements)
    block_width = channels * n_queries * n_keys
    mask = within < block_width

    c = within % channels
    h = (within // channels) % n_keys
    q = within // (channels * n_keys)
    metadata_stride = num_samples + 1
    sample = tl.load(block_to_sample_ptr + output_block).to(tl.int64)
    local_block = output_block - tl.load(metadata_ptr + metadata_stride + sample).to(tl.int64)

    token_start = tl.load(token_offsets_ptr + sample).to(tl.int64)
    token_end = tl.load(token_offsets_ptr + sample + 1).to(tl.int64)
    tokens = token_end - token_start
    atom_start = tl.load(atom_offsets_ptr + sample).to(tl.int64)
    atom_end = tl.load(atom_offsets_ptr + sample + 1).to(tl.int64)
    atoms = atom_end - atom_start

    q_atom = local_block * n_queries + q
    q_index = tl.load(
        atom_to_token_ptr + atom_start + q_atom,
        mask=mask & (q_atom < atoms),
        other=0,
    )
    if has_num_valid_atoms:
        valid_atoms = tl.load(metadata_ptr + 2 * metadata_stride + sample).to(tl.int64)
        centered_start = local_block * n_queries - pad_left
        key_start = tl.minimum(tl.maximum(centered_start, 0), tl.maximum(valid_atoms - n_keys, 0))
        k_atom = key_start + h
        k_mask = mask & (k_atom >= 0) & (k_atom < valid_atoms)
    else:
        k_atom = local_block * n_queries + h - pad_left
        k_mask = mask & (k_atom >= 0) & (k_atom < atoms)
    k_index = tl.load(atom_to_token_ptr + atom_start + k_atom, mask=k_mask, other=0)

    q_token = _bounded_token(q_index, tokens)
    k_token = _bounded_token(k_index, tokens)
    pair_start = tl.load(metadata_ptr + sample).to(tl.int64)
    token_pair_offsets = (pair_start + q_token * tokens + k_token) * channels + c
    values = tl.load(token_pair_ptr + token_pair_offsets, mask=mask)
    out_offsets = output_block * block_width + within
    tl.store(out_ptr + out_offsets, values, mask=mask)


@triton.jit
def _nested_token_pair_to_atom_window_gather_backward(
    grad_out_ptr,
    atom_to_token_ptr,
    token_offsets_ptr,
    atom_offsets_ptr,
    metadata_ptr,
    block_to_sample_ptr,
    grad_token_pair_ptr,
    num_samples,
    channels: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    pad_left: tl.constexpr,
    has_num_valid_atoms: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:

    output_block = tl.program_id(0).to(tl.int64)
    within = tl.program_id(1).to(tl.int64) * block_elements + tl.arange(0, block_elements)
    block_width = channels * n_queries * n_keys
    mask = within < block_width

    c = within % channels
    h = (within // channels) % n_keys
    q = within // (channels * n_keys)
    metadata_stride = num_samples + 1
    sample = tl.load(block_to_sample_ptr + output_block).to(tl.int64)
    local_block = output_block - tl.load(metadata_ptr + metadata_stride + sample).to(tl.int64)

    token_start = tl.load(token_offsets_ptr + sample).to(tl.int64)
    token_end = tl.load(token_offsets_ptr + sample + 1).to(tl.int64)
    tokens = token_end - token_start
    atom_start = tl.load(atom_offsets_ptr + sample).to(tl.int64)
    atom_end = tl.load(atom_offsets_ptr + sample + 1).to(tl.int64)
    atoms = atom_end - atom_start

    q_atom = local_block * n_queries + q
    q_index = tl.load(
        atom_to_token_ptr + atom_start + q_atom,
        mask=mask & (q_atom < atoms),
        other=0,
    )
    if has_num_valid_atoms:
        valid_atoms = tl.load(metadata_ptr + 2 * metadata_stride + sample).to(tl.int64)
        centered_start = local_block * n_queries - pad_left
        key_start = tl.minimum(tl.maximum(centered_start, 0), tl.maximum(valid_atoms - n_keys, 0))
        k_atom = key_start + h
        k_mask = mask & (k_atom >= 0) & (k_atom < valid_atoms)
    else:
        k_atom = local_block * n_queries + h - pad_left
        k_mask = mask & (k_atom >= 0) & (k_atom < atoms)
    k_index = tl.load(atom_to_token_ptr + atom_start + k_atom, mask=k_mask, other=0)

    q_token = _bounded_token(q_index, tokens)
    k_token = _bounded_token(k_index, tokens)
    pair_start = tl.load(metadata_ptr + sample).to(tl.int64)
    token_pair_offsets = (pair_start + q_token * tokens + k_token) * channels + c
    out_offsets = output_block * block_width + within
    values = tl.load(grad_out_ptr + out_offsets, mask=mask, other=0.0)
    tl.atomic_add(grad_token_pair_ptr + token_pair_offsets, values, sem="relaxed", mask=mask)


@triton.jit
def _token_pair_to_atom_window_gather_regular_backward(
    grad_out_ptr,
    grad_token_pair_ptr,
    atoms,
    tokens,
    channels: tl.constexpr,
    n_blocks,
    atoms_per_token: tl.constexpr,
    max_key_slots: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    pad_left: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    key_slot_pid = tl.program_id(0)
    if wide_offsets:
        key_slot_pid = key_slot_pid.to(tl.int64)
        offset_tokens = tokens.to(tl.int64)
    else:
        offset_tokens = tokens
    channel_pid = tl.program_id(1)

    key_slot = key_slot_pid % max_key_slots
    q_token = (key_slot_pid // max_key_slots) % offset_tokens
    batch = key_slot_pid // (max_key_slots * offset_tokens)

    channels_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channels_offsets < channels

    q_start = q_token * atoms_per_token
    q_end = q_start + atoms_per_token
    min_block = q_start // n_queries
    max_block = (q_end - 1) // n_queries

    valid_key_min = min_block * n_queries - pad_left
    valid_key_max = max_block * n_queries - pad_left + n_keys - 1
    valid_key_min = tl.maximum(valid_key_min, 0)
    valid_key_max = tl.minimum(valid_key_max, atoms - 1)
    valid_token_min = valid_key_min // atoms_per_token
    valid_token_max = valid_key_max // atoms_per_token

    key_token = tl.maximum(valid_token_min, 1) + key_slot
    active = (key_token <= valid_token_max) & (key_token < tokens)

    acc = tl.zeros((block_channels,), dtype=tl.float32)
    h_rel = tl.arange(0, block_atoms)

    for q_rel in range(atoms_per_token):
        q_atom = q_start + q_rel
        block = q_atom // n_queries
        q = q_atom - block * n_queries
        key_start = key_token * atoms_per_token
        h_start = key_start - block * n_queries + pad_left
        h = h_start + h_rel
        k_atom = block * n_queries + h - pad_left
        h_mask = (h >= 0) & (h < n_keys) & (k_atom >= key_start) & (k_atom < key_start + atoms_per_token)
        offsets = (((batch * n_blocks + block) * n_queries + q) * n_keys + h[:, None]) * channels
        offsets = offsets + channels_offsets[None, :]
        values = tl.load(
            grad_out_ptr + offsets,
            mask=active & h_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(values.to(tl.float32), axis=0)

    out_offsets = ((batch * tokens + q_token) * tokens + key_token) * channels + channels_offsets
    tl.store(grad_token_pair_ptr + out_offsets, acc, mask=active & channel_mask)


@triton.jit
def _token_pair_to_atom_window_gather_regular_zero_backward(
    grad_out_ptr,
    grad_token_pair_ptr,
    atoms,
    tokens,
    channels: tl.constexpr,
    n_blocks,
    atoms_per_token: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    pad_left: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    q_token_pid = tl.program_id(0)
    if wide_offsets:
        q_token_pid = q_token_pid.to(tl.int64)
        offset_tokens = tokens.to(tl.int64)
    else:
        offset_tokens = tokens
    channel_pid = tl.program_id(1)

    q_token = q_token_pid % offset_tokens
    batch = q_token_pid // offset_tokens

    channels_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channels_offsets < channels
    h_rel = tl.arange(0, block_atoms)
    acc = tl.zeros((block_channels,), dtype=tl.float32)

    q_start = q_token * atoms_per_token
    for q_rel in range(atoms_per_token):
        q_atom = q_start + q_rel
        block = q_atom // n_queries
        q = q_atom - block * n_queries
        for h_base in range(0, n_keys, block_atoms):
            h = h_base + h_rel
            k_atom = block * n_queries + h - pad_left
            h_mask = (h < n_keys) & ((k_atom < atoms_per_token) | (k_atom >= atoms))
            offsets = (((batch * n_blocks + block) * n_queries + q) * n_keys + h[:, None]) * channels
            offsets = offsets + channels_offsets[None, :]
            values = tl.load(
                grad_out_ptr + offsets,
                mask=h_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(values.to(tl.float32), axis=0)

    out_offsets = ((batch * tokens + q_token) * tokens) * channels + channels_offsets
    tl.store(grad_token_pair_ptr + out_offsets, acc, mask=channel_mask)
