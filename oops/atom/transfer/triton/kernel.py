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
def _atom_to_token_reduce_regular_forward(
    atom_ptr,
    atom_mask_ptr,
    out_ptr,
    atoms,
    tokens,
    channels,
    atoms_per_token: tl.constexpr,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    token_pid = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)

    token = token_pid % tokens
    batch = token_pid // tokens
    atom_offsets = token * atoms_per_token + tl.arange(0, block_atoms)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    atom_mask = atom_offsets < (token + 1) * atoms_per_token
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else out_ptr.dtype.element_ty

    values = tl.load(
        atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
        mask=atom_mask[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(accum_dtype)
    if has_atom_mask:
        mask_batch = batch // mask_batch_repeat
        weights = tl.load(
            atom_mask_ptr + mask_batch * atoms + atom_offsets,
            mask=atom_mask,
            other=0.0,
        ).to(accum_dtype)
        values = (values * weights[:, None]).to(accum_dtype)
        count = tl.sum(weights, axis=0, dtype=accum_dtype)
    else:
        count = tl.full((), atoms_per_token, accum_dtype)
    numerator = tl.sum(values, axis=0, dtype=accum_dtype)
    if normalize:
        denominator = tl.where(
            count > 0,
            count + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        out = (numerator / denominator).to(accum_dtype)
    else:
        out = numerator
    tl.store(
        out_ptr + (batch * tokens + token) * channels + channel_offsets,
        out,
        mask=channel_mask,
    )


@triton.jit
def _atom_to_token_reduce_regular_backward(
    grad_out_ptr,
    atom_mask_ptr,
    grad_atom_ptr,
    atoms,
    tokens,
    channels,
    atoms_per_token: tl.constexpr,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    token_pid = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)

    token = token_pid % tokens
    batch = token_pid // tokens
    atom_offsets = token * atoms_per_token + tl.arange(0, block_atoms)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    atom_mask = atom_offsets < (token + 1) * atoms_per_token
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else grad_atom_ptr.dtype.element_ty

    grad = tl.load(
        grad_out_ptr + (batch * tokens + token) * channels + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(accum_dtype)
    if has_atom_mask:
        mask_batch = batch // mask_batch_repeat
        weights = tl.load(
            atom_mask_ptr + mask_batch * atoms + atom_offsets,
            mask=atom_mask,
            other=0.0,
        ).to(accum_dtype)
        count = tl.sum(weights, axis=0, dtype=accum_dtype)
    else:
        count = tl.full((), atoms_per_token, accum_dtype)
    if has_atom_mask:
        if normalize:
            denominator = tl.where(
                count > 0,
                count + tl.full((), denominator_offset, accum_dtype),
                tl.full((), 1.0, accum_dtype),
            )
            values = (grad[None, :] * (weights[:, None] / denominator).to(accum_dtype)).to(accum_dtype)
        else:
            values = (grad[None, :] * weights[:, None]).to(accum_dtype)
    else:
        if normalize:
            denominator = tl.where(
                count > 0,
                count + tl.full((), denominator_offset, accum_dtype),
                tl.full((), 1.0, accum_dtype),
            )
            values = (grad[None, :] / denominator).to(accum_dtype)
        else:
            values = grad[None, :]

    tl.store(
        grad_atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
        values,
        mask=atom_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _atom_to_token_reduce_regular_vjp(
    atom_ptr,
    atom_mask_ptr,
    grad_out_ptr,
    grad_atom_ptr,
    grad_mask_ptr,
    atoms,
    tokens,
    channels,
    atoms_per_token: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    token_pid = tl.program_id(0).to(tl.int64)
    token = token_pid % tokens
    batch = token_pid // tokens
    atom_offsets = token * atoms_per_token + tl.arange(0, block_atoms)
    channel_offsets = tl.arange(0, block_channels)
    active_atoms = atom_offsets < (token + 1) * atoms_per_token
    active_channels = channel_offsets < channels
    weights = tl.load(
        atom_mask_ptr + (batch // mask_batch_repeat) * atoms + atom_offsets,
        mask=active_atoms,
        other=0.0,
    ).to(tl.float32)
    values = tl.load(
        atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
        mask=active_atoms[:, None] & active_channels[None, :],
        other=0.0,
    ).to(tl.float32)
    grad = tl.load(
        grad_out_ptr + (batch * tokens + token) * channels + channel_offsets,
        mask=active_channels,
        other=0.0,
    ).to(tl.float32)
    if normalize:
        count = tl.sum(weights, axis=0, dtype=tl.float32)
        active = count > 0
        denominator = tl.where(active, count + denominator_offset, 1.0)
        numerator = tl.sum(values * weights[:, None], axis=0, dtype=tl.float32)
        grad_numerator = grad / denominator
        denominator_sq = denominator * denominator
        correction = -grad * numerator
        correction = correction / denominator_sq
        denominator_correction = tl.sum(correction, axis=0)
        grad_atom = grad[None, :] * (weights[:, None] / denominator)
    else:
        grad_numerator = grad
        grad_atom = grad[None, :] * weights[:, None]
    atom_dot = tl.sum(values * grad_numerator[None, :], axis=1)
    if normalize:
        grad_mask = atom_dot + tl.where(active, denominator_correction, 0.0)
    else:
        grad_mask = atom_dot
    tl.store(
        grad_atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
        grad_atom,
        mask=active_atoms[:, None] & active_channels[None, :],
    )
    tl.store(grad_mask_ptr + batch * atoms + atom_offsets, grad_mask, mask=active_atoms)


@triton.jit
def _atom_to_token_sorted_lower_bound(
    atom_to_token_ptr,
    atoms,
    token,
    max_iters: tl.constexpr,
):
    lo = tl.full((), 0, tl.int64)
    hi = tl.cast(atoms, tl.int64)
    for _ in range(max_iters):
        mid = (lo + hi) // 2
        value = tl.load(atom_to_token_ptr + mid, mask=mid < atoms, other=2147483647).to(tl.int64)
        move_right = value < token
        lo = tl.where(move_right, mid + 1, lo)
        hi = tl.where(move_right, hi, mid)
    return lo


@triton.jit
def _atom_to_token_reduce_mask_state(
    atom_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    numerator_ptr,
    raw_denominator_ptr,
    atoms,
    tokens,
    channels,
    plan_kind: tl.constexpr,
    width: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    max_iters: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
    permutation_ptr=None,
    has_permutation: tl.constexpr = False,
) -> None:

    token_pid = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)
    token = token_pid % tokens
    batch = token_pid // tokens
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    if plan_kind == 0:
        atom_offsets = token * width + lanes
        segment_mask = lanes < width
    elif plan_kind == 1:
        atom_offsets = token + lanes * tokens
        segment_mask = lanes < width
    else:
        start = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token, max_iters)
        end = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token + 1, max_iters)
        atom_offsets = start + lanes
        segment_mask = atom_offsets < end

    if has_permutation:
        atom_offsets = tl.load(permutation_ptr + atom_offsets, mask=segment_mask, other=0).to(tl.int64)

    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    accum_dtype = numerator_ptr.dtype.element_ty
    mask_batch = batch // mask_batch_repeat
    weights = tl.load(
        atom_mask_ptr + mask_batch * atoms + atom_offsets,
        mask=segment_mask,
        other=0.0,
    ).to(accum_dtype)
    values = tl.load(
        atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
        mask=segment_mask[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(accum_dtype)
    weighted = (values * weights[:, None]).to(accum_dtype)
    numerator = tl.sum(weighted, axis=0, dtype=accum_dtype)
    raw_denominator = tl.sum(weights, axis=0, dtype=accum_dtype)
    tl.store(
        numerator_ptr + (token_pid * channels + channel_offsets),
        numerator,
        mask=channel_mask,
    )
    tl.store(raw_denominator_ptr + token_pid, raw_denominator, mask=channel_pid == 0)


@triton.jit
def _atom_to_token_reduce_mask_backward(
    atom_ptr,
    atom_to_token_ptr,
    numerator_ptr,
    raw_denominator_ptr,
    grad_out_ptr,
    grad_mask_ptr,
    atoms,
    tokens,
    channels,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    plan_kind: tl.constexpr,
    width: tl.constexpr,
    block_channels: tl.constexpr,
    permutation_ptr=None,
    has_permutation: tl.constexpr = False,
) -> None:

    pid = tl.program_id(0).to(tl.int64)
    atom_index = pid % atoms
    batch = pid // atoms
    if plan_kind == 0:
        token = atom_index // width
    elif plan_kind == 1:
        token = atom_index % tokens
    else:
        token = tl.load(atom_to_token_ptr + atom_index).to(tl.int64)

    if has_permutation:
        atom_index = tl.load(permutation_ptr + atom_index).to(tl.int64)
    accum_dtype = numerator_ptr.dtype.element_ty
    if normalize:
        raw_denominator = tl.load(raw_denominator_ptr + batch * tokens + token).to(accum_dtype)
        active = raw_denominator > 0
        denominator = tl.where(
            active,
            raw_denominator + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        denominator_sq = (denominator * denominator).to(accum_dtype)
    atom_dot = tl.zeros((), dtype=tl.float32)
    if normalize:
        denominator_correction = tl.zeros((), dtype=tl.float32)

    for channel_start in range(0, channels, block_channels):
        channel_offsets = channel_start + tl.arange(0, block_channels)
        channel_mask = channel_offsets < channels
        grad = tl.load(
            grad_out_ptr + (batch * tokens + token) * channels + channel_offsets,
            mask=channel_mask,
            other=0.0,
        ).to(accum_dtype)
        atom = tl.load(
            atom_ptr + (batch * atoms + atom_index) * channels + channel_offsets,
            mask=channel_mask,
            other=0.0,
        ).to(accum_dtype)
        if normalize:
            numerator = tl.load(
                numerator_ptr + (batch * tokens + token) * channels + channel_offsets,
                mask=channel_mask,
                other=0.0,
            ).to(accum_dtype)
            grad_numerator = (grad / denominator).to(accum_dtype)
            correction = (-grad * numerator).to(accum_dtype)
            correction = (correction / denominator_sq).to(accum_dtype)
            denominator_correction += tl.sum(correction.to(tl.float32), axis=0)
        else:
            grad_numerator = grad
        atom_dot += tl.sum(atom.to(tl.float32) * grad_numerator.to(tl.float32), axis=0)

    atom_dot_value = atom_dot.to(accum_dtype)
    if normalize:
        correction_value = denominator_correction.to(accum_dtype)
        grad_mask = (atom_dot_value + tl.where(active, correction_value, 0.0)).to(accum_dtype)
    else:
        grad_mask = atom_dot_value
    tl.store(grad_mask_ptr + batch * atoms + atom_index, grad_mask)


@triton.jit
def _atom_to_token_reduce_sorted_forward(
    atom_ptr,
    atom_to_token_ptr,
    atom_mask_ptr,
    out_ptr,
    atoms,
    tokens,
    channels,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    max_iters: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
    permutation_ptr=None,
    has_permutation: tl.constexpr = False,
    token_offsets_ptr=None,
    token_offsets_stride: tl.constexpr = 1,
    has_token_offsets: tl.constexpr = False,
) -> None:
    token_pid = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)

    token = token_pid % tokens
    batch = token_pid // tokens
    if has_token_offsets:
        start = tl.load(token_offsets_ptr + token * token_offsets_stride).to(tl.int64)
        end = tl.load(token_offsets_ptr + (token + 1) * token_offsets_stride).to(tl.int64)
    else:
        start = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token, max_iters)
        end = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token + 1, max_iters)
    atom_offsets = start + tl.arange(0, block_atoms)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    atom_mask = atom_offsets < end
    if has_permutation:
        source_atoms = tl.load(permutation_ptr + atom_offsets, mask=atom_mask, other=0).to(tl.int64)
    else:
        source_atoms = atom_offsets
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else out_ptr.dtype.element_ty

    values = tl.load(
        atom_ptr + (batch * atoms + source_atoms[:, None]) * channels + channel_offsets[None, :],
        mask=atom_mask[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(accum_dtype)
    if has_atom_mask:
        mask_batch = batch // mask_batch_repeat
        weights = tl.load(
            atom_mask_ptr + mask_batch * atoms + source_atoms,
            mask=atom_mask,
            other=0.0,
        ).to(accum_dtype)
        values = (values * weights[:, None]).to(accum_dtype)
        count = tl.sum(weights, axis=0, dtype=accum_dtype)
    else:
        count = (end - start).to(accum_dtype)
    numerator = tl.sum(values, axis=0, dtype=accum_dtype)
    if normalize:
        denominator = tl.where(
            count > 0,
            count + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        out = (numerator / denominator).to(accum_dtype)
    else:
        out = numerator
    tl.store(
        out_ptr + (batch * tokens + token) * channels + channel_offsets,
        out,
        mask=channel_mask,
    )


@triton.jit
def _atom_to_token_reduce_sorted_backward(
    grad_out_ptr,
    atom_to_token_ptr,
    atom_mask_ptr,
    grad_atom_ptr,
    atoms,
    tokens,
    channels,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    max_iters: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
    permutation_ptr=None,
    has_permutation: tl.constexpr = False,
    token_offsets_ptr=None,
    token_offsets_stride: tl.constexpr = 1,
    has_token_offsets: tl.constexpr = False,
) -> None:
    token_pid = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)

    token = token_pid % tokens
    batch = token_pid // tokens
    if has_token_offsets:
        start = tl.load(token_offsets_ptr + token * token_offsets_stride).to(tl.int64)
        end = tl.load(token_offsets_ptr + (token + 1) * token_offsets_stride).to(tl.int64)
    else:
        start = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token, max_iters)
        end = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token + 1, max_iters)
    atom_offsets = start + tl.arange(0, block_atoms)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    atom_mask = atom_offsets < end
    if has_permutation:
        source_atoms = tl.load(permutation_ptr + atom_offsets, mask=atom_mask, other=0).to(tl.int64)
    else:
        source_atoms = atom_offsets
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else grad_atom_ptr.dtype.element_ty

    grad = tl.load(
        grad_out_ptr + (batch * tokens + token) * channels + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(accum_dtype)
    if has_atom_mask:
        mask_batch = batch // mask_batch_repeat
        weights = tl.load(
            atom_mask_ptr + mask_batch * atoms + source_atoms,
            mask=atom_mask,
            other=0.0,
        ).to(accum_dtype)
        count = tl.sum(weights, axis=0, dtype=accum_dtype)
    else:
        count = (end - start).to(accum_dtype)
    if has_atom_mask:
        if normalize:
            denominator = tl.where(
                count > 0,
                count + tl.full((), denominator_offset, accum_dtype),
                tl.full((), 1.0, accum_dtype),
            )
            values = (grad[None, :] * (weights[:, None] / denominator).to(accum_dtype)).to(accum_dtype)
        else:
            values = (grad[None, :] * weights[:, None]).to(accum_dtype)
    else:
        if normalize:
            denominator = tl.where(
                count > 0,
                count + tl.full((), denominator_offset, accum_dtype),
                tl.full((), 1.0, accum_dtype),
            )
            values = (grad[None, :] / denominator).to(accum_dtype)
        else:
            values = grad[None, :]

    tl.store(
        grad_atom_ptr + (batch * atoms + source_atoms[:, None]) * channels + channel_offsets[None, :],
        values,
        mask=atom_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _atom_to_token_reduce_interleaved_forward(
    atom_ptr,
    atom_mask_ptr,
    out_ptr,
    atoms,
    tokens,
    channels,
    atoms_per_token: tl.constexpr,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    token_pid = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)

    token = token_pid % tokens
    batch = token_pid // tokens
    atom_offsets = token + tl.arange(0, block_atoms).to(tl.int64) * tokens
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    atom_mask = tl.arange(0, block_atoms) < atoms_per_token
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else out_ptr.dtype.element_ty

    values = tl.load(
        atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
        mask=atom_mask[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(accum_dtype)
    if has_atom_mask:
        mask_batch = batch // mask_batch_repeat
        weights = tl.load(
            atom_mask_ptr + mask_batch * atoms + atom_offsets,
            mask=atom_mask,
            other=0.0,
        ).to(accum_dtype)
        values = (values * weights[:, None]).to(accum_dtype)
        count = tl.sum(weights, axis=0, dtype=accum_dtype)
    else:
        count = tl.full((), atoms_per_token, accum_dtype)
    numerator = tl.sum(values, axis=0, dtype=accum_dtype)
    if normalize:
        denominator = tl.where(
            count > 0,
            count + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        out = (numerator / denominator).to(accum_dtype)
    else:
        out = numerator
    tl.store(
        out_ptr + (batch * tokens + token) * channels + channel_offsets,
        out,
        mask=channel_mask,
    )


@triton.jit
def _atom_to_token_reduce_interleaved_backward(
    grad_out_ptr,
    atom_mask_ptr,
    grad_atom_ptr,
    atoms,
    tokens,
    channels,
    atoms_per_token: tl.constexpr,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    token_pid = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)

    token = token_pid % tokens
    batch = token_pid // tokens
    atom_offsets = token + tl.arange(0, block_atoms).to(tl.int64) * tokens
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    atom_mask = tl.arange(0, block_atoms) < atoms_per_token
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else grad_atom_ptr.dtype.element_ty

    grad = tl.load(
        grad_out_ptr + (batch * tokens + token) * channels + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(accum_dtype)
    if has_atom_mask:
        mask_batch = batch // mask_batch_repeat
        weights = tl.load(
            atom_mask_ptr + mask_batch * atoms + atom_offsets,
            mask=atom_mask,
            other=0.0,
        ).to(accum_dtype)
        count = tl.sum(weights, axis=0, dtype=accum_dtype)
    else:
        count = tl.full((), atoms_per_token, accum_dtype)
    if has_atom_mask:
        if normalize:
            denominator = tl.where(
                count > 0,
                count + tl.full((), denominator_offset, accum_dtype),
                tl.full((), 1.0, accum_dtype),
            )
            values = (grad[None, :] * (weights[:, None] / denominator).to(accum_dtype)).to(accum_dtype)
        else:
            values = (grad[None, :] * weights[:, None]).to(accum_dtype)
    else:
        if normalize:
            denominator = tl.where(
                count > 0,
                count + tl.full((), denominator_offset, accum_dtype),
                tl.full((), 1.0, accum_dtype),
            )
            values = (grad[None, :] / denominator).to(accum_dtype)
        else:
            values = grad[None, :]

    tl.store(
        grad_atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
        values,
        mask=atom_mask[:, None] & channel_mask[None, :],
    )


_BATCHED_KIND_REGULAR = tl.constexpr(1)
_BATCHED_KIND_INTERLEAVED = tl.constexpr(2)
_BATCHED_KIND_SORTED = tl.constexpr(3)
_BATCHED_KIND_GENERAL = tl.constexpr(4)

_BATCHED_BAD_INDEX = tl.constexpr(1)
_BATCHED_BAD_REGULAR = tl.constexpr(2)
_BATCHED_BAD_INTERLEAVED = tl.constexpr(4)
_BATCHED_BAD_SORTED = tl.constexpr(8)


@triton.jit
def _batched_atom_to_token_classify(
    atom_to_token_ptr,
    mapping_flags_ptr,
    sort_keys_ptr,
    total_mapping_atoms,
    atoms,
    tokens,
    block_atoms: tl.constexpr,
) -> None:
    linear = tl.program_id(0).to(tl.int64) * block_atoms + tl.arange(0, block_atoms).to(tl.int64)
    active = linear < total_mapping_atoms
    atom = linear % atoms
    mapping_batch = linear // atoms
    value = tl.load(atom_to_token_ptr + linear, mask=active, other=0).to(tl.int64)
    tl.store(sort_keys_ptr + linear, tl.maximum(tl.minimum(value, tokens - 1), 0), mask=active)
    previous = tl.load(
        atom_to_token_ptr + linear - 1,
        mask=active & (atom > 0),
        other=value,
    ).to(tl.int64)
    width = atoms // tokens
    uniform = (width > 0) & (atoms == tokens * width)
    flags = tl.where((value < 0) | (value >= tokens), _BATCHED_BAD_INDEX, 0)
    flags |= tl.where(~uniform | (value != atom // tl.maximum(width, 1)), _BATCHED_BAD_REGULAR, 0)
    flags |= tl.where(~uniform | (value != atom % tokens), _BATCHED_BAD_INTERLEAVED, 0)
    flags |= tl.where((atom > 0) & (value < previous), _BATCHED_BAD_SORTED, 0)
    if atoms >= block_atoms:
        first_row = tl.program_id(0).to(tl.int64) * block_atoms // atoms
        first_flags = tl.reduce(tl.where(active & (mapping_batch == first_row), flags, 0), 0, _mapping_flag_or)
        second_flags = tl.reduce(tl.where(active & (mapping_batch != first_row), flags, 0), 0, _mapping_flag_or)
        tl.atomic_or(mapping_flags_ptr + first_row, first_flags)
        tl.atomic_or(mapping_flags_ptr + first_row + 1, second_flags, mask=second_flags != 0)
    else:
        tl.atomic_or(mapping_flags_ptr + mapping_batch, flags, mask=active)


@triton.jit
def _batched_atom_to_token_kind(flags):
    index_valid = (flags & _BATCHED_BAD_INDEX) == 0
    regular = index_valid & ((flags & _BATCHED_BAD_REGULAR) == 0)
    interleaved = index_valid & ((flags & _BATCHED_BAD_INTERLEAVED) == 0)
    sorted_row = index_valid & ((flags & _BATCHED_BAD_SORTED) == 0)
    kind = tl.where(
        regular,
        _BATCHED_KIND_REGULAR,
        tl.where(
            interleaved,
            _BATCHED_KIND_INTERLEAVED,
            tl.where(sorted_row, _BATCHED_KIND_SORTED, _BATCHED_KIND_GENERAL),
        ),
    )
    return kind


@triton.jit
def _mapping_flag_or(left, right):
    return left | right


@triton.jit
def _stable_atom_token_sort(
    atom_to_token_ptr,
    sorted_mapping_ptr,
    permutation_ptr,
    mapping_start,
    atoms,
    tokens,
    permutation_base,
    block_atoms: tl.constexpr,
):
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    active = lanes < atoms
    token = tl.load(atom_to_token_ptr + mapping_start + lanes, mask=active, other=0).to(tl.int64)
    token = tl.maximum(tl.minimum(token, tl.maximum(tokens - 1, 0)), 0)
    # Low 32 bits hold the original atom index and fix token tie order.
    keys = tl.where(active, (token << 32) | lanes, 9223372036854775807)
    ordered = tl.sort(keys, descending=False)
    tl.store(sorted_mapping_ptr + mapping_start + lanes, ordered >> 32, mask=active)
    tl.store(permutation_ptr + mapping_start + lanes, permutation_base + (ordered & 4294967295), mask=active)


@triton.jit
def _batched_atom_to_token_classify_and_invert(
    atom_to_token_ptr,
    mapping_flags_ptr,
    sorted_mapping_ptr,
    permutation_ptr,
    atoms,
    tokens,
    block_atoms: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    active = lanes < atoms
    start = row * atoms
    value = tl.load(atom_to_token_ptr + start + lanes, mask=active, other=0).to(tl.int64)
    previous = tl.load(atom_to_token_ptr + start + lanes - 1, mask=active & (lanes > 0), other=value).to(tl.int64)
    width = atoms // tokens
    uniform = (width > 0) & (atoms == tokens * width)
    flags = tl.where((value < 0) | (value >= tokens), _BATCHED_BAD_INDEX, 0)
    flags |= tl.where(~uniform | (value != lanes // tl.maximum(width, 1)), _BATCHED_BAD_REGULAR, 0)
    flags |= tl.where(~uniform | (value != lanes % tokens), _BATCHED_BAD_INTERLEAVED, 0)
    flags |= tl.where((lanes > 0) & (value < previous), _BATCHED_BAD_SORTED, 0)
    row_flags = tl.reduce(tl.where(active, flags, 0), 0, _mapping_flag_or)
    tl.store(mapping_flags_ptr + row, row_flags)
    if _batched_atom_to_token_kind(row_flags) == _BATCHED_KIND_GENERAL:
        _stable_atom_token_sort(
            atom_to_token_ptr, sorted_mapping_ptr, permutation_ptr, start, atoms, tokens, 0, block_atoms
        )


@triton.jit
def _batched_atom_token_tile_classify_and_count(
    atom_to_token_ptr,
    tile_flags_ptr,
    counts_ptr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    tiles: tl.constexpr,
    block_atoms: tl.constexpr,
    block_tokens: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    lanes = tl.arange(0, block_atoms)
    atom = tile * block_atoms + lanes
    active = atom < atoms
    value = tl.load(atom_to_token_ptr + row * atoms + atom, mask=active, other=0).to(tl.int64)
    previous = tl.load(atom_to_token_ptr + row * atoms + atom - 1, mask=active & (atom > 0), other=value).to(tl.int64)
    width = atoms // tokens
    uniform = (width > 0) & (atoms == tokens * width)
    flags = tl.where((value < 0) | (value >= tokens), _BATCHED_BAD_INDEX, 0)
    flags |= tl.where(~uniform | (value != atom // tl.maximum(width, 1)), _BATCHED_BAD_REGULAR, 0)
    flags |= tl.where(~uniform | (value != atom % tokens), _BATCHED_BAD_INTERLEAVED, 0)
    flags |= tl.where((atom > 0) & (value < previous), _BATCHED_BAD_SORTED, 0)
    tile_flags = tl.reduce(tl.where(active, flags, 0), 0, _mapping_flag_or)
    tl.store(tile_flags_ptr + row * tiles + tile, tile_flags)
    token = tl.maximum(tl.minimum(value, tokens - 1), 0).to(tl.int32)
    counts = tl.histogram(token, block_tokens, mask=active)
    tl.store(counts_ptr + (row * tiles + tile) * block_tokens + tl.arange(0, block_tokens), counts)


@triton.jit
def _batched_atom_token_tile_prefix(
    tile_flags_ptr,
    flags_ptr,
    counts_ptr,
    totals_ptr,
    tiles: tl.constexpr,
    block_tokens: tl.constexpr,
    block_tiles: tl.constexpr,
    token_tile: tl.constexpr,
):
    row = tl.program_id(1)
    tile_lanes = tl.arange(0, block_tiles)
    tile_flags = tl.load(tile_flags_ptr + row * tiles + tile_lanes, mask=tile_lanes < tiles, other=0)
    flags = tl.reduce(tile_flags, 0, _mapping_flag_or)
    if tl.program_id(0) == 0:
        tl.store(flags_ptr + row, flags)
    if _batched_atom_to_token_kind(flags) == _BATCHED_KIND_GENERAL:
        tile = tl.arange(0, block_tiles)
        token = tl.program_id(0) * token_tile + tl.arange(0, token_tile)
        offsets = (row * tiles + tile[:, None]) * block_tokens + token[None, :]
        count = tl.load(counts_ptr + offsets, mask=(tile[:, None] < tiles) & (token[None, :] < block_tokens), other=0)
        exclusive = tl.cumsum(count, 0) - count
        tl.store(counts_ptr + offsets, exclusive, mask=(tile[:, None] < tiles) & (token[None, :] < block_tokens))
        total = tl.sum(count, 0)
        tl.store(totals_ptr + row * block_tokens + token, total, mask=token < block_tokens)


@triton.jit
def _batched_atom_token_token_prefix(flags_ptr, totals_ptr, block_tokens: tl.constexpr):
    row = tl.program_id(0)
    flags = tl.load(flags_ptr + row)
    if _batched_atom_to_token_kind(flags) == _BATCHED_KIND_GENERAL:
        token = tl.arange(0, block_tokens)
        total = tl.load(totals_ptr + row * block_tokens + token)
        tl.store(totals_ptr + row * block_tokens + token, tl.cumsum(total, 0) - total)


@triton.jit
def _batched_atom_token_tile_scatter(
    atom_to_token_ptr,
    flags_ptr,
    counts_ptr,
    totals_ptr,
    sorted_mapping_ptr,
    permutation_ptr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    tiles: tl.constexpr,
    block_atoms: tl.constexpr,
    block_tokens: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    flags = tl.load(flags_ptr + row)
    if _batched_atom_to_token_kind(flags) == _BATCHED_KIND_GENERAL:
        lanes = tl.arange(0, block_atoms)
        atoms_local = tile * block_atoms + lanes
        active = atoms_local < atoms
        token = tl.load(atom_to_token_ptr + row * atoms + atoms_local, mask=active, other=0).to(tl.int32)
        token = tl.maximum(tl.minimum(token, tokens - 1), 0)
        keys = tl.where(active, token * block_atoms + lanes, 2147483647)
        ordered = tl.sort(keys, descending=False)
        sorted_token = ordered // block_atoms
        original_lane = ordered % block_atoms
        previous = tl.gather(sorted_token, tl.maximum(lanes - 1, 0), 0)
        beginning = tl.where((lanes == 0) | (sorted_token != previous), lanes, 0)
        local_rank = lanes - tl.associative_scan(beginning, 0, _maximum)
        active_sorted = tile * block_atoms + lanes < atoms
        offset = tl.load(counts_ptr + (row * tiles + tile) * block_tokens + sorted_token, mask=active_sorted, other=0)
        token_offset = tl.load(totals_ptr + row * block_tokens + sorted_token, mask=active_sorted, other=0)
        position = offset + token_offset + local_rank
        tl.store(sorted_mapping_ptr + row * atoms + position, sorted_token, mask=active_sorted)
        tl.store(permutation_ptr + row * atoms + position, tile * block_atoms + original_lane, mask=active_sorted)


@triton.jit
def _maximum(left, right):
    return tl.maximum(left, right)


@triton.jit
def _batched_atom_to_token_lower_bound(
    atom_to_token_ptr,
    mapping_start,
    atoms,
    token,
):
    lo = tl.full((), 0, tl.int64)
    hi = tl.full((), 0, tl.int64) + atoms
    while lo < hi:
        active = lo < hi
        mid = (lo + hi) // 2
        value = tl.load(
            atom_to_token_ptr + mapping_start + mid,
            mask=active & (mid < atoms),
            other=2147483647,
        ).to(tl.int64)
        move_right = active & (value < token)
        hi = tl.where(active & ~move_right, mid, hi)
        lo = tl.where(move_right, mid + 1, lo)
    return lo


@triton.jit
def _batched_atom_to_token_runtime_descriptor(
    atom_to_token_ptr,
    mapping_flags_ptr,
    sorted_mapping_ptr,
    batch,
    token,
    atoms,
    tokens,
    mapping_batch_repeat: tl.constexpr,
):
    mapping_batch = batch // mapping_batch_repeat
    mapping_start = mapping_batch * atoms
    flags = tl.load(mapping_flags_ptr + mapping_batch).to(tl.int32)
    kind = _batched_atom_to_token_kind(flags)
    width = atoms // tokens
    sorted_start = tl.full((), 0, tl.int64)
    sorted_end = tl.full((), 0, tl.int64) + atoms
    if kind == _BATCHED_KIND_SORTED:
        sorted_start = _batched_atom_to_token_lower_bound(atom_to_token_ptr, mapping_start, atoms, token)
        sorted_end = _batched_atom_to_token_lower_bound(atom_to_token_ptr, mapping_start, atoms, token + 1)
    elif kind == _BATCHED_KIND_GENERAL:
        sorted_start = _batched_atom_to_token_lower_bound(sorted_mapping_ptr, mapping_start, atoms, token)
        sorted_end = _batched_atom_to_token_lower_bound(sorted_mapping_ptr, mapping_start, atoms, token + 1)
    start = tl.where(kind == _BATCHED_KIND_REGULAR, token * width, sorted_start)
    start = tl.where(kind == _BATCHED_KIND_INTERLEAVED, token, start)
    scan_count = tl.where(
        (kind == _BATCHED_KIND_REGULAR) | (kind == _BATCHED_KIND_INTERLEAVED),
        width,
        sorted_end - sorted_start,
    )
    stride = tl.where(kind == _BATCHED_KIND_INTERLEAVED, tokens, 1)
    return mapping_start, kind, start, stride, scan_count


@triton.jit
def _batched_atom_selection(
    permutation_ptr,
    mapping_start,
    atom_offsets,
    lane_mask,
    kind,
):
    if kind == _BATCHED_KIND_GENERAL:
        atom_offsets = tl.load(
            permutation_ptr + mapping_start + atom_offsets,
            mask=lane_mask,
            other=0,
        ).to(tl.int64)
    return atom_offsets


@triton.jit
def _batched_atom_to_token_reduce_forward(
    atom_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    mapping_flags_ptr,
    sorted_mapping_ptr,
    permutation_ptr,
    out_ptr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    mapping_batch_repeat: tl.constexpr,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    segment = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)
    token = segment % tokens
    batch = segment // tokens
    mapping_start, kind, start, stride, scan_count = _batched_atom_to_token_runtime_descriptor(
        atom_to_token_ptr,
        mapping_flags_ptr,
        sorted_mapping_ptr,
        batch,
        token,
        atoms,
        tokens,
        mapping_batch_repeat,
    )
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32
    numerator = tl.zeros((block_channels,), dtype=accum_dtype)
    raw_denominator = tl.zeros((), dtype=accum_dtype)
    assigned_count = tl.zeros((), dtype=tl.int64)
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atom_offsets = start + relative * stride
        lane_mask = relative < scan_count
        atom_offsets = _batched_atom_selection(permutation_ptr, mapping_start, atom_offsets, lane_mask, kind)
        values = tl.load(
            atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
            mask=lane_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(accum_dtype)
        if has_atom_mask:
            mask_batch = batch // mask_batch_repeat
            weights = tl.load(
                atom_mask_ptr + mask_batch * atoms + atom_offsets,
                mask=lane_mask,
                other=0.0,
            ).to(accum_dtype)
            values = (values * weights[:, None]).to(accum_dtype)
            raw_denominator += tl.sum(weights, axis=0, dtype=accum_dtype)
        numerator += tl.sum(values, axis=0, dtype=accum_dtype)
        assigned_count += tl.sum(lane_mask.to(tl.int64), axis=0)
    raw_denominator = raw_denominator if has_atom_mask else assigned_count.to(accum_dtype)
    if normalize:
        denominator = tl.where(
            raw_denominator > 0,
            raw_denominator + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        output = (numerator / denominator).to(accum_dtype)
    else:
        output = numerator
    tl.store(
        out_ptr + (batch * tokens + token) * channels + channel_offsets,
        output,
        mask=channel_mask,
    )


@triton.jit
def _batched_atom_to_token_reduce_backward(
    grad_out_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    mapping_flags_ptr,
    sorted_mapping_ptr,
    permutation_ptr,
    grad_atom_ptr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    mapping_batch_repeat: tl.constexpr,
    has_atom_mask: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    segment = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)
    token = segment % tokens
    batch = segment // tokens
    mapping_start, kind, start, stride, scan_count = _batched_atom_to_token_runtime_descriptor(
        atom_to_token_ptr,
        mapping_flags_ptr,
        sorted_mapping_ptr,
        batch,
        token,
        atoms,
        tokens,
        mapping_batch_repeat,
    )
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    accum_dtype = tl.float32
    raw_denominator = tl.zeros((), dtype=accum_dtype)
    assigned_count = tl.zeros((), dtype=tl.int64)
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atom_offsets = start + relative * stride
        lane_mask = relative < scan_count
        atom_offsets = _batched_atom_selection(permutation_ptr, mapping_start, atom_offsets, lane_mask, kind)
        if has_atom_mask:
            mask_batch = batch // mask_batch_repeat
            weights = tl.load(
                atom_mask_ptr + mask_batch * atoms + atom_offsets,
                mask=lane_mask,
                other=0.0,
            ).to(accum_dtype)
            raw_denominator += tl.sum(weights, axis=0, dtype=accum_dtype)
        assigned_count += tl.sum(lane_mask.to(tl.int64), axis=0)
    raw_denominator = raw_denominator if has_atom_mask else assigned_count.to(accum_dtype)
    if normalize:
        denominator = tl.where(
            raw_denominator > 0,
            raw_denominator + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    grad = tl.load(
        grad_out_ptr + (batch * tokens + token) * channels + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(accum_dtype)
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atom_offsets = start + relative * stride
        lane_mask = relative < scan_count
        atom_offsets = _batched_atom_selection(permutation_ptr, mapping_start, atom_offsets, lane_mask, kind)
        if has_atom_mask:
            mask_batch = batch // mask_batch_repeat
            weights = tl.load(
                atom_mask_ptr + mask_batch * atoms + atom_offsets,
                mask=lane_mask,
                other=0.0,
            ).to(accum_dtype)
            if normalize:
                values = (grad[None, :] * (weights[:, None] / denominator).to(accum_dtype)).to(accum_dtype)
            else:
                values = (grad[None, :] * weights[:, None]).to(accum_dtype)
        else:
            if normalize:
                values = (grad[None, :] / denominator).to(accum_dtype)
            else:
                values = grad[None, :]
        tl.store(
            grad_atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
            values,
            mask=lane_mask[:, None] & channel_mask[None, :],
        )


@triton.jit
def _batched_atom_to_token_reduce_mask_state(
    atom_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    mapping_flags_ptr,
    sorted_mapping_ptr,
    permutation_ptr,
    numerator_ptr,
    raw_denominator_ptr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    mapping_batch_repeat: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    segment = tl.program_id(0).to(tl.int64)
    channel_pid = tl.program_id(1)
    token = segment % tokens
    batch = segment // tokens
    mapping_start, kind, start, stride, scan_count = _batched_atom_to_token_runtime_descriptor(
        atom_to_token_ptr,
        mapping_flags_ptr,
        sorted_mapping_ptr,
        batch,
        token,
        atoms,
        tokens,
        mapping_batch_repeat,
    )
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    accum_dtype = raw_denominator_ptr.dtype.element_ty
    numerator = tl.zeros((block_channels,), dtype=accum_dtype)
    denominator = tl.zeros((), dtype=accum_dtype)
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atom_offsets = start + relative * stride
        lane_mask = relative < scan_count
        atom_offsets = _batched_atom_selection(permutation_ptr, mapping_start, atom_offsets, lane_mask, kind)
        mask_batch = batch // mask_batch_repeat
        weights = tl.load(
            atom_mask_ptr + mask_batch * atoms + atom_offsets,
            mask=lane_mask,
            other=0.0,
        ).to(accum_dtype)
        values = tl.load(
            atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
            mask=lane_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(accum_dtype)
        numerator += tl.sum((values * weights[:, None]).to(accum_dtype), axis=0, dtype=accum_dtype)
        denominator += tl.sum(weights, axis=0, dtype=accum_dtype)
    tl.store(
        numerator_ptr + segment * channels + channel_offsets,
        numerator,
        mask=channel_mask,
    )
    tl.store(raw_denominator_ptr + segment, denominator, mask=channel_pid == 0)


@triton.jit
def _batched_atom_to_token_reduce_mask_backward(
    atom_ptr,
    atom_to_token_ptr,
    numerator_ptr,
    raw_denominator_ptr,
    grad_out_ptr,
    grad_mask_ptr,
    atoms: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    mapping_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
    total_atoms,
) -> None:
    pid = tl.program_id(0).to(tl.int64) * block_atoms + tl.arange(0, block_atoms).to(tl.int64)
    atom_mask = pid < total_atoms
    atom_index = pid % atoms
    batch = pid // atoms
    mapping_batch = batch // mapping_batch_repeat
    source_token = tl.load(atom_to_token_ptr + mapping_batch * atoms + atom_index, mask=atom_mask, other=0).to(tl.int64)
    token = tl.maximum(tl.minimum(source_token, tokens - 1), 0)
    accum_dtype = raw_denominator_ptr.dtype.element_ty
    if normalize:
        raw_denominator = tl.load(raw_denominator_ptr + batch * tokens + token, mask=atom_mask, other=0).to(accum_dtype)
        active = raw_denominator > 0
        denominator = tl.where(
            active,
            raw_denominator + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        denominator_sq = (denominator * denominator).to(accum_dtype)
    atom_dot = tl.zeros((block_atoms,), dtype=tl.float32)
    if normalize:
        denominator_correction = tl.zeros((block_atoms,), dtype=tl.float32)
    for channel_start in range(0, channels, block_channels):
        channel_offsets = channel_start + tl.arange(0, block_channels)
        channel_mask = channel_offsets < channels
        grad = tl.load(
            grad_out_ptr + (batch[:, None] * tokens + token[:, None]) * channels + channel_offsets[None, :],
            mask=atom_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(accum_dtype)
        atom = tl.load(
            atom_ptr + (batch[:, None] * atoms + atom_index[:, None]) * channels + channel_offsets[None, :],
            mask=atom_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(accum_dtype)
        if normalize:
            numerator = tl.load(
                numerator_ptr + (batch[:, None] * tokens + token[:, None]) * channels + channel_offsets[None, :],
                mask=atom_mask[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(accum_dtype)
            grad_numerator = (grad / denominator[:, None]).to(accum_dtype)
            correction = (-grad * numerator).to(accum_dtype)
            correction = (correction / denominator_sq[:, None]).to(accum_dtype)
            denominator_correction += tl.sum(correction.to(tl.float32), axis=1)
        else:
            grad_numerator = grad
        atom_dot += tl.sum(atom.to(tl.float32) * grad_numerator.to(tl.float32), axis=1)
    if normalize:
        gradient = atom_dot + tl.where(active, denominator_correction, 0.0)
    else:
        gradient = atom_dot
    tl.store(grad_mask_ptr + pid, gradient.to(grad_mask_ptr.dtype.element_ty), mask=atom_mask)


_NESTED_KIND_INVALID = tl.constexpr(0)
_NESTED_KIND_REGULAR = tl.constexpr(1)
_NESTED_KIND_INTERLEAVED = tl.constexpr(2)
_NESTED_KIND_SORTED = tl.constexpr(3)
_NESTED_KIND_GENERAL = tl.constexpr(4)
_NESTED_KIND_INDEXED = tl.constexpr(5)

_NESTED_BAD_INDEX = tl.constexpr(1)
_NESTED_BAD_REGULAR = tl.constexpr(2)
_NESTED_BAD_INTERLEAVED = tl.constexpr(4)
_NESTED_BAD_SORTED = tl.constexpr(8)


@triton.jit
def _nested_atom_to_token_lower_bound(
    atom_to_token_ptr,
    atom_start,
    atoms,
    token,
):

    lo = tl.full((), 0, tl.int64)
    hi = atoms.to(tl.int64)
    # At most INT32_MAX atoms: 32 steps cover every admitted lower bound.
    for _ in range(32):
        active = lo < hi
        mid = (lo + hi) // 2
        value = tl.load(
            atom_to_token_ptr + atom_start + mid,
            mask=active & (mid < atoms),
            other=2147483647,
        ).to(tl.int64)
        move_right = active & (value < token)
        hi = tl.where(active & ~move_right, mid, hi)
        lo = tl.where(move_right, mid + 1, lo)
    return lo


@triton.jit
def _nested_atom_to_token_count(
    atom_to_token_ptr,
    atom_batch_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    token_count_ptr,
    sample_flags_ptr,
    sort_keys_ptr,
    total_atoms,
    write_sort_keys: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:

    atom = tl.program_id(0).to(tl.int64) * block_atoms + tl.arange(0, block_atoms).to(tl.int64)
    atom_mask = atom < total_atoms
    sample = tl.load(atom_batch_ptr + atom, mask=atom_mask, other=0).to(tl.int64)
    atom_start = tl.load(atom_offsets_ptr + sample, mask=atom_mask, other=0).to(tl.int64)
    atom_end = tl.load(atom_offsets_ptr + sample + 1, mask=atom_mask, other=0).to(tl.int64)
    token_start = tl.load(token_offsets_ptr + sample, mask=atom_mask, other=0).to(tl.int64)
    token_end = tl.load(token_offsets_ptr + sample + 1, mask=atom_mask, other=0).to(tl.int64)
    tokens = token_end - token_start
    sample_atoms = atom_end - atom_start
    local_atom = atom - atom_start
    local_token = tl.load(atom_to_token_ptr + atom, mask=atom_mask, other=0).to(tl.int64)
    valid = atom_mask & (tokens > 0) & (local_token >= 0) & (local_token < tokens)
    bounded = tl.maximum(tl.minimum(local_token, tl.maximum(tokens - 1, 0)), 0)
    if write_sort_keys:
        tl.store(sort_keys_ptr + atom, token_start + bounded, mask=atom_mask)
    tl.atomic_add(token_count_ptr + token_start + bounded, 1, mask=atom_mask & (tokens > 0))
    flags = tl.where(valid, 0, _NESTED_BAD_INDEX)
    safe_tokens = tl.maximum(tokens, 1)
    width = sample_atoms // safe_tokens
    uniform = (tokens > 0) & (width > 0) & (sample_atoms == tokens * width)
    regular_ok = uniform & (local_token == local_atom // tl.maximum(width, 1))
    interleaved_ok = uniform & (local_token == local_atom % safe_tokens)
    flags |= tl.where(regular_ok, 0, _NESTED_BAD_REGULAR) | tl.where(interleaved_ok, 0, _NESTED_BAD_INTERLEAVED)
    has_previous = atom_mask & (atom > atom_start)
    previous = tl.load(atom_to_token_ptr + atom - 1, mask=has_previous, other=local_token).to(tl.int64)
    flags |= tl.where(has_previous & (local_token < previous), _NESTED_BAD_SORTED, 0)
    first_sample = tl.min(tl.where(atom_mask, sample, 2147483647), 0)
    last_sample = tl.max(tl.where(atom_mask, sample, 0), 0)
    if first_sample == last_sample:
        combined = tl.reduce(tl.where(atom_mask, flags, 0), 0, _mapping_flag_or)
        tl.atomic_or(sample_flags_ptr + first_sample, combined)
    else:
        tl.atomic_or(sample_flags_ptr + sample, flags, mask=atom_mask)


@triton.jit
def _nested_atom_to_token_finalize_and_invert(
    atom_to_token_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    sample_flags_ptr,
    sample_kind_ptr,
    sorted_mapping_ptr,
    permutation_ptr,
    num_samples,
    global_sort: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    for sample in range(tl.program_id(0).to(tl.int64), num_samples, tl.num_programs(0).to(tl.int64)):
        atom_start = tl.load(atom_offsets_ptr + sample).to(tl.int64)
        atoms = tl.load(atom_offsets_ptr + sample + 1).to(tl.int64) - atom_start
        token_start = tl.load(token_offsets_ptr + sample).to(tl.int64)
        tokens = tl.load(token_offsets_ptr + sample + 1).to(tl.int64) - token_start
        flags = tl.load(sample_flags_ptr + sample).to(tl.int32)
        empty = (atoms == 0) & (tokens >= 0)
        index_valid = (flags & _NESTED_BAD_INDEX) == 0
        regular = index_valid & ((flags & _NESTED_BAD_REGULAR) == 0)
        interleaved = index_valid & ((flags & _NESTED_BAD_INTERLEAVED) == 0)
        sorted_layout = index_valid & ((flags & _NESTED_BAD_SORTED) == 0)
        kind = tl.where(
            empty,
            _NESTED_KIND_SORTED,
            tl.where(
                regular,
                _NESTED_KIND_REGULAR,
                tl.where(
                    interleaved,
                    _NESTED_KIND_INTERLEAVED,
                    tl.where(sorted_layout, _NESTED_KIND_SORTED, _NESTED_KIND_GENERAL),
                ),
            ),
        )
        if not global_sort and (kind == _NESTED_KIND_GENERAL) & (tokens > 0):
            _stable_atom_token_sort(
                atom_to_token_ptr,
                sorted_mapping_ptr,
                permutation_ptr,
                atom_start,
                atoms,
                tokens,
                atom_start,
                block_atoms,
            )
            kind = _NESTED_KIND_INDEXED
        tl.store(sample_kind_ptr + sample, kind)


@triton.jit
def _nested_atom_to_token_descriptor(
    atom_to_token_ptr,
    sorted_mapping_ptr,
    token_batch_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    sample_kind_ptr,
    segment_start_ptr,
    total_tokens,
    global_sort: tl.constexpr,
) -> None:
    global_token = tl.program_id(0).to(tl.int64)
    mask = global_token < total_tokens
    sample = tl.load(token_batch_ptr + global_token, mask=mask, other=0).to(tl.int64)
    atom_start = tl.load(atom_offsets_ptr + sample, mask=mask, other=0).to(tl.int64)
    atom_end = tl.load(atom_offsets_ptr + sample + 1, mask=mask, other=0).to(tl.int64)
    token_start = tl.load(token_offsets_ptr + sample, mask=mask, other=0).to(tl.int64)
    token_end = tl.load(token_offsets_ptr + sample + 1, mask=mask, other=1).to(tl.int64)
    atoms = atom_end - atom_start
    tokens = token_end - token_start
    token = global_token - token_start
    width = atoms // tl.maximum(tokens, 1)
    kind = tl.load(sample_kind_ptr + sample, mask=mask, other=_NESTED_KIND_INVALID).to(tl.int32)
    sorted_start = atom_start
    if kind == _NESTED_KIND_SORTED:
        sorted_start += _nested_atom_to_token_lower_bound(atom_to_token_ptr, atom_start, atoms, token)
    elif kind >= _NESTED_KIND_GENERAL:
        key = global_token if global_sort else token
        sorted_start += _nested_atom_to_token_lower_bound(sorted_mapping_ptr, atom_start, atoms, key)
    regular_start = atom_start + token * width
    interleaved_start = atom_start + token
    start = tl.where(kind == _NESTED_KIND_SORTED, sorted_start, regular_start)
    start = tl.where(kind == _NESTED_KIND_INTERLEAVED, interleaved_start, start)
    start = tl.where(kind >= _NESTED_KIND_GENERAL, sorted_start, start)
    tl.store(segment_start_ptr + global_token, start, mask=mask)


@triton.jit
def _nested_atom_selection(
    atom_to_token_ptr,
    permutation_ptr,
    atoms,
    lane_mask,
    kind,
):
    if kind >= _NESTED_KIND_GENERAL:
        atoms = tl.load(permutation_ptr + atoms, mask=lane_mask, other=0).to(tl.int64)
    return atoms


@triton.jit
def _nested_runtime_descriptor(
    token,
    token_batch_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    sample_kind_ptr,
    segment_start_ptr,
    token_count_ptr,
):
    sample = tl.load(token_batch_ptr + token).to(tl.int64)
    token_start = tl.load(token_offsets_ptr + sample).to(tl.int64)
    token_end = tl.load(token_offsets_ptr + sample + 1).to(tl.int64)
    sample_tokens = token_end - token_start
    kind = tl.load(sample_kind_ptr + sample).to(tl.int32)
    assigned_count = tl.load(token_count_ptr + token).to(tl.int64)
    start = tl.load(segment_start_ptr + token).to(tl.int64)
    stride = tl.where(kind == _NESTED_KIND_INTERLEAVED, sample_tokens, 1)
    scan_count = assigned_count
    return start, stride, scan_count, assigned_count, kind


@triton.jit
def _nested_atom_to_token_reduce_forward(
    atom_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    permutation_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    token_batch_ptr,
    sample_kind_ptr,
    segment_start_ptr,
    token_count_ptr,
    out_ptr,
    total_tokens,
    sample_size: tl.constexpr,
    mask_sample_size: tl.constexpr,
    channels,
    has_atom_mask: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    segment = tl.program_id(0).to(tl.int64)
    token = segment // sample_size
    group = segment % sample_size
    channel_pid = tl.program_id(1)
    start, stride, scan_count, assigned_count, kind = _nested_runtime_descriptor(
        token,
        token_batch_ptr,
        atom_offsets_ptr,
        token_offsets_ptr,
        sample_kind_ptr,
        segment_start_ptr,
        token_count_ptr,
    )
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else out_ptr.dtype.element_ty
    numerator = tl.zeros((block_channels,), dtype=accum_dtype)
    raw_denominator = tl.zeros((), dtype=accum_dtype)
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atoms = start + relative * stride
        lane_mask = relative < scan_count
        atoms = _nested_atom_selection(atom_to_token_ptr, permutation_ptr, atoms, lane_mask, kind)
        values = tl.load(
            atom_ptr + (atoms[:, None] * sample_size + group) * channels + channel_offsets[None, :],
            mask=lane_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(accum_dtype)
        if has_atom_mask:
            weights = tl.load(
                atom_mask_ptr + atoms * mask_sample_size + group % mask_sample_size,
                mask=lane_mask,
                other=0.0,
            ).to(accum_dtype)
            values = (values * weights[:, None]).to(accum_dtype)
            raw_denominator += tl.sum(weights, axis=0, dtype=accum_dtype)
        numerator += tl.sum(values, axis=0, dtype=accum_dtype)
    raw_denominator = raw_denominator if has_atom_mask else assigned_count.to(accum_dtype)
    if normalize:
        denominator = tl.where(
            raw_denominator > 0,
            raw_denominator + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        output = (numerator / denominator).to(accum_dtype)
    else:
        output = numerator
    tl.store(
        out_ptr + (token * sample_size + group) * channels + channel_offsets,
        output,
        mask=channel_mask,
    )


@triton.jit
def _nested_atom_to_token_reduce_backward(
    grad_out_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    permutation_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    token_batch_ptr,
    sample_kind_ptr,
    segment_start_ptr,
    token_count_ptr,
    grad_atom_ptr,
    total_tokens,
    sample_size: tl.constexpr,
    mask_sample_size: tl.constexpr,
    channels,
    has_atom_mask: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    accumulate_fp32: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    segment = tl.program_id(0).to(tl.int64)
    token = segment // sample_size
    group = segment % sample_size
    channel_pid = tl.program_id(1)
    start, stride, scan_count, assigned_count, kind = _nested_runtime_descriptor(
        token,
        token_batch_ptr,
        atom_offsets_ptr,
        token_offsets_ptr,
        sample_kind_ptr,
        segment_start_ptr,
        token_count_ptr,
    )
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    accum_dtype = tl.float32 if accumulate_fp32 else grad_atom_ptr.dtype.element_ty
    grad = tl.load(
        grad_out_ptr + (token * sample_size + group) * channels + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(accum_dtype)
    if has_atom_mask:
        raw_denominator = tl.zeros((), dtype=accum_dtype)
        for atom_base in range(0, scan_count, block_atoms):
            relative = atom_base + lanes
            atoms = start + relative * stride
            lane_mask = relative < scan_count
            atoms = _nested_atom_selection(atom_to_token_ptr, permutation_ptr, atoms, lane_mask, kind)
            weights = tl.load(
                atom_mask_ptr + atoms * mask_sample_size + group % mask_sample_size,
                mask=lane_mask,
                other=0.0,
            ).to(accum_dtype)
            raw_denominator += tl.sum(weights, axis=0, dtype=accum_dtype)
    else:
        raw_denominator = assigned_count.to(accum_dtype)
    if normalize:
        denominator = tl.where(
            raw_denominator > 0,
            raw_denominator + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atoms = start + relative * stride
        lane_mask = relative < scan_count
        atoms = _nested_atom_selection(atom_to_token_ptr, permutation_ptr, atoms, lane_mask, kind)
        if has_atom_mask:
            weights = tl.load(
                atom_mask_ptr + atoms * mask_sample_size + group % mask_sample_size,
                mask=lane_mask,
                other=0.0,
            ).to(accum_dtype)
            if normalize:
                values = (grad[None, :] * (weights[:, None] / denominator).to(accum_dtype)).to(accum_dtype)
            else:
                values = (grad[None, :] * weights[:, None]).to(accum_dtype)
        else:
            if normalize:
                values = (grad[None, :] / denominator).to(accum_dtype)
            else:
                values = grad[None, :]
        tl.store(
            grad_atom_ptr + (atoms[:, None] * sample_size + group) * channels + channel_offsets[None, :],
            values,
            mask=lane_mask[:, None] & channel_mask[None, :],
        )


@triton.jit
def _nested_atom_to_token_reduce_mask_state(
    atom_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    permutation_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    token_batch_ptr,
    sample_kind_ptr,
    segment_start_ptr,
    token_count_ptr,
    numerator_ptr,
    raw_denominator_ptr,
    total_tokens,
    sample_size: tl.constexpr,
    mask_sample_size: tl.constexpr,
    channels,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    segment = tl.program_id(0).to(tl.int64)
    token = segment // sample_size
    group = segment % sample_size
    channel_pid = tl.program_id(1)
    start, stride, scan_count, _assigned_count, kind = _nested_runtime_descriptor(
        token,
        token_batch_ptr,
        atom_offsets_ptr,
        token_offsets_ptr,
        sample_kind_ptr,
        segment_start_ptr,
        token_count_ptr,
    )
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    channel_offsets = channel_pid * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    accum_dtype = numerator_ptr.dtype.element_ty
    numerator = tl.zeros((block_channels,), dtype=accum_dtype)
    raw_denominator = tl.zeros((), dtype=accum_dtype)
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atoms = start + relative * stride
        lane_mask = relative < scan_count
        atoms = _nested_atom_selection(atom_to_token_ptr, permutation_ptr, atoms, lane_mask, kind)
        weights = tl.load(
            atom_mask_ptr + atoms * mask_sample_size + group % mask_sample_size,
            mask=lane_mask,
            other=0.0,
        ).to(accum_dtype)
        values = tl.load(
            atom_ptr + (atoms[:, None] * sample_size + group) * channels + channel_offsets[None, :],
            mask=lane_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(accum_dtype)
        numerator += tl.sum((values * weights[:, None]).to(accum_dtype), axis=0, dtype=accum_dtype)
        raw_denominator += tl.sum(weights, axis=0, dtype=accum_dtype)
    tl.store(
        numerator_ptr + (token * sample_size + group) * channels + channel_offsets,
        numerator,
        mask=channel_mask,
    )
    tl.store(
        raw_denominator_ptr + token * sample_size + group,
        raw_denominator,
        mask=channel_pid == 0,
    )


@triton.jit
def _nested_atom_to_token_reduce_mask_backward(
    atom_ptr,
    atom_to_token_ptr,
    permutation_ptr,
    atom_offsets_ptr,
    token_offsets_ptr,
    token_batch_ptr,
    sample_kind_ptr,
    segment_start_ptr,
    token_count_ptr,
    numerator_ptr,
    raw_denominator_ptr,
    grad_out_ptr,
    grad_mask_ptr,
    partial_ptr,
    total_tokens,
    sample_size: tl.constexpr,
    mask_sample_size: tl.constexpr,
    write_partial: tl.constexpr,
    channels,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    block_channels: tl.constexpr,
    block_atoms: tl.constexpr,
) -> None:
    segment = tl.program_id(0).to(tl.int64)
    token = segment // sample_size
    group = segment % sample_size
    start, stride, scan_count, _assigned_count, kind = _nested_runtime_descriptor(
        token,
        token_batch_ptr,
        atom_offsets_ptr,
        token_offsets_ptr,
        sample_kind_ptr,
        segment_start_ptr,
        token_count_ptr,
    )
    lanes = tl.arange(0, block_atoms).to(tl.int64)
    accum_dtype = numerator_ptr.dtype.element_ty
    if normalize:
        raw_denominator = tl.load(raw_denominator_ptr + token * sample_size + group).to(accum_dtype)
        active = raw_denominator > 0
        denominator = tl.where(
            active,
            raw_denominator + tl.full((), denominator_offset, accum_dtype),
            tl.full((), 1.0, accum_dtype),
        )
        denominator_sq = (denominator * denominator).to(accum_dtype)
    for atom_base in range(0, scan_count, block_atoms):
        relative = atom_base + lanes
        atoms = start + relative * stride
        lane_mask = relative < scan_count
        atoms = _nested_atom_selection(atom_to_token_ptr, permutation_ptr, atoms, lane_mask, kind)
        atom_dot = tl.zeros((block_atoms,), dtype=tl.float32)
        if normalize:
            correction = tl.zeros((), dtype=tl.float32)
        for channel_start in range(0, channels, block_channels):
            channel_offsets = channel_start + tl.arange(0, block_channels)
            channel_mask = channel_offsets < channels
            grad = tl.load(
                grad_out_ptr + (token * sample_size + group) * channels + channel_offsets,
                mask=channel_mask,
                other=0.0,
            ).to(accum_dtype)
            values = tl.load(
                atom_ptr + (atoms[:, None] * sample_size + group) * channels + channel_offsets[None, :],
                mask=lane_mask[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(accum_dtype)
            if normalize:
                numerator = tl.load(
                    numerator_ptr + (token * sample_size + group) * channels + channel_offsets,
                    mask=channel_mask,
                    other=0.0,
                ).to(accum_dtype)
                grad_numerator = (grad / denominator).to(accum_dtype)
                correction_values = (-grad * numerator).to(accum_dtype)
                correction_values = (correction_values / denominator_sq).to(accum_dtype)
                correction += tl.sum(correction_values.to(tl.float32), axis=0)
            else:
                grad_numerator = grad
            atom_dot += tl.sum(
                values.to(tl.float32) * grad_numerator[None, :].to(tl.float32),
                axis=1,
            )

        if normalize:
            result = (atom_dot.to(accum_dtype) + tl.where(active, correction.to(accum_dtype), 0.0)).to(accum_dtype)
        else:
            result = atom_dot.to(accum_dtype)
        if write_partial:
            tl.store(
                partial_ptr + atoms * sample_size + group,
                result.to(tl.float32),
                mask=lane_mask,
            )
        else:
            tl.store(
                grad_mask_ptr + atoms * mask_sample_size + group % mask_sample_size,
                result,
                mask=lane_mask,
            )


@triton.jit
def _nested_atom_to_token_reduce_mask_reduce_samples(
    partial_ptr,
    grad_mask_ptr,
    total_atoms,
    sample_size: tl.constexpr,
    block_samples: tl.constexpr,
) -> None:

    atom = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, block_samples)
    total = tl.zeros((), dtype=tl.float32)
    for sample_start in tl.range(0, sample_size, block_samples):
        samples = sample_start + lanes
        values = tl.load(
            partial_ptr + atom * sample_size + samples,
            mask=(atom < total_atoms) & (samples < sample_size),
            other=0.0,
        )
        total += tl.sum(values, axis=0)
    tl.store(grad_mask_ptr + atom, total, mask=atom < total_atoms)


@triton.jit
def _atom_to_token_reduce_sorted_vjp(
    atom_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    permutation_ptr,
    grad_out_ptr,
    grad_atom_ptr,
    grad_mask_ptr,
    atoms,
    tokens,
    channels: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    normalize: tl.constexpr,
    denominator_offset: tl.constexpr,
    need_grad_atom: tl.constexpr,
    has_permutation: tl.constexpr,
    max_iters: tl.constexpr,
    block_atoms: tl.constexpr,
    block_channels: tl.constexpr,
    token_offsets_ptr=None,
    token_offsets_stride: tl.constexpr = 1,
    has_token_offsets: tl.constexpr = False,
) -> None:
    pid = tl.program_id(0).to(tl.int64)
    batch = pid // tokens
    token = pid % tokens
    if has_token_offsets:
        start = tl.load(token_offsets_ptr + token * token_offsets_stride).to(tl.int64)
        end = tl.load(token_offsets_ptr + (token + 1) * token_offsets_stride).to(tl.int64)
    else:
        start = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token, max_iters)
        end = _atom_to_token_sorted_lower_bound(atom_to_token_ptr, atoms, token + 1, max_iters)
    positions = start + tl.arange(0, block_atoms)
    active_atoms = positions < end
    if has_permutation:
        sources = tl.load(permutation_ptr + positions, mask=active_atoms, other=0).to(tl.int64)
    else:
        sources = positions
    weights = tl.load(
        atom_mask_ptr + (batch // mask_batch_repeat) * atoms + sources,
        mask=active_atoms,
        other=0.0,
    ).to(tl.float32)
    if normalize:
        count = tl.sum(weights, axis=0)
        denominator = tl.where(count > 0, count + denominator_offset, 1.0)
        denominator_sq = denominator * denominator
    mask_gradient = tl.zeros((block_atoms,), tl.float32)
    correction = tl.full((), 0.0, tl.float32)
    for channel_start in range(0, channels, block_channels):
        channel_offsets = channel_start + tl.arange(0, block_channels)
        active_channels = channel_offsets < channels
        gradient = tl.load(
            grad_out_ptr + (batch * tokens + token) * channels + channel_offsets,
            mask=active_channels,
            other=0.0,
        ).to(tl.float32)
        atom = tl.load(
            atom_ptr + (batch * atoms + sources[:, None]) * channels + channel_offsets[None, :],
            mask=active_atoms[:, None] & active_channels[None, :],
            other=0.0,
        ).to(tl.float32)
        if normalize:
            numerator = tl.sum(atom * weights[:, None], axis=0)
            grad_numerator = gradient / denominator
            correction += tl.sum((-gradient * numerator) / denominator_sq, axis=0)
        else:
            grad_numerator = gradient
        mask_gradient += tl.sum(atom * grad_numerator[None, :], axis=1)
        if need_grad_atom:
            if normalize:
                grad_atom = gradient[None, :] * (weights[:, None] / denominator)
            else:
                grad_atom = gradient[None, :] * weights[:, None]
            tl.store(
                grad_atom_ptr + (batch * atoms + sources[:, None]) * channels + channel_offsets[None, :],
                grad_atom,
                mask=active_atoms[:, None] & active_channels[None, :],
            )
    if normalize:
        mask_gradient += tl.where(count > 0, correction, 0.0)
    tl.store(grad_mask_ptr + batch * atoms + sources, mask_gradient, mask=active_atoms)


@triton.jit
def _atom_to_token_reduce_sorted_sum_vjp(
    atom_ptr,
    atom_mask_ptr,
    atom_to_token_ptr,
    inverse_permutation_ptr,
    grad_out_ptr,
    grad_atom_ptr,
    grad_mask_ptr,
    atoms,
    tokens,
    channels: tl.constexpr,
    mask_batch_repeat: tl.constexpr,
    need_grad_atom: tl.constexpr,
    has_inverse_permutation: tl.constexpr,
    block_atoms: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(1).to(tl.int64)
    atom_offsets = tl.program_id(0).to(tl.int64) * block_atoms + tl.arange(0, block_atoms)
    active_atoms = atom_offsets < atoms
    if has_inverse_permutation:
        positions = tl.load(inverse_permutation_ptr + atom_offsets, mask=active_atoms, other=0).to(tl.int64)
    else:
        positions = atom_offsets
    positions = tl.minimum(tl.maximum(positions, 0), atoms - 1)
    token = tl.load(atom_to_token_ptr + positions, mask=active_atoms, other=0).to(tl.int64)
    token = tl.minimum(tl.maximum(token, 0), tokens - 1)
    weights = tl.load(
        atom_mask_ptr + (batch // mask_batch_repeat) * atoms + atom_offsets,
        mask=active_atoms,
        other=0.0,
    ).to(tl.float32)
    mask_gradient = tl.zeros((block_atoms,), tl.float32)
    for channel_start in range(0, channels, block_channels):
        channel_offsets = channel_start + tl.arange(0, block_channels)
        active = active_atoms[:, None] & (channel_offsets[None, :] < channels)
        gradient = tl.load(
            grad_out_ptr + (batch * tokens + token[:, None]) * channels + channel_offsets[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        atom = tl.load(
            atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        mask_gradient += tl.sum(atom * gradient, axis=1)
        if need_grad_atom:
            tl.store(
                grad_atom_ptr + (batch * atoms + atom_offsets[:, None]) * channels + channel_offsets[None, :],
                gradient * weights[:, None],
                mask=active,
            )
    tl.store(grad_mask_ptr + batch * atoms + atom_offsets, mask_gradient, mask=active_atoms)
