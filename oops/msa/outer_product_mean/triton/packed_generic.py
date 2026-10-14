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

import torch

from oops.runtime import cuda_device_context, require_native_matmul_precision_state

from ..composed import _execute_contraction, _execute_contraction_backward


def _topology(left, mask, sizes, offsets):
    if sizes.device.type != "cpu" or sizes.dtype is not torch.int64 or sizes.dim() != 2 or sizes.shape[1] != 3:
        raise ValueError("outer_product_mean requires CPU int64 MSA element sizes")
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError("outer_product_mean requires CPU int64 pair offsets")
    topology = sizes.tolist()
    pairs = offsets.tolist()
    expected = [0]
    for depth, tokens, channels in topology:
        if depth < 0 or tokens < 0 or channels <= 0:
            raise ValueError("outer_product_mean MSA element sizes disagree with the public domain")
        expected.append(expected[-1] + tokens * tokens)
    if pairs != expected or sum(m * n for m, n, _ in topology) != left.shape[0] or mask.shape[0] != left.shape[0]:
        raise ValueError("outer_product_mean packed rows disagree with element sizes and pair offsets")
    return topology, expected[-1]


def _execute_forward(
    left,
    right,
    mask,
    weight,
    bias,
    pair_offsets_cpu,
    msa_sizes_cpu,
    input_precision,
    state,
    divide_before,
    eps,
    clamp,
    save_state,
):
    with cuda_device_context(left.device), torch.autocast("cuda", enabled=False):
        require_native_matmul_precision_state("outer_product_mean", state)
        topology, pairs = _topology(left, mask, msa_sizes_cpu, pair_offsets_cpu)
        output = torch.empty((pairs, weight.shape[0]), device=left.device, dtype=weight.dtype)
        outer = torch.empty((pairs, left.shape[1] ** 2), device=left.device, dtype=left.dtype) if save_state else None
        numerator = torch.empty_like(output) if save_state else None
        counts = torch.empty((pairs, 1), device=left.device, dtype=torch.float32) if save_state else None
        row_start = pair_start = 0
        for depth, tokens, _channels in topology:
            row_stop, pair_stop = row_start + depth * tokens, pair_start + tokens * tokens
            left_sample = left[row_start:row_stop].reshape(1, depth, tokens, left.shape[1])
            right_sample = right[row_start:row_stop].reshape_as(left_sample)
            keep = mask[row_start:row_stop].float().reshape(1, depth, tokens)
            count = torch.einsum("bsi,bsj->bij", keep, keep).unsqueeze(-1)
            result, item_outer, item_numerator = _execute_contraction(
                left_sample,
                right_sample,
                count,
                weight,
                bias,
                input_precision,
                state,
                divide_before,
                eps,
                clamp,
                save_state,
            )
            output[pair_start:pair_stop].copy_(result.reshape(-1, weight.shape[0]))
            if save_state:
                outer[pair_start:pair_stop].copy_(item_outer.reshape(-1, left.shape[1] ** 2))
                numerator[pair_start:pair_stop].copy_(item_numerator.reshape(-1, weight.shape[0]))
                counts[pair_start:pair_stop].copy_(count.reshape(-1, 1))
            row_start, pair_start = row_stop, pair_stop
        return output, outer, numerator, counts


def _execute_backward(
    gradient,
    left,
    right,
    mask,
    weight,
    bias,
    pair_offsets_cpu,
    msa_sizes_cpu,
    outer,
    numerator,
    counts,
    input_precision,
    state,
    divide_before,
    eps,
    clamp,
    need_left,
    need_right,
    need_mask,
    need_weight,
    need_bias,
):
    with cuda_device_context(left.device), torch.autocast("cuda", enabled=False):
        require_native_matmul_precision_state("outer_product_mean", state)
        topology, _ = _topology(left, mask, msa_sizes_cpu, pair_offsets_cpu)
        grad_left = torch.empty_like(left) if need_left else None
        grad_right = torch.empty_like(right) if need_right else None
        grad_mask = torch.empty_like(mask) if need_mask else None
        grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if need_weight else None
        grad_bias = (
            torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if need_bias and bias is not None else None
        )
        row_start = pair_start = 0
        for depth, tokens, _channels in topology:
            row_stop, pair_stop = row_start + depth * tokens, pair_start + tokens * tokens
            left_sample = left[row_start:row_stop].reshape(1, depth, tokens, left.shape[1])
            right_sample = right[row_start:row_stop].reshape_as(left_sample)
            item_outer = outer[pair_start:pair_stop].reshape(1, tokens, left.shape[1], tokens, left.shape[1])
            item_numerator = numerator[pair_start:pair_stop].reshape(1, tokens, tokens, weight.shape[0])
            count = counts[pair_start:pair_stop].reshape(1, tokens, tokens, 1)
            grads = _execute_contraction_backward(
                gradient[pair_start:pair_stop].reshape_as(item_numerator),
                left_sample,
                right_sample,
                count,
                weight,
                bias,
                item_outer,
                item_numerator,
                input_precision,
                state,
                divide_before,
                eps,
                clamp,
                need_left,
                need_right,
                need_mask,
                need_weight,
                need_bias,
            )
            if need_left:
                grad_left[row_start:row_stop].copy_(grads[0].reshape(-1, left.shape[1]))
            if need_right:
                grad_right[row_start:row_stop].copy_(grads[1].reshape(-1, right.shape[1]))
            if need_mask:
                keep = mask[row_start:row_stop].float().reshape(1, depth, tokens)
                incoming = grads[2].squeeze(-1)
                result = torch.einsum("bsj,bij->bsi", keep, incoming) + torch.einsum("bsi,bij->bsj", keep, incoming)
                grad_mask[row_start:row_stop].copy_(result.reshape(-1))
            if need_weight:
                grad_weight.add_(grads[3])
            if grad_bias is not None:
                grad_bias.add_(grads[4])
            row_start, pair_start = row_stop, pair_stop
        return (
            grad_left,
            grad_right,
            grad_mask,
            None if grad_weight is None else grad_weight.to(weight.dtype),
            None if grad_bias is None else grad_bias.to(bias.dtype),
        )
