from __future__ import annotations

import torch
from torch import Tensor

from oops.core._math import _next_power_of_two
from oops.runtime import (
    CUDA_MAX_GRID_X,
    cuda_device_caps_if_needed,
    cuda_device_context,
    is_compiling,
    is_external_fake,
)

from .._metadata import _validate_msa_layout, _validate_token_offsets
from . import kernel
from .profiles import MSA_ANY_SCHEDULE, PAIR_MASK_SCHEDULE
from .schedule import PackedMsaAnyProblem, PackedPairMaskProblem


def _any(values: Tensor, sizes: list[list[int]], offsets: list[int]) -> Tensor:
    config = MSA_ANY_SCHEDULE.generic.default
    block_tokens = config.block_tokens
    layout = []
    merge = []
    out_begin = partial_begin = tile_begin = merge_begin = 0
    for (m, n), begin in zip(sizes, offsets, strict=False):
        if n:
            block_rows = _next_power_of_two(max(1, min(m, config.block_msa)))
            parts = max(1, (m + block_rows - 1) // block_rows)
            layout.append((begin, out_begin, m, n, block_rows, parts, tile_begin, partial_begin))
            tile_begin += parts * ((n + block_tokens - 1) // block_tokens)
            if parts > 1:
                merge.append((partial_begin, out_begin, n, parts, merge_begin))
                merge_begin += (n + block_tokens - 1) // block_tokens
                partial_begin += parts * n
        out_begin += n
    if max(tile_begin, merge_begin) > CUDA_MAX_GRID_X:
        raise ValueError("token_mask_from_msa_mask tile grid exceeds CUDA launch limits")
    problem = PackedMsaAnyProblem(len(sizes), values.numel(), out_begin, tile_begin, merge_begin)
    target = cuda_device_caps_if_needed(values.device, MSA_ANY_SCHEDULE.target_changes_plan(problem))
    config = MSA_ANY_SCHEDULE.resolve(problem, target=target).config
    out = torch.empty(out_begin, device=values.device, dtype=torch.bool)
    partial = torch.empty(partial_begin, device=values.device, dtype=torch.bool) if partial_begin else out
    if tile_begin:
        kernel._msa_any_multi[(tile_begin,)](
            values,
            out,
            partial,
            tuple(layout),
            block_tokens,
            values.numel() >= 2**31,
            num_warps=config.partial_num_warps,
        )
    if merge_begin:
        kernel._msa_merge_multi[(merge_begin,)](
            partial, out, tuple(merge), block_tokens, num_warps=config.output_num_warps
        )
    return out


def _pair(values: Tensor, lengths: list[int]) -> Tensor:
    block = PAIR_MASK_SCHEDULE.generic.default.block_tokens**2
    layout = []
    begin = out_begin = tile_begin = 0
    for n in lengths:
        if n > 3_037_000_499:
            raise ValueError("pair_mask_from_token_mask token length overflows int64 square storage")
        if n:
            layout.append((begin, out_begin, n, tile_begin))
            tile_begin += (n * n + block - 1) // block
        begin += n
        out_begin += n * n
    if tile_begin > CUDA_MAX_GRID_X:
        raise ValueError("pair_mask_from_token_mask tile grid exceeds CUDA launch limits")
    problem = PackedPairMaskProblem(len(lengths), begin, out_begin, tile_begin)
    target = cuda_device_caps_if_needed(values.device, PAIR_MASK_SCHEDULE.target_changes_plan(problem))
    config = PAIR_MASK_SCHEDULE.resolve(problem, target=target).config
    out = torch.empty(out_begin, device=values.device, dtype=torch.bool)
    if tile_begin:
        kernel._pair_mask_multi[(tile_begin,)](
            values,
            out,
            tuple(layout),
            config.block_tokens**2,
            max(begin, out_begin) >= 2**31,
            num_warps=config.num_warps,
        )
    return out


def _msa_impl(values: Tensor, sizes: Tensor, offsets: Tensor, pair: bool) -> Tensor:
    with cuda_device_context(values.device):
        shape, bounds = _validate_msa_layout(sizes, offsets, values.shape[0])
        lengths = [n for m, n in shape]
        token = _any(values, shape, bounds)
        return _pair(token, lengths) if pair else token


torch.library.define(
    "ooperators::_packed_msa_mask", "(Tensor values, Tensor sizes, Tensor offsets, bool pair) -> Tensor"
)


@torch.library.impl("ooperators::_packed_msa_mask", "cuda")
def _packed_msa_mask(values: Tensor, sizes: Tensor, offsets: Tensor, pair: bool) -> Tensor:
    return _msa_impl(values, sizes, offsets, pair)


@torch.library.register_fake("ooperators::_packed_msa_mask")
def _packed_msa_mask_fake(values: Tensor, sizes: Tensor, offsets: Tensor, pair: bool) -> Tensor:
    del offsets, pair
    total = torch.library.get_ctx().new_dynamic_size(min=0)
    return values.new_empty((total,), dtype=torch.bool)


def packed_msa_any(values: Tensor, sizes: Tensor, offsets: Tensor) -> Tensor:
    execute = torch.ops.ooperators._packed_msa_mask.default if is_compiling() or is_external_fake(values) else _msa_impl
    return execute(values, sizes, offsets, False)


def packed_msa_pair_mask(values: Tensor, sizes: Tensor, offsets: Tensor) -> Tensor:
    execute = torch.ops.ooperators._packed_msa_mask.default if is_compiling() or is_external_fake(values) else _msa_impl
    return execute(values, sizes, offsets, True)


def _pair_impl(values: Tensor, offsets: Tensor) -> Tensor:
    with cuda_device_context(values.device):
        bounds = _validate_token_offsets(offsets, values.shape[0])
        lengths = [right - left for left, right in zip(bounds, bounds[1:])]
        return _pair(values, lengths)


torch.library.define("ooperators::_packed_pair_mask", "(Tensor values, Tensor offsets) -> Tensor")


@torch.library.impl("ooperators::_packed_pair_mask", "cuda")
def _packed_pair_mask(values: Tensor, offsets: Tensor) -> Tensor:
    return _pair_impl(values, offsets)


@torch.library.register_fake("ooperators::_packed_pair_mask")
def _packed_pair_mask_fake(values: Tensor, offsets: Tensor) -> Tensor:
    total = torch.library.get_ctx().new_dynamic_size(min=0)
    return values.new_empty((total,), dtype=torch.bool)


def packed_pair_mask(values: Tensor, offsets: Tensor) -> Tensor:
    execute = (
        torch.ops.ooperators._packed_pair_mask.default if is_compiling() or is_external_fake(values) else _pair_impl
    )
    return execute(values, offsets)
