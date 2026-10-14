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

from math import prod

import torch
import triton
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import guard_or_true, statically_known_false

from oops.core import TargetDescriptor
from oops.core._math import _ceil_div
from oops.runtime import (
    CUDA_MAX_GRID_X,
    FLOAT32_DOT_TF32X3,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps,
    cuda_device_context,
    require_triton_dot_precision,
    shared_memory_budget,
)

from . import kernel
from .kernel import ROW_ALIGNMENT, SAMPLE_TABLE_WIDTH
from .profiles import tile_table
from .schedule import MAX_HEAD_SIZE, AttentionTile, RowTile, TileTable, TriangleAttentionPhase, full_width

_OPERATOR = "triangle_attention"


_FLOAT32_DOT = FLOAT32_DOT_TF32X3


_DOT_PHASES = tuple(
    phase for phase in TriangleAttentionPhase if phase is not TriangleAttentionPhase.BACKWARD_PREPROCESS
)


def _kernel_operand(tensor: Tensor) -> Tensor:
    # Raw pointer loads cannot observe logical view bits.
    return tensor.resolve_neg().resolve_conj().contiguous()


def supports_triangle_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    *,
    uniform_groups: int = 0,
    uniform_cols: int = 0,
) -> bool:

    return (
        query.is_cuda
        and query.dim() in (3, 5)
        and query.dtype in STANDARD_FLOAT_DTYPES
        and key.shape == query.shape
        and value.shape == query.shape
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and all(
            operand.device == query.device for operand in (key, value, mask, bias, sample_table) if operand is not None
        )
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and guard_or_true(query.numel() > 0)
        and query.shape[-2] > 0
        and 0 < query.shape[-1] <= MAX_HEAD_SIZE
        and (mask is None or (mask.dtype is torch.int16 and mask.dim() == 1))
        and (
            bias is None
            or (bias.dtype in (query.dtype, torch.float32) and bias.dim() in (1, 5) and bias.is_contiguous())
        )
        and sample_table.dtype is torch.int64
        and sample_table.dim() == 2
        and sample_table.shape[1] == SAMPLE_TABLE_WIDTH
        and sample_table.shape[0] > 0
        and _phases_fit(
            query.dtype,
            query.shape[-1],
            query.device,
            num_heads=query.shape[-2],
            num_samples=sample_table.shape[0],
            uniform_groups=uniform_groups,
            uniform_cols=uniform_cols,
        )
    )


def _phases_fit(
    dtype: torch.dtype,
    head_size: int,
    device: torch.device,
    *,
    num_heads: int,
    num_samples: int,
    uniform_groups: int,
    uniform_cols: int,
) -> bool:

    target = cuda_device_caps(device)
    table = tile_table(target)
    budget = shared_memory_budget(target)
    ragged = isinstance(uniform_cols, int) and uniform_cols == 0
    tiles = {phase: table.tile(phase, dtype, head_size, budget, ragged=ragged) for phase in _DOT_PHASES}
    if any(tile is None for tile in tiles.values()):
        return False
    if ragged:
        return True
    rows = table.row_tile(dtype, head_size, ragged=False)
    launchable = uniform_groups * num_heads * _ceil_div(uniform_cols, rows.block_rows) <= CUDA_MAX_GRID_X
    for phase, tile in tiles.items():
        assert tile is not None
        if phase is TriangleAttentionPhase.BACKWARD_SHARED_BIAS:
            programs = (
                num_samples
                * num_heads
                * _ceil_div(uniform_cols, tile.block_query)
                * _ceil_div(uniform_cols, tile.block_key)
            )
        else:
            block = tile.block_key if phase is TriangleAttentionPhase.BACKWARD_KEY_VALUE else tile.block_query
            programs = uniform_groups * num_heads * _ceil_div(uniform_cols, block)
        launchable = launchable & (programs <= CUDA_MAX_GRID_X)
    if launchable is True or launchable is False:
        return launchable
    if statically_known_false(launchable):
        return False
    torch._check(launchable)
    return True


def _phase_tile(
    table: TileTable,
    phase: TriangleAttentionPhase,
    dtype: torch.dtype,
    head_size: int,
    target: TargetDescriptor,
    *,
    ragged: bool,
) -> AttentionTile:

    tile = table.tile(phase, dtype, head_size, shared_memory_budget(target), ragged=ragged)
    if tile is None:
        raise RuntimeError(f"triangle attention {phase.value} tile disappeared after admission")
    return tile


def _run_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
    *,
    table: TileTable | None = None,
) -> tuple[Tensor, Tensor]:

    del bias_shared_only
    output = torch.empty_like(query, dtype=torch.float32)
    lse = torch.empty((prod(query.shape[:-2]), query.shape[-2]), device=query.device, dtype=torch.float32)
    _launch_forward(
        kernel._forward,
        TriangleAttentionPhase.FORWARD,
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
        output,
        lse,
        table,
    )
    return output, lse


def _run_inference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
    *,
    table: TileTable | None = None,
) -> Tensor:

    del bias_shared_only
    output = torch.empty_like(query)
    _launch_forward(
        kernel._inference,
        TriangleAttentionPhase.INFERENCE,
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
        output,
        None,
        table,
    )
    return output


def _launch_forward(
    entry: triton.JITFunction,
    phase: TriangleAttentionPhase,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
    output: Tensor,
    lse: Tensor | None,
    table: TileTable | None,
) -> None:
    with cuda_device_context(query.device):
        query, key, value = (_kernel_operand(tensor) for tensor in (query, key, value))
        mask = None if mask is None else _kernel_operand(mask)
        bias = None if bias is None else _kernel_operand(bias)
        sample_table = _kernel_operand(sample_table)
        require_triton_dot_precision(_OPERATOR, query.dtype, _FLOAT32_DOT, precision)
        num_heads, head_size = query.shape[-2:]
        target = cuda_device_caps(query.device)
        tile = _phase_tile(
            tile_table(target) if table is None else table,
            phase,
            query.dtype,
            head_size,
            target,
            ragged=uniform_cols == 0,
        )
        num_programs = _programs(target, tile, uniform_groups * num_heads * _ceil_div(uniform_cols, tile.block_query))
        operands = [query, key, value, _pointer(mask, query), _pointer(bias, query), output]
        if lse is not None:
            operands.append(lse)
        entry[(num_programs,)](
            *operands,
            sample_table,
            sample_table.shape[0],
            num_heads,
            num_programs,
            _resolve_scale(head_size, scale),
            head_size,
            mask is not None,
            bias is not None,
            qkv_transposed,
            mask_transposed,
            _aligned_rows(uniform_cols),
            _even_tiles(uniform_cols, tile.block_query, tile.block_key),
            tile.block_query,
            tile.block_key,
            full_width(head_size),
            input_precision=precision,
            num_warps=tile.num_warps,
            num_stages=tile.num_stages,
        )


def _run_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    lse: Tensor,
    sample_table: Tensor,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
    *,
    table: TileTable | None = None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:

    with cuda_device_context(query.device):
        grad_output = _kernel_operand(grad_output.to(query.dtype))
        query, key = _kernel_operand(query), _kernel_operand(key)
        if need_grad_query or need_grad_key or need_grad_bias:
            value, output = _kernel_operand(value), _kernel_operand(output)
        mask = None if mask is None else _kernel_operand(mask)
        bias = None if bias is None else _kernel_operand(bias)
        lse, sample_table = _kernel_operand(lse), _kernel_operand(sample_table)
        num_heads, head_size = query.shape[-2:]
        num_samples = sample_table.shape[0]
        target = cuda_device_caps(query.device)
        if table is None:
            table = tile_table(target)
        effective_scale = _resolve_scale(head_size, scale)
        block_channel = full_width(head_size)
        keep_mask = _pointer(mask, query)
        kernel_bias = _pointer(bias, query)
        need_grad_bias = need_grad_bias and bias is not None
        aligned = _aligned_rows(uniform_cols)
        common = (
            head_size,
            mask is not None,
            bias is not None,
            qkv_transposed,
            mask_transposed,
            aligned,
        )

        ragged = uniform_cols == 0

        delta: Tensor | None = None
        if need_grad_query or need_grad_key or need_grad_bias:
            delta = torch.empty_like(lse)
            if not (need_grad_query or (need_grad_bias and not bias_shared_only)):
                rows = table.row_tile(query.dtype, head_size, ragged=ragged)
                num_programs = _programs(
                    target, rows, uniform_groups * num_heads * _ceil_div(uniform_cols, rows.block_rows)
                )
                kernel._backward_preprocess[(num_programs,)](
                    output,
                    grad_output,
                    delta,
                    sample_table,
                    num_samples,
                    num_heads,
                    num_programs,
                    head_size,
                    qkv_transposed,
                    aligned,
                    _even_tiles(uniform_cols, rows.block_rows, rows.block_rows),
                    rows.block_rows,
                    block_channel,
                    num_warps=rows.num_warps,
                )

        grad_bias: Tensor | None = None
        if need_grad_bias and not bias_shared_only:
            assert bias is not None
            grad_bias = torch.empty_like(bias)
        grad_query: Tensor | None = None
        if need_grad_query or (need_grad_bias and not bias_shared_only):
            assert delta is not None
            tile = _phase_tile(
                table, TriangleAttentionPhase.BACKWARD_QUERY, query.dtype, head_size, target, ragged=ragged
            )
            num_programs = _programs(
                target, tile, uniform_groups * num_heads * _ceil_div(uniform_cols, tile.block_query)
            )
            grad_query = torch.empty_like(query) if need_grad_query else None
            kernel._backward_query[(num_programs,)](
                query,
                key,
                value,
                keep_mask,
                kernel_bias,
                lse,
                output,
                delta,
                grad_output,
                _pointer(grad_query, query),
                _pointer(grad_bias, query),
                sample_table,
                num_samples,
                num_heads,
                num_programs,
                effective_scale,
                *common,
                _even_tiles(uniform_cols, tile.block_query, tile.block_key),
                need_grad_query,
                need_grad_bias and not bias_shared_only,
                tile.block_query,
                tile.block_key,
                block_channel,
                input_precision=precision,
                num_warps=tile.num_warps,
                num_stages=tile.num_stages,
            )

        grad_key: Tensor | None = None
        grad_value: Tensor | None = None
        if need_grad_key or need_grad_value:
            tile = _phase_tile(
                table, TriangleAttentionPhase.BACKWARD_KEY_VALUE, query.dtype, head_size, target, ragged=ragged
            )
            num_programs = _programs(target, tile, uniform_groups * num_heads * _ceil_div(uniform_cols, tile.block_key))
            grad_key = torch.empty_like(key) if need_grad_key else None
            grad_value = torch.empty(value.shape, dtype=value.dtype, device=value.device) if need_grad_value else None
            kernel._backward_key_value[(num_programs,)](
                query,
                key,
                value,
                keep_mask,
                kernel_bias,
                lse,
                _pointer(delta, lse),
                grad_output,
                _pointer(grad_key, query),
                _pointer(grad_value, query),
                sample_table,
                num_samples,
                num_heads,
                num_programs,
                effective_scale,
                *common,
                _even_tiles(uniform_cols, tile.block_query, tile.block_key),
                need_grad_key,
                need_grad_value,
                tile.block_query,
                tile.block_key,
                block_channel,
                input_precision=precision,
                num_warps=(
                    max(tile.num_warps, 8 if block_channel >= 256 else 4)
                    if query.dtype == torch.float32 and precision == "tf32x3" and need_grad_key and need_grad_value
                    else tile.num_warps
                ),
                num_stages=tile.num_stages,
            )

        if need_grad_bias and (bias_shared_only or ragged):
            assert bias is not None and delta is not None
            tile = _phase_tile(
                table, TriangleAttentionPhase.BACKWARD_SHARED_BIAS, query.dtype, head_size, target, ragged=ragged
            )
            score_tiles = _ceil_div(uniform_cols, tile.block_query) * _ceil_div(uniform_cols, tile.block_key)
            split = table.shared_bias_split
            group_parts = (
                split.parts(num_samples * num_heads * score_tiles, uniform_groups, bias.numel(), bias.element_size())
                if split is not None and bias_shared_only and not ragged
                else 1
            )
            if group_parts > 1:
                partials = torch.empty((group_parts, *bias.shape), device=bias.device, dtype=torch.float32)
            else:
                if grad_bias is None:
                    grad_bias = torch.empty_like(bias)
                partials = grad_bias
            num_programs = _programs(target, tile, num_samples * group_parts * num_heads * score_tiles)
            kernel._backward_shared_bias[(num_programs,)](
                query,
                key,
                value,
                keep_mask,
                bias,
                lse,
                delta,
                grad_output,
                partials,
                sample_table,
                num_samples,
                num_heads,
                num_programs,
                effective_scale,
                group_parts,
                bias.numel() if group_parts > 1 else 0,
                head_size,
                mask is not None,
                qkv_transposed,
                mask_transposed,
                aligned,
                _even_tiles(uniform_cols, tile.block_query, tile.block_key),
                tile.block_query,
                tile.block_key,
                block_channel,
                input_precision=precision,
                num_warps=tile.num_warps,
                num_stages=tile.num_stages,
            )
            if group_parts > 1:
                grad_bias = partials.sum(0).to(bias.dtype)

        return grad_query, grad_key, grad_value, grad_bias


def _programs(target: TargetDescriptor, tile: AttentionTile | RowTile, uniform_tiles: int) -> int:

    if uniform_tiles > 0:
        if uniform_tiles > CUDA_MAX_GRID_X:
            raise RuntimeError(f"triangle attention launch of {uniform_tiles} programs escaped grid admission")
        return uniform_tiles
    multiprocessors = target.resources.multiprocessor_count
    if multiprocessors is None:
        raise RuntimeError("triangle attention persistent kernels need the target multiprocessor count")
    return multiprocessors * tile.programs_per_multiprocessor


def _pointer(tensor: Tensor | None, placeholder: Tensor) -> Tensor:

    return placeholder if tensor is None else tensor


def _resolve_scale(head_size: int, scale: float | None) -> float:
    return float(head_size**-0.5 if scale is None else scale)


def _aligned_rows(uniform_cols: int) -> bool:

    return uniform_cols > 0 and uniform_cols % ROW_ALIGNMENT == 0


def _even_tiles(uniform_cols: int, block_query: int, block_key: int) -> bool:

    return uniform_cols > 0 and uniform_cols % block_query == 0 and uniform_cols % block_key == 0
