from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from math import prod

import torch
import triton
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import guard_or_true, statically_known_false
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.core import TargetDescriptor
from oops.core._math import _ceil_div
from oops.runtime import (
    CUDA_MAX_GRID_X,
    FLOAT32_DOT_TF32X3,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps,
    cuda_device_context,
    forward_mode_active,
    functorch_transforms_active,
    is_compiling,
    matmul_precision,
    require_triton_dot_precision,
    shared_memory_budget,
)

from . import kernel
from .kernel import ROW_ALIGNMENT, SAMPLE_TABLE_WIDTH
from .profiles import tile_table
from .schedule import MAX_HEAD_SIZE, AttentionTile, RowTile, TileTable, TriangleAttentionPhase, full_width

_OPERATOR = "triangle_attention"
_FLOAT32_DOT = FLOAT32_DOT_TF32X3
_PLAIN_TENSOR_TYPES = (Tensor, torch.nn.Parameter)
_DOT_PHASES = tuple(
    phase for phase in TriangleAttentionPhase if phase is not TriangleAttentionPhase.BACKWARD_PREPROCESS
)


def _kernel_operand(tensor: Tensor) -> Tensor:
    # Raw pointer loads cannot observe logical view bits.
    return tensor.resolve_neg().resolve_conj().contiguous()


def uniform_sample_table(
    num_samples: int,
    groups: int,
    cols: int,
    num_heads: int,
    bias_groups: int,
    device: torch.device,
) -> Tensor:
    r"""Table columns: cell_start, cols, groups, bias_start, bias_groups,
    bias_group_stride, bias_head_stride.
    """

    block = cols * cols
    if (
        is_compiling()
        or is_in_torch_dispatch_mode()
        or functorch_transforms_active()
        or device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        sample = torch.arange(num_samples, device=device, dtype=torch.int64)
        columns = (
            sample * (groups * cols),
            sample * 0 + cols,
            sample * 0 + groups,
            sample * (bias_groups * num_heads * block),
            sample * 0 + max(bias_groups, 1),
            sample * 0 + num_heads * block,
            sample * 0 + block,
        )
        return torch.stack(columns, dim=1)
    return _cached_uniform_sample_table(num_samples, groups, cols, num_heads, bias_groups, device)


@lru_cache(maxsize=256)
def _cached_uniform_sample_table(
    num_samples: int, groups: int, cols: int, num_heads: int, bias_groups: int, device: torch.device
) -> Tensor:
    block = cols * cols
    rows = [
        (
            sample * groups * cols,
            cols,
            groups,
            sample * bias_groups * num_heads * block,
            max(bias_groups, 1),
            num_heads * block,
            block,
        )
        for sample in range(num_samples)
    ]
    return torch.tensor(rows, dtype=torch.int64, device=device)


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


def triton_triangle_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    sample_table: Tensor,
    *,
    uniform_groups: int,
    uniform_cols: int,
    bias_shared_only: bool,
    qkv_transposed: bool,
    mask_transposed: bool,
    scale: float | None,
    semantic: bool,
) -> Tensor:

    precision = matmul_precision(query.dtype, _FLOAT32_DOT)
    operands = tuple(operand for operand in (query, key, value, bias) if operand is not None)
    needs_grad = torch.is_grad_enabled() and any(operand.requires_grad for operand in operands)
    lean = (
        not semantic
        and all(type(operand) in _PLAIN_TENSOR_TYPES for operand in (*operands, sample_table))
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
        and not forward_mode_active(*operands)
    )
    arguments = (
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
    )
    if lean:
        if needs_grad:
            output, _lse = _TriangleAttention.apply(*arguments)
            return output.to(query.dtype)
        return _run_inference(*arguments)
    if needs_grad:
        output, _lse = _triangle_attention_forward(*arguments)
        return output.to(query.dtype)
    return _triangle_attention_inference(*arguments)


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


_FORWARD_SCHEMA = (
    "(Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor? bias, Tensor sample_table, "
    "SymInt uniform_groups, SymInt uniform_cols, bool bias_shared_only, bool qkv_transposed, "
    "bool mask_transposed, float? scale, str precision) -> (Tensor, Tensor)"
)
# Gradient-demand flags cross the opaque ABI; undemanded slots are None.
_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor? bias, "
    "Tensor output, Tensor lse, Tensor sample_table, SymInt uniform_groups, SymInt uniform_cols, "
    "bool bias_shared_only, bool qkv_transposed, bool mask_transposed, "
    "float? scale, str precision, bool need_grad_query, bool need_grad_key, bool need_grad_value, "
    "bool need_grad_bias) -> (Tensor?, Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_triangle_attention_forward", mutates_args=(), device_types="cuda", schema=_FORWARD_SCHEMA
)
def _triangle_attention_forward(
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
) -> tuple[Tensor, Tensor]:
    return _run_forward(
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
    )


@torch.library.register_fake(_triangle_attention_forward)
def _triangle_attention_forward_fake(
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
) -> tuple[Tensor, Tensor]:
    output = torch.empty_like(query, dtype=torch.float32)
    lse = torch.empty((prod(query.shape[:-2]), query.shape[-2]), device=query.device, dtype=torch.float32)
    return output, lse


_INFERENCE_SCHEMA = (
    "(Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor? bias, Tensor sample_table, "
    "SymInt uniform_groups, SymInt uniform_cols, bool bias_shared_only, bool qkv_transposed, "
    "bool mask_transposed, float? scale, str precision) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_triangle_attention_inference", mutates_args=(), device_types="cuda", schema=_INFERENCE_SCHEMA
)
def _triangle_attention_inference(
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
) -> Tensor:
    return _run_inference(
        query,
        key,
        value,
        mask,
        bias,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
    )


@torch.library.register_fake(_triangle_attention_inference)
def _triangle_attention_inference_fake(
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
) -> Tensor:
    return torch.empty_like(query)


@torch.library.custom_op(
    "ooperators::_triangle_attention_backward", mutates_args=(), device_types="cuda", schema=_BACKWARD_SCHEMA
)
def _triangle_attention_backward(
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
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return _run_backward(
        grad_output,
        query,
        key,
        value,
        mask,
        bias,
        output,
        lse,
        sample_table,
        uniform_groups,
        uniform_cols,
        bias_shared_only,
        qkv_transposed,
        mask_transposed,
        scale,
        precision,
        need_grad_query,
        need_grad_key,
        need_grad_value,
        need_grad_bias,
    )


@torch.library.register_fake(_triangle_attention_backward)
def _triangle_attention_backward_fake(
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
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return (
        torch.empty_like(query) if need_grad_query else None,
        torch.empty_like(key) if need_grad_key else None,
        torch.empty_like(value) if need_grad_value else None,
        torch.empty_like(bias) if need_grad_bias and bias is not None else None,
    )


def _setup_context(ctx, inputs, output) -> None:
    query, key, value, mask, bias, sample_table, *scalars = inputs
    attention_output, lse = output
    # dV needs no delta; dQ, dK, and dBias do.
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[4]
    ctx.scalars = scalars
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(lse)
    ctx.save_for_backward(query, key, value, mask, bias, attention_output if needs_delta else query, lse, sample_table)


def _backward_from_context(ctx, grad_output: Tensor, grad_lse: Tensor | None, *, semantic: bool):
    del grad_lse
    query, key, value, mask, bias, output, lse, sample_table = ctx.saved_tensors
    ctx.maybe_clear_saved_tensors()
    backward: Callable = _triangle_attention_backward if semantic else _run_backward
    gradients = backward(
        grad_output,
        query,
        key,
        value,
        mask,
        bias,
        output,
        lse,
        sample_table,
        *ctx.scalars,
        need_grad_query=ctx.needs_input_grad[0],
        need_grad_key=ctx.needs_input_grad[1],
        need_grad_value=ctx.needs_input_grad[2],
        need_grad_bias=ctx.needs_input_grad[4],
    )
    grad_query, grad_key, grad_value, grad_bias = gradients
    return grad_query, grad_key, grad_value, None, grad_bias, *((None,) * 8)


def _autograd_backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
    return _backward_from_context(ctx, grad_output, grad_lse, semantic=True)


def _plain_autograd_backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
    return _backward_from_context(ctx, grad_output, grad_lse, semantic=torch.is_grad_enabled())


class _TriangleAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = _run_forward(*inputs)
        _setup_context(ctx, inputs, output)
        return output

    backward = staticmethod(_plain_autograd_backward)


torch.library.register_autograd(_triangle_attention_forward, _autograd_backward, setup_context=_setup_context)


def _pointer(tensor: Tensor | None, placeholder: Tensor) -> Tensor:

    return placeholder if tensor is None else tensor


def _resolve_scale(head_size: int, scale: float | None) -> float:
    return float(head_size**-0.5 if scale is None else scale)


def _aligned_rows(uniform_cols: int) -> bool:

    return uniform_cols > 0 and uniform_cols % ROW_ALIGNMENT == 0


def _even_tiles(uniform_cols: int, block_query: int, block_key: int) -> bool:

    return uniform_cols > 0 and uniform_cols % block_query == 0 and uniform_cols % block_key == 0
