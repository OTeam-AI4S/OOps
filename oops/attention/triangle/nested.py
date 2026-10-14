from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import functorch_transforms_active, is_compiling, unsupported


def _square_row_sizes(operator: str, operand: str, row_offsets: Tensor, packed_offsets: Tensor, total: int) -> Tensor:

    if row_offsets.numel() <= 1:
        raise unsupported(operator, "does not serve an empty NestedTensor batch")
    if packed_offsets.shape != row_offsets.shape:
        raise ValueError(f"{operator} {operand} packed offsets must contain one boundary per sample")
    row_sizes = row_offsets[1:] - row_offsets[:-1]
    if bool(torch.any(row_sizes <= 0)):
        raise unsupported(operator, "requires every nested pair element to have N > 0")
    if not torch.equal(packed_offsets[1:] - packed_offsets[:-1], row_sizes.square()):
        raise ValueError(f"{operator} {operand} token dims must be square (N, N) per sample")
    if total <= 0:
        raise unsupported(operator, "requires non-empty packed pair values")
    return row_sizes


_TRIANGLE_ATTENTION_SAMPLE_TABLE_SCHEMA = (
    "(Tensor query_rows, Tensor query_packed, SymInt query_total_cells, "
    "Tensor key_rows, Tensor key_packed, SymInt key_total_cells, "
    "Tensor value_rows, Tensor value_packed, SymInt value_total_cells, "
    "Tensor? mask_rows, Tensor? mask_packed, SymInt mask_total_cells, "
    "Tensor? bias_groups, Tensor? bias_packed, int bias_mode, "
    "SymInt bias_total_values) -> Tensor"
)


def triangle_attention_sample_table(
    query_rows: Tensor,
    query_packed: Tensor,
    query_total_cells: int,
    key_rows: Tensor,
    key_packed: Tensor,
    key_total_cells: int,
    value_rows: Tensor,
    value_packed: Tensor,
    value_total_cells: int,
    mask_rows: Tensor | None,
    mask_packed: Tensor | None,
    mask_total_cells: int,
    bias_groups: Tensor | None,
    bias_packed: Tensor | None,
    bias_mode: int,
    bias_total_values: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    inputs = (
        query_rows,
        query_packed,
        query_total_cells,
        key_rows,
        key_packed,
        key_total_cells,
        value_rows,
        value_packed,
        value_total_cells,
        mask_rows,
        mask_packed,
        mask_total_cells,
        bias_groups,
        bias_packed,
        bias_mode,
        bias_total_values,
    )
    tensors = tuple(value for value in inputs if isinstance(value, Tensor))
    plain = (
        not is_compiling()
        and not is_in_torch_dispatch_mode()
        and not functorch_transforms_active()
        and all(
            type(tensor) is Tensor
            and tensor.device.type == "cpu"
            and tensor.dim() == 1
            and tensor.dtype in (torch.int32, torch.int64)
            for tensor in tensors
        )
        and not torch.overrides.has_torch_function(tensors)
    )
    capturing = plain and device is not None and device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    if plain and not capturing and query_rows.numel() <= 1025 and sum(tensor.numel() for tensor in tensors) <= 8192:
        key = tuple(
            (value.dtype, tuple(value.shape), tuple(value.tolist())) if isinstance(value, Tensor) else value
            for value in inputs
        )
        result = _cached_sample_table(key, device)
        if result.device.type == "cuda":
            result.record_stream(torch.cuda.current_stream(result.device))
        return result
    if capturing:
        # Copy scalar geometry by value into capture-owned storage. A cached CUDA
        # table can be evicted after capture, and a pageable H2D copy is invalid.
        rows = _build_sample_table(*inputs).tolist()
        result = torch.empty((len(rows), len(rows[0])), dtype=torch.int64, device=device)
        for sample, row in enumerate(rows):
            for column, value in enumerate(row):
                result[sample, column].fill_(value)
        return result
    result = _triangle_attention_sample_table(*inputs)
    return result if device is None else result.to(device)


@lru_cache(maxsize=128)
def _cached_sample_table(key: tuple, device: torch.device | None) -> Tensor:
    inputs: tuple[Any, ...] = tuple(
        torch.tensor(value[2], dtype=value[0]).reshape(value[1]) if isinstance(value, tuple) else value for value in key
    )
    result = _build_sample_table(*inputs)
    return result if device is None else result.to(device)


@torch.library.custom_op(
    "ooperators::_triangle_attention_sample_table",
    mutates_args=(),
    device_types="cpu",
    schema=_TRIANGLE_ATTENTION_SAMPLE_TABLE_SCHEMA,
)
def _triangle_attention_sample_table(
    query_rows: Tensor,
    query_packed: Tensor,
    query_total_cells: int,
    key_rows: Tensor,
    key_packed: Tensor,
    key_total_cells: int,
    value_rows: Tensor,
    value_packed: Tensor,
    value_total_cells: int,
    mask_rows: Tensor | None,
    mask_packed: Tensor | None,
    mask_total_cells: int,
    bias_groups: Tensor | None,
    bias_packed: Tensor | None,
    bias_mode: int,
    bias_total_values: int,
) -> Tensor:
    r"""Table columns: cell_start, cols, groups, bias_start, bias_groups,
    bias_group_stride, bias_head_stride. Bias modes: 0 absent, 1 shared, 2 per row.
    """

    return _build_sample_table(
        query_rows,
        query_packed,
        query_total_cells,
        key_rows,
        key_packed,
        key_total_cells,
        value_rows,
        value_packed,
        value_total_cells,
        mask_rows,
        mask_packed,
        mask_total_cells,
        bias_groups,
        bias_packed,
        bias_mode,
        bias_total_values,
    )


def _build_sample_table(
    query_rows: Tensor,
    query_packed: Tensor,
    query_total_cells: int,
    key_rows: Tensor,
    key_packed: Tensor,
    key_total_cells: int,
    value_rows: Tensor,
    value_packed: Tensor,
    value_total_cells: int,
    mask_rows: Tensor | None,
    mask_packed: Tensor | None,
    mask_total_cells: int,
    bias_groups: Tensor | None,
    bias_packed: Tensor | None,
    bias_mode: int,
    bias_total_values: int,
) -> Tensor:
    operator = "triangle_attention"
    sizes = _square_row_sizes(operator, "query", query_rows, query_packed, query_total_cells)
    for operand, rows, packed, total in (
        ("key", key_rows, key_packed, key_total_cells),
        ("value", value_rows, value_packed, value_total_cells),
    ):
        if total != query_total_cells or not torch.equal(rows, query_rows) or not torch.equal(packed, query_packed):
            raise ValueError(f"triangle_attention {operand} layout must match the query layout per sample")

    if mask_rows is None:
        if mask_packed is not None or mask_total_cells != 0:
            raise ValueError("triangle_attention mask metadata is incomplete")
    else:
        if mask_packed is None:
            raise ValueError("triangle_attention mask metadata is incomplete")
        if (
            mask_total_cells != query_total_cells
            or not torch.equal(mask_rows, query_rows)
            or not torch.equal(mask_packed, query_packed)
        ):
            raise ValueError("triangle_attention mask layout must match the query layout per sample")

    bias_start = torch.zeros_like(sizes)
    bias_group_counts = torch.ones_like(sizes)
    if bias_mode == 0:
        if bias_groups is not None or bias_packed is not None or bias_total_values != 0:
            raise ValueError("triangle_attention triangle_bias metadata is incomplete")
    elif bias_mode == 1:
        if bias_groups is not None or bias_packed is None:
            raise ValueError("triangle_attention shared triangle_bias metadata is incomplete")
        if bias_packed.shape != query_packed.shape or not torch.equal(bias_packed, query_packed):
            raise ValueError("triangle_attention shared triangle_bias layout must match query per sample")
        if bias_total_values != query_total_cells:
            raise ValueError("triangle_attention shared triangle_bias packed values do not match its layout")
        bias_start = bias_packed[:-1]
    elif bias_mode == 2:
        if bias_groups is None or bias_packed is None:
            raise ValueError("triangle_attention triangle_bias metadata is incomplete")
        if bias_groups.shape != query_rows.shape or bias_packed.shape != query_packed.shape:
            raise ValueError("triangle_attention triangle_bias must have one packed boundary per sample")
        bias_group_counts = bias_groups[1:] - bias_groups[:-1]
        valid_group_counts = (bias_group_counts == 1) | (bias_group_counts == sizes)
        if not bool(torch.all(valid_group_counts)):
            raise ValueError("triangle_attention triangle_bias group axis must be 1 or N_i per sample")
        expected_values = bias_group_counts * sizes.square()
        if not torch.equal(bias_packed[1:] - bias_packed[:-1], expected_values):
            raise ValueError("triangle_attention triangle_bias query/key axes must both have length N_i per sample")
        if bias_total_values <= 0:
            raise ValueError("triangle_attention triangle_bias packed values must be non-empty")
        bias_start = bias_packed[:-1]
    else:
        raise ValueError(f"triangle_attention received invalid triangle_bias metadata mode {bias_mode}")

    columns = (
        query_packed[:-1],
        sizes,
        sizes,
        bias_start,
        bias_group_counts,
        sizes.square(),
        torch.full_like(sizes, bias_total_values),
    )
    return torch.stack(columns, dim=1).contiguous()


@torch.library.register_fake(_triangle_attention_sample_table)
def _triangle_attention_sample_table_fake(
    query_rows: Tensor,
    query_packed: Tensor,
    query_total_cells: int,
    key_rows: Tensor,
    key_packed: Tensor,
    key_total_cells: int,
    value_rows: Tensor,
    value_packed: Tensor,
    value_total_cells: int,
    mask_rows: Tensor | None,
    mask_packed: Tensor | None,
    mask_total_cells: int,
    bias_groups: Tensor | None,
    bias_packed: Tensor | None,
    bias_mode: int,
    bias_total_values: int,
) -> Tensor:
    return torch.empty((query_rows.shape[0] - 1, 7), device=query_rows.device, dtype=torch.int64)


_BIAS_ORDER_SCHEMA = "(Tensor bias, Tensor? bias_groups, Tensor bias_packed, Tensor query_rows) -> Tensor"


def _bias_blocks(bias_groups: Tensor | None, bias_packed: Tensor, query_rows: Tensor) -> list[tuple[int, int, int]]:

    sizes = (query_rows[1:] - query_rows[:-1]).tolist()
    counts = [1] * len(sizes) if bias_groups is None else (bias_groups[1:] - bias_groups[:-1]).tolist()
    return list(zip(sizes, counts, bias_packed[:-1].tolist()))


@torch.library.custom_op("ooperators::_triangle_attention_bias_query_major", mutates_args=(), schema=_BIAS_ORDER_SCHEMA)
def triangle_attention_bias_query_major(
    bias: Tensor, bias_groups: Tensor | None, bias_packed: Tensor, query_rows: Tensor
) -> Tensor:

    heads = bias.shape[1]
    pieces = []
    for size, count, start in _bias_blocks(bias_groups, bias_packed, query_rows):
        block = bias[start : start + count * size * size].reshape(count, size, size, heads)
        pieces.append(block.permute(3, 0, 2, 1).reshape(heads, -1))
    return torch.cat(pieces, dim=1)


@torch.library.register_fake(triangle_attention_bias_query_major)
def _triangle_attention_bias_query_major_fake(
    bias: Tensor, bias_groups: Tensor | None, bias_packed: Tensor, query_rows: Tensor
) -> Tensor:
    return torch.empty((bias.shape[1], bias.shape[0]), device=bias.device, dtype=bias.dtype)


@torch.library.custom_op("ooperators::_triangle_attention_bias_key_major", mutates_args=(), schema=_BIAS_ORDER_SCHEMA)
def triangle_attention_bias_key_major(
    bias: Tensor, bias_groups: Tensor | None, bias_packed: Tensor, query_rows: Tensor
) -> Tensor:

    heads = bias.shape[0]
    pieces = []
    for size, count, start in _bias_blocks(bias_groups, bias_packed, query_rows):
        block = bias[:, start : start + count * size * size].reshape(heads, count, size, size)
        pieces.append(block.permute(1, 3, 2, 0).reshape(-1, heads))
    return torch.cat(pieces, dim=0)


@torch.library.register_fake(triangle_attention_bias_key_major)
def _triangle_attention_bias_key_major_fake(
    bias: Tensor, bias_groups: Tensor | None, bias_packed: Tensor, query_rows: Tensor
) -> Tensor:
    return torch.empty((bias.shape[1], bias.shape[0]), device=bias.device, dtype=bias.dtype)


def _bias_query_major_setup_context(ctx, inputs, output) -> None:
    _bias, bias_groups, bias_packed, query_rows = inputs
    ctx.save_for_backward(bias_groups, bias_packed, query_rows)


def _bias_query_major_backward(ctx, grad_output: Tensor):
    bias_groups, bias_packed, query_rows = ctx.saved_tensors
    return triangle_attention_bias_key_major(grad_output, bias_groups, bias_packed, query_rows), None, None, None


torch.library.register_autograd(
    triangle_attention_bias_query_major, _bias_query_major_backward, setup_context=_bias_query_major_setup_context
)
