from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import unsupported


def canonical_square_cell_offsets(
    operator: str,
    operand: str,
    row_offsets: Tensor,
    inner_offsets: Tensor | None,
    packed_rank: int,
    static_width: int,
    total_positions: int,
) -> Tensor:

    _validate_cpu_offsets(operator, operand, "row", row_offsets)
    if row_offsets.numel() <= 1:
        raise unsupported(operator, "does not serve an empty NestedTensor batch")
    row_sizes = row_offsets[1:] - row_offsets[:-1]
    if bool(torch.any(row_sizes <= 0)):
        raise unsupported(operator, "requires every nested pair element to have N > 0")

    if packed_rank == 2:
        if inner_offsets is not None:
            raise ValueError(f"{operator} {operand} has inconsistent packed rank metadata")
        if static_width <= 0 or not torch.equal(row_sizes, torch.full_like(row_sizes, static_width)):
            raise ValueError(f"{operator} {operand} token dims must be square (N, N) per sample")
        cell_offsets = row_offsets * static_width
    elif packed_rank == 1:
        if inner_offsets is None:
            raise ValueError(f"{operator} {operand} inner offsets are required for flattened pair storage")
        _validate_cpu_offsets(operator, operand, "inner", inner_offsets)
        if inner_offsets.numel() != int(row_offsets[-1]) + 1:
            raise ValueError(f"{operator} {operand} inner offsets do not cover every packed row")
        row_widths = inner_offsets[1:] - inner_offsets[:-1]
        sample_indices = torch.repeat_interleave(
            torch.arange(row_sizes.numel(), dtype=torch.int64),
            row_sizes,
        )
        expected_widths = row_sizes.index_select(0, sample_indices)
        if not torch.equal(row_widths, expected_widths):
            raise ValueError(f"{operator} {operand} token dims must be square (N, N) per sample")
        cell_offsets = inner_offsets.index_select(0, row_offsets)
    else:
        raise ValueError(f"{operator} {operand} packed layout must retain one or two pair axes; got rank {packed_rank}")

    if int(cell_offsets[-1]) != total_positions:
        raise ValueError(
            f"{operator} {operand} packed layout covers {int(cell_offsets[-1])} positions, "
            f"but packed values contain {total_positions}"
        )
    return cell_offsets.contiguous()


def _validate_cpu_offsets(operator: str, operand: str, level: str, offsets: Tensor) -> None:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError(f"{operator} {operand} {level} offsets must be a one-dimensional CPU int64 tensor")
    if offsets.numel() == 0 or int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ValueError(f"{operator} {operand} {level} offsets must be monotonic and start at zero")


_OPERATOR = "triangle_multiplication"


_NESTED_TRIANGLE_METADATA_SCHEMA = (
    "(Tensor pair_sizes, SymInt pair_channels, Tensor pair_row_offsets, Tensor? pair_inner_offsets, "
    "int pair_packed_rank, "
    "SymInt pair_static_width, SymInt pair_total_positions, Tensor? mask_row_offsets, "
    "Tensor? mask_inner_offsets, int mask_packed_rank, SymInt mask_static_width, "
    "SymInt mask_total_positions) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_canonicalize_nested_triangle_metadata",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_TRIANGLE_METADATA_SCHEMA,
)
def canonicalize_nested_triangle_metadata(
    pair_sizes: Tensor,
    pair_channels: int,
    pair_row_offsets: Tensor,
    pair_inner_offsets: Tensor | None,
    pair_packed_rank: int,
    pair_static_width: int,
    pair_total_positions: int,
    mask_row_offsets: Tensor | None,
    mask_inner_offsets: Tensor | None,
    mask_packed_rank: int,
    mask_static_width: int,
    mask_total_positions: int,
) -> Tensor:

    if bool(torch.any(pair_sizes[:, -1] != pair_channels)):
        raise ValueError("triangle_multiplication nested pair channels must match packed values")
    pair_cells = canonical_square_cell_offsets(
        _OPERATOR,
        "pair",
        pair_row_offsets,
        pair_inner_offsets,
        pair_packed_rank,
        pair_static_width,
        pair_total_positions,
    )
    if mask_row_offsets is None:
        if mask_inner_offsets is not None or mask_packed_rank != 0 or mask_total_positions != 0:
            raise ValueError("triangle_multiplication pair_mask metadata is incomplete")
        return pair_row_offsets.clone()

    mask_cells = canonical_square_cell_offsets(
        _OPERATOR,
        "pair_mask",
        mask_row_offsets,
        mask_inner_offsets,
        mask_packed_rank,
        mask_static_width,
        mask_total_positions,
    )
    if not torch.equal(pair_row_offsets, mask_row_offsets) or not torch.equal(pair_cells, mask_cells):
        raise ValueError("triangle_multiplication pair_mask layout must match the pair layout per sample")
    return pair_row_offsets.clone()


@torch.library.register_fake(canonicalize_nested_triangle_metadata)
def _canonicalize_nested_triangle_metadata_fake(
    pair_sizes: Tensor,
    pair_channels: int,
    pair_row_offsets: Tensor,
    pair_inner_offsets: Tensor | None,
    pair_packed_rank: int,
    pair_static_width: int,
    pair_total_positions: int,
    mask_row_offsets: Tensor | None,
    mask_inner_offsets: Tensor | None,
    mask_packed_rank: int,
    mask_static_width: int,
    mask_total_positions: int,
) -> Tensor:
    return torch.empty_like(pair_row_offsets, dtype=torch.int64)
