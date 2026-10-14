from __future__ import annotations

import torch
from torch import Tensor

_NESTED_CATEGORICAL_MSA_METADATA_SCHEMA = (
    "(Tensor token_sizes, Tensor token_rows, Tensor token_cells, Tensor token_packed, "
    "SymInt token_total, Tensor continuous_sizes, Tensor continuous_rows, "
    "Tensor continuous_cells, Tensor continuous_packed, SymInt continuous_total) -> Tensor"
)


def _validate_offset_metadata(name: str, offsets: Tensor) -> None:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError(f"categorical_msa_projection {name} offsets must be a one-dimensional CPU int64 tensor")
    if offsets.numel() == 0:
        raise ValueError(f"categorical_msa_projection {name} offsets must include the initial zero")


def _validate_element_size_metadata(name: str, element_sizes: Tensor, batch_size: int, element_rank: int) -> None:
    if (
        element_sizes.device.type != "cpu"
        or element_sizes.dtype is not torch.int64
        or element_sizes.dim() != 2
        or element_sizes.shape != (batch_size, element_rank)
    ):
        raise ValueError(
            f"categorical_msa_projection {name} element sizes must be a "
            f"(batch_size, {element_rank}) CPU int64 tensor"
        )


@torch.library.custom_op(
    "ooperators::_validate_nested_categorical_msa_projection_metadata",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_CATEGORICAL_MSA_METADATA_SCHEMA,
)
def validate_nested_categorical_msa_projection_metadata(
    token_sizes: Tensor,
    token_rows: Tensor,
    token_cells: Tensor,
    token_packed: Tensor,
    token_total: int,
    continuous_sizes: Tensor,
    continuous_rows: Tensor,
    continuous_cells: Tensor,
    continuous_packed: Tensor,
    continuous_total: int,
) -> Tensor:

    batch_size = token_rows.numel() - 1
    _validate_element_size_metadata("tokens", token_sizes, batch_size, 2)
    _validate_element_size_metadata("continuous", continuous_sizes, batch_size, 3)
    for name, offsets in (
        ("token row", token_rows),
        ("token cell", token_cells),
        ("token packed", token_packed),
        ("continuous row", continuous_rows),
        ("continuous cell", continuous_cells),
        ("continuous packed", continuous_packed),
    ):
        _validate_offset_metadata(name, offsets)

    if not bool(torch.all(continuous_sizes[:, 2] == 2)):
        raise ValueError("categorical_msa_projection continuous element feature dimension must be 2")
    if (
        token_total != continuous_total
        or not torch.equal(token_sizes, continuous_sizes[:, :2])
        or not torch.equal(token_rows, continuous_rows)
        or not torch.equal(token_cells, continuous_cells)
        or not torch.equal(token_packed, continuous_packed)
    ):
        raise ValueError(
            "categorical_msa_projection tokens and continuous must have the same (M_i, N_i) layout per sample"
        )

    if bool(torch.any(token_sizes < 0)):
        raise ValueError("categorical_msa_projection token element sizes must be non-negative")
    if token_packed.shape != token_rows.shape:
        raise ValueError("categorical_msa_projection packed offsets must contain one boundary per sample")
    if int(token_rows[0]) != 0 or int(token_cells[0]) != 0 or int(token_packed[0]) != 0:
        raise ValueError("categorical_msa_projection offsets must start at zero")
    if (
        int(token_rows[-1]) != token_cells.numel() - 1
        or int(token_cells[-1]) != token_total
        or int(token_packed[-1]) != token_total
    ):
        raise ValueError("categorical_msa_projection offsets do not cover the packed values")

    msa_sizes = token_sizes[:, 0]
    sequence_sizes = token_sizes[:, 1]
    if not torch.equal(token_rows[1:] - token_rows[:-1], msa_sizes):
        raise ValueError("categorical_msa_projection row offsets do not match element sizes")
    row_tokens = torch.repeat_interleave(sequence_sizes, msa_sizes, output_size=token_cells.numel() - 1)
    if not torch.equal(token_cells[1:] - token_cells[:-1], row_tokens):
        raise ValueError("categorical_msa_projection cell offsets do not match element sizes")
    if not torch.equal(token_packed[1:] - token_packed[:-1], msa_sizes * sequence_sizes):
        raise ValueError("categorical_msa_projection two-level offsets and packed offsets are inconsistent")
    return torch.ones((), dtype=torch.bool, device=token_rows.device)


@torch.library.register_fake(validate_nested_categorical_msa_projection_metadata)
def _validate_nested_categorical_msa_projection_metadata_fake(
    token_sizes: Tensor,
    token_rows: Tensor,
    token_cells: Tensor,
    token_packed: Tensor,
    token_total: int,
    continuous_sizes: Tensor,
    continuous_rows: Tensor,
    continuous_cells: Tensor,
    continuous_packed: Tensor,
    continuous_total: int,
) -> Tensor:
    del (
        token_sizes,
        token_cells,
        token_packed,
        token_total,
        continuous_sizes,
        continuous_rows,
        continuous_cells,
        continuous_packed,
        continuous_total,
    )
    return torch.empty((), dtype=torch.bool, device=token_rows.device)
