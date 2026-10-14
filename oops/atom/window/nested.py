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
from torch import Tensor

from oops.atom.layout import _validate_packed_offsets

_NESTED_ATOM_WINDOW_METADATA_SCHEMA = (
    "(Tensor pair_row_offsets, Tensor pair_inner_offsets, Tensor atom_offsets, Tensor? num_valid_atoms, "
    "SymInt pair_total, SymInt atom_total, int n_queries, int n_keys) -> "
    "(Tensor, Tensor, Tensor, Tensor)"
)

_NESTED_VALID_ATOM_EXTENTS_SCHEMA = "(Tensor num_valid_atoms, Tensor atom_offsets) -> Tensor"


@torch.library.custom_op(
    "ooperators::_canonicalize_nested_atom_window_metadata",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_ATOM_WINDOW_METADATA_SCHEMA,
)
def canonicalize_nested_atom_window_metadata(
    pair_row_offsets: Tensor,
    pair_inner_offsets: Tensor,
    atom_offsets: Tensor,
    num_valid_atoms: Tensor | None,
    pair_total: int,
    atom_total: int,
    n_queries: int,
    n_keys: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    return canonicalize_nested_atom_window_metadata_eager(
        pair_row_offsets,
        pair_inner_offsets,
        atom_offsets,
        num_valid_atoms,
        pair_total,
        atom_total,
        n_queries,
        n_keys,
    )


def canonicalize_nested_atom_window_metadata_eager(
    pair_row_offsets: Tensor,
    pair_inner_offsets: Tensor,
    atom_offsets: Tensor,
    num_valid_atoms: Tensor | None,
    pair_total: int,
    atom_total: int,
    n_queries: int,
    n_keys: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    _validate_packed_offsets(
        "token_pair_to_atom_window_gather",
        pair_row_offsets,
        pair_inner_offsets.numel() - 1,
        operand="pair rows",
    )
    _validate_packed_offsets(
        "token_pair_to_atom_window_gather",
        pair_inner_offsets,
        pair_total,
        operand="pair cells",
    )
    _validate_packed_offsets(
        "token_pair_to_atom_window_gather",
        atom_offsets,
        atom_total,
        operand="atom_to_token",
    )
    if pair_row_offsets.numel() != atom_offsets.numel():
        raise ValueError("token_pair_to_atom_window_gather nested pair and atom_to_token batches must match")

    token_lengths = pair_row_offsets[1:] - pair_row_offsets[:-1]
    atom_lengths = atom_offsets[1:] - atom_offsets[:-1]
    total_rows = int(pair_row_offsets[-1])
    row_sample = torch.repeat_interleave(
        torch.arange(token_lengths.numel(), dtype=torch.long),
        token_lengths,
        output_size=total_rows,
    )
    row_widths = pair_inner_offsets[1:] - pair_inner_offsets[:-1]
    expected_widths = token_lengths.index_select(0, row_sample)
    if not torch.equal(row_widths, expected_widths):
        raise ValueError("token_pair_to_atom_window_gather nested pair token dimensions must be square (N_i, N_i)")
    if bool(torch.any((token_lengths == 0) & (atom_lengths > 0))):
        raise ValueError(
            "token_pair_to_atom_window_gather each non-empty atom_to_token element requires at least one token"
        )

    pair_cell_offsets = pair_inner_offsets.index_select(0, pair_row_offsets)
    block_lengths = torch.div(atom_lengths + n_queries - 1, n_queries, rounding_mode="floor")
    block_offsets = torch.cat((block_lengths.new_zeros(1), block_lengths.cumsum(0)))
    total_blocks = int(block_offsets[-1])
    block_to_sample = torch.repeat_interleave(
        torch.arange(block_lengths.numel(), dtype=torch.long),
        block_lengths,
        output_size=total_blocks,
    )

    if num_valid_atoms is None:
        normalized_valid_atoms = atom_lengths.new_zeros((atom_lengths.numel() + 1,))
    else:
        if (
            num_valid_atoms.device.type != "cpu"
            or num_valid_atoms.dtype is not torch.int64
            or num_valid_atoms.dim() != 1
            or num_valid_atoms.shape != atom_lengths.shape
        ):
            raise ValueError(
                "token_pair_to_atom_window_gather num_valid_atoms must be a one-dimensional CPU int64 "
                "tensor with one value per sample"
            )
        if bool(torch.any((num_valid_atoms < 0) | (num_valid_atoms > atom_lengths))):
            raise ValueError("num_valid_atoms must lie in [0, A_i] for every nested sample")
        normalized_valid_atoms = torch.cat((num_valid_atoms, num_valid_atoms.new_zeros(1)))

    sample_metadata = torch.cat(
        (
            pair_cell_offsets,
            block_offsets,
            normalized_valid_atoms,
        )
    ).contiguous()
    validation = torch.ones((), dtype=torch.bool, device=pair_row_offsets.device)
    return (
        sample_metadata,
        block_lengths.contiguous(),
        block_to_sample.contiguous(),
        validation,
    )


@torch.library.register_fake(canonicalize_nested_atom_window_metadata)
def _canonicalize_nested_atom_window_metadata_fake(
    pair_row_offsets: Tensor,
    pair_inner_offsets: Tensor,
    atom_offsets: Tensor,
    num_valid_atoms: Tensor | None,
    pair_total: int,
    atom_total: int,
    n_queries: int,
    n_keys: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del pair_inner_offsets, atom_total, pair_total, n_queries, n_keys
    samples = atom_offsets.shape[0] - 1
    total_blocks = torch.library.get_ctx().new_dynamic_size(min=0)
    del num_valid_atoms
    return (
        torch.empty((3 * (samples + 1),), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((samples,), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((total_blocks,), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((), dtype=torch.bool, device=pair_row_offsets.device),
    )


@torch.library.custom_op(
    "ooperators::_validate_nested_valid_atom_extents",
    mutates_args=(),
    device_types="cuda",
    schema=_NESTED_VALID_ATOM_EXTENTS_SCHEMA,
)
def validate_nested_valid_atom_extents(num_valid_atoms: Tensor, atom_offsets: Tensor) -> Tensor:

    atom_lengths = atom_offsets[1:] - atom_offsets[:-1]
    return torch.all((num_valid_atoms >= 0) & (num_valid_atoms <= atom_lengths))


@torch.library.register_fake(validate_nested_valid_atom_extents)
def _validate_nested_valid_atom_extents_fake(num_valid_atoms: Tensor, atom_offsets: Tensor) -> Tensor:
    del atom_offsets
    return torch.empty((), dtype=torch.bool, device=num_valid_atoms.device)
