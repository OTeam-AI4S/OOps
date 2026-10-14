from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor

LDDTPairIndices: TypeAlias = Tensor | tuple[Tensor, Tensor]


def flat_pair_indices(pair_indices: LDDTPairIndices) -> Tensor:

    if isinstance(pair_indices, tuple):
        if len(pair_indices) != 2:
            raise ValueError("tuple pair_indices must contain exactly (first_atom_index, second_atom_index)")
        first_atom_index, second_atom_index = pair_indices
        if (
            first_atom_index.dim() != 1
            or second_atom_index.dim() != 1
            or first_atom_index.numel() != second_atom_index.numel()
        ):
            raise ValueError("tuple pair_indices must contain two 1D index tensors of equal length")
        return torch.stack((first_atom_index, second_atom_index), dim=-1).reshape(-1)

    if pair_indices.dim() == 1:
        if pair_indices.numel() % 2 != 0:
            raise ValueError("flat pair_indices must contain an even number of atom indices")
        return pair_indices
    if pair_indices.dim() == 2 and pair_indices.shape[-1] == 2:
        return pair_indices.reshape(-1)
    raise ValueError("pair_indices must be flat, shaped (N_pair, 2), or a tuple (first_atom_index, second_atom_index)")
