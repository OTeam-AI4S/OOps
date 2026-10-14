from __future__ import annotations

import torch
from torch import Tensor

from oops.atom.layout import _validate_packed_offsets

_NESTED_TOKEN_ATOM_OFFSETS_SCHEMA = (
    "(str operator, Tensor token_offsets, Tensor atom_offsets, SymInt token_total, SymInt atom_total) -> Tensor"
)

_NESTED_ATOM_TO_TOKEN_REDUCTION_OFFSETS_SCHEMA = (
    "(Tensor token_offsets, Tensor atom_offsets, Tensor mapping_offsets, Tensor? mask_offsets, "
    "SymInt token_total, SymInt atom_total, SymInt mapping_total, SymInt mask_total) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_validate_nested_token_atom_offsets",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_TOKEN_ATOM_OFFSETS_SCHEMA,
)
def validate_nested_token_atom_offsets(
    operator: str,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    token_total: int,
    atom_total: int,
) -> Tensor:

    _validate_packed_offsets(operator, token_offsets, token_total, operand="token")
    _validate_packed_offsets(operator, atom_offsets, atom_total, operand="atom_to_token")
    if token_offsets.numel() != atom_offsets.numel():
        raise ValueError(f"{operator} token and atom_to_token NestedTensor batches must match")

    token_lengths = token_offsets[1:] - token_offsets[:-1]
    atom_lengths = atom_offsets[1:] - atom_offsets[:-1]
    if bool(torch.any((token_lengths == 0) & (atom_lengths > 0))):
        raise ValueError(f"{operator} each non-empty atom_to_token element requires at least one token")
    return torch.ones((), dtype=torch.bool, device=token_offsets.device)


@torch.library.register_fake(validate_nested_token_atom_offsets)
def _validate_nested_token_atom_offsets_fake(
    operator: str,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    token_total: int,
    atom_total: int,
) -> Tensor:
    del operator, atom_offsets, token_total, atom_total
    return torch.empty((), dtype=torch.bool, device=token_offsets.device)


@torch.library.custom_op(
    "ooperators::_validate_nested_atom_to_token_reduce_offsets",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_ATOM_TO_TOKEN_REDUCTION_OFFSETS_SCHEMA,
)
def validate_nested_atom_to_token_reduce_offsets(
    token_offsets: Tensor,
    atom_offsets: Tensor,
    mapping_offsets: Tensor,
    mask_offsets: Tensor | None,
    token_total: int,
    atom_total: int,
    mapping_total: int,
    mask_total: int,
) -> Tensor:

    _validate_packed_offsets("atom_to_token_mean", token_offsets, token_total, operand="token_reference")
    _validate_packed_offsets("atom_to_token_mean", atom_offsets, atom_total, operand="atom")
    _validate_packed_offsets("atom_to_token_mean", mapping_offsets, mapping_total, operand="atom_to_token")
    if token_offsets.numel() != atom_offsets.numel() or atom_offsets.numel() != mapping_offsets.numel():
        raise ValueError("atom_to_token_mean nested token, atom, and atom_to_token batches must match")
    if not torch.equal(atom_offsets, mapping_offsets):
        raise ValueError("atom_to_token_mean nested atom and atom_to_token layouts must match")
    if mask_offsets is not None:
        _validate_packed_offsets("atom_to_token_mean", mask_offsets, mask_total, operand="atom_mask")
        if not torch.equal(atom_offsets, mask_offsets):
            raise ValueError("atom_to_token_mean nested atom and atom_mask layouts must match")

    token_lengths = token_offsets[1:] - token_offsets[:-1]
    atom_lengths = atom_offsets[1:] - atom_offsets[:-1]
    if bool(torch.any((token_lengths == 0) & (atom_lengths > 0))):
        raise ValueError("atom_to_token_mean each non-empty atom element requires at least one token")
    return torch.ones((), dtype=torch.bool, device=token_offsets.device)


@torch.library.register_fake(validate_nested_atom_to_token_reduce_offsets)
def _validate_nested_atom_to_token_reduce_offsets_fake(
    token_offsets: Tensor,
    atom_offsets: Tensor,
    mapping_offsets: Tensor,
    mask_offsets: Tensor | None,
    token_total: int,
    atom_total: int,
    mapping_total: int,
    mask_total: int,
) -> Tensor:
    del (
        atom_offsets,
        mapping_offsets,
        mask_offsets,
        token_total,
        atom_total,
        mapping_total,
        mask_total,
    )
    return torch.empty((), dtype=torch.bool, device=token_offsets.device)
