from __future__ import annotations

from torch import Tensor


def degenerate_shape(tensor: Tensor, rank: int) -> tuple[int, ...]:

    return (1,) * (rank - tensor.dim()) + tuple(tensor.shape)


def degenerate_strides(tensor: Tensor, rank: int) -> tuple[int, ...]:
    r"""Omitted leading axes have extent one and stride zero."""

    return (0,) * (rank - tensor.dim()) + tuple(tensor.stride())


def check_head_major_qkv(query: Tensor, key: Tensor, value: Tensor) -> None:

    if query.dim() < 3:
        raise ValueError("query must have layout (num_heads, query_length, head_size) under an optional prefix")
    if key.dim() != query.dim() or value.dim() != query.dim():
        raise ValueError("query, key, and value must have the same rank")
    if key.shape[:-2] != query.shape[:-2] or value.shape[:-2] != key.shape[:-2]:
        raise ValueError("query, key, and value must share prefix and num_heads dimensions")
    if key.shape[-1] != query.shape[-1]:
        raise ValueError("query and key head_size dimensions must match")
    if value.shape[-1] != query.shape[-1]:
        raise ValueError("query and value head_size dimensions must match")
    if value.shape[-2] != key.shape[-2]:
        raise ValueError("key and value key_len dimensions must match")


def optional_unit_batch(tensor: Tensor | None) -> Tensor | None:

    return None if tensor is None else tensor.unsqueeze(0)
