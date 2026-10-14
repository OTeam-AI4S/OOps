r"""
Tensor metadata keys for runtime caches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TensorCacheKey:
    r"""
    Cache key tied to tensor storage identity, metadata, and in-place version.
    """

    data_ptr: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, slots=True)
class TensorMetadataKey:
    r"""Tensor facts that determine a shape/stride/dtype/device-only execution plan."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    contiguous: bool


def tensor_cache_key(tensor: Tensor) -> TensorCacheKey | None:
    r"""Return a storage/version key, or ``None`` for an inference tensor."""

    if torch.is_inference(tensor):
        return None
    return TensorCacheKey(
        data_ptr=tensor.untyped_storage().data_ptr(),
        version=tensor._version,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def tensor_metadata_key(tensor: Tensor) -> TensorMetadataKey:
    r"""Return metadata safe for caches whose result never depends on tensor identity or data."""

    return TensorMetadataKey(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
        contiguous=tensor.is_contiguous(),
    )
