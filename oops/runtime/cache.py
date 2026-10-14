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
