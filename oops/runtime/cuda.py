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

r"""CUDA device capabilities and launch-grid validation."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import cache

import torch

from oops.core import TargetDescriptor, TargetResources

# CUDA launch-grid limits from the CUDA Programming Guide.
CUDA_MAX_GRID_X = (1 << 31) - 1
CUDA_MAX_GRID_YZ = (1 << 16) - 1
INT32_MAX = (1 << 31) - 1


def cuda_device_context(device: torch.device) -> AbstractContextManager[None]:
    r"""Run CUDA launches on the operand device and restore the caller's device."""

    return nullcontext() if device.index == torch.cuda.current_device() else torch.cuda.device(device)


def cuda_device_caps(device: torch.device) -> TargetDescriptor:
    r"""Return device capabilities for an explicit or current CUDA device."""

    index = torch.cuda.current_device() if device.index is None else device.index
    if torch.compiler.is_compiling():
        return _read_cuda_device_caps(index)
    return _cuda_device_caps(index)


def cuda_device_caps_if_needed(device: torch.device, needed: bool) -> TargetDescriptor | None:
    r"""Resolve a CUDA target only when a launch phase has device-specific deltas."""

    return cuda_device_caps(device) if needed else None


@cache
def _cuda_device_caps(index: int) -> TargetDescriptor:
    return _read_cuda_device_caps(index)


def _read_cuda_device_caps(index: int) -> TargetDescriptor:
    properties = torch.cuda.get_device_properties(index)
    optin = properties.shared_memory_per_block_optin  # type: ignore[attr-defined]
    shared_memory_per_block = int(optin or properties.shared_memory_per_block)
    major = int(properties.major)
    minor = int(properties.minor)
    return TargetDescriptor(
        backend="cuda",
        architecture=f"sm{major}{minor}",
        # Runtime compute capability does not identify accelerated variants such as sm90a.
        variant=None,
        features=_cuda_architecture_features(major, minor),
        resources=TargetResources(
            shared_memory_per_block=shared_memory_per_block,
            shared_memory_per_multiprocessor=int(properties.shared_memory_per_multiprocessor),
            registers_per_multiprocessor=int(properties.regs_per_multiprocessor),  # type: ignore[attr-defined]
            max_threads_per_block=int(properties.max_threads_per_block),  # type: ignore[attr-defined]
            max_threads_per_multiprocessor=int(properties.max_threads_per_multi_processor),
            warp_size=int(properties.warp_size),
            multiprocessor_count=int(properties.multi_processor_count),
        ),
    )


def _cuda_architecture_features(major: int, minor: int) -> frozenset[str]:
    r"""Return only CUDA ISA-family capabilities established by compute capability."""

    capability = (major, minor)
    features: set[str] = set()
    if capability >= (7, 0):
        features.add("tensor_core")
    if capability >= (8, 0):
        features.update(("bf16_tensor_core", "cp_async", "tf32_tensor_core"))
    return frozenset(features)


def tiled_grid_dim_fits(length: int, block: int, limit: int) -> bool:
    r"""Return whether ``ceil(length / block)`` is a valid non-negative grid dimension."""

    if block <= 0 or limit < 0:
        return False
    # Keep symbolic lengths in an expression; Python branching would specialize them.
    return (length >= 0) & (length <= limit * block)


def _grid_1d_fits(grid: tuple[int]) -> bool:
    return (0 < grid[0]) & (grid[0] <= CUDA_MAX_GRID_X)


def _grid_2d_fits(grid: tuple[int, int]) -> bool:
    return (0 < grid[0]) & (grid[0] <= CUDA_MAX_GRID_X) & (0 < grid[1]) & (grid[1] <= CUDA_MAX_GRID_YZ)


def _grid_3d_fits(grid: tuple[int, int, int]) -> bool:
    return (
        (0 < grid[0])
        & (grid[0] <= CUDA_MAX_GRID_X)
        & (0 < grid[1])
        & (grid[1] <= CUDA_MAX_GRID_YZ)
        & (0 < grid[2])
        & (grid[2] <= CUDA_MAX_GRID_YZ)
    )


def _optional_grid_1d_fits(grid: tuple[int]) -> bool:
    return 0 <= grid[0] <= CUDA_MAX_GRID_X


def _optional_grid_2d_fits(grid: tuple[int, int]) -> bool:
    if grid[0] == 0:
        return grid[1] >= 0
    return (grid[0] <= CUDA_MAX_GRID_X) & (0 < grid[1]) & (grid[1] <= CUDA_MAX_GRID_YZ)
