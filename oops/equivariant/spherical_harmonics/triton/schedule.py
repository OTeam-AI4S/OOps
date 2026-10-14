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

from dataclasses import dataclass
from functools import lru_cache

import torch

from oops.core._math import _ceil_div
from oops.runtime import CUDA_MAX_GRID_X


@dataclass(frozen=True, slots=True)
class SphericalHarmonicsProblem:
    num_vectors: int
    lmax: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.num_vectors <= 0:
            raise ValueError("spherical harmonics elements must be positive")
        if not 0 <= self.lmax <= 2:
            raise ValueError("spherical harmonics lmax must be in [0, 2]")


@dataclass(frozen=True, slots=True)
class SphericalHarmonicsConfig:
    block_vectors: int = 256
    num_warps: int = 4

    def __post_init__(self) -> None:
        if self.block_vectors <= 0 or self.block_vectors & (self.block_vectors - 1):
            raise ValueError("spherical harmonics block_elements must be a positive power of two")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("spherical harmonics num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class SphericalHarmonicsPlan:
    grid: tuple[int]
    coeffs: int
    block_vectors: int
    num_warps: int

    def grid_fits(self) -> bool:
        return (0 < self.grid[0]) & (self.grid[0] <= CUDA_MAX_GRID_X)


@dataclass(frozen=True, slots=True)
class SphericalHarmonicsSchedule:
    config: SphericalHarmonicsConfig

    def resolve(self, problem: SphericalHarmonicsProblem, *, cache: bool = True) -> SphericalHarmonicsPlan:
        if not cache or torch.compiler.is_compiling():
            return _resolve_spherical_harmonics(problem, self.config)
        return _cached_spherical_harmonics(problem, self.config)

    def supports(self, problem: SphericalHarmonicsProblem, *, cache: bool = True) -> bool:
        if not cache or torch.compiler.is_compiling():
            return _supports_spherical_harmonics(problem, self.config)
        return _cached_supports_spherical_harmonics(problem, self.config)


def _resolve_spherical_harmonics(
    problem: SphericalHarmonicsProblem,
    config: SphericalHarmonicsConfig,
) -> SphericalHarmonicsPlan:
    plan = _make_spherical_harmonics_plan(problem, config)
    if not plan.grid_fits():
        raise ValueError("spherical harmonics has no launchable plan")
    return plan


@lru_cache(maxsize=1024)
def _cached_spherical_harmonics(
    problem: SphericalHarmonicsProblem,
    config: SphericalHarmonicsConfig,
) -> SphericalHarmonicsPlan:
    return _resolve_spherical_harmonics(problem, config)


def _supports_spherical_harmonics(
    problem: SphericalHarmonicsProblem,
    config: SphericalHarmonicsConfig,
) -> bool:
    return problem.num_vectors <= CUDA_MAX_GRID_X * config.block_vectors


@lru_cache(maxsize=1024)
def _cached_supports_spherical_harmonics(
    problem: SphericalHarmonicsProblem,
    config: SphericalHarmonicsConfig,
) -> bool:
    return _supports_spherical_harmonics(problem, config)


def _make_spherical_harmonics_plan(
    problem: SphericalHarmonicsProblem,
    config: SphericalHarmonicsConfig,
) -> SphericalHarmonicsPlan:
    return SphericalHarmonicsPlan(
        grid=(_ceil_div(problem.num_vectors, config.block_vectors),),
        coeffs=(problem.lmax + 1) * (problem.lmax + 1),
        block_vectors=config.block_vectors,
        num_warps=config.num_warps,
    )
