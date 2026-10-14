from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from oops.core._math import _ceil_div
from oops.runtime import CUDA_MAX_GRID_X

from ...spherical_harmonics.triton.schedule import (
    SphericalHarmonicsConfig,
    SphericalHarmonicsPlan,
    SphericalHarmonicsProblem,
    _make_spherical_harmonics_plan,
)


@dataclass(frozen=True, slots=True)
class SE3BasisProblem:
    edges: int
    max_degree: int
    use_pad_trick: bool
    fully_fused: bool
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.edges <= 0:
            raise ValueError("SE(3) basis edges must be positive")
        if self.max_degree != 1:
            raise ValueError("SE(3) basis max_degree must be 1")


@dataclass(frozen=True, slots=True)
class SE3BasisConfig:
    block_edges: int = 16
    grad_block_edges: int = 32
    num_warps: int = 4
    spherical_harmonics: SphericalHarmonicsConfig = SphericalHarmonicsConfig()

    def __post_init__(self) -> None:
        for name, value in (
            ("block_edges", self.block_edges),
            ("grad_block_edges", self.grad_block_edges),
        ):
            if value <= 0 or value & (value - 1):
                raise ValueError(f"SE(3) basis {name} must be a positive power of two")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("SE(3) basis num_warps must be one of 1, 2, 4, or 8")


@dataclass(frozen=True, slots=True)
class SE3BasisOutputSpec:
    key: str
    shape: tuple[int, int, int]
    per_edge: int
    kind: int


@dataclass(frozen=True, slots=True)
class SE3BasisPlan:
    problem: SE3BasisProblem
    outputs: tuple[SE3BasisOutputSpec, ...]
    keys: tuple[str, ...]
    fill_grid: tuple[int]
    grad_harmonics_grid: tuple[int]
    spherical_plan: SphericalHarmonicsPlan
    config: SE3BasisConfig

    def grid_fits(self) -> bool:
        supported = self.spherical_plan.grid_fits()
        supported = supported & (0 < self.fill_grid[0]) & (self.fill_grid[0] <= CUDA_MAX_GRID_X)
        return supported & (0 < self.grad_harmonics_grid[0]) & (self.grad_harmonics_grid[0] <= CUDA_MAX_GRID_X)


@dataclass(frozen=True, slots=True)
class SE3BasisSchedule:
    config: SE3BasisConfig

    def resolve(self, problem: SE3BasisProblem, *, cache: bool = True) -> SE3BasisPlan:
        if not cache or torch.compiler.is_compiling():
            return _resolve_se3_basis(problem, self.config)
        return _cached_se3_basis(problem, self.config)

    def supports(self, problem: SE3BasisProblem, *, cache: bool = True) -> bool:
        if not cache or torch.compiler.is_compiling():
            return _supports_se3_basis(problem, self.config)
        return _cached_supports_se3_basis(problem, self.config)


def _resolve_se3_basis(problem: SE3BasisProblem, config: SE3BasisConfig) -> SE3BasisPlan:
    plan = _make_se3_basis_plan(problem, config)
    if not plan.grid_fits():
        raise ValueError("SE(3) basis has no launchable plan")
    return plan


@lru_cache(maxsize=1024)
def _cached_se3_basis(problem: SE3BasisProblem, config: SE3BasisConfig) -> SE3BasisPlan:
    return _resolve_se3_basis(problem, config)


def _supports_se3_basis(problem: SE3BasisProblem, config: SE3BasisConfig) -> bool:
    return _make_se3_basis_plan(problem, config).grid_fits()


@lru_cache(maxsize=1024)
def _cached_supports_se3_basis(problem: SE3BasisProblem, config: SE3BasisConfig) -> bool:
    return _supports_se3_basis(problem, config)


# (key, per-edge shape, pad-final-axis, placement kind).
_SE3_BASIS_L1_LAYOUT = (
    ("0,0", (1, 1, 1), True, 0),
    ("0,1", (1, 1, 3), True, 1),
    ("1,0", (3, 1, 1), True, 2),
    ("1,1", (3, 3, 3), True, 3),
    ("out0_fused", (4, 2, 1), True, 4),
    ("out1_fused", (4, 4, 3), True, 5),
    ("in0_fused", (1, 2, 4), False, 6),
    ("in1_fused", (3, 4, 4), False, 7),
    ("fully_fused", (4, 6, 4), False, 8),
)


def _make_se3_basis_plan(problem: SE3BasisProblem, config: SE3BasisConfig) -> SE3BasisPlan:
    rows = _SE3_BASIS_L1_LAYOUT if problem.fully_fused else _SE3_BASIS_L1_LAYOUT[:-1]
    outputs = []
    for key, (outer, middle, columns), padded, kind in rows:
        shape = (outer, middle, columns + int(padded and problem.use_pad_trick))
        per_edge = shape[0] * shape[1] * shape[2]
        outputs.append(
            SE3BasisOutputSpec(
                key=key,
                shape=shape,
                per_edge=per_edge,
                kind=kind,
            )
        )
    spherical_plan = _make_spherical_harmonics_plan(
        SphericalHarmonicsProblem(problem.edges, 2, problem.dtype),
        config.spherical_harmonics,
    )
    output_specs = tuple(outputs)
    return SE3BasisPlan(
        problem=problem,
        outputs=output_specs,
        keys=tuple(spec.key for spec in output_specs),
        fill_grid=(_ceil_div(problem.edges, config.block_edges),),
        grad_harmonics_grid=(_ceil_div(problem.edges, config.grad_block_edges),),
        spherical_plan=spherical_plan,
        config=config,
    )
