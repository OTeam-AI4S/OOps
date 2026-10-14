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

from functools import lru_cache

import torch
from torch import Tensor

from oops.runtime import STANDARD_FLOAT_DTYPES, cuda_device_context, is_compiling, is_external_fake

from . import kernel
from .profiles import SPHERICAL_HARMONICS_SCHEDULE
from .schedule import SphericalHarmonicsPlan, SphericalHarmonicsProblem

_SPHERICAL_HARMONICS_NORMALIZATIONS = {"integral": 0, "component": 1, "norm": 2}


def spherical_harmonics_backward(
    grad_out: Tensor,
    vectors: Tensor,
    lmax: int,
    normalize: bool,
    normalization: str,
    eps: float,
    plan: SphericalHarmonicsPlan,
) -> Tensor:
    r"""Return the vector VJP for the supplied harmonics plan.

    vectors is contiguous CUDA storage shaped (..., 3), with lmax in 0..2.
    grad_out has shape (..., (lmax + 1)**2) and activation or float32 dtype.
    The plan describes these vectors. The result has the vectors' shape and dtype;
    inputs are not mutated.
    """
    with cuda_device_context(vectors.device):
        vectors = vectors.resolve_neg().contiguous()
        grad_out = grad_out.resolve_neg()
        grad_out = grad_out.contiguous()
        num_vectors = int(vectors.numel() // 3)
        grad_vectors = torch.empty_like(vectors)
        kernel._spherical_harmonics_backward[plan.grid](
            vectors,
            grad_out,
            grad_vectors,
            num_vectors,
            lmax,
            normalize,
            _SPHERICAL_HARMONICS_NORMALIZATIONS[normalization],
            eps,
            plan.block_vectors,
            num_warps=plan.num_warps,
        )
        return grad_vectors


def supports_spherical_harmonics(
    vectors: Tensor,
    lmax: int,
    *,
    normalization: str,
) -> bool:

    supported = (
        vectors.is_cuda
        and vectors.is_floating_point()
        and vectors.dtype in STANDARD_FLOAT_DTYPES
        and vectors.dim() >= 2
        and vectors.shape[-1] == 3
        and vectors.is_contiguous()
        and 0 <= lmax <= 2
        and normalization in _SPHERICAL_HARMONICS_NORMALIZATIONS
        and vectors.numel() > 0
    )
    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(vectors)
    return supported and SPHERICAL_HARMONICS_SCHEDULE.supports(
        SphericalHarmonicsProblem(vectors.numel() // 3, lmax, vectors.dtype),
        cache=not external_fake,
    )


def prepare_spherical_harmonics(
    vectors: Tensor,
    lmax: int,
    *,
    normalization: str,
) -> SphericalHarmonicsPlan | None:

    if not (
        vectors.is_cuda
        and vectors.dtype in STANDARD_FLOAT_DTYPES
        and vectors.dim() >= 2
        and vectors.shape[-1] == 3
        and vectors.is_contiguous()
        and vectors.numel() > 0
        and 0 <= lmax <= 2
        and normalization in _SPHERICAL_HARMONICS_NORMALIZATIONS
    ):
        return None
    return _prepared_spherical_harmonics_plan(int(vectors.numel() // 3), int(lmax), vectors.dtype)


@lru_cache(maxsize=1024)
def _prepared_spherical_harmonics_plan(
    num_vectors: int,
    lmax: int,
    dtype: torch.dtype,
) -> SphericalHarmonicsPlan:

    return SPHERICAL_HARMONICS_SCHEDULE.resolve(SphericalHarmonicsProblem(num_vectors, lmax, dtype))


def _spherical_harmonics_forward_impl(
    vectors: Tensor,
    lmax: int,
    normalize: bool,
    normalization: str,
    eps: float,
    *,
    plan: SphericalHarmonicsPlan | None = None,
) -> Tensor:
    with cuda_device_context(vectors.device):
        vectors = vectors.resolve_neg().contiguous()
        num_vectors = int(vectors.numel() // 3)
        if plan is None:
            plan = SPHERICAL_HARMONICS_SCHEDULE.resolve(SphericalHarmonicsProblem(num_vectors, lmax, vectors.dtype))
        out = torch.empty((*vectors.shape[:-1], plan.coeffs), device=vectors.device, dtype=vectors.dtype)
        kernel._spherical_harmonics_forward[plan.grid](
            vectors,
            out,
            num_vectors,
            lmax,
            normalize,
            _SPHERICAL_HARMONICS_NORMALIZATIONS[normalization],
            eps,
            plan.block_vectors,
            num_warps=plan.num_warps,
        )
        return out


def _spherical_harmonics_backward_impl(
    grad_out: Tensor,
    vectors: Tensor,
    lmax: int,
    normalize: bool,
    normalization: str,
    eps: float,
) -> Tensor:
    problem = SphericalHarmonicsProblem(vectors.numel() // 3, lmax, vectors.dtype)
    plan = SPHERICAL_HARMONICS_SCHEDULE.resolve(problem)
    return spherical_harmonics_backward(
        grad_out,
        vectors,
        lmax,
        normalize,
        normalization,
        eps,
        plan,
    )
