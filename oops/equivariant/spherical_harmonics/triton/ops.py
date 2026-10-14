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

from oops.runtime import is_compiling, is_external_fake, needs_grad

from .generic import _spherical_harmonics_backward_impl, _spherical_harmonics_forward_impl
from .schedule import SphericalHarmonicsPlan


def triton_spherical_harmonics(
    vectors: Tensor,
    lmax: int,
    *,
    normalize: bool,
    normalization: str,
    eps: float,
    plan: SphericalHarmonicsPlan | None = None,
) -> Tensor:

    if plan is None:
        compiling = is_compiling()
        external_fake = False if compiling else is_external_fake(vectors)
    else:
        compiling = external_fake = False
    if needs_grad(vectors) or compiling or external_fake:
        return _spherical_harmonics_forward(vectors, int(lmax), bool(normalize), normalization, float(eps))
    return _spherical_harmonics_forward_impl(
        vectors,
        int(lmax),
        bool(normalize),
        normalization,
        float(eps),
        plan=plan,
    )


@torch.library.custom_op("ooperators::_spherical_harmonics_forward", mutates_args=(), device_types="cuda")
def _spherical_harmonics_forward(vectors: Tensor, lmax: int, normalize: bool, normalization: str, eps: float) -> Tensor:
    return _spherical_harmonics_forward_impl(vectors, lmax, normalize, normalization, eps)


@torch.library.register_fake(_spherical_harmonics_forward)
def _spherical_harmonics_forward_fake(
    vectors: Tensor,
    lmax: int,
    normalize: bool,
    normalization: str,
    eps: float,
) -> Tensor:
    del normalize, normalization, eps
    return torch.empty(
        (*vectors.shape[:-1], (lmax + 1) * (lmax + 1)),
        device=vectors.device,
        dtype=vectors.dtype,
    )


@torch.library.custom_op("ooperators::_spherical_harmonics_backward", mutates_args=(), device_types="cuda")
def _spherical_harmonics_backward(
    grad_out: Tensor,
    vectors: Tensor,
    lmax: int,
    normalize: bool,
    normalization: str,
    eps: float,
) -> Tensor:
    return _spherical_harmonics_backward_impl(grad_out, vectors, lmax, normalize, normalization, eps)


@torch.library.register_fake(_spherical_harmonics_backward)
def _spherical_harmonics_backward_fake(
    grad_out: Tensor,
    vectors: Tensor,
    lmax: int,
    normalize: bool,
    normalization: str,
    eps: float,
) -> Tensor:
    del grad_out, lmax, normalize, normalization, eps
    return torch.empty_like(vectors)


def _spherical_harmonics_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, int, bool, str, float],
    output: Tensor,
) -> None:
    vectors, lmax, normalize, normalization, eps = inputs
    del output
    ctx.save_for_backward(vectors)
    ctx.lmax = lmax
    ctx.normalize = normalize
    ctx.normalization = normalization
    ctx.eps = eps


def _spherical_harmonics_forward_autograd_backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None, None]:
    (vectors,) = ctx.saved_tensors
    grad_vectors = _spherical_harmonics_backward(
        grad_out,
        vectors,
        ctx.lmax,
        ctx.normalize,
        ctx.normalization,
        ctx.eps,
    )
    return grad_vectors, None, None, None, None


torch.library.register_autograd(
    _spherical_harmonics_forward,
    _spherical_harmonics_forward_autograd_backward,
    setup_context=_spherical_harmonics_forward_setup_context,
)
