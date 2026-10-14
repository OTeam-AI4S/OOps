from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

from oops.runtime import STANDARD_FLOAT_DTYPES, cuda_device_context, is_compiling, is_external_fake, needs_grad

from . import kernel
from .backend import _SPHERICAL_HARMONICS_NORMALIZATIONS, spherical_harmonics_backward
from .profiles import SPHERICAL_HARMONICS_SCHEDULE
from .schedule import SphericalHarmonicsPlan, SphericalHarmonicsProblem


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
