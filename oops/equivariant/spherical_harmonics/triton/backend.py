from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import cuda_device_context

from . import kernel
from .schedule import SphericalHarmonicsPlan

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
