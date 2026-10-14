from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import is_compiling, is_external_fake, require_no_forward_mode, unsupported

from . import triton as _triton_equivariant

_SUPPORTED_NORMALIZATIONS = {"integral", "component", "norm"}


def spherical_harmonics(
    vectors: Tensor,
    lmax: int,
    *,
    normalize: bool = True,
    normalization: str = "integral",
    eps: float = 1e-12,
    return_list: bool = False,
) -> Tensor | list[Tensor]:
    r"""Compute real spherical harmonics for degrees zero through ``lmax``.

    Args:
        vectors: Contiguous, non-empty CUDA float16, bfloat16, or float32 tensor
            shaped ``(..., 3)``, with rank at least two.
        lmax: Maximum degree, from zero through two.
        normalize: Normalize each vector before evaluating the harmonics.
        normalization: ``"integral"``, ``"component"``, or ``"norm"``, following
            e3nn's real spherical-harmonic conventions.
        eps: Positive lower bound on vector length when ``normalize=True``.
        return_list: Return separate tensors for each degree instead of one
            concatenated tensor.

    Returns:
        Harmonics on the input device and in its dtype, shaped
        ``(..., (lmax + 1)**2)``. With ``return_list=True``, returns ``lmax + 1``
        tensors whose final widths are ``2 * degree + 1``.

    Raises:
        ValueError: The final vector width or normalization name is invalid.
        TypeError: The vectors are not floating point.
        OOpsUnsupportedError: Degree, device, dtype, layout, tensor extent, or
            forward-mode differentiation is unsupported.

    Notes:
        Arithmetic uses float32 before rounding to the input dtype; CUDA autocast
        does not change the input or output dtype. The degree blocks follow e3nn's
        real basis order. Relative to integral normalization, component
        normalization multiplies every degree by ``sqrt(4*pi)``; norm normalization
        multiplies degree ``l`` by ``sqrt(4*pi / (2*l + 1))``.

        Normalization evaluates ``v / max(norm(v), eps)``. First-order reverse mode
        differentiates that clamped map, including its ``1 / eps`` branch below the
        clamp. Forward-mode and higher-order differentiation are unsupported.
        CPU and float64 inputs have no fallback.

    Examples:
        >>> import torch
        >>> from oops import spherical_harmonics
        >>> if torch.cuda.is_available():
        ...     vectors = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
        ...     harmonics = spherical_harmonics(vectors, 1)
        ...     assert harmonics.shape == (1, 4)
        ...     by_degree = spherical_harmonics(vectors, 1, return_list=True)
        ...     assert [value.shape[-1] for value in by_degree] == [1, 3]
    """

    require_no_forward_mode("spherical_harmonics", vectors)
    if vectors.shape[-1:] != (3,):
        raise ValueError(f"vectors must have shape (..., 3), got {tuple(vectors.shape)}")
    if lmax < 0 or lmax > 2:
        raise unsupported("spherical_harmonics", "serves 0 <= lmax <= 2", lmax=lmax)
    if normalization not in _SUPPORTED_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {sorted(_SUPPORTED_NORMALIZATIONS)}, got {normalization!r}")
    if not vectors.is_floating_point():
        raise TypeError(f"vectors must be floating point, got {vectors.dtype}")

    plan = None
    compiling = is_compiling()
    vector_type = type(vectors)
    external_fake = (
        not compiling
        and vector_type is not Tensor
        and vector_type is not torch.nn.Parameter
        and is_external_fake(vectors)
    )
    if compiling or external_fake:
        supported = _triton_equivariant.supports_spherical_harmonics(vectors, lmax, normalization=normalization)
    else:
        plan = _triton_equivariant.prepare_spherical_harmonics(vectors, lmax, normalization=normalization)
        supported = plan is not None
    if not supported:
        raise unsupported(
            "spherical_harmonics",
            "requires a contiguous non-empty CUDA float16/bfloat16/float32 (..., 3) tensor "
            "of rank at least 2 whose launch grid fits CUDA limits",
            dtype=vectors.dtype,
            device=vectors.device,
            shape=tuple(vectors.shape),
            contiguous=vectors.is_contiguous(),
        )

    out = _triton_equivariant.triton_spherical_harmonics(
        vectors,
        lmax,
        normalize=normalize,
        normalization=normalization,
        eps=eps,
        plan=plan,
    )
    if return_list:
        return [out[..., degree * degree : (degree + 1) * (degree + 1)] for degree in range(lmax + 1)]
    return out
