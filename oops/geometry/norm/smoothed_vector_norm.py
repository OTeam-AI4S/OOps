from __future__ import annotations

import math
import sys
from numbers import Real

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import is_compiling, require_no_forward_mode, unsupported

from .triton import vector_norm


def smoothed_vector_norm(
    vectors: Tensor | NestedTensor,
    *,
    eps: float = 1e-8,
    keepdim: bool = False,
) -> Tensor | NestedTensor:
    r"""Compute ``sqrt(sum(vectors**2, dim=-1) + eps)``.

    Args:
        vectors: CPU or CUDA float16, bfloat16, float32, or float64 vectors.
            Dense inputs may be strided. Nested inputs require a static final
            element dimension and at least one ragged axis.
        eps: Finite non-negative real scalar added to the squared norm.
        keepdim: Retain the final vector dimension with length one.

    Returns:
        Norms with the input dtype, device, and leading dimensions. Nested
        inputs retain their ragged topology.

    Raises:
        TypeError: ``eps`` is not a real scalar or ``keepdim`` is not boolean.
        ValueError: ``eps`` is negative or non-finite, or vectors have no axis.
        OOpsUnsupportedError: The dtype, device, layout, or forward-mode
            or higher-order differentiation is unsupported.

    Notes:
        The norm and coupled first-order VJP ``grad * vectors / norm`` retain
        representable finite results when separate squares, products, or ratios
        would overflow or underflow.
        Output and gradient storage casts follow the input dtype. Zero vectors
        with ``eps=0`` have zero gradient. Empty dimensions are supported.
        Autocast preserves the input dtype. Full-graph compilation is supported.

    Examples:
        >>> import torch
        >>> from oops.geometry.norm import smoothed_vector_norm
        >>> smoothed_vector_norm(torch.tensor([[3.0, 4.0]]), eps=0)
        tensor([5.])
    """

    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError("smoothed_vector_norm eps must be a real scalar")
    eps = float(eps)
    if is_compiling():
        torch._check(eps >= 0)
        torch._check(eps <= sys.float_info.max)
    elif not math.isfinite(eps) or eps < 0:
        raise ValueError("smoothed_vector_norm eps must be finite and non-negative")
    if type(keepdim) is not bool:
        raise TypeError("smoothed_vector_norm keepdim must be a bool")
    if isinstance(vectors, NestedTensor):
        element_rank = vectors.dim() - 1
        if element_rank < 1 or not vectors.ragged_dims or element_rank - 1 in vectors.ragged_dims:
            raise unsupported("smoothed_vector_norm", "requires a static final element dimension")
        order = vectors.packed_dim_order
        if order == tuple(range(element_rank)):
            output = _dense_norm(vectors.concat, eps, keepdim)
            return vectors.packed_with_static_tail(output)
        batch_dim = 0 if vectors.batch_first else 1
        element_dims = tuple(dim for dim in range(vectors.dim()) if dim != batch_dim)
        permutation = list(range(vectors.dim()))
        for packed_dim, element_dim in enumerate(order):
            permutation[element_dims[packed_dim]] = element_dims[element_dim]
        reference = torch.permute(vectors, tuple(permutation))
        vector_dim = order.index(element_rank - 1) - len(vectors.ragged_dims) + 1
        output = _dense_norm(reference.concat.movedim(vector_dim, -1), eps, True).movedim(-1, vector_dim)
        result = reference.packed_with_static_tail(output)
        result = torch.permute(result, tuple(permutation.index(dim) for dim in range(vectors.dim())))
        return result if keepdim else torch.squeeze(result, -1)
    return _dense_norm(vectors, eps, keepdim)


def _dense_norm(vectors: Tensor, eps: float, keepdim: bool) -> Tensor:
    if not isinstance(vectors, Tensor):
        raise TypeError("smoothed_vector_norm vectors must be a Tensor or NestedTensor")
    if vectors.dim() == 0:
        raise ValueError("smoothed_vector_norm vectors must have at least one axis")
    if (
        vectors.device.type not in ("cpu", "cuda")
        or vectors.layout != torch.strided
        or vectors.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    ):
        raise unsupported("smoothed_vector_norm", "requires strided CPU/CUDA floating-point vectors")
    require_no_forward_mode("smoothed_vector_norm", vectors)
    output = vector_norm(vectors, eps)
    return output.unsqueeze(-1) if keepdim else output
