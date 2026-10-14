from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import HALF_DTYPES, require_no_forward_mode, require_same_device, unsupported

from .triton import supports_layer_norm, triton_layer_norm


def layer_norm(
    input: Tensor | NestedTensor,
    normalized_shape: int | Iterable[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor | NestedTensor:
    r"""Normalize the final feature dimension of CUDA activations.

    Args:
        input: CUDA float16, bfloat16, or float32 tensor of shape ``(..., C)``, or a
            nested tensor with a static final feature axis. Dense inputs may be strided;
            nested packed values must be contiguous with features physically last.
        normalized_shape: ``C`` or the one-element sequence ``(C,)``, with ``1 <= C <= 8192``.
        weight: Optional scale of shape ``(C,)`` on the input device. Strided views are accepted.
        bias: Optional bias of shape ``(C,)``, on the input device and in the same dtype
            as ``weight`` when both are supplied.
        eps: Stabilizer added to the variance before computing the reciprocal standard deviation.

    Returns:
        Normalized values with the input shape, storage dtype, device, and nested topology.

    Raises:
        ValueError: Operands use different devices or nested affine parameters are supplied.
        OOpsUnsupportedError: The device, dtype, shape, empty input, or nested layout is
            unsupported, or a forward-mode tangent is supplied.

    Notes:
        Statistics and affine arithmetic accumulate in float32; the result rounds to the
        activation dtype. For a half activation, all supplied affine parameters must use
        either that activation dtype or float32. A float32 activation with half affine
        parameters is unsupported even inside CUDA autocast.
        This functional API serves CUDA inputs only. ``LayerNorm`` supplies the native
        PyTorch fallback for other inputs. First-order reverse-mode gradients are supported;
        forward-mode AD is unsupported.

    Examples:
        >>> import torch
        >>> from oops import layer_norm
        >>> if torch.cuda.is_available():
        ...     x = torch.ones(2, 8, device="cuda")
        ...     y = layer_norm(x, 8)
        ...     assert y.shape == x.shape and y.dtype == x.dtype
    """

    if type(input) is not Tensor and isinstance(input, NestedTensor):
        return _layer_norm_nested(input, normalized_shape, weight, bias, eps)

    require_no_forward_mode("layer_norm", input, weight, bias)
    require_same_device("layer_norm", input, weight, bias)

    shape = _iter_shape(normalized_shape)
    normalized_size = math.prod(shape)
    if len(shape) != 1:
        raise unsupported(
            "layer_norm",
            "normalizes over the final dimension only",
            normalized_shape=shape,
            input_shape=tuple(input.shape),
        )

    if _uses_mixed_autocast(input, weight, bias) and input.dtype not in HALF_DTYPES:
        raise unsupported(
            "layer_norm",
            "does not reimplement autocast promotion for a float32 activation with half affine",
            input_dtype=input.dtype,
            weight_dtype=None if weight is None else weight.dtype,
            bias_dtype=None if bias is None else bias.dtype,
        )

    if not supports_layer_norm(input, normalized_size, weight, bias):
        raise unsupported(
            "layer_norm",
            "requires a CUDA half/float32 activation with a matching contiguous affine row",
            input_dtype=input.dtype,
            device=input.device,
            normalized_size=normalized_size,
            weight_dtype=None if weight is None else weight.dtype,
            bias_dtype=None if bias is None else bias.dtype,
        )

    return triton_layer_norm(input, weight, bias, eps)


def _layer_norm_nested(
    input: NestedTensor,
    normalized_shape: int | Iterable[int],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> NestedTensor:
    if isinstance(weight, NestedTensor) or isinstance(bias, NestedTensor):
        raise ValueError("layer_norm NestedTensor input requires dense affine parameters")
    if input.dim() < 2 or not input.ragged_dims:
        raise unsupported(
            "layer_norm",
            "requires a NestedTensor with explicit packed element topology",
            ragged_dims=input.ragged_dims,
        )
    element_rank = input.dim() - 1
    normalized_dim = element_rank - 1
    if normalized_dim in input.ragged_dims or input.packed_dim_order[-1] != normalized_dim:
        raise unsupported(
            "layer_norm",
            "requires the final logical element dimension to be static and physically last",
            ragged_dims=input.ragged_dims,
            packed_dim_order=input.packed_dim_order,
        )

    packed = input.concat
    if packed.dim() < 1:
        raise unsupported("layer_norm", "requires packed values with a retained static final dimension")
    if not packed.is_contiguous():
        raise unsupported("layer_norm", "requires contiguous NestedTensor packed values")
    output = cast(Tensor, layer_norm(packed, _iter_shape(normalized_shape), weight, bias, eps))
    return input.packed_like(output)


def _iter_shape(normalized_shape: int | Iterable[int]) -> tuple[int, ...]:
    return (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)


def _uses_mixed_autocast(input: Tensor, weight: Tensor | None, bias: Tensor | None) -> bool:
    if not torch.is_autocast_enabled("cuda"):
        return False
    return any(
        tensor is not None and tensor.is_floating_point() and tensor.dtype != input.dtype for tensor in (weight, bias)
    )
