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
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import HALF_DTYPES, STANDARD_FLOAT_DTYPES, require_no_forward_mode, require_same_device, unsupported

from .composed import _dispatch_outer_product_mean


def outer_product_mean(
    msa: Tensor | NestedTensor,
    msa_mask: Tensor | NestedTensor | None = None,
    *,
    msa_norm_weight: Tensor,
    msa_norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    output_weight: Tensor,
    left_bias: Tensor | None = None,
    right_bias: Tensor | None = None,
    output_bias: Tensor | None = None,
    msa_norm_eps: float = 1e-5,
    denominator_eps: float = 1e-3,
    divide_before_projection: bool = True,
    clamp_denominator: bool = False,
    outer_dtype: torch.dtype | None = None,
) -> Tensor | NestedTensor:
    r"""Apply an outer-product mean update to an MSA representation.

    MSA sequences are reduced to square pair features, with configurable projection
    bias and denominator conventions for AF2/AF3-style blocks.

    Args:
        msa: Dense ``(B, S, N, C)`` features or canonical DanLing ``NestedTensor``
            elements ``(S_i, N_i, C)`` with both ragged axes declared explicitly.
        msa_mask: Optional multiplicative mask ``(B, S, N)`` or matching nested
            elements. ``None`` uses unit weights; floating masks are differentiable.
        msa_norm_weight: Contiguous normalization scale ``(C,)`` on the MSA device.
        msa_norm_bias: Contiguous normalization bias ``(C,)`` on the MSA device.
        left_weight: Left projection matrix ``(H, C)``.
        right_weight: Right projection matrix ``(H, C)``.
        output_weight: Pair projection matrix ``(P, H * H)``.
        left_bias: Optional left projection bias ``(H,)``.
        right_bias: Optional right projection bias ``(H,)``.
        output_bias: Optional pair projection bias ``(P,)``.
        msa_norm_eps: Epsilon for MSA normalization.
        denominator_eps: Added to the pair count for post-projection division.
        divide_before_projection: If ``True``, divide the outer product by the
            count clamped to at least one, then apply the output projection. If
            ``False``, apply the projection before denominator division.
        clamp_denominator: For post-projection division, clamp
            ``count + denominator_eps`` to at least one.
        outer_dtype: Dtype of the left/right projection features used for the
            outer-product update. ``None`` uses the projection dtype.

    Returns:
        Pair updates ``(B, N, N, P)`` on the input device in the prepared output
        projection dtype. Nested elements have shape ``(N_i, N_i, P)``. No residual
        addition is applied.

    Raises:
        ValueError: Shapes, devices, representations, or nested topologies disagree.
        OOpsUnsupportedError: Device, dtype, layout, tensor shape or forward-mode differentiation is unsupported.

    Notes:
        Zero MSA depth preserves square pair shapes; zero token dimensions return
        empty pair dimensions. Dividing before or after a biased output projection
        gives different affine semantics. A projection bias can give nonzero output
        for a fully masked pair.

    Examples:
        >>> import torch
        >>> from oops import outer_product_mean
        >>> if torch.cuda.is_available():
        ...     msa = torch.ones(1, 3, 4, 16, device="cuda")
        ...     parameters = dict(
        ...         msa_norm_weight=torch.ones(16, device="cuda"),
        ...         msa_norm_bias=torch.zeros(16, device="cuda"),
        ...         left_weight=torch.ones(4, 16, device="cuda"),
        ...         right_weight=torch.ones(4, 16, device="cuda"),
        ...         output_weight=torch.ones(8, 16, device="cuda"),
        ...     )
        ...     result = outer_product_mean(msa, **parameters)
        ...     assert result.shape == (1, 4, 4, 8)
    """
    dense_exact = type(msa) is Tensor and (msa_mask is None or type(msa_mask) is Tensor)
    if not dense_exact:
        if isinstance(msa, NestedTensor):
            if msa_mask is not None and not isinstance(msa_mask, NestedTensor):
                raise ValueError("outer_product_mean msa and msa_mask must both be NestedTensor or both be dense")
            from .nested import outer_product_mean_nested

            return outer_product_mean_nested(
                msa,
                msa_mask,
                msa_norm_weight=msa_norm_weight,
                msa_norm_bias=msa_norm_bias,
                left_weight=left_weight,
                right_weight=right_weight,
                output_weight=output_weight,
                left_bias=left_bias,
                right_bias=right_bias,
                output_bias=output_bias,
                msa_norm_eps=msa_norm_eps,
                denominator_eps=denominator_eps,
                divide_before_projection=divide_before_projection,
                clamp_denominator=clamp_denominator,
                outer_dtype=outer_dtype,
            )
        if isinstance(msa_mask, NestedTensor):
            raise ValueError("outer_product_mean msa and msa_mask must both be NestedTensor or both be dense")

    weights = (msa_norm_weight, msa_norm_bias, left_weight, right_weight, output_weight)
    projection_biases = (left_bias, right_bias)
    _check_shapes(msa, msa_mask, weights, projection_biases, output_bias)
    require_no_forward_mode("outer_product_mean", msa, msa_mask, *weights, *projection_biases, output_bias)
    operands = (msa, msa_mask, *weights, *projection_biases, output_bias)
    require_same_device("outer_product_mean", msa, *operands[1:])
    if not msa.is_cuda:
        raise unsupported(
            "outer_product_mean",
            "requires CUDA operands",
            device=msa.device,
        )
    if msa.dtype not in STANDARD_FLOAT_DTYPES:
        raise unsupported(
            "outer_product_mean",
            "requires a float16, bfloat16, or float32 msa",
            msa_dtype=msa.dtype,
        )
    (
        left_weight,
        right_weight,
        output_weight,
        left_bias,
        right_bias,
        output_bias,
    ) = _prepare_projection_dtypes(
        msa,
        left_weight,
        right_weight,
        output_weight,
        left_bias,
        right_bias,
        output_bias,
    )
    left_weight = left_weight.contiguous()
    right_weight = right_weight.contiguous()
    output_weight = output_weight.contiguous()
    left_bias = None if left_bias is None else left_bias.contiguous()
    right_bias = None if right_bias is None else right_bias.contiguous()
    output_bias = None if output_bias is None else output_bias.contiguous()
    norm_dtypes = (msa_norm_weight.dtype, msa_norm_bias.dtype)
    if any(dtype not in (msa.dtype, torch.float32) for dtype in norm_dtypes):
        raise unsupported(
            "outer_product_mean",
            "requires each msa_norm affine operand in the msa dtype or float32",
            msa_dtype=msa.dtype,
            norm_dtypes=norm_dtypes,
        )
    return _dispatch_outer_product_mean(
        msa,
        msa_mask,
        msa_norm_weight,
        msa_norm_bias,
        left_weight,
        right_weight,
        output_weight,
        left_bias,
        right_bias,
        output_bias,
        msa_norm_eps=msa_norm_eps,
        denominator_eps=denominator_eps,
        divide_before_projection=divide_before_projection,
        clamp_denominator=clamp_denominator,
        outer_dtype=outer_dtype,
    )


def _check_shapes(
    msa: Tensor,
    msa_mask: Tensor | None,
    weights: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    projection_biases: tuple[Tensor | None, Tensor | None],
    output_bias: Tensor | None,
) -> None:
    if msa.dim() != 4:
        raise ValueError("msa must have shape (B, S, N, msa_size)")
    if msa_mask is not None and tuple(msa_mask.shape) != tuple(msa.shape[:-1]):
        raise ValueError(f"msa_mask shape {tuple(msa_mask.shape)} must match msa.shape[:-1] = {tuple(msa.shape[:-1])}")
    msa_norm_weight, msa_norm_bias, left_weight, right_weight, output_weight = weights
    msa_size = msa.shape[-1]
    hidden_size = left_weight.shape[0]
    if hidden_size <= 0:
        raise ValueError("left_weight output channels must be positive")
    expected = (
        (msa_size,),
        (msa_size,),
        (hidden_size, msa_size),
        (hidden_size, msa_size),
        (output_weight.shape[0], hidden_size * hidden_size),
    )
    if output_weight.shape[0] <= 0:
        raise ValueError("output_weight output channels must be positive")
    names = ("msa_norm_weight", "msa_norm_bias", "left_weight", "right_weight", "output_weight")
    for name, weight, want in zip(names, weights, expected):
        if tuple(weight.shape) != want:
            raise ValueError(f"{name} must have shape {want}; got {tuple(weight.shape)}")
    for name, bias in zip(("left_bias", "right_bias"), projection_biases):
        if bias is not None and tuple(bias.shape) != (hidden_size,):
            raise ValueError(f"{name} must have shape {(hidden_size,)}; got {tuple(bias.shape)}")
    if output_bias is not None and tuple(output_bias.shape) != (output_weight.shape[0],):
        raise ValueError(f"output_bias must have shape {(output_weight.shape[0],)}; got {tuple(output_bias.shape)}")


def _prepare_projection_dtypes(
    msa: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    output_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    output_bias: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    projection_operands = (
        left_weight,
        right_weight,
        output_weight,
        left_bias,
        right_bias,
        output_bias,
    )
    provided = tuple(operand for operand in projection_operands if operand is not None)
    autocast_dtype = _cuda_projection_autocast_dtype(msa)
    if autocast_dtype is None:
        if any(operand.dtype != msa.dtype for operand in provided):
            raise unsupported(
                "outer_product_mean",
                "outside CUDA float16/bfloat16 autocast requires every projection weight and bias in the msa dtype",
                msa_dtype=msa.dtype,
                projection_dtypes=tuple(operand.dtype for operand in provided),
            )
        return projection_operands
    if any(operand.dtype not in STANDARD_FLOAT_DTYPES for operand in provided):
        raise unsupported(
            "outer_product_mean",
            "inside CUDA autocast requires float16/bfloat16/float32 projection weights and biases",
            autocast_dtype=autocast_dtype,
            projection_dtypes=tuple(operand.dtype for operand in provided),
        )

    def cast_weight(weight: Tensor) -> Tensor:
        return weight if weight.dtype == autocast_dtype else weight.to(dtype=autocast_dtype)

    def cast_bias(bias: Tensor | None) -> Tensor | None:
        return None if bias is None else cast_weight(bias)

    return (
        cast_weight(left_weight),
        cast_weight(right_weight),
        cast_weight(output_weight),
        cast_bias(left_bias),
        cast_bias(right_bias),
        cast_bias(output_bias),
    )


def _cuda_projection_autocast_dtype(msa: Tensor) -> torch.dtype | None:
    if not msa.is_cuda or not torch.is_autocast_enabled("cuda"):
        return None
    autocast_dtype = torch.get_autocast_dtype("cuda")
    if msa.dtype not in STANDARD_FLOAT_DTYPES or autocast_dtype not in HALF_DTYPES:
        return None
    return autocast_dtype
