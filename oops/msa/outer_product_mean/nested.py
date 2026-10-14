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

from math import prod

import torch
from danling.tensors import NestedTensor
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true

from oops.runtime import (
    FLOAT32_DOT_TF32X3,
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    NativeMatmulDomain,
    captured_native_matmul_precision_state,
    empty_with_dependencies,
    is_compiling,
    native_matmul_domain,
    require_native_matmul_precision_state,
    require_no_forward_mode,
    require_same_device,
    require_shape,
    triton_dot_precision,
    unsupported,
)

from .layout import (
    _empty_nested_pair_values,
    canonical_packed_values,
    n_only_storage,
    nested_pair_output,
    validate_nested_outer_product_mean_metadata,
)
from .packed import segmented_outer_product_mean
from .triton.generic import supports_segmented_outer_product_mean


def outer_product_mean_nested(
    msa: NestedTensor,
    msa_mask: NestedTensor | None,
    *,
    msa_norm_weight: Tensor,
    msa_norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    output_weight: Tensor,
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    output_bias: Tensor | None,
    msa_norm_eps: float,
    denominator_eps: float,
    divide_before_projection: bool,
    clamp_denominator: bool,
    outer_dtype: torch.dtype | None,
) -> NestedTensor:

    if not msa.batch_first or (msa_mask is not None and not msa_mask.batch_first):
        raise ValueError("outer_product_mean NestedTensor batch must be the leading logical dimension")
    if len(msa) == 0:
        raise ValueError("outer_product_mean does not serve an empty NestedTensor wrapper batch")
    prefix_rank = msa.dim() - 4
    if prefix_rank < 0:
        raise ValueError(
            "outer_product_mean nested msa elements must have shape (*prefix, M_i, N_i, msa_size); "
            f"got logical rank {msa.dim()}"
        )
    n_only = n_only_storage(msa, prefix_rank, True)
    msa_values = msa.concat
    if msa_values.dim() != prefix_rank + 2 + int(n_only) or msa_values.stride(-1) != 1:
        raise unsupported(
            "outer_product_mean",
            "requires channel-contiguous packed msa values with shape (sum M_i*N_i, *prefix, msa_size)",
        )

    if msa_mask is None:
        mask_values = msa_values.new_ones(msa_values.shape[:-1])
        mask_sizes = None
        mask_n_only = n_only
    else:
        if msa_mask.dim() != msa.dim() - 1:
            raise ValueError("outer_product_mean nested msa_mask elements must have shape (*prefix, M_i, N_i)")
        mask_n_only = n_only_storage(msa_mask, prefix_rank, False)
        mask_values = msa_mask.concat
        if mask_values.dim() != prefix_rank + 1 + int(mask_n_only):
            raise unsupported("outer_product_mean", "nested msa_mask must retain the complete packed static prefix")
        mask_prefix = mask_values.shape[1:-1] if mask_n_only else mask_values.shape[1:]
        msa_prefix = msa_values.shape[1:-2] if n_only else msa_values.shape[1:-1]
        for mask_size, msa_size in zip(mask_prefix, msa_prefix, strict=True):
            require_shape(mask_size == msa_size, "outer_product_mean packed msa_mask shape must match msa")
        mask_cells = mask_values.shape[0] * (mask_values.shape[-1] if mask_n_only else 1)
        msa_cells = msa_values.shape[0] * (msa_values.shape[-2] if n_only else 1)
        require_shape(mask_cells == msa_cells, "outer_product_mean packed msa_mask must cover every msa cell")
        mask_sizes = msa_mask.element_sizes()

    msa_sizes = msa.element_sizes()
    validation = validate_nested_outer_product_mean_metadata(msa_sizes, mask_sizes)
    if is_compiling():
        torch._check_tensor_all(validation)

    weights = (
        msa_norm_weight,
        msa_norm_bias,
        left_weight,
        right_weight,
        output_weight,
    )
    _check_weight_shapes(msa_values.shape[-1], weights, left_bias, right_bias, output_bias)
    require_no_forward_mode(
        "outer_product_mean",
        msa_values,
        mask_values,
        *weights,
        left_bias,
        right_bias,
        output_bias,
    )
    require_same_device(
        "outer_product_mean",
        msa_values,
        mask_values,
        *weights,
        left_bias,
        right_bias,
        output_bias,
    )
    if not msa_values.is_cuda:
        raise unsupported("outer_product_mean", "requires CUDA operands", device=msa_values.device)
    if msa_values.dtype not in STANDARD_FLOAT_DTYPES:
        raise unsupported(
            "outer_product_mean",
            "requires a float16, bfloat16, or float32 msa",
            msa_dtype=msa_values.dtype,
        )

    projection_operands = _prepare_projection_dtypes(
        msa_values,
        left_weight,
        right_weight,
        output_weight,
        left_bias,
        right_bias,
        output_bias,
    )
    left_weight = projection_operands[0].contiguous()
    right_weight = projection_operands[1].contiguous()
    output_weight = projection_operands[2].contiguous()
    left_bias = None if projection_operands[3] is None else projection_operands[3].contiguous()
    right_bias = None if projection_operands[4] is None else projection_operands[4].contiguous()
    output_bias = None if projection_operands[5] is None else projection_operands[5].contiguous()
    if any(dtype not in (msa_values.dtype, torch.float32) for dtype in (msa_norm_weight.dtype, msa_norm_bias.dtype)):
        raise unsupported(
            "outer_product_mean",
            "requires each msa_norm affine operand in the msa dtype or float32",
            msa_dtype=msa_values.dtype,
            norm_dtypes=(msa_norm_weight.dtype, msa_norm_bias.dtype),
        )

    static_prefix = tuple(msa_values.shape[1:-2] if n_only else msa_values.shape[1:-1])
    if is_compiling():
        for axis in range(1, msa_values.dim()):
            torch._dynamo.mark_static(msa_values, axis)
    groups = prod(static_prefix)
    if statically_known_true(groups == 0):
        values = _empty_nested_pair_values(output_weight.detach(), msa_sizes, list(static_prefix))
        values = empty_with_dependencies(values, msa_values, mask_values, *weights, left_bias, right_bias, output_bias)
        return nested_pair_output(msa, values, msa_sizes[:, -2], prefix_rank)

    projection_dtype = left_weight.dtype
    contraction_dtype = projection_dtype if outer_dtype is None else outer_dtype
    domains = int(NativeMatmulDomain.FP32)
    domains |= int(native_matmul_domain(projection_dtype))
    domains |= int(native_matmul_domain(contraction_dtype))
    domains |= int(native_matmul_domain(output_weight.dtype))
    native_matmul_state = captured_native_matmul_precision_state(domains)
    input_precision = triton_dot_precision(contraction_dtype, FLOAT32_DOT_TF32X3, native_matmul_state)
    dual_precision = triton_dot_precision(projection_dtype, FLOAT32_DOT_TF32X3, native_matmul_state)
    count_precision = triton_dot_precision(torch.float32, FLOAT32_DOT_TF32X3, native_matmul_state)
    if not is_compiling():
        require_native_matmul_precision_state("outer_product_mean", native_matmul_state)

    msa_values = canonical_packed_values(msa_values, msa_sizes, prefix_rank, n_only, True)
    mask_values = canonical_packed_values(mask_values, msa_sizes, prefix_rank, mask_n_only, False)

    from .composed import _packed_msa_dual_projection

    left, right = _packed_msa_dual_projection(
        msa_values,
        msa_norm_weight,
        msa_norm_bias,
        left_weight,
        right_weight,
        left_bias,
        right_bias,
        float(msa_norm_eps),
        dual_precision,
        native_matmul_state,
    )
    projection_mask = mask_values.to(dtype=projection_dtype).unsqueeze(-1)
    left = left * projection_mask
    right = right * projection_mask
    if outer_dtype is not None:
        left = left.to(outer_dtype)
        right = right.to(outer_dtype)

    device = msa_values.device
    lengths = msa_sizes[:, -2]
    segment_sizes = msa_sizes[:, -3:]
    if prefix_rank == 0 and not n_only:
        msa_outer = msa.ragged_level_offsets(0, device=device, dtype=torch.int64)
        msa_packed = msa.packed_offsets(device=device, dtype=torch.int64)
    else:
        segment_sizes = segment_sizes.repeat(groups, 1)
        msa_outer = torch.cat((torch.zeros(1, dtype=torch.int64), segment_sizes[:, 0].cumsum(0))).to(device=device)
        msa_cells = segment_sizes[:, 0] * segment_sizes[:, 1]
        msa_packed = torch.cat((torch.zeros(1, dtype=torch.int64), msa_cells.cumsum(0))).to(device=device)
    pair_offsets_cpu = torch.cat((torch.zeros(1, dtype=torch.int64), segment_sizes[:, 1].square().cumsum(0)))
    pair_offsets = pair_offsets_cpu.to(device=device)
    if is_compiling():
        for value in (left, right, output_weight):
            torch._dynamo.mark_static(value, -1)
        torch._dynamo.mark_static(output_weight, 0)

    if not supports_segmented_outer_product_mean(
        left,
        right,
        mask_values,
        output_weight,
        output_bias,
        msa_outer,
        msa_packed,
        pair_offsets,
    ):
        raise unsupported(
            "outer_product_mean",
            "requires canonical packed CUDA segments and launchable segmented phases",
            msa_values_shape=tuple(msa_values.shape),
            hidden=left_weight.shape[0],
            output_channels=output_weight.shape[0],
        )

    output_values = segmented_outer_product_mean(
        left,
        right,
        mask_values,
        output_weight,
        output_bias,
        msa_outer,
        msa_packed,
        pair_offsets,
        pair_offsets_cpu,
        segment_sizes,
        input_precision,
        dual_precision,
        count_precision,
        native_matmul_state,
        divide_before_projection,
        float(denominator_eps),
        clamp_denominator,
    )
    if not statically_known_true(groups == 1):
        require_shape(output_values.shape[0] % groups == 0, "outer_product_mean output must preserve all static groups")
    pair_rows = output_values.shape[0] // groups
    output_values = output_values.reshape(*static_prefix, pair_rows, output_values.shape[-1]).movedim(-2, 0)
    return nested_pair_output(msa, output_values, lengths, prefix_rank)


def _check_weight_shapes(
    channels: int,
    weights: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    left_bias: Tensor | None,
    right_bias: Tensor | None,
    output_bias: Tensor | None,
) -> None:
    norm_weight, norm_bias, left_weight, right_weight, output_weight = weights
    if left_weight.dim() != 2:
        raise ValueError("left_weight must have shape (hidden_size, msa_size)")
    hidden = left_weight.shape[0]
    expected = (
        (channels,),
        (channels,),
        (hidden, channels),
        (hidden, channels),
        (output_weight.shape[0], hidden * hidden),
    )
    for name, value, shape in zip(
        ("msa_norm_weight", "msa_norm_bias", "left_weight", "right_weight", "output_weight"),
        weights,
        expected,
    ):
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if hidden <= 0 or output_weight.shape[0] <= 0:
        raise ValueError("outer_product_mean hidden and output channel dimensions must be positive")
    for name, bias_value in (("left_bias", left_bias), ("right_bias", right_bias)):
        if bias_value is not None and tuple(bias_value.shape) != (hidden,):
            raise ValueError(f"{name} must have shape {(hidden,)}; got {tuple(bias_value.shape)}")
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
    operands = (left_weight, right_weight, output_weight, left_bias, right_bias, output_bias)
    provided = tuple(value for value in operands if value is not None)
    autocast_dtype = None
    if msa.is_cuda and torch.is_autocast_enabled("cuda"):
        candidate = torch.get_autocast_dtype("cuda")
        if msa.dtype in STANDARD_FLOAT_DTYPES and candidate in HALF_DTYPES:
            autocast_dtype = candidate
    if autocast_dtype is None:
        if any(value.dtype != msa.dtype for value in provided):
            raise unsupported(
                "outer_product_mean",
                "outside CUDA float16/bfloat16 autocast requires every projection operand in the msa dtype",
                msa_dtype=msa.dtype,
                projection_dtypes=tuple(value.dtype for value in provided),
            )
        return operands
    if any(value.dtype not in STANDARD_FLOAT_DTYPES for value in provided):
        raise unsupported(
            "outer_product_mean",
            "inside CUDA autocast requires float16/bfloat16/float32 projection operands",
            projection_dtypes=tuple(value.dtype for value in provided),
        )
    return (
        left_weight.to(dtype=autocast_dtype),
        right_weight.to(dtype=autocast_dtype),
        output_weight.to(dtype=autocast_dtype),
        None if left_bias is None else left_bias.to(dtype=autocast_dtype),
        None if right_bias is None else right_bias.to(dtype=autocast_dtype),
        None if output_bias is None else output_bias.to(dtype=autocast_dtype),
    )
