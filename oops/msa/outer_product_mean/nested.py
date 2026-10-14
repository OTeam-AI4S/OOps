from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import (
    FLOAT32_DOT_TF32X3,
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    NativeMatmulDomain,
    captured_native_matmul_precision_state,
    is_compiling,
    native_matmul_domain,
    require_native_matmul_precision_state,
    require_no_forward_mode,
    require_same_device,
    triton_dot_precision,
    unsupported,
)

from .packed import segmented_outer_product_mean
from .triton.ops import supports_segmented_outer_product_mean


def _require_sizes(name: str, sizes: Tensor, rank: int) -> None:
    if sizes.device.type != "cpu" or sizes.dtype is not torch.int64 or sizes.dim() != 2:
        raise ValueError(f"outer_product_mean {name} element_sizes must be a two-dimensional CPU int64 tensor")
    if sizes.shape[1] != rank:
        raise ValueError(
            f"outer_product_mean {name} elements must have rank {rank}; "
            f"got element_sizes shape {tuple(sizes.shape)}"
        )
    if bool(torch.any(sizes < 0)):
        raise ValueError(f"outer_product_mean {name} element sizes must be nonnegative")


_NESTED_OUTER_PRODUCT_MEAN_METADATA_SCHEMA = "(Tensor msa_sizes, Tensor? mask_sizes) -> Tensor"


@torch.library.custom_op(
    "ooperators::_validate_nested_outer_product_mean_metadata",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_OUTER_PRODUCT_MEAN_METADATA_SCHEMA,
)
def validate_nested_outer_product_mean_metadata(
    msa_sizes: Tensor,
    mask_sizes: Tensor | None,
) -> Tensor:

    _require_sizes("msa", msa_sizes, 3)
    if msa_sizes.shape[0] == 0:
        raise ValueError("outer_product_mean does not serve an empty NestedTensor wrapper batch")
    if bool(torch.any(msa_sizes[:, 2] <= 0)):
        raise ValueError("outer_product_mean requires the msa channel dimension to be positive")
    if not bool(torch.all(msa_sizes[:, 2] == msa_sizes[0, 2])):
        raise ValueError("outer_product_mean requires one static msa channel size across nested samples")
    if mask_sizes is not None:
        _require_sizes("msa_mask", mask_sizes, 2)
        if not torch.equal(mask_sizes, msa_sizes[:, :2]):
            raise ValueError("outer_product_mean nested msa_mask must match M and N for every sample")

    return torch.ones((), dtype=torch.bool)


@torch.library.register_fake(validate_nested_outer_product_mean_metadata)
def _validate_nested_outer_product_mean_metadata_fake(
    msa_sizes: Tensor,
    mask_sizes: Tensor | None,
) -> Tensor:
    del mask_sizes
    return torch.empty((), device=msa_sizes.device, dtype=torch.bool)


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
    if msa.dim() != 4:
        raise ValueError(
            "outer_product_mean nested msa elements must have shape (M_i, N_i, msa_size); "
            f"got logical rank {msa.dim()}"
        )
    if msa.ragged_dims != (0, 1) or msa.packed_dim_order != (0, 1, 2):
        raise unsupported(
            "outer_product_mean",
            "requires explicit canonical (M_i, N_i, msa_size) nested msa packing",
            ragged_dims=msa.ragged_dims,
            packed_dim_order=msa.packed_dim_order,
        )
    msa_values = msa.concat
    if msa_values.dim() != 2 or not msa_values.is_contiguous():
        raise unsupported(
            "outer_product_mean",
            "requires contiguous packed msa values with shape (sum M_i*N_i, msa_size)",
        )

    if msa_mask is None:
        mask_values = msa_values.new_ones((msa_values.shape[0],))
        mask_sizes = None
    else:
        if msa_mask.dim() != 3:
            raise ValueError("outer_product_mean nested msa_mask elements must have shape (M_i, N_i)")
        if msa_mask.ragged_dims != (0, 1) or msa_mask.packed_dim_order != (0, 1):
            raise unsupported(
                "outer_product_mean",
                "requires explicit canonical (M_i, N_i) nested msa_mask packing",
                ragged_dims=msa_mask.ragged_dims,
                packed_dim_order=msa_mask.packed_dim_order,
            )
        mask_values = msa_mask.concat
        if mask_values.dim() != 1 or not mask_values.is_contiguous():
            raise unsupported("outer_product_mean", "requires contiguous flattened nested msa_mask values")
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
    _check_weight_shapes(msa_values.shape[1], weights, left_bias, right_bias, output_bias)
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

    from .composed import _packed_msa_dual_projection

    projected = _packed_msa_dual_projection(
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
    left, right = projected
    projection_mask = mask_values.to(dtype=projection_dtype).unsqueeze(-1)
    left = left * projection_mask
    right = right * projection_mask
    if outer_dtype is not None:
        left = left.to(outer_dtype)
        right = right.to(outer_dtype)

    device = msa_values.device
    msa_outer = msa.ragged_level_offsets(0, device=device, dtype=torch.int64)
    msa_packed = msa.packed_offsets(device=device, dtype=torch.int64)
    lengths = msa_sizes[:, 1]
    pair_offsets_cpu = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.square().cumsum(0)))
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
        msa_sizes,
        input_precision,
        dual_precision,
        count_precision,
        native_matmul_state,
        divide_before_projection,
        float(denominator_eps),
        clamp_denominator,
    )
    return msa.packed_with_square_lengths(output_values, lengths)


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
