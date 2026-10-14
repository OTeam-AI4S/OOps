from __future__ import annotations

from typing import Literal

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import (
    is_compiling,
    prepare_cuda_autocast_operands,
    require_no_forward_mode,
    require_same_device,
    unsupported,
)

from . import triton as _triton
from .nested import canonicalize_nested_triangle_metadata
from .parameters import TriangleMultiplicationBiases, TriangleMultiplicationWeights

_OPERATOR = "triangle_multiplication"


def triangle_multiplication(
    pair: Tensor | NestedTensor,
    pair_mask: Tensor | NestedTensor | None = None,
    *,
    direction: Literal["incoming", "outgoing"] = "outgoing",
    input_norm_weight: Tensor,
    input_norm_bias: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    left_value_bias: Tensor | None = None,
    left_gate_bias: Tensor | None = None,
    right_value_bias: Tensor | None = None,
    right_gate_bias: Tensor | None = None,
    output_bias: Tensor | None = None,
    output_gate_bias: Tensor | None = None,
    input_norm_eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
) -> Tensor | NestedTensor:
    r"""
    Apply an incoming or outgoing triangle multiplicative update to pair features.

    Pair features have a square token grid; each leading-prefix element is an
    independent execution group. The output gate reads the input-normalized pair.

    Args:
        pair: Dense ``(*prefix, N, N, C)`` pairs or canonical DanLing
            ``NestedTensor`` elements ``(*prefix, N_i, N_i, C)``.
        pair_mask: Optional multiplicative mask broadcastable to
            ``(*prefix, N, N)``, or matching nested elements. Floating masks are
            differentiable; ``None`` uses unit weights.
        direction: ``"outgoing"`` or ``"incoming"``, selecting the contraction
            orientation.
        input_norm_weight: Contiguous input-normalization scale ``(C,)``.
        input_norm_bias: Contiguous input-normalization bias ``(C,)``.
        left_value_weight: Contiguous left value matrix ``(H, C)``.
        left_gate_weight: Contiguous left gate matrix ``(H, C)``.
        right_value_weight: Contiguous right value matrix ``(H, C)``.
        right_gate_weight: Contiguous right gate matrix ``(H, C)``.
        output_norm_weight: Contiguous contraction-normalization scale ``(H,)``.
        output_norm_bias: Contiguous contraction-normalization bias ``(H,)``.
        output_weight: Contiguous output projection ``(C, H)``.
        output_gate_weight: Contiguous output gate projection ``(C, C)``.
        left_value_bias: Optional contiguous left value bias ``(H,)``.
        left_gate_bias: Optional contiguous left gate bias ``(H,)``.
        right_value_bias: Optional contiguous right value bias ``(H,)``.
        right_gate_bias: Optional contiguous right gate bias ``(H,)``.
        output_bias: Optional contiguous output projection bias ``(C,)``.
        output_gate_bias: Optional contiguous output gate bias ``(C,)``.
        input_norm_eps: Epsilon for input normalization.
        output_norm_eps: Epsilon for contraction normalization.

    Returns:
        Pair updates with the input shape, device, and prepared projection dtype.
        Nested results preserve the input topology. No residual addition is applied.

    Raises:
        ValueError: Direction, shapes, devices, representations, or nested
            topologies disagree.
        OOpsUnsupportedError: Device, dtype, layout, channel/launch limits, or
            forward-mode differentiation is unsupported, or captured matmul
            precision no longer matches the execution.

    Notes:
        CUDA float16, bfloat16, and float32 pairs are served with
        ``1 <= C, H <= 1024`` within the normalization/launch domain. All operands
        share a device. Projection weights and biases match the pair dtype outside
        autocast; normalization affine pairs may use FP32. CUDA FP16/BF16 autocast
        casts pairs and projections to its target dtype while retaining supplied
        normalization affine dtypes. Casts preserve master-parameter gradients.
        Projection biases are added before sigmoid gates.

        First-order reverse mode includes floating masks and all weights/biases;
        forward-mode AD is unsupported. PyTorch's scoped FP32 and relevant
        FP16/BF16 native matmul settings are captured at forward. Changed settings
        before backward or a stale compiled call are rejected. Unsupported calls
        have no reference fallback.

        Nested masks match the pair layout per sample; shared static prefixes
        represent independent groups. CUDA Graph capture supports a fixed nested
        topology after representative public-call warmup. Replay may update packed
        values, masks, and weights; changed CPU structural metadata requires a new
        warmup and capture.

    Examples:
        >>> import torch
        >>> from oops import triangle_multiplication
        >>> if torch.cuda.is_available():
        ...     pair = torch.ones(2, 2, 16, device="cuda")
        ...     scale = torch.ones(16, device="cuda")
        ...     bias = torch.zeros(16, device="cuda")
        ...     weight = torch.eye(16, device="cuda")
        ...     parameters = dict(
        ...         input_norm_weight=scale, input_norm_bias=bias,
        ...         left_value_weight=weight, left_gate_weight=weight,
        ...         right_value_weight=weight, right_gate_weight=weight,
        ...         output_norm_weight=scale, output_norm_bias=bias,
        ...         output_weight=weight, output_gate_weight=weight,
        ...     )
        ...     result = triangle_multiplication(pair, direction="outgoing", **parameters)
        ...     assert result.shape == pair.shape
    """
    if direction not in ("incoming", "outgoing"):
        raise ValueError("direction must be 'incoming' or 'outgoing'")
    weights = TriangleMultiplicationWeights(
        input_norm_weight,
        input_norm_bias,
        left_value_weight,
        left_gate_weight,
        right_value_weight,
        right_gate_weight,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        output_gate_weight,
    )
    biases = TriangleMultiplicationBiases(
        left_value_bias,
        left_gate_bias,
        right_value_bias,
        right_gate_bias,
        output_bias,
        output_gate_bias,
    )
    if isinstance(pair, NestedTensor):
        if pair_mask is not None and not isinstance(pair_mask, NestedTensor):
            raise ValueError("triangle_multiplication pair and pair_mask must both use the same NestedTensor layout")
        return _nested(pair, pair_mask, direction, weights, biases, input_norm_eps, output_norm_eps)
    if isinstance(pair_mask, NestedTensor):
        raise ValueError("triangle_multiplication pair and pair_mask must both use the same NestedTensor layout")
    require_no_forward_mode(_OPERATOR, pair, pair_mask, *weights, *biases)
    _check_shapes(pair, pair_mask, weights)
    _check_bias_shapes(pair.shape[-1], weights.left_value_weight.shape[0], biases)
    require_same_device(_OPERATOR, pair, pair_mask, *weights, *biases)
    pair, weights, biases = _prepare_autocast(pair, weights, biases)

    num_tokens = pair.shape[-2]
    grouped_pair = pair.reshape(-1, num_tokens * num_tokens, pair.shape[-1])
    grouped_mask = (
        None if pair_mask is None else pair_mask.expand(pair.shape[:-1]).reshape(grouped_pair.shape[:2]).contiguous()
    )
    output = _dispatch(
        grouped_pair, grouped_mask, None, num_tokens, direction, weights, biases, input_norm_eps, output_norm_eps
    )
    return output.reshape(pair.shape)


def _nested(
    pair: NestedTensor,
    pair_mask: NestedTensor | None,
    direction: str,
    weights: TriangleMultiplicationWeights,
    biases: TriangleMultiplicationBiases,
    input_norm_eps: float,
    output_norm_eps: float,
) -> NestedTensor:
    pair_values = pair.concat
    mask_values = None if pair_mask is None else pair_mask.concat
    require_no_forward_mode(_OPERATOR, pair_values, mask_values, *weights, *biases)
    pair_values, mask_values, sample_row_offsets, pair_positions, prefix_shape = _canonical_nested_inputs(
        pair, pair_mask, weights
    )
    _check_bias_shapes(pair_values.shape[-1], weights.left_value_weight.shape[0], biases)
    require_same_device(_OPERATOR, pair_values, mask_values, *weights, *biases)
    pair_values, weights, biases = _prepare_autocast(pair_values, weights, biases)

    base_row_offsets = (
        sample_row_offsets.to(device=pair_values.device, dtype=torch.int64)
        if is_compiling()
        else pair.ragged_level_offsets(0, device=pair_values.device, dtype=torch.int64)
    )
    group_count = 1
    for prefix_size in prefix_shape:
        group_count *= prefix_size
    if group_count == 1:
        device_row_offsets = base_row_offsets
        # Inference tensors cannot be saved for backward; copy cached metadata on device.
        if not is_compiling() and base_row_offsets.is_inference() and not torch.is_inference_mode_enabled():
            device_row_offsets = base_row_offsets.clone()
    else:
        row_lengths = base_row_offsets[1:] - base_row_offsets[:-1]
        grouped_row_lengths = row_lengths.unsqueeze(0).expand(group_count, -1).reshape(-1)
        device_row_offsets = torch.cat((base_row_offsets.new_zeros(1), grouped_row_lengths.cumsum(0)))
    grouped_pair = pair_values.reshape(1, -1, pair_values.shape[-1])
    grouped_mask = None if mask_values is None else mask_values.reshape(1, -1)
    output_values = _dispatch(
        grouped_pair, grouped_mask, device_row_offsets, 0, direction, weights, biases, input_norm_eps, output_norm_eps
    )
    packed_output = (
        output_values.reshape(*prefix_shape, pair_positions, output_values.shape[-1])
        .movedim(-2, 0)
        .contiguous()
        .reshape(pair.concat.shape)
    )
    return pair.packed_like(packed_output)


def _dispatch(
    pair: Tensor,
    mask: Tensor | None,
    sample_row_offsets: Tensor | None,
    num_tokens: int,
    direction: str,
    weights: TriangleMultiplicationWeights,
    biases: TriangleMultiplicationBiases,
    input_norm_eps: float,
    output_norm_eps: float,
) -> Tensor:

    if not _triton.supports_triangle_multiplication(pair, mask, sample_row_offsets, weights):
        raise unsupported(
            _OPERATOR,
            "requires a CUDA float16/bfloat16/float32 pair, channel widths within the fused launch "
            "domain, and contiguous weights on the pair's device in the pair's dtype (float32 for "
            "the two LayerNorms)",
            pair=_describe(pair),
            pair_mask=_describe(mask),
            samples=None if sample_row_offsets is None else sample_row_offsets.shape[0] - 1,
            hidden_size=weights.left_value_weight.shape[0],
            weight_dtypes=tuple(sorted({str(weight.dtype) for weight in weights})),
            weight_devices=tuple(sorted({str(weight.device) for weight in weights})),
            contiguous_weights=all(weight.is_contiguous() for weight in weights),
        )
    precision, native_matmul_state = _triton.resolve_triangle_multiplication_precisions(pair.dtype)
    return _triton.triton_triangle_multiplication(
        pair,
        mask,
        sample_row_offsets,
        direction,
        weights,
        biases,
        num_tokens,
        input_norm_eps,
        output_norm_eps,
        precision,
        native_matmul_state,
        semantic=is_compiling(),
    )


def _prepare_autocast(
    pair: Tensor,
    weights: TriangleMultiplicationWeights,
    biases: TriangleMultiplicationBiases,
) -> tuple[Tensor, TriangleMultiplicationWeights, TriangleMultiplicationBiases]:

    pair, *projections = prepare_cuda_autocast_operands(pair, (pair, *weights.projections))
    weights = weights.with_projections(tuple(projections))
    present = biases.present
    if present:
        biases = biases.replace_present(prepare_cuda_autocast_operands(pair, present))
        for name, bias in zip(biases._fields, biases):
            if bias is not None and (bias.dtype != pair.dtype or not bias.is_contiguous()):
                raise unsupported(
                    _OPERATOR,
                    "requires contiguous projection biases in the pair's dtype",
                    bias=name,
                    operand=_describe(bias),
                )
    return pair, weights, biases


def _canonical_nested_inputs(
    pair: NestedTensor,
    pair_mask: NestedTensor | None,
    weights: TriangleMultiplicationWeights,
) -> tuple[Tensor, Tensor | None, Tensor, int, tuple[int, ...]]:
    if not pair.batch_first:
        raise ValueError("triangle_multiplication NestedTensor batch must be the leading logical dimension")
    if pair.dim() < 4:
        raise ValueError(
            "triangle_multiplication nested pair elements must have shape (*prefix, N_i, N_i, pair_size); "
            f"got logical rank {pair.dim()}"
        )
    prefix_rank = pair.dim() - 4
    token_dims = (prefix_rank, prefix_rank + 1)
    expected_pair_order = (*token_dims, *range(prefix_rank), prefix_rank + 2)
    if pair.packed_dim_order != expected_pair_order:
        raise unsupported(
            _OPERATOR,
            "requires token-pair-major nested (*prefix, N_i, N_i, pair_size) packing",
            packed_dim_order=pair.packed_dim_order,
        )
    if pair.ragged_dims != token_dims:
        raise unsupported(
            _OPERATOR,
            "requires both nested token axes to be declared ragged",
            ragged_dims=pair.ragged_dims,
        )
    packed = pair.concat
    packed_rank = packed.dim() - prefix_rank - 1
    if packed_rank not in (1, 2) or packed.shape[-1] <= 0:
        raise ValueError(
            "triangle_multiplication nested pair elements must retain their static prefix and channel axis"
        )
    if not packed.is_contiguous():
        raise unsupported(_OPERATOR, "requires contiguous NestedTensor packed pair values")
    _check_weight_shapes(packed.shape[-1], weights)
    pair_positions = packed.shape[0] if packed_rank == 1 else packed.shape[0] * packed.shape[1]
    prefix_shape = tuple(packed.shape[packed_rank:-1])
    cell_major_pair = packed.flatten(0, packed_rank - 1)
    pair_order = (*range(1, prefix_rank + 1), 0, prefix_rank + 1)
    pair_values = cell_major_pair.permute(*pair_order).contiguous().flatten(0, -2)
    pair_rows = pair.ragged_level_offsets(0, dtype=torch.int64)
    pair_inner = pair.ragged_level_offsets(1, dtype=torch.int64) if packed_rank == 1 else None
    pair_static_width = packed.shape[1] if packed_rank == 2 else 0

    if pair_mask is None:
        mask_values = None
        mask_rows = None
        mask_inner = None
        mask_rank = 0
        mask_static_width = 0
        mask_total_positions = 0
    else:
        if not pair_mask.batch_first or pair_mask.dim() != pair.dim() - 1:
            raise ValueError(
                "triangle_multiplication nested pair_mask elements must have shape (*prefix, N_i, N_i) "
                "with a leading wrapper batch"
            )
        expected_mask_order = (*token_dims, *range(prefix_rank))
        if pair_mask.packed_dim_order != expected_mask_order:
            raise unsupported(
                _OPERATOR,
                "requires token-pair-major nested (*prefix, N_i, N_i) mask packing",
                packed_dim_order=pair_mask.packed_dim_order,
            )
        if pair_mask.ragged_dims != token_dims:
            raise unsupported(
                _OPERATOR,
                "requires both nested pair_mask token axes to be declared ragged",
                ragged_dims=pair_mask.ragged_dims,
            )
        mask_packed = pair_mask.concat
        mask_rank = mask_packed.dim() - prefix_rank
        mask_prefix_shape = tuple(mask_packed.shape[mask_rank:])
        if mask_rank not in (1, 2) or mask_prefix_shape != prefix_shape or not mask_packed.is_contiguous():
            raise unsupported(
                _OPERATOR,
                "requires a contiguous nested pair_mask with the pair's complete static prefix",
            )
        mask_total_positions = mask_packed.shape[0] if mask_rank == 1 else mask_packed.shape[0] * mask_packed.shape[1]
        cell_major_mask = mask_packed.flatten(0, mask_rank - 1)
        mask_order = (*range(1, prefix_rank + 1), 0)
        mask_values = cell_major_mask.permute(*mask_order).contiguous().flatten()
        mask_rows = pair_mask.ragged_level_offsets(0, dtype=torch.int64)
        mask_inner = pair_mask.ragged_level_offsets(1, dtype=torch.int64) if mask_rank == 1 else None
        mask_static_width = mask_packed.shape[1] if mask_rank == 2 else 0

    validated_row_offsets = canonicalize_nested_triangle_metadata(
        pair.element_sizes(),
        packed.shape[-1],
        pair_rows,
        pair_inner,
        packed_rank,
        pair_static_width,
        pair_positions,
        mask_rows,
        mask_inner,
        mask_rank,
        mask_static_width,
        mask_total_positions,
    )
    return pair_values, mask_values, validated_row_offsets, pair_positions, prefix_shape


def _describe(tensor: Tensor | None) -> str | None:
    return None if tensor is None else f"{tuple(tensor.shape)} {tensor.dtype} {tensor.device}"


def _check_bias_shapes(pair_size: int, hidden_size: int, biases: TriangleMultiplicationBiases) -> None:
    widths = (hidden_size, hidden_size, hidden_size, hidden_size, pair_size, pair_size)
    for name, bias, width in zip(biases._fields, biases, widths, strict=True):
        if bias is not None and tuple(bias.shape) != (width,):
            raise ValueError(f"{name} must have shape {(width,)}; got {tuple(bias.shape)}")


def _check_shapes(pair: Tensor, pair_mask: Tensor | None, weights: TriangleMultiplicationWeights) -> None:
    if pair.dim() < 3:
        raise ValueError(f"pair must have shape (..., N, N, pair_size); got rank {pair.dim()}")
    if pair.shape[-2] != pair.shape[-3]:
        raise ValueError(f"pair token dims must be square (N, N); got {tuple(pair.shape[-3:-1])}")
    if pair_mask is not None and not _broadcasts_to(pair_mask.shape, pair.shape[:-1]):
        raise ValueError(
            f"pair_mask shape {tuple(pair_mask.shape)} must broadcast to pair[..., N, N] = {tuple(pair.shape[:-1])}"
        )
    _check_weight_shapes(pair.shape[-1], weights)


def _broadcasts_to(shape: torch.Size, target: torch.Size) -> bool:

    return len(shape) <= len(target) and all(size in (1, full) for size, full in zip(reversed(shape), reversed(target)))


def _check_weight_shapes(pair_size: int, weights: TriangleMultiplicationWeights) -> None:
    hidden_size = weights.left_value_weight.shape[0]
    projection = (hidden_size, pair_size)
    expected = (
        (pair_size,),
        (pair_size,),
        projection,
        projection,
        projection,
        projection,
        (hidden_size,),
        (hidden_size,),
        (pair_size, hidden_size),
        (pair_size, pair_size),
    )
    for name, weight, shape in zip(weights._fields, weights, expected, strict=True):
        if tuple(weight.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}; got {tuple(weight.shape)}")
