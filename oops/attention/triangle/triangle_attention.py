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

from oops.runtime import is_compiling, require_no_forward_mode, require_same_device, unsupported

from . import triton as _triton
from .nested import triangle_attention_bias_query_major, triangle_attention_sample_table

_OPERATOR = "triangle_attention"


def triangle_attention(
    query: Tensor | NestedTensor,
    key: Tensor | NestedTensor,
    value: Tensor | NestedTensor,
    mask: Tensor | NestedTensor | None = None,
    triangle_bias: Tensor | NestedTensor | None = None,
    *,
    scale: float | None = None,
) -> Tensor | NestedTensor:
    r"""
    Apply row-wise triangle attention over the column axis of pair features.

    Each row has an independent softmax domain. Dense row and column lengths may
    differ.

    Args:
        query: Dense ``(*prefix, R, C, H, D)`` queries, with any number of prefix
            axes, ``1 <= D <= 256``, and nonempty dimensions, or a DanLing ``NestedTensor`` containing square
            ``(N_i, N_i, H, D)`` elements.
        key: Keys matching the query shape, dtype, device, and nested topology.
        value: Values matching the query shape, dtype, device, and nested topology.
        mask: Optional bool keep-mask broadcastable to ``(*prefix, R, C)`` or a
            matching nested mask. ``False`` excludes a key column; ``None`` keeps
            every key.
        triangle_bias: Optional additive bias ``(*prefix, R|1, H, C, C)`` or nested
            ``(N_i|1, H, N_i, N_i)`` elements. Its dtype may match Q/K/V or be FP32,
            including outside autocast.
        scale: Python scalar multiplying Q.K before bias addition. ``None`` uses
            ``D ** -0.5``.

    Returns:
        Attention values with the query shape, dtype, and device. A nested result
        preserves the query topology. Fully masked rows return zero.

    Raises:
        ValueError: Shapes, devices, dense/nested representations, or nested
            topologies disagree.
        OOpsUnsupportedError: The device, dtype, layout, nonempty size domain,
            launch limits, or requested forward-mode differentiation is unsupported.

    Examples:
        >>> import torch
        >>> from oops import triangle_attention
        >>> if torch.cuda.is_available():
        ...     q = torch.ones(2, 3, 1, 16, device="cuda")
        ...     result = triangle_attention(q, q, q)
        ...     assert result.shape == q.shape
    """
    if isinstance(query, NestedTensor):
        if not isinstance(key, NestedTensor) or not isinstance(value, NestedTensor):
            raise ValueError("triangle_attention query, key, and value must all use the same NestedTensor layout")
        if mask is not None and not isinstance(mask, NestedTensor):
            raise ValueError("triangle_attention mask must use the same NestedTensor layout as nested query/key/value")
        if triangle_bias is not None and not isinstance(triangle_bias, NestedTensor):
            raise ValueError(
                "triangle_attention triangle_bias must use a compatible NestedTensor layout with nested query/key/value"
            )
        return _nested(query, key, value, mask, triangle_bias, scale)
    if isinstance(key, NestedTensor) or isinstance(value, NestedTensor):
        raise ValueError("triangle_attention query, key, and value must all use the same NestedTensor layout")
    if isinstance(mask, NestedTensor) or isinstance(triangle_bias, NestedTensor):
        raise ValueError("triangle_attention dense query/key/value cannot use NestedTensor mask or triangle_bias")
    require_no_forward_mode(_OPERATOR, query, key, value, triangle_bias)
    return _dense(query, key, value, mask, triangle_bias, scale)


def _dense(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    triangle_bias: Tensor | None,
    scale: float | None,
) -> Tensor:
    _check_dense_shapes(query, key, value, mask, triangle_bias)
    require_same_device(_OPERATOR, query, key, value, mask, triangle_bias)
    _require_bool_mask(mask)
    shape = query.shape
    batchless = query.dim() == 4
    prefixed = query.dim() > 5
    if batchless or prefixed:

        def batched(tensor: Tensor) -> Tensor:
            return tensor.unsqueeze(0) if batchless else tensor.flatten(0, -5)

        batched_query = batched(query)
        batched_key = batched_query if key is query else batched(key)
        batched_value = batched_query if value is query else batched_key if value is key else batched(value)
        query, key, value = batched_query, batched_key, batched_value
        triangle_bias = None if triangle_bias is None else batched(triangle_bias)
    batch, rows, cols, num_heads, head_size = query.shape
    if is_compiling():
        torch._dynamo.mark_static(query, -1)
    packed_query, packed_key, packed_value, transposed = _packed_qkv(query, key, value)
    if mask is None:
        packed_mask = None
    else:
        packed_mask = mask.expand(shape[:-2]).to(torch.int16, memory_format=torch.contiguous_format).view(-1)
    if triangle_bias is None:
        packed_bias = None
        bias_groups = 0
    else:
        bias_groups = triangle_bias.shape[1]
        packed_bias = triangle_bias.contiguous()
    sample_table = _triton.uniform_sample_table(batch, rows, cols, num_heads, bias_groups, query.device)
    if not _triton.supports_triangle_attention(
        packed_query,
        packed_key,
        packed_value,
        packed_mask,
        packed_bias,
        sample_table,
        uniform_groups=rows,
        uniform_cols=cols,
    ):
        raise unsupported(
            _OPERATOR,
            "requires CUDA float16/bfloat16/float32 q/k/v of one dtype and shape with head_size <= 256, "
            "a bool mask broadcastable to (*prefix, rows, cols), and a (*prefix, rows|1, num_heads, cols, cols) "
            "bias in the q/k/v dtype or float32",
            query=_describe(query),
            key=_describe(key),
            value=_describe(value),
            mask=_describe(mask),
            triangle_bias=_describe(triangle_bias),
        )
    packed_output = _triton.triton_triangle_attention(
        packed_query,
        packed_key,
        packed_value,
        packed_mask,
        packed_bias,
        sample_table,
        uniform_groups=rows,
        uniform_cols=cols,
        bias_shared_only=bias_groups == 1,
        qkv_transposed=transposed,
        mask_transposed=False,
        scale=scale,
        semantic=is_compiling(),
    )
    if transposed:
        output = packed_output.transpose(1, 2)
    else:
        output = packed_output
    if batchless:
        return output.squeeze(0)
    return output.reshape(shape) if prefixed else output


def _require_bool_mask(mask: Tensor | None) -> None:

    if mask is not None and mask.dtype is not torch.bool:
        raise unsupported(_OPERATOR, "requires a bool keep-mask", mask_dtype=mask.dtype)


def _check_dense_shapes(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    triangle_bias: Tensor | None,
) -> None:
    if query.dim() < 4:
        raise ValueError(
            "triangle_attention query must be (*prefix, rows, cols, num_heads, head_size) of rank at least 4, "
            f"got rank {query.dim()}"
        )
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("triangle_attention query, key, and value must have the same shape")
    if mask is not None:
        target = tuple(query.shape[:-2])
        if mask.dim() > len(target) or any(
            size not in (1, extent) for size, extent in zip(reversed(mask.shape), reversed(target))
        ):
            raise ValueError(
                f"triangle_attention mask shape {tuple(mask.shape)} must broadcast to the query's "
                f"(*prefix, rows, cols) = {target}"
            )
    if triangle_bias is not None:
        *prefix, rows, cols, num_heads, _head_size = query.shape
        expected = (*prefix, "rows|1", num_heads, cols, cols)
        shape = tuple(triangle_bias.shape)
        if (
            triangle_bias.dim() != query.dim()
            or shape[: len(prefix)] != tuple(prefix)
            or shape[len(prefix)] not in (1, rows)
            or shape[len(prefix) + 1 :] != (num_heads, cols, cols)
        ):
            raise ValueError(f"triangle_attention triangle_bias shape {shape} must be {expected}")


def _packed_qkv(query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor, bool]:

    orientations = tuple(_pair_orientation(tensor) for tensor in (query, key, value))
    transposed = orientations[0] is True and all(orientation is True for orientation in orientations)
    coalesce = not transposed and not all(orientation is False for orientation in orientations)

    def pack(tensor: Tensor) -> Tensor:
        if coalesce:
            tensor = tensor.contiguous()
        if transposed:
            tensor = tensor.transpose(1, 2)
        return tensor

    packed_query = pack(query)
    packed_key = packed_query if key is query else pack(key)
    packed_value = packed_query if value is query else packed_key if value is key else pack(value)
    return packed_query, packed_key, packed_value, transposed


def _pair_orientation(tensor: Tensor) -> bool | None:
    r"""False: row-major; True: swapped pair axes; None: neither layout."""

    if tensor.is_contiguous():
        return False
    batch, rows, cols, num_heads, head_size = tensor.shape
    expected = (rows * cols * num_heads * head_size, num_heads * head_size, rows * num_heads * head_size, head_size, 1)
    if all(size == 1 or stride == wanted for size, stride, wanted in zip(tensor.shape, tensor.stride(), expected)):
        return True
    return None


def _nested(
    query: NestedTensor,
    key: NestedTensor,
    value: NestedTensor,
    mask: NestedTensor | None,
    triangle_bias: NestedTensor | None,
    scale: float | None,
) -> NestedTensor:
    query_values, transposed = _canonical_nested_qkv("query", query)
    key_values, key_transposed = _canonical_nested_qkv("key", key)
    value_values, value_transposed = _canonical_nested_qkv("value", value)
    require_no_forward_mode(
        _OPERATOR,
        query_values,
        key_values,
        value_values,
        None if triangle_bias is None else triangle_bias.concat,
    )
    if key_values.shape[1:] != query_values.shape[1:] or value_values.shape[1:] != query_values.shape[1:]:
        raise ValueError("triangle_attention nested query, key, and value must share num_heads and head_size")
    if key_transposed != transposed or value_transposed != transposed:
        raise ValueError("triangle_attention nested query, key, and value must share packed pair-axis orientation")

    mask_values: Tensor | None
    if mask is None:
        mask_values = None
        mask_rows = mask_packed = None
        mask_total_cells = 0
    else:
        if not mask.batch_first or mask.dim() != 3:
            raise ValueError("triangle_attention nested mask elements must have shape (N_i, N_i)")
        canonical_mask = mask.ragged_dims == (0, 1) and mask.packed_dim_order == (0, 1)
        transposed_mask = mask.ragged_dims == (1, 0) and mask.packed_dim_order == (1, 0)
        if not canonical_mask and not transposed_mask:
            raise unsupported(
                _OPERATOR,
                "requires explicit canonical or transposed (N_i, N_i) nested mask packing",
                ragged_dims=mask.ragged_dims,
                packed_dim_order=mask.packed_dim_order,
            )
        if transposed_mask != transposed:
            raise ValueError("triangle_attention nested mask must share q/k/v packed pair-axis orientation")
        mask_values = mask.concat
        if mask_values.dim() != 1:
            raise ValueError("triangle_attention nested mask must have flattened square pair storage")
        _require_bool_mask(mask_values)
        mask_values = mask_values.to(torch.int16)
        mask_rows = mask.ragged_level_offsets(0, dtype=torch.int64)
        mask_packed = mask.packed_offsets(dtype=torch.int64)
        mask_total_cells = mask_values.shape[0]

    bias_values: Tensor | None
    if triangle_bias is None:
        bias_values = None
        bias_groups = bias_packed = None
        bias_mode = 0
        bias_total_values = 0
        bias_transposed = False
    else:
        bias_values, bias_groups, bias_packed, bias_mode, bias_transposed = _canonical_nested_triangle_bias(
            triangle_bias, query_values.shape[1]
        )
        bias_total_values = bias_values.shape[0]
        if bias_transposed:
            bias_values = triangle_attention_bias_query_major(
                bias_values, bias_groups, bias_packed, query.ragged_level_offsets(0, dtype=torch.int64)
            ).view(-1)
        else:
            bias_values = bias_values.transpose(0, 1).contiguous().view(-1)

    device = query_values.device
    sample_table = triangle_attention_sample_table(
        query.ragged_level_offsets(0, dtype=torch.int64),
        query.packed_offsets(dtype=torch.int64),
        query_values.shape[0],
        key.ragged_level_offsets(0, dtype=torch.int64),
        key.packed_offsets(dtype=torch.int64),
        key_values.shape[0],
        value.ragged_level_offsets(0, dtype=torch.int64),
        value.packed_offsets(dtype=torch.int64),
        value_values.shape[0],
        mask_rows,
        mask_packed,
        mask_total_cells,
        bias_groups,
        bias_packed,
        bias_mode,
        bias_total_values,
        device=device,
    )
    if is_compiling():
        torch._dynamo.mark_static(query_values, -1)
    query_values, key_values, value_values = (
        tensor.contiguous() for tensor in (query_values, key_values, value_values)
    )
    if not _triton.supports_triangle_attention(
        query_values, key_values, value_values, mask_values, bias_values, sample_table
    ):
        raise unsupported(
            _OPERATOR,
            "requires CUDA float16/bfloat16/float32 canonical/transposed packed NestedTensor q/k/v with "
            "head_size <= 256, a matching bool nested mask, and independently canonical/transposed "
            "nested triangle bias in the q/k/v dtype or float32",
            query=_describe(query_values),
            key=_describe(key_values),
            value=_describe(value_values),
            mask=_describe(mask_values),
            triangle_bias=_describe(bias_values),
        )
    output_values = _triton.triton_triangle_attention(
        query_values,
        key_values,
        value_values,
        mask_values,
        bias_values,
        sample_table,
        uniform_groups=0,
        uniform_cols=0,
        bias_shared_only=bias_mode == 1,
        qkv_transposed=transposed,
        mask_transposed=transposed,
        scale=scale,
        semantic=is_compiling(),
    )
    return query.packed_like(output_values)


def _canonical_nested_qkv(operand: str, nested: NestedTensor) -> tuple[Tensor, bool]:
    if not nested.batch_first or nested.dim() != 5:
        raise ValueError(f"triangle_attention nested {operand} elements must have shape (N_i, N_i, H, D)")
    canonical = nested.ragged_dims == (0, 1) and nested.packed_dim_order == (0, 1, 2, 3)
    transposed = nested.ragged_dims == (1, 0) and nested.packed_dim_order == (1, 0, 2, 3)
    if not canonical and not transposed:
        raise unsupported(
            _OPERATOR,
            f"requires explicit canonical or transposed (N_i, N_i, H, D) nested {operand} packing",
            ragged_dims=nested.ragged_dims,
            packed_dim_order=nested.packed_dim_order,
        )
    values = nested.concat
    if values.dim() != 3 or values.shape[1] <= 0 or values.shape[2] <= 0:
        raise ValueError(f"triangle_attention nested {operand} must retain positive num_heads and head_size axes")
    return values, transposed


def _canonical_nested_triangle_bias(
    triangle_bias: NestedTensor,
    num_heads: int,
) -> tuple[Tensor, Tensor | None, Tensor, int, bool]:
    if not triangle_bias.batch_first or triangle_bias.dim() != 5:
        raise ValueError("triangle_attention nested triangle_bias elements must have shape (rows|1, H, N_i, N_i)")
    ragged_dims, order = triangle_bias.ragged_dims, triangle_bias.packed_dim_order
    if ragged_dims == (2, 3) and order == (2, 3, 0, 1):
        shared, transposed = True, False
    elif ragged_dims == (3, 2) and order == (3, 2, 0, 1):
        shared, transposed = True, True
    elif ragged_dims == (0, 2, 3) and order == (0, 2, 3, 1):
        shared, transposed = False, False
    elif ragged_dims == (0, 3, 2) and order == (0, 3, 2, 1):
        shared, transposed = False, True
    else:
        raise unsupported(
            _OPERATOR,
            "requires literal dense-order nested triangle_bias elements with canonical or transposed shared/full "
            "pair-axis ragged dimensions",
            ragged_dims=ragged_dims,
            packed_dim_order=order,
        )
    if shared and triangle_bias.shape[1] != 1:
        raise ValueError("triangle_attention row-shared nested triangle_bias must have a unit group axis")
    bias_groups = None if shared else triangle_bias.ragged_level_offsets(0, dtype=torch.int64)
    packed = triangle_bias.concat
    if packed.shape[-1] != num_heads:
        raise unsupported(
            _OPERATOR,
            "requires packed triangle_bias num_heads matching q/k/v",
            triangle_bias=_describe(packed),
            num_heads=num_heads,
        )
    return (
        packed.contiguous().view(-1, num_heads),
        bias_groups,
        triangle_bias.packed_offsets(dtype=torch.int64),
        1 if shared else 2,
        transposed,
    )


def _describe(tensor: Tensor | None) -> str | None:
    return None if tensor is None else f"{tuple(tensor.shape)} {tensor.dtype} {tensor.device}"
