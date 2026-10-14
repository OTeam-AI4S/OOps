from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.attention.flash import (
    MASK_KEY,
    MASK_NONE,
    MASK_PAIR,
    DenseAttentionSpec,
    check_head_major_qkv,
    pair_mask_kind,
    run_dense_attention,
)
from oops.attention.flash.triton import MAX_HEAD_SIZE
from oops.runtime import (
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
)
from oops.runtime import needs_grad as tensors_need_grad
from oops.runtime import require_no_forward_mode, unsupported

from . import triton as _triton
from .composed import native_pair_bias_attention
from .nested import canonicalize_nested_pair_bias_attention_metadata
from .triton.ops import _prepare_eager_pair_bias_attention


def pair_bias_attention(
    query: Tensor | NestedTensor,
    key: Tensor | NestedTensor,
    value: Tensor | NestedTensor,
    mask: Tensor | NestedTensor | None = None,
    pair_bias: Tensor | NestedTensor | None = None,
    *,
    scale: float | None = None,
) -> Tensor | NestedTensor:
    r"""
    Apply scaled dot-product attention with an optional additive pair bias.

    Q/K/V use head-major layouts. Each prefix element and group has an independent
    softmax domain; query and key lengths may differ.

    Args:
        query: Dense ``(*prefix, S, H, Q, D)`` queries or a DanLing ``NestedTensor``
            with ``(S, H, Q_i, D)`` elements in token-major packed storage. Rank 3
            omits prefix and group; rank 4 omits the prefix. Nested operands may
            consistently omit ``S`` to use one group.
        key: Keys with the query prefix, groups, heads, channels, and dtype,
            replacing ``Q`` with ``K``. Nested key lengths may vary per sample.
        value: Values matching the key shape, dtype, device, and nested topology.
        mask: Optional bool keep-mask ``(*prefix, S|1, K)``,
            ``(*prefix, S|1, Q|1, K)``, or ``(*prefix, S|1, 1, Q|1, K)``.
            Omit prefix/group axes consistently with Q/K/V. ``False`` excludes a
            key or pair; ``None`` keeps all pairs. A nested mask has matching
            ``(S|1, K_i)`` or ``(S|1, Q_i|1, K_i)`` elements.
        pair_bias: Optional additive bias ``(*prefix, S|1, H|1, Q|1, K|1)``.
            Outer prefix axes match Q/K/V; for rank-5 inputs a non-unit batch axis
            has nonzero stride. Nested elements have ``(S|1, H|1, Q_i, K_i)`` shape.
            The dtype matches Q/K/V; FP32 bias over half Q/K/V is also accepted
            inside CUDA autocast matching the Q/K/V dtype.
        scale: Python scalar multiplying Q.K before bias addition. ``None`` uses
            ``D ** -0.5``.

    Returns:
        Attention values with the query shape, dtype, and device. A nested result
        preserves the query topology. Fully masked query rows return zero.

    Raises:
        ValueError: Shapes, devices, dense/nested representations, or nested
            topologies disagree.
        OOpsUnsupportedError: The device, dtype, layout, nonempty size domain,
            launch limits, or requested forward-mode differentiation is unsupported.

    Notes:
        Q/K/V share one CUDA device and float16, bfloat16, or float32 dtype, with
        ``1 <= D <= 256``. Dense inputs may be strided; every axis is nonempty.
        Nested group/head sharing is uniform across samples; mixed sharing and
        singleton query/key bias axes are unsupported. Masks and bias share the
        Q/K/V device.

        First-order reverse mode differentiates Q/K/V and bias. Broadcast bias
        gradients are summed to the original shape and returned in its dtype.
        FP32 contractions capture the effective scoped PyTorch matmul precision at
        forward and reuse it in backward even if the setting changes. The bool
        mask is not differentiable; forward-mode AD is unsupported. Unsupported
        calls raise instead of selecting a reference fallback.

    Examples:
        >>> import torch
        >>> from oops import pair_bias_attention
        >>> if torch.cuda.is_available():
        ...     q = torch.ones(1, 2, 16, device="cuda")
        ...     result = pair_bias_attention(q, q, q)
        ...     assert result.shape == q.shape
    """
    if type(query) is Tensor:
        require_no_forward_mode("pair_bias_attention", query, key, value, pair_bias)
        return _pair_bias_attention_dense(query, key, value, mask, pair_bias, scale)
    if isinstance(query, NestedTensor):
        if not isinstance(key, NestedTensor) or not isinstance(value, NestedTensor):
            raise ValueError("pair_bias_attention query, key, and value must all use NestedTensor")
        if mask is not None and not isinstance(mask, NestedTensor):
            raise ValueError("pair_bias_attention nested query/key/value require a NestedTensor mask")
        if pair_bias is not None and not isinstance(pair_bias, NestedTensor):
            raise ValueError("pair_bias_attention nested query/key/value require a NestedTensor pair_bias")
        return _pair_bias_attention_nested(query, key, value, mask, pair_bias, scale)
    if isinstance(key, NestedTensor) or isinstance(value, NestedTensor):
        raise ValueError("pair_bias_attention query, key, and value must all use NestedTensor")
    if isinstance(mask, NestedTensor) or isinstance(pair_bias, NestedTensor):
        raise ValueError("pair_bias_attention dense query/key/value cannot use NestedTensor mask or pair_bias")
    require_no_forward_mode("pair_bias_attention", query, key, value, pair_bias)
    return _pair_bias_attention_dense(query, key, value, mask, pair_bias, scale)


def _pair_bias_attention_dense(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    scale: float | None,
) -> Tensor:
    if mask is not None and mask.dim() == query.dim():
        if mask.shape[-3] != 1:
            raise unsupported(
                "pair_bias_attention",
                "serves a mask head axis only of size 1",
                mask_shape=tuple(mask.shape),
                query_shape=tuple(query.shape),
            )
        mask = mask.squeeze(-3)
    if mask is not None and mask.dim() == query.dim() - 1 and mask.shape[-2] == 1:
        mask = mask.squeeze(-2)
    if query.dim() > _PAIR_BIAS_ATTENTION.maximum_rank:
        return _pair_bias_attention_prefixed(query, key, value, mask, pair_bias, scale)
    return run_dense_attention(_PAIR_BIAS_ATTENTION, query, key, value, mask, pair_bias, scale=scale)


def _pair_bias_attention_prefixed(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    scale: float | None,
) -> Tensor:

    check_head_major_qkv(query, key, value)
    batch_prefix = query.shape[:-4]
    groups = query.shape[-4]
    mask_kind = pair_mask_kind(query, mask)
    if mask is not None:
        if mask_kind == MASK_KEY:
            valid_mask = (
                mask.shape[:-2] == batch_prefix and mask.shape[-2] in (1, groups) and mask.shape[-1] == key.shape[-2]
            )
        elif mask_kind == MASK_PAIR:
            valid_mask = (
                mask.shape[:-3] == batch_prefix
                and mask.shape[-3] in (1, groups)
                and mask.shape[-2:] == (query.shape[-2], key.shape[-2])
            )
        else:
            valid_mask = False
        if not valid_mask:
            raise ValueError(
                "pair_bias_attention mask must have shape "
                "(*batch_prefix, group|1, key_len) or "
                "(*batch_prefix, group|1, query_length|1, key_len), optionally with a unit head "
                "axis before query_length, matching query/key"
            )
    if pair_bias is not None and (
        pair_bias.dim() != query.dim()
        or pair_bias.shape[:-4] != batch_prefix
        or pair_bias.shape[-4] not in (1, groups)
        or pair_bias.shape[-3] not in (1, query.shape[-3])
        or pair_bias.shape[-2] not in (1, query.shape[-2])
        or pair_bias.shape[-1] not in (1, key.shape[-2])
    ):
        raise ValueError(
            "pair_bias_attention pair_bias must have shape "
            "(*batch_prefix, group|1, num_heads|1, query_length|1, key_len|1) matching query/key"
        )

    grouped_query = query.flatten(0, -5)
    grouped_key = grouped_query if key is query else key.flatten(0, -5)
    if value is query:
        grouped_value = grouped_query
    elif value is key:
        grouped_value = grouped_key
    else:
        grouped_value = value.flatten(0, -5)
    if mask is None:
        grouped_mask = None
    elif mask_kind == MASK_KEY:
        grouped_mask = mask.flatten(0, -3)
    else:
        grouped_mask = mask.flatten(0, -4)
    grouped_bias = None if pair_bias is None else pair_bias.flatten(0, -5)
    output = run_dense_attention(
        _PAIR_BIAS_ATTENTION,
        grouped_query,
        grouped_key,
        grouped_value,
        grouped_mask,
        grouped_bias,
        scale=scale,
    )
    return output.reshape(query.shape)


def _pair_bias_attention_nested(
    query: NestedTensor,
    key: NestedTensor,
    value: NestedTensor,
    mask: NestedTensor | None,
    pair_bias: NestedTensor | None,
    scale: float | None,
) -> NestedTensor:
    query_values, implicit_group = _canonical_nested_qkv("query", query)
    key_values, key_implicit_group = _canonical_nested_qkv("key", key)
    value_values, value_implicit_group = _canonical_nested_qkv("value", value)
    require_no_forward_mode(
        "pair_bias_attention",
        query_values,
        key_values,
        value_values,
        None if pair_bias is None else pair_bias.concat,
    )
    if key_values.shape[1:] != query_values.shape[1:] or value_values.shape[1:] != query_values.shape[1:]:
        raise ValueError("pair_bias_attention nested Q/K/V must share S, H, and D")
    if key_implicit_group != implicit_group or value_implicit_group != implicit_group:
        raise ValueError("pair_bias_attention nested Q/K/V must all either include or omit the unit sample axis")

    if mask is None:
        mask_values = None
        mask_rows = mask_inner = mask_packed = None
        mask_kind = MASK_NONE
        mask_groups = 0
        mask_total_values = 0
    else:
        (
            mask_values,
            mask_rows,
            mask_inner,
            mask_packed,
            mask_kind,
            mask_groups,
        ) = _canonical_nested_mask(mask)
        mask_total_values = mask_values.numel()

    if pair_bias is None:
        bias_values = None
        bias_rows = bias_inner = bias_packed = None
        bias_groups = bias_heads = 0
        bias_total_values = 0
    else:
        (
            bias_values,
            bias_rows,
            bias_inner,
            bias_packed,
            bias_groups,
            bias_heads,
        ) = _canonical_nested_bias(pair_bias)
        bias_total_values = bias_values.numel()

    query_offsets = query.packed_offsets(dtype=torch.int64)
    key_offsets = key.packed_offsets(dtype=torch.int64)
    validation = canonicalize_nested_pair_bias_attention_metadata(
        query_offsets,
        query_values.shape[0],
        key_offsets,
        key_values.shape[0],
        value.packed_offsets(dtype=torch.int64),
        value_values.shape[0],
        query_values.shape[1],
        query_values.shape[2],
        mask_rows,
        mask_inner,
        mask_packed,
        mask_kind,
        mask_total_values,
        mask_groups,
        bias_rows,
        bias_inner,
        bias_packed,
        bias_total_values,
        bias_groups,
        bias_heads,
    )
    if is_compiling():
        torch._check_tensor_all(validation)

    device = query_values.device
    device_query_offsets = query.packed_offsets(device=device, dtype=torch.int64)
    device_key_offsets = key.packed_offsets(device=device, dtype=torch.int64)
    device_mask_offsets = (
        device_query_offsets if mask is None else mask.packed_offsets(device=device, dtype=torch.int64)
    )
    device_bias_offsets = (
        device_query_offsets if pair_bias is None else pair_bias.packed_offsets(device=device, dtype=torch.int64)
    )
    if is_compiling():
        torch._dynamo.mark_static(query_values, -1)
    if not _triton.supports_segmented_pair_bias_attention(
        query_values,
        key_values,
        value_values,
        mask_values,
        bias_values,
        device_query_offsets,
        device_key_offsets,
        device_mask_offsets,
        device_bias_offsets,
        mask_kind=mask_kind,
        mask_groups=mask_groups,
        bias_groups=bias_groups,
        bias_heads=bias_heads,
    ):
        raise unsupported(
            "pair_bias_attention",
            "requires CUDA float16/bfloat16/float32 canonical token-major NestedTensor Q/K/V with "
            f"head_size <= {MAX_HEAD_SIZE}, a uniform key/pair bool mask, and a literal full-Q/K "
            "uniform sample/head pair bias in the Q/K/V dtype or matching-autocast float32",
            query=_describe(query_values),
            key=_describe(key_values),
            value=_describe(value_values),
            mask=_describe(mask_values),
            pair_bias=_describe(bias_values),
        )
    output_values = _triton.triton_segmented_pair_bias_attention(
        query_values,
        key_values,
        value_values,
        mask_values,
        bias_values,
        device_query_offsets,
        device_key_offsets,
        device_mask_offsets,
        device_bias_offsets,
        mask_kind=mask_kind,
        mask_groups=mask_groups,
        bias_groups=bias_groups,
        bias_heads=bias_heads,
        scale=scale,
    )
    if implicit_group:
        output_values = output_values.squeeze(1)
    return query.packed_like(output_values)


def _canonical_nested_qkv(operand: str, nested: NestedTensor) -> tuple[Tensor, bool]:
    if not nested.batch_first:
        raise ValueError(f"pair_bias_attention nested {operand} must be batch-first")
    implicit_group = nested.dim() == 4 and nested.ragged_dims == (1,) and nested.packed_dim_order == (1, 0, 2)
    explicit_group = nested.dim() == 5 and nested.ragged_dims == (2,) and nested.packed_dim_order == (2, 0, 1, 3)
    if not implicit_group and not explicit_group:
        raise unsupported(
            "pair_bias_attention",
            f"requires token-major (H, N_i, D) or (S, H, N_i, D) nested {operand} packing",
            ragged_dims=nested.ragged_dims,
            packed_dim_order=nested.packed_dim_order,
        )
    values = nested.concat
    if implicit_group:
        if values.dim() != 3 or any(size <= 0 for size in values.shape[1:]):
            raise ValueError(f"pair_bias_attention nested {operand} must retain positive H and D axes")
        values = values.unsqueeze(1)
    elif values.dim() != 4 or any(size <= 0 for size in values.shape[1:]):
        raise ValueError(f"pair_bias_attention nested {operand} must retain positive S, H, and D axes")
    return values, implicit_group


def _canonical_nested_mask(
    mask: NestedTensor,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor, int, int]:
    if not mask.batch_first:
        raise ValueError("pair_bias_attention nested mask must be batch-first")
    implicit_group = singleton_query = False
    if mask.dim() == 2 and mask.ragged_dims == (0,) and mask.packed_dim_order == (0,):
        mask_kind = MASK_KEY
        rows = mask.ragged_level_offsets(0, dtype=torch.int64)
        inner = None
        implicit_group = True
    elif mask.dim() == 3 and mask.ragged_dims == (0, 1) and mask.packed_dim_order == (0, 1):
        mask_kind = MASK_PAIR
        rows = mask.ragged_level_offsets(0, dtype=torch.int64)
        inner = mask.ragged_level_offsets(1, dtype=torch.int64)
        implicit_group = True
    elif mask.dim() == 3 and mask.ragged_dims == (1,) and mask.packed_dim_order == (1, 0):
        mask_kind = MASK_KEY
        rows = mask.ragged_level_offsets(0, dtype=torch.int64)
        inner = None
    elif mask.dim() == 4 and mask.ragged_dims == (1, 2) and mask.packed_dim_order == (1, 2, 0):
        mask_kind = MASK_PAIR
        rows = mask.ragged_level_offsets(0, dtype=torch.int64)
        inner = mask.ragged_level_offsets(1, dtype=torch.int64)
    elif mask.dim() == 4 and mask.ragged_dims == (2,) and mask.packed_dim_order == (2, 0, 1):
        mask_kind = MASK_KEY
        rows = mask.ragged_level_offsets(0, dtype=torch.int64)
        inner = None
        singleton_query = True
    else:
        raise unsupported(
            "pair_bias_attention",
            "requires a key/pair mask with an implicit unit sample axis or a uniformly shared/full "
            "explicit sample axis; per-element mixed sample sharing is not served",
            ragged_dims=mask.ragged_dims,
            packed_dim_order=mask.packed_dim_order,
        )
    values = mask.concat
    if implicit_group:
        if values.dim() != 1:
            raise ValueError("pair_bias_attention implicit-sample nested mask must have flattened packed storage")
        values = values.unsqueeze(1)
    elif singleton_query:
        if values.dim() != 3 or values.shape[2] != 1:
            raise unsupported(
                "pair_bias_attention",
                "requires a nested mask query axis that is ragged as Q_i or a singleton",
                packed_shape=tuple(values.shape),
            )
        values = values.squeeze(2)
    elif values.dim() != 2:
        raise ValueError("pair_bias_attention nested mask must retain one static sample axis")
    return values, rows, inner, mask.packed_offsets(dtype=torch.int64), mask_kind, values.shape[1]


def _canonical_nested_bias(
    pair_bias: NestedTensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, int]:
    if not pair_bias.batch_first:
        raise ValueError("pair_bias_attention nested pair_bias must be batch-first")
    implicit_group = (
        pair_bias.dim() == 4 and pair_bias.ragged_dims == (1, 2) and pair_bias.packed_dim_order == (1, 2, 0)
    )
    explicit_group = (
        pair_bias.dim() == 5 and pair_bias.ragged_dims == (2, 3) and pair_bias.packed_dim_order == (2, 3, 0, 1)
    )
    if not implicit_group and not explicit_group:
        raise unsupported(
            "pair_bias_attention",
            "requires a literal full-Q/K pair_bias with an implicit unit sample axis or uniform explicit "
            "sample/head sharing; per-element mixed S/H sharing and singleton Q/K broadcast are not served",
            ragged_dims=pair_bias.ragged_dims,
            packed_dim_order=pair_bias.packed_dim_order,
        )
    values = pair_bias.concat
    if implicit_group:
        if values.dim() != 2 or values.shape[1] <= 0:
            raise ValueError("pair_bias_attention implicit-sample nested pair_bias must retain a positive head axis")
        values = values.unsqueeze(1)
    elif values.dim() != 3 or values.shape[1] <= 0 or values.shape[2] <= 0:
        raise ValueError("pair_bias_attention nested pair_bias must retain positive sample and head axes")
    return (
        values,
        pair_bias.ragged_level_offsets(0, dtype=torch.int64),
        pair_bias.ragged_level_offsets(1, dtype=torch.int64),
        pair_bias.packed_offsets(dtype=torch.int64),
        values.shape[1],
        values.shape[2],
    )


def _describe(tensor: Tensor | None) -> str | None:
    return None if tensor is None else f"{tuple(tensor.shape)} {tensor.dtype} {tensor.device}"


def _run_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    scale: float | None,
) -> Tensor:
    prepared = None
    compiling = is_compiling()
    query_type = type(query)
    external_fake = (
        not compiling and query_type is not Tensor and query_type is not torch.nn.Parameter and is_external_fake(query)
    )
    if compiling or external_fake:
        supported = _triton.supports_pair_bias_attention(query, key, value, mask, pair_bias)
    else:
        prepared = _prepare_eager_pair_bias_attention(query, key, value, mask, pair_bias)
        supported = prepared is not None
    if not supported:
        raise unsupported(
            "pair_bias_attention",
            "requires CUDA float16/bfloat16/float32 query/key/value sharing one dtype with "
            f"head_size <= {MAX_HEAD_SIZE}, a pair bias in that dtype -- or a same-device "
            "float32 bias over half-precision Q/K/V inside matching CUDA autocast -- at the "
            "query's own rank "
            "shaped (batch, sample|1, num_heads|1, query_length|1, key_len|1) whose batch axis is "
            "the query's own and not expanded over a zero stride, and a bool keep "
            "mask shaped (batch, sample|1, key_len) or (batch, sample|1, query_length|1, key_len), "
            "optionally with a unit head axis before query_length, "
            "each dropping its batch axis where the query drops its own, with every active "
            "resolved attention launch fitting CUDA grid limits",
            device=query.device,
            query_dtype=query.dtype,
            key_dtype=key.dtype,
            value_dtype=value.dtype,
            query_shape=tuple(query.shape),
            key_shape=tuple(key.shape),
            pair_bias_dtype=None if pair_bias is None else pair_bias.dtype,
            pair_bias_shape=None if pair_bias is None else tuple(pair_bias.shape),
            # A zero-stride non-unit batch axis still broadcasts despite its apparent shape.
            pair_bias_stride=None if pair_bias is None else tuple(pair_bias.stride()),
            mask_dtype=None if mask is None else mask.dtype,
            mask_shape=None if mask is None else tuple(mask.shape),
        )
    if (
        prepared is not None
        and pair_bias is not None
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape[-1] % 8 == 0
        and query.shape[-2] % 8 == 0
        and key.shape[-2] % 8 == 0
        and all(tensor.requires_grad and tensor.is_contiguous() for tensor in (query, key, value, pair_bias))
        and tuple(pair_bias.shape[-2:]) == (query.shape[-2], key.shape[-2])
        and tensors_need_grad(query, key, value, pair_bias)
        and not is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in (query, key, value, pair_bias))
        and (mask is None or type(mask) is Tensor)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function((query, key, value, pair_bias, mask))
        and not functorch_transforms_active()
        and pair_bias.dtype == query.dtype
        and query.shape[-1] <= 128
        and query.numel() // (query.shape[-2] * query.shape[-1]) <= 16
        and 128 <= query.shape[-2] <= 1024
        and 128 <= key.shape[-2] <= 1024
    ):
        with torch.cuda.device(query.device):
            if not torch.cuda.is_current_stream_capturing():
                native = native_pair_bias_attention(query, key, value, mask, pair_bias, scale)
                if native is not None:
                    return native

    return _triton.triton_pair_bias_attention(
        query,
        key,
        value,
        mask,
        pair_bias,
        scale=scale,
        prepared=prepared,
    )


_PAIR_BIAS_ATTENTION = DenseAttentionSpec(
    name="pair_bias_attention",
    batchless_rank=3,
    maximum_rank=5,
    check_qkv=check_head_major_qkv,
    run=_run_pair_bias_attention,
)
