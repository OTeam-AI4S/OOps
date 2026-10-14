from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor
from torch.nn import functional as F

from oops.runtime import is_compiling, require_same_device, unsupported

from ._contract import _check_gate_output
from .composed import _gate_attention_values
from .nested import canonicalize_nested_gate_attention_metadata


def gate_attention_output(
    attention: Tensor | NestedTensor,
    single: Tensor | NestedTensor,
    gate_weight: Tensor,
    output_weight: Tensor,
) -> Tensor | NestedTensor:
    r"""Apply a sigmoid gate and output projection to head-major attention values.

    The gate acts on the flattened head/channel axis; the result has one output
    feature vector per query token.

    Args:
        attention: Dense ``(*prefix, H, Q, D)`` values with rank 3, 4, or 5, or a
            DanLing ``NestedTensor`` containing ``(H, Q_i, D)`` or ``(S, H, Q_i, D)``
            elements in token-major packed storage.
        single: Gating input ``(*prefix, Q, C)`` or matching nested ``(Q_i, C)`` or
            ``(S, Q_i, C)`` elements, on the attention device.
        gate_weight: Matrix ``(H * D, C)`` on that device.
        output_weight: Matrix ``(O, H * D)`` on that device.

    Returns:
        Values shaped ``(*prefix, Q, O)`` on the input device, with dtype determined
        by the PyTorch linear/multiply operations and active autocast. A nested
        result preserves the single topology with channel width ``O``.

    Raises:
        ValueError: Ranks, shapes, devices, dense/nested representations, or nested
            topologies disagree.
        OOpsUnsupportedError: A nested operand has an unsupported packed layout,
            or a compiled call's captured matmul precision no longer matches
            the active policy.

    Notes:
        The equation is ``linear(sigmoid(linear(single, gate_weight)) *
        attention_flat, output_weight)``. Gate logits and sigmoid values retain the
        rounding boundaries of the corresponding PyTorch operations. CUDA FP32
        projection follows the effective scoped matmul precision; IEEE precision
        does not promise bitwise agreement with the PyTorch composition.

        CPU, other dtypes, mixed-dtype multiplication, higher-order reverse mode,
        and forward-mode AD are supported. Dense prefixes
        contain at most batch and group axes. Nested operands must consistently
        include or omit the group axis and agree on group counts and token lengths.

    Examples:
        Zero gate logits halve the attention values:

        >>> import torch
        >>> from oops import gate_attention_output
        >>> attention = torch.ones(1, 1, 2)
        >>> single = torch.zeros(1, 3)
        >>> gate_weight = torch.zeros(2, 3)
        >>> output_weight = torch.ones(1, 2)
        >>> gate_attention_output(attention, single, gate_weight, output_weight).tolist()
        [[1.0]]
    """

    if isinstance(attention, NestedTensor):
        if not isinstance(single, NestedTensor):
            raise ValueError("gate_attention_output nested attention requires a NestedTensor single")
        return _gate_nested_attention_output(attention, single, gate_weight, output_weight)
    if isinstance(single, NestedTensor):
        raise ValueError("gate_attention_output dense attention cannot use a NestedTensor single")
    _check_gate_output(attention, single, gate_weight, output_weight)
    require_same_device("gate_attention_output", attention, single, gate_weight, output_weight)
    if attention.dim() == 3:
        return gate_attention_output(
            attention.unsqueeze(0),
            single.unsqueeze(0),
            gate_weight,
            output_weight,
        ).squeeze(0)
    return _gate_attention_values(attention.transpose(-3, -2), single, gate_weight, output_weight)


def _gate_nested_attention_output(
    attention: NestedTensor,
    single: NestedTensor,
    gate_weight: Tensor,
    output_weight: Tensor,
) -> NestedTensor:
    if not attention.batch_first:
        raise ValueError("gate_attention_output nested attention must be batch-first")
    implicit_attention_group = (
        attention.dim() == 4 and attention.ragged_dims == (1,) and attention.packed_dim_order == (1, 0, 2)
    )
    explicit_attention_group = (
        attention.dim() == 5 and attention.ragged_dims == (2,) and attention.packed_dim_order == (2, 0, 1, 3)
    )
    if not implicit_attention_group and not explicit_attention_group:
        raise unsupported(
            "gate_attention_output",
            "requires token-major nested attention elements shaped (H, N_i, D) or (S, H, N_i, D)",
            ragged_dims=attention.ragged_dims,
            packed_dim_order=attention.packed_dim_order,
        )
    if not single.batch_first:
        raise ValueError("gate_attention_output nested single must be batch-first")
    implicit_single_group = single.dim() == 3 and single.ragged_dims == (0,) and single.packed_dim_order == (0, 1)
    explicit_single_group = single.dim() == 4 and single.ragged_dims == (1,) and single.packed_dim_order == (1, 0, 2)
    if not implicit_single_group and not explicit_single_group:
        raise unsupported(
            "gate_attention_output",
            "requires token-major nested single elements shaped (N_i, C) or (S, N_i, C)",
            ragged_dims=single.ragged_dims,
            packed_dim_order=single.packed_dim_order,
        )
    if implicit_attention_group != implicit_single_group:
        raise ValueError("gate_attention_output nested single and attention must both include or omit S")

    attention_values = attention.concat
    single_values = single.concat
    if implicit_attention_group:
        if attention_values.dim() != 3 or single_values.dim() != 2:
            raise ValueError("gate_attention_output implicit-sample nested packed values have invalid ranks")
        attention_values = attention_values.unsqueeze(1)
        single_values = single_values.unsqueeze(1)
    elif attention_values.dim() != 4 or single_values.dim() != 3:
        raise ValueError("gate_attention_output explicit-sample nested packed values have invalid ranks")
    num_groups, num_heads, head_size = attention_values.shape[1:]
    if single_values.shape[1] != num_groups:
        raise ValueError("gate_attention_output nested single and attention must share S")
    if gate_weight.dim() != 2 or output_weight.dim() != 2:
        raise ValueError("gate_weight and output_weight must be rank-2 matrices")
    gate_channels = num_heads * head_size
    if gate_weight.shape != (gate_channels, single_values.shape[-1]):
        raise ValueError("gate_weight must have shape (num_heads * head_size, single_channels)")
    if output_weight.shape[1] != gate_channels:
        raise ValueError("output_weight input channels must equal num_heads * head_size")
    require_same_device(
        "gate_attention_output",
        attention_values,
        single_values,
        gate_weight,
        output_weight,
    )
    validation = canonicalize_nested_gate_attention_metadata(
        attention.packed_offsets(dtype=torch.int64),
        attention_values.shape[0],
        num_groups,
        single.packed_offsets(dtype=torch.int64),
        single_values.shape[0],
        single_values.shape[1],
    )
    if is_compiling():
        torch._check_tensor_all(validation)
    output_values = _gate_attention_values(attention_values, single_values, gate_weight, output_weight)
    if implicit_single_group:
        output_values = output_values.squeeze(1)
    return single.packed_with_static_tail(output_values)


def _gate_attention_output_reference(
    attention: Tensor,
    single: Tensor,
    gate_weight: Tensor,
    output_weight: Tensor,
) -> Tensor:
    *prefix, num_heads, query_length, head_size = attention.shape
    attention_flat = attention.transpose(-3, -2).reshape(*prefix, query_length, num_heads * head_size)
    return F.linear(torch.sigmoid(F.linear(single, gate_weight)) * attention_flat, output_weight)
