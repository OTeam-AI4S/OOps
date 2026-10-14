from __future__ import annotations

from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import require_no_forward_mode, require_same_device

from ..gate.gate_attention_output import gate_attention_output
from .pair_bias_attention import pair_bias_attention


def gated_pair_bias_attention(
    single: Tensor | NestedTensor,
    query: Tensor | NestedTensor,
    key: Tensor | NestedTensor,
    value: Tensor | NestedTensor,
    mask: Tensor | NestedTensor | None = None,
    pair_bias: Tensor | NestedTensor | None = None,
    gate_weight: Tensor | None = None,
    output_weight: Tensor | None = None,
    *,
    scale: float | None = None,
) -> Tensor | NestedTensor:
    r"""
    Apply pair-bias attention, sigmoid gating, and an output projection.

    This composes ``pair_bias_attention`` with ``gate_attention_output``
    and returns one projected feature vector per query token.

    Args:
        single: Dense ``(*prefix, Q, C)`` gating input or matching nested
            ``(Q_i, C)`` or ``(S, Q_i, C)`` elements.
        query: Head-major ``(*prefix, H, Q, D)`` queries, with at most batch and
            group prefix axes, or nested queries accepted by pair-bias attention.
        key: Keys matching the query prefix, heads, and channels, with key length
            ``K`` in place of ``Q``.
        value: Values matching the key shape, dtype, device, and nested topology.
        mask: Optional bool key or pair keep-mask accepted by pair-bias attention.
            ``None`` keeps all keys; an all-true mask is equivalent.
        pair_bias: Required additive bias accepted by pair-bias attention. Pass by
            keyword to omit the optional mask without a positional placeholder.
        gate_weight: Required matrix ``(H * D, C)``.
        output_weight: Required matrix ``(O, H * D)``.
        scale: Python scalar multiplying Q.K before bias addition. ``None`` uses
            ``D ** -0.5``.

    Returns:
        Gated attention values ``(*prefix, Q, O)`` on the input device, with dtype
        determined by the PyTorch projection operations and active autocast. A
        nested result preserves the single topology with channel width ``O``.

    Raises:
        ValueError: Required bias/weights are absent, or shapes, ranks, devices,
            representations, or nested topologies disagree.
        OOpsUnsupportedError: Attention inputs or nested layouts are unsupported,
            or forward-mode AD is requested through Q/K/V or pair bias, or captured
            CUDA matmul precision no longer matches the active policy.

    Notes:
        Dense Q/K/V have rank 3, 4, or 5; all operands share a device. Attention
        dtype, bias, mask, and precision rules follow ``pair_bias_attention``.
        The gate stage preserves its projection and sigmoid rounding boundaries.

        First-order reverse mode differentiates the attention inputs, single, and
        weights. Forward-mode tangents confined to single, gate weights, or output
        weights are supported; tangents on Q/K/V or pair bias are rejected. Nested
        attention uses token-major packed storage, with single elements matching
        its query lengths and group counts.

    Examples:
        >>> import torch
        >>> from oops import gated_pair_bias_attention
        >>> if torch.cuda.is_available():
        ...     q = torch.ones(1, 2, 16, device="cuda")
        ...     single = torch.zeros(2, 16, device="cuda")
        ...     bias = torch.zeros(1, 2, 2, device="cuda")
        ...     weight = torch.eye(16, device="cuda")
        ...     result = gated_pair_bias_attention(
        ...         single, q, q, q, pair_bias=bias,
        ...         gate_weight=weight, output_weight=weight,
        ...     )
        ...     assert result.shape == (2, 16)
    """

    if pair_bias is None:
        raise ValueError("pair_bias is required")
    if gate_weight is None:
        raise ValueError("gate_weight is required")
    if output_weight is None:
        raise ValueError("output_weight is required")
    if (type(query) is not Tensor or type(single) is not Tensor) and isinstance(query, NestedTensor) != isinstance(
        single, NestedTensor
    ):
        raise ValueError("gated_pair_bias_attention query and single must both be dense or both NestedTensor")
    if isinstance(query, NestedTensor):
        require_no_forward_mode(
            "gated_pair_bias_attention",
            query.concat,
            key.concat if isinstance(key, NestedTensor) else key,
            value.concat if isinstance(value, NestedTensor) else value,
            pair_bias.concat if isinstance(pair_bias, NestedTensor) else pair_bias,
        )
    else:
        require_no_forward_mode("gated_pair_bias_attention", query, key, value, pair_bias)
    require_same_device(
        "gated_pair_bias_attention",
        query,
        key,
        value,
        mask,
        pair_bias,
        single,
        gate_weight,
        output_weight,
    )

    attention = pair_bias_attention(query, key, value, mask, pair_bias, scale=scale)
    return gate_attention_output(attention, single, gate_weight, output_weight)
