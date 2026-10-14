from __future__ import annotations

from torch import Tensor


def _check_gate_output(attention: Tensor, single: Tensor, gate_weight: Tensor, output_weight: Tensor) -> None:
    if attention.dim() not in (3, 4, 5):
        raise ValueError(
            "attention must have layout (num_heads, query_length, head_size) under optional batch and sample axes"
        )
    if single.dim() != attention.dim() - 1:
        raise ValueError("single must have layout (..., query_length, single_channels)")
    if gate_weight.dim() != 2 or output_weight.dim() != 2:
        raise ValueError("gate_weight and output_weight must be rank-2 matrices")

    *prefix, num_heads, query_length, head_size = attention.shape
    expected_single_prefix = (*prefix, query_length)
    gate_channels = num_heads * head_size
    if tuple(single.shape[:-1]) != expected_single_prefix:
        raise ValueError("single must share the attention leading axes and its query_length")
    if gate_weight.shape != (gate_channels, single.shape[-1]):
        raise ValueError("gate_weight must have shape (num_heads * head_size, single_channels)")
    if output_weight.shape[1] != gate_channels:
        raise ValueError("output_weight input channels must equal num_heads * head_size")
