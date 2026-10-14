from __future__ import annotations

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import functorch_transforms_active, is_compiling, unsupported


def _validate_token_offsets(
    operand: str,
    offsets: Tensor,
    total_tokens: int,
) -> Tensor:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError(f"gate_attention_output {operand} packed offsets must be a one-dimensional CPU int64 tensor")
    if offsets.numel() <= 1:
        raise unsupported("gate_attention_output", "does not serve an empty NestedTensor batch")
    if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ValueError(f"gate_attention_output {operand} packed offsets must be monotonic and start at zero")
    sizes = offsets[1:] - offsets[:-1]
    if bool(torch.any(sizes <= 0)):
        raise unsupported(
            "gate_attention_output",
            f"requires every nested {operand} element to have length > 0",
        )
    if int(offsets[-1]) != total_tokens:
        raise ValueError(
            f"gate_attention_output {operand} packed layout covers {int(offsets[-1])} tokens, "
            f"but packed values contain {total_tokens}"
        )
    return sizes


_NESTED_GATE_METADATA_SCHEMA = (
    "(Tensor attention_offsets, SymInt attention_total_tokens, SymInt attention_groups, "
    "Tensor single_offsets, SymInt single_total_tokens, SymInt single_groups) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_canonicalize_nested_gate_attention_metadata",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_GATE_METADATA_SCHEMA,
)
def _canonicalize_nested_gate_attention_metadata(
    attention_offsets: Tensor,
    attention_total_tokens: int,
    attention_groups: int,
    single_offsets: Tensor,
    single_total_tokens: int,
    single_groups: int,
) -> Tensor:

    return _validate_nested_gate_attention_metadata(
        attention_offsets, attention_total_tokens, attention_groups, single_offsets, single_total_tokens, single_groups
    )


def _validate_nested_gate_attention_metadata(
    attention_offsets: Tensor,
    attention_total_tokens: int,
    attention_groups: int,
    single_offsets: Tensor,
    single_total_tokens: int,
    single_groups: int,
) -> Tensor:
    _validate_token_offsets("attention", attention_offsets, attention_total_tokens)
    _validate_token_offsets("single", single_offsets, single_total_tokens)
    if not torch.equal(attention_offsets, single_offsets):
        raise ValueError("gate_attention_output nested single lengths must match attention query lengths")
    if attention_groups != single_groups:
        raise ValueError("gate_attention_output nested single and attention must share S")
    return torch.ones((), dtype=torch.bool)


@torch.library.register_fake(_canonicalize_nested_gate_attention_metadata)
def _canonicalize_nested_gate_attention_metadata_fake(
    attention_offsets: Tensor,
    attention_total_tokens: int,
    attention_groups: int,
    single_offsets: Tensor,
    single_total_tokens: int,
    single_groups: int,
) -> Tensor:
    del (
        attention_total_tokens,
        attention_groups,
        single_offsets,
        single_total_tokens,
        single_groups,
    )
    return torch.empty((), device=attention_offsets.device, dtype=torch.bool)


def canonicalize_nested_gate_attention_metadata(
    attention_offsets: Tensor,
    attention_total_tokens: int,
    attention_groups: int,
    single_offsets: Tensor,
    single_total_tokens: int,
    single_groups: int,
) -> Tensor:

    if (
        not is_compiling()
        and type(attention_offsets) is Tensor
        and type(single_offsets) is Tensor
        and not is_in_torch_dispatch_mode()
        and not functorch_transforms_active()
    ):
        return _validate_nested_gate_attention_metadata(
            attention_offsets,
            attention_total_tokens,
            attention_groups,
            single_offsets,
            single_total_tokens,
            single_groups,
        )
    return _canonicalize_nested_gate_attention_metadata(
        attention_offsets, attention_total_tokens, attention_groups, single_offsets, single_total_tokens, single_groups
    )
