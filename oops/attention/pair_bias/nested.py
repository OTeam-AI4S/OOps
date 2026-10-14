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
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.attention.flash import MASK_KEY, MASK_NONE, MASK_PAIR
from oops.runtime import functorch_transforms_active, is_compiling, unsupported


def _validate_token_offsets(
    operand: str,
    offsets: Tensor,
    total_tokens: int,
) -> Tensor:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError(f"pair_bias_attention {operand} packed offsets must be a one-dimensional CPU int64 tensor")
    if offsets.numel() <= 1:
        raise unsupported("pair_bias_attention", "does not serve an empty NestedTensor batch")
    if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ValueError(f"pair_bias_attention {operand} packed offsets must be monotonic and start at zero")
    sizes = offsets[1:] - offsets[:-1]
    if bool(torch.any(sizes <= 0)):
        raise unsupported(
            "pair_bias_attention",
            f"requires every nested {operand} element to have length > 0",
        )
    if int(offsets[-1]) != total_tokens:
        raise ValueError(
            f"pair_bias_attention {operand} packed layout covers {int(offsets[-1])} tokens, "
            f"but packed values contain {total_tokens}"
        )
    return sizes


def _validate_pair_offsets(
    operand: str,
    row_offsets: Tensor,
    inner_offsets: Tensor,
    packed_offsets: Tensor,
    query_sizes: Tensor,
    key_sizes: Tensor,
) -> int:
    for level, offsets in (
        ("row", row_offsets),
        ("inner", inner_offsets),
        ("packed", packed_offsets),
    ):
        if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
            raise ValueError(
                f"pair_bias_attention {operand} {level} offsets must be a one-dimensional CPU int64 tensor"
            )
        if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
            raise ValueError(f"pair_bias_attention {operand} {level} offsets must be monotonic and start at zero")
    if row_offsets.shape[0] != query_sizes.shape[0] + 1 or packed_offsets.shape != row_offsets.shape:
        raise ValueError(f"pair_bias_attention {operand} must have one packed boundary per sample")
    expected_query_offsets = torch.cat((query_sizes.new_zeros((1,)), query_sizes.cumsum(0)))
    if not torch.equal(row_offsets, expected_query_offsets):
        raise ValueError(f"pair_bias_attention {operand} query axis must match Q_i for every sample")
    if inner_offsets.numel() != int(row_offsets[-1]) + 1:
        raise ValueError(f"pair_bias_attention {operand} inner offsets do not cover every query row")
    sample_indices = torch.repeat_interleave(
        torch.arange(query_sizes.numel(), dtype=torch.int64),
        query_sizes,
    )
    expected_key_widths = key_sizes.index_select(0, sample_indices)
    if not torch.equal(inner_offsets[1:] - inner_offsets[:-1], expected_key_widths):
        raise ValueError(f"pair_bias_attention {operand} key axis must match K_i for every query row")
    expected_packed = inner_offsets.index_select(0, row_offsets)
    if not torch.equal(packed_offsets, expected_packed):
        raise ValueError(f"pair_bias_attention {operand} query/key axes must match Q_i and K_i for every sample")
    return int(packed_offsets[-1])


_NESTED_PAIR_BIAS_METADATA_SCHEMA = (
    "(Tensor query_offsets, SymInt query_total_tokens, Tensor key_offsets, SymInt key_total_tokens, "
    "Tensor value_offsets, SymInt value_total_tokens, SymInt num_groups, SymInt num_heads, "
    "Tensor? mask_rows, Tensor? mask_inner, Tensor? mask_packed, int mask_kind, "
    "SymInt mask_total_values, SymInt mask_groups, Tensor? bias_rows, Tensor? bias_inner, "
    "Tensor? bias_packed, SymInt bias_total_values, SymInt bias_groups, SymInt bias_heads) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_canonicalize_nested_pair_bias_attention_metadata",
    mutates_args=(),
    device_types="cpu",
    schema=_NESTED_PAIR_BIAS_METADATA_SCHEMA,
)
def _canonicalize_nested_pair_bias_attention_metadata(
    query_offsets: Tensor,
    query_total_tokens: int,
    key_offsets: Tensor,
    key_total_tokens: int,
    value_offsets: Tensor,
    value_total_tokens: int,
    num_groups: int,
    num_heads: int,
    mask_rows: Tensor | None,
    mask_inner: Tensor | None,
    mask_packed: Tensor | None,
    mask_kind: int,
    mask_total_values: int,
    mask_groups: int,
    bias_rows: Tensor | None,
    bias_inner: Tensor | None,
    bias_packed: Tensor | None,
    bias_total_values: int,
    bias_groups: int,
    bias_heads: int,
) -> Tensor:

    return _validate_nested_pair_bias_attention_metadata(
        query_offsets,
        query_total_tokens,
        key_offsets,
        key_total_tokens,
        value_offsets,
        value_total_tokens,
        num_groups,
        num_heads,
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


def _validate_nested_pair_bias_attention_metadata(
    query_offsets: Tensor,
    query_total_tokens: int,
    key_offsets: Tensor,
    key_total_tokens: int,
    value_offsets: Tensor,
    value_total_tokens: int,
    num_groups: int,
    num_heads: int,
    mask_rows: Tensor | None,
    mask_inner: Tensor | None,
    mask_packed: Tensor | None,
    mask_kind: int,
    mask_total_values: int,
    mask_groups: int,
    bias_rows: Tensor | None,
    bias_inner: Tensor | None,
    bias_packed: Tensor | None,
    bias_total_values: int,
    bias_groups: int,
    bias_heads: int,
) -> Tensor:

    query_sizes = _validate_token_offsets("query", query_offsets, query_total_tokens)
    key_sizes = _validate_token_offsets("key", key_offsets, key_total_tokens)
    _validate_token_offsets("value", value_offsets, value_total_tokens)
    if key_offsets.shape != query_offsets.shape:
        raise ValueError("pair_bias_attention query and key must contain the same number of samples")
    if not torch.equal(value_offsets, key_offsets) or value_total_tokens != key_total_tokens:
        raise ValueError("pair_bias_attention value layout must match the key layout per sample")
    if num_groups <= 0 or num_heads <= 0:
        raise ValueError("pair_bias_attention nested Q/K/V must retain positive sample and head axes")

    if mask_rows is None:
        if (
            mask_inner is not None
            or mask_packed is not None
            or mask_kind != MASK_NONE
            or mask_total_values != 0
            or mask_groups != 0
        ):
            raise ValueError("pair_bias_attention mask metadata is incomplete")
    else:
        if mask_packed is None:
            raise ValueError("pair_bias_attention mask metadata is incomplete")
        if mask_groups not in (1, num_groups):
            raise ValueError("pair_bias_attention mask sample axis must be uniformly 1 or S")
        if mask_kind == MASK_KEY:
            if (
                mask_inner is not None
                or not torch.equal(mask_rows, key_offsets)
                or not torch.equal(mask_packed, key_offsets)
            ):
                raise ValueError("pair_bias_attention key mask lengths must match K_i per sample")
            mask_positions = key_total_tokens
        elif mask_kind == MASK_PAIR:
            if mask_inner is None:
                raise ValueError("pair_bias_attention pair mask metadata is incomplete")
            mask_positions = _validate_pair_offsets(
                "pair mask",
                mask_rows,
                mask_inner,
                mask_packed,
                query_sizes,
                key_sizes,
            )
        else:
            raise ValueError("pair_bias_attention mask must use key or pair form")
        if mask_total_values != mask_positions * mask_groups:
            raise ValueError("pair_bias_attention mask packed values do not match its declared layout")

    if bias_rows is None:
        if (
            bias_inner is not None
            or bias_packed is not None
            or bias_total_values != 0
            or bias_groups != 0
            or bias_heads != 0
        ):
            raise ValueError("pair_bias_attention pair_bias metadata is incomplete")
    else:
        if bias_inner is None or bias_packed is None:
            raise ValueError("pair_bias_attention pair_bias metadata is incomplete")
        if bias_groups not in (1, num_groups):
            raise ValueError("pair_bias_attention pair_bias sample axis must be uniformly 1 or S")
        if bias_heads not in (1, num_heads):
            raise ValueError("pair_bias_attention pair_bias head axis must be uniformly 1 or H")
        bias_positions = _validate_pair_offsets(
            "pair_bias",
            bias_rows,
            bias_inner,
            bias_packed,
            query_sizes,
            key_sizes,
        )
        if bias_total_values != bias_positions * bias_groups * bias_heads:
            raise ValueError("pair_bias_attention pair_bias packed values do not match its declared layout")

    return torch.ones((), dtype=torch.bool)


@torch.library.register_fake(_canonicalize_nested_pair_bias_attention_metadata)
def _canonicalize_nested_pair_bias_attention_metadata_fake(
    query_offsets: Tensor,
    query_total_tokens: int,
    key_offsets: Tensor,
    key_total_tokens: int,
    value_offsets: Tensor,
    value_total_tokens: int,
    num_groups: int,
    num_heads: int,
    mask_rows: Tensor | None,
    mask_inner: Tensor | None,
    mask_packed: Tensor | None,
    mask_kind: int,
    mask_total_values: int,
    mask_groups: int,
    bias_rows: Tensor | None,
    bias_inner: Tensor | None,
    bias_packed: Tensor | None,
    bias_total_values: int,
    bias_groups: int,
    bias_heads: int,
) -> Tensor:
    del (
        query_total_tokens,
        key_offsets,
        key_total_tokens,
        value_offsets,
        value_total_tokens,
        num_groups,
        num_heads,
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
    return torch.empty((), device=query_offsets.device, dtype=torch.bool)


def canonicalize_nested_pair_bias_attention_metadata(
    query_offsets: Tensor,
    query_total_tokens: int,
    key_offsets: Tensor,
    key_total_tokens: int,
    value_offsets: Tensor,
    value_total_tokens: int,
    num_groups: int,
    num_heads: int,
    mask_rows: Tensor | None,
    mask_inner: Tensor | None,
    mask_packed: Tensor | None,
    mask_kind: int,
    mask_total_values: int,
    mask_groups: int,
    bias_rows: Tensor | None,
    bias_inner: Tensor | None,
    bias_packed: Tensor | None,
    bias_total_values: int,
    bias_groups: int,
    bias_heads: int,
) -> Tensor:

    metadata = (
        query_offsets,
        key_offsets,
        value_offsets,
        mask_rows,
        mask_inner,
        mask_packed,
        bias_rows,
        bias_inner,
        bias_packed,
    )
    if (
        not is_compiling()
        and all(tensor is None or type(tensor) is Tensor for tensor in metadata)
        and not is_in_torch_dispatch_mode()
        and not functorch_transforms_active()
    ):
        return _validate_nested_pair_bias_attention_metadata(
            query_offsets,
            query_total_tokens,
            key_offsets,
            key_total_tokens,
            value_offsets,
            value_total_tokens,
            num_groups,
            num_heads,
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
    return _canonicalize_nested_pair_bias_attention_metadata(
        query_offsets,
        query_total_tokens,
        key_offsets,
        key_total_tokens,
        value_offsets,
        value_total_tokens,
        num_groups,
        num_heads,
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
