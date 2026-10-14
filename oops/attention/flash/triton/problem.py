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

from dataclasses import dataclass
from enum import Enum

import torch
import triton
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true

from oops.core.launch import validate_positive_parameters
from oops.runtime import HALF_DTYPES, STANDARD_FLOAT_DTYPES, is_compiling, tensors_share_device

from ..mask import MASK_KEY, MASK_NONE, pair_mask_kind
from ..qkv import degenerate_shape, degenerate_strides

# tl.dot requires each matrix dimension to be at least 16.
MIN_BLOCK_CHANNEL = 16

MAX_HEAD_SIZE = 256
SUPPORTED_DTYPES = STANDARD_FLOAT_DTYPES


def block_channel_size(head_size: int) -> int:
    return max(MIN_BLOCK_CHANNEL, triton.next_power_of_2(head_size))


def resolve_scale(head_size: int, scale: float | None) -> float:
    return float(head_size**-0.5 if scale is None else scale)


@dataclass(frozen=True, slots=True)
class AttentionLaunchContext:

    num_heads: int
    query_length: int
    key_length: int
    head_size: int
    mask_kind: int = 0
    element_size: int = 4

    def __post_init__(self) -> None:
        validate_positive_parameters(
            num_heads=self.num_heads,
            query_length=self.query_length,
            key_length=self.key_length,
            head_size=self.head_size,
            element_size=self.element_size,
        )

    @property
    def score_elements(self) -> int:

        return self.query_length * self.key_length

    @property
    def block_channel(self) -> int:

        return block_channel_size(self.head_size)

    @property
    def has_mask(self) -> bool:

        return self.mask_kind != 0


Strides4 = tuple[int, int, int, int]
Strides5 = tuple[int, int, int, int, int]


class AttentionTensorLayout(Enum):

    HEAD_TOKEN = "head_token"


def attention_tensor_strides(layout: AttentionTensorLayout, tensor: Tensor) -> Strides5:
    r"""Logical stride order: ``(batch, group, token, head, channel)``."""

    del layout
    batch, group, head, token, channel = degenerate_strides(tensor, 5)
    return batch, group, token, head, channel


def attention_lse_strides(tensor: Tensor | None) -> Strides4:
    r"""Logical stride order: ``(batch, group, head, token)``."""

    if tensor is None:
        return 0, 0, 0, 0
    batch, group, head, token = degenerate_strides(tensor, 4)
    return batch, group, head, token


def attention_bias_strides(tensor: Tensor | None) -> Strides5:
    r"""Logical stride order: ``(batch, group, head, query, key)``."""

    if tensor is None:
        return 0, 0, 0, 0, 0
    batch, group, head, query, key = degenerate_strides(tensor, 5)
    return batch, group, head, query, key


def attention_shared_bias_strides(tensor: Tensor) -> Strides4:

    batch, _group, head, query, key = degenerate_strides(tensor, 5)
    return batch, head, query, key


@dataclass(frozen=True, slots=True)
class AttentionProblem:

    batch_size: int
    num_groups: int
    context: AttentionLaunchContext
    tensor_layout: AttentionTensorLayout
    lse_shape: tuple[int, ...]
    query_strides: Strides5
    key_strides: Strides5
    value_strides: Strides5
    mask_strides: Strides4
    bias_strides: Strides5
    shared_bias_strides: Strides4
    bias_groups: int

    @property
    def num_heads(self) -> int:
        return self.context.num_heads

    @property
    def query_length(self) -> int:
        return self.context.query_length

    @property
    def key_length(self) -> int:
        return self.context.key_length

    @property
    def head_size(self) -> int:
        return self.context.head_size

    @property
    def mask_kind(self) -> int:
        return self.context.mask_kind

    @property
    def batch_group_head_count(self) -> int:
        return self.batch_size * self.num_groups * self.num_heads


def supports_head_major_qkv(query: Tensor, key: Tensor, value: Tensor) -> bool:

    return (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and tensors_share_device(query, key, value)
        and query.dim() in (4, 5)
        and key.dim() == query.dim()
        and value.dim() == query.dim()
        and query.dtype in SUPPORTED_DTYPES
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.shape[:-2] == key.shape[:-2] == value.shape[:-2]
        and key.shape[-1] == query.shape[-1]
        and value.shape[-1] == query.shape[-1]
        and value.shape[-2] == key.shape[-2]
        and query.shape[-3] > 0
        and query.shape[-2] > 0
        and key.shape[-2] > 0
        and 0 < query.shape[-1] <= MAX_HEAD_SIZE
    )


def optional_tensor(tensor: Tensor | None, placeholder: Tensor) -> Tensor:

    return placeholder if tensor is None else tensor


def supports_pair_mask(query: Tensor, key: Tensor, mask: Tensor | None) -> bool:

    if mask is None:
        return True
    kind = pair_mask_kind(query, mask)
    if not (mask.is_cuda and tensors_share_device(query, mask) and mask.dtype == torch.bool and kind != MASK_NONE):
        return False
    trailing = (key.shape[-2],) if kind == MASK_KEY else (query.shape[-2], key.shape[-2])
    batch, samples = degenerate_shape(query, 5)[:2]
    shape = degenerate_shape(mask, len(trailing) + 2)
    return shape[0] == batch and shape[1] in (1, samples) and shape[2:] == trailing


def pair_mask_strides(mask_kind: int, mask: Tensor | None) -> tuple[int, int, int, int]:

    if mask_kind == MASK_NONE or mask is None:
        return (0, 0, 0, 0)
    if mask_kind == MASK_KEY:
        batch_stride, sample_stride, key_stride = degenerate_strides(mask, 3)
        samples = degenerate_shape(mask, 3)[1]
        query_stride = 0
    else:
        batch_stride, sample_stride, query_stride, key_stride = degenerate_strides(mask, 4)
        samples = degenerate_shape(mask, 4)[1]
    return (
        batch_stride,
        0 if statically_known_true(samples == 1) else sample_stride,
        query_stride,
        key_stride,
    )


def supports_pair_bias(query: Tensor, key: Tensor, pair_bias: Tensor | None) -> bool:

    if pair_bias is None:
        return True
    if pair_bias.dim() != query.dim():
        return False
    target = degenerate_shape(query, 5)[:-1] + (key.shape[-2],)
    shape = degenerate_shape(pair_bias, 5)
    batch_stride = degenerate_strides(pair_bias, 5)[0]
    autocast_float32_bias = (
        query.dtype in HALF_DTYPES
        and pair_bias.dtype == torch.float32
        and pair_bias.device == query.device
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == query.dtype
    )
    if not (
        pair_bias.is_cuda
        and tensors_share_device(query, pair_bias)
        and (pair_bias.dtype == query.dtype or autocast_float32_bias)
    ):
        return False
    if is_compiling():
        torch._check(shape[0] == target[0])
        torch._check((target[0] == 1) | (batch_stride != 0))
        for size, target_size in zip(shape[1:], target[1:]):
            torch._check((size == 1) | (size == target_size))
        return True
    return (
        shape[0] == target[0]
        and (target[0] == 1 or batch_stride != 0)
        and all(size in (1, target_size) for size, target_size in zip(shape[1:], target[1:]))
    )


def expand_pair_bias(query: Tensor, key: Tensor, pair_bias: Tensor | None) -> Tensor | None:
    r"""Keep the sample axis at ``S|1``; expansion reduces only head/query/key sharing."""

    if pair_bias is None:
        return None
    target = (*pair_bias.shape[:-3], query.shape[-3], query.shape[-2], key.shape[-2])
    if pair_bias.shape == target and pair_bias.is_contiguous():
        return pair_bias
    return pair_bias.contiguous().expand(*target)
