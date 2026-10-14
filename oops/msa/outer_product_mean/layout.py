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

from math import prod

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import unsupported


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

    rank = msa_sizes.shape[1]
    if rank < 3:
        raise ValueError("outer_product_mean msa elements require (*prefix, M_i, N_i, C) axes")
    _require_sizes("msa", msa_sizes, rank)
    if msa_sizes.shape[0] == 0:
        raise ValueError("outer_product_mean does not serve an empty NestedTensor wrapper batch")
    if bool(torch.any(msa_sizes[:, -1] <= 0)):
        raise ValueError("outer_product_mean requires the msa channel dimension to be positive")
    if not bool(torch.all(msa_sizes[:, -1] == msa_sizes[0, -1])):
        raise ValueError("outer_product_mean requires one static msa channel size across nested samples")
    if not bool(torch.all(msa_sizes[:, :-3] == msa_sizes[0, :-3])):
        raise ValueError("outer_product_mean requires one static prefix across nested samples")
    if mask_sizes is not None:
        _require_sizes("msa_mask", mask_sizes, rank - 1)
        if not torch.equal(mask_sizes, msa_sizes[:, :-1]):
            raise ValueError("outer_product_mean nested msa_mask must match the prefix, M and N for every sample")

    return torch.ones((), dtype=torch.bool)


@torch.library.register_fake(validate_nested_outer_product_mean_metadata)
def _validate_nested_outer_product_mean_metadata_fake(
    msa_sizes: Tensor,
    mask_sizes: Tensor | None,
) -> Tensor:
    del mask_sizes
    return torch.empty((), device=msa_sizes.device, dtype=torch.bool)


@torch.library.custom_op(
    "ooperators::_empty_nested_outer_product_mean",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor anchor, Tensor msa_sizes, SymInt[] static_prefix) -> Tensor",
)
def _empty_nested_pair_values(anchor: Tensor, msa_sizes: Tensor, static_prefix: list[int]) -> Tensor:
    rows = int(msa_sizes[:, -2].square().sum())
    return anchor.new_empty((rows, *static_prefix, anchor.shape[0]))


@_empty_nested_pair_values.register_fake
def _empty_nested_pair_values_fake(anchor: Tensor, msa_sizes: Tensor, static_prefix: list[int]) -> Tensor:
    del msa_sizes
    rows = torch.library.get_ctx().new_dynamic_size(min=0)
    return anchor.new_empty((rows, *static_prefix, anchor.shape[0]))


def nested_pair_output(msa: NestedTensor, values: Tensor, lengths: Tensor, prefix_rank: int) -> NestedTensor:
    output = msa.packed_with_square_lengths(values, lengths)
    return output.permute(0, *range(3, prefix_rank + 3), 1, 2, prefix_rank + 3) if prefix_rank else output


def n_only_storage(value: NestedTensor, prefix_rank: int, channels: bool) -> bool:
    tail = (prefix_rank + 2,) if channels else ()
    if value.ragged_dims == (prefix_rank, prefix_rank + 1) and value.packed_dim_order == (
        prefix_rank,
        prefix_rank + 1,
        *range(prefix_rank),
        *tail,
    ):
        return False
    if value.ragged_dims == (prefix_rank + 1,) and value.packed_dim_order == (
        prefix_rank + 1,
        *range(prefix_rank),
        prefix_rank,
        *tail,
    ):
        return True
    raise unsupported(
        "outer_product_mean",
        "requires ragged-first packed M/N or fixed-M N-ragged elements",
        ragged_dims=value.ragged_dims,
        packed_dim_order=value.packed_dim_order,
    )


@torch.library.custom_op(
    "ooperators::_outer_product_mean_sequence_major_order",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor msa_sizes, SymInt cells) -> Tensor",
)
def sequence_major_order(msa_sizes: Tensor, cells: int) -> Tensor:
    sizes = msa_sizes[:, -3:].tolist()
    total_tokens = sum(tokens for _, tokens, _ in sizes)
    token_start = 0
    orders = []
    for depth, tokens, _ in sizes:
        rows = torch.arange(depth, dtype=torch.int64)[:, None] * total_tokens
        cols = torch.arange(tokens, dtype=torch.int64)[None, :] + token_start
        orders.append((rows + cols).reshape(-1))
        token_start += tokens
    order = torch.cat(orders)
    if order.numel() != cells:
        raise ValueError("outer_product_mean packed rows disagree with MSA element sizes")
    return order


@sequence_major_order.register_fake
def _sequence_major_order_fake(msa_sizes: Tensor, cells: int) -> Tensor:
    return msa_sizes.new_empty((cells,))


def canonical_packed_values(value: Tensor, sizes: Tensor, prefix_rank: int, n_only: bool, channels: bool) -> Tensor:
    prefix = value.shape[1 : 1 + prefix_rank]
    groups = prod(prefix)
    cells = value.shape[0] * (value.shape[-2 if channels else -1] if n_only else 1)
    packed = value.movedim(0, -2 if channels else -1)
    shape = (groups, cells, value.shape[-1]) if channels else (groups, cells)
    packed = packed.reshape(shape)
    if n_only:
        order = sequence_major_order(sizes, cells).to(device=value.device)
        packed = packed.index_select(1, order)
    output_shape = (groups * cells, value.shape[-1]) if channels else (groups * cells,)
    return packed.reshape(output_shape).contiguous()
