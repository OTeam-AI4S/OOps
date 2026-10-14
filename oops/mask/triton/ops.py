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

from oops.runtime import is_compiling, is_external_fake

from .generic import _msa_impl, _pair_impl

torch.library.define(
    "ooperators::_packed_msa_mask", "(Tensor values, Tensor sizes, Tensor offsets, bool pair) -> Tensor"
)


@torch.library.impl("ooperators::_packed_msa_mask", "cuda")
def _packed_msa_mask(values: Tensor, sizes: Tensor, offsets: Tensor, pair: bool) -> Tensor:
    return _msa_impl(values, sizes, offsets, pair)


@torch.library.register_fake("ooperators::_packed_msa_mask")
def _packed_msa_mask_fake(values: Tensor, sizes: Tensor, offsets: Tensor, pair: bool) -> Tensor:
    del offsets, pair
    total = torch.library.get_ctx().new_dynamic_size(min=0)
    return values.new_empty((total,), dtype=torch.bool)


def packed_msa_any(values: Tensor, sizes: Tensor, offsets: Tensor) -> Tensor:
    execute = torch.ops.ooperators._packed_msa_mask.default if is_compiling() or is_external_fake(values) else _msa_impl
    return execute(values, sizes, offsets, False)


def packed_msa_pair_mask(values: Tensor, sizes: Tensor, offsets: Tensor) -> Tensor:
    execute = torch.ops.ooperators._packed_msa_mask.default if is_compiling() or is_external_fake(values) else _msa_impl
    return execute(values, sizes, offsets, True)


torch.library.define("ooperators::_packed_pair_mask", "(Tensor values, Tensor offsets) -> Tensor")


@torch.library.impl("ooperators::_packed_pair_mask", "cuda")
def _packed_pair_mask(values: Tensor, offsets: Tensor) -> Tensor:
    return _pair_impl(values, offsets)


@torch.library.register_fake("ooperators::_packed_pair_mask")
def _packed_pair_mask_fake(values: Tensor, offsets: Tensor) -> Tensor:
    total = torch.library.get_ctx().new_dynamic_size(min=0)
    return values.new_empty((total,), dtype=torch.bool)


def packed_pair_mask(values: Tensor, offsets: Tensor) -> Tensor:
    execute = (
        torch.ops.ooperators._packed_pair_mask.default if is_compiling() or is_external_fake(values) else _pair_impl
    )
    return execute(values, offsets)
