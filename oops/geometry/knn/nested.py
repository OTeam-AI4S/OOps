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

from oops.runtime import functorch_transforms_active, is_compiling

_SCHEMA = "(Tensor coordinate_offsets, Tensor mask_offsets, SymInt coordinate_total, " "SymInt mask_total) -> Tensor"


def _validate_offsets(name: str, offsets: Tensor, total: int) -> None:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError(f"masked_knn {name} offsets must be a one-dimensional CPU int64 tensor")
    if offsets.numel() == 0:
        raise ValueError(f"masked_knn {name} offsets must include the initial zero")
    if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ValueError(f"masked_knn {name} offsets must be monotonic and start at zero")
    if int(offsets[-1]) != total:
        raise ValueError(f"masked_knn {name} offsets do not cover their packed values")


def _validate_metadata(
    coordinate_offsets: Tensor,
    mask_offsets: Tensor,
    coordinate_total: int,
    mask_total: int,
) -> None:

    _validate_offsets("coordinates", coordinate_offsets, coordinate_total)
    _validate_offsets("mask", mask_offsets, mask_total)
    if not torch.equal(coordinate_offsets, mask_offsets):
        raise ValueError("masked_knn coordinates and mask must have matching per-sample lengths")


@torch.library.custom_op(
    "ooperators::_validate_nested_masked_knn_metadata",
    mutates_args=(),
    device_types="cpu",
    schema=_SCHEMA,
)
def _validate_nested_masked_knn_metadata(
    coordinate_offsets: Tensor,
    mask_offsets: Tensor,
    coordinate_total: int,
    mask_total: int,
) -> Tensor:
    _validate_metadata(coordinate_offsets, mask_offsets, coordinate_total, mask_total)
    return torch.ones((), dtype=torch.bool, device=coordinate_offsets.device)


@torch.library.register_fake(_validate_nested_masked_knn_metadata)
def _validate_nested_masked_knn_metadata_fake(
    coordinate_offsets: Tensor,
    mask_offsets: Tensor,
    coordinate_total: int,
    mask_total: int,
) -> Tensor:
    del mask_offsets, coordinate_total, mask_total
    return torch.empty((), dtype=torch.bool, device=coordinate_offsets.device)


def validate_nested_masked_knn_metadata(
    coordinate_offsets: Tensor,
    mask_offsets: Tensor,
    coordinate_total: int,
    mask_total: int,
) -> Tensor | None:
    offsets = (coordinate_offsets, mask_offsets)
    if (
        not is_compiling()
        and all(
            type(value) is Tensor and value.device.type == "cpu" and value.layout is torch.strided for value in offsets
        )
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(offsets)
        and not functorch_transforms_active()
    ):
        _validate_metadata(coordinate_offsets, mask_offsets, coordinate_total, mask_total)
        return None
    return _validate_nested_masked_knn_metadata(coordinate_offsets, mask_offsets, coordinate_total, mask_total)
