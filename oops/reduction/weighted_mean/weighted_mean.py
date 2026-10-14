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
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import require_no_forward_mode, require_same_device, unsupported

from .triton.ops import run_weighted_mean


def weighted_mean(values: Tensor | NestedTensor, weights: Tensor | NestedTensor) -> Tensor | NestedTensor:
    r"""Compute the weighted mean over the final observation axis.

    Args:
        values: CPU or CUDA float16, bfloat16, float32, or float64 observations,
            or a batch-first DanLing NestedTensor with a final ragged axis
            and an optional leading ragged sample axis.
        weights: Finite nonnegative coefficients or Boolean/integer masks,
            broadcastable with the observations, on the same device.

    Returns:
        ``sum(weights * values) / sum(weights)`` after broadcasting, with the
        final axis removed. Floating operands follow ordinary dtype promotion;
        Boolean/integer weights preserve the values' dtype. Empty observations
        or zero total weight return NaN with connected zero gradients for both
        operands. Nested inputs reduce true observation lengths. A remaining ragged sample axis is preserved
        in the returned NestedTensor; otherwise the result is dense
        ``(B, *prefix)``. Broadcasting must preserve nested sample lengths.

    Raises:
        ValueError: Devices differ, operands have no observation axis, or nested
            observation lengths disagree.
        RuntimeError: Operand shapes cannot broadcast.
        OOpsUnsupportedError: A dtype, device, nested layout, or AD mode is unsupported.

    Notes:
        Nonfinite observations at zero weight do not contribute to the result.
        Positive-weight nonfinite observations propagate.

    Examples:
        >>> import torch
        >>> from oops import weighted_mean
        >>> weighted_mean(torch.tensor([2., 8.]), torch.tensor([3., 1.])).item()
        3.5
    """
    require_no_forward_mode("weighted_mean", values, weights)
    require_same_device("weighted_mean", values, weights)
    if values.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise unsupported("weighted_mean", "requires floating observations", dtype=values.dtype)
    if weights.is_complex() or weights.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise unsupported("weighted_mean", "requires real floating coefficients or masks", dtype=weights.dtype)
    if values.device.type not in ("cpu", "cuda"):
        raise unsupported("weighted_mean", "requires CPU or CUDA observations", device=values.device)
    if isinstance(values, NestedTensor) or isinstance(weights, NestedTensor):
        from .nested import nested_weighted_mean

        return nested_weighted_mean(values, weights)
    if max(values.ndim, weights.ndim) == 0:
        raise ValueError("weighted_mean requires a final observation axis")
    return run_weighted_mean(values, weights)
