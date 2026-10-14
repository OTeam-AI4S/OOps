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

import math

import torch
import triton
from torch import Tensor

from oops.runtime import cuda_device_context

from . import kernel


def _epsilon(eps: float) -> tuple[float, int]:
    mantissa, exponent = math.frexp(math.sqrt(eps))
    return 2 * mantissa, exponent - 1 if mantissa else 0


def _cpu_state(vectors: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    work = vectors.to(torch.float64 if vectors.dtype is torch.float64 else torch.float32)
    maximum = work.abs().amax(-1)
    mantissa, exponent = torch.frexp(maximum)
    mantissa, exponent = mantissa * 2, exponent - 1
    epsilon_mantissa, epsilon_exponent = _epsilon(eps)
    use_epsilon = (epsilon_mantissa != 0) & (
        (maximum == 0)
        | (epsilon_exponent > exponent)
        | ((epsilon_exponent == exponent) & (epsilon_mantissa > mantissa))
    )
    mantissa = torch.where(use_epsilon, epsilon_mantissa, mantissa)
    exponent = torch.where(use_epsilon, epsilon_exponent, exponent)
    safe_scale = torch.where(mantissa == 0, 1, mantissa)
    scaled = torch.ldexp(work, -exponent.unsqueeze(-1)) / safe_scale.unsqueeze(-1)
    epsilon = torch.ldexp(torch.full_like(mantissa, epsilon_mantissa), epsilon_exponent - exponent) / safe_scale
    factor = (scaled.square().sum(-1) + epsilon.square()).sqrt()
    return mantissa, exponent, factor


def _config(width: int) -> tuple[int, int]:
    channels = min(triton.next_power_of_2(width), 256)
    return channels, max(1, 128 // channels)


def _run_forward(vectors: Tensor, eps: float) -> Tensor:
    shape = vectors.shape[:-1]
    rows = math.prod(shape)
    width = vectors.shape[-1]
    if rows == 0:
        return vectors.new_empty(shape)
    if width == 0:
        return torch.tensor(math.sqrt(eps), device=vectors.device, dtype=vectors.dtype).expand(shape).clone()
    if not vectors.is_cuda:
        mantissa, exponent, factor = _cpu_state(vectors, eps)
        output = torch.ldexp(mantissa * factor, exponent)
        output = torch.where(vectors.abs().amax(-1).isinf(), float("inf"), output)
        return output.to(vectors.dtype)
    with cuda_device_context(vectors.device):
        prepared = vectors.resolve_neg().contiguous()
        output = vectors.new_empty(shape)
        channels, block_rows = _config(width)
        epsilon_mantissa, epsilon_exponent = _epsilon(eps)
        kernel._forward[(triton.cdiv(rows, block_rows),)](
            prepared,
            output,
            rows,
            width,
            epsilon_mantissa,
            epsilon_exponent,
            channels,
            block_rows,
            vectors.dtype is torch.float64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output


def _run_backward(vectors: Tensor, upstream: Tensor, eps: float) -> Tensor:
    rows = math.prod(vectors.shape[:-1])
    width = vectors.shape[-1]
    if rows == 0 or width == 0:
        return vectors.new_empty(vectors.shape)
    if not vectors.is_cuda:
        mantissa, exponent, factor = _cpu_state(vectors, eps)
        denominator = torch.where(factor == 0, 1, mantissa * factor)
        work_dtype = torch.float64 if vectors.dtype is torch.float64 else torch.float32
        value_mantissa, value_exponent = torch.frexp(vectors.to(work_dtype))
        upstream_mantissa, upstream_exponent = torch.frexp(upstream.to(work_dtype))
        result = torch.ldexp(
            upstream_mantissa.unsqueeze(-1) * value_mantissa / denominator.unsqueeze(-1),
            upstream_exponent.unsqueeze(-1) + value_exponent - exponent.unsqueeze(-1),
        )
        return result.to(vectors.dtype).contiguous()
    with cuda_device_context(vectors.device):
        prepared = vectors.resolve_neg().contiguous()
        gradient = torch.empty_like(prepared)
        channels, block_rows = _config(width)
        epsilon_mantissa, epsilon_exponent = _epsilon(eps)
        kernel._backward[(triton.cdiv(rows, block_rows),)](
            prepared,
            upstream.resolve_neg().contiguous(),
            gradient,
            rows,
            width,
            epsilon_mantissa,
            epsilon_exponent,
            channels,
            block_rows,
            vectors.dtype is torch.float64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return gradient
