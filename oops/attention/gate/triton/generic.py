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

from contextlib import nullcontext
from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.core import TargetDescriptor
from oops.core._math import _next_power_of_two
from oops.runtime import (
    FLOAT32_DOT_TF32X3,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps,
    cuda_device_caps_if_needed,
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    matmul_precision,
    shared_memory_budget,
)

from . import kernel
from .profiles import SCHEDULES
from .schedule import GateMultiplyProblem, GatePhaseProfile, GateProjectionConfig, GateProjectionProblem

_OPERATOR = "gate_attention_output"


_FLOAT32_DOT = FLOAT32_DOT_TF32X3


def _kernel_operand(tensor: Tensor) -> Tensor:
    # Raw pointer loads cannot observe logical view bits.
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    return tensor


def _geometry(attention: Tensor) -> tuple[int, int, int, int, tuple[int, ...]]:

    shape = (*(1 for _ in range(5 - attention.dim())), *attention.shape)
    strides = (*(0 for _ in range(5 - attention.dim())), *attention.stride())
    return shape[1], shape[2], shape[3], shape[4], strides


def _select(profile: GatePhaseProfile, problem, device: torch.device):
    return profile.select(problem, cuda_device_caps_if_needed(device, profile.target_changes_config(problem)))


def _attention_gate_multiply_forward_impl(attention: Tensor, logits: Tensor) -> Tensor:
    with cuda_device_context(attention.device):
        attention, logits = _kernel_operand(attention), _kernel_operand(logits)
        output = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
        groups, tokens, heads, width, strides = _geometry(attention)
        rows = logits.numel() // (heads * width)
        if rows:
            problem = GateMultiplyProblem(rows, heads, width, logits.dtype)
            tile = _select(SCHEDULES.multiply_forward, problem, attention.device)
            block_rows, block_columns = tile.blocks(problem.gate_channels)
            kernel._gate_forward.run(
                attention,
                logits,
                output,
                rows,
                tokens,
                groups,
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                strides[4],
                heads,
                width,
                block_rows,
                block_columns,
                num_warps=tile.num_warps,
                grid=tile.grid(rows, problem.gate_channels),
                warmup=False,
            )
        return output


def _attention_gate_multiply_backward_impl(
    grad: Tensor, attention: Tensor, logits: Tensor, need_attention: bool, need_logits: bool
) -> tuple[Tensor | None, Tensor | None]:

    with cuda_device_context(attention.device):
        grad, logits = _kernel_operand(grad), _kernel_operand(logits).contiguous()
        if need_logits:
            attention = _kernel_operand(attention)
        grad_attention = torch.empty_like(attention, memory_format=torch.contiguous_format) if need_attention else None
        grad_logits = torch.empty_like(logits) if need_logits else None
        groups, tokens, heads, width, strides = _geometry(attention)
        rows = logits.numel() // (heads * width)
        if rows:
            problem = GateMultiplyProblem(rows, heads, width, logits.dtype)
            tile = _select(SCHEDULES.multiply_backward, problem, attention.device)
            block_rows, block_columns = tile.blocks(problem.gate_channels)
            kernel._gate_backward.run(
                attention,
                logits,
                grad.contiguous(),
                grad_attention if grad_attention is not None else logits,
                grad_logits if grad_logits is not None else logits,
                rows,
                tokens,
                groups,
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                strides[4],
                heads,
                width,
                block_rows,
                block_columns,
                need_attention,
                need_logits,
                num_warps=tile.num_warps,
                grid=tile.grid(rows, problem.gate_channels),
                warmup=False,
            )
        return grad_attention, grad_logits


def projection_operands(
    attention: Tensor, single: Tensor, weight: Tensor
) -> tuple[Tensor, Tensor, GateProjectionProblem] | None:

    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else single.dtype
    if not (
        attention.is_cuda
        and dtype in STANDARD_FLOAT_DTYPES
        and attention.dtype == dtype
        and (weight.dtype == dtype or torch.is_autocast_enabled("cuda"))
        and single.dtype in STANDARD_FLOAT_DTYPES
        and weight.dtype in STANDARD_FLOAT_DTYPES
    ):
        return None
    heads, width = attention.shape[-2:]
    channels = single.shape[-1]
    if heads == 0 or width == 0 or channels == 0:
        return None
    problem = GateProjectionProblem(
        attention.numel() // (heads * width), channels, heads, width, dtype, matmul_precision(dtype, _FLOAT32_DOT)
    )
    budget = shared_memory_budget(cuda_device_caps(attention.device))
    plain = (
        not is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in (attention, single, weight))
        and not is_in_torch_dispatch_mode()
        and not functorch_transforms_active()
    )
    if plain:
        admitted = _projection_admitted(problem, budget)
    else:
        admitted = SCHEDULES.projection_admitted(problem, budget)
    if not admitted:
        return None
    if plain and not torch.is_grad_enabled() and SCHEDULES.fuses(problem):
        return single, weight, problem
    return (
        single if single.dtype == dtype else single.to(dtype),
        weight if weight.dtype == dtype else weight.to(dtype),
        problem,
    )


@lru_cache(maxsize=256)
def _projection_admitted(problem: GateProjectionProblem, shared_memory_bytes: int) -> bool:
    return SCHEDULES.projection_admitted(problem, shared_memory_bytes)


def _projection_problem(attention: Tensor, single: Tensor, precision: str) -> GateProjectionProblem:
    heads, width = attention.shape[-2:]
    channels = single.shape[-1]
    return GateProjectionProblem(single.numel() // channels, channels, heads, width, single.dtype, precision)


def _projection_config(problem: GateProjectionProblem, target: TargetDescriptor) -> GateProjectionConfig:
    profile = SCHEDULES.projection
    return profile.select(problem, target if profile.target_changes_config(problem) else None)


def _launch_projection(
    entry,
    problem: GateProjectionProblem,
    attention: Tensor,
    single: Tensor,
    weight: Tensor,
    output: Tensor,
    logits: Tensor | None,
) -> None:
    with cuda_device_context(attention.device):
        if problem.rows == 0:
            return
        attention, single, weight = (
            _kernel_operand(attention),
            _kernel_operand(single),
            _kernel_operand(weight),
        )
        groups, tokens, heads, width, strides = _geometry(attention)
        single_rows = single.reshape(problem.rows, problem.single_channels)
        config = _projection_config(problem, cuda_device_caps(attention.device))
        entry.run(
            single_rows,
            weight,
            attention,
            output,
            *(() if logits is None else (logits,)),
            problem.rows,
            problem.single_channels,
            single_rows.stride(0),
            single_rows.stride(1),
            weight.stride(0),
            weight.stride(1),
            tokens,
            groups,
            *strides,
            heads,
            width,
            config.block_rows,
            min(config.block_gate, max(16, _next_power_of_two(problem.gate_channels))),
            config.block_reduction,
            input_precision=problem.input_precision,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            grid=config.grid(problem.rows, problem.gate_channels),
            warmup=False,
        )


def _native_logits(single: Tensor, weight: Tensor) -> Tensor:
    # Ambient autocast must not recast already-prepared operands.
    autocast = torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled("cuda") else nullcontext()
    with autocast:
        return F.linear(single, weight).contiguous()


def _gated_shape(single: Tensor, weight: Tensor) -> tuple[int, ...]:
    return (*single.shape[:-1], weight.shape[0])


def _projection_forward_impl(
    attention: Tensor, single: Tensor, weight: Tensor, precision: str, problem: GateProjectionProblem | None = None
) -> tuple[Tensor, Tensor]:

    if problem is None:
        problem = _projection_problem(attention, single, precision)
    if not SCHEDULES.fuses(problem):
        logits = _native_logits(single, weight)
        return _attention_gate_multiply_forward_impl(attention, logits), logits
    output = torch.empty(_gated_shape(single, weight), dtype=single.dtype, device=single.device)
    logits = torch.empty_like(output)
    _launch_projection(kernel._gate_projection_forward, problem, attention, single, weight, output, logits)
    return output, logits


def _projection_inference_impl(
    attention: Tensor, single: Tensor, weight: Tensor, precision: str, problem: GateProjectionProblem | None = None
) -> Tensor:

    if problem is None:
        problem = _projection_problem(attention, single, precision)
    if not SCHEDULES.fuses(problem):
        return _attention_gate_multiply_forward_impl(attention, _native_logits(single, weight))
    output = torch.empty(_gated_shape(single, weight), dtype=problem.compute_dtype, device=single.device)
    _launch_projection(kernel._gate_projection_inference, problem, attention, single, weight, output, None)
    return output


def _projection_linear_gradients(grad_logits, single, weight, need_single, need_weight):
    autocast = torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled("cuda") else nullcontext()
    with autocast:
        grad_single = torch.matmul(grad_logits, weight) if need_single else None
        grad_weight = (
            grad_logits.reshape(-1, weight.shape[0]).t().mm(single.reshape(-1, weight.shape[1]))
            if need_weight
            else None
        )
    return grad_single, grad_weight


def _projection_backward_impl(
    grad: Tensor,
    attention: Tensor,
    single: Tensor,
    weight: Tensor,
    logits: Tensor,
    need_attention: bool,
    need_single: bool,
    need_weight: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:

    grad_attention, grad_logits = _attention_gate_multiply_backward_impl(
        grad, attention, logits, need_attention, need_single or need_weight
    )
    if grad_logits is None:
        return grad_attention, None, None
    grad_single, grad_weight = _projection_linear_gradients(grad_logits, single, weight, need_single, need_weight)
    return grad_attention if need_attention else None, grad_single, grad_weight
