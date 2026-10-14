from __future__ import annotations

import math

import torch
import triton
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import (
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
    needs_grad,
    unsupported,
)

from . import kernel


def vector_norm(vectors: Tensor, eps: float) -> Tensor:
    if needs_grad(vectors):
        if (
            not is_compiling()
            and type(vectors) in (Tensor, torch.nn.Parameter)
            and not is_in_torch_dispatch_mode()
            and not functorch_transforms_active()
        ):
            return _VectorNorm.apply(vectors, eps)
        return _forward(vectors, eps)
    if is_compiling() or is_external_fake(vectors):
        return _forward(vectors, eps)
    return _run_forward(vectors, eps)


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


@torch.library.custom_op("ooperators::_smoothed_vector_norm_forward", mutates_args=())
def _forward(vectors: Tensor, eps: float) -> Tensor:
    return _run_forward(vectors, eps)


@_forward.register_fake
def _forward_fake(vectors: Tensor, eps: float) -> Tensor:
    del eps
    return vectors.new_empty(vectors.shape[:-1])


def _setup_context(ctx, inputs, output):
    del output
    vectors, ctx.eps = inputs
    ctx.save_for_backward(vectors)
    ctx.plain = False


def _autograd_backward(ctx, gradient):
    if torch.is_grad_enabled():
        raise unsupported("smoothed_vector_norm", "supports first-order reverse-mode differentiation only")
    backward = _run_backward if ctx.plain else _backward
    return backward(ctx.saved_tensors[0], gradient, ctx.eps), None


torch.library.register_autograd(_forward, _autograd_backward, setup_context=_setup_context)


@torch.library.custom_op("ooperators::_smoothed_vector_norm_backward", mutates_args=())
def _backward(vectors: Tensor, upstream: Tensor, eps: float) -> Tensor:
    return _run_backward(vectors, upstream, eps)


@_backward.register_fake
def _backward_fake(vectors: Tensor, upstream: Tensor, eps: float) -> Tensor:
    del upstream, eps
    return vectors.new_empty(vectors.shape)


class _VectorNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vectors, eps):
        output = _run_forward(vectors, eps)
        _setup_context(ctx, (vectors, eps), output)
        ctx.plain = True
        return output

    backward = staticmethod(_autograd_backward)
