from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from typing import TypeAlias

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.core import TargetDescriptor
from oops.runtime import (
    CUDA_MAX_GRID_X,
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps,
    cuda_device_context,
    functorch_transforms_active,
    is_external_fake,
    needs_grad,
)

from . import kernel
from .backend import (
    _PACKED_ROWS,
    _matrix_shape,
    _row_addressing,
    layer_norm_forward_into,
    layer_norm_forward_with_stats,
)
from .profiles import SCHEDULES
from .schedule import MAX_LAYER_NORM_WIDTH, LayerNormPlan, LayerNormProblem

_AffineMetadata: TypeAlias = tuple[torch.device, torch.dtype, tuple[int, ...]]
_LayerNormAdmissionKey: TypeAlias = tuple[
    torch.device,
    torch.dtype,
    tuple[int, ...],
    int,
    _AffineMetadata | None,
    _AffineMetadata | None,
]


def supports_layer_norm(
    input: Tensor,
    normalized_size: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
) -> bool:
    compiling = torch.compiler.is_compiling()
    if compiling:
        return _supports_layer_norm_tensors(input, normalized_size, weight, bias, dynamic_rows=True)
    input_type = type(input)
    if input_type is not Tensor and input_type is not torch.nn.Parameter and is_external_fake(input):
        return _supports_layer_norm_tensors(input, normalized_size, weight, bias, cache=False, dynamic_rows=True)
    return _cached_supports_layer_norm(_layer_norm_admission_key(input, normalized_size, weight, bias))


def _supports_layer_norm_tensors(
    input: Tensor,
    normalized_size: int,
    weight: Tensor | None,
    bias: Tensor | None,
    *,
    cache: bool = True,
    dynamic_rows: bool = False,
) -> bool:
    affine = tuple(param for param in (weight, bias) if param is not None)
    if (
        input.dtype in HALF_DTYPES
        and any(param.dtype is torch.float32 for param in affine)
        and any(param.dtype is not torch.float32 for param in affine)
    ):
        return False
    supported = (
        input.is_cuda
        and input.dim() >= 1
        and normalized_size == input.shape[-1]
        and normalized_size > 0
        and normalized_size <= MAX_LAYER_NORM_WIDTH
        and input.dtype in STANDARD_FLOAT_DTYPES
        and _affine_supported(weight, input, normalized_size)
        and _affine_supported(bias, input, normalized_size)
    )
    if not supported:
        return False
    num_rows = input.numel() // normalized_size
    if dynamic_rows:
        torch._check(num_rows > 0)
        torch._check(num_rows <= CUDA_MAX_GRID_X)
        return True
    if num_rows <= 0 or num_rows > CUDA_MAX_GRID_X:
        return False
    problem = LayerNormProblem(num_rows, normalized_size, input.dtype)
    return SCHEDULES.supports(problem) if cache else SCHEDULES.supports(problem, cache=False)


def _supports_layer_norm_metadata(
    key: _LayerNormAdmissionKey,
) -> bool:
    input_device, input_dtype, input_shape, normalized_size, weight, bias = key
    affine = tuple(param for param in (weight, bias) if param is not None)
    if (
        input_dtype in HALF_DTYPES
        and any(param[1] is torch.float32 for param in affine)
        and any(param[1] is not torch.float32 for param in affine)
    ):
        return False
    supported = (
        input_device.type == "cuda"
        and len(input_shape) >= 1
        and normalized_size == input_shape[-1]
        and 0 < normalized_size <= MAX_LAYER_NORM_WIDTH
        and input_dtype in STANDARD_FLOAT_DTYPES
        and _affine_metadata_supported(weight, input_device, input_dtype, normalized_size)
        and _affine_metadata_supported(bias, input_device, input_dtype, normalized_size)
    )
    if not supported:
        return False
    num_rows = _numel(input_shape) // normalized_size
    return num_rows > 0 and SCHEDULES.supports(LayerNormProblem(num_rows, normalized_size, input_dtype))


@lru_cache(maxsize=1024)
def _cached_supports_layer_norm(
    key: _LayerNormAdmissionKey,
) -> bool:
    return _supports_layer_norm_metadata(key)


def layer_norm_backward_grid_fits(
    num_rows: int,
    normalized_size: int,
    dtype: torch.dtype = torch.float32,
    *,
    cache: bool = True,
) -> bool:

    if num_rows <= 0 or normalized_size <= 0 or normalized_size > MAX_LAYER_NORM_WIDTH:
        return False
    problem = LayerNormProblem(num_rows, normalized_size, dtype)
    return SCHEDULES.supports_backward(problem) if cache else SCHEDULES.supports_backward(problem, cache=False)


_NATIVE_AFFINE_TILE_ROWS = 256
_NATIVE_STARTUP_TILES = 64


@lru_cache(maxsize=None)
def _native_cache_bytes(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).L2_cache_size


def _native_startup_supported(input: Tensor) -> bool:
    elements = input.numel()
    if elements <= 8192:
        return True
    if not input.is_contiguous() or elements // input.shape[-1] > _NATIVE_AFFINE_TILE_ROWS * _NATIVE_STARTUP_TILES:
        return False
    return elements * input.element_size() <= _native_cache_bytes(input.device)


def triton_layer_norm(input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:

    operands = tuple(tensor for tensor in (input, weight, bias) if tensor is not None)
    plain = (
        not torch.compiler.is_compiling()
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not functorch_transforms_active()
        and not is_in_torch_dispatch_mode()
    )
    if (
        plain
        and all(tensor.dtype is input.dtype for tensor in operands)
        and (not needs_grad(*operands) or not torch._C._autograd._top_saved_tensors_default_hooks(True))
        and _native_startup_supported(input)
    ):
        input = input.resolve_neg()
        if not input.is_contiguous():
            input, _ = _row_addressing(input)
        weight = None if weight is None else weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        with torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled("cuda") else nullcontext():
            return F.layer_norm(input, (input.shape[-1],), weight, bias, eps)
    input, _ = _row_addressing(input)
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if plain:
        if needs_grad(*operands):
            return _EagerLayerNorm.apply(input, weight, bias, eps)[0]
        output = torch.empty(input.shape, device=input.device, dtype=input.dtype)
        layer_norm_forward_into(input, weight, bias, output, eps)
        return output
    if needs_grad(*operands):
        return _layer_norm_forward(input, weight, bias, eps)[0]
    return _layer_norm_inference(input, weight, bias, eps)


def triton_layer_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    need_grad_input: bool,
    need_grad_weight: bool,
    need_grad_bias: bool,
    *,
    plan: LayerNormPlan | None = None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:

    if plan is None:
        return _layer_norm_backward(
            grad_output,
            input,
            weight,
            bias,
            mean,
            rstd,
            eps,
            input.dtype,
            need_grad_input,
            need_grad_weight,
            need_grad_bias,
        )
    return _run_layer_norm_backward(
        grad_output,
        input,
        weight,
        bias,
        mean,
        rstd,
        eps,
        need_grad_input,
        need_grad_weight,
        need_grad_bias,
        plan=plan,
    )


def resolve_layer_norm_backward_plan(input: Tensor) -> LayerNormPlan:

    return resolve_layer_norm_backward_problem(
        input.numel() // input.shape[-1],
        input.shape[-1],
        input.dtype,
        target=cuda_device_caps(input.device),
    )


def resolve_layer_norm_backward_problem(
    num_rows: int,
    num_columns: int,
    dtype: torch.dtype,
    *,
    target: TargetDescriptor | None,
    cache: bool = True,
) -> LayerNormPlan:

    problem = LayerNormProblem(num_rows, num_columns, dtype)
    if cache:
        return SCHEDULES.resolve_backward(problem, target=target)
    return SCHEDULES.resolve_backward(problem, target=target, cache=False)


def _affine_supported(param: Tensor | None, input: Tensor, normalized_size: int) -> bool:
    if param is None:
        return True
    compatible = param.dtype == input.dtype or (param.dtype is torch.float32 and input.dtype in HALF_DTYPES)
    return param.device == input.device and compatible and param.shape == (normalized_size,)


def _affine_metadata_supported(
    param: _AffineMetadata | None,
    input_device: torch.device,
    input_dtype: torch.dtype,
    normalized_size: int,
) -> bool:
    if param is None:
        return True
    device, dtype, shape = param
    compatible = dtype == input_dtype or (dtype is torch.float32 and input_dtype in HALF_DTYPES)
    return device == input_device and compatible and shape == (normalized_size,)


def _layer_norm_admission_key(
    input: Tensor,
    normalized_size: int,
    weight: Tensor | None,
    bias: Tensor | None,
) -> _LayerNormAdmissionKey:
    def affine_metadata(tensor: Tensor | None) -> _AffineMetadata | None:
        return None if tensor is None else (tensor.device, tensor.dtype, tuple(tensor.shape))

    return (
        input.device,
        input.dtype,
        tuple(input.shape),
        normalized_size,
        affine_metadata(weight),
        affine_metadata(bias),
    )


def _numel(shape: tuple[int, ...]) -> int:
    elements = 1
    for extent in shape:
        elements *= extent
    return elements


@torch.library.custom_op("ooperators::_layer_norm_forward", mutates_args=(), device_types="cuda")
def _layer_norm_forward(
    input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    return layer_norm_forward_with_stats(input, weight, bias, eps)


@torch.library.custom_op("ooperators::_layer_norm_inference", mutates_args=(), device_types="cuda")
def _layer_norm_inference(input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    output = torch.empty(input.shape, device=input.device, dtype=input.dtype)
    layer_norm_forward_into(input, weight, bias, output, eps)
    return output


_LAYER_NORM_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor rstd, float eps, "
    "ScalarType input_dtype, bool need_grad_input, bool need_grad_weight, bool need_grad_bias) "
    "-> (Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_layer_norm_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_LAYER_NORM_BACKWARD_SCHEMA,
)
def _layer_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    input_dtype: torch.dtype,
    need_grad_input: bool,
    need_grad_weight: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:

    _validate_layer_norm_backward_input(input, input_dtype, need_grad_input, need_grad_weight)
    target = cuda_device_caps(input.device)
    num_rows, num_columns = _matrix_shape(input)
    plan = SCHEDULES.resolve_backward(LayerNormProblem(num_rows, num_columns, input_dtype), target=target)
    return _run_layer_norm_backward(
        grad_output,
        input,
        weight,
        bias,
        mean,
        rstd,
        eps,
        need_grad_input,
        need_grad_weight,
        need_grad_bias,
        plan=plan,
    )


def _run_layer_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    need_grad_input: bool,
    need_grad_weight: bool,
    need_grad_bias: bool,
    *,
    plan: LayerNormPlan,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    grad_input = torch.empty(input.shape, device=input.device, dtype=input.dtype) if need_grad_input else None
    grad_weight, grad_bias = triton_layer_norm_backward_into(
        grad_output,
        input,
        weight,
        bias,
        mean,
        rstd,
        eps,
        grad_input,
        need_grad_weight,
        need_grad_bias,
        plan=plan,
    )
    return (
        grad_input,
        grad_weight.to(weight.dtype) if grad_weight is not None and weight is not None else None,
        grad_bias.to(bias.dtype) if grad_bias is not None and bias is not None else None,
    )


def triton_layer_norm_backward_into(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    grad_input: Tensor | None,
    need_grad_weight: bool,
    need_grad_bias: bool,
    *,
    plan: LayerNormPlan,
) -> tuple[Tensor | None, Tensor | None]:
    r"""Write the input VJP into caller storage and return float32 affine VJPs.

    grad_input is contiguous, matches input shape/device/dtype, and does not alias
    inputs; None omits that VJP. The plan describes the rows. Statistics are both
    supplied as FP32 saved forward values. Unrequested affine VJPs are None.
    """
    with cuda_device_context(input.device):
        grad_output = grad_output.resolve_neg().contiguous()
        weight = None if weight is None else weight.resolve_neg().contiguous()
        mean = mean.resolve_neg().contiguous()
        rstd = rstd.resolve_neg().contiguous()
        need_grad_input = grad_input is not None
        has_weight = weight is not None
        need_grad_weight = need_grad_weight and has_weight
        need_grad_bias = need_grad_bias and bias is not None
        if need_grad_input or need_grad_weight:
            input, row_map = _row_addressing(input)
        else:
            row_map = _PACKED_ROWS
        weight_pointer = input if weight is None else weight
        num_rows, num_columns = _matrix_shape(input)
        num_blocks = plan.grid[0]
        grad_affine_partial = (
            torch.empty(
                (int(need_grad_weight) + int(need_grad_bias), num_blocks, num_columns),
                device=input.device,
                dtype=torch.float32,
            )
            if need_grad_weight or need_grad_bias
            else input
        )

        backward_kernel = (
            kernel._layer_norm_backward_wide
            if plan.block_columns == 8192
            and plan.block_rows > 1
            and (need_grad_input or need_grad_weight)
            and (input.dtype is torch.float32 or need_grad_input)
            else kernel._layer_norm_backward
        )
        backward_kernel[plan.grid](
            grad_output,
            input,
            weight_pointer,
            mean,
            rstd,
            input if grad_input is None else grad_input,
            grad_affine_partial,
            num_rows,
            num_columns,
            *row_map,
            has_weight,
            need_grad_input,
            need_grad_weight,
            need_grad_bias,
            plan.block_rows,
            plan.block_columns,
            num_warps=(
                max(plan.num_warps, 8)
                if backward_kernel is kernel._layer_norm_backward_wide
                and need_grad_input
                and (need_grad_weight or need_grad_bias)
                else plan.num_warps
            ),
        )

        grad_weight = torch.empty(num_columns, device=input.device, dtype=torch.float32) if need_grad_weight else None
        grad_bias = torch.empty(num_columns, device=input.device, dtype=torch.float32) if need_grad_bias else None
        if grad_weight is not None or grad_bias is not None:
            kernel._layer_norm_affine_backward_reduce.run(
                grad_affine_partial,
                input if grad_weight is None else grad_weight,
                input if grad_bias is None else grad_bias,
                num_blocks,
                num_columns,
                need_grad_weight,
                need_grad_bias,
                64,
                16,
                grid=((num_columns + 15) // 16,),
                warmup=False,
                num_warps=2,
            )
        return grad_weight, grad_bias


@torch.library.register_fake(_layer_norm_forward)
def _layer_norm_forward_fake(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    num_rows, _ = _matrix_shape(input)
    return (
        torch.empty(input.shape, device=input.device, dtype=input.dtype),
        torch.empty((num_rows,), device=input.device, dtype=torch.float32),
        torch.empty((num_rows,), device=input.device, dtype=torch.float32),
    )


@torch.library.register_fake(_layer_norm_inference)
def _layer_norm_inference_fake(input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    return torch.empty(input.shape, device=input.device, dtype=input.dtype)


@torch.library.register_fake(_layer_norm_backward)
def _layer_norm_backward_fake(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    mean: Tensor,
    rstd: Tensor,
    eps: float,
    input_dtype: torch.dtype,
    need_grad_input: bool,
    need_grad_weight: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    _validate_layer_norm_backward_input(input, input_dtype, need_grad_input, need_grad_weight)
    del grad_output, mean, rstd, eps
    return (
        torch.empty(input.shape, device=input.device, dtype=input.dtype) if need_grad_input else None,
        torch.empty_like(weight) if need_grad_weight and weight is not None else None,
        torch.empty_like(bias) if need_grad_bias and bias is not None else None,
    )


def _validate_layer_norm_backward_input(
    input: Tensor,
    input_dtype: torch.dtype,
    need_grad_input: bool,
    need_grad_weight: bool,
) -> None:

    if input_dtype not in STANDARD_FLOAT_DTYPES:
        raise ValueError(f"invalid LayerNorm input dtype metadata: {input_dtype}")
    if (need_grad_input or need_grad_weight) and input.dtype is not input_dtype:
        raise ValueError("LayerNorm backward input dtype metadata does not match its required input tensor")


def _setup_context(
    ctx, inputs: tuple[Tensor, Tensor | None, Tensor | None, float], output: tuple[Tensor, Tensor, Tensor]
) -> None:
    input, weight, bias, eps = inputs
    _output, mean, rstd = output
    ctx.mark_non_differentiable(mean, rstd)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(input, weight, bias, mean, rstd)
    ctx.eps = eps
    ctx.input_dtype = input.dtype
    ctx.need_grad_input, ctx.need_grad_weight, ctx.need_grad_bias = ctx.needs_input_grad[:3]


def _backward(ctx, grad_output: Tensor, _grad_mean: Tensor | None, _grad_rstd: Tensor | None):
    input, weight, bias, mean, rstd = ctx.saved_tensors
    grad_input, grad_weight, grad_bias = _layer_norm_backward(
        grad_output.contiguous(),
        input,
        weight,
        bias,
        mean,
        rstd,
        ctx.eps,
        ctx.input_dtype,
        ctx.need_grad_input,
        ctx.need_grad_weight,
        ctx.need_grad_bias,
    )
    return grad_input, grad_weight, grad_bias, None


torch.library.register_autograd(_layer_norm_forward, _backward, setup_context=_setup_context)


class _EagerLayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        output = layer_norm_forward_with_stats(input, weight, bias, eps)
        _setup_context(ctx, (input, weight, bias, eps), output)
        ctx.plan = resolve_layer_norm_backward_plan(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor, _grad_mean: Tensor | None, _grad_rstd: Tensor | None):
        if torch.is_grad_enabled():
            return _backward(ctx, grad_output, _grad_mean, _grad_rstd)
        input, weight, bias, mean, rstd = ctx.saved_tensors
        grads = _run_layer_norm_backward(
            grad_output,
            input,
            weight,
            bias,
            mean,
            rstd,
            ctx.eps,
            ctx.need_grad_input,
            ctx.need_grad_weight,
            ctx.need_grad_bias,
            plan=ctx.plan,
        )
        return (*grads, None)
