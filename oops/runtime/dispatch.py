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

r"""
Shared dispatch predicates and dtype coercions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import overload

import torch
import torch.autograd.forward_ad as forward_ad
from torch import Tensor
from torch._subclasses.fake_tensor import is_fake

from oops._errors import OOpsUnsupportedError
from oops.core import TargetDescriptor

from .cuda import cuda_device_caps
from .dtypes import (
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    DTypeSpec,
    NativeMatmulDomain,
    dtype_spec,
    registered_native_state_domains,
)


def _cuda_out_dtype_supported(operator: str) -> bool:
    r"""Whether this torch build has the CUDA matmul overload with a wider output dtype."""

    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(f"aten::{operator}.dtype", "CUDA")
    except (AttributeError, RuntimeError):
        return False


_CUDA_MM_OUT_DTYPE_SUPPORTED = _cuda_out_dtype_supported("mm")
_CUDA_BMM_OUT_DTYPE_SUPPORTED = _cuda_out_dtype_supported("bmm")


def float32_accumulating_matmul(left: Tensor, right: Tensor, *, batched: bool = False) -> Tensor:
    r"""Contract matrices with an output and accumulator no narrower than float32.

    The compute dtype is the promoted operand dtype, at least float32, independent of CUDA
    autocast. ``batched=False`` accepts matrices; ``batched=True`` accepts matrix batches."""

    compute_dtype = torch.promote_types(torch.promote_types(left.dtype, right.dtype), torch.float32)
    same_half_dtype = left.dtype is right.dtype and left.dtype in HALF_DTYPES
    # CUDA out_dtype overloads do not accept FP16 accumulation.
    wide_half_output = same_half_dtype and not (
        left.is_cuda and left.dtype is torch.float16 and torch.backends.cuda.matmul.allow_fp16_accumulation
    )
    with torch.autocast("cuda", enabled=False):
        if batched:
            if wide_half_output and left.is_cuda and right.is_cuda and _CUDA_BMM_OUT_DTYPE_SUPPORTED:
                return torch.bmm(left, right, out_dtype=compute_dtype)
            return torch.bmm(left.to(compute_dtype), right.to(compute_dtype))
        if wide_half_output and left.is_cuda and right.is_cuda and _CUDA_MM_OUT_DTYPE_SUPPORTED:
            return torch.mm(left, right, out_dtype=compute_dtype)
        return torch.mm(left.to(compute_dtype), right.to(compute_dtype))


FLOAT32_DOT_IEEE = "ieee"
FLOAT32_DOT_TF32X3 = "tf32x3"


_NATIVE_MATMUL_STATE_VERSION = 1
_NATIVE_MATMUL_KNOWN_DOMAINS = (
    NativeMatmulDomain.FP32
    | NativeMatmulDomain.FP16
    | NativeMatmulDomain.BF16
    | NativeMatmulDomain.FP8
    | NativeMatmulDomain.FP4
)
_NATIVE_MATMUL_FP32_DOMAIN = int(NativeMatmulDomain.FP32)
_NATIVE_MATMUL_FP16_DOMAIN = int(NativeMatmulDomain.FP16)
_NATIVE_MATMUL_BF16_DOMAIN = int(NativeMatmulDomain.BF16)
_NATIVE_MATMUL_DOMAIN_MASK = int(_NATIVE_MATMUL_KNOWN_DOMAINS)

# Token bits: version [56:62], controls [8:55], domains [0:7]; bit 63 is clear.
_NATIVE_MATMUL_STATE_VERSION_SHIFT = 56
_NATIVE_MATMUL_STATE_VERSION_MASK = 0x7F << _NATIVE_MATMUL_STATE_VERSION_SHIFT
_NATIVE_MATMUL_FP32_TF32 = 1 << 8
_NATIVE_MATMUL_FP16_REDUCED_REDUCTION = 1 << 9
_NATIVE_MATMUL_FP16_SPLIT_K_REDUCTION = 1 << 10
_NATIVE_MATMUL_FP16_ACCUMULATION = 1 << 11
_NATIVE_MATMUL_BF16_REDUCED_REDUCTION = 1 << 12
_NATIVE_MATMUL_BF16_SPLIT_K_REDUCTION = 1 << 13
_NATIVE_MATMUL_STATE_PAYLOAD_MASK = (
    _NATIVE_MATMUL_DOMAIN_MASK
    | _NATIVE_MATMUL_FP32_TF32
    | _NATIVE_MATMUL_FP16_REDUCED_REDUCTION
    | _NATIVE_MATMUL_FP16_SPLIT_K_REDUCTION
    | _NATIVE_MATMUL_FP16_ACCUMULATION
    | _NATIVE_MATMUL_BF16_REDUCED_REDUCTION
    | _NATIVE_MATMUL_BF16_SPLIT_K_REDUCTION
)
_NATIVE_MATMUL_STATE_MASK = _NATIVE_MATMUL_STATE_VERSION_MASK | _NATIVE_MATMUL_STATE_PAYLOAD_MASK


def native_matmul_domain(dtype: torch.dtype) -> NativeMatmulDomain:
    r"""Return the mutable precision domain for one native CUDA matmul operand dtype."""

    return _dtype_spec_or_unsupported(dtype, "native_matmul_domain").native_matmul_domain


def _dtype_spec_or_unsupported(dtype: torch.dtype, operator: str) -> DTypeSpec:
    try:
        return dtype_spec(dtype)
    except KeyError:
        raise unsupported(
            operator,
            "has no registered precision family; register the dtype before admitting native matmul",
            dtype=dtype,
        ) from None


def _scoped_float32_matmul_state() -> str:
    r"""Resolve the effective scoped CUDA matmul setting without reading deprecated allow_tf32."""

    backend = torch.backends.cuda.matmul.fp32_precision
    if backend != "none":
        return backend
    generic = torch.backends.fp32_precision  # type: ignore[attr-defined]
    if generic != "none":
        return generic
    # Unspecified scoped controls inherit IEEE.
    return "ieee"


def current_native_matmul_precision_state(domains: int | NativeMatmulDomain) -> int:
    r"""Read a versioned token for the requested mutable native CUDA matmul state."""

    selected = int(domains)
    if selected <= 0 or selected & ~_NATIVE_MATMUL_DOMAIN_MASK:
        raise ValueError(f"invalid native matmul precision domains: {selected}")
    unregistered = selected & ~int(registered_native_state_domains())
    if unregistered:
        raise unsupported(
            "native matmul precision",
            "has a registered dtype family but no state reader until its kernels and controls are implemented",
            domains=unregistered,
        )

    matmul = torch.backends.cuda.matmul
    state = (_NATIVE_MATMUL_STATE_VERSION << _NATIVE_MATMUL_STATE_VERSION_SHIFT) | selected
    if selected & _NATIVE_MATMUL_FP32_DOMAIN:
        fp32 = _scoped_float32_matmul_state()
        if fp32 == "tf32":
            state |= _NATIVE_MATMUL_FP32_TF32
        elif fp32 != "ieee":
            raise unsupported(
                "native matmul precision",
                "received an unknown effective scoped FP32 precision",
                precision=fp32,
            )
    if selected & _NATIVE_MATMUL_FP16_DOMAIN:
        state |= int(matmul.allow_fp16_reduced_precision_reduction) * _NATIVE_MATMUL_FP16_REDUCED_REDUCTION
        state |= int(matmul.allow_fp16_reduced_precision_reduction_split_k) * _NATIVE_MATMUL_FP16_SPLIT_K_REDUCTION
        state |= int(matmul.allow_fp16_accumulation) * _NATIVE_MATMUL_FP16_ACCUMULATION
    if selected & _NATIVE_MATMUL_BF16_DOMAIN:
        state |= int(matmul.allow_bf16_reduced_precision_reduction) * _NATIVE_MATMUL_BF16_REDUCED_REDUCTION
        state |= int(matmul.allow_bf16_reduced_precision_reduction_split_k) * _NATIVE_MATMUL_BF16_SPLIT_K_REDUCTION
    return state


@torch.compiler.assume_constant_result
def captured_native_matmul_precision_state(domains: int | NativeMatmulDomain) -> int:
    r"""Capture native matmul state in a compiled semantic custom-op ABI."""

    return current_native_matmul_precision_state(domains)


def _invalid_native_matmul_state(state: object) -> None:
    raise unsupported(
        "native matmul precision",
        "received an invalid or stale precision-state ABI",
        state=state,
    )


def _native_matmul_domains_from_state(state: int) -> int:
    version = (state & _NATIVE_MATMUL_STATE_VERSION_MASK) >> _NATIVE_MATMUL_STATE_VERSION_SHIFT
    domains = state & _NATIVE_MATMUL_DOMAIN_MASK
    if state < 0 or state & ~_NATIVE_MATMUL_STATE_MASK or version != _NATIVE_MATMUL_STATE_VERSION or domains == 0:
        _invalid_native_matmul_state(state)
    return domains


def require_native_matmul_precision_state(operator: str, expected: int) -> None:
    r"""Reject a semantic phase when mutable native matmul state changed after capture."""

    if type(expected) is not int:
        _invalid_native_matmul_state(expected)
    current = current_native_matmul_precision_state(_native_matmul_domains_from_state(expected))
    if current != expected:
        raise unsupported(
            operator,
            "requires native matmul precision state to remain unchanged for one operator execution",
            forward_precision=expected,
            backward_precision=current,
        )


def native_matmul_fp16_accumulation_enabled(state: int) -> bool:
    r"""Decode the validated FP16 accumulation control from a native-matmul state token."""

    if type(state) is not int:
        _invalid_native_matmul_state(state)
    domains = _native_matmul_domains_from_state(state)
    if not domains & _NATIVE_MATMUL_FP16_DOMAIN:
        raise ValueError("native matmul state does not carry the FP16 domain")
    return bool(state & _NATIVE_MATMUL_FP16_ACCUMULATION)


def triton_dot_precision(dtype: torch.dtype, float32_dot: str, state: int) -> str:
    r"""Resolve one Triton dot lowering from a captured native precision-state token."""

    if dtype in HALF_DTYPES:
        return FLOAT32_DOT_IEEE
    if dtype is not torch.float32:
        spec = _dtype_spec_or_unsupported(dtype, "triton_dot_precision")
        if not spec.triton_dot:
            raise unsupported(
                "triton_dot_precision",
                "has no registered Triton dot policy; implement the lowering before admitting a kernel",
                dtype=dtype,
            )
        raise AssertionError("registered Triton dot dtype has no implemented precision resolver")
    if not (_native_matmul_domains_from_state(state) & _NATIVE_MATMUL_FP32_DOMAIN):
        raise ValueError("FP32 Triton dot precision requires the FP32 native-matmul state domain")
    if state & _NATIVE_MATMUL_FP32_TF32:
        return "tf32"
    return float32_dot


def _current_triton_float32_dot_precision(float32_dot: str) -> str:
    r"""Map the live effective scoped FP32 setting without constructing a native-state token."""

    fp32 = _scoped_float32_matmul_state()
    if fp32 == "tf32":
        return "tf32"
    if fp32 == "ieee":
        return float32_dot
    raise unsupported("triton_dot_precision", "received an unknown FP32 precision state", state=fp32)


def require_triton_dot_precision(
    operator: str,
    dtype: torch.dtype,
    float32_dot: str,
    expected: str,
) -> None:
    r"""Reject a captured precision policy that differs from the active policy."""

    if dtype in HALF_DTYPES:
        current = FLOAT32_DOT_IEEE
    elif dtype is torch.float32:
        current = _current_triton_float32_dot_precision(float32_dot)
    else:
        _dtype_spec_or_unsupported(dtype, operator)
        raise unsupported(operator, "has no implemented Triton dot precision policy", dtype=dtype)
    if current != expected:
        raise unsupported(
            operator,
            "requires effective Triton dot precision to match the captured forward lowering",
            captured_precision=expected,
            current_precision=current,
        )


def require_upcast_matmul_precision(
    operator: str,
    source_dtype: torch.dtype,
    float32_dot: str,
    expected: str,
) -> None:
    r"""Reject a stale lowering for operands widened to FP32 inside a Triton dot."""

    if source_dtype in HALF_DTYPES:
        current = "tf32"
    elif source_dtype is torch.float32:
        current = _current_triton_float32_dot_precision(float32_dot)
    else:
        _dtype_spec_or_unsupported(source_dtype, operator)
        raise unsupported(operator, "has no implemented upcast Triton dot precision policy", dtype=source_dtype)
    if current != expected:
        raise unsupported(
            operator,
            "requires effective upcast Triton dot precision to match the captured forward lowering",
            captured_precision=expected,
            current_precision=current,
        )


@torch.compiler.assume_constant_result
def matmul_precision(dtype: torch.dtype, float32_dot: str = FLOAT32_DOT_IEEE) -> str:
    r"""Return the dot precision for ``dtype`` under the active CUDA matmul policy."""

    if dtype in HALF_DTYPES:
        return FLOAT32_DOT_IEEE
    if dtype is torch.float32:
        return _current_triton_float32_dot_precision(float32_dot)
    _dtype_spec_or_unsupported(dtype, "matmul_precision")
    raise unsupported("matmul_precision", "has no implemented Triton dot precision policy", dtype=dtype)


def upcast_matmul_precision(source_dtype: torch.dtype, float32_dot: str = FLOAT32_DOT_IEEE) -> str:
    r"""Return dot precision for operands widened from ``source_dtype`` to float32."""

    if source_dtype in HALF_DTYPES:
        return "tf32"
    if source_dtype is not torch.float32:
        _dtype_spec_or_unsupported(source_dtype, "upcast_matmul_precision")
        raise unsupported(
            "upcast_matmul_precision", "has no implemented Triton dot precision policy", dtype=source_dtype
        )
    return matmul_precision(torch.float32, float32_dot)


_SHARED_MEMORY_RESERVE_BYTES = 1024


def shared_memory_budget(target: torch.device | TargetDescriptor) -> int:
    r"""Return the available per-block shared-memory budget."""

    descriptor = target if isinstance(target, TargetDescriptor) else cuda_device_caps(target)
    hardware_limit = descriptor.resources.shared_memory_per_block
    if hardware_limit is None:
        raise RuntimeError("target descriptor is missing shared_memory_per_block")
    return max(hardware_limit - _SHARED_MEMORY_RESERVE_BYTES, 0)


# NVIDIA MMA requires a reduction granularity of 16.
MIN_DOT_REDUCTION = 16

MIN_DOT_BLOCK = 16


def dot_tile_shared_bytes(
    rows: int,
    reduction: int,
    columns: int,
    element_size: int,
    operands: int = 2,
    *,
    buffers: int = 1,
) -> int:
    r"""Estimate shared-memory bytes for an activation tile and ``operands`` weight tiles.

    ``element_size`` is the operand storage width; ``buffers`` is the staging count."""

    activation = buffers * rows * reduction
    weights = operands * max(buffers - 1, 1) * reduction * columns
    padding = operands * (buffers - 1) * reduction
    return element_size * (activation + weights + padding) + 4 * rows * columns


def halved_dot_blocks(block: int, floor: int = MIN_DOT_BLOCK) -> Iterator[int]:
    r"""
    Yield ``block`` and each halving of it down to ``floor``, widest first.
    """

    while block >= floor:
        yield block
        block //= 2


def narrowed_dot_tile(
    candidates: Iterable[tuple[int, int]],
    reduction: int,
    element_size: int,
    budget: int,
    operands: int = 2,
    *,
    buffers: int = 1,
) -> tuple[int, int] | None:
    r"""Return the first legal ``(rows, columns)`` candidate fitting ``budget``, or ``None``."""

    if reduction < MIN_DOT_REDUCTION:
        return None
    for rows, columns in candidates:
        if dot_tile_shared_bytes(rows, reduction, columns, element_size, operands, buffers=buffers) <= budget:
            return rows, columns
    return None


def tensors_require_grad(*tensors: Tensor | None) -> bool:
    r"""
    Return whether any tensor participates in autograd.
    """

    return any(tensor is not None and tensor.requires_grad for tensor in tensors)


def needs_grad(*tensors: Tensor | None) -> bool:
    r"""
    Return whether the current call needs an autograd-aware path.
    """

    return torch.is_grad_enabled() and tensors_require_grad(*tensors)


def tensors_share_device(reference: Tensor, *tensors: Tensor | None) -> bool:
    r"""Return whether every provided tensor lives on ``reference``'s device."""

    return all(tensor is None or tensor.device == reference.device for tensor in tensors)


def empty_with_dependencies(output: Tensor, *tensors: Tensor | None) -> Tensor:
    r"""Attach zero-valued autograd edges to an already-shaped empty output."""

    for tensor in tensors:
        if tensor is not None and tensor.is_floating_point():
            # Empty views retain autograd edges without reading storage.
            output = output + tensor.as_strided((0,), (1,)).sum()
    return output


def require_same_device(operator: str, reference: Tensor, *tensors: Tensor | None) -> None:
    r"""Raise ``ValueError`` if provided operands have different devices."""

    if not tensors_share_device(reference, *tensors):
        devices = (reference.device, *(tensor.device for tensor in tensors if tensor is not None))
        raise ValueError(f"{operator} operands must be on the same device, got {devices}")


def is_compiling() -> bool:
    r"""
    Return whether TorchDynamo/Torch compiler is tracing the current call.
    """

    return torch.compiler.is_compiling()


def is_external_fake(tensor: Tensor) -> bool:
    r"""Return whether ``tensor`` is an external FakeTensor."""

    tensor_type = type(tensor)
    return tensor_type is not Tensor and tensor_type is not torch.nn.Parameter and is_fake(tensor)


def cast_floating_tensor(tensor: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    r"""
    Cast a floating tensor on ``device`` to ``dtype``; leave all other tensors unchanged.
    """

    if tensor.is_floating_point() and tensor.device == device and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def cast_floating_tensors_like(reference: Tensor, tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    r"""
    Cast floating tensors on ``reference``'s device to ``reference``'s dtype.
    """

    return tuple(cast_floating_tensor(tensor, reference.dtype, reference.device) for tensor in tensors)


def prepare_cuda_autocast_operands(reference: Tensor, operands: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    r"""Cast eligible same-device floating operands to the active CUDA half-autocast dtype.

    Operands outside that device/dtype domain remain unchanged."""

    if not reference.is_cuda or not torch.is_autocast_enabled("cuda"):
        return operands
    target_dtype = torch.get_autocast_dtype("cuda")
    if target_dtype not in HALF_DTYPES:
        return operands
    return tuple(
        (
            cast_floating_tensor(operand, target_dtype, reference.device)
            if operand.dtype in STANDARD_FLOAT_DTYPES
            else operand
        )
        for operand in operands
    )


def require_cuda(operator: str, *tensors: Tensor | None) -> None:
    r"""
    Raise ``OOpsUnsupportedError`` unless all provided tensors are CUDA tensors.
    """

    for tensor in tensors:
        if tensor is not None and not tensor.is_cuda:
            raise unsupported(operator, "requires CUDA tensors")


def unsupported(operator: str, reason: str, **context: object) -> OOpsUnsupportedError:
    r"""
    Build a consistent unsupported-dispatch error.
    """

    suffix = ""
    separator = " ("
    for key, value in context.items():
        suffix = f"{suffix}{separator}{key}={value}"
        separator = ", "
    if context:
        suffix = f"{suffix})"
    return OOpsUnsupportedError(f"{operator} {reason}{suffix}")


def has_forward_mode_tangent(*tensors: Tensor | None) -> bool:
    r"""Return whether any operand carries a forward-mode tangent."""

    if forward_ad._current_level < 0:
        return False
    return any(tensor is not None and forward_ad.unpack_dual(tensor).tangent is not None for tensor in tensors)


def functorch_transforms_active() -> bool:
    r"""Whether any torch.func interpreter is active, using Torch's direct C++ fast check."""

    return torch._C._are_functorch_transforms_active()


def unwrap_functorch_tensor(tensor: Tensor) -> Tensor:
    r"""Remove an active functorch wrapper behind the centralized Torch-version boundary."""

    functorch = getattr(torch._C, "_functorch", None)
    unwrap = None if functorch is None else getattr(functorch, "get_unwrapped", None)
    if unwrap is None:
        raise RuntimeError("this PyTorch build does not expose the required functorch unwrap boundary")
    return unwrap(tensor)


@contextmanager
def disable_functorch() -> Iterator[None]:
    r"""Run raw-storage VJP work below the active functorch interpreter."""

    disable = getattr(torch._C, "_DisableFuncTorch", None)
    if disable is None:
        raise RuntimeError("this PyTorch build does not expose the required functorch disable boundary")
    with disable():
        yield


def _functorch_transform_keys(transforms_active: bool | None = None) -> tuple[object, ...] | None:
    r"""Return the active functorch keys, or ``None`` when Torch cannot expose them."""

    transforms_active = functorch_transforms_active() if transforms_active is None else transforms_active
    if not transforms_active:
        return ()
    functorch = getattr(torch._C, "_functorch", None)
    if functorch is None:
        return None
    get_stack = getattr(functorch, "get_interpreter_stack", None)
    if get_stack is not None:
        stack = get_stack()
        if stack is not None:
            return tuple(interpreter.key() for interpreter in stack)
    peek = getattr(functorch, "peek_interpreter_stack", None)
    interpreter = None if peek is None else peek()
    return None if interpreter is None else (interpreter.key(),)


def forward_mode_active(*tensors: Tensor | None) -> bool:
    r"""Return whether a direct dual or a ``torch.func.jvp`` transform is active."""

    keys = _functorch_transform_keys()
    if keys == ():
        return has_forward_mode_tangent(*tensors)
    functorch = getattr(torch._C, "_functorch", None)
    transform_type = None if functorch is None else getattr(functorch, "TransformType", None)
    jvp = None if transform_type is None else getattr(transform_type, "Jvp", None)
    if keys is not None and jvp is not None and jvp in keys:
        return True
    # Wrapped dual tensors cannot be inspected with unpack_dual.
    return forward_ad._current_level >= 0


def autograd_function_active(
    *tensors: Tensor | None,
    transforms_active: bool | None = None,
) -> bool:
    r"""Return whether the call requires a differentiation transform rule."""

    keys = _functorch_transform_keys(transforms_active)
    if keys == ():
        return has_forward_mode_tangent(*tensors)
    if keys is None:
        return True
    functorch = getattr(torch._C, "_functorch", None)
    transform_type = None if functorch is None else getattr(functorch, "TransformType", None)
    if transform_type is None:
        return True
    differentiation = tuple(
        key for key in (getattr(transform_type, "Jvp", None), getattr(transform_type, "Grad", None)) if key is not None
    )
    return any(key in differentiation for key in keys) or forward_ad._current_level >= 0


def require_no_forward_mode(operator: str, *tensors: Tensor | None) -> None:
    r"""Raise ``OOpsUnsupportedError`` if forward-mode AD is active."""

    if forward_mode_active(*tensors):
        raise unsupported(
            operator,
            "does not support forward-mode AD",
            hint="use reverse-mode autograd or the equivalent Torch operation for a Jacobian-vector product",
        )


@overload
def _kernel_operand(tensor: None) -> None: ...


@overload
def _kernel_operand(tensor: Tensor) -> Tensor: ...


def _kernel_operand(tensor: Tensor | None) -> Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.resolve_neg()
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _require_dot_block(name: str, value: int) -> None:
    if value < MIN_DOT_BLOCK or value & (value - 1):
        raise ValueError(f"{name} must be a power of two at least {MIN_DOT_BLOCK}; got {value}")


def _require_num_warps(name: str, value: int) -> None:
    if value not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be one of 1, 2, 4, or 8; got {value}")
