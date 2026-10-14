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

r"""Storage dtype and arithmetic policy descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag
from types import MappingProxyType

import torch


class DTypeFamily(Enum):
    r"""Logical floating-point families relevant to kernel arithmetic."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"


class NativeMatmulDomain(IntFlag):
    r"""Mutable native-matmul precision families carried through semantic ABIs."""

    FP32 = 1
    FP16 = 2
    BF16 = 4
    FP8 = 8
    FP4 = 16


@dataclass(frozen=True, slots=True)
class DTypeSpec:
    r"""Registered storage and precision-control properties of one exact torch dtype."""

    torch_dtype: torch.dtype
    family: DTypeFamily
    logical_bits: int
    logical_elements_per_storage: int
    native_matmul_domain: NativeMatmulDomain
    supports_scaled_storage: bool
    native_state_reader: bool
    triton_dot: bool

    def __post_init__(self) -> None:
        if self.logical_bits <= 0:
            raise ValueError("logical_bits must be positive")
        if self.logical_elements_per_storage <= 0:
            raise ValueError("logical_elements_per_storage must be positive")
        if not isinstance(self.torch_dtype, torch.dtype):
            raise TypeError("torch_dtype must be a torch.dtype")
        storage_bits = self.torch_dtype.itemsize * 8
        if self.logical_bits * self.logical_elements_per_storage != storage_bits:
            raise ValueError("logical dtype width and pack factor must fill exactly one storage element")
        if not self.native_matmul_domain or self.native_matmul_domain & (self.native_matmul_domain - 1):
            raise ValueError("one DTypeSpec must belong to exactly one native-matmul domain")


class ScaleGranularity(Enum):
    r"""How scale factors map onto a stored tensor."""

    NONE = "none"
    TENSOR = "tensor"
    AXIS = "axis"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class ScaleLayout:
    r"""Scale dtype and logical axis/block layout for a math policy."""

    granularity: ScaleGranularity = ScaleGranularity.NONE
    scale_dtype: torch.dtype | None = None
    axes: tuple[int, ...] = ()
    block_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.granularity is ScaleGranularity.NONE:
            if self.scale_dtype is not None or self.axes or self.block_shape:
                raise ValueError("an unscaled layout cannot carry scale metadata")
            return
        if self.scale_dtype is None or not self.scale_dtype.is_floating_point:
            raise ValueError("a scaled layout requires a floating scale_dtype")
        if self.granularity is ScaleGranularity.TENSOR:
            if self.axes or self.block_shape:
                raise ValueError("per-tensor scaling cannot carry axes or a block shape")
        elif self.granularity is ScaleGranularity.AXIS:
            if not self.axes or self.block_shape:
                raise ValueError("per-axis scaling requires axes and no block shape")
            if len(set(self.axes)) != len(self.axes):
                raise ValueError("scale axes must be unique")
        elif self.granularity is ScaleGranularity.BLOCK and (
            self.axes or not self.block_shape or any(size <= 0 for size in self.block_shape)
        ):
            raise ValueError("per-block scaling requires a positive block shape and no axes")

    @property
    def is_scaled(self) -> bool:
        return self.granularity is not ScaleGranularity.NONE


NO_SCALE = ScaleLayout()


@dataclass(frozen=True, slots=True)
class PackingLayout:
    r"""Mapping from logical scalar elements onto one physical storage element."""

    logical_elements_per_storage: int = 1
    axis: int | None = None

    def __post_init__(self) -> None:
        if self.logical_elements_per_storage <= 0:
            raise ValueError("logical_elements_per_storage must be positive")
        if self.logical_elements_per_storage == 1:
            if self.axis is not None:
                raise ValueError("unpacked storage cannot declare a packed axis")
        elif not isinstance(self.axis, int):
            raise ValueError("packed storage requires a logical packing axis")

    @property
    def is_packed(self) -> bool:
        return self.logical_elements_per_storage > 1


NO_PACKING = PackingLayout()


@dataclass(frozen=True, slots=True)
class OperandPolicy:
    r"""Storage and scale contract for one logical matrix operand."""

    storage: DTypeSpec
    scale_layout: ScaleLayout = NO_SCALE
    packing_layout: PackingLayout = NO_PACKING
    scale_required: bool = False

    def __post_init__(self) -> None:
        if self.scale_layout.is_scaled and not self.storage.supports_scaled_storage:
            raise ValueError(f"{self.storage.family.value} storage has no registered scaled-storage contract")
        if self.scale_required and not self.scale_layout.is_scaled:
            raise ValueError("this operand policy requires an explicit scale layout")
        if self.packing_layout.logical_elements_per_storage != self.storage.logical_elements_per_storage:
            raise ValueError("operand packing does not match the registered storage dtype")

    @classmethod
    def from_dtype(
        cls,
        storage_dtype: torch.dtype,
        *,
        scale_layout: ScaleLayout = NO_SCALE,
        packing_layout: PackingLayout = NO_PACKING,
        scale_required: bool = False,
    ) -> OperandPolicy:
        return cls(
            storage=dtype_spec(storage_dtype),
            scale_layout=scale_layout,
            packing_layout=packing_layout,
            scale_required=scale_required,
        )

    @property
    def storage_dtype(self) -> torch.dtype:
        return self.storage.torch_dtype


@dataclass(frozen=True, slots=True)
class MathPolicy:
    r"""A kernel's per-operand storage/scale and shared arithmetic contract."""

    operands: tuple[OperandPolicy, ...]
    compute: DTypeSpec
    accumulator: DTypeSpec
    output: DTypeSpec

    def __post_init__(self) -> None:
        if not self.operands:
            raise ValueError("a math policy requires at least one operand")

    @classmethod
    def from_dtypes(
        cls,
        *,
        storage_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
        output_dtype: torch.dtype,
        scale_layout: ScaleLayout = NO_SCALE,
        packing_layout: PackingLayout = NO_PACKING,
        scale_required: bool = False,
    ) -> MathPolicy:
        r"""Build the common single-storage-operand policy."""

        return cls(
            operands=(
                OperandPolicy.from_dtype(
                    storage_dtype,
                    scale_layout=scale_layout,
                    packing_layout=packing_layout,
                    scale_required=scale_required,
                ),
            ),
            compute=dtype_spec(compute_dtype),
            accumulator=dtype_spec(accumulator_dtype),
            output=dtype_spec(output_dtype),
        )

    @property
    def storage_dtype(self) -> torch.dtype:
        if len(self.operands) != 1:
            raise AttributeError("a multi-operand math policy has no single storage_dtype")
        return self.operands[0].storage_dtype

    @property
    def scale_layout(self) -> ScaleLayout:
        if len(self.operands) != 1:
            raise AttributeError("a multi-operand math policy has no single scale_layout")
        return self.operands[0].scale_layout

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.compute.torch_dtype

    @property
    def accumulator_dtype(self) -> torch.dtype:
        return self.accumulator.torch_dtype

    @property
    def output_dtype(self) -> torch.dtype:
        return self.output.torch_dtype


def dtype_spec(dtype: torch.dtype) -> DTypeSpec:
    r"""Return the exact registered description for ``dtype``."""

    try:
        return _DTYPE_SPECS[dtype]
    except KeyError as error:
        raise KeyError(f"no dtype specification is registered for {dtype}") from error


def registered_dtype_specs() -> tuple[DTypeSpec, ...]:
    r"""Return an immutable snapshot of registered dtype descriptions."""

    return tuple(_DTYPE_SPECS.values())


def _make_builtin_dtype_specs() -> MappingProxyType:
    r"""Build and freeze the exact builtin dtype registry."""

    specs: dict[torch.dtype, DTypeSpec] = {}

    def register(
        dtype: torch.dtype,
        family: DTypeFamily,
        logical_bits: int,
        domain: NativeMatmulDomain,
        *,
        logical_elements_per_storage: int = 1,
        supports_scaled_storage: bool,
        native_state_reader: bool,
        triton_dot: bool,
    ) -> None:
        spec = DTypeSpec(
            torch_dtype=dtype,
            family=family,
            logical_bits=logical_bits,
            logical_elements_per_storage=logical_elements_per_storage,
            native_matmul_domain=domain,
            supports_scaled_storage=supports_scaled_storage,
            native_state_reader=native_state_reader,
            triton_dot=triton_dot,
        )
        existing = specs.get(dtype)
        if existing is not None and existing != spec:
            raise ValueError(f"dtype already has a different registered specification: {dtype}")
        specs[dtype] = spec

    register(
        torch.float32,
        DTypeFamily.FP32,
        32,
        NativeMatmulDomain.FP32,
        supports_scaled_storage=False,
        native_state_reader=True,
        triton_dot=True,
    )
    register(
        torch.float16,
        DTypeFamily.FP16,
        16,
        NativeMatmulDomain.FP16,
        supports_scaled_storage=False,
        native_state_reader=True,
        triton_dot=True,
    )
    register(
        torch.bfloat16,
        DTypeFamily.BF16,
        16,
        NativeMatmulDomain.BF16,
        supports_scaled_storage=False,
        native_state_reader=True,
        triton_dot=True,
    )

    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            register(
                dtype,
                DTypeFamily.FP8,
                8,
                NativeMatmulDomain.FP8,
                supports_scaled_storage=True,
                native_state_reader=False,
                triton_dot=False,
            )

    float4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if isinstance(float4_dtype, torch.dtype):
        register(
            float4_dtype,
            DTypeFamily.FP4,
            4,
            NativeMatmulDomain.FP4,
            logical_elements_per_storage=2,
            supports_scaled_storage=True,
            native_state_reader=False,
            triton_dot=False,
        )
    return MappingProxyType(specs)


_DTYPE_SPECS = _make_builtin_dtype_specs()
del _make_builtin_dtype_specs

HALF_DTYPES = (torch.float16, torch.bfloat16)
STANDARD_FLOAT_DTYPES = (*HALF_DTYPES, torch.float32)

_REGISTERED_NATIVE_STATE_DOMAINS = NativeMatmulDomain(0)
for _spec in _DTYPE_SPECS.values():
    if _spec.native_state_reader:
        _REGISTERED_NATIVE_STATE_DOMAINS |= _spec.native_matmul_domain


def registered_native_state_domains() -> NativeMatmulDomain:
    r"""Return domains with at least one implemented native precision-state reader."""

    return _REGISTERED_NATIVE_STATE_DOMAINS
