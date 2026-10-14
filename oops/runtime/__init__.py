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
Runtime support used by public operator dispatch and caches.
"""

from __future__ import annotations

from .cache import TensorCacheKey, TensorMetadataKey, tensor_cache_key, tensor_metadata_key
from .cuda import (
    CUDA_MAX_GRID_X,
    CUDA_MAX_GRID_YZ,
    INT32_MAX,
    cuda_device_caps,
    cuda_device_caps_if_needed,
    cuda_device_context,
    tiled_grid_dim_fits,
)
from .dispatch import (
    FLOAT32_DOT_IEEE,
    FLOAT32_DOT_TF32X3,
    MIN_DOT_BLOCK,
    MIN_DOT_REDUCTION,
    autograd_function_active,
    captured_native_matmul_precision_state,
    cast_floating_tensor,
    cast_floating_tensors_like,
    current_native_matmul_precision_state,
    disable_functorch,
    dot_tile_shared_bytes,
    empty_with_dependencies,
    float32_accumulating_matmul,
    forward_mode_active,
    functorch_transforms_active,
    halved_dot_blocks,
    has_forward_mode_tangent,
    is_compiling,
    is_external_fake,
    matmul_precision,
    narrowed_dot_tile,
    native_matmul_domain,
    native_matmul_fp16_accumulation_enabled,
    needs_grad,
    prepare_cuda_autocast_operands,
    require_cuda,
    require_native_matmul_precision_state,
    require_no_forward_mode,
    require_same_device,
    require_triton_dot_precision,
    require_upcast_matmul_precision,
    shared_memory_budget,
    tensors_require_grad,
    tensors_share_device,
    triton_dot_precision,
    unsupported,
    unwrap_functorch_tensor,
    upcast_matmul_precision,
)
from .dtypes import (
    HALF_DTYPES,
    NO_PACKING,
    NO_SCALE,
    STANDARD_FLOAT_DTYPES,
    DTypeFamily,
    DTypeSpec,
    MathPolicy,
    NativeMatmulDomain,
    OperandPolicy,
    PackingLayout,
    ScaleGranularity,
    ScaleLayout,
    dtype_spec,
    registered_dtype_specs,
    registered_native_state_domains,
)

__all__ = [
    "require_cuda",
    "has_forward_mode_tangent",
    "require_no_forward_mode",
    "require_same_device",
    "unsupported",
    "FLOAT32_DOT_IEEE",
    "FLOAT32_DOT_TF32X3",
    "HALF_DTYPES",
    "STANDARD_FLOAT_DTYPES",
    "NO_PACKING",
    "NO_SCALE",
    "CUDA_MAX_GRID_X",
    "CUDA_MAX_GRID_YZ",
    "cuda_device_context",
    "INT32_MAX",
    "TensorCacheKey",
    "TensorMetadataKey",
    "autograd_function_active",
    "cast_floating_tensor",
    "cast_floating_tensors_like",
    "cuda_device_caps",
    "cuda_device_caps_if_needed",
    "is_compiling",
    "is_external_fake",
    "MIN_DOT_BLOCK",
    "MIN_DOT_REDUCTION",
    "DTypeFamily",
    "DTypeSpec",
    "MathPolicy",
    "NativeMatmulDomain",
    "OperandPolicy",
    "PackingLayout",
    "ScaleGranularity",
    "ScaleLayout",
    "dot_tile_shared_bytes",
    "empty_with_dependencies",
    "float32_accumulating_matmul",
    "forward_mode_active",
    "functorch_transforms_active",
    "halved_dot_blocks",
    "narrowed_dot_tile",
    "matmul_precision",
    "captured_native_matmul_precision_state",
    "current_native_matmul_precision_state",
    "disable_functorch",
    "dtype_spec",
    "native_matmul_domain",
    "native_matmul_fp16_accumulation_enabled",
    "needs_grad",
    "prepare_cuda_autocast_operands",
    "registered_dtype_specs",
    "registered_native_state_domains",
    "require_native_matmul_precision_state",
    "require_triton_dot_precision",
    "require_upcast_matmul_precision",
    "shared_memory_budget",
    "tensor_cache_key",
    "tensor_metadata_key",
    "tiled_grid_dim_fits",
    "tensors_require_grad",
    "tensors_share_device",
    "triton_dot_precision",
    "unwrap_functorch_tensor",
    "upcast_matmul_precision",
]
