from __future__ import annotations

import torch

from oops.runtime import (
    captured_native_matmul_precision_state,
    native_matmul_domain,
    require_native_matmul_precision_state,
)


def resolve_categorical_msa_projection_native_matmul_state(dtype: torch.dtype) -> int:
    return captured_native_matmul_precision_state(native_matmul_domain(dtype))


def require_categorical_msa_projection_native_matmul_state(expected: int) -> None:
    require_native_matmul_precision_state("categorical_msa_projection", expected)
