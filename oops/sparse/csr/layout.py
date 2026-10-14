from __future__ import annotations

from torch import Tensor

SUPPORTED_LAYOUT = (
    "requires contiguous CUDA float16/bfloat16/float32/float64 values with a non-empty "
    "edge axis, output-segment and flattened feature grids within CUDA launch limits, and a contiguous CUDA "
    "int32/int64 row-offset vector"
)


def num_edges(values: Tensor, name: str) -> int:
    if values.dim() < 1:
        raise ValueError(f"{name} must have shape (E, ...)")
    return values.shape[0]


def layout_context(values: Tensor, offsets: Tensor) -> dict[str, object]:
    return {
        "dtype": values.dtype,
        "device": values.device,
        "shape": tuple(values.shape),
        "contiguous": values.is_contiguous(),
        "row_offsets_dtype": offsets.dtype,
    }
