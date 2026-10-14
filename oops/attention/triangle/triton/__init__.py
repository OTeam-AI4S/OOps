from __future__ import annotations

from .ops import supports_triangle_attention, triton_triangle_attention, uniform_sample_table

__all__ = [
    "supports_triangle_attention",
    "triton_triangle_attention",
    "uniform_sample_table",
]
