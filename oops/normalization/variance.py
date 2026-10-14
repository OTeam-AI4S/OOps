from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def centered_variance(centered, count):
    r"""Masked lanes in centered must be zero."""

    return tl.sum(centered * centered, axis=1) / count
