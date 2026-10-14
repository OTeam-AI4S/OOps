from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _maximum(left, right):
    return tl.maximum(left, right, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _divide(left, right):
    if left.dtype == tl.float32:
        return tl.div_rn(left, right)
    return left / right


@triton.jit
def _ldexp(value, exponent):
    if value.dtype == tl.float32:
        bits = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        subnormal = (bits > 0) & (bits < 0x00800000)
        # CUDA libdevice's float32 exponent helpers flush subnormal inputs.
        value = tl.where(subnormal, value * 16777216.0, value)
        exponent -= tl.where(subnormal, 24, 0)
    return libdevice.ldexp(value, exponent)


@triton.jit
def _decompose(value):
    shifted = value
    adjustment = tl.full(value.shape, 0, tl.int32)
    if value.dtype == tl.float32:
        bits = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        subnormal = (bits > 0) & (bits < 0x00800000)
        shifted = tl.where(subnormal, value * 16777216.0, value)
        adjustment = tl.where(subnormal, 24, 0)
    exponent = tl.where((value != 0) & (tl.abs(value) != float("inf")), libdevice.ilogb(shifted) - adjustment, 0)
    return _ldexp(value, -exponent), exponent


@triton.jit
def _norm_state(
    vectors,
    row,
    rows,
    WIDTH: tl.constexpr,
    EPS_MANTISSA: tl.constexpr,
    EPS_EXPONENT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    channels = tl.arange(0, BLOCK_C)
    work: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    maximum = tl.full((row.shape[0],), 0, work)
    for start in range(tl.cdiv(WIDTH, BLOCK_C)):
        index = start * BLOCK_C + channels
        values = tl.load(
            vectors + row[:, None] * WIDTH + index[None, :], (row[:, None] < rows) & (index[None, :] < WIDTH), 0
        ).to(work)
        maximum = _maximum(maximum, tl.reduce(tl.abs(values), 1, _maximum))
    scale_mantissa, scale_exponent = _decompose(maximum)
    use_epsilon = (EPS_MANTISSA != 0) & (
        (maximum == 0)
        | (EPS_EXPONENT > scale_exponent)
        | ((EPS_EXPONENT == scale_exponent) & (EPS_MANTISSA > scale_mantissa))
    )
    scale_mantissa = tl.where(use_epsilon, EPS_MANTISSA, scale_mantissa)
    scale_exponent = tl.where(use_epsilon, EPS_EXPONENT, scale_exponent)
    safe_scale = tl.where(scale_mantissa == 0, 1, scale_mantissa)
    square_sum = tl.full((row.shape[0],), 0, work)
    for start in range(tl.cdiv(WIDTH, BLOCK_C)):
        index = start * BLOCK_C + channels
        values = tl.load(
            vectors + row[:, None] * WIDTH + index[None, :], (row[:, None] < rows) & (index[None, :] < WIDTH), 0
        ).to(work)
        scaled = _divide(_ldexp(values, -scale_exponent[:, None]), safe_scale[:, None])
        square_sum += tl.sum(scaled * scaled, 1)
    epsilon = _divide(_ldexp(tl.full((row.shape[0],), EPS_MANTISSA, work), EPS_EXPONENT - scale_exponent), safe_scale)
    factor = libdevice.sqrt(square_sum + epsilon * epsilon)
    return scale_mantissa, scale_exponent, factor, maximum


@triton.jit
def _forward(
    vectors,
    output,
    rows,
    WIDTH: tl.constexpr,
    EPS_MANTISSA: tl.constexpr,
    EPS_EXPONENT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mantissa, exponent, factor, maximum = _norm_state(
        vectors, row, rows, WIDTH, EPS_MANTISSA, EPS_EXPONENT, BLOCK_C, DOUBLE
    )
    value = _ldexp(mantissa * factor, exponent)
    value = tl.where(maximum == float("inf"), float("inf"), value)
    tl.store(output + row, value, row < rows)


@triton.jit
def _backward(
    vectors,
    upstream,
    gradient,
    rows,
    WIDTH: tl.constexpr,
    EPS_MANTISSA: tl.constexpr,
    EPS_EXPONENT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_C)
    work: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    mantissa, exponent, factor, _ = _norm_state(vectors, row, rows, WIDTH, EPS_MANTISSA, EPS_EXPONENT, BLOCK_C, DOUBLE)
    denominator = tl.where(factor == 0, 1, mantissa * factor)
    cotangent = tl.load(upstream + row, row < rows, 0).to(work)
    cotangent_mantissa, cotangent_exponent = _decompose(cotangent)
    for start in range(tl.cdiv(WIDTH, BLOCK_C)):
        index = start * BLOCK_C + channels
        mask = (row[:, None] < rows) & (index[None, :] < WIDTH)
        value = tl.load(vectors + row[:, None] * WIDTH + index[None, :], mask, 0).to(work)
        value_mantissa, value_exponent = _decompose(value)
        result = _ldexp(
            _divide(cotangent_mantissa[:, None] * value_mantissa, denominator[:, None]),
            cotangent_exponent[:, None] + value_exponent - exponent[:, None],
        )
        tl.store(gradient + row[:, None] * WIDTH + index[None, :], result, mask)
