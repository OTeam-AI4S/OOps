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

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _row_offset(row, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full((), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (row % SHAPE[axis]).to(tl.int64) * STRIDES[axis]
        row //= SHAPE[axis]
    return offset


@triton.jit
def _frexp(value):
    finite_nonzero = (value != 0) & (tl.abs(value) != float("inf")) & (value == value)
    exponent = tl.where(finite_nonzero, libdevice.ilogb(tl.abs(value)) + 1, 0)
    mantissa = libdevice.ldexp(value, -exponent)
    return mantissa, exponent


@triton.jit
def _partial(
    X,
    W,
    PARTIAL,
    OUT,
    STATS,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    PARTS: tl.constexpr,
    SCALED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0) // PARTS, tl.program_id(0) % PARTS
    index = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + _row_offset(row, SHAPE, XS) + index * XS[-1], index < N, 0).to(tl.float64)
    w = tl.load(W + _row_offset(row, SHAPE, WS) + index * WS[-1], index < N, 0).to(tl.float64)
    x = tl.where((w == 0) & ((tl.abs(x) == float("inf")) | (x != x)), 0.0, x)
    if SCALED:
        xm, xe = _frexp(x)
        wm, we = _frexp(w)
        de = tl.max(tl.where(w != 0, we, -4096), 0)
        ne = tl.max(tl.where((x != 0) & (w != 0), xe + we, -4096), 0)
        d = tl.sum(libdevice.ldexp(wm, we - de), 0)
        n = tl.sum(libdevice.ldexp(xm * wm, xe + we - ne), 0)
    else:
        d, n = tl.sum(w, 0), tl.sum(x * w, 0)
        de, ne = 0, 0
    if PARTS == 1:
        _store_mean(OUT, STATS, row, n, d, ne, de)
    else:
        base = (row * PARTS + part) * 4
        tl.store(PARTIAL + base, n)
        tl.store(PARTIAL + base + 1, d)
        tl.store(PARTIAL + base + 2, ne)
        tl.store(PARTIAL + base + 3, de)


@triton.jit
def _store_mean(OUT, STATS, row, n, d, ne, de):
    nm, nen = _frexp(n)
    dm, den = _frexp(d)
    mm = tl.where(d > 0, nm / tl.where(d > 0, dm, 1.0), 0.0)
    me = ne - de + nen - den
    tl.store(OUT + row, tl.where(d > 0, libdevice.ldexp(mm, me), float("nan")))
    tl.store(STATS + row * 4, mm)
    tl.store(STATS + row * 4 + 1, me)
    tl.store(STATS + row * 4 + 2, d)
    tl.store(STATS + row * 4 + 3, de)


@triton.jit
def _finish(PARTIAL, NEXT, OUT, STATS, PARTS: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    row, group = tl.program_id(0) // GROUPS, tl.program_id(0) % GROUPS
    index = group * BLOCK + tl.arange(0, BLOCK)
    base = (row * PARTS + index) * 4
    n = tl.load(PARTIAL + base, index < PARTS, 0)
    d = tl.load(PARTIAL + base + 1, index < PARTS, 0)
    ne = tl.load(PARTIAL + base + 2, index < PARTS, -4096).to(tl.int32)
    de = tl.load(PARTIAL + base + 3, index < PARTS, -4096).to(tl.int32)
    ne_max = tl.max(tl.where(n != 0, ne, -4096), 0)
    de_max = tl.max(tl.where(d > 0, de, -4096), 0)
    n = tl.sum(libdevice.ldexp(n, ne - ne_max), 0)
    d = tl.sum(libdevice.ldexp(d, de - de_max), 0)
    if GROUPS == 1:
        _store_mean(OUT, STATS, row, n, d, ne_max, de_max)
    else:
        next_base = (row * GROUPS + group) * 4
        tl.store(NEXT + next_base, n)
        tl.store(NEXT + next_base + 1, d)
        tl.store(NEXT + next_base + 2, ne_max)
        tl.store(NEXT + next_base + 3, de_max)


@triton.jit
def _backward(
    X,
    W,
    STATS,
    G,
    GX,
    GW,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    GS: tl.constexpr,
    SCALED: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    parts: tl.constexpr = tl.cdiv(N, BLOCK)
    row, part = tl.program_id(0) // parts, tl.program_id(0) % parts
    index = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + _row_offset(row, SHAPE, XS) + index * XS[-1], index < N, 0).to(tl.float64)
    w = tl.load(W + _row_offset(row, SHAPE, WS) + index * WS[-1], index < N, 0).to(tl.float64)
    x = tl.where((w == 0) & ((tl.abs(x) == float("inf")) | (x != x)), 0.0, x)
    g = tl.load(G + _row_offset(row, SHAPE, GS)).to(tl.float64)
    mm = tl.load(STATS + row * 4)
    me = tl.load(STATS + row * 4 + 1).to(tl.int32)
    d = tl.load(STATS + row * 4 + 2)
    de = tl.load(STATS + row * 4 + 3).to(tl.int32)
    denom = tl.where(d > 0, d, 1.0)
    if SCALED:
        gm, ge = _frexp(g)
        wm, we = _frexp(w)
        gx = libdevice.ldexp(gm * wm / denom, ge + we - de)
        xm, xe = _frexp(x)
        delta_e = tl.maximum(tl.where(x != 0, xe, -4096), tl.where(mm != 0, me, -4096))
        delta = libdevice.ldexp(xm, xe - delta_e) - libdevice.ldexp(mm, me - delta_e)
        gw = libdevice.ldexp(gm * delta / denom, ge + delta_e - de)
    else:
        mu = libdevice.ldexp(mm, me)
        gx, gw = g * w / denom, g * (x - mu) / denom
    if NEED_X:
        tl.store(GX + row * N + index, tl.where((d > 0) & (w != 0), gx, 0.0), index < N)
    if NEED_W:
        tl.store(GW + row * N + index, tl.where(d > 0, gw, 0.0), index < N)
