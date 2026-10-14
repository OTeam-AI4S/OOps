from __future__ import annotations

import triton
import triton.language as tl

_SE3_BASIS_NORM_EPS = tl.constexpr(1.0e-12)


@triton.jit
def _se3_basis_l1_value(
    rem,
    kind: tl.constexpr,
    use_pad_trick: tl.constexpr,
    h0,
    h1x,
    h1y,
    h1z,
    h2xz,
    h2xy,
    h2mid,
    h2yz,
    h2diff,
):
    r"""Map a packed basis entry to a linear combination of the nine harmonics."""

    value = tl.zeros_like(h0)
    last = 3
    if use_pad_trick:
        last = 4

    if kind == 0:
        value = tl.where(rem == 0, h0, value)
    elif kind == 1:
        value = tl.where(rem == 0, h1x * 0.5773502691896258, value)
        value = tl.where(rem == 1, h1y * 0.5773502691896258, value)
        value = tl.where(rem == 2, h1z * 0.5773502691896258, value)
    elif kind == 2:
        k = rem % (1 + use_pad_trick)
        in_dim = rem // (1 + use_pad_trick)
        value = tl.where((k == 0) & (in_dim == 0), h1x * 0.5773502691896258, value)
        value = tl.where((k == 0) & (in_dim == 1), h1y * 0.5773502691896258, value)
        value = tl.where((k == 0) & (in_dim == 2), h1z * 0.5773502691896258, value)
    elif kind == 3:
        k = rem % last
        block_l = rem // last
        block = block_l % 3
        in_dim = block_l // 3
        value = _se3_basis_pair_1_1_value(in_dim, block, k, h0, h1x, h1y, h1z, h2xz, h2xy, h2mid, h2yz, h2diff)
    elif kind == 4:
        k = rem % (1 + use_pad_trick)
        remaining_index = rem // (1 + use_pad_trick)
        freq = remaining_index % 2
        dim = remaining_index // 2
        value = tl.where((k == 0) & (dim == 0) & (freq == 0), h0, value)
        value = tl.where((k == 0) & (dim == 1) & (freq == 1), h1x * 0.5773502691896258, value)
        value = tl.where((k == 0) & (dim == 2) & (freq == 1), h1y * 0.5773502691896258, value)
        value = tl.where((k == 0) & (dim == 3) & (freq == 1), h1z * 0.5773502691896258, value)
    elif kind == 5:
        k = rem % last
        remaining_index = rem // last
        freq = remaining_index % 4
        dim = remaining_index // 4
        value = tl.where((k == 0) & (dim == 0) & (freq == 0), h1x * 0.5773502691896258, value)
        value = tl.where((k == 1) & (dim == 0) & (freq == 0), h1y * 0.5773502691896258, value)
        value = tl.where((k == 2) & (dim == 0) & (freq == 0), h1z * 0.5773502691896258, value)
        value = tl.where(
            (dim > 0) & (freq > 0),
            _se3_basis_pair_1_1_value(dim - 1, freq - 1, k, h0, h1x, h1y, h1z, h2xz, h2xy, h2mid, h2yz, h2diff),
            value,
        )
    elif kind == 6:
        dim = rem % 4
        freq = (rem // 4) % 2
        value = tl.where((freq == 0) & (dim == 0), h0, value)
        value = tl.where((freq == 1) & (dim == 1), h1x * 0.5773502691896258, value)
        value = tl.where((freq == 1) & (dim == 2), h1y * 0.5773502691896258, value)
        value = tl.where((freq == 1) & (dim == 3), h1z * 0.5773502691896258, value)
    elif kind == 7:
        dim = rem % 4
        remaining_index = rem // 4
        freq = remaining_index % 4
        in_dim = remaining_index // 4
        value = tl.where((freq == 0) & (dim == 0) & (in_dim == 0), h1x * 0.5773502691896258, value)
        value = tl.where((freq == 0) & (dim == 0) & (in_dim == 1), h1y * 0.5773502691896258, value)
        value = tl.where((freq == 0) & (dim == 0) & (in_dim == 2), h1z * 0.5773502691896258, value)
        value = tl.where(
            (freq > 0) & (dim > 0),
            _se3_basis_pair_1_1_value(
                in_dim,
                freq - 1,
                dim - 1,
                h0,
                h1x,
                h1y,
                h1z,
                h2xz,
                h2xy,
                h2mid,
                h2yz,
                h2diff,
            ),
            value,
        )
    else:
        dim_out = rem % 4
        remaining_index = rem // 4
        freq = remaining_index % 6
        dim = remaining_index // 6
        value = tl.where((freq == 0) & (dim_out == 0) & (dim == 0), h0, value)
        value = tl.where((freq == 1) & (dim_out == 0) & (dim == 1), h1x * 0.5773502691896258, value)
        value = tl.where((freq == 1) & (dim_out == 0) & (dim == 2), h1y * 0.5773502691896258, value)
        value = tl.where((freq == 1) & (dim_out == 0) & (dim == 3), h1z * 0.5773502691896258, value)
        value = tl.where((freq == 2) & (dim == 0) & (dim_out == 1), h1x * 0.5773502691896258, value)
        value = tl.where((freq == 2) & (dim == 0) & (dim_out == 2), h1y * 0.5773502691896258, value)
        value = tl.where((freq == 2) & (dim == 0) & (dim_out == 3), h1z * 0.5773502691896258, value)
        value = tl.where(
            (freq > 2) & (dim > 0) & (dim_out > 0),
            _se3_basis_pair_1_1_value(
                dim - 1,
                freq - 3,
                dim_out - 1,
                h0,
                h1x,
                h1y,
                h1z,
                h2xz,
                h2xy,
                h2mid,
                h2yz,
                h2diff,
            ),
            value,
        )
    return value


@triton.jit
def _se3_basis_pair_1_1_value(in_dim, block, k, h0, h1x, h1y, h1z, h2xz, h2xy, h2mid, h2yz, h2diff):
    value = tl.zeros_like(h0)
    value = tl.where((block == 0) & (in_dim == k), h0 * 0.5773502691896258, value)

    value = tl.where((block == 1) & (in_dim == 1) & (k == 0), -h1z * 0.4082482904638631, value)
    value = tl.where((block == 1) & (in_dim == 2) & (k == 0), h1y * 0.4082482904638631, value)
    value = tl.where((block == 1) & (in_dim == 0) & (k == 1), h1z * 0.4082482904638631, value)
    value = tl.where((block == 1) & (in_dim == 2) & (k == 1), -h1x * 0.4082482904638631, value)
    value = tl.where((block == 1) & (in_dim == 0) & (k == 2), -h1y * 0.4082482904638631, value)
    value = tl.where((block == 1) & (in_dim == 1) & (k == 2), h1x * 0.4082482904638631, value)

    value = tl.where(
        (block == 2) & (in_dim == 0) & (k == 0),
        -h2mid * 0.18257418583505536 - h2diff * 0.31622776601683794,
        value,
    )
    value = tl.where((block == 2) & (in_dim == 1) & (k == 0), h2xy * 0.31622776601683794, value)
    value = tl.where((block == 2) & (in_dim == 2) & (k == 0), h2xz * 0.31622776601683794, value)
    value = tl.where((block == 2) & (in_dim == 0) & (k == 1), h2xy * 0.31622776601683794, value)
    value = tl.where((block == 2) & (in_dim == 1) & (k == 1), h2mid * 0.3651483716701107, value)
    value = tl.where((block == 2) & (in_dim == 2) & (k == 1), h2yz * 0.31622776601683794, value)
    value = tl.where((block == 2) & (in_dim == 0) & (k == 2), h2xz * 0.31622776601683794, value)
    value = tl.where((block == 2) & (in_dim == 1) & (k == 2), h2yz * 0.31622776601683794, value)
    value = tl.where(
        (block == 2) & (in_dim == 2) & (k == 2),
        -h2mid * 0.18257418583505536 + h2diff * 0.31622776601683794,
        value,
    )
    return value


@triton.jit
def _se3_basis_l1_fill(
    relative_positions_ptr,
    outputs,
    edges,
    per_edges: tl.constexpr,
    use_pad_trick: tl.constexpr,
    block_edges: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64) * block_edges + tl.arange(0, block_edges)
    active = edge < edges
    x = tl.load(relative_positions_ptr + edge * 3, mask=active, other=0.0).to(tl.float32)
    y = tl.load(relative_positions_ptr + edge * 3 + 1, mask=active, other=0.0).to(tl.float32)
    z = tl.load(relative_positions_ptr + edge * 3 + 2, mask=active, other=0.0).to(tl.float32)
    inv_norm = 1.0 / tl.maximum(tl.sqrt(x * x + y * y + z * z), _SE3_BASIS_NORM_EPS)
    x *= inv_norm
    y *= inv_norm
    z *= inv_norm
    h0 = tl.full((block_edges, 1), 0.28209479177387814, tl.float32)
    h1x = (0.4886025119029199 * x)[:, None]
    h1y = (0.4886025119029199 * y)[:, None]
    h1z = (0.4886025119029199 * z)[:, None]
    h2xz = (1.0925484305920792 * x * z)[:, None]
    h2xy = (1.0925484305920792 * x * y)[:, None]
    h2mid = (0.31539156525252005 * (2 * y * y - x * x - z * z))[:, None]
    h2yz = (1.0925484305920792 * y * z)[:, None]
    h2diff = (0.5462742152960396 * (z * z - x * x))[:, None]
    for kind in tl.static_range(len(per_edges)):
        entries = tl.arange(0, triton.next_power_of_2(per_edges[kind]))
        value = _se3_basis_l1_value(
            entries[None, :], kind, use_pad_trick, h0, h1x, h1y, h1z, h2xz, h2xy, h2mid, h2yz, h2diff
        )
        tl.store(
            outputs[kind] + edge[:, None] * per_edges[kind] + entries[None, :],
            value,
            mask=active[:, None] & (entries < per_edges[kind])[None, :],
        )


@triton.jit
def _se3_basis_l1_grad_harmonics(
    inputs,
    grad_harmonics_ptr,
    edges,
    per_edges: tl.constexpr,
    present: tl.constexpr,
    use_pad_trick: tl.constexpr,
    block_edges: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64) * block_edges + tl.arange(0, block_edges)
    active = edge < edges
    g0 = tl.zeros((block_edges,), tl.float32)
    g1 = tl.zeros((block_edges,), tl.float32)
    g2 = tl.zeros((block_edges,), tl.float32)
    g3 = tl.zeros((block_edges,), tl.float32)
    g4 = tl.zeros((block_edges,), tl.float32)
    g5 = tl.zeros((block_edges,), tl.float32)
    g6 = tl.zeros((block_edges,), tl.float32)
    g7 = tl.zeros((block_edges,), tl.float32)
    g8 = tl.zeros((block_edges,), tl.float32)
    for kind in tl.static_range(len(per_edges)):
        if present[kind]:
            entries = tl.arange(0, triton.next_power_of_2(per_edges[kind]))
            grad = tl.load(
                inputs[kind] + edge[:, None] * per_edges[kind] + entries[None, :],
                mask=active[:, None] & (entries < per_edges[kind])[None, :],
                other=0,
            ).to(tl.float32)
            one = tl.full((triton.next_power_of_2(per_edges[kind]),), 1, tl.float32)
            zero = tl.full((triton.next_power_of_2(per_edges[kind]),), 0, tl.float32)
            coeff0 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, one, zero, zero, zero, zero, zero, zero, zero, zero
            )
            g0 += tl.sum(grad * coeff0[None, :], axis=1)
            coeff1 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, one, zero, zero, zero, zero, zero, zero, zero
            )
            g1 += tl.sum(grad * coeff1[None, :], axis=1)
            coeff2 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, zero, one, zero, zero, zero, zero, zero, zero
            )
            g2 += tl.sum(grad * coeff2[None, :], axis=1)
            coeff3 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, zero, zero, one, zero, zero, zero, zero, zero
            )
            g3 += tl.sum(grad * coeff3[None, :], axis=1)
            coeff4 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, zero, zero, zero, one, zero, zero, zero, zero
            )
            g4 += tl.sum(grad * coeff4[None, :], axis=1)
            coeff5 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, zero, zero, zero, zero, one, zero, zero, zero
            )
            g5 += tl.sum(grad * coeff5[None, :], axis=1)
            coeff6 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, zero, zero, zero, zero, zero, one, zero, zero
            )
            g6 += tl.sum(grad * coeff6[None, :], axis=1)
            coeff7 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, zero, zero, zero, zero, zero, zero, one, zero
            )
            g7 += tl.sum(grad * coeff7[None, :], axis=1)
            coeff8 = _se3_basis_l1_value(
                entries, kind, use_pad_trick, zero, zero, zero, zero, zero, zero, zero, zero, one
            )
            g8 += tl.sum(grad * coeff8[None, :], axis=1)
    tl.store(grad_harmonics_ptr + edge * 9 + 0, g0, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 1, g1, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 2, g2, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 3, g3, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 4, g4, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 5, g5, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 6, g6, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 7, g7, mask=active)
    tl.store(grad_harmonics_ptr + edge * 9 + 8, g8, mask=active)
