from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _spherical_harmonics_forward(
    vectors_ptr,
    out_ptr,
    num_vectors,
    lmax: tl.constexpr,
    normalize: tl.constexpr,
    normalization: tl.constexpr,
    eps: tl.constexpr,
    block_vectors: tl.constexpr,
) -> None:
    block = tl.program_id(0).to(tl.int64)
    offsets = block * block_vectors + tl.arange(0, block_vectors)
    mask = offsets < num_vectors
    element_base = offsets

    x = tl.load(vectors_ptr + element_base * 3, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(vectors_ptr + element_base * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(vectors_ptr + element_base * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    if normalize:
        norm = tl.sqrt(x * x + y * y + z * z)
        inv_norm = 1.0 / tl.maximum(norm, eps)
        x *= inv_norm
        y *= inv_norm
        z *= inv_norm

    coeffs = (lmax + 1) * (lmax + 1)
    scale0 = 1.0
    scale1 = 1.0
    scale2 = 1.0
    if normalization == 1:
        scale0 = 3.5449077018110318
        scale1 = 3.5449077018110318
        scale2 = 3.5449077018110318
    elif normalization == 2:
        scale0 = 3.5449077018110318
        scale1 = 2.046653415892977
        scale2 = 1.5853309190550713

    tl.store(out_ptr + element_base * coeffs, 0.28209479177387814 * scale0, mask=mask)
    if lmax >= 1:
        degree1_scale = 0.4886025119029199 * scale1
        base = element_base * coeffs + 1
        tl.store(out_ptr + base, degree1_scale * x, mask=mask)
        tl.store(out_ptr + base + 1, degree1_scale * y, mask=mask)
        tl.store(out_ptr + base + 2, degree1_scale * z, mask=mask)

    if lmax >= 2:
        cross_scale = 1.0925484305920792 * scale2
        middle_scale = 0.31539156525252005 * scale2
        difference_scale = 0.5462742152960396 * scale2
        base = element_base * coeffs + 4
        xx = x * x
        yy = y * y
        zz = z * z
        tl.store(out_ptr + base, cross_scale * x * z, mask=mask)
        tl.store(out_ptr + base + 1, cross_scale * x * y, mask=mask)
        tl.store(out_ptr + base + 2, middle_scale * (2.0 * yy - xx - zz), mask=mask)
        tl.store(out_ptr + base + 3, cross_scale * y * z, mask=mask)
        tl.store(out_ptr + base + 4, difference_scale * (zz - xx), mask=mask)


@triton.jit
def _spherical_harmonics_backward(
    vectors_ptr,
    grad_out_ptr,
    grad_vectors_ptr,
    num_vectors,
    lmax: tl.constexpr,
    normalize: tl.constexpr,
    normalization: tl.constexpr,
    eps: tl.constexpr,
    block_vectors: tl.constexpr,
) -> None:
    block = tl.program_id(0).to(tl.int64)
    offsets = block * block_vectors + tl.arange(0, block_vectors)
    mask = offsets < num_vectors
    element_base = offsets

    x = tl.load(vectors_ptr + element_base * 3, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(vectors_ptr + element_base * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(vectors_ptr + element_base * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(x * x + y * y + z * z)
    inv_norm = 1.0 / tl.maximum(norm, eps)
    if normalize:
        x *= inv_norm
        y *= inv_norm
        z *= inv_norm

    coeffs = (lmax + 1) * (lmax + 1)
    # Degree zero is constant, so its vector derivative is zero.
    scale1 = 1.0
    scale2 = 1.0
    if normalization == 1:
        scale1 = 3.5449077018110318
        scale2 = 3.5449077018110318
    elif normalization == 2:
        scale1 = 2.046653415892977
        scale2 = 1.5853309190550713

    grad_x = tl.zeros((block_vectors,), dtype=tl.float32)
    grad_y = tl.zeros((block_vectors,), dtype=tl.float32)
    grad_z = tl.zeros((block_vectors,), dtype=tl.float32)

    if lmax >= 1:
        degree1_scale = 0.4886025119029199 * scale1
        base = element_base * coeffs + 1
        grad_1x = tl.load(grad_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        grad_1y = tl.load(grad_out_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        grad_1z = tl.load(grad_out_ptr + base + 2, mask=mask, other=0.0).to(tl.float32)
        grad_x += degree1_scale * grad_1x
        grad_y += degree1_scale * grad_1y
        grad_z += degree1_scale * grad_1z

    if lmax >= 2:
        cross_scale = 1.0925484305920792 * scale2
        middle_scale = 0.31539156525252005 * scale2
        difference_scale = 0.5462742152960396 * scale2
        base = element_base * coeffs + 4
        grad_xz = tl.load(grad_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        grad_xy = tl.load(grad_out_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        grad_mid = tl.load(grad_out_ptr + base + 2, mask=mask, other=0.0).to(tl.float32)
        grad_yz = tl.load(grad_out_ptr + base + 3, mask=mask, other=0.0).to(tl.float32)
        grad_difference = tl.load(grad_out_ptr + base + 4, mask=mask, other=0.0).to(tl.float32)
        middle = middle_scale * grad_mid
        difference = difference_scale * grad_difference
        grad_x += cross_scale * (z * grad_xz + y * grad_xy) - 2.0 * x * (middle + difference)
        grad_y += cross_scale * (x * grad_xy + z * grad_yz) + 4.0 * y * middle
        grad_z += cross_scale * (x * grad_xz + y * grad_yz) - 2.0 * z * (middle - difference)

    if normalize:
        # Normalization Jacobian: (I - u u^T) / norm above eps; I / eps at or below eps.
        projection = x * grad_x + y * grad_y + z * grad_z
        above_clamp = norm > eps
        grad_x = inv_norm * tl.where(above_clamp, grad_x - x * projection, grad_x)
        grad_y = inv_norm * tl.where(above_clamp, grad_y - y * projection, grad_y)
        grad_z = inv_norm * tl.where(above_clamp, grad_z - z * projection, grad_z)

    tl.store(grad_vectors_ptr + element_base * 3, grad_x, mask=mask)
    tl.store(grad_vectors_ptr + element_base * 3 + 1, grad_y, mask=mask)
    tl.store(grad_vectors_ptr + element_base * 3 + 2, grad_z, mask=mask)
