from __future__ import annotations

import torch
from torch import Tensor


def native_forward(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    x, w = values.double(), weights.double()
    x = torch.where((w == 0) & ~torch.isfinite(x), 0.0, x)
    if values.dtype == torch.float64 or weights.dtype == torch.float64:
        xm, xe = torch.frexp(x)
        wm, we = torch.frexp(w)
        identity = torch.full_like(w.sum(-1, keepdim=True), -4096, dtype=torch.int32)
        de = torch.cat((torch.where(w != 0, we, -4096), identity), -1).amax(-1, keepdim=True)
        ne = torch.cat((torch.where((x != 0) & (w != 0), xe + we, -4096), identity), -1).amax(-1, keepdim=True)
        d = torch.ldexp(wm, we - de).sum(-1, keepdim=True)
        n = torch.ldexp(xm * wm, xe + we - ne).sum(-1, keepdim=True)
        nm, nen = torch.frexp(n)
        dm, den = torch.frexp(d)
        mm = nm / torch.where(d > 0, dm, 1.0)
        me = ne - de + nen - den
        mu = torch.ldexp(mm, me)
    else:
        d = w.sum(-1, keepdim=True)
        de = torch.zeros_like(d, dtype=torch.int32)
        mu = (x * w).sum(-1, keepdim=True) / torch.where(d > 0, d, 1.0)
        mm, me = torch.frexp(mu)
    present = d > 0
    mu = torch.where(present, mu, 0.0)
    dtype = torch.promote_types(values.dtype, weights.dtype) if weights.is_floating_point() else values.dtype
    output = torch.where(present, mu, float("nan")).squeeze(-1).to(dtype)
    return output, torch.cat((torch.where(present, mm, 0.0), me.double(), d, de.double()), -1)


def native_backward(values: Tensor, weights: Tensor, stats: Tensor, grad: Tensor) -> tuple[Tensor, Tensor]:
    x, w, g = values.double(), weights.double(), grad.double().unsqueeze(-1)
    mm, me, d, de = (stats[..., i : i + 1] for i in range(4))
    mu = torch.ldexp(mm, me.to(torch.int32))
    x = torch.where((w == 0) & ~torch.isfinite(x), 0.0, x)
    present = d > 0
    denominator = torch.where(present, d, 1.0)
    if values.dtype == torch.float64 or weights.dtype == torch.float64:
        gm, ge = torch.frexp(g)
        wm, we = torch.frexp(w)
        gx = torch.ldexp(gm * wm / denominator, ge + we - de.to(torch.int32))
        xm, xe = torch.frexp(x)
        me = me.to(torch.int32)
        delta_e = torch.maximum(torch.where(x != 0, xe, -4096), torch.where(mm != 0, me, -4096))
        delta = torch.ldexp(xm, xe - delta_e) - torch.ldexp(mm.expand_as(xm), me - delta_e)
        gw = torch.ldexp(gm * delta / denominator, ge + delta_e - de.to(torch.int32))
    else:
        gx = g * w / denominator
        gw = g * (x - mu) / denominator
    gx = torch.where(present & (w != 0), gx, 0.0)
    gw = torch.where(present, gw, 0.0)
    return gx, gw
