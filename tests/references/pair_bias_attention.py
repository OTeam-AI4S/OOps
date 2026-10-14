"""Independent PyTorch pair-bias attention equation shared by attention tests."""

from __future__ import annotations

import torch


def pair_bias_attention_reference(query, key, value, mask=None, pair_bias=None, scale=None):
    with torch.autocast("cuda", enabled=False):
        scale = query.shape[-1] ** -0.5 if scale is None else scale
        scores = (query.float() @ key.float().transpose(-1, -2)) * scale
        if pair_bias is not None:
            scores = scores + pair_bias.float()
        keep = None
        if mask is not None:
            keep = mask.bool()
            if keep.dim() == query.dim() - 2:
                keep = keep.unsqueeze(-2).unsqueeze(-2)
            else:
                while keep.dim() < scores.dim():
                    keep = keep.unsqueeze(-3)
            scores = scores.masked_fill(~keep, float("-inf"))
            scores = torch.where(keep.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
        probabilities = torch.softmax(scores, dim=-1)
        if keep is not None:
            probabilities = probabilities.masked_fill(~keep, 0.0)
        return probabilities @ value.float()
