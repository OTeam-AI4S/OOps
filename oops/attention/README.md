# Attention

Use pair-bias attention to combine token features with an additive pair signal. Use triangle attention
when each row of a pair representation needs to attend across its columns. The gating functions turn
attention values into projected token features.

## Usage

This CUDA example applies two attention heads to four tokens, keeping only the first two keys:

```python
import torch
from oops import pair_bias_attention

input = torch.randn(2, 4, 8, device="cuda")
mask = torch.tensor([True, True, False, False], device="cuda")
pair_bias = torch.zeros(2, 4, 4, device="cuda")
output = pair_bias_attention(input, input, input, mask=mask, pair_bias=pair_bias)
assert output.shape == input.shape
```

## Key Concepts

`pair_bias_attention` adds pair bias to token attention logits. `triangle_attention` attends over the
columns of each pair-feature row.

A boolean mask selects keys or pairs: `True` keeps them and `False` excludes them. A fully masked row
returns zero. Pair bias is additive.

`gate_attention_output` applies a sigmoid gate from the single features and then an output projection.
`gated_pair_bias_attention` combines that stage with pair-bias attention.

## Public API

| API                         | When to Use                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `pair_bias_attention`       | Attend over token features with an additive pair bias.         |
| `triangle_attention`        | Attend over the columns of each pair-feature row.              |
| `gate_attention_output`     | Gate and project existing attention values.                    |
| `gated_pair_bias_attention` | Compute pair-bias attention and its gated projection together. |
