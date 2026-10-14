# MSA Operators

Turn categorical multiple-sequence alignment (MSA) tokens into model features, or summarize MSA features
as a pair update.
OFoldX handles alignment acquisition and sequence pairing before these feature computations.

## Usage

Project three token classes and two continuous channels into four feature channels on CUDA:

```python
import torch
from oops import categorical_msa_projection

input = torch.tensor([[0, 1, 2], [2, 1, 0]], device="cuda")
continuous = torch.zeros(2, 3, 2, device="cuda")
weight = torch.randn(4, 5, device="cuda")
output = categorical_msa_projection(input, continuous, weight)
assert output.shape == (2, 3, 4)
```

## Key Concepts

Categorical projection is equivalent to a bias-free linear projection of one-hot token features
concatenated with the supplied continuous features.

`outer_product_mean` reduces alignment sequences into a square token-pair update. Its MSA mask is
multiplicative: floating weights contribute to both the contraction and its counts. Add any residual
outside the function.

Choose `divide_before_projection`, projection biases, and denominator settings to match the model's
OPM equation. Dividing the outer product before a biased output projection differs from dividing the
projected value afterward.

## Public API

| API                          | When to Use                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| `categorical_msa_projection` | Embed categorical alignment tokens and continuous features. |
| `outer_product_mean`         | Reduce MSA features into a pair representation.             |

## Related API

Prepare validity with [MSA/token masks](../mask/README.md), then see [pair updates](../pair/README.md) and
[attention](../attention/README.md) for downstream computations. Return to the [operator overview](../README.md).
