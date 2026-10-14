# Weighted Reduction

Average observations with real weights or validity masks, including variable-length `NestedTensor` inputs.

## Usage

Compute a CPU weighted mean and differentiate both the observations and their weights:

```python
import torch
from oops import weighted_mean

input = torch.tensor([2.0, 8.0], requires_grad=True)
weights = torch.tensor([3.0, 1.0], requires_grad=True)
output = weighted_mean(input, weights)
output.backward()
assert output.item() == 3.5
```

## Key Concepts

The function broadcasts values and weights, then evaluates `sum(weights * values) / sum(weights)` over
the final observation axis. Pass finite, nonnegative weights.

Empty observations or zero total weight return NaN. Zero-weight observations do not contribute to the
mean, including nonfinite observations; positive-weight nonfinite values propagate.

## Public API

| API             | When to Use                                                    |
| --------------- | -------------------------------------------------------------- |
| `weighted_mean` | Average the final observation axis with coefficients or masks. |

## Related API

See [CSR reductions](../sparse/README.md) to group edges by row, [atom aggregation](../atom/README.md) for
atom–token pooling, and [losses](../losses/README.md) for model objectives. Return to the
[operator overview](../README.md).
