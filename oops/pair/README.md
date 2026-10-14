# Pair Updates

Use triangle multiplication to update a token-pair representation through a third token. The same
function implements incoming and outgoing contractions; choose the direction used by your model.

## Usage

Supply the model's normalization and projection parameters explicitly. This small CUDA example uses
identity projections to show the complete call:

```python
import torch
from oops import triangle_multiplication

input = torch.randn(4, 4, 8, device="cuda")
projection = torch.eye(8, device="cuda")
scale = torch.ones(8, device="cuda")
bias = torch.zeros(8, device="cuda")
parameters = {
    "input_norm_weight": scale,
    "input_norm_bias": bias,
    "left_value_weight": projection,
    "left_gate_weight": projection,
    "right_value_weight": projection,
    "right_gate_weight": projection,
    "output_norm_weight": scale,
    "output_norm_bias": bias,
    "output_weight": projection,
    "output_gate_weight": projection,
}
output = triangle_multiplication(input, direction="outgoing", **parameters)
assert output.shape == input.shape
```

## Key Concepts

The two token axes are square. A pair mask multiplies the projected features.

The function returns an update. Add the residual in model code when the architecture calls for it.
The output gate reads the input-normalized pair, and projection biases precede sigmoid gates. These
choices define the triangle update together with the contraction direction.

## Public API

| API                       | When to Use                                                 |
| ------------------------- | ----------------------------------------------------------- |
| `triangle_multiplication` | Compute an incoming or outgoing multiplicative pair update. |

## Related API

[Triangle attention](../attention/README.md) and [MSA outer-product mean](../msa/README.md) also update pair
features. See [normalization](../normalization/README.md) for standalone `LayerNorm` and the
[operator overview](../README.md) for shared conventions.
