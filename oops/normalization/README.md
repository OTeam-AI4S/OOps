# Normalization

Use `LayerNorm` as a module in a neural network, or call `layer_norm` when supplying normalization
parameters directly to the CUDA implementation.

## Usage

Normalize the final feature axis and differentiate through the input:

```python
import torch
from oops import LayerNorm

layer = LayerNorm(16)
input = torch.randn(2, 4, 16, requires_grad=True)
output = layer(input)
output.square().mean().backward()
```

The [LayerNorm script](../../examples/layer_norm.py) runs dense and variable-length examples.

## Key Concepts

`LayerNorm` accepts the constructor arguments of `torch.nn.LayerNorm` and works with dense and
variable-length tensors. `layer_norm` is the functional CUDA interface.

Normalization operates over the final feature axis.

## Public API

| API          | When to Use                                         |
| ------------ | --------------------------------------------------- |
| `LayerNorm`  | Use a PyTorch-compatible normalization module.      |
| `layer_norm` | Normalize features with explicit affine parameters. |
