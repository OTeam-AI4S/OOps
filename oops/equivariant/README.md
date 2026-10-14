# Equivariant Bases

Build directional features for a geometric model. `spherical_harmonics` evaluates real harmonics;
`se3_basis` supplies the degree-one SE(3) basis convention used by RFdiffusion.

## Usage

Evaluate degrees zero through two for two vectors on CUDA:

```python
import torch
from oops import spherical_harmonics

input = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
output = spherical_harmonics(input, lmax=2)
assert output.shape == (2, 9)
```

## Key Concepts

Choose the normalization convention expected by the model: integral, component, or norm. `normalize`
controls vector normalization, and `return_list=True` separates the result into one tensor per degree.
The concatenated result has final width `(lmax + 1)**2`.

`se3_basis` returns RFdiffusion-compatible degree-one SE(3) bases.

## Public API

| API                   | When to Use                                             |
| --------------------- | ------------------------------------------------------- |
| `spherical_harmonics` | Evaluate real harmonics for `0 <= lmax <= 2`.           |
| `se3_basis`           | Build RFdiffusion-compatible bases with `max_degree=1`. |
