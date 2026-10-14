# Operators

Choose an operator for the representation you are working with: tokens, pair features, alignments,
atoms, or sparse edges. Public functions can be imported from `oops` or their family namespace.

## Usage

For example, compute vector lengths while retaining gradients to the input:

```python
import torch
from oops import smoothed_vector_norm

input = torch.tensor([[3.0, 4.0]], requires_grad=True)
output = smoothed_vector_norm(input, eps=0)
output.sum().backward()
assert output.tolist() == [5.0]
```

## Operator Map

| Family                                   | When to Use                                                   |
| ---------------------------------------- | ------------------------------------------------------------- |
| [Attention](attention/README.md)         | Attend to token features or pair-feature rows.                |
| [Pair](pair/README.md)                   | Build incoming or outgoing triangle updates.                  |
| [MSA](msa/README.md)                     | Project alignment tokens or reduce MSA features into pairs.   |
| [Normalization](normalization/README.md) | Normalize feature channels through a module or function.      |
| [Atom](atom/README.md)                   | Transfer between atom/token features and gather atom windows. |
| [Sparse](sparse/README.md)               | Group edges and aggregate features within CSR rows.           |
| [Geometry](geometry/README.md)           | Select neighbors or compute differentiable vector lengths.    |
| [Equivariant](equivariant/README.md)     | Build spherical-harmonic features and SE(3) bases.            |
| [Losses](losses/README.md)               | Compute classification or structural objectives.              |
| [Mask](mask/README.md)                   | Convert MSA/token validity into boolean masks.                |
| [Reduction](reduction/README.md)         | Average observations with weights or masks.                   |

## Masks

Attention uses boolean keep-masks. OPM and triangle multiplication can use floating multiplicative weights
with gradients. The [mask functions](mask/README.md) convert nonzero values to boolean validity, so retain
the original weights when their magnitudes matter.

## OFoldX Integration

Use the [OFoldX backend guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/modules/backend/README.md)
to select `auto`, `eager`, or `oops` in an OFoldX workflow.

## Related API

See the [examples](../examples/README.md) for runnable scripts.
