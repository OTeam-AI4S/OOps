# Geometry

Find neighbors within variable-length molecular samples, or compute vector lengths while retaining
coordinate gradients.

## Usage

Find two neighbors for every point in one CUDA sample:

```python
import torch
from danling import NestedTensor
from oops import masked_knn

input = NestedTensor(
    [torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda")],
    ragged_dims=(0,),
)
mask = NestedTensor([torch.tensor([True, True], device="cuda")], ragged_dims=(0,))
output = masked_knn(input, mask, num_neighbors=2, eps=0)
distances, indices, valid = output
```

The three returned nested tensors contain distances, sample-local indices, and validity for each slot.

## Key Concepts

`masked_knn` searches within each sample, includes a valid query itself, and breaks equal-distance ties by
local index. `valid` identifies real neighbors; other slots contain filler values. Empty samples are
supported, and samples remain independent.

Distances retain coordinate gradients; neighbor indices are discrete.

`smoothed_vector_norm` computes `sqrt(sum(input**2, dim=-1) + eps)`. `keepdim` controls whether the
length-one vector axis is retained.

## Public API

| API                    | When to Use                                                |
| ---------------------- | ---------------------------------------------------------- |
| `masked_knn`           | Select a fixed number of masked neighbors per point.       |
| `smoothed_vector_norm` | Compute differentiable vector lengths over the final axis. |

## Related API

Pass neighbor features to [sparse graph operations](../sparse/README.md), use relative vectors for
[equivariant bases](../equivariant/README.md), or compare predicted geometry through
[structural losses](../losses/README.md). See the [operator overview](../README.md).
