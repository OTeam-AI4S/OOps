# Losses

Train against class labels or compare predicted molecular geometry with a reference. Choose dense
smooth-LDDT for its distance-cutoff selection, or sparse smooth-LDDT when supplying the pair set yourself.

## Usage

Compute per-observation class losses on CUDA, then average them for backward:

```python
import torch
from oops import cross_entropy

input = torch.randn(2, 4, device="cuda", requires_grad=True)
labels = torch.tensor([0, 3], device="cuda")
output = cross_entropy(input, labels)
output.mean().backward()
```

## Key Concepts

Classification logits keep classes on the final axis, and labels contain integer class or bin indices.
`cross_entropy` returns unreduced losses and has no `ignore_index` argument. Distogram loss uses the
required token mask to weight off-diagonal pairs by the product of their token weights; its `pair_mask`
argument is unsupported.

Dense smooth-LDDT excludes diagonals and pairs outside the row's reference-distance cutoff. Nucleotide
indicators select the cutoff, and floating atom weights control pair contributions. Sparse smooth-LDDT
uses the supplied pair order, duplicates, and diagonal entries without implicit cutoff filtering.

Both smooth-LDDT losses return one when no valid or selected pairs remain.

## Public API

| API                          | When to Use                                                       |
| ---------------------------- | ----------------------------------------------------------------- |
| `cross_entropy`              | Compute unreduced class-index cross entropy.                      |
| `distogram_cross_entropy`    | Train token-masked distance-bin predictions.                      |
| `smooth_lddt_loss_dense`     | Compare dense atom pairs selected by reference cutoffs and masks. |
| `smooth_lddt_loss_sparse`    | Compare an explicit selected pair set.                            |
| `smooth_lddt_pair_indices`   | Reuse a boolean pair mask as an ordered index list.               |
| `smooth_lddt_pair_distances` | Compute differentiable distances for a pair list.                 |

## Related API

Prepare validity with [masks](../mask/README.md), compute positions and distances with
[geometry](../geometry/README.md), and aggregate observations with [weighted reduction](../reduction/README.md).
See the [operator overview](../README.md).
