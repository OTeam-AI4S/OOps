# Sparse Graph Operators

Group edges by row and aggregate their features using compressed sparse row (CSR) metadata. This is
useful when a molecular graph or neighborhood list is stored as edges rather than a dense pair grid.

## Usage

Sum three edge values into three rows on CUDA. The repeated offset identifies an empty middle row:

```python
import torch
from oops import segment_sum_csr

input = torch.tensor([1.0, 3.0, 2.0], device="cuda")
row_offsets = torch.tensor([0, 2, 2, 3], device="cuda")
output = segment_sum_csr(input, row_offsets)
assert output.shape == (3,)
```

The row sums are `[4.0, 0.0, 2.0]`.

## Key Concepts

Each row occupies the edge slice between two consecutive offsets. `CSRGraph` groups these row
boundaries and edge indices. `from_edges(sort=True)` records the stable sort permutation in `src_index`
so edge features can follow the graph's order.

Empty segments return zero for sums and weighted sums, and negative infinity for maxima. Segment softmax
keeps the edge shape, so an empty row has no probability entries.

`edge_dot_node` connects each edge feature to its destination node feature before reducing channels.

## Public API

| API                        | When to Use                                       |
| -------------------------- | ------------------------------------------------- |
| `CSRGraph`                 | Describe CSR row boundaries and edge indices.     |
| `regular_csr_graph`        | Construct a fixed-degree graph.                   |
| `validate_csr_metadata`    | Check CSR boundaries and row indices.             |
| `segment_sum_csr`          | Add edge features within each row.                |
| `segment_max_csr`          | Select each row's feature-wise maxima.            |
| `segment_softmax_csr`      | Normalize edges independently within each row.    |
| `segment_weighted_sum_csr` | Reduce weighted edge features by row.             |
| `edge_dot_node`            | Dot edge features with destination node features. |

## Related API

Use [geometry](../geometry/README.md) to select neighbors and [atom mappings](../atom/README.md) for
atom–token mappings. [Weighted reduction](../reduction/README.md) averages observations along a tensor axis.
See the [operator overview](../README.md) for shared conventions.
