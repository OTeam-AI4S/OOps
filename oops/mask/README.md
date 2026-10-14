# Validity Masks

Turn MSA or token validity into boolean masks for attention and pair computations. Use these functions
when the consumer needs to know which observations exist, independently of their weight magnitudes.

## Usage

Reduce MSA validity over its sequence axis, then build pair validity on CPU:

```python
import torch
from oops import pair_mask_from_msa_mask

input = torch.tensor([[1, 0], [0, 0]])
output = pair_mask_from_msa_mask(input)
assert output.tolist() == [[True, False], [False, False]]
```

## Key Concepts

Every nonzero value counts as true. `token_mask_from_msa_mask` uses logical-any over MSA rows;
`pair_mask_from_token_mask` forms the outer conjunction of token validity. `pair_mask_from_msa_mask`
combines the two steps.

Boolean conversion discards weight magnitude. Retain the original floating weights when passing a
multiplicative mask to OPM or triangle multiplication.

An empty MSA axis yields false token validity, and an empty token axis stays empty.

## Public API

| API                         | When to Use                                          |
| --------------------------- | ---------------------------------------------------- |
| `token_mask_from_msa_mask`  | Find tokens observed in at least one alignment row.  |
| `pair_mask_from_token_mask` | Keep pairs whose two tokens are valid.               |
| `pair_mask_from_msa_mask`   | Build token-pair validity directly from an MSA mask. |
