# Atom–Token Transfer

Use an atom-to-token assignment to pool atom features into token features, broadcast token features back
to atoms, or gather token-pair features for local atom attention.

## Usage

Here two atoms contribute to the first token and one contributes to the second. Dense assignment weights
let this small example run on CPU:

```python
import torch
from oops import atom_to_token_mean

input = torch.tensor([[2.0], [4.0], [9.0]])
assignment = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
output = atom_to_token_mean(input, assignment)
assert output.tolist() == [[3.0], [9.0]]
```

## Key Concepts

Transfers accept dense assignment weights or local integer mappings. `AtomTokenLayout` describes which
token each atom belongs to. Bare reduction indices require `num_tokens`; nested reductions use
`token_reference` to determine the output token lengths.

Sums and means return zero for empty tokens. For a mean, `atom_mask` weights both the feature sum and its
count; `denominator_offset` is added to positive counts.

Window gathering expands token-pair features into atom query/key windows. Padded slots select token zero,
so their features come from that token rather than being zero-filled.

## Public API

| API                                | When to Use                                       |
| ---------------------------------- | ------------------------------------------------- |
| `AtomTokenLayout`                  | Describe the mapping from atoms to tokens.        |
| `validate_atom_to_token`           | Check mapping values against declared extents.    |
| `token_to_atom_broadcast`          | Transfer token features to their assigned atoms.  |
| `atom_to_token_sum`                | Sum weighted atom features into tokens.           |
| `atom_to_token_mean`               | Average atom features within each token.          |
| `token_pair_to_atom_window_gather` | Prepare local atom-pair windows from token pairs. |
