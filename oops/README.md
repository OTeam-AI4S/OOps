# Operators

Choose an operator for the representation you are working with: tokens, pair features, alignments,
atoms, or sparse edges. Public functions can be imported from `oops` or their family namespace.

## Operator Map

| Family                                   | When to Use                                                   |
| ---------------------------------------- | ------------------------------------------------------------- |
| [Attention](attention/README.md)         | Attend to token features or pair-feature rows.                |
| [Pair](pair/README.md)                   | Build incoming or outgoing triangle updates.                  |
| [Normalization](normalization/README.md) | Normalize feature channels through a module or function.      |
| [Atom](atom/README.md)                   | Transfer between atom/token features and gather atom windows. |
| [Sparse](sparse/README.md)               | Group edges and aggregate features within CSR rows.           |
| [Losses](losses/README.md)               | Compute classification or structural objectives.              |

## OFoldX Integration

Use the [OFoldX backend guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/modules/backend/README.md)
to select `auto`, `eager`, or `oops` in an OFoldX workflow.

## Related API

See the [examples](../examples/README.md) for runnable scripts.
