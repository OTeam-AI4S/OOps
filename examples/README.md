# Examples

Run small examples of the public APIs after installing OOps. Start with `LayerNorm` to see dense and
variable-length inputs in the same program.

## Usage

From the repository root:

```bash
python examples/layer_norm.py --device cpu
```

The script applies one `LayerNorm` module to a dense tensor and sequences of different lengths, then runs
backward for both.

## Choosing a Device

Select the supported CUDA path with:

```bash
python examples/layer_norm.py --device cuda
```

See the [operator overview](../oops/README.md) and [normalization](../oops/normalization/README.md).
