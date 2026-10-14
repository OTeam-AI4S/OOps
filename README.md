# [OOps](https://oops.valtech.ai/)

> [!TIP]
> More OOps, zero OOMs.

OOps provides PyTorch operators for biomolecular models, including attention, pair updates, atom–token transfer,
sparse graph aggregation, and structural losses. It is the operator backend for
[OFoldX](https://github.com/OTeam-AI4S/OFoldX) and can also be used directly in PyTorch programs.

## Get Started

Install from PyPI:

```shell
pip install ooperators
```

Use `LayerNorm` with a dense tensor:

```python
import torch
from oops import LayerNorm

layer = LayerNorm(16)
input = torch.randn(2, 4, 16, requires_grad=True)
output = layer(input)
output.square().mean().backward()
```

Use DanLing `NestedTensor` for variable-length sequences:

```python
import torch
from danling import NestedTensor
from oops import LayerNorm

input = NestedTensor([torch.randn(length, 16, requires_grad=True) for length in (3, 5)], ragged_dims=(0,))
output = LayerNorm(16)(input)
output.square().mean().backward()
```

See the [operator guide](oops) and [examples](examples) for more usage.

Install the latest source version when you need unreleased changes:

```shell
pip install git+https://github.com/OTeam-AI4S/OOps
```

## Community

- [GitHub Issues](https://github.com/OTeam-AI4S/OOps/issues): usage questions, bug reports, and feature requests.

## Citation

> [!NOTE]
> The artifacts distributed in this repository are part of the OFoldX project.
> If OOps supports your research, please cite OOps as follows:

```bibtex
@software{chen2026oops,
  author = {Chen, Zhiyuan},
  title  = {OOps},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OOps},
}
```

## License

OOps is distributed under the [AGPL-3.0-or-later license](license.md); the [License FAQ](license-faq.md)
covers additional terms.
