# OOps

OOps is a high-performance GPU kernel library built to turbocharge biomolecular foundation models like OFold and ODesign. By pushing hardware efficiency to the limit, OOps drastically accelerates both training and inference while shattering the memory wall for ultra-long sequences — keeping your GPUs humming where standard frameworks simply run out of memory.

## Installation

```shell
pip install ooperators
```

## Usage

### Dense Tensors

`LayerNorm` accepts the same constructor arguments as `torch.nn.LayerNorm`.
This CUDA example normalizes a tensor and computes gradients:

```python
import torch
from oops import LayerNorm

norm = LayerNorm(256, device="cuda")
x = torch.randn(2, 128, 256, device="cuda", requires_grad=True)
y = norm(x)
y.square().sum().backward()
```

### Ragged Tensors

Use a `NestedTensor` for sequences of different lengths. `ragged_dims=(0,)`
marks the length axis within each sequence:

```python
import torch
from danling import NestedTensor
from oops import LayerNorm

norm = LayerNorm(256, device="cuda")
sequences = [torch.randn(length, 256, device="cuda", requires_grad=True) for length in (128, 192)]
x = NestedTensor(sequences, ragged_dims=(0,))
y = norm(x)
y.square().sum().backward()
```

## License

OOps is available under the [GNU Affero General Public License v3.0 or later](license.md). See the [License FAQ](license-faq.md) for commercial licensing.

## Privacy

See the [Privacy Notice](privacy.md) for information about data processing on the OOps documentation website.
