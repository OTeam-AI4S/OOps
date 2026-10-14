# OOps

OOps 是一个高性能 GPU 算子库，旨在加速 OFold、ODesign 等生物分子基础模型。
OOps 充分发挥硬件效率，显著加速训练和推理，突破超长序列的显存瓶颈，让 GPU 在常规框架显存不足的场景下继续运行。

## 安装

```shell
pip install ooperators
```

## 使用

### 普通 Tensor

`LayerNorm` 接受与 `torch.nn.LayerNorm` 相同的构造参数。
下面的 CUDA 示例对 Tensor 做归一化并计算梯度：

```python
import torch
from oops import LayerNorm

norm = LayerNorm(256, device="cuda")
x = torch.randn(2, 128, 256, device="cuda", requires_grad=True)
y = norm(x)
y.square().sum().backward()
```

### 不等长序列

不同长度的序列可以组成 `NestedTensor`。`ragged_dims=(0,)`
表示每个序列的长度轴：

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

## 许可证

OOps 使用 [GNU Affero General Public License v3.0 或更新版本](license.zh.md)。商业许可请参阅[许可协议常见问题](license-faq.zh.md)。

## 隐私

OOps 文档网站的数据处理信息请参阅[隐私声明](privacy.zh.md)。
