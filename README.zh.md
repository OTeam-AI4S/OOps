# [OOps](https://oops.valtech.ai/zh/)

> [!TIP]
> More OOps, zero OOMs.

OOps 为生物分子模型提供 PyTorch 算子。它是 [OFoldX](https://github.com/OTeam-AI4S/OFoldX) 的算子后端，也可直接用于 PyTorch 程序。

## 快速开始

从 PyPI 安装：

```shell
pip install ooperators
```

对普通张量应用 `LayerNorm`：

```python
import torch
from oops import LayerNorm

layer = LayerNorm(16)
input = torch.randn(2, 4, 16, requires_grad=True)
output = layer(input)
output.square().mean().backward()
```

使用 DanLing `NestedTensor` 表示变长序列：

```python
import torch
from danling import NestedTensor
from oops import LayerNorm

input = NestedTensor([torch.randn(length, 16, requires_grad=True) for length in (3, 5)], ragged_dims=(0,))
output = LayerNorm(16)(input)
output.square().mean().backward()
```

更多用法请参阅[算子指南](oops)和[示例](examples)。

如果需要尚未发布的修改，可以从源码安装：

```shell
pip install git+https://github.com/OTeam-AI4S/OOps
```

## 社区

- [GitHub Issues](https://github.com/OTeam-AI4S/OOps/issues)：使用问题、错误报告与功能请求。

## 引用

> [!NOTE]
> 本仓库提供的内容是 OFoldX 项目的一部分。
> 如果 OOps 对你的研究有帮助，请按如下方式引用 OOps。

```bibtex
@software{chen2026oops,
  author = {Chen, Zhiyuan},
  title  = {OOps},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OOps},
}
```

## 许可

OOps 使用 [AGPL-3.0-or-later 许可证](license.md)，附加条款见[许可协议常见问题](license-faq.md)。
