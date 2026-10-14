from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import is_compiling, is_external_fake, require_no_forward_mode, require_same_device, unsupported

from . import triton as _triton_sparse

_INDEX_DTYPES = (torch.int32, torch.int64)

_EDGE_DOT_NODE_SUPPORTED_LAYOUT = (
    "requires contiguous CUDA float16/bfloat16/float32/float64 edge_values and node_values "
    "sharing one dtype, a contiguous CUDA int32/int64 dst_index, and non-empty edge and channel "
    "axes with the fused output grid inside CUDA launch limits"
)


def edge_dot_node(edge_values: Tensor, node_values: Tensor, dst_index: Tensor) -> Tensor:
    r"""Dot each edge feature with its destination node feature.

    Args:
        edge_values: Contiguous CUDA float16, bfloat16, float32, or float64
            features shaped ``(E, ..., C)``, with non-empty edge/channel axes.
        node_values: Same-device, same-dtype contiguous features shaped
            ``(N, ..., C)``, with the same feature dimensions as edge values.
        dst_index: Contiguous CUDA int32/int64 destination ids shaped ``(E,)``.
            Entries must lie in ``[0, N)`` when nodes are present.

    Returns:
        Dot products shaped ``(E, ...)`` in the feature dtype. An empty node
        axis represents an all-zero operand and produces zeros.

    Raises:
        ValueError: Ranks, feature shapes, devices, index length, or index
            dtype disagree.
        OOpsUnsupportedError: The device, feature dtype, layout, launch
            extent, or forward-mode AD is unsupported, or a required strict
            deterministic node reduction cannot be served.

    Notes:
        Reverse mode differentiates both feature operands. Half arithmetic uses
        FP32 before casts to source dtypes. Ordinary node gradients may vary in
        final bits. Strict and warn-only deterministic modes give repeatable
        node gradients on one device when their launch domain is supported.
        Outside that domain, strict mode rejects backward; warn-only mode warns
        and proceeds without repeatability. Final bits may differ across GPU
        models.

        Full-graph compilation is supported; forward mode is unsupported.
        Destination index contents are not validated here. Out-of-range ids
        are outside the numerical contract.

    Examples:
        >>> import torch
        >>> from oops import edge_dot_node
        >>> if torch.cuda.is_available():
        ...     edges = torch.tensor([[1.0, 2.0]], device="cuda")
        ...     nodes = torch.tensor([[3.0, 4.0]], device="cuda")
        ...     destinations = torch.tensor([0], device="cuda")
        ...     assert edge_dot_node(edges, nodes, destinations).tolist() == [11.0]
    """

    require_no_forward_mode("edge_dot_node", edge_values, node_values)
    require_same_device("edge_dot_node", edge_values, node_values, dst_index)
    if edge_values.dim() < 2:
        raise ValueError("edge_values must have shape (E, ..., C)")
    if node_values.dim() != edge_values.dim():
        raise ValueError("node_values must have the same rank as edge_values")
    if tuple(node_values.shape[1:]) != tuple(edge_values.shape[1:]):
        raise ValueError("node_values.shape[1:] must match edge_values.shape[1:]")
    if dst_index.dim() != 1:
        raise ValueError("dst_index must have shape (E,)")
    if dst_index.shape[0] != edge_values.shape[0]:
        raise ValueError("dst_index length must match edge_values.shape[0]")
    if dst_index.dtype not in _INDEX_DTYPES:
        raise ValueError("dst_index must be an int32 or int64 tensor")

    plan = None
    compiling = is_compiling()
    edge_type = type(edge_values)
    external_fake = (
        not compiling
        and edge_type is not Tensor
        and edge_type is not torch.nn.Parameter
        and is_external_fake(edge_values)
    )
    if compiling or external_fake:
        supported = _triton_sparse.supports_edge_dot_node(edge_values, node_values, dst_index)
    else:
        plan = _triton_sparse.prepare_edge_dot_node(edge_values, node_values, dst_index)
        supported = plan is not None
    if not supported:
        raise unsupported(
            "edge_dot_node",
            _EDGE_DOT_NODE_SUPPORTED_LAYOUT,
            **_edge_dot_node_forward_context(edge_values, node_values, dst_index),
        )
    return _triton_sparse.triton_edge_dot_node(edge_values, node_values, dst_index, plan=plan)


def _edge_dot_node_forward_context(edge_values: Tensor, node_values: Tensor, dst_index: Tensor) -> dict[str, object]:
    return {
        "dtype": edge_values.dtype,
        "node_values_dtype": node_values.dtype,
        "device": edge_values.device,
        "edge_values_shape": tuple(edge_values.shape),
        "node_values_shape": tuple(node_values.shape),
        "contiguous": (edge_values.is_contiguous(), node_values.is_contiguous(), dst_index.is_contiguous()),
        "dst_index_device": dst_index.device,
        "dst_index_dtype": dst_index.dtype,
    }
