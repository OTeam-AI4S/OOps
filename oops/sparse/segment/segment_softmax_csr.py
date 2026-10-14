from __future__ import annotations

from torch import Tensor

from oops.runtime import require_no_forward_mode, unsupported

from ..csr import SUPPORTED_LAYOUT as _SUPPORTED_LAYOUT
from ..csr import canonical_csr_metadata as _canonical_csr_metadata
from ..csr import layout_context as _layout_context
from ..csr import num_edges as _num_edges
from ..csr.graph import CSRGraph
from . import triton as _triton_sparse


def segment_softmax_csr(
    logits: Tensor,
    row_offsets: Tensor | CSRGraph,
    *,
    segment_ids: Tensor | None = None,
) -> Tensor:
    r"""Normalize logits with a softmax over each CSR segment's edge axis.

    Args:
        logits: Contiguous CUDA float16, bfloat16, float32, or float64 logits
            shaped ``(E, ...)``, with a non-empty edge axis.
        row_offsets: Contiguous CUDA int32/int64 boundaries shaped ``(N + 1,)``,
            or a ``CSRGraph`` carrying them.
        segment_ids: Optional int32/int64 cache shaped ``(E,)`` on the offset
            device, derived from ``row_offsets``. Omit it for a ``CSRGraph``.

    Returns:
        Probabilities with the shape and dtype of ``logits``. Each feature
        normalizes independently over its segment. Empty segments have no
        output entries; uncovered prefix/tail entries are written as zero.

    Raises:
        TypeError: ``segment_ids`` is supplied separately with a graph.
        ValueError: Logits are non-floating, or shapes, metadata dtypes,
            devices, or checked host contents violate the contract.
        OOpsUnsupportedError: The device, floating dtype, layout, non-empty
            edge storage, launch extent, or forward-mode AD is unsupported.

    Notes:
        Offsets must start at zero, be non-decreasing, and end at ``E``.
        Cached ids must agree exactly with those offsets. CPU contents are
        checked eagerly; CUDA contents are not inspected by operator calls.
        Validate untrusted metadata with ``validate_csr_metadata``. Uncovered
        prefix/tail edges have zero gradients, but incomplete offsets do not
        satisfy the CSR validation contract.

        Half arithmetic uses FP32 before casts to source dtypes. Results repeat
        on one device; final bits may differ across GPU models. Reverse mode,
        higher-order reverse mode, and full-graph compilation are supported;
        forward mode is unsupported.

    Examples:
        >>> import torch
        >>> from oops import segment_softmax_csr
        >>> if torch.cuda.is_available():
        ...     logits = torch.zeros(3, device="cuda")
        ...     offsets = torch.tensor([0, 2, 3], device="cuda")
        ...     result = segment_softmax_csr(logits, offsets)
        ...     torch.testing.assert_close(result, logits.new_tensor([0.5, 0.5, 1.0]))
    """

    require_no_forward_mode("segment_softmax_csr", logits)
    if not logits.is_floating_point():
        raise ValueError("segment_softmax_csr requires floating-point logits")
    num_edges = _num_edges(logits, "logits")
    offsets, inferred_segments, _out_segments, segment_ids = _canonical_csr_metadata(
        row_offsets,
        device=logits.device,
        num_edges=num_edges,
        num_segments=None,
        segment_ids=segment_ids,
    )
    if not _triton_sparse.supports_segment_softmax(logits, offsets, inferred_segments):
        raise unsupported("segment_softmax_csr", _SUPPORTED_LAYOUT, **_layout_context(logits, offsets))
    return _triton_sparse.triton_segment_softmax(logits, offsets, inferred_segments, segment_ids)
