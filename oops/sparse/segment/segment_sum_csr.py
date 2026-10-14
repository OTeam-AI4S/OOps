from __future__ import annotations

from torch import Tensor

from oops.runtime import require_no_forward_mode, unsupported

from ..csr import SUPPORTED_LAYOUT as _SUPPORTED_LAYOUT
from ..csr import canonical_csr_metadata as _canonical_csr_metadata
from ..csr import layout_context as _layout_context
from ..csr import num_edges as _num_edges
from ..csr.graph import CSRGraph
from . import triton as _triton_sparse


def segment_sum_csr(
    values: Tensor,
    row_offsets: Tensor | CSRGraph,
    *,
    num_segments: int | None = None,
    segment_ids: Tensor | None = None,
) -> Tensor:
    r"""Sum edge values within each CSR segment.

    Args:
        values: Contiguous CUDA float16, bfloat16, float32, or float64 values
            shaped ``(E, ...)``, with a non-empty edge axis.
        row_offsets: Contiguous CUDA int32/int64 boundaries shaped ``(N + 1,)``,
            or a ``CSRGraph`` carrying them.
        num_segments: Optional output segment count, at least ``N``. Extra
            segments are empty.
        segment_ids: Optional int32/int64 cache shaped ``(E,)`` on the offset
            device, derived from ``row_offsets``. Omit it for a ``CSRGraph``.

    Returns:
        A tensor shaped ``(num_segments or N, ...)`` in the value dtype.
        Empty segments contain zero.

    Raises:
        TypeError: ``segment_ids`` is supplied separately with a graph.
        ValueError: Shapes, metadata dtypes, devices, segment counts, or
            checked host contents violate the contract.
        OOpsUnsupportedError: Value dtype, contiguity, non-empty edge storage,
            device, launch extent, or forward-mode AD is unsupported.

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
        >>> from oops import segment_sum_csr
        >>> if torch.cuda.is_available():
        ...     values = torch.tensor([1.0, 3.0, 2.0], device="cuda")
        ...     offsets = torch.tensor([0, 2, 3], device="cuda")
        ...     result = segment_sum_csr(values, offsets)
        ...     assert result.tolist() == [4.0, 2.0]
    """

    require_no_forward_mode("segment_sum_csr", values)
    num_edges = _num_edges(values, "values")
    offsets, inferred_segments, out_segments, segment_ids = _canonical_csr_metadata(
        row_offsets,
        device=values.device,
        num_edges=num_edges,
        num_segments=num_segments,
        segment_ids=segment_ids,
    )
    if not _triton_sparse.supports_segment_sum(values, offsets, out_segments):
        raise unsupported("segment_sum_csr", _SUPPORTED_LAYOUT, **_layout_context(values, offsets))
    return _triton_sparse.triton_segment_sum(
        values,
        offsets,
        out_segments,
        inferred_segments,
        segment_ids,
    )
