from __future__ import annotations

from torch import Tensor

from oops.runtime import require_no_forward_mode, require_same_device, unsupported

from ..csr import SUPPORTED_LAYOUT as _SUPPORTED_LAYOUT
from ..csr import canonical_csr_metadata as _canonical_csr_metadata
from ..csr import layout_context as _layout_context
from ..csr import num_edges as _num_edges
from ..csr.graph import CSRGraph
from . import triton as _triton_sparse


def csr_weighted_sum(
    values: Tensor,
    weights: Tensor,
    row_offsets: Tensor | CSRGraph,
    *,
    num_segments: int | None = None,
    segment_ids: Tensor | None = None,
) -> Tensor:
    r"""Sum weighted edge values within each CSR segment.

    Args:
        values: Contiguous CUDA float16, bfloat16, float32, or float64 values
            shaped ``(E, ...)``, with a non-empty edge axis.
        weights: Same-device, same-dtype contiguous weights. Supported shapes
            are scalar ``(E,)``, head-prefix ``(E, H)`` or ``(E, 1)`` for
            ``(E, H, ...)`` values, and the full value shape.
        row_offsets: Contiguous CUDA int32/int64 boundaries shaped ``(N + 1,)``,
            or a ``CSRGraph`` carrying them.
        num_segments: Optional output segment count, at least ``N``. Extra
            segments are empty.
        segment_ids: Optional int32/int64 cache shaped ``(E,)`` on the offset
            device, derived from ``row_offsets``. Omit it for a ``CSRGraph``.

    Returns:
        Weighted sums shaped ``(num_segments or N, ...)`` in the value dtype.
        Empty segments are zero.

    Raises:
        TypeError: ``segment_ids`` is supplied separately with a graph.
        ValueError: Shapes, devices, metadata dtypes, segment counts, weight
            alignment, or checked host contents violate the contract.
        OOpsUnsupportedError: The device, dtype, weight broadcast layout,
            contiguity, edge storage, launch extent, or forward mode is unsupported.

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

        Gradients flow to values and weights. Weight gradients retain their
        input shape.

    Examples:
        >>> import torch
        >>> from oops import csr_weighted_sum
        >>> if torch.cuda.is_available():
        ...     values = torch.tensor([1.0, 3.0, 2.0], device="cuda")
        ...     weights = torch.tensor([2.0, 1.0, 4.0], device="cuda")
        ...     offsets = torch.tensor([0, 2, 3], device="cuda")
        ...     assert csr_weighted_sum(values, weights, offsets).tolist() == [5.0, 8.0]
    """

    require_no_forward_mode("csr_weighted_sum", values, weights)
    num_edges = _num_edges(values, "values")
    if weights.dim() < 1:
        raise ValueError("weights must have shape (E, ...)")
    if weights.shape[0] != values.shape[0]:
        raise ValueError("weights.shape[0] must match values.shape[0]")
    require_same_device("csr_weighted_sum", values, weights)
    _validate_weight_alignment(weights, values)
    offsets, inferred_segments, out_segments, segment_ids = _canonical_csr_metadata(
        row_offsets,
        device=values.device,
        num_edges=num_edges,
        num_segments=num_segments,
        segment_ids=segment_ids,
    )
    if not _triton_sparse.supports_csr_weighted_sum(values, weights, offsets, out_segments):
        raise unsupported(
            "csr_weighted_sum",
            _SUPPORTED_LAYOUT + ", with weights that are scalar (E,), head-prefix (E, H), or shaped like values",
            weights_dtype=weights.dtype,
            weights_shape=tuple(weights.shape),
            **_layout_context(values, offsets),
        )
    return _triton_sparse.triton_csr_weighted_sum(
        values,
        weights,
        offsets,
        out_segments,
        inferred_segments,
        segment_ids,
    )


def _validate_weight_alignment(weights: Tensor, values: Tensor) -> None:
    if weights.dim() > values.dim():
        raise ValueError("weights rank may not exceed values rank")
    weight_tail = tuple(weights.shape[1:])
    value_tail = tuple(values.shape[1:])
    if len(weight_tail) > len(value_tail):
        raise ValueError("weights feature rank may not exceed values feature rank")
    if not all(weight_dim in (1, value_dim) for weight_dim, value_dim in zip(weight_tail, value_tail)):
        raise ValueError("weights feature dimensions must broadcast over a prefix of values feature dimensions")
