# OOps
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OOps.

# OOps is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OOps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://oops.valtech.ai/about/license-faq/>.

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from oops.runtime import INT32_MAX

_INDEX_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True, eq=False)
class CSRGraph:
    r"""Store edges grouped by source row in compressed sparse row form.

    Args:
        row_offsets: Int32/int64 boundaries shaped ``(num_rows + 1,)``.
        col_indices: Optional same-device int32/int64 edge columns shaped
            ``(num_edges,)``.
        segment_ids: Optional same-device int32/int64 row ids shaped
            ``(num_edges,)``, derived exactly from the offsets.
        src_index: Optional same-device int32/int64 original edge positions,
            shaped ``(num_edges,)``. ``from_edges(sort=True)`` records the
            stable sort permutation here.
        num_rows: Row count; defaults to ``len(row_offsets) - 1`` and must
            agree with it when supplied.
        num_edges: Edge count, inferred from edge indices or the terminal offset
            when omitted.
        check_invariants: Whether to inspect CUDA CSR contents. CPU contents
            are always validated.

    Attributes:
        row_offsets: Edge boundaries for each row.
        col_indices: Optional edge columns.
        segment_ids: Optional edge-to-row ids.
        src_index: Optional original edge positions.
        num_rows: Resolved integer row count.
        num_edges: Resolved integer edge count.
        check_invariants: Construction-time CUDA validation choice.

    Raises:
        TypeError: A supplied index field is not a torch tensor.
        ValueError: Index metadata or declared counts disagree, checked
            contents violate CSR invariants, or checked CUDA construction
            lacks an independent edge count.

    Notes:
        Graph connectivity must remain unchanged throughout a differentiable computation.

    Examples:
        >>> import torch
        >>> from oops import CSRGraph
        >>> graph = CSRGraph(torch.tensor([0, 2, 3]))
        >>> (graph.num_rows, graph.num_edges)
        (2, 3)
    """

    row_offsets: Tensor
    col_indices: Tensor | None = None
    segment_ids: Tensor | None = None
    src_index: Tensor | None = None
    num_rows: int | None = None
    num_edges: int | None = None
    check_invariants: bool = field(default=False, kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_index_tensor(self.row_offsets, "row_offsets")
        if self.row_offsets.dim() != 1:
            raise ValueError("row_offsets must have shape (num_rows + 1,)")
        if self.row_offsets.numel() == 0:
            raise ValueError("row_offsets must contain at least one offset")

        inferred_rows = int(self.row_offsets.numel()) - 1
        if self.num_rows is None:
            object.__setattr__(self, "num_rows", inferred_rows)
        elif int(self.num_rows) != inferred_rows:
            raise ValueError("num_rows must match row_offsets.shape[0] - 1")
        else:
            object.__setattr__(self, "num_rows", int(self.num_rows))

        edge_aligned = (self.col_indices, self.segment_ids, self.src_index)
        if (
            self.check_invariants
            and self.row_offsets.is_cuda
            and self.num_edges is None
            and all(tensor is None for tensor in edge_aligned)
        ):
            raise ValueError("check_invariants=True with CUDA row_offsets requires num_edges or an edge-aligned tensor")

        inferred_edges = _infer_num_edges(
            self.row_offsets,
            self.col_indices,
            self.segment_ids,
            self.src_index,
            self.num_edges,
        )
        object.__setattr__(self, "num_edges", inferred_edges)

        for name, tensor in (
            ("col_indices", self.col_indices),
            ("segment_ids", self.segment_ids),
            ("src_index", self.src_index),
        ):
            _check_optional_edge_tensor(tensor, name, self.row_offsets.device, inferred_edges)

        if self.check_invariants or self.row_offsets.device.type == "cpu":
            validate_csr_metadata(self.row_offsets, num_edges=inferred_edges, segment_ids=self.segment_ids)

    @classmethod
    def from_row_offsets(
        cls,
        row_offsets: Tensor,
        col_indices: Tensor | None = None,
        segment_ids: Tensor | None = None,
        *,
        num_edges: int | None = None,
        check_invariants: bool = False,
    ) -> CSRGraph:
        r"""Construct a graph from existing CSR row boundaries.

        Args:
            row_offsets: Int32/int64 boundaries shaped ``(N + 1,)``.
            col_indices: Optional same-device int32/int64 columns shaped ``(E,)``.
            segment_ids: Optional same-device int32/int64 row ids shaped
                ``(E,)``, corresponding exactly to the offsets.
            num_edges: Optional edge count; inferred under ``CSRGraph``'s
                construction rules when omitted.
            check_invariants: Inspect CUDA contents when true. CPU contents are
                always checked.

        Returns:
            A ``CSRGraph`` retaining the supplied tensors, with resolved integer
            row and edge counts.

        Raises:
            TypeError: An index field is not a torch tensor.
            ValueError: Metadata or checked contents violate ``CSRGraph``'s
                construction contract.

        Examples:
            >>> import torch
            >>> from oops import CSRGraph
            >>> CSRGraph.from_row_offsets(torch.tensor([0, 2, 3])).num_edges
            3
        """

        return cls(
            row_offsets=row_offsets,
            col_indices=col_indices,
            segment_ids=segment_ids,
            num_edges=num_edges,
            check_invariants=check_invariants,
        )

    @classmethod
    def from_edges(
        cls,
        row_index: Tensor,
        col_index: Tensor | None = None,
        num_rows: int | None = None,
        sort: bool = True,
    ) -> CSRGraph:
        r"""Build CSR boundaries and row ids from edge-aligned source rows.

        Args:
            row_index: One-dimensional int32/int64 source rows shaped ``(E,)``.
            col_index: Optional same-device int32/int64 columns shaped ``(E,)``.
            num_rows: Non-negative row count; inferred from the maximum row id
                when omitted, or zero for no edges.
            sort: Stably group edges by row. When false, input rows must already
                be non-decreasing.

        Returns:
            A ``CSRGraph`` with row boundaries, grouped ``segment_ids``, optional
            reordered columns, and ``src_index`` containing the stable permutation
            when sorting. Offsets become int64 if the edge count exceeds the
            int32 range; otherwise they use the row-index dtype.

        Raises:
            TypeError: An index operand is not a torch tensor.
            ValueError: Index shapes/dtypes/devices or declared row counts are
                invalid, or checked CPU rows violate range/order requirements.

        Notes:
            Reorder edge values with ``src_index`` when sorting their metadata.

        Examples:
            >>> import torch
            >>> from oops import CSRGraph
            >>> graph = CSRGraph.from_edges(torch.tensor([1, 0, 1]), num_rows=2)
            >>> graph.row_offsets.tolist(), graph.src_index.tolist()
            ([0, 1, 3], [1, 0, 2])
        """

        _check_index_tensor(row_index, "row_index")
        if row_index.dim() != 1:
            raise ValueError("row_index must have shape (num_edges,)")
        if col_index is not None:
            _check_index_tensor(col_index, "col_index")
            if col_index.dim() != 1:
                raise ValueError("col_index must have shape (num_edges,)")
            if col_index.device != row_index.device:
                raise ValueError("col_index must be on the same device as row_index")
            if col_index.shape[0] != row_index.shape[0]:
                raise ValueError("col_index length must match row_index length")

        edge_count = int(row_index.shape[0])
        rows = _infer_rows_from_edges(row_index, num_rows)
        if row_index.device.type == "cpu":
            _validate_cpu_rows(row_index, rows)
            if not sort and (row_index[1:] < row_index[:-1]).any().item():
                raise ValueError("row_index must be sorted when sort=False")

        if sort:
            order = torch.argsort(row_index.to(dtype=torch.long), stable=True)
            row_index = row_index.index_select(0, order)
            col_index = col_index.index_select(0, order) if col_index is not None else None
            src_index: Tensor | None = order
        else:
            src_index = None

        counts = _row_counts(row_index.to(dtype=torch.long), rows)
        row_offsets = torch.zeros(rows + 1, device=row_index.device, dtype=torch.long)
        if rows:
            row_offsets[1:] = torch.cumsum(counts, dim=0)
        offset_dtype = torch.int64 if row_index.dtype == torch.int32 and edge_count > INT32_MAX else row_index.dtype
        row_offsets = row_offsets.to(dtype=offset_dtype)
        segment_ids = row_index

        return cls(
            row_offsets=row_offsets,
            col_indices=col_index,
            segment_ids=segment_ids,
            src_index=src_index,
            num_rows=rows,
            num_edges=edge_count,
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> CSRGraph:
        r"""Move graph metadata to a device and preserve declared counts.

        Args:
            device: Destination for every metadata tensor.
            non_blocking: Request asynchronous copies where supported. CPU
                destinations always use blocking copies before content validation.

        Returns:
            A ``CSRGraph`` on the requested device, with the same row/edge counts.
            Copies follow ordinary ``Tensor.to`` behavior and may share tensors
            when no device change is needed.

        Raises:
            ValueError: Destination construction detects inconsistent metadata or
                invalid CPU contents.

        Notes:
            Destination construction applies the normal policy: CPU contents are
            validated; CUDA contents are not inspected. ``check_invariants`` is
            not propagated as a request to validate the destination.

        Examples:
            >>> import torch
            >>> from oops import CSRGraph
            >>> graph = CSRGraph(torch.tensor([0, 1]))
            >>> graph.to("cpu").row_offsets.tolist()
            [0, 1]
        """

        target = torch.device(device)
        copy_non_blocking = non_blocking and target.type != "cpu"
        return CSRGraph(
            row_offsets=self.row_offsets.to(device=target, non_blocking=copy_non_blocking),
            col_indices=_to_device(self.col_indices, target, copy_non_blocking),
            segment_ids=_to_device(self.segment_ids, target, copy_non_blocking),
            src_index=_to_device(self.src_index, target, copy_non_blocking),
            num_rows=self.num_rows,
            num_edges=self.num_edges,
        )


def regular_csr_graph(
    num_rows: int,
    fanout: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> CSRGraph:
    r"""Build CSR metadata with the same edge count in every row.

    Args:
        num_rows: Non-negative row count.
        fanout: Non-negative number of edges per row.
        device: Metadata device; None follows PyTorch's default device.
        dtype: Offset and row-id dtype, either torch.int32 or torch.int64.

    Returns:
        A ``CSRGraph`` with offsets ``arange(num_rows + 1) * fanout`` and
        edge-aligned segment ids. Its edge count is ``num_rows * fanout``.

    Raises:
        ValueError: Counts or index dtype are invalid, or requested int32
            offsets cannot represent the terminal edge count.

    Notes:
        Empty rows and graphs are valid metadata, even when an individual
        operator requires non-empty edge storage. CUDA construction does not
        synchronize metadata contents with the host.

    Examples:
        >>> from oops import regular_csr_graph
        >>> graph = regular_csr_graph(2, 3)
        >>> graph.row_offsets.tolist()
        [0, 3, 6]
    """

    if dtype not in _INDEX_DTYPES:
        raise ValueError("dtype must be torch.int32 or torch.int64")
    rows = int(num_rows)
    degree = int(fanout)
    if rows < 0:
        raise ValueError("num_rows must be non-negative")
    if degree < 0:
        raise ValueError("fanout must be non-negative")
    if dtype == torch.int32 and rows * degree > INT32_MAX:
        raise ValueError("int32 row_offsets cannot represent num_rows * fanout; use torch.int64")

    row_offsets = torch.arange(rows + 1, device=device, dtype=dtype) * degree
    segment_ids = torch.repeat_interleave(torch.arange(rows, device=device, dtype=dtype), degree)
    return CSRGraph(
        row_offsets=row_offsets,
        segment_ids=segment_ids,
        num_rows=rows,
        num_edges=rows * degree,
    )


def validate_csr_metadata(
    row_offsets: Tensor,
    *,
    num_edges: int | None = None,
    segment_ids: Tensor | None = None,
    dst_index: Tensor | None = None,
    num_nodes: int | None = None,
) -> None:
    r"""Validate CSR coverage, row ids, and optional destination bounds.

    Args:
        row_offsets: Int32/int64 row boundaries shaped ``(N + 1,)``.
        num_edges: Expected edge count ``E``. Defaults first to the length of
            supplied ``segment_ids``, then to supplied ``dst_index``, otherwise
            to the terminal offset.
        segment_ids: Optional same-device int32/int64 row ids shaped ``(E,)``,
            checked for exact agreement with the offsets.
        dst_index: Optional same-device int32/int64 destination ids shaped
            ``(E,)``, checked against ``num_nodes``.
        num_nodes: Non-negative destination node count, required with
            ``dst_index``.

    Returns:
        None when all requested invariants hold: offsets begin at zero, are
        non-decreasing, lie in ``[0, E]``, and end at ``E``; row ids
        place each edge in exactly its offset interval; destination ids lie
        in ``[0, num_nodes)``.

    Raises:
        TypeError: An index operand is not a torch tensor.
        ValueError: Tensor metadata or a requested invariant is invalid.
            Content errors identify the violated clause and offending values.

    Examples:
        >>> import torch
        >>> from oops import validate_csr_metadata
        >>> offsets = torch.tensor([0, 2, 3])
        >>> row_ids = torch.tensor([0, 0, 1])
        >>> validate_csr_metadata(offsets, num_edges=3, segment_ids=row_ids)
    """

    _check_index_tensor(row_offsets, "row_offsets")
    if row_offsets.dim() != 1:
        raise ValueError("row_offsets must have shape (num_rows + 1,)")
    if row_offsets.numel() == 0:
        raise ValueError("row_offsets must contain at least one offset")
    if segment_ids is not None:
        _check_index_tensor(segment_ids, "segment_ids")
        if segment_ids.dim() != 1:
            raise ValueError("segment_ids must have shape (num_edges,)")
        if segment_ids.device != row_offsets.device:
            raise ValueError("segment_ids must be on the same device as row_offsets")
    if dst_index is not None:
        _check_index_tensor(dst_index, "dst_index")
        if dst_index.dim() != 1:
            raise ValueError("dst_index must have shape (num_edges,)")
        if dst_index.device != row_offsets.device:
            raise ValueError("dst_index must be on the same device as row_offsets")
        if num_nodes is None:
            raise ValueError("num_nodes is required when dst_index is provided")

    num_rows = int(row_offsets.numel()) - 1
    nodes = None if num_nodes is None else int(num_nodes)
    if nodes is not None and nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    edges: int | None = None
    if num_edges is not None:
        edges = int(num_edges)
    elif segment_ids is not None:
        edges = int(segment_ids.shape[0])
    elif dst_index is not None:
        edges = int(dst_index.shape[0])
    if edges is not None and edges < 0:
        raise ValueError("num_edges must be non-negative")
    if edges is not None and segment_ids is not None and int(segment_ids.shape[0]) != edges:
        raise ValueError(f"segment_ids length must match num_edges ({edges}), got {int(segment_ids.shape[0])}")
    if edges is not None and dst_index is not None and int(dst_index.shape[0]) != edges:
        raise ValueError(f"dst_index length must match num_edges ({edges}), got {int(dst_index.shape[0])}")

    offsets = row_offsets.to(dtype=torch.long)
    stats = _csr_clause_stats(offsets, num_rows, segment_ids, dst_index)
    (
        first,
        last,
        low_index,
        low_value,
        high_index,
        high_value,
        has_decrease,
        decrease_index,
        decrease_left,
        decrease_right,
        has_misplaced_id,
        misplaced_index,
        misplaced_value,
        misplaced_start,
        misplaced_stop,
        low_dst_index,
        low_dst_value,
        high_dst_index,
        high_dst_value,
    ) = (int(value) for value in stats.tolist())
    if edges is None:
        edges = last

    if first != 0:
        raise ValueError(f"row_offsets must start at zero, got row_offsets[0] = {first}")
    # A negative offset also decreases; report its range violation first.
    if low_value < 0:
        raise ValueError(f"every row_offset must lie in [0, {edges}], got row_offsets[{low_index}] = {low_value}")
    if has_decrease:
        raise ValueError(
            "row_offsets must be monotonically non-decreasing, got "
            f"row_offsets[{decrease_index}] = {decrease_left} > "
            f"row_offsets[{decrease_index + 1}] = {decrease_right}"
        )
    if high_value > edges:
        raise ValueError(f"every row_offset must lie in [0, {edges}], got row_offsets[{high_index}] = {high_value}")
    if last != edges:
        raise ValueError(f"row_offsets[-1] must equal num_edges ({edges}), got {last}")
    if has_misplaced_id:
        if misplaced_value < 0 or misplaced_value >= num_rows:
            raise ValueError(
                f"segment_ids must match row_offsets, got segment_ids[{misplaced_index}] = "
                f"{misplaced_value}, which is outside [0, {num_rows})"
            )
        raise ValueError(
            f"segment_ids must match row_offsets, got segment_ids[{misplaced_index}] = "
            f"{misplaced_value}, whose edges are row_offsets[{misplaced_value}] = "
            f"{misplaced_start} through {misplaced_stop}"
        )
    if dst_index is not None and dst_index.numel() > 0:
        assert nodes is not None
        if low_dst_value < 0:
            raise ValueError(
                f"every dst_index must lie in [0, {nodes}), got " f"dst_index[{low_dst_index}] = {low_dst_value}"
            )
        if high_dst_value >= nodes:
            raise ValueError(
                f"every dst_index must lie in [0, {nodes}), got " f"dst_index[{high_dst_index}] = {high_dst_value}"
            )


def _csr_clause_stats(
    offsets: Tensor,
    num_rows: int,
    segment_ids: Tensor | None,
    dst_index: Tensor | None,
) -> Tensor:

    device = offsets.device
    zero = torch.zeros(1, device=device, dtype=torch.long)
    low_index = offsets.argmin().reshape(1)
    high_index = offsets.argmax().reshape(1)

    if num_rows > 0:
        decreasing = offsets[1:] < offsets[:-1]
        has_decrease = decreasing.any().reshape(1)
        decrease_index = decreasing.long().argmax().reshape(1)
        decrease_left = offsets.index_select(0, decrease_index)
        decrease_right = offsets.index_select(0, decrease_index + 1)
    else:
        has_decrease, decrease_index, decrease_left, decrease_right = zero, zero, zero, zero

    if segment_ids is None or segment_ids.numel() == 0 or num_rows == 0:
        has_misplaced_id, misplaced_index = zero, zero
        misplaced_value, misplaced_start, misplaced_stop = zero, zero, zero
    else:
        ids = segment_ids.to(dtype=torch.long)
        row = ids.clamp(0, num_rows - 1)
        start = offsets.index_select(0, row)
        stop = offsets.index_select(0, row + 1)
        edge = torch.arange(ids.numel(), device=device, dtype=torch.long)
        misplaced = (ids < 0) | (ids >= num_rows) | (edge < start) | (edge >= stop)
        has_misplaced_id = misplaced.any().reshape(1)
        misplaced_index = misplaced.long().argmax().reshape(1)
        misplaced_value = ids.index_select(0, misplaced_index)
        misplaced_start = start.index_select(0, misplaced_index)
        misplaced_stop = stop.index_select(0, misplaced_index)

    if dst_index is None or dst_index.numel() == 0:
        low_dst_index, low_dst_value = zero, zero
        high_dst_index, high_dst_value = zero, zero
    else:
        dst = dst_index.to(dtype=torch.long)
        low_dst_index = dst.argmin().reshape(1)
        high_dst_index = dst.argmax().reshape(1)
        low_dst_value = dst.index_select(0, low_dst_index)
        high_dst_value = dst.index_select(0, high_dst_index)

    return torch.cat(
        [
            offsets[:1],
            offsets[-1:],
            low_index,
            offsets.index_select(0, low_index),
            high_index,
            offsets.index_select(0, high_index),
            has_decrease.long(),
            decrease_index,
            decrease_left,
            decrease_right,
            has_misplaced_id.long(),
            misplaced_index,
            misplaced_value,
            misplaced_start,
            misplaced_stop,
            low_dst_index,
            low_dst_value,
            high_dst_index,
            high_dst_value,
        ]
    )


def _check_index_tensor(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in _INDEX_DTYPES:
        raise ValueError(f"{name} must be an int32 or int64 tensor")


def _check_optional_edge_tensor(tensor: Tensor | None, name: str, device: torch.device, num_edges: int) -> None:
    if tensor is None:
        return
    _check_index_tensor(tensor, name)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must have shape (num_edges,)")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as row_offsets")
    if tensor.shape[0] != num_edges:
        raise ValueError(f"{name} length must match num_edges")


def _infer_num_edges(
    row_offsets: Tensor,
    col_indices: Tensor | None,
    segment_ids: Tensor | None,
    src_index: Tensor | None,
    num_edges: int | None,
) -> int:
    edge_tensor = col_indices if col_indices is not None else segment_ids if segment_ids is not None else src_index
    if num_edges is None:
        if row_offsets.device.type == "cpu":
            edges = int(row_offsets[-1].item())
        elif edge_tensor is not None:
            edges = int(edge_tensor.shape[0])
        else:
            edges = int(row_offsets[-1].item())
    else:
        edges = int(num_edges)
    if edges < 0:
        raise ValueError("num_edges must be non-negative")
    return edges


def _row_counts(row_index: Tensor, num_rows: int) -> Tensor:
    buckets = torch.zeros(num_rows + 1, device=row_index.device, dtype=torch.long)
    in_range = (row_index >= 0) & (row_index < num_rows)
    target = torch.where(in_range, row_index, torch.full_like(row_index, num_rows))
    buckets.scatter_add_(0, target, torch.ones_like(row_index))
    return buckets[:num_rows]


def _infer_rows_from_edges(row_index: Tensor, num_rows: int | None) -> int:
    if num_rows is not None:
        rows = int(num_rows)
        if rows < 0:
            raise ValueError("num_rows must be non-negative")
        return rows
    if row_index.numel() == 0:
        return 0
    return int(row_index.max().item()) + 1


def _validate_cpu_rows(row_index: Tensor, num_rows: int) -> None:
    if row_index.numel() == 0:
        return
    row_long = row_index.to(dtype=torch.long)
    if (row_long < 0).any().item():
        raise ValueError("row_index must be non-negative")
    if (row_long >= num_rows).any().item():
        raise ValueError("row_index entries must be smaller than num_rows")


def _to_device(tensor: Tensor | None, device: torch.device | str, non_blocking: bool) -> Tensor | None:
    return None if tensor is None else tensor.to(device=device, non_blocking=non_blocking)
