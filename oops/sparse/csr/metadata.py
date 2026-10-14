from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import is_compiling

from .graph import CSRGraph, validate_csr_metadata

INDEX_DTYPES = (torch.int32, torch.int64)


def validate_row_offsets(
    row_offsets: Tensor,
    *,
    num_edges: int,
    edge_name: str = "values",
) -> int:

    if row_offsets.dim() != 1:
        raise ValueError("row_offsets must have shape (N + 1,)")
    if row_offsets.dtype not in INDEX_DTYPES:
        raise ValueError("row_offsets must be an int32 or int64 tensor")
    if row_offsets.numel() == 0:
        raise ValueError("row_offsets must contain at least one offset")
    if row_offsets.device.type == "cpu" and not is_compiling():
        if row_offsets[0].item() != 0:
            raise ValueError("row_offsets must start at zero")
        if (row_offsets[1:] < row_offsets[:-1]).any().item():
            raise ValueError("row_offsets must be monotonically non-decreasing")
        if row_offsets[-1].item() != num_edges:
            raise ValueError(f"row_offsets[-1] must equal {edge_name}.shape[0]")
    return row_offsets.numel() - 1


def canonical_row_offsets(
    row_offsets: Tensor,
    *,
    device: torch.device,
    num_edges: int,
    num_segments: int | None,
) -> tuple[Tensor, int, int]:
    inferred_segments = validate_row_offsets(row_offsets, num_edges=num_edges)
    if row_offsets.device != device:
        raise ValueError(f"row_offsets must be on {device}, got {row_offsets.device}")
    out_segments = inferred_segments if num_segments is None else int(num_segments)
    if out_segments < inferred_segments:
        raise ValueError("num_segments may not be smaller than row_offsets.shape[0] - 1")
    return row_offsets, inferred_segments, out_segments


def canonical_csr_metadata(
    metadata: Tensor | CSRGraph,
    *,
    device: torch.device,
    num_edges: int,
    num_segments: int | None,
    segment_ids: Tensor | None,
) -> tuple[Tensor, int, int, Tensor | None]:

    if isinstance(metadata, CSRGraph):
        if segment_ids is not None:
            raise TypeError("segment_ids must not be passed separately when row_offsets is a CSRGraph")
        if metadata.num_edges != num_edges:
            raise ValueError(f"CSRGraph.num_edges must match values.shape[0] ({num_edges})")
        row_offsets = metadata.row_offsets
        segment_ids = metadata.segment_ids
    else:
        row_offsets = metadata

    offsets, inferred_segments, out_segments = canonical_row_offsets(
        row_offsets,
        device=device,
        num_edges=num_edges,
        num_segments=num_segments,
    )
    validate_cached_segment_ids(segment_ids, row_offsets, num_edges)
    return offsets, inferred_segments, out_segments, segment_ids


def validate_cached_segment_ids(
    segment_ids: Tensor | None,
    row_offsets: Tensor,
    num_edges: int,
) -> None:
    if segment_ids is None:
        return
    if segment_ids.dim() != 1:
        raise ValueError("segment_ids must have shape (E,)")
    if segment_ids.dtype not in INDEX_DTYPES:
        raise ValueError("segment_ids must be an int32 or int64 tensor")
    if segment_ids.shape[0] != num_edges:
        raise ValueError("segment_ids length must match row_offsets[-1]")
    if segment_ids.device != row_offsets.device:
        raise ValueError("segment_ids must be on the same device as row_offsets")
    if row_offsets.device.type == "cpu" and segment_ids.device.type == "cpu" and not is_compiling():
        validate_csr_metadata(row_offsets, num_edges=num_edges, segment_ids=segment_ids)


def canonical_segment_ids(
    row_offsets: Tensor,
    num_edges: int,
    segment_ids: Tensor | None,
) -> Tensor:

    positions = torch.arange(num_edges, device=row_offsets.device, dtype=torch.int64)
    ids = segment_ids
    if ids is None:
        ids = torch.searchsorted(
            row_offsets[1:],
            positions,
            right=True,
            out_int32=row_offsets.dtype == torch.int32,
        )
    covered = (positions >= row_offsets[0]) & (positions < row_offsets[-1])
    return torch.where(covered, ids, row_offsets.numel() - 1)
