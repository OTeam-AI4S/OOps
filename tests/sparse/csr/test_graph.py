from __future__ import annotations

import pytest
import torch

from oops import CSRGraph, regular_csr_graph, validate_csr_metadata


class TestCSRGraph:

    @pytest.mark.parametrize(
        ("kind", "device"),
        [
            ("offsets", "cpu"),
            ("offsets", "cuda"),
            ("sorted_edges", "cpu"),
            ("sorted_edges", "cuda"),
            ("unsorted_edges", "cpu"),
            ("unsorted_edges", "cuda"),
            ("move", "cuda"),
            ("regular", "cpu"),
            ("empty_regular", "cpu"),
        ],
    )
    def test_forward(self, kind, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        if kind in ("regular", "empty_regular"):
            fanout = 3 if kind == "regular" else 0
            graph = regular_csr_graph(4, fanout, dtype=torch.int32)
            expected = torch.arange(5, dtype=torch.int32) * fanout
            assert graph.num_rows == 4 and graph.num_edges == 4 * fanout
            torch.testing.assert_close(graph.row_offsets, expected)
            return
        if kind == "offsets":
            offsets = torch.tensor([0, 2, 2, 5], device=device)
            columns = torch.tensor([5, 6, 2, 3, 4], device=device)
            graph = CSRGraph.from_row_offsets(offsets, col_indices=columns)
            expected_offsets = offsets
            expected_columns = columns
        else:
            rows = torch.tensor(
                [2, 0, 1, 0, 2] if kind in ("unsorted_edges", "move") else [0, 0, 1, 2, 2],
                device="cpu" if kind == "move" else device,
            )
            columns = torch.tensor([1, 2, 3, 4, 5], device=rows.device)
            sort = kind in ("unsorted_edges", "move")
            graph = CSRGraph.from_edges(rows, columns, num_rows=4, sort=sort)
            if kind == "move":
                graph = graph.to(device)
            expected_offsets = torch.tensor([0, 2, 3, 5, 5], device=device)
            expected_columns = torch.tensor([2, 4, 3, 1, 5] if sort else [1, 2, 3, 4, 5], device=device)
            torch.testing.assert_close(graph.segment_ids, torch.tensor([0, 0, 1, 2, 2], device=device))
        torch.testing.assert_close(graph.row_offsets, expected_offsets)
        torch.testing.assert_close(graph.col_indices, expected_columns)
        assert graph.row_offsets.device.type == device
        validate_csr_metadata(graph.row_offsets, num_edges=5, segment_ids=graph.segment_ids)

    @pytest.mark.parametrize("case", ["offset_start", "offset_order", "columns", "edge_count", "int32_overflow"])
    def test_contract(self, case):
        if case == "int32_overflow":
            with pytest.raises(ValueError, match="int32"):
                regular_csr_graph(torch.iinfo(torch.int32).max, 2, dtype=torch.int32)
            return
        offsets = (
            torch.tensor([1, 2])
            if case == "offset_start"
            else (torch.tensor([0, 2, 1]) if case == "offset_order" else torch.tensor([0, 2, 4]))
        )
        if case == "edge_count":
            with pytest.raises(ValueError):
                validate_csr_metadata(offsets, num_edges=3)
        else:
            columns = torch.arange(3 if case == "columns" else int(offsets[-1]))
            with pytest.raises(ValueError):
                CSRGraph.from_row_offsets(offsets, col_indices=columns)
