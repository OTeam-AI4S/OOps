from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext

import pytest
import torch
from danling.tensors import NestedTensor
from torch import Tensor
from torch.autograd import forward_ad
from torch.utils.checkpoint import checkpoint

from oops import OOpsUnsupportedError, masked_knn

_DTYPE_TOLERANCE = {torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2), torch.float32: (2e-6, 2e-6)}


@contextmanager
def _deterministic_algorithms(*, warn_only: bool = False):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def _segment_oracle(
    coordinates: Tensor,
    mask: Tensor,
    num_neighbors: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Independent per-segment oracle; Python loops are test-only."""

    length = coordinates.shape[0]
    if length == 0:
        return (
            coordinates.new_empty((0, num_neighbors)),
            torch.empty((0, num_neighbors), device=coordinates.device, dtype=torch.long),
            torch.empty((0, num_neighbors), device=coordinates.device, dtype=torch.bool),
        )

    truth = mask.bool()
    candidates = torch.nonzero(truth, as_tuple=False).flatten()
    distance_rows: list[Tensor] = []
    index_rows: list[Tensor] = []
    valid_rows: list[Tensor] = []
    for query in range(length):
        if not bool(truth[query]):
            distance_rows.append(torch.ones(num_neighbors, device=coordinates.device, dtype=torch.float32))
            index_rows.append(torch.zeros(num_neighbors, device=coordinates.device, dtype=torch.long))
            valid_rows.append(torch.zeros(num_neighbors, device=coordinates.device, dtype=torch.bool))
            continue

        delta = coordinates[query].float() - coordinates[candidates].float()
        candidate_distance_sq = delta.square().sum(dim=-1)
        candidate_distances = (
            torch.linalg.vector_norm(delta, dim=-1) if eps == 0 else torch.sqrt(candidate_distance_sq + float(eps))
        )
        # Order by squared distance; norm rounding can create ties.
        order = torch.argsort(candidate_distance_sq, stable=True)
        ordered_indices = candidates[order]
        ordered_distances = candidate_distances[order]
        real_count = min(num_neighbors, int(ordered_indices.numel()))
        filler_count = num_neighbors - real_count
        row_max = candidate_distances.max() if candidate_distances.numel() else candidate_distances.new_tensor(0.0)
        distance_rows.append(
            torch.cat(
                (
                    ordered_distances[:real_count],
                    # Filler slots have zero VJP.
                    (row_max.detach() + 1.0).expand(filler_count),
                )
            )
        )
        index_rows.append(
            torch.cat(
                (
                    ordered_indices[:real_count],
                    torch.zeros(filler_count, device=coordinates.device, dtype=torch.long),
                )
            )
        )
        valid_rows.append(
            torch.cat(
                (
                    torch.ones(real_count, device=coordinates.device, dtype=torch.bool),
                    torch.zeros(filler_count, device=coordinates.device, dtype=torch.bool),
                )
            )
        )

    return (
        torch.stack(distance_rows).to(dtype=coordinates.dtype),
        torch.stack(index_rows),
        torch.stack(valid_rows),
    )


def _oracle(
    coordinate_parts: Sequence[Tensor],
    mask_parts: Sequence[Tensor],
    num_neighbors: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    outputs = tuple(
        _segment_oracle(coordinates, mask, num_neighbors, eps)
        for coordinates, mask in zip(coordinate_parts, mask_parts, strict=True)
    )
    return tuple(torch.cat(parts, dim=0) for parts in zip(*outputs, strict=True))


def _case(
    coordinate_parts: Sequence[Tensor],
    mask_parts: Sequence[Tensor],
    *,
    dtype: torch.dtype = torch.float32,
    mask_dtype: torch.dtype = torch.bool,
    device: torch.device | str = "cuda",
    requires_grad: bool = False,
) -> tuple[NestedTensor, NestedTensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    lengths = tuple(part.shape[0] for part in coordinate_parts)
    coordinate_template = NestedTensor(
        [torch.empty(length, 3) for length in lengths],
        ragged_dims=(0,),
    )
    mask_template = NestedTensor(
        [torch.empty(length, dtype=mask_dtype) for length in lengths],
        ragged_dims=(0,),
    )
    values = torch.cat(tuple(part.to(dtype=dtype) for part in coordinate_parts), dim=0).to(device)
    values.requires_grad_(requires_grad)
    mask_values = torch.cat(tuple(part.to(dtype=mask_dtype) for part in mask_parts), dim=0).to(device)
    coordinates = coordinate_template.packed_like(values)
    mask = mask_template.packed_like(mask_values)
    coordinate_views = tuple(values.split(lengths))
    mask_views = tuple(mask_values.split(lengths))
    return coordinates, mask, values, coordinate_views, mask_views


def _inputs(kind, dtype, training, variant=0):
    torch.manual_seed(93 + variant)
    if kind == "ties":
        parts = (torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),)
        masks = (torch.ones(4),)
    elif kind == "empty":
        parts = (torch.empty(0, 3), torch.randn(3, 3), torch.randn(2, 3))
        masks = (torch.empty(0), torch.zeros(3), torch.tensor([0, 1]))
    elif kind == "zero":
        parts, masks = (torch.zeros(3, 3),), (torch.ones(3),)
    elif kind == "rounded_norm":
        parts = (torch.tensor([[1.0, 3e-4, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),)
        masks = (torch.ones(3),)
    else:
        lengths = (4, 3) if variant == 0 else (5, 7)
        parts = tuple(torch.randn(length, 3) for length in lengths)
        masks = tuple(torch.arange(length).remainder(3) != 1 for length in lengths)
    return _case(parts, masks, dtype=dtype, mask_dtype=dtype, requires_grad=training)


def _run(coordinates, mask, neighbors, eps):
    outputs = masked_knn(coordinates, mask, neighbors, eps)
    return tuple(output.concat for output in outputs)


class TestMaskedKNN:

    @pytest.mark.parametrize(
        ("dtype", "kind", "neighbors", "mode", "eps"),
        [
            (torch.float16, "ordinary", 1, "eager", 1e-6),
            (torch.bfloat16, "ordinary", 5, "eager", 1e-6),
            (torch.float32, "empty", 5, "eager", 1e-6),
            (torch.float32, "ties", 4, "eager", 1e-6),
            (torch.float32, "ordinary", 48, "compile", 1e-6),
            (torch.float32, "ordinary", 128, "eager", 1e-6),
            (torch.float32, "rounded_norm", 2, "eager", 0.0),
            (torch.float32, "rounded_norm", 2, "compile", 0.0),
        ],
    )
    def test_forward(self, cuda, dtype, kind, neighbors, mode, eps):
        del cuda
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        for variant in (0, 1) if mode == "compile" else (0,):
            coordinates, mask, _, coordinate_parts, mask_parts = _inputs(kind, dtype, False, variant)
            actual = operation(coordinates, mask, neighbors, eps)
            expected = _oracle(coordinate_parts, mask_parts, neighbors, eps)
            atol, rtol = _DTYPE_TOLERANCE[dtype]
            torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
            torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
            assert actual[0].dtype is dtype and actual[1].dtype is torch.int64 and actual[2].dtype is torch.bool

    @pytest.mark.parametrize(
        ("dtype", "kind", "mode"),
        [
            (torch.float16, "ordinary", "eager"),
            (torch.bfloat16, "ordinary", "checkpoint"),
            (torch.float32, "ordinary", "compile"),
            (torch.float32, "ordinary", "deterministic"),
            (torch.float32, "zero", "eager"),
            (torch.float32, "ordinary", "saved-strided"),
            (torch.float32, "empty", "eager"),
            pytest.param(torch.float16, "ordinary", "eager", marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
        ],
    )
    def test_backward(self, cuda, dtype, kind, mode, saved_strided_hook):
        del cuda
        coordinates, mask, values, coordinate_parts, mask_parts = _inputs(kind, dtype, True)
        eps = 0.0 if kind == "zero" else 1e-6
        expected, _, _ = _oracle(coordinate_parts, mask_parts, 5, eps)
        operation = torch.compile(_run, fullgraph=True) if mode == "compile" else _run
        with _deterministic_algorithms() if mode == "deterministic" else nullcontext():
            with saved_strided_hook() if mode == "saved-strided" else nullcontext():
                actual = (
                    checkpoint(operation, coordinates, mask, 5, eps, use_reentrant=False)
                    if mode == "checkpoint"
                    else operation(coordinates, mask, 5, eps)
                )
            upstream = torch.randn_like(actual[0])
            gradient = torch.autograd.grad(actual[0], values, upstream, retain_graph=mode == "deterministic")[0]
            if mode == "deterministic":
                repeated = torch.autograd.grad(actual[0], values, upstream)[0]
                torch.testing.assert_close(gradient, repeated, atol=0, rtol=0)
        reference = (
            torch.autograd.grad(expected, values, upstream)[0] if expected.requires_grad else torch.zeros_like(values)
        )
        atol, rtol = _DTYPE_TOLERANCE[dtype]
        torch.testing.assert_close(gradient, reference, atol=atol, rtol=rtol)
        assert torch.isfinite(gradient).all()

    @pytest.mark.parametrize(
        "case",
        [
            "cpu",
            "coordinate_shape",
            "mask_shape",
            "topology",
            "neighbors",
            "epsilon",
            "forward_ad",
            "per_sample_topology",
            "per_sample_topology_compile",
        ],
    )
    def test_contract(self, case, request):
        coordinates = NestedTensor([torch.randn(3, 4 if case == "coordinate_shape" else 3)], ragged_dims=(0,))
        mask = NestedTensor(
            [torch.ones(3, 1) if case == "mask_shape" else torch.ones(4 if case == "topology" else 3)], ragged_dims=(0,)
        )
        if case in ("per_sample_topology", "per_sample_topology_compile"):
            device = request.getfixturevalue("cuda")
            coordinates = NestedTensor(
                [torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)], ragged_dims=(0,)
            )
            valid_mask = NestedTensor([torch.ones(3, device=device), torch.ones(3, device=device)], ragged_dims=(0,))
            invalid_mask = NestedTensor([torch.ones(2, device=device), torch.ones(4, device=device)], ragged_dims=(0,))
            operation = (
                torch.compile(masked_knn, fullgraph=True) if case == "per_sample_topology_compile" else masked_knn
            )
            operation(coordinates, valid_mask, 3)
            with pytest.raises(ValueError, match="matching per-sample lengths"):
                operation(coordinates, invalid_mask, 3)
        elif case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(coordinates.concat, torch.ones_like(coordinates.concat))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode"):
                    masked_knn(coordinates.packed_like(dual), mask, 3)
        else:
            with pytest.raises(OOpsUnsupportedError if case == "cpu" else ValueError):
                masked_knn(
                    coordinates, mask, 129 if case == "neighbors" else 3, float("nan") if case == "epsilon" else 1e-6
                )
