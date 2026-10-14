from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

import pytest
import torch
from torch import Tensor
from torch.autograd import forward_ad

from oops import OOpsUnsupportedError, smooth_lddt_pair_distances


@dataclass(frozen=True)
class Case:
    dtype: torch.dtype = torch.float32
    geometry: str = "batched"
    layout: str = "flat"
    compiled: bool = False
    logical_negative: str = ""
    saved_hook: bool = False
    saved_strided: bool = False
    autocast: torch.dtype | None = None
    deterministic: bool = False
    mutation: str = ""
    index_dtype: torch.dtype = torch.int64


def _inputs(case: Case, device: torch.device, pairs: int | None = None) -> tuple[Tensor, Tensor]:
    shape: tuple[int, ...]
    if case.geometry in ("sparse-rows", "unreferenced"):
        atoms = 1025 if pairs in (None, 256) else 1033
        coordinates = 4 * torch.randn(2, atoms, 3, device=device, dtype=case.dtype)
        if case.geometry == "unreferenced":
            coordinates[0, 0] = torch.nan
            coordinates[1, 0] = torch.inf
        return coordinates, torch.randint(
            1 if case.geometry == "unreferenced" else 0, atoms, (2 * (257 if pairs is None else pairs),), device=device
        )
    if case.geometry == "coincident":
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-3.0, 0.5, 4.0]]],
            device=device,
            dtype=case.dtype,
        )
        return coordinates, torch.tensor([0, 1, 2, 2, 0, 3], device=device)
    if case.geometry == "empty":
        return torch.randn(2, 8, 3, device=device, dtype=case.dtype), torch.empty(0, device=device, dtype=torch.int64)
    if pairs is not None:
        shape, atoms = ((2, 16, 3), 16) if pairs == 256 else ((3, 17, 3), 17)
    else:
        atoms, pairs = (24, 60) if case.geometry == "batchless" else (8, 1600)
        shape = (atoms, 3) if case.geometry == "batchless" else (4, atoms, 3)
    coordinates = 4 * torch.randn(shape, device=device, dtype=case.dtype)
    return coordinates, torch.randint(atoms, (2 * pairs,), device=device)


def _runs(case: Case, cuda: torch.device, grad: bool, saved_strided_hook):
    torch.manual_seed(2)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(case.deterministic)
        operation = (
            torch.compile(smooth_lddt_pair_distances, fullgraph=True, dynamic=True)
            if case.compiled
            else smooth_lddt_pair_distances
        )
        sizes = (256, 257) if case.compiled and case.geometry != "empty" else (None,)
        for pairs in sizes:
            coordinates, flat = _inputs(case, cuda, pairs)
            if case.logical_negative:
                coordinates = torch._neg_view(coordinates)
            coordinates.requires_grad_(grad)
            reference_coordinates = coordinates.double().detach().requires_grad_(grad)
            matrix = flat.reshape(-1, 2)
            indices: Tensor | tuple[Tensor, Tensor] = flat
            if case.layout == "matrix":
                indices = matrix
            elif case.layout == "tuple":
                indices = matrix[:, 0], matrix[:, 1]
            with (
                torch.autocast("cuda", dtype=case.autocast or case.dtype, enabled=case.autocast is not None),
                (
                    saved_strided_hook()
                    if case.saved_strided
                    else (
                        torch.autograd.graph.saved_tensors_hooks(
                            lambda t: -t if t.is_floating_point() else t,
                            lambda t: torch._neg_view(t) if t.is_floating_point() else t,
                        )
                        if case.saved_hook
                        else nullcontext()
                    )
                ),
            ):
                actual = operation(coordinates, indices)
            expected = torch.linalg.vector_norm(
                reference_coordinates[..., matrix[:, 0], :] - reference_coordinates[..., matrix[:, 1], :], dim=-1
            )
            assert actual.shape == expected.shape
            assert actual.device == coordinates.device
            assert actual.dtype is torch.float32
            torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)
            yield actual, expected, coordinates, reference_coordinates
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def _check_updated_pairs(cuda: torch.device, case: Case) -> None:
    stream = torch.cuda.Stream(device=cuda)
    with torch.cuda.stream(stream):
        coordinates = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 4.0, 1.0], [3.0, -1.0, 2.0]],
            device=cuda,
            requires_grad=True,
        )
        pairs = torch.tensor([0, 1, 1, 2, 2, 3], device=cuda, dtype=case.index_dtype)
        original = pairs.clone()
        updated = torch.tensor([0, 2, 0, 3, 1, 3], device=cuda, dtype=case.index_dtype)
        if case.mutation == "graph-writer":
            writer = torch.cuda.CUDAGraph()
            with torch.cuda.graph(writer, stream=stream):
                pairs.copy_(updated)
            pairs.copy_(original)
        for _ in range(3):
            value = smooth_lddt_pair_distances(coordinates, pairs)
            torch.autograd.grad(value.sum(), coordinates)
            del value
        if case.mutation == "graph-input":
            consumer = torch.cuda.CUDAGraph()
            with torch.cuda.graph(consumer, stream=stream):
                actual = smooth_lddt_pair_distances(coordinates, pairs)
                gradient = torch.autograd.grad(actual.sum(), coordinates)[0]
            pairs.copy_(updated)
            consumer.replay()
        else:
            writer.replay()
            actual = smooth_lddt_pair_distances(coordinates, pairs)
            gradient = torch.autograd.grad(actual.sum(), coordinates)[0]
        reference = coordinates.detach().double().requires_grad_()
        matrix = updated.reshape(-1, 2).long()
        expected = torch.linalg.vector_norm(reference[matrix[:, 0]] - reference[matrix[:, 1]], dim=-1)
        expected_gradient = torch.autograd.grad(expected.sum(), reference)[0]
    stream.synchronize()
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(gradient.double(), expected_gradient, atol=2e-6, rtol=2e-6)


@pytest.mark.usefixtures("compiler_cache")
class TestSmoothLDDTPairDistances:

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(Case(torch.float16), id="fp16-batched"),
            pytest.param(Case(logical_negative="coordinates"), id="logical-negative"),
            pytest.param(Case(), id="fp32-batched"),
            pytest.param(Case(torch.bfloat16, "batchless"), id="bf16-batchless"),
            pytest.param(Case(layout="matrix"), id="pair-matrix"),
            pytest.param(Case(layout="tuple"), id="pair-tuple"),
            pytest.param(Case(geometry="coincident"), id="coincident-and-self-pairs"),
            pytest.param(Case(autocast=torch.float16), id="fp16-autocast-preserves-fp32"),
            pytest.param(Case(autocast=torch.bfloat16), id="bf16-autocast-preserves-fp32"),
            pytest.param(Case(torch.float16, "empty", compiled=True), id="compiled-empty"),
        ],
    )
    def test_forward(
        self, cuda: torch.device, case: Case, saved_strided_hook: Callable[[], AbstractContextManager[None]]
    ) -> None:
        for actual, _expected, _coordinates, _reference in _runs(
            case, cuda, grad=False, saved_strided_hook=saved_strided_hook
        ):
            assert not actual.requires_grad

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(Case(torch.float16), id="fp16-batched"),
            pytest.param(Case(logical_negative="coordinates"), id="logical-negative"),
            pytest.param(Case(saved_hook=True), id="saved-hook"),
            pytest.param(Case(saved_strided=True), id="saved-strided"),
            pytest.param(Case(torch.bfloat16, "batchless"), id="bf16-batchless"),
            pytest.param(Case(), id="fp32-duplicate-pairs"),
            pytest.param(Case(compiled=True), id="dynamic-duplicate-pairs"),
            pytest.param(Case(geometry="coincident"), id="coincident-and-self-pairs"),
            pytest.param(Case(torch.float16, "empty", compiled=True), id="compiled-empty"),
            pytest.param(Case(torch.bfloat16, compiled=True, deterministic=True), id="dynamic-deterministic"),
            pytest.param(Case(geometry="unreferenced"), id="unreferenced-nonfinite-row"),
            pytest.param(Case(geometry="sparse-rows", compiled=True, deterministic=True), id="sparse-row-tail-dynamic"),
            pytest.param(Case(torch.bfloat16, geometry="sparse-rows"), id="sparse-row-tail-bf16"),
            pytest.param(Case(mutation="graph-input", index_dtype=torch.int32), id="graph-input-int32"),
            pytest.param(Case(mutation="graph-input"), id="graph-input-int64"),
            pytest.param(Case(mutation="graph-writer", index_dtype=torch.int32), id="graph-writer-int32"),
            pytest.param(Case(mutation="graph-writer"), id="graph-writer-int64"),
            pytest.param(Case(torch.float16), marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
        ],
    )
    def test_backward(
        self, cuda: torch.device, case: Case, saved_strided_hook: Callable[[], AbstractContextManager[None]]
    ) -> None:
        if case.mutation:
            _check_updated_pairs(cuda, case)
            return
        for actual, expected, coordinates, reference in _runs(
            case, cuda, grad=True, saved_strided_hook=saved_strided_hook
        ):
            upstream = torch.randn_like(actual)
            if case.logical_negative or case.saved_hook:
                upstream = torch._neg_view(upstream)
            gradient = torch.autograd.grad(actual, coordinates, upstream)[0]
            target = torch.autograd.grad(expected, reference, upstream.double())[0]
            assert gradient.shape == coordinates.shape
            assert gradient.dtype is case.dtype
            torch.testing.assert_close(
                gradient,
                target.to(case.dtype),
                atol=4e-3 if case.dtype is torch.bfloat16 else 6e-4,
                rtol=2e-2 if case.dtype is torch.bfloat16 else 3e-3,
            )

    @pytest.mark.parametrize("fault", ["cpu", "odd-flat", "invalid-matrix", "invalid-tuple", "forward-ad"])
    def test_contract(self, fault: str) -> None:
        coordinates = torch.randn(2, 8, 3)
        indices: Tensor | tuple[Tensor, ...] = torch.tensor([0, 3, 1, 6])
        if fault == "forward-ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(coordinates, torch.ones_like(coordinates))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    smooth_lddt_pair_distances(dual, cast(Tensor, indices))
            return
        if fault == "odd-flat":
            indices = torch.tensor([0, 3, 1])
        elif fault == "invalid-matrix":
            indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
        elif fault == "invalid-tuple":
            indices = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
        with pytest.raises(OOpsUnsupportedError if fault == "cpu" else ValueError):
            smooth_lddt_pair_distances(coordinates, cast(Tensor | tuple[Tensor, Tensor], indices))
