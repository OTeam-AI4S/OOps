from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import pytest
import torch
from torch.autograd import forward_ad

from oops import OOpsUnsupportedError, smooth_lddt_loss_sparse

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@dataclass(frozen=True)
class Case:
    route: str = "coordinates"
    predicted_dtype: torch.dtype = torch.float32
    truth_dtype: torch.dtype = torch.float32
    batched_truth: bool = False
    empty: bool = False
    autocast: torch.dtype | None = None
    compiled: bool = False
    logical_negative: str = ""
    saved_hook: bool = False
    saved_strided: bool = False
    demand: tuple[bool, bool] = (True, True)
    deterministic: bool = False
    wide: bool = False
    unreferenced: bool = False


_FORWARD_CASES = [
    pytest.param(Case("mask", logical_negative="predicted"), id="negative-mask-prediction"),
    pytest.param(Case("distances", logical_negative="truth"), id="negative-distances-truth"),
    pytest.param(Case("coordinates", logical_negative="truth"), id="negative-coordinates-truth"),
    pytest.param(Case("mask"), id="mask-fp32"),
    pytest.param(Case("distances"), id="distances-fp32"),
    pytest.param(Case("coordinates"), id="coordinates-fp32"),
    pytest.param(Case("mask", torch.float16, autocast=torch.float16), id="mask-autocast"),
    pytest.param(Case("distances", torch.bfloat16, autocast=torch.bfloat16), id="distances-autocast"),
    pytest.param(Case("coordinates", torch.bfloat16, autocast=torch.bfloat16), id="coordinates-autocast"),
    pytest.param(Case("mask", empty=True, deterministic=True), id="deterministic-empty-mask"),
    pytest.param(Case("distances", deterministic=True), id="deterministic-distances"),
    pytest.param(Case("coordinates", deterministic=True), id="deterministic-coordinates"),
]
_BACKWARD_CASES = [
    pytest.param(Case("mask", logical_negative="predicted"), id="negative-mask-prediction"),
    pytest.param(Case("distances", logical_negative="truth"), id="negative-distances-truth"),
    pytest.param(Case("coordinates", logical_negative="truth"), id="negative-coordinates-truth"),
    pytest.param(Case("mask", saved_hook=True), id="mask-saved-hook"),
    pytest.param(Case("distances", saved_hook=True), id="distances-saved-hook"),
    pytest.param(Case("coordinates", saved_hook=True), id="coordinates-saved-hook"),
    pytest.param(Case("mask", saved_strided=True), id="mask-saved-strided"),
    pytest.param(Case("distances", saved_strided=True), id="distances-saved-strided"),
    pytest.param(Case("coordinates", saved_strided=True), id="coordinates-saved-strided"),
    pytest.param(Case("mask", torch.float16), id="mask-fp16"),
    pytest.param(Case("distances", torch.bfloat16, torch.float16), id="distances-bf16"),
    pytest.param(Case("distances", truth_dtype=torch.bfloat16, batched_truth=True), id="batched-distances"),
    pytest.param(Case(), id="coordinates-fp32"),
    pytest.param(Case("distances", torch.float16, torch.bfloat16, empty=True), id="empty-mixed"),
    pytest.param(Case("distances", empty=True, deterministic=True), id="empty-deterministic"),
    pytest.param(Case("mask", torch.bfloat16, batched_truth=True, compiled=True), id="mask-dynamic"),
    pytest.param(Case("distances", torch.bfloat16, batched_truth=True, compiled=True), id="distances-dynamic"),
    pytest.param(Case("coordinates", torch.bfloat16, batched_truth=True, compiled=True), id="coordinates-dynamic"),
    pytest.param(Case("distances", demand=(True, False)), id="prediction-only"),
    pytest.param(Case("coordinates", demand=(False, True)), id="truth-only"),
    pytest.param(Case(wide=True, unreferenced=True, demand=(True, False)), id="unreferenced-nonfinite-row"),
    pytest.param(Case(wide=True, compiled=True), id="wide-row-tail-dynamic"),
    pytest.param(
        Case(wide=True, predicted_dtype=torch.bfloat16, truth_dtype=torch.bfloat16, demand=(True, False)),
        id="wide-row-tail-bf16-prediction",
    ),
    pytest.param(Case("mask", torch.float16), marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
]


def _reference(predicted: torch.Tensor, pairs: torch.Tensor, true_distances: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return predicted.new_ones(()) + predicted.reshape(-1)[:0].sum() + true_distances.reshape(-1)[:0].sum()
    indices = pairs.reshape(-1, 2)
    predicted_distances = torch.linalg.vector_norm(predicted[:, indices[:, 0]] - predicted[:, indices[:, 1]], dim=-1)
    error = (predicted_distances - true_distances).abs()
    score = torch.zeros_like(error)
    for threshold in (0.5, 1.0, 2.0, 4.0):
        score += torch.sigmoid(threshold - error)
    score *= 0.25
    return 1.0 - score.mean()


def _distances(coordinates: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    indices = pairs.reshape(-1, 2)
    return torch.linalg.vector_norm(coordinates[..., indices[:, 0], :] - coordinates[..., indices[:, 1], :], dim=-1)


def _runs(case: Case, grad: bool, cuda: torch.device, saved_strided_hook):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def apply(predicted, truth, pair_data):
        if case.route == "mask":
            return smooth_lddt_loss_sparse(predicted, truth, pair_data)
        if case.route == "coordinates":
            return smooth_lddt_loss_sparse(predicted, truth, pair_indices=pair_data)
        return smooth_lddt_loss_sparse(predicted, pair_indices=pair_data, true_pair_distances=truth)

    function = torch.compile(apply, fullgraph=True, dynamic=True) if case.compiled else apply
    sizes: tuple[tuple[int, int], ...]
    if case.wide:
        sizes = ((1025, 4096), (1033, 4097)) if case.compiled else ((1025, 4096),)
    elif case.compiled:
        sizes = ((11, 32), (12, 33)) if case.route == "mask" else ((16, 32), (16, 33))
    else:
        sizes = ((4 if case.deterministic else 9, 2),)
    try:
        torch.use_deterministic_algorithms(case.deterministic)
        for atoms, pair_count in sizes:
            torch.manual_seed(pair_count if case.compiled else 41)
            true_base = 2.0 * torch.randn((2, atoms, 3) if case.batched_truth else (atoms, 3), device=cuda)
            predicted_base = true_base + 0.2 * torch.randn(2, atoms, 3, device=cuda)
            mask = torch.rand(atoms, atoms, device=cuda) < 0.3
            if case.compiled and case.route == "mask":
                mask.fill_(True)
            mask.fill_diagonal_(False)
            if case.empty:
                mask.zero_()
            pairs = torch.nonzero(mask).flatten()
            if (case.compiled or case.wide) and case.route != "mask":
                pairs = torch.randint(1 if case.unreferenced else 0, atoms, (2 * pair_count,), device=cuda)
            elif case.deterministic and not case.empty:
                pairs = torch.tensor([0, 1, 1, 2], device=cuda)
            if case.unreferenced:
                predicted_base[0, 0] = torch.nan
                predicted_base[1, 0] = torch.inf
            truth_source = _distances(true_base, pairs) if case.route == "distances" else true_base
            predicted = predicted_base.to(case.predicted_dtype)
            truth = truth_source.to(case.truth_dtype)
            if case.logical_negative == "predicted":
                predicted = torch._neg_view(predicted)
            elif case.logical_negative == "truth":
                truth = torch._neg_view(truth)
            predicted.requires_grad_(grad and case.demand[0])
            truth.requires_grad_(grad and case.demand[1])
            reference_predicted = predicted.detach().float().requires_grad_(predicted.requires_grad)
            reference_truth = truth.detach().float().requires_grad_(truth.requires_grad)
            pair_data = mask if case.route == "mask" else pairs
            if case.compiled:
                for tensor in (predicted, truth, pair_data):
                    torch._dynamo.mark_dynamic(tensor, 0)
                    if (
                        tensor.ndim == 3
                        or (case.route == "mask" and tensor is pair_data)
                        or (case.route == "distances" and case.batched_truth and tensor is truth)
                    ):
                        torch._dynamo.mark_dynamic(tensor, 1)
            with (
                torch.autocast("cuda", dtype=case.autocast or torch.float16, enabled=case.autocast is not None),
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
                actual = function(predicted, truth, pair_data)
            reference_distances = reference_truth if case.route == "distances" else _distances(reference_truth, pairs)
            expected = _reference(reference_predicted, pairs, reference_distances)
            assert actual.shape == expected.shape == torch.Size([])
            assert actual.dtype is torch.float32
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            if case.empty:
                torch.testing.assert_close(actual, torch.ones((), device=cuda, dtype=torch.float32))
            yield actual, expected, (predicted, truth), (reference_predicted, reference_truth)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.usefixtures("compiler_cache")
class TestSmoothLDDTLossSparse:

    @_CUDA
    @pytest.mark.parametrize("case", _FORWARD_CASES)
    def test_forward(
        self, cuda: torch.device, case: Case, saved_strided_hook: Callable[[], AbstractContextManager[None]]
    ) -> None:
        for actual, _expected, _leaves, _reference_leaves in _runs(
            case, grad=False, cuda=cuda, saved_strided_hook=saved_strided_hook
        ):
            assert not actual.requires_grad

    @_CUDA
    @pytest.mark.parametrize("case", _BACKWARD_CASES)
    def test_backward(
        self, cuda: torch.device, case: Case, saved_strided_hook: Callable[[], AbstractContextManager[None]]
    ) -> None:
        for actual, expected, leaves, reference_leaves in _runs(
            case, grad=True, cuda=cuda, saved_strided_hook=saved_strided_hook
        ):
            actual.backward()
            expected.backward()
            for leaf, reference in zip(leaves, reference_leaves, strict=True):
                if not leaf.requires_grad:
                    assert leaf.grad is None
                    continue
                assert leaf.grad is not None and reference.grad is not None
                assert leaf.grad.dtype is leaf.dtype
                assert leaf.grad.shape == leaf.shape
                torch.testing.assert_close(
                    leaf.grad,
                    reference.grad.to(leaf.dtype),
                    atol=4e-3 if leaf.dtype is torch.bfloat16 else 6e-4,
                    rtol=2e-2 if leaf.dtype is torch.bfloat16 else 3e-3,
                )
                if case.empty:
                    assert torch.count_nonzero(leaf.grad) == 0

    @pytest.mark.parametrize(
        "contract",
        [
            "forward-ad-prediction",
            "forward-ad-truth",
            "missing",
            "redundant-pairs",
            "redundant-truth",
            "device",
            "deterministic-forward",
            "deterministic-backward",
        ],
    )
    def test_contract(self, contract: str) -> None:
        deterministic = contract.startswith("deterministic-")
        if deterministic and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        device = "cuda" if deterministic else "cpu"
        predicted = torch.randn(2, 12 if contract == "deterministic-forward" else 4, 3, device=device)
        truth = torch.randn(predicted.shape[1], 3, device=device)
        pairs = torch.tensor([0, 1, 2, 3], device=device)
        if contract.startswith("forward-ad-"):
            with forward_ad.dual_level():
                if contract == "forward-ad-prediction":
                    predicted = forward_ad.make_dual(predicted, torch.ones_like(predicted))
                else:
                    truth = forward_ad.make_dual(truth, torch.ones_like(truth))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    smooth_lddt_loss_sparse(predicted, truth, pair_indices=pairs)
            return
        if deterministic:
            previous = torch.are_deterministic_algorithms_enabled()
            previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                if contract == "deterministic-forward":
                    torch.use_deterministic_algorithms(True)
                    with pytest.raises(OOpsUnsupportedError, match="does not have a deterministic implementation"):
                        smooth_lddt_loss_sparse(
                            predicted,
                            pair_indices=torch.tensor([0, 1] * 33, device=device),
                            true_pair_distances=torch.rand(33, device=device),
                        )
                else:
                    torch.use_deterministic_algorithms(False)
                    predicted.requires_grad_()
                    distances = torch.rand(2, device=device, requires_grad=True)
                    output = smooth_lddt_loss_sparse(predicted, pair_indices=pairs, true_pair_distances=distances)
                    torch.use_deterministic_algorithms(True)
                    with pytest.raises(OOpsUnsupportedError, match="does not have a deterministic implementation"):
                        torch.autograd.grad(output, (predicted, distances))
            finally:
                torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
            return
        mask = torch.ones(4, 4, device=device, dtype=torch.bool)
        error = OOpsUnsupportedError if contract == "device" else TypeError
        with pytest.raises(error):
            smooth_lddt_loss_sparse(
                predicted,
                None if contract == "missing" else truth,
                mask if contract in ("redundant-pairs", "device") else None,
                pair_indices=pairs if contract in ("redundant-pairs", "redundant-truth") else None,
                true_pair_distances=torch.ones(2, device=device) if contract == "redundant-truth" else None,
            )
