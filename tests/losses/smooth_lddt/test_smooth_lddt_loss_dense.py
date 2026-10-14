from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import pytest
import torch
from torch.autograd import forward_ad

from oops import OOpsUnsupportedError, smooth_lddt_loss_dense

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@dataclass(frozen=True)
class Case:
    predicted_dtype: torch.dtype = torch.float32
    true_dtype: torch.dtype = torch.float32
    nucleotide_dtype: torch.dtype = torch.float32
    mask_dtype: torch.dtype = torch.float32
    autocast: torch.dtype | None = None
    multiplicity: int = 1
    compiled: bool = False
    logical_negative: str = ""
    saved_hook: bool = False
    saved_strided: bool = False
    demand: tuple[bool, bool, bool] = (True, True, True)
    deterministic: bool = False


_FORWARD_CASES = [
    pytest.param(Case(), id="fp32"),
    pytest.param(Case(logical_negative="predicted"), id="negative-prediction"),
    pytest.param(Case(logical_negative="true"), id="negative-truth"),
    pytest.param(Case(torch.float16, autocast=torch.float16), id="fp16-autocast"),
    pytest.param(Case(torch.bfloat16, autocast=torch.bfloat16), id="bf16-autocast"),
    pytest.param(Case(deterministic=True), id="deterministic-small"),
]
_BACKWARD_CASES = [
    pytest.param(Case(logical_negative="predicted"), id="negative-prediction"),
    pytest.param(Case(logical_negative="true"), id="negative-truth"),
    pytest.param(Case(logical_negative="mask"), id="negative-mask"),
    pytest.param(Case(saved_hook=True), id="saved-hook"),
    pytest.param(Case(saved_strided=True), id="saved-strided"),
    pytest.param(Case(torch.float16, torch.float32, torch.bool, torch.bool), id="bool-mask"),
    pytest.param(Case(torch.bfloat16, torch.float16, torch.float16, torch.bfloat16), id="bf16-mask"),
    pytest.param(Case(torch.float32, torch.bfloat16, torch.bfloat16, torch.float32), id="fp32-mask"),
    pytest.param(Case(mask_dtype=torch.float16), id="fp16-mask"),
    pytest.param(Case(multiplicity=3), id="multiplicity"),
    pytest.param(
        Case(torch.bfloat16, nucleotide_dtype=torch.bool, mask_dtype=torch.float16, compiled=True), id="dynamic"
    ),
    pytest.param(Case(demand=(True, False, False)), id="prediction-only"),
    pytest.param(Case(demand=(False, True, False)), id="truth-only"),
    pytest.param(Case(demand=(False, False, True)), id="mask-only"),
    pytest.param(
        Case(torch.float16, torch.float32, torch.bool, torch.bool),
        marks=pytest.mark.cuda_device(1),
        id="secondary_cuda",
    ),
]


def _reference(
    predicted: torch.Tensor,
    true: torch.Tensor,
    nucleotide: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    true_distances = torch.cdist(true, true)
    predicted_distances = torch.cdist(predicted, predicted)
    cutoff = torch.where(nucleotide != 0, 30.0, 15.0).unsqueeze(-1)
    pairs = (true_distances < cutoff).to(true.dtype)
    pairs.diagonal(dim1=-2, dim2=-1).zero_()
    pairs *= mask.unsqueeze(-1) * mask.unsqueeze(-2)
    error = (true_distances - predicted_distances).abs()
    score = sum(torch.sigmoid(threshold - error) for threshold in (0.5, 1.0, 2.0, 4.0)) * 0.25
    return 1.0 - (score * pairs).sum((-1, -2)).div(pairs.sum((-1, -2)).clamp_min(1)).mean()


def _runs(case: Case, grad: bool, cuda: torch.device, saved_strided_hook):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    function = (
        torch.compile(smooth_lddt_loss_dense, fullgraph=True, dynamic=True) if case.compiled else smooth_lddt_loss_dense
    )
    shapes = (
        ((256, 11), (257, 12))
        if case.compiled
        else ((2 * case.multiplicity, 8 if case.deterministic else 16 if case.multiplicity > 1 else 12),)
    )
    try:
        torch.use_deterministic_algorithms(case.deterministic)
        for batch, atoms in shapes:
            torch.manual_seed(batch if case.compiled else 31)
            true_base = 3.0 * torch.randn(batch, atoms, 3, device=cuda)
            predicted = (true_base + 0.2 * torch.randn_like(true_base)).to(case.predicted_dtype)
            true = true_base.to(case.true_dtype)
            nucleotide = (torch.rand(batch // case.multiplicity, atoms, device=cuda) > 0.7).to(case.nucleotide_dtype)
            mask = (torch.rand(batch // case.multiplicity, atoms, device=cuda) > 0.2).to(case.mask_dtype)
            if case.autocast is not None or case.compiled:
                mask = torch.rand_like(mask)
            if case.multiplicity > 1:
                mask.fill_(1)
                mask[:, ::5] = 0.5
            if case.logical_negative == "predicted":
                predicted = torch._neg_view(predicted)
            elif case.logical_negative == "true":
                true = torch._neg_view(true)
            elif case.logical_negative == "mask":
                mask = torch._neg_view(mask)
            leaves = (predicted, true, mask)
            for leaf, demanded in zip(leaves, case.demand, strict=True):
                if leaf.is_floating_point():
                    leaf.requires_grad_(grad and demanded)
            reference_leaves = tuple(leaf.detach().float().requires_grad_(leaf.requires_grad) for leaf in leaves)
            reference_predicted, reference_true, reference_mask = reference_leaves
            if case.compiled:
                for tensor in (predicted, true, nucleotide, mask):
                    torch._dynamo.mark_dynamic(tensor, 0)
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
                actual = function(predicted, true, nucleotide, mask, multiplicity=case.multiplicity)
            expected = _reference(
                reference_predicted,
                reference_true,
                nucleotide.repeat_interleave(case.multiplicity, 0),
                reference_mask.repeat_interleave(case.multiplicity, 0),
            )
            assert actual.shape == expected.shape == torch.Size([])
            assert actual.dtype is torch.float32
            tolerance = 1e-4 if case.compiled else 2e-5
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            yield actual, expected, leaves, reference_leaves
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.usefixtures("compiler_cache")
class TestSmoothLDDTLossDense:

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
                    atol=2e-6 if case.multiplicity > 1 else 4e-3 if leaf.dtype is torch.bfloat16 else 6e-4,
                    rtol=2e-3 if case.multiplicity > 1 else 2e-2 if leaf.dtype is torch.bfloat16 else 3e-3,
                )

    @pytest.mark.parametrize(
        "contract", ["device", "shape", "multiplicity", "forward-ad", "deterministic-strict", "deterministic-warn"]
    )
    def test_contract(self, contract: str) -> None:
        deterministic = contract.startswith("deterministic-")
        if deterministic and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        device = "cuda" if deterministic else "cpu"
        shape = (8, 3) if contract == "shape" else (2, 12 if deterministic else 8, 3)
        predicted = torch.randn(shape, device=device)
        true = torch.randn_like(predicted)
        metadata = torch.ones(predicted.shape[:-1], device=device)
        if contract == "forward-ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(predicted, torch.ones_like(predicted))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    smooth_lddt_loss_dense(dual, true, metadata, metadata)
            return
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=contract == "deterministic-warn")
            if contract == "deterministic-warn":
                with pytest.warns(UserWarning, match="does not have a deterministic implementation"):
                    smooth_lddt_loss_dense(predicted, true, metadata, metadata)
            else:
                error = ValueError if contract in ("shape", "multiplicity") else OOpsUnsupportedError
                with pytest.raises(
                    error, match="does not have a deterministic implementation" if deterministic else None
                ):
                    smooth_lddt_loss_dense(
                        predicted, true, metadata, metadata, multiplicity=0 if contract == "multiplicity" else 1
                    )
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
