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

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

import pytest
import torch
from torch.autograd import forward_ad
from torch.nn import functional as F

from oops import OOpsUnsupportedError, distogram_cross_entropy


@dataclass(frozen=True)
class DistogramCase:
    dtype: torch.dtype = torch.float32
    reduction: str | None = "mean"
    shapes: tuple[tuple[int, int, int], ...] = ((2, 16, 8),)
    mask_kind: str = "float"
    strided: bool = False
    autocast: bool = False
    compiled: bool = False
    deterministic: bool = False
    logical_negative: str = ""
    saved_hook: bool = False
    saved_strided: bool = False
    active: tuple[bool, bool] = (True, True)


def _inputs(case: DistogramCase, shape: tuple[int, int, int], device: torch.device):
    batch, tokens, classes = shape
    logits = torch.randn(batch, tokens, tokens, classes, device=device, dtype=case.dtype)
    labels = torch.randint(classes, (batch, tokens, tokens), device=device)
    mask_dtype = case.dtype if case.mask_kind == "float" else torch.float32
    mask = torch.rand(batch, tokens, device=device, dtype=mask_dtype)
    if case.mask_kind == "bool":
        mask = mask > 0.2
    if case.strided:
        logits = logits.transpose(1, 2)
        labels = labels.transpose(1, 2)
        mask = mask.T.contiguous().T
    if case.logical_negative == "logits":
        logits = torch._neg_view(logits)
    elif case.logical_negative == "mask":
        mask = torch._neg_view(mask)
    if case.compiled:
        for operand in (logits, labels, mask):
            for dim in range(operand.ndim):
                torch._dynamo.mark_dynamic(operand, dim)
    return logits, labels, mask


def _reference(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    reduction: str | None,
) -> torch.Tensor:
    errors = F.cross_entropy(logits.flatten(0, -2), labels.flatten(), reduction="none").reshape_as(labels)
    weights = mask.float()
    pairs = weights[:, None, :] * weights[:, :, None]
    pairs = pairs * ~torch.eye(mask.shape[-1], device=mask.device, dtype=torch.bool)
    losses = (errors * pairs).sum((-1, -2)) / (pairs.sum((-1, -2)) + 1e-5)
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    return losses


@contextmanager
def _operation(case: DistogramCase):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(case.deterministic)
        yield (
            torch.compile(distogram_cross_entropy, fullgraph=True, dynamic=True)
            if case.compiled
            else distogram_cross_entropy
        )
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.usefixtures("compiler_cache")
class TestDistogramCrossEntropy:

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(DistogramCase(torch.float16, "mean", ((2, 32, 16),), "bool", autocast=True), id="fp16-mean"),
            pytest.param(DistogramCase(torch.bfloat16, "sum", ((2, 32, 16),), "bool", autocast=True), id="bf16-sum"),
            pytest.param(
                DistogramCase(torch.bfloat16, None, ((2, 32, 16),), "bool", autocast=True), id="bf16-unreduced"
            ),
            pytest.param(DistogramCase(mask_kind="bool", strided=True), id="strided"),
            pytest.param(DistogramCase(shapes=((1, 257, 129),)), id="wide-classes"),
            pytest.param(DistogramCase(logical_negative="logits"), id="negative-logits"),
            pytest.param(DistogramCase(logical_negative="mask"), id="negative-mask"),
            pytest.param(DistogramCase(), id="fp32"),
            pytest.param(DistogramCase(torch.bfloat16), id="bf16"),
            pytest.param(DistogramCase(shapes=((1, 257, 256),), deterministic=True), id="wide-deterministic"),
            pytest.param(
                DistogramCase(shapes=((3, 48, 16),), mask_kind="bool", deterministic=True), id="deterministic-mean"
            ),
            pytest.param(
                DistogramCase(reduction=None, shapes=((3, 48, 16),), mask_kind="bool", deterministic=True),
                id="deterministic-unreduced",
            ),
            pytest.param(
                DistogramCase(shapes=((3, 48, 16),), mask_kind="bool", compiled=True, deterministic=True),
                id="compiled-deterministic-mean",
            ),
            pytest.param(
                DistogramCase(
                    reduction=None, shapes=((3, 48, 16),), mask_kind="bool", compiled=True, deterministic=True
                ),
                id="compiled-deterministic-unreduced",
            ),
        ],
    )
    def test_forward(self, cuda: torch.device, case: DistogramCase) -> None:
        torch.manual_seed(0)
        tolerance = 2e-5 if case.dtype is torch.float32 else 2e-2
        with _operation(case) as execute:
            for shape in case.shapes:
                logits, labels, mask = _inputs(case, shape, cuda)
                with torch.autocast("cuda", dtype=case.dtype, enabled=case.autocast):
                    actual = execute(logits, labels, token_mask=mask, reduction=case.reduction)
                    expected = _reference(logits, labels, mask, case.reduction)
                expected_dtype = torch.float32 if case.autocast else case.dtype
                assert actual.dtype is expected_dtype
                torch.testing.assert_close(actual, expected.to(expected_dtype), atol=tolerance, rtol=tolerance)
                if case.deterministic:
                    for _ in range(3):
                        repeated = execute(logits, labels, token_mask=mask, reduction=case.reduction)
                        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(DistogramCase(), id="mean"),
            pytest.param(DistogramCase(reduction="sum"), id="sum"),
            pytest.param(DistogramCase(shapes=((1, 257, 129),)), id="wide-classes"),
            pytest.param(DistogramCase(logical_negative="logits"), id="negative-logits"),
            pytest.param(DistogramCase(logical_negative="mask"), id="negative-mask"),
            pytest.param(DistogramCase(saved_hook=True), id="saved-hook"),
            pytest.param(DistogramCase(torch.bfloat16, saved_strided=True), id="saved-strided"),
            pytest.param(DistogramCase(shapes=((1, 257, 256),), deterministic=True), id="wide-deterministic"),
            pytest.param(DistogramCase(reduction=None), id="unreduced"),
            pytest.param(
                DistogramCase(torch.bfloat16, shapes=((256, 16, 16), (257, 17, 17)), compiled=True),
                id="compiled-dynamic-bf16",
            ),
            pytest.param(DistogramCase(mask_kind="bool", strided=True, active=(True, False)), id="strided-logits-only"),
            pytest.param(DistogramCase(reduction=None, strided=True, active=(False, True)), id="strided-mask-only"),
            pytest.param(
                DistogramCase(
                    shapes=((1, 65, 128), (2, 67, 129)), active=(False, True), compiled=True, deterministic=True
                ),
                id="wide-dynamic-mask-only",
            ),
            pytest.param(
                DistogramCase(torch.bfloat16, shapes=((1, 65, 128),), active=(False, True), autocast=True),
                id="wide-autocast-mask-only",
            ),
            pytest.param(DistogramCase(), marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
        ],
    )
    def test_backward(
        self, cuda: torch.device, case: DistogramCase, saved_strided_hook: Callable[[], AbstractContextManager[None]]
    ) -> None:
        torch.manual_seed(1)
        tolerance = 2e-5 if case.dtype is torch.float32 else 2e-2
        with _operation(case) as execute:
            for shape in case.shapes:
                logits, labels, mask = _inputs(case, shape, cuda)
                operands = (logits.requires_grad_(case.active[0]), mask.requires_grad_(case.active[1]))
                references = tuple(
                    operand.detach().clone().requires_grad_(active)
                    for operand, active in zip(operands, case.active, strict=True)
                )
                with torch.autocast("cuda", dtype=case.dtype, enabled=case.autocast):
                    with (
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
                    ):
                        actual = execute(logits, labels, token_mask=mask, reduction=case.reduction)
                    expected = _reference(references[0], labels, references[1], case.reduction)
                expected_dtype = torch.float32 if case.autocast else case.dtype
                assert actual.dtype is expected_dtype
                expected = expected.to(expected_dtype)
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                grad = torch.randn_like(actual)
                if case.logical_negative or case.saved_hook:
                    grad = torch._neg_view(grad)
                actual.backward(grad)
                expected.backward(grad)
                for operand, reference, active in zip(operands, references, case.active, strict=True):
                    if not active:
                        assert operand.grad is None
                        continue
                    assert operand.grad is not None and reference.grad is not None
                    assert operand.grad.dtype is operand.dtype
                    torch.testing.assert_close(operand.grad, reference.grad, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize(
        ("fault", "error"),
        [
            ("dual-logits", OOpsUnsupportedError),
            ("dual-mask", OOpsUnsupportedError),
            ("no-mask", OOpsUnsupportedError),
            ("pair-mask", OOpsUnsupportedError),
            ("reduction", ValueError),
        ],
    )
    def test_contract(self, fault: str, error: type[Exception]) -> None:
        logits, labels, mask = _inputs(DistogramCase(), (1, 4, 8), torch.device("cpu"))
        if fault.startswith("dual-"):
            with forward_ad.dual_level():
                if fault == "dual-logits":
                    logits = forward_ad.make_dual(logits, torch.ones_like(logits))
                else:
                    mask = forward_ad.make_dual(mask, torch.ones_like(mask))
                with pytest.raises(error, match="forward-mode AD"):
                    distogram_cross_entropy(logits, labels, token_mask=mask)
            return
        token_mask = mask if fault == "reduction" else None
        pair_mask = torch.ones_like(labels, dtype=torch.bool) if fault == "pair-mask" else None
        reduction = "median" if fault == "reduction" else "mean"
        with pytest.raises(error):
            distogram_cross_entropy(logits, labels, token_mask=token_mask, pair_mask=pair_mask, reduction=reduction)
