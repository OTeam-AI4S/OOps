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

from oops import OOpsUnsupportedError, cross_entropy


@dataclass(frozen=True)
class CrossEntropyCase:
    dtype: torch.dtype = torch.float32
    shapes: tuple[tuple[int, int], ...] = ((128, 32),)
    strided: bool = False
    compiled: bool = False
    deterministic: bool = False
    logical_negative: str = ""
    saved_hook: bool = False
    saved_strided: bool = False


def _inputs(case: CrossEntropyCase, shape: tuple[int, int], device: torch.device):
    rows, classes = shape
    logits = torch.randn((classes, rows) if case.strided else shape, device=device, dtype=case.dtype)
    if case.strided:
        logits = logits.T
    labels = torch.randint(classes, (rows * 2 if case.strided else rows,), device=device)
    if case.strided:
        labels = labels[::2]
    if case.logical_negative:
        logits = torch._neg_view(logits)
    if case.compiled:
        torch._dynamo.mark_dynamic(logits, 0)
        torch._dynamo.mark_dynamic(logits, 1)
        torch._dynamo.mark_dynamic(labels, 0)
    return logits, labels


@contextmanager
def _operation(case: CrossEntropyCase):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(case.deterministic)
        yield torch.compile(cross_entropy, fullgraph=True, dynamic=True) if case.compiled else cross_entropy
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.usefixtures("compiler_cache")
class TestCrossEntropy:

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(CrossEntropyCase(torch.float16), id="fp16-autocast"),
            pytest.param(CrossEntropyCase(torch.bfloat16), id="bf16-autocast"),
            pytest.param(CrossEntropyCase(), id="fp32"),
            pytest.param(CrossEntropyCase(logical_negative="logits"), id="logical-negative"),
            pytest.param(CrossEntropyCase(shapes=((33, 17),), strided=True), id="strided"),
        ],
    )
    def test_forward(
        self, cuda: torch.device, case: CrossEntropyCase, saved_strided_hook: Callable[[], AbstractContextManager[None]]
    ) -> None:
        torch.manual_seed(0)
        with _operation(case) as execute:
            for shape in case.shapes:
                logits, labels = _inputs(case, shape, cuda)
                with torch.autocast("cuda", dtype=case.dtype, enabled=case.dtype is not torch.float32):
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
                        actual = execute(logits, labels)
                    expected = F.cross_entropy(logits, labels, reduction="none")
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(CrossEntropyCase(shapes=((128, 64), (33, 17), (129, 255))), id="fp32"),
            pytest.param(CrossEntropyCase(torch.bfloat16, logical_negative="logits"), id="logical-negative"),
            pytest.param(CrossEntropyCase(torch.bfloat16, saved_hook=True), id="saved-hook"),
            pytest.param(CrossEntropyCase(torch.bfloat16, saved_strided=True), id="saved-strided"),
            pytest.param(CrossEntropyCase(shapes=((33, 17),), strided=True), id="strided"),
            pytest.param(
                CrossEntropyCase(torch.bfloat16, ((32, 16), (33, 17)), compiled=True, deterministic=True),
                id="compiled-dynamic-deterministic-autocast",
            ),
            pytest.param(CrossEntropyCase(shapes=((128, 64),)), marks=pytest.mark.cuda_device(1), id="secondary_cuda"),
        ],
    )
    def test_backward(
        self, cuda: torch.device, case: CrossEntropyCase, saved_strided_hook: Callable[[], AbstractContextManager[None]]
    ) -> None:
        torch.manual_seed(1)
        tolerance = 1e-5 if case.dtype is torch.float32 else 2e-2
        with _operation(case) as execute:
            for shape in case.shapes:
                logits, labels = _inputs(case, shape, cuda)
                logits.requires_grad_()
                reference = logits.detach().clone().requires_grad_()
                with torch.autocast("cuda", dtype=case.dtype, enabled=case.dtype is not torch.float32):
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
                        actual = execute(logits, labels)
                    expected = F.cross_entropy(reference, labels, reduction="none")
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
                grad = torch.randn_like(actual)
                if case.logical_negative or case.saved_hook:
                    grad = torch._neg_view(grad)
                torch.testing.assert_close(
                    torch.autograd.grad(actual, logits, grad)[0],
                    torch.autograd.grad(expected, reference, grad)[0],
                    atol=tolerance,
                    rtol=tolerance,
                )

    @pytest.mark.parametrize(
        ("fault", "error"),
        [
            ("forward-ad", OOpsUnsupportedError),
            ("cpu", OOpsUnsupportedError),
            ("shape", ValueError),
            ("probability-labels", OOpsUnsupportedError),
        ],
    )
    def test_contract(self, fault: str, error: type[Exception]) -> None:
        logits, labels = _inputs(CrossEntropyCase(), (8, 4), torch.device("cpu"))
        if fault == "forward-ad":
            with forward_ad.dual_level():
                logits = forward_ad.make_dual(logits, torch.ones_like(logits))
                with pytest.raises(error, match="forward-mode AD"):
                    cross_entropy(logits, labels)
            return
        if fault == "shape":
            labels = labels[:-1]
        elif fault == "probability-labels":
            labels = torch.randn_like(logits)
        with pytest.raises(error):
            cross_entropy(logits, labels)
