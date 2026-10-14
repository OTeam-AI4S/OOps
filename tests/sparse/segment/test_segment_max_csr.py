from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.autograd import forward_ad

from oops import CSRGraph, OOpsUnsupportedError, segment_max_csr


def _case(cuda, dtype, lengths, tail):
    torch.manual_seed(6)
    values = torch.randn(sum(lengths), *tail, device=cuda, dtype=dtype)
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=cuda)
    return values, offsets


def _reference(values, lengths):
    rows = [
        part.amax(0) if part.shape[0] else values.new_full(values.shape[1:], -float("inf"))
        for part in values.split(lengths)
    ]
    rows += [values.new_full(values.shape[1:], -float("inf")) for _ in range(2)]
    return torch.stack(rows)


class TestSegmentMaxCSR:

    @pytest.mark.parametrize(
        ("dtype", "lengths", "tail", "metadata", "mode"),
        [
            (torch.float16, (0, 1, 17, 129, 2049), (3,), "offsets", "eager"),
            (torch.bfloat16, (0, 8193, 4099), (65,), "offsets", "eager"),
            (torch.float32, (0, 3, 0, 8), (2, 3), "offsets", "compile"),
            (torch.float64, (0, 2, 5), (3,), "segments", "eager"),
            (torch.float32, (2, 0, 3), (4,), "graph", "eager"),
            (torch.float32, (8193, *((1,) * 511)), (3,), "offsets", "eager"),
            (torch.float32, (0, 17, 8193), (17,), "offsets", "negative"),
        ],
    )
    def test_forward(self, cuda, dtype, lengths, tail, metadata, mode, tolerance, saved_strided_hook):
        operation = (
            torch.compile(segment_max_csr, fullgraph=True, dynamic=True) if mode == "compile" else segment_max_csr
        )
        for current in (lengths, (1, 0, 9)) if mode == "compile" else (lengths,):
            lengths = current
            values, offsets = _case(cuda, dtype, lengths, tail)
            if mode == "negative":
                values = torch._neg_view(values)
            segment_ids = torch.repeat_interleave(
                torch.arange(len(lengths), device=cuda), torch.tensor(lengths, device=cuda)
            )
            descriptor = CSRGraph.from_row_offsets(offsets, segment_ids=segment_ids) if metadata == "graph" else offsets
            kwargs = {"segment_ids": segment_ids.int()} if metadata == "segments" else {}
            actual = operation(values, descriptor, num_segments=len(lengths) + 2, **kwargs)
            expected = _reference(values.double(), lengths)
            torch.testing.assert_close(actual.double(), expected, **tolerance(dtype))
            repeated = operation(values, descriptor, num_segments=len(lengths) + 2, **kwargs)
            torch.testing.assert_close(actual, repeated, atol=0, rtol=0)

    @pytest.mark.parametrize(
        ("dtype", "lengths", "tail", "mode"),
        [
            (torch.float16, (0, 1, 17, 129, 2049), (3,), "eager"),
            (torch.bfloat16, (0, 8193, 4099), (65,), "ties"),
            (torch.float32, (0, 3, 0, 8), (2, 3), "compile"),
            (torch.float32, (8193, *((1,) * 511)), (3,), "eager"),
            (torch.float32, (0, 17, 8193), (17,), "negative"),
            (torch.float32, (0, 17, 8193), (17,), "saved-hook"),
            (torch.float32, (0, 17, 8193), (17,), "saved-strided"),
            (torch.float32, (1, 0, 8193, *((1,) * 511)), (65,), "ties"),
            (torch.float64, (1, 6, 10), (3,), "ties"),
            (torch.float64, (8193, *((1,) * 31)), (17,), "ties"),
            pytest.param(
                torch.float16,
                (0, 1, 17, 129, 2049),
                (3,),
                "eager",
                marks=pytest.mark.cuda_device(1),
                id="secondary_cuda",
            ),
        ],
    )
    def test_backward(self, cuda, dtype, lengths, tail, mode, tolerance, saved_strided_hook):
        values, offsets = _case(cuda, dtype, lengths, tail)
        if mode == "negative":
            values = torch._neg_view(values)
        values = values.round().requires_grad_() if mode == "ties" else values
        values.requires_grad_()
        reference_values = values.detach().double().requires_grad_()
        operation = (
            torch.compile(segment_max_csr, fullgraph=True, dynamic=True) if mode == "compile" else segment_max_csr
        )
        hooks = (
            saved_strided_hook()
            if mode == "saved-strided"
            else (
                torch.autograd.graph.saved_tensors_hooks(
                    lambda tensor: -tensor if tensor.is_floating_point() else tensor,
                    lambda tensor: torch._neg_view(tensor) if tensor.is_floating_point() else tensor,
                )
                if mode == "saved-hook"
                else nullcontext()
            )
        )
        with hooks:
            output = operation(values, offsets, num_segments=len(lengths) + 2)
        reference = _reference(reference_values, lengths)
        upstream = torch.randn_like(output)
        if mode in ("negative", "saved-hook"):
            upstream = torch._neg_view(upstream)
        higher = mode == "higher_order"
        actual = torch.autograd.grad(output, values, upstream, create_graph=higher)[0]
        expected = torch.autograd.grad(reference, reference_values, upstream.double(), create_graph=higher)[0]
        torch.testing.assert_close(actual.double(), expected, **tolerance(dtype))

    @pytest.mark.parametrize("case", ["cpu", "integer", "empty", "forward_ad"])
    def test_contract(self, request, case):
        device = "cpu" if case in ("cpu", "forward_ad") else request.getfixturevalue("cuda")
        values = torch.ones(
            0 if case == "empty" else 4, 2, device=device, dtype=torch.int64 if case == "integer" else torch.float32
        )
        offsets = torch.tensor([0, values.shape[0]], device=device)
        if case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(values, torch.ones_like(values))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    segment_max_csr(dual, offsets)
        else:
            with pytest.raises(OOpsUnsupportedError):
                segment_max_csr(values, offsets)
