from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.autograd import forward_ad

from oops import CSRGraph, OOpsUnsupportedError, csr_weighted_sum


def _case(cuda, dtype, layout, edges=137, skew=False, gaps=False, single=False, width=(3, 65)):
    torch.manual_seed(8)
    lengths = (1,) if single else (8193, *((1,) * 511)) if skew else (0, 1, 36, edges - 37)
    prefix = 4 if gaps else 0
    edges = sum(lengths) + (10 if gaps else 0)
    values = torch.randn(edges, *width, device=cuda, dtype=dtype)
    shape = {
        "scalar": (edges,),
        "head": (edges, width[0]),
        "broadcast": (edges, 1),
        "elementwise": values.shape,
    }[layout]
    weights = torch.randn(shape, device=cuda, dtype=dtype)
    if gaps:
        for operand in (values, weights):
            operand[:prefix] = torch.nan
            operand[prefix + sum(lengths) :] = torch.nan
    offsets = torch.tensor([prefix, *(prefix + torch.tensor(lengths).cumsum(0)).tolist()], device=cuda)
    return values, weights, offsets, lengths


def _reference(values, weights, lengths, gaps=False):
    values, weights = (operand.narrow(0, 4 if gaps else 0, sum(lengths)) for operand in (values, weights))
    aligned = weights.reshape(*weights.shape, *((1,) * (values.ndim - weights.ndim)))
    products = values * aligned
    return torch.stack([part.sum(0) for part in products.split(lengths)])


class TestCSRWeightedSum:

    @pytest.mark.parametrize(
        ("dtype", "layout", "metadata", "mode", "skew"),
        [
            (torch.float16, "scalar", "offsets", "eager", False),
            (torch.bfloat16, "head", "graph", "eager", False),
            (torch.float32, "broadcast", "offsets", "compile", False),
            (torch.float64, "elementwise", "segments", "eager", False),
            (torch.float32, "scalar", "offsets", "eager", True),
            (torch.float32, "scalar", "offsets", "negative-values", True),
            (torch.float32, "head", "offsets", "negative-weights", False),
            (torch.float32, "head", "offsets", "eager", True),
            (torch.bfloat16, "elementwise", "offsets", "eager", True),
        ],
    )
    def test_forward(self, cuda, dtype, layout, metadata, mode, skew, tolerance):
        operation = (
            torch.compile(csr_weighted_sum, fullgraph=True, dynamic=True) if mode == "compile" else csr_weighted_sum
        )
        for edges in (137, 151) if mode == "compile" else (137,):
            values, weights, offsets, lengths = _case(cuda, dtype, layout, edges, skew=skew)
            if mode == "negative-values":
                values = torch._neg_view(values)
            elif mode == "negative-weights":
                weights = torch._neg_view(weights)
            ids = torch.repeat_interleave(torch.arange(len(lengths), device=cuda), offsets.diff())
            descriptor = CSRGraph.from_row_offsets(offsets, segment_ids=ids) if metadata == "graph" else offsets
            kwargs = {"segment_ids": ids.int()} if metadata == "segments" else {}
            actual = operation(values, weights, descriptor, **kwargs)
            expected = _reference(values.double(), weights.double(), lengths)
            torch.testing.assert_close(actual.double(), expected, **tolerance(dtype))

    @pytest.mark.parametrize(
        ("dtype", "layout", "wanted", "mode", "skew", "gaps", "width"),
        [
            (torch.float16, "scalar", (0,), "eager", False, False, (3, 65)),
            (torch.float32, "scalar", (0, 1), "single", False, False, (3, 65)),
            (torch.float16, "scalar", (0, 1), "higher_order_half", False, False, (3, 65)),
            (torch.bfloat16, "head", (1,), "eager", False, False, (3, 65)),
            (torch.float32, "broadcast", (0, 1), "compile", False, False, (3, 65)),
            (torch.float64, "elementwise", (0, 1), "eager", False, False, (3, 65)),
            (torch.float64, "scalar", (0, 1), "higher_order", False, True, (3, 65)),
            (torch.float32, "scalar", (0, 1), "eager", True, False, (3, 65)),
            (torch.float32, "scalar", (0, 1), "negative-values", True, False, (3, 65)),
            (torch.float32, "elementwise", (0, 1), "negative-weights", True, False, (4, 17)),
            (torch.float32, "head", (0, 1), "saved-hook", False, False, (3, 65)),
            (torch.float32, "head", (0, 1), "saved-strided", False, False, (3, 65)),
            (torch.float32, "elementwise", (0, 1), "compile", True, False, (3, 65)),
            (torch.float32, "scalar", (0, 1), "eager", True, True, (3, 65)),
            (torch.float64, "head", (1,), "eager", False, True, (3, 65)),
            (torch.float16, "head", (0, 1), "eager", False, True, (4, 16)),
            (torch.bfloat16, "broadcast", (0,), "compile", True, False, (4, 17)),
            (torch.float32, "elementwise", (1,), "eager", True, True, (4, 16)),
            (torch.float64, "scalar", (0, 1), "higher_order", False, True, (4, 17)),
            pytest.param(
                torch.float16,
                "scalar",
                (0,),
                "eager",
                False,
                False,
                (3, 65),
                marks=pytest.mark.cuda_device(1),
                id="secondary_cuda",
            ),
        ],
    )
    def test_backward(self, cuda, dtype, layout, wanted, mode, skew, gaps, width, tolerance, saved_strided_hook):
        values, weights, offsets, lengths = _case(
            cuda, dtype, layout, skew=skew, gaps=gaps, single=mode == "single", width=width
        )
        if mode == "negative-values":
            values = torch._neg_view(values)
        elif mode == "negative-weights":
            weights = torch._neg_view(weights)
        if mode == "higher_order_half":
            values.zero_()
            values[0].flatten()[:3] = torch.tensor([300.0, -300.0, 1.0], device=cuda, dtype=dtype)
            weights.fill_(0.0001)
        operands = (values, weights)
        for index, operand in enumerate(operands):
            operand.requires_grad_(index in wanted)
        references = tuple(
            operand.detach().double().requires_grad_(index in wanted) for index, operand in enumerate(operands)
        )
        operation = torch.compile(csr_weighted_sum, fullgraph=True) if mode == "compile" else csr_weighted_sum
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
            actual = operation(values, weights, offsets)
        expected = _reference(*references, lengths, gaps=gaps)
        upstream = torch.randn_like(actual)
        if mode in ("negative-values", "negative-weights", "saved-hook"):
            upstream = torch._neg_view(upstream)
        if mode == "higher_order_half":
            upstream.zero_()
            upstream[1].flatten()[:3] = torch.tensor([300.0, 300.0, 1.0], device=cuda, dtype=dtype)
        upstream.requires_grad_(mode.startswith("higher_order") and gaps)
        inputs, ref_inputs = tuple(operands[i] for i in wanted), tuple(references[i] for i in wanted)
        gradients = torch.autograd.grad(actual, inputs, upstream, create_graph=mode.startswith("higher_order"))
        ref_gradients = torch.autograd.grad(
            expected, ref_inputs, upstream.double(), create_graph=mode.startswith("higher_order")
        )
        for result, reference in zip(gradients, ref_gradients, strict=True):
            torch.testing.assert_close(result.double(), reference, **tolerance(dtype))
        if mode.startswith("higher_order") and gaps:
            directions = tuple(torch.randn_like(operands[index]) for index in wanted)
            second = torch.autograd.grad(gradients, upstream, directions, retain_graph=True)[0]
            ref_second = torch.autograd.grad(
                ref_gradients, upstream, tuple(direction.double() for direction in directions), retain_graph=True
            )[0]
            torch.testing.assert_close(second.double(), ref_second, **tolerance(dtype))
        if mode.startswith("higher_order"):
            direction = torch.randn_like(values)
            mixed = torch.autograd.grad(gradients[0], weights, direction)[0]
            ref_mixed = torch.autograd.grad(ref_gradients[0], references[1], direction.double())[0]
            torch.testing.assert_close(mixed.double(), ref_mixed, **tolerance(dtype))

    @pytest.mark.parametrize("case", ["cpu", "weight_shape", "forward_ad"])
    def test_contract(self, case):
        values, weights = torch.randn(5, 3, 2), torch.randn(5, 4) if case == "weight_shape" else torch.randn(5)
        offsets = torch.tensor([0, 2, 5])
        if case == "forward_ad":
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(values, torch.ones_like(values))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    csr_weighted_sum(dual, weights, offsets)
        else:
            with pytest.raises(ValueError if case == "weight_shape" else OOpsUnsupportedError):
                csr_weighted_sum(values, weights, offsets)
