from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch
from danling.tensors import NestedTensor
from torch import Tensor, nn
from torch.autograd import forward_ad
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oops import LayerNorm, OOpsUnsupportedError, layer_norm

_TOLERANCE = {torch.float32: 1e-4, torch.float16: 3e-3, torch.bfloat16: 2e-2}
_GRAD_TOLERANCE = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 4e-2}


@dataclass(frozen=True)
class Case:
    dtype: torch.dtype = torch.float32
    layout: str = "dense"
    affine: str = "both"
    master: bool = False
    execution: str = "eager"
    demand: tuple[bool, bool, bool] = (True, True, True)
    autocast: bool = False
    width: int = 32
    rows: int = 3
    logical_negative: bool = False
    saved_hook: bool = False
    saved_strided: bool = False


_FORWARD_CASES = [
    pytest.param(Case(), id="dense-fp32"),
    pytest.param(Case(width=128, rows=67, logical_negative=True), id="logical-negative"),
    pytest.param(Case(torch.float16, affine="weight"), id="dense-fp16-weight"),
    pytest.param(Case(torch.bfloat16, affine="none"), id="dense-bf16-plain"),
    pytest.param(Case(torch.float16, master=True), id="dense-fp16-master"),
    pytest.param(Case(torch.bfloat16, "transposed", master=True), id="transposed"),
    pytest.param(Case(torch.bfloat16, "sliced", master=True), id="sliced"),
    pytest.param(Case(torch.bfloat16, "expanded", master=True), id="expanded"),
    pytest.param(Case(layout="nested"), id="nested-fp32"),
    pytest.param(Case(torch.float16, "nonleading"), id="nested-fp16-nonleading"),
    pytest.param(Case(torch.bfloat16, "batch-last"), id="nested-bf16-batch-last"),
    pytest.param(Case(torch.float16, "nested", master=True, autocast=True), id="nested-fp16-autocast"),
    pytest.param(Case(torch.bfloat16, "nested", master=True, autocast=True), id="nested-bf16-autocast"),
    pytest.param(Case(torch.bfloat16, "mixed-empty"), id="nested-mixed-empty"),
    pytest.param(Case(affine="none", execution="dynamic"), id="dynamic-optional"),
]
_BACKWARD_CASES = [
    pytest.param(Case(), marks=pytest.mark.cuda_device(1), id="secondary-cuda"),
    pytest.param(Case(width=8192, rows=1025), id="wide-fp32"),
    pytest.param(
        Case(torch.bfloat16, "nested", master=True, execution="dynamic", width=4097, rows=257),
        id="wide-nested-bf16-dynamic",
    ),
    pytest.param(Case(), id="dense-fp32"),
    pytest.param(Case(width=128, rows=67, logical_negative=True), id="logical-negative"),
    pytest.param(Case(width=128, rows=67, saved_hook=True), id="saved-hook"),
    pytest.param(Case(width=128, rows=67, saved_strided=True), id="saved-strided"),
    pytest.param(Case(width=32, rows=3, saved_strided=True), id="saved-strided-small"),
    pytest.param(Case(torch.float16), id="dense-fp16"),
    pytest.param(Case(torch.bfloat16), id="dense-bf16"),
    pytest.param(Case(affine="weight"), id="weight-affine"),
    pytest.param(Case(affine="bias"), id="bias-affine"),
    pytest.param(Case(affine="none", execution="dynamic"), id="dynamic-no-affine"),
    pytest.param(Case(demand=(True, False, False)), id="input-only"),
    pytest.param(Case(torch.float16, demand=(False, True, False)), id="weight-only"),
    pytest.param(Case(torch.bfloat16, demand=(False, False, True)), id="bias-only"),
    pytest.param(
        Case(torch.bfloat16, affine="weight", master=True, execution="dynamic", demand=(False, True, False)),
        id="compiled-frozen-input-weight",
    ),
    pytest.param(
        Case(torch.bfloat16, affine="bias", master=True, execution="dynamic", demand=(False, False, True)),
        id="compiled-frozen-input-bias",
    ),
    pytest.param(Case(torch.bfloat16, master=True, execution="dynamic"), id="dynamic-master"),
    pytest.param(Case(execution="checkpoint"), id="dense-checkpoint"),
    pytest.param(Case(torch.bfloat16, execution="unbacked"), id="unbacked-rows"),
    pytest.param(Case(torch.bfloat16, "transposed", master=True), id="transposed"),
    pytest.param(Case(torch.bfloat16, "sliced", master=True), id="sliced"),
    pytest.param(Case(torch.bfloat16, "expanded", master=True), id="expanded"),
    pytest.param(Case(layout="nested"), id="nested-fp32"),
    pytest.param(Case(torch.float16, "nonleading"), id="nested-fp16-nonleading"),
    pytest.param(Case(torch.bfloat16, "batch-last"), id="nested-bf16-batch-last"),
    pytest.param(Case(torch.bfloat16, "nested", demand=(True, False, False)), id="nested-input-only"),
    pytest.param(Case(torch.bfloat16, "nested", demand=(False, True, False)), id="nested-weight-only"),
    pytest.param(Case(torch.bfloat16, "nested", demand=(False, False, True)), id="nested-bias-only"),
    pytest.param(Case(torch.float16, "nested", master=True, autocast=True), id="nested-fp16-autocast"),
    pytest.param(Case(torch.bfloat16, "nested", master=True, autocast=True), id="nested-bf16-autocast"),
    pytest.param(Case(torch.bfloat16, "nonleading", execution="dynamic"), id="nested-dynamic"),
    pytest.param(Case(torch.bfloat16, "nested", execution="checkpoint"), id="nested-checkpoint"),
    pytest.param(Case(torch.bfloat16, "mixed-empty"), id="nested-mixed-empty"),
]


@torch.library.custom_op("oops_test::_unbacked_row_prefix", mutates_args=())
def _unbacked_row_prefix(values: Tensor, lengths: Tensor) -> Tensor:
    return values[: int(lengths.sum())].clone()


@torch.library.register_fake(_unbacked_row_prefix)
def _unbacked_row_prefix_fake(values: Tensor, lengths: Tensor) -> Tensor:
    del lengths
    rows = torch.library.get_ctx().new_dynamic_size(min=0)
    return values.new_empty((rows, values.shape[-1]))


def _setup_prefix_context(ctx, inputs, output) -> None:
    del output
    ctx.input_rows = inputs[0].shape[0]


def _prefix_backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
    return F.pad(grad_output, (0, 0, 0, ctx.input_rows - grad_output.shape[0])), None


torch.library.register_autograd(_unbacked_row_prefix, _prefix_backward, setup_context=_setup_prefix_context)


def _make_inputs(case: Case, device: torch.device, rows: int, grad: bool):
    if case.layout in ("nested", "nonleading", "batch-last", "mixed-empty", "all-empty"):
        shapes: tuple[tuple[int, ...], ...] = ((2, rows, case.width), (4, rows + 2, case.width))
        if case.layout == "nonleading":
            shapes = ((2, rows, case.width), (2, rows + 2, case.width))
        elif case.layout == "mixed-empty":
            shapes = ((0, 3, case.width), (2, 0, case.width), (1, 4, case.width))
        elif case.layout == "all-empty":
            shapes = ((0, 3, case.width), (2, 0, case.width))
        template = NestedTensor(
            [torch.empty(shape) for shape in shapes],
            ragged_dims=(1,) if case.layout == "nonleading" else (0, 1),
            batch_first=case.layout != "batch-last",
        )
        values = torch.randn(template.concat.shape, device=device, dtype=case.dtype)
    else:
        template = None
        values = torch.randn(16 if case.execution == "unbacked" else rows, case.width, device=device, dtype=case.dtype)
        if case.layout in ("transposed", "sliced", "expanded"):
            values = torch.randn(1, 7, 9, case.width, device=device, dtype=case.dtype)
            if case.layout == "transposed":
                values = values.transpose(-3, -2)
            elif case.layout == "sliced":
                values = values[:, :5]
            else:
                values = values.expand(4, -1, -1, -1)
        values = values.detach()
    parameter_dtype = torch.float32 if case.master else case.dtype
    weight = (
        torch.randn(case.width, device=device, dtype=parameter_dtype) if case.affine in ("both", "weight") else None
    )
    bias = torch.randn(case.width, device=device, dtype=parameter_dtype) if case.affine in ("both", "bias") else None
    if case.logical_negative:
        values = torch._neg_view(values)
        weight = None if weight is None else torch._neg_view(weight)
        bias = None if bias is None else torch._neg_view(bias)
    leaves = (values, weight, bias)
    for value, requested in zip(leaves, case.demand, strict=True):
        if value is not None:
            value.requires_grad_(grad and requested)
    reference_leaves = tuple(
        None if value is None else value.detach().clone().requires_grad_(value.requires_grad) for value in leaves
    )
    input = values if template is None else template.packed_like(values)
    reference_input = reference_leaves[0] if template is None else template.packed_like(reference_leaves[0])
    return (input, weight, bias), (reference_input, *reference_leaves[1:]), leaves, reference_leaves


def _reference(input, weight, bias):
    def dense(value):
        return F.layer_norm(
            value.float(),
            (value.shape[-1],),
            None if weight is None else weight.float(),
            None if bias is None else bias.float(),
        ).to(value.dtype)

    if not isinstance(input, NestedTensor):
        return dense(input)
    outputs = [dense(element) for element in torch.unbind(input, dim=0 if input.batch_first else 1)]
    return NestedTensor(outputs, ragged_dims=input.ragged_dims, batch_first=input.batch_first).concat


def _runs(case: Case, cuda: torch.device, grad: bool, saved_strided_hook=None):
    torch.manual_seed(40)
    compiled = case.execution != "eager"

    def body(input, weight, bias, lengths):
        if case.execution == "unbacked":
            input = _unbacked_row_prefix(input, lengths)
        return layer_norm(input, case.width, weight, bias)

    def apply(*operands):
        if case.execution == "checkpoint":
            return checkpoint(body, *operands, use_reentrant=False)
        return body(*operands)

    function = torch.compile(apply, fullgraph=True, dynamic=True) if compiled else apply
    for rows in ((case.rows, case.rows + 2) if compiled else (case.rows,)):
        operands, reference_operands, leaves, reference_leaves = _make_inputs(case, cuda, rows, grad)
        lengths = torch.tensor([rows, rows + 1], device=cuda)
        hooks = (
            saved_strided_hook()
            if case.saved_strided
            else (
                torch.autograd.graph.saved_tensors_hooks(lambda tensor: -tensor, torch._neg_view)
                if case.saved_hook
                else nullcontext()
            )
        )
        with hooks, torch.autocast("cuda", dtype=case.dtype, enabled=case.autocast):
            output = function(*operands, lengths)
        if case.execution == "unbacked":
            reference_operands = (reference_operands[0][: 2 * rows + 1], *reference_operands[1:])
        expected = _reference(*reference_operands)
        if isinstance(operands[0], NestedTensor):
            assert isinstance(output, NestedTensor)
            assert output.ragged_dims == operands[0].ragged_dims
            assert output.packed_dim_order == operands[0].packed_dim_order
            assert output.batch_first == operands[0].batch_first
            batch_dim = 0 if output.batch_first else 1
            assert [tuple(x.shape) for x in torch.unbind(output, dim=batch_dim)] == [
                tuple(x.shape) for x in torch.unbind(operands[0], dim=batch_dim)
            ]
            output = output.concat
        assert output.shape == expected.shape
        assert output.dtype is case.dtype
        torch.testing.assert_close(output, expected, atol=_TOLERANCE[case.dtype], rtol=_TOLERANCE[case.dtype])
        yield output, expected, leaves, reference_leaves


@pytest.mark.usefixtures("compiler_cache")
class TestLayerNorm:

    @pytest.mark.parametrize("case", _FORWARD_CASES)
    def test_forward(self, cuda: torch.device, case: Case) -> None:
        for output, _expected, _leaves, _reference_leaves in _runs(case, cuda, grad=False):
            assert not output.requires_grad

    @pytest.mark.parametrize("case", _BACKWARD_CASES)
    def test_backward(self, cuda: torch.device, case: Case, saved_strided_hook) -> None:
        for output, expected, leaves, reference_leaves in _runs(
            case, cuda, grad=True, saved_strided_hook=saved_strided_hook
        ):
            grad_output = torch.randn_like(output)
            if case.logical_negative or case.saved_hook:
                grad_output = torch._neg_view(grad_output)
            output.backward(grad_output)
            expected.backward(grad_output)
            for leaf, reference in zip(leaves, reference_leaves, strict=True):
                if leaf is None:
                    continue
                if not leaf.requires_grad:
                    assert leaf.grad is None
                    continue
                assert leaf.grad is not None and reference.grad is not None
                assert leaf.grad.dtype is leaf.dtype
                assert leaf.grad.shape == leaf.shape
                torch.testing.assert_close(
                    leaf.grad, reference.grad, atol=_GRAD_TOLERANCE[case.dtype], rtol=_GRAD_TOLERANCE[case.dtype]
                )

    @pytest.mark.parametrize(
        "contract",
        [
            "module-cpu",
            "module-empty",
            "forward-ad",
            "nested-forward-ad",
            "nested-empty",
            "ragged-channel",
            "packed-stride",
            "nested-affine",
            "normalized-shape",
        ],
    )
    def test_contract(self, request: pytest.FixtureRequest, contract: str) -> None:
        if contract.startswith("module-"):
            device = request.getfixturevalue("cuda") if contract == "module-empty" else torch.device("cpu")
            module, reference = LayerNorm(16).to(device), nn.LayerNorm(16).to(device)
            module.load_state_dict(reference.state_dict())
            input = torch.randn(0 if contract == "module-empty" else 4, 16, device=device, requires_grad=True)
            reference_input = input.detach().clone().requires_grad_()
            with pytest.raises(OOpsUnsupportedError):
                layer_norm(input, 16, module.weight, module.bias)
            output, expected = module(input), reference(reference_input)
            torch.testing.assert_close(output, expected)
            output.sum().backward()
            expected.sum().backward()
            for actual_leaf, reference_leaf in zip(
                (input, module.weight, module.bias), (reference_input, reference.weight, reference.bias), strict=True
            ):
                torch.testing.assert_close(actual_leaf.grad, reference_leaf.grad)
            return
        case = Case(torch.bfloat16, "all-empty" if contract == "nested-empty" else "nested")
        device = torch.device("cpu") if contract == "forward-ad" else request.getfixturevalue("cuda")
        (input, weight, bias), _, leaves, _ = _make_inputs(case, device, 3, False)
        error: type[Exception] = OOpsUnsupportedError
        match: str | None = None
        normalized_shape: int | tuple[int, ...] = 32
        if contract in ("forward-ad", "nested-forward-ad"):
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(leaves[0], torch.ones_like(leaves[0]))
                argument = input.packed_like(dual) if contract == "nested-forward-ad" else dual
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    layer_norm(argument, 32, weight, bias)
            return
        if contract == "ragged-channel":
            input = NestedTensor([torch.empty(3), torch.empty(5)], ragged_dims=(0,)).packed_like(
                torch.empty(8, device=device)
            )
            normalized_shape, weight, bias = 1, None, None
            match = "final logical element dimension"
        elif contract == "packed-stride":
            rows, width = input.concat.shape
            input = input.packed_like(torch.empty(width, rows, device=device, dtype=case.dtype).T)
            match = "contiguous NestedTensor packed"
        elif contract == "nested-affine":
            weight = NestedTensor([torch.empty(32), torch.empty(32)], ragged_dims=(0,))
            error, match = ValueError, "dense affine parameters"
        elif contract == "normalized-shape":
            normalized_shape, match = (1, 32), "final dimension only"
        with pytest.raises(error, match=match):
            layer_norm(input, normalized_shape, weight, bias)
