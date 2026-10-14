from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor
from torch.autograd import forward_ad
from torch.utils.checkpoint import checkpoint

from oops import AtomTokenLayout, OOpsUnsupportedError, token_to_atom_broadcast


def _nested_case(
    token_lengths: tuple[int, ...],
    mappings: tuple[tuple[int, ...], ...],
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    channels: int = 3,
    requires_grad: bool = False,
) -> tuple[NestedTensor, NestedTensor, torch.Tensor, torch.Tensor]:
    token_reference = NestedTensor([torch.empty(1, channels) for _ in token_lengths], ragged_dims=(0,))
    token_values = torch.randn((sum(token_lengths), channels), dtype=dtype, device=device)
    token_values.requires_grad_(requires_grad)
    token = token_reference.packed_with_lengths(token_values, torch.tensor(token_lengths, dtype=torch.long))

    mapping_reference = NestedTensor([torch.empty(1, dtype=torch.long) for _ in mappings], ragged_dims=(0,))
    if mappings:
        mapping_values = torch.tensor([index for mapping in mappings for index in mapping], device=device)
    else:
        mapping_values = torch.empty(0, dtype=torch.long, device=device)
    atom_to_token = mapping_reference.packed_with_lengths(
        mapping_values,
        torch.tensor(tuple(map(len, mappings)), dtype=torch.long),
    )
    return token, atom_to_token, token_values, mapping_values


def _nested_reference(
    token_values: torch.Tensor,
    mapping_values: torch.Tensor,
    token_lengths: tuple[int, ...],
    atom_lengths: tuple[int, ...],
) -> torch.Tensor:
    token_parts = token_values.split(token_lengths)
    mapping_parts = mapping_values.split(atom_lengths)
    outputs = tuple(
        tokens.index_select(0, mapping.clamp(0, max(tokens.shape[0] - 1, 0)))
        for tokens, mapping in zip(token_parts, mapping_parts)
    )
    return torch.cat(outputs) if outputs else token_values.new_empty((0, token_values.shape[-1]))


def _sampled_nested_case(
    token_lengths: tuple[int, ...],
    mappings: tuple[tuple[int, ...], ...],
    *,
    samples: int = 3,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    channels: int = 4,
    requires_grad: bool = False,
) -> tuple[NestedTensor, NestedTensor, torch.Tensor, torch.Tensor]:
    token_reference = NestedTensor(
        [torch.empty(samples, length, channels) for length in token_lengths],
        ragged_dims=(1,),
    )
    token_values = torch.randn((sum(token_lengths), samples, channels), dtype=dtype, device=device)
    token_values.requires_grad_(requires_grad)
    token = token_reference.packed_like(token_values)

    mapping_reference = NestedTensor([torch.empty(1, dtype=torch.long) for _ in mappings], ragged_dims=(0,))
    if mappings:
        mapping_values = torch.tensor([index for mapping in mappings for index in mapping], device=device)
    else:
        mapping_values = torch.empty(0, dtype=torch.long, device=device)
    atom_to_token = mapping_reference.packed_with_lengths(
        mapping_values,
        torch.tensor(tuple(map(len, mappings)), dtype=torch.long),
    )
    return token, atom_to_token, token_values, mapping_values


def _sampled_nested_reference(
    token_values: torch.Tensor,
    mapping_values: torch.Tensor,
    token_lengths: tuple[int, ...],
    atom_lengths: tuple[int, ...],
) -> torch.Tensor:
    token_parts = token_values.split(token_lengths)
    mapping_parts = mapping_values.split(atom_lengths)
    outputs = tuple(
        tokens.index_select(0, mapping.clamp(0, max(tokens.shape[0] - 1, 0)))
        for tokens, mapping in zip(token_parts, mapping_parts)
    )
    if outputs:
        return torch.cat(outputs)
    return token_values.new_empty((0, token_values.shape[1], token_values.shape[2]))


def _case(kind, dtype, device, training, variant=0):
    lengths = (2, 5, 3) if variant == 0 else (3, 4, 2)
    mappings = ((1, 0, 1), (2, 1), (0, 1, 0, 1))
    if kind == "clamped":
        mappings = ((-3, 9, 0), (100, -1), (0, 9, -2, 1))
    elif kind == "empty":
        lengths, mappings = (0, 3), ((), (2, 0))
    sampled = kind == "sampled"
    factory = _sampled_nested_case if sampled else _nested_case
    tokens, mapping, values, indices = factory(lengths, mappings, dtype=dtype, device=device, requires_grad=training)
    reference = values.detach().double().requires_grad_(training)
    atom_lengths = tuple(map(len, mappings))
    oracle = _sampled_nested_reference if sampled else _nested_reference
    return tokens, mapping, values, reference, lambda tensor: oracle(tensor, indices, lengths, atom_lengths)


def _run(tokens, mapping):
    result = token_to_atom_broadcast(tokens, mapping)
    return result.concat if isinstance(result, NestedTensor) else result


class TestTokenToAtomBroadcast:

    @pytest.mark.parametrize(
        ("kind", "dtype", "mode"),
        [
            ("nested", torch.float32, "eager"),
            ("sampled", torch.bfloat16, "compile"),
            ("clamped", torch.float16, "eager"),
            ("empty", torch.float32, "eager"),
            ("dense", torch.float64, "cpu"),
            ("layout", torch.float32, "eager"),
            ("assignment", torch.float32, "eager"),
        ],
    )
    def test_forward(self, request, kind, dtype, mode):
        device = "cpu" if mode == "cpu" else request.getfixturevalue("cuda")
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        if kind in ("dense", "layout", "assignment"):
            tokens = torch.randn(2, 4, 3, device=device, dtype=dtype)
            index = torch.tensor([0, 2, 3, 1, 0, 2], device=device)
            mapping = AtomTokenLayout.from_atom_to_token(index, num_tokens=4) if kind == "layout" else index
            if kind == "assignment":
                mapping = torch.nn.functional.one_hot(index, 4).to(dtype)
            torch.testing.assert_close(operation(tokens, mapping), tokens.index_select(-2, index))
            return
        for variant in (0, 1) if mode == "compile" else (0,):
            tokens, mapping, _, reference, oracle = _case(kind, dtype, device, False, variant)
            torch.testing.assert_close(operation(tokens, mapping), oracle(reference).to(dtype))

    @pytest.mark.parametrize(
        ("kind", "dtype", "mode"),
        [
            ("nested", torch.float32, "compile"),
            ("sampled", torch.bfloat16, "checkpoint"),
            ("clamped", torch.float16, "eager"),
            ("empty", torch.float32, "eager"),
            ("nested", torch.float64, "jvp"),
            ("sampled", torch.float32, "jvp"),
        ],
    )
    def test_backward(self, request, kind, dtype, mode):
        device = "cpu" if mode == "jvp" else request.getfixturevalue("cuda")
        tokens, mapping, values, reference, oracle = _case(kind, dtype, device, mode != "jvp")
        expected = oracle(reference)
        if mode == "jvp":
            tangent = torch.randn_like(values)
            with forward_ad.dual_level():
                dual = forward_ad.make_dual(values, tangent)
                primal, actual_tangent = forward_ad.unpack_dual(_run(tokens.packed_like(dual), mapping))
            torch.testing.assert_close(primal, expected.to(dtype))
            torch.testing.assert_close(actual_tangent, oracle(tangent))
            return
        operation = torch.compile(_run, fullgraph=True) if mode == "compile" else _run
        actual = (
            checkpoint(operation, tokens, mapping, use_reentrant=False)
            if mode == "checkpoint"
            else operation(tokens, mapping)
        )
        upstream = torch.randn_like(actual)
        gradient = torch.autograd.grad(actual, values, upstream)[0]
        expected_gradient = torch.autograd.grad(expected, reference, upstream.double())[0]
        torch.testing.assert_close(actual, expected.to(dtype))
        torch.testing.assert_close(gradient, expected_gradient.to(dtype))

    @pytest.mark.parametrize("case", ["shape", "empty_tokens", "sample_axis", "nested_mismatch"])
    def test_contract(self, case):
        if case == "nested_mismatch":
            tokens = NestedTensor([torch.ones(2, 3), torch.ones(3, 3)], ragged_dims=(0,))
            mapping = NestedTensor([torch.tensor([0])], ragged_dims=(0,))
        else:
            tokens = torch.ones(2, 0 if case == "sample_axis" else 3, 0 if case == "empty_tokens" else 4, 5)
            mapping = (
                torch.zeros(2, 1, 2, 7, dtype=torch.long) if case == "shape" else torch.zeros(2, 7, dtype=torch.long)
            )
        with pytest.raises((ValueError, TypeError, OOpsUnsupportedError)):
            token_to_atom_broadcast(tokens, mapping)
