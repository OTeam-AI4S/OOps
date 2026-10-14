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

import math

import pytest
import torch
from danling.tensors import NestedTensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oops import AtomTokenLayout, OOpsUnsupportedError, token_pair_to_atom_window_gather


def _indices(atom_to_token, n_queries, n_keys):
    atoms = atom_to_token.numel()
    blocks = math.ceil(atoms / n_queries)
    query = F.pad(atom_to_token, (0, blocks * n_queries - atoms)).reshape(blocks, n_queries)
    left = (n_keys - n_queries) // 2
    right = (blocks - 1) * n_queries + n_keys - left - atoms
    key = F.pad(atom_to_token, (left, right)).unfold(0, n_keys, n_queries)
    return query, key


def _reference(token_pair, atom_to_token, n_queries, n_keys):
    query, key = _indices(atom_to_token, n_queries, n_keys)
    return token_pair[
        ...,
        query[:, :, None].expand(-1, -1, n_keys),
        key[:, None, :].expand(-1, n_queries, -1),
        :,
    ]


def _clamped_reference(token_pair, atom_to_token, n_queries, n_keys, num_valid_atoms):
    atoms = atom_to_token.numel()
    blocks = math.ceil(atoms / n_queries)
    query = F.pad(atom_to_token, (0, blocks * n_queries - atoms)).reshape(blocks, n_queries)
    left = (n_keys - n_queries) // 2
    centered = torch.arange(blocks, device=atom_to_token.device) * n_queries - left
    starts = centered.clamp(0, max(num_valid_atoms - n_keys, 0))
    key_atoms = starts[:, None] + torch.arange(n_keys, device=atom_to_token.device)[None]
    valid_keys = (key_atoms >= 0) & (key_atoms < num_valid_atoms)
    key = atom_to_token[key_atoms.clamp(0, atom_to_token.numel() - 1)]
    key = torch.where(valid_keys, key, 0)
    return token_pair[
        query[:, :, None].expand(-1, -1, n_keys),
        key[:, None, :].expand(-1, n_queries, -1),
        :,
    ]


def _nested_inputs(token_lengths, mappings, dtype, device, *, channels=7, requires_grad=True):
    pair_reference = NestedTensor(
        [torch.empty(length, length, channels) for length in token_lengths],
        ragged_dims=(0, 1),
    )
    pair_values = torch.randn(pair_reference.concat.shape, device=device, dtype=dtype, requires_grad=requires_grad)
    pair = pair_reference.packed_like(pair_values)
    mapping_values = torch.cat(tuple(mapping.to(device=device, dtype=torch.long) for mapping in mappings))
    mapping_lengths = torch.tensor([mapping.numel() for mapping in mappings], dtype=torch.int64)
    mapping = pair_reference.packed_with_lengths(mapping_values, mapping_lengths)
    return pair, mapping, pair_values, mapping_values


def _nested_reference(pair_values, mapping_values, token_lengths, atom_lengths, n_queries, n_keys, valid=None):
    channels = pair_values.shape[-1]
    pairs = pair_values.split(tuple(length * length for length in token_lengths))
    mappings = mapping_values.split(tuple(atom_lengths))
    outputs = []
    for sample, (pair, mapping, tokens) in enumerate(zip(pairs, mappings, token_lengths, strict=True)):
        pair = pair.reshape(tokens, tokens, channels)
        if mapping.numel() == 0:
            outputs.append(pair.new_empty((0, n_queries, n_keys, channels)) + pair.sum() * 0)
        elif valid is None:
            outputs.append(_reference(pair, mapping, n_queries, n_keys))
        else:
            outputs.append(_clamped_reference(pair, mapping, n_queries, n_keys, int(valid[sample])))
    return torch.cat(outputs)


def _case(kind, dtype, device, training, variant=0, channels=8):
    torch.manual_seed(8421 + variant)
    if kind in ("nested", "empty"):
        tokens = (0, 3) if kind == "empty" else ((3, 5) if not variant else (4, 6))
        mappings = tuple(
            torch.empty(0, dtype=torch.long) if kind == "empty" else torch.arange(2 * n) % n for n in tokens
        )
        pair, mapping, values, indices = _nested_inputs(tokens, mappings, dtype, device, requires_grad=training)
        reference = values.detach().double().requires_grad_(training)
        lengths = tuple(index.numel() for index in mappings)

        def oracle():
            return _nested_reference(reference, indices, tokens, lengths, 2, 4)

        return (pair, mapping, 2, 4, None), values, reference, oracle
    atoms, tokens, queries, keys = (40, 10, 32, 128) if kind == "clamped" else (128, 32, 32, 64)
    index = torch.arange(atoms, device=device) // 4
    if kind == "irregular":
        index = torch.repeat_interleave(torch.arange(tokens, device=device), torch.tensor([5, 3] * 16, device=device))
    mapping = index if kind == "raw" else AtomTokenLayout.from_atom_to_token(index, num_tokens=tokens)
    values = torch.randn(1, tokens, tokens, channels, device=device, dtype=dtype, requires_grad=training)
    reference = values.detach().double().requires_grad_(training)
    valid = 37 if kind == "clamped" else None
    oracle = (
        (lambda: torch.stack([_clamped_reference(part, index, queries, keys, int(valid)) for part in reference]))
        if valid is not None
        else (lambda: _reference(reference, index, queries, keys))
    )
    return (values, mapping, queries, keys, valid), values, reference, oracle


def _run(pair, mapping, queries, keys, valid):
    result = token_pair_to_atom_window_gather(pair, mapping, queries, keys, num_valid_atoms=valid)
    return result.concat if isinstance(result, NestedTensor) else result


class TestTokenPairToAtomWindowGather:

    @pytest.mark.parametrize(
        ("kind", "dtype", "mode"),
        [
            ("regular", torch.float32, "eager"),
            ("irregular", torch.float16, "eager"),
            ("clamped", torch.float32, "cpu"),
            ("clamped", torch.bfloat16, "compile"),
            ("nested", torch.bfloat16, "compile"),
            ("regular", torch.float32, "graph"),
            ("empty", torch.float32, "eager"),
            ("regular", torch.float32, "negative"),
        ],
    )
    def test_forward(self, request, kind, dtype, mode, logical_negative_view):
        device = "cpu" if mode == "cpu" else request.getfixturevalue("cuda")
        operation = torch.compile(_run, fullgraph=True, dynamic=True) if mode == "compile" else _run
        for variant in (0, 1) if mode == "compile" else (0,):
            inputs, values, reference, oracle = _case(kind, dtype, device, False, variant)
            if mode == "negative":
                inputs = (logical_negative_view(inputs[0]), *inputs[1:])
            if mode == "graph":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        operation(*inputs)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = operation(*inputs)
                values.add_(0.25)
                reference.copy_(values.double())
                graph.replay()
            else:
                actual = operation(*inputs)
            torch.testing.assert_close(actual.double(), oracle(), atol=0, rtol=0)

    @pytest.mark.parametrize(
        ("kind", "dtype", "mode", "atol", "rtol", "channels"),
        [
            pytest.param(
                "clamped", torch.float32, "eager", 1e-5, 1e-5, 8, marks=pytest.mark.cuda_device(1), id="secondary-cuda"
            ),
            ("regular", torch.bfloat16, "compile", 1e-2, 1e-2, 8),
            ("regular", torch.float32, "deterministic", 1e-5, 1e-5, 8),
            ("irregular", torch.float16, "eager", 1.5e-1, 2e-2, 8),
            ("irregular", torch.float32, "saved-strided", 1e-5, 1e-5, 8),
            ("clamped", torch.float32, "eager", 1e-5, 1e-5, 8),
            ("nested", torch.bfloat16, "checkpoint", 1.5e-1, 2e-2, 8),
            ("empty", torch.float32, "deterministic", 0, 0, 8),
            ("raw", torch.float32, "eager", 1e-5, 1e-5, 64),
            ("raw", torch.float32, "compile", 1e-5, 1e-5, 65),
            ("regular", torch.float32, "negative", 1e-5, 1e-5, 8),
            ("nested", torch.bfloat16, "negative", 1.5e-1, 2e-2, 8),
            ("nested", torch.bfloat16, "saved-strided", 1.5e-1, 2e-2, 8),
        ],
    )
    def test_backward(self, cuda, kind, dtype, mode, atol, rtol, channels, logical_negative_view, saved_strided_hook):
        inputs, values, reference, oracle = _case(kind, dtype, cuda, True, channels=channels)
        if mode == "negative":
            inputs = (logical_negative_view(inputs[0]), *inputs[1:])
        operation = torch.compile(_run, fullgraph=True) if mode == "compile" else _run
        previous, previous_warn = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        try:
            torch.use_deterministic_algorithms(mode == "deterministic")
            if mode == "saved-strided":
                with saved_strided_hook():
                    actual = operation(*inputs)
            else:
                actual = (
                    checkpoint(operation, *inputs, use_reentrant=False) if mode == "checkpoint" else operation(*inputs)
                )
            upstream = torch.randn_like(actual)
            gradient = torch.autograd.grad(actual, values, upstream, retain_graph=mode == "deterministic")[0]
            if mode == "deterministic":
                repeated = torch.autograd.grad(actual, values, upstream)[0]
                torch.testing.assert_close(gradient, repeated, atol=0, rtol=0)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
        expected = oracle()
        expected_gradient = torch.autograd.grad(expected, reference, upstream.double())[0]
        torch.testing.assert_close(actual.double(), expected, atol=0, rtol=0)
        torch.testing.assert_close(gradient.double(), expected_gradient, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("case", ["cpu", "valid_extent", "square", "forward_ad", "deterministic"])
    def test_contract(self, request, case):
        if case == "deterministic":
            inputs, values, _, _ = _case("nested", torch.float32, request.getfixturevalue("cuda"), True)
            output = _run(*inputs)
            previous, previous_warn = (
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            try:
                torch.use_deterministic_algorithms(True)
                with pytest.raises(OOpsUnsupportedError, match="deterministic"):
                    torch.autograd.grad(output.sum(), values, retain_graph=True)
                torch.use_deterministic_algorithms(True, warn_only=True)
                with pytest.warns(UserWarning, match="deterministic"):
                    torch.autograd.grad(output.sum(), values)
            finally:
                torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
            return
        pair = torch.randn(1, 4, 3 if case == "square" else 4, 3)
        mapping = torch.arange(8) // 2
        if case == "forward_ad":
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(pair, torch.ones_like(pair))
                with pytest.raises(OOpsUnsupportedError, match="forward-mode AD"):
                    token_pair_to_atom_window_gather(dual, mapping, 2, 4)
        else:
            with pytest.raises(OOpsUnsupportedError if case == "cpu" else ValueError):
                token_pair_to_atom_window_gather(
                    pair, mapping, 2, 4, num_valid_atoms=9 if case == "valid_extent" else None
                )
