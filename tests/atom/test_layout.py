from __future__ import annotations

import pytest
import torch

from oops import AtomTokenLayout, validate_atom_to_token

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps kernels require CUDA")


class TestAtomTokenLayout:

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
    def test_sorted_layout_reorders_an_arbitrary_mapping(self, device: str):
        r"""``sort=True`` records the reordering that makes an arbitrary mapping contiguous per token."""

        arbitrary = torch.tensor([3, 0, 2, 0, 1, 3, 3, 2], device=device)
        layout = AtomTokenLayout.from_atom_to_token(arbitrary, num_tokens=4, sort=True)
        sorted_layout = layout.sorted_layout
        assert layout.atoms_per_token is None
        assert sorted_layout is not None and sorted_layout.permutation is not None
        assert sorted_layout.inverse_permutation is not None

        assert torch.equal(arbitrary.index_select(0, sorted_layout.permutation), sorted_layout.sorted_atom_to_token)
        assert sorted_layout.max_atoms_per_token == 3
        offsets = sorted_layout.token_offsets
        assert offsets.tolist() == [0, 2, 3, 5, 8]
        for token in range(4):
            segment = sorted_layout.sorted_atom_to_token[offsets[token] : offsets[token + 1]]
            assert torch.equal(segment, torch.full_like(segment, token))

        values = torch.randn(2, 8, 3, device=device)
        gathered = values.index_select(-2, sorted_layout.permutation)
        assert torch.equal(gathered.index_select(-2, sorted_layout.inverse_permutation), values)

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
    def test_validate_atom_to_token(self, device: str):
        r"""Validate the range and regular-layout clauses on host and device metadata."""

        atom_to_token = torch.arange(8, device=device) // 2
        validate_atom_to_token(atom_to_token, num_tokens=4)
        validate_atom_to_token(AtomTokenLayout(atom_to_token, 4, 2))
        validate_atom_to_token(atom_to_token.expand(3, -1), num_tokens=4)
        validate_atom_to_token(torch.zeros(0, dtype=torch.long, device=device), num_tokens=0)
        with pytest.raises(ValueError, match="num_tokens must be non-negative"):
            validate_atom_to_token(torch.zeros(0, dtype=torch.long, device=device), num_tokens=-1)

        with pytest.raises(ValueError, match=r"lie in \[0, 4\), got \[7\] = 4"):
            validate_atom_to_token(torch.tensor([0, 0, 1, 1, 2, 2, 3, 4], device=device), num_tokens=4)
        with pytest.raises(ValueError, match=r"lie in \[0, 4\), got \[0\] = -1"):
            validate_atom_to_token(torch.tensor([-1, 0, 1, 1, 2, 2, 3, 3], device=device), num_tokens=4)
        with pytest.raises(ValueError, match=r"atom_to_token\[i\] == i // 2, got \[2\] = 0 where 1 was expected"):
            validate_atom_to_token(AtomTokenLayout(torch.tensor([0, 0, 0, 1, 2, 2, 3, 3], device=device), 4, 2))
        with pytest.raises(ValueError, match="num_tokens is required"):
            validate_atom_to_token(atom_to_token)
