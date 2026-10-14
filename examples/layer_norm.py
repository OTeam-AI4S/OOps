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

r"""Run dense and variable-length LayerNorm forward/backward on the selected device."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from danling import NestedTensor

from oops import LayerNorm


def main(argv: Sequence[str] | None = None) -> None:
    r"""Run the LayerNorm examples.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    device = torch.device(arguments.device)
    torch.manual_seed(0)
    norm = LayerNorm(16, device=device)

    dense = torch.randn(2, 4, 16, device=device, requires_grad=True)
    dense_output = norm(dense)
    dense_output.square().mean().backward()
    assert dense.grad is not None
    print(f"Dense output: {tuple(dense_output.shape)}; input gradient: {tuple(dense.grad.shape)}")

    norm.zero_grad(set_to_none=True)
    sequences = [torch.randn(length, 16, device=device, requires_grad=True) for length in (3, 5)]
    nested = NestedTensor(sequences, ragged_dims=(0,))
    nested_output = norm(nested)
    nested_output.square().mean().backward()
    assert all(sequence.grad is not None for sequence in sequences)
    print(f"Nested output: {tuple(nested_output.concat.shape)} packed; sequence lengths: {[len(x) for x in sequences]}")


if __name__ == "__main__":
    main()
