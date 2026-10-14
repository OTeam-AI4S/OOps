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

import torch
from torch import Tensor

from oops.runtime import is_compiling, is_external_fake
from oops.runtime import needs_grad as tensors_need_grad

from ...layout import AtomTokenLayout
from .generic import (
    _grad_window_problem,
    _launch_token_pair_to_atom_window_gather,
    _resolve_token_pair_window_plan,
    _run_nested_token_pair_to_atom_window_gather_backward,
    _run_nested_token_pair_to_atom_window_gather_forward,
    _run_token_pair_to_atom_window_gather_backward,
    _run_token_pair_to_atom_window_gather_forward,
)
from .profiles import TOKEN_PAIR_WINDOW


def triton_token_pair_to_atom_window_gather(
    token_pair: Tensor,
    layout: AtomTokenLayout,
    n_queries: int,
    n_keys: int,
) -> Tensor:

    if tensors_need_grad(token_pair):
        return _token_pair_to_atom_window_gather_forward(
            token_pair,
            layout.atom_to_token,
            layout.atoms_per_token,
            n_queries,
            n_keys,
        )
    compiling = is_compiling()
    pair_type = type(token_pair)
    external_fake = (
        not compiling
        and pair_type is not Tensor
        and pair_type is not torch.nn.Parameter
        and is_external_fake(token_pair)
    )
    if compiling or external_fake:
        return _token_pair_to_atom_window_gather_forward(
            token_pair,
            layout.atom_to_token,
            layout.atoms_per_token,
            n_queries,
            n_keys,
        )
    plan = _resolve_token_pair_window_plan(token_pair, layout, n_queries, n_keys)
    return _launch_token_pair_to_atom_window_gather(token_pair, layout.atom_to_token, plan)


@torch.library.custom_op("ooperators::_token_pair_to_atom_window_gather_forward", mutates_args=(), device_types="cuda")
def _token_pair_to_atom_window_gather_forward(
    token_pair: Tensor,
    atom_to_token: Tensor,
    atoms_per_token: int | None,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    return _run_token_pair_to_atom_window_gather_forward(token_pair, atom_to_token, atoms_per_token, n_queries, n_keys)


@torch.library.register_fake(_token_pair_to_atom_window_gather_forward)
def _token_pair_to_atom_window_gather_forward_fake(
    token_pair: Tensor,
    atom_to_token: Tensor,
    atoms_per_token: int | None,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    del atoms_per_token
    n_blocks = (atom_to_token.numel() + n_queries - 1) // n_queries
    return torch.empty(
        (*token_pair.shape[:-3], n_blocks, n_queries, n_keys, token_pair.shape[-1]),
        device=token_pair.device,
        dtype=token_pair.dtype,
    )


@torch.library.custom_op(
    "ooperators::_token_pair_to_atom_window_gather_backward",
    mutates_args=(),
    device_types="cuda",
)
def _token_pair_to_atom_window_gather_backward(
    grad_out: Tensor,
    atom_to_token: Tensor,
    tokens: int,
    atoms_per_token: int | None,
) -> Tensor:
    return _run_token_pair_to_atom_window_gather_backward(
        grad_out,
        atom_to_token,
        tokens,
        atoms_per_token,
    )


@torch.library.register_fake(_token_pair_to_atom_window_gather_backward)
def _token_pair_to_atom_window_gather_backward_fake(
    grad_out: Tensor,
    atom_to_token: Tensor,
    tokens: int,
    atoms_per_token: int | None,
) -> Tensor:
    del atom_to_token, atoms_per_token
    return torch.empty(
        (*grad_out.shape[:-4], tokens, tokens, grad_out.shape[-1]),
        device=grad_out.device,
        dtype=grad_out.dtype,
    )


def _setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor, int | None, int, int],
    output: Tensor,
) -> None:
    token_pair, atom_to_token, atoms_per_token, _n_queries, _n_keys = inputs
    ctx.tokens = token_pair.shape[-2]
    ctx.atoms_per_token = atoms_per_token
    compiling = is_compiling()
    pair_type = type(token_pair)
    ctx.semantic_boundary = compiling or (
        pair_type is not Tensor and pair_type is not torch.nn.Parameter and is_external_fake(token_pair)
    )
    if not ctx.semantic_boundary:
        ctx.backward_plan = TOKEN_PAIR_WINDOW.resolve(
            _grad_window_problem(output, ctx.tokens, atom_to_token.numel(), atoms_per_token)
        )
    ctx.save_for_backward(atom_to_token)


def _backward(ctx, grad_out: Tensor) -> tuple[Tensor, None, None, None, None]:
    (atom_to_token,) = ctx.saved_tensors
    if ctx.semantic_boundary:
        grad_token_pair = _token_pair_to_atom_window_gather_backward(
            grad_out,
            atom_to_token,
            ctx.tokens,
            ctx.atoms_per_token,
        )
    else:
        grad_token_pair = _run_token_pair_to_atom_window_gather_backward(
            grad_out,
            atom_to_token,
            ctx.tokens,
            ctx.atoms_per_token,
            ctx.backward_plan,
        )
    return grad_token_pair, None, None, None, None


torch.library.register_autograd(_token_pair_to_atom_window_gather_forward, _backward, setup_context=_setup_context)

_NESTED_TOKEN_PAIR_WINDOW_FORWARD_SCHEMA = (
    "(Tensor pair, Tensor atom_to_token, Tensor token_offsets, Tensor atom_offsets, "
    "Tensor packed_metadata, Tensor block_to_sample, bool has_num_valid_atoms, int n_queries, int n_keys) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_nested_token_pair_to_atom_window_gather_forward",
    mutates_args=(),
    device_types="cuda",
    schema=_NESTED_TOKEN_PAIR_WINDOW_FORWARD_SCHEMA,
)
def _nested_token_pair_to_atom_window_gather_forward(
    pair: Tensor,
    atom_to_token: Tensor,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    packed_metadata: Tensor,
    block_to_sample: Tensor,
    has_num_valid_atoms: bool,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    return _run_nested_token_pair_to_atom_window_gather_forward(
        pair,
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        n_queries,
        n_keys,
    )


@torch.library.register_fake(_nested_token_pair_to_atom_window_gather_forward)
def _nested_token_pair_to_atom_window_gather_forward_fake(
    pair: Tensor,
    atom_to_token: Tensor,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    packed_metadata: Tensor,
    block_to_sample: Tensor,
    has_num_valid_atoms: bool,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    del (
        atom_to_token,
        token_offsets,
        has_num_valid_atoms,
    )
    return torch.empty(
        (block_to_sample.shape[0], n_queries, n_keys, pair.shape[1]),
        device=pair.device,
        dtype=pair.dtype,
    )


_NESTED_TOKEN_PAIR_WINDOW_BACKWARD_SCHEMA = (
    "(Tensor grad_out, Tensor atom_to_token, Tensor token_offsets, Tensor atom_offsets, "
    "Tensor packed_metadata, Tensor block_to_sample, bool has_num_valid_atoms, "
    "SymInt pair_positions, int n_queries, int n_keys) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_nested_token_pair_to_atom_window_gather_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_NESTED_TOKEN_PAIR_WINDOW_BACKWARD_SCHEMA,
)
def _nested_token_pair_to_atom_window_gather_backward(
    grad_out: Tensor,
    atom_to_token: Tensor,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    packed_metadata: Tensor,
    block_to_sample: Tensor,
    has_num_valid_atoms: bool,
    pair_positions: int,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    return _run_nested_token_pair_to_atom_window_gather_backward(
        grad_out,
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        pair_positions,
        n_queries,
        n_keys,
    )


@torch.library.register_fake(_nested_token_pair_to_atom_window_gather_backward)
def _nested_token_pair_to_atom_window_gather_backward_fake(
    grad_out: Tensor,
    atom_to_token: Tensor,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    packed_metadata: Tensor,
    block_to_sample: Tensor,
    has_num_valid_atoms: bool,
    pair_positions: int,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    del (
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        n_queries,
        n_keys,
    )
    return torch.empty(
        (pair_positions, grad_out.shape[-1]),
        device=grad_out.device,
        dtype=grad_out.dtype,
    )


def _nested_token_pair_window_setup_context(ctx, inputs, output) -> None:
    (
        pair,
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        n_queries,
        n_keys,
    ) = inputs
    del output
    ctx.save_for_backward(
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
    )
    ctx.has_num_valid_atoms = has_num_valid_atoms
    ctx.pair_positions = pair.shape[0]
    ctx.n_queries = n_queries
    ctx.n_keys = n_keys


def _nested_token_pair_window_autograd_backward(ctx, grad_out: Tensor):
    (
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
    ) = ctx.saved_tensors
    grad_pair = _nested_token_pair_to_atom_window_gather_backward(
        grad_out,
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        ctx.has_num_valid_atoms,
        ctx.pair_positions,
        ctx.n_queries,
        ctx.n_keys,
    )
    return grad_pair, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    _nested_token_pair_to_atom_window_gather_forward,
    _nested_token_pair_window_autograd_backward,
    setup_context=_nested_token_pair_window_setup_context,
)


def triton_nested_token_pair_to_atom_window_gather(
    pair: Tensor,
    atom_to_token: Tensor,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    packed_metadata: Tensor,
    block_to_sample: Tensor,
    has_num_valid_atoms: bool,
    n_queries: int,
    n_keys: int,
) -> Tensor:

    if not tensors_need_grad(pair):
        compiling = is_compiling()
        pair_type = type(pair)
        external_fake = (
            not compiling and pair_type is not Tensor and pair_type is not torch.nn.Parameter and is_external_fake(pair)
        )
        if not compiling and not external_fake:
            return _run_nested_token_pair_to_atom_window_gather_forward(
                pair,
                atom_to_token,
                token_offsets,
                atom_offsets,
                packed_metadata,
                block_to_sample,
                has_num_valid_atoms,
                n_queries,
                n_keys,
            )
    return _nested_token_pair_to_atom_window_gather_forward(
        pair,
        atom_to_token,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        n_queries,
        n_keys,
    )
