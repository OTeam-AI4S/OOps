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
import warnings

import torch
from torch import Tensor

from oops.runtime import (
    INT32_MAX,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps_if_needed,
    cuda_device_context,
    tensors_share_device,
    unsupported,
)

from ...layout import AtomTokenLayout
from . import kernel
from .profiles import NESTED_TOKEN_PAIR_WINDOW_BACKWARD, NESTED_TOKEN_PAIR_WINDOW_FORWARD, TOKEN_PAIR_WINDOW
from .schedule import (
    NestedTokenPairWindowPlan,
    NestedTokenPairWindowProblem,
    NestedTokenPairWindowSchedule,
    TokenPairWindowAtomicBackwardPlan,
    TokenPairWindowPlan,
    TokenPairWindowProblem,
    TokenPairWindowRegularBackwardPlan,
)

_NONDETERMINISTIC_ATOM_WINDOW_BACKWARD = (
    "does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True)'. The irregular/partial atom-window backward "
    "uses relaxed floating atomics. A qualifying complete regular AtomTokenLayout selects the "
    "deterministic reduce only when width is at least 4, atoms fill query blocks exactly, key "
    "slots do not exceed 64, and its reduction grids fit; otherwise use warn_only=True if "
    "nondeterminism is acceptable."
)


def _require_atomic_backward_allowed() -> None:
    if not torch.are_deterministic_algorithms_enabled():
        return
    rejection = unsupported(
        "token_pair_to_atom_window_gather backward",
        _NONDETERMINISTIC_ATOM_WINDOW_BACKWARD,
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(str(rejection), UserWarning, stacklevel=3)
        return
    raise rejection


def supports_token_pair_to_atom_window_gather(
    token_pair: Tensor,
    layout: AtomTokenLayout,
    n_queries: int,
    n_keys: int,
) -> bool:
    atom_to_token = layout.atom_to_token
    layout_supported = (
        token_pair.is_cuda
        and atom_to_token.is_cuda
        and tensors_share_device(token_pair, atom_to_token)
        and token_pair.is_contiguous()
        and atom_to_token.is_contiguous()
        and token_pair.dim() >= 3
        and token_pair.shape[-3] == token_pair.shape[-2]
        and layout.num_tokens == token_pair.shape[-2]
        and token_pair.dtype in STANDARD_FLOAT_DTYPES
        and atom_to_token.dtype == torch.long
        and atom_to_token.numel() > 0
        and token_pair.numel() > 0
        and token_pair.shape[-1] > 0
        and token_pair.shape[-2] > 0
        and n_queries > 0
        and n_keys > 0
        and n_keys >= n_queries
        and n_queries % 2 == 0
        and n_keys % 2 == 0
    )
    return layout_supported and TOKEN_PAIR_WINDOW.supports(
        _token_pair_window_problem(token_pair, layout, n_queries, n_keys)
    )


def _token_pair_window_problem(
    token_pair: Tensor,
    layout: AtomTokenLayout,
    n_queries: int,
    n_keys: int,
) -> TokenPairWindowProblem:
    batch = math.prod(token_pair.shape[:-3]) if token_pair.dim() > 3 else 1
    return TokenPairWindowProblem(
        batch=batch,
        tokens=token_pair.shape[-2],
        atoms=layout.atom_to_token.numel(),
        channels=token_pair.shape[-1],
        n_queries=n_queries,
        n_keys=n_keys,
        atoms_per_token=layout.atoms_per_token,
        dtype=token_pair.dtype,
    )


def _grad_window_problem(
    grad_out: Tensor,
    tokens: int,
    atoms: int,
    atoms_per_token: int | None,
) -> TokenPairWindowProblem:
    return TokenPairWindowProblem(
        batch=math.prod(grad_out.shape[:-4]),
        tokens=tokens,
        atoms=atoms,
        channels=grad_out.shape[-1],
        n_queries=grad_out.shape[-3],
        n_keys=grad_out.shape[-2],
        atoms_per_token=atoms_per_token,
        dtype=grad_out.dtype,
    )


def _run_token_pair_to_atom_window_gather_forward(
    token_pair: Tensor,
    atom_to_token: Tensor,
    atoms_per_token: int | None,
    n_queries: int,
    n_keys: int,
) -> Tensor:
    tokens = token_pair.shape[-2]
    layout = AtomTokenLayout(atom_to_token, tokens, atoms_per_token)
    plan = TOKEN_PAIR_WINDOW.resolve(_token_pair_window_problem(token_pair, layout, n_queries, n_keys))
    return _launch_token_pair_to_atom_window_gather(token_pair, atom_to_token, plan)


def _launch_token_pair_to_atom_window_gather(
    token_pair: Tensor,
    atom_to_token: Tensor,
    plan: TokenPairWindowPlan,
) -> Tensor:
    input_shape = tuple(token_pair.shape)
    tokens = input_shape[-2]
    token_pair_flat = token_pair.reshape(
        plan.problem.batch,
        tokens,
        tokens,
        plan.problem.channels,
    )
    out = _token_pair_to_atom_window_gather_forward_impl(
        token_pair_flat,
        atom_to_token,
        plan,
    )
    return out.reshape(
        *input_shape[:-3],
        plan.forward.n_blocks,
        plan.problem.n_queries,
        plan.problem.n_keys,
        plan.problem.channels,
    )


def _run_token_pair_to_atom_window_gather_backward(
    grad_out: Tensor,
    atom_to_token: Tensor,
    tokens: int,
    atoms_per_token: int | None,
    plan: TokenPairWindowPlan | None = None,
) -> Tensor:
    with cuda_device_context(grad_out.device):
        grad_out = grad_out.resolve_neg()
        atom_to_token = atom_to_token.resolve_neg().contiguous()
        atoms = atom_to_token.numel()
        if plan is None:
            plan = TOKEN_PAIR_WINDOW.resolve(_grad_window_problem(grad_out, tokens, atoms, atoms_per_token))
        output_shape = (*grad_out.shape[:-4], tokens, tokens, plan.problem.channels)
        if isinstance(plan.backward, TokenPairWindowRegularBackwardPlan):
            grad_token_pair = _token_pair_to_atom_window_gather_regular_backward(
                grad_out.contiguous().reshape(plan.forward.flat_output_shape),
                plan,
            )
            return grad_token_pair.reshape(output_shape)

        _require_atomic_backward_allowed()
        grad_token_pair = _token_pair_to_atom_window_gather_atomic_backward(
            grad_out.contiguous().reshape(plan.forward.flat_output_shape),
            atom_to_token,
            plan,
        )
        return grad_token_pair.reshape(output_shape)


def _token_pair_to_atom_window_gather_forward_impl(
    token_pair: Tensor,
    atom_to_token: Tensor,
    plan: TokenPairWindowPlan,
) -> Tensor:
    with cuda_device_context(token_pair.device):
        token_pair = token_pair.resolve_neg()
        atom_to_token = atom_to_token.resolve_neg().contiguous()
        out = torch.empty(
            plan.forward.flat_output_shape,
            device=token_pair.device,
            dtype=token_pair.dtype,
        )
        kernel._token_pair_to_atom_window_gather_forward[plan.forward.grid](
            token_pair,
            atom_to_token,
            out,
            plan.forward.total_out,
            plan.problem.atoms,
            plan.problem.tokens,
            plan.problem.channels,
            plan.forward.n_blocks,
            plan.problem.n_queries,
            plan.problem.n_keys,
            plan.forward.pad_left,
            plan.forward.block_elements,
            num_warps=plan.forward.num_warps,
            use_int64=plan.forward.grid[0] * plan.forward.block_elements > INT32_MAX or token_pair.numel() > INT32_MAX,
        )
        return out


def _token_pair_to_atom_window_gather_atomic_backward(
    grad_out: Tensor,
    atom_to_token: Tensor,
    plan: TokenPairWindowPlan,
) -> Tensor:
    backward = plan.backward
    if not isinstance(backward, TokenPairWindowAtomicBackwardPlan):
        raise ValueError("atomic atom-window launcher requires an atomic plan")
    problem = plan.problem
    grad_token_pair = torch.zeros(
        (problem.batch, problem.tokens, problem.tokens, problem.channels),
        device=grad_out.device,
        dtype=plan.accumulation_dtype,
    )
    if backward.block_points:
        kernel._token_pair_to_atom_window_grouped_backward[backward.grid](
            grad_out,
            atom_to_token,
            grad_token_pair,
            plan.forward.total_out // problem.channels,
            problem.atoms,
            problem.tokens,
            problem.channels,
            plan.forward.n_blocks,
            problem.n_queries,
            problem.n_keys,
            plan.forward.pad_left,
            backward.block_points,
            backward.block_channels,
            use_int64=backward.grid[0] * backward.block_points * problem.channels > INT32_MAX
            or problem.batch * problem.tokens * problem.tokens * problem.channels > INT32_MAX,
            num_warps=backward.num_warps,
        )
    else:
        kernel._token_pair_to_atom_window_gather_backward[backward.grid](
            grad_out,
            atom_to_token,
            grad_token_pair,
            plan.forward.total_out,
            problem.atoms,
            problem.tokens,
            problem.channels,
            plan.forward.n_blocks,
            problem.n_queries,
            problem.n_keys,
            plan.forward.pad_left,
            backward.block_elements,
            num_warps=backward.num_warps,
            use_int64=backward.grid[0] * backward.block_elements > INT32_MAX
            or problem.batch * problem.tokens * problem.tokens * problem.channels > INT32_MAX,
        )
    return grad_token_pair.to(grad_out.dtype) if grad_token_pair.dtype != grad_out.dtype else grad_token_pair


def _token_pair_to_atom_window_gather_regular_backward(
    grad_out: Tensor,
    plan: TokenPairWindowPlan,
) -> Tensor:
    backward = plan.backward
    if not isinstance(backward, TokenPairWindowRegularBackwardPlan):
        raise ValueError("regular atom-window launcher requires a regular plan")
    problem = plan.problem
    atoms_per_token = problem.atoms_per_token
    if atoms_per_token is None:
        raise ValueError("regular atom-window plan requires atoms_per_token")
    grad_token_pair = torch.zeros(
        (problem.batch, problem.tokens, problem.tokens, problem.channels),
        device=grad_out.device,
        dtype=plan.accumulation_dtype,
    )
    kernel._token_pair_to_atom_window_gather_regular_zero_backward[backward.zero_grid](
        grad_out,
        grad_token_pair,
        problem.atoms,
        problem.tokens,
        problem.channels,
        plan.forward.n_blocks,
        atoms_per_token,
        problem.n_queries,
        problem.n_keys,
        plan.forward.pad_left,
        backward.wide_offsets,
        backward.block_channels,
        backward.block_atoms,
        num_warps=backward.num_warps,
    )
    if backward.reduce_grid is not None:
        kernel._token_pair_to_atom_window_gather_regular_backward[backward.reduce_grid](
            grad_out,
            grad_token_pair,
            problem.atoms,
            problem.tokens,
            problem.channels,
            plan.forward.n_blocks,
            atoms_per_token,
            backward.max_key_slots,
            problem.n_queries,
            problem.n_keys,
            plan.forward.pad_left,
            backward.wide_offsets,
            backward.block_channels,
            backward.block_atoms,
            num_warps=backward.num_warps,
        )
    return grad_token_pair.to(grad_out.dtype) if grad_token_pair.dtype != grad_out.dtype else grad_token_pair


_NONDETERMINISTIC_NESTED_ATOM_WINDOW_BACKWARD = (
    "does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True)'. A NestedTensor mapping has bare-index "
    "semantics and its packed atom-window backward uses relaxed floating atomics; use "
    "warn_only=True if nondeterminism is acceptable."
)


def _require_nested_atomic_backward_allowed() -> None:
    if not torch.are_deterministic_algorithms_enabled():
        return
    rejection = unsupported(
        "token_pair_to_atom_window_gather backward",
        _NONDETERMINISTIC_NESTED_ATOM_WINDOW_BACKWARD,
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(str(rejection), UserWarning, stacklevel=3)
        return
    raise rejection


def _nested_token_pair_window_problem(
    pair: Tensor,
    atom_to_token: Tensor,
    atom_offsets: Tensor,
    packed_metadata: Tensor,
    block_to_sample: Tensor,
    has_num_valid_atoms: bool,
    n_queries: int,
    n_keys: int,
) -> NestedTokenPairWindowProblem:
    return NestedTokenPairWindowProblem(
        num_samples=atom_offsets.shape[0] - 1,
        total_pair_positions=pair.shape[0],
        total_atoms=atom_to_token.shape[0],
        total_blocks=block_to_sample.shape[0],
        channels=pair.shape[1],
        n_queries=n_queries,
        n_keys=n_keys,
        dtype=pair.dtype,
        has_num_valid_atoms=has_num_valid_atoms,
    )


def _resolve_nested_token_pair_window_plan(
    schedule: NestedTokenPairWindowSchedule,
    problem: NestedTokenPairWindowProblem,
    device: torch.device,
) -> NestedTokenPairWindowPlan:
    target = cuda_device_caps_if_needed(device, schedule.target_changes_plan(problem))
    return schedule.resolve(problem, target=target)


def supports_nested_token_pair_to_atom_window_gather(
    pair: Tensor,
    atom_to_token: Tensor,
    token_offsets: Tensor,
    atom_offsets: Tensor,
    packed_metadata: Tensor,
    block_to_sample: Tensor,
    has_num_valid_atoms: bool,
    n_queries: int,
    n_keys: int,
    *,
    check_plan: bool = True,
) -> bool:

    metadata = (token_offsets, atom_offsets, packed_metadata, block_to_sample)
    samples = atom_offsets.shape[0] - 1
    supported = (
        pair.is_cuda
        and pair.dim() == 2
        and pair.is_contiguous()
        and pair.dtype in STANDARD_FLOAT_DTYPES
        and pair.shape[1] > 0
        and atom_to_token.is_cuda
        and atom_to_token.dtype is torch.long
        and atom_to_token.dim() == 1
        and atom_to_token.is_contiguous()
        and tensors_share_device(pair, atom_to_token, *metadata)
        and samples > 0
        and token_offsets.dtype is torch.int64
        and atom_offsets.dtype is torch.int64
        and packed_metadata.dtype is torch.int64
        and block_to_sample.dtype is torch.int64
        and all(tensor.dim() == 1 and tensor.is_contiguous() for tensor in metadata)
        and token_offsets.shape == atom_offsets.shape
        and packed_metadata.shape[0] == 3 * atom_offsets.shape[0]
        and n_queries > 0
        and n_keys >= n_queries
        and n_queries % 2 == 0
        and n_keys % 2 == 0
    )
    if not supported or not check_plan:
        return supported
    problem = _nested_token_pair_window_problem(
        pair,
        atom_to_token,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        n_queries,
        n_keys,
    )
    return NESTED_TOKEN_PAIR_WINDOW_FORWARD.supports(problem) and NESTED_TOKEN_PAIR_WINDOW_BACKWARD.supports(problem)


def _run_nested_token_pair_to_atom_window_gather_forward(
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
    with cuda_device_context(pair.device):
        pair = pair.resolve_neg()
        atom_to_token = atom_to_token.resolve_neg().contiguous()
        plan = _resolve_nested_token_pair_window_plan(
            NESTED_TOKEN_PAIR_WINDOW_FORWARD,
            _nested_token_pair_window_problem(
                pair,
                atom_to_token,
                atom_offsets,
                packed_metadata,
                block_to_sample,
                has_num_valid_atoms,
                n_queries,
                n_keys,
            ),
            pair.device,
        )
        problem = plan.problem
        output = torch.empty(
            (problem.total_blocks, problem.n_queries, problem.n_keys, problem.channels),
            device=pair.device,
            dtype=pair.dtype,
        )
        if problem.total_blocks > 0:
            kernel._nested_token_pair_to_atom_window_gather_forward[plan.grid](
                pair,
                atom_to_token,
                token_offsets,
                atom_offsets,
                packed_metadata,
                block_to_sample,
                output,
                problem.num_samples,
                problem.channels,
                problem.n_queries,
                problem.n_keys,
                (problem.n_keys - problem.n_queries) // 2,
                problem.has_num_valid_atoms,
                plan.block_elements,
                num_warps=plan.num_warps,
            )
        return output


def _run_nested_token_pair_to_atom_window_gather_backward(
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
    with cuda_device_context(grad_out.device):
        grad_out = grad_out.resolve_neg()
        atom_to_token = atom_to_token.resolve_neg().contiguous()
        token_offsets = token_offsets.contiguous()
        atom_offsets = atom_offsets.contiguous()
        packed_metadata = packed_metadata.contiguous()
        block_to_sample = block_to_sample.contiguous()
        problem = NestedTokenPairWindowProblem(
            num_samples=atom_offsets.shape[0] - 1,
            total_pair_positions=pair_positions,
            total_atoms=atom_to_token.shape[0],
            total_blocks=block_to_sample.shape[0],
            channels=grad_out.shape[-1],
            n_queries=n_queries,
            n_keys=n_keys,
            dtype=grad_out.dtype,
            has_num_valid_atoms=has_num_valid_atoms,
        )
        plan = _resolve_nested_token_pair_window_plan(NESTED_TOKEN_PAIR_WINDOW_BACKWARD, problem, grad_out.device)
        if problem.total_blocks > 0:
            _require_nested_atomic_backward_allowed()
        grad_pair = torch.zeros(
            (problem.total_pair_positions, problem.channels),
            device=grad_out.device,
            dtype=plan.accumulation_dtype,
        )
        if problem.total_blocks > 0:
            kernel._nested_token_pair_to_atom_window_gather_backward[plan.grid](
                grad_out.contiguous(),
                atom_to_token,
                token_offsets,
                atom_offsets,
                packed_metadata,
                block_to_sample,
                grad_pair,
                problem.num_samples,
                problem.channels,
                problem.n_queries,
                problem.n_keys,
                (problem.n_keys - problem.n_queries) // 2,
                problem.has_num_valid_atoms,
                plan.block_elements,
                num_warps=plan.num_warps,
            )
        return grad_pair.to(grad_out.dtype) if grad_pair.dtype != grad_out.dtype else grad_pair


def _resolve_token_pair_window_plan(
    token_pair: Tensor, layout: AtomTokenLayout, n_queries: int, n_keys: int
) -> TokenPairWindowPlan:
    return TOKEN_PAIR_WINDOW.resolve(_token_pair_window_problem(token_pair, layout, n_queries, n_keys))
