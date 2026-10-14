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

import torch
from torch import Tensor

from oops.runtime import is_compiling, is_external_fake
from oops.runtime import needs_grad as tensors_need_grad

from ...layout import AtomTokenPlan
from .generic import (
    _MASK_VJP_INTERLEAVED,
    _MASK_VJP_REGULAR,
    _MASK_VJP_SORTED,
    _atom_to_token_reduce_interleaved_backward,
    _atom_to_token_reduce_interleaved_forward,
    _atom_to_token_reduce_sorted_backward,
    _atom_to_token_reduce_sorted_forward,
    _bind_prepared_atom_to_token_plan,
    _compatible_atom_to_token_offsets,
    _flatten_grad_out,
    _gather_atom_mask,
    _launch_atom_to_token_reduce_regular_forward,
    _launch_atom_to_token_reduce_regular_vjp,
    _native_prepared_atom_to_token_reduce,
    _plain_atom_call,
    _prepare_shared_general_backward_mapping,
    _prepare_shared_general_mask_backward_mapping,
    _resolve_atom_to_token_plan,
    _run_atom_to_token_reduce_mask_backward,
    _run_atom_to_token_reduce_regular_backward,
    _run_atom_to_token_reduce_regular_forward,
    _run_atom_to_token_reduce_sorted_vjp,
    _run_batched_atom_to_token_reduce_backward,
    _run_batched_atom_to_token_reduce_forward,
    _run_batched_atom_to_token_reduce_mask_backward,
    _run_nested_atom_to_token_reduce_backward,
    _run_nested_atom_to_token_reduce_forward,
    _run_nested_atom_to_token_reduce_mask_backward,
)
from .schedule import AtomToTokenReductionPlan


def triton_prepared_atom_to_token_reduce(
    atom: Tensor,
    atom_mask: Tensor | None,
    mapping: Tensor,
    permutation: Tensor | None,
    inverse_permutation: Tensor | None,
    layout_plan: AtomTokenPlan | None,
    denominator_offset: float,
    normalize: bool,
    token_offsets: Tensor | None = None,
    original_mapping: Tensor | None = None,
) -> Tensor | None:
    if (
        layout_plan is None
        or is_compiling()
        or not _plain_atom_call(
            atom, atom_mask, mapping, permutation, inverse_permutation, token_offsets, original_mapping
        )
    ):
        return None
    launch_plan = _bind_prepared_atom_to_token_plan(atom, atom_mask, mapping, layout_plan)
    if launch_plan is None:
        return None
    offset = float(denominator_offset)
    native_output = _native_prepared_atom_to_token_reduce(
        atom, atom_mask, original_mapping, layout_plan, launch_plan, offset, normalize
    )
    if native_output is not None:
        return native_output
    if layout_plan.kind == "regular":
        if tensors_need_grad(atom, atom_mask):
            return _EagerRegularAtomToTokenReduction.apply(
                atom,
                atom_mask,
                layout_plan.num_tokens,
                layout_plan.width,
                offset,
                normalize,
                (layout_plan, launch_plan),
            )
        return _launch_atom_to_token_reduce_regular_forward(atom, atom_mask, launch_plan, offset, normalize)
    token_offsets = _compatible_atom_to_token_offsets(token_offsets, mapping, layout_plan.num_tokens)
    shape = atom.shape
    atom_flat = atom if atom.dim() == 3 else atom.reshape(launch_plan.flat_atom_shape)
    if tensors_need_grad(atom, atom_mask):
        out = _SortedAtomToTokenReduction.apply(
            atom_flat,
            atom_mask,
            mapping,
            permutation,
            inverse_permutation,
            layout_plan,
            offset,
            normalize,
            launch_plan,
            token_offsets,
        )
    else:
        out = _atom_to_token_reduce_sorted_forward(
            atom_flat,
            atom_mask,
            mapping,
            permutation,
            layout_plan.num_tokens,
            layout_plan.width,
            offset,
            normalize,
            launch_plan,
            token_offsets,
        )
    return out if len(shape) == 3 else out.reshape(*shape[:-2], layout_plan.num_tokens, shape[-1])


def triton_atom_to_token_reduce(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    permutation: Tensor | None,
    inverse_permutation: Tensor | None,
    plan: AtomTokenPlan,
    denominator_offset: float,
    normalize: bool,
    token_offsets: Tensor | None = None,
) -> Tensor:

    if plan.kind == "regular":
        if not is_compiling() and _plain_atom_call(atom, atom_mask) and tensors_need_grad(atom, atom_mask):
            return _EagerRegularAtomToTokenReduction.apply(
                atom, atom_mask, plan.num_tokens, plan.width, float(denominator_offset), normalize
            )
        compiling = is_compiling()
        atom_type = type(atom)
        external_fake = (
            not compiling and atom_type is not Tensor and atom_type is not torch.nn.Parameter and is_external_fake(atom)
        )
        if not tensors_need_grad(atom, atom_mask) and not compiling and not external_fake:
            launch_plan = _resolve_atom_to_token_plan(atom, atom_mask, plan)
            return _launch_atom_to_token_reduce_regular_forward(
                atom,
                atom_mask,
                launch_plan,
                float(denominator_offset),
                normalize,
            )
        return _atom_to_token_reduce_regular_forward(
            atom,
            atom_mask,
            plan.num_tokens,
            plan.width,
            float(denominator_offset),
            normalize,
        )
    if plan.kind == "interleaved":
        return _triton_atom_to_token_reduce_interleaved(
            atom,
            atom_mask,
            plan,
            float(denominator_offset),
            normalize,
        )
    return _triton_atom_to_token_reduce_sorted(
        atom,
        atom_mask,
        atom_to_token,
        permutation,
        inverse_permutation,
        plan,
        float(denominator_offset),
        normalize,
        _compatible_atom_to_token_offsets(token_offsets, atom_to_token, plan.num_tokens),
    )


def triton_atom_to_token_reduce_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    permutation: Tensor | None,
    inverse_permutation: Tensor | None,
    plan: AtomTokenPlan,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    atoms = atom_to_token.numel()
    if plan.kind == "regular":
        return _atom_to_token_reduce_regular_backward(
            grad_out,
            atom_mask,
            atoms,
            plan.width,
            float(denominator_offset),
            normalize,
        )
    grad_out_flat, _batch, output_shape = _flatten_grad_out(grad_out, plan.num_tokens, atoms)
    if plan.kind == "interleaved":
        grad_atom = _atom_to_token_reduce_interleaved_backward(
            grad_out_flat,
            atom_mask,
            plan.num_tokens,
            plan.width,
            float(denominator_offset),
            normalize,
        )
    else:
        grad_atom = _atom_to_token_reduce_sorted_backward(
            grad_out_flat,
            atom_mask,
            atom_to_token,
            permutation,
            inverse_permutation,
            plan.num_tokens,
            plan.width,
            float(denominator_offset),
            normalize,
        )
    return grad_atom.reshape(output_shape)


@torch.library.custom_op("ooperators::_atom_to_token_reduce_mask_backward", mutates_args=(), device_types="cuda")
def _atom_to_token_reduce_mask_backward(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor | None,
    grad_out: Tensor,
    num_tokens: int,
    plan_kind: int,
    width: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    return _run_atom_to_token_reduce_mask_backward(
        atom, atom_mask, atom_to_token, grad_out, num_tokens, plan_kind, width, denominator_offset, normalize
    )


@torch.library.register_fake(_atom_to_token_reduce_mask_backward)
def _atom_to_token_reduce_mask_backward_fake(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor | None,
    grad_out: Tensor,
    num_tokens: int,
    plan_kind: int,
    width: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    del atom, atom_to_token, grad_out, num_tokens, plan_kind, width, denominator_offset, normalize
    return torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)


def triton_atom_to_token_reduce_mask_backward(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor | None,
    permutation: Tensor | None,
    inverse_permutation: Tensor | None,
    grad_out: Tensor,
    plan: AtomTokenPlan,
    denominator_offset: float,
    normalize: bool,
    semantic: bool = True,
    launch_plan: AtomToTokenReductionPlan | None = None,
) -> Tensor:

    plan_kind = {
        "regular": _MASK_VJP_REGULAR,
        "interleaved": _MASK_VJP_INTERLEAVED,
        "sorted": _MASK_VJP_SORTED,
    }[plan.kind]
    if not semantic and not torch.is_grad_enabled():
        return _run_atom_to_token_reduce_mask_backward(
            atom,
            atom_mask,
            atom_to_token if plan.kind == "sorted" else None,
            grad_out,
            plan.num_tokens,
            plan_kind,
            plan.width,
            float(denominator_offset),
            normalize,
            permutation,
            launch_plan,
        )
    if permutation is not None:
        atom = atom.index_select(-2, permutation)
        atom_mask = _gather_atom_mask(atom_mask, permutation)
    backward = (
        _atom_to_token_reduce_mask_backward
        if semantic or torch.is_grad_enabled()
        else _run_atom_to_token_reduce_mask_backward
    )
    grad_mask = backward(
        atom,
        atom_mask,
        atom_to_token if plan.kind == "sorted" else None,
        grad_out,
        plan.num_tokens,
        plan_kind,
        plan.width,
        float(denominator_offset),
        normalize,
    )
    if inverse_permutation is None:
        return grad_mask
    return grad_mask.index_select(-1, inverse_permutation)


@torch.library.custom_op("ooperators::_batched_atom_to_token_reduce_forward", mutates_args=(), device_types="cuda")
def _batched_atom_to_token_reduce_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    num_tokens: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _run_batched_atom_to_token_reduce_forward(
        atom, atom_mask, atom_to_token, num_tokens, mapping_batch_repeat, denominator_offset, normalize
    )


@torch.library.register_fake(_batched_atom_to_token_reduce_forward)
def _batched_atom_to_token_reduce_forward_fake(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    num_tokens: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del atom_mask, mapping_batch_repeat, denominator_offset, normalize
    mapping_batch = atom_to_token.numel() // atom.shape[-2]
    return (
        torch.empty(
            (*atom.shape[:-2], num_tokens, atom.shape[-1]),
            device=atom.device,
            dtype=atom.dtype,
        ),
        torch.empty((mapping_batch,), device=atom.device, dtype=torch.int32),
        torch.empty((mapping_batch, atom.shape[-2]), device=atom.device, dtype=torch.int64),
        torch.empty((mapping_batch, atom.shape[-2]), device=atom.device, dtype=torch.int64),
    )


@torch.library.custom_op("ooperators::_batched_atom_to_token_reduce_backward", mutates_args=(), device_types="cuda")
def _batched_atom_to_token_reduce_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    mapping_flags: Tensor,
    sorted_mapping: Tensor,
    permutation: Tensor,
    atoms: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    return _run_batched_atom_to_token_reduce_backward(
        grad_out,
        atom_mask,
        atom_to_token,
        mapping_flags,
        sorted_mapping,
        permutation,
        atoms,
        mapping_batch_repeat,
        denominator_offset,
        normalize,
    )


@torch.library.register_fake(_batched_atom_to_token_reduce_backward)
def _batched_atom_to_token_reduce_backward_fake(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    mapping_flags: Tensor,
    sorted_mapping: Tensor,
    permutation: Tensor,
    atoms: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    del atom_mask, atom_to_token, mapping_flags, mapping_batch_repeat, denominator_offset, normalize
    return torch.empty(
        (*grad_out.shape[:-2], atoms, grad_out.shape[-1]),
        device=grad_out.device,
        dtype=grad_out.dtype,
    )


@torch.library.custom_op(
    "ooperators::_batched_atom_to_token_reduce_mask_backward",
    mutates_args=(),
    device_types="cuda",
)
def _batched_atom_to_token_reduce_mask_backward(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor,
    mapping_flags: Tensor,
    sorted_mapping: Tensor,
    permutation: Tensor,
    grad_out: Tensor,
    num_tokens: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    return _run_batched_atom_to_token_reduce_mask_backward(
        atom,
        atom_mask,
        atom_to_token,
        mapping_flags,
        sorted_mapping,
        permutation,
        grad_out,
        num_tokens,
        mapping_batch_repeat,
        denominator_offset,
        normalize,
    )


@torch.library.register_fake(_batched_atom_to_token_reduce_mask_backward)
def _batched_atom_to_token_reduce_mask_backward_fake(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor,
    mapping_flags: Tensor,
    sorted_mapping: Tensor,
    permutation: Tensor,
    grad_out: Tensor,
    num_tokens: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    del (
        atom,
        atom_to_token,
        mapping_flags,
        grad_out,
        num_tokens,
        mapping_batch_repeat,
        denominator_offset,
        normalize,
    )
    return torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)


def _batched_atom_to_token_reduce_setup_context(ctx, inputs, output) -> None:
    atom, atom_mask, atom_to_token, num_tokens, mapping_batch_repeat, denominator_offset, normalize = inputs
    _result, mapping_flags, sorted_mapping, permutation = output
    ctx.semantic = True
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(mapping_flags, sorted_mapping, permutation)
    need_grad_mask = atom_mask is not None and ctx.needs_input_grad[1]
    ctx.save_for_backward(
        atom_mask,
        atom_to_token,
        atom if need_grad_mask else None,
        mapping_flags,
        sorted_mapping,
        permutation,
    )
    ctx.need_grad_atom = ctx.needs_input_grad[0]
    ctx.need_grad_mask = need_grad_mask
    ctx.atoms = atom.shape[-2]
    ctx.num_tokens = num_tokens
    ctx.mapping_batch_repeat = mapping_batch_repeat
    ctx.denominator_offset = denominator_offset
    ctx.normalize = normalize


def _batched_atom_to_token_reduce_autograd_backward(
    ctx,
    grad_out: Tensor,
    grad_mapping_flags: Tensor | None,
    grad_sorted_mapping: Tensor | None,
    grad_permutation: Tensor | None,
):
    del grad_mapping_flags, grad_sorted_mapping, grad_permutation
    atom_mask, atom_to_token, atom, mapping_flags, sorted_mapping, permutation = ctx.saved_tensors
    semantic = ctx.semantic or torch.is_grad_enabled()
    atom_backward = _batched_atom_to_token_reduce_backward if semantic else _run_batched_atom_to_token_reduce_backward
    mask_backward = (
        _batched_atom_to_token_reduce_mask_backward if semantic else _run_batched_atom_to_token_reduce_mask_backward
    )
    grad_atom = None
    if ctx.need_grad_atom:
        grad_atom = atom_backward(
            grad_out,
            atom_mask,
            atom_to_token,
            mapping_flags,
            sorted_mapping,
            permutation,
            ctx.atoms,
            ctx.mapping_batch_repeat,
            ctx.denominator_offset,
            ctx.normalize,
        )
    grad_mask = None
    if ctx.need_grad_mask:
        grad_mask = mask_backward(
            atom,
            atom_mask,
            atom_to_token,
            mapping_flags,
            sorted_mapping,
            permutation,
            grad_out,
            ctx.num_tokens,
            ctx.mapping_batch_repeat,
            ctx.denominator_offset,
            ctx.normalize,
        )
    return grad_atom, grad_mask, None, None, None, None, None


torch.library.register_autograd(
    _batched_atom_to_token_reduce_forward,
    _batched_atom_to_token_reduce_autograd_backward,
    setup_context=_batched_atom_to_token_reduce_setup_context,
)


class _BatchedAtomToTokenReduction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atom, atom_mask, atom_to_token, num_tokens, mapping_batch_repeat, denominator_offset, normalize):
        arguments = (atom, atom_mask, atom_to_token, num_tokens, mapping_batch_repeat, denominator_offset, normalize)
        output = _run_batched_atom_to_token_reduce_forward(*arguments)
        _batched_atom_to_token_reduce_setup_context(ctx, arguments, output)
        ctx.semantic = False
        return output

    @staticmethod
    def backward(ctx, grad_out, grad_flags, grad_sorted, grad_permutation):
        return _batched_atom_to_token_reduce_autograd_backward(ctx, grad_out, grad_flags, grad_sorted, grad_permutation)


def triton_batched_atom_to_token_reduce(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    *,
    num_tokens: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:

    compiling = is_compiling()
    atom_type = type(atom)
    external_fake = (
        not compiling and atom_type is not Tensor and atom_type is not torch.nn.Parameter and is_external_fake(atom)
    )
    arguments = (
        atom,
        atom_mask,
        atom_to_token,
        int(num_tokens),
        int(mapping_batch_repeat),
        float(denominator_offset),
        normalize,
    )
    if not compiling and not external_fake and _plain_atom_call(atom, atom_mask, atom_to_token):
        if tensors_need_grad(atom, atom_mask):
            return _BatchedAtomToTokenReduction.apply(*arguments)[0]
        return _run_batched_atom_to_token_reduce_forward(*arguments)[0]
    return _batched_atom_to_token_reduce_forward(*arguments)[0]


def triton_shared_general_atom_to_token_reduce(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    *,
    num_tokens: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:

    batch = math.prod(atom.shape[:-2]) if atom.dim() > 2 else 1
    return triton_batched_atom_to_token_reduce(
        atom,
        atom_mask,
        atom_to_token,
        num_tokens=int(num_tokens),
        mapping_batch_repeat=batch,
        denominator_offset=float(denominator_offset),
        normalize=normalize,
    )


def triton_shared_general_atom_to_token_reduce_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    *,
    atoms: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:

    batch, mapping_flags, sorted_mapping, permutation = _prepare_shared_general_backward_mapping(
        grad_out, atom_mask, atom_to_token, atoms
    )
    return _batched_atom_to_token_reduce_backward(
        grad_out,
        atom_mask,
        atom_to_token,
        mapping_flags,
        sorted_mapping,
        permutation,
        int(atoms),
        batch,
        float(denominator_offset),
        normalize,
    )


def triton_shared_general_atom_to_token_reduce_mask_backward(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor,
    grad_out: Tensor,
    *,
    num_tokens: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:

    batch, mapping_flags, sorted_mapping, permutation = _prepare_shared_general_mask_backward_mapping(
        atom, atom_mask, atom_to_token, num_tokens
    )
    return _batched_atom_to_token_reduce_mask_backward(
        atom,
        atom_mask,
        atom_to_token,
        mapping_flags,
        sorted_mapping,
        permutation,
        grad_out,
        int(num_tokens),
        batch,
        float(denominator_offset),
        normalize,
    )


def _triton_atom_to_token_reduce_sorted(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    permutation: Tensor | None,
    inverse_permutation: Tensor | None,
    plan: AtomTokenPlan,
    denominator_offset: float,
    normalize: bool,
    token_offsets: Tensor | None = None,
) -> Tensor:
    input_shape = tuple(atom.shape)
    atoms = input_shape[-2]
    channels = input_shape[-1]
    batch = math.prod(input_shape[:-2]) if len(input_shape) > 2 else 1
    atom_flat = atom.reshape(batch, atoms, channels)
    if tensors_need_grad(atom, atom_mask):
        out = _SortedAtomToTokenReduction.apply(
            atom_flat,
            atom_mask,
            atom_to_token,
            permutation,
            inverse_permutation,
            plan,
            float(denominator_offset),
            normalize,
            None,
            token_offsets,
        )
    else:
        out = _atom_to_token_reduce_sorted_forward(
            atom_flat,
            atom_mask,
            atom_to_token,
            permutation,
            plan.num_tokens,
            plan.width,
            float(denominator_offset),
            normalize,
            None,
            token_offsets,
        )
    return out.reshape(*input_shape[:-2], plan.num_tokens, channels)


def _triton_atom_to_token_reduce_interleaved(
    atom: Tensor,
    atom_mask: Tensor | None,
    plan: AtomTokenPlan,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    input_shape = tuple(atom.shape)
    atoms = input_shape[-2]
    channels = input_shape[-1]
    batch = math.prod(input_shape[:-2]) if len(input_shape) > 2 else 1
    atom_flat = atom.reshape(batch, atoms, channels)
    if tensors_need_grad(atom, atom_mask):
        out = _InterleavedAtomToTokenReduction.apply(
            atom_flat,
            atom_mask,
            plan,
            float(denominator_offset),
            normalize,
        )
    else:
        out = _atom_to_token_reduce_interleaved_forward(
            atom_flat,
            atom_mask,
            plan.num_tokens,
            plan.width,
            float(denominator_offset),
            normalize,
        )
    return out.reshape(*input_shape[:-2], plan.num_tokens, channels)


@torch.library.custom_op("ooperators::_atom_to_token_reduce_regular_forward", mutates_args=(), device_types="cuda")
def _atom_to_token_reduce_regular_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    num_tokens: int,
    atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    return _run_atom_to_token_reduce_regular_forward(
        atom, atom_mask, num_tokens, atoms_per_token, denominator_offset, normalize
    )


@torch.library.custom_op("ooperators::_atom_to_token_reduce_regular_backward", mutates_args=(), device_types="cuda")
def _atom_to_token_reduce_regular_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atoms: int,
    atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    return _run_atom_to_token_reduce_regular_backward(
        grad_out, atom_mask, atoms, atoms_per_token, denominator_offset, normalize
    )


@torch.library.register_fake(_atom_to_token_reduce_regular_forward)
def _atom_to_token_reduce_regular_forward_fake(
    atom: Tensor,
    atom_mask: Tensor | None,
    num_tokens: int,
    atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    del atom_mask, atoms_per_token, denominator_offset, normalize
    return torch.empty(
        (*atom.shape[:-2], num_tokens, atom.shape[-1]),
        device=atom.device,
        dtype=atom.dtype,
    )


@torch.library.register_fake(_atom_to_token_reduce_regular_backward)
def _atom_to_token_reduce_regular_backward_fake(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atoms: int,
    atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    del atom_mask, atoms_per_token, denominator_offset, normalize
    return torch.empty(
        (*grad_out.shape[:-2], atoms, grad_out.shape[-1]),
        device=grad_out.device,
        dtype=grad_out.dtype,
    )


def _atom_to_token_reduce_regular_setup_context(
    ctx,
    inputs: tuple[Tensor, Tensor | None, int, int, float, bool],
    output: Tensor,
) -> None:
    atom, atom_mask, num_tokens, atoms_per_token, denominator_offset, normalize = inputs
    need_grad_mask = atom_mask is not None and ctx.needs_input_grad[1]
    ctx.save_for_backward(atom_mask, atom if need_grad_mask else None)
    ctx.need_grad_atom = ctx.needs_input_grad[0]
    ctx.need_grad_mask = need_grad_mask
    ctx.atoms = atom.shape[-2]
    ctx.num_tokens = num_tokens
    ctx.atoms_per_token = atoms_per_token
    ctx.denominator_offset = denominator_offset
    ctx.normalize = normalize


def _atom_to_token_reduce_regular_autograd_backward(ctx, grad_out: Tensor):
    atom_mask, atom = ctx.saved_tensors
    semantic = getattr(ctx, "semantic", True) or torch.is_grad_enabled()
    launch_plan = None if semantic else getattr(ctx, "launch_plan", None)
    if (
        not semantic
        and ctx.need_grad_atom
        and ctx.need_grad_mask
        and launch_plan is not None
        and grad_out.dtype == launch_plan.problem.dtype
        and launch_plan.problem.channels <= launch_plan.block_channels
    ):
        grad_atom, grad_mask = _launch_atom_to_token_reduce_regular_vjp(
            atom, atom_mask, grad_out, launch_plan, ctx.denominator_offset, ctx.normalize
        )
        return grad_atom, grad_mask, None, None, None, None
    grad_atom = None
    if ctx.need_grad_atom:
        if semantic:
            grad_atom = _atom_to_token_reduce_regular_backward(
                grad_out, atom_mask, ctx.atoms, ctx.atoms_per_token, ctx.denominator_offset, ctx.normalize
            )
        else:
            grad_atom = _run_atom_to_token_reduce_regular_backward(
                grad_out, atom_mask, ctx.atoms, ctx.atoms_per_token, ctx.denominator_offset, ctx.normalize, launch_plan
            )
    grad_mask = None
    if ctx.need_grad_mask:
        plan = getattr(ctx, "layout_plan", None)
        if plan is None:
            plan = AtomTokenPlan("regular", ctx.num_tokens, ctx.atoms_per_token)
        grad_mask = triton_atom_to_token_reduce_mask_backward(
            atom,
            atom_mask,
            None,
            None,
            None,
            grad_out,
            plan,
            ctx.denominator_offset,
            ctx.normalize,
            semantic,
            launch_plan,
        )
    return grad_atom, grad_mask, None, None, None, None


torch.library.register_autograd(
    _atom_to_token_reduce_regular_forward,
    _atom_to_token_reduce_regular_autograd_backward,
    setup_context=_atom_to_token_reduce_regular_setup_context,
)


class _EagerRegularAtomToTokenReduction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atom, atom_mask, num_tokens, atoms_per_token, denominator_offset, normalize, prepared_plan=None):
        if prepared_plan is None:
            layout = AtomTokenPlan("regular", num_tokens, atoms_per_token)
            launch = _resolve_atom_to_token_plan(atom, atom_mask, layout)
        else:
            layout, launch = prepared_plan
        output = _launch_atom_to_token_reduce_regular_forward(atom, atom_mask, launch, denominator_offset, normalize)
        _atom_to_token_reduce_regular_setup_context(
            ctx, (atom, atom_mask, num_tokens, atoms_per_token, denominator_offset, normalize), output
        )
        ctx.semantic = False
        ctx.layout_plan = layout
        ctx.launch_plan = launch
        return output

    @staticmethod
    def backward(ctx, grad_out):
        gradients = _atom_to_token_reduce_regular_autograd_backward(ctx, grad_out)
        return gradients + ((None,) if len(ctx.needs_input_grad) == 7 else ())


class _SortedAtomToTokenReduction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        atom: Tensor,
        atom_mask: Tensor | None,
        atom_to_token: Tensor,
        permutation: Tensor | None,
        inverse_permutation: Tensor | None,
        plan: AtomTokenPlan,
        denominator_offset: float,
        normalize: bool,
        launch_plan: AtomToTokenReductionPlan | None = None,
        token_offsets: Tensor | None = None,
    ) -> Tensor:
        need_grad_mask = atom_mask is not None and ctx.needs_input_grad[1]
        semantic = launch_plan is None and (is_compiling() or not _plain_atom_call(atom, atom_mask))
        if launch_plan is None and not semantic:
            launch_plan = _resolve_atom_to_token_plan(atom, atom_mask, plan)
        output = _atom_to_token_reduce_sorted_forward(
            atom,
            atom_mask,
            atom_to_token,
            permutation,
            plan.num_tokens,
            plan.width,
            denominator_offset,
            normalize,
            launch_plan,
            token_offsets,
        )
        ctx.save_for_backward(
            atom_mask,
            atom_to_token,
            permutation,
            inverse_permutation,
            atom if need_grad_mask else None,
            token_offsets,
        )
        ctx.need_grad_atom = ctx.needs_input_grad[0]
        ctx.need_grad_mask = need_grad_mask
        ctx.semantic = semantic
        ctx.launch_plan = launch_plan
        ctx.plan = plan
        ctx.denominator_offset = denominator_offset
        ctx.normalize = normalize
        return output

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        atom_mask, atom_to_token, permutation, inverse_permutation, atom, token_offsets = ctx.saved_tensors
        launch_plan = None if ctx.semantic or torch.is_grad_enabled() else ctx.launch_plan
        if ctx.need_grad_mask and launch_plan is not None and grad_out.dtype == launch_plan.problem.dtype:
            grad_atom, grad_mask = _run_atom_to_token_reduce_sorted_vjp(
                atom,
                atom_mask,
                atom_to_token,
                permutation,
                inverse_permutation,
                grad_out,
                ctx.plan,
                ctx.denominator_offset,
                ctx.normalize,
                ctx.need_grad_atom,
                launch_plan,
                token_offsets,
            )
            gradients = (grad_atom, grad_mask, None, None, None, None, None, None)
            return gradients + (None,) * (len(ctx.needs_input_grad) - len(gradients))
        grad_atom = None
        if ctx.need_grad_atom:
            grad_atom = _atom_to_token_reduce_sorted_backward(
                grad_out.contiguous(),
                atom_mask,
                atom_to_token,
                permutation,
                inverse_permutation,
                ctx.plan.num_tokens,
                ctx.plan.width,
                ctx.denominator_offset,
                ctx.normalize,
                launch_plan,
                token_offsets,
            )
        grad_mask = None
        if ctx.need_grad_mask:
            grad_mask = triton_atom_to_token_reduce_mask_backward(
                atom,
                atom_mask,
                atom_to_token,
                permutation,
                inverse_permutation,
                grad_out,
                ctx.plan,
                ctx.denominator_offset,
                ctx.normalize,
                ctx.semantic,
                launch_plan,
            )
        gradients = (grad_atom, grad_mask, None, None, None, None, None, None)
        return gradients + (None,) * (len(ctx.needs_input_grad) - len(gradients))


class _InterleavedAtomToTokenReduction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        atom: Tensor,
        atom_mask: Tensor | None,
        plan: AtomTokenPlan,
        denominator_offset: float,
        normalize: bool,
    ) -> Tensor:
        need_grad_mask = atom_mask is not None and ctx.needs_input_grad[1]
        output = _atom_to_token_reduce_interleaved_forward(
            atom,
            atom_mask,
            plan.num_tokens,
            plan.width,
            denominator_offset,
            normalize,
        )
        ctx.save_for_backward(atom_mask, atom if need_grad_mask else None)
        ctx.need_grad_atom = ctx.needs_input_grad[0]
        ctx.need_grad_mask = need_grad_mask
        ctx.semantic = is_compiling() or not _plain_atom_call(atom, atom_mask)
        ctx.plan = plan
        ctx.denominator_offset = denominator_offset
        ctx.normalize = normalize
        return output

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        atom_mask, atom = ctx.saved_tensors
        grad_atom = None
        if ctx.need_grad_atom:
            grad_atom = _atom_to_token_reduce_interleaved_backward(
                grad_out.contiguous(),
                atom_mask,
                ctx.plan.num_tokens,
                ctx.plan.width,
                ctx.denominator_offset,
                ctx.normalize,
            )

        grad_mask = None
        if ctx.need_grad_mask:
            grad_mask = triton_atom_to_token_reduce_mask_backward(
                atom,
                atom_mask,
                None,
                None,
                None,
                grad_out,
                ctx.plan,
                ctx.denominator_offset,
                ctx.normalize,
                ctx.semantic,
            )
        return grad_atom, grad_mask, None, None, None


_NESTED_ATOM_TO_TOKEN_FORWARD_SCHEMA = (
    "(Tensor atom, Tensor? atom_mask, Tensor atom_to_token, Tensor atom_offsets, "
    "Tensor token_offsets, Tensor atom_batch, Tensor token_batch, float denominator_offset, bool normalize) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.custom_op(
    "ooperators::_nested_atom_to_token_reduce_forward",
    mutates_args=(),
    device_types="cuda",
    schema=_NESTED_ATOM_TO_TOKEN_FORWARD_SCHEMA,
)
def _nested_atom_to_token_reduce_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    atom_batch: Tensor,
    token_batch: Tensor,
    denominator_offset: float,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _run_nested_atom_to_token_reduce_forward(
        atom,
        atom_mask,
        atom_to_token,
        atom_offsets,
        token_offsets,
        atom_batch,
        token_batch,
        denominator_offset,
        normalize,
    )


@torch.library.register_fake(_nested_atom_to_token_reduce_forward)
def _nested_atom_to_token_reduce_forward_fake(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    atom_batch: Tensor,
    token_batch: Tensor,
    denominator_offset: float,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del atom_mask, atom_to_token, token_offsets, atom_batch, denominator_offset, normalize
    tokens = token_batch.shape[0]
    return (
        torch.empty((tokens, *atom.shape[1:]), device=atom.device, dtype=atom.dtype),
        torch.empty((atom_offsets.shape[0] - 1,), device=atom.device, dtype=torch.int32),
        torch.empty((tokens,), device=atom.device, dtype=torch.int64),
        torch.empty((tokens,), device=atom.device, dtype=torch.int32),
        torch.empty((atom.shape[0],), device=atom.device, dtype=torch.int64),
    )


_NESTED_ATOM_TO_TOKEN_BACKWARD_SCHEMA = (
    "(Tensor grad_out, Tensor? atom_mask, Tensor atom_to_token, Tensor permutation, "
    "Tensor atom_offsets, Tensor token_offsets, "
    "Tensor token_batch, Tensor sample_kind, Tensor segment_start, Tensor token_count, "
    "SymInt total_atoms, float denominator_offset, bool normalize) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_nested_atom_to_token_reduce_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_NESTED_ATOM_TO_TOKEN_BACKWARD_SCHEMA,
)
def _nested_atom_to_token_reduce_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    permutation: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    token_batch: Tensor,
    sample_kind: Tensor,
    segment_start: Tensor,
    token_count: Tensor,
    total_atoms: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    return _run_nested_atom_to_token_reduce_backward(
        grad_out,
        atom_mask,
        atom_to_token,
        permutation,
        atom_offsets,
        token_offsets,
        token_batch,
        sample_kind,
        segment_start,
        token_count,
        total_atoms,
        denominator_offset,
        normalize,
    )


@torch.library.register_fake(_nested_atom_to_token_reduce_backward)
def _nested_atom_to_token_reduce_backward_fake(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    permutation: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    token_batch: Tensor,
    sample_kind: Tensor,
    segment_start: Tensor,
    token_count: Tensor,
    total_atoms: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    del (
        atom_mask,
        atom_to_token,
        atom_offsets,
        token_offsets,
        token_batch,
        sample_kind,
        segment_start,
        token_count,
        denominator_offset,
        normalize,
    )
    return torch.empty((total_atoms, *grad_out.shape[1:]), device=grad_out.device, dtype=grad_out.dtype)


_NESTED_ATOM_TO_TOKEN_MASK_BACKWARD_SCHEMA = (
    "(Tensor atom, Tensor atom_mask, Tensor atom_to_token, Tensor permutation, Tensor grad_out, Tensor atom_offsets, "
    "Tensor token_offsets, Tensor token_batch, Tensor sample_kind, Tensor segment_start, "
    "Tensor token_count, float denominator_offset, bool normalize) -> Tensor"
)


@torch.library.custom_op(
    "ooperators::_nested_atom_to_token_reduce_mask_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_NESTED_ATOM_TO_TOKEN_MASK_BACKWARD_SCHEMA,
)
def _nested_atom_to_token_reduce_mask_backward(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor,
    permutation: Tensor,
    grad_out: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    token_batch: Tensor,
    sample_kind: Tensor,
    segment_start: Tensor,
    token_count: Tensor,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    return _run_nested_atom_to_token_reduce_mask_backward(
        atom,
        atom_mask,
        atom_to_token,
        permutation,
        grad_out,
        atom_offsets,
        token_offsets,
        token_batch,
        sample_kind,
        segment_start,
        token_count,
        denominator_offset,
        normalize,
    )


@torch.library.register_fake(_nested_atom_to_token_reduce_mask_backward)
def _nested_atom_to_token_reduce_mask_backward_fake(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor,
    permutation: Tensor,
    grad_out: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    token_batch: Tensor,
    sample_kind: Tensor,
    segment_start: Tensor,
    token_count: Tensor,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    del (
        atom,
        atom_to_token,
        grad_out,
        atom_offsets,
        token_offsets,
        token_batch,
        sample_kind,
        segment_start,
        token_count,
        denominator_offset,
        normalize,
    )
    return torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)


def _nested_atom_to_token_reduce_setup_context(ctx, inputs, output) -> None:
    (
        atom,
        atom_mask,
        mapping,
        atom_offsets,
        token_offsets,
        _atom_batch,
        token_batch,
        denominator_offset,
        normalize,
    ) = inputs
    _result, sample_kind, segment_start, token_count, permutation = output
    ctx.semantic = True
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(sample_kind, segment_start, token_count, permutation)
    need_grad_mask = atom_mask is not None and ctx.needs_input_grad[1]
    ctx.save_for_backward(
        atom_mask,
        atom if need_grad_mask else None,
        mapping,
        permutation,
        atom_offsets,
        token_offsets,
        token_batch,
        sample_kind,
        segment_start,
        token_count,
    )
    ctx.need_grad_atom = ctx.needs_input_grad[0]
    ctx.need_grad_mask = need_grad_mask
    ctx.total_atoms = atom.shape[0]
    ctx.denominator_offset = denominator_offset
    ctx.normalize = normalize


def _nested_atom_to_token_reduce_autograd_backward(
    ctx,
    grad_out: Tensor,
    grad_sample_kind: Tensor | None,
    grad_segment_start: Tensor | None,
    grad_token_count: Tensor | None,
    grad_permutation: Tensor | None,
):
    del grad_sample_kind, grad_segment_start, grad_token_count, grad_permutation
    (
        atom_mask,
        atom,
        mapping,
        permutation,
        atom_offsets,
        token_offsets,
        token_batch,
        sample_kind,
        segment_start,
        token_count,
    ) = ctx.saved_tensors
    semantic = ctx.semantic or torch.is_grad_enabled()
    atom_backward = _nested_atom_to_token_reduce_backward if semantic else _run_nested_atom_to_token_reduce_backward
    mask_backward = (
        _nested_atom_to_token_reduce_mask_backward if semantic else _run_nested_atom_to_token_reduce_mask_backward
    )
    grad_atom = None
    if ctx.need_grad_atom:
        grad_atom = atom_backward(
            grad_out,
            atom_mask,
            mapping,
            permutation,
            atom_offsets,
            token_offsets,
            token_batch,
            sample_kind,
            segment_start,
            token_count,
            ctx.total_atoms,
            ctx.denominator_offset,
            ctx.normalize,
        )
    grad_mask = None
    if ctx.need_grad_mask:
        grad_mask = mask_backward(
            atom,
            atom_mask,
            mapping,
            permutation,
            grad_out,
            atom_offsets,
            token_offsets,
            token_batch,
            sample_kind,
            segment_start,
            token_count,
            ctx.denominator_offset,
            ctx.normalize,
        )
    return grad_atom, grad_mask, None, None, None, None, None, None, None


torch.library.register_autograd(
    _nested_atom_to_token_reduce_forward,
    _nested_atom_to_token_reduce_autograd_backward,
    setup_context=_nested_atom_to_token_reduce_setup_context,
)


class _NestedAtomToTokenReduction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        atom,
        atom_mask,
        atom_to_token,
        atom_offsets,
        token_offsets,
        atom_batch,
        token_batch,
        denominator_offset,
        normalize,
    ):
        arguments = (
            atom,
            atom_mask,
            atom_to_token,
            atom_offsets,
            token_offsets,
            atom_batch,
            token_batch,
            denominator_offset,
            normalize,
        )
        output = _run_nested_atom_to_token_reduce_forward(*arguments)
        _nested_atom_to_token_reduce_setup_context(ctx, arguments, output)
        ctx.semantic = False
        return output

    @staticmethod
    def backward(ctx, grad_out, grad_kind, grad_start, grad_count, grad_permutation):
        return _nested_atom_to_token_reduce_autograd_backward(
            ctx, grad_out, grad_kind, grad_start, grad_count, grad_permutation
        )


def triton_nested_atom_to_token_reduce(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    atom_batch: Tensor,
    token_batch: Tensor,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:

    arguments = (
        atom,
        atom_mask,
        atom_to_token,
        atom_offsets,
        token_offsets,
        atom_batch,
        token_batch,
        float(denominator_offset),
        normalize,
    )
    if not is_compiling() and _plain_atom_call(
        atom, atom_mask, atom_to_token, atom_offsets, token_offsets, atom_batch, token_batch
    ):
        if tensors_need_grad(atom, atom_mask):
            return _NestedAtomToTokenReduction.apply(*arguments)[0]
        return _run_nested_atom_to_token_reduce_forward(*arguments)[0]
    return _nested_atom_to_token_reduce_forward(*arguments)[0]
