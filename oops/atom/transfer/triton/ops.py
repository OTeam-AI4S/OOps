from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from typing import overload

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.core._math import _next_power_of_two
from oops.runtime import (
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps,
    cuda_device_caps_if_needed,
    cuda_device_context,
    forward_mode_active,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
)
from oops.runtime import needs_grad as tensors_need_grad
from oops.runtime import shared_memory_budget, tensors_share_device
from oops.runtime.dispatch import _kernel_operand

from ...layout import AtomTokenPlan
from . import kernel
from .profiles import ATOM_TO_TOKEN_REDUCTION, BATCHED_ATOM_TO_TOKEN_REDUCTION, NESTED_ATOM_TO_TOKEN_REDUCTION
from .schedule import (
    AtomToTokenReductionKind,
    AtomToTokenReductionPlan,
    AtomToTokenReductionProblem,
    AtomToTokenReductionSchedule,
    BatchedAtomToTokenReductionPlan,
    BatchedAtomToTokenReductionProblem,
    NestedAtomToTokenReductionPlan,
    NestedAtomToTokenReductionProblem,
)

_MASK_VJP_REGULAR = 0
_MASK_VJP_INTERLEAVED = 1
_MASK_VJP_SORTED = 2
_MASK_VJP_KINDS: dict[int, AtomToTokenReductionKind] = {
    _MASK_VJP_REGULAR: "regular",
    _MASK_VJP_INTERLEAVED: "interleaved",
    _MASK_VJP_SORTED: "sorted",
}


def _plain_atom_call(*operands: Tensor | None) -> bool:
    tensors = tuple(tensor for tensor in operands if tensor is not None)
    return (
        all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in tensors)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(tensors)
        and not functorch_transforms_active()
        and not forward_mode_active(*tensors)
    )


@overload
def _kernel_mask(atom_mask: None) -> None: ...


@overload
def _kernel_mask(atom_mask: Tensor) -> Tensor: ...


def _kernel_mask(atom_mask: Tensor | None) -> Tensor | None:
    if atom_mask is None:
        return None
    atom_mask = atom_mask.resolve_neg()
    if atom_mask.is_contiguous() or _mask_batch_repeat(atom_mask) is not None:
        return atom_mask
    return atom_mask.contiguous()


def _mask_execution_plan(plan: AtomToTokenReductionPlan, atom_mask: Tensor | None) -> AtomToTokenReductionPlan:
    repeat = 1 if atom_mask is None or atom_mask.is_contiguous() else _required_mask_batch_repeat(atom_mask)
    if repeat == plan.problem.mask_batch_repeat:
        return plan
    return ATOM_TO_TOKEN_REDUCTION.resolve(replace(plan.problem, mask_batch_repeat=repeat))


@lru_cache(maxsize=16)
def _l2_cache_size(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).L2_cache_size)


@lru_cache(maxsize=256)
def _prepared_atom_to_token_plan(
    schedule: AtomToTokenReductionSchedule,
    layout_plan: AtomTokenPlan,
    atom_shape: tuple[int, ...],
    atom_stride: tuple[int, ...],
    atom_dtype: torch.dtype,
    atom_device: torch.device,
    mapping_shape: tuple[int, ...],
    mapping_stride: tuple[int, ...],
    mapping_dtype: torch.dtype,
    mapping_device: torch.device,
    mask_shape: tuple[int, ...] | None,
    mask_stride: tuple[int, ...] | None,
    mask_dtype: torch.dtype | None,
    mask_device: torch.device | None,
) -> AtomToTokenReductionPlan | None:
    if (
        atom_device.type != "cuda"
        or atom_dtype not in STANDARD_FLOAT_DTYPES
        or len(atom_shape) < 2
        or not _contiguous_shape_stride(atom_shape, atom_stride)
        or mapping_device != atom_device
        or mapping_dtype != torch.long
        or len(mapping_shape) != 1
        or mapping_shape[0] != atom_shape[-2]
        or mapping_shape[0] <= 0
        or not _contiguous_shape_stride(mapping_shape, mapping_stride)
        or layout_plan.kind not in ("regular", "sorted")
    ):
        return None
    channels = atom_shape[-1]
    batch = math.prod(atom_shape[:-2])
    if channels <= 0 or batch <= 0:
        return None
    repeat = 1
    if mask_shape is not None:
        if (
            mask_device != atom_device
            or mask_shape != atom_shape[:-1]
            or mask_dtype is None
            or mask_dtype.is_complex
            or mask_stride is None
        ):
            return None
        mask_repeat = _mask_shape_batch_repeat(mask_shape, mask_stride)
        if mask_repeat is None or mask_repeat <= 0 or batch % mask_repeat:
            return None
        repeat = mask_repeat
    problem = AtomToTokenReductionProblem(
        batch=batch,
        atoms=atom_shape[-2],
        num_tokens=layout_plan.num_tokens,
        channels=channels,
        dtype=atom_dtype,
        kind=layout_plan.kind,
        width=layout_plan.width,
        has_atom_mask=mask_shape is not None,
        mask_batch_repeat=repeat,
        mask_vjp=mask_dtype is not None and mask_dtype.is_floating_point,
    )
    return schedule.resolve(problem) if schedule.supports(problem) else None


def _contiguous_shape_stride(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size == 0:
            return True
        if size != 1:
            if step != expected:
                return False
            expected *= size
    return True


def _mask_shape_batch_repeat(shape: tuple[int, ...], stride: tuple[int, ...]) -> int | None:
    if stride[-1] != 1:
        return None
    expected = shape[-1]
    repeat = 1
    broadcast_suffix = True
    for size, step in zip(reversed(shape[:-1]), reversed(stride[:-1])):
        if size == 1:
            continue
        if broadcast_suffix and step == 0:
            repeat *= size
            continue
        broadcast_suffix = False
        if step != expected:
            return None
        expected *= size
    return repeat


def _compatible_atom_to_token_offsets(
    token_offsets: Tensor | None,
    mapping: Tensor,
    num_tokens: int,
) -> Tensor | None:
    if (
        token_offsets is None
        or token_offsets.dim() != 1
        or token_offsets.numel() != num_tokens + 1
        or token_offsets.dtype not in (torch.int32, torch.int64)
        or token_offsets.device != mapping.device
        or token_offsets.stride(0) < 0
    ):
        return None
    return token_offsets


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
    launch_plan = _prepared_atom_to_token_plan(
        ATOM_TO_TOKEN_REDUCTION,
        layout_plan,
        tuple(atom.shape),
        atom.stride(),
        atom.dtype,
        atom.device,
        tuple(mapping.shape),
        mapping.stride(),
        mapping.dtype,
        mapping.device,
        None if atom_mask is None else tuple(atom_mask.shape),
        None if atom_mask is None else atom_mask.stride(),
        None if atom_mask is None else atom_mask.dtype,
        None if atom_mask is None else atom_mask.device,
    )
    if launch_plan is None:
        return None
    offset = float(denominator_offset)
    problem = launch_plan.problem
    native_bytes = 2 * problem.batch * (problem.atoms + problem.num_tokens) * problem.channels * 4
    if (
        atom_mask is None
        and atom.dtype is torch.float32
        and (not normalize or layout_plan.kind == "regular")
        and native_bytes <= _l2_cache_size(atom.device)
    ):
        if layout_plan.kind == "regular":
            output = atom.unflatten(-2, (layout_plan.num_tokens, layout_plan.width)).sum(-2)
        elif original_mapping is not None:
            index = original_mapping.reshape(*(1,) * (atom.dim() - 2), -1, 1).expand_as(atom)
            output = atom.new_zeros((*atom.shape[:-2], layout_plan.num_tokens, atom.shape[-1]))
            output.scatter_add_(-2, index, atom)
        else:
            output = None
        if output is not None:
            return output / (layout_plan.width + offset) if normalize else output
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


def _inversion_sort_bound(device: torch.device) -> int:
    capacity = shared_memory_budget(cuda_device_caps(device)) // 16
    return min(8192, 1 << (max(capacity, 1).bit_length() - 1))


def _batched_mapping_metadata(
    atom_to_token: Tensor, plan: BatchedAtomToTokenReductionPlan
) -> tuple[Tensor, Tensor, Tensor]:
    atom_to_token = _kernel_operand(atom_to_token)
    problem = plan.problem

    shape = (problem.mapping_batch, problem.atoms)
    bound = _inversion_sort_bound(atom_to_token.device)
    if problem.atoms <= bound:
        sorted_mapping = torch.empty(shape, device=atom_to_token.device, dtype=torch.int64)
        permutation = torch.empty_like(sorted_mapping)
        block = max(32, _next_power_of_two(problem.atoms))
        flags = torch.empty(plan.mapping_flag_shape, device=atom_to_token.device, dtype=torch.int32)
        kernel._batched_atom_to_token_classify_and_invert[(problem.mapping_batch,)](
            atom_to_token,
            flags,
            sorted_mapping,
            permutation,
            problem.atoms,
            problem.num_tokens,
            block,
            num_warps=8 if block >= 4096 else 4,
        )
    else:
        tiles = (problem.atoms + 255) // 256
        block_tokens = _next_power_of_two(problem.num_tokens)
        if tiles <= 256 and block_tokens <= 4096 and problem.mapping_batch * tiles * block_tokens <= 1 << 22:
            flags = torch.empty(plan.mapping_flag_shape, device=atom_to_token.device, dtype=torch.int32)
            tile_flags = torch.empty((problem.mapping_batch, tiles), device=atom_to_token.device, dtype=torch.int32)
            sorted_mapping = torch.empty(shape, device=atom_to_token.device, dtype=torch.int64)
            permutation = torch.empty_like(sorted_mapping)
            counts = torch.empty(
                (problem.mapping_batch, tiles, block_tokens), device=atom_to_token.device, dtype=torch.int32
            )
            totals = torch.empty((problem.mapping_batch, block_tokens), device=atom_to_token.device, dtype=torch.int32)
            kernel._batched_atom_token_tile_classify_and_count[(tiles, problem.mapping_batch)](
                atom_to_token,
                tile_flags,
                counts,
                problem.atoms,
                problem.num_tokens,
                tiles,
                256,
                block_tokens,
                num_warps=4,
            )
            kernel._batched_atom_token_tile_prefix[((block_tokens + 31) // 32, problem.mapping_batch)](
                tile_flags,
                flags,
                counts,
                totals,
                tiles,
                block_tokens,
                _next_power_of_two(tiles),
                32,
                num_warps=4,
            )
            kernel._batched_atom_token_token_prefix[(problem.mapping_batch,)](
                flags,
                totals,
                block_tokens,
                num_warps=4,
            )
            kernel._batched_atom_token_tile_scatter[(tiles, problem.mapping_batch)](
                atom_to_token,
                flags,
                counts,
                totals,
                sorted_mapping,
                permutation,
                problem.atoms,
                problem.num_tokens,
                tiles,
                256,
                block_tokens,
                num_warps=4,
            )
        else:
            narrow_keys = problem.num_tokens <= 2147483647
            keys = torch.empty(shape, device=atom_to_token.device, dtype=torch.int32 if narrow_keys else torch.int64)
            flags = _batched_mapping_flags(atom_to_token, plan, keys)
            sorted_keys, permutation = torch.sort(keys, dim=-1, stable=True)
            sorted_mapping = sorted_keys.to(torch.int64)
    return flags, sorted_mapping, permutation


def _nested_mapping_metadata(
    atom_to_token: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    atom_batch: Tensor,
    token_batch: Tensor,
    plan: NestedAtomToTokenReductionPlan,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    atom_to_token = _kernel_operand(atom_to_token)
    atom_offsets = _kernel_operand(atom_offsets)
    token_offsets = _kernel_operand(token_offsets)
    atom_batch = _kernel_operand(atom_batch)
    token_batch = _kernel_operand(token_batch)
    problem = plan.problem
    device = atom_to_token.device
    block = max(32, min(_inversion_sort_bound(device), _next_power_of_two(max(problem.total_atoms, 1))))
    global_sort = problem.total_atoms > block
    sorted_mapping = torch.empty(atom_to_token.shape, device=device, dtype=torch.int32)

    token_count = torch.zeros(plan.token_count_shape, device=device, dtype=torch.int32)
    sample_flags = torch.zeros(plan.sample_flag_shape, device=device, dtype=torch.int32)
    if plan.classify_grid[0] > 0:
        kernel._nested_atom_to_token_count[plan.classify_grid](
            atom_to_token,
            atom_batch,
            atom_offsets,
            token_offsets,
            token_count,
            sample_flags,
            sorted_mapping,
            problem.total_atoms,
            global_sort,
            plan.classify_block_atoms,
            num_warps=plan.num_warps,
        )
    sample_kind = torch.empty(plan.sample_flag_shape, device=device, dtype=torch.int32)
    if global_sort:
        sorted_mapping, permutation = torch.sort(sorted_mapping, stable=True)
    else:
        permutation = torch.empty(atom_to_token.shape, device=device, dtype=torch.int64)
    kernel._nested_atom_to_token_finalize_and_invert[plan.finalize_grid](
        atom_to_token,
        atom_offsets,
        token_offsets,
        sample_flags,
        sample_kind,
        sorted_mapping,
        permutation,
        problem.num_samples,
        global_sort,
        block,
        num_warps=8 if block >= 4096 else 4,
    )
    segment_start = torch.empty((problem.total_tokens,), device=device, dtype=torch.int64)
    if plan.descriptor_grid[0] > 0:
        kernel._nested_atom_to_token_descriptor[plan.descriptor_grid](
            atom_to_token,
            sorted_mapping,
            token_batch,
            atom_offsets,
            token_offsets,
            sample_kind,
            segment_start,
            problem.total_tokens,
            global_sort,
            num_warps=plan.num_warps,
        )
    return sample_kind, segment_start, token_count, permutation


def _supports_regular_atom_to_token_reduce(
    atom: Tensor,
    atom_to_token: Tensor,
    atom_mask: Tensor | None,
    *,
    num_tokens: int,
    atoms_per_token: int,
) -> bool:

    if (
        not atom.is_cuda
        or not tensors_share_device(atom, atom_to_token, atom_mask)
        or not atom.is_contiguous()
        or atom.dim() < 2
        or atom.dtype not in STANDARD_FLOAT_DTYPES
        or not atom_to_token.is_cuda
        or not atom_to_token.is_contiguous()
        or atom_to_token.dtype != torch.long
        or atom_to_token.dim() != 1
        or atom_to_token.numel() != atom.shape[-2]
        or atom_to_token.numel() == 0
        or atoms_per_token <= 0
    ):
        return False
    if not _supports_atom_mask(atom, atom_mask):
        return False
    return _supports_execution_plan(
        atom,
        atom_mask,
        AtomTokenPlan("regular", num_tokens, atoms_per_token),
    )


def _supports_sorted_atom_to_token_reduce(
    atom: Tensor,
    atom_to_token: Tensor,
    atom_mask: Tensor | None,
    *,
    num_tokens: int,
    max_atoms_per_token: int,
) -> bool:

    if (
        not atom.is_cuda
        or not tensors_share_device(atom, atom_to_token, atom_mask)
        or not atom.is_contiguous()
        or atom.dim() < 2
        or atom.dtype not in STANDARD_FLOAT_DTYPES
        or not atom_to_token.is_cuda
        or not atom_to_token.is_contiguous()
        or atom_to_token.dtype != torch.long
        or atom_to_token.dim() != 1
        or atom_to_token.numel() != atom.shape[-2]
        or atom_to_token.numel() == 0
        or max_atoms_per_token <= 0
    ):
        return False
    if not _supports_atom_mask(atom, atom_mask):
        return False
    return _supports_execution_plan(
        atom,
        atom_mask,
        AtomTokenPlan("sorted", num_tokens, max_atoms_per_token),
    )


def _supports_interleaved_atom_to_token_reduce(
    atom: Tensor,
    atom_mask: Tensor | None,
    *,
    num_tokens: int,
    atoms_per_token: int,
) -> bool:

    if (
        not atom.is_cuda
        or not atom.is_contiguous()
        or atom.dim() < 2
        or atom.dtype not in STANDARD_FLOAT_DTYPES
        or atom.shape[-2] != int(num_tokens) * int(atoms_per_token)
        or atoms_per_token <= 0
    ):
        return False
    if not _supports_atom_mask(atom, atom_mask):
        return False
    return _supports_execution_plan(
        atom,
        atom_mask,
        AtomTokenPlan("interleaved", num_tokens, atoms_per_token),
    )


def _supports_atom_mask(atom: Tensor, atom_mask: Tensor | None) -> bool:
    if atom_mask is None:
        return True
    repeat = _mask_batch_repeat(atom_mask)
    batch = math.prod(atom.shape[:-2]) if atom.dim() > 2 else 1
    return (
        atom_mask.is_cuda
        and atom_mask.device == atom.device
        and tuple(atom_mask.shape) == tuple(atom.shape[:-1])
        and not atom_mask.is_complex()
        and repeat is not None
        and repeat > 0
        and batch % repeat == 0
    )


def _mask_batch_repeat(atom_mask: Tensor) -> int | None:

    if atom_mask.stride(-1) != 1:
        return None
    expected_stride = atom_mask.shape[-1]
    repeat = 1
    in_broadcast_suffix = True
    for size, stride in reversed(tuple(zip(atom_mask.shape[:-1], atom_mask.stride()[:-1]))):
        if size == 1:
            continue
        if in_broadcast_suffix and stride == 0:
            repeat *= size
            continue
        in_broadcast_suffix = False
        if stride != expected_stride:
            return None
        expected_stride *= size
    return repeat


def _required_mask_batch_repeat(atom_mask: Tensor | None) -> int:
    if atom_mask is None:
        return 1
    repeat = _mask_batch_repeat(atom_mask)
    if repeat is None:
        raise ValueError("atom_mask must be contiguous or broadcast over trailing batch dimensions")
    return repeat


def _gather_atom_mask(atom_mask: Tensor, permutation: Tensor) -> Tensor:

    physical = atom_mask
    for dim, (size, stride) in enumerate(zip(atom_mask.shape[:-1], atom_mask.stride()[:-1])):
        if size > 1 and stride == 0:
            physical = physical.narrow(dim, 0, 1)
    return physical.contiguous().index_select(-1, permutation).expand(atom_mask.shape)


def _supports_execution_plan(
    atom: Tensor,
    atom_mask: Tensor | None,
    layout_plan: AtomTokenPlan,
) -> bool:
    channels = atom.shape[-1]
    batch = math.prod(atom.shape[:-2]) if atom.dim() > 2 else 1
    if channels <= 0 or batch <= 0:
        return False
    return ATOM_TO_TOKEN_REDUCTION.supports(_atom_to_token_problem(atom, atom_mask, layout_plan))


def _atom_to_token_problem(
    atom: Tensor,
    atom_mask: Tensor | None,
    layout_plan: AtomTokenPlan,
) -> AtomToTokenReductionProblem:
    batch = math.prod(atom.shape[:-2]) if atom.dim() > 2 else 1
    return _make_atom_to_token_problem(
        batch=batch,
        atoms=atom.shape[-2],
        channels=atom.shape[-1],
        dtype=atom.dtype,
        atom_mask=atom_mask,
        layout_plan=layout_plan,
    )


def _grad_out_atom_to_token_problem(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atoms: int,
    layout_plan: AtomTokenPlan,
) -> AtomToTokenReductionProblem:
    batch = math.prod(grad_out.shape[:-2]) if grad_out.dim() > 2 else 1
    return _make_atom_to_token_problem(
        batch=batch,
        atoms=atoms,
        channels=grad_out.shape[-1],
        dtype=grad_out.dtype,
        atom_mask=atom_mask,
        layout_plan=layout_plan,
    )


def _make_atom_to_token_problem(
    *,
    batch: int,
    atoms: int,
    channels: int,
    dtype: torch.dtype,
    atom_mask: Tensor | None,
    layout_plan: AtomTokenPlan,
) -> AtomToTokenReductionProblem:
    return AtomToTokenReductionProblem(
        batch=batch,
        atoms=atoms,
        num_tokens=layout_plan.num_tokens,
        channels=channels,
        dtype=dtype,
        kind=layout_plan.kind,
        width=layout_plan.width,
        has_atom_mask=atom_mask is not None,
        mask_batch_repeat=_required_mask_batch_repeat(atom_mask),
        mask_vjp=atom_mask is not None and atom_mask.is_floating_point(),
    )


def supports_atom_to_token_reduce(
    atom: Tensor,
    atom_to_token: Tensor,
    atom_mask: Tensor | None,
    plan: AtomTokenPlan,
) -> bool:

    if plan.kind == "regular":
        return _supports_regular_atom_to_token_reduce(
            atom,
            atom_to_token,
            atom_mask,
            num_tokens=plan.num_tokens,
            atoms_per_token=plan.width,
        )
    if plan.kind == "interleaved":
        return _supports_interleaved_atom_to_token_reduce(
            atom,
            atom_mask,
            num_tokens=plan.num_tokens,
            atoms_per_token=plan.width,
        )
    return _supports_sorted_atom_to_token_reduce(
        atom,
        atom_to_token,
        atom_mask,
        num_tokens=plan.num_tokens,
        max_atoms_per_token=plan.width,
    )


def supports_atom_to_token_reduce_layout(plan: AtomTokenPlan, *, atoms: int) -> bool:

    return ATOM_TO_TOKEN_REDUCTION.supports_layout(
        plan.kind,
        atoms=int(atoms),
        num_tokens=plan.num_tokens,
        width=plan.width,
    )


def _batched_atom_to_token_problem(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    mapping_batch_repeat: int,
    num_tokens: int,
) -> BatchedAtomToTokenReductionProblem:
    batch = math.prod(atom.shape[:-2]) if atom.dim() > 2 else 1
    atoms = atom.shape[-2]
    mapping_batch = atom_to_token.numel() // atoms
    return BatchedAtomToTokenReductionProblem(
        batch=batch,
        mapping_batch=mapping_batch,
        mapping_batch_repeat=mapping_batch_repeat,
        atoms=atoms,
        num_tokens=num_tokens,
        channels=atom.shape[-1],
        dtype=atom.dtype,
        has_atom_mask=atom_mask is not None,
        mask_batch_repeat=_required_mask_batch_repeat(atom_mask),
        mask_vjp=atom_mask is not None and atom_mask.is_floating_point(),
    )


def supports_batched_atom_to_token_reduce(
    atom: Tensor,
    atom_to_token: Tensor,
    atom_mask: Tensor | None,
    *,
    mapping_batch_repeat: int,
    num_tokens: int,
) -> bool:

    if (
        not atom.is_cuda
        or not tensors_share_device(atom, atom_to_token, atom_mask)
        or not atom.is_contiguous()
        or atom.dim() < 3
        or atom.dtype not in STANDARD_FLOAT_DTYPES
        or atom.shape[-2] <= 0
        or atom.shape[-1] <= 0
        or num_tokens <= 0
        or not atom_to_token.is_cuda
        or not atom_to_token.is_contiguous()
        or atom_to_token.dtype != torch.long
        or atom_to_token.dim() < 2
        or atom_to_token.shape[-1] != atom.shape[-2]
        or mapping_batch_repeat <= 0
        or not _supports_atom_mask(atom, atom_mask)
    ):
        return False
    problem = _batched_atom_to_token_problem(
        atom,
        atom_mask,
        atom_to_token,
        mapping_batch_repeat,
        num_tokens,
    )
    return BATCHED_ATOM_TO_TOKEN_REDUCTION.supports(problem)


def supports_shared_general_atom_to_token_reduce(
    atom: Tensor,
    atom_to_token: Tensor,
    atom_mask: Tensor | None,
    *,
    num_tokens: int,
) -> bool:

    if (
        not atom.is_cuda
        or not tensors_share_device(atom, atom_to_token, atom_mask)
        or not atom.is_contiguous()
        or atom.dim() < 2
        or atom.dtype not in STANDARD_FLOAT_DTYPES
        or atom.shape[-2] <= 0
        or atom.shape[-1] <= 0
        or num_tokens <= 0
        or not atom_to_token.is_cuda
        or not atom_to_token.is_contiguous()
        or atom_to_token.dtype != torch.long
        or atom_to_token.dim() != 1
        or atom_to_token.shape[0] != atom.shape[-2]
        or not _supports_atom_mask(atom, atom_mask)
    ):
        return False
    batch = math.prod(atom.shape[:-2]) if atom.dim() > 2 else 1
    problem = _batched_atom_to_token_problem(
        atom,
        atom_mask,
        atom_to_token,
        batch,
        num_tokens,
    )
    return BATCHED_ATOM_TO_TOKEN_REDUCTION.supports(problem)


def _resolve_batched_atom_to_token_plan(
    problem: BatchedAtomToTokenReductionProblem,
    device: torch.device,
) -> BatchedAtomToTokenReductionPlan:
    target = cuda_device_caps_if_needed(device, BATCHED_ATOM_TO_TOKEN_REDUCTION.target_changes_plan(problem))
    return BATCHED_ATOM_TO_TOKEN_REDUCTION.resolve(problem, target=target)


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
            launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, plan))
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


def _run_atom_to_token_reduce_mask_backward(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor | None,
    grad_out: Tensor,
    num_tokens: int,
    plan_kind: int,
    width: int,
    denominator_offset: float,
    normalize: bool,
    permutation: Tensor | None = None,
    launch_plan: AtomToTokenReductionPlan | None = None,
) -> Tensor:
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_mask(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        grad_out = _kernel_operand(grad_out)
        permutation = _kernel_operand(permutation)
        if launch_plan is not None:
            launch_plan = _mask_execution_plan(launch_plan, atom_mask)
        kind = _MASK_VJP_KINDS.get(plan_kind)
        if kind is None:
            raise ValueError(f"unknown atom-to-token mask plan kind: {plan_kind}")
        if launch_plan is None:
            layout_plan = AtomTokenPlan(kind, num_tokens, width)
            launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, layout_plan))
        mask_plan = launch_plan.mask_vjp_plan
        if mask_plan is None:
            raise ValueError("atom-to-token mask backward requires a mask-VJP plan")

        problem = launch_plan.problem
        atom_flat = atom.reshape(launch_plan.flat_atom_shape)
        grad_out_flat = grad_out.contiguous().reshape(launch_plan.flat_output_shape)
        if normalize:
            numerator = torch.empty(mask_plan.numerator_shape, device=atom.device, dtype=torch.float32)
            raw_denominator = torch.empty(mask_plan.denominator_shape, device=atom.device, dtype=torch.float32)
        else:
            numerator = torch.empty((), device=atom.device, dtype=torch.float32)
            raw_denominator = numerator
        grad_mask = torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)
        index_pointer = atom if atom_to_token is None else atom_to_token
        if normalize:
            kernel._atom_to_token_reduce_mask_state[mask_plan.state_grid](
                atom_flat,
                atom_mask,
                index_pointer,
                numerator,
                raw_denominator,
                problem.atoms,
                problem.num_tokens,
                problem.channels,
                launch_plan.mask_plan_kind,
                problem.width,
                problem.mask_batch_repeat,
                launch_plan.max_iters,
                launch_plan.block_channels,
                launch_plan.block_atoms,
                permutation_ptr=permutation,
                has_permutation=permutation is not None,
                num_warps=launch_plan.num_warps,
            )
        kernel._atom_to_token_reduce_mask_backward[mask_plan.reduce_grid](
            atom_flat,
            index_pointer,
            numerator,
            raw_denominator,
            grad_out_flat,
            grad_mask,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            normalize,
            denominator_offset,
            launch_plan.mask_plan_kind,
            problem.width,
            launch_plan.block_channels,
            permutation_ptr=permutation,
            has_permutation=permutation is not None,
            num_warps=launch_plan.num_warps,
        )
        return grad_mask


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


def _batched_mapping_flags(
    atom_to_token: Tensor,
    plan: BatchedAtomToTokenReductionPlan,
    sort_keys: Tensor,
) -> Tensor:
    atom_to_token = _kernel_operand(atom_to_token)
    sort_keys = _kernel_operand(sort_keys)
    problem = plan.problem
    flags = torch.zeros(plan.mapping_flag_shape, device=atom_to_token.device, dtype=torch.int32)
    kernel._batched_atom_to_token_classify[plan.classify_grid](
        atom_to_token,
        flags,
        sort_keys,
        problem.mapping_batch * problem.atoms,
        problem.atoms,
        problem.num_tokens,
        plan.classify_block_atoms,
        num_warps=plan.num_warps,
    )
    return flags


def _launch_batched_atom_to_token_reduce_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    plan: BatchedAtomToTokenReductionPlan,
    denominator_offset: float,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    atom = _kernel_operand(atom)
    atom_mask = _kernel_mask(atom_mask)
    atom_to_token = _kernel_operand(atom_to_token)
    repeat = _required_mask_batch_repeat(atom_mask)
    if repeat != plan.problem.mask_batch_repeat:
        plan = _resolve_batched_atom_to_token_plan(replace(plan.problem, mask_batch_repeat=repeat), atom.device)
    problem = plan.problem
    atom_flat = atom.reshape(plan.flat_atom_shape)
    output = torch.empty(plan.flat_output_shape, device=atom.device, dtype=atom.dtype)
    mapping_flags, sorted_mapping, permutation = _batched_mapping_metadata(atom_to_token, plan)
    mask_pointer = atom_flat if atom_mask is None else atom_mask
    kernel._batched_atom_to_token_reduce_forward[plan.segment_grid](
        atom_flat,
        mask_pointer,
        atom_to_token,
        mapping_flags,
        sorted_mapping,
        permutation,
        output,
        problem.atoms,
        problem.num_tokens,
        problem.channels,
        problem.mapping_batch_repeat,
        problem.has_atom_mask,
        problem.mask_batch_repeat,
        normalize,
        float(denominator_offset),
        plan.block_channels,
        plan.block_atoms,
        num_warps=plan.num_warps,
    )
    return (
        output.reshape(*atom.shape[:-2], problem.num_tokens, problem.channels),
        mapping_flags,
        sorted_mapping,
        permutation,
    )


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


def _run_batched_atom_to_token_reduce_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    num_tokens: int,
    mapping_batch_repeat: int,
    denominator_offset: float,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with cuda_device_context(atom.device):
        problem = _batched_atom_to_token_problem(
            atom,
            atom_mask,
            atom_to_token,
            int(mapping_batch_repeat),
            int(num_tokens),
        )
        plan = _resolve_batched_atom_to_token_plan(problem, atom.device)
        return _launch_batched_atom_to_token_reduce_forward(
            atom,
            atom_mask,
            atom_to_token,
            plan,
            float(denominator_offset),
            normalize,
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


def _run_batched_atom_to_token_reduce_backward(
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
    with cuda_device_context(grad_out.device):
        grad_out = _kernel_operand(grad_out)
        atom_mask = _kernel_mask(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        mapping_flags = _kernel_operand(mapping_flags)
        sorted_mapping = _kernel_operand(sorted_mapping)
        permutation = _kernel_operand(permutation)
        grad_out = grad_out.contiguous()
        batch = math.prod(grad_out.shape[:-2]) if grad_out.dim() > 2 else 1
        mapping_batch = atom_to_token.numel() // int(atoms)
        problem = BatchedAtomToTokenReductionProblem(
            batch=batch,
            mapping_batch=mapping_batch,
            mapping_batch_repeat=int(mapping_batch_repeat),
            atoms=int(atoms),
            num_tokens=grad_out.shape[-2],
            channels=grad_out.shape[-1],
            dtype=grad_out.dtype,
            has_atom_mask=atom_mask is not None,
            mask_batch_repeat=_required_mask_batch_repeat(atom_mask),
            mask_vjp=atom_mask is not None and atom_mask.is_floating_point(),
        )
        plan = _resolve_batched_atom_to_token_plan(problem, grad_out.device)
        grad_out_flat = grad_out.reshape(plan.flat_output_shape)
        grad_atom = torch.empty(plan.flat_atom_shape, device=grad_out.device, dtype=grad_out.dtype)
        mask_pointer = grad_out_flat if atom_mask is None else atom_mask
        kernel._batched_atom_to_token_reduce_backward[plan.segment_grid](
            grad_out_flat,
            mask_pointer,
            atom_to_token,
            mapping_flags,
            sorted_mapping,
            permutation,
            grad_atom,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.mapping_batch_repeat,
            problem.has_atom_mask,
            problem.mask_batch_repeat,
            normalize,
            float(denominator_offset),
            plan.block_channels,
            plan.block_atoms,
            num_warps=plan.num_warps,
        )
        return grad_atom.reshape(*grad_out.shape[:-2], problem.atoms, problem.channels)


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


def _run_batched_atom_to_token_reduce_mask_backward(
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
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_mask(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        mapping_flags = _kernel_operand(mapping_flags)
        sorted_mapping = _kernel_operand(sorted_mapping)
        permutation = _kernel_operand(permutation)
        grad_out = _kernel_operand(grad_out)
        grad_out = grad_out.contiguous()
        problem = _batched_atom_to_token_problem(
            atom,
            atom_mask,
            atom_to_token,
            int(mapping_batch_repeat),
            int(num_tokens),
        )
        plan = _resolve_batched_atom_to_token_plan(problem, atom.device)
        atom_flat = atom.reshape(plan.flat_atom_shape)
        grad_out_flat = grad_out.reshape(plan.flat_output_shape)
        if normalize:
            if plan.numerator_shape is None or plan.denominator_shape is None:
                raise ValueError("batched atom-to-token mask backward requires state scratch")
            numerator = torch.empty(plan.numerator_shape, device=atom.device, dtype=torch.float32)
            raw_denominator = torch.empty(plan.denominator_shape, device=atom.device, dtype=torch.float32)
        else:
            numerator = torch.empty((), device=atom.device, dtype=torch.float32)
            raw_denominator = numerator
        grad_mask = torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)
        if normalize:
            kernel._batched_atom_to_token_reduce_mask_state[plan.segment_grid](
                atom_flat,
                atom_mask,
                atom_to_token,
                mapping_flags,
                sorted_mapping,
                permutation,
                numerator,
                raw_denominator,
                problem.atoms,
                problem.num_tokens,
                problem.channels,
                problem.mapping_batch_repeat,
                problem.mask_batch_repeat,
                plan.block_channels,
                plan.block_atoms,
                num_warps=plan.num_warps,
            )
        kernel._batched_atom_to_token_reduce_mask_backward[plan.mask_vjp_grid](
            atom_flat,
            atom_to_token,
            numerator,
            raw_denominator,
            grad_out_flat,
            grad_mask,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.mapping_batch_repeat,
            normalize,
            float(denominator_offset),
            plan.block_channels,
            plan.mask_vjp_block_atoms,
            problem.batch * problem.atoms,
            num_warps=4,
        )
        return grad_mask


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

    batch = math.prod(grad_out.shape[:-2]) if grad_out.dim() > 2 else 1
    problem = BatchedAtomToTokenReductionProblem(
        batch=batch,
        mapping_batch=1,
        mapping_batch_repeat=batch,
        atoms=int(atoms),
        num_tokens=grad_out.shape[-2],
        channels=grad_out.shape[-1],
        dtype=grad_out.dtype,
        has_atom_mask=atom_mask is not None,
        mask_batch_repeat=_required_mask_batch_repeat(atom_mask),
        mask_vjp=atom_mask is not None and atom_mask.is_floating_point(),
    )
    plan = _resolve_batched_atom_to_token_plan(problem, grad_out.device)
    mapping_flags, sorted_mapping, permutation = _batched_mapping_metadata(atom_to_token, plan)
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

    batch = math.prod(atom.shape[:-2]) if atom.dim() > 2 else 1
    problem = _batched_atom_to_token_problem(
        atom,
        atom_mask,
        atom_to_token,
        batch,
        int(num_tokens),
    )
    plan = _resolve_batched_atom_to_token_plan(problem, atom.device)
    mapping_flags, sorted_mapping, permutation = _batched_mapping_metadata(atom_to_token, plan)
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


def _flatten_grad_out(grad_out: Tensor, num_tokens: int, atoms: int) -> tuple[Tensor, int, tuple[int, ...]]:

    input_shape = tuple(grad_out.shape)
    channels = input_shape[-1]
    batch = math.prod(input_shape[:-2]) if len(input_shape) > 2 else 1
    return (
        grad_out.contiguous().reshape(batch, num_tokens, channels),
        batch,
        (*input_shape[:-2], atoms, channels),
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
    layout_plan = AtomTokenPlan("regular", num_tokens, atoms_per_token)
    launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, layout_plan))
    return _launch_atom_to_token_reduce_regular_forward(
        atom,
        atom_mask,
        launch_plan,
        denominator_offset,
        normalize,
    )


def _launch_atom_to_token_reduce_regular_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    launch_plan: AtomToTokenReductionPlan,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_mask(atom_mask)
        if launch_plan is not None:
            launch_plan = _mask_execution_plan(launch_plan, atom_mask)
        input_shape = tuple(atom.shape)
        problem = launch_plan.problem
        atom_flat = atom if atom.dim() == 3 else atom.reshape(launch_plan.flat_atom_shape)
        out = torch.empty(launch_plan.flat_output_shape, device=atom.device, dtype=atom.dtype)
        # Disabled pointer arguments still require a tensor placeholder.
        placeholder = atom_flat if atom_mask is None else atom_mask
        kernel._atom_to_token_reduce_regular_forward[launch_plan.segment_grid](
            atom_flat,
            placeholder,
            out,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.width,
            problem.has_atom_mask,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            launch_plan.accumulate_fp32,
            launch_plan.block_channels,
            launch_plan.block_atoms,
            num_warps=launch_plan.num_warps,
        )
        return out if atom.dim() == 3 else out.reshape(*input_shape[:-2], problem.num_tokens, problem.channels)


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


def _run_atom_to_token_reduce_regular_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atoms: int,
    atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
    launch_plan: AtomToTokenReductionPlan | None = None,
) -> Tensor:
    with cuda_device_context(grad_out.device):
        grad_out = _kernel_operand(grad_out)
        atom_mask = _kernel_mask(atom_mask)
        if launch_plan is not None:
            launch_plan = _mask_execution_plan(launch_plan, atom_mask)
        grad_out = grad_out.contiguous()
        if launch_plan is None or launch_plan.problem.dtype != grad_out.dtype:
            num_tokens = grad_out.shape[-2]
            layout_plan = AtomTokenPlan("regular", num_tokens, atoms_per_token)
            launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(
                _grad_out_atom_to_token_problem(grad_out, atom_mask, atoms, layout_plan)
            )
        problem = launch_plan.problem
        output_shape = (*grad_out.shape[:-2], problem.atoms, problem.channels)
        grad_out_flat = grad_out.reshape(launch_plan.flat_output_shape)
        grad_atom = torch.empty(launch_plan.flat_atom_shape, device=grad_out.device, dtype=grad_out.dtype)
        placeholder = grad_out_flat if atom_mask is None else atom_mask
        kernel._atom_to_token_reduce_regular_backward[launch_plan.segment_grid](
            grad_out_flat,
            placeholder,
            grad_atom,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.width,
            problem.has_atom_mask,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            launch_plan.accumulate_fp32,
            launch_plan.block_channels,
            launch_plan.block_atoms,
            num_warps=launch_plan.num_warps,
        )
        return grad_atom.reshape(output_shape)


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


def _launch_atom_to_token_reduce_regular_vjp(
    atom: Tensor,
    atom_mask: Tensor,
    grad_out: Tensor,
    launch_plan: AtomToTokenReductionPlan,
    denominator_offset: float,
    normalize: bool,
) -> tuple[Tensor, Tensor]:
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_mask(atom_mask)
        grad_out = _kernel_operand(grad_out)
        if launch_plan is not None:
            launch_plan = _mask_execution_plan(launch_plan, atom_mask)
        problem = launch_plan.problem
        grad_out = grad_out.contiguous()
        grad_out_flat = grad_out if grad_out.dim() == 3 else grad_out.reshape(launch_plan.flat_output_shape)
        atom_flat = atom if atom.dim() == 3 else atom.reshape(launch_plan.flat_atom_shape)
        grad_atom = torch.empty(atom.shape, device=atom.device, dtype=atom.dtype)
        grad_mask = torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)
        kernel._atom_to_token_reduce_regular_vjp[(problem.batch * problem.num_tokens,)](
            atom_flat,
            atom_mask,
            grad_out_flat,
            grad_atom,
            grad_mask,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.width,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            launch_plan.block_channels,
            launch_plan.block_atoms,
            num_warps=launch_plan.num_warps,
            enable_fp_fusion=False,
        )
        return grad_atom, grad_mask


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
            launch = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, layout))
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


def _atom_to_token_reduce_sorted_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    permutation: Tensor | None,
    num_tokens: int,
    max_atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
    launch_plan: AtomToTokenReductionPlan | None = None,
    token_offsets: Tensor | None = None,
) -> Tensor:

    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_mask(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        permutation = _kernel_operand(permutation)
        token_offsets = _kernel_operand(token_offsets)
        if launch_plan is not None:
            launch_plan = _mask_execution_plan(launch_plan, atom_mask)
        direct_permutation = permutation is not None and (
            launch_plan is not None or (not is_compiling() and _plain_atom_call(atom, atom_mask))
        )
        if permutation is not None and not direct_permutation:
            atom = atom.index_select(-2, permutation)
            if atom_mask is not None:
                atom_mask = _gather_atom_mask(atom_mask, permutation)
        if launch_plan is None:
            layout_plan = AtomTokenPlan("sorted", num_tokens, max_atoms_per_token)
            launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, layout_plan))
        problem = launch_plan.problem
        out = torch.empty(launch_plan.flat_output_shape, device=atom.device, dtype=atom.dtype)
        placeholder = atom if atom_mask is None else atom_mask
        kernel._atom_to_token_reduce_sorted_forward[launch_plan.segment_grid](
            atom,
            atom_to_token,
            placeholder,
            out,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.has_atom_mask,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            launch_plan.accumulate_fp32,
            launch_plan.max_iters,
            launch_plan.block_channels,
            launch_plan.block_atoms,
            permutation_ptr=permutation if direct_permutation else None,
            has_permutation=direct_permutation,
            num_warps=launch_plan.num_warps,
            token_offsets_ptr=token_offsets,
            token_offsets_stride=1 if token_offsets is None else token_offsets.stride(0),
            has_token_offsets=token_offsets is not None,
        )
        return out


def _atom_to_token_reduce_sorted_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    permutation: Tensor | None,
    inverse_permutation: Tensor | None,
    num_tokens: int,
    max_atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
    launch_plan: AtomToTokenReductionPlan | None = None,
    token_offsets: Tensor | None = None,
) -> Tensor:

    with cuda_device_context(grad_out.device):
        grad_out = _kernel_operand(grad_out)
        atom_mask = _kernel_mask(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        permutation = _kernel_operand(permutation)
        inverse_permutation = _kernel_operand(inverse_permutation)
        token_offsets = _kernel_operand(token_offsets)
        if launch_plan is not None:
            launch_plan = _mask_execution_plan(launch_plan, atom_mask)
        direct_permutation = (
            permutation is not None
            and not torch.is_grad_enabled()
            and (launch_plan is not None or (not is_compiling() and _plain_atom_call(grad_out, atom_mask)))
        )
        if permutation is not None and atom_mask is not None and not direct_permutation:
            atom_mask = _gather_atom_mask(atom_mask, permutation)
        atoms = atom_to_token.numel()
        if launch_plan is None or launch_plan.problem.dtype != grad_out.dtype:
            layout_plan = AtomTokenPlan("sorted", num_tokens, max_atoms_per_token)
            launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(
                _grad_out_atom_to_token_problem(grad_out, atom_mask, atoms, layout_plan)
            )
        problem = launch_plan.problem
        grad_atom = torch.empty(launch_plan.flat_atom_shape, device=grad_out.device, dtype=grad_out.dtype)
        placeholder = grad_out if atom_mask is None else atom_mask
        kernel._atom_to_token_reduce_sorted_backward[launch_plan.segment_grid](
            grad_out,
            atom_to_token,
            placeholder,
            grad_atom,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.has_atom_mask,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            launch_plan.accumulate_fp32,
            launch_plan.max_iters,
            launch_plan.block_channels,
            launch_plan.block_atoms,
            permutation_ptr=permutation if direct_permutation else None,
            has_permutation=direct_permutation,
            num_warps=launch_plan.num_warps,
            token_offsets_ptr=token_offsets,
            token_offsets_stride=1 if token_offsets is None else token_offsets.stride(0),
            has_token_offsets=token_offsets is not None,
        )
        if inverse_permutation is None or direct_permutation:
            return grad_atom
        return grad_atom.index_select(-2, inverse_permutation)


def _run_atom_to_token_reduce_sorted_vjp(
    atom: Tensor,
    atom_mask: Tensor,
    atom_to_token: Tensor,
    permutation: Tensor | None,
    inverse_permutation: Tensor | None,
    grad_out: Tensor,
    plan: AtomTokenPlan,
    denominator_offset: float,
    normalize: bool,
    need_grad_atom: bool,
    launch_plan: AtomToTokenReductionPlan | None = None,
    token_offsets: Tensor | None = None,
) -> tuple[Tensor | None, Tensor]:
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_mask(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        permutation = _kernel_operand(permutation)
        inverse_permutation = _kernel_operand(inverse_permutation)
        grad_out = _kernel_operand(grad_out)
        token_offsets = _kernel_operand(token_offsets)
        if launch_plan is not None:
            launch_plan = _mask_execution_plan(launch_plan, atom_mask)
        if launch_plan is None:
            launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, plan))
        problem = launch_plan.problem
        grad_atom = torch.empty_like(atom) if need_grad_atom else None
        grad_mask = torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)
        if not normalize:
            kernel._atom_to_token_reduce_sorted_sum_vjp[(((problem.atoms + 15) // 16), problem.batch)](
                atom,
                atom_mask,
                atom_to_token,
                inverse_permutation,
                grad_out.contiguous(),
                grad_atom,
                grad_mask,
                problem.atoms,
                problem.num_tokens,
                problem.channels,
                problem.mask_batch_repeat,
                need_grad_atom,
                inverse_permutation is not None,
                16,
                launch_plan.block_channels,
                num_warps=launch_plan.num_warps,
                enable_fp_fusion=False,
            )
            return grad_atom, grad_mask
        kernel._atom_to_token_reduce_sorted_vjp[(problem.batch * problem.num_tokens,)](
            atom,
            atom_mask,
            atom_to_token,
            permutation,
            grad_out.contiguous(),
            grad_atom,
            grad_mask,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            need_grad_atom,
            permutation is not None,
            launch_plan.max_iters,
            launch_plan.block_atoms,
            launch_plan.block_channels,
            num_warps=launch_plan.num_warps,
            enable_fp_fusion=False,
            token_offsets_ptr=token_offsets,
            token_offsets_stride=1 if token_offsets is None else token_offsets.stride(0),
            has_token_offsets=token_offsets is not None,
        )
        return grad_atom, grad_mask


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
            launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, plan))
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


def _atom_to_token_reduce_interleaved_forward(
    atom: Tensor,
    atom_mask: Tensor | None,
    num_tokens: int,
    atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_mask(atom_mask)
        layout_plan = AtomTokenPlan("interleaved", num_tokens, atoms_per_token)
        launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(_atom_to_token_problem(atom, atom_mask, layout_plan))
        problem = launch_plan.problem
        out = torch.empty(launch_plan.flat_output_shape, device=atom.device, dtype=atom.dtype)
        placeholder = atom if atom_mask is None else atom_mask
        kernel._atom_to_token_reduce_interleaved_forward[launch_plan.segment_grid](
            atom,
            placeholder,
            out,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.width,
            problem.has_atom_mask,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            launch_plan.accumulate_fp32,
            launch_plan.block_channels,
            launch_plan.block_atoms,
            num_warps=launch_plan.num_warps,
        )
        return out


def _atom_to_token_reduce_interleaved_backward(
    grad_out: Tensor,
    atom_mask: Tensor | None,
    num_tokens: int,
    atoms_per_token: int,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:
    with cuda_device_context(grad_out.device):
        grad_out = _kernel_operand(grad_out)
        atom_mask = _kernel_mask(atom_mask)
        atoms = num_tokens * atoms_per_token
        layout_plan = AtomTokenPlan("interleaved", num_tokens, atoms_per_token)
        launch_plan = ATOM_TO_TOKEN_REDUCTION.resolve(
            _grad_out_atom_to_token_problem(grad_out, atom_mask, atoms, layout_plan)
        )
        problem = launch_plan.problem
        grad_atom = torch.empty(launch_plan.flat_atom_shape, device=grad_out.device, dtype=grad_out.dtype)
        placeholder = grad_out if atom_mask is None else atom_mask
        kernel._atom_to_token_reduce_interleaved_backward[launch_plan.segment_grid](
            grad_out,
            placeholder,
            grad_atom,
            problem.atoms,
            problem.num_tokens,
            problem.channels,
            problem.width,
            problem.has_atom_mask,
            problem.mask_batch_repeat,
            normalize,
            denominator_offset,
            launch_plan.accumulate_fp32,
            launch_plan.block_channels,
            launch_plan.block_atoms,
            num_warps=launch_plan.num_warps,
        )
        return grad_atom


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


def _nested_atom_to_token_problem(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_offsets: Tensor,
    token_batch: Tensor,
) -> NestedAtomToTokenReductionProblem:
    sample_size = 1 if atom.dim() == 2 else atom.shape[1]
    mask_vjp = atom_mask is not None and atom_mask.is_floating_point()
    return NestedAtomToTokenReductionProblem(
        num_samples=atom_offsets.shape[0] - 1,
        total_atoms=atom.shape[0],
        total_tokens=token_batch.shape[0],
        sample_size=sample_size,
        channels=atom.shape[-1],
        dtype=atom.dtype,
        has_atom_mask=atom_mask is not None,
        mask_vjp=mask_vjp,
        shared_mask_vjp=(mask_vjp and atom.dim() == 3 and sample_size > 1 and _nested_mask_sample_size(atom_mask) == 1),
    )


def _resolve_nested_atom_to_token_plan(
    problem: NestedAtomToTokenReductionProblem,
    device: torch.device,
) -> NestedAtomToTokenReductionPlan:
    target = cuda_device_caps_if_needed(device, NESTED_ATOM_TO_TOKEN_REDUCTION.target_changes_plan(problem))
    return NESTED_ATOM_TO_TOKEN_REDUCTION.resolve(problem, target=target)


def _nested_mask_sample_size(atom_mask: Tensor | None) -> int:
    return 1 if atom_mask is None or atom_mask.dim() == 1 else atom_mask.shape[1]


def supports_nested_atom_to_token_reduce(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    atom_offsets: Tensor,
    token_offsets: Tensor,
    atom_batch: Tensor,
    token_batch: Tensor,
) -> bool:

    tensors = (
        atom,
        atom_to_token,
        atom_offsets,
        token_offsets,
        atom_batch,
        token_batch,
    )
    if (
        not atom.is_cuda
        or atom.dim() not in (2, 3)
        or not atom.is_contiguous()
        or atom.dtype not in STANDARD_FLOAT_DTYPES
        or atom.shape[-1] <= 0
        or (atom.dim() == 3 and atom.shape[1] <= 0)
        or not tensors_share_device(*tensors, atom_mask)
        or atom_to_token.dtype is not torch.long
        or atom_to_token.dim() != 1
        or not atom_to_token.is_contiguous()
        or atom_to_token.shape[0] != atom.shape[0]
        or atom_offsets.dtype is not torch.int64
        or token_offsets.dtype is not torch.int64
        or atom_offsets.dim() != 1
        or token_offsets.dim() != 1
        or atom_offsets.shape != token_offsets.shape
        or atom_offsets.shape[0] <= 1
        or not atom_offsets.is_contiguous()
        or not token_offsets.is_contiguous()
        or atom_batch.dtype not in (torch.int32, torch.int64)
        or token_batch.dtype not in (torch.int32, torch.int64)
        or atom_batch.dim() != 1
        or token_batch.dim() != 1
        or not atom_batch.is_contiguous()
        or not token_batch.is_contiguous()
        or atom_batch.shape[0] != atom.shape[0]
    ):
        return False
    if atom_mask is not None and (
        atom_mask.dim() not in (1, 2)
        or atom_mask.shape[0] != atom.shape[0]
        or (atom.dim() == 2 and atom_mask.dim() != 1)
        or not atom_mask.is_contiguous()
        or atom_mask.is_complex()
    ):
        return False
    supported = NESTED_ATOM_TO_TOKEN_REDUCTION.supports(
        _nested_atom_to_token_problem(atom, atom_mask, atom_offsets, token_batch)
    )
    if atom_mask is not None and atom_mask.dim() == 2:
        sample_size = 1 if atom.dim() == 2 else atom.shape[1]
        mask_sample_size = atom_mask.shape[1]
        supported = supported & ((mask_sample_size == 1) | (mask_sample_size == sample_size))
    return supported


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


def _run_nested_atom_to_token_reduce_forward(
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
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_operand(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        atom_offsets = _kernel_operand(atom_offsets)
        token_offsets = _kernel_operand(token_offsets)
        atom_batch = _kernel_operand(atom_batch)
        token_batch = _kernel_operand(token_batch)
        plan = _resolve_nested_atom_to_token_plan(
            _nested_atom_to_token_problem(atom, atom_mask, atom_offsets, token_batch),
            atom.device,
        )
        problem = plan.problem
        device = atom.device
        sample_kind, segment_start, token_count, permutation = _nested_mapping_metadata(
            atom_to_token, atom_offsets, token_offsets, atom_batch, token_batch, plan
        )

        output_shape = (
            (problem.total_tokens, problem.channels)
            if atom.dim() == 2
            else (problem.total_tokens, problem.sample_size, problem.channels)
        )
        output = torch.empty(output_shape, device=device, dtype=atom.dtype)
        if plan.segment_grid[0] > 0:
            placeholder = atom if atom_mask is None else atom_mask
            mask_sample_size = _nested_mask_sample_size(atom_mask)
            kernel._nested_atom_to_token_reduce_forward[plan.segment_grid](
                atom,
                placeholder,
                atom_to_token,
                permutation,
                atom_offsets,
                token_offsets,
                token_batch,
                sample_kind,
                segment_start,
                token_count,
                output,
                problem.total_tokens,
                problem.sample_size,
                mask_sample_size,
                problem.channels,
                problem.has_atom_mask,
                normalize,
                float(denominator_offset),
                atom.dtype in HALF_DTYPES,
                plan.block_channels,
                plan.block_atoms,
                num_warps=plan.num_warps,
            )
        return output, sample_kind, segment_start, token_count, permutation


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


def _run_nested_atom_to_token_reduce_backward(
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
    with cuda_device_context(grad_out.device):
        grad_out = _kernel_operand(grad_out)
        atom_mask = _kernel_operand(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        permutation = _kernel_operand(permutation)
        atom_offsets = _kernel_operand(atom_offsets)
        token_offsets = _kernel_operand(token_offsets)
        token_batch = _kernel_operand(token_batch)
        sample_kind = _kernel_operand(sample_kind)
        segment_start = _kernel_operand(segment_start)
        token_count = _kernel_operand(token_count)
        problem = NestedAtomToTokenReductionProblem(
            num_samples=atom_offsets.shape[0] - 1,
            total_atoms=total_atoms,
            total_tokens=grad_out.shape[0],
            sample_size=1 if grad_out.dim() == 2 else grad_out.shape[1],
            channels=grad_out.shape[-1],
            dtype=grad_out.dtype,
            has_atom_mask=atom_mask is not None,
            mask_vjp=False,
            shared_mask_vjp=False,
        )
        plan = _resolve_nested_atom_to_token_plan(problem, grad_out.device)
        grad_atom = torch.empty(
            (problem.total_atoms, *grad_out.shape[1:]),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        if plan.segment_grid[0] > 0:
            placeholder = grad_out if atom_mask is None else atom_mask
            mask_sample_size = _nested_mask_sample_size(atom_mask)
            kernel._nested_atom_to_token_reduce_backward[plan.segment_grid](
                grad_out.contiguous(),
                placeholder,
                atom_to_token,
                permutation,
                atom_offsets,
                token_offsets,
                token_batch,
                sample_kind,
                segment_start,
                token_count,
                grad_atom,
                problem.total_tokens,
                problem.sample_size,
                mask_sample_size,
                problem.channels,
                problem.has_atom_mask,
                normalize,
                float(denominator_offset),
                grad_atom.dtype in HALF_DTYPES,
                plan.block_channels,
                plan.block_atoms,
                num_warps=plan.num_warps,
            )
        return grad_atom


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


def _run_nested_atom_to_token_reduce_mask_backward(
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
    with cuda_device_context(atom.device):
        atom = _kernel_operand(atom)
        atom_mask = _kernel_operand(atom_mask)
        atom_to_token = _kernel_operand(atom_to_token)
        permutation = _kernel_operand(permutation)
        grad_out = _kernel_operand(grad_out)
        atom_offsets = _kernel_operand(atom_offsets)
        token_offsets = _kernel_operand(token_offsets)
        token_batch = _kernel_operand(token_batch)
        sample_kind = _kernel_operand(sample_kind)
        segment_start = _kernel_operand(segment_start)
        token_count = _kernel_operand(token_count)
        problem = _nested_atom_to_token_problem(atom, atom_mask, atom_offsets, token_batch)
        plan = _resolve_nested_atom_to_token_plan(problem, atom.device)
        if normalize:
            numerator = torch.empty(grad_out.shape, device=atom.device, dtype=torch.float32)
            raw_denominator = torch.empty(grad_out.shape[:-1], device=atom.device, dtype=torch.float32)
        else:
            numerator = torch.empty((), device=atom.device, dtype=torch.float32)
            raw_denominator = numerator
        grad_mask = torch.empty(atom_mask.shape, device=atom_mask.device, dtype=atom_mask.dtype)
        mask_sample_size = _nested_mask_sample_size(atom_mask)
        mask_partial = (
            torch.empty(plan.mask_partial_shape, device=atom.device, dtype=torch.float32)
            if plan.mask_partial_shape is not None
            else grad_mask
        )
        if plan.mask_vjp_grid[0] > 0:
            if normalize:
                kernel._nested_atom_to_token_reduce_mask_state[plan.segment_grid](
                    atom,
                    atom_mask,
                    atom_to_token,
                    permutation,
                    atom_offsets,
                    token_offsets,
                    token_batch,
                    sample_kind,
                    segment_start,
                    token_count,
                    numerator,
                    raw_denominator,
                    problem.total_tokens,
                    problem.sample_size,
                    mask_sample_size,
                    problem.channels,
                    plan.block_channels,
                    plan.block_atoms,
                    num_warps=plan.num_warps,
                )
            kernel._nested_atom_to_token_reduce_mask_backward[plan.mask_vjp_grid](
                atom,
                atom_to_token,
                permutation,
                atom_offsets,
                token_offsets,
                token_batch,
                sample_kind,
                segment_start,
                token_count,
                numerator,
                raw_denominator,
                grad_out.contiguous(),
                grad_mask,
                mask_partial,
                problem.total_tokens,
                problem.sample_size,
                mask_sample_size,
                problem.shared_mask_vjp,
                problem.channels,
                normalize,
                float(denominator_offset),
                plan.block_channels,
                plan.block_atoms,
                num_warps=plan.num_warps,
            )
            if plan.mask_reduce_grid[0] > 0:
                kernel._nested_atom_to_token_reduce_mask_reduce_samples[plan.mask_reduce_grid](
                    mask_partial,
                    grad_mask,
                    problem.total_atoms,
                    problem.sample_size,
                    plan.mask_reduce_block_samples,
                    num_warps=plan.num_warps,
                )
        return grad_mask


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
