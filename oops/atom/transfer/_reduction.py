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

import weakref
from functools import lru_cache
from typing import Literal, NamedTuple

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import (
    TensorCacheKey,
    is_compiling,
    is_external_fake,
    require_no_forward_mode,
    require_same_device,
    tensor_cache_key,
    unsupported,
)

from ..layout import (
    AtomTokenLayout,
    AtomTokenPlan,
    AtomTokenSortedLayout,
    _is_non_decreasing,
    _regular_atoms_per_token,
    _sorted_atom_token_layout,
)
from .nested import validate_nested_atom_to_token_reduce_offsets
from .triton import (
    supports_atom_to_token_reduce,
    supports_atom_to_token_reduce_layout,
    supports_batched_atom_to_token_reduce,
    supports_nested_atom_to_token_reduce,
    supports_shared_general_atom_to_token_reduce,
    triton_atom_to_token_reduce,
    triton_batched_atom_to_token_reduce,
    triton_nested_atom_to_token_reduce,
    triton_prepared_atom_to_token_reduce,
    triton_shared_general_atom_to_token_reduce,
)

_SharedLayoutKey = tuple[int, TensorCacheKey, int, int]
_INDEX_STATS_CACHE: dict[
    _SharedLayoutKey,
    tuple[weakref.ReferenceType[Tensor], _IndexStats, Tensor | None],
] = {}


class _IndexStats(NamedTuple):
    num_tokens: int
    is_valid: bool
    is_sorted: bool
    max_atoms_per_token: int
    regular_atoms_per_token: int | None = None
    interleaved_atoms_per_token: int | None = None
    sorted_layout: AtomTokenSortedLayout | None = None


def atom_to_token_reduce(
    atom: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor | AtomTokenLayout,
    *,
    num_tokens: int | None = None,
    token_reference: NestedTensor | None = None,
    atom_mask: Tensor | NestedTensor | None = None,
    normalize: bool,
    denominator_offset: float,
) -> Tensor | NestedTensor:

    operator = "atom_to_token_mean" if normalize else "atom_to_token_sum"
    if isinstance(atom, NestedTensor):
        if not isinstance(atom_to_token, NestedTensor):
            raise ValueError(f"{operator} nested atom requires a NestedTensor atom_to_token mapping")
        if token_reference is None:
            raise ValueError(f"{operator} nested atom requires token_reference=NestedTensor(...)")
        if atom_mask is not None and not isinstance(atom_mask, NestedTensor):
            raise ValueError(f"{operator} nested atom requires a NestedTensor atom_mask")
        if num_tokens is not None:
            raise ValueError(f"{operator} nested atom uses token_reference instead of num_tokens")
        return _atom_to_token_reduce_nested(
            atom,
            atom_to_token,
            token_reference,
            atom_mask,
            normalize,
            denominator_offset,
        )
    if isinstance(atom_to_token, NestedTensor) or isinstance(atom_mask, NestedTensor):
        raise ValueError(f"{operator} nested operands require a NestedTensor atom input")
    if token_reference is not None:
        raise ValueError(f"{operator} token_reference is only valid for NestedTensor atom input")

    if atom.dim() < 2:
        raise ValueError("atom must have shape (..., N_atom, C)")
    if atom_mask is not None and atom_mask.shape != atom.shape[:-1]:
        raise ValueError("atom_mask must have shape (..., N_atom)")
    require_no_forward_mode(operator, atom, atom_mask)
    index = atom_to_token.atom_to_token if isinstance(atom_to_token, AtomTokenLayout) else atom_to_token
    require_same_device(operator, atom, index, atom_mask)
    if isinstance(atom_to_token, AtomTokenLayout):
        layout = _check_num_tokens(atom_to_token, num_tokens)
        plan, mapping, permutation, inverse_permutation = _layout_route(layout)
        token_offsets = None if layout.sorted_layout is None else layout.sorted_layout.token_offsets
        prepared = triton_prepared_atom_to_token_reduce(
            atom,
            atom_mask,
            mapping,
            permutation,
            inverse_permutation,
            plan,
            denominator_offset,
            normalize,
            token_offsets,
            layout.atom_to_token,
        )
        if prepared is not None:
            return prepared
        if plan is None or not supports_atom_to_token_reduce(atom, mapping, atom_mask, plan):
            raise unsupported(
                operator,
                "requires a contiguous CUDA float16/bfloat16/float32 atom within CUDA launch "
                "limits, an optional CUDA atom_mask that is contiguous or "
                "broadcast over trailing batch dimensions, and an AtomTokenLayout carrying "
                "either atoms_per_token the regular kernel can serve or, from "
                "AtomTokenLayout.from_atom_to_token(..., sort=True), a sorted_layout whose "
                "widest token the sorted kernel can serve",
                atom_dtype=atom.dtype,
                device=atom.device,
                atom_mask_shape=None if atom_mask is None else tuple(atom_mask.shape),
                atom_mask_stride=None if atom_mask is None else atom_mask.stride(),
                atom_mask_device=None if atom_mask is None else atom_mask.device,
                atom_mask_requires_grad=None if atom_mask is None else atom_mask.requires_grad,
                num_tokens=layout.num_tokens,
                atoms_per_token=layout.atoms_per_token,
                max_atoms_per_token=(
                    None if layout.sorted_layout is None else layout.sorted_layout.max_atoms_per_token
                ),
                atoms=layout.atom_to_token.numel(),
            )
        return triton_atom_to_token_reduce(
            atom,
            atom_mask,
            mapping,
            permutation,
            inverse_permutation,
            plan,
            denominator_offset,
            normalize,
            token_offsets,
        )
    if atom_to_token.dtype == torch.long:
        if num_tokens is None:
            raise unsupported(
                operator,
                "requires num_tokens for a bare long index",
                atom_to_token_shape=tuple(atom_to_token.shape),
                hint="pass num_tokens=..., or wrap regular metadata in AtomTokenLayout",
            )
        if _shares_one_index(atom_to_token):
            compiling = is_compiling()
            atom_type = type(atom)
            external_fake = (
                not compiling
                and atom_type is not Tensor
                and atom_type is not torch.nn.Parameter
                and is_external_fake(atom)
            )
            if compiling or external_fake or (atom.is_cuda and torch.cuda.is_current_stream_capturing()):
                shared_index = _shared_index(atom_to_token)
                if not supports_shared_general_atom_to_token_reduce(
                    atom, shared_index, atom_mask, num_tokens=int(num_tokens)
                ):
                    raise unsupported(
                        operator,
                        "requires contiguous CUDA float16/bfloat16/float32 atoms "
                        "and a contiguous CUDA long shared mapping",
                        atom_shape=tuple(atom.shape),
                        atom_dtype=atom.dtype,
                        atom_to_token_shape=tuple(atom_to_token.shape),
                        num_tokens=int(num_tokens),
                    )
                return triton_shared_general_atom_to_token_reduce(
                    atom,
                    atom_mask,
                    shared_index,
                    num_tokens=int(num_tokens),
                    denominator_offset=float(denominator_offset),
                    normalize=normalize,
                )
            shared_atom_to_token = _shared_index(atom_to_token)
            stats = _index_stats(atom_to_token, shared_atom_to_token, num_tokens)
            return _dispatch_bare_index(
                atom,
                atom_mask,
                shared_atom_to_token,
                stats,
                denominator_offset,
                normalize,
            )

        mapping_batch_repeat = _dense_mapping_batch_repeat(atom, atom_to_token)
        if not supports_batched_atom_to_token_reduce(
            atom,
            atom_to_token,
            atom_mask,
            mapping_batch_repeat=mapping_batch_repeat,
            num_tokens=int(num_tokens),
        ):
            raise unsupported(
                operator,
                "requires a contiguous CUDA float16/bfloat16/float32 atom and contiguous CUDA long "
                "mapping whose batch dimensions either match atom exactly or omit only its final "
                "leading sample axis",
                atom_shape=tuple(atom.shape),
                atom_dtype=atom.dtype,
                atom_device=atom.device,
                atom_to_token_shape=tuple(atom_to_token.shape),
                atom_to_token_stride=atom_to_token.stride(),
                atom_mask_shape=None if atom_mask is None else tuple(atom_mask.shape),
                atom_mask_stride=None if atom_mask is None else atom_mask.stride(),
                num_tokens=int(num_tokens),
                mapping_batch_repeat=mapping_batch_repeat,
            )
        return triton_batched_atom_to_token_reduce(
            atom,
            atom_mask,
            atom_to_token,
            num_tokens=int(num_tokens),
            mapping_batch_repeat=mapping_batch_repeat,
            denominator_offset=float(denominator_offset),
            normalize=normalize,
        )
    if atom_to_token.dim() < 2:
        raise ValueError("dense atom_to_token must have shape (..., N_atom, N_token)")
    if atom_to_token.shape[-2] != atom.shape[-2]:
        raise ValueError("dense atom_to_token atom dimension must match atom")
    with torch.autocast(atom.device.type, enabled=False):
        weights = atom_to_token.to(dtype=atom.dtype)
        if atom_mask is not None:
            weights = weights * atom_mask.to(dtype=atom.dtype).unsqueeze(-1)
        numerator = torch.matmul(weights.transpose(-1, -2), atom)
        if not normalize:
            return numerator
        count = weights.sum(dim=-2)
        denominator = torch.where(
            count > 0,
            count + denominator_offset,
            torch.ones_like(count),
        ).unsqueeze(-1)
        return numerator / denominator


def _atom_to_token_reduce_nested(
    atom: NestedTensor,
    atom_to_token: NestedTensor,
    token_reference: NestedTensor,
    atom_mask: NestedTensor | None,
    normalize: bool,
    denominator_offset: float,
) -> NestedTensor:
    operator = "atom_to_token_mean" if normalize else "atom_to_token_sum"
    atom_values, mapping_values, token_values, mask_values = _canonical_nested_reduce_inputs(
        atom,
        atom_to_token,
        token_reference,
        atom_mask,
        operator,
    )
    require_no_forward_mode(operator, atom_values, mask_values)
    require_same_device(operator, atom_values, mapping_values, mask_values)

    token_cpu_offsets = token_reference.ragged_level_offsets(0, dtype=torch.int64)
    atom_cpu_offsets = atom.ragged_level_offsets(0, dtype=torch.int64)
    mapping_cpu_offsets = atom_to_token.ragged_level_offsets(0, dtype=torch.int64)
    mask_cpu_offsets = None if atom_mask is None else atom_mask.ragged_level_offsets(0, dtype=torch.int64)
    validation = validate_nested_atom_to_token_reduce_offsets(
        token_cpu_offsets,
        atom_cpu_offsets,
        mapping_cpu_offsets,
        mask_cpu_offsets,
        token_values.shape[0],
        atom_values.shape[0],
        mapping_values.shape[0],
        0 if mask_values is None else mask_values.shape[0],
    )
    if is_compiling() or not is_external_fake(validation):
        torch._check_tensor_all(validation)
    else:
        torch._assert_async(validation)

    device = atom_values.device
    atom_offsets = atom.ragged_level_offsets(0, device=device, dtype=torch.int64)
    token_offsets = token_reference.ragged_level_offsets(0, device=device, dtype=torch.int64)
    atom_batch = atom.packed_batch_indices(device=device, dtype=torch.int32)
    token_batch = token_reference.packed_batch_indices(device=device, dtype=torch.int32)
    if not supports_nested_atom_to_token_reduce(
        atom_values,
        mask_values,
        mapping_values,
        atom_offsets,
        token_offsets,
        atom_batch,
        token_batch,
    ):
        raise unsupported(
            operator,
            "requires canonical packed CUDA float16/bfloat16/float32 NestedTensor atom/token layouts, "
            "a matching local-long mapping and optional contiguous non-complex atom mask within launch limits",
            atom_shape=tuple(atom_values.shape),
            atom_dtype=atom_values.dtype,
            device=atom_values.device,
            mapping_shape=tuple(mapping_values.shape),
            mask_shape=None if mask_values is None else tuple(mask_values.shape),
            token_shape=tuple(token_values.shape),
        )
    output_values = triton_nested_atom_to_token_reduce(
        atom_values,
        mask_values,
        mapping_values,
        atom_offsets,
        token_offsets,
        atom_batch,
        token_batch,
        float(denominator_offset),
        normalize,
    )
    return _nested_reduce_output(token_reference, atom_values, output_values)


def _canonical_nested_reduce_inputs(
    atom: NestedTensor,
    atom_to_token: NestedTensor,
    token_reference: NestedTensor,
    atom_mask: NestedTensor | None,
    operator: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    operands = (atom, atom_to_token, token_reference, *((atom_mask,) if atom_mask is not None else ()))
    if any(not operand.batch_first for operand in operands):
        raise ValueError(f"{operator} NestedTensor batch must be the leading logical dimension")
    if len(atom) == 0:
        raise unsupported(
            operator,
            "does not serve an empty NestedTensor wrapper batch without retained element rank",
        )
    if (
        len(atom) != len(atom_to_token)
        or len(atom) != len(token_reference)
        or (atom_mask is not None and len(atom) != len(atom_mask))
    ):
        raise ValueError(f"{operator} nested token, atom, mapping, and mask batches must match")
    implicit_atom_samples = atom.dim() == 3 and atom.ragged_dims == (0,) and atom.packed_dim_order == (0, 1)
    explicit_atom_samples = atom.dim() == 4 and atom.ragged_dims == (1,) and atom.packed_dim_order == (1, 0, 2)
    if not implicit_atom_samples and not explicit_atom_samples:
        raise ValueError(f"{operator} nested atom elements must use atom-major (A_i, C) or (S, A_i, C) topology")
    if atom_to_token.dim() != 2 or atom_to_token.ragged_dims != (0,) or atom_to_token.packed_dim_order != (0,):
        raise ValueError(f"{operator} nested atom_to_token elements must use canonical (A_i,) topology")
    implicit_reference_samples = (
        token_reference.dim() == 3
        and token_reference.ragged_dims == (0,)
        and token_reference.packed_dim_order == (0, 1)
    )
    explicit_reference_samples = (
        token_reference.dim() == 4
        and token_reference.ragged_dims == (1,)
        and token_reference.packed_dim_order == (1, 0, 2)
    )
    if not implicit_reference_samples and not explicit_reference_samples:
        raise ValueError(
            f"{operator} token_reference elements must use token-major " "(N_i, C_ref) or (S_ref, N_i, C_ref) topology"
        )
    if implicit_atom_samples and not implicit_reference_samples:
        raise ValueError(f"{operator} unsampled atom requires an unsampled token_reference")

    implicit_mask_samples = atom_mask is None or (
        atom_mask.dim() == 2 and atom_mask.ragged_dims == (0,) and atom_mask.packed_dim_order == (0,)
    )
    explicit_mask_samples = atom_mask is not None and (
        atom_mask.dim() == 3 and atom_mask.ragged_dims == (1,) and atom_mask.packed_dim_order == (1, 0)
    )
    if not implicit_mask_samples and not explicit_mask_samples:
        raise ValueError(f"{operator} nested atom_mask elements must use atom-major (A_i,) or (S_mask, A_i) topology")
    if implicit_atom_samples and explicit_mask_samples:
        raise ValueError(f"{operator} unsampled atom requires an unsampled atom_mask")

    atom_values = atom.concat
    mapping_values = atom_to_token.concat
    token_values = token_reference.concat
    mask_values = None if atom_mask is None else atom_mask.concat
    if atom_values.dim() not in (2, 3) or atom_values.shape[-1] <= 0:
        raise ValueError(f"{operator} nested atom packed values must have shape (sum A_i, [S,] C)")
    sample_size = 1 if atom_values.dim() == 2 else atom_values.shape[1]
    if atom_values.dim() == 3 and sample_size <= 0:
        raise ValueError(f"{operator} nested sampled atom requires S > 0")
    if mapping_values.dim() != 1 or mapping_values.dtype is not torch.long:
        raise ValueError(f"{operator} nested atom_to_token packed values must be one-dimensional long")
    if token_values.dim() not in (2, 3):
        raise ValueError(f"{operator} token_reference packed values must have shape (sum N_i, [S_ref,] C_ref)")
    if token_values.dim() == 3:
        _check_nested_reduce_sample_axis(operator, "token_reference", token_values.shape[1], sample_size)
    if mask_values is not None and (mask_values.dim() not in (1, 2) or mask_values.is_complex()):
        raise ValueError(
            f"{operator} nested atom_mask packed values must have shape (sum A_i, [S_mask]) " "and be non-complex"
        )
    if mask_values is not None and mask_values.dim() == 2:
        _check_nested_reduce_sample_axis(operator, "atom_mask", mask_values.shape[1], sample_size)
    if (
        not atom_values.is_contiguous()
        or not mapping_values.is_contiguous()
        or (mask_values is not None and not mask_values.is_contiguous())
    ):
        raise unsupported(operator, "requires contiguous NestedTensor packed atom, mapping, and mask values")
    return atom_values, mapping_values, token_values, mask_values


def _check_nested_reduce_sample_axis(operator: str, operand: str, observed: int, sample_size: int) -> None:
    message = f"{operator} {operand} sample axis must be uniformly 1 or S"
    valid = (observed == 1) | (observed == sample_size)
    if is_compiling():
        torch._check(valid, lambda: message)
    elif not valid:
        raise ValueError(message)


def _nested_reduce_output(
    token_reference: NestedTensor,
    atom_values: Tensor,
    output_values: Tensor,
) -> NestedTensor:

    output = token_reference.packed_with_static_tail(output_values)
    if atom_values.dim() == 3 and token_reference.dim() == 3:
        # The physical sampled output remains (sum N_i, S, C).
        return output.movedim(2, 1)
    return output


def _check_num_tokens(layout: AtomTokenLayout, num_tokens: int | None) -> AtomTokenLayout:
    if num_tokens is not None and int(num_tokens) != layout.num_tokens:
        raise ValueError("num_tokens must match AtomTokenLayout.num_tokens")
    return layout


@lru_cache(maxsize=256)
def _layout_plan(kind: Literal["regular", "interleaved", "sorted"], num_tokens: int, width: int) -> AtomTokenPlan:
    return AtomTokenPlan(kind, num_tokens, width)


def _layout_route(
    layout: AtomTokenLayout,
) -> tuple[AtomTokenPlan | None, Tensor, Tensor | None, Tensor | None]:

    plain = (None, layout.atom_to_token, None, None)
    if layout.num_tokens <= 0:
        return plain
    atoms_per_token = layout.atoms_per_token
    if atoms_per_token is not None and atoms_per_token > 0:
        return _layout_plan("regular", layout.num_tokens, atoms_per_token), layout.atom_to_token, None, None
    sorted_layout = layout.sorted_layout
    if sorted_layout is None:
        return plain
    return (
        _layout_plan("sorted", layout.num_tokens, sorted_layout.max_atoms_per_token),
        sorted_layout.sorted_atom_to_token,
        sorted_layout.permutation,
        sorted_layout.inverse_permutation,
    )


def _dispatch_bare_index(
    atom: Tensor,
    atom_mask: Tensor | None,
    atom_to_token: Tensor,
    stats: _IndexStats,
    denominator_offset: float,
    normalize: bool,
) -> Tensor:

    plan = _bare_index_plan(atom_to_token, stats, atom.shape[-2])
    index, permutation, inverse_permutation = _bare_index_route(atom_to_token, stats, plan)
    if plan is not None and supports_atom_to_token_reduce(atom, index, atom_mask, plan):
        return triton_atom_to_token_reduce(
            atom,
            atom_mask,
            index,
            permutation,
            inverse_permutation,
            plan,
            denominator_offset,
            normalize,
        )
    if (
        plan is None
        and stats.is_valid
        and supports_shared_general_atom_to_token_reduce(
            atom,
            atom_to_token,
            atom_mask,
            num_tokens=stats.num_tokens,
        )
    ):
        return triton_shared_general_atom_to_token_reduce(
            atom,
            atom_mask,
            atom_to_token,
            num_tokens=stats.num_tokens,
            denominator_offset=denominator_offset,
            normalize=normalize,
        )
    raise unsupported(
        "atom_to_token_mean" if normalize else "atom_to_token_sum",
        "requires a contiguous CUDA float16/bfloat16/float32 atom within CUDA launch limits, an "
        "optional CUDA atom_mask that is contiguous or broadcast over trailing "
        "batch dimensions, and a valid regular, interleaved, sorted or arbitrary index",
        atom_dtype=atom.dtype,
        device=atom.device,
        atom_mask_shape=None if atom_mask is None else tuple(atom_mask.shape),
        atom_mask_stride=None if atom_mask is None else atom_mask.stride(),
        atom_mask_device=None if atom_mask is None else atom_mask.device,
        atom_mask_requires_grad=None if atom_mask is None else atom_mask.requires_grad,
        num_tokens=stats.num_tokens,
        index_valid=stats.is_valid,
        atoms_per_token=stats.regular_atoms_per_token,
        interleaved_atoms_per_token=stats.interleaved_atoms_per_token,
        is_sorted=stats.is_sorted,
        max_atoms_per_token=stats.max_atoms_per_token,
    )


def _bare_index_plan(atom_to_token: Tensor, stats: _IndexStats, atoms: int) -> AtomTokenPlan | None:

    index_readable = atom_to_token.is_cuda and atom_to_token.is_contiguous() and atom_to_token.numel() == atoms
    if stats.regular_atoms_per_token is not None and index_readable:
        plan = AtomTokenPlan("regular", stats.num_tokens, stats.regular_atoms_per_token)
        if supports_atom_to_token_reduce_layout(plan, atoms=atoms):
            return plan
    interleaved = stats.interleaved_atoms_per_token
    if interleaved is not None and atoms == stats.num_tokens * interleaved:
        plan = AtomTokenPlan("interleaved", stats.num_tokens, interleaved)
        if supports_atom_to_token_reduce_layout(plan, atoms=atoms):
            return plan
    if stats.is_sorted and index_readable and atoms > 0 and stats.max_atoms_per_token > 0:
        plan = AtomTokenPlan("sorted", stats.num_tokens, stats.max_atoms_per_token)
        if supports_atom_to_token_reduce_layout(plan, atoms=atoms):
            return plan
    sorted_layout = stats.sorted_layout
    if sorted_layout is not None and index_readable and atoms > 0:
        plan = AtomTokenPlan("sorted", stats.num_tokens, sorted_layout.max_atoms_per_token)
        if supports_atom_to_token_reduce_layout(plan, atoms=atoms):
            return plan
    return None


def _bare_index_route(
    atom_to_token: Tensor,
    stats: _IndexStats,
    plan: AtomTokenPlan | None,
) -> tuple[Tensor, Tensor | None, Tensor | None]:

    sorted_layout = stats.sorted_layout
    if sorted_layout is None or plan is None or plan.kind != "sorted":
        return atom_to_token, None, None
    return (
        sorted_layout.sorted_atom_to_token,
        sorted_layout.permutation,
        sorted_layout.inverse_permutation,
    )


def _shares_one_index(atom_to_token: Tensor) -> bool:

    if atom_to_token.dim() == 1:
        return True
    if atom_to_token.dim() < 2 or atom_to_token.shape[-1] == 0:
        return False
    return all(size == 1 or stride == 0 for size, stride in zip(atom_to_token.shape[:-1], atom_to_token.stride()[:-1]))


def _dense_mapping_batch_repeat(atom: Tensor, atom_to_token: Tensor) -> int:

    if atom_to_token.dim() < 2 or atom_to_token.shape[-1] != atom.shape[-2]:
        raise ValueError("index atom_to_token atom dimension must match atom")
    atom_batch = atom.shape[:-2]
    mapping_batch = atom_to_token.shape[:-1]
    if mapping_batch == atom_batch:
        return 1
    if atom.dim() >= 4 and mapping_batch == atom.shape[:-3]:
        sample_size = atom.shape[-3]
        if sample_size <= 0:
            raise ValueError("atom_to_token_mean explicit sample axis S must be positive")
        return sample_size
    raise ValueError(
        "index atom_to_token must have shape (N_atom,), match all atom batch dimensions, "
        "or omit only the final leading sample axis"
    )


def _shared_index(atom_to_token: Tensor) -> Tensor:

    return atom_to_token if atom_to_token.dim() == 1 else atom_to_token[(0,) * (atom_to_token.dim() - 1)]


def _cached_index_stats(owner: Tensor, index: Tensor, num_tokens: int) -> _IndexStats:
    cache_key = _shared_layout_cache_key(owner, num_tokens)
    if cache_key is None:
        return _compute_index_stats(index, num_tokens)
    cached = _INDEX_STATS_CACHE.get(cache_key)
    # CUDA writes need not advance the Python version counter.
    if cached is not None and cached[0]() is owner and (cached[2] is None or torch.equal(index, cached[2])):
        return cached[1]
    # Integer metadata must remain saveable by later autograd calls.
    with torch.inference_mode(False):
        stats = _compute_index_stats(index, num_tokens)
        snapshot = index.clone() if index.is_cuda else None
    if len(_INDEX_STATS_CACHE) >= 16 and cache_key not in _INDEX_STATS_CACHE:
        del _INDEX_STATS_CACHE[next(iter(_INDEX_STATS_CACHE))]

    def discard_stats(_owner: weakref.ReferenceType[Tensor]) -> None:
        _INDEX_STATS_CACHE.pop(cache_key, None)

    owner_ref = weakref.ref(owner, discard_stats)
    _INDEX_STATS_CACHE[cache_key] = (owner_ref, stats, snapshot)
    return stats


def _index_stats(owner: Tensor, atom_to_token: Tensor, num_tokens: int) -> _IndexStats:

    return _cached_index_stats(owner, atom_to_token, num_tokens)


def _compute_index_stats(atom_to_token: Tensor, num_tokens: int) -> _IndexStats:
    tokens = int(num_tokens)
    if atom_to_token.numel() == 0 or tokens <= 0:
        return _IndexStats(tokens, is_valid=False, is_sorted=True, max_atoms_per_token=0)
    min_token = int(atom_to_token.min().item())
    max_token = int(atom_to_token.max().item())
    if min_token < 0 or max_token >= tokens:
        return _IndexStats(tokens, is_valid=False, is_sorted=False, max_atoms_per_token=0)
    regular_atoms_per_token = _regular_atoms_per_token(atom_to_token, tokens)
    interleaved_atoms_per_token = _interleaved_atoms_per_token(atom_to_token, tokens)
    if regular_atoms_per_token is not None:
        return _IndexStats(
            tokens,
            is_valid=True,
            is_sorted=True,
            max_atoms_per_token=regular_atoms_per_token,
            regular_atoms_per_token=regular_atoms_per_token,
            interleaved_atoms_per_token=interleaved_atoms_per_token,
        )
    if interleaved_atoms_per_token is not None and tokens > 1:
        return _IndexStats(
            tokens,
            is_valid=True,
            is_sorted=False,
            max_atoms_per_token=0,
            interleaved_atoms_per_token=interleaved_atoms_per_token,
        )
    is_sorted = _is_non_decreasing(atom_to_token)
    if not is_sorted:
        return _IndexStats(
            tokens,
            is_valid=True,
            is_sorted=False,
            max_atoms_per_token=0,
            regular_atoms_per_token=regular_atoms_per_token,
            interleaved_atoms_per_token=interleaved_atoms_per_token,
            sorted_layout=_sorted_atom_token_layout(atom_to_token, tokens, sort=True),
        )
    counts = torch.bincount(atom_to_token, minlength=tokens)
    max_atoms_per_token = int(counts.max().item()) if counts.numel() else 0
    return _IndexStats(
        tokens,
        is_valid=True,
        is_sorted=True,
        max_atoms_per_token=max_atoms_per_token,
        regular_atoms_per_token=regular_atoms_per_token,
        interleaved_atoms_per_token=interleaved_atoms_per_token,
    )


def _interleaved_atoms_per_token(atom_to_token: Tensor, num_tokens: int) -> int | None:
    atoms = atom_to_token.numel()
    if num_tokens <= 0 or atoms % num_tokens:
        return None
    atoms_per_token = atoms // num_tokens
    if atoms_per_token <= 0:
        return None
    expected = torch.arange(atoms, device=atom_to_token.device, dtype=atom_to_token.dtype) % num_tokens
    return atoms_per_token if torch.equal(atom_to_token, expected) else None


def _shared_layout_cache_key(
    atom_to_token: Tensor,
    num_tokens: int,
) -> _SharedLayoutKey | None:
    if atom_to_token.shape[-1] > 65536 or num_tokens > 65536:
        return None
    tensor_key = tensor_cache_key(atom_to_token)
    if tensor_key is None:
        return None
    # Descriptors cannot be reused across unsynchronized CUDA streams.
    stream = torch.cuda.current_stream(atom_to_token.device).cuda_stream if atom_to_token.is_cuda else 0
    return (id(atom_to_token), tensor_key, int(num_tokens), stream)
