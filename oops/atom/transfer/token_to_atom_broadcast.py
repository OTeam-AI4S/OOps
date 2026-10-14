from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import is_compiling, is_external_fake, require_same_device, unsupported

from ..layout import AtomTokenLayout
from .nested import validate_nested_token_atom_offsets


def token_to_atom_broadcast(
    token: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor | AtomTokenLayout,
) -> Tensor | NestedTensor:
    r"""Broadcast token features to their assigned atoms.

    Args:
        token: Features shaped ``(..., N, C)``, or a NestedTensor containing
            ``(N_i, C)`` or explicit-sample ``(S, N_i, C)`` elements.
        atom_to_token: An ``AtomTokenLayout``, local-long indices shaped
            ``(A,)`` or ``(..., A)``, or dense assignment weights
            ``(..., A, N)``. A sampled dense input ``(*prefix, S, N, C)``
            also accepts ``(*prefix, A)`` indices shared over ``S``. Nested
            inputs require matching local-long ``(A_i,)`` mapping elements.

    Returns:
        Atom features in the token dtype, shaped ``(..., A, C)`` or containing
        nested ``(A_i, C)`` / ``(S, A_i, C)`` elements. Long indices are clamped
        to the local token range before selecting features. Assignment matrices
        apply their declared weights.

    Raises:
        ValueError: Shapes, devices, mapping topology, or a layout's token
            count disagree, or atoms reference an empty token axis.
        OOpsUnsupportedError: Nested packing is unsupported or an empty
            wrapper batch does not retain element rank.

    Notes:
        The result preserves the token dtype under autocast. Supported devices,
        dtypes, and differentiation follow PyTorch operand rules. Full-graph
        compilation supports dynamic token/atom lengths at a fixed positive
        sample count ``S``.

        Besides globally shared indices, only an exact batch match or omission
        of the final sample axis is accepted. Nested sample topology is preserved.
        Index contents are not validated here. Validate untrusted mappings with
        ``validate_atom_to_token``.

    Examples:
        >>> import torch
        >>> from oops import token_to_atom_broadcast
        >>> tokens = torch.tensor([[2.0], [9.0]])
        >>> mapping = torch.tensor([0, 0, 1])
        >>> token_to_atom_broadcast(tokens, mapping).tolist()
        [[2.0], [2.0], [9.0]]
    """

    if isinstance(token, NestedTensor):
        if not isinstance(atom_to_token, NestedTensor):
            raise ValueError(
                "token_to_atom_broadcast token and atom_to_token must both be NestedTensor or both be dense"
            )
        return _token_to_atom_broadcast_nested(token, atom_to_token)
    if isinstance(atom_to_token, NestedTensor):
        raise ValueError("token_to_atom_broadcast token and atom_to_token must both be NestedTensor or both be dense")

    if token.dim() < 2:
        raise ValueError("token must have shape (..., N_token, C)")
    if isinstance(atom_to_token, AtomTokenLayout):
        if atom_to_token.num_tokens != token.shape[-2]:
            raise ValueError("AtomTokenLayout.num_tokens must match token.shape[-2]")
        atom_to_token = atom_to_token.atom_to_token
    require_same_device("token_to_atom_broadcast", token, atom_to_token)
    if atom_to_token.dtype == torch.long:
        return _token_to_atom_broadcast_index(token, atom_to_token)
    if atom_to_token.dim() < 2:
        raise ValueError("dense atom_to_token must have shape (..., N_atom, N_token)")
    if atom_to_token.shape[-1] != token.shape[-2]:
        raise ValueError("dense atom_to_token token dimension must match token")
    with torch.autocast(token.device.type, enabled=False):
        return torch.matmul(atom_to_token.to(dtype=token.dtype), token)


def _token_to_atom_broadcast_index(token: Tensor, atom_to_token: Tensor) -> Tensor:
    tokens = token.shape[-2]
    if tokens == 0 and atom_to_token.numel() != 0:
        raise ValueError("token must contain at least one token when atom_to_token is non-empty")
    if atom_to_token.dim() == 1:
        gather_index = atom_to_token.clamp(0, tokens - 1)
        return token.index_select(-2, gather_index)
    if atom_to_token.shape[:-1] == token.shape[:-2]:
        gather_index = atom_to_token.clamp(0, tokens - 1)
        return token.gather(-2, gather_index.unsqueeze(-1).expand(*atom_to_token.shape, token.shape[-1]))
    if token.dim() >= 4 and atom_to_token.shape[:-1] == token.shape[:-3]:
        sample_size = token.shape[-3]
        if sample_size <= 0:
            raise ValueError("token_to_atom_broadcast explicit sample axis S must be positive")
        gather_index = atom_to_token.clamp(0, tokens - 1)
        expanded_index = (
            gather_index.unsqueeze(-2)
            .unsqueeze(-1)
            .expand(
                *token.shape[:-2],
                atom_to_token.shape[-1],
                token.shape[-1],
            )
        )
        return token.gather(-2, expanded_index)
    raise ValueError(
        "index atom_to_token must have shape (N_atom,), match all token batch dimensions, "
        "or omit only the final leading sample axis"
    )


def _token_to_atom_broadcast_nested(token: NestedTensor, atom_to_token: NestedTensor) -> NestedTensor:
    if not token.batch_first or not atom_to_token.batch_first:
        raise ValueError("token_to_atom_broadcast NestedTensor batch must be the leading logical dimension")
    # Batch alignment must remain checked even when no atom values are read.
    if len(token) != len(atom_to_token):
        raise ValueError("token_to_atom_broadcast token and atom_to_token NestedTensor batches must match")
    if len(token) == 0:
        raise unsupported(
            "token_to_atom_broadcast",
            "does not serve an empty NestedTensor wrapper batch without retained element rank",
        )
    unsampled_rank = token.dim() == 3
    sampled_rank = token.dim() == 4
    if not unsampled_rank and not sampled_rank:
        raise ValueError(
            "token_to_atom_broadcast nested token elements must have shape (N_i, C) or (S, N_i, C); "
            f"got logical rank {token.dim()}"
        )
    unsampled = unsampled_rank and token.ragged_dims == (0,) and token.packed_dim_order == (0, 1)
    sampled = sampled_rank and token.ragged_dims == (1,) and token.packed_dim_order == (1, 0, 2)
    if not unsampled and not sampled:
        raise unsupported(
            "token_to_atom_broadcast",
            "nested token requires canonical (N_i, C) or token-major (S, N_i, C) packing",
            ragged_dims=token.ragged_dims,
            packed_dim_order=token.packed_dim_order,
        )
    if atom_to_token.dim() != 2:
        raise ValueError(
            "token_to_atom_broadcast nested atom_to_token elements must have shape (A_i,); "
            f"got logical rank {atom_to_token.dim()}"
        )
    if atom_to_token.ragged_dims != (0,) or atom_to_token.packed_dim_order != (0,):
        raise unsupported(
            "token_to_atom_broadcast",
            "nested atom_to_token requires canonical (A_i,) packing with one ragged axis",
            ragged_dims=atom_to_token.ragged_dims,
            packed_dim_order=atom_to_token.packed_dim_order,
        )

    token_values = token.concat
    mapping_values = atom_to_token.concat
    expected_token_value_rank = 2 if unsampled else 3
    if token_values.dim() != expected_token_value_rank:
        packed_contract = "(sum N_i, C)" if unsampled else "(sum N_i, S, C)"
        raise ValueError(f"token_to_atom_broadcast nested token packed values must have shape {packed_contract}")
    if sampled and token_values.shape[1] == 0:
        raise ValueError("token_to_atom_broadcast explicit sample axis S must be positive")
    if mapping_values.dim() != 1 or mapping_values.dtype is not torch.long:
        raise ValueError("token_to_atom_broadcast nested atom_to_token packed values must be one-dimensional long")
    require_same_device("token_to_atom_broadcast", token_values, mapping_values)

    token_cpu_offsets = token.ragged_level_offsets(0, dtype=torch.int64)
    atom_cpu_offsets = atom_to_token.ragged_level_offsets(0, dtype=torch.int64)
    validation_token = validate_nested_token_atom_offsets(
        "token_to_atom_broadcast",
        token_cpu_offsets,
        atom_cpu_offsets,
        token_values.shape[0],
        mapping_values.shape[0],
    )
    # Keep empty-output validation live and FakeTensor checks tensor-valued.
    if is_compiling() or not is_external_fake(validation_token):
        torch._check_tensor_all(validation_token)
    else:
        torch._assert_async(validation_token)
    device = token_values.device
    token_offsets = token.ragged_level_offsets(0, device=device, dtype=torch.int64)
    token_lengths = token_offsets[1:] - token_offsets[:-1]
    atom_offsets = atom_to_token.ragged_level_offsets(0, device=device, dtype=torch.int64)
    atom_lengths = atom_offsets[1:] - atom_offsets[:-1]

    total_atoms = mapping_values.shape[0]
    sample_bounds = torch.stack((token_offsets[:-1], token_lengths - 1), dim=-1)
    atom_bounds = torch.repeat_interleave(sample_bounds, atom_lengths, dim=0, output_size=total_atoms)
    token_starts, token_limits = atom_bounds.unbind(-1)
    local_index = torch.minimum(mapping_values.clamp_min(0), token_limits)
    output_values = token_values.index_select(0, token_starts + local_index)
    output = atom_to_token.packed_with_static_tail(output_values)
    if unsampled:
        return output
    # Sampled values remain physically (sum A_i, S, C).
    return torch.movedim(output, -2, 1)
