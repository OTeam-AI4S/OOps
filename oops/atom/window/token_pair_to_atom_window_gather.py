from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor
from torch.nn import functional as F

from oops.runtime import is_compiling, is_external_fake, require_no_forward_mode, require_same_device, unsupported

from ..layout import AtomTokenLayout
from .nested import (
    canonicalize_nested_atom_window_metadata,
    canonicalize_nested_atom_window_metadata_eager,
    validate_nested_valid_atom_extents,
)
from .triton import (
    supports_nested_token_pair_to_atom_window_gather,
    supports_token_pair_to_atom_window_gather,
    triton_nested_token_pair_to_atom_window_gather,
    triton_token_pair_to_atom_window_gather,
)


def token_pair_to_atom_window_gather(
    token_pair: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor | AtomTokenLayout,
    n_queries: int = 32,
    n_keys: int = 128,
    num_valid_atoms: int | Tensor | None = None,
) -> Tensor | NestedTensor:
    r"""Gather token-pair features into local atom-pair windows.

    Args:
        token_pair: Features shaped ``(..., N, N, C)``, or a NestedTensor
            with canonical ``(N_i, N_i, C)`` elements.
        atom_to_token: An ``AtomTokenLayout`` or local-long mapping shaped
            ``(A,)`` / ``(..., A)``. The default dense layout requires a
            mapping shared over the batch. Nested pair inputs require matching
            local ``(A_i,)`` mapping elements.
        n_queries: Positive even query-window width.
        n_keys: Even key-window width, at least ``n_queries``.
        num_valid_atoms: Optional valid atom extent selecting clamped key
            starts. Dense extents lie in ``[1, A]``. Nested calls accept a
            shared integer or a CPU/CUDA integer vector with one extent in
            ``[0, A_i]`` per sample.

    Returns:
        Features in the input dtype, shaped
        ``(..., ceil(A / n_queries), n_queries, n_keys, C)`` or containing
        corresponding nested ``(ceil(A_i / n_queries), n_queries, n_keys, C)``
        elements. Padded query/key slots select token zero.

    Raises:
        TypeError: A nested ``num_valid_atoms`` has an invalid type or dtype.
        ValueError: Window widths, extents, shapes, devices, or nested
            topology violate the contract.
        OOpsUnsupportedError: The device, dtype, layout, or launch extent is
            unsupported, forward-mode differentiation is requested, or strict
            deterministic mode cannot serve the requested backward.
        RuntimeError: PyTorch's compiled determinism guard rejects backward, or
            CUDA extent validation fails, possibly asynchronously.

    Notes:
        The default dense layout requires contiguous CUDA float16, bfloat16,
        or float32 features and contiguous shared long indices. Nested calls
        require canonical contiguous packed CUDA storage. Dense extent-aware
        layouts also accept CPU inputs and per-example mappings.

        The default key window is centered and selects token zero outside the
        atom extent. With ``num_valid_atoms``, starts are clamped to
        ``[0, max(num_valid_atoms - n_keys, 0)]``. When fewer atoms are valid
        than ``n_keys``, starts are zero and later key slots select token zero.

        Reverse mode differentiates ``token_pair``. Strict deterministic mode
        serves eligible complete regular layouts and empty packed outputs;
        other VJPs are rejected. Warn-only mode warns and proceeds for those
        VJPs, whose final bits may vary. Forward mode is unsupported on all
        routes. Index contents are not validated here. Out-of-range token ids
        are outside the numerical contract. Validate untrusted mappings with
        ``validate_atom_to_token``.

    Examples:
        >>> import torch
        >>> from oops import token_pair_to_atom_window_gather
        >>> if torch.cuda.is_available():
        ...     pair = torch.zeros(2, 2, 4, device="cuda")
        ...     mapping = torch.tensor([0, 0, 1, 1], device="cuda")
        ...     result = token_pair_to_atom_window_gather(pair, mapping, 2, 4)
        ...     assert result.shape == (2, 2, 4, 4)
    """
    _check_window(n_queries, n_keys)
    if isinstance(token_pair, NestedTensor):
        if not isinstance(atom_to_token, NestedTensor):
            raise ValueError(
                "token_pair_to_atom_window_gather token_pair and atom_to_token must both be NestedTensor "
                "or both be dense"
            )
        return _token_pair_to_atom_window_gather_nested(
            token_pair,
            atom_to_token,
            n_queries,
            n_keys,
            num_valid_atoms,
        )
    if isinstance(atom_to_token, NestedTensor):
        raise ValueError(
            "token_pair_to_atom_window_gather token_pair and atom_to_token must both be NestedTensor or both be dense"
        )
    if token_pair.dim() < 3:
        raise ValueError("token_pair must have shape (..., N_token, N_token, C)")
    if token_pair.shape[-2] != token_pair.shape[-3]:
        raise ValueError("token_pair token dimensions must be square")
    require_no_forward_mode("token_pair_to_atom_window_gather", token_pair)
    index = atom_to_token.atom_to_token if isinstance(atom_to_token, AtomTokenLayout) else atom_to_token
    require_same_device("token_pair_to_atom_window_gather", token_pair, index)
    tokens = token_pair.shape[-2]
    if isinstance(atom_to_token, Tensor):
        shared_atom_to_token = _shared_atom_to_token(atom_to_token)
        if shared_atom_to_token is None:
            if atom_to_token.shape[:-1] != token_pair.shape[:-3]:
                raise ValueError("batched atom_to_token dimensions must match token_pair")
            if num_valid_atoms is None:
                raise unsupported(
                    "token_pair_to_atom_window_gather",
                    "requires an atom_to_token index shared across the batch, not one index per example",
                    atom_to_token_shape=tuple(atom_to_token.shape),
                    token_pair_shape=tuple(token_pair.shape),
                )
            _check_num_valid_atoms(num_valid_atoms, atom_to_token.shape[-1])
            return _token_pair_to_atom_window_gather_clamped(
                token_pair,
                atom_to_token,
                n_queries,
                n_keys,
                num_valid_atoms,
            )
        atom_to_token = shared_atom_to_token
    if num_valid_atoms is not None:
        index = atom_to_token.atom_to_token if isinstance(atom_to_token, AtomTokenLayout) else atom_to_token
        if isinstance(atom_to_token, AtomTokenLayout) and atom_to_token.num_tokens != tokens:
            raise ValueError("AtomTokenLayout.num_tokens must match token_pair")
        _check_num_valid_atoms(num_valid_atoms, index.numel())
        return _token_pair_to_atom_window_gather_clamped(
            token_pair,
            index,
            n_queries,
            n_keys,
            num_valid_atoms,
        )
    layout = _as_layout(atom_to_token, tokens=tokens)
    if not supports_token_pair_to_atom_window_gather(token_pair, layout, n_queries, n_keys):
        raise unsupported(
            "token_pair_to_atom_window_gather",
            "requires a contiguous CUDA float16/bfloat16/float32 token_pair and a contiguous CUDA "
            "atom_to_token within CUDA launch limits",
            token_pair_dtype=token_pair.dtype,
            token_pair_contiguous=token_pair.is_contiguous(),
            device=token_pair.device,
            atom_to_token_device=layout.atom_to_token.device,
            atom_to_token_contiguous=layout.atom_to_token.is_contiguous(),
        )
    return triton_token_pair_to_atom_window_gather(token_pair, layout, n_queries, n_keys)


def _token_pair_to_atom_window_gather_nested(
    token_pair: NestedTensor,
    atom_to_token: NestedTensor,
    n_queries: int,
    n_keys: int,
    num_valid_atoms: int | Tensor | None,
) -> NestedTensor:
    pair_values, mapping_values = _canonical_nested_window_inputs(token_pair, atom_to_token)
    require_no_forward_mode("token_pair_to_atom_window_gather", pair_values)
    require_same_device("token_pair_to_atom_window_gather", pair_values, mapping_values)
    compiling = is_compiling()
    pair_type = type(pair_values)
    external_fake = (
        not compiling
        and pair_type is not Tensor
        and pair_type is not torch.nn.Parameter
        and is_external_fake(pair_values)
    )

    pair_row_offsets = token_pair.ragged_level_offsets(0, dtype=torch.int64)
    pair_inner_offsets = token_pair.ragged_level_offsets(1, dtype=torch.int64)
    atom_cpu_offsets = atom_to_token.ragged_level_offsets(0, dtype=torch.int64)
    normalized_valid = _normalize_nested_num_valid_atoms(num_valid_atoms, len(token_pair))
    has_num_valid_atoms = normalized_valid is not None
    valid_cpu = normalized_valid if normalized_valid is None or normalized_valid.device.type == "cpu" else None
    metadata_op = (
        canonicalize_nested_atom_window_metadata
        if compiling or external_fake
        else canonicalize_nested_atom_window_metadata_eager
    )
    packed_metadata, block_lengths, block_to_sample, validation = metadata_op(
        pair_row_offsets,
        pair_inner_offsets,
        atom_cpu_offsets,
        valid_cpu,
        pair_values.shape[0],
        mapping_values.shape[0],
        n_queries,
        n_keys,
    )
    if compiling:
        torch._check_tensor_all(validation)
    elif external_fake:
        torch._assert_async(validation)

    device = pair_values.device
    token_offsets = token_pair.ragged_level_offsets(0, device=device, dtype=torch.int64)
    atom_offsets = atom_to_token.ragged_level_offsets(0, device=device, dtype=torch.int64)
    packed_metadata = packed_metadata.to(device=device)
    block_to_sample = block_to_sample.to(device=device)
    if normalized_valid is not None and normalized_valid.device.type != "cpu":
        require_same_device("token_pair_to_atom_window_gather", pair_values, normalized_valid)
        valid_device = normalized_valid.to(dtype=torch.int64)
        extent_validation = validate_nested_valid_atom_extents(valid_device, atom_offsets)
        if compiling:
            torch._check_tensor_all(extent_validation)
        else:
            torch._assert_async(extent_validation)
        samples = len(token_pair)
        metadata_extent = samples + 1
        packed_metadata = torch.cat(
            (
                packed_metadata[: 2 * metadata_extent],
                valid_device,
                valid_device.new_zeros(1),
            )
        )
    if not supports_nested_token_pair_to_atom_window_gather(
        pair_values,
        mapping_values,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        n_queries,
        n_keys,
        check_plan=not compiling and not external_fake,
    ):
        raise unsupported(
            "token_pair_to_atom_window_gather",
            "requires canonical packed CUDA float16/bfloat16/float32 pair and local-long atom mapping "
            "NestedTensors within segmented launch limits",
            pair_shape=tuple(pair_values.shape),
            pair_dtype=pair_values.dtype,
            mapping_shape=tuple(mapping_values.shape),
            device=pair_values.device,
            samples=len(token_pair),
        )
    output_values = triton_nested_token_pair_to_atom_window_gather(
        pair_values,
        mapping_values,
        token_offsets,
        atom_offsets,
        packed_metadata,
        block_to_sample,
        has_num_valid_atoms,
        n_queries,
        n_keys,
    )
    return atom_to_token.packed_with_lengths(output_values, block_lengths)


def _canonical_nested_window_inputs(
    token_pair: NestedTensor,
    atom_to_token: NestedTensor,
) -> tuple[Tensor, Tensor]:
    if not token_pair.batch_first or not atom_to_token.batch_first:
        raise ValueError("token_pair_to_atom_window_gather NestedTensor batch must be the leading logical dimension")
    if len(token_pair) == 0:
        raise unsupported(
            "token_pair_to_atom_window_gather",
            "does not serve an empty NestedTensor wrapper batch without retained element rank",
        )
    if len(token_pair) != len(atom_to_token):
        raise ValueError("token_pair_to_atom_window_gather nested pair and atom_to_token batches must match")
    if token_pair.dim() != 4 or token_pair.ragged_dims != (0, 1) or token_pair.packed_dim_order != (0, 1, 2):
        raise ValueError(
            "token_pair_to_atom_window_gather nested pair elements must use canonical explicit "
            "(N_i, N_i, C) topology"
        )
    if atom_to_token.dim() != 2 or atom_to_token.ragged_dims != (0,) or atom_to_token.packed_dim_order != (0,):
        raise ValueError(
            "token_pair_to_atom_window_gather nested atom_to_token elements must use canonical (A_i,) topology"
        )

    pair_values = token_pair.concat
    mapping_values = atom_to_token.concat
    if pair_values.dim() != 2 or pair_values.shape[1] <= 0:
        raise ValueError("token_pair_to_atom_window_gather nested pair packed values must have shape (sum N_i^2, C)")
    if mapping_values.dim() != 1 or mapping_values.dtype is not torch.long:
        raise ValueError(
            "token_pair_to_atom_window_gather nested atom_to_token packed values must be one-dimensional long"
        )
    if not pair_values.is_contiguous() or not mapping_values.is_contiguous():
        raise unsupported(
            "token_pair_to_atom_window_gather",
            "requires contiguous NestedTensor packed pair and atom_to_token values",
        )
    return pair_values, mapping_values


def _normalize_nested_num_valid_atoms(
    num_valid_atoms: int | Tensor | None,
    samples: int,
) -> Tensor | None:
    if num_valid_atoms is None:
        return None
    if isinstance(num_valid_atoms, bool):
        raise TypeError("num_valid_atoms must be an int, a CPU/CUDA integer tensor, or None")
    if isinstance(num_valid_atoms, int):
        return torch.full((samples,), num_valid_atoms, dtype=torch.int64)
    if isinstance(num_valid_atoms, NestedTensor) or not isinstance(num_valid_atoms, Tensor):
        raise TypeError("num_valid_atoms must be an int, a CPU/CUDA integer tensor, or None")
    if num_valid_atoms.device.type not in ("cpu", "cuda"):
        raise ValueError("nested num_valid_atoms must be a CPU or CUDA tensor")
    if (
        num_valid_atoms.dtype.is_floating_point
        or num_valid_atoms.dtype.is_complex
        or num_valid_atoms.dtype is torch.bool
    ):
        raise TypeError("nested num_valid_atoms must use an integer dtype")
    if num_valid_atoms.dim() != 1 or num_valid_atoms.shape[0] != samples:
        raise ValueError("nested num_valid_atoms must contain one value per sample")
    return num_valid_atoms.to(dtype=torch.int64)


def _check_window(n_queries: int, n_keys: int) -> None:
    if n_queries <= 0 or n_keys <= 0:
        raise ValueError("window sizes must be positive")
    if n_keys < n_queries:
        raise ValueError("n_keys must be >= n_queries")
    if n_queries % 2 or n_keys % 2:
        raise ValueError("window sizes must be even")


def _check_num_valid_atoms(num_valid_atoms: int | Tensor, atoms: int) -> None:
    if num_valid_atoms <= 0 or num_valid_atoms > atoms:
        raise ValueError("num_valid_atoms must be in [1, N_atom]")


def _as_layout(
    atom_to_token: Tensor | AtomTokenLayout,
    *,
    tokens: int,
) -> AtomTokenLayout:
    if isinstance(atom_to_token, AtomTokenLayout):
        if atom_to_token.num_tokens != tokens:
            raise ValueError("AtomTokenLayout.num_tokens must match token_pair")
        return atom_to_token
    return AtomTokenLayout.from_atom_to_token(atom_to_token, num_tokens=tokens, detect_regular=False)


def _token_pair_to_atom_window_gather_clamped(
    token_pair: Tensor,
    atom_to_token: Tensor,
    n_queries: int,
    n_keys: int,
    num_valid_atoms: int | Tensor,
) -> Tensor:
    atoms = atom_to_token.shape[-1]
    n_blocks = (atoms + n_queries - 1) // n_queries
    q_pad = n_blocks * n_queries - atoms
    pad_left = (n_keys - n_queries) // 2
    atom_to_token = atom_to_token.clamp(0, token_pair.shape[-2] - 1)
    query_indices = F.pad(atom_to_token, (0, q_pad), value=0).reshape(*atom_to_token.shape[:-1], n_blocks, n_queries)
    centered = torch.arange(n_blocks, device=atom_to_token.device) * n_queries - pad_left
    # Clamp the upper bound first so zero wins when valid_atoms < n_keys.
    starts = centered.clamp(max=num_valid_atoms - n_keys).clamp(min=0)
    key_atom_indices = starts[:, None] + torch.arange(n_keys, device=atom_to_token.device)[None, :]
    key_indices = atom_to_token[..., key_atom_indices.clamp(max=atoms - 1)]
    key_indices = key_indices.masked_fill(key_atom_indices >= num_valid_atoms, 0)

    query_indices = query_indices.to(dtype=torch.long)
    key_indices = key_indices.to(dtype=torch.long)
    prefix = tuple(token_pair.shape[:-3])
    channels = token_pair.shape[-1]
    flat_token_pair = token_pair.reshape(-1, token_pair.shape[-3], token_pair.shape[-2], channels)
    flat_query_indices = query_indices.expand(*prefix, n_blocks, n_queries).reshape(-1, n_blocks, n_queries)
    flat_key_indices = key_indices.expand(*prefix, n_blocks, n_keys).reshape(-1, n_blocks, n_keys)
    batch_index = torch.arange(flat_token_pair.shape[0], device=token_pair.device).view(-1, 1, 1, 1)
    row_index = flat_query_indices.unsqueeze(-1).expand(-1, -1, -1, n_keys)
    column_index = flat_key_indices.unsqueeze(-2).expand(-1, -1, n_queries, -1)
    gathered = flat_token_pair[batch_index, row_index, column_index]
    return gathered.reshape(*prefix, n_blocks, n_queries, n_keys, channels)


def _shared_atom_to_token(atom_to_token: Tensor) -> Tensor | None:
    if atom_to_token.dim() == 1:
        return atom_to_token
    if atom_to_token.dim() < 2 or atom_to_token.shape[-1] == 0:
        return None
    if not all(size == 1 or stride == 0 for size, stride in zip(atom_to_token.shape[:-1], atom_to_token.stride()[:-1])):
        return None
    return atom_to_token[(0,) * (atom_to_token.dim() - 1)]
