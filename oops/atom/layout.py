from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from oops.sparse.csr.graph import CSRGraph


@dataclass(frozen=True, slots=True)
class AtomTokenPlan:

    kind: Literal["regular", "interleaved", "sorted"]
    num_tokens: int
    width: int

    def __post_init__(self) -> None:
        if self.kind not in ("regular", "interleaved", "sorted"):
            raise ValueError(f"unknown atom-token plan kind: {self.kind}")
        num_tokens = int(self.num_tokens)
        width = int(self.width)
        if num_tokens <= 0:
            raise ValueError("atom-token plan num_tokens must be positive")
        if width <= 0:
            raise ValueError("atom-token plan width must be positive")
        object.__setattr__(self, "num_tokens", num_tokens)
        object.__setattr__(self, "width", width)


@dataclass(frozen=True, slots=True, eq=False)
class AtomTokenSortedLayout:
    r"""``permutation[s]`` identifies the original atom at sorted position ``s``.

    ``inverse_permutation`` undoes that ordering; offsets bound sorted tokens.
    ``max_atoms_per_token`` is the largest offset gap.
    """

    sorted_atom_to_token: Tensor
    token_offsets: Tensor
    max_atoms_per_token: int
    permutation: Tensor | None = None
    inverse_permutation: Tensor | None = None

    def __post_init__(self) -> None:
        if self.sorted_atom_to_token.dtype != torch.long or self.sorted_atom_to_token.dim() != 1:
            raise ValueError("sorted_atom_to_token must be a 1-D long tensor")
        if not self.sorted_atom_to_token.is_contiguous():
            raise ValueError("sorted_atom_to_token must be contiguous")
        atoms = int(self.sorted_atom_to_token.numel())
        if self.token_offsets.dtype != torch.long or self.token_offsets.dim() != 1:
            raise ValueError("token_offsets must be a 1-D long tensor")
        if self.token_offsets.numel() == 0:
            raise ValueError("token_offsets must have shape (num_tokens + 1,)")
        if self.token_offsets.device != self.sorted_atom_to_token.device:
            raise ValueError("token_offsets must be on the same device as sorted_atom_to_token")
        max_atoms_per_token = int(self.max_atoms_per_token)
        if max_atoms_per_token <= 0:
            raise ValueError("max_atoms_per_token must be positive")
        object.__setattr__(self, "max_atoms_per_token", max_atoms_per_token)
        if (self.permutation is None) != (self.inverse_permutation is None):
            raise ValueError("permutation and inverse_permutation must be given together")
        for name, tensor in (
            ("permutation", self.permutation),
            ("inverse_permutation", self.inverse_permutation),
        ):
            if tensor is None:
                continue
            _check_sorted_layout_permutation(tensor, name, atoms)
            if tensor.device != self.sorted_atom_to_token.device:
                raise ValueError(f"{name} must be on the same device as sorted_atom_to_token")


@dataclass(frozen=True, slots=True, eq=False)
class AtomTokenLayout:
    r"""Describe a shared local atom-to-token mapping and reusable layout metadata.

    Args:
        atom_to_token: One-dimensional long indices shaped ``(A,)``.
        num_tokens: Non-negative token count.
        atoms_per_token: Optional positive width declaring the canonical
            mapping ``atom_index // atoms_per_token``. Its atom count must be
            ``num_tokens * atoms_per_token``.
        sorted_layout: Optional sorted mapping, token offsets, and paired
            permutations for an irregular layout, with the same atom count
            and device as ``atom_to_token``.

    Attributes:
        atom_to_token: Caller-owned mapping tensor.
        num_tokens: Declared token extent.
        atoms_per_token: Canonical regular width, or None.
        sorted_layout: Reordering metadata, or None.

    Raises:
        ValueError: Index rank/dtype, token extent, declared regular width,
            or sorted metadata shapes/devices are inconsistent.

    Notes:
        Construction checks tensor metadata, not all index values. Regular
        and sorted declarations are trusted: use ``regular`` or
        ``from_atom_to_token`` to construct them and ``validate_atom_to_token``
        for untrusted contents. The dataclass is frozen, but its tensors are
        mutable; callers must keep trusted declarations consistent with them.
        Instances compare by identity because tensor value equality is
        ambiguous for a metadata container.

    Examples:
        >>> from oops import AtomTokenLayout
        >>> layout = AtomTokenLayout.regular(2, 2)
        >>> layout.atom_to_token.tolist()
        [0, 0, 1, 1]
    """

    atom_to_token: Tensor
    num_tokens: int
    atoms_per_token: int | None = None
    sorted_layout: AtomTokenSortedLayout | None = None

    def __post_init__(self) -> None:
        if self.atom_to_token.dtype != torch.long:
            raise ValueError("atom_to_token must be a long tensor")
        if self.atom_to_token.dim() != 1:
            raise ValueError("atom_to_token must be a 1-D tensor")
        num_tokens = int(self.num_tokens)
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        object.__setattr__(self, "num_tokens", num_tokens)

        if self.sorted_layout is not None:
            sorted_layout = self.sorted_layout
            if sorted_layout.sorted_atom_to_token.numel() != self.atom_to_token.numel():
                raise ValueError("sorted_layout must reorder exactly as many atoms as atom_to_token")
            if sorted_layout.sorted_atom_to_token.device != self.atom_to_token.device:
                raise ValueError("sorted_layout must be on the same device as atom_to_token")
            if sorted_layout.token_offsets.numel() != num_tokens + 1:
                raise ValueError("sorted_layout token_offsets size must equal num_tokens + 1")

        if self.atoms_per_token is None:
            return
        atoms_per_token = int(self.atoms_per_token)
        if atoms_per_token <= 0:
            raise ValueError("atoms_per_token must be positive")
        if self.atom_to_token.numel() != num_tokens * atoms_per_token:
            raise ValueError("regular atom_to_token size must equal num_tokens * atoms_per_token")
        object.__setattr__(self, "atoms_per_token", atoms_per_token)

    @classmethod
    def regular(
        cls,
        num_tokens: int,
        atoms_per_token: int,
        *,
        device: torch.device | str | None = None,
    ) -> AtomTokenLayout:
        r"""Construct the canonical contiguous atom-to-token layout.

        Args:
            num_tokens: Non-negative number of tokens.
            atoms_per_token: Positive number of consecutive atoms for each token.
            device: Mapping device; None follows PyTorch's default device.

        Returns:
            An ``AtomTokenLayout`` with long indices
            ``arange(num_tokens * atoms_per_token) // atoms_per_token`` and
            trusted regular metadata.

        Raises:
            ValueError: ``num_tokens`` is negative or ``atoms_per_token`` is not
                positive.

        Examples:
            >>> from oops import AtomTokenLayout
            >>> AtomTokenLayout.regular(2, 2).atom_to_token.tolist()
            [0, 0, 1, 1]
        """

        num_tokens = int(num_tokens)
        atoms_per_token = int(atoms_per_token)
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if atoms_per_token <= 0:
            raise ValueError("atoms_per_token must be positive")
        atoms = num_tokens * atoms_per_token
        atom_to_token = torch.arange(atoms, device=device, dtype=torch.long) // atoms_per_token
        return cls(atom_to_token, num_tokens, atoms_per_token)

    @classmethod
    def from_atom_to_token(
        cls,
        atom_to_token: Tensor,
        *,
        num_tokens: int | None = None,
        detect_regular: bool = True,
        sort: bool = False,
    ) -> AtomTokenLayout:
        r"""Build reusable layout metadata from existing atom-to-token indices.

        Args:
            atom_to_token: One-dimensional long mapping shaped ``(A,)``.
            num_tokens: Token count; inferred from the maximum id when omitted,
                with zero tokens for an empty mapping.
            detect_regular: Check for canonical contiguous ordering and record
                its ``atoms_per_token`` when present.
            sort: Record sorted metadata and stable permutations for a layout
                not detected as regular. An already sorted mapping needs no
                permutation.

        Returns:
            An ``AtomTokenLayout`` retaining the input mapping and optional
            regular or sorted metadata on its device.

        Raises:
            ValueError: The mapping rank/dtype or declared extent is invalid,
                or checked CPU sorted construction receives out-of-range ids.

        Notes:
            Inferring token count, detecting regularity, or sorting metadata can
            inspect CUDA contents on the host. Build reusable layouts upstream;
            this construction is not a full-graph operator. Without ``sort=True``,
            arbitrary index contents are not validated by construction. CUDA
            sorted metadata may be absent for out-of-range ids; validate untrusted
            indices explicitly.

        Examples:
            >>> import torch
            >>> from oops import AtomTokenLayout
            >>> mapping = torch.tensor([1, 0, 1])
            >>> layout = AtomTokenLayout.from_atom_to_token(mapping, num_tokens=2, sort=True)
            >>> layout.sorted_layout.sorted_atom_to_token.tolist()
            [0, 1, 1]
        """

        if atom_to_token.dtype != torch.long:
            raise ValueError("atom_to_token must be a long tensor")
        if atom_to_token.dim() != 1:
            raise ValueError("atom_to_token must be a 1-D tensor")
        tokens = _num_tokens(atom_to_token, num_tokens)
        atoms_per_token = _regular_atoms_per_token(atom_to_token, tokens) if detect_regular else None
        sorted_layout = None
        if sort and atoms_per_token is None:
            sorted_layout = _sorted_atom_token_layout(
                atom_to_token,
                tokens,
                sort=not _is_non_decreasing(atom_to_token),
            )
        return cls(atom_to_token, tokens, atoms_per_token, sorted_layout)


def validate_atom_to_token(
    atom_to_token: Tensor | AtomTokenLayout,
    *,
    num_tokens: int | None = None,
) -> None:
    r"""Validate token-id range and any declared canonical regular layout.

    Args:
        atom_to_token: Long indices shaped ``(A,)`` or ``(..., A)``, or an
            ``AtomTokenLayout`` carrying its token count and optional regular
            width.
        num_tokens: Token count, required for bare indices. If supplied with
            a layout, it must equal the layout's count.

    Returns:
        None when every index lies in ``[0, num_tokens)`` and a declared
        regular width exactly matches ``atom_index // atoms_per_token``.

    Raises:
        ValueError: Rank/dtype/count is invalid, an id is outside its extent,
            or a regular declaration disagrees with the mapping. Content
            errors include the offending position and value.

    Notes:
        CPU and CUDA mappings are accepted. CUDA validation synchronizes with
        the host and is not full-graph traceable. Validate reusable metadata
        where it is constructed. Invalid mappings are outside transfer
        operators' numerical contract. Sorted-layout permutations are not
        checked here.

    Examples:
        >>> import torch
        >>> from oops import validate_atom_to_token
        >>> validate_atom_to_token(torch.tensor([0, 0, 1]), num_tokens=2)
    """

    atoms_per_token = None
    if isinstance(atom_to_token, AtomTokenLayout):
        if num_tokens is not None and int(num_tokens) != atom_to_token.num_tokens:
            raise ValueError("num_tokens must match AtomTokenLayout.num_tokens")
        atoms_per_token = atom_to_token.atoms_per_token
        num_tokens = atom_to_token.num_tokens
        atom_to_token = atom_to_token.atom_to_token
    if atom_to_token.dtype != torch.long:
        raise ValueError("atom_to_token must be a long tensor")
    if atom_to_token.dim() < 1:
        raise ValueError("atom_to_token must have shape (N_atom,) or (..., N_atom)")
    if num_tokens is None:
        raise ValueError("num_tokens is required to check a bare atom_to_token index")
    tokens = int(num_tokens)
    if tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if atom_to_token.numel() == 0:
        return

    stats = _atom_to_token_clause_stats(atom_to_token, atoms_per_token)
    low_atom, low_token, high_atom, high_token, irregular_atom, irregular_token = (
        int(value) for value in stats.tolist()
    )

    if low_token < 0:
        raise ValueError(f"every atom_to_token entry must lie in [0, {tokens}), got [{low_atom}] = {low_token}")
    if high_token >= tokens:
        raise ValueError(f"every atom_to_token entry must lie in [0, {tokens}), got [{high_atom}] = {high_token}")
    if atoms_per_token is not None and irregular_atom >= 0:
        raise ValueError(
            f"atoms_per_token={atoms_per_token} means atom_to_token[i] == i // {atoms_per_token}, got "
            f"[{irregular_atom}] = {irregular_token} where {irregular_atom // atoms_per_token} was expected"
        )


def _atom_to_token_clause_stats(atom_to_token: Tensor, atoms_per_token: int | None) -> Tensor:

    flat = atom_to_token.reshape(-1)
    device = flat.device
    low_atom = flat.argmin().reshape(1)
    high_atom = flat.argmax().reshape(1)

    if atoms_per_token is None:
        # -1 denotes no irregular atom; zero is a valid atom index.
        irregular_atom = torch.full((1,), -1, device=device, dtype=torch.long)
        irregular_token = torch.zeros(1, device=device, dtype=torch.long)
    else:
        expected = torch.arange(flat.numel(), device=device, dtype=flat.dtype) // atoms_per_token
        irregular = flat != expected
        first = irregular.long().argmax().reshape(1)
        irregular_atom = torch.where(irregular.any().reshape(1), first, torch.full_like(first, -1))
        irregular_token = flat.index_select(0, first)

    return torch.cat(
        (
            low_atom,
            flat.index_select(0, low_atom),
            high_atom,
            flat.index_select(0, high_atom),
            irregular_atom,
            irregular_token,
        )
    )


def _num_tokens(atom_to_token: Tensor, num_tokens: int | None) -> int:
    if num_tokens is not None:
        return int(num_tokens)
    if atom_to_token.numel() == 0:
        return 0
    return int(atom_to_token.max().item()) + 1


def _regular_atoms_per_token(atom_to_token: Tensor, num_tokens: int) -> int | None:
    atoms = atom_to_token.numel()
    if num_tokens <= 0 or atoms % num_tokens:
        return None
    atoms_per_token = atoms // num_tokens
    if atoms_per_token <= 0:
        return None
    expected = torch.arange(atoms, device=atom_to_token.device, dtype=atom_to_token.dtype) // atoms_per_token
    return atoms_per_token if torch.equal(atom_to_token, expected) else None


def _is_non_decreasing(atom_to_token: Tensor) -> bool:

    if atom_to_token.numel() < 2:
        return True
    return bool(torch.all(atom_to_token[1:] >= atom_to_token[:-1]).item())


def _sorted_atom_token_layout(
    atom_to_token: Tensor,
    num_tokens: int,
    *,
    sort: bool,
) -> AtomTokenSortedLayout | None:

    if atom_to_token.numel() == 0 or num_tokens <= 0:
        return None
    graph = CSRGraph.from_edges(atom_to_token, num_rows=num_tokens, sort=sort)
    token_offsets = graph.row_offsets
    counts = token_offsets[1:] - token_offsets[:-1]
    covered, max_atoms_per_token = (int(value) for value in torch.stack((token_offsets[-1], counts.max())).tolist())
    if covered != int(atom_to_token.numel()) or max_atoms_per_token <= 0:
        return None
    sorted_atom_to_token = graph.segment_ids
    assert sorted_atom_to_token is not None
    permutation = graph.src_index
    return AtomTokenSortedLayout(
        sorted_atom_to_token=sorted_atom_to_token,
        token_offsets=token_offsets.to(dtype=torch.long),
        max_atoms_per_token=max_atoms_per_token,
        permutation=permutation,
        inverse_permutation=None if permutation is None else torch.argsort(permutation, stable=True),
    )


def _check_sorted_layout_permutation(tensor: Tensor, name: str, atoms: int) -> None:
    if tensor.dtype != torch.long or tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1-D long tensor")
    if int(tensor.numel()) != atoms:
        raise ValueError(f"{name} must have one entry per atom")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_packed_offsets(operator: str, offsets: Tensor, total: int, *, operand: str) -> None:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError(f"{operator} {operand} offsets must be a one-dimensional CPU int64 tensor")
    if offsets.numel() == 0:
        raise ValueError(f"{operator} {operand} offsets must include the initial zero")
    if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ValueError(f"{operator} {operand} offsets must be monotonic and start at zero")
    if int(offsets[-1]) != total:
        raise ValueError(
            f"{operator} {operand} offsets cover {int(offsets[-1])} packed values, "
            f"but the packed tensor contains {total}"
        )
