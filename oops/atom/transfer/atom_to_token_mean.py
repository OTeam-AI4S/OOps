from __future__ import annotations

from danling.tensors import NestedTensor
from torch import Tensor

from ..layout import AtomTokenLayout
from ._reduction import atom_to_token_reduce


def atom_to_token_mean(
    atom: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor | AtomTokenLayout,
    *,
    num_tokens: int | None = None,
    token_reference: NestedTensor | None = None,
    atom_mask: Tensor | NestedTensor | None = None,
    denominator_offset: float = 0.0,
) -> Tensor | NestedTensor:
    r"""Compute a guarded weighted mean of atom features for each token.

    Args:
        atom: Features shaped ``(..., A, C)``, or a NestedTensor with
            ``(A_i, C)`` or ``(S, A_i, C)`` elements.
        atom_to_token: Dense assignment weights shaped ``(..., A, N)``,
            local-long indices, or an ``AtomTokenLayout``. Dense indices may be
            globally shared ``(A,)``, match all feature batch axes, or omit only
            the sample axis in ``(*prefix, S, A, C)``. Nested mappings contain
            local ``(A_i,)`` indices.
        num_tokens: Output token count, required for bare dense indices.
            A layout carries this count; nested calls use ``token_reference``.
        token_reference: NestedTensor defining output token lengths. Sampled
            atoms accept reference elements ``(N_i, C_ref)``,
            ``(1, N_i, C_ref)``, or ``(S, N_i, C_ref)``.
        atom_mask: Optional non-complex atom weights shaped ``(..., A)``.
            Nested sampled masks may contain ``(A_i,)``, ``(1, A_i)``, or
            ``(S, A_i)`` elements. Floating weights are differentiable.
        denominator_offset: Offset added to positive weighted counts. The
            default gives AF3/Protenix semantics; ``1e-6`` gives Boltz semantics.

    Returns:
        Token features in the atom dtype: ``(..., N, C)`` for dense inputs,
        or nested ``(N_i, C)`` / ``(S, N_i, C)`` elements. The mean is
        ``sum / where(count > 0, count + denominator_offset, 1)``;
        ``atom_mask`` weights both sum and count. Empty tokens return zero.

    Raises:
        ValueError: Operand shapes, devices, mapping lengths, or nested
            topology disagree, or a checked eager mapping has invalid ids.
        OOpsUnsupportedError: An index-based call has an unsupported device,
            dtype, layout, or launch extent, or a forward-mode tangent is supplied
            on ``atom`` or ``atom_mask``.

    Notes:
        Dense assignments follow PyTorch's device and dtype rules and preserve
        the atom dtype under autocast. CUDA index layouts serve float16,
        bfloat16, and float32 features. Half reductions and VJP arithmetic use
        FP32 before casts to source dtypes. Reverse mode and full-graph
        compilation are supported. Forward-mode tangents on ``atom`` or
        ``atom_mask`` are unsupported.

        Bare mappings use live values on every call, including CUDA Graph replay.
        Explicit layouts must remain consistent with their tensors. Eager shared
        indices are range-checked; compiled, Graph, per-batch, and nested calls
        do not validate index contents. Out-of-range ids are outside the
        numerical contract. Validate untrusted metadata with
        ``validate_atom_to_token``.
        Sampled ``S`` must be positive and is static under compilation;
        atom/token lengths may vary.

    Examples:
        >>> import torch
        >>> from oops import atom_to_token_mean
        >>> atoms = torch.tensor([[2.0], [4.0], [9.0]])
        >>> assignment = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> atom_to_token_mean(atoms, assignment).tolist()
        [[3.0], [9.0]]
    """

    return atom_to_token_reduce(
        atom,
        atom_to_token,
        num_tokens=num_tokens,
        token_reference=token_reference,
        atom_mask=atom_mask,
        normalize=True,
        denominator_offset=denominator_offset,
    )
