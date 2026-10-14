from __future__ import annotations

import warnings

import torch

from oops.runtime import unsupported

NONDETERMINISTIC_REDUCTION = (
    "does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this "
    "operation, or you can use the 'warn_only=True' option, if that's acceptable for your "
    "application. The selected fused reduction uses relaxed float32 atomics, so the order of "
    "conflicting additions -- and the last bits of the result -- may vary between runs on "
    "identical input; use the composed torch reduction when the value has to be reproducible."
)


def require_nondeterminism_allowed(operator: str) -> None:

    if not torch.are_deterministic_algorithms_enabled():
        return
    rejection = unsupported(operator, NONDETERMINISTIC_REDUCTION)
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(str(rejection), UserWarning, stacklevel=3)
        return
    raise rejection
