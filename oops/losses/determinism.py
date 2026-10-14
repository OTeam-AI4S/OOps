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
