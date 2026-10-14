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

from .generic import (
    supports_atom_to_token_reduce,
    supports_atom_to_token_reduce_layout,
    supports_batched_atom_to_token_reduce,
    supports_nested_atom_to_token_reduce,
    supports_shared_general_atom_to_token_reduce,
)
from .ops import (
    triton_atom_to_token_reduce,
    triton_atom_to_token_reduce_backward,
    triton_atom_to_token_reduce_mask_backward,
    triton_batched_atom_to_token_reduce,
    triton_nested_atom_to_token_reduce,
    triton_prepared_atom_to_token_reduce,
    triton_shared_general_atom_to_token_reduce,
    triton_shared_general_atom_to_token_reduce_backward,
    triton_shared_general_atom_to_token_reduce_mask_backward,
)

__all__ = [
    "supports_atom_to_token_reduce",
    "supports_atom_to_token_reduce_layout",
    "supports_batched_atom_to_token_reduce",
    "supports_nested_atom_to_token_reduce",
    "supports_shared_general_atom_to_token_reduce",
    "triton_atom_to_token_reduce",
    "triton_atom_to_token_reduce_backward",
    "triton_atom_to_token_reduce_mask_backward",
    "triton_batched_atom_to_token_reduce",
    "triton_nested_atom_to_token_reduce",
    "triton_prepared_atom_to_token_reduce",
    "triton_shared_general_atom_to_token_reduce",
    "triton_shared_general_atom_to_token_reduce_backward",
    "triton_shared_general_atom_to_token_reduce_mask_backward",
]
