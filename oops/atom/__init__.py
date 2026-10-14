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

from .layout import AtomTokenLayout, validate_atom_to_token

from .transfer import atom_to_token_mean, token_to_atom_broadcast

from .window import token_pair_to_atom_window_gather


__all__ = ['AtomTokenLayout', 'atom_to_token_mean', 'token_pair_to_atom_window_gather', 'token_to_atom_broadcast', 'validate_atom_to_token']
