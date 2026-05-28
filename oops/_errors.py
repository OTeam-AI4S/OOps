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

r"""
Errors raised by OOps operators.
"""

from __future__ import annotations


class OOpsUnsupportedError(NotImplementedError):
    r"""Signal that an operator cannot serve the requested execution contract.

    Operators use this ``NotImplementedError`` subclass for unsupported devices,
    dtypes, layouts, launch limits, differentiation modes, or precision states.
    Invalid argument relationships may instead raise ``ValueError`` or ``TypeError``.
    Modules with a documented native fallback may catch this exception.
    """
