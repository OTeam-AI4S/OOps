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

from ._edition import _warn_internal_users
from ._errors import OOpsUnsupportedError
from ._version import __version__
from .attention import gate_attention_output, gated_pair_bias_attention, pair_bias_attention, triangle_attention
from .msa import categorical_msa_projection
from .normalization import LayerNorm, layer_norm
from .pair import triangle_multiplication

__all__ = [
    "categorical_msa_projection",
    "__version__",
    "OOpsUnsupportedError",
    "LayerNorm",
    "layer_norm",
    "triangle_multiplication",
    "gate_attention_output",
    "pair_bias_attention",
    "gated_pair_bias_attention",
    "triangle_attention",
]

_warn_internal_users()
