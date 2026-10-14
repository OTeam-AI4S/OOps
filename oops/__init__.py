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

from .sparse import CSRGraph, regular_csr_graph, validate_csr_metadata

from .atom import AtomTokenLayout, validate_atom_to_token

from .atom import token_pair_to_atom_window_gather

from .atom import token_to_atom_broadcast

from .atom import atom_to_token_mean

from .atom import atom_to_token_sum

from .sparse import segment_sum_csr

from .sparse import segment_max_csr

from .sparse import segment_softmax_csr

from .sparse import edge_dot_node

from .sparse import segment_weighted_sum_csr

from .losses import cross_entropy

from .losses import distogram_cross_entropy

from .losses import smooth_lddt_loss_dense

from .losses import smooth_lddt_pair_indices

from .losses import smooth_lddt_pair_distances

__all__ = ['categorical_msa_projection', '__version__', 'OOpsUnsupportedError', 'LayerNorm', 'layer_norm', 'triangle_multiplication', 'gate_attention_output', 'pair_bias_attention', 'gated_pair_bias_attention', 'triangle_attention', 'CSRGraph', 'regular_csr_graph', 'validate_csr_metadata', 'AtomTokenLayout', 'validate_atom_to_token', 'token_pair_to_atom_window_gather', 'token_to_atom_broadcast', 'atom_to_token_mean', 'atom_to_token_sum', 'segment_sum_csr', 'segment_max_csr', 'segment_softmax_csr', 'edge_dot_node', 'segment_weighted_sum_csr', 'cross_entropy', 'distogram_cross_entropy', 'smooth_lddt_loss_dense', 'smooth_lddt_pair_indices', 'smooth_lddt_pair_distances']
