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
from .atom import (
    AtomTokenLayout,
    atom_to_token_mean,
    atom_to_token_sum,
    token_pair_to_atom_window_gather,
    token_to_atom_broadcast,
    validate_atom_to_token,
)
from .attention import gate_attention_output, gated_pair_bias_attention, pair_bias_attention, triangle_attention
from .equivariant import se3_basis, spherical_harmonics
from .geometry import masked_knn, smoothed_vector_norm
from .losses import (
    cross_entropy,
    distogram_cross_entropy,
    smooth_lddt_loss_dense,
    smooth_lddt_loss_sparse,
    smooth_lddt_pair_distances,
    smooth_lddt_pair_indices,
)
from .mask import pair_mask_from_msa_mask, pair_mask_from_token_mask, token_mask_from_msa_mask
from .msa import categorical_msa_projection
from .normalization import LayerNorm, layer_norm
from .pair import triangle_multiplication
from .reduction import weighted_mean
from .sparse import (
    CSRGraph,
    segment_weighted_sum_csr,
    edge_dot_node,
    regular_csr_graph,
    segment_max_csr,
    segment_softmax_csr,
    segment_sum_csr,
    validate_csr_metadata,
)

_warn_internal_users()

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
    "CSRGraph",
    "regular_csr_graph",
    "validate_csr_metadata",
    "AtomTokenLayout",
    "validate_atom_to_token",
    "token_pair_to_atom_window_gather",
    "token_to_atom_broadcast",
    "atom_to_token_mean",
    "atom_to_token_sum",
    "segment_sum_csr",
    "segment_max_csr",
    "segment_softmax_csr",
    "edge_dot_node",
    "segment_weighted_sum_csr",
    "cross_entropy",
    "distogram_cross_entropy",
    "smooth_lddt_loss_dense",
    "smooth_lddt_pair_indices",
    "smooth_lddt_pair_distances",
    "smooth_lddt_loss_sparse",
    "spherical_harmonics",
    "se3_basis",
    "pair_mask_from_token_mask",
    "token_mask_from_msa_mask",
    "pair_mask_from_msa_mask",
    "masked_knn",
    "smoothed_vector_norm",
    "weighted_mean",
]
