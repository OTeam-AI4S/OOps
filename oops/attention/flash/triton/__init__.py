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

from .contract import (
    AttentionAdmissionKey,
    FlashAttentionBackend,
    PreparedAttention,
    attention_admission_key,
)
from .generic import (
    attention_grids_launchable,
    prepare_attention,
    run_attention_backward,
    run_attention_forward,
)
from .ops import (
    dispatch_attention,
    fake_attention_backward,
    fake_attention_forward,
    make_attention_backward,
    make_attention_bias_backward,
    setup_attention_bias_context,
    setup_attention_context,
)
from .problem import (
    MAX_HEAD_SIZE,
    SUPPORTED_DTYPES,
    AttentionLaunchContext,
    AttentionProblem,
    AttentionTensorLayout,
    attention_bias_strides,
    attention_lse_strides,
    attention_shared_bias_strides,
    attention_tensor_strides,
    block_channel_size,
    expand_pair_bias,
    optional_tensor,
    pair_mask_strides,
    resolve_scale,
    supports_head_major_qkv,
    supports_pair_bias,
    supports_pair_mask,
)
from .schedule import (
    AttentionLaunchPlan,
    AttentionLaunchRule,
    AttentionPhase,
    AttentionPhaseProfile,
    AttentionScheduleCatalog,
    AttentionStagePolicy,
    AttentionStageRule,
    AttentionTileConfig,
    ResolvedAttentionTile,
    fit_block,
)

__all__ = [
    "AttentionAdmissionKey",
    "AttentionLaunchContext",
    "AttentionLaunchPlan",
    "AttentionLaunchRule",
    "AttentionPhase",
    "AttentionPhaseProfile",
    "AttentionProblem",
    "AttentionScheduleCatalog",
    "AttentionStagePolicy",
    "AttentionStageRule",
    "AttentionTensorLayout",
    "AttentionTileConfig",
    "FlashAttentionBackend",
    "MAX_HEAD_SIZE",
    "PreparedAttention",
    "ResolvedAttentionTile",
    "SUPPORTED_DTYPES",
    "attention_bias_strides",
    "attention_admission_key",
    "attention_grids_launchable",
    "attention_lse_strides",
    "attention_shared_bias_strides",
    "attention_tensor_strides",
    "block_channel_size",
    "dispatch_attention",
    "expand_pair_bias",
    "fake_attention_backward",
    "fake_attention_forward",
    "fit_block",
    "make_attention_backward",
    "make_attention_bias_backward",
    "optional_tensor",
    "pair_mask_strides",
    "prepare_attention",
    "resolve_scale",
    "run_attention_backward",
    "run_attention_forward",
    "setup_attention_bias_context",
    "setup_attention_context",
    "supports_head_major_qkv",
    "supports_pair_bias",
    "supports_pair_mask",
]
