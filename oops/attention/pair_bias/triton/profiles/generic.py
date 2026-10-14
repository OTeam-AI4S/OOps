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

from dataclasses import replace

import torch

from oops.attention.flash.triton import (
    AttentionLaunchContext,
    AttentionLaunchRule,
    AttentionStagePolicy,
    AttentionStageRule,
    AttentionTileConfig,
)
from oops.attention.flash.triton.schedule import (
    SegmentedAttentionConfig,
    SegmentedAttentionProblem,
    SegmentedAttentionRule,
)
from oops.core import LaunchProfile

_STAGES_3_2_1 = AttentionStagePolicy(
    rules=(AttentionStageRule(64, 3), AttentionStageRule(128, 2)),
    fallback=1,
)
_STAGES_2_2_1 = AttentionStagePolicy(
    rules=(AttentionStageRule(64, 2), AttentionStageRule(128, 2)),
    fallback=1,
)

FORWARD_FALLBACK = LaunchProfile[AttentionLaunchContext, AttentionTileConfig](
    AttentionTileConfig(
        name="pair_bias_forward_generic",
        block_query=64,
        block_key=32,
        num_warps=4,
        stage_policy=_STAGES_3_2_1,
        query_budget_bytes=16384,
        key_budget_bytes=8192,
    )
)
PREPROCESS_FALLBACK = LaunchProfile[AttentionLaunchContext, AttentionTileConfig](
    AttentionTileConfig(
        name="pair_bias_backward_preprocess_generic",
        block_query=128,
        block_key=None,
        num_warps=4,
        stage_policy=_STAGES_3_2_1,
        query_budget_bytes=16384,
    )
)
QUERY_FALLBACK = LaunchProfile[AttentionLaunchContext, AttentionTileConfig](
    AttentionTileConfig(
        name="pair_bias_backward_query_generic",
        block_query=128,
        block_key=32,
        num_warps=4,
        stage_policy=_STAGES_3_2_1,
        query_budget_bytes=16384,
    )
)
QUERY_TF32X3_FALLBACK = LaunchProfile[AttentionLaunchContext, AttentionTileConfig](
    AttentionTileConfig(
        name="pair_bias_backward_query_tf32x3_generic",
        block_query=128,
        block_key=32,
        num_warps=4,
        stage_policy=_STAGES_3_2_1,
        query_budget_bytes=16384,
    )
)
KEY_VALUE_FALLBACK = LaunchProfile[AttentionLaunchContext, AttentionTileConfig](
    AttentionTileConfig(
        name="pair_bias_backward_key_value_generic",
        block_query=32,
        block_key=64,
        num_warps=2,
        stage_policy=_STAGES_3_2_1,
        query_budget_bytes=8192,
        key_budget_bytes=8192,
    )
)
KEY_VALUE_TF32X3_FALLBACK = LaunchProfile[AttentionLaunchContext, AttentionTileConfig](
    AttentionTileConfig(
        name="pair_bias_backward_key_value_tf32x3_generic",
        block_query=32,
        block_key=64,
        num_warps=2,
        stage_policy=_STAGES_3_2_1,
        query_budget_bytes=8192,
        key_budget_bytes=8192,
    )
)
_GRAD_BIAS = AttentionTileConfig(
    name="pair_bias_grad_bias_generic",
    block_query=32,
    block_key=64,
    num_warps=4,
    stage_policy=_STAGES_2_2_1,
    query_budget_bytes=16384,
)
_GRAD_BIAS_PARTS = replace(_GRAD_BIAS, name="pair_bias_grad_bias_parts", group_parts=8, group_workspace_bytes=16 << 20)
GRAD_BIAS_FALLBACK = LaunchProfile[AttentionLaunchContext, AttentionTileConfig](
    _GRAD_BIAS,
    rules=(AttentionLaunchRule(_GRAD_BIAS_PARTS, max_query_length=256, max_block_channel=32, element_sizes=(2,)),),
)

SEGMENTED_FORWARD_FALLBACK = LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig](
    SegmentedAttentionConfig(
        AttentionTileConfig(
            name="pair_bias_segmented_forward_generic",
            block_query=64,
            block_key=32,
            num_warps=4,
            stage_policy=_STAGES_3_2_1,
            query_budget_bytes=16384,
            key_budget_bytes=8192,
        ),
        1,
    )
)
SEGMENTED_PREPROCESS_FALLBACK = LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig](
    SegmentedAttentionConfig(
        AttentionTileConfig(
            name="pair_bias_segmented_backward_preprocess_generic",
            block_query=128,
            block_key=None,
            num_warps=4,
            stage_policy=_STAGES_3_2_1,
            query_budget_bytes=16384,
        ),
        1,
    )
)
SEGMENTED_QUERY_FALLBACK = LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig](
    SegmentedAttentionConfig(
        AttentionTileConfig(
            name="pair_bias_segmented_backward_query_generic",
            block_query=128,
            block_key=32,
            num_warps=4,
            stage_policy=_STAGES_3_2_1,
            query_budget_bytes=16384,
        ),
        1,
    )
)
SEGMENTED_KEY_VALUE_FALLBACK = LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig](
    SegmentedAttentionConfig(
        AttentionTileConfig(
            name="pair_bias_segmented_backward_key_value_generic",
            block_query=32,
            block_key=64,
            num_warps=2,
            stage_policy=_STAGES_3_2_1,
            query_budget_bytes=8192,
            key_budget_bytes=8192,
        ),
        1,
    )
)
SEGMENTED_KEY_VALUE_TF32X3_FALLBACK = LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig](
    SegmentedAttentionConfig(
        AttentionTileConfig(
            name="pair_bias_segmented_backward_key_value_tf32x3_generic",
            block_query=32,
            block_key=64,
            num_warps=2,
            stage_policy=_STAGES_3_2_1,
            query_budget_bytes=8192,
            key_budget_bytes=8192,
        ),
        1,
    )
)
SEGMENTED_GRAD_BIAS_FALLBACK = LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig](
    SegmentedAttentionConfig(
        AttentionTileConfig(
            name="pair_bias_segmented_grad_bias_generic",
            block_query=32,
            block_key=64,
            num_warps=4,
            stage_policy=_STAGES_2_2_1,
            query_budget_bytes=16384,
        ),
        1,
    )
)


def _resident_profile(profile, *, key_value=False, bias=False, half=False):

    def tile(config):
        updates = {"query_budget_bytes": 8192}
        if key_value:
            updates["num_warps"] = max(config.num_warps, 4)
        if bias:
            updates["key_budget_bytes"] = 8192
        return replace(config, **updates)

    default = profile.default
    if isinstance(default, SegmentedAttentionConfig):
        bounded = replace(default, tile=tile(default.tile))
        rules = tuple(replace(rule, config=replace(rule.config, tile=tile(rule.config.tile))) for rule in profile.rules)
        return replace(
            profile,
            rules=(
                *rules,
                SegmentedAttentionRule(
                    bounded, (torch.float32, torch.float16, torch.bfloat16) if half else (torch.float32,)
                ),
            ),
        )
    bounded = tile(default)
    rules = (
        profile.rules if bias and not half else tuple(replace(rule, config=tile(rule.config)) for rule in profile.rules)
    )
    return replace(profile, rules=(*rules, AttentionLaunchRule(bounded, element_sizes=(2, 4) if half else (4,))))


FORWARD_FALLBACK = _resident_profile(FORWARD_FALLBACK)
QUERY_FALLBACK = _resident_profile(QUERY_FALLBACK, half=True)
QUERY_TF32X3_FALLBACK = _resident_profile(QUERY_TF32X3_FALLBACK)
KEY_VALUE_FALLBACK = _resident_profile(KEY_VALUE_FALLBACK, key_value=True, half=True)
KEY_VALUE_TF32X3_FALLBACK = _resident_profile(KEY_VALUE_TF32X3_FALLBACK, key_value=True)
GRAD_BIAS_FALLBACK = _resident_profile(GRAD_BIAS_FALLBACK, bias=True)
SEGMENTED_FORWARD_FALLBACK = _resident_profile(SEGMENTED_FORWARD_FALLBACK)
SEGMENTED_QUERY_FALLBACK = _resident_profile(SEGMENTED_QUERY_FALLBACK, half=True)
SEGMENTED_KEY_VALUE_FALLBACK = _resident_profile(SEGMENTED_KEY_VALUE_FALLBACK, key_value=True, half=True)
SEGMENTED_KEY_VALUE_TF32X3_FALLBACK = _resident_profile(SEGMENTED_KEY_VALUE_TF32X3_FALLBACK, key_value=True)
SEGMENTED_GRAD_BIAS_FALLBACK = _resident_profile(SEGMENTED_GRAD_BIAS_FALLBACK, bias=True)
