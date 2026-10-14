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

from dataclasses import replace
from typing import cast

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
from oops.core import LaunchProfile, LaunchProfileDelta, TargetProfile, TargetSelector
from oops.runtime import HALF_DTYPES

from . import generic
from .generic import FORWARD_FALLBACK, KEY_VALUE_TF32X3_FALLBACK, QUERY_FALLBACK, QUERY_TF32X3_FALLBACK

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
_FLOAT32_STAGES = AttentionStagePolicy(rules=(AttentionStageRule(64, 2),), fallback=1)
_SEGMENTED_RESIDENT_WARPS = 8


def _delta(*rules: AttentionLaunchRule) -> tuple[TargetProfile, ...]:
    return (TargetProfile(_SM80, LaunchProfileDelta[AttentionLaunchContext, AttentionTileConfig](rules=rules)),)


def _segmented_resident_delta(
    profile: LaunchProfile[SegmentedAttentionProblem, SegmentedAttentionConfig],
    *rules: SegmentedAttentionRule,
) -> tuple[TargetProfile, ...]:
    def resident(config: SegmentedAttentionConfig) -> SegmentedAttentionConfig:
        return replace(
            config,
            programs_per_multiprocessor=max(1, _SEGMENTED_RESIDENT_WARPS // config.tile.num_warps),
        )

    return (
        TargetProfile(
            _SM80,
            LaunchProfileDelta(
                default=resident(profile.default),
                rules=tuple(
                    replace(cast(SegmentedAttentionRule, rule), config=resident(rule.config))
                    for rule in (*rules, *profile.rules)
                ),
            ),
        ),
    )


def _short_query(tile: AttentionTileConfig, name: str) -> AttentionLaunchRule:
    return AttentionLaunchRule(
        replace(tile, name=name, block_query=32, stage_policy=_FLOAT32_STAGES),
        max_query_length=32,
        max_block_channel=32,
        element_sizes=(4,),
    )


FORWARD_TARGETS = _delta(
    AttentionLaunchRule(
        replace(FORWARD_FALLBACK.default, name="pair_bias_forward_float32", key_budget_bytes=16384),
        min_block_channel=128,
        max_block_channel=128,
        element_sizes=(4,),
    )
)
QUERY_TARGETS = _delta(_short_query(QUERY_FALLBACK.default, "pair_bias_backward_query_short_float32"))
QUERY_TF32X3_TARGETS = _delta(
    _short_query(QUERY_TF32X3_FALLBACK.default, "pair_bias_backward_query_short_tf32x3"),
    AttentionLaunchRule(
        replace(
            QUERY_TF32X3_FALLBACK.default,
            name="pair_bias_backward_query_tf32x3",
            num_warps=8,
            stage_policy=_FLOAT32_STAGES,
        ),
        min_block_channel=32,
        max_block_channel=32,
        element_sizes=(4,),
    ),
)
KEY_VALUE_TARGETS = _delta(
    AttentionLaunchRule(
        replace(generic.KEY_VALUE_FALLBACK.default, name="pair_bias_backward_key_value_half_sm80", block_query=16),
        min_block_channel=64,
        max_block_channel=64,
        element_sizes=(2,),
    )
)
KEY_VALUE_TF32X3_TARGETS = _delta(
    AttentionLaunchRule(
        replace(
            KEY_VALUE_TF32X3_FALLBACK.default,
            name="pair_bias_backward_key_value_tf32x3",
            block_key=32,
            stage_policy=_FLOAT32_STAGES,
        ),
        min_block_channel=32,
        max_block_channel=32,
        element_sizes=(4,),
    )
)
GRAD_BIAS_TARGETS = ()
PREPROCESS_TARGETS = ()
SEGMENTED_FORWARD_TARGETS = _segmented_resident_delta(generic.SEGMENTED_FORWARD_FALLBACK)
SEGMENTED_GRAD_BIAS_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta(
            rules=(
                SegmentedAttentionRule(
                    replace(
                        generic.SEGMENTED_GRAD_BIAS_FALLBACK.default,
                        tile=replace(generic.SEGMENTED_GRAD_BIAS_FALLBACK.default.tile, block_query=16),
                    ),
                    (torch.float32,),
                ),
            )
        ),
    ),
)
SEGMENTED_KEY_VALUE_TARGETS = _segmented_resident_delta(
    generic.SEGMENTED_KEY_VALUE_FALLBACK,
    SegmentedAttentionRule(
        replace(
            generic.SEGMENTED_KEY_VALUE_FALLBACK.default,
            tile=replace(generic.SEGMENTED_KEY_VALUE_FALLBACK.default.tile, block_query=16),
        ),
        HALF_DTYPES,
    ),
)
SEGMENTED_KEY_VALUE_TF32X3_TARGETS = _segmented_resident_delta(generic.SEGMENTED_KEY_VALUE_TF32X3_FALLBACK)
SEGMENTED_PREPROCESS_TARGETS = ()
SEGMENTED_QUERY_TARGETS = _segmented_resident_delta(generic.SEGMENTED_QUERY_FALLBACK)
