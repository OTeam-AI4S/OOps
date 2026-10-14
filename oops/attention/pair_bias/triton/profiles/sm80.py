from dataclasses import replace

import torch

from oops.attention.flash.triton import (
    AttentionLaunchContext,
    AttentionLaunchRule,
    AttentionStagePolicy,
    AttentionStageRule,
    AttentionTileConfig,
)
from oops.attention.flash.triton.schedule import SegmentedAttentionRule
from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector
from oops.runtime import HALF_DTYPES

from . import generic
from .generic import FORWARD_FALLBACK, KEY_VALUE_TF32X3_FALLBACK, QUERY_FALLBACK, QUERY_TF32X3_FALLBACK

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
_FLOAT32_STAGES = AttentionStagePolicy(rules=(AttentionStageRule(64, 2),), fallback=1)


def _delta(*rules: AttentionLaunchRule) -> tuple[TargetProfile, ...]:
    return (TargetProfile(_SM80, LaunchProfileDelta[AttentionLaunchContext, AttentionTileConfig](rules=rules)),)


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
SEGMENTED_FORWARD_TARGETS = ()
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
SEGMENTED_KEY_VALUE_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta(
            rules=(
                SegmentedAttentionRule(
                    replace(
                        generic.SEGMENTED_KEY_VALUE_FALLBACK.default,
                        tile=replace(generic.SEGMENTED_KEY_VALUE_FALLBACK.default.tile, block_query=16),
                    ),
                    HALF_DTYPES,
                ),
            )
        ),
    ),
)
SEGMENTED_KEY_VALUE_TF32X3_TARGETS = ()
SEGMENTED_PREPROCESS_TARGETS = ()
SEGMENTED_QUERY_TARGETS = ()
