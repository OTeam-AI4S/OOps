from dataclasses import replace

import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector
from oops.runtime import HALF_DTYPES

from ..schedule import (
    GateMultiplyConfig,
    GateMultiplyProblem,
    GateMultiplyRule,
    GateProjectionConfig,
    GateProjectionProblem,
    GateProjectionRule,
)
from . import generic

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

PROJECTION_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[GateProjectionProblem, GateProjectionConfig](
            rules=(
                GateProjectionRule(
                    replace(generic.PROJECTION_FALLBACK.default, block_gate=64), HALF_DTYPES, min_values=1 << 19
                ),
                GateProjectionRule(
                    replace(generic.PROJECTION_FALLBACK.default, block_gate=128, block_reduction=32),
                    (torch.float32,),
                    min_values=1 << 21,
                ),
            )
        ),
    ),
)
MULTIPLY_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[GateMultiplyProblem, GateMultiplyConfig](
            rules=(GateMultiplyRule(replace(generic.MULTIPLY_FORWARD_FALLBACK.default, block_rows=16), HALF_DTYPES),)
        ),
    ),
)
MULTIPLY_BACKWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[GateMultiplyProblem, GateMultiplyConfig](
            rules=(GateMultiplyRule(replace(generic.MULTIPLY_BACKWARD_FALLBACK.default, block_rows=16), HALF_DTYPES),)
        ),
    ),
)
