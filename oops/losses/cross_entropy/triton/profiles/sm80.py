import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import (
    CrossEntropyConfig,
    CrossEntropyProblem,
    CrossEntropyRule,
    CrossEntropySchedule,
    DistogramCrossEntropyConfig,
    DistogramCrossEntropyProblem,
    DistogramCrossEntropyRule,
    DistogramCrossEntropySchedule,
)
from .generic import DISTOGRAM_FINALIZE

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
CROSS_ENTROPY_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[CrossEntropyProblem, CrossEntropySchedule](
            rules=(
                CrossEntropyRule(
                    CrossEntropySchedule(CrossEntropyConfig(block_rows=64)),
                    min_rows=1 << 20,
                    min_classes=128,
                    max_classes=128,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
DISTOGRAM_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[DistogramCrossEntropyProblem, DistogramCrossEntropySchedule](
            rules=(
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=128, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=64,
                    max_classes=64,
                    dtypes=(torch.float32,),
                ),
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=128, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=128,
                    max_classes=128,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
DISTOGRAM_BACKWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[DistogramCrossEntropyProblem, DistogramCrossEntropySchedule](
            rules=(
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=32, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=64,
                    max_classes=64,
                    dtypes=(torch.float32,),
                ),
                DistogramCrossEntropyRule(
                    DistogramCrossEntropySchedule(
                        DistogramCrossEntropyConfig(block_rows=32, accumulate_num_warps=4), DISTOGRAM_FINALIZE
                    ),
                    min_tokens=256,
                    min_classes=128,
                    max_classes=128,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
