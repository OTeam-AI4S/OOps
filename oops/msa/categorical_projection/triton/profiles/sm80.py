import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import (
    CategoricalProjectionConfig,
    CategoricalProjectionProblem,
    CategoricalProjectionRule,
    CategoricalProjectionSchedule,
)

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[CategoricalProjectionProblem, CategoricalProjectionSchedule](
            rules=(
                CategoricalProjectionRule(
                    CategoricalProjectionSchedule(
                        CategoricalProjectionConfig(block_rows=64, block_channels=32, num_warps=4)
                    ),
                    min_rows=1 << 16,
                    max_channels=128,
                    dtypes=(torch.float32, torch.bfloat16),
                ),
                CategoricalProjectionRule(
                    CategoricalProjectionSchedule(
                        CategoricalProjectionConfig(block_rows=16, block_channels=64, num_warps=4)
                    ),
                    max_rows=1024,
                    max_channels=64,
                    dtypes=(torch.bfloat16,),
                ),
            )
        ),
    ),
)
