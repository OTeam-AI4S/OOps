import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import (
    SmoothLddtDenseConfig,
    SmoothLddtDenseProblem,
    SmoothLddtDenseRule,
    SmoothLddtDenseSchedule,
    SmoothLddtMaskConfig,
    SmoothLddtMaskProblem,
    SmoothLddtMaskRule,
    SmoothLddtMaskSchedule,
)
from .generic import SMOOTH_LDDT_DENSE_FINALIZE

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))
DENSE_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[SmoothLddtDenseProblem, SmoothLddtDenseSchedule](
            rules=(
                SmoothLddtDenseRule(
                    SmoothLddtDenseSchedule(
                        SmoothLddtDenseConfig(block_pairs=256, row_block_atoms=256, num_warps=8),
                        SMOOTH_LDDT_DENSE_FINALIZE,
                    ),
                    min_atoms=512,
                    dtypes=(torch.float32, torch.bfloat16),
                ),
            )
        ),
    ),
)
MASK_FORWARD_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[SmoothLddtMaskProblem, SmoothLddtMaskSchedule](
            rules=(
                SmoothLddtMaskRule(
                    SmoothLddtMaskSchedule(SmoothLddtMaskConfig(block_pairs=512, num_warps=8)),
                    min_atoms=1024,
                    dtypes=(torch.float32, torch.bfloat16),
                ),
            )
        ),
    ),
)
