from dataclasses import replace

import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import NestedTokenPairWindowConfig, NestedTokenPairWindowProblem, NestedTokenPairWindowRule
from .generic import NESTED_TOKEN_PAIR_WINDOW_BACKWARD_FALLBACK

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

_NESTED_TOKEN_PAIR_WINDOW_BACKWARD = LaunchProfileDelta[NestedTokenPairWindowProblem, NestedTokenPairWindowConfig](
    rules=(
        NestedTokenPairWindowRule(
            config=replace(NESTED_TOKEN_PAIR_WINDOW_BACKWARD_FALLBACK.default, block_elements=512, num_warps=8),
            dtypes=(torch.bfloat16,),
            channels=64,
            n_queries=32,
            n_keys=128,
            has_num_valid_atoms=False,
        ),
    )
)

NESTED_TOKEN_PAIR_WINDOW_BACKWARD_TARGETS = (TargetProfile(_SM80, _NESTED_TOKEN_PAIR_WINDOW_BACKWARD),)
