import torch

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import MaskedKnnConfig, MaskedKnnProblem, MaskedKnnRule

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))

_MASKED_KNN_FORWARD = LaunchProfileDelta[MaskedKnnProblem, MaskedKnnConfig](
    rules=(
        MaskedKnnRule(
            config=MaskedKnnConfig(block_key=512, num_warps=4),
            min_topk=128,
            max_topk=128,
            dtypes=(torch.float32, torch.bfloat16),
            min_points_per_sample=512,
            max_total_points=8192,
        ),
    )
)

MASKED_KNN_TARGETS = (TargetProfile(_SM80, _MASKED_KNN_FORWARD),)
MASKED_KNN_BACKWARD_TARGETS = ()
