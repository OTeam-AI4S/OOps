from oops.core import LaunchProfile

from ..schedule import MaskedKnnBackwardConfig, MaskedKnnConfig, MaskedKnnProblem, MaskedKnnRule

MASKED_KNN_FALLBACK = LaunchProfile[MaskedKnnProblem, MaskedKnnConfig](
    default=MaskedKnnConfig(block_key=256, num_warps=8),
    rules=(MaskedKnnRule(MaskedKnnConfig(block_key=512, num_warps=8), max_topk=64),),
)
MASKED_KNN_BACKWARD_FALLBACK = LaunchProfile[MaskedKnnProblem, MaskedKnnBackwardConfig](
    default=MaskedKnnBackwardConfig(atomic_block_edges=256, deterministic_target_block=128, num_warps=4),
)
