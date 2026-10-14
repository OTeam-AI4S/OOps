from oops.core import LaunchProfile

from ....batch_finalize import BatchFinalizeConfig, BatchFinalizeSchedule
from ..schedule import (
    CrossEntropyConfig,
    CrossEntropyProblem,
    CrossEntropySchedule,
    DistogramCrossEntropyConfig,
    DistogramCrossEntropyProblem,
    DistogramCrossEntropySchedule,
)

CROSS_ENTROPY = CrossEntropySchedule(CrossEntropyConfig())
DISTOGRAM_FINALIZE = BatchFinalizeSchedule(BatchFinalizeConfig())
DISTOGRAM_CROSS_ENTROPY = DistogramCrossEntropySchedule(
    DistogramCrossEntropyConfig(max_block_elements=8192),
    DISTOGRAM_FINALIZE,
)


CROSS_ENTROPY_FORWARD_FALLBACK = LaunchProfile[CrossEntropyProblem, CrossEntropySchedule](CROSS_ENTROPY)
DISTOGRAM_FORWARD_FALLBACK = LaunchProfile[DistogramCrossEntropyProblem, DistogramCrossEntropySchedule](
    DISTOGRAM_CROSS_ENTROPY
)
