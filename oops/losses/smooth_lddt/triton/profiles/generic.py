from oops.core import LaunchProfile

from ....batch_finalize import BatchFinalizeConfig, BatchFinalizeSchedule
from ..schedule import (
    SmoothLddtDenseConfig,
    SmoothLddtDenseProblem,
    SmoothLddtDenseSchedule,
    SmoothLddtMaskConfig,
    SmoothLddtMaskProblem,
    SmoothLddtMaskSchedule,
    SmoothLddtPairDistanceConfig,
    SmoothLddtPairDistanceSchedule,
    SmoothLddtSparseConfig,
    SmoothLddtSparseCoordsSchedule,
    SmoothLddtSparseSchedule,
)

SMOOTH_LDDT_DENSE_FINALIZE = BatchFinalizeSchedule(BatchFinalizeConfig())
SMOOTH_LDDT_DENSE = SmoothLddtDenseSchedule(SmoothLddtDenseConfig(), SMOOTH_LDDT_DENSE_FINALIZE)
SMOOTH_LDDT_MASK = SmoothLddtMaskSchedule(SmoothLddtMaskConfig())
SMOOTH_LDDT_PAIR_DISTANCE = SmoothLddtPairDistanceSchedule(SmoothLddtPairDistanceConfig())
SMOOTH_LDDT_SPARSE = SmoothLddtSparseSchedule(SmoothLddtSparseConfig())
SMOOTH_LDDT_SPARSE_COORDS = SmoothLddtSparseCoordsSchedule(
    SMOOTH_LDDT_PAIR_DISTANCE,
    SMOOTH_LDDT_SPARSE,
)


DENSE_FALLBACK = LaunchProfile[SmoothLddtDenseProblem, SmoothLddtDenseSchedule](SMOOTH_LDDT_DENSE)
MASK_FORWARD_FALLBACK = LaunchProfile[SmoothLddtMaskProblem, SmoothLddtMaskSchedule](SMOOTH_LDDT_MASK)
