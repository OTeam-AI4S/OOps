# OOps
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OOps.

# OOps is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OOps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://oops.valtech.ai/about/license-faq/>.

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
