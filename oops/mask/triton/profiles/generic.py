from __future__ import annotations

from oops.core import LaunchProfile

from ..schedule import (
    PackedMsaAnyConfig,
    PackedMsaAnyProblem,
    PackedPairMaskConfig,
    PackedPairMaskProblem,
)

PAIR_MASK_FALLBACK = LaunchProfile[PackedPairMaskProblem, PackedPairMaskConfig](
    PackedPairMaskConfig(),
)
MSA_ANY_FALLBACK = LaunchProfile[PackedMsaAnyProblem, PackedMsaAnyConfig](
    PackedMsaAnyConfig(),
)
