from __future__ import annotations

from ..schedule import PackedMsaAnySchedule, PackedPairMaskSchedule
from .generic import MSA_ANY_FALLBACK, PAIR_MASK_FALLBACK

PAIR_MASK_SCHEDULE = PackedPairMaskSchedule(PAIR_MASK_FALLBACK)
MSA_ANY_SCHEDULE = PackedMsaAnySchedule(MSA_ANY_FALLBACK)

__all__ = ["MSA_ANY_SCHEDULE", "PAIR_MASK_SCHEDULE"]
