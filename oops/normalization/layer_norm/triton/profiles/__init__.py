from __future__ import annotations

from ..schedule import LayerNormPhaseProfile, LayerNormScheduleCatalog
from .generic import BACKWARD_FALLBACK, FORWARD_FALLBACK
from .sm80 import BACKWARD_TARGETS

SCHEDULES = LayerNormScheduleCatalog(
    forward=LayerNormPhaseProfile(FORWARD_FALLBACK),
    backward=LayerNormPhaseProfile(BACKWARD_FALLBACK, BACKWARD_TARGETS),
)

__all__ = [
    "BACKWARD_FALLBACK",
    "BACKWARD_TARGETS",
    "FORWARD_FALLBACK",
    "SCHEDULES",
]
