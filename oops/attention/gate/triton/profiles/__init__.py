from ..schedule import GateAttentionOutputScheduleCatalog, GatePhaseProfile
from . import generic as _generic
from . import sm80 as _sm80

SCHEDULES = GateAttentionOutputScheduleCatalog(
    projection=GatePhaseProfile(_generic.PROJECTION_FALLBACK, _sm80.PROJECTION_TARGETS),
    native_linear=_generic.NATIVE_LINEAR,
    multiply_forward=GatePhaseProfile(_generic.MULTIPLY_FORWARD_FALLBACK, _sm80.MULTIPLY_FORWARD_TARGETS),
    multiply_backward=GatePhaseProfile(_generic.MULTIPLY_BACKWARD_FALLBACK, _sm80.MULTIPLY_BACKWARD_TARGETS),
)

__all__ = ["SCHEDULES"]
