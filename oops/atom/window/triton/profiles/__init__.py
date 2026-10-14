from ..schedule import NestedTokenPairWindowSchedule
from . import generic as _generic
from . import sm80 as _sm80
from .generic import TOKEN_PAIR_WINDOW

NESTED_TOKEN_PAIR_WINDOW_FORWARD = NestedTokenPairWindowSchedule(_generic.NESTED_TOKEN_PAIR_WINDOW_FORWARD_FALLBACK)
NESTED_TOKEN_PAIR_WINDOW_BACKWARD = NestedTokenPairWindowSchedule(
    _generic.NESTED_TOKEN_PAIR_WINDOW_BACKWARD_FALLBACK,
    _sm80.NESTED_TOKEN_PAIR_WINDOW_BACKWARD_TARGETS,
)

__all__ = [
    "NESTED_TOKEN_PAIR_WINDOW_BACKWARD",
    "NESTED_TOKEN_PAIR_WINDOW_FORWARD",
    "TOKEN_PAIR_WINDOW",
]
