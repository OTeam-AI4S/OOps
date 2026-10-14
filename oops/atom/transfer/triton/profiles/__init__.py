from ..schedule import BatchedAtomToTokenReductionSchedule
from . import generic as _generic
from .generic import ATOM_TO_TOKEN_REDUCTION, NESTED_ATOM_TO_TOKEN_REDUCTION

BATCHED_ATOM_TO_TOKEN_REDUCTION = BatchedAtomToTokenReductionSchedule(
    _generic.BATCHED_ATOM_TO_TOKEN_REDUCTION_FALLBACK,
)

__all__ = [
    "ATOM_TO_TOKEN_REDUCTION",
    "BATCHED_ATOM_TO_TOKEN_REDUCTION",
    "NESTED_ATOM_TO_TOKEN_REDUCTION",
]
