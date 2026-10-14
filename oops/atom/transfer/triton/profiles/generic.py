from oops.core import LaunchProfile

from ..schedule import (
    AtomToTokenReductionConfig,
    AtomToTokenReductionSchedule,
    BatchedAtomToTokenReductionConfig,
    BatchedAtomToTokenReductionProblem,
    NestedAtomToTokenReductionConfig,
    NestedAtomToTokenReductionSchedule,
)

ATOM_TO_TOKEN_REDUCTION = AtomToTokenReductionSchedule(AtomToTokenReductionConfig())
BATCHED_ATOM_TO_TOKEN_REDUCTION_FALLBACK = LaunchProfile[
    BatchedAtomToTokenReductionProblem, BatchedAtomToTokenReductionConfig
](BatchedAtomToTokenReductionConfig())
NESTED_ATOM_TO_TOKEN_REDUCTION = NestedAtomToTokenReductionSchedule(LaunchProfile(NestedAtomToTokenReductionConfig()))
