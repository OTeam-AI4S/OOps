import torch

from oops.core import LaunchProfile
from oops.runtime import HALF_DTYPES

from ..schedule import (
    GateMultiplyConfig,
    GateMultiplyProblem,
    GateNativeLinearRule,
    GateProjectionConfig,
    GateProjectionProblem,
)

PROJECTION_FALLBACK = LaunchProfile[GateProjectionProblem, GateProjectionConfig](GateProjectionConfig(64, 32, 64, 4, 3))
NATIVE_LINEAR = (
    GateNativeLinearRule(HALF_DTYPES, min_intensity=150, min_multiply_accumulates=1 << 30),
    GateNativeLinearRule((torch.float32,), ("tf32",), min_intensity=150, min_multiply_accumulates=1 << 28),
)
MULTIPLY_FORWARD_FALLBACK = LaunchProfile[GateMultiplyProblem, GateMultiplyConfig](GateMultiplyConfig(4, 64, 4))
MULTIPLY_BACKWARD_FALLBACK = LaunchProfile[GateMultiplyProblem, GateMultiplyConfig](GateMultiplyConfig(4, 64, 4))
