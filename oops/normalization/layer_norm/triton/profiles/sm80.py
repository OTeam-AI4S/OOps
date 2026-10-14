from dataclasses import replace

from oops.core import LaunchProfileDelta, TargetProfile, TargetSelector

from ..schedule import PORTABLE_POLICY, LayerNormLaunchPolicy

_POLICY = replace(
    PORTABLE_POLICY,
    backward_warp_columns=256,
    backward_max_num_warps=8,
)
_PROFILE: LaunchProfileDelta[int, LayerNormLaunchPolicy] = LaunchProfileDelta(default=_POLICY)
BACKWARD_TARGETS = (TargetProfile(TargetSelector(backend="cuda", architectures=("sm80",)), _PROFILE),)
