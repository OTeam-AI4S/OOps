from __future__ import annotations

from oops.core import LaunchProfile

from ..schedule import PORTABLE_POLICY, LayerNormLaunchPolicy

FORWARD_FALLBACK: LaunchProfile[int, LayerNormLaunchPolicy] = LaunchProfile(PORTABLE_POLICY)
BACKWARD_FALLBACK: LaunchProfile[int, LayerNormLaunchPolicy] = LaunchProfile(PORTABLE_POLICY)
