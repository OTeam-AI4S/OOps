from dataclasses import replace

from oops.core import (
    CoreLaunchContext,
    CoreLaunchRule,
    CoreTileConfig,
    LaunchProfileDelta,
    TargetProfile,
    TargetSelector,
)
from oops.runtime import HALF_DTYPES

from . import generic

_SM80 = TargetSelector(backend="cuda", architectures=("sm80",))


DUAL_PROJECTION_TARGETS = ()


_TAIL_HALF = replace(
    generic.TAIL_FALLBACK.default,
    name="outer_product_mean_tail_half",
    block_channels=128,
    block_reduction=64,
    num_stages=4,
)


TAIL_TARGETS = (
    TargetProfile(
        _SM80,
        LaunchProfileDelta[CoreLaunchContext, CoreTileConfig](
            rules=(CoreLaunchRule(config=_TAIL_HALF, dtypes=HALF_DTYPES),)
        ),
    ),
)
