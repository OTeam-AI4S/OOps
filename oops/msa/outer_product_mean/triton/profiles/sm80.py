# OOps
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OOps.

# OOps is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OOps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://oops.valtech.ai/about/license-faq/>.

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
