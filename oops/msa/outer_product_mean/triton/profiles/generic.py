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

import torch

from oops.core import CoreLaunchContext, CoreLaunchRule, CoreTileConfig, LaunchProfile

DUAL_PROJECTION_FALLBACK = LaunchProfile[CoreLaunchContext, CoreTileConfig](
    CoreTileConfig(
        name="outer_product_mean_dual_projection_generic",
        block_rows=16,
        block_reduction=64,
        num_warps=1,
        num_stages=1,
    ),
    rules=(
        CoreLaunchRule(
            CoreTileConfig(
                name="outer_product_mean_dual_projection_fp32_generic",
                block_rows=16,
                block_reduction=64,
                num_warps=2,
                num_stages=1,
            ),
            dtypes=(torch.float32,),
        ),
    ),
)


TAIL_FALLBACK = LaunchProfile[CoreLaunchContext, CoreTileConfig](
    CoreTileConfig(
        name="outer_product_mean_tail_generic",
        block_rows=64,
        block_channels=64,
        block_reduction=16,
        num_warps=4,
        num_stages=1,
    )
)
