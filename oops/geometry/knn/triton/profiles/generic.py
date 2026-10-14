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

from oops.core import LaunchProfile

from ..schedule import MaskedKnnBackwardConfig, MaskedKnnConfig, MaskedKnnProblem, MaskedKnnRule

MASKED_KNN_FALLBACK = LaunchProfile[MaskedKnnProblem, MaskedKnnConfig](
    default=MaskedKnnConfig(block_key=256, num_warps=8),
    rules=(MaskedKnnRule(MaskedKnnConfig(block_key=512, num_warps=8), max_topk=64),),
)
MASKED_KNN_BACKWARD_FALLBACK = LaunchProfile[MaskedKnnProblem, MaskedKnnBackwardConfig](
    default=MaskedKnnBackwardConfig(atomic_block_edges=256, deterministic_target_block=128, num_warps=4),
)
