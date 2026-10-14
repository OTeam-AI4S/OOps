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

from dataclasses import dataclass
from functools import lru_cache

from oops.core import validate_positive_parameters
from oops.runtime import CUDA_MAX_GRID_X


@dataclass(frozen=True, slots=True)
class SoftmaxAdaptiveConfig:
    long_row_edges: int
    block_features: int
    block_edges: int

    def __post_init__(self) -> None:
        validate_positive_parameters(long_row_edges=self.long_row_edges)
        for name in ("block_features", "block_edges"):
            value = getattr(self, name)
            if value <= 0 or value & (value - 1):
                raise ValueError(f"softmax {name} must be a positive power of two")


@lru_cache(maxsize=1024)
def adaptive_geometry(
    config: SoftmaxAdaptiveConfig,
    num_edges: int,
    segments: int,
    features: int,
    block_features: int,
) -> tuple[int, int] | None:

    if num_edges < config.long_row_edges or features <= 1:
        return None
    long_features = min(config.block_features, block_features)
    programs = segments + (num_edges // config.long_row_edges) * (block_features // long_features)
    return (programs, long_features) if programs <= CUDA_MAX_GRID_X else None
