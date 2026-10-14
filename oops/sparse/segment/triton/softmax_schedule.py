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
