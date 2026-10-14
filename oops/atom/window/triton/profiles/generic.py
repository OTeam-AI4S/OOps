from oops.core import LaunchProfile

from ..schedule import (
    NestedTokenPairWindowConfig,
    NestedTokenPairWindowProblem,
    TokenPairWindowConfig,
    TokenPairWindowSchedule,
)

NESTED_TOKEN_PAIR_WINDOW_FORWARD_FALLBACK = LaunchProfile[NestedTokenPairWindowProblem, NestedTokenPairWindowConfig](
    NestedTokenPairWindowConfig()
)
NESTED_TOKEN_PAIR_WINDOW_BACKWARD_FALLBACK = LaunchProfile[NestedTokenPairWindowProblem, NestedTokenPairWindowConfig](
    NestedTokenPairWindowConfig()
)
TOKEN_PAIR_WINDOW = TokenPairWindowSchedule(TokenPairWindowConfig())
