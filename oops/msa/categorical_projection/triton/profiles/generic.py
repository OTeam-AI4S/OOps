from oops.core import LaunchProfile

from ..schedule import CategoricalProjectionConfig, CategoricalProjectionProblem, CategoricalProjectionSchedule

SCHEDULE = CategoricalProjectionSchedule(CategoricalProjectionConfig())


FALLBACK = LaunchProfile[CategoricalProjectionProblem, CategoricalProjectionSchedule](SCHEDULE)
