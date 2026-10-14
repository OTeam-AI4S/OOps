from ....csr.triton.schedule import CSR_SUM_SCHEDULE, SegmentConfig, SegmentSchedule
from ..softmax_schedule import SoftmaxAdaptiveConfig

SEGMENT_SUM_SCHEDULE = CSR_SUM_SCHEDULE
SEGMENT_MAX_SCHEDULE = SegmentSchedule(SegmentConfig())
SEGMENT_SOFTMAX_SCHEDULE = SegmentSchedule(SegmentConfig())
SEGMENT_SOFTMAX_ADAPTIVE = SoftmaxAdaptiveConfig(long_row_edges=8192, block_features=8, block_edges=256)
