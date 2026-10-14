from oops.attention.flash.triton import AttentionPhaseProfile, AttentionScheduleCatalog
from oops.attention.flash.triton.schedule import SegmentedAttentionPhaseProfile, SegmentedAttentionScheduleCatalog

from . import generic as _generic
from . import sm80 as _sm80

SCHEDULES = AttentionScheduleCatalog(
    forward=AttentionPhaseProfile(_generic.FORWARD_FALLBACK, _sm80.FORWARD_TARGETS),
    backward_preprocess=AttentionPhaseProfile(_generic.PREPROCESS_FALLBACK, _sm80.PREPROCESS_TARGETS),
    backward_query=AttentionPhaseProfile(_generic.QUERY_FALLBACK, _sm80.QUERY_TARGETS),
    backward_query_tf32x3=AttentionPhaseProfile(_generic.QUERY_TF32X3_FALLBACK, _sm80.QUERY_TF32X3_TARGETS),
    backward_key_value=AttentionPhaseProfile(_generic.KEY_VALUE_FALLBACK, _sm80.KEY_VALUE_TARGETS),
    backward_key_value_tf32x3=AttentionPhaseProfile(
        _generic.KEY_VALUE_TF32X3_FALLBACK,
        _sm80.KEY_VALUE_TF32X3_TARGETS,
    ),
    grad_bias=AttentionPhaseProfile(_generic.GRAD_BIAS_FALLBACK, _sm80.GRAD_BIAS_TARGETS),
)

SEGMENTED_SCHEDULES = SegmentedAttentionScheduleCatalog(
    forward=SegmentedAttentionPhaseProfile(
        _generic.SEGMENTED_FORWARD_FALLBACK,
        _sm80.SEGMENTED_FORWARD_TARGETS,
    ),
    backward_preprocess=SegmentedAttentionPhaseProfile(
        _generic.SEGMENTED_PREPROCESS_FALLBACK,
        _sm80.SEGMENTED_PREPROCESS_TARGETS,
    ),
    backward_query=SegmentedAttentionPhaseProfile(
        _generic.SEGMENTED_QUERY_FALLBACK,
        _sm80.SEGMENTED_QUERY_TARGETS,
    ),
    backward_key_value=SegmentedAttentionPhaseProfile(
        _generic.SEGMENTED_KEY_VALUE_FALLBACK,
        _sm80.SEGMENTED_KEY_VALUE_TARGETS,
    ),
    backward_key_value_tf32x3=SegmentedAttentionPhaseProfile(
        _generic.SEGMENTED_KEY_VALUE_TF32X3_FALLBACK,
        _sm80.SEGMENTED_KEY_VALUE_TF32X3_TARGETS,
    ),
    grad_bias=SegmentedAttentionPhaseProfile(
        _generic.SEGMENTED_GRAD_BIAS_FALLBACK,
        _sm80.SEGMENTED_GRAD_BIAS_TARGETS,
    ),
)

__all__ = ["SCHEDULES", "SEGMENTED_SCHEDULES"]
