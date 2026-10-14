from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import (
    FLOAT32_DOT_TF32X3,
    HALF_DTYPES,
    STANDARD_FLOAT_DTYPES,
    cuda_device_caps,
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
    matmul_precision,
)
from oops.runtime import needs_grad as tensors_need_grad
from oops.runtime import require_triton_dot_precision, tensors_share_device
from oops.runtime.cuda import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ

from ...flash import degenerate_shape, pair_mask_kind
from ...flash.triton import (
    AttentionAdmissionKey,
    AttentionLaunchContext,
    AttentionPhase,
    AttentionProblem,
    AttentionTensorLayout,
    FlashAttentionBackend,
    PreparedAttention,
    attention_admission_key,
    attention_bias_strides,
    attention_grids_launchable,
    attention_shared_bias_strides,
    attention_tensor_strides,
    dispatch_attention,
    expand_pair_bias,
    fake_attention_backward,
    fake_attention_forward,
    make_attention_backward,
    make_attention_bias_backward,
    optional_tensor,
    pair_mask_strides,
    prepare_attention,
    resolve_scale,
    run_attention_backward,
    run_attention_forward,
    setup_attention_bias_context,
    setup_attention_context,
    supports_head_major_qkv,
    supports_pair_bias,
    supports_pair_mask,
)
from ...flash.triton.ops import _resolve_attention_operand
from ...flash.triton.schedule import SegmentedAttentionPlan, SegmentedAttentionProblem
from . import kernel
from .profiles import SCHEDULES, SEGMENTED_SCHEDULES

_PAIR_BIAS_FLOAT32_DOT = FLOAT32_DOT_TF32X3
_PAIR_BIAS_EAGER_ADMISSION: tuple[AttentionAdmissionKey, PreparedAttention] | None = None
_PAIR_BIAS_EAGER_ADMISSIONS: dict[AttentionAdmissionKey, PreparedAttention] = {}


def _supports_pair_bias_contract(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
) -> bool:
    return (
        supports_head_major_qkv(query, key, value)
        and supports_pair_mask(query, key, mask)
        and supports_pair_bias(query, key, pair_bias)
    )


def supports_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
) -> bool:

    return _supports_pair_bias_contract(
        query,
        key,
        value,
        mask,
        pair_bias,
    ) and _pair_bias_attention_grids_launchable(query, key, value, mask, pair_bias)


def prepare_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
) -> PreparedAttention | None:

    if is_compiling():
        raise RuntimeError("pair-bias eager admission must not run while compiling")
    return _prepare_eager_pair_bias_attention(query, key, value, mask, pair_bias)


def _prepare_eager_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
) -> PreparedAttention | None:

    global _PAIR_BIAS_EAGER_ADMISSION

    precision = matmul_precision(query.dtype, _PAIR_BIAS_FLOAT32_DOT)
    key_metadata = attention_admission_key(
        (query, key, value, mask, pair_bias),
        precision=precision,
    )
    cached = _PAIR_BIAS_EAGER_ADMISSION
    if cached is not None and cached[0] == key_metadata:
        return cached[1]
    prepared = _PAIR_BIAS_EAGER_ADMISSIONS.get(key_metadata)
    if prepared is not None:
        _PAIR_BIAS_EAGER_ADMISSION = key_metadata, prepared
        return prepared
    if not _supports_pair_bias_contract(query, key, value, mask, pair_bias):
        return None
    kernel_bias = expand_pair_bias(query, key, pair_bias)
    prepared = prepare_attention(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        mask,
        kernel_bias,
        precision=precision,
        maximum_grid_x=CUDA_MAX_GRID_X,
        maximum_grid_yz=CUDA_MAX_GRID_YZ,
    )
    if prepared is not None:
        if len(_PAIR_BIAS_EAGER_ADMISSIONS) >= 32:
            _PAIR_BIAS_EAGER_ADMISSIONS.clear()
        _PAIR_BIAS_EAGER_ADMISSIONS[key_metadata] = prepared
        _PAIR_BIAS_EAGER_ADMISSION = key_metadata, prepared
    return prepared


def _pair_bias_attention_grids_launchable(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
) -> bool:
    if is_compiling():
        torch._dynamo.mark_static(query, -1)
    return attention_grids_launchable(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        mask,
        pair_bias,
        maximum_grid_x=CUDA_MAX_GRID_X,
        maximum_grid_yz=CUDA_MAX_GRID_YZ,
    )


def triton_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    pair_bias: Tensor | None = None,
    *,
    scale: float | None = None,
    prepared: PreparedAttention | None = None,
) -> Tensor:

    pair_bias = expand_pair_bias(query, key, pair_bias)
    operands = tuple(tensor for tensor in (query, key, value, mask, pair_bias) if tensor is not None)
    if (
        not is_compiling()
        and tensors_need_grad(query, key, value, pair_bias)
        and all(type(tensor) in (Tensor, torch.nn.Parameter) for tensor in operands)
        and not is_in_torch_dispatch_mode()
        and not torch.overrides.has_torch_function(operands)
        and not functorch_transforms_active()
    ):
        if mask is not None:
            previous_strides = mask.stride()
            mask = mask.to(dtype=torch.int16, memory_format=torch.contiguous_format)
            if prepared is not None and mask.stride() != previous_strides:
                prepared = replace(
                    prepared,
                    problem=_PAIR_BIAS_BACKEND.bind_problem(
                        query, key, value, mask, pair_bias, bind_launch_strides=True
                    ),
                )
        precision = (
            prepared.precision if prepared is not None else matmul_precision(query.dtype, _PAIR_BIAS_FLOAT32_DOT)
        )
        return _EagerPairBiasAttention.apply(query, key, value, mask, pair_bias, scale, precision, prepared)
    return dispatch_attention(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        mask,
        pair_bias,
        scale,
        _pair_bias_attention_inference,
        _pair_bias_attention_bias_forward,
        _pair_bias_attention_forward,
        prepared=prepared,
    )


class _EagerPairBiasAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask, pair_bias, scale, precision, prepared):
        with cuda_device_context(query.device):
            output, lse = run_attention_forward(
                _PAIR_BIAS_BACKEND,
                query,
                key,
                value,
                mask,
                pair_bias,
                scale=scale,
                precision=precision,
                prepared=prepared,
            )
        needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[4]
        ctx.scale = scale
        ctx.precision = precision
        ctx.prepared = prepared
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            query, key, value if needs_delta else key, mask, pair_bias, output if needs_delta else query, lse
        )
        return output.to(query.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, mask, pair_bias, output, lse = ctx.saved_tensors
        need_query, need_key, need_value = ctx.needs_input_grad[:3]
        need_bias = ctx.needs_input_grad[4]
        if torch.is_grad_enabled():
            if pair_bias is None:
                gradients = _pair_bias_attention_backward(
                    grad_output,
                    query,
                    key,
                    value,
                    mask,
                    output,
                    lse,
                    ctx.scale,
                    ctx.precision,
                    need_query,
                    need_key,
                    need_value,
                )
                grad_query, grad_key, grad_value = gradients
                grad_bias = None
            else:
                grad_query, grad_key, grad_value, grad_bias = _pair_bias_attention_bias_backward(
                    grad_output,
                    query,
                    key,
                    value,
                    mask,
                    pair_bias,
                    output,
                    lse,
                    ctx.scale,
                    ctx.precision,
                    need_query,
                    need_key,
                    need_value,
                    need_bias,
                )
        else:
            with cuda_device_context(query.device):
                grad_query, grad_key, grad_value, grad_bias = run_attention_backward(
                    _PAIR_BIAS_BACKEND,
                    grad_output,
                    query,
                    key,
                    value,
                    mask,
                    pair_bias,
                    output,
                    lse,
                    scale=ctx.scale,
                    precision=ctx.precision,
                    need_grad_query=need_query,
                    need_grad_key=need_key,
                    need_grad_value=need_value,
                    need_grad_bias=need_bias,
                    prepared=ctx.prepared,
                )
        return grad_query, grad_key, grad_value, None, grad_bias, None, None, None


@torch.library.custom_op("ooperators::_pair_bias_attention_bias_forward", mutates_args=(), device_types="cuda")
def _pair_bias_attention_bias_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    output, lse = run_attention_forward(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        mask,
        pair_bias,
        scale=scale,
        precision=precision,
    )
    assert lse is not None
    return output, lse


@torch.library.custom_op("ooperators::_pair_bias_attention_forward", mutates_args=(), device_types="cuda")
def _pair_bias_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    output, lse = run_attention_forward(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        mask,
        None,
        scale=scale,
        precision=precision,
    )
    assert lse is not None
    return output, lse


@torch.library.custom_op("ooperators::_pair_bias_attention_inference", mutates_args=(), device_types="cuda")
def _pair_bias_attention_inference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    scale: float | None,
    precision: str,
) -> Tensor:
    output, _ = run_attention_forward(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        mask,
        pair_bias,
        scale=scale,
        precision=precision,
        write_lse=False,
    )
    return output


# Gradient-demand flags cross the opaque ABI; unrequested Tensor? slots are None.
_PAIR_BIAS_BIAS_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor pair_bias,"
    " Tensor output, Tensor lse, float? scale, str precision, bool need_grad_query,"
    " bool need_grad_key, bool need_grad_value, bool need_grad_pair_bias)"
    " -> (Tensor?, Tensor?, Tensor?, Tensor?)"
)
_PAIR_BIAS_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor output,"
    " Tensor lse, float? scale, str precision, bool need_grad_query, bool need_grad_key,"
    " bool need_grad_value) -> (Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_pair_bias_attention_bias_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_PAIR_BIAS_BIAS_BACKWARD_SCHEMA,
)
def _pair_bias_attention_bias_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    output: Tensor,
    lse: Tensor,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_pair_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return run_attention_backward(
        _PAIR_BIAS_BACKEND,
        grad_output,
        query,
        key,
        value,
        mask,
        pair_bias,
        output,
        lse,
        scale=scale,
        precision=precision,
        need_grad_query=need_grad_query,
        need_grad_key=need_grad_key,
        need_grad_value=need_grad_value,
        need_grad_bias=need_grad_pair_bias,
    )


@torch.library.custom_op(
    "ooperators::_pair_bias_attention_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_PAIR_BIAS_BACKWARD_SCHEMA,
)
def _pair_bias_attention_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    output: Tensor,
    lse: Tensor,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    grad_query, grad_key, grad_value, _grad_pair_bias = run_attention_backward(
        _PAIR_BIAS_BACKEND,
        grad_output,
        query,
        key,
        value,
        mask,
        None,
        output,
        lse,
        scale=scale,
        precision=precision,
        need_grad_query=need_grad_query,
        need_grad_key=need_grad_key,
        need_grad_value=need_grad_value,
        need_grad_bias=False,
    )
    return grad_query, grad_key, grad_value


@torch.library.register_fake(_pair_bias_attention_bias_forward)
def _pair_bias_attention_bias_forward_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    return fake_attention_forward(_PAIR_BIAS_BACKEND, query, key, value, mask, pair_bias)


@torch.library.register_fake(_pair_bias_attention_forward)
def _pair_bias_attention_forward_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    return fake_attention_forward(_PAIR_BIAS_BACKEND, query, key, value, mask, None)


@torch.library.register_fake(_pair_bias_attention_inference)
def _pair_bias_attention_inference_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    scale: float | None,
    precision: str,
) -> Tensor:
    return torch.empty_like(query)


@torch.library.register_fake(_pair_bias_attention_bias_backward)
def _pair_bias_attention_bias_backward_fake(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    output: Tensor,
    lse: Tensor,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_pair_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return fake_attention_backward(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        pair_bias,
        need_grad_query,
        need_grad_key,
        need_grad_value,
        need_grad_pair_bias,
    )


@torch.library.register_fake(_pair_bias_attention_backward)
def _pair_bias_attention_backward_fake(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    output: Tensor,
    lse: Tensor,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    grad_query, grad_key, grad_value, _grad_bias = fake_attention_backward(
        _PAIR_BIAS_BACKEND,
        query,
        key,
        value,
        None,
        need_grad_query,
        need_grad_key,
        need_grad_value,
        False,
    )
    return grad_query, grad_key, grad_value


torch.library.register_autograd(
    _pair_bias_attention_bias_forward,
    make_attention_bias_backward(_pair_bias_attention_bias_backward),
    setup_context=setup_attention_bias_context,
)
torch.library.register_autograd(
    _pair_bias_attention_forward,
    make_attention_backward(_pair_bias_attention_backward),
    setup_context=setup_attention_context,
)


# The sample axis S is the shared kernel group axis, not its batch axis.


def _bind_pair_bias_problem(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    *,
    bind_launch_strides: bool,
) -> AttentionProblem:
    batch_size, num_samples, num_heads, query_length, head_size = degenerate_shape(query, 5)
    key_length = key.shape[-2]
    tensor_layout = AttentionTensorLayout.HEAD_TOKEN
    mask_kind = pair_mask_kind(query, mask)
    bias_groups = 1 if pair_bias is None else pair_bias.shape[-4]
    bias_strides = attention_bias_strides(pair_bias) if bind_launch_strides else (0, 0, 0, 0, 0)
    if bind_launch_strides and statically_known_true(bias_groups == 1):
        batch_stride, _group_stride, head_stride, query_stride, key_stride = bias_strides
        bias_strides = batch_stride, 0, head_stride, query_stride, key_stride
    return AttentionProblem(
        batch_size=batch_size,
        num_groups=num_samples,
        context=AttentionLaunchContext(
            num_heads=num_heads,
            query_length=query_length,
            key_length=key_length,
            head_size=head_size,
            mask_kind=mask_kind,
            element_size=query.element_size(),
        ),
        tensor_layout=tensor_layout,
        lse_shape=tuple(query.shape[:-1]),
        query_strides=(attention_tensor_strides(tensor_layout, query) if bind_launch_strides else (0, 0, 0, 0, 0)),
        key_strides=attention_tensor_strides(tensor_layout, key) if bind_launch_strides else (0, 0, 0, 0, 0),
        value_strides=(attention_tensor_strides(tensor_layout, value) if bind_launch_strides else (0, 0, 0, 0, 0)),
        mask_strides=pair_mask_strides(mask_kind, mask) if bind_launch_strides else (0, 0, 0, 0),
        bias_strides=bias_strides,
        shared_bias_strides=(
            attention_shared_bias_strides(pair_bias) if bind_launch_strides and pair_bias is not None else (0, 0, 0, 0)
        ),
        bias_groups=bias_groups,
    )


_PAIR_BIAS_BACKEND = FlashAttentionBackend(
    operator="pair_bias_attention",
    float32_dot=_PAIR_BIAS_FLOAT32_DOT,
    bind_problem=_bind_pair_bias_problem,
    schedules=SCHEDULES,
    contiguous_fake_bias_gradient=True,
)

_SEGMENTED_PAIR_BIAS_FLOAT32_DOT = FLOAT32_DOT_TF32X3
_SEGMENTED_PAIR_BIAS_EAGER_ADMISSION: tuple[AttentionAdmissionKey, int, int, int, int] | None = None


def _segmented_problem(
    query: Tensor,
    mask_kind: int,
    query_sample_offsets: Tensor,
    precision: str,
) -> SegmentedAttentionProblem:
    return SegmentedAttentionProblem(
        num_samples=query_sample_offsets.shape[0] - 1,
        total_positions=query.shape[0] * query.shape[1],
        num_heads=query.shape[2],
        head_size=query.shape[3],
        dtype=query.dtype,
        element_size=query.element_size(),
        mask_kind=mask_kind,
        input_precision=precision,
    )


def _supports_bias_dtype(query: Tensor, pair_bias: Tensor | None) -> bool:
    if pair_bias is None:
        return True
    autocast_float32_bias = (
        query.dtype in HALF_DTYPES
        and pair_bias.dtype is torch.float32
        and pair_bias.device == query.device
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == query.dtype
    )
    return pair_bias.dtype == query.dtype or autocast_float32_bias


def supports_segmented_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    *,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
) -> bool:

    global _SEGMENTED_PAIR_BIAS_EAGER_ADMISSION

    offsets = (query_sample_offsets, key_sample_offsets, mask_sample_offsets, bias_sample_offsets)
    supported = (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and tensors_share_device(query, key, value, mask, pair_bias, *offsets)
        and query.dim() == key.dim() == value.dim() == 4
        and query.shape[1:] == key.shape[1:] == value.shape[1:]
        and query.dtype in STANDARD_FLOAT_DTYPES
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and query.shape[0] > 0
        and key.shape[0] > 0
        and query.shape[1] > 0
        and query.shape[2] > 0
        and 0 < query.shape[3] <= 256
        and all(
            offset.is_cuda and offset.dtype is torch.int64 and offset.dim() == 1 and offset.is_contiguous()
            for offset in offsets
        )
        and query_sample_offsets.shape
        == key_sample_offsets.shape
        == mask_sample_offsets.shape
        == bias_sample_offsets.shape
        and query_sample_offsets.shape[0] > 1
        and (
            mask is None
            or (mask.dtype is torch.bool and mask.dim() == 2 and mask.shape[1] == mask_groups and mask.is_contiguous())
        )
        and (
            pair_bias is None
            or (
                pair_bias.dim() == 3
                and pair_bias.shape[1] == bias_groups
                and pair_bias.shape[2] == bias_heads
                and pair_bias.is_contiguous()
                and _supports_bias_dtype(query, pair_bias)
            )
        )
    )
    if not supported:
        return False
    precision = matmul_precision(query.dtype, _SEGMENTED_PAIR_BIAS_FLOAT32_DOT)
    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(query)
    admission_key = None
    if not compiling and not external_fake:
        tensor_key = attention_admission_key(
            (query, key, value, mask, pair_bias, *offsets),
            precision=precision,
        )
        admission_key = tensor_key, mask_kind, mask_groups, bias_groups, bias_heads
        if admission_key == _SEGMENTED_PAIR_BIAS_EAGER_ADMISSION:
            return True
    problem = _segmented_problem(query, mask_kind, query_sample_offsets, precision)
    phases: tuple[AttentionPhase, ...] = (
        AttentionPhase.FORWARD,
        AttentionPhase.BACKWARD_PREPROCESS,
        AttentionPhase.BACKWARD_QUERY,
        AttentionPhase.BACKWARD_KEY_VALUE,
    )
    if pair_bias is not None:
        phases += (AttentionPhase.GRAD_BIAS,)
    supported = all(SEGMENTED_SCHEDULES.supports(problem, phase) for phase in phases)
    if supported and admission_key is not None:
        _SEGMENTED_PAIR_BIAS_EAGER_ADMISSION = admission_key
    return supported


def _resolve_segmented_plan(
    problem: SegmentedAttentionProblem,
    phase: AttentionPhase,
    device: torch.device,
) -> SegmentedAttentionPlan:
    return SEGMENTED_SCHEDULES.resolve(problem, phase, target=cuda_device_caps(device))


def _run_segmented_pair_bias_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    *,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
    write_lse: bool = True,
) -> tuple[Tensor, Tensor | None]:
    with cuda_device_context(query.device):
        original_query, original_key = query, key
        query = _resolve_attention_operand(query)
        key = query if key is original_query else _resolve_attention_operand(key)
        value = (
            query if value is original_query else key if value is original_key else _resolve_attention_operand(value)
        )
        mask = None if mask is None else _resolve_attention_operand(mask)
        pair_bias = None if pair_bias is None else _resolve_attention_operand(pair_bias)
        query_sample_offsets = _resolve_attention_operand(query_sample_offsets).contiguous()
        key_sample_offsets = _resolve_attention_operand(key_sample_offsets).contiguous()
        mask_sample_offsets = _resolve_attention_operand(mask_sample_offsets).contiguous()
        bias_sample_offsets = _resolve_attention_operand(bias_sample_offsets).contiguous()
        require_triton_dot_precision(
            "pair_bias_attention",
            query.dtype,
            _SEGMENTED_PAIR_BIAS_FLOAT32_DOT,
            precision,
        )
        problem = _segmented_problem(query, mask_kind, query_sample_offsets, precision)
        plan = _resolve_segmented_plan(problem, AttentionPhase.FORWARD, query.device)
        tile = plan.tile
        block_key = tile.block_key
        assert block_key is not None
        # Backward centering reads the normalized output before storage-dtype rounding.
        output = torch.empty_like(query, dtype=torch.float32 if write_lse else query.dtype)
        lse = torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32) if write_lse else None
        kernel._segmented_pair_bias_attention_forward[plan.grid](
            query,
            key,
            value,
            optional_tensor(mask, query),
            optional_tensor(pair_bias, query),
            output,
            optional_tensor(lse, query),
            query_sample_offsets,
            key_sample_offsets,
            mask_sample_offsets,
            bias_sample_offsets,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *((0, 0) if mask is None else mask.stride()),
            *((0, 0, 0) if pair_bias is None else pair_bias.stride()),
            *output.stride(),
            *((0, 0, 0) if lse is None else lse.stride()),
            problem.num_samples,
            query.shape[1],
            problem.num_heads,
            mask_groups,
            bias_groups,
            bias_heads,
            problem.head_size,
            plan.grid[0],
            resolve_scale(problem.head_size, scale),
            mask_kind,
            pair_bias is not None,
            write_lse,
            tile.block_query,
            block_key,
            tile.block_channel,
            input_precision=precision,
            num_warps=tile.num_warps,
            num_stages=tile.num_stages,
        )
        return output, lse


def _run_segmented_pair_bias_attention_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    output: Tensor,
    lse: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    *,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    with cuda_device_context(query.device):
        original_query, original_key = query, key
        query = _resolve_attention_operand(query)
        key = query if key is original_query else _resolve_attention_operand(key)
        needs_delta = need_grad_query or need_grad_key or need_grad_bias
        if needs_delta:
            value = (
                query
                if value is original_query
                else key if value is original_key else _resolve_attention_operand(value)
            )
            output = _resolve_attention_operand(output)
        mask = None if mask is None else _resolve_attention_operand(mask)
        pair_bias = None if pair_bias is None else _resolve_attention_operand(pair_bias)
        lse = _resolve_attention_operand(lse)
        grad_output = _resolve_attention_operand(grad_output.to(query.dtype))
        query_sample_offsets = _resolve_attention_operand(query_sample_offsets).contiguous()
        key_sample_offsets = _resolve_attention_operand(key_sample_offsets).contiguous()
        mask_sample_offsets = _resolve_attention_operand(mask_sample_offsets).contiguous()
        bias_sample_offsets = _resolve_attention_operand(bias_sample_offsets).contiguous()
        problem = _segmented_problem(query, mask_kind, query_sample_offsets, precision)
        effective_scale = resolve_scale(problem.head_size, scale)
        keep_mask = optional_tensor(mask, query)
        kernel_bias = optional_tensor(pair_bias, query)
        needs_delta = need_grad_query or need_grad_key or need_grad_bias
        delta: Tensor | None = None
        if needs_delta:
            preprocess_plan = _resolve_segmented_plan(problem, AttentionPhase.BACKWARD_PREPROCESS, query.device)
            preprocess_tile = preprocess_plan.tile
            delta = torch.empty_like(lse)
            kernel._segmented_pair_bias_attention_backward_preprocess[preprocess_plan.grid](
                output,
                grad_output,
                delta,
                query_sample_offsets,
                *output.stride(),
                *grad_output.stride(),
                *delta.stride(),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                problem.head_size,
                preprocess_plan.grid[0],
                preprocess_tile.block_query,
                preprocess_tile.block_channel,
                num_warps=preprocess_tile.num_warps,
            )

        grad_bias = torch.empty_like(pair_bias) if pair_bias is not None and need_grad_bias else None
        full_bias_gradient = grad_bias is not None and bias_groups == query.shape[1] and bias_heads == query.shape[2]
        grad_query: Tensor | None = None
        if need_grad_query or full_bias_gradient:
            assert delta is not None
            query_plan = _resolve_segmented_plan(problem, AttentionPhase.BACKWARD_QUERY, query.device)
            query_tile = query_plan.tile
            query_block_key = query_tile.block_key
            assert query_block_key is not None
            grad_query = torch.empty_like(query) if need_grad_query else None
            kernel._segmented_pair_bias_attention_backward_query[query_plan.grid](
                query,
                key,
                value,
                keep_mask,
                kernel_bias,
                lse,
                delta,
                grad_output,
                optional_tensor(grad_query, query),
                optional_tensor(grad_bias, query),
                query_sample_offsets,
                key_sample_offsets,
                mask_sample_offsets,
                bias_sample_offsets,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *((0, 0) if mask is None else mask.stride()),
                *((0, 0, 0) if pair_bias is None else pair_bias.stride()),
                *lse.stride(),
                *delta.stride(),
                *grad_output.stride(),
                *((0, 0, 0, 0) if grad_query is None else grad_query.stride()),
                *((0, 0, 0) if grad_bias is None else grad_bias.stride()),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                mask_groups,
                bias_groups,
                bias_heads,
                problem.head_size,
                query_plan.grid[0],
                effective_scale,
                mask_kind,
                pair_bias is not None,
                need_grad_query,
                full_bias_gradient,
                query_tile.block_query,
                query_block_key,
                query_tile.block_channel,
                input_precision=precision,
                num_warps=query_tile.num_warps,
                num_stages=query_tile.num_stages,
            )

        grad_key: Tensor | None = None
        grad_value: Tensor | None = None
        if need_grad_key or need_grad_value:
            key_value_plan = _resolve_segmented_plan(problem, AttentionPhase.BACKWARD_KEY_VALUE, query.device)
            key_value_tile = key_value_plan.tile
            key_value_block_key = key_value_tile.block_key
            assert key_value_block_key is not None
            grad_key = torch.empty_like(key) if need_grad_key else None
            grad_value = torch.empty_like(value) if need_grad_value else None
            kernel._segmented_pair_bias_attention_backward_key_value[key_value_plan.grid](
                query,
                key,
                value,
                keep_mask,
                kernel_bias,
                lse,
                optional_tensor(delta, lse),
                grad_output,
                optional_tensor(grad_key, key),
                optional_tensor(grad_value, value),
                query_sample_offsets,
                key_sample_offsets,
                mask_sample_offsets,
                bias_sample_offsets,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *((0, 0) if mask is None else mask.stride()),
                *((0, 0, 0) if pair_bias is None else pair_bias.stride()),
                *lse.stride(),
                *((0, 0, 0) if delta is None else delta.stride()),
                *grad_output.stride(),
                *((0, 0, 0, 0) if grad_key is None else grad_key.stride()),
                *((0, 0, 0, 0) if grad_value is None else grad_value.stride()),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                mask_groups,
                bias_groups,
                bias_heads,
                problem.head_size,
                key_value_plan.grid[0],
                effective_scale,
                mask_kind,
                pair_bias is not None,
                need_grad_key,
                need_grad_value,
                key_value_tile.block_query,
                key_value_block_key,
                key_value_tile.block_channel,
                input_precision=precision,
                num_warps=key_value_tile.num_warps,
                num_stages=key_value_tile.num_stages,
            )

        if grad_bias is not None and not full_bias_gradient:
            assert delta is not None
            assert pair_bias is not None
            grad_bias_plan = _resolve_segmented_plan(problem, AttentionPhase.GRAD_BIAS, query.device)
            grad_bias_tile = grad_bias_plan.tile
            grad_bias_block_key = grad_bias_tile.block_key
            assert grad_bias_block_key is not None
            kernel._segmented_pair_bias_attention_grad_bias[grad_bias_plan.grid](
                query,
                key,
                value,
                keep_mask,
                pair_bias,
                lse,
                delta,
                grad_output,
                grad_bias,
                query_sample_offsets,
                key_sample_offsets,
                mask_sample_offsets,
                bias_sample_offsets,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *((0, 0) if mask is None else mask.stride()),
                *pair_bias.stride(),
                *lse.stride(),
                *delta.stride(),
                *grad_output.stride(),
                *grad_bias.stride(),
                problem.num_samples,
                query.shape[1],
                problem.num_heads,
                mask_groups,
                bias_groups,
                bias_heads,
                problem.head_size,
                grad_bias_plan.grid[0],
                effective_scale,
                mask_kind,
                bias_groups == 1,
                bias_heads == 1,
                grad_bias_tile.block_query,
                grad_bias_block_key,
                grad_bias_tile.block_channel,
                input_precision=precision,
                num_warps=grad_bias_tile.num_warps,
                num_stages=grad_bias_tile.num_stages,
            )

        return (
            grad_query if need_grad_query else None,
            grad_key if need_grad_key else None,
            grad_value if need_grad_value else None,
            grad_bias,
        )


@torch.library.custom_op(
    "ooperators::_segmented_pair_bias_attention_bias_forward", mutates_args=(), device_types="cuda"
)
def _segmented_pair_bias_attention_bias_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    output, lse = _run_segmented_pair_bias_attention_forward(
        query,
        key,
        value,
        mask,
        pair_bias,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind=mask_kind,
        mask_groups=mask_groups,
        bias_groups=bias_groups,
        bias_heads=bias_heads,
        scale=scale,
        precision=precision,
    )
    assert lse is not None
    return output, lse


@torch.library.custom_op("ooperators::_segmented_pair_bias_attention_forward", mutates_args=(), device_types="cuda")
def _segmented_pair_bias_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    output, lse = _run_segmented_pair_bias_attention_forward(
        query,
        key,
        value,
        mask,
        None,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind=mask_kind,
        mask_groups=mask_groups,
        bias_groups=0,
        bias_heads=0,
        scale=scale,
        precision=precision,
    )
    assert lse is not None
    return output, lse


@torch.library.custom_op("ooperators::_segmented_pair_bias_attention_inference", mutates_args=(), device_types="cuda")
def _segmented_pair_bias_attention_inference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
) -> Tensor:
    return _run_segmented_pair_bias_attention_forward(
        query,
        key,
        value,
        mask,
        pair_bias,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind=mask_kind,
        mask_groups=mask_groups,
        bias_groups=bias_groups,
        bias_heads=bias_heads,
        scale=scale,
        precision=precision,
        write_lse=False,
    )[0]


@torch.library.register_fake(_segmented_pair_bias_attention_bias_forward)
def _segmented_pair_bias_attention_bias_forward_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    del (
        key,
        value,
        mask,
        pair_bias,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind,
        mask_groups,
        bias_groups,
        bias_heads,
        scale,
        precision,
    )
    return (
        torch.empty_like(query, dtype=torch.float32),
        torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32),
    )


@torch.library.register_fake(_segmented_pair_bias_attention_forward)
def _segmented_pair_bias_attention_forward_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    scale: float | None,
    precision: str,
) -> tuple[Tensor, Tensor]:
    del (
        key,
        value,
        mask,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind,
        mask_groups,
        scale,
        precision,
    )
    return (
        torch.empty_like(query, dtype=torch.float32),
        torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32),
    )


@torch.library.register_fake(_segmented_pair_bias_attention_inference)
def _segmented_pair_bias_attention_inference_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
) -> Tensor:
    del (
        key,
        value,
        mask,
        pair_bias,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind,
        mask_groups,
        bias_groups,
        bias_heads,
        scale,
        precision,
    )
    return torch.empty_like(query)


_SEGMENTED_PAIR_BIAS_BIAS_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor pair_bias, "
    "Tensor output, Tensor lse, Tensor query_sample_offsets, Tensor key_sample_offsets, "
    "Tensor mask_sample_offsets, Tensor bias_sample_offsets, SymInt mask_kind, SymInt mask_groups, "
    "SymInt bias_groups, SymInt bias_heads, float? scale, str precision, bool need_grad_query, "
    "bool need_grad_key, bool need_grad_value, bool need_grad_bias) "
    "-> (Tensor?, Tensor?, Tensor?, Tensor?)"
)
_SEGMENTED_PAIR_BIAS_BACKWARD_SCHEMA = (
    "(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor? mask, Tensor output, "
    "Tensor lse, Tensor query_sample_offsets, Tensor key_sample_offsets, Tensor mask_sample_offsets, "
    "Tensor bias_sample_offsets, SymInt mask_kind, SymInt mask_groups, float? scale, str precision, "
    "bool need_grad_query, bool need_grad_key, bool need_grad_value) -> (Tensor?, Tensor?, Tensor?)"
)


@torch.library.custom_op(
    "ooperators::_segmented_pair_bias_attention_bias_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_SEGMENTED_PAIR_BIAS_BIAS_BACKWARD_SCHEMA,
)
def _segmented_pair_bias_attention_bias_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    output: Tensor,
    lse: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    return _run_segmented_pair_bias_attention_backward(
        grad_output,
        query,
        key,
        value,
        mask,
        pair_bias,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind=mask_kind,
        mask_groups=mask_groups,
        bias_groups=bias_groups,
        bias_heads=bias_heads,
        scale=scale,
        precision=precision,
        need_grad_query=need_grad_query,
        need_grad_key=need_grad_key,
        need_grad_value=need_grad_value,
        need_grad_bias=need_grad_bias,
    )


@torch.library.custom_op(
    "ooperators::_segmented_pair_bias_attention_backward",
    mutates_args=(),
    device_types="cuda",
    schema=_SEGMENTED_PAIR_BIAS_BACKWARD_SCHEMA,
)
def _segmented_pair_bias_attention_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    output: Tensor,
    lse: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    grad_query, grad_key, grad_value, _ = _run_segmented_pair_bias_attention_backward(
        grad_output,
        query,
        key,
        value,
        mask,
        None,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind=mask_kind,
        mask_groups=mask_groups,
        bias_groups=0,
        bias_heads=0,
        scale=scale,
        precision=precision,
        need_grad_query=need_grad_query,
        need_grad_key=need_grad_key,
        need_grad_value=need_grad_value,
        need_grad_bias=False,
    )
    return grad_query, grad_key, grad_value


@torch.library.register_fake(_segmented_pair_bias_attention_bias_backward)
def _segmented_pair_bias_attention_bias_backward_fake(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor,
    output: Tensor,
    lse: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    del (
        grad_output,
        mask,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind,
        mask_groups,
        bias_groups,
        bias_heads,
        scale,
        precision,
    )
    return (
        torch.empty_like(query) if need_grad_query else None,
        torch.empty_like(key) if need_grad_key else None,
        torch.empty_like(value) if need_grad_value else None,
        torch.empty_like(pair_bias) if need_grad_bias else None,
    )


@torch.library.register_fake(_segmented_pair_bias_attention_backward)
def _segmented_pair_bias_attention_backward_fake(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    output: Tensor,
    lse: Tensor,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    mask_kind: int,
    mask_groups: int,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    del (
        grad_output,
        mask,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind,
        mask_groups,
        scale,
        precision,
    )
    return (
        torch.empty_like(query) if need_grad_query else None,
        torch.empty_like(key) if need_grad_key else None,
        torch.empty_like(value) if need_grad_value else None,
    )


def _segmented_pair_bias_setup_context(ctx, inputs, output) -> None:
    (
        query,
        key,
        value,
        mask,
        pair_bias,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind,
        mask_groups,
        bias_groups,
        bias_heads,
        scale,
        precision,
    ) = inputs
    attention_output, lse = output
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[4]
    ctx.mask_kind = mask_kind
    ctx.mask_groups = mask_groups
    ctx.bias_groups = bias_groups
    ctx.bias_heads = bias_heads
    ctx.scale = scale
    ctx.precision = precision
    ctx.needs_delta = needs_delta
    ctx.mark_non_differentiable(lse)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        query,
        key,
        value if needs_delta else key,
        mask,
        pair_bias,
        attention_output if needs_delta else query,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
    )


def _segmented_pair_setup_context(ctx, inputs, output) -> None:
    (
        query,
        key,
        value,
        mask,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        mask_kind,
        mask_groups,
        scale,
        precision,
    ) = inputs
    attention_output, lse = output
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
    ctx.mask_kind = mask_kind
    ctx.mask_groups = mask_groups
    ctx.scale = scale
    ctx.precision = precision
    ctx.needs_delta = needs_delta
    ctx.mark_non_differentiable(lse)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        query,
        key,
        value if needs_delta else key,
        mask,
        attention_output if needs_delta else query,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
    )


def _segmented_pair_bias_autograd_backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
    del grad_lse
    (
        query,
        key,
        value,
        mask,
        pair_bias,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
    ) = ctx.saved_tensors
    if not ctx.needs_delta:
        value = key
    grad_query, grad_key, grad_value, grad_bias = _segmented_pair_bias_attention_bias_backward(
        grad_output,
        query,
        key,
        value,
        mask,
        pair_bias,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        ctx.mask_kind,
        ctx.mask_groups,
        ctx.bias_groups,
        ctx.bias_heads,
        ctx.scale,
        ctx.precision,
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[1],
        ctx.needs_input_grad[2],
        ctx.needs_input_grad[4],
    )
    return (
        grad_query,
        grad_key,
        grad_value,
        None,
        grad_bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _segmented_pair_autograd_backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
    del grad_lse
    (
        query,
        key,
        value,
        mask,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
    ) = ctx.saved_tensors
    if not ctx.needs_delta:
        value = key
    grad_query, grad_key, grad_value = _segmented_pair_bias_attention_backward(
        grad_output,
        query,
        key,
        value,
        mask,
        output,
        lse,
        query_sample_offsets,
        key_sample_offsets,
        mask_sample_offsets,
        bias_sample_offsets,
        ctx.mask_kind,
        ctx.mask_groups,
        ctx.scale,
        ctx.precision,
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[1],
        ctx.needs_input_grad[2],
    )
    return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    _segmented_pair_bias_attention_bias_forward,
    _segmented_pair_bias_autograd_backward,
    setup_context=_segmented_pair_bias_setup_context,
)
torch.library.register_autograd(
    _segmented_pair_bias_attention_forward,
    _segmented_pair_autograd_backward,
    setup_context=_segmented_pair_setup_context,
)


def triton_segmented_pair_bias_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    pair_bias: Tensor | None,
    query_sample_offsets: Tensor,
    key_sample_offsets: Tensor,
    mask_sample_offsets: Tensor,
    bias_sample_offsets: Tensor,
    *,
    mask_kind: int,
    mask_groups: int,
    bias_groups: int,
    bias_heads: int,
    scale: float | None,
) -> Tensor:

    precision = matmul_precision(query.dtype, _SEGMENTED_PAIR_BIAS_FLOAT32_DOT)
    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(query)
    if not tensors_need_grad(query, key, value, pair_bias):
        if not compiling and not external_fake:
            return _run_segmented_pair_bias_attention_forward(
                query,
                key,
                value,
                mask,
                pair_bias,
                query_sample_offsets,
                key_sample_offsets,
                mask_sample_offsets,
                bias_sample_offsets,
                mask_kind=mask_kind,
                mask_groups=mask_groups,
                bias_groups=bias_groups,
                bias_heads=bias_heads,
                scale=scale,
                precision=precision,
                write_lse=False,
            )[0]
        return _segmented_pair_bias_attention_inference(
            query,
            key,
            value,
            mask,
            pair_bias,
            query_sample_offsets,
            key_sample_offsets,
            mask_sample_offsets,
            bias_sample_offsets,
            mask_kind,
            mask_groups,
            bias_groups,
            bias_heads,
            scale,
            precision,
        )
    if pair_bias is not None:
        output, _lse = _segmented_pair_bias_attention_bias_forward(
            query,
            key,
            value,
            mask,
            pair_bias,
            query_sample_offsets,
            key_sample_offsets,
            mask_sample_offsets,
            bias_sample_offsets,
            mask_kind,
            mask_groups,
            bias_groups,
            bias_heads,
            scale,
            precision,
        )
    else:
        output, _lse = _segmented_pair_bias_attention_forward(
            query,
            key,
            value,
            mask,
            query_sample_offsets,
            key_sample_offsets,
            mask_sample_offsets,
            bias_sample_offsets,
            mask_kind,
            mask_groups,
            scale,
            precision,
        )
    return output.to(query.dtype)
