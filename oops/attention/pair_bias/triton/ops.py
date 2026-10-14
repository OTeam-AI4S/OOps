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

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import (
    cuda_device_context,
    functorch_transforms_active,
    is_compiling,
    is_external_fake,
    matmul_precision,
)
from oops.runtime import needs_grad as tensors_need_grad
from oops.runtime.cuda import CUDA_MAX_GRID_X, CUDA_MAX_GRID_YZ

from ...flash.triton import (
    AttentionAdmissionKey,
    PreparedAttention,
    attention_admission_key,
    dispatch_attention,
    expand_pair_bias,
    fake_attention_backward,
    fake_attention_forward,
    make_attention_backward,
    make_attention_bias_backward,
    prepare_attention,
    run_attention_backward,
    run_attention_forward,
    setup_attention_bias_context,
    setup_attention_context,
    supports_head_major_qkv,
    supports_pair_bias,
    supports_pair_mask,
)
from .generic import (
    _PAIR_BIAS_BACKEND,
    _PAIR_BIAS_FLOAT32_DOT,
    _SEGMENTED_PAIR_BIAS_FLOAT32_DOT,
    _pair_bias_attention_grids_launchable,
    _run_segmented_pair_bias_attention_backward,
    _run_segmented_pair_bias_attention_forward,
)

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
