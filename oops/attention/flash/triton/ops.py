from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, TypeAlias

import torch
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_false, statically_known_true

from oops.runtime import (
    CUDA_MAX_GRID_X,
    INT32_MAX,
    cuda_device_caps_if_needed,
    cuda_device_context,
    is_compiling,
    is_external_fake,
    matmul_precision,
)
from oops.runtime import needs_grad as tensors_need_grad
from oops.runtime import require_triton_dot_precision, unsupported

from ..mask import MASK_PAIR
from . import kernel
from .problem import (
    AttentionProblem,
    Strides5,
    attention_bias_strides,
    attention_lse_strides,
    attention_shared_bias_strides,
    attention_tensor_strides,
    optional_tensor,
    pair_mask_strides,
    resolve_scale,
)
from .schedule import AttentionLaunchPlan, AttentionPhase, AttentionScheduleCatalog


class ProblemBinder(Protocol):
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        bias: Tensor | None,
        /,
        *,
        bind_launch_strides: bool,
    ) -> AttentionProblem: ...


TensorMetadata: TypeAlias = tuple[
    torch.device,
    torch.dtype,
    torch.layout,
    tuple[int, ...],
    tuple[int, ...],
]
AttentionAdmissionKey: TypeAlias = tuple[
    tuple[TensorMetadata | None, ...],
    bool,
    torch.dtype | None,
    str,
]


@dataclass(frozen=True, slots=True)
class FlashAttentionBackend:

    operator: str
    float32_dot: str
    bind_problem: ProblemBinder
    schedules: AttentionScheduleCatalog
    contiguous_fake_bias_gradient: bool = False


@dataclass(frozen=True, slots=True)
class PreparedAttention:

    problem: AttentionProblem
    forward_plan: AttentionLaunchPlan
    precision: str


def attention_admission_key(
    tensors: tuple[Tensor | None, ...],
    *,
    precision: str,
) -> AttentionAdmissionKey:

    metadata = tuple(
        (
            None
            if tensor is None
            else (
                tensor.device,
                tensor.dtype,
                tensor.layout,
                tensor.shape,
                tensor.stride(),
            )
        )
        for tensor in tensors
    )
    autocast_enabled = torch.is_autocast_enabled("cuda")
    autocast_dtype = torch.get_autocast_dtype("cuda") if autocast_enabled else None
    return metadata, autocast_enabled, autocast_dtype, precision


def attention_grids_launchable(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    *,
    maximum_grid_x: int,
    maximum_grid_yz: int,
) -> bool:

    compiling = is_compiling()
    external_fake = False if compiling else is_external_fake(query)
    problem = backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=False)
    precision = matmul_precision(query.dtype, backend.float32_dot)
    return _attention_problem_grids_launchable(
        backend,
        problem,
        bias is not None,
        precision,
        cache=not external_fake,
        maximum_grid_x=maximum_grid_x,
        maximum_grid_yz=maximum_grid_yz,
    )


def prepare_attention(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    *,
    precision: str,
    maximum_grid_x: int,
    maximum_grid_yz: int,
) -> PreparedAttention | None:

    problem = backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=True)
    if not _attention_problem_grids_launchable(
        backend,
        problem,
        bias is not None,
        precision,
        maximum_grid_x=maximum_grid_x,
        maximum_grid_yz=maximum_grid_yz,
    ):
        return None
    require_in_slice_offsets_fit(backend, problem, bias is not None)
    target = cuda_device_caps_if_needed(
        query.device,
        backend.schedules.target_changes_plan(
            problem,
            (AttentionPhase.FORWARD,),
            precision=precision,
        ),
    )
    forward_plan = backend.schedules.resolve(
        problem,
        AttentionPhase.FORWARD,
        precision=precision,
        target=target,
    )
    return PreparedAttention(problem=problem, forward_plan=forward_plan, precision=precision)


def _attention_problem_grids_launchable(
    backend: FlashAttentionBackend,
    problem: AttentionProblem,
    has_bias: bool,
    precision: str,
    *,
    cache: bool = True,
    maximum_grid_x: int,
    maximum_grid_yz: int,
) -> bool:
    if is_compiling() or not cache:
        batch_group_heads = problem.batch_size * problem.num_groups * problem.num_heads
        torch._check(batch_group_heads > 0)
        torch._check(batch_group_heads <= maximum_grid_x)
        batch_heads = problem.batch_size * problem.num_heads
        torch._check(batch_heads > 0)
        torch._check(batch_heads <= maximum_grid_x)
        problem = replace(problem, batch_size=1, num_groups=1)
    phases: tuple[AttentionPhase, ...] = (
        AttentionPhase.FORWARD,
        AttentionPhase.BACKWARD_PREPROCESS,
        AttentionPhase.BACKWARD_QUERY,
        AttentionPhase.BACKWARD_KEY_VALUE,
    )
    if has_bias and statically_known_true(problem.bias_groups == 1):
        phases += (AttentionPhase.GRAD_BIAS,)
    launchable = True
    for phase in phases:
        launchable = launchable & backend.schedules.grids_fit(
            problem,
            phase,
            precision=precision,
            maximum_grid_x=maximum_grid_x,
            maximum_grid_yz=maximum_grid_yz,
            cache=cache,
        )
    if launchable is True or launchable is False:
        return launchable
    if statically_known_false(launchable):
        return False
    # Avoid Torch 2.13's implication-simplification bug for compound predicates.
    torch._check(torch.sym_ite(launchable, 1, 0) == 1)
    return True


_MAX_TILE = 256


def _in_slice_span(first_extent: int, first_stride: int, second_extent: int, second_stride: int) -> int:
    r"""The span includes masked lanes in the final padded tile."""

    def last(extent: int) -> int:
        return -(-extent // _MAX_TILE) * _MAX_TILE - 1

    return last(first_extent) * abs(first_stride) + last(second_extent) * abs(second_stride)


def in_slice_offsets_fit(problem: AttentionProblem, has_bias: bool, *token_major: Strides5) -> bool:
    r"""Int32 lane offsets, including masked tail lanes, must not overflow; slice bases use int64."""

    queries, keys, channels = problem.query_length, problem.key_length, problem.head_size
    spans = [
        _in_slice_span(queries, problem.query_strides[2], channels, problem.query_strides[4]),
        _in_slice_span(keys, problem.key_strides[2], channels, problem.key_strides[4]),
        _in_slice_span(keys, problem.value_strides[2], channels, problem.value_strides[4]),
        *(_in_slice_span(queries, strides[2], channels, strides[4]) for strides in token_major),
    ]
    if problem.mask_kind == MASK_PAIR or has_bias:
        spans.append(_in_slice_span(queries, keys, keys, 1))
    if has_bias:
        spans.append(_in_slice_span(queries, problem.bias_strides[3], keys, problem.bias_strides[4]))
    return max(spans) <= INT32_MAX


def require_in_slice_offsets_fit(
    backend: FlashAttentionBackend,
    problem: AttentionProblem,
    has_bias: bool,
    *token_major: Strides5,
) -> None:

    if not in_slice_offsets_fit(problem, has_bias, *token_major):
        raise unsupported(
            backend.operator,
            "requires every (batch, group, head) slice of each operand to span fewer than 2**31 elements",
            query_length=problem.query_length,
            key_length=problem.key_length,
            head_size=problem.head_size,
            query_strides=problem.query_strides,
            key_strides=problem.key_strides,
            value_strides=problem.value_strides,
            bias_strides=problem.bias_strides if has_bias else None,
        )


def dispatch_attention(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    scale: float | None,
    inference_op,
    bias_forward_op,
    forward_op,
    prepared: PreparedAttention | None = None,
) -> Tensor:

    if mask is not None:
        previous_strides = mask.stride()
        mask = mask.to(dtype=torch.int16, memory_format=torch.contiguous_format)
        if prepared is not None and mask.stride() != previous_strides:
            prepared = replace(
                prepared,
                problem=backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=True),
            )
    precision = prepared.precision if prepared is not None else matmul_precision(query.dtype, backend.float32_dot)
    if prepared is None:
        compiling = is_compiling()
        external_fake = False if compiling else is_external_fake(query)
    else:
        compiling = external_fake = False
    if not tensors_need_grad(query, key, value, bias):
        if not compiling and not external_fake:
            return run_attention_forward(
                backend,
                query,
                key,
                value,
                mask,
                bias,
                scale=scale,
                precision=precision,
                write_lse=False,
                prepared=prepared,
            )[0]
        return inference_op(query, key, value, mask, bias, scale, precision)
    if bias is not None:
        output, _lse = bias_forward_op(query, key, value, mask, bias, scale, precision)
    else:
        output, _lse = forward_op(query, key, value, mask, scale, precision)
    return output.to(query.dtype)


def _resolve_attention_operand(tensor: Tensor) -> Tensor:
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    return tensor


def run_attention_forward(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    *,
    scale: float | None,
    precision: str,
    write_lse: bool = True,
    prepared: PreparedAttention | None = None,
) -> tuple[Tensor, Tensor | None]:

    with cuda_device_context(query.device):
        originals = (query, key, value, mask, bias)
        query = _resolve_attention_operand(query)
        key = query if key is originals[0] else _resolve_attention_operand(key)
        value = query if value is originals[0] else key if value is originals[1] else _resolve_attention_operand(value)
        mask = None if mask is None else _resolve_attention_operand(mask)
        bias = None if bias is None else _resolve_attention_operand(bias)
        if prepared is not None and any(
            resolved is not original
            for resolved, original in zip((query, key, value, mask, bias), originals, strict=True)
        ):
            prepared = None
        require_triton_dot_precision(backend.operator, query.dtype, backend.float32_dot, precision)
        problem = (
            prepared.problem
            if prepared is not None
            else backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=True)
        )
        if prepared is None:
            require_in_slice_offsets_fit(backend, problem, bias is not None)
            target = cuda_device_caps_if_needed(
                query.device,
                backend.schedules.target_changes_plan(
                    problem,
                    (AttentionPhase.FORWARD,),
                    precision=precision,
                ),
            )
            plan = backend.schedules.resolve(problem, AttentionPhase.FORWARD, precision=precision, target=target)
        else:
            plan = prepared.forward_plan
        output = torch.empty_like(query, dtype=torch.float32 if write_lse else query.dtype)
        lse = torch.empty(problem.lse_shape, device=query.device, dtype=torch.float32) if write_lse else None
        tile = plan.tile
        block_key = tile.block_key
        assert block_key is not None

        kernel._flash_attention_forward.run(
            query,
            key,
            value,
            optional_tensor(mask, query),
            optional_tensor(bias, query),
            output,
            optional_tensor(lse, query),
            *problem.query_strides,
            *problem.key_strides,
            *problem.value_strides,
            *problem.mask_strides,
            *problem.bias_strides,
            *attention_tensor_strides(problem.tensor_layout, output),
            *attention_lse_strides(lse),
            problem.num_groups,
            problem.query_length,
            problem.key_length,
            problem.num_heads,
            problem.head_size,
            problem.bias_groups,
            resolve_scale(problem.head_size, scale),
            problem.mask_kind,
            bias is not None,
            write_lse,
            problem.query_length % tile.block_query == 0,
            problem.key_length % block_key == 0,
            problem.head_size == tile.block_channel,
            tile.block_query,
            block_key,
            tile.block_channel,
            input_precision=precision,
            num_warps=tile.num_warps,
            num_stages=tile.num_stages,
            grid=plan.grid,
            warmup=False,
        )
        return output, lse


def run_attention_backward(
    backend: FlashAttentionBackend,
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    lse: Tensor,
    *,
    scale: float | None,
    precision: str,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
    prepared: PreparedAttention | None = None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:

    with cuda_device_context(query.device):
        originals = (query, key, value, mask, bias)
        query = _resolve_attention_operand(query)
        key = query if key is originals[0] else _resolve_attention_operand(key)
        needs_delta = need_grad_query or need_grad_key or (bias is not None and need_grad_bias)
        if needs_delta:
            value = (
                query if value is originals[0] else key if value is originals[1] else _resolve_attention_operand(value)
            )
            output = _resolve_attention_operand(output)
        mask = None if mask is None else _resolve_attention_operand(mask)
        bias = None if bias is None else _resolve_attention_operand(bias)
        # The backward kernels share LSE and delta strides.
        lse = _resolve_attention_operand(lse).contiguous()
        grad_output = _resolve_attention_operand(grad_output.to(query.dtype))
        if prepared is not None and any(
            resolved is not original
            for resolved, original in zip((query, key, value, mask, bias), originals, strict=True)
        ):
            prepared = None
        if prepared is not None:
            # Saved-tensor hooks may unpack the same values into a new layout.
            bound = prepared.problem
            bias_strides = attention_bias_strides(bias)
            if bound.bias_groups == 1:
                bias_strides = bias_strides[0], 0, *bias_strides[2:]
            if (
                attention_tensor_strides(bound.tensor_layout, query) != bound.query_strides
                or attention_tensor_strides(bound.tensor_layout, key) != bound.key_strides
                or (needs_delta and attention_tensor_strides(bound.tensor_layout, value) != bound.value_strides)
                or pair_mask_strides(bound.mask_kind, mask) != bound.mask_strides
                or bias_strides != bound.bias_strides
            ):
                prepared = None
        problem = (
            prepared.problem
            if prepared is not None
            else backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=True)
        )
        require_in_slice_offsets_fit(
            backend, problem, bias is not None, attention_tensor_strides(problem.tensor_layout, grad_output)
        )
        target = cuda_device_caps_if_needed(
            query.device,
            backend.schedules.target_changes_plan(
                problem,
                (
                    AttentionPhase.BACKWARD_PREPROCESS,
                    AttentionPhase.BACKWARD_QUERY,
                    AttentionPhase.BACKWARD_KEY_VALUE,
                    AttentionPhase.GRAD_BIAS,
                ),
                precision=precision,
            ),
        )
        effective_scale = resolve_scale(problem.head_size, scale)
        keep_mask = optional_tensor(mask, query)
        kernel_bias = optional_tensor(bias, query)

        wants_bias_gradient = bias is not None and need_grad_bias
        # dV = P.T @ dO needs neither V nor delta.
        delta: Tensor | None = None
        if need_grad_query or need_grad_key or wants_bias_gradient:
            preprocess_plan = backend.schedules.resolve(
                problem,
                AttentionPhase.BACKWARD_PREPROCESS,
                precision=precision,
                target=target,
            )
            preprocess_tile = preprocess_plan.tile
            delta = torch.empty_like(lse)
            kernel._flash_attention_backward_preprocess.run(
                output,
                grad_output,
                delta,
                *attention_tensor_strides(problem.tensor_layout, output),
                *attention_tensor_strides(problem.tensor_layout, grad_output),
                *attention_lse_strides(delta),
                problem.num_groups,
                problem.query_length,
                problem.num_heads,
                problem.head_size,
                problem.query_length % preprocess_tile.block_query == 0,
                problem.head_size == preprocess_tile.block_channel,
                preprocess_tile.block_query,
                preprocess_tile.block_channel,
                num_warps=preprocess_tile.num_warps,
                grid=preprocess_plan.grid,
                warmup=False,
            )

        broadcast_bias = wants_bias_gradient and problem.bias_groups == 1
        query_writes_bias = wants_bias_gradient and not broadcast_bias
        grad_bias_dtype = bias.dtype if bias is not None and query_writes_bias else torch.float32
        grad_bias = (
            torch.empty(bias.shape, device=query.device, dtype=grad_bias_dtype)
            if bias is not None and need_grad_bias
            else None
        )
        query_grad_bias = grad_bias if query_writes_bias else None

        grad_query: Tensor | None = None
        if need_grad_query or query_writes_bias:
            assert delta is not None
            query_plan = backend.schedules.resolve(
                problem,
                AttentionPhase.BACKWARD_QUERY,
                precision=precision,
                target=target,
            )
            query_tile = query_plan.tile
            query_block_key = query_tile.block_key
            assert query_block_key is not None
            grad_query = torch.empty_like(query) if need_grad_query else None
            kernel._flash_attention_backward_query.run(
                query,
                key,
                value,
                keep_mask,
                kernel_bias,
                lse,
                delta,
                grad_output,
                optional_tensor(grad_query, query),
                optional_tensor(query_grad_bias, query),
                *problem.query_strides,
                *problem.key_strides,
                *problem.value_strides,
                *problem.mask_strides,
                *problem.bias_strides,
                *attention_lse_strides(lse),
                *attention_tensor_strides(problem.tensor_layout, grad_output),
                *attention_tensor_strides(problem.tensor_layout, optional_tensor(grad_query, query)),
                *attention_bias_strides(query_grad_bias),
                problem.num_groups,
                problem.query_length,
                problem.key_length,
                problem.num_heads,
                problem.head_size,
                problem.bias_groups,
                effective_scale,
                problem.mask_kind,
                bias is not None,
                need_grad_query,
                query_writes_bias,
                problem.query_length % query_tile.block_query == 0,
                problem.key_length % query_block_key == 0,
                problem.head_size == query_tile.block_channel,
                query_tile.block_query,
                query_block_key,
                query_tile.block_channel,
                input_precision=precision,
                num_warps=query_tile.num_warps,
                num_stages=query_tile.num_stages,
                grid=query_plan.grid,
                warmup=False,
            )

        grad_key: Tensor | None = None
        grad_value: Tensor | None = None
        if need_grad_key or need_grad_value:
            key_value_plan = backend.schedules.resolve(
                problem,
                AttentionPhase.BACKWARD_KEY_VALUE,
                precision=precision,
                target=target,
            )
            key_value_tile = key_value_plan.tile
            key_value_block_key = key_value_tile.block_key
            assert key_value_block_key is not None
            grad_key = torch.empty_like(key) if need_grad_key else None
            grad_value = torch.empty_like(value) if need_grad_value else None
            bias_transposed = bias is not None and problem.bias_groups == 1
            if bias_transposed:
                assert bias is not None
                key_value_bias = bias.transpose(-1, -2).contiguous()
                batch_stride, group_stride, head_stride, key_stride, query_stride = attention_bias_strides(
                    key_value_bias
                )
                key_value_bias_strides = (batch_stride, group_stride, head_stride, query_stride, key_stride)
            else:
                key_value_bias = kernel_bias
                key_value_bias_strides = problem.bias_strides
            kernel._flash_attention_backward_key_value.run(
                query,
                key,
                value,
                keep_mask,
                key_value_bias,
                lse,
                optional_tensor(delta, lse),
                grad_output,
                optional_tensor(grad_key, key),
                optional_tensor(grad_value, value),
                *problem.query_strides,
                *problem.key_strides,
                *problem.value_strides,
                *problem.mask_strides,
                *key_value_bias_strides,
                *attention_lse_strides(lse),
                *attention_tensor_strides(problem.tensor_layout, grad_output),
                *attention_tensor_strides(problem.tensor_layout, optional_tensor(grad_key, key)),
                *attention_tensor_strides(problem.tensor_layout, optional_tensor(grad_value, value)),
                problem.num_groups,
                problem.query_length,
                problem.key_length,
                problem.num_heads,
                problem.head_size,
                problem.bias_groups,
                effective_scale,
                problem.mask_kind,
                bias is not None,
                bias_transposed,
                need_grad_key,
                need_grad_value,
                problem.query_length % key_value_tile.block_query == 0,
                problem.key_length % key_value_block_key == 0,
                problem.head_size == key_value_tile.block_channel,
                key_value_tile.block_query,
                key_value_block_key,
                key_value_tile.block_channel,
                input_precision=precision,
                num_warps=key_value_tile.num_warps,
                num_stages=key_value_tile.num_stages,
                grid=key_value_plan.grid,
                warmup=False,
            )
            del key_value_bias

        if broadcast_bias:
            assert bias is not None
            assert grad_bias is not None
            assert delta is not None
            grad_bias_plan = backend.schedules.resolve(
                problem,
                AttentionPhase.GRAD_BIAS,
                precision=precision,
                target=target,
            )
            grad_bias_tile = grad_bias_plan.tile
            grad_bias_block_key = grad_bias_tile.block_key
            assert grad_bias_block_key is not None
            group_parts = grad_bias_tile.group_parts
            partial_bytes = grad_bias.numel() * grad_bias.element_size()
            workspace_limit = grad_bias_tile.group_workspace_bytes
            grad_bias_grid = grad_bias_plan.grid
            if grad_bias_grid[0] * grad_bias_grid[1] * grad_bias_grid[2] >= 256:
                group_parts = 1
            while group_parts > 1 and (
                group_parts * 4 > problem.num_groups
                or (workspace_limit is not None and group_parts * partial_bytes > workspace_limit)
                or grad_bias_grid[0] * group_parts > CUDA_MAX_GRID_X
            ):
                group_parts //= 2
            partials = (
                torch.empty((group_parts, *grad_bias.shape), device=grad_bias.device, dtype=torch.float32)
                if group_parts > 1
                else grad_bias
            )
            kernel._flash_attention_grad_bias.run(
                query,
                key,
                value,
                keep_mask,
                bias,
                lse,
                delta,
                grad_output,
                partials,
                *problem.query_strides,
                *problem.key_strides,
                *problem.value_strides,
                *problem.mask_strides,
                *problem.shared_bias_strides,
                *attention_lse_strides(lse),
                *attention_tensor_strides(problem.tensor_layout, grad_output),
                *attention_shared_bias_strides(grad_bias),
                problem.num_groups,
                problem.query_length,
                problem.key_length,
                problem.num_heads,
                problem.head_size,
                effective_scale,
                problem.mask_kind,
                problem.query_length % grad_bias_tile.block_query == 0,
                problem.key_length % grad_bias_block_key == 0,
                problem.head_size == grad_bias_tile.block_channel,
                grad_bias_tile.block_query,
                grad_bias_block_key,
                grad_bias_tile.block_channel,
                input_precision=precision,
                group_parts=group_parts,
                partial_stride=grad_bias.numel() if group_parts > 1 else 0,
                num_warps=grad_bias_tile.num_warps,
                num_stages=grad_bias_tile.num_stages,
                grid=(grad_bias_grid[0] * group_parts, *grad_bias_grid[1:]),
                warmup=False,
            )
            if group_parts > 1:
                kernel._reduce_attention_bias_parts.run(
                    partials,
                    grad_bias,
                    grad_bias.numel(),
                    group_parts=group_parts,
                    block_size=256,
                    grid=((grad_bias.numel() + 255) // 256,),
                    warmup=False,
                )

        return (
            grad_query if need_grad_query else None,
            grad_key if need_grad_key else None,
            grad_value if need_grad_value else None,
            None if bias is None or grad_bias is None else grad_bias.to(bias.dtype),
        )


def fake_attention_forward(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor | None,
) -> tuple[Tensor, Tensor]:

    problem = backend.bind_problem(query, key, value, mask, bias, bind_launch_strides=False)
    return (
        torch.empty_like(query, dtype=torch.float32),
        torch.empty(problem.lse_shape, device=query.device, dtype=torch.float32),
    )


def fake_attention_backward(
    backend: FlashAttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Tensor | None,
    need_grad_query: bool,
    need_grad_key: bool,
    need_grad_value: bool,
    need_grad_bias: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:

    grad_bias = None
    if bias is not None and need_grad_bias:
        memory_format = torch.contiguous_format if backend.contiguous_fake_bias_gradient else torch.preserve_format
        grad_bias = torch.empty_like(bias, memory_format=memory_format)
    return (
        torch.empty_like(query) if need_grad_query else None,
        torch.empty_like(key) if need_grad_key else None,
        torch.empty_like(value) if need_grad_value else None,
        grad_bias,
    )


def setup_attention_bias_context(ctx, inputs, output) -> None:

    query, key, value, mask, bias, scale, precision = inputs
    attention_output, lse = output
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[4]
    ctx.scale = scale
    ctx.precision = precision
    ctx.mark_non_differentiable(lse)
    ctx.set_materialize_grads(False)
    # A dV-only backward never dereferences the V/output placeholder slots.
    ctx.save_for_backward(
        query, key, value if needs_delta else key, mask, bias, attention_output if needs_delta else query, lse
    )


def setup_attention_context(ctx, inputs, output) -> None:

    query, key, value, mask, scale, precision = inputs
    attention_output, lse = output
    needs_delta = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
    ctx.scale = scale
    ctx.precision = precision
    ctx.mark_non_differentiable(lse)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        query, key, value if needs_delta else key, mask, attention_output if needs_delta else query, lse
    )


def make_attention_bias_backward(backward_op):

    def backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
        query, key, value, mask, bias, output, lse = ctx.saved_tensors
        need_grad_query, need_grad_key, need_grad_value = ctx.needs_input_grad[:3]
        grad_query, grad_key, grad_value, grad_bias = backward_op(
            grad_output,
            query,
            key,
            value,
            mask,
            bias,
            output,
            lse,
            ctx.scale,
            ctx.precision,
            need_grad_query,
            need_grad_key,
            need_grad_value,
            ctx.needs_input_grad[4],
        )
        return grad_query, grad_key, grad_value, None, grad_bias, None, None

    return backward


def make_attention_backward(backward_op):

    def backward(ctx, grad_output: Tensor, grad_lse: Tensor | None):
        query, key, value, mask, output, lse = ctx.saved_tensors
        need_grad_query, need_grad_key, need_grad_value = ctx.needs_input_grad[:3]
        grad_query, grad_key, grad_value = backward_op(
            grad_output,
            query,
            key,
            value,
            mask,
            output,
            lse,
            ctx.scale,
            ctx.precision,
            need_grad_query,
            need_grad_key,
            need_grad_value,
        )
        return grad_query, grad_key, grad_value, None, None, None

    return backward
