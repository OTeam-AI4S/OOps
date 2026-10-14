from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import cuda_device_context

from . import kernel
from .profiles import SCHEDULES
from .schedule import LayerNormProblem


def layer_norm_forward_into(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    eps: float,
    *,
    mean: Tensor | None = None,
    rstd: Tensor | None = None,
) -> None:
    r"""Write LayerNorm into caller-owned contiguous output storage.

    Input is admitted nonempty CUDA storage, normalized over its final dimension.
    Affine tensors are contiguous rows. Output matches input shape, device and dtype,
    and must not alias inputs. Optional FP32 statistics have one element per row
    and must both be supplied. This executor creates no autograd node.
    """
    if (mean is None) != (rstd is None):
        raise ValueError("mean and rstd must both be supplied")
    with cuda_device_context(input.device):
        weight = None if weight is None else weight.resolve_neg().contiguous()
        bias = None if bias is None else bias.resolve_neg().contiguous()
        has_weight, has_bias = weight is not None, bias is not None
        input, row_map = _row_addressing(input)
        # input is an inert pointer for absent affine parameters.
        weight = input if weight is None else weight
        bias = input if bias is None else bias
        num_rows, num_columns = _matrix_shape(input)
        plan = SCHEDULES.resolve_forward(LayerNormProblem(num_rows, num_columns, input.dtype))
        kernel._layer_norm_forward[plan.grid](
            input,
            weight,
            bias,
            output,
            input if mean is None else mean,
            input if rstd is None else rstd,
            num_rows,
            num_columns,
            *row_map,
            eps,
            has_weight,
            has_bias,
            mean is not None,
            plan.block_rows,
            plan.block_columns,
            num_warps=plan.num_warps,
        )


def _matrix_shape(input: Tensor) -> tuple[int, int]:
    num_columns = input.shape[-1]
    return input.numel() // num_columns, num_columns


_PACKED_ROWS = (False, 0, 0, 0)


def _row_addressing(input: Tensor) -> tuple[Tensor, tuple[bool, int, int, int]]:
    r"""Return input and its row-start map.

    For two leading stride groups, row r starts at
    (r // inner_rows) * outer_stride + (r % inner_rows) * inner_stride.
    """

    input = input.resolve_neg()
    if input.is_contiguous():
        return input, _PACKED_ROWS
    if input.stride(-1) != 1:
        return input.contiguous(), _PACKED_ROWS
    extents: list[int] = []
    strides: list[int] = []
    for extent, stride in zip(input.shape[:-1], input.stride()[:-1]):
        if extent == 1:
            continue
        if extents and strides[-1] == stride * extent:
            extents[-1] *= extent
            strides[-1] = stride
        else:
            extents.append(extent)
            strides.append(stride)
    if not extents:
        return input, _PACKED_ROWS
    if len(extents) > 2:
        return input.contiguous(), _PACKED_ROWS
    if len(extents) == 1:
        return input, (True, extents[0], strides[0], 0)
    return input, (True, extents[1], strides[1], strides[0])


def layer_norm_forward_with_stats(
    input: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    num_rows, _ = _matrix_shape(input)
    output = input.new_empty(input.shape)
    mean = input.new_empty((num_rows,), dtype=torch.float32)
    rstd = mean.new_empty(mean.shape)
    layer_norm_forward_into(input, weight, bias, output, eps, mean=mean, rstd=rstd)
    return output, mean, rstd
