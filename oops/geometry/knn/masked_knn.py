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

import math
from numbers import Real

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import (
    is_compiling,
    is_external_fake,
    require_no_forward_mode,
    require_same_device,
    unsupported,
)

from .nested import validate_nested_masked_knn_metadata
from .triton import segmented_masked_knn, supports_segmented_masked_knn

_FLOAT32_MAX = torch.finfo(torch.float32).max


def masked_knn(
    coordinates: NestedTensor,
    mask: NestedTensor,
    num_neighbors: int,
    eps: float = 1e-6,
) -> tuple[NestedTensor, NestedTensor, NestedTensor]:
    r"""Find fixed-width masked nearest neighbors within each packed sample.

    Args:
        coordinates: Batch-first NestedTensor with canonical ``(N_i, 3)``
            CUDA float16, bfloat16, or float32 coordinate elements. Packed
            values may be non-contiguous views.
        mask: Matching NestedTensor with canonical ``(N_i,)`` elements.
            Nonzero values are true. Supported CUDA dtypes are bool, uint8,
            signed int8/int16/int32/int64, float16, bfloat16, and float32.
        num_neighbors: Integer number of output slots, from 1 through 128.
        eps: Finite non-negative real scalar no larger than FP32 maximum,
            added to squared distances before square root.

    Returns:
        A tuple of distance, sample-local index, and validity NestedTensors,
        each containing ``(N_i, num_neighbors)`` elements. Distances retain
        the coordinate dtype, indices are int64, and validity is bool.
        Samples stay independent and the input topology is preserved.

    Raises:
        TypeError: Inputs are not NestedTensors, ``num_neighbors`` is not
            an int, or ``eps`` is not a non-boolean real number.
        ValueError: Counts, scalar range, sample alignment, devices, or
            logical topology violate the contract, or the wrapper batch is empty.
        OOpsUnsupportedError: Device, dtype, packed layout, launch extent,
            or forward-mode differentiation is unsupported.

    Notes:
        Each true query considers all true targets in its own sample, including
        itself. Neighbors sort by squared distance, then local index. Valid
        distances are ``sqrt(distance_sq + eps)``.

        Missing slots have index zero, false validity, and the largest selected
        distance plus one. Rows with no valid targets use one. Empty samples are
        supported. Distances retain coordinate gradients; indices and validity
        are discrete.

    Examples:
        >>> import torch
        >>> from danling.tensors import NestedTensor
        >>> from oops import masked_knn
        >>> if torch.cuda.is_available():
        ...     values = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda")
        ...     points = NestedTensor([values], ragged_dims=(0,))
        ...     mask = NestedTensor([torch.ones(2, dtype=torch.bool, device="cuda")], ragged_dims=(0,))
        ...     distances, indices, valid = masked_knn(points, mask, 2, eps=0)
        ...     assert distances.concat.shape == (2, 2)
        ...     assert indices.concat[0].tolist() == [0, 1]
        ...     assert valid.concat.all().item()
    """

    if not isinstance(coordinates, NestedTensor) or not isinstance(mask, NestedTensor):
        raise TypeError("masked_knn coordinates and mask must both be NestedTensor")
    if type(num_neighbors) is not int:
        raise TypeError("num_neighbors must be an int")
    if not 1 <= num_neighbors <= 128:
        raise ValueError("num_neighbors must lie in [1, 128]")
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError("eps must be a real number")
    if is_compiling():
        torch._check(eps >= 0)
        torch._check(eps <= _FLOAT32_MAX)
    else:
        eps = _validated_eps(eps)

    coordinate_values, mask_values = _canonical_inputs(coordinates, mask)
    require_no_forward_mode("masked_knn", coordinate_values, mask_values)
    require_same_device("masked_knn", coordinate_values, mask_values)

    coordinate_cpu_offsets = coordinates.ragged_level_offsets(0, dtype=torch.int64)
    mask_cpu_offsets = mask.ragged_level_offsets(0, dtype=torch.int64)
    validation = validate_nested_masked_knn_metadata(
        coordinate_cpu_offsets,
        mask_cpu_offsets,
        coordinate_values.shape[0],
        mask_values.shape[0],
    )
    if validation is not None:
        if is_compiling() or not is_external_fake(validation):
            torch._check_tensor_all(validation)
        else:
            torch._assert_async(validation)

    device = coordinate_values.device
    offsets = coordinates.ragged_level_offsets(0, device=device, dtype=torch.int64)
    batch_indices = coordinates.packed_batch_indices(device=device, dtype=torch.int32)
    if not supports_segmented_masked_knn(
        coordinate_values,
        mask_values,
        offsets,
        batch_indices,
        num_neighbors,
    ):
        raise unsupported(
            "masked_knn",
            "requires canonical packed CUDA float16/bfloat16/float32 coordinates, a matching truthy mask, "
            "and launchable segmented metadata",
            coordinate_shape=tuple(coordinate_values.shape),
            coordinate_dtype=coordinate_values.dtype,
            mask_shape=tuple(mask_values.shape),
            mask_dtype=mask_values.dtype,
            device=coordinate_values.device,
            num_neighbors=num_neighbors,
        )
    distances, local_indices, valid = segmented_masked_knn(
        coordinate_values,
        mask_values,
        offsets,
        batch_indices,
        num_neighbors,
        eps,
    )
    return (
        coordinates.packed_with_static_tail(distances),
        coordinates.packed_with_static_tail(local_indices),
        coordinates.packed_with_static_tail(valid),
    )


def _canonical_inputs(coordinates: NestedTensor, mask: NestedTensor) -> tuple[Tensor, Tensor]:
    if not coordinates.batch_first or not mask.batch_first:
        raise ValueError("masked_knn NestedTensor batch must be the leading logical dimension")
    if len(coordinates) == 0 or len(mask) == 0:
        raise ValueError("masked_knn requires a non-empty NestedTensor wrapper batch")
    if len(coordinates) != len(mask):
        raise ValueError("masked_knn coordinates and mask batches must match")
    if coordinates.dim() != 3 or coordinates.ragged_dims != (0,) or coordinates.packed_dim_order != (0, 1):
        raise ValueError("masked_knn coordinate elements must use canonical (N_i, 3) topology")
    if mask.dim() != 2 or mask.ragged_dims != (0,) or mask.packed_dim_order != (0,):
        raise ValueError("masked_knn mask elements must use canonical (N_i,) topology")
    coordinate_values = coordinates.concat
    mask_values = mask.concat
    if coordinate_values.dim() != 2 or coordinate_values.shape[1] != 3:
        raise ValueError("masked_knn packed coordinates must have shape (sum N_i, 3)")
    if mask_values.dim() != 1 or mask_values.is_complex():
        raise ValueError("masked_knn packed mask must be one-dimensional and non-complex")
    return coordinate_values.contiguous(), mask_values.contiguous()


def _validated_eps(eps: object) -> float:

    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError("eps must be a real number")
    value = float(eps)
    if not math.isfinite(value) or value < 0 or value > _FLOAT32_MAX:
        raise ValueError("eps must be a finite non-negative float no larger than the float32 maximum")
    return value
