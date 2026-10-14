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

from torch import Tensor

from oops.runtime import require_no_forward_mode, require_same_device, unsupported

from .triton import supports_cross_entropy, triton_cross_entropy


def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    r"""Compute unreduced cross entropy with classes on the final axis.

    Args:
        logits: Nonempty CUDA float16, bfloat16, or float32 tensor ``(..., C)`` with
            ``1 <= C <= 256``.
        labels: Integer class indices of shape ``logits.shape[:-1]`` on the logits device.
            Labels must lie in ``[0, C)``; probability and one-hot labels are unsupported.

    Returns:
        A loss tensor shaped like ``labels``.

    Raises:
        ValueError: Devices differ or class-index label shapes disagree.
        OOpsUnsupportedError: Input device, dtype, shape, or class count is unsupported,
            probability labels are supplied, or logits carry a forward-mode tangent.

    Notes:
        There is no ``ignore_index``; every label must be a valid class index.

    Examples:
        >>> import torch
        >>> from oops import cross_entropy
        >>> if torch.cuda.is_available():
        ...     logits = torch.zeros(2, 4, device="cuda")
        ...     labels = torch.tensor([0, 3], device="cuda")
        ...     loss = cross_entropy(logits, labels)
        ...     assert loss.shape == labels.shape and loss.dtype == torch.float32
    """

    require_no_forward_mode("cross_entropy", logits)
    require_same_device("cross_entropy", logits, labels)

    if labels.shape != logits.shape[:-1]:
        if labels.shape == logits.shape:
            raise unsupported(
                "cross_entropy",
                "requires integer class indices shaped logits.shape[:-1], not probability labels",
                labels_shape=tuple(labels.shape),
                logits_shape=tuple(logits.shape),
            )
        raise ValueError("labels must have shape logits.shape[:-1]")

    if not supports_cross_entropy(logits, labels):
        raise unsupported(
            "cross_entropy",
            "requires CUDA float16/bfloat16/float32 logits with integer labels, at most 256 "
            "classes, and a launchable row grid",
            logits_dtype=logits.dtype,
            labels_dtype=labels.dtype,
            device=logits.device,
            logits_shape=tuple(logits.shape),
        )
    return triton_cross_entropy(logits, labels)
