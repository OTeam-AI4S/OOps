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
        A loss tensor shaped like ``labels``. CUDA half autocast returns float32;
        otherwise the result has the logits dtype.

    Raises:
        ValueError: Devices differ or class-index label shapes disagree.
        OOpsUnsupportedError: Input device, dtype, shape, or class count is unsupported,
            probability labels are supplied, or logits carry a forward-mode tangent.

    Notes:
        First-order reverse mode differentiates logits; labels are non-differentiable.
        Label values are not validated. Out-of-range labels are outside the numerical
        contract. There is no ``ignore_index``.

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
