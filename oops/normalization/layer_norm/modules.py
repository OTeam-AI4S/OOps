from __future__ import annotations

from torch import Tensor, nn

from oops._errors import OOpsUnsupportedError

from .layer_norm import layer_norm


class LayerNorm(nn.LayerNorm):
    r"""Normalize activations with OOps kernels and a native PyTorch fallback.

    Args:
        normalized_shape (int | list[int] | tuple[int, ...]): Normalized dimensions,
            following ``torch.nn.LayerNorm``.
        eps (float): Stabilizer added to the variance.
        elementwise_affine (bool): Whether to learn a scale and optional bias for each feature.
        bias (bool): Whether to learn the affine bias when ``elementwise_affine`` is true.
        device (torch.device | str | int | None): Optional device for affine parameters.
        dtype (torch.dtype | None): Optional affine parameter dtype.

    Raises:
        RuntimeError: Native PyTorch validation fails for an invalid normalization shape,
            parameter dtype, or parameter device.
        ValueError: Operands violate the functional API's device or topology contract.

    Notes:
        Calling the module returns normalized values with the input shape. The selected
        OOps or native PyTorch operation determines dtype and differentiation behavior.

        Constructor and native behavior follow ``torch.nn.LayerNorm``. Served CUDA calls use
        ``layer_norm``; an ``OOpsUnsupportedError`` selects native LayerNorm, including on
        CPU and for float64. Call the functional API when unsupported inputs should raise.

    Examples:
        >>> import torch
        >>> from oops import LayerNorm
        >>> norm = LayerNorm(4)
        >>> norm(torch.zeros(2, 4)).shape
        torch.Size([2, 4])
    """

    def forward(self, input: Tensor) -> Tensor:
        r"""Normalize ``input`` using the module's affine parameters.

        Args:
            input: Activation tensor compatible with the configured normalized dimensions.

        Returns:
            Normalized values with the input shape; see the class contract for fallback behavior.
        """
        try:
            return layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        except OOpsUnsupportedError:
            return super().forward(input)
