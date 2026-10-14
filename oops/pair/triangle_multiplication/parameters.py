from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class TriangleMultiplicationWeights(NamedTuple):

    input_norm_weight: Tensor
    input_norm_bias: Tensor
    left_value_weight: Tensor
    left_gate_weight: Tensor
    right_value_weight: Tensor
    right_gate_weight: Tensor
    output_norm_weight: Tensor
    output_norm_bias: Tensor
    output_weight: Tensor
    output_gate_weight: Tensor

    @property
    def projections(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        return (
            self.left_value_weight,
            self.left_gate_weight,
            self.right_value_weight,
            self.right_gate_weight,
            self.output_weight,
            self.output_gate_weight,
        )

    def with_projections(self, projections: tuple[Tensor, ...]) -> TriangleMultiplicationWeights:

        left_value, left_gate, right_value, right_gate, output, output_gate = projections
        return self._replace(
            left_value_weight=left_value,
            left_gate_weight=left_gate,
            right_value_weight=right_value,
            right_gate_weight=right_gate,
            output_weight=output,
            output_gate_weight=output_gate,
        )


class TriangleMultiplicationBiases(NamedTuple):

    left_value_bias: Tensor | None
    left_gate_bias: Tensor | None
    right_value_bias: Tensor | None
    right_gate_bias: Tensor | None
    output_bias: Tensor | None
    output_gate_bias: Tensor | None

    @property
    def present(self) -> tuple[Tensor, ...]:
        return tuple(bias for bias in self if bias is not None)

    def replace_present(self, replacements: tuple[Tensor, ...]) -> TriangleMultiplicationBiases:

        replaced = iter(replacements)
        return TriangleMultiplicationBiases(*(None if bias is None else next(replaced) for bias in self))
