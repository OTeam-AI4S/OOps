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

import torch
from torch import Tensor

from oops.runtime import is_compiling, is_external_fake, needs_grad

from .generic import _run_se3_basis_l1_forward, _se3_basis_l1_backward_impl, _se3_basis_l1_forward_impl
from .profiles import SE3_BASIS_L1_SCHEDULE
from .schedule import SE3BasisProblem


def triton_se3_basis_l1(
    relative_positions: Tensor,
    *,
    use_pad_trick: bool,
    fully_fused: bool,
    compiling: bool | None = None,
    external_fake: bool | None = None,
) -> dict[str, Tensor]:

    use_pad_trick, fully_fused = bool(use_pad_trick), bool(fully_fused)
    problem = SE3BasisProblem(
        relative_positions.shape[0],
        1,
        use_pad_trick,
        fully_fused,
        relative_positions.dtype,
    )
    if compiling is None:
        compiling = is_compiling()
    if external_fake is None:
        external_fake = False if compiling else is_external_fake(relative_positions)
    plan = SE3_BASIS_L1_SCHEDULE.resolve(problem, cache=not external_fake)
    if needs_grad(relative_positions) or compiling or external_fake:
        tensors = _se3_basis_l1_forward(relative_positions, use_pad_trick, fully_fused)
    else:
        tensors = _se3_basis_l1_forward_impl(relative_positions, plan)
    return dict(zip(plan.keys, tensors, strict=True))


@torch.library.custom_op("ooperators::_se3_basis_l1_forward", mutates_args=(), device_types="cuda")
def _se3_basis_l1_forward(relative_positions: Tensor, use_pad_trick: bool, fully_fused: bool) -> list[Tensor]:
    return _run_se3_basis_l1_forward(relative_positions, use_pad_trick, fully_fused)


@torch.library.register_fake(_se3_basis_l1_forward)
def _se3_basis_l1_forward_fake(relative_positions: Tensor, use_pad_trick: bool, fully_fused: bool) -> list[Tensor]:
    problem = SE3BasisProblem(relative_positions.shape[0], 1, use_pad_trick, fully_fused, relative_positions.dtype)
    plan = SE3_BASIS_L1_SCHEDULE.resolve(problem, cache=False)
    return [
        torch.empty(
            (relative_positions.shape[0], *spec.shape),
            device=relative_positions.device,
            dtype=relative_positions.dtype,
        )
        for spec in plan.outputs
    ]


@torch.library.custom_op("ooperators::_se3_basis_l1_backward", mutates_args=(), device_types="cuda")
def _se3_basis_l1_backward(
    grad_basis: list[Tensor | None],
    relative_positions: Tensor,
    use_pad_trick: bool,
    fully_fused: bool,
) -> Tensor:
    return _se3_basis_l1_backward_impl(grad_basis, relative_positions, use_pad_trick, fully_fused)


@torch.library.register_fake(_se3_basis_l1_backward)
def _se3_basis_l1_backward_fake(
    grad_basis: list[Tensor | None],
    relative_positions: Tensor,
    use_pad_trick: bool,
    fully_fused: bool,
) -> Tensor:
    del grad_basis, use_pad_trick, fully_fused
    return torch.empty_like(relative_positions)


def _se3_basis_l1_forward_setup_context(
    ctx,
    inputs: tuple[Tensor, bool, bool],
    output: list[Tensor],
) -> None:
    relative_positions, use_pad_trick, fully_fused = inputs
    del output
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(relative_positions)
    ctx.use_pad_trick = use_pad_trick
    ctx.fully_fused = fully_fused


def _se3_basis_l1_forward_autograd_backward(ctx, grad_basis: list[Tensor | None]) -> tuple[Tensor, None, None]:
    (relative_positions,) = ctx.saved_tensors
    grad_relative_positions = _se3_basis_l1_backward(
        list(grad_basis),
        relative_positions,
        ctx.use_pad_trick,
        ctx.fully_fused,
    )
    return grad_relative_positions, None, None


torch.library.register_autograd(
    _se3_basis_l1_forward,
    _se3_basis_l1_forward_autograd_backward,
    setup_context=_se3_basis_l1_forward_setup_context,
)
