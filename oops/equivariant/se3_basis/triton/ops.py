from __future__ import annotations

import torch
from torch import Tensor

from oops.runtime import STANDARD_FLOAT_DTYPES, cuda_device_context, is_compiling, is_external_fake, needs_grad

from ...spherical_harmonics.triton.backend import spherical_harmonics_backward
from . import kernel
from .kernel import _SE3_BASIS_NORM_EPS
from .profiles import SE3_BASIS_L1_SCHEDULE
from .schedule import SE3BasisPlan, SE3BasisProblem

_SE3_BASIS_HARMONIC_LMAX = 2
_SE3_BASIS_HARMONICS = (_SE3_BASIS_HARMONIC_LMAX + 1) * (_SE3_BASIS_HARMONIC_LMAX + 1)


def supports_se3_basis_l1(
    relative_positions: Tensor,
    max_degree: int,
    *,
    use_pad_trick: bool,
    fully_fused: bool,
    cache: bool = True,
) -> bool:

    supported = (
        relative_positions.is_cuda
        and relative_positions.is_floating_point()
        and relative_positions.dtype in STANDARD_FLOAT_DTYPES
        and relative_positions.dim() == 2
        and relative_positions.shape[-1] == 3
        and relative_positions.is_contiguous()
        and max_degree == 1
        and relative_positions.shape[0] > 0
    )
    return supported and SE3_BASIS_L1_SCHEDULE.supports(
        SE3BasisProblem(
            relative_positions.shape[0],
            max_degree,
            bool(use_pad_trick),
            bool(fully_fused),
            relative_positions.dtype,
        ),
        cache=cache,
    )


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


def _se3_basis_l1_forward_impl(relative_positions: Tensor, plan: SE3BasisPlan) -> list[Tensor]:
    with cuda_device_context(relative_positions.device):
        relative_positions = relative_positions.resolve_neg().contiguous()
        edges = int(relative_positions.shape[0])
        device, dtype = relative_positions.device, relative_positions.dtype
        basis = [torch.empty((edges, *spec.shape), device=device, dtype=dtype) for spec in plan.outputs]
        kernel._se3_basis_l1_fill[plan.fill_grid](
            relative_positions,
            tuple(basis),
            edges,
            tuple(spec.per_edge for spec in plan.outputs),
            plan.problem.use_pad_trick,
            plan.config.block_edges,
            num_warps=plan.config.num_warps,
        )
        return basis


@torch.library.custom_op("ooperators::_se3_basis_l1_forward", mutates_args=(), device_types="cuda")
def _se3_basis_l1_forward(relative_positions: Tensor, use_pad_trick: bool, fully_fused: bool) -> list[Tensor]:
    problem = SE3BasisProblem(relative_positions.shape[0], 1, use_pad_trick, fully_fused, relative_positions.dtype)
    return _se3_basis_l1_forward_impl(relative_positions, SE3_BASIS_L1_SCHEDULE.resolve(problem))


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
    with cuda_device_context(relative_positions.device):
        relative_positions = relative_positions.resolve_neg().contiguous()
        grad_basis = [None if grad is None else grad.resolve_neg() for grad in grad_basis]
        edges = int(relative_positions.shape[0])
        grad_harmonics = torch.empty(
            (edges, _SE3_BASIS_HARMONICS), device=relative_positions.device, dtype=torch.float32
        )
        problem = SE3BasisProblem(relative_positions.shape[0], 1, use_pad_trick, fully_fused, relative_positions.dtype)
        plan = SE3_BASIS_L1_SCHEDULE.resolve(problem)
        present = tuple(grad is not None for grad in grad_basis)
        inputs = tuple(relative_positions if grad is None else grad.contiguous() for grad in grad_basis)
        kernel._se3_basis_l1_grad_harmonics[plan.grad_harmonics_grid](
            inputs,
            grad_harmonics,
            edges,
            tuple(spec.per_edge for spec in plan.outputs),
            present,
            plan.problem.use_pad_trick,
            plan.config.grad_block_edges,
            num_warps=plan.config.num_warps,
        )
        return spherical_harmonics_backward(
            grad_harmonics,
            relative_positions,
            _SE3_BASIS_HARMONIC_LMAX,
            True,
            "integral",
            _SE3_BASIS_NORM_EPS.value,
            plan.spherical_plan,
        )


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
