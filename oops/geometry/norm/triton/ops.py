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
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from oops.runtime import functorch_transforms_active, is_compiling, is_external_fake, needs_grad, unsupported

from .generic import _run_backward, _run_forward


def vector_norm(vectors: Tensor, eps: float) -> Tensor:
    if needs_grad(vectors):
        if (
            not is_compiling()
            and type(vectors) in (Tensor, torch.nn.Parameter)
            and not is_in_torch_dispatch_mode()
            and not functorch_transforms_active()
        ):
            return _VectorNorm.apply(vectors, eps)
        return _forward(vectors, eps)
    if is_compiling() or is_external_fake(vectors):
        return _forward(vectors, eps)
    return _run_forward(vectors, eps)


@torch.library.custom_op("ooperators::_smoothed_vector_norm_forward", mutates_args=())
def _forward(vectors: Tensor, eps: float) -> Tensor:
    return _run_forward(vectors, eps)


@_forward.register_fake
def _forward_fake(vectors: Tensor, eps: float) -> Tensor:
    del eps
    return vectors.new_empty(vectors.shape[:-1])


def _setup_context(ctx, inputs, output):
    del output
    vectors, ctx.eps = inputs
    ctx.save_for_backward(vectors)
    ctx.plain = False


def _autograd_backward(ctx, gradient):
    if torch.is_grad_enabled():
        raise unsupported("smoothed_vector_norm", "supports first-order reverse-mode differentiation only")
    backward = _run_backward if ctx.plain else _backward
    return backward(ctx.saved_tensors[0], gradient, ctx.eps), None


torch.library.register_autograd(_forward, _autograd_backward, setup_context=_setup_context)


@torch.library.custom_op("ooperators::_smoothed_vector_norm_backward", mutates_args=())
def _backward(vectors: Tensor, upstream: Tensor, eps: float) -> Tensor:
    return _run_backward(vectors, upstream, eps)


@_backward.register_fake
def _backward_fake(vectors: Tensor, upstream: Tensor, eps: float) -> Tensor:
    del upstream, eps
    return vectors.new_empty(vectors.shape)


class _VectorNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vectors, eps):
        output = _run_forward(vectors, eps)
        _setup_context(ctx, (vectors, eps), output)
        ctx.plain = True
        return output

    backward = staticmethod(_autograd_backward)
