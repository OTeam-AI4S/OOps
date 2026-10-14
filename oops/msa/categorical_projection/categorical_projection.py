from __future__ import annotations

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from oops.runtime import (
    STANDARD_FLOAT_DTYPES,
    empty_with_dependencies,
    is_compiling,
    is_external_fake,
    prepare_cuda_autocast_operands,
    require_no_forward_mode,
    require_same_device,
    unsupported,
)

from .nested import validate_nested_categorical_msa_projection_metadata
from .precision import resolve_categorical_msa_projection_native_matmul_state
from .triton import (
    prepare_eager_categorical_msa_projection,
    supports_prepared_categorical_msa_projection,
    triton_categorical_msa_projection,
)


def categorical_msa_projection(
    tokens: Tensor | NestedTensor,
    continuous: Tensor | NestedTensor,
    weight: Tensor,
) -> Tensor | NestedTensor:
    r"""
    Project categorical MSA tokens and two continuous features.

    The result equals a bias-free linear projection of concatenated one-hot tokens
    and continuous features.

    Args:
        tokens: Contiguous int64 token indices ``(*shape,)`` or canonical DanLing
            ``NestedTensor`` elements ``(M_i, N_i)``. Indices lie in ``[0, V)``.
        continuous: Contiguous floating features ``(*shape, 2)`` or matching nested
            ``(M_i, N_i, 2)`` elements, on the token device.
        weight: Contiguous projection matrix ``(C, V + 2)`` on the input device,
            with positive vocabulary size ``V``.

    Returns:
        Features ``(*shape, C)`` on the input device in the prepared projection
        dtype. Nested results preserve the token topology with channel width ``C``.

    Raises:
        ValueError: Shapes, devices, representations, or nested topologies disagree.
        OOpsUnsupportedError: Index dtype, floating dtype, layout, CUDA launch
            domain, or forward-mode differentiation is unsupported, or native
            matmul precision changes during an execution.
        RuntimeError: A launched CUDA kernel encounters an out-of-range token;
            the device assertion may be reported asynchronously.

    Notes:
        Nonempty calls require CUDA int64 tokens and float16, bfloat16, or float32
        continuous features and weights sharing one prepared dtype. CUDA FP16/BF16
        autocast selects the floating projection/output dtype. Empty outputs retain
        shape, dtype, and zero autograd dependencies, including outside the
        nonempty CUDA domain.
        Nested MSA ragged axes are declared explicitly and use canonical packed
        order.

        First-order reverse mode and ``torch.compile(fullgraph=True)`` are
        supported; forward-mode AD is unsupported. PyTorch's scoped FP32 and
        relevant FP16/BF16 native matmul controls must remain unchanged throughout
        forward/backward, including cached compiled backward. Unsupported calls
        have no reference fallback.

    Examples:
        >>> import torch
        >>> from oops import categorical_msa_projection
        >>> if torch.cuda.is_available():
        ...     tokens = torch.tensor([[0, 1], [1, 0]], device="cuda")
        ...     continuous = torch.zeros(2, 2, 2, device="cuda")
        ...     weight = torch.ones(8, 4, device="cuda")
        ...     result = categorical_msa_projection(tokens, continuous, weight)
        ...     assert result.shape == (2, 2, 8)
    """

    if isinstance(tokens, NestedTensor):
        if not isinstance(continuous, NestedTensor):
            raise ValueError(
                "categorical_msa_projection tokens and continuous must both be NestedTensor or both be dense"
            )
        return _categorical_msa_projection_nested(tokens, continuous, weight)
    if isinstance(continuous, NestedTensor):
        raise ValueError("categorical_msa_projection tokens and continuous must both be NestedTensor or both be dense")

    _check_shapes(tokens, continuous, weight)
    require_no_forward_mode("categorical_msa_projection", continuous, weight)
    require_same_device("categorical_msa_projection", tokens, continuous, weight)

    continuous, weight = prepare_cuda_autocast_operands(continuous, (continuous, weight))
    if tokens.dtype is not torch.int64:
        raise unsupported(
            "categorical_msa_projection",
            "requires int64 token indices",
            token_dtype=tokens.dtype,
        )
    if continuous.dtype not in STANDARD_FLOAT_DTYPES or weight.dtype is not continuous.dtype:
        raise unsupported(
            "categorical_msa_projection",
            "requires float16/bfloat16/float32 continuous features and weights in one prepared dtype",
            continuous_dtype=continuous.dtype,
            weight_dtype=weight.dtype,
        )

    output_shape = (*tokens.shape, weight.shape[0])
    if tokens.numel() == 0 or weight.shape[0] == 0:
        output = torch.empty(output_shape, device=continuous.device, dtype=continuous.dtype)
        return empty_with_dependencies(output, continuous, weight)

    compiling = is_compiling()
    fake = False if compiling else is_external_fake(tokens)
    plan = None
    if compiling or fake:
        supported = supports_prepared_categorical_msa_projection(tokens, continuous, weight)
    else:
        plan = prepare_eager_categorical_msa_projection(tokens, continuous, weight)
        supported = plan is not None
    if not supported:
        raise unsupported(
            "categorical_msa_projection",
            "requires contiguous CUDA operands with a launchable row/channel grid",
            token_shape=tuple(tokens.shape),
            continuous_shape=tuple(continuous.shape),
            weight_shape=tuple(weight.shape),
            device=tokens.device,
        )
    native_matmul_state = resolve_categorical_msa_projection_native_matmul_state(continuous.dtype)
    return triton_categorical_msa_projection(tokens, continuous, weight, native_matmul_state, plan)


def _categorical_msa_projection_nested(
    tokens: NestedTensor,
    continuous: NestedTensor,
    weight: Tensor,
) -> NestedTensor:
    if not tokens.batch_first or not continuous.batch_first:
        raise ValueError("categorical_msa_projection NestedTensor batch must be the leading logical dimension")
    if len(tokens) != len(continuous):
        raise ValueError("categorical_msa_projection tokens and continuous NestedTensor batches must match")
    if len(tokens) == 0:
        raise unsupported(
            "categorical_msa_projection",
            "does not serve an empty NestedTensor wrapper batch without retained element rank",
        )
    if tokens.dim() != 3:
        raise ValueError(
            "categorical_msa_projection nested tokens elements must have shape (M_i, N_i); "
            f"got logical rank {tokens.dim()}"
        )
    if continuous.dim() != 4:
        raise ValueError(
            "categorical_msa_projection nested continuous elements must have shape (M_i, N_i, 2); "
            f"got logical rank {continuous.dim()}"
        )
    if tokens.ragged_dims != (0, 1) or tokens.packed_dim_order != (0, 1):
        raise unsupported(
            "categorical_msa_projection",
            "requires explicit canonical (M_i, N_i) nested token packing",
            ragged_dims=tokens.ragged_dims,
            packed_dim_order=tokens.packed_dim_order,
        )
    if continuous.ragged_dims != (0, 1) or continuous.packed_dim_order != (0, 1, 2):
        raise unsupported(
            "categorical_msa_projection",
            "requires explicit canonical (M_i, N_i, 2) nested continuous packing",
            ragged_dims=continuous.ragged_dims,
            packed_dim_order=continuous.packed_dim_order,
        )

    token_values = tokens.concat
    continuous_values = continuous.concat
    if token_values.dim() != 1:
        raise ValueError("categorical_msa_projection nested token packed values must have shape (sum M_i*N_i,)")
    if continuous_values.dim() != 2 or continuous_values.shape[-1] != 2:
        raise ValueError("categorical_msa_projection nested continuous packed values must have shape (sum M_i*N_i, 2)")

    validation = validate_nested_categorical_msa_projection_metadata(
        tokens.element_sizes(),
        tokens.ragged_level_offsets(0, dtype=torch.int64),
        tokens.ragged_level_offsets(1, dtype=torch.int64),
        tokens.packed_offsets(dtype=torch.int64),
        token_values.shape[0],
        continuous.element_sizes(),
        continuous.ragged_level_offsets(0, dtype=torch.int64),
        continuous.ragged_level_offsets(1, dtype=torch.int64),
        continuous.packed_offsets(dtype=torch.int64),
        continuous_values.shape[0],
    )
    if is_compiling():
        torch._check_tensor_all(
            validation,
            lambda: "categorical_msa_projection nested metadata validation must execute",
        )

    output_values = categorical_msa_projection(token_values, continuous_values, weight)
    return continuous.packed_with_static_tail(output_values)


def _check_shapes(tokens: Tensor, continuous: Tensor, weight: Tensor) -> None:
    if tokens.dim() < 1:
        raise ValueError("tokens must have at least one dimension")
    if continuous.shape != (*tokens.shape, 2):
        raise ValueError(f"continuous must have shape {(*tokens.shape, 2)}; got {tuple(continuous.shape)}")
    if weight.dim() != 2 or weight.shape[1] <= 2:
        raise ValueError("weight must have shape (C, V + 2) with V > 0")
