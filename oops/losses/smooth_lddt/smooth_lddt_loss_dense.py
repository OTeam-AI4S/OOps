from __future__ import annotations

from torch import SymInt, Tensor

from oops.runtime import require_no_forward_mode, require_same_device, unsupported

from .triton import supports_smooth_lddt_loss_dense, triton_smooth_lddt_loss_dense


def smooth_lddt_loss_dense(
    predicted_coordinates: Tensor,
    true_coordinates: Tensor,
    is_nucleotide: Tensor,
    coordinates_mask: Tensor,
    *,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
    true_distances: Tensor | None = None,
) -> Tensor:
    r"""Compute a masked smooth-LDDT loss over dense atom pairs.

    Args:
        predicted_coordinates: Contiguous CUDA coordinates of shape ``(B, A, 3)``
            and dtype float16, bfloat16, or float32.
        true_coordinates: Contiguous coordinates with the same shape and device.
            Their supported storage dtype is independent of the prediction dtype.
        is_nucleotide: Contiguous boolean or float16/bfloat16/float32 indicators of shape
            ``(B, A)``. Nonzero values select ``nucleic_acid_cutoff`` for each row.
        coordinates_mask: Contiguous boolean or float16/bfloat16/float32 atom weights of shape
            ``(B, A)``, on the coordinate device.
        nucleic_acid_cutoff: Ground-truth distance cutoff for nucleotide rows.
        other_cutoff: Ground-truth distance cutoff for other rows.
        multiplicity: Positive integer sample multiplicity. Indicators and masks
            may have batch size ``B // multiplicity`` and are repeated per sample,
            or may already have batch size ``B``. ``B`` must be divisible by it.
        true_distances: Unsupported; must be ``None``.

    Returns:
        A scalar float32 loss on the coordinate device. For each batch item, the
        score is the weighted mean of ``sigmoid(t - abs(d_pred - d_true))`` over
        thresholds ``t = 0.5, 1, 2, 4`` and valid pairs. Diagonal pairs and pairs
        at or beyond the row's ground-truth cutoff are excluded. Pair weights are
        ``mask_i * mask_j``; the denominator is their sum clamped to at least one.
        The loss is one minus the mean batch score. No valid pairs give loss one.

    Raises:
        ValueError: If devices differ, prediction rank is invalid, or multiplicity
            and metadata batch sizes are incompatible.
        OOpsUnsupportedError: If precomputed distances are supplied, inputs are
            outside the supported dtype, shape, layout, or device domain,
            forward-mode AD is requested, or strict determinism is unsupported
            for the supplied shape.

    Notes:
        Coordinate, score, and reduction arithmetic uses float32. CUDA autocast
        does not narrow caller-owned inputs or change the output dtype.
        First-order reverse-mode gradients are supported for prediction, truth,
        and floating masks and return in each input's storage dtype. Boolean masks
        and the nonzero nucleotide indicator are not differentiable. Forward-mode
        and higher-order AD are unsupported.
        Results may vary in their last bits between calls. For shapes with
        non-deterministic results, strict deterministic mode rejects the call;
        ``warn_only=True`` warns and continues.

    Examples:
        >>> import torch
        >>> from oops import smooth_lddt_loss_dense
        >>> if torch.cuda.is_available():
        ...     coordinates = torch.zeros(1, 2, 3, device="cuda")
        ...     nucleotide = torch.zeros(1, 2, dtype=torch.bool, device="cuda")
        ...     mask = torch.ones(1, 2, device="cuda")
        ...     loss = smooth_lddt_loss_dense(coordinates, coordinates, nucleotide, mask)
        ...     assert loss.shape == () and loss.dtype == torch.float32
    """

    require_no_forward_mode("smooth_lddt_loss_dense", predicted_coordinates, true_coordinates, coordinates_mask)
    require_same_device(
        "smooth_lddt_loss_dense",
        predicted_coordinates,
        true_coordinates,
        is_nucleotide,
        coordinates_mask,
        true_distances,
    )
    if true_distances is not None:
        raise unsupported(
            "smooth_lddt_loss_dense",
            "computes the ground-truth distances inside the kernel and cannot consume precomputed ones",
            true_distances_shape=tuple(true_distances.shape),
        )
    if predicted_coordinates.dim() != 3:
        raise ValueError(
            f"predicted_coordinates must have shape (batch, atoms, 3), got {tuple(predicted_coordinates.shape)}"
        )
    if not isinstance(multiplicity, int) or isinstance(multiplicity, bool) or multiplicity <= 0:
        raise ValueError("multiplicity must be a positive integer")

    batch = predicted_coordinates.shape[0]
    is_nucleotide = _expand_multiplicity(
        is_nucleotide,
        multiplicity=multiplicity,
        total_batch=batch,
        name="is_nucleotide",
    )
    coordinates_mask = _expand_multiplicity(
        coordinates_mask,
        multiplicity=multiplicity,
        total_batch=batch,
        name="coordinates_mask",
    )
    if not supports_smooth_lddt_loss_dense(
        predicted_coordinates, true_coordinates, is_nucleotide, coordinates_mask, None
    ):
        raise unsupported(
            "smooth_lddt_loss_dense",
            "requires contiguous CUDA float16/bfloat16/float32 coordinates and boolean or "
            "float16/bfloat16/float32 indicators/masks within CUDA launch limits",
            dtype=predicted_coordinates.dtype,
            device=predicted_coordinates.device,
            predicted_coordinates_shape=tuple(predicted_coordinates.shape),
            true_coordinates_shape=tuple(true_coordinates.shape),
            is_nucleotide_shape=tuple(is_nucleotide.shape),
            coordinates_mask_shape=tuple(coordinates_mask.shape),
        )
    return triton_smooth_lddt_loss_dense(
        predicted_coordinates,
        true_coordinates,
        is_nucleotide,
        coordinates_mask,
        nucleic_acid_cutoff=nucleic_acid_cutoff,
        other_cutoff=other_cutoff,
    )


def _expand_multiplicity(
    values: Tensor,
    *,
    multiplicity: int,
    total_batch: int | SymInt,
    name: str,
) -> Tensor:
    if multiplicity == 1:
        return values
    if total_batch % multiplicity != 0:
        raise ValueError(f"expanded batch {total_batch} is not divisible by multiplicity {multiplicity}")
    base_batch = total_batch // multiplicity
    if values.shape[0] == base_batch:
        return values.repeat_interleave(multiplicity, 0)
    if values.shape[0] == total_batch:
        return values
    raise ValueError(f"{name} batch must be {base_batch} or {total_batch}, got {values.shape[0]}")
