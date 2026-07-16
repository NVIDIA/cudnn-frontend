"""High-level, torch-tensor dispatch for the FROST norm kernels.

This is the ergonomic entry point used by tests and by the (optional) cuDNN
graph engine. It takes torch tensors in their natural layout, derives the
:class:`RowwiseSpec` / :class:`BatchNormSpec`, reshapes to the canonical view
each kernel expects, and launches.

All five variants are covered:
  - LayerNorm / RMSNorm / InstanceNorm / GroupNorm -> one shared row-wise kernel
  - BatchNorm -> its own across-batch kernel
"""

from __future__ import annotations

from typing import Optional, Sequence

from .config import (
    NormVariant,
    ROWWISE_VARIANTS,
    batchnorm_spec,
    choose_block_threads,
    choose_cpasync_config,
    rowwise_spec,
    vector_width,
)
from .dtypes import DTYPE_BYTES, torch_dtype_to_str


def _as_variant(v) -> NormVariant:
    return v if isinstance(v, NormVariant) else NormVariant(v)


def _rowwise_use_vec(impl: str, spec, x) -> tuple:
    """Return (use_vec, V). Vectorized path needs impl='vec'/'auto' and M % V == 0."""
    if impl not in ("vec", "auto"):
        return False, 1
    V = vector_width(DTYPE_BYTES[torch_dtype_to_str(x.dtype)])
    if V > 1 and spec.M % V == 0:
        return True, V
    return False, 1


def norm_forward(
    variant,
    x,
    gamma=None,
    beta=None,
    *,
    eps: float = 1e-5,
    normalized_shape: Optional[Sequence[int]] = None,
    num_groups: Optional[int] = None,
    momentum: float = 0.1,
    training: bool = True,
    running_mean=None,
    running_var=None,
    impl: str = "auto",
):
    """Forward pass for any of the five norm variants.

    Returns ``(y, mean, rstd)``. ``y`` has the input's shape; ``mean``/``rstd``
    are fp32 per-group statistics (flat, length = number of groups / channels).
    For RMSNorm ``mean`` is a zero placeholder (no centering).

    ``impl`` selects the kernel: ``"scalar"``, ``"vec"`` (128-bit vectorized), or
    ``"auto"`` (vectorized when the shape allows, else scalar).
    """
    import torch

    variant = _as_variant(variant)
    x = x.contiguous()

    if variant in ROWWISE_VARIANTS:
        spec = rowwise_spec(
            variant, x.shape, normalized_shape=normalized_shape, num_groups=num_groups
        )
        if gamma is None:
            gamma = torch.ones(spec.gamma_len, dtype=x.dtype, device=x.device)
        xv = x.reshape(spec.R, spec.M)
        V = vector_width(DTYPE_BYTES[torch_dtype_to_str(x.dtype)])

        if impl in ("cpasync", "tma"):
            ok, bt = choose_cpasync_config(spec.M, V, DTYPE_BYTES[torch_dtype_to_str(x.dtype)])
            if ok:
                if impl == "cpasync":
                    from cudnn.norm.fprop.frost.rowwise_cpasync import rowwise_forward_cpasync

                    y2, mean, rstd = rowwise_forward_cpasync(
                        spec, xv, gamma, beta, eps=eps, V=V, block_threads=bt
                    )
                else:
                    from cudnn.norm.fprop.frost.rowwise_tma import rowwise_forward_tma

                    y2, mean, rstd = rowwise_forward_tma(
                        spec, xv, gamma, beta, eps=eps, V=V, block_threads=bt
                    )
                return y2.reshape(x.shape), mean, rstd
            impl = "auto"  # fall through to vec/scalar

        use_vec, V = _rowwise_use_vec(impl, spec, x)
        if use_vec:
            from cudnn.norm.fprop.frost.rowwise_vec import rowwise_forward_vec

            bt = choose_block_threads(spec.M // V)
            y2, mean, rstd = rowwise_forward_vec(spec, xv, gamma, beta, eps=eps, V=V, block_threads=bt)
        else:
            from cudnn.norm.fprop.frost.rowwise import rowwise_forward

            bt = choose_block_threads(spec.M)
            y2, mean, rstd = rowwise_forward(spec, xv, gamma, beta, eps=eps, block_threads=bt)
        return y2.reshape(x.shape), mean, rstd

    if variant == NormVariant.BATCH_NORM:
        from cudnn.norm.fprop.frost.batchnorm import batchnorm_forward

        spec = batchnorm_spec(x.shape)
        if gamma is None:
            gamma = torch.ones(spec.C, dtype=x.dtype, device=x.device)
        xv = x.reshape(spec.N, spec.C, spec.S)
        bt = choose_block_threads(spec.count)
        y3, sm, sr = batchnorm_forward(
            spec, xv, gamma, beta, eps=eps, momentum=momentum, training=training,
            running_mean=running_mean, running_var=running_var, block_threads=bt,
        )
        return y3.reshape(x.shape), sm, sr

    raise ValueError(f"unknown norm variant {variant}")


def norm_backward(
    variant,
    dy,
    x,
    gamma,
    mean,
    rstd,
    *,
    normalized_shape: Optional[Sequence[int]] = None,
    num_groups: Optional[int] = None,
    has_beta: bool = True,
    impl: str = "auto",
):
    """Backward pass for any of the five norm variants.

    ``mean``/``rstd`` are the fp32 statistics returned by :func:`norm_forward`.
    Returns ``(dx, dgamma, dbeta)`` (``dbeta`` is ``None`` when ``has_beta`` is
    False). ``dx`` has the input's shape; ``dgamma``/``dbeta`` are fp32.

    ``impl`` selects the kernel: ``"scalar"``, ``"vec"``, or ``"auto"``.
    """
    variant = _as_variant(variant)
    x = x.contiguous()
    dy = dy.contiguous()

    if variant in ROWWISE_VARIANTS:
        spec = rowwise_spec(
            variant, x.shape, normalized_shape=normalized_shape, num_groups=num_groups
        )
        xv = x.reshape(spec.R, spec.M)
        dyv = dy.reshape(spec.R, spec.M)
        use_vec, V = _rowwise_use_vec(impl, spec, x)
        if use_vec:
            from cudnn.norm.bprop.frost.rowwise_vec import rowwise_backward_vec

            bt = choose_block_threads(spec.M // V)
            dx, dgamma, dbeta = rowwise_backward_vec(
                spec, dyv, xv, gamma, mean, rstd, has_beta=has_beta, V=V, block_threads=bt
            )
        else:
            from cudnn.norm.bprop.frost.rowwise import rowwise_backward

            bt = choose_block_threads(spec.M)
            dx, dgamma, dbeta = rowwise_backward(
                spec, dyv, xv, gamma, mean, rstd, has_beta=has_beta, block_threads=bt
            )
        return dx.reshape(x.shape), dgamma, dbeta

    if variant == NormVariant.BATCH_NORM:
        from cudnn.norm.bprop.frost.batchnorm import batchnorm_backward

        spec = batchnorm_spec(x.shape)
        xv = x.reshape(spec.N, spec.C, spec.S)
        dyv = dy.reshape(spec.N, spec.C, spec.S)
        bt = choose_block_threads(spec.count)
        dx, dgamma, dbeta = batchnorm_backward(
            spec, dyv, xv, gamma, mean, rstd, has_beta=has_beta, block_threads=bt
        )
        return dx.reshape(x.shape), dgamma, dbeta

    raise ValueError(f"unknown norm variant {variant}")
