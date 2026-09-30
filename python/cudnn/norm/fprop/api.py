# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level torch-tensor dispatch for the sm_100 norm forward kernels.

Takes torch tensors in their natural layout, derives the
:class:`~cudnn.norm.config_sm100.RowwiseSpec` / :class:`BatchNormSpec`, resolves
the launch :class:`Cfg`, reshapes to the canonical view each kernel expects, and
launches. This is the direct (non-graph) entry point used by the tests; the
cuDNN graph engine (``cudnn.norm.fprop.engines``) lowers onto the same kernels.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..config_sm100 import (
    STAGE_NONE,
    Cfg,
    NormVariant,
    ROWWISE_VARIANTS,
    TemplateParams,
    batchnorm_spec,
    choose_block_threads,
    make_cfg,
    make_warp_cfg,
    rowwise_spec,
    vector_width,
)

# LN/RMS reduce the last dim (M == C == gamma_len) and can use the warp-per-row
# kernel; GN/IN have a different affine mapping and stay on the row-reduce kernel.
_WARP_VARIANTS = (NormVariant.LAYER_NORM, NormVariant.RMS_NORM)
from ..dtypes import DTYPE_BYTES, torch_dtype_to_str
from .kernels import (
    batchnorm_nchw_sm100,
    batchnorm_nhwc_sm100,
    batchnorm_sm100,
    groupnorm_sm100,
    instancenorm_sm100,
    layernorm_sm100,
    rmsnorm_sm100,
)

_ROWWISE_KERNEL = {
    NormVariant.LAYER_NORM: layernorm_sm100,
    NormVariant.RMS_NORM: rmsnorm_sm100,
    NormVariant.GROUP_NORM: groupnorm_sm100,
    NormVariant.INSTANCE_NORM: instancenorm_sm100,
}


def _as_variant(v) -> NormVariant:
    return v if isinstance(v, NormVariant) else NormVariant(v)


def _is_nhwc(x) -> bool:
    """True for a 4-D tensor whose memory is channels-last (NHWC)."""
    import torch

    return x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)


def norm_fprop(
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
):
    """Forward pass for any of the five norm variants.

    Returns ``(y, mean, rstd)``. ``y`` has the input's shape; ``mean``/``rstd``
    are fp32 per-group statistics (flat). For RMSNorm ``mean`` is a placeholder.
    """
    import torch

    variant = _as_variant(variant)
    # BatchNorm has a dedicated channels-last kernel, so preserve NHWC inputs
    # instead of paying a transpose in ``.contiguous()``.
    if not (variant == NormVariant.BATCH_NORM and _is_nhwc(x)):
        x = x.contiguous()
    io = torch_dtype_to_str(x.dtype)

    if variant in ROWWISE_VARIANTS:
        spec = rowwise_spec(variant, x.shape, normalized_shape=normalized_shape, num_groups=num_groups)
        if gamma is None:
            gamma = torch.ones(spec.gamma_len, dtype=x.dtype, device=x.device)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=(beta is not None))
        x2d = x.reshape(spec.R, spec.M)

        # Warp-per-row kernel for LN/RMS when the row fits the register budget.
        if variant in _WARP_VARIANTS:
            wcfg = make_warp_cfg(params, spec.M)
            if wcfg is not None:
                from .kernels import layernorm_warp_sm100

                y2, mean, rstd = layernorm_warp_sm100.forward(spec, x2d, gamma, beta, eps=eps, wcfg=wcfg, params=params)
                return y2.reshape(x.shape), mean, rstd

        cfg = make_cfg(params, spec.M)
        y2, mean, rstd = _ROWWISE_KERNEL[variant].forward(spec, x2d, gamma, beta, eps=eps, cfg=cfg, params=params)
        return y2.reshape(x.shape), mean, rstd

    if variant == NormVariant.BATCH_NORM:
        spec = batchnorm_spec(x.shape)
        if gamma is None:
            gamma = torch.ones(spec.C, dtype=x.dtype, device=x.device)
        update_running = bool(training and running_mean is not None and running_var is not None)
        params = TemplateParams(
            variant=variant,
            io_dtype=io,
            has_beta=(beta is not None),
            training=training,
            update_running=update_running,
        )
        eb = DTYPE_BYTES[io]

        # --- NHWC (channels-last): fused cooperative split-K kernel over [M=N*H*W, C] ---
        if _is_nhwc(x) and batchnorm_nhwc_sm100.nhwc_cfg(spec.C, eb) is not None:
            N, C, H, W = (int(v) for v in x.shape)
            x2d = x.permute(0, 2, 3, 1).reshape(N * H * W, C)  # view, no copy
            y2d, sm, sr = batchnorm_nhwc_sm100.forward(
                spec,
                x2d,
                gamma,
                beta,
                eps=eps,
                momentum=momentum,
                training=training,
                running_mean=running_mean,
                running_var=running_var,
                cfg=None,
                params=params,
            )
            y = y2d.reshape(N, H, W, C).permute(0, 3, 1, 2)
            return y, sm, sr

        # --- NCHW: split-K over the (n, c) rows, each of S contiguous elements ---
        if batchnorm_nchw_sm100.nchw_cfg(spec.C, spec.N, spec.S, eb) is not None:
            x3d = x.reshape(spec.N, spec.C, spec.S)
            y3, sm, sr = batchnorm_nchw_sm100.forward(
                spec,
                x3d,
                gamma,
                beta,
                eps=eps,
                momentum=momentum,
                training=training,
                running_mean=running_mean,
                running_var=running_var,
                cfg=None,
                params=params,
            )
            return y3.reshape(x.shape), sm, sr

        cfg = Cfg(block_threads=choose_block_threads(spec.count), V=vector_width(eb), stage_mode=STAGE_NONE, vec=False, elem_bytes=eb)
        x3d = x.reshape(spec.N, spec.C, spec.S)
        y3, sm, sr = batchnorm_sm100.forward(
            spec,
            x3d,
            gamma,
            beta,
            eps=eps,
            momentum=momentum,
            training=training,
            running_mean=running_mean,
            running_var=running_var,
            cfg=cfg,
            params=params,
        )
        return y3.reshape(x.shape), sm, sr

    raise ValueError(f"unknown norm variant {variant}")
