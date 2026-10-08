# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level torch-tensor dispatch for the sm_100 norm backward kernels.

Takes ``dy`` plus the forward's saved ``x``, ``gamma``, ``mean``, ``rstd``,
derives the layout + launch :class:`Cfg`, and launches. Returns
``(dx, dgamma, dbeta)`` (``dbeta`` is ``None`` when ``has_beta`` is False);
``dgamma``/``dbeta`` are fp32.
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
    rowwise_spec,
    vector_width,
)
from ..dtypes import DTYPE_BYTES, torch_dtype_to_str
from .kernels import (
    batchnorm_nchw_sm100,
    batchnorm_nhwc_sm100,
    groupnorm_fast_sm100,
    groupnorm_nhwc_sm100,
    instancenorm_nhwc_sm100,
    instancenorm_warp_sm100,
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


def _is_nhwc(t) -> bool:
    """True for a 4-D tensor whose memory is channels-last (NHWC)."""
    import torch

    return t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last)


def norm_bprop(
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
):
    """Backward pass for any of the five norm variants.

    ``mean``/``rstd`` are the fp32 statistics returned by ``norm_fprop``.
    Returns ``(dx, dgamma, dbeta)``.
    """
    variant = _as_variant(variant)
    # BatchNorm has a native channels-last backward, so preserve NHWC instead of
    # paying the transpose that .contiguous() would do.
    _nhwc_native = _as_variant(variant) in (NormVariant.BATCH_NORM, NormVariant.INSTANCE_NORM, NormVariant.GROUP_NORM)
    if not (_nhwc_native and _is_nhwc(x) and _is_nhwc(dy)):
        x = x.contiguous()
        dy = dy.contiguous()
    io = torch_dtype_to_str(x.dtype)

    if variant in ROWWISE_VARIANTS:
        spec = rowwise_spec(variant, x.shape, normalized_shape=normalized_shape, num_groups=num_groups)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=has_beta)
        cfg = make_cfg(params, spec.M, staged_rows=2)  # backward stages X and DY
        # GroupNorm, channels-last: a group is cpg ADJACENT channels, contiguous within
        # a pixel in NHWC, so the channel tile is the group itself.
        if variant == NormVariant.GROUP_NORM and _is_nhwc(x):
            N, C, H, W = (int(v) for v in x.shape)
            if groupnorm_nhwc_sm100.eligible(C, int(spec.channels_per_group), DTYPE_BYTES[io]):
                x3 = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
                dy3 = dy.permute(0, 2, 3, 1).reshape(N, H * W, C)
                dx3, dgamma, dbeta = groupnorm_nhwc_sm100.backward(spec, dy3, x3, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
                return dx3.reshape(N, H, W, C).permute(0, 3, 1, 2), dgamma, dbeta
        # InstanceNorm, channels-last: an IN group is strided by C in NHWC, so the
        # rowwise view does not exist -- use the per-image NHWC kernel instead of
        # transposing.
        if variant == NormVariant.INSTANCE_NORM and _is_nhwc(x):
            N, C, H, W = (int(v) for v in x.shape)
            x3 = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
            dy3 = dy.permute(0, 2, 3, 1).reshape(N, H * W, C)
            if instancenorm_nhwc_sm100.nhwc_cfg(C, DTYPE_BYTES[io]) is not None:
                dx3, dgamma, dbeta = instancenorm_nhwc_sm100.backward(spec, dy3, x3, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
                return dx3.reshape(N, H, W, C).permute(0, 3, 1, 2), dgamma, dbeta
        x2d = x.reshape(spec.R, spec.M)
        dy2d = dy.reshape(spec.R, spec.M)
        # GN/IN: the atomic-free map (channel set fixed per CTA) when it applies.
        # InstanceNorm with short rows: a WARP per row, so the row reduction is a
        # shuffle rather than a block reduce through shared memory.
        if variant == NormVariant.INSTANCE_NORM and instancenorm_warp_sm100.eligible(spec, DTYPE_BYTES[io]):
            dx, dgamma, dbeta = instancenorm_warp_sm100.backward(spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
            return dx.reshape(x.shape), dgamma, dbeta
        if variant in (NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM) and groupnorm_fast_sm100.eligible(spec, DTYPE_BYTES[io]):
            dx, dgamma, dbeta = groupnorm_fast_sm100.backward(spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
            return dx.reshape(x.shape), dgamma, dbeta
        # LN/RMS with a long row: split it across a CGA and reduce c1/c2 through
        # distributed shared memory, exactly as the forward does. Only past the
        # measured crossover -- the pipelined kernel below is better for short rows.
        if variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
            from .kernels import layernorm_cga_sm100

            if layernorm_cga_sm100.should_use(spec.M, spec.R, DTYPE_BYTES[io]):
                dx, dgamma, dbeta = layernorm_cga_sm100.backward(
                    spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, params=params
                )
                return dx.reshape(x.shape), dgamma, dbeta
        dx, dgamma, dbeta = _ROWWISE_KERNEL[variant].backward(spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
        return dx.reshape(x.shape), dgamma, dbeta

    if variant == NormVariant.BATCH_NORM:
        spec = batchnorm_spec(x.shape)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=has_beta)
        eb = DTYPE_BYTES[io]
        cfg = Cfg(block_threads=choose_block_threads(spec.count), V=vector_width(eb), stage_mode=STAGE_NONE, vec=False, elem_bytes=eb)
        x3d = x.reshape(spec.N, spec.C, spec.S)
        dy3d = dy.reshape(spec.N, spec.C, spec.S)
        # NHWC: the forward's fused cooperative split-K with both streams cached.
        if _is_nhwc(x) and batchnorm_nhwc_sm100.nhwc_cfg(spec.C, eb) is not None:
            N, C, H, W = (int(v) for v in x.shape)
            x2d = x.permute(0, 2, 3, 1).reshape(N * H * W, C)
            dy2d = dy.permute(0, 2, 3, 1).reshape(N * H * W, C)
            dx2, dgamma, dbeta = batchnorm_nhwc_sm100.backward(spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
            return dx2.reshape(N, H, W, C).permute(0, 3, 1, 2), dgamma, dbeta
        # NCHW: split-K over the batch, fixed channels per warp (no atomics).
        if batchnorm_nchw_sm100.nchw_cfg(spec.C, spec.N, spec.S, eb) is not None:
            dx, dgamma, dbeta = batchnorm_nchw_sm100.backward(spec, dy3d, x3d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
            return dx.reshape(x.shape), dgamma, dbeta
        dx, dgamma, dbeta = batchnorm_sm100.backward(spec, dy3d, x3d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
        return dx.reshape(x.shape), dgamma, dbeta

    raise ValueError(f"unknown norm variant {variant}")
