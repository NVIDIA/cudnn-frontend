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
    x = x.contiguous()
    dy = dy.contiguous()
    io = torch_dtype_to_str(x.dtype)

    if variant in ROWWISE_VARIANTS:
        spec = rowwise_spec(variant, x.shape, normalized_shape=normalized_shape, num_groups=num_groups)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=has_beta)
        cfg = make_cfg(params, spec.M, staged_rows=2)  # backward stages X and DY
        x2d = x.reshape(spec.R, spec.M)
        dy2d = dy.reshape(spec.R, spec.M)
        dx, dgamma, dbeta = _ROWWISE_KERNEL[variant].backward(spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
        return dx.reshape(x.shape), dgamma, dbeta

    if variant == NormVariant.BATCH_NORM:
        spec = batchnorm_spec(x.shape)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=has_beta)
        eb = DTYPE_BYTES[io]
        cfg = Cfg(block_threads=choose_block_threads(spec.count), V=vector_width(eb), stage_mode=STAGE_NONE, vec=False, elem_bytes=eb)
        x3d = x.reshape(spec.N, spec.C, spec.S)
        dy3d = dy.reshape(spec.N, spec.C, spec.S)
        dx, dgamma, dbeta = batchnorm_sm100.backward(spec, dy3d, x3d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params)
        return dx.reshape(x.shape), dgamma, dbeta

    raise ValueError(f"unknown norm variant {variant}")
