# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sm_100 norm forward kernels (CUTLASS primitives).

One entry module per flavor; the shared cp.async row staging + block reduction
live in :mod:`_common_sm100`.

    layernorm_sm100     LayerNorm  (also the shared LN/RMS reduce-last-dim kernel)
    rmsnorm_sm100       RMSNorm    (LayerNorm kernel, has_mean=False)
    groupnorm_sm100     GroupNorm  (shared row-wise kernel, affine c=(r%gps)*cpg+j//span)
    instancenorm_sm100  InstanceNorm (GroupNorm with num_groups=C)
    batchnorm_sm100     BatchNorm  (generic fallback, one CTA per channel)
    batchnorm_nhwc_sm100 BatchNorm NHWC (fused cooperative split-K over [M=NHW, C])
    batchnorm_nchw_sm100 BatchNorm NCHW (warp-per-row split-K over the batch)

Each module exposes ``forward(spec, x, gamma, beta, *, eps, cfg, params)`` and
caches the compiled kernel per (io_dtype, structural flags, block_threads).
"""

from . import (  # noqa: F401
    batchnorm_nchw_sm100,
    batchnorm_nhwc_sm100,
    batchnorm_sm100,
    groupnorm_sm100,
    instancenorm_sm100,
    layernorm_sm100,
    layernorm_warp_sm100,
    rmsnorm_sm100,
)
