# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sm_100 norm forward kernels (CUTLASS primitives).

One entry module per flavor; the shared cp.async row staging + block reduction
live in :mod:`_common_sm100`.

    layernorm_sm100     LayerNorm  (also the shared LN/RMS reduce-last-dim kernel)
    layernorm_cga_sm100 LN/RMS with one row split across a CGA (DSMEM reduction, large C)
    rmsnorm_sm100       RMSNorm    (LayerNorm kernel, has_mean=False)
    groupnorm_sm100     GroupNorm  (shared row-wise kernel, affine c=(r%gps)*cpg+j//span)
    instancenorm_sm100  InstanceNorm (GroupNorm with num_groups=C)
    batchnorm_sm100     BatchNorm  (generic fallback, one CTA per channel)
    batchnorm_nhwc_sm100 BatchNorm NHWC (fused cooperative split-K over [M=NHW, C])
    batchnorm_nchw_sm100 BatchNorm NCHW (warp-per-row split-K over the batch)
    groupnorm_nhwc_sm100  GroupNorm NHWC (coalescing-sized channel tile, segmented group reduce)
    instancenorm_nhwc_sm100 InstanceNorm NHWC (per-image BN-NHWC map, stats per (n, c))

Each module exposes ``forward(spec, x, gamma, beta, *, eps, cfg, params)`` and
caches the compiled kernel per (io_dtype, structural flags, block_threads).
"""

# Deliberately NOT imported here. Each kernel module imports ``cutlass.primitives``
# at its top, so a package-level fan-out would load the CuTe DSL for anything that
# merely touches this package -- including the engine, before its gate runs.
# Callers import the one module they need; the tests enumerate with pkgutil.
