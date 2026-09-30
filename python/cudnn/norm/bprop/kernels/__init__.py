# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sm_100 norm backward kernels (CUTLASS primitives).

One entry module per flavor; the shared cp.async row staging + block reduction
live in :mod:`cudnn.norm._common_sm100`.

    layernorm_sm100     LayerNorm  (also the shared LN/RMS backward kernel)
    rmsnorm_sm100       RMSNorm    (LayerNorm backward, has_mean=False)
    groupnorm_sm100     GroupNorm  (shared row-wise backward, per-channel atomics)
    instancenorm_sm100  InstanceNorm (GroupNorm with num_groups=C)
    batchnorm_sm100     BatchNorm  (across-batch, one CTA per channel, no atomics)

Each module exposes ``backward(spec, dy, x, gamma, mean, rstd, *, has_beta, cfg, params)``.
"""

from . import (  # noqa: F401
    batchnorm_sm100,
    groupnorm_sm100,
    instancenorm_sm100,
    layernorm_sm100,
    rmsnorm_sm100,
)
