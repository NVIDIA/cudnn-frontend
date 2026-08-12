"""sm_100 norm forward kernels (CUTLASS primitives).

One entry module per flavor; the shared cp.async row staging + block reduction
live in :mod:`_common_sm100`.

    layernorm_sm100     LayerNorm  (also the shared LN/RMS reduce-last-dim kernel)
    rmsnorm_sm100       RMSNorm    (LayerNorm kernel, has_mean=False)
    groupnorm_sm100     GroupNorm  (shared row-wise kernel, affine c=(r%gps)*cpg+j//span)
    instancenorm_sm100  InstanceNorm (GroupNorm with num_groups=C)
    batchnorm_sm100     BatchNorm  (across-batch, one CTA per channel)

Each module exposes ``forward(spec, x, gamma, beta, *, eps, cfg, params)`` and
caches the compiled kernel per (io_dtype, structural flags, block_threads).
"""

from . import (  # noqa: F401
    batchnorm_sm100,
    groupnorm_sm100,
    instancenorm_sm100,
    layernorm_sm100,
    rmsnorm_sm100,
)
