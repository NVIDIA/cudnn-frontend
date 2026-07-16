"""cudnn.norm.frost: JIT norm kernels (LayerNorm / RMSNorm / GroupNorm /
BatchNorm / InstanceNorm) written in CUTLASS CuTe-DSL primitives.

Public API::

    from cudnn.norm.frost import NormVariant, norm_forward, norm_backward

    y, mean, rstd = norm_forward(NormVariant.LAYER_NORM, x, gamma, beta, eps=1e-5)
    dx, dgamma, dbeta = norm_backward(NormVariant.LAYER_NORM, dy, x, gamma, mean, rstd)

fprop kernels live under ``cudnn.norm.fprop.frost`` and bprop kernels under
``cudnn.norm.bprop.frost``. The four per-sample norms share one row-wise kernel;
BatchNorm has its own across-batch kernel.

Optional cuDNN-graph engine registration lives in :mod:`cudnn.norm.frost.engine`
and is imported explicitly (it depends on a built cuDNN frontend); the kernel
library above is usable standalone.
"""

from .api import norm_backward, norm_forward
from .config import NormVariant

# Engine name for graph.select_engines(["frost_norm_eng0"]) once the graph
# engine is wired up (see cudnn.norm.frost.engine).
ENGINE_NAME = "frost_norm_eng0"

__all__ = ["NormVariant", "norm_forward", "norm_backward", "ENGINE_NAME"]
