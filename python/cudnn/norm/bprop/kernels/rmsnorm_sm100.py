"""RMSNorm backward, sm_100.

RMSNorm backward is LayerNorm backward with no centering term (``a == 0``),
served by the LayerNorm kernel with ``spec.has_mean == False``. Per-flavor entry
point.
"""

from __future__ import annotations

from . import layernorm_sm100


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, cfg, params):
    """Launch RMSNorm backward. Returns ``(dx, dgamma, dbeta)``."""
    return layernorm_sm100.backward(
        spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params
    )
