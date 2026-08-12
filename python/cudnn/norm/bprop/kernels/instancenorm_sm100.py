"""InstanceNorm backward, sm_100.

InstanceNorm is GroupNorm with ``num_groups == C``; its backward reuses the
GroupNorm kernel with the InstanceNorm
:class:`~cudnn.norm.config_sm100.RowwiseSpec`. Per-flavor entry point.
"""

from __future__ import annotations

from . import groupnorm_sm100


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, cfg, params):
    """Launch InstanceNorm backward. Returns ``(dx, dgamma, dbeta)``."""
    return groupnorm_sm100.backward(
        spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, cfg=cfg, params=params
    )
