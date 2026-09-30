# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm forward, sm_100.

RMSNorm has the identical row reduction as LayerNorm but no mean subtraction
(``var = mean(x^2)``). It is served by the LayerNorm kernel with
``spec.has_mean == False`` rather than duplicating the ~90-line kernel for a
single compile-time flag. This module is the per-flavor entry point.
"""

from __future__ import annotations

from . import layernorm_sm100


def forward(spec, x2d, gamma, beta, *, eps, cfg, params):
    """Launch RMSNorm forward. Returns ``(y, mean, rstd)`` (``mean`` unused)."""
    return layernorm_sm100.forward(spec, x2d, gamma, beta, eps=eps, cfg=cfg, params=params)
