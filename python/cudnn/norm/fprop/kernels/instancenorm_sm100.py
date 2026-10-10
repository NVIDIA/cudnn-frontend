# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InstanceNorm forward, sm_100.

InstanceNorm normalizes each ``(sample, channel)`` over its spatial extent — i.e.
GroupNorm with ``num_groups == C`` (one channel per group). It reuses the
GroupNorm kernel: the derived :class:`~cudnn.norm.config_sm100.RowwiseSpec` has
``groups_per_sample = C``, ``channels_per_group = 1``, ``gamma_inner_span = HW``,
so the affine channel reduces to ``r % C``. This module is the per-flavor entry
point.
"""

from __future__ import annotations

from . import groupnorm_sm100


def forward(spec, x2d, gamma, beta, *, eps, cfg, params):
    """Launch InstanceNorm forward. Returns ``(y, mean, rstd)``."""
    return groupnorm_sm100.forward(spec, x2d, gamma, beta, eps=eps, cfg=cfg, params=params)
