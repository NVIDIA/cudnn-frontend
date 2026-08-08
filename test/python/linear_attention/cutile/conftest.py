# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cutile suite validates the cuTile engines and their kernels, so op
calls made here pin them — under default ranking the FROST engines would serve
the FROST-eligible shapes instead.

The pin is enforced where it is applied: engine_utils.apply_pin() raises if the
pinned engine produced no plan, so a pin that stops working fails the first op
call rather than silently testing another engine. This suite once ran for
months against whichever engine the router picked, because the seam it pinned
through was dead and nothing checked."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="package")
def _pin_cutile_engines():
    from cudnn.linear_attention import engine_utils
    from cudnn.linear_attention.ops import gdn, kda

    saved = engine_utils.pin_engines(("gdn_cutile", "kda_cutile"))
    for m in (gdn, kda):
        m._fwd_graph_cache.clear()
        m._bwd_graph_cache.clear()
    yield
    engine_utils.pin_engines(saved)
    for m in (gdn, kda):
        m._fwd_graph_cache.clear()
        m._bwd_graph_cache.clear()
