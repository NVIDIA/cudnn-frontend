# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cutile suite validates the cuTile engines and their kernels, so op
calls made here pin them — under default routing the FROST engines (ranked
first) would serve the FROST-eligible shapes instead."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="package")
def _pin_cutile_engines():
    from cudnn.engines import GdnCuTileEngine, KdaCuTileEngine
    from cudnn.linear_attention.ops import gdn, kda

    saved = gdn._engines, kda._engines
    gdn._engines = [GdnCuTileEngine()]
    kda._engines = [KdaCuTileEngine()]
    for m in (gdn, kda):
        m._fwd_graph_cache.clear()
        m._bwd_graph_cache.clear()
    _assert_pin_took_effect()
    yield
    gdn._engines, kda._engines = saved
    for m in (gdn, kda):
        m._fwd_graph_cache.clear()
        m._bwd_graph_cache.clear()


def _assert_pin_took_effect():
    """A pin nobody verifies is a suite that silently tests another engine.

    This suite existed for months while the seam it pins through was dead, and
    every test still passed — against whichever engine the router picked."""
    import torch

    import cudnn
    from cudnn.linear_attention.ops import gdn

    if not torch.cuda.is_available():
        return
    graph = cudnn.pygraph()
    for engine in gdn._gdn_engines() or ():
        graph.register_backend(engine)
    names = [e.name for e in graph._candidate_engines()]
    assert names == ["gdn_cutile"], f"the cuTile pin did not take effect; candidates are {names}"
