# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An engine that cannot serve a graph must say so with a decline type.

``build_plans()`` walks the ranked plan list and skips an entry that raises one
of ``engines.base.decline_types()``, moving on to the next plan and ultimately
to the cuDNN backend. Anything else propagates and aborts the walk, so a graph
the backend could have served fails outright.

"This machine has no CUDA device" and "the driver did not report the property I
need to size a pipeline" are declines: the engine cannot serve the graph, but
another entry in the list can. They were raising RuntimeError, which is not a
decline type and is not caught by the engines' own ``build_plan`` handlers
either, so a probe failure took down the whole walk instead of falling back.
"""

import pytest

import cudnn
from cudnn.engines.base import decline_types


@pytest.mark.L0
def test_device_probes_decline_when_no_driver(monkeypatch):
    from cudnn.frost import device

    monkeypatch.setattr(device, "_driver", lambda: None)

    with pytest.raises(decline_types()):
        device.current_device()
    with pytest.raises(decline_types()):
        device._device_handle(0)
    with pytest.raises(decline_types()):
        with device.device_context(0):
            pass


@pytest.mark.L0
@pytest.mark.parametrize("probe", ["_sm_smem_budget_bytes_of", "_l2_swizzle_budget_bytes_of"])
def test_tile_config_probes_decline_when_unavailable(monkeypatch, probe):
    from cudnn.frost import device as frost_device
    from cudnn.gemm.frost import tile_config

    fn = getattr(tile_config, probe)
    # Both probes are @lru_cache'd, so an earlier test that already queried this
    # device would serve a cached answer and never reach the raise.
    fn.cache_clear()
    monkeypatch.setattr(frost_device, "is_available", lambda: False)
    try:
        with pytest.raises(decline_types()):
            fn(0)
    finally:
        fn.cache_clear()


@pytest.mark.L0
def test_decline_types_are_what_build_plans_skips():
    """The tuple is the contract; keep it and the walk in agreement."""
    assert NotImplementedError in decline_types()
    assert cudnn.cudnnGraphNotSupportedError in decline_types()
    assert ImportError in decline_types()
    # RuntimeError must NOT be a decline: it is how an engine reports a bug,
    # and swallowing it would hide real failures behind a silent fallback.
    assert RuntimeError not in decline_types()
