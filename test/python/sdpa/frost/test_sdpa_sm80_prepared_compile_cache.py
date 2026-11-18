# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared SM80 hosts must survive artifact export and reuse on a fresh plan."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires SM80")]


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256)])
def test_replan_reloads_prepared_artifact(direction, d, dv, tmp_path, monkeypatch, split=1, pack_gqa=False):
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    if direction == "fwd":
        from sdpa.frost.test_sdpa_prepared_sm80 import _case, _check

        import cudnn

        knobs = {cudnn.knob_type.SPLIT_KV: split} if split > 1 else None
        if pack_gqa:
            knobs = {**(knobs or {}), cudnn.knob_type.PACK_GQA: 1}
        build = lambda: _case(d, dv, features=d == 128, graph_knobs=knobs)
    else:
        from sdpa.frost.test_sdpa_bwd_prepared_sm80 import _case, _check

        build = lambda: _case(d, dv, hk=4 if d == 64 else 2, causal=d != 64, features=d == 128)
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")  # the reload path is under test; the memo would hand plan 2 the object plan 1 got
    first = build()
    first.graph.execute(first.pack, first.workspace)
    _check(first)
    counts = compiled_cache.stats()

    def forbidden(*args, **kwargs):
        raise AssertionError("replanning an identical prepared SM80 graph invoked JIT")

    monkeypatch.setattr(cute, "compile", forbidden)
    second = build()
    plan = second.graph._compiled_plans[second.graph._plan_index]
    assert plan._prepared is not None
    assert hasattr(plan._prepared.spec.artifact, "_compiled_cache_raw")
    assert compiled_cache.stats()["hits"] > counts["hits"]
    assert compiled_cache.stats()["misses"] == counts["misses"]
    second.graph.execute(second.pack, second.workspace)
    _check(second)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            second.graph.execute(second.pack, second.workspace)
        names = ("o", "stats") if direction == "fwd" else tuple(second.expected)
        for name in names:
            second.bufs[name].fill_(float("nan"))
        capture.replay()
        _check(second)
    finally:
        capture.reset()


@pytest.mark.parametrize("d", [128, 512])
def test_split_replan_reloads_prepared_artifact(d, tmp_path, monkeypatch):
    test_replan_reloads_prepared_artifact("fwd", d, d, tmp_path, monkeypatch, split=3)


@pytest.mark.parametrize("d", [128, 512])
def test_pack_replan_reloads_prepared_artifact(d, tmp_path, monkeypatch):
    test_replan_reloads_prepared_artifact("fwd", d, d, tmp_path, monkeypatch, split=3, pack_gqa=True)
