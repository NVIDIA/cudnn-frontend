# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native geometry survives public graph planning without retuning defaults."""

from dataclasses import replace
from types import SimpleNamespace

import cudnn
import pytest

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import engines, heuristics

pytestmark = pytest.mark.L0

_CAPS = engines._sm80_spec().capabilities
_POINTS = [
    (d, dv, m, n)
    for d, dv, ms, ns in (
        (64, 64, (64, 128), (64, 128)),
        (128, 128, (64, 128), (64, 128)),
        (192, 128, (64, 128), (64, 128)),
        (256, 256, (64, 128), (32, 64)),
        (512, 512, (32,), (16,)),
        (576, 512, (32,), (16,)),
    )
    for m in ms
    for n in ns
]


def _facts(d=192, dv=128, mode="dense", causal=False):
    def port(heads, seq, width):
        return SimpleNamespace(
            get_dim=lambda: (1, heads, seq, width),
            get_stride=lambda: (seq * heads * width, width, heads * width, 1),
            get_data_type=lambda: cudnn.data_type.BFLOAT16,
            ragged_offset=object() if mode == "thd" else None,
        )

    return ga.SdpaGraphFacts(
        b=1,
        h_q=64,
        h_kv=8,
        s_q=192,
        s_kv=2048,
        d_qk=d,
        d_v=dv,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(8, 0),
        causal=causal,
        bottom_right=causal,
        thd=mode == "thd",
        has_cu_seq_len=mode == "thd",
        cu_seq_q_t=object() if mode == "thd" else None,
        cu_seq_kv_t=object() if mode == "thd" else None,
        has_paged_kv=mode == "paged",
        page_size=16 if mode == "paged" else 0,
        padded=mode == "paged",
        q_t=port(64, 192, d),
        k_t=port(8, 2048, d),
        v_t=port(8, 2048, dv),
        o_t=port(64, 192, dv),
    )


@pytest.mark.parametrize("d,dv,m,n", _POINTS + [(48, 32, 64, 128), (96, 80, 64, 64), (136, 112, 64, 128), (248, 200, 64, 32)])
@pytest.mark.parametrize("mode", ["dense", "thd", "paged"])
def test_explicit_graph_geometry(d, dv, m, n, mode):
    facts = _facts(d, dv, mode)
    knobs = engines.SdpaFwdKnobs(tile_m=m, tile_n=n, cga=1, sched_policy=0)
    assert engines.mismatch(_CAPS, facts, knobs) is None
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs


@pytest.mark.parametrize(
    "d,dv,kw,label",
    [
        (192, 128, {"tile_m": 32}, "tile_m"),
        (192, 128, {"tile_n": 32}, "tile_n"),
        (256, 256, {"tile_n": 128}, "tile_n"),
        (248, 200, {"tile_n": 128}, "tile_n"),
        (512, 512, {"tile_m": 64}, "tile_m"),
        (576, 512, {"tile_n": 64}, "tile_n"),
        (192, 128, {"cga": 2}, "cga"),
        (192, 128, {"split_kv": 2}, "split"),
        (192, 128, {"pack_gqa": True}, "pack_gqa"),
    ],
)
def test_unsupported_geometry_declines(d, dv, kw, label):
    reason = engines.mismatch(_CAPS, _facts(d, dv), engines.SdpaFwdKnobs(**kw))
    assert reason is not None and label in reason


@pytest.mark.parametrize("mode", ["dense", "thd", "paged"])
@pytest.mark.parametrize("causal", [False, True])
def test_default_and_scheduler_only_domains_stay_implicit(mode, causal):
    facts = _facts(mode=mode, causal=causal)
    for knobs in (None, engines.SdpaFwdKnobs(sched_policy=0)):
        assert engines.mismatch(_CAPS, facts, knobs) is None
        assert engines.effective_tile_ms(_CAPS, facts, knobs) == frozenset()
        assert engines.effective_tile_ns(_CAPS, facts, knobs) == frozenset()
    assert engines.effective_cgas(_CAPS, facts) == frozenset()
    for kind in ("A", "FALLBACK"):
        plans = heuristics.recommend(kind, facts, {"sdpa_fwd_prefill_sm80": 1})
        assert plans
        assert all(p.knobs.tile_m is p.knobs.tile_n is p.knobs.cga is None for p in plans)


def test_explicit_tiles_keep_feature_and_layout_gates():
    knobs = engines.SdpaFwdKnobs(tile_m=64, tile_n=64, cga=1)
    assert "natural scheduler" in engines.mismatch(_CAPS, _facts(mode="thd"), replace(knobs, sched_policy=1))
    assert engines.mismatch(_CAPS, _facts(193, 128), knobs) is not None
    assert engines.mismatch(_CAPS, replace(_facts(), has_dropout=True), knobs) is not None


def _require_sm80():
    import torch
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("requires supported CuTeDSL")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires native SM80")


@pytest.mark.parametrize("d,dv,m,n", _POINTS)
def test_public_graph_geometry_rebind_replay(d, dv, m, n, cudnn_handle, monkeypatch):
    _require_sm80()
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
    from test_sdpa_prepared_sm80 import _case, _check

    observed = []
    original = SdpaFwdDslSm80.check_support

    def record(api):
        result = original(api)
        observed.append((api.kernel_tile_m, api.kernel_tile_n, api.kernel_num_warps))
        return result

    monkeypatch.setattr(SdpaFwdDslSm80, "check_support", record)
    scheduler = 2 if n == 64 else 0
    public = {cudnn.knob_type.TILE_M: m, cudnn.knob_type.TILE_N: n, cudnn.knob_type.TILE_CGA_M: 1, cudnn.knob_type.SCHED_POLICY: scheduler}
    case = _case(d, dv, sq=192, layout="gapped", stats=m != 128, causal=scheduler == 2, graph_knobs=public)
    g = case.graph
    assert g.get_engine_and_knobs_at_index(g._plan_index)[1] == public
    warps = 4 if m == 64 or d == 64 or d > 256 else 8
    assert observed and observed[-1] == (m, n, warps)
    for binding in range(2):
        if binding:
            for name, old in list(case.bufs.items()):
                new = torch.empty_strided(old.shape, old.stride(), dtype=old.dtype, device=old.device).copy_(old)
                case.bufs[name] = new
                case.pack[case.refs[name]] = new
        case.bufs["v"].mul_(-0.7)
        g.execute(case.pack, case.workspace, handle=cudnn_handle)
        _check(case)
        capture = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        previous = cudnn.get_stream(cudnn_handle)
        try:
            cudnn.set_stream(cudnn_handle, stream.cuda_stream)
            with torch.cuda.graph(capture, stream=stream):
                g.execute(case.pack, case.workspace, handle=cudnn_handle)
            torch.cuda.current_stream().wait_stream(stream)
            case.bufs["v"].mul_(0.5)
            case.bufs["o"].fill_(float("nan"))
            if "stats" in case.bufs:
                case.bufs["stats"].fill_(float("nan"))
            capture.replay()
            _check(case)
        finally:
            cudnn.set_stream(cudnn_handle, previous)
            capture.reset()
    assert torch.all(case.storage["o"][..., dv:] == 123)


@pytest.mark.parametrize("d,dv,m,n", [(192, 128, 64, 128), (256, 256, 64, 32)])
@pytest.mark.parametrize("thd", [False, True])
def test_public_paged_graph_geometry_rebind_replay(d, dv, m, n, thd, cudnn_handle, monkeypatch):
    _require_sm80()
    import torch
    from test_sdpa_fwd_paged_sm80 import test_native_paged_adapter_rebind_replay as run

    public = {cudnn.knob_type.TILE_M: m, cudnn.knob_type.TILE_N: n, cudnn.knob_type.TILE_CGA_M: 1, cudnn.knob_type.SCHED_POLICY: 0}
    run((d, dv, 16, "HND", "NHD", "br_swa"), torch.bfloat16, thd, True, cudnn_handle, monkeypatch, case_index=25, graph_knobs=public)
