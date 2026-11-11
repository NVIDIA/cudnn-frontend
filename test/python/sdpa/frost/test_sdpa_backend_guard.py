# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-forward family's backend planning guard (``sdpa/fwd/backend_guard.py``, declared through
``manifest.EngineFamily.backend_guard``): on cc 10.7 the cuDNN backend crashes the process while planning (9.26 / 9.27:
the heuristics) or building (9.28: the plan build) a single-query MXFP8 graph without a sink token, so planning records a
backend decline instead of asking.  Host-only: the
facts are synthetic, or the device is modelled as cc 10.7 with the sm_107a DSL target (``rubin_host``, the way
test_sdpa_fwd_heuristics.py::sm107_metadata_target models it), and nothing is ever built or executed."""

import math

import pytest

import cudnn
from cudnn._pygraph import cudnn_graph_not_supported, pygraph
from cudnn.engines import manifest
from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID, is_python_engine
from cudnn.sdpa import graph_analyzer
from cudnn.sdpa.fwd.backend_guard import SQ1_MXFP8_PLANNING_CRASH_FIXED_IN, backend_guard
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm107_mxfp8"
_GUARD = "backend crashes the process while planning or building single-query MXFP8 SDPA graphs on cc 10.0 / 10.3 / 10.7"
_TAIL = "the backend is not consulted for this graph"
_FAMILY = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")


def _facts(**over):
    base = dict(
        b=4,
        h_q=8,
        h_kv=2,
        s_q=1,
        s_kv=2048,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_mxfp8=True,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.fixture
def crashing_backend(monkeypatch):
    """A cuDNN 9.26.0 backend -- below any SQ1_MXFP8_PLANNING_CRASH_FIXED_IN that may be recorded, so the guard is armed --
    modelled on any host.  The guard reads ``cudnn.backend_version()`` at call time and the CI lanes run a newer cuDNN (9.28)
    than the login box (9.26), so every test of the ARMED guard pins the version instead of inheriting the installed
    library's: the day a fix version is recorded these tests keep describing the armed regime (test_version_bound covers the
    boundary itself)."""
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92600)
    monkeypatch.setattr(cudnn, "backend_version_string", lambda: "9.26.0")


@pytest.fixture
def rubin_host(crashing_backend, monkeypatch):
    """A cc 10.7 device with the sm_107a DSL target, the cuDNN 9.26 backend of ``crashing_backend`` and the flag deleted,
    modelled on any host."""
    from cudnn.frost import buffers

    monkeypatch.setattr(graph_analyzer, "_device_cc", lambda: (10, 7))
    monkeypatch.setattr(graph_analyzer, "_device_sm_count", lambda: 216)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)


def _mxfp8_graph(b=4, hq=8, hk=2, sq=1, skv=2048, d=128, dv=128, bshd=True, stats=False, sink=False, unaligned_k=False):
    """A dense graph.sdpa_mxfp8 forward declared as test/python/sdpa/mxfp8.py declares it (BSHD or BHSD strides)."""
    ceil_div = lambda a, m: -(-a // m)  # noqa: E731
    itype = cudnn.data_type.FP8_E4M3
    g = pygraph(io_data_type=itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    d_scale_pad = ceil_div(ceil_div(d, 32), 4) * 4
    dv_pad = ceil_div(dv, 128) * 128
    s_q_pad, s_kv_pad, s_kv_scale_pad = ceil_div(sq, 128) * 128, ceil_div(skv, 128) * 128, ceil_div(ceil_div(skv, 32), 4) * 4

    def sf(dims, name):
        return g.tensor(
            dim=dims,
            stride=(dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1),
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
            name=name,
        )

    q_stride = (sq * hq * d, d, hq * d, 1) if bshd else (hq * sq * d, sq * d, d, 1)
    o_stride = (sq * hq * dv, dv, hq * dv, 1) if bshd else (hq * sq * dv, sq * dv, dv, 1)
    k_stride = (skv * hk * d, d, hk * d, 1) if bshd else (hk * skv * d, skv * d, d, 1)
    if unaligned_k:
        k_stride = (hk * skv * (d + 1), skv * (d + 1), d + 1, 1)
    v_stride = (skv * hk * dv, dv, hk * dv, 1) if bshd else (hk * skv * dv, skv * dv, dv, 1)
    q = g.tensor(dim=(b, hq, sq, d), stride=q_stride, data_type=itype, name="q")
    k = g.tensor(dim=(b, hk, skv, d), stride=k_stride, data_type=itype, name="k")
    v = g.tensor(dim=(b, hk, skv, dv), stride=v_stride, data_type=itype, name="v")
    kw = dict(
        q=q,
        k=k,
        v=v,
        descale_q=sf((b, hq, s_q_pad, d_scale_pad), "sf_q"),
        descale_k=sf((b, hk, s_kv_pad, d_scale_pad), "sf_k"),
        descale_v=sf((b, hk, s_kv_scale_pad, dv_pad), "sf_v"),
        attn_scale=1.0 / math.sqrt(d),
        generate_stats=stats,
    )
    if sink:
        kw["sink_token"] = g.tensor(dim=(1, hq, 1, 1), stride=(hq, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="sink")
    o, st, amax = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim((b, hq, sq, dv)).set_stride(o_stride).set_data_type(cudnn.data_type.BFLOAT16)
    if stats:
        st.set_output(True).set_dim((b, hq, sq, 1)).set_stride((hq * sq, sq, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    return g


def _plan_names(g):
    return [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]


# --- the pure function on synthetic facts ---------------------------------------------------------------------------


def test_guard_names_the_backend_and_the_reason_on_the_measured_domain(crashing_backend):
    reason = backend_guard(None, _facts())
    assert reason is not None
    assert _GUARD in reason and _TAIL in reason and "cuDNN 9.26.0 " in reason


@pytest.mark.parametrize(
    "over",
    [
        dict(),
        dict(wants_stats=True),
        dict(thd=True, padded=True),
        dict(dtype_o=cudnn.data_type.HALF),
        dict(dtype_o=cudnn.data_type.FP8_E4M3),
        dict(dtype=cudnn.data_type.FP8_E5M2),
        dict(bshd_layout=False),
        dict(causal=True, right_bound=0),
        dict(b=1, h_q=1, h_kv=1, s_kv=128),
        dict(d_qk=256, d_v=256),
        dict(d_qk=192, d_v=128),
        dict(device_cc=(10, 0), device_sm_count=148),
        dict(device_cc=(10, 3), device_sm_count=148),
    ],
    ids=lambda over: ",".join(f"{k}={v}" for k, v in over.items()) or "base",
)
def test_every_single_query_mxfp8_graph_without_a_sink_is_guarded(crashing_backend, over):
    """The measured crash domain: dense and THD, Stats on or off, every O dtype, both inputs, every head dim alike, on cc 10.7
    and on cc 10.0 / 10.3 (the SM100 decode emitter)."""
    assert backend_guard(None, _facts(**over)) is not None


@pytest.mark.parametrize(
    "over",
    [
        dict(s_q=2),
        dict(s_q=8),
        dict(s_q=4096),
        dict(has_sink=True),
        dict(has_paged_kv=True, page_size=128, padded=True),
        dict(device_cc=(9, 0)),
        dict(device_cc=(10, 8)),
        dict(device_cc=(12, 0)),
        dict(device_cc=None),
        dict(is_mxfp8=False, is_fp8=True),
        dict(is_mxfp8=False, dtype=cudnn.data_type.BFLOAT16, dtype_o=None),
        dict(is_backward=True),
        dict(invalid="malformed"),
    ],
    ids=lambda over: ",".join(f"{k}={v}" for k, v in over.items()),
)
def test_one_deviation_from_the_domain_lets_the_backend_be_consulted(crashing_backend, over):
    """Outside the measured domain the backend is queried as usual -- with the guard ARMED (a crashing backend), so the
    deviation is what lets it through: more than one query row, a sink token (the backend plans), paged pools (the
    backend's own C++ validate declines them first), another device, per-tensor FP8 or half graphs, backward graphs,
    malformed graphs."""
    assert backend_guard(None, _facts(**over)) is None


def test_no_facts_no_guard():
    assert backend_guard(None, None) is None


def test_version_bound(monkeypatch):
    """The recorded fix version is the exact boundary: armed one below it, lifted at it; while none is recorded no version
    number lifts the guard (the installed library's own version decides for a real graph)."""
    if SQ1_MXFP8_PLANNING_CRASH_FIXED_IN is None:
        # Every known build crashes: a future version number does not lift the guard until one is measured clean.
        monkeypatch.setattr(cudnn, "backend_version", lambda: 99999)
        assert backend_guard(None, _facts()) is not None
    else:
        monkeypatch.setattr(cudnn, "backend_version", lambda: SQ1_MXFP8_PLANNING_CRASH_FIXED_IN)
        assert backend_guard(None, _facts()) is None
        monkeypatch.setattr(cudnn, "backend_version", lambda: SQ1_MXFP8_PLANNING_CRASH_FIXED_IN - 1)
        assert backend_guard(None, _facts()) is not None


def test_manifest_declares_the_guard_for_the_sdpa_forward_family():
    assert manifest.resolve_backend_guard(_FAMILY) is backend_guard
    assert all(manifest.resolve_backend_guard(f) is None for f in manifest.MANIFEST if f.name != "frost_sdpa_fwd")


# --- the planning sequence on real graphs (host-only: nothing is built) ---------------------------------------------


@pytest.mark.parametrize("bshd", [False, True])
def test_native_single_query_graph_plans_on_the_row_without_lowering_the_backend(rubin_host, monkeypatch, bshd):
    """The guarded graph plans on the FROST row alone: no C++ lowering happens (the tripwire would fail the test), the
    plan list holds the row's python plans only, and the guard's text is recorded as the backend's decline."""
    monkeypatch.setattr(pygraph, "_lower_backend_graph", lambda self: pytest.fail("the backend must not be lowered for a guarded graph"))
    g = _mxfp8_graph(bshd=bshd)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    names = _plan_names(g)
    assert names and all(n == _ROW or n.startswith(_ROW + "[") for n in names), names
    assert all(is_python_engine(p.engine_id) for p in g.plans)
    assert g._lowered_graph is None
    assert _GUARD in str(g._backend_declined) and _TAIL in str(g._backend_declined)


def test_unaligned_single_query_graph_declines_with_the_guard_text(rubin_host, monkeypatch):
    """A graph the row does not serve (unaligned K) and the guarded backend is not asked about: a typed decline whose text
    carries the guard's reason, with no C++ lowering on the way."""
    monkeypatch.setattr(pygraph, "_lower_backend_graph", lambda self: pytest.fail("the backend must not be lowered for a guarded graph"))
    g = _mxfp8_graph(bshd=False, unaligned_k=True)
    g.validate()
    g.build_operation_graph()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError) as exc:
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert _GUARD in str(exc.value) and _TAIL in str(exc.value), str(exc.value)
    # ... and the row's own reason, both sides in one message.
    assert "python engines declined:" in str(exc.value) and f"{_ROW}: k: native dense forward" in str(exc.value), str(exc.value)
    assert g._lowered_graph is None


def test_sink_single_query_graph_consults_the_backend(rubin_host, monkeypatch):
    """The guard's one exception, a sink token: the backend is lowered and queried as for any graph (the stub stands in
    for it and declines, so the row's plans are what remains)."""
    lowered = []

    def stub(self):
        lowered.append(True)
        raise cudnn_graph_not_supported("stub backend: declined")

    monkeypatch.setattr(pygraph, "_lower_backend_graph", stub)
    g = _mxfp8_graph(sink=True)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert lowered, "a sink graph is outside the guard's domain: the backend must be consulted"
    assert all(is_python_engine(p.engine_id) for p in g.plans)
    assert "stub backend: declined" in str(g._backend_declined)


def test_multi_query_graph_consults_the_backend(rubin_host, monkeypatch):
    lowered = []

    def stub(self):
        lowered.append(True)
        raise cudnn_graph_not_supported("stub backend: declined")

    monkeypatch.setattr(pygraph, "_lower_backend_graph", stub)
    g = _mxfp8_graph(sq=8)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert lowered


# --- the paths outside the planning sequence that create backend plans (host-only: nothing is built) ---------------


def test_explicit_backend_pin_on_a_guarded_graph_is_a_typed_decline(rubin_host, monkeypatch):
    """create_execution_plan(<backend engine id>, knobs) -- a replayed backend record -- on a guarded graph: the C++
    engine-config path crashes exactly as the heuristics query does (rc 139 measured on the board), so the append
    refuses with the guard's text before any C++ lowering; the plan list is unchanged and the row's plans still serve."""
    monkeypatch.setattr(pygraph, "_lower_backend_graph", lambda self: pytest.fail("the backend must not be lowered for a guarded graph"))
    g = _mxfp8_graph()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    before = list(g.plans)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError) as exc:
        g.create_execution_plan(16, {})  # the backend's eng16 record, as the sibling s_q == 8 contract reports it
    assert _GUARD in str(exc.value) and _TAIL in str(exc.value), str(exc.value)
    assert list(g.plans) == before and g._lowered_graph is None
    assert _GUARD in str(g._backend_declined)
    assert all(is_python_engine(p.engine_id) for p in g.plans)


def test_late_backend_lowering_on_a_guarded_graph_is_a_typed_decline(rubin_host, monkeypatch):
    """A classic query that lowers the backend late and runs its heuristics (_lower_backend_plan) refuses on a guarded
    graph instead of running the crashing query."""
    monkeypatch.setattr(pygraph, "_lower_backend_graph", lambda self: pytest.fail("the backend must not be lowered for a guarded graph"))
    g = _mxfp8_graph()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=_GUARD):
        g._lower_backend_plan()
    assert g._lowered_graph is None


def test_a_guarded_graph_whose_row_fails_to_build_names_the_guard_in_the_exhaustion_error(rubin_host, monkeypatch):
    """When every FROST plan of a guarded graph declines at BUILD, the exhaustion error carries the recorded backend
    decline (the guard's reason) next to the build failures, so the reader sees why no backend plan was on the list."""
    from cudnn.sdpa.fwd.engine import FrostSdpaFwdEngine

    def declining_build(self, graph, cfg, ctx):
        raise NotImplementedError(f"{self.name}: stub build decline")

    monkeypatch.setattr(pygraph, "_lower_backend_graph", lambda self: pytest.fail("the backend must not be lowered for a guarded graph"))
    monkeypatch.setattr(FrostSdpaFwdEngine, "build_plan", declining_build)
    g = _mxfp8_graph()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError) as exc:
        g.build_plans()
    msg = str(exc.value)
    assert "no plan in the list could be built" in msg and "stub build decline" in msg and _GUARD in msg, msg


# ---------------------------------------------------------------------------------------------- backend engine guard
# The native SM100 / SM107 forward engines (10, 18) build O from the K buffer when d_qk <= 128 < d_v
# (NVIDIA/cudnn-frontend#2401). sdpa/fwd/backend_guard.py::backend_engine_guard keeps those two engines off such graphs
# on every backend from 9.23.1 until a recorded fix; the backend's SM80 engine (8) serves them.

from cudnn.sdpa.fwd import backend_guard as _bg  # noqa: E402


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("dims", [(128, 256), (64, 256), (128, 192)])
def test_wide_v_engine_guard_arms_on_blackwell_from_9_23_1(monkeypatch, cc, dims):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92301)
    monkeypatch.setattr(cudnn, "backend_version_string", lambda: "9.23.1")
    facts = _facts(d_qk=dims[0], d_v=dims[1], dtype=cudnn.data_type.BFLOAT16, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=False, device_cc=cc, s_q=512)
    excluded = _bg.backend_engine_guard(None, facts)
    assert sorted(excluded) == [10, 18]
    assert all("#2401" in reason for reason in excluded.values())


@pytest.mark.parametrize(
    "over",
    [
        dict(d_qk=128, d_v=128),  # d_v within the K buffer's width
        dict(d_qk=192, d_v=256),  # d_qk past 128: the chunk loop stages V in smem_K, which the descriptor names
        dict(d_qk=256, d_v=128),
        dict(device_cc=(9, 0)),
        dict(device_cc=(12, 0)),
        dict(is_backward=True),
    ],
)
def test_wide_v_engine_guard_leaves_other_graphs_alone(monkeypatch, over):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 93000)
    base = dict(d_qk=128, d_v=256, dtype=cudnn.data_type.BFLOAT16, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=False, device_cc=(10, 0), s_q=512)
    assert _bg.backend_engine_guard(None, _facts(**{**base, **over})) == {}


def test_wide_v_engine_guard_version_bounds(monkeypatch):
    facts = _facts(d_qk=128, d_v=256, dtype=cudnn.data_type.BFLOAT16, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=False, device_cc=(10, 0), s_q=512)
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92300)  # 9.23.0.39 measured correct
    assert _bg.backend_engine_guard(None, facts) == {}
    monkeypatch.setattr(cudnn, "backend_version", lambda: 93100)
    monkeypatch.setattr(_bg, "WIDE_V_WRONG_O_FIXED_IN", 93100)
    assert _bg.backend_engine_guard(None, facts) == {}
    monkeypatch.setattr(_bg, "WIDE_V_WRONG_O_FIXED_IN", 93101)
    assert sorted(_bg.backend_engine_guard(None, facts)) == [10, 18]


def test_wide_v_engine_guard_is_declared_by_the_forward_family():
    assert manifest.resolve_backend_engine_guard(_FAMILY) is _bg.backend_engine_guard


def _wide_v_graph():
    import torch

    b, hq, hkv, s, dqk, dv = 2, 16, 4, 1024, 128, 256
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, hq, s, dqk, device="cuda", generator=gen).to(torch.bfloat16)
    k = torch.randn(b, hkv, s, dqk, device="cuda", generator=gen).to(torch.bfloat16)
    v = torch.randn(b, hkv, s, dv, device="cuda", generator=gen).to(torch.bfloat16)
    o = torch.empty(b, hq, s, dv, device="cuda", dtype=torch.bfloat16)
    g = pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q, K, V = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    O, _ = g.sdpa(q=Q, k=K, v=V, attn_scale=1 / math.sqrt(dqk), use_causal_mask=True, generate_stats=False)
    O.set_output(True).set_dim(o.size()).set_stride(o.stride())
    g.validate()
    g.build_operation_graph()
    r = hq // hkv
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float().repeat_interleave(r, 1), v.float().repeat_interleave(r, 1), is_causal=True, scale=1 / math.sqrt(dqk)
    )
    return g, {Q: q, K: k, V: v, O: o}, o, ref


def _run_plan(g, pack, index):
    import torch

    g.select_plan(index)
    g.check_support()
    g.build_plans()
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8))
    torch.cuda.synchronize()


def _requires_wide_v_device():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the wide-V engine guard covers cc 10.x devices")
    if cudnn.backend_version() < _bg.WIDE_V_WRONG_O_SINCE:
        pytest.skip(f"cuDNN {cudnn.backend_version()} predates the defect (9.23.1)")


def test_wide_v_graph_never_offers_the_wrong_engines_and_its_default_is_right(monkeypatch):
    _requires_wide_v_device()
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    g, pack, o, ref = _wide_v_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    armed = bool(_bg.backend_engine_guard(g, g._facts_for(graph_analyzer.analyze)))
    offered = [int(p.engine_id) for p in g.plans if not is_python_engine(p.engine_id)]
    if armed:
        assert not set(offered) & set(_bg.WIDE_V_WRONG_O_ENGINES), offered
        with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="#2401"):
            g.create_execution_plan(10, {})
    _run_plan(g, pack, 0)
    assert (o.float() - ref).abs().max().item() < 0.05


def test_wide_v_guard_withholds_the_opensource_delegate(monkeypatch):
    """OPENSOURCE in the heuristics adds a delegating entry that lets C++ pick from its own list, engines 10 / 18
    included. An armed guard withholds it, and every backend plan offered returns the right O."""
    _requires_wide_v_device()
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    g, pack, o, ref = _wide_v_graph()
    g.create_execution_plans([cudnn.heur_mode.OPENSOURCE, cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    if not _bg.backend_engine_guard(g, g._facts_for(graph_analyzer.analyze)):
        pytest.skip(f"the wide-V guard is not armed on cuDNN {cudnn.backend_version()}")
    backend = [i for i, p in enumerate(g.plans) if not is_python_engine(p.engine_id)]
    assert backend and all(int(g.plans[i].engine_id) != BACKEND_HEURISTIC_ENGINE_ID for i in backend)
    for index in backend:
        _run_plan(g, pack, index)
        assert (o.float() - ref).abs().max().item() < 0.05, g.get_plan_name_at_index(index)


def test_wide_v_engine_guard_is_current(monkeypatch):
    """Detector for WIDE_V_WRONG_O_FIXED_IN: with the guard lifted, the installed backend's engine 10 / 18 plans for the
    dense causal 128/256 graph run against torch's fp32 SDPA. While no fix version is recorded they must still be wrong (a
    right O fails with the version to record); at or above a recorded version they must be right."""
    _requires_wide_v_device()
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    recorded = _bg.WIDE_V_WRONG_O_FIXED_IN
    monkeypatch.setattr(_bg, "WIDE_V_WRONG_O_FIXED_IN", 0)  # lift the guard
    g, pack, o, ref = _wide_v_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    native = [i for i, p in enumerate(g.plans) if int(p.engine_id) in _bg.WIDE_V_WRONG_O_ENGINES]
    if not native:
        pytest.skip(f"cuDNN {cudnn.backend_version()} offers neither engine 10 nor 18 for this graph")
    errors = []
    for index in native:
        _run_plan(g, pack, index)
        errors.append((int(g.plans[index].engine_id), (o.float() - ref).abs().max().item()))
    wrong = any(err > 0.05 for _, err in errors)
    if recorded is None:
        assert wrong, (
            f"cuDNN {cudnn.backend_version()} engines {errors} return the right O: record WIDE_V_WRONG_O_FIXED_IN = "
            f"{cudnn.backend_version()} in sdpa/fwd/backend_guard.py"
        )
    elif cudnn.backend_version() >= recorded:
        assert not wrong, f"cuDNN {cudnn.backend_version()} engines {errors} at or above the recorded fix {recorded}"
