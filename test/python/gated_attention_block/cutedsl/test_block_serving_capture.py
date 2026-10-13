# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block under a serving stack's CUDA-graph defaults, and its projection GEMMs without the opt-in flag.

A serving stack captures the block's FIRST execute of a fresh declaration inside ``torch.cuda.graph`` on a NEW stream with
ZERO eager warm-ups.  What the block owes it, pinned here:

* on the forced-tile JIT route (every projection whose ``N`` the 256-wide tile divides) that first captured execute launches
  nothing, and the replay is BITWISE a fresh eager run -- under ``torch.cuda.set_sync_debug_mode("error")``;
* the GRAPH GEMM route (no forced tile) creates a cuDNN handle at its first execute per stream -- under capture that is a
  typed ``RuntimeError`` naming the route and the remedy, and NO handle is created inside the capture;
* ``compile()`` under capture is refused typed (the GEMM plan builds a cuDNN graph and JIT-compiles a kernel there);
* the serving declaration ``build_proj_gemm(capture_safe=True)`` never enters the ranked list -- no backend plan, no
  FROST-engine compile, whether the opt-in flag is set or not -- so it ends on the JIT route by construction, and a refused
  forced compile is a typed ``NotImplementedError`` naming the tile and the reason, never the warning + graph-heuristic
  fallback (the flag-set case is the one that matters: there a plain declaration's ``build_plans`` compiles the same kernel
  inside the FROST GEMM engine and a refusal surfaces raw from the pin);
* a dense bf16 projection at every served geometry compiles with ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` UNSET (the flag is read
  live and a host module must never set it) on the ``jit-only`` route -- no ``frost_gemm`` ever enters the ranked list, no
  cuDNN handle is created -- while a shape WITHOUT a forced tile keeps today's typed decline;
* ``cudnn.frost.capture`` is the ONE spelling of the status query and the refusal; the grouped GEMM's debug validation
  delegates to it and keeps its own remedy wording.

The accept cells need the SM107 (Rubin) GPU the block targets (``requires_rubin``); the helper, refusal and decline cells run
on any CUDA device (or on the host, where marked).
"""

import logging
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block serving-capture tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.kernels import proj_gemm  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import _FORCED_TILE_NAME, _NoFrostPlan, build_proj_gemm, run_proj_gemm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, make_inputs  # noqa: E402

_FLAG = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"
_SM107 = (10, 7)

requires_rubin = pytest.mark.requires_rubin  # registered by this directory's conftest: skipped wherever the device is not SM107
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device (a stream to capture)")

# The block geometry the capture cells run: every projection on the forced tile (n_qkvg = (2*8 + 2*2) * 256 = 5120, h_q*d_head
# = 2048, d_model 512 -- all multiples of 256), small enough to compile and run in seconds.
_GEOM_KW = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)

# The served geometries of the two models and their tensor-parallel shards, as (h_q, h_kv, d_model): n_qkvg = (2*h_q + 2*h_kv) * 256
# is 17408 / 13312 / 8704 / 4608 / 2560 -- the 32/2 full layer, the 24/2 layer at d_model 2560, and the 16/1, 8/1, 4/1 shards.
# Every N and K the two projections see (n_qkvg, d_model, h_q*d_head) is a multiple of 256: the forced tile applies to all of them.
_SERVED_GEOMETRIES = [
    pytest.param(32, 2, 4096, id="32-2-d4096-n17408"),
    pytest.param(24, 2, 2560, id="24-2-d2560-n13312"),
    pytest.param(16, 1, 4096, id="16-1-d4096-n8704"),
    pytest.param(8, 1, 4096, id="8-1-d4096-n4608"),
    pytest.param(4, 1, 4096, id="4-1-d4096-n2560"),
]


def _n_qkvg(h_q, h_kv, d_head=256):
    return (2 * h_q + 2 * h_kv) * d_head


def _make_block(geom_kw=_GEOM_KW, batch=1, seq_len=256, dtype=torch.bfloat16, seed=0, **kw):
    """A declared bf16 block plus the inputs it was declared for (not compiled)."""
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    inp = make_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len, dtype=dtype, seed=seed)
    out = torch.empty(batch, seq_len, block_geom.d_model, device="cuda", dtype=dtype)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom, **kw)
    return blk, inp, out


def _execute(blk, inp, out, ws):
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws)


class _Capture:
    """A low-level stream capture on a fresh side stream, with one node of its own so it is never empty, that survives a typed
    error raised inside it: ``capture_end`` runs in ``__exit__`` whatever happened (the body's exception is kept -- with
    ``torch.cuda.graph`` a ``capture_end`` error would REPLACE it), and the graph is reset explicitly (an instantiated graph left
    to the cyclic GC can invalidate a LATER test's capture).  ``torch.cuda.graph`` is used where a REPLAY is wanted."""

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.graph = torch.cuda.CUDAGraph()
        self._buf = torch.zeros(1, device="cuda")
        self._ctx = None

    def __enter__(self):
        torch.cuda.synchronize()
        self._ctx = torch.cuda.stream(self.stream)
        self._ctx.__enter__()
        self.graph.capture_begin()
        self._buf.add_(1)  # a node of this capture's own: never an empty capture
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.graph.capture_end()
        except Exception:  # noqa: BLE001 -- the body's error is the one under test; a capture_end error must not replace it
            if exc is None:
                raise
        finally:
            self._ctx.__exit__(None, None, None)
            self.graph.reset()
        return False


# ---------------------------------------------------------------------------
# cudnn.frost.capture -- the one helper, its message, and the grouped GEMM's delegation
# ---------------------------------------------------------------------------


def test_shared_helper_names_route_and_remedy_and_the_grouped_gemm_delegates_to_it(monkeypatch):
    """Host-only (no device): the status query is stubbed to 'capturing'.  The shared refusal names the ROUTE (``what``) and a
    REMEDY (the default one: compile at warm-up / one eager execute on the capture stream) and never the grouped GEMM's env var;
    the grouped GEMM's ``_check_not_capturing`` keeps its name on its module (the stub-by-name test there still reaches it) and
    its own remedy wording, through the shared helper."""
    from cudnn.frost import capture
    from cudnn.gemm.cutedsl.grouped import backend_utils

    seen = []
    monkeypatch.setattr(capture, "is_capturing", lambda stream: seen.append(int(stream)) or True)
    with pytest.raises(RuntimeError) as e:
        capture.check_not_capturing(0x1234, "frost gemm graph route (cuDNN handle creation)")
    msg = str(e.value)
    assert "frost gemm graph route" in msg and "0x1234" in msg, msg
    assert "warm-up" in msg and "eager execute on the capture stream" in msg, msg
    assert backend_utils.DEBUG_VALIDATE_DEVICE_VALUES_ENV not in msg, msg
    with pytest.raises(RuntimeError) as e2:
        backend_utils._check_not_capturing(0x1234, "validating padded_offsets values")
    msg2 = str(e2.value)
    assert msg2.startswith("validating padded_offsets values: refused under CUDA-graph capture"), msg2
    assert f"{backend_utils.DEBUG_VALIDATE_DEVICE_VALUES_ENV}=1 reads device values back to the host" in msg2, msg2
    assert "unset it for captured graphs" in msg2, msg2
    assert seen == [0x1234, 0x1234], seen
    # not capturing: a no-op for both spellings
    monkeypatch.setattr(capture, "is_capturing", lambda stream: False)
    capture.check_not_capturing(0, "anything")
    backend_utils._check_not_capturing(0, "anything")


@requires_cuda
def test_capture_state_reads_a_live_capture_and_the_refusal_fires_inside_it():
    """A real capture on a fresh side stream: inside it the stream reads ``(True, id)`` and the refusal fires (typed, with the
    route and the remedy); another stream reads not-capturing; after the capture the stream reads not-capturing again.  The
    status query is legal inside the capture (the capture is not invalidated by it: ``capture_end`` succeeds)."""
    from cudnn.frost.capture import capture_state, check_not_capturing, is_capturing

    other = torch.cuda.Stream()
    with _Capture() as cap:
        capturing, cid = capture_state(cap.stream.cuda_stream)
        assert capturing and isinstance(cid, int), (capturing, cid)
        assert is_capturing(cap.stream.cuda_stream)
        assert not is_capturing(other.cuda_stream)
        with pytest.raises(RuntimeError, match="frost gemm graph route.*refused under CUDA-graph capture.*eager execute on the capture stream"):
            check_not_capturing(cap.stream.cuda_stream, "frost gemm graph route (cuDNN handle creation)")
    assert capture_state(cap.stream.cuda_stream) == (False, None)
    check_not_capturing(cap.stream.cuda_stream, "frost gemm graph route")  # a no-op outside the capture


# ---------------------------------------------------------------------------
# compile() under capture, and the serving declaration's typed declines (any CUDA device)
# ---------------------------------------------------------------------------


@requires_cuda
def test_build_proj_gemm_is_refused_under_capture_before_any_cudnn_graph_exists(monkeypatch):
    """``build_proj_gemm`` inside a capture is a typed ``RuntimeError`` naming the compile route and the remedy, raised BEFORE the
    ``cudnn.pygraph`` constructor (whose handle creation would be the illegal call): the constructor is stubbed to fail loudly."""
    import cudnn

    def no_graph(*a, **k):
        raise AssertionError("cudnn.pygraph was constructed inside a capture")

    monkeypatch.setattr(cudnn, "pygraph", no_graph)
    with _Capture() as cap:
        assert torch.cuda.current_stream().cuda_stream == cap.stream.cuda_stream
        with pytest.raises(RuntimeError, match="cap_test: FROST GEMM compile.*refused under CUDA-graph capture.*compile\\(\\) at warm-up"):
            build_proj_gemm(m=256, k=512, n=512, dtype=torch.bfloat16, label="cap_test")


def test_capture_safe_refuses_the_graph_route_at_compile_time_before_any_backend_call():
    """The serving declaration cannot end on the graph route: a shape the forced tile does not divide (``n=224``) is a typed
    ``NotImplementedError`` naming the way out, and ``pin_frost=False`` (the graph route by request) a ``ValueError`` -- both raised
    before the ``cudnn.pygraph`` constructor and before any backend call (the capture check is a no-op without a device), so they
    hold on a host without a GPU too."""
    with pytest.raises(ValueError, match="capture_safe=True needs the forced-tile JIT route.*pin_frost=False"):
        build_proj_gemm(m=256, k=512, n=512, dtype=torch.bfloat16, label="cs_graph", pin_frost=False, capture_safe=True)
    with pytest.raises(NotImplementedError, match="capture_safe=True needs the forced-tile JIT route.*graph route.*an N that 256 divides"):
        build_proj_gemm(m=256, k=512, n=224, dtype=torch.bfloat16, label="cs_224", capture_safe=True)


def _never_ranked(names):
    """Stands in for ``proj_gemm._frost_plan_index`` where a ranked list must never be inspected (a ``capture_safe`` plan)."""
    raise AssertionError(f"a capture_safe plan inspected a ranked list: {names}")


def _set_flag(monkeypatch, flag):
    if flag == "unset":
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, "1")


@requires_cuda
@pytest.mark.parametrize("flag", ["set", "unset"])
def test_forced_refusal_under_capture_safe_is_typed_at_compile_and_never_a_warning(monkeypatch, caplog, flag):
    """A forced-tile compile the compiler refuses (``jit_from_cudnn_graph`` stubbed to raise) under ``capture_safe=True`` is a
    typed ``NotImplementedError`` at ``compile()`` naming the refused tile and the compiler's reason, and NO warning is logged --
    the plain declaration's warning + graph-heuristic fallback (which would create a handle inside a capture) never happens.
    Holds with the opt-in flag SET (the suite's default) and UNSET alike, because a ``capture_safe`` plan never enters the ranked
    list: with the flag set and ``frost_gemm`` on the list, a plain declaration's ``build_plans`` compiles the SAME
    ``jit_from_cudnn_graph`` inside the FROST GEMM engine and the stub's error surfaces RAW from the pin (the Rubin failure this
    cell first exposed); the serving declaration's only compile is the driver's forced-tile one."""
    from cudnn.gemm.frost import compiler

    _set_flag(monkeypatch, flag)

    def refused(graph, config=None):
        raise RuntimeError("forced refusal for the test")

    monkeypatch.setattr(compiler, "jit_from_cudnn_graph", refused)
    monkeypatch.setattr(proj_gemm, "_frost_plan_index", _never_ranked)
    caplog.set_level(logging.DEBUG, logger=proj_gemm.__name__)
    with pytest.raises(NotImplementedError) as e:
        build_proj_gemm(m=256, k=512, n=512, dtype=torch.bfloat16, label=f"cs_refused_{flag}", capture_safe=True)
    msg = str(e.value)
    assert "capture_safe=True" in msg and _FORCED_TILE_NAME in msg, msg
    assert "forced refusal for the test" in msg and "graph-heuristic fallback is refused" in msg, msg
    assert isinstance(e.value.__cause__, RuntimeError), repr(e.value.__cause__)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == proj_gemm.__name__]
    assert not warnings, [r.getMessage() for r in warnings]


@requires_cuda
def test_dense_plan_without_the_flag_and_a_refused_compile_keeps_the_typed_decline(monkeypatch):
    """With the flag UNSET the dense jit-only arm has no backend plan to fall back onto: a refused forced compile re-raises the
    typed decline (``_NoFrostPlan``, naming the flag) with the compiler's refusal chained -- never a plan with no kernel."""
    from cudnn.gemm.frost import compiler

    monkeypatch.delenv(_FLAG, raising=False)

    def refused(graph, config=None):
        raise RuntimeError("forced refusal for the test")

    monkeypatch.setattr(compiler, "jit_from_cudnn_graph", refused)
    with pytest.raises(_NoFrostPlan) as e:
        build_proj_gemm(m=256, k=512, n=512, dtype=torch.bfloat16, label="dense_noflag_refused")
    msg = str(e.value)
    assert _FLAG in msg and "forced refusal for the test" in msg and "no route can serve this plan" in msg, msg


@requires_cuda
def test_a_shape_without_a_forced_tile_keeps_todays_decline_without_the_flag(monkeypatch):
    """``n=224``: no forced tile, so the graph route IS the plan and an unset flag is the same typed decline as before -- the
    message names the flag and refuses to fall back onto a backend plan."""
    monkeypatch.delenv(_FLAG, raising=False)
    with pytest.raises(_NoFrostPlan, match="CUDNN_FRONTEND_ENABLE_FROST_ENGINES.*Refusing to fall back"):
        build_proj_gemm(m=256, k=512, n=224, dtype=torch.bfloat16, label="no_tile_noflag")


# ---------------------------------------------------------------------------
# Rubin accept cells: the flag-free dense route, and the block under a zero-warm-up capture
# ---------------------------------------------------------------------------


def _spy_ranked_list(monkeypatch):
    """Record every ranked list the pin inspects (``_frost_plan_index`` is the ONE call the pin makes with it)."""
    seen = []
    real = proj_gemm._frost_plan_index

    def spy(names):
        seen.append(list(names))
        return real(names)

    monkeypatch.setattr(proj_gemm, "_frost_plan_index", spy)
    return seen


def _assert_jit_only(plan, seen):
    assert plan.route == "jit-only" and plan.jit is not None, (plan.label, plan.route, plan.jit)
    assert plan.tile_config_name == _FORCED_TILE_NAME, (plan.label, plan.tile_config_name)
    assert seen, "the pin never inspected a ranked list"
    for names in seen:
        assert not any(n == "frost_gemm" or n.startswith("frost_gemm[") for n in names), names


@requires_rubin
@pytest.mark.parametrize("h_q, h_kv, d_model", _SERVED_GEOMETRIES)
def test_dense_bf16_projections_compile_and_run_without_the_flag_at_every_served_geometry(monkeypatch, h_q, h_kv, d_model):
    """Both projection GEMMs of every served geometry, built with the flag UNSET: the ``jit-only`` route at the forced tile, no
    ``frost_gemm`` in any ranked list the pin saw, no cuDNN handle created, and the kernel RUNS (cos > 0.999 vs torch, finite)."""
    monkeypatch.delenv(_FLAG, raising=False)
    seen = _spy_ranked_list(monkeypatch)
    handles_before = dict(proj_gemm._GRAPH_HANDLES)
    m, d_head = 128, 256
    for label, k, n in (("qkv_gate_proj", d_model, _n_qkvg(h_q, h_kv, d_head)), ("out_proj", h_q * d_head, d_model)):
        plan = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label=f"{label}_{h_q}_{h_kv}")
        _assert_jit_only(plan, seen)
        torch.manual_seed(0)
        a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1).to(torch.bfloat16)
        w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
        run_proj_gemm(plan, a, w, out, ws)
        torch.cuda.synchronize()
        ref = torch.nn.functional.linear(a.float(), w.float())
        assert torch.isfinite(out.float()).all(), label
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
        assert cos > 0.999, (label, cos)
    assert dict(proj_gemm._GRAPH_HANDLES) == handles_before, "the jit-only route created a cuDNN handle"


@requires_rubin
@pytest.mark.parametrize("h_q, h_kv, d_model", _SERVED_GEOMETRIES)
def test_bf16_block_compiles_without_the_flag_at_every_served_geometry(monkeypatch, h_q, h_kv, d_model):
    """The whole bf16 block at each served geometry (B=1, S=128), declared and compiled with the flag UNSET: both projections on the
    ``jit-only`` route at the forced tile, no ``frost_gemm`` in any ranked list, one execute finite."""
    monkeypatch.delenv(_FLAG, raising=False)
    seen = _spy_ranked_list(monkeypatch)
    blk, inp, out = _make_block(dict(d_model=d_model, h_q=h_q, h_kv=h_kv, d_head=256, rope_dim=64), batch=1, seq_len=128)
    blk.check_support()
    blk.compile()
    _assert_jit_only(blk._proj._plan, seen)
    _assert_jit_only(blk._out_proj._plan, seen)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    out.fill_(float("nan"))
    _execute(blk, inp, out, ws)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()


@requires_rubin
@pytest.mark.parametrize("flag", ["set", "unset"])
def test_capture_safe_plan_never_enters_the_ranked_list_whatever_the_flag(monkeypatch, flag):
    """The serving declaration with the opt-in flag SET (the suite's default) and UNSET: the plan's graph holds NO ranked list
    (``create_execution_plans`` never ran: ``plan.graph.plans == []``), the pin never inspected one, the route is ``jit-only`` at
    the forced tile, ``workspace_bytes`` is the JIT's own, no cuDNN handle exists for it, and the kernel RUNS (cos > 0.999 vs
    torch).  The flag-set half is the literal acceptance: with ``frost_gemm`` on a ranked list a plain declaration pins
    ``graph+jit`` and compiles once more inside the engine; the serving declaration does neither."""
    _set_flag(monkeypatch, flag)
    monkeypatch.setattr(proj_gemm, "_frost_plan_index", _never_ranked)
    handles_before = dict(proj_gemm._GRAPH_HANDLES)
    m, k, n = 128, 512, 512
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label=f"cs_{flag}", capture_safe=True)
    assert plan.capture_safe and plan.route == "jit-only" and plan.jit is not None, (plan.route, plan.jit)
    assert plan.tile_config_name == _FORCED_TILE_NAME, plan.tile_config_name
    assert plan.graph.plans == [], "the serving declaration built a ranked list"
    assert plan.workspace_bytes == max(int(getattr(plan.jit, "workspace_bytes", 0) or 0), 1), plan.workspace_bytes
    torch.manual_seed(0)
    a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    run_proj_gemm(plan, a, w, out, ws)
    torch.cuda.synchronize()
    ref = torch.nn.functional.linear(a.float(), w.float())
    assert torch.isfinite(out.float()).all()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.999, cos
    assert dict(proj_gemm._GRAPH_HANDLES) == handles_before, "the capture_safe plan created a cuDNN handle"


@requires_rubin
def test_first_execute_captured_with_zero_warmups_replays_bitwise_eager(monkeypatch):
    """vLLM's defaults: a FRESH bf16 declaration, compiled with the opt-in flag UNSET (the serving stack's configuration: both
    projections on the ``jit-only`` route -- on the tree before this route that compile is the ``_NoFrostPlan`` decline), and its
    FIRST execute captured inside ``torch.cuda.graph`` on a NEW stream with ZERO eager warm-ups, the execute under
    ``set_sync_debug_mode("error")`` (no host sync inside the capture).  The capture launches nothing (the NaN sentinel survives
    it), the replay is finite and BITWISE a fresh eager run on the default stream.  The forced-tile JIT route is what makes this
    possible: no handle, no lazy resource at the first execute."""
    monkeypatch.delenv(_FLAG, raising=False)
    blk, inp, out = _make_block()
    blk.check_support()
    blk.compile()
    assert blk._proj._plan.jit is not None and blk._out_proj._plan.jit is not None, "the capture cell needs the forced-tile JIT route"
    assert blk._proj._plan.route == "jit-only" and blk._out_proj._plan.route == "jit-only", (blk._proj._plan.route, blk._out_proj._plan.route)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    out.fill_(float("nan"))
    ws.fill_(0xFF)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()  # NEW: the block never ran on it, nothing ran on it
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                _execute(blk, inp, out, ws)  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(out).all(), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        assert torch.isfinite(replayed.float()).all()
        out.fill_(float("nan"))
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        _execute(blk, inp, out, ws)  # eager, on the default stream
        torch.cuda.synchronize()
        assert torch.equal(out, replayed), "the replay of the captured first execute differs from the eager run"
        # new inputs through the captured pointers replay to the eager result over them
        inp2 = make_inputs(RefGeometry(**_GEOM_KW), batch=1, seq_len=256, dtype=torch.bfloat16, seed=1)
        inp["h"].copy_(inp2["h"])
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2 = out.clone()
        out.fill_(float("nan"))
        _execute(blk, inp, out, ws)
        torch.cuda.synchronize()
        assert torch.equal(out, replayed2), "the replay over new inputs differs from the eager run over them"
    finally:
        graph.reset()


@requires_rubin
def test_graph_route_first_execute_under_capture_is_refused_typed_and_creates_no_handle(monkeypatch):
    """The forced tile disabled (``_forced_tile_config`` -> None): both projections run the GRAPH route, whose first execute per
    stream creates a cuDNN handle.  Captured on a NEW stream with no warm-up, the first execute is a typed ``RuntimeError`` naming
    the graph route and the remedy, raised before any handle exists for that stream -- and the capture itself survives (the
    refusal is a status query, not an illegal call).  Eagerly, on the same new stream afterwards, the block runs."""
    monkeypatch.setattr(proj_gemm, "_forced_tile_config", lambda n: None)
    blk, inp, out = _make_block()
    blk.check_support()
    blk.compile()
    assert blk._proj._plan.route == "graph" and blk._proj._plan.jit is None, (blk._proj._plan.route, blk._proj._plan.jit)
    assert blk._out_proj._plan.route == "graph" and blk._out_proj._plan.jit is None
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    dev = torch.cuda.current_device()
    with _Capture() as cap:
        key = (dev, int(cap.stream.cuda_stream))
        assert key not in proj_gemm._GRAPH_HANDLES, "a fresh stream must have no cached handle"
        with pytest.raises(RuntimeError, match="frost gemm graph route.*refused under CUDA-graph capture.*eager execute") as e:
            _execute(blk, inp, out, ws)
        assert key not in proj_gemm._GRAPH_HANDLES, "a handle was created inside the capture"
    assert "capture_safe" in str(e.value), str(e.value)  # the remedy names the serving declaration too
    # eager on that same (now idle) stream: the handle is created outside any capture and the block runs
    out.fill_(float("nan"))
    with torch.cuda.stream(cap.stream):
        _execute(blk, inp, out, ws)
    torch.cuda.synchronize()
    assert key in proj_gemm._GRAPH_HANDLES and torch.isfinite(out.float()).all()


@requires_rubin
def test_block_compile_under_capture_is_refused_typed():
    """``GatedAttentionBlockFwd.compile()`` inside a capture on a fresh stream: the first stage to build a GEMM plan refuses typed
    (the compile route, the remedy), before any cuDNN graph exists."""
    blk, inp, out = _make_block()
    blk.check_support()
    with _Capture():
        with pytest.raises(RuntimeError, match="FROST GEMM compile.*refused under CUDA-graph capture.*compile\\(\\) at warm-up"):
            blk.compile()
