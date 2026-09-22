# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing, Rule 8 and CUDA-graph behaviour of the sm90 KDA engines.

The numerics of ``kda_hopper_cuda`` are covered by the shared ``backend``
fixture in ``test_la.py`` (``hopper_cuda``). What is covered here is everything
that fixture cannot reach:

* that the KDA heuristics hook actually makes it the default on Hopper, since
  that is a default-behaviour change for every sm90 KDA user;
* that ranking it first does NOT change who is ELIGIBLE. ``_build_plan_at()``
  builds a ranked entry without re-affirming ``check_support``, so a graph this
  128-only engine rejects must never reach it. When the hook first landed it
  did, and a head-dim-64 graph launched the kernel on 64-dim tensors and died
  with an illegal memory access. That is what ``test_declined_graph_falls_through``
  pins;
* that the path captures into a CUDA graph. Per-call host dispatch is ~130 us
  against a ~60 us kernel, so capture is how the overhead is actually removed
  (measured 143 us -> 2.8 us of host time), and capture breaks silently the
  moment anything on the path allocates or synchronises per call;
* Rule 8, for BOTH sm90 engines (``kda_hopper_cuda`` and the CuTe DSL
  ``kda_hopper``): the plans own no device memory. Every scratch byte -- the
  final state the kernel always writes, the zero seed for an absent
  ``initial_state``, the DSL kernel's scratch, the dtype/layout staging -- is
  declared by ``get_workspace_size()`` and carved from the caller's workspace;
  execute allocates nothing and never synchronises; a missing workspace is the
  contract error, not an allocation; an UNDECLARED padded input raises rather
  than being silently repacked.
"""

import pytest
import torch

import cudnn  # noqa: F401 -- import-order requirement, see test/python/conftest.py
from cudnn.frost.workspace import align_up
from cudnn.linear_attention import kimi_delta_attention
from cudnn.linear_attention.ops.common import select_plan
from cudnn.linear_attention.ops.kda import build_bprop_graph, build_fprop_graph

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

ENGINE = "kda_hopper_cuda"
DSL_ENGINE = "kda_hopper"
HOPPER = (9, 0)
SCALE = 128**-0.5
CUDNN_DTYPE = {
    torch.float32: cudnn.data_type.FLOAT,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.int32: cudnn.data_type.INT32,
    torch.int64: cudnn.data_type.INT64,
}
# (initial_state declared, final_state requested)
STATE_PORTS = {"both": (True, True), "no_fs": (True, False), "no_s0": (False, True), "neither": (False, False)}
# (g dtype, cu_seqlens dtype, state dtype); the second is what FlashInfer / vLLM hand cuDNN
DTYPES = {"fp32_int32": (torch.float32, torch.int32, torch.float32), "bf16_gate_int64_cu_bf16_state": (torch.bfloat16, torch.int64, torch.bfloat16)}


def _hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == HOPPER


requires_hopper = pytest.mark.skipif(not _hopper(), reason="the sm90 CUDA KDA engine needs Hopper")


def make_case(T=2048, H=12, N=1, dim=128, lower_bound=-5.0, seed=7):
    """A production-gate KDA case: L2-normalised q/k, gate on (-5, 0), non-zero
    initial state. The gate range matters -- a kernel tuned at -1 can be 100%
    NaN at -5."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.float32)
    k = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.float32)
    q = (q / q.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    k = (k / k.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    v = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    raw = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.float32)
    gate = (lower_bound * torch.sigmoid(raw)).contiguous()
    beta = torch.sigmoid(torch.randn(T, H, generator=g, device="cuda", dtype=torch.float32)).contiguous()
    state = (torch.randn(N, H, dim, dim, generator=g, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    span = T // N
    bounds = [0]
    for i in range(N):
        bounds.append(bounds[-1] + (T - span * (N - 1) if i == N - 1 else span))
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    return q, k, v, gate, beta, cu, state


def launched_kernels(fn):
    """Device kernels one call launches, busiest first."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    return [e.key for e in sorted(events, key=lambda e: -e.self_device_time_total)]


# ---------------------------------------------------------------------------
# Graph-API scaffolding: the public pygraph route, which is how FlashInfer
# drives cuDNN and the only route that declares dtypes and strides.
# ---------------------------------------------------------------------------


def pin(graph, plan_name):
    """Pin ``plan_name`` and build its plan; an engine that declines waives the test."""
    try:
        select_plan(graph, plan_name)
    except (cudnn.cudnnGraphNotSupportedError, NotImplementedError) as exc:
        pytest.skip(f"{plan_name} declined: {exc}")
    graph.build_plans()
    return graph


def fwd_graph(plan_name, case, *, seed=True, want_state=True, g_dtype=torch.float32, cu_dtype=torch.int32, state_dtype=torch.float32):
    """A single-node KDA forward graph pinned to ``plan_name`` and its variant
    pack: ``(graph, pack, o, final_state_or_None)``."""
    q, k, v, g, beta, cu, s0 = case
    T, H, _ = q.shape
    N = cu.shape[0] - 1
    graph, t = build_fprop_graph(
        T,
        N,
        H,
        H,
        H,
        128,
        128,
        cudnn.data_type.BFLOAT16,
        CUDNN_DTYPE[g_dtype],
        cudnn.data_type.FLOAT,
        CUDNN_DTYPE[state_dtype] if seed else None,
        CUDNN_DTYPE[cu_dtype],
        SCALE,
        want_state,
        False,
        False,
        False,
        False,
        False,
        None,
        "log",
        0,
    )
    if want_state:
        t["fs"].set_data_type(CUDNN_DTYPE[state_dtype])
    pin(graph, plan_name)
    o = torch.empty_like(v)
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: g.to(g_dtype), t["beta"]: beta, t["cu"]: cu.to(cu_dtype), t["O"]: o}
    if seed:
        pack[t["state0"]] = s0.to(state_dtype)
    fs = None
    if want_state:
        fs = torch.empty(s0.shape, dtype=state_dtype, device=s0.device)
        pack[t["fs"]] = fs
    return graph, pack, o, fs


def bwd_graph(case, dO, *, seed=True, d_final_state=None, initial_state=None):
    """A single-node KDA_BWD graph pinned to ``kda_hopper_cuda``:
    ``(graph, pack, {gradient port: buffer})``. ``initial_state`` overrides the
    case's seed when ``seed`` is set."""
    q, k, v, g, beta, cu, s0 = case
    if initial_state is not None:
        s0 = initial_state
    T, H, _ = q.shape
    N = cu.shape[0] - 1
    f32 = cudnn.data_type.FLOAT
    graph, t = build_bprop_graph(
        T,
        N,
        H,
        H,
        H,
        128,
        128,
        cudnn.data_type.BFLOAT16,
        f32,
        f32,
        f32 if seed else None,
        f32 if d_final_state is not None else None,
        cudnn.data_type.INT32,
        None,
        SCALE,
        False,
        False,
    )
    pin(graph, ENGINE)
    outs = dict(dQ=torch.empty_like(q), dK=torch.empty_like(k), dV=torch.empty_like(v), dG=torch.empty_like(g), dBeta=torch.empty_like(beta))
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: g, t["beta"]: beta, t["cu"]: cu, t["dO"]: dO}
    for name, buf in outs.items():
        pack[t[name]] = buf
    if seed:
        pack[t["state0"]] = s0
        outs["d_initial_state"] = torch.empty_like(s0)
        pack[t["dstate0"]] = outs["d_initial_state"]
    if d_final_state is not None:
        pack[t["dfs"]] = d_final_state
    return graph, pack, outs


def workspace_for(graph):
    return torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")


def assert_executes_allocate_nothing(graph, pack, ws, repeats=5):
    """R9: warm once, then no torch allocation and no torch-visible sync across ``repeats`` executes.
    The allocation COUNTER is the assertion: ``memory_allocated()`` also moves when Python frees an
    earlier test's tensors mid-test, so equality on it reports phantom failures under xdist."""
    graph.execute(pack, ws)  # warm: NVRTC / cute.compile, port binding, the carve memo
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(repeats):
            graph.execute(pack, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations, "execute() allocated (and freed) device memory"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@requires_hopper
def test_is_the_default_on_hopper():
    """An unpinned call picks the fused CUDA engine, not slot-1 cuTile."""
    q, k, v, g, beta, cu, s0 = make_case()
    kernels = launched_kernels(lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True))
    assert any("kda_fused" in name for name in kernels), f"expected kda_fused, got {kernels[:3]}"


@requires_hopper
def test_plan_name_still_overrides():
    """Ranking is a default, not a lock: naming another engine still runs it.

    Asserted positively -- that cuTile's own kernels ran -- because merely not
    seeing ``kda_fused`` would also pass if the call had fallen through to some
    unrelated engine.
    """
    q, k, v, g, beta, cu, s0 = make_case()
    kernels = launched_kernels(lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name="kda_cutile"))
    assert any("chunk_" in name for name in kernels), f"expected cuTile chunk_* kernels, got {kernels[:3]}"
    assert not any("kda_fused" in name for name in kernels), f"plan_name ignored, got {kernels[:3]}"


@requires_hopper
def test_declined_graph_falls_through():
    """A head dim the sm90 kernels do not serve must not reach them.

    RED-then-green history: with ranking but no eligibility filter this launched
    the 128-only kernel on 64-dim tensors and raised an illegal memory access.
    """
    q, k, v, g, beta, cu, s0 = make_case(T=512, H=4, dim=64)
    out, final_state = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True)[:2]
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(final_state).all()
    kernels = launched_kernels(lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True))
    assert not any("kda_fused" in name for name in kernels), "the 128-only kernel served a 64-dim graph"


# ---------------------------------------------------------------------------
# Layouts: serve the DECLARED layout, never adapt to the observed one
# ---------------------------------------------------------------------------


def _padded_q(q):
    """q as a slice of a wider fused-projection buffer: innermost contiguous, outer stride padded."""
    wide = torch.empty(q.shape[0], q.shape[1] * 2, q.shape[2], dtype=q.dtype, device=q.device)
    wide[:, : q.shape[1]] = q
    strided = wide[:, : q.shape[1]]
    assert not strided.is_contiguous()
    return strided


@requires_hopper
def test_undeclared_padded_input_raises():
    """The plan can only stage what the graph DECLARED: a padded q handed to a
    graph that declared q packed is refused, naming the port -- not silently
    repacked into a per-call copy (Rule 5). The torch op above it keeps its
    contract: it repacks a fused-projection slice itself, in the op layer, and
    answers exactly like the packed call.

    History: the kernels take addresses and never see a stride, so before the
    layout guard an ordinary fused-projection slice was read as if it were
    packed (reviewer-measured relative errors of 7.0 on o). The guard then
    repacked it with a hidden ``torch.empty`` + copy per call inside the plan
    (Rule 8); that copy now lives in the op layer, where allocation is allowed.
    """
    case = make_case(T=512, H=4, N=2)
    q, k, v, g, beta, cu, s0 = case
    graph, pack, _, _ = fwd_graph(ENGINE, case)
    t_q = next(t for t in pack if pack[t] is q)
    pack[t_q] = _padded_q(q)
    with pytest.raises(NotImplementedError, match="'q'.*reserved no staging"):
        graph.execute(pack, workspace_for(graph))

    packed = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name=ENGINE)
    sliced = kimi_delta_attention(_padded_q(q), k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name=ENGINE)
    for a, b in zip(packed, sliced):
        assert torch.equal(a, b), "the op layer repacks a fused-projection slice; the answer is the packed one"


@requires_hopper
def test_declared_padded_input_is_repacked_from_workspace():
    """A graph that DECLARES q's padded strides gets it repacked through staging
    the plan sized at build, and the answer matches the packed graph bit for bit."""
    case = make_case(T=512, H=4, N=2)
    q, k, v, g, beta, cu, s0 = case
    T, H = q.shape[:2]
    N = cu.shape[0] - 1
    bf16, f32, i32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT, cudnn.data_type.INT32

    graph = cudnn.pygraph()
    q_t = graph.tensor([T, H, 128], stride=[2 * H * 128, 128, 1], data_type=bf16, name="q")
    k_t = graph.tensor([T, H, 128], data_type=bf16, name="k")
    v_t = graph.tensor([T, H, 128], data_type=bf16, name="v")
    g_t = graph.tensor([T, H, 128], data_type=f32, name="g")
    beta_t = graph.tensor([T, H], data_type=f32, name="beta")
    cu_t = graph.tensor([N + 1], data_type=i32, name="cu_seqlens")
    s0_t = graph.tensor([N, H, 128, 128], data_type=f32, name="initial_state")
    o_t, fs_t, _ = graph.kda(q=q_t, k=k_t, v=v_t, g=g_t, beta=beta_t, cu_seqlens=cu_t, initial_state=s0_t, scale=SCALE, output_final_state=True, name="kda")
    fs_t.set_data_type(f32)
    pin(graph, ENGINE)
    assert graph.get_workspace_size() == T * H * 128 * 2, "the bf16 q staging is the only workspace region"

    o, fs = torch.empty_like(v), torch.empty_like(s0)
    pack = {q_t: _padded_q(q), k_t: k, v_t: v, g_t: g, beta_t: beta, cu_t: cu, s0_t: s0, o_t: o, fs_t: fs}
    ws = workspace_for(graph)
    graph.execute(pack, ws)

    ref_graph, ref_pack, ref_o, ref_fs = fwd_graph(ENGINE, case)
    ref_graph.execute(ref_pack, workspace_for(ref_graph))
    torch.cuda.synchronize()
    assert torch.equal(o, ref_o) and torch.equal(fs, ref_fs)


# ---------------------------------------------------------------------------
# Zero seeds
# ---------------------------------------------------------------------------


@requires_hopper
def test_absent_initial_state_is_zeroed_on_the_execution_stream():
    """Omitting initial_state must equal passing an explicit zero state.

    The seed is a workspace carve re-zeroed on the execution stream every call.
    Filling it with torch.zeros on the ambient stream left no dependency on the
    stream the kernel runs on, so a fresh plan could read whatever the
    allocation held. Running on a side stream after dirtying the allocator is
    what makes that visible.
    """
    q, k, v, g, beta, cu, s0 = make_case(T=512, H=4)
    zeros = torch.zeros_like(s0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # Dirty the allocator so a recycled block is full of non-zeros.
        junk = torch.full_like(s0, 3.5)
        del junk
        seeded_o, seeded_fs = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=zeros, output_final_state=True, plan_name=ENGINE)[:2]
        bare_o, bare_fs = kimi_delta_attention(q, k, v, g, beta, cu, output_final_state=True, plan_name=ENGINE)[:2]
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    torch.testing.assert_close(bare_o.float(), seeded_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bare_fs, seeded_fs, atol=2e-2, rtol=2e-2)


@requires_hopper
def test_bwd_absent_optional_states_are_zeroed_each_execute():
    """A backward without initial_state / d_final_state equals one seeded with
    explicit zeros, on a side stream, even after the workspace the seeds are
    carved from has been filled with junk between executes."""
    case = make_case(T=512, H=4, N=2)
    q, k, v, g, beta, cu, s0 = case
    dO = torch.randn_like(v) * 0.5
    zeros = torch.zeros_like(s0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        seeded, seeded_pack, seeded_out = bwd_graph(case, dO, seed=True, d_final_state=zeros, initial_state=zeros)
        seeded.execute(seeded_pack, workspace_for(seeded))
        bare, bare_pack, bare_out = bwd_graph(case, dO, seed=False, d_final_state=None)
        ws = workspace_for(bare)
        for _ in range(2):
            ws.fill_(255)  # the seeds must be re-zeroed, not trusted from the previous execute
            bare.execute(bare_pack, ws)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    for name in ("dQ", "dK", "dV", "dG", "dBeta"):
        torch.testing.assert_close(bare_out[name].float(), seeded_out[name].float(), atol=1e-4, rtol=1e-4, msg=name)


# ---------------------------------------------------------------------------
# Rule 8: the plans own no device memory
# ---------------------------------------------------------------------------


@requires_hopper
@pytest.mark.parametrize("dtypes", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("state_ports", list(STATE_PORTS), ids=list(STATE_PORTS))
@pytest.mark.parametrize("engine", [ENGINE, DSL_ENGINE])
def test_execute_allocates_nothing_and_never_synchronizes(engine, state_ports, dtypes):
    """Warmed-path check over every state-port combination: further executes add
    no torch allocation and trigger no torch-visible synchronization. The
    converted-dtype cases exercise the staging carve (kda_hopper declines them)."""
    seed, want_state = STATE_PORTS[state_ports]
    g_dtype, cu_dtype, state_dtype = DTYPES[dtypes]
    graph, pack, _, _ = fwd_graph(
        engine, make_case(T=512, H=4, N=2), seed=seed, want_state=want_state, g_dtype=g_dtype, cu_dtype=cu_dtype, state_dtype=state_dtype
    )
    assert_executes_allocate_nothing(graph, pack, workspace_for(graph))


@requires_hopper
@pytest.mark.parametrize("seed", [True, False], ids=["with_d_initial_state", "without_d_initial_state"])
def test_bwd_execute_allocates_nothing(seed):
    case = make_case(T=512, H=4, N=2)
    dO = torch.randn_like(case[2]) * 0.5
    graph, pack, _ = bwd_graph(case, dO, seed=seed, d_final_state=torch.zeros_like(case[6]) if seed else None)
    assert_executes_allocate_nothing(graph, pack, workspace_for(graph))


@requires_hopper
def test_workspace_size_reflects_state_ports_and_staging():
    """Every scratch byte is declared: state carves for absent ports, staging for
    every DECLARED dtype the kernel does not read natively, the DSL kernel's own
    scratch. Sizes are the exact build-time formulas."""
    T, H, N = 512, 4, 2
    case = make_case(T=T, H=H, N=N)
    state = N * H * 128 * 128 * 4

    def cuda(**kw):
        return fwd_graph(ENGINE, case, **kw)[0].get_workspace_size()

    assert cuda() == 0
    assert cuda(want_state=False) == align_up(state)
    assert cuda(want_state=False, seed=False) == 2 * align_up(state)
    assert cuda(g_dtype=torch.bfloat16) == align_up(T * H * 128 * 4)
    assert cuda(cu_dtype=torch.int64) == align_up((N + 1) * 4)
    assert cuda(state_dtype=torch.bfloat16) == 2 * align_up(state), "bf16 initial_state in, bf16 final_state out: one fp32 staging each"

    kernel = pytest.importorskip("cudnn.linear_attention.hopper.kernel.kda_prefill_sm90", reason="the sm90 CuTe DSL kernel needs the cutedsl extra")
    scratch = kernel.workspace_layout(T, H, N).size

    def dsl(**kw):
        return fwd_graph(DSL_ENGINE, case, **kw)[0].get_workspace_size()

    assert dsl() == scratch > 0
    assert dsl(want_state=False) == scratch + align_up(state)
    assert dsl(want_state=False, seed=False) == scratch + 2 * align_up(state)
    assert kernel.workspace_layout(2048, 12, 1).size == 48_783_360  # 46.52 MiB: what _WS used to allocate per shape


@requires_hopper
@pytest.mark.parametrize("engine, want_state", [(ENGINE, False), (DSL_ENGINE, True)], ids=["kda_hopper_cuda(no_fs)", "kda_hopper"])
def test_execute_without_workspace_raises_contract_error(engine, want_state):
    """A non-zero workspace contract is enforced with the shared Workspace error,
    never met with an allocation."""
    graph, pack, _, _ = fwd_graph(engine, make_case(T=512, H=4, N=2), want_state=want_state)
    assert graph.get_workspace_size() > 0
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace but execute\(\) received none"):
        graph.execute(pack)


@requires_hopper
@pytest.mark.parametrize("engine", [ENGINE, DSL_ENGINE])
def test_alternating_workspaces_recarve(engine):
    """A caller that rotates workspaces gets correct results from each: the
    carves (and the DSL plan's CuTe descriptor memo) follow the base pointer."""
    case = make_case(T=512, H=4, N=2)
    q, k, v, g, beta, cu, s0 = case
    ref_o = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=False, plan_name=engine)[0]
    graph, pack, o, _ = fwd_graph(engine, case, want_state=False)  # absent final_state: the kernel writes the carve
    workspaces = [workspace_for(graph), workspace_for(graph)]
    assert workspaces[0].data_ptr() != workspaces[1].data_ptr()
    for i in range(4):
        o.zero_()
        graph.execute(pack, workspaces[i % 2])
        torch.cuda.synchronize()
        torch.testing.assert_close(o.float(), ref_o.float(), atol=2e-2, rtol=2e-2, msg=f"execute {i}")


# ---------------------------------------------------------------------------
# CUDA graph capture
# ---------------------------------------------------------------------------


@requires_hopper
@pytest.mark.parametrize("shape", [(2048, 12, 1), (4096, 12, 1)], ids=["2048x12", "4096x12"])
def test_cuda_graph_capture_replays(shape):
    """The engine captures, and replay reproduces eager numerics.

    Capture is the answer to this path's host-dispatch cost, and it only works
    while steady state allocates and synchronises nothing -- so this fails the
    moment a per-call allocation creeps back in.
    """
    T, H, N = shape
    q, k, v, g, beta, cu, s0 = make_case(T=T, H=H, N=N)
    call = lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name=ENGINE)  # noqa: E731

    # Warm everything that is lazy: NVRTC compile, graph build, plan buffers.
    for _ in range(5):
        eager_o, eager_fs = call()[:2]
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        captured_o, captured_fs = call()[:2]

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_o.float(), eager_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(captured_fs, eager_fs, atol=2e-2, rtol=2e-2)


@requires_hopper
@pytest.mark.parametrize("engine", [ENGINE, DSL_ENGINE])
def test_first_execute_of_a_shape_under_capture(engine):
    """The lazy build -- NVRTC + module load, or cute.compile -- runs inside
    ``torch.cuda.graph`` (a serving stack's graph-cache miss), and replay
    matches eager. The seed carve is absent here, so the per-execute memset is
    captured as a graph node and replays."""
    case = make_case(T=784, H=4, N=1, seed=11)  # a shape no other test compiles
    graph, pack, o, fs = fwd_graph(engine, case, seed=False)
    ws = workspace_for(graph)
    if engine == ENGINE:
        from cudnn.linear_attention.hopper import cuda_host

        cuda_host._LIBRARIES.clear()  # force the module load under capture even when the cubin is cached
    else:
        from cudnn.linear_attention.hopper.kernel import kda_prefill_sm90

        kda_prefill_sm90._CACHE.pop((784, 4, 1, 128), None)

    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        graph.execute(pack, ws)
    o.zero_()
    fs.zero_()
    captured.replay()
    torch.cuda.synchronize()
    replayed_o, replayed_fs = o.clone(), fs.clone()

    graph.execute(pack, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed_o.float(), o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(replayed_fs, fs, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(o).all() and fs.abs().sum() > 0


# ---------------------------------------------------------------------------
# Serving dtypes and in-kernel fusions
# ---------------------------------------------------------------------------


@requires_hopper
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"])
@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64], ids=["int32_cu", "int64_cu"])
@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16], ids=["fp32_gate", "bf16_gate"])
def test_serving_dtypes_reach_the_fused_kernel(state_dtype, cu_dtype, gate_dtype):
    """bf16 state, int64 cu_seqlens and a bf16 gate must all reach this engine.

    That combination is what FlashInfer and vLLM's Kimi path hand cuDNN. The
    kernel itself takes an int32 chunk table and fp32 gate/beta/state, so these
    are converted at execute through staging carved from the workspace (an
    interim Rule 2 exception recorded in marshal.py); declining them instead
    sent the whole call to an engine roughly 3.5x slower. Both halves are
    asserted: that the fused kernel actually ran, and that converting did not
    change the answer.
    """
    q, k, v, g, beta, cu, s0 = make_case(T=512, H=4, N=2)

    ref_o, ref_fs = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name=ENGINE)[:2]

    call = lambda: kimi_delta_attention(  # noqa: E731
        q,
        k,
        v,
        g.to(gate_dtype),
        beta,
        cu.to(cu_dtype),
        initial_state=s0.to(state_dtype),
        output_final_state=True,
        plan_name=ENGINE,
    )
    got_o, got_fs = call()[:2]
    torch.cuda.synchronize()

    assert any("kda_fused" in name for name in launched_kernels(call)), "converted call left the fused engine"
    torch.testing.assert_close(got_o.float(), ref_o.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_fs.float(), ref_fs.float(), atol=3e-2, rtol=3e-2)


@requires_hopper
@pytest.mark.parametrize("seq_lens", [[65, 33], [0, 65, 0, 33], [17]], ids=["ragged", "with_empty", "single_short"])
def test_in_kernel_gate_masks_the_partial_tail(seq_lens):
    """The in-kernel safe gate and beta sigmoid must not touch padding rows.

    RED-then-green: the load predicate zero-fills rows past the end of a
    sequence, and the rest of the kernel relies on a padded gate being 0. Both
    fusions map 0 to something non-zero -- the gate to lb*sigmoid(ea*dt_bias),
    the sigmoid to 0.5 -- so an unmasked transform corrupts the chunk-cumulative
    decay of any PARTIAL tail chunk. Equal-split varlen shapes (2048 tokens over
    4 sequences) have no partial tail and never saw it; FlashInfer's [0, 65, 0,
    33] case did, as a 0.41 relative error on final_state.
    """
    total = sum(seq_lens)
    gen = torch.Generator(device="cuda").manual_seed(61)
    n = len(seq_lens)
    H = 4
    q = torch.randn(total, H, 128, generator=gen, device="cuda", dtype=torch.float32)
    q = (q / q.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    k = torch.randn(total, H, 128, generator=gen, device="cuda", dtype=torch.float32)
    k = (k / k.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    v = torch.randn(total, H, 128, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.5
    raw = torch.randn(total, H, 128, generator=gen, device="cuda", dtype=torch.float32)
    beta_raw = torch.randn(total, H, generator=gen, device="cuda", dtype=torch.float32)
    a_log = 0.1 * torch.randn(H, generator=gen, device="cuda", dtype=torch.float32)
    dt_bias = 0.1 * torch.randn(H, 128, generator=gen, device="cuda", dtype=torch.float32)
    s0 = (torch.randn(n, H, 128, 128, generator=gen, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    bounds, acc = [0], 0
    for length in seq_lens:
        acc += length
        bounds.append(acc)
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)

    # The fused call: the kernel applies the safe gate and the beta sigmoid.
    fused_o, fused_fs = kimi_delta_attention(
        q,
        k,
        v,
        raw.contiguous(),
        beta_raw.contiguous(),
        cu,
        initial_state=s0,
        output_final_state=True,
        safe_gate=True,
        a_log=a_log,
        dt_bias=dt_bias,
        gate_lower_bound=-5.0,
        use_beta_sigmoid_in_kernel=True,
        plan_name=ENGINE,
    )[:2]

    # The same maths applied outside, then the plain contract. Any disagreement
    # is the fusion, since everything else is identical.
    g_pre = (-5.0 * torch.sigmoid(a_log.exp()[:, None] * (raw + dt_bias))).contiguous()
    beta_pre = torch.sigmoid(beta_raw).contiguous()
    ref_o, ref_fs = kimi_delta_attention(q, k, v, g_pre, beta_pre, cu, initial_state=s0, output_final_state=True, plan_name=ENGINE)[:2]
    torch.cuda.synchronize()

    torch.testing.assert_close(fused_o.float(), ref_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused_fs, ref_fs, atol=2e-2, rtol=2e-2)
