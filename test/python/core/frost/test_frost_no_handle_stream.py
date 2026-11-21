# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST plans executed without a handle run on the caller's current stream, like the backend.

Each case builds a FROST plan, then executes it with no handle (a) inside a side stream right
behind a slow producer of its input and (b) inside a global-mode CUDA graph capture followed by a
changed-input replay. An engine that launches on stream 0 reads the stale input in (a) and leaves
the capture empty in (b).
"""

import pytest

torch = pytest.importorskip("torch")

import cudnn
from cudnn.engines.engine_ids import FAMILY_BLOCK, FROST_SDPA_FWD_ID_BASE

pytestmark = pytest.mark.L0

_SLEEP_CYCLES = 100_000_000


def _cutedsl_or_skip():
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("requires a supported CuTe DSL")


def _sdpa_case(monkeypatch):
    cc = torch.cuda.get_device_capability()
    if cudnn.backend_version() < 92400 or cc not in ((8, 0), (9, 0), (10, 0), (10, 3)):
        pytest.skip("the FROST decode SDPA plan is exercised on cc 8.0 / 9.0 / 10.0 / 10.3")
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = torch.randn(2, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    tq, tk, tv = [graph.tensor_like(x) for x in (q, k, v)]
    to, _ = graph.sdpa(q=tq, k=tk, v=tv, generate_stats=False, attn_scale=128**-0.5)
    o = torch.empty_like(q)
    to.set_output(True).set_dim(o.shape).set_stride(o.stride())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, cfg in enumerate(graph.plans) if FROST_SDPA_FWD_ID_BASE <= cfg.engine_id < FROST_SDPA_FWD_ID_BASE + FAMILY_BLOCK]
    for index in frost:
        graph.select_plan(index)
        try:
            graph.build_plans()
        except (cudnn.cudnnGraphNotSupportedError, NotImplementedError):
            continue
        break
    else:
        pytest.skip("no FROST SDPA plan built for the decode graph")
    assert graph.selected_engine is not None

    def reference():
        scores = q.float() @ k.float().transpose(-2, -1) * 128**-0.5
        return scores.softmax(-1) @ v.float()

    return graph, {tq: q, tk: k, tv: v, to: o}, q, o, reference, dict(atol=2e-2, rtol=2e-2)


def _gemm_case(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FROST GEMM is exercised on SM10x")
    M = N = K = 256
    _bf, _f32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    a = torch.randint(-3, 4, (1, M, K), device="cuda").to(torch.bfloat16)
    b = torch.randint(-3, 4, (1, N, K), device="cuda").to(torch.bfloat16)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    graph = cudnn.pygraph(io_data_type=_bf, intermediate_data_type=_f32, compute_data_type=_f32)
    A = graph.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = graph.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = graph.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(_bf)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    frost = [i for i, n in enumerate(names) if n == "frost_gemm" or n.startswith("frost_gemm[")]
    if not frost:
        pytest.skip(f"no FROST GEMM plan for the dense graph (plans={names})")
    graph.select_plan(frost[0])
    graph.build_plans()
    assert graph.selected_engine.name == "frost_gemm"

    def reference():
        return (a.float() @ b.float().transpose(-2, -1)).bfloat16().float()

    return graph, {A: a, B: b, C: c}, a, c, reference, dict(atol=0, rtol=0)


_CASES = {"sdpa": _sdpa_case, "gemm": _gemm_case}


@pytest.fixture(params=list(_CASES))
def frost_case(request, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    _cutedsl_or_skip()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    graph, pack, src, out, reference, tol = _CASES[request.param](monkeypatch)
    ws = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    graph.execute(pack, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), reference(), **tol)
    return graph, pack, src, out, reference, tol, ws


def test_no_handle_execute_waits_for_a_side_stream_producer(frost_case):
    graph, pack, src, out, reference, tol, ws = frost_case
    fresh = src * -1
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SLEEP_CYCLES)
        src.copy_(fresh)
        out.zero_()
        graph.execute(pack, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), reference(), **tol, msg=lambda d: "the plan ran without waiting for the side-stream producer: " + d)


def test_no_handle_execute_is_capturable(frost_case):
    graph, pack, src, out, reference, tol, ws = frost_case
    cg = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(cg):
            graph.execute(pack, ws)
        src.mul_(-1)
        out.zero_()
        cg.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), reference(), **tol, msg=lambda d: "the no-handle capture did not record the launch: " + d)
    finally:
        cg.reset()
