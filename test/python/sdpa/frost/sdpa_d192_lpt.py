# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D192 half scheduling: actual work decode, admission and retained replay."""

import ast
import inspect
import math
from itertools import accumulate
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("policy", [0, 1, 2])
@pytest.mark.parametrize("lengths", [(1537,), (1537, 0, 513, 2049)])
def test_thd_work_order(bounded, policy, lengths):
    """Execute the production decoder source; numerics alone cannot detect ignored LPT."""
    from cudnn.sdpa.fwd.kernels._common_blackwell import make_sdpa_helpers

    factory = ast.parse(inspect.getsource(make_sdpa_helpers)).body[0]
    name = "_thd_decode_bounded" if bounded else "_thd_decode_prefix"
    fn = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    scope = dict(
        cutlass=SimpleNamespace(Int32=int, const_expr=bool, make_array_view=lambda x: x),
        CFG=SimpleNamespace(CGA_M=2, SPLIT_KV=1, TILE_K=192, DTYPE_QKV=2, SCHEDULER_POLICY=policy, PACK_GQA=False, QH_PER_KH=4),
        SCHED_LPT=1,
        SCHED_LPT_L2=2,
        thd_token_tile_m=512,
        _q_supers_per_cluster=2,
        _packed_heads_per_kv=1,
        THD_WORK_PREFIX_OFF=lambda b: 4 * b + 4,
        THD_REMAP_OFF=lambda b: 3 * b + 2,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "<production THD decoder>", "exec"), scope)
    b, heads = len(lengths), 8
    for qlens in (lengths, tuple(reversed(lengths)), (0,) * b):
        counts = [(q + 511) // 512 for q in qlens]
        remap = sorted(range(b), key=lambda i: (-qlens[i], i))
        meta = [0] * (5 * b + 5)
        meta[b : 2 * b + 1] = [0, *accumulate(qlens)]
        meta[3 * b + 2 : 4 * b + 2] = remap
        meta[4 * b + 4 :] = [0, *accumulate(counts[i] * heads for i in remap)]
        expected = []
        for batch in remap:
            rows = counts[batch]
            if policy == 0:
                tasks = [(row, head) for head in range(heads) for row in range(rows)]
            elif policy == 1:
                tasks = [(row, head) for row in reversed(range(rows)) for head in range(heads)]
            else:
                tasks = [(row, head) for first in range(0, heads, 4) for row in reversed(range(rows)) for head in range(first, first + 4)]
            expected.extend((row * 2, head, batch) for row, head in tasks)
        expected += [(0, 0, b)] * 3
        got = [scope[name](i * 2, meta, b, heads, 0) for i in range(len(expected))]
        assert got == expected


@pytest.mark.parametrize("dtype", ["HALF", "BFLOAT16"])
@pytest.mark.parametrize("shape", [(192, 128), (160, 96)])
@pytest.mark.parametrize("leg", ["dense", "thd", "paged", "paged_thd"])
def test_native_admission(dtype, shape, leg):
    import cudnn
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, mismatch

    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    f = SdpaGraphFacts(
        b=3,
        h_q=8,
        h_kv=2,
        s_q=1537,
        s_kv=2048,
        d_qk=shape[0],
        d_v=shape[1],
        dtype=getattr(cudnn.data_type, dtype),
        device_cc=(10, 7),
        causal=True,
        bottom_right=True,
        thd="thd" in leg,
        padded="thd" in leg or "paged" in leg,
        has_paged_kv="paged" in leg,
        page_size=16 if "paged" in leg else 0,
    )
    why = mismatch(caps, f, SdpaFwdKnobs(cga=2, tile_m=128, tile_n=128, split_kv=1, pack_gqa=False, sched_policy=1))
    if "thd" in leg and shape != (192, 128):
        assert why is not None and "dense-only" in why
        return
    assert why is None, why
    why = mismatch(caps, f, SdpaFwdKnobs(cga=2, sched_policy=2))
    assert why is not None and "sched_policy" in why


def _require_device():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("requires SM100/SM103/SM107")
    return torch.cuda.get_device_capability()


def _reference(q, k, v, *, causal, bottom, window, sinks=None):
    import torch

    q, k, v = (t.double().cpu() for t in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    nq, nk = q.shape[2], k.shape[2]
    pos = torch.arange(nq)[:, None] + (nk - nq if bottom else 0)
    keys = torch.arange(nk)[None]
    if causal:
        scores.masked_fill_(keys > pos, -torch.inf)
    if window is not None:
        scores.masked_fill_(keys < pos - window, -torch.inf)
    full = scores if sinks is None else torch.cat((scores, sinks.double().cpu().view(1, -1, 1, 1).expand(*scores.shape[:3], 1)), -1)
    return torch.nan_to_num(full.softmax(-1), nan=0.0)[..., :nk] @ v, full.logsumexp(-1)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("case", ["plain", "causal", "bottom_tail", "window_sink", "split_empty", "logical_width", "output_only"])
def test_dense_replay(dtype_name, case):
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    cc = _require_device()
    torch.manual_seed(1207)
    b, h, hk = 2, 8, 2
    d, dv = (160, 96) if case == "logical_width" else (192, 128)
    sq, sk = (769, 1024) if case in ("plain", "causal", "output_only") else (769, 257)
    causal, bottom = case != "plain", case not in ("plain", "causal", "output_only")
    window = 63 if case == "window_sink" else None
    split = 8 if case == "split_empty" else 1
    stats = case != "output_only"
    dtype = getattr(torch, dtype_name)

    def inputs():
        return [
            torch.randn(b, n + 16, heads, width, device="cuda", dtype=dtype)[:, :n].transpose(1, 2)
            for n, heads, width in ((sq, h, d), (sk, hk, d), (sk, hk, dv))
        ]

    q, k, v = inputs()
    fresh = inputs()
    sinks = torch.linspace(-1, 1, h, device="cuda") if case == "window_sink" else None
    out = torch.empty(b, sq, h, dv, device="cuda", dtype=dtype).transpose(1, 2)
    lse = torch.empty(b, h, sq, device="cuda") if stats else None
    captures, workspaces, apis, stats_atols = [], [], [], []
    expected = _reference(q, k, v, causal=causal, bottom=bottom, window=window, sinks=sinks)
    try:
        eager = []
        for policy in ((0, 1) if cc == (10, 7) else (0, 1, 2)):
            api = SdpaFwdDslSm100(
                q,
                k,
                v,
                out,
                lse,
                scale_softmax=d**-0.5,
                is_causal=causal,
                causal_bottom_right=bottom,
                window_size_left=window,
                has_sink=sinks is not None,
                cga=2,
                split_kv=split,
                pack_gqa=False,
                sched_policy=policy,
            )
            assert api.check_support()
            api.compile()
            assert api._k_mod.CFG.SCHEDULER_POLICY == policy and api.flavor == (192, 128)
            # The legacy dense-causal tail exp2 polynomial (used on SM103)
            # already exceeds 4e-4 on the Natural parent for some BF16 rows.
            # Bound that form at 1e-3; all schedules must also match bitwise.
            legacy_exp2 = getattr(api._k_mod, "_DENSE_CAUSAL_PUBLIC_EXP2_MIX", False) and not getattr(api._k_mod, "_E2E_ENABLED", False)
            lse_atol = 1e-3 if legacy_exp2 else 4e-4
            stats_atols.append(lse_atol)
            ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
            api.execute(q, k, v, out, lse_tensor=lse, sinks=sinks, workspace=ws)
            torch.cuda.synchronize()
            torch.testing.assert_close(out.double().cpu(), expected[0], atol=4e-3, rtol=4e-3)
            if stats:
                torch.testing.assert_close(lse.double().cpu(), expected[1], atol=lse_atol, rtol=0)
            eager.append((out.clone(), lse.clone() if stats else None))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                api.execute(q, k, v, out, lse_tensor=lse, sinks=sinks, workspace=ws)
            captures.append(graph)
            workspaces.append(ws)
            apis.append(api)
            api.execute(*fresh, out, lse_tensor=lse, sinks=sinks, workspace=ws)
        for result in eager[1:]:
            assert torch.equal(eager[0][0], result[0])
            if stats:
                assert torch.equal(eager[0][1], result[1])
        v.mul_(0.75)
        expected = _reference(q, k, v, causal=causal, bottom=bottom, window=window, sinks=sinks)
        previous = None
        for graph, ws, lse_atol in zip(captures, workspaces, stats_atols):
            out.fill_(torch.nan)
            ws.fill_(255)
            if stats:
                lse.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out.double().cpu(), expected[0], atol=4e-3, rtol=4e-3)
            if stats:
                torch.testing.assert_close(lse.double().cpu(), expected[1], atol=lse_atol, rtol=0)
            if previous is not None:
                assert torch.equal(out, previous[0])
                if stats:
                    assert torch.equal(lse, previous[1])
            previous = (out.clone(), lse.clone() if stats else None)
    finally:
        for graph in captures:
            graph.reset()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("case", ["causal", "bottom", "window", "runtime_waves", "persistent_tail"])
def test_ragged_replay(dtype_name, case):
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    cc = _require_device()
    torch.manual_seed(1211)
    wave_case = case in ("runtime_waves", "persistent_tail")
    if wave_case and cc != (10, 7):
        pytest.skip("native SM107 D192 runtime initial waves")
    b, h, hk, sq, sk = 3, (24 if wave_case else 8), (6 if wave_case else 2), 1537, 2048
    d, dv = (160, 96) if case == "logical_width" else (192, 128)
    dtype = getattr(torch, dtype_name)
    bottom, window = case != "causal", 127 if case == "window" else None
    q = torch.randn(b * sq, h, d, device="cuda", dtype=dtype)
    k = torch.randn(b * sk, hk, d, device="cuda", dtype=dtype)
    v = torch.randn(b * sk, hk, dv, device="cuda", dtype=dtype)
    o = torch.empty(b * sq, h, dv, device="cuda", dtype=dtype)
    lse = torch.empty(b * sq, h, device="cuda")
    cq = torch.arange(b + 1, device="cuda", dtype=torch.int32) * sq
    ck = torch.arange(b + 1, device="cuda", dtype=torch.int32) * sk

    if case == "runtime_waves":
        cq.copy_(torch.tensor([0, 1537, 1537, 2050], device="cuda", dtype=torch.int32))
        ck.copy_(torch.tensor([0, 2048, 2048, 2817], device="cuda", dtype=torch.int32))

    def view(t, n):
        return t.view(b, n, t.shape[1], t.shape[2]).transpose(1, 2)

    apis, workspaces, graphs = [], [], []
    try:
        for policy in ((0, 1) if cc == (10, 7) else (0, 1, 2)):
            api = SdpaFwdDslSm100(
                view(q, sq),
                view(k, sk),
                view(v, sk),
                view(o, sq),
                lse.view(b, sq, h).transpose(1, 2),
                thd=True,
                cu_seq_q_lens=True,
                cu_seq_kv_lens=True,
                seq_kv_lens_present=True,
                cga=2,
                pack_gqa=False,
                split_kv=1,
                sched_policy=policy,
                scale_softmax=d**-0.5,
                is_causal=True,
                causal_bottom_right=bottom,
                window_size_left=window,
            )
            assert api.check_support()
            api.compile()
            assert api._k_mod.CFG.SCHEDULER_POLICY == policy
            assert api._k_mod.THD_PERSISTENT
            plan = api._thd_plan()
            resident = torch.cuda.get_device_properties(0).multi_processor_count // 2
            assert plan.units <= getattr(api._k_mod, "THD_INITIAL_WAVES", 1) * resident
            if wave_case:
                assert plan.wave_policy is not None and plan.wave_policy[2] == 2
            ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
            if case == "persistent_tail":
                from cudnn.sdpa.fwd import prepared

                buffers = (q, k, v, o, cq, ck, lse, None, None, None)
                facts = {name: prepared.facts_of_tensor(tensor) for name, tensor in zip(prepared._NATIVE_THD_ROLES, buffers) if tensor is not None}
                spec = api._thd_spec
                frame = spec.native.bind(
                    prepared._native_pack_from_facts(facts), prepared._NATIVE_THD_INDICES, ws.data_ptr(), torch.cuda.current_stream().cuda_stream
                )
                # Prove valid work exceeds the actual bound launch, rather than
                # only checking the sample plan's nominal wave limit.
                live_units = b * h * math.ceil(sq / api._k_mod.CGA_TILE_M)
                assert live_units > frame[spec.index["n_thd_units"]]

            def run():
                # Bind a smaller current capacity than the graph-free sample
                # declaration. The common binder must reconsider initial waves.
                capacity = 2050 if case == "runtime_waves" else q.shape[0]
                api.execute(q[:capacity], k, v, o[:capacity], lse_tensor=lse[:capacity], seq_q_lens=cq, seq_kv_lens=ck, workspace=ws)

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            apis.append(api)
            workspaces.append(ws)
            graphs.append(graph)
        replay_lengths = [([1537, 0, 513], [2048, 0, 769]), ([17, 513, 0], [65, 0, 0]), ([0, 0, 0], [0, 0, 0])]
        if case == "persistent_tail":
            replay_lengths.insert(0, ([sq] * b, [sk] * b))
        for qs, ks in replay_lengths:
            qprefix, kprefix = [0, *accumulate(qs)], [0, *accumulate(ks)]
            cq.copy_(torch.tensor(qprefix, device="cuda", dtype=torch.int32))
            ck.copy_(torch.tensor(kprefix, device="cuda", dtype=torch.int32))
            v.mul_(0.75)
            expected = []
            for i, (nq, nk) in enumerate(zip(qs, ks)):
                if nq:
                    expected.append(
                        (
                            i,
                            _reference(
                                q[qprefix[i] : qprefix[i + 1]].transpose(0, 1)[None],
                                k[kprefix[i] : kprefix[i + 1]].transpose(0, 1)[None],
                                v[kprefix[i] : kprefix[i + 1]].transpose(0, 1)[None],
                                causal=True,
                                bottom=bottom,
                                window=window,
                            ),
                        )
                    )
            previous = None
            for graph, ws in zip(graphs, workspaces):
                o.fill_(19)
                lse.fill_(29)
                ws.fill_(255)
                graph.replay()
                torch.cuda.synchronize()
                for i, (ro, rs) in expected:
                    lo, hi = qprefix[i : i + 2]
                    torch.testing.assert_close(o[lo:hi].double().cpu(), ro[0].transpose(0, 1), atol=4e-3, rtol=4e-3)
                    torch.testing.assert_close(lse[lo:hi].double().cpu(), rs[0].T, atol=4e-4, rtol=0)
                assert torch.all(o[qprefix[-1] :] == 19) and torch.all(lse[qprefix[-1] :] == 29)
                if previous is not None:
                    assert torch.equal(o, previous[0]) and torch.equal(lse, previous[1])
                previous = (o.clone(), lse.clone())
    finally:
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("packed", [False, True])
def test_paged_replay(dtype_name, packed, monkeypatch):
    import cudnn
    import torch
    from native_d192_paged_sm107_test_utils import test_dense_replay as run

    if _require_device() != (10, 7):
        pytest.skip("native SM107 paged D192")
    original = cudnn.pygraph.create_execution_plan

    def lpt(graph, engine_id, knobs=None):
        return original(graph, engine_id, {**(knobs or {}), cudnn.knob_type.SCHED_POLICY: 1})

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", lpt)
    run(getattr(torch, dtype_name), packed, True, 64, True, (192, 128), monkeypatch)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("stats_layout", ["head", "token"])
def test_paged_independent_origins(dtype_name, stats_layout, monkeypatch):
    import cudnn
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as run

    if _require_device() != (10, 7):
        pytest.skip("native SM107 paged D192 physical origins")
    original = cudnn.pygraph.create_execution_plan

    def lpt(graph, engine_id, knobs=None):
        return original(graph, engine_id, {**(knobs or {}), cudnn.knob_type.SCHED_POLICY: 1})

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", lpt)
    run((192, 128), dtype_name, "mixed", stats_layout, 2, 1, True, page_size=64, window_left=47, sink=-120.0)
