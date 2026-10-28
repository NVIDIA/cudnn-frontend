# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 D256 THD backward: actual graph/standalone contracts and independent physical origins.

The existing FP64 per-sequence reference is shared with the other half backward
row. No architecture or adapter substitution is used: every graph pins the SM100
engine and fails if it is not offered. Poisoned gaps expose reads from compact
prefixes when a port instead declares its own physical offsets.
"""

import pytest
import torch

import cudnn
import test_sdpa_bwd_thd_sm107 as checks
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from cudnn.sdpa.bwd.api_dsl_sm100_d256 import SdpaBwdDslSm100D256
from cudnn.sdpa.bwd.prepared_sm107 import THD_ORIGIN_PORTS

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]
_ENGINE = "sdpa_bwd_sm100_d256"


def make_payload(
    dt, token_major, lq, lk, phase, prefix, *, hq=8, hkv=2, causal=True, bottom_right=True, window_left=None, d=256, d_v=256, physical_origins=True
):
    c = checks._thd_case(
        lq,
        lk,
        hq,
        d,
        dt,
        cap_q=1000,
        cap_kv=1500,
        poison=True,
        seed=19 + phase,
        causal=causal,
        bottom_right=bottom_right,
        window_left=window_left,
        hkv=hkv,
        d_v=d_v,
    )
    data = {}
    starts = {}
    offsets = {}
    spec = {}
    live = {}
    for row, port in enumerate(THD_ORIGIN_PORTS):
        kv = port in ("k", "v", "dk", "dv")
        lens = lk if kv else lq
        cap = 1500 if kv else 1000
        if not physical_origins:
            # Compact bindings use the plan's packed capacity, tightened by
            # the fixed three-sequence envelope. Physical origins retain holes.
            cap = min(cap, 3 * (383 if kv else 257))
        heads = hkv if kv else hq
        org = [13 + 3 * row + phase * 7 + i * ((383 if kv else 257) + 17 + row * 5) for i in range(3)]
        if not physical_origins:
            org = (c.cu_k if kv else c.cu_q)[:-1]
        starts[port] = org
        mask = torch.zeros(cap, dtype=torch.bool, device="cuda")
        for start, n in zip(org, lens):
            mask[start : start + n] = True
        live[port] = mask
        if port == "stats":
            t = torch.full((cap, hq) if token_major else (1, hq, cap), float("nan"), device="cuda", dtype=torch.float32)
            for i, (start, n) in enumerate(zip(org, lens)):
                values = c.lse[0, :, c.cu_q[i] : c.cu_q[i] + n]
                if token_major:
                    t[start : start + n] = values.T
                else:
                    t[0, :, start : start + n] = values
            stride = hq if token_major else 1
        else:
            width = d if port in ("q", "k", "dq", "dk") else d_v
            t = torch.full((1, cap, heads, width), -31 if port in ("dq", "dk", "dv") else float("nan"), device="cuda", dtype=dt)
            if port in ("dq", "dk", "dv"):
                t[0, mask] = float("nan")
            else:
                src = getattr(c, port)
                cu = c.cu_k if kv else c.cu_q
                for i, (start, n) in enumerate(zip(org, lens)):
                    t[0, start : start + n] = src[0, cu[i] : cu[i] + n]
            stride = heads * width
        data[port] = t
        mult = 2 if stride % 2 == 0 else 1
        ro_dtype = torch.int32 if row % 2 == 0 else torch.int64
        offsets[port] = torch.tensor([x * stride // mult for x in org] + [(org[-1] + lens[-1]) * stride // mult], dtype=ro_dtype, device="cuda")
        spec[port] = (ro_dtype, mult)
    if not physical_origins:
        offsets, spec = {}, {}
    ql = torch.tensor(c.cu_q if prefix else lq, dtype=torch.int32, device="cuda")
    kl = torch.tensor(c.cu_k if prefix else lk, dtype=torch.int32, device="cuda")
    return dict(c=c, data=data, starts=starts, offsets=offsets, spec=spec, live=live, ql=ql, kl=kl, prefix=prefix)


def poison(p):
    p["ws"].fill_(255)
    for port in ("dq", "dk", "dv"):
        p["data"][port].fill_(-31)
        p["data"][port][0, p["live"][port]] = float("nan")


def run(api, p):
    args = [t.permute(0, 2, 1, 3) if port != "stats" else t for port in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv") for t in [p["data"][port]]]
    api.execute(*args, workspace=p["ws"], seq_q_lens=p["ql"], seq_kv_lens=p["kl"], thd_ragged_offsets=p["offsets"])


def check(p):
    c = p["c"]
    packed = []
    for port in ("dq", "dk", "dv"):
        lens = c.lens_kv if port != "dq" else c.lens_q
        packed.append(torch.cat([p["data"][port][:, start : start + n] for start, n in zip(p["starts"][port], lens)], dim=1))
        assert torch.all(p["data"][port][0, ~p["live"][port]] == -31), "output padding changed: " + port
    checks._check(c, *packed)
    for i, (nq, nk) in enumerate(zip(c.lens_q, c.lens_kv)):
        if not nq or not nk:
            checks._assert_empty_sequence_exactly_zero(c, *packed, i)
    for port, snap in p["snapshots"].items():
        torch.testing.assert_close(p["data"][port], snap, atol=0, rtol=0, equal_nan=True)


def snapshots(p):
    return {port: p["data"][port].clone() for port in ("q", "k", "v", "o", "do", "stats")}


def make_adapter(p, token_major, deterministic=False):
    c = p["c"]
    eq, ek, es = checks._envelope_samples(c)
    eo = torch.empty(c.b, eq.shape[2], c.h, c.d_v, device=eq.device, dtype=c.dtype).permute(0, 2, 1, 3)
    ev = torch.empty(c.b, ek.shape[2], c.hkv, c.d_v, device=ek.device, dtype=c.dtype).permute(0, 2, 1, 3)
    api = SdpaBwdDslSm100D256(
        sample_q=eq,
        sample_k=ek,
        sample_v=ev,
        sample_o=eo,
        sample_do=eo,
        sample_stats=es,
        sample_dq=eq,
        sample_dk=ek,
        sample_dv=ev,
        scale_softmax=c.scale,
        is_causal=c.causal,
        causal_bottom_right=c.bottom_right,
        window_size_left=None if c.window_left is None else c.window_left - 1,
        thd=True,
        max_total_seq_len_q=1000,
        max_total_seq_len_kv=1500,
        thd_stats_token_major=token_major,
        thd_ragged_offsets=p["spec"],
        deterministic=deterministic,
    )
    assert api.check_support()
    api.compile()
    return api


def make_graph(p, token_major, cudnn_handle=None, deterministic=False):
    c = p["c"]
    io = cudnn.data_type.HALF if c.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle)
    tensors, ragged = {}, {}

    def describe(port, tensor=None):
        kv = port in ("k", "v", "dk", "dv")
        heads, seq = (c.hkv, 383) if kv else (c.h, 257)
        width = c.d if port in ("q", "k", "dq", "dk") else c.d_v
        dim, stride, dtype = [3, heads, seq, width], [seq * heads * width, width, heads * width, 1], io
        if port == "stats":
            dim, dtype = [3, c.h, 257, 1], cudnn.data_type.FLOAT
            stride = [257 * c.h, 1, c.h, 1] if token_major else [c.h * 1000, 1000, 1, 1]
        if tensor is None:
            tensor = g.tensor(name=port, dim=dim, stride=stride, data_type=dtype)
        else:
            tensor.set_output(True).set_data_type(dtype).set_dim(dim).set_stride(stride)
        ro_dt, multiplier = p["spec"][port]
        ro = g.tensor(
            name=port + "_origin", dim=[4, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32 if ro_dt == torch.int32 else cudnn.data_type.INT64
        )
        tensor.set_ragged_offset(ro).set_ragged_offset_multiplier(multiplier)
        tensors[port], ragged[port] = tensor, ro
        return tensor

    for port in ("q", "k", "v", "o", "do", "stats"):
        describe(port)
    ql = g.tensor(name="seq_len_q", dim=[3, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    kl = g.tensor(name="seq_len_kv", dim=[3, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    mask = {}
    if c.causal:
        mask["use_causal_mask_bottom_right" if c.bottom_right else "use_causal_mask"] = True
    if c.window_left is not None:
        mask["sliding_window_length"] = c.window_left
    outs = g.sdpa_backward(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        o=tensors["o"],
        dO=tensors["do"],
        stats=tensors["stats"],
        attn_scale=c.scale,
        use_deterministic_algorithm=deterministic,
        use_padding_mask=True,
        seq_len_q=ql,
        seq_len_kv=kl,
        max_total_seq_len_q=1000,
        max_total_seq_len_kv=1500,
        **mask,
    )
    for port, out in zip(("dq", "dk", "dv"), outs):
        describe(port, out)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = checks._plan_index(g, _ENGINE)
    assert idx is not None, f"{_ENGINE} missing: {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()

    def execute(payload):
        assert not payload["prefix"], "the graph length ports take B lengths, not B+1 prefixes"
        vp = {tensors[port]: payload["data"][port] for port in THD_ORIGIN_PORTS}
        vp.update({ragged[port]: payload["offsets"][port] for port in THD_ORIGIN_PORTS})
        vp.update({ql: payload["ql"], kl: payload["kl"]})
        if cudnn_handle is not None:
            cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        g.execute(vp, payload["ws"], handle=cudnn_handle)

    return g, execute


@pytest.mark.parametrize("dt,token_major", [(torch.bfloat16, True), (torch.float16, False)], ids=["bf16-token-stats", "fp16-head-stats"])
@pytest.mark.parametrize("hq,hkv", [(2, 2), (8, 2)], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "causal,bottom_right,window_left",
    [(False, False, None), (True, False, None), (True, True, None), (True, True, 129)],
    ids=["unmasked", "top-left", "bottom-right", "window"],
)
def test_graph_independent_origins(dt, token_major, hq, hkv, causal, bottom_right, window_left):
    p = make_payload(
        dt, token_major, (257, 93, 128), (383, 129, 267), 0, False, hq=hq, hkv=hkv, causal=causal, bottom_right=bottom_right, window_left=window_left
    )
    g, execute = make_graph(p, token_major)
    p["ws"] = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda")
    p["snapshots"] = snapshots(p)
    poison(p)
    execute(p)
    torch.cuda.synchronize()
    check(p)


@pytest.mark.parametrize("dt,token_major", [(torch.bfloat16, True), (torch.float16, False)], ids=["bf16-token-stats", "fp16-head-stats"])
@pytest.mark.parametrize("surface", ["standalone", "graph"])
def test_origins_rebind_and_replay(monkeypatch, dt, token_major, surface):
    check_origins_rebind_and_replay(monkeypatch, dt, token_major, surface)


def check_origins_rebind_and_replay(monkeypatch, dt, token_major, surface, *, d=256, d_v=256, physical_origins=True, cudnn_handle=None, deterministic=False):
    import cutlass.cute as cute
    from functools import partial

    payload = partial(make_payload, d=d, d_v=d_v, physical_origins=physical_origins)

    # Force two complete GQA groups through reused scratch to expose cross-chunk
    # addressing. The production workspace policy is not under test here.
    monkeypatch.setattr(SdpaBwdDslSm100D256, "_ws_budget_bytes", lambda self: 1 << 20)
    p = payload(dt, token_major, (257, 93, 128), (383, 129, 267), 0, False)
    if surface == "standalone":
        plan = make_adapter(p, token_major, deterministic=deterministic)
        assert plan._qh_chunk == 4
        execute = lambda payload: run(plan, payload)
        ws_bytes = plan.scratch_workspace_bytes()
    else:
        plan, execute = make_graph(p, token_major, cudnn_handle, deterministic=deterministic)
        ws_bytes = plan.get_workspace_size()
    p["ws"] = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
    poison(p)
    p["snapshots"] = snapshots(p)
    execute(p)
    torch.cuda.synchronize()
    check(p)

    def no_compile(*args, **kwargs):
        raise AssertionError("a prepared THD execute recompiled")

    monkeypatch.setattr(cute, "compile", no_compile)
    stream = torch.cuda.Stream()
    retained = []
    try:
        for binding in range(2):
            p = payload(dt, token_major, (257, 93, 128), (383, 129, 267), binding, surface == "standalone" and bool(binding))
            p["ws"] = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
            poison(p)
            p["snapshots"] = snapshots(p)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                execute(p)
            stream.synchronize()
            check(p)
            capture = torch.cuda.CUDAGraph()
            retained.append((capture, p))
            with torch.cuda.graph(capture, stream=stream):
                previous_sync_mode = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    execute(p)
                finally:
                    torch.cuda.set_sync_debug_mode(previous_sync_mode)
        for phase, (lq, lk) in enumerate([((129, 0, 17), (0, 259, 127)), ((0, 0, 0), (0, 0, 0)), ((17, 93, 128), (127, 129, 267))], 2):
            for capture, p in retained:
                changed = payload(dt, token_major, lq, lk, phase, p["prefix"])
                for port in THD_ORIGIN_PORTS:
                    p["data"][port].copy_(changed["data"][port])
                for port in p["offsets"]:
                    p["offsets"][port].copy_(changed["offsets"][port])
                p["ql"].copy_(changed["ql"])
                p["kl"].copy_(changed["kl"])
                for key in ("c", "live", "starts"):
                    p[key] = changed[key]
                p["snapshots"] = snapshots(p)
                poison(p)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    capture.replay()
                stream.synchronize()
                check(p)
                expected = [p["data"][port].clone() for port in ("dq", "dk", "dv")]
                poison(p)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    capture.replay()
                stream.synchronize()
                check(p)
                for port, want in zip(("dq", "dk", "dv"), expected):
                    if deterministic:
                        assert torch.equal(p["data"][port].view(torch.int16), want.view(torch.int16)), port
                    else:
                        torch.testing.assert_close(p["data"][port], want, atol=0, rtol=0)
    finally:
        stream.synchronize()
        for capture, _ in retained:
            capture.reset()


@pytest.mark.parametrize("first_dtype", [torch.int32, torch.int64], ids=["i32-to-i64", "i64-to-i32"])
@pytest.mark.parametrize("port", ["q", "stats"])
def test_origin_integer_types_separate_compiled_artifacts(monkeypatch, tmp_path, first_dtype, port):
    """Offset width changes the compiled pointer ABI even when every stride/multiplier is identical."""
    from cudnn.frost import compiled_cache

    previous_cache = compiled_cache.get_cache_dir()
    monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "0")
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "1")
    compiled_cache.clear_memo()
    compiled_cache.set_cache_dir(tmp_path)
    second_dtype = torch.int64 if first_dtype == torch.int32 else torch.int32
    plans, payloads, entries = [], [], []
    captures = []
    try:
        for offset_dtype in (first_dtype, second_dtype):
            p = make_payload(torch.bfloat16, True, (257, 93, 128), (383, 129, 267), 0, False, hq=2, hkv=2)
            p["offsets"][port] = p["offsets"][port].to(offset_dtype)
            p["spec"][port] = (offset_dtype, p["spec"][port][1])
            with compiled_cache.record_entries() as recorded:
                api = make_adapter(p, True)
            assert recorded, "the prepared host did not persist an artifact"
            if plans:
                # Check BEFORE launching a colliding entry: the old bug could
                # interpret current valid offsets using the previous pointer width.
                assert api._compiled.entry is not plans[0]._compiled.entry, "different origin integer widths reused the same compiled host"
                assert set(recorded).isdisjoint(entries[0]), "different origin integer widths reused the same disk artifact"
            plans.append(api)
            payloads.append(p)
            entries.append(set(recorded))
            p["ws"] = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
            p["snapshots"] = snapshots(p)
            poison(p)
            run(api, p)
            torch.cuda.synchronize()
            check(p)

        # Forget process memoization and require actual disk reload for BOTH
        # declarations. A previously built launchable cannot satisfy this probe.
        compiled_cache.clear_memo()
        monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
        before = compiled_cache.stats()
        with compiled_cache.refuse_compiles():
            for p, expected_entries in zip(payloads, entries):
                with compiled_cache.record_entries() as reloaded_entries:
                    api = make_adapter(p, True)
                assert set(reloaded_entries) == expected_entries
                poison(p)
                run(api, p)
                torch.cuda.synchronize()
                check(p)
                capture = torch.cuda.CUDAGraph()
                captures.append(capture)
                with torch.cuda.graph(capture):
                    run(api, p)
                poison(p)
                capture.replay()
                torch.cuda.synchronize()
                check(p)
        assert compiled_cache.stats()["hits"] >= before["hits"] + 2, "the replan did not reload both disk artifacts"
    finally:
        torch.cuda.synchronize()
        for capture in captures:
            capture.reset()
        compiled_cache.clear_memo()
        compiled_cache.set_cache_dir(previous_cache)
