# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 sink splits fold the sink once and retain current binding ownership."""

import dataclasses
import inspect
import math

import pytest

from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L1, requires_dsl]

# dims, group, Q, KV, CGA, splits, packed, causal, left window, kernel
CASES = {
    "decode64": ((64, 64), 8, 4, 1025, 1, 8, True, True, None, "sm107/prefill_d128_f16.py"),
    "decode128": ((128, 128), 4, 7, 769, 1, 3, True, True, 127, "sm100/decode_d128_f16.py"),
    "native128": ((128, 128), 4, 129, 513, 2, 8, True, True, None, "sm107/prefill_d128_f16.py"),
    "partial6": ((96, 80), 6, 129, 385, 2, 4, True, True, 31, "sm107/prefill_d128_f16.py"),
    "plain128": ((128, 128), 1, 129, 257, 2, 4, False, False, None, "sm107/prefill_d128_f16.py"),
    "native192": ((192, 128), 2, 257, 385, 2, 8, False, True, 47, "sm107/prefill_d192_d128_f16.py"),
    "keyless": ((128, 128), 4, 257, 65, 2, 8, True, True, None, "sm107/prefill_d128_f16.py"),
    "decode256": ((256, 256), 4, 4, 513, 2, 4, True, True, None, "sm107/decode_d256_f16.py"),
    "native64plain": ((64, 64), 1, 1, 4096, 1, 8, False, False, None, "sm107/prefill_d128_f16.py"),
    "singleq64partial": ((48, 32), 12, 129, 385, 1, 3, True, True, 47, "sm100/decode_d128_f16.py"),
    "native64partial": ((48, 32), 12, 129, 385, 2, 3, True, True, 47, "sm107/prefill_d128_f16.py"),
    "native64keyless": ((64, 64), 4, 257, 65, 2, 8, True, True, None, "sm107/prefill_d128_f16.py"),
    "native64empty": ((64, 64), 4, 65, 65, 2, 8, True, True, None, "sm107/prefill_d128_f16.py"),
}
# The public graph rejects bottom-right Q capacity larger than KV capacity.
GRAPH_CASES = [name for name, case in CASES.items() if case[2] <= case[3]]


@pytest.mark.parametrize("case", GRAPH_CASES)
def test_sink_split_graph_admission(case, monkeypatch):
    from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts
    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    dims, group, sq, sk, cga, splits, packed, causal, window, _ = CASES[case]
    facts = _f16_facts(
        b=2, h_q=2 * group, h_kv=2, s_q=sq, s_kv=sk, d_qk=dims[0], d_v=dims[1], has_sink=True, causal=causal, bottom_right=causal, window_left=window
    )
    knobs = engines.SdpaFwdKnobs(cga=cga, split_kv=splits, pack_gqa=packed, tile_m=128, tile_n=128, sched_policy=0)
    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), facts, knobs) is None
    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), dataclasses.replace(facts, has_epilogue_gate=True), knobs) is not None


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_sink_gate_split_standalone_declines_at_support(dtype_name, monkeypatch):
    import torch
    from test_sdpa_fwd_dsl_sm107 import _fake_cc, _gate_api
    from cudnn.frost import buffers

    _fake_cc(monkeypatch, (10, 7))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    options = dict(s=4, dtype=getattr(torch, dtype_name), cga=2, pack_gqa=True, split_kv=4, is_causal=True)
    # Each feature is supported separately; only their combination is absent.
    assert _gate_api(**options, has_sink=True, with_gate=False).check_support()
    assert _gate_api(**options, has_sink=False, with_gate=True).check_support()
    with pytest.raises(ValueError, match="attention sink"):
        _gate_api(**options, has_sink=True, with_gate=True).check_support()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("stats", ["none", "natural", "log2"])
@pytest.mark.parametrize("case", list(CASES))
def test_sink_split_standalone_replay(case, dtype_name, stats):
    _check_sink_split(case, dtype_name, stats, graph_api=False)


@pytest.mark.parametrize("case", GRAPH_CASES)
def test_sink_split_graph_rebind(case, cudnn_handle):
    _check_sink_split(case, "bfloat16", "log2", graph_api=True, handle=cudnn_handle)


@pytest.mark.parametrize("case", ["native128", "keyless"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_sink_split_narrow_standalone_replay(case, dtype_name):
    _check_sink_split(case, dtype_name, "log2", graph_api=False, tile_n=64)


@pytest.mark.parametrize("case", ["native128", "partial6"])
def test_sink_split_narrow_graph_rebind(case, cudnn_handle):
    _check_sink_split(case, "bfloat16", "log2", graph_api=True, handle=cudnn_handle, tile_n=64)


def _check_sink_split(case, dtype_name, stats, *, graph_api, handle=None, tile_n=128, sink=True):
    import cudnn
    import torch
    from cuda.bindings import driver as cuda
    from test_sdpa_fwd_dsl_sm107 import _dense_mask, _half_oracle, _half_prefold_api, _rubin_only

    _rubin_only()
    dims, group, sq, sk, cga, splits, packed, causal, window, kernel = CASES[case]
    api, bufs = _half_prefold_api(
        *dims,
        getattr(torch, dtype_name),
        prefolded=False,
        with_stats=stats != "none",
        split_kv=splits,
        b=2,
        hq=2 * group,
        hkv=2,
        s_q=sq,
        s_kv=sk,
        causal=causal,
        bottom_right=causal,
        window_left=window,
        sink=sink,
        pack_gqa=packed,
    )
    api.cga = cga
    api.tile_n = tile_n
    api.sched_policy = 0 if stats == "none" else 1
    api.stats_log2 = stats == "log2"
    assert api.check_support()
    stream = torch.cuda.Stream()
    saved_stream = cudnn.get_stream(handle) if graph_api else None
    captures, bindings = [], []
    if graph_api:
        dt = cudnn.data_type.BFLOAT16 if dtype_name == "bfloat16" else cudnn.data_type.HALF
        graph = cudnn.pygraph(io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        ts = {n: graph.tensor_like(bufs[n]) for n in ("q", "k", "v")}
        if sink:
            ts["sinks"] = graph.tensor_like(bufs["sinks"]).set_dim([1, 2 * group, 1, 1]).set_stride([2 * group, 1, 1, 1])
        ts["out"], lse = graph.sdpa(
            q=ts["q"],
            k=ts["k"],
            v=ts["v"],
            generate_stats=stats != "none",
            attn_scale=bufs["attn_scale"],
            use_causal_mask_bottom_right=causal,
            sliding_window_length=None if window is None else window + 1,
            sink_token=ts.get("sinks"),
            stats_use_log2=stats == "log2",
        )
        ts["out"].set_output(True).set_dim(list(bufs["out"].shape)).set_stride(list(bufs["out"].stride()))
        if stats != "none":
            bufs["lse"] = bufs["lse"].unsqueeze(-1)
            ts["lse"] = lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(list(bufs["lse"].shape)).set_stride(list(bufs["lse"].stride()))
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        idx = next(i for i in range(graph.get_execution_plan_count()) if graph.get_plan_name_at_index(i).startswith("sdpa_fwd_prefill_sm107"))
        engine, knobs = graph.get_engine_and_knobs_at_index(idx)
        knobs.update(
            {
                cudnn.knob_type.TILE_M: 128,
                cudnn.knob_type.TILE_N: tile_n,
                cudnn.knob_type.TILE_CGA_M: cga,
                cudnn.knob_type.SPLIT_KV: splits,
                cudnn.knob_type.PACK_GQA: int(packed),
                cudnn.knob_type.SCHED_POLICY: 0 if dims == (192, 128) else 1,
            }
        )
        graph.create_execution_plan(engine, knobs)
        graph.select_plan(graph.get_execution_plan_count() - 1)
        graph.check_support()
        graph.build_plans()
        api = inspect.getclosurevars(graph._compiled_plans[graph._plan_index]._compiled.default_stream).nonlocals["api"]
        ws_bytes = graph.get_workspace_size()
        cudnn.set_stream(handle, stream.cuda_stream)
    else:
        api.compile()
        ws_bytes = api.scratch_workspace_bytes()
    if max(dims) <= 64 and cga == 1 and tile_n == 128 and (splits > 1 or api._d64_decode_tile()):
        kernel = "sm100/decode_d128_f16.py"
        assert api._k_mod.CFG.TILES_Q == 1
    assert api._k_mod.__file__.endswith(kernel), api._k_mod.__file__
    assert api._k_mod.CFG.SPLIT_KV == splits
    if max(dims) <= 64:
        assert api.flavor == (64, 64) and api._k_mod.CFG.TILE_K == api._k_mod.CFG.TILE_O == 64
        assert api._k_mod.CFG.CTA_MMA == cga
        if splits > 1:
            if kernel.startswith("sm107/"):
                assert not api._k_mod.FOLD_SINK
            assert api._fp32_partial_split() and api._partial_torch_dtype() == torch.float32
        else:
            assert not api._fp32_partial_split()
    if tile_n == 64:
        assert api._k_mod.CFG.TILE_N == 64 and api._k_mod.CFG.S_SLOTS == 2
    try:
        for ordinal in range(2):
            current = {n: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="cuda") for n, t in bufs.items() if isinstance(t, torch.Tensor)}
            for n, t in current.items():
                t.copy_(bufs[n])
            current["v"].mul_(1 if ordinal == 0 else -1)
            if sink:
                current["sinks"].add_(ordinal * 2)
            ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device="cuda")

            def execute():
                if graph_api:
                    graph.execute({ts[n]: t for n, t in current.items()}, ws, handle=handle)
                else:
                    api.execute(
                        q_tensor=current["q"],
                        k_tensor=current["k"],
                        v_tensor=current["v"],
                        o_tensor=current["out"],
                        lse_tensor=current.get("lse"),
                        sinks=current.get("sinks"),
                        workspace=ws,
                        current_stream=cuda.CUstream(stream.cuda_stream),
                    )

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                execute()
            stream.synchronize()
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                execute()
            captures.append(capture)
            bindings.append((current, ws))
        for capture, (current, ws) in zip(captures, bindings):
            current["v"].neg_()
            if sink:
                current["sinks"].add_(0.75)
            current["out"].fill_(torch.nan)
            ws.fill_(0xA5)
            if "lse" in current:
                current["lse"].fill_(torch.nan)
            torch.cuda.synchronize()
            capture.replay()
            torch.cuda.synchronize()
            ref_o, ref_lse = _half_oracle(
                current["q"],
                current["k"],
                current["v"],
                logit_scale=bufs["attn_scale"],
                masked=_dense_mask(sq, sk, causal=causal, bottom_right=causal, window_left=window),
                sinks=current.get("sinks"),
            )
            torch.testing.assert_close(current["out"].double(), ref_o, atol=4e-3, rtol=4e-3)
            if "lse" in current:
                expected = ref_lse / math.log(2) if stats == "log2" else ref_lse
                torch.testing.assert_close(current["lse"].reshape_as(expected).double(), expected, atol=6e-4, rtol=0)
    finally:
        for capture in captures:
            capture.reset()
        if graph_api:
            cudnn.set_stream(handle, saved_stream)


@pytest.mark.parametrize("s_q", [1, 4])
@pytest.mark.parametrize("sink_logit", [-2.0, -1000.0])
def test_sink_split_paged_decode_keyless(s_q, sink_logit):
    import cudnn
    import torch
    from test_sdpa_fwd_decode_d256_sm107 import _run_graph
    from test_sdpa_fwd_dsl_sm107 import _rubin_only

    _rubin_only()
    plan = _run_graph(
        B=3,
        H=16,
        KH=2,
        s_q=s_q,
        lens=[700, 0, 1],
        page=16,
        sink=sink_logit,
        causal_br=True,
        dtype=torch.bfloat16,
        stats_log2=True,
        explicit_knobs={cudnn.knob_type.SPLIT_KV: 8, cudnn.knob_type.PACK_GQA: 1},
    )
    assert plan.knobs.split_kv == 8 and plan.knobs.pack_gqa is True
