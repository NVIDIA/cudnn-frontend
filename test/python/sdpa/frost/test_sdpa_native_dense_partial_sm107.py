# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph and standalone native partial PackGQA admission and retained captures."""

import inspect
import math

import pytest

from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L1, requires_dsl]


@pytest.mark.parametrize("group", [6, 12, 24, 192])
def test_native_dense_partial_admission(group, monkeypatch):
    """A partial-group pin reaches the native tile and keeps other route boundaries."""
    import dataclasses

    from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts

    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    caps = _caps("sdpa_fwd_prefill_sm107")
    facts = _f16_facts(h_q=group * 2, h_kv=2)
    knobs = engines.SdpaFwdKnobs(cga=2, pack_gqa=True, split_kv=1)
    for dims in ((128, 128), (64, 64), (96, 80)):
        f = dataclasses.replace(facts, d_qk=dims[0], d_v=dims[1])
        assert engines.mismatch(caps, f, knobs) is None
        assert engines.mismatch(caps, f, dataclasses.replace(knobs, split_kv=2)) is None
    assert engines.mismatch(caps, facts, dataclasses.replace(knobs, cga=1)) is not None
    for changed in (dict(attn_scale_prefolded=True), dict(h_q=6), dict(d_qk=256, d_v=256), dict(has_paged_kv=True)):
        assert engines.mismatch(caps, dataclasses.replace(facts, **changed), knobs) is not None


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("with_stats", [False, True])
@pytest.mark.parametrize("sched_policy", [0, 1, 2])
@pytest.mark.parametrize("geometry", ["partial6", "partial12_sink", "partial24_split"])
def test_native_dense_partial_graph_rebind(cudnn_handle, dtype_name, with_stats, sched_policy, geometry):
    """Rebind both captures; retain the graph's existing LPT-L2 rejection."""
    import cudnn
    import torch

    from test_sdpa_fwd_dsl_sm107 import _dense_mask, _half_oracle, _half_prefold_api, _rubin_only

    _rubin_only()
    group, dims, sq, sk, splits, causal, sink = {
        "partial6": (6, (128, 128), 129, 257, 1, True, False),
        "partial12_sink": (12, (96, 80), 257, 257, 1, True, True),
        "partial24_split": (24, (64, 64), 129, 385, 2, True, False),
    }[geometry]
    api, bufs = _half_prefold_api(
        *dims,
        getattr(torch, dtype_name),
        prefolded=False,
        with_stats=with_stats,
        split_kv=splits,
        b=2,
        hq=2 * group,
        hkv=2,
        s_q=sq,
        s_kv=sk,
        causal=causal,
        bottom_right=causal,
        sink=sink,
        pack_gqa=True,
    )
    assert api.check_support(), "standalone admission must match the public graph"
    if with_stats:
        bufs["lse"] = bufs["lse"].unsqueeze(-1)
    if geometry == "partial12_sink":
        # Dense bottom-right graphs require Q capacity <= KV capacity. Device
        # lengths still exercise keyless rows and changing live KV prefixes.
        bufs["seq_q"] = torch.full((2, 1, 1, 1), sq, dtype=torch.int32, device="cuda")
        bufs["seq_kv"] = torch.full((2, 1, 1, 1), 65, dtype=torch.int32, device="cuda")
    dt = cudnn.data_type.BFLOAT16 if dtype_name == "bfloat16" else cudnn.data_type.HALF
    g = cudnn.pygraph(io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tensors = {n: g.tensor_like(bufs[n]) for n in ("q", "k", "v")}
    if "seq_kv" in bufs:
        tensors.update({n: g.tensor_like(bufs[n]) for n in ("seq_q", "seq_kv")})
    if sink:
        tensors["sinks"] = g.tensor_like(bufs["sinks"]).set_dim([1, 2 * group, 1, 1]).set_stride([2 * group, 1, 1, 1])
    tensors["out"], stats = g.sdpa(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        generate_stats=with_stats,
        attn_scale=bufs["attn_scale"],
        use_causal_mask_bottom_right=causal,
        sink_token=tensors.get("sinks"),
        use_padding_mask="seq_kv" in tensors,
        seq_len_q=tensors.get("seq_q"),
        seq_len_kv=tensors.get("seq_kv"),
    )
    tensors["out"].set_output(True).set_dim(list(bufs["out"].shape)).set_stride(list(bufs["out"].stride()))
    if with_stats:
        tensors["lse"] = stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        stats.set_dim(list(bufs["lse"].shape)).set_stride(list(bufs["lse"].stride()))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    index = next(i for i in range(g.get_execution_plan_count()) if g.get_plan_name_at_index(i).startswith("sdpa_fwd_prefill_sm107"))
    engine, knobs = g.get_engine_and_knobs_at_index(index)
    # Pin the same tile knobs for native D64 and the larger flavors.
    chosen = {
        **knobs,
        cudnn.knob_type.TILE_M: 128,
        cudnn.knob_type.TILE_N: 128,
        cudnn.knob_type.TILE_CGA_M: 2,
        cudnn.knob_type.PACK_GQA: 1,
        cudnn.knob_type.SPLIT_KV: splits,
        cudnn.knob_type.SCHED_POLICY: sched_policy,
    }
    g.create_execution_plan(engine, chosen)
    g.select_plan(g.get_execution_plan_count() - 1)
    g.check_support()
    if sched_policy == 2:
        # The direct dense kernel accepts LPT-L2, but the public graph's
        # established scheduler domain is NATURAL/LPT. Partial-group admission
        # must not silently widen that independent contract.
        with pytest.raises(NotImplementedError, match=r"sched_policy=2 is outside this engine's domain \[0, 1\]"):
            g.build_plans()
        return
    g.build_plans()
    selected = g._compiled_plans[g._plan_index]
    compiled_api = inspect.getclosurevars(selected._compiled.default_stream).nonlocals["api"]
    assert compiled_api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
    cfg = compiled_api._k_mod.CFG
    assert (cfg.PACK_G, cfg.CGA_M, cfg.SPLIT_KV) == (math.gcd(group, 128), 2, splits)
    stream = torch.cuda.Stream()
    saved_stream = cudnn.get_stream(cudnn_handle)
    captures, bindings = [], []
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        for ordinal in range(2):
            current = {n: torch.empty_strided(bufs[n].shape, bufs[n].stride(), dtype=bufs[n].dtype, device="cuda") for n in tensors}
            for n, value in current.items():
                value.copy_(bufs[n])
            current["v"].mul_(1 if ordinal == 0 else -1)
            ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
            pack = {tensors[n]: value for n, value in current.items()}
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                g.execute(pack, ws, handle=cudnn_handle)
            stream.synchronize()
            capture = torch.cuda.CUDAGraph()
            captures.append(capture)
            with torch.cuda.graph(capture, stream=stream):
                g.execute(pack, ws, handle=cudnn_handle)
            bindings.append((current, ws))
        assert bindings[0][0]["q"].data_ptr() != bindings[1][0]["q"].data_ptr()
        for capture, (current, ws) in zip(captures, bindings):
            for replay in range(2):
                current["v"].neg_()
                live_k = sk
                if "seq_kv" in current:
                    live_k = (65, 33)[replay]
                    current["seq_kv"].fill_(live_k)
                    current["k"].copy_(bufs["k"])
                    current["v"].copy_(bufs["v"])
                    current["v"].mul_(1 if replay == 0 else -1)
                    current["k"][..., live_k:, :].fill_(31)
                    current["v"][..., live_k:, :].fill_(37)
                current["out"].fill_(torch.nan)
                ws.fill_(0xA5)
                if with_stats:
                    current["lse"].fill_(torch.nan)
                torch.cuda.synchronize()
                capture.replay()
                torch.cuda.synchronize()
                ref_o, ref_lse = _half_oracle(
                    current["q"],
                    current["k"][..., :live_k, :],
                    current["v"][..., :live_k, :],
                    logit_scale=bufs["attn_scale"],
                    masked=_dense_mask(sq, live_k, causal=causal, bottom_right=causal),
                    sinks=current.get("sinks"),
                )
                torch.testing.assert_close(current["out"].double(), ref_o, atol=4e-3, rtol=4e-3)
                if with_stats:
                    torch.testing.assert_close(current["lse"].squeeze(-1).double(), ref_lse, atol=4e-4, rtol=0)
    finally:
        for capture in captures:
            capture.reset()
        cudnn.set_stream(cudnn_handle, saved_stream)
