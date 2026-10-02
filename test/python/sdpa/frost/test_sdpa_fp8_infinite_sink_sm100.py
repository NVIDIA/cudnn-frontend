# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 d64/d128 sink limits, with constant logits and an analytic oracle."""

import math
import pytest
import torch
from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl, select_engine
from cudnn.sdpa.fwd.engines import engine_name

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]


@pytest.mark.L0
@pytest.mark.parametrize(
    "d,in_dtype,out_dtype,hkv,stats_log2",
    [
        (128, torch.float8_e4m3fn, torch.float16, 8, False),
        (64, torch.float8_e5m2, torch.bfloat16, 8, True),
        (64, torch.float8_e4m3fn, torch.float8_e4m3fn, 2, False),
        (128, torch.float8_e5m2, torch.float8_e5m2, 2, None),
    ],
    ids=["d128-e4m3-half", "d64-e5m2-bfloat16-log2", "d64-e4m3-gqa", "d128-e5m2-gqa-no-stats"],
)
def test_infinite_sink_rebind_and_replay(d, in_dtype, out_dtype, hkv, stats_log2, monkeypatch):
    import cudnn
    import cutlass.cute as cute

    b, hq, sq, skv = 3, 8, 256, 256
    # Q.K / d = 1. V is constant within each KV head; FP8 P quantization is exact.
    q = torch.ones(b, sq, hq, d, device="cuda").to(in_dtype).transpose(1, 2)
    k = torch.ones(b, skv, hkv, d, device="cuda").to(in_dtype).transpose(1, 2)
    values = torch.arange(1, hkv + 1, device="cuda", dtype=torch.float32) / 8
    v = values.view(1, 1, hkv, 1).expand(b, skv, hkv, d).contiguous().to(in_dtype).transpose(1, 2)
    o = torch.empty(b, sq, hq, d, device="cuda", dtype=out_dtype).transpose(1, 2)
    lse = torch.empty(b, hq, sq, 1, device="cuda")
    amax = torch.empty(1, 1, 1, 1, device="cuda")
    one = torch.ones(1, 1, 1, 1, device="cuda")
    q_lens = torch.tensor([129, 256, 0], device="cuda", dtype=torch.int32).reshape(b, 1, 1, 1)
    kv_lens = torch.tensor([0, 137, 256], device="cuda", dtype=torch.int32).reshape(b, 1, 1, 1)
    initial_sinks = torch.zeros(1, hq, 1, 1, device="cuda")
    sinks = torch.tensor([float("inf"), 1000.0, float("-inf"), 0.0, 1.0, float("inf"), -1000.0, 3.0], device="cuda").reshape(1, hq, 1, 1)
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    qt, kt, vt, st, qlt, klt, unit = [g.tensor_like(x) for x in (q, k, v, initial_sinks, q_lens, kv_lens, one)]
    kwargs = {} if stats_log2 is None else {"stats_use_log2": stats_log2}
    ot, lt, _, at = g.sdpa_fp8(
        q=qt,
        k=kt,
        v=vt,
        descale_q=unit,
        descale_k=unit,
        descale_v=unit,
        descale_s=unit,
        scale_s=unit,
        scale_o=unit,
        attn_scale=1.0 / d,
        sink_token=st,
        generate_stats=stats_log2 is not None,
        use_padding_mask=True,
        seq_len_q=qlt,
        seq_len_kv=klt,
        use_causal_mask_bottom_right=True,
        **kwargs,
    )
    out_type = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
    }[out_dtype]
    ot.set_output(True).set_dim(list(o.shape)).set_stride(list(o.stride())).set_data_type(out_type)
    at.set_output(True).set_dim(list(amax.shape)).set_stride(list(amax.stride())).set_data_type(cudnn.data_type.FLOAT)
    vp = {qt: q, kt: k, vt: v, st: initial_sinks, qlt: q_lens, klt: kv_lens, unit: one, ot: o, at: amax}
    if stats_log2 is not None:
        lt.set_output(True).set_dim(list(lse.shape)).set_stride(list(lse.stride())).set_data_type(cudnn.data_type.FLOAT)
        vp[lt] = lse
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine_name(arch="sm100", fp8=True), pack_gqa=hkv != hq)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute(vp, ws)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    vp[st] = sinks  # Fresh storage, same declared graph.
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must reuse its compiled plan"))
    # BR alignment: row i sees clamp(i + KV - Q + 1, 0, KV) keys.
    rows = torch.arange(sq, device="cuda")[None, :]
    live = rows < q_lens.reshape(b, 1)
    counts = (rows + kv_lens.reshape(b, 1) - q_lens.reshape(b, 1) + 1).clamp_min(0)
    counts = torch.minimum(counts, kv_lens.reshape(b, 1))
    base_lse = counts.double().log() + 1.0
    head_values = values.repeat_interleave(hq // hkv).double()

    def check():
        expected_lse = torch.logaddexp(base_lse[:, None, :], sinks.double().view(1, hq, 1))
        factor = torch.where(counts[:, None, :] > 0, (base_lse[:, None, :] - expected_lse).exp(), 0.0)
        expected_o = torch.where(live[:, None, :], factor * head_values[None, :, None], 0.0)
        expected_lse = torch.where(live[:, None, :], expected_lse, float("-inf"))
        expected_cast = expected_o[..., None].expand_as(o).to(out_dtype).float()
        atol = {torch.float16: 1e-3, torch.bfloat16: 4e-3, torch.float8_e4m3fn: 0.0, torch.float8_e5m2: 0.0}[out_dtype]
        torch.testing.assert_close(o.float(), expected_cast, atol=atol, rtol=0)
        assert (o.float()[:, torch.isposinf(sinks.view(hq))] == 0).all(), "positive-infinite sinks must produce exact zero"
        torch.testing.assert_close(amax, expected_o.abs().max().float().reshape_as(amax), atol=1e-3, rtol=0)
        if stats_log2 is not None:
            if stats_log2:
                expected_lse = expected_lse * math.log2(math.e)
            torch.testing.assert_close(lse.squeeze(-1), expected_lse.float(), atol=1e-3, rtol=0)

    def execute():
        prior_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            g.execute(vp, ws)
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        finally:
            torch.cuda.set_sync_debug_mode(prior_mode)

    execute()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured, stream=stream):
            execute()
        torch.cuda.current_stream().wait_stream(stream)
        for value in (float("inf"), float("-inf"), 2.0, 1000.0):
            sinks.fill_(value)
            o.fill_(float("nan"))
            lse.fill_(float("nan"))
            amax.fill_(999.0)
            captured.replay()
            check()
    finally:
        captured.reset()


@pytest.mark.L0
@pytest.mark.parametrize("block_scaled_o", ["mxfp8", "nvfp4"])
def test_infinite_sink_block_scaled_output(block_scaled_o):
    from sdpa.frost.test_sdpa_fwd_fp8_sm100 import _run

    sink = torch.full((1, 8, 1, 1), float("inf"), device="cuda")
    result = _run(2, 8, 8, 256, 256, "e4m3", torch.float8_e4m3fn, scale=1.0 / math.sqrt(128), sdpa_kwargs={}, sink=sink, block_scaled_o=block_scaled_o)
    # The usual FP8 reference need not evaluate inf-inf: this limit is exact.
    assert (result.output == 0).all()
    assert result.amax == 0.0
    assert result.sf_pad_ok


@pytest.mark.L0
def test_infinite_sink_packed_empty_sequences():
    from sdpa.frost.test_sdpa_fwd_fp8_sm100 import _run_thd

    sink = torch.full((1, 8, 1, 1), float("inf"), device="cuda")
    output, _, amax, _, stats, _ = _run_thd([129, 0, 200], [0, 256, 137], 8, 8, "e4m3", scale=1.0 / math.sqrt(128), bottom_right=True, sink=sink, stats=True)
    assert (output == 0).all()
    assert amax == 0.0
    assert torch.isposinf(stats).all()
