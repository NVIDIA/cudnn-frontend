# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free unit tests for the SDPA facts analyzer and per-engine capability probes."""

from __future__ import annotations

import cudnn
import pytest
import torch

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.bwd import engines as bwd_engines
from cudnn.sdpa.fwd import engines


def _eligible(graph, knobs=None):
    """Names of the FROST SDPA engines whose caps match this graph.

    ``knobs`` is passed straight to the probe: graph.set_engine_knobs() was
    removed with the monkey-patch dispatch layer, and a knob request is a
    property of a PLAN (engines.base.PlanConfig.knobs), not of the graph.
    """
    return {s.name for s in engines.ENGINE_SPECS if engines.analyze_for(s, graph, knobs)[1] is None}


# The default pytest.ini addopts is `-m L0`; mark the whole module so it runs.
pytestmark = pytest.mark.L0

B, H, S, D = 2, 8, 256, 512
DIMS = (B, H, S, D)
STRIDES = (S * H * D, D, H * D, 1)
DTYPE = cudnn.data_type.HALF


@pytest.fixture(autouse=True)
def _fake_sm100(monkeypatch):
    """Fake an SM100 device so the device-family gate passes without a real GPU."""
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))


def _mk_graph() -> cudnn.pygraph:
    return cudnn.pygraph(
        io_data_type=DTYPE,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )


def _mk_qkv(g: cudnn.pygraph, d: int = D):
    dims = (B, H, S, d)
    strides = (S * H * d, d, H * d, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    return q, k, v, dims, strides


def _finish_output(o, dims, strides, dtype=DTYPE):
    """Set O's dims/stride/dtype directly, standing in for build_operation_graph()."""
    o.set_output(True).set_dim(dims).set_stride(strides)
    o.set_data_type(dtype)


def _facts(graph):
    facts = ga.analyze(graph)
    assert facts is not None, "expected a single SDPA node on the graph"
    assert facts.invalid is None, facts.invalid
    return facts


def test_engines_registered():
    """The family ships in the library's static engine table, with an id block
    wide enough for every spec (ids are identity and never move)."""
    from cudnn.engines import MANIFEST, is_python_engine

    (row,) = [r for r in MANIFEST if r.factory == "FrostSdpaFwdEngines"]
    assert is_python_engine(row.engine_id)
    assert row.id_end - row.engine_id >= len(engines.ENGINE_SPECS)
    assert engines.engine_name() == "sdpa_fwd_prefill_sm100"
    assert engines.engine_name(arch="sm107", fp8=True) == "sdpa_fwd_prefill_sm107_fp8"


def test_single_sdpa_node_found():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, dims, strides)
    node = ga._single_sdpa_node(g)
    assert node is not None
    rec = ga._record_from_node(node)
    assert rec["q"] is q and rec["k"] is k and rec["v"] is v
    assert rec["o"] is o
    assert rec["use_causal_mask"] is True
    assert rec["attn_scale"] == 0.1


def test_probe_accepts_dsv4_causal():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, dims, strides)
    assert engines.engine_name() in _eligible(g)


def test_probe_accepts_bf16():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dims = (B, H, S, D)
    strides = (S * H * D, D, H * D, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.BFLOAT16, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.BFLOAT16, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.BFLOAT16, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides, dtype=cudnn.data_type.BFLOAT16)
    assert engines.engine_name() in _eligible(g)


def test_probe_rejects_uncoverable_head_dim():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=520)  # beyond the largest (d512) envelope
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_misaligned_head_dim():
    # Envelope zero-padding requires d % 8 == 0 (TMA 16-byte global-stride rule
    # at 2 bytes/elem); d=60 is covered by every flavor but misaligned.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=60)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_envelope_covers_small_head_dim():
    # d=64 <= every f16 flavor's envelope: all three are eligible, and the
    # registration order (smallest-first) makes d128 the auto-select winner.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=64)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    elig = _eligible(g)
    assert engines.engine_name() in elig
    ordered = [s.name for s in engines.ENGINE_SPECS if s.name in elig]
    assert ordered[0] == engines.engine_name()


def test_probe_envelope_mixed_dims_pick_covering_flavor():
    # (d_qk=192, d_v=128) uses the native d192/d128 flavor; larger envelopes
    # remain eligible for explicit A/B selection.
    g = _mk_graph()
    d_qk, d_v = 192, 128
    q = g.tensor(dim=(B, H, S, d_qk), stride=(S * H * d_qk, d_qk, H * d_qk, 1), data_type=DTYPE, name="q")
    k = g.tensor(dim=(B, H, S, d_qk), stride=(S * H * d_qk, d_qk, H * d_qk, 1), data_type=DTYPE, name="k")
    v = g.tensor(dim=(B, H, S, d_v), stride=(S * H * d_v, d_v, H * d_v, 1), data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, (B, H, S, d_v), (S * H * d_v, d_v, H * d_v, 1))
    elig = _eligible(g)
    assert engines.engine_name() in elig
    ordered = [s.name for s in engines.ENGINE_SPECS if s.name in elig]
    assert ordered[0] == engines.engine_name()


def test_d192_fp8_sink_dtype_support():
    spec = next(s for s in engines.ENGINE_SPECS if s.name == engines.engine_name(fp8=True))

    def facts(dtype, *, sink):
        return ga.SdpaGraphFacts(
            b=1,
            h_q=8,
            h_kv=8,
            s_q=256,
            s_kv=256,
            d_qk=192,
            d_v=128,
            dtype=dtype,
            dtype_o=cudnn.data_type.HALF,
            is_fp8=True,
            has_sink=sink,
            device_cc=(10, 0),
        )

    assert engines.mismatch(spec.capabilities, facts(cudnn.data_type.FP8_E4M3, sink=True)) is None
    assert engines.mismatch(spec.capabilities, facts(cudnn.data_type.FP8_E5M2, sink=False)) is None
    assert engines.mismatch(spec.capabilities, facts(cudnn.data_type.FP8_E5M2, sink=True)) is None


def test_probe_rejects_wrong_device_family(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (8, 0))
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_bias():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    bias = g.tensor(dim=(1, H, S, S), stride=(H * S * S, S * S, S, 1), data_type=cudnn.data_type.FLOAT, name="bias")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, bias=bias)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_alibi():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_alibi_mask=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_second_op_on_graph():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    r = g.relu(input=o, name="r")
    r.set_output(True).set_dim(dims).set_stride(strides)
    assert not _eligible(g)


def test_probe_rejects_padding_mask_without_seq_len_kv():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_padding_mask=True)
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_accepts_seq_len_q_with_padding_mask():
    # padding_mask requires a seq_len_q companion; the engine accepts it (KV-only trim).
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_padding_mask=True,
        seq_len_kv=seq_kv,
        seq_len_q=seq_q,
    )
    _finish_output(o, dims, strides)
    assert engines.engine_name() in _eligible(g)


def test_probe_rejects_non_int32_seq_len():
    """The kernels consume per-batch lengths as int32 directly — no implicit
    cast anywhere on the execute path — so an int64 seq_len is ineligible."""
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="seq_kv64")
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_padding_mask=True,
        seq_len_kv=seq_kv,
    )
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_accepts_bottom_right_with_padded_seq_len_q():
    # The kernels anchor the BR diagonal at the per-batch
    # (seq_len_q[b], seq_len_kv[b]) corner, so dense padding with per-batch
    # seq_len_q is served (it used to be gated while the diagonal was anchored
    # at the global S_q).
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        seq_len_kv=seq_kv,
        seq_len_q=seq_q,
    )
    _finish_output(o, dims, strides)
    assert engines.engine_name() in _eligible(g)


def test_probe_rejects_seq_len_q_without_padding_mask():
    # Bare seq_len_q is per-batch Q trimming, which the kernel has no path for.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        seq_len_q=seq_q,
    )
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def _mk_thd_qkvo(g, *, mask_kwargs, d: int = D):
    """Build a ragged (THD) sdpa graph with the given mask kwargs."""
    dims = (B, H, S, d)
    strides = (S * H * d, d, H * d, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    ro = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="ro")
    q.set_ragged_offset(ro)
    k.set_ragged_offset(ro)
    v.set_ragged_offset(ro)
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="sq")
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="skv")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_padding_mask=True, seq_len_q=seq_q, seq_len_kv=seq_kv, **mask_kwargs)
    o.set_output(True).set_dim(dims).set_stride(strides)
    o.set_data_type(DTYPE)
    o.set_ragged_offset(ro)


def test_probe_accepts_thd_top_left_causal():
    g = _mk_graph()
    _mk_thd_qkvo(g, mask_kwargs=dict(use_causal_mask=True))
    assert engines.engine_name() in _eligible(g)


def test_probe_accepts_thd_bottom_right():
    # The SM100 kernels anchor the THD bottom-right diagonal at each sequence's
    # own (seq_len_q[b], seq_len_kv[b]) via the cu_seqlen metadata.
    g = _mk_graph()
    _mk_thd_qkvo(g, mask_kwargs=dict(use_causal_mask_bottom_right=True))
    assert engines.engine_name() in _eligible(g)


def test_probe_accepts_thd_stats():
    """The SM100 epilogue writes cuDNN's ragged Stats directly (token-major
    or head-major packed LSE), so THD + generate_stats is eligible."""
    g = _mk_graph()
    dims = (B, H, S, D)
    strides = (S * H * D, D, H * D, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    ro = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="ro")
    q.set_ragged_offset(ro)
    k.set_ragged_offset(ro)
    v.set_ragged_offset(ro)
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="sq")
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="skv")
    o, stats = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        generate_stats=True,
        use_causal_mask=True,
        use_padding_mask=True,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
    )
    _finish_output(o, dims, strides)
    o.set_ragged_offset(ro)
    assert stats is not None
    stats.set_output(True).set_dim((B, H, S, 1)).set_stride((S * H, 1, H, 1))
    stats.set_data_type(cudnn.data_type.FLOAT)
    stats_ro = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="stats_ro")
    stats.set_ragged_offset(stats_ro)
    assert engines.engine_name() in _eligible(g)


def test_probe_accepts_right_band_widening():
    # diagonal_band_right_bound > 0 lowers as MASK_CAUSAL with a compile-time
    # BAND_RIGHT diagonal offset.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, diagonal_band_right_bound=16)
    _finish_output(o, dims, strides)
    assert engines.engine_name() in _eligible(g)
    facts = ga.analyze(g)
    assert facts.right_band_widening and facts.right_bound == 16 and not facts.causal


def test_probe_rejects_negative_right_band():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    try:
        o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, diagonal_band_right_bound=-4)
    except (RuntimeError, ValueError):
        return  # the pygraph binding may reject it before the probe ever runs
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_bad_sink_shape():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    bad_sink = g.tensor(dim=(1, H, 2, 1), stride=(2 * H, 2, 1, 1), data_type=cudnn.data_type.FLOAT, name="badsink")
    try:
        o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, sink_token=bad_sink)
    except TypeError:
        pytest.skip("this cuDNN wheel's sdpa() binding predates sink_token")
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_rejects_non_fp32_sink():
    """The kernels consume fp32 sink logits directly — no implicit cast anywhere
    on the execute path — so a non-fp32 sink token is ineligible up front."""
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    bad_sink = g.tensor(dim=(1, H, 1, 1), stride=(H, 1, 1, 1), data_type=cudnn.data_type.BFLOAT16, name="bf16sink")
    try:
        o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, sink_token=bad_sink)
    except TypeError:
        pytest.skip("this cuDNN wheel's sdpa() binding predates sink_token")
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_resolve_causal_plus_swa():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_causal_mask=True,
        sliding_window_length=128,
    )
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.causal is True
    assert cfg.bottom_right is False
    # cuDNN's sliding_window_length is a length; Frost's swa_window is an offset
    # (length - 1), so 128 maps to 127.
    assert cfg.window_left == 127


def test_resolve_plain_swa_no_causal():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, sliding_window_length=64)
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.causal is False
    assert cfg.window_left == 63


def test_resolve_causal_bottom_right():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask_bottom_right=True)
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.causal is True
    assert cfg.bottom_right is True


def test_resolve_padding_mask_with_seq_len_kv():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_padding_mask=True, seq_len_kv=seq_kv)
    _finish_output(o, dims, strides)
    cfg = _facts(g)
    assert cfg.padded is True
    assert engines.engine_name() in _eligible(g)


def test_resolve_generate_stats():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, stats = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, generate_stats=True)
    _finish_output(o, dims, strides)
    if stats is not None:
        stats.set_output(True).set_dim((B, H, S, 1)).set_stride((H * S, S, 1, 1))
        stats.set_data_type(cudnn.data_type.FLOAT)
    cfg = _facts(g)
    assert cfg.wants_stats is True
    assert cfg.stats_t is not None


def test_probe_rejects_bottom_right_swa_only():
    # Kernel gap: CAUSAL_BOTTOM_RIGHT requires MASK_CAUSAL; BOTTOM_RIGHT
    # alignment with only a left band has no causal bit.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        diagonal_band_left_bound=64,
        diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT,
    )
    _finish_output(o, dims, strides)
    assert not _eligible(g)


def test_probe_accepts_ragged_skv_via_synth_padding():
    # KV tail (S_kv % 128 != 0) with no covering mask: the f16 rows opt into
    # skv_tail_via_padding — the lowering synthesizes full-length per-batch KV
    # lengths and the padded path masks the tail (the FP8 row's mechanism).
    g = _mk_graph()
    s_kv = 300
    q = g.tensor(dim=(B, H, S, D), stride=(S * H * D, D, H * D, 1), data_type=DTYPE, name="q")
    k = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="k")
    v = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, (B, H, S, D), (S * H * D, D, H * D, 1))
    assert engines.engine_name() in _eligible(g)


def test_probe_accepts_ragged_skv_with_top_left_causal():
    # Top-left causal with S_q <= S_kv provably masks the KV tail columns.
    g = _mk_graph()
    s_kv = 300
    q = g.tensor(dim=(B, H, S, D), stride=(S * H * D, D, H * D, 1), data_type=DTYPE, name="q")
    k = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="k")
    v = g.tensor(dim=(B, H, s_kv, D), stride=(s_kv * H * D, D, H * D, 1), data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, (B, H, S, D), (S * H * D, D, H * D, 1))
    assert engines.engine_name() in _eligible(g)


def _mk_eligible_graph():
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, dims, strides)
    return g


def test_knob_request_within_domain_keeps_engine_eligible():
    g = _mk_eligible_graph()
    assert engines.engine_name() in _eligible(g, engines.SdpaFwdKnobs(sched_policy=0, tile_m=128, tile_n=128, cga=2))


def test_knob_request_outside_domain_rejects_engine():
    # A value no row's domain contains: honored or ineligible, never degraded.
    g = _mk_eligible_graph()
    assert not _eligible(g, engines.SdpaFwdKnobs(sched_policy=99))
    # softmax_precision=1 is cudnn.data_type.DOUBLE — in no row's domain (the
    # fp8 rows serve FLOAT, and the sm107 row additionally HALF), so an
    # explicit request declines everywhere.
    assert not _eligible(g, engines.SdpaFwdKnobs(softmax_precision=1))


def test_knob_request_lpt_sched_is_in_domain():
    # The SM100 rows advertise all three scheduler policies (the static/CLC
    # remap serves them); an explicit LPT request stays eligible.
    g = _mk_eligible_graph()
    assert engines.engine_name() in _eligible(g, engines.SdpaFwdKnobs(sched_policy=1))


def test_knob_request_unsupported_tile_rejects_engine():
    g = _mk_eligible_graph()
    assert not _eligible(g, engines.SdpaFwdKnobs(tile_n=64))


def test_knob_request_unsupported_q_tile_rejects_engine():
    g = _mk_eligible_graph()
    assert not _eligible(g, engines.SdpaFwdKnobs(tile_m=64))


def test_knob_request_wrong_vocabulary_rejects_engine():
    # A different op's knob object must not silently pass.
    g = _mk_eligible_graph()
    assert not _eligible(g, object())


def test_knob_request_none_fields_are_no_preference():
    g = _mk_eligible_graph()
    assert engines.engine_name() in _eligible(g, engines.SdpaFwdKnobs())


def _mk_gqa_graph(h_q, h_kv, d=128):
    """Causal GQA graph (BSHD-physical strides) for the pack_gqa knob tests."""
    g = _mk_graph()
    dims_q, strides_q = (B, h_q, S, d), (S * h_q * d, d, h_q * d, 1)
    dims_kv, strides_kv = (B, h_kv, S, d), (S * h_kv * d, d, h_kv * d, 1)
    q = g.tensor(dim=dims_q, stride=strides_q, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims_kv, stride=strides_kv, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims_kv, stride=strides_kv, data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, dims_q, strides_q)
    return g


def test_knob_request_pack_gqa_eligible_on_gqa():
    # The f16 SM100 rows declare pack_gqas={False, True}; a dense GQA graph
    # with a power-of-2 group admits the packed request.
    g = _mk_gqa_graph(8, 2)
    assert engines.engine_name() in _eligible(g, engines.SdpaFwdKnobs(pack_gqa=True))


def test_knob_request_pack_gqa_on_mha_is_identity():
    # MHA: G == 1 compiles the identity (bit-exact unpacked fold), so an
    # explicit pack_gqa=True is honorable and the engine stays eligible.
    g = _mk_eligible_graph()
    assert engines.engine_name() in _eligible(g, engines.SdpaFwdKnobs(pack_gqa=True))


def test_knob_request_pack_gqa_no_pow2_group_rejects_engine():
    # GQA ratio 3 does not divide tile_m (128) so pack_gqa_supported is False.
    g = _mk_gqa_graph(6, 2)
    assert not _eligible(g, engines.SdpaFwdKnobs(pack_gqa=True))


def test_knob_request_pack_gqa_false_always_eligible():
    # Running unpacked is trivially honorable — on MHA graphs too.
    assert engines.engine_name() in _eligible(_mk_eligible_graph(), engines.SdpaFwdKnobs(pack_gqa=False))
    assert engines.engine_name() in _eligible(_mk_gqa_graph(8, 2), engines.SdpaFwdKnobs(pack_gqa=False))


def test_knob_request_pack_gqa_outside_domain_rejects_row():
    # The mxfp8/SM80 rows keep the default {False} domain, so an explicit
    # packed request must never make them eligible; the packable set is
    # pinned exactly by test_pack_gqa_capability_domains.
    g = _mk_gqa_graph(8, 2)
    eligible = _eligible(g, engines.SdpaFwdKnobs(pack_gqa=True))
    assert eligible <= {s.name for s in engines.ENGINE_SPECS if True in s.capabilities.pack_gqas}, eligible


# ---------------------------------------------------------------------------
# SM120 engine row (sdpa_fwd_prefill_sm120) — same probes under a faked
# SM120/SM121 device. Executable coverage lives in test_sdpa_fwd_dsl_sm120.py.
# ---------------------------------------------------------------------------

_SM120 = engines.engine_name(arch="sm120")


def _mk_sm120_graph(d: int = 128, **sdpa_kwargs):
    """A dense fp16 BSHD graph inside the SM120 row's envelope (D <= 256)."""
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=d)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, **sdpa_kwargs)
    _finish_output(o, dims, strides)
    return g


def _mk_dense_stats_graph(stats_stride, *, stats_dim=(B, H, S, 1), stats_dtype=cudnn.data_type.FLOAT):
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=128)
    o, stats = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, generate_stats=True)
    _finish_output(o, dims, strides)
    assert stats is not None
    stats.set_output(True).set_dim(stats_dim).set_stride(stats_stride)
    stats.set_data_type(stats_dtype)
    return g


def test_sm120_probe_accepts_causal_swa_on_both_minors(monkeypatch):
    for cc in ((12, 0), (12, 1)):
        monkeypatch.setattr(ga, "_device_cc", lambda cc=cc: cc)
        elig = _eligible(_mk_sm120_graph(use_causal_mask=True, sliding_window_length=64))
        assert _SM120 in elig
        assert not any("sm100" in name for name in elig)


def test_probe_rejects_requested_amax_s():
    # The FP8 kernels no longer compute Amax_S; a graph that DECLARES the
    # output (set_output(True), non-virtual) must go elsewhere. The port the
    # op returns unconditionally does NOT count (is_virtual stays True).
    import math

    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    dims, strides = (B, H, S, 128), (S * H * 128, 128, H * 128, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="v")
    sc = [g.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT) for _ in range(6)]
    kw = dict(
        q=q,
        k=k,
        v=v,
        descale_q=sc[0],
        descale_k=sc[1],
        descale_v=sc[2],
        descale_s=sc[3],
        scale_s=sc[4],
        scale_o=sc[5],
        attn_scale=1.0 / math.sqrt(128),
        generate_stats=False,
        use_causal_mask=True,
    )

    def build(request_amax_s):
        gg = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        qq = gg.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="q")
        kk = gg.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="k")
        vv = gg.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="v")
        ss = [gg.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT) for _ in range(6)]
        o, _stats, amx_s, amx_o = gg.sdpa_fp8(
            q=qq,
            k=kk,
            v=vv,
            descale_q=ss[0],
            descale_k=ss[1],
            descale_v=ss[2],
            descale_s=ss[3],
            scale_s=ss[4],
            scale_o=ss[5],
            attn_scale=1.0 / math.sqrt(128),
            generate_stats=False,
            use_causal_mask=True,
        )
        _finish_output(o, dims, strides, dtype=cudnn.data_type.HALF)
        amx_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
        if request_amax_s:
            amx_s.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
        return gg

    fp8_name = engines.engine_name(fp8=True)
    assert fp8_name in _eligible(build(request_amax_s=False))
    assert fp8_name not in _eligible(build(request_amax_s=True))


def test_probe_accepts_bottom_right_with_swa():
    # The band shifts wholesale with the diagonal: the SM100 kernels apply the
    # same causal_diag offset to the SWA lower limit as to the causal upper one.
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask_bottom_right=True, sliding_window_length=64)
    _finish_output(o, dims, strides)
    assert engines.engine_name() in _eligible(g)


def test_sm120_probe_accepts_bottom_right_with_swa(monkeypatch):
    # BR + SWA is served on both families now; this pins the SM120 row's claim.
    kwargs = dict(use_causal_mask_bottom_right=True, sliding_window_length=64)
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _SM120 in _eligible(_mk_sm120_graph(**kwargs))


def test_sm120_probe_rejects_on_sm100_family():
    # The autouse fixture fakes (10, 0); the SM120 row must stay ineligible.
    assert _SM120 not in _eligible(_mk_sm120_graph())


def test_sm120_probe_head_dim_envelope(monkeypatch):
    # d_envelope: any multiple of 8 up to the 256 cap is served via TMA
    # zero-padding; only sub-8 alignment (TMA 16-byte global-stride rule)
    # stays ineligible.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _SM120 in _eligible(_mk_sm120_graph(d=192))
    assert _SM120 in _eligible(_mk_sm120_graph(d=136))  # multiple of 8, not of 16
    assert not _eligible(_mk_sm120_graph(d=132))  # multiple of 4, not of 8


def test_sm120_probe_accepts_right_band_widening(monkeypatch):
    # diagonal_band_right_bound > 0 is served by the SM120 row (the causal
    # machinery with a widened diagonal); the SM100 rows keep rejecting it.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    for align in (cudnn.diagonal_alignment.TOP_LEFT, cudnn.diagonal_alignment.BOTTOM_RIGHT):
        g = _mk_graph()
        q, k, v, dims, strides = _mk_qkv(g, d=128)
        o, _ = g.sdpa(
            name="s",
            q=q,
            k=k,
            v=v,
            attn_scale=0.1,
            is_inference=True,
            diagonal_band_right_bound=16,
            diagonal_alignment=align,
        )
        _finish_output(o, dims, strides)
        assert _SM120 in _eligible(g), align


def test_sm120_probe_accepts_ragged_skv_without_padding_or_causal(monkeypatch):
    # No KV-tail rule on the SM120 row (skv_tile=0): the kernel's first
    # (masked) step always covers the rightmost — and therefore any partial —
    # KV tile, so a dense unmasked graph with S_kv % 128 != 0 is served
    # natively. The SM100 f16 row keeps rejecting this shape.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    s_kv, d = 300, 128
    q = g.tensor(dim=(B, H, S, d), stride=(S * H * d, d, H * d, 1), data_type=DTYPE, name="q")
    k = g.tensor(dim=(B, H, s_kv, d), stride=(s_kv * H * d, d, H * d, 1), data_type=DTYPE, name="k")
    v = g.tensor(dim=(B, H, s_kv, d), stride=(s_kv * H * d, d, H * d, 1), data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, (B, H, S, d), (S * H * d, d, H * d, 1))
    assert _SM120 in _eligible(g)


def test_sm120_probe_accepts_mixed_head_dims(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    d_qk, d_v = 192, 128
    g = _mk_graph()
    q = g.tensor(dim=(B, H, S, d_qk), stride=(S * H * d_qk, d_qk, H * d_qk, 1), data_type=DTYPE, name="q")
    k = g.tensor(dim=(B, H, S, d_qk), stride=(S * H * d_qk, d_qk, H * d_qk, 1), data_type=DTYPE, name="k")
    v = g.tensor(dim=(B, H, S, d_v), stride=(S * H * d_v, d_v, H * d_v, 1), data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True)
    _finish_output(o, (B, H, S, d_v), (S * H * d_v, d_v, H * d_v, 1))
    assert _SM120 in _eligible(g)


def test_sm120_probe_accepts_stats_output(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert engines.engine_name(arch="sm120") in _eligible(_mk_dense_stats_graph((H * S, S, 1, 1)))


@pytest.mark.parametrize(
    ("cc", "engine"),
    [
        ((8, 0), engines.engine_name(arch="sm80")),
        ((10, 0), engines.engine_name(arch="sm100")),
        ((12, 0), engines.engine_name(arch="sm120")),
    ],
    ids=["sm80", "sm100", "sm120"],
)
def test_fwd_probe_accepts_strided_stats(monkeypatch, cc, engine):
    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    # B is innermost, then H, then S; the extra two head rows leave a gap
    # between adjacent S positions.
    stats_stride = (1, B, (H + 2) * B, 1)
    assert engine in _eligible(_mk_dense_stats_graph(stats_stride))


@pytest.mark.parametrize(
    ("cc", "engine"),
    [
        ((8, 0), engines.engine_name(arch="sm80")),
        ((10, 0), engines.engine_name(arch="sm100")),
        ((12, 0), engines.engine_name(arch="sm120")),
    ],
    ids=["sm80", "sm100", "sm120"],
)
@pytest.mark.parametrize("stats_stride", [(0, S, 1, 1), (1, 1, 1, 1)], ids=["broadcast", "overlapping"])
def test_fwd_probe_rejects_aliasing_stats(monkeypatch, cc, engine, stats_stride):
    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    assert engine not in _eligible(_mk_dense_stats_graph(stats_stride))


@pytest.mark.parametrize(
    ("stats_dim", "stats_stride", "stats_dtype", "reason"),
    [
        ((B, H, S, 1), (H * S, S, 1, 1), cudnn.data_type.HALF, "stats must be fp32"),
        ((B, H, S), (H * S, S, 1), cudnn.data_type.FLOAT, "stats must be (B, H_q, S_q, 1)"),
    ],
    ids=["dtype", "shape"],
)
def test_fwd_probe_rejects_invalid_stats_metadata(monkeypatch, stats_dim, stats_stride, stats_dtype, reason):
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    facts = ga.analyze(_mk_dense_stats_graph(stats_stride, stats_dim=stats_dim, stats_dtype=stats_dtype))
    capabilities = next(spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == engines.engine_name(arch="sm100"))
    assert reason in engines.mismatch(capabilities, facts)


def test_sm120_probe_accepts_padded_stats(monkeypatch):
    """Padding + generate_stats needs the per-batch seq_len_q LSE trim."""
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=128)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    o, stats = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        generate_stats=True,
        use_padding_mask=True,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
    )
    _finish_output(o, dims, strides)
    assert stats is not None
    stats.set_output(True).set_dim((B, H, S, 1)).set_stride((H * S, S, 1, 1))
    stats.set_data_type(cudnn.data_type.FLOAT)
    assert engines.engine_name(arch="sm120") in _eligible(g)


def test_sm120_probe_accepts_sink(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=128)
    sink = g.tensor(dim=(1, H, 1, 1), stride=(H, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="sink")
    try:
        o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, sink_token=sink)
    except TypeError:
        pytest.skip("this cuDNN wheel's sdpa() binding predates sink_token")
    _finish_output(o, dims, strides)
    assert engines.engine_name(arch="sm120") in _eligible(g)


def test_sm120_probe_accepts_thd(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    _mk_thd_qkvo(g, mask_kwargs=dict(use_causal_mask=True), d=128)
    assert engines.engine_name(arch="sm120") in _eligible(g)


def test_sm120_probe_accepts_thd_bottom_right(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    _mk_thd_qkvo(g, mask_kwargs=dict(use_causal_mask_bottom_right=True), d=128)
    assert engines.engine_name(arch="sm120") in _eligible(g)


def test_sm120_probe_accepts_thd_stats(monkeypatch):
    """The SM120 epilogue writes cuDNN's token-major ragged Stats directly,
    so THD + generate_stats is eligible."""
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    dims = (B, H, S, 128)
    strides = (S * H * 128, 128, H * 128, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    ro = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="ro")
    q.set_ragged_offset(ro)
    k.set_ragged_offset(ro)
    v.set_ragged_offset(ro)
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="sq")
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="skv")
    o, stats = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        generate_stats=True,
        use_causal_mask=True,
        use_padding_mask=True,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
    )
    _finish_output(o, dims, strides)
    o.set_ragged_offset(ro)
    assert stats is not None
    stats.set_output(True).set_dim((B, H, S, 1)).set_stride((S * H, 1, H, 1))
    stats.set_data_type(cudnn.data_type.FLOAT)
    stats_ro = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="stats_ro")
    stats.set_ragged_offset(stats_ro)
    assert engines.engine_name(arch="sm120") in _eligible(g)


def _mk_thd_cu_graph(*, extra_seq_len=False):
    """Ragged (THD) graph carrying the cu_seq_len_q/kv (B+1,) prefix-sum form."""
    g = _mk_graph()
    dims = (B, H, S, D)
    strides = (S * H * D, D, H * D, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    ro = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name="ro")
    q.set_ragged_offset(ro)
    k.set_ragged_offset(ro)
    v.set_ragged_offset(ro)
    cu_q = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="cu_q")
    cu_kv = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="cu_kv")
    kw = dict(cu_seq_len_q=cu_q, cu_seq_len_kv=cu_kv)
    if extra_seq_len:
        skv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="skv")
        kw["seq_len_kv"] = skv
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True, use_padding_mask=True, **kw)
    _finish_output(o, dims, strides)
    o.set_ragged_offset(ro)
    return g


def test_probe_accepts_thd_cu_seq_len():
    """THD with the (B+1,) cu_seq_len prefix-sum form (cuDNN 9.24+) is served:
    the lowering derives per-batch lengths host-side from its inherent tolist
    round-trip."""
    assert engines.engine_name() in _eligible(_mk_thd_cu_graph())


def test_probe_rejects_thd_cu_plus_seq_len():
    """Both forms on one side is ambiguous (the backend has its own
    precedence, which the python engines do not replicate) — declined."""
    assert not _eligible(_mk_thd_cu_graph(extra_seq_len=True))


@pytest.mark.parametrize("side", ["cu_seq_len_q", "cu_seq_len_kv"])
def test_cu_seq_len_is_declined(side):
    """cu_seq_len_* (cuDNN 9.24+) are prefix sums — a different contract from
    seq_len_* and from ragged_offset, and these kernels implement neither.
    Reading such a graph as plain padded silently produced wrong output: 14.9%
    of O on test_sdpa_mixed_seq_len_forms_L0[cu_q_brcm].

    A FACT, not a verdict: ``invalid`` means malformed-for-everyone, so putting
    this there would also bar the engine that eventually implements it."""
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=128)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    cu = g.tensor(dim=(B + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name=side)
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_padding_mask=True,
        seq_len_kv=seq_kv,
        seq_len_q=seq_q,
        **{side: cu},
    )
    _finish_output(o, dims, strides)
    facts = ga.analyze(g)
    assert facts.invalid is None, facts.invalid
    assert facts.has_cu_seq_len
    assert not _eligible(g), "no engine may claim a graph carrying cu_seq_len"


def test_sm120_probe_accepts_padding_mask_with_seq_lens(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=128)
    seq_kv = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
    seq_q = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
    o, _ = g.sdpa(
        name="s",
        q=q,
        k=k,
        v=v,
        attn_scale=0.1,
        is_inference=True,
        use_padding_mask=True,
        seq_len_kv=seq_kv,
        seq_len_q=seq_q,
    )
    _finish_output(o, dims, strides)
    assert _SM120 in _eligible(g)


def _mk_sm120_layout_graph(strides, d=128):
    g = _mk_graph()
    dims = (B, H, S, d)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    return g


def test_sm120_probe_accepts_dense_flex_layouts(monkeypatch):
    # Same dense_flex envelope as the SM100 rows: any B/H/S order with the head
    # dim innermost — the adapter normalizes to compact BSHD (one copy when the
    # caller's layout is not already BSHD-physical).
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    d = 128
    bhsd_contig = (H * S * d, S * d, d, 1)
    assert _SM120 in _eligible(_mk_sm120_layout_graph(bhsd_contig))
    # Head dim NOT innermost (S innermost instead) is outside dense_flex.
    s_innermost = (H * S * d, S * d, 1, S)
    assert not _eligible(_mk_sm120_layout_graph(s_innermost))


def test_sm120_knob_domains(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_sm120_graph(use_causal_mask=True)
    assert _SM120 in _eligible(g, engines.SdpaFwdKnobs(tile_m=64, tile_n=64, cga=1))
    assert not _eligible(g, engines.SdpaFwdKnobs(cga=2))
    # All three scheduler policies are in the SM120 domain (static-grid
    # remap); a value outside the vocabulary still declines.
    assert _SM120 in _eligible(g, engines.SdpaFwdKnobs(sched_policy=1))
    assert not _eligible(g, engines.SdpaFwdKnobs(sched_policy=99))
    # split_kv: the SM120 row serves {1, 2, 4} (inline chunking + the shared
    # combine); a value outside the domain still declines.
    assert _SM120 in _eligible(g, engines.SdpaFwdKnobs(split_kv=1))
    assert _SM120 in _eligible(g, engines.SdpaFwdKnobs(split_kv=4))
    assert not _eligible(g, engines.SdpaFwdKnobs(split_kv=8))


# ---------------------------------------------------------------------------
# SDPA_BWD: facts extraction + sdpa_bwd_sm120 probe gating. Executable
# coverage lives in test_sdpa_bwd_dsl_sm120.py.
# ---------------------------------------------------------------------------

_BWD_ENGINE = "sdpa_bwd_sm120"
_BWD_D = 64


def _bwd_eligible(graph, knobs=None):
    """Names of the FROST SDPA-backward engines whose caps match this graph."""
    return {s.name for s in bwd_engines.ENGINE_SPECS if bwd_engines.analyze_for(s, graph, knobs)[1] is None}


def _bshd_strides(h: int, s: int, d: int) -> tuple[int, int, int, int]:
    return (s * h * d, d, h * d, 1)


def _mk_bwd_graph(
    d: int = _BWD_D,
    h_kv: int = H,
    s_q: int = S,
    s_kv: int = S,
    kv_transposed_view: bool = False,
    stats_stride: tuple | None = None,
    grad_strides: tuple | None = None,
    bias: bool = False,
    dbias: bool = False,
    sink: bool = False,
    dsink: bool = False,
    seq_lens: str | None = None,  # "kv" / "both" (padding mask) or "q_only"
    **bwd_kwargs,
):
    g = _mk_graph()
    q_dims, q_strides = (B, H, s_q, d), _bshd_strides(H, s_q, d)
    kv_dims, kv_strides = (B, h_kv, s_kv, d), _bshd_strides(h_kv, s_kv, d)
    if kv_transposed_view:
        # Mimic the post-build_operation_graph state: the backward node's K/V
        # ports are rewritten to transposed (B, H, D, S) views of the same
        # canonical BSHD buffer.
        kv_dims = (B, h_kv, d, s_kv)
        kv_strides = (s_kv * h_kv * d, d, 1, h_kv * d)
    q = g.tensor(dim=q_dims, stride=q_strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=kv_dims, stride=kv_strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=kv_dims, stride=kv_strides, data_type=DTYPE, name="v")
    o = g.tensor(dim=q_dims, stride=_bshd_strides(H, s_q, d), data_type=DTYPE, name="o")
    do = g.tensor(dim=q_dims, stride=_bshd_strides(H, s_q, d), data_type=DTYPE, name="dO")
    stats = g.tensor(
        dim=(B, H, s_q, 1),
        stride=stats_stride or (H * s_q, s_q, 1, 1),
        data_type=cudnn.data_type.FLOAT,
        name="stats",
    )
    if bias:
        bias_t = g.tensor(dim=(1, H, s_q, s_kv), stride=(H * s_q * s_kv, s_q * s_kv, s_kv, 1), data_type=DTYPE, name="bias")
        bwd_kwargs.update(bias=bias_t)
    if dbias:
        dbias_t = g.tensor(dim=(1, H, s_q, s_kv), stride=(H * s_q * s_kv, s_q * s_kv, s_kv, 1), data_type=DTYPE, name="dBias")
        bwd_kwargs.update(dBias=dbias_t)
    if sink:
        sink_t = g.tensor(dim=(1, H, 1, 1), stride=(H, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="sink")
        bwd_kwargs.update(sink_token=sink_t)
    if dsink:
        dsink_t = g.tensor(dim=(1, H, 1, 1), stride=(H, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="dSink")
        bwd_kwargs.update(dSink_token=dsink_t)
    if seq_lens in ("kv", "both"):
        seq_kv_t = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_kv")
        bwd_kwargs.update(use_padding_mask=True, seq_len_kv=seq_kv_t)
    if seq_lens in ("both", "q_only"):
        seq_q_t = g.tensor(dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_q")
        bwd_kwargs.update(seq_len_q=seq_q_t)
    dq, dk, dv = g.sdpa_backward(name="sb", q=q, k=k, v=v, o=o, dO=do, stats=stats, attn_scale=0.125, **bwd_kwargs)
    _finish_output(dq, q_dims, grad_strides or _bshd_strides(H, s_q, d))
    _finish_output(dk, (B, h_kv, s_kv, d), grad_strides or _bshd_strides(h_kv, s_kv, d))
    _finish_output(dv, (B, h_kv, s_kv, d), grad_strides or _bshd_strides(h_kv, s_kv, d))
    return g


def test_bwd_engines_registered():
    from cudnn.engines import MANIFEST, is_python_engine

    (row,) = [r for r in MANIFEST if r.factory == "FrostSdpaBwdEngines"]
    assert is_python_engine(row.engine_id)
    assert row.id_end - row.engine_id >= len(bwd_engines.ENGINE_SPECS)
    assert bwd_engines.engine_name() == _BWD_ENGINE


def test_bwd_facts_extracted():
    g = _mk_bwd_graph(use_causal_mask=True)
    facts = _facts(g)
    assert facts.is_backward
    assert facts.causal and not facts.bottom_right
    assert facts.right_bound == 0
    assert not facts.deterministic and not facts.has_dbias and not facts.has_dsink
    assert (facts.b, facts.h_q, facts.h_kv, facts.s_q, facts.s_kv, facts.d_qk, facts.d_v) == (B, H, H, S, S, _BWD_D, _BWD_D)
    assert facts.dtype == cudnn.data_type.HALF and facts.uniform_dtype  # facts speak cudnn.data_type, not torch
    assert facts.bshd_layout
    for ref in (facts.do_t, facts.dq_t, facts.dk_t, facts.dv_t, facts.stats_t):
        assert ref is not None
    assert facts.scale == 0.125


def test_bwd_facts_kv_transposed_view_canonicalized():
    # After build_operation_graph the bwd node's K/V ports describe transposed
    # (B, H, D, S) views; the analyzer canonicalizes dims AND strides back so
    # geometry and the BSHD layout gate hold before and after the native build.
    facts = _facts(_mk_bwd_graph(kv_transposed_view=True))
    assert (facts.s_kv, facts.d_qk) == (S, _BWD_D)
    assert facts.bshd_layout
    # port_layouts (what bwd lowering consumes) has the rewrite undone too.
    ports = {name: (dim, stride) for name, dim, stride in facts.port_layouts}
    assert ports["k"] == ((B, H, S, _BWD_D), _bshd_strides(H, S, _BWD_D))
    assert ports["v"] == ((B, H, S, _BWD_D), _bshd_strides(H, S, _BWD_D))


def test_bwd_probe_accepts(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph())
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(use_causal_mask=True))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(s_q=S // 2, use_causal_mask_bottom_right=True))
    for d in (32, 128):
        assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(d=d))


def test_bwd_probe_rejects_forward_graph(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g, d=_BWD_D)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
    _finish_output(o, dims, strides)
    assert not _bwd_eligible(g)
    # ... and symmetrically, the forward engines decline a backward graph.
    assert not _eligible(_mk_bwd_graph())


def test_bwd_probe_gqa(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(h_kv=H // 2))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(h_kv=1))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(h_kv=H // 2, use_deterministic_algorithm=True))
    # H_q must be a multiple of H_kv
    assert not _bwd_eligible(_mk_bwd_graph(h_kv=3))


def test_bwd_probe_rejects_unsupported_head_dim(monkeypatch):
    # Envelope: any multiple of 8 up to 256 (adapter pads); reject the rest.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert not _bwd_eligible(_mk_bwd_graph(d=100))
    assert not _bwd_eligible(_mk_bwd_graph(d=264))


def test_bwd_probe_causal_notches(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(s_q=S // 2, use_causal_mask=True))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(s_q=2 * S, use_causal_mask=True))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(s_q=2 * S, use_causal_mask_bottom_right=True))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(use_causal_mask=True, sliding_window_length=64))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(sliding_window_length=64))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(s_q=S // 2, use_causal_mask_bottom_right=True, sliding_window_length=64))


def test_bwd_probe_accepts_right_band_widening(monkeypatch):
    # diagonal_band_right_bound > 0 lowers as causal with a right offset.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    import cudnn

    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(diagonal_band_right_bound=16))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(s_q=S // 2, diagonal_band_right_bound=16, diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT))


def test_bwd_probe_accepts_deterministic(monkeypatch):
    # use_deterministic_algorithm is served by the ordered-relay dQ path.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(use_deterministic_algorithm=True))


def test_bwd_probe_accepts_padding_mask(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(seq_lens="kv"))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(seq_lens="both"))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(seq_lens="both", use_causal_mask_bottom_right=True))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(seq_lens="both", use_causal_mask=True, sliding_window_length=64))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(seq_lens="both", use_deterministic_algorithm=True))


def test_bwd_probe_rejects_seq_len_q_without_padding_mask(monkeypatch):
    # Bare seq_len_q is per-batch Q trimming, which the kernel has no path for.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert not _bwd_eligible(_mk_bwd_graph(seq_lens="q_only"))


def test_bwd_probe_accepts_sink(monkeypatch):
    # dSink without the sink input is rejected.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(sink=True))
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(sink=True, dsink=True, use_causal_mask=True))
    assert not _bwd_eligible(_mk_bwd_graph(dsink=True))


def test_bwd_probe_rejects_bias(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert not _bwd_eligible(_mk_bwd_graph(bias=True))


def test_bwd_probe_rejects_dbias(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert not _bwd_eligible(_mk_bwd_graph(dbias=True))


def test_bwd_probe_accepts_dense_flex_layouts(monkeypatch):
    # Same dense_flex envelope as the forward rows: any B/H/S order with the
    # head dim innermost; the declared strides are served natively.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    bhsd_contig = (H * S * _BWD_D, S * _BWD_D, _BWD_D, 1)
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(grad_strides=bhsd_contig))
    # Head dim NOT innermost (S innermost instead) is outside dense_flex.
    s_innermost = (H * S * _BWD_D, S * _BWD_D, 1, S)
    assert not _bwd_eligible(_mk_bwd_graph(grad_strides=s_innermost))


def test_bwd_probe_accepts_strided_stats(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    # Padded stats strides bind natively (baked into the compiled kernel).
    assert _BWD_ENGINE in _bwd_eligible(_mk_bwd_graph(stats_stride=(2 * H * S, 2 * S, 2, 1)))


def test_bwd_knob_domains(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    g = _mk_bwd_graph()
    # In-domain requests are eligible (final per-head-dim feasibility is the
    # kernel constructor's, at build).
    assert _BWD_ENGINE in _bwd_eligible(g, bwd_engines.SdpaBwdKnobs(tile_m=64, tile_n=128))
    assert _BWD_ENGINE in _bwd_eligible(g, bwd_engines.SdpaBwdKnobs())  # all-None = no preference
    # Out-of-domain values are rejected.
    assert not _bwd_eligible(g, bwd_engines.SdpaBwdKnobs(tile_m=48))
    assert not _bwd_eligible(g, bwd_engines.SdpaBwdKnobs(tile_n=32))
    # Another operation's vocabulary is rejected wholesale.
    assert not _bwd_eligible(g, engines.SdpaFwdKnobs(tile_m=64))


def test_bwd_mismatch_reason_strings(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    caps = bwd_engines.ENGINE_SPECS[0].capabilities
    reason = bwd_engines.mismatch(caps, _facts(_mk_bwd_graph(d=100)))
    assert reason is not None and "100" in reason
    reason = bwd_engines.mismatch(caps, _facts(_mk_bwd_graph(use_causal_mask=True, use_alibi_mask=True)))
    assert reason is not None and "ALiBi" in reason
    reason = bwd_engines.mismatch(caps, _facts(_mk_bwd_graph()), engines.SdpaFwdKnobs(tile_m=64))
    assert reason is not None and "knob" in reason
    reason = bwd_engines.mismatch(caps, _facts(_mk_bwd_graph()), bwd_engines.SdpaBwdKnobs(tile_m=48))
    assert reason is not None and "tile_m=48" in reason
    assert bwd_engines.mismatch(caps, _facts(_mk_bwd_graph()), bwd_engines.SdpaBwdKnobs(tile_m=64, tile_n=128)) is None


# ---------------------------------------------------------------------------
# The SM80 backward row (sdpa_bwd_sm80): its envelope covers exactly the
# features the SM120 row's kernel rejects.
# ---------------------------------------------------------------------------

_BWD_SM80 = "sdpa_bwd_sm80"


def test_bwd_sm80_probe_accepts_the_sm120_rejections(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (8, 0))
    assert _BWD_SM80 in _bwd_eligible(_mk_bwd_graph())
    assert _BWD_SM80 in _bwd_eligible(_mk_bwd_graph(use_causal_mask=True))
    assert _BWD_SM80 in _bwd_eligible(_mk_bwd_graph(h_kv=H // 2))  # GQA
    assert _BWD_SM80 in _bwd_eligible(_mk_bwd_graph(use_deterministic_algorithm=True))
    # Flavor envelope: any head dim <= 256 (no multiple-of-16 rule) and
    # top-left causal with S_q != S_kv.
    assert _BWD_SM80 in _bwd_eligible(_mk_bwd_graph(d=96))
    assert _BWD_SM80 in _bwd_eligible(_mk_bwd_graph(use_causal_mask=True, s_q=S // 2))


def test_bwd_sm80_probe_rejections(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (8, 0))
    assert not _bwd_eligible(_mk_bwd_graph(s_q=1))  # decode-shaped: prefill kernels only
    assert not _bwd_eligible(_mk_bwd_graph(d=257))  # beyond the qwen (256, 256) envelope


def test_bwd_sm80_probe_rejects_off_arch(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _BWD_SM80 not in _bwd_eligible(_mk_bwd_graph())


def test_bwd_dsink_fact():
    d = 128
    g = _mk_graph()
    dims = (B, H, S, d)
    strides = (S * H * d, d, H * d, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
    o = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="o")
    do = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="dO")
    stats = g.tensor(dim=(B, H, S, 1), stride=(H * S, S, 1, 1), data_type=cudnn.data_type.FLOAT, name="stats")
    sink = g.tensor(dim=(1, H, 1, 1), stride=(H, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="sink")
    dsink = g.tensor(dim=(1, H, 1, 1), stride=(H, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="dsink")
    dq, dk, dv = g.sdpa_backward(q=q, k=k, v=v, o=o, dO=do, stats=stats, attn_scale=0.5, use_causal_mask=True, sink_token=sink, dSink_token=dsink)
    for t in (dq, dk, dv):
        _finish_output(t, dims, strides)
    facts = _facts(g)
    assert facts.has_sink and facts.has_dsink
    assert facts.sink_t is not None and facts.dsink_t is not None
