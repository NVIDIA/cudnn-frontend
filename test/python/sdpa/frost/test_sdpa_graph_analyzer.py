# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free unit tests for the SDPA facts analyzer and per-engine capability probes."""

from __future__ import annotations

import math

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


@pytest.mark.parametrize("stats_use_log2", [False, True])
def test_probe_stats_log2_keeps_eligibility(stats_use_log2):
    """A base-2 Stats request (stats_use_log2) is a plan-time epilogue specialization
    of every FROST forward kernel, so the fact is recorded and the graph stays eligible
    in both bases."""
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, stats = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, generate_stats=True, stats_use_log2=stats_use_log2)
    _finish_output(o, dims, strides)
    _finish_output(stats, (B, H, S, 1), (H * S, S, 1, 1), cudnn.data_type.FLOAT)
    assert _facts(g).has_stats_log2 is stats_use_log2
    assert _eligible(g)


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


def _mk_softmax_precision_graph(precision):
    g = _mk_graph()
    q, k, v, dims, strides = _mk_qkv(g)
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True, softmax_precision=precision)
    _finish_output(o, dims, strides)
    return g


def test_softmax_precision_is_an_op_attribute_not_a_knob():
    """sdpa(softmax_precision=) is numerics-changing, so it is a graph FACT the
    capability rows gate on — not a tuning axis an autotuner could pick."""
    import cudnn as _c

    assert "softmax_precision" not in engines.SdpaFwdKnobs.__dataclass_fields__
    # FLOAT is the pipeline every row runs: same eligibility as no request.
    assert _eligible(_mk_softmax_precision_graph(_c.data_type.FLOAT)) == _eligible(_mk_eligible_graph())
    # HALF exists only in the per-tensor-FP8 SM107 arm; a bf16 graph declines everywhere.
    g_half = _mk_softmax_precision_graph(_c.data_type.HALF)
    assert ga.analyze(g_half).softmax_precision == _c.data_type.HALF
    assert not _eligible(g_half)
    # Anything else is a malformed request, reported on the facts.
    g_bad = _mk_softmax_precision_graph(_c.data_type.DOUBLE)
    assert "softmax_precision must be" in (ga.analyze(g_bad).invalid or "")
    assert not _eligible(g_bad)
    # The attribute never reaches the cuDNN backend: a SET value makes the node backend-unlowerable.
    assert g_half._unlowerable_node() is not None
    assert _mk_softmax_precision_graph(None)._unlowerable_node() is None
    # serialize() is the backend format, which has no field for the attribute:
    # refused rather than emitted as (and later executed as) the f32 pipeline.
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="serialize"):
        g_half.serialize()


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
    # GQA ratios 3 and 5 share no factor with tile_m (128): nothing can be
    # packed, so an explicit pack_gqa=True is not honorable (never degraded).
    g = _mk_gqa_graph(6, 2)
    assert not _eligible(g, engines.SdpaFwdKnobs(pack_gqa=True))
    assert not _eligible(_mk_gqa_graph(10, 2), engines.SdpaFwdKnobs(pack_gqa=True))


def test_knob_request_pack_gqa_partial_group_on_wired_flavors_only():
    # Partial PackGQA (Capabilities.pack_gqa_partial_d_shapes): a ratio that
    # shares a factor with tile_m but does not divide it packs that factor on
    # the d128 and d256 f16 flavors (G=12 -> 4 heads per row group, G=6 -> 2) ...
    for h_q, h_kv in ((24, 2), (12, 2)):
        assert engines.engine_name() in _eligible(_mk_gqa_graph(h_q, h_kv), engines.SdpaFwdKnobs(pack_gqa=True))
        assert engines.engine_name() in _eligible(_mk_gqa_graph(h_q, h_kv, d=256), engines.SdpaFwdKnobs(pack_gqa=True))
    # ... while the d512 flavor keeps the full-ratio contract (its kernel packs
    # HEADS_PER_TILE = G), so the same request is declined there.
    assert not _eligible(_mk_gqa_graph(24, 2, d=512), engines.SdpaFwdKnobs(pack_gqa=True))
    assert engines.engine_name() in _eligible(_mk_gqa_graph(32, 2, d=512), engines.SdpaFwdKnobs(pack_gqa=True))
    # A group larger than the tile (256/1 MQA) packs the whole tile (p = 128,
    # two packed heads per KV head) on the partial flavors; declined on d512.
    assert engines.engine_name() in _eligible(_mk_gqa_graph(256, 1), engines.SdpaFwdKnobs(pack_gqa=True))
    assert not _eligible(_mk_gqa_graph(256, 1, d=512), engines.SdpaFwdKnobs(pack_gqa=True))


def test_pack_gqa_partial_d_shapes_in_lockstep_with_the_adapter():
    # The standalone adapter (api_dsl.SdpaFwdDslSm100.check_support) mirrors
    # Capabilities.pack_gqa_partial_d_shapes as a module tuple because the
    # adapter has no engine row in hand when it validates a knob request.  Pin
    # the two together so the gate cannot drift: the f16 SM100 row declares
    # exactly the adapter's flavors, and no other row (fp8 / mxfp8, the cc 10.7
    # line, SM120) declares partial packing -- the adapter excludes those too.
    from cudnn.sdpa.fwd.api_dsl import _SM100_PARTIAL_PACK_GQA_FLAVORS

    by_name = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    assert by_name[engines.engine_name()].pack_gqa_partial_d_shapes == frozenset(_SM100_PARTIAL_PACK_GQA_FLAVORS)
    assert frozenset(_SM100_PARTIAL_PACK_GQA_FLAVORS) == frozenset({(128, 128), (256, 256)})
    others = {name: caps.pack_gqa_partial_d_shapes for name, caps in by_name.items() if name != engines.engine_name()}
    assert all(v is None for v in others.values()), others


def test_capabilities_positional_prefix_is_append_only():
    # Capabilities evolves APPEND-ONLY (the contract stated above
    # pack_gqa_d_shapes): a positional construction written against an older
    # field order must keep binding the same fields.  Prove it the way it
    # breaks -- construct positionally in the pre-partial-PackGQA order with
    # thd_padded_stats=True and check nothing rebinds (inserted mid-class, the
    # True landed on pack_gqa_partial_d_shapes, thd_padded_stats fell back to
    # False and the flavor membership test raised TypeError on a bool) -- then
    # pin the legacy tail and the new field's place after it.
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(engines.Capabilities)}
    names = list(fields)
    required = {"sm_lo": 100, "sm_hi": 100, "phase": "prefill", "d_shapes": frozenset({(128, 128)})}

    def legacy_value(name):
        f = fields[name]
        if name == "thd_padded_stats":
            return True
        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:
            return f.default_factory()
        return required[name]

    legacy_order = [n for n in names if n not in ("pack_gqa_partial_d_shapes", "paged_d_shapes")]
    caps = engines.Capabilities(*[legacy_value(n) for n in legacy_order])
    assert caps.thd_padded_stats is True
    assert caps.pack_gqa_partial_d_shapes is None
    assert caps.paged_d_shapes is None
    assert caps.epilogue_gate is False
    assert engines.pack_gqa_partial(caps, ga.SdpaGraphFacts(d_qk=128, d_v=128)) is False

    legacy_tail = ["pack_gqa_d_shapes", "thd_padded_stats", "epilogue_gate", "epilogue_gate_d_shapes", "epilogue_gate_dtypes"]
    start = names.index("pack_gqa_d_shapes")
    assert names[start : start + len(legacy_tail)] == legacy_tail, names[start:]
    # ... and every later field is appended after it, in the order it landed.
    assert names[start + len(legacy_tail) :] == ["pack_gqa_partial_d_shapes", "paged_d_shapes"], names[start:]


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


def _mk_sm120_graph(d: int = 128, *, d_v: int | None = None, **sdpa_kwargs):
    """Build a dense fp16 BSHD graph with independent Q/K and V/O dimensions."""
    g = _mk_graph()
    q_dims, q_strides = (B, H, S, d), (S * H * d, d, H * d, 1)
    d_v = d if d_v is None else d_v
    dims, strides = (B, H, S, d_v), (S * H * d_v, d_v, H * d_v, 1)
    q = g.tensor(dim=q_dims, stride=q_strides, data_type=DTYPE, name="q")
    k = g.tensor(dim=q_dims, stride=q_strides, data_type=DTYPE, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="v")
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
    # stays ineligible. Both dimensions in (256, 512] use the d512 envelope.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert _SM120 in _eligible(_mk_sm120_graph(d=192))
    assert _SM120 in _eligible(_mk_sm120_graph(d=136))  # multiple of 8, not of 16
    assert not _eligible(_mk_sm120_graph(d=132))  # multiple of 4, not of 8
    assert _SM120 in _eligible(_mk_sm120_graph(d=512))
    assert _SM120 in _eligible(_mk_sm120_graph(d=504))
    assert _SM120 in _eligible(_mk_sm120_graph(d=264))
    assert _SM120 in _eligible(_mk_sm120_graph(d=496))


def test_sm120_probe_full_d512_envelope_pairs(monkeypatch):
    """Every 8-aligned Q/K and V/O pair in (256, 512] is independently eligible."""
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    for d_qk in range(264, 513, 8):
        for d_v in range(264, 513, 8):
            assert _SM120 in _eligible(_mk_sm120_graph(d=d_qk, d_v=d_v)), (d_qk, d_v)
    for d_qk, d_v in ((256, 512), (512, 256), (248, 264), (264, 248), (260, 264), (264, 260), (520, 512), (512, 520)):
        assert _SM120 not in _eligible(_mk_sm120_graph(d=d_qk, d_v=d_v)), (d_qk, d_v)


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
    # split_kv: the SM120 row WIRES the split path (inline chunking + the
    # shared combine), which is a boolean gate — the kernel has no upper bound
    # on the split count, so 8 is admissible too. WHICH splits get proposed is
    # split_kv_candidates' device-derived ladder, not a per-row domain. A
    # non-count still declines.
    assert _SM120 in _eligible(g, engines.SdpaFwdKnobs(split_kv=1))
    assert _SM120 in _eligible(g, engines.SdpaFwdKnobs(split_kv=4))
    assert _SM120 in _eligible(g, engines.SdpaFwdKnobs(split_kv=8))
    assert not _eligible(g, engines.SdpaFwdKnobs(split_kv=0))


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


def test_bwd_probe_rejects_deterministic_broadcast_dbias(monkeypatch):
    # A batch-broadcast bias reduces dBias over B through unordered atomics.
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    assert not _bwd_eligible(_mk_bwd_graph(bias=True, dbias=True, use_deterministic_algorithm=True))


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


# --- paged KV caches (issue #920) -------------------------------------------


def _mk_paged_graph(*, d=128, d_v=None, page_size=16, max_pages=8, hnd=True, padding=True, max_seq_len=None, one_table=False, sink=False):
    """cuDNN's paged-cache contract: K/V page pools [num_pages, H_kv, page_size, D]
    (HND compact, or NHD storage declared via strides) + (B, 1, max_pages, 1)
    int32 block tables + per-batch lengths. ``d_v`` (default ``d``) is the V
    pool's row width: the two pools may differ (MLA-style d_qk != d_v). ``sink``
    adds the (1, H, 1, 1) fp32 sink_token (a decode graph: s_q = 1)."""
    g = _mk_graph()
    b, h, kh = B, H, 2
    d_v = d if d_v is None else d_v
    q = g.tensor(dim=(b, h, 1, d), stride=(h * d, d, d, 1), data_type=DTYPE, name="q")
    num_pages = b * max_pages

    def _pool_strides(dd):
        return (kh * page_size * dd, page_size * dd, dd, 1) if hnd else (page_size * kh * dd, dd, kh * dd, 1)

    k = g.tensor(dim=(num_pages, kh, page_size, d), stride=_pool_strides(d), data_type=DTYPE, name="k")
    v = g.tensor(dim=(num_pages, kh, page_size, d_v), stride=_pool_strides(d_v), data_type=DTYPE, name="v")
    tk = g.tensor(dim=(b, 1, max_pages, 1), stride=(max_pages, max_pages, 1, 1), data_type=cudnn.data_type.INT32, name="tk")
    tv = tk if one_table else g.tensor(dim=(b, 1, max_pages, 1), stride=(max_pages, max_pages, 1, 1), data_type=cudnn.data_type.INT32, name="tv")
    slq = g.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="slq")
    slk = g.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="slk")
    kw = dict(paged_attention_k_table=tk, paged_attention_v_table=tv)
    if max_seq_len is not None:
        kw["paged_attention_max_seq_len_kv"] = max_seq_len
    if sink:
        kw["sink_token"] = g.tensor(dim=(1, h, 1, 1), stride=(h, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="sink")
    o, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_padding_mask=padding, seq_len_q=slq, seq_len_kv=slk, **kw)
    _finish_output(o, (b, h, 1, d_v), (h * d_v, d_v, d_v, 1))
    return g


def test_paged_facts_hnd_and_nhd():
    for hnd in (True, False):
        facts = _facts(_mk_paged_graph(hnd=hnd))
        assert facts.has_paged_kv and facts.page_size == 16
        assert facts.s_kv == 8 * 16 and facts.h_kv == 2 and facts.padded
        assert tuple(facts.paged_k_table_t.get_dim()) == (B, 1, 8, 1)
        assert facts.paged_k_table_t is not None and facts.paged_v_table_t is not None
        assert engines.engine_name() in _eligible(_mk_paged_graph(hnd=hnd))


def test_paged_facts_declared_max_seq_len_and_single_table():
    facts = _facts(_mk_paged_graph(max_seq_len=100, one_table=True))
    assert facts.s_kv == 100
    assert facts.paged_k_table_t is facts.paged_v_table_t
    assert engines.engine_name() in _eligible(_mk_paged_graph(max_seq_len=100, one_table=True))


def test_paged_probe_declines():
    assert not _eligible(_mk_paged_graph(page_size=48)), "page_size must divide 128 or be a multiple of it"
    assert engines.engine_name() in _eligible(_mk_paged_graph(d=192)), "d=192 rides the d256 flavor envelope"
    assert not _eligible(_mk_paged_graph(d=512)), "paged KV rides the d128 / d192x128 / d256 flavors only"
    assert not _eligible(_mk_paged_graph(d=512, d_v=128)), "(512, 128) selects the d512 flavor, which carries no PAGED_KV specialization"
    assert not _eligible(_mk_paged_graph(padding=False)), "paged KV needs the padding mask (per-batch KV lengths)"
    # Lifted decline: the sink is an epilogue fold, orthogonal to the paged loader.
    facts = _facts(_mk_paged_graph(sink=True))
    assert facts.has_paged_kv and facts.has_sink and facts.s_q == 1
    assert engines.engine_name() in _eligible(_mk_paged_graph(sink=True)), "paged KV with an attention sink at decode is served"
    facts = ga.analyze(_mk_paged_graph(max_seq_len=8 * 16 + 1))
    assert facts.invalid is not None, "a declared max S_kv beyond the block table's reach is invalid"


def test_paged_mixed_head_dims_are_served():
    """The head-dim gate tests the flavor the lowering SELECTS
    (``Capabilities.paged_d_shapes``), not the raw dims: (192, 128) is the
    native d192x128 flavor, (256, 128) / (64, 192) / (136, 72) ride the d256 and
    d192x128 envelopes. The previous "exactly one dim > 128" approximation
    declined all of them (INVERTED from a decline)."""
    for d_qk, d_v in ((192, 128), (256, 128), (64, 192), (136, 72)):
        assert engines.engine_name() in _eligible(_mk_paged_graph(d=d_qk, d_v=d_v)), f"paged ({d_qk}, {d_v}) must be served"


def test_paged_split_kv_is_proposed_on_a_decode_launch(monkeypatch):
    """The padded exclusion on split-KV is lifted for paged graphs: a B=2, H_kv=2
    decode launch over 128k tokens must be offered a split plan."""
    monkeypatch.setattr(ga, "_device_sm_count", lambda: 148)
    from cudnn.sdpa.fwd.heuristics import _knob_sets

    g = _mk_paged_graph(page_size=128, max_pages=1024)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == engines.engine_name())
    facts = _facts(g)
    knob_sets = _knob_sets(spec, facts)
    assert any(k.split_kv and k.split_kv > 1 for k in knob_sets), knob_sets


@pytest.mark.parametrize("max_seq_len", [4000, 4001, 16641])
def test_paged_split_kv_is_proposed_with_a_ragged_declared_max(monkeypatch, max_seq_len):
    """A declared ``paged_attention_max_seq_len_kv`` that is NOT a multiple of
    the 128-row KV tile (FlashInfer passes its true max verbatim, e.g. 4000)
    must not cost the paged decode launch its split: the per-batch lengths
    bound the walk on device, so a paged graph never rides the synthesized
    KV-tail padding that excludes the split on a mask-free dense graph. The
    heuristic must still propose a split on the same B=2, H_kv=2 launch (tables
    padded past the declared max, as frameworks do), and a pinned split must
    still pass the knob probe."""
    monkeypatch.setattr(ga, "_device_sm_count", lambda: 148)
    from cudnn.sdpa.fwd.heuristics import _knob_sets

    g = _mk_paged_graph(page_size=16, max_pages=-(-max_seq_len // 16) + 6, max_seq_len=max_seq_len)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == engines.engine_name())
    facts = _facts(g)
    assert facts.s_kv == max_seq_len and facts.s_kv % 128 != 0 and facts.padded and facts.has_paged_kv
    knob_sets = _knob_sets(spec, facts)
    assert any(k.split_kv and k.split_kv > 1 for k in knob_sets), knob_sets
    assert engines.engine_name() in _eligible(g, engines.SdpaFwdKnobs(split_kv=2))


def test_packed_layout_ignores_the_batch_stride_a_ragged_tensor_never_reads():
    """FlashInfer declares its packed THD Q/O with the batch stride equal to the
    token stride (h * d): not BSHD-physical over all four axes, but every
    sequence base comes from the ragged offsets, so only the (H, S, D) order
    the THD lowering addresses has to hold."""
    from cudnn.sdpa.graph_analyzer import bshd_layout_ok, packed_layout_ok

    b, h, s, d = 2, 8, 87, 128
    flashinfer_q = ((b, h, s, d), (h * d, d, h * d, 1))
    assert not bshd_layout_ok(*flashinfer_q) and packed_layout_ok(*flashinfer_q)
    dense_bshd = ((b, h, s, d), (s * h * d, d, h * d, 1))
    assert bshd_layout_ok(*dense_bshd) and packed_layout_ok(*dense_bshd)
    bhsd = ((b, h, s, d), (h * s * d, s * d, d, 1))  # heads outside tokens: not the packed order
    assert not packed_layout_ok(*bhsd)
    assert packed_layout_ok((1, h, s, d), (h * d, d, h * d, 1))  # b == 1 wildcards as before


# ---------------------------------------------------------------------------
# Epilogue-gate tail: sdpa(virtual O_v) -> sigmoid(G) -> mul(O_v, s)  (PR-A)
#
# The analyzer recognises EXACTLY this three-node shape (cudnn._sdpa_tail),
# records it as FACTS (has_epilogue_gate, the G ref, its dtype, shape-ok, the
# virtual O_v's dtype and declaration state) and rebinds ``o_t`` to the mul
# output.  The rows judge: only the Rubin d256 rows claim it, at exact dims.
# ---------------------------------------------------------------------------

_D256 = 256


def _mk_gated_graph(
    *,
    d=_D256,
    dtype=DTYPE,
    intermediate=cudnn.data_type.FLOAT,
    mul_order="o_first",
    declare_o_v=True,
    o_v_dtype=None,
    gate_dims=None,
    sdpa_kwargs=None,
    gate_strides=None,
):
    """(graph, tensors) for the gate tail; ``o`` (the mul output) is the REAL output.
    ``gate_strides`` overrides G's BSHD layout (None = compact BSHD for ``gate_dims``)."""
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=intermediate, compute_data_type=cudnn.data_type.FLOAT)
    dims = (B, H, S, d)
    strides = (S * H * d, d, H * d, 1)
    q = g.tensor(dim=dims, stride=strides, data_type=dtype, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=dtype, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=dtype, name="v")
    gdims = tuple(gate_dims) if gate_dims is not None else dims
    gstrides = tuple(gate_strides) if gate_strides is not None else _bshd_strides(*gdims[1:])  # default: G rides O's BSHD layout
    gate = g.tensor(dim=gdims, stride=gstrides, data_type=dtype, name="gate")
    o_v, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True, use_causal_mask=True, **(sdpa_kwargs or {}))
    if declare_o_v:
        o_v.set_dim(dims).set_stride(strides)
    if o_v_dtype is not None:
        o_v.set_data_type(o_v_dtype)
    s = g.sigmoid(input=gate, name="sig")
    o = g.mul(a=o_v, b=s, name="gated") if mul_order == "o_first" else g.mul(a=s, b=o_v, name="gated")
    _finish_output(o, dims, strides, dtype=dtype)
    return g, dict(q=q, k=k, v=v, gate=gate, o_v=o_v, s=s, o=o)


def _rubin(monkeypatch):
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 7))


@pytest.mark.parametrize("mul_order", ["o_first", "s_first"])
@pytest.mark.parametrize("intermediate", [cudnn.data_type.FLOAT, None], ids=["intermediate-FLOAT", "intermediate-inherits-io"])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16], ids=["fp16", "bf16"])
def test_gate_tail_is_recognised(monkeypatch, mul_order, intermediate, dtype):
    """The three-node tail is ONE sdpa graph to the analyzer: ``has_epilogue_gate``,
    ``epilogue_gate_t`` IS the graph's G, ``o_t`` IS the mul output (the kernel
    writes the real O; O_v / s are never bound), in either mul operand order and
    with or without an explicit FLOAT intermediate dtype (plan S7 Q7: O_v may be
    FLOAT or Q's dtype -- the kernel gates the fp32 pre-cast accumulator either
    way).  Only the Rubin f16 row is eligible at d=256 on cc10.7; at d=128 no
    row is, and the reason names the gate."""
    _rubin(monkeypatch)
    g, ts = _mk_gated_graph(mul_order=mul_order, intermediate=intermediate, dtype=dtype)
    facts = _facts(g)
    assert facts.has_epilogue_gate is True
    assert facts.epilogue_gate_t is ts["gate"]
    assert facts.epilogue_gate_dtype == dtype
    assert facts.epilogue_gate_shape_ok is True
    assert facts.epilogue_gate_layout_ok is True, "a compact BSHD G is TMA-loadable zero-copy"
    assert facts.o_t is ts["o"], "the REAL O is the mul output"
    assert facts.o_t is not ts["o_v"]
    assert facts.sdpa_o_virtual_declared is True
    assert facts.sdpa_o_virtual_dtype in (None, cudnn.data_type.FLOAT, dtype)
    assert facts.q_t is ts["q"] and facts.d_qk == facts.d_v == _D256 and facts.causal is True
    assert ga._single_sdpa_node(g) is not None, "the sdpa node is still reachable through the single-node accessor"
    assert _eligible(g) == {"sdpa_fwd_prefill_sm107"}
    # The 4-node SdpaBinding must demand G and never O_v / s.
    assert "gate" in ga.SdpaBinding.__dataclass_fields__

    g128, _ = _mk_gated_graph(d=128, mul_order=mul_order, intermediate=intermediate, dtype=dtype)
    assert _facts(g128).has_epilogue_gate is True
    assert not _eligible(g128)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    _, why = engines.analyze_for(spec, g128)
    assert why is not None and "epilogue gate" in why, why
    # The same tail on the SM100 line (the autouse cc) is declined by every row.
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    g100, _ = _mk_gated_graph(mul_order=mul_order, intermediate=intermediate, dtype=dtype)
    assert not _eligible(g100)


def test_gate_tail_o_v_dtype_decision(monkeypatch):
    """O_v carries whatever dtype the IR gives a virtual (the intermediate dtype
    at validate(); None before): FLOAT and Q's dtype are ACCEPTED, anything else
    is declined by message -- the kernel gates the fp32 pre-cast accumulator and
    never materialises O_v, so only those two describe the math it does."""
    _rubin(monkeypatch)
    for ok in (None, cudnn.data_type.FLOAT, cudnn.data_type.HALF):
        g, _ = _mk_gated_graph(o_v_dtype=ok)
        assert _facts(g).sdpa_o_virtual_dtype == ok
        assert _eligible(g) == {"sdpa_fwd_prefill_sm107"}, ok
    g, _ = _mk_gated_graph(o_v_dtype=cudnn.data_type.BFLOAT16)
    assert not _eligible(g)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    assert "virtual O" in engines.analyze_for(spec, g)[1]


def _malformed_cases():
    """(id, builder) -> a graph the tail matcher must NOT fuse, or fuse-then-decline."""

    def non_virtual_o_v():
        g, ts = _mk_gated_graph()
        ts["o_v"].set_output(True)  # a REAL O_v: two outputs, nothing to fuse
        return g

    def sigmoid_of_o_v_times_g():
        g = _mk_graph()
        q, k, v, dims, strides = _mk_qkv(g, d=_D256)
        gate = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="gate")
        o_v, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
        o_v.set_dim(dims).set_stride(strides)
        o = g.mul(a=g.sigmoid(input=o_v), b=gate)
        _finish_output(o, dims, strides)
        return g

    def g_produced_by_a_node():
        # G = the sdpa node's own Stats: three nodes, but G is not a graph INPUT.
        g = _mk_graph()
        q, k, v, dims, strides = _mk_qkv(g, d=_D256)
        o_v, stats = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, generate_stats=True)
        o_v.set_dim(dims).set_stride(strides)
        o = g.mul(a=o_v, b=g.sigmoid(input=stats))
        _finish_output(o, dims, strides)
        return g

    def sigmoid_output_is_an_output():
        g, ts = _mk_gated_graph()
        ts["s"].set_output(True).set_dim((B, H, S, _D256)).set_stride((S * H * _D256, _D256, H * _D256, 1))
        return g

    def second_sdpa():
        g, ts = _mk_gated_graph()
        o2, _ = g.sdpa(name="s2", q=ts["q"], k=ts["k"], v=ts["v"], attn_scale=0.1, is_inference=True)
        _finish_output(o2, (B, H, S, _D256), (S * H * _D256, _D256, H * _D256, 1))
        return g

    def relu_tail():
        g, ts = _mk_gated_graph()
        r = g.relu(input=ts["o"], name="r")
        r.set_output(True).set_dim((B, H, S, _D256)).set_stride((S * H * _D256, _D256, H * _D256, 1))
        return g

    def mul_by_g_without_sigmoid():
        g = _mk_graph()
        q, k, v, dims, strides = _mk_qkv(g, d=_D256)
        gate = g.tensor(dim=dims, stride=strides, data_type=DTYPE, name="gate")
        o_v, _ = g.sdpa(name="s", q=q, k=k, v=v, attn_scale=0.1, is_inference=True)
        o_v.set_dim(dims).set_stride(strides)
        _finish_output(g.mul(a=o_v, b=gate), dims, strides)
        return g

    return [
        ("non-virtual-o_v", non_virtual_o_v),
        ("sigmoid(o_v)*g", sigmoid_of_o_v_times_g),
        ("g-produced-by-a-node", g_produced_by_a_node),
        ("s-marked-output", sigmoid_output_is_an_output),
        ("second-sdpa", second_sdpa),
        ("relu-tail", relu_tail),
        ("mul-without-sigmoid", mul_by_g_without_sigmoid),
    ]


@pytest.mark.parametrize("case", [pytest.param(b, id=i) for i, b in _malformed_cases()])
def test_gate_tail_rejects_malformed_shapes(monkeypatch, case):
    """Anything that is not EXACTLY the tail is "not ours": analyze() returns
    None (the graph is not a single sdpa forward), so no FROST row is eligible
    and the classic path serves it."""
    _rubin(monkeypatch)
    g = case()
    assert ga.analyze(g) is None
    assert not _eligible(g)


def test_gate_tail_declines_a_broadcast_g_and_an_undeclared_o_v(monkeypatch):
    """Two LEGAL graphs (a broadcasting pointwise mul; a virtual O_v the user did
    not declare) that the tail matcher DOES recognise and every row then
    declines BY MESSAGE -- never facts.invalid, so the backend still serves them."""
    _rubin(monkeypatch)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    g, ts = _mk_gated_graph(gate_dims=(B, 1, S, _D256))
    facts = _facts(g)
    assert facts.has_epilogue_gate is True and facts.epilogue_gate_shape_ok is False
    assert not _eligible(g)
    assert "shape" in engines.analyze_for(spec, g)[1]
    g, ts = _mk_gated_graph(declare_o_v=False)
    facts = _facts(g)
    assert facts.has_epilogue_gate is True and facts.sdpa_o_virtual_declared is False
    assert not _eligible(g)
    assert "set_dim" in engines.analyze_for(spec, g)[1]


@pytest.mark.parametrize(
    "gate_strides",
    [(H * S * _D256, S * _D256, _D256, 1), (S * H * 260, 260, H * 260, 1)],
    ids=["head-major", "unaligned-head-stride"],
)
def test_gate_tail_declines_a_g_the_kernel_cannot_tma_load(monkeypatch, gate_strides):
    """G is TMA-loaded ZERO-COPY by the fused kernel (Q/K/V/O have a normalisation
    copy in the lowering; G has none), so a rank-4, O-SHAPED G whose strides TMA
    cannot express -- head-major (the seq stride does not cover the heads), or a
    head stride that is not a 16-byte multiple -- is a legal graph that the tail
    matcher recognises, ``epilogue_gate_layout_ok`` records as False, and the
    row declines naming the zero-copy rule.  This is the analyzer half of the
    adapter's ``TMA-expressible`` ValueError (rule 8b): without it the row
    would admit a G ``check_support`` then rejects."""
    _rubin(monkeypatch)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    g, ts = _mk_gated_graph(gate_strides=gate_strides)
    facts = _facts(g)
    assert facts.has_epilogue_gate is True and facts.epilogue_gate_t is ts["gate"]
    assert facts.epilogue_gate_shape_ok is True, "the shape is O's -- only the LAYOUT is at fault"
    assert facts.epilogue_gate_layout_ok is False
    assert not _eligible(g)
    assert "zero-copy" in engines.analyze_for(spec, g)[1]
    # The same G declared BSHD-compact is served: the layout fact is the only difference.
    assert _eligible(_mk_gated_graph()[0]) == {"sdpa_fwd_prefill_sm107"}


def test_virtual_amax_o_is_not_a_fact():
    """``sdpa_fp8`` RETURNS its Amax_O port unconditionally; only a real
    (set_output(True), non-virtual) tensor is a requested output.  A virtual one
    used to enter SdpaBinding and make the plan demand a buffer nobody has
    ("missing buffers" at execute) -- and it is what has_amax_o=False folds out."""
    import math

    dims, strides = (B, H, S, 128), (S * H * 128, 128, H * 128, 1)

    def build(request_amax_o):
        gg = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        qq = gg.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="q")
        kk = gg.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="k")
        vv = gg.tensor(dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3, name="v")
        ss = [gg.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT) for _ in range(6)]
        o, _stats, _amx_s, amx_o = gg.sdpa_fp8(
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
        if request_amax_o:
            amx_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
        return gg, amx_o

    g, amx_o = build(request_amax_o=False)
    assert amx_o.is_virtual
    facts = _facts(g)
    assert facts.amax_o_t is None, "a virtual Amax_O is not a requested output"
    assert engines.engine_name(fp8=True) in _eligible(g), "and its absence declines nothing"
    g, amx_o = build(request_amax_o=True)
    assert _facts(g).amax_o_t is amx_o
    assert engines.engine_name(fp8=True) in _eligible(g)


def test_mxfp8_virtual_amax_o_is_inferred_and_not_a_fact():
    """``sdpa_mxfp8`` infers O / Stats / Amax_O dims the way ``sdpa_fp8`` does
    (PR-B S9): an UNREQUESTED Amax_O -- virtual, never ``set_dim``'d -- passes
    validate() (it used to fail the IR-level Tensor.validate() with
    "dims not set" unless the caller declared the port it never asked for) and
    is not a fact (``amax_o_t is None``), which is what has_amax_o=False folds
    out.  The inference is PROVISIONAL: a caller's explicit set_dim / set_stride
    on every output survives validate() byte-for-byte (a non-row-major Stats
    layout is the witness), and a requested Amax_O is still the fact.  The
    LOWERING mechanism is pinned too, because it was mis-stated once: the
    sdpa-family arm of ``_lower_to_cpp`` pushes whatever dim/stride the IR
    carries, inferred or user-set (only ``push_output_attrs`` is
    user-assigned-only), so C++ receives the virtual ``[1, 1, 1, 1]`` scalar
    -- the state ``sdpa_fp8`` has always produced -- not an undimensioned port."""
    d = 256
    dims, bshd = (B, H, S, d), (S * H * d, d, H * d, 1)
    stats_dims, stats_bsh1 = (B, H, S, 1), (S * H, 1, H, 1)  # deliberately NOT row-major
    e8m0, f8_128x4 = cudnn.data_type.FP8_E8M0, cudnn.tensor_reordering.F8_128x4

    def build(*, request_amax_o):
        gg = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        qq = gg.tensor(dim=dims, stride=bshd, data_type=cudnn.data_type.FP8_E4M3, name="q")
        kk = gg.tensor(dim=dims, stride=bshd, data_type=cudnn.data_type.FP8_E4M3, name="k")
        vv = gg.tensor(dim=dims, stride=bshd, data_type=cudnn.data_type.FP8_E4M3, name="v")

        def sf(sd):  # F8_128x4 scale-factor tensors: Q/K rowwise [B,H,S,d/32], V columnwise [B,H,S/32,d]
            return gg.tensor(dim=list(sd), stride=[sd[1] * sd[2] * sd[3], sd[2] * sd[3], sd[3], 1], data_type=e8m0, reordering_type=f8_128x4)

        o, stats, amx_o = gg.sdpa_mxfp8(
            q=qq,
            k=kk,
            v=vv,
            descale_q=sf((B, H, S, d // 32)),
            descale_k=sf((B, H, S, d // 32)),
            descale_v=sf((B, H, S // 32, d)),
            attn_scale=1.0 / math.sqrt(d),
            generate_stats=True,
            use_causal_mask=True,
        )
        # Builder-time inference (graph-input dims are known): the three ports carry provisional dims.
        assert list(o.dim) == list(dims) and list(stats.dim) == list(stats_dims) and list(amx_o.dim) == [1, 1, 1, 1]
        assert amx_o.is_virtual and stats.is_virtual and o.is_virtual
        # The caller declares what it requests -- O and Stats -- exactly as the block / harness do.
        _finish_output(o, dims, bshd, dtype=cudnn.data_type.HALF)
        stats.set_output(True).set_dim(stats_dims).set_stride(stats_bsh1).set_data_type(cudnn.data_type.FLOAT)
        if request_amax_o:
            amx_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
        return gg, o, stats, amx_o

    mxfp8_name = engines.engine_name(mxfp8=True)

    # (a) every output declared explicitly, Amax_O requested: validate() passes, nothing is overridden.
    g, o, stats, amx_o = build(request_amax_o=True)
    g.validate()
    assert g._lowered_graph is None, "an mxfp8 graph a FROST row serves validates natively"
    assert tuple(o.stride) == bshd and tuple(stats.stride) == stats_bsh1, "explicit set_stride must survive the inference"
    assert tuple(o.dim) == dims and tuple(stats.dim) == stats_dims and tuple(amx_o.dim) == (1, 1, 1, 1)
    assert not amx_o.is_virtual
    facts = _facts(g)
    assert facts.amax_o_t is amx_o and facts.is_mxfp8
    assert mxfp8_name in _eligible(g)

    # (b) Amax_O left virtual and UNSET: validate() passes on the inferred [1,1,1,1]; it is not a fact.
    g, o, stats, amx_o = build(request_amax_o=False)
    g.validate()
    assert g._lowered_graph is None
    assert amx_o.is_virtual and list(amx_o.dim) == [1, 1, 1, 1] and list(amx_o.stride) == [1, 1, 1, 1]
    assert not amx_o.dim_assigned and not amx_o.stride_assigned, "inferred, not user-assigned (a set_dim/set_stride would flip these)"
    assert tuple(o.stride) == bshd and tuple(stats.stride) == stats_bsh1
    facts = _facts(g)
    assert facts.amax_o_t is None, "a virtual Amax_O is not a requested output"
    assert facts.is_mxfp8
    assert mxfp8_name in _eligible(g), "and its absence declines nothing"
    # (c) What the classic backend would receive (device-free: lowering builds the C++ graph, it does
    # not validate it): the sdpa-family lowering pushes the IR's dim/stride whether inferred or
    # user-set, so the undeclared Amax_O lands in C++ as a dimensioned VIRTUAL scalar -- the same
    # state sdpa_fp8 has always produced -- and the user's O / Stats layout arrives byte-for-byte.
    g._lower_to_cpp()
    cpp_amax, cpp_o, cpp_stats = (g._cpp_tensors[t.uid] for t in (amx_o, o, stats))
    assert cpp_amax.get_is_virtual() and list(cpp_amax.get_dim()) == [1, 1, 1, 1] and list(cpp_amax.get_stride()) == [1, 1, 1, 1]
    assert not cpp_o.get_is_virtual() and tuple(cpp_o.get_dim()) == dims and tuple(cpp_o.get_stride()) == bshd
    assert tuple(cpp_stats.get_dim()) == stats_dims and tuple(cpp_stats.get_stride()) == stats_bsh1
