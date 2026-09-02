# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend integration for the SM80 (A100) FROST SDPA engines
(``sdpa_fwd_prefill_sm80`` / ``sdpa_bwd_sm80``): registration, probe
eligibility/rejection, and graph-level forward + backward end-to-end against
a torch reference.  The end-to-end tests need an SM80 device (they run on the
Ampere leg of the frost CI job); registration/probe tests run anywhere."""

from __future__ import annotations

import dataclasses
import math
import pytest
import torch

import cudnn
import cudnn.sdpa  # noqa: F401 — the SM80 capability tables live here
from cudnn.engines import MANIFEST, is_python_engine
from frost_test_utils import select_engine

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.bwd import engines as engines_bwd_sm80
from cudnn.sdpa.fwd import engines as engines_fwd

_FWD = "sdpa_fwd_prefill_sm80"
_BWD = "sdpa_bwd_sm80"
_BWD_SPEC = next(sp for sp in engines_bwd_sm80.ENGINE_SPECS if sp.name == _BWD)
_FWD_CAPS = next(sp for sp in engines_fwd.ENGINE_SPECS if sp.name == _FWD).capabilities


def _is_sm80() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(torch.cuda.current_device()) == (8, 0)


_SM80 = pytest.mark.skipif(not _is_sm80(), reason="needs an SM80 (A100) GPU")

B, H, S, D = 2, 4, 512, 128
_HALF = cudnn.data_type.HALF
_SCALE = 1.0 / math.sqrt(D)


def _bshd_stride(b, h, s, d):
    return (s * h * d, d, h * d, 1)


def _buf(d=D):
    return torch.randn(B, S, H, d, dtype=torch.float16, device="cuda").permute(0, 2, 1, 3)


def _build_fwd_graph(*, d=D, stats_stride=None, **sdpa_kwargs):
    g = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    st = _bshd_stride(B, H, S, d)
    q = g.tensor(name="q", dim=(B, H, S, d), stride=st, data_type=_HALF)
    k = g.tensor(name="k", dim=(B, H, S, d), stride=st, data_type=_HALF)
    v = g.tensor(name="v", dim=(B, H, S, d), stride=st, data_type=_HALF)
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(d), use_causal_mask=True, generate_stats=True, **sdpa_kwargs)
    o.set_output(True).set_dim((B, H, S, d)).set_stride(st).set_data_type(_HALF)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    if stats_stride is not None:
        stats.set_dim((B, H, S, 1)).set_stride(stats_stride)
    return g, q, k, v, o, stats


def _native_then_pin(g, engine):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine)
    g.check_support()
    g.build_plans()


@pytest.mark.L0
def test_sm80_engines_registered():
    from cudnn.sdpa.bwd.engine import FrostSdpaBwdEngines
    from cudnn.sdpa.fwd.engine import FrostSdpaFwdEngines

    fwd_row = next(r for r in MANIFEST if r.factory == "FrostSdpaFwdEngines")
    bwd_row = next(r for r in MANIFEST if r.factory == "FrostSdpaBwdEngines")
    fwd_ids = fwd_row.offered_ids()
    bwd_ids = bwd_row.offered_ids()
    assert _FWD in fwd_ids and _BWD in bwd_ids  # opt-in fixture enables them
    fwd = {e.name: e for e in FrostSdpaFwdEngines(fwd_ids)}
    bwd = {e.name: e for e in FrostSdpaBwdEngines(bwd_ids)}
    assert _FWD in fwd and fwd_row.owns(fwd[_FWD].engine_id)
    assert _BWD in bwd and bwd_row.owns(bwd[_BWD].engine_id)
    assert is_python_engine(fwd[_FWD].engine_id) and is_python_engine(bwd[_BWD].engine_id)


@pytest.mark.L0
def test_probe_rejects_unsupported_features():
    """A dropout graph must leave the SM80 forward engine ineligible for a
    feature (not an arch) reason — normalize device_cc so this runs anywhere."""
    g = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    st = _bshd_stride(B, H, S, D)
    q = g.tensor(name="q", dim=(B, H, S, D), stride=st, data_type=_HALF)
    k = g.tensor(name="k", dim=(B, H, S, D), stride=st, data_type=_HALF)
    v = g.tensor(name="v", dim=(B, H, S, D), stride=st, data_type=_HALF)
    seed = g.tensor(name="seed", dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
    offset = g.tensor(name="offset", dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
    o, _ = g.sdpa(q=q, k=k, v=v, generate_stats=False, dropout=(0.1, seed, offset))
    o.set_output(True).set_data_type(_HALF)

    facts = ga.analyze(g)
    assert facts is not None and not facts.invalid
    assert facts.has_dropout
    facts = dataclasses.replace(facts, device_cc=(8, 0))
    reason = engines_fwd.mismatch(_FWD_CAPS, facts)
    assert reason is not None and "dropout" in reason


@pytest.mark.L0
def test_direction_cross_rejection():
    """The fwd engine must reject sdpa_backward graphs and vice versa."""
    g, *_ = _build_fwd_graph()
    facts = dataclasses.replace(ga.analyze(g), device_cc=(8, 0))
    assert engines_fwd.mismatch(_FWD_CAPS, facts) is None
    assert "sdpa_backward" in engines_bwd_sm80.mismatch(_BWD_SPEC.capabilities, facts)


@_SM80
@pytest.mark.L0
def test_fwd_engine_end_to_end():
    g, q, k, v, o, stats = _build_fwd_graph()
    _native_then_pin(g, _FWD)

    torch.manual_seed(0)
    q_buf, k_buf, v_buf = _buf(), _buf(), _buf()
    o_buf = torch.empty_like(q_buf)
    stats_buf = torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda")
    g.execute({q: q_buf, k: k_buf, v: v_buf, o: o_buf, stats: stats_buf}, None)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(q_buf.float(), k_buf.float(), v_buf.float(), is_causal=True, scale=_SCALE).to(torch.float16)
    torch.testing.assert_close(o_buf, ref, rtol=1e-2, atol=4e-3)
    assert torch.isfinite(stats_buf).all()


def _check_fwd_engine_strided_stats(d):
    sentinel = -12345.0
    stats_storage = torch.full((S + 7, H + 2, B), sentinel, dtype=torch.float32, device="cuda")
    strided_stats_buf = stats_storage.permute(2, 1, 0)[:, :H, :S].unsqueeze(-1)
    compact_stats_buf = torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda")
    compact_graph = _build_fwd_graph(d=d, stats_stride=compact_stats_buf.stride())
    strided_graph = _build_fwd_graph(d=d, stats_stride=strided_stats_buf.stride())
    _native_then_pin(compact_graph[0], _FWD)
    _native_then_pin(strided_graph[0], _FWD)

    torch.manual_seed(0)
    q_buf, k_buf, v_buf = _buf(d), _buf(d), _buf(d)
    for graph_tensors, stats_buf in ((compact_graph, compact_stats_buf), (strided_graph, strided_stats_buf)):
        graph, q, k, v, o, stats = graph_tensors
        graph.execute({q: q_buf, k: k_buf, v: v_buf, o: torch.empty_like(q_buf), stats: stats_buf}, None)
    torch.cuda.synchronize()

    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q_buf.float(), k_buf.float().transpose(-1, -2)) * scale
    causal_mask = torch.ones(S, S, dtype=torch.bool, device="cuda").triu(diagonal=1)
    stats_ref = torch.logsumexp(scores.masked_fill(causal_mask, float("-inf")), dim=-1)
    torch.testing.assert_close(strided_stats_buf, compact_stats_buf, rtol=0, atol=0)
    torch.testing.assert_close(strided_stats_buf.squeeze(-1), stats_ref, rtol=3e-2, atol=5e-2)

    gaps = torch.ones_like(stats_storage, dtype=torch.bool)
    gaps[:S, :H, :] = False
    assert torch.all(stats_storage[gaps] == sentinel), "the LSE store touched padding outside its declared view"


@_SM80
@pytest.mark.L0
def test_fwd_engine_strided_stats():
    """The SM80 L0 half flavor writes LSE into a permuted, gapped layout."""
    _check_fwd_engine_strided_stats(128)


@_SM80
@pytest.mark.L1
def test_fwd_engine_strided_stats_d256():
    """The SM80 D256 half flavor preserves dense Stats strides."""
    _check_fwd_engine_strided_stats(256)


@_SM80
@pytest.mark.L0
def test_bwd_engine_end_to_end():
    # Forward via the SM80 engine to produce O / stats.
    g, q, k, v, o, stats = _build_fwd_graph()
    _native_then_pin(g, _FWD)
    torch.manual_seed(0)
    q_buf, k_buf, v_buf = _buf(), _buf(), _buf()
    o_buf = torch.empty_like(q_buf)
    stats_buf = torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda")
    g.execute({q: q_buf, k: k_buf, v: v_buf, o: o_buf, stats: stats_buf}, None)

    gb = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    st = _bshd_stride(B, H, S, D)
    qb = gb.tensor(name="q", dim=(B, H, S, D), stride=st, data_type=_HALF)
    kb = gb.tensor(name="k", dim=(B, H, S, D), stride=st, data_type=_HALF)
    vb = gb.tensor(name="v", dim=(B, H, S, D), stride=st, data_type=_HALF)
    ob = gb.tensor(name="o", dim=(B, H, S, D), stride=st, data_type=_HALF)
    dob = gb.tensor(name="dO", dim=(B, H, S, D), stride=st, data_type=_HALF)
    statsb = gb.tensor(name="stats", dim=(B, H, S, 1), stride=(H * S, S, 1, 1), data_type=cudnn.data_type.FLOAT)
    dq, dk, dv = gb.sdpa_backward(q=qb, k=kb, v=vb, o=ob, dO=dob, stats=statsb, attn_scale=_SCALE, use_causal_mask=True)
    for t in (dq, dk, dv):
        t.set_output(True).set_data_type(_HALF)
    _native_then_pin(gb, _BWD)

    do_buf = _buf()
    dq_buf, dk_buf, dv_buf = torch.empty_like(q_buf), torch.empty_like(k_buf), torch.empty_like(v_buf)
    gb.execute(
        {qb: q_buf, kb: k_buf, vb: v_buf, ob: o_buf, dob: do_buf, statsb: stats_buf, dq: dq_buf, dk: dk_buf, dv: dv_buf},
        None,
    )
    torch.cuda.synchronize()

    q_ref = q_buf.detach().float().requires_grad_()
    k_ref = k_buf.detach().float().requires_grad_()
    v_ref = v_buf.detach().float().requires_grad_()
    o_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True, scale=_SCALE)
    o_ref.backward(do_buf.float())

    torch.testing.assert_close(dq_buf.float(), q_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(dk_buf.float(), k_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(dv_buf.float(), v_ref.grad, rtol=3e-2, atol=3e-2)


@_SM80
@pytest.mark.L0
@pytest.mark.parametrize("gqa", [1, 4], ids=["mha", "gqa4x"])
def test_fwd_engine_bhsd_contiguous_layout(gqa):
    """dense_flex delivery: BHSD-contiguous buffers (the test_mhas_v2 norm)
    and GQA head expansion must both be normalized by the lowering — this was
    the CI 'stride order' failure of 2026-07-29."""
    h_kv = H // gqa
    g = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    st_q = (H * S * D, S * D, D, 1)  # BHSD-contiguous
    st_kv = (h_kv * S * D, S * D, D, 1)
    q = g.tensor(name="q", dim=(B, H, S, D), stride=st_q, data_type=_HALF)
    k = g.tensor(name="k", dim=(B, h_kv, S, D), stride=st_kv, data_type=_HALF)
    v = g.tensor(name="v", dim=(B, h_kv, S, D), stride=st_kv, data_type=_HALF)
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=_SCALE, use_causal_mask=True, generate_stats=True)
    o.set_output(True).set_data_type(_HALF)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _native_then_pin(g, _FWD)

    torch.manual_seed(0)
    q_buf = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    k_buf = torch.randn(B, h_kv, S, D, dtype=torch.float16, device="cuda")
    v_buf = torch.randn(B, h_kv, S, D, dtype=torch.float16, device="cuda")
    o_buf = torch.empty_like(q_buf)
    stats_buf = torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda")
    g.execute({q: q_buf, k: k_buf, v: v_buf, o: o_buf, stats: stats_buf}, None)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_buf.float(), k_buf.float().repeat_interleave(gqa, dim=1), v_buf.float().repeat_interleave(gqa, dim=1), is_causal=True, scale=_SCALE
    ).to(torch.float16)
    torch.testing.assert_close(o_buf, ref, rtol=1e-2, atol=4e-3)


@_SM80
@pytest.mark.L0
def test_bwd_engine_bhsd_contiguous_layout():
    """Backward counterpart of the dense_flex layout regression."""
    st = (H * S * D, S * D, D, 1)  # BHSD-contiguous
    dims = (B, H, S, D)
    torch.manual_seed(0)
    bufs = {n: torch.randn(dims, dtype=torch.float16, device="cuda") for n in ("q", "k", "v", "do")}

    # LSE from a matching fwd run (wrapper path; layout-independent).
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80

    def phys(t):
        return t.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)

    fwd = sdpa_fwd_wrapper_sm80(q_tensor=phys(bufs["q"]), k_tensor=phys(bufs["k"]), v_tensor=phys(bufs["v"]), is_causal=True, scale_softmax=_SCALE)
    o_buf = fwd["o_tensor"].contiguous()  # BHSD-contiguous view of O
    stats_buf = fwd["lse_tensor"].reshape(B, H, S, 1).contiguous()

    gb = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    qb = gb.tensor(name="q", dim=dims, stride=st, data_type=_HALF)
    kb = gb.tensor(name="k", dim=dims, stride=st, data_type=_HALF)
    vb = gb.tensor(name="v", dim=dims, stride=st, data_type=_HALF)
    ob = gb.tensor(name="o", dim=dims, stride=st, data_type=_HALF)
    dob = gb.tensor(name="dO", dim=dims, stride=st, data_type=_HALF)
    statsb = gb.tensor(name="stats", dim=(B, H, S, 1), stride=(H * S, S, 1, 1), data_type=cudnn.data_type.FLOAT)
    dq, dk, dv = gb.sdpa_backward(q=qb, k=kb, v=vb, o=ob, dO=dob, stats=statsb, attn_scale=_SCALE, use_causal_mask=True)
    for t in (dq, dk, dv):
        t.set_output(True).set_data_type(_HALF)
    _native_then_pin(gb, _BWD)

    dq_buf, dk_buf, dv_buf = (torch.empty(dims, dtype=torch.float16, device="cuda") for _ in range(3))
    gb.execute(
        {qb: bufs["q"], kb: bufs["k"], vb: bufs["v"], ob: o_buf, dob: bufs["do"], statsb: stats_buf, dq: dq_buf, dk: dk_buf, dv: dv_buf},
        None,
    )
    torch.cuda.synchronize()

    q_ref = bufs["q"].detach().float().requires_grad_()
    k_ref = bufs["k"].detach().float().requires_grad_()
    v_ref = bufs["v"].detach().float().requires_grad_()
    o_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True, scale=_SCALE)
    o_ref.backward(bufs["do"].float())
    torch.testing.assert_close(dq_buf.float(), q_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(dk_buf.float(), k_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(dv_buf.float(), v_ref.grad, rtol=3e-2, atol=3e-2)
