# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM120 DSL per-tensor FP8 SDPA-forward engine.

Drives ``graph.sdpa_fp8`` (FP8 E4M3 Q/K/V + scalar per-tensor descales) routed to
the ``sdpa_fwd_prefill_sm120_fp8`` engine, and validates O against an fp32-dequant
reference. ``Amax_S`` and ``Amax_O`` are both produced in-kernel (bitcast-int32
atomicMax over the pre-cast fp32 values); both are checked.

SM120 v1 envelope (see engines._sm120_fp8_spec): E4M3 in / FP16 out only, exact
d128, causal / bottom-right / SWA / KV-padding masks; no sink (Amax_S semantics),
no THD. E5M2 and fp8 outputs are covered by negative tests.

Requires: SM120/SM121 (consumer Blackwell), cutlass-dsl. Skips otherwise.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_blackwell_geforce, requires_dsl


def _select_engine(graph, name):
    """Pin the ranked entry named ``name``. A pin is strict: check_support /
    build_plans raise if that engine declines the graph."""
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    assert name in names, f"engine {name!r} did not claim this graph; plans={names}"
    graph.select_plan(names.index(name))
    return graph


pytestmark = [requires_blackwell_geforce, requires_dsl]

_E4M3_MAX = 448.0


def _quant(x):
    dq = (x.abs().amax().clamp_min(1e-8) / _E4M3_MAX).item()
    return (x / dq).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn), dq


def _ref(qd, kd, vd, *, scale, is_causal=False, bottom_right=False, swa_window=None, seq_lens_kv=None):
    b, h_q, s_q, _ = qd.shape
    _, h_kv, s_kv, _ = vd.shape
    dev = qd.device
    g = h_q // h_kv
    k_e = kd.repeat_interleave(g, dim=1)
    v_e = vd.repeat_interleave(g, dim=1)
    scores = torch.matmul(qd, k_e.transpose(-1, -2)) * scale
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    masked = torch.zeros(1, 1, s_q, s_kv, dtype=torch.bool, device=dev)
    if is_causal:
        lim = i + (s_kv - s_q) if bottom_right else i
        masked = masked | (j > lim)
    if swa_window is not None:
        masked = masked | (j < i - swa_window)
    if seq_lens_kv is not None:
        slk = torch.as_tensor(seq_lens_kv, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        masked = masked | (j >= slk)
    scores = scores.masked_fill(masked, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_e), probs.max().item()


def _run(B, H_q, H_kv, S_q, S_kv, *, scale, sdpa_kwargs, seq_lens_kv=None):
    import cudnn

    dev = "cuda"
    D = 128
    Qf = torch.randn(B, H_q, S_q, D, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S_kv, D, device=dev) * 0.5
    Vf = torch.randn(B, H_kv, S_kv, D, device=dev) * 0.5
    Q8, dq = _quant(Qf)
    K8, dk = _quant(Kf)
    V8, dv = _quant(Vf)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    Ob = torch.empty(B, S_q, H_q, D, device=dev, dtype=torch.float16).transpose(1, 2)
    lse = torch.empty(B, H_q, S_q, 1, device=dev, dtype=torch.float32)
    amax_s = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
    amax_o = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    def sc(val):
        return torch.tensor([[[[val]]]], dtype=torch.float32, device=dev)

    dqt, dkt, dvt, dst, sst, sot = sc(dq), sc(dk), sc(dv), sc(1.0), sc(1.0), sc(1.0)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor_like(Qb)
    k = g.tensor_like(Kb)
    v = g.tensor_like(Vb)

    def _stns():
        return g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

    dqn, dkn, dvn, dsn, ssn, son = (_stns() for _ in range(6))
    kw = dict(q=q, k=k, v=v, descale_q=dqn, descale_k=dkn, descale_v=dvn, descale_s=dsn, scale_s=ssn, scale_o=son, attn_scale=scale, generate_stats=True)
    vp = {q: Qb, k: Kb, v: Vb, dqn: dqt, dkn: dkt, dvn: dvt, dsn: dst, ssn: sst, son: sot}
    if seq_lens_kv is not None:
        slq = torch.full((B, 1, 1, 1), S_q, dtype=torch.int32, device=dev)
        slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        sq_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        skv_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_q=sq_h, seq_len_kv=skv_h)
        vp[sq_h] = slq
        vp[skv_h] = slk
    kw.update(sdpa_kwargs)
    o, stats, amx_s, amx_o = g.sdpa_fp8(**kw)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(cudnn.data_type.HALF)
    stats.set_output(True).set_dim([B, H_q, S_q, 1]).set_stride([H_q * S_q, S_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_s.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(arch="sm120", fp8=True))
    g.check_support()
    g.build_plans()
    vp.update({o: Ob, stats: lse, amx_s: amax_s, amx_o: amax_o})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    ref_kw = _ref_kwargs(sdpa_kwargs)
    o_ref, amax_s_ref = _ref(Q8.float() * dq, K8.float() * dk, V8.float() * dv, scale=scale, seq_lens_kv=seq_lens_kv, **ref_kw)
    return Ob, o_ref, amax_s.item(), amax_s_ref, amax_o.item(), o_ref.abs().max().item()


def _ref_kwargs(sdpa_kwargs):
    out = {}
    if sdpa_kwargs.get("use_causal_mask"):
        out["is_causal"] = True
    if sdpa_kwargs.get("use_causal_mask_bottom_right"):
        out["is_causal"] = True
        out["bottom_right"] = True
    lb = sdpa_kwargs.get("left_bound")
    if lb is not None:
        out["swa_window"] = lb - 1
    return out


def _check(O, O_ref, amax_s, amax_s_ref, amax_o, amax_o_ref):
    diff = (O.float() - O_ref).abs().max().item()
    assert diff <= 5e-2, f"max|O-ref|={diff:.4f} > 0.05"
    assert abs(amax_s - amax_s_ref) <= 0.03, f"amax_s {amax_s:.4f} vs ref {amax_s_ref:.4f}"
    assert abs(amax_o - amax_o_ref) <= 0.03, f"amax_o {amax_o:.4f} vs ref {amax_o_ref:.4f}"


_MASKS = {
    "none": {},
    "causal": dict(use_causal_mask=True),
    "causal_br": dict(use_causal_mask_bottom_right=True),
    "swa": dict(use_causal_mask=True, left_bound=65),
}


@pytest.mark.L0
@pytest.mark.parametrize("mask", list(_MASKS))
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_masks(mask):
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=_MASKS[mask])
    _check(O, O_ref, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_gqa():
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 2, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(O, O_ref, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_bottom_right_rectangular():
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 8, 128, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask_bottom_right=True))
    _check(O, O_ref, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_multi_tile_long_seq():
    # 1k x 1k exercises the multi-KV-tile online-softmax rescale path.
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(1, 4, 4, 1024, 1024, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(O, O_ref, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_padding(causal):
    # KV padding: batch 0 uses all 256 KV cols, batch 1 only 192 (partial tile).
    scale = 1.0 / math.sqrt(128)
    sk = dict(use_causal_mask=True) if causal else {}
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=sk, seq_lens_kv=[256, 192])
    _check(O, O_ref, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_e5m2_not_offered():
    """The v1 kernel hardcodes the e4m3 MMA tag; E5M2 graphs must not route here."""
    import cudnn

    dev = "cuda"
    B, H, S, D = 1, 4, 256, 128
    X = torch.randn(B, S, H, D, device=dev).to(torch.float8_e5m2).transpose(1, 2)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E5M2, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(X), g.tensor_like(X), g.tensor_like(X)
    scalars = [g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT) for _ in range(6)]
    o, stats, amx_s, amx_o = g.sdpa_fp8(
        q=q,
        k=k,
        v=v,
        descale_q=scalars[0],
        descale_k=scalars[1],
        descale_v=scalars[2],
        descale_s=scalars[3],
        scale_s=scalars[4],
        scale_o=scalars[5],
        attn_scale=1.0 / math.sqrt(D),
        generate_stats=True,
    )
    o.set_output(True).set_dim([B, H, S, D]).set_stride([S * H * D, D, H * D, 1]).set_data_type(cudnn.data_type.HALF)
    stats.set_output(True).set_dim([B, H, S, 1]).set_stride([H * S, S, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_s.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    try:
        g.create_execution_plans([cudnn.heur_mode.A])
    except cudnn.cudnnGraphNotSupportedError:
        # Nothing — python engine or backend — serves E5M2 here: also a pass
        # (the point is only that the e4m3-tagged sm120 fp8 cell declined).
        return
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    assert engine_name(arch="sm120", fp8=True) not in names, f"E5M2 graph must not offer the sm120 fp8 engine; plans={names}"
