# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL per-tensor FP8 SDPA-forward engine.

Drives ``graph.sdpa_fp8`` (FP8 E4M3/E5M2 Q/K/V + scalar per-tensor descales) routed to
the ``sdpa_fwd_prefill_sm100_d128_fp8`` engine, and validates O against an fp32-dequant
reference. ``Amax_S`` and ``Amax_O`` are both produced in-kernel (atomicMax over the
pre-cast fp32 values); both are checked.

cuDNN's ``sdpa_fp8`` op exposes causal / bottom-right / sliding-window masks, attention
sink, and a padding mask (per-batch ``seq_len_kv`` → KV-side masking, tested here). THD /
ragged inputs are still deferred (engine declares thd=False).

Requires: SM100 (Blackwell), cutlass-dsl, cuDNN >= 9.21 (fp8 SDPA). Skips otherwise.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(torch.cuda.current_device()) == (10, 0)


def _deps_available() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


def _select_engine(graph, name):
    """Pin the ranked entry named ``name`` (graph.plans holds the backend's
    plans and the python engines' in one list). A pin is strict: check_support /
    build_plans raise if that engine declines the graph."""
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    assert name in names, f"engine {name!r} did not claim this graph; plans={names}"
    graph.select_plan(names.index(name))
    return graph


pytestmark = pytest.mark.skipif(
    not (_is_sm100() and _deps_available()),
    reason="FP8 SDPA engine requires an SM100 (Blackwell) device + cutlass-dsl.",
)

_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
_OUT = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_CUDNN_OTYPE = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float8_e4m3fn: "FP8_E4M3", torch.float8_e5m2: "FP8_E5M2"}


def _quant(x, in_key):
    fp8, fmax = _FP8[in_key], _FP8_MAX[in_key]
    dq = (x.abs().amax().clamp_min(1e-8) / fmax).item()
    return (x / dq).clamp(-fmax, fmax).to(fp8), dq


def _ref(qd, kd, vd, *, scale, is_causal=False, bottom_right=False, swa_window=None, sinks=None, seq_lens_kv=None):
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
        # Per-batch KV padding: columns j >= seq_len_kv[b] are padding -> masked.
        slk = torch.as_tensor(seq_lens_kv, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        masked = masked | (j >= slk)
    scores = scores.masked_fill(masked, float("-inf"))
    if sinks is not None:
        col = sinks.view(1, h_q, 1, 1).float().expand(b, h_q, s_q, 1).to(dev)
        probs = torch.softmax(torch.cat([scores, col], dim=-1), dim=-1)
        return torch.matmul(probs[..., :s_kv], v_e), probs[..., :s_kv].max().item()
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_e), probs.max().item()


def _run(B, H_q, H_kv, S_q, S_kv, in_key, out_dt, *, scale, sdpa_kwargs, sink=None, seq_lens_kv=None):
    import cudnn

    dev = "cuda"
    D = 128
    Qf = torch.randn(B, H_q, S_q, D, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S_kv, D, device=dev) * 0.5
    Vf = torch.randn(B, H_kv, S_kv, D, device=dev) * 0.5
    Q8, dq = _quant(Qf, in_key)
    K8, dk = _quant(Kf, in_key)
    V8, dv = _quant(Vf, in_key)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    Ob = torch.empty(B, S_q, H_q, D, device=dev, dtype=out_dt).transpose(1, 2)
    lse = torch.empty(B, H_q, S_q, 1, device=dev, dtype=torch.float32)
    amax_s = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
    amax_o = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    def sc(val):
        return torch.tensor([[[[val]]]], dtype=torch.float32, device=dev)

    dqt, dkt, dvt, dst, sst, sot = sc(dq), sc(dk), sc(dv), sc(1.0), sc(1.0), sc(1.0)

    g = cudnn.pygraph(
        io_data_type=getattr(cudnn.data_type, _CUDNN_ITYPE[in_key]), intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
    )
    q = g.tensor_like(Qb)
    k = g.tensor_like(Kb)
    v = g.tensor_like(Vb)

    def _stns():
        return g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

    dqn, dkn, dvn, dsn, ssn, son = (_stns() for _ in range(6))
    kw = dict(q=q, k=k, v=v, descale_q=dqn, descale_k=dkn, descale_v=dvn, descale_s=dsn, scale_s=ssn, scale_o=son, attn_scale=scale, generate_stats=True)
    vp = {q: Qb, k: Kb, v: Vb, dqn: dqt, dkn: dkt, dvn: dvt, dsn: dst, ssn: sst, son: sot}
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    if seq_lens_kv is not None:
        # KV padding: full (unpadded) query lengths + per-batch valid KV lengths.
        slq = torch.full((B, 1, 1, 1), S_q, dtype=torch.int32, device=dev)
        slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        sq_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        skv_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_q=sq_h, seq_len_kv=skv_h)
        vp[sq_h] = slq
        vp[skv_h] = slk
    kw.update(sdpa_kwargs)
    o, stats, amx_s, amx_o = g.sdpa_fp8(**kw)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    stats.set_output(True).set_dim([B, H_q, S_q, 1]).set_stride([H_q * S_q, S_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_s.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(128, fp8=True))
    g.check_support()
    g.build_plans()
    vp.update({o: Ob, stats: lse, amx_s: amax_s, amx_o: amax_o})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    ref_kw = _ref_kwargs(sdpa_kwargs)
    if sink is not None:
        ref_kw["sinks"] = sink.flatten()
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


def _check(O, O_ref, out_dt, in_key, amax_s, amax_s_ref, amax_o, amax_o_ref):
    atol = 7e-2 if in_key == "e5m2" else 5e-2
    diff = (O.float() - O_ref).abs().max().item()
    if out_dt in (torch.float8_e4m3fn, torch.float8_e5m2):
        floor = (O_ref - O_ref.to(out_dt).float()).abs().max().item()
        atol = max(atol, 3.0 * floor)
    assert diff <= atol, f"max|O-ref|={diff:.4f} > {atol:.4f}"
    # Amax_S and Amax_O are both produced in-kernel (atomicMax over the pre-cast fp32
    # values), so they match the exact fp32 reference for every output dtype, incl. FP8.
    assert abs(amax_s - amax_s_ref) <= 0.03, f"amax_s {amax_s:.4f} vs ref {amax_s_ref:.4f}"
    assert abs(amax_o - amax_o_ref) <= 0.03, f"amax_o {amax_o:.4f} vs ref {amax_o_ref:.4f}"


_INS = ["e4m3", "e5m2"]
_MASKS = {
    "none": {},
    "causal": dict(use_causal_mask=True),
    "causal_br": dict(use_causal_mask_bottom_right=True),
    # sdpa_fp8 spells the SWA left window as `left_bound` (maps to the
    # diagonal_band_left_bound node param that the analyzer reads).
    "swa": dict(use_causal_mask=True, left_bound=65),
}


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("mask", list(_MASKS))
@torch_fork_set_rng(seed=0)
def test_fp8_masks(in_key, mask):
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=_MASKS[mask])
    _check(O, O_ref, torch.float16, in_key, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("out_key", ["fp16", "bf16", "e4m3", "e5m2"])
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_fp8_output_dtypes(in_key, out_key):
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(1, 8, 8, 512, 512, in_key, _OUT[out_key], scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(O, O_ref, _OUT[out_key], in_key, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_fp8_sink(in_key):
    scale = 1.0 / math.sqrt(128)
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), sink=sink)
    _check(O, O_ref, torch.float16, in_key, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_fp8_gqa(in_key):
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 2, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(O, O_ref, torch.float16, in_key, a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_bottom_right_rectangular():
    scale = 1.0 / math.sqrt(128)
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 8, 128, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask_bottom_right=True))
    _check(O, O_ref, torch.float16, "e4m3", a_s, a_s_ref, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_fp8_padding(in_key, causal):
    # KV padding mask: batch 0 uses all 256 KV cols, batch 1 only 192 (a partial
    # KV tile).  Exercises per-batch eff_seqlen_kv masking AND the in-kernel amax_s
    # / amax_o over the padding-masked scores (padded cols/rows must not leak or
    # poison the global amax).
    scale = 1.0 / math.sqrt(128)
    sk = dict(use_causal_mask=True) if causal else {}
    O, O_ref, a_s, a_s_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=sk, seq_lens_kv=[256, 192])
    _check(O, O_ref, torch.float16, in_key, a_s, a_s_ref, a_o, a_o_ref)
