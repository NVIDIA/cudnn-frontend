# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL per-tensor FP8 SDPA-forward engine.

Drives ``graph.sdpa_fp8`` (FP8 E4M3/E5M2 Q/K/V + scalar per-tensor descales) routed to
the exact d128/d128 or d192/d128 engine, and validates O against an fp32-dequant
reference. ``Amax_O`` is produced in-kernel (atomicMax over the pre-cast fp32 values)
and checked.

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
from frost_test_utils import requires_blackwell, requires_dsl


from frost_test_utils import select_engine as _select_engine  # noqa: F401

pytestmark = [requires_blackwell, requires_dsl]

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
        return torch.matmul(probs[..., :s_kv], v_e)
    row_has_kv = torch.isfinite(scores).any(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(row_has_kv, probs, torch.zeros_like(probs))
    return torch.matmul(probs, v_e)


def _run(
    B,
    H_q,
    H_kv,
    S_q,
    S_kv,
    in_key,
    out_dt,
    *,
    scale,
    sdpa_kwargs,
    sink=None,
    seq_lens_kv=None,
    s_scale=1.0,
    s_descale_gain=1.0,
    stats=True,
    sync_debug=False,
    d_qk=128,
    d_v=128,
):
    import cudnn

    dev = "cuda"
    Qf = torch.randn(B, H_q, S_q, d_qk, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S_kv, d_qk, device=dev) * 0.5
    Vf = torch.randn(B, H_kv, S_kv, d_v, device=dev) * 0.5
    Q8, dq = _quant(Qf, in_key)
    K8, dk = _quant(Kf, in_key)
    V8, dv = _quant(Vf, in_key)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    Ob = torch.empty(B, S_q, H_q, d_v, device=dev, dtype=out_dt).transpose(1, 2)
    lse = torch.empty(B, H_q, S_q, 1, device=dev, dtype=torch.float32)
    amax_o = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    def sc(val):
        return torch.tensor([[[[val]]]], dtype=torch.float32, device=dev)

    # Reciprocal S scales: this kernel casts P unscaled, which is exact for any
    # reciprocal pair. s_descale_gain breaks the reciprocity to test the guard.
    dqt, dkt, dvt, dst, sst, sot = sc(dq), sc(dk), sc(dv), sc(s_descale_gain / s_scale), sc(s_scale), sc(1.0)

    g = cudnn.pygraph(
        io_data_type=getattr(cudnn.data_type, _CUDNN_ITYPE[in_key]), intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
    )
    q = g.tensor_like(Qb)
    k = g.tensor_like(Kb)
    v = g.tensor_like(Vb)

    def _stns():
        return g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

    dqn, dkn, dvn, dsn, ssn, son = (_stns() for _ in range(6))
    kw = dict(q=q, k=k, v=v, descale_q=dqn, descale_k=dkn, descale_v=dvn, descale_s=dsn, scale_s=ssn, scale_o=son, attn_scale=scale, generate_stats=stats)
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
    o, stats_t, _amx_s_unused, amx_o = g.sdpa_fp8(**kw)  # Amax_S: not requested (engines decline graphs that declare it)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    if stats:
        stats_t.set_output(True).set_dim([B, H_q, S_q, 1]).set_stride([H_q * S_q, S_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(d_qk, d_v=d_v, fp8=True))
    g.check_support()
    g.build_plans()
    if not stats:
        # No Stats output: the kernel compiles the LSE store out (has_lse=False)
        # — no dummy buffer exists at any level, so the dense workspace is 0.
        assert g.get_workspace_size() == 0
    vp.update({o: Ob, amx_o: amax_o})
    if stats:
        vp[stats_t] = lse
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    if sync_debug:
        # Rule 3 pin: execute must not read the scale tensors (or anything
        # else) back to the host.
        prev_sync_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode(2)
    try:
        g.execute(vp, ws)
    finally:
        if sync_debug:
            torch.cuda.set_sync_debug_mode(prev_sync_mode)
    torch.cuda.synchronize()

    ref_kw = _ref_kwargs(sdpa_kwargs)
    if sink is not None:
        ref_kw["sinks"] = sink.flatten()
    o_ref = _ref(Q8.float() * dq, K8.float() * dk, V8.float() * dv, scale=scale, seq_lens_kv=seq_lens_kv, **ref_kw)
    return Ob, o_ref, amax_o.item(), o_ref.abs().max().item()


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


def _check(out, o_ref, out_dt, in_key, amax_o, amax_o_ref):
    atol = 7e-2 if in_key == "e5m2" else 5e-2
    diff = (out.float() - o_ref).abs().max().item()
    if out_dt in (torch.float8_e4m3fn, torch.float8_e5m2):
        floor = (o_ref - o_ref.to(out_dt).float()).abs().max().item()
        atol = max(atol, 3.0 * floor)
    assert diff <= atol, f"max|O-ref|={diff:.4f} > {atol:.4f}"
    # Amax_O is produced in-kernel (atomicMax over the pre-cast fp32
    # values), so they match the exact fp32 reference for every output dtype, incl. FP8.
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
    out, o_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=_MASKS[mask])
    _check(out, o_ref, torch.float16, in_key, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("out_key", ["fp16", "bf16", "e4m3", "e5m2"])
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_fp8_output_dtypes(in_key, out_key):
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref = _run(1, 8, 8, 512, 512, in_key, _OUT[out_key], scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(out, o_ref, _OUT[out_key], in_key, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("out_key", ["fp16", "bf16", "e4m3", "e5m2"])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
@torch_fork_set_rng(seed=0)
def test_fp8_d192_d128_output_dtypes(in_key, out_key):
    """Exact DSv3 shape: FP8 Q/K use d192 while V and O use d128."""
    scale = 1.0 / math.sqrt(192)
    out, o_ref, a_o, a_o_ref = _run(
        1,
        8,
        8,
        512,
        512,
        in_key,
        _OUT[out_key],
        scale=scale,
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=192,
        d_v=128,
    )
    _check(out, o_ref, _OUT[out_key], in_key, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal_br", "swa"])
@torch_fork_set_rng(seed=0)
def test_fp8_d192_d128_masks(mask):
    # B*H_q=16 selects the grouped LPT decoder used by the target workload.
    scale = 1.0 / math.sqrt(192)
    out, o_ref, a_o, a_o_ref = _run(
        2,
        8,
        8,
        256,
        256,
        "e4m3",
        torch.float16,
        scale=scale,
        sdpa_kwargs=_MASKS[mask],
        d_qk=192,
        d_v=128,
    )
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_d192_d128_zero_length_kv():
    """A zero-length KV batch must produce a finite zero output."""
    scale = 1.0 / math.sqrt(192)
    out, o_ref, a_o, a_o_ref = _run(
        2,
        8,
        8,
        256,
        256,
        "e4m3",
        torch.float16,
        scale=scale,
        sdpa_kwargs={},
        seq_lens_kv=[256, 0],
        d_qk=192,
        d_v=128,
    )
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_fp8_sink(in_key):
    scale = 1.0 / math.sqrt(128)
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    out, o_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), sink=sink)
    _check(out, o_ref, torch.float16, in_key, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_fp8_gqa(in_key):
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref = _run(2, 8, 2, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(out, o_ref, torch.float16, in_key, a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_bottom_right_rectangular():
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref = _run(2, 8, 8, 128, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask_bottom_right=True))
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_fp8_padding(in_key, causal):
    # KV padding mask: batch 0 uses all 256 KV cols, batch 1 only 192 (a partial
    # KV tile).  Exercises per-batch eff_seqlen_kv masking AND the in-kernel amax_o
    # / amax_o over the padding-masked scores (padded cols/rows must not leak or
    # poison the global amax).
    scale = 1.0 / math.sqrt(128)
    sk = dict(use_causal_mask=True) if causal else {}
    out, o_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=sk, seq_lens_kv=[256, 192])
    _check(out, o_ref, torch.float16, in_key, a_o, a_o_ref)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_fp8_stats_less_zero_workspace(in_key):
    """No Stats output: the kernel compiles the LSE store out (has_lse=False),
    no dummy buffer exists at any level, and the dense graph reports
    ``get_workspace_size() == 0`` (asserted inside ``_run``). The Amax_O
    atomicMax write is independent of the LSE and still produced."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), stats=False)
    _check(out, o_ref, torch.float16, in_key, a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm100_execute_lse_contract():
    """Strict lse_tensor execute contract, both directions (f16 parity):
    has_lse is keyed on sample_lse at compile time, so a requested LSE must be
    bound and an unrequested one is rejected (the store is compiled out —
    there is no LSE slot, and no cached dummy fallback either)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 1, 4, 256, 128
    dev = "cuda"
    qf = torch.randn(b, h, s, d, device=dev) * 0.5
    q8, dq = _quant(qf, "e4m3")
    q = q8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)
    k, v = q.clone(), q.clone()
    o = torch.empty(b, s, h, d, device=dev, dtype=torch.float16).transpose(1, 2)
    lse = torch.empty(b, h, s, dtype=torch.float32, device=dev)
    dsc = torch.tensor([dq], dtype=torch.float32, device=dev)

    kw = dict(sample_q=q, sample_k=k, sample_v=v, sample_o=o, is_causal=True, pertensor_fp8=True, dtype_o=torch.float16)
    ex = dict(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, descale_q=dsc, descale_k=dsc, descale_v=dsc)

    api = SdpaFwdDslSm100(sample_lse=lse, **kw)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="lse_tensor is required"):
        api.execute(**ex)

    api = SdpaFwdDslSm100(**kw)
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="without an LSE output"):
        api.execute(lse_tensor=lse, **ex)
    api.execute(**ex)
    torch.cuda.synchronize()
    o_ref = _ref(q8.float() * dq, q8.float() * dq, q8.float() * dq, scale=api.scale_softmax, is_causal=True)
    torch.testing.assert_close(o.float(), o_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("gain", [2.0, 0.5])
@torch_fork_set_rng(seed=0)
def test_fp8_sm100_s_scales_ignored(gain):
    """Scale_S/Descale_S are unsupported knobs on this cell (P is cast
    unscaled): their values are accepted and IGNORED -- nothing reads them,
    host or device (the old execute-time reciprocal check was itself a Rule 3
    readback). A wild non-reciprocal pair therefore produces the bitwise-same
    O as unit scales.
    """
    scale = 1.0 / math.sqrt(128)
    torch.manual_seed(2024)
    unit = _run(2, 8, 8, 256, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    torch.manual_seed(2024)
    wild = _run(2, 8, 8, 256, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), s_scale=8.0, s_descale_gain=gain)
    assert torch.equal(unit[0], wild[0]), "descale_s/scale_s values must not reach the kernel"


@pytest.mark.L0
@pytest.mark.parametrize("s_scale", [8.0, 448.0])
@torch_fork_set_rng(seed=0)
def test_fp8_sm100_accepts_reciprocal_s_scales(s_scale):
    """Any reciprocal pair is accepted and gives the same O as unit scales --
    the scales cancel, and the kernel applies neither.
    """
    scale = 1.0 / math.sqrt(128)
    torch.manual_seed(2024)
    unit = _run(2, 8, 8, 256, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    torch.manual_seed(2024)
    scaled = _run(2, 8, 8, 256, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), s_scale=s_scale)
    assert torch.equal(unit[0], scaled[0]), "reciprocal S scales must not change O"


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_fp8_sm100_device_scales_execute_reads_no_device_memory():
    """The graph path binds DEVICE scale tensors and the kernel folds
    descale_q*descale_k / descale_v*scale_o in-kernel; amax_o divides by the
    device scale_o -- no .item()/D2H readback anywhere (Rule 3), pinned by
    sync-debug mode 2 around the execute. Numerics vs the fp32 reference are
    unchanged."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref = _run(2, 8, 8, 256, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), sync_debug=True)
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)
