# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL per-tensor FP8 SDPA-forward engine.

Drives ``graph.sdpa_fp8`` (FP8 E4M3/E5M2 Q/K/V + scalar per-tensor descales) routed to
the d128/d128 engine (which also serves the dense d<=128 head-dim ENVELOPE via TMA
zero-padding) or the exact d192/d128 engine, and validates O against an fp32-dequant
reference. ``Amax_O`` is produced in-kernel (atomicMax over the pre-cast fp32 values)
and checked.

cuDNN's ``sdpa_fp8`` op exposes causal / bottom-right / sliding-window masks, attention
sink, a padding mask (per-batch ``seq_len_kv`` → KV-side masking, tested here), and THD /
ragged inputs (packed Q/K/V/O + per-operand ragged_offset + seq_len_q/kv or the
cu_seq_len prefix-sum form, tested here — write_thd_meta envelope design, issue #552;
ragged Stats in the packed token-major TH1 layout).

Requires: SM100 (Blackwell), cutlass-dsl, cuDNN >= 9.21 (fp8 SDPA). Skips otherwise.
"""

import math
from typing import NamedTuple

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import _SM, make_dense_stats, requires_blackwell, requires_dsl


from frost_test_utils import select_engine as _select_engine  # noqa: F401

pytestmark = [requires_blackwell, requires_dsl]

# ONE per-tensor FP8 engine per arch line (sm100 = pre-Rubin Blackwell
# 100-106, sm107 = the Rubin line): pin the engine that serves the device
# under test. The d192xd128 kernel flavor exists on the sm100 engine only.
_D128_ARCH = "sm107" if _SM == 107 else "sm100"
_skip_on_rubin = pytest.mark.skipif(_SM == 107, reason="the d192xd128 per-tensor FP8 flavor has no Rubin kernel (sm107 serves d128 only)")

_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
_OUT = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_CUDNN_OTYPE = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float8_e4m3fn: "FP8_E4M3", torch.float8_e5m2: "FP8_E5M2"}


class _ReferenceWithStats(NamedTuple):
    output: torch.Tensor
    stats: torch.Tensor


class _RunResult(NamedTuple):
    output: torch.Tensor
    reference: torch.Tensor
    amax: float
    reference_amax: float


class _RunWithStatsResult(NamedTuple):
    output: torch.Tensor
    reference: torch.Tensor
    amax: float
    reference_amax: float
    stats: torch.Tensor
    reference_stats: torch.Tensor


def _quant(x, in_key):
    fp8, fmax = _FP8[in_key], _FP8_MAX[in_key]
    dq = (x.abs().amax().clamp_min(1e-8) / fmax).item()
    return (x / dq).clamp(-fmax, fmax).to(fp8), dq


def _ref(qd, kd, vd, *, scale, is_causal=False, bottom_right=False, swa_window=None, sinks=None, seq_lens_kv=None, return_stats=False):
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
        ext = torch.cat([scores, col], dim=-1)
        probs = torch.softmax(ext, dim=-1)
        o = torch.matmul(probs[..., :s_kv], v_e)
        if return_stats:
            return _ReferenceWithStats(o, torch.logsumexp(ext, dim=-1))
        return o
    row_has_kv = torch.isfinite(scores).any(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(row_has_kv, probs, torch.zeros_like(probs))
    o = torch.matmul(probs, v_e)
    if return_stats:
        return _ReferenceWithStats(o, torch.logsumexp(scores, dim=-1))
    return o


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
    pack_gqa=None,
    stats_layout="contiguous",
    return_lse=False,
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
    lse = make_dense_stats(B, H_q, S_q, stats_layout)
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
        stats_t.set_output(True).set_dim([B, H_q, S_q, 1]).set_stride(list(lse.stride())).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    # ONE fp8 engine per arch line — kernel-flavor choice (d128 vs d192xd128,
    # and the head-dim envelope) happens inside the lowering.
    _select_engine(g, engine_name(arch=_D128_ARCH, fp8=True), pack_gqa=pack_gqa)
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
    if return_lse:
        assert stats, "return_lse requires stats=True"
        o_ref, lse_ref = _ref(Q8.float() * dq, K8.float() * dk, V8.float() * dv, scale=scale, seq_lens_kv=seq_lens_kv, return_stats=True, **ref_kw)
        return _RunWithStatsResult(Ob, o_ref, amax_o.item(), o_ref.abs().max().item(), lse.squeeze(-1), lse_ref)
    o_ref = _ref(Q8.float() * dq, K8.float() * dk, V8.float() * dv, scale=scale, seq_lens_kv=seq_lens_kv, **ref_kw)
    return _RunResult(Ob, o_ref, amax_o.item(), o_ref.abs().max().item())


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


def _half_atol(in_key):
    return 7e-2 if in_key == "e5m2" else 5e-2


def _check(out, o_ref, out_dt, in_key, amax_o, amax_o_ref):
    atol = _half_atol(in_key)
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


def _check_fp8_strided_stats(d_qk, d_v, in_key):
    if torch.cuda.get_device_capability() == (10, 7) and d_qk != 128:
        pytest.skip("SM107 per-tensor FP8 supports only d128")
    kwargs = dict(
        B=2,
        H_q=4,
        H_kv=2,
        S_q=128,
        S_kv=128,
        in_key=in_key,
        out_dt=torch.float16,
        scale=1.0 / math.sqrt(d_qk),
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=d_qk,
        d_v=d_v,
        return_lse=True,
    )
    torch.manual_seed(59)
    contiguous = _run(**kwargs, stats_layout="contiguous")
    torch.manual_seed(59)
    strided = _run(**kwargs, stats_layout="strided")
    _check(strided.output, strided.reference, torch.float16, in_key, strided.amax, strided.reference_amax)
    torch.testing.assert_close(strided.stats, contiguous.stats, rtol=0, atol=0)
    torch.testing.assert_close(strided.stats, strided.reference_stats, rtol=3e-2, atol=_half_atol(in_key))


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_fp8_strided_stats():
    """The per-tensor FP8 L0 flavor preserves dense Stats strides."""
    _check_fp8_strided_stats(128, 128, "e4m3")


@pytest.mark.L1
@pytest.mark.parametrize(
    ("d_qk", "d_v", "in_key"),
    [(128, 128, "e5m2"), (192, 128, "e4m3"), (192, 128, "e5m2")],
    ids=["d128_e5m2", "d192_d128_e4m3", "d192_d128_e5m2"],
)
@torch_fork_set_rng(seed=59)
def test_fp8_strided_stats_other_flavors(d_qk, d_v, in_key):
    """The remaining per-tensor FP8 flavors preserve dense Stats strides."""
    _check_fp8_strided_stats(d_qk, d_v, in_key)


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


@_skip_on_rubin
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


@_skip_on_rubin
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


@_skip_on_rubin
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
@pytest.mark.parametrize("mask", ["none", "causal"])
@pytest.mark.parametrize("dims", [(80, 80), (96, 64)], ids=["d80", "d96_d64"])
@torch_fork_set_rng(seed=0)
def test_fp8_head_dim_envelope(dims, mask):
    """Per-tensor FP8 serves the dense d<=128 head-dim ENVELOPE on the d128
    row (TMA zero-padding — exact in FP8, arch-independent since the descales
    are scalars): the ViT d=72-in-80 contract's landing zone. Head dims must
    be multiples of 16 (TMA 16-byte global-stride rule at 1 byte/elem)."""
    d_qk, d_v = dims
    scale = 1.0 / math.sqrt(d_qk)
    out, o_ref, a_o, a_o_ref = _run(
        2,
        4,
        4,
        384,
        384,
        "e4m3",
        torch.bfloat16,
        scale=scale,
        sdpa_kwargs=_MASKS[mask],
        d_qk=d_qk,
        d_v=d_v,
    )
    _check(out, o_ref, torch.bfloat16, "e4m3", a_o, a_o_ref)


@pytest.mark.L0
@_skip_on_rubin
@torch_fork_set_rng(seed=0)
def test_fp8_head_dim_envelope_d192_flavor():
    """The d192xd128 flavor serves its envelope too: (160, 96) rides the
    (192, 128) kernel with TMA zero-padding on both sides.

    Direct adapter API: the pygraph ``sdpa_fp8`` node validator still bounds
    the GRAPH route at d_qk <= 128 (%16) / exact (192, 128) — a pre-FE-OSS
    shape whitelist in the C++ frontend — so this region of the envelope is
    reachable through the standalone API only until that validation is
    relaxed to describe rather than judge."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, S, d_qk, d_v = 2, 4, 384, 160, 96
    dev = "cuda"
    Q8 = (torch.randn(B, H, S, d_qk, device=dev) * 0.5).to(torch.float8_e4m3fn)
    K8 = (torch.randn(B, H, S, d_qk, device=dev) * 0.5).to(torch.float8_e4m3fn)
    V8 = (torch.randn(B, H, S, d_v, device=dev) * 0.5).to(torch.float8_e4m3fn)
    out = torch.empty(B, H, S, d_v, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(B, H, S, device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d_qk)

    api = SdpaFwdDslSm100(sample_q=Q8, sample_k=K8, sample_v=V8, sample_o=out, sample_lse=lse, scale_softmax=scale, pertensor_fp8=True)
    assert api.check_support()
    api.compile()
    api.execute(q_tensor=Q8, k_tensor=K8, v_tensor=V8, o_tensor=out, lse_tensor=lse)
    torch.cuda.synchronize()

    o_ref = torch.softmax(Q8.float() @ K8.float().transpose(-1, -2) * scale, dim=-1) @ V8.float()
    err = (out.float() - o_ref).abs().max().item()
    assert err <= 5e-2 + 0.05 * o_ref.abs().max().item(), f"(160,96) envelope mismatch: max err {err}"
    assert not torch.isnan(out.float()).any()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_head_dim_envelope_padded():
    """The envelope composed with the KV padding mask — the ViT production
    shape (non-causal, seqlen padded up to a tile multiple, d=80)."""
    out, o_ref, a_o, a_o_ref = _run(
        2,
        4,
        4,
        384,
        384,
        "e4m3",
        torch.bfloat16,
        scale=1.0 / math.sqrt(72),
        sdpa_kwargs={},
        seq_lens_kv=[384, 250],
        d_qk=80,
        d_v=80,
    )
    _check(out, o_ref, torch.bfloat16, "e4m3", a_o, a_o_ref)


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


# --- PackGQA: TILE_M/G tokens x G query heads per tile -------
@pytest.mark.L0
@pytest.mark.parametrize(
    "h_q,h_kv",
    [(8, 4), (8, 2), (8, 1), (16, 1)],
    ids=["g2", "g4", "g8_mqa", "g16_mqa"],
)
@torch_fork_set_rng(seed=0)
def test_fp8_pack_gqa_ratios(h_q, h_kv):
    """Packed plans across GQA ratios, causal, tile-unaligned s_q, LSE checked."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_ref, lse_v, lse_ref = _run(
        2, h_q, h_kv, 40, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=True, return_lse=True
    )
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_ref)
    torch.testing.assert_close(lse_v, lse_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [8, 64, 100], ids=["subspan", "exact_span", "tail"])
@torch_fork_set_rng(seed=0)
def test_fp8_pack_gqa_tiles(s_q):
    """Packed tile-geometry edges at G=8 (s_q*G below / at / past the 512-row CGA span)."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_ref, lse_v, lse_ref = _run(
        1, 64, 8, s_q, 256, "e4m3", torch.bfloat16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=True, return_lse=True
    )
    _check(out, o_ref, torch.bfloat16, "e4m3", a_o, a_ref)
    torch.testing.assert_close(lse_v, lse_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("out_dt", [torch.float16, torch.float8_e4m3fn], ids=["f16_out", "e4m3_out"])
@pytest.mark.parametrize(
    "mask",
    ["none_padded", "causal", "causal_br", "swa", "sink_causal"],
)
@torch_fork_set_rng(seed=0)
def test_fp8_pack_gqa_features(mask, out_dt):
    """Packed plans x the fp8 mask/sink envelope x both output dtypes."""
    scale = 1.0 / math.sqrt(128)
    kw = dict()
    sink = None
    seq_lens_kv = None
    if mask == "none_padded":
        seq_lens_kv = [180, 240]
    elif mask == "causal":
        kw = dict(use_causal_mask=True)
    elif mask == "causal_br":
        kw = dict(use_causal_mask_bottom_right=True)
    elif mask == "swa":
        kw = dict(use_causal_mask=True, left_bound=17)
    elif mask == "sink_causal":
        kw = dict(use_causal_mask=True)
        sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    out, o_ref, a_o, a_ref, lse_v, lse_ref = _run(
        2, 8, 2, 40, 256, "e4m3", out_dt, scale=scale, sdpa_kwargs=kw, sink=sink, seq_lens_kv=seq_lens_kv, pack_gqa=True, return_lse=True
    )
    _check(out, o_ref, out_dt, "e4m3", a_o, a_ref)
    torch.testing.assert_close(lse_v, lse_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_pack_gqa_e5m2():
    """Packed e5m2 input path."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_ref = _run(2, 8, 2, 40, 256, "e5m2", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=True)
    _check(out, o_ref, torch.float16, "e5m2", a_o, a_ref)


@_skip_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("h_q,h_kv", [(8, 4), (8, 2), (16, 2)], ids=["g2", "g4", "g8"])
@torch_fork_set_rng(seed=0)
def test_fp8_pack_gqa_d192_d128_ratios(h_q, h_kv):
    """Packed d192xd128 flavor across GQA ratios, causal, tile-unaligned s_q."""
    scale = 1.0 / math.sqrt(192)
    out, o_ref, a_o, a_ref, lse_v, lse_ref = _run(
        2,
        h_q,
        h_kv,
        40,
        256,
        "e4m3",
        torch.float16,
        scale=scale,
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=192,
        d_v=128,
        pack_gqa=True,
        return_lse=True,
    )
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_ref)
    torch.testing.assert_close(lse_v, lse_ref, atol=5e-2, rtol=3e-2)


@_skip_on_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_pack_gqa_d192_d128_grouped_lpt():
    """Packed d192xd128 through the grouped-LPT decoder: the head-group and
    reverse-row (lpt_q_tiles) knobs must see the PACKED launch geometry —
    B * (h_q/G) = 32 selects the group-of-8 decoder, and s_q*G spans two
    reverse-row CGA tiles."""
    scale = 1.0 / math.sqrt(192)
    out, o_ref, a_o, a_ref, lse_v, lse_ref = _run(
        8,
        8,
        4,
        300,
        512,
        "e4m3",
        torch.bfloat16,
        scale=scale,
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=192,
        d_v=128,
        pack_gqa=True,
        return_lse=True,
    )
    _check(out, o_ref, torch.bfloat16, "e4m3", a_o, a_ref)
    torch.testing.assert_close(lse_v, lse_ref, atol=5e-2, rtol=3e-2)


@_skip_on_rubin
@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_pack_gqa_d192_d128_e5m2_sink():
    """Packed d192xd128 E5M2 + sink: the sink logit seeds the softmax per
    ROW, so under packing the seed is lane-varying (row's true query head)."""
    scale = 1.0 / math.sqrt(192)
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    out, o_ref, a_o, a_ref = _run(
        2, 8, 2, 40, 256, "e5m2", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), sink=sink, d_qk=192, d_v=128, pack_gqa=True
    )
    _check(out, o_ref, torch.float16, "e5m2", a_o, a_ref)


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


def _run_thd(seq_lens_q, seq_lens_kv, H_q, H_kv, in_key, *, scale, causal=False, sink=None, stats=False, cu_lens=False):
    """THD/varlen: packed [T,H,D] Q/K/V/O + per-operand ragged_offset + per-batch
    lengths (or their cu prefix-sum form).

    Per-tensor quantization of the packed tokens (one scalar per operand —
    identical semantics to the dense per-tensor path). Returns the packed frost
    O, the packed fp32 reference, the (amax_o, amax_o_ref) pair, and — with
    ``stats`` — the packed token-major (T, H) LSE next to its natural-log
    reference."""
    import cudnn

    dev = "cuda"
    D = 128
    B = len(seq_lens_q)
    S_max_q, S_max_kv = max(seq_lens_q), max(seq_lens_kv)
    T_q, T_kv = sum(seq_lens_q), sum(seq_lens_kv)

    def _cu(sl):
        c = [0]
        for s in sl:
            c.append(c[-1] + s)
        return c

    cu_q, cu_k = _cu(seq_lens_q), _cu(seq_lens_kv)

    q_pk = torch.randn(T_q, H_q, D, device=dev) * 0.5
    k_pk = torch.randn(T_kv, H_kv, D, device=dev) * 0.5
    v_pk = torch.randn(T_kv, H_kv, D, device=dev) * 0.5
    q8, dq = _quant(q_pk, in_key)
    k8, dk = _quant(k_pk, in_key)
    v8, dv = _quant(v_pk, in_key)

    def _dense_buf(packed, s_max, h, dt):
        # Dense-capacity storage; packed tokens in the leading elements (THD
        # contract). The capacity tail is NaN-POISONED (test_mhas_v2 parity):
        # the last sequence's KV tile steps past the packed total, and those
        # tail loads must land as zeros through the setup kernel's
        # packed-total-clamped K/V descriptors — a leaked NaN would wipe the
        # tile via BMM2's P·V (0 · NaN == NaN).
        stride = (s_max * h * D, D, h * D, 1)
        stor = torch.full((B * s_max * h * D,), float("nan"), device=dev, dtype=torch.float32).to(dt)
        stor[: packed.numel()] = packed.reshape(-1)
        return stor, stor.as_strided((B, h, s_max, D), stride), stride

    _, q_gpu, stride_q = _dense_buf(q8, S_max_q, H_q, q8.dtype)
    _, k_gpu, stride_kv = _dense_buf(k8, S_max_kv, H_kv, k8.dtype)
    _, v_gpu, _ = _dense_buf(v8, S_max_kv, H_kv, v8.dtype)
    o_stor = torch.zeros(B * S_max_q * H_q * D, device=dev, dtype=torch.float16)
    o_gpu = o_stor.as_strided((B, H_q, S_max_q, D), stride_q)
    amax_o = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    slq = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    cuq_t = torch.tensor(cu_q, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    cuk_t = torch.tensor(cu_k, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    ro_q = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * D).view(B + 1, 1, 1, 1)
    ro_k = (torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * D).view(B + 1, 1, 1, 1)

    def sc(val):
        return torch.tensor([[[[val]]]], dtype=torch.float32, device=dev)

    io = getattr(cudnn.data_type, _CUDNN_ITYPE[in_key])
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H_q, S_max_q, D], stride=list(stride_q), data_type=io, name="q")
    tk = g.tensor(dim=[B, H_kv, S_max_kv, D], stride=list(stride_kv), data_type=io, name="k")
    tv = g.tensor(dim=[B, H_kv, S_max_kv, D], stride=list(stride_kv), data_type=io, name="v")
    sq_h = g.tensor_like(cuq_t if cu_lens else slq)
    skv_h = g.tensor_like(cuk_t if cu_lens else slk)
    qro, kro, vro, oro = (g.tensor_like(ro_q) for _ in range(4))
    tq.set_ragged_offset(qro)
    tk.set_ragged_offset(kro)
    tv.set_ragged_offset(vro)

    def _stns():
        return g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

    dqn, dkn, dvn, dsn, ssn, son = (_stns() for _ in range(6))
    kw = dict(
        q=tq,
        k=tk,
        v=tv,
        descale_q=dqn,
        descale_k=dkn,
        descale_v=dvn,
        descale_s=dsn,
        scale_s=ssn,
        scale_o=son,
        attn_scale=scale,
        generate_stats=stats,
        use_padding_mask=True,
    )
    if cu_lens:
        kw.update(cu_seq_len_q=sq_h, cu_seq_len_kv=skv_h)
    else:
        kw.update(seq_len_q=sq_h, seq_len_kv=skv_h)
    if causal:
        kw["use_causal_mask"] = True
    vp = {
        tq: q_gpu,
        tk: k_gpu,
        tv: v_gpu,
        dqn: sc(dq),
        dkn: sc(dk),
        dvn: sc(dv),
        dsn: sc(1.0),
        ssn: sc(1.0),
        son: sc(1.0),
        sq_h: (cuq_t if cu_lens else slq),
        skv_h: (cuk_t if cu_lens else slk),
        qro: ro_q,
        kro: ro_k,
        vro: ro_k,
        oro: ro_q,
    }
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    o, stats_t, _amx_s_unused, amx_o = g.sdpa_fp8(**kw)  # Amax_S: not requested (engines decline graphs that declare it)
    o.set_output(True).set_dim([B, H_q, S_max_q, D]).set_stride(list(stride_q)).set_data_type(cudnn.data_type.HALF)
    o.set_ragged_offset(oro)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    stats_stor = None
    if stats:
        # Ragged Stats, packed token-major TH1 ([t, h]; offsets = cu_q * h_q).
        stats_stor = torch.zeros(B * S_max_q * H_q, dtype=torch.float32, device=dev)
        stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        stats_t.set_dim((B, H_q, S_max_q, 1)).set_stride((S_max_q * H_q, 1, H_q, 1))
        stats_ro_t = (ro_q.flatten() // D).view(B + 1, 1, 1, 1).contiguous()
        stats_ro = g.tensor_like(stats_ro_t, name="stats_ro")
        stats_t.set_ragged_offset(stats_ro)
        vp[stats_ro] = stats_ro_t
        vp[stats_t] = stats_stor

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(arch=_D128_ARCH, fp8=True))
    g.check_support()
    g.build_plans()
    vp.update({o: o_gpu, amx_o: amax_o})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    o_ref = torch.zeros(T_q, H_q, D, device=dev, dtype=torch.float32)
    lse_ref = torch.zeros(T_q, H_q, dtype=torch.float32, device=dev)
    for b in range(B):
        if cu_q[b + 1] == cu_q[b]:
            continue
        if cu_k[b + 1] == cu_k[b]:
            # Zero-length KV: every Q row of the sequence is dead — O := 0
            # (o_ref is pre-zeroed) and, sink-less, LSE := -inf.
            if stats:
                lse_ref[cu_q[b] : cu_q[b + 1]] = sink.flatten().to(lse_ref) if sink is not None else float("-inf")
            continue
        qb = (q8[cu_q[b] : cu_q[b + 1]].float() * dq).permute(1, 0, 2).unsqueeze(0)
        kb = (k8[cu_k[b] : cu_k[b + 1]].float() * dk).permute(1, 0, 2).unsqueeze(0)
        vb = (v8[cu_k[b] : cu_k[b + 1]].float() * dv).permute(1, 0, 2).unsqueeze(0)
        ref_kw = dict(is_causal=True) if causal else {}
        if sink is not None:
            ref_kw["sinks"] = sink.flatten()
        ob = _ref(qb, kb, vb, scale=scale, **ref_kw)
        o_ref[cu_q[b] : cu_q[b + 1]] = ob.squeeze(0).permute(1, 0, 2)
        if stats:
            lse_ref[cu_q[b] : cu_q[b + 1]] = _ref_lse(qb, kb, scale=scale, causal=causal, sinks=(sink.flatten() if sink is not None else None)).squeeze(0).T

    o_out = o_stor[: T_q * H_q * D].reshape(T_q, H_q, D)
    lse_out = stats_stor[: T_q * H_q].reshape(T_q, H_q) if stats else None
    return o_out, o_ref, amax_o.item(), o_ref.abs().max().item(), lse_out, (lse_ref if stats else None)


def _ref_lse(qd, kd, *, scale, causal, sinks=None):
    """Natural-log LSE reference over per-sequence scores, [1, H, S_q]."""
    _, h_q, s_q, _ = qd.shape
    _, h_kv, s_kv, _ = kd.shape
    dev = qd.device
    k_e = kd.repeat_interleave(h_q // h_kv, dim=1)
    scores = torch.matmul(qd, k_e.transpose(-1, -2)) * scale
    if causal:
        i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
        j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
        scores = scores.masked_fill(j > i, float("-inf"))
    if sinks is not None:
        col = sinks.view(1, h_q, 1, 1).float().expand(1, h_q, s_q, 1).to(dev)
        scores = torch.cat([scores, col], dim=-1)
    return torch.logsumexp(scores, dim=-1)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_fp8_thd(in_key, causal):
    """THD/varlen self-attention: two packed sequences of unequal, tile-ragged length."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref, _, _ = _run_thd([200, 150], [200, 150], 8, 8, in_key, scale=scale, causal=causal)
    _check(out, o_ref, torch.float16, in_key, a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_thd_cross_gqa():
    """THD cross-attention (unequal packed Q and K/V totals) with GQA heads."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref, _, _ = _run_thd([64, 200], [256, 128], 8, 2, "e4m3", scale=scale)
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_thd_sink():
    """THD causal + attention sink."""
    scale = 1.0 / math.sqrt(128)
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    out, o_ref, a_o, a_o_ref, _, _ = _run_thd([200, 150], [200, 150], 8, 8, "e4m3", scale=scale, causal=True, sink=sink)
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_thd_stats():
    """THD + generate_stats: the ragged token-major TH1 LSE is written next to O."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref, lse, lse_ref = _run_thd([200, 150], [200, 150], 8, 8, "e4m3", scale=scale, causal=True, stats=True)
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_thd_zero_len_kv():
    """Zero-length Q and KV sequences (test_mhas_v2 ragged parity, e.g.
    seq_len_q=[126, 0, 60] / seq_len_kv=[0, 83, 77]): the zero-KV sequence's
    rows are dead — the epilogue must come back O := 0 / LSE := -inf, not the
    unwritten O TMEM (garbage survives `* inv_sum(=0)` when it is NaN)."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref, lse, lse_ref = _run_thd([126, 40, 60], [0, 83, 77], 8, 8, "e4m3", scale=scale, stats=True)
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2, equal_nan=False)
    out, o_ref, a_o, a_o_ref, _, _ = _run_thd([126, 0, 60], [0, 83, 77], 8, 8, "e5m2", scale=scale)
    _check(out, o_ref, torch.float16, "e5m2", a_o, a_o_ref)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_thd_cu_seq_len():
    """THD via the (B+1,) cu_seq_len prefix-sum length form."""
    scale = 1.0 / math.sqrt(128)
    out, o_ref, a_o, a_o_ref, _, _ = _run_thd([200, 150], [180, 120], 8, 8, "e4m3", scale=scale, cu_lens=True)
    _check(out, o_ref, torch.float16, "e4m3", a_o, a_o_ref)


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
