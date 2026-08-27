# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the FROST SM120 DSL per-tensor FP8 SDPA-forward engine.

Drives ``graph.sdpa_fp8`` (FP8 E4M3 Q/K/V + scalar per-tensor descales) routed to
the ``sdpa_fwd_prefill_sm120_fp8`` engine, and validates O against an fp32-dequant
reference. ``Amax_O`` is produced in-kernel (bitcast-int32 atomicMax over the
pre-cast fp32 values) and checked; there is no Amax_S output (graphs that
declare one are declined and route to the native backend).

SM120 envelope (see engines._sm120_fp8_spec): E4M3/E5M2 in, FP16/BF16/FP8
out (fp8 O via a direct quantizing store, Scale_O applied pre-cast), head
TILES any multiple of 32 up to 256 with the QK^T and P@V sides independent,
actual head dims any multiple of 16 up to the tile via TMA zero-padding
(what graphs can reach is further gated by the C++ sdpa_fp8 node:
d_qk <= 128 x d_v <= 128 plus the (192, 128) MLA pair), causal /
bottom-right / SWA / right-band / KV-padding masks, per-batch seq_len_q trim,
ragged S_kv without a padding mask (skv_tile=0), dense_flex layouts, THD
(ragged) with token- or head-major Stats, and attention sinks
(sink-extended denominator, no sink column in P).

Requires: SM120/SM121 (Blackwell GeForce), cutlass-dsl. Skips otherwise.
"""

import math
from typing import NamedTuple

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import make_dense_stats, requires_blackwell_geforce, requires_dsl, select_engine as _select_engine, offers_engine

pytestmark = [requires_blackwell_geforce, requires_dsl]

_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
_FP8_MAX = {torch.float8_e4m3fn: _E4M3_MAX, torch.float8_e5m2: _E5M2_MAX}


class _RunResult(NamedTuple):
    output: torch.Tensor
    reference: torch.Tensor
    amax: float
    reference_amax: float
    stats: torch.Tensor
    reference_stats: torch.Tensor


def _quant(x, dtype=torch.float8_e4m3fn):
    fmax = _FP8_MAX[dtype]
    dq = (x.abs().amax().clamp_min(1e-8) / fmax).item()
    return (x / dq).clamp(-fmax, fmax).to(dtype), dq


def _ref(qd, kd, vd, *, scale, is_causal=False, bottom_right=False, swa_window=None, right_bound=0, seq_lens_kv=None, seq_lens_q=None, sinks=None):
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
        lim = (i + (s_kv - s_q) if bottom_right else i) + right_bound
        masked = masked | (j > lim)
    if swa_window is not None:
        masked = masked | (j < i - swa_window)
    if seq_lens_kv is not None:
        slk = torch.as_tensor(seq_lens_kv, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        masked = masked | (j >= slk)
    scores = scores.masked_fill(masked, float("-inf"))
    if sinks is not None:
        sink_col = sinks.flatten().float().view(1, h_q, 1, 1).expand(b, h_q, s_q, 1)
        scores = torch.cat([scores, sink_col], dim=-1)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)[..., :s_kv]
    out = torch.matmul(probs, v_e)
    lse = torch.logsumexp(scores, dim=-1)
    if seq_lens_q is not None:
        # cuDNN dense padded-Q trim: rows at/past seq_len_q[b] write O := 0 /
        # LSE := -inf (softmax of a fully -inf row is NaN in torch; the trim
        # replaces it).
        slq = torch.as_tensor(seq_lens_q, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        row_dead = i >= slq
        out = torch.where(row_dead, torch.zeros((), device=dev), out)
        lse = lse.masked_fill(row_dead.squeeze(-1), float("-inf"))
    # Fully-masked rows are -inf in logsumexp; the kernel writes -inf too.
    return out, lse


def _run(
    B,
    H_q,
    H_kv,
    S_q,
    S_kv,
    *,
    scale,
    sdpa_kwargs,
    seq_lens_kv=None,
    seq_lens_q=None,
    tiles=None,
    pack_gqa=None,
    s_descale_gain=1.0,
    sync_debug=False,
    D=128,
    D_v=None,
    io_dtype=torch.float8_e4m3fn,
    o_dtype=torch.float16,
    so_val=1.0,
    layout="bshd",
    sinks=None,
    stats_layout="contiguous",
):
    import cudnn

    dev = "cuda"
    D_v = D if D_v is None else D_v
    Qf = torch.randn(B, H_q, S_q, D, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S_kv, D, device=dev) * 0.5
    Vf = torch.randn(B, H_kv, S_kv, D_v, device=dev) * 0.5
    Q8, dq = _quant(Qf, io_dtype)
    K8, dk = _quant(Kf, io_dtype)
    V8, dv = _quant(Vf, io_dtype)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    if layout == "bshd":
        Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
        Ob = torch.empty(B, S_q, H_q, D_v, device=dev, dtype=o_dtype).transpose(1, 2)
    elif layout == "bhsd":
        # BHSD-contiguous (torch's natural layout) -- NOT BSHD-physical; the
        # dense_flex relaxation normalizes it via one gather / scatter copy.
        Qb, Kb, Vb = Q8.contiguous(), K8.contiguous(), V8.contiguous()
        Ob = torch.empty(B, H_q, S_q, D_v, device=dev, dtype=o_dtype)
    elif layout == "padded_s":
        # Padded S stride: rows allocated with slack between sequences.
        def pad_s(x8, s):
            buf = torch.zeros(x8.shape[0], x8.shape[1], s + 24, x8.shape[3], device=dev, dtype=x8.dtype)
            buf[:, :, :s, :] = x8
            return buf[:, :, :s, :]

        Qb, Kb, Vb = pad_s(Q8, S_q), pad_s(K8, S_kv), pad_s(V8, S_kv)
        Ob = torch.zeros(B, H_q, S_q + 24, D_v, device=dev, dtype=o_dtype)[:, :, :S_q, :]
    else:
        raise ValueError(f"unknown layout {layout!r}")
    lse = make_dense_stats(B, H_q, S_q, stats_layout)
    amax_o = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    def sc(val):
        return torch.tensor([[[[val]]]], dtype=torch.float32, device=dev)

    # Scale_S maps P (in (0,1] after the row-max subtraction) onto the io
    # dtype's range; Descale_S undoes it in the epilogue. Non-unit on
    # purpose -- unit values would not exercise the scaling at all.
    s_scale = _FP8_MAX[io_dtype]
    dqt, dkt, dvt, dst, sst, sot = sc(dq), sc(dk), sc(dv), sc(s_descale_gain / s_scale), sc(s_scale), sc(so_val)

    io_cudnn = {torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3, torch.float8_e5m2: cudnn.data_type.FP8_E5M2}[io_dtype]
    g = cudnn.pygraph(io_data_type=io_cudnn, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor_like(Qb)
    k = g.tensor_like(Kb)
    v = g.tensor_like(Vb)

    def _stns():
        return g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

    dqn, dkn, dvn, dsn, ssn, son = (_stns() for _ in range(6))
    kw = dict(q=q, k=k, v=v, descale_q=dqn, descale_k=dkn, descale_v=dvn, descale_s=dsn, scale_s=ssn, scale_o=son, attn_scale=scale, generate_stats=True)
    vp = {q: Qb, k: Kb, v: Vb, dqn: dqt, dkn: dkt, dvn: dvt, dsn: dst, ssn: sst, son: sot}
    if seq_lens_q is not None and seq_lens_kv is None:
        seq_lens_kv = [S_kv] * B
    if seq_lens_kv is not None:
        slq = (
            torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
            if seq_lens_q is not None
            else torch.full((B, 1, 1, 1), S_q, dtype=torch.int32, device=dev)
        )
        slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        sq_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        skv_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_q=sq_h, seq_len_kv=skv_h)
        vp[sq_h] = slq
        vp[skv_h] = slk
    if sinks is not None:
        sink_t = g.tensor_like(sinks, name="sink")
        kw["sink_token"] = sink_t
        vp[sink_t] = sinks
    kw.update(sdpa_kwargs)
    o, stats, _amx_s_unused, amx_o = g.sdpa_fp8(**kw)  # Amax_S: not requested (FROST does not produce it)
    o_cudnn = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
    }[o_dtype]
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(o_cudnn)
    stats.set_output(True).set_dim([B, H_q, S_q, 1]).set_stride(list(lse.stride())).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(arch="sm120", fp8=True), tiles=tiles, pack_gqa=pack_gqa)
    g.check_support()
    g.build_plans()
    vp.update({o: Ob, stats: lse, amx_o: amax_o})
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
    o_ref, lse_ref = _ref(Q8.float() * dq, K8.float() * dk, V8.float() * dv, scale=scale, seq_lens_kv=seq_lens_kv, seq_lens_q=seq_lens_q, sinks=sinks, **ref_kw)
    # O carries Scale_O; Amax_O is the pre-scale amax (the kernel divides it
    # back out), so both compare against the unscaled reference.
    return _RunResult(Ob.float() / so_val, o_ref, amax_o.item(), o_ref.abs().max().item(), lse.squeeze(-1), lse_ref)


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
    rb = sdpa_kwargs.get("right_bound")
    if rb is not None:
        # A band graph is NON-causal in cuDNN's vocabulary; the kernel lowers
        # it as the causal machinery with the diagonal translated right by R.
        out["is_causal"] = True
        out["right_bound"] = rb
    return out


def _check(out, o_ref, amax_o, amax_o_ref, lse=None, lse_ref=None, tol_o=5e-2):
    diff = (out.float() - o_ref).abs().max().item()
    assert diff <= tol_o, f"max|O-ref|={diff:.4f} > {tol_o}"
    assert abs(amax_o - amax_o_ref) <= 0.03, f"amax_o {amax_o:.4f} vs ref {amax_o_ref:.4f}"
    if lse is not None:
        finite = torch.isfinite(lse_ref)
        assert torch.equal(torch.isfinite(lse), finite), "LSE -inf pattern differs from the reference"
        d = (lse[finite] - lse_ref[finite]).abs().max().item() if finite.any() else 0.0
        assert d <= 3e-2, f"max|LSE-ref|={d:.4f} > 0.03"


# --- PackGQA: q_tile/G tokens x G query heads per tile -------
@pytest.mark.L0
@pytest.mark.parametrize(
    "h_q,h_kv",
    [(8, 4), (8, 2), (8, 1), (16, 1)],
    ids=["g2", "g4", "g8_mqa", "g16_mqa"],
)
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_pack_gqa_ratios(h_q, h_kv):
    """Packed plans across GQA ratios (incl. MQA)."""
    out, o_ref, a_o, a_ref, lse, lse_ref = _run(2, h_q, h_kv, 40, 256, scale=1.0 / math.sqrt(128), sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=True)
    _check(out, o_ref, a_o, a_ref, lse, lse_ref)


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [4, 16, 25], ids=["subspan", "exact_span", "tail"])
@torch_fork_set_rng(seed=1)
def test_fp8_sm120_pack_gqa_tiles(s_q):
    """Packed tile-geometry edges at G=8, q_tile=128 (token span 16/tile)."""
    out, o_ref, a_o, a_ref, lse, lse_ref = _run(
        1, 64, 8, s_q, 256, scale=1.0 / math.sqrt(128), sdpa_kwargs=dict(use_causal_mask=True), tiles=(128, 128), pack_gqa=True
    )
    _check(out, o_ref, a_o, a_ref, lse, lse_ref)


@pytest.mark.L0
@pytest.mark.parametrize("o_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16_out", "e4m3_out"])
@pytest.mark.parametrize("mask", ["none_padded", "causal", "causal_br", "swa", "sink_causal"])
@torch_fork_set_rng(seed=2)
def test_fp8_sm120_pack_gqa_features(mask, o_dtype):
    """Packed plans x the fp8 mask/sink envelope x both output-dtype epilogues."""
    kw = dict()
    sinks = None
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
        sinks = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    out, o_ref, a_o, a_ref, lse, lse_ref = _run(
        2, 8, 2, 40, 256, scale=1.0 / math.sqrt(128), sdpa_kwargs=kw, seq_lens_kv=seq_lens_kv, sinks=sinks, o_dtype=o_dtype, pack_gqa=True
    )
    tol_o = 5e-2
    if o_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        floor = (o_ref - o_ref.to(o_dtype).float()).abs().max().item()
        tol_o = max(tol_o, 3 * floor)
    _check(out, o_ref, a_o, a_ref, lse, lse_ref, tol_o=tol_o)


@pytest.mark.L1
@torch_fork_set_rng(seed=3)
def test_fp8_sm120_pack_gqa_tile64():
    """Packed at q_tile=64 (G must divide the smaller tile: 8/2 -> G=4)."""
    out, o_ref, a_o, a_ref, lse, lse_ref = _run(
        2, 8, 2, 24, 192, scale=1.0 / math.sqrt(128), sdpa_kwargs=dict(use_causal_mask=True), tiles=(64, 64), pack_gqa=True
    )
    _check(out, o_ref, a_o, a_ref, lse, lse_ref)


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
    res = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=_MASKS[mask])
    _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [1, 2], ids=["mqa", "gqa"])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_grouped_query(h_kv):
    """H_kv=1 is MQA (one shared KV head), H_kv=2 is GQA; both take the
    repeat_interleave path in the kernel's head mapping."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 8, h_kv, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(*res)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_bottom_right_rectangular():
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 8, 8, 128, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask_bottom_right=True))
    _check(*res)


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_multi_tile_long_seq():
    # 1k x 1k exercises the multi-KV-tile online-softmax rescale path.
    scale = 1.0 / math.sqrt(128)
    res = _run(1, 4, 4, 1024, 1024, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_padding(causal):
    # KV padding: batch 0 uses all 256 KV cols, batch 1 only 192 (partial tile).
    scale = 1.0 / math.sqrt(128)
    sk = dict(use_causal_mask=True) if causal else {}
    res = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=sk, seq_lens_kv=[256, 192])
    _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "none_dense", "causal"])
@pytest.mark.parametrize("s", [96, 97, 110], ids=lambda s: f"s{s}")
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_single_kv_tile(mask, s):
    """Single-KV-tile shapes (s_kv <= kv_tile), repeated.

    "none" reaches the shape through the padded route (per-batch KV length);
    "none_dense" through the dense route that skv_tile=0 opened -- both bound
    the KV trip count at compile time, i.e. both are collapse-guard
    populations."""
    scale = 1.0 / math.sqrt(128)
    seq_lens_kv = [s] if mask == "none" else None
    kwargs = {**_MASKS, "none_dense": _MASKS["none"]}[mask]
    for _ in range(3):
        res = _run(1, 4, 4, s, s, scale=scale, sdpa_kwargs=kwargs, seq_lens_kv=seq_lens_kv)
        _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["bhsd", "padded_s"])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_dense_flex_layout(layout):
    """Non-BSHD dense layouts (the dense_flex relaxation): BHSD-contiguous
    (torch's natural layout) and padded S strides. The shared adapter
    normalizes to the kernel's compact BSHD via one gather / scatter copy."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 4, 4, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), layout=layout)
    _check(*res)


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_fp8_sm120_strided_stats():
    """Dense LSE is written directly through a permuted, gapped layout."""

    scale = 1.0 / math.sqrt(128)
    kwargs = dict(B=2, H_q=4, H_kv=2, S_q=128, S_kv=128, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    torch.manual_seed(59)
    contiguous = _run(**kwargs)
    torch.manual_seed(59)
    strided = _run(**kwargs, stats_layout="strided")
    _check(*strided)
    torch.testing.assert_close(strided.stats, contiguous.stats, atol=0, rtol=0)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_dense_flex_offered():
    """The dense_flex capability is a DECLARED offer: a BHSD-contiguous (non
    BSHD-physical) graph must be claimed by the row, not merely survive
    engine selection. Guards the ``layouts={"bshd", "dense_flex"}`` entry."""
    import cudnn

    assert _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, layout="bhsd")


_O_TOL = {torch.bfloat16: 5e-2, torch.float8_e4m3fn: 1.5e-1, torch.float8_e5m2: 3.5e-1}


@pytest.mark.L0
@pytest.mark.parametrize("o_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16])
@pytest.mark.parametrize("mask", ["none", "causal"])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_o_dtype(o_dtype, mask):
    """Non-FP16 O dtypes. fp8 O exercises the direct quantizing store with a
    NON-UNIT Scale_O (applied before the cast; Amax_O stays the pre-scale
    fp32 amax); bf16 rides the staging epilogue. Tolerances widen with the
    output mantissa (e5m2 keeps 2 bits)."""
    scale = 1.0 / math.sqrt(128)
    kwargs = dict(use_causal_mask=True) if mask == "causal" else {}
    res = _run(2, 4, 2, 256, 256, scale=scale, sdpa_kwargs=kwargs, o_dtype=o_dtype, so_val=2.0)
    _check(*res, tol_o=_O_TOL[o_dtype])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_fp8_out_mixed_in():
    """E5M2 in with E4M3 out: the MMA tag follows the input, the quantizing
    store the independent O dtype."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 2, 2, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), io_dtype=torch.float8_e5m2, o_dtype=torch.float8_e4m3fn, so_val=2.0)
    _check(*res, tol_o=_O_TOL[torch.float8_e4m3fn])


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_fp8_out_envelope():
    """fp8 O x d_envelope: the 2-byte column-pair stores clip at the actual
    head dim (pairs never straddle it -- dims are multiples of 16)."""
    scale = 1.0 / math.sqrt(80)
    res = _run(2, 4, 2, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), D=80, o_dtype=torch.float8_e4m3fn, so_val=2.0)
    _check(*res, tol_o=_O_TOL[torch.float8_e4m3fn])


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_fp8_out_dense_flex():
    """fp8 O x dense_flex: the adapter's O normalization allocates an fp8
    scratch and ``o_view.copy_(o_scratch)`` does a strided fp8 copy-back — a
    torch path with historically uneven fp8 support."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 4, 4, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), layout="bhsd", o_dtype=torch.float8_e4m3fn, so_val=2.0)
    _check(*res, tol_o=_O_TOL[torch.float8_e4m3fn])


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_fp8_out_sink():
    """fp8 O x sink: the sink rescales ``row_sum_inv`` before the direct
    quantizing store, so a sink-scaled value must still land inside
    ``cvt.rn.satfinite``'s saturation rather than corrupt the pair pack."""
    scale = 1.0 / math.sqrt(128)
    sinks = torch.randn(1, 4, 1, 1, dtype=torch.float32, device="cuda")
    res = _run(2, 4, 2, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), sinks=sinks, o_dtype=torch.float8_e4m3fn, so_val=2.0)
    _check(*res, tol_o=_O_TOL[torch.float8_e4m3fn])


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_fp8_out_seq_q_trim():
    """fp8 O x per-batch Q trim: rows at/past seq_len_q[b] zero-fill through
    the direct-store path."""
    scale = 1.0 / math.sqrt(128)
    res = _run(3, 2, 2, 256, 256, scale=scale, sdpa_kwargs={}, seq_lens_q=[256, 129, 0], o_dtype=torch.float8_e4m3fn, so_val=2.0)
    _check(*res, tol_o=_O_TOL[torch.float8_e4m3fn])


@pytest.mark.L0
@pytest.mark.parametrize("D,D_v", [(48, 48), (80, 80), (112, 112), (112, 80)])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_d_envelope(D, D_v):
    """Actual head dims narrower than the 32-granule head tiles (multiples of
    16): TMA zero-fills the pad K/V columns, the Q load and O store guards
    clip to the actual widths. Causal exercises the masked-tile path too."""
    scale = 1.0 / math.sqrt(D)
    res = _run(2, 4, 2, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), D=D, D_v=D_v)
    _check(*res)


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_d_envelope_offering():
    """Multiples of 16 are served via the envelope; other alignments decline
    (TMA 16-byte global-stride rule at 1 byte/elem)."""
    import cudnn

    assert _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, D=80)
    assert not _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, D=72)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal"])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_sink(mask):
    """Attention sink: per-Q-head logit in the softmax denominator (virtual
    column with no V row). O rescales and the LSE extends by the sink term."""
    scale = 1.0 / math.sqrt(128)
    sinks = torch.randn(1, 4, 1, 1, dtype=torch.float32, device="cuda")
    kwargs = dict(use_causal_mask=True) if mask == "causal" else {}
    res = _run(2, 4, 2, 256, 256, scale=scale, sdpa_kwargs=kwargs, sinks=sinks)
    _check(*res)


@pytest.mark.L0
@torch_fork_set_rng(seed=20)
def test_fp8_sm120_sink_dead_rows():
    """With a sink, rows with no visible key keep a finite LSE (the sink
    alone) and O := 0; rows past seq_len_q[b] still trim to -inf."""
    scale = 1.0 / math.sqrt(128)
    sinks = torch.randn(1, 4, 1, 1, dtype=torch.float32, device="cuda")
    res = _run(2, 4, 4, 128, 128, scale=scale, sdpa_kwargs={}, seq_lens_q=[96, 128], seq_lens_kv=[0, 64], sinks=sinks)
    _check(*res)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_right_band():
    """diagonal band right bound R > 0 (keep j <= diag + R, inclusive): the
    causal machinery with the diagonal TRANSLATED right -- the frontier width
    is R-independent. Cases: plain band; full band (left + right bounds);
    a band across a ragged KV tail; degenerate R >= S_kv (clamps to full
    visibility)."""
    scale = 1.0 / math.sqrt(128)
    for kwargs, s_q, s_kv in (
        (dict(right_bound=32), 256, 256),
        (dict(right_bound=32, left_bound=65), 256, 256),
        (dict(right_bound=48), 256, 300),
        (dict(right_bound=500), 128, 128),
    ):
        res = _run(2, 4, 4, s_q, s_kv, scale=scale, sdpa_kwargs=kwargs)
        _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True], ids=["none", "causal"])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_seq_q_trim(causal):
    """Dense padded graphs with per-batch Q lengths shorter than S_q: rows at
    or past seq_len_q[b] write O = 0 and LSE = -inf. Batch 2 is ZERO-length --
    the ragged population that crashes the backend's eng11; the frost path
    must handle it exactly."""
    scale = 1.0 / math.sqrt(128)
    kw = dict(use_causal_mask=True) if causal else {}
    res = _run(3, 4, 4, 256, 256, scale=scale, sdpa_kwargs=kw, seq_lens_q=[256, 129, 0], seq_lens_kv=[256, 200, 256])
    _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("s_q,s_kv", [(256, 130), (255, 321)], ids=lambda v: str(v))
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_dense_kv_tail(s_q, s_kv):
    """Dense no-mask with S_kv not a whole number of KV tiles (skv_tile=0):
    the kernel's first (masked) KV step bounds columns against the
    compile-time seqlen_k regardless of mask flags, so ragged S_kv is served
    natively -- no padding mask, no synthesized lengths. Rectangular S_q
    exercises the Q-row trim alongside the KV tail."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 4, 4, s_q, s_kv, scale=scale, sdpa_kwargs={})
    _check(*res)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_single_kv_tile_padded():
    """Padded lengths inside a single KV tile (the other collapse-exposed
    population: min(seq_kv_lens[b], shape) also bounds the trip count, so the
    same collapsed-pipeline layout is emitted here)."""
    scale = 1.0 / math.sqrt(128)
    for _ in range(3):
        res = _run(2, 4, 4, 97, 97, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), seq_lens_kv=[97, 60])
        _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal"])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_e5m2(mask):
    """E5M2 inputs: the MMA tag and the P-quantization target follow the
    input dtype (P is cast to e5m2, Scale_S maps it onto e5m2's range).
    E5M2's 2-bit mantissa doubles the P-quantization floor, hence the wider
    O tolerance; Amax_O is a fp32 pre-cast value and keeps its own."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 4, 4, 256, 256, scale=scale, sdpa_kwargs=_MASKS[mask], io_dtype=torch.float8_e5m2)
    _check(*res, tol_o=1e-1)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_e5m2_single_kv_tile():
    """The collapse-guard shape on the E5M2 specialization: a different MMA
    tag and P conversion make it a distinct compiled artifact, so it needs
    its own toolchain guard (repeated -- the guarded failure mode was
    nondeterministic)."""
    scale = 1.0 / math.sqrt(128)
    for _ in range(3):
        res = _run(1, 4, 4, 97, 97, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), io_dtype=torch.float8_e5m2)
        _check(*res, tol_o=1e-1)


def _fp8_graph_offers_sm120(io_dtype, o_dtype, D=128, D_v=None, sink=False, layout="bshd"):
    """Build one sdpa_fp8 graph and report whether the sm120 fp8 cell claims it.

    A capability rejection is the point, so nothing is executed; a graph that
    no engine at all serves counts as declined too.
    """
    import cudnn

    dev = "cuda"
    B, H, S = 1, 4, 256
    D_v = D if D_v is None else D_v
    torch_in = torch.float8_e5m2 if io_dtype == cudnn.data_type.FP8_E5M2 else torch.float8_e4m3fn
    if layout == "bhsd":
        # BHSD-contiguous: dense but NOT BSHD-physical.
        X = torch.randn(B, H, S, D, device=dev).to(torch_in)
        Xv = torch.randn(B, H, S, D_v, device=dev).to(torch_in)
    else:
        X = torch.randn(B, S, H, D, device=dev).to(torch_in).transpose(1, 2)
        Xv = torch.randn(B, S, H, D_v, device=dev).to(torch_in).transpose(1, 2)
    g = cudnn.pygraph(io_data_type=io_dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(X), g.tensor_like(X), g.tensor_like(Xv)
    scalars = [g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT) for _ in range(6)]
    kw = dict(
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
    if sink:
        kw["sink_token"] = g.tensor(dim=[1, H, 1, 1], stride=[H, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
    o, stats, _amx_s_unused, amx_o = g.sdpa_fp8(**kw)  # Amax_S: not requested (FROST does not produce it)
    o.set_output(True).set_dim([B, H, S, D_v]).set_stride([S * H * D_v, D_v, H * D_v, 1]).set_data_type(o_dtype)
    stats.set_output(True).set_dim([B, H, S, 1]).set_stride([H * S, S, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    try:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError, ValueError):
        # The op itself may refuse the shape before any engine is consulted;
        # for "this cell must not claim it" that is the same answer.
        return False
    return offers_engine(g, engine_name(arch="sm120", fp8=True))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_sink_offered():
    """Sink graphs are served (the sink extends the softmax denominator as a
    virtual column; P carries no sink column)."""
    import cudnn

    assert _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, sink=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_fp8_output_offered():
    """fp8 O is served via the direct quantizing store (Scale_O before the
    cast; Amax_O pre-cast)."""
    import cudnn

    assert _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E4M3)


@pytest.mark.L0
@pytest.mark.parametrize("D", [24, 144, 288])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_off_granule_head_dim_not_offered(D):
    """Declined head dims: 24 breaks the envelope alignment (multiples of 16,
    the TMA 16-byte global-stride rule at 1 byte/elem); 144/288 exceed the
    C++ sdpa_fp8 node's front door (d <= 128) / the row's 256 cap."""
    import cudnn

    assert not _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, D=D)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_head_dim_domain_offered():
    """Every graph the C++ front door admits is served: the d_qk <= 128 x
    d_v <= 128 cross (multiples of 32) plus the (192, 128) MLA pair."""
    import cudnn

    for D in range(32, 129, 32):
        for D_v in range(32, 129, 32):
            assert _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, D=D, D_v=D_v), f"({D}, {D_v}) not offered"
    assert _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, D=192, D_v=128), "(192, 128) MLA not offered"


@pytest.mark.L0
@pytest.mark.parametrize("D", [32, 64, 96], ids=lambda d: f"d{d}")
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_head_dims(D):
    """Correctness across the widened head-dim domain (exact, no padding).

    The graph front door (the C++ sdpa_fp8 node) admits d_qk <= 128 x
    d_v <= 128 plus the (192, 128) MLA pair, so >128 uniform dims cannot
    reach any engine; the kernel itself serves multiples of 32 up to 256."""
    scale = 1.0 / math.sqrt(D)
    res = _run(2, 4, 4, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), D=D)
    _check(*res)


@pytest.mark.L1
@pytest.mark.parametrize("D", [32, 96], ids=lambda d: f"d{d}")
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_head_dims_no_mask(D):
    scale = 1.0 / math.sqrt(D)
    res = _run(1, 4, 4, 256, 256, scale=scale, sdpa_kwargs={}, D=D)
    _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize(
    "D,D_v,mask",
    [
        (192, 128, "causal"),
        (192, 128, "causal_br"),
        (192, 128, "swa"),
        (192, 128, "padded"),
        (64, 128, "none"),
        (128, 32, "causal"),
    ],
)
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_mixed_head_dims(D, D_v, mask):
    """The QK^T and P@V sides are independent in the kernel; the 192/128
    shape is the MLA population the f16 cell already serves (the front
    door's only >128 carve-out), so it is crossed with every mask family."""
    scale = 1.0 / math.sqrt(D)
    seq_lens_kv = [256, 192] if mask == "padded" else None
    kw = {} if mask == "padded" else _MASKS[mask]
    res = _run(2, 4, 4, 256, 256, scale=scale, sdpa_kwargs=kw, seq_lens_kv=seq_lens_kv, D=D, D_v=D_v)
    _check(*res)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["causal_br", "swa", "padded"])
@pytest.mark.parametrize("D", [32, 96], ids=lambda d: f"d{d}")
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_head_dim_mask_cross(D, mask):
    """Non-d128 head dims crossed with the mask families (the base head-dims
    test covers plain causal). Masking and the head-dim envelope are
    independent axes in the kernel; a regression coupling them shows here."""
    scale = 1.0 / math.sqrt(D)
    seq_lens_kv = [256, 192] if mask == "padded" else None
    kw = {} if mask == "padded" else _MASKS[mask]
    res = _run(2, 4, 4, 256, 256, scale=scale, sdpa_kwargs=kw, seq_lens_kv=seq_lens_kv, D=D)
    _check(*res)


def _run_template_tail(D, D_v, *, mask, S=256):
    """Compile and launch the fp8 kernel template directly (production loader
    and adapter ABI) for head dims the graph front door cannot reach.

    The engine row declares the kernel's full domain — multiples of 32 up to
    256, QK^T/P@V sides independent — but the C++ sdpa_fp8 node admits only
    d_qk <= 128 x d_v <= 128 plus (192, 128) today, so the >128 tail is
    protected here at the template level.

    scale_s is 1.0 (P is cast to e4m3 unscaled and the reference does not
    model that cast), so the error floor is the bare P-quantization step
    (~0.055 measured on SM120); 0.15 separates it cleanly from real
    corruption (a dropped 32-column group or swizzle fault lands > 1).
    """
    import os

    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm120 import TemplateParams

    B, H = (2, 2) if mask == "padded" else (1, 2)
    kw = {"dtype_qkv": DTYPE_E4M3}
    swa_window = None
    seq_kv_lens = None
    if mask in ("causal", "swa"):
        kw["window_right"] = 0
    if mask == "swa":
        swa_window = 64  # kernel window_left=W keeps kv in [q-W, q]; same W as _ref's swa_window
        kw["window_left"] = swa_window
    if mask == "padded":
        kw["seq_kv_lens_present"] = True
        seq_kv_lens = [S, S - 73]  # batch 1 ends inside a KV tile at an odd offset
    path = os.path.join(os.path.dirname(os.path.abspath(api_dsl.__file__)), "kernels", "prefill_fp8_sm120.py")
    module = load_template(path, TemplateParams(**kw), tag=f"fp8_tail_d{D}_d{D_v}_{mask}")
    fn = module.compile(compute_capability=torch.cuda.get_device_capability(), b=B, qh=H, kh=H, sq=S, skv=S, d_qk=D, d_v=D_v, has_lse=False)

    dev = "cuda"

    def mk(*shape):
        return (torch.randn(*shape, device=dev) * 0.5).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)

    q8, k8, v8 = mk(B, S, H, D), mk(B, S, H, D), mk(B, S, H, D_v)  # compact BSHD, the kernel contract
    o = torch.zeros(B, S, H, D_v, device=dev, dtype=torch.float16)
    seq_q = torch.full((B,), S, dtype=torch.int32, device=dev)
    seq_kv = torch.tensor(seq_kv_lens, dtype=torch.int32, device=dev) if seq_kv_lens else seq_q
    amax_o = torch.zeros(1, dtype=torch.float32, device=dev)
    scale = 1.0 / math.sqrt(D)
    one = torch.ones(1, dtype=torch.float32, device=dev)  # identity device scales
    fn(
        q8,  # native fp8 element types across the ABI
        k8,
        v8,
        o,
        None,  # lse (has_lse=False)
        None,  # sinks (unsupported; ABI slot)
        seq_q,
        seq_kv,
        amax_o.view(torch.int32),  # bitcast-int32 atomicMax storage
        cutlass.Float32(scale * math.log2(math.e)),  # softmax_scale_log2 base (descale_q*descale_k fold in-kernel)
        cutlass.Float32(1.0),  # o_scale_fused base (descale_v*scale_o and the P-cast 2^-4 fold in-kernel)
        one,  # descale_q_t
        one,  # descale_k_t
        one,  # descale_v_t
        one,  # scale_o_t
        cutlass.Int32(0),  # thd_max_sq (dense: ignored)
        None,  # thd_q_lens (dense: folded out of the ABI)
        None,  # thd_kv_lens
        None,  # thd_lens_form
        cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    def bhsd(t):
        return t.float().permute(0, 2, 1, 3)

    o_ref, _ = _ref(bhsd(q8), bhsd(k8), bhsd(v8), scale=scale, is_causal=mask in ("causal", "swa"), swa_window=swa_window, seq_lens_kv=seq_kv_lens)
    of = bhsd(o)
    assert not torch.isnan(of).any(), "NaN in O"
    diff = (of - o_ref).abs().max().item()
    assert diff <= 0.15, f"max|O-ref|={diff:.4f} > 0.15 (P-quant floor ~0.055)"


@pytest.mark.L1
@pytest.mark.parametrize(
    "D,D_v,mask",
    [
        (160, 160, "causal"),
        (160, 160, "swa"),
        (160, 160, "none"),
        (256, 256, "causal"),
        (256, 256, "none"),
        (256, 256, "swa"),
        (256, 256, "padded"),
        (256, 128, "causal"),
        (128, 256, "causal"),
        (224, 160, "causal"),
    ],
)
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_head_dim_tail_direct(D, D_v, mask):
    """Correctness of the front-door-unreachable >128 head-dim tail (see
    _run_template_tail): uncommon uniform dims (160), the 256 domain max
    crossed with every mask family, 256 on each side alone (the QK^T and
    P@V swizzle/fragment paths size independently), and a mixed >128 pair."""
    _run_template_tail(D, D_v, mask=mask)


@pytest.mark.L0
@pytest.mark.parametrize("tiles", [(64, 64), (64, 128), (128, 64), (128, 128)])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_every_enumerated_tile(tiles):
    """propose_plans offers the whole tile domain, but a shape only ever runs
    one point of it, so the rest would ship untested. S_q=256 keeps both q_tile
    values meaningful (one full tile at 128, two at 64)."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), tiles=tiles)
    _check(*res)


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_fp8_sm120_device_scales_execute_reads_no_device_memory():
    """The graph path binds DEVICE scale tensors and the kernel folds
    dq*dk / ds*dv*so / ss in-kernel; amax_o divides by the device scale_o --
    no .item()/D2H readback anywhere (Rule 3), pinned by sync-debug mode 2
    around the execute. Numerics vs the fp32 reference are unchanged."""
    scale = 1.0 / math.sqrt(128)
    res = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), sync_debug=True)
    _check(*res)


_THD_SENTINEL = 2048.0


def _pack_thd(seqs, s_max, dtype):
    """Pack per-sequence ``(1, H, L_i, D)`` tensors into THD storage.

    Returns ``(dense_view, storage, ragged_offset)``: the ``(B, H, S_max, D)``
    packed-stride view over dense-sized storage whose first ``T*H*D`` elements
    hold the packed tokens, the raw storage, and the ``(B+1, 1, 1, 1)`` int64
    element-unit offsets (``cu_tokens * H * D``).
    """
    b, h, d = len(seqs), seqs[0].shape[1], seqs[0].shape[3]
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + s.shape[2])
    storage = torch.zeros(b * s_max * h * d, dtype=dtype, device="cuda")
    packed = storage[: max(cu[-1], 1) * h * d].view(max(cu[-1], 1), h, d)
    for i, s in enumerate(seqs):
        packed[cu[i] : cu[i + 1]].copy_(s[0].permute(1, 0, 2))
    view = storage.as_strided((b, h, s_max, d), (s_max * h * d, d, h * d, 1))
    ro = (torch.tensor(cu, dtype=torch.int64, device="cuda") * h * d).view(b + 1, 1, 1, 1)
    return view, storage, ro


def _run_thd_fp8(
    *,
    seq_q_lens,
    seq_kv_lens,
    h_q=8,
    h_kv=8,
    D=128,
    is_causal=True,
    bottom_right=False,
    window_size_right=None,
    with_sink=False,
    o_dtype=torch.float16,
    so_val=1.0,
    check_stats=False,
    stats_layout="head_major",
    raw_bind=False,
    poison_pad=False,
):
    """Run a ragged FP8 graph on the SM120 engine vs per-sequence references.

    Per-tensor FP8 means ONE descale per tensor for the whole packed batch, so
    the quantization scale is taken over every sequence at once.
    """
    import cudnn

    dev = "cuda"
    batch = len(seq_q_lens)
    s_q_max, s_kv_max = max(max(seq_q_lens), 1), max(max(seq_kv_lens), 1)
    scale = 1.0 / math.sqrt(D)

    def _quant_seqs(seqs):
        amax = max((s.abs().amax().item() for s in seqs if s.numel()), default=1.0)
        d = max(amax, 1e-8) / _E4M3_MAX
        return [(s / d).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn) for s in seqs], d

    q_f = [torch.randn(1, h_q, max(n, 1), D, device=dev)[:, :, :n] * 0.5 for n in seq_q_lens]
    k_f = [torch.randn(1, h_kv, max(n, 1), D, device=dev)[:, :, :n] * 0.5 for n in seq_kv_lens]
    v_f = [torch.randn(1, h_kv, max(n, 1), D, device=dev)[:, :, :n] * 0.5 for n in seq_kv_lens]
    q_8, dq = _quant_seqs([s.contiguous() for s in q_f])
    k_8, dk = _quant_seqs([s.contiguous() for s in k_f])
    v_8, dv = _quant_seqs([s.contiguous() for s in v_f])

    q_view, q_storage, q_ro = _pack_thd(q_8, s_q_max, torch.float8_e4m3fn)
    k_view, k_storage, k_ro = _pack_thd(k_8, s_kv_max, torch.float8_e4m3fn)
    v_view, v_storage, v_ro = _pack_thd(v_8, s_kv_max, torch.float8_e4m3fn)
    o_view, o_storage, o_ro = _pack_thd([torch.zeros(1, h_q, max(n, 1), D, dtype=o_dtype, device=dev)[:, :, :n] for n in seq_q_lens], s_q_max, o_dtype)
    if poison_pad:
        # Model uninitialized capacity: fp8 NaN bit patterns past the packed
        # KV tokens (real binders hand rounded-up allocations). The kernel
        # must keep them out of P @ V — masking S alone is not enough
        # (P = 0 times a NaN V row is still NaN).
        t_kv_total = sum(seq_kv_lens)
        for st, heads in ((k_storage, h_kv), (v_storage, h_kv)):
            st[t_kv_total * heads * D :] = torch.tensor(float("nan")).to(st.dtype)
    # The kernel writes every valid packed O token; everything else must come
    # back untouched. (Clamped into the O dtype's range for fp8 outputs.)
    sentinel = min(_THD_SENTINEL, torch.finfo(o_dtype).max)
    o_storage.fill_(sentinel)

    sq_t = torch.tensor(seq_q_lens, dtype=torch.int32, device=dev).view(batch, 1, 1, 1)
    skv_t = torch.tensor(seq_kv_lens, dtype=torch.int32, device=dev).view(batch, 1, 1, 1)
    amax_o = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    def sc(val):
        return torch.tensor([[[[val]]]], dtype=torch.float32, device=dev)

    s_scale = _E4M3_MAX
    dqt, dkt, dvt, dst, sst, sot = sc(dq), sc(dk), sc(dv), sc(1.0 / s_scale), sc(s_scale), sc(so_val)
    sinks = torch.randn(1, h_q, 1, 1, dtype=torch.float32, device=dev) if with_sink else None

    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q_view), g.tensor_like(k_view), g.tensor_like(v_view)
    rq, rk, rv, ro = (g.tensor_like(x) for x in (q_ro, k_ro, v_ro, o_ro))
    tq.set_ragged_offset(rq)
    tk.set_ragged_offset(rk)
    tv.set_ragged_offset(rv)
    sq_h = g.tensor_like(sq_t)
    skv_h = g.tensor_like(skv_t)

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
        generate_stats=check_stats,
        use_padding_mask=True,
        seq_len_q=sq_h,
        seq_len_kv=skv_h,
    )
    if window_size_right is not None:
        # A band graph is NON-causal in cuDNN's vocabulary (right_bound = R).
        kw["right_bound"] = window_size_right
    elif bottom_right:
        kw["use_causal_mask_bottom_right"] = True
    elif is_causal:
        kw["use_causal_mask"] = True
    sink_t = None
    if sinks is not None:
        sink_t = g.tensor_like(sinks, name="sink")
        kw["sink_token"] = sink_t

    o, stats, _amx_s_unused, amx_o = g.sdpa_fp8(**kw)  # Amax_S: not requested (FROST does not produce it)
    o_cudnn = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
    }[o_dtype]
    o.set_output(True).set_dim(list(o_view.shape)).set_stride(list(o_view.stride())).set_data_type(o_cudnn)
    o.set_ragged_offset(ro)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    if raw_bind:
        # mhas-style runtime binding: RAW rank-3 (T_cap, H, D) buffers rather
        # than the declared-rank BSHD views. The adapter must read/write the
        # packed storage directly (as_strided), never normalize through the
        # dense _to_bshd path -- its semantic-copy scratch write-back
        # scrambles packed bytes for h > 1 (the mhas test16/25/28/32 class).
        q_bind = q_storage.view(-1, h_q, D)
        k_bind = k_storage.view(-1, h_kv, D)
        v_bind = v_storage.view(-1, h_kv, D)
        o_bind = o_storage.view(-1, h_q, D)
    else:
        q_bind, k_bind, v_bind, o_bind = q_view, k_view, v_view, o_view
    vp = {
        tq: q_bind,
        tk: k_bind,
        tv: v_bind,
        rq: q_ro,
        rk: k_ro,
        rv: v_ro,
        ro: o_ro,
        sq_h: sq_t,
        skv_h: skv_t,
        dqn: dqt,
        dkn: dkt,
        dvn: dvt,
        dsn: dst,
        ssn: sst,
        son: sot,
        o: o_bind,
        amx_o: amax_o,
    }
    if sink_t is not None:
        vp[sink_t] = sinks
    cu = [0]
    for n in seq_q_lens:
        cu.append(cu[-1] + n)
    t_cap = max(64, -(-sum(seq_q_lens) // 64) * 64)
    stats_storage = None
    if check_stats:
        assert stats is not None
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        # One flat fp32 buffer covers either ragged Stats layout; the ragged
        # offsets are ELEMENT offsets under the declared strides, so each
        # layout scales cu_q by its own token stride.
        stats_storage = torch.full((h_q * t_cap,), _THD_SENTINEL, dtype=torch.float32, device=dev)
        if stats_layout == "head_major":
            # (H, t_cap): tokens contiguous within a head row (stride_s = 1);
            # element offsets = cu_q.
            stats.set_dim((batch, h_q, s_q_max, 1)).set_stride((h_q * t_cap, t_cap, 1, 1))
            stats_ro_t = (q_ro.flatten() // (D * h_q)).view(batch + 1, 1, 1, 1).contiguous()
        else:
            # token-major packed (T, H): heads contiguous within a token row
            # (stride_s = h_q); element offsets = cu_q * h_q.
            stats.set_dim((batch, h_q, s_q_max, 1)).set_stride((s_q_max * h_q, 1, h_q, 1))
            stats_ro_t = (q_ro.flatten() // D).view(batch + 1, 1, 1, 1).contiguous()
        stats_ro = g.tensor_like(stats_ro_t)
        stats.set_ragged_offset(stats_ro)
        vp[stats_ro] = stats_ro_t
        vp[stats] = stats_storage

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(arch="sm120", fp8=True))
    g.check_support()
    g.build_plans()
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    packed_o = o_storage[: max(cu[-1], 1) * h_q * D].view(max(cu[-1], 1), h_q, D)
    for i, (nq, nkv) in enumerate(zip(seq_q_lens, seq_kv_lens)):
        if nq == 0:
            continue
        o_ref, lse_ref = _ref(
            q_8[i].float() * dq,
            k_8[i].float() * dk,
            v_8[i].float() * dv,
            scale=scale,
            is_causal=is_causal or bottom_right or window_size_right is not None,
            bottom_right=bottom_right,
            right_bound=window_size_right if window_size_right is not None else 0,
            sinks=sinks,
        )
        # O carries Scale_O; compare against the unscaled reference.
        got = packed_o[cu[i] : cu[i + 1]].float() / so_val
        want = o_ref[0].permute(1, 0, 2).float()
        diff = (got - want).abs().max().item()
        tol_o = _O_TOL.get(o_dtype, 5e-2)
        assert diff <= tol_o, f"seq {i}: max|O-ref|={diff:.4f}"
        if check_stats:
            if stats_layout == "head_major":
                got_lse = stats_storage.view(h_q, t_cap)[:, cu[i] : cu[i + 1]]
            else:
                got_lse = stats_storage[: cu[-1] * h_q].view(cu[-1], h_q)[cu[i] : cu[i + 1]].transpose(0, 1)
            # Mirror _check: match the +-inf pattern first (zero-KV rows are
            # -inf on both sides), then compare only the finite entries.
            finite = torch.isfinite(lse_ref[0])
            assert torch.equal(torch.isfinite(got_lse), finite), f"seq {i}: LSE -inf pattern differs from the reference"
            ld = (got_lse[finite] - lse_ref[0][finite]).abs().max().item() if finite.any() else 0.0
            assert ld <= 5e-2, f"seq {i}: max|LSE-ref|={ld:.4f}"

    # Nothing outside the packed token range may be written.
    assert (o_storage[cu[-1] * h_q * D :] == sentinel).all(), "wrote past the packed O tokens"


@pytest.mark.L0
@torch_fork_set_rng(seed=40)
def test_fp8_sm120_thd():
    """THD self-attention: packed ragged batch vs per-sequence references."""
    _run_thd_fp8(seq_q_lens=[200, 150], seq_kv_lens=[200, 150], is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=41)
def test_fp8_sm120_thd_cross():
    """THD cross-attention + bottom-right: unequal packed Q and KV totals.

    Q shorter than KV per sequence on purpose -- with bottom-right alignment a
    longer Q leaves its leading rows with no valid column, and the all--inf
    softmax in the reference is NaN while the kernel writes the dead-row O=0.
    Dead rows have their own coverage; this case is about the packing.
    """
    _run_thd_fp8(seq_q_lens=[100, 60], seq_kv_lens=[180, 120], is_causal=True, bottom_right=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_thd_raw_buffer_binding():
    """THD with mhas-style RAW rank-3 runtime buffers (not the declared-rank
    BSHD views), h > 1, no mask: regression for the dense _to_bshd_writable
    scratch copy-back scrambling packed O bytes (mhas fp8 ragged
    test16/25/28/32)."""
    _run_thd_fp8(seq_q_lens=[35, 75], seq_kv_lens=[289, 190], h_q=3, h_kv=3, is_causal=False, raw_bind=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=56)
def test_fp8_sm120_thd_dirty_kv_pad():
    """KV storage CAPACITY beyond the packed tokens is uninitialized in real
    binders (rounded-up allocations), and fp8 has NaN bit patterns. The
    S-side column mask survives them (a select), but P @ V would multiply
    P = 0 into a NaN V row and 0 * NaN = NaN poisons every output row; the
    masked step must zero the sV tail rows before the PV MMA."""
    _run_thd_fp8(seq_q_lens=[35, 75], seq_kv_lens=[289, 190], h_q=3, h_kv=3, is_causal=False, poison_pad=True)


@pytest.mark.L0
@pytest.mark.parametrize("stats_layout", ["head_major", "token_major"])
@torch_fork_set_rng(seed=42)
def test_fp8_sm120_thd_stats(stats_layout):
    """THD + generate_stats in BOTH ragged-Stats layouts: head-major
    (H, head_stride) and token-major packed (T, H) -- the kernel specializes
    on either."""
    _run_thd_fp8(seq_q_lens=[200, 150], seq_kv_lens=[200, 150], is_causal=True, check_stats=True, stats_layout=stats_layout)


@pytest.mark.L1
@torch_fork_set_rng(seed=43)
def test_fp8_sm120_thd_gqa():
    """THD + grouped-query: the ragged path and the head mapping compose."""
    _run_thd_fp8(seq_q_lens=[128, 64, 192], seq_kv_lens=[128, 64, 192], h_q=8, h_kv=2, is_causal=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=27)
def test_fp8_sm120_thd_no_mask():
    """THD with NO causal/band mask (padding only), tile-unaligned per-sequence
    KV lengths: the first masked step must trim each sequence's ragged KV tail
    (the mhas fp8 ragged family's shape class)."""
    _run_thd_fp8(seq_q_lens=[35, 75], seq_kv_lens=[289, 190], h_q=3, h_kv=3, is_causal=False)


@pytest.mark.L0
@torch_fork_set_rng(seed=28)
def test_fp8_sm120_thd_fp8_out():
    """THD x fp8 O: the direct quantizing store's thd_varlen branch (packed
    rows, no dense zero-fill) with a NON-UNIT Scale_O applied before the
    cast."""
    _run_thd_fp8(seq_q_lens=[200, 150], seq_kv_lens=[200, 150], is_causal=True, o_dtype=torch.float8_e4m3fn, so_val=2.0)


@pytest.mark.L1
@torch_fork_set_rng(seed=29)
def test_fp8_sm120_thd_d_envelope():
    """THD x head-dim envelope (actual d = 112 < tile 128): the rank-4
    envelope TMA descriptor addresses packed K/V rows via kv_row_base and
    zero-fills the pad columns per sequence (the mhas test16 shape)."""
    _run_thd_fp8(seq_q_lens=[35, 75], seq_kv_lens=[289, 190], h_q=3, h_kv=3, D=112, is_causal=False)


@pytest.mark.L1
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
@torch_fork_set_rng(seed=25)
def test_fp8_sm120_thd_gqa_sink(stats_layout):
    """THD + GQA + attention sink through the packed epilogue fold, with the
    sink entering the ragged Stats (both declared layouts)."""
    _run_thd_fp8(seq_q_lens=[130, 70], seq_kv_lens=[130, 70], h_q=8, h_kv=2, is_causal=True, with_sink=True, check_stats=True, stats_layout=stats_layout)


@pytest.mark.L1
@torch_fork_set_rng(seed=26)
def test_fp8_sm120_thd_zero_length_sequence():
    """A zero-length sequence contributes no tokens and must not perturb its
    packed neighbors (O and ragged Stats). The last sequence has Q tokens but
    ZERO keys inside a live launch: its rows must come back O := 0 with
    LSE := -inf through the kernel's row_sum <= 0 guard, not stale memory."""
    _run_thd_fp8(seq_q_lens=[128, 0, 64], seq_kv_lens=[100, 0, 0], is_causal=True, check_stats=True, stats_layout="token_major")


@pytest.mark.L0
@torch_fork_set_rng(seed=57)
def test_fp8_sm120_thd_all_kv_zero():
    """EVERY sequence has zero keys: all Q rows are dead and must come back
    O := 0 / LSE := -inf through the kernel's row_sum <= 0 guard. This shape
    also exercises the adapter's all-KV-zero clamp, whose V binding is a
    zero stub carrying the INPUT dtype (Q's storage cannot back V when
    kh*d_v exceeds it, and O's dtype is independent of the input's)."""
    _run_thd_fp8(seq_q_lens=[200, 150], seq_kv_lens=[0, 0], is_causal=False, check_stats=True, stats_layout="token_major")


@pytest.mark.L0
@torch_fork_set_rng(seed=30)
def test_fp8_sm120_thd_right_band():
    """THD + TOP_LEFT right band: per-sequence diagonals each widened by R."""
    _run_thd_fp8(seq_q_lens=[130, 70], seq_kv_lens=[130, 70], is_causal=False, window_size_right=24)
