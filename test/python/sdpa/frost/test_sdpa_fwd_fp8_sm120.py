# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the FROST SM120 DSL per-tensor FP8 SDPA-forward engine.

Drives ``graph.sdpa_fp8`` (FP8 E4M3 Q/K/V + scalar per-tensor descales) routed to
the ``sdpa_fwd_prefill_sm120_fp8`` engine, and validates O against an fp32-dequant
reference. ``Amax_S`` and ``Amax_O`` are both produced in-kernel (bitcast-int32
atomicMax over the pre-cast fp32 values); both are checked.

SM120 envelope (see engines._sm120_fp8_spec): E4M3 in / FP16 out only, exact
d128, causal / bottom-right / SWA / KV-padding masks, THD (ragged) with
token-major Stats; no sink (Amax_S semantics), no head-major ragged Stats.
Everything outside that envelope is a capability-row decline, checked by the
*_not_offered tests below.

Requires: SM120/SM121 (consumer Blackwell), cutlass-dsl. Skips otherwise.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_blackwell_geforce, requires_dsl, select_engine as _select_engine

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
    # Fully-masked rows are -inf in logsumexp; the kernel writes -inf too.
    return torch.matmul(probs, v_e), probs.max().item(), torch.logsumexp(scores, dim=-1)


def _run(B, H_q, H_kv, S_q, S_kv, *, scale, sdpa_kwargs, seq_lens_kv=None, tiles=None, s_descale_gain=1.0):
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

    # Scale_S maps P (in (0,1] after the row-max subtraction) onto E4M3's
    # range; Descale_S undoes it in the epilogue. Non-unit on purpose -- unit
    # values would not exercise the scaling at all.
    s_scale = _E4M3_MAX
    dqt, dkt, dvt, dst, sst, sot = sc(dq), sc(dk), sc(dv), sc(s_descale_gain / s_scale), sc(s_scale), sc(1.0)

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
    _select_engine(g, engine_name(arch="sm120", fp8=True), tiles=tiles)
    g.check_support()
    g.build_plans()
    vp.update({o: Ob, stats: lse, amx_s: amax_s, amx_o: amax_o})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    ref_kw = _ref_kwargs(sdpa_kwargs)
    o_ref, amax_s_ref, lse_ref = _ref(Q8.float() * dq, K8.float() * dk, V8.float() * dv, scale=scale, seq_lens_kv=seq_lens_kv, **ref_kw)
    return Ob, o_ref, amax_s.item(), amax_s_ref, amax_o.item(), o_ref.abs().max().item(), lse.squeeze(-1), lse_ref


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


def _check(out, o_ref, amax_s, amax_s_ref, amax_o, amax_o_ref, lse=None, lse_ref=None):
    diff = (out.float() - o_ref).abs().max().item()
    assert diff <= 5e-2, f"max|O-ref|={diff:.4f} > 0.05"
    assert abs(amax_s - amax_s_ref) <= 0.03, f"amax_s {amax_s:.4f} vs ref {amax_s_ref:.4f}"
    assert abs(amax_o - amax_o_ref) <= 0.03, f"amax_o {amax_o:.4f} vs ref {amax_o_ref:.4f}"
    if lse is not None:
        finite = torch.isfinite(lse_ref)
        assert torch.equal(torch.isfinite(lse), finite), "LSE -inf pattern differs from the reference"
        d = (lse[finite] - lse_ref[finite]).abs().max().item() if finite.any() else 0.0
        assert d <= 3e-2, f"max|LSE-ref|={d:.4f} > 0.03"


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


def _fp8_graph_offers_sm120(io_dtype, o_dtype, D=128, sink=False):
    """Build one sdpa_fp8 graph and report whether the sm120 fp8 cell claims it.

    A capability rejection is the point, so nothing is executed; a graph that
    no engine at all serves counts as declined too.
    """
    import cudnn

    dev = "cuda"
    B, H, S = 1, 4, 256
    torch_in = torch.float8_e5m2 if io_dtype == cudnn.data_type.FP8_E5M2 else torch.float8_e4m3fn
    X = torch.randn(B, S, H, D, device=dev).to(torch_in).transpose(1, 2)
    g = cudnn.pygraph(io_data_type=io_dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(X), g.tensor_like(X), g.tensor_like(X)
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
    o, stats, amx_s, amx_o = g.sdpa_fp8(**kw)
    o.set_output(True).set_dim([B, H, S, D]).set_stride([S * H * D, D, H * D, 1]).set_data_type(o_dtype)
    stats.set_output(True).set_dim([B, H, S, 1]).set_stride([H * S, S, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    for t in (amx_s, amx_o):
        t.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    try:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError, ValueError):
        # The op itself may refuse the shape before any engine is consulted;
        # for "this cell must not claim it" that is the same answer.
        return False
    return engine_name(arch="sm120", fp8=True) in [g.get_plan_name_at_index(i) for i in range(len(g.plans))]


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_sink_not_offered():
    """Amax_S over a softmax that includes a sink column has no agreed meaning,
    so the row declines rather than reporting an amax for a different quantity."""
    import cudnn

    assert not _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, sink=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_fp8_output_not_offered():
    """The epilogue stores FP16; an fp8 O would need a quantizing store."""
    import cudnn

    assert not _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E4M3)


@pytest.mark.L0
@pytest.mark.parametrize("D", [64, 256])
@torch_fork_set_rng(seed=0)
def test_fp8_sm120_non_128_head_dim_not_offered(D):
    """The 8-bit fragment path has no zero-padding envelope, so d is exact."""
    import cudnn

    assert not _fp8_graph_offers_sm120(cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, D=D)


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
@torch_fork_set_rng(seed=5)
def test_fp8_sm120_s_scales_are_actually_applied():
    """Scale_S and Descale_S are reciprocal in normal use, so a correct kernel
    and one that ignores BOTH produce the same O — every other test here would
    pass either way. Break the reciprocity: with Descale_S = k/Scale_S the
    output must scale by exactly k, which only holds if both reach the math.
    """
    scale = 1.0 / math.sqrt(128)
    k = 2.0
    # _run draws its own Q/K/V, so both runs must start from the same RNG state
    # or the comparison says nothing about the scales.
    torch.manual_seed(1234)
    base = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    torch.manual_seed(1234)
    gained = _run(2, 8, 8, 256, 256, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), s_descale_gain=k)

    o_base, o_gain = base[0].float(), gained[0].float()
    ref = o_base * k
    diff = (o_gain - ref).abs().max().item()
    assert diff <= 5e-2 * k, f"Descale_S gain {k} not reflected in O: max|O_gain - {k}*O_base| = {diff:.4f}"
    # and the pair is not being cancelled away: the two runs must differ
    assert (o_gain - o_base).abs().max().item() > 1e-3


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


def _run_thd_fp8(*, seq_q_lens, seq_kv_lens, h_q=8, h_kv=8, is_causal=True, bottom_right=False, check_stats=False):
    """Run a ragged FP8 graph on the SM120 engine vs per-sequence references.

    Per-tensor FP8 means ONE descale per tensor for the whole packed batch, so
    the quantization scale is taken over every sequence at once.
    """
    import cudnn

    dev, D = "cuda", 128
    batch = len(seq_q_lens)
    s_q_max, s_kv_max = max(max(seq_q_lens), 1), max(max(seq_kv_lens), 1)
    scale = 1.0 / math.sqrt(D)

    def _quant_seqs(seqs):
        amax = max((s.abs().amax().item() for s in seqs), default=1.0)
        d = max(amax, 1e-8) / _E4M3_MAX
        return [(s / d).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn) for s in seqs], d

    q_f = [torch.randn(1, h_q, max(n, 1), D, device=dev)[:, :, :n] * 0.5 for n in seq_q_lens]
    k_f = [torch.randn(1, h_kv, max(n, 1), D, device=dev)[:, :, :n] * 0.5 for n in seq_kv_lens]
    v_f = [torch.randn(1, h_kv, max(n, 1), D, device=dev)[:, :, :n] * 0.5 for n in seq_kv_lens]
    q_8, dq = _quant_seqs([s.contiguous() for s in q_f])
    k_8, dk = _quant_seqs([s.contiguous() for s in k_f])
    v_8, dv = _quant_seqs([s.contiguous() for s in v_f])

    q_view, _, q_ro = _pack_thd(q_8, s_q_max, torch.float8_e4m3fn)
    k_view, _, k_ro = _pack_thd(k_8, s_kv_max, torch.float8_e4m3fn)
    v_view, _, v_ro = _pack_thd(v_8, s_kv_max, torch.float8_e4m3fn)
    o_view, o_storage, o_ro = _pack_thd(
        [torch.zeros(1, h_q, max(n, 1), D, dtype=torch.float16, device=dev)[:, :, :n] for n in seq_q_lens], s_q_max, torch.float16
    )
    # The kernel writes every valid packed O token; everything else must come
    # back untouched.
    o_storage.fill_(_THD_SENTINEL)

    sq_t = torch.tensor(seq_q_lens, dtype=torch.int32, device=dev).view(batch, 1, 1, 1)
    skv_t = torch.tensor(seq_kv_lens, dtype=torch.int32, device=dev).view(batch, 1, 1, 1)
    amax_s = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
    amax_o = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    def sc(val):
        return torch.tensor([[[[val]]]], dtype=torch.float32, device=dev)

    s_scale = _E4M3_MAX
    dqt, dkt, dvt, dst, sst, sot = sc(dq), sc(dk), sc(dv), sc(1.0 / s_scale), sc(s_scale), sc(1.0)

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
    if bottom_right:
        kw["use_causal_mask_bottom_right"] = True
    elif is_causal:
        kw["use_causal_mask"] = True

    o, stats, amx_s, amx_o = g.sdpa_fp8(**kw)
    o.set_output(True).set_dim(list(o_view.shape)).set_stride(list(o_view.stride())).set_data_type(cudnn.data_type.HALF)
    o.set_ragged_offset(ro)
    amx_s.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amx_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    vp = {
        tq: q_view,
        tk: k_view,
        tv: v_view,
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
        o: o_view,
        amx_s: amax_s,
        amx_o: amax_o,
    }
    cu = [0]
    for n in seq_q_lens:
        cu.append(cu[-1] + n)
    t_cap = max(64, -(-sum(seq_q_lens) // 64) * 64)
    stats_storage = None
    if check_stats:
        assert stats is not None
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        # head-major [h, t]: tokens contiguous within a head, heads strided by
        # the padded token capacity; offsets = cu_q * stride_s = cu_q. This is
        # the only ragged Stats layout the fp8 kernel stores.
        stats_storage = torch.full((h_q * t_cap,), _THD_SENTINEL, dtype=torch.float32, device=dev)
        stats.set_dim((batch, h_q, s_q_max, 1)).set_stride((h_q * t_cap, t_cap, 1, 1))
        stats_ro_t = (q_ro.flatten() // (D * h_q)).view(batch + 1, 1, 1, 1).contiguous()
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
    amax_s_ref = 0.0
    for i, (nq, nkv) in enumerate(zip(seq_q_lens, seq_kv_lens)):
        if nq == 0:
            continue
        o_ref, a_s_ref, lse_ref = _ref(
            q_8[i].float() * dq, k_8[i].float() * dk, v_8[i].float() * dv, scale=scale, is_causal=is_causal or bottom_right, bottom_right=bottom_right
        )
        amax_s_ref = max(amax_s_ref, a_s_ref)
        got = packed_o[cu[i] : cu[i + 1]].float()
        want = o_ref[0].permute(1, 0, 2).float()
        diff = (got - want).abs().max().item()
        assert diff <= 5e-2, f"seq {i}: max|O-ref|={diff:.4f}"
        if check_stats:
            got_lse = stats_storage.view(h_q, t_cap)[:, cu[i] : cu[i + 1]]
            ld = (got_lse - lse_ref[0]).abs().max().item()
            assert ld <= 5e-2, f"seq {i}: max|LSE-ref|={ld:.4f}"

    # Nothing outside the packed token range may be written.
    assert (o_storage[cu[-1] * h_q * D :] == _THD_SENTINEL).all(), "wrote past the packed O tokens"
    assert abs(amax_s.item() - amax_s_ref) <= 5e-2 * max(amax_s_ref, 1e-3), f"amax_s {amax_s.item()} vs {amax_s_ref}"


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
@torch_fork_set_rng(seed=42)
def test_fp8_sm120_thd_stats():
    """THD + generate_stats: the ragged Stats output is token-major [t, h]."""
    _run_thd_fp8(seq_q_lens=[200, 150], seq_kv_lens=[200, 150], is_causal=True, check_stats=True)


@pytest.mark.L1
@torch_fork_set_rng(seed=43)
def test_fp8_sm120_thd_gqa():
    """THD + grouped-query: the ragged path and the head mapping compose."""
    _run_thd_fp8(seq_q_lens=[128, 64, 192], seq_kv_lens=[128, 64, 192], h_q=8, h_kv=2, is_causal=True)
