# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the FROST SM120 block-scaled MXFP8 SDPA-forward engine.

Drives ``graph.sdpa_mxfp8`` (FP8 E4M3/E5M2 Q/K/V + per-32-block E8M0 scale
factors in cuDNN's F8_128x4 order) routed to ``sdpa_fwd_prefill_sm120_mxfp8``,
and validates O, Stats and Amax_O against an fp32 reference over the
dequantized inputs; Amax_O goes through ``sdpa.fp8.assert_amax_o`` (the
reference's pre-cast amax and the kernel's own stored O), with O and Amax_O
NaN-prefilled so an output never written fails. Inputs are quantized with the
torch-only quantizer in ``test/python/sdpa/mxfp8_quant.py``.

By default every 32-element block of Q, K (along D) and V (along S) is scaled
by its own power of two before quantization, so the stored block scales differ
strongly from block to block: a scale read for the wrong row, column, block or
plane changes O by up to the spread factor instead of hiding behind
near-uniform scales.

SM120 envelope (see engines._sm120_mxfp8_spec): E4M3/E5M2 in, FP16/BF16/FP8
out, head tiles any multiple of 32 up to 256 with independent QK^T and P@V
sides, head dims any multiple of 16 via TMA zero-padding, causal /
bottom-right / SWA / right-band / KV-padding masks, the per-batch seq_len_q
trim, attention sinks, GQA/MQA with PackGQA, dense split-KV, any dense
B/H/S storage order with the head dim innermost, THD (ragged) with
token- or head-major Stats, and the block-scaled MXFP8 O (E4M3 + UE8M0
per 32 in sf_o) at d = 128. Both head dims in (256, 512] ride the d512
flavor (``sm120/prefill_d512_mxfp8.py``, 64 x 64 CTA), covered by its own
section. The NVFP4 O is declined. Base-2 Stats are exercised through the direct adapter: the
``sdpa_mxfp8`` graph op has no ``stats_use_log2`` argument.

Requires: an SM120-line GPU (cc 12.x), cutlass-dsl. Skips otherwise.
"""

import math
from dataclasses import replace
from typing import NamedTuple, Optional

import pytest
import torch
import cudnn

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import make_dense_stats, offers_engine, requires_blackwell_geforce, requires_dsl, select_engine as _select_engine
from sdpa.fp8 import assert_amax_o

pytestmark = [requires_blackwell_geforce, requires_dsl]

_ENGINE = engine_name(arch="sm120", mxfp8=True)
_BLOCK = 32
_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_CUDNN_OTYPE = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float8_e4m3fn: "FP8_E4M3", torch.float8_e5m2: "FP8_E5M2"}
_MASKS = {
    "none": {},
    "causal": dict(use_causal_mask=True),
    "causal_br": dict(use_causal_mask_bottom_right=True),
    "swa": dict(use_causal_mask=True, diagonal_band_left_bound=65),  # window = 64
    "right_band": dict(diagonal_band_right_bound=48),
}


class _RunResult(NamedTuple):
    output: torch.Tensor  # O as stored: a BHSD view, or packed (T, H, D) under THD; a block-scaled O dequantized and divided by scale_o
    reference: torch.Tensor  # fp32 reference O, dead rows zero
    input_dtype: torch.dtype  # FP8 Q/K/V element type
    amax_o: torch.Tensor  # the graph's Amax_O buffer (NaN-prefilled)
    stats: Optional[torch.Tensor]  # (B, H, S_q), or (T, H) under THD
    reference_stats: Optional[torch.Tensor]
    sf_o_pad_ok: Optional[bool] = None  # block-scaled O: every kernel-owned SF_O pad row came back zero


class _Quantized(NamedTuple):
    """One MXFP8-quantized operand."""

    data: torch.Tensor  # fp8 [b, h, s, d]
    sf: torch.Tensor  # swizzled F8_128x4 scale bytes, shaped by ``sf_dims``
    dequant: torch.Tensor  # fp32 [b, h, s, d], data times its block scales
    sf_dims: tuple  # logical dims of the graph's scale tensor


def _cdiv(a, b):
    return (a + b - 1) // b


def _block_spread(shape, axis, lo, hi, generator):
    """Per-32-element-block power-of-two factors along ``axis`` (every other index varies freely)."""
    blocks = list(shape)
    blocks[axis] = _cdiv(shape[axis], _BLOCK)
    exps = torch.randint(lo, hi + 1, blocks, device="cuda", generator=generator).float()
    return torch.exp2(exps).repeat_interleave(_BLOCK, dim=axis).narrow(axis, 0, shape[axis])


def _qk_sf_cols(d):
    """Columns of a Q/K scale tensor: ceil(D / 32) blocks, padded to a multiple of 4."""
    return _cdiv(_cdiv(d, _BLOCK), 4) * 4


def _quantize(t, fp8, *, columnwise) -> _Quantized:
    """MXFP8-quantize ``t`` [b, h, s, d] fp32; ``columnwise`` scales along S (V), else along D (Q/K)."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    b, h, s, d = t.shape
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t, b, h, s, d, _BLOCK, fp8, with_ref=True)
    if columnwise:
        dims = (b, h, _cdiv(_cdiv(s, _BLOCK), 4) * 4, _cdiv(d, 128) * 128)
        return _Quantized(data_s, swz_s.view(torch.uint8).reshape(dims).contiguous(), data_s.float() * dq_s.reshape(b, h, s, d), dims)
    dims = (b, h, _cdiv(s, 128) * 128, _qk_sf_cols(d))
    return _Quantized(data_d, swz_d.view(torch.uint8).reshape(dims).contiguous(), data_d.float() * dq_d.reshape(b, h, s, d), dims)


def _draw(B, H_q, H_kv, S_q, S_kv, d_qk, d_v, fp8):
    """Random Q/K/V, each 32-element block scaled by its own power of two, quantized: (Q, K, V) as ``_Quantized``."""
    gen = torch.Generator(device="cuda").manual_seed(torch.initial_seed() % (2**31))
    qf = torch.randn(B, H_q, S_q, d_qk, device="cuda", generator=gen) * 0.5
    kf = torch.randn(B, H_kv, S_kv, d_qk, device="cuda", generator=gen) * 0.5
    vf = torch.randn(B, H_kv, S_kv, d_v, device="cuda", generator=gen) * 0.5
    qf = qf * _block_spread(qf.shape, 3, -2, 2, gen)
    kf = kf * _block_spread(kf.shape, 3, -2, 2, gen)
    vf = vf * _block_spread(vf.shape, 2, -3, 3, gen)
    return _quantize(qf, fp8, columnwise=False), _quantize(kf, fp8, columnwise=False), _quantize(vf, fp8, columnwise=True)


def _bshd(x):
    """A BHSD view over BSHD storage."""
    return x.permute(0, 2, 1, 3).contiguous().transpose(1, 2)


def _nan_output(shape, dtype):
    """Output storage prefilled with NaN, so an element never written fails the finiteness check."""
    return torch.full(shape, float("nan"), device="cuda").to(dtype)


def _padded_pitch(x, padding=16):
    """A BHSD view over BSHD storage with a padded physical head pitch."""
    b, h, s, d = x.shape
    view = torch.full((b, s, h, d + padding), float("nan"), device=x.device).to(x.dtype)[..., :d].transpose(1, 2)
    view.copy_(x)
    return view


# Storage orders for Q/K/V/O: a BHSD view holding ``x``'s values.
_LAYOUTS = {"bshd": lambda x: x, "padded_pitch": _padded_pitch, "unaligned_pitch": lambda x: _padded_pitch(x, 1), "bhsd": lambda x: x.contiguous()}


def _sf_tensor(g, dims):
    """A compact graph E8M0 scale tensor in the F8_128x4 order."""
    import cudnn

    return g.tensor(
        dim=list(dims),
        stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )


def _rowwise_sf_offsets(b, h, s, d, rows, blocks):
    """Flat byte offsets of the rowwise (Q/K) scales of ``rows`` x ``blocks`` for every (batch, head):
    atoms ordered (b, h, s-tile, d-plane), byte ``(r % 32) * 16 + ((r // 32) % 4) * 4 + c % 4``."""
    s_tiles, planes = _cdiv(s, 128), _cdiv(_cdiv(d, _BLOCK), 4)
    bh = torch.arange(b * h, device="cuda").view(-1, 1, 1)
    r = torch.as_tensor(rows, device="cuda").view(1, -1, 1)
    c = torch.as_tensor(blocks, device="cuda").view(1, 1, -1)
    atom = (bh * s_tiles + r // 128) * planes + c // 4
    return (atom * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + c % 4).view(b, h, len(rows), len(blocks))


def _columnwise_sf_offsets(b, h, s, d, blocks, cols):
    """Flat byte offsets of the columnwise (V) scales of key ``blocks`` x D ``cols``: atoms ordered
    (d-plane, b, h, s-tile), byte ``(n % 32) * 16 + ((n // 32) % 4) * 4 + j % 4``."""
    s_tiles = _cdiv(_cdiv(_cdiv(s, _BLOCK), 4) * 4 * _BLOCK, 128)
    bh = torch.arange(b * h, device="cuda").view(-1, 1, 1)
    j = torch.as_tensor(blocks, device="cuda").view(1, -1, 1)
    n = torch.as_tensor(cols, device="cuda").view(1, 1, -1)
    atom = (n // 128) * (b * h * s_tiles) + bh * s_tiles + j // 4
    return (atom * 512 + (n % 32) * 16 + ((n // 32) % 4) * 4 + j % 4).view(b, h, len(blocks), len(cols))


def _ref(qd, kd, vd, *, scale, is_causal=False, bottom_right=False, swa_window=None, right_bound=0, sinks=None, seq_lens_q=None, seq_lens_kv=None):
    """fp32 reference with the kernel's mask / sink / trim semantics (BHSD, GQA-aware); returns (O, LSE)."""
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
    diag = 0
    if bottom_right:
        # The diagonal anchors at each batch's (seq_len_q, seq_len_kv) corner.
        slq_d = torch.as_tensor(seq_lens_q if seq_lens_q is not None else [s_q] * b, device=dev).view(b, 1, 1, 1)
        slk_d = torch.as_tensor(seq_lens_kv if seq_lens_kv is not None else [s_kv] * b, device=dev).view(b, 1, 1, 1)
        diag = slk_d - slq_d
    if is_causal:
        masked = masked | (j > i + diag + right_bound)
    if swa_window is not None:
        masked = masked | (j < i + diag - swa_window)
    if seq_lens_kv is not None:
        masked = masked | (j >= torch.as_tensor(seq_lens_kv, device=dev).view(b, 1, 1, 1))
    scores = scores.masked_fill(masked, float("-inf"))
    if sinks is not None:
        scores = torch.cat([scores, sinks.float().view(1, h_q, 1, 1).expand(b, h_q, s_q, 1)], dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)[..., :s_kv]
    out = torch.matmul(probs, v_e)
    if seq_lens_q is not None:
        dead = i >= torch.as_tensor(seq_lens_q, device=dev).view(b, 1, 1, 1)
        out = torch.where(dead, torch.zeros((), device=dev), out)
        lse = lse.masked_fill(dead.squeeze(-1), float("-inf"))
    return out, lse


def _ref_kwargs(sdpa_kwargs):
    out = {}
    if sdpa_kwargs.get("use_causal_mask"):
        out["is_causal"] = True
    if sdpa_kwargs.get("use_causal_mask_bottom_right"):
        out.update(is_causal=True, bottom_right=True)
    if sdpa_kwargs.get("diagonal_band_right_bound") is not None:
        # A band graph lowers as the causal machinery with the diagonal widened right by R.
        out.update(is_causal=True, right_bound=sdpa_kwargs["diagonal_band_right_bound"])
    if sdpa_kwargs.get("diagonal_band_left_bound") is not None:
        out["swa_window"] = sdpa_kwargs["diagonal_band_left_bound"] - 1
    return out


def _run(
    B,
    H_q,
    H_kv,
    S_q,
    S_kv,
    *,
    d_qk=128,
    d_v=None,
    in_key="e4m3",
    out_dt=torch.float16,
    scale=None,
    sdpa_kwargs=None,
    seq_lens_q=None,
    seq_lens_kv=None,
    sinks=None,
    with_stats=True,
    tiles=None,
    pack_gqa=None,
    split_kv=None,
    poison_pad=False,
    poison_v_sf_pad_cols=False,
    poison_qk_sf_pad_blocks=False,
    sync_debug=False,
    layout="bshd",
    sf_o_layout=None,
    scale_o=None,
    sched_policy=None,
):
    """Quantize, build the sdpa_mxfp8 graph, pin the SM120 MXFP8 engine (and ``split_kv``), execute.
    ``layout`` names the storage order (``_LAYOUTS``) of Q/K/V/O. ``sf_o_layout`` ("planes" / "token_major")
    requests the block-scaled MXFP8 O (E4M3 + UE8M0 per 32 d in ``sf_o``), with the optional global ``scale_o``."""
    import cudnn

    dev = "cuda"
    d_v = d_qk if d_v is None else d_v
    scale = d_qk**-0.5 if scale is None else scale
    sdpa_kwargs = sdpa_kwargs or {}
    fp8 = _FP8[in_key]
    q_op, k_op, v_op = _draw(B, H_q, H_kv, S_q, S_kv, d_qk, d_v, fp8)
    Qb, Kb, Vb = _bshd(q_op.data), _bshd(k_op.data), _bshd(v_op.data)
    sfq, sfk, sfv = q_op.sf, k_op.sf, v_op.sf
    if poison_pad:
        # Pad storage past each batch's length carries NaN data and NaN (0xFF)
        # scale encodings; the reference reads the clean tensors, the kernel must
        # keep the pad rows and wholly-padded scale blocks out of O and Stats.
        assert seq_lens_kv is not None
        nan8 = torch.tensor(float("nan")).to(fp8).view(torch.uint8)
        Qb, Kb, Vb = Qb.clone(), Kb.clone(), Vb.clone()
        sfq, sfk, sfv = sfq.clone(), sfk.clone(), sfv.clone()
        for b, n in enumerate(seq_lens_kv):
            Kb[b, :, n:, :].view(torch.uint8).fill_(nan8)
            Vb[b, :, n:, :].view(torch.uint8).fill_(nan8)
            rows = list(range(n, S_kv))
            if rows:
                off = _rowwise_sf_offsets(B, H_kv, S_kv, d_qk, rows, list(range(_cdiv(d_qk, _BLOCK))))[b]
                sfk.view(-1)[off.reshape(-1)] = 0xFF
            dead_blocks = list(range(_cdiv(n, _BLOCK), _cdiv(S_kv, _BLOCK)))
            if dead_blocks:
                off = _columnwise_sf_offsets(B, H_kv, S_kv, d_v, dead_blocks, list(range(d_v)))[b]
                sfv.view(-1)[off.reshape(-1)] = 0xFF
        for b, n in enumerate(seq_lens_q or []):
            Qb[b, :, n:, :].view(torch.uint8).fill_(nan8)
            rows = list(range(n, S_q))
            if rows:
                off = _rowwise_sf_offsets(B, H_q, S_q, d_qk, rows, list(range(_cdiv(d_qk, _BLOCK))))[b]
                sfq.view(-1)[off.reshape(-1)] = 0xFF
    if poison_v_sf_pad_cols:
        # V-scale rows past d_v inside its last 128-row atom carry the NaN encoding (caller-owned pad bytes).
        sfv = sfv.clone()
        off = _columnwise_sf_offsets(B, H_kv, S_kv, d_v, list(range(_cdiv(S_kv, _BLOCK))), list(range(d_v, _cdiv(d_v, 128) * 128)))
        sfv.view(-1)[off.reshape(-1)] = 0xFF
    if poison_qk_sf_pad_blocks:
        # Q/K-scale blocks past d_qk inside the last plane carry the NaN encoding (caller-owned pad bytes).
        pad_blocks = list(range(_cdiv(d_qk, _BLOCK), _qk_sf_cols(d_qk)))
        if pad_blocks:
            sfq, sfk = sfq.clone(), sfk.clone()
            sfq.view(-1)[_rowwise_sf_offsets(B, H_q, S_q, d_qk, list(range(S_q)), pad_blocks).reshape(-1)] = 0xFF
            sfk.view(-1)[_rowwise_sf_offsets(B, H_kv, S_kv, d_qk, list(range(S_kv)), pad_blocks).reshape(-1)] = 0xFF

    Ob = _nan_output((B, S_q, H_q, d_v), out_dt).transpose(1, 2)
    Qb, Kb, Vb, Ob = (_LAYOUTS[layout](x) for x in (Qb, Kb, Vb, Ob))
    lse = make_dense_stats(B, H_q, S_q, "contiguous") if with_stats else None
    amax_buf = torch.full((1, 1, 1, 1), float("nan"), device=dev)

    g = cudnn.pygraph(
        io_data_type=getattr(cudnn.data_type, _CUDNN_ITYPE[in_key]),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q, k, v = g.tensor_like(Qb), g.tensor_like(Kb), g.tensor_like(Vb)
    dq, dk, dv = (_sf_tensor(g, op.sf_dims) for op in (q_op, k_op, v_op))
    kw = dict(q=q, k=k, v=v, descale_q=dq, descale_k=dk, descale_v=dv, attn_scale=scale, generate_stats=with_stats)
    vp = {q: Qb, k: Kb, v: Vb, dq: sfq, dk: sfk, dv: sfv}
    if sinks is not None:
        st = g.tensor_like(sinks)
        kw["sink_token"] = st
        vp[st] = sinks
    if seq_lens_q is not None and seq_lens_kv is None:
        seq_lens_kv = [S_kv] * B
    if seq_lens_kv is not None:
        slq = torch.tensor(seq_lens_q if seq_lens_q is not None else [S_q] * B, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        sq_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        skv_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_q=sq_h, seq_len_kv=skv_h)
        vp.update({sq_h: slq, skv_h: slk})
    if sf_o_layout is not None:
        from sdpa.block_scale_o_ref import round_up

        # SF_O pre-filled with 0xAA (scale bytes never written stay visible); token-major rows past B*S_q are caller-owned.
        C = max(4, round_up(d_v // _BLOCK, 4))
        if sf_o_layout == "planes":
            R = round_up(S_q, 128)
            sf_o_buf = torch.full((B, H_q, R, C), 0xAA, device=dev, dtype=torch.uint8)
            sf_o_stride = [H_q * R * C, R * C, C, 1]
        else:
            R = S_q
            sf_o_buf = torch.full((round_up(B * S_q, 128), H_q * C), 0xAA, device=dev, dtype=torch.uint8)
            sf_o_buf[B * S_q :] = 0
            sf_o_stride = [R * H_q * C, C, H_q * C, 1]
        kw["sf_o"] = g.tensor(dim=[B, H_q, R, C], stride=sf_o_stride, data_type=cudnn.data_type.FP8_E8M0)
        vp[kw["sf_o"]] = sf_o_buf
        if scale_o is not None:
            kw["scale_o"] = g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
            vp[kw["scale_o"]] = torch.full((1,), scale_o, device=dev)
    kw.update(sdpa_kwargs)
    o, stats_t, amax_o = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    if with_stats:
        stats_t.set_output(True).set_dim([B, H_q, S_q, 1]).set_stride(list(lse.stride())).set_data_type(cudnn.data_type.FLOAT)
        vp[stats_t] = lse
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    vp[amax_o] = amax_buf

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    chosen = _select_engine(g, _ENGINE, tiles=tiles, pack_gqa=pack_gqa, sched_policy=sched_policy)
    if split_kv is not None and (chosen.knobs.split_kv or 1) != split_kv:
        # The ranked plans need not carry this split count: re-plan the engine's own knob set with it.
        g.create_execution_plan(chosen.engine_id, replace(chosen.knobs, split_kv=split_kv, sched_policy=0))
        g.select_plan(len(g.plans) - 1)
    assert split_kv is None or (g.plans[g._plan_index].knobs.split_kv or 1) == split_kv
    g.check_support()
    g.build_plans()
    vp[o] = Ob
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    if sync_debug:
        g.execute(vp, ws)  # warm the plan's lazy state outside the probe
        torch.cuda.synchronize()
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        if sync_debug:
            torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()

    o_ref, lse_ref = _ref(
        q_op.dequant, k_op.dequant, v_op.dequant, scale=scale, sinks=sinks, seq_lens_q=seq_lens_q, seq_lens_kv=seq_lens_kv, **_ref_kwargs(sdpa_kwargs)
    )
    stats = lse.squeeze(-1) if with_stats else None
    if sf_o_layout is not None:
        from sdpa.block_scale_o_ref import dequant_block_scaled_o

        o_deq, pad_ok = dequant_block_scaled_o(Ob, sf_o_buf, _BLOCK, sf_o_layout, B, H_q, S_q, d_v, C)
        return _RunResult(o_deq / (1.0 if scale_o is None else scale_o), o_ref, fp8, amax_buf, stats, lse_ref if with_stats else None, pad_ok)
    return _RunResult(Ob, o_ref, fp8, amax_buf, stats, lse_ref if with_stats else None)


def _o_tol(res: _RunResult, d_qk=128):
    """Half-output tolerance relative to the reference's magnitude; FP8 outputs add three times the
    output dtype's own quantization floor of the reference."""
    e5m2 = res.input_dtype == torch.float8_e5m2
    base = 8e-2 if e5m2 and d_qk > 128 else 7e-2 if e5m2 else 5e-2
    tol = base * max(1.0, res.reference.abs().max().item())
    if res.output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        floor = (res.reference - res.reference.to(res.output.dtype).float()).abs().max().item()
        tol = max(tol, 3.0 * floor)
    return tol


def _check(res: _RunResult, d_qk=128):
    out = res.output.float()
    assert torch.isfinite(out).all(), "non-finite O (NaN = an element never written)"
    diff = (out - res.reference).abs().max().item()
    tol = _o_tol(res, d_qk)
    assert diff <= tol, f"max|O-ref|={diff:.4f} > {tol:.4f}"
    assert_amax_o(res.amax_o, res.reference.abs().max().item(), torch_itype=res.input_dtype, torch_otype=res.output.dtype, o_gpu=res.output)
    if res.stats is not None:
        finite = torch.isfinite(res.reference_stats)
        assert torch.equal(torch.isfinite(res.stats), finite), "Stats -inf pattern differs from the reference"
        d = (res.stats[finite] - res.reference_stats[finite]).abs().max().item() if finite.any() else 0.0
        assert d <= 3e-2, f"max|LSE-ref|={d:.4f} > 0.03"


# --- masks, dtypes, head dims ------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("mask", list(_MASKS))
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
@torch_fork_set_rng(seed=11)
def test_mxfp8_sm120_masks(mask, in_key):
    _check(_run(2, 4, 2, 256, 256, in_key=in_key, sdpa_kwargs=_MASKS[mask]))


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal"])
@torch_fork_set_rng(seed=139)
def test_mxfp8_sm120_negative_scale(mask):
    """A negative attn_scale runs at |scale| with Q's sign bits flipped as Q loads (#1435)."""
    _check(_run(2, 4, 2, 130, 200, scale=-0.7 / math.sqrt(128), sdpa_kwargs=_MASKS[mask]))


@pytest.mark.L0
@pytest.mark.parametrize("out_dt", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2], ids=["fp16", "bf16", "e4m3", "e5m2"])
@torch_fork_set_rng(seed=13)
def test_mxfp8_sm120_o_dtype(out_dt):
    _check(_run(1, 4, 4, 192, 320, out_dt=out_dt, sdpa_kwargs=_MASKS["causal"]))


@pytest.mark.L0
@pytest.mark.parametrize(
    "d_qk,d_v",
    [(64, 64), (128, 128), (192, 128), (256, 256), (32, 32), (96, 160), pytest.param(224, 64, marks=pytest.mark.L1)],
    ids=lambda d: str(d),
)
@torch_fork_set_rng(seed=17)
def test_mxfp8_sm120_head_dims(d_qk, d_v):
    """Native head tiles: each 32-column scale block is one k32 MMA step; D=256 spans two scale planes."""
    _check(_run(1, 4, 2, 256, 384, d_qk=d_qk, d_v=d_v, sdpa_kwargs=_MASKS["causal"]), d_qk=d_qk)


@pytest.mark.L0
@pytest.mark.parametrize("kv_tile", [64, 128], ids=lambda n: f"kv{n}")
@pytest.mark.parametrize("d_v", [256, 128], ids=lambda d: f"dv{d}")
@torch_fork_set_rng(seed=18)
def test_mxfp8_sm120_d256_qk_on_both_kv_tiles(d_v, kv_tile):
    """A 256-column Q/K head on either KV tile."""
    _check(_run(1, 4, 2, 256, 384, d_qk=256, d_v=d_v, tiles=(128, kv_tile), sdpa_kwargs=_MASKS["causal"]), d_qk=256)


@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(80, 80), (112, 112), (112, 80), pytest.param(144, 208, marks=pytest.mark.L1)], ids=lambda d: str(d))
@torch_fork_set_rng(seed=19)
def test_mxfp8_sm120_head_dim_envelope(d_qk, d_v):
    """Multiples of 16 ride the next 32-multiple tile: TMA zero-fills the columns inside the last scaled block,
    and the NaN-encoded V-scale rows past d_v stay out of O and Amax_O."""
    _check(_run(1, 2, 2, 192, 256, d_qk=d_qk, d_v=d_v, sdpa_kwargs=_MASKS["causal"], poison_v_sf_pad_cols=True), d_qk=d_qk)


@pytest.mark.L0
@pytest.mark.parametrize("tiles", [(64, 64), (64, 128), (128, 64), (128, 128)], ids=lambda t: f"{t[0]}x{t[1]}")
@torch_fork_set_rng(seed=23)
def test_mxfp8_sm120_every_tile(tiles):
    """A 64-key tile reads half of its 128-row K/V scale tile; both halves must pick their own blocks."""
    _check(_run(1, 2, 2, 320, 448, tiles=tiles, sdpa_kwargs=_MASKS["causal_br"]))


@pytest.mark.L0
@pytest.mark.parametrize("h_q,h_kv", [(8, 2), (8, 1)], ids=["g4", "g8"])
@pytest.mark.parametrize("mask", ["causal", "swa"])
@torch_fork_set_rng(seed=31)
def test_mxfp8_sm120_pack_gqa(h_q, h_kv, mask):
    """PackGQA rows interleave query heads: each lane's Q scale row follows its (token, head)."""
    _check(_run(2, h_q, h_kv, 40, 300, pack_gqa=True, sdpa_kwargs=_MASKS[mask]))


@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(80, 112), (144, 208)], ids=lambda d: str(d))
@torch_fork_set_rng(seed=127)
def test_mxfp8_sm120_pack_gqa_envelope_head_dims(d_qk, d_v):
    """PackGQA over zero-padded envelope head dims: the packed Q rows keep their scale rows while the last
    scale block of each side covers the zero-filled columns."""
    _check(_run(2, 8, 2, 40, 300, d_qk=d_qk, d_v=d_v, pack_gqa=True, sdpa_kwargs=_MASKS["causal"]), d_qk)


# --- PackGQA with tile_q % G != 0: q_tile // G tokens, the rows past them dead
@pytest.mark.L0
@pytest.mark.parametrize(
    "h_q,d,tiles",
    [
        (24, 128, (128, 128)),  # G=6: 21 tokens in 126 live rows
        (24, 128, (64, 64)),  # G=6: 10 tokens in 60 live rows; the tile at token 60 crosses key 64
        (28, 256, (128, 64)),  # G=7: 18 tokens in 126 live rows; the tile at token 54 crosses key 64
        (20, 512, (64, 64)),  # G=5 on the D=512 template: 12 tokens in 60 live rows; token 60 crosses key 64
    ],
    ids=["g6_128x128", "g6_64x64", "g7_d256", "g5_d512"],
)
@torch_fork_set_rng(seed=37)
def test_mxfp8_sm120_pack_gqa_tile_q_not_divisible(h_q, d, tiles):
    """A group the tile does not divide packs: each live row's Q scale follows its (token, head), a dead row
    takes the unit scale and never stores. Top-left causal over 70 tokens starts packed tiles off the KV grid."""
    _check(_run(2, h_q, 4, 70, 280, d_qk=d, tiles=tiles, pack_gqa=True, sdpa_kwargs=_MASKS["causal"]), d)


@pytest.mark.L0
@pytest.mark.parametrize("out_dt", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "e4m3"])
@pytest.mark.parametrize("d,tiles", [(128, None), (256, (128, 64)), (512, (64, 64))], ids=["d128", "d256", "d512"])
@torch_fork_set_rng(seed=39)
def test_mxfp8_sm120_pack_gqa_tile_q_not_divisible_split(d, tiles, out_dt):
    """GQA 6 packed under a pinned split over two (128-row) or four (64-row) Q tiles, D=128, 256 and 512: the
    live rows' partials recombine."""
    _check(_run(2, 24, 4, 33, 1536, d_qk=d, tiles=tiles, out_dt=out_dt, pack_gqa=True, split_kv=2, sdpa_kwargs=_MASKS["causal_br"]), d)


@pytest.mark.L0
@pytest.mark.parametrize("d,tiles", [(128, (64, 64)), (512, (64, 64))], ids=["d128", "d512"])
@torch_fork_set_rng(seed=41)
def test_mxfp8_sm120_pack_gqa_dead_rows_stay_out_of_amax(monkeypatch, d, tiles):
    """A packed tile's dead rows map onto the next tile's first token, a token below S_q, so Amax_O must skip
    them. Key 0 carries a large V row the live queries never weight (their Q and its K oppose in column 0),
    while a dead row's zero Q weights every key alike: its O would exceed every live one."""

    def draw(B, H_q, H_kv, S_q, S_kv, d_qk, d_v, fp8):
        q = torch.randn(B, H_q, S_q, d_qk, device="cuda") * 0.5
        q[..., 0] = 2.0
        k = torch.randn(B, H_kv, S_kv, d_qk, device="cuda") * 0.5
        k[:, :, 0, 0] = -16.0 * math.sqrt(d_qk)
        v = torch.randn(B, H_kv, S_kv, d_v, device="cuda") * 0.5
        v[:, :, 0] = 400.0
        return _quantize(q, fp8, columnwise=False), _quantize(k, fp8, columnwise=False), _quantize(v, fp8, columnwise=True)

    monkeypatch.setitem(globals(), "_draw", draw)
    res = _run(1, 24, 4, 40, 256, d_qk=d, tiles=tiles, pack_gqa=True)
    assert res.reference.abs().max().item() < 0.5, "the live rows must not weight key 0"
    _check(res, d)


@pytest.mark.L0
@pytest.mark.parametrize("out_dt", [torch.float16, torch.float8_e4m3fn], ids=["f16", "e4m3"])
@pytest.mark.parametrize("d,tiles,s_q", [(128, (128, 128), 30), (512, (64, 64), 20)], ids=["d128", "d512"])
@torch_fork_set_rng(seed=43)
def test_mxfp8_sm120_pack_gqa_dead_rows_never_store(d, tiles, s_q, out_dt):
    """A packed tile's dead rows map onto the next tile's first token. Under LPT every head's last Q tile is
    dispatched before any first one, and 384 of them outlast one wave, so a dead-row store would land after
    the next tile's live store and overwrite it."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT

    _check(_run(96, 24, 4, s_q, 256, d_qk=d, out_dt=out_dt, tiles=tiles, pack_gqa=True, sched_policy=SCHED_LPT, sdpa_kwargs=_MASKS["causal"]), d)


@pytest.mark.L0
@pytest.mark.parametrize("s_q,s_kv", [(256, 130), (255, 321), (97, 97), (1, 200)], ids=lambda v: str(v))
@torch_fork_set_rng(seed=37)
def test_mxfp8_sm120_ragged_extents(s_q, s_kv):
    """S_q / S_kv off the tile and scale-block grids; the tail blocks of the last scale tile are padding."""
    _check(_run(2, 2, 2, s_q, s_kv))


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal_br"])
@torch_fork_set_rng(seed=43)
def test_mxfp8_sm120_seq_q_trim(mask):
    _check(_run(3, 2, 2, 256, 256, seq_lens_q=[256, 130, 7], seq_lens_kv=[256, 250, 64], sdpa_kwargs=_MASKS[mask]))


@pytest.mark.L0
@pytest.mark.parametrize("tiles", [(128, 128), (128, 64)], ids=lambda t: f"kv{t[1]}")
@torch_fork_set_rng(seed=47)
def test_mxfp8_sm120_poisoned_pad(tiles):
    """Pad rows carry NaN data and NaN (0xFF) scale encodings past every batch's lengths: the dense
    padding mask, the V-scale tail guard and the Q-scale row guard keep them out of O and Stats."""
    res = _run(3, 4, 2, 192, 256, seq_lens_q=[192, 100, 40], seq_lens_kv=[256, 77, 129], tiles=tiles, poison_pad=True)
    _check(res)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal"])
@torch_fork_set_rng(seed=53)
def test_mxfp8_sm120_sink(mask):
    sinks = torch.randn(1, 4, 1, 1, device="cuda") * 2
    _check(_run(2, 4, 2, 256, 256, sinks=sinks, sdpa_kwargs=_MASKS[mask]))


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_mxfp8_sm120_sink_dead_rows():
    """A row with no visible key keeps the sink alone: O := 0, a finite LSE."""
    sinks = torch.randn(1, 2, 1, 1, device="cuda")
    _check(_run(2, 2, 2, 128, 128, sinks=sinks, seq_lens_kv=[128, 0]))


@pytest.mark.L0
@pytest.mark.parametrize("split_kv", [2, 3])
@pytest.mark.parametrize("out_dt", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "e4m3"])
@torch_fork_set_rng(seed=63)
def test_mxfp8_sm120_split_kv_graph(split_kv, out_dt):
    """A pinned split count (GQA): each split walks its own range of K/V scale tiles; the combine casts
    the half partials once and produces O, Stats and the recombined O's Amax_O."""
    _check(_run(2, 4, 2, 64, 1536, out_dt=out_dt, split_kv=split_kv, sdpa_kwargs=_MASKS["causal_br"]))


@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(256, 256), (192, 128), (80, 112)], ids=lambda d: str(d))
@torch_fork_set_rng(seed=65)
def test_mxfp8_sm120_split_kv_graph_head_dims(d_qk, d_v):
    """Split-KV with two V scale planes (D_V = 256), asymmetric QK^T / P@V tiles and zero-padded envelope
    head dims, over a KV length off the tile grid (split-KV declines padding masks)."""
    _check(_run(1, 4, 1, 96, 1100, d_qk=d_qk, d_v=d_v, out_dt=torch.bfloat16, split_kv=2, sdpa_kwargs=_MASKS["causal"]), d_qk=d_qk)


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["padded_pitch", "bhsd", "unaligned_pitch"])
@pytest.mark.parametrize("out_dt", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "e4m3"])
@torch_fork_set_rng(seed=113)
def test_mxfp8_sm120_graph_native_layout(layout, out_dt):
    """Aligned padded and BHSD layouts bind natively; byte-unaligned pitches decline."""
    if layout == "unaligned_pitch":
        with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=r"q: native dense forward.*16-byte outer-stride alignment"):
            _run(2, 4, 2, 200, 330, out_dt=out_dt, layout=layout, sdpa_kwargs=_MASKS["causal"])
        return
    _check(_run(2, 4, 2, 200, 330, out_dt=out_dt, layout=layout, sdpa_kwargs=_MASKS["causal"]))


@pytest.mark.L0
@pytest.mark.parametrize(
    "sf_o_layout,scale_o,tiles",
    [("planes", None, None), ("token_major", None, None), ("planes", 2.5, None), ("planes", None, (64, 128))],
    ids=["planes", "token_major", "scale_o", "tile64"],
)
@torch_fork_set_rng(seed=137)
def test_mxfp8_sm120_block_scaled_output(sf_o_layout, scale_o, tiles):
    """MXFP8 O (E4M3 + one UE8M0 scale per 32 d elements in sf_o) against the reference, within three times the
    block-quantization floor; S_q = 300 leaves pad rows in the last 128-row scale atom, which must come back zero."""
    from sdpa.block_scale_o_ref import quantize_o_mxfp8

    res = _run(
        2, 4, 2, 300, 256, out_dt=torch.float8_e4m3fn, with_stats=False, tiles=tiles, sf_o_layout=sf_o_layout, scale_o=scale_o, sdpa_kwargs=_MASKS["causal"]
    )
    so = 1.0 if scale_o is None else scale_o
    floor = (quantize_o_mxfp8(res.reference * so)[2] / so - res.reference).abs().max().item()
    diff = (res.output - res.reference).abs().max().item()
    tol = max(5e-2 * max(1.0, res.reference.abs().max().item()), 3.0 * floor)
    assert torch.isfinite(res.output).all(), "non-finite O (NaN = an element never written)"
    assert diff <= tol, f"max|O-ref|={diff:.4f} > {tol:.4f} (quantization floor {floor:.4f})"
    assert res.sf_o_pad_ok, "sf_o pad rows past S_q must be zero"
    assert_amax_o(res.amax_o, res.reference.abs().max().item(), torch_itype=res.input_dtype, torch_otype=torch.float8_e4m3fn, o_gpu=res.output)


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 512])
@torch_fork_set_rng(seed=79)
def test_mxfp8_sm120_execute_reads_no_device_memory(d):
    """Rule 3: execute performs no blocking device-to-host read, on the general and d512 templates."""
    _check(_run(1, 2, 2, 128, 128, d_qk=d, sync_debug=True), d_qk=d)


@pytest.mark.L1
@torch_fork_set_rng(seed=83)
def test_mxfp8_sm120_long_sequence():
    _check(_run(1, 4, 2, 2048, 2048, sdpa_kwargs=_MASKS["causal"]))


@pytest.mark.L1
@torch_fork_set_rng(seed=89)
def test_mxfp8_sm120_e5m2_wide_heads():
    _check(_run(2, 4, 2, 384, 512, d_qk=256, d_v=256, in_key="e5m2", seq_lens_kv=[512, 300], sdpa_kwargs=_MASKS["causal_br"]), 256)


# --- d512 flavor: both head dims in (256, 512] on 512 x 512 tiles, 64 x 64 CTA -------------------
# Two warps split each 16-row Q slab's head dim, so every warp reads its own two scale planes of Q,
# K and V; shapes keep S > 128 and B * H > 1 so a plane, tile or head misaddressed scale shows up.


@pytest.mark.L0
@pytest.mark.parametrize("mask", list(_MASKS))
@pytest.mark.parametrize("in_key", ["e4m3", pytest.param("e5m2", marks=pytest.mark.L1)])
@torch_fork_set_rng(seed=211)
def test_mxfp8_sm120_d512_masks(mask, in_key):
    _check(_run(2, 4, 2, 192, 320, d_qk=512, in_key=in_key, sdpa_kwargs=_MASKS[mask]), d_qk=512)


@pytest.mark.L0
@pytest.mark.parametrize(
    "d_qk,d_v",
    [(272, 496), (512, 272), (496, 512), (384, 320), pytest.param(448, 448, marks=pytest.mark.L1), pytest.param(320, 320, marks=pytest.mark.L1)],
    ids=lambda d: str(d),
)
@torch_fork_set_rng(seed=223)
def test_mxfp8_sm120_d512_envelope(d_qk, d_v):
    """Head dims inside (256, 512] ride the 512 x 512 tiles with TMA zero-fill. 3-plane scale tiles
    (<= 384) have no fourth plane, and the NaN-encoded scale blocks of the zero-filled Q/K columns and
    V-scale rows past d_v stay out of O, Stats and Amax_O."""
    res = _run(2, 4, 2, 192, 320, d_qk=d_qk, d_v=d_v, sdpa_kwargs=_MASKS["causal"], poison_v_sf_pad_cols=d_v % 128 != 0, poison_qk_sf_pad_blocks=True)
    _check(res, d_qk=d_qk)


@pytest.mark.L0
@pytest.mark.parametrize(
    "out_dt", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, pytest.param(torch.float16, marks=pytest.mark.L1)], ids=["bf16", "e4m3", "e5m2", "fp16"]
)
@torch_fork_set_rng(seed=227)
def test_mxfp8_sm120_d512_o_dtype(out_dt):
    _check(_run(1, 4, 2, 65, 193, d_qk=512, out_dt=out_dt, sdpa_kwargs=_MASKS["causal"]), d_qk=512)


@pytest.mark.L0
@torch_fork_set_rng(seed=228)
def test_mxfp8_sm120_d512_no_stats():
    """Without a Stats output the LSE store compiles out; O and Amax_O are unchanged."""
    _check(_run(1, 8, 2, 65, 193, d_qk=512, out_dt=torch.float8_e4m3fn, with_stats=False, sdpa_kwargs=_MASKS["causal"]), d_qk=512)


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["padded_pitch", "bhsd", "unaligned_pitch"])
@torch_fork_set_rng(seed=230)
def test_mxfp8_sm120_d512_native_layout(layout):
    """D512 writes quantized output through native aligned padded or BHSD strides."""
    if layout == "unaligned_pitch":
        with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=r"q: native dense forward.*16-byte outer-stride alignment"):
            _run(1, 4, 2, 65, 193, d_qk=512, out_dt=torch.float8_e4m3fn, layout=layout, sdpa_kwargs=_MASKS["causal"])
        return
    _check(_run(1, 4, 2, 65, 193, d_qk=512, out_dt=torch.float8_e4m3fn, layout=layout, sdpa_kwargs=_MASKS["causal"]), d_qk=512)


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal"])
@torch_fork_set_rng(seed=229)
def test_mxfp8_sm120_d512_negative_scale(mask):
    _check(_run(2, 4, 2, 130, 200, d_qk=512, scale=-0.7 / math.sqrt(512), sdpa_kwargs=_MASKS[mask]), d_qk=512)


@pytest.mark.L0
@torch_fork_set_rng(seed=233)
def test_mxfp8_sm120_d512_poisoned_pad_sink():
    """Per-batch Q/KV lengths over NaN-poisoned pad data and scales, including a batch with no key and
    one with no query, with attention sinks: the V tail guard picks its blocks at both 64-key halves of
    the scale span."""
    sinks = torch.randn(1, 8, 1, 1, device="cuda")
    res = _run(3, 8, 2, 192, 256, d_qk=512, seq_lens_q=[192, 100, 0], seq_lens_kv=[256, 77, 129], sinks=sinks, poison_pad=True)
    _check(res, d_qk=512)


@pytest.mark.L0
@torch_fork_set_rng(seed=239)
def test_mxfp8_sm120_d512_pack_gqa():
    """MQA packed as 64 heads x 1 token per 64-row tile: each lane's Q scale row follows its (token, head)."""
    _check(_run(1, 64, 1, 65, 193, d_qk=512, pack_gqa=True, sdpa_kwargs=_MASKS["causal"]), d_qk=512)


@pytest.mark.L0
@torch_fork_set_rng(seed=241)
def test_mxfp8_sm120_d512_persistent_units(monkeypatch):
    """Dense d512 CTAs are persistent: with two CTAs each walks several units, re-arming the K/V barriers
    and reloading the Q scale words and V scale tile per unit."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    monkeypatch.setattr(SdpaFwdDslSm120, "_persistent_ctas", lambda self, device: 2)
    _check(_run(2, 8, 2, 193, 193, d_qk=512, sdpa_kwargs=_MASKS["causal"]), d_qk=512)


@pytest.mark.L0
@pytest.mark.parametrize("split_kv", [2, 3])
@torch_fork_set_rng(seed=251)
def test_mxfp8_sm120_d512_split_kv(split_kv):
    _check(_run(1, 4, 1, 96, 1100, d_qk=512, out_dt=torch.bfloat16, split_kv=split_kv, sdpa_kwargs=_MASKS["causal"]), d_qk=512)


@pytest.mark.L0
@pytest.mark.parametrize("h_q,sinks", [(64, False), (64, True), pytest.param(128, False, marks=pytest.mark.L1)], ids=["h64", "h64_sink", "h128"])
@torch_fork_set_rng(seed=257)
def test_mxfp8_sm120_d512_mqa_window128(h_q, sinks):
    """MQA under a 128-token causal window, the shape the d512 flavor's tile and unit walk are tuned for."""
    sink_t = torch.randn(1, h_q, 1, 1, device="cuda") if sinks else None
    _check(_run(1, h_q, 1, 320, 320, d_qk=512, sinks=sink_t, sdpa_kwargs=dict(use_causal_mask=True, diagonal_band_left_bound=129)), d_qk=512)


@pytest.mark.L0
@pytest.mark.parametrize(
    "d_qk,d_v,mask,stats,sink,poison",
    [
        (512, 512, "causal", "token_major", False, False),
        (512, 512, "swa", "head_major", True, True),
        (384, 320, "causal", "head_major", False, True),
        (272, 336, "causal_br", "token_major", True, False),
    ],
    ids=["512_causal", "512_swa_sink_poison", "384x320_head_major_poison", "272x336_sink"],
)
@torch_fork_set_rng(seed=263)
def test_mxfp8_sm120_d512_thd(d_qk, d_v, mask, stats, sink, poison, monkeypatch):
    """THD packs each sequence's scale tiles after every earlier one's, 3 or 4 planes per tile as the head
    dim needs. Two persistent CTAs claim several units each, so the K/V barriers carry their phase from
    one unit to the next and every unit reloads its scale coordinates."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    monkeypatch.setattr(SdpaFwdDslSm120, "_persistent_ctas", lambda self, device: 2)
    sinks = torch.randn(1, 4, 1, 1, device="cuda") if sink else None
    res = _run_thd(*_THD_LENS, 4, 2, d_qk=d_qk, d_v=d_v, stats=stats, sinks=sinks, poison_pad=poison, sdpa_kwargs=_MASKS[mask])
    _check(res, d_qk=d_qk)


@pytest.mark.L0
def test_mxfp8_sm120_d512_flavor_routing():
    """Both head dims in (256, 512] run the d512 template; any wider or one-sided pair declines."""
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
    from cudnn.sdpa.fwd.config_sm120 import FP8_D512_TILE, TemplateParams

    def _api(d_qk, d_v):
        q, k = (torch.zeros(1, 128, 4, d_qk, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2) for _ in range(2))
        v = torch.zeros(1, 128, 4, d_v, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
        o = torch.empty(1, 128, 4, d_v, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        return SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o)

    for d_qk, d_v in ((512, 512), (272, 496), (384, 320)):
        api = _api(d_qk, d_v)
        assert api.check_support() and api.flavor == (512, 512) and (api.q_tile, api.kv_tile) == FP8_D512_TILE, (d_qk, d_v)
    for d_qk, d_v in ((256, 512), (512, 256), (528, 512)):
        with pytest.raises(ValueError):
            _api(d_qk, d_v).check_support()
    mod = api_dsl._load_sm120_kernel_module((512, 512), TemplateParams(dtype_qkv=DTYPE_E4M3, q_tile=64, kv_tile=64), fp8=True)
    assert mod.__file__.replace("\\", "/").endswith("sm120/prefill_d512_mxfp8.py")
    with pytest.raises(ValueError, match="head dimensions must both be in"):
        mod.compile(compute_capability=torch.cuda.get_device_capability(), b=1, qh=1, kh=1, sq=128, skv=128, d_qk=256, d_v=256, has_lse=False)


# --- direct adapter --------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("d", [64, 256, 512])
def test_mxfp8_sm120_stats_log2_direct(d):
    """Both Stats bases agree with analytic constant logits; O is unchanged between them."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    b, h, s = 1, 2, 256
    qf = torch.full((b, h, s, d), 0.125, device="cuda")
    kf = torch.full_like(qf, 0.25)
    vf = torch.full((b, h, s, d), 0.5, device="cuda")
    q_op = _quantize(qf, torch.float8_e4m3fn, columnwise=False)
    k_op = _quantize(kf, torch.float8_e4m3fn, columnwise=False)
    v_op = _quantize(vf, torch.float8_e4m3fn, columnwise=True)
    sfq, sfk, sfv = q_op.sf, k_op.sf, v_op.sf
    expected = q_op.dequant[0, 0, 0].double().dot(k_op.dequant[0, 0, 0].double()) * d**-0.5 + math.log(s)
    q, k, v = _bshd(q_op.data), _bshd(k_op.data), _bshd(v_op.data)
    outputs = []
    for log2 in (False, True):
        o = torch.empty(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)
        lse = torch.full((b, h, s), float("nan"), device="cuda")
        api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, scale_softmax=d**-0.5, stats_log2=log2)
        assert api.check_support()
        api.compile()
        workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=workspace)
        torch.cuda.synchronize()
        want = expected * (math.log2(math.e) if log2 else 1.0)
        torch.testing.assert_close(lse, torch.full_like(lse, want.item()), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(o, torch.full_like(o, 0.5), atol=1e-2, rtol=1e-2)
        outputs.append(o)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


@pytest.mark.L0
def test_mxfp8_sm120_direct_rejects_per_tensor_descales():
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    b, h, s, d = 1, 2, 128, 128
    q_op = _quantize(torch.randn(b, h, s, d, device="cuda"), torch.float8_e4m3fn, columnwise=False)
    v_op = _quantize(torch.randn(b, h, s, d, device="cuda"), torch.float8_e4m3fn, columnwise=True)
    q, sfq, v, sfv = _bshd(q_op.data), q_op.sf, _bshd(v_op.data), v_op.sf
    o = torch.empty(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=q, sample_v=v, sample_o=o)
    assert api.check_support()
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    one = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="block scale factors"):
        api.execute(q_tensor=q, k_tensor=q, v_tensor=v, o_tensor=o, sf_q=sfq, sf_k=sfq, sf_v=sfv, descale_q=one, workspace=workspace)


@pytest.mark.L0
def test_mxfp8_sm120_direct_d256_admits_both_kv_tiles():
    """(256, 256) heads accept an explicit tile_n of 64 or 128; an unset one picks a kernel-table tile."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
    from cudnn.sdpa.fwd.config_sm120 import tile_domain

    def _api(d_qk, d_v, **kw):
        q, k = (torch.zeros(1, 128, 4, d_qk, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2) for _ in range(2))
        v = torch.zeros(1, 128, 4, d_v, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
        o = torch.empty(1, 128, 4, d_v, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        return SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, **kw)

    for d_qk, d_v in ((256, 256), (240, 240), (256, 240)):
        for tile_n in (64, 128):
            api = _api(d_qk, d_v, tile_n=tile_n)
            assert api.check_support() and api.kv_tile == tile_n, (d_qk, d_v, tile_n)
        api = _api(d_qk, d_v)
        assert api.check_support() and (api.q_tile, api.kv_tile) in tile_domain(d_qk, d_v, fp8=True), (d_qk, d_v)


# --- engine admission -------------------------------------------------------------------------


def _graph_offers(d_qk=128, d_v=128):
    """Build an sdpa_mxfp8 graph declaration only (no buffers) and report whether the engine is ranked."""
    import cudnn

    B, H, S = 2, 2, 256
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=[B, H, S, d_qk], stride=[S * H * d_qk, d_qk, H * d_qk, 1], data_type=cudnn.data_type.FP8_E4M3)
    k = g.tensor(dim=[B, H, S, d_qk], stride=[S * H * d_qk, d_qk, H * d_qk, 1], data_type=cudnn.data_type.FP8_E4M3)
    v = g.tensor(dim=[B, H, S, d_v], stride=[S * H * d_v, d_v, H * d_v, 1], data_type=cudnn.data_type.FP8_E4M3)

    qk_sf = (B, H, S, _qk_sf_cols(d_qk))
    kw = dict(
        q=q, k=k, v=v, descale_q=_sf_tensor(g, qk_sf), descale_k=_sf_tensor(g, qk_sf), descale_v=_sf_tensor(g, (B, H, S // _BLOCK, _cdiv(d_v, 128) * 128))
    )
    kw.update(attn_scale=d_qk**-0.5, generate_stats=False)
    o, _, _ = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim([B, H, S, d_v]).set_stride([S * H * d_v, d_v, H * d_v, 1]).set_data_type(cudnn.data_type.HALF)
    try:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
    except cudnn.cudnnGraphNotSupportedError:
        # No engine at all serves the graph: for "this engine must not claim it" that is the same answer.
        return False
    return offers_engine(g, _ENGINE)


@pytest.mark.L0
@pytest.mark.parametrize(
    "kwargs,offered",
    [
        (dict(d_qk=128, d_v=128), True),
        (dict(d_qk=512, d_v=512), True),
        (dict(d_qk=384, d_v=320), True),
        (dict(d_qk=256, d_v=512), False),
        (dict(d_qk=528, d_v=528), False),
        (dict(d_qk=24, d_v=24), False),
    ],
    ids=["d128", "d512", "d384x320", "d256x512", "d528", "off_granule"],
)
def test_mxfp8_sm120_offers_only_inside_the_envelope(kwargs, offered):
    assert _graph_offers(**kwargs) == offered


# --- THD (ragged) ----------------------------------------------------------------------------


class _SeqQuantized(NamedTuple):
    """One sequence's MXFP8-quantized operand for the THD packing."""

    data: torch.Tensor  # fp8 [1, h, s, d]
    dequant: torch.Tensor  # fp32 [1, h, s, d]
    sf_tiles: torch.Tensor  # [h, ceil(s / 128), tile_bytes], every tile's planes contiguous


class _SeqOperands(NamedTuple):
    q: _SeqQuantized
    k: _SeqQuantized
    v: _SeqQuantized


def _poison_sf_tail(sf_tiles, s, *, columnwise):
    """Write the NaN encoding 0xFF into the scales of a sequence's last tile that lie past its length ``s``:
    rowwise (Q/K) rows ``s .. tiles * 128``, columnwise (V) 32-key blocks wholly past ``s``."""
    _, tiles, size = sf_tiles.shape
    dev = sf_tiles.device
    plane = (torch.arange(size // 512, device=dev) * 512).view(1, -1, 1)
    if columnwise:
        blocks = torch.arange(_cdiv(s, _BLOCK), tiles * 4, device=dev)
        n = torch.arange(128, device=dev)
        tile = (blocks // 4).view(-1, 1, 1)
        byte = plane + ((n % 32) * 16 + ((n // 32) % 4) * 4).view(1, 1, -1) + (blocks % 4).view(-1, 1, 1)
    else:
        rows = torch.arange(s, tiles * 128, device=dev)
        rr = rows % 128
        tile = (rows // 128).view(-1, 1, 1)
        byte = plane + ((rr % 32) * 16 + ((rr // 32) % 4) * 4).view(-1, 1, 1) + torch.arange(4, device=dev).view(1, 1, -1)
    sf_tiles[:, tile.expand_as(byte), byte] = 0xFF


def _quantize_seq(t, fp8, *, columnwise, poison_tail=False) -> _SeqQuantized:
    """Per-sequence quantization for the THD packing. Columnwise F8_128x4 is plane-major across a tensor,
    so its [planes, h, tiles, 512] atoms are regrouped to keep every tile's planes contiguous.
    ``poison_tail`` writes 0xFF scales past the sequence's length (``_poison_sf_tail``)."""
    _, h, s, d = t.shape
    if s == 0:
        planes = _cdiv(d, 128)
        return _SeqQuantized(
            t.new_zeros((1, h, 0, d)).to(fp8), t.new_zeros((1, h, 0, d)), torch.zeros((h, 0, 512 * planes), dtype=torch.uint8, device=t.device)
        )
    op = _quantize(t, fp8, columnwise=columnwise)
    tiles, planes = _cdiv(s, 128), _cdiv(d, 128)
    sf = op.sf.reshape(planes, h, tiles, 512).permute(1, 2, 0, 3) if columnwise else op.sf
    sf = sf.contiguous().reshape(h, tiles, -1)
    if poison_tail:
        _poison_sf_tail(sf, s, columnwise=columnwise)
    return _SeqQuantized(op.data, op.dequant, sf)


def _run_thd(
    seq_lens_q,
    seq_lens_kv,
    H_q,
    H_kv,
    *,
    d_qk=128,
    d_v=None,
    in_key="e4m3",
    out_dt=torch.float16,
    sdpa_kwargs=None,
    sinks=None,
    stats=None,
    tiles=None,
    poison_pad=False,
):
    """Packed [T, H, D] Q/K/V/O with ragged offsets in dense-capacity storage whose tail past the packed
    total is NaN-poisoned, per-batch lengths, and packed per-sequence-tile scale factors. ``stats``:
    None, "token_major" ((T, H)) or "head_major" ((H, T_capacity)) ragged Stats. ``poison_pad`` writes
    0xFF scales past every sequence's Q and KV length inside its last scale tile."""
    import cudnn

    dev = "cuda"
    d_v = d_qk if d_v is None else d_v
    sdpa_kwargs = sdpa_kwargs or {}
    scale = d_qk**-0.5
    fp8 = _FP8[in_key]
    B = len(seq_lens_q)
    S_max_q, S_max_kv = max(seq_lens_q), max(seq_lens_kv)
    T_q = sum(seq_lens_q)
    cu_q = [0] + list(torch.tensor(seq_lens_q).cumsum(0).tolist())
    cu_k = [0] + list(torch.tensor(seq_lens_kv).cumsum(0).tolist())
    gen = torch.Generator(device=dev).manual_seed(torch.initial_seed() % (2**31))
    seqs = []
    for s_q, s_kv in zip(seq_lens_q, seq_lens_kv):
        qf = torch.randn(1, H_q, s_q, d_qk, device=dev, generator=gen) * 0.5
        kf = torch.randn(1, H_kv, s_kv, d_qk, device=dev, generator=gen) * 0.5
        vf = torch.randn(1, H_kv, s_kv, d_v, device=dev, generator=gen) * 0.5
        if s_q:
            qf = qf * _block_spread(qf.shape, 3, -2, 2, gen)
        if s_kv:
            kf = kf * _block_spread(kf.shape, 3, -2, 2, gen)
            vf = vf * _block_spread(vf.shape, 2, -3, 3, gen)
        seqs.append(
            _SeqOperands(
                q=_quantize_seq(qf, fp8, columnwise=False, poison_tail=poison_pad),
                k=_quantize_seq(kf, fp8, columnwise=False, poison_tail=poison_pad),
                v=_quantize_seq(vf, fp8, columnwise=True, poison_tail=poison_pad),
            )
        )

    def pack_tokens(role):
        return torch.cat([getattr(seq, role).data.squeeze(0).permute(1, 0, 2) for seq in seqs], dim=0)

    def dense_buf(packed, s_max, h, d):
        stride = (s_max * h * d, d, h * d, 1)
        stor = torch.full((B * s_max * h * d,), float("nan"), device=dev).to(packed.dtype)
        stor[: packed.numel()] = packed.reshape(-1)
        return stor.as_strided((B, h, s_max, d), stride), stride

    q_gpu, stride_q = dense_buf(pack_tokens("q"), S_max_q, H_q, d_qk)
    k_gpu, stride_k = dense_buf(pack_tokens("k"), S_max_kv, H_kv, d_qk)
    v_gpu, stride_v = dense_buf(pack_tokens("v"), S_max_kv, H_kv, d_v)
    sfq, sfk, sfv = (torch.cat([getattr(seq, role).sf_tiles for seq in seqs], dim=1).contiguous() for role in ("q", "k", "v"))
    stride_o = (S_max_q * H_q * d_v, d_v, H_q * d_v, 1)
    o_stor = _nan_output((B * S_max_q * H_q * d_v,), out_dt)
    o_gpu = o_stor.as_strided((B, H_q, S_max_q, d_v), stride_o)
    amax = torch.full((1, 1, 1, 1), float("nan"), device=dev)

    io = getattr(cudnn.data_type, _CUDNN_ITYPE[in_key])
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H_q, S_max_q, d_qk], stride=list(stride_q), data_type=io)
    tk = g.tensor(dim=[B, H_kv, S_max_kv, d_qk], stride=list(stride_k), data_type=io)
    tv = g.tensor(dim=[B, H_kv, S_max_kv, d_v], stride=list(stride_v), data_type=io)
    slq = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    offsets = {
        "q": torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * d_qk,
        "k": torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * d_qk,
        "v": torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * d_v,
        "o": torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * d_v,
    }
    offsets = {name: t.view(B + 1, 1, 1, 1) for name, t in offsets.items()}
    ro = {name: g.tensor_like(t) for name, t in offsets.items()}
    for name, t in (("q", tq), ("k", tk), ("v", tv)):
        t.set_ragged_offset(ro[name])
    sq_h, skv_h = g.tensor_like(slq), g.tensor_like(slk)

    dq = _sf_tensor(g, (B, H_q, _cdiv(S_max_q, 128) * 128, _qk_sf_cols(d_qk)))
    dk = _sf_tensor(g, (B, H_kv, _cdiv(S_max_kv, 128) * 128, _qk_sf_cols(d_qk)))
    dv = _sf_tensor(g, (B, H_kv, _cdiv(S_max_kv, 128) * 4, _cdiv(d_v, 128) * 128))
    kw = dict(q=tq, k=tk, v=tv, descale_q=dq, descale_k=dk, descale_v=dv, attn_scale=scale, generate_stats=stats is not None)
    kw.update(use_padding_mask=True, seq_len_q=sq_h, seq_len_kv=skv_h, **sdpa_kwargs)
    vp = {tq: q_gpu, tk: k_gpu, tv: v_gpu, dq: sfq, dk: sfk, dv: sfv, sq_h: slq, skv_h: slk}
    vp.update({ro[name]: offsets[name] for name in offsets})
    if sinks is not None:
        st = g.tensor_like(sinks)
        kw["sink_token"] = st
        vp[st] = sinks
    o, stats_t, amax_o = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim([B, H_q, S_max_q, d_v]).set_stride(list(stride_o)).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    o.set_ragged_offset(ro["o"])
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    stats_stor = None
    if stats is not None:
        stats_stor = torch.full((B * S_max_q * H_q,), float("nan"), device=dev)
        stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        if stats == "token_major":
            stats_t.set_dim((B, H_q, S_max_q, 1)).set_stride((S_max_q * H_q, 1, H_q, 1))
            stats_ro_t = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q).view(B + 1, 1, 1, 1)
        else:
            stats_t.set_dim((B, H_q, S_max_q, 1)).set_stride((S_max_q, B * S_max_q, 1, 1))
            stats_ro_t = torch.tensor(cu_q, dtype=torch.int64, device=dev).view(B + 1, 1, 1, 1)
        stats_ro = g.tensor_like(stats_ro_t)
        stats_t.set_ragged_offset(stats_ro)
        vp.update({stats_ro: stats_ro_t, stats_t: stats_stor})
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, _ENGINE, tiles=tiles)
    g.check_support()
    g.build_plans()
    vp.update({o: o_gpu, amax_o: amax})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    o_ref = torch.zeros(T_q, H_q, d_v, device=dev)
    lse_ref = torch.full((T_q, H_q), float("-inf"), device=dev)
    ref_kw = _ref_kwargs(sdpa_kwargs)
    for b, seq in enumerate(seqs):
        if seq_lens_q[b] == 0:
            continue
        ob, lb = _ref(seq.q.dequant, seq.k.dequant, seq.v.dequant, scale=scale, sinks=sinks, **ref_kw)
        o_ref[cu_q[b] : cu_q[b + 1]] = ob.squeeze(0).permute(1, 0, 2)
        lse_ref[cu_q[b] : cu_q[b + 1]] = lb.squeeze(0).permute(1, 0)
    o_out = o_stor[: T_q * H_q * d_v].reshape(T_q, H_q, d_v)
    lse_out = None
    if stats == "token_major":
        lse_out = stats_stor[: T_q * H_q].reshape(T_q, H_q)
    elif stats == "head_major":
        lse_out = stats_stor.reshape(H_q, B * S_max_q)[:, :T_q].transpose(0, 1)
    return _RunResult(o_out, o_ref, fp8, amax, lse_out, lse_ref if stats is not None else None)


_THD_LENS = ([300, 0, 17, 129, 256], [256, 40, 0, 200, 384])


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["none", "causal", "causal_br", "swa"])
@torch_fork_set_rng(seed=97)
def test_mxfp8_sm120_thd(mask):
    """Each sequence's scale tiles start after every earlier sequence's ceil(S / 128) tiles; empty
    sequences contribute none."""
    _check(_run_thd(*_THD_LENS, 4, 2, sdpa_kwargs=_MASKS[mask]))


@pytest.mark.L0
@torch_fork_set_rng(seed=149)
def test_mxfp8_sm120_thd_many_sequences():
    """More sequences than a warp has lanes: each lane sums the tile counts of several earlier sequences
    (empty and multi-tile ones included) before the warp combines them."""
    lens_q = [(0, 17, 130, 64, 1, 257)[i % 6] for i in range(70)]
    lens_kv = [(31, 0, 129, 200, 64, 300)[i % 6] for i in range(70)]
    _check(_run_thd(lens_q, lens_kv, 2, 1, sdpa_kwargs=_MASKS["causal_br"]))


@pytest.mark.L0
@pytest.mark.parametrize("stats", ["token_major", "head_major"])
@torch_fork_set_rng(seed=101)
def test_mxfp8_sm120_thd_stats_sink(stats):
    sinks = torch.randn(1, 4, 1, 1, device="cuda")
    _check(_run_thd(*_THD_LENS, 4, 1, stats=stats, sinks=sinks, sdpa_kwargs=_MASKS["causal"]))


@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(64, 64), (256, 256), (192, 128), (80, 112)], ids=lambda d: str(d))
@torch_fork_set_rng(seed=103)
def test_mxfp8_sm120_thd_head_dims(d_qk, d_v):
    """A two-plane V tile (D = 256) keeps its planes contiguous per sequence tile under THD."""
    _check(_run_thd([140, 70, 300], [200, 33, 260], 2, 2, d_qk=d_qk, d_v=d_v, out_dt=torch.bfloat16, stats="token_major"), d_qk=d_qk)


@pytest.mark.L0
@pytest.mark.parametrize("tiles", [(128, 128), (128, 64)], ids=lambda t: f"kv{t[1]}")
@torch_fork_set_rng(seed=109)
def test_mxfp8_sm120_thd_poisoned_pad(tiles):
    """Each packed sequence's last scale tile carries 0xFF (NaN) scales past its Q and KV length: the
    Q-scale row guard, the V-scale tail guard and the mask's select keep them out of O, Stats and
    Amax_O."""
    _check(_run_thd(*_THD_LENS, 4, 2, stats="token_major", tiles=tiles, poison_pad=True, sdpa_kwargs=_MASKS["causal"]))


@pytest.mark.L1
@torch_fork_set_rng(seed=107)
def test_mxfp8_sm120_thd_fp8_out():
    _check(_run_thd(*_THD_LENS, 4, 2, out_dt=torch.float8_e4m3fn, sdpa_kwargs=_MASKS["causal_br"]))
