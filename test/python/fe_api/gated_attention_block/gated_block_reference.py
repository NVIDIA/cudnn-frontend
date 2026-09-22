# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch references for the gated attention block.

Two of them, for two different jobs — do not use one for the other's job:

``gated_attention_block_reference``
    The **correctness oracle**. FP32 throughout, chunked over query tiles so it
    does not materialize an ``[S, S]`` score matrix, and it returns every
    intermediate so a failure localizes to a stage instead of to "the block".
    Slow by construction; that is fine.

``qk_norm_rope_reference``
    The oracle for stages (2)+(3) alone — the fused, single-rounding form the
    FROST kernel implements.

``gated_attention_block_baseline``
    The **perf baseline**: the same math in bf16 through the ops a framework
    would actually call — ``F.linear`` (cuBLAS) and
    ``F.scaled_dot_product_attention`` (cuDNN/FA). This is the five-launch shape
    the fused block has to beat, and two of those five are already-fused work we
    would be replacing rather than adding to.

Both follow the contracts in ``python/cudnn/gated_attention_block/api.py``: the
``(Q, GATE, K, V)`` column order of the fused projection, and the NeoX /
``rotate_half`` RoPE tables of shape ``[B, S, ROPE_DIM]`` with duplicated halves.

Geometry is Qwen3.5's gated attention. The forward order — project, QK-RMSNorm
(V unnormed), partial RoPE, SDPA, ``* sigmoid(GATE)``, out-project — is the
model's, and the gate multiply lands BEFORE the out projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Geometry + input construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefGeometry:
    """Mirror of ``GatedAttentionBlockGeometry``, kept independent on purpose.

    A reference that imports the thing it validates can agree with a bug.
    """

    d_model: int
    h_q: int
    h_kv: int
    d_head: int
    rope_dim: int
    qk_norm_eps: float = 1e-6
    attn_scale: Optional[float] = None
    is_causal: bool = True
    rope_base: float = 1_000_000.0
    # Appended LAST (this is not a field-for-field mirror of the block's
    # Geometry). False: RoPE-only Q/K -- no RMSNorm, ``make_inputs`` yields
    # ``None`` norm weights, the oracles skip the norm and return no rstd.
    qk_norm: bool = True

    @property
    def scale(self) -> float:
        return self.attn_scale if self.attn_scale is not None else float(self.d_head) ** -0.5

    @property
    def n_qkvg(self) -> int:
        return (2 * self.h_q + 2 * self.h_kv) * self.d_head

    @property
    def offsets(self) -> Tuple[int, int, int, int]:
        q = 0
        g = q + self.h_q * self.d_head
        k = g + self.h_q * self.d_head
        v = k + self.h_kv * self.d_head
        return (q, g, k, v)


# One full-attention layer of Qwen3.5-397B at TP=1: d_model 4096, 32 query heads
# over 2 KV heads (GQA 16x), head dim 256 of which the leading 64 rotate.
GEOMETRY_D4096_H32_KV2_D256 = RefGeometry(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)

# A shrunk shape with the same structure, for oracle-speed correctness runs.
GEOMETRY_SMALL = RefGeometry(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16)


def build_rope_tables(
    seq_len: int,
    rope_dim: int,
    *,
    base: float = 1_000_000.0,
    batch: int = 1,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos, sin`` of shape ``[batch, seq_len, rope_dim]``, halves duplicated.

    This is byte-for-byte what HF ``apply_rotary_pos_emb`` and vLLM's default
    ``RotaryEmbedding`` consume, which is the point: a caller passes what it
    already holds and no conversion happens on the hot path.

    ``positions`` (``[batch, seq_len]``) exists so an mRoPE caller can supply
    per-section positions. The block never learns which flavour of RoPE built
    the table — that is what keeps mRoPE, YaRN and NTK scaling out of the kernel.
    """
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    half = rope_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    if positions is None:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).expand(batch, seq_len)
    positions = positions.to(device=device, dtype=torch.float32)
    freqs = positions[..., None] * inv_freq  # [B, S, rope_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, rope_dim]
    assert emb.shape[-1] == 2 * half
    return emb.cos().to(dtype), emb.sin().to(dtype)


def make_inputs(
    geom: RefGeometry,
    batch: int,
    seq_len: int,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> dict:
    """Random inputs in the block's declared layouts. Weights are ``nn.Linear``
    shaped (``[out, in]``), so the block's GEMMs read them transposed.

    ``geom.qk_norm=False`` yields ``None`` for both norm weights -- the block
    (and every oracle here) takes ``None`` in those slots for RoPE-only Q/K.
    The random draws are identical either way (the weights are constants, not
    draws), so norm-on and norm-off tests see the same ``h`` / ``w_qkvg`` / ``w_o``."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, std=0.02):
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * std).to(dtype)

    cos, sin = build_rope_tables(seq_len, geom.rope_dim, base=geom.rope_base, batch=batch, device=device, dtype=dtype)
    norm_w = (lambda: torch.ones(geom.d_head, device=device, dtype=dtype)) if geom.qk_norm else (lambda: None)
    return {
        "h": randn(batch, seq_len, geom.d_model, std=1.0),
        "w_qkvg": randn(geom.n_qkvg, geom.d_model),
        "w_q_norm": norm_w(),
        "w_k_norm": norm_w(),
        "cos": cos,
        "sin": sin,
        "w_o": randn(geom.d_model, geom.h_q * geom.d_head),
    }


# ---------------------------------------------------------------------------
# Stage primitives — shared by both references so they cannot drift
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Rotate ``x[..., :rope_dim]``, pass ``x[..., rope_dim:]`` through.

    ``x`` is ``[B, S, H, D]``; ``cos`` / ``sin`` are ``[B, S, rope_dim]`` and
    broadcast across heads.
    """
    if rope_dim == 0:
        return x
    if rope_dim > x.shape[-1]:
        raise ValueError(f"rope_dim {rope_dim} exceeds head dim {x.shape[-1]}")
    c = cos[:, :, None, :].to(x.dtype)
    s = sin[:, :, None, :].to(x.dtype)
    rot, passthrough = x[..., :rope_dim], x[..., rope_dim:]
    rot = rot * c + _rotate_half(rot) * s
    return torch.cat((rot, passthrough), dim=-1) if passthrough.shape[-1] else rot


def qk_norm_rope_reference(
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    eps: float,
    *,
    qk_norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stages (2)+(3) fused, the way the FROST kernel computes them.

    FP32 norm, FP32 rotation, **one** cast at the end. That last part is not a
    detail: an unfused torch chain rounds to bf16 after the norm AND again after
    the rotation, so it is a slightly different function. This is the oracle the
    kernel is checked against; ``gated_attention_block_baseline`` keeps the
    two-rounding chain on purpose, because that is what a framework does.

    Returns ``(y, rstd)`` with ``y`` in ``x``'s dtype and ``rstd`` fp32 — the
    reciprocal RMS the backward needs, which is why the kernel emits it.

    ``qk_norm=False`` (the block's ``geometry.qk_norm=False``): no norm, ``w`` is
    ignored (pass ``None``), ``rstd`` is ``None``, and ``y`` is the fp32 partial
    RoPE of ``x`` cast once -- the passthrough dims ``[rope_dim, D)`` are then
    bit-identical to ``x``'s (widen + narrow of the same value), which is what
    the kernels are held to.
    """
    x32 = x.float()
    if qk_norm:
        if w is None:
            raise ValueError("qk_norm=True needs a [D] norm weight; pass qk_norm=False for RoPE-only Q/K")
        rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        y = x32 * rstd * w.float()
    else:
        rstd = None
        y = x32
    if rope_dim:
        c = cos[:, :, None, :rope_dim].float()
        s = sin[:, :, None, :rope_dim].float()
        rot, passthrough = y[..., :rope_dim], y[..., rope_dim:]
        rot = rot * c + _rotate_half(rot) * s
        y = torch.cat((rot, passthrough), dim=-1) if passthrough.shape[-1] else rot
    return y.to(x.dtype), (None if rstd is None else rstd.squeeze(-1).float())


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row RMSNorm over the last dim, computed in fp32.

    Returns ``(y, rstd)``; ``rstd`` is the ``[..., 1]``-squeezed fp32 reciprocal
    RMS the backward needs, which is why it comes out of the forward at all.
    """
    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    y = (x32 * rstd) * w.float()
    return y.to(x.dtype), rstd.squeeze(-1).float()


def split_qkvg(proj: torch.Tensor, geom: RefGeometry) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[B, S, N] -> (Q, GATE, K, V)`` in the block's contract column order."""
    b, s, _ = proj.shape
    o_q, o_g, o_k, o_v = geom.offsets
    hq, hkv, d = geom.h_q, geom.h_kv, geom.d_head
    q = proj[..., o_q:o_g].view(b, s, hq, d)
    gate = proj[..., o_g:o_k].view(b, s, hq, d)
    k = proj[..., o_k:o_v].view(b, s, hkv, d)
    v = proj[..., o_v:].view(b, s, hkv, d)
    return q, gate, k, v


def _key_padding_and_causal_mask(
    s_q: int,
    s_kv: int,
    *,
    is_causal: bool,
    seq_lens: Optional[torch.Tensor],
    batch_index: int,
    q_lo: int,
    device,
) -> Optional[torch.Tensor]:
    """``[q_chunk, s_kv]`` bool mask, True where a column is ALLOWED."""
    q_idx = torch.arange(q_lo, q_lo + s_q, device=device)
    k_idx = torch.arange(s_kv, device=device)
    allowed = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
    if is_causal:
        allowed &= k_idx[None, :] <= q_idx[:, None]
    if seq_lens is not None:
        allowed &= k_idx[None, :] < int(seq_lens[batch_index])
    return allowed


# ---------------------------------------------------------------------------
# The FP32 oracle
# ---------------------------------------------------------------------------


@dataclass
class RefOutputs:
    """Everything the oracle computed, so a mismatch localizes to a stage."""

    out: torch.Tensor  # [B, S, d_model]
    q_pre: torch.Tensor  # [B, S, H_q, D]   post-projection, pre-norm
    k_pre: torch.Tensor  # [B, S, H_kv, D]
    gate: torch.Tensor  # [B, S, H_q, D]   pre-sigmoid
    v: torch.Tensor  # [B, S, H_kv, D]
    q: torch.Tensor  # [B, S, H_q, D]   post-norm, post-RoPE
    k: torch.Tensor  # [B, S, H_kv, D]
    rstd_q: Optional[torch.Tensor]  # [B, S, H_q]  fp32; None under geom.qk_norm=False
    rstd_k: Optional[torch.Tensor]  # [B, S, H_kv] fp32; idem
    o: torch.Tensor  # [B, S, H_q, D]   SDPA output, PRE-gate
    o_gated: torch.Tensor  # [B, S, H_q, D]
    lse: torch.Tensor  # [B, H_q, S]  fp32, natural log


def gated_attention_block_reference(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geom: RefGeometry,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    q_chunk: int = 512,
) -> RefOutputs:
    """FP32 oracle for the whole block, chunked over query tiles.

    ``seq_lens`` (``[B]`` int32) marks per-batch valid KV length; columns at or
    beyond it are masked. It is here because the degenerate cases are the ones
    the kernel gets wrong: a row with NO allowed column must produce ``O = 0``
    and ``LSE = -inf`` exactly, never a floored denominator's ``-69.08`` and
    never accumulator residue scaled by a sigmoid.

    ``geom.qk_norm=False``: the norm weights are ``None``, stage (2) is skipped
    (RoPE only) and ``rstd_q`` / ``rstd_k`` come back ``None``.
    """
    b, s, d_model = h.shape
    if d_model != geom.d_model:
        raise ValueError(f"h last dim {d_model} != geometry d_model {geom.d_model}")
    if w_qkvg.shape != (geom.n_qkvg, geom.d_model):
        raise ValueError(f"w_qkvg must be [{geom.n_qkvg}, {geom.d_model}], got {tuple(w_qkvg.shape)}")
    if w_o.shape != (geom.d_model, geom.h_q * geom.d_head):
        raise ValueError(f"w_o must be [{geom.d_model}, {geom.h_q * geom.d_head}], got {tuple(w_o.shape)}")

    dev = h.device
    out_dtype = h.dtype
    d = geom.d_head
    rep = geom.h_q // geom.h_kv

    # (1) fused QKV+GATE projection, fp32 accumulate.
    proj = (h.float() @ w_qkvg.float().t()).to(out_dtype)
    q_pre, gate, k_pre, v = split_qkvg(proj, geom)

    # (2)+(3) QK-RMSNorm then partial RoPE -- V is NOT normed. One fp32 pass
    # with a single final rounding, matching the fused kernel. qk_norm=False:
    # RoPE only, rstd None.
    q, rstd_q = qk_norm_rope_reference(q_pre, w_q_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    k, rstd_k = qk_norm_rope_reference(k_pre, w_k_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm)

    # (4) SDPA, chunked over q tiles, fp32.
    o = torch.zeros(b, s, geom.h_q, d, device=dev, dtype=torch.float32)
    lse = torch.full((b, geom.h_q, s), float("-inf"), device=dev, dtype=torch.float32)

    k_b = k.float().repeat_interleave(rep, dim=2)  # [B, S, H_q, D]
    v_b = v.float().repeat_interleave(rep, dim=2)

    for bi in range(b):
        for lo in range(0, s, q_chunk):
            hi = min(lo + q_chunk, s)
            qc = q[bi, lo:hi].float().transpose(0, 1)  # [H_q, s_q, D]
            kc = k_b[bi].transpose(0, 1)  # [H_q, S, D]
            vc = v_b[bi].transpose(0, 1)
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * geom.scale  # [H_q, s_q, S]

            allowed = _key_padding_and_causal_mask(hi - lo, s, is_causal=geom.is_causal, seq_lens=seq_lens, batch_index=bi, q_lo=lo, device=dev)
            scores = scores.masked_fill(~allowed[None], float("-inf"))

            row_max = scores.amax(dim=-1)  # [H_q, s_q]
            dead = torch.isinf(row_max) & (row_max < 0)  # no allowed column at all
            safe_max = torch.where(dead, torch.zeros_like(row_max), row_max)
            p = torch.exp(scores - safe_max[..., None])
            p = torch.where(allowed[None], p, torch.zeros_like(p))
            denom = p.sum(dim=-1)  # [H_q, s_q]

            # SELECT, not a multiply by zero: residue can be a NaN bit pattern,
            # and NaN * 0 is NaN. Same rule the kernel epilogue must follow.
            o_chunk = torch.matmul(p, vc) / torch.where(dead, torch.ones_like(denom), denom)[..., None]
            o_chunk = torch.where(dead[..., None], torch.zeros_like(o_chunk), o_chunk)
            o[bi, lo:hi] = o_chunk.transpose(0, 1)

            lse_chunk = safe_max + torch.log(denom)
            lse[bi, :, lo:hi] = torch.where(dead, torch.full_like(lse_chunk, float("-inf")), lse_chunk)

    o = o.to(out_dtype)

    # (5) gate -- AFTER the dead-row substitution above, never before.
    o_gated = (o.float() * torch.sigmoid(gate.float())).to(out_dtype)

    # (6) out projection.
    o_flat = o_gated.reshape(b, s, geom.h_q * d)
    out = (o_flat.float() @ w_o.float().t()).to(out_dtype)

    return RefOutputs(
        out=out,
        q_pre=q_pre,
        k_pre=k_pre,
        gate=gate,
        v=v,
        q=q,
        k=k,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        o=o,
        o_gated=o_gated,
        lse=lse,
    )


# ---------------------------------------------------------------------------
# The perf baseline
# ---------------------------------------------------------------------------


def gated_attention_block_baseline(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geom: RefGeometry,
) -> torch.Tensor:
    """bf16 baseline through the ops a framework actually launches.

    Five kernels' worth of work, matching the vLLM shipped graph::

        [GEMM] fused QKV+gate -> [fused qk_norm + partial RoPE + gate copy]
        -> [SDPA] -> [elementwise gate] -> [GEMM] o_proj

    Two of those five are already-fused work the block would be REPLACING, not
    adding to, which is why this and not a six-launch strawman is the number to
    beat. The norm+RoPE here is several torch ops rather than one Triton kernel,
    so it flatters the block slightly; treat it as an upper bound on the
    baseline's launch count, and prefer the real framework graph when one is
    available.

    Measurement protocol, because these stages are individually sub-millisecond:
    build and warm every artifact OUTSIDE the timed region, size iterations so
    each window is >= 60 ms, interleave >= 7 slots and take the MEDIAN, and print
    a CONTROL PAIR (the same config timed twice). A result no larger than the
    control is noise.
    """
    b, s, _ = h.shape
    d = geom.d_head

    proj = F.linear(h, w_qkvg)
    q, gate, k, v = split_qkvg(proj, geom)

    if geom.qk_norm:  # RoPE-only blocks have no norm and no norm weights
        q, _ = rms_norm(q, w_q_norm, geom.qk_norm_eps)
        k, _ = rms_norm(k, w_k_norm, geom.qk_norm_eps)
    q = apply_partial_rope(q, cos, sin, geom.rope_dim)
    k = apply_partial_rope(k, cos, sin, geom.rope_dim)

    # [B, S, H, D] -> [B, H, S, D] for SDPA; enable_gqa broadcasts H_kv -> H_q.
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=geom.is_causal,
        scale=geom.scale,
        enable_gqa=True,
    ).transpose(1, 2)

    o_gated = o * torch.sigmoid(gate)
    return F.linear(o_gated.reshape(b, s, geom.h_q * d), w_o)


# ---------------------------------------------------------------------------
# FP8 (E4M3, static per-tensor scales) -- the shared fake-quant reference
# ---------------------------------------------------------------------------
#
# ONE copy of the recipe the FP8 block test, block_perf_table and
# block_fusion_table all need: amax scales for h / W_qkvg / W_o, e4m3 casts
# with the exact saturating semantics the kernels use, and the fp32 oracle with
# the block's own quantization points.  A harness that gates FP8 arms against
# the bf16 fp32 oracle would zero them as WRONG (e4m3 h/W alone costs cosine),
# so it gates against THIS instead.

FP8_E4M3 = torch.float8_e4m3fn
FP8_E4M3_MAX = 448.0


def amax_scale(x: torch.Tensor) -> float:
    """Static per-tensor scale from this tensor's amax (a stand-in for offline calibration)."""
    return float(FP8_E4M3_MAX / x.float().abs().amax().clamp_min(1e-8))


def quant_e4m3(x: torch.Tensor, scale: float) -> torch.Tensor:
    """``sat_e4m3(x * scale)`` -- bit-exact vs the kernels' ``cvt.rn.satfinite.e4m3x2.f32``."""
    return (x.float() * scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_E4M3)


def dequant_e4m3(x8: torch.Tensor, descale: float) -> torch.Tensor:
    return x8.float() * descale


def quantize_block_inputs(inp: dict) -> Tuple[dict, dict]:
    """``make_inputs`` output -> the same dict with ``h`` / ``w_qkvg`` / ``w_o`` as
    e4m3 (amax scales), plus ``{descale_h, descale_w_qkvg, descale_w_o}``.

    Norm weights, cos/sin stay bf16 (the block's activation dtype).  Built from
    an EXISTING input dict so a harness can hand the bf16 arms and the FP8 arms
    the same data.
    """
    s_h, s_wq, s_wo = amax_scale(inp["h"]), amax_scale(inp["w_qkvg"]), amax_scale(inp["w_o"])
    fp8 = dict(inp)
    fp8["h"], fp8["w_qkvg"], fp8["w_o"] = quant_e4m3(inp["h"], s_h), quant_e4m3(inp["w_qkvg"], s_wq), quant_e4m3(inp["w_o"], s_wo)
    return fp8, dict(descale_h=1.0 / s_h, descale_w_qkvg=1.0 / s_wq, descale_w_o=1.0 / s_wo)


def make_fp8_inputs(geom: RefGeometry, batch: int, seq_len: int, *, device: torch.device | str = "cuda", seed: int = 0) -> Tuple[dict, dict]:
    """bf16 inputs from :func:`make_inputs`, then :func:`quantize_block_inputs`."""
    return quantize_block_inputs(make_inputs(geom, batch=batch, seq_len=seq_len, device=device, dtype=torch.bfloat16, seed=seed))


def gated_attention_block_fp8_reference(
    inp: dict,
    geom: RefGeometry,
    *,
    descale_h: float,
    descale_w_qkvg: float,
    descale_w_o: float,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    scale_o: float,
    seq_lens: Optional[torch.Tensor] = None,
    fused: bool = False,
) -> torch.Tensor:
    """fp32 chain with the FP8 block's exact quantization points -> bf16 ``[B, S, d_model]``.

    ``fused=False`` mirrors the UNFUSED FP8 pipeline: the projection epilogue
    rounds ``alpha * acc`` to bf16 (the slab), the SDPA writes bf16 O, the gate
    kernel writes bf16 O_gated, and each is quantized from that bf16 value.

    ``fused=True`` mirrors the FULLY FUSED pipeline's numerics contract (one
    rounding per output): the projection fork norms/rotates the fp32
    ``alpha * acc`` and casts Q/K/V to e4m3 ONCE (GATE is rounded to bf16 --
    ``gate16`` IS a bf16 buffer), the SDPA fork multiplies its fp32 O by
    ``sigmoid(gate16)`` and casts to e4m3 ONCE.  Neither variant replicates the
    FP8 SDPA's e4m3 cast of P, which is why the block tests gate on a cosine.
    """
    b, s, dm = inp["h"].shape
    t = b * s
    hq, hkv, d, r = geom.h_q, geom.h_kv, geom.d_head, geom.rope_dim
    h32 = dequant_e4m3(inp["h"], descale_h).view(t, dm)
    w32 = dequant_e4m3(inp["w_qkvg"], descale_w_qkvg)
    proj = h32 @ w32.t()
    if not fused:
        proj = proj.to(torch.bfloat16)  # the GEMM epilogue rounds (acc * alpha) to bf16
    o_q, o_g, o_k, o_v = geom.offsets
    q = proj[:, o_q : o_q + hq * d].reshape(b, s, hq, d)
    gate = proj[:, o_g : o_g + hq * d].reshape(b, s, hq, d)
    if fused:
        gate = gate.to(torch.bfloat16)  # gate16: the fork's bf16 GATE buffer
    k = proj[:, o_k : o_k + hkv * d].reshape(b, s, hkv, d)
    v = proj[:, o_v : o_v + hkv * d].reshape(b, s, hkv, d)
    qn, _ = qk_norm_rope_reference(q, inp["w_q_norm"], inp["cos"], inp["sin"], r, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    kn, _ = qk_norm_rope_reference(k, inp["w_k_norm"], inp["cos"], inp["sin"], r, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    q32 = dequant_e4m3(quant_e4m3(qn, scale_q), 1.0 / scale_q)
    k32 = dequant_e4m3(quant_e4m3(kn, scale_k), 1.0 / scale_k)
    v32 = dequant_e4m3(quant_e4m3(v, scale_v), 1.0 / scale_v)
    # fp32 SDPA with GQA broadcast on the dequantized fp8 operands
    rep = hq // hkv
    qb = q32.transpose(1, 2)  # [b, hq, s, d]
    kb = k32.transpose(1, 2).repeat_interleave(rep, 1)
    vb = v32.transpose(1, 2).repeat_interleave(rep, 1)
    logits = torch.einsum("bhqd,bhkd->bhqk", qb, kb) * geom.scale
    mask = torch.ones(s, s, dtype=torch.bool, device=logits.device)
    if geom.is_causal:
        mask = torch.tril(mask)
    allowed = mask[None, None]
    if seq_lens is not None:
        kv_ok = torch.arange(s, device=logits.device)[None, :] < seq_lens.to(logits.device)[:, None]  # [b, s]
        allowed = allowed & kv_ok[:, None, None, :]
    logits = logits.masked_fill(~allowed, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)  # fully-masked rows -> O = 0, as the kernel substitutes
    o = torch.einsum("bhqk,bhkd->bhqd", p, vb).transpose(1, 2)  # [b, s, hq, d]
    if fused:
        og = o.float() * torch.sigmoid(gate.float())  # fp32 O * sigmoid(gate16), ONE e4m3 cast below
    else:
        o = o.to(torch.bfloat16)  # the SDPA writes bf16 O
        og = (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)  # the gate kernel writes bf16 O_gated
    og32 = dequant_e4m3(quant_e4m3(og, scale_o), 1.0 / scale_o).reshape(t, hq * d)  # o was transposed: reshape copies
    wo32 = dequant_e4m3(inp["w_o"], descale_w_o)
    return (og32 @ wo32.t()).to(torch.bfloat16).view(b, s, dm)


# ---------------------------------------------------------------------------
# MXFP8 (E4M3 codes + per-32-block E8M0 scales, cuDNN F8_128x4) -- the shared fake-quant reference
# ---------------------------------------------------------------------------
#
# ONE copy of the MXFP8 recipe the MXFP8 block test, block_perf_table and
# block_fusion_table all need: the caller-side quantization of ``h`` / ``W_qkvg``
# into e4m3 codes + PADDED F8_128x4 E8M0 blobs (exactly the contract
# ``GatedAttentionBlockFwd(quant=MxQuantSpec, sample_h_sf=, sample_w_qkvg_sf=)``
# takes -- ``kernels.proj_gemm.sf_blob_bytes`` bytes), a per-tensor amax scale
# for ``W_o`` (PR-B D1), and the fp32 oracle with the block's own quantization
# points: block-dequant of h / W, Q / K block-quantized ROWWISE along D, V
# COLUMNWISE along S per (b, h) on the S-padded tensor, per-tensor O and W_o.
# The block scales are TE / cuDNN semantics via ``test/python/sdpa/mxfp8_quant.py``
# (E8M0 exponent rounded UP, exact power-of-two dequant) -- the same module the
# FROST MXFP8 SDPA suites quantize with, so the oracle and the kernels agree on
# every scale byte.  Neither oracle replicates the MXFP8 SDPA's unit-scale e4m3
# P, which is why the block tests gate on a cosine (0.99 floor).

MX_BLOCK = 32
MX_ATOM_ROWS = 128  # rows of one F8_128x4 atom (== the SDPA's Q / KV tile height)
MX_ATOM_COLS = 4  # 32-element blocks per atom row


def _mxfp8_quant():
    """``test/python/sdpa/mxfp8_quant.py`` -- as ``sdpa.mxfp8_quant`` when ``test/python`` is
    importable (pytest from there; the SDPA suites' spelling), else loaded BY PATH so a
    standalone driver (``frost_dev/block_*_table.py``) gets it without touching ``sys.path``."""
    try:
        from sdpa import mxfp8_quant as mq

        return mq
    except ImportError:
        import importlib.util
        import os

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sdpa", "mxfp8_quant.py")
        spec = importlib.util.spec_from_file_location("_gated_block_mxfp8_quant", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def mx_sf_padded_dims(rows: int, k: int, block: int = MX_BLOCK) -> Tuple[int, int]:
    """``(rows_pad, blocks_pad)`` of one F8_128x4 scale matrix over ``rows x K`` at ``block``
    elements per scale (32 for MXFP8 / MXFP4, 16 for NVFP4): whole 128-row x 4-block atoms.
    A deliberately INDEPENDENT mirror of ``kernels.proj_gemm.sf_padded_dims`` (the reference
    must not import the thing it checks)."""
    if k % block:
        raise ValueError(f"K={k} must be a multiple of the {block}-element scale block")
    return -(-rows // MX_ATOM_ROWS) * MX_ATOM_ROWS, -(-(k // block) // MX_ATOM_COLS) * MX_ATOM_COLS


def mx_quantize_rowwise_2d(x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[rows, K]`` -> (e4m3 codes ``[rows, K]``, logical E8M0 bytes ``[rows, K/32]``): one scale per
    32-element block along K (TE ``quantize_blocks`` semantics)."""
    mq = _mxfp8_quant()
    rows, k = x2d.shape
    if k % MX_BLOCK:
        raise ValueError(f"K={k} must be a multiple of {MX_BLOCK}")
    codes, e = mq.quantize_blocks(x2d.float().reshape(rows, k // MX_BLOCK, MX_BLOCK), FP8_E4M3)
    return codes.reshape(rows, k), e.contiguous()


def mx_swizzle_sf_rowwise_padded(e: torch.Tensor, block: int = MX_BLOCK) -> torch.Tensor:
    """Logical ``[rows, K/block]`` scale BYTES -> the PADDED F8_128x4 blob (flat uint8) the block's
    ``sample_h_sf`` / ``sample_w_qkvg_sf`` / ``sample_w_o_sf`` contract takes: rows padded to 128,
    blocks to 4, pad ``0x00``, then cuDNN's F8_128x4 reorder (``swizzle_sf_rowwise`` == the FROST
    GEMM suite's ``to_blocked``).  ``numel == kernels.proj_gemm.sf_blob_bytes(rows, K, block)``.

    The ONE blob builder for every scale format: ``block`` only sizes the logical matrix (the
    atom rule is format-agnostic), and an ``e`` handed over as ``float8_e8m0fnu`` / ``float8_e4m3fn``
    is re-VIEWED as its bytes, never value-cast (``0.0137.to(uint8)`` would be ``0``)."""
    mq = _mxfp8_quant()
    if e.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        e = e.view(torch.uint8)
    rows, cols = e.shape
    rows_pad, cols_pad = mx_sf_padded_dims(rows, cols * block, block)
    pad = torch.zeros(rows_pad, cols_pad, dtype=torch.uint8, device=e.device)
    pad[:rows, :cols] = e.to(torch.uint8)
    return mq.swizzle_sf_rowwise(pad).contiguous().flatten()


def mx_unswizzle_sf_rowwise(blob: torch.Tensor, rows: int, k: int, block: int = MX_BLOCK) -> torch.Tensor:
    """The inverse of :func:`mx_swizzle_sf_rowwise_padded`: a PADDED F8_128x4 blob -> the logical
    ``[rows, K/block]`` scale bytes (pad rows / blocks dropped).

    ``_swizzle_128x4`` views ``[R, C]`` as ``(rt, rg, rr, ct, cc)`` = ``(R/128, 4, 32, C/4, 4)`` and
    stores it as ``(rt, ct, rr, rg, cc)``; reading the blob back in that order and permuting
    ``(0, 3, 2, 1, 4)`` restores the logical matrix.  The oracle dequantizes ``h`` / ``W_qkvg``
    THROUGH this, so it reads exactly the bytes the kernel reads (a wrong caller blob is a
    wrong oracle too, never a silent agreement)."""
    rows_pad, cols_pad = mx_sf_padded_dims(rows, k, block)
    if blob.numel() != rows_pad * cols_pad:
        raise ValueError(f"blob has {blob.numel()} bytes, the padded F8_128x4 matrix over {rows} x K={k} at block {block} is {rows_pad}x{cols_pad}")
    v = blob.reshape(rows_pad // MX_ATOM_ROWS, cols_pad // MX_ATOM_COLS, 32, 4, MX_ATOM_COLS)  # (rt, ct, rr, rg, cc)
    e = v.permute(0, 3, 2, 1, 4).reshape(rows_pad, cols_pad)  # (rt, rg, rr, ct, cc)
    return e[:rows, : k // block].contiguous()


def mx_dequant_rowwise_2d(codes: torch.Tensor, blob: torch.Tensor) -> torch.Tensor:
    """e4m3 codes ``[rows, K]`` x their PADDED F8_128x4 blob -> fp32 ``[rows, K]`` (exact 2^e scales)."""
    mq = _mxfp8_quant()
    rows, k = codes.shape
    e = mx_unswizzle_sf_rowwise(blob.to(torch.uint8), rows, k)
    scale = mq.e8m0_to_float(e).repeat_interleave(MX_BLOCK, dim=-1)  # [rows, K]
    return codes.float() * scale


# ---------------------------------------------------------------------------
# FP4 (E2M1 codes, two per byte) -- NVFP4 (E4M3 scale per 16) and MXFP4 (E8M0 scale per 32)
# ---------------------------------------------------------------------------
#
# torch 2.13 has ``float4_e2m1fn_x2`` but cannot cast to or from it (``copy_kernel`` is not
# implemented), so the reference rounds on fp32 BY HAND and compares codes as uint8.  The two
# formats share the E2M1 code grid and the F8_128x4 blob builder above; they differ only in the
# scale rule (``fp4_quantize_rowwise_2d``).  Both scale rules multiply by the fp32 rounding of
# ``1/6`` (E2M1's max) exactly as the kernel does (``opaque_fp4_max_rcp``): dividing by 6 is one
# ulp apart on some amax values and would move an E8M0 exponent or an E4M3 code.

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # positive codes 0..7; code | 0x8 is the negative
E2M1_MAX = E2M1_GRID[-1]
_E2M1_MIDPOINTS = tuple((a + b) / 2 for a, b in zip(E2M1_GRID[:-1], E2M1_GRID[1:]))  # 0.25 .. 5.0
FP4_INV_MAX = 1.0 / E2M1_MAX  # rounded to fp32 at use (bits 0x3E2AAAAB), the kernel's constant
E4M3_MIN_SUBNORMAL = 2.0**-9  # the NVFP4 scale floor: keeps an all-zero block's encode scale finite
FP4_FORMATS = {
    "nvfp4": (16, torch.float8_e4m3fn),  # kernel_registry: fp4_e2m1 x fp8_e4m3, block 16
    "mxfp4": (32, torch.float8_e8m0fnu),  # kernel_registry: fp4_e2m1 x fp8_e8m0, block 32
}


def fp4_format(fmt) -> Tuple[str, int, torch.dtype]:
    """``"nvfp4"`` / ``"mxfp4"`` (or an enum member NAMED so, e.g. the block's ``Fp4Format``) ->
    ``(name, block, sf_dtype)``.  The reference must not import the thing it checks, so the
    format is spelled by name here and the block / scale dtype are derived from that name."""
    name = str(getattr(fmt, "name", fmt)).lower()
    if name not in FP4_FORMATS:
        raise ValueError(f"unknown fp4 format {fmt!r}; the served formats are {sorted(FP4_FORMATS)}")
    block, sf_dtype = FP4_FORMATS[name]
    return name, block, sf_dtype


def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> 4-bit E2M1 codes (uint8 ``0..15``, sign in bit 3): round to NEAREST on the grid,
    ties to the EVEN code (``0.25 -> 0``, ``0.75 -> 1.0``, ``1.25 -> 1.0``, ``1.75 -> 2``,
    ``2.5 -> 2``, ``3.5 -> 4``, ``5.0 -> 4``), saturate at ``6`` (``cvt.rn.satfinite.e2m1x2`` --
    ``+-inf`` saturates too), sign kept (``-0.2 -> -0.0``).  NaN is REFUSED (``ValueError``): the torchao
    port this is pinned against has no NaN code, ``satfinite``'s NaN byte is unspecified, and a reference
    that encoded NaN as ``6.0`` would report a kernel CODE mismatch instead of a bad input."""
    if bool(torch.isnan(x).any()):
        raise ValueError("e2m1_codes: NaN input is out of the fp4 quantizer's contract (the reference does not fabricate a code for it)")
    a = x.float().abs()
    mids = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=a.device)
    lo = torch.bucketize(a, mids, right=False)  # a == midpoint -> the lower code
    hi = torch.bucketize(a, mids, right=True)  # a == midpoint -> the upper code; equal to ``lo`` otherwise
    code = torch.where(lo % 2 == 0, lo, hi)  # ties to the even code
    return (code.to(torch.uint8) | (torch.signbit(x).to(torch.uint8) << 3)).contiguous()


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (uint8 ``0..15``) -> their fp32 values (``-0.0`` for code ``8``)."""
    lut = torch.tensor(E2M1_GRID + tuple(-v for v in E2M1_GRID), dtype=torch.float32, device=codes.device)
    return lut[(codes & 0xF).long()]


def e2m1_rne(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp32 on the E2M1 grid ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` (sign kept): the value
    :func:`e2m1_codes` encodes."""
    return e2m1_values(e2m1_codes(x))


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 codes ``[..., K]`` -> bytes ``[..., K/2]``: the LOW nibble is the EVEN ``k`` (what the
    GEMM's ``float4_e2m1fn_x2`` operand and the suite's ``unpack_fp4`` read)."""
    if codes.shape[-1] % 2:
        raise ValueError(f"K={codes.shape[-1]} must be even to pack two E2M1 codes per byte")
    c = codes.to(torch.uint8) & 0xF
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Bytes ``[..., K/2]`` -> fp32 E2M1 values ``[..., K]`` (low nibble first)."""
    u8 = packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed
    lo = e2m1_values(u8 & 0xF)
    hi = e2m1_values(u8 >> 4)
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def fp4_quantize_rowwise_2d(x2d: torch.Tensor, fmt) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[rows, K]`` -> (packed E2M1 bytes ``[rows, K/2]``, logical scale BYTES ``[rows, K/block]``),
    one scale per ``block`` elements along K.

    MXFP4: ``e = e8m0_ceil(amax * fp32(1/6))`` (TE / cuDNN semantics, exponent rounded UP -- the
    shared ``mxfp8_quant.e8m0_ceil``), ``q = e2m1_rne(x * 2^(127-e))`` (exact power-of-two scaling).
    NVFP4: ``s = (amax * fp32(1/6)).clamp_min(2^-9).to(float8_e4m3fn)`` (RN, the E4M3 min-subnormal
    floor keeps an all-zero block's scale ``0x01`` rather than ``0``), ``q = e2m1_rne(x / s)`` with a
    TRUE division -- a reciprocal-multiply perturbs exact E2M1 midpoints (``0.5859375 / 0.46875 ==
    1.25``) and flips a whole code; the kernel uses ``div.rn.f32`` for the same reason.  No global
    (per-tensor) scale in either format.  The scale bytes are the kernel's SF bytes; feed them to
    :func:`mx_swizzle_sf_rowwise_padded` with the format's ``block``."""
    name, block, sf_dtype = fp4_format(fmt)
    mq = _mxfp8_quant()
    rows, k = x2d.shape
    if k % block:
        raise ValueError(f"K={k} must be a multiple of the {block}-element {name} block")
    if not bool(torch.isfinite(x2d).all()):
        # A non-finite element makes its block's amax non-finite: the E8M0 exponent would come out 0xFF, the
        # E4M3 scale 0x7F, and every code in the block a fabricated 6.0 -- a bitwise tier comparing the kernel
        # against that would report a CODE mismatch where the finding is "bad input".  Refuse instead.
        raise ValueError(f"{name}: non-finite input is out of the fp4 quantizer's contract")
    x = x2d.float().reshape(rows, k // block, block)
    amax = x.abs().amax(dim=-1)  # [rows, K/block]
    inv_max = torch.tensor(FP4_INV_MAX, dtype=torch.float32, device=x.device)
    if name == "mxfp4":
        e = mq.e8m0_ceil(amax * inv_max)
        q = x * mq.exp2_rcp(e).unsqueeze(-1)
        sf_bytes = e
    else:
        s = (amax * inv_max).clamp_min(E4M3_MIN_SUBNORMAL).to(sf_dtype)
        q = x / s.float().unsqueeze(-1)
        sf_bytes = s.view(torch.uint8)
    codes = e2m1_codes(q).reshape(rows, k)
    return pack_e2m1(codes), sf_bytes.contiguous()


def fp4_dequant_rowwise_2d(packed: torch.Tensor, blob: torch.Tensor, fmt, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Packed E2M1 bytes ``[rows, K/2]`` x their PADDED F8_128x4 scale blob -> ``out_dtype`` ``[rows, K]``.

    THROUGH the blob (:func:`mx_unswizzle_sf_rowwise` at the format's block), so the oracle reads
    exactly the scale bytes the GEMM reads -- a wrong caller blob is a wrong oracle too, never a
    silent agreement.  E8M0 scales dequantize as ``2^(e-127)``, E4M3 scales as their value.  Every
    product ``code x scale`` is exact in fp64; in fp32 it is exact wherever it is representable, but
    an MXFP4 block whose amax sits in bf16's top octave can dequantize to ``4 x 2^126 = 2^128`` and
    overflows to ``inf`` -- the same ``inf`` the GEMM's fp32 accumulator would produce from those
    codes.  Pass ``out_dtype=torch.float64`` to read such a blob back finite."""
    name, block, sf_dtype = fp4_format(fmt)
    mq = _mxfp8_quant()
    rows, k_half = packed.shape
    k = 2 * k_half
    sf = mx_unswizzle_sf_rowwise(blob.view(torch.uint8) if blob.dtype != torch.uint8 else blob, rows, k, block)
    scale = (mq.e8m0_to_float(sf) if name == "mxfp4" else sf.view(sf_dtype).float()).to(out_dtype)
    return unpack_e2m1(packed).to(out_dtype) * scale.repeat_interleave(block, dim=-1)


def quantize_block_inputs_mxfp8(inp: dict, *, o_fp4=None) -> Tuple[dict, dict]:
    """``make_inputs`` output -> the same dict with ``h`` / ``w_qkvg`` as e4m3 MXFP8 CODES plus
    their PADDED F8_128x4 blobs ``h_sf`` / ``w_qkvg_sf`` (the block's ``sample_h_sf`` /
    ``sample_w_qkvg_sf`` and execute ``h_sf=`` / ``w_qkvg_sf=``), ``w_o`` per-tensor e4m3 (D1),
    and ``{descale_w_o}`` -- the ``MxQuantSpec`` constructor kwargs (``scale_o`` is the caller's).

    ``o_fp4`` (appended; ``"nvfp4"`` / ``"mxfp4"`` or the block's ``Fp4Format`` member): the fp4 O
    mode's ``W_o`` instead -- packed E2M1 codes ``[d_model, K/2]`` viewed ``float4_e2m1fn_x2`` plus
    the format's PADDED F8_128x4 blob ``w_o_sf`` (the block's ``sample_w_o_sf`` / ``w_o_sf``), and
    ``descale_w_o=1.0`` (no per-tensor scale on a block-scaled weight).

    Norm weights, cos/sin stay bf16 (the block's activation dtype).  Built from an EXISTING
    input dict so a harness can hand the bf16, FP8 and MXFP8 arms the same data."""
    h = inp["h"]
    b, s, dm = h.shape
    h_codes, h_e = mx_quantize_rowwise_2d(h.reshape(b * s, dm))
    w_codes, w_e = mx_quantize_rowwise_2d(inp["w_qkvg"])
    mx = dict(inp)
    mx["h"] = h_codes.reshape(b, s, dm).contiguous()
    mx["h_sf"] = mx_swizzle_sf_rowwise_padded(h_e)
    mx["w_qkvg"] = w_codes.contiguous()
    mx["w_qkvg_sf"] = mx_swizzle_sf_rowwise_padded(w_e)
    if o_fp4 is None:
        s_wo = amax_scale(inp["w_o"])
        mx["w_o"] = quant_e4m3(inp["w_o"], s_wo)
        return mx, dict(descale_w_o=1.0 / s_wo)
    _, block, _ = fp4_format(o_fp4)
    packed, e = fp4_quantize_rowwise_2d(inp["w_o"].float(), o_fp4)
    mx["w_o"] = packed.view(torch.float4_e2m1fn_x2)
    mx["w_o_sf"] = mx_swizzle_sf_rowwise_padded(e, block)
    return mx, dict(descale_w_o=1.0)


def make_mxfp8_inputs(geom: RefGeometry, batch: int, seq_len: int, *, device: torch.device | str = "cuda", seed: int = 0) -> Tuple[dict, dict]:
    """bf16 inputs from :func:`make_inputs`, then :func:`quantize_block_inputs_mxfp8`."""
    return quantize_block_inputs_mxfp8(make_inputs(geom, batch=batch, seq_len=seq_len, device=device, dtype=torch.bfloat16, seed=seed))


def mx_fake_quant_rowwise(x: torch.Tensor) -> torch.Tensor:
    """Block-quantize ``x[..., D]`` along its LAST dim in 32-element blocks and dequantize (fp32):
    what the block's rowwise ``quantize_mxfp8`` (Q / K) does to the SDPA's operands."""
    mq = _mxfp8_quant()
    *lead, d = x.shape
    if d % MX_BLOCK:
        raise ValueError(f"last dim {d} must be a multiple of {MX_BLOCK}")
    codes, e = mq.quantize_blocks(x.float().reshape(*lead, d // MX_BLOCK, MX_BLOCK), FP8_E4M3)
    return (codes.float() * mq.e8m0_to_float(e).unsqueeze(-1)).reshape(*lead, d)


def mx_fake_quant_v_columnwise(v: torch.Tensor) -> torch.Tensor:
    """V ``[B, S, H, D]`` block-quantized along S (32-token blocks per (b, h, d), on the tensor
    S-padded to whole 128-row tiles as the kernel and ``quantize_to_mxfp8`` see it) and
    dequantized -- the COLUMNWISE quantization the BMM2 operand takes.  Pad rows are zero and
    never change a block's amax, so the un-padded result is the kernel's exactly."""
    mq = _mxfp8_quant()
    b, s, h, d = v.shape
    s_pad = -(-s // MX_ATOM_ROWS) * MX_ATOM_ROWS
    x = v.float().permute(0, 2, 3, 1)  # [b, h, d, s]
    if s_pad != s:
        x = F.pad(x, (0, s_pad - s))
    codes, e = mq.quantize_blocks(x.reshape(b, h, d, s_pad // MX_BLOCK, MX_BLOCK), FP8_E4M3)
    deq = (codes.float() * mq.e8m0_to_float(e).unsqueeze(-1)).reshape(b, h, d, s_pad)[..., :s]
    return deq.permute(0, 3, 1, 2).contiguous()  # [b, s, h, d]


def _mxfp8_gated_o(inp_mx: dict, geom: RefGeometry, *, seq_lens: Optional[torch.Tensor], fused: bool) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Stages (1)..(5) of the MXFP8 chain in fp32 with the block's rounding points -> the gated O
    (fp32 under ``fused``, bf16 otherwise -- the value the per-tensor O quantization sees)."""
    b, s, dm = inp_mx["h"].shape
    t = b * s
    hq, hkv, d, r = geom.h_q, geom.h_kv, geom.d_head, geom.rope_dim
    # (1) block-dequantized codes (E8M0 exact) x block-dequantized weights, fp32 accumulate.
    h32 = mx_dequant_rowwise_2d(inp_mx["h"].reshape(t, dm), inp_mx["h_sf"])
    w32 = mx_dequant_rowwise_2d(inp_mx["w_qkvg"], inp_mx["w_qkvg_sf"])
    proj = h32 @ w32.t()
    if not fused:
        proj = proj.to(torch.bfloat16)  # the block-scale GEMM writes a bf16 slab
    o_q, o_g, o_k, o_v = geom.offsets
    q = proj[:, o_q : o_q + hq * d].reshape(b, s, hq, d)
    gate = proj[:, o_g : o_g + hq * d].reshape(b, s, hq, d)
    if fused:
        gate = gate.to(torch.bfloat16)  # gate16: the fork's bf16 GATE buffer
    k = proj[:, o_k : o_k + hkv * d].reshape(b, s, hkv, d)
    v = proj[:, o_v : o_v + hkv * d].reshape(b, s, hkv, d)
    # (2)+(3) norm (optional) + RoPE on Q / K -- fp32, one rounding to the slab dtype
    # (bf16 unfused; the fused fork quantizes straight off fp32, so no rounding there).
    qn, _ = qk_norm_rope_reference(q, inp_mx["w_q_norm"], inp_mx["cos"], inp_mx["sin"], r, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    kn, _ = qk_norm_rope_reference(k, inp_mx["w_k_norm"], inp_mx["cos"], inp_mx["sin"], r, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    # (3q) Q / K rowwise along D, V columnwise along S -- block-quantized then dequantized.
    q32 = mx_fake_quant_rowwise(qn)
    k32 = mx_fake_quant_rowwise(kn)
    v32 = mx_fake_quant_v_columnwise(v)
    # (4) fp32 SDPA with GQA broadcast on the dequantized MXFP8 operands.
    rep = hq // hkv
    qb = q32.transpose(1, 2)  # [b, hq, s, d]
    kb = k32.transpose(1, 2).repeat_interleave(rep, 1)
    vb = v32.transpose(1, 2).repeat_interleave(rep, 1)
    logits = torch.einsum("bhqd,bhkd->bhqk", qb, kb) * geom.scale
    mask = torch.ones(s, s, dtype=torch.bool, device=logits.device)
    if geom.is_causal:
        mask = torch.tril(mask)
    allowed = mask[None, None]
    if seq_lens is not None:
        kv_ok = torch.arange(s, device=logits.device)[None, :] < seq_lens.to(logits.device)[:, None]  # [b, s]
        allowed = allowed & kv_ok[:, None, None, :]
    logits = logits.masked_fill(~allowed, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)  # fully-masked rows -> O = 0, as the kernel substitutes (SELECT)
    o = torch.einsum("bhqk,bhkd->bhqd", p, vb).transpose(1, 2)  # [b, s, hq, d]
    # (5) gate AFTER the dead-row substitution.
    if fused:
        og = o.float() * torch.sigmoid(gate.float())  # fp32 O * sigmoid(gate16); ONE e4m3 cast in the caller
    else:
        o = o.to(torch.bfloat16)  # the SDPA writes bf16 O
        og = (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)  # the gate kernel writes bf16 O_gated
    return og, (b, s, dm)


def mxfp8_calibrated_scale_o(inp_mx: dict, geom: RefGeometry, *, seq_lens: Optional[torch.Tensor] = None) -> float:
    """A static per-tensor ``scale_o`` for the UNFUSED MXFP8 block, calibrated on this data through
    the oracle's own gated O (offline-calibration stand-in; the FULLY FUSED path is pinned to 1.0
    by PR-B D8 and needs no calibration)."""
    og, _ = _mxfp8_gated_o(inp_mx, geom, seq_lens=seq_lens, fused=False)
    return amax_scale(og)


def gated_attention_block_mxfp8_reference(
    inp_mx: dict,
    geom: RefGeometry,
    *,
    descale_w_o: float,
    scale_o: float,
    seq_lens: Optional[torch.Tensor] = None,
    fused: bool = False,
    o_fp4=None,
) -> torch.Tensor:
    """fp32 chain with the MXFP8 block's exact quantization points -> bf16 ``[B, S, d_model]``.

    ``inp_mx`` is :func:`quantize_block_inputs_mxfp8`'s dict (codes + PADDED F8_128x4 blobs; the
    oracle dequantizes THROUGH the blobs, so it reads what the kernels read).

    ``fused=False`` mirrors the UNFUSED pipeline: the block-scale GEMM rounds its (already
    dequantized) accumulator to the bf16 slab, Q / K are normed + rotated from bf16 and
    block-quantized rowwise, V columnwise, the SDPA writes bf16 O, the gate kernel bf16
    O_gated, and ``quantize_o`` casts it per-tensor with ``scale_o``.

    ``fused=True`` mirrors the FULLY FUSED pipeline (one rounding per output): the fork norms /
    rotates the fp32 accumulator and block-quantizes ONCE (GATE rounded to bf16 -- ``gate16`` IS
    a bf16 buffer), the gated MXFP8 SDPA multiplies its fp32 O by ``sigmoid(gate16)`` and casts
    to e4m3 ONCE, UNSCALED -- pass ``scale_o=1.0`` (D8).  Neither variant replicates the MXFP8
    SDPA's unit-scale e4m3 P, which is why the block tests gate on a cosine.

    ``o_fp4`` (appended; ``"nvfp4"`` / ``"mxfp4"`` or the block's ``Fp4Format``) mirrors the fp4 O
    mode: the gated O is a **bf16** tensor on BOTH pipelines (the gate kernel's, or -- fused -- the
    gated MXFP8 SDPA writing bf16 O, config row 10), block-quantized to E2M1 + the format's scale
    (:func:`fp4_quantize_rowwise_2d` over ``[T, K = H_q*D]``) and dequantized THROUGH its padded
    blob, then multiplied by ``W_o`` dequantized through ``inp_mx["w_o_sf"]`` (the dict of
    :func:`quantize_block_inputs_mxfp8` with the same ``o_fp4``) -- the oracle reads exactly the
    scale bytes the block-scale GEMM reads.  Neither side carries a per-tensor scale:
    ``scale_o`` / ``descale_w_o`` must be 1.0 (``ValueError`` otherwise, so a mis-specified oracle
    cannot silently agree with a block that pins them)."""
    og, (b, s, dm) = _mxfp8_gated_o(inp_mx, geom, seq_lens=seq_lens, fused=fused)
    hq, d = geom.h_q, geom.d_head
    if o_fp4 is None:
        og32 = dequant_e4m3(quant_e4m3(og, scale_o), 1.0 / scale_o).reshape(b * s, hq * d)  # o was transposed: reshape copies
        wo32 = dequant_e4m3(inp_mx["w_o"], descale_w_o)
    else:
        if float(scale_o) != 1.0 or float(descale_w_o) != 1.0:
            raise ValueError(f"o_fp4: a block-scaled O / W_o has no per-tensor scale; got scale_o={scale_o}, descale_w_o={descale_w_o}")
        _, block, _ = fp4_format(o_fp4)
        og16 = og.to(torch.bfloat16).float().reshape(b * s, hq * d)  # the bf16 gated O both pipelines hand the fp4 quantizer
        packed, e = fp4_quantize_rowwise_2d(og16, o_fp4)
        og32 = fp4_dequant_rowwise_2d(packed, mx_swizzle_sf_rowwise_padded(e, block), o_fp4)
        w_o = inp_mx["w_o"]
        wo32 = fp4_dequant_rowwise_2d(w_o.view(torch.uint8) if w_o.dtype != torch.uint8 else w_o, inp_mx["w_o_sf"], o_fp4)
    return (og32 @ wo32.t()).to(torch.bfloat16).view(b, s, dm)
