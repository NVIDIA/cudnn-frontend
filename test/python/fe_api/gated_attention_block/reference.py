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
    shaped (``[out, in]``), so the block's GEMMs read them transposed."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, std=0.02):
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * std).to(dtype)

    cos, sin = build_rope_tables(seq_len, geom.rope_dim, base=geom.rope_base, batch=batch, device=device, dtype=dtype)
    return {
        "h": randn(batch, seq_len, geom.d_model, std=1.0),
        "w_qkvg": randn(geom.n_qkvg, geom.d_model),
        "w_q_norm": torch.ones(geom.d_head, device=device, dtype=dtype),
        "w_k_norm": torch.ones(geom.d_head, device=device, dtype=dtype),
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
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stages (2)+(3) fused, the way the FROST kernel computes them.

    FP32 norm, FP32 rotation, **one** cast at the end. That last part is not a
    detail: an unfused torch chain rounds to bf16 after the norm AND again after
    the rotation, so it is a slightly different function. This is the oracle the
    kernel is checked against; ``gated_attention_block_baseline`` keeps the
    two-rounding chain on purpose, because that is what a framework does.

    Returns ``(y, rstd)`` with ``y`` in ``x``'s dtype and ``rstd`` fp32 — the
    reciprocal RMS the backward needs, which is why the kernel emits it.
    """
    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    y = x32 * rstd * w.float()
    if rope_dim:
        c = cos[:, :, None, :rope_dim].float()
        s = sin[:, :, None, :rope_dim].float()
        rot, passthrough = y[..., :rope_dim], y[..., rope_dim:]
        rot = rot * c + _rotate_half(rot) * s
        y = torch.cat((rot, passthrough), dim=-1) if passthrough.shape[-1] else rot
    return y.to(x.dtype), rstd.squeeze(-1).float()


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
    rstd_q: torch.Tensor  # [B, S, H_q]  fp32
    rstd_k: torch.Tensor  # [B, S, H_kv] fp32
    o: torch.Tensor  # [B, S, H_q, D]   SDPA output, PRE-gate
    o_gated: torch.Tensor  # [B, S, H_q, D]
    lse: torch.Tensor  # [B, H_q, S]  fp32, natural log


def gated_attention_block_reference(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: torch.Tensor,
    w_k_norm: torch.Tensor,
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
    # with a single final rounding, matching the fused kernel.
    q, rstd_q = qk_norm_rope_reference(q_pre, w_q_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps)
    k, rstd_k = qk_norm_rope_reference(k_pre, w_k_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps)

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
    w_q_norm: torch.Tensor,
    w_k_norm: torch.Tensor,
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
    qn, _ = qk_norm_rope_reference(q, inp["w_q_norm"], inp["cos"], inp["sin"], r, geom.qk_norm_eps)
    kn, _ = qk_norm_rope_reference(k, inp["w_k_norm"], inp["cos"], inp["sin"], r, geom.qk_norm_eps)
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
