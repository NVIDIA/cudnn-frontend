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
from typing import List, Optional, Tuple

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
    # Appended LAST: the mask band the block's SDPA lowers, in the block
    # Geometry's own vocabulary. ``window_left = W`` keeps ``k >= q + diag - W``
    # (W = L - 1 for a cuDNN window LENGTH L); ``window_right = R`` widens the
    # causal upper bound to ``k <= q + diag + R`` (needs ``is_causal``);
    # ``causal_bottom_right`` puts the diagonal at ``diag = S_kv - S_q`` instead
    # of 0 (a no-op for the block's self-attention; dense-only, see the mask
    # helper). -1 = unbounded. ``window_right`` / ``causal_bottom_right`` without
    # ``is_causal`` are refused by the mask helper, as ``api.py`` refuses them.
    window_left: int = -1
    window_right: int = -1
    causal_bottom_right: bool = False

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
    acc_dtype: torch.dtype = torch.float32,
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

    ``acc_dtype`` (appended) is the dtype every ``.float()`` site computes
    in -- fp32 by default (the kernel's arithmetic), ``torch.float64`` for an
    autograd oracle of the backward kernels: with fp64 inputs the chain then
    stays unrounded (the final cast is to ``x.dtype`` = fp64) and
    ``torch.autograd.grad`` through it is the exact adjoint.
    """
    x32 = x.to(acc_dtype)
    if qk_norm:
        if w is None:
            raise ValueError("qk_norm=True needs a [D] norm weight; pass qk_norm=False for RoPE-only Q/K")
        rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        y = x32 * rstd * w.to(acc_dtype)
    else:
        rstd = None
        y = x32
    if rope_dim:
        c = cos[:, :, None, :rope_dim].to(acc_dtype)
        s = sin[:, :, None, :rope_dim].to(acc_dtype)
        rot, passthrough = y[..., :rope_dim], y[..., rope_dim:]
        rot = rot * c + _rotate_half(rot) * s
        y = torch.cat((rot, passthrough), dim=-1) if passthrough.shape[-1] else rot
    return y.to(x.dtype), (None if rstd is None else rstd.squeeze(-1).to(acc_dtype))


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, *, acc_dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row RMSNorm over the last dim, computed in ``acc_dtype`` (fp32 by default).

    Returns ``(y, rstd)``; ``rstd`` is the ``[..., 1]``-squeezed ``acc_dtype``
    reciprocal RMS the backward needs, which is why it comes out of the forward
    at all. ``acc_dtype`` is appended for the fp64 backward oracles.
    """
    x32 = x.to(acc_dtype)
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    y = (x32 * rstd) * w.to(acc_dtype)
    return y.to(x.dtype), rstd.squeeze(-1).to(acc_dtype)


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
    window_left: int = -1,
    window_right: int = -1,
    causal_bottom_right: bool = False,
    s_q_total: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """``[q_chunk, s_kv]`` bool mask, True where a column is ALLOWED.

    Rows ``[q_lo, q_lo + s_q)`` of the full ``[S_q, S_kv]`` mask. The band is the
    ONE the block's SDPA lowers (``SdpaFwdDslSm100``: per-side offsets from a
    diagonal, ``None`` / ``-1`` = unbounded): with ``diag = S_kv - S_q_total``
    under ``causal_bottom_right`` and ``0`` otherwise,

    * ``is_causal``: keep ``k <= q + diag + max(window_right, 0)``
      (``window_right`` WIDENS the causal bound; it needs ``is_causal``);
    * ``window_left = W >= 0``: keep ``k >= q + diag - W`` (``W = L - 1`` for a
      cuDNN window LENGTH ``L``); with ``is_causal=False`` a sliding window alone;
    * ``seq_lens``: keep ``k < seq_lens[batch_index]`` (orthogonal to the band).

    ``s_q_total`` (the full query length; defaults to ``s_kv``, the block's
    self-attention) only matters under ``causal_bottom_right``. That arm is
    DENSE-only: its diagonal is anchored at the PADDED lengths ``S_kv -
    S_q_total`` and ignores ``seq_lens`` (cuDNN's bottom-right under padding
    anchors at the per-batch ACTUAL lengths; for the block's self-attention --
    equal Q / KV lengths per batch -- both anchors give ``diag = 0``, the
    plain-causal band, so the two agree on everything the block lowers).
    ``window_right`` or ``causal_bottom_right`` without ``is_causal`` is REFUSED,
    exactly as ``GatedAttentionBlockGeometry`` refuses it (``api.py``): a graph
    flag the reference silently ignored would agree with nothing the SDPA lowers.
    Appended kwargs default to the previous behaviour (plain causal + padding).
    Pinned against ``F.scaled_dot_product_attention`` in ``test_qk_norm_rope_bwd.py``.
    """
    if int(window_right) >= 0 and not is_causal:
        raise ValueError("window_right requires is_causal=True (it widens the causal diagonal, it does not create one)")
    if causal_bottom_right and not is_causal:
        raise ValueError("causal_bottom_right=True requires is_causal=True")
    q_idx = torch.arange(q_lo, q_lo + s_q, device=device)
    k_idx = torch.arange(s_kv, device=device)
    allowed = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
    diag = (s_kv - (s_kv if s_q_total is None else int(s_q_total))) if causal_bottom_right else 0
    if is_causal:
        allowed &= k_idx[None, :] <= q_idx[:, None] + diag + max(int(window_right), 0)
    if window_left >= 0:
        allowed &= k_idx[None, :] >= q_idx[:, None] + diag - int(window_left)
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
    acc_dtype: torch.dtype = torch.float32,
) -> RefOutputs:
    """FP32 oracle for the whole block, chunked over query tiles.

    ``seq_lens`` (``[B]`` int32) marks per-batch valid KV length; columns at or
    beyond it are masked. It is here because the degenerate cases are the ones
    the kernel gets wrong: a row with NO allowed column must produce ``O = 0``
    and ``LSE = -inf`` exactly, never a floored denominator's ``-69.08`` and
    never accumulator residue scaled by a sigmoid.

    ``geom.qk_norm=False``: the norm weights are ``None``, stage (2) is skipped
    (RoPE only) and ``rstd_q`` / ``rstd_k`` come back ``None``.

    ``acc_dtype`` (appended): the dtype of every accumulation -- fp32 by
    default; ``torch.float64`` with fp64 inputs gives the unrounded oracle the
    backward tests differentiate through. The intermediate roundings stay keyed
    on ``h.dtype`` (``out_dtype``), so an fp64 ``h`` is never rounded.
    ``geom.window_left / window_right / causal_bottom_right`` select the mask
    band (``_key_padding_and_causal_mask``).
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
    proj = (h.to(acc_dtype) @ w_qkvg.to(acc_dtype).t()).to(out_dtype)
    q_pre, gate, k_pre, v = split_qkvg(proj, geom)

    # (2)+(3) QK-RMSNorm then partial RoPE -- V is NOT normed. One fp32 pass
    # with a single final rounding, matching the fused kernel. qk_norm=False:
    # RoPE only, rstd None.
    q, rstd_q = qk_norm_rope_reference(q_pre, w_q_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=acc_dtype)
    k, rstd_k = qk_norm_rope_reference(k_pre, w_k_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=acc_dtype)

    # (4) SDPA, chunked over q tiles, fp32.
    o = torch.zeros(b, s, geom.h_q, d, device=dev, dtype=acc_dtype)
    lse = torch.full((b, geom.h_q, s), float("-inf"), device=dev, dtype=acc_dtype)

    k_b = k.to(acc_dtype).repeat_interleave(rep, dim=2)  # [B, S, H_q, D]
    v_b = v.to(acc_dtype).repeat_interleave(rep, dim=2)

    for bi in range(b):
        for lo in range(0, s, q_chunk):
            hi = min(lo + q_chunk, s)
            qc = q[bi, lo:hi].to(acc_dtype).transpose(0, 1)  # [H_q, s_q, D]
            kc = k_b[bi].transpose(0, 1)  # [H_q, S, D]
            vc = v_b[bi].transpose(0, 1)
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * geom.scale  # [H_q, s_q, S]

            allowed = _key_padding_and_causal_mask(
                hi - lo,
                s,
                is_causal=geom.is_causal,
                seq_lens=seq_lens,
                batch_index=bi,
                q_lo=lo,
                device=dev,
                window_left=geom.window_left,
                window_right=geom.window_right,
                causal_bottom_right=geom.causal_bottom_right,
                s_q_total=s,
            )
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
    o_gated = (o.to(acc_dtype) * torch.sigmoid(gate.to(acc_dtype))).to(out_dtype)

    # (6) out projection.
    o_flat = o_gated.reshape(b, s, geom.h_q * d)
    out = (o_flat.to(acc_dtype) @ w_o.to(acc_dtype).t()).to(out_dtype)

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


def fp4_quantize_rowwise_2d(x2d: torch.Tensor, fmt, global_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[rows, K]`` -> (packed E2M1 bytes ``[rows, K/2]``, logical scale BYTES ``[rows, K/block]``),
    one scale per ``block`` elements along K.

    ``global_scale`` (appended, default ``1.0`` = the forward's single-level cast): quantize ``fp32(x2d) * global_scale``
    -- ONE fp32 multiply before the block amax, the kernel's pre-scale slot read (``quantize_fp4.py``, the two-level
    cast of a GRADIENT: the live power-of-two ``scale_dy`` lifts a raw output gradient above the NVFP4 e4m3 scale floor;
    the consumer undoes it downstream).  Exact for a power of two; a non-power-of-two value is served too, rounded
    once on both sides.

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
    x = x2d.float()
    if float(global_scale) != 1.0:
        x = x * torch.tensor(float(global_scale), dtype=torch.float32, device=x.device)  # the kernel's one fp32 multiply before the amax
    x = x.reshape(rows, k // block, block)
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


def quantize_block_inputs_mxfp8(inp: dict, *, o_fp4=None, backward: bool = False, w_qkvg_fp4: bool = False) -> Tuple[dict, dict]:
    """``make_inputs`` output -> the same dict with ``h`` / ``w_qkvg`` as e4m3 MXFP8 CODES plus
    their PADDED F8_128x4 blobs ``h_sf`` / ``w_qkvg_sf`` (the block's ``sample_h_sf`` /
    ``sample_w_qkvg_sf`` and execute ``h_sf=`` / ``w_qkvg_sf=``), ``w_o`` per-tensor e4m3 (D1),
    and ``{descale_w_o}`` -- the ``MxQuantSpec`` constructor kwargs (``scale_o`` is the caller's).

    ``o_fp4`` (appended; ``"nvfp4"`` / ``"mxfp4"`` or the block's ``Fp4Format`` member): the fp4 O
    mode's ``W_o`` instead -- packed E2M1 codes ``[d_model, K/2]`` viewed ``float4_e2m1fn_x2`` plus
    the format's PADDED F8_128x4 blob ``w_o_sf`` (the block's ``sample_w_o_sf`` / ``w_o_sf``), and
    ``descale_w_o=1.0`` (no per-tensor scale on a block-scaled weight).

    ``backward`` (appended): also the MXFP8 BACKWARD's four caller artifacts, built from the SAME bf16
    inputs along the OTHER contraction axis (the caller contract of a transposed-weight-gradient
    training step): ``h_t`` = e4m3 codes ``[d_model, T]`` of ``h`` re-quantized along the TOKENS
    (``mx_quantize_rowwise_2d(h.reshape(T, dm).t().contiguous())``, K-major: strides ``(T, 1)``) with its
    blob ``h_t_sf`` over ``(rows = dm, K = T)``; ``w_qkvg_t`` = e4m3 ``[d_model, N]`` of ``W_qkvg`` re-quantized
    along its ROW axis N with ``w_qkvg_t_sf`` over ``(rows = dm, K = N)``.  Never a ``.t()`` view of the
    forward's codes and never the forward's blob: the byte count of a blob is the same for ``(rows, K)``
    and ``(K, rows)``, so a wrong-orientation blob passes every host check and only the numerics see it.
    ``h_t`` / ``h_t_sf`` exist only when ``T = batch*seq_len`` is a multiple of 32 (the token axis cannot be
    block-quantized otherwise, and the block declines a weight gradient at such a ``T``); ``w_qkvg_t`` /
    ``w_qkvg_t_sf`` are built at every ``T`` (their axes are ``d_model`` and ``N``).

    ``w_qkvg_fp4`` (appended): the MXFP4 weight mode's ``W_qkvg`` instead of the e4m3 one -- packed E2M1 codes ``[N, d_model / 2]``
    viewed ``float4_e2m1fn_x2`` (``fp4_quantize_rowwise_2d(W, "mxfp4")``, the forward suite's ``_fp4w_inputs``) with its E8M0 / 32
    blob under the UNCHANGED key ``w_qkvg_sf``; with ``backward=True`` the transposed artifact ``w_qkvg_t`` is then the packed e2m1
    ``[d_model, N / 2]`` of ``W_qkvg^T`` re-quantized along N in the same format (the SAME key as the e4m3 artifact, its dtype keyed on
    the mode) with ``w_qkvg_t_sf`` over ``(rows = d_model, K = N)``; and under ``o_fp4`` with ``backward=True`` the dict also carries
    ``w_o_t`` -- the packed e2m1 ``[H_q*D, d_model / 2]`` of ``W_o^T`` re-quantized along ``d_model`` in ``o_fp4``'s format -- with
    ``w_o_t_sf`` over ``(rows = H_q*D, K = d_model)`` at that format's block (the fp4 weight modes' backward artifacts: two fake-quants
    of one master weight along its two axes, the fp4 training recipe).

    Norm weights, cos/sin stay bf16 (the block's activation dtype).  Built from an EXISTING
    input dict so a harness can hand the bf16, FP8 and MXFP8 arms the same data."""
    h = inp["h"]
    b, s, dm = h.shape
    h_codes, h_e = mx_quantize_rowwise_2d(h.reshape(b * s, dm))
    mx = dict(inp)
    mx["h"] = h_codes.reshape(b, s, dm).contiguous()
    mx["h_sf"] = mx_swizzle_sf_rowwise_padded(h_e)
    if w_qkvg_fp4:
        packed_w, w_e = fp4_quantize_rowwise_2d(inp["w_qkvg"].float(), "mxfp4")
        mx["w_qkvg"] = packed_w.view(torch.float4_e2m1fn_x2)
        mx["w_qkvg_sf"] = mx_swizzle_sf_rowwise_padded(w_e, MX_BLOCK)
    else:
        w_codes, w_e = mx_quantize_rowwise_2d(inp["w_qkvg"])
        mx["w_qkvg"] = w_codes.contiguous()
        mx["w_qkvg_sf"] = mx_swizzle_sf_rowwise_padded(w_e)
    if backward:
        # the backward's caller artifacts: the SAME bf16 tensors re-quantized along the OTHER contraction axis, from the bf16
        # values (never a .t() of the forward's codes), each with the blob of ITS OWN orientation
        t = b * s
        n = int(inp["w_qkvg"].shape[0])
        if w_qkvg_fp4:
            w_t_packed, w_t_e = fp4_quantize_rowwise_2d(inp["w_qkvg"].t().contiguous().float(), "mxfp4")  # [dm, N]: blocks along N, packed [dm, N/2]
            mx["w_qkvg_t"] = w_t_packed.view(torch.float4_e2m1fn_x2)
            mx["w_qkvg_t_sf"] = mx_swizzle_sf_rowwise_padded(w_t_e, MX_BLOCK)
        else:
            w_t_codes, w_t_e = mx_quantize_rowwise_2d(inp["w_qkvg"].t().contiguous())  # [dm, N]: blocks along N
            mx["w_qkvg_t"] = w_t_codes.contiguous()
            mx["w_qkvg_t_sf"] = mx_swizzle_sf_rowwise_padded(w_t_e)
        checks = [("w_qkvg_t", mx["w_qkvg_t"], mx["w_qkvg_t_sf"], dm, n, MX_BLOCK)]
        if t % MX_BLOCK == 0:  # the token axis block-quantizes only in whole 32-blocks: no h_t at a ragged T (no weight gradient there)
            h_t_codes, h_t_e = mx_quantize_rowwise_2d(h.reshape(t, dm).t().contiguous())  # [dm, T]: blocks along the tokens
            mx["h_t"] = h_t_codes.contiguous()
            mx["h_t_sf"] = mx_swizzle_sf_rowwise_padded(h_t_e)
            checks.append(("h_t", mx["h_t"], mx["h_t_sf"], dm, t, MX_BLOCK))
        if o_fp4 is not None:
            _, o_block, _ = fp4_format(o_fp4)
            hd = int(inp["w_o"].shape[1])
            wo_t_packed, wo_t_e = fp4_quantize_rowwise_2d(inp["w_o"].t().contiguous().float(), o_fp4)  # [HD, dm]: blocks along d_model, packed [HD, dm/2]
            mx["w_o_t"] = wo_t_packed.view(torch.float4_e2m1fn_x2)
            mx["w_o_t_sf"] = mx_swizzle_sf_rowwise_padded(wo_t_e, o_block)
            checks.append(("w_o_t", mx["w_o_t"], mx["w_o_t_sf"], hd, dm, o_block))
        for name, codes, blob, rows, k, block in checks:
            # self-check of the builder: K-major codes of the stated shape (packed [rows, K/2] for e2m1), the blob of the stated
            # orientation's byte count at its block
            fp4 = codes.dtype == torch.float4_e2m1fn_x2
            k_store = k // 2 if fp4 else k
            if tuple(codes.shape) != (rows, k_store) or codes.stride() != (k_store, 1) or codes.dtype not in (FP8_E4M3, torch.float4_e2m1fn_x2):
                raise ValueError(f"{name}: expected contiguous codes [{rows}, {k_store}], got {tuple(codes.shape)} strides {codes.stride()} {codes.dtype}")
            rows_pad, blocks_pad = mx_sf_padded_dims(rows, k, block)
            if blob.numel() != rows_pad * blocks_pad:
                raise ValueError(f"{name}_sf: {blob.numel()} bytes, the padded blob over ({rows}, K={k}) at block {block} is {rows_pad} x {blocks_pad}")
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


# ---------------------------------------------------------------------------
# Packed (THD / varlen) inputs and the per-sequence reference
# ---------------------------------------------------------------------------
#
# Under THD the block takes ONE packed token matrix ``[T, d_model]`` (or ``[1, T, d_model]``) holding ``B`` sequences back to
# back, per-token RoPE tables whose positions restart at every sequence, and the per-sequence lengths as an int32 tensor --
# ``[B]`` lengths or ``[B+1]`` prefix sums.  The reference for such an input is the DENSE oracle run on every sequence on
# its own (each sequence's mask is its own geometry: the causal diagonal, a window, bottom-right alignment all live inside
# the sequence), so a THD bug that leaks across a sequence boundary shows up as one sequence's rows contaminated by its
# neighbour's -- which a whole-tensor cosine would average away.  Everything here is torch-only (no pytest, no block
# import): the builders are shared by the test modules and the perf harnesses.

TAIL_SENTINEL = -7.0  # finite, bf16 / fp16 / e4m3-exact, never a value a correct output row holds over a whole tail


def sequence_slices(lens) -> list:
    """``[(lo, hi), ...]`` per sequence of a packing, in order (``hi == lo`` for an empty sequence)."""
    out, lo = [], 0
    for n in lens:
        n = int(n)
        out.append((lo, lo + n))
        lo += n
    return out


def cu_seqlens_of(lens, *, base: int = 0) -> list:
    """The ``[B+1]`` prefix sums of a length list (``base`` added to every entry: a prefix tensor sliced from a larger one)."""
    cu, acc = [int(base)], int(base)
    for n in lens:
        acc += int(n)
        cu.append(acc)
    return cu


def assert_packing_contract(seq_lens, t_total: int, max_seq_len: int, num_sequences: int, *, cu: bool = False) -> list:
    """TEST-ONLY detector of the block's packed-lengths contract (the library never reads a length on the host).

    ``seq_lens`` is a list of ints or an int32 tensor (READ BACK HERE -- a D2H sync; never call this from library code):
    ``[B]`` lengths, or with ``cu=True`` ``[B+1]`` non-decreasing prefix sums at any base.  Asserts, naming the fact:
    exactly ``B = num_sequences`` sequences, every length in ``[0, max_seq_len]``, the lengths summing to ``t_total``
    (``cu[B] - cu[0] == T``), ``2 <= max_seq_len <= T``, and ``B * max_seq_len >= T`` -- the last one because a smaller
    product SILENTLY caps the SDPA chain's packed capacity below ``T`` (the tokens past ``B * max_seq_len`` are simply not
    processed, no message), which is why the block declines it at declaration.  Returns the lengths as a list of ints.
    """
    vals = seq_lens.detach().cpu().tolist() if isinstance(seq_lens, torch.Tensor) else [int(x) for x in seq_lens]
    b, t_total, max_seq_len = int(num_sequences), int(t_total), int(max_seq_len)
    if cu:
        assert len(vals) == b + 1, f"a prefix-sum tensor has B+1 = {b + 1} entries, got {len(vals)}"
        assert all(vals[i] <= vals[i + 1] for i in range(b)), f"prefix sums must be non-decreasing, got {vals}"
        lens = [vals[i + 1] - vals[i] for i in range(b)]
        assert vals[b] - vals[0] == t_total, f"cu[B] - cu[0] = {vals[b] - vals[0]} must equal the packed token total T = {t_total}"
    else:
        assert len(vals) == b, f"a lengths tensor has B = {b} entries, got {len(vals)}"
        lens = vals
    assert all(0 <= n <= max_seq_len for n in lens), f"every length must lie in [0, max_seq_len={max_seq_len}], got {lens}"
    assert sum(lens) == t_total, f"the lengths {lens} sum to {sum(lens)}, the packed token total is T = {t_total} (sum == T is the caller contract)"
    assert 2 <= max_seq_len <= t_total, f"2 <= max_seq_len <= T is required, got max_seq_len={max_seq_len}, T={t_total}"
    assert b * max_seq_len >= t_total, f"num_sequences * max_seq_len = {b * max_seq_len} < T = {t_total}: the SDPA chain would silently cap its packed capacity"
    return lens


def packed_rope_tables(
    lens, rope_dim: int, *, base: float = 1_000_000.0, device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos, sin`` of shape ``[T, rope_dim]`` for a packing: :func:`build_rope_tables` per sequence (positions
    ``0 .. len_i - 1``, restarting at every sequence boundary), concatenated along the token axis.  An empty sequence
    contributes no rows.  These are the PER-TOKEN tables a packed block takes (``[T, rope_dim]`` or ``[1, T, rope_dim]``);
    handing it a dense ``[B, S, rope_dim]`` table, or positions that run across sequences, is a plausible-but-wrong RoPE
    after the first sequence."""
    parts = [build_rope_tables(int(n), rope_dim, base=base, batch=1, device=device, dtype=dtype) for n in lens if int(n) > 0]
    if not parts:
        empty = torch.empty(0, rope_dim, device=device, dtype=dtype)
        return empty, empty.clone()
    cos = torch.cat([c[0] for c, _ in parts], dim=0)
    sin = torch.cat([s[0] for _, s in parts], dim=0)
    return cos.contiguous(), sin.contiguous()


def make_packed_inputs(
    geom: RefGeometry,
    lens,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    max_seq_len: Optional[int] = None,
    cu_base: int = 0,
) -> Tuple[dict, dict]:
    """Packed inputs for the sequences ``lens`` plus the packing metadata.

    ``inp`` is :func:`make_inputs` at ``batch=1, seq_len=T`` with ``cos`` / ``sin`` replaced by the per-token
    :func:`packed_rope_tables` (as ``[1, T, rope_dim]``): ``h`` and the weights are drawn by the SAME generator as the
    dense ``B=1, S=T`` block's (and, element for element, as a dense ``B, S`` block's with ``B*S == T`` -- the draw order is
    by element, not by shape), so the packed block and the dense one see the same bytes.

    ``meta``: ``lens`` (ints), ``cu`` (the ``[B+1]`` prefix sums at ``cu_base``), ``t`` (= T), ``b``, ``max_seq_len``
    (``max(lens)`` unless given), ``slices``, and the int32 device tensors in BOTH forms -- ``seq_lens`` ``[B]`` and
    ``cu_seqlens`` ``[B+1]`` (at ``cu_base``; the kernels normalize a prefix tensor to its first entry).  The packing is
    checked against the contract (:func:`assert_packing_contract`) before anything is allocated.
    """
    lens = [int(n) for n in lens]
    t = sum(lens)
    b = len(lens)
    s_max = int(max_seq_len) if max_seq_len is not None else max(lens) if lens else 0
    assert_packing_contract(lens, t, s_max, b)
    inp = make_inputs(geom, batch=1, seq_len=t, device=device, dtype=dtype, seed=seed)
    cos, sin = packed_rope_tables(lens, geom.rope_dim, base=geom.rope_base, device=device, dtype=dtype)
    inp["cos"], inp["sin"] = cos.view(1, t, geom.rope_dim), sin.view(1, t, geom.rope_dim)
    cu = cu_seqlens_of(lens, base=cu_base)
    meta = dict(
        lens=lens,
        cu=cu,
        t=t,
        b=b,
        max_seq_len=s_max,
        slices=sequence_slices(lens),
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device),
    )
    return inp, meta


def _packed_rows(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """A packed activation as ``[1, T, ...]`` whether it was handed over rank-2 ``[T, ...]`` or rank-3 ``[1, T, ...]``."""
    if x is None:
        return None
    return x.unsqueeze(0) if x.dim() == 2 else x


def gated_attention_block_reference_packed(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geom: RefGeometry,
    lens,
    *,
    q_chunk: int = 512,
    acc_dtype: torch.dtype = torch.float32,
) -> list:
    """The dense oracle per sequence of a packing: ``gated_attention_block_reference`` on ``h[:, lo:hi]`` with that
    sequence's own ``cos`` / ``sin`` rows, so its mask (causal / bottom-right / window) is the sequence's OWN geometry.
    Returns one :class:`RefOutputs` per sequence (``[1, len_i, ...]`` tensors) or ``None`` for an empty one (no rows exist
    -- its neighbours are what a test checks there).  ``h``, ``cos``, ``sin`` may be rank-2 ``[T, .]`` or rank-3 ``[1, T, .]``."""
    h3, cos3, sin3 = _packed_rows(h), _packed_rows(cos), _packed_rows(sin)
    refs = []
    for lo, hi in sequence_slices(lens):
        if hi == lo:
            refs.append(None)
            continue
        refs.append(
            gated_attention_block_reference(
                h3[:, lo:hi], w_qkvg, w_q_norm, w_k_norm, cos3[:, lo:hi], sin3[:, lo:hi], w_o, geom, q_chunk=q_chunk, acc_dtype=acc_dtype
            )
        )
    return refs


def compare_packed(refs: list, lens, check) -> list:
    """Run ``check(i, lo, hi, ref_i)`` for every NON-empty sequence of a packing, COLLECTING the verdicts instead of stopping
    at the first miss (a cross-sequence leak then names its sequence).  ``check`` raises ``AssertionError`` on a failure.
    Prints one line per sequence and returns the list of FAILURE texts (empty = every sequence passed; the caller asserts
    ``not failures``)."""
    failures = []
    for i, ((lo, hi), ref) in enumerate(zip(sequence_slices(lens), refs)):
        if ref is None or hi == lo:
            print(f"seq {i} [{lo}:{hi}]: empty (no rows; neighbours checked)")
            continue
        try:
            check(i, lo, hi, ref)
        except AssertionError as exc:
            text = f"seq {i} [{lo}:{hi}]: {str(exc).splitlines()[0]}"
            print(text)
            failures.append(text)
        else:
            print(f"seq {i} [{lo}:{hi}]: ok")
    return failures


# ---------------------------------------------------------------------------
# The per-tensor fp8 BACKWARD oracle: the quantized block backward's quantization points over the fp64 chain
# ---------------------------------------------------------------------------
#
# What the quantized block backward computes (``api_bwd.py`` under ``quant``), in launch order: dY -> e4m3 ``dy8``
# (``scale_dy``); B2 ``dO_gated = dy8 @ W_o8 * alpha`` (bf16); B3 the gate backward in bf16 (``dO``, ``dG``, ``og8 =
# e4m3(bf16(O * s) * scale_o)``, ``delta = rowsum(bf16 dO * bf16 O)``); ``do8 = e4m3(bf16 dO * scale_do)``; B1 ``dW_o =
# dy8^T @ og8 * alpha``; the fp8 SDPA row over the e4m3 ``q8 / k8 / v8`` (the forward's static scales), ``do8``, the
# forward's exact LSE and the block's delta -- e4m3 P (``scale_s``) into dV, e4m3 dS (``scale_dp``) into dQ / dK, bf16
# gradients out; B5+B6 in bf16; ``dqkvg8 = e4m3(bf16 dqkvg * scale_dqkvg)``; B7 / B8 over it.
#
# The oracle runs the fp64 chain of ``gated_attention_block_reference`` with exactly those points: forward
# STRAIGHT-THROUGH points at q / k / v / og (the VALUE the kernels consume -- ``deq(e4m3(bf16(x) * scale))`` -- with the
# gradient passing through), ``_QuantGrad`` points on dY and on the slab gradient (backward only), and the SDPA row as an
# autograd Function whose backward IS the row's own reference (``test/python/sdpa/fp8_ref.compute_ref_backward`` over the
# same e4m3 payloads, LSE, delta and scales).  The GEMMs' fp32-accumulate + bf16-output roundings, the norm backward's
# bf16 bands and the gate kernel's bf16 products stay unmodelled -- they are what the bf16 suite's bound covers.


def _rope_adjoint_ref(dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The exact RoPE adjoint on ``[B, S, H, D]``: ``dy*cos - rotate_half(dy*sin)`` on the leading ``rope_dim`` columns,
    pass-through beyond."""
    rot, rest = dy[..., :rope_dim], dy[..., rope_dim:]
    c, sn = cos[:, :, None, :rope_dim], sin[:, :, None, :rope_dim]
    ys = rot * sn
    half = rope_dim // 2
    g_rot = rot * c - torch.cat((-ys[..., half:], ys[..., :half]), dim=-1)
    return torch.cat((g_rot, rest), dim=-1) if rest.shape[-1] else g_rot


def dw_norm_noise_mass(dq_post: torch.Tensor, x_pre: torch.Tensor, rstd: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Per column ``d``: ``sqrt(sum over rows of (g * x_hat)^2)`` in fp64 -- the natural unit of the bf16 rounding noise of
    ``dW[d] = sum_rows g * x_hat`` when both factors carry bf16-rounded inputs (``g = RoPE^T(dQ)``, ``x_hat = x * rstd``);
    the bf16 backward suite's ``dW_norm`` bound is ``noise * mass + rtol * |ref|``."""
    g = _rope_adjoint_ref(dq_post.double(), cos, sin, rope_dim)
    x_hat = x_pre.double() * rstd.double()[..., None]
    terms = (g * x_hat).reshape(-1, x_pre.shape[-1])
    return terms.pow(2).sum(0).sqrt()


FP64_ATTENTION_CHUNK_BYTES = 2 << 30


def fp64_attention_head_chunk(b: int, s_q: int, s_kv: int, h_q: int, budget_bytes: int = FP64_ATTENTION_CHUNK_BYTES) -> int:
    """The q heads per pass of :func:`fp64_attention` that keep ONE fp64 ``[B, chunk, S_q, S_kv]`` tensor within ``budget_bytes``
    (four of them are live at a pass's peak), never fewer than TWO (``h_q`` permitting), at most ``h_q``.  32 heads at S = 2K are
    one pass (1 GiB per tensor); S = 8K is 4 heads per pass (2 GiB); S = 16K two heads (4 GiB, 16 GiB peak); S = 32K two heads at
    16 GiB (64 GiB peak).  Two is the floor because a pass over ONE head hands cuBLAS a batch of one, for which it picks a
    different fp64 GEMM than for any batch of two or more: numerically equivalent, but O then differs from the batched form in
    the last fp64 bit (4e-16 at |O| 1.6, LSE exact; measured on Rubin at S = 2K and 4K), and this oracle is held to BITWISE.
    The floor bounds the chunk, not the trailing REMAINDER: ``h_q % chunk == 1`` would leave ONE head for the last pass (32 heads
    at S in [2897, 2942] -> chunk 31; 8 heads at S in [5793, 6192] -> chunk 7), so :func:`fp64_attention_head_passes` folds that
    head into the preceding pass (``chunk + 1`` heads there, 1/chunk over the budget once) -- no pass is a batch of one at any S."""
    return max(min(2, int(h_q)), min(int(h_q), int(budget_bytes) // (int(b) * int(s_q) * int(s_kv) * 8)))


def fp64_attention_head_passes(h_q: int, chunk: int) -> List[Tuple[int, int]]:
    """The ``(h0, h1)`` q-head ranges :func:`fp64_attention` runs for ``h_q`` heads at ``chunk`` heads per pass, in order:
    ``chunk`` heads each and the remainder last, except that a remainder of ONE head is folded into the preceding pass (which
    then holds ``chunk + 1`` heads).  A pass is a batch of one -- the one form cuBLAS's fp64 GEMM choice does not reproduce bit
    for bit (see :func:`fp64_attention`) -- only when asked (``chunk == 1``) or at ``h_q == 1``.  ``(32, 31)`` -> one pass of 32;
    ``(7, 2)`` -> passes of 2, 2 and 3; ``(10, 3)`` -> 3, 3 and 4; ``(8, 3)`` -> 3, 3 and 2 (a remainder of two or more stays)."""
    h_q, chunk = int(h_q), max(1, min(int(h_q), int(chunk)))
    passes = [(h0, min(h0 + chunk, h_q)) for h0 in range(0, h_q, chunk)]
    if chunk >= 2 and len(passes) >= 2 and passes[-1][1] - passes[-1][0] == 1:
        passes[-2:] = [(passes[-2][0], passes[-1][1])]
    return passes


def fp64_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: torch.Tensor, scale: float, *, head_chunk: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp64 attention over ``[B, S, H, D]`` operands with the GQA broadcast and the ``[S_q, S_kv]`` ``allowed`` mask ->
    ``(O [B, S, H_q, D], LSE [B, H_q, S])`` in natural log; a row with no allowed column is SELECTED to ``O = 0``,
    ``LSE = -inf`` (never a floored denominator).

    ``head_chunk`` (appended, keyword-only; ``None`` = every head in one pass) is the number of q heads whose fp64
    ``[B, chunk, S, S]`` score / probability matrices are held at once: a pass holds FOUR of them at its peak, so the
    one-pass form costs ``4 * B * H_q * S^2 * 8`` bytes -- 256 GiB for 32 heads at S = 16K, 1 TiB at 32K -- and no GPU
    runs it past S ~ 8K.  The passes are :func:`fp64_attention_head_passes`: ``head_chunk`` heads each, a trailing remainder
    of ONE head folded into the pass before it.  The chunking is over the BATCH axis of every kernel involved (the batched
    matmuls, the row max and row sum, the elementwise ops): each head's operand layouts, GEMM shapes and reduction order are
    those of the one-pass form, so the result is bitwise the same for every pass of TWO OR MORE heads.  Verified on Rubin
    against the one-pass form -- this function alone (``O`` and LSE): S = 2K over 32 heads with chunks of 2, 3, 4, 8 and 16;
    S = 16K over 8 heads with chunks of 2 and 4; S = 32K over 4 heads with a chunk of 2; the folded pass at S = 2K over 7 heads
    with chunks of 2 and 3 (passes of 2, 2, 3 and of 3, 4) and over 10 heads with a chunk of 3 (3, 3, 4), and at S = 16K over 7
    heads with chunks of 2 and 3.  Through the block oracle (every tensor it returns): chunks of 2, 3 and 7 at S = 512 and 2K
    over 8 heads, chunks of 2, 3 and 31 at S = 2K and 4K over 32 heads, chunks of 2, 3 and 9 at S = 2K over 10 heads.  A pass of ONE
    head is the exception: cuBLAS selects a different fp64 GEMM for a batch of one, so ``O`` then differs in its last bit (1 ulp,
    4e-16 at |O| 1.6; LSE exact, at every S) -- numerically equivalent, not bitwise; only ``head_chunk=1`` (or ``H_q == 1``)
    produces one.  :func:`fp64_attention_head_chunk` is the oracle's auto choice.  Measured peaks of this function alone,
    256-wide heads: two heads per pass 16.7 GiB at S = 16K and 65.6 GiB at S = 32K (an 8-head pass at 16K is 65 GiB; 32 heads
    at once would be ~260 GiB)."""
    b, s, hq, _d = q.shape
    rep = hq // k.shape[2]
    kb_all, vb_all = k.transpose(1, 2), v.transpose(1, 2)  # [B, H_kv, S, D]
    passes = fp64_attention_head_passes(hq, hq if head_chunk is None else int(head_chunk))
    not_allowed = ~allowed[None, None]
    o = torch.empty(b, hq, s, v.shape[-1], dtype=q.dtype, device=q.device)
    lse = torch.empty(b, hq, s, dtype=q.dtype, device=q.device)
    for h0, h1 in passes:
        kv_heads = torch.arange(h0, h1, device=q.device) // rep  # the kv head each q head of the chunk reads (contiguous groups)
        qb = q[:, :, h0:h1].transpose(1, 2)
        kb = kb_all.index_select(1, kv_heads)
        vb = vb_all.index_select(1, kv_heads)
        scores = torch.matmul(qb, kb.transpose(-1, -2)) * float(scale)
        scores = scores.masked_fill(not_allowed, float("-inf"))
        row_max = scores.amax(dim=-1)
        dead = torch.isinf(row_max) & (row_max < 0)
        safe_max = torch.where(dead, torch.zeros_like(row_max), row_max)
        p = torch.exp(scores - safe_max[..., None])
        p = torch.where(allowed[None, None], p, torch.zeros_like(p))
        denom = p.sum(dim=-1)
        o_c = torch.matmul(p, vb) / torch.where(dead, torch.ones_like(denom), denom)[..., None]
        o[:, h0:h1] = torch.where(dead[..., None], torch.zeros_like(o_c), o_c)
        lse[:, h0:h1] = torch.where(dead, torch.full_like(safe_max, float("-inf")), safe_max + torch.log(denom))
    return o.transpose(1, 2), lse


def _compute_ref_backward():
    """The fp8 SDPA row's reference (``test/python/sdpa/fp8_ref.py``), imported lazily: ``sdpa`` is a namespace package of
    the python test tree, on ``sys.path`` under pytest (the tree's root conftest) and put there for a standalone harness."""
    try:
        from sdpa.fp8_ref import compute_ref_backward
    except ImportError:
        import os
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
        from sdpa.fp8_ref import compute_ref_backward
    return compute_ref_backward


class _QuantGrad(torch.autograd.Function):
    """Identity forward; the backward is one of the block's GRADIENT quantization points: ``g -> deq(e4m3(bf16(g) * scale),
    1 / scale)`` -- the bf16 rounding FIRST (the quantize launches read bf16 buffers), then the saturating RNE e4m3 cast
    (:func:`quant_e4m3`, bit-exact vs the kernels), then the exact dequantization in fp64."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        g8 = quant_e4m3(g.to(torch.bfloat16), ctx.scale)
        return (g8.to(torch.float64) * (1.0 / ctx.scale)).to(g.dtype), None


def _ste_e4m3(x: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """A FORWARD quantization point as a straight-through estimator: the VALUE is what the kernels consume -- the e4m3 code
    of the bf16-rounded ``x`` at ``scale``, dequantized exactly in fp64 -- and the gradient passes through unchanged.
    Returns ``(x_ste, x8)`` with ``x8`` the e4m3 codes (``[B, S, H, D]``, contiguous)."""
    x8 = quant_e4m3(x.detach().to(torch.bfloat16), scale).contiguous()
    fq = x8.to(torch.float64) * (1.0 / float(scale))
    return x + (fq - x.detach()), x8


def _ste_codes(x: torch.Tensor, x8_given: Optional[torch.Tensor], scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`_ste_e4m3` with the e4m3 codes GIVEN -- the record's own operand (``[T, H, D]`` or ``[B, S, H, D]`` e4m3, the codes
    the record's LSE / O were computed from and the backward recomputes bit-exactly): the VALUE at the point is their exact fp64
    dequantization at ``scale``, the gradient passes through unchanged.  ``None`` falls back to this oracle's own cast."""
    if x8_given is None:
        return _ste_e4m3(x, scale)
    if x8_given.dtype != FP8_E4M3:
        raise ValueError(f"a given forward operand must be e4m3 codes, got {x8_given.dtype}")
    x8 = x8_given.detach().reshape(x.shape).contiguous()
    fq = x8.to(torch.float64) * (1.0 / float(scale))
    return x + (fq - x.detach()), x8


@dataclass(frozen=True)
class _Fp8RowCfg:
    """The plan-time facts the SDPA row Function needs: the block's band, the forward's static scales, the step's gradient
    scales and the mode."""

    scale: float
    is_causal: bool
    causal_bottom_right: bool
    window_left: int
    window_right: int
    scale_q: float
    scale_k: float
    scale_v: float
    scale_s: float
    scale_do: float
    scale_dp: float
    modelled: bool
    # Appended, defaulted: the MEMORY shape of the SDPA node at long S (the arithmetic is the same at every value).
    fwd_head_chunk: Optional[int] = None  # q heads per fp64 attention pass (:func:`fp64_attention`'s ``head_chunk``; None = all)
    bwd_group_chunk: Optional[int] = None  # KV-head GROUPS per ``compute_ref_backward`` call of the modelled backward (None = all)


def fp8_row_mask_args(is_causal: bool, causal_bottom_right: bool, window_left: int) -> tuple:
    """``compute_ref_backward``'s ``(left_bound, right_bound, diag_align)`` for the block's band.  The block's ``window_left =
    W`` keeps ``k >= q + diag - W``; the row's reference masks ``rel <= diag - left_bound`` (``rel = k - q``), so
    ``left_bound = W + 1`` (the adapter takes ``window_size_left = left_bound - 1``); ``right_bound = 0`` under a causal
    mask (``window_right > 0`` is declined by the block and the row alike), ``None`` dense; the diagonal alignment is
    cuDNN's enum (``BOTTOM_RIGHT`` only with a causal mask; a no-op for the block's self-attention)."""
    import cudnn

    right = 0 if is_causal else None
    left = None if int(window_left) < 0 else int(window_left) + 1
    align = None
    if is_causal:
        align = cudnn.diagonal_alignment.BOTTOM_RIGHT if causal_bottom_right else cudnn.diagonal_alignment.TOP_LEFT
    return left, right, align


class _Fp8SdpaRow(torch.autograd.Function):
    """The SDPA stage of the quantized block backward as ONE autograd node.

    Forward: the fp64 attention (:func:`fp64_attention`) over the straight-through ``q / k / v`` (their values are the
    dequantized e4m3 operands the kernels read) -> ``O`` and the exact LSE.  Backward, by mode:

    * ``modelled`` (M): ``do8 = e4m3(bf16(dO) * scale_do)`` (the gate backward writes bf16 dO, the dO quantize reads it),
      ``delta`` = the given tensor (the block's own, fp32 ``[B, H_q, S]``) or ``rowsum(bf16(dO) * bf16(O))`` in fp32, then
      the row's reference ``compute_ref_backward`` over the e4m3 ``q8 / k8 / v8 / do8`` with the forward's LSE (the given
      fp32 one, else this node's fp64 LSE), the block's band, ``scale_s`` on P, ``scale_dp`` on dS (``quantize_ds=True``: the
      shipped e4m3-dS chain), unit gradient scales; the fp32 dQ / dK / dV are ROUNDED to bf16 -- the row's output dtype --
      and returned in fp64.  ``amax_dp`` is ``max |dS|`` in fp32 before the ``scale_dp`` cast (the row's ``amax_dP``
      contract), taken off the reference's own ``ds_scaled`` intermediate.  The row's dead ``o`` / ``descale_o`` are fed
      ``do8`` / 1.0 -- read by nothing once ``delta`` is given, exactly as the stage binds them.
    * unmodelled (U): the exact fp64 adjoint over the dequantized operands with the exact fp64 ``delta = rowsum(dO * O)``
      -- no e4m3 P / dS / dO point (informational); ``amax_dp = max |dS|`` in fp64.
    * ``seeded``: the block's OWN bf16 ``dq / dk / dv`` are returned (in fp64), so everything downstream of the SDPA stage is
      judged under the bf16 block's bound; ``amax_dp`` is ``None``.
    * ``o_record`` (appended, last, defaulted so a twelve-argument caller still works): the record's bf16 pre-gate O.  When given, a ``delta`` computed here is
      ``rowsum(bf16(dO) * O_record)`` -- the gate backward's operands -- instead of this node's fp64 O rounded to bf16.
    * ``do_record`` (appended after it, defaulted): the block's own bf16 dO -- the gate backward's output, the tensor the dO quantize
      READ and the one the block's ``delta`` is the row-sum of.  When given, the modelled branch casts IT (``do8`` is then bitwise the
      block's) and a ``delta`` computed here uses it; this node's incoming fp64 gradient is used only by the unmodelled branch.  Without
      it the modelled stage pairs the block's ``delta`` with a ``do8`` cast from a once-rounded fp64 dO -- the two bf16 roundings of the
      block's chain (B2, then the gate multiply) flip a few per cent of the codes, and ``dS = P (dP - delta)`` is then inconsistent.

    Every quantity the mode produced lands in ``holder`` (``o``, ``lse``, ``do8``, ``delta``, ``dq / dk / dv``,
    ``amax_dp``) for the caller's stage-localised assertions.  The GQA grouping is the block's: q head ``i`` reads kv head
    ``i // (H_q / H_kv)``, the reference's contiguous groups.

    Memory: the forward runs :func:`fp64_attention` ``cfg.fwd_head_chunk`` q heads at a time and the modelled backward
    runs the row's reference one KV-head group (``cfg.bwd_group_chunk`` of them) at a time with amax-only intermediates --
    nothing of size ``[B, H_q, S, S]`` is held at either default the block oracle picks (its one-pass form needed 256 GiB
    for 32 heads at S = 16K); the arithmetic is the same at every chunking.  The unmodelled (U) branch still holds the fp64
    ``[B, H_q, S, S]`` matrices: informational, small shapes only."""

    @staticmethod
    def forward(ctx, q, k, v, q8, k8, v8, allowed, cfg, lse_given, delta_given, seeded, holder, o_record=None, do_record=None):
        o, lse = fp64_attention(q, k, v, allowed, cfg.scale, head_chunk=cfg.fwd_head_chunk)
        ctx.save_for_backward(q, k, v, q8, k8, v8, o, lse, allowed)
        ctx.cfg, ctx.lse_given, ctx.delta_given, ctx.seeded, ctx.holder, ctx.o_record = cfg, lse_given, delta_given, seeded, holder, o_record
        ctx.do_record = do_record
        holder.update(o=o.detach(), lse=lse.detach())
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, q8, k8, v8, o, lse, allowed = ctx.saved_tensors
        cfg, holder = ctx.cfg, ctx.holder
        b, s, hq, d = q.shape
        hkv = k.shape[2]
        do64 = do.contiguous()
        # The block's bf16 dO when given (the tensor the dO quantize read and delta is the row-sum of), else this node's gradient rounded once.
        do_bf16 = (do64 if ctx.do_record is None else ctx.do_record.detach().reshape(do64.shape)).to(torch.bfloat16)
        o_bf16 = (o if ctx.o_record is None else ctx.o_record.detach().reshape(o.shape)).to(torch.bfloat16)
        # The block's delta (B3): rowsum(bf16 dO * bf16 O) in fp32 per (b, h, q) -- the SAME tensor the kernel consumed when given.
        if ctx.delta_given is not None:
            delta = ctx.delta_given.detach().float().reshape(b, hq, s).contiguous()
        else:
            delta = (do_bf16.float() * o_bf16.float()).sum(-1).permute(0, 2, 1).contiguous()
        do8 = quant_e4m3(do_bf16, cfg.scale_do).contiguous()
        holder.update(do8=do8, do_bf16=do_bf16, delta=delta)
        if ctx.seeded is not None:
            dq, dk, dv = (ctx.seeded[n].detach().to(torch.float64).reshape(x.shape) for n, x in (("dq", q), ("dk", k), ("dv", v)))
            holder.update(dq=dq, dk=dk, dv=dv, amax_dp=None)
        elif cfg.modelled:
            compute_ref_backward = _compute_ref_backward()
            stats = (ctx.lse_given.detach().float() if ctx.lse_given is not None else lse.float()).reshape(b, hq, s, 1).contiguous()
            left, right, align = fp8_row_mask_args(cfg.is_causal, cfg.causal_bottom_right, cfg.window_left)
            # Per KV-head GROUP (one kv head and the q heads that read it: dK / dV sum over exactly those), on head SLICES of the
            # fp32 operands converted ONCE -- the row reference's own `_prepare` cast, hoisted: a slice of the full fp32 tensor
            # keeps the strides the whole-tensor call hands cuBLAS (lda = H_q * D), so every per-head GEMM of the reference sees
            # the operands it would see unchunked, and the result is bitwise the one-call form's.  `return_intermediates="amax"`
            # keeps only the running max |ds_scaled| (the one number read below) instead of the (b, h_q, s_q, s_kv) collect
            # (72 GiB for 32 heads at S = 16K, twice that while it concatenates).
            rep = hq // hkv
            gpc = hkv if cfg.bwd_group_chunk is None else max(1, min(hkv, int(cfg.bwd_group_chunk)))
            q32, k32, v32, do32 = q8.float(), k8.float(), v8.float(), do8.float()
            dq = torch.empty(b, s, hq, d, dtype=torch.float64, device=q.device)
            dk = torch.empty(b, s, hkv, d, dtype=torch.float64, device=q.device)
            dv = torch.empty(b, s, hkv, v.shape[-1], dtype=torch.float64, device=q.device)
            ds_amax, dp_amax_raw = None, 0.0
            for kv0 in range(0, hkv, gpc):
                kv1 = min(kv0 + gpc, hkv)
                h0, h1 = kv0 * rep, kv1 * rep
                out = compute_ref_backward(
                    q32[:, :, h0:h1], k32[:, :, kv0:kv1], v32[:, :, kv0:kv1], do32[:, :, h0:h1], do32[:, :, h0:h1], cfg.scale,
                    1.0 / cfg.scale_q, 1.0 / cfg.scale_k, 1.0 / cfg.scale_v, cfg.scale_s, 1.0 / cfg.scale_s, FP8_E4M3, 1.0, 1.0 / cfg.scale_do,
                    torch.bfloat16, left_bound=left, right_bound=right, diag_align=align, stats=stats[:, h0:h1], return_intermediates="amax",
                    quantize_ds=True, dP_scale=cfg.scale_dp, quantize_grads=False, delta=delta[:, h0:h1],
                )  # fmt: skip
                dq32, dk32, dv32, _dsink, dp_amax_c, _dq_amax, _dk_amax, _dv_amax, inter = out
                dp_amax_raw = max(dp_amax_raw, float(dp_amax_c))
                ds_amax = inter["ds_scaled"] if ds_amax is None else torch.maximum(ds_amax, inter["ds_scaled"])
                dq[:, :, h0:h1] = dq32.to(torch.bfloat16).to(torch.float64)
                dk[:, :, kv0:kv1] = dk32.to(torch.bfloat16).to(torch.float64)
                dv[:, :, kv0:kv1] = dv32.to(torch.bfloat16).to(torch.float64)
            amax_dp = float(ds_amax.item()) / cfg.scale_dp
            holder.update(dq=dq, dk=dk, dv=dv, amax_dp=amax_dp, amax_dp_raw=dp_amax_raw)
        else:
            rep = hq // hkv
            qb, dob, ob = q.transpose(1, 2), do64.transpose(1, 2), o.transpose(1, 2)
            kb = k.transpose(1, 2).repeat_interleave(rep, 1)
            vb = v.transpose(1, 2).repeat_interleave(rep, 1)
            scores = torch.matmul(qb, kb.transpose(-1, -2)) * cfg.scale
            scores = scores.masked_fill(~allowed[None, None], float("-inf"))
            live = torch.isfinite(lse)
            p = torch.exp(scores - torch.where(live, lse, torch.zeros_like(lse))[..., None])
            p = torch.where(allowed[None, None] & live[..., None], p, torch.zeros_like(p))
            delta64 = (dob * ob).sum(-1, keepdim=True)
            dvb = torch.matmul(p.transpose(-1, -2), dob)
            dpb = torch.matmul(dob, vb.transpose(-1, -2))
            ds = p * (dpb - delta64) * cfg.scale
            dqb = torch.matmul(ds, kb)
            dkb = torch.matmul(ds.transpose(-1, -2), qb)
            dq = dqb.transpose(1, 2)
            dk = dkb.reshape(b, hkv, rep, s, d).sum(2).transpose(1, 2)
            dv = dvb.reshape(b, hkv, rep, s, d).sum(2).transpose(1, 2)
            holder.update(dq=dq, dk=dk, dv=dv, amax_dp=float(ds.abs().max().item()))
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


def gated_attention_block_fp8_bwd_reference(
    inp_q: dict,
    geom: RefGeometry,
    spec,
    dy: torch.Tensor,
    *,
    scale_dy: float,
    scale_do: float,
    scale_dqkvg: float,
    scale_s: float,
    scale_dp: float,
    delta: Optional[torch.Tensor] = None,
    modelled: bool = True,
    seeded: Optional[dict] = None,
    lse: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    q8: Optional[torch.Tensor] = None,
    k8: Optional[torch.Tensor] = None,
    v8: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    fwd_head_chunk: Optional[int] = None,
    bwd_group_chunk: Optional[int] = None,
) -> dict:
    """The oracle of the per-tensor fp8 (e4m3) block BACKWARD over the quantized training record.

    ``inp_q`` is the quantized input dict (e4m3 ``h`` / ``w_qkvg`` / ``w_o``, bf16 norm weights, ``cos`` / ``sin``),
    ``spec`` the forward's ``QuantSpec`` (the descales and the static ``scale_q / k / v / o``), ``dy`` the bf16 output
    gradient.  ``scale_dy / scale_do / scale_dqkvg`` are the block's own gradient scales READ BACK from its scalar block
    (so the oracle models the block's exact e4m3 points); ``scale_s = 2 ** FP8_SCALE_S_LOG2``; ``scale_dp`` the caller's.
    ``delta`` (fp32 ``[B, H_q, S]``) is the block's own ``rowsum(bf16(dO) * bf16(O))`` -- the SAME tensor the kernel
    consumed; ``None`` computes it here.  ``modelled=True`` models every backward quantization point (the fp8 SDPA row's
    reference over the e4m3 ``q8 / k8 / v8 / do8`` with the fp8 dS, then the e4m3 ``dqkvg`` and ``dY`` points);
    ``modelled=False`` keeps the forward's straight-through points only and runs the backward in fp64 (informational).
    ``seeded = dict(dq=, dk=, dv=)`` substitutes the block's OWN bf16 SDPA gradients, so ``dh / dw_*`` are judged under
    the bf16 block's bound.  ``lse`` (appended; fp32 ``[B, H_q, S]``) is the record's exact LSE -- the one the kernel
    recomputes P from; ``None`` uses this oracle's own fp64 LSE over the same dequantized operands (they agree to fp32
    rounding, which the e4m3 P cast can turn into the rare midpoint flip the fp8 comparison helper budgets).  ``o``
    (appended; the record's bf16 pre-gate O ``[B, S, H_q, D]``) is the O the gate backward READS -- the fp8 forward's output,
    P's e4m3 cast included -- substituted straight-through as the VALUE of the SDPA stage's output (the gradient still flows to
    the SDPA node), so dG, og8 and hence dW_o are composed from the same O as the kernels'; ``None`` keeps this oracle's own
    fp64 attention O, and the forward's unmodelled P cast then reaches dG / og8 / dW_o (a difference of several bf16 bounds at
    the test geometry, measured on the first full run of the block's accept matrix).  ``gate`` (appended; the record's bf16
    GATE band, ``[T, H_q, D]`` or ``[B, S, H_q, D]``) is the gate the gate backward READS -- the forward GEMM's bf16 rounding of
    the projection -- substituted straight-through likewise (the gradient w.r.t. the gate still reaches the slab point); ``None``
    keeps this oracle's exact fp64 projection, whose last bits move ``bf16(O * sigmoid(gate))`` across an e4m3 midpoint on
    0.1-0.6 % of the og8 codes and so a whole column of dW_o each (measured on the same run).  ``q8 / k8 / v8`` (appended; the
    record's e4m3 SDPA operands, ``[T, H, D]`` or ``[B, S, H, D]``: the forward's codes, which the backward recomputes
    bit-exactly) are the operands the record's LSE, O and the block's delta were computed from -- substituted straight-through as
    the VALUES of the three forward STE points, so the modelled SDPA stage runs ``compute_ref_backward`` on the SAME e4m3 operands
    as the kernel (the row suite's own composition); ``None`` keeps this oracle's cast of its fp64 chain, whose bf16-level
    disagreement with the bf16 recompute flips a few per cent of the codes -- and a P recomputed from flipped codes under the
    record's LSE is no longer normalised, so ``dS = P (dP - delta)`` loses its zero-sum structure on every affected row (dh at cos
    0.996 with 75 % of its rows outside the bf16 bound at S = 512 on the first full run, while the same chain SEEDED with the
    block's own dQ / dK / dV sat at 0.16-0.69 of the bound on every cell of that run).  ``do`` (appended; the block's own bf16 dO,
    the gate backward's output, ``[T, H_q, D]`` or ``[B, S, H_q, D]``) is the tensor the dO quantize READ and the block's ``delta`` is
    the row-sum of: the modelled SDPA stage casts it (``do8`` bitwise the block's) so that ``do8`` and ``delta`` are the consistent
    pair the kernel consumed; ``None`` casts this oracle's once-rounded fp64 dO, which disagrees with the block's twice-rounded bf16
    dO on a large fraction of the elements and flips a few per cent of the ``do8`` codes against the block's ``delta`` (with the
    record's codes fed but not the dO, the modelled dh still sat at cos 0.998 with 44 % of its rows outside at S = 512, qk_norm).
    With ``lse / o / gate / q8 / k8 / v8 / do`` all given, the modelled SDPA stage runs the row's reference on exactly the kernel's
    inputs, and the end-to-end difference to the block is the SDPA stage's kernel-vs-reference difference propagated through the
    bf16 chain's modelled casts -- the comparison the modelled oracle is for.

    Returns ``dh, dw_qkvg, dw_o, dw_q_norm, dw_k_norm`` (fp64), ``dq, dk, dv`` (the SDPA stage's bf16-rounded outputs,
    in fp64; ``dv`` is also the slab's V band, which the norm backward copies bit-exactly), ``amax_dp`` (``max |dS|`` in
    fp32 before the cast; ``None`` when seeded), the bands ``dq_pre / dg / dk_pre / do`` (fp64 gradients w.r.t. the slab's
    Q / GATE / K bands and the pre-gate O, ``[T, H, D]``), ``dw_q_norm_mass / dw_k_norm_mass`` (the bf16 suite's noise
    unit) -- plus the quantities a stage-localised check compares the block's own buffers against: ``q8 / k8 / v8 / og8``
    (the forward STE points' e4m3 codes), ``do8``, ``delta`` (fp32 ``[B, H_q, S]``, the one the SDPA stage consumed),
    ``lse`` (this oracle's fp64 LSE), ``o`` (fp64 pre-gate O), ``dq_post / dk_post`` (the post-norm gradients).

    ``fwd_head_chunk`` / ``bwd_group_chunk`` (appended; ``None`` = auto) shape the SDPA node's MEMORY, never its arithmetic:
    the q heads per pass of its fp64 attention forward (auto: :func:`fp64_attention_head_chunk` -- one fp64 ``[B, chunk, S, S]``
    tensor within 2 GiB but never fewer than two heads, so 32 heads at S = 2K are one pass and S = 16K / 32K two heads per
    pass; a trailing one-head remainder is folded into the pass before it, :func:`fp64_attention_head_passes`) and the KV-head
    groups per ``compute_ref_backward`` call of its modelled backward (auto: one).  The one-pass form of the node needed ``4 * B * H_q
    * S^2 * 8`` bytes in the forward (256 GiB at 32 heads, S = 16K) and a 9 B/cell ``[B, H_q, S, S]`` intermediate collect in
    the backward (144 GiB at the concatenation): the chart gate above S = 4K ran against torch for want of a reference.  The
    result is bitwise the same at every chunking whose passes hold two or more heads (verified on Rubin against the one-pass
    form: chunks of 2, 3 and 7 at S = 512 and 2K on the 8-head test geometry, of 2, 3 and 31 at S = 2K and 4K on the 32-head
    397B geometry, of 2, 3 and 9 on a 10-head geometry at S = 2K); ``fwd_head_chunk=1`` is numerically equivalent but moves the last
    fp64 bit of ``o`` (see :func:`fp64_attention`).

    Known modelled difference: the kernel forms dP from the e4m3 ``do8`` while ``delta`` comes from the bf16 dO, so the
    softmax identity ``sum_j P_ij dP_ij = delta_i`` holds only to the dO quantization error; fed the same ``delta`` the
    matrix is consistent, and a residual of that size is the contract, not a bug.

    Composition (M), in the block's order: fp64 chain on ``deq(h8)``, ``deq(W8)`` with ``requires_grad`` leaves; forward
    STE points ``x + (fq(x) - x).detach()`` at ``q / k / v`` (``scale_q / k / v``, on the bf16-rounded values the quantize
    launches read) and ``og`` (``scale_o``, on the bf16-rounded og); the SDPA row as :class:`_Fp8SdpaRow`; backward points as
    :class:`_QuantGrad` at the slab (``scale_dqkvg``) and on the output (``dY``, ``scale_dy``).  Every fp8 cast is the
    saturating RNE :func:`quant_e4m3` (bit-exact vs the kernels).
    """
    b, s, dm = inp_q["h"].shape
    t = b * s
    hq, hkv, d, rd = geom.h_q, geom.h_kv, geom.d_head, geom.rope_dim
    dev = inp_q["h"].device
    if tuple(dy.shape) != (b, s, dm):
        raise ValueError(f"dy must be [B, S, d_model] = {(b, s, dm)}, got {tuple(dy.shape)}")
    if inp_q["h"].dtype != FP8_E4M3 or inp_q["w_qkvg"].dtype != FP8_E4M3 or inp_q["w_o"].dtype != FP8_E4M3:
        raise ValueError("the fp8 backward oracle takes the QUANTIZED input dict: e4m3 h / w_qkvg / w_o")
    if seeded is not None and set(seeded) != {"dq", "dk", "dv"}:
        raise ValueError(f"seeded must be dict(dq=, dk=, dv=), got keys {sorted(seeded)}")

    def leaf(x):
        return None if x is None else x.detach().to(torch.float64).requires_grad_(True)

    # Exact dequantization in fp64 (the e4m3 codes are exact in fp64; the descale is the QuantSpec's plan-time constant).
    h = leaf(inp_q["h"].to(torch.float64) * float(spec.descale_h))
    w_qkvg = leaf(inp_q["w_qkvg"].to(torch.float64) * float(spec.descale_w_qkvg))
    w_o = leaf(inp_q["w_o"].to(torch.float64) * float(spec.descale_w_o))
    w_q, w_k = (leaf(inp_q["w_q_norm"]), leaf(inp_q["w_k_norm"])) if geom.qk_norm else (None, None)
    cos, sin = inp_q["cos"].to(torch.float64), inp_q["sin"].to(torch.float64)
    o_q, o_g, o_k, o_v = geom.offsets

    # (1) the projection; the dqkvg quantization point sits on its GRADIENT (B5+B6 write bf16 bands, the quantize reads them).
    proj = h.reshape(t, dm) @ w_qkvg.t()
    proj_q = _QuantGrad.apply(proj, scale_dqkvg) if modelled else proj
    q_pre = proj_q[:, o_q:o_g].reshape(b, s, hq, d)
    gate_proj = proj_q[:, o_g:o_k].reshape(b, s, hq, d)
    # The record's bf16 GATE band as the VALUE the gate stages read (B3 forms sigmoid(gate) from the record, not from the
    # exact projection), straight-through: the gradient w.r.t. the gate reaches the slab's quantization point unchanged.
    gate = gate_proj if gate is None else gate_proj + (gate.detach().to(torch.float64).reshape(gate_proj.shape) - gate_proj.detach())
    k_pre = proj_q[:, o_k:o_v].reshape(b, s, hkv, d)
    v = proj_q[:, o_v:].reshape(b, s, hkv, d)
    # (2)+(3) norm + RoPE in fp64 (one rounding in the kernel; unrounded here), then the forward's static e4m3 points.
    q, rstd_q = qk_norm_rope_reference(q_pre, w_q, cos, sin, rd, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=torch.float64)
    k, rstd_k = qk_norm_rope_reference(k_pre, w_k, cos, sin, rd, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=torch.float64)
    # The record's own codes when given (the operands its LSE / O belong to), else this oracle's cast of its fp64 chain.
    q_ste, q8 = _ste_codes(q, q8, spec.scale_q)
    k_ste, k8 = _ste_codes(k, k8, spec.scale_k)
    v_ste, v8 = _ste_codes(v, v8, spec.scale_v)
    # (4) the SDPA row, over the block's band.
    allowed = _key_padding_and_causal_mask(
        s,
        s,
        is_causal=geom.is_causal,
        seq_lens=None,
        batch_index=0,
        q_lo=0,
        device=dev,
        window_left=geom.window_left,
        window_right=geom.window_right,
        causal_bottom_right=geom.causal_bottom_right,
        s_q_total=s,
    )
    cfg = _Fp8RowCfg(
        scale=float(geom.scale),
        is_causal=bool(geom.is_causal),
        causal_bottom_right=bool(geom.causal_bottom_right),
        window_left=int(geom.window_left),
        window_right=int(geom.window_right),
        scale_q=float(spec.scale_q),
        scale_k=float(spec.scale_k),
        scale_v=float(spec.scale_v),
        scale_s=float(scale_s),
        scale_do=float(scale_do),
        scale_dp=float(scale_dp),
        modelled=bool(modelled),
        fwd_head_chunk=fp64_attention_head_chunk(b, s, s, hq) if fwd_head_chunk is None else int(fwd_head_chunk),
        bwd_group_chunk=1 if bwd_group_chunk is None else int(bwd_group_chunk),
    )
    holder: dict = {}
    o_record = None if o is None else o.detach()
    do_record = None if do is None else do.detach()
    o_sdpa = _Fp8SdpaRow.apply(
        q_ste,
        k_ste,
        v_ste,
        q8,
        k8,
        v8,
        allowed,
        cfg,
        None if lse is None else lse.detach(),
        None if delta is None else delta.detach(),
        seeded,
        holder,
        o_record,
        do_record,
    )
    # (4b) the record's pre-gate O as the VALUE the gate stages read (B3 forms dG and og8 from the bf16 record O -- the fp8
    # forward's output -- not from an fp64 attention), straight-through: the gradient passes to the SDPA node unchanged.
    o = o_sdpa if o_record is None else o_sdpa + (o_record.to(torch.float64).reshape(o_sdpa.shape) - o_sdpa.detach())
    # (5) the gate, then the forward's og point on the bf16-rounded og (the gate kernel writes bf16 og; quantize_o reads it).
    og = o * torch.sigmoid(gate)
    og_ste, og8 = _ste_e4m3(og, spec.scale_o)
    # (6) the out projection; the dY quantization point sits on the OUTPUT's gradient.
    out = og_ste.reshape(t, hq * d) @ w_o.t()
    out_q = _QuantGrad.apply(out, scale_dy) if modelled else out

    wanted = [h, w_qkvg, w_o] + ([w_q, w_k] if geom.qk_norm else []) + [q_pre, k_pre, gate, v, o, q, k]
    grads = list(torch.autograd.grad(out_q, wanted, dy.detach().to(torch.float64).reshape(t, dm)))
    res = dict(dh=grads.pop(0), dw_qkvg=grads.pop(0), dw_o=grads.pop(0))
    if geom.qk_norm:
        res.update(dw_q_norm=grads.pop(0), dw_k_norm=grads.pop(0))
    else:
        res.update(dw_q_norm=None, dw_k_norm=None)
    res.update(
        dq_pre=grads.pop(0).reshape(t, hq, d),
        dk_pre=grads.pop(0).reshape(t, hkv, d),
        dg=grads.pop(0).reshape(t, hq, d),
        dv_band=grads.pop(0).reshape(t, hkv, d),
        do=grads.pop(0).reshape(t, hq, d),
    )
    dq_post, dk_post = grads.pop(0), grads.pop(0)  # w.r.t. the post-norm / post-RoPE q / k: the SDPA stage's outputs, the norm backward's inputs
    if geom.qk_norm:
        res["dw_q_norm_mass"] = dw_norm_noise_mass(dq_post, q_pre, rstd_q, cos, sin, rd)
        res["dw_k_norm_mass"] = dw_norm_noise_mass(dk_post, k_pre, rstd_k, cos, sin, rd)
    else:
        res["dw_q_norm_mass"] = res["dw_k_norm_mass"] = None
    res.update(
        dq=holder["dq"],
        dk=holder["dk"],
        dv=holder["dv"],
        amax_dp=holder["amax_dp"],
        q8=q8,
        k8=k8,
        v8=v8,
        og8=og8,
        do8=holder["do8"],
        delta=holder["delta"],
        lse=holder["lse"],
        o=holder["o"],
        dq_post=dq_post.detach(),
        dk_post=dk_post.detach(),
    )
    return res


# ---------------------------------------------------------------------------
# The MXFP8 block BACKWARD oracle (the fp8 oracle's sibling over the MXFP8 training record)
# ---------------------------------------------------------------------------


def _mxfp8_compute_ref_backward():
    """The MXFP8 SDPA row's reference (``test/python/sdpa/mxfp8_ref.py``), imported lazily like :func:`_compute_ref_backward`."""
    try:
        from sdpa.mxfp8_ref import compute_ref_backward
    except ImportError:
        import os
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
        from sdpa.mxfp8_ref import compute_ref_backward
    return compute_ref_backward


def _mx_unswizzle_128x4(blob: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The inverse of ``mxfp8_quant._swizzle_128x4`` on a ``[rows, cols]`` logical scale matrix (``rows % 128 == 0``, ``cols % 4 == 0``)
    whose F8_128x4 bytes are ``blob`` (any shape of the same byte count): the atoms are stored ``(rt, ct, rr, rg, cc)``, the logical
    matrix is ``(rt, rg, rr, ct, cc)``."""
    if rows % MX_ATOM_ROWS or cols % MX_ATOM_COLS:
        raise ValueError(f"an F8_128x4 matrix needs rows % 128 == 0 and cols % 4 == 0, got {rows} x {cols}")
    if blob.numel() != rows * cols:
        raise ValueError(f"blob has {blob.numel()} bytes, the F8_128x4 matrix is {rows} x {cols}")
    v = blob.reshape(rows // MX_ATOM_ROWS, cols // MX_ATOM_COLS, 32, 4, MX_ATOM_COLS)
    return v.permute(0, 3, 2, 1, 4).reshape(rows, cols)


def mx_sf_ref_of(blob: torch.Tensor, layout: str, b: int, h: int, s_real: int, d: int, block: int = MX_BLOCK) -> torch.Tensor:
    """Per-element fp32 dequant scales ``[b*h, s_real, d]`` (the ``sf_*_ref`` form ``mxfp8_ref.compute_ref_backward`` takes) of a
    kernel scale-factor blob in either SDPA layout -- the two layouts the row suite's ``_Quant`` hands the graph and the block's
    quantizer kernel writes:

    * ``"rowwise"``: the ``[B, H, ceil128(S), D/32]`` F8_128x4 blob of a Q / K / V / dO payload (one 1 KiB tile per (b, h, 128-row
      tile); ``mxfp8_quant.swizzle_sf_rowwise`` of the logical ``[B*H*ceil128(S), D/32]`` bytes);
    * ``"columnwise"``: the ``[B, H, ceil128(S)/32, D]`` blob of a q_T / k_T / dO_T payload, D-plane-major
      (``swizzle_sf_columnwise``: the 128x4 rule on the TRANSPOSED logical matrix ``[D, B*H*ceil128(S)/32]``).

    An E8M0 byte ``e`` is the exact scale ``2^(e-127)``, so the result is bitwise ``quantize_to_mxfp8``'s ``sf_d_ref`` / ``sf_s_ref``
    of the source the blob was quantized from (pinned by the stage tests)."""
    mq = _mxfp8_quant()
    l = b * h
    s_pad = -(-s_real // MX_ATOM_ROWS) * MX_ATOM_ROWS
    groups = d // block
    if d % block:
        raise ValueError(f"D={d} must be a multiple of the {block}-element block")
    raw = blob.detach()
    if raw.dtype != torch.uint8:
        raw = raw.view(torch.uint8) if raw.dtype in (torch.int8, torch.float8_e8m0fnu, torch.float8_e4m3fn) else raw.to(torch.uint8)
    raw = raw.reshape(-1)
    if layout == "rowwise":
        e = _mx_unswizzle_128x4(raw, l * s_pad, groups)  # [l*s_pad, D/32]
        sf = torch.repeat_interleave(mq.e8m0_to_float(e).reshape(l, s_pad, groups), block, dim=2)
        return sf[:, :s_real].contiguous()
    if layout == "columnwise":
        e_t = _mx_unswizzle_128x4(raw, d, l * s_pad // block)  # [D, l*s_pad/32]: the transposed logical matrix the swizzle was applied to
        e = e_t.transpose(0, 1).contiguous()  # [l*s_pad/32, D]
        sf = torch.repeat_interleave(mq.e8m0_to_float(e).reshape(l, s_pad // block, d), block, dim=1)
        return sf[:, :s_real].contiguous()
    raise ValueError(f"layout must be 'rowwise' or 'columnwise', got {layout!r}")


def _mx_points(x_bshd, *, codes_d=None, sf_d=None, codes_s=None, sf_s=None) -> dict:
    """The two MX quantizations of a bf16-ROUNDED ``[B, S, H, D]`` activation as the SDPA row consumes them -- ``"d"``: ROWWISE
    (32-element blocks along D: q / k / v / dO), ``"s"``: COLUMNWISE (32-token blocks along S on the tensor S-padded to whole 128-row
    tiles, as the kernel sees it: q_T / k_T / dO_T) -- each as ``(codes, ref)``: e4m3 codes ``[B, H, S, D]`` (the reference's layout)
    and the per-element fp32 dequant scales ``[B, H, S, D]``.  This oracle's own cast is ``quantize_to_mxfp8`` (bitwise the row
    suite's ``_Quant`` and the block's quantizer kernel); the block's OWN codes (``[B, S, H, D]`` or ``[T, H, D]`` e4m3) with their
    SDPA-layout blob substitute it when given -- both of a pair or neither (a code without its blob cannot be dequantized)."""
    b, s, h, d = x_bshd.shape
    mq = _mxfp8_quant()
    own = None
    if codes_d is None or codes_s is None:
        fd, sd_ref, _blob_d, fs, ss_ref, _blob_s = mq.quantize_to_mxfp8(x_bshd.detach().float().permute(0, 2, 1, 3), b, h, s, d, fp8_dtype=FP8_E4M3)
        own = {"d": (fd, sd_ref), "s": (fs, ss_ref)}
    out = {}
    for key, codes, blob, layout in (("d", codes_d, sf_d, "rowwise"), ("s", codes_s, sf_s, "columnwise")):
        if (codes is None) != (blob is None):
            raise ValueError(
                f"the {layout} MX point takes its e4m3 codes AND their scale-factor blob, or neither (got codes {codes is not None}, blob {blob is not None})"
            )
        if codes is None:
            c, ref = own[key]
        else:
            if codes.dtype != FP8_E4M3:
                raise ValueError(f"a given MX payload must be e4m3 codes, got {codes.dtype}")
            c = codes.detach().reshape(b, s, h, d).permute(0, 2, 1, 3).contiguous()
            ref = mx_sf_ref_of(blob, layout, b, h, s, d)
        out[key] = (c.contiguous(), ref.reshape(b, h, s, d).contiguous())
    return out


def _mx_deq_bshd(point) -> torch.Tensor:
    """``(codes [B, H, S, D] e4m3, ref [B, H, S, D] fp32)`` -> the exact fp64 value ``[B, S, H, D]`` (a code times a power of two)."""
    codes, ref = point
    return (codes.to(torch.float64) * ref.to(torch.float64)).permute(0, 2, 1, 3)


@dataclass(frozen=True)
class _MxRowCfg:
    """The plan-time facts the MXFP8 SDPA row Function needs: the block's band, the mode, the fold model, the memory shape."""

    scale: float
    is_causal: bool
    causal_bottom_right: bool
    window_left: int
    window_right: int
    modelled: bool
    # "kernel": the row's fold modelled PER GRADIENT -- dK once-rounded from fp32 partials, dV from per-Q-head bf16 partials summed
    # in the fold kernel's fixed order; "once": every gradient once-rounded (the row's reference as it is; informational).
    fold: str = "kernel"
    fwd_head_chunk: Optional[int] = None  # q heads per fp64 attention pass (:func:`fp64_attention`'s ``head_chunk``; None = all)
    bwd_group_chunk: Optional[int] = None  # KV-head GROUPS per ``compute_ref_backward`` call (None = one)


def mx_fold_dv_kernel_order(dv_parts: torch.Tensor, group: int) -> torch.Tensor:
    """The MXFP8 row's dV fold as ``dkv_reduce`` spells it (``bprop_chain_common._reduce_group_vec``): per KV head, the group's
    per-Q-head partials -- ``dv_parts`` ``[B, S, H_q, D]`` in any float dtype, the kernel's being bf16 -- each rounded to bf16, summed
    in fp32 from zero with the q heads ASCENDING (head ``kv * group + g`` for ``g = 0 .. group - 1``), the sum rounded to bf16
    once.  Returns bf16 ``[B, S, H_kv, D]``.  Pinned bitwise against ``dkv_reduce_host`` run on those partials."""
    b, s, hq, d = dv_parts.shape
    if hq % group:
        raise ValueError(f"H_q={hq} is not a multiple of the group {group}")
    hkv = hq // group
    acc = torch.zeros(b, s, hkv, d, dtype=torch.float32, device=dv_parts.device)
    for g in range(group):
        acc = acc + dv_parts[:, :, g::group].to(torch.bfloat16).float()
    return acc.to(torch.bfloat16)


class _MxProjection(torch.autograd.Function):
    """The projection ``h @ W_qkvg^T`` of the MXFP8 block backward as ONE autograd node.  Forward: the fp64 product of the
    dequantized leaves.  Backward (M): the block's two dQKVG quantization points -- the gradient ``g`` ``[T, N]`` is rounded to bf16
    (the GEMM pack writes bf16 bands) and block-quantized TWICE, once along N (``dqkvg8``, the dgrad's A operand) and once along T
    (``dqkvg_t8``, the wgrad's A operand) -- and each product reads the CALLER's transposed artifact dequantized through its blob:
    ``dh = fq_N(bf16 g) @ deq(w_qkvg_t)^T`` (``w_qkvg_t`` ``[d_model, N]``), ``dW = fq_T(bf16 g)^T @ deq(h_t)^T`` (``h_t`` ``[d_model, T]``).
    A single :class:`_QuantGrad` cannot express two gradients of one tensor, hence the Function.  Every quantity lands in ``holder``
    (the e4m3 codes and logical E8M0 bytes of both casts) for the stage-localised bitwise checks."""

    @staticmethod
    def forward(ctx, h2d, w, deq_w_t, deq_h_t, holder):
        ctx.save_for_backward(deq_w_t, deq_h_t)
        ctx.holder = holder
        return h2d @ w.t()

    @staticmethod
    def backward(ctx, g):
        deq_w_t, deq_h_t = ctx.saved_tensors
        mq = _mxfp8_quant()
        g16 = g.detach().to(torch.bfloat16).float().contiguous()  # [T, N], the bf16 bands the quantize launches read
        codes_n, e_n = mx_quantize_rowwise_2d(g16)  # along N: the dgrad's A
        g_n = codes_n.to(torch.float64) * torch.repeat_interleave(mq.e8m0_to_float(e_n), MX_BLOCK, dim=-1).to(torch.float64)  # [T, N]
        dh = g_n @ deq_w_t.t()  # [T, dm]
        if deq_h_t is None:
            # no wgrad model (see deq_t): at a T with no whole 32-token blocks the block refuses dW_qkvg and quantizes no dQKVG^T
            codes_t = e_t = dw = None
        else:
            codes_t, e_t = mx_quantize_rowwise_2d(g16.t().contiguous())  # along T, physically [N, T]: the wgrad's A
            g_t = codes_t.to(torch.float64) * torch.repeat_interleave(mq.e8m0_to_float(e_t), MX_BLOCK, dim=-1).to(torch.float64)  # [N, T]
            dw = g_t @ deq_h_t.t()  # [N, dm]
        ctx.holder.update(dqkvg8=codes_n, dqkvg_e=e_n, dqkvg_t8=codes_t, dqkvg_t_e=e_t)
        return dh, dw, None, None, None


class _Fp4OutProj(torch.autograd.Function):
    """The out projection ``O_gated @ W_o^T`` of the MXFP8 block backward under an fp4 ``W_o`` (``MxQuantSpec.o_fp4``) as ONE autograd
    node -- the two dY points of that arm are DIFFERENT, which a single :class:`_QuantGrad` cannot express.  Forward: the fp64 product
    of the fp4 STRAIGHT-THROUGH value of the gated O (``og_fp4``: the bf16 ``O_gated`` block-quantized to e2m1 in ``fmt`` and
    dequantized through its padded blob -- what the forward's fp4 out projection multiplied) and the dequantized e2m1 ``W_o`` leaf.
    Backward (M), ``g`` ``[T, d_model]`` rounded to bf16 first (the quantize launches read a bf16 dY):

    * B1 ``dW_o = fq_pt(dY)^T @ og8``: the PER-TENSOR e4m3 point at ``scale_dy`` (``quant_e4m3(bf16 g, scale_dy) / scale_dy``) against the
      gate backward's per-tensor e4m3 ``og8`` (``scale_o = 1`` under ``o_fp4``), dequantized -- the weight gradient stays 8-bit;
    * B2 ``dO_gated = fq_A(dY) @ deq(w_o_t, w_o_t_sf)``: the BLOCK point of the caller's transposed e2m1 ``W_o`` row -- NVFP4: the
      TWO-LEVEL cast ``fq_nvfp4(bf16 g, global_scale=scale_dy)`` (the kernel's pre-scale) dequantized through its padded e4m3-per-16
      blob, the product multiplied by ``1 / scale_dy`` (the gate backward's descale arm; exact for the power-of-two ``scale_dy``);
      MXFP4: the MX-rowwise e4m3 point (``mx_quantize_rowwise_2d``, E8M0 per 32) against ``deq(w_o_t)``, no scale anywhere.

    Every quantity lands in ``holder``: ``dy8`` (the per-tensor codes ``[T, d_model]``), ``dy4`` / ``dy4_e`` / ``dy4_sf`` (NVFP4: packed
    e2m1 uint8 ``[T, d_model/2]``, the logical e4m3 scale bytes ``[T, d_model/16]``, the padded blob) or ``dy_mx8`` / ``dy_mx_e`` /
    ``dy_mx_sf`` (MXFP4: e4m3 codes ``[T, d_model]``, E8M0 bytes ``[T, d_model/32]``, the padded canonical blob) -- the stage-localised
    bitwise checks' comparands."""

    @staticmethod
    def forward(ctx, og2d, w_o, og_fp4, og8_deq, deq_w_o_t, scale_dy, fmt_name, holder):
        ctx.save_for_backward(og8_deq, deq_w_o_t)
        ctx.scale_dy, ctx.fmt_name, ctx.holder = float(scale_dy), str(fmt_name), holder
        return og_fp4 @ w_o.t()

    @staticmethod
    def backward(ctx, g):
        og8_deq, deq_w_o_t = ctx.saved_tensors
        mq = _mxfp8_quant()
        sdy = ctx.scale_dy
        g16 = g.detach().to(torch.bfloat16).float().contiguous()  # [T, d_model]: the bf16 dY the quantize launches read
        # B1's point: the per-tensor e4m3 cast at scale_dy, dequantized exactly (alpha_b1 = descale_dy x descale_o, descale_o = 1 under o_fp4)
        dy8 = quant_e4m3(g16, sdy)
        g_pt = dy8.to(torch.float64) * (1.0 / sdy)
        dw_o = g_pt.t() @ og8_deq  # [d_model, H_q*D]
        # B2's point: the block cast of the caller's e2m1 W_o row
        if ctx.fmt_name == "nvfp4":
            packed, e = fp4_quantize_rowwise_2d(g16, "nvfp4", global_scale=sdy)  # the two-level cast: scale_dy x dY, e4m3 scales per 16
            blob = mx_swizzle_sf_rowwise_padded(e, 16)
            g_blk = fp4_dequant_rowwise_2d(packed, blob, "nvfp4", out_dtype=torch.float64)  # = the dequantized scale_dy x dY
            dog = (g_blk @ deq_w_o_t.t()) * (1.0 / sdy)  # the gate backward's descale arm, exact for a power of two
            ctx.holder.update(dy8=dy8, dy4=packed, dy4_e=e, dy4_sf=blob, dy_mx8=None, dy_mx_e=None, dy_mx_sf=None)
        else:
            codes, e = mx_quantize_rowwise_2d(g16)  # MX rowwise: E8M0 per 32 along d_model
            g_blk = codes.to(torch.float64) * torch.repeat_interleave(mq.e8m0_to_float(e), MX_BLOCK, dim=-1).to(torch.float64)
            dog = g_blk @ deq_w_o_t.t()
            ctx.holder.update(dy8=dy8, dy4=None, dy4_e=None, dy4_sf=None, dy_mx8=codes, dy_mx_e=e, dy_mx_sf=mx_swizzle_sf_rowwise_padded(e))
        return dog, dw_o, None, None, None, None, None, None


class _MxSdpaRow(torch.autograd.Function):
    """The SDPA stage of the MXFP8 block backward as ONE autograd node (the sibling of :class:`_Fp8SdpaRow`).

    Forward: the fp64 attention (:func:`fp64_attention`) over the straight-through ``q / k / v`` -> ``O`` and the exact LSE.
    Backward, by mode:

    * ``modelled`` (M): the row's reference ``mxfp8_ref.compute_ref_backward`` over the e4m3 MX payloads the kernel reads -- the
      rowwise ``q8 / k8 / v8`` (the backward's V is quantized ROWWISE: the dP operand), the rowwise ``do8`` and the columnwise
      ``q_T8 / k_T8 / do_T8`` with their per-element scales -- with the forward's LSE (the given fp32 one, else this node's fp64 LSE),
      the block's band, P at the fixed 2^8 scale, dS quantized per 1x32 block both ways (``quantize_ds=True``: the shipped
      block-scaled chain) and the SAME ``delta`` the kernel consumed (given; else ``rowsum(bf16 dO * bf16 O)`` in fp32, which IS the
      row's own pre-pass).  ``dO``'s two points are cast from the block's bf16 ``dO`` when given (``do_record``: the gate backward's
      output, the tensor the quantize launches read), else from this node's gradient rounded to bf16.  Per KV-head group: ONE
      reference call over the group -> dQ (per head) and dK (the group's partials summed in fp32 and rounded ONCE: what the row
      computes from its fp32 dK partials, and the reference as it is) -- and, under ``fold == "kernel"``, a SECOND call with K / V
      repeated per Q head (every head its own group of one) -> the per-Q-head fp32 partials, from which dV is folded as the row's
      ``dkv_reduce`` folds its bf16 partials (:func:`mx_fold_dv_kernel_order`: each partial rounded to bf16, the group summed in fp32
      in head order, rounded once more).  ``fold == "once"`` takes the group call's once-rounded dV (informational).  Every gradient
      is returned in fp64, rounded to bf16 (the row's output dtype).
    * unmodelled (U): the exact fp64 adjoint over the dequantized operands with the exact fp64 ``delta = rowsum(dO * O)`` -- no MX
      point in the backward (informational).
    * ``seeded``: the block's OWN bf16 ``dq / dk / dv`` are returned (in fp64), so everything downstream is judged under the bf16
      block's bound.

    Everything the mode produced lands in ``holder``: ``o``, ``lse``, ``do8 / do_T8`` (codes, ``[B, S, H, D]``) with ``sf_do / sf_do_T``
    (per-element scales, ``[B, H, S, D]``), ``delta``, ``dq / dk / dv``, ``dv_once`` (the once-rounded dV, for the per-cell distance),
    ``dq_parts / dk_parts / dv_parts`` (the per-Q-head fp32 partials of the per-head call, ``[B, S, H_q, D]`` fp64; None under
    ``fold == "once"``).  The GQA grouping is the block's: q head ``i`` reads kv head ``i // (H_q / H_kv)``.  Memory: the forward runs
    ``cfg.fwd_head_chunk`` q heads per fp64 pass, the backward ``cfg.bwd_group_chunk`` KV-head groups per reference call (the
    reference is blocked over KV and never holds a ``[B, H, S, S]`` matrix); the (U) branch does, small shapes only."""

    @staticmethod
    def forward(ctx, q, k, v, allowed, cfg, lse_given, delta_given, seeded, holder, o_record, do_record, points):
        o, lse = fp64_attention(q, k, v, allowed, cfg.scale, head_chunk=cfg.fwd_head_chunk)
        ctx.save_for_backward(q, k, v, o, lse, allowed)
        ctx.cfg, ctx.lse_given, ctx.delta_given, ctx.seeded, ctx.holder = cfg, lse_given, delta_given, seeded, holder
        ctx.o_record, ctx.do_record, ctx.points = o_record, do_record, points
        holder.update(o=o.detach(), lse=lse.detach())
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, allowed = ctx.saved_tensors
        cfg, holder, pts = ctx.cfg, ctx.holder, ctx.points
        b, s, hq, d = q.shape
        hkv = k.shape[2]
        rep = hq // hkv
        do64 = do.contiguous()
        do_bf16 = (do64 if ctx.do_record is None else ctx.do_record.detach().reshape(do64.shape)).to(torch.bfloat16)
        o_bf16 = (o if ctx.o_record is None else ctx.o_record.detach().reshape(o.shape)).to(torch.bfloat16)
        # The block's delta: rowsum(bf16 dO * bf16 O) in fp32 per (b, h, q) -- the SAME tensor the kernel consumed when given; on this
        # row that row-sum IS the kernel's own pre-pass (dot_do_o over the bf16 ports), so the two spellings agree bitwise.
        if ctx.delta_given is not None:
            delta = ctx.delta_given.detach().float().reshape(b, hq, s).contiguous()
        else:
            delta = (do_bf16.float() * o_bf16.float()).sum(-1).permute(0, 2, 1).contiguous()
        do_pts = _mx_points(do_bf16, codes_d=pts.get("do8"), sf_d=pts.get("sf_do"), codes_s=pts.get("do_T8"), sf_s=pts.get("sf_do_T"))
        holder.update(
            do_bf16=do_bf16,
            delta=delta,
            do8=do_pts["d"][0].permute(0, 2, 1, 3).contiguous(),
            do_T8=do_pts["s"][0].permute(0, 2, 1, 3).contiguous(),
            sf_do=do_pts["d"][1],
            sf_do_T=do_pts["s"][1],
        )
        if ctx.seeded is not None:
            dq, dk, dv = (ctx.seeded[n].detach().to(torch.float64).reshape(x.shape) for n, x in (("dq", q), ("dk", k), ("dv", v)))
            holder.update(dq=dq, dk=dk, dv=dv, dv_once=None, dq_parts=None, dk_parts=None, dv_parts=None)
        elif cfg.modelled:
            compute_ref_backward = _mxfp8_compute_ref_backward()
            stats = (ctx.lse_given.detach().float() if ctx.lse_given is not None else lse.float()).reshape(b, hq, s, 1).contiguous()
            left, right, align = fp8_row_mask_args(cfg.is_causal, cfg.causal_bottom_right, cfg.window_left)
            q8, sf_q = pts["q"]
            q_T8, sf_q_T = pts["q_T"]
            k8, sf_k = pts["k"]
            k_T8, sf_k_T = pts["k_T"]
            v8, sf_v = pts["v"]
            do8, sf_do = do_pts["d"]
            do_T8, sf_do_T = do_pts["s"]
            o_ref, do_ref = o_bf16.permute(0, 2, 1, 3).contiguous(), do_bf16.permute(0, 2, 1, 3).contiguous()  # [B, H, S, D]: read by nothing (delta given)
            gpc = hkv if cfg.bwd_group_chunk is None else max(1, min(hkv, int(cfg.bwd_group_chunk)))
            dq = torch.empty(b, s, hq, d, dtype=torch.float64, device=q.device)
            dk = torch.empty(b, s, hkv, d, dtype=torch.float64, device=q.device)
            dv = torch.empty(b, s, hkv, v.shape[-1], dtype=torch.float64, device=q.device)
            dv_once = torch.empty_like(dv)
            per_head = cfg.fold == "kernel"
            dq_parts = torch.empty(b, s, hq, d, dtype=torch.float64, device=q.device) if per_head else None
            dk_parts = torch.empty_like(dq_parts) if per_head else None
            dv_parts = torch.empty_like(dq_parts) if per_head else None
            common = dict(torch_itype=FP8_E4M3, left_bound=left, right_bound=right, diag_align=align, quantize_ds=True)
            for kv0 in range(0, hkv, gpc):
                kv1 = min(kv0 + gpc, hkv)
                h0, h1 = kv0 * rep, kv1 * rep
                hs, ks = slice(h0, h1), slice(kv0, kv1)
                # (i) the group: dQ per head and the once-rounded dK / dV -- the row's reference as it is (bitwise the row suite's)
                out = compute_ref_backward(
                    q8[:, hs], q_T8[:, hs], k8[:, ks], k_T8[:, ks], v8[:, ks], o_ref[:, hs], do_ref[:, hs], do8[:, hs], do_T8[:, hs], cfg.scale,
                    sf_q[:, hs].contiguous(), sf_q_T[:, hs].contiguous(), sf_k[:, ks].contiguous(), sf_k_T[:, ks].contiguous(), sf_v[:, ks].contiguous(),
                    sf_do[:, hs].contiguous(), sf_do_T[:, hs].contiguous(), torch_otype=torch.bfloat16, stats=stats[:, hs], delta=delta[:, hs], **common,
                )  # fmt: skip
                dq[:, :, hs] = out[0].permute(0, 2, 1, 3).to(torch.float64)
                dk[:, :, ks] = out[1].permute(0, 2, 1, 3).to(torch.float64)
                dv_once[:, :, ks] = out[2].permute(0, 2, 1, 3).to(torch.float64)
                if per_head:
                    # (ii) every q head its own group of one (K / V repeated per head): the per-Q-head fp32 partials the kernel folds
                    rk, rkt, rv = (x[:, ks].repeat_interleave(rep, dim=1).contiguous() for x in (k8, k_T8, v8))
                    rsk, rskt, rsv = (x[:, ks].repeat_interleave(rep, dim=1).contiguous() for x in (sf_k, sf_k_T, sf_v))
                    outh = compute_ref_backward(
                        q8[:, hs], q_T8[:, hs], rk, rkt, rv, o_ref[:, hs], do_ref[:, hs], do8[:, hs], do_T8[:, hs], cfg.scale,
                        sf_q[:, hs].contiguous(), sf_q_T[:, hs].contiguous(), rsk, rskt, rsv, sf_do[:, hs].contiguous(), sf_do_T[:, hs].contiguous(),
                        torch_otype=torch.float32, stats=stats[:, hs], delta=delta[:, hs], **common,
                    )  # fmt: skip
                    dq_parts[:, :, hs] = outh[0].permute(0, 2, 1, 3).to(torch.float64)
                    dk_parts[:, :, hs] = outh[1].permute(0, 2, 1, 3).to(torch.float64)
                    dv_parts[:, :, hs] = outh[2].permute(0, 2, 1, 3).to(torch.float64)
                    dv[:, :, ks] = mx_fold_dv_kernel_order(outh[2].permute(0, 2, 1, 3), rep).to(torch.float64)
                else:
                    dv[:, :, ks] = dv_once[:, :, ks]
            holder.update(dq=dq, dk=dk, dv=dv, dv_once=dv_once, dq_parts=dq_parts, dk_parts=dk_parts, dv_parts=dv_parts)
        else:
            qb, dob, ob = q.transpose(1, 2), do64.transpose(1, 2), o.transpose(1, 2)
            kb = k.transpose(1, 2).repeat_interleave(rep, 1)
            vb = v.transpose(1, 2).repeat_interleave(rep, 1)
            scores = torch.matmul(qb, kb.transpose(-1, -2)) * cfg.scale
            scores = scores.masked_fill(~allowed[None, None], float("-inf"))
            live = torch.isfinite(lse)
            p = torch.exp(scores - torch.where(live, lse, torch.zeros_like(lse))[..., None])
            p = torch.where(allowed[None, None] & live[..., None], p, torch.zeros_like(p))
            delta64 = (dob * ob).sum(-1, keepdim=True)
            dvb = torch.matmul(p.transpose(-1, -2), dob)
            dpb = torch.matmul(dob, vb.transpose(-1, -2))
            ds = p * (dpb - delta64) * cfg.scale
            dqb = torch.matmul(ds, kb)
            dkb = torch.matmul(ds.transpose(-1, -2), qb)
            dq = dqb.transpose(1, 2)
            dk = dkb.reshape(b, hkv, rep, s, d).sum(2).transpose(1, 2)
            dv = dvb.reshape(b, hkv, rep, s, d).sum(2).transpose(1, 2)
            holder.update(dq=dq, dk=dk, dv=dv, dv_once=None, dq_parts=None, dk_parts=None, dv_parts=None)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def gated_attention_block_mxfp8_bwd_reference(
    inp_mx: dict,
    geom: RefGeometry,
    spec,
    dy: torch.Tensor,
    *,
    scale_dy: float,
    delta: Optional[torch.Tensor] = None,
    modelled: bool = True,
    seeded: Optional[dict] = None,
    lse: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    q8: Optional[torch.Tensor] = None,
    sf_q: Optional[torch.Tensor] = None,
    q_T8: Optional[torch.Tensor] = None,
    sf_q_T: Optional[torch.Tensor] = None,
    k8: Optional[torch.Tensor] = None,
    sf_k: Optional[torch.Tensor] = None,
    k_T8: Optional[torch.Tensor] = None,
    sf_k_T: Optional[torch.Tensor] = None,
    v8: Optional[torch.Tensor] = None,
    sf_v: Optional[torch.Tensor] = None,
    do8: Optional[torch.Tensor] = None,
    sf_do: Optional[torch.Tensor] = None,
    do_T8: Optional[torch.Tensor] = None,
    sf_do_T: Optional[torch.Tensor] = None,
    h_t: Optional[torch.Tensor] = None,
    h_t_sf: Optional[torch.Tensor] = None,
    w_qkvg_t: Optional[torch.Tensor] = None,
    w_qkvg_t_sf: Optional[torch.Tensor] = None,
    fold: str = "kernel",
    fwd_head_chunk: Optional[int] = None,
    bwd_group_chunk: Optional[int] = None,
    w_o_t: Optional[torch.Tensor] = None,
    w_o_t_sf: Optional[torch.Tensor] = None,
    o_fp4=None,
) -> dict:
    """The oracle of the MXFP8 block BACKWARD over the MXFP8 training record -- the sibling of
    :func:`gated_attention_block_fp8_bwd_reference` (read its docstring for the shared conventions: the record's ``lse / o / gate``
    and the block's ``do`` substituted straight-through, the ``seeded`` mode, the memory knobs).

    ``inp_mx`` is :func:`quantize_block_inputs_mxfp8`'s dict (e4m3 ``h`` / ``w_qkvg`` CODES with their PADDED F8_128x4 blobs ``h_sf``
    / ``w_qkvg_sf``, the per-tensor e4m3 ``w_o``; bf16 norm weights, ``cos`` / ``sin``), ``spec`` the forward's ``MxQuantSpec``
    (``descale_w_o``, ``scale_o``), ``dy`` the bf16 output gradient, ``scale_dy`` the block's "current" dY scale READ BACK from its
    scalar block (the one per-tensor gradient point of this pipeline).  ``delta`` (fp32 ``[B, H_q, S]``) is the block's own
    ``rowsum(bf16 dO * bf16 O)`` -- the SAME tensor the kernel consumed; ``None`` computes it here, and on this row the two are
    bitwise (the MXFP8 kernel's own pre-pass is that dot over its bf16 ports, unlike the fp8 row's).  ``modelled=True`` models every
    backward quantization point; ``modelled=False`` keeps the forward's straight-through points only and runs the backward in fp64
    (informational).

    **The block's own payloads.**  ``q8 / sf_q`` .. ``do_T8 / sf_do_T`` (the kernel's seven e4m3 payloads with their SDPA-layout
    scale-factor blobs, ``[B, S, H, D]`` or ``[T, H, D]`` codes) substitute this oracle's own MX casts straight-through as the VALUES
    of the forward points (``q / k`` rowwise) and as the operands of the modelled SDPA stage (every point), so the stage runs the
    row's reference on exactly the kernel's inputs; a pair (codes, blob) is taken whole or not at all.  ``v8 / sf_v`` is the
    backward's ROWWISE V (the dP operand) -- the forward consumed V columnwise, and the forward STE value of V here is the
    columnwise fake-quant (:func:`mx_fake_quant_v_columnwise`), the quantization the record's O / LSE came from.

    **The caller's artifacts.**  ``h_t / h_t_sf`` (e4m3 ``[d_model, T]`` + the padded blob over ``(rows = d_model, K = T)``) and
    ``w_qkvg_t / w_qkvg_t_sf`` (``[d_model, N]`` + the blob over ``(d_model, N)``) are what the block's dgrad and wgrad read; the (M)
    oracle dequantizes them THROUGH their blobs (:func:`mx_dequant_rowwise_2d`), never the forward's ``h`` / ``W_qkvg`` -- a wrong
    caller blob is a wrong oracle, never a silent agreement.  ``None`` quantizes the dequantized leaves along the transposed axis
    here (the oracle's stand-in for a caller); at a ``T`` that is no multiple of 32 there is no stand-in for ``h_t`` (no whole
    32-token blocks) and the block refuses the projection weight gradient there, so the oracle returns ``dw_qkvg=None`` and no
    transposed dQKVG quantization (``dqkvg_t8 / dqkvg_t_e / dqkvg_t_sf`` ``None``) -- the dgrad-only block's shape of the result.

    **The fold.**  Under GQA the MXFP8 SDPA backward folds its per-Q-head dK partials in fp32 and rounds the sum once, like the
    reference, while its per-Q-head dV partials are bf16 (the kernel stores them from its epilogue; fp32 ones do not fit its 327 KiB
    shared-memory budget), so dV carries one bf16 rounding per group member where a once-rounded reference carries one in total
    (relative RMS about 3e-3 at a group of 4, the geometry the tests run, measured on the per-tensor fp8 row before it moved to
    fp32 partials); the modelled oracle folds dV the same way and the distance to a once-rounded fold is reported per cell.
    ``fold="kernel"`` (default) is that model -- dK once-rounded, dV through :func:`mx_fold_dv_kernel_order` over the per-Q-head
    partials of a per-head reference call --, ``fold="once"`` rounds every gradient once (the row's reference as it is, bitwise the
    row suite's; informational).  The result carries BOTH dV forms (``dv`` and ``dv_once``) and the per-Q-head partials.

    Known modelled difference, as the fp8 oracle's: the kernel forms dP from the rowwise e4m3 ``do8`` while ``delta`` is the bf16
    dO's row-sum, so ``sum_j P_ij dP_ij = delta_i`` holds only to the dO quantization error; fed the same ``delta`` the matrix is
    consistent, and a residual of that size is the contract, not a bug.

    **The fp4 weight modes** (appended: ``w_o_t`` / ``w_o_t_sf`` / ``o_fp4``, and an e2m1 ``inp_mx["w_qkvg"]``).  An MXFP4 ``W_qkvg``
    (``inp_mx["w_qkvg"]`` packed ``float4_e2m1fn_x2`` ``[N, d_model/2]`` with its E8M0 / 32 ``w_qkvg_sf``) is a leaf dequantized through its
    blob (:func:`fp4_dequant_rowwise_2d`), and ``w_qkvg_t`` is then the caller's packed e2m1 ``[d_model, N/2]`` (``w_qkvg_t_sf`` E8M0 / 32)
    dequantized the same way -- the (M) dgrad reads it, never the forward's row-quantized weight: two fake-quants of one master weight
    along its two axes, the fp4 training recipe's straight-through estimator (the stand-in at ``None`` quantizes the dequantized leaf's
    transpose to MXFP4).  An fp4 ``W_o`` (``o_fp4`` an ``Fp4Format`` member or its name; ``inp_mx["w_o"]`` packed e2m1 ``[d_model, H_q*D/2]``
    with ``inp_mx["w_o_sf"]`` in that format; ``spec.scale_o == spec.descale_w_o == 1.0``) makes the forward STE point of the gated O the
    fp4 one (the bf16 ``O_gated`` block-quantized in ``o_fp4``'s format and dequantized through its padded blob -- what the forward's fp4
    out projection multiplied, as :func:`gated_attention_block_mxfp8_reference` models it) and the out projection the ONE node
    :class:`_Fp4OutProj` with the arm's two dY points -- B1 per-tensor e4m3 at ``scale_dy`` against the per-tensor ``og8``, B2 the block
    cast of the caller's e2m1 ``w_o_t`` row: NVFP4 the TWO-LEVEL cast ``fq_nvfp4(bf16 dY, global_scale=scale_dy)`` with the product
    descaled by ``1 / scale_dy`` (exactly the kernel chain: the quantize's pre-scale slot read, the gate backward's descale arm), MXFP4 the
    MX-rowwise e4m3 point.  ``w_o_t`` / ``w_o_t_sf`` (packed e2m1 ``[H_q*D, d_model/2]`` + the blob over ``(H_q*D, d_model)`` at the format's
    block) are dequantized through their blob; ``None`` quantizes the dequantized ``W_o`` leaf's transpose in ``o_fp4``'s format (the
    stand-in).  ``modelled=False`` keeps the fp4 forward point and differentiates the out projection in fp64.  The result then also
    carries ``w_o_t_deq`` (fp64 ``[H_q*D, d_model]``), ``og_fp4`` (the fp4 STE value of the gated O, fp64 ``[T, H_q*D]``), ``o_fp4`` (the
    format's name) and the dY block point's codes / scale bytes / blob (``dy4`` / ``dy4_e`` / ``dy4_sf`` under NVFP4, ``dy_mx8`` / ``dy_mx_e``
    / ``dy_mx_sf`` under MXFP4; ``dy8`` the per-tensor codes) under ``modelled=True``.

    Returns the fp8 oracle's dict (``dh, dw_qkvg, dw_o, dw_q_norm, dw_k_norm`` fp64; ``dq, dk, dv`` the SDPA stage's bf16-rounded
    outputs in fp64; the bands; the norm noise masses; ``q8 / k8 / v8 / og8 / do8 / delta / lse / o / dq_post / dk_post``) plus:
    ``amax_dp = None`` (no dP scalar on this row), ``q_T8 / k_T8 / do_T8`` (the columnwise codes, ``[B, S, H, D]``), ``sf_q / sf_q_T
    / sf_k / sf_k_T / sf_v / sf_do / sf_do_T`` (per-element fp32 scales ``[B, H, S, D]`` of every MX point), ``dv_once``, ``dq_parts /
    dk_parts / dv_parts`` (fp64 ``[B, S, H_q, D]``, the per-Q-head partials of the per-head call; None under ``fold="once"``),
    ``dqkvg8 / dqkvg_e`` (the dQKVG gradient's e4m3 codes ``[T, N]`` and logical E8M0 bytes ``[T, N/32]``, blocks along N), ``dqkvg_t8 /
    dqkvg_t_e`` (``[N, T]`` / ``[N, T/32]``, blocks along T) and ``dqkvg_sf / dqkvg_t_sf`` (their padded canonical F8_128x4 blobs,
    :func:`mx_swizzle_sf_rowwise_padded`), ``h_t_deq / w_qkvg_t_deq`` (fp64, what the (M) dgrad / wgrad read)."""
    b, s, dm = inp_mx["h"].shape
    t = b * s
    hq, hkv, d, rd = geom.h_q, geom.h_kv, geom.d_head, geom.rope_dim
    dev = inp_mx["h"].device
    n_qkvg = geom.n_qkvg
    if tuple(dy.shape) != (b, s, dm):
        raise ValueError(f"dy must be [B, S, d_model] = {(b, s, dm)}, got {tuple(dy.shape)}")
    fp4x2 = torch.float4_e2m1fn_x2
    w_qkvg_fp4 = inp_mx["w_qkvg"].dtype == fp4x2
    if inp_mx["h"].dtype != FP8_E4M3 or inp_mx["w_qkvg"].dtype not in (FP8_E4M3, fp4x2) or "h_sf" not in inp_mx or "w_qkvg_sf" not in inp_mx:
        raise ValueError(
            "the MXFP8 backward oracle takes the MXFP8 input dict: e4m3 h CODES and e4m3 (or, under the MXFP4 weight mode, packed e2m1) w_qkvg "
            "CODES with their h_sf / w_qkvg_sf blobs"
        )
    fmt_name = None if o_fp4 is None else fp4_format(o_fp4)[0]
    if o_fp4 is None:
        if inp_mx["w_o"].dtype != FP8_E4M3:
            raise ValueError(f"the MXFP8 backward oracle takes the per-tensor e4m3 w_o without o_fp4 (an fp4 W_o is the o_fp4 arm), got {inp_mx['w_o'].dtype}")
        if w_o_t is not None or w_o_t_sf is not None:
            raise ValueError("w_o_t / w_o_t_sf are the fp4 W_o arm's artifacts (o_fp4); without o_fp4 the out projection is per-tensor e4m3 and reads w_o")
    else:
        if inp_mx["w_o"].dtype != fp4x2 or "w_o_sf" not in inp_mx:
            raise ValueError(f"o_fp4={fmt_name}: the oracle takes the packed e2m1 w_o (float4_e2m1fn_x2) with its w_o_sf blob, got {inp_mx['w_o'].dtype}")
        if float(spec.scale_o) != 1.0 or float(spec.descale_w_o) != 1.0:
            raise ValueError(f"o_fp4: a block-scaled O / W_o has no per-tensor scale; got scale_o={spec.scale_o}, descale_w_o={spec.descale_w_o}")
        if (w_o_t is None) != (w_o_t_sf is None):
            raise ValueError("w_o_t takes its e2m1 codes AND their scale-factor blob, or neither")
    if fold not in ("kernel", "once"):
        raise ValueError(f"fold must be 'kernel' (the row's fold modelled per gradient) or 'once' (every gradient once-rounded), got {fold!r}")
    if seeded is not None and set(seeded) != {"dq", "dk", "dv"}:
        raise ValueError(f"seeded must be dict(dq=, dk=, dv=), got keys {sorted(seeded)}")
    for name, codes, blob in (("h_t", h_t, h_t_sf), ("w_qkvg_t", w_qkvg_t, w_qkvg_t_sf)):
        if (codes is None) != (blob is None):
            raise ValueError(f"{name} takes its codes AND their scale-factor blob, or neither")
    mq = _mxfp8_quant()

    def leaf(x):
        return None if x is None else x.detach().to(torch.float64).requires_grad_(True)

    def as_u8(codes):
        return codes.view(torch.uint8) if codes.dtype != torch.uint8 else codes

    # Exact dequantization in fp64 THROUGH the blobs (a code times a power of two is exact; an e2m1 value times its e4m3 or E8M0 scale too).
    h_deq = mx_dequant_rowwise_2d(inp_mx["h"].reshape(t, dm), inp_mx["h_sf"])  # [T, dm] fp32
    if w_qkvg_fp4:
        w_deq = fp4_dequant_rowwise_2d(as_u8(inp_mx["w_qkvg"]), inp_mx["w_qkvg_sf"], "mxfp4")  # [N, dm] fp32: the MXFP4 weight through its blob
    else:
        w_deq = mx_dequant_rowwise_2d(inp_mx["w_qkvg"], inp_mx["w_qkvg_sf"])  # [N, dm] fp32
    h = leaf(h_deq.reshape(b, s, dm))
    w_qkvg = leaf(w_deq)
    if o_fp4 is None:
        w_o_deq = dequant_e4m3(inp_mx["w_o"], float(spec.descale_w_o))
    else:
        w_o_deq = fp4_dequant_rowwise_2d(as_u8(inp_mx["w_o"]), inp_mx["w_o_sf"], o_fp4)  # [dm, HD] fp32: the fp4 W_o through its blob
    w_o = leaf(w_o_deq)
    w_q, w_k = (leaf(inp_mx["w_q_norm"]), leaf(inp_mx["w_k_norm"])) if geom.qk_norm else (None, None)
    cos, sin = inp_mx["cos"].to(torch.float64), inp_mx["sin"].to(torch.float64)
    o_q, o_g, o_k, o_v = geom.offsets

    # The caller's transposed artifacts, dequantized through THEIR blobs; the oracle's stand-in quantizes the dequantized leaves along
    # the transposed axis (tokens for h, N for W_qkvg, d_model for W_o) exactly as a caller would its bf16 tensors -- in the weight's
    # own format (e4m3 MX, or the fp4 format of the mode: `fp4_fmt`).
    def deq_t(codes, blob, own_src, rows, k, fp4_fmt=None):
        if codes is None:
            if k % MX_BLOCK != 0:
                # No stand-in exists: the transposed axis has no whole 32-element blocks at this k, so the GEMM that would read
                # the artifact is the one the block refuses there (the projection weight gradient at T % 32 != 0) -- its gradient
                # is not modelled: the oracle returns dw_qkvg=None and no transposed dQKVG quantization, as the dgrad-only block
                # takes no dw_qkvg buffer at such a T.
                return None
            src_t = own_src.detach().float().t().contiguous()
            if fp4_fmt is not None:
                _, blk, _ = fp4_format(fp4_fmt)
                packed, e = fp4_quantize_rowwise_2d(src_t, fp4_fmt)
                return fp4_dequant_rowwise_2d(packed, mx_swizzle_sf_rowwise_padded(e, blk), fp4_fmt, out_dtype=torch.float64).contiguous()
            c, e = mx_quantize_rowwise_2d(src_t)
            return (c.to(torch.float64) * torch.repeat_interleave(mq.e8m0_to_float(e), MX_BLOCK, dim=-1).to(torch.float64)).contiguous()
        if fp4_fmt is not None:
            if codes.dtype not in (fp4x2, torch.uint8) or tuple(codes.shape) != (rows, k // 2):
                raise ValueError(
                    f"a transposed e2m1 artifact must be packed [{rows}, {k} // 2] (K-major, two codes per byte), got {codes.dtype} {tuple(codes.shape)}"
                )
            return fp4_dequant_rowwise_2d(as_u8(codes), blob, fp4_fmt, out_dtype=torch.float64).contiguous()
        if codes.dtype != FP8_E4M3 or tuple(codes.shape) != (rows, k):
            raise ValueError(f"a transposed artifact must be e4m3 [{rows}, {k}] (K-major), got {codes.dtype} {tuple(codes.shape)}")
        return mx_dequant_rowwise_2d(codes, blob).to(torch.float64).contiguous()

    deq_h_t = deq_t(h_t, h_t_sf, h_deq, dm, t)  # [dm, T]
    deq_w_t = deq_t(w_qkvg_t, w_qkvg_t_sf, w_deq, dm, n_qkvg, "mxfp4" if w_qkvg_fp4 else None)  # [dm, N]
    deq_w_o_t = deq_t(w_o_t, w_o_t_sf, w_o_deq, hq * d, dm, o_fp4) if o_fp4 is not None else None  # [HD, dm]
    holder: dict = {}
    # (1) the projection; the two dQKVG quantization points sit on its GRADIENT (B5+B6 write bf16 bands, the quantize launches read them).
    h2d = h.reshape(t, dm)
    proj = _MxProjection.apply(h2d, w_qkvg, deq_w_t, deq_h_t, holder) if modelled else h2d @ w_qkvg.t()
    q_pre = proj[:, o_q:o_g].reshape(b, s, hq, d)
    gate_proj = proj[:, o_g:o_k].reshape(b, s, hq, d)
    gate = gate_proj if gate is None else gate_proj + (gate.detach().to(torch.float64).reshape(gate_proj.shape) - gate_proj.detach())
    k_pre = proj[:, o_k:o_v].reshape(b, s, hkv, d)
    v = proj[:, o_v:].reshape(b, s, hkv, d)
    # (2)+(3) norm + RoPE in fp64 (one rounding in the kernel; unrounded here), then the forward's MX points on the bf16-rounded values
    # the quantize launches read: q / k ROWWISE (the record's own codes when given), v COLUMNWISE (the forward's BMM2 operand).
    q, rstd_q = qk_norm_rope_reference(q_pre, w_q, cos, sin, rd, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=torch.float64)
    k, rstd_k = qk_norm_rope_reference(k_pre, w_k, cos, sin, rd, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=torch.float64)
    q_pts = _mx_points(q.detach().to(torch.bfloat16), codes_d=q8, sf_d=sf_q, codes_s=q_T8, sf_s=sf_q_T)
    k_pts = _mx_points(k.detach().to(torch.bfloat16), codes_d=k8, sf_d=sf_k, codes_s=k_T8, sf_s=sf_k_T)
    v_pts = _mx_points(v.detach().to(torch.bfloat16), codes_d=v8, sf_d=sf_v)  # the backward's rowwise V; its columnwise point is unused
    q_ste = q + (_mx_deq_bshd(q_pts["d"]) - q.detach())
    k_ste = k + (_mx_deq_bshd(k_pts["d"]) - k.detach())
    v_ste = v + (mx_fake_quant_v_columnwise(v.detach().to(torch.bfloat16).float()).to(torch.float64) - v.detach())
    points = dict(q=q_pts["d"], q_T=q_pts["s"], k=k_pts["d"], k_T=k_pts["s"], v=v_pts["d"], do8=do8, sf_do=sf_do, do_T8=do_T8, sf_do_T=sf_do_T)
    # (4) the SDPA row, over the block's band.
    allowed = _key_padding_and_causal_mask(
        s,
        s,
        is_causal=geom.is_causal,
        seq_lens=None,
        batch_index=0,
        q_lo=0,
        device=dev,
        window_left=geom.window_left,
        window_right=geom.window_right,
        causal_bottom_right=geom.causal_bottom_right,
        s_q_total=s,
    )
    cfg = _MxRowCfg(
        scale=float(geom.scale),
        is_causal=bool(geom.is_causal),
        causal_bottom_right=bool(geom.causal_bottom_right),
        window_left=int(geom.window_left),
        window_right=int(geom.window_right),
        modelled=bool(modelled),
        fold=fold,
        fwd_head_chunk=fp64_attention_head_chunk(b, s, s, hq) if fwd_head_chunk is None else int(fwd_head_chunk),
        bwd_group_chunk=1 if bwd_group_chunk is None else int(bwd_group_chunk),
    )
    o_record = None if o is None else o.detach()
    do_record = None if do is None else do.detach()
    o_sdpa = _MxSdpaRow.apply(
        q_ste,
        k_ste,
        v_ste,
        allowed,
        cfg,
        None if lse is None else lse.detach(),
        None if delta is None else delta.detach(),
        seeded,
        holder,
        o_record,
        do_record,
        points,
    )
    # (4b) the record's pre-gate O as the VALUE the gate stages read, straight-through.
    o = o_sdpa if o_record is None else o_sdpa + (o_record.to(torch.float64).reshape(o_sdpa.shape) - o_sdpa.detach())
    # (5) the gate, then the forward's per-tensor og point on the bf16-rounded og (the gate kernel writes bf16 og; quantize_o reads it).
    og = o * torch.sigmoid(gate)
    og_ste, og8 = _ste_e4m3(og, float(spec.scale_o))
    og_fp4 = None
    if o_fp4 is None:
        # (6) the out projection (per-tensor fp8, D1); the dY quantization point sits on the OUTPUT's gradient.
        out = og_ste.reshape(t, hq * d) @ w_o.t()
        out_q = _QuantGrad.apply(out, scale_dy) if modelled else out
    else:
        # (5q') / (6') the fp4 W_o arm: the forward's VALUE is the fp4 fake-quant of the bf16 gated O (the fp4 quantize's input; what the
        # fp4 out projection multiplied), B1's operand the per-tensor og8 above (scale_o = 1); the two dY points sit on the out
        # projection's own node (_Fp4OutProj) -- in (U) the fp4 forward point alone, the backward plain fp64.
        _, o_block, _ = fp4_format(o_fp4)
        og16 = og.detach().to(torch.bfloat16).float().reshape(t, hq * d)
        og_packed, og_e = fp4_quantize_rowwise_2d(og16, o_fp4)
        og_fp4 = fp4_dequant_rowwise_2d(og_packed, mx_swizzle_sf_rowwise_padded(og_e, o_block), o_fp4, out_dtype=torch.float64)
        og2d = og.reshape(t, hq * d)
        if modelled:
            og8_deq = og8.reshape(t, hq * d).to(torch.float64) * (1.0 / float(spec.scale_o))
            out = _Fp4OutProj.apply(og2d, w_o, og_fp4, og8_deq, deq_w_o_t, float(scale_dy), fmt_name, holder)
        else:
            out = (og2d + (og_fp4 - og2d.detach())) @ w_o.t()
        out_q = out

    wanted = [h, w_qkvg, w_o] + ([w_q, w_k] if geom.qk_norm else []) + [q_pre, k_pre, gate, v, o, q, k]
    # allow_unused only where the wgrad is not modelled (deq_h_t None): w_qkvg then receives no gradient and dw_qkvg reads None
    grads = list(torch.autograd.grad(out_q, wanted, dy.detach().to(torch.float64).reshape(t, dm), allow_unused=deq_h_t is None))
    res = dict(dh=grads.pop(0), dw_qkvg=grads.pop(0), dw_o=grads.pop(0))
    if geom.qk_norm:
        res.update(dw_q_norm=grads.pop(0), dw_k_norm=grads.pop(0))
    else:
        res.update(dw_q_norm=None, dw_k_norm=None)
    res.update(
        dq_pre=grads.pop(0).reshape(t, hq, d),
        dk_pre=grads.pop(0).reshape(t, hkv, d),
        dg=grads.pop(0).reshape(t, hq, d),
        dv_band=grads.pop(0).reshape(t, hkv, d),
        do=grads.pop(0).reshape(t, hq, d),
    )
    dq_post, dk_post = grads.pop(0), grads.pop(0)
    if geom.qk_norm:
        res["dw_q_norm_mass"] = dw_norm_noise_mass(dq_post, q_pre, rstd_q, cos, sin, rd)
        res["dw_k_norm_mass"] = dw_norm_noise_mass(dk_post, k_pre, rstd_k, cos, sin, rd)
    else:
        res["dw_q_norm_mass"] = res["dw_k_norm_mass"] = None

    def bshd(point):
        return point[0].permute(0, 2, 1, 3).contiguous()

    res.update(
        dq=holder["dq"],
        dk=holder["dk"],
        dv=holder["dv"],
        dv_once=holder.get("dv_once"),
        dq_parts=holder.get("dq_parts"),
        dk_parts=holder.get("dk_parts"),
        dv_parts=holder.get("dv_parts"),
        amax_dp=None,
        q8=bshd(q_pts["d"]),
        q_T8=bshd(q_pts["s"]),
        k8=bshd(k_pts["d"]),
        k_T8=bshd(k_pts["s"]),
        v8=bshd(v_pts["d"]),
        og8=og8,
        do8=holder["do8"],
        do_T8=holder["do_T8"],
        sf_q=q_pts["d"][1],
        sf_q_T=q_pts["s"][1],
        sf_k=k_pts["d"][1],
        sf_k_T=k_pts["s"][1],
        sf_v=v_pts["d"][1],
        sf_do=holder["sf_do"],
        sf_do_T=holder["sf_do_T"],
        delta=holder["delta"],
        lse=holder["lse"],
        o=holder["o"],
        dq_post=dq_post.detach(),
        dk_post=dk_post.detach(),
        h_t_deq=deq_h_t,
        w_qkvg_t_deq=deq_w_t,
        fold=fold,
        w_o_t_deq=deq_w_o_t,
        og_fp4=og_fp4,
        o_fp4=fmt_name,
    )
    if modelled:
        res.update(
            dqkvg8=holder["dqkvg8"],
            dqkvg_e=holder["dqkvg_e"],
            dqkvg_t8=holder["dqkvg_t8"],
            dqkvg_t_e=holder["dqkvg_t_e"],
            dqkvg_sf=mx_swizzle_sf_rowwise_padded(holder["dqkvg_e"]),
            dqkvg_t_sf=None if holder["dqkvg_t_e"] is None else mx_swizzle_sf_rowwise_padded(holder["dqkvg_t_e"]),
        )
    else:
        res.update(dqkvg8=None, dqkvg_e=None, dqkvg_t8=None, dqkvg_t_e=None, dqkvg_sf=None, dqkvg_t_sf=None)
    # the fp4 W_o arm's dY block point (the _Fp4OutProj node's record; None without o_fp4 or under modelled=False)
    for key in ("dy8", "dy4", "dy4_e", "dy4_sf", "dy_mx8", "dy_mx_e", "dy_mx_sf"):
        res[key] = holder.get(key)
    return res
