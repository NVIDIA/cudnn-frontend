# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from .helpers import get_fp8_scale_factor, get_fp8_descale_factor
from .fp16_ref import _ScoreMask, _score_blocks, _qk, _pv, _kv_reduce, _init_softmax_state, _prepare

# fmt: off

# Blocked over KV like fp16_ref (see the note there): the (b, h, s_q, s_kv)
# score/probability matrices are never materialized.


def gqa_kv_head(q_head, h_q, h_kv):
    """The kv head that q head ``q_head`` reads.  Both references group q heads CONTIGUOUSLY -- ``_grouped``
    views (b, h_q, s, d) as (b, h_kv, h_q // h_kv, s, d), so kv head ``h`` serves q heads ``h*g .. h*g+g-1``
    with ``g = h_q // h_kv`` -- and anything that names a (q head, kv head) pair of this reference (the fp8
    flip attribution in ``fp8.assert_close_fp8_grad``) must use this same mapping."""
    return q_head // (h_q // h_kv)


class _Intermediates:
    """Collects, per KV block, the SCALED fp32 values a reference hands to ``.to(torch_itype)`` -- what a
    kernel-vs-reference fp8 rounding flip is a flip OF -- without materializing anything the caller did not
    ask for.

    ``selection`` is ``True`` (the full (b, h_q, s_q, s_kv) matrices; small problems only -- the references
    are blocked over KV precisely so those are never held, b=8 h=8 s=8192 is 17 GB per matrix) or a dict
    with ONE key:
      ``q_rows``:  LongTensor[n, 3] of (b, q_head, i)  -> every collected tensor is [n, s_kv]: row i, all
                   keys j;
      ``kv_cols``: LongTensor[n, 3] of (b, kv_head, j) -> every collected tensor is [n, g, s_q]: column j
                   for the g = h_q // h_kv q heads of that kv head in ascending order (``gqa_kv_head``),
                   all queries i.
    Those are exactly the reduction axes of one output d-row: O[i] / dQ[i] sum over j; dK[j] / dV[j] sum
    over (q head in the group, i)."""

    def __init__(self, selection, b, h_q, h_kv, s_q, s_kv, device):
        if h_q % h_kv != 0:
            raise ValueError(f"h_q={h_q} is not a multiple of h_kv={h_kv}")
        self.h_q, self.h_kv, self.s_q, self.s_kv = h_q, h_kv, s_q, s_kv
        self.g = h_q // h_kv
        self.parts = {}
        if selection is True:
            self.mode = "full"
        elif isinstance(selection, dict) and len(selection) == 1 and next(iter(selection)) in ("q_rows", "kv_cols"):
            self.mode, sel = next(iter(selection.items()))
            self.sel = torch.as_tensor(sel, device=device, dtype=torch.long).reshape(-1, 3)
            heads = h_q if self.mode == "q_rows" else h_kv
            rows = s_q if self.mode == "q_rows" else s_kv
            if self.sel.numel() and bool(((self.sel < 0).any() | (self.sel[:, 0] >= b).any()
                                          | (self.sel[:, 1] >= heads).any() | (self.sel[:, 2] >= rows).any()).item()):
                raise ValueError(f"{self.mode} selection out of range for b={b}, heads={heads}, rows={rows}")
            if self.mode == "kv_cols":
                # [n, g]: the q heads of each selected kv head, gqa_kv_head's inverse.
                self.heads = self.sel[:, 1:2] * self.g + torch.arange(self.g, device=device)[None, :]
        else:
            raise ValueError("return_intermediates must be False, True, {'q_rows': [n, 3]} or {'kv_cols': [n, 3]}")

    def add(self, name, start, end, blk):
        """Record ``blk`` [b, h_q, s_q, end - start], the KV block's slice of intermediate ``name``."""
        if self.mode == "full":
            self.parts.setdefault(name, []).append(blk)
        elif self.mode == "q_rows":
            self.parts.setdefault(name, []).append(blk[self.sel[:, 0], self.sel[:, 1], self.sel[:, 2], :])
        else:
            out = self.parts.get(name)
            if out is None:
                out = self.parts[name] = torch.zeros((self.sel.shape[0], self.g, self.s_q), dtype=blk.dtype, device=blk.device)
            hit = ((self.sel[:, 2] >= start) & (self.sel[:, 2] < end)).nonzero().flatten()
            if hit.numel():
                out[hit] = blk[self.sel[hit, 0][:, None], self.heads[hit], :, (self.sel[hit, 2] - start)[:, None]]

    def rows(self, t):
        """A per-(b, q head, i) tensor [b, h_q, s_q, 1] (running max, row sum) in the collected layout,
        broadcastable against the block tensors."""
        if self.mode == "full":
            return t
        if self.mode == "q_rows":
            return t[self.sel[:, 0], self.sel[:, 1], self.sel[:, 2], :]  # [n, 1]
        return t[self.sel[:, 0][:, None], self.heads, :, 0]  # [n, g, s_q]

    def finalize(self, **extra):
        out = {k: (torch.cat(v, dim=-1) if isinstance(v, list) else v) for k, v in self.parts.items()}
        out.update(extra)
        out.update(h_q=self.h_q, h_kv=self.h_kv, mode=self.mode)
        return out


def compute_ref(q, k, v, attn_scale,
                q_descale, k_descale, v_descale,
                s_scale, s_descale, torch_itype,
                torch_otype,
                padding=None, bias=None,
                left_bound=None, right_bound=None, diag_align=None, sink_token=None,
                rescale_threshold=0.0,
                dtype=torch.float32, quantize_o=True, sink_in_max=True, return_intermediates=False):
    """Compute forward pass reference with online softmax tiling.
    Returns (o_quant, stats, o_amax); ``quantize_o=False`` returns O in fp32 instead of torch_otype.

    ``return_intermediates`` (default False: nothing changes) APPENDS a dict to the return value with the
    SCALED intermediates exactly as this function quantizes them, in the layout ``_Intermediates`` describes
    (``True`` = full [b, h_q, s_q, s_kv]; ``{"q_rows": [n, 3]}`` / ``{"kv_cols": [n, 3]}`` = gathered):
      ``p_scaled``  the value handed to ``.to(torch_itype)``, ``p * s_scale * 2**-rescale_threshold``;
      ``valid``     the score is finite -- unmasked, inside the sequence, not a padded q row.  Every other
                    position holds P = 0 by construction, not by rounding, so it cannot flip;
      ``gain``      the fp32 factor from ONE dequantized P code to O, ``2**rescale_threshold *
                    exp(m_block - m_final) / max(l_final, 1)``: a P code moved by ``u`` at (i, j) moves
                    O[i] by ``u * s_descale * gain[i, j] * V[j] * v_descale`` (0 where the row has no key);
      ``h_q``, ``h_kv``, ``mode``.

    ``dtype`` is the accumulation type. The DLFW containers run fp32 matmul in TF32
    (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1), a 3e-4 relative error on Q@K^T that a smooth softmax
    hides but the fp8 P cast turns into rounding flips; pass float64 when the compare has no
    mismatch budget.

    ``sink_in_max=False`` adds the sink to the row sum after the KV loop instead of seeding the
    running max with it (the FROST SM100 kernels). Same LSE, but the running max sets the scale
    the fp8 P cast rounds at, so the two conventions round P differently on sink rows."""
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape
    device = q.device

    q, k, v = _prepare(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), padding, device, dtype)
    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=bias.float() if bias is not None else None, block_mask=None, is_alibi=False,
                      padding=padding, diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)

    m_old, l_old = _init_softmax_state(b, h_q, s_q, sink_token if sink_in_max else None, device)
    m_old, l_old = m_old.to(dtype), l_old.to(dtype)
    o = torch.zeros((b, h_q, s_q, d_v), dtype=dtype, device=device)

    s_scale_effective = s_scale * (2.0 ** (-rescale_threshold))
    s_descale_effective = s_descale * (2.0 ** rescale_threshold)
    NEG_INF = float('-inf')
    collect = _Intermediates(return_intermediates, b, h_q, h_v, s_q, s_kv, device) if return_intermediates is not False else None

    # Q (FP8) @ K^T (FP8) -> S (FP32)
    for start, end, s_block in _score_blocks(q, k, q_descale * k_descale * attn_scale, mask):
        m_block = s_block.max(dim=-1, keepdim=True).values

        is_first = (m_old == NEG_INF)
        # The kernel's online softmax runs in the log2 domain, so the rescale
        # threshold is in log2 units.
        exceeds_threshold = (m_block - m_old > rescale_threshold * math.log(2))
        should_update = is_first | exceeds_threshold
        m_new = torch.where(should_update, m_block, m_old)

        exp_input = m_old - m_new
        needs_correction = (exp_input < -rescale_threshold * math.log(2))
        correction = torch.where(needs_correction, torch.exp(exp_input), torch.ones_like(exp_input))
        correction = correction.nan_to_num()

        o = o * correction
        l_old = l_old * correction

        p_block = torch.exp(s_block - m_new).nan_to_num()
        if mask.q_row_mask is not None:
            p_block = p_block.masked_fill(mask.q_row_mask, 0.0)
        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # P (FP32) -> P (FP8)
        p_scaled = p_block * s_scale_effective
        p_block_quant = (p_scaled.to(torch_itype)).to(dtype)
        if collect is not None:
            collect.add("p_scaled", start, end, p_scaled)
            collect.add("valid", start, end, torch.isfinite(s_block))
            collect.add("m_at", start, end, m_new.expand(s_block.shape))  # the max this block's P is relative to

        o = o + _pv(p_block_quant, v[:, :, start:end, :], h_v) * v_descale * s_descale_effective
        m_old = m_new
        l_old = l_new

    # o accumulated every block as p_quant * exp(m_block - m_exit) (the corrections telescope); the sink fold
    # below moves l, not o, and the division uses the folded l.
    m_exit = m_old

    if sink_token is not None and not sink_in_max:
        sink = sink_token.to(dtype=dtype, device=device).expand(b, h_q, s_q, 1)
        # A row whose every key is masked has m = -inf: it is all sink (O = 0, LSE = sink).
        m_fin = torch.where(m_old == NEG_INF, sink, m_old)
        l_old = l_old * torch.exp(m_old - m_fin).nan_to_num() + torch.exp(sink - m_fin)
        m_old = m_fin
        if mask.q_row_mask is not None:
            # Padded query rows are dead, not sink-only: the kernels write LSE = -inf there.
            m_old = m_old.masked_fill(mask.q_row_mask, NEG_INF)
            l_old = l_old.masked_fill(mask.q_row_mask, 0.0)

    intermediates = None
    if collect is not None:
        parts = collect.finalize()
        gain = (2.0 ** rescale_threshold) * torch.exp(parts.pop("m_at") - collect.rows(m_exit)) / collect.rows(l_old).clamp(min=1.0)
        intermediates = dict(parts, gain=gain.nan_to_num(0.0))

    o = o / l_old.clamp(min=1.0)
    stats = (m_old + torch.log(l_old)).float()
    o = o.transpose(1, 2)

    o_amax = o.abs().max().item()
    if not quantize_o:
        return (o.float(), stats, o_amax) + ((intermediates,) if collect is not None else ())
    o_scale = get_fp8_scale_factor(o_amax, torch_otype)
    o_quant = (o * o_scale).to(torch_otype)

    return (o_quant, stats, o_amax) + ((intermediates,) if collect is not None else ())


def compute_ref_backward(q, k, v, o, dO, attn_scale,
                         q_descale, k_descale, v_descale,
                         s_scale, s_descale, torch_itype,
                         o_descale, dO_descale,
                         torch_otype,
                         padding=None, bias=None,
                         left_bound=None, right_bound=None, diag_align=None, sink_token=None,
                         stats=None, return_intermediates=False, quantize_ds=True):
    """Compute backward pass reference.
    Returns (dQ, dK, dV, dSink_token, dP_amax, dQ_amax, dK_amax, dV_amax).

    ``quantize_ds`` (default True: the cuDNN backend's recipe -- dS rounded to ``torch_itype`` with ``dP_scale`` before
    the dQ / dK products) set False holds dS in fp32 for those products: the reference for an engine whose dQ / dK
    consume dS in a wider dtype (the FROST sm107 d256 fp8 chain writes dS as bf16), which an e4m3-dS reference would
    misreport by the dS rounding noise alone (0.56 % of dQ / 0.54 % of dK outside atol 0.08 at B1 H2 S512, max |diff|
    0.17, dV untouched).  P is quantized either way.

    ``return_intermediates`` (default False: nothing changes) APPENDS a dict, as ``compute_ref`` does:
    ``p_scaled`` (``p * s_scale``, quantized into dV), ``ds_scaled`` (``dS * dP_scale``, quantized into
    dQ and dK), ``valid``, ``gain=None`` (the gradients carry no normalization: a code moved by ``u`` at
    (i, j) moves dK[j] by ``u * dP_descale * Q[i] * q_descale``, dQ[i] by ``u * dP_descale * K[j] *
    k_descale``, dV[j] by ``u * s_descale * dO[i] * dO_descale``), ``h_q``, ``h_kv``, ``mode``.  Needs
    ``h_k == h_v`` (one GQA group size serves dK and dV)."""
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape
    device = q.device

    q, k, v = _prepare(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), padding, device)
    dO = dO.float().transpose(1, 2)
    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=bias.float() if bias is not None else None, block_mask=None, is_alibi=False,
                      padding=padding, diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)
    qk_scale = q_descale * k_descale * attn_scale
    collect = None
    if return_intermediates is not False:
        if h_k != h_v:
            raise ValueError(f"return_intermediates needs h_k == h_v (got {h_k} / {h_v})")
        collect = _Intermediates(return_intermediates, b, h_q, h_k, s_q, s_kv, device)

    # The backward kernel does not renormalize: it recomputes P = exp(S - stats)
    # from the forward's log-sum-exp, which already accounts for the sink.
    if stats is not None:
        lse = stats.float()
    else:
        m_old, l_old = _init_softmax_state(b, h_q, s_q, sink_token, device)
        for _, _, s_block in _score_blocks(q, k, qk_scale, mask):
            m_new = torch.maximum(m_old, s_block.max(dim=-1, keepdim=True).values)
            l_old = l_old * torch.exp(m_old - m_new).nan_to_num() + torch.exp(s_block - m_new).nan_to_num().sum(dim=-1, keepdim=True)
            m_old = m_new
        lse = m_old + torch.log(l_old)

    D = (o.float() * dO.transpose(1, 2)).sum(dim=-1, keepdim=True).transpose(1, 2) * o_descale * dO_descale


    def dP_block(start, end):
        # dO (FP8) @ V (FP8) -> dP (FP32)
        dP = _qk(dO, v[:, :, start:end, :], h_v)
        return dP * dO_descale * v_descale

    # dP is quantized with one global scale, so its amax needs a pass of its own.
    dP_amax = 0.0
    for start in range(0, s_kv, 128):
        dP_amax = max(dP_amax, dP_block(start, min(start + 128, s_kv)).abs().max().item())
    dP_scale = get_fp8_scale_factor(dP_amax, torch_otype)
    dP_descale = get_fp8_descale_factor(dP_amax, torch_itype)

    dQ = torch.zeros((b, h_q, s_q, d_qk), dtype=torch.float32, device=device)
    dK = torch.zeros((b, h_k, s_kv, d_qk), dtype=torch.float32, device=device)
    dV = torch.zeros((b, h_v, s_kv, d_v), dtype=torch.float32, device=device)

    for start, end, s_block in _score_blocks(q, k, qk_scale, mask):
        p = torch.exp(s_block - lse).nan_to_num()
        if mask.q_row_mask is not None:
            p = p.masked_fill(mask.q_row_mask, 0.0)

        # P (FP32) -> P (FP8); P (FP8) @ dO (FP8) -> dV (FP32)
        p_scaled = p * s_scale
        p_quant = p_scaled.to(torch_itype).float()
        dV[:, :, start:end, :] = _kv_reduce(p_quant, dO, h_v) * s_descale * dO_descale

        dS = p * (dP_block(start, end) - D) * attn_scale
        # dS (FP32) -> dS (FP8)
        ds_scaled = dS * dP_scale
        dS_quant = (ds_scaled.to(torch_itype)).float() if quantize_ds else ds_scaled
        if collect is not None:
            collect.add("p_scaled", start, end, p_scaled)
            collect.add("ds_scaled", start, end, ds_scaled)
            collect.add("valid", start, end, torch.isfinite(s_block))

        # dS (FP8) @ K (FP8) -> dQ (FP32); dS^T (FP8) @ Q (FP8) -> dK (FP32)
        dQ = dQ + _pv(dS_quant, k[:, :, start:end, :], h_k) * k_descale * dP_descale
        dK[:, :, start:end, :] = _kv_reduce(dS_quant, q, h_k) * q_descale * dP_descale

    # Compute dSink_token if sink_token was provided
    # Formula: dSink = -exp(sink - logsumexp) * D summed over batch and sequence
    # Note: attn_scale is NOT applied here because sink_token is added directly to scores,
    # not multiplied by attn_scale like Q @ K.T
    dSink_token = None
    if sink_token is not None:
        p_sink = torch.exp(sink_token.float().expand(b, h_q, s_q, 1) - lse).nan_to_num()
        if mask.q_row_mask is not None:
            p_sink = p_sink.masked_fill(mask.q_row_mask, 0.0)
        dSink_token = (-p_sink * D).sum(dim=(0, 2), keepdim=True)

    dQ = dQ.transpose(1, 2)
    dK = dK.transpose(1, 2)
    dV = dV.transpose(1, 2)

    dQ_amax = dQ.abs().max().item()
    dK_amax = dK.abs().max().item()
    dV_amax = dV.abs().max().item()

    # dQ (FP32) -> dQ (FP8)
    dQ = (dQ * get_fp8_scale_factor(dQ_amax, torch_otype)).to(torch_otype)
    # dK (FP32) -> dK (FP8)
    dK = (dK * get_fp8_scale_factor(dK_amax, torch_otype)).to(torch_otype)
    # dV (FP32) -> dV (FP8)
    dV = (dV * get_fp8_scale_factor(dV_amax, torch_otype)).to(torch_otype)

    out = (dQ, dK, dV, dSink_token, dP_amax, dQ_amax, dK_amax, dV_amax)
    return out if collect is None else out + (collect.finalize(gain=None),)
