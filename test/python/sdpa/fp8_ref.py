# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from .helpers import get_fp8_scale_factor, get_fp8_descale_factor
from .fp16_ref import _ScoreMask, _score_blocks, _pv, _grouped, _init_softmax_state, _prepare

# fmt: off

# Blocked over KV like fp16_ref (see the note there): the (b, h, s_q, s_kv)
# score/probability matrices are never materialized.


def compute_ref(q, k, v, attn_scale,
                q_descale, k_descale, v_descale,
                s_scale, s_descale, torch_itype,
                torch_otype,
                padding=None, bias=None,
                left_bound=None, right_bound=None, diag_align=None, sink_token=None,
                rescale_threshold=0.0):
    """Compute forward pass reference with online softmax tiling.
    Returns (o_quant, stats, o_amax)."""
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape
    device = q.device

    q, k, v = _prepare(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), padding, device)
    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=bias.float() if bias is not None else None, block_mask=None, is_alibi=False,
                      padding=padding, diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)

    m_old, l_old = _init_softmax_state(b, h_q, s_q, sink_token, device)
    o = torch.zeros((b, h_q, s_q, d_v), dtype=torch.float32, device=device)

    s_scale_effective = s_scale * (2.0 ** (-rescale_threshold))
    s_descale_effective = s_descale * (2.0 ** rescale_threshold)
    NEG_INF = float('-inf')

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
        p_block_quant = ((p_block * s_scale_effective).to(torch_itype)).float()

        o = o + _pv(p_block_quant, v[:, :, start:end, :], h_v) * v_descale * s_descale_effective
        m_old = m_new
        l_old = l_new

    o = o / l_old.clamp(min=1.0)
    stats = (m_old + torch.log(l_old)).float()
    o = o.transpose(1, 2)

    o_amax = o.abs().max().item()
    o_scale = get_fp8_scale_factor(o_amax, torch_otype)
    o_quant = (o * o_scale).to(torch_otype)

    return o_quant, stats, o_amax


def compute_ref_backward(q, k, v, o, dO, attn_scale,
                         q_descale, k_descale, v_descale,
                         s_scale, s_descale, torch_itype,
                         o_descale, dO_descale,
                         torch_otype,
                         padding=None, bias=None,
                         left_bound=None, right_bound=None, diag_align=None, sink_token=None,
                         stats=None):
    """Compute backward pass reference.
    Returns (dQ, dK, dV, dSink_token, dP_amax, dQ_amax, dK_amax, dV_amax)."""
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape
    device = q.device

    q, k, v = _prepare(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), padding, device)
    dO = dO.float().transpose(1, 2)
    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=bias.float() if bias is not None else None, block_mask=None, is_alibi=False,
                      padding=padding, diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)
    qk_scale = q_descale * k_descale * attn_scale

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

    dO_g_v = _grouped(dO, h_v)
    q_g_k = _grouped(q, h_k)

    def dP_block(start, end):
        # dO (FP8) @ V (FP8) -> dP (FP32)
        dP = torch.einsum("bhgqd,bhkd->bhgqk", dO_g_v, v[:, :, start:end, :]).reshape(b, h_q, s_q, end - start)
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
        p_quant = (p * s_scale).to(torch_itype).float()
        dV[:, :, start:end, :] = torch.einsum("bhgqk,bhgqd->bhkd", _grouped(p_quant, h_v), dO_g_v) * s_descale * dO_descale

        dS = p * (dP_block(start, end) - D) * attn_scale
        # dS (FP32) -> dS (FP8)
        dS_quant = ((dS * dP_scale).to(torch_itype)).float()

        # dS (FP8) @ K (FP8) -> dQ (FP32); dS^T (FP8) @ Q (FP8) -> dK (FP32)
        dQ = dQ + _pv(dS_quant, k[:, :, start:end, :], h_k) * k_descale * dP_descale
        dK[:, :, start:end, :] = torch.einsum("bhgqk,bhgqd->bhkd", _grouped(dS_quant, h_k), q_g_k) * q_descale * dP_descale

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

    return dQ, dK, dV, dSink_token, dP_amax, dQ_amax, dK_amax, dV_amax
