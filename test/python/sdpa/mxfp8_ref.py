# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from .fp16_ref import _ScoreMask, _score_blocks, _pv, _grouped, _init_softmax_state

# fmt: off

# Blocked over KV like fp16_ref (see the note there): the (b, h, s_q, s_kv)
# score/probability matrices are never materialized.


def _dequant(x_fp8, sf_ref):
    # sf_*_ref are per-element fp32 dequant scales in [b*h, s, d] layout.
    return (x_fp8.float() * sf_ref.view(x_fp8.shape)).nan_to_num()


def compute_ref(q_fp8, k_fp8, v_fp8, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, torch_itype=torch.float8_e4m3fn, output_type=torch.bfloat16,
                left_bound=None, right_bound=None, diag_align=None, sink_token=None, rescale_threshold=4.0):
    """
    Compute reference SDPA with MXFP8 dequantization.
    Takes FP8 inputs and converts to FP32 to match cuDNN behavior.
    Supports GQA/MQA where K and V have fewer heads than Q.

    If sink_token is provided (shape: 1, h_q, 1, 1), it's used as a virtual attention
    score that competes with Q*K scores in softmax but contributes zero to the output
    (no V for sink). This is implemented via online softmax initialization:
    m_old = sink_token, l_old = 1.
    """
    b, h_q, s_q, d_qk = q_fp8.shape
    _, h_k, s_kv, _ = k_fp8.shape
    _, h_v, _, d_vo = v_fp8.shape
    device = q_fp8.device

    # Dequantize Q and K (scale factors apply to d_qk dimension), V (scale factors apply to s_kv dimension)
    q_dq = _dequant(q_fp8, sf_q_ref)
    k_dq = _dequant(k_fp8, sf_k_ref)
    v_dq = _dequant(v_fp8, sf_v_ref)

    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=None, block_mask=None, is_alibi=False, padding=None,
                      diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)

    m_old, l_old = _init_softmax_state(b, h_q, s_q, sink_token.float().reshape(1, h_q, 1, 1) if sink_token is not None else None, device)
    o = torch.zeros((b, h_q, s_q, d_vo), dtype=torch.float32, device=device)
    NEG_INF = float('-inf')

    for start, end, s_block in _score_blocks(q_dq, k_dq, attn_scale, mask):
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
        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # P (FP32) -> P (FP8)
        s_scale = 16.0
        inv_s_scale = 1.0 / 16.0
        p_block_quant = (p_block * s_scale).to(torch_itype).float().nan_to_num()
        p_block_quant = p_block_quant * inv_s_scale

        o = o + _pv(p_block_quant, v_dq[:, :, start:end, :], h_v)
        m_old = m_new
        l_old = l_new

    o = o / l_old.clamp(min=1.0)

    # O (FP32) -> O (output)
    o_ref = o.to(output_type).float()
    stats_ref = (m_old + torch.log(l_old)).float()

    return o_ref, stats_ref


def compute_ref_backward(q_fp8, q_t_fp8, k_fp8, k_t_fp8, v_fp8, o_f16, dO_f16, dO_fp8, dO_t_fp8, attn_scale,
                         sf_q_ref, sf_q_t_ref, sf_k_ref, sf_k_t_ref, sf_v_ref, sf_dO_ref, sf_dO_t_ref,
                         torch_itype=torch.float8_e4m3fn, torch_otype=torch.bfloat16,
                         left_bound=None, right_bound=None, diag_align=None, sink_token=None,
                         stats=None):
    """
    Compute backward pass reference for MXFP8 SDPA.

    If sink_token is provided, the virtual sink is included in softmax normalization
    and dSink_token is computed: dS_sink = -p_sink * D (no attn_scale), then summed
    over batch and query dimensions.
    """
    b, h_q, s_q, d_qk = q_fp8.shape
    _, h_k, s_kv, _ = k_fp8.shape
    _, h_v, _, d_vo = v_fp8.shape
    device = q_fp8.device

    # Dequantize for BMM1 (Q @ K^T): D-dimension scale factors
    q_dq = _dequant(q_fp8, sf_q_ref)
    k_dq = _dequant(k_fp8, sf_k_ref)
    # Dequantize for dO @ V^T: D-scale for dO, S-scale for V
    dO_dq = _dequant(dO_fp8, sf_dO_ref)
    v_dq = _dequant(v_fp8, sf_v_ref)
    # Dequantize for P^T @ dO_T -> dV: S-scale for dO_T
    dO_t_dq = _dequant(dO_t_fp8, sf_dO_t_ref)
    # Dequantize for dS @ K_T -> dQ: S-scale for K_T
    k_t_dq = _dequant(k_t_fp8, sf_k_t_ref)
    # Dequantize for dS^T @ Q_T -> dK: S-scale for Q_T
    q_t_dq = _dequant(q_t_fp8, sf_q_t_ref)

    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=None, block_mask=None, is_alibi=False, padding=None,
                      diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)
    sink = sink_token.float().reshape(1, h_q, 1, 1).expand(b, h_q, s_q, 1) if sink_token is not None else None

    # Unscaled scores are kept: the kernel folds attn_scale * log2(e) into ONE
    # multiplier and evaluates P in the log2 domain (see below).
    if stats is not None:
        lse = stats.float().reshape(b, h_q, s_q, 1)
    else:
        m_old, l_old = _init_softmax_state(b, h_q, s_q, sink, device)
        for _, _, s_block in _score_blocks(q_dq, k_dq, attn_scale, mask):
            m_new = torch.maximum(m_old, s_block.max(dim=-1, keepdim=True).values)
            l_old = l_old * torch.exp(m_old - m_new).nan_to_num() + torch.exp(s_block - m_new).nan_to_num().sum(dim=-1, keepdim=True)
            m_old = m_new
        lse = m_old + torch.log(l_old)

    # Use BF16 inputs for D
    D = (o_f16.float() * dO_f16.float()).reshape(b, h_q, s_q, d_vo).sum(dim=-1, keepdim=True)

    from .mxfp8 import quantize_to_mxfp8
    _log2e = math.log2(math.e)

    dQ = torch.zeros((b, h_q, s_q, d_qk), dtype=torch.float32, device=device)
    dK = torch.zeros((b, h_k, s_kv, d_qk), dtype=torch.float32, device=device)
    dV = torch.zeros((b, h_v, s_kv, d_vo), dtype=torch.float32, device=device)
    dO_g_v = _grouped(dO_dq, h_v)
    dO_t_g_v = _grouped(dO_t_dq, h_v)
    q_t_g_k = _grouped(q_t_dq, h_k)

    for start, end, s_raw in _score_blocks(q_dq, k_dq, None, mask):
        n = end - start
        # The backward kernel does not renormalize: it recomputes P from the
        # forward's log-sum-exp, which already accounts for the sink. It does so in
        # the log2 domain, P = 2^(S_raw * (attn_scale * log2 e) - stats * log2 e),
        # and this reference follows that arithmetic on purpose: P is then rounded
        # to E4M3 (3 mantissa bits) for the dV MMA on both sides, and a P computed
        # as exp(S - stats) lands on the other side of an E4M3 rounding boundary
        # often enough that a "sink-like" key (hundreds of query rows attending it
        # with P near 1) drifts ~0.3 in dV while every other row agrees to 1e-3
        # (test_sdpa_mxfp8_bwd_L0 at s=2404 with a 1184 sliding window). With the
        # kernel's formulation the two round alike and the tight tolerance holds.
        # torch.pow rather than torch.exp2: exp2 on a tensor this size raises a CUDA
        # "invalid argument" in some torch nightlies.
        if stats is not None:
            p = torch.pow(2.0, s_raw * (attn_scale * _log2e) - lse * _log2e).nan_to_num().float()
        else:
            p = torch.exp(s_raw * attn_scale - lse).nan_to_num().float()

        p_fp8 = (p * 256.0).to(torch_itype).float() * (1.0 / 256.0)

        # dO @ V -> dP
        dP = torch.einsum("bhgqd,bhkd->bhgqk", dO_g_v, v_dq[:, :, start:end, :]).reshape(b, h_q, s_q, n)

        # dS = P * (dP - D) * attn_scale
        dS = p * (dP - D) * attn_scale

        # MXFP8 scales are per 32 elements along either axis, so quantizing this
        # 128-wide KV block gives exactly the full-matrix quantization.
        dS_fp8, sf_dS_ref, _, dS_fp8_t, sf_dS_t_ref, _ = quantize_to_mxfp8(
            dS, b, h_q, s_q, n, block_size=32, fp8_dtype=torch_itype
        )
        # D-quantized dS (along s_kv) and S-quantized dS (along s_q)
        dS_fp32 = _dequant(dS_fp8, sf_dS_ref)
        dS_fp32_t = _dequant(dS_fp8_t, sf_dS_t_ref)

        # P @ dO -> dV; dS @ K -> dQ; dS^T @ Q -> dK (GQA reduction folded into the einsum)
        dV[:, :, start:end, :] = torch.einsum("bhgqk,bhgqd->bhkd", _grouped(p_fp8, h_v), dO_t_g_v)
        dQ = dQ + _pv(dS_fp32, k_t_dq[:, :, start:end, :], h_k)
        dK[:, :, start:end, :] = torch.einsum("bhgqk,bhgqd->bhkd", _grouped(dS_fp32_t, h_k), q_t_g_k)

    # Compute dSink_token if sink_token was provided
    # Formula: dSink = -exp(sink - logsumexp) * D summed over batch and sequence
    # Note: attn_scale is NOT applied because sink_token is added directly to scores,
    # not multiplied by attn_scale like Q @ K.T
    dSink_token = None
    if sink is not None:
        p_sink = torch.exp(sink - lse).nan_to_num().float()
        dSink_token = (-p_sink * D).sum(dim=(0, 2), keepdim=True)  # (1, h_q, 1, 1)

    dQ = dQ.to(torch_otype).float()
    dK = dK.to(torch_otype).float()
    dV = dV.to(torch_otype).float()

    return dQ, dK, dV, dSink_token
