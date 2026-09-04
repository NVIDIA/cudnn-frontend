# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import math
import cudnn

# fmt: off

# Both references are blocked over the KV dimension (flash-attention style) so
# peak memory is O(b * h * s_q * BLOCK) instead of O(b * h * s_q * s_kv): the
# largest test_mhas_v2 configs (b=8, h=8, 4096x4096) would otherwise hold several
# 4 GiB fp32 score matrices at once and OOM sibling xdist workers.
BLOCK = 128


def _alibi_slopes(h_q, device):
    n = 2 ** math.floor(math.log2(h_q))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < h_q:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (h_q - n), 2))
        m = torch.cat([m, m_hat])
    return m.view(1, -1, 1, 1).to(device=device, dtype=torch.float32)


class _ScoreMask:
    """Per-KV-block additive bias and boolean mask, built from index arithmetic
    so no (s_q, s_kv) tensor is ever materialized."""

    def __init__(self, b, h_q, s_q, s_kv, *, bias, block_mask, is_alibi, padding,
                 diag_align, left_bound, right_bound, device):
        self.bias = bias
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.rows = torch.arange(s_q, device=device).view(1, 1, s_q, 1)
        self.slopes = _alibi_slopes(h_q, device) if is_alibi else None

        self.seq_kv = None
        self.q_row_mask = None
        if padding is not None:
            seq_len_q, seq_len_kv = padding
            seq_q = torch.as_tensor(seq_len_q, device=device).view(b, 1, 1, 1)
            self.seq_kv = torch.as_tensor(seq_len_kv, device=device).view(b, 1, 1, 1)
            self.q_row_mask = self.rows >= seq_q

        # Diagonal offset: TOP_LEFT aligns row 0 with col 0; BOTTOM_RIGHT aligns
        # the last (valid) query with the last (valid) key, per batch when padded.
        if diag_align == cudnn.diagonal_alignment.BOTTOM_RIGHT:
            self.diag_offset = (self.seq_kv - seq_q) if padding else (s_kv - s_q)
        else:
            self.diag_offset = 0

        self.block_bits = None
        if block_mask is not None:
            bm = block_mask.to(dtype=torch.uint8, device=device)
            bits = (bm[..., None] & (1 << torch.arange(8, device=device, dtype=torch.uint8))) != 0
            self.block_bits = bits.reshape(bm.shape[0], bm.shape[1], bm.shape[2], bm.shape[3] * 8)
            self.q_blk = (torch.arange(s_q, device=device) // BLOCK).view(-1, 1)

    def __call__(self, start, end):
        cols = torch.arange(start, end, device=self.rows.device).view(1, 1, 1, end - start)
        rel = cols - self.rows

        add = None
        if self.bias is not None:
            add = self.bias[:, :, :, start:end]
        if self.slopes is not None:
            alibi = rel.to(torch.float32) * self.slopes
            add = alibi if add is None else add + alibi

        masked = None
        def _or(m, new):
            return new if m is None else (m | new)
        if self.seq_kv is not None:
            masked = _or(masked, cols >= self.seq_kv)
        if self.right_bound is not None:
            masked = _or(masked, rel > self.diag_offset + self.right_bound)
        if self.left_bound is not None:
            masked = _or(masked, rel <= self.diag_offset - self.left_bound)
        if self.block_bits is not None:
            k_blk = (cols.view(-1) // BLOCK).view(1, -1)
            masked = _or(masked, ~self.block_bits[:, :, self.q_blk, k_blk])
        if self.q_row_mask is not None:
            masked = _or(masked, self.q_row_mask)
        return add, masked


def _grouped(x, h_kv):
    b, h_q, s, d = x.shape
    assert h_q % h_kv == 0
    return x.view(b, h_kv, h_q // h_kv, s, d)


def _qk(q, k_block, h_k):
    # q: (b, h_q, s_q, d), k_block: (b, h_k, n, d) -> (b, h_q, s_q, n) without expanding K.
    b, h_q, s_q, _ = q.shape
    s = torch.einsum("bhgqd,bhkd->bhgqk", _grouped(q, h_k), k_block)
    return s.reshape(b, h_q, s_q, -1)


def _pv(p, v_block, h_v):
    b, h_q, s_q, _ = p.shape
    o = torch.einsum("bhgqk,bhkd->bhgqd", _grouped(p, h_v), v_block)
    return o.reshape(b, h_q, s_q, -1)


def _score_blocks(q, k, attn_scale, mask):
    s_kv = k.shape[2]
    for start in range(0, s_kv, BLOCK):
        end = min(start + BLOCK, s_kv)
        s = _qk(q, k[:, :, start:end, :], k.shape[1])
        if attn_scale is not None:
            s = s * attn_scale
        add, masked = mask(start, end)
        if add is not None:
            s = s + add
        if masked is not None:
            s = s.masked_fill(masked, float("-inf"))
        yield start, end, s


def _prepare(q, k, v, padding, device):
    q = q.to(dtype=torch.float32, device=device)
    k = k.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)
    if padding is not None:
        b = q.shape[0]
        seq_len_q, seq_len_kv = padding
        rows_q = torch.arange(q.shape[2], device=device).view(1, 1, -1, 1)
        rows_kv = torch.arange(k.shape[2], device=device).view(1, 1, -1, 1)
        seq_q = torch.as_tensor(seq_len_q, device=device).view(b, 1, 1, 1)
        seq_kv = torch.as_tensor(seq_len_kv, device=device).view(b, 1, 1, 1)
        q = q.masked_fill(rows_q >= seq_q, 0.0)
        k = k.masked_fill(rows_kv >= seq_kv, 0.0)
        v = v.masked_fill(rows_kv >= seq_kv, 0.0)
    return q, k, v


def _init_softmax_state(b, h_q, s_q, sink_token, device):
    if sink_token is not None:
        m = sink_token.to(dtype=torch.float32, device=device).expand(b, h_q, s_q, 1).clone()
        l = torch.ones((b, h_q, s_q, 1), dtype=torch.float32, device=device)
    else:
        m = torch.full((b, h_q, s_q, 1), float("-inf"), dtype=torch.float32, device=device)
        l = torch.zeros((b, h_q, s_q, 1), dtype=torch.float32, device=device)
    return m, l


def compute_ref(
    q,
    k,
    v,
    attn_scale=None,
    bias=None,
    block_mask=None,
    is_alibi=False,
    padding=None,
    diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=None,
    right_bound=None,
    dropout_prob=0.0,
    dropout_mask=None,
    sink_token=None,
    torch_type=torch.float16,
    device="cuda",
):
    b, h_q, s_q, d_qk = q.shape
    _, h_k, s_kv, _ = k.shape
    _, h_v, _, d_v = v.shape

    assert k.shape == (b, h_k, s_kv, d_qk)
    assert v.shape == (b, h_v, s_kv, d_v)

    q, k, v = _prepare(q, k, v, padding, device)
    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=bias, block_mask=block_mask, is_alibi=is_alibi, padding=padding,
                      diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)

    m_old, l_old = _init_softmax_state(b, h_q, s_q, sink_token, device)
    o = torch.zeros((b, h_q, s_q, d_v), dtype=torch.float32, device=device)

    for start, end, s_block in _score_blocks(q, k, attn_scale, mask):
        m_block = s_block.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_old, m_block)

        correction = torch.exp(m_old - m_new).nan_to_num()
        o = o * correction
        l_old = l_old * correction

        p_block = torch.exp(s_block - m_new).nan_to_num().to(torch_type).float()
        if mask.q_row_mask is not None:
            p_block = p_block.masked_fill(mask.q_row_mask, 0.0)

        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # apply dropout mask over softmax outputs
        if dropout_prob != 0.0:
            assert dropout_mask is not None, "PyTorch reference must have dropout_mask for dropout"
            p_block = (p_block * dropout_mask[:, :, :, start:end]) / (1 - dropout_prob)

        o = o + _pv(p_block, v[:, :, start:end, :], h_v)
        m_old = m_new
        l_old = l_new

    o_ref = o / l_old.clamp(min=1.0)
    o_ref = o_ref.to(torch_type).float()

    score_max_ref = m_old
    score_sum_exp_ref = l_old
    stats_ref = torch.log(score_sum_exp_ref) + score_max_ref

    return o_ref, stats_ref, score_max_ref, score_sum_exp_ref


def compute_ref_backward(
    q,
    k,
    v,
    o,
    dO,
    attn_scale=None,
    bias=None,
    is_alibi=False,
    padding=None,
    diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=None,
    right_bound=None,
    dropout_prob=0.0,
    dropout_mask=None,
    sink_token=None,
    torch_type=torch.float16,
    device="cuda",
):
    b, h_q, s_q, d_qk = q.shape
    _, h_k, s_kv, _ = k.shape
    _, h_v, _, d_v = v.shape

    q, k, v = _prepare(q, k, v, padding, device)
    o = o.to(dtype=torch.float32, device=device)
    dO = dO.to(dtype=torch.float32, device=device)
    mask = _ScoreMask(b, h_q, s_q, s_kv, bias=bias, block_mask=None, is_alibi=is_alibi, padding=padding,
                      diag_align=diag_align, left_bound=left_bound, right_bound=right_bound, device=device)

    if dropout_prob != 0.0:
        assert dropout_mask is not None, "PyTorch reference must have dropout_mask for dropout"

    # Pass 1: row log-sum-exp (includes the sink as a virtual first column).
    m_old, l_old = _init_softmax_state(b, h_q, s_q, sink_token, device)
    for _, _, s_block in _score_blocks(q, k, attn_scale, mask):
        m_new = torch.maximum(m_old, s_block.max(dim=-1, keepdim=True).values)
        l_old = l_old * torch.exp(m_old - m_new).nan_to_num() + torch.exp(s_block - m_new).nan_to_num().sum(dim=-1, keepdim=True)
        m_old = m_new
    lse = m_old + torch.log(l_old)

    # D = sum(o * dO, dim=-1)
    D = (o * dO).sum(dim=-1, keepdim=True)

    dQ = torch.zeros((b, h_q, s_q, d_qk), dtype=torch.float32, device=device)
    dK = torch.zeros((b, h_k, s_kv, d_qk), dtype=torch.float32, device=device)
    dV = torch.zeros((b, h_v, s_kv, d_v), dtype=torch.float32, device=device)
    dBias = torch.zeros((1, h_q, s_q, s_kv), dtype=torch.float32, device=device) if bias is not None else None

    q_g_k = _grouped(q, h_k)
    dO_g_v = _grouped(dO, h_v)

    # Pass 2: P = exp(S - lse) per block; all-masked rows give exp(-inf - -inf) -> nan -> 0.
    for start, end, s_block in _score_blocks(q, k, attn_scale, mask):
        p = torch.exp(s_block - lse).nan_to_num()

        # dP = dO @ V^T, then apply dropout mask
        dP = torch.einsum("bhgqd,bhkd->bhgqk", dO_g_v, v[:, :, start:end, :]).reshape(b, h_q, s_q, end - start)
        if dropout_prob != 0.0:
            drop = dropout_mask[:, :, :, start:end]
            dP = (dP * drop) / (1 - dropout_prob)
            p_dropped = (p * drop) / (1 - dropout_prob)
        else:
            p_dropped = p
        p_dropped = p_dropped.to(torch_type).float()

        # dS = P * (dP - D) * attn_scale
        dS_raw = p * (dP - D)
        if dBias is not None:
            # dBias = dS / attn_scale (undo the scale baked into dS), summed over batch
            dBias[:, :, :, start:end] = dS_raw.sum(dim=0, keepdim=True)
        dS = dS_raw * attn_scale if attn_scale is not None else dS_raw
        dS = dS.to(torch_type).float()

        # dQ += dS @ K_block
        dQ = dQ + _pv(dS, k[:, :, start:end, :], h_k)
        # dK_block = dS^T @ Q, dV_block = P_dropped^T @ dO (both reduce over the GQA group directly)
        dK[:, :, start:end, :] = torch.einsum("bhgqk,bhgqd->bhkd", _grouped(dS, h_k), q_g_k)
        dV[:, :, start:end, :] = torch.einsum("bhgqk,bhgqd->bhkd", _grouped(p_dropped, h_v), dO_g_v)

    # dSink_token: gradient for sink token
    dSink_token = None
    if sink_token is not None:
        sink = sink_token.to(dtype=torch.float32, device=device).expand(b, h_q, s_q, 1)
        p_sink = torch.exp(sink - lse).nan_to_num()
        if mask.q_row_mask is not None:
            p_sink = p_sink.masked_fill(mask.q_row_mask, 0.0)
        dSink_token = (-p_sink * D).sum(dim=(0, 2), keepdim=True)

    dQ = dQ.to(torch_type).float()
    dK = dK.to(torch_type).float()
    dV = dV.to(torch_type).float()
    if dBias is not None:
        dBias = dBias.to(torch_type).float()

    return dQ, dK, dV, dBias, dSink_token
