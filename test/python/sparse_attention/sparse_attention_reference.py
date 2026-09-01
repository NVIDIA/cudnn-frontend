# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normative PyTorch reference for cudnn.sparse_attention forward.

This is the executable form of the contract in
``python/cudnn/sparse_attention/fwd/api.py`` and the oracle every registered
kernel is validated against. It lives with the tests on purpose: the API
package must stay framework-neutral and kernel-only, while the oracle is
torch by construction.
"""

import math
from typing import Optional, Tuple

import torch


def reference_sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    index_granularity: int = 1,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contract semantics, reference speed. Returns ``(out, lse)``.

    Layouts follow the API: THD ``q (T_q, H_q, D_k)`` / ``k, v (T_kv, H_kv, D)``
    or BSHD with a leading batch dim. ``topk_idxs`` is ``(*lead, [G,] topk)``
    with storage-native ids, -1 padded; ``lse`` is KV-only, base-e, FP32;
    ``attn_sink`` joins the softmax denominator only; dead rows produce
    ``lse = -inf`` and ``out = 0``.
    """
    g = index_granularity
    is_thd = q.ndim == 3

    if is_thd:
        t_q, h_q, d_k = q.shape
        t_kv, h_kv, _ = k.shape
        d_v = v.shape[-1]
        q_flat, k_flat, v_flat = q, k, v
        idxs = topk_idxs.reshape(t_q, -1, topk_idxs.shape[-1]) if topk_idxs.ndim == 3 else topk_idxs.reshape(t_q, 1, -1)
        kv_bound = torch.full((t_q,), t_kv, dtype=torch.int64, device=q.device)
        kv_base = torch.zeros((t_q,), dtype=torch.int64, device=q.device)
    else:
        b, s_q, h_q, d_k = q.shape
        _, s_kv, h_kv, _ = k.shape
        d_v = v.shape[-1]
        t_q = b * s_q
        q_flat = q.reshape(t_q, h_q, d_k)
        k_flat = k.reshape(b * s_kv, h_kv, d_k)
        v_flat = v.reshape(b * s_kv, h_kv, d_v)
        idxs = topk_idxs.reshape(t_q, -1, topk_idxs.shape[-1])
        if topk_idxs.ndim == 3:  # (B, S_q, topk) -> G = 1
            idxs = idxs.reshape(t_q, 1, -1)
        batch_of_row = torch.arange(b, device=q.device, dtype=torch.int64).repeat_interleave(s_q)
        kv_bound = torch.full((t_q,), s_kv, dtype=torch.int64, device=q.device)
        kv_base = batch_of_row * s_kv

    n_groups = idxs.shape[1]
    topk_max = idxs.shape[-1]
    heads_per_kv = h_q // h_kv
    group_scope = n_groups

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_k)

    length = None
    if topk_length is not None:
        length = topk_length.reshape(t_q, n_groups) if group_scope != 1 else topk_length.reshape(t_q, 1)

    idxs = idxs.to(torch.int64)
    slot = torch.arange(topk_max, device=q.device)
    valid = idxs >= 0
    if length is not None:
        valid = valid & (slot.view(1, 1, -1) < length.unsqueeze(-1))

    token_ids = idxs.unsqueeze(-1) * g + torch.arange(g, device=q.device).view(1, 1, 1, g)
    token_valid = valid.unsqueeze(-1) & (token_ids < kv_bound.view(-1, 1, 1, 1))
    token_ids = token_ids.reshape(t_q, n_groups, topk_max * g)
    token_valid = token_valid.reshape(t_q, n_groups, topk_max * g)
    gather_ids = (token_ids.clamp(min=0) + kv_base.view(-1, 1, 1)).clamp(max=k_flat.shape[0] - 1)

    out_t = torch.zeros(t_q, h_q, d_v, dtype=q.dtype, device=q.device)
    lse_t = torch.full((t_q, h_q), float("-inf"), dtype=torch.float32, device=q.device)

    for h in range(h_q):
        kv_head = h // heads_per_kv
        if group_scope == 1:
            grp = 0
        elif group_scope == h_kv:
            grp = kv_head
        else:  # group_scope == h_q
            grp = h
        ids_h = gather_ids[:, grp, :]
        valid_h = token_valid[:, grp, :]
        kk = k_flat[:, kv_head, :].float()[ids_h]
        vv = v_flat[:, kv_head, :].float()[ids_h]
        s = torch.einsum("td,tkd->tk", q_flat[:, h, :].float(), kk) * softmax_scale
        s = torch.where(valid_h, s, torch.full_like(s, float("-inf")))

        row_lse = torch.logsumexp(s, dim=-1)
        denom_lse = row_lse
        if attn_sink is not None:
            denom_lse = torch.logaddexp(row_lse, attn_sink[h].float().expand_as(row_lse))
        p = torch.exp(s - denom_lse.unsqueeze(-1))
        p = torch.where(valid_h, p, torch.zeros_like(p))
        out_t[:, h, :] = torch.einsum("tk,tkd->td", p, vv).to(q.dtype)
        lse_t[:, h] = row_lse

    if not is_thd:
        out_t = out_t.reshape(b, s_q, h_q, d_v)
        lse_t = lse_t.reshape(b, s_q, h_q)
    return out_t, lse_t
