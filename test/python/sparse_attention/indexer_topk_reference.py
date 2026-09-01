# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normative PyTorch reference for cudnn.sparse_attention.indexer_topk.

Executable form of the contract in
``python/cudnn/sparse_attention/indexer_topk/api.py`` and the oracle every
registered kernel is validated against. Row-looped for clarity, not speed.
"""

from typing import Optional, Tuple

import torch


def reference_indexer_topk(
    q_index: torch.Tensor,  # (T_q, H_i, D_i) THD
    k_index: torch.Tensor,  # (T_e, H_ik, D_i)
    top_k: int,
    *,
    weights: Optional[torch.Tensor] = None,  # (T_q, H_i) fp32
    activation: str = "relu",
    head_groups: int = 1,
    ratio: int = 1,
    score_pool: int = 1,
    force_first: int = 0,
    force_last: int = 0,
    cu_seqlens_q: torch.Tensor = None,  # (B+1,) int32
    cu_seqlens_k: torch.Tensor = None,  # (B+1,) int32, entry counts per sequence
    q_causal_offsets: Optional[torch.Tensor] = None,  # (B,) int32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(indices, topk_length, logits)`` per the normative contract.

    ``indices`` (T_q, [G,] top_k) int32: global pooled-entry ids, compact,
    ascending, -1 beyond ``topk_length``. ``logits`` slot-aligned fp32.
    Sequences must start ``ratio * score_pool``-aligned in the packed key
    stream (the aligned-packing rule).
    """
    t_q, h_i, d_i = q_index.shape
    h_ik = k_index.shape[1]
    G = head_groups
    hpg = h_i // G
    device = q_index.device
    b = cu_seqlens_q.shape[0] - 1

    out_shape = (t_q, G, top_k) if G > 1 else (t_q, top_k)
    indices = torch.full((t_q, G, top_k), -1, dtype=torch.int32, device=device)
    logits = torch.full((t_q, G, top_k), float("-inf"), dtype=torch.float32, device=device)
    lengths = torch.zeros(t_q, G, dtype=torch.int32, device=device)

    for bi in range(b):
        q_lo, q_hi = int(cu_seqlens_q[bi]), int(cu_seqlens_q[bi + 1])
        k_lo, k_hi = int(cu_seqlens_k[bi]), int(cu_seqlens_k[bi + 1])
        assert k_lo % (ratio * score_pool) == 0 or score_pool * ratio == 1, "sequences must be granularity-aligned in the packed key stream"
        n_entries = k_hi - k_lo
        offset = int(q_causal_offsets[bi]) if q_causal_offsets is not None else 0
        pool_base = k_lo // score_pool  # global pooled-entry id of this sequence's entry 0

        for t in range(q_lo, q_hi):
            p = offset + (t - q_lo)  # global token position
            n_valid = min((p + 1) // ratio, n_entries)  # fully-past entries
            n_pooled = n_valid // score_pool
            if n_pooled == 0:
                continue

            for g in range(G):
                kh = 0 if h_ik == 1 else g
                kk = k_index[k_lo : k_lo + n_valid, kh, :].float()  # (n_valid, D)
                qq = q_index[t, g * hpg : (g + 1) * hpg, :].float()  # (hpg, D)
                s = qq @ kk.t()  # (hpg, n_valid)
                if activation == "relu":
                    s = torch.relu(s)
                if weights is not None:
                    s = s * weights[t, g * hpg : (g + 1) * hpg].float().unsqueeze(-1)
                score = s.sum(0)  # (n_valid,)
                pooled = score[: n_pooled * score_pool].reshape(n_pooled, score_pool).amax(-1)  # (n_pooled,)

                k_eff = min(top_k, n_pooled)
                forced = set(range(min(force_first, n_pooled))) | set(range(max(0, n_pooled - force_last), n_pooled))
                forced = set(list(forced)[:k_eff]) if len(forced) > k_eff else forced
                budget = k_eff - len(forced)
                # exact top-k over the non-forced candidates, ties -> smallest id
                rest = [e for e in range(n_pooled) if e not in forced]
                rest.sort(key=lambda e: (-float(pooled[e]), e))
                chosen = sorted(forced | set(rest[:budget]))  # ascending ids
                ids = torch.tensor(chosen, dtype=torch.int32, device=device)
                indices[t, g, : len(chosen)] = ids + pool_base
                logits[t, g, : len(chosen)] = pooled[ids.long()]
                lengths[t, g] = len(chosen)

    if G == 1:
        return indices.reshape(t_q, top_k), lengths.reshape(t_q), logits.reshape(t_q, top_k)
    return indices, lengths, logits
