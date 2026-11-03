# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch references for pooled DSA training.

Layouts follow the DSA module: ``index_q`` (B, S, H, D), ``weights`` (B, S, H),
``keys``/``gates`` (B, S_k, D), ``ape`` (P, D), pool ids (B, S, K) INT32 with
``-1`` padding, token ids (B, S, K*P + P - 1).
"""

import math
from typing import Optional, Tuple

import torch

from deepseek_sparse_attention.cutedsl.dsa_reference import (
    _indexer_predict_distribution,
    ref_sparse_attn_score_recompute,
)

CLIP_LOG_MIN = -100.0


def pool_keys(keys: torch.Tensor, gates: torch.Tensor, ape: torch.Tensor, pool_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax(gates + ape) weighted sum over each complete pool, FP32.

    Returns ``(pooled (B, P_k, D), alpha (B, P_k, pool_size, D))``.
    """
    b, s_k, d = keys.shape
    n_pools = s_k // pool_size
    use = n_pools * pool_size
    k = keys[:, :use].float().view(b, n_pools, pool_size, d)
    logits = gates[:, :use].float().view(b, n_pools, pool_size, d) + ape.float().view(1, 1, pool_size, d)
    alpha = torch.softmax(logits, dim=2)
    return (alpha * k).sum(dim=2), alpha


def round_bf16_ste(x: torch.Tensor) -> torch.Tensor:
    """bf16 rounding in the forward, identity in the backward."""
    return x + (x.to(torch.bfloat16).float() - x).detach()


def pooled_dense_scores(index_q: torch.Tensor, weights: torch.Tensor, pooled: torch.Tensor, sm_scale: float, chunk: int = 1024) -> torch.Tensor:
    """sum_h relu(q_h . k_j) * (w_h * sm_scale) over every pool, FP32 (B, S, P_k)."""
    b, s, h, d = index_q.shape
    out = torch.empty(b, s, pooled.shape[1], dtype=torch.float32, device=index_q.device)
    k_f = pooled.float()
    w_f = weights.float() * sm_scale
    for start in range(0, s, chunk):
        end = min(start + chunk, s)
        sc = torch.einsum("bqhd,bjd->bqhj", index_q[:, start:end].float(), k_f).relu_()
        out[:, start:end] = (sc * w_f[:, start:end].unsqueeze(-1)).sum(dim=2)
    return out


def eligible_pools(positions: torch.Tensor, n_pools: int, pool_size: int) -> torch.Tensor:
    """(B, S, P_k) bool: pool j is complete and causal when j < (pos + 1) // pool_size."""
    completed = (positions + 1) // pool_size
    j = torch.arange(n_pools, device=positions.device).view(1, 1, -1)
    return j < completed.unsqueeze(-1)


def select_pools(scores: torch.Tensor, eligible: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k eligible pools per row, INT32, ``-1`` for the missing ones (always last)."""
    masked = scores.masked_fill(~eligible, float("-inf"))
    vals, ids = masked.topk(k, dim=-1)
    return ids.masked_fill(vals == float("-inf"), -1).to(torch.int32)


def expand_pools_compact(pool_ids: torch.Tensor, positions: torch.Tensor, pool_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Training expansion: [valid pools' tokens][tail tokens][-1 ...].

    Returns ``(token_ids (B, S, K*P + P - 1) INT32, topk_length (B, S) INT32)``.
    Slot group ``k`` (slots ``k*P .. k*P+P-1``) holds pool ``pool_ids[k]`` for
    every valid pool; the incomplete tail follows the last valid pool.
    """
    b, s, k = pool_ids.shape
    p = pool_size
    device = pool_ids.device
    length = positions + 1
    completed = length // p
    tail = length - completed * p
    valid = pool_ids >= 0
    n_valid = valid.sum(dim=-1)
    width = k * p + p - 1
    hist = (pool_ids.clamp(min=0).unsqueeze(-1) * p + torch.arange(p, device=device)).reshape(b, s, k * p)
    hist = hist.masked_fill(~valid.repeat_interleave(p, dim=-1), -1)
    out = torch.full((b, s, width), -1, dtype=torch.int64, device=device)
    out[..., : k * p] = hist
    slot = torch.arange(width, device=device).view(1, 1, width)
    tail_start = (n_valid * p).unsqueeze(-1)
    in_tail = (slot >= tail_start) & (slot < tail_start + tail.unsqueeze(-1))
    tail_tok = (completed * p).unsqueeze(-1) + (slot - tail_start)
    out = torch.where(in_tail, tail_tok, out)
    return out.to(torch.int32), (n_valid * p + tail).to(torch.int32)


def expand_pools_inference(pool_ids: torch.Tensor, positions: torch.Tensor, pool_size: int) -> torch.Tensor:
    """PR #1847 ``pooled_expand_forward`` semantics (short rows ascending)."""
    b, s, k = pool_ids.shape
    p = pool_size
    device = pool_ids.device
    length = positions + 1
    completed = length // p
    tail = length - completed * p
    width = k * p + p - 1
    valid = (pool_ids >= 0) & (pool_ids < completed.unsqueeze(-1))
    hist = (pool_ids.clamp(min=0).unsqueeze(-1) * p + torch.arange(p, device=device)).reshape(b, s, k * p)
    hist = hist.masked_fill(~valid.repeat_interleave(p, dim=-1), -1)
    slot = torch.arange(width, device=device).view(1, 1, width)
    short = (length <= k * p).unsqueeze(-1)
    ascending = torch.where(slot < length.unsqueeze(-1), slot, torch.full_like(slot, -1)).expand(b, s, width)
    out = torch.full((b, s, width), -1, dtype=torch.int64, device=device)
    out[..., : k * p] = hist
    tail_tok = (completed * p).unsqueeze(-1) + (slot - k * p)
    out = torch.where((slot >= k * p) & (slot - k * p < tail.unsqueeze(-1)), tail_tok, out)
    return torch.where(short, ascending, out).to(torch.int32)


def argsort_compact_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Argsort compaction of inference-style index rows into valid-first rows with lengths."""
    valid = indices != -1
    topk_length = valid.sum(dim=-1, dtype=torch.int32).contiguous()
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)
    topk_idxs = indices.gather(-1, order).clamp_min(0).to(torch.int32).contiguous()
    return topk_idxs, topk_length


def pool_target(token_target: torch.Tensor, pool_ids: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Sum the token teacher mass of each selected pool and renormalize (tail dropped)."""
    b, s, k = pool_ids.shape
    grouped = token_target[..., : k * pool_size].reshape(b, s, k, pool_size).sum(dim=-1)
    grouped = grouped.masked_fill(pool_ids < 0, 0.0)
    return grouped / grouped.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def pooled_indexer_loss(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    pool_ids: torch.Tensor,
    target: torch.Tensor,
    sm_scale: float,
    loss_coeff: float,
    pool_size: int,
) -> torch.Tensor:
    """loss_coeff * mean_rows KL(target || softmax(selected pooled scores)); differentiable."""
    pooled, _ = pool_keys(keys, gates, ape, pool_size)
    pooled = round_bf16_ste(pooled)
    predict, scores = _indexer_predict_distribution(index_q.float(), pooled, weights.float(), pool_ids, sm_scale, return_scores=True)
    valid = pool_ids >= 0
    log_predict = torch.log_softmax(scores, dim=-1).masked_fill(~valid, 0.0)
    t = target.masked_fill(~valid, 0.0)
    kl = (t * (torch.log(t.clamp(min=math.exp(CLIP_LOG_MIN))) - log_predict)).sum(dim=-1)
    return loss_coeff * kl.mean()


def oracle_pooled_indexer_backward(index_q, weights, keys, gates, ape, pool_ids, target, sm_scale, loss_coeff, pool_size, grad_loss=1.0):
    """Autograd gradients for (index_q, weights, keys, gates, ape), FP32."""
    leaves = [t.detach().float().requires_grad_(True) for t in (index_q, weights, keys, gates, ape)]
    loss = pooled_indexer_loss(*leaves, pool_ids, target, sm_scale, loss_coeff, pool_size)
    return torch.autograd.grad(loss, leaves, grad_outputs=torch.full_like(loss, grad_loss))


def naive_pooled_indexer_backward(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    token_ids: torch.Tensor,
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    sm_scale: float,
    softmax_scale: float,
    loss_coeff: float,
    grad_loss: float,
    pool_size: int,
    chunk: int = 512,
):
    """Dense naive reference backward on pools.

    Dense pooled scores over every pool, the teacher from a full attention
    recompute masked to the selected tokens, then chunked einsum gradients and
    autograd through the pooling.
    """
    b, s, h, d = index_q.shape
    s_k = keys.shape[1]
    n_pools = s_k // pool_size
    k_elig = eligible_pools(positions, n_pools, pool_size)
    with torch.enable_grad():
        keys_l, gates_l, ape_l = (t.detach().float().requires_grad_(True) for t in (keys, gates, ape))
        pooled, _ = pool_keys(keys_l, gates_l, ape_l, pool_size)
        pooled_bf = round_bf16_ste(pooled)
    pooled_c = pooled_bf.detach()
    # dense pooled index scores (B, S, P_k)
    index_scores = pooled_dense_scores(index_q, weights, pooled_c, sm_scale).masked_fill(~k_elig, float("-inf"))
    # teacher: full attention recompute, masked to the selected tokens
    width = token_ids.shape[-1]
    token_mask = torch.zeros(b, s, s_k + 1, dtype=torch.bool, device=index_q.device)
    token_mask.scatter_(-1, token_ids.clamp(min=-1).long() + 1, True)
    token_mask = token_mask[..., 1:]
    logits = torch.einsum("bqhd,bkd->bhqk", q_attn.float(), k_attn.float()) * softmax_scale
    logits = logits.masked_fill(~token_mask.unsqueeze(1), float("-inf"))
    probs = torch.softmax(logits, dim=-1).sum(dim=1)  # (B, S, S_k)
    del logits
    # pool-level teacher (tail excluded), aligned to the dense pool axis
    pooled_mass = probs[..., : n_pools * pool_size].reshape(b, s, n_pools, pool_size).sum(dim=-1)
    sel_mask = torch.zeros(b, s, n_pools + 1, dtype=torch.bool, device=index_q.device)
    sel_mask.scatter_(-1, pool_ids.long() + 1, True)
    sel_mask = sel_mask[..., 1:]
    pooled_mass = pooled_mass.masked_fill(~sel_mask, 0.0)
    target = pooled_mass / pooled_mass.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    predict = torch.softmax(index_scores.masked_fill(~sel_mask, float("-inf")), dim=-1)
    grad_rows = grad_loss * loss_coeff / (b * s)
    grad_logits = (predict * target.sum(dim=-1, keepdim=True) - target) * grad_rows
    grad_logits = grad_logits.masked_fill(~sel_mask, 0.0)  # (B, S, P_k)
    # chunked backward through sum_h relu(q.k) * w
    k_f = pooled_c.float()
    w_f = weights.float() * sm_scale
    grad_q = torch.empty(b, s, h, d, dtype=torch.float32, device=index_q.device)
    grad_w = torch.empty(b, s, h, dtype=torch.float32, device=index_q.device)
    grad_k = torch.zeros(b, n_pools, d, dtype=torch.float32, device=index_q.device)
    for start in range(0, s, chunk):
        end = min(start + chunk, s)
        q_c = index_q[:, start:end].float()
        sc = torch.einsum("bqhd,bjd->bqhj", q_c, k_f)
        relu_mask = sc > 0
        sc.relu_()
        g = grad_logits[:, start:end].unsqueeze(2)  # (B, c, 1, P_k)
        grad_w[:, start:end] = (g * sc).sum(dim=-1) * sm_scale
        gs = (g * w_f[:, start:end].unsqueeze(-1)).masked_fill(~relu_mask, 0.0)
        grad_q[:, start:end] = torch.einsum("bqhj,bjd->bqhd", gs, k_f)
        grad_k += torch.einsum("bqhj,bqhd->bjd", gs, q_c)
    grad_keys, grad_gates, grad_ape = torch.autograd.grad(pooled_bf, (keys_l, gates_l, ape_l), grad_outputs=grad_k)
    return grad_q, grad_w, grad_keys, grad_gates, grad_ape


def make_case(batch: int, seqlen: int, device, *, index_heads=32, index_dim=128, attn_heads=64, attn_dim=512, pool_size=4, topk_tokens=2048, seed=0):
    """Random pooled DSA layer inputs (NoPE MLA latent)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rn = lambda *shape, dtype=torch.bfloat16, scale=1.0: (torch.randn(*shape, device=device, generator=gen) * scale).to(dtype)
    index_q = rn(batch, seqlen, index_heads, index_dim, scale=index_dim**-0.5)
    weights = rn(batch, seqlen, index_heads, scale=0.5)
    keys = rn(batch, seqlen, index_dim)
    gates = rn(batch, seqlen, index_dim, scale=0.5)
    ape = rn(pool_size, index_dim, dtype=torch.float32, scale=0.1)
    q_attn = rn(batch, seqlen, attn_heads, attn_dim, scale=attn_dim**-0.5)
    k_attn = rn(batch, seqlen, attn_dim)
    positions = torch.arange(seqlen, device=device).view(1, seqlen).expand(batch, seqlen).contiguous()
    return dict(
        index_q=index_q,
        weights=weights,
        keys=keys,
        gates=gates,
        ape=ape,
        q_attn=q_attn,
        k_attn=k_attn,
        positions=positions,
        pool_size=pool_size,
        k_pools=topk_tokens // pool_size,
    )
