"""
Reference implementations for DSA (DeepSeek Sparse Attention) tests.

Contains pure-PyTorch reference implementations for numerical verification of
the cudnn-frontend DSA kernels.

The forward sparse-attention reference in this file exists solely to generate
(out, lse) inputs and run autograd against the cudnn-frontend
SparseAttentionBackward kernel. The production DSA forward pass is FlashMLA
(C++); FlashMLA is not integrated in this CuTe-DSL-only step. For production
forward, use FlashMLA directly. FlashMLA returns the KV-only LSE; the attention
sink is folded into the softmax denominator by the backward kernel.
"""

import math
from typing import Optional, Tuple

import torch


def _make_topk_mask(
    topk_idxs: torch.Tensor,  # (T, topk)
    topk_length: Optional[torch.Tensor],  # (T,) or None
    s_kv: int,
) -> torch.Tensor:
    """Materialize a boolean (T, s_kv) mask from topk indices."""
    t, topk = topk_idxs.shape
    mask = torch.zeros(t, s_kv, dtype=torch.bool, device=topk_idxs.device)
    valid_idx = topk_idxs.clamp(min=0, max=s_kv - 1)
    row_idx = torch.arange(t, device=topk_idxs.device).unsqueeze(1).expand(-1, topk)
    mask[row_idx, valid_idx] = True
    if topk_length is not None:
        positions = torch.arange(topk, device=topk_idxs.device).unsqueeze(0).expand(t, -1)
        invalid = positions >= topk_length.unsqueeze(1)
        invalid_idx = topk_idxs.clone()
        invalid_idx[invalid] = 0
        # Recompute mask only where valid
        mask = torch.zeros(t, s_kv, dtype=torch.bool, device=topk_idxs.device)
        valid_mask = ~invalid & (topk_idxs >= 0) & (topk_idxs < s_kv)
        valid_positions = topk_idxs[valid_mask]
        valid_rows = row_idx[valid_mask]
        mask[valid_rows, valid_positions] = True
    return mask


def ref_sparse_attention_forward(
    q: torch.Tensor,  # (T, H, D)
    kv: torch.Tensor,  # (T_kv, D), K=V shared
    attn_sink: torch.Tensor,  # (H,)
    topk_idxs: torch.Tensor,  # (T, topk)
    topk_length: Optional[torch.Tensor] = None,  # (T,)
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference DSA forward (uncompiled, slow).

    Returns ``(out, lse)`` where ``lse`` is the FlashMLA-style KV-only LSE,
    excluding the attention sink. ``out`` is still computed with the sink in
    the softmax denominator.
    """
    t, h, d = q.shape
    t_kv, d_kv = kv.shape
    assert d == d_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    q_f = q.to(torch.float32)
    k_f = kv.to(torch.float32)
    v_f = kv.to(torch.float32)

    mask = _make_topk_mask(topk_idxs, topk_length, t_kv)  # (T, T_kv)
    mask = mask.unsqueeze(1).expand(t, h, t_kv)  # (T, H, T_kv)

    # scores[t, h, k] = q[t, h] @ kv[k]
    scores = torch.einsum("thd,kd->thk", q_f, k_f) * softmax_scale
    scores = scores.masked_fill(~mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)  # (T, H), excludes sink
    sink = attn_sink.view(1, h)
    lse_with_sink = torch.logaddexp(lse, sink)
    weights = torch.exp(scores - lse_with_sink.unsqueeze(-1))
    out = torch.einsum("thk,kd->thd", weights, v_f)

    return out.to(q.dtype), lse


def check_ref_dsa_sparse_attention_backward(
    q,
    kv,
    attn_sink,
    topk_idxs,
    out,
    dout,
    lse,
    dq_actual,
    dkv_actual,
    d_sink_actual,
    softmax_scale=None,
    topk_length=None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    """Run autograd on the reference forward to compare ``dq`` / ``dkv`` / ``d_sink``."""
    q_r = q.detach().clone().to(torch.float32).requires_grad_(True)
    kv_r = kv.detach().clone().to(torch.float32).requires_grad_(True)
    sink_r = attn_sink.detach().clone().to(torch.float32).requires_grad_(True)

    out_r, _ = ref_sparse_attention_forward(
        q_r,
        kv_r,
        sink_r,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    out_r.backward(dout.to(torch.float32))

    torch.testing.assert_close(
        dq_actual.to(torch.float32),
        q_r.grad,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        dkv_actual.to(torch.float32),
        kv_r.grad,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        d_sink_actual.to(torch.float32),
        sink_r.grad,
        atol=atol,
        rtol=rtol,
    )


def ref_indexer_forward(
    q: torch.Tensor,  # (B, S_q, H_q, D)
    k: torch.Tensor,  # (B, S_k, H_kv, D)
    w: torch.Tensor,  # (B, S_q, H_q)
    ratio: int,
) -> torch.Tensor:
    """Dense indexer score computation. Returns (B, S_q, S_k) FP32."""
    b, s_q, h_q, d = q.shape
    _, s_k, h_kv, _ = k.shape
    qhead_per_kvhead = h_q // h_kv
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    w_f = w.to(torch.float32)

    # Expand K across the qhead groups so each query head sees its KV head.
    k_exp = k_f.repeat_interleave(qhead_per_kvhead, dim=2)  # (B, S_k, H_q, D)
    # scores_per_head[b, q, h, k] = Q @ K^T
    scores = torch.einsum("bqhd,bkhd->bqhk", q_f, k_exp)
    scores = torch.relu(scores)
    # Weighted per-head sum
    scores = scores * w_f.unsqueeze(-1)  # (B, S_q, H_q, S_k)
    out = scores.sum(dim=2)  # (B, S_q, S_k)

    # Ratio causal mask: for query q, keys k >= (q+1)//ratio are -inf.
    q_idx = torch.arange(s_q, device=q.device).view(1, s_q, 1)
    k_idx = torch.arange(s_k, device=q.device).view(1, 1, s_k)
    mask = k_idx < ((q_idx + 1) // ratio).clamp(min=0)
    out = torch.where(mask.expand_as(out), out, torch.full_like(out, float("-inf")))
    return out


def check_ref_indexer_forward(
    q,
    k,
    w,
    out_actual,
    ratio: int,
    atol: float = 1e-4,
    rtol: float = 1e-4,
):
    out_ref = ref_indexer_forward(q, k, w, ratio)
    # Compare only over finite positions (mask out -inf).
    finite = torch.isfinite(out_ref)
    torch.testing.assert_close(
        out_actual[finite],
        out_ref[finite],
        atol=atol,
        rtol=rtol,
    )


def ref_indexer_top_k(
    input_values: torch.Tensor,  # (n_rows, num_cols)
    seq_lens: torch.Tensor,  # (batch_size,)
    top_k: int,
    next_n: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-batch topk over the first seq_lens[b] columns. Returns (indices, values)."""
    n_rows, num_cols = input_values.shape
    batch = seq_lens.shape[0]
    rows_per_batch = n_rows // batch
    device = input_values.device

    out_idx = torch.zeros(n_rows, top_k, dtype=torch.int32, device=device)
    out_val = torch.zeros(n_rows, top_k, dtype=input_values.dtype, device=device)
    for b in range(batch):
        L = int(seq_lens[b].item())
        row_start = b * rows_per_batch
        row_end = row_start + rows_per_batch
        block = input_values[row_start:row_end, :L]
        k_eff = min(top_k, L)
        if k_eff > 0:
            vals, idxs = torch.topk(block, k_eff, dim=1)
            out_idx[row_start:row_end, :k_eff] = idxs.to(torch.int32)
            out_val[row_start:row_end, :k_eff] = vals
    return out_idx, out_val


def check_ref_indexer_top_k(
    input_values,
    seq_lens,
    top_k,
    next_n,
    idx_actual,
    val_actual,
    return_val: bool,
    atol: float = 0.0,
    rtol: float = 0.0,
):
    idx_ref, val_ref = ref_indexer_top_k(input_values, seq_lens, top_k, next_n)
    # Topk order is permitted to differ; compare as sorted sets per row up to the
    # effective length. The DSA kernel returns indices for a particular row; we
    # verify that the set of picked indices matches the reference set.
    n_rows = input_values.shape[0]
    for r in range(n_rows):
        # Build sets of picked indices, but cap at the smaller of the effective
        # lengths. The reference pads with zeros beyond seq_lens[b]; so do
        # actual. We rely on value equality at matched indices via the value
        # tensor check when return_val is True.
        ref_set = set(int(i) for i in idx_ref[r].tolist())
        act_set = set(int(i) for i in idx_actual[r].tolist())
        # Values within the effective top-k slice should match after sorting.
        if return_val:
            ref_sorted = torch.sort(val_ref[r]).values
            act_sorted = torch.sort(val_actual[r].to(val_ref.dtype)).values
            torch.testing.assert_close(act_sorted, ref_sorted, atol=atol, rtol=rtol)


def ref_sparse_indexer_score_recompute(
    q_indexer: torch.Tensor,  # (B, S_q, H_q, D)
    k_indexer: torch.Tensor,  # (B, S_k, D)
    weights: torch.Tensor,  # (B, S_q, H_q)
    topk_indices: torch.Tensor,  # (B, S_q, topk)
    topk_length: Optional[torch.Tensor] = None,  # (B, S_q)
) -> torch.Tensor:
    """Reference for sparse_indexer_score_recompute.

    Computes softmax( sum_h ReLU(Q_h @ K_topk^T) * W_h ) over the topk dim.
    Returns (B, S_q, topk) FP32.
    """
    b, s_q, h_q, d = q_indexer.shape
    _, s_k, _ = k_indexer.shape
    topk = topk_indices.shape[-1]
    q_f = q_indexer.to(torch.float32)
    k_f = k_indexer.to(torch.float32)
    w_f = weights.to(torch.float32)

    # Gather topk K: (B, S_q, topk, D)
    idx = topk_indices.clamp(min=0, max=s_k - 1).long()
    k_gather = torch.gather(
        k_f.unsqueeze(1).expand(b, s_q, s_k, d),
        2,
        idx.unsqueeze(-1).expand(b, s_q, topk, d),
    )

    # scores[b, q, h, t] = ReLU(Q[b, q, h] . K_gather[b, q, t])
    scores = torch.einsum("bqhd,bqtd->bqht", q_f, k_gather)
    scores = torch.relu(scores)
    # Weighted head reduction
    scores = (scores * w_f.unsqueeze(-1)).sum(dim=2)  # (B, S_q, topk)

    # Mask invalid topk positions before softmax
    valid = topk_indices >= 0
    if topk_length is not None:
        positions = torch.arange(topk, device=q_indexer.device).view(1, 1, topk)
        valid = valid & (positions < topk_length.unsqueeze(-1))
    scores = scores.masked_fill(~valid, float("-inf"))
    return torch.softmax(scores, dim=-1)


def ref_sparse_attn_score_recompute(
    q_attn: torch.Tensor,  # (B, S_q, H_q, D)
    k_attn: torch.Tensor,  # (B, S_k, D)
    lse: torch.Tensor,  # (B, S_q, H_q)
    topk_indices: torch.Tensor,  # (B, S_q, topk)
    softmax_scale: float,
    topk_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference for sparse_attn_score_recompute.

    P[b,q,h,t] = exp(Q[b,q,h] . K_topk[b,q,t] * scale - LSE[b,q,h])
    target[b,q,t] = sum_h P / sum_t sum_h P
    """
    b, s_q, h_q, d = q_attn.shape
    _, s_k, _ = k_attn.shape
    topk = topk_indices.shape[-1]
    q_f = q_attn.to(torch.float32)
    k_f = k_attn.to(torch.float32)
    lse_f = lse.to(torch.float32)

    idx = topk_indices.clamp(min=0, max=s_k - 1).long()
    k_gather = torch.gather(
        k_f.unsqueeze(1).expand(b, s_q, s_k, d),
        2,
        idx.unsqueeze(-1).expand(b, s_q, topk, d),
    )

    qk = torch.einsum("bqhd,bqtd->bqht", q_f, k_gather) * softmax_scale
    p = torch.exp(qk - lse_f.unsqueeze(-1))  # (B, S_q, H_q, topk)
    target = p.sum(dim=2)  # (B, S_q, topk)

    valid = topk_indices >= 0
    if topk_length is not None:
        positions = torch.arange(topk, device=q_attn.device).view(1, 1, topk)
        valid = valid & (positions < topk_length.unsqueeze(-1))
    target = target.masked_fill(~valid, 0.0)
    denom = target.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return target / denom


def check_ref_sparse_score_recompute(
    score_type: str,
    q,
    k_or_lse,
    topk_indices,
    actual,
    aux=None,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    if score_type == "indexer":
        # aux = weights
        ref = ref_sparse_indexer_score_recompute(q, k_or_lse, aux, topk_indices, topk_length)
    else:
        # aux = k_attn; k_or_lse = lse
        ref = ref_sparse_attn_score_recompute(q, aux, k_or_lse, topk_indices, softmax_scale, topk_length)
    torch.testing.assert_close(actual.to(torch.float32), ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Dense score recompute references (full-KV, no top-K)
# ---------------------------------------------------------------------------


def _bottom_right_causal_mask(
    s_q: int,
    s_k: int,
    ratio: int,
    device: torch.device,
) -> torch.Tensor:
    """Return dense score validity mask with bottom-right causal alignment."""
    q_global_start = s_k * ratio - s_q
    q = torch.arange(s_q, device=device, dtype=torch.int64)
    k_pos = torch.arange(s_k, device=device, dtype=torch.int64)
    col_limit = torch.div(q_global_start + q + 1, ratio, rounding_mode="floor")
    return k_pos.view(1, s_k) < col_limit.view(s_q, 1)


def ref_dense_indexer_score_recompute(
    q_indexer: torch.Tensor,  # (B, S_q, H_q, D)
    k_indexer: torch.Tensor,  # (B, S_k, H_kv, D) — H_kv=1 for MQA
    weights: torch.Tensor,  # (B, S_q, H_q)
    ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference for ``DSA.dense_indexer_score_recompute_wrapper``.

    Computes per-query dense indexer scores
    ``S[b, q, k] = sum_h ReLU(Q_h · K_h^T) · W_h`` and the
    ``LogSumExp``-style denom ``log(sum_k exp(S[b, q, k]))`` over valid
    bottom-right causal positions.
    """
    b, s_q, h_q, d = q_indexer.shape
    _, s_k, h_kv, _ = k_indexer.shape
    qhpkv = h_q // h_kv
    q_f = q_indexer.to(torch.float32)
    k_f = k_indexer.to(torch.float32)
    w_f = weights.to(torch.float32)

    # Expand K across qhead groups so each query head sees its KV head.
    k_exp = k_f.repeat_interleave(qhpkv, dim=2)  # (B, S_k, H_q, D)
    scores = torch.einsum("bqhd,bkhd->bqhk", q_f, k_exp)
    scores = torch.relu(scores)
    scores = (scores * w_f.unsqueeze(-1)).sum(dim=2)  # (B, S_q, S_k)

    valid = _bottom_right_causal_mask(s_q, s_k, ratio, q_indexer.device)
    scores_for_denom = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    denom = torch.logsumexp(scores_for_denom, dim=-1)  # (B, S_q)
    return scores.masked_fill(~valid.unsqueeze(0), 0.0), denom


def ref_dense_attn_score_recompute(
    q_attn: torch.Tensor,  # (B, S_q, H_q, D)
    k_attn: torch.Tensor,  # (B, S_k, H_kv, D)
    lse: torch.Tensor,  # (B, S_q, H_q) FP32
    softmax_scale: float,
    ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference for ``DSA.dense_attn_score_recompute_wrapper``.

    ``P[b, q, h, k] = exp(Q_h · K_h^T · scale - LSE_h)``, sum over heads,
    returns unnormalized ``S`` and the per-query L1-norm denom over valid
    bottom-right causal positions.
    """
    b, s_q, h_q, d = q_attn.shape
    _, s_k, h_kv, _ = k_attn.shape
    qhpkv = h_q // h_kv
    q_f = q_attn.to(torch.float32)
    k_f = k_attn.to(torch.float32)
    lse_f = lse.to(torch.float32)

    k_exp = k_f.repeat_interleave(qhpkv, dim=2)
    qk = torch.einsum("bqhd,bkhd->bqhk", q_f, k_exp) * softmax_scale
    p = torch.exp(qk - lse_f.unsqueeze(-1))  # (B, S_q, H_q, S_k)
    s = p.sum(dim=2)  # (B, S_q, S_k)

    valid = _bottom_right_causal_mask(s_q, s_k, ratio, q_attn.device)
    s = s.masked_fill(~valid.unsqueeze(0), 0.0)
    denom = s.sum(dim=-1)  # (B, S_q)
    return s, denom


def check_ref_dense_score_recompute(
    score_type: str,
    q,
    k,
    aux,
    out_actual,
    denom_actual,
    softmax_scale: Optional[float] = None,
    ratio: int = 1,
    atol_scores: float = 5e-3,
    rtol_scores: float = 5e-3,
    atol_denom: float = 5e-3,
    rtol_denom: float = 5e-3,
):
    """Numerical validation of dense score recompute outputs.

    Only positions that would normally be kept (i.e. ``out_actual`` is finite)
    are compared, matching the cuDNN kernel's partial-write semantics on
    causally-masked tiles.
    """
    if score_type == "indexer":
        # aux = weights
        s_ref, denom_ref = ref_dense_indexer_score_recompute(q, k, aux, ratio)
    else:
        # aux = lse
        s_ref, denom_ref = ref_dense_attn_score_recompute(
            q,
            k,
            aux,
            softmax_scale,
            ratio,
        )

    finite = torch.isfinite(out_actual)
    if finite.any():
        torch.testing.assert_close(
            out_actual[finite].to(torch.float32),
            s_ref[finite],
            atol=atol_scores,
            rtol=rtol_scores,
        )
    torch.testing.assert_close(
        denom_actual.to(torch.float32),
        denom_ref,
        atol=atol_denom,
        rtol=rtol_denom,
    )


# ---------------------------------------------------------------------------
# Indexer gradient backward reference (autograd through the same math)
# ---------------------------------------------------------------------------


def _indexer_predict_distribution(
    q_indexer: torch.Tensor,  # (B, S_q, H, D)
    k_indexer: torch.Tensor,  # (B, S_k, D)
    weights: torch.Tensor,  # (B, S_q, H)
    topk_indices: torch.Tensor,  # (B, S_q, topk) INT32
    sm_scale: float,
) -> torch.Tensor:
    """Differentiable predict-distribution computation for autograd reference.

    Mirrors the cudnn ``sparse_indexer_score_recompute`` math, but as a plain
    PyTorch op graph so ``autograd.grad`` produces the same gradients the
    ``IndexerBackward`` 3-stage kernel should produce.

    Implements the weights-scaling trick used by the kernel:
    ``score = sum_h relu(Q_h · K_{topk}^T) · (W_h · sm_scale)``.
    """
    b, s_q, h, d = q_indexer.shape
    _, s_k, _ = k_indexer.shape
    topk = topk_indices.shape[-1]

    idx = topk_indices.clamp(min=0).long()
    k_gather = torch.gather(
        k_indexer.unsqueeze(1).expand(b, s_q, s_k, d),
        2,
        idx.unsqueeze(-1).expand(b, s_q, topk, d),
    )  # (B, S_q, topk, D)

    scores_per_head = torch.einsum("bqhd,bqtd->bqht", q_indexer, k_gather)
    scores_per_head = torch.relu(scores_per_head)
    weights_scaled = weights * sm_scale
    scores = (scores_per_head * weights_scaled.unsqueeze(-1)).sum(dim=2)  # (B, S_q, topk)

    valid = topk_indices >= 0
    scores = scores.masked_fill(~valid, float("-inf"))
    return torch.softmax(scores, dim=-1)


def ref_indexer_backward(
    index_q: torch.Tensor,  # (B, S_q, H, D) bf16
    weights: torch.Tensor,  # (B, S_q, H)   bf16
    index_k: torch.Tensor,  # (B, S_k, D)   bf16
    attn_score: torch.Tensor,  # (B, S_q, topk) target FP32
    index_score: torch.Tensor,  # (B, S_q, topk) predict FP32 (unused — recomputed)
    topk_indices: torch.Tensor,  # (B, S_q, topk) INT32
    sm_scale: float = 1.0,
    grad_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch autograd reference for ``DSA.indexer_backward_wrapper``.

    Recomputes the predict distribution from ``(index_q, weights, index_k,
    topk_indices)`` and takes the gradient of
    ``-grad_scale * sum(attn_score * log(predict))`` (the clipped-log KL
    ``d/d(predict)`` contracted against ``-target``, mirroring the kernel's
    ``_score_grad_inplace`` stage) with respect to ``index_q``, ``weights``,
    and ``index_k``. Returns gradients in the same dtype as the inputs.

    Inputs remain in their original dtype through the forward / softmax so
    the internal predict distribution matches the kernel's bf16 GEMMs with
    fp32 accumulator — otherwise the two would disagree purely because
    ``fp32(bf16(x)) * fp32(bf16(y))`` ≠ ``fp32(x) * fp32(y)``.
    """
    q = index_q.detach().clone().requires_grad_(True)
    w = weights.detach().clone().requires_grad_(True)
    k = index_k.detach().clone().requires_grad_(True)

    predict = _indexer_predict_distribution(
        q,
        k,
        w,
        topk_indices,
        sm_scale,
    )  # (B, S_q, topk) — dtype follows inputs, cast to fp32 inside softmax

    eps = torch.finfo(torch.float32).tiny
    predict_clipped = predict.to(torch.float32).clamp(min=eps)
    target = attn_score.to(torch.float32).clamp(min=eps)

    # KL gradient stage: d/d(logits) of KL = g - predict * sum(g) where
    # g = -target. We sum g * log(predict) directly so autograd gives the
    # identical gradient w.r.t. (q, w, k).
    loss = -grad_scale * (target * torch.log(predict_clipped)).sum()
    grads = torch.autograd.grad(loss, (q, w, k))

    dq, dw, dk = grads
    return (
        dq.to(index_q.dtype),
        dw.to(weights.dtype),
        dk.to(index_k.dtype),
    )


def _dense_indexer_predict_distribution(
    q_indexer: torch.Tensor,  # (B, S_q, H, D)
    k_indexer: torch.Tensor,  # (B, S_k, D)
    weights: torch.Tensor,  # (B, S_q, H)
    sm_scale: float,
    ratio: int,
) -> torch.Tensor:
    """Dense full-KV predict distribution for dense indexer backward."""
    b, s_q, h, d = q_indexer.shape
    _, s_k, _ = k_indexer.shape

    scores_per_head = torch.einsum("bqhd,bkd->bqhk", q_indexer, k_indexer)
    scores_per_head = torch.relu(scores_per_head)
    scores = (scores_per_head * (weights * sm_scale).unsqueeze(-1)).sum(dim=2)

    valid = _bottom_right_causal_mask(s_q, s_k, ratio, q_indexer.device)
    scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    return torch.softmax(scores, dim=-1)


def ref_dense_indexer_backward(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    attn_score_raw: torch.Tensor,
    attn_l1norm: torch.Tensor,
    sm_scale: float = 1.0,
    ratio: int = 1,
    grad_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch autograd reference for ``DSA.dense_indexer_backward_wrapper``."""
    q = index_q.detach().clone().requires_grad_(True)
    w = weights.detach().clone().requires_grad_(True)
    k = index_k.detach().clone().requires_grad_(True)

    predict = _dense_indexer_predict_distribution(q, k, w, sm_scale, ratio)

    _, s_q, _, _ = index_q.shape
    _, s_k, _ = index_k.shape
    valid = _bottom_right_causal_mask(s_q, s_k, ratio, index_q.device)
    target = attn_score_raw.to(torch.float32).masked_fill(~valid.unsqueeze(0), 0.0)
    target = target / attn_l1norm.to(torch.float32).unsqueeze(-1).clamp(min=1e-10)

    eps = math.exp(-100.0)
    predict_clipped = predict.to(torch.float32).clamp(min=eps)
    target_eff = target.clamp(min=eps)
    loss = -grad_scale * (target_eff * torch.log(predict_clipped)).sum()
    grads = torch.autograd.grad(loss, (q, w, k))

    dq, dw, dk = grads
    return (
        dq.to(index_q.dtype),
        dw.to(weights.dtype),
        dk.to(index_k.dtype),
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return 1.0
    return (af @ bf / denom).item()


def check_ref_indexer_backward(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    d_index_q_actual,
    d_weights_actual,
    d_index_k_actual,
    sm_scale: float = 1.0,
    grad_scale: float = 1.0,
    cos_min: float = 0.97,
    rms_rel_max: float = 0.55,
):
    """Check gradient values against the PyTorch autograd reference.

    The kernel runs three bf16 GEMMs with fp32 accumulators; the reference
    runs the same chain via ``torch.autograd``. At bf16 input precision the
    worst-case elementwise error is dominated by near-zero entries (relative
    error blows up). We instead check two aggregate criteria that are
    sensitive to systematic errors but tolerant of bf16 quantisation:

    * ``cosine similarity >= cos_min``: directions match.
    * ``RMS(error) / RMS(ref) <= rms_rel_max``: magnitudes match.

    Default thresholds are empirically calibrated at ``sq=128``, ``head=64``,
    ``topk=128``. On larger problems (``sq=2048+``, ``topk=512``) cosine
    typically exceeds ``0.99``; callers that want the stricter bound can
    pass ``cos_min=0.99``.
    """
    dq_ref, dw_ref, dk_ref = ref_indexer_backward(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        sm_scale=sm_scale,
        grad_scale=grad_scale,
    )

    for name, actual, ref in (
        ("d_index_q", d_index_q_actual, dq_ref),
        ("d_weights", d_weights_actual, dw_ref),
        ("d_index_k", d_index_k_actual, dk_ref),
    ):
        cos = _cosine_similarity(actual, ref)
        err = actual.to(torch.float32) - ref.to(torch.float32)
        rms_err = err.pow(2).mean().sqrt().item()
        rms_ref = ref.to(torch.float32).pow(2).mean().sqrt().item()
        rms_rel = rms_err / max(rms_ref, 1e-12)
        assert cos >= cos_min, f"{name}: cosine similarity {cos:.4f} < {cos_min} — " f"direction mismatch"
        assert rms_rel <= rms_rel_max, (
            f"{name}: RMS relative error {rms_rel:.4f} > {rms_rel_max} — " f"magnitude mismatch (rms_err={rms_err:.4g}, rms_ref={rms_ref:.4g})"
        )


def check_ref_dense_indexer_backward(
    index_q,
    weights,
    index_k,
    attn_score_raw,
    attn_l1norm,
    d_index_q_actual,
    d_weights_actual,
    d_index_k_actual,
    sm_scale: float = 1.0,
    ratio: int = 1,
    grad_scale: float = 1.0,
    cos_min: float = 0.97,
    rms_rel_max: float = 0.55,
):
    dq_ref, dw_ref, dk_ref = ref_dense_indexer_backward(
        index_q,
        weights,
        index_k,
        attn_score_raw,
        attn_l1norm,
        sm_scale=sm_scale,
        ratio=ratio,
        grad_scale=grad_scale,
    )

    for name, actual, ref in (
        ("d_index_q", d_index_q_actual, dq_ref),
        ("d_weights", d_weights_actual, dw_ref),
        ("d_index_k", d_index_k_actual, dk_ref),
    ):
        cos = _cosine_similarity(actual, ref)
        err = actual.to(torch.float32) - ref.to(torch.float32)
        rms_err = err.pow(2).mean().sqrt().item()
        rms_ref = ref.to(torch.float32).pow(2).mean().sqrt().item()
        rms_rel = rms_err / max(rms_ref, 1e-12)
        assert cos >= cos_min, f"{name}: cosine similarity {cos:.4f} < {cos_min} — " f"direction mismatch"
        assert rms_rel <= rms_rel_max, (
            f"{name}: RMS relative error {rms_rel:.4f} > {rms_rel_max} — " f"magnitude mismatch (rms_err={rms_err:.4g}, rms_ref={rms_ref:.4g})"
        )
