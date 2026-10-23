# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 indexer forward over a paged K pool (block_table + kv_lens) on SM100.

Every case scatters per-sequence dense keys into a page pool through a random page permutation, keeps the
unused pool rows NaN, and checks the kernel against the fp64 reference computed on the dense keys.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_reference import (
    check_ref_compressed_topk,
    ref_indexer_forward,
)
from deepseek_sparse_attention.cutedsl.test_DSA_indexer_forward_compressed import _assert_topk_sets_match

D = 128
TOL = 2e-3


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ GPU required")


def _dsa():
    try:
        from cudnn import DSA

        return DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


def _paged_pool(kv_lens_host, page_size, device, *, spare_pages=3):
    """Dense per-sequence keys scattered into a NaN-filled pool through a random page permutation.

    Returns ``(pool, block_table, kv_lens, dense_k)``; ``block_table`` slots past a sequence's live pages are -1.
    """
    bs = len(kv_lens_host)
    max_pages = max((L + page_size - 1) // page_size for L in kv_lens_host)
    num_pages = bs * max_pages + spare_pages
    pool = torch.full((num_pages, page_size, 1, D), float("nan"), dtype=torch.bfloat16, device=device)
    perm = torch.randperm(num_pages, device=device)[: bs * max_pages].view(bs, max_pages)
    block_table = torch.full((bs, max_pages), -1, dtype=torch.int32, device=device)
    dense_k = []
    for b, L in enumerate(kv_lens_host):
        n_pages_b = (L + page_size - 1) // page_size
        pages = perm[b, :n_pages_b]
        block_table[b, :n_pages_b] = pages.to(torch.int32)
        kb = torch.randn(L, 1, D, dtype=torch.bfloat16, device=device)
        dense_k.append(kb)
        rows = pool[pages].view(-1, 1, D)
        rows[:L] = kb
        pool[pages] = rows.view(n_pages_b, page_size, 1, D)
    kv_lens = torch.tensor(kv_lens_host, dtype=torch.int32, device=device)
    return pool, block_table, kv_lens, dense_k


def _check_batch(q_b, k_b, w_b, ratio, offset, sm_scale, top_k, indices, logits, scores, L):
    """One sequence: fused top-k set + values against the dense fp64 reference, dense scores and -inf past ``L``."""
    ref = ref_indexer_forward(q_b, k_b, w_b, ratio, q_causal_offsets=offset, compute_dtype=torch.float64) * sm_scale
    if top_k <= L:
        check_ref_compressed_topk(ref, indices.unsqueeze(0), logits.unsqueeze(0), top_k, atol=TOL, rtol=TOL)
    _assert_topk_sets_match(ref.squeeze(0), indices, top_k, tie_tol=TOL, logits=logits)
    finite = torch.isfinite(ref[0])
    assert torch.equal(torch.isneginf(scores[:, :L]), torch.isneginf(ref[0]))
    torch.testing.assert_close(scores[:, :L][finite], ref[0][finite], atol=TOL, rtol=TOL)
    assert bool(torch.isneginf(scores[:, L:]).all())


@pytest.mark.L0
@torch_fork_set_rng(seed=61)
@pytest.mark.parametrize("page_size", [64, 256])
def test_DSA_indexer_forward_paged_bshd(page_size):
    """BSHD queries over a paged pool: ragged key counts (partial last page, one and several pages per sequence),
    chunked causal offsets so every row's reach ends inside its sequence, and a sequence whose keys stop short of
    the causal reach (rows clamp to kv_lens)."""
    _require_sm100()
    DSA = _dsa()
    device = torch.device("cuda")
    kv_lens_host = [300, 97, 256, 40]
    ratio, top_k, s_q, h_q, sm_scale = 4, 24, 64, 32, D**-0.5
    pool, block_table, kv_lens, dense_k = _paged_pool(kv_lens_host, page_size, device)
    # The last sequence's offset asks for 100 keys but only 40 exist.
    offsets_host = [L * ratio - s_q for L in kv_lens_host[:-1]] + [100 * ratio - s_q]
    q_causal_offsets = torch.tensor(offsets_host, dtype=torch.int32, device=device)
    bs = len(kv_lens_host)
    q = torch.randn(bs, s_q, h_q, D, dtype=torch.bfloat16, device=device)
    w = torch.randn(bs, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1

    fused = DSA.indexer_forward_top_k_wrapper(
        q,
        pool,
        w,
        top_k=top_k,
        ratio=ratio,
        sm_scale=sm_scale,
        q_causal_offsets=q_causal_offsets,
        block_table=block_table,
        kv_lens=kv_lens,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
    )
    scores = DSA.indexer_forward_wrapper(
        q,
        pool,
        w,
        ratio=ratio,
        sm_scale=sm_scale,
        q_causal_offsets=q_causal_offsets,
        block_table=block_table,
        kv_lens=kv_lens,
    )["scores"]
    torch.cuda.synchronize()
    assert fused["indices"].shape == (bs, s_q, top_k)
    assert scores.shape == (bs, s_q, block_table.shape[1] * page_size)
    for b, L in enumerate(kv_lens_host):
        _check_batch(
            q[b : b + 1],
            dense_k[b].unsqueeze(0),
            w[b : b + 1],
            ratio,
            q_causal_offsets[b : b + 1],
            sm_scale,
            top_k,
            fused["indices"][b],
            fused["logits"][b],
            scores[b],
            L,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=62)
@pytest.mark.parametrize("page_size", [16, 64])
def test_DSA_indexer_forward_paged_thd(page_size):
    """THD queries over a paged pool (no cu_seqlens_k): per-sequence query lengths and offsets, a one-token
    sequence, and max_seqlen_k left to its default of max_pages * page_size."""
    _require_sm100()
    DSA = _dsa()
    device = torch.device("cuda")
    # (seqlen_q, kv_len, q_causal_offset)
    shapes = [(64, 200, 200 * 4 - 64), (1, 1, 0), (100, 97, 97 * 4 - 100), (30, 300, 1100)]
    ratio, top_k, h_q, sm_scale = 4, 12, 16, D**-0.5
    kv_lens_host = [s[1] for s in shapes]
    pool, block_table, kv_lens, dense_k = _paged_pool(kv_lens_host, page_size, device)
    cu_q = torch.tensor([0, *torch.tensor([s[0] for s in shapes]).cumsum(0).tolist()], dtype=torch.int32, device=device)
    q_causal_offsets = torch.tensor([s[2] for s in shapes], dtype=torch.int32, device=device)
    total_q = int(cu_q[-1])
    q = torch.randn(total_q, h_q, D, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    common = dict(
        ratio=ratio,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(s[0] for s in shapes),
        q_causal_offsets=q_causal_offsets,
        block_table=block_table,
        kv_lens=kv_lens,
    )

    fused = DSA.indexer_forward_top_k_wrapper(q, pool, w, top_k=top_k, topk_indices_global=False, return_softmax=False, deterministic=True, **common)
    scores = DSA.indexer_forward_wrapper(q, pool, w, **common)["scores"]
    torch.cuda.synchronize()
    assert fused["indices"].shape == (total_q, top_k)
    assert scores.shape == (total_q, block_table.shape[1] * page_size)
    cu_host = cu_q.tolist()
    for b, (_, L, _) in enumerate(shapes):
        q0, q1 = cu_host[b : b + 2]
        _check_batch(
            q[q0:q1].unsqueeze(0),
            dense_k[b].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            q_causal_offsets[b : b + 1],
            sm_scale,
            top_k,
            fused["indices"][q0:q1],
            fused["logits"][q0:q1],
            scores[q0:q1],
            L,
        )
    # The one-token sequence has no complete block: nothing selected.
    assert bool((fused["indices"][cu_host[1]] == -1).all())


@pytest.mark.L0
@torch_fork_set_rng(seed=63)
def test_DSA_indexer_forward_paged_matches_dense_and_ignores_page_order():
    """The same keys through two different page permutations give bitwise-identical scores, both equal the
    dense BSHD kernel on the gathered keys (the paged path changes addressing only), and a 3-D pool reads the same."""
    _require_sm100()
    DSA = _dsa()
    device = torch.device("cuda")
    kv_lens_host = [384, 384]
    ratio, top_k, s_q, h_q, page_size, sm_scale = 4, 32, 128, 32, 128, D**-0.5
    pool, block_table, kv_lens, dense_k = _paged_pool(kv_lens_host, page_size, device)
    # Second pool: the same dense keys under another permutation.
    pool2, block_table2, _, _ = _paged_pool(kv_lens_host, page_size, device)
    for b, kb in enumerate(dense_k):
        pages = block_table2[b, : kb.shape[0] // page_size].long()
        pool2[pages] = kb.view(-1, page_size, 1, D)
    bs = len(kv_lens_host)
    offsets = torch.full((bs,), 384 * ratio - s_q, dtype=torch.int32, device=device)
    q = torch.randn(bs, s_q, h_q, D, dtype=torch.bfloat16, device=device)
    w = torch.randn(bs, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    k_dense = torch.stack(dense_k)

    def run(k, **paged):
        fused = DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=top_k,
            ratio=ratio,
            sm_scale=sm_scale,
            q_causal_offsets=offsets,
            topk_indices_global=False,
            return_softmax=False,
            deterministic=True,
            **paged,
        )
        scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, sm_scale=sm_scale, q_causal_offsets=offsets, **paged)["scores"]
        torch.cuda.synchronize()
        return torch.sort(fused["indices"], dim=-1).values, torch.sort(fused["logits"], dim=-1).values, scores

    idx_a, val_a, scores_a = run(pool, block_table=block_table, kv_lens=kv_lens)
    idx_b, val_b, scores_b = run(pool2, block_table=block_table2, kv_lens=kv_lens)
    idx_d, val_d, scores_d = run(k_dense)
    assert torch.equal(idx_a, idx_b) and torch.equal(val_a, val_b) and torch.equal(scores_a, scores_b)
    assert torch.equal(idx_a, idx_d) and torch.equal(val_a, val_d) and torch.equal(scores_a, scores_d)
    # A single-head key cache may omit the head axis, as the QSA compressed-key cache does.
    idx_3, val_3, scores_3 = run(pool.squeeze(2), block_table=block_table, kv_lens=kv_lens)
    assert torch.equal(idx_a, idx_3) and torch.equal(val_a, val_3) and torch.equal(scores_a, scores_3)


@pytest.mark.L0
def test_DSA_indexer_forward_paged_declines():
    """Off-contract paged requests decline before any launch: a page that does not hold whole 128-row boxes,
    cu_seqlens_k next to a block table, a missing kv_lens, global ids, MXFP8 and explicit microbatching."""
    _require_sm100()
    DSA = _dsa()
    device = torch.device("cuda")
    bs, s_q, h_q, page_size = 2, 8, 4, 128
    pool = torch.randn(bs + 1, page_size, 1, D, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(bs, dtype=torch.int32, device=device).view(bs, 1)
    kv_lens = torch.full((bs,), 2, dtype=torch.int32, device=device)
    q = torch.randn(bs, s_q, h_q, D, dtype=torch.bfloat16, device=device)
    w = torch.randn(bs, s_q, h_q, dtype=torch.bfloat16, device=device)
    kwargs = dict(top_k=1, block_table=block_table, kv_lens=kv_lens, topk_indices_global=False, return_softmax=False)

    with pytest.raises(ValueError, match="multiple of 8 that divides 128"):
        DSA.indexer_forward_top_k_wrapper(q, pool[:, :48].contiguous(), w, **kwargs)
    with pytest.raises(ValueError, match="kv_lens"):
        DSA.indexer_forward_top_k_wrapper(q, pool, w, **{**kwargs, "kv_lens": None})
    with pytest.raises(ValueError, match="kv_lens requires block_table"):
        DSA.indexer_forward_top_k_wrapper(q, pool[:bs], w, **{**kwargs, "block_table": None})
    with pytest.raises(ValueError, match="cu_seqlens_k must be None"):
        cu = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        DSA.indexer_forward_top_k_wrapper(
            q.view(bs * s_q, h_q, D), pool, w.view(bs * s_q, h_q), cu_seqlens_q=cu * s_q, cu_seqlens_k=cu, max_seqlen_q=s_q, **kwargs
        )
    with pytest.raises(NotImplementedError, match="local key ids"):
        DSA.indexer_forward_top_k_wrapper(q, pool, w, **{**kwargs, "topk_indices_global": True})
    with pytest.raises(NotImplementedError, match="microbatch_rows"):
        DSA.indexer_forward_top_k_wrapper(q, pool, w, microbatch_rows=4, **kwargs)
    with pytest.raises(NotImplementedError, match="BF16"):
        q8, k8 = q.to(torch.float8_e4m3fn), pool.to(torch.float8_e4m3fn)
        sf = torch.ones((1, 128, 4), device=device).to(torch.float8_e8m0fnu)
        DSA.indexer_forward_top_k_wrapper(q8, k8, w, precision="mxfp8", q_scale=sf, k_scale=sf, **kwargs)
    with pytest.raises(ValueError, match="multiple of 8 that divides 128"):
        DSA.indexer_forward_wrapper(q, pool[:, :48].contiguous(), w, block_table=block_table, kv_lens=kv_lens)
    with pytest.raises(NotImplementedError, match="BF16"):
        q8, k8 = q.to(torch.float8_e4m3fn), pool.to(torch.float8_e4m3fn)
        sf = torch.ones((1, 128, 4), device=device).to(torch.float8_e8m0fnu)
        DSA.indexer_forward_wrapper(q8, k8, w, precision="mxfp8", q_scale=sf, k_scale=sf, block_table=block_table, kv_lens=kv_lens)
