# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pooled DSA training backward: pooling, expansion, indexer-loss chain, attention backward."""

import pytest
import torch

from cudnn.deepseek_sparse_attention.pooled import backward as PB
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
from deepseek_sparse_attention.cutedsl import pooled_training_reference as R
from deepseek_sparse_attention.cutedsl.dsa_reference import ref_sparse_attn_score_recompute

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(torch.cuda.get_device_capability() not in ((10, 0), (10, 3)), reason="Pooled DSA requires Blackwell"),
    pytest.mark.skipif(cutedsl_too_old(cutedsl_state()[1]), reason="Pooled DSA requires a supported CuTe DSL version"),
]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-20)).item()


def _dense_causal_lse(q_attn, k_attn, softmax_scale):
    s = q_attn.shape[1]
    causal = torch.ones(s, s, dtype=torch.bool, device=q_attn.device).tril().view(1, s, 1, s)
    logits = (torch.einsum("bqhd,bkd->bqhk", q_attn.float(), k_attn.float()) * softmax_scale).masked_fill(~causal, float("-inf"))
    return torch.logsumexp(logits, dim=-1)


def _selected(c, k_pools):
    pool_size = c["pool_size"]
    pooled = PB.pooled_keys_forward(c["keys"], c["gates"], c["ape"], pool_size)["pooled"]
    scores = R.pooled_dense_scores(c["index_q"], c["weights"], pooled, 1.0)
    pool_ids = R.select_pools(scores, R.eligible_pools(c["positions"], pooled.shape[1], pool_size), k_pools)
    return pooled, pool_ids


@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_pooled_keys_forward_matches_reference(pool_size, out_dtype):
    c = R.make_case(2, 1000, torch.device("cuda"), pool_size=pool_size, seed=3)
    out = torch.empty(2, 1000 // pool_size, 128, dtype=out_dtype, device="cuda")
    got = PB.pooled_keys_forward(c["keys"], c["gates"], c["ape"], pool_size, out=out)["pooled"]
    ref, _ = R.pool_keys(c["keys"], c["gates"], c["ape"], pool_size)
    tol = 2**-7 if out_dtype == torch.bfloat16 else 1e-5
    assert (got.float() - ref).abs().max().item() <= tol * ref.abs().max().item() + 1e-6


@pytest.mark.parametrize("batch,seqlen,k_pools", [(1, 1024, 128), (2, 777, 32), (3, 4096, 512)])
def test_expand_compact_matches_reference(batch, seqlen, k_pools):
    c = R.make_case(batch, seqlen, torch.device("cuda"), topk_tokens=k_pools * 4, seed=batch)
    _, pool_ids = _selected(c, k_pools)
    ref_tok, ref_len = R.expand_pools_compact(pool_ids, c["positions"], 4)
    got = PB.pooled_expand_compact(pool_ids, c["positions"], 4, seqlen_k=seqlen, global_ids=True)
    width = ref_tok.shape[-1]
    assert got["token_ids"].shape[-1] % 64 == 0 and got["token_ids"].shape[-1] >= width
    offset = (torch.arange(batch, device="cuda") * seqlen).view(batch, 1, 1)
    assert torch.equal(got["token_ids"][..., :width], torch.where(ref_tok >= 0, ref_tok + offset, ref_tok).to(torch.int32))
    assert (got["token_ids"][..., width:] == -1).all()
    assert torch.equal(got["topk_length"], ref_len.to(torch.int32))
    local = PB.pooled_expand_compact(pool_ids, c["positions"], 4, global_ids=False, width_multiple=1)
    assert torch.equal(local["token_ids"], ref_tok.to(torch.int32))


@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
def test_unpool_backward_matches_autograd(grad_dtype):
    c = R.make_case(2, 520, torch.device("cuda"), seed=5)
    keys, gates, ape = c["keys"], c["gates"], c["ape"]
    d_pooled = torch.randn(2, 130, 128, device="cuda")
    leaves = [t.detach().float().requires_grad_(True) for t in (keys, gates, ape)]
    pooled, _ = R.pool_keys(*leaves, 4)
    ref = torch.autograd.grad(pooled, leaves, grad_outputs=d_pooled)
    got = PB.unpool_backward(
        keys,
        gates,
        ape,
        d_pooled,
        4,
        d_keys=torch.empty(2, 520, 128, dtype=grad_dtype, device="cuda"),
        d_gates=torch.empty(2, 520, 128, dtype=grad_dtype, device="cuda"),
    )
    tol = 2e-2 if grad_dtype == torch.bfloat16 else 1e-4
    assert _rel(got["d_keys"], ref[0]) < tol
    assert _rel(got["d_gates"], ref[1]) < tol
    assert _rel(got["d_ape"], ref[2]) < 1e-4


def test_pooled_indexer_backward_matches_oracle():
    c = R.make_case(1, 1024, torch.device("cuda"), topk_tokens=512, seed=1)
    pool_size, k_pools = 4, 128
    pooled, pool_ids = _selected(c, k_pools)
    softmax_scale = c["q_attn"].shape[-1] ** -0.5
    token_ids, length = R.expand_pools_compact(pool_ids, c["positions"], pool_size)
    lse = _dense_causal_lse(c["q_attn"], c["k_attn"], softmax_scale)
    token_target = ref_sparse_attn_score_recompute(c["q_attn"], c["k_attn"], lse, token_ids, softmax_scale, topk_length=length)
    target = PB.pooled_indexer_target(token_target, pool_ids, pool_size)["target"]
    has_pool = ((c["positions"] + 1) // pool_size) >= 1
    assert torch.allclose(target.sum(-1)[has_pool], torch.ones_like(target.sum(-1)[has_pool]), atol=1e-4)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    got = PB.pooled_indexer_backward(c["index_q"], c["weights"], c["keys"], c["gates"], c["ape"], pool_ids, target.clone(), grad_loss, pool_size, 1.0, 1.0)
    ref = R.oracle_pooled_indexer_backward(c["index_q"], c["weights"], c["keys"], c["gates"], c["ape"], pool_ids, target, 1.0, 1.0, pool_size)
    for name, r in zip(("d_index_q", "d_weights", "d_keys", "d_gates", "d_ape"), ref):
        assert _rel(got[name], r) < 1e-2, name


def test_pooled_sparse_attention_backward_matches_compaction_route():
    from cudnn.deepseek_sparse_attention.sparse_attention_backward.api import sparse_attention_backward_wrapper
    from cudnn.deepseek_sparse_attention.sparse_attention_forward.api import sparse_attention_forward_wrapper

    c = R.make_case(1, 2048, torch.device("cuda"), topk_tokens=512, seed=7)
    pool_size, k_pools = 4, 128
    _, pool_ids = _selected(c, k_pools)
    q, kv, positions = c["q_attn"], c["k_attn"], c["positions"]
    s, h, d = q.shape[1:]
    softmax_scale = d**-0.5
    expanded = PB.pooled_expand_compact(pool_ids, positions, pool_size, seqlen_k=s, global_ids=True)
    fwd = sparse_attention_forward_wrapper(
        q.reshape(s, h, d), kv.reshape(s, d), expanded["token_ids"].reshape(s, -1), topk_length=expanded["topk_length"].reshape(s), softmax_scale=softmax_scale
    )
    out, lse = fwd["out"].reshape(1, s, h, -1), fwd["lse"].reshape(1, s, h).float()
    dout = torch.randn_like(out)
    got = PB.pooled_sparse_attention_backward(q, kv, out, dout, lse, pool_ids, positions, pool_size, softmax_scale=softmax_scale)
    inference_ids = R.expand_pools_inference(pool_ids, positions, pool_size).reshape(s, -1)
    idx_c, len_c = R.xtuner_compact_indices(inference_ids)
    sink = torch.full((h,), float("-inf"), dtype=torch.float32, device="cuda")
    ref = sparse_attention_backward_wrapper(
        q.reshape(s, h, d),
        kv.reshape(s, d),
        out.reshape(s, h, -1),
        dout.reshape(s, h, -1),
        lse.reshape(s, h),
        sink,
        idx_c,
        softmax_scale=softmax_scale,
        topk_length=len_c,
    )
    assert _rel(got["dq"].reshape(s, h, d), ref["dq"]) < 2e-2
    assert _rel(got["dkv"].reshape(s, d), ref["dkv"]) < 2e-2
