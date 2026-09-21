# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_sparse_score_recompute_params
from fe_api.dsa.dsa_reference import check_ref_sparse_score_recompute


def _allocate(cfg, score_type: str, has_topk_length: bool):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    topk = cfg["topk"]
    device = "cuda"

    q = torch.randn(b, s_q, qhpkv, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, d, dtype=torch.bfloat16, device=device)

    # Random top-K indices in [0, s_k). Use a guaranteed-valid range.
    topk_k = min(topk, s_k)
    topk_indices = torch.stack([torch.stack([torch.randperm(s_k, device=device)[:topk_k] for _ in range(s_q)]) for _ in range(b)]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((b, s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk_k + 1, (b, s_q), dtype=torch.int32, device=device)
        positions = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, topk)
        topk_indices = topk_indices.masked_fill(positions >= topk_length.unsqueeze(-1), -1)

    if score_type == "indexer":
        weights = torch.randn(b, s_q, qhpkv, dtype=torch.bfloat16, device=device)
        return q, k, weights, topk_indices, topk_length
    else:
        lse = torch.randn(b, s_q, qhpkv, dtype=torch.float32, device=device)
        return q, k, lse, topk_indices, topk_length


def _local_to_global_topk_indices(topk_indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    batch_offsets = torch.arange(topk_indices.shape[0], device=topk_indices.device, dtype=torch.int32).view(-1, 1, 1).mul_(int(seqlen_k))
    return torch.where(topk_indices >= 0, topk_indices + batch_offsets, topk_indices)


@pytest.mark.L1
@torch_fork_set_rng(seed=1234)
@pytest.mark.parametrize("num_heads,head_dim,topk", [(32, 128, 128), (64, 512, 512), (128, 512, 1024), (64, 576, 1024), (128, 576, 512)])
@pytest.mark.parametrize("has_topk_length", [False, True])
@pytest.mark.parametrize("use_global", [False, True])
def test_sparse_attn_sm100_tiles_match_reference(num_heads, head_dim, topk, has_topk_length, use_global):
    from fe_api.dsa.dsa_utils import _require_exact_sm100

    _require_exact_sm100()
    from cudnn import DSA

    cfg = {"b": 2, "s_q": 32, "s_kv": 2048, "head_dim": head_dim, "qhead_per_kv_head": num_heads, "topk": topk}
    q, k, lse, local_ids, lengths = _allocate(cfg, "attention", has_topk_length)
    # Exercise partial final tiles across n128/n64 and K-split specializations,
    # preserving the invalid sentinel when converting to global IDs.
    valid_topk = topk * 3 // 4 + 5
    local_ids[..., valid_topk:] = -1
    if lengths is not None:
        lengths.clamp_(max=valid_topk)
    ids = _local_to_global_topk_indices(local_ids, cfg["s_kv"]) if use_global else local_ids
    scale = 1.0 / math.sqrt(cfg["head_dim"])
    actual = DSA.sparse_attn_score_recompute_wrapper(q, k, lse, ids, scale, topk_length=lengths, topk_indices_global=use_global)["target"]
    check_ref_sparse_score_recompute("attention", q, lse, local_ids, actual, aux=k, softmax_scale=scale, topk_length=lengths)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_sparse_score_recompute_params
def test_DSA_sparse_score_recompute_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    score_type,
    has_topk_length,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        score_type=score_type,
        has_topk_length=has_topk_length,
        min_compute_capability=90,
        s_q_default=256,
        s_kv_default=2048,
    )
    q, k, aux, topk_indices, topk_length = _allocate(
        cfg,
        score_type,
        has_topk_length=has_topk_length,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        if score_type == "indexer":
            result = DSA.sparse_indexer_score_recompute_wrapper(
                q,
                k,
                aux,
                topk_indices,
                qhead_per_kv_head=qhead_per_kv_head,
                topk_length=topk_length,
                stream=stream,
            )
            actual = result["predict"]
        else:
            softmax_scale = 1.0 / math.sqrt(head_dim)
            result = DSA.sparse_attn_score_recompute_wrapper(
                q,
                k,
                aux,
                topk_indices,
                softmax_scale,
                qhead_per_kv_head=qhead_per_kv_head,
                topk_length=topk_length,
                stream=stream,
            )
            actual = result["target"]
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    if not cfg["skip_ref"]:
        if score_type == "indexer":
            check_ref_sparse_score_recompute(
                "indexer",
                q,
                k,
                topk_indices,
                actual,
                aux=aux,
                topk_length=topk_length,
            )
        else:
            check_ref_sparse_score_recompute(
                "attention",
                q,
                aux,
                topk_indices,
                actual,
                aux=k,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
            )


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@pytest.mark.parametrize("has_topk_length", [False, True])
def test_DSA_sparse_score_recompute_wrapper_batch_gt_one(score_type, has_topk_length, request):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=128,
        qhead_per_kv_head=32,
        topk=128,
        score_type=score_type,
        has_topk_length=has_topk_length,
        min_compute_capability=90,
        b_default=2,
        s_q_default=32,
        s_kv_default=256,
    )
    q, k, aux, topk_indices, topk_length = _allocate(
        cfg,
        score_type,
        has_topk_length=has_topk_length,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    for use_global in (False, True):
        topk_for_kernel = _local_to_global_topk_indices(topk_indices, cfg["s_kv"]) if use_global else topk_indices
        try:
            if score_type == "indexer":
                result = DSA.sparse_indexer_score_recompute_wrapper(
                    q,
                    k,
                    aux,
                    topk_for_kernel,
                    qhead_per_kv_head=cfg["qhead_per_kv_head"],
                    topk_length=topk_length,
                    topk_indices_global=use_global,
                    stream=stream,
                )
                actual = result["predict"]
            else:
                softmax_scale = 1.0 / math.sqrt(cfg["head_dim"])
                result = DSA.sparse_attn_score_recompute_wrapper(
                    q,
                    k,
                    aux,
                    topk_for_kernel,
                    softmax_scale,
                    qhead_per_kv_head=cfg["qhead_per_kv_head"],
                    topk_length=topk_length,
                    topk_indices_global=use_global,
                    stream=stream,
                )
                actual = result["target"]
        except (ValueError, NotImplementedError, RuntimeError) as e:
            pytest.skip(f"Unsupported testcase: {e}")

        if not cfg["skip_ref"]:
            if score_type == "indexer":
                check_ref_sparse_score_recompute(
                    "indexer",
                    q,
                    k,
                    topk_indices,
                    actual,
                    aux=aux,
                    topk_length=topk_length,
                )
            else:
                check_ref_sparse_score_recompute(
                    "attention",
                    q,
                    aux,
                    topk_indices,
                    actual,
                    aux=k,
                    softmax_scale=softmax_scale,
                    topk_length=topk_length,
                )


# ---------------------------------------------------------------------------
# Rule 8 detectors on the classes (recipes R5 / R3 / R9)
# ---------------------------------------------------------------------------

_SMALL_CFG = {"b": 2, "s_q": 8, "s_kv": 256, "head_dim": 128, "qhead_per_kv_head": 32, "topk": 128}


def _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out, topk_length=None):
    if score_type == "indexer":
        return DSA.SparseIndexerScoreRecompute(
            sample_q_indexer=q,
            sample_k_indexer=k,
            sample_weights=aux,
            sample_topk_indices=topk_indices,
            sample_out=out,
            sample_topk_length=topk_length,
            qhead_per_kv_head=q.shape[2],
        )
    return DSA.SparseAttnScoreRecompute(
        sample_q_attn=q,
        sample_k_attn=k,
        sample_lse=aux,
        sample_topk_indices=topk_indices,
        sample_out=out,
        softmax_scale=q.shape[-1] ** -0.5,
        sample_topk_length=topk_length,
        qhead_per_kv_head=q.shape[2],
    )


def _sparse_case(score_type, has_topk_length=False):
    from cudnn import DSA

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Sparse score recompute requires SM90+")
    q, k, aux, topk_indices, topk_length = _allocate(_SMALL_CFG, score_type, has_topk_length)
    out = torch.empty(topk_indices.shape, dtype=torch.float32, device="cuda")
    return DSA, q, k, aux, topk_indices, topk_length, out


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_sparse_score_recompute_output_contiguity_declined(score_type):
    """R5: a layout the kernel cannot address natively is declined in check_support(), never copied."""
    DSA, q, k, aux, topk_indices, _, out = _sparse_case(score_type)
    b, s_q, topk = topk_indices.shape
    out_t = torch.empty(b, topk, s_q, dtype=torch.float32, device="cuda").transpose(1, 2)  # (b, s_q, topk) view, non-contiguous
    with pytest.raises(NotImplementedError, match="out must be contiguous"):
        _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out_t).check_support()
    idx_t = torch.empty(b, topk, s_q, dtype=torch.int32, device="cuda").transpose(1, 2)
    with pytest.raises(NotImplementedError, match="topk_indices must be contiguous"):
        _sparse_plan(DSA, score_type, q, k, aux, idx_t, out).check_support()
    q_strided = torch.empty(*q.shape[:-1], 2 * q.shape[-1], dtype=q.dtype, device="cuda")[..., ::2]  # innermost stride 2
    with pytest.raises(NotImplementedError, match="Q must have a unit innermost stride"):
        _sparse_plan(DSA, score_type, q_strided, k, aux, topk_indices, out).check_support()
    assert _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_sparse_score_recompute_execute_requires_outputs(score_type, compile_allocates_nothing):
    """out is a required execute argument, re-validated live (Rule 1): missing raises, a strided view raises."""
    DSA, q, k, aux, topk_indices, _, out = _sparse_case(score_type)
    plan = _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    with pytest.raises(TypeError):
        plan.execute(q, k, aux, topk_indices)
    b, s_q, topk = topk_indices.shape
    out_t = torch.empty(b, topk, s_q, dtype=torch.float32, device="cuda").transpose(1, 2)
    with pytest.raises(ValueError, match="out must be contiguous"):
        plan.execute(q, k, aux, topk_indices, out_t)
    with pytest.raises(ValueError, match="topk_length was not declared"):
        plan.execute(q, k, aux, topk_indices, out, topk_length=torch.ones(b, s_q, dtype=torch.int32, device="cuda"))


def _executes_allocate_nothing(plan, run, repeats=3):
    """R9: warm once, then no torch allocation and no host sync across ``repeats`` executes."""
    run()  # warm: the score backend compiles on the first execute
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(repeats):
            run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocation.all.allocated"] - before


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@pytest.mark.parametrize("has_topk_length", [False, True])
@torch_fork_set_rng(seed=7)
def test_DSA_sparse_score_recompute_execute_allocates_nothing_and_never_synchronizes(score_type, has_topk_length, compile_allocates_nothing):
    from cuda.bindings import driver as cuda

    DSA, q, k, aux, topk_indices, topk_length, out = _sparse_case(score_type, has_topk_length)
    plan = _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out, topk_length)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    delta = _executes_allocate_nothing(plan, lambda: plan.execute(q, k, aux, topk_indices, out, topk_length=topk_length, current_stream=stream))
    if torch.cuda.get_device_capability()[0] == 9 and delta != 0:
        pytest.xfail("SM90 keeps the per-head (B,S,H)->(B,H,S) transpose copy (Rule 8 batch item D4, deferred)")
    assert delta == 0, f"{type(plan).__name__}.execute() made {delta} torch allocation(s) across 3 warm executes"
    check_ref_sparse_score_recompute(
        score_type,
        q,
        k if score_type == "indexer" else aux,
        topk_indices,
        out,
        aux=aux if score_type == "indexer" else k,
        softmax_scale=q.shape[-1] ** -0.5,
        topk_length=topk_length,
    )


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=8)
def test_DSA_sparse_score_recompute_sm100_topk_length_none_compiles_out(score_type, monkeypatch):
    """R3: without topk_length the SM100 mTopkLength slot is None at compile and launch (no (1,1) placeholder), keyed have_topk_length=False."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100 sparse score kernel")
    from cuda.bindings import driver as cuda
    from cudnn.deepseek_sparse_attention.score_recompute import _interface_sm100

    fn = _interface_sm100._sparse_indexer_score_recompute if score_type == "indexer" else _interface_sm100._sparse_attn_score_recompute
    monkeypatch.setattr(fn, "compile_cache", {})
    DSA, q, k, aux, topk_indices, _, out = _sparse_case(score_type)
    plan = _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out)
    assert plan.check_support()
    plan.compile()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    delta = _executes_allocate_nothing(plan, lambda: plan.execute(q, k, aux, topk_indices, out, current_stream=stream))
    assert delta == 0, f"a dead topk_length slot must be compiled out, not allocated per execute ({delta} allocations)"
    (key,) = fn.compile_cache.keys()
    assert key[-3] is False, f"compile key must carry have_topk_length=False: {key}"
