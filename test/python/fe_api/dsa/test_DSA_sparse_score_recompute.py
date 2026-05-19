import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_score_recompute_params
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

    if score_type == "indexer":
        weights = torch.randn(b, s_q, qhpkv, dtype=torch.bfloat16, device=device)
        return q, k, weights, topk_indices, topk_length
    else:
        lse = torch.randn(b, s_q, qhpkv, dtype=torch.float32, device=device)
        return q, k, lse, topk_indices, topk_length


def _local_to_global_topk_indices(topk_indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    batch_offsets = torch.arange(topk_indices.shape[0], device=topk_indices.device, dtype=torch.int32).view(-1, 1, 1).mul_(int(seqlen_k))
    return torch.where(topk_indices >= 0, topk_indices + batch_offsets, topk_indices)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_score_recompute_params
def test_DSA_sparse_score_recompute_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    score_type,
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
        min_compute_capability=90,
        s_q_default=256,
        s_kv_default=2048,
    )
    q, k, aux, topk_indices, topk_length = _allocate(cfg, score_type, has_topk_length=False)
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
def test_DSA_sparse_score_recompute_wrapper_batch_gt_one(score_type, request):
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
        min_compute_capability=90,
        b_default=2,
        s_q_default=32,
        s_kv_default=256,
    )
    q, k, aux, topk_indices, topk_length = _allocate(
        cfg,
        score_type,
        has_topk_length=False,
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
