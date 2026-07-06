import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_score_recompute_params
from fe_api.dsa.dsa_reference import (
    _batched_ratio_causal_mask,
    _ratio_causal_mask,
    check_ref_dense_score_recompute,
)


@pytest.mark.L0
def test_DSA_ratio_causal_mask_offsets_reference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_default = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=torch.bool,
        device=device,
    )
    expected_cp = torch.tensor(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=torch.bool,
        device=device,
    )
    expected_batched = torch.stack(
        [
            expected_default[:5],
            expected_cp,
            torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                dtype=torch.bool,
                device=device,
            ),
        ]
    )

    assert torch.equal(_ratio_causal_mask(10, 3, 4, device), expected_default)
    assert torch.equal(_ratio_causal_mask(5, 3, 4, device, q_causal_offset=4), expected_cp)
    offsets = torch.tensor([0, 4, 9], dtype=torch.int32, device=device)
    assert torch.equal(_batched_ratio_causal_mask(5, 3, 4, device, 3, offsets), expected_batched)


def _allocate(cfg, score_type: str):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    device = "cuda"

    q = torch.randn(b, s_q, h_kv * qhpkv, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    if score_type == "indexer":
        weights = torch.randn(b, s_q, h_kv * qhpkv, dtype=torch.bfloat16, device=device)
        return q, k, weights
    lse = torch.randn(b, s_q, h_kv * qhpkv, dtype=torch.float32, device=device)
    return q, k, lse


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_score_recompute_params
def test_DSA_dense_score_recompute_wrapper(
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
        s_kv_default=1024,
    )
    q, k, aux = _allocate(cfg, score_type)
    q_causal_offsets = torch.full((cfg["b"],), 8, dtype=torch.int32, device=q.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        if score_type == "indexer":
            result = DSA.dense_indexer_score_recompute_wrapper(
                q,
                k,
                aux,
                qhead_per_kv_head=qhead_per_kv_head,
                q_causal_offsets=q_causal_offsets,
                stream=stream,
            )
        else:
            softmax_scale = 1.0 / math.sqrt(head_dim)
            result = DSA.dense_attn_score_recompute_wrapper(
                q,
                k,
                aux,
                softmax_scale,
                qhead_per_kv_head=qhead_per_kv_head,
                q_causal_offsets=q_causal_offsets,
                stream=stream,
            )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    out = result["out"]
    denom = result["denom"]

    assert out.shape == (cfg["b"], cfg["s_q"], cfg["s_kv"])
    assert denom.shape == (cfg["b"], cfg["s_q"])
    assert torch.isfinite(out).any()
    assert (torch.isfinite(out) | torch.isneginf(out)).all()
    assert torch.isfinite(denom).all()

    if not cfg["skip_ref"]:
        if score_type == "indexer":
            check_ref_dense_score_recompute(
                "indexer",
                q,
                k,
                aux,
                out,
                denom,
                q_causal_offsets=q_causal_offsets,
            )
        else:
            check_ref_dense_score_recompute(
                "attention",
                q,
                k,
                aux,
                out,
                denom,
                softmax_scale=softmax_scale,
                q_causal_offsets=q_causal_offsets,
            )
