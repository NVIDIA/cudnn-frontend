import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import (
    dsa_init,
    expand_mxfp8_scale,
    make_random_mxfp8_scale,
    quantize_mxfp8,
    with_dsa_score_recompute_params,
)
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


def _allocate_mxfp8_qk(cfg, scale_utils):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv
    sf_groups = (d + 31) // 32
    device = "cuda"

    q_ref = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    q_scale_logical = make_random_mxfp8_scale(
        (b, s_q, h_q, sf_groups),
        device=device,
        seed=201,
        exponent_min=-2,
        exponent_max=3,
    )
    k_scale_logical = make_random_mxfp8_scale(
        (b, s_k, h_kv, sf_groups),
        device=device,
        seed=203,
        exponent_min=-2,
        exponent_max=3,
    )
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=qhpkv)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    return q, k, q_deq, k_deq, q_scale, k_scale


def _dense_attn_lse(q, k, softmax_scale):
    h_q = q.shape[2]
    h_kv = k.shape[2]
    qhpkv = h_q // h_kv
    k_exp = k.repeat_interleave(qhpkv, dim=2)
    qk = torch.einsum("bqhd,bkhd->bqhk", q.float(), k_exp.float()) * softmax_scale
    return torch.logsumexp(qk, dim=-1)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_dense_score_recompute_thd_q_causal_offsets_alignment():
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This regression is specific to the SM90 THD adapter")

    batch, seqlen_q, seqlen_k, heads, head_dim = 2, 256, 1024, 32, 128
    # Use a non-default stream so this test exercises only the offset contract.
    torch_stream = torch.cuda.Stream()
    with torch.cuda.stream(torch_stream):
        q = torch.randn(batch, seqlen_q, heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen_k, 1, head_dim, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(batch, seqlen_q, heads, dtype=torch.bfloat16, device="cuda")
        cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.arange(0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device="cuda")
        q_causal_offsets = cu_seqlens_q[:-1]

        assert q_causal_offsets[1:].data_ptr() % 4 == 0
        assert q_causal_offsets[1:].data_ptr() % 16 != 0

        result = DSA.dense_indexer_score_recompute_wrapper(
            q.flatten(0, 1),
            k.flatten(0, 1),
            weights.flatten(0, 1),
            qhead_per_kv_head=heads,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            q_causal_offsets=q_causal_offsets,
            stream=cuda.CUstream(torch_stream.cuda_stream),
        )

    torch_stream.synchronize()
    assert result["out"].shape == (batch * seqlen_q, seqlen_k)
    assert result["denom"].shape == (batch * seqlen_q,)
    assert torch.isfinite(result["out"]).any()
    assert (torch.isfinite(result["out"]) | torch.isneginf(result["out"])).all()
    assert torch.isfinite(result["denom"]).all()


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


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=13)
def test_DSA_dense_score_recompute_wrapper_mxfp8_matches_dequant_reference(
    score_type,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    scale_utils = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils")
    cfg = dsa_init(
        request=request,
        head_dim=128,
        qhead_per_kv_head=64,
        score_type=score_type,
        min_compute_capability=100,
        b_default=1,
        s_q_default=128,
        s_kv_default=128,
    )
    ratio = 4
    q, k, q_deq, k_deq, q_scale, k_scale = _allocate_mxfp8_qk(cfg, scale_utils)
    q_causal_offsets = torch.full((cfg["b"],), 16, dtype=torch.int32, device=q.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if score_type == "indexer":
        aux = (
            torch.randn(
                cfg["b"],
                cfg["s_q"],
                cfg["h_kv"] * cfg["qhead_per_kv_head"],
                dtype=torch.bfloat16,
                device="cuda",
            ).abs()
            * 0.1
        )
        result = DSA.dense_indexer_score_recompute_wrapper(
            q,
            k,
            aux,
            qhead_per_kv_head=cfg["qhead_per_kv_head"],
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            precision="mxfp8",
            q_scale=q_scale,
            k_scale=k_scale,
            stream=stream,
        )
        check_ref_dense_score_recompute(
            "indexer",
            q_deq,
            k_deq,
            aux,
            result["out"],
            result["denom"],
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            atol_scores=5e-3,
            rtol_scores=5e-3,
            atol_denom=5e-3,
            rtol_denom=5e-3,
        )
    else:
        softmax_scale = 1.0 / math.sqrt(cfg["head_dim"])
        aux = _dense_attn_lse(q_deq, k_deq, softmax_scale)
        result = DSA.dense_attn_score_recompute_wrapper(
            q,
            k,
            aux,
            softmax_scale,
            qhead_per_kv_head=cfg["qhead_per_kv_head"],
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            precision="mxfp8",
            q_scale=q_scale,
            k_scale=k_scale,
            stream=stream,
        )
        check_ref_dense_score_recompute(
            "attention",
            q_deq,
            k_deq,
            aux,
            result["out"],
            result["denom"],
            softmax_scale=softmax_scale,
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            atol_scores=5e-3,
            rtol_scores=5e-3,
            atol_denom=5e-3,
            rtol_denom=5e-3,
        )
