import inspect

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import (
    _require_sm90,
    dsa_init,
    expand_mxfp8_scale,
    make_random_mxfp8_scale,
    quantize_fp8_1x128,
    quantize_mxfp8,
    with_dsa_indexer_forward_params,
)
from fe_api.dsa.dsa_reference import (
    check_ref_compressed_topk,
    check_ref_indexer_forward,
    ref_indexer_forward,
)


@pytest.mark.L0
def test_DSA_indexer_forward_api_is_split_from_top_k():
    try:
        from cudnn import DSA
        import cudnn.deepseek_sparse_attention as dsa
        import cudnn.deepseek_sparse_attention.indexer_forward as indexer_forward
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    dense_parameters = inspect.signature(DSA.indexer_forward_wrapper).parameters
    fused_parameters = inspect.signature(DSA.indexer_forward_top_k_wrapper).parameters
    compressed_only = {
        "top_k",
        "return_softmax",
        "softmax_out",
        "microbatch_rows",
        "topk_indices_global",
        "cand_buffer",
        "out_indices",
        "out_logits",
        "cand_batch_offsets",
    }

    assert compressed_only.isdisjoint(dense_parameters)
    assert "is_compressed_logits" not in dense_parameters
    assert "is_compressed_logits" not in fused_parameters
    assert tuple(fused_parameters)[:4] == ("q", "k", "w", "top_k")
    assert compressed_only <= fused_parameters.keys()
    assert "indexer_forward_top_k_wrapper" in dsa.__all__
    assert "indexer_forward_top_k_wrapper" in indexer_forward.__all__
    assert DSA.indexer_forward_top_k_wrapper is indexer_forward.indexer_forward_top_k_wrapper


def _allocate_inputs(cfg):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv

    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device="cuda")
    return q, k, w


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_forward_params
def test_DSA_indexer_forward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    ratio,
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
        ratio=ratio,
        s_q_default=256,
        s_kv_default=512,
        min_compute_capability=90,
    )
    q, k, w = _allocate_inputs(cfg)
    q_causal_offsets = torch.full((cfg["b"],), 4, dtype=torch.int32, device=q.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        result = DSA.indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            q_causal_offsets=q_causal_offsets,
            stream=stream,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    scores = result["scores"]
    if not cfg["skip_ref"]:
        check_ref_indexer_forward(q, k, w, scores, ratio, q_causal_offsets=q_causal_offsets)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
def test_DSA_indexer_forward_wrapper_qh16_causal_block_boundary():
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    b, s_q, s_k, h_q, h_kv, d = 1, 8, 128, 16, 1, 128
    ratio = 4
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)
    q_causal_offsets = torch.tensor([252], dtype=torch.int32, device=device)

    scores = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        q_causal_offsets=q_causal_offsets,
    )["scores"]
    torch.cuda.synchronize()

    # The causal limit moves from 63 to 64 inside the CTA. In particular,
    # k=63 is masked for q=0..2, then becomes valid at q=3.
    assert bool(torch.isneginf(scores[0, :3, 63]).all())
    assert bool(torch.isfinite(scores[0, 3:, 63]).all())
    assert bool(torch.isneginf(scores[0, :7, 64]).all())
    assert bool(torch.isfinite(scores[0, 7, 64]))
    check_ref_indexer_forward(
        q,
        k,
        w,
        scores,
        ratio,
        q_causal_offsets=q_causal_offsets,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
def test_DSA_indexer_forward_wrapper_mxfp8_matches_dequant_reference(request):
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
        ratio=4,
        min_compute_capability=100,
        b_default=1,
        s_q_default=128,
        s_kv_default=128,
    )

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
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale(
        (b, s_q, h_q, sf_groups),
        device=device,
        seed=101,
        exponent_min=-2,
        exponent_max=3,
    )
    k_scale_logical = make_random_mxfp8_scale(
        (b, s_k, h_kv, sf_groups),
        device=device,
        seed=103,
        exponent_min=-2,
        exponent_max=3,
    )
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=qhpkv)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    q_causal_offsets = torch.full((b,), 16, dtype=torch.int32, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=cfg["ratio"],
        qhead_per_kv_head=qhpkv,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        stream=stream,
    )
    torch.cuda.synchronize()

    check_ref_indexer_forward(
        q_deq,
        k_deq,
        w,
        result["scores"],
        cfg["ratio"],
        q_causal_offsets=q_causal_offsets,
        atol=5e-3,
        rtol=5e-3,
    )

    compressed = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=min(32, s_k),
        ratio=cfg["ratio"],
        qhead_per_kv_head=qhpkv,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        topk_indices_global=False,
        stream=stream,
    )
    torch.cuda.synchronize()
    check_ref_compressed_topk(
        result["scores"],
        compressed["indices"],
        compressed["logits"],
        min(32, s_k),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
def test_DSA_indexer_forward_wrapper_qh16_thd_varlen_tails():
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    shapes = [(7, 67), (11, 70)]
    ratio, h_q, h_kv, d = 4, 16, 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    total_q, total_k = int(cu_seqlens_q[-1]), int(cu_seqlens_k[-1])
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device)
    q_causal_offsets = torch.tensor(
        [s_k * ratio - s_q for s_q, s_k in shapes],
        dtype=torch.int32,
        device=device,
    )

    scores = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        q_causal_offsets=q_causal_offsets,
    )["scores"]
    torch.cuda.synchronize()

    cu_q_host = cu_seqlens_q.tolist()
    cu_k_host = cu_seqlens_k.tolist()
    max_seqlen_k = max(k_lengths)
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        check_ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            scores[q0:q1, :s_k].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
        )
        if s_k < max_seqlen_k:
            assert bool(torch.isneginf(scores[q0:q1, s_k:]).all())


@pytest.mark.L0
@torch_fork_set_rng(seed=12)
def test_DSA_indexer_forward_wrapper_fp8_sm90_matches_dequant_reference(request):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip("SM90 FP8 indexer forward path requires Hopper")
    cfg = dsa_init(
        request=request,
        head_dim=128,
        qhead_per_kv_head=64,
        ratio=4,
        min_compute_capability=90,
        b_default=1,
        s_q_default=128,
        s_kv_default=128,
    )

    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv
    device = "cuda"

    q_ref = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q, q_scale = quantize_fp8_1x128(q_ref)
    k, k_scale = quantize_fp8_1x128(k_ref)
    q_deq = q.float() * q_scale.unsqueeze(-1)
    k_deq = k.float() * k_scale.unsqueeze(-1)
    q_causal_offsets = torch.full((b,), 16, dtype=torch.int32, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=cfg["ratio"],
        qhead_per_kv_head=qhpkv,
        q_causal_offsets=q_causal_offsets,
        precision="fp8",
        q_scale=q_scale,
        k_scale=k_scale,
        return_lse=True,
        stream=stream,
    )
    torch.cuda.synchronize()

    scores_ref = ref_indexer_forward(
        q_deq,
        k_deq,
        w,
        cfg["ratio"],
        q_causal_offsets=q_causal_offsets,
    )
    check_ref_indexer_forward(
        q_deq,
        k_deq,
        w,
        result["scores"],
        cfg["ratio"],
        q_causal_offsets=q_causal_offsets,
        atol=2e-3,
        rtol=2e-3,
    )
    lse_ref = torch.logsumexp(scores_ref, dim=-1)
    assert torch.equal(torch.isfinite(result["lse"]), torch.isfinite(lse_ref))
    finite_lse = torch.isfinite(lse_ref)
    torch.testing.assert_close(
        result["lse"][finite_lse],
        lse_ref[finite_lse],
        atol=5e-3,
        rtol=5e-3,
    )
