# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import (
    _require_sm90,
    dsa_init,
    expand_mxfp8_scale,
    make_random_mxfp8_scale,
    pack_mxfp8_scales_thd,
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
        "deterministic",
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

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        q_causal_offsets=q_causal_offsets,
        return_lse=True,
    )
    scores = result["scores"]
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
    scores_ref = ref_indexer_forward(
        q,
        k,
        w,
        ratio,
        q_causal_offsets=q_causal_offsets,
    )
    torch.testing.assert_close(
        result["lse"],
        torch.logsumexp(scores_ref, dim=-1),
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
@pytest.mark.parametrize(
    "qhead_per_kv_head,s_q_default,ratio",
    [
        pytest.param(64, 128, 4, id="qh64"),
        pytest.param(32, 5, 4, id="qh32_query_tail"),
        pytest.param(32, 5, 1, id="qh32_ratio1_query_tail"),
    ],
)
def test_DSA_indexer_forward_wrapper_mxfp8_matches_dequant_reference(
    request,
    qhead_per_kv_head,
    s_q_default,
    ratio,
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
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        min_compute_capability=100,
        b_default=1,
        s_q_default=s_q_default,
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

    result = DSA.indexer_forward_wrapper(
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
        return_lse=True,
    )
    scores = result["scores"]
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
        scores_ref = ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
        ).squeeze(0)
        torch.testing.assert_close(
            result["lse"][q0:q1],
            torch.logsumexp(scores_ref, dim=-1),
            atol=5e-3,
            rtol=5e-3,
        )
        if s_k < max_seqlen_k:
            assert bool(torch.isneginf(scores[q0:q1, s_k:]).all())


@pytest.mark.L0
def test_DSA_indexer_forward_wrapper_mxfp8_requires_mqa():
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 indexer forward requires SM100+")

    scale_utils = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils")
    device = torch.device("cuda")
    b, s_q, s_k, h_q, h_kv, d = 1, 128, 32, 128, 2, 128
    qhead_per_kv_head = h_q // h_kv
    q = torch.zeros((b, s_q, h_q, d), dtype=torch.float8_e4m3fn, device=device)
    k = torch.zeros((b, s_k, h_kv, d), dtype=torch.float8_e4m3fn, device=device)
    w = torch.ones((b, s_q, h_q), dtype=torch.bfloat16, device=device)
    q_scale_logical = torch.ones((b, s_q, h_q, d // 32), device=device).to(torch.float8_e8m0fnu)
    k_scale_logical = torch.ones((b, s_k, h_kv, d // 32), device=device).to(torch.float8_e8m0fnu)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    common = {
        "qhead_per_kv_head": qhead_per_kv_head,
        "precision": "mxfp8",
        "q_scale": q_scale,
        "k_scale": k_scale,
    }

    with pytest.raises(ValueError, match="requires n_heads_kv=1"):
        DSA.indexer_forward_wrapper(q, k, w, **common)

    cu_q = torch.tensor([0, s_q], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, s_k], dtype=torch.int32, device=device)
    cu_q_scale = scale_utils.make_scale_cu_seqlens_padded(cu_q, token_alignment=2)
    cu_k_scale = scale_utils.make_scale_cu_seqlens_padded(cu_k, token_alignment=128)
    with pytest.raises(ValueError, match="requires n_heads_kv=1"):
        DSA.indexer_forward_wrapper(
            q[0],
            k[0],
            w[0],
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=s_q,
            max_seqlen_k=s_k,
            cu_seqlens_q_scale_padded=cu_q_scale,
            cu_seqlens_k_scale_padded=cu_k_scale,
            **common,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
@pytest.mark.parametrize("h_q", [64, 32], ids=["qh64", "qh32"])
def test_DSA_indexer_forward_wrapper_thd_mxfp8_compact_padded_scales(h_q):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("THD MXFP8 indexer forward requires SM100+")

    device = torch.device("cuda")
    shapes = [(127, 32), (129, 64)]
    ratio, h_kv, d = 4, 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    max_q, max_k = max(q_lengths), max(k_lengths)

    q_ref = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale(
        (total_q, h_q, d // 32),
        device=device,
        seed=107,
        exponent_min=-2,
        exponent_max=3,
    )
    k_scale_logical = make_random_mxfp8_scale(
        (total_k, h_kv, d // 32),
        device=device,
        seed=109,
        exponent_min=-2,
        exponent_max=3,
    )
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale, k_scale, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(
        q_scale_logical,
        k_scale_logical,
        cu_q,
        cu_k,
        h_q // h_kv,
        q_alignment=256,
        k_alignment=256,
    )
    expected_scale_prefix = torch.tensor(
        [0, 256, 512],
        dtype=torch.int32,
        device=device,
    )
    assert torch.equal(cu_q_scale, expected_scale_prefix)
    assert torch.equal(cu_k_scale, expected_scale_prefix)
    q_causal_offsets = torch.tensor([1, 17], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q // h_kv,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    assert result["scores"].shape == (total_q, max_k)
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        check_ref_indexer_forward(
            q_deq[q0:q1].unsqueeze(0),
            k_deq[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            result["scores"][q0:q1, :s_k].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
            atol=5e-3,
            rtol=5e-3,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=12)
@pytest.mark.parametrize("qhead_per_kv_head", [32, 64])
def test_DSA_indexer_forward_wrapper_fp8_sm90_matches_dequant_reference(request, qhead_per_kv_head):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    cfg = dsa_init(
        request=request,
        head_dim=128,
        qhead_per_kv_head=qhead_per_kv_head,
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


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_DSA_indexer_forward_wrapper_fp8_sm90_thd_matches_dequant_reference():
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    shapes = [(5, 67), (7, 70)]
    ratio, h_q, h_kv, d = 4, 32, 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])

    q_ref = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q, q_scale = quantize_fp8_1x128(q_ref)
    k, k_scale = quantize_fp8_1x128(k_ref)
    q_deq = q.float() * q_scale.unsqueeze(-1)
    k_deq = k.float() * k_scale.unsqueeze(-1)
    q_causal_offsets = torch.tensor(
        [s_k * ratio - s_q for s_q, s_k in shapes],
        dtype=torch.int32,
        device=device,
    )

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        q_causal_offsets=q_causal_offsets,
        precision="fp8",
        q_scale=q_scale,
        k_scale=k_scale,
        return_lse=True,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    assert result["scores"].shape == (total_q, max(k_lengths))
    assert result["lse"].shape == (total_q,)
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        scores_ref = ref_indexer_forward(
            q_deq[q0:q1].unsqueeze(0),
            k_deq[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
        ).squeeze(0)
        torch.testing.assert_close(
            result["scores"][q0:q1, :s_k],
            scores_ref,
            atol=2e-3,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            result["lse"][q0:q1],
            torch.logsumexp(scores_ref, dim=-1),
            atol=5e-3,
            rtol=5e-3,
        )
        if s_k < max(k_lengths):
            assert bool(torch.isneginf(result["scores"][q0:q1, s_k:]).all())
