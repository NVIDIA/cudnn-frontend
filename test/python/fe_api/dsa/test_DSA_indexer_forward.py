# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import (
    _require_sm90,
    dsa_init,
    with_dsa_indexer_forward_params,
)
from fe_api.dsa.dsa_reference import check_ref_indexer_forward


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
