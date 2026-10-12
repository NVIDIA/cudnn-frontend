# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest

from test_utils import torch_fork_set_rng

from native_sparse_attention.cutedsl.nsa_utils import (
    with_nsa_topk_reduction_params,
    nsa_init,
    allocate_input_tensors,
    allocate_output_tensors,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_topk_reduction_params
def test_nsa_topk_reduction_compile_execute(
    layout,
    dtype,
    acc_dtype,
    selection_block_size,
    compress_stride,
    k_value,
    is_causal,
    mma_tiler_mn,
    request,
):
    try:
        from cudnn import NSA
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        selection_block_size=selection_block_size,
        compress_stride=compress_stride,
        k_value=k_value,
        is_causal=is_causal,
        mma_tiler_mn=mma_tiler_mn,
        s_q_default_override=4096,
        s_kv_default_override=128,
    )

    Q, K, _, LSE, _, _, cum_seqlen_q, cum_seqlen_kv, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    _, _, _, topk_scores, topk_indices = allocate_output_tensors(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    topk_reduction = NSA.TopKReduction(
        sample_q=Q,
        sample_k=K,
        sample_lse=LSE,
        sample_topk_scores=topk_scores,
        sample_topk_indices=topk_indices,
        sample_cum_seqlen_q=cum_seqlen_q,
        sample_cum_seqlen_k=cum_seqlen_kv,
        max_s_q=max_s_q,
        max_s_k=max_s_kv,
        acc_dtype=cfg["acc_dtype"],
        k_value=cfg["k_value"],
        selection_block_size=cfg["selection_block_size"],
        compress_stride=cfg["compress_stride"],
        is_causal=cfg["is_causal"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        scale_softmax=None,
    )
    assert topk_reduction.check_support()
    topk_reduction.compile()
    topk_reduction.execute(
        q_tensor=Q,
        k_tensor=K,
        lse_tensor=LSE,
        topk_scores_tensor=topk_scores,
        topk_indices_tensor=topk_indices,
        cumulative_s_q_tensor=cum_seqlen_q,
        cumulative_s_k_tensor=cum_seqlen_kv,
        current_stream=stream,
    )

    print("No reference check for Top-K Reduction")


@pytest.mark.L0
@pytest.mark.parametrize("is_causal", [True, False])
@torch_fork_set_rng(seed=0)
def test_nsa_topk_reduction_thd_ignores_keys_past_the_sequence(is_causal):
    """A sequence's top-k must not depend on the next sequence's compressed keys (T,H,D).

    There is no reference for this kernel, so this is a noninterference check: perturb the first compressed
    key of sequence 2 and require sequence 1's scores and indices to stay bit-identical. With
    s_k < s_q / compress_stride - 2 * selection_block_size / compress_stride, the selection block that holds
    key index s_k (one past sequence 1) is still a top-k candidate for sequence 1's later queries, and
    s_k % 128 != 0 puts that key in the same 128-column key tile as sequence 1's last keys.
    """
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

    s_q, s_k, h_q, h_k, d = 4096, 64, 4, 1, 128
    selection_block_size, compress_stride, k_value = 64, 32, 16
    assert s_k < s_q // compress_stride - 2 * selection_block_size // compress_stride and s_k % 128 != 0

    q = torch.randn(2 * s_q, h_q, d, dtype=torch.float16).cuda()
    k = torch.randn(2 * s_k, h_k, d, dtype=torch.float16).cuda()
    lse = (-1.0 * torch.randn(1, h_q, 2 * s_q, dtype=torch.float32)).transpose(0, 2).cuda()  # (T, H_q, 1), T at unit stride
    cum_seqlen_q = torch.tensor([0, s_q, 2 * s_q], dtype=torch.int32).cuda()
    cum_seqlen_k = torch.tensor([0, s_k, 2 * s_k], dtype=torch.int32).cuda()

    def sequence_1_topk(k_tensor):
        result = NSA.topk_reduction_wrapper(
            q_tensor=q,
            k_tensor=k_tensor,
            lse_tensor=lse,
            cum_seqlen_q_tensor=cum_seqlen_q,
            cum_seqlen_k_tensor=cum_seqlen_k,
            max_s_q=s_q,
            max_s_k=s_k,
            k_value=k_value,
            selection_block_size=selection_block_size,
            compress_stride=compress_stride,
            is_causal=is_causal,
        )
        return result["topk_scores_tensor"][:s_q].clone(), result["topk_indices_tensor"][:s_q].clone()

    scores, indices = sequence_1_topk(k)
    # The last query has more than k_value candidate blocks with keys: every slot holds a real block.
    assert (indices[-1] > 0).all() and torch.isfinite(scores[-1]).all()

    k_perturbed = k.clone()
    k_perturbed[s_k] *= 4  # the first compressed key of sequence 2
    scores_perturbed, indices_perturbed = sequence_1_topk(k_perturbed)

    torch.testing.assert_close(indices_perturbed, indices, rtol=0, atol=0)
    torch.testing.assert_close(scores_perturbed, scores, rtol=0, atol=0, equal_nan=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_topk_reduction_params
def test_nsa_topk_reduction_wrapper(
    layout,
    dtype,
    acc_dtype,
    selection_block_size,
    compress_stride,
    k_value,
    is_causal,
    mma_tiler_mn,
    request,
):
    try:
        from cudnn import NSA
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        selection_block_size=selection_block_size,
        compress_stride=compress_stride,
        k_value=k_value,
        is_causal=is_causal,
        mma_tiler_mn=mma_tiler_mn,
        s_q_default_override=4096,
        s_kv_default_override=128,
    )

    Q, K, _, LSE, _, _, cum_seqlen_q, cum_seqlen_kv, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    topk_scores, topk_indices = NSA.topk_reduction_wrapper(
        q_tensor=Q,
        k_tensor=K,
        lse_tensor=LSE,
        cum_seqlen_q_tensor=cum_seqlen_q,
        cum_seqlen_k_tensor=cum_seqlen_kv,
        max_s_q=max_s_q,
        max_s_k=max_s_kv,
        acc_dtype=cfg["acc_dtype"],
        k_value=cfg["k_value"],
        selection_block_size=cfg["selection_block_size"],
        compress_stride=cfg["compress_stride"],
        is_causal=cfg["is_causal"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        scale_softmax=None,
        current_stream=stream,
    )

    print("No reference check for Top-K Reduction")
