import torch

import pytest

from test_utils import torch_fork_set_rng

from fe_api.nsa.nsa_utils import (
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
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )

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

    Q, K, _, LSE, _, _, cum_seqlen_q, cum_seqlen_kv, max_s_q, max_s_kv = (
        allocate_input_tensors(cfg)
    )
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
    topk_reduction.compile(current_stream=stream)
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
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )

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

    Q, K, _, LSE, _, _, cum_seqlen_q, cum_seqlen_kv, max_s_q, max_s_kv = (
        allocate_input_tensors(cfg)
    )
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
