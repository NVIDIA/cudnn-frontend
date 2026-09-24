# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    # Rule 8 holds BUILD to the execute standard: no device read (the T,H,D envelope is
    # a required host int) and no hidden copy at init / check_support / compile / execute.
    torch.cuda.set_sync_debug_mode("error")
    try:
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
    finally:
        torch.cuda.set_sync_debug_mode("default")

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


def _thd_topk_case(request):
    """One T,H,D Top-K configuration (the L0 shape) for the contract tests."""
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = nsa_init(
        request=request,
        layout="thd",
        dtype=torch.float16,
        acc_dtype=torch.float32,
        selection_block_size=64,
        compress_stride=32,
        k_value=16,
        is_causal=True,
        mma_tiler_mn=(128, 128),
        s_q_default_override=4096,
        s_kv_default_override=128,
    )
    Q, K, _, LSE, _, _, cum_seqlen_q, cum_seqlen_kv, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    _, _, _, topk_scores, topk_indices = allocate_output_tensors(cfg)
    kwargs = dict(
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
    )
    tensors = dict(Q=Q, K=K, LSE=LSE, topk_scores=topk_scores, topk_indices=topk_indices, cum_seqlen_q=cum_seqlen_q, cum_seqlen_kv=cum_seqlen_kv)
    return NSA, cfg, kwargs, tensors


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_topk_reduction_thd_requires_max_s(request):
    """Rule 8: the T,H,D launch envelope is a required host int, never read back from cum_seqlen at build."""
    NSA, _, kwargs, tensors = _thd_topk_case(request)
    for missing in ("max_s_q", "max_s_k"):
        bad = dict(kwargs, **{missing: None})
        with pytest.raises(ValueError, match="max_s_q and max_s_k are required"):
            NSA.TopKReduction(**bad)


def _wrapper_kwargs(kwargs, tensors):
    return dict(
        q_tensor=tensors["Q"],
        k_tensor=tensors["K"],
        lse_tensor=tensors["LSE"],
        cum_seqlen_q_tensor=tensors["cum_seqlen_q"],
        cum_seqlen_k_tensor=tensors["cum_seqlen_kv"],
        max_s_q=kwargs["max_s_q"],
        max_s_k=kwargs["max_s_k"],
        acc_dtype=kwargs["acc_dtype"],
        k_value=kwargs["k_value"],
        selection_block_size=kwargs["selection_block_size"],
        compress_stride=kwargs["compress_stride"],
        is_causal=kwargs["is_causal"],
        mma_tiler_mn=kwargs["mma_tiler_mn"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_topk_reduction_wrapper_infers_max_s(request):
    """The eager wrapper keeps develop's surface: the T,H,D envelope is inferred from cum_seqlen when omitted."""
    NSA, _, kwargs, tensors = _thd_topk_case(request)
    ref = NSA.topk_reduction_wrapper(**_wrapper_kwargs(kwargs, tensors))
    got = NSA.topk_reduction_wrapper(**dict(_wrapper_kwargs(kwargs, tensors), max_s_q=None, max_s_k=None))
    torch.testing.assert_close(got["topk_indices_tensor"], ref["topk_indices_tensor"])
    torch.testing.assert_close(got["topk_scores_tensor"], ref["topk_scores_tensor"], equal_nan=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_topk_reduction_wrapper_accepts_row_major_lse(request):
    """The eager wrapper stages a row-major (T, H_q[, 1]) LSE into the layout the class accepts; results match."""
    NSA, _, kwargs, tensors = _thd_topk_case(request)
    ref = NSA.topk_reduction_wrapper(**_wrapper_kwargs(kwargs, tensors))
    for row_major in (tensors["LSE"].contiguous(), tensors["LSE"].squeeze(-1).contiguous()):
        assert row_major.is_contiguous() and row_major.stride(0) != 1
        got = NSA.topk_reduction_wrapper(**dict(_wrapper_kwargs(kwargs, tensors), lse_tensor=row_major))
        torch.testing.assert_close(got["topk_indices_tensor"], ref["topk_indices_tensor"])
        torch.testing.assert_close(got["topk_scores_tensor"], ref["topk_scores_tensor"], equal_nan=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_topk_reduction_wrapper_lse_staging_outlives_the_released_original(request):
    """R1 staging: the row-major LSE copy on an explicit side stream reads the caller's tensor
    asynchronously; the wrapper records the original there, so releasing it right after the call
    and reusing its block (poisoned here) on the caller's stream cannot corrupt the copy."""
    from cuda.bindings import driver as cuda

    NSA, _, kwargs, tensors = _thd_topk_case(request)
    ref = NSA.topk_reduction_wrapper(**_wrapper_kwargs(kwargs, tensors))
    row_major = tensors["LSE"].contiguous()
    assert row_major.stride(0) != 1
    # Warm the window's kernels first: under CUDA lazy loading a first launch waits for the device to drain (no race).
    NSA.topk_reduction_wrapper(**dict(_wrapper_kwargs(kwargs, tensors), lse_tensor=row_major))
    torch.empty(1, dtype=row_major.dtype, device="cuda").fill_(float("nan"))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # no other cached block of this size: the released one is the only candidate for reuse
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(1_000_000_000)  # the wrapper's staging copy queues behind this
    got = NSA.topk_reduction_wrapper(**dict(_wrapper_kwargs(kwargs, tensors), lse_tensor=row_major, current_stream=cuda.CUstream(side.cuda_stream)))
    shape, dtype = row_major.shape, row_major.dtype
    del row_major  # the caller releases its reference while the copy is still pending
    poison = torch.empty(shape, dtype=dtype, device="cuda")
    # Same size as the released block: with the original recorded, the allocator defers the block's reuse
    # (or waits for the pending copy first) instead of letting this fill race it.
    poison.fill_(float("nan"))
    torch.cuda.synchronize()
    torch.testing.assert_close(got["topk_indices_tensor"], ref["topk_indices_tensor"])
    torch.testing.assert_close(got["topk_scores_tensor"], ref["topk_scores_tensor"], equal_nan=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_topk_reduction_declines_row_major_lse(request):
    """R5: a natural row-major (T, H_q) LSE is declined in check_support (the kernel hard-codes a (1, S_q) stride)."""
    NSA, _, kwargs, tensors = _thd_topk_case(request)
    t, h_q = tensors["LSE"].shape[0], tensors["LSE"].shape[1]
    row_major_lse = torch.empty(t, h_q, dtype=torch.float32, device=tensors["LSE"].device)
    topk = NSA.TopKReduction(**dict(kwargs, sample_lse=row_major_lse))
    with pytest.raises(NotImplementedError, match="sample_lse"):
        topk.check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_topk_reduction_execute_rejects_lse_layout_mismatch(request):
    """Rule 1: a live LSE whose layout differs from the plan's sample is rejected, never made contiguous."""
    NSA, _, kwargs, tensors = _thd_topk_case(request)
    topk = NSA.TopKReduction(**kwargs)
    assert topk.check_support()
    topk.compile()
    t, h_q = tensors["LSE"].shape[0], tensors["LSE"].shape[1]
    row_major_lse = torch.empty(t, h_q, 1, dtype=torch.float32, device=tensors["LSE"].device)
    with pytest.raises(ValueError, match="lse_tensor layout differs"):
        topk.execute(
            q_tensor=tensors["Q"],
            k_tensor=tensors["K"],
            lse_tensor=row_major_lse,
            topk_scores_tensor=tensors["topk_scores"],
            topk_indices_tensor=tensors["topk_indices"],
            cumulative_s_q_tensor=tensors["cum_seqlen_q"],
            cumulative_s_k_tensor=tensors["cum_seqlen_kv"],
        )
