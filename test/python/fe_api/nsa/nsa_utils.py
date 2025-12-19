"""
Utilities for NSA (Native Sparse Attention) tests.
"""

import torch
import pytest
from typing import Optional, Tuple
import math

# Common parameterization marks for all NSA tests
NSA_PARAM_MARKS = [
    pytest.mark.parametrize("layout", ["bshd", "thd"]),
    pytest.mark.parametrize("dtype", [torch.float16]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
]

# Parameterization marks for NSA Top-K Reduction tests
NSA_TOPK_REDUCTION_PARAM_MARKS = [
    pytest.mark.parametrize("selection_block_size", [64]),
    pytest.mark.parametrize("compress_stride", [32]),
    pytest.mark.parametrize("k_value", [16]),
    pytest.mark.parametrize("is_causal", [True]),
    pytest.mark.parametrize("mma_tiler_mn", [(128, 128)]),
]

# Parameterization marks for NSA Compression Attention tests
NSA_COMPRESSION_ATTENTION_PARAM_MARKS = [
    pytest.mark.parametrize("mma_tiler_mn", [(128, 128)]),
    pytest.mark.parametrize("is_persistent", [False]),
    pytest.mark.parametrize("scale_q", [1.0]),
    pytest.mark.parametrize("scale_k", [1.0]),
    pytest.mark.parametrize("scale_v", [1.0]),
    pytest.mark.parametrize("inv_scale_o", [1.0]),
    pytest.mark.parametrize("scale_softmax", [None]),
]

# Parameterization marks for NSA Sliding Window Attention tests
NSA_SWA_PARAM_MARKS = [
    pytest.mark.parametrize("window_size", [64, 512]),
    pytest.mark.parametrize("scale_softmax", [None]),
]

# Parameterization marks for NSA Selection Attention tests
NSA_SELECTION_ATTENTION_PARAM_MARKS = [
    pytest.mark.parametrize("topk_size", [16]),
    pytest.mark.parametrize("block_size", [64]),
]


def with_nsa_topk_reduction_params(func):
    for mark in reversed(NSA_PARAM_MARKS + NSA_TOPK_REDUCTION_PARAM_MARKS):
        func = mark(func)
    return func


def with_nsa_compression_attention_params(func):
    for mark in reversed(NSA_PARAM_MARKS + NSA_COMPRESSION_ATTENTION_PARAM_MARKS):
        func = mark(func)
    return func


def with_nsa_swa_params(func):
    for mark in reversed(NSA_PARAM_MARKS + NSA_SWA_PARAM_MARKS):
        func = mark(func)
    return func


def with_nsa_selection_attention_params(func):
    for mark in reversed(NSA_PARAM_MARKS + NSA_SELECTION_ATTENTION_PARAM_MARKS):
        func = mark(func)
    return func


def nsa_init(
    request: pytest.FixtureRequest,
    layout: str = "bshd",
    dtype: Optional[torch.dtype] = None,
    acc_dtype: Optional[torch.dtype] = None,
    selection_block_size: Optional[int] = None,
    compress_stride: Optional[int] = None,
    k_value: Optional[int] = None,
    is_causal: Optional[bool] = None,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    is_persistent: Optional[bool] = None,
    scale_q: Optional[float] = None,
    scale_k: Optional[float] = None,
    scale_v: Optional[float] = None,
    inv_scale_o: Optional[float] = None,
    scale_softmax: Optional[float] = None,
    window_size: Optional[int] = None,
    topk_size: Optional[int] = None,
    block_size: Optional[int] = None,
    s_q_default_override: Optional[int] = None,
    s_kv_default_override: Optional[int] = None,
):
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            f"Environment not supported: requires compute capability >= 10, found {major}"
        )

    b = (
        int(request.config.getoption("--nsa-b"))
        if request.config.getoption("--nsa-b") is not None
        else 2
    )
    s_q = (
        int(request.config.getoption("--nsa-s_q"))
        if request.config.getoption("--nsa-s_q") is not None
        else 1024 if s_q_default_override is None else s_q_default_override
    )
    s_kv = (
        int(request.config.getoption("--nsa-s_kv"))
        if request.config.getoption("--nsa-s_kv") is not None
        else 1024 if s_kv_default_override is None else s_kv_default_override
    )
    d_qk = (
        int(request.config.getoption("--nsa-d_qk"))
        if request.config.getoption("--nsa-d_qk") is not None
        else 128
    )
    d_v = (
        int(request.config.getoption("--nsa-d_v"))
        if request.config.getoption("--nsa-d_v") is not None
        else 128
    )
    h_q = (
        int(request.config.getoption("--nsa-h_q"))
        if request.config.getoption("--nsa-h_q") is not None
        else 4
    )
    h_k = (
        int(request.config.getoption("--nsa-h_k"))
        if request.config.getoption("--nsa-h_k") is not None
        else 1
    )
    h_v = (
        int(request.config.getoption("--nsa-h_v"))
        if request.config.getoption("--nsa-h_v") is not None
        else 1
    )

    actual_s_q = (
        torch.tensor([s_q] * b, dtype=torch.int32).cuda() if layout == "thd" else None
    )
    actual_s_kv = (
        torch.tensor([s_kv] * b, dtype=torch.int32).cuda() if layout == "thd" else None
    )
    topk_sizes = (
        torch.tensor([topk_size] * b, dtype=torch.int32).cuda()
        if (layout == "thd" and topk_size is not None)
        else None
    )

    scale_softmax = 1.0 / math.sqrt(d_qk) if scale_softmax is None else scale_softmax

    skip_ref = request.config.getoption("--nsa-skip-ref", default=False)

    return {
        "b": b,
        "s_q": s_q,
        "s_kv": s_kv,
        "d_qk": d_qk,
        "d_v": d_v,
        "h_q": h_q,
        "h_k": h_k,
        "h_v": h_v,
        "actual_s_q": actual_s_q,
        "actual_s_kv": actual_s_kv,
        "layout": layout,
        "dtype": dtype,
        "acc_dtype": acc_dtype,
        "selection_block_size": selection_block_size,
        "compress_stride": compress_stride,
        "k_value": k_value,
        "is_causal": is_causal,
        "mma_tiler_mn": mma_tiler_mn,
        "is_persistent": is_persistent,
        "scale_q": scale_q,
        "scale_k": scale_k,
        "scale_v": scale_v,
        "inv_scale_o": inv_scale_o,
        "scale_softmax": scale_softmax,
        "skip_ref": skip_ref,
        "window_size": window_size,
        "topk_size": topk_size,
        "topk_sizes": topk_sizes,
        "block_size": block_size,
    }


def allocate_input_tensors(cfg):
    layout = cfg["layout"]

    b = cfg["b"]
    s_q = cfg["s_q"]
    s_kv = cfg["s_kv"]
    d_qk = cfg["d_qk"]
    d_v = cfg["d_v"]
    h_q = cfg["h_q"]
    h_k = cfg["h_k"]
    h_v = cfg["h_v"]
    actual_s_q = cfg["actual_s_q"]
    actual_s_kv = cfg["actual_s_kv"]

    dtype = cfg["dtype"]
    acc_dtype = cfg["acc_dtype"]
    selection_block_size = cfg["selection_block_size"]
    compress_stride = cfg["compress_stride"]
    k_value = cfg["k_value"]

    (
        Q,
        K,
        V,
        LSE,
        cum_seqlen_q,
        cum_seqlen_kv,
        max_s_q,
        max_s_kv,
    ) = (None, None, None, None, None, None, None, None)
    if layout == "bshd":
        Q = torch.randn(b, s_q, h_q, d_qk, dtype=dtype).transpose(1, 2).cuda()
        K = torch.randn(b, s_kv, h_k, d_qk, dtype=dtype).transpose(1, 2).cuda()
        V = torch.randn(b, s_kv, h_k, d_v, dtype=dtype).transpose(1, 2).cuda()
        LSE = (
            -1.0
            * torch.randn(b, s_q, h_q, dtype=torch.float32)
            .transpose(1, 2)
            .contiguous()
            .cuda()
        )

        block_counts, block_indices = None, None  # TODO
    elif layout == "thd":
        cum_seqlen_q = (
            torch.cat([torch.tensor([0]).cuda(), torch.cumsum(actual_s_q, dim=0)])
            .to(torch.int32)
            .cuda()
        )
        cum_seqlen_kv = (
            torch.cat([torch.tensor([0]).cuda(), torch.cumsum(actual_s_kv, dim=0)])
            .to(torch.int32)
            .cuda()
        )
        max_s_q = max(actual_s_q).item()
        max_s_kv = max(actual_s_kv).item()

        total_seq_len_q = max(actual_s_q.sum().item(), actual_s_kv.sum().item())
        total_seq_len_kv = actual_s_kv.sum().item()
        # Q: (T, H_q, D_qk)
        Q = torch.randn((total_seq_len_q, h_q, d_qk), dtype=dtype).cuda()
        # K: (T, H_kv, D_qk)
        K = torch.randn((total_seq_len_kv, h_k, d_qk), dtype=dtype).cuda()
        # V: (T, H_kv, D_v)
        V = torch.randn((total_seq_len_kv, h_k, d_v), dtype=dtype).cuda()
        # LSE: (T, H_q, 1)
        LSE = (
            -1.0
            * torch.randn((1, h_q, total_seq_len_q), dtype=torch.float32)
            .transpose(0, 2)
            .cuda()
        )

        # block_counts: (T, H_kv), block_indices: (T, H_kv, max(topk_sizes))
        block_counts, block_indices = None, None  # TODO

    return (
        Q,
        K,
        V,
        LSE,
        actual_s_q,
        actual_s_kv,
        cum_seqlen_q,
        cum_seqlen_kv,
        max_s_q,
        max_s_kv,
    )


def allocate_output_tensors(cfg):
    layout = cfg["layout"]
    b = cfg["b"]
    s_q = cfg["s_q"]
    actual_s_q = cfg["actual_s_q"]
    h_q = cfg["h_q"]
    d_v = cfg["d_v"]
    h_k = cfg["h_k"]
    k_value = cfg["k_value"]
    acc_dtype = cfg["acc_dtype"]
    dtype = cfg["dtype"]

    O, L, M = None, None, None
    topk_scores, topk_indices = None, None
    if layout == "bshd":
        O = torch.empty(b, s_q, h_q, d_v, dtype=dtype).transpose(1, 2).cuda()
        L = torch.empty(b, s_q, h_q, 1, dtype=torch.float32).transpose(1, 2).cuda()
        M = torch.empty(b, s_q, h_q, 1, dtype=torch.float32).transpose(1, 2).cuda()

        if k_value is not None:
            topk_scores = (
                torch.empty(b, s_q, h_k, k_value, dtype=acc_dtype)
                .transpose(1, 2)
                .cuda()
            )
            topk_indices = (
                torch.empty(b, s_q, h_k, k_value, dtype=torch.int32)
                .transpose(1, 2)
                .cuda()
            )
    elif layout == "thd":
        total_seq_len = actual_s_q.sum().item()

        O = torch.empty(total_seq_len, h_q, d_v, dtype=dtype).cuda()
        L = torch.empty(total_seq_len, h_q, 1, dtype=torch.float32).cuda()
        M = torch.empty(total_seq_len, h_q, 1, dtype=torch.float32).cuda()

        if k_value is not None:
            topk_scores = torch.empty(
                total_seq_len, h_k, k_value, dtype=acc_dtype
            ).cuda()
            topk_indices = torch.empty(
                total_seq_len, h_k, k_value, dtype=torch.int32
            ).cuda()

    return (
        O,
        L,
        M,
        topk_scores,
        topk_indices,
    )


def _compute_exclusive_prefix_sum(tensor):
    assert list(tensor.size())[1:] == [1, 1, 1]
    # We need to provide a tuple of two tensors to torch.cat().
    return torch.cat(
        (
            torch.zeros(1, 1, 1, 1, dtype=tensor.dtype, device=tensor.device),
            torch.cumsum(tensor, dim=0),
        )
    )


def generate_ragged_offset(cfg):
    if cfg["layout"] != "thd":
        return None, None, None, None, None

    h_q = cfg["h_q"]
    h_k = cfg["h_k"]
    h_v = cfg["h_v"]
    d_qk = cfg["d_qk"]
    d_v = cfg["d_v"]
    seq_len_q = cfg["actual_s_q"]
    seq_len_kv = cfg["actual_s_kv"]

    # Only for thd_thd_thd
    if seq_len_q.ndim == 1:
        seq_len_q = seq_len_q.view(-1, 1, 1, 1)
    if seq_len_kv.ndim == 1:
        seq_len_kv = seq_len_kv.view(-1, 1, 1, 1)

    q_ragged_offset = _compute_exclusive_prefix_sum(seq_len_q) * h_q * d_qk
    k_ragged_offset = _compute_exclusive_prefix_sum(seq_len_kv) * h_k * d_qk
    v_ragged_offset = _compute_exclusive_prefix_sum(seq_len_kv) * h_v * d_v
    o_ragged_offset = _compute_exclusive_prefix_sum(seq_len_q) * h_q * d_v
    stats_ragged_offset = _compute_exclusive_prefix_sum(seq_len_q) * h_q

    # Convert to int64 for cuDNN 9.6.0
    q_ragged_offset = q_ragged_offset.to(dtype=torch.int64).cuda()
    k_ragged_offset = k_ragged_offset.to(dtype=torch.int64).cuda()
    v_ragged_offset = v_ragged_offset.to(dtype=torch.int64).cuda()
    o_ragged_offset = o_ragged_offset.to(dtype=torch.int64).cuda()
    stats_ragged_offset = stats_ragged_offset.to(dtype=torch.int64).cuda()

    return (
        q_ragged_offset,
        k_ragged_offset,
        v_ragged_offset,
        o_ragged_offset,
        stats_ragged_offset,
    )


def generate_block_indices(
    seq_lens: list[int], num_kv_heads: int, topk_sizes: list[int], block_size: int
):
    """
    Generate block indices and counts for sparse attention.

    Args:
        seq_lens: List of sequence lengths for each batch
        num_kv_heads: Number of key/value heads
        topk_sizes: List of top-k sizes for each batch
        block_size: Size of each block

    Returns:
        Tuple of (block_counts, block_indices) tensors on CUDA
    """
    total_seq_len = sum(seq_lens)
    max_topk_size = max(topk_sizes)
    block_counts = torch.zeros(total_seq_len, num_kv_heads, dtype=torch.int32)
    block_indices = torch.zeros(
        total_seq_len, num_kv_heads, max_topk_size, dtype=torch.int32
    )

    seq_len_offset = 0
    for i in range(len(seq_lens)):
        seq_len = seq_lens[i]
        topk_size = topk_sizes[i]
        max_index = seq_len // block_size
        assert (
            topk_size <= max_index
        ), "topk_size must be less than or equal to the number of blocks"
        for t in range(seq_len):
            for h in range(num_kv_heads):
                block_indices[seq_len_offset + t, h, :topk_size] = (
                    torch.randperm(max_index)[:topk_size].sort().values
                )
                block_counts[seq_len_offset + t, h] = topk_size
        seq_len_offset += seq_len

    return block_counts.cuda(), block_indices.cuda()
