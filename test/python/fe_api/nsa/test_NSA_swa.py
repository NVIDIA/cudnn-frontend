# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import cudnn

import pytest
from test_utils import torch_fork_set_rng

from fe_api.nsa.nsa_utils import (
    nsa_init,
    allocate_input_tensors,
    allocate_output_tensors,
    with_nsa_swa_params,
    generate_ragged_offset,
)

from fe_api.nsa.nsa_reference import check_ref_nsa_swa


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_swa_params
def test_nsa_swa_compile_execute(
    layout,
    dtype,
    acc_dtype,
    window_size,
    scale_softmax,
    request,
):
    try:
        from cudnn import NSA
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        scale_softmax=scale_softmax,
        window_size=window_size,
    )

    (
        Q,
        K,
        V,
        _,
        actual_s_q,
        actual_s_kv,
        cum_seqlen_q,
        cum_seqlen_kv,
        max_s_q,
        max_s_kv,
    ) = allocate_input_tensors(cfg)

    O, Stats, _, _, _ = allocate_output_tensors(cfg)
    cudnn_handle = cudnn.create_handle()

    swa = NSA.SlidingWindowAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_stats=Stats,
        sample_seq_len_q=actual_s_q,
        sample_seq_len_kv=actual_s_kv,
        max_seq_len_q=max_s_q,
        max_seq_len_kv=max_s_kv,
        left_bound=cfg["window_size"],
        right_bound=0,
        attn_scale=cfg["scale_softmax"],
        intermediate_data_type=cfg["acc_dtype"],
        compute_data_type=cfg["acc_dtype"],
        cudnn_handle=cudnn_handle,
    )

    assert swa.check_support() is True
    swa.compile()
    # Rule 8: execute() launches asynchronously on the handle's stream and never
    # blocks the host; torch's sync debug mode turns a torch.cuda.synchronize()
    # inside it into an error.
    previous_sync_debug_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        swa.execute(
            q_tensor=Q,
            k_tensor=K,
            v_tensor=V,
            seq_len_q_tensor=actual_s_q,
            seq_len_kv_tensor=actual_s_kv,
            o_tensor=O,
            stats_tensor=Stats,
        )
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_debug_mode)

    check_ref_nsa_swa(
        Q,
        K,
        V,
        O,
        Stats,
        actual_s_q,
        actual_s_kv,
        max_s_q,
        max_s_kv,
        cfg,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_execute_allocates_scratch_on_the_handle_stream(request, monkeypatch):
    """The handle is re-streamed to a side stream while torch stays on its default stream.
    execute()'s per-call scratch must be allocated on the handle's (launch) stream (R1):
    the caching allocator orders a block's reuse only against the stream it was allocated
    on, so scratch allocated on the default stream could be handed to the next allocation
    while the graph is still reading it on the side stream."""
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = nsa_init(
        request=request,
        layout="bshd",
        dtype=torch.float16,
        acc_dtype=torch.float32,
        scale_softmax=None,
        window_size=64,
    )
    Q, K, V, _, actual_s_q, actual_s_kv, _, _, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    O, Stats, _, _, _ = allocate_output_tensors(cfg)
    cudnn_handle = cudnn.create_handle()
    swa = NSA.SlidingWindowAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_stats=Stats,
        sample_seq_len_q=actual_s_q,
        sample_seq_len_kv=actual_s_kv,
        max_seq_len_q=max_s_q,
        max_seq_len_kv=max_s_kv,
        left_bound=cfg["window_size"],
        right_bound=0,
        attn_scale=cfg["scale_softmax"],
        intermediate_data_type=cfg["acc_dtype"],
        compute_data_type=cfg["acc_dtype"],
        cudnn_handle=cudnn_handle,
    )
    assert swa.check_support() is True
    swa.compile()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())  # the inputs were written on the default stream
    allocation_streams = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        allocation_streams.append(torch.cuda.current_stream().cuda_stream)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", recording_empty)
    swa.execute(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        o_tensor=O,
        stats_tensor=Stats,
        current_stream=side.cuda_stream,
    )
    seen = set(allocation_streams)
    assert allocation_streams, "execute() allocated nothing; the test no longer covers the scratch allocation"
    assert seen == {side.cuda_stream}, f"scratch allocated on stream(s) {seen}, the graph runs on {side.cuda_stream}"
    assert cudnn.get_stream(cudnn_handle) == side.cuda_stream

    torch.cuda.current_stream().wait_stream(side)
    check_ref_nsa_swa(
        Q,
        K,
        V,
        O,
        Stats,
        actual_s_q,
        actual_s_kv,
        max_s_q,
        max_s_kv,
        cfg,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_swa_params
def test_nsa_swa_wrapper(
    layout,
    dtype,
    acc_dtype,
    window_size,
    scale_softmax,
    request,
):
    try:
        from cudnn import NSA
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        scale_softmax=scale_softmax,
        window_size=window_size,
    )

    (
        Q,
        K,
        V,
        _,
        actual_s_q,
        actual_s_kv,
        cum_seqlen_q,
        cum_seqlen_kv,
        max_s_q,
        max_s_kv,
    ) = allocate_input_tensors(cfg)
    cudnn_handle = cudnn.create_handle()

    O, Stats = NSA.sliding_window_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        left_bound=cfg["window_size"],
        right_bound=0,
        is_infer=False,
        attn_scale=cfg["scale_softmax"],
        o_dtype=cfg["dtype"],
        intermediate_data_type=cfg["acc_dtype"],
        compute_data_type=cfg["acc_dtype"],
        cudnn_handle=cudnn_handle,
    )

    check_ref_nsa_swa(
        Q,
        K,
        V,
        O,
        Stats,
        actual_s_q,
        actual_s_kv,
        max_s_q,
        max_s_kv,
        cfg,
    )
