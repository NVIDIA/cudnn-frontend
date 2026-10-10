# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import cudnn

import pytest
from test_utils import torch_fork_set_rng

from native_sparse_attention.cutedsl.nsa_utils import (
    nsa_init,
    allocate_input_tensors,
    allocate_output_tensors,
    with_nsa_swa_params,
    generate_ragged_offset,
)

from native_sparse_attention.cutedsl.nsa_reference import check_ref_nsa_swa


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
@pytest.mark.parametrize("layout", ["bshd", "thd"])
@pytest.mark.parametrize("with_stats", [False, True])
def test_nsa_swa_ordered_bindings_rebind(layout, with_stats, request, monkeypatch):
    """Changing caller buffers preserves the mapping API's layouts and optional ports."""
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=torch.float16,
        acc_dtype=torch.float32,
        window_size=64,
        s_q_default_override=32,
        s_kv_default_override=32,
    )
    q, k, v, _, q_lens, kv_lens, _, _, max_q, max_kv = allocate_input_tensors(cfg)
    out, stats, _, _, _ = allocate_output_tensors(cfg)
    stats = stats if with_stats else None
    handle = cudnn.create_handle()
    swa = NSA.SlidingWindowAttention(
        q,
        k,
        v,
        out,
        sample_stats=stats,
        left_bound=cfg["window_size"],
        sample_seq_len_q=q_lens,
        sample_seq_len_kv=kv_lens,
        max_seq_len_q=max_q,
        max_seq_len_kv=max_kv,
        cudnn_handle=handle,
    )
    try:
        assert swa.check_support()
        swa.compile()
        graph = swa._cudnn_swa_graph
        execute = graph.execute
        calls = []

        def recording(buffers, workspace, *, handle, tensor_uids=None):
            assert tensor_uids is not None, "the wrapper rebuilt a tensor mapping"
            assert isinstance(buffers, (tuple, list))
            assert len(tensor_uids) == len(set(tensor_uids)) == len(buffers)
            bindings = dict(zip(tensor_uids, buffers))
            assert all(isinstance(uid, int) for uid in bindings)
            calls.append((bindings, workspace))
            return execute(buffers, workspace, handle=handle, tensor_uids=tensor_uids)

        monkeypatch.setattr(graph, "execute", recording)
        for iteration in range(2):
            # Keep the first call's allocations alive so stale-pointer reuse is observable.
            q, k, v = (torch.randn_like(tensor) for tensor in (q, k, v))
            out = torch.full_like(out, float("nan"))
            stats = torch.full_like(stats, float("nan")) if with_stats else None
            swa.execute(q, k, v, out, stats, seq_len_q_tensor=q_lens, seq_len_kv_tensor=kv_lens)
            assert len(calls) == iteration + 1
            bindings, workspace = calls[-1]
            expected = ((swa.q_cudnn, q), (swa.k_cudnn, k), (swa.v_cudnn, v), (swa.o_cudnn, out))
            if with_stats:
                expected += ((swa.stats_cudnn, stats),)
            for tensor, buffer in expected:
                assert bindings[tensor.get_uid()].data_ptr() == buffer.data_ptr()
            assert len(bindings) == 4 + int(with_stats) + (6 + int(with_stats) if layout == "thd" else 0)
            ordered_out = out.clone()
            ordered_stats = stats.clone() if with_stats else None
            out.fill_(float("nan"))
            if with_stats:
                stats.fill_(float("nan"))
            execute(bindings, workspace, handle=handle)
            torch.testing.assert_close(out, ordered_out, rtol=0, atol=0)
            if with_stats:
                torch.testing.assert_close(stats, ordered_stats, rtol=0, atol=0)
    finally:
        cudnn.destroy_handle(handle)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("layout", ["bshd", "thd"])
def test_nsa_swa_execute_allocates_scratch_on_the_handle_stream(layout, request, monkeypatch):
    """The handle is re-streamed to a side stream while torch stays on its default stream.
    Everything execute() allocates per call -- the backend workspace and, for THD without
    caller offsets, the five internally generated ragged-offset tensors -- must be produced
    on the handle's (launch) stream (R1): the caching allocator orders a block's reuse only
    against the stream it was allocated on, so a buffer produced on the default stream could
    be handed to the next allocation while the graph is still reading it on the side stream."""
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = nsa_init(
        request=request,
        layout=layout,
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
    for name in ("empty", "zeros", "cat", "cumsum"):  # the workspace + the ragged-offset producers
        real = getattr(torch, name)

        def recording(*args, _real=real, **kwargs):
            allocation_streams.append(torch.cuda.current_stream().cuda_stream)
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, recording)
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
    expected_calls = 1 if layout == "bshd" else 1 + 5 * 2  # workspace (+ zeros/cumsum per generated offset, cat once each)
    assert len(allocation_streams) >= expected_calls, f"execute() made {len(allocation_streams)} allocating calls; the test no longer covers them all"
    assert seen == {side.cuda_stream}, f"allocations on stream(s) {seen}, the graph runs on {side.cuda_stream}"
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
