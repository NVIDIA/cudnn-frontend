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
    # T,H,D: the caller owns the ragged offsets (all None for bshd).
    q_off, k_off, v_off, o_off, stats_off = generate_ragged_offset(cfg)

    # Rule 8 holds BUILD to the execute standard too: no device read and no
    # torch-visible sync in __init__ / check_support / compile.
    torch.cuda.set_sync_debug_mode("error")
    try:
        swa = NSA.SlidingWindowAttention(
            sample_q=Q,
            sample_k=K,
            sample_v=V,
            sample_o=O,
            sample_stats=Stats,
            sample_seq_len_q=actual_s_q,
            sample_seq_len_kv=actual_s_kv,
            sample_q_ragged_offset=q_off,
            sample_k_ragged_offset=k_off,
            sample_v_ragged_offset=v_off,
            sample_o_ragged_offset=o_off,
            sample_stats_ragged_offset=stats_off,
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
    finally:
        torch.cuda.set_sync_debug_mode("default")

    # R2: the caller allocates the backend workspace once and hands it to every execute.
    ws = torch.empty(max(swa.get_workspace_size(), 1), dtype=torch.uint8, device=Q.device)
    execute_kwargs = dict(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        o_tensor=O,
        stats_tensor=Stats,
        q_ragged_offset_tensor=q_off,
        k_ragged_offset_tensor=k_off,
        v_ragged_offset_tensor=v_off,
        o_ragged_offset_tensor=o_off,
        stats_ragged_offset_tensor=stats_off,
        workspace=ws,
    )
    # Rule 8: execute() launches asynchronously on the handle's stream and never
    # blocks the host; torch's sync debug mode turns a torch.cuda.synchronize()
    # inside it into an error.
    previous_sync_debug_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        swa.execute(**execute_kwargs)
        # R9 detector: three warm executes allocate nothing and never sync.
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            swa.execute(**execute_kwargs)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations, "SlidingWindowAttention.execute allocated device memory"
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
def test_nsa_swa_execute_on_a_side_stream_allocates_nothing(request, monkeypatch):
    """The handle is re-streamed to a side stream while torch stays on its default stream.
    The caller owns the workspace (R2) and allocates it on the launch stream (R1: the caching
    allocator orders a block's reuse only against the stream it was allocated on); execute()
    itself allocates nothing, so there is no scratch whose lifetime the class could get wrong."""
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
    with torch.cuda.stream(side):  # the caller allocates on the launch stream (R1)
        ws = torch.empty(max(swa.get_workspace_size(), 1), dtype=torch.uint8, device=Q.device)

    allocations = []
    for name in ("empty", "zeros", "ones", "empty_like", "zeros_like", "full", "tensor", "arange"):
        real = getattr(torch, name)

        def recording(*args, _real=real, _name=name, **kwargs):
            allocations.append((_name, torch.cuda.current_stream().cuda_stream))
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
        workspace=ws,
        current_stream=side.cuda_stream,
    )
    seen = list(allocations)
    assert not seen, f"execute() allocated {seen}; the caller owns every buffer (R2)"
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
        max_seq_len_q=max_s_q,  # required for thd (None for bshd)
        max_seq_len_kv=max_s_kv,
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


def _thd_swa_case(request, window_size=64):
    """One compiled T,H,D SWA graph (training mode) plus its execute arguments, for the contract tests."""
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = nsa_init(request=request, layout="thd", dtype=torch.float16, acc_dtype=torch.float32, scale_softmax=None, window_size=window_size)
    Q, K, V, _, actual_s_q, actual_s_kv, _, _, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    O, Stats, _, _, _ = allocate_output_tensors(cfg)
    q_off, k_off, v_off, o_off, stats_off = generate_ragged_offset(cfg)
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
        cudnn_handle=cudnn.create_handle(),
    )
    assert swa.check_support() is True
    swa.compile()
    execute_kwargs = dict(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        o_tensor=O,
        stats_tensor=Stats,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        q_ragged_offset_tensor=q_off,
        k_ragged_offset_tensor=k_off,
        v_ragged_offset_tensor=v_off,
        o_ragged_offset_tensor=o_off,
        stats_ragged_offset_tensor=stats_off,
    )
    return NSA, cfg, swa, execute_kwargs, (Q, K, V, O, Stats, actual_s_q, actual_s_kv, max_s_q, max_s_kv)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_thd_execute_requires_ragged_offsets(request):
    """Rule 1/8: the engine never derives the T,H,D ragged offsets (per-execute cat/cumsum); they are a required set."""
    NSA, _, swa, execute_kwargs, _ = _thd_swa_case(request)
    ws = torch.empty(max(swa.get_workspace_size(), 1), dtype=torch.uint8, device=execute_kwargs["q_tensor"].device)
    none_offsets = {k: None for k in execute_kwargs if k.endswith("_ragged_offset_tensor")}
    with pytest.raises(ValueError, match="ragged_offset.*required for the T,H,D layout"):
        swa.execute(**dict(execute_kwargs, **none_offsets), workspace=ws)
    # a partial set is rejected as a set, not tolerated
    with pytest.raises(ValueError, match="missing"):
        swa.execute(**dict(execute_kwargs, v_ragged_offset_tensor=None), workspace=ws)
    # wrong dtype is rejected before launch
    bad = execute_kwargs["q_ragged_offset_tensor"].to(torch.int32)
    with pytest.raises(ValueError, match="int64"):
        swa.execute(**dict(execute_kwargs, q_ragged_offset_tensor=bad), workspace=ws)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_execute_requires_workspace(request):
    """R2: execute() takes the caller's get_workspace_size() buffer and never allocates one."""
    NSA, _, swa, execute_kwargs, _ = _thd_swa_case(request)
    required = swa.get_workspace_size()
    if required == 0:
        pytest.skip("this plan needs no workspace; the None contract is not exercised")
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace but execute\(\) received none"):
        swa.execute(**execute_kwargs)
    device = execute_kwargs["q_tensor"].device
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace"):
        swa.execute(**execute_kwargs, workspace=torch.empty(required - 1, dtype=torch.uint8, device=device))
    swa.execute(**execute_kwargs, workspace=torch.empty(required, dtype=torch.uint8, device=device))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_wrapper_thd_derives_max_seq_len_and_restrides_offsets(request):
    """The wrapper keeps develop's eager surface: with the envelope ints omitted it derives them from seq_len_*,
    and caller-provided strided ragged offsets are made contiguous. The class itself still rejects both."""
    NSA, cfg, swa, execute_kwargs, (Q, K, V, O, Stats, actual_s_q, actual_s_kv, max_s_q, max_s_kv) = _thd_swa_case(request)
    # stride-2 views of the reference offsets: declined by the class, normalised by the wrapper
    strided = {k: torch.stack([v, v], dim=1).flatten(0, 1)[::2] for k, v in execute_kwargs.items() if k.endswith("_ragged_offset_tensor")}
    assert all(not v.is_contiguous() for v in strided.values())
    ws = torch.empty(max(swa.get_workspace_size(), 1), dtype=torch.uint8, device=Q.device)
    with pytest.raises(ValueError, match="must be contiguous"):
        swa.execute(**dict(execute_kwargs, **strided), workspace=ws)
    no_envelope = NSA.SlidingWindowAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_stats=Stats,
        sample_seq_len_q=actual_s_q,
        sample_seq_len_kv=actual_s_kv,
        cudnn_handle=swa._cudnn_handle,
    )
    with pytest.raises(ValueError, match="max_seq_len_q and max_seq_len_kv must be provided"):
        no_envelope.check_support()

    O_w, Stats_w = NSA.sliding_window_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        **strided,
        left_bound=cfg["window_size"],
        right_bound=0,
        attn_scale=cfg["scale_softmax"],
        cudnn_handle=cudnn.create_handle(),
    )
    check_ref_nsa_swa(Q, K, V, O_w, Stats_w, actual_s_q, actual_s_kv, max_s_q, max_s_kv, cfg)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_wrapper_handle_only_thd_derives_offsets_on_the_handle_stream(request):
    """Handle-only call (``stream=None``, handle re-streamed to a side stream): the wrapper derives the packed
    ragged offsets, allocates the outputs and the workspace and launches on the HANDLE's stream, so a busy ambient
    torch stream neither delays the offsets behind the launch nor lets the launch read them before they exist
    (Rule 5). The explicit-stream call is the control."""
    NSA, cfg, _, _, (Q, K, V, _, _, actual_s_q, actual_s_kv, max_s_q, max_s_kv) = _thd_swa_case(request)
    common = dict(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        left_bound=cfg["window_size"],
        right_bound=0,
        attn_scale=cfg["scale_softmax"],
        max_seq_len_q=max_s_q,
        max_seq_len_kv=max_s_kv,
    )
    side = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, side.cuda_stream)
    torch.cuda.synchronize()  # the inputs are complete; from here the ambient stream only carries the delay below
    torch.cuda._sleep(1_000_000_000)  # ambient torch stream busy: work enqueued there runs after the handle-stream launch
    O_h, Stats_h = NSA.sliding_window_attention_wrapper(**common, cudnn_handle=handle)  # stream=None: the handle's stream
    O_s, Stats_s = NSA.sliding_window_attention_wrapper(**common, cudnn_handle=cudnn.create_handle(), stream=side.cuda_stream)
    torch.cuda.synchronize()
    assert cudnn.get_stream(handle) == side.cuda_stream
    check_ref_nsa_swa(Q, K, V, O_h, Stats_h, actual_s_q, actual_s_kv, max_s_q, max_s_kv, cfg)
    check_ref_nsa_swa(Q, K, V, O_s, Stats_s, actual_s_q, actual_s_kv, max_s_q, max_s_kv, cfg)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_packed_thd_ragged_offsets_match_reference(request):
    """NSA.packed_thd_ragged_offsets reproduces the test reference offsets for a fully packed batch."""
    NSA, cfg, _, execute_kwargs, (Q, K, V, O, Stats, actual_s_q, actual_s_kv, _, _) = _thd_swa_case(request)
    got = NSA.packed_thd_ragged_offsets(actual_s_q, actual_s_kv, Q, K, V, O, Stats)
    ref = generate_ragged_offset(cfg)
    for name, g, r in zip(("q", "k", "v", "o", "stats"), got, ref):
        assert g.dtype == torch.int64 and tuple(g.shape) == tuple(r.shape), name
        torch.testing.assert_close(g, r, msg=name)
