# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real-backend empty-query compatibility and retained-capture regression."""

import os
import subprocess
import sys

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L0


def _exercise(dtype, dim, stats, mask, handle, *, pin_engine=None, layout="ragged"):
    from sdpa import fp16
    from sdpa.random_config import ExecConfig

    cfg = ExecConfig(
        data_type=dtype,
        rng_geom_seed=1741,
        rng_data_seed=1741,
        is_padding=True,
        is_ragged=True,
        is_paged=layout == "paged",
        is_cu_seq_len=layout == "cumulative",
        block_size=64 if layout == "paged" else None,
        batches=3,
        h_q=4,
        h_k=2,
        h_v=2,
        s_q=129,
        s_kv=257,
        d_qk=dim,
        d_v=dim,
        seq_len_q=[129, 129, 129],
        seq_len_kv=[193, 257, 129],
        total_q=411,
        total_kv=801,
        ragged_origin_rows={"q": [0, 137, 274, 411], "o": [0, 137, 274, 411], "stats": [0, 137, 274, 411], "k": [0, 267, 534, 801], "v": [0, 267, 534, 801]},
        with_ragged_token_gap=False,
        with_ragged_head_gap=False,
        fwd_stats=stats,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT if mask == "window" else cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=97 if mask == "window" else None,
        right_bound=0 if mask != "none" else None,
    )
    if layout == "paged":
        cfg.ragged_origin_rows = {name: rows for name, rows in cfg.ragged_origin_rows.items() if name not in ("k", "v")}
    cfg.fill_derived_fields()
    allocs, tensors, _, _ = fp16.allocate_tensors(cfg, torch.Generator(device="cuda").manual_seed(1741), perf=False)
    uid = fp16.TensorUid
    captured = None
    q_payload = [tensors[uid.q][start : start + 129].clone() for start in cfg.ragged_origin_rows["q"][:-1]]
    try:

        def check_plans(graph):
            records = [graph.get_engine_and_knobs_at_index(i) for i in range(graph.get_execution_plan_count())]
            if pin_engine is not None:
                knobs = next(knobs for engine, knobs in records if engine == pin_engine)
                graph.create_execution_plan(pin_engine, knobs)
                graph.select_plan(graph.get_execution_plan_count() - 1)
            elif cudnn.backend_version() < 92600:
                assert all(engine != 11 for engine, _ in records)
                with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="zero-length Q"):
                    graph.create_execution_plan(11, {})

        check_plans.pins = pin_engine is not None
        graph, pack = fp16.create_forward_graph(cfg, tensors, handle, plan_hook=check_plans)
        workspace = torch.full((max(1, graph.get_workspace_size()),), 255, device="cuda", dtype=torch.uint8)
        graph.execute(pack, workspace)
        torch.cuda.synchronize()
        fp16.compute_and_compare_reference(cfg, allocs, tensors, False)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=torch.cuda.current_stream()):
            graph.execute(pack, workspace)
        for iteration, lengths in enumerate(([97, 0, 129], [0, 65, 33], [0, 0, 0], [129, 97, 63]), 1):
            cfg.seq_len_q = lengths
            prefixes = [0]
            for length in lengths:
                prefixes.append(prefixes[-1] + length)
            # Keep physical origins consistent with compact packed callers,
            # including duplicate offsets for empty sequences.
            for name in ("q", "o", "stats"):
                cfg.ragged_origin_rows[name] = prefixes
                offset_uid = getattr(uid, name + "_ragged_offset")
                stride = getattr(cfg, "stride_" + name)[2]
                tensors[offset_uid].copy_(torch.tensor(prefixes, device="cuda", dtype=torch.int64) * stride)
            for batch, length in enumerate(lengths):
                start = prefixes[batch]
                tensors[uid.q][start : start + length].copy_(q_payload[batch][:length] * (0.75**iteration))
            if layout == "cumulative":
                tensors[uid.cu_seq_len_q].copy_(torch.tensor(prefixes, device="cuda", dtype=torch.int32))
            else:
                tensors[uid.seq_len_q].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            tensors[uid.v].add_(0.125)
            if layout == "paged":
                # The harness reference uses dense V, while the graph binds
                # its independent page-pool copy. Change both identically.
                tensors[uid.container_v].add_(0.125)
            for launch in (lambda: graph.execute(pack, workspace), captured.replay):
                tensors[uid.o].fill_(float("nan"))
                if stats:
                    tensors[uid.stats].fill_(float("nan"))
                if pin_engine is not None:
                    print("EMPTY_Q_EXECUTE", cudnn.backend_version(), pin_engine, lengths, flush=True)
                torch.cuda.set_sync_debug_mode("error")
                try:
                    launch()
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                torch.cuda.synchronize()
                fp16.compute_and_compare_reference(cfg, allocs, tensors, False)
    finally:
        if captured is not None:
            captured.reset()
        fp16.cleanup_tensors(allocs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
def test_sm120_empty_query_default_and_changed_capture(dtype, dim, stats, mask, cudnn_handle, monkeypatch):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 backend compatibility")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    original = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(original)
    try:
        with torch.cuda.stream(stream):
            cudnn.set_stream(handle=cudnn_handle, stream=stream.cuda_stream)
            _exercise(dtype, dim, stats, mask, cudnn_handle)
    finally:
        original.wait_stream(stream)
        cudnn.set_stream(handle=cudnn_handle, stream=original.cuda_stream)


@pytest.mark.parametrize("layout", ["paged", "cumulative"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats", [False, True])
def test_sm120_empty_query_length_transports(layout, dtype, stats, cudnn_handle, monkeypatch):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 backend compatibility")
    if layout == "cumulative" and cudnn.backend_version() < 92400:
        pytest.skip("cumulative length API requires cuDNN 9.24")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    original = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(original)
    try:
        with torch.cuda.stream(stream):
            cudnn.set_stream(handle=cudnn_handle, stream=stream.cuda_stream)
            _exercise(dtype, 128, stats, "window", cudnn_handle, layout=layout)
    finally:
        original.wait_stream(stream)
        cudnn.set_stream(handle=cudnn_handle, stream=original.cuda_stream)


def _cold_empty(dtype, stats, handle, *, pin_engine=None):
    from sdpa import fp16
    from sdpa.random_config import ExecConfig

    cfg = ExecConfig(
        data_type=dtype,
        rng_geom_seed=1741,
        rng_data_seed=1741,
        is_padding=True,
        is_ragged=True,
        batches=3,
        h_q=4,
        h_k=2,
        h_v=2,
        s_q=129,
        s_kv=257,
        d_qk=64,
        d_v=64,
        seq_len_q=[97, 0, 129],
        seq_len_kv=[193, 257, 129],
        with_ragged_token_gap=False,
        with_ragged_head_gap=False,
        fwd_stats=stats,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    )
    cfg.fill_derived_fields()
    allocs, tensors, _, _ = fp16.allocate_tensors(cfg, torch.Generator(device="cuda").manual_seed(1741), perf=False)
    try:

        def select(graph):
            records = [graph.get_engine_and_knobs_at_index(i) for i in range(graph.get_execution_plan_count())]
            if pin_engine is not None:
                knobs = next(knobs for engine, knobs in records if engine == pin_engine)
                graph.create_execution_plan(pin_engine, knobs)
                graph.select_plan(graph.get_execution_plan_count() - 1)
            elif cudnn.backend_version() < 92600:
                assert all(engine != 11 for engine, _ in records)

        select.pins = pin_engine is not None
        graph, pack = fp16.create_forward_graph(cfg, tensors, handle, plan_hook=select)
        # Match the original harness: caller workspace contents are undefined.
        # Retain its 0xff initialization so zero-filled allocator pages cannot
        # hide an execution defect.
        workspace = torch.full((max(1, graph.get_workspace_size()),), 255, device="cuda", dtype=torch.uint8)
        print("EMPTY_Q_EXECUTE", cudnn.backend_version(), pin_engine, "cold compact empty sequence", flush=True)
        graph.execute(pack, workspace)
        torch.cuda.synchronize()
        fp16.compute_and_compare_reference(cfg, allocs, tensors, False)
    finally:
        fp16.cleanup_tensors(allocs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats", [False, True])
def test_sm120_cold_packed_empty_query(dtype, stats, cudnn_handle, monkeypatch):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 backend compatibility")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)
    _cold_empty(dtype, stats, cudnn_handle)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("stats", [False, True])
def test_sm120_empty_query_backend_fix_boundary(dtype, stats):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 backend compatibility")
    # A raw-engine failure poisons its CUDA context. Keep it in a fresh process,
    # and require the exact execution-stage error, never an arbitrary nonzero exit.
    code = f"""
import sys, os
sys.meta_path[:] = [f for f in sys.meta_path if '__editable__' not in str(f)]
sys.path[:] = {sys.path!r}
import cudnn, torch
assert cudnn.__file__ == {cudnn.__file__!r}
assert cudnn.backend_version() == {cudnn.backend_version()}
from cudnn.sdpa.fwd import backend_guard
backend_guard.SM120_EMPTY_Q_CRASH_FIXED_IN = 0
from sdpa.graph.test_sdpa_sm120_empty_q_compat import _cold_empty
handle = cudnn.create_handle()
stream = torch.cuda.Stream()
try:
    with torch.cuda.stream(stream):
        cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
        _cold_empty(getattr(torch, {dtype!r}), {stats!r}, handle, pin_engine=11)
except RuntimeError as error:
    if 'illegal instruction' in str(error):
        print('EMPTY_Q_ILLEGAL_INSTRUCTION', flush=True)
        os._exit(71)
    raise
cudnn.destroy_handle(handle)
print('EMPTY_Q_REFERENCE_PASS', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=dict(os.environ, CUDNN_FRONTEND_ENABLE_FROST_ENGINES="")
    )
    diagnostic = result.stdout + result.stderr
    if cudnn.backend_version() < 92600:
        assert result.returncode == 71 and "EMPTY_Q_EXECUTE" in diagnostic and "EMPTY_Q_ILLEGAL_INSTRUCTION" in diagnostic, diagnostic
    else:
        assert result.returncode == 0 and "EMPTY_Q_REFERENCE_PASS" in diagnostic, diagnostic
