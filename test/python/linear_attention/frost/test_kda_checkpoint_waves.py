# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent-head checkpoint reuse: all gradients, canaries and poisoned graph replay."""

from itertools import accumulate

import pytest
import torch
import cutlass.cute as cute

import cudnn
from cudnn.linear_attention.frost import kda_engine
from determinism_utils import assert_bitwise, snapshot
from linear_attention.determinism import pack, sequences
from linear_attention.frost.test_kda_piece_capacity import _prepared
from linear_attention.test_la import frost_la_unavailable

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.parametrize(
    "dtype,safe,dim",
    [(torch.bfloat16, False, 64), (torch.bfloat16, True, 128), (torch.float16, False, 128), (torch.float16, True, 64)],
    ids=["bf16-log-d64", "bf16-safe-d128", "fp16-log-d128", "fp16-safe-d64"],
)
@pytest.mark.parametrize("divisor", [2, 4])
@pytest.mark.parametrize("head_pattern", ["equal", "gqa", "gva"])
@pytest.mark.parametrize("state_waves", [False, True], ids=["checkpoint-only", "all-states"])
def test_checkpoint_waves_bitwise_and_capture(monkeypatch, dtype, safe, dim, divisor, head_pattern, state_waves):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", 0 if state_waves else 1024**3)
    inputs = pack(sequences("kda", [0, 1, 15, 16, 17, 8193] + [33] * 31, dtype=dtype, heads=4, key_dim=dim, value_dim=dim))
    if head_pattern == "gqa":
        inputs["q"] = inputs["q"][:, :2].contiguous()
        inputs["k"] = inputs["k"][:, :2].contiguous()
    elif head_pattern == "gva":
        inputs["v"] = inputs["v"][:, :2].contiguous()
    inputs["cu_seqlens"] = inputs.pop("cu")
    inputs["dO"] = inputs.pop("do")
    inputs["d_final_state"] = inputs.pop("dfs")
    inputs["g"].fill_(-4 if safe else -0.001)
    if safe:
        inputs["a_log"] = torch.zeros(4, device="cuda")
        inputs["dt_bias"] = torch.zeros(4, dim, device="cuda")
    handle = cudnn.create_handle()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: 1)
            baseline_call, baseline_result, baseline_storage, full_size = _prepared(inputs, "bwd", safe, handle)
            baseline_call()
            expected = snapshot(baseline_result)
            assert all(torch.isfinite(value).all() for value in expected.values())
            monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: divisor)
            execute, result, storage, size = _prepared(inputs, "bwd", safe, handle)
            assert size < full_size
            execute()
            assert_bitwise(snapshot(result), expected, "head wave eager")
            with torch.cuda.graph(graph, stream=stream):
                execute()
            for _ in range(8):
                storage[:size].fill_(0xFF)
                for value in result.values():
                    value.fill_(float("nan"))
                graph.replay()
                assert_bitwise(snapshot(result), expected, "head wave poisoned replay")
                assert (storage[size:] == 0xA5).all(), "workspace overrun"
            # Changed values and metadata at stable addresses exercise the
            # full capacity bound, including empty identities and ragged tails.
            inputs["dO"].neg_()
            lengths = [0, 1, 15, 16, 17, 8193] + [33] * 31
            for changed_lengths in ([sum(lengths)] + [0] * (len(lengths) - 1), list(reversed(lengths))):
                inputs["cu_seqlens"].copy_(torch.tensor([0, *accumulate(changed_lengths)], device="cuda", dtype=torch.int32))
                baseline_call()
                changed = snapshot(baseline_result)
                storage[:size].fill_(0xAB)
                graph.replay()
                assert_bitwise(snapshot(result), changed, "head wave changed-input replay")
                assert (storage[size:] == 0xA5).all(), "candidate workspace overrun"
                assert (baseline_storage[full_size:] == 0xA5).all(), "baseline workspace overrun"
    finally:
        stream.synchronize()
        graph.reset()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)


def test_checkpoint_wave_capacity_policy():
    gib = 1024**3
    budget = kda_engine._KDA_CHECKPOINT_WAVE_BYTES
    for heads in (1, 2, 3, 4, 8, 12, 16, 32, 64):
        for size in (0, budget, budget + 1, 2 * budget, 2 * budget + 1, 32 * gib):
            divisor = kda_engine._checkpoint_head_divisor(heads, size, 1024, 148)
            assert divisor in (1, 2, 4)
            assert heads % divisor == 0
            assert divisor == 1 or size > (divisor // 2) * budget
    assert kda_engine._checkpoint_head_divisor(32, 2 * budget + 1, 16, 148) == 4
    assert kda_engine._checkpoint_head_divisor(4, 2 * budget + 1, 16, 148) == 1
    assert kda_engine._checkpoint_head_divisor(16, 16 * gib, 16, 148) == 2


def test_checkpoint_wave_compile_count(monkeypatch):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.frost.kernel import kda_chain_backward_f16 as chain

    original_compile = cute.compile
    calls = []

    def record_compile(*args, **kwargs):
        calls.append(args[0])
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", record_compile)
    monkeypatch.setattr(chain, "chain_backward_cache", {})
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    try:
        for offset, threshold in ((0, 1024**3), (2, 0)):
            monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", threshold)
            for divisor in (2, 4):
                monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: divisor)
                for lengths in ([8193] + [17] * 33, [8193, 31] + [33] * 65, [12289] + [65] * 37):
                    inputs = pack(sequences("kda", lengths, heads=4))
                    inputs["cu_seqlens"] = inputs.pop("cu")
                    inputs["dO"] = inputs.pop("do")
                    inputs["d_final_state"] = inputs.pop("dfs")
                    execute, result, storage, size = _prepared(inputs, "bwd", False, handle)
                    execute()
                    assert all(torch.isfinite(value).all() for value in result.values())
                    assert (storage[size:] == 0xA5).all()
                    del execute, result, storage
                assert len(calls) == offset + (1 if divisor == 2 else 2), "only the bounded head divisor may specialize these shapes"
    finally:
        torch.cuda.synchronize()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("divisor", [2, 4])
@pytest.mark.parametrize("state_waves", [False, True], ids=["checkpoint-only", "all-states"])
def test_checkpoint_wave_runtime_heads(monkeypatch, divisor, state_waves):
    """A compiled wave must use the current head count, including non-power-of-two counts."""
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.frost.kernel import kda_chain_backward_f16 as chain

    monkeypatch.setattr(chain, "chain_backward_cache", {})
    monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", 0 if state_waves else 1024**3)
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    try:
        compiled = None
        for heads in (divisor, 3 * divisor, 2 * divisor, 4 * divisor):
            inputs = pack(sequences("kda", [0, 8193, 17, 513], heads=heads))
            inputs["cu_seqlens"] = inputs.pop("cu")
            inputs["dO"] = inputs.pop("do")
            inputs["d_final_state"] = inputs.pop("dfs")
            inputs["g"].fill_(-0.001)
            monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: 1)
            baseline_call, baseline_result, baseline_storage, full_size = _prepared(inputs, "bwd", False, handle)
            baseline_call()
            expected = snapshot(baseline_result)
            monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: divisor)
            execute, result, storage, size = _prepared(inputs, "bwd", False, handle)
            assert size < full_size
            storage[:size].fill_(0xFF)
            execute()
            assert_bitwise(snapshot(result), expected, f"runtime heads={heads}, divisor={divisor}")
            assert (storage[size:] == 0xA5).all()
            assert (baseline_storage[full_size:] == 0xA5).all()
            assert len(chain.chain_backward_cache) == 2, "head count must not specialize the baseline or wave"
            current = tuple(chain.chain_backward_cache.values())
            if compiled is not None:
                assert all(a is b for a, b in zip(current, compiled)), "runtime heads must reuse the same compiled launches"
            compiled = current
            del baseline_call, baseline_result, baseline_storage, execute, result, storage
    finally:
        torch.cuda.synchronize()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("state_waves", [False, True], ids=["checkpoint-only", "all-states"])
def test_checkpoint_wave_independent_streams(monkeypatch, state_waves):
    """Shared compiled code must not share queues or state across independent graph replays."""
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.frost.kernel import kda_chain_backward_f16 as chain

    monkeypatch.setattr(chain, "chain_backward_cache", {})
    monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", 0 if state_waves else 1024**3)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    handles = [cudnn.create_handle(), cudnn.create_handle()]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    plans = []
    try:
        for index, (stream, handle, graph) in enumerate(zip(streams, handles, graphs)):
            stream.wait_stream(torch.cuda.current_stream())
            cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
            with torch.cuda.stream(stream):
                inputs = pack(sequences("kda", [0, 1, 15, 16, 17, 8193] + [33] * (31 + index), heads=8))
                inputs["cu_seqlens"] = inputs.pop("cu")
                inputs["dO"] = inputs.pop("do")
                inputs["d_final_state"] = inputs.pop("dfs")
                inputs["g"].fill_(-0.001)
                if index:
                    inputs["dO"].neg_()
                original_inputs = snapshot(inputs)
                monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: 1)
                baseline_call, baseline_result, baseline_storage, full_size = _prepared(inputs, "bwd", False, handle)
                baseline_call()
                expected = snapshot(baseline_result)
                assert (baseline_storage[full_size:] == 0xA5).all()
                monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: 4)
                execute, result, storage, size = _prepared(inputs, "bwd", False, handle)
                execute()
                with torch.cuda.graph(graph, stream=stream):
                    execute()
                plans.append((execute, result, storage, size, expected, inputs, original_inputs))
                del baseline_call, baseline_result, baseline_storage
        assert len(chain.chain_backward_cache) == 2, "independent streams must reuse compiled code"
        a, b = (plan[2] for plan in plans)
        assert a.data_ptr() + a.numel() <= b.data_ptr() or b.data_ptr() + b.numel() <= a.data_ptr()
        for _ in range(8):
            for _ in range(8):
                for stream, graph, (_, result, storage, size, _, _, _) in zip(streams, graphs, plans):
                    with torch.cuda.stream(stream):
                        storage[:size].fill_(0xFF)
                        for value in result.values():
                            value.fill_(float("nan"))
                        graph.replay()
            # Submit both streams before waiting; comparisons must not serialize the replays.
            for stream in streams:
                stream.synchronize()
            for _, result, storage, size, expected, inputs, original_inputs in plans:
                assert_bitwise(snapshot(result), expected, "independent stream replay")
                assert_bitwise(snapshot(inputs), original_inputs, "read-only input ports")
                assert (storage[size:] == 0xA5).all()
    finally:
        for stream, graph, handle in zip(streams, graphs, handles):
            stream.synchronize()
            graph.reset()
            torch.cuda.current_stream().wait_stream(stream)
            cudnn.destroy_handle(handle)


def test_state_wave_warm_stress(monkeypatch):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from warm_stress import StressCase, assert_warm_stress, build_launchers

    monkeypatch.setattr(kda_engine, "multiprocessor_count", lambda device: 2)
    monkeypatch.setattr(kda_engine, "_checkpoint_head_divisor", lambda *args: 4)
    monkeypatch.setattr(kda_engine, "_KDA_STATE_WAVE_BYTES", 0)
    inputs = pack(sequences("kda", [265] * 32 + [8193], heads=4))
    inputs["cu_seqlens"] = inputs.pop("cu")
    inputs["dO"] = inputs.pop("do")
    inputs["d_final_state"] = inputs.pop("dfs")
    inputs["g"].fill_(-0.001)
    build_launchers()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            execute, result, storage, size = _prepared(inputs, "bwd", False, handle)
            execute()
            with torch.cuda.graph(captured, stream=stream):
                execute()
            report = assert_warm_stress(
                StressCase("KDA full head-wave reuse", captured.replay, result, thread_blocks=2, refills_per_launch=4),
                budget_s=300,
            )
            print(report.text())
            assert (storage[size:] == 0xA5).all()
    finally:
        stream.synchronize()
        captured.reset()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)
