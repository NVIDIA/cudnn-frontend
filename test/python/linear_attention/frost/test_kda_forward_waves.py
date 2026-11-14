# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward head waves preserve state, output and capture bytes with bounded scratch."""

from itertools import accumulate

import pytest
import torch

import cudnn
from cudnn.linear_attention.frost import kda_engine
from determinism_utils import assert_bitwise, snapshot
from linear_attention.determinism import pack, sequences
from linear_attention.frost.test_kda_piece_capacity import _prepared
from linear_attention.test_la import frost_la_unavailable

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.parametrize("divisor", [2, 4, 8])
@pytest.mark.parametrize(
    "dtype,safe,key_dim,value_dim,head_pattern",
    [(torch.bfloat16, False, 64, 64, "equal"), (torch.bfloat16, True, 128, 64, "gqa"), (torch.float16, False, 64, 128, "gva")],
    ids=["bf16-log-d64", "bf16-safe-k128-v64", "fp16-log-k64-v128"],
)
def test_forward_waves_bitwise_and_capture(monkeypatch, divisor, dtype, safe, key_dim, value_dim, head_pattern):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    lengths = [0, 1, 15, 16, 17, 8193, 513, 33]
    inputs = pack(sequences("kda", lengths, dtype=dtype, heads=16, key_dim=key_dim, value_dim=value_dim))
    if head_pattern == "gqa":
        inputs["q"] = inputs["q"][:, :8].contiguous()
        inputs["k"] = inputs["k"][:, :8].contiguous()
    elif head_pattern == "gva":
        inputs["v"] = inputs["v"][:, :8].contiguous()
    inputs["cu_seqlens"] = inputs.pop("cu")
    inputs.pop("do")
    inputs.pop("dfs")
    inputs["g"].fill_(-4 if safe else -0.001)
    if safe:
        inputs["a_log"] = torch.zeros(16, device="cuda")
        inputs["dt_bias"] = torch.zeros(16, key_dim, device="cuda")
    handle = cudnn.create_handle()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: 1)
            baseline, baseline_result, baseline_storage, full_size = _prepared(inputs, "fwd", safe, handle)
            baseline()
            expected = snapshot(baseline_result)
            assert all(torch.isfinite(value).all() for value in expected.values())
            original_inputs = snapshot(inputs)
            monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: divisor)
            execute, result, storage, size = _prepared(inputs, "fwd", safe, handle)
            assert size < full_size
            execute()
            assert_bitwise(snapshot(result), expected, "forward head wave")
            with torch.cuda.graph(graph, stream=stream):
                execute()
            for _ in range(8):
                storage[:size].fill_(0xFF)
                for value in result.values():
                    value.fill_(float("nan"))
                graph.replay()
                assert_bitwise(snapshot(result), expected, "forward wave poisoned replay")
                assert (storage[size:] == 0xA5).all()
            assert_bitwise(snapshot(inputs), original_inputs, "read-only forward inputs")
            for changed in ([sum(lengths)] + [0] * (len(lengths) - 1), list(reversed(lengths))):
                inputs["cu_seqlens"].copy_(torch.tensor([0, *accumulate(changed)], device="cuda", dtype=torch.int32))
                inputs["v"].neg_()
                inputs["g"].mul_(0.8)
                baseline()
                expected = snapshot(baseline_result)
                assert all(torch.isfinite(value).all() for value in expected.values())
                storage[:size].fill_(0xAB)
                graph.replay()
                assert_bitwise(snapshot(result), expected, "forward wave changed metadata")
                assert (storage[size:] == 0xA5).all()
                assert (baseline_storage[full_size:] == 0xA5).all()
    finally:
        stream.synchronize()
        graph.reset()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("divisor", [2, 4, 8])
def test_forward_wave_runtime_heads(monkeypatch, divisor):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.frost.kernel import kda_chain_forward_f16 as chain

    monkeypatch.setattr(chain, "chain_forward_cache", {})
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    compiled = None
    try:
        for heads in (divisor, 3 * divisor, 96):
            inputs = pack(sequences("kda", [0, 8193, 17, 513], heads=heads))
            inputs["cu_seqlens"] = inputs.pop("cu")
            inputs["g"].fill_(-0.001)
            monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: 1)
            baseline, expected, baseline_storage, full_size = _prepared(inputs, "fwd", False, handle)
            baseline()
            monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: divisor)
            execute, result, storage, size = _prepared(inputs, "fwd", False, handle)
            storage[:size].fill_(0xFF)
            execute()
            assert_bitwise(snapshot(result), snapshot(expected), f"runtime forward H{heads}")
            assert size < full_size
            assert (storage[size:] == 0xA5).all()
            assert (baseline_storage[full_size:] == 0xA5).all()
            assert len(chain.chain_forward_cache) == 2
            current = tuple(chain.chain_forward_cache.values())
            if compiled is not None:
                assert all(a is b for a, b in zip(current, compiled)), "runtime heads must reuse compiled launches"
            compiled = current
            del baseline, expected, baseline_storage, execute, result, storage
    finally:
        torch.cuda.synchronize()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("cadence", [16, 64], ids=["chunk", "coarse"])
def test_forward_wave_checkpoint_autograd(monkeypatch, dtype, cadence):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from cudnn.linear_attention.ops import kda, kimi_delta_attention

    inputs = pack(sequences("kda", [0, 17, 8193, 513], dtype=dtype, heads=4, value_heads=8, key_dim=64, value_dim=128))
    inputs["g"].fill_(-0.001)
    names = ("q", "k", "v", "g", "beta", "initial_state")

    def call():
        kda.fprop_cache.clear()
        kda.bprop_cache.clear()
        leaves = {name: inputs[name].detach().clone().requires_grad_() for name in names}
        out, state, checkpoints = kimi_delta_attention(
            *(leaves[name] for name in names[:5]),
            inputs["cu"],
            initial_state=leaves["initial_state"],
            output_final_state=True,
            batch_invariant=True,
            checkpoint_every_n_tokens=cadence,
            checkpoint_dtype=torch.float32,
            plan_name="kda_frost",
        )
        grads = torch.autograd.grad((out, state), tuple(leaves.values()), (inputs["do"], inputs["dfs"]))
        return snapshot({"out": out, "state": state, "checkpoints": checkpoints, **dict(zip(names, grads))})

    monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: 1)
    expected = call()
    monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: 8)
    assert_bitwise(call(), expected, "forward waves retain public checkpoints and all gradients")


def test_forward_wave_warm_stress(monkeypatch):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    from warm_stress import StressCase, assert_warm_stress, build_launchers

    monkeypatch.setattr(kda_engine, "multiprocessor_count", lambda device: 2)
    inputs = pack(sequences("kda", [265] * 32 + [8193], heads=16))
    inputs["cu_seqlens"] = inputs.pop("cu")
    inputs["g"].fill_(-0.001)
    build_launchers()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: 1)
            baseline, baseline_result, baseline_storage, full_size = _prepared(inputs, "fwd", False, handle)
            baseline()
            expected = snapshot(baseline_result)
            monkeypatch.setattr(kda_engine, "_forward_head_divisor", lambda *args: 2)
            execute, result, storage, size = _prepared(inputs, "fwd", False, handle)
            execute()
            with torch.cuda.graph(captured, stream=stream):
                execute()
            report = assert_warm_stress(
                StressCase("KDA forward head-wave reuse", captured.replay, result, thread_blocks=2, refills_per_launch=2, reference=expected),
                budget_s=300,
            )
            print(report.text())
            assert (storage[size:] == 0xA5).all()
            assert (baseline_storage[full_size:] == 0xA5).all()
    finally:
        stream.synchronize()
        captured.reset()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)
