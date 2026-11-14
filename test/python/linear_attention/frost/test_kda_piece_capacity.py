# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Length-rule workspace bounds, raw-byte parity and prepared replay."""

import itertools
import math

import cutlass.cute as cute
import pytest
import torch

import cudnn
from cudnn.linear_attention.frost.common import piece_chain
from cudnn.linear_attention.frost.common.split_k import WORK_ITEM_FIELDS
from cudnn.linear_attention.frost.kernel.kda_chain_prologue_f16 import run_chain_prologue

from determinism_utils import assert_bitwise, snapshot
from linear_attention.determinism import pack, sequences
from linear_attention.test_la import CUDNN_DTYPE, build_and_pin, frost_la_unavailable

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def supported_runtime():
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)


def test_piece_capacity_bounds_empty_and_threshold_lengths():
    """Include empty identity slots and both sides of the 8192-token rule."""
    lengths = (0, 1, 15, 16, 17, 8191, 8192, 8193, 131072, 131073)
    for batch in itertools.product(lengths, repeat=3):
        total = sum(batch)
        pieces = min(max(1, math.ceil(total / 8192)), 16)
        bound = piece_chain.piece_capacity(num_seqs=len(batch), pieces=pieces, total_tokens=total, batch_invariant=True)
        needed = sum(max(1, min(pieces, math.ceil(length / 8192))) for length in batch)
        assert needed <= bound <= len(batch) * pieces
        assert piece_chain.piece_capacity(num_seqs=len(batch), pieces=pieces, total_tokens=total, batch_invariant=False) == len(batch) * pieces
    assert piece_chain.piece_capacity(num_seqs=432, pieces=16, total_tokens=131072, batch_invariant=True) == 448


@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64], ids=["cu32", "cu64"])
def test_piece_prologue_capacity_canary_and_compile_count(monkeypatch, cu_dtype):
    """One symbolic prologue handles different batch/token/capacity sizes without touching a trailing canary."""
    compiled = []
    original = cute.compile

    def count_compile(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", count_compile)
    cache = {}
    # Include an all-empty batch, a saturated extra-slot bound and more sequences than one CTA scan round.
    for lengths in ([0, 0, 0], [0, 1, 8191, 8192, 8193, 17001], [0] * 9 + [16385], [0, 17, 131073] + [33] * 1025):
        batch, total = len(lengths), sum(lengths)
        pieces = min(max(1, math.ceil(total / 8192)), 16)
        capacity = piece_chain.piece_capacity(num_seqs=batch, pieces=pieces, total_tokens=total, batch_invariant=True)
        cu = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=cu_dtype)
        results = []
        for slots in (batch * pieces, capacity):
            table = piece_chain.piece_table_layout(batch, pieces, 1, capacity=slots)
            assert table.item_rows == slots
            storage = torch.full((slots + 1 + 32,), -12345, device="cuda", dtype=torch.int32)
            rows = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
            summaries = torch.empty_like(rows)
            work = torch.zeros((slots, WORK_ITEM_FIELDS), device="cuda", dtype=torch.int32)
            # TMA requires a nonzero physical extent even when cu_seqlens describes an entirely empty batch.
            # Public KDA does not select the chain for zero tokens; this direct prologue test supplies its backing storage.
            k = torch.empty((max(1, total), 1, 64), device="cuda", dtype=torch.bfloat16)
            run_chain_prologue(
                cache,
                pieces=pieces,
                unit_chunks=1,
                b_t=16,
                length_rule=True,
                heads_out=1,
                cu_seqlens=cu,
                cu_pieces=storage[: slots + 1],
                main_rows=rows,
                summary_rows=summaries,
                main_count=rows[-1:],
                summary_count=summaries[-1:],
                work_items=work,
                scheduler=torch.empty(6, device="cuda", dtype=torch.int32),
                k=k,
                gate=k,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            used = rows[-1].item()
            assert used <= slots
            assert (storage[slots + 1 :] == -12345).all(), "prologue wrote beyond its declared capacity"
            assert (storage[used : slots + 1] == total).all()
            results.append({"cu": storage[: used + 1].clone(), "rows": rows, "summary_rows": summaries, "work": work[:used].clone()})
        assert_bitwise(results[1], results[0], "compact piece table")
    assert len(compiled) == 1, "runtime sequence/token/capacity extents must remain symbolic"


def _prepared(inputs, phase, safe_gate, handle):
    """Build all storage before capture, including a canary after the exact workspace extent."""
    graph = cudnn.pygraph(handle=handle)
    names = {
        "fwd": ("q", "k", "v", "g", "beta", "initial_state"),
        "bwd": ("q", "k", "v", "g", "beta", "initial_state", "dO", "d_final_state"),
        "summary": ("k", "v", "g", "beta", "initial_state"),
        "summary_bwd": ("q", "k", "g", "beta", "dO", "d_final_state"),
    }[phase]
    names = (*names, "cu_seqlens", *(("a_log", "dt_bias") if safe_gate else ()))
    ports = {name: graph.tensor(list(inputs[name].shape), data_type=CUDNN_DTYPE[inputs[name].dtype], name=name) for name in names}
    attrs = dict(batch_invariant=True, safe_gate=safe_gate)
    if phase == "fwd":
        outputs = graph.kda(**ports, **attrs, output_final_state=True)
        dtypes = [inputs["q"].dtype, torch.float32, inputs["q"].dtype, torch.float32]
    elif phase == "bwd":
        outputs = graph.kda_bwd(**ports, **attrs)
        dtypes = [inputs[n].dtype for n in ("q", "k", "v", "g", "beta", "initial_state")] + [torch.float32, torch.float32]
    else:
        outputs = getattr(graph, "kda_" + phase)(**ports, **attrs, output_transition=True)
        dtypes = [torch.float32] * len(outputs)
    live = [(port, dtype) for port, dtype in zip(outputs, dtypes) if port is not None]
    for port, dtype in live:
        port.set_output(True).set_data_type(CUDNN_DTYPE[dtype])
    build_and_pin(graph, "kda_summary_frost" if phase.startswith("summary") else "kda_frost")
    result = {str(i): torch.empty(tuple(port.get_dim()), dtype=dtype, device="cuda") for i, (port, dtype) in enumerate(live)}
    variant_pack = {ports[name]: inputs[name] for name in names}
    variant_pack.update({port: result[str(i)] for i, (port, _) in enumerate(live)})
    size = graph.get_workspace_size()
    storage = torch.full((size + 256,), 0xA5, dtype=torch.uint8, device="cuda")
    workspace = storage[:size]

    def execute():
        graph.execute(variant_pack, workspace, handle=handle)

    return execute, result, storage, size


@pytest.mark.parametrize("phase", ["fwd", "bwd", "summary", "summary_bwd"])
@pytest.mark.parametrize(
    "dim,dtype,safe_gate",
    [(64, torch.bfloat16, False), (64, torch.float16, True), (128, torch.bfloat16, True), (128, torch.float16, False)],
    ids=["d64-bf16-log", "d64-fp16-safe", "d128-bf16-safe", "d128-fp16-log"],
)
def test_kda_shared_stage_reuse(phase, dim, dtype, safe_gate):
    """Small Racecheck target: wrap shared-memory rings, cross a tail and retain empty identity slots.

    Run under ``compute-sanitizer --tool racecheck --error-exitcode 99``. The original elected generic-memory
    handoffs report RAW/WAR hazards; every shared-memory reader/writer must publish its own barrier credit.
    The longer compaction tests separately exercise persistent work-item scheduling and capture/replay.
    """
    inputs = pack(sequences("kda", [265, 129, 0, 1], dtype=dtype, heads=1, key_dim=dim, value_dim=dim))
    inputs["cu_seqlens"] = inputs.pop("cu")
    inputs["dO"] = inputs.pop("do")
    inputs["d_final_state"] = inputs.pop("dfs")
    inputs["g"].fill_(-4 if safe_gate else -0.001)
    if safe_gate:
        inputs["a_log"] = torch.zeros(1, device="cuda")
        inputs["dt_bias"] = torch.zeros(1, dim, device="cuda")
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    try:
        execute, result, storage, size = _prepared(inputs, phase, safe_gate, handle)
        execute()
        expected = snapshot(result)
        assert all(torch.isfinite(value).all() for value in expected.values())
        storage[:size].fill_(0xFF)
        for value in result.values():
            value.fill_(float("nan"))
        execute()
        actual = snapshot(result)
        assert all(torch.isfinite(value).all() for value in actual.values())
        assert_bitwise(actual, expected, "shared-stage reuse")
        assert (storage[size:] == 0xA5).all(), "workspace overrun"
    finally:
        torch.cuda.synchronize()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("phase", ["fwd", "bwd", "summary", "summary_bwd"])
def test_kda_shared_stage_warm_stress(monkeypatch, phase):
    """Repeated native graph launches with raw-ring and scheduler-ring reuse.

    Twenty-four documents run through two persistent CTAs, forcing at least one to wrap the eight-slot ticket ring. Each document has
    seventeen chunks, also wrapping the raw operand ring. This deliberately small grid stresses repeated ownership
    changes; it is not an occupancy/performance measurement. Cold byte stability complements the separate FP64 tests.
    """
    from cudnn.linear_attention.frost import kda_engine
    from warm_stress import StressCase, assert_warm_stress, build_launchers

    monkeypatch.setattr(kda_engine, "multiprocessor_count", lambda device: 2)
    inputs = pack(sequences("kda", [265] * 24, dtype=torch.bfloat16, heads=1))
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
            execute, result, storage, size = _prepared(inputs, phase, False, handle)
            execute()
            with torch.cuda.graph(captured, stream=stream):
                execute()
            report = assert_warm_stress(StressCase(f"KDA {phase} shared-stage reuse", captured.replay, result, thread_blocks=2, refills_per_launch=4))
            print(report.text())
            assert (storage[size:] == 0xA5).all(), "workspace overrun"
    finally:
        stream.synchronize()
        captured.reset()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("phase", ["fwd", "bwd", "summary", "summary_bwd"])
@pytest.mark.parametrize(
    "dtype,safe_gate",
    [(torch.bfloat16, False), (torch.bfloat16, True), (torch.float16, False), (torch.float16, True)],
    ids=["bf16-log", "bf16-safe", "fp16-log", "fp16-safe"],
)
def test_compact_kda_matches_full_capacity_and_changed_replay(monkeypatch, phase, dtype, safe_gate):
    """Capacity only: every output/gradient/state stays byte-identical, even after poisoned-workspace graph replay."""
    lengths = [0, 1, 15, 16, 17, 8191, 8192, 8193, 17001] + [33] * 257
    inputs = pack(sequences("kda", lengths, dtype=dtype, heads=1))
    inputs["cu_seqlens"] = inputs.pop("cu")
    inputs["dO"] = inputs.pop("do")
    inputs["d_final_state"] = inputs.pop("dfs")
    inputs["g"].fill_(-4 if safe_gate else -0.001)
    if safe_gate:
        inputs["a_log"] = torch.zeros(1, device="cuda")
        inputs["dt_bias"] = torch.zeros(1, 64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            with monkeypatch.context() as patch:
                patch.setattr(piece_chain, "piece_capacity", lambda *, num_seqs, pieces, **kw: num_seqs * pieces)
                baseline, expected, baseline_storage, old_size = _prepared(inputs, phase, safe_gate, handle)
            actual_call, actual, storage, new_size = _prepared(inputs, phase, safe_gate, handle)
            assert new_size < old_size * 0.75, (old_size, new_size)
            baseline()
            actual_call()
            assert_bitwise(snapshot(actual), snapshot(expected), "full-capacity baseline")

            def unexpected_preparation(*args, **kwargs):
                pytest.fail("prepared execution allocated storage or compiled a kernel")

            with torch.cuda.graph(captured, stream=stream):
                old_debug = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    with monkeypatch.context() as patch:
                        for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like"):
                            patch.setattr(torch, name, unexpected_preparation)
                        patch.setattr(cute, "compile", unexpected_preparation)
                        actual_call()
                finally:
                    torch.cuda.set_sync_debug_mode(old_debug)
            replay_lengths = ([sum(lengths)] + [0] * (len(lengths) - 1), list(reversed(lengths)))
            for change, new_lengths in zip((0.125, -0.25), replay_lengths):
                inputs["q"].add_(change)
                inputs["v"].add_(change)
                inputs["dO"].add_(change)
                # Same declared shapes, but first consume the entire bound: one long document plus empty identities.
                # Then restore the ragged mixture in reverse order, keeping the captured metadata address stable.
                inputs["cu_seqlens"].copy_(torch.tensor([0, *itertools.accumulate(new_lengths)], device="cuda", dtype=torch.int32))
                baseline()
                storage[:new_size].fill_(0xFF)
                for value in actual.values():
                    value.fill_(float("nan"))
                captured.replay()
                assert_bitwise(snapshot(actual), snapshot(expected), "changed-input replay")
                assert (storage[new_size:] == 0xA5).all(), "compact workspace overrun"
                assert (baseline_storage[old_size:] == 0xA5).all(), "baseline workspace overrun"
    finally:
        stream.synchronize()
        captured.reset()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("cadence", [16, 64], ids=["chunk-checkpoints", "coarse-checkpoints"])
def test_compact_kda_checkpoint_autograd(monkeypatch, dtype, cadence):
    """Caller checkpoints retain all gradient bytes, including grouped heads and unequal K/V dimensions."""
    from cudnn.linear_attention.ops import kda, kimi_delta_attention

    inputs = pack(sequences("kda", [0, 17, 8193, 17001], dtype=dtype, heads=1, value_heads=2, key_dim=64, value_dim=128))
    inputs["g"].fill_(-0.001)
    names = ("q", "k", "v", "g", "beta", "initial_state")

    def call():
        kda.fprop_cache.clear()
        kda.bprop_cache.clear()
        leaves = {name: inputs[name].detach().clone().requires_grad_() for name in names}
        out, state, _ = kimi_delta_attention(
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
        return snapshot({"out": out, "state": state, **dict(zip(names, grads))})

    with monkeypatch.context() as patch:
        patch.setattr(piece_chain, "piece_capacity", lambda *, num_seqs, pieces, **kw: num_seqs * pieces)
        expected = call()
    assert_bitwise(call(), expected, "compact checkpoint autograd")
