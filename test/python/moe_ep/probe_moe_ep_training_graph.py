#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed-resource SM107 multi-rank CUDA Graph communication probe.

Run from the project root on one node with two Rubin GPUs. The container
launcher selects an architecture-native Python/PyTorch environment::

    data/script/run_moe_ep_bf16_combine_container.sh \
      --my-version-root "$PWD/my-version" \
      training-graph-probe

Set ``MOE_EP_GRAPH_PROBE_NPROC=4`` (or another local EP size) on the host to
reuse the same probe beyond EP2.

The probe exercises only the public fixed-resource ordinary/capture path,
including fixed-address staging/reset operations, forward/backward CuTeDSL
callables, and a one-scalar NCCL overflow OR.
"""

from __future__ import annotations

import argparse
import gc
import os
import socket
import time
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

from cudnn import MoeEp, MoeEpTrainingWeights
from cudnn.moe_ep._megamoe_backend.mxfp8._adapter import (
    _quantize_plain_mxfp8,
)
from cudnn.moe_ep._megamoe_backend._runtime import (
    _RuntimeWatchdog,
    get_runtime_manager,
)


def _debug_phase(rank: int, phase: str) -> None:
    if os.environ.get("MOE_EP_DEBUG_RUNTIME", "0") != "1":
        return
    print(
        "[moe-ep-probe] "
        f"time={time.monotonic():.6f} host={socket.gethostname()} "
        f"pid={os.getpid()} rank={rank} phase={phase}",
        flush=True,
    )


@contextmanager
def _debug_phase_scope(rank: int, phase: str):
    _debug_phase(rank, f"{phase}.begin")
    try:
        yield
    finally:
        _debug_phase(rank, f"{phase}.end")


def _synchronize_with_watchdog(
    rank: int,
    device: torch.device,
    phase: str,
) -> None:
    watchdog = _RuntimeWatchdog(phase)
    watchdog.start()
    _debug_phase(rank, f"{phase}.begin")
    try:
        torch.cuda.synchronize(device)
    finally:
        watchdog.close()
    _debug_phase(rank, f"{phase}.end")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-replays", type=int, default=2)
    parser.add_argument("--burst-replays", type=int, default=100)
    parser.add_argument(
        "--multistream-replays",
        type=int,
        default=10,
        help=(
            "two-lane cross-stream graph replays; use a larger value such as "
            "100 for dedicated stress runs"
        ),
    )
    parser.add_argument(
        "--max-recv-size-per-rank",
        type=int,
        default=1,
        help=(
            "bounded receive capacity; must remain below the forced-overflow "
            "route count so the probe retains overflow coverage"
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help=(
            "create/capture/destroy cycles; the first is exhaustive and later "
            "cycles use a minimal replay to verify teardown/re-init"
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--skip-multistream",
        action="store_true",
        help="skip the two-lane ordered cross-stream resource probe",
    )
    parser.add_argument(
        "--expect-overflow-assert",
        action="store_true",
        help=(
            "run only the fatal drop_on_overflow=False graph assertion probe; "
            "success requires every rank to observe the expected CUDA error"
        ),
    )
    return parser.parse_args()


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _assert_replay_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Compare graph replay outputs with dtype-appropriate semantics."""

    low_precision = {
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
        torch.uint8,
        torch.int32,
        torch.int64,
    }
    if actual.dtype in low_precision:
        if not torch.equal(actual, expected):
            raise AssertionError(
                f"{name} is not bitwise equal after graph replay"
            )
        return
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
        msg=f"{name} differs after graph replay",
    )


def _make_inputs(
    rank: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    token_count = 8
    hidden = 128
    intermediate = 256
    experts_per_rank = 2
    top_k = 2
    generator = torch.Generator(device=device).manual_seed(20260828 + rank)

    activation = (
        torch.randn(
            token_count,
            hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 8
    )
    fc1_weight = (
        torch.randn(
            experts_per_rank,
            hidden,
            2 * intermediate,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 16
    )
    fc2_weight = (
        torch.randn(
            experts_per_rank,
            intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 16
    )
    topk_idx = torch.full(
        (token_count, top_k),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.zeros(
        (token_count, top_k),
        dtype=torch.float32,
        device=device,
    )
    # Exactly one route is received by each rank during eager warmup, so every
    # positive max_recv_size_per_rank remains within capacity.
    topk_idx[0, 0] = rank * experts_per_rank
    topk_weights[0, 0] = 1.0
    grad_output = (
        torch.randn(
            token_count,
            hidden,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 8
    )
    return (
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
    ), grad_output


def _route_pattern(
    kind: str,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = 8
    top_k = 2
    experts_per_rank = 2
    indices = torch.full(
        (token_count, top_k),
        -1,
        dtype=torch.int32,
        device=device,
    )
    weights = torch.zeros(
        (token_count, top_k),
        dtype=torch.float32,
        device=device,
    )
    if kind == "local":
        indices[0, 0] = rank * experts_per_rank
        weights[0, 0] = 1.0
    elif kind == "remote":
        peer = (rank + 1) % world_size
        indices[0, 0] = peer * experts_per_rank
        weights[0, 0] = 1.0
    elif kind == "overflow":
        # Every source sends all routes to rank 0. Its raw receive count is
        # therefore much larger than max_recv_size_per_rank=1.
        indices.fill_(0)
        weights.fill_(0.5)
    else:
        raise ValueError(f"unknown route pattern {kind!r}")
    return indices, weights


def _make_two_slot_inputs(
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    tuple[torch.Tensor, ...],
    torch.Tensor,
    tuple[torch.Tensor, ...],
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    args0, grad0 = _make_inputs(rank, device)
    local = _route_pattern("local", rank, world_size, device)
    remote = _route_pattern("remote", rank, world_size, device)
    args0 = (*args0[:3], local[0].clone(), local[1].clone())
    args1 = (
        args0[0].clone(),
        args0[1],
        args0[2],
        remote[0].clone(),
        remote[1].clone(),
    )
    return args0, grad0, args1, grad0.clone(), local, remote


def _make_training_weights(
    args: tuple[torch.Tensor, ...],
) -> MoeEpTrainingWeights:
    return MoeEpTrainingWeights(
        forward_fc1=_quantize_plain_mxfp8(args[1], axis=1),
        forward_fc2=_quantize_plain_mxfp8(args[2], axis=1),
        backward_w2_transpose=_quantize_plain_mxfp8(
            args[2].transpose(1, 2).contiguous(),
            axis=1,
        ),
        backward_w1_transpose=_quantize_plain_mxfp8(
            args[1].transpose(1, 2).contiguous(),
            axis=1,
        ),
    )


def _make_operator(
    *,
    world_size: int,
    group,
    max_recv_size_per_rank: int,
    drop_on_overflow: bool,
) -> MoeEp:
    return MoeEp(
        num_experts=2 * world_size,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        ep_group=group,
        max_tokens_per_rank=8,
        max_recv_size_per_rank=max_recv_size_per_rank,
        drop_on_overflow=drop_on_overflow,
        combine_format="bf16",
    )


def _close_probe_operator(
    *,
    device: torch.device,
    group,
    op: MoeEp,
) -> None:
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    op.close()
    gc.collect()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)


def _run_training_resource_probe(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group,
    diagnostic_replays: int,
    burst_replays: int,
    max_recv_size_per_rank: int,
    full_probe: bool,
) -> None:
    """Exercise full graph behavior or a minimal teardown/re-init replay."""

    args0, grad0, args1, grad1, local, remote = _make_two_slot_inputs(
        rank,
        world_size,
        device,
    )
    # Keep immutable baseline patterns separate from the graph-bound input
    # tensors. Overflow injection mutates the latter in place.
    weights = _make_training_weights(args0)
    op = _make_operator(
        world_size=world_size,
        group=group,
        max_recv_size_per_rank=max_recv_size_per_rank,
        drop_on_overflow=True,
    )
    graph = None
    try:
        resources = op.prepare_training_resources(
            weights,
            slot_count=2,
            lane_count=1,
        )
        slot0, slot1 = resources.slots
        lane0 = resources.lanes[0]

        # Ordinary execution is the collective warmup for all fused staging,
        # MegaMoE, and fixed-capacity WGrad export compile caches.
        resources.refresh_weights()
        y0 = resources.forward(
            slot0,
            lane0,
            args0[0],
            args0[3],
            args0[4],
        )
        y1 = resources.forward(
            slot1,
            lane0,
            args1[0],
            args1[3],
            args1[4],
        )
        dx0, dp0, operands0 = resources.backward(slot0, lane0, grad0)
        dx1, dp1, operands1 = resources.backward(slot1, lane0, grad1)
        overflow_status = resources.finalize_overflow((slot0, slot1))
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        if int(overflow_status.item()) != 0:
            raise AssertionError("ordinary fixed-resource warmup overflowed")

        comparison_names = (
            "y0",
            "y1",
            "dx0",
            "dx1",
            "dprob0",
            "dprob1",
            "slot0.fc1_a",
            "slot0.fc1_b",
            "slot0.fc2_a",
            "slot0.fc2_b",
            "slot1.fc1_a",
            "slot1.fc1_b",
            "slot1.fc2_a",
            "slot1.fc2_b",
        )
        ordinary = {
            name: tensor.clone()
            for name, tensor in zip(
                comparison_names,
                (
                y0,
                y1,
                dx0,
                dx1,
                dp0,
                dp1,
                operands0.fc1_a,
                operands0.fc1_b,
                operands0.fc2_a,
                operands0.fc2_b,
                operands1.fc1_a,
                operands1.fc1_b,
                operands1.fc2_a,
                operands1.fc2_b,
                ),
            )
        }
        ordinary_offsets = (
            operands0.expert_offsets.clone(),
            operands1.expert_offsets.clone(),
        )

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            resources.refresh_weights()
            graph_y0 = resources.forward(
                slot0,
                lane0,
                args0[0],
                args0[3],
                args0[4],
            )
            graph_y1 = resources.forward(
                slot1,
                lane0,
                args1[0],
                args1[3],
                args1[4],
            )
            graph_dx0, graph_dp0, graph_operands0 = resources.backward(
                slot0,
                lane0,
                grad0,
            )
            graph_dx1, graph_dp1, graph_operands1 = resources.backward(
                slot1,
                lane0,
                grad1,
            )
            graph_overflow = resources.finalize_overflow((slot0, slot1))
        dist.barrier(group=group)

        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        dist.barrier(group=group)
        if int(graph_overflow.item()) != 0:
            raise AssertionError("captured fixed-resource graph overflowed")

        captured = {
            name: tensor
            for name, tensor in zip(
                comparison_names,
                (
                    graph_y0,
                    graph_y1,
                    graph_dx0,
                    graph_dx1,
                    graph_dp0,
                    graph_dp1,
                    graph_operands0.fc1_a,
                    graph_operands0.fc1_b,
                    graph_operands0.fc2_a,
                    graph_operands0.fc2_b,
                    graph_operands1.fc1_a,
                    graph_operands1.fc1_b,
                    graph_operands1.fc2_a,
                    graph_operands1.fc2_b,
                ),
            )
        }
        for name in comparison_names:
            _assert_replay_tensor(name, captured[name], ordinary[name])
        torch.testing.assert_close(
            graph_operands0.expert_offsets,
            ordinary_offsets[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            graph_operands1.expert_offsets,
            ordinary_offsets[1],
            rtol=0,
            atol=0,
        )

        if full_probe:
            # Diagnostic mode aligns ranks after every replay and verifies that
            # fixed-slot dprob reset prevents history accumulation.
            dprob_reference = graph_dp0.clone()
            for _ in range(diagnostic_replays):
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                dist.barrier(group=group)
                if int(graph_overflow.item()) != 0:
                    raise AssertionError(
                        "fixed-resource diagnostic replay overflowed"
                    )
                torch.testing.assert_close(
                    graph_dp0,
                    dprob_reference,
                    rtol=1e-5,
                    atol=1e-6,
                )

            # Production-like burst: no synchronization or host collective in
            # the loop. The graph contains the captured scalar overflow OR.
            with torch.cuda.stream(stream):
                for _ in range(burst_replays):
                    graph.replay()
            stream.synchronize()
            dist.barrier(group=group)
            if int(graph_overflow.item()) != 0:
                raise AssertionError("fixed-resource replay burst overflowed")
            torch.testing.assert_close(
                graph_dp0,
                ordinary["dprob0"],
                rtol=1e-5,
                atol=1e-6,
            )

            # Overflow both slots, then restore their distinct valid patterns.
            overflow = _route_pattern("overflow", rank, world_size, device)
            with torch.cuda.stream(stream):
                args0[3].copy_(overflow[0])
                args0[4].copy_(overflow[1])
                args1[3].copy_(overflow[0])
                args1[4].copy_(overflow[1])
                graph.replay()
            stream.synchronize()
            dist.barrier(group=group)
            if int(graph_overflow.item()) != 1:
                raise AssertionError("fixed-resource overflow was not global")

            with torch.cuda.stream(stream):
                args0[3].copy_(local[0])
                args0[4].copy_(local[1])
                args1[3].copy_(remote[0])
                args1[4].copy_(remote[1])
                graph.replay()
            stream.synchronize()
            dist.barrier(group=group)
            recovered_overflow = int(graph_overflow.item())
            if recovered_overflow != 0:
                raise AssertionError(
                    "fixed-resource graph did not recover: "
                    f"rank={rank}, global_overflow={recovered_overflow}, "
                    f"slot0_routing_restored="
                    f"{torch.equal(args0[3], local[0])}, "
                    f"slot1_routing_restored="
                    f"{torch.equal(args1[3], remote[0])}"
                )

        if rank == 0:
            mode = "full" if full_probe else "reinit"
            effective_burst = burst_replays if full_probe else 0
            print(
                f"MOE_EP_EP{world_size}_TRAINING_RESOURCES_GRAPH_PASS "
                f"mode={mode} burst={effective_burst}",
                flush=True,
            )
    finally:
        if graph is not None:
            del graph
        _close_probe_operator(device=device, group=group, op=op)


def _run_multistream_resource_probe(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group,
    replays: int,
    max_recv_size_per_rank: int,
) -> None:
    """Capture two independent lanes with deterministic cross-rank ordering."""

    args0, grad0, args1, grad1, _, _ = _make_two_slot_inputs(
        rank,
        world_size,
        device,
    )
    op = _make_operator(
        world_size=world_size,
        group=group,
        max_recv_size_per_rank=max_recv_size_per_rank,
        drop_on_overflow=True,
    )
    graph = None
    try:
        with _debug_phase_scope(rank, "multistream.prepare"):
            resources = op.prepare_training_resources(
                _make_training_weights(args0),
                slot_count=2,
                lane_count=2,
            )
        slot0, slot1 = resources.slots
        lane0, lane1 = resources.lanes
        with _debug_phase_scope(rank, "multistream.refresh-weights"):
            resources.refresh_weights()

        with _debug_phase_scope(rank, "multistream.lane0-forward"):
            eager_y0 = resources.forward(
                slot0, lane0, args0[0], args0[3], args0[4]
            )
        with _debug_phase_scope(rank, "multistream.lane0-backward"):
            eager_dx0, eager_dp0, _ = resources.backward(
                slot0,
                lane0,
                grad0,
            )
        with _debug_phase_scope(rank, "multistream.lane0-finalize"):
            resources.finalize_overflow((slot0,), lane0)
        _synchronize_with_watchdog(
            rank,
            device,
            "multistream.lane0-synchronize",
        )
        with _debug_phase_scope(rank, "multistream.lane0-barrier"):
            dist.barrier(group=group)

        with _debug_phase_scope(rank, "multistream.lane1-forward"):
            eager_y1 = resources.forward(
                slot1, lane1, args1[0], args1[3], args1[4]
            )
        with _debug_phase_scope(rank, "multistream.lane1-backward"):
            eager_dx1, eager_dp1, _ = resources.backward(
                slot1,
                lane1,
                grad1,
            )
        with _debug_phase_scope(rank, "multistream.lane1-finalize"):
            resources.finalize_overflow((slot1,), lane1)
        _synchronize_with_watchdog(
            rank,
            device,
            "multistream.lane1-synchronize",
        )
        with _debug_phase_scope(rank, "multistream.lane1-barrier"):
            dist.barrier(group=group)
        expected = tuple(
            tensor.clone()
            for tensor in (
                eager_y0,
                eager_dx0,
                eager_dp0,
                eager_y1,
                eager_dx1,
                eager_dp1,
            )
        )

        capture_stream = torch.cuda.Stream(device=device)
        lane_stream0 = torch.cuda.Stream(device=device)
        lane_stream1 = torch.cuda.Stream(device=device)
        fork_event = torch.cuda.Event()
        done_event0 = torch.cuda.Event()
        done_event1 = torch.cuda.Event()
        capture_stream.wait_stream(torch.cuda.current_stream(device))

        # One outer graph visits two lane-bound streams, rejoins them, then
        # emits exactly one NCCL overflow finalizer. The MegaMoE kernels use
        # device-side cross-rank software synchronization and consume one CTA
        # slot per SM. Launching both lanes concurrently can let different
        # ranks schedule different lanes first, leaving each lane waiting for
        # peers whose matching kernel cannot be scheduled. Chain lane 1 after
        # lane 0 so every rank observes the same collective order while still
        # validating independent per-stream lane storage and graph edges.
        graph = torch.cuda.CUDAGraph()
        capture_watchdog = _RuntimeWatchdog("multistream.capture")
        capture_watchdog.start()
        with _debug_phase_scope(rank, "multistream.capture"):
            try:
                with torch.cuda.graph(graph, stream=capture_stream):
                    fork_event.record(capture_stream)
                    lane_stream0.wait_event(fork_event)
                    with torch.cuda.stream(lane_stream0):
                        graph_y0 = resources.forward(
                            slot0,
                            lane0,
                            args0[0],
                            args0[3],
                            args0[4],
                        )
                        graph_dx0, graph_dp0, _ = resources.backward(
                            slot0,
                            lane0,
                            grad0,
                        )
                        done_event0.record(lane_stream0)
                    lane_stream1.wait_event(done_event0)
                    with torch.cuda.stream(lane_stream1):
                        graph_y1 = resources.forward(
                            slot1,
                            lane1,
                            args1[0],
                            args1[3],
                            args1[4],
                        )
                        graph_dx1, graph_dp1, _ = resources.backward(
                            slot1,
                            lane1,
                            grad1,
                        )
                        done_event1.record(lane_stream1)
                    capture_stream.wait_event(done_event1)
                    graph_overflow = resources.finalize_overflow(
                        (slot0, slot1),
                        lane0,
                    )
            finally:
                capture_watchdog.close()
        with _debug_phase_scope(rank, "multistream.capture-barrier"):
            dist.barrier(group=group)

        with _debug_phase_scope(rank, "multistream.replay"):
            with torch.cuda.stream(capture_stream):
                for _ in range(replays):
                    graph.replay()
            replay_watchdog = _RuntimeWatchdog(
                "multistream.replay-synchronize"
            )
            replay_watchdog.start()
            with _debug_phase_scope(
                rank,
                "multistream.replay-synchronize",
            ):
                try:
                    capture_stream.synchronize()
                finally:
                    replay_watchdog.close()
            with _debug_phase_scope(rank, "multistream.replay-barrier"):
                dist.barrier(group=group)
        if int(graph_overflow.item()) != 0:
            raise AssertionError("multi-stream fixed-resource graph overflowed")

        actual = (
            graph_y0,
            graph_dx0,
            graph_dp0,
            graph_y1,
            graph_dx1,
            graph_dp1,
        )
        for index, (value, reference) in enumerate(zip(actual, expected)):
            _assert_replay_tensor(
                f"multistream[{index}]",
                value,
                reference,
            )
        if rank == 0:
            print(
                f"MOE_EP_EP{world_size}_MULTISTREAM_GRAPH_PASS "
                f"replays={replays}",
                flush=True,
            )
    finally:
        if graph is not None:
            del graph
        _close_probe_operator(device=device, group=group, op=op)


def _run_error_mode_assert_probe(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group,
    max_recv_size_per_rank: int,
) -> None:
    """Require a captured global overflow to assert on every rank."""

    args, grad_output = _make_inputs(rank, device)
    local = _route_pattern("local", rank, world_size, device)
    overflow = _route_pattern("overflow", rank, world_size, device)
    route_indices = local[0].clone()
    route_weights = local[1].clone()
    op = _make_operator(
        world_size=world_size,
        group=group,
        max_recv_size_per_rank=max_recv_size_per_rank,
        drop_on_overflow=False,
    )
    resources = op.prepare_training_resources(
        _make_training_weights(args),
        slot_count=1,
        lane_count=1,
    )
    slot = resources.slots[0]
    lane = resources.lanes[0]

    # Warm every kernel and prove the assertion accepts a valid execution.
    resources.refresh_weights()
    resources.forward(
        slot,
        lane,
        args[0],
        route_indices,
        route_weights,
    )
    resources.backward(slot, lane, grad_output)
    resources.finalize_overflow((slot,), lane)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        resources.refresh_weights()
        resources.forward(
            slot,
            lane,
            args[0],
            route_indices,
            route_weights,
        )
        resources.backward(slot, lane, grad_output)
        resources.finalize_overflow((slot,), lane)
    dist.barrier(group=group)

    # A valid replay confirms capture before intentionally poisoning the
    # context with the fatal error-mode assertion.
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    dist.barrier(group=group)

    try:
        with torch.cuda.stream(stream):
            route_indices.copy_(overflow[0])
            route_weights.copy_(overflow[1])
            graph.replay()
        stream.synchronize()
    except BaseException as exc:
        print(
            f"MOE_EP_EP{world_size}_ERROR_MODE_ASSERT_PASS "
            f"rank={rank} error={type(exc).__name__}",
            flush=True,
        )
        # CUDA device assertions poison the process context. Do not run Python
        # destructors, NCCL collectives, or NVSHMEM finalization afterward.
        os._exit(0)

    print(
        f"MOE_EP_EP{world_size}_ERROR_MODE_ASSERT_MISSING rank={rank}",
        flush=True,
    )
    os._exit(1)


def main() -> None:
    args = _parse_args()
    _require_positive("diagnostic_replays", args.diagnostic_replays)
    _require_positive("burst_replays", args.burst_replays)
    _require_positive("multistream_replays", args.multistream_replays)
    _require_positive("cycles", args.cycles)
    _require_positive(
        "max_recv_size_per_rank",
        args.max_recv_size_per_rank,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size < 2:
        raise RuntimeError(
            f"this probe requires WORLD_SIZE >= 2, got {world_size}"
        )
    forced_overflow_routes = world_size * 8 * 2
    if args.max_recv_size_per_rank >= forced_overflow_routes:
        raise ValueError(
            "max_recv_size_per_rank must remain below the probe's forced "
            f"overflow route count {forced_overflow_routes}, got "
            f"{args.max_recv_size_per_rank}"
        )

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 7):
        raise RuntimeError(
            "this probe requires Rubin SM107; "
            f"rank {rank} found compute capability {capability}"
        )
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    try:
        if args.expect_overflow_assert:
            _run_error_mode_assert_probe(
                rank=rank,
                world_size=world_size,
                device=device,
                group=dist.group.WORLD,
                max_recv_size_per_rank=args.max_recv_size_per_rank,
            )
            raise AssertionError("fatal overflow assertion probe returned")
        for cycle in range(args.cycles):
            with _debug_phase_scope(
                rank,
                f"training-resources-cycle-{cycle}",
            ):
                _run_training_resource_probe(
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    group=dist.group.WORLD,
                    diagnostic_replays=args.diagnostic_replays,
                    burst_replays=args.burst_replays,
                    max_recv_size_per_rank=args.max_recv_size_per_rank,
                    full_probe=cycle == 0,
                )
        if not args.skip_multistream:
            with _debug_phase_scope(rank, "multistream"):
                _run_multistream_resource_probe(
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    group=dist.group.WORLD,
                    replays=args.multistream_replays,
                    max_recv_size_per_rank=args.max_recv_size_per_rank,
                )
        if rank == 0:
            print(
                f"MOE_EP_EP{world_size}_CUDA_GRAPH_PROBE_PASS",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            try:
                with _debug_phase_scope(rank, "runtime-shutdown"):
                    get_runtime_manager().shutdown()
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
