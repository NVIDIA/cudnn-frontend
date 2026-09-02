# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Collective coordination and timing helpers for explicit MoeEP sweeps."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
import torch.distributed as dist

from ._tuning import MoeEpAutotuneCandidateResult, MoeEpTuningConfig
from ._types import MoeEpTrainingBackwardOutputs, MoeEpTrainingForwardOutputs

_T = TypeVar("_T")
_MAX_AUTOTUNE_CANDIDATES = 32


def normalize_candidates(
    baseline: MoeEpTuningConfig,
    candidates: Sequence[MoeEpTuningConfig],
    *,
    warmup_iters: int,
    timed_iters: int,
    max_candidates: int,
) -> tuple[MoeEpTuningConfig, ...]:
    """Validate, de-duplicate, and prepend the current configuration."""

    if (
        isinstance(warmup_iters, bool)
        or not isinstance(warmup_iters, int)
        or warmup_iters < 0
    ):
        raise ValueError(
            f"warmup_iters must be a non-negative integer, got {warmup_iters!r}"
        )
    if (
        isinstance(timed_iters, bool)
        or not isinstance(timed_iters, int)
        or timed_iters <= 0
    ):
        raise ValueError(f"timed_iters must be a positive integer, got {timed_iters!r}")
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or not 1 <= max_candidates <= _MAX_AUTOTUNE_CANDIDATES
    ):
        raise ValueError(
            f"max_candidates must be an integer in [1, {_MAX_AUTOTUNE_CANDIDATES}], "
            f"got {max_candidates!r}"
        )
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of MoeEpTuningConfig values")
    if not candidates:
        raise ValueError("candidates must not be empty")

    ordered: list[MoeEpTuningConfig] = [baseline]
    seen = {baseline}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, MoeEpTuningConfig):
            raise TypeError(
                "candidates must contain only MoeEpTuningConfig values; "
                f"candidates[{index}] is {type(candidate).__name__}"
            )
        if candidate.reduce_topk_in_kernel != baseline.reduce_topk_in_kernel:
            raise ValueError(
                "autotune does not sweep reduce_topk_in_kernel; "
                f"candidate {index} has {candidate.reduce_topk_in_kernel}, "
                f"baseline has {baseline.reduce_topk_in_kernel}"
            )
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

    if len(ordered) > max_candidates:
        raise ValueError(
            f"autotune has {len(ordered)} unique candidates including the baseline, "
            f"exceeding max_candidates={max_candidates}"
        )
    return tuple(ordered)


def verify_candidates_across_ranks(
    candidates: tuple[MoeEpTuningConfig, ...],
    group: dist.ProcessGroup | None,
) -> None:
    """Fail before runtime allocation when EP ranks supplied different lists."""

    if group is None:
        return
    gathered: list[object] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, candidates, group=group)
    if any(value != candidates for value in gathered):
        raise RuntimeError(
            f"MoeEp autotune candidates must match on every EP rank; "
            f"rank candidate lists: {gathered}"
        )


def verify_state_across_ranks(
    state: tuple[object, ...],
    group: dist.ProcessGroup | None,
) -> None:
    """Require matching operator lifecycle state before collective teardown."""

    if group is None:
        return
    gathered: list[object] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, state, group=group)
    if any(value != state for value in gathered):
        raise RuntimeError(
            f"MoeEp autotune requires matching lifecycle state on every EP rank; "
            f"rank states: {gathered}"
        )


def raise_preflight_errors(
    error: BaseException | None,
    *,
    phase: str,
    group: dist.ProcessGroup | None,
) -> None:
    """Turn rank-local preflight failures into one collective failure."""

    if group is None:
        if error is not None:
            raise error
        return
    local = None if error is None else (type(error).__name__, str(error))
    gathered: list[object] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    failures = [
        (rank, value) for rank, value in enumerate(gathered) if value is not None
    ]
    if failures:
        raise RuntimeError(
            f"MoeEp autotune {phase} failed before runtime entry; rank errors: {failures}"
        ) from error


def synchronize_candidate(
    device: torch.device,
    group: dist.ProcessGroup | None,
) -> None:
    """Drain device work and align ranks at a healthy candidate boundary."""

    torch.cuda.synchronize(device)
    if group is not None:
        dist.barrier(group=group)


def benchmark_candidate(
    run: Callable[[], _T],
    *,
    device: torch.device,
    group: dist.ProcessGroup | None,
    timed_iters: int,
) -> tuple[float, tuple[float, ...]]:
    """Return median(per-iteration rank-MAX) in milliseconds."""

    stream = torch.cuda.current_stream(device)
    local_samples: list[float] = []
    for _ in range(timed_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        run()
        end.record(stream)
        end.synchronize()
        local_samples.append(float(start.elapsed_time(end)))

    slow_rank_samples = torch.tensor(local_samples, dtype=torch.float64, device=device)
    if group is not None:
        dist.all_reduce(slow_rank_samples, op=dist.ReduceOp.MAX, group=group)
    samples = tuple(float(value) for value in slow_rank_samples.cpu().tolist())
    latency_ms = float(statistics.median(samples))
    if not math.isfinite(latency_ms):
        raise RuntimeError(
            f"MoeEp autotune produced a non-finite latency: {latency_ms}"
        )
    return latency_ms, samples


def select_winner(
    results: Sequence[MoeEpAutotuneCandidateResult],
) -> MoeEpAutotuneCandidateResult:
    """Choose the first minimum-latency candidate for stable tie-breaking."""

    if not results:
        raise ValueError("cannot select an autotune winner without results")
    return min(results, key=lambda result: result.latency_ms)


def allocate_training_outputs(
    requirements,
    device: torch.device,
) -> tuple[MoeEpTrainingForwardOutputs, MoeEpTrainingBackwardOutputs]:
    """Allocate private one-lane outputs from the production ABI contract."""

    def allocate(name: str) -> torch.Tensor:
        shape, stride, dtype, alignment = requirements[name]
        tensor = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if tensor.data_ptr() % alignment:
            raise RuntimeError(
                f"autotune output {name} is not {alignment}-byte aligned"
            )
        return tensor

    forward = MoeEpTrainingForwardOutputs(
        fc1_preact=allocate("fc1_preact"),
        output=allocate("output"),
        fc1_a=allocate("fc1_a"),
        fc1_sfa=allocate("fc1_sfa"),
        valid_route_counts=allocate("valid_route_counts"),
        expert_offsets=allocate("expert_offsets"),
    )
    backward = MoeEpTrainingBackwardOutputs(
        grad_activation=allocate("grad_activation"),
        dprob=allocate("dprob"),
        fc1_b=allocate("fc1_b"),
        fc1_sfb=allocate("fc1_sfb"),
        fc2_a=allocate("fc2_a"),
        fc2_sfa=allocate("fc2_sfa"),
        fc2_b=allocate("fc2_b"),
        fc2_sfb=allocate("fc2_sfb"),
    )
    return forward, backward


__all__ = [
    "allocate_training_outputs",
    "benchmark_candidate",
    "normalize_candidates",
    "raise_preflight_errors",
    "select_winner",
    "synchronize_candidate",
    "verify_candidates_across_ranks",
    "verify_state_across_ranks",
]
