# SPDX-License-Identifier: BSD-3-Clause

"""Static-mask training benchmark for cuDNN Frontend Flex Attention."""

from __future__ import annotations

import argparse
from array import array
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import random
import socket
import statistics
import subprocess
import time
from typing import Any, Callable, Literal, Sequence

import torch

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
STANDARD_SEQLEN = 128 * 1024
HSTU_DOCUMENT_MIN = 8 * 1024
HSTU_DOCUMENT_MAX = 16 * 1024
DOCUMENT_LENGTHS_128K = (
    11858,
    8270,
    12765,
    11038,
    4578,
    14018,
    11721,
    11988,
    4942,
    8393,
    7541,
    13495,
    10465,
)
MASK_NAMES = (
    "causal",
    "document_causal",
    "local",
    "sink_local",
    "tree_dfs",
    "tree_bfs",
    "longformer",
    "hstu",
)
PHASE_NAMES = ("forward", "backward", "combined")
HEAD_DIM_CONFIGS = {
    128: (128, 128),
    192: (192, 128),
    256: (256, 256),
}
SUPPORTED_ARCHES = (90, 100, 103)


@dataclass(frozen=True)
class Workload:
    batch_size: int = 1
    seqlen: int = STANDARD_SEQLEN
    num_q_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int = 128
    head_dim_v: int = 128
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError("the static-mask benchmark requires batch_size=1")
        if self.seqlen <= 0:
            raise ValueError("seqlen must be positive")
        if self.num_q_heads != self.num_kv_heads:
            raise ValueError("the static-mask benchmark requires Hq=Hkv")
        if (self.head_dim, self.head_dim_v) not in HEAD_DIM_CONFIGS.values():
            raise ValueError("the static-mask benchmark supports (Dqk,Dv) in " f"{tuple(HEAD_DIM_CONFIGS.values())}")
        if self.dtype != torch.bfloat16:
            raise ValueError("the static-mask benchmark requires BF16")


@dataclass(frozen=True)
class MaskSpec:
    name: str
    title: str
    endpoints: torch.Tensor
    details: dict[str, Any]
    visible_pairs: int

    @property
    def nfunc(self) -> int:
        return self.endpoints.shape[0]

    @property
    def seqlen(self) -> int:
        return self.endpoints.shape[1]

    @property
    def density(self) -> float:
        return self.visible_pairs / (self.seqlen * self.seqlen)


@dataclass
class TimingStats:
    samples_ms: list[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    def to_json(self) -> dict[str, Any]:
        return {
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "samples_ms": self.samples_ms,
        }


def _jittered_lengths(total: int, count: int, *, seed: int, jitter: float) -> list[int]:
    if count <= 0 or total < count:
        raise ValueError("length count must be positive and no greater than total")
    generator = random.Random(seed)
    weights = [generator.uniform(1.0 - jitter, 1.0 + jitter) for _ in range(count)]
    scaled = [weight * total / sum(weights) for weight in weights]
    lengths = [max(1, math.floor(value)) for value in scaled]
    difference = total - sum(lengths)
    fractional_order = sorted(
        range(count),
        key=lambda idx: scaled[idx] - math.floor(scaled[idx]),
        reverse=True,
    )
    for index in range(difference):
        lengths[fractional_order[index % count]] += 1
    if sum(lengths) != total:
        raise AssertionError("failed to normalize jittered lengths")
    return lengths


def _bounded_partition(
    total: int,
    minimum: int,
    maximum: int,
    *,
    seed: int,
    prefer_multiple: bool = False,
) -> list[int]:
    if total < minimum:
        return [total]
    min_parts = math.ceil(total / maximum)
    max_parts = total // minimum
    count = round(total / ((minimum + maximum) / 2))
    count = min(max(count, min_parts), max_parts)
    if prefer_multiple and count == 1 and max_parts >= 2:
        count = 2
    generator = random.Random(seed)
    lengths = [generator.randint(minimum, maximum) for _ in range(count)]
    difference = total - sum(lengths)
    while difference:
        order = list(range(count))
        generator.shuffle(order)
        progressed = False
        for index in order:
            if difference > 0:
                delta = min(difference, maximum - lengths[index])
            else:
                delta = -min(-difference, lengths[index] - minimum)
            if delta:
                lengths[index] += delta
                difference -= delta
                progressed = True
            if difference == 0:
                break
        if not progressed:
            raise AssertionError("bounded partition cannot satisfy the requested total")
    return lengths


def _hstu_document_lengths(total: int, *, seed: int) -> tuple[list[int], list[int]]:
    contexts = _bounded_partition(
        total // 2,
        HSTU_DOCUMENT_MIN // 2,
        HSTU_DOCUMENT_MAX // 2,
        seed=seed,
    )
    targets = contexts.copy()
    targets[-1] += total % 2
    return contexts, targets


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for begin, end in sorted(intervals):
        if not 0 <= begin <= end:
            raise ValueError(f"invalid interval [{begin}, {end})")
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return [(begin, end) for begin, end in merged]


def _encode_intervals(
    intervals: Sequence[tuple[int, int]],
    *,
    max_intervals: int,
    starts_at_zero: bool,
) -> list[int]:
    merged = _merge_intervals(intervals)
    if not merged or len(merged) > max_intervals:
        raise ValueError("interval union does not fit the selected nfunc")
    if starts_at_zero:
        if merged[0][0] != 0:
            raise ValueError("the first interval must start at zero")
        encoded = [merged[0][1]]
        encoded.extend(value for pair in merged[1:] for value in pair)
        expected = 2 * max_intervals - 1
    else:
        encoded = [0]
        encoded.extend(value for pair in merged for value in pair)
        expected = 2 * max_intervals + 1
    encoded.extend([encoded[-1]] * (expected - len(encoded)))
    return encoded


def _validate_endpoints(endpoints: torch.Tensor, seqlen: int) -> None:
    if endpoints.dtype != torch.int32 or endpoints.ndim != 2 or not endpoints.is_contiguous():
        raise ValueError("endpoints must be a contiguous int32 [nfunc, seqlen] tensor")
    if endpoints.shape[1] != seqlen or endpoints.shape[0] % 2 != 1:
        raise ValueError("endpoints must have odd nfunc and exactly seqlen columns")
    if endpoints.shape[0] >= 33:
        raise ValueError("the static-mask benchmark requires nfunc < 33")
    if bool(((endpoints < 0) | (endpoints > seqlen)).any()):
        raise ValueError("endpoint lies outside the sequence")
    if endpoints.shape[0] > 1 and bool((endpoints[1:] < endpoints[:-1]).any()):
        raise ValueError("endpoints must be nondecreasing for every query")


def visible_pair_count(endpoints: torch.Tensor) -> int:
    counts = endpoints[0].to(torch.int64)
    for index in range(1, endpoints.shape[0], 2):
        counts = counts + endpoints[index + 1] - endpoints[index]
    return int(counts.sum().item())


def endpoint_visible(endpoints: torch.Tensor, q_idx: int, kv_idx: int) -> bool:
    row = endpoints[:, q_idx]
    visible = kv_idx < int(row[0])
    for index in range(1, row.numel(), 2):
        visible |= int(row[index]) <= kv_idx < int(row[index + 1])
    return visible


def _make_causal(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    return torch.arange(1, seqlen + 1, dtype=torch.int32).view(1, -1), {}


def _make_document_causal(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    lengths = list(DOCUMENT_LENGTHS_128K) if seqlen == STANDARD_SEQLEN else _bounded_partition(seqlen, 4096, 16384, seed=42, prefer_multiple=True)
    endpoints = torch.empty((3, seqlen), dtype=torch.int32)
    offset = 0
    for length in lengths:
        end = offset + length
        endpoints[0, offset:end] = 0
        endpoints[1, offset:end] = offset
        endpoints[2, offset:end] = torch.arange(offset + 1, end + 1, dtype=torch.int32)
        offset = end
    return endpoints, {"document_lengths": lengths, "seed": 42}


def _make_local(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    q_idx = torch.arange(seqlen, dtype=torch.int32)
    end = q_idx + 1
    begin = torch.clamp(q_idx - 512, min=0)
    return torch.stack((torch.zeros_like(begin), begin, end)), {"window_left": 512}


def _make_sink_local(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    q_idx = torch.arange(seqlen, dtype=torch.int32)
    end = q_idx + 1
    sink_end = torch.minimum(end, torch.scalar_tensor(4, dtype=torch.int32))
    local_begin = torch.clamp(q_idx - 512, min=4)
    local_begin = torch.minimum(torch.maximum(local_begin, sink_end), end)
    return torch.stack((sink_end, local_begin, end)), {"sink_tokens": 4, "window_left": 512}


def _tree_order(depth: int, traversal: Literal["dfs", "bfs"]) -> list[int]:
    node_count = 2**depth - 1
    if traversal == "bfs":
        return list(range(node_count))
    order: list[int] = []

    def visit(node: int) -> None:
        if node >= node_count:
            return
        order.append(node)
        visit(2 * node + 1)
        visit(2 * node + 2)

    visit(0)
    return order


def _tree_ancestors(node: int) -> list[int]:
    result = []
    while True:
        result.append(node)
        if node == 0:
            break
        node = (node - 1) // 2
    return list(reversed(result))


def _make_tree(seqlen: int, traversal: Literal["dfs", "bfs"]) -> tuple[torch.Tensor, dict[str, Any]]:
    depth = 7
    node_count = 2**depth - 1
    lengths = _jittered_lengths(seqlen, node_count, seed=42, jitter=0.2)
    order = _tree_order(depth, traversal)
    starts: dict[int, int] = {}
    offset = 0
    for node in order:
        starts[node] = offset
        offset += lengths[node]
    endpoints = torch.empty((2 * depth - 1, seqlen), dtype=torch.int32)
    for node in order:
        start = starts[node]
        end = start + lengths[node]
        row_end = torch.arange(start + 1, end + 1, dtype=torch.int32)
        ancestor_intervals = [(starts[ancestor], starts[ancestor] + lengths[ancestor]) for ancestor in _tree_ancestors(node)[:-1]]
        template = _merge_intervals((*ancestor_intervals, (start, end)))
        encoded = _encode_intervals(template, max_intervals=depth, starts_at_zero=True)
        current_end_slot = 2 * len(template) - 2
        for slot, value in enumerate(encoded):
            endpoints[slot, start:end] = row_end if slot >= current_end_slot else value
    return endpoints, {
        "depth": depth,
        "node_count": node_count,
        "node_lengths": lengths,
        "node_order": order,
        "traversal": traversal,
        "seed": 42,
        "length_jitter": 0.2,
    }


def _make_longformer(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    radius = 256
    global_tokens = [int((index + 0.5) * seqlen / 8) for index in range(8)]
    global_tokens = sorted({min(token, seqlen - 1) for token in global_tokens})
    flat = array("i")
    global_set = set(global_tokens)
    for q_idx in range(seqlen):
        if q_idx in global_set:
            intervals = [(0, seqlen)]
        else:
            intervals = [(max(0, q_idx - radius), min(seqlen, q_idx + radius + 1))]
            intervals.extend((token, token + 1) for token in global_tokens)
        flat.extend(_encode_intervals(intervals, max_intervals=9, starts_at_zero=False))
    endpoints = torch.tensor(flat, dtype=torch.int32).view(seqlen, 19).t().contiguous()
    return endpoints, {
        "local_radius": radius,
        "global_tokens": global_tokens,
        "global_token_count": len(global_tokens),
    }


def _make_hstu(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    contexts, targets = _hstu_document_lengths(seqlen, seed=42)
    endpoints = torch.empty((5, seqlen), dtype=torch.int32)
    offset = 0
    for context, target in zip(contexts, targets):
        context_end = offset + context
        document_end = context_end + target
        context_rows = torch.arange(offset + 1, context_end + 1, dtype=torch.int32)
        endpoints[0, offset:context_end] = 0
        endpoints[1, offset:context_end] = offset
        endpoints[2, offset:context_end] = context_rows
        endpoints[3, offset:context_end] = context_rows
        endpoints[4, offset:context_end] = context_rows
        target_rows = torch.arange(context_end, document_end, dtype=torch.int32)
        endpoints[0, context_end:document_end] = 0
        endpoints[1, context_end:document_end] = offset
        endpoints[2, context_end:document_end] = context_end
        endpoints[3, context_end:document_end] = target_rows
        endpoints[4, context_end:document_end] = target_rows + 1
        offset = document_end
    return endpoints, {
        "context_lengths": contexts,
        "target_lengths": targets,
        "document_lengths": [context + target for context, target in zip(contexts, targets)],
        "document_length_bounds": [HSTU_DOCUMENT_MIN, HSTU_DOCUMENT_MAX],
        "target_context_ratio": 1,
        "seed": 42,
    }


_MASK_BUILDERS: dict[str, tuple[str, Callable[[int], tuple[torch.Tensor, dict[str, Any]]]]] = {
    "causal": ("Causal", _make_causal),
    "document_causal": ("Varlen document causal", _make_document_causal),
    "local": ("Causal local W=512", _make_local),
    "sink_local": ("Sink S=4 + local W=512", _make_sink_local),
    "tree_dfs": ("Tree attention DFS", lambda length: _make_tree(length, "dfs")),
    "tree_bfs": ("Tree attention BFS", lambda length: _make_tree(length, "bfs")),
    "longformer": ("Longformer", _make_longformer),
    "hstu": ("Packed HSTU context/target", _make_hstu),
}


def make_mask_spec(name: str, seqlen: int) -> MaskSpec:
    if name not in _MASK_BUILDERS:
        raise ValueError(f"unknown mask {name!r}; expected one of {MASK_NAMES}")
    title, builder = _MASK_BUILDERS[name]
    endpoints, details = builder(seqlen)
    endpoints = endpoints.contiguous()
    _validate_endpoints(endpoints, seqlen)
    return MaskSpec(
        name=name,
        title=title,
        endpoints=endpoints,
        details=details,
        visible_pairs=visible_pair_count(endpoints),
    )


class FlexRunner:
    def __init__(self, workload: Workload, mask_func: torch.Tensor) -> None:
        from cudnn.flex_attention import create_mask_plan, flex_attn_func

        torch.manual_seed(0)
        q_shape = (
            workload.batch_size,
            workload.seqlen,
            workload.num_q_heads,
            workload.head_dim,
        )
        kv_shape = (
            workload.batch_size,
            workload.seqlen,
            workload.num_kv_heads,
            workload.head_dim,
        )
        v_shape = (*kv_shape[:-1], workload.head_dim_v)
        out_shape = (*q_shape[:-1], workload.head_dim_v)
        self.q = torch.randn(q_shape, dtype=workload.dtype, device="cuda", requires_grad=True)
        self.k = torch.randn(kv_shape, dtype=workload.dtype, device="cuda", requires_grad=True)
        self.v = torch.randn(v_shape, dtype=workload.dtype, device="cuda", requires_grad=True)
        generator = torch.Generator(device="cuda").manual_seed(1)
        self.dout = torch.randn(out_shape, dtype=workload.dtype, device="cuda", generator=generator)
        self.mask_func = mask_func
        self.softmax_scale = 1.0 / math.sqrt(workload.head_dim)
        self._create_mask_plan = create_mask_plan
        self._flex_attn_func = flex_attn_func
        self.plan: Any = None
        self.backward_output: torch.Tensor | None = None

    @property
    def qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q, self.k, self.v

    def build_metadata(self):
        return self._create_mask_plan(
            self.mask_func,
            self.q,
            self.k,
            self.v,
            pack_gqa=False,
            build_backward=True,
        )

    def forward(self, plan=None) -> torch.Tensor:
        return self._flex_attn_func(
            self.q,
            self.k,
            self.v,
            mask_plan=self.plan if plan is None else plan,
            softmax_scale=self.softmax_scale,
            deterministic=False,
        )

    def run_forward(self) -> torch.Tensor:
        return self.forward()

    def prepare_backward_graph(self) -> None:
        self.backward_output = self.forward()

    def run_backward(self) -> tuple[torch.Tensor, ...]:
        if self.backward_output is None:
            raise RuntimeError("backward graph has not been prepared")
        output = self.backward_output
        self.backward_output = None
        return torch.autograd.grad(output, self.qkv, self.dout, retain_graph=False)

    def run_combined(self) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(self.forward(), self.qkv, self.dout)

    def run_metadata_forward(self) -> torch.Tensor:
        return self.forward(self.build_metadata())

    def run_metadata_combined(self) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(self.forward(self.build_metadata()), self.qkv, self.dout)


def _create_l2_flush_buffer() -> torch.Tensor:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(getattr(properties, "L2_cache_size", 128 * 1024 * 1024))
    return torch.empty(max(2 * l2_bytes, 256 * 1024 * 1024), dtype=torch.uint8, device="cuda")


def _flush_l2(buffer: torch.Tensor) -> None:
    buffer.zero_()


def _compile_step(callable_: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started_at) * 1e3


def _flex_block_stats(plan: Any, seqlen: int) -> dict[str, Any]:
    packed_plan = plan._runtime_args[0]
    partial = int(packed_plan.mask_block_cnt.sum().item())
    full = int(packed_plan.full_block_cnt.sum().item()) if packed_plan.full_block_cnt is not None else 0
    block_size = tuple(int(value) for value in packed_plan.block_size)
    total = packed_plan.mask_block_cnt.numel() * math.ceil(seqlen / block_size[1])
    empty = total - partial - full
    return {
        "block_size": list(block_size),
        "partial": partial,
        "full": full,
        "empty": empty,
        "active": partial + full,
        "total": total,
    }


def _prepare_runner(runner: FlexRunner, phases: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:
    compile_ms: dict[str, float] = {}
    runner.plan, compile_ms["metadata"] = _compile_step(runner.build_metadata)
    block_stats = _flex_block_stats(runner.plan, runner.q.shape[1])
    _, compile_ms["forward"] = _compile_step(runner.run_forward)
    if "backward" in phases or "combined" in phases:
        runner.prepare_backward_graph()
        torch.cuda.synchronize()
        _, compile_ms["backward"] = _compile_step(runner.run_backward)
    print(
        "compiled: " + ", ".join(f"{name}={value:.1f}ms" for name, value in compile_ms.items()),
        flush=True,
    )
    return compile_ms, block_stats


def _time_one_cuda(callable_: Callable[[], Any], flush_buffer: torch.Tensor) -> float:
    _flush_l2(flush_buffer)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _time_one_wall(callable_: Callable[[], Any], flush_buffer: torch.Tensor) -> float:
    _flush_l2(flush_buffer)
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started_at) * 1e3


def _measure(
    callable_: Callable[[], Any],
    *,
    setup: Callable[[], Any] | None,
    flush_buffer: torch.Tensor,
    warmup: int,
    runs: int,
    wall_clock: bool,
) -> TimingStats:
    for _ in range(warmup):
        if setup is not None:
            setup()
        _flush_l2(flush_buffer)
        callable_()
        torch.cuda.synchronize()
    timer = _time_one_wall if wall_clock else _time_one_cuda
    samples = []
    for _ in range(runs):
        if setup is not None:
            setup()
        samples.append(timer(callable_, flush_buffer))
    return TimingStats(samples)


def _metric_actions(runner: FlexRunner, phases: Sequence[str]) -> list[tuple[str, Callable[[], Any], bool, Callable[[], Any] | None]]:
    actions = [("metadata", runner.build_metadata, True, None)]
    if "forward" in phases:
        actions.extend(
            (
                ("forward", runner.run_forward, False, None),
                ("metadata_forward", runner.run_metadata_forward, True, None),
            )
        )
    if "backward" in phases:
        actions.append(("backward", runner.run_backward, False, runner.prepare_backward_graph))
    if "combined" in phases:
        actions.extend(
            (
                ("combined", runner.run_combined, False, None),
                ("metadata_combined", runner.run_metadata_combined, True, None),
            )
        )
    return actions


def _phase_flops(workload: Workload, visible_pairs: int, phase: str) -> int:
    head_pairs = workload.num_q_heads * visible_pairs
    if phase == "forward":
        return 2 * head_pairs * (workload.head_dim + workload.head_dim_v)
    if phase == "backward":
        return 2 * head_pairs * (3 * workload.head_dim + 2 * workload.head_dim_v)
    if phase == "combined":
        return 2 * head_pairs * (4 * workload.head_dim + 3 * workload.head_dim_v)
    raise ValueError(f"phase {phase!r} has no FLOP definition")


def _tflops(flops: int, milliseconds: float) -> float:
    return flops / (milliseconds * 1e9)


def benchmark_mask(
    workload: Workload,
    spec: MaskSpec,
    phases: Sequence[str],
    *,
    warmup: int,
    runs: int,
    metadata_runs: int,
) -> dict[str, Any]:
    print(
        f"\n[{spec.name}] nfunc={spec.nfunc} density={spec.density:.4%} " f"visible_pairs={spec.visible_pairs:,}",
        flush=True,
    )
    mask_func = spec.endpoints.unsqueeze(0).to(device="cuda", non_blocking=True).contiguous()
    runner = FlexRunner(workload, mask_func)
    compile_ms, block_stats = _prepare_runner(runner, phases)
    flush_buffer = _create_l2_flush_buffer()
    metrics: dict[str, dict[str, Any]] = {}
    for metric, action, wall_clock, setup in _metric_actions(runner, phases):
        stats = _measure(
            action,
            setup=setup,
            flush_buffer=flush_buffer,
            warmup=0 if metric == "metadata" else warmup,
            runs=metadata_runs if metric == "metadata" else runs,
            wall_clock=wall_clock,
        )
        metrics[metric] = stats.to_json()
        if metric in PHASE_NAMES:
            metrics[metric]["active_tflops"] = _tflops(_phase_flops(workload, spec.visible_pairs, metric), stats.median_ms)
        suffix = f" active_tflops={metrics[metric]['active_tflops']:.1f}" if metric in PHASE_NAMES else ""
        print(f"{metric}: {stats.median_ms:.3f}ms{suffix}", flush=True)
    result = {
        "mask": spec.name,
        "title": spec.title,
        "head_dim": workload.head_dim,
        "head_dim_v": workload.head_dim_v,
        "nfunc": spec.nfunc,
        "visible_pairs": spec.visible_pairs,
        "element_density": spec.density,
        "details": spec.details,
        "compile_ms": compile_ms,
        "block_stats": block_stats,
        "metrics": metrics,
    }
    del flush_buffer, runner, mask_func
    torch.cuda.empty_cache()
    return result


def _safe_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(("git", "-C", str(root), *args), text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        driver = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "driver": driver,
        "torch": torch.__version__,
        "cutlass_dsl": _safe_version("nvidia-cutlass-dsl"),
        "cudnn_frontend": _safe_version("nvidia-cudnn-frontend"),
        "repository_commit": _git_output(root, "rev-parse", "HEAD"),
        "repository_dirty": bool(_git_output(root, "status", "--porcelain", "--untracked-files=no")),
    }


def _workload_json(workload: Workload) -> dict[str, Any]:
    result = asdict(workload)
    result["dtype"] = str(workload.dtype).removeprefix("torch.")
    return result


def write_csv(path: Path, cases: Sequence[dict[str, Any]]) -> None:
    fields = (
        "mask",
        "head_dim",
        "head_dim_v",
        "density",
        "nfunc",
        "metadata_ms",
        "forward_ms",
        "forward_tflops",
        "backward_ms",
        "backward_tflops",
        "combined_ms",
        "combined_tflops",
        "metadata_forward_ms",
        "metadata_combined_ms",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            metrics = case["metrics"]
            writer.writerow(
                {
                    "mask": case["mask"],
                    "head_dim": case["head_dim"],
                    "head_dim_v": case["head_dim_v"],
                    "density": case["element_density"],
                    "nfunc": case["nfunc"],
                    "metadata_ms": metrics.get("metadata", {}).get("median_ms"),
                    "forward_ms": metrics.get("forward", {}).get("median_ms"),
                    "forward_tflops": metrics.get("forward", {}).get("active_tflops"),
                    "backward_ms": metrics.get("backward", {}).get("median_ms"),
                    "backward_tflops": metrics.get("backward", {}).get("active_tflops"),
                    "combined_ms": metrics.get("combined", {}).get("median_ms"),
                    "combined_tflops": metrics.get("combined", {}).get("active_tflops"),
                    "metadata_forward_ms": metrics.get("metadata_forward", {}).get("median_ms"),
                    "metadata_combined_ms": metrics.get("metadata_combined", {}).get("median_ms"),
                }
            )


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "| Mask | Density | Plan ms | FWD ms / active TFLOPS | BWD ms / active TFLOPS | Train ms / active TFLOPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def metric_cell(metrics: dict[str, Any], name: str) -> str:
        metric = metrics.get(name)
        if metric is None:
            return "N/A"
        return f"{metric['median_ms']:.3f} / {metric['active_tflops']:.1f}"

    for case in results["benchmark"]:
        metrics = case["metrics"]
        lines.append(
            f"| {case['title']} | {case['element_density']:.2%} | "
            f"{metrics['metadata']['median_ms']:.3f} | {metric_cell(metrics, 'forward')} | "
            f"{metric_cell(metrics, 'backward')} | {metric_cell(metrics, 'combined')} |"
        )
    return "\n".join(lines) + "\n"


def _write_result_files(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(output_dir / "results.csv", results["benchmark"])
    (output_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")


def _parse_selection(value: str, allowed: Sequence[str], *, label: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(allowed)
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(selected) - set(allowed))
    if not selected or invalid:
        raise ValueError(f"invalid {label}: {invalid or value}; expected {allowed}")
    return selected


def _validate_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the Flex Attention benchmark requires CUDA")
    capability = torch.cuda.get_device_capability()
    arch = capability[0] * 10 + capability[1]
    if arch not in SUPPORTED_ARCHES:
        raise RuntimeError("cudnn.flex_attention supports SM90, SM100, and SM103; " f"got {torch.cuda.get_device_name()} SM{arch}")


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark cuDNN Frontend Flex Attention across eight static masks")
    parser.add_argument("--mask", default="all", help="all or a comma-separated mask list")
    parser.add_argument(
        "--phase",
        default="all",
        help="all or a comma-separated subset of forward,backward,combined",
    )
    parser.add_argument("--seqlen", type=int, default=STANDARD_SEQLEN)
    parser.add_argument(
        "--head-dim",
        type=int,
        choices=tuple(HEAD_DIM_CONFIGS),
        default=128,
        help="128=(128,128), 192=(192,128), 256=(256,256)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--metadata-runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        masks = _parse_selection(args.mask, MASK_NAMES, label="mask")
        phases = PHASE_NAMES if args.phase == "all" else _parse_selection(args.phase, PHASE_NAMES, label="phase")
    except ValueError as error:
        parser.error(str(error))
    if args.seqlen <= 0:
        parser.error("seqlen must be positive")
    if args.warmup < 0 or args.runs <= 0 or args.metadata_runs <= 0:
        parser.error("warmup must be non-negative; runs and metadata-runs must be positive")
    head_dim, head_dim_v = HEAD_DIM_CONFIGS[args.head_dim]
    if args.dry_run:
        print(f"seqlen={args.seqlen} masks={len(masks)} Dqk={head_dim} Dv={head_dim_v} " f"backend=flex_attention phases={','.join(phases)}")
        for name in masks:
            spec = make_mask_spec(name, args.seqlen)
            print(f"{spec.name}: nfunc={spec.nfunc} density={spec.density:.6%} " f"visible_pairs={spec.visible_pairs}")
        return

    _validate_environment()
    workload = Workload(seqlen=args.seqlen, head_dim=head_dim, head_dim_v=head_dim_v)
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    results: dict[str, Any] = {
        "schema_version": 1,
        "backend": "cudnn.flex_attention",
        "provenance": collect_provenance(),
        "workload": _workload_json(workload),
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "metadata_runs": args.metadata_runs,
            "l2_flush_before_each_sample": True,
            "phases": list(phases),
            "masks": list(masks),
        },
        "benchmark": [],
    }
    print(
        f"gpu={results['provenance']['gpu']} backend=cudnn.flex_attention output={output_dir}",
        flush=True,
    )
    for name in masks:
        spec = make_mask_spec(name, workload.seqlen)
        results["benchmark"].append(
            benchmark_mask(
                workload,
                spec,
                phases,
                warmup=args.warmup,
                runs=args.runs,
                metadata_runs=args.metadata_runs,
            )
        )
        _write_result_files(output_dir, results)
    print(f"results={output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
