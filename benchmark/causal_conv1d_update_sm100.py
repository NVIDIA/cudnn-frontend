# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode microbenchmark for the causal-convolution update operation.

This compares the direct low-level cuDNN Frontend API against FLA 0.5.2's
public Triton update operation at representative Qwen3.5/Qwen3.8 decode
shapes.  The official model applies one fused-QKV depthwise convolution before
splitting Q, K, and V, so the benchmark uses the full convolution channel
counts D={6144, 8192, 10240, 12288, 20480}; every model uses W=4.
It is a single-process, same-input, AB/BA-interleaved comparison.  Both
implementations are compiled and CUDA-graph-captured before timing; in
particular, the ``torch.empty_like`` in FLA's wrapper runs during capture rather
than between the timing events.  Every timed replay starts from the same state
bits.

The script deliberately refuses to print timing results unless both output and
the final mutable cache pass an explicit FP32/BF16 reference gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import cudnn
import torch
import torch.nn.functional as F

# Execute the Python sources from this checkout while retaining the installed
# package's compiled extension.  This mirrors the focused pytest overlay and
# makes the source-hash route proof independent of site-package sync state.
_SOURCE_CUDNN = Path(__file__).resolve().parents[1] / "python" / "cudnn"
if str(_SOURCE_CUDNN) not in cudnn.__path__:
    cudnn.__path__.insert(0, str(_SOURCE_CUDNN))

from cudnn._causal_conv1d_arch import (
    is_supported_causal_conv1d_update_compute_capability,
    supported_causal_conv1d_update_compute_capabilities_text,
)
from cudnn.causal_conv1d_update_sm100 import _CausalConv1dUpdatePlan
from fla.modules.conv.triton.ops import causal_conv1d_update as fla_causal_conv1d_update

DEFAULT_BATCH_SIZES = (1, 8, 32, 128)
DEFAULT_CHANNELS = (6144, 8192, 10240, 12288, 20480)
DEFAULT_SHAPES = tuple((batch, channels) for channels in DEFAULT_CHANNELS for batch in DEFAULT_BATCH_SIZES)
OUTPUT_ATOL = 3e-2
OUTPUT_RTOL = 3e-2


class CapturedArm(NamedTuple):
    name: str
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    state: torch.Tensor


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        n_rows, n_channels = (int(piece) for piece in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must be N x D, for example 8x8192") from exc
    if n_rows <= 0 or n_channels <= 0:
        raise argparse.ArgumentTypeError("N and D must both be positive")
    return n_rows, n_channels


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _slurm_provenance() -> dict[str, str | None]:
    """Return optional scheduler metadata without restricting where the benchmark runs."""

    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurmd_node_name": os.environ.get("SLURMD_NODENAME"),
    }


def _module_path(module_name: str) -> Path:
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if path is None:
        raise RuntimeError(f"cannot resolve source path for loaded module {module_name}")
    return Path(path).resolve()


def _nvidia_driver_for_uuid(device_uuid: str) -> str:
    """Resolve the actual NVIDIA driver, matching the CUDA-visible GPU UUID."""

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    normalized_uuid = device_uuid.lower().removeprefix("gpu-")
    for line in output.splitlines():
        gpu_uuid, separator, driver_version = line.partition(",")
        if separator and gpu_uuid.strip().lower().removeprefix("gpu-") == normalized_uuid:
            return driver_version.strip()
    raise RuntimeError(f"nvidia-smi did not report CUDA-visible GPU UUID {device_uuid}")


def _capture(call: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]], *, warmup: int):
    """Warm and capture ``call``; all Python allocation stays outside timing."""

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        result = call()
    torch.cuda.synchronize()
    return graph, result


def _reference(x: torch.Tensor, weight: torch.Tensor, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    final_state = torch.cat((initial_state[..., 1:], x.unsqueeze(-1)), dim=-1)
    accumulator = (final_state.float() * weight.float().unsqueeze(0)).sum(dim=-1)
    output = F.silu(accumulator).to(torch.bfloat16)
    return output, final_state


def _gate_arm(
    arm: CapturedArm,
    initial_state: torch.Tensor,
    reference_output: torch.Tensor,
    reference_state: torch.Tensor,
) -> dict:
    arm.state.copy_(initial_state)
    arm.graph.replay()
    torch.cuda.synchronize()

    output_diff = (arm.output.float() - reference_output.float()).abs()
    state_bits_equal = torch.equal(arm.state.view(torch.int16), reference_state.view(torch.int16))
    output_close = torch.allclose(arm.output, reference_output, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    gate = {
        "output_close": bool(output_close),
        "output_max_abs": float(output_diff.max().item()),
        "output_mean_abs": float(output_diff.mean().item()),
        "state_bits_equal": bool(state_bits_equal),
    }
    if not output_close or not state_bits_equal:
        raise AssertionError(f"{arm.name} correctness gate failed: {gate}")
    return gate


def _prime_events(event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> None:
    """Force lazy CUDA event creation before any measured interval."""

    for start, end in event_pairs:
        start.record()
        end.record()
    torch.cuda.synchronize()


def _interleaved_samples(
    native: CapturedArm,
    fla: CapturedArm,
    initial_state: torch.Tensor,
    *,
    samples: int,
) -> tuple[list[float], list[float], list[str]]:
    """Measure one graph replay per sample, alternating AB and BA order."""

    arms = {native.name: native, fla.name: fla}
    timings = {native.name: [], fla.name: []}
    event_pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(samples * 2)]
    _prime_events(event_pairs)
    orders = []
    event_index = 0

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for sample_index in range(samples):
            order = (native.name, fla.name) if sample_index % 2 == 0 else (fla.name, native.name)
            orders.append("/".join(order))
            pending = []
            for name in order:
                arm = arms[name]
                # This reset is ordered before the start event on the same
                # stream, hence excluded from the measured interval.
                arm.state.copy_(initial_state)
                start, end = event_pairs[event_index]
                event_index += 1
                start.record()
                arm.graph.replay()
                end.record()
                pending.append((name, start, end))
            torch.cuda.synchronize()
            for name, start, end in pending:
                timings[name].append(float(start.elapsed_time(end) * 1000.0))
    finally:
        if gc_was_enabled:
            gc.enable()

    return timings[native.name], timings[fla.name], orders


def _summary(samples_us: list[float]) -> dict:
    return {
        "median_us": statistics.median(samples_us),
        "q25_us": _quantile(samples_us, 0.25),
        "q75_us": _quantile(samples_us, 0.75),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


@torch.no_grad()
def _benchmark_shape(n_rows: int, n_channels: int, *, samples: int, warmup: int, seed: int) -> dict:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = (
        torch.randn(
            (n_rows, n_channels),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )
    weight = torch.randn((n_channels, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    initial_state = (
        torch.randn(
            (n_rows, n_channels, 4),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )
    native_state = initial_state.clone()
    fla_state = initial_state.clone()
    native_output = torch.empty_like(x)

    api = _CausalConv1dUpdatePlan(x, weight, native_state, native_output, activation="silu")
    if not api.check_support():
        raise RuntimeError("native check_support unexpectedly returned false")
    api.compile()

    def native_call() -> torch.Tensor:
        return api.execute(x, weight, native_state, native_output)

    def fla_call() -> tuple[torch.Tensor, torch.Tensor]:
        # Use FLA's public Triton op directly.  Its real cache contract is
        # [N, D, W], including the newest element, so W=4 here.
        return fla_causal_conv1d_update(
            x=x,
            cache=fla_state,
            residual=None,
            weight=weight,
            bias=None,
            activation="silu",
        )

    native_graph, captured_native_output = _capture(native_call, warmup=warmup)
    fla_graph, captured_fla = _capture(fla_call, warmup=warmup)
    fla_output, returned_fla_state = captured_fla
    if captured_native_output.data_ptr() != native_output.data_ptr():
        raise AssertionError("native direct execute did not return its preallocated output")
    if returned_fla_state.data_ptr() != fla_state.data_ptr():
        raise AssertionError("FLA did not return the mutable cache passed by the benchmark")

    native_arm = CapturedArm("native", native_graph, native_output, native_state)
    fla_arm = CapturedArm("fla", fla_graph, fla_output, fla_state)
    reference_output, reference_state = _reference(x, weight, initial_state)
    correctness = {
        "native": _gate_arm(native_arm, initial_state, reference_output, reference_state),
        "fla": _gate_arm(fla_arm, initial_state, reference_output, reference_state),
    }
    cross_output_close = torch.allclose(native_output, fla_output, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    cross_state_bits_equal = torch.equal(native_state.view(torch.int16), fla_state.view(torch.int16))
    correctness["native_vs_fla"] = {
        "output_close": bool(cross_output_close),
        "state_bits_equal": bool(cross_state_bits_equal),
        "output_max_abs": float((native_output.float() - fla_output.float()).abs().max().item()),
    }
    if not cross_output_close or not cross_state_bits_equal:
        raise AssertionError(f"native/FLA cross gate failed: {correctness['native_vs_fla']}")

    # Graphs and events are warmed.  Every measured arm below is reset from the
    # same immutable state, outside of its event pair.
    for arm in (native_arm, fla_arm):
        for _ in range(warmup):
            arm.state.copy_(initial_state)
            arm.graph.replay()
    torch.cuda.synchronize()

    native_samples, fla_samples, orders = _interleaved_samples(
        native_arm,
        fla_arm,
        initial_state,
        samples=samples,
    )
    native_summary = _summary(native_samples)
    fla_summary = _summary(fla_samples)
    paired_ratios = [fla_us / native_us for native_us, fla_us in zip(native_samples, fla_samples, strict=True)]
    return {
        "shape": {"N": n_rows, "D": n_channels, "W": 4},
        "correctness": correctness,
        "native": {**native_summary, "samples_us": native_samples},
        "fla": {**fla_summary, "samples_us": fla_samples},
        "ratio_of_medians_fla_over_native": fla_summary["median_us"] / native_summary["median_us"],
        "paired_fla_over_native": {
            "median": statistics.median(paired_ratios),
            "min": min(paired_ratios),
            "max": max(paired_ratios),
            "samples": paired_ratios,
        },
        "sample_orders": orders,
    }


def _metadata(repo: Path, shapes: tuple[tuple[int, int], ...], args: argparse.Namespace) -> dict:
    executed_native_api_path = _module_path(_CausalConv1dUpdatePlan.__module__)
    executed_native_kernel_path = _module_path("cudnn.causal_conv1d_update_sm100.kernel")
    repo_native_api_path = (repo / "python/cudnn/causal_conv1d_update_sm100/api.py").resolve()
    repo_native_kernel_path = (repo / "python/cudnn/causal_conv1d_update_sm100/kernel.py").resolve()
    native_api_sha256 = _sha256(executed_native_api_path)
    native_kernel_sha256 = _sha256(executed_native_kernel_path)
    if native_api_sha256 != _sha256(repo_native_api_path) or native_kernel_sha256 != _sha256(repo_native_kernel_path):
        raise RuntimeError("executed native short-conv sources do not match this worktree")
    fla_source_path = _module_path(fla_causal_conv1d_update.__module__)
    fla_kernel_path = _module_path("fla.modules.conv.triton.kernels")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    device_uuid = str(properties.uuid)
    return {
        "schema_version": 1,
        "benchmark": "causal_conv1d_update_decode_sm100",
        "timing_contract": "warm cache-hit CUDA-graph replay; one update per sample; compile, capture, allocation, and state reset outside events",
        "comparison_contract": "single process; AB/BA interleaved; identical x/weight/initial state; BF16 N,D,W=4 no-bias SiLU",
        "shape_contract": ("official Qwen3.5/Qwen3.8 fused-QKV short conv: one update over " "D in {6144,8192,10240,12288,20480}; W=4"),
        "native_route": f"{_CausalConv1dUpdatePlan.__module__}.{_CausalConv1dUpdatePlan.__qualname__}.execute",
        "fla_route": f"{fla_causal_conv1d_update.__module__}.{fla_causal_conv1d_update.__name__}",
        "provenance": {
            "benchmark": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "native_api": {
                "executed_path": str(executed_native_api_path),
                "repo_path": str(repo_native_api_path),
                "sha256": native_api_sha256,
            },
            "native_kernel": {
                "executed_path": str(executed_native_kernel_path),
                "repo_path": str(repo_native_kernel_path),
                "sha256": native_kernel_sha256,
            },
            "fla_op": {
                "path": str(fla_source_path),
                "sha256": _sha256(fla_source_path),
                "license": "MIT",
            },
            "fla_kernel": {
                "path": str(fla_kernel_path),
                "sha256": _sha256(fla_kernel_path),
                "license": "MIT",
            },
        },
        "hardware": {
            "name": properties.name,
            "architecture": f"sm_{capability[0]}{capability[1]}",
            "compute_capability": list(capability),
            "device_index": device,
            "uuid": device_uuid,
            "driver": _nvidia_driver_for_uuid(device_uuid),
            "total_memory_bytes": properties.total_memory,
        },
        "slurm": _slurm_provenance(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version(),
            "cudnn_frontend": cudnn.__version__,
            "cudnn_backend": cudnn.backend_version(),
            "cudnn_backend_string": cudnn.backend_version_string(),
            "flash_linear_attention": _package_version("flash-linear-attention"),
        },
        "repository": {
            "root": str(repo),
            "head": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "dirty": bool(_git(repo, "status", "--porcelain")),
        },
        "parameters": {
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
            "shapes": [list(shape) for shape in shapes],
        },
    }


def _validate_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = tuple(torch.cuda.get_device_capability())
    if not is_supported_causal_conv1d_update_compute_capability(capability):
        raise RuntimeError(
            "benchmark requires a functionally supported causal-conv compute capability "
            f"({supported_causal_conv1d_update_compute_capabilities_text()}), found {capability[0]}.{capability[1]}"
        )
    if _CausalConv1dUpdatePlan.__module__ != "cudnn.causal_conv1d_update_sm100.api":
        raise RuntimeError(f"unexpected native route: {_CausalConv1dUpdatePlan.__module__}")
    if fla_causal_conv1d_update.__module__ != "fla.modules.conv.triton.ops":
        raise RuntimeError(f"unexpected FLA route: {fla_causal_conv1d_update.__module__}")
    if _package_version("flash-linear-attention") != "0.5.2":
        raise RuntimeError("this controlled comparison requires flash-linear-attention==0.5.2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="N x D; repeat for multiple shapes (default: Qwen decode matrix)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=51,
        help="timed samples per implementation and shape",
    )
    parser.add_argument("--warmup", type=int, default=10, help="warm replays before measurement")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-json", type=Path, help="also write the complete result to this file")
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if args.warmup < 1:
        parser.error("--warmup must be positive")
    shapes = tuple(args.shape) if args.shape else DEFAULT_SHAPES

    _validate_environment()
    repo = Path(__file__).resolve().parents[1]
    result = _metadata(repo, shapes, args)
    result["results"] = []
    for shape_index, (n_rows, n_channels) in enumerate(shapes):
        row = _benchmark_shape(
            n_rows,
            n_channels,
            samples=args.samples,
            warmup=args.warmup,
            seed=args.seed + shape_index,
        )
        result["results"].append(row)
        print(
            f"N={n_rows:3d} D={n_channels:4d}: native={row['native']['median_us']:.3f} us, "
            f"FLA={row['fla']['median_us']:.3f} us, paired FLA/native={row['paired_fla_over_native']['median']:.3f}x",
            flush=True,
        )

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print("RESULT_JSON_BEGIN")
    print(encoded)
    print("RESULT_JSON_END")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n")


if __name__ == "__main__":
    sys.exit(main())
