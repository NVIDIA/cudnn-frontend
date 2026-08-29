# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact FLA-shim benchmark for the causal-convolution decode update.

This benchmark measures the public callable installed by
``cudnn.fla.accelerate_fla(targets="short_conv")`` against the saved FLA 0.5.2
public callable.  It patches once, keeps both arms resident in one process, and
restores once after all measurements.  No target toggling occurs in a sample.

Two intentionally separate steady-state metrics are reported:

* ``graph_replay_us`` is CUDA-event elapsed time for one warmed CUDA-graph
  replay.  Python dispatch, output allocation, compilation, and state reset are
  outside the replay interval.
* ``eager_host_enqueue_us`` is host wall time from entering the public Python
  callable until it returns after enqueuing the update.  The stream is drained
  before each sample and synchronized after the timer.  This includes Python
  validation, cache lookup, output allocation, and launch overhead, but it is
  explicitly *not* kernel time or completed device latency.

Timing is fail-closed unless the exact monkeypatch route is proven and both
arms pass full-output FP32-reference, bitwise-state, cache-identity, and cross
implementation gates.  The timed Qwen decode layout is ``[N, 1, D]`` for
``N={1,8,32,128}``, ``D={2048,4096}``, and ``W=4``.  The other admitted input
layouts, ``[N,D]`` and ``[1,N,D]``, receive additional correctness/route smoke
coverage.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

import torch

# Import the shared timing/reference helpers from the adjacent low-level
# benchmark without relying on ``benchmark`` being an installed package.
_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
import causal_conv1d_update_sm100 as _base
import cudnn
import cudnn.fla as cudnn_fla
import fla.modules.conv.triton.ops as fla_ops

DEFAULT_SHAPES = _base.DEFAULT_SHAPES
SMOKE_N = 8
SMOKE_D = 2048


class EagerArm(NamedTuple):
    name: str
    call: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    state: torch.Tensor


def _callable_code_path(function: Callable) -> Path:
    code = getattr(function, "__code__", None)
    filename = getattr(code, "co_filename", None)
    if filename is None:
        raise RuntimeError(f"cannot resolve code source for {function!r}")
    return Path(filename).resolve()


def _tensor_gate(
    name: str,
    output: torch.Tensor,
    state: torch.Tensor,
    reference_output: torch.Tensor,
    reference_state: torch.Tensor,
) -> dict:
    output_diff = (output.float() - reference_output.float()).abs()
    output_close = torch.allclose(output, reference_output, atol=_base.OUTPUT_ATOL, rtol=_base.OUTPUT_RTOL)
    state_bits_equal = torch.equal(state.view(torch.int16), reference_state.view(torch.int16))
    gate = {
        "output_close": bool(output_close),
        "output_max_abs": float(output_diff.max().item()),
        "output_mean_abs": float(output_diff.mean().item()),
        "state_bits_equal": bool(state_bits_equal),
    }
    if not output_close or not state_bits_equal:
        raise AssertionError(f"{name} correctness gate failed: {gate}")
    return gate


def _cross_gate(
    lhs_name: str,
    lhs_output: torch.Tensor,
    lhs_state: torch.Tensor,
    rhs_name: str,
    rhs_output: torch.Tensor,
    rhs_state: torch.Tensor,
) -> dict:
    output_close = torch.allclose(lhs_output, rhs_output, atol=_base.OUTPUT_ATOL, rtol=_base.OUTPUT_RTOL)
    state_bits_equal = torch.equal(lhs_state.view(torch.int16), rhs_state.view(torch.int16))
    gate = {
        "output_close": bool(output_close),
        "output_max_abs": float((lhs_output.float() - rhs_output.float()).abs().max().item()),
        "state_bits_equal": bool(state_bits_equal),
    }
    if not output_close or not state_bits_equal:
        raise AssertionError(f"{lhs_name}/{rhs_name} cross gate failed: {gate}")
    return gate


def _reference_for_layout(x: torch.Tensor, weight: torch.Tensor, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows, n_channels = initial_state.shape[:2]
    output, final_state = _base._reference(x.view(n_rows, n_channels), weight, initial_state)
    return output.view(x.shape), final_state


def _make_inputs(
    n_rows: int,
    n_channels: int,
    *,
    layout: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    flat_x = torch.randn((n_rows, n_channels), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    if layout == "N1D":
        x = flat_x.view(n_rows, 1, n_channels)
    elif layout == "ND":
        x = flat_x
    elif layout == "1ND":
        x = flat_x.view(1, n_rows, n_channels)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    weight = torch.randn((n_channels, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    initial_state = torch.randn((n_rows, n_channels, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    return x, weight, initial_state


def _make_call(
    function: Callable,
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    def call() -> tuple[torch.Tensor, torch.Tensor]:
        return function(
            x=x,
            cache=state,
            residual=None,
            weight=weight,
            bias=None,
            activation="silu",
        )

    return call


def _gate_eager_arm(
    arm: EagerArm,
    initial_state: torch.Tensor,
    reference_output: torch.Tensor,
    reference_state: torch.Tensor,
    *,
    require_native_route: bool,
) -> tuple[torch.Tensor, dict]:
    arm.state.copy_(initial_state)
    output, returned_state = arm.call()
    if returned_state is not arm.state:
        raise AssertionError(f"{arm.name} did not return the exact mutable cache object")
    if output.shape != reference_output.shape:
        raise AssertionError(f"{arm.name} output shape mismatch: expected {reference_output.shape}, got {output.shape}")
    if require_native_route and cudnn_fla.short_conv_last_path() != "native":
        raise AssertionError(f"exact shim did not take the native route: {cudnn_fla.short_conv_last_path()}")
    torch.cuda.synchronize()
    gate = _tensor_gate(arm.name, output, arm.state, reference_output, reference_state)
    gate["cache_identity"] = True
    gate["output_shape_preserved"] = True
    if require_native_route:
        gate["shim_route"] = "native"
    return output, gate


def _warm_eager(arms: tuple[EagerArm, EagerArm], initial_state: torch.Tensor, *, warmup: int) -> None:
    for arm in arms:
        for _ in range(warmup):
            arm.state.copy_(initial_state)
            output, returned_state = arm.call()
            if returned_state is not arm.state:
                raise AssertionError(f"{arm.name} did not return the exact mutable cache object")
            if arm.name == "shim" and cudnn_fla.short_conv_last_path() != "native":
                raise AssertionError(f"exact shim warmup did not take the native route: {cudnn_fla.short_conv_last_path()}")
            del output, returned_state
        torch.cuda.synchronize()


def _interleaved_eager_host_samples(
    shim: EagerArm,
    fla: EagerArm,
    initial_state: torch.Tensor,
    *,
    samples: int,
) -> tuple[list[float], list[float], list[str]]:
    """Measure warm public-call host enqueue latency; never label it kernel time."""

    arms = {shim.name: shim, fla.name: fla}
    timings = {shim.name: [], fla.name: []}
    orders = []

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for sample_index in range(samples):
            order = (shim.name, fla.name) if sample_index % 2 == 0 else (fla.name, shim.name)
            orders.append("/".join(order))
            for name in order:
                arm = arms[name]
                arm.state.copy_(initial_state)
                # Drain the reset and any prior work outside the timed region.
                torch.cuda.synchronize()
                start_ns = time.perf_counter_ns()
                output, returned_state = arm.call()
                elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
                if returned_state is not arm.state:
                    raise AssertionError(f"{name} did not return the exact mutable cache object")
                if name == shim.name and cudnn_fla.short_conv_last_path() != "native":
                    raise AssertionError(f"exact shim eager sample did not take the native route: {cudnn_fla.short_conv_last_path()}")
                timings[name].append(elapsed_us)
                # Completion and output lifetime are deliberately outside the
                # host-enqueue timer.
                torch.cuda.synchronize()
                del output, returned_state
    finally:
        if gc_was_enabled:
            gc.enable()

    return timings[shim.name], timings[fla.name], orders


def _benchmark_shape(
    shim_function: Callable,
    original_fla: Callable,
    n_rows: int,
    n_channels: int,
    *,
    samples: int,
    warmup: int,
    seed: int,
) -> dict:
    x, weight, initial_state = _make_inputs(n_rows, n_channels, layout="N1D", seed=seed)
    reference_output, reference_state = _reference_for_layout(x, weight, initial_state)

    shim_graph_state = initial_state.clone()
    fla_graph_state = initial_state.clone()
    shim_graph_call = _make_call(shim_function, x, weight, shim_graph_state)
    fla_graph_call = _make_call(original_fla, x, weight, fla_graph_state)

    shim_graph, captured_shim = _base._capture(shim_graph_call, warmup=warmup)
    shim_output, returned_shim_state = captured_shim
    if returned_shim_state is not shim_graph_state:
        raise AssertionError("captured exact shim did not return the exact mutable cache object")
    if shim_output.shape != x.shape:
        raise AssertionError(f"captured exact shim output shape mismatch: expected {x.shape}, got {shim_output.shape}")
    if cudnn_fla.short_conv_last_path() != "native":
        raise AssertionError(f"exact shim capture did not take the native route: {cudnn_fla.short_conv_last_path()}")

    fla_graph, captured_fla = _base._capture(fla_graph_call, warmup=warmup)
    fla_output, returned_fla_state = captured_fla
    if returned_fla_state is not fla_graph_state:
        raise AssertionError("captured FLA did not return the exact mutable cache object")
    if fla_output.shape != x.shape:
        raise AssertionError(f"captured FLA output shape mismatch: expected {x.shape}, got {fla_output.shape}")

    shim_graph_arm = _base.CapturedArm("shim", shim_graph, shim_output, shim_graph_state)
    fla_graph_arm = _base.CapturedArm("fla", fla_graph, fla_output, fla_graph_state)
    shim_graph_gate = _base._gate_arm(shim_graph_arm, initial_state, reference_output, reference_state)
    shim_graph_gate.update(cache_identity=True, output_shape_preserved=True, shim_route_at_capture="native")
    fla_graph_gate = _base._gate_arm(fla_graph_arm, initial_state, reference_output, reference_state)
    fla_graph_gate.update(cache_identity=True, output_shape_preserved=True)
    graph_correctness = {"shim": shim_graph_gate, "fla": fla_graph_gate}
    graph_correctness["shim_vs_fla"] = _cross_gate(
        "shim",
        shim_output,
        shim_graph_state,
        "fla",
        fla_output,
        fla_graph_state,
    )

    for arm in (shim_graph_arm, fla_graph_arm):
        for _ in range(warmup):
            arm.state.copy_(initial_state)
            arm.graph.replay()
    torch.cuda.synchronize()

    shim_graph_samples, fla_graph_samples, graph_orders = _base._interleaved_samples(
        shim_graph_arm,
        fla_graph_arm,
        initial_state,
        samples=samples,
    )
    shim_graph_summary = _base._summary(shim_graph_samples)
    fla_graph_summary = _base._summary(fla_graph_samples)
    graph_paired = [fla_us / shim_us for shim_us, fla_us in zip(shim_graph_samples, fla_graph_samples, strict=True)]

    shim_eager_state = initial_state.clone()
    fla_eager_state = initial_state.clone()
    shim_eager_arm = EagerArm("shim", _make_call(shim_function, x, weight, shim_eager_state), shim_eager_state)
    fla_eager_arm = EagerArm("fla", _make_call(original_fla, x, weight, fla_eager_state), fla_eager_state)
    _warm_eager((shim_eager_arm, fla_eager_arm), initial_state, warmup=warmup)
    shim_eager_output, shim_eager_gate = _gate_eager_arm(
        shim_eager_arm,
        initial_state,
        reference_output,
        reference_state,
        require_native_route=True,
    )
    fla_eager_output, fla_eager_gate = _gate_eager_arm(
        fla_eager_arm,
        initial_state,
        reference_output,
        reference_state,
        require_native_route=False,
    )
    eager_correctness = {
        "shim": shim_eager_gate,
        "fla": fla_eager_gate,
        "shim_vs_fla": _cross_gate(
            "shim",
            shim_eager_output,
            shim_eager_state,
            "fla",
            fla_eager_output,
            fla_eager_state,
        ),
    }
    del shim_eager_output, fla_eager_output

    shim_eager_samples, fla_eager_samples, eager_orders = _interleaved_eager_host_samples(
        shim_eager_arm,
        fla_eager_arm,
        initial_state,
        samples=samples,
    )
    shim_eager_summary = _base._summary(shim_eager_samples)
    fla_eager_summary = _base._summary(fla_eager_samples)
    eager_paired = [fla_us / shim_us for shim_us, fla_us in zip(shim_eager_samples, fla_eager_samples, strict=True)]

    return {
        "shape": {"layout": "N1D", "x": list(x.shape), "N": n_rows, "D": n_channels, "W": 4},
        "correctness": {"graph": graph_correctness, "eager": eager_correctness},
        "graph_replay_us": {
            "shim": {**shim_graph_summary, "samples_us": shim_graph_samples},
            "fla": {**fla_graph_summary, "samples_us": fla_graph_samples},
            "paired_fla_over_shim": {
                "median": statistics.median(graph_paired),
                "min": min(graph_paired),
                "max": max(graph_paired),
                "samples": graph_paired,
            },
            "sample_orders": graph_orders,
        },
        "eager_host_enqueue_us": {
            "shim": {**shim_eager_summary, "samples_us": shim_eager_samples},
            "fla": {**fla_eager_summary, "samples_us": fla_eager_samples},
            "paired_fla_over_shim": {
                "median": statistics.median(eager_paired),
                "min": min(eager_paired),
                "max": max(eager_paired),
                "samples": eager_paired,
            },
            "sample_orders": eager_orders,
            "is_kernel_time": False,
        },
    }


def _layout_smoke(
    shim_function: Callable,
    original_fla: Callable,
    *,
    layout: str,
    seed: int,
) -> dict:
    x, weight, initial_state = _make_inputs(SMOKE_N, SMOKE_D, layout=layout, seed=seed)
    reference_output, reference_state = _reference_for_layout(x, weight, initial_state)
    shim_state = initial_state.clone()
    fla_state = initial_state.clone()
    shim_arm = EagerArm("shim", _make_call(shim_function, x, weight, shim_state), shim_state)
    fla_arm = EagerArm("fla", _make_call(original_fla, x, weight, fla_state), fla_state)

    shim_output, shim_gate = _gate_eager_arm(
        shim_arm,
        initial_state,
        reference_output,
        reference_state,
        require_native_route=True,
    )
    fla_output, fla_gate = _gate_eager_arm(
        fla_arm,
        initial_state,
        reference_output,
        reference_state,
        require_native_route=False,
    )
    cross = _cross_gate("shim", shim_output, shim_state, "fla", fla_output, fla_state)
    return {
        "layout": layout,
        "shape": list(x.shape),
        "shim": shim_gate,
        "fla": fla_gate,
        "shim_vs_fla": cross,
        "cache_identity": {"shim": True, "fla": True},
        "shim_route": "native",
    }


def _validate_environment() -> None:
    _base._validate_environment()
    if _base._package_version("flash-linear-attention") != "0.5.2":
        raise RuntimeError("exact FLA short-conv shim benchmark requires flash-linear-attention==0.5.2")
    if cudnn_fla.is_accelerated("short_conv"):
        raise RuntimeError("short_conv was already accelerated before the benchmark; refusing ambiguous ownership")
    if fla_ops.causal_conv1d_update is not _base.fla_causal_conv1d_update:
        raise RuntimeError("FLA ops attribute changed before the benchmark could save the exact original callable")


def _route_and_metadata(repo: Path, benchmark_path: Path, original_fla: Callable, shim_function: Callable) -> dict:
    applied = cudnn_fla._ORIGINALS.get("short_conv")
    if applied is None:
        raise RuntimeError("short_conv registry has no applied patch after activation")
    if applied.owner is not fla_ops or applied.original is not original_fla or applied.replacement is not shim_function:
        raise RuntimeError("short_conv registry ownership does not match the live FLA ops attribute")
    if fla_ops.causal_conv1d_update is not shim_function:
        raise RuntimeError("live FLA ops attribute is not the registered exact shim replacement")
    if getattr(shim_function, "__wrapped__", None) is not original_fla:
        raise RuntimeError("exact shim does not wrap the saved FLA original")

    repo_shim_path = (repo / "python/cudnn/fla/short_conv.py").resolve()
    executed_shim_path = _base._module_path("cudnn.fla.short_conv")
    shim_code_path = _callable_code_path(shim_function)
    repo_base_benchmark_path = (repo / "benchmark/causal_conv1d_update_sm100.py").resolve()
    executed_base_benchmark_path = Path(_base.__file__).resolve()
    repo_registry_path = (repo / "python/cudnn/fla/__init__.py").resolve()
    executed_registry_path = _base._module_path("cudnn.fla")
    repo_native_api_path = (repo / "python/cudnn/causal_conv1d_update_sm100/api.py").resolve()
    repo_native_kernel_path = (repo / "python/cudnn/causal_conv1d_update_sm100/kernel.py").resolve()
    executed_native_api_path = _base._module_path("cudnn.causal_conv1d_update_sm100.api")
    executed_native_kernel_path = _base._module_path("cudnn.causal_conv1d_update_sm100.kernel")

    for name, executed, expected in (
        ("shim module", executed_shim_path, repo_shim_path),
        ("shim callable code", shim_code_path, repo_shim_path),
        ("shared benchmark helpers", executed_base_benchmark_path, repo_base_benchmark_path),
        ("FLA adapter registry", executed_registry_path, repo_registry_path),
        ("native API", executed_native_api_path, repo_native_api_path),
        ("native kernel", executed_native_kernel_path, repo_native_kernel_path),
    ):
        if _base._sha256(executed) != _base._sha256(expected):
            raise RuntimeError(f"executed {name} source does not match this worktree: {executed} != {expected}")

    fla_op_path = _base._module_path("fla.modules.conv.triton.ops")
    fla_kernel_path = _base._module_path("fla.modules.conv.triton.kernels")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    device_uuid = str(properties.uuid)
    return {
        "schema_version": 1,
        "benchmark": "fla_short_conv_exact_shim_sm100",
        "timing_contracts": {
            "graph_replay_us": "warm cache-hit CUDA-graph replay; Python, compile, capture, allocation, and state reset outside CUDA events",
            "eager_host_enqueue_us": "warm public-call host wall time through enqueue/return; includes Python validation, cache lookup, output allocation, and launch; pre-drained stream and post-timer synchronize; not kernel time",
        },
        "comparison_contract": "single process; patch once/restore once; saved FLA original versus live exact shim; AB/BA interleaved; identical BF16 x/weight/initial-state; W=4 no-bias SiLU",
        "route_proof": {
            "target": "short_conv",
            "live_ops_attribute_is_registry_replacement": True,
            "registry_owner_is_fla_ops_module": True,
            "replacement_wraps_saved_original": True,
            "is_accelerated": cudnn_fla.is_accelerated("short_conv"),
            "shim_code_path": str(shim_code_path),
            "last_path_required_per_eager_call_and_at_capture": "native",
            "restored_after_measurement": False,
        },
        "provenance": {
            "benchmark": {"path": str(benchmark_path), "sha256": _base._sha256(benchmark_path)},
            "shared_benchmark_helpers": {
                "path": str(executed_base_benchmark_path),
                "sha256": _base._sha256(executed_base_benchmark_path),
                "license": "Apache-2.0",
            },
            "shim": {"path": str(executed_shim_path), "sha256": _base._sha256(executed_shim_path), "license": "Apache-2.0"},
            "adapter_registry": {
                "path": str(executed_registry_path),
                "sha256": _base._sha256(executed_registry_path),
                "license": "Apache-2.0",
            },
            "native_api": {"path": str(executed_native_api_path), "sha256": _base._sha256(executed_native_api_path)},
            "native_kernel": {"path": str(executed_native_kernel_path), "sha256": _base._sha256(executed_native_kernel_path)},
            "fla_op": {"path": str(fla_op_path), "sha256": _base._sha256(fla_op_path), "license": "MIT"},
            "fla_kernel": {"path": str(fla_kernel_path), "sha256": _base._sha256(fla_kernel_path), "license": "MIT"},
        },
        "hardware": {
            "name": properties.name,
            "architecture": f"sm_{capability[0]}{capability[1]}",
            "compute_capability": list(capability),
            "device_index": device,
            "uuid": device_uuid,
            "driver": _base._nvidia_driver_for_uuid(device_uuid),
            "total_memory_bytes": properties.total_memory,
        },
        "slurm": _base._slurm_provenance(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version(),
            "cudnn_frontend": cudnn.__version__,
            "cudnn_backend": cudnn.backend_version(),
            "cudnn_backend_string": cudnn.backend_version_string(),
            "flash_linear_attention": _base._package_version("flash-linear-attention"),
        },
        "repository": {
            "root": str(repo),
            "head": _base._git(repo, "rev-parse", "HEAD"),
            "branch": _base._git(repo, "branch", "--show-current"),
            "dirty": bool(_base._git(repo, "status", "--porcelain")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=_base._parse_shape,
        help="N x D; repeat for multiple timed N1D shapes (default: Qwen decode matrix)",
    )
    parser.add_argument("--samples", type=int, default=51, help="timed samples per arm, metric, and shape")
    parser.add_argument("--warmup", type=int, default=10, help="warm calls/replays before measurement")
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
    benchmark_path = Path(__file__).resolve()
    original_fla = fla_ops.causal_conv1d_update
    patched = False
    result = None
    try:
        cudnn_fla.accelerate_fla(verbose=False, targets="short_conv")
        patched = True
        shim_function = fla_ops.causal_conv1d_update
        result = _route_and_metadata(repo, benchmark_path, original_fla, shim_function)
        result["parameters"] = {
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
            "timed_layout": "N1D",
            "shapes": [list(shape) for shape in shapes],
            "smoke": {"layouts": ["ND", "1ND"], "N": SMOKE_N, "D": SMOKE_D, "W": 4},
        }
        result["layout_smoke"] = [
            _layout_smoke(shim_function, original_fla, layout="ND", seed=args.seed + 1000),
            _layout_smoke(shim_function, original_fla, layout="1ND", seed=args.seed + 1001),
        ]
        result["results"] = []
        for shape_index, (n_rows, n_channels) in enumerate(shapes):
            row = _benchmark_shape(
                shim_function,
                original_fla,
                n_rows,
                n_channels,
                samples=args.samples,
                warmup=args.warmup,
                seed=args.seed + shape_index,
            )
            result["results"].append(row)
    finally:
        if patched:
            cudnn_fla.restore_fla(targets="short_conv")
            if fla_ops.causal_conv1d_update is not original_fla:
                raise RuntimeError("restore_fla did not restore the exact saved FLA original")

    if result is None:
        raise RuntimeError("benchmark produced no result")
    result["route_proof"]["restored_after_measurement"] = True
    # Emit timings only after every gate passed and the original FLA callable
    # was restored, so a partial/failed run cannot be mistaken for a result.
    for row in result["results"]:
        n_rows, n_channels = row["shape"]["N"], row["shape"]["D"]
        print(
            f"N={n_rows:3d} D={n_channels:4d}: "
            f"graph shim={row['graph_replay_us']['shim']['median_us']:.3f} us, "
            f"FLA={row['graph_replay_us']['fla']['median_us']:.3f} us, "
            f"paired={row['graph_replay_us']['paired_fla_over_shim']['median']:.3f}x; "
            f"eager-host shim={row['eager_host_enqueue_us']['shim']['median_us']:.3f} us, "
            f"FLA={row['eager_host_enqueue_us']['fla']['median_us']:.3f} us, "
            f"paired={row['eager_host_enqueue_us']['paired_fla_over_shim']['median']:.3f}x",
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
