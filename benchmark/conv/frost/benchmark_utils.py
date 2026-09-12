# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared infrastructure for the Frost convolution sweep benchmarks."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import cudnn
import torch

from cudnn.conv.frost.tile_config import ConvTileConfig, by_name
from cudnn.engines.engine_ids import is_backend_engine

_L2_BYTES = 126 * 1024 * 1024  # B200 class
_AUTO_POOL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024
_AUTO_NBUF_CAP = 1024


@dataclass(frozen=True)
class ConvShape:
    n: int
    c: int
    d: int
    h: int
    w: int
    k: int
    t: int
    r: int
    s: int

    @classmethod
    def parse(cls, text: str) -> "ConvShape":
        try:
            values = tuple(int(value.strip()) for value in text.split(","))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--shape must contain integers") from exc
        if len(values) != 9:
            raise argparse.ArgumentTypeError("--shape must be N,C,D,H,W,K,T,R,S (nine values)")
        if any(value <= 0 for value in values):
            raise argparse.ArgumentTypeError("all --shape extents must be positive")
        return cls(*values)

    def __str__(self) -> str:
        return ",".join(str(value) for value in (self.n, self.c, self.d, self.h, self.w, self.k, self.t, self.r, self.s))


@dataclass(frozen=True)
class KernelStats:
    name: str
    total_ms: float
    instances: int


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    elapsed_ms: float
    stats: tuple[KernelStats, ...] = ()
    error: str = ""


def select_tile_configs(spec: Optional[str], heuristic_config: str, configs: Sequence[ConvTileConfig]) -> list[str]:
    """Resolve ``--configs`` into compatible catalog names in catalog order."""
    config_by_name = {config.name: config for config in configs}
    if spec is None or spec.strip().lower() == "all":
        return list(config_by_name)
    if spec.strip().lower() == "default":
        return [heuristic_config]

    selected = []
    for token in (value.strip() for value in spec.split(",")):
        if not token:
            continue
        if token.lower() == "default":
            selected.append(heuristic_config)
            continue
        if any(char in token for char in "*?["):
            matches = [name for name in config_by_name if fnmatch.fnmatchcase(name, token)]
            if not matches:
                raise ValueError(f"--configs pattern {token!r} matched no convolution tile config")
            selected.extend(matches)
            continue
        try:
            config = by_name(token)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown convolution tile config {token!r}") from exc
        if config.name not in config_by_name:
            raise ValueError(f"convolution tile config {token!r} is not sweepable for this shape")
        selected.append(config.name)
    if not selected:
        raise ValueError("--configs selected no convolution tile configs")
    return list(dict.fromkeys(selected))


def triplet(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated integers") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return values


def output_spatial(
    shape: ConvShape,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[int, int, int]:
    out = tuple(
        (image + lower + upper - dil * (filt - 1) - 1) // step + 1
        for image, filt, lower, upper, step, dil in zip(
            (shape.d, shape.h, shape.w),
            (shape.t, shape.r, shape.s),
            pre_padding,
            post_padding,
            stride,
            dilation,
        )
    )
    if any(value <= 0 for value in out):
        raise ValueError(f"convolution parameters produce an invalid output shape {out}")
    return out


def channels_last_stride(shape: Sequence[int]) -> tuple[int, ...]:
    """Return a compact NDHWC physical stride for a logical NCDHW shape."""
    _n, c, d, h, w = shape
    return (c * d * h * w, 1, h * w * c, w * c, c)


def resolve_nbuf(spec: str, per_set_bytes: int) -> int:
    if spec.strip().lower() != "auto":
        try:
            return max(1, int(spec))
        except ValueError as exc:
            raise ValueError("--rotate-buffers must be 'auto' or a positive integer") from exc

    nbuf = max(2, -(-int(1.5 * _L2_BYTES) // per_set_bytes))
    budget = _AUTO_POOL_BUDGET_BYTES
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        budget = min(budget, free // 2)
    return max(1, min(nbuf, max(1, budget // per_set_bytes), _AUTO_NBUF_CAP))


def build_cudnn_plan(graph):
    """Build the first usable native cuDNN plan, excluding OSS delegates."""
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])

    candidates = [i for i, plan in enumerate(graph.plans) if is_backend_engine(plan.engine_id) and plan.cpp_index is not None]
    if not candidates:
        names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
        raise RuntimeError(f"no cudnn plan is available; graph plans: {names}")

    print("available plans:", [graph.get_plan_name_at_index(i) for i in candidates])
    failures = []
    for index in candidates:
        name = graph.get_plan_name_at_index(index)
        try:
            graph.select_plan(index)
            graph.check_support()
            graph.build_plans()
            workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
            return workspace, name
        except (NotImplementedError, cudnn.cudnnGraphNotSupportedError) as exc:
            failures.append(f"{name}: {exc}")
    raise RuntimeError(f"no cudnn plan could be built: {'; '.join(failures)}")


def parse_nsys_stats(text: str) -> list[KernelStats]:
    lines = text.splitlines()
    header_index = None
    columns = None
    for index, line in enumerate(lines):
        row = next(csv.reader([line]), [])
        if "Name" in row and "Instances" in row and any(re.fullmatch(r"Total Time \(\w+\)", column) for column in row):
            header_index, columns = index, row
            break
    if header_index is None or columns is None:
        raise RuntimeError("could not find the cuda_gpu_kern_sum CSV header in nsys output")

    total_index = next(i for i, column in enumerate(columns) if re.fullmatch(r"Total Time \(\w+\)", column))
    instances_index = columns.index("Instances")
    name_index = columns.index("Name")
    unit = re.fullmatch(r"Total Time \((\w+)\)", columns[total_index]).group(1)
    per_ms = {"ns": 1e6, "us": 1e3, "ms": 1.0, "s": 1e-3}.get(unit)
    if per_ms is None:
        raise RuntimeError(f"unsupported nsys time unit {unit!r}")

    stats = []
    for row in csv.reader(lines[header_index + 1 :]):
        if len(row) <= max(total_index, instances_index, name_index):
            continue
        try:
            total_ms = float(row[total_index].replace(",", "")) / per_ms
            instances = int(row[instances_index].replace(",", ""))
        except ValueError:
            continue
        stats.append(KernelStats(row[name_index], total_ms, instances))
    if not stats:
        raise RuntimeError("nsys captured no CUDA kernels")
    return stats


def _report_tag(implementation: str, tile_config: Optional[str]) -> str:
    if tile_config is None:
        return implementation
    return f"{implementation}_{re.sub(r'[^A-Za-z0-9_.-]', '_', tile_config)}"


@contextmanager
def _report_prefix(script: str, output: Optional[str], implementation: str, tile_config: Optional[str] = None):
    tag = _report_tag(implementation, tile_config)
    if output is None:
        stem = os.path.splitext(os.path.basename(script))[0]
        with tempfile.TemporaryDirectory(prefix=f"{stem}_{tag}_") as workdir:
            yield os.path.join(workdir, "report")
        return

    base = os.path.abspath(os.path.expanduser(output))
    if base.endswith(".nsys-rep"):
        base = base[: -len(".nsys-rep")]
    report_prefix = f"{base}_{tag}"
    os.makedirs(os.path.dirname(report_prefix), exist_ok=True)
    yield report_prefix


def profile(
    script: str,
    implementation: str,
    args,
    shape: ConvShape,
    nbuf: int,
    tile_config: Optional[str] = None,
) -> list[KernelStats]:
    nsys = "/usr/local/bin/nsys" if os.path.exists("/usr/local/bin/nsys") else shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("nsys not found; install Nsight Systems to run this benchmark")

    inner = [
        sys.executable,
        "-u",
        os.path.abspath(script),
        "--_nsys-worker",
        "--_implementation",
        implementation,
        "--shape",
        str(shape),
        "--pre-padding",
        ",".join(map(str, args.pre_padding)),
        "--post-padding",
        ",".join(map(str, args.post_padding)),
        "--stride",
        ",".join(map(str, args.stride)),
        "--dilation",
        ",".join(map(str, args.dilation)),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--rotate-buffers",
        str(nbuf),
    ]
    if getattr(args, "check_correctness", False):
        inner += ["--check-correctness"]
    if tile_config is not None:
        inner += ["--_tile-config", tile_config]

    with _report_prefix(script, args.output, implementation, tile_config) as report_prefix:
        profile_command = [
            nsys,
            "profile",
            "-o",
            report_prefix,
            "--force-overwrite=true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-um-cpu-page-faults=false",
            "--cuda-um-gpu-page-faults=false",
            "--trace=cuda",
            *inner,
        ]
        print(f"  + {shlex.join(profile_command)}", flush=True)
        env = os.environ.copy()
        env.setdefault("TMPDIR", tempfile.gettempdir())

        if args.verbose:
            result = subprocess.run(profile_command, stdout=None, stderr=subprocess.PIPE, text=True, env=env)
        else:
            result = subprocess.run(profile_command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            label = tile_config or implementation
            raise RuntimeError(
                f"nsys profile for {label} exited {result.returncode}\n" f"stdout:\n{'See above' if args.verbose else result.stdout}\nstderr:\n{result.stderr}"
            )

        stats_command = [
            nsys,
            "stats",
            "--report",
            "cuda_gpu_kern_sum",
            "--format",
            "csv",
            "--force-export=true",
            report_prefix + ".nsys-rep",
        ]
        result = subprocess.run(stats_command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"nsys stats for {implementation} exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        if args.output is not None:
            print(f"  [saved nsys report: {report_prefix}.nsys-rep]")
        return parse_nsys_stats(result.stdout)


def print_kernel_breakdown(implementation: str, stats: Sequence[KernelStats], iters: int) -> None:
    print(f"\n  {implementation} captured kernels:")
    for item in sorted(stats, key=lambda value: value.total_ms, reverse=True):
        name = item.name if len(item.name) <= 96 else item.name[:93] + "..."
        print(f"    {item.total_ms / iters:8.4f} ms/iteration  {item.instances:4d} launches  {name}")


def make_parser(description: str, default_shape: str, configs_help: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--shape", type=ConvShape.parse, default=ConvShape.parse(default_shape), help="N,C,D,H,W,K,T,R,S")
    parser.add_argument("--pre-padding", type=triplet, default=(0, 0, 0), help="lower D,H,W padding (default: 0,0,0)")
    parser.add_argument("--post-padding", type=triplet, default=(0, 0, 0), help="upper D,H,W padding (default: 0,0,0)")
    parser.add_argument("--stride", type=triplet, default=(1, 1, 1), help="D,H,W stride (default: 1,1,1)")
    parser.add_argument("--dilation", type=triplet, default=(1, 1, 1), help="D,H,W dilation (default: 1,1,1)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--configs", default="all", metavar="CONFIGS", help=configs_help)
    parser.add_argument(
        "--rotate-buffers",
        default="auto",
        metavar="N",
        help="number of independent tensor sets to rotate; 'auto' makes the pool larger than L2 (default: auto)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PREFIX",
        help="keep PREFIX_cudnn.nsys-rep and one PREFIX_frost_CONFIG_*.nsys-rep per tile (default: temporary reports)",
    )
    parser.add_argument("--_nsys-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_implementation", choices=("frost", "cudnn"), help=argparse.SUPPRESS)
    parser.add_argument("--_tile-config", help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true", help="If enabled, print more debug messages")
    return parser


def validate_args(args) -> None:
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if any(value < 0 for value in (*args.pre_padding, *args.post_padding)):
        raise SystemExit("padding values must be non-negative")
    if any(value <= 0 for value in (*args.stride, *args.dilation)):
        raise SystemExit("stride and dilation values must be positive")
    if args._nsys_worker and args._implementation is None:
        raise SystemExit("internal worker mode requires --_implementation")
    if args._nsys_worker and args._implementation == "frost" and args._tile_config is None:
        raise SystemExit("internal Frost worker mode requires --_tile-config")


def run_sweep(
    script: str,
    args,
    shape: ConvShape,
    nbuf: int,
    config_names: Sequence[str],
    flops: int,
    *,
    allow_missing_cudnn: bool = False,
) -> int:
    """Profile the native baseline and every selected Frost tile, then report."""
    cudnn_ms = None
    try:
        cudnn_stats = profile(script, "cudnn", args, shape, nbuf)
        cudnn_ms = sum(item.total_ms for item in cudnn_stats) / args.iters
        print_kernel_breakdown("cuDNN", cudnn_stats, args.iters)
    except RuntimeError as exc:
        if not allow_missing_cudnn:
            raise
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
        message = lines[-1] if lines else type(exc).__name__
        print(f"  [native cuDNN baseline unavailable: {message[:160]}]", flush=True)

    frost_results: list[BenchmarkResult] = []
    for index, config_name in enumerate(config_names, start=1):
        print(f"\n  [{index:02d}/{len(config_names):02d}] profiling {config_name}", flush=True)
        try:
            stats = profile(script, "frost", args, shape, nbuf, tile_config=config_name)
            elapsed_ms = sum(item.total_ms for item in stats) / args.iters
            result = BenchmarkResult(config_name, elapsed_ms, tuple(stats))
            if args.verbose:
                print_kernel_breakdown(config_name, stats, args.iters)
        except RuntimeError as exc:
            lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
            message = lines[-1] if lines else type(exc).__name__
            result = BenchmarkResult(config_name, float("inf"), error=message[:160])
            print(f"  [failed: {result.error}]", flush=True)
        frost_results.append(result)

    successful = sorted((result for result in frost_results if not result.error), key=lambda result: result.elapsed_ms)
    failed = [result for result in frost_results if result.error]
    ordered = successful + failed
    width = max(18, min(96, max(len(result.name) for result in frost_results)))
    rule = "=" * (width + (43 if cudnn_ms is not None else 26))
    print("\n" + rule)
    if cudnn_ms is not None:
        print(f"  {'configuration':{width}s} {'TFLOPS':>10s} {'kernel ms':>11s} {'vs cuDNN':>10s}")
    else:
        print(f"  {'configuration':{width}s} {'TFLOPS':>10s} {'kernel ms':>11s}")
    print(rule)
    if cudnn_ms is not None:
        cudnn_tflops = flops / (cudnn_ms * 1e-3) / 1e12
        print(f"  {'cuDNN backend':{width}s} {cudnn_tflops:10.2f} {cudnn_ms:11.4f} {1.0:9.2f}x")
    for result in ordered:
        if result.error:
            suffix = f"  ERR {result.error}" if cudnn_ms is not None else f" ERR {result.error}"
            print(f"  {result.name:{width}s} {'':>10s} {'':>11s}{suffix}")
            continue
        tflops = flops / (result.elapsed_ms * 1e-3) / 1e12
        if cudnn_ms is not None:
            ratio = cudnn_ms / result.elapsed_ms
            print(f"  {result.name:{width}s} {tflops:10.2f} {result.elapsed_ms:11.4f} {ratio:9.2f}x")
        else:
            print(f"  {result.name:{width}s} {tflops:10.2f} {result.elapsed_ms:11.4f}")
    print(rule)

    if successful:
        best = successful[0]
        comparison = f", {cudnn_ms / best.elapsed_ms:.2f}x vs cuDNN" if cudnn_ms is not None else ""
        print(f"  best: {best.name} ({best.elapsed_ms:.4f} ms{comparison})")
        if failed:
            print(f"  failed configs: {len(failed)}/{len(frost_results)}")
        return 0
    print("  no Frost tile configuration completed successfully", file=sys.stderr)
    return 1
