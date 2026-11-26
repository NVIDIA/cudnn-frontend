# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Benchmark the FROST weight-gradient kernel against native cuDNN with nsys.

The shape is ``N,C,D,H,W,K,T,R,S``: the image ``X`` and the output gradient
``dY`` are the operands, ``dW`` is the ``K,C,T,R,S`` filter gradient. Inputs and
``dY`` use compact channels-last storage while retaining cuDNN's logical
NCDHW/KCTRS shapes, and the reduction is ``L = N*Z*P*Q`` over the output pixels.

Requires an SM100 or SM103 GPU and Nsight Systems on ``PATH``. Nothing else about
the machine is assumed: shape, padding, stride, dilation and dtype all come from
the command line, and no peak rate or clock frequency is built in.

Methodology:

* ``--warmup`` warmup iterations (default 10), then ``--iters`` measured
  iterations (default 100) with automatic tensor-buffer rotation, so every
  measured launch reads a buffer that is cold in L2.
* Nsight Systems measures GPU kernel time over a ``cudaProfilerStart`` range:
  graph construction, JIT compilation, allocation and warmup are excluded, and
  the reported figure is the summed kernel time divided by the iterations.
* The cuDNN baseline is the first supported native plan from heuristic mode A.
* Speedup is cuDNN time divided by FROST time.
* FLOPs are ``2 * (N * Z * P * Q) * K * (C * T * R * S)``; achieved TFLOPS is
  FLOPs over the measured kernel time, so no peak rate is assumed.

``--shape`` is required and may be repeated to measure several shapes under the
same padding, stride and dilation. The weight-gradient template serves FP16 and
BF16 operands only: there is no NVFP4/block-scaled weight gradient.

FROST is measured through the public engine selection (``create_execution_plan``
with the weight-gradient engine id and ``TILE_M``/``TILE_N``/``TILEK``/
``CTA_GROUP``/``STAGES`` knobs), not through the heuristic ranking: the
weight-gradient heuristics deliberately place the FROST plan behind the backend
plan, so a fixed-configuration comparison has to pin it. Both engines are
prepared and launched through the same graph contract, exactly as a caller would.

The result is a table on stdout: one section per tile configuration with the
FROST and cuDNN kernel time, achieved TFLOPS, speedup and the SM clock sampled
while that measurement ran (from nvidia-smi, reported as ``-`` when that is
unavailable). No JSON, CSV or Markdown artifact is written and no run directory
is created; nsys reports are temporary unless ``-o PREFIX`` retains them.

Examples:

    # one shape, one fixed tile, BF16
    python benchmark/conv/frost/benchmark_conv_wgrad.py \
        --shape 1,1024,1,44,80,1024,3,3,3 --pre-padding 2,1,1 --post-padding 0,1,1

    # the same shape, correctness-checked before timing
    python benchmark/conv/frost/benchmark_conv_wgrad.py \
        --shape 1,1024,1,44,80,1024,3,3,3 --pre-padding 2,1,1 --post-padding 0,1,1 \
        --check-correctness

    # a small tile sweep over two shapes
    python benchmark/conv/frost/benchmark_conv_wgrad.py \
        --shape 1,1024,2,88,160,1024,3,3,3 --shape 1,1024,1,88,160,1024,3,3,3 \
        --pre-padding 2,1,1 --post-padding 0,1,1 \
        --configs default,CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

# FROST convolution is opt-in while it is experimental. Respect an explicit
# user choice, but make the benchmark work without a separate environment knob.
os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

import cudnn
import cutlass
import torch

from cudnn.conv.frost.tile_config import ConvTileConfig, by_name
from cudnn.conv.frost.utils import _channels_last_3d_stride
from cudnn.engines.engine_ids import FROST_CONV_WGRAD_ID_BASE

from benchmark_utils import (
    ConvShape,
    build_cudnn_plan,
    frost_conv_unsupported_reason,
    output_spatial as _output_spatial,
    print_kernel_breakdown,
    profile,
    resolve_nbuf as _resolve_nbuf,
    triplet,
    validate_args,
)

# Default tile configuration for ``--configs default``: 128x256x256 CTA tile,
# 128x256x32 MMA shape, 2x1 cluster, 2-CTA MMA.
FIXED_CONFIG = "CONFIG_sm100_128x256x256_128x256x32_cluster2x1_2ctamma"

# torch dtype -> (cuDNN data type)
DTYPES = {
    "bf16": (torch.bfloat16, cudnn.data_type.BFLOAT16),
    "fp16": (torch.float16, cudnn.data_type.HALF),
}
# The template's ``check_config`` takes the DSL output dtype, not the torch one.
FROST_OUTPUT_DTYPE = {"bf16": cutlass.BFloat16, "fp16": cutlass.Float16}


@dataclass(frozen=True)
class Case:
    """One measured shape with everything derived from the geometry."""

    name: str
    shape: ConvShape
    spatial: tuple[int, int, int]
    flops: int
    per_set_bytes: int


def _cases(args) -> list[Case]:
    """Derive the measured geometry for every requested shape."""
    item_size = DTYPES[args.dtype][0].itemsize
    cases = []
    for index, shape in enumerate(args.shape, start=1):
        try:
            spatial = _output_spatial(shape, args.pre_padding, args.post_padding, args.stride, args.dilation)
        except ValueError as exc:
            raise ValueError(f"--shape {shape}: {exc}") from exc
        flops = 2 * shape.n * shape.k * spatial[0] * spatial[1] * spatial[2] * shape.c * shape.t * shape.r * shape.s
        x_elements = shape.n * shape.c * shape.d * shape.h * shape.w
        dy_elements = shape.n * shape.k * spatial[0] * spatial[1] * spatial[2]
        dw_elements = shape.k * shape.c * shape.t * shape.r * shape.s
        per_set = item_size * (x_elements + dy_elements + dw_elements)
        cases.append(Case(f"S{index}", shape, spatial, flops, per_set))
    return cases


def _make_data(case: Case, torch_dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape, (z, p, q) = case.shape, case.spatial
    image = torch.empty((shape.n, shape.c, shape.d, shape.h, shape.w), dtype=torch_dtype, device="cuda", memory_format=torch.channels_last_3d)
    loss = torch.empty((shape.n, shape.k, z, p, q), dtype=torch_dtype, device="cuda", memory_format=torch.channels_last_3d)
    weight = torch.empty((shape.k, shape.c, shape.t, shape.r, shape.s), dtype=torch_dtype, device="cuda", memory_format=torch.channels_last_3d)
    return image, loss, weight


def _declaration(case: Case, fe_dtype, args):
    """Declare X + dY -> dW on the graph API: the one contract both engines serve."""
    shape = case.shape
    xshape = (shape.n, shape.c, shape.d, shape.h, shape.w)
    yshape = (shape.n, shape.k, *case.spatial)
    wshape = (shape.k, shape.c, shape.t, shape.r, shape.s)
    graph = cudnn.pygraph(
        io_data_type=fe_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = graph.tensor(dim=xshape, stride=_channels_last_3d_stride(xshape), data_type=fe_dtype)
    dy = graph.tensor(dim=yshape, stride=_channels_last_3d_stride(yshape), data_type=fe_dtype)
    dw = graph.conv_wgrad(
        x,
        dy,
        pre_padding=args.pre_padding,
        post_padding=args.post_padding,
        stride=args.stride,
        dilation=args.dilation,
    )
    dw.set_output(True).set_data_type(fe_dtype).set_dim(wshape).set_stride(_channels_last_3d_stride(wshape))
    return graph, (x, dy, dw)


def _knobs(config: ConvTileConfig) -> dict:
    k = cudnn.knob_type
    return {
        k.TILE_M: config.cta_tile_m,
        k.TILE_N: config.cta_tile_n,
        k.TILEK: config.cta_tile_k_bytes,
        k.CTA_GROUP: config.cta_group,
        k.STAGES: 6,
    }


def _build_frost(config_name: str, case: Case, fe_dtype, args):
    """Pin the weight-gradient engine to one tile configuration of the catalog."""
    graph, (x, dy, dw) = _declaration(case, fe_dtype, args)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(FROST_CONV_WGRAD_ID_BASE, _knobs(by_name(config_name)))
    graph.select_plan(len(graph.plans) - 1)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")

    def execute(data):
        image, loss, weight = data
        graph.execute({x: image, dy: loss, dw: weight}, workspace)

    return graph.get_plan_name_at_index(len(graph.plans) - 1), execute


def _build_cudnn(case: Case, fe_dtype, args):
    """The baseline: first supported native plan from heuristic mode A."""
    graph, (x, dy, dw) = _declaration(case, fe_dtype, args)
    workspace, name = build_cudnn_plan(graph)

    def execute(data):
        image, loss, weight = data
        graph.execute({x: image, dy: loss, dw: weight}, workspace)

    return name, execute


def _check_correctness(args, case: Case, data, torch_dtype, fe_dtype) -> None:
    """Compare FROST against the native cuDNN backend outside the capture range."""
    image, loss, weight = data
    generator = torch.Generator(device="cuda").manual_seed(0)
    image.normal_(generator=generator)
    # Keep the filter-gradient scale near one as the reduction grows.
    reduction = case.shape.c * case.shape.t * case.shape.r * case.shape.s
    loss.normal_(std=reduction**-0.5, generator=generator)
    reference = torch.full_like(weight, float("nan"))
    _, cudnn_execute = _build_cudnn(case, fe_dtype, args)
    cudnn_execute((image, loss, reference))
    # Detect missing stores as well as incorrect values.
    weight.fill_(float("nan"))
    _, frost_execute = _build_frost(args._tile_config, case, fe_dtype, args)
    frost_execute((image, loss, weight))
    torch.cuda.synchronize()
    tolerance = 2 * torch.finfo(torch_dtype).eps
    try:
        torch.testing.assert_close(weight, reference, rtol=tolerance, atol=tolerance)
    except AssertionError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise RuntimeError(f"correctness FAILED: {args._tile_config} vs cuDNN backend (rtol=atol={tolerance})") from exc
    print(f"[correctness] {args._tile_config}: PASS vs cuDNN backend (rtol=atol={tolerance})", flush=True)


def _worker(args, case: Case, nbuf: int) -> int:
    torch_dtype, fe_dtype = DTYPES[args.dtype]
    warmup_data = _make_data(case, torch_dtype)

    if args._implementation == "frost":
        plan_name, execute = _build_frost(args._tile_config, case, fe_dtype, args)
        if args.check_correctness:
            _check_correctness(args, case, warmup_data, torch_dtype, fe_dtype)
    else:
        plan_name, execute = _build_cudnn(case, fe_dtype, args)

    pool = [_make_data(case, torch_dtype) for _ in range(nbuf)]
    for _ in range(args.warmup):
        execute(warmup_data)
    torch.cuda.synchronize()
    print(f"[worker] {args._implementation}: selected {plan_name!r}; capturing {args.iters} iterations", flush=True)

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        for iteration in range(args.iters):
            execute(pool[iteration % nbuf])
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()
    return 0


def _parse_configs(spec: str) -> list[str]:
    names = []
    for token in (value.strip() for value in spec.split(",")):
        if not token:
            continue
        name = FIXED_CONFIG if token.lower() == "default" else token
        by_name(name)  # unknown names fail here, before any GPU work
        names.append(name)
    if not names:
        raise SystemExit("--configs selected no tile configuration")
    return list(dict.fromkeys(names))


def _shell(command: Sequence[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001 - context only, never fatal
        return ""


def _mhz(text: Optional[str]) -> Optional[int]:
    digits = "".join(character for character in (text or "") if character.isdigit())
    return int(digits) if digits else None


def _device_uuid(index: Optional[int] = None) -> str:
    """UUID of a visible device, as nvidia-smi prints it without its ``GPU-`` prefix."""
    index = torch.cuda.current_device() if index is None else index
    return str(getattr(torch.cuda.get_device_properties(index), "uuid", "") or "")


def _nvidia_smi_rows(fields: str) -> dict[str, list[str]]:
    """Query nvidia-smi fields per GPU, keyed by UUID; empty when nvidia-smi is unusable."""
    text = _shell(["nvidia-smi", f"--query-gpu=uuid,{fields}", "--format=csv,noheader,nounits"])
    rows = {}
    for line in text.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) > 1:
            rows[parts[0].removeprefix("GPU-")] = parts[1:]
    return rows


def _mhz_text(value: Optional[int]) -> str:
    return f"{value} MHz" if value is not None else "unknown"


def _clock_status(args) -> dict:
    """SM clock of the benchmark device, and whether it matches the reference clock."""
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    sm_clock, max_sm_clock, mem_clock = _nvidia_smi_rows("clocks.sm,clocks.max.sm,clocks.mem").get(_device_uuid(index), (None, None, None))
    device = {
        "index": index,
        "name": properties.name,
        "sm_clock_mhz": _mhz(sm_clock),
        "max_sm_clock_mhz": _mhz(max_sm_clock),
        "mem_clock_mhz": _mhz(mem_clock),
    }
    expected = args.expect_clock_mhz
    locked = None if not expected else device["max_sm_clock_mhz"] == expected
    return {"expected_sm_clock_mhz": expected, "locked": locked, "device": device}


def _print_clock_status(clocks: dict) -> None:
    device = clocks["device"]
    print(
        f"  [clock: gpu{device['index']} {device['name']} sm={_mhz_text(device['sm_clock_mhz'])} "
        f"max_sm={_mhz_text(device['max_sm_clock_mhz'])} mem={_mhz_text(device['mem_clock_mhz'])}]"
    )
    if clocks["locked"] is False:
        expected = clocks["expected_sm_clock_mhz"]
        print(f"  [WARNING: max SM clock is not the {expected} MHz reference; lock the clock (nvidia-smi -lgc {expected},{expected}) for comparable runs]")
    elif clocks["locked"]:
        print(f"  [clock: max SM clock matches the {clocks['expected_sm_clock_mhz']} MHz reference]")


class _ClockSampler:
    """Sample the benchmark device's SM clock and power while the benchmark runs.

    Nsight measures GPU kernel time, which says nothing about the clock the
    kernel ran at. On an unlocked GPU two measurements taken seconds apart can
    sit at different boost states, so every measurement records the clock it
    actually ran at. The window spans the whole nsys invocation (process start,
    JIT, the measured loop), so the peak sample is the clock the loop ran at.
    """

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.uuid = _device_uuid()
        self.samples: list[tuple[float, Optional[int], Optional[float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="clock-sampler", daemon=True)

    def _sample(self) -> tuple[Optional[int], Optional[float]]:
        rows = _nvidia_smi_rows("clocks.sm,power.draw")
        # Report the device this process runs on, not whichever GPU nvidia-smi lists first.
        fields = rows.get(self.uuid, []) if self.uuid else next(iter(rows.values()), [])
        power = None
        if len(fields) > 1:
            try:
                power = float(fields[1])
            except ValueError:
                power = None
        return (_mhz(fields[0]) if fields else None), power

    def _run(self) -> None:
        while not self._stop.is_set():
            clock, power = self._sample()
            self.samples.append((time.monotonic(), clock, power))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def window(self, start: float, end: float) -> dict:
        """Clock/power statistics over one measurement window."""
        window = [sample for sample in self.samples if start <= sample[0] <= end]
        if not window:
            return {}
        clocks = [clock for _, clock, _ in window if clock]
        powers = [power for _, _, power in window if power is not None]
        return {
            "samples": len(window),
            "sm_clock_mhz_mean": round(statistics.fmean(clocks)) if clocks else None,
            "sm_clock_mhz_min": min(clocks) if clocks else None,
            "sm_clock_mhz_max": max(clocks) if clocks else None,
            "power_w_mean": round(statistics.fmean(powers), 1) if powers else None,
        }


def _tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def _clock_mhz(record: Optional[dict]) -> str:
    value = ((record or {}).get("clock") or {}).get("sm_clock_mhz_max")
    return str(value) if value else "-"


def _clock_text(record: Optional[dict]) -> str:
    value = ((record or {}).get("clock") or {}).get("sm_clock_mhz_max")
    return f"{value:6d}" if value else "     -"


def _row(case: Case, frost: dict, baseline: Optional[dict]) -> str:
    frost_us = frost["kernel_ms"] * 1e3
    if baseline is None:
        return f"  {case.name:6s} {frost_us:10.1f} {frost['tflops']:10.2f} {'':>10s} {'':>10s} {'':>8s} {_clock_text(frost)}"
    base_us = baseline["kernel_ms"] * 1e3
    return (
        f"  {case.name:6s} {frost_us:10.1f} {frost['tflops']:10.2f} "
        f"{base_us:10.1f} {baseline['tflops']:10.2f} {baseline['kernel_ms'] / frost['kernel_ms']:7.2f}x {_clock_text(frost)}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shape",
        type=ConvShape.parse,
        action="append",
        required=True,
        metavar="N,C,D,H,W,K,T,R,S",
        help="shape to benchmark; repeat the flag to measure several shapes under the same padding, stride and dilation",
    )
    parser.add_argument("--pre-padding", type=triplet, default=(0, 0, 0), help="lower D,H,W padding (default: 0,0,0)")
    parser.add_argument("--post-padding", type=triplet, default=(0, 0, 0), help="upper D,H,W padding (default: 0,0,0)")
    parser.add_argument("--stride", type=triplet, default=(1, 1, 1), help="D,H,W stride (default: 1,1,1)")
    parser.add_argument("--dilation", type=triplet, default=(1, 1, 1), help="D,H,W dilation (default: 1,1,1)")
    parser.add_argument("--dtype", default="bf16", choices=tuple(DTYPES), help="operand dtype (default: bf16)")
    parser.add_argument("--configs", default="default", help=f"comma-separated tile config names; 'default' is {FIXED_CONFIG}")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--rotate-buffers",
        default="auto",
        metavar="N",
        help="number of independent tensor sets to rotate; 'auto' makes the pool larger than L2 (default: auto)",
    )
    parser.add_argument(
        "--expect-clock-mhz",
        type=int,
        default=None,
        help="max SM clock that --require-clock demands and reports when it is wrong (default: no reference clock)",
    )
    parser.add_argument("--require-clock", action="store_true", help="refuse to run unless the max SM clock equals --expect-clock-mhz")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PREFIX",
        help="keep the nsys reports under PREFIX, one per shape and tile configuration (default: temporary reports)",
    )
    parser.add_argument("--check-correctness", action="store_true", help="check each FROST config against the cuDNN backend before timing")
    parser.add_argument("--_nsys-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_implementation", choices=("frost", "cudnn"), help=argparse.SUPPRESS)
    parser.add_argument("--_tile-config", help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true", help="print worker output, kernel breakdowns and full failures")
    return parser


def _print_table(cases: Sequence[Case], configs: Sequence[str], cudnn_results: dict, per_config: dict) -> None:
    """Print one FROST-versus-cuDNN table per tile configuration."""
    header = f"  {'shape':6s} {'FROST µs':>10s} {'TFLOPS':>10s} {'cuDNN µs':>10s} {'TFLOPS':>10s} {'speedup':>8s} {'SM MHz':>7s}"
    rule = "=" * len(header)
    print("\n" + rule)
    for index, config_name in enumerate(configs):
        measured = per_config[config_name]
        if index:
            print()
        print(f"  {config_name}")
        print(header)
        for case in cases:
            frost = measured.get(case.name)
            if frost is None:
                print(f"  {case.name:6s} {'FAILED':>10s}")
                continue
            print(_row(case, frost, cudnn_results.get(case.name)))
        ratios = [
            cudnn_results[case.name]["kernel_ms"] / measured[case.name]["kernel_ms"] for case in cases if case.name in measured and case.name in cudnn_results
        ]
        if ratios:
            print(f"  geomean speedup vs cuDNN: {(torch.tensor(ratios).log().mean().exp().item()):.2f}x over {len(ratios)} shape(s)")
    print(rule)


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        print("No CUDA device is available.", file=sys.stderr)
        return 1
    reason = frost_conv_unsupported_reason()
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1
    try:
        cases = _cases(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    validate_args(args)
    if args.require_clock and args.expect_clock_mhz is None:
        raise SystemExit("--require-clock needs --expect-clock-mhz")

    if args._nsys_worker:
        case = cases[0]
        nbuf = _resolve_nbuf(args.rotate_buffers, case.per_set_bytes)
        return _worker(args, case, nbuf)

    gpu_name = torch.cuda.get_device_name()
    configs = _parse_configs(args.configs)

    from cudnn.conv.frost.templates.sm100_conv_wgrad import check_config

    for name in configs:
        try:
            check_config(by_name(name), FROST_OUTPUT_DTYPE[args.dtype])
        except ValueError as exc:
            raise SystemExit(f"{name}: {exc}") from exc

    clocks = _clock_status(args)
    sampler = _ClockSampler()
    print(f"\n=== conv wgrad NCDHW -> KCTRS, {args.dtype.upper()} ===")
    print(f"  [gpu: {gpu_name}, sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}, {torch.cuda.device_count()} visible]")
    _print_clock_status(clocks)
    print("  [timer: nsys cudaProfilerApi capture range, cuda_gpu_kern_sum; graph build/JIT/allocation/warmup excluded]")
    print(f"  [clock: sampled every {sampler.interval:.1f} s; the SM MHz column is the highest SM clock sampled during that measurement]")
    print(f"  [warmup: {args.warmup}; measured iterations: {args.iters}]")
    print("  [speedup: cuDNN kernel time / FROST kernel time]")
    print(f"  [{len(cases)} shape(s), {len(configs)} tile configuration(s)]")
    if args.check_correctness:
        print("  [correctness: each FROST config checked against the cuDNN backend before the capture range; a mismatch fails that config]")
    if args.require_clock and clocks["locked"] is not True:
        print(
            f"  [refusing to run: the max SM clock is not the {args.expect_clock_mhz} MHz reference. "
            f"Lock it with `nvidia-smi -lgc {args.expect_clock_mhz},{args.expect_clock_mhz}`, or drop --require-clock "
            f"to measure at the current clock]",
            file=sys.stderr,
        )
        return 1

    cudnn_results: dict = {}
    per_config: dict = {name: {} for name in configs}
    nsys_base = None if args.output is None else os.path.abspath(os.path.expanduser(args.output))
    status = 0
    sampler.start()

    try:
        for case in cases:
            shape = case.shape
            nbuf = _resolve_nbuf(args.rotate_buffers, case.per_set_bytes)
            print(
                f"\n--- {case.name}: NCDHW={shape.n}x{shape.c}x{shape.d}x{shape.h}x{shape.w} "
                f"KCTRS={shape.k}x{shape.c}x{shape.t}x{shape.r}x{shape.s} "
                f"-> ZPQ={case.spatial[0]}x{case.spatial[1]}x{case.spatial[2]} "
                f"L={case.shape.n * case.spatial[0] * case.spatial[1] * case.spatial[2]} "
                f"[{case.flops / 1e9:.1f} GFLOP, {nbuf} buffer set(s)]",
                flush=True,
            )
            args.output = None if nsys_base is None else os.path.join(nsys_base, case.name)
            started = time.monotonic()
            try:
                stats = profile(__file__, "cudnn", args, shape, nbuf, extra_inner=["--dtype", args.dtype])
                kernel_ms = sum(item.total_ms for item in stats) / args.iters
                cudnn_results[case.name] = {
                    "plan": next((item.name for item in stats if item.instances), None),
                    "kernel_ms": kernel_ms,
                    "tflops": _tflops(case.flops, kernel_ms),
                    "clock": sampler.window(started, time.monotonic()),
                }
                if args.verbose:
                    print_kernel_breakdown("cuDNN", stats, args.iters)
            except RuntimeError as exc:
                if args.verbose:
                    print(str(exc), file=sys.stderr, flush=True)
                print(f"  [native cuDNN baseline unavailable: {str(exc).splitlines()[-1][:160]}]")

            for index, config_name in enumerate(configs, start=1):
                print(f"  [{index:02d}/{len(configs):02d}] profiling {config_name}", flush=True)
                started = time.monotonic()
                try:
                    stats = profile(__file__, "frost", args, shape, nbuf, tile_config=config_name, extra_inner=["--dtype", args.dtype])
                    kernel_ms = sum(item.total_ms for item in stats) / args.iters
                    tflops = _tflops(case.flops, kernel_ms)
                    per_config[config_name][case.name] = {
                        "kernel_ms": kernel_ms,
                        "tflops": tflops,
                        "clock": sampler.window(started, time.monotonic()),
                    }
                    if args.verbose:
                        print_kernel_breakdown(config_name, stats, args.iters)
                except RuntimeError as exc:
                    if args.verbose:
                        print(str(exc), file=sys.stderr, flush=True)
                    print(f"  [failed: {str(exc).splitlines()[-1][:160]}]", flush=True)
                    status = 1

            if case is cases[0] and case.name not in cudnn_results and not any(case.name in per_config[name] for name in configs):
                print(
                    "  [aborting after the first shape: every engine failed, so the environment (Nsight Systems, the GPU, or the "
                    "requested clock lock) is the problem rather than the kernel; see the messages above]",
                    file=sys.stderr,
                )
                return 1
    finally:
        sampler.stop()

    _print_table(cases, configs, cudnn_results, per_config)
    return status


if __name__ == "__main__":
    sys.exit(main())
