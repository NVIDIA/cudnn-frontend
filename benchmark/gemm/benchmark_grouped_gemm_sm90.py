# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the cooperative and pingpong warp-group schedules in the SM90 grouped GEMM kernels.

DeepSeek-V3 default: H=7168, F=2048, 256 experts, top-8, one 4096-token
microbatch on each of 8 EP ranks, TP=1. Uniform routing gives 32 local
experts with M=1024 each. This is a synthetic balanced-routing benchmark.

python benchmark/gemm/benchmark_grouped_gemm_sm90.py --direction all

Both schedules share identical BF16 operands. The schedule is forced through
the test hook moe_grouped_gemm_sm90._SCHEDULE_OVERRIDE; pingpong runs its only
configuration, a 128x192 tile on a (2, 1) cluster, and cooperative a 128x256
tile on the same cluster. Input generation and JIT are outside timing. Class
execution uses fixed output buffers and a preallocated workspace; graph timing
is device time for an actual captured bundle, while eager timing includes host
dispatch and synchronization. Routing and FC2 forward are excluded.
Forward measures the FC1 GEMM without GLU (N=2F, K=H).
Glu measures FC1 forward + SwiGLU, writing C.
Dglu measures FC2 dgrad + dSwiGLU + dprob, including its buffer reset, without
dbias; its kernel has one (cooperative) schedule and ignores the override.
Glu and dglu use --glu-layout (interleaved_32 by default, or gate_then_up).
Dgrad measures the FC2 dgrad (N=F, K=H), reading the forward weights N-major.
Schedule checks are exact.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import statistics
import time
from pathlib import Path

import torch
import cudnn
from cudnn.gemm.cutedsl.grouped.unfused import moe_grouped_gemm_sm90

SCHEDULES = ("cooperative", "pingpong")
#: direction -> its public wrapper, its class API and the schedules its kernel runs.
DIRECTIONS = dict(
    forward=("grouped_gemm_wrapper_sm90", "GroupedGemmSm90", SCHEDULES),
    glu=("grouped_gemm_glu_wrapper_sm90", "GroupedGemmGluSm90", SCHEDULES),
    dglu=("grouped_gemm_dglu_wrapper_sm90", "GroupedGemmDgluSm90", ("cooperative",)),
    dgrad=("grouped_gemm_wrapper_sm90", "GroupedGemmSm90", SCHEDULES),
)


def problem(args, direction, schedule):
    torch.manual_seed(args.seed)
    groups = 256 // args.ep
    valid_m = math.ceil(args.context * args.microbatch * args.ep * 8 / 256)
    m = math.ceil(valid_m / 256) * 256
    h, f = 7168, 2048
    n, k = (2 * f, h) if direction in ("forward", "glu") else (f, h)
    rows = groups * m
    # Pingpong runs only its 128x192 tile on a (2, 1) cluster.
    tile_n = 192 if schedule == "pingpong" else 256
    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.125
    kwargs = dict(
        a_tensor=a,
        padded_offsets=torch.arange(1, groups + 1, device="cuda", dtype=torch.int32) * m,
        alpha_tensor=torch.ones(groups, device="cuda"),
        prob_tensor=torch.ones(rows, device="cuda"),
        mma_tiler_mn=(128, tile_n),
        cluster_shape_mn=(2, 1),
        use_dynamic_sched=True,
    )
    if valid_m != m:
        for expert in range(groups):
            a[expert * m + valid_m : (expert + 1) * m].zero_()
            kwargs["prob_tensor"][expert * m + valid_m : (expert + 1) * m].zero_()
    if direction == "forward":
        kwargs.update(
            b_tensor=torch.randn(groups, n, k, device="cuda", dtype=torch.bfloat16) * 0.125,
            b_major="k",
            # Without generate_c the kernel does not write C, but the class API takes a sample.
            c_tensor=torch.empty(rows, n, device="cuda", dtype=torch.bfloat16),
        )
    if direction == "glu":
        kwargs.update(
            b_tensor=torch.randn(groups, n, k, device="cuda", dtype=torch.bfloat16) * 0.125,
            b_major="k",
            sfa_tensor=None,
            c_dtype=torch.bfloat16,
            generate_c=True,
            act_func="swiglu",
            glu_layout=args.glu_layout,
        )
    if direction == "dglu":
        kwargs.update(
            b_tensor=torch.randn(groups, k, n, device="cuda", dtype=torch.bfloat16) * 0.125,
            b_major="n",
            sfa_tensor=None,
            # The forward's saved C (gate | up, 2N wide, in --glu-layout order).
            c_tensor=torch.randn(rows, 2 * n, device="cuda", dtype=torch.bfloat16) * 0.125,
            beta_tensor=torch.ones(groups, device="cuda"),
            dprob_tensor=torch.zeros(rows, device="cuda", dtype=torch.float32),
            act_func="dswiglu",
            glu_layout=args.glu_layout,
        )
    if direction == "dgrad":
        kwargs.update(
            # The forward weights (E, K, N), read N-major.
            b_tensor=torch.randn(groups, k, n, device="cuda", dtype=torch.bfloat16) * 0.125,
            b_major="n",
            c_tensor=torch.empty(rows, n, device="cuda", dtype=torch.bfloat16),
        )
    metadata = dict(groups=groups, M=m, valid_M=valid_m, N=n, K=k, total_rows=rows, mma_tiler_mn=(128, tile_n), cluster_shape_mn=kwargs["cluster_shape_mn"])
    return kwargs, metadata


def prepare(kwargs, direction, schedule):
    wrapper, cls = (getattr(cudnn, name) for name in DIRECTIONS[direction][:2])
    # Test hook: every SM90 plan built while it is set runs this schedule; restored once the plans are built.
    saved, moe_grouped_gemm_sm90._SCHEDULE_OVERRIDE = moe_grouped_gemm_sm90._SCHEDULE_OVERRIDE, schedule
    try:
        outputs = wrapper(**kwargs)
        operands = {**kwargs, **{name: value for name, value in outputs.items() if value is not None}}
        samples = {"sample_" + name.removesuffix("_tensor"): value for name, value in operands.items()}
        constructor = inspect.signature(cls).parameters
        init = {name: kwargs[name] for name in constructor if name in kwargs}
        for name, parameter in constructor.items():
            if name.startswith("sample_"):
                if name in samples:
                    init[name] = samples[name]
                elif parameter.default is inspect.Parameter.empty:
                    init[name] = None
        api = cls(**init)
        if not api.check_support():
            raise RuntimeError(f"Unsupported requested configuration: {direction} {schedule}")
        api.compile()
    finally:
        moe_grouped_gemm_sm90._SCHEDULE_OVERRIDE = saved
    # Bind through the real backend's signature; no wrapper allocation in timing.
    implementation = api._implementation
    execute = implementation.execute
    arguments = {name: operands[name] for name in inspect.signature(execute).parameters if name in operands}
    for name, parameter in inspect.signature(execute).parameters.items():
        if name not in arguments and parameter.default is inspect.Parameter.empty:
            arguments[name] = None
    if "workspace" in inspect.signature(execute).parameters:
        arguments["workspace"] = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    if "_workspace" in inspect.signature(execute).parameters:
        # The kernel scratch the wrapper allocates per call.
        arguments["_workspace"] = torch.empty(implementation._scratch_nbytes, device="cuda", dtype=torch.uint8)

    def run():
        if kwargs.get("dprob_tensor") is not None:
            kwargs["dprob_tensor"].zero_()
        execute(**arguments)

    run()
    torch.cuda.synchronize()
    return run, outputs, api


def snapshots(outputs):
    # The wrapper returns c_tensor=None without generate_c, so forward/dgrad's empty C sample is never compared or hashed.
    return {name: value.detach().clone() for name, value in outputs.items() if value is not None}


def check(actual, expected):
    for name, target in expected.items():
        value = actual[name]
        if not torch.isfinite(value.float()).all():
            raise AssertionError(f"Nonfinite {name}")
        torch.testing.assert_close(value.float(), target.float(), rtol=0, atol=0)


def output_hashes(values):
    """Cross-process controls; copying/hashing is outside timing.

    dprob uses an existing atomic reduction. Its bytes are not a deterministic
    regression contract.
    """
    return {
        name: hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        for name, value in values.items()
        if name != "dprob_tensor"
    }


def distribution(samples):
    quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
    return dict(median=statistics.median(samples), q25=quartiles[0], q75=quartiles[2], samples=samples)


def measure(runs, args):
    # Compile both schedules before warming or sampling either of them. Adjacent
    # pairs alternate AB/BA so clock drift cannot separate the two schedule
    # measurement windows.
    graphs, events = {}, {}
    samples = {schedule: dict(graph_device_us=[], eager_wall_us=[]) for schedule in runs}
    for schedule, run in runs.items():
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.bundle):
                run()
        graphs[schedule] = graph
        events[schedule] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for _ in range(args.warmup):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    for repeat in range(args.repeats):
        order = tuple(runs) if repeat % 2 == 0 else tuple(reversed(runs))
        for schedule in order:
            begin, end = events[schedule]
            begin.record()
            graphs[schedule].replay()
            end.record()
            end.synchronize()
            samples[schedule]["graph_device_us"].append(begin.elapsed_time(end) * 1000 / args.bundle)
        for schedule in order:
            start = time.perf_counter()
            for _ in range(args.bundle):
                runs[schedule]()
            torch.cuda.synchronize()
            samples[schedule]["eager_wall_us"].append((time.perf_counter() - start) * 1e6 / args.bundle)
    measurements = {}
    for schedule, metrics in samples.items():
        measurements[schedule] = {name: statistics.median(values) for name, values in metrics.items()}
        measurements[schedule]["distributions"] = {name: distribution(values) for name, values in metrics.items()}
    if "cooperative" in runs and "pingpong" in runs:
        measurements["pingpong"]["paired_ratio_to_cooperative"] = {
            name: distribution([p / c for p, c in zip(samples["pingpong"][name], values)]) for name, values in samples["cooperative"].items()
        }
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("all", *DIRECTIONS), default="all")
    parser.add_argument("--schedule", choices=("compare", *SCHEDULES), default="compare")
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--bundle", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--glu-layout", choices=("interleaved_32", "gate_then_up"), default="interleaved_32")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.ep < 1 or 256 % args.ep or min(args.context, args.microbatch, args.warmup, args.bundle, args.repeats) < 1:
        parser.error("EP must divide 256; context, microbatch and timing counts must be positive")
    if args.direction == "dglu" and args.schedule == "pingpong":
        parser.error("dglu has one (cooperative) schedule")
    runtime = dict(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn_source=cudnn.__file__,
        cudnn_frontend=cudnn.__version__,
        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
    )
    print(json.dumps(dict(runtime=runtime, configuration=vars(args)), default=str), flush=True)
    directions = tuple(DIRECTIONS) if args.direction == "all" else (args.direction,)
    rows = []
    for direction in directions:
        expected = None
        runs, cases = {}, {}
        for schedule in (schedule for schedule in DIRECTIONS[direction][2] if args.schedule in ("compare", schedule)):
            kwargs, shape = problem(args, direction, schedule)
            run, outputs, api = prepare(kwargs, direction, schedule)
            values = snapshots(outputs)
            if expected is None:
                expected = values
            else:
                check(values, expected)
            runs[schedule] = run
            cases[schedule] = dict(direction=direction, schedule=schedule, **shape, output_sha256=output_hashes(values))
            del run, outputs, api, values, kwargs
        measurements = measure(runs, args)
        for schedule, case in cases.items():
            row = {**case, **measurements[schedule]}
            rows.append(row)
            print(json.dumps(row), flush=True)
        del expected, runs, cases
    report = dict(runtime=runtime, configuration=vars(args), measurements=rows)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
