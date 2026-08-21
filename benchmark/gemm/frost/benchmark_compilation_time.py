# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure no-cache compile time for representative GEMM graphs.

Run this from the cudnn-frontend repo root:
    python benchmark/gemm/frost/benchmark_compilation_time.py --case matmul --shape 1,1024,1024,1024 --warmup 2 --iters 5

Pick a graph with ``--case``. The script sets up the compiler and CUDA once,
then compiles in a loop with CuTe DSL caches turned off. It reads the
``[JIT-TIMER]`` logs for the main JIT stages and reports them under the elapsed
``jit_from_cudnn_graph`` call. Graph construction happens before that timer.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from io import StringIO
import os
import re
import statistics
import tempfile
import time
from dataclasses import dataclass

_CONFIG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1"
_CASES = ("matmul", "block_scale_matmul", "moe_grouped_matmul")
_JIT_TIMER = re.compile(r"\[JIT-TIMER\] Function: (\S+) \| Execution Time: ([0-9.]+) (\S+)")
_UNIT_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "\N{MICRO SIGN}s": 1e-3,
    "\N{GREEK SMALL LETTER MU}s": 1e-3,
    "ms": 1.0,
    "s": 1e3,
}


@dataclass(frozen=True)
class CompilationTiming:
    elapsed_ms: float
    graph_analysis_validation_ms: float
    codegen_render_ms: float
    module_import_ms: float
    mod_compile_ms: float
    python_to_ir_ms: float
    compile_and_jit_ms: float
    lookup_ms: float

    @property
    def total_ms(self) -> float:
        return self.python_to_ir_ms + self.compile_and_jit_ms + self.lookup_ms

    @property
    def other_python_ms(self) -> float:
        return self.elapsed_ms - self.graph_analysis_validation_ms - self.codegen_render_ms - self.module_import_ms - self.mod_compile_ms

    @property
    def mod_compile_other_ms(self) -> float:
        return self.mod_compile_ms - self.total_ms


def _build_graph(case: str, batch: int, m: int, n: int, k: int):
    import cudnn

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    if case == "matmul":
        a = graph.tensor(name="A", dim=[batch, m, k], stride=[m * k, k, 1])
        b = graph.tensor(name="B", dim=[batch, k, n], stride=[k * n, 1, k])
        c = graph.matmul(A=a, B=b, name="mm")
    elif case == "block_scale_matmul":
        if k % 16:
            raise ValueError("block_scale_matmul requires K to be divisible by 16")
        sf_k = k // 16
        a = graph.tensor(
            name="A",
            dim=[batch, m, k],
            stride=[m * k, k, 1],
            data_type=cudnn.data_type.FP4_E2M1,
        )
        b = graph.tensor(
            name="B",
            dim=[batch, k, n],
            stride=[k * n, 1, k],
            data_type=cudnn.data_type.FP4_E2M1,
        )
        sfa = graph.tensor(
            name="SFA",
            dim=[batch, m, sf_k],
            stride=[m * sf_k, sf_k, 1],
            data_type=cudnn.data_type.FP8_E4M3,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        sfb = graph.tensor(
            name="SFB",
            dim=[batch, sf_k, n],
            stride=[sf_k * n, 1, sf_k],
            data_type=cudnn.data_type.FP8_E4M3,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        a = graph.block_scale_dequantize(input=a, descale=sfa, block_size=[1, 16])
        b = graph.block_scale_dequantize(input=b, descale=sfb, block_size=[16, 1])
        c = graph.matmul(A=a, B=b, name="mm")
    elif case == "moe_grouped_matmul":
        tokens = batch * m
        token = graph.tensor(name="token", dim=[1, tokens, k], stride=[tokens * k, k, 1])
        weight = graph.tensor(name="weight", dim=[batch, k, n], stride=[k * n, 1, k])
        first_token_offset = graph.tensor(
            name="first_token_offset",
            dim=[batch, 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        c = graph.moe_grouped_matmul(
            token,
            weight,
            first_token_offset,
            mode=cudnn.moe_grouped_matmul_mode.NONE,
            compute_data_type=cudnn.data_type.FLOAT,
            name="moe",
        )

    c.set_data_type(cudnn.data_type.BFLOAT16)
    c.set_output(True)
    return graph


def _initialize() -> None:
    import cutlass
    import cudnn.gemm.frost  # noqa: F401

    cutlass.cuda.initialize_cuda_context()


def _compile(case: str, batch: int, m: int, n: int, k: int) -> tuple[int, float]:
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    graph = _build_graph(case, batch, m, n, k)
    start = time.perf_counter_ns()
    jit_from_cudnn_graph(
        graph,
        config=by_name(_CONFIG),
        cta_group=2,
    )
    return start, (time.perf_counter_ns() - start) / 1e6


def _parse_timing(
    output: str,
    *,
    elapsed_ms: float,
    graph_analysis_validation_ms: float,
    codegen_render_ms: float,
    module_import_ms: float,
    mod_compile_ms: float,
) -> CompilationTiming:
    stages: dict[str, list[float]] = {}
    for stage, duration, unit in _JIT_TIMER.findall(output):
        if unit not in _UNIT_TO_MS:
            raise RuntimeError(f"unsupported CuTe DSL JIT timer unit: {unit!r}")
        stages.setdefault(stage, []).append(float(duration) * _UNIT_TO_MS[unit])

    required = ("build_ir_module", "compile_and_jit", "lookup")
    invalid = {stage: stages.get(stage, []) for stage in required if len(stages.get(stage, [])) != 1}
    if invalid:
        raise RuntimeError(f"expected one timing for each CuTe DSL JIT stage, got {invalid}:\n" f"{output}")

    return CompilationTiming(
        elapsed_ms=elapsed_ms,
        graph_analysis_validation_ms=graph_analysis_validation_ms,
        codegen_render_ms=codegen_render_ms,
        module_import_ms=module_import_ms,
        mod_compile_ms=mod_compile_ms,
        python_to_ir_ms=stages["build_ir_module"][0],
        compile_and_jit_ms=stages["compile_and_jit"][0],
        lookup_ms=stages["lookup"][0],
    )


def _measure(case: str, batch: int, m: int, n: int, k: int, count: int, *, verbose: bool) -> list[CompilationTiming]:
    from cudnn.gemm.frost import compiler

    stage_times: dict[str, float] = {}
    stage_starts: dict[str, int] = {}

    def timed(name: str, fn, *, record_start: bool = False):
        def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            if record_start:
                stage_starts[name] = start
            try:
                return fn(*args, **kwargs)
            finally:
                stage_times[name] = (time.perf_counter_ns() - start) / 1e6

        return wrapper

    original_generate = compiler.generate
    original_render = compiler._render_template
    original_block_scale_render = compiler._render_block_scale_template
    original_import = compiler._import_kernel

    def timed_import(*args, **kwargs):
        start = time.perf_counter_ns()
        try:
            module = original_import(*args, **kwargs)
        finally:
            stage_times["import"] = (time.perf_counter_ns() - start) / 1e6
        module.compile = timed("mod_compile", module.compile)
        return module

    with tempfile.TemporaryDirectory(prefix=f"compile-{case}-") as kernel_cache:
        os.environ.update(
            {
                "CUDNN_FRONTEND_GEMM_KERNEL_CACHE": kernel_cache,
                "CUTE_DSL_DISABLE_FILE_CACHING": "1",
                "CUTE_DSL_NO_CACHE": "1",
                "CUTE_DSL_JIT_TIME_PROFILING": "1",
            }
        )
        log = StringIO()
        with redirect_stderr(log):
            _initialize()
            if verbose:
                print(log.getvalue(), end="")
            offset = log.tell()
            timings = []
            compiler.generate = timed("generate", original_generate, record_start=True)
            compiler._render_template = timed("render", original_render)
            compiler._render_block_scale_template = timed("render", original_block_scale_render)
            compiler._import_kernel = timed_import
            try:
                for _ in range(count):
                    stage_times.clear()
                    stage_starts.clear()
                    jit_start, elapsed_ms = _compile(case, batch, m, n, k)
                    expected_stages = {"generate", "render", "import", "mod_compile"}
                    if stage_times.keys() != expected_stages:
                        raise RuntimeError("expected one codegen, render, import, and mod.compile " f"call per iteration, got {sorted(stage_times)}")

                    output = log.getvalue()[offset:]
                    if verbose:
                        print(output, end="")
                    timings.append(
                        _parse_timing(
                            output,
                            elapsed_ms=elapsed_ms,
                            graph_analysis_validation_ms=(stage_starts["generate"] - jit_start) / 1e6,
                            codegen_render_ms=(stage_times["generate"] + stage_times["render"]),
                            module_import_ms=stage_times["import"],
                            mod_compile_ms=stage_times["mod_compile"],
                        )
                    )
                    offset = log.tell()
            finally:
                compiler.generate = original_generate
                compiler._render_template = original_render
                compiler._render_block_scale_template = original_block_scale_render
                compiler._import_kernel = original_import
    return timings


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must contain integers") from error
    if len(shape) != 4 or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape must be B,M,N,K with positive dimensions")
    return shape


def _format_timing(timing: CompilationTiming) -> str:
    return (
        f"  jit_from_cudnn_graph elapsed    {timing.elapsed_ms:9.3f} ms\n"
        f"  ├── graph analysis / validation {timing.graph_analysis_validation_ms:9.3f} ms\n"
        f"  ├── codegen / render             {timing.codegen_render_ms:9.3f} ms\n"
        f"  ├── generated-module import      {timing.module_import_ms:9.3f} ms\n"
        f"  ├── other Python overhead        {timing.other_python_ms:9.3f} ms\n"
        f"  └── mod.compile()                {timing.mod_compile_ms:9.3f} ms\n"
        f"      ├── build_ir_module          {timing.python_to_ir_ms:9.3f} ms  Python -> IR\n"
        f"      ├── compile_and_jit          {timing.compile_and_jit_ms:9.3f} ms  IR -> cubin\n"
        f"      ├── lookup                   {timing.lookup_ms:9.3f} ms  resolve entry symbol\n"
        f"      └── other                    {timing.mod_compile_other_ms:9.3f} ms"
    )


def _summary(name: str, samples: list[float]) -> str:
    return (
        f"{name:<38} median={statistics.median(samples):9.3f} ms  "
        f"min={min(samples):9.3f} ms  max={max(samples):9.3f} ms  "
        f"mean={statistics.mean(samples):9.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--case",
        choices=_CASES,
        default="matmul",
        help="graph to compile (default: matmul)",
    )
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        default=(1, 1024, 1024, 1024),
        help=("shape as B,M,N,K; for moe_grouped_matmul, B is groups and M is " "tokens per group " "(default: 1,1024,1024,1024)"),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="runs excluded from summary statistics (default: 2)",
    )
    parser.add_argument("--iters", type=int, default=5, help="measured runs (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="print complete CuTe DSL logs")
    args = parser.parse_args()
    batch, m, n, k = args.shape

    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be non-negative and --iters must be positive")

    print(f"Case: {args.case}")
    if args.case == "moe_grouped_matmul":
        print(f"Shape: groups={batch}, tokens/group={m}, N={n}, K={k}")
    else:
        print(f"Shape: B={batch}, M={m}, N={n}, K={k}")
    print(f"Config: {_CONFIG}, cta_group=2")
    count = args.warmup + args.iters

    timings = _measure(args.case, batch, m, n, k, count, verbose=args.verbose)
    for index, timing in enumerate(timings[: args.warmup], 1):
        print(f"warmup {index}/{args.warmup}:")
        print(_format_timing(timing))

    samples = timings[args.warmup :]
    for index, timing in enumerate(samples, 1):
        print(f"sample {index}/{args.iters}:")
        print(_format_timing(timing))

    print("\nSummary")
    rows = (
        ("jit_from_cudnn_graph elapsed", "elapsed_ms"),
        ("├── graph analysis / validation", "graph_analysis_validation_ms"),
        ("├── codegen / render", "codegen_render_ms"),
        ("├── generated-module import", "module_import_ms"),
        ("├── other Python overhead", "other_python_ms"),
        ("└── mod.compile()", "mod_compile_ms"),
        ("    ├── build_ir_module", "python_to_ir_ms"),
        ("    ├── compile_and_jit", "compile_and_jit_ms"),
        ("    ├── lookup", "lookup_ms"),
        ("    └── other", "mod_compile_other_ms"),
    )
    for label, field in rows:
        print(_summary(label, [getattr(sample, field) for sample in samples]))


if __name__ == "__main__":
    main()
