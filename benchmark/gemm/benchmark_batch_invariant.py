# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure batch-invariant GEMM forward against native cuDNN on the same graph.

    python benchmark/gemm/benchmark_batch_invariant.py --formats bf16 mxfp8 --m 1 64 256 1024 --nk 4096 4096
    python benchmark/gemm/benchmark_batch_invariant.py --nk 128 8192 --flashinfer-legacy --csv gemm.csv

For each (format, N, K), all M share the same weights, scales and activation rows.
The native baseline uses the first buildable backend plan recommended for the
graph without the numerical note. --flashinfer-legacy compares the original
FlashInfer kernels directly. Each timed CUDA graph replay includes the full
forward, including any conversions or split-K reduction performed by that
backend. JIT, input packing, capture and bitwise comparisons precede timing.
Shared rows and the original FlashInfer kernels must agree bitwise. The native
baseline must produce finite results with a relative L2 error at most 0.02.
"""

import argparse
import csv
import statistics
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch

import cudnn
from cudnn.engines import is_backend_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/python/gemm/batch_invariant"))
from quantized_matmul import FORMATS, TENSOR_SCALE, make_case, quantized_graph


def bf16_graph(inputs, out):
    a, b = inputs
    graph = cudnn.pygraph(compute_data_type=cudnn.data_type.FLOAT)
    x, w = graph.tensor_like(a.unsqueeze(0)), graph.tensor_like(b.unsqueeze(0))
    y = graph.matmul(A=x, B=w).set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return graph, {x: a.unsqueeze(0), w: b.unsqueeze(0), y: out.unsqueeze(0)}


def problems(fmt, sizes, n, k, quant_dtype):
    sizes = sorted(set(sizes), reverse=True)
    if fmt == "bf16":
        rng = torch.Generator(device="cuda").manual_seed(913)
        activations = torch.randn((sizes[0], k), device="cuda", dtype=torch.bfloat16, generator=rng)
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=rng).T
        for m in sizes:
            inputs = (activations[:m], weight)
            out = activations.new_empty((m, n))
            yield inputs, out, partial(bf16_graph, inputs, out), None
        return

    full = make_case(fmt, "cuda", m=sizes[0], n=n, k=k, batch=1)
    for m in sizes:
        case = full if m == sizes[0] else full.select_rows(range(m))
        inputs = case.inputs()
        out = torch.empty((*case.x.shape[:-1], n), device=case.x.device, dtype=quant_dtype)
        yield inputs, out, partial(quantized_graph, case, inputs, out, rank_three=True), case


def cudnn_plan(make_graph, out, *, native):
    graph, bindings = make_graph()
    if native:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        reason = "no native cuDNN plan proposed"
        for index, plan in enumerate(graph.plans):
            if is_backend_engine(plan.engine_id):
                try:
                    graph.build_plan_at_index(index)
                except cudnn.cudnnGraphNotSupportedError as exc:
                    reason = str(exc)
                    continue
                engine = graph.get_plan_name_at_index(index)
                break
        else:
            raise cudnn.cudnnGraphNotSupportedError(reason)
    else:
        graph.select_numeric_notes([cudnn.numerical_note.BATCH_INVARIANT])
        graph.build()
        engine = graph.selected_engine.name if graph.selected_engine is not None else None
        if engine != "batch_invariant_gemm":
            raise RuntimeError(f"BATCH_INVARIANT selected {engine!r}")
    workspace = torch.empty(graph.get_workspace_size(), device=out.device, dtype=torch.uint8)
    return engine, partial(graph.execute, bindings, workspace)


def flashinfer_plan(fmt, inputs, out, case, k):
    from flashinfer.gemm.batch_invariant import api

    m, n = out.shape[-2:]
    kw = {"out": out}
    cls = fmt
    if fmt == "bf16":
        call = api.mm_bf16_batch_invariant
    elif fmt in TENSOR_SCALE:
        call = api.bmm_fp8_batch_invariant
        kw["dtype"] = out.dtype
    else:
        kw["out_dtype"] = out.dtype
        if fmt == "groupwise":
            a, b, sa, sb = inputs
            return partial(api.gemm_fp8_nt_groupwise_batch_invariant, a, b.T, sa, sb, scale_major_mode="K", **kw)
        if fmt == "mxfp8":
            call = api.mm_mxfp8_batch_invariant
            cls = api.mx_format_for(out.device)
        elif fmt == "mxfp8_mxfp4":
            call = api.mm_mxfp8_mxfp4_batch_invariant
            cls = api.mxfp4w8a_format_for(out.device)
            kw["alpha"] = case.alpha
        else:
            call = api.mm_fp4_batch_invariant
            use_nvfp4 = fmt == "nvfp4"
            cls = api.fp4_format_for(out.device, use_nvfp4, n, k)
            kw.update(alpha=case.alpha, use_nvfp4=use_nvfp4, block_size=case.block)
    chunks = api.chunks_for(n, k, cls)
    if chunks > 1:
        kw["workspace"] = torch.empty((chunks, m, n), device=out.device, dtype=torch.float32)
    return partial(call, *inputs, **kw)


def check_output(out, expected, label):
    if not torch.isfinite(out).all().item():
        raise ValueError(f"{label} produced a non-finite or unwritten result")
    if expected is not None and not torch.equal(out.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise ValueError(f"{label} differs bitwise from the batch-invariant result")


def check_native(out, expected):
    check_output(out, None, "native")
    # A different reduction order can change individual outputs near cancellation.
    reference = expected.float()
    relative_error = ((out.float() - reference).norm() / reference.norm().clamp_min(1e-12)).item()
    if relative_error > 0.02:
        raise ValueError(f"native result differs from the batch-invariant result (relative L2 error {relative_error:.4g})")


def measure(runs, out, expected, args):
    captures = {}
    for name, (_, run) in runs.items():
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.bundle):
                run()
        out.fill_(float("nan"))
        graph.replay()
        if name == "native":
            check_native(out, expected)
        else:
            check_output(out, expected, name)
        captures[name] = graph

    samples = {name: [] for name in captures}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for i in range(args.repeats):
        order = captures if i % 2 == 0 else reversed(captures)
        for name in order:
            start.record()
            captures[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.bundle)
    return {name: statistics.median(times) for name, times in samples.items()}


def benchmark(writer, fmt, n, k, args):
    reference = None
    quant_dtype = torch.bfloat16 if args.quant_output == "bf16" else torch.float16
    for inputs, out, make_graph, case in problems(fmt, args.m, n, k, quant_dtype):
        m = out.size(-2)
        runs = {"batch_invariant": cudnn_plan(make_graph, out, native=False)}
        out.fill_(float("nan"))
        runs["batch_invariant"][1]()
        check_output(out, None if reference is None else reference[..., :m, :], f"{fmt} M={m}")
        if reference is None:
            reference = out.clone()

        missing = None
        try:
            runs["native"] = cudnn_plan(make_graph, out, native=True)
        except cudnn.cudnnGraphNotSupportedError as exc:
            lines = str(exc).splitlines()
            reason = next((line.split("Reason: ", 1)[1] for line in lines if "Reason: " in line), lines[0])
            missing = reason.split("the backend declined: ")[-1][:250]
        if args.flashinfer_legacy:
            runs["flashinfer_legacy"] = ("flashinfer", flashinfer_plan(fmt, inputs, out, case, k))

        us = measure(runs, out, reference[..., :m, :], args)
        prefix = [fmt, str(out.dtype).removeprefix("torch."), m, n, k]
        for name, elapsed in us.items():
            tflops = 2 * m * n * k / (elapsed * 1e6)
            writer.writerow([*prefix, name, runs[name][0], f"{elapsed:.3f}", f"{tflops:.3f}", f"{elapsed / us['batch_invariant']:.3f}", "OK"])
        if missing is not None:
            writer.writerow([*prefix, "native", "", "", "", "", "N/A: " + missing])


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--formats", nargs="+", choices=("bf16", *FORMATS), default=("bf16", *FORMATS))
    p.add_argument("--m", nargs="+", type=positive_int, default=[1, 64, 256, 1024], help="Activation row counts compared for each weight shape")
    p.add_argument(
        "--nk", action="append", nargs=2, type=positive_int, metavar=("N", "K"), help="Repeat for additional weight shapes. Defaults: 4096 4096 and 128 8192"
    )
    p.add_argument("--quant-output", choices=("bf16", "fp16"), default="bf16", help="Output type for quantized inputs. The BF16 input path returns BF16.")
    p.add_argument("--flashinfer-legacy", action="store_true", help="Compare and bitwise-check the original FlashInfer implementations")
    p.add_argument("--warmup", type=positive_int, default=3)
    p.add_argument("--bundle", type=positive_int, default=8, help="Forward calls per CUDA graph replay")
    p.add_argument("--repeats", type=positive_int, default=9, help="Timed replays per backend")
    p.add_argument("--csv", type=Path, help="Write CSV to this file (default: stdout)")
    args = p.parse_args()
    shapes = args.nk or [(4096, 4096), (128, 8192)]
    print(
        f"GPU={torch.cuda.get_device_name()} CC={torch.cuda.get_device_capability()} torch={torch.__version__} CUDA={torch.version.cuda} "
        f"cuDNN={cudnn.backend_version_string()} warmup={args.warmup} bundle={args.bundle} repeats={args.repeats}",
        file=sys.stderr,
    )
    output = args.csv.open("w", newline="") if args.csv else nullcontext(sys.stdout)
    with output as dest:
        writer = csv.writer(dest)
        writer.writerow(
            ["format", "out_dtype", "m", "n", "k", "backend", "engine", "median_us", "effective_tflops", "latency_ratio_to_batch_invariant", "status"]
        )
        for fmt in args.formats:
            for n, k in shapes:
                benchmark(writer, fmt, n, k, args)
                dest.flush()


if __name__ == "__main__":
    main()
