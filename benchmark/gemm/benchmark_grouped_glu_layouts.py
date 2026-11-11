# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare persistent GLU layouts in the existing grouped GEMM kernels.

DeepSeek-V3 default: H=7168, F=2048, 256 experts, top-8, one 4096-token
microbatch on each of 8 EP ranks, TP=1. Uniform routing gives 32 local
experts with M=1024 each. This is a synthetic balanced-routing benchmark.

python benchmark/gemm/benchmark_grouped_glu_layouts.py --precision all

Both layouts share identical logical payloads/scales. Conversion and JIT are
outside timing. Class execution uses fixed output buffers; graph timing is
device time for an actual captured bundle, while eager timing includes host
dispatch and synchronization. Routing, weight quantization and FC2 forward
are excluded. Backward measures FC2 dgrad + dSwiGLU + dprob, including its
buffer reset, without dbias. Input-layout checks are exact except atomic dprob.
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


def paired(tensor, axis, *, inverse=False, block=32):
    """Test/initialization-only permutation; never executed in timed regions."""
    moved = tensor.movedim(axis, -1)
    shape = moved.shape
    byte_view = moved.view(torch.uint8) if tensor.element_size() == 1 else moved
    if inverse:
        value = byte_view.reshape(*shape[:-1], shape[-1] // (2 * block), 2, block).transpose(-3, -2)
    else:
        value = byte_view.reshape(*shape[:-1], 2, shape[-1] // (2 * block), block).transpose(-3, -2)
    return value.contiguous().reshape(shape).view(tensor.dtype).movedim(-1, axis)


def pack_sf(scales):
    e, rows, cols = scales.shape
    return scales.reshape(e, rows // 128, 4, 32, cols // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().flatten()


def payload(shape, precision):
    if precision == "nvfp4":
        raw = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), device="cuda", dtype=torch.uint8)
        return raw.view(torch.float4_e2m1fn_x2)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.125
    return value if precision == "bf16" else value.to(torch.float8_e4m3fn)


def problem(args, precision, direction, layout):
    torch.manual_seed(args.seed)
    groups = 256 // args.ep
    valid_m = math.ceil(args.context * args.microbatch * args.ep * 8 / 256)
    m = math.ceil(valid_m / 256) * 256
    h, f = 7168, 2048
    n, k = (2 * f, h) if direction == "forward" else (f, h)
    rows = groups * m
    tile_m = args.tile_m
    canonical_forward_tile = getattr(args, "canonical_forward_tile_m", None)
    if layout == "gate_then_up" and direction == "forward" and precision != "bf16" and canonical_forward_tile is not None:
        tile_m = canonical_forward_tile
    a, b = payload((rows, k), precision), payload((groups, n, k), precision)
    kwargs = dict(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=None,
        padded_offsets=torch.arange(1, groups + 1, device="cuda", dtype=torch.int32) * m,
        alpha_tensor=torch.full((groups,), 0.01 if precision == "nvfp4" else 1.0, device="cuda"),
        prob_tensor=torch.ones(rows, device="cuda"),
        glu_layout=layout,
        b_major="k",
        mma_tiler_mn=(tile_m, 256),
        cluster_shape_mn=(2, 1) if tile_m == 256 else (1, 1),
        use_dynamic_sched=True,
        d_dtype=torch.float8_e4m3fn if precision == "mxfp8" else torch.bfloat16,
    )
    if valid_m != m:
        for expert in range(groups):
            a[expert * m + valid_m : (expert + 1) * m].view(torch.uint8).zero_()
            kwargs["prob_tensor"][expert * m + valid_m : (expert + 1) * m].zero_()
    if precision != "bf16":
        sf_vec = 32 if precision == "mxfp8" else 16
        sf_dtype = torch.float8_e8m0fnu if precision == "mxfp8" else torch.float8_e4m3fn
        # Nonuniform scales expose accidental branch-scale mismatches.
        sa = torch.randint(-2, 1, (1, rows, k // sf_vec), device="cuda").float().exp2().to(sf_dtype)
        sb = torch.randint(-2, 1, (groups, n, k // sf_vec), device="cuda").float().exp2().to(sf_dtype)
        if direction == "forward" and layout == "interleaved_32":
            sb = paired(sb, 1)
        kwargs.update(sfa_tensor=pack_sf(sa), sfb_tensor=pack_sf(sb), sf_vec_size=sf_vec)
        if precision == "mxfp8":
            kwargs["norm_const_tensor"] = torch.ones(1, device="cuda")
    if direction == "forward":
        if layout == "interleaved_32":
            kwargs["b_tensor"] = paired(b, 1).contiguous()
        kwargs.update(c_dtype=torch.bfloat16, generate_c=True, act_func="swiglu")
    else:
        c = torch.randn(rows, 2 * f, device="cuda", dtype=torch.bfloat16) * 0.125
        kwargs.update(
            c_tensor=paired(c, 1).contiguous() if layout == "interleaved_32" else c,
            beta_tensor=torch.ones(groups, device="cuda"),
            dprob_tensor=torch.zeros(rows, device="cuda", dtype=torch.float32),
            act_func="dswiglu",
        )
    metadata = dict(groups=groups, M=m, valid_M=valid_m, N=n, K=k, total_rows=rows, mma_tiler_mn=(tile_m, 256), cluster_shape_mn=kwargs["cluster_shape_mn"])
    return kwargs, metadata


def prepare(kwargs, direction):
    wrapper = cudnn.grouped_gemm_glu_wrapper_sm100 if direction == "forward" else cudnn.grouped_gemm_dglu_wrapper_sm100
    cls = cudnn.GroupedGemmGluSm100 if direction == "forward" else cudnn.GroupedGemmDgluSm100
    # The wrapper derives discrete expert count/shape from pointers and N;
    # the reusable class needs those constructor descriptors explicitly.
    wrapper_args = {name: value for name, value in kwargs.items() if name not in ("num_experts", "b_shape")}
    if "glu_layout" not in inspect.signature(wrapper).parameters:
        if wrapper_args.pop("glu_layout") != "interleaved_32":
            raise RuntimeError("Imported frontend does not support gate_then_up")
    outputs = wrapper(**wrapper_args)
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
        raise RuntimeError(f"Unsupported requested configuration: {kwargs['glu_layout']}")
    api.compile()
    # Bind through the real backend's signature; no wrapper allocation in timing.
    implementation = api._implementation
    execute = implementation.execute
    arguments = {name: operands[name] for name in inspect.signature(execute).parameters if name in operands}
    for name, parameter in inspect.signature(execute).parameters.items():
        if name not in arguments and parameter.default is inspect.Parameter.empty:
            arguments[name] = None
    if "workspace" in inspect.signature(execute).parameters:
        arguments["workspace"] = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)

    def run():
        if kwargs.get("dprob_tensor") is not None:
            kwargs["dprob_tensor"].zero_()
        if outputs.get("amax_tensor") is not None:
            outputs["amax_tensor"].fill_(float("-inf"))
        execute(**arguments)

    run()
    torch.cuda.synchronize()
    return run, outputs, api


def snapshots(outputs, direction, layout):
    result = {}
    for name in ("c_tensor", "d_tensor", "d_row_tensor", "d_col_tensor"):
        if name == "d_col_tensor" and outputs.get("sfd_col_tensor") is None:
            continue  # The block-scaled BF16-output path does not write D_col.
        value = outputs.get(name)
        if value is None:
            continue
        value = value.detach().clone()
        if layout == "interleaved_32" and (name == "c_tensor" or direction == "backward"):
            value = paired(value, 1, inverse=True)
        result[name] = value
    d = outputs.get("d_tensor") if direction == "forward" else outputs.get("d_row_tensor")
    if d is not None:
        m, n = d.shape[:2]
        for name, rows, cols in (("sfd_row_tensor", m, n), ("sfd_col_tensor", n, m)):
            scale = outputs.get(name)
            if scale is None:
                continue
            scale = scale.reshape(1, rows // 128, cols // 128, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(rows, cols // 32)
            if direction == "backward" and layout == "interleaved_32":
                scale = paired(scale, 1 if name == "sfd_row_tensor" else 0, inverse=True, block=1 if name == "sfd_row_tensor" else 32)
            result[name] = scale.detach().clone()
    if outputs.get("dprob_tensor") is not None:
        result["dprob_tensor"] = outputs["dprob_tensor"].detach().clone()
    if outputs.get("amax_tensor") is not None:
        result["amax_tensor"] = outputs["amax_tensor"].detach().clone()
    return result


def check(actual, expected):
    for name, target in expected.items():
        value = actual[name]
        if not torch.isfinite(value.float()).all():
            raise AssertionError(f"Nonfinite {name}")
        torch.testing.assert_close(value.float(), target.float(), rtol=1e-5 if name == "dprob_tensor" else 0, atol=1e-6 if name == "dprob_tensor" else 0)


def output_hashes(values):
    """Cross-process legacy controls; copying/hashing is outside timing.

    dprob uses an existing atomic reduction and is checked with a tolerance
    within a process. Its bytes are not a deterministic regression contract.
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
    # Compile both layouts before warming or sampling either of them. Adjacent
    # pairs alternate AB/BA so canonical JIT and clock drift cannot separate
    # the two layout measurement windows.
    graphs, events = {}, {}
    samples = {layout: dict(graph_device_us=[], eager_wall_us=[]) for layout in runs}
    for layout, run in runs.items():
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.bundle):
                run()
        graphs[layout] = graph
        events[layout] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for _ in range(args.warmup):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    for repeat in range(args.repeats):
        order = tuple(runs) if repeat % 2 == 0 else tuple(reversed(runs))
        for layout in order:
            begin, end = events[layout]
            begin.record()
            graphs[layout].replay()
            end.record()
            end.synchronize()
            samples[layout]["graph_device_us"].append(begin.elapsed_time(end) * 1000 / args.bundle)
        for layout in order:
            start = time.perf_counter()
            for _ in range(args.bundle):
                runs[layout]()
            torch.cuda.synchronize()
            samples[layout]["eager_wall_us"].append((time.perf_counter() - start) * 1e6 / args.bundle)
    measurements = {}
    for layout, metrics in samples.items():
        measurements[layout] = {name: statistics.median(values) for name, values in metrics.items()}
        measurements[layout]["distributions"] = {name: distribution(values) for name, values in metrics.items()}
    if "interleaved_32" in runs and "gate_then_up" in runs:
        measurements["gate_then_up"]["paired_ratio_to_interleaved_32"] = {
            name: distribution([c / i for c, i in zip(samples["gate_then_up"][name], values)]) for name, values in samples["interleaved_32"].items()
        }
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("all", "bf16", "nvfp4", "mxfp8"), default="all")
    parser.add_argument("--direction", choices=("both", "forward", "backward"), default="both")
    parser.add_argument("--layout", choices=("compare", "interleaved_32", "gate_then_up"), default="compare")
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--tile-m", type=int, choices=(128, 256), default=256)
    parser.add_argument(
        "--canonical-forward-tile-m",
        type=int,
        choices=(128, 256),
        help="Override only canonical block-scaled forward tiles, for paired comparisons of TE's selected configurations",
    )
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--bundle", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.ep < 1 or 256 % args.ep or min(args.context, args.microbatch, args.warmup, args.bundle, args.repeats) < 1:
        parser.error("EP must divide 256; context, microbatch and timing counts must be positive")
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
    precisions = ("bf16", "nvfp4", "mxfp8") if args.precision == "all" else (args.precision,)
    directions = ("forward", "backward") if args.direction == "both" else (args.direction,)
    rows = []
    for precision in precisions:
        for direction in directions:
            expected = None
            runs, cases = {}, {}
            for layout in ("interleaved_32", "gate_then_up") if args.layout == "compare" else (args.layout,):
                kwargs, shape = problem(args, precision, direction, layout)
                run, outputs, api = prepare(kwargs, direction)
                values = snapshots(outputs, direction, layout)
                if expected is None:
                    expected = values
                else:
                    check(values, expected)
                runs[layout] = run
                cases[layout] = dict(precision=precision, direction=direction, layout=layout, **shape, output_sha256=output_hashes(values))
                del run, outputs, api, values, kwargs
            measurements = measure(runs, args)
            for layout, case in cases.items():
                row = {**case, **measurements[layout]}
                rows.append(row)
                print(json.dumps(row), flush=True)
            del expected, runs, cases
    report = dict(runtime=runtime, configuration=vars(args), measurements=rows)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
