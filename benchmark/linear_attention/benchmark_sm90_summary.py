# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Graph timings for the public SM90 KDA summary graph plan.

Optionally compare a frozen pre-staging FP32 kernel supplied with
``--baseline-source`` (the 128-thread PORTS ABI from a136913d). Each case
checks output bytes against that baseline before timing. No training or
communication performance is inferred from these summary-only measurements.
``--profile-one`` emits one baseline and one candidate launch, without event
timing, for Nsight Compute. Imports/compilation/allocations precede capture.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

import cudnn
from cudnn.linear_attention.cake import compiler
from cudnn.linear_attention.ops.common import select_plan


def make_case(lengths, heads, gate):
    rng = torch.Generator().manual_seed(713)
    total = sum(lengths)
    q, k = [torch.nn.functional.normalize(torch.randn(total, heads, 128, generator=rng), dim=-1).bfloat16().cuda() for _ in range(2)]
    v, do = [torch.randn(total, heads, 128, generator=rng).bfloat16().cuda() for _ in range(2)]
    return dict(
        q=q,
        k=k,
        v=v,
        dO=do,
        g=torch.full((total, heads, 128), gate, device="cuda"),
        beta=torch.full((total, heads), 0.5, device="cuda"),
        cu_seqlens=torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"),
    )


def candidate(x, backward, handle):
    graph = cudnn.pygraph()
    names = ("q", "k", "g", "beta", "dO", "cu_seqlens") if backward else ("k", "v", "g", "beta", "cu_seqlens")
    tensors = {name: graph.tensor_like(x[name]) for name in names}
    h, m = graph.kda_summary_bwd(**tensors, output_transition=True) if backward else graph.kda_summary(**tensors, output_transition=True)
    h.set_data_type(cudnn.data_type.FLOAT).set_output(True)
    m.set_data_type(cudnn.data_type.FLOAT).set_output(True)
    select_plan(graph, "kda_summary_hopper_cuda")
    graph.build_plans()
    shape = (x["cu_seqlens"].numel() - 1, x["k"].shape[1], 128, 128)
    outputs = tuple(torch.empty(shape, device="cuda", dtype=torch.float32) for _ in range(2))
    bindings = {tensor: x[name] for name, tensor in tensors.items()}
    bindings.update({h: outputs[0], m: outputs[1]})
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run():
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph.execute(bindings, workspace, handle=handle)

    return run, outputs, workspace.numel()


def baseline(x, backward, library):
    n, heads = x["cu_seqlens"].numel() - 1, x["k"].shape[1]
    outputs = tuple(torch.empty((n, heads, 128, 128), device="cuda", dtype=torch.float32) for _ in range(2))
    params = compiler.Params()
    for t in (x["q"], x["k"], x["dO"] if backward else x["v"], x["g"], x["beta"], x["cu_seqlens"], None, *outputs):
        params.ptr(0 if t is None else t.data_ptr())
        for stride in (0,) * 4 if t is None else (0,) * (4 - t.ndim) + t.stride():
            params.i64(stride)
    params.i32(heads).i32(0).f32(128**-0.5)
    function = library.function("kda_summary_bwd" if backward else "kda_summary_fwd")

    def run():
        compiler.launch(function, (n * heads * 64, 1, 1), (128, 1, 1), 0, torch.cuda.current_stream().cuda_stream, params, "baseline")

    return run, outputs


def measure(run, repeats, groups):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run()
        samples = []
        for _ in range(groups):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / repeats)
        return statistics.median(samples)
    finally:
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gate", type=float, default=-0.001)
    parser.add_argument("--thd", action="store_true", help="Also measure 32 short documents and a 20-short/2-long pack")
    parser.add_argument("--profile-one", action="store_true")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if args.heads <= 0 or min(args.tokens) <= 0 or min(args.repeats, args.groups) <= 0:
        parser.error("heads, tokens, repeats and groups must be positive")
    library = compiler.KernelLibrary(args.baseline_source, "sm_90", torch.cuda.current_device(), fast_math=False) if args.baseline_source else None
    print(
        json.dumps(
            dict(
                cudnn=cudnn.__file__,
                torch=torch.__version__,
                baseline_sha256=hashlib.sha256(args.baseline_source.read_bytes()).hexdigest() if library else None,
            )
        ),
        flush=True,
    )
    layouts = [(t,) for t in args.tokens]
    if args.thd:
        layouts += [(33,) * 32, (17,) * 20 + (1024, 4096)]
    handle = cudnn.create_handle()
    try:
        for lengths in layouts:
            x = make_case(lengths, args.heads, args.gate)
            for backward in (False, True):
                run, actual, workspace = candidate(x, backward, handle)
                old = baseline(x, backward, library) if library else None
                if old:
                    old[0]()
                run()
                torch.cuda.synchronize()
                if old:
                    for got, ref in zip(actual, old[1]):
                        if not torch.equal(got.view(torch.uint8), ref.view(torch.uint8)):
                            raise AssertionError("staging changed the direct FP32 result")
                if args.profile_one:
                    return
                row = dict(lengths=lengths, heads=args.heads, gate=args.gate, backward=backward, workspace_bytes=workspace)
                row["summary_us"] = measure(run, args.repeats, args.groups)
                if old:
                    row.update(baseline_us=measure(old[0], args.repeats, args.groups), bitwise_baseline=True)
                print(json.dumps(row), flush=True)
    finally:
        cudnn.destroy_handle(handle)


if __name__ == "__main__":
    main()
