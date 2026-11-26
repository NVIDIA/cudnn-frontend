# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired FE FC1 latency; not a TE full-MLP or training throughput benchmark."""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import cudnn

from gemm.cutedsl.test_grouped_gemm_glu_branch_layout import make_problem, pack_scales, canonicalize_c
from gemm.cutedsl.test_grouped_gemm_glu_hadamard_layout import prepare as prepare_amax
from gemm.cutedsl.test_grouped_gemm_glu_hadamard_quant_gate_then_up import kwargs_for as prepare_quant
from determinism_utils import assert_bitwise


def prepare(problem, case, layout):
    if case == "nvfp4_hamax":
        kwargs, owners = prepare_amax(problem, layout=layout, discrete=True)
        kwargs["bias_tensor"] = None
        return lambda: cudnn.grouped_gemm_glu_hadamard_wrapper_sm100(**kwargs), owners
    if case == "nvfp4_hquant":
        kwargs, owners = prepare_quant(problem, "col_fp4", discrete=True, layout=layout)
        kwargs.update(norm_const=1.0, rht_norm_const=1.0, bias_tensor=None)
        return lambda: cudnn.grouped_gemm_glu_hadamard_quant_wrapper_sm100(**kwargs), owners
    kwargs, b, sb, _, _ = problem
    kwargs = dict(kwargs)
    if layout == "interleaved_32":
        n = b.shape[1]
        order = torch.arange(n, device="cuda").view(2, n // 64, 32).permute(1, 0, 2).reshape(-1)
        b = b.view(torch.uint8).index_select(1, order).view(b.dtype)
        if sb is not None:
            sb = sb.view(torch.uint8).index_select(1, order).view(sb.dtype)
        kwargs["bias_tensor"] = kwargs["bias_tensor"].t().index_select(1, order).t()
    scales = pack_scales(sb)
    kwargs["bias_tensor"] = None
    kwargs.update(
        b_ptrs=torch.tensor([x.data_ptr() for x in b], device="cuda", dtype=torch.int64),
        sfb_ptrs=(
            None
            if scales is None
            else torch.tensor([scales.data_ptr() + i * scales.numel() // len(b) for i in range(len(b))], device="cuda", dtype=torch.int64)
        ),
        n=b.shape[1],
        b_dtype=b.dtype,
        glu_layout=layout,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        use_dynamic_sched=True,
    )
    return lambda: cudnn.grouped_gemm_glu_wrapper_sm100(**kwargs), (b, scales)


def benchmark(args, case):
    problem = make_problem(f=args.ffn, k=args.hidden, counts=(args.m,) * args.groups, precision=case.split("_")[0])
    layouts = ["interleaved_32", "gate_then_up"]
    prepared = {layout: prepare(problem, case, layout) for layout in layouts}
    snapshots, graphs = {}, {}
    for layout in layouts:
        invoke, owners = prepared[layout]
        out = invoke()
        torch.cuda.synchronize()
        snapshots[layout] = {key: value.clone() for key, value in out.items() if value is not None and key != "d_col_tensor"}
        if case in ("bf16", "nvfp4"):
            snapshots[layout] = {k: v for k, v in snapshots[layout].items() if not k.startswith("sfd_")}
    old_c = snapshots["interleaved_32"]["c_tensor"]
    snapshots["interleaved_32"]["c_tensor"] = canonicalize_c(old_c.squeeze(-1) if old_c.ndim == 3 else old_c).reshape(old_c.shape)
    assert_bitwise(snapshots["gate_then_up"], snapshots["interleaved_32"], case)
    del snapshots
    for layout in layouts:
        invoke, owners = prepared[layout]
        for _ in range(12):
            invoke()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.replays):
                invoke()
        graphs[layout] = graph
    # Settle clocks after compilation; the measured rounds remain paired.
    deadline = time.perf_counter() + args.warmup_seconds
    while time.perf_counter() < deadline:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    rows, rng = [], random.Random(401)
    for _ in range(args.rounds):
        order = layouts.copy()
        rng.shuffle(order)
        row = {}
        for layout in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[layout].replay()
            end.record()
            end.synchronize()
            row[layout] = start.elapsed_time(end) * 1000 / args.replays
        rows.append(row)
    medians = {layout: statistics.median(row[layout] for row in rows) for layout in layouts}
    delta = statistics.median(100 * (row["gate_then_up"] / row["interleaved_32"] - 1) for row in rows)
    return dict(
        case=case, median_us=medians, paired_delta_percent=delta, rounds=rows, tile=[256, 256], ctas=2, replays=args.replays, warmup_seconds=args.warmup_seconds
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["bf16", "mxfp8", "nvfp4", "nvfp4_hamax", "nvfp4_hquant"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (10, 7)
    result = dict(
        scope="FE FC1 only; layout preparation and compilation excluded",
        gpu=torch.cuda.get_device_name(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        frontend=cudnn.__file__,
        diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"])).hexdigest(),
        clocks="unlocked",
        shape=dict(groups=args.groups, m=args.m, hidden=args.hidden, ffn=args.ffn),
        result=benchmark(args, args.case),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result["result"].items() if k != "rounds"}), flush=True)


if __name__ == "__main__":
    main()
