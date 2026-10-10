# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native Sage FP8 KV128 with FP8 KV64 on exactly the same sparse mask.

Run from the repository root on an idle SM120 GPU. Results use paired,
alternating CUDA-graph replays, not profiler times. Metadata conversion is
baseline setup only and is never used by the native implementation.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from cudnn import BSA
from cudnn.block_sparse_attention import _interface
from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd
from benchmark_sm120_blk128 import _make_block_indices, _rounded_topk


def _capture(fn, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    return graph, output


def _paired_times(graphs, repeats):
    events = [[], []]
    for repeat in range(repeats):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[index].replay()
            end.record()
            events[index].append((start, end))
    torch.cuda.synchronize()
    return [[start.elapsed_time(end) for start, end in pairs] for pairs in events]


def _sample_error(q, k, v, outputs, indices):
    refs, actual = [], [[], []]
    for head, row in ((0, 0), (0, 127), (q.shape[1] - 1, q.shape[2] - 1), (q.shape[1] - 1, q.shape[2] // 2)):
        selected = indices[0, head, row // 128].long()
        tokens = (selected[:, None] * 128 + torch.arange(128, device=q.device)).flatten()
        scores = k[0, head, tokens].float() @ q[0, head, row].float() * 128**-0.5
        refs.append(torch.softmax(scores, dim=0) @ v[0, head, tokens].float())
        for target, output in zip(actual, outputs):
            target.append(output[0, head, row].float())
    reference = torch.stack(refs)
    errors = [(torch.stack(rows) - reference).norm().div(reference.norm().clamp_min(1e-8)).item() for rows in actual]
    if not all(error < 0.1 for error in errors):
        raise AssertionError(f"sampled FP8 relative L2 error exceeded 10%: {errors}")
    return errors


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=142720)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--densities", type=float, nargs="+", default=(0.15, 0.2))
    parser.add_argument("--patterns", choices=("strided", "local"), nargs="+", default=("strided", "local"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("this benchmark requires SM120")
    if args.sequence <= 0 or args.sequence % 128 or args.heads < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("sequence must be a positive multiple of 128; heads/repeats must be positive; warmup must be nonnegative")
    if any(not 0 < density <= 1 for density in args.densities):
        raise ValueError("density must be in (0, 1]")
    torch.manual_seed(120128)
    shape = (1, args.heads, args.sequence, 128)
    q, k, v = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    quantized = {
        64: _quantize_sage_bhsd(q, k, v),
        128: _quantize_sage_bhsd(q, k, v, v_block_size=128),
    }
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cutlass": str(_interface.cutlass.__version__),
        "shape": shape,
        "timing": "paired alternating CUDA-graph replay; medians; milliseconds",
        "cases": [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    for density in args.densities:
        topk = _rounded_topk(args.sequence // 128, density)
        for pattern in args.patterns:
            indices128 = _make_block_indices(args.heads, args.sequence // 128, topk, pattern, q.device)
            indices64 = torch.stack((2 * indices128, 2 * indices128 + 1), dim=-1).flatten(-2).repeat_interleave(2, dim=2)
            case = {"density": density, "effective_density": topk / (args.sequence // 128), "pattern": pattern, "topk128": topk}
            for mode in ("attention_only", "with_quantization"):
                graphs, outputs = [], []
                for block_size, indices in ((64, indices64), (128, indices128)):

                    def run(block_size=block_size, indices=indices):
                        if mode == "attention_only":
                            return _interface._bsa_attn_fwd_sm120_fp8(
                                *quantized[block_size],
                                indices,
                                indices.shape[-1],
                                128**-0.5,
                                sparse_block_size=block_size,
                                v_block_size=128 if block_size == 128 else 0,
                            )[0]
                        return BSA.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=block_size)["o_tensor"]

                    graph, output = _capture(run, args.warmup)
                    graphs.append(graph)
                    outputs.append(output)
                samples = _paired_times(graphs, args.repeats)
                medians = [statistics.median(values) for values in samples]
                case[mode] = {
                    "blk64_ms": medians[0],
                    "blk128_ms": medians[1],
                    "speedup": medians[0] / medians[1],
                    "sample_relative_l2": _sample_error(q, k, v, outputs, indices128),
                    "samples_ms": samples,
                }
                del graphs, outputs, graph, output
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
