# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the installed native FP8 blk128 kernel with an archived revision.

Both revisions use identical quantized inputs, original blk128 metadata,
independent compile caches, and alternating CUDA-graph timing. Complete O
and centered-K LSE must be bitwise identical before performance is reported.
The public path includes each revision's Q/K/V quantization and V layout write.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

from cudnn import BSA
from cudnn.block_sparse_attention import _interface
from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd
from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128 import bsa_fwd_sm120_fp8 as native
from benchmark_sm120_blk128 import _make_block_indices, _rounded_topk
from benchmark_sm120_fp8_blk128 import _capture, _paired_times


def _load_baseline(path):
    spec = importlib.util.spec_from_file_location("bsa_fp8_revision_baseline", path)
    if spec is None or spec.loader is None:
        raise ValueError("baseline-kernel must point to a Python kernel source file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BlockSparseAttnForwardFp8Sm120Blk128


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-kernel", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=142720)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--densities", type=float, nargs="+", default=(0.15, 0.2))
    parser.add_argument("--patterns", choices=("strided", "local"), nargs="+", default=("strided", "local"))
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=120128)
    parser.add_argument("--profile", choices=("baseline", "candidate"), help="launch one attention kernel for profiler collection, without timing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("this benchmark requires SM120")
    if args.sequence <= 0 or args.sequence % 128 or args.heads < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("sequence must be a positive multiple of 128; heads/repeats must be positive; warmup must be nonnegative")
    if any(not 0 < density <= 1 for density in args.densities):
        raise ValueError("density must be in (0, 1]")

    baseline = _load_baseline(args.baseline_kernel)
    candidate = native.BlockSparseAttnForwardFp8Sm120Blk128
    original_cache = getattr(_interface._bsa_attn_fwd_sm120_fp8, "compile_cache", None)
    caches = ({}, {})
    torch.manual_seed(args.seed)
    shape = (1, args.heads, args.sequence, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    v_blocks = tuple(128 if getattr(kernel, "supports_blocked_v", False) else 0 for kernel in (baseline, candidate))
    quantized = tuple(_quantize_sage_bhsd(q, k, v, v_block_size=block) for block in v_blocks)
    for tensor_index, (a, b) in enumerate(zip(*quantized)):
        if tensor_index == 2:
            if v_blocks[0]:
                a = a.transpose(-1, -2).reshape(shape)
            if v_blocks[1]:
                b = b.transpose(-1, -2).reshape(shape)
        if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
            raise AssertionError(f"Quantized tensor {tensor_index} changed")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": str(torch.__version__),
        "cutlass": str(_interface.cutlass.__version__),
        "shape": shape,
        "seed": args.seed,
        "quantized_values_bitwise_equal": True,
        "v_block_sizes": v_blocks,
        "configured_baseline_compile_options": getattr(baseline, "_compile_options", "") if v_blocks[0] else "",
        "configured_candidate_compile_options": getattr(candidate, "_compile_options", "") if v_blocks[1] else "",
        "compile_min_blocks": tuple(getattr(kernel, "_compile_min_blocks", 0) for kernel in (baseline, candidate)),
        "baseline_sha256": hashlib.sha256(args.baseline_kernel.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "quantizer_sha256": {
            name: hashlib.sha256((Path(_interface.__file__).parent / name).read_bytes()).hexdigest()
            for name in ("_fp8_quant.py", "csrc/fwd/sm120_blk128/sage_v_quant.py")
        },
        "timing": "paired alternating CUDA-graph replay; medians; milliseconds",
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cases": [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    try:
        for density in args.densities:
            num_blocks = args.sequence // 128
            topk = _rounded_topk(num_blocks, density)
            compile_options = tuple(
                getattr(kernel, "_compile_options", "") if v_blocks[index] and topk >= getattr(kernel, "_compile_min_blocks", 0) else ""
                for index, kernel in enumerate((baseline, candidate))
            )
            for pattern in args.patterns:
                indices = _make_block_indices(args.heads, num_blocks, topk, pattern, q.device)
                for mode in ("attention_only", "with_quantization"):
                    outputs, graphs = [], []
                    for index, kernel in enumerate((baseline, candidate)):
                        if args.profile and index != (0 if args.profile == "baseline" else 1):
                            continue
                        native.BlockSparseAttnForwardFp8Sm120Blk128 = kernel
                        _interface._bsa_attn_fwd_sm120_fp8.compile_cache = caches[index]

                        def run():
                            if mode == "attention_only":
                                return _interface._bsa_attn_fwd_sm120_fp8(
                                    *quantized[index], indices, topk, 128**-0.5, sparse_block_size=128, v_block_size=v_blocks[index]
                                )
                            if index == 0:
                                baseline_quantized = _quantize_sage_bhsd(q, k, v, v_block_size=v_blocks[0])
                                out, _ = _interface._bsa_attn_fwd_sm120_fp8(
                                    *baseline_quantized, indices, topk, 128**-0.5, sparse_block_size=128, v_block_size=v_blocks[0]
                                )
                                return (out,)
                            return (BSA.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=128)["o_tensor"],)

                        if args.profile:
                            run()
                            torch.cuda.synchronize()
                            print("PROFILE_COMPLETE", flush=True)
                            return 0
                        graph, output = _capture(run, args.warmup)
                        graphs.append(graph)
                        outputs.append(output)
                    for graph in graphs:
                        graph.replay()
                    torch.cuda.synchronize()
                    exact = [torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) for a, b in zip(*outputs)]
                    if not all(exact):
                        raise AssertionError(f"Complete output/LSE differs from baseline: bitwise_equal={exact}")
                    samples = _paired_times(graphs, args.repeats)
                    medians = [statistics.median(values) for values in samples]
                    result = {
                        "density": density,
                        "effective_density": topk / num_blocks,
                        "topk128": topk,
                        "pattern": pattern,
                        "mode": mode,
                        "baseline_compile_options": compile_options[0],
                        "candidate_compile_options": compile_options[1],
                        "baseline_ms": medians[0],
                        "candidate_ms": medians[1],
                        "speedup": medians[0] / medians[1],
                        "bitwise_equal": exact,
                        "samples_ms": samples,
                    }
                    report["cases"].append(result)
                    print(json.dumps({key: value for key, value in result.items() if key != "samples_ms"}), flush=True)
                    del graphs, outputs, graph, output
    finally:
        native.BlockSparseAttnForwardFp8Sm120Blk128 = candidate
        if original_cache is None:
            delattr(_interface._bsa_attn_fwd_sm120_fp8, "compile_cache")
        else:
            _interface._bsa_attn_fwd_sm120_fp8.compile_cache = original_cache
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
