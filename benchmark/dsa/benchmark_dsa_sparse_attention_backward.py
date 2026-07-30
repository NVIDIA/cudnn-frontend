#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark DeepSeek Sparse Attention (DSA) backward in cuDNN Frontend.

Times the public ``cudnn.DSA.sparse_attention_backward_wrapper`` API, which
dispatches to the Hopper (SM90) or Blackwell (SM100) CuTe DSL kernel based on
the active CUDA device. Inputs are flat FlashMLA-shaped tensors (shared K=V
buffer, global per-query top-k indices). The forward ``out``/``lse`` consumed
by the backward kernel are produced by a chunked PyTorch reference; the
production forward (FlashMLA) is out of scope here.

Usage:
    python benchmark_dsa_sparse_attention_backward.py
    python benchmark_dsa_sparse_attention_backward.py --seqlens 4096,8192 --topks 128,512,2048
    python benchmark_dsa_sparse_attention_backward.py --head-dim 576 --csv results.csv
    python benchmark_dsa_sparse_attention_backward.py profile --seqlens 8192 --topks 2048

``profile`` mode runs one warmed-up backward call wrapped in
``cudaProfilerStart/Stop`` and an NVTX range for nsys/ncu capture; it uses the
first value of ``--seqlens`` and the last value of ``--topks``.
"""

import argparse
import csv
import os

import torch

try:
    from cudnn import DSA
except ImportError as e:  # e.g. cudnn not installed, or binary incompatible with this node
    DSA, _CUDNN_IMPORT_ERROR = None, e
else:
    _CUDNN_IMPORT_ERROR = None

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def flops_bwd(seqlen_q, topk, nheads, head_dim, head_dim_v):
    """BWD = 5 matmuls: recompute S (d_qk), dV (d_v), dP (d_v), dQ (d_qk), dK (d_qk)."""
    return 2 * seqlen_q * nheads * topk * (3 * head_dim + 2 * head_dim_v)


def make_inputs(seqlen_q, topk, seqlen_kv, nheads, head_dim, head_dim_v, dtype, use_attn_sink, use_topk_length):
    """Build flat-layout backward inputs with unique random top-k indices per query."""
    device = "cuda"
    q = torch.randn(seqlen_q, nheads, head_dim, device=device, dtype=dtype) / 10
    kv = torch.randn(seqlen_kv, head_dim, device=device, dtype=dtype) / 10
    dout = torch.randn(seqlen_q, nheads, head_dim_v, device=device, dtype=dtype) / 10
    if use_attn_sink:
        attn_sink = torch.linspace(-2.0, 2.0, nheads, device=device, dtype=torch.float32)
    else:
        # exp(-inf) contributes nothing to the softmax denominator: sink disabled.
        attn_sink = torch.full((nheads,), float("-inf"), device=device, dtype=torch.float32)

    topk_idxs = torch.rand(seqlen_q, seqlen_kv, device=device).argsort(dim=-1)[:, :topk].to(torch.int32)
    topk_length = None
    if use_topk_length:
        topk_length = torch.full((seqlen_q,), topk, dtype=torch.int32, device=device)
    return q, kv, attn_sink, dout, topk_idxs, topk_length


@torch.no_grad()
def reference_forward(q, kv, attn_sink, topk_idxs, softmax_scale, head_dim_v, chunk=128):
    """Chunked sparse attention forward; returns (out, KV-only lse) for the backward kernel."""
    seqlen_q, nheads, _ = q.shape
    out = torch.empty(seqlen_q, nheads, head_dim_v, dtype=q.dtype, device=q.device)
    lse = torch.empty(seqlen_q, nheads, dtype=torch.float32, device=q.device)
    for i in range(0, seqlen_q, chunk):
        idx = topk_idxs[i : i + chunk].long()  # (C, K)
        kv_g = kv[idx].float()  # (C, K, D)
        scores = torch.einsum("chd,ckd->chk", q[i : i + chunk].float(), kv_g) * softmax_scale
        lse_c = torch.logsumexp(scores, dim=-1)  # (C, H), KV-only (no sink)
        lse_with_sink = torch.logaddexp(lse_c, attn_sink.view(1, nheads))
        p = torch.exp(scores - lse_with_sink.unsqueeze(-1))
        out[i : i + chunk] = torch.einsum("chk,ckv->chv", p, kv_g[..., :head_dim_v]).to(q.dtype)
        lse[i : i + chunk] = lse_c
    return out, lse


def setup_case(seqlen_q, seqlen_kv, topk, args):
    """Allocate inputs, run the reference forward, and return a zero-arg backward closure."""
    dtype = DTYPES[args.dtype]
    softmax_scale = args.head_dim**-0.5
    q, kv, attn_sink, dout, topk_idxs, topk_length = make_inputs(
        seqlen_q,
        topk,
        seqlen_kv,
        args.nheads,
        args.head_dim,
        args.head_dim_v,
        dtype,
        use_attn_sink=not args.no_attn_sink,
        use_topk_length=not args.no_topk_length,
    )
    out, lse = reference_forward(q, kv, attn_sink, topk_idxs, softmax_scale, args.head_dim_v)
    dq = torch.empty_like(q)
    dkv = torch.zeros_like(kv)

    def run():
        DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq,
            dkv=dkv,
        )

    return run


def bench_one(seqlen_q, seqlen_kv, topk, args):
    """Benchmark one config; returns (bwd_ms, bwd_tflops)."""
    run = setup_case(seqlen_q, seqlen_kv, topk, args)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeat):
        run()
    end.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(end) / args.repeat
    bwd_tflops = flops_bwd(seqlen_q, topk, args.nheads, args.head_dim, args.head_dim_v) / (bwd_ms * 1e-3) / 1e12
    return bwd_ms, bwd_tflops


def run_benchmark_suite(args):
    device_name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()

    sep = "=" * 80
    print(sep)
    print(f"DSA Sparse Attention BWD Benchmark on {device_name} (SM{major}{minor})")
    print(
        f"nheads={args.nheads}, d_qk={args.head_dim}, d_v={args.head_dim_v}, dtype={args.dtype}, "
        f"attn_sink={not args.no_attn_sink}, topk_length={not args.no_topk_length}, "
        f"warmup={args.warmup}, repeat={args.repeat}"
    )
    print(sep)
    print(f"{'seqlen_q':<10} {'seqlen_kv':<10} {'topk':<8} {'BWD ms':<12} {'BWD TFLOPS':<12}")
    print("-" * 80)

    rows = []
    for seqlen_q in args.seqlens:
        seqlen_kv = seqlen_q
        for topk in args.topks:
            if topk > seqlen_kv:
                continue
            try:
                torch.cuda.empty_cache()
                bwd_ms, bwd_tf = bench_one(seqlen_q, seqlen_kv, topk, args)
                print(f"{seqlen_q:<10} {seqlen_kv:<10} {topk:<8} {bwd_ms:<12.3f} {bwd_tf:<12.2f}")
                rows.append(
                    {
                        "seqlen_q": seqlen_q,
                        "seqlen_kv": seqlen_kv,
                        "topk": topk,
                        "nheads": args.nheads,
                        "head_dim": args.head_dim,
                        "head_dim_v": args.head_dim_v,
                        "bwd_ms": round(bwd_ms, 4),
                        "bwd_tflops": round(bwd_tf, 2),
                    }
                )
            except Exception as e:
                print(f"{seqlen_q:<10} {seqlen_kv:<10} {topk:<8} ERROR: {e}")
                torch.cuda.synchronize()
    print(sep)

    if args.csv and rows:
        csv_dir = os.path.dirname(args.csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to {args.csv}")


def run_profile(args):
    """Single warmed-up backward call under cudaProfilerStart/Stop for ncu/nsys."""
    seqlen_q = args.seqlens[0]
    topk = args.topks[-1]
    run = setup_case(seqlen_q, seqlen_q, topk, args)

    # warmup (also triggers compile) outside the profiled range
    run()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range("dsa_sparse_attention_bwd_profile"):
        run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"Profile run done: seqlen_q={seqlen_q}, topk={topk}, nheads={args.nheads}, d_qk={args.head_dim}, d_v={args.head_dim_v}, dtype={args.dtype}")


def comma_separated_ints(s):
    return [int(x) for x in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "mode",
        nargs="?",
        default="benchmark",
        choices=["benchmark", "profile"],
        help="benchmark: sweep seqlens x topks; profile: single run under cudaProfilerStart/Stop",
    )
    p.add_argument(
        "--seqlens",
        type=comma_separated_ints,
        default=[4096, 8192],
        help="comma-separated total query lengths; seqlen_kv = seqlen_q (default: 4096,8192)",
    )
    p.add_argument(
        "--topks",
        type=comma_separated_ints,
        default=[128, 512, 1024, 2048],
        help="comma-separated top-k values (default: 128,512,1024,2048)",
    )
    p.add_argument("--nheads", type=int, default=64, help="number of query heads (default: 64)")
    p.add_argument(
        "--head-dim",
        type=int,
        default=512,
        choices=[512, 576],
        help="QK head dim; 576 = 512 value dims + 64 RoPE dims. head_dim_v is derived (512)",
    )
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES), help="input dtype (default: bfloat16)")
    p.add_argument("--no-attn-sink", action="store_true", help="disable the attention sink (pass -inf sink logits)")
    p.add_argument(
        "--no-topk-length",
        action="store_true",
        help="omit the topk_length tensor (benchmarks the dense-top-k kernel variant)",
    )
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations per config (default: 10)")
    p.add_argument("--repeat", type=int, default=50, help="timed iterations per config (default: 50)")
    p.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    return p.parse_args()


def main():
    args = parse_args()
    args.head_dim_v = 512 if args.head_dim == 576 else args.head_dim  # matches the kernel-side derivation

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return
    major, minor = torch.cuda.get_device_capability()
    if major not in (9, 10):
        print(f"SKIP: DSA backward requires a Hopper (SM90) or Blackwell (SM100) GPU, found SM{major}{minor}")
        return
    if DSA is None:
        print(f"SKIP: cudnn is not importable ({_CUDNN_IMPORT_ERROR})")
        print("Install with: pip install nvidia-cudnn-frontend[cutedsl]")
        return
    try:
        _ = DSA.sparse_attention_backward_wrapper  # resolve the lazy import up front
    except ImportError as e:
        print(f"SKIP: cudnn[cutedsl] extras not available ({e})")
        print("Install with: pip install nvidia-cudnn-frontend[cutedsl]")
        return

    torch.manual_seed(0)
    if args.mode == "profile":
        run_profile(args)
    else:
        run_benchmark_suite(args)


if __name__ == "__main__":
    main()
