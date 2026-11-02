#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark DeepSeek Sparse Attention (DSA) backward in cuDNN Frontend.

Builds a ``cudnn.DSA.SparseAttentionBackward`` plan and reusable output and
workspace buffers during setup. Timed iterations call ``plan.execute()`` and
dispatch to the Hopper (SM90) or Blackwell (SM100) CuTe DSL kernel based on
the active CUDA device. Inputs are flat MQA tensors (shared K=V buffer, global
per-query top-k indices). Unless ``--synthetic-forward`` is set, the forward
``out``/``lse`` consumed by the backward kernel come from a chunked PyTorch
reference; that option uses synthetic tensors instead. The forward launch is
intentionally excluded so the benchmark isolates backward.

Usage:
    python benchmark_dsa_sparse_attention_backward.py
    python benchmark_dsa_sparse_attention_backward.py --seqlens 4096,8192 --seqlens-kv 65536 --topks 2048 --deterministic both
    python benchmark_dsa_sparse_attention_backward.py --nheads 96 --head-dim 576 --synthetic-forward --csv results.csv
    python benchmark_dsa_sparse_attention_backward.py --seqlens 4096,8192,16384 --seqlens-kv 65536 --segment-lengths-kv 27726,17644,12603,7563 --sorted-topk --topks 2048 --nheads 96 --head-dim 576 --deterministic both --synthetic-forward
    python benchmark_dsa_sparse_attention_backward.py profile --seqlens 8192 --seqlens-kv 65536 --topks 2048 --deterministic on

``profile`` mode runs one warmed-up backward call wrapped in
``cudaProfilerStart/Stop`` and an NVTX range for nsys/ncu capture; it uses the
first value of ``--seqlens`` and the last value of ``--topks``.
"""

import argparse
import csv
import math
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


def proportional_query_segments(seqlen_q, segment_lengths_kv):
    """Assign query rows in proportion to packed KV segments, preserving the total."""
    total_s_kv = sum(segment_lengths_kv)
    segment_lengths_q = []
    cumulative_kv = 0
    previous_q_boundary = 0
    for segment_kv in segment_lengths_kv:
        cumulative_kv += segment_kv
        # Round each cumulative boundary with integer arithmetic so fractional
        # rows are distributed across segments instead of collected at the end.
        q_boundary = (seqlen_q * cumulative_kv + total_s_kv // 2) // total_s_kv
        segment_lengths_q.append(q_boundary - previous_q_boundary)
        previous_q_boundary = q_boundary
    return segment_lengths_q


def _make_segment_topk_indices(seqlen_q, seqlen_kv, topk, device):
    """Build unique, dispersed indices local to one packed sequence."""
    # A stride coprime with KV visits every key exactly once. Construct the
    # multiplied vectors in int64, reduce them, then keep the Q-by-TopK result
    # in int32 so production-sized cases do not need an int64 scratch matrix.
    stride = 2654435761 % seqlen_kv
    while math.gcd(stride, seqlen_kv) != 1:
        stride = (stride - 1) % seqlen_kv
    columns = ((torch.arange(topk, device=device, dtype=torch.int64) * stride) % seqlen_kv).to(torch.int32)
    offsets = ((torch.arange(seqlen_q, device=device, dtype=torch.int64) * 104729) % seqlen_kv).to(torch.int32)
    return (offsets[:, None] + columns[None, :]).remainder_(seqlen_kv)


def make_topk_indices(seqlen_q, seqlen_kv, topk, device, segment_lengths_kv=None, sorted_topk=False):
    """Build unique global indices, optionally constrained to packed sequences."""
    if segment_lengths_kv is None:
        segment_lengths_kv = [seqlen_kv]
    if not segment_lengths_kv or any(length <= 0 for length in segment_lengths_kv):
        raise ValueError("--segment-lengths-kv must contain positive lengths")
    if sum(segment_lengths_kv) != seqlen_kv:
        raise ValueError(f"--segment-lengths-kv must sum to seqlen_kv ({seqlen_kv}), got {sum(segment_lengths_kv)}")

    segment_lengths_q = proportional_query_segments(seqlen_q, segment_lengths_kv)
    pieces = []
    kv_begin = 0
    for segment_q, segment_kv in zip(segment_lengths_q, segment_lengths_kv):
        if segment_q and topk > segment_kv:
            raise ValueError(f"topk ({topk}) must not exceed each active KV segment; found segment length {segment_kv}")
        if segment_q:
            indices = _make_segment_topk_indices(segment_q, segment_kv, topk, device)
            if sorted_topk:
                indices = indices.sort(dim=-1).values
            pieces.append(indices.add_(kv_begin))
        kv_begin += segment_kv
    return torch.cat(pieces, dim=0)


def make_inputs(
    seqlen_q,
    topk,
    seqlen_kv,
    nheads,
    head_dim,
    head_dim_v,
    dtype,
    use_attn_sink,
    use_topk_length,
    segment_lengths_kv=None,
    sorted_topk=False,
):
    """Build flat-layout backward inputs with unique dispersed top-k rows."""
    device = "cuda"
    q = torch.randn(seqlen_q, nheads, head_dim, device=device, dtype=dtype) / 10
    kv = torch.randn(seqlen_kv, head_dim, device=device, dtype=dtype) / 10
    dout = torch.randn(seqlen_q, nheads, head_dim_v, device=device, dtype=dtype) / 10
    if use_attn_sink:
        attn_sink = torch.linspace(-2.0, 2.0, nheads, device=device, dtype=torch.float32)
    else:
        # exp(-inf) contributes nothing to the softmax denominator: sink disabled.
        attn_sink = torch.full((nheads,), float("-inf"), device=device, dtype=torch.float32)

    topk_idxs = make_topk_indices(
        seqlen_q,
        seqlen_kv,
        topk,
        device,
        segment_lengths_kv=segment_lengths_kv,
        sorted_topk=sorted_topk,
    )
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


def setup_case(seqlen_q, seqlen_kv, topk, deterministic, args):
    """Prepare one allocation-free backward plan and its reusable buffers."""
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
        segment_lengths_kv=args.segment_lengths_kv,
        sorted_topk=args.sorted_topk,
    )
    if args.synthetic_forward:
        # Scores are close to zero for the scaled inputs above, so log(topk)
        # is a representative finite LSE. This avoids a large reference-only
        # gather when profiling production sequence lengths.
        out = torch.zeros(seqlen_q, args.nheads, args.head_dim_v, dtype=dtype, device=q.device)
        lse = torch.full((seqlen_q, args.nheads), math.log(topk), dtype=torch.float32, device=q.device)
    else:
        out, lse = reference_forward(q, kv, attn_sink, topk_idxs, softmax_scale, args.head_dim_v)
    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)
    d_sink = torch.empty_like(attn_sink)

    plan = DSA.SparseAttentionBackward(
        sample_q=q,
        sample_kv=kv,
        sample_out=out,
        sample_dout=dout,
        sample_lse=lse,
        sample_attn_sink=attn_sink,
        sample_topk_idxs=topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        deterministic_relay_cluster_size=args.deterministic_relay_cluster_size if deterministic else None,
    )
    if not plan.check_support():
        raise RuntimeError(
            "Unsupported DSA backward benchmark case: "
            f"seqlen_q={seqlen_q}, seqlen_kv={seqlen_kv}, topk={topk}, "
            f"nheads={args.nheads}, head_dim={args.head_dim}, head_dim_v={args.head_dim_v}, "
            f"dtype={args.dtype}, deterministic={deterministic}, "
            f"deterministic_relay_cluster_size={args.deterministic_relay_cluster_size if deterministic else None}"
        )
    plan.compile()
    workspace_bytes = plan.scratch_workspace_bytes()
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device) if workspace_bytes else None

    def run():
        plan.execute(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            dq,
            dkv,
            topk_length=topk_length,
            workspace=workspace,
            d_sink=d_sink,
        )

    return run, workspace_bytes


def bench_one(seqlen_q, seqlen_kv, topk, deterministic, args):
    """Benchmark one config; returns (bwd_ms, bwd_tflops, workspace_bytes)."""
    run, workspace_bytes = setup_case(seqlen_q, seqlen_kv, topk, deterministic, args)

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
    return bwd_ms, bwd_tflops, workspace_bytes


def sequence_pairs(args):
    """Pair Q and KV extents, broadcasting one KV extent when requested."""
    if args.seqlens_kv is None:
        return [(seqlen, seqlen) for seqlen in args.seqlens]
    if len(args.seqlens_kv) == 1:
        return [(seqlen, args.seqlens_kv[0]) for seqlen in args.seqlens]
    if len(args.seqlens_kv) != len(args.seqlens):
        raise ValueError("--seqlens-kv must contain one value or the same number of values as --seqlens")
    return list(zip(args.seqlens, args.seqlens_kv))


def deterministic_modes(args):
    return {"off": [False], "on": [True], "both": [False, True]}[args.deterministic]


def run_benchmark_suite(args):
    device_name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()

    sep = "=" * 80
    print(sep)
    print(f"DSA Sparse Attention BWD Benchmark on {device_name} (SM{major}{minor})")
    print(
        f"nheads={args.nheads}, d_qk={args.head_dim}, d_v={args.head_dim_v}, dtype={args.dtype}, "
        f"attn_sink={not args.no_attn_sink}, topk_length={not args.no_topk_length}, "
        f"sorted_topk={args.sorted_topk}, segment_lengths_kv={args.segment_lengths_kv}, "
        f"deterministic_relay_cluster_size={args.deterministic_relay_cluster_size}, "
        f"warmup={args.warmup}, repeat={args.repeat}"
    )
    print(sep)
    print(f"{'seqlen_q':<10} {'seqlen_kv':<10} {'topk':<8} {'det':<5} {'BWD ms':<12} {'BWD TFLOPS':<12} {'workspace GiB':<14}")
    print("-" * 80)

    rows = []
    for seqlen_q, seqlen_kv in sequence_pairs(args):
        for topk in args.topks:
            if topk > seqlen_kv:
                continue
            for deterministic in deterministic_modes(args):
                try:
                    torch.cuda.empty_cache()
                    bwd_ms, bwd_tf, workspace_bytes = bench_one(seqlen_q, seqlen_kv, topk, deterministic, args)
                    workspace_gib = workspace_bytes / 2**30
                    print(f"{seqlen_q:<10} {seqlen_kv:<10} {topk:<8} {str(deterministic):<5} {bwd_ms:<12.3f} {bwd_tf:<12.2f} {workspace_gib:<14.3f}")
                    rows.append(
                        {
                            "seqlen_q": seqlen_q,
                            "seqlen_kv": seqlen_kv,
                            "topk": topk,
                            "nheads": args.nheads,
                            "head_dim": args.head_dim,
                            "head_dim_v": args.head_dim_v,
                            "deterministic": deterministic,
                            "deterministic_relay_cluster_size": args.deterministic_relay_cluster_size if deterministic else None,
                            "bwd_ms": round(bwd_ms, 4),
                            "bwd_tflops": round(bwd_tf, 2),
                            "workspace_gib": round(workspace_gib, 4),
                            "sorted_topk": args.sorted_topk,
                            "segment_lengths_q": (
                                ",".join(map(str, proportional_query_segments(seqlen_q, args.segment_lengths_kv))) if args.segment_lengths_kv else ""
                            ),
                            "segment_lengths_kv": ",".join(map(str, args.segment_lengths_kv)) if args.segment_lengths_kv else "",
                        }
                    )
                except Exception as e:
                    print(f"{seqlen_q:<10} {seqlen_kv:<10} {topk:<8} {str(deterministic):<5} ERROR: {e}")
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
    seqlen_q, seqlen_kv = sequence_pairs(args)[0]
    topk = args.topks[-1]
    cases = []
    for deterministic in deterministic_modes(args):
        run, workspace_bytes = setup_case(seqlen_q, seqlen_kv, topk, deterministic, args)
        run()
        cases.append((deterministic, run, workspace_bytes))
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for deterministic, run, _ in cases:
        with torch.cuda.nvtx.range(f"dsa_sparse_attention_bwd_profile[deterministic={deterministic}]"):
            run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    for deterministic, _, workspace_bytes in cases:
        print(
            f"Profile run done: seqlen_q={seqlen_q}, seqlen_kv={seqlen_kv}, topk={topk}, "
            f"nheads={args.nheads}, d_qk={args.head_dim}, d_v={args.head_dim_v}, dtype={args.dtype}, "
            f"deterministic={deterministic}, sorted_topk={args.sorted_topk}, "
            f"deterministic_relay_cluster_size={args.deterministic_relay_cluster_size if deterministic else None}, "
            f"segment_lengths_q={proportional_query_segments(seqlen_q, args.segment_lengths_kv) if args.segment_lengths_kv else None}, "
            f"segment_lengths_kv={args.segment_lengths_kv}, workspace={workspace_bytes / 2**30:.3f} GiB"
        )


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
        "--seqlens-kv",
        type=comma_separated_ints,
        default=None,
        help="one KV length broadcast to all Q lengths, or one per --seqlens value (default: match Q)",
    )
    p.add_argument(
        "--segment-lengths-kv",
        type=comma_separated_ints,
        default=None,
        help="packed KV sequence lengths; Q segments are assigned proportionally (default: one flat sequence)",
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
    p.add_argument(
        "--deterministic",
        choices=("off", "on", "both"),
        default="off",
        help="benchmark deterministic execution off, on, or both (default: off)",
    )
    p.add_argument(
        "--deterministic-relay-cluster-size",
        type=int,
        choices=(4, 8, 16, 32, 64),
        default=None,
        help="force the deterministic large-context relay to 4, 8, 16, 32, or 64 logical CTAs per group (default: automatic)",
    )
    p.add_argument(
        "--sorted-topk",
        action="store_true",
        help="sort each row of generated Top-K indices (default: dispersed order)",
    )
    p.add_argument(
        "--synthetic-forward",
        action="store_true",
        help="use finite synthetic out/LSE tensors instead of the reference forward (recommended for long sequences)",
    )
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
    if args.deterministic_relay_cluster_size is not None and args.deterministic == "off":
        raise ValueError("--deterministic-relay-cluster-size requires --deterministic on or both")
    args.head_dim_v = 512 if args.head_dim == 576 else args.head_dim  # matches the kernel-side derivation

    if args.segment_lengths_kv:
        segment_total = sum(args.segment_lengths_kv)
        incompatible = [seqlen_kv for _, seqlen_kv in sequence_pairs(args) if seqlen_kv != segment_total]
        if incompatible:
            raise ValueError(f"--segment-lengths-kv sums to {segment_total}, but every --seqlens-kv value must match; " f"found {sorted(set(incompatible))}")

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
        _ = DSA.SparseAttentionBackward  # resolve the lazy import up front
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
