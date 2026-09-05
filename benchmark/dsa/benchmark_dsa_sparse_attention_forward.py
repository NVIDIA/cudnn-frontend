#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the public SM100 DSA sparse-attention forward API.

The benchmark reports two cuDNN Frontend GPU paths: an already-compiled
``SparseAttentionForward.execute`` call with preallocated outputs, and the
public wrapper. CUDA-event timing does not separately capture Python/host-side
allocation overhead.
"""

import argparse
import math

import torch

try:
    from cudnn import DSA
except ImportError as exc:
    DSA, _CUDNN_IMPORT_ERROR = None, exc
else:
    _CUDNN_IMPORT_ERROR = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seqlen-q", type=int, default=4096, help="number of query rows (default: 4096)")
    parser.add_argument("--seqlen-kv", type=int, default=5120, help="number of KV rows (default: 5120)")
    parser.add_argument("--heads", "--nheads", dest="heads", type=int, default=64, choices=[64, 128], help="query heads (default: 64)")
    parser.add_argument("--head-dim", type=int, default=512, choices=[512, 576], help="QK head dimension (default: 512)")
    parser.add_argument("--topk", "--k", dest="topk", type=int, default=640, help="logical sparse K (default: 640)")
    parser.add_argument(
        "--indexer-topk",
        type=int,
        default=512,
        choices=[0, 512, 1024, 2048],
        help="prefix length for indexer LSE (default: 512)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations per path (default: 10)")
    parser.add_argument("--repeat", type=int, default=50, help="timed iterations per path (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="input RNG seed (default: 0)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--no-attn-sink", action="store_true", help="omit the attention sink")
    parser.add_argument("--use-topk-length", action="store_true", help="pass a full-length topk_length tensor")
    return parser.parse_args()


def validate_args(args):
    if args.seqlen_q <= 0 or args.seqlen_kv <= 0 or args.topk <= 0:
        raise ValueError("seqlen-q, seqlen-kv, and topk must all be positive")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be nonnegative and repeat must be positive")
    if args.indexer_topk > args.topk:
        raise ValueError(f"indexer_topk ({args.indexer_topk}) must not exceed topk ({args.topk})")
    if args.heads == 128:
        if args.head_dim != 512:
            raise ValueError("H128 supports only head_dim=512")
        if args.indexer_topk == 2048:
            raise ValueError("H128 supports indexer_topk only in {0, 512, 1024}")


def make_inputs(args, device):
    q = torch.randn((args.seqlen_q, args.heads, args.head_dim), dtype=torch.bfloat16, device=device).mul_(0.1)
    kv = torch.randn((args.seqlen_kv, args.head_dim), dtype=torch.bfloat16, device=device).mul_(0.1)
    topk_idxs = torch.randint(0, args.seqlen_kv, (args.seqlen_q, args.topk), dtype=torch.int32, device=device)
    attn_sink = None
    if not args.no_attn_sink:
        attn_sink = torch.linspace(-2.0, 2.0, args.heads, dtype=torch.float32, device=device)
    topk_length = None
    if args.use_topk_length:
        topk_length = torch.full((args.seqlen_q,), args.topk, dtype=torch.int32, device=device)
    return q, kv, topk_idxs, attn_sink, topk_length


def time_cuda(run, warmup, repeat):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def make_dsa_paths(args, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale):
    op = DSA.SparseAttentionForward(
        q,
        kv,
        topk_idxs,
        sample_attn_sink=attn_sink,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
        indexer_topk=args.indexer_topk,
    )
    op.check_support()
    op.compile()

    out = torch.empty((args.seqlen_q, args.heads, 512), dtype=torch.bfloat16, device=q.device)
    max_logits = torch.empty((args.seqlen_q, args.heads), dtype=torch.float32, device=q.device)
    lse = torch.empty_like(max_logits)
    lse_indexer = torch.empty_like(max_logits) if args.indexer_topk else None

    def run_execute():
        return op.execute(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            out=out,
            max_logits=max_logits,
            lse=lse,
            lse_indexer=lse_indexer,
        )

    def run_wrapper():
        return DSA.sparse_attention_forward_wrapper(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            indexer_topk=args.indexer_topk,
        )

    # ``compile()`` is the API lifecycle gate; this first execute performs the
    # concrete CuTe JIT before either measured path starts.
    run_execute()
    torch.cuda.synchronize()
    return run_execute, run_wrapper


# Keep ordinary versioned tensors: the public API safely caches successful
# value validation by tensor identity/version, while inference tensors have no
# mutation counter and must be revalidated on every call.
@torch.no_grad()
def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return
    if DSA is None:
        print(f"SKIP: cudnn is not importable ({_CUDNN_IMPORT_ERROR})")
        print("Install with: pip install nvidia-cudnn-frontend[cutedsl]")
        return

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        print(f"SKIP: DSA sparse forward requires an SM100-family GPU, found SM{major}{minor}")
        return
    try:
        _ = DSA.sparse_attention_forward_wrapper
    except ImportError as exc:
        print(f"SKIP: cudnn[cutedsl] extras not available ({exc})")
        return

    torch.manual_seed(args.seed)
    q, kv, topk_idxs, attn_sink, topk_length = make_inputs(args, device)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)
    padded_topk = ((args.topk + 63) // 64) * 64
    print(f"DSA Sparse Attention Forward on {torch.cuda.get_device_name(device)} (SM{major}{minor})")
    print(
        f"S_q={args.seqlen_q}, S_kv={args.seqlen_kv}, H={args.heads}, D_qk={args.head_dim}, D_v=512, "
        f"K={args.topk}, padded_K={padded_topk}, indexer_topk={args.indexer_topk}, BF16, "
        f"sink={attn_sink is not None}, topk_length={topk_length is not None}, seed={args.seed}, "
        f"warmup={args.warmup}, repeat={args.repeat}"
    )
    print("Compiling and warming up...")
    run_execute, run_wrapper = make_dsa_paths(args, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale)

    execute_ms = time_cuda(run_execute, args.warmup, args.repeat)
    wrapper_ms = time_cuda(run_wrapper, args.warmup, args.repeat)
    print(f"{'DSA execute path (preallocated)':<42} {execute_ms:>10.3f} ms")
    print(f"{'DSA public wrapper':<42} {wrapper_ms:>10.3f} ms")


if __name__ == "__main__":
    main()
