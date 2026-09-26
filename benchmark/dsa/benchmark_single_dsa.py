#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single DSA sparse-attention benchmark.

Runs one (pass, shape) case of the cuDNN Frontend DeepSeek Sparse Attention
kernels and prints a parseable ``RESULT,`` line. Called as a subprocess by
runner.py so each case gets a clean CUDA context (CuTe DSL compile state
included) and failures stay independent.

Passes:
  fwd   ``cudnn.DSA.SparseAttentionForward.execute`` with preallocated
        outputs (SM100-family only).
  bwd   ``cudnn.DSA.SparseAttentionBackward.execute`` with preallocated
        gradients and workspace (SM90 / SM100). The ``out``/``lse`` it
        consumes come from a chunked PyTorch reference, so no forward launch
        sits in the timed region.

Inputs are flat MQA tensors: ``q (S_q, H, d_qk)``, one shared ``kv (S_kv,
d_qk)`` record read as both K and V, and ``topk_idxs (S_q, topk)`` holding
unique random rows of the KV pool per query. Timing is CUDA events around
each steady-state call after a 256 MiB L2 flush, median of
``--num_iterations``; kernel compilation happens during warmup.

Exit code 3 (``UNSUPPORTED,<reason>`` on stdout) means the kernel's own
support check rejected the case — the runner records it as skipped.
"""

import argparse
import math
import sys
from typing import Optional

import torch

UNSUPPORTED_EXIT_CODE = 3

# Dense-MMA FLOPs per clock per SM by compute-capability major, for the % SOL
# denominator (same table as attention_training/benchmark_single_sdpa.py).
_FLOPS_PER_CLOCK_PER_SM = {
    9: {"bfloat16": 4096, "float16": 4096},
    10: {"bfloat16": 8192, "float16": 8192},
    12: {"bfloat16": 1024, "float16": 1024},
}

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def flops_fwd(s_q, num_heads, topk, d_qk, d_v):
    """QK^T over the gathered rows and PV: 2 matmuls."""
    return 2.0 * s_q * num_heads * topk * (d_qk + d_v)


def flops_bwd(s_q, num_heads, topk, d_qk, d_v):
    """5 matmuls: recompute S (d_qk), dV (d_v), dP (d_v), dQ (d_qk), dK (d_qk)."""
    return 2.0 * s_q * num_heads * topk * (3 * d_qk + 2 * d_v)


def sample_sm_clock_mhz() -> Optional[float]:
    """Current SM clock of the torch device via NVML, or None without pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
            handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
            return float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def peak_mma_tflops(data_type: str, sm_mhz: Optional[float]) -> Optional[float]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    per_clk = _FLOPS_PER_CLOCK_PER_SM.get(props.major, {}).get(data_type)
    if per_clk is None or sm_mhz is None:
        return None
    return per_clk * props.multi_processor_count * sm_mhz / 1e6


def make_topk_idxs(s_q, s_kv, topk, device, chunk=1024):
    """Unique random KV rows per query, generated in row chunks to bound memory."""
    idxs = torch.empty(s_q, topk, dtype=torch.int32, device=device)
    for i in range(0, s_q, chunk):
        n = min(chunk, s_q - i)
        idxs[i : i + n] = torch.rand(n, s_kv, device=device).argsort(dim=-1)[:, :topk].to(torch.int32)
    return idxs


@torch.no_grad()
def reference_forward(q, kv, attn_sink, topk_idxs, softmax_scale, d_v, chunk=128):
    """Chunked sparse attention forward -> (out, KV-only lse) for the backward kernel."""
    s_q, num_heads, _ = q.shape
    out = torch.empty(s_q, num_heads, d_v, dtype=q.dtype, device=q.device)
    lse = torch.empty(s_q, num_heads, dtype=torch.float32, device=q.device)
    for i in range(0, s_q, chunk):
        idx = topk_idxs[i : i + chunk].long()
        kv_g = kv[idx].float()  # (C, K, D)
        scores = torch.einsum("chd,ckd->chk", q[i : i + chunk].float(), kv_g) * softmax_scale
        lse_c = torch.logsumexp(scores, dim=-1)  # KV-only, excludes the sink
        lse_with_sink = torch.logaddexp(lse_c, attn_sink.view(1, num_heads))
        p = torch.exp(scores - lse_with_sink.unsqueeze(-1))
        out[i : i + chunk] = torch.einsum("chk,ckv->chv", p, kv_g[..., :d_v]).to(q.dtype)
        lse[i : i + chunk] = lse_c
    return out, lse


def time_fn(fn, num_warmup, num_iters) -> float:
    """Median per-call GPU time of fn() in ms via CUDA events, L2 flushed before each call."""
    l2_flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        l2_flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def setup_fwd(args, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale):
    from cudnn import DSA

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

    s_q, num_heads, _ = q.shape
    out = torch.empty((s_q, num_heads, args.head_dim_vo), dtype=q.dtype, device=q.device)
    max_logits = torch.empty((s_q, num_heads), dtype=torch.float32, device=q.device)
    lse = torch.empty_like(max_logits)
    lse_indexer = torch.empty_like(max_logits) if args.indexer_topk else None

    def run():
        op.execute(
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

    padded_topk = ((args.topk + 63) // 64) * 64
    detail = f"SparseAttentionForward.execute padded_K={padded_topk} indexer_topk={args.indexer_topk}"
    return run, detail


def setup_bwd(args, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale):
    from cudnn import DSA

    s_q, num_heads, _ = q.shape
    if attn_sink is None:
        # exp(-inf) adds nothing to the denominator: sink disabled.
        attn_sink = torch.full((num_heads,), float("-inf"), dtype=torch.float32, device=q.device)
    dout = torch.randn(s_q, num_heads, args.head_dim_vo, dtype=q.dtype, device=q.device).mul_(0.1)
    out, lse = reference_forward(q, kv, attn_sink, topk_idxs, softmax_scale, args.head_dim_vo)
    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)  # every route's finalizer overwrites all of dkv
    d_sink = torch.empty_like(attn_sink)

    op = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        sample_dq=dq,
        sample_dkv=dkv,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
        deterministic=args.deterministic,
    )
    op.check_support()
    op.compile()
    workspace_bytes = op.scratch_workspace_bytes()
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device) if workspace_bytes else None

    def run():
        op.execute(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            dq=dq,
            dkv=dkv,
            d_sink=d_sink,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            workspace=workspace,
        )

    detail = f"SparseAttentionBackward.execute workspace_bytes={workspace_bytes} deterministic={args.deterministic}"
    return run, detail


def unsupported(reason: str):
    print(f"UNSUPPORTED,{reason.replace(',', ';')}")
    sys.exit(UNSUPPORTED_EXIT_CODE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile_pass", required=True, choices=["fwd", "bwd"])
    p.add_argument("--backend", default="cudnn", choices=["cudnn"])
    p.add_argument("--q_seqlen", type=int, required=True)
    p.add_argument("--kv_seqlen", type=int, required=True)
    p.add_argument("--num_q_heads", type=int, required=True)
    p.add_argument("--head_dim_qk", type=int, default=512)
    p.add_argument("--head_dim_vo", type=int, default=512)
    p.add_argument("--topk", type=int, required=True, help="logical K: KV rows gathered per query")
    p.add_argument("--indexer_topk", type=int, default=0, help="indexer-selected prefix of topk with its own LSE (fwd only)")
    p.add_argument("--data_type", default="bfloat16", choices=sorted(DTYPES))
    p.add_argument("--has_sink", action="store_true")
    p.add_argument("--use_topk_length", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="deterministic backward (SM100)")
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--num_warmup_iterations", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.topk > args.kv_seqlen:
        unsupported(f"topk ({args.topk}) exceeds kv_seqlen ({args.kv_seqlen})")
    if not torch.cuda.is_available():
        unsupported("CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    if args.profile_pass == "fwd" and major != 10:
        unsupported(f"DSA sparse forward requires an SM100-family GPU, found SM{major}{minor}")
    if args.profile_pass == "bwd" and major not in (9, 10):
        unsupported(f"DSA sparse backward requires SM90 or SM100, found SM{major}{minor}")
    try:
        import cudnn
        from cudnn import DSA

        _ = DSA.SparseAttentionForward if args.profile_pass == "fwd" else DSA.SparseAttentionBackward
    except (ImportError, AttributeError) as e:
        unsupported(f"cudnn[cutedsl] not available: {e}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = DTYPES[args.data_type]
    softmax_scale = 1.0 / math.sqrt(args.head_dim_qk)

    q = torch.randn(args.q_seqlen, args.num_q_heads, args.head_dim_qk, dtype=dtype, device=device).mul_(0.1)
    kv = torch.randn(args.kv_seqlen, args.head_dim_qk, dtype=dtype, device=device).mul_(0.1)
    topk_idxs = make_topk_idxs(args.q_seqlen, args.kv_seqlen, args.topk, device)
    attn_sink = torch.linspace(-2.0, 2.0, args.num_q_heads, dtype=torch.float32, device=device) if args.has_sink else None
    topk_length = torch.full((args.q_seqlen,), args.topk, dtype=torch.int32, device=device) if args.use_topk_length else None

    setup = setup_fwd if args.profile_pass == "fwd" else setup_bwd
    try:
        fn, detail = setup(args, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale)
    except (ValueError, RuntimeError) as e:
        # APIBase.check_support raises these for shape/arch contracts the
        # kernels do not serve; anything else is a real failure and propagates.
        unsupported(str(e))

    ms = time_fn(fn, args.num_warmup_iterations, args.num_iterations)
    sm_mhz = sample_sm_clock_mhz()

    flops = (flops_fwd if args.profile_pass == "fwd" else flops_bwd)(args.q_seqlen, args.num_q_heads, args.topk, args.head_dim_qk, args.head_dim_vo)
    tflops = flops / (ms * 1e-3) / 1e12
    peak = peak_mma_tflops(args.data_type, sm_mhz)
    peak_str = f"{peak:.3f}" if peak else ""
    gpu = torch.cuda.get_device_name().replace(",", ";")
    print(f"RESULT,{ms:.6f},{tflops:.3f},{peak_str},{gpu},{cudnn.__version__},{detail.replace(',', ';')}")


if __name__ == "__main__":
    main()
