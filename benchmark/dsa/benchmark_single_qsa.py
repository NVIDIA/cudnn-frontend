#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single QSA block-sparse attention benchmark.

Runs one (pass, shape) case of the cuDNN Frontend Query Sparse Attention
kernels (``cudnn.ops.qsa_forward`` / ``qsa_backward``: packed THD, BF16,
d=256, four-token selection blocks) and prints the same ``RESULT,`` line as
``benchmark_single_dsa.py``, so ``runner.py`` drives both workers alike.

The case is one packed sequence of ``--q_seqlen`` tokens (``--kv_seqlen`` must
match: QSA is self-attention over the packed tokens). Each query selects up to
``--topk`` complete causal blocks at random plus its own partial tail block —
the indexer's block budget — so queries early in the sequence gather fewer
blocks than the budget, as in the model. The bitmap, token bounds and block
spans follow the bitmap API's contract (docs/operations/sparse_attention/qsa.md).

Passes:
  fwd   ``qsa_forward``: the public call — output allocation, selection
        compaction and the attention kernel(s). QSA has no preallocated
        execute path, so the timed region is what a training step pays.
  bwd   ``qsa_backward``, likewise; the ``out_fp32``/``lse`` it consumes come
        from one untimed forward call.

TFLOPS count the tokens actually gathered (selected block tokens, clipped to
each query's causal bound, summed over queries) with the DSA suite's per-row
matmul formulas, so the number reads against the same dense-MMA peak.

Exit code 3 (``UNSUPPORTED,<reason>`` on stdout) means the QSA contract
declined the case — a head geometry or architecture the kernels do not serve
— and the runner records it as skipped.
"""

import argparse
import os
import sys

import torch

# Shared with the DSA worker: timing loop, SOL denominator, FLOP formulas and the UNSUPPORTED protocol.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_single_dsa import DTYPES, flops_bwd, flops_fwd, peak_mma_tflops, sample_sm_clock_mhz, time_fn, unsupported  # noqa: E402


def make_selection(total, block_size, budget, device, generator, chunk=2048):
    """Per-query bitmap of up to ``budget`` random complete causal blocks plus the tail block.

    Returns ``(sel_blocks, block_first, block_last, gathered_tokens)``; the bitmap is built in
    query-row chunks so the random scores never exceed ``chunk x blocks`` floats.
    """
    blocks = (total + block_size - 1) // block_size
    first = torch.arange(blocks, device=device, dtype=torch.int32) * block_size
    last = torch.clamp(first + block_size - 1, max=total - 1)
    sel = torch.zeros((total, blocks), dtype=torch.uint8, device=device)
    gathered = 0
    for i in range(0, total, chunk):
        t = torch.arange(i, min(i + chunk, total), device=device, dtype=torch.int32)
        scores = torch.rand((t.numel(), blocks), generator=generator, device=device)
        complete = (first + (block_size - 1))[None, :] <= t[:, None]  # whole block at or before the query
        scores.masked_fill_(~complete, -1.0)
        top = scores.topk(min(budget, blocks), dim=1)
        pick = torch.zeros_like(complete).scatter_(1, top.indices, top.values >= 0)
        pick[torch.arange(t.numel(), device=device), t // block_size] = True  # the causal tail block
        sel[i : i + t.numel()] = pick
        span = torch.minimum(last[None, :], t[:, None]) - first[None, :] + 1
        gathered += int((span * pick).sum())
    return sel, first, last, gathered


def unsupported_if_declined(call):
    """Run ``call`` once; map the QSA contract's refusals to the runner's skipped row."""
    try:
        return call()
    except (NotImplementedError, ValueError, TypeError, ImportError) as e:
        # check_inputs / _implementation raise these for geometries, dtypes and
        # architectures the kernels do not serve; anything else is a real failure.
        unsupported(str(e))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile_pass", required=True, choices=["fwd", "bwd"])
    p.add_argument("--backend", default="cudnn", choices=["cudnn"])
    p.add_argument("--q_seqlen", type=int, required=True, help="tokens in the packed sequence")
    p.add_argument("--kv_seqlen", type=int, required=True, help="must equal --q_seqlen (self-attention over the packed tokens)")
    p.add_argument("--num_q_heads", type=int, required=True)
    p.add_argument("--num_kv_heads", type=int, required=True)
    p.add_argument("--head_dim_qk", type=int, default=256)
    p.add_argument("--head_dim_vo", type=int, default=256)
    p.add_argument("--topk", type=int, required=True, help="block budget: complete causal blocks selected per query (the tail block comes on top)")
    p.add_argument("--block_size", type=int, default=4)
    p.add_argument("--data_type", default="bfloat16", choices=sorted(DTYPES))
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--num_warmup_iterations", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.kv_seqlen != args.q_seqlen:
        unsupported(f"QSA is self-attention over the packed tokens: kv_seqlen ({args.kv_seqlen}) must equal q_seqlen ({args.q_seqlen})")
    if args.head_dim_qk != 256 or args.head_dim_vo != 256:
        unsupported(f"QSA serves d=256, got {args.head_dim_qk}/{args.head_dim_vo}")
    if args.data_type != "bfloat16":
        unsupported(f"QSA training is BF16, got {args.data_type}")
    if not torch.cuda.is_available():
        unsupported("CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        unsupported(f"QSA requires SM100, SM103 or SM107, found SM{major}{minor}")
    try:
        import cudnn
        from cudnn.ops import qsa_backward, qsa_forward
    except (ImportError, AttributeError) as e:
        unsupported(f"cudnn QSA ops not available: {e}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = DTYPES[args.data_type]
    total, d = args.q_seqlen, args.head_dim_qk

    sel_blocks, block_first, block_last, gathered = make_selection(total, args.block_size, args.topk, device, generator)
    shape = lambda heads: (total, heads, d)
    inputs = dict(
        q=torch.randn(shape(args.num_q_heads), generator=generator, device=device).to(dtype),
        k=torch.randn(shape(args.num_kv_heads), generator=generator, device=device).to(dtype),
        v=torch.randn(shape(args.num_kv_heads), generator=generator, device=device).to(dtype),
        sel_blocks=sel_blocks,
        lo=torch.zeros(total, dtype=torch.int32, device=device),
        hi=torch.arange(total, dtype=torch.int32, device=device),  # causal: keys up to and including the query
        blk_base=torch.zeros(total, dtype=torch.int32, device=device),  # one packed sequence
        tok_base=torch.zeros(total, dtype=torch.int32, device=device),
        block_first=block_first,
        block_last=block_last,
        scale=1.0 / (d**0.5),
        block_size=args.block_size,
    )

    if args.profile_pass == "fwd":
        fn = lambda: qsa_forward(**inputs)
        unsupported_if_declined(fn)
        detail = "qsa_forward public call (allocation + compaction + kernels)"
    else:
        saved = unsupported_if_declined(lambda: qsa_forward(**inputs))
        grad_out = torch.randn(shape(args.num_q_heads), generator=generator, device=device).to(dtype)
        fn = lambda: qsa_backward(grad_out, **inputs, out_fp32=saved["out_fp32"], lse=saved["lse"])
        unsupported_if_declined(fn)
        detail = "qsa_backward public call (allocation + compaction + kernels)"
    detail += f" gathered_tokens={gathered} blocks_per_query<={args.topk}+tail"

    ms = time_fn(fn, args.num_warmup_iterations, args.num_iterations)
    sm_mhz = sample_sm_clock_mhz()

    # The gathered-token total takes the place of s_q * topk in the DSA per-row formulas.
    flops = (flops_fwd if args.profile_pass == "fwd" else flops_bwd)(1, args.num_q_heads, gathered, d, d)
    tflops = flops / (ms * 1e-3) / 1e12
    peak = peak_mma_tflops(args.data_type, sm_mhz)
    peak_str = f"{peak:.3f}" if peak else ""
    gpu = torch.cuda.get_device_name().replace(",", ";")
    print(f"RESULT,{ms:.6f},{tflops:.3f},{peak_str},{gpu},{cudnn.__version__},{detail.replace(',', ';')}")


if __name__ == "__main__":
    main()
