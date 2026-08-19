#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark block-sparse attention inference (forward-only) in cuDNN Frontend.

Times the public ``cudnn.block_sparse_attention_forward`` API against the
FA4-lineage CuTe DSL block-sparse forward (``flash_attn.cute``, optional) on
identical block masks, across sparse-block granularities of 64, 128, and 256
tokens. A dense run (every block selected) is included per case to show the
kernel's peak as an upper reference for the sparse bars.

The workload models video-diffusion sparse attention (e.g. VSA): batch 1,
head_dim 128, bf16, non-causal, per-head data-dependent block masks selected
by top-k with the diagonal block always kept. The default cases are Wan2.1
text-to-video shapes — 1.3B (12 heads) and 14B (40 heads) at 480P
(S = 24x32x52 = 39936 latent tokens) and 720P (S = 24x48x80 = 92160).

Reported TFLOPS count only the selected blocks:

    FLOPs = 4 * H * D * S * keep_blocks_per_row * block_tokens

Both arms are fed the same selected-token set in every cell. Where a kernel
cannot express the requested granularity natively, the mask is losslessly
re-expressed at the granularity it supports (see ``--help`` and the README),
so TFLOPS stay comparable as work-done-per-second on the requested mask.

Usage:
    python benchmark_sparse_attention_inference.py
    python benchmark_sparse_attention_inference.py --cases wan14b-480p --sparsities 0.8,0.9
    python benchmark_sparse_attention_inference.py --granularities 64,128 --csv results.csv
    python benchmark_sparse_attention_inference.py --check
    python benchmark_sparse_attention_inference.py --plot results.png
"""

import argparse
import csv
import math
import sys

import torch

try:
    from cudnn.block_sparse_attention import block_sparse_attention_forward
except ImportError as e:
    block_sparse_attention_forward, _CUDNN_IMPORT_ERROR = None, e
else:
    _CUDNN_IMPORT_ERROR = None

HEAD_DIM = 128

# (label, num_heads, seqlen): Wan2.1 T2V latent shapes, padded to the VSA tile
# grid. 1.3B has 12 heads, 14B has 40; both use head_dim 128.
CASES = {
    "wan1.3b-480p": (12, 39936),
    "wan14b-480p": (40, 39936),
    "wan14b-720p": (40, 92160),
    # MiniMax-H3 (open weights): 56 heads x d128; ~31k visual tokens for a
    # 1344x768 124-frame clip, ~91k for a 15 s output (padded to 256-token
    # blocks). The release ships full attention with sparse support planned,
    # so the dense bar is its current cost and the sparse bars the headroom.
    "minimax-h3-5s": (56, 31488),
    "minimax-h3-15s": (56, 91392),
}


def make_block_mask(num_heads, seqlen, granularity, sparsity, seed=0):
    """Per-head boolean block mask: top-k of NB blocks per row, diagonal kept."""
    nb = seqlen // granularity
    keep = nb if sparsity == 0.0 else max(1, round((1.0 - sparsity) * nb))
    gen = torch.Generator("cpu").manual_seed(seed)
    scores = torch.rand(num_heads, nb, nb, generator=gen)
    eye = torch.eye(nb, dtype=torch.bool).unsqueeze(0).expand_as(scores)
    scores = scores.masked_fill(eye, 2.0)  # diagonal block always selected
    top = torch.topk(scores, keep, -1).indices
    mask = torch.zeros(num_heads, nb, nb, dtype=torch.bool).scatter_(-1, top, True)
    return mask, keep


def expand_mask(mask, factor):
    """Re-express a block mask at a granularity ``factor`` times finer."""
    return mask.repeat_interleave(factor, -1).repeat_interleave(factor, -2)


def mask_to_indices(mask):
    """Boolean (H, NBq, NBk) mask -> sorted per-row column indices + counts."""
    counts = mask.sum(-1)
    cap = int(counts.max())
    idx = mask.to(torch.int8).argsort(dim=-1, descending=True, stable=True)[..., :cap]
    idx, _ = idx.sort(dim=-1)
    return idx.to(torch.int32).contiguous(), counts.to(torch.int32).contiguous(), cap


def make_qkv(num_heads, seqlen, device="cuda"):
    torch.manual_seed(42)
    q = torch.randn(1, seqlen, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    return q, torch.randn_like(q), torch.randn_like(q)


# --------------------------------------------------------------------------- arms


def run_cudnn(q, k, v, mask, granularity):
    """cuDNN BSA forward. 64/128 are native; 256 is re-expressed on 128 blocks."""
    if granularity == 256:
        mask, sparse_block_size = expand_mask(mask, 2), 128
    else:
        sparse_block_size = granularity
    idx, counts, cap = mask_to_indices(mask)
    idx = idx.unsqueeze(0).cuda()
    nums = None
    if sparse_block_size == 128 and cap % 2:
        # The blk128 fixed-count path requires an even count; odd per-row
        # counts go through the variable-count contract with padded capacity.
        idx = torch.cat([idx, idx[..., -1:]], -1)
        nums = counts.unsqueeze(0).cuda()
        cap += 1
    return lambda: block_sparse_attention_forward(
        q, k, v, idx, block_sparse_num=cap, q2k_block_nums=nums,
        sparse_block_size=sparse_block_size, layout="bshd",
    )


def _load_fa4():
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    return BlockSparseTensorsTorch, _flash_attn_fwd


def run_fa4(q, k, v, mask, granularity):
    """FA4-lineage CuTe BSA forward (flash_attn.cute).

    The SM100 kernel selects at 256-token Q granularity and its KV tile caps
    at 128, so masks finer than that are aggregated on the Q side (rows in a
    256-token group attend the union of their blocks — extra work the shared
    FLOP count does not credit) and re-expressed on 128-token KV blocks for
    granularity 256.
    """
    BlockSparseTensorsTorch, _flash_attn_fwd = _load_fa4()
    if granularity == 256:
        mask, kv_block = expand_mask(mask, 2), 128
    else:
        kv_block = granularity
    q_group = 256 // kv_block
    if q_group > 1:
        h, nbq, nbk = mask.shape
        mask = mask.view(h, nbq // q_group, q_group, nbk).any(2)
    idx, counts, cap = mask_to_indices(mask)
    empty = torch.zeros_like(counts)
    sparse_tensors = BlockSparseTensorsTorch(
        full_block_cnt=counts.unsqueeze(0).cuda(),
        full_block_idx=idx.unsqueeze(0).cuda(),
        mask_block_cnt=empty.unsqueeze(0).cuda(),
        mask_block_idx=idx[..., :0].unsqueeze(0).cuda(),
        block_size=(256, kv_block),
    )
    return lambda: _flash_attn_fwd(
        q, k, v, tile_mn=(128, kv_block), block_sparse_tensors=sparse_tensors,
        causal=False, return_lse=True,
    )


ARMS = {"cudnn": run_cudnn, "fa4": run_fa4}


# --------------------------------------------------------------------------- bench


def time_fn(fn, warmup=3, target_ms=500.0, max_iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    est = start.elapsed_time(end)
    iters = max(5, min(max_iters, int(target_ms / max(est, 1e-3))))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_cell(arm, num_heads, seqlen, granularity, sparsity):
    mask, keep = make_block_mask(num_heads, seqlen, granularity, sparsity)
    q, k, v = make_qkv(num_heads, seqlen)
    fn = ARMS[arm](q, k, v, mask, granularity)
    ms = time_fn(fn)
    flops = 4 * num_heads * HEAD_DIM * seqlen * keep * granularity
    return ms, flops / ms * 1e-9


def check_parity(arms):
    """Small-shape parity of every arm/granularity against an fp32 reference."""
    num_heads, seqlen, sparsity = 4, 4096, 0.9
    failures = 0
    for granularity in (64, 128, 256):
        mask, _ = make_block_mask(num_heads, seqlen, granularity, sparsity)
        q, k, v = make_qkv(num_heads, seqlen)
        qf, kf, vf = (t.squeeze(0).permute(1, 0, 2).float() for t in (q, k, v))
        scores = qf @ kf.transpose(-1, -2) / math.sqrt(HEAD_DIM)
        token_mask = expand_mask(mask, granularity).cuda()
        scores = scores.masked_fill(~token_mask, float("-inf"))
        ref = (scores.softmax(-1) @ vf).permute(1, 0, 2)
        floor = (ref.to(torch.bfloat16).float() - ref).abs().max().item()
        for arm in arms:
            if arm == "fa4" and granularity < 256:
                continue  # FA4's Q aggregation attends a superset; not comparable
            out = ARMS[arm](q, k, v, mask, granularity)()[0]
            err = (out.squeeze(0).float() - ref).abs().max().item()
            ok = err < 8 * max(floor, 1e-3)
            failures += not ok
            print(f"parity {arm} blk{granularity}: max_err={err:.3e} "
                  f"bf16_floor={floor:.3e} -> {'PASS' if ok else 'FAIL'}")
    return failures


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Repo chart palette (benchmark/sdpa_benchmark_training/charts.py)
    colors = {"cudnn": "#76b900", "fa4": "#FFD700"}
    names = {"cudnn": "cuDNN BSA", "fa4": "FAv4 CuTe BSA"}
    cases = list(dict.fromkeys(r["case"] for r in rows))
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.5), squeeze=False)
    for ax, case in zip(axes[0], cases):
        cells = [r for r in rows if r["case"] == case]
        labels = list(dict.fromkeys(
            "dense" if r["sparsity"] == 0.0 else f"blk{r['granularity']}\n{int(r['sparsity'] * 100)}%"
            for r in cells))
        arms = list(dict.fromkeys(r["arm"] for r in cells))
        width = 0.8 / len(arms)
        for i, arm in enumerate(arms):
            vals = []
            for label in labels:
                match = [r["tflops"] for r in cells if r["arm"] == arm and (
                    ("dense" if r["sparsity"] == 0.0 else f"blk{r['granularity']}\n{int(r['sparsity'] * 100)}%") == label)]
                vals.append(match[0] if match else 0.0)
            xs = [j + (i - (len(arms) - 1) / 2) * width for j in range(len(labels))]
            bars = ax.bar(xs, vals, width, label=names[arm], color=colors[arm])
            ax.bar_label(bars, fmt="%.0f", fontsize=6)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylabel("TFLOP/s (selected blocks only)", fontsize=9)
        ax.set_title(case, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Block-sparse attention inference forward", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default=",".join(CASES),
                        help="comma-separated case names, or HxS pairs like 12x39936")
    parser.add_argument("--granularities", default="64,128,256")
    parser.add_argument("--sparsities", default="0.9",
                        help="comma-separated block-sparsity fractions (0 = dense)")
    parser.add_argument("--arms", default="cudnn,fa4")
    parser.add_argument("--no-dense", action="store_true",
                        help="skip the dense peak-reference run per case")
    parser.add_argument("--csv", help="write results to this CSV file")
    parser.add_argument("--plot", help="write a grouped-bar chart PNG to this path")
    parser.add_argument("--check", action="store_true",
                        help="run small-shape fp32 parity checks and exit")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "cudnn" in arms and block_sparse_attention_forward is None:
        sys.exit(f"cudnn import failed: {_CUDNN_IMPORT_ERROR}")
    if "fa4" in arms:
        try:
            _load_fa4()
        except ImportError:
            print("fa4 arm skipped: flash_attn.cute with block-sparsity support "
                  "not importable (needs flash-attn CuTe DSL + nvidia-cutlass-dsl "
                  "+ quack-kernels)")
            arms.remove("fa4")

    if args.check:
        sys.exit(check_parity(arms))

    cases = {}
    for name in args.cases.split(","):
        name = name.strip()
        if name in CASES:
            cases[name] = CASES[name]
        else:
            heads, seqlen = name.split("x")
            cases[name] = (int(heads), int(seqlen))
    granularities = [int(g) for g in args.granularities.split(",")]
    sparsities = [float(s) for s in args.sparsities.split(",")]

    rows = []
    print(f"{'case':<14} {'arm':<6} {'blk':>4} {'sparsity':>8} {'ms':>9} {'TFLOP/s':>8}")
    for case, (num_heads, seqlen) in cases.items():
        grid = [(g, s) for s in sparsities for g in granularities]
        if not args.no_dense:
            grid.append((128, 0.0))  # dense peak reference, granularity-independent
        for granularity, sparsity in grid:
            for arm in arms:
                ms, tflops = bench_cell(arm, num_heads, seqlen, granularity, sparsity)
                rows.append(dict(case=case, arm=arm, granularity=granularity,
                                 sparsity=sparsity, ms=round(ms, 3), tflops=round(tflops, 1)))
                label = "dense" if sparsity == 0.0 else f"{sparsity:.0%}"
                print(f"{case:<14} {arm:<6} {granularity:>4} {label:>8} "
                      f"{ms:>9.3f} {tflops:>8.0f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
