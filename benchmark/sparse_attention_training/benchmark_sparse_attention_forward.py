#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the generic sparse-attention forward API for shipped variants.

Times ``cudnn.sparse_attention_forward_wrapper`` under the geometry of three
production sparse-attention architectures:

* ``dsv4`` — DeepSeek-V4 DSA/CSA core attention: MQA over a shared 512-d
  latent (K aliased as V, ``D_k = D_v = 512``; RoPE lives in-place on dims
  448-511, no widened head), token-level top-2048, attention sink.
* ``qwen3.8`` — Qwen3.8-Flash-Next QSA: GQA 24Q/2KV, d=256, micro-block
  granularity 4, 2048-token budget (512 entries), shared indices.
* ``minimax`` — MiniMax-M3 MSA: GQA 64Q/4KV, d=128, block granularity 128,
  top-16 blocks per KV-head group (per-group indices).
* ``glm5.2`` — GLM-5/5.1/5.2 DSA: V3.2-shaped MQA latent (576-d K = 512
  latent + 64 RoPE, 512-d V), token top-2048, no sink. (5.2's IndexShare is
  indexer-side only; the core attention call shape is unchanged.)
* ``glm5.3-flash`` — GLM-5.3-Flash DSA layers: NoPE MLA, rope-free
  ``D_k = D_v = 512`` shared latent, token top-2048, no sink (11 of 45
  layers; the rest are KDA linear attention, out of scope here).

Indices are causal-realistic: query row ``i`` selects unique random entries
from its causal prefix (``i // granularity + 1`` candidates), up to the
variant's top-k; ``topk_length`` carries the per-row valid count. Index
generation is row-chunked so no ``S x S`` buffer is ever materialized.

Runs through the dispatched device kernels; variants whose configuration no
registered kernel serves fail with ``NotImplementedError``. ``--q-chunk``
splits each call over query-row chunks — correct under this API because
indices are storage-native (global) ids, so a row's selection is independent
of how rows are batched into calls.

Usage:
    python benchmark_sparse_attention_forward.py --variant dsv4 --seqlens 4096
    python benchmark_sparse_attention_forward.py --variant dsv4,glm5.2 --seqlens 4096,8192 --csv out.csv
    python benchmark_sparse_attention_forward.py profile --variant dsv4 --seqlens 8192

``profile`` mode runs one warmed-up forward call wrapped in
``cudaProfilerStart/Stop`` and an NVTX range for nsys/ncu capture; it uses
the first value of ``--seqlens``.
"""

import argparse
import csv
import dataclasses
import math
from typing import Optional

import torch

try:
    from cudnn.sparse_attention import sparse_attention_forward_wrapper
except ImportError as e:  # e.g. cudnn not installed, or binary incompatible with this node
    sparse_attention_forward_wrapper, _CUDNN_IMPORT_ERROR = None, e
else:
    _CUDNN_IMPORT_ERROR = None

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclasses.dataclass(frozen=True)
class VariantConfig:
    name: str
    h_q: int
    h_kv: int
    d_k: int
    d_v: int
    granularity: int  # tokens per index entry
    topk: int  # entries per row (per group)
    group_scope: int  # 1 = shared across heads, h_kv = per KV-head group, h_q = per head
    attn_sink: bool
    kv_aliased: bool  # single latent tensor serves as K and V


VARIANTS = {
    # DeepSeek-V4 DSA/CSA core attention: 64-head geometry, shared 512-d
    # latent (K = V, RoPE in-place on dims 448-511 — head is NOT widened to
    # 576 as in V3.2).
    "dsv4": VariantConfig("dsv4", h_q=64, h_kv=1, d_k=512, d_v=512, granularity=1, topk=2048, group_scope=1, attn_sink=True, kv_aliased=True),
    # Qwen3.8-Flash-Next QSA: 24Q/2KV @ 256, r=4 micro-blocks, K=2048 tokens.
    "qwen3.8": VariantConfig("qwen3.8", h_q=24, h_kv=2, d_k=256, d_v=256, granularity=4, topk=512, group_scope=1, attn_sink=False, kv_aliased=False),
    # MiniMax-M3 MSA: 64Q/4KV @ 128, block=128, top-16 blocks per GQA group.
    "minimax": VariantConfig("minimax", h_q=64, h_kv=4, d_k=128, d_v=128, granularity=128, topk=16, group_scope=4, attn_sink=False, kv_aliased=False),
    # GLM-5/5.1/5.2 DSA (V3.2 shape): 64 heads over 512-latent + 64-RoPE =
    # 576-d K, 512-d V, token top-2048, no sink.
    "glm5.2": VariantConfig("glm5.2", h_q=64, h_kv=1, d_k=576, d_v=512, granularity=1, topk=2048, group_scope=1, attn_sink=False, kv_aliased=True),
    # GLM-5.3-Flash DSA layers: NoPE MLA, rope-free 512-d shared latent
    # (qk_rope_head_dim=0), token top-2048, no sink.
    "glm5.3-flash": VariantConfig("glm5.3-flash", h_q=64, h_kv=1, d_k=512, d_v=512, granularity=1, topk=2048, group_scope=1, attn_sink=False, kv_aliased=True),
}


def make_causal_topk(seqlen_q: int, cfg: VariantConfig, device: str, row_chunk: int = 4096):
    """Unique random entry ids from each row's causal prefix, -1 padded.

    Returns ``(topk_idxs (S_q, [G,] topk) int32, topk_length (S_q[, G]) int32)``.
    Row ``i`` may select from entries ``0 .. i // g`` (its causal prefix at
    entry granularity); rows select ``min(topk, prefix)`` entries.
    """
    g = cfg.granularity
    n_groups = cfg.group_scope
    n_entries = (seqlen_q + g - 1) // g
    idxs = torch.full((seqlen_q, n_groups, cfg.topk), -1, dtype=torch.int32, device=device)
    lengths = torch.zeros(seqlen_q, n_groups, dtype=torch.int32, device=device)
    for lo in range(0, seqlen_q, row_chunk):
        hi = min(lo + row_chunk, seqlen_q)
        rows = hi - lo
        prefix = (torch.arange(lo, hi, device=device) // g + 1).clamp(max=n_entries)  # (rows,)
        scores = torch.rand(rows, n_groups, n_entries, device=device)
        # Push out-of-prefix entries past every in-prefix entry, then argsort:
        # the first `prefix` positions of each row are a random permutation of
        # the causal prefix.
        scores += (torch.arange(n_entries, device=device).view(1, 1, -1) >= prefix.view(-1, 1, 1)).float() * 2.0
        order = scores.argsort(dim=-1)[:, :, : cfg.topk].to(torch.int32)
        n_valid = prefix.clamp(max=cfg.topk).to(torch.int32)  # (rows,)
        idxs[lo:hi] = order
        slot = torch.arange(cfg.topk, device=device).view(1, 1, -1)
        idxs[lo:hi] = torch.where(slot < n_valid.view(-1, 1, 1), idxs[lo:hi], torch.full_like(idxs[lo:hi], -1))
        lengths[lo:hi] = n_valid.view(-1, 1).expand(rows, n_groups)
    if n_groups == 1:
        return idxs.squeeze(1).contiguous(), lengths.squeeze(1).contiguous()
    return idxs.contiguous(), lengths.contiguous()


def make_inputs(seqlen_q: int, cfg: VariantConfig, dtype: torch.dtype, device: str = "cuda"):
    q = torch.randn(seqlen_q, cfg.h_q, cfg.d_k, device=device, dtype=dtype) / 10
    if cfg.kv_aliased:
        kv = torch.randn(seqlen_q, cfg.h_kv, cfg.d_k, device=device, dtype=dtype) / 10
        k, v = kv, kv[:, :, : cfg.d_v]
    else:
        k = torch.randn(seqlen_q, cfg.h_kv, cfg.d_k, device=device, dtype=dtype) / 10
        v = torch.randn(seqlen_q, cfg.h_kv, cfg.d_v, device=device, dtype=dtype) / 10
    topk_idxs, topk_length = make_causal_topk(seqlen_q, cfg, device)
    attn_sink = torch.linspace(-2.0, 2.0, cfg.h_q, device=device, dtype=torch.float32) if cfg.attn_sink else None
    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
    return q, k, v, topk_idxs, topk_length, attn_sink, cu_seqlens_q


def flops_fwd(cfg: VariantConfig, topk_length: torch.Tensor) -> int:
    """Exact 2-matmul FLOP count (QK^T + PV) from the generated valid lengths."""
    heads_per_group = cfg.h_q if cfg.group_scope == 1 else cfg.h_q // cfg.group_scope if cfg.group_scope == cfg.h_kv else 1
    selected_tokens = int(topk_length.to(torch.int64).sum()) * cfg.granularity
    return 2 * selected_tokens * heads_per_group * (cfg.d_k + cfg.d_v)


def run_forward(inputs, cfg: VariantConfig, q_chunk: Optional[int]):
    q, k, v, topk_idxs, topk_length, attn_sink, cu_seqlens_q = inputs
    if q_chunk is None or q_chunk >= q.shape[0]:
        return sparse_attention_forward_wrapper(
            q,
            k,
            v,
            topk_idxs,
            topk_length=topk_length,
            index_granularity=cfg.granularity,
            attn_sink=attn_sink,
            cu_seqlens_q=cu_seqlens_q,
        )
    # Storage-native (global) ids make query chunking exact: each chunk sees
    # the full K/V and its own slice of rows/indices.
    device = q.device
    for lo in range(0, q.shape[0], q_chunk):
        hi = min(lo + q_chunk, q.shape[0])
        cu = torch.tensor([0, hi - lo], dtype=torch.int32, device=device)
        result = sparse_attention_forward_wrapper(
            q[lo:hi],
            k,
            v,
            topk_idxs[lo:hi].contiguous(),
            topk_length=topk_length[lo:hi].contiguous(),
            index_granularity=cfg.granularity,
            attn_sink=attn_sink,
            cu_seqlens_q=cu,
        )
    return result


def bench_config(cfg: VariantConfig, seqlen_q: int, dtype: torch.dtype, q_chunk: Optional[int], warmup: int, repeat: int):
    inputs = make_inputs(seqlen_q, cfg, dtype)
    flops = flops_fwd(cfg, inputs[4])

    for _ in range(warmup):
        run_forward(inputs, cfg, q_chunk)
    torch.cuda.synchronize()

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run_forward(inputs, cfg, q_chunk)
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / repeat
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops, flops


def profile_config(cfg: VariantConfig, seqlen_q: int, dtype: torch.dtype, q_chunk: Optional[int]):
    inputs = make_inputs(seqlen_q, cfg, dtype)
    run_forward(inputs, cfg, q_chunk)  # warm + compile
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range(f"sparse_attention_fwd_{cfg.name}_s{seqlen_q}"):
        run_forward(inputs, cfg, q_chunk)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="bench", choices=["bench", "profile"])
    parser.add_argument("--variant", default="dsv4,qwen3.8,minimax,glm5.2,glm5.3-flash", help="comma-separated subset of: " + ",".join(VARIANTS))
    parser.add_argument("--seqlens", default="4096,8192", help="comma-separated seqlen_q (= seqlen_kv) values")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    parser.add_argument("--backend", default="default", help='"default" (device kernels) or "reference" (PyTorch)')
    parser.add_argument("--q-chunk", type=int, default=None, help="split each call over query-row chunks (bounds reference memory)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if sparse_attention_forward_wrapper is None:
        raise SystemExit(f"cudnn import failed: {_CUDNN_IMPORT_ERROR}")

    dtype = DTYPES[args.dtype]
    variants = [VARIANTS[v.strip()] for v in args.variant.split(",")]
    seqlens = [int(s) for s in args.seqlens.split(",")]

    if args.mode == "profile":
        profile_config(variants[0], seqlens[0], dtype, args.q_chunk)
        return

    rows = []
    header = f"{'variant':>8} {'seqlen':>8} {'heads':>9} {'d_k/d_v':>9} {'gran':>5} {'topk':>5} {'ms':>10} {'TFLOPS':>9}"
    print(header)
    print("-" * len(header))
    for cfg in variants:
        for s in seqlens:
            ms, tflops, flops = bench_config(cfg, s, dtype, args.q_chunk, args.warmup, args.repeat)
            print(
                f"{cfg.name:>8} {s:>8} {f'{cfg.h_q}/{cfg.h_kv}':>9} {f'{cfg.d_k}/{cfg.d_v}':>9} "
                f"{cfg.granularity:>5} {cfg.topk:>5} {ms:>10.3f} {tflops:>9.2f}"
            )
            rows.append(
                dict(
                    variant=cfg.name,
                    seqlen=s,
                    h_q=cfg.h_q,
                    h_kv=cfg.h_kv,
                    d_k=cfg.d_k,
                    d_v=cfg.d_v,
                    granularity=cfg.granularity,
                    topk=cfg.topk,
                    group_scope=cfg.group_scope,
                    dtype=args.dtype,
                    ms=round(ms, 4),
                    tflops=round(tflops, 2),
                    flops=flops,
                )
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
