#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the generic sparse-attention forward API for shipped variants.

Times ``cudnn.sparse_attention_forward_wrapper`` (device kernels) or the
normative PyTorch oracle (``reference_sparse_attention_forward``) under the
geometry of named, production sparse-attention architectures:

* ``dsv4`` — DeepSeek-V4 DSA core attention: MQA over a shared 512-d latent
  (K aliased as V, ``D_k = D_v = 512``; RoPE lives in-place on dims 448-511,
  no widened head), token-level top-2048, attention sink. Device kernel
  (SM100 DSA-prefill envelope).
* ``csa-dsv4-1024`` / ``csa-dsv4-512`` (#825) — DeepSeek-V4 CSA: selection
  over *compressed* entries (m=4, so ``granularity=4`` over the same 512-d
  aliased latent), top-1024/512 entries. Benchmarks the compressed-entry
  slice only — the two-stream union with a last-128-token sliding window +
  sink (the deferred ``ExtraKV`` argument) is **not** implemented by the
  wrapper yet and is not benchmarked here; see the ``VariantConfig`` comment
  for the documented gap. No registered kernel (``granularity=4`` falls
  outside the DSA-prefill envelope) — reference backend only.
* ``qwen3.8`` (#826) — Qwen3.8-Flash-Next QSA, issue-literal shape: GQA
  24Q/2KV, d=256, micro-block granularity 4, 2048-token budget (512
  entries), indices *shared* across heads, forced-tail-block index
  generation (the row's own incomplete trailing block is always attended).
  Shared indices don't match the GQA-substrate kernel's ``G == H_kv``
  requirement — reference backend only.
* ``qwen3.8-gqa`` (#826) — same shape, per-KV-head-group indices
  (``G = H_kv = 2``) — the shape the registered SM100 GQA-substrate kernel
  (PR4) actually serves, so this row gets real device-kernel numbers.
* ``minimax`` — MiniMax-M3 MSA: GQA 64Q/4KV, d=128, block granularity 128,
  top-16 blocks per KV-head group (per-group indices). Device kernel
  (SM100 GQA-substrate envelope).
* ``glm5.2`` (#827) — GLM-5/5.1/5.2 DSA: V3.2-shaped MQA latent (576-d K =
  512 latent + 64 RoPE, 512-d V), token top-2048, no sink. (5.2's IndexShare
  is indexer-side only; the core attention call shape is unchanged, so this
  reports attention-only cost as the issue asks.) Device kernel.
* ``glm5.3-flash`` (#827) — GLM-5.3-Flash DSA layers: NoPE MLA, rope-free
  ``D_k = D_v = 512`` shared latent, token top-2048, no sink (11 of 45
  layers; the rest are KDA linear attention, out of scope here). Device
  kernel.

Indices are causal-realistic: query row ``i`` selects unique random entries
from its causal prefix (``i // granularity + 1`` candidates), up to the
variant's top-k; ``topk_length`` carries the per-row valid count. Index
generation is row-chunked so no ``S x S`` buffer is ever materialized.

``--backend default`` (the default) dispatches each variant automatically:
device kernel when its shape is in a registered envelope
(``VariantConfig.expect_device_kernel``, kept in lockstep with
``python/cudnn/sparse_attention/fwd/api.py``'s ``check_support``), else the
PyTorch reference oracle — no variant errors out. ``--backend reference``
forces the oracle for every variant. ``--q-chunk`` splits each call over
query-row chunks (needed to bound reference-oracle memory at longer
seqlens) — correct under this API because indices are storage-native
(global) ids, so a row's selection is independent of how rows are batched
into calls. Indexer/top-k cost (the ``ExtraKV`` window branch above, and
issues #829-831) is explicitly out of scope for this harness.

Usage:
    python benchmark_sparse_attention_forward.py --variant dsv4 --seqlens 4096
    python benchmark_sparse_attention_forward.py --variant dsv4,glm5.2 --seqlens 4096,8192 --csv out.csv
    python benchmark_sparse_attention_forward.py --variant csa-dsv4-1024,qwen3.8 --seqlens 8192 --q-chunk 512
    python benchmark_sparse_attention_forward.py profile --variant dsv4 --seqlens 8192

``profile`` mode runs one warmed-up forward call wrapped in
``cudaProfilerStart/Stop`` and an NVTX range for nsys/ncu capture; it uses
the first value of ``--seqlens``.
"""

import argparse
import csv
import dataclasses
import math
import os
import sys
from typing import Optional

import torch

try:
    from cudnn.sparse_attention import sparse_attention_forward_wrapper
except ImportError as e:  # e.g. cudnn not installed, or binary incompatible with this node
    sparse_attention_forward_wrapper, _CUDNN_IMPORT_ERROR = None, e
else:
    _CUDNN_IMPORT_ERROR = None

# The normative PyTorch oracle (test/python/sparse_attention/sparse_attention_reference.py)
# doubles as the "reference backend" here: variants no device kernel serves yet
# (see VariantConfig.expect_device_kernel) are benchmarked against it directly
# rather than through sparse_attention_forward_wrapper (whose frozen signature
# has no backend-selection knob -- it always dispatches to a device kernel or
# raises NotImplementedError, it never silently falls back).
_TEST_PYTHON_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "test", "python"))
if _TEST_PYTHON_DIR not in sys.path:
    sys.path.insert(0, _TEST_PYTHON_DIR)
try:
    from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward
except ImportError as e:
    reference_sparse_attention_forward, _REFERENCE_IMPORT_ERROR = None, e
else:
    _REFERENCE_IMPORT_ERROR = None

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
    force_tail: bool = False  # always attend the row's own (incomplete) trailing block, QSA-style
    expect_device_kernel: bool = True  # False = no registered kernel serves this envelope yet (see api.py check_support); benchmark falls back to the reference oracle
    note: str = ""  # short free-text annotation carried through to the printed table / CSV


VARIANTS = {
    # DeepSeek-V4 DSA/CSA core attention: 64-head geometry, shared 512-d
    # latent (K = V, RoPE in-place on dims 448-511 — head is NOT widened to
    # 576 as in V3.2). Token-granularity DSA slice of DSv4 -- hits the SM100
    # DSA-prefill envelope (H_kv=1, K aliased as V, granularity=1).
    "dsv4": VariantConfig("dsv4", h_q=64, h_kv=1, d_k=512, d_v=512, granularity=1, topk=2048, group_scope=1, attn_sink=True, kv_aliased=True),
    # DeepSeek-V4 CSA (#825): selection over *compressed* entries -- an m=4
    # learned compression makes each index entry cover 4 raw tokens of the
    # latent stream, so kernel-side this is the same aliased-latent geometry
    # as `dsv4` with granularity=4 and a T/4-length effective KV axis
    # (`seqlen_q` below is still the raw token count; the entry count the
    # generator draws from is `seqlen_q / 4`, matching a real compressed
    # stream). Two budgets per the issue (top-1024 / top-512 entries).
    #
    # GAP (documented, not fabricated): full CSA additionally unions these
    # compressed entries with a last-128-raw-token sliding window and the
    # per-head sink logits in *one* softmax denominator -- the deferred
    # `ExtraKV` argument in the forward contract (python/cudnn/sparse_attention/
    # fwd/api.py has no such parameter yet). This variant benchmarks the
    # compressed-entry-only slice; the window/ExtraKV branch is out of scope
    # until the wrapper grows that argument. granularity=4 also falls outside
    # the DSA-prefill envelope (which pins granularity==1), so no device
    # kernel serves this shape yet either -- reference-backend only.
    "csa-dsv4-1024": VariantConfig(
        "csa-dsv4-1024", h_q=64, h_kv=1, d_k=512, d_v=512, granularity=4, topk=1024, group_scope=1, attn_sink=True, kv_aliased=True,
        expect_device_kernel=False, note="compressed-entries only; window/ExtraKV union not benchmarked (contract gap)",
    ),
    "csa-dsv4-512": VariantConfig(
        "csa-dsv4-512", h_q=64, h_kv=1, d_k=512, d_v=512, granularity=4, topk=512, group_scope=1, attn_sink=True, kv_aliased=True,
        expect_device_kernel=False, note="compressed-entries only; window/ExtraKV union not benchmarked (contract gap)",
    ),
    # Qwen3.8-Flash-Next QSA (#826), literal issue shape: 24Q/2KV @ 256,
    # r=4 micro-blocks, K=2048 tokens (512 entries), indices *shared* across
    # heads (group_scope=1) as the issue states. Forced-tail-block semantics
    # (QSA always attends the row's own incomplete trailing block) are
    # applied regardless of index scope. Shared indices (group_scope=1) fall
    # outside the GQA-substrate envelope (which pins group_scope==H_kv), so
    # this shape has no registered device kernel -- reference-backend only.
    "qwen3.8": VariantConfig(
        "qwen3.8", h_q=24, h_kv=2, d_k=256, d_v=256, granularity=4, topk=512, group_scope=1, attn_sink=False, kv_aliased=False,
        force_tail=True, expect_device_kernel=False, note="shared indices (issue-literal); group_scope=1 != H_kv, no GQA-substrate match",
    ),
    # Qwen3.8-shaped QSA on the PR4 GQA-substrate envelope: identical shape
    # to `qwen3.8` but with per-KV-head-group indices (group_scope=H_kv=2)
    # instead of shared -- the shape the registered SM100 GQA-substrate
    # kernel (G=H_kv, granularity in (4, 64, 128), BF16) actually serves, so
    # this row reports real device-kernel numbers instead of reference-only.
    "qwen3.8-gqa": VariantConfig(
        "qwen3.8-gqa", h_q=24, h_kv=2, d_k=256, d_v=256, granularity=4, topk=512, group_scope=2, attn_sink=False, kv_aliased=False,
        force_tail=True, expect_device_kernel=True, note="per-KV-group indices (G=H_kv) to hit the registered GQA-substrate kernel",
    ),
    # MiniMax-M3 MSA: 64Q/4KV @ 128, block=128, top-16 blocks per GQA group.
    # G=H_kv, granularity=128 -- already on the GQA-substrate envelope.
    "minimax": VariantConfig("minimax", h_q=64, h_kv=4, d_k=128, d_v=128, granularity=128, topk=16, group_scope=4, attn_sink=False, kv_aliased=False),
    # GLM-5/5.1/5.2 DSA (V3.2 shape, #827): 64 heads over 512-latent + 64-RoPE
    # = 576-d K, 512-d V, token top-2048, no sink. Hits the DSA-prefill
    # envelope (D_k=576 splits QK=576/V=512). #827 also asks that this report
    # attention-only cost (5.2's IndexShare runs the indexer on 1 of 4 layers
    # but does not change the core-attention call shape) -- this harness only
    # ever measures the core-attention call, so no extra accounting is needed
    # here; the sharing shows up in the (out-of-scope, #829-831) indexer bench.
    "glm5.2": VariantConfig("glm5.2", h_q=64, h_kv=1, d_k=576, d_v=512, granularity=1, topk=2048, group_scope=1, attn_sink=False, kv_aliased=True),
    # GLM-5.3-Flash DSA layers (#827): NoPE MLA, rope-free 512-d shared latent
    # (qk_rope_head_dim=0), token top-2048, no sink; 11 of 45 layers run this
    # (the rest are KDA linear attention, out of scope for this harness) --
    # per-call cost is what's reported; model-level 11/45 amortization is a
    # documentation-level multiplier, not something this op-level bench does.
    "glm5.3-flash": VariantConfig("glm5.3-flash", h_q=64, h_kv=1, d_k=512, d_v=512, granularity=1, topk=2048, group_scope=1, attn_sink=False, kv_aliased=True),
}


def make_causal_topk(seqlen_q: int, cfg: VariantConfig, device: str, row_chunk: int = 4096):
    """Unique random entry ids from each row's causal prefix, -1 padded.

    Returns ``(topk_idxs (S_q, [G,] topk) int32, topk_length (S_q[, G]) int32)``.
    Row ``i`` may select from entries ``0 .. i // g`` (its causal prefix at
    entry granularity); rows select ``min(topk, prefix)`` entries.

    ``cfg.force_tail`` (QSA, #826) additionally pins each row's own
    (possibly incomplete) trailing block -- entry ``i // g`` -- into slot 0
    whenever it wasn't already drawn, matching QSA's "always attend the
    current block" semantics; it never reduces ``topk_length`` since that
    entry is always within the causal prefix.
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
        if cfg.force_tail:
            own_entry = (torch.arange(lo, hi, device=device) // g).clamp(max=n_entries - 1).to(torch.int32)  # (rows,)
            already_selected = (idxs[lo:hi] == own_entry.view(-1, 1, 1)).any(dim=-1)  # (rows, n_groups)
            forced_slot0 = torch.where(already_selected, idxs[lo:hi, :, 0], own_entry.view(-1, 1).expand(rows, n_groups))
            idxs[lo:hi, :, 0] = forced_slot0
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
    """Exact 2-matmul FLOP count (QK^T + PV) from the generated valid lengths.

    RECONCILIATION vs speed-of-light's ``pr4_flops(t_q, h_q, topk, gran, d_k,
    d_v) = 2*t_q*h_q*topk*gran*(d_k+d_v)`` (sparse_attention_training_fprop/
    generate.py): the two are the SAME formula in the group_scope==h_kv
    (GQA-substrate / PR4) case --

        flops_fwd            = 2 * (sum(topk_length) * gran) * (h_q // h_kv) * (d_k + d_v)
        pr4_flops             = 2 *  t_q * topk        * gran *  h_q         * (d_k + d_v)

    which are algebraically identical iff ``sum(topk_length) * (h_q // h_kv)
    == t_q * topk * h_q``, i.e. iff ``sum(topk_length) == t_q * h_kv * topk``
    -- true ONLY when every (row, kv-group) selects the full nominal `topk`.
    ``make_causal_topk()`` clamps early rows to `min(i // gran + 1, topk)`
    valid entries (row i's causal prefix), so at the seqlens this harness
    actually runs (4096/8192 vs e.g. qwen3.8-gqa's topk*gran = 512*4 = 2048
    ramp length), a real fraction of rows are still ramping up and
    ``sum(topk_length) < t_q * h_kv * topk``. Measured on this box
    (seqlen_q=4096, bf16, default/device backend):
      * qwen3.8-gqa (h_q=24,h_kv=2,d=256,gran=4,topk=512): flops_fwd() gives
        ~1.55e11 FLOPs vs pr4_flops()'s causal-blind ~2.06e11 (nominal is
        ~1.33x actual -- expected, NOT a bug in either formula: pr4_flops()
        is deliberately the "every row saturated" closed form, per its own
        docstring; flops_fwd() is deliberately the exact as-generated count).
      * Both configs' *measured* wall-clock TFLOPS via this file's
        bench_config() (flops_fwd() / elapsed ms) reproduce at 1.18 TFLOPS
        (qwen3.8-gqa, seqlen=4096) and 0.94 TFLOPS (minimax, seqlen=4096;
        see results/device_kernel_sweep.csv -- minimax's measured TFLOPS
        ranges 0.94-1.26 across seqlens 4096-65536 in that file) against a
        pr4_gqa_ai()+roofline_bound() BW-bound ceiling of 96/128 TFLOPS on
        GB300 -- i.e. ~1-1.4% MFU, consistent with the current scalar-FFMA
        (no tensor-core) kernel plus per-call Python/host dispatch overhead
        for this index-driven gather. A prior round's independently
        "re-measured" 68.9-267.5 TFLOPS for this same scalar kernel is NOT
        reproducible via this function + bench_config()'s wall-clock timing
        on this hardware (58x-227x higher than what's measured here) and
        exceeds the BW-bound roofline ceiling for one of the two configs --
        it did not come from flops_fwd() and a real elapsed-time measurement
        of sparse_attention_forward_wrapper; treat any future report that
        doesn't reproduce through THIS function + bench_config() as suspect.
        See pr4_flops()/pr4_gqa_ai()/roofline_bound() in speed-of-light's
        sparse_attention_training_fprop/generate.py for the theoretical
        (causal-blind) side of this cross-check.
    """
    heads_per_group = cfg.h_q if cfg.group_scope == 1 else cfg.h_q // cfg.group_scope if cfg.group_scope == cfg.h_kv else 1
    selected_tokens = int(topk_length.to(torch.int64).sum()) * cfg.granularity
    return 2 * selected_tokens * heads_per_group * (cfg.d_k + cfg.d_v)


def _resolve_backend(cfg: VariantConfig, requested: str) -> str:
    """"default" tries the device kernel and only serves what the frozen
    wrapper's ``check_support`` actually registers; "reference" always uses
    the PyTorch oracle. Since the wrapper itself has no backend-selection
    knob (it dispatches to a device kernel or raises ``NotImplementedError``,
    never a silent fallback), "default" here is driven by
    ``cfg.expect_device_kernel`` -- kept in lockstep with api.py's
    ``check_support`` envelopes by the comments on each ``VariantConfig``.
    """
    if requested == "reference":
        return "reference"
    if requested == "default":
        return "default" if cfg.expect_device_kernel else "reference"
    raise ValueError(f"unknown --backend {requested!r}")


def run_forward(inputs, cfg: VariantConfig, q_chunk: Optional[int], backend: str):
    q, k, v, topk_idxs, topk_length, attn_sink, cu_seqlens_q = inputs
    if backend == "reference":
        if reference_sparse_attention_forward is None:
            raise SystemExit(f"reference oracle import failed: {_REFERENCE_IMPORT_ERROR}")

        def call(qq, kk, vv, ii, ll, ss, cu):
            out, lse = reference_sparse_attention_forward(
                qq, kk, vv, ii, topk_length=ll, attn_sink=ss, index_granularity=cfg.granularity
            )
            return {"out": out, "lse": lse}

    else:

        def call(qq, kk, vv, ii, ll, ss, cu):
            return sparse_attention_forward_wrapper(
                qq, kk, vv, ii, topk_length=ll, index_granularity=cfg.granularity, attn_sink=ss, cu_seqlens_q=cu
            )

    if q_chunk is None or q_chunk >= q.shape[0]:
        return call(q, k, v, topk_idxs, topk_length, attn_sink, cu_seqlens_q)
    # Storage-native (global) ids make query chunking exact: each chunk sees
    # the full K/V and its own slice of rows/indices.
    device = q.device
    for lo in range(0, q.shape[0], q_chunk):
        hi = min(lo + q_chunk, q.shape[0])
        cu = torch.tensor([0, hi - lo], dtype=torch.int32, device=device)
        result = call(q[lo:hi], k, v, topk_idxs[lo:hi].contiguous(), topk_length[lo:hi].contiguous(), attn_sink, cu)
    return result


def bench_config(cfg: VariantConfig, seqlen_q: int, dtype: torch.dtype, q_chunk: Optional[int], warmup: int, repeat: int, requested_backend: str = "default"):
    backend = _resolve_backend(cfg, requested_backend)
    inputs = make_inputs(seqlen_q, cfg, dtype)
    flops = flops_fwd(cfg, inputs[4])

    for _ in range(warmup):
        run_forward(inputs, cfg, q_chunk, backend)
    torch.cuda.synchronize()

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run_forward(inputs, cfg, q_chunk, backend)
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / repeat
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops, flops, backend


def profile_config(cfg: VariantConfig, seqlen_q: int, dtype: torch.dtype, q_chunk: Optional[int], requested_backend: str = "default"):
    backend = _resolve_backend(cfg, requested_backend)
    inputs = make_inputs(seqlen_q, cfg, dtype)
    run_forward(inputs, cfg, q_chunk, backend)  # warm + compile
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range(f"sparse_attention_fwd_{cfg.name}_s{seqlen_q}_{backend}"):
        run_forward(inputs, cfg, q_chunk, backend)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="bench", choices=["bench", "profile"])
    parser.add_argument(
        "--variant",
        default="dsv4,csa-dsv4-1024,csa-dsv4-512,qwen3.8,qwen3.8-gqa,minimax,glm5.2,glm5.3-flash",
        help="comma-separated subset of: " + ",".join(VARIANTS),
    )
    parser.add_argument("--seqlens", default="4096,8192", help="comma-separated seqlen_q (= seqlen_kv) values")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    parser.add_argument(
        "--backend",
        default="default",
        choices=["default", "reference"],
        help='"default" uses the device kernel when the variant is in a registered envelope '
        "(VariantConfig.expect_device_kernel), else falls back to the PyTorch reference oracle; "
        '"reference" always uses the oracle.',
    )
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
        profile_config(variants[0], seqlens[0], dtype, args.q_chunk, args.backend)
        return

    rows = []
    header = f"{'variant':>14} {'seqlen':>8} {'heads':>9} {'d_k/d_v':>9} {'gran':>5} {'topk':>5} {'ms':>10} {'TFLOPS':>9} {'backend':>9}"
    print(header)
    print("-" * len(header))
    for cfg in variants:
        for s in seqlens:
            ms, tflops, flops, backend = bench_config(cfg, s, dtype, args.q_chunk, args.warmup, args.repeat, args.backend)
            print(
                f"{cfg.name:>14} {s:>8} {f'{cfg.h_q}/{cfg.h_kv}':>9} {f'{cfg.d_k}/{cfg.d_v}':>9} "
                f"{cfg.granularity:>5} {cfg.topk:>5} {ms:>10.3f} {tflops:>9.2f} {backend:>9}"
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
                    backend=backend,
                    ms=round(ms, 4),
                    tflops=round(tflops, 2),
                    flops=flops,
                    note=cfg.note,
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
