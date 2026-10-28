#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single DSA sparse-attention benchmark.

Runs one case of the cuDNN Frontend DeepSeek Sparse Attention kernels and
prints a parseable ``RESULT,`` line. Called as a subprocess by runner.py so
each case gets a clean CUDA context (CuTe DSL compile state included) and
failures stay independent.

A case is one ``(component, phase)`` cell of the suite:

  core / context      ``cudnn.DSA.SparseAttentionForward.execute`` with
                      preallocated outputs over a ``kv_seqlen``-row pool (one
                      request). Every query gathers ``topk`` unique random
                      rows of the pool (the per-token upper bound, as the
                      committed results were measured), so ``topk_length ==
                      topk`` on every row and the forward kernel's indexer
                      prefix contract (``topk_length >= indexer_topk``) holds.
  core / generation   The same kernel over ``batch_size`` requests of
                      ``kv_seqlen`` cached tokens and ``q_seqlen`` query rows
                      each (decode / MTP rows); ids stay inside the row's own
                      request and causal to its position (full lists at every
                      shipped ``kv_len``); one CTA per query row on the
                      prefill kernel, no split-KV path. Reports the bytes the
                      gather touches so the runner can derive GB/s.
  core / bprop        ``cudnn.DSA.SparseAttentionBackward.execute`` with
                      preallocated gradients and workspace. The ``out``/``lse``
                      it consumes come from a chunked PyTorch reference, so no
                      forward launch sits in the timed region.
  indexer / context   ``cudnn.DSA.indexer_forward_top_k_wrapper`` (fused
  indexer / generation  scoring + top-k) with caller-owned candidate and output
                      buffers; ``bfloat16`` keys, or ``mxfp8`` where Q and K
                      are e4m3 with packed e8m0 block scales. Generation rows
                      pass ``q_causal_offsets`` so each request's rows sit at
                      the end of its cache.
  indexer hierarchical  The DSv4.1 consumer step: ``SparseIndexerScoreRecompute``
                      over a synthesized causal-valid candidate list of 16384
                      ids (newest 8-token block pinned) followed by
                      ``IndexerTopK``; ``time_ms`` is the sum of the two
                      kernels (the source layer's block top-k is not a cuDNN
                      kernel and is not timed).
  indexer / bprop     ``cudnn.DSA.IndexerBackward.execute`` on the default
                      backend; the kernel overwrites ``attn_score`` in place,
                      so a pristine copy is restored before every call,
                      outside the timed window.

The KV pool is flat (``rows x D``) or paged through a block table
(``--paged``); either way the kernels gather flat ``int32`` row ids. Timing
is CUDA events around each steady-state call after a 256 MiB L2 flush,
median of ``--num_iterations``; kernel compilation happens in one untimed
probe call before warmup.

Exit code 3 (``UNSUPPORTED,<reason>`` on stdout) means the case is not
served: either the suite's own support rules (``support.decide_support``,
the same function the runner evaluates before spawning) or the kernel's own
support check rejected it. The runner records both as skipped. Any other
exception is a real failure and propagates with its traceback.

The shared CuTe DSL compiler and the indexer APIs read the CURRENT device
(``gpu_arch_flag`` is cached per process on it); run the worker with
``CUDA_VISIBLE_DEVICES`` pinned to one GPU, as the runner does.
"""

import argparse
import math
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    from . import kv_pool, support
except ImportError:  # run as a script by the runner: resolve the suite package from the repo root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dsa import kv_pool, support

UNSUPPORTED_EXIT_CODE = 3

# Dense-MMA FLOPs per clock per SM by compute-capability major, for the % SOL
# denominator (same table as attention_training/benchmark_single_sdpa.py).
# "mxfp8" is the block-scaled fp8 rate the mxfp8 indexer rows issue; no row
# issues an fp4 MMA, so there is no nvfp4 entry.
_FLOPS_PER_CLOCK_PER_SM = {
    9: {"bfloat16": 4096, "float16": 4096},
    10: {"bfloat16": 8192, "float16": 8192, "mxfp8": 16384},
    12: {"bfloat16": 1024, "float16": 1024},
}

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
KV_DTYPES = ("bfloat16", "mxfp8", "nvfp4")
INDEXER_HEAD_DIM = 128
MXFP8_SF_VEC_SIZE = 32

# Exceptions the kernels raise for contracts they do not serve. Stage A (construction,
# check_support) also maps RuntimeError: APIBase's arch and dtype gates raise it. Stage B
# (the untimed probe call that compiles) maps only the validation types; a RuntimeError
# from compile or launch is a real failure and propagates.
_STAGE_A_UNSUPPORTED = (ValueError, RuntimeError, TypeError, AssertionError)
_STAGE_B_UNSUPPORTED = (ValueError, TypeError, NotImplementedError, AssertionError)


def flops_fwd(s_q, num_heads, topk, d_qk, d_v, gathered=None):
    """QK^T over the gathered rows and PV: 2 matmuls over ``gathered`` KV rows (default ``s_q * topk``)."""
    n = s_q * topk if gathered is None else gathered
    return 2.0 * num_heads * n * (d_qk + d_v)


def flops_bwd(s_q, num_heads, topk, d_qk, d_v, gathered=None):
    """5 matmuls over ``gathered`` KV rows (default ``s_q * topk``): recompute S (d_qk), dV (d_v), dP (d_v), dQ (d_qk), dK (d_qk)."""
    n = s_q * topk if gathered is None else gathered
    return 2.0 * num_heads * n * (3 * d_qk + 2 * d_v)


def gathered_ids(rows, topk, topk_length) -> int:
    """KV rows the core kernels gather: ``sum(topk_length)`` under the compact id contract, else ``rows * topk``.

    Context / bprop lists are full, so both counts agree there; generation
    lists are causal to the row's position, so a row of a cache barely longer
    than ``topk`` holds fewer valid ids and the nominal count would overstate
    the TFLOPS on gathered rows.
    """
    return int(topk_length.sum()) if topk_length is not None else rows * topk


def flops_indexer_logits(indexer_heads, visible_keys):
    """Indexer scoring: one ``(H_idx x 128) . 128`` product per visible key; the top-k adds no MMA work."""
    return 2.0 * indexer_heads * INDEXER_HEAD_DIM * visible_keys


def flops_indexer_bwd(s_q, indexer_heads, topk):
    """Indexer backward: three GEMMs of ``2 * S_q * H_idx * 128 * topk`` (dQ, dK, dW), as benchmark_dsa_indexer_h32.py counts them."""
    return 6.0 * s_q * indexer_heads * INDEXER_HEAD_DIM * topk


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


def make_candidate_ids(num_requests, q_tokens, kv_len, phase, count, block, device, chunk_rows=1024):
    """Candidate lists for the hierarchical indexer consumer: ``(num_requests, q_tokens, count)`` int32 global ids.

    Row ``j`` of a request sits at ``pos`` (see :func:`kv_pool.row_positions`) and
    receives ``min(pos + 1, count)`` unique causal-valid positions: the newest
    ``block``-token block (the one holding ``pos``) is pinned, the rest are drawn
    at random from the older positions, and the remaining slots are ``-1``. Ids
    are ``request * kv_len + position`` (``topk_indices_global=True``). Also
    returns the valid count per row as ``(num_requests, q_tokens)`` int32.
    Generated ``chunk_rows`` rows at a time over the visible range only.
    """
    rows = int(num_requests) * int(q_tokens)
    pos_all = kv_pool.row_positions(q_tokens, kv_len, phase).repeat(int(num_requests)).to(device)
    req_all = torch.arange(rows, device=device, dtype=torch.int64) // int(q_tokens)
    n_valid = torch.clamp(pos_all + 1, max=int(count))
    ids = torch.full((rows, int(count)), -1, dtype=torch.int32, device=device)
    start = 0
    while start < rows:
        max_vis = int(pos_all[start : start + chunk_rows].max()) + 1
        n = max(1, min(chunk_rows, rows - start, (1 << 25) // max_vis))
        pos = pos_all[start : start + n].view(n, 1)
        width = int(pos.max()) + 1
        col = torch.arange(width, device=device).view(1, width)
        scores = torch.rand(n, width, device=device)
        scores.masked_fill_(col > pos, 2.0)  # invisible positions sort last
        scores.masked_fill_((col >= pos - pos % block) & (col <= pos), -1.0)  # newest block sorts first: pinned
        take = min(width, int(count))
        picks = scores.argsort(dim=-1)[:, :take]
        picks.masked_fill_(col[:, :take] >= n_valid[start : start + n].view(n, 1), -1)
        flat = req_all[start : start + n].view(n, 1) * int(kv_len) + picks
        ids[start : start + n, :take] = torch.where(picks >= 0, flat, -1).to(torch.int32)
        start += n
    return ids.view(int(num_requests), int(q_tokens), int(count)), n_valid.to(torch.int32).view(int(num_requests), int(q_tokens))


@torch.no_grad()
def reference_forward(q, kv, attn_sink, topk_idxs, softmax_scale, d_v, chunk=128):
    """Chunked sparse attention forward -> (out, KV-only lse) for the backward kernel. ``-1`` ids are masked."""
    s_q, num_heads, _ = q.shape
    out = torch.empty(s_q, num_heads, d_v, dtype=q.dtype, device=q.device)
    lse = torch.empty(s_q, num_heads, dtype=torch.float32, device=q.device)
    for i in range(0, s_q, chunk):
        idx = topk_idxs[i : i + chunk].long()
        kv_g = kv[idx.clamp(min=0)].float()  # (C, K, D)
        scores = torch.einsum("chd,ckd->chk", q[i : i + chunk].float(), kv_g) * softmax_scale
        scores.masked_fill_((idx < 0).unsqueeze(1), float("-inf"))
        lse_c = torch.logsumexp(scores, dim=-1)  # KV-only, excludes the sink
        lse_with_sink = torch.logaddexp(lse_c, attn_sink.view(1, num_heads))
        p = torch.exp(scores - lse_with_sink.unsqueeze(-1))
        out[i : i + chunk] = torch.einsum("chk,ckv->chv", p, kv_g[..., :d_v]).to(q.dtype)
        lse[i : i + chunk] = lse_c
    return out, lse


def time_fn(fn, num_warmup, num_iters, pre_iter: Optional[Callable[[], None]] = None) -> float:
    """Median per-call GPU time of fn() in ms via CUDA events, L2 flushed before each call.

    ``pre_iter`` runs before every warmup and timed call, outside the events
    (restoring an input a kernel overwrites in place).
    """
    l2_flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    for _ in range(num_warmup):
        if pre_iter is not None:
            pre_iter()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        if pre_iter is not None:
            pre_iter()
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


def profile_once(fn, num_warmup, label, pre_iter: Optional[Callable[[], None]] = None) -> float:
    """Warm fn(), then time exactly one call inside the CUDA profiler range (for nsys / ncu) and return its ms."""
    l2_flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    for _ in range(num_warmup):
        if pre_iter is not None:
            pre_iter()
        fn()
    torch.cuda.synchronize()
    if pre_iter is not None:
        pre_iter()
    l2_flush.zero_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range(label):
            start.record()
            fn()
            end.record()
        end.synchronize()
    finally:
        torch.cuda.profiler.stop()
    return start.elapsed_time(end)


class Prepared:
    """A case ready to time: inputs allocated and plans validated (stage A), nothing compiled yet.

    ``probe()`` compiles and runs every kernel once (stage B); ``steps`` are
    the closures timed separately (``time_ms`` is their sum; one step for
    every row but the hierarchical indexer); ``run()`` executes all steps in
    order for warmup and profile mode; ``pre_iter`` restores in-place inputs
    before each call; ``detail`` is the RESULT detail prefix and ``flops`` the
    row's useful work; ``tensors`` names the preallocated inputs and outputs
    the steps read and write (for the GPU smoke's reference checks).
    """

    def __init__(
        self,
        steps: List[Callable[[], None]],
        probe: Callable[[], None],
        detail: str,
        flops: float,
        pre_iter: Optional[Callable[[], None]] = None,
        tensors: Optional[Dict[str, object]] = None,
    ):
        self.steps = steps
        self.probe = probe
        self.detail = detail
        self.flops = float(flops)
        self.pre_iter = pre_iter
        self.tensors = tensors or {}

    def run(self) -> None:
        for step in self.steps:
            step()


def _flat_or_paged_pool(args, num_requests: int, kv_len: int, dtype, device):
    """The KV pool and the id-list mapping arguments for a flat or (``--paged``) page-table pool."""
    if args.paged:
        paged = kv_pool.make_paged_pool(num_requests, kv_len, args.page_size, args.head_dim_qk, dtype, device)
        return paged.pool, dict(block_table=paged.block_table, page_size=args.page_size)
    pool = torch.randn(num_requests * kv_len, args.head_dim_qk, dtype=dtype, device=device).mul_(0.1)
    return pool, {}


def _core_inputs(args, phase: str, dtype, device):
    """q, kv pool, ids, topk_length, sink and scale for a core row (context: one request; generation: ``batch_size``).

    Context / bprop lists are full (every row gathers ``topk`` rows of the
    pool); generation lists are causal to the row's position inside its own
    request. Two contracts are checked here, before any plan is built: a row
    shorter than ``topk`` needs the length tensor (without it the kernels read
    every slot), and with ``indexer_topk > 0`` every row must cover the
    indexer prefix (``docs/fe-oss-apis/dsa.md``, "Indexer prefix constraint").
    A violation is a ValueError, i.e. a skipped row with the reason.
    """
    num_requests = args.batch_size if phase == "generation" else 1
    rows = num_requests * args.q_seqlen
    q = torch.randn(rows, args.num_q_heads, args.head_dim_qk, dtype=dtype, device=device).mul_(0.1)
    kv, mapping = _flat_or_paged_pool(args, num_requests, args.kv_seqlen, dtype, device)
    topk_idxs, topk_length = kv_pool.make_topk_ids(
        num_requests,
        args.q_seqlen,
        args.kv_seqlen,
        args.topk,
        phase=phase,
        causal=phase == "generation",
        index_granularity=args.index_granularity,
        device=device,
        **mapping,
    )
    shortest = int(topk_length.min())
    if shortest < args.topk and not args.use_topk_length:
        raise ValueError(
            f"a row with {shortest} < topk ({args.topk}) visible tokens needs --use_topk_length: without a length tensor the kernels read every slot"
        )
    if args.indexer_topk and shortest < args.indexer_topk:
        raise ValueError(f"every row must have topk_length >= indexer_topk ({args.indexer_topk}) for the indexer LSE; the shortest row has {shortest}")
    attn_sink = torch.linspace(-2.0, 2.0, args.num_q_heads, dtype=torch.float32, device=device) if args.has_sink else None
    topk_length = topk_length if args.use_topk_length else None
    softmax_scale = 1.0 / math.sqrt(args.head_dim_qk)
    return q, kv, topk_idxs, attn_sink, topk_length, softmax_scale


def _page_token(args) -> str:
    return f";page={args.page_size}" if args.paged else ""


def _core_forward(args, DSA, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale) -> Tuple[Callable, Callable, str, Dict[str, object]]:
    """Plan + preallocated outputs for SparseAttentionForward; returns (run, probe, legacy detail, tensors)."""
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

    rows, num_heads, _ = q.shape
    out = torch.empty((rows, num_heads, args.head_dim_vo), dtype=q.dtype, device=q.device)
    max_logits = torch.empty((rows, num_heads), dtype=torch.float32, device=q.device)
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

    def probe():
        op.compile()  # a lifecycle flag: the interface compiles on this first execute
        run()

    padded_topk = ((args.topk + 63) // 64) * 64
    detail = f"SparseAttentionForward.execute padded_K={padded_topk} indexer_topk={args.indexer_topk}"
    tensors = dict(
        q=q,
        kv=kv,
        topk_idxs=topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        out=out,
        max_logits=max_logits,
        lse=lse,
        lse_indexer=lse_indexer,
    )
    return run, probe, detail, tensors


def setup_core_context(args, case, DSA, device) -> Prepared:
    dtype = DTYPES[args.data_type]
    q, kv, topk_idxs, attn_sink, topk_length, softmax_scale = _core_inputs(args, "context", dtype, device)
    run, probe, detail, tensors = _core_forward(args, DSA, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale)
    gathered = gathered_ids(args.q_seqlen, args.topk, topk_length)
    flops = flops_fwd(args.q_seqlen, args.num_q_heads, args.topk, args.head_dim_qk, args.head_dim_vo, gathered=gathered)
    return Prepared([run], probe, detail + _page_token(args), flops, tensors=tensors)


def setup_core_generation(args, case, DSA, device) -> Prepared:
    dtype = DTYPES[args.data_type]
    q, kv, topk_idxs, attn_sink, topk_length, softmax_scale = _core_inputs(args, "generation", dtype, device)
    run, probe, detail, tensors = _core_forward(args, DSA, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale)
    rows = args.batch_size * args.q_seqlen
    flops = flops_fwd(rows, args.num_q_heads, args.topk, args.head_dim_qk, args.head_dim_vo, gathered=gathered_ids(rows, args.topk, topk_length))
    record = support.record_bytes(args.head_dim_qk, args.kv_dtype, args.kv_layout, args.num_kv_heads)
    # KV records gathered + the int32 id lists read + Q read and O written (2-byte elements).
    bytes_gathered = rows * args.topk * record + rows * args.topk * 4 + rows * args.num_q_heads * (args.head_dim_qk + args.head_dim_vo) * 2
    ctas = f";CTAs={rows}" if args.num_q_heads == 64 else ""  # the H128 variant runs persistent 2-CTA clusters instead
    detail += f";flat-row prefill kernel;rows={rows}{ctas};no split-KV;bytes_gathered={bytes_gathered}" + _page_token(args)
    return Prepared([run], probe, detail, flops, tensors=tensors)


def setup_core_bprop(args, case, DSA, device) -> Prepared:
    dtype = DTYPES[args.data_type]
    q, kv, topk_idxs, attn_sink, topk_length, softmax_scale = _core_inputs(args, "context", dtype, device)
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

    def probe():
        op.compile()  # a lifecycle flag: the backend primes on this first execute
        run()

    detail = f"SparseAttentionBackward.execute workspace_bytes={workspace_bytes} deterministic={args.deterministic}" + _page_token(args)
    gathered = gathered_ids(args.q_seqlen, args.topk, topk_length)
    flops = flops_bwd(args.q_seqlen, args.num_q_heads, args.topk, args.head_dim_qk, args.head_dim_vo, gathered=gathered)
    tensors = dict(q=q, kv=kv, topk_idxs=topk_idxs, topk_length=topk_length, out=out, dout=dout, lse=lse, attn_sink=attn_sink, dq=dq, dkv=dkv, d_sink=d_sink)
    return Prepared([run], probe, detail, flops, tensors=tensors)


def _indexer_geometry(args, phase: str):
    """(requests, q rows per request, cached tokens per request, index keys per request) of an indexer row.

    The divisibility check below (and the pool-ratio check of the hierarchical
    setup) are the kernels' argument contracts on a preset's own fields, not
    enumerated suite axes, so they stay next to the construction they guard;
    no shipped preset reaches them.
    """
    num_requests = args.batch_size if phase == "generation" else 1
    ctx = args.kv_seqlen
    if ctx % args.indexer_pool_ratio:
        raise ValueError(f"kv_seqlen ({ctx}) must be a multiple of indexer_pool_ratio ({args.indexer_pool_ratio})")
    return num_requests, args.q_seqlen, ctx, ctx // args.indexer_pool_ratio


def _e8m0_ones(shape, device) -> torch.Tensor:
    """Unit E8M0 block scales (values are irrelevant for timing)."""
    return torch.ones(shape, dtype=torch.float32, device=device).to(torch.float8_e8m0fnu)


def setup_indexer_dense(args, case, DSA, device, phase: str) -> Prepared:
    """Fused indexer scoring + top-k (context and generation), bf16 or mxfp8 keys."""
    dtype = DTYPES[args.data_type]
    heads, ratio = args.indexer_heads, args.indexer_pool_ratio
    top_k = support.indexer_topk_select(case)
    num_requests, s_q, ctx, s_k = _indexer_geometry(args, phase)
    q = torch.randn(num_requests, s_q, heads, INDEXER_HEAD_DIM, dtype=dtype, device=device).mul_(0.1)
    k = torch.randn(num_requests, s_k, 1, INDEXER_HEAD_DIM, dtype=dtype, device=device).mul_(0.1)
    w = torch.randn(num_requests, s_q, heads, dtype=dtype, device=device).mul_(0.1)
    # Generation rows are the last s_q positions of each request's ctx-token cache.
    q_causal_offsets = torch.full((num_requests,), ctx - s_q, dtype=torch.int32, device=device) if phase == "generation" else None

    precision, q_scale, k_scale, microbatch_rows = "bf16", None, None, -1
    if args.kv_dtype == "mxfp8":
        # The MXFP8 indexer takes e4m3 Q AND K with packed e8m0 scales per 32 lanes (w stays bf16); there is
        # no bf16-Q x mxfp8-K form. Packed once here; mxfp8 never microbatches, so size and launch with 0.
        from cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils import pack_qk_scale_bshd

        precision, microbatch_rows = "mxfp8", 0
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        groups = INDEXER_HEAD_DIM // MXFP8_SF_VEC_SIZE
        q_scale, k_scale = pack_qk_scale_bshd(_e8m0_ones((num_requests, s_q, heads, groups), device), _e8m0_ones((num_requests, s_k, 1, groups), device), heads)
        torch.cuda.synchronize()

    # Caller-owned buffers so the timed call allocates nothing and reads nothing back to the host
    # (the size query below does the one .item() under offsets, here in setup).
    cand_floats = DSA.compress_topk_cand_buffer_size(num_requests, s_q, s_k, ratio, microbatch_rows=microbatch_rows, q_causal_offsets=q_causal_offsets)
    cand_buffer = torch.empty(cand_floats, dtype=torch.float32, device=device)
    out_indices = torch.empty((num_requests, s_q, top_k), dtype=torch.int32, device=device)
    out_logits = torch.empty((num_requests, s_q, top_k), dtype=torch.float32, device=device)

    def run():
        DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k,
            ratio=ratio,
            qhead_per_kv_head=heads,
            max_seqlen_q=s_q,
            max_seqlen_k=s_k,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
            sf_vec_size=MXFP8_SF_VEC_SIZE,
            return_softmax=False,
            microbatch_rows=microbatch_rows,
            topk_indices_global=True,
            cand_buffer=cand_buffer,
            out_indices=out_indices,
            out_logits=out_logits,
        )

    visible = num_requests * int(kv_pool.causal_visible_counts(s_q, ctx, phase, ratio).sum())
    flops = flops_indexer_logits(heads, visible)
    detail = f"indexer_forward_top_k_wrapper precision={precision} top_k={top_k} ratio={ratio} heads={heads};fused"
    if phase == "generation":
        key_bytes = int(round(support.index_key_bytes(args.kv_dtype, INDEXER_HEAD_DIM, 1)))
        detail += f";rows={num_requests * s_q};bytes_gathered={num_requests * s_k * key_bytes}"  # the key scan of every request's pool
    tensors = dict(
        q=q,
        k=k,
        w=w,
        q_scale=q_scale,
        k_scale=k_scale,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        microbatch_rows=microbatch_rows,
        out_indices=out_indices,
        out_logits=out_logits,
    )
    return Prepared([run], run, detail, flops, tensors=tensors)


def setup_indexer_hierarchical(args, case, DSA, device, phase: str) -> Prepared:
    """DSv4.1 hierarchical consumer: candidate-restricted scoring over CANDIDATE_COUNT ids, then IndexerTopK."""
    dtype = DTYPES[args.data_type]
    heads = args.indexer_heads
    top_k = support.indexer_topk_select(case)
    count, block = support.CANDIDATE_COUNT, support.CANDIDATE_BLOCK
    if args.indexer_pool_ratio != 1:
        raise ValueError(f"the candidate-restricted indexer scoring kernel takes unpooled keys (indexer_pool_ratio 1, got {args.indexer_pool_ratio})")
    num_requests, s_q, ctx, s_k = _indexer_geometry(args, phase)
    rows = num_requests * s_q
    q = torch.randn(num_requests, s_q, heads, INDEXER_HEAD_DIM, dtype=dtype, device=device).mul_(0.1)
    k = torch.randn(num_requests, s_k, INDEXER_HEAD_DIM, dtype=dtype, device=device).mul_(0.1)
    w = torch.randn(num_requests, s_q, heads, dtype=dtype, device=device).mul_(0.1)
    candidates, n_valid = make_candidate_ids(num_requests, s_q, ctx, phase, count, block, device)
    # Zero-filled once in setup: the recompute kernel is only guaranteed to write a row's valid prefix, and
    # IndexerTopK below scans all `count` columns, so the padded slots must hold a defined value.
    scores = torch.zeros((num_requests, s_q, count), dtype=torch.float32, device=device)

    recompute = DSA.SparseIndexerScoreRecompute(q, k, w, candidates, scores, sample_topk_length=n_valid, qhead_per_kv_head=heads, topk_indices_global=True)
    recompute.check_support()

    scores_2d = scores.view(rows, count)
    seq_lens = torch.full((rows,), count, dtype=torch.int32, device=device)  # the consumer selects over the full candidate width
    select = DSA.IndexerTopK(scores_2d, seq_lens, top_k=top_k, next_n=1, return_val=False)
    select.check_support()
    workspace = torch.empty(select.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
    selected = torch.empty((rows, top_k), dtype=torch.int32, device=device)

    def step_recompute():
        recompute.execute(q, k, w, candidates, scores, topk_length=n_valid)

    def step_topk():
        select.execute(scores_2d, seq_lens, out_indices=selected, workspace=workspace)

    def probe():
        recompute.compile()  # a lifecycle flag: the backend compiles on this first execute
        step_recompute()
        select.compile()
        step_topk()

    # Useful work: every row scores its valid candidates (count, or pos + 1 for short rows).
    flops = flops_indexer_logits(heads, int(n_valid.sum()))
    detail = f"SparseIndexerScoreRecompute+IndexerTopK candidates={count} block={block} top_k={top_k} heads={heads};consumer"
    tensors = dict(q=q, k=k, w=w, candidates=candidates, n_valid=n_valid, scores=scores_2d, seq_lens=seq_lens, selected=selected)
    return Prepared([step_recompute, step_topk], probe, detail, flops, tensors=tensors)


def setup_indexer_forward(args, case, DSA, device, phase: str) -> Prepared:
    if args.indexer_mode == "hierarchical":
        return setup_indexer_hierarchical(args, case, DSA, device, phase)
    return setup_indexer_dense(args, case, DSA, device, phase)


def setup_indexer_bprop(args, case, DSA, device) -> Prepared:
    """IndexerBackward on the default backend with random targets; attn_score restored before every call."""
    dtype = DTYPES[args.data_type]
    heads = args.indexer_heads
    top_k = support.indexer_topk_select(case)
    _, s_q, _, s_k = _indexer_geometry(args, "context")
    index_q = torch.randn(1, s_q, heads, INDEXER_HEAD_DIM, dtype=dtype, device=device).mul_(0.1)
    weights = torch.randn(1, s_q, heads, dtype=dtype, device=device).mul_(0.1)
    index_k = torch.randn(1, s_k, INDEXER_HEAD_DIM, dtype=dtype, device=device).mul_(0.1)
    attn_pristine = torch.softmax(torch.randn(1, s_q, top_k, dtype=torch.float32, device=device), dim=-1)  # target distribution
    attn_score = attn_pristine.clone()  # kernel 1 overwrites it with the score gradient
    index_score = torch.softmax(torch.randn(1, s_q, top_k, dtype=torch.float32, device=device), dim=-1)  # predict distribution
    topk_indices = torch.randint(0, s_k, (1, s_q, top_k), dtype=torch.int32, device=device)  # global == local for one request
    grad_loss = torch.ones((), dtype=torch.float32, device=device)
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k)

    op = DSA.IndexerBackward(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        index_score,
        topk_indices,
        sm_scale=1.0,
        block_I=128,
        topk_indices_global=True,
        backend="default",
    )
    op.check_support()
    workspace_bytes = op.scratch_workspace_bytes()
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device) if workspace_bytes else None

    def run():
        op.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn_score,
            index_score,
            topk_indices,
            grad_loss,
            loss_coeff=float(s_q),
            workspace=workspace,
        )

    def restore():
        attn_score.copy_(attn_pristine)

    def probe():
        op.compile()
        run()

    flops = flops_indexer_bwd(s_q, heads, top_k)
    detail = f"IndexerBackward.execute workspace_bytes={workspace_bytes} backend=default top_k={top_k} heads={heads};attn_score restore excluded"
    tensors = dict(
        op=op,
        index_q=index_q,
        weights=weights,
        index_k=index_k,
        attn_score=attn_score,
        attn_pristine=attn_pristine,
        index_score=index_score,
        topk_indices=topk_indices,
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k,
    )
    return Prepared([run], probe, detail, flops, pre_iter=restore, tensors=tensors)


SETUPS: Dict[Tuple[str, str], Callable] = {
    ("core", "context"): setup_core_context,
    ("core", "generation"): setup_core_generation,
    ("core", "bprop"): setup_core_bprop,
    ("indexer", "context"): lambda args, case, DSA, device: setup_indexer_forward(args, case, DSA, device, "context"),
    ("indexer", "generation"): lambda args, case, DSA, device: setup_indexer_forward(args, case, DSA, device, "generation"),
    ("indexer", "bprop"): setup_indexer_bprop,
}


def unsupported(reason: str):
    print(f"UNSUPPORTED,{' '.join(str(reason).split()).replace(',', ';')}")
    sys.exit(UNSUPPORTED_EXIT_CODE)


def case_from_args(args) -> dict:
    """The flat case dict support.decide_support / support.static_columns read."""
    return dict(
        component=args.component,
        phase=args.phase,
        profile_pass=args.profile_pass,
        kv_dtype=args.kv_dtype,
        paged=args.paged,
        batch_size=args.batch_size,
        indexer_mode=args.indexer_mode if args.component == "indexer" else None,
        data_type=args.data_type,
        q_seqlen=args.q_seqlen,
        kv_seqlen=args.kv_seqlen,
        num_q_heads=args.num_q_heads,
        head_dim_qk=args.head_dim_qk,
        head_dim_vo=args.head_dim_vo,
        topk=args.topk,
        indexer_topk=args.indexer_topk,
        num_kv_heads=args.num_kv_heads,
        kv_layout=args.kv_layout,
        index_granularity=args.index_granularity,
        indexer_heads=args.indexer_heads,
        indexer_pool_ratio=args.indexer_pool_ratio,
        mtp_rows=args.mtp_rows,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile_pass", required=True, choices=["fwd", "bwd"])
    p.add_argument("--backend", default="cudnn", choices=["cudnn"])
    p.add_argument("--q_seqlen", type=int, required=True, help="query rows per request")
    p.add_argument("--kv_seqlen", type=int, required=True, help="KV pool rows per request")
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
    # Suite axes (defaults reproduce the legacy core context / bprop rows).
    p.add_argument("--component", default="core", choices=["core", "indexer"])
    p.add_argument("--phase", default=None, choices=list(support.PHASES), help="default: context for --profile_pass fwd, bprop for bwd")
    p.add_argument("--kv_dtype", default="bfloat16", choices=list(KV_DTYPES), help="storage format of the KV record / index keys")
    p.add_argument("--paged", action="store_true", help="page the KV pool through a block table")
    p.add_argument("--page_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1, help="requests per generation case")
    # Preset geometry the indexer rows and the support rules read.
    p.add_argument("--num_kv_heads", type=int, default=1)
    p.add_argument("--kv_layout", default="shared", choices=["shared", "separate"])
    p.add_argument("--index_granularity", type=int, default=1, help="KV rows per index entry (4 = four-token blocks)")
    p.add_argument("--indexer_heads", type=int, default=0, help="lightning-indexer query heads (0 = no indexer rows)")
    p.add_argument("--indexer_pool_ratio", type=int, default=1, help="raw tokens per index key (the indexer kernels' ratio)")
    p.add_argument("--indexer_mode", default="dense", choices=["dense", "hierarchical"])
    p.add_argument("--mtp_rows", type=int, default=1)
    p.add_argument(
        "--profile", action="store_true", help="one warmed call inside torch.cuda.profiler start/stop + NVTX range; RESULT carries that single timing"
    )
    args = p.parse_args(argv)
    if args.phase is None:
        args.phase = "context" if args.profile_pass == "fwd" else "bprop"
    elif support.PROFILE_PASS_FOR_PHASE[args.phase] != args.profile_pass:
        p.error(f"--phase {args.phase} goes with --profile_pass {support.PROFILE_PASS_FOR_PHASE[args.phase]}")
    if args.batch_size < 1 or args.page_size < 1:
        p.error("--batch_size and --page_size must be positive")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    case = case_from_args(args)
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    reason = support.decide_support(case, cc, support.dsl_target_error(cc) if cc == (10, 7) else None)
    if reason:
        unsupported(reason)
    if not torch.cuda.is_available():
        unsupported("CUDA not available")
    try:
        import cudnn
        from cudnn import DSA

        _ = DSA.SparseAttentionForward
    except (ImportError, AttributeError) as e:
        unsupported(f"cudnn[cutedsl] not available: {e}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Stage A: inputs, plans and the kernels' own check_support. Their shape / arch / dtype
    # declines are ValueError / RuntimeError / TypeError (and a few asserts) -> skipped row.
    # Running out of device memory while building the inputs is a RuntimeError subclass but
    # not a decline: it propagates as an honest failure.
    try:
        prepared = SETUPS[(args.component, args.phase)](args, case, DSA, device)
    except torch.cuda.OutOfMemoryError:
        raise
    except _STAGE_A_UNSUPPORTED as e:
        unsupported(str(e) or type(e).__name__)  # a bare kernel assert has no text
    # Stage B: one untimed probe call, where the CuTe compile happens. Validation types
    # still map to a skip; a RuntimeError from compile or launch propagates as a failure.
    try:
        prepared.probe()
        torch.cuda.synchronize()
    except _STAGE_B_UNSUPPORTED as e:
        unsupported(str(e) or type(e).__name__)

    detail = prepared.detail
    if args.profile:
        ms = profile_once(prepared.run, args.num_warmup_iterations, f"dsa_{args.component}_{args.phase}", prepared.pre_iter)
        detail += ";profile=single"
    else:
        step_ms = [time_fn(step, args.num_warmup_iterations, args.num_iterations, prepared.pre_iter) for step in prepared.steps]
        ms = sum(step_ms)
        # Per-step tokens the runner lifts into time_step1_ms / time_step2_ms; the legacy core
        # rows keep their detail text unchanged.
        if args.component == "indexer" or len(step_ms) > 1:
            detail += "".join(f";time_step{i + 1}_ms={t:.6f}" for i, t in enumerate(step_ms))
    sm_mhz = sample_sm_clock_mhz()

    tflops = prepared.flops / (ms * 1e-3) / 1e12 if ms > 0 else 0.0
    peak_dtype = args.kv_dtype if args.component == "indexer" else args.data_type  # the MMA dtype the row issues
    peak = peak_mma_tflops(peak_dtype, sm_mhz)
    peak_str = f"{peak:.3f}" if peak else ""
    gpu = torch.cuda.get_device_name().replace(",", ";")
    print(f"RESULT,{ms:.6f},{tflops:.3f},{peak_str},{gpu},{cudnn.__version__},{detail.replace(',', ';')}")


if __name__ == "__main__":
    main()
