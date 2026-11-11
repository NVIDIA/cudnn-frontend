# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SSA-contract wrapper around the vendored GVR self-sampling top-k.

GVR selects the exact top-K columns of a materialized score row but emits
raw (unordered) indices over the WHOLE row and no predict. This adapts it to
the SSA indexer contract (per-row [start, end) bounds, GLOBAL ascending ids,
clamp(end-1, 0) padding, predict = softmax over the selected scores):
  - GVR's native varlen mode scans [start[r], end[r]) per row with no -inf
    mask; row_starts offsets the DRAM base and kv_lens supplies the valid
    length. A fused epilogue kernel (postproc.py) converts local positions to
    global ascending ids and computes predict. This covers both starts == 0
    causal-tail rows and starts != 0 packed/two-document rows.
"""

import torch

from . import gvr_host, gvr_kernel

_ws_cache = {}


def gvr_topk_prefill(
    score,  # [rows, num_cols] fp32, materialized S (row-major)
    row_starts,  # [rows] int32
    row_ends,  # [rows] int32
    top_k,
    *,
    logical_num_cols,
    out_indices,  # [rows, top_k] int32  -- GLOBAL ascending ids
    out_probs,  # [rows, top_k] fp32   -- predict softmax
):
    rows, ncols = score.shape
    dev = score.device
    assert ncols == logical_num_cols

    # GVR is hint-free (TRT-LLM PR#18410): the threshold bracket comes from
    # the current row's self-sample, so prefill (which has no prior top-k)
    # needs no temporal hint tensor at all.
    raw = torch.empty(rows, top_k, dtype=torch.int32, device=dev)

    # ---- native per-row bounds via GVR's varlen mode: row r scans
    # [start_r, end_r) with NO -inf mask (x_starts offsets the DRAM base per
    # row; kv_lens = the valid length). The -inf-copy adapter both cost a
    # full 8GB S pass AND stalled GVR's sampling ladder on the blanked
    # columns (measured 2.4 -> 29 ms); this avoids both, and handles
    # packed-THD starts != 0. ----
    starts = row_starts.contiguous()
    ends = row_ends.contiguous()
    lengths = (ends - starts).clamp_min(0).to(torch.int32)
    gvr_host.run_varlen(
        score,
        lengths,
        raw,
        next_n=1,
        compress_ratio=1,
        max_seq_len=ncols,
        row_starts=starts,
    )
    # GVR emits column ids LOCAL to [start_r, end_r); add the per-row start to
    # get GLOBAL ids (pads stay -1). No-op when starts == 0 (causal-tail).
    raw = torch.where(raw >= 0, raw + starts.view(rows, 1), raw)

    # ---- fused SSA epilogue: gather scores, ascending bitmap rank-scatter,
    # clamp(end-1, 0) padding, predict softmax (one kernel; replaces the torch
    # sort+gather+softmax that cost 3.5 ms at 256K/K=2048). GLOBAL ids in,
    # handles per-row [start, end) bounds via mStarts/mEnds. ----
    from .postproc import gvr_sort_predict

    gvr_sort_predict(score, raw.contiguous(), starts, ends, out_indices, out_probs)


def gvr_compact_select(
    candidx,  # [rows, tcap] int32 -- GLOBAL col id of each collected cand
    candval,  # [rows, tcap] fp32  -- its score (row stride must be %4)
    counts,  # [rows] int32       -- per-row #collected (cursor count)
    row_starts,  # [rows] int32
    row_ends,  # [rows] int32
    top_k,
    *,
    out_indices,  # [rows, top_k] int32  -- GLOBAL ascending ids
    out_probs,  # [rows, top_k] fp32   -- predict softmax
    flags=None,  # [rows+1] int32 -- count, then compact bad row ids
    tie_cap=None,  # logical candidate cap for the certificate (native's
    # _tie_cap); defaults to candval's physical width
    workspace=None,  # optional persistent buffers + per-stream GVR slab
    fallback_pred=None,  # bool device predicates for sparse repair graph
):
    """Select the top-K of the filtered-fusion COMPACT candidate list, reusing
    GVR's exact selection + the sort/predict epilogue instead of a bespoke
    select kernel. GVR picks the top-K POSITIONS (+ values) of candval[r,
    0:counts[r]]; positions are remapped to GLOBAL col ids via candidx, then the
    shared gvr_sort_predict (value-provided path -- S is never materialized)
    produces ascending ids + predict. Exactness is the same cursor certificate
    the native compact select uses."""
    from .postproc import gvr_sort_predict

    rows, tcap = candval.shape
    dev = candval.device
    counts = counts.to(torch.int32).contiguous()
    # GVR can only select from the tcap physical slots; the RAW cursor count
    # (possibly > tcap on overflow) is kept for the certificate below.
    if workspace is None:
        gvr_counts = counts.clamp_max(tcap).contiguous()
        positions = torch.empty(rows, top_k, dtype=torch.int32, device=dev)
        sel_vals = torch.empty(rows, top_k, dtype=torch.float32, device=dev)
    else:
        gvr_counts = workspace["gvr_counts"]
        torch.clamp(counts, max=tcap, out=gvr_counts)
        positions = workspace["positions"]
        sel_vals = workspace["sel_vals"]
    # ---- GVR selects top-K positions of candval[r, 0:count] + their values.
    # kv_lens = per-row candidate count (next_n=1, cr=1 => n_r = counts[r]). ----
    gvr_host.run_varlen(
        candval,
        gvr_counts,
        positions,
        next_n=1,
        compress_ratio=1,
        values=sel_vals,
        max_seq_len=tcap,
        workspace=(None if workspace is None else workspace.get("gvr_workspace")),
    )
    # ---- one fused kernel: remap positions -> GLOBAL col id (candidx gather,
    # ON-DEVICE), ascending-by-id rank-scatter, predict (values provided, no S),
    # and the cursor certificate (raw counts + tcap) -- no host glue. ----
    if flags is None:
        flags = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
    gvr_sort_predict(
        candval,
        positions,
        row_starts.contiguous(),
        row_ends.contiguous(),
        out_indices,
        out_probs,
        values=sel_vals,
        candidx=candidx,
        counts=counts,
        flags=flags,
        fallback_pred=fallback_pred,
        cert_cap=(tcap if tie_cap is None else int(tie_cap)),
    )
