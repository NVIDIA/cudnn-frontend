# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_topk_decode``: the block-sparse indexer's decode top-k (one thread block per row, the row in registers, one wide
histogram pass plus the narrowing passes over the k-th bin's candidates, the list in canonical order).

Every accept cell compares the kernel BITWISE against an exact canonical reference built from the contract alone (``torch.sort``
of the visible logits, descending and stable -- so a tie keeps the smaller id -- the first ``min(top_k, counts)`` ids re-sorted
descending, ``-1`` past them), checks the SET against ``torch.topk`` of the fp64 copy of the same values (the sharp oracle up to
exact ties at the k-th value), and launches twice (bitwise).  The radix top-k kernel the standalone decode selector binds today
is a second oracle for the set on a cc 9.0+ device.  Every knob value (the thread count, the PDL pair) is the same function and is
pinned bitwise against the others.

The CuTe DSL floor gates the kernel-module import (Rule 7): below it every cell of this module skips with the version named;
with the floor met the version decline itself is pinned through a modeled sub-floor state.  The PDL arm and the row-split cluster
form (``cluster_ctas`` 4 / 8 / 16: the histogram merged over a thread-block cluster by ``st.async`` + mbarrier) need a cc 9.0+ device
and skip elsewhere; the single-CTA form runs on any CuTe DSL device, so the host tier covers the select itself.
"""

import hashlib
import importlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest
import torch

import cudnn.frost.compiled_cache as cc
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

_CUTEDSL_REQUIREMENT = cutedsl_requirement_error("qsa_topk_decode")  # None at or above the floor
pytestmark = [pytest.mark.L0, pytest.mark.skipif(_CUTEDSL_REQUIREMENT is not None, reason=str(_CUTEDSL_REQUIREMENT))]

# the kernel module imports the DSL at import time: imported only once the floor is met (every cell skips above otherwise)
topk_mod = importlib.import_module("cudnn.gated_attention_block.kernels.qsa_topk_decode") if _CUTEDSL_REQUIREMENT is None else None


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is needed")
needs_pdl_gpu = pytest.mark.skipif(_cc() is None or _cc() < (9, 0), reason=f"griddepcontrol needs cc 9.0+; found {_cc()}")
needs_cluster_gpu = pytest.mark.skipif(_cc() is None or _cc() < (9, 0), reason=f"thread-block clusters and DSMEM need cc 9.0+; found {_cc()}")
_TARGET = cutedsl_arch_requirement_error(_cc()) if _cc() is not None else None
needs_dsl_target = pytest.mark.skipif(_TARGET is not None, reason=str(_TARGET))

K = 512
SENTINEL = -2  # never produced by the kernel (ids are >= 0, the padding is -1): a slot still holding it after a launch was never written
THREADS = (512, 1024)


def _cluster_sizes():
    cc = _cc()
    return (1,) if cc is None or cc < (9, 0) else (1, 4, 8, 16)


def reference_canonical(logits: torch.Tensor, counts: torch.Tensor, top_k: int) -> torch.Tensor:
    """The contract: per row the exact top-``min(top_k, n)`` of the visible fp32 logits (ties -> the smaller ids), ids DESCENDING,
    ``-1`` past them; ``n = clamp(counts, 0, n_blocks_max)``."""
    rows, cols = logits.shape
    out = torch.full((rows, top_k), -1, dtype=torch.int32, device=logits.device)
    for t in range(rows):
        n = int(min(max(int(counts[t]), 0), cols))
        if n == 0:
            continue
        order = torch.sort(logits[t, :n], descending=True, stable=True).indices  # stable: equal values keep ascending ids
        sel = order[: min(top_k, n)]
        out[t, : sel.numel()] = torch.sort(sel, descending=True).values.to(torch.int32)
    return out


def assert_set_matches_fp64_topk(logits: torch.Tensor, counts: torch.Tensor, out: torch.Tensor) -> None:
    """The brief's oracle: per row the selected VALUES equal the fp64 ``torch.topk`` values exactly (fp32 -> fp64 is exact), and
    every id differing from ``torch.topk``'s choice carries exactly the k-th value (an exact tie)."""
    rows, cols = logits.shape
    top_k = out.shape[1]
    for t in range(rows):
        n = int(min(max(int(counts[t]), 0), cols))
        k_eff = min(top_k, n)
        ids = out[t]
        valid = ids >= 0
        assert int(valid.sum()) == k_eff, f"row {t}: {int(valid.sum())} valid ids, expected {k_eff}"
        assert bool((ids[k_eff:] == -1).all()), f"row {t}: the padding is not -1"
        if k_eff == 0:
            continue
        sel = ids[:k_eff].long()
        assert bool((sel < n).all()) and sel.unique().numel() == k_eff, f"row {t}: an id past the visible count or a duplicate"
        ref = torch.topk(logits[t, :n].double(), k_eff)
        got_vals = torch.sort(logits[t, sel].double(), descending=True).values
        assert torch.equal(got_vals, ref.values), f"row {t}: the selected values differ from the fp64 top-k"
        kth = ref.values[-1]
        diff = set(sel.tolist()) ^ set(ref.indices.tolist())
        for i in diff:
            assert float(logits[t, i].double()) == float(kth), f"row {t}: id {i} differs from torch.topk beyond an exact tie"


def _case(rows: int, cols: int, top_k: int, *, seed: int, device, kind: str = "randn"):
    """A row block whose counts cover: the full row, a partial row, a dead row, the identity boundary (``k``), one past it (``k + 1``),
    a single visible block; the columns past ``counts`` hold NaN or huge garbage (loaded by the kernel, never used)."""
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(rows, cols, device=device, dtype=torch.float32, generator=g) * 3.0
    if kind == "neg":
        logits = -logits.abs() - 0.5
    elif kind == "ties":
        # row 0 all equal, row 1 two values (every third id high), the rest randn
        logits[0, :] = 1.25
        logits[1, :] = torch.where(torch.arange(cols, device=device) % 3 == 0, torch.tensor(2.0, device=device), torch.tensor(-1.0, device=device))
    elif kind == "boundary_tie":
        # exactly top_k + 1 maxima at the smallest-id positions, above every random value -> one of them is dropped (the largest id)
        m = min(top_k + 1, cols)
        logits[:, :m] = 1.0e4
    elif kind == "wide":
        # many octaves between the maximum and the k-th value: the k-th lands in the window's saturated bin 0 (the slow exact path)
        logits = torch.exp(torch.randn(rows, cols, device=device, generator=g) * 6.0)
    elif kind == "relu":
        # the scorer's functional form: relu over 4 heads, summed, scaled
        logits = torch.relu(torch.randn(rows, cols, 4, device=device, generator=g)).sum(-1) / math.sqrt(128.0)
    elif kind == "recency":
        # the top-k = the LAST top_k ids (distinct, increasing): every staged word of the fast path falls into the one or two warps
        # that own the row's tail -- the per-warp region bound (top_k + RANK_CAP words) at its worst case
        lo_id = max(cols - top_k, 0)
        logits[:, lo_id:] = 50.0 + torch.arange(cols - lo_id, device=device, dtype=torch.float32) * 0.01
    elif kind == "whole_wide":
        # exactly top_k identical maxima spread over the row: the k-th bin is taken WHOLE with top_k > RANK_CAP candidates in it (the
        # fast path stages every one of them; the first RANK_CAP are rank-list entries as well, the rest plain ids)
        ids = torch.arange(top_k, device=device) * max(cols // top_k, 1)
        logits[:, ids[ids < cols]] = 77.0
    elif kind == "spread_tie":
        # n_above distinct values strictly above a tie value T = 100.0 (octaves above it), n_tie copies of T spread EVENLY over the
        # ids (every slice of a 4- / 8- / 16-CTA cluster holds a few of them, under UNION_CAP), the rest randn far below T:
        # n_above < top_k < n_above + n_tie -> the k-th key is EXACT with a PARTIAL take (the top_k - n_above smallest tied ids) and
        # the k-th bin holds n_tie > RANK_CAP candidates: the single-CTA list passes + the 2 x 12-bit id select; the cluster form's
        # union-list passes + the id rank behind its R2 barrier (the one path no other kind reaches)
        n_above = (3 * top_k) // 4
        n_tie = max(getattr(topk_mod, "RANK_CAP", 128), top_k - n_above) + 72
        tie_ids = torch.arange(n_tie, device=device) * max(2 * (cols // (2 * n_tie)), 2)  # even ids
        above_ids = torch.arange(n_above, device=device) * max(2 * (cols // (2 * n_above)), 2) + 1  # odd ids: disjoint from the ties
        tie_ids, above_ids = tie_ids[tie_ids < cols], above_ids[above_ids < cols]
        logits[:, tie_ids] = 100.0
        logits[:, above_ids] = 1000.0 + torch.arange(above_ids.numel(), device=device, dtype=torch.float32) * 0.5
    base = [cols, max(cols // 2 + 3, 1), 0, min(cols, top_k), min(cols, top_k + 1), 1]
    counts = torch.tensor([base[i % len(base)] for i in range(rows)], dtype=torch.int32, device=device)
    for t in range(rows):
        n = int(counts[t])
        if n < cols:
            logits[t, n:] = float("nan") if t % 2 else 1e30
    return logits.contiguous(), counts


def _run(logits, counts, top_k, **kw):
    out = torch.full((logits.shape[0], top_k), SENTINEL, dtype=torch.int32, device=logits.device)
    topk_mod.qsa_topk_decode(logits, counts, out, **kw)
    return out


def _check(logits, counts, top_k, **kw):
    out = _run(logits, counts, top_k, **kw)
    out2 = _run(logits, counts, top_k, **kw)
    torch.cuda.synchronize()
    assert torch.equal(out, out2), "two launches differ"
    assert bool((out != SENTINEL).all()), "a slot was never written (the sentinel survives)"
    ref = reference_canonical(logits, counts, top_k)
    if not torch.equal(out, ref):
        bad = (out != ref).nonzero()
        t, j = (int(x) for x in bad[0])
        raise AssertionError(
            f"{bad.shape[0]} slots differ from the canonical reference; first at row {t} slot {j}: got {int(out[t, j])}, want {int(ref[t, j])} (counts {int(counts[t])})"
        )
    assert_set_matches_fp64_topk(logits, counts, out)
    return out


# --------------------------------------------------------------------------------------------- accept cells


@needs_cuda
@needs_dsl_target
@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("cols", [8, 520, 2048, 8192, 32768])
def test_matches_the_exact_canonical_reference(cols, threads):
    """Six rows per case (full / partial / dead / k / k + 1 / one block visible; NaN or 1e30 past the counts) at every row width
    of the serving cells (8K / 32K / 128K tokens = 2048 / 8192 / 32768 blocks) and two small widths: bitwise the canonical reference,
    the set the fp64 top-k, two launches bitwise."""
    device = torch.device("cuda")
    logits, counts = _case(6, cols, K, seed=cols + threads, device=device)
    _check(logits, counts, K, threads_per_cta=threads)


@needs_cuda
@needs_dsl_target
@pytest.mark.parametrize("kind", ["ties", "boundary_tie", "neg", "wide", "relu", "spread_tie", "recency", "whole_wide"])
def test_ties_wide_ranges_negative_and_scorer_shaped_logits(kind):
    """A row of identical logits (more candidates in the k-th bin than the candidate list holds: the row-sweep fallback), a
    two-valued row, exactly ``k + 1`` equal maxima (the largest of them is the one dropped), all-negative rows, a value range of
    many octaves (the k-th value in the window's saturated bin: three more passes), the scorer's own relu-sum shape and a spread
    tie at the k-th key with a partial take (the list passes, then the 2 x 12-bit id select), the top-k concentrated in the row's
    last ids (one warp's staging region at its bound) and exactly k identical maxima (the k-th bin taken whole, wider than the rank
    list): the k-th-value ties resolve to the smaller ids, every form bitwise the reference."""
    device = torch.device("cuda")
    for threads in THREADS:
        logits, counts = _case(6, 8192, K, seed=11, device=device, kind=kind)
        out = _check(logits, counts, K, threads_per_cta=threads)
        if kind == "ties":
            # row 0: every visible block ties (8192 > CAND_CAP = 4096 candidates in the k-th bin) -> the smallest min(k, n) ids
            assert int(counts[0]) > topk_mod.CAND_CAP, "the premise: more tied candidates than the candidate list holds"
            n0 = int(counts[0])
            want = torch.arange(min(K, n0) - 1, -1, -1, device=device, dtype=torch.int32)
            assert torch.equal(out[0, : want.numel()], want)
        if kind == "boundary_tie":
            for t in range(6):
                n = int(counts[t])
                if n > K:
                    # the K smallest of the K + 1 tied maxima: id K is the dropped one
                    assert int(out[t, 0]) == K - 1 and int(out[t, K - 1]) == 0 and K not in out[t].tolist()
        if kind == "recency":
            n0 = int(counts[0])
            assert torch.equal(out[0], torch.arange(n0 - 1, n0 - 1 - K, -1, device=device, dtype=torch.int32)), "the premise: the last k ids"
        if kind == "whole_wide":
            n0 = int(counts[0])
            assert int((logits[0, :n0] == 77.0).sum()) == K > topk_mod.RANK_CAP, "the premise: exactly k maxima, more than the rank list"


@needs_cuda
@needs_dsl_target
def test_candidate_list_boundary_and_the_fallback_agree():
    """Exactly ``CAND_CAP`` tied candidates in the k-th bin (the list path, full) and ``CAND_CAP + 1`` (the row-sweep fallback):
    both bitwise the reference; the two paths are the same function."""
    device = torch.device("cuda")
    cap = topk_mod.CAND_CAP
    cols = 2 * cap
    g = torch.Generator(device=device).manual_seed(5)
    for tied in (cap, cap + 1):
        logits = torch.randn(2, cols, device=device, generator=g).abs() + 10.0  # every other value above the tied block
        logits[:, :tied] = 1.0  # the tied block, below the rest: the k-th bin holds exactly `tied` candidates when n - tied < k
        counts = torch.tensor([tied + K // 2, cols], dtype=torch.int32, device=device)
        _check(logits, counts, K, threads_per_cta=1024)


@needs_cuda
@needs_dsl_target
@pytest.mark.parametrize("top_k", [1, 4, 100, 2048])
def test_other_list_widths(top_k):
    """``top_k`` is a compile-time constant read off ``block_ids.shape[1]``: 1, the block's smallest multiple of 4, an odd width, the
    radix kernel's bound."""
    device = torch.device("cuda")
    logits, counts = _case(6, 4096, top_k, seed=top_k, device=device)
    _check(logits, counts, top_k, threads_per_cta=1024)


@needs_cuda
@needs_dsl_target
def test_every_knob_value_is_bitwise_the_same_function():
    """The thread count is a performance knob: every value gives the same bits (and ``cluster_ctas = 1``, the served value)."""
    device = torch.device("cuda")
    logits, counts = _case(8, 8192, K, seed=5, device=device)
    ref = reference_canonical(logits, counts, K)
    for threads in topk_mod.THREAD_COUNTS:
        out = _run(logits, counts, K, cluster_ctas=1, threads_per_cta=threads)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"threads_per_cta={threads} differs"
    for cluster_ctas in _cluster_sizes()[1:]:
        out = _run(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=1024)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"cluster_ctas={cluster_ctas} differs"


@needs_cuda
@needs_cluster_gpu
@needs_dsl_target
@pytest.mark.parametrize("cluster_ctas", [4, 8, 16])
@pytest.mark.parametrize("cols", [520, 8192, 32768])
def test_the_cluster_form_matches_the_exact_canonical_reference(cols, cluster_ctas):
    """The row split over 4 / 8 / 16 CTAs (16 = the non-portable size): the same six-row cases, bitwise the reference, two launches
    bitwise; the tie kinds and a 128-row block as well (the per-CTA selected counts exchanged once, the exit drain)."""
    device = torch.device("cuda")
    logits, counts = _case(6, cols, K, seed=cols + 7 * cluster_ctas, device=device)
    _check(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=1024)
    for kind in ("ties", "boundary_tie", "wide"):
        logits, counts = _case(6, cols, K, seed=13, device=device, kind=kind)
        _check(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=1024)
    logits, counts = _case(128, cols, K, seed=128 + cluster_ctas, device=device)
    _check(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=1024)


@needs_cuda
@needs_cluster_gpu
@needs_dsl_target
@pytest.mark.parametrize("cluster_ctas", [4, 8, 16])
@pytest.mark.parametrize("cols", [8192, 32768])
def test_the_cluster_form_union_list_passes_and_the_id_rank_on_a_spread_tie(cols, cluster_ctas):
    """The k-th bin's candidates spread over every CTA (each under ``UNION_CAP``: the union path, not the merged-pass one), more of
    them than ``RANK_CAP`` (the narrowing passes over the union list, not the all-pairs rank) and an EXACT k-th key with a PARTIAL
    take (the id rank behind the R2 barrier names ``idstar``): the one path of the cluster form no other cell reaches.  Eight
    launches, each bitwise the single-CTA form's result and the canonical reference; the fp64 oracle on the set.  (The 2026-10-21
    barrier audit's HIGH finding lived here: the finder published ``idstar`` in the word every thread had just read as ``above``
    after the last pass's scan barrier, with no barrier between that read and the write -- a latent race; the kernel now publishes
    it in the fourth word of the record, read nowhere else.)"""
    device = torch.device("cuda")
    logits, counts = _case(6, cols, K, seed=29 + cluster_ctas, device=device, kind="spread_tie")
    n0 = int(counts[0])
    vals = torch.sort(logits[0, :n0], descending=True).values
    assert (
        float(vals[K - 1]) == 100.0 == float(vals[K]) and int((logits[0, :n0] > 100.0).sum()) == (3 * K) // 4
    ), "the premise: an exact k-th key, a partial take"
    assert int((logits[0, :n0] == 100.0).sum()) > topk_mod.RANK_CAP, "the premise: more tied candidates than the all-pairs rank takes"
    want = _check(logits, counts, K, cluster_ctas=1, threads_per_cta=1024)
    outs = [_run(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=1024) for _ in range(8)]
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert torch.equal(out, want), f"launch {i} of the cluster form differs from the single-CTA form"
    assert_set_matches_fp64_topk(logits, counts, outs[0])


@needs_cuda
@needs_dsl_target
def test_many_rows_one_block_each():
    """128 rows (the B 32 x S_q 4 cell) at 8192 blocks: one CTA per row, every row independent."""
    device = torch.device("cuda")
    logits, counts = _case(128, 8192, K, seed=128, device=device)
    _check(logits, counts, K, threads_per_cta=1024)


@needs_cuda
@needs_dsl_target
def test_identity_region_dead_rows_and_clamped_counts():
    """``counts`` of 0 / 1 / k - 1 / k -> the identity list then ``-1``; a negative count clamps to 0; a count past the row width clamps
    to the width; the output buffer is fully written (no sentinel survives in any slot)."""
    device = torch.device("cuda")
    cols = 1024
    g = torch.Generator(device=device).manual_seed(3)
    logits = torch.randn(7, cols, device=device, generator=g)
    counts = torch.tensor([0, 1, K - 1, K, -5, cols + 100, cols], dtype=torch.int32, device=device)
    out = _check(logits, counts, K, threads_per_cta=1024)
    assert bool((out[0] == -1).all()) and bool((out[4] == -1).all())
    assert int(out[1, 0]) == 0 and bool((out[1, 1:] == -1).all())
    assert torch.equal(out[3], torch.arange(K - 1, -1, -1, device=device, dtype=torch.int32))
    clamped = reference_canonical(logits[5:6], torch.tensor([cols], dtype=torch.int32, device=device), K)[0]
    assert torch.equal(out[5], clamped), "a count past the row width is clamped to the width"
    assert bool((out[5] >= 0).all()) and bool((out[6] >= 0).all()), "a row with more visible blocks than k fills every slot"
    assert int(out[2, K - 1]) == -1 and int(out[2, K - 2]) == 0, "k - 1 visible blocks: the identity list then one -1"


@needs_cuda
@needs_dsl_target
def test_counts_spelled_as_the_list_length_is_the_documented_trap_not_a_second_spelling():
    """The contract's trap, pinned for the integrator: ``counts`` is the VISIBLE count.  Spelled as the list length
    ``min(top_k, n_vis)`` -- the output-length vocabulary of the scorer and the block -- every row with ``n_vis > top_k`` lands in
    the identity region and the kernel returns the OLDEST ``top_k`` ids (``top_k - 1 .. 0``), bitwise reproducibly, with no error:
    the two spellings are NOT interchangeable, and nothing in the kernel can tell them apart.  The logits rise with the id, so the
    true top-k of a row with ``n_vis > top_k`` is its NEWEST ids and never the identity list (no random premise)."""
    device = torch.device("cuda")
    cols = 2048
    logits = torch.arange(cols, device=device, dtype=torch.float32).expand(6, cols).contiguous()
    counts = torch.tensor([cols, cols // 2 + 3, 0, K, K + 1, 1], dtype=torch.int32, device=device)
    over = counts > K
    assert bool(over.any()) and bool((~over).any())
    wrong = torch.clamp_max(counts, K)  # the list-length spelling
    out = _run(logits, wrong, K, threads_per_cta=1024)
    torch.cuda.synchronize()
    right = reference_canonical(logits, counts, K)
    identity = torch.arange(K - 1, -1, -1, device=device, dtype=torch.int32)
    for t in range(6):
        if bool(over[t]):
            assert torch.equal(out[t], identity), f"row {t}: the list-length spelling must yield the identity list"
            assert not torch.equal(out[t], right[t]), f"row {t}: the identity list is not the row's top-k"
        else:
            assert torch.equal(out[t], right[t]), f"row {t}: a row with counts <= top_k reads the same under both spellings"


@needs_cuda
@needs_dsl_target
def test_columns_past_the_count_may_hold_anything_and_a_short_row_in_a_wide_block():
    """The kernel loads the row's leading columns together with the count and the rest clamped to the last visible column: a row of
    8 visible blocks in a 32768-block tensor (the speculated columns past the count are uninitialised memory) and a 600-block row
    (the loads past the count re-read column 599) both read the reference."""
    device = torch.device("cuda")
    cols = 32768
    logits = torch.empty(3, cols, device=device)  # uninitialised past the counts on purpose
    g = torch.Generator(device=device).manual_seed(8)
    logits[0, :8] = torch.randn(8, device=device, generator=g)
    logits[1, :600] = torch.randn(600, device=device, generator=g)
    logits[2, :] = torch.randn(cols, device=device, generator=g)
    counts = torch.tensor([8, 600, cols], dtype=torch.int32, device=device)
    for threads in THREADS:
        _check(logits, counts, K, threads_per_cta=threads)


@needs_cuda
@needs_pdl_gpu
@needs_dsl_target
def test_use_pdl_arm_is_bitwise_the_plain_arm():
    """The griddepcontrol pair is traced under ``use_pdl`` and the launch carries the attribute; the two forms are bitwise."""
    device = torch.device("cuda")
    logits, counts = _case(6, 8192, K, seed=77, device=device)
    plain = _run(logits, counts, K, threads_per_cta=1024, use_pdl=False)
    pdl = _run(logits, counts, K, threads_per_cta=1024, use_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(plain, pdl)
    assert torch.equal(plain, reference_canonical(logits, counts, K))


@needs_cuda
@needs_pdl_gpu
@needs_dsl_target
def test_the_radix_top_k_kernel_of_the_standalone_selector_selects_the_same_set():
    """The second oracle: the DSA radix top-k (``deterministic=True``, ``tie_break=1``: the smaller ids at the cutoff) over the same
    visible logits selects the same SET per row (its order is ascending, ours descending)."""
    from cudnn.deepseek_sparse_attention.indexer_top_k import indexer_top_k_wrapper

    device = torch.device("cuda")
    logits, counts = _case(6, 8192, K, seed=21, device=device)
    clean = logits.clone()
    for t in range(6):
        clean[t, int(counts[t]) :] = 0.0  # the radix kernel reads its row up to counts only; keep its input finite regardless
    out = _check(logits, counts, K, threads_per_cta=1024)
    dsa = indexer_top_k_wrapper(clean, counts, K, next_n=1, return_val=False, tie_break=1, deterministic=True)["indices"]
    torch.cuda.synchronize()
    for t in range(6):
        ours = sorted(i for i in out[t].tolist() if i >= 0)
        theirs = sorted(i for i in dsa[t].tolist() if i >= 0)
        assert ours == theirs, f"row {t}: the sets differ ({len(set(ours) ^ set(theirs))} ids)"


# --------------------------------------------------------------------------------------------- declines (any CUDA device)


@needs_cuda
def test_declines_below_the_dsl_floor_by_version_before_any_compile(monkeypatch):
    """Rule 7: the version gate runs first and names the version; no kernel compile is reached (the compile hook raises if called)."""
    device = torch.device("cuda")
    logits, counts = _case(1, 256, K, seed=0, device=device)
    out = torch.empty(1, K, dtype=torch.int32, device=device)
    monkeypatch.setattr(topk_mod, "cutedsl_requirement_error", lambda what: f"{what} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    monkeypatch.setattr(topk_mod, "compile_cached", lambda *a, **k: (_ for _ in ()).throw(AssertionError("compile reached below the floor")))
    with pytest.raises(NotImplementedError, match="qsa_topk_decode requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        topk_mod.qsa_topk_decode(logits, counts, out, threads_per_cta=1024)
    with pytest.raises(NotImplementedError, match="found 4.6.2"):
        topk_mod.compile_qsa_topk_decode(cols_cap=1024, top_k=K, threads_per_cta=1024)


@needs_cuda
def test_declines_a_dsl_without_the_device_target(monkeypatch):
    monkeypatch.setattr(
        topk_mod, "cutedsl_arch_requirement_error", lambda cc: "SM107 requires a CuTe DSL build supporting sm_107a; found 4.7.0 without that target"
    )
    with pytest.raises(NotImplementedError, match="sm_107a"):
        topk_mod.compile_qsa_topk_decode(cols_cap=1024, top_k=K, threads_per_cta=1024)


@needs_cuda
def test_declines_clusters_and_pdl_below_cc_9(monkeypatch):
    """Thread-block clusters, DSMEM and griddepcontrol exist from Hopper: both arms decline by name on an older part (modeled); the
    single-CTA plain arm has no such gate."""
    monkeypatch.setattr(topk_mod, "compute_capability", lambda device: (8, 0))
    monkeypatch.setattr(topk_mod, "cutedsl_arch_requirement_error", lambda cc: None)
    with pytest.raises(NotImplementedError, match=r"cluster_ctas = 8 needs thread-block clusters and DSMEM \(cc 9\.0\+\); found cc 8\.0"):
        topk_mod.compile_qsa_topk_decode(cols_cap=8192, top_k=K, cluster_ctas=8, threads_per_cta=1024)
    with pytest.raises(NotImplementedError, match=r"use_pdl=True needs griddepcontrol \(cc 9\.0\+\); found cc 8\.0"):
        topk_mod.compile_qsa_topk_decode(cols_cap=1024, top_k=K, threads_per_cta=1024, use_pdl=True)


@needs_cuda
def test_declines_a_shared_memory_need_above_the_device_limit(monkeypatch):
    monkeypatch.setattr(topk_mod, "shared_memory_per_block_optin", lambda device: 1024)
    monkeypatch.setattr(topk_mod, "compute_capability", lambda device: (10, 7))
    monkeypatch.setattr(topk_mod, "cutedsl_arch_requirement_error", lambda cc_: None)
    need = topk_mod.smem_bytes_for(2048, 1, 1024, K)
    with pytest.raises(NotImplementedError, match=f"needs {need} B of shared memory per CTA; the device allows 1024 B"):
        topk_mod.compile_qsa_topk_decode(cols_cap=2048, top_k=K, threads_per_cta=1024)


def test_geometry_refusals_name_the_rule():
    """The served knob domains (reject side): a cluster size other than 1 names the follow-up, a thread count outside the domain, a
    list width outside ``[1, 2048]``, a capacity that is not a power of two or not whole rounds, a register tile past ``MAX_EPT``."""
    with pytest.raises(ValueError, match=r"cluster_ctas must be one of \(1, 4, 8, 16\)"):
        topk_mod.validate_topk_geometry(2048, K, 2, 1024)
    with pytest.raises(ValueError, match=r"cluster_ctas must be one of \(1, 4, 8, 16\)"):
        topk_mod.validate_topk_geometry(2048, K, 32, 1024)
    with pytest.raises(ValueError, match=r"threads_per_cta must be one of \(512, 1024\)"):
        topk_mod.validate_topk_geometry(2048, K, 1, 256)
    with pytest.raises(ValueError, match=r"the cluster form \(cluster_ctas = 8\) runs 1024 threads per CTA; got threads_per_cta = 512"):
        topk_mod.validate_topk_geometry(8192, K, 8, 512)
    with pytest.raises(ValueError, match="multiple of cluster_ctas x threads_per_cta"):
        topk_mod.validate_topk_geometry(8192, K, 16, 1024)
    with pytest.raises(ValueError, match=r"top_k must be in \[1, 2048\]"):
        topk_mod.validate_topk_geometry(2048, 0, 1, 1024)
    with pytest.raises(ValueError, match=r"top_k must be in \[1, 2048\]"):
        topk_mod.validate_topk_geometry(2048, 2049, 1, 1024)
    with pytest.raises(ValueError, match="power of two"):
        topk_mod.validate_topk_geometry(3000, K, 1, 1024)
    with pytest.raises(ValueError, match="multiple of cluster_ctas x threads_per_cta"):
        topk_mod.validate_topk_geometry(512, K, 1, 1024)
    with pytest.raises(ValueError, match=r"128 keys per thread overflow the register tile \(64 per thread\); use threads_per_cta = 1024"):
        topk_mod.validate_topk_geometry(1 << 16, K, 1, 512)
    with pytest.raises(ValueError, match=r"overflow the register tile.*wider than 65536 blocks, past this kernel's reach"):
        topk_mod.validate_topk_geometry(1 << 17, K, 1, 1024)
    with pytest.raises(ValueError, match="n_blocks_max must be >= 1"):
        topk_mod.cols_cap_for(0)
    assert topk_mod.cols_cap_for(8, 1, 1024) == 1024 and topk_mod.cols_cap_for(2049, 1, 1024) == 4096 and topk_mod.cols_cap_for(32768, 1, 512) == 32768
    assert topk_mod.cols_cap_for(8, 16, 1024) == 16384 and topk_mod.cols_cap_for(32768, 8, 1024) == 32768
    assert topk_mod.CLUSTER_SIZES == (1, 4, 8, 16) and topk_mod.DEFAULT_CLUSTER_CTAS == 1 and topk_mod.THREAD_COUNTS == (512, 1024)


@needs_cuda
@needs_dsl_target
def test_execute_form_checks_refuse_by_name():
    """Rule 1 in both directions at the run entry: dtype, rank, device, stride, alignment, width, capacity -- refused, never converted."""
    device = torch.device("cuda")
    r = topk_mod.compile_qsa_topk_decode(cols_cap=2048, top_k=K, threads_per_cta=1024)
    logits, counts = _case(2, 2048, K, seed=1, device=device)
    out = torch.empty(2, K, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="counts must be a contiguous 1-D int32 CUDA tensor"):
        topk_mod.run_qsa_topk_decode(r, logits, counts.to(torch.int64), out, stream=None)
    with pytest.raises(ValueError, match="counts must be a contiguous 1-D int32 CUDA tensor"):
        topk_mod.run_qsa_topk_decode(r, logits, counts.cpu(), out, stream=None)
    with pytest.raises(ValueError, match="logits must have dtype torch.float32"):
        topk_mod.run_qsa_topk_decode(r, logits.to(torch.bfloat16), counts, out, stream=None)
    with pytest.raises(ValueError, match="logits must hold 2 rows"):
        topk_mod.run_qsa_topk_decode(r, logits[:1], counts, out, stream=None)
    with pytest.raises(ValueError, match="logits needs a unit column stride"):
        topk_mod.run_qsa_topk_decode(r, logits.t().contiguous().t(), counts, out, stream=None)
    with pytest.raises(ValueError, match="logits must be 16-B aligned"):
        topk_mod.run_qsa_topk_decode(r, logits[:, 1:], counts, out, stream=None)
    with pytest.raises(ValueError, match=r"block_ids must be \[T, top_k = 512\]"):
        topk_mod.run_qsa_topk_decode(r, logits, counts, out[:, :256], stream=None)
    with pytest.raises(ValueError, match="block_ids must have dtype torch.int32"):
        topk_mod.run_qsa_topk_decode(r, logits, counts, out.to(torch.int64), stream=None)
    wide, wide_counts = _case(2, 4096, K, seed=2, device=device)
    with pytest.raises(ValueError, match="compiled for cols_cap = 2048"):
        topk_mod.run_qsa_topk_decode(r, wide, wide_counts, out, stream=None)
    with pytest.raises(ValueError, match="block_ids must be a"):
        topk_mod.qsa_topk_decode(logits, counts, out[0])
    with pytest.raises(ValueError, match=r"cluster_ctas must be one of \(1, 4, 8, 16\)"):
        topk_mod.qsa_topk_decode(logits, counts, out, cluster_ctas=2)
    # the row count is capped by the launch grid (one CTA per row along grid.y): refused by name before any launch -- the
    # row-stride-0 expanded views keep the probe small (the form checks ask for a unit column stride only)
    too_many = topk_mod.MAX_ROWS + 1
    with pytest.raises(ValueError, match=f"holds {too_many} rows; the launch grid caps the row count at {topk_mod.MAX_ROWS}"):
        topk_mod.run_qsa_topk_decode(
            r,
            torch.zeros(1, 2048, device=device).expand(too_many, 2048),
            torch.zeros(too_many, dtype=torch.int32, device=device),
            torch.zeros(1, K, dtype=torch.int32, device=device).expand(too_many, K),
            stream=None,
        )
    # the row-strided forms ARE served (a wider row stride, a sliced output): the kernel addresses them natively
    padded = torch.empty(2, 2048 + 64, device=device)[:, :2048]
    padded.copy_(logits)
    out_wide = torch.full((2, K + 32), SENTINEL, dtype=torch.int32, device=device)
    topk_mod.run_qsa_topk_decode(r, padded, counts, out_wide[:, :K], stream=None)
    torch.cuda.synchronize()
    assert torch.equal(out_wide[:, :K], reference_canonical(logits, counts, K)) and bool((out_wide[:, K:] == SENTINEL).all())


# --------------------------------------------------------------------------------------------- the contract text and the rendering


def test_module_docstring_carries_the_two_pre_development_tables_and_the_contract():
    doc = topk_mod.__doc__
    assert "Barrier table" in doc and "SMEM table" in doc, "the two pre-development tables are REQUIRED in the module docstring"
    assert "SUM(issuing lanes) == THREADS" in doc and "THRESHOLD PAIR" in doc
    assert "``logits [T, n_blocks_max]`` fp32" in doc and "``counts [T]`` int32" in doc and "``block_ids [T, top_k]`` int32" in doc
    assert "STRICTLY DESCENDING" in doc and "SMALLER block ids" in doc
    assert "No performance number is claimed" in doc
    # the one trap of the contract: `counts` is the VISIBLE count; the list-length spelling is forbidden, never "either spelling"
    assert "either spelling" not in doc and "NEVER the list length" in doc, "the counts clause must name the visible count and forbid the list-length spelling"
    # the read-past-the-count clause changed with this form: the columns may be READ, they never influence the result
    assert "MAY READ them" in doc and "NEVER influence the result" in doc
    assert "once per CTA right after the ``counts`` read" in doc, "the PDL-sites paragraph must describe the trigger placement the code has"


def test_compile_keys_on_the_file_digest_and_the_memo_dict_key(monkeypatch):
    """``compile_qsa_topk_decode`` hands ``compile_cached`` ``template_key(globals(), {"key": <the dict key>}, "compile_qsa_topk_decode")``
    -- the file digest the module names itself (``FROST_SOURCE_DIGEST``), the tag and exactly the plain tuple the module dict memoises
    under -- with the site's symbol and options; a repeat request is served by the dict (no second ``compile_cached`` call)."""
    calls = []
    monkeypatch.setattr(topk_mod, "compile_cached", lambda fn, *a, **k: calls.append(k) or object())
    monkeypatch.setattr(topk_mod, "compiled_cache", {})
    monkeypatch.setattr(topk_mod, "current_device", lambda: 0)
    monkeypatch.setattr(topk_mod, "compute_capability", lambda device: (10, 7))
    monkeypatch.setattr(topk_mod, "cutedsl_arch_requirement_error", lambda cc_: None)
    monkeypatch.setattr(topk_mod, "shared_memory_per_block_optin", lambda device: 1 << 20)
    assert topk_mod.FROST_SOURCE_DIGEST == hashlib.sha256(Path(topk_mod.__file__).read_bytes()).hexdigest()[:16]
    r = topk_mod.compile_qsa_topk_decode(cols_cap=4096, top_k=K, threads_per_cta=1024, use_pdl=True)
    (key,) = topk_mod.compiled_cache.keys()
    assert key == (4096, K, 1, 1024, True, 0) and all(isinstance(v, (int, bool)) for v in key)
    assert calls[0]["cache_key"] == f"{topk_mod.FROST_SOURCE_DIGEST}|compile_qsa_topk_decode|{[('key', key)]!r}"
    assert calls[0]["cache_key"] == cc.template_key(vars(topk_mod), {"key": key}, "compile_qsa_topk_decode")
    assert (calls[0]["symbol"], calls[0]["options"]) == ("cudnn_qsa_topk_decode", "--enable-tvm-ffi")
    assert r.compiled is topk_mod.compiled_cache[key]
    assert topk_mod.compile_qsa_topk_decode(cols_cap=4096, top_k=K, threads_per_cta=1024, use_pdl=True).compiled is r.compiled
    assert len(calls) == 1, "a repeat request is served by the module dict, not by another compile_cached call"


_FACADE_PROBE = textwrap.dedent("""
    import sys
    import cudnn.frost.buffers as b
    b._DSL_STATE = (True, ("nvidia-cutlass-dsl", "4.6.2"))  # the public wheel the serving stacks pin, below the FROST floor
    import cudnn.gated_attention_block.qsa_topk as F
    try:
        F.qsa_topk_decode(None, None, None)
    except NotImplementedError as e:
        print("DECLINE", str(e))
    else:
        print("NO_DECLINE")
    try:
        F.qsa_topk_decode_kernel_module()
    except NotImplementedError as e:
        print("DECLINE_MODULE", str(e))
    else:
        print("NO_DECLINE_MODULE")
    print("KERNEL_IMPORTED", "cudnn.gated_attention_block.kernels.qsa_topk_decode" in sys.modules)
    print("TILE_DSL_IMPORTED", any(m in sys.modules for m in ("cudnn.frost.tile_dsl.barrier", "cudnn.frost.tile_dsl.tma")))
    """)


def test_the_facade_declines_a_too_old_dsl_by_version_without_importing_the_kernel_module():
    """Rule 7 at the entry a caller hits: in a fresh process whose DSL discovery is modeled as the public 4.6.2 wheel, the DSL-free
    facade raises the typed ``NotImplementedError`` naming the version -- not the bare ``ModuleNotFoundError`` the kernel module's own
    import would raise from inside the DSL -- and neither the kernel module nor the ``tile_dsl`` modules it pulls is imported."""
    proc = subprocess.run([sys.executable, "-c", _FACADE_PROBE], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = proc.stdout
    assert "DECLINE qsa_topk_decode requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2" in out, out
    assert "DECLINE_MODULE qsa_topk_decode requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2" in out, out
    assert "ModuleNotFoundError" not in out + proc.stderr, out + proc.stderr
    assert "KERNEL_IMPORTED False" in out and "TILE_DSL_IMPORTED False" in out, out


@needs_cuda
@needs_dsl_target
def test_the_facade_serves_the_kernel_module_at_or_above_the_floor():
    """With the floor met the facade's lazy import IS the kernel module and its convenience entry is the module's (bitwise)."""
    from cudnn.gated_attention_block import qsa_topk as facade

    assert facade.qsa_topk_decode_kernel_module() is topk_mod
    device = torch.device("cuda")
    logits, counts = _case(6, 4096, K, seed=33, device=device)
    via_facade = torch.full((6, K), SENTINEL, dtype=torch.int32, device=device)
    assert facade.qsa_topk_decode(logits, counts, via_facade, threads_per_cta=1024) is via_facade
    torch.cuda.synchronize()
    assert torch.equal(via_facade, _check(logits, counts, K, threads_per_cta=1024))


_RENDER_PROBE = textwrap.dedent("""
    import glob, os, re, sys
    import cutlass, cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_compact_tensor
    from cudnn.gated_attention_block.kernels import qsa_topk_decode as M
    arch, dump, thr, pdl, cap, C = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5]), int(sys.argv[6])
    cute.compile(M.qsa_topk_decode_launch, M._fake_rows(cutlass.Int32),
                 make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=4),
                 M._fake_rows(cutlass.Int32), cutlass.Int32(1), cap, 512, C, thr, pdl, M.smem_bytes_for(cap, C, thr),
                 make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-ptx")
    ptx = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    txt = open(ptx[-1]).read()
    for k in ("bar.sync", "barrier.cluster.arrive.relaxed", "barrier.cluster.arrive.release", "barrier.cluster.wait", "ld.shared::cluster",
              "st.async", "mbarrier.init", "mbarrier.arrive.expect_tx", "mbarrier.try_wait", "mapa", "match.any.sync", "atom.shared",
              "red.shared", "vote.sync.ballot", "redux.sync", "griddepcontrol.wait", "griddepcontrol.launch_dependents", ".local",
              "lanemask_lt", "ld.global", "st.global"):
        print("COUNT", k, len(re.findall(re.escape(k), txt)))
    """)


def _arch_known(arch: str) -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string(arch)
        return True
    except Exception:
        return False


def _render(tmp_path, threads, use_pdl, cap, cluster_ctas):
    dump = tmp_path / f"t{threads}_p{int(use_pdl)}_c{cap}_k{cluster_ctas}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump))
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_PROBE, "sm_107a", str(dump), str(threads), "1" if use_pdl else "0", str(cap), str(cluster_ctas)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"COUNT (\S+) (\d+)", proc.stdout)}


@pytest.mark.parametrize("threads,use_pdl,cap", [(1024, False, 32768), (1024, True, 8192), (512, False, 32768)])
def test_sm_107a_rendering_of_the_single_cta_form_carries_the_tables_structure(tmp_path, threads, use_pdl, cap):
    """A host trace-compile for sm_107a (no GPU): the single-CTA form holds NO cluster barrier, NO DSMEM access, NO mbarrier, NO
    ``match.any``; its hot path is ``bar.sync`` sites only (the barrier table: B0, B1 / B2 / B3, C1, R1, F on the common path, the
    rare paths' own); shared atomics and warp ``redux`` present; the griddepcontrol pair iff ``use_pdl``; no ``%lanemask_lt`` read (the
    mask is derived); the row read by ``EPT + 1`` global loads (the count and one per round); no local-memory spill at 8192 columns
    (the 32-key tile at 32768 x 1024 is allowed its few)."""
    if not _arch_known("sm_107a"):
        pytest.skip("this CuTe DSL has no sm_107a target")
    counts = _render(tmp_path, threads, use_pdl, cap, 1)
    ept = cap // threads
    assert counts["bar.sync"] >= 8, counts
    for k in (
        "barrier.cluster.arrive.relaxed",
        "barrier.cluster.arrive.release",
        "barrier.cluster.wait",
        "ld.shared::cluster",
        "st.async",
        "mbarrier.init",
        "mapa",
        "match.any.sync",
        "lanemask_lt",
    ):
        assert counts[k] == 0, (k, counts)
    if cap <= 8192:
        assert counts[".local"] == 0, counts
    assert counts["atom.shared"] + counts["red.shared"] > 0 and counts["redux.sync"] >= 2 and counts["vote.sync.ballot"] >= 2 * ept, counts
    assert counts["griddepcontrol.wait"] == counts["griddepcontrol.launch_dependents"] == (1 if use_pdl else 0), counts
    assert (
        counts["ld.global"] == 2 * ept + 1 and counts["st.global"] >= ept + 2
    ), counts  # two load paths (the row as wide as the capacity / narrower), the count


@pytest.mark.parametrize("cluster_ctas,use_pdl", [(8, False), (16, True)])
def test_sm_107a_rendering_of_the_cluster_form_carries_the_exchange_structure(tmp_path, cluster_ctas, use_pdl):
    """The cluster form: exactly two relaxed cluster barriers (the mbarrier-init publish and the exit drain) and NO release arrive (a
    GPU-scope membar per site); the exchanges are ``st.async`` to ``mapa``-translated peers completing mbarriers (the five mbarrier
    inits, one ``arrive.expect_tx`` per exchange site), no ``ld.shared::cluster``; the griddepcontrol pair iff ``use_pdl``."""
    if not _arch_known("sm_107a"):
        pytest.skip("this CuTe DSL has no sm_107a target")
    counts = _render(tmp_path, 1024, use_pdl, 32768, cluster_ctas)
    assert counts["barrier.cluster.arrive.relaxed"] == 2 and counts["barrier.cluster.wait"] == 2 and counts["barrier.cluster.arrive.release"] == 0, counts
    assert counts["st.async"] >= 5 and counts["mapa"] >= 5 and counts["mbarrier.init"] == 5 and counts["mbarrier.arrive.expect_tx"] >= 5, counts
    assert counts["ld.shared::cluster"] == 0 and counts["match.any.sync"] == 0 and counts["lanemask_lt"] == 0, counts
    assert counts["griddepcontrol.wait"] == counts["griddepcontrol.launch_dependents"] == (1 if use_pdl else 0), counts
