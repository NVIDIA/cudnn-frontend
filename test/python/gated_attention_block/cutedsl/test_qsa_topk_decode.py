# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_topk_decode``: the block-sparse indexer's decode top-k (one thread-block cluster per row, an exact radix select over
fp32 block logits, the list in canonical order).

Every accept cell compares the kernel BITWISE against an exact canonical reference built from the contract alone (``torch.sort``
of the visible logits, descending and stable -- so a tie keeps the smaller id -- the first ``min(top_k, counts)`` ids re-sorted
descending, ``-1`` past them), checks the SET against ``torch.topk`` of the fp64 copy of the same values (the sharp oracle up to
exact ties at the k-th value), and launches twice (bitwise).  The radix top-k kernel the standalone decode selector binds today
is a second oracle for the set on a cluster-capable device.  Every knob value (cluster size, thread count, the PDL pair) is the
same function and is pinned bitwise against the others.

The CuTe DSL floor gates the kernel-module import (Rule 7): below it every cell of this module skips with the version named;
with the floor met the version decline itself is pinned through a modeled sub-floor state.  The cluster arms (``cluster_ctas >
1``) and the PDL arm need a cc 9.0+ device and skip elsewhere; the non-cluster arm runs on any CuTe DSL device, so the host tier
covers the radix select itself.
"""

import hashlib
import importlib
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
needs_cluster_gpu = pytest.mark.skipif(_cc() is None or _cc() < (9, 0), reason=f"thread-block clusters, DSMEM and griddepcontrol need cc 9.0+; found {_cc()}")
_TARGET = cutedsl_arch_requirement_error(_cc()) if _cc() is not None else None
needs_dsl_target = pytest.mark.skipif(_TARGET is not None, reason=str(_TARGET))

K = 512
SENTINEL = -2  # never produced by the kernel (ids are >= 0, the padding is -1): a slot still holding it after a launch was never written


def _cluster_sizes():
    cc = _cc()
    return (1,) if cc is None or cc < (9, 0) else (1, 2, 4, 8)


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
    a single visible block; the columns past ``counts`` hold NaN or huge garbage (never read)."""
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
    base = [cols, max(cols // 2 + 3, 1), 0, min(cols, top_k), min(cols, top_k + 1), 1]
    counts = torch.tensor([base[i % len(base)] for i in range(rows)], dtype=torch.int32, device=device)
    for t in range(rows):
        n = int(counts[t])
        if n < cols:
            logits[t, n:] = float("nan") if t % 2 else 1e30
    return logits, counts


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
@pytest.mark.parametrize("cluster_ctas", _cluster_sizes())
@pytest.mark.parametrize("cols", [8, 520, 2048, 8192, 32768])
def test_matches_the_exact_canonical_reference(cols, cluster_ctas):
    """Six rows per case (full / partial / dead / k / k + 1 / one block visible; NaN or 1e30 past the counts) at every row width
    of the serving cells (8K / 32K / 128K tokens = 2048 / 8192 / 32768 blocks) and two small widths: bitwise the canonical reference,
    the set the fp64 top-k, two launches bitwise."""
    device = torch.device("cuda")
    logits, counts = _case(6, cols, K, seed=cols + cluster_ctas, device=device)
    _check(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=256)


@needs_cuda
@needs_dsl_target
@pytest.mark.parametrize("kind", ["ties", "boundary_tie", "neg"])
def test_ties_keep_the_smaller_ids_and_negative_logits_order_by_value(kind):
    """A row of identical logits, a two-valued row, exactly ``k + 1`` equal maxima (the largest of them is the one dropped), all-negative
    rows: the k-th-value ties resolve to the smaller ids and the negative keys order by value."""
    device = torch.device("cuda")
    for cluster_ctas in _cluster_sizes():
        logits, counts = _case(6, 4096, K, seed=11, device=device, kind=kind)
        out = _check(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=256)
        if kind == "ties":
            # row 0: every visible block ties -> the smallest min(k, n) ids, descending
            n0 = int(counts[0])
            want = torch.arange(min(K, n0) - 1, -1, -1, device=device, dtype=torch.int32)
            assert torch.equal(out[0, : want.numel()], want)
        if kind == "boundary_tie":
            for t in range(6):
                n = int(counts[t])
                if n > K:
                    # the K smallest of the K + 1 tied maxima: id K is the dropped one
                    assert int(out[t, 0]) == K - 1 and int(out[t, K - 1]) == 0 and K not in out[t].tolist()


@needs_cuda
@needs_dsl_target
@pytest.mark.parametrize("top_k", [1, 4, 100, 2048])
def test_other_list_widths(top_k):
    """``top_k`` is a compile-time constant read off ``block_ids.shape[1]``: 1, the block's smallest multiple of 4, an odd width, the
    radix kernel's bound."""
    device = torch.device("cuda")
    logits, counts = _case(6, 4096, top_k, seed=top_k, device=device)
    _check(logits, counts, top_k, cluster_ctas=_cluster_sizes()[-1], threads_per_cta=256)


@needs_cuda
@needs_dsl_target
def test_every_knob_value_is_bitwise_the_same_function():
    """The cluster size and the thread count are performance knobs: every value gives the same bits."""
    device = torch.device("cuda")
    logits, counts = _case(8, 8192, K, seed=5, device=device)
    ref = reference_canonical(logits, counts, K)
    for cluster_ctas in _cluster_sizes():
        for threads in topk_mod.THREAD_COUNTS:
            out = _run(logits, counts, K, cluster_ctas=cluster_ctas, threads_per_cta=threads)
            torch.cuda.synchronize()
            assert torch.equal(out, ref), f"cluster_ctas={cluster_ctas} threads_per_cta={threads} differs"


@needs_cuda
@needs_dsl_target
def test_many_rows_one_cluster_each():
    """128 rows (the B 32 x S_q 4 cell) at 8192 blocks: one cluster per row, every row independent."""
    device = torch.device("cuda")
    logits, counts = _case(128, 8192, K, seed=128, device=device)
    _check(logits, counts, K, cluster_ctas=_cluster_sizes()[-1], threads_per_cta=256)


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
    out = _check(logits, counts, K, cluster_ctas=_cluster_sizes()[-1], threads_per_cta=256)
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
    out = _run(logits, wrong, K, cluster_ctas=_cluster_sizes()[-1], threads_per_cta=256)
    torch.cuda.synchronize()
    right = reference_canonical(logits, counts, K)
    identity = torch.arange(K - 1, -1, -1, device=device, dtype=torch.int32)
    for t in range(6):
        if bool(over[t]):
            assert torch.equal(out[t], identity), f"row {t}: the list-length spelling must yield the identity list"
            assert not torch.equal(out[t], right[t]), f"row {t}: the identity list is not the row's top-k"
        else:
            assert torch.equal(out[t], right[t]), f"row {t}: a row with counts <= top_k reads the same under both spellings"


def _slice_lengths(n: int, cluster_ctas: int) -> list:
    """The kernel's slice rule (its docstring): CTA ``q`` owns ``[q x ceil(n / C), ...)`` clamped to the row -- the per-CTA element counts."""
    s = -(-n // cluster_ctas)
    return [max(0, min(s, n - q * s)) for q in range(cluster_ctas)]


@needs_cuda
@needs_dsl_target
def test_a_slice_without_elements_and_an_uneven_split():
    """Two row blocks on the largest cluster: ``n`` just above ``k = 512`` (an uneven split -- the CTAs hold different counts and none
    is empty, the premise checked with the kernel's slice rule), then ``n`` in ``5 .. 11`` at ``k = 4`` (``ceil(n / C)`` of 1 or 2: the
    last CTAs own NO element and contribute zero totals -- at least one empty slice per row ASSERTED with the slice rule, not assumed;
    on the non-cluster host the second block is a small-``k`` cell)."""
    device = torch.device("cuda")
    C = _cluster_sizes()[-1]
    g = torch.Generator(device=device).manual_seed(9)
    logits = torch.randn(6, 2048, device=device, generator=g)
    counts = torch.tensor([K + 1, K + 7, K + 2 * C + 1, K + 3, K + 5, K + 9], dtype=torch.int32, device=device)
    uneven = [_slice_lengths(int(n), C) for n in counts]
    if C > 1:
        assert all(len(set(lens)) > 1 for lens in uneven), uneven
    assert all(0 not in lens for lens in uneven), f"no slice is empty at k = 512 (ceil(n / C) >= 64 fills every CTA): {uneven}"
    _check(logits, counts, K, cluster_ctas=C, threads_per_cta=256)
    small_k = 4
    small = torch.tensor([5, 6, 7, 9, 10, 11], dtype=torch.int32, device=device)  # 8 left out: ceil(8 / 8) = 1 fills all eight CTAs
    empties = [_slice_lengths(int(n), C) for n in small]
    if C > 1:
        assert all(0 in lens for lens in empties), f"the premise: every row leaves at least one CTA without an element: {empties}"
    _check(logits, small, small_k, cluster_ctas=C, threads_per_cta=256)


@needs_cuda
@needs_cluster_gpu
@needs_dsl_target
def test_use_pdl_arm_is_bitwise_the_plain_arm():
    """The griddepcontrol pair is traced under ``use_pdl`` and the launch carries the attribute; the two forms are bitwise."""
    device = torch.device("cuda")
    logits, counts = _case(6, 8192, K, seed=77, device=device)
    plain = _run(logits, counts, K, cluster_ctas=8, threads_per_cta=256, use_pdl=False)
    pdl = _run(logits, counts, K, cluster_ctas=8, threads_per_cta=256, use_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(plain, pdl)
    assert torch.equal(plain, reference_canonical(logits, counts, K))


@needs_cuda
@needs_cluster_gpu
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
    out = _check(logits, counts, K, cluster_ctas=8, threads_per_cta=256)
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
        topk_mod.qsa_topk_decode(logits, counts, out, cluster_ctas=1, threads_per_cta=256)
    with pytest.raises(NotImplementedError, match="found 4.6.2"):
        topk_mod.compile_qsa_topk_decode(cols_cap=256, top_k=K, cluster_ctas=1, threads_per_cta=256)


@needs_cuda
def test_declines_a_dsl_without_the_device_target(monkeypatch):
    monkeypatch.setattr(
        topk_mod, "cutedsl_arch_requirement_error", lambda cc: "SM107 requires a CuTe DSL build supporting sm_107a; found 4.7.0 without that target"
    )
    with pytest.raises(NotImplementedError, match="sm_107a"):
        topk_mod.compile_qsa_topk_decode(cols_cap=256, top_k=K, cluster_ctas=1, threads_per_cta=256)


@needs_cuda
def test_declines_clusters_and_pdl_below_cc_9(monkeypatch):
    """Thread-block clusters, DSMEM and griddepcontrol exist from Hopper: both arms decline by name on an older part (modeled)."""
    monkeypatch.setattr(topk_mod, "compute_capability", lambda device: (8, 0))
    monkeypatch.setattr(topk_mod, "cutedsl_arch_requirement_error", lambda cc: None)
    with pytest.raises(NotImplementedError, match=r"cluster_ctas = 2 needs thread-block clusters and DSMEM \(cc 9\.0\+\); found cc 8\.0"):
        topk_mod.compile_qsa_topk_decode(cols_cap=512, top_k=K, cluster_ctas=2, threads_per_cta=256)
    with pytest.raises(NotImplementedError, match=r"use_pdl=True needs griddepcontrol \(cc 9\.0\+\); found cc 8\.0"):
        topk_mod.compile_qsa_topk_decode(cols_cap=256, top_k=K, cluster_ctas=1, threads_per_cta=256, use_pdl=True)


@needs_cuda
def test_declines_a_shared_memory_need_above_the_device_limit(monkeypatch):
    monkeypatch.setattr(topk_mod, "shared_memory_per_block_optin", lambda device: 1024)
    need = topk_mod.smem_bytes_for(2048, 1, 256)
    with pytest.raises(NotImplementedError, match=f"needs {need} B of shared memory per CTA; the device allows 1024 B -- use a larger cluster"):
        topk_mod.compile_qsa_topk_decode(cols_cap=2048, top_k=K, cluster_ctas=1, threads_per_cta=256)
    # at the largest portable cluster no larger one can be advised (the cc gate precedes the shared-memory gate: modeled cc 10.7 so the
    # shared-memory decline is the one reached on any host)
    monkeypatch.setattr(topk_mod, "compute_capability", lambda device: (10, 7))
    monkeypatch.setattr(topk_mod, "cutedsl_arch_requirement_error", lambda cc_: None)
    need8 = topk_mod.smem_bytes_for(2048, 8, 256)
    with pytest.raises(NotImplementedError, match=f"needs {need8} B of shared memory per CTA; the device allows 1024 B -- no larger portable cluster exists"):
        topk_mod.compile_qsa_topk_decode(cols_cap=2048, top_k=K, cluster_ctas=8, threads_per_cta=256)


def test_geometry_refusals_name_the_rule():
    with pytest.raises(ValueError, match=r"cluster_ctas must be one of \(1, 2, 4, 8\)"):
        topk_mod.validate_topk_geometry(2048, K, 3, 256)
    with pytest.raises(ValueError, match=r"threads_per_cta must be one of \(256, 512, 1024\)"):
        topk_mod.validate_topk_geometry(2048, K, 1, 128)
    with pytest.raises(ValueError, match=r"top_k must be in \[1, 2048\]"):
        topk_mod.validate_topk_geometry(2048, 0, 1, 256)
    with pytest.raises(ValueError, match=r"top_k must be in \[1, 2048\]"):
        topk_mod.validate_topk_geometry(2048, 2049, 1, 256)
    with pytest.raises(ValueError, match="power of two"):
        topk_mod.validate_topk_geometry(3000, K, 1, 256)
    with pytest.raises(ValueError, match="multiple of cluster_ctas x threads_per_cta"):
        topk_mod.validate_topk_geometry(1024, K, 8, 256)
    with pytest.raises(ValueError, match="overflows the 16-bit packed counts.*use a larger cluster"):
        topk_mod.validate_topk_geometry(1 << 17, K, 1, 256)
    with pytest.raises(ValueError, match="overflows the 16-bit packed counts.*wider than 8 x 65535 blocks"):
        topk_mod.validate_topk_geometry(1 << 20, K, 8, 256)  # at the largest portable cluster no larger one can be advised
    with pytest.raises(ValueError, match="n_blocks_max must be >= 1"):
        topk_mod.cols_cap_for(0)
    assert topk_mod.cols_cap_for(8, 8, 256) == 2048 and topk_mod.cols_cap_for(2049, 1, 256) == 4096 and topk_mod.cols_cap_for(32768, 8, 256) == 32768


@needs_cuda
@needs_dsl_target
def test_execute_form_checks_refuse_by_name():
    """Rule 1 in both directions at the run entry: dtype, rank, device, stride, alignment, width, capacity -- refused, never converted."""
    device = torch.device("cuda")
    r = topk_mod.compile_qsa_topk_decode(cols_cap=2048, top_k=K, cluster_ctas=1, threads_per_cta=256)
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
    # the row count is capped by the launch grid (one cluster per row along grid.y): refused by name before any launch -- the
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
    assert "no mbarrier is initialised" in doc and "SUM(issuing lanes) == expected" in doc
    assert "``logits [T, n_blocks_max]`` fp32" in doc and "``counts [T]`` int32" in doc and "``block_ids [T, top_k]`` int32" in doc
    assert "STRICTLY DESCENDING" in doc and "SMALLER block ids" in doc
    assert "No performance number is claimed" in doc
    # the one trap of the contract: `counts` is the VISIBLE count; the list-length spelling is forbidden, never "either spelling"
    assert "either spelling" not in doc and "NEVER the list length" in doc, "the counts clause must name the visible count and forbid the list-length spelling"
    assert "once per CTA right after the ``counts`` read" in doc, "the PDL-sites paragraph must describe the trigger placement the code has"


def test_compile_keys_on_the_file_digest_and_the_memo_dict_key(monkeypatch):
    """``compile_qsa_topk_decode`` hands ``compile_cached`` ``template_key(globals(), {"key": <the dict key>}, "compile_qsa_topk_decode")``
    -- the file digest the module names itself (``FROST_SOURCE_DIGEST``), the tag and exactly the plain tuple the module dict memoises
    under -- with the site's symbol and options; a repeat request is served by the dict (no second ``compile_cached`` call).  Without the
    digest ``template_key`` returns None and ``compile_cached`` silently bypasses the persistent cache (every process re-traces)."""
    calls = []
    monkeypatch.setattr(topk_mod, "compile_cached", lambda fn, *a, **k: calls.append(k) or object())
    monkeypatch.setattr(topk_mod, "compiled_cache", {})
    monkeypatch.setattr(topk_mod, "current_device", lambda: 0)
    monkeypatch.setattr(topk_mod, "compute_capability", lambda device: (10, 7))
    monkeypatch.setattr(topk_mod, "cutedsl_arch_requirement_error", lambda cc_: None)
    monkeypatch.setattr(topk_mod, "shared_memory_per_block_optin", lambda device: 1 << 20)
    assert topk_mod.FROST_SOURCE_DIGEST == hashlib.sha256(Path(topk_mod.__file__).read_bytes()).hexdigest()[:16]
    r = topk_mod.compile_qsa_topk_decode(cols_cap=4096, top_k=K, cluster_ctas=8, threads_per_cta=256, use_pdl=True)
    (key,) = topk_mod.compiled_cache.keys()
    assert key == (4096, K, 8, 256, True, 0) and all(isinstance(v, (int, bool)) for v in key)
    assert calls[0]["cache_key"] == f"{topk_mod.FROST_SOURCE_DIGEST}|compile_qsa_topk_decode|{[('key', key)]!r}"
    assert calls[0]["cache_key"] == cc.template_key(vars(topk_mod), {"key": key}, "compile_qsa_topk_decode")
    assert (calls[0]["symbol"], calls[0]["options"]) == ("cudnn_qsa_topk_decode", "--enable-tvm-ffi")
    assert r.compiled is topk_mod.compiled_cache[key]
    assert topk_mod.compile_qsa_topk_decode(cols_cap=4096, top_k=K, cluster_ctas=8, threads_per_cta=256, use_pdl=True).compiled is r.compiled
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
    assert facade.qsa_topk_decode(logits, counts, via_facade, cluster_ctas=_cluster_sizes()[-1], threads_per_cta=256) is via_facade
    torch.cuda.synchronize()
    assert torch.equal(via_facade, _check(logits, counts, K, cluster_ctas=_cluster_sizes()[-1], threads_per_cta=256))


_RENDER_PROBE = textwrap.dedent("""
    import glob, os, re, sys
    import cutlass, cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_compact_tensor
    from cudnn.gated_attention_block.kernels import qsa_topk_decode as M
    arch, dump, C, thr, pdl = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
    cap = M.cols_cap_for(32768, C, thr)
    cute.compile(M.qsa_topk_decode_launch, M._fake_rows(cutlass.Float32),
                 make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=4),
                 M._fake_rows(cutlass.Int32), cutlass.Int32(1), cap, 512, C, thr, pdl, M.smem_bytes_for(cap, C, thr),
                 make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-ptx")
    ptx = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    txt = open(ptx[-1]).read()
    for k in ("ld.shared::cluster", "st.shared::cluster", "barrier.cluster.arrive", "barrier.cluster.arrive.relaxed", "barrier.cluster.wait",
              "match.any.sync", "atom.shared", "vote.sync.ballot", "griddepcontrol.wait", "griddepcontrol.launch_dependents", "mbarrier",
              ".local", "lanemask_lt"):
        print("COUNT", k, len(re.findall(re.escape(k), txt)))
    """)


def _arch_known(arch: str) -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string(arch)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("cluster_ctas,threads,use_pdl", [(8, 256, False), (8, 256, True), (1, 1024, False)])
def test_sm_107a_rendering_carries_the_cluster_barriers_dsmem_traffic_and_pdl_pair_of_the_tables(tmp_path, cluster_ctas, threads, use_pdl):
    """A host trace-compile for sm_107a (no GPU): the PTX holds exactly the structure the barrier table states -- five cluster
    arrive / wait pairs (four passes + the totals exchange) and ``4 x C`` DSMEM histogram loads plus one DSMEM store at ``C > 1``,
    none of them at ``C = 1``; no RELAXED cluster arrive anywhere (the merge rests on the release form); the griddepcontrol pair iff
    ``use_pdl``; no mbarrier; no local-memory spill; no ``%lanemask_lt`` read (the mask is derived)."""
    if not _arch_known("sm_107a"):
        pytest.skip("this CuTe DSL has no sm_107a target")
    dump = tmp_path / f"c{cluster_ctas}_t{threads}_p{int(use_pdl)}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump))
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_PROBE, "sm_107a", str(dump), str(cluster_ctas), str(threads), "1" if use_pdl else "0"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    counts = {m.group(1): int(m.group(2)) for m in re.finditer(r"COUNT (\S+) (\d+)", proc.stdout)}
    if cluster_ctas > 1:
        assert counts["barrier.cluster.arrive"] == counts["barrier.cluster.wait"] == topk_mod.PASSES + 1, counts
        assert counts["ld.shared::cluster"] == topk_mod.PASSES * cluster_ctas and counts["st.shared::cluster"] == 1, counts
    else:
        assert counts["barrier.cluster.arrive"] == counts["ld.shared::cluster"] == counts["st.shared::cluster"] == 0, counts
    # the histogram merge rests on the RELEASE form of the cluster arrive (the DSL's default; `cluster_arrive_relaxed` is a separate
    # API): the relaxed form would race the DSMEM reads -- pinned absent whatever the cluster size
    assert counts["barrier.cluster.arrive.relaxed"] == 0, counts
    assert counts["lanemask_lt"] == 0, "the below-me warp mask is derived arithmetically (no %lanemask_lt read): one DSL intrinsic less"
    assert counts["match.any.sync"] > 0 and counts["atom.shared"] > 0 and counts["vote.sync.ballot"] > 0, counts
    assert counts["griddepcontrol.wait"] == counts["griddepcontrol.launch_dependents"] == (1 if use_pdl else 0), counts
    assert counts["mbarrier"] == 0 and counts[".local"] == 0, counts
