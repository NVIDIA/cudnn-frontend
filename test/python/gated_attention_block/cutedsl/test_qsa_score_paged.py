# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``kernels/qsa_score_paged.py`` -- the decode indexer's scorer over the PAGED compressed-key cache -- and its DSL-free host entry
``gated_attention_block/qsa_select_paged.py`` (the standalone selector + the kernel's host-side facts).

Host cells (any machine): the MMA row map of the (s, h) query rows (a bijection onto the padded rows; the head-sum geometry per
head count), the knob's default rule at both sides of every threshold, the compile-key envelope (typed ``ValueError``), the byte model, the
compile site's persistent-cache key (the module digest), the Rule 7 gates (the DSL floor, the ``sm_107a`` target, the radix top-k's
cc 9.0 and PDL's cc 9.0 -- each a ``NotImplementedError`` naming the version or the device, raised BEFORE any compile), the e4m3 pair
refused BY NAME, the form checks naming their fault; with a CUDA device: the front door's import-time contract in a FRESH process (the
DSL modeled as the public 4.6.2 wheel: the typed decline, the kernel module never imported) and the launch-stream resolution with both
kernels stubbed (a ``torch.cuda.Stream`` / a raw handle / ``None``: every torch operation on the launch stream).

Scorer cells (ANY CuTe DSL device from sm_80 -- the kernel is one warp-level ``mma.sync.m16n8k16`` per 8 keys x 16 D, loads,
shuffles, no barrier): the fp32 logits of every
visible column against an fp64 re-derivation of the indexer score on the SAME bf16 inputs gathered through the table
(``relu(q_h . kbar_j)`` summed over heads, times the scale), the per-row counts against the QSA oracle's count rule, the
columns at or past a row's count NEVER written (a sentinel proves it), two launches bitwise, the ``units_per_warp`` knob and the
PDL arm bitwise the plain launch, the degenerate matrix of the module docstring (a padding sequence, a sequence shorter than the
step, a visible count inside a page, a ``-1`` / out-of-range table page under a visible block and past the visible range), at
``n_blocks_max`` {8, 520, 4096, 8192, 32768} x rows 1..4 x page sizes {16, 32, 64}; every branch of the row map (``H_i`` 1 / 2 / 3 /
8 / 16 / 32, several m-tiles, a query spanning two tiles) against the same reference; a fresh variant reaching the compiled-plan cache.

Selector cells (cc 9.0+: the radix top-k): the selected SET of every row equals ``torch.topk`` of the fp64 reference up to
ties at the k-th boundary (the dense decode selector's oracle), the ids ASCENDING with ``-1`` past ``block_lens``, the identity
region (fewer than ``top_k`` visible blocks selects every one), two calls BITWISE, preallocated outputs used in place, the selection
on a side stream with the ambient stream parked bitwise a synchronised control; the agreement with the dense-cache selector runs
where that module is present (probed by name, skipped by name otherwise).
"""

import hashlib
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cudnn.frost import compiled_cache as cc
from cudnn.frost.buffers import cutedsl_requirement_error

pytestmark = pytest.mark.L0

# Rule 7, the module's own gate: the KERNEL module imports the DSL at import time (like the tile_dsl primitives), which a DSL below
# FROST's floor does not have -- so the version gate runs BEFORE the kernel import and the module SKIPS with the installed and the
# required version named instead of dying at collection.  The front door imports no DSL; its import-time contract below the floor is
# pinned by the fresh-process cell, which models the floor in a child interpreter.
requirement_error = cutedsl_requirement_error("the paged decode scorer's tests (kernels/qsa_score_paged.py)")
if requirement_error is not None:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block import qsa_select_paged as fd  # noqa: E402  (the DSL-free front door: the selector + the host-side facts)
from cudnn.gated_attention_block.kernels import qsa_score_paged as sp  # noqa: E402  (the kernel module: the compile + run entries)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import full_block_ids  # noqa: E402

D, H, BS = fd.INDEX_HEAD_DIM, 4, fd.BLOCK_SIZE
TOP_K = fd.QSA_TOP_K
SENTINEL = 1.5e30  # a finite magnitude no logit reaches: a column still holding it was never written
LOGIT_ATOL = 1e-4  # the fp32 kernel vs the fp64 reference: O(1) scores (scale 1 / sqrt(128) on 128-term dots of N(0, 1) bf16 values)
LOGIT_RTOL = 1e-5
TIE_TOL = 1e-4  # the set oracle's tie window at the k-th boundary (the dense decode selector's test constant)
_E4M3 = torch.float8_e4m3fn
_DEV = torch.device("cuda") if torch.cuda.is_available() else None
_KERNEL_MODULE = "cudnn.gated_attention_block.kernels.qsa_score_paged"


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def _scorer_reason():
    if not torch.cuda.is_available():
        return "the scorer kernel needs a CUDA device"
    return sp.dsl_decline_reason()


_SCORER_WHY = _scorer_reason()
_TOPK_WHY = None if (_cc() is not None and _cc() >= (9, 0)) else f"the radix top-k kernel needs cc 9.0+ (found {_cc()})"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is needed for the input tensors")
needs_scorer = pytest.mark.skipif(_SCORER_WHY is not None, reason=_SCORER_WHY or "")
needs_selector = pytest.mark.skipif(_SCORER_WHY is not None or _TOPK_WHY is not None, reason=_SCORER_WHY or _TOPK_WHY or "")
needs_pdl = pytest.mark.skipif(_cc() is None or _cc() < (9, 0), reason=f"griddepcontrol (PDL) needs cc 9.0+ (found {_cc()})")


def _stream():
    return torch.cuda.current_stream().cuda_stream


class _Case:
    """One decode step over a paged cache: ``batch`` sequences, ``s_q`` rows each, a shuffled page table into a pool with spare
    pages, and the fp64 reference through the SAME table (independent of the kernel's addressing)."""

    def __init__(self, batch, s_q, n_blocks_max, page_size, kv_lens, *, seed, h=H, extra_pages=5):
        self.batch, self.s_q, self.n_blocks_max, self.page_size, self.h = batch, s_q, n_blocks_max, page_size, h
        self.rpp = page_size // BS
        self.max_pages = -(-n_blocks_max // self.rpp)
        g = torch.Generator(device=_DEV).manual_seed(seed)
        self.qi = torch.randn(batch * s_q, h, D, dtype=torch.bfloat16, device=_DEV, generator=g)
        self.num_pages = batch * self.max_pages + extra_pages
        self.pool = torch.randn(self.num_pages, self.rpp, 1, D, dtype=torch.bfloat16, device=_DEV, generator=g)
        self.table = torch.randperm(self.num_pages, device=_DEV, generator=g)[: batch * self.max_pages].view(batch, self.max_pages).to(torch.int32).contiguous()
        self.kv_lens = torch.tensor(list(kv_lens), dtype=torch.int32, device=_DEV)
        assert self.kv_lens.numel() == batch

    def dense(self, table=None):
        """``[B, n_blocks_max, D]`` bf16: the compressed keys gathered through the table (a ``-1`` / out-of-range page -> zeros)."""
        table = self.table if table is None else table
        idx = table.long()
        ok = (idx >= 0) & (idx < self.num_pages)
        rows = self.pool.select(2, 0)[idx.clamp(0, self.num_pages - 1).view(-1)].view(self.batch, self.max_pages, self.rpp, D)
        rows = torch.where(ok.view(self.batch, self.max_pages, 1, 1), rows, torch.zeros_like(rows))
        return rows.reshape(self.batch, self.max_pages * self.rpp, D)[:, : self.n_blocks_max]

    def positions(self):
        return (self.kv_lens.long() - self.s_q)[:, None] + torch.arange(self.s_q, device=_DEV)[None, :]

    def n_vis(self):
        return torch.div(self.positions() + 1, BS, rounding_mode="floor").clamp(0, self.n_blocks_max)

    def reference(self, scale, table=None):
        """fp64 ``[B, s_q, n_blocks_max]``, ``-inf`` at or past the row's visible count."""
        q = self.qi.view(self.batch, self.s_q, self.h, D).double()
        scores = torch.relu(torch.einsum("brhd,bnd->brhn", q, self.dense(table).double())).sum(dim=2) * scale
        visible = torch.arange(self.n_blocks_max, device=_DEV).view(1, 1, -1) < self.n_vis().view(self.batch, self.s_q, 1)
        return scores.masked_fill(~visible, float("-inf"))

    def outputs(self, with_lens=True):
        t = self.batch * self.s_q
        logits = torch.full((t, self.n_blocks_max), SENTINEL, dtype=torch.float32, device=_DEV)
        counts = torch.full((t,), -7, dtype=torch.int32, device=_DEV)
        lens = torch.full((t,), -7, dtype=torch.int32, device=_DEV) if with_lens else None
        return logits, counts, lens

    def run(self, recipe, *, table=None, top_k=TOP_K, scale=None, with_lens=True):
        logits, counts, lens = self.outputs(with_lens)
        sp.run_qsa_score_paged(
            recipe,
            self.qi,
            self.pool,
            self.table if table is None else table,
            self.kv_lens,
            logits,
            counts,
            lens,
            top_k=top_k,
            scale=D**-0.5 if scale is None else scale,
            stream=_stream(),
        )
        torch.cuda.synchronize()
        return logits, counts, lens


def _bits(x):
    return x.view(torch.int32)


def _assert_scorer(case, logits, counts, lens, *, top_k=TOP_K, scale=None, table=None):
    """The scorer's contract on one case: counts = the oracle's visible count per row, block_lens = min(top_k, counts), every visible
    column within the fp32 budget of the fp64 reference, every other column still the sentinel."""
    scale = D**-0.5 if scale is None else scale
    n_vis = case.n_vis()
    _, want_counts = full_block_ids(case.positions(), case.n_blocks_max, BS)  # the oracle's count rule at the capacity: min(capacity, floor((p + 1) / 4))
    assert torch.equal(counts, want_counts.view(-1).to(torch.int32)), "counts differ from the oracle's count rule"
    assert torch.equal(counts, n_vis.view(-1).to(torch.int32))
    if lens is not None:
        assert torch.equal(lens, torch.minimum(counts, torch.full_like(counts, top_k))), "block_lens differs from min(top_k, counts)"
    ref = case.reference(scale, table)
    for b in range(case.batch):
        for s in range(case.s_q):
            n = int(n_vis[b, s])
            row = logits[b * case.s_q + s]
            if n:
                torch.testing.assert_close(
                    row[:n].double(), ref[b, s, :n], atol=LOGIT_ATOL, rtol=LOGIT_RTOL, msg=f"sequence {b} row {s}: logits off the fp64 reference"
                )
            assert bool((row[n:] == SENTINEL).all()), f"sequence {b} row {s}: a column at or past the visible count {n} was written"


def _assert_rows(block_ids, scores, block_lens, ref, top_k, tie_tol=TIE_TOL):
    """Per row of ``ref [rows, n_blocks_max]`` (the dense decode selector's oracle): the valid count equals ``block_lens`` =
    ``min(top_k, visible)``, ids are unique local ids below the visible count, the scores are the reference scores of the selected
    ids (fp32 of an fp64 value), and the SET equals torch.topk's up to flips within ``tie_tol`` of the k-th reference score."""
    rows, n_blocks_max = ref.shape
    assert block_ids.shape == scores.shape == (rows, top_k)
    assert block_ids.dtype == torch.int32 and scores.dtype == torch.float32 and block_lens.dtype == torch.int32
    valid = block_ids >= 0
    visible = torch.isfinite(ref).sum(dim=-1)
    assert torch.equal(valid.sum(dim=-1).to(torch.int32), block_lens), "the valid prefix differs from block_lens"
    assert torch.equal(block_lens.long(), visible.clamp(max=top_k)), "block_lens differs from min(top_k, visible)"
    prefix = torch.arange(top_k, device=ref.device)[None, :] < block_lens[:, None].long()
    assert torch.equal(valid, prefix), "the valid ids are not a prefix followed by -1"
    assert bool(((block_ids < visible[:, None]) | ~valid).all()), "an id at or past the row's visible count"
    assert bool(torch.isneginf(scores[~valid]).all())
    row_idx = torch.arange(rows, device=ref.device).unsqueeze(1).expand_as(block_ids)
    gathered = ref[row_idx[valid], block_ids[valid].long()]
    torch.testing.assert_close(scores[valid].double(), gathered, atol=tie_tol, rtol=tie_tol)
    in_actual = torch.zeros_like(ref, dtype=torch.bool)
    in_actual[row_idx[valid], block_ids[valid].long()] = True
    assert torch.equal(in_actual.sum(dim=-1), valid.sum(dim=-1)), "a row lists a block twice"
    k_eff = min(top_k, n_blocks_max)
    ref_topk = torch.topk(ref, k_eff, dim=-1)
    exp_valid = torch.isfinite(ref_topk.values)
    in_expected = torch.zeros_like(in_actual)
    rows_k = torch.arange(rows, device=ref.device).unsqueeze(1).expand(-1, k_eff)
    in_expected[rows_k[exp_valid], ref_topk.indices[exp_valid]] = True
    kth = torch.where(exp_valid, ref_topk.values, torch.full_like(ref_topk.values, float("inf"))).min(dim=-1).values
    differs = in_actual ^ in_expected
    if bool(differs.any()):
        gap = (ref - kth.unsqueeze(1)).abs()[differs]
        assert bool((gap <= tie_tol).all()), f"{int(differs.sum())} ids differ from torch.topk beyond a tie (max gap {float(gap.max()):.3e})"


def _assert_canonical(block_ids, block_lens):
    """The selected ids come back in ASCENDING order (the deterministic radix kernel's canonical order), the padding last."""
    for r in range(block_ids.shape[0]):
        n = int(block_lens[r])
        ids = block_ids[r, :n]
        assert bool((ids[1:] > ids[:-1]).all()), f"row {r}: the selected ids are not strictly ascending"
        assert bool((block_ids[r, n:] == -1).all()), f"row {r}: a non -1 entry past block_lens"


def _step_kv_lens(n_blocks_max, rows, batch=4):
    """Step-0 positions covering every residue modulo the block size, the last sequence seeing every block of the cache."""
    base = BS * n_blocks_max - rows - 3
    return [max(0, base + r) + rows for r in range(batch)]


# --------------------------------------------------------------------------------------------- host cells


def test_query_row_map_is_a_bijection_and_the_head_sum_geometry_follows_the_head_count():
    """The (s, h) -> (m-tile, padded row) map the kernel loads its A fragments by: every (s, h) lands on exactly one padded row, every
    other padded row is a padding row (``row_of`` None), the inverse agrees; the heads of one query sit on the lane's two rows
    (``g``, ``g + 8``) and ``group`` consecutive ``g`` values -- ``stages`` butterfly shuffles (H_i 4: ONE); a query wider than a
    tile spans ``tiles_per_query`` tiles.  The facts live in the DSL-free front door and the kernel module uses the very same
    objects (one definition, Rule 7's placement)."""
    for h_i, s_q, want in (
        (4, 1, dict(padded=4, tile_rows=4, queries_per_tile=4, tiles_per_query=1, m_tiles=1, group=2, stages=1)),
        (4, 4, dict(padded=4, tile_rows=4, queries_per_tile=4, tiles_per_query=1, m_tiles=1, group=2, stages=1)),
        (4, 5, dict(padded=4, tile_rows=4, queries_per_tile=4, tiles_per_query=1, m_tiles=2, group=2, stages=1)),
        (3, 2, dict(padded=4, tile_rows=4, queries_per_tile=4, tiles_per_query=1, m_tiles=1, group=2, stages=1)),
        (1, 20, dict(padded=1, tile_rows=1, queries_per_tile=16, tiles_per_query=1, m_tiles=2, group=0, stages=0)),
        (2, 3, dict(padded=2, tile_rows=2, queries_per_tile=8, tiles_per_query=1, m_tiles=1, group=1, stages=0)),
        (8, 2, dict(padded=8, tile_rows=8, queries_per_tile=2, tiles_per_query=1, m_tiles=1, group=4, stages=2)),
        (16, 3, dict(padded=16, tile_rows=16, queries_per_tile=1, tiles_per_query=1, m_tiles=3, group=8, stages=3)),
        (17, 2, dict(padded=32, tile_rows=16, queries_per_tile=1, tiles_per_query=2, m_tiles=4, group=8, stages=3)),
        (64, 1, dict(padded=64, tile_rows=16, queries_per_tile=1, tiles_per_query=4, m_tiles=4, group=8, stages=3)),
    ):
        rm = fd.query_row_map(s_q, h_i)
        assert rm.h_i == h_i and rm.s_q == s_q and {k: getattr(rm, k) for k in want} == want, (h_i, s_q, rm)
        seen = {}
        for s in range(s_q):
            for h in range(h_i):
                t, r = rm.tile_row(s, h)
                assert 0 <= t < rm.m_tiles and 0 <= r < 16
                assert rm.row_of(t, r) == (s, h), (h_i, s_q, s, h, t, r, rm.row_of(t, r))
                seen[(t, r)] = (s, h)
        assert len(seen) == s_q * h_i, "two (s, h) share a padded row"
        for t in range(rm.m_tiles):
            for r in range(16):
                if (t, r) not in seen:
                    assert rm.row_of(t, r) is None, (h_i, s_q, t, r)
        if rm.tile_rows > 1:
            # one query's heads: the lane's two rows (g and g + 8) x the `group` consecutive g values x its tiles -- nothing else
            for s in range(s_q):
                rows = {rm.tile_row(s, h) for h in range(h_i)}
                gs = {r % 8 for _, r in rows}
                tiles = {t for t, _ in rows}
                assert len(tiles) == rm.tiles_per_query and max(gs) - min(gs) < rm.group and min(gs) % rm.group == 0, (h_i, s_q, s, rows)
    with pytest.raises(ValueError, match="s_q x h_i"):
        fd.query_row_map(17, 4) and fd.validate_params(h_i=4, s_q=17, page_size=16, units_per_warp=1)
    assert [fd.padded_heads(h) for h in (1, 2, 3, 4, 5, 8, 9, 16, 17, 64)] == [1, 2, 4, 4, 8, 8, 16, 16, 32, 64]
    assert [fd.keys_per_warp(u) for u in fd.UNITS_PER_WARP_CHOICES] == [32, 64, 128, 256]
    assert [fd.blocks_per_cta(u) for u in fd.UNITS_PER_WARP_CHOICES] == [128, 256, 512, 1024]
    assert fd.ROWS_PER_UNIT == fd.KEYS_PER_NTILE * fd.NTILES_PER_UNIT == 32 and fd.K_STEPS == 8 and fd.MAX_QUERY_ROWS == 64
    assert sp.query_row_map is fd.query_row_map and sp.blocks_per_cta is fd.blocks_per_cta and sp.validate_params is fd.validate_params
    assert sp.keys_per_warp is fd.keys_per_warp and sp.THREADS == fd.THREADS == 128 and sp.INDEX_HEAD_DIM == D


def test_default_units_is_one_everywhere_and_the_rule_keeps_its_arguments():
    """The knob's default (fit on the 2026-10-21 decode ranking): ONE unit per warp on every cell -- the function keeps the step's key
    count, the SM count and the rows per sequence for a refit; the one-argument call still resolves."""
    for sms in (212, 204):
        for b, n, s_q in ((1, 2048, 1), (1, 32768, 1), (4, 32768, 4), (32, 2048, 1), (32, 32768, 4), (32, 32768, 1), (1, 10**9, 4)):
            assert fd.default_units_per_warp(n, b, sms, s_q) == fd.UNITS_DEFAULT == 1, (sms, b, n, s_q)
    assert fd.default_units_per_warp(8) == 1 and fd.default_units_per_warp(32768, 32) == 1 and fd.UNITS_DEFAULT in fd.UNITS_PER_WARP_CHOICES


def test_validate_params_rejects_outside_the_envelope_and_accepts_inside():
    with pytest.raises(ValueError, match="s_q x h_i"):
        fd.validate_params(h_i=4, s_q=17, page_size=16, units_per_warp=1)
    with pytest.raises(ValueError, match="s_q x h_i"):
        fd.validate_params(h_i=64, s_q=2, page_size=16, units_per_warp=1)
    for page in (4, 12, 24, 20):
        with pytest.raises(ValueError, match="page_size must be a power of two >= 8"):
            fd.validate_params(h_i=4, s_q=1, page_size=page, units_per_warp=1)
    with pytest.raises(ValueError, match="units_per_warp must be one of"):
        fd.validate_params(h_i=4, s_q=1, page_size=16, units_per_warp=3)
    with pytest.raises(ValueError, match="h_i must be >= 1"):
        fd.validate_params(h_i=0, s_q=1, page_size=16, units_per_warp=1)
    with pytest.raises(ValueError, match="s_q must be >= 1"):
        fd.validate_params(h_i=4, s_q=0, page_size=16, units_per_warp=1)
    with pytest.raises(ValueError, match=r"s_q x h_i = 12 x 5 pads to 96"):
        fd.validate_params(h_i=5, s_q=12, page_size=16, units_per_warp=1)  # 60 (s, h) rows, but 5 heads pad to 8: past the envelope
    with pytest.raises(ValueError, match="units_per_warp must be one of"):
        fd.validate_params(h_i=4, s_q=1, page_size=16, units_per_warp=16)
    for depth in (3, 16, 0):
        with pytest.raises(ValueError, match="ring_depth must be one of"):
            fd.validate_params(h_i=4, s_q=1, page_size=16, units_per_warp=1, ring_depth=depth)
    for depth in fd.RING_DEPTH_CHOICES:
        fd.validate_params(h_i=4, s_q=1, page_size=16, units_per_warp=1, ring_depth=depth)
    for h_i, s_q, page, units in ((4, 4, 16, 1), (4, 16, 64, 4), (64, 1, 8, 2), (8, 8, 32, 2), (16, 4, 256, 1), (3, 16, 16, 8), (5, 8, 16, 1)):
        fd.validate_params(h_i=h_i, s_q=s_q, page_size=page, units_per_warp=units)


def test_moved_bytes_counts_the_streamed_rows_and_the_written_logits():
    """Two sequences of two rows: the compressed rows of a sequence are counted once (its last row's count), the logits per row."""
    n = fd.moved_bytes([9, 10, 0, 0], s_q=2, h_i=4, n_blocks_max=64, page_size=16, units_per_warp=1)
    rows = 10 * D * 2  # the first sequence streams 10 compressed rows (the padding sequence none)
    logits = (9 + 10) * 4
    queries = 1 * 16 * D * 2  # one live CTA (128 rows per CTA) stages the padded 16-row query tile (2 x 4 rows pad to one m-tile)
    table = -(-10 // 4) * 4  # one word per page
    assert n == rows + logits + queries + table + 2 * 4 * 4
    n2 = fd.moved_bytes([200, 200], s_q=1, h_i=4, n_blocks_max=512, page_size=16, units_per_warp=1)
    assert n2 == 200 * D * 2 * 2 + 200 * 4 * 2 + 2 * 16 * D * 2 * 2 + -(-200 // 4) * 4 * 2 + 2 * 2 * 4  # 2 live CTAs of 128 rows per sequence
    with pytest.raises(ValueError, match="not a multiple of s_q"):
        fd.moved_bytes([1, 2, 3], s_q=2, h_i=4, n_blocks_max=64, page_size=16)


def test_compile_site_keys_on_the_module_digest():
    """The persistent compiled-plan cache needs a key: the kernel module names its own source identity (``FROST_SOURCE_DIGEST``, the
    file's sha256 prefix, the neighbours' line), so ``template_key`` yields one for the compile site -- ``None`` would send every compile
    down the ``bypassed`` branch (a plain trace per process, no persistent object, no in-process memo).  The site passes that key and
    its symbol and leaves no bare ``cute.compile(``."""
    assert sp.FROST_SOURCE_DIGEST == hashlib.sha256(Path(sp.__file__).read_bytes()).hexdigest()[:16]
    key = (4, 1, 16, 1, True, False, 0)
    assert cc.template_key(vars(sp), {"key": key}, "compile_qsa_score_paged") == f"{sp.FROST_SOURCE_DIGEST}|compile_qsa_score_paged|[('key', {key!r})]"
    src = Path(sp.__file__).read_text()
    assert "cute.compile(" not in src and 'symbol="cudnn_block_qsa_score_paged"' in src and '"compile_qsa_score_paged")' in src


@needs_cuda
def test_front_door_declines_by_version_in_a_fresh_process_without_importing_the_kernel_module():
    """Rule 7 at the DOCUMENTED import path, in a fresh interpreter (in this process the kernel module is already imported, so an in-process
    check would pass for the wrong reason).  The installed DSL is modeled as the public 4.6.2 wheel two ways at once -- the version
    discovery answers 4.6.2 AND ``cutlass.experimental`` (the 4.7.0-only API every tile_dsl helper imports at module scope) is made
    unimportable -- then the front door is imported and the selector called on real CUDA operands: the import touched neither the kernel
    module nor ``cudnn.frost.tile_dsl``, the call raises the typed ``NotImplementedError`` naming the floor and the found version, and
    afterwards the kernel module, ``cudnn.frost.tile_dsl`` and the radix top-k's package are still not imported."""
    probe = r"""
import importlib.abc
import sys


class _NoExperimental(importlib.abc.MetaPathFinder):
    # the public 4.6.2 wheel has no cutlass.experimental: an import of it from inside the DSL is the untyped failure Rule 7 forbids
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cutlass.experimental" or fullname.startswith("cutlass.experimental."):
            raise ModuleNotFoundError("No module named 'cutlass.experimental' (modeled: the public 4.6.2 wheel)")
        return None


sys.meta_path.insert(0, _NoExperimental())
import torch
import cudnn.frost.buffers as buffers

buffers._DSL_STATE = (True, ("nvidia-cutlass-dsl", "4.6.2"))
from cudnn.gated_attention_block import qsa_select_paged as fd

KERNEL = "cudnn.gated_attention_block.kernels.qsa_score_paged"
assert KERNEL not in sys.modules, "importing the front door imported the kernel module"
assert "cudnn.frost.tile_dsl" not in sys.modules, "importing the front door imported tile_dsl"
qi = torch.zeros(1, 4, 128, dtype=torch.bfloat16, device="cuda")
pool = torch.zeros(3, 4, 1, 128, dtype=torch.bfloat16, device="cuda")
table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
kv_lens = torch.tensor([9], dtype=torch.int32, device="cuda")
try:
    fd.qsa_select_decode_paged(qi, pool, table, kv_lens, seq_q=1)
except NotImplementedError as error:
    assert "qsa_select_decode_paged requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2" in str(error), str(error)
else:
    raise AssertionError("the selector ran on a sub-floor DSL")
assert KERNEL not in sys.modules, "the decline imported the kernel module"
assert "cudnn.frost.tile_dsl" not in sys.modules, "the decline imported tile_dsl"
assert not any(m.startswith("cudnn.deepseek_sparse_attention") for m in sys.modules), "the decline imported the radix top-k's package"
print("@@FRONT_DOOR_OK")
"""
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, f"rc {run.returncode}\n{run.stderr[-3000:]}"
    assert "@@FRONT_DOOR_OK" in run.stdout, run.stdout[-2000:]


@needs_cuda
def test_rule7_gates_run_before_any_compile(monkeypatch):
    """The DSL floor, then the sm_107a target, then the radix top-k's and PDL's cc 9.0: each a NotImplementedError that names the
    version or the device, raised by the compile entry (the kernel module's own gate) and by the selector (the front door's gate)
    BEFORE any kernel is compiled or imported -- each module's binding patched, since each gates for itself."""
    case = _Case(1, 1, 8, 16, [9], seed=1)
    kw = dict(seq_q=1)
    for mod in (sp, fd):
        monkeypatch.setattr(mod, "cutedsl_requirement_error", lambda what: f"{what} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    with pytest.raises(NotImplementedError, match="qsa_score_paged requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        sp.compile_qsa_score_paged(h_i=H, s_q=1, page_size=16)
    with pytest.raises(NotImplementedError, match="qsa_select_decode_paged requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, **kw)
    for mod in (sp, fd):
        monkeypatch.setattr(mod, "cutedsl_requirement_error", lambda what: None)
        monkeypatch.setattr(
            mod, "cutedsl_arch_requirement_error", lambda cc: "SM107 requires a CuTe DSL build supporting sm_107a; found 4.7.0 without that target"
        )
    with pytest.raises(NotImplementedError, match="sm_107a"):
        sp.compile_qsa_score_paged(h_i=H, s_q=1, page_size=16)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, **kw)
    for mod in (sp, fd):
        monkeypatch.setattr(mod, "cutedsl_arch_requirement_error", lambda cc: None)
    monkeypatch.setattr(sp, "compute_capability", lambda dev: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    with pytest.raises(NotImplementedError, match=r"cc 9\.0 or later; found cc 8\.0"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, **kw)
    with pytest.raises(NotImplementedError, match=r"griddepcontrol.*cc 8\.0"):
        sp.compile_qsa_score_paged(h_i=H, s_q=1, page_size=16, use_pdl=True)


@needs_cuda
def test_e4m3_pair_is_refused_by_name_and_a_mixed_pair_is_a_form_error():
    case = _Case(1, 1, 8, 16, [9], seed=2)
    with pytest.raises(NotImplementedError, match="e4m3"):
        fd.qsa_select_decode_paged(case.qi.to(_E4M3), case.pool.to(_E4M3), case.table, case.kv_lens, seq_q=1)
    with pytest.raises(ValueError, match="both be bfloat16"):
        fd.qsa_select_decode_paged(case.qi.to(_E4M3), case.pool, case.table, case.kv_lens, seq_q=1)
    with pytest.raises(ValueError, match="both be bfloat16"):
        fd.qsa_select_decode_paged(case.qi, case.pool.to(_E4M3), case.table, case.kv_lens, seq_q=1)
    with pytest.raises(ValueError, match="both be bfloat16"):
        fd.qsa_select_decode_paged(case.qi.half(), case.pool, case.table, case.kv_lens, seq_q=1)


@needs_cuda
def test_selector_form_checks_name_the_fault():
    """Every form check fires ahead of the gates and the compile (no kernel runs for a malformed call)."""
    case = _Case(2, 2, 16, 16, [9, 40], seed=3)
    kw = dict(seq_q=2)
    with pytest.raises(ValueError, match=r"head_dim 128"):
        fd.qsa_select_decode_paged(case.qi[..., :64], case.pool, case.table, case.kv_lens, **kw)
    with pytest.raises(ValueError, match="T must be B x seq_q"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=3)
    with pytest.raises(ValueError, match="seq_q must be >= 1"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=0)
    with pytest.raises(ValueError, match="multiple of 8"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, n_blocks_max=12, **kw)
    with pytest.raises(ValueError, match="top_k must be in"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, top_k=0, **kw)
    with pytest.raises(ValueError, match="top_k must be in"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, top_k=fd.TOP_K_KERNEL_MAX + 1, **kw)
    with pytest.raises(ValueError, match="scale must be positive"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, scale=0.0, **kw)
    with pytest.raises(ValueError, match="block_table must be an int32"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table.to(torch.int64), case.kv_lens, **kw)
    with pytest.raises(ValueError, match="units_per_warp must be one of"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, units_per_warp=3, **kw)
    with pytest.raises(ValueError, match="one compressed row"):
        fd.qsa_select_decode_paged(case.qi, case.pool.expand(-1, -1, 2, -1), case.table, case.kv_lens, **kw)
    t, n = case.qi.shape[0], case.n_blocks_max
    buf = torch.empty(t * n + 4, dtype=torch.float32, device=_DEV)
    with pytest.raises(ValueError, match="32-byte aligned"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, logits_out=buf[4:].view(t, n), **kw)
    with pytest.raises(ValueError, match="logits_out must be a contiguous"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, logits_out=torch.empty(t, n + 8, device=_DEV)[:, :n], **kw)
    with pytest.raises(ValueError, match="counts_out must be a contiguous"):
        fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, counts_out=torch.empty(t, dtype=torch.int64, device=_DEV), **kw)


@needs_cuda
def test_selector_resolves_one_launch_stream_and_runs_every_torch_op_on_it(monkeypatch):
    """Rule 5 by construction, on the host with both kernels STUBBED (any CUDA device; the cc 9.0 gate modeled open): ``stream`` as a
    ``torch.cuda.Stream``, as its raw handle and as ``None`` -- the selector resolves ONE torch stream first; the scorer's launch and the
    top-k wrapper both receive that stream's raw handle, and torch's CURRENT stream at both calls (where the output allocations and the
    wrapper's own are made) IS the launch stream -- torch's current stream for ``None`` -- and the context is left afterwards."""
    import cudnn.deepseek_sparse_attention.indexer_top_k as topk_mod

    case = _Case(1, 1, 8, 16, [9], seed=15)
    seen = []

    def fake_compile(**kw):
        seen.append(("compile", kw))
        return object()

    def fake_run(recipe, qi, kbar, block_table, kv_lens, logits, counts, block_lens, *, top_k, scale, stream):
        seen.append(("scorer", int(stream), torch.cuda.current_stream(qi.device).cuda_stream, logits.shape, counts.shape, block_lens.shape))

    def fake_topk(logits, counts, top_k, *, next_n, return_val, stream, tie_break, deterministic):
        seen.append(("topk", int(stream), torch.cuda.current_stream(logits.device).cuda_stream, next_n, return_val, tie_break, deterministic))
        rows = logits.shape[0]
        return {
            "indices": torch.full((rows, top_k), -1, dtype=torch.int32, device=logits.device),
            "values": torch.full((rows, top_k), float("-inf"), dtype=torch.float32, device=logits.device),
        }

    monkeypatch.setattr(sp, "compile_qsa_score_paged", fake_compile)
    monkeypatch.setattr(sp, "run_qsa_score_paged", fake_run)
    monkeypatch.setattr(topk_mod, "indexer_top_k_wrapper", fake_topk)
    monkeypatch.setattr(fd, "cutedsl_requirement_error", lambda what: None)
    monkeypatch.setattr(fd, "cutedsl_arch_requirement_error", lambda cc: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    ambient, side = torch.cuda.current_stream(), torch.cuda.Stream()
    assert side.cuda_stream != ambient.cuda_stream
    for arg, want in ((side, side.cuda_stream), (side.cuda_stream, side.cuda_stream), (None, ambient.cuda_stream)):
        seen.clear()
        out = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=1, stream=arg)
        assert [s[0] for s in seen] == ["compile", "scorer", "topk"], seen
        assert seen[1][1] == want and seen[1][2] == want, (arg, seen[1])
        assert seen[2][1] == want and seen[2][2] == want, (arg, seen[2])
        assert seen[2][3:] == (1, True, 0, True), "the deterministic, canonically ordered top-k"
        assert torch.cuda.current_stream() == ambient, "the launch-stream context was not left"
        assert out["block_ids"].shape == (1, TOP_K) and out["logits"].shape == (1, 8) and out["counts"].shape == (1,) and out["block_lens"].shape == (1,)
    torch.cuda.synchronize()


@needs_scorer
def test_run_form_checks_name_the_fault():
    """The kernel-level entry's checks: the artifact's fixed facts (H_i, the page's row count, the block_lens presence) and every
    operand's shape / dtype / device / stride / alignment, the table's reach, top_k, scale."""
    case = _Case(2, 2, 16, 16, [9, 40], seed=4)
    r = sp.compile_qsa_score_paged(h_i=H, s_q=2, page_size=16)
    logits, counts, lens = case.outputs()
    ok = dict(top_k=TOP_K, scale=D**-0.5, stream=_stream())
    with pytest.raises(ValueError, match=r"H_i=4"):
        sp.run_qsa_score_paged(r, case.qi[:, :2], case.pool, case.table, case.kv_lens, logits, counts, lens, **ok)
    with pytest.raises(ValueError, match="unit last stride"):
        sp.run_qsa_score_paged(r, case.qi.transpose(1, 2).contiguous().transpose(1, 2), case.pool, case.table, case.kv_lens, logits, counts, lens, **ok)
    with pytest.raises(ValueError, match="pages of 4 compressed rows"):
        sp.run_qsa_score_paged(r, case.qi, torch.empty(4, 8, 1, D, dtype=torch.bfloat16, device=_DEV), case.table, case.kv_lens, logits, counts, lens, **ok)
    with pytest.raises(ValueError, match="T must be B x s_q"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table[:1], case.kv_lens[:1], logits, counts, lens, **ok)
    with pytest.raises(ValueError, match="kv_lens must be an int32"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table, case.kv_lens.to(torch.int64), logits, counts, lens, **ok)
    with pytest.raises(ValueError, match="fewer than the"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table[:, :2].contiguous(), case.kv_lens, logits, counts, lens, **ok)
    with pytest.raises(ValueError, match="unit column stride"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table.t().contiguous().t(), case.kv_lens, logits, counts, lens, **ok)
    with pytest.raises(ValueError, match="logits must be an fp32"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table, case.kv_lens, logits.half(), counts, lens, **ok)
    with pytest.raises(ValueError, match=r"logits must be \[T="):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table, case.kv_lens, logits[:2], counts, lens, **ok)
    with pytest.raises(ValueError, match="compiled with block_lens"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table, case.kv_lens, logits, counts, None, **ok)
    with pytest.raises(ValueError, match="counts must be an int32"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table, case.kv_lens, logits, counts[:2], lens, **ok)
    with pytest.raises(ValueError, match="top_k must be in"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table, case.kv_lens, logits, counts, lens, top_k=0, scale=1.0, stream=_stream())
    with pytest.raises(ValueError, match="scale must be positive"):
        sp.run_qsa_score_paged(r, case.qi, case.pool, case.table, case.kv_lens, logits, counts, lens, top_k=4, scale=-1.0, stream=_stream())
    with pytest.raises(ValueError, match="qi must be bfloat16"):
        sp.run_qsa_score_paged(r, case.qi.half(), case.pool, case.table, case.kv_lens, logits, counts, lens, **ok)
    assert bool((logits == SENTINEL).all()) and bool((counts == -7).all()), "a refused call wrote an output"


# --------------------------------------------------------------------------------------------- the scorer (any CuTe DSL device)


@needs_scorer
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("n_blocks_max", [8, 520, 4096, 8192, 32768])
def test_scorer_matches_the_fp64_reference_and_writes_only_the_visible_columns(n_blocks_max, rows, page_size):
    """Four sequences whose step-0 positions cover every residue modulo the block size (the last sees every block of the cache), the
    knob at its default: counts = the oracle's count rule, block_lens = min(top_k, counts), every visible logit
    within the fp32 budget of the fp64 reference gathered through the SAME shuffled table, every other column untouched, and a
    second launch bitwise."""
    case = _Case(4, rows, n_blocks_max, page_size, _step_kv_lens(n_blocks_max, rows), seed=n_blocks_max + 7 * rows + page_size)
    r = sp.compile_qsa_score_paged(h_i=H, s_q=rows, page_size=page_size, units_per_warp=fd.default_units_per_warp(n_blocks_max, 4, None, rows))
    logits, counts, lens = case.run(r)
    _assert_scorer(case, logits, counts, lens)
    logits2, counts2, lens2 = case.run(r)
    assert torch.equal(_bits(logits), _bits(logits2)) and torch.equal(counts, counts2) and torch.equal(lens, lens2), "two launches differ"


@needs_scorer
def test_a_fresh_variant_reaches_the_compiled_plan_cache_not_the_bypass():
    """The dynamic half of the digest pin: a variant no other cell compiles goes through ``compile_cached`` with a key -- counted as a
    miss or a hit of the persistent cache (or a memo hit), never ``bypassed`` -- where the cache is enabled and the environment
    identifies itself; with the cache disabled the one compile is the counted bypass."""
    dev = torch.cuda.current_device()
    key = (3, 1, 8, 4, True, False, dev)
    sp.compiled_cache.pop(key, None)  # the module memo: the site must reach compile_cached
    before = dict(cc.stats())
    r = sp.compile_qsa_score_paged(h_i=3, s_q=1, page_size=8, units_per_warp=4, device=dev)
    delta = {k: v - before[k] for k, v in cc.stats().items()}
    assert r.h_i == 3 and r.rows_per_page == 2 and key in sp.compiled_cache
    persistent = cc.enabled() and "unknown" not in cc.environment_manifest(dev).values()
    if persistent:
        assert delta["bypassed"] == 0, delta
        assert delta["misses"] + delta["hits"] + delta["memo_hits"] == 1, delta
    else:
        assert delta["bypassed"] + delta["memo_hits"] == 1, delta
    case = _Case(1, 1, 8, 8, [4 * 8 + 1], seed=17, h=3)
    logits, counts, lens = case.run(r)
    _assert_scorer(case, logits, counts, lens)


@needs_scorer
def test_degenerate_rows_and_dead_pages():
    """The module docstring's matrix: a padding sequence (kv_lens 0, its table all -1: nothing written but its zero counts), a
    sequence shorter than the step (the rows at a negative position read 0), a visible count inside a page, a -1 / out-of-range page
    PAST the visible range (bitwise harmless) and UNDER a visible block (that block's logit exactly 0, every other column unchanged)."""
    rows, n_blocks_max, page = 3, 520, 16
    kv = [0, 5, 4 * 50 + 1, 4 * n_blocks_max + rows - 1, 4 * 130 + 2]  # the fourth sequence: every row sees the whole cache
    case = _Case(5, rows, n_blocks_max, page, kv, seed=5)
    case.table[0] = -1  # a padding sequence's table is never consulted
    r = sp.compile_qsa_score_paged(h_i=H, s_q=rows, page_size=page)
    logits, counts, lens = case.run(r)
    _assert_scorer(case, logits, counts, lens)
    assert counts[:rows].tolist() == [0, 0, 0] and lens[:rows].tolist() == [0, 0, 0] and bool((logits[:rows] == SENTINEL).all())
    assert counts[rows : 2 * rows].tolist() == [0, 1, 1]  # kv 5, rows at positions 2, 3, 4
    assert counts[2 * rows : 3 * rows].tolist() == [
        49,
        50,
        50,
    ]  # kv 201: positions 198 / 199 / 200 -> 49 / 50 / 50 rows, inside the 13th page (4 rows per page)
    assert counts[3 * rows : 4 * rows].tolist() == [n_blocks_max] * rows
    # sequence 4 (positions 519, 520, 521 -> 130 visible blocks = pages 0..32 of its table): a -1 page past the range is harmless
    table = case.table.clone()
    table[4, 40] = -1
    table[4, 41] = case.num_pages + 3
    logits_b, counts_b, lens_b = case.run(r, table=table)
    assert (
        torch.equal(_bits(logits), _bits(logits_b)) and torch.equal(counts, counts_b) and torch.equal(lens, lens_b)
    ), "a dead page past the visible range changed an output"
    # a -1 page and an out-of-range page UNDER visible blocks (blocks 40..43 and 44..47 of sequence 4): zero keys, logit exactly 0
    table[4, 10] = -1
    table[4, 11] = case.num_pages + 3
    logits_c, counts_c, lens_c = case.run(r, table=table)
    _assert_scorer(case, logits_c, counts_c, lens_c, table=table)
    for s in range(rows):
        row = logits_c[4 * rows + s]
        assert bool((row[40:48] == 0.0).all()), "a dead page under visible blocks did not read as zero keys"
        keep = torch.ones(n_blocks_max, dtype=torch.bool, device=_DEV)
        keep[40:48] = False
        assert torch.equal(_bits(row[keep]), _bits(logits[4 * rows + s][keep])), "a dead page changed another column"
    assert torch.equal(counts, counts_c) and torch.equal(lens, lens_c)


@needs_scorer
@pytest.mark.parametrize(
    "h_i,s_q,n_blocks_max,page_size", [(1, 20, 72, 16), (2, 3, 72, 32), (3, 2, 136, 16), (8, 2, 72, 16), (16, 3, 72, 64), (17, 2, 136, 16), (4, 5, 136, 8)]
)
def test_scorer_row_map_branches_match_the_fp64_reference(h_i, s_q, n_blocks_max, page_size):
    """Every branch of the row map against the same fp64 reference: ``P' = 1`` (two queries per lane, no sum), the pair-only sum
    (``H_i`` 2), a padded head count (3 -> 4), two and three shuffle stages (8 / 16 heads), a query spanning two m-tiles (17 heads),
    several m-tiles of queries (``s_q`` 20 at one head, 5 at four) -- counts, sentinel and a second launch bitwise."""
    kv = [4 * n_blocks_max + s_q - 1, 4 * 30 + s_q + 2, s_q]
    case = _Case(3, s_q, n_blocks_max, page_size, kv, seed=100 * h_i + s_q, h=h_i)
    r = sp.compile_qsa_score_paged(h_i=h_i, s_q=s_q, page_size=page_size)
    logits, counts, lens = case.run(r)
    _assert_scorer(case, logits, counts, lens)
    logits2, counts2, lens2 = case.run(r)
    assert torch.equal(_bits(logits), _bits(logits2)) and torch.equal(counts, counts2) and torch.equal(lens, lens2)


@needs_scorer
@pytest.mark.parametrize("page_size", [16, 64])
def test_units_per_warp_is_a_knob(page_size):
    """32 / 64 / 128 / 256 compressed rows per warp: the same MMA chain per key whichever warp owns it, so the logits are BITWISE across
    the knob (the capacity is not a multiple of 1024: every value walks a ragged last chunk)."""
    case = _Case(3, 2, 1032, page_size, [4 * 1032 - 1, 4 * 700 + 2, 4 * 3 + 3], seed=6)
    outs = []
    for units in fd.UNITS_PER_WARP_CHOICES:
        r = sp.compile_qsa_score_paged(h_i=H, s_q=2, page_size=page_size, units_per_warp=units)
        outs.append(case.run(r))
    _assert_scorer(case, *outs[0])
    for logits, counts, lens in outs[1:]:
        assert torch.equal(_bits(outs[0][0]), _bits(logits)) and torch.equal(outs[0][1], counts) and torch.equal(outs[0][2], lens)


@needs_scorer
def test_ring_depth_is_a_knob():
    """2 / 4 / 8 n-tiles of keys in flight per warp: the same copies and the same MMA chain per key, so the logits are BITWISE
    across the knob; the recipe records the depth and the compile key keeps the default's identity."""
    case = _Case(2, 3, 1032, 16, [4 * 1032 - 1, 4 * 300 + 2], seed=21)
    outs = []
    for depth in fd.RING_DEPTH_CHOICES:
        r = sp.compile_qsa_score_paged(h_i=H, s_q=3, page_size=16, units_per_warp=2, ring_depth=depth)
        assert r.ring_depth == depth
        outs.append(case.run(r))
    _assert_scorer(case, *outs[0])
    for logits, counts, lens in outs[1:]:
        assert torch.equal(_bits(outs[0][0]), _bits(logits)) and torch.equal(outs[0][1], counts) and torch.equal(outs[0][2], lens)
    with pytest.raises(ValueError, match="ring_depth must be one of"):
        sp.compile_qsa_score_paged(h_i=H, s_q=3, page_size=16, units_per_warp=2, ring_depth=5)


@needs_scorer
def test_block_lens_output_is_optional_and_the_artifact_pins_its_presence():
    case = _Case(2, 2, 64, 32, [4 * 64 - 1, 4 * 10], seed=8)
    with_lens = sp.compile_qsa_score_paged(h_i=H, s_q=2, page_size=32, has_block_lens=True)
    without = sp.compile_qsa_score_paged(h_i=H, s_q=2, page_size=32, has_block_lens=False)
    logits_a, counts_a, lens_a = case.run(with_lens)
    logits_b, counts_b, lens_b = case.run(without, with_lens=False)
    assert lens_b is None and torch.equal(_bits(logits_a), _bits(logits_b)) and torch.equal(counts_a, counts_b)
    _assert_scorer(case, logits_a, counts_a, lens_a)
    logits, counts, lens = case.outputs()
    with pytest.raises(ValueError, match="compiled without block_lens"):
        sp.run_qsa_score_paged(without, case.qi, case.pool, case.table, case.kv_lens, logits, counts, lens, top_k=TOP_K, scale=D**-0.5, stream=_stream())


@needs_scorer
@needs_pdl
def test_use_pdl_arm_is_bitwise_the_plain_launch():
    """The programmatic-dependent-launch pair (the wait ahead of every global read, the release after the staging) changes the
    schedule only: logits, counts and block_lens bitwise the plain launch."""
    case = _Case(3, 4, 2048, 16, [4 * 2048, 4 * 600 + 1, 0], seed=9)
    plain = sp.compile_qsa_score_paged(h_i=H, s_q=4, page_size=16, use_pdl=False)
    pdl = sp.compile_qsa_score_paged(h_i=H, s_q=4, page_size=16, use_pdl=True)
    assert pdl.use_pdl and not plain.use_pdl
    a = case.run(plain)
    b = case.run(pdl)
    _assert_scorer(case, *a)
    assert torch.equal(_bits(a[0]), _bits(b[0])) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


@needs_scorer
def test_scale_scales_the_logits_and_a_strided_query_slab_reads_in_place():
    """``scale`` multiplies every logit (the ordering is scale-free); a column slice of a wider bf16 slab (the band's layout) is
    read in place -- bitwise the compact copy."""
    case = _Case(2, 2, 64, 16, [4 * 64 - 1, 4 * 10 + 2], seed=10)
    r = sp.compile_qsa_score_paged(h_i=H, s_q=2, page_size=16)
    one = case.run(r, scale=1.0)
    half = case.run(r, scale=0.5)
    _assert_scorer(case, *one, scale=1.0)
    _assert_scorer(case, *half, scale=0.5)
    slab = torch.zeros(case.qi.shape[0], 640, dtype=torch.bfloat16, device=_DEV)
    slab[:, 64 : 64 + H * D] = case.qi.view(case.qi.shape[0], H * D)
    strided = slab[:, 64 : 64 + H * D].view(case.qi.shape[0], H, D)
    assert strided.data_ptr() == slab.data_ptr() + 128 and strided.stride(0) == 640
    logits, counts, lens = case.outputs()
    sp.run_qsa_score_paged(r, strided, case.pool, case.table, case.kv_lens, logits, counts, lens, top_k=TOP_K, scale=1.0, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(_bits(logits), _bits(one[0])) and torch.equal(counts, one[1])


# --------------------------------------------------------------------------------------------- the selector (cc 9.0+)


@needs_selector
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("n_blocks_max", [8, 520, 4096, 8192, 32768])
def test_paged_selection_matches_the_fp64_topk_set_up_to_ties(n_blocks_max, rows, page_size):
    """The standalone selector: every row's SET equals torch.topk of the fp64 reference up to ties, the ids ascending with -1 past
    block_lens = min(top_k, counts), the counts the oracle's count rule, the scores the reference scores of the selected ids; the
    identity region (fewer than top_k visible blocks) selects every visible block; two calls BITWISE."""
    case = _Case(4, rows, n_blocks_max, page_size, _step_kv_lens(n_blocks_max, rows), seed=n_blocks_max + 11 * rows + page_size)
    # the capacity is passed where the table's reach (whole pages) exceeds it (8 blocks on a 16-row page); where it equals, the default is exercised
    cap = dict(n_blocks_max=n_blocks_max) if case.max_pages * case.rpp != n_blocks_max else {}
    out = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=rows, **cap)
    torch.cuda.synchronize()
    t = 4 * rows
    assert out["block_ids"].shape == (t, TOP_K) and out["scores"].shape == (t, TOP_K) and out["logits"].shape == (t, n_blocks_max)
    assert out["counts"].shape == (t,) and out["block_lens"].shape == (t,)
    _, want_counts = full_block_ids(case.positions(), n_blocks_max, BS)
    assert torch.equal(out["counts"], want_counts.view(-1).to(torch.int32))
    ref = case.reference(D**-0.5).view(t, n_blocks_max)
    _assert_rows(out["block_ids"], out["scores"], out["block_lens"], ref, TOP_K)
    _assert_canonical(out["block_ids"], out["block_lens"])
    if n_blocks_max <= TOP_K:
        want_ids, want_lens = full_block_ids(case.positions(), TOP_K, BS)
        assert torch.equal(out["block_ids"], want_ids.view(t, TOP_K).to(torch.int32)) and torch.equal(out["block_lens"], want_lens.view(-1).to(torch.int32))
    again = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=rows, **cap)
    torch.cuda.synchronize()
    for key in ("block_ids", "counts", "block_lens"):
        assert torch.equal(out[key], again[key]), f"two calls differ in {key}"
    assert torch.equal(_bits(out["scores"]), _bits(again["scores"]))
    live = torch.arange(n_blocks_max, device=_DEV)[None, :] < out["counts"][:, None]
    assert torch.equal(_bits(out["logits"][live]), _bits(again["logits"][live]))


@needs_selector
def test_selector_preallocated_outputs_are_written_in_place_and_match():
    case = _Case(3, 2, 1024, 32, [4 * 1024 - 1, 4 * 300 + 2, 0], seed=12)
    t = 6
    logits = torch.full((t, 1024), SENTINEL, dtype=torch.float32, device=_DEV)
    counts = torch.full((t,), -7, dtype=torch.int32, device=_DEV)
    lens = torch.full((t,), -7, dtype=torch.int32, device=_DEV)
    out = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=2, logits_out=logits, counts_out=counts, block_lens_out=lens)
    base = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=2)
    torch.cuda.synchronize()
    assert out["logits"].data_ptr() == logits.data_ptr() and out["counts"].data_ptr() == counts.data_ptr() and out["block_lens"].data_ptr() == lens.data_ptr()
    _assert_scorer(case, logits, counts, lens)
    assert torch.equal(out["block_ids"], base["block_ids"]) and torch.equal(_bits(out["scores"]), _bits(base["scores"]))
    assert counts[4:].tolist() == [0, 0] and bool((out["block_ids"][4:] == -1).all()) and bool(torch.isneginf(out["scores"][4:]).all())


@needs_selector
def test_selector_knobs_are_the_same_function():
    """``units_per_warp`` and ``use_pdl`` (cc 9.0+) are performance knobs of the selector: block_ids and scores bitwise."""
    case = _Case(2, 3, 4096, 16, [4 * 4096 - 2, 4 * 1000 + 1], seed=13)
    base = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=3)
    for kw in (dict(units_per_warp=1), dict(units_per_warp=4), dict(use_pdl=True)):
        other = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=3, **kw)
        torch.cuda.synchronize()
        assert torch.equal(base["block_ids"], other["block_ids"]) and torch.equal(_bits(base["scores"]), _bits(other["scores"])), kw


@needs_selector
def test_selector_on_a_side_stream_with_the_ambient_stream_parked_matches_the_synchronised_control():
    """The delayed-producer pin with the REAL kernels (Rule 5): a synchronised control on torch's default stream first (it compiles both
    kernels, so the parked window below holds launches only); then the default (ambient) stream is parked by a long spin and the SAME
    selection is issued on a side stream, passed as a ``torch.cuda.Stream``; the side stream ALONE is synchronised and the outputs are
    compared ON the side stream while the ambient stream is still parked -- anything the call had wrongly enqueued on the ambient
    stream (an allocation's fill, the top-k, a copy) would not have run yet, and the comparison would see it.  block_ids, scores,
    counts, block_lens and the live logits bitwise the control; the premise -- the ambient stream busy at the issue and at the
    comparison -- is asserted."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is the park; the matmul fallback is too short")
    from gated_block_stream_probe import park_the_default_stream

    case = _Case(3, 2, 4096, 16, [4 * 4096 - 2, 4 * 1000 + 1, 4 * 20 + 2], seed=16)
    ambient, side = torch.cuda.default_stream(), torch.cuda.Stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    control = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=2)
    torch.cuda.synchronize()
    ref = case.reference(D**-0.5).view(6, 4096)
    _assert_rows(control["block_ids"], control["scores"], control["block_lens"], ref, TOP_K)
    park_the_default_stream(3.0)  # the AMBIENT stream is busy from here, through the call and past the comparison
    out = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=2, stream=side)
    parked_at_issue = not ambient.query()
    side.synchronize()  # the LAUNCH stream only
    with torch.cuda.stream(side):
        same = {key: torch.equal(out[key], control[key]) for key in ("block_ids", "counts", "block_lens")}
        same["scores"] = torch.equal(_bits(out["scores"]), _bits(control["scores"]))
        live = torch.arange(4096, device=_DEV)[None, :] < control["counts"][:, None]
        same["logits"] = torch.equal(_bits(out["logits"][live]), _bits(control["logits"][live]))
    parked_at_compare = not ambient.query()
    torch.cuda.synchronize()
    print(f"premise: parked_at_issue={parked_at_issue} parked_at_compare={parked_at_compare}; same={same}")
    assert parked_at_issue and parked_at_compare, "the premise: the ambient stream must still be busy at the issue and at the comparison"
    assert all(same.values()), f"the selection on the side stream differs from the synchronised control: {same}"


@needs_selector
def test_paged_entry_agrees_with_the_dense_selector_when_that_module_is_present():
    """The dense-cache decode selector (``cudnn.gated_attention_block.qsa_select``) is a sibling change: probed BY NAME -- skipped by
    name where the tree lacks it, imported unguarded where it is present -- then both selectors over the same keys (the dense form
    through the gathered cache) select the same SET per row up to ties at the k-th boundary."""
    if importlib.util.find_spec("cudnn.gated_attention_block.qsa_select") is None:
        pytest.skip("the dense decode selector module cudnn.gated_attention_block.qsa_select is absent from this tree (a sibling change)")
    dense_mod = importlib.import_module("cudnn.gated_attention_block.qsa_select")
    rows, n_blocks_max = 4, 2048
    case = _Case(3, rows, n_blocks_max, 16, [4 * 1500 + 1 + rows, 4 * 2000 + 3 + rows, 4 * 300 + rows], seed=14)
    paged = fd.qsa_select_decode_paged(case.qi, case.pool, case.table, case.kv_lens, seq_q=rows)
    try:
        dense = dense_mod.qsa_select_decode(case.qi.view(3, rows, H, D), case.dense().contiguous(), case.kv_lens, list_per_sequence=False)
    except NotImplementedError as error:
        pytest.skip(f"the dense selector declines on this device: {error}")
    torch.cuda.synchronize()
    ref = case.reference(D**-0.5).view(-1, n_blocks_max)
    dense_ids = dense["block_ids"].view(-1, TOP_K)
    _assert_rows(dense_ids, dense["scores"].view(-1, TOP_K), dense["block_lens"].view(-1), ref, TOP_K)
    _assert_rows(paged["block_ids"], paged["scores"], paged["block_lens"], ref, TOP_K)
    kth = torch.topk(ref, TOP_K, dim=-1).values[:, -1]
    for r in range(3 * rows):
        a, b = set(paged["block_ids"][r].tolist()) - {-1}, set(dense_ids[r].tolist()) - {-1}
        if a != b:
            diff = torch.tensor(sorted(a ^ b), device=_DEV)
            assert bool(((ref[r, diff] - kth[r]).abs() <= TIE_TOL).all()), f"row {r}: {len(a ^ b)} ids differ beyond a tie"
