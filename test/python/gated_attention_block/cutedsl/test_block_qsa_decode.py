# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's block-sparse DECODE mode: a dense ``[B, S_q]`` declaration on a ``QsaSpec`` geometry with
``kv_source="paged"`` over a paged KV cache -- the step's tokens written through, then stage (4) is the index-list sparse core
READING THE POOLS: per sequence the keys of the listed 4-token blocks and the open tail over the first ``kv_lens[b]`` tokens of
its ``block_table`` row, on the bottom-right diagonal (row ``j`` at ``pos_j = kv_lens[b] - S_q + j``).  The list form is a
DECLARATION fact derived from ``S_q``: at ``S_q <= 4`` (the sparse core's shared-list cap) ONE list per sequence,
``block_ids [B, top_k]`` (+ ``block_lens [B]``), selected at the STEP-0 position ``pos_0 = kv_lens[b] - S_q`` and shared by the
step's rows -- the speculative (MTP) rows see the listed blocks plus EVERY token from the step-0 tail start to their own position
(the contract page's definition; the oracle's ``pos0``); past four rows per sequence the same paged read with one list per row
(``[B, S_q, top_k]``): the extend-with-prefix form over the pools (a chunked prefill, a prefix-cache hit) at any ``S_q``.  The
split (``decode_split_kv``: fp32 partials in the workspace + the split combine, the gate riding the combine) is the one
performance knob; the LSE is served over the pools (the core writes it).

Host cells (any device; the declaration cells even on the CPU): the declaration FACTS (the sparse stage in the paged form, the
list form per ``S_q``, no dense route, where the gate runs, the split's default rule, ``seq_lens_present`` forced, the stage
order); the typed declines (the packed form under ``QsaSpec`` over the pools -- the serving table's ``sparse`` row --, a block
table sample that is not contiguous, a split past the item's tile count, ``max_kv_len`` past the table's reach, the bottom-right
diagonal on the block's own K / V, the in-block indexer over the pools); the static ``serving_envelope`` (the DECODE form served
on a Rubin reading with ``sparse_core``, the EXTEND form declined with the ``sparse`` row); the execute-time form checks on an
UNCOMPILED block under ``set_sync_debug_mode("error")`` (the list form per declaration, ``seq_lens`` refused, the whole table
required, ``kv_lens`` required); the split rule.

Rubin cells (``requires_rubin``; one compiled block per cell): the decode ladder against the fp32 oracle -- the dense block
reference over the WHOLE sequence gives the post-norm / post-RoPE Q, K, V and the pre-sigmoid gate; the prefix rows fill the
pools (NaN everywhere else: a gathered unwritten row would poison O), the step's rows are the block's own write-through; the
step rows' attention is the oracle's over ``qsa_visible_mask(..., pos0=)`` (shared) or per row (per-token), then the gate and
the out projection -- ``cos(out) > 0.999``, the gated O (the workspace slot) and the LSE within ``atol 2e-2`` (the sparse
end-to-end module's bars, never widened), a dead row (``kv_lens[b] = 0``; a row whose position is negative because the
sequence is shorter than the step) EXACTLY ``out = 0`` / ``O = 0`` / ``LSE = -inf``, the pools' step rows BITWISE the slab's
post-RoPE K / V bands, the prefix rows and the NULL block untouched, two launches bitwise.  The ladder: ``S_q`` 1 .. 4 (the
shared list; B 1 / 2 / 4; ``kv_lens`` up to 20000 incl. 2052, a dead entry, a one-token sequence and a sequence shorter than
the step), ``S_q`` 8 and 64 (per-token lists), bf16 and f16, pages 16 HND / 32 unified / 64 NHD, splits 1 / 2 / 4 / 17 and
the default rule, ``fuse_gate`` (unsplit: the core's epilogue; split: the combine) and the separate gate launch, ``return_lse``
on / off, ``block_lens`` given / absent, ``fuse_norm_rope``, the indexer band's raw-key write-through, the Flash-Next 24 / 2
geometry at ``d_model`` 2560; the MTP matrix at every residue of the step-0 position on a short sequence (the ``4k + 3`` row
sees the block completed after step 0: the step-0-anchored oracle within budget, the per-row-tail reading farther); the
block's form vs the standalone adapter on the SAME Q / gate views, pools, table and list: BITWISE; ``get_workspace_size()``
honest under a split (a sentinel past the carve untouched); CUDA-graph capture of the FIRST execute on a NEW stream with no
warm-up under ``set_sync_debug_mode("error")``, the replay bitwise the eager run, and the replay over NEW lengths / tables /
lists / slots / inputs bitwise the eager run over them.

The in-block indexer's DECODE step over the pools (``QsaSpec(index_source="indexer")`` with ``kv_source="paged"``; the ``_IndexerCase``
cells, 2026-10-21): the block selects per row from the band, the stack's raw-key ring and its paged compressed-key cache -- the prefix
blocks compressed from the oracle's raw band through the compress kernel, the ring primed with the last ``cap`` raw keys, the step's
rows the block's own prepare (or the unfused twin); the TWO-STAGE oracle -- (1) the selection: the counts ``min(top_k, floor((pos + 1) /
4))``, the list a strictly descending valid prefix then ``-1``, every id below the row's visible count, the SET the fp64 top-k of the
kernel's OWN ``(ix_q, kbar)`` up to ties, the indexer queries BITWISE the pool-1 compress of the slab's band, the ring rows BITWISE the raw
band, the block a row completes BITWISE the compress kernel over the same four raw keys; (2) the attention within the module's bars of
the fp32 oracle fed the KERNEL's list per row -- dead rows exact (``kv_lens`` 0, rows at a negative position: no ring / cache write, the
null ring block and page 0 untouched, ``counts`` 0, every slot ``-1``); the identity region (512 visible blocks: the list ``511 .. 0``,
bitwise the caller-list block fed the full list); ``fuse_prepare`` on vs off BITWISE with the census 8 / 6 vs 11 / 9 launches by kernel
name, no allocation on the execute path; the caller-list block fed the indexer's ``block_ids_out`` BITWISE; the FIRST execute captured
with no warm-up and replayed bitwise, over a new step too; the RoPE-table ceiling by form; the ring-slot form bitwise the derived
one; the position-table arm bitwise the per-token form; the per-row MTP rows; the workspace sentinel; the typed declines, each equal
to the envelope's row; the execute inputs refused in both directions.
"""

import contextlib
import math
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    KV_SOURCE_PAGED,
    KV_SOURCE_SELF,
    DecodeEnvelope,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    index_k_raw_view,
)
from cudnn.gated_attention_block.api import _EXTEND_DECLINES, _cols, _IndexerDecode, _qsa_decode_decline_texts, _QsaPrepare, _SparseSdpa, _view, ProjBlock

requirement_error = cutedsl_requirement_error("the gated attention block's sparse decode tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_census import cupti_kernels as _cupti_kernels  # noqa: E402
from gated_block_qsa_reference import qsa_visible_mask, random_block_ids  # noqa: E402
from gated_block_reference import RefGeometry, build_rope_tables, gated_attention_block_reference  # noqa: E402

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SM107 = (10, 7)
BF16, F16 = torch.bfloat16, torch.float16
D = 256
ROPE = 64
BS = 4  # tokens per selectable block
COS_OUT = 0.999  # the end-to-end suite's bar on `out`
ATOL = 2e-2  # the SDPA stage test's bar on O / LSE -- never widened here
SHARED_MAX_S_Q = 4  # the sparse core's shared-list cap (SparseCapabilities.index_shared_list_max_tokens)
GEOM_SMALL = dict(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=ROPE)  # group 4, cheap projections
GEOM_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=D, rope_dim=ROPE)  # Qwen3.8-Flash-Next TP 1 (group 12)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


@contextlib.contextmanager
def _no_host_sync():
    """Every declaration-time and form check stays on the host without a device sync (Rule 3) -- enforced, not read."""
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The 16-bit patterns of a bf16 / f16 tensor: a NaN-safe bitwise comparison."""
    return x.contiguous().view(torch.int16)


def _sentinel(dtype):
    return float(torch.finfo(dtype).max) / 4


def _pools(n_pages, h, page, layout, dtype, device=_DEV, fill=float("nan")):
    """Two pools as ``[P, H, page, D]`` views: ``hnd`` (compact), ``nhd`` (by strides), ``unified`` (one ``[P, H, page, 2D]`` content
    row per token, K at ``+0``, V at ``+D``).  NaN-filled: the sparse core gathers listed blocks and the open tail only, so a NaN
    reaching O names a row the kernel should never have read (the sparse module's own detector)."""
    if layout == "hnd":
        return tuple(torch.full((n_pages, h, page, D), fill, device=device, dtype=dtype) for _ in range(2))
    if layout == "nhd":
        return tuple(torch.full((n_pages, page, h, D), fill, device=device, dtype=dtype).permute(0, 2, 1, 3) for _ in range(2))
    assert layout == "unified"
    row = torch.full((n_pages, h, page, 2 * D), fill, device=device, dtype=dtype)
    return row[..., :D], row[..., D:]


def _geom(geom_kw, top_k=512, index_band=False, index_list_form=None, **over):
    kw = dict(geom_kw, causal_bottom_right=True)
    kw.update(over)
    return GatedAttentionBlockGeometry(**kw, qsa=QsaSpec(top_k=top_k, index_band=index_band, index_list_form=index_list_form)), RefGeometry(**kw)


def _declare(
    geom_kw,
    batch,
    s_q,
    *,
    page=16,
    layout="hnd",
    n_pages=8,
    dtype=BF16,
    top_k=512,
    index_band=False,
    table=None,
    geom_over=None,
    index_list_form=None,
    **blk_kw,
):
    """A declared (not compiled) sparse DECODE block on EMPTY tensors: the declaration cells (any device)."""
    geom, _ = _geom(geom_kw, top_k=top_k, index_band=index_band, index_list_form=index_list_form, **(geom_over or {}))
    mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=_DEV)  # noqa: E731
    kc, vc = _pools(n_pages, geom.h_kv, page, layout, dtype)
    bt = torch.zeros(batch, n_pages, dtype=torch.int32, device=_DEV) if table is None else table
    kw = dict(paged_kv_page_size=page, kv_source=KV_SOURCE_PAGED, sample_k_cache=kc, sample_v_cache=vc, sample_block_table=bt)
    kw.update(blk_kw)
    blk = GatedAttentionBlockFwd(
        mk(batch, s_q, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=dtype, device=_DEV),
        torch.ones(D, dtype=dtype, device=_DEV),
        mk(batch, s_q, ROPE),
        mk(batch, s_q, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(batch, s_q, geom.d_model),
        geom,
        **kw,
    )
    return blk, dict(k_cache=kc, v_cache=vc, block_table=bt, geom=geom)


def _row(name):
    return next(r for r in _EXTEND_DECLINES if r[0] == name)


# ---------------------------------------------------------------------------
# Host: the declaration facts, the typed declines, the envelope, the execute form checks, the split rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s_q", [1, 4, 5, 64])
@pytest.mark.parametrize("fuse_gate", [False, True], ids=["gate_launch", "fuse_gate"])
def test_declaration_facts_of_the_sparse_decode_mode(s_q, fuse_gate):
    """A dense ``[B, S_q]`` QsaSpec declaration over the pools CONSTRUCTS (no host sync) with the SPARSE stage in its paged form:
    the shared list at ``S_q <= 4`` and per-token lists past it, NO dense decode route (the sparse core has no row cap: 64 rows
    construct), the gate where it was declared (the sparse core fuses it unsplit and routes it into the combine under a split --
    stage (5) is not built either way), the per-sequence KV length forced on, the write-through right before the sparse stage, the
    split the rule picks (or the pinned one), ``max_kv_len`` the table's reach by default."""
    from cudnn.gated_attention_block.api import sparse_decode_split_choice

    with _no_host_sync():
        blk, aux = _declare(GEOM_SMALL, 4, s_q, n_pages=6, fuse_gate=fuse_gate)
    st = blk._sdpa
    assert isinstance(st, _SparseSdpa) and st.paged and st.kv_source == KV_SOURCE_PAGED
    assert st.list_per_sequence is (s_q <= SHARED_MAX_S_Q) and blk.qsa_shared_list is (s_q <= SHARED_MAX_S_Q) and blk.sparse_decode
    assert blk.decode_route is None and blk.gate_fused is fuse_gate and (blk._gate is None) is fuse_gate and st.fuse_gate is fuse_gate
    assert blk.seq_lens_present and st.seq_lens_present
    assert blk._stages.index(blk._cache_write) == blk._stages.index(blk._sdpa) - 1
    assert blk.max_kv_len == 6 * 16
    sm = torch.cuda.get_device_properties(torch.device(_DEV)).multi_processor_count if _DEV == "cuda" else 0
    assert blk.decode_split_kv == sparse_decode_split_choice(batch=4, h_kv=2, s_q=s_q, top_k=512, sm_count=sm) == st.split_kv
    assert 1 <= st.split_kv <= 17
    with _no_host_sync():
        pinned, _ = _declare(GEOM_SMALL, 4, s_q, n_pages=6, fuse_gate=fuse_gate, decode_split_kv=3, max_kv_len=70, return_lse=True)
    assert pinned.decode_split_kv == 3 == pinned._sdpa.split_kv and pinned.max_kv_len == 70 and pinned.return_lse and pinned._sdpa.want_lse
    # the serving declaration's capture-safe projections: set per stage iff the GEMM driver's "auto" resolver names a JIT tile for the
    # stage's own shape -- the forced tile where 256 divides N (this geometry's two projections), the small-M route's tile at T <= 128
    # (the indexer band's 13952-column slab of the 24 / 2 geometry: its own cell below).  THIS small geometry's band slab (N 5120 + 640
    # = 5760 = 45 x 128, off the 256 grid) is not that case on any served part: its 90 64-row tiles FIT one wave at 90 SMs and more,
    # where the route keeps the driver's graph route (eager steps; a capture is the driver's typed refusal); a part with fewer SMs
    # overflows the wave and takes the route's 128-row tile at T <= 128.  So the expectation is read off the live resolver, as the
    # 13952 cell reads it, and the one-wave fact is asserted where the live part has it; the out projection (N 512, on the 256 grid)
    # is capture-safe throughout
    from cudnn.gated_attention_block.kernels.proj_gemm import SMALL_M_MAX, _auto_tile_name
    from cudnn.gemm.frost import arch_family

    assert blk._proj.capture_safe and blk._out_proj.capture_safe
    with _no_host_sync():
        band, baux = _declare(GEOM_SMALL, 4, s_q, n_pages=6, index_band=True)
    g = baux["geom"]
    assert g.n_qkvg % 256 != 0 and g.n_qkvg % 128 == 0
    name = _auto_tile_name(4 * s_q, g.n_qkvg, g.d_model, dtype=BF16, a_major="k", b_major="k", block_scale=False, alpha=False)
    resolves = name is not None
    if _DEV == "cuda" and arch_family.active_family() == "sm100":
        one_wave = -(-g.n_qkvg // 64) <= sm  # the slab's 90 64-row tiles fit one wave: every served part
        assert resolves is (4 * s_q <= SMALL_M_MAX and not one_wave), (sm, name)
    else:
        assert not resolves, name  # the sm120 family's forced-tile fallback has no tile off the 256 grid
    assert getattr(band._proj, "capture_safe", False) is resolves and band._proj.resolves_a_jit_tile() is resolves and band._out_proj.capture_safe


@pytest.mark.parametrize(
    "batch, s_q, jit", [(1, 1, True), (4, 1, True), (32, 4, True), (4, 64, False)], ids=["B1-Sq1-T1", "B4-Sq1-T4", "B32-Sq4-T128", "B4-Sq64-T256"]
)
def test_the_indexer_band_declaration_over_the_pools_is_capture_safe_at_decode_row_counts(batch, s_q, jit):
    """The 24 / 2 geometry WITH the indexer band over the pools (N = 13312 + 640 = 13952, off the 256 grid): at decode row counts
    (``T = B x S_q <= 128``) the slab GEMM takes the small-M route's 128-row swap-AB tile (13952 = 109 x 128; its 218 64-row tiles
    overflow a wave on every served part), so BOTH projection stages are declared capture-safe -- exactly where the driver's own
    resolver names a tile (the predicate is read off it, never re-spelled); past 128 rows the band slab has no tile (256 does not
    divide it) and keeps the graph route: the slab stage is not flagged (a capture is the driver's typed refusal), the out projection
    (N 2560, on the 256 grid) is.  Declared on empty tensors with no host sync (any device)."""
    from cudnn.gated_attention_block.kernels.proj_gemm import SMALL_M_MAX, _auto_tile_name
    from cudnn.gemm.frost import arch_family
    from cudnn.gemm.frost.tile_config import _sm_count

    with _no_host_sync():
        blk, aux = _declare(GEOM_FLASH_NEXT, batch, s_q, n_pages=6, index_band=True)
    g = aux["geom"]
    t = batch * s_q
    assert g.n_qkvg == 13952 and g.n_qkvg % 256 != 0 and g.n_qkvg % 128 == 0 and (t <= SMALL_M_MAX) is jit
    # the resolver the block's predicate reads (its arch-family gate included), at the live part's SM count: on the sm100 family the
    # small-M route's 128-row swap-AB tile below 218 SMs; the sm120 family keeps the forced-tile fallback, which has no tile off the 256 grid
    name = _auto_tile_name(t, g.n_qkvg, g.d_model, dtype=BF16, a_major="k", b_major="k", block_scale=False, alpha=False)
    resolves = name is not None
    if arch_family.active_family() == "sm100":
        assert name is None or (name.startswith("CONFIG_sm100_128x") and name.endswith("_swapAB"))
        # the one-wave boundary the route reads off the live SM count: the slab's 218 64-row tiles overflow the wave of every served part
        # (<= 212 SMs) -> the small-M tile at T <= 128; a part with 218 SMs or more holds them in one wave and keeps the graph route
        assert resolves is (jit and _sm_count() < -(-g.n_qkvg // 64)), (_sm_count(), name)
    assert getattr(blk._proj, "capture_safe", False) is resolves and blk._out_proj.capture_safe is True
    assert blk._proj.resolves_a_jit_tile() is resolves and blk._out_proj.resolves_a_jit_tile() is True


def test_the_sparse_split_rule_fills_the_machine_and_stays_inside_the_item():
    """The split the sparse decode mode lowers with when ``decode_split_kv=None``: enough chunks to put one work item per SM
    (``ceil(SMs / (B x H_kv x S_q))``), never more than the item's tile count ``ceil((top_k + 1) / 32)`` nor than the measured cap of 16
    (the B 1 / B 4 x 32K x S_q 1 split sweeps on a 212-SM part with the pinned decode-grid combine: the step keeps falling to 16-17
    chunks where the former combine entry, whose cost grows with the partial count, turned up past 16), 1 for a full launch and
    for a non-CUDA reading (0 SMs) -- a performance choice only: the same function under every value."""
    from cudnn.gated_attention_block.api import SPARSE_DECODE_SPLIT_CAP
    from cudnn.gated_attention_block.api import sparse_decode_split_choice as pick

    assert SPARSE_DECODE_SPLIT_CAP == 16
    assert pick(batch=4, h_kv=2, s_q=1, top_k=512, sm_count=212) == 16  # 212 / 8 = 27 -> the measured cap of 16 (below the 17 tiles of a 512-list item)
    assert pick(batch=1, h_kv=2, s_q=1, top_k=512, sm_count=212) == 16  # 2 items: the cap again
    assert pick(batch=1, h_kv=2, s_q=1, top_k=256, sm_count=212) == 9  # ceil(257 / 32) = 9 tiles at top_k 256: the item's tile count binds under the cap
    assert pick(batch=32, h_kv=2, s_q=1, top_k=512, sm_count=212) == 4  # ceil(212 / 64) = 4: the fill rule, under either cap
    assert pick(batch=4, h_kv=2, s_q=4, top_k=512, sm_count=212) == 7  # ceil(212 / 32)
    assert pick(batch=32, h_kv=2, s_q=4, top_k=512, sm_count=212) == 1  # 256 items: a full launch never splits
    assert pick(batch=2, h_kv=2, s_q=1, top_k=64, sm_count=212) == 3  # ceil(65 / 32) = 3 tiles at top_k 64
    assert pick(batch=1, h_kv=2, s_q=1, top_k=512, sm_count=0) == 1  # no CUDA device: declined at check_support anyway
    assert pick(batch=1, h_kv=1, s_q=1, top_k=4, sm_count=212) == 1  # a 4-list item is one tile


def test_typed_declines_of_the_sparse_decode_mode():
    """Every request the mode cannot serve, declined at DECLARATION with the feature named: the PACKED form over the pools (the
    serving table's ``sparse`` row -- the core serves THD and paged pools one at a time), a block-table sample that is not
    contiguous (the sparse core binds the whole ``(B, max_pages)`` table), a split past the item's tile count, ``max_kv_len``
    past the table's reach, the bottom-right diagonal on the block's OWN K / V (the decode mode is the paged one), the in-block
    indexer over the pools, the prepare fusion over the pools."""
    name, exc, _pred, msg = _row("sparse")
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    geom, _ = _geom(GEOM_SMALL)
    kc, vc = _pools(4, 2, 16, "hnd", BF16)
    with pytest.raises(exc, match="thd=True") as e:
        GatedAttentionBlockFwd(
            mk(32, geom.d_model),
            mk(geom.n_qkvg, geom.d_model),
            mk(D),
            mk(D),
            mk(32, ROPE),
            mk(32, ROPE),
            mk(geom.d_model, geom.h_q * D),
            mk(32, geom.d_model),
            geom,
            thd=True,
            num_sequences=2,
            max_seq_len=32,
            token_capacity=32,
            paged_kv_page_size=16,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=kc,
            sample_v_cache=vc,
            sample_block_table=torch.zeros(2, 4, dtype=torch.int32, device=_DEV),
        )
    assert str(e.value) == msg and "one at a time" in msg
    wide = torch.zeros(2, 9, dtype=torch.int32, device=_DEV)[:, :4]  # a row slice of a wider table: not contiguous
    with pytest.raises(ValueError, match="CONTIGUOUS under QsaSpec"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, table=wide)
    with pytest.raises(NotImplementedError, match=r"ceil\(\(top_k \+ 1\) / 32\)"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, decode_split_kv=18)
    with pytest.raises(NotImplementedError, match=r"ceil\(\(top_k \+ 1\) / 32\)"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, top_k=64, decode_split_kv=4)
    with pytest.raises(ValueError, match="exceeds the block table's reach"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, max_kv_len=65)
    with pytest.raises(ValueError, match="decode_split_kv must be"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, decode_split_kv=0)
    # the block's own K / V with the bottom-right diagonal: the decode mode is the paged one -- the decline names it
    with pytest.raises(NotImplementedError, match="kv_source='paged'") as e:
        GatedAttentionBlockFwd(
            mk(1, 8, geom.d_model),
            mk(geom.n_qkvg, geom.d_model),
            mk(D),
            mk(D),
            mk(1, 8, ROPE),
            mk(1, 8, ROPE),
            mk(geom.d_model, geom.h_q * D),
            mk(1, 8, geom.d_model),
            geom,
        )
    assert "causal_bottom_right=True" in str(e.value) and "sparse adapter's record declines" not in str(e.value)
    # INVERTED 2026-10-21: the in-block indexer over the pools CONSTRUCTS -- its decode step (the fused prepare's paged arms, the paged scorer,
    # the decode top-k) in the indexer's slot, the list per row; the three side-cache samples declare it (the host cells below)
    blk = _declare_indexer()
    assert isinstance(blk._indexer_decode, _IndexerDecode) and blk._indexer is None and blk.indexer_decode and blk.qsa_list_form == "per_row"
    # INVERTED 2026-10-21: the prepare fusion over the pools is served on the block-sparse decode mode (the caller-list band form and the
    # indexer's decode step, each bitwise its unfused twin); the dense-geometry DECODE form keeps a decline naming the mode that serves it
    assert isinstance(_declare(GEOM_SMALL, 2, 1, n_pages=4, fuse_prepare=True)[0]._prepare, _QsaPrepare)
    with pytest.raises(NotImplementedError, match="fuse_prepare=True with kv_source='paged' on a dense geometry"):
        mk2 = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
        dense = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True)
        GatedAttentionBlockFwd(
            mk2(1, 1, dense.d_model),
            mk2(dense.n_qkvg, dense.d_model),
            mk2(D),
            mk2(D),
            mk2(1, 1, ROPE),
            mk2(1, 1, ROPE),
            mk2(dense.d_model, dense.h_q * D),
            mk2(1, 1, dense.d_model),
            dense,
            paged_kv_page_size=16,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=kc,
            sample_v_cache=vc,
            sample_block_table=torch.zeros(1, 4, dtype=torch.int32, device=_DEV),
            fuse_norm_rope=False,
            fuse_prepare=True,
        )


def _indexer_samples(batch, page=16, n_pages=4, cap=8, dtype=BF16, qsa_over=None):
    """The decode indexer's three side-cache samples on EMPTY tensors: the paged compressed-key cache ``[n_pages, P / 4, 1, 128]``, the ring
    ``[batch + 1, cap, 1, 128]`` (block 0 the null block), the ring block table ``[B, 1]``."""
    d_i = QsaSpec(**(qsa_over or {})).index_head_dim
    return dict(
        sample_index_kv_cache=torch.empty(n_pages, page // BS, 1, d_i, dtype=dtype, device=_DEV),
        sample_index_state_cache=torch.empty(batch + 1, cap, 1, d_i, dtype=dtype, device=_DEV),
        sample_index_state_block_table=torch.zeros(batch, 1, dtype=torch.int32, device=_DEV),
    )


def _declare_indexer(geom_kw=GEOM_FLASH_NEXT, batch=1, s_q=1, *, page=16, n_pages=4, cap=8, dtype=BF16, qsa_over=None, geom_over=None, samples=True, **blk_kw):
    """A declared (not compiled) decode-indexer block on EMPTY tensors: ``QsaSpec(index_source="indexer", index_band=True)`` over the
    pools with the three side-cache samples (``samples=False`` leaves them out: the missing-sample refusal)."""
    qsa = QsaSpec(index_source="indexer", index_band=True, **(qsa_over or {}))
    geom = GatedAttentionBlockGeometry(**dict(geom_kw, causal_bottom_right=True, **(geom_over or {})), qsa=qsa)
    mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=_DEV)  # noqa: E731
    kc, vc = _pools(n_pages, geom.h_kv, page, "hnd", dtype)
    kw = dict(
        paged_kv_page_size=page,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(batch, n_pages, dtype=torch.int32, device=_DEV),
    )
    if samples:
        kw.update(_indexer_samples(batch, page=page, n_pages=n_pages, cap=cap, dtype=dtype, qsa_over=qsa_over))
    kw.update(blk_kw)
    tables = (None, None) if blk_kw.get("rope_position_tables_only") else (mk(batch, s_q, geom.rope_dim), mk(batch, s_q, geom.rope_dim))
    return GatedAttentionBlockFwd(
        mk(batch, s_q, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=dtype, device=_DEV),
        torch.ones(D, dtype=dtype, device=_DEV),
        *tables,
        mk(geom.d_model, geom.h_q * D),
        mk(batch, s_q, geom.d_model),
        geom,
        **kw,
    )


def test_the_envelope_serves_the_sparse_decode_mode_and_declines_the_packed_form(monkeypatch):
    """The static record a serving stack reads: on a Rubin reading a QsaSpec geometry with the bottom-right diagonal SERVES the
    DECODE form (``sparse_core`` names stage (4); the requires spell the shared list and the per-token form) and declines the
    EXTEND form with the table's ``sparse`` row (the very message the packed declaration raises); the top-left QsaSpec geometry
    meets the ``diagonal`` row like a dense one; on this device the verdict reads the part."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom, _ = _geom(GEOM_SMALL)
    env = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV, device_cc=_SM107)
    assert isinstance(env.decode, DecodeEnvelope) and env.decode.served and env.decode.sparse_core and env.decode.reason is None
    assert env.forms == ("decode",) and not env.extend.served and env.extend.reason == _row("sparse")[3]
    assert any("[B, top_k]" in r for r in env.decode.requires) and any("[B, S_q, top_k]" in r for r in env.decode.requires)
    assert env.decode.routes == () and "sparse" not in env.decode.declines and env.extend.declines["sparse"] == _row("sparse")[3]
    top_left = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec()), BF16, _DEV, device_cc=_SM107)
    assert not top_left.decode.served and top_left.decode.reason == _row("diagonal")[3] and top_left.decode.sparse_core  # a geometry fact, served or not
    dense = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True), BF16, _DEV, device_cc=_SM107)
    assert dense.decode.served and not dense.decode.sparse_core and dense.decode.routes
    monkeypatch.undo()
    here = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV)
    assert here.decode.served is (_cc() == _SM107 and here.cutedsl_ok)


def test_execute_form_checks_of_the_sparse_decode_mode():
    """On an UNCOMPILED block (the index tensors and the paged inputs are a declaration-vs-argument contract: checked first, no
    plan, no sync): the shared-list form takes ``block_ids [B, top_k]`` / ``block_lens [B]`` and refuses the per-token spelling
    by name, the per-token form the reverse; ``seq_lens`` is refused over the pools; the table is the WHOLE declared
    ``(B, max_pages)`` table; ``kv_lens`` is required; an int64 list is refused, never converted; a ``block_ids`` view off the 16-byte
    base alignment is refused on the shared form as on the per-row form (the aligned storage-offset control passes)."""
    with _no_host_sync():
        blk, aux = _declare(GEOM_SMALL, 2, 2, n_pages=4)
        per_tok, aux8 = _declare(GEOM_SMALL, 2, 8, n_pages=4)
    geom = aux["geom"]
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
    z = lambda *shape: torch.zeros(*shape, dtype=BF16, device=_DEV)  # noqa: E731

    def run(b, *, s_q, ids, lens=None, kv_lens=None, table=None, seq_lens=None, slot=None, pools=None):
        pools = aux if pools is None else pools
        return b.execute(
            z(2, s_q, geom.d_model),
            z(geom.n_qkvg, geom.d_model),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            z(2, s_q, ROPE),
            z(2, s_q, ROPE),
            z(geom.d_model, geom.h_q * D),
            z(2, s_q, geom.d_model),
            torch.empty(1 << 20, dtype=torch.uint8, device=_DEV),
            seq_lens=seq_lens,
            block_ids=ids,
            block_lens=lens,
            k_cache=pools["k_cache"],
            v_cache=pools["v_cache"],
            block_table=pools["block_table"] if table is None else table,
            kv_lens=i32(2) if kv_lens is None else kv_lens,
            slot_mapping=i32(2 * s_q) if slot is None else slot,
        )

    with _no_host_sync():
        with pytest.raises(ValueError, match=r"block_ids must be \[2, 512\]"):
            run(blk, s_q=2, ids=i32(4, 512))  # the per-token spelling on the shared-list form
        with pytest.raises(ValueError, match=r"block_ids must be \[2, 512\]"):
            run(blk, s_q=2, ids=i32(2, 2, 512))
        with pytest.raises(ValueError, match=r"block_lens must be \[2\]"):
            run(blk, s_q=2, ids=i32(2, 512), lens=i32(4))
        with pytest.raises(ValueError, match="int64"):
            run(blk, s_q=2, ids=torch.zeros(2, 512, dtype=torch.int64, device=_DEV))
        with pytest.raises(ValueError, match=r"block_ids must be \[16, 512\] or \[2, 8, 512\]"):
            run(per_tok, s_q=8, ids=i32(2, 512), pools=aux8)  # the shared spelling on the per-token form
        with pytest.raises(ValueError, match="seq_lens"):
            run(blk, s_q=2, ids=i32(2, 512), seq_lens=i32(2))
        with pytest.raises(ValueError, match="kv_lens"):
            run(blk, s_q=2, ids=i32(2, 512), kv_lens=i32(3))
        with pytest.raises(ValueError, match="block_table"):
            run(blk, s_q=2, ids=i32(2, 512), table=i32(2, 6))  # wider than the declared table: the sparse core binds exactly (B, max_pages)
        with pytest.raises(ValueError, match="block_table"):
            run(blk, s_q=2, ids=i32(2, 512), table=i32(2, 8)[:, :4])
        # The 16-byte base alignment of block_ids (the sparse core binds it as a TMA / bulk-copy base) is refused BEFORE any stage on the
        # SHARED form too -- a storage-offset view one id (4 bytes) in; the control four ids (16 bytes) in passes every form check and stops at
        # the uncompiled block's own gate, as does the plain list.  (A review found the shared form returning before this check.)
        base = i32(4 + 2 * 512)
        with pytest.raises(ValueError, match="16-byte aligned"):
            run(blk, s_q=2, ids=base[1 : 1 + 2 * 512].view(2, 512))
        with pytest.raises(RuntimeError, match="compile"):
            run(blk, s_q=2, ids=base[4 : 4 + 2 * 512].view(2, 512))
        with pytest.raises(RuntimeError, match="compile"):
            run(blk, s_q=2, ids=i32(2, 512))  # every form check passed: the uncompiled block stops at its own gate


def test_index_list_form_is_a_declaration_attribute_the_envelope_publishes():
    """``QsaSpec.index_list_form`` (appended; a DECLARATION attribute, numerics-changing): ``"per_row"`` binds one list per row at ANY
    ``S_q`` (the sparse core's per-token arm, the serving stacks' per-row selection -- ``qsa_shared_list`` False, ``execute`` takes
    ``[T, top_k]`` / ``[B, S_q, top_k]``); ``None`` keeps the ``S_q`` rule (shared at ``S_q <= 4``, per row past it); ``"shared"`` past the
    cap is a typed decline whose text EQUALS the envelope's ``shared_list_rows`` row (one shared token: ``tail covers at most two blocks``);
    ``"shared"`` on the block's own K / V is refused by name; the vocabulary is pinned at ``QsaSpec.validate``; the envelope publishes the
    forms per geometry (``index_list_forms`` / ``index_list_form_default``) and lists the row under ``decode.declines``."""
    import re

    from cudnn.gated_attention_block.api import QSA_INDEX_LIST_FORMS

    assert QSA_INDEX_LIST_FORMS == (None, "shared", "per_row")
    with pytest.raises(ValueError, match="index_list_form must be None"):
        QsaSpec(index_list_form="rows").validate()
    with _no_host_sync():
        for s_q in (1, 2, 4, 8):
            per_row, _ = _declare(GEOM_SMALL, 2, s_q, n_pages=4, index_list_form="per_row")
            assert per_row.sparse_decode and not per_row.qsa_shared_list and per_row.qsa_list_form == "per_row" and not per_row._sdpa.list_per_sequence
            auto, _ = _declare(GEOM_SMALL, 2, s_q, n_pages=4)
            assert auto.qsa_shared_list is (s_q <= SHARED_MAX_S_Q) and auto.qsa_list_form == ("shared" if s_q <= SHARED_MAX_S_Q else "per_row")
            if s_q <= SHARED_MAX_S_Q:
                shared, _ = _declare(GEOM_SMALL, 2, s_q, n_pages=4, index_list_form="shared")
                assert shared.qsa_shared_list and shared.qsa_list_form == "shared" and shared._sdpa.list_per_sequence
        env = GatedAttentionBlockFwd.serving_envelope(_geom(GEOM_SMALL)[0], BF16, _DEV, device_cc=_SM107)
        assert env.decode.index_list_forms == ("shared", "per_row") and env.decode.index_list_form_default == "shared"
        assert "shared_list_rows" in env.decode.declines and "tail covers at most two blocks" in env.decode.declines["shared_list_rows"]
        with pytest.raises(NotImplementedError, match=re.escape("tail covers at most two blocks")) as e:
            _declare(GEOM_SMALL, 2, 8, n_pages=4, index_list_form="shared")
        assert str(e.value) == env.decode.declines["shared_list_rows"]
        assert (
            "shared_list_rows"
            not in GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL), BF16, _DEV, device_cc=_SM107).decode.declines
        )
        assert GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL), BF16, _DEV, device_cc=_SM107).decode.index_list_forms == ()
        # the block's own K / V: every row carries its own list -- "shared" is refused by name, "per_row" names what happens
        geom_self = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec(index_list_form="shared"))
        mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
        args = lambda g: (
            mk(1, 8, g.d_model),
            mk(g.n_qkvg, g.d_model),
            mk(D),
            mk(D),
            mk(1, 8, ROPE),
            mk(1, 8, ROPE),
            mk(g.d_model, g.h_q * D),
            mk(1, 8, g.d_model),
            g,
        )  # noqa: E731
        with pytest.raises(ValueError, match="kv_source='self'"):
            GatedAttentionBlockFwd(*args(geom_self))
        own = GatedAttentionBlockFwd(*args(GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec(index_list_form="per_row"))))
        assert not own.sparse_decode and own.qsa_list_form is None and not own.qsa_shared_list
        # the per-row form's execute contract at S_q 2: both per-row spellings pass the form checks (the uncompiled block stops at its own gate),
        # the shared spelling is refused by name
        blk, aux = _declare(GEOM_SMALL, 2, 2, n_pages=4, index_list_form="per_row")
        geom = aux["geom"]
        i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
        z = lambda *shape: torch.zeros(*shape, dtype=BF16, device=_DEV)  # noqa: E731

        def run(ids):
            return blk.execute(
                z(2, 2, geom.d_model),
                z(geom.n_qkvg, geom.d_model),
                torch.ones(D, dtype=BF16, device=_DEV),
                torch.ones(D, dtype=BF16, device=_DEV),
                z(2, 2, ROPE),
                z(2, 2, ROPE),
                z(geom.d_model, geom.h_q * D),
                z(2, 2, geom.d_model),
                torch.empty(1 << 20, dtype=torch.uint8, device=_DEV),
                block_ids=ids,
                k_cache=aux["k_cache"],
                v_cache=aux["v_cache"],
                block_table=aux["block_table"],
                kv_lens=i32(2),
                slot_mapping=i32(4),
            )

        for ids in (i32(4, 512), i32(2, 2, 512)):
            with pytest.raises(RuntimeError, match="compile"):
                run(ids)
        with pytest.raises(ValueError, match=r"block_ids must be \[4, 512\] or \[2, 2, 512\].*index_list_form"):
            run(i32(2, 512))


# ---------------------------------------------------------------------------
# Rubin: the oracle, the cells
# ---------------------------------------------------------------------------


def _rope_tables(n_pos, dtype, ref_geom):
    cos, sin = build_rope_tables(n_pos, ROPE, base=ref_geom.rope_base, batch=1, device=_DEV, dtype=dtype)
    return cos, sin  # [1, n_pos, ROPE]


def _attend(ref, allowed, p_live, scale, dtype, h_q, G):
    """The fp32 oracle's attention of the LIVE rows of one sequence (positions ``p_live``) over its keys through the visible mask
    ``allowed`` ``[n_live, L]`` (the dense reference's post-norm / post-RoPE Q, K, V and pre-sigmoid gate), then the gate: ``(o_gated
    [n_live, h_q, D]`` in ``dtype`` -- one rounding of O, the gate in fp32, one rounding --, ``lse [h_q, n_live]`` fp32)``; a row with no
    visible key is exactly 0 / -inf."""
    q_rows = ref.q[0][p_live].float()  # [n_live, h_q, D]
    gate_rows = ref.gate[0][p_live].float()
    n_live = int(q_rows.shape[0])
    o = torch.zeros(n_live, h_q, D, dtype=torch.float32, device=_DEV)
    lse = torch.full((h_q, n_live), float("-inf"), dtype=torch.float32, device=_DEV)
    for hh in range(h_q):
        kk, vv = ref.k[0][:, hh // G].float(), ref.v[0][:, hh // G].float()
        sc = (q_rows[:, hh] @ kk.t()) * scale
        sc = sc.masked_fill(~allowed, float("-inf"))
        rmax = sc.amax(dim=-1)
        dead = torch.isinf(rmax) & (rmax < 0)
        safe = torch.where(dead, torch.zeros_like(rmax), rmax)
        p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
        den = p.sum(dim=-1)
        den_safe = torch.where(dead, torch.ones_like(den), den)
        oc = (p @ vv) / den_safe[:, None]
        o[:, hh] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
        lse[hh] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
    o16 = o.to(dtype)  # the oracle's single rounding of O, then the gate in fp32, then one rounding
    og = (o16.float() * torch.sigmoid(gate_rows)).to(dtype)
    return og, lse


class _Case:
    """One sparse DECODE step over a batch of sequences with lengths ``kv_lens`` (0 = a dead entry; a length below ``s_q`` leaves
    the rows with a negative position dead): the whole-sequence inputs per entry, the step's inputs (its last ``s_q`` positions),
    the NaN-filled pools holding the ORACLE's prefix K / V, the block table (page 0 = the NULL block of a dead entry, ``-1`` past a
    sequence's live pages), the slot mapping of the step's tokens (-1 on a dead row), the lists (one per sequence at the step-0
    position, or one per row), and the per-sequence fp32 oracle of the step's ``out`` rows, gated O and LSE."""

    def __init__(
        self,
        geom_kw,
        s_q,
        kv_lens,
        *,
        page=16,
        layout="hnd",
        dtype=BF16,
        seed=0,
        top_k=512,
        block_lens=True,
        return_lse=False,
        fuse_gate=False,
        fuse_norm_rope=False,
        split=None,
        index_band=False,
        n_pages=None,
        pos0_anchor=True,
        shuffle=True,
        index_list_form=None,
        fuse_prepare=None,
    ):
        self.s_q, self.kv_lens, self.page, self.layout, self.dtype, self.top_k = int(s_q), [int(x) for x in kv_lens], page, layout, dtype, top_k
        self.fuse_prepare = fuse_prepare
        self.block_lens_given, self.return_lse, self.fuse_gate, self.index_band = bool(block_lens), bool(return_lse), bool(fuse_gate), bool(index_band)
        # the list form: the S_q rule (one list per sequence up to the shared-list cap), or QsaSpec.index_list_form as declared
        self.index_list_form = index_list_form
        self.shared = (self.s_q <= SHARED_MAX_S_Q) if index_list_form is None else (index_list_form == "shared")
        self.geom, self.ref_geom = _geom(geom_kw, top_k=top_k, index_band=index_band, index_list_form=index_list_form)
        geom, ref_geom = self.geom, self.ref_geom
        h_q, h_kv, d_model = geom.h_q, geom.h_kv, geom.d_model
        G = h_q // h_kv
        B = len(self.kv_lens)
        g = torch.Generator(device=_DEV).manual_seed(seed)
        self.gen = g

        def randn(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        n_dense = sum(geom.qkvg_block_widths[:4])
        w_qkvg = randn(geom.n_qkvg, d_model)
        self.w = dict(
            w_qkvg=w_qkvg, w_q_norm=torch.ones(D, dtype=dtype, device=_DEV), w_k_norm=torch.ones(D, dtype=dtype, device=_DEV), w_o=randn(d_model, h_q * D)
        )
        L_max = max(max(self.kv_lens), s_q)
        cos_t, sin_t = _rope_tables(L_max, dtype, ref_geom)
        scale = geom.scale
        # the step's inputs: row j of entry b is at position p_j = L - s_q + j; a negative position is a DEAD row (finite inputs, RoPE row 0)
        self.h = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.cos = cos_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.sin = sin_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.ref_out = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.ref_o_gated = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_lse = torch.full((B, h_q, s_q), float("-inf"), dtype=torch.float32, device=_DEV)
        self.ref_lse_rowtail = self.ref_lse.clone()
        self.ref_o_gated_rowtail = self.ref_o_gated.clone()
        self.ref_rows = {}
        self.live_rows = torch.zeros(B, s_q, dtype=torch.bool, device=_DEV)
        ids_rows, lens_rows = [], []
        for b, L in enumerate(self.kv_lens):
            pos = torch.arange(s_q, device=_DEV) + (L - s_q)
            live = pos >= 0
            self.live_rows[b] = live
            pos0 = torch.tensor([L - s_q], device=_DEV)
            if self.shared:
                ids_b, lens_b = random_block_ids(pos0, top_k, BS, generator=g, kv_lens=torch.tensor([L], device=_DEV), shuffle=shuffle)
            else:
                ids_b, lens_b = random_block_ids(pos, top_k, BS, generator=g, kv_lens=torch.tensor([L], device=_DEV).expand(s_q), shuffle=shuffle)
            ids_rows.append(ids_b)
            lens_rows.append(lens_b)
            if L == 0:
                continue  # a dead entry: no sequence, no reference, block 0, slots -1
            h_full = randn(1, L, d_model, std=1.0)
            ref = gated_attention_block_reference(
                h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], self.w["w_o"], ref_geom
            )
            self.ref_rows[b] = (ref.k[0], ref.v[0])  # [L, h_kv, D]: the prefix rows fill the pools
            p_live = pos[live]
            self.h[b, live] = h_full[0, p_live]
            self.cos[b, live] = cos_t[0, p_live]
            self.sin[b, live] = sin_t[0, p_live]
            # the step rows' attention over the sequence's keys through the visible mask (the shared list anchored at pos_0, or per row)
            rows = ids_b.expand(s_q, top_k) if self.shared else ids_b
            lens_arg = (lens_b.expand(s_q) if self.shared else lens_b) if block_lens else None
            for anchored, (o_dst, lse_dst) in ((True, (self.ref_o_gated, self.ref_lse)), (False, (self.ref_o_gated_rowtail, self.ref_lse_rowtail))):
                if not anchored and not self.shared:
                    continue
                allowed = qsa_visible_mask(rows, lens_arg, pos, L, L, BS, top_k=top_k, pos0=(L - s_q) if (anchored and self.shared) else None)[live]
                og, lse = _attend(ref, allowed, p_live, scale, dtype, h_q, G)
                o_dst[b, live] = og
                lse_dst[b, :, live] = lse
                if anchored:
                    self.ref_out[b, live] = (og.reshape(-1, h_q * D).float() @ self.w["w_o"].float().t()).to(dtype)
        self.block_ids = torch.stack([x.reshape(top_k) if self.shared else x.reshape(s_q, top_k) for x in ids_rows]).contiguous()
        self.block_lens = torch.stack([x.reshape(()) if self.shared else x.reshape(s_q) for x in lens_rows]).contiguous()
        assert self.block_ids.dtype == torch.int32 and self.block_lens.dtype == torch.int32
        # pages: block 0 reserved (NULL); each live entry gets ceil(L / P) pages from a permutation of 1 .. n_pages - 1; -1 past them
        need = sum(-(-L // page) for L in self.kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=g, device=_DEV) + 1).tolist()
        self.max_pages = max(-(-L_max // page), 1)
        self.block_table = torch.full((B, self.max_pages), -1, dtype=torch.int32, device=_DEV)
        it = iter(perm)
        slots = torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _pools(self.n_pages, h_kv, page, layout, dtype)
        self.index_cache = torch.full((self.n_pages, page, geom.qsa.index_head_dim), float("nan"), dtype=dtype, device=_DEV) if index_band else None
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                self.block_table[b] = 0  # the dead entry: the NULL block, slots -1
                continue
            pages = [next(it) for _ in range(-(-L // page))]
            self.block_table[b, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=_DEV)
            k_rows, v_rows = self.ref_rows[b]
            for p in range(max(L - s_q, 0)):  # the PREFIX: the oracle's rows into the pools
                pg, off = pages[p // page], p % page
                self.k_cache[pg, :, off] = k_rows[p]
                self.v_cache[pg, :, off] = v_rows[p]
            for j in range(s_q):  # the step's live tokens: written through by the block
                p = L - s_q + j
                if p >= 0:
                    slots[b * s_q + j] = pages[p // page] * page + p % page
        self.slot_mapping = slots
        self.kv_lens_t = torch.tensor(self.kv_lens, dtype=torch.int32, device=_DEV)
        self.prefix_bits = (_bits(self.k_cache).clone(), _bits(self.v_cache).clone())
        kw = dict(
            paged_kv_page_size=page,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            fuse_gate=fuse_gate,
            fuse_norm_rope=fuse_norm_rope,
            return_lse=return_lse,
            decode_split_kv=split,
        )
        if fuse_prepare is not None:
            kw["fuse_prepare"] = fuse_prepare
        self.blk_kw = kw
        self.out = torch.empty(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.lse = torch.empty(B, h_q, s_q, dtype=torch.float32, device=_DEV) if return_lse else None
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, self.geom, **kw
        )
        assert isinstance(self.blk._sdpa, _SparseSdpa) and self.blk.sparse_decode and self.blk.qsa_shared_list is self.shared
        self.ws = None

    def compile(self):
        assert self.blk.check_support()
        self.blk.compile()
        self.ws = torch.full((self.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
        return self

    def execute(
        self, *, out=None, ws=None, lse=None, h=None, cos=None, sin=None, kv_lens=None, block_table=None, slot_mapping=None, ids=None, lens=None, stream=None
    ):
        out = self.out if out is None else out
        self.blk.execute(
            self.h if h is None else h,
            self.w["w_qkvg"],
            self.w["w_q_norm"],
            self.w["w_k_norm"],
            self.cos if cos is None else cos,
            self.sin if sin is None else sin,
            self.w["w_o"],
            out,
            self.ws if ws is None else ws,
            lse=(self.lse if lse is None else lse) if self.return_lse else None,
            block_ids=self.block_ids if ids is None else ids,
            block_lens=(self.block_lens if lens is None else lens) if self.block_lens_given else None,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.block_table if block_table is None else block_table,
            kv_lens=self.kv_lens_t if kv_lens is None else kv_lens,
            slot_mapping=self.slot_mapping if slot_mapping is None else slot_mapping,
            index_k_raw=self.index_cache,
            current_stream=stream,
        )
        return out

    def run(self):
        if self.ws is None:
            self.compile()
        self.out.fill_(_sentinel(self.dtype))
        if self.lse is not None:
            self.lse.fill_(_sentinel(self.dtype))
        self.execute()
        torch.cuda.synchronize()
        return self.out

    def slab_views(self, ws=None):
        """The post-norm-RoPE Q band, the GATE band, the K / V bands of the slab the last execute left in the workspace, as the
        sparse stage's ``[B, S_q, H, D]`` views at the slab's token stride, and the O slot."""
        ws = self.ws if ws is None else ws
        g, lay = self.geom, self.blk._ws
        B, S = self.blk.batch, self.blk.seq_len
        t = B * S
        proj = _view(ws, lay.proj, (t, g.n_qkvg), self.dtype)
        o_q, o_g, o_k, o_v = g.qkvg_offsets[:4]
        q = _cols(proj, o_q, g.h_q, g.d_head).view(B, S, g.h_q, g.d_head)
        gate = _cols(proj, o_g, g.h_q, g.d_head).view(B, S, g.h_q, g.d_head)
        k = _cols(proj, o_k, g.h_kv, g.d_head).view(B, S, g.h_kv, g.d_head)
        v = _cols(proj, o_v, g.h_kv, g.d_head).view(B, S, g.h_kv, g.d_head)
        o = _view(ws, lay.o, (t, g.h_q, g.d_head), self.dtype).view(B, S, g.h_q, g.d_head)
        return q, gate, k, v, o, proj

    def label(self):
        form = f"{'shared' if self.shared else 'per-row'}{'' if self.index_list_form is None else ' (declared)'}"
        return f"[{self.geom.h_q}/{self.geom.h_kv} S_q={self.s_q} {form} p{self.page} {self.layout} {self.dtype} split={self.blk.decode_split_kv} gate={'fused' if self.fuse_gate else 'launch'} lse={self.return_lse} lens={self.kv_lens}]"

    def check(self, out=None, ws=None, lse=None, *, what="out"):
        """The standard assertions: finite, no sentinel, dead rows exact, within the budget vs the oracle (out, the gated O slot, the
        LSE), the pools' step rows bitwise the slab's K / V bands, the prefix rows and the NULL block untouched."""
        out = self.out if out is None else out
        ws = self.ws if ws is None else ws
        lse = self.lse if lse is None else lse
        sent = _sentinel(self.dtype)
        live = self.live_rows
        dead = ~live
        q, gate, k, v, o, proj = self.slab_views(ws)
        assert torch.isfinite(out.float()).all(), f"{what}: non-finite out"
        assert not (out.float() == sent).any(), f"{what}: a sentinel survived in out"
        assert torch.isfinite(o.float()).all(), f"{what}: non-finite gated O (a row the sparse core never wrote keeps the workspace fill)"
        if dead.any():
            assert (out[dead] == 0).all() and (o[dead] == 0).all(), f"{what}: a dead row's out and O must be exactly 0"
        c = _cos(out[live], self.ref_out[live])
        d_out = float((out[live].float() - self.ref_out[live].float()).abs().max())
        d_o = float((o[live].float() - self.ref_o_gated[live].float()).abs().max())
        d_lse = 0.0
        if self.return_lse:
            assert lse is not None and not (lse == sent).any(), f"{what}: a sentinel survived in LSE"
            dead_bhs = dead[:, None, :].expand_as(lse)
            if dead.any():
                assert torch.isinf(lse[dead_bhs]).all() and (lse[dead_bhs] < 0).all(), f"{what}: a dead row's LSE must be -inf exactly"
            d_lse = float((lse[~dead_bhs] - self.ref_lse[~dead_bhs]).abs().max()) if live.any() else 0.0
            assert d_lse <= ATOL, f"{what}: LSE off the oracle by {d_lse} (budget {ATOL})"
        assert c > COS_OUT, f"{what}: cos(out) {c} <= {COS_OUT}"
        assert d_o <= ATOL, f"{what}: gated O off the oracle by {d_o} (budget {ATOL})"
        # the write-through: the step's live rows are the slab's post-RoPE K / V rows, bitwise; nothing else moved
        slot = self.slot_mapping
        lv = slot >= 0
        s_ = slot[lv].long()
        kf, vf = k.reshape(-1, self.geom.h_kv, D), v.reshape(-1, self.geom.h_kv, D)
        assert torch.equal(_bits(self.k_cache[s_ // self.page, :, s_ % self.page, :]), _bits(kf[lv])), f"{what}: the K pool rows are not the slab's rows"
        assert torch.equal(_bits(self.v_cache[s_ // self.page, :, s_ % self.page, :]), _bits(vf[lv])), f"{what}: the V pool rows are not the slab's rows"
        for pool, before in zip((self.k_cache, self.v_cache), self.prefix_bits):
            now = _bits(pool)
            touched = (now != before).any(dim=-1).any(dim=1)  # [n_pages, page] rows whose bits moved
            rows_moved = touched.nonzero()
            written = set((int(p) for p in (s_ // self.page * self.page + s_ % self.page).tolist()))
            assert all(int(pg) * self.page + int(off) in written for pg, off in rows_moved), f"{what}: a pool row outside the step's slots moved"
            assert not touched[0].any(), f"{what}: the NULL block moved"
        if self.index_cache is not None:
            raw = index_k_raw_view(proj, self.geom, self.blk.batch, self.blk.seq_len)  # [B, S, 1, index_head_dim]
            assert torch.equal(_bits(self.index_cache[s_ // self.page, s_ % self.page, :]), _bits(raw.reshape(-1, raw.shape[-1])[lv]))
            assert (~torch.isnan(self.index_cache.float())).all(-1).sum().item() == int(lv.sum())
        print(f"\n{self.label()} {what}: cos(out) {c:.6f} max|d out| {d_out:.4e} max|d O_gated| {d_o:.4e} max|d LSE| {d_lse:.2e} dead rows {int(dead.sum())}")
        return dict(cos=c, d_out=d_out, d_o=d_o, d_lse=d_lse, dead=int(dead.sum()))


def _two_launches(case: _Case, what: str):
    a = case.run().clone()
    o_a = case.slab_views()[4].clone()
    lse_a = None if case.lse is None else case.lse.clone()
    b = case.run().clone()
    o_b = case.slab_views()[4].clone()
    assert torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(o_a), _bits(o_b)), f"{what}: two launches are not bitwise"
    if lse_a is not None:
        assert torch.equal(lse_a, case.lse), f"{what}: two launches are not bitwise on LSE"
    return case.check(what=what)


# (id, geometry, S_q, kv_lens, dtype, page, layout, split, fuse_gate, return_lse, block_lens, fuse_norm_rope, index_band, seed)
_LADDER = [
    pytest.param(GEOM_SMALL, 1, (2052, 8192, 0, 4097), BF16, 16, "hnd", None, False, True, True, False, False, 1, id="Sq1-B4-shared-p16-hnd-default-split-lse"),
    pytest.param(
        GEOM_SMALL, 2, (300, 0, 1000, 77), BF16, 64, "nhd", 1, True, False, True, True, False, 2, id="Sq2-B4-shared-p64-nhd-unsplit-fused-gate-fused-norm-rope"
    ),
    pytest.param(
        GEOM_SMALL, 3, (8192, 2052), F16, 32, "unified", 4, True, True, False, False, False, 3, id="Sq3-B2-shared-p32-unified-f16-split4-no-block-lens"
    ),
    pytest.param(
        GEOM_SMALL, 4, (8191, 8190, 8189, 8188), BF16, 16, "hnd", 17, True, True, True, False, False, 4, id="Sq4-B4-shared-every-residue-split17-fused-gate-lse"
    ),
    pytest.param(GEOM_FLASH_NEXT, 4, (4096,), BF16, 16, "hnd", None, True, False, True, True, False, 5, id="Sq4-B1-flash-next-24-2-shared-default-split-fused"),
    pytest.param(
        GEOM_SMALL, 4, (2, 4, 0, 20000), BF16, 16, "hnd", 2, False, True, True, False, False, 6, id="Sq4-B4-shared-shorter-than-the-step-dead-20000-split2"
    ),
    pytest.param(GEOM_SMALL, 1, (1024, 60), BF16, 16, "hnd", 1, False, True, True, False, True, 7, id="Sq1-B2-shared-index-band-raw-key-write-through"),
    pytest.param(GEOM_SMALL, 8, (300, 100), BF16, 16, "hnd", 1, False, True, True, False, False, 8, id="Sq8-B2-per-token-p16-unsplit-lse"),
    pytest.param(GEOM_SMALL, 64, (2500, 64), BF16, 64, "nhd", 2, True, False, True, False, False, 9, id="Sq64-B2-per-token-p64-nhd-split2-fused-gate"),
]

# (geometry, S_q, kv_lens, page, layout, split, fuse_gate, return_lse, block_lens, seed): the DECLARED per-row form at S_q <= 4 --
# QsaSpec(index_list_form="per_row") -- one list per row at the row's own position (the serving stacks' per-row selection), where the
# S_q rule would have bound the shared step-0 list; the control arm the in-block indexer's decode step is compared against
_PER_ROW_LADDER = [
    pytest.param(GEOM_SMALL, 2, (300, 0, 2052, 77), 16, "hnd", 4, True, True, True, 11, id="Sq2-B4-per-row-declared-p16-split4-fused-gate-lse"),
    pytest.param(
        GEOM_SMALL, 4, (8191, 8190, 2, 8188), 64, "nhd", None, False, True, False, 12, id="Sq4-B4-per-row-declared-p64-nhd-default-split-no-block-lens"
    ),
    pytest.param(GEOM_FLASH_NEXT, 4, (4096,), 16, "hnd", 1, True, False, True, 13, id="Sq4-B1-per-row-declared-flash-next-24-2-unsplit-fused-gate"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, s_q, kv_lens, page, layout, split, fuse_gate, return_lse, block_lens, seed", _PER_ROW_LADDER)
def test_the_declared_per_row_form_at_mtp_widths_matches_the_per_row_oracle(
    geom_kw, s_q, kv_lens, page, layout, split, fuse_gate, return_lse, block_lens, seed
):
    """``QsaSpec(index_list_form="per_row")`` at ``S_q`` 2 / 4 (the widths the ``S_q`` rule binds to the shared step-0 list): one list per
    row at the row's own position, the step within the budget of the PER-ROW oracle (every row's list anchored at its own position: the
    row at ``4k + 3`` sees block ``k`` only through its own list), dead rows exact, the pools bitwise, two launches bitwise -- the
    serving stacks' per-row selection as the sparse core computes it."""
    case = _Case(
        geom_kw,
        s_q,
        kv_lens,
        page=page,
        layout=layout,
        seed=seed,
        split=split,
        fuse_gate=fuse_gate,
        return_lse=return_lse,
        block_lens=block_lens,
        index_list_form="per_row",
    )
    assert not case.shared and not case.blk.qsa_shared_list and case.blk.qsa_list_form == "per_row" and case.block_ids.shape == (len(kv_lens), s_q, 512)
    case.compile()
    _two_launches(case, "per-row declared")


@requires_rubin
@pytest.mark.parametrize("geom_kw, s_q, kv_lens, dtype, page, layout, split, fuse_gate, return_lse, block_lens, fuse_norm_rope, index_band, seed", _LADDER)
def test_sparse_decode_matches_the_oracle(
    geom_kw, s_q, kv_lens, dtype, page, layout, split, fuse_gate, return_lse, block_lens, fuse_norm_rope, index_band, seed
):
    """The decode ladder (module docstring): every cell within the budget vs the step-0-anchored oracle (or the per-row oracle on
    the per-token form), dead rows exact, the write-through rows bitwise the slab, two launches bitwise."""
    case = _Case(
        geom_kw,
        s_q,
        kv_lens,
        page=page,
        layout=layout,
        dtype=dtype,
        seed=seed,
        split=split,
        fuse_gate=fuse_gate,
        return_lse=return_lse,
        block_lens=block_lens,
        fuse_norm_rope=fuse_norm_rope,
        index_band=index_band,
    )
    case.compile()
    if split is not None:
        assert case.blk.decode_split_kv == split
    _two_launches(case, "ladder")


@requires_rubin
@pytest.mark.parametrize("r", [0, 1, 2, 3], ids=lambda r: f"pos0-mod4-{r}")
@pytest.mark.parametrize("s_q", [2, 4], ids=lambda s: f"Sq{s}")
def test_mtp_rows_share_the_step0_list_and_see_the_block_completed_after_step0(s_q, r):
    """The MTP (speculative) rows of a SHORT sequence (every one of its ~60 keys matters) at every residue of the step-0 position:
    the block's rows are within the budget of the STEP-0-ANCHORED oracle (the contract page's definition: the listed blocks plus
    every token from the step-0 tail start to the row's own position), and the per-row-tail reading (vLLM's expand form, the
    oracle at ``pos0=None``), which hides the block completed between step 0 and a row at ``4k + 3``, is farther from the block
    wherever such a row exists (``r < 3`` at ``S_q = 4``; ``r = 2`` at ``S_q = 2``) and coincides otherwise."""
    L = 56 + r + s_q  # pos_0 = L - s_q = 56 + r -> pos_0 mod 4 = r
    case = _Case(GEOM_SMALL, s_q, (L, L + 4), page=16, seed=10 + r, split=1, fuse_gate=True, return_lse=True)
    case.compile()
    m = _two_launches(case, f"mtp r={r}")
    o = case.slab_views()[4]
    live = case.live_rows
    d_pos0 = float((o[live].float() - case.ref_o_gated[live].float()).abs().max())
    d_rowtail = float((o[live].float() - case.ref_o_gated_rowtail[live].float()).abs().max())
    lse_rowtail = float((case.lse - case.ref_lse_rowtail).abs().max())
    hidden_row_exists = any(((L - s_q + j) % 4 == 3) and j > 0 and ((L - s_q + 1) // 4) < ((L - s_q + j + 1) // 4) for j in range(s_q))
    print(
        f"\nmtp S_q={s_q} pos0 mod 4 = {r}: vs the pos0 oracle max|dO| {d_pos0:.3e}; vs the per-row-tail reading max|dO| {d_rowtail:.3e} max|dLSE| {lse_rowtail:.3e}; hidden row {hidden_row_exists}"
    )
    assert d_pos0 <= ATOL
    if hidden_row_exists:
        assert d_rowtail > d_pos0 and lse_rowtail > m["d_lse"], "the per-row-tail reading should be the farther one where a block completed after step 0"
    else:
        assert torch.equal(case.ref_o_gated_rowtail, case.ref_o_gated) and torch.equal(case.ref_lse_rowtail, case.ref_lse)


@requires_rubin
def test_the_block_form_equals_the_standalone_adapter_bitwise_on_the_same_views():
    """The block adds nothing of its own to the sparse core: the standalone adapter on the block's OWN post-norm / post-RoPE Q
    view and GATE view (the slab bands at the slab's token stride), the same pools, table, lengths, list and split writes the
    gated O (and the LSE) BITWISE the block's O slot -- unsplit (the core's epilogue gate) and split (the gate in the combine)."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    for split in (1, 4):
        case = _Case(GEOM_SMALL, 2, (700, 0, 2052, 130), page=16, seed=20 + split, split=split, fuse_gate=True, return_lse=True)
        case.compile()
        case.run()
        case.check(what=f"adapter twin split {split}")
        q, gate, _k, _v, o, _proj = case.slab_views()
        B, S = case.blk.batch, case.blk.seq_len
        o2 = torch.full_like(o, _sentinel(case.dtype))
        lse2 = torch.full_like(case.lse, _sentinel(case.dtype))
        a = SparseGqaFwdDslSm107(
            q=q,
            k=case.k_cache,
            v=case.v_cache,
            o=o2,
            lse=lse2,
            block_ids=case.block_ids,
            block_lens=case.block_lens,
            seq_kv_lens=case.kv_lens_t,
            top_k=case.top_k,
            scale=case.geom.scale,
            paged_kv=True,
            page_size=case.page,
            block_table=case.block_table,
            list_per_sequence=True,
            bottom_right=True,
            split_kv=split,
            epilogue_gate=gate,
        )
        assert a.check_support()
        a.compile()
        ws = torch.empty(max(a.scratch_workspace_bytes(), 16), dtype=torch.uint8, device=_DEV)
        a.execute(workspace=ws)
        torch.cuda.synchronize()
        assert torch.equal(_bits(o2), _bits(o)) and torch.equal(
            lse2, case.lse
        ), f"split {split}: the block's sparse stage differs from the standalone adapter on the same views"


@requires_rubin
def test_get_workspace_size_is_honest_and_holds_the_split_partials():
    """The split partials (the sparse adapter's scratch) sit inside ``get_workspace_size()`` (the engine region behind the carve);
    nothing past the size is written: a sentinel tail survives an execute; the dense sparse block's workspace is unchanged by the
    paged read (no scratch when unsplit)."""
    case = _Case(GEOM_SMALL, 1, (4096, 512, 0), page=16, seed=31, split=8, fuse_gate=True)
    case.compile()
    blk = case.blk
    lay = blk._layout()
    scratch = blk._sdpa.scratch_workspace_bytes()
    assert scratch > 0 and blk.get_workspace_size() >= lay.total_bytes + scratch
    need = blk.get_workspace_size()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    ws[:need].fill_(0x7F)
    out = torch.empty_like(case.out)
    case.execute(out=out, ws=ws)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.check(out, ws, what="sentinel-tailed workspace")
    unsplit = _Case(GEOM_SMALL, 1, (4096, 512, 0), page=16, seed=31, split=1, fuse_gate=True).compile()
    assert unsplit.blk._sdpa.scratch_workspace_bytes() == 0 and unsplit.blk.get_workspace_size() < need
    unsplit.run()
    unsplit.check(what="unsplit twin")


def _written(pool):
    return int((~torch.isnan(pool.float())).all(-1).sum())


@requires_rubin
def test_capture_of_the_first_execute_replays_bitwise_and_takes_new_lengths_tables_lists_slots_and_inputs():
    """The serving stack's defaults: the FIRST execute of a fresh declaration captured on a NEW stream with no eager warm-up under
    ``set_sync_debug_mode("error")`` (no host sync in the path: lengths, tables, lists and slots are device data); the capture
    launches nothing, the replay is BITWISE a fresh eager run; new ``kv_lens`` / ``block_table`` / ``block_ids`` / ``block_lens`` /
    ``slot_mapping`` / ``h`` / RoPE rows through the captured tensors replay bitwise the eager run over them (a split plan: the
    partials in the captured workspace)."""
    case = _Case(GEOM_SMALL, 2, (300, 0, 1000, 77), page=16, seed=41, split=4, fuse_gate=True, return_lse=True)  # fresh: never executed
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.full((case.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    case.lse.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_lse = case.out.clone(), case.lse.clone()
        case.check(replayed, lse=replayed_lse, what="replay")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed)) and torch.equal(
            case.lse, replayed_lse
        ), "the replay of the captured first execute differs from the eager run"
        # a NEW step through the captured tensors: new lengths (one entry a token longer: its slot and its list move), a re-laid-out
        # table, new lists / counts / slots / h / RoPE rows -- the oracle of the new step through a sibling case over the same shapes
        sib = _Case(GEOM_SMALL, 2, (301, 0, 1000, 77), page=16, seed=43, split=4, fuse_gate=True, return_lse=True, n_pages=case.n_pages)
        assert sib.block_table.shape == case.block_table.shape and sib.k_cache.shape == case.k_cache.shape
        case.k_cache.copy_(sib.k_cache)
        case.v_cache.copy_(sib.v_cache)
        case.block_table.copy_(sib.block_table)
        case.kv_lens_t.copy_(sib.kv_lens_t)
        case.slot_mapping.copy_(sib.slot_mapping)
        case.block_ids.copy_(sib.block_ids)
        case.block_lens.copy_(sib.block_lens)
        case.h.copy_(sib.h)
        case.cos.copy_(sib.cos)
        case.sin.copy_(sib.sin)
        case.w["w_qkvg"].copy_(sib.w["w_qkvg"])
        case.w["w_o"].copy_(sib.w["w_o"])
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.ref_out, case.ref_o_gated, case.ref_lse, case.live_rows, case.kv_lens = sib.ref_out, sib.ref_o_gated, sib.ref_lse, sib.live_rows, sib.kv_lens
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2, replayed2_lse = case.out.clone(), case.lse.clone()
        case.check(replayed2, lse=replayed2_lse, what="replay over new lengths / table / lists / slots / inputs")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed2)) and torch.equal(
            case.lse, replayed2_lse
        ), "the replay over new lengths / tables / lists / slots differs from the eager run over them"
    finally:
        graph.reset()


# ---------------------------------------------------------------------------
# Rubin: the split combine's FORM per grid -- the census names the kernel, two executes bitwise
# ---------------------------------------------------------------------------

# The kernel names the CUDA profiler reports for the shared combine module's forms (its own pins name them the same way).
_COMBINE_KERNELS = {
    "decode_grid": ("cudnn_kernel__combine_decode_grid_kernel",),
    "lean_walk": ("cudnn_kernel__combine_lean_walk_kernel",),
    "own": ("cudnn_kernel__combine_kernel", "cudnn_kernel__combine_decode_kernel"),
}
# (geometry, S_q, kv_lens, pinned split, the form the sparse stage pins its combine entry to): the decode-grid kernel on a small grid
# from split 8 (B 2 x 8 heads: 16 lean blocks / 16 rows), the module's own forms on the same grid below split 8, the lean group walk on
# an overflow grid at split <= 4 (B 6 x 24 heads: 144 lean blocks, past the decode-grid bound of 128), the module's own forms on the
# B 4 x S_q 4 grid at split 7 (128 rows -- a small grid, but below split 8).
_COMBINE_FORM_CELLS = [
    pytest.param(GEOM_SMALL, 1, (2052, 300), 8, "decode_grid", 51, id="B2-Sq1-split8-decode-grid"),
    pytest.param(GEOM_SMALL, 1, (2052, 300), 4, "own", 52, id="B2-Sq1-split4-own-forms"),
    pytest.param(GEOM_FLASH_NEXT, 1, (2052, 300, 1000, 77, 4097, 64), 4, "lean_walk", 53, id="B6-Sq1-24-2-split4-lean-walk"),
    pytest.param(GEOM_SMALL, 4, (2052, 300, 1000, 77), 7, "own", 54, id="B4-Sq4-split7-own-forms"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, s_q, kv_lens, split, form, seed", _COMBINE_FORM_CELLS)
def test_the_split_combine_takes_the_pinned_form_per_grid_and_two_executes_are_bitwise(geom_kw, s_q, kv_lens, split, form, seed):
    """The block's sparse stage compiles its split combine with the FORM the adapter pins per plan on the sparse grid ``(B, H_q, S_q)``
    (``SparseGqaFwdDslSm107.combine_form``: the dense adapter's rule -- the decode-grid kernel on a small grid from split 8, the lean
    group walk on an overflow grid at split <= 4, the module's own forms otherwise); the CUDA profiler's census of one execute names
    exactly that combine kernel; the step is within the budget vs the oracle (the gate in the combine), two executes bitwise."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    want = {"decode_grid": comb.DECODE_GRID_AUTO, "lean_walk": comb.DECODE_GRID_LEAN_WALK, "own": comb.DECODE_GRID_OFF}[form]
    case = _Case(geom_kw, s_q, kv_lens, page=16, seed=seed, split=split, fuse_gate=True, return_lse=True)
    case.compile()
    impl = case.blk._sdpa._impl
    assert case.blk.decode_split_kv == split == impl.split_kv and impl.combine_form() == want == impl._combine_form, (form, impl._combine_form)
    _two_launches(case, f"combine form {form}")
    kernels, _ = _cupti_kernels(lambda: case.execute())
    if kernels is None:
        pytest.skip("torch.profiler recorded no CUDA event (CUPTI unavailable on this node): the combine kernel's name is unverified here")
    combines = [k for k in kernels if k.startswith("cudnn_kernel__combine")]
    assert len(combines) == 1 and combines[0].startswith(_COMBINE_KERNELS[form]), (form, combines)
    others = tuple(k for f, ks in _COMBINE_KERNELS.items() if f != form for k in ks)
    assert not combines[0].startswith(others), (form, combines)
    print(f"\n{case.label()} combine form {form}: the census saw {combines[0]} among {len(kernels)} kernels")


_FLAG = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"


@requires_rubin
def test_capture_of_the_first_execute_on_the_indexer_band_geometry_replays_bitwise_with_the_slab_on_the_small_m_tile(monkeypatch):
    """The 24 / 2 geometry WITH the indexer band over the pools (N 13952) under the serving stack's defaults: the block compiled with
    ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` DELETED from the environment (the flag is read live at compile; the suite's conftest sets
    it per test) -- both projections capture-safe on the ``jit-only`` route, the slab on the small-M route's 128-row swap-AB tile
    (13952 = 109 x 128), no ranked list, no cuDNN handle created -- then its FIRST execute captured on a NEW stream with no eager
    warm-up under ``set_sync_debug_mode("error")``: the capture launches nothing (``out``, the pools and the raw-key pool untouched),
    the replay is BITWISE a fresh eager run (``out``, the LSE, the raw indexer key's pool rows), and a new step through the captured
    tensors (one length a token shorter and one a token longer, a re-laid-out table, new lists / slots / inputs) replays bitwise the
    eager run over it."""
    from cudnn.gated_attention_block.kernels import proj_gemm
    from cudnn.gated_attention_block.kernels.proj_gemm import small_m_tile_config

    monkeypatch.delenv(_FLAG, raising=False)
    handles_before = dict(proj_gemm._GRAPH_HANDLES)
    case = _Case(GEOM_FLASH_NEXT, 1, (4096, 60), page=16, seed=51, split=8, fuse_gate=True, return_lse=True, index_band=True)  # fresh: never executed
    assert case.blk.geom.n_qkvg == 13952 and case.blk._proj.capture_safe and case.blk._out_proj.capture_safe
    case.blk.check_support()
    case.blk.compile()
    for st in (case.blk._proj, case.blk._out_proj):
        assert st._plan.capture_safe and st._plan.route == "jit-only" and st._plan.jit is not None and st._plan.graph.plans == [], (st.name, st._plan.route)
    slab = case.blk._proj._plan.tile_config_name
    assert slab == small_m_tile_config(2, 13952, 2560, dtype=BF16) and slab.startswith("CONFIG_sm100_128x8x128_") and slab.endswith("_swapAB"), slab
    assert dict(proj_gemm._GRAPH_HANDLES) == handles_before, "the jit-only route created a cuDNN handle"
    case.ws = torch.full((case.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    case.lse.fill_(float("nan"))
    k_before, ix_before = _written(case.k_cache), _written(case.index_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before and _written(case.index_cache) == ix_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_lse, replayed_ix = case.out.clone(), case.lse.clone(), _bits(case.index_cache).clone()
        case.check(replayed, lse=replayed_lse, what="band replay")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.index_cache.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert (
            torch.equal(_bits(case.out), _bits(replayed)) and torch.equal(case.lse, replayed_lse) and torch.equal(_bits(case.index_cache), replayed_ix)
        ), "the replay of the captured first execute differs from the eager run"
        # a NEW step through the captured tensors: one length a token shorter and one a token longer (every slot, list and tail moves;
        # the page counts stay 256 + 4, so the captured table keeps its shape), a re-laid-out table, new lists / counts / slots / h / RoPE
        # rows / weights -- the oracle of the new step through a sibling case over the same shapes
        sib = _Case(GEOM_FLASH_NEXT, 1, (4095, 61), page=16, seed=53, split=8, fuse_gate=True, return_lse=True, index_band=True, n_pages=case.n_pages)
        assert sib.block_table.shape == case.block_table.shape and sib.k_cache.shape == case.k_cache.shape
        for name in ("k_cache", "v_cache", "block_table", "kv_lens_t", "slot_mapping", "block_ids", "block_lens", "h", "cos", "sin", "index_cache"):
            getattr(case, name).copy_(getattr(sib, name))
        case.w["w_qkvg"].copy_(sib.w["w_qkvg"])
        case.w["w_o"].copy_(sib.w["w_o"])
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.ref_out, case.ref_o_gated, case.ref_lse, case.live_rows, case.kv_lens = sib.ref_out, sib.ref_o_gated, sib.ref_lse, sib.live_rows, sib.kv_lens
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2, replayed2_lse, replayed2_ix = case.out.clone(), case.lse.clone(), _bits(case.index_cache).clone()
        case.check(replayed2, lse=replayed2_lse, what="band replay over new lengths / table / lists / slots / inputs")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.index_cache.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert (
            torch.equal(_bits(case.out), _bits(replayed2)) and torch.equal(case.lse, replayed2_lse) and torch.equal(_bits(case.index_cache), replayed2_ix)
        ), "the replay over new lengths / tables / lists / slots differs from the eager run over them"
    finally:
        graph.reset()


@requires_rubin
def test_the_band_chain_census_is_one_launch_above_the_band_off_chain_with_the_slab_on_one_small_m_kernel():
    """The CUPTI census of the 24 / 2 decode step (B 1 x KV 4096, S_q 1, split 8, the gate in the combine) with and without the indexer
    band: the band-ON chain is exactly ONE kernel longer -- the raw indexer key's separate cache write -- and its slab GEMM is ONE
    kernel carrying the small-M route's 128-row swap-AB geometry for N 13952 (the kernel symbol spells the config's geometry token).
    The band-OFF slab carries the route's tile for N 13312 at the part's SM count -- the 64-row ``cluster2x1`` tile where its 208
    64-row tiles fit one wave (a 212-SM part), the same 128-row tile where they overflow it (a 204-SM part) -- so the two chains differ
    by the raw-key write alone, or by it plus the slab's name.  No split-K reducer on the slab, no graph-route kernel, no hidden copy
    on either execute path.  The pin names THIS tree's chain -- slab, norm + RoPE, the K / V cache write [+ the raw-key write], the
    sparse tile, the combine, the out projection and its split-K reducer: 7 / 8 kernels; ``out`` within the oracle's budget on both."""
    from collections import Counter

    from cudnn.gemm.frost.tile_config import by_name
    from cudnn.gated_attention_block.kernels.proj_gemm import small_m_tile_config

    cells = {}
    for band in (False, True):
        cells[band] = _Case(GEOM_FLASH_NEXT, 1, (4096,), page=16, seed=61, split=8, fuse_gate=True, index_band=band).compile()
    k_off, m_off = _cupti_kernels(lambda: cells[False].execute())
    k_on, m_on = _cupti_kernels(lambda: cells[True].execute())
    if k_off is None or k_on is None:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    print(f"\nband OFF: {len(k_off)} kernels + {len(m_off)} memset/memcpy:\n  " + "\n  ".join(n[:140] for n in k_off))
    print(f"band ON:  {len(k_on)} kernels + {len(m_on)} memset/memcpy:\n  " + "\n  ".join(n[:140] for n in k_on))
    geo_on = by_name(small_m_tile_config(1, 13952, 2560, dtype=BF16)).geometry_name  # 128x8x128_..._cluster1x1_1ctamma_swapAB
    geo_off = by_name(small_m_tile_config(1, 13312, 2560, dtype=BF16)).geometry_name  # 64x8 cluster2x1 at 212 SMs, 128x8 at 204 SMs
    assert geo_on.startswith("128x8x128_") and "splitK" not in geo_on and geo_on.endswith("_swapAB"), geo_on
    assert sum(geo_on in n for n in k_on) == 1 and sum(geo_off in n for n in k_off) == 1, (geo_on, geo_off)
    same_slab = geo_on == geo_off
    print(f"slab geometry: band ON {geo_on}; band OFF {geo_off} ({'the same tile' if same_slab else 'a different tile'} at this SM count)")
    if not same_slab:
        assert not any(geo_on in n for n in k_off) and not any(geo_off in n for n in k_on)
    extra = Counter(k_on) - Counter(k_off)
    gone = Counter(k_off) - Counter(k_on)
    # + the raw-key write (and the slab's name, where the two N resolve to different tiles)
    assert sum(extra.values()) == (1 if same_slab else 2) and sum(gone.values()) == (0 if same_slab else 1), (extra, gone)
    assert sum(v for n, v in extra.items() if "frost_cache_write" in n) == 1, extra
    assert sum("frost_cache_write" in n for n in k_on) == sum("frost_cache_write" in n for n in k_off) + 1
    assert not any("nvjet" in n.lower() or "cublas" in n.lower() for n in k_on + k_off), "a non-FROST GEMM kernel on the chain"
    assert not [n for n in m_on + m_off if "memcpy" in n.lower()], "a hidden copy on an execute path"
    assert (len(k_off), len(k_on)) == (7, 8), (len(k_off), len(k_on))
    torch.cuda.synchronize()
    for band, case in cells.items():
        case.run()
        case.check(what=f"census band={'on' if band else 'off'}")


# ---------------------------------------------------------------------------
# The in-block indexer's DECODE step over the pools (QsaSpec(index_source="indexer") with kv_source="paged"; 2026-10-21)
# ---------------------------------------------------------------------------

_INDEXER_DECLINE_TOKENS = {
    "indexer_f16": "needs bf16 activations",
    "indexer_head_dim": "packs 128-wide heads",
    "indexer_thd": "scores dense [B, S] prompts",
    "indexer_query_rows": "query rows per sequence (at most 64",
    "indexer_page_size": "a power-of-two count of compressed rows",
    "indexer_e4m3_cache": "e4m3 read arm",
    "indexer_e4m3_pool": "in-kernel dequant arm",
    "indexer_shared_list": "selects per row",
    "shared_list_rows": "tail covers at most two blocks",
    "fuse_prepare_rope_tables_unfused": "the unfused query norm reads the per-token",
}


def test_decode_indexer_declaration_facts_carve_and_the_envelope_rows(monkeypatch):
    """``QsaSpec(index_source="indexer")`` over the pools CONSTRUCTS (no host sync, any device): the decode step ``qsa_indexer`` sits in the
    indexer's slot before the sparse core, the prefill indexer is not built, the list form is per row; under ``fuse_prepare=False`` the
    unfused twin -- the norm + RoPE pass, a K / V-only cache write (no raw-key pool arm), the stage's four launches --, under ``True`` the
    prepare launch with the three paged arms and the stage's two launches; the knob's AUTO reading (``None``: the MEASURED default --
    ``SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS == ((10, 7),)`` since the 2026-10-21 end-to-end gate, so cc 10.7 reads True on the indexer's
    decode step and every other part / the caller-list band form read False; admissibility-aware: out of place -- ``inplace_qkv=False`` -- and on
    the ``rope_dim=0`` geometry it reads False on every part, the unfused twin, never a refusal blaming a knob the caller never set, while an
    explicit True keeps its typed decline there) and its readback; the switch's AUTO reading False and an explicit ``use_pdl=True`` the existing typed
    decline (the switch is the sparse chain's follow-up); the carve = the caller-list band block's + the five slots appended at the END in
    the stage's order; the envelope's indexer rows on a Rubin reading (every one read off the host facts, the FlashInfer-read names pinned
    by signature), ``kv_cache_dtypes`` narrowed to the activation dtype under the indexer; the ring rule's two spellings equal."""
    import dataclasses

    from cudnn.gated_attention_block.api import (
        SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS,
        _align_up,
        _index_state_ring_extra_rows,
        _itemsize,
        _qsa_indexer_decode_slots,
    )

    assert SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS == ((10, 7),)
    with _no_host_sync():
        for s_q, batch in ((1, 1), (4, 2), (16, 1)):
            off = _declare_indexer(GEOM_SMALL, batch, s_q, cap=s_q + 3, fuse_prepare=False)
            on = _declare_indexer(GEOM_SMALL, batch, s_q, cap=s_q + 3, fuse_prepare=True)
            for blk in (off, on):
                assert blk.indexer_decode and blk.sparse_decode and not blk.qsa_shared_list and blk.qsa_list_form == "per_row"
                assert isinstance(blk._indexer_decode, _IndexerDecode) and blk._indexer is None and not blk._sdpa.list_per_sequence
                names = [st.name for st in blk._stages]
                assert names.index("qsa_indexer") == names.index("sdpa_sparse") - (1 if blk.fuse_prepare else 2), names
                assert blk.use_pdl is False and blk.use_pdl_requested is None and blk.pdl_stages == () and not blk.use_pdl_effective
                assert blk._indexer_decode.n_blocks_max == 4 * (16 // BS) and blk._indexer_decode.fused is blk.fuse_prepare
            assert off.fuse_prepare is False and off.fuse_prepare_requested is False and off._prepare is None
            assert off._cache_write is not None and not off._cache_write.index_band  # the K / V-only write: no raw-key pool arm
            assert [st.name for st in off._stages][:3] == ["qkv_gate_proj", "qk_norm_rope", "qsa_indexer"] and "cache_write" in [st.name for st in off._stages]
            assert off._indexer_decode.launches_per_execute() == 4 and off._indexer_decode.pdl_names == ("qsa_compress_step", "qsa_score", "qsa_topk")
            assert on.fuse_prepare is True and on.fuse_prepare_requested is True and isinstance(on._prepare, _QsaPrepare) and on._prepare.decode_arms
            assert on._cache_write is None and on._norm_rope is None and "cache_write" not in [st.name for st in on._stages]
            assert on._indexer_decode.launches_per_execute() == 2 and on._indexer_decode.pdl_names == ("qsa_score", "qsa_topk")
            arms = on._prepare._index_arms()
            assert (arms.raw_pool, arms.query, arms.compress, arms.paged, arms.ring_slots, arms.rope_table, arms.n_commit) == (
                False,
                True,
                True,
                True,
                False,
                False,
                False,
            )
            # the carve: the caller-list band block's layout + the five slots at the END, in the stage's order
            caller = _declare(GEOM_SMALL, batch, s_q, n_pages=4, index_band=True, index_list_form="per_row")[0]
            lay, lay_c = off._layout(), caller._layout()
            assert on._layout() == lay
            e, t = _itemsize(BF16), batch * s_q
            slots = _qsa_indexer_decode_slots(off.geom, batch, s_q, e, off._indexer_decode.n_blocks_max)
            assert [nm for nm, _ in slots] == ["ix_q", "ix_logits", "ix_counts", "ix_ids", "ix_cslot"]
            assert (
                dict(slots)["ix_logits"] == t * off._indexer_decode.n_blocks_max * 4
                and dict(slots)["ix_ids"] == t * 512 * 4
                and dict(slots)["ix_cslot"] == batch * s_q * 4
            )
            acc = lay_c.total_bytes
            for nm, nbytes in slots:
                assert getattr(lay, nm) == acc, nm
                acc += _align_up(nbytes)
            assert lay.total_bytes == acc and lay.ix_kbar == -1 and lay.ix_scores == -1 and lay.ix_cand == -1
            assert lay_c.ix_logits == -1 and lay_c.ix_counts == -1 and lay_c.ix_cslot == -1
        # the switch stays the sparse chain's follow-up: an explicit True is the existing typed decline, AUTO reads False
        with pytest.raises(NotImplementedError, match="use_pdl=True is served for the bf16 / f16 block only"):
            _declare_indexer(GEOM_SMALL, 1, 1, use_pdl=True)
        # the knob's AUTO reading: the measured part (cc 10.7) resolves True on the indexer's decode step, every other part False, the
        # caller-list band form False everywhere; an explicit value wins on every part; the declaration is kept beside the resolved knob
        import cudnn.gated_attention_block.api as _api

        for cc, want in (((10, 7), True), ((10, 0), False), ((10, 3), False), ((9, 0), False), (None, False)):
            monkeypatch.setattr(_api, "_device_cc_of", lambda device, _cc=cc: _cc)
            auto = _declare_indexer(GEOM_SMALL, 1, 1)
            assert auto.fuse_prepare is want and auto.fuse_prepare_requested is None and (auto._prepare is not None) is want, cc
            assert (
                (auto._cache_write is None) is want and auto._indexer_decode.fused is want and auto._indexer_decode.launches_per_execute() == (2 if want else 4)
            ), cc
            assert (
                _declare_indexer(GEOM_SMALL, 1, 1, fuse_prepare=False).fuse_prepare is False
                and _declare_indexer(GEOM_SMALL, 1, 1, fuse_prepare=True).fuse_prepare is True
            ), cc
            band = _declare(GEOM_SMALL, 1, 1, n_pages=4, index_band=True, index_list_form="per_row")[0]
            assert band.fuse_prepare is False and band.fuse_prepare_requested is None and band._prepare is None, cc
            # the AUTO reading is ADMISSIBILITY-AWARE (never True where an explicit True is declined): out of place the fused launch is refused
            # by its own row, so the measured default reads False there on EVERY part -- the unfused twin (the K / V-only write, the stage's
            # four launches) -- and an explicit True keeps its ValueError; the fused stage's rope_dim row (its check_support decline) the same way
            oop = _declare_indexer(GEOM_SMALL, 1, 1, inplace_qkv=False)
            assert oop.fuse_prepare is False and oop.fuse_prepare_requested is None and oop._prepare is None and not oop.inplace_qkv, cc
            assert oop._cache_write is not None and oop._indexer_decode.fused is False and oop._indexer_decode.launches_per_execute() == 4, cc
            with pytest.raises(ValueError, match="inplace_qkv=False"):
                _declare_indexer(GEOM_SMALL, 1, 1, inplace_qkv=False, fuse_prepare=True)
            rope0 = _declare_indexer(GEOM_SMALL, 1, 1, geom_over=dict(rope_dim=0, qk_norm=True))
            assert rope0.fuse_prepare is False and rope0.fuse_prepare_requested is None and rope0._prepare is None and rope0.geom.rope_dim == 0, cc
            with pytest.raises(ValueError, match="fuse_prepare=True with geometry.rope_dim=0"):
                _declare_indexer(GEOM_SMALL, 1, 1, geom_over=dict(rope_dim=0, qk_norm=True), fuse_prepare=True)._prepare.check_support()
        monkeypatch.undo()
        # the ring rule's DSL-free spelling
        assert _index_state_ring_extra_rows(BS) == 3
        # the envelope on a Rubin reading: the indexer rows a serving stack READS (never re-spells), pinned by name
        geom = _declare_indexer(GEOM_SMALL, 1, 1).geom
        env = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV, device_cc=_SM107)
        d = env.decode
        assert d.served and d.sparse_core
        assert d.index_sources == ("caller", "indexer") and d.index_cache_dtypes == (BF16,) and d.index_head_dims == (128,) and d.index_kv_heads == (1,)
        assert d.index_block_sizes == (4,) and d.index_top_k_range == (4, 512, 4) and d.index_query_rows_max == 64 and d.index_state_rows_extra == 3
        assert d.index_page_sizes == (16, 32, 64, 128) and d.fuse_prepare_default is True and d.kv_cache_dtypes == (BF16,)  # the measured part
        assert d.index_list_forms == ("per_row",) and d.index_list_form_default == "per_row" and d.pdl_stages == () and d.pdl_default is False
        assert set(_INDEXER_DECLINE_TOKENS) <= set(d.declines) and all(tok in d.declines[nm] for nm, tok in _INDEXER_DECLINE_TOKENS.items())
        # the record's tail is append-only (a serving stack reads these by name; a new field goes after `fuse_prepare_default`)
        assert [f.name for f in dataclasses.fields(DecodeEnvelope)][-12:] == [
            "index_list_forms",
            "index_list_form_default",
            "index_sources",
            "index_cache_dtypes",
            "index_head_dims",
            "index_kv_heads",
            "index_block_sizes",
            "index_top_k_range",
            "index_query_rows_max",
            "index_state_rows_extra",
            "index_page_sizes",
            "fuse_prepare_default",
        ]
        assert GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV, device_cc=(10, 0)).decode.fuse_prepare_default is False  # an unmeasured part
        f16 = GatedAttentionBlockFwd.serving_envelope(geom, F16, _DEV, device_cc=_SM107).decode
        assert f16.index_sources == ("caller",) and f16.index_cache_dtypes == () and f16.fuse_prepare_default is False
        caller_env = GatedAttentionBlockFwd.serving_envelope(_geom(GEOM_SMALL)[0], BF16, _DEV, device_cc=_SM107).decode
        assert (
            caller_env.index_sources == ("caller", "indexer")
            and caller_env.kv_cache_dtypes == (BF16, torch.float8_e4m3fn)
            and caller_env.fuse_prepare_default is False
        )
        dense_env = GatedAttentionBlockFwd.serving_envelope(
            GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True), BF16, _DEV, device_cc=_SM107
        ).decode
        assert (
            dense_env.index_sources == ()
            and dense_env.index_page_sizes == ()
            and dense_env.index_query_rows_max == 0
            and "indexer_f16" not in dense_env.declines
        )


def test_decode_indexer_typed_declines_equal_the_envelope_row_by_row():
    """One trip per row of the decode indexer's declines, each on a REAL declaration: the raised text EQUALS the envelope's row and carries
    the row's one shared token -- ``indexer_query_rows`` (S_q 17 at 4 heads; 8 heads at S_q 9), ``indexer_page_size`` (page 48),
    ``indexer_e4m3_cache``, ``indexer_e4m3_pool``, ``indexer_shared_list``, ``fuse_prepare_rope_tables_unfused``, ``indexer_f16``,
    ``indexer_head_dim``, ``shared_list_rows``; the ring rule (``cap < S_q + 3``), a missing sample, the commit flag, the
    side caches off their form, and the samples / flags on a caller-list or a dense declaration -- ``ValueError`` naming them."""
    import re

    rows = {
        "indexer_query_rows": lambda: _declare_indexer(GEOM_SMALL, 1, 17, cap=20),
        "indexer_page_size": lambda: _declare_indexer(GEOM_SMALL, 1, 1, page=48),
        "indexer_e4m3_cache": lambda: _declare_indexer(GEOM_SMALL, 1, 1, qsa_over=dict(index_cache_dtype=torch.float8_e4m3fn)),
        "indexer_shared_list": lambda: _declare_indexer(GEOM_SMALL, 1, 1, qsa_over=dict(index_list_form="shared")),
        "fuse_prepare_rope_tables_unfused": lambda: _declare_indexer(GEOM_SMALL, 1, 1, rope_position_tables_only=True, fuse_prepare=False),
        "indexer_f16": lambda: _declare_indexer(GEOM_SMALL, 1, 1, dtype=F16),
        "indexer_head_dim": lambda: _declare_indexer(GEOM_SMALL, 1, 1, qsa_over=dict(index_head_dim=64)),
        "shared_list_rows": lambda: _declare(GEOM_SMALL, 2, 8, n_pages=4, index_list_form="shared"),
    }
    with _no_host_sync():
        for name, trip in rows.items():
            geom_dtype = F16 if name == "indexer_f16" else BF16
            probe = (
                QsaSpec(index_source="indexer", index_band=True, index_head_dim=64)
                if name == "indexer_head_dim"
                else QsaSpec(index_source="indexer", index_band=True, index_list_form="shared") if name == "indexer_shared_list" else QsaSpec(index_band=True)
            )
            env = GatedAttentionBlockFwd.serving_envelope(
                GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True, qsa=probe), geom_dtype, _DEV, device_cc=_SM107
            )
            with pytest.raises(NotImplementedError, match=re.escape(_INDEXER_DECLINE_TOKENS[name])) as e:
                trip()
            assert str(e.value) == env.decode.declines[name], (name, str(e.value))
        # the e4m3 pool: the sample pools in e4m3 under the attribute
        kc, vc = (torch.empty(4, 2, 16, D, dtype=torch.float8_e4m3fn, device=_DEV) for _ in range(2))
        with pytest.raises(NotImplementedError, match=re.escape(_INDEXER_DECLINE_TOKENS["indexer_e4m3_pool"])) as e:
            _declare_indexer(GEOM_SMALL, 1, 1, kv_cache_dtype=torch.float8_e4m3fn, sample_k_cache=kc, sample_v_cache=vc)
        env = GatedAttentionBlockFwd.serving_envelope(_declare_indexer(GEOM_SMALL, 1, 1).geom, BF16, _DEV, device_cc=_SM107)
        assert str(e.value) == env.decode.declines["indexer_e4m3_pool"]
        # the query-row envelope at 8 heads: S_q 8 x 8 = 64 rows serves, 9 x 8 does not
        assert _declare_indexer(GEOM_SMALL, 1, 8, cap=11, qsa_over=dict(index_heads=8))._indexer_decode is not None
        with pytest.raises(NotImplementedError, match="at most 8 rows per sequence"):
            _declare_indexer(GEOM_SMALL, 1, 9, cap=12, qsa_over=dict(index_heads=8))
        # the ring rule, the commit flag, a missing sample, the side caches off their form, the flags' type
        with pytest.raises(ValueError, match=r"needs at least S_q \+ 4 - 1 = 7 rows"):
            _declare_indexer(GEOM_SMALL, 1, 4, cap=6)
        assert _declare_indexer(GEOM_SMALL, 1, 4, cap=7)._indexer_decode is not None  # the bound met with equality
        with pytest.raises(NotImplementedError, match="index_commit_present=True .* is a follow-up"):
            _declare_indexer(GEOM_SMALL, 1, 1, index_commit_present=True)
        with pytest.raises(ValueError, match="needs sample_index_kv_cache, sample_index_state_cache, sample_index_state_block_table at declaration"):
            _declare_indexer(GEOM_SMALL, 1, 1, samples=False)
        samples = _indexer_samples(1, cap=8)
        with pytest.raises(ValueError, match="must hold a page's compressed rows contiguously"):
            _declare_indexer(GEOM_SMALL, 1, 1, sample_index_kv_cache=torch.empty(4, 4, 1, 256, dtype=BF16, device=_DEV)[..., :128])
        with pytest.raises(ValueError, match=r"must be \[blocks >= 1, exactly 4, 1, 128\]"):
            _declare_indexer(GEOM_SMALL, 1, 1, sample_index_kv_cache=torch.empty(4, 8, 1, 128, dtype=BF16, device=_DEV))
        with pytest.raises(ValueError, match="never quantized"):
            _declare_indexer(GEOM_SMALL, 1, 1, sample_index_state_cache=samples["sample_index_state_cache"].to(torch.float8_e4m3fn))
        with pytest.raises(ValueError, match="blocks contiguous"):
            _declare_indexer(
                GEOM_SMALL, 1, 1, index_state_slot_mapping_present=True, sample_index_state_cache=torch.empty(2, 12, 1, 128, dtype=BF16, device=_DEV)[:, :8]
            )
        with pytest.raises(ValueError, match=r"must be a \[B=1, 1\] int32 tensor"):
            _declare_indexer(GEOM_SMALL, 1, 1, sample_index_state_block_table=torch.zeros(1, 2, dtype=torch.int32, device=_DEV))
        with pytest.raises(TypeError, match="rope_position_tables_only must be a bool"):
            _declare_indexer(GEOM_SMALL, 1, 1, rope_position_tables_only=1)
        # the samples and the flags off the form: a caller-list block, a dense block -- refused by name
        with pytest.raises(ValueError, match="refused rather than silently ignored"):
            _declare(GEOM_SMALL, 1, 1, n_pages=4, index_band=True, **_indexer_samples(1))
        with pytest.raises(ValueError, match="index_state_slot_mapping_present.*refused rather than silently ignored"):
            _declare(GEOM_SMALL, 1, 1, n_pages=4, index_state_slot_mapping_present=True)
        with pytest.raises(ValueError, match="rope_position_tables_only.*refused rather than silently ignored"):
            _declare(GEOM_SMALL, 1, 1, n_pages=4, rope_position_tables_only=True)
        # the caller-list block and the e4m3 pool: the per-row list form over e4m3 pools is the gathered cast's decline
        kc8, vc8 = (torch.empty(4, 2, 16, D, dtype=torch.float8_e4m3fn, device=_DEV) for _ in range(2))
        with pytest.raises(NotImplementedError, match="index_list_form='per_row'.*in-kernel dequant arm"):
            _declare(
                GEOM_SMALL,
                1,
                1,
                n_pages=4,
                index_list_form="per_row",
                kv_cache_dtype=torch.float8_e4m3fn,
                sample_k_cache=kc8,
                sample_v_cache=vc8,
                fuse_norm_rope=False,
            )


def test_decode_indexer_execute_inputs_are_required_and_refused_in_both_directions():
    """On an UNCOMPILED decode-indexer block under ``set_sync_debug_mode("error")`` (the inputs are a declaration-vs-argument contract: no
    plan, no sync): the well-formed set stops at the block's own compile gate; each required input missing is a ``ValueError`` naming it;
    ``index_k_raw`` (the raw key lives in the ring), ``index_k_compressed`` (the cache is paged), ``block_ids`` / ``block_lens``, ``n_commit``
    and the ring slots without their flag are refused by name; under the flag the slots are required and must share ``slot_mapping``'s
    dtype; a position table one row short of ``max_pages x P`` names the ceiling; ``cos`` / ``sin`` are refused under the table arm and
    required without it; a side cache off its form or off the 16-byte base is refused BEFORE any stage; a caller-list block refuses every
    one of the seven."""
    with _no_host_sync():
        blk = _declare_indexer(GEOM_SMALL, 2, 1, cap=4)
        slots = _declare_indexer(GEOM_SMALL, 2, 1, cap=4, index_state_slot_mapping_present=True)
        table = _declare_indexer(GEOM_SMALL, 2, 1, cap=4, fuse_prepare=True, rope_position_tables_only=True)
    geom = blk.geom
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
    z = lambda *shape: torch.zeros(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    kc, vc = _pools(4, 2, 16, "hnd", BF16)
    base = dict(
        k_cache=kc,
        v_cache=vc,
        block_table=i32(2, 4),
        kv_lens=i32(2),
        slot_mapping=i32(2),
        w_iq_norm=torch.ones(128, dtype=BF16, device=_DEV),
        w_ik_norm=torch.ones(128, dtype=BF16, device=_DEV),
        index_kv_cache=z(4, 4, 1, 128),
        index_state_cache=z(3, 4, 1, 128),
        index_state_block_table=i32(2, 1),
        cos_pos=z(1, 64, ROPE),
        sin_pos=z(1, 64, ROPE),
    )

    def run(b=blk, *, tables=True, **over):
        kw = dict(base)
        kw.update(over)
        for nm in [nm for nm, x in kw.items() if x is None]:
            del kw[nm]
        cs = (z(2, 1, ROPE), z(2, 1, ROPE)) if tables else (None, None)
        return b.execute(
            z(2, 1, geom.d_model),
            z(geom.n_qkvg, geom.d_model),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            *cs,
            z(geom.d_model, geom.h_q * D),
            z(2, 1, geom.d_model),
            torch.empty(1 << 20, dtype=torch.uint8, device=_DEV),
            **kw,
        )

    with _no_host_sync():
        with pytest.raises(RuntimeError, match="compile"):
            run()
        for nm in ("index_kv_cache", "index_state_cache", "index_state_block_table", "cos_pos", "sin_pos", "w_iq_norm", "w_ik_norm"):
            with pytest.raises(ValueError, match=nm):
                run(**{nm: None})
        with pytest.raises(ValueError, match="raw key lives in the ring"):
            run(index_k_raw=z(4, 16, 128))
        with pytest.raises(ValueError, match="PAGED cache"):
            run(index_k_compressed=z(2, 4, 128))
        with pytest.raises(ValueError, match="derives the selection"):
            run(block_ids=i32(2, 512))
        with pytest.raises(ValueError, match="n_commit"):
            run(n_commit=i32(2))
        with pytest.raises(ValueError, match="declared index_state_slot_mapping_present=False"):
            run(index_state_slot_mapping=i32(2))
        with pytest.raises(ValueError, match="index_state_slot_mapping_present=True: execute needs index_state_slot_mapping"):
            run(slots)
        with pytest.raises(ValueError, match="share the slot mapping's dtype"):
            run(slots, index_state_slot_mapping=torch.zeros(2, dtype=torch.int64, device=_DEV))
        with pytest.raises(RuntimeError, match="compile"):
            run(slots, index_state_slot_mapping=i32(2))
        with pytest.raises(RuntimeError, match="compile"):
            run(slots, slot_mapping=torch.zeros(2, dtype=torch.int64, device=_DEV), index_state_slot_mapping=torch.zeros(2, dtype=torch.int64, device=_DEV))
        with pytest.raises(ValueError, match="the table is the ceiling on positions"):
            run(cos_pos=z(1, 63, ROPE), sin_pos=z(1, 63, ROPE))
        with pytest.raises(ValueError, match="cos_pos and sin_pos must share a shape"):
            run(cos_pos=z(1, 64, ROPE), sin_pos=z(1, 65, ROPE))
        with pytest.raises(ValueError, match="cos / sin are refused under rope_position_tables_only=True"):
            run(table)
        with pytest.raises(RuntimeError, match="compile"):
            run(table, tables=False)
        with pytest.raises(ValueError, match="cos / sin .* are required"):
            run(tables=False)
        with pytest.raises(ValueError, match="needs at least"):
            run(index_state_cache=z(3, 3, 1, 128))
        with pytest.raises(ValueError, match=r"must be \[blocks >= 1, exactly 4, 1, 128\]"):
            run(index_kv_cache=z(4, 8, 1, 128))
        flat = z(4 * 4 * 128 + 8)
        with pytest.raises(ValueError, match="16-byte aligned address"):
            run(index_kv_cache=flat[4 : 4 + 4 * 4 * 128].view(4, 4, 1, 128))  # 8 bytes off the vector base
        with pytest.raises(RuntimeError, match="compile"):
            run(index_kv_cache=flat[8 : 8 + 4 * 4 * 128].view(4, 4, 1, 128))  # the aligned storage-offset control passes every form check
        with pytest.raises(ValueError, match="never quantized"):
            run(index_state_cache=z(3, 4, 1, 128).to(torch.float8_e4m3fn))
        # the mirror image: a caller-list band block refuses the seven rather than ignoring them
        caller, aux = _declare(GEOM_SMALL, 2, 1, n_pages=4, index_band=True)
        for nm in ("index_kv_cache", "index_state_cache", "index_state_block_table", "index_state_slot_mapping", "cos_pos", "sin_pos", "n_commit"):
            with pytest.raises(ValueError, match=f"{nm}.*refused rather than silently ignored"):
                caller.execute(
                    z(2, 1, geom.d_model),
                    z(geom.n_qkvg, geom.d_model),
                    torch.ones(D, dtype=BF16, device=_DEV),
                    torch.ones(D, dtype=BF16, device=_DEV),
                    z(2, 1, ROPE),
                    z(2, 1, ROPE),
                    z(geom.d_model, geom.h_q * D),
                    z(2, 1, geom.d_model),
                    torch.empty(1 << 20, dtype=torch.uint8, device=_DEV),
                    block_ids=i32(2, 512),
                    k_cache=aux["k_cache"],
                    v_cache=aux["v_cache"],
                    block_table=aux["block_table"],
                    kv_lens=i32(2),
                    slot_mapping=i32(2),
                    index_k_raw=z(8, 16, 128),
                    **{nm: base.get(nm, i32(2))},
                )


_TIE_TOL = 1e-4  # the scorer suite's tie window at the k-th boundary (fp32 logits of O(1) vs the fp64 reference)
_SENT_I32 = -7


def _set_vs_topk(in_set: torch.Tensor, scores: torch.Tensor, top_k: int, tol: float) -> tuple:
    """Per row: the SET ``in_set`` ``[R, N]`` against the top-k of ``scores`` ``[R, N]`` (fp64; ``-inf`` = not visible); every member of the
    symmetric difference accepted iff its score lies within ``tol`` of the row's k-th score (the margin rule).  Returns ``(flips, rows,
    max_gap)``; asserts the rule (the end-to-end module's helper)."""
    rows, n = scores.shape
    k_eff = min(top_k, n)
    tk = torch.topk(scores, k_eff, dim=-1)
    exp_valid = torch.isfinite(tk.values)
    in_exp = torch.zeros(rows, n + 1, dtype=torch.bool, device=scores.device)
    in_exp.scatter_(1, torch.where(exp_valid, tk.indices, torch.full_like(tk.indices, n)), torch.ones_like(exp_valid))
    in_exp = in_exp[:, :n]
    kth = torch.where(exp_valid, tk.values, torch.full_like(tk.values, float("inf"))).min(dim=-1).values
    differs = in_set ^ in_exp
    flips = int(differs.sum())
    if flips == 0:
        return 0, 0, 0.0
    gap = (scores - kth[:, None]).abs()[differs]
    assert torch.isfinite(gap).all(), f"{int((~torch.isfinite(gap)).sum())} flipped ids lie outside the row's visible range"
    assert bool((gap <= tol).all()), f"{flips} ids differ from the top-k beyond the margin: max gap {float(gap.max()):.3e} > tol {tol:.3e}"
    return flips, int(differs.any(dim=-1).sum()), float(gap.max())


class _IndexerCase(_Case):
    """One decode step of the IN-BLOCK INDEXER over the pools: ``_Case``'s batch of sequences (the whole-sequence inputs per entry, the
    dense reference's Q / K / V / gate, the NaN pools holding the oracle's prefix K / V, the table, the slots) plus the indexer's state --
    the band's raw keys of every position from the oracle's own projection, the PREFIX blocks compressed through the compress kernel
    into the paged compressed-key cache (page 0 the null page, NaN everywhere unwritten), the ring primed with the last ``cap`` raw
    keys below the step (block 0 the null block), the position tables over the table's whole reach -- and the block declared
    ``QsaSpec(index_source="indexer")`` over them.  The oracle is formed AFTER the run from the KERNEL's own list per row (the
    two-stage oracle's second stage); ``check_selection`` is its first stage."""

    def __init__(
        self,
        geom_kw,
        s_q,
        kv_lens,
        *,
        page=16,
        layout="hnd",
        dtype=BF16,
        seed=0,
        top_k=512,
        return_lse=False,
        fuse_gate=False,
        split=None,
        n_pages=None,
        fuse_prepare=None,
        cap=None,
        slot_form=False,
        rope_tables_only=False,
        outputs=True,
        lens_out=True,
    ):
        self.s_q, self.kv_lens, self.page, self.layout, self.dtype, self.top_k = int(s_q), [int(x) for x in kv_lens], page, layout, dtype, top_k
        self.block_lens_given, self.return_lse, self.fuse_gate, self.index_band = False, bool(return_lse), bool(fuse_gate), True
        self.index_list_form, self.shared, self.fuse_prepare = None, False, fuse_prepare
        self.slot_form, self.rope_tables_only, self.outputs, self.lens_out = bool(slot_form), bool(rope_tables_only), bool(outputs), bool(lens_out)
        kw_g = dict(geom_kw, causal_bottom_right=True)
        self.geom = GatedAttentionBlockGeometry(**kw_g, qsa=QsaSpec(top_k=top_k, index_source="indexer", index_band=True))
        self.ref_geom = RefGeometry(**kw_g)
        geom, ref_geom = self.geom, self.ref_geom
        h_q, h_kv, d_model, d_i, h_i = geom.h_q, geom.h_kv, geom.d_model, geom.qsa.index_head_dim, geom.qsa.index_heads
        G = h_q // h_kv
        B = len(self.kv_lens)
        g = torch.Generator(device=_DEV).manual_seed(seed)
        self.gen = g

        def randn(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        n_dense = sum(geom.qkvg_block_widths[:4])
        w_qkvg = randn(geom.n_qkvg, d_model)
        self.w = dict(
            w_qkvg=w_qkvg,
            w_q_norm=torch.ones(D, dtype=dtype, device=_DEV),
            w_k_norm=torch.ones(D, dtype=dtype, device=_DEV),
            w_o=randn(d_model, h_q * D),
            w_iq_norm=(1.0 + 0.03 * randn(d_i, std=1.0).float()).to(dtype),
            w_ik_norm=(1.0 + 0.02 * randn(d_i, std=1.0).float()).to(dtype),
        )
        self.w_band_k = w_qkvg[n_dense + h_i * d_i :]  # [d_i, d_model]: the band's raw-key head
        L_max = max(max(self.kv_lens), s_q)
        self.max_pages = max(-(-L_max // page), 1)
        self.n_pos = self.max_pages * page  # the position tables cover the table's whole reach (the RoPE-table ceiling met with equality)
        cos_t, sin_t = _rope_tables(self.n_pos, dtype, ref_geom)
        self.cos_pos, self.sin_pos = cos_t, sin_t  # [1, n_pos, ROPE]
        scale = geom.scale
        self.scale, self.G = scale, G
        self.h = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.cos = cos_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.sin = sin_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.ref_out = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.ref_o_gated = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_lse = torch.full((B, h_q, s_q), float("-inf"), dtype=torch.float32, device=_DEV)
        self.ref_rows, self.refs, self.k_raw_full = {}, {}, {}
        self.live_rows = torch.zeros(B, s_q, dtype=torch.bool, device=_DEV)
        for b, L in enumerate(self.kv_lens):
            pos = torch.arange(s_q, device=_DEV) + (L - s_q)
            live = pos >= 0
            self.live_rows[b] = live
            if L == 0:
                continue
            h_full = randn(1, L, d_model, std=1.0)
            ref = gated_attention_block_reference(
                h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], self.w["w_o"], ref_geom
            )
            self.refs[b] = ref
            self.ref_rows[b] = (ref.k[0], ref.v[0])
            # the band's raw key of every position: the oracle's own projection, ONE rounding (the prefix blocks and the ring are built from it)
            self.k_raw_full[b] = (h_full[0].float() @ self.w_band_k.float().t()).to(dtype)  # [L, d_i]
            p_live = pos[live]
            self.h[b, live] = h_full[0, p_live]
            self.cos[b, live] = cos_t[0, p_live]
            self.sin[b, live] = sin_t[0, p_live]
        # pages: block 0 reserved (NULL); each live entry gets ceil(L / P) pages from a permutation of 1 .. n_pages - 1; -1 past them
        need = sum(-(-L // page) for L in self.kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=g, device=_DEV) + 1).tolist()
        self.block_table = torch.full((B, self.max_pages), -1, dtype=torch.int32, device=_DEV)
        it = iter(perm)
        slots = torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _pools(self.n_pages, h_kv, page, layout, dtype)
        self.index_cache = None  # the raw-key pool of the caller-list band form: nothing here (the raw key lives in the ring)
        self.rpp = page // BS
        self.cap = int(cap) if cap is not None else (4 if s_q == 1 else 8)  # the serving stacks' ring: 4 without speculative rows, 8 with 1 .. 4
        self.index_kv_cache = torch.full((self.n_pages, self.rpp, 1, d_i), float("nan"), dtype=dtype, device=_DEV)
        self.index_state_cache = torch.full((B + 1, self.cap, 1, d_i), float("nan"), dtype=dtype, device=_DEV)
        self.index_state_block_table = torch.zeros(B, 1, dtype=torch.int32, device=_DEV)
        self.pages = {}
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                self.block_table[b] = 0  # the dead entry: the NULL block / the null ring block, slots -1
                continue
            pages = [next(it) for _ in range(-(-L // page))]
            self.pages[b] = pages
            self.block_table[b, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=_DEV)
            self.index_state_block_table[b, 0] = b + 1
            k_rows, v_rows = self.ref_rows[b]
            pos0 = L - s_q
            for p in range(max(pos0, 0)):  # the PREFIX: the oracle's rows into the pools
                pg, off = pages[p // page], p % page
                self.k_cache[pg, :, off] = k_rows[p]
                self.v_cache[pg, :, off] = v_rows[p]
            for j in range(s_q):  # the step's live tokens: written through by the block
                p = pos0 + j
                if p >= 0:
                    slots[b * s_q + j] = pages[p // page] * page + p % page
            # the ring: the last `cap` raw keys below the step, position-indexed
            for p in range(max(0, pos0 - self.cap), max(pos0, 0)):
                self.index_state_cache[b + 1, p % self.cap, 0] = self.k_raw_full[b][p]
        self.slot_mapping = slots
        self.kv_lens_t = torch.tensor(self.kv_lens, dtype=torch.int32, device=_DEV)
        self.prefix_bits = (_bits(self.k_cache).clone(), _bits(self.v_cache).clone())
        # the ring's WRITE rows as the stack spells them (block x cap + pos % cap, -1 on a dead row), slot_mapping's dtype
        pos_all = torch.tensor([[L - s_q + j for j in range(s_q)] for L in self.kv_lens], device=_DEV, dtype=torch.long)
        blk_of = self.index_state_block_table.long()
        self.ring_slots = (
            torch.where((pos_all >= 0) & (blk_of > 0), blk_of * self.cap + pos_all % self.cap, torch.full_like(pos_all, -1)).reshape(-1).to(torch.int32)
        )
        self.block_ids_out = torch.full((B, s_q, top_k), _SENT_I32, dtype=torch.int32, device=_DEV) if outputs else None
        self.block_lens_out = torch.full((B, s_q), _SENT_I32, dtype=torch.int32, device=_DEV) if lens_out else None
        kw = dict(
            paged_kv_page_size=page,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            sample_index_kv_cache=self.index_kv_cache,
            sample_index_state_cache=self.index_state_cache,
            sample_index_state_block_table=self.index_state_block_table,
            fuse_gate=fuse_gate,
            return_lse=return_lse,
            decode_split_kv=split,
            fuse_prepare=fuse_prepare,
            index_state_slot_mapping_present=self.slot_form,
            rope_position_tables_only=self.rope_tables_only,
        )
        self.blk_kw = kw
        self.out = torch.empty(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.lse = torch.empty(B, h_q, s_q, dtype=torch.float32, device=_DEV) if return_lse else None
        tables = (None, None) if self.rope_tables_only else (self.cos, self.sin)
        self.blk = GatedAttentionBlockFwd(self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], *tables, self.w["w_o"], self.out, self.geom, **kw)
        assert (
            isinstance(self.blk._sdpa, _SparseSdpa)
            and self.blk.indexer_decode
            and not self.blk.qsa_shared_list
            and isinstance(self.blk._indexer_decode, _IndexerDecode)
        )
        self.ws = None
        self._compress = {}

    def prime_prefix(self):
        """The PREFIX blocks of every live sequence (``j < floor(pos0 / 4)``) into the paged compressed-key cache through the compress kernel
        (the prefill compress over the oracle's raw band: what a FROST prefill would have written), once per case."""
        if getattr(self, "_primed", False):
            return
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            nb = max(L - self.s_q, 0) // BS
            if nb == 0:
                continue
            kb = self.compress(self.k_raw_full[b][None, : nb * BS], self.cos_pos, self.sin_pos)  # [1, nb, d_i]
            for j in range(nb):
                self.index_kv_cache[self.block_table[b, j // self.rpp].item(), j % self.rpp, 0] = kb[0, j]
        torch.cuda.synchronize()
        self._primed = True

    def compress(self, k_raw, cos, sin, pool=BS):
        """The compress kernel (``kernels/qsa_compress.py``, the prefill form) over ``k_raw [B', S, d_i]`` at ``pool`` with the tables
        ``cos / sin [1 or B', S_pos, ROPE]`` -> ``[B', floor(S / pool), d_i]``; the one-rounding chain the block's own arms compute."""
        from cudnn.gated_attention_block.kernels.qsa_compress import compile_qsa_compress, run_qsa_compress

        d_i = self.geom.qsa.index_head_dim
        if pool not in self._compress:
            self._compress[pool] = compile_qsa_compress(dtype=self.dtype, d=d_i, rope_dim=ROPE, pool=pool, has_seq_lens=False, table_dtype=self.dtype)
        k_raw = k_raw.contiguous()
        out = torch.full((int(k_raw.shape[0]), int(k_raw.shape[1]) // pool, d_i), float("nan"), dtype=self.dtype, device=_DEV)
        w = self.w["w_ik_norm"] if pool == BS else self.w["w_iq_norm"]
        run_qsa_compress(
            self._compress[pool],
            k_raw,
            out,
            w,
            cos,
            sin,
            None,
            eps=float(self.geom.qsa.index_norm_eps),
            norm_weight_offset=0.0,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        return out

    def compile(self):
        self.prime_prefix()
        return super().compile()

    def execute(self, *, out=None, ws=None, lse=None, h=None, cos=None, sin=None, kv_lens=None, block_table=None, slot_mapping=None, stream=None, **over):
        out = self.out if out is None else out
        # explicit cos / sin always reach the block (the refusal cell hands them to a position-table declaration); else the declared form's
        tables = (cos, sin) if (cos is not None or sin is not None) else ((None, None) if self.rope_tables_only else (self.cos, self.sin))
        kw = dict(
            lse=(self.lse if lse is None else lse) if self.return_lse else None,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.block_table if block_table is None else block_table,
            kv_lens=self.kv_lens_t if kv_lens is None else kv_lens,
            slot_mapping=self.slot_mapping if slot_mapping is None else slot_mapping,
            w_iq_norm=self.w["w_iq_norm"],
            w_ik_norm=self.w["w_ik_norm"],
            index_kv_cache=self.index_kv_cache,
            index_state_cache=self.index_state_cache,
            index_state_block_table=self.index_state_block_table,
            index_state_slot_mapping=self.ring_slots if self.slot_form else None,
            cos_pos=self.cos_pos,
            sin_pos=self.sin_pos,
            block_ids_out=self.block_ids_out,
            block_lens_out=self.block_lens_out,
            current_stream=stream,
        )
        kw.update(over)
        self.blk.execute(
            self.h if h is None else h,
            self.w["w_qkvg"],
            self.w["w_q_norm"],
            self.w["w_k_norm"],
            *tables,
            self.w["w_o"],
            out,
            self.ws if ws is None else ws,
            **kw,
        )
        return out

    def run(self):
        if self.ws is None:
            self.compile()
        self.out.fill_(_sentinel(self.dtype))
        if self.lse is not None:
            self.lse.fill_(_sentinel(self.dtype))
        if self.block_ids_out is not None:
            self.block_ids_out.fill_(_SENT_I32)
        if self.block_lens_out is not None:
            self.block_lens_out.fill_(_SENT_I32)
        self.execute()
        torch.cuda.synchronize()
        self.oracle_from_list()
        return self.out

    def views(self):
        return self.blk._indexer_decode.views(self.ws, self.blk._ws)

    def kernel_ids(self) -> torch.Tensor:
        """The step's list ``[B, S_q, top_k]`` int32: the caller's buffer, or the workspace slot."""
        ids = self.block_ids_out if self.block_ids_out is not None else self.views()["ids"]
        return ids.view(len(self.kv_lens), self.s_q, self.top_k)

    def kernel_counts(self) -> torch.Tensor:
        return self.views()["counts"].view(len(self.kv_lens), self.s_q)

    def n_vis(self) -> torch.Tensor:
        """The visible complete-block count per row, ``clamp(floor((pos + 1) / 4), 0, n_blocks_max)`` ``[B, S_q]`` (host arithmetic)."""
        n_max = self.blk._indexer_decode.n_blocks_max
        return torch.tensor([[max(0, min((L - self.s_q + j + 1) // BS, n_max)) for j in range(self.s_q)] for L in self.kv_lens], device=_DEV)

    def oracle_from_list(self):
        """The two-stage oracle's second stage: the attention of every live row over the KERNEL's list per row (the row's own position, no
        step-0 anchor), then the gate and the out projection -- ``ref_o_gated`` / ``ref_lse`` / ``ref_out``."""
        ids = self.kernel_ids()
        h_q, D_ = self.geom.h_q, D
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            pos = torch.arange(self.s_q, device=_DEV) + (L - self.s_q)
            live = pos >= 0
            allowed = qsa_visible_mask(ids[b], None, pos, L, L, BS, top_k=self.top_k, pos0=None)[live]
            og, lse = _attend(self.refs[b], allowed, pos[live], self.scale, self.dtype, h_q, self.G)
            self.ref_o_gated[b, live] = og
            self.ref_lse[b, :, live] = lse
            self.ref_out[b, live] = (og.reshape(-1, h_q * D_).float() @ self.w["w_o"].float().t()).to(self.dtype)

    def check_selection(self, what: str = "selection") -> dict:
        """The two-stage oracle's FIRST stage on the kernel's own outputs: the counts rule (``min(top_k, n_vis)`` in ``block_lens_out``, ``n_vis``
        in the workspace counts), the list a strictly descending valid prefix then ``-1``, every id below the row's visible count, dead rows
        (``counts 0``, every slot ``-1``), the SET == the fp64 top-k of the kernel's OWN ``(ix_q, kbar)`` up to ties; the indexer queries
        BITWISE the pool-1 compress over the slab's band; the ring rows of the step BITWISE the slab's raw band; the block a row completes
        BITWISE the compress kernel over its four raw keys (the ring's and the band's); the null ring block and page 0 untouched."""
        B, s_q, top_k = len(self.kv_lens), self.s_q, self.top_k
        t = B * s_q
        ids, counts, n_vis = self.kernel_ids(), self.kernel_counts(), self.n_vis()
        assert torch.equal(counts.cpu(), n_vis.cpu().to(torch.int32)), f"{what}: the scorer's counts are not the visible counts"
        if self.block_lens_out is not None:
            assert torch.equal(
                self.block_lens_out.cpu(), torch.minimum(n_vis, torch.full_like(n_vis, top_k)).cpu().to(torch.int32)
            ), f"{what}: block_lens_out != min(top_k, n_vis)"
        valid = ids >= 0
        n_sel = torch.minimum(n_vis, torch.full_like(n_vis, top_k))
        assert torch.equal(
            valid, torch.arange(top_k, device=_DEV)[None, None, :] < n_sel.to(_DEV)[..., None]
        ), f"{what}: the valid ids are not the prefix min(top_k, n_vis)"
        assert not (ids == _SENT_I32).any(), f"{what}: a sentinel survived in the list"
        diffs = (ids[..., :-1] - ids[..., 1:]).masked_fill(~valid[..., 1:], 1)
        assert bool((diffs > 0).all()), f"{what}: a row's list is not strictly descending"
        assert bool((torch.where(valid, ids, torch.full_like(ids, -1)).amax(-1) < n_vis.to(_DEV)).all()), f"{what}: an id at or past the row's visible count"
        # the kernel's own operands: ix_q (every row) and the visible kbar rows (through the table)
        v = self.views()
        g = self.geom
        lay = self.blk._ws
        proj = _view(self.ws, lay.proj, (t, g.n_qkvg), self.dtype)
        h_i, d_i = g.qsa.index_heads, g.qsa.index_head_dim
        q_raw = torch.as_strided(proj, (t, h_i, d_i), (g.n_qkvg, d_i, 1), proj.storage_offset() + g.qkvg_offsets[ProjBlock.INDEX])
        cos_q = self.cos.view(t, 1, ROPE).expand(t, h_i, ROPE)
        sin_q = self.sin.view(t, 1, ROPE).expand(t, h_i, ROPE)
        qi_ref = self.compress(q_raw.reshape(t, h_i, d_i), cos_q, sin_q, pool=1)
        assert torch.equal(_bits(v["qi"]), _bits(qi_ref)), f"{what}: the indexer queries are not the pool-1 compress of the slab's band"
        raw = index_k_raw_view(proj, g, B, s_q)[:, :, 0]  # [B, S_q, d_i]: the step's raw keys (the FROST GEMM's band)
        flips = rows = 0
        gap = 0.0
        n_completed = 0
        for b, L in enumerate(self.kv_lens):
            pos0 = L - s_q
            if L == 0:
                assert (
                    torch.isnan(self.index_state_cache[0].float()).all() and torch.isnan(self.index_kv_cache[0].float()).all()
                ), f"{what}: the null ring block / page 0 moved"
                assert bool((counts[b] == 0).all()) and bool((ids[b] == -1).all()), f"{what}: a dead sequence scored"
                continue
            for j in range(s_q):
                p = pos0 + j
                if p < 0:
                    assert int(counts[b, j]) == 0 and bool((ids[b, j] == -1).all()), f"{what}: a row at a negative position scored"
                    continue
                # the ring row of the step's live row: the slab's raw key, bitwise
                assert torch.equal(
                    _bits(self.index_state_cache[b + 1, p % self.cap, 0]), _bits(raw[b, j])
                ), f"{what}: ring row of position {p} is not the step's raw key"
                if (p + 1) % BS == 0:
                    blk = p // BS
                    k4 = torch.stack(
                        [(self.index_state_cache[b + 1, q % self.cap, 0] if q < pos0 else raw[b, q - pos0]) for q in range(BS * blk, BS * blk + BS)]
                    )
                    kb = self.compress(k4[None], self.cos_pos[:, BS * blk : BS * blk + 1], self.sin_pos[:, BS * blk : BS * blk + 1])[0, 0]
                    row = self.index_kv_cache[int(self.block_table[b, blk // self.rpp]), blk % self.rpp, 0]
                    assert torch.equal(
                        _bits(row), _bits(kb)
                    ), f"{what}: sequence {b} row {j}: the block {blk} the row completes is not the compress of its four raw keys"
                    n_completed += 1
            # the SET: the fp64 top-k of the kernel's own operands over the row's visible blocks
            n = int(n_vis[b].max())
            if n == 0:
                continue
            pages = self.block_table[b, : -(-n // self.rpp)].long()
            kbar = self.index_kv_cache[pages][:, :, 0].reshape(-1, d_i)[:n].double()  # [n, d_i] through the table
            qi = v["qi"].view(B, s_q, h_i, d_i)[b].double()  # [S_q, h_i, d_i]
            sc = torch.relu(torch.einsum("shd,jd->shj", qi, kbar)).sum(1) / math.sqrt(d_i)  # [S_q, n]
            limit = torch.arange(n, device=_DEV)[None, :] < n_vis[b].to(_DEV)[:, None]
            sc = sc.masked_fill(~limit, float("-inf"))
            live = torch.tensor([pos0 + j >= 0 for j in range(s_q)], device=_DEV)
            m = torch.zeros(s_q, n + 1, dtype=torch.bool, device=_DEV)
            m.scatter_(1, torch.where(ids[b] >= 0, ids[b].long(), torch.full_like(ids[b], n).long()), torch.ones_like(ids[b], dtype=torch.bool))
            f, r, gp = _set_vs_topk(m[:, :n][live], sc[live], top_k, _TIE_TOL)
            flips, rows, gap = flips + f, rows + r, max(gap, gp)
        print(
            f"\n{self.label()} {what}: {n_completed} block(s) completed by the step, {flips} tie flips vs the fp64 top-k of the kernel's operands ({rows} rows, gap {gap:.2e})"
        )
        return dict(flips=flips, rows=rows, gap=gap, completed=n_completed)

    def label(self):
        return (
            f"[indexer {self.geom.h_q}/{self.geom.h_kv} S_q={self.s_q} p{self.page} {self.layout} split={self.blk.decode_split_kv} "
            f"gate={'fused' if self.fuse_gate else 'launch'} lse={self.return_lse} fuse_prepare={self.blk.fuse_prepare} cap={self.cap} lens={self.kv_lens}]"
        )


def _two_launches_indexer(case: _IndexerCase, what: str):
    a = case.run().clone()
    ids_a, o_a = case.kernel_ids().clone(), case.slab_views()[4].clone()
    lse_a = None if case.lse is None else case.lse.clone()
    cache_a, ring_a = _bits(case.index_kv_cache).clone(), _bits(case.index_state_cache).clone()
    b = case.run().clone()
    assert (
        torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(o_a), _bits(case.slab_views()[4])) and torch.equal(ids_a, case.kernel_ids())
    ), f"{what}: two launches are not bitwise"
    assert torch.equal(cache_a, _bits(case.index_kv_cache)) and torch.equal(
        ring_a, _bits(case.index_state_cache)
    ), f"{what}: the side caches differ between two launches"
    if lse_a is not None:
        assert torch.equal(lse_a, case.lse), f"{what}: two launches are not bitwise on LSE"
    m1 = case.check_selection(what)
    m2 = case.check(what=what)
    return dict(m1, **m2)


# (geometry, S_q, kv_lens, page, layout, split, fuse_gate, return_lse, fuse_prepare, cap, seed)
_INDEXER_LADDER = [
    pytest.param(GEOM_SMALL, 1, (2052, 8192, 0, 4097), 16, "hnd", None, False, True, None, None, 101, id="Sq1-B4-p16-hnd-default-split-lse-auto-knob"),
    pytest.param(GEOM_SMALL, 2, (300, 0, 2051, 77), 64, "nhd", 1, True, False, True, None, 102, id="Sq2-B4-p64-nhd-unsplit-fused-gate-fused-prepare"),
    pytest.param(GEOM_SMALL, 3, (8192, 2052), 32, "unified", 4, True, True, True, None, 103, id="Sq3-B2-p32-unified-split4-fused-gate-lse-fused-prepare"),
    pytest.param(
        GEOM_SMALL, 4, (8191, 8190, 8189, 8188), 16, "hnd", 16, True, True, False, None, 104, id="Sq4-B4-every-residue-split16-fused-gate-lse-unfused"
    ),
    pytest.param(GEOM_FLASH_NEXT, 4, (4096,), 16, "hnd", None, True, False, True, None, 105, id="Sq4-B1-flash-next-24-2-default-split-fused"),
    pytest.param(GEOM_SMALL, 4, (2, 4, 0, 20000), 16, "hnd", 2, False, True, None, None, 106, id="Sq4-B4-shorter-than-the-step-dead-20000-split2"),
    pytest.param(GEOM_SMALL, 1, (32768, 300), 16, "hnd", None, True, True, True, 4, 107, id="Sq1-B2-32K-cluster-top-k-fused-prepare"),
    pytest.param(GEOM_SMALL, 2, (8195, 8194), 16, "hnd", 1, False, True, False, 5, 108, id="Sq2-B2-cap5-mid-block-unfused"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, s_q, kv_lens, page, layout, split, fuse_gate, return_lse, fuse_prepare, cap, seed", _INDEXER_LADDER)
def test_decode_indexer_matches_the_two_stage_oracle(geom_kw, s_q, kv_lens, page, layout, split, fuse_gate, return_lse, fuse_prepare, cap, seed):
    """The decode indexer's ladder (module docstring): per cell the selection against the kernel's own operands (stage 1), the attention
    within the budget of the fp32 oracle fed the kernel's list per row (stage 2), dead rows exact, the pools' step rows bitwise the slab,
    the ring and the completed blocks bitwise the compress kernel, two launches bitwise -- fused and unfused prepare, both knob readings."""
    case = _IndexerCase(
        geom_kw, s_q, kv_lens, page=page, layout=layout, seed=seed, split=split, fuse_gate=fuse_gate, return_lse=return_lse, fuse_prepare=fuse_prepare, cap=cap
    )
    case.compile()
    if split is not None:
        assert case.blk.decode_split_kv == split
    from cudnn.gated_attention_block.api import SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS, _device_cc_of

    want = (
        fuse_prepare if fuse_prepare is not None else (_device_cc_of(_DEV) in SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS)
    )  # the AUTO knob: the measured part reads True
    assert case.blk.fuse_prepare is want and case.blk.fuse_prepare_requested is fuse_prepare
    m = _two_launches_indexer(case, "indexer ladder")
    print(f"\n{case.label()}: {m}")


@requires_rubin
def test_decode_indexer_identity_region_is_the_full_list_and_bitwise_the_caller_list_block():
    """512 visible blocks (``kv_lens`` 2048 at ``S_q`` 1 = ``top_k``): the list is the identity ``511 .. 0`` on every row, and the block is
    BITWISE the caller-list band block fed the full list (one core, one list, the same slab)."""
    case = _IndexerCase(GEOM_SMALL, 1, (2048, 2048, 2048, 2048), page=16, seed=111, split=8, fuse_gate=True, return_lse=True, fuse_prepare=True)
    case.compile()
    _two_launches_indexer(case, "identity")
    ids = case.kernel_ids()
    want = torch.arange(511, -1, -1, dtype=torch.int32, device=_DEV).expand(4, 1, 512)
    assert torch.equal(ids, want) and bool((case.kernel_counts() == 512).all())
    full = _Case(
        GEOM_SMALL,
        1,
        (2048, 2048, 2048, 2048),
        page=16,
        seed=111,
        split=8,
        fuse_gate=True,
        return_lse=True,
        index_band=True,
        index_list_form="per_row",
        shuffle=False,
    )
    full.k_cache.copy_(case.k_cache)
    full.v_cache.copy_(case.v_cache)
    full.block_table.copy_(case.block_table)
    full.slot_mapping.copy_(case.slot_mapping)
    full.block_ids.copy_(ids.view(4, 1, 512))
    full.block_lens.copy_(case.block_lens_out.view(4, 1))
    full.h.copy_(case.h)
    full.cos.copy_(case.cos)
    full.sin.copy_(case.sin)
    full.w["w_qkvg"].copy_(case.w["w_qkvg"])
    full.w["w_o"].copy_(case.w["w_o"])
    full.prefix_bits = (_bits(full.k_cache).clone(), _bits(full.v_cache).clone())
    full.compile()
    full.run()
    assert torch.equal(_bits(full.out), _bits(case.out)) and torch.equal(
        full.lse, case.lse
    ), "the identity region is not bitwise the caller-list block on the full list"


@requires_rubin
@pytest.mark.parametrize("s_q, kv_lens", [(1, (4096, 300, 0, 2051)), (4, (4095, 2, 0, 2052))], ids=["Sq1", "Sq4"])
def test_decode_indexer_fused_and_unfused_prepare_are_bitwise_with_the_census_by_kernel_name(s_q, kv_lens):
    """``fuse_prepare`` on vs off on the same inputs: ``out``, the LSE, the gated O, the pools, the ring, the paged cache, the list and the
    counts BITWISE; the CUPTI census by kernel NAME on the 24 / 2 geometry against the caller-list band block on the same shapes (the
    CONTROL: the slab's route and the out projection's reducer are the route's business) -- fused = the control's count (the prepare launch
    replaces the norm + RoPE pass, the K / V write and the raw-key write; the scorer ``cudnn_block_qsa_score_paged`` and the top-k
    ``cudnn_qsa_topk_decode`` come in): 8 where the control is 8 (the slab, ``frost_qsa_prepare``, the scorer, the top-k, the sparse tile,
    the combine, the out projection and its split-K reducer); unfused = the control + 3 (``frost_qk_norm_rope``, the K / V-only
    ``frost_cache_write``, the pool-1 ``frost_qsa_compress``, ``frost_qsa_compress_step``, the scorer and the top-k; no raw-key write): 11
    where the control is 8; no memcpy / memset on any of the three execute paths; no allocation, no host sync over three executes."""
    from collections import Counter

    on = _IndexerCase(GEOM_FLASH_NEXT, s_q, kv_lens, page=16, seed=120 + s_q, split=4, fuse_gate=True, return_lse=True, fuse_prepare=True)
    off = _IndexerCase(GEOM_FLASH_NEXT, s_q, kv_lens, page=16, seed=120 + s_q, split=4, fuse_gate=True, return_lse=True, fuse_prepare=False)
    ctl = _Case(GEOM_FLASH_NEXT, s_q, kv_lens, page=16, seed=120 + s_q, split=4, fuse_gate=True, return_lse=True, index_band=True, index_list_form="per_row")
    ctl.compile()
    ctl.run()
    for case in (on, off):
        case.compile()
        _two_launches_indexer(case, f"fuse_prepare={case.blk.fuse_prepare}")
    assert torch.equal(_bits(on.out), _bits(off.out)) and torch.equal(on.lse, off.lse), "fuse_prepare changed out / LSE"
    assert torch.equal(_bits(on.slab_views()[4]), _bits(off.slab_views()[4])), "fuse_prepare changed the gated O"
    assert torch.equal(_bits(on.k_cache.contiguous()), _bits(off.k_cache.contiguous())) and torch.equal(
        _bits(on.v_cache.contiguous()), _bits(off.v_cache.contiguous())
    )
    assert torch.equal(_bits(on.index_state_cache), _bits(off.index_state_cache)) and torch.equal(_bits(on.index_kv_cache), _bits(off.index_kv_cache))
    assert (
        torch.equal(on.kernel_ids(), off.kernel_ids())
        and torch.equal(on.block_lens_out, off.block_lens_out)
        and torch.equal(on.kernel_counts(), off.kernel_counts())
    )
    assert torch.equal(_bits(on.views()["qi"]), _bits(off.views()["qi"]))
    k_on, m_on = _cupti_kernels(lambda: on.execute())
    k_off, m_off = _cupti_kernels(lambda: off.execute())
    k_ctl, m_ctl = _cupti_kernels(lambda: ctl.execute())
    if k_on is None or k_off is None or k_ctl is None:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the bitwise cells stand")
    print(f"\nfused: {len(k_on)} kernels + {len(m_on)} memset/memcpy:\n  " + "\n  ".join(n[:140] for n in k_on))
    print(f"unfused: {len(k_off)} kernels + {len(m_off)} memset/memcpy:\n  " + "\n  ".join(n[:140] for n in k_off))
    print(f"control (caller-list band, unfused): {len(k_ctl)} kernels + {len(m_ctl)} memset/memcpy:\n  " + "\n  ".join(n[:140] for n in k_ctl))
    assert not (m_on or m_off or m_ctl), "a memset / memcpy on the execute path"
    has = lambda names, tok: sum(tok in n for n in names)  # noqa: E731
    for names in (k_on, k_off):
        assert has(names, "qsa_score_paged") == 1 and has(names, "qsa_topk_decode") == 1, names
    for names in (k_on, k_off, k_ctl):
        assert not any("nvjet" in n.lower() or "cublas" in n.lower() for n in names), "the slab left the FROST route"
    assert (
        has(k_on, "frost_qsa_prepare") == 1
        and has(k_on, "frost_qk_norm_rope") == 0
        and has(k_on, "frost_cache_write") == 0
        and has(k_on, "frost_qsa_compress") == 0
    )
    assert has(k_off, "frost_qsa_prepare") == 0 and has(k_off, "frost_qk_norm_rope") == 1 and has(k_off, "frost_cache_write") == 1
    assert has(k_off, "frost_qsa_compress_step") == 1 and has(k_off, "frost_qsa_compress") == 2  # the pool-1 query norm + the step form
    assert has(k_ctl, "frost_qk_norm_rope") == 1 and has(k_ctl, "frost_cache_write") == 2 and has(k_ctl, "qsa_score_paged") == 0  # K / V + the raw key
    assert len(k_off) - len(k_on) == 3, (len(k_off), len(k_on))
    assert len(k_on) == len(k_ctl) and len(k_off) == len(k_ctl) + 3, (len(k_on), len(k_off), len(k_ctl))
    if len(k_ctl) == 8:  # the measured chain at these rows: slab, norm + RoPE, the K / V write, the raw-key write, tile, combine, out projection, its reducer
        assert (len(k_on), len(k_off)) == (8, 11), (len(k_on), len(k_off))
    extra, gone = Counter(k_off) - Counter(k_on), Counter(k_on) - Counter(k_off)
    assert sum(extra.values()) == 4 and sum(gone.values()) == 1, (extra, gone)
    # no allocation of the block's own on the execute path, no host sync (the R9 census; the fused and the unfused form alike)
    for case in (on, off):
        torch.cuda.synchronize()
        stat = lambda: torch.cuda.memory_stats()["allocation.all.allocated"]  # noqa: E731
        n0 = stat()
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                case.execute()
        finally:
            torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert stat() == n0, f"fuse_prepare={case.blk.fuse_prepare}: the decode indexer block made {stat() - n0} CUDA allocation(s) over three executes"


@requires_rubin
def test_decode_indexer_list_fed_to_a_caller_list_block_is_bitwise():
    """The control arm of every measurement: the caller-list band block (``index_list_form="per_row"``) fed the indexer block's own
    ``block_ids_out`` / ``block_lens_out`` on the same slab, pools and table reproduces ``out`` / the gated O / the LSE BITWISE."""
    case = _IndexerCase(GEOM_SMALL, 2, (8192, 300, 0, 4097), page=16, seed=131, split=4, fuse_gate=True, return_lse=True, fuse_prepare=True)
    case.compile()
    _two_launches_indexer(case, "indexer arm")
    ctl = _Case(GEOM_SMALL, 2, (8192, 300, 0, 4097), page=16, seed=131, split=4, fuse_gate=True, return_lse=True, index_band=True, index_list_form="per_row")
    for name in ("k_cache", "v_cache", "block_table", "slot_mapping", "h", "cos", "sin"):
        getattr(ctl, name).copy_(getattr(case, name))
    ctl.w["w_qkvg"].copy_(case.w["w_qkvg"])
    ctl.w["w_o"].copy_(case.w["w_o"])
    ctl.block_ids.copy_(case.kernel_ids())
    ctl.block_lens.copy_(case.block_lens_out)
    ctl.prefix_bits = (_bits(ctl.k_cache).clone(), _bits(ctl.v_cache).clone())
    ctl.compile()
    ctl.run()
    assert torch.equal(_bits(ctl.out), _bits(case.out)) and torch.equal(ctl.lse, case.lse), "the caller-list block fed the indexer's list is not bitwise"
    assert torch.equal(_bits(ctl.slab_views()[4]), _bits(case.slab_views()[4]))


@requires_rubin
@pytest.mark.parametrize("fuse_prepare", [True, False], ids=["fused", "unfused"])
def test_decode_indexer_first_execute_captures_without_warm_up_and_replays_bitwise(fuse_prepare):
    """The serving stack's defaults on the decode indexer: the FIRST execute of a fresh declaration captured on a NEW stream with no eager
    warm-up under ``set_sync_debug_mode("error")`` (no plan constant, no allocation, no host read on the path), the capture launches
    nothing, the replay is BITWISE a fresh eager run (``out``, the LSE, the list, the ring, the cache), and a new step through the captured
    tensors (new lengths / tables / slots / ring state / cache / inputs) replays bitwise the eager run over them."""
    # the 24 / 2 band geometry: its slab (N 13952) rides the small-M route's 128-row tile at B x S_q <= 128 rows, capture-safe from the
    # first execute (the band capture cell above); the 8 / 2 band slab (N 5760) takes the GEMM graph route, whose handle is created eagerly
    case = _IndexerCase(GEOM_FLASH_NEXT, 2, (300, 0, 1000, 77), page=16, seed=141, split=4, fuse_gate=True, return_lse=True, fuse_prepare=fuse_prepare)
    case.prime_prefix()
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.full((case.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    case.lse.fill_(float("nan"))
    k_before, ring_before = _written(case.k_cache), _bits(case.index_state_cache).clone()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert (
            torch.isnan(case.out).all() and _written(case.k_cache) == k_before and torch.equal(_bits(case.index_state_cache), ring_before)
        ), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        case.oracle_from_list()
        replayed, replayed_lse, replayed_ids = case.out.clone(), case.lse.clone(), case.kernel_ids().clone()
        cache_r, ring_r = _bits(case.index_kv_cache).clone(), _bits(case.index_state_cache).clone()
        case.check_selection("replay")
        case.check(replayed, lse=replayed_lse, what="replay")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed)) and torch.equal(case.lse, replayed_lse) and torch.equal(case.kernel_ids(), replayed_ids)
        assert torch.equal(_bits(case.index_kv_cache), cache_r) and torch.equal(_bits(case.index_state_cache), ring_r), "the replay differs from the eager run"
        # a NEW step through the captured tensors: one length a token longer (its slot, its list, its ring position move), a re-laid-out
        # table, a new ring state and prefix cache, new inputs -- the oracle of the new step through a sibling case over the same shapes
        sib = _IndexerCase(
            GEOM_FLASH_NEXT, 2, (301, 0, 1000, 77), page=16, seed=143, split=4, fuse_gate=True, return_lse=True, fuse_prepare=fuse_prepare, n_pages=case.n_pages
        )
        sib.prime_prefix()
        assert (
            sib.block_table.shape == case.block_table.shape
            and sib.k_cache.shape == case.k_cache.shape
            and sib.index_kv_cache.shape == case.index_kv_cache.shape
        )
        for name in (
            "k_cache",
            "v_cache",
            "block_table",
            "kv_lens_t",
            "slot_mapping",
            "h",
            "cos",
            "sin",
            "index_kv_cache",
            "index_state_cache",
            "index_state_block_table",
        ):
            getattr(case, name).copy_(getattr(sib, name))
        for name in ("w_qkvg", "w_o", "w_iq_norm", "w_ik_norm"):
            case.w[name].copy_(sib.w[name])
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.refs, case.k_raw_full, case.kv_lens, case.live_rows, case.pages = sib.refs, sib.k_raw_full, sib.kv_lens, sib.live_rows, sib.pages
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        case.oracle_from_list()
        replayed2, replayed2_lse, replayed2_ids = case.out.clone(), case.lse.clone(), case.kernel_ids().clone()
        cache_r2, ring_r2 = _bits(case.index_kv_cache).clone(), _bits(case.index_state_cache).clone()
        case.check_selection("replay over the new step")
        case.check(replayed2, lse=replayed2_lse, what="replay over the new step")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed2)) and torch.equal(case.lse, replayed2_lse) and torch.equal(case.kernel_ids(), replayed2_ids)
        assert torch.equal(_bits(case.index_kv_cache), cache_r2) and torch.equal(
            _bits(case.index_state_cache), ring_r2
        ), "the replay over the new step differs from the eager run"
    finally:
        graph.reset()


@requires_rubin
def test_decode_indexer_rope_table_ceiling_by_form():
    """The RoPE-table ceiling (the compress kernel's rule, enforced by FORM at execute): a table of EXACTLY ``max_pages x P`` rows is
    accepted (the cells above run at that bound), one row short is a ``ValueError`` naming the bound BEFORE any stage (nothing of the step
    written); the stage's -2 report for a live row past the table is the kernel suites' pin (a conforming table never meets it)."""
    case = _IndexerCase(GEOM_SMALL, 1, (2052, 300), page=16, seed=151, split=1, fuse_prepare=True)
    case.compile()
    assert int(case.cos_pos.shape[1]) == case.max_pages * case.page
    _two_launches_indexer(case, "exact ceiling")
    before = (_bits(case.index_state_cache).clone(), _bits(case.index_kv_cache).clone(), _bits(case.k_cache).clone())
    with pytest.raises(ValueError, match="the table is the ceiling on positions"):
        case.execute(cos_pos=case.cos_pos[:, :-1].contiguous(), sin_pos=case.sin_pos[:, :-1].contiguous())
    torch.cuda.synchronize()
    assert (
        torch.equal(before[0], _bits(case.index_state_cache))
        and torch.equal(before[1], _bits(case.index_kv_cache))
        and torch.equal(before[2], _bits(case.k_cache))
    )


@requires_rubin
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64], ids=["slots_i32", "slots_i64"])
@pytest.mark.parametrize("fuse_prepare", [True, False], ids=["fused", "unfused"])
def test_decode_indexer_ring_slot_form_is_bitwise_the_derived_form(slot_dtype, fuse_prepare):
    """``index_state_slot_mapping_present=True``: the ring WRITE rows named by the stack's own slots (``block x cap + pos % cap``, ``-1`` on a
    dead row; ``slot_mapping``'s dtype, int32 and int64) against the rows the kernels derive from the ring table -- the ring, the cache, the
    list and ``out`` BITWISE; a slot tensor of the other dtype refused by name."""
    kw = dict(page=16, seed=161, split=2, fuse_gate=True, return_lse=True, fuse_prepare=fuse_prepare)
    derived = _IndexerCase(GEOM_SMALL, 4, (8191, 0, 2052, 77), **kw)
    explicit = _IndexerCase(GEOM_SMALL, 4, (8191, 0, 2052, 77), slot_form=True, **kw)
    for case in (derived, explicit):
        case.slot_mapping = case.slot_mapping.to(slot_dtype)
        case.ring_slots = case.ring_slots.to(slot_dtype)
        case.compile()
        _two_launches_indexer(case, f"ring slots {'explicit' if case.slot_form else 'derived'} {slot_dtype}")
    assert torch.equal(_bits(derived.out), _bits(explicit.out)) and torch.equal(derived.lse, explicit.lse)
    assert torch.equal(_bits(derived.index_state_cache), _bits(explicit.index_state_cache)) and torch.equal(
        _bits(derived.index_kv_cache), _bits(explicit.index_kv_cache)
    )
    assert torch.equal(derived.kernel_ids(), explicit.kernel_ids())
    other = torch.int64 if slot_dtype == torch.int32 else torch.int32
    with pytest.raises(ValueError, match="share the slot mapping's dtype"):
        explicit.execute(index_state_slot_mapping=explicit.ring_slots.to(other))


@requires_rubin
def test_decode_indexer_rope_position_tables_only_is_bitwise_the_per_token_form():
    """``rope_position_tables_only=True`` (the fused launch rotates Q / K and the indexer queries from ``cos_pos`` / ``sin_pos`` at the
    device-derived position; ``cos`` / ``sin`` refused): ``out``, the LSE, the pools, the ring, the cache, the list BITWISE the per-token
    form; the unfused twin under the flag is the typed decline."""
    kw = dict(page=16, seed=171, split=4, fuse_gate=True, return_lse=True, fuse_prepare=True)
    tok = _IndexerCase(GEOM_SMALL, 4, (8191, 2, 0, 300), **kw)
    tab = _IndexerCase(GEOM_SMALL, 4, (8191, 2, 0, 300), rope_tables_only=True, **kw)
    assert tab.blk.rope_position_tables_only and tab.blk._prepare._index_arms().rope_table
    for case in (tok, tab):
        case.compile()
        _two_launches_indexer(case, f"tables_only={case.rope_tables_only}")
    assert torch.equal(_bits(tok.out), _bits(tab.out)) and torch.equal(tok.lse, tab.lse) and torch.equal(tok.kernel_ids(), tab.kernel_ids())
    assert torch.equal(_bits(tok.index_state_cache), _bits(tab.index_state_cache)) and torch.equal(_bits(tok.index_kv_cache), _bits(tab.index_kv_cache))
    assert torch.equal(_bits(tok.k_cache), _bits(tab.k_cache)) and torch.equal(_bits(tok.views()["qi"]), _bits(tab.views()["qi"]))
    with pytest.raises(ValueError, match="cos / sin are refused under rope_position_tables_only=True"):
        tab.execute(cos=tok.cos, sin=tok.sin)


@requires_rubin
@pytest.mark.parametrize("r", [0, 1, 2, 3], ids=lambda r: f"pos0-mod4-{r}")
def test_decode_indexer_mtp_rows_select_for_themselves(r):
    """Per-row MTP semantics at ``S_q`` 4 on a short sequence at every residue of the step-0 position: every row's list is its own (the
    ``4k + 3`` row sees the block completed INSIDE the step only through its own selection -- that block is in its visible count and in
    its list iff its score ranks, never in an earlier row's list), the per-row oracle within budget on every row."""
    L = 56 + r + 4
    case = _IndexerCase(GEOM_SMALL, 4, (L, L + 4), page=16, seed=180 + r, split=1, fuse_gate=True, return_lse=True, fuse_prepare=(r % 2 == 0))
    case.compile()
    _two_launches_indexer(case, f"mtp r={r}")
    ids, n_vis = case.kernel_ids(), case.n_vis()
    for b, Lb in enumerate(case.kv_lens):
        pos0 = Lb - 4
        for j in range(4):
            p = pos0 + j
            if (p + 1) % 4 == 0:
                k = p // 4
                assert int(n_vis[b, j]) == k + 1 and bool(
                    (ids[b, j] == k).any()
                ), f"row {j} at 4k + 3 does not see / select its own block {k} (every block fits: the identity region)"
                for jj in range(j):
                    assert not bool((ids[b, jj] == k).any()), f"row {jj} lists block {k}, completed after its position"


@requires_rubin
def test_decode_indexer_workspace_size_is_honest():
    """``get_workspace_size()`` holds the five indexer slots and the split partials; a sentinel past it survives an execute; the slots sit
    where the carve says (the logits' rows 16-byte aligned)."""
    case = _IndexerCase(GEOM_SMALL, 1, (4096, 512, 0), page=16, seed=191, split=8, fuse_gate=True, fuse_prepare=False)
    case.compile()
    blk = case.blk
    lay = blk._layout()
    assert lay.ix_logits >= 0 and lay.ix_counts >= 0 and lay.ix_ids >= 0 and lay.ix_cslot >= 0 and lay.ix_q >= 0
    assert (lay.ix_logits % 16, lay.ix_ids % 16) == (0, 0) and blk._indexer_decode.n_blocks_max % 4 == 0
    need = blk.get_workspace_size()
    assert need >= lay.total_bytes
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    ws[:need].fill_(0x7F)
    out = torch.empty_like(case.out)
    case.ws = ws
    case.execute(out=out)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.oracle_from_list()
    case.check_selection("sentinel-tailed workspace")
    case.check(out, ws, what="sentinel-tailed workspace")


@requires_rubin
@pytest.mark.parametrize("s_q, page", [(1, 16), (4, 64)], ids=["Sq1-p16", "Sq4-p64"])
def test_caller_list_band_block_with_fuse_prepare_is_bitwise_the_two_stage_form(s_q, page):
    """The prepare fusion over the pools on the CALLER-LIST band form (the raw-key pool arm; the decline lifted 2026-10-21 with this cell):
    ``out``, the LSE, the gated O, the pools and the raw-key pool BITWISE the two-stage form; the stage list one launch shorter per band."""
    kw = dict(page=page, seed=200 + s_q, split=2, fuse_gate=True, return_lse=True, index_band=True)
    off = _Case(GEOM_SMALL, s_q, (2052, 300, 0, 4097), **kw)
    on = _Case(GEOM_SMALL, s_q, (2052, 300, 0, 4097), fuse_prepare=True, **kw)
    assert isinstance(on.blk._prepare, _QsaPrepare) and not on.blk._prepare.decode_arms and on.blk._cache_write is None and off.blk._prepare is None
    for case in (off, on):
        case.compile()
        _two_launches(case, f"caller-list band fuse_prepare={case.blk.fuse_prepare}")
    assert torch.equal(_bits(on.out), _bits(off.out)) and torch.equal(on.lse, off.lse) and torch.equal(_bits(on.slab_views()[4]), _bits(off.slab_views()[4]))
    assert torch.equal(_bits(on.k_cache.contiguous()), _bits(off.k_cache.contiguous())) and torch.equal(
        _bits(on.v_cache.contiguous()), _bits(off.v_cache.contiguous())
    )
    assert torch.equal(_bits(on.index_cache), _bits(off.index_cache)), "the raw-key pool differs"
    names_on, names_off = [st.name for st in on.blk._stages], [st.name for st in off.blk._stages]
    assert "qsa_prepare" in names_on and "cache_write" not in names_on and "qk_norm_rope" not in names_on and len(names_on) == len(names_off) - 1
