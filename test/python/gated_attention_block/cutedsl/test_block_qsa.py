# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``QsaSpec`` -- the block-sparse attention declaration of the gated attention block -- and the fifth (indexer) band.

Host cells (CPU arithmetic, the declaration-time declines on any GPU, the index-tensor form checks under
``torch.cuda.set_sync_debug_mode("error")``, the block's declines read off the sparse adapter's capabilities record and the
adapter's own accept / decline set equal to that record, the CuTe DSL floor declining through the block BEFORE any kernel
import, the in-block indexer's declaration -- its stage in the list, its workspace carve, the execute contract that flips with
``index_source`` -- and the scorer's own declines) plus ONE Rubin cell: the unfused projection serving the 13952-column slab at
the 24-query-head / 2-KV-head geometry, its indexer band checked against the fp64 matmul and the raw indexer key read back as a
zero-copy view.  The sparse SDPA stage end to end -- the block against the QSA oracle, the accept matrix, the degenerate rows,
the in-block indexer against the two-stage oracle -- is ``test_block_qsa_end_to_end.py``.
"""

import contextlib
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.frost.tile_dsl.requirements import tma_gather4_requirement_error
from cudnn.gated_attention_block import (
    GatedAttentionBlockBwd,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    ProjBlock,
    QsaSpec,
    QuantSpec,
    SavedForBackward,
    build_fused_qkvg_weight,
    index_k_raw_view,
    qkvg_from_hf,
    saved_slab_views,
)
from cudnn.gated_attention_block.api import (
    QSA_BLOCK_SIZE,
    QSA_TOP_K_ALIGN,
    QSA_TOP_K_MAX,
    _align_up,
    _CacheWrite,
    _FusedQkvProjection,
    _itemsize,
    _plan_workspace,
    _SparseSdpa,
    _view,
)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

_DEV = "cuda" if torch.cuda.is_available() else "cpu"

# The sibling block modules skip as a WHOLE below FROST's CuTe DSL floor (``cutedsl_requirement_error``).  This module's
# declaration cells need no DSL at all, so they keep running on a release DSL below the floor; only the arms that ask the
# block's stages, the sparse adapter or the in-block indexer's declaration to answer (they import their CuTe DSL kernels, or
# read the scorer's package) skip there -- ``_needs_dsl_floor`` on the test, or a ``pytest.skip`` ahead of the one such arm.
_CUTEDSL_REQUIREMENT = cutedsl_requirement_error("the gated attention block's stages")  # None at or above the floor
_needs_dsl_floor = pytest.mark.skipif(_CUTEDSL_REQUIREMENT is not None, reason=str(_CUTEDSL_REQUIREMENT))
# The adapter's TARGET half of the same gate: a DSL at or above the floor may still lack the Rubin target (the public 4.7 wheels do), and the
# sparse adapter then declines a modeled cc 10.7 request by name.  The host ACCEPTANCE test models that part and asserts acceptance, so it
# skips exactly where the adapter would decline -- through the adapter's own requirement (the floor, then the ``sm_107a`` target); the
# dedicated decline test below keeps running on any DSL at the floor (it substitutes the target state itself).
_SM107_TARGET_REQUIREMENT = tma_gather4_requirement_error((10, 7))  # None when the installed DSL renders the Rubin target
_needs_sm107_target = pytest.mark.skipif(_SM107_TARGET_REQUIREMENT is not None, reason=str(_SM107_TARGET_REQUIREMENT))

# The 24-query-head / 2-KV-head d256 geometry of the block-sparse model at TP 1 (d_model 2560, rope 64), with the
# indexer band: (4 + 1) x 128 = 640 columns below V -> N = 13312 + 640 = 13952.
_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64)
GEOM_BAND = GatedAttentionBlockGeometry(**_FLASH_NEXT, qsa=QsaSpec(index_band=True))
GEOM_DENSE = GatedAttentionBlockGeometry(**_FLASH_NEXT)
# A shrunk d256 geometry for the declaration cells (the sparse core is the d256 body; the other numbers are free).
_SMALL_D256 = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
# A tiny geometry (d_head 64) for CPU layout / loader arithmetic -- the band keeps its 64-column alignment at index_head_dim 64.
_TINY = dict(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16)
_TINY_QSA = QsaSpec(index_band=True, index_heads=2, index_head_dim=64)  # band = (2 + 1) x 64 = 192 columns


@contextlib.contextmanager
def _no_host_sync():
    """Every declaration-time and form check must stay on the host without a device sync (Rule 3) -- enforced, not read."""
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


# ---------------------------------------------------------------------------
# QsaSpec and the geometry: the fifth band's arithmetic, by name
# ---------------------------------------------------------------------------


def test_qsa_spec_defaults_and_derived_numbers():
    q = QsaSpec()
    assert (q.block_size, q.top_k, q.index_source, q.index_band) == (4, 512, "caller", False)
    assert (q.index_heads, q.index_kv_heads, q.index_head_dim, q.index_norm_eps) == (4, 1, 128, 1e-6)
    assert q.index_cache_dtype is None and not q.index_cache_e4m3  # appended: the indexer cache's dtype, None = the activation dtype
    assert (QSA_BLOCK_SIZE, QSA_TOP_K_MAX, QSA_TOP_K_ALIGN) == (4, 512, 4)
    assert q.index_band_cols == 640
    assert q.identity_bound == 2051  # floor(n / 4) <= 512 <=> n <= 2051
    assert QsaSpec(top_k=256).identity_bound == 1027 and QsaSpec(top_k=4).identity_bound == 19
    q.validate()
    QsaSpec(top_k=4).validate()
    QsaSpec(index_source="indexer", index_band=True).validate()
    QsaSpec(index_source="indexer", index_band=True, index_cache_dtype=torch.bfloat16).validate()
    e4 = QsaSpec(index_source="indexer", index_band=True, index_cache_dtype=torch.float8_e4m3fn)
    e4.validate()
    assert e4.index_cache_e4m3
    assert hash(GEOM_BAND) != hash(GEOM_DENSE)  # frozen dataclasses: a compile key tells the two apart
    # the fp8 indexer cache is a different plan from the bf16 indexer (numerics-changing: an attribute, never a knob)
    assert hash(GatedAttentionBlockGeometry(**_SMALL_D256, qsa=e4)) != hash(
        GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_source="indexer", index_band=True))
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(block_size=8), "block_size"),
        (dict(top_k=0), "top_k"),
        (dict(top_k=3), "top_k"),
        (dict(top_k=516), "top_k"),
        (dict(top_k=1024), "top_k"),
        (dict(index_source="tokens"), "index_source"),
        (dict(index_source="indexer", index_band=False), "index_band=True"),
        (dict(index_heads=0), "index_heads"),
        (dict(index_kv_heads=2), "index_kv_heads"),
        (dict(index_head_dim=100), "index_head_dim"),
        (dict(index_norm_eps=0.0), "index_norm_eps"),
        # the fp8 indexer cache: e5m2 named, another dtype refused, the attribute under caller lists refused by name
        (dict(index_source="indexer", index_band=True, index_cache_dtype=torch.float8_e5m2), "e5m2"),
        (dict(index_source="indexer", index_band=True, index_cache_dtype=torch.float16), "index_cache_dtype must be"),
        (dict(index_source="indexer", index_band=True, index_cache_dtype=torch.float32), "index_cache_dtype must be"),
        (dict(index_cache_dtype=torch.float8_e4m3fn), "requires index_source='indexer'"),
        (dict(index_band=True, index_cache_dtype=torch.float8_e4m3fn), "requires index_source='indexer'"),
        (dict(index_cache_dtype=torch.bfloat16), "requires index_source='indexer'"),  # ANY value under caller lists, the bf16 spelling too
        (dict(index_band=True, index_cache_dtype=torch.bfloat16), "whatever the dtype"),
    ],
)
def test_qsa_spec_validate_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        QsaSpec(**kwargs).validate()
    # ... and through the geometry, which runs QsaSpec's own rows.
    with pytest.raises(ValueError, match=match):
        GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(**kwargs)).validate()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(rope_dim=256), "rope_dim"),  # wider than the indexer head (128)
        (dict(is_causal=False), "is_causal=True"),
        (dict(window_left=128), "sliding window"),
        (dict(window_right=0), "sliding window"),
    ],
)
def test_geometry_cross_field_rejects_under_qsa(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GatedAttentionBlockGeometry(**{**_SMALL_D256, **kwargs}, qsa=QsaSpec()).validate()
    GatedAttentionBlockGeometry(**{**_SMALL_D256, **kwargs}).validate() if "window_right" not in kwargs else None  # dense: legal


def test_geometry_qsa_must_be_a_qsa_spec():
    with pytest.raises(TypeError, match="QsaSpec"):
        GatedAttentionBlockGeometry(**_SMALL_D256, qsa={"top_k": 512}).validate()


def test_flash_next_five_band_layout():
    """Every number of the five-band contract at the 24 / 2 geometry, by name."""
    g = GEOM_BAND
    g.validate()
    assert g.index_band and g.qsa.index_band_cols == 640
    assert g.qkvg_blocks == (ProjBlock.Q, ProjBlock.GATE, ProjBlock.K, ProjBlock.V, ProjBlock.INDEX)
    assert g.n_qkvg == 13952 == 109 * 128 == 218 * 64
    assert g.qkvg_block_widths == (6144, 6144, 512, 512, 640)
    assert g.qkvg_offsets == (0, 6144, 12288, 12800, 13312)
    assert g.qkvg_heads == (24, 24, 2, 2, 5)
    assert g.qkvg_head_dims == (256, 256, 256, 256, 128)
    assert g.index_k_raw_offset == 13312 + 4 * 128 == 13824
    assert g.n_qkv == 6144 + 512 + 512 == GEOM_DENSE.n_qkv  # Q + K + V: neither the gate nor the band
    assert g.gqa_ratio == 12 and g.scale == pytest.approx(1 / 16)
    # The band's columns decode to the indexer's heads at the indexer's head dim.
    assert g.block_for_column(13312) == (ProjBlock.INDEX, 0, 0)
    assert g.block_for_column(13312 + 4 * 128 - 1) == (ProjBlock.INDEX, 3, 127)  # the last query-head column
    assert g.block_for_column(13824) == (ProjBlock.INDEX, 4, 0)  # the raw key head
    assert g.block_for_column(13951) == (ProjBlock.INDEX, 4, 127)
    assert g.block_for_column(13311) == (ProjBlock.V, 1, 255)
    with pytest.raises(ValueError):
        g.block_for_column(13952)
    # Tile plans: 64 / 128 well defined with the band as the last contiguous run; 256 is the typed refusal.
    for tile_n, band_tiles in ((64, 10), (128, 5)):
        plan = g.qkvg_tile_plan(tile_n)
        assert len(plan) == 13952 // tile_n and plan.count(ProjBlock.INDEX) == band_tiles
        runs = [b for i, b in enumerate(plan) if i == 0 or plan[i - 1] != b]
        assert runs == list(g.qkvg_blocks)
    with pytest.raises(ValueError, match="must be positive and divide N=13952"):
        g.qkvg_tile_plan(256)
    # TP 2 / TP 4 with the band: 6656 + 640 = 7296 (57 x 128), 3584 + 640 = 4224 (33 x 128).
    for h_q, h_kv, n in ((12, 1, 7296), (6, 1, 4224)):
        tp = GatedAttentionBlockGeometry(**{**_FLASH_NEXT, "h_q": h_q, "h_kv": h_kv}, qsa=QsaSpec(index_band=True))
        tp.validate()
        assert tp.n_qkvg == n and n % 128 == 0 and n % 256 == 128
        assert len(tp.qkvg_tile_plan(128)) == n // 128


def test_a_qsa_spec_without_the_band_leaves_the_four_band_layout_untouched():
    """``qsa`` set but ``index_band=False`` (caller lists over a plain four-band W_qkvg): the layout is the dense one."""
    g = GatedAttentionBlockGeometry(**_FLASH_NEXT, qsa=QsaSpec())
    g.validate()
    assert not g.index_band
    for attr in ("qkvg_blocks", "qkvg_block_widths", "qkvg_offsets", "qkvg_heads", "qkvg_head_dims", "n_qkvg", "n_qkv"):
        assert getattr(g, attr) == getattr(GEOM_DENSE, attr), attr
    assert g.qkvg_tile_plan(256) == GEOM_DENSE.qkvg_tile_plan(256)
    with pytest.raises(ValueError, match="declares no indexer band"):
        g.index_k_raw_offset


# ---------------------------------------------------------------------------
# The loader and the slab views (CPU)
# ---------------------------------------------------------------------------


def test_build_fused_qkvg_weight_appends_the_indexer_band():
    torch.manual_seed(0)
    g = GatedAttentionBlockGeometry(**_TINY, qsa=_TINY_QSA)
    g.validate()
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model
    w_qg, w_k, w_v = torch.randn(2 * hq * d, dm), torch.randn(hkv * d, dm), torch.randn(hkv * d, dm)
    w_i = torch.randn(g.qsa.index_band_cols, dm)
    fused = build_fused_qkvg_weight(w_qg, w_k, w_v, g, q_gate_layout="per_head", index_qk_proj_weight=w_i)
    assert fused.shape == (g.n_qkvg, dm) == (hq * d * 2 + hkv * d * 2 + 192, dm)
    h = torch.randn(2, 8, dm)
    proj = h @ fused.t()
    o_i = g.qkvg_offsets[ProjBlock.INDEX]
    torch.testing.assert_close(proj[..., o_i:], h @ w_i.t())  # the band is the indexer weight, rows as stored
    torch.testing.assert_close(proj[..., :o_i], h @ build_fused_qkvg_weight(w_qg, w_k, w_v, GatedAttentionBlockGeometry(**_TINY), q_gate_layout="per_head").t())
    # The raw key is the band's LAST index_kv_heads x index_head_dim columns; the query heads precede it.
    torch.testing.assert_close(proj[..., g.index_k_raw_offset :], h @ w_i[g.qsa.index_heads * g.qsa.index_head_dim :].t())
    with pytest.raises(ValueError, match="index_qk_proj_weight is required"):
        build_fused_qkvg_weight(w_qg, w_k, w_v, g, q_gate_layout="per_head")
    with pytest.raises(ValueError, match=r"must be \[192, 256\]"):
        build_fused_qkvg_weight(w_qg, w_k, w_v, g, q_gate_layout="per_head", index_qk_proj_weight=torch.randn(640, dm))
    with pytest.raises(ValueError, match="dtype"):
        build_fused_qkvg_weight(w_qg, w_k, w_v, g, q_gate_layout="per_head", index_qk_proj_weight=w_i.to(torch.bfloat16))
    with pytest.raises(ValueError, match="declares no indexer band"):
        build_fused_qkvg_weight(w_qg, w_k, w_v, GatedAttentionBlockGeometry(**_TINY), q_gate_layout="per_head", index_qk_proj_weight=w_i)


def test_qkvg_from_hf_carries_the_indexer_band():
    torch.manual_seed(0)
    g = GatedAttentionBlockGeometry(**_TINY, qsa=_TINY_QSA)
    d, dm = g.d_head, g.d_model
    w_qg, w_k, w_v = torch.randn(2 * g.h_q * d, dm), torch.randn(g.h_kv * d, dm), torch.randn(g.h_kv * d, dm)
    w_i, w_n = torch.randn(g.qsa.index_band_cols, dm), torch.randn(d) * 0.01
    w_qkvg, wq, wk = qkvg_from_hf(w_qg, w_k, w_v, w_n, w_n, g, index_qk_proj_weight=w_i, act_dtype=torch.float32)
    assert w_qkvg.shape == (g.n_qkvg, dm)
    assert torch.equal(w_qkvg, build_fused_qkvg_weight(w_qg, w_k, w_v, g, q_gate_layout="per_head", index_qk_proj_weight=w_i))
    assert torch.equal(w_qkvg[g.qkvg_offsets[ProjBlock.INDEX] :], w_i)
    assert torch.equal(wq, 1.0 + w_n)  # the norm form is untouched by the band
    with pytest.raises(ValueError, match="index_qk_proj_weight is required"):
        qkvg_from_hf(w_qg, w_k, w_v, w_n, w_n, g)
    # The band's bf16 rounding is the loader's single rounding, like every other band.
    w16, _, _ = qkvg_from_hf(w_qg, w_k, w_v, w_n, w_n, g, index_qk_proj_weight=w_i)
    assert torch.equal(w16[g.qkvg_offsets[ProjBlock.INDEX] :], w_i.to(torch.bfloat16))


def test_saved_slab_views_skips_the_band_and_index_k_raw_view_reads_it():
    g = GatedAttentionBlockGeometry(**_TINY, qsa=_TINY_QSA)
    b, s = 2, 3
    t = b * s
    slab = torch.arange(t * g.n_qkvg, dtype=torch.float32).view(t, g.n_qkvg)
    views = saved_slab_views(slab, g, b, s)
    assert len(views) == 4  # the training record's four bands; never the indexer band
    for view, blk in zip(views, (ProjBlock.Q, ProjBlock.GATE, ProjBlock.K, ProjBlock.V)):
        off, heads = g.qkvg_offsets[blk], g.qkvg_heads[blk]
        assert view.shape == (b, s, heads, g.d_head) and view.data_ptr() == slab.data_ptr() + off * 4
        assert torch.equal(view.reshape(t, heads * g.d_head), slab[:, off : off + heads * g.d_head])
    k_raw = index_k_raw_view(slab, g, b, s)
    off, w = g.index_k_raw_offset, g.qsa.index_kv_heads * g.qsa.index_head_dim
    assert k_raw.shape == (b, s, 1, 64) and k_raw.data_ptr() == slab.data_ptr() + off * 4  # zero-copy: the slab's storage
    assert torch.equal(k_raw.reshape(t, w), slab[:, off : off + w])
    assert torch.equal(index_k_raw_view(slab.view(b, s, g.n_qkvg), g, b, s), k_raw)  # the [B, S, N] spelling
    # The slab contract and the band are both required.
    with pytest.raises(ValueError, match="declares no indexer band"):
        index_k_raw_view(slab[:, : GatedAttentionBlockGeometry(**_TINY).n_qkvg].contiguous(), GatedAttentionBlockGeometry(**_TINY), b, s)
    with pytest.raises(ValueError, match="elements"):
        index_k_raw_view(slab[:-1].contiguous(), g, b, s)
    with pytest.raises(ValueError, match="contiguous"):
        index_k_raw_view(slab.t(), g, b, s)


def test_workspace_carve_is_the_dense_one_plus_the_band_columns():
    """The carve of a geometry WITHOUT the band is unchanged (literals); with it the slab grows by exactly the band."""
    b, s = 2, 128
    t, e = b * s, _itemsize(torch.bfloat16)
    dense = GatedAttentionBlockGeometry(**_SMALL_D256)
    band = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_band=True))
    lay_d = _plan_workspace(dense, b, s, torch.bfloat16, False, False, True)
    lay_b = _plan_workspace(band, b, s, torch.bfloat16, False, False, True)
    assert (lay_d.proj, lay_d.o, lay_d.engine_scratch, lay_d.total_bytes) == (0, 2_621_440, 3_670_016, 3_670_016)  # 256 x 5120 x 2 ; + 256 x 2048 x 2
    assert lay_d.q == lay_d.k == lay_d.v == -1
    assert (lay_b.proj, lay_b.o, lay_b.engine_scratch, lay_b.total_bytes) == (0, 2_949_120, 3_997_696, 3_997_696)  # 256 x 5760 x 2
    assert lay_b.o - lay_d.o == _align_up(t * 640 * e) == t * band.qsa.index_band_cols * e


def test_workspace_carve_of_the_decode_indexer_is_the_caller_list_carve_plus_five_slots_at_the_end():
    """The in-block indexer's DECODE step over the pools (2026-10-21) carves FIVE slots after everything the caller-list band block's
    decode carve holds -- `ix_q [T, index_heads, index_head_dim]`, `ix_logits [T, n_blocks_max]` fp32, `ix_counts [T]` int32, `ix_ids
    [T, top_k]` int32, `ix_cslot [B, S_q]` int32 -- in that order, each 16-byte aligned, and nothing else moves: every other field of the
    two carves is equal, the prefill indexer's slots stay unused (-1), and the total grows by exactly the five aligned sizes.  The planner's
    `qsa_indexer_decode` is the logits width (the table's reach in compressed rows); the DSL-free ring rule equals the kernel's."""
    import dataclasses

    from cudnn.gated_attention_block.api import _index_state_ring_extra_rows, _qsa_indexer_decode_slots

    b, s, n_blocks_max, e = 2, 4, 32 * 4, _itemsize(torch.bfloat16)
    t = b * s
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, causal_bottom_right=True, qsa=QsaSpec(index_source="indexer", index_band=True))
    lay_c = _plan_workspace(geom, b, s, torch.bfloat16, True, False, True)
    lay_i = _plan_workspace(geom, b, s, torch.bfloat16, True, False, True, qsa_indexer_decode=n_blocks_max)
    five = ["ix_q", "ix_logits", "ix_counts", "ix_ids", "ix_cslot"]
    slots = _qsa_indexer_decode_slots(geom, b, s, e, n_blocks_max)
    assert [nm for nm, _ in slots] == five
    q = geom.qsa
    assert [nbytes for _, nbytes in slots] == [t * q.index_heads * q.index_head_dim * e, t * n_blocks_max * 4, t * 4, t * q.top_k * 4, b * s * 4]
    for f in dataclasses.fields(lay_c):
        if f.name in five or f.name in ("engine_scratch", "total_bytes"):
            continue  # the five slots sit before the engines' scratch, which the planner keeps at the END of the carve
        assert getattr(lay_c, f.name) == getattr(lay_i, f.name), f.name
    assert all(getattr(lay_c, nm) == -1 for nm in five) and lay_i.ix_kbar == lay_i.ix_scores == lay_i.ix_cand == lay_i.ix_ids_sorted == -1
    acc = lay_c.total_bytes
    for nm, nbytes in slots:
        assert getattr(lay_i, nm) == acc and acc % 16 == 0, nm
        acc += _align_up(nbytes)
    assert lay_i.total_bytes == lay_i.engine_scratch == acc and lay_c.engine_scratch == lay_c.total_bytes
    assert _index_state_ring_extra_rows(q.block_size) == q.block_size - 1 == 3  # == kernels.qsa_compress.min_ring_rows(0, 4), checked at plan time


def test_decode_indexer_declines_are_published_row_by_row_and_the_envelope_names_the_indexer_facts(monkeypatch):
    """The decode indexer's decline rows (`_qsa_decode_decline_texts`) reach the envelope VERBATIM under a sparse geometry on every reading,
    each carrying its one shared token, and the FlashInfer-read indexer facts are published by name on a Rubin reading: `index_sources`
    (`"caller"`, `"indexer"` at bf16 x head dim 128; `"caller"` alone at f16), the cache dtype, the head dims / KV heads / block sizes, the
    `top_k` range, the 64-row query envelope, the ring's extra rows, the page sizes the paged scorer serves, `fuse_prepare_default` True on the measured part (cc 10.7,
    2026-10-21) for the bf16 indexer geometry and False off it; the cache dtypes `(bf16, e4m3)` since the e4m3 indexer cache is served at decode (2026-10-21; the
    `indexer_e4m3_cache` decline row retired) -- the DSL state MODELED as the served one (the facts are a reading of the SERVED form, never of this
    host's wheel); a DENSE geometry publishes none of them (any device, no host sync)."""
    import dataclasses

    from cudnn.frost import buffers as _buffers
    from cudnn.gated_attention_block import DecodeEnvelope
    from cudnn.gated_attention_block.api import SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS, _qsa_decode_decline_texts

    # the DSL state MODELED as the served one (the public 4.8.0 wheel with the sm_107a target, the version-gate cell's convention): the facts
    # published for the measured part are a reading of the SERVED form, never of this host's wheel -- under a DSL at the floor but without
    # the Rubin target (the public 4.7 wheels) `serving_envelope(..., device_cc=(10, 7))` reports decode unserved and the knob's AUTO reading
    # False, the rule this cell is not about
    monkeypatch.setattr(_buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(_buffers, "_cutedsl_has_sm107", lambda: True)
    tokens = {
        "shared_list_rows": "tail covers at most two blocks",
        "indexer_f16": "needs bf16 activations",
        "indexer_head_dim": "packs 128-wide heads",
        "indexer_thd": "scores dense [B, S] prompts",
        "indexer_query_rows": "query rows per sequence (at most 64",
        "indexer_page_size": "a power-of-two count of compressed rows",
        "indexer_e4m3_pool": "in-kernel dequant arm",
        "indexer_shared_list": "selects per row",
        "fuse_prepare_rope_tables_unfused": "the unfused query norm reads the per-token",
        # the EXTEND step's rows (2026-10-29)
        "indexer_extend_fuse_prepare": "the T-row chunk's ring read and its tail write live in two launches",
        "indexer_extend_ring_slots": "the extend step takes the stack's ring slots",
        "indexer_extend_rope_tables_only": "the extend step is always unfused and its query norm reads the per-token",
        "indexer_extend_ring_rows": "at least block_size - 1 = 3 ring rows",
        "indexer_extend_chunk_rows": "a chunk longer than the table's reach",
    }
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, causal_bottom_right=True, qsa=QsaSpec(index_source="indexer", index_band=True))
    for dtype in (torch.bfloat16, torch.float16):
        texts = _qsa_decode_decline_texts(geom, dtype)
        assert set(texts) == set(tokens) and all(tok in texts[nm] for nm, tok in tokens.items())
        d = GatedAttentionBlockFwd.serving_envelope(geom, dtype, _DEV, device_cc=(10, 7)).decode
        assert all(d.declines[nm] == texts[nm] for nm in texts), "a decline row reaches the envelope reworded"
        assert d.index_sources == (("caller", "indexer") if dtype == torch.bfloat16 else ("caller",))
        assert d.index_cache_dtypes == (
            (torch.bfloat16, torch.float8_e4m3fn) if dtype == torch.bfloat16 else ()
        )  # the e4m3 indexer cache served at decode (2026-10-21)
        assert "indexer_e4m3_cache" not in d.declines  # the row retired: a served form has no decline row
        assert d.index_head_dims == (128,) and d.index_kv_heads == (1,) and d.index_block_sizes == (4,) and d.index_top_k_range == (4, 512, 4)
        assert d.index_query_rows_max == 64 and d.index_state_rows_extra == 3 and d.index_page_sizes == (16, 32, 64, 128)
        assert d.fuse_prepare_default is (dtype == torch.bfloat16) and SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS == (
            (10, 7),
        )  # the measured part; f16: not served
        assert d.index_list_forms == ("per_row",) and d.index_list_form_default == "per_row"
    dense = GatedAttentionBlockFwd.serving_envelope(
        GatedAttentionBlockGeometry(**_SMALL_D256, causal_bottom_right=True), torch.bfloat16, _DEV, device_cc=(10, 7)
    ).decode
    assert dense.index_sources == () and dense.index_page_sizes == () and dense.index_query_rows_max == 0 and dense.fuse_prepare_default is False
    assert GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=(10, 0)).decode.fuse_prepare_default is False  # an unmeasured part
    assert not (set(tokens) & set(dense.declines))
    # the append-only tail: the EXTEND step's four fields after route_cc (2026-10-29); `quant_modes` appended after them 2026-10-30 (the quant
    # spellings the DENSE paged forms serve on the part -- the FP8-projection + bf16-attention mode; `()` on every QsaSpec geometry)
    assert [f.name for f in dataclasses.fields(DecodeEnvelope)][-19:] == [
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
        "sparse_tiles",
        "sparse_tile_lean_default",
        "device_cc",
        "route_cc",
        "index_select_forms",
        "index_extend_ring_rows_min",
        "index_extend_select_budget_bytes",
        "index_compress_roundings",
        "quant_modes",
    ]
    assert dense.quant_modes == (("QuantSpec(attention_dtype=torch.bfloat16)",) if dense.served else ()), "a served dense form on cc 10.7 names the mode"


# ---------------------------------------------------------------------------
# Declaration-time declines (any GPU; no host sync)
# ---------------------------------------------------------------------------


def _samples(geom, *, batch=1, seq_len=8, dtype=torch.bfloat16, act=None, device=_DEV):
    act = act or dtype
    z = lambda *shape, dt=dtype: torch.zeros(*shape, dtype=dt, device=device)  # noqa: E731
    return dict(
        sample_h=z(batch, seq_len, geom.d_model),
        sample_w_qkvg=z(geom.n_qkvg, geom.d_model),
        sample_w_q_norm=z(geom.d_head, dt=act),
        sample_w_k_norm=z(geom.d_head, dt=act),
        sample_cos=z(batch, seq_len, geom.rope_dim, dt=act),
        sample_sin=z(batch, seq_len, geom.rope_dim, dt=act),
        sample_w_o=z(geom.d_model, geom.h_q * geom.d_head),
        sample_out=z(batch, seq_len, geom.d_model, dt=act),
    )


_E4M3 = torch.float8_e4m3fn
_UNIT_QUANT = QuantSpec(descale_h=1.0, descale_w_qkvg=1.0, descale_w_o=1.0, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0)

# (geometry overrides, QsaSpec, block kwargs, sample dtype, exception, match) -- one row per typed decline of the sparse path.
# The row ``thd`` (NotImplementedError "thd=True") INVERTED on 2026-10-08 into test_thd_under_qsa_declares_the_packed_sparse_stage
# when the block's sparse stage bound packed operands; the two THD compositions the core does not carry keep their rows below.
_DECLINES = [
    pytest.param({}, QsaSpec(), dict(save_for_backward=True), torch.bfloat16, NotImplementedError, "save_for_backward=True", id="training"),
    pytest.param({}, QsaSpec(), dict(quant=_UNIT_QUANT), _E4M3, NotImplementedError, "quantized pipelines", id="quant"),
    # fuse_gate under QsaSpec is SERVED (the sparse core's epilogue gate; its accept cells are the end-to-end module's)
    pytest.param({}, QsaSpec(index_band=True), dict(fuse_norm_rope=True), torch.bfloat16, NotImplementedError, "256-column tiles", id="fused_proj_x_band"),
    pytest.param(dict(causal_bottom_right=True), QsaSpec(), {}, torch.bfloat16, NotImplementedError, "causal_bottom_right=True", id="bottom_right"),
    pytest.param({}, QsaSpec(), {}, torch.float32, NotImplementedError, "bf16 / f16", id="dtype"),
    pytest.param(dict(h_q=48), QsaSpec(), {}, torch.bfloat16, NotImplementedError, r"h_q // h_kv <= 16", id="gqa_group"),
    # the in-block indexer: the scorer's dtype (bf16), its head dim and its head groups -- each named in the message
    pytest.param({}, QsaSpec(index_source="indexer", index_band=True), {}, torch.float16, NotImplementedError, "bf16 activations", id="indexer_f16"),
    pytest.param(
        {},
        QsaSpec(index_source="indexer", index_band=True, index_head_dim=256),
        {},
        torch.bfloat16,
        NotImplementedError,
        "128-wide heads",
        id="indexer_head_dim",
    ),
    pytest.param(
        {},
        QsaSpec(index_source="indexer", index_band=True, index_heads=3),
        {},
        torch.bfloat16,
        NotImplementedError,
        "head groups",
        id="indexer_heads",
        marks=_needs_dsl_floor,  # the head-group read sits behind the floor gate: below it the version decline answers first
    ),
    pytest.param(dict(d_head=128), QsaSpec(), {}, torch.bfloat16, NotImplementedError, "d_head == 256", id="d_head"),
    # the fp8 indexer cache on f16 activations: the indexer's own dtype row speaks (the scorer packs bf16 queries and keys); on the decode form
    # over the pools the e4m3 cache itself is served since 2026-10-21 (test_block_qsa_decode.py) -- the f16 row is what stays
    pytest.param(
        {},
        QsaSpec(index_source="indexer", index_band=True, index_cache_dtype=_E4M3),
        {},
        torch.float16,
        NotImplementedError,
        "bf16 activations",
        id="indexer_e4m3_cache_on_f16_activations",
    ),
    # THD composed with what the core does not carry under packed sequences: the in-block indexer (the paged write-through composes
    # with the packed sparse block since 2026-10-14: test_write_through_under_thd_builds_the_same_stage_beside_the_packed_sparse_stage)
    pytest.param(
        {},
        QsaSpec(index_source="indexer", index_band=True),
        dict(thd=True, num_sequences=1, max_seq_len=8),
        torch.bfloat16,
        NotImplementedError,
        "packed-sequence form",
        id="indexer_x_thd",
    ),
    pytest.param(dict(is_causal=False), QsaSpec(), {}, torch.bfloat16, ValueError, "is_causal=True", id="bidirectional"),
    # a negative attn_scale: the block's geometry refuses it by name for every pipeline (the standalone sparse adapter SERVES it signed)
    pytest.param(dict(attn_scale=-0.0625), QsaSpec(), {}, torch.bfloat16, ValueError, "attn_scale must be > 0", id="negative_attn_scale"),
    pytest.param(dict(window_left=64), QsaSpec(), {}, torch.bfloat16, ValueError, "sliding window", id="window"),
    pytest.param({}, QsaSpec(top_k=516), {}, torch.bfloat16, ValueError, "top_k", id="top_k_516"),
    pytest.param({}, QsaSpec(top_k=0), {}, torch.bfloat16, ValueError, "top_k", id="top_k_0"),
    pytest.param({}, QsaSpec(top_k=3), {}, torch.bfloat16, ValueError, "top_k", id="top_k_3"),
]


@pytest.mark.parametrize("geom_kw, qsa, blk_kw, dtype, exc, match", _DECLINES)
def test_declaration_declines_every_sparse_request_it_cannot_serve(geom_kw, qsa, blk_kw, dtype, exc, match):
    geom = GatedAttentionBlockGeometry(**{**_SMALL_D256, **geom_kw}, qsa=qsa)
    kw = _samples(geom, dtype=dtype, act=torch.bfloat16 if dtype == _E4M3 else dtype)
    with _no_host_sync():
        with pytest.raises(exc, match=match):
            GatedAttentionBlockFwd(**kw, geometry=geom, **blk_kw)
    # The SAME request without the sparse declaration is not refused by any of these rows (the dense block's own
    # declines, if any, read differently), so each row is the sparse path's.
    if geom_kw.get("is_causal", True) and "window_left" not in geom_kw:
        dense = GatedAttentionBlockGeometry(**{**_SMALL_D256, **geom_kw})
        kw_d = _samples(dense, dtype=dtype, act=torch.bfloat16 if dtype == _E4M3 else dtype)
        try:
            GatedAttentionBlockFwd(**kw_d, geometry=dense, **blk_kw)
        except (NotImplementedError, ValueError) as e:  # a dense decline of its own is fine; the sparse wording is not
            assert "QsaSpec" not in str(e)


def test_thd_under_qsa_declares_the_packed_sparse_stage():
    """INVERTED 2026-10-08 (the ``thd`` row of ``_DECLINES`` until the block's sparse stage bound packed operands): a ``QsaSpec``
    block declared ``thd=True`` CONSTRUCTS (no host sync) with its sparse stage in the packed form -- internally ``batch 1,
    seq_len T``, the SAME ``[1, T, H, D]`` operand descriptors the dense arm declares (the slab column slice at the token
    stride), ONE lengths descriptor of exactly ``num_sequences`` (``num_sequences + 1`` under ``cu_seqlens``) int32 entries
    bound to BOTH length sides, the flat ``[T, top_k]`` list and the head-major ``[1, H_q, T]`` LSE -- at both packed input
    ranks; a dense declaration carries neither length descriptor; the block's own packed contract still holds
    (``seq_lens_present`` exclusive, the knobs refused without ``thd``); the stage's own guards hold without the block; the
    one composition the core does not carry under packed sequences stays a typed decline naming the feature (the row of
    ``_DECLINES``: the in-block indexer; the paged write-through composes -- the next test)."""
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec())
    t, top_k, b = 16, geom.qsa.top_k, 3
    with _no_host_sync():
        for cu, rank2 in ((False, False), (True, True)):
            kw = _samples(geom, batch=1, seq_len=t)
            if rank2:
                for nm in ("sample_h", "sample_cos", "sample_sin", "sample_out"):
                    kw[nm] = kw[nm][0]
            blk = GatedAttentionBlockFwd(**kw, geometry=geom, thd=True, num_sequences=b, max_seq_len=8, cu_seqlens=cu, return_lse=True)
            st = blk._sdpa
            assert isinstance(st, _SparseSdpa) and st.thd and st.cu_seqlens == cu and (st.num_sequences, st.max_seq_len) == (b, 8)
            assert (blk.batch, blk.seq_len, blk.thd) == (1, t, True) and not st.seq_lens_present
            ops = st._operands()
            assert ops["q"].shape == (1, t, geom.h_q, geom.d_head) and ops["o"].shape == (1, t, geom.h_q, geom.d_head)
            assert ops["k"].shape == ops["v"].shape == (1, t, geom.h_kv, geom.d_head)
            assert ops["q"].stride() == (t * geom.n_qkvg, geom.n_qkvg, geom.d_head, 1)  # the slab column slice, in place
            n = b + int(cu)
            assert ops["seq_q_lens"].shape == ops["seq_kv_lens"].shape == (n,) and str(ops["seq_q_lens"].dtype) == "torch.int32"
            assert ops["block_ids"].shape == (t, top_k) and ops["block_lens"].shape == (t,) and ops["lse"].shape == (1, geom.h_q, t)
        dense = GatedAttentionBlockFwd(**_samples(geom, batch=1, seq_len=t), geometry=geom)
        ops_d = dense._sdpa._operands()
        assert not dense._sdpa.thd and ops_d["seq_q_lens"] is None and ops_d["seq_kv_lens"] is None
        # the block's own packed contract, unchanged under QsaSpec
        with pytest.raises(ValueError, match="mutually exclusive"):
            GatedAttentionBlockFwd(**_samples(geom, batch=1, seq_len=t), geometry=geom, thd=True, num_sequences=b, max_seq_len=8, seq_lens_present=True)
        with pytest.raises(ValueError, match="THD-only"):
            GatedAttentionBlockFwd(**_samples(geom, batch=1, seq_len=t), geometry=geom, num_sequences=b)
        # the stage's own guards (it stays honest without the block): the packed form is batch 1, the sequence count is a
        # declaration fact, the two length contracts are exclusive, the knobs are THD-only, and a launch without the
        # lengths or the metadata arm is refused BEFORE the adapter is touched
        st_kw = dict(dtype=torch.bfloat16, device=_DEV, want_lse=True)
        with pytest.raises(ValueError, match="batch 1"):
            _SparseSdpa(geom, batch=2, seq_len=8, thd=True, num_sequences=2, **st_kw)
        with pytest.raises(ValueError, match="num_sequences >= 1"):
            _SparseSdpa(geom, batch=1, seq_len=t, thd=True, **st_kw)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _SparseSdpa(geom, batch=1, seq_len=t, thd=True, num_sequences=b, seq_lens_present=True, **st_kw)
        with pytest.raises(ValueError, match="THD-only"):
            _SparseSdpa(geom, batch=1, seq_len=t, cu_seqlens=True, **st_kw)
        st = _SparseSdpa(geom, batch=1, seq_len=t, thd=True, num_sequences=b, **st_kw)
        st._impl, st._compiled = object(), True  # the guards run ahead of any adapter call
        z = lambda *shape, dt=torch.bfloat16: torch.zeros(*shape, dtype=dt, device=_DEV)  # noqa: E731
        q, kv, o = z(1, t, geom.h_q, geom.d_head), z(1, t, geom.h_kv, geom.d_head), z(1, t, geom.h_q, geom.d_head)
        ids, lse, lens = z(t, top_k, dt=torch.int32), z(1, geom.h_q, t, dt=torch.float32), z(b, dt=torch.int32)
        with pytest.raises(ValueError, match="requires seq_lens"):
            st.execute(q, kv, kv, o, lse=lse, workspace=z(64, dt=torch.uint8), block_ids=ids)
        with pytest.raises(ValueError, match="requires workspace"):
            st.execute(q, kv, kv, o, lse=lse, seq_lens=lens, block_ids=ids)


def test_write_through_under_thd_builds_the_same_stage_beside_the_packed_sparse_stage():
    """INVERTED 2026-10-14 (the ``write_through_x_thd`` row of ``_DECLINES`` until the write-through served packed sequences):
    a ``QsaSpec`` block declared ``thd=True`` with ``paged_kv_page_size`` CONSTRUCTS (no host sync) with the write-through stage
    at the dense block's pipeline position -- after norm + RoPE, right before the packed sparse stage -- sized at the packed
    token count (the block is ``B = 1, S = T`` inside, so the stage's ``[T, H_kv, D]`` views are the dense arm's), at both
    packed input ranks and with the indexer band (its raw-key pool rides the same stage); the dense ``QsaSpec`` block builds the
    same stage (the pools are the caller's: the workspace is unchanged -- the Rubin cell asserts it).  The in-block indexer stays the
    packed form's one typed decline, with or without the write-through.  The composition's end-to-end acceptance is the packed
    write-through cell of ``test_block_qsa_end_to_end.py`` (Rubin)."""
    t, b, ps = 16, 3, 16
    with _no_host_sync():
        for band, rank2 in ((False, False), (False, True), (True, False)):
            geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_band=band))
            kw = _samples(geom, batch=1, seq_len=t)
            if rank2:
                for nm in ("sample_h", "sample_cos", "sample_sin", "sample_out"):
                    kw[nm] = kw[nm][0]
            blk = GatedAttentionBlockFwd(**kw, geometry=geom, thd=True, num_sequences=b, max_seq_len=8, paged_kv_page_size=ps)
            plain = GatedAttentionBlockFwd(**kw, geometry=geom, thd=True, num_sequences=b, max_seq_len=8)
            cw = blk._cache_write
            assert blk.paged_kv_page_size == ps and isinstance(cw, _CacheWrite) and cw.page_size == ps and cw.index_band is band
            assert blk.thd and (blk.batch, blk.seq_len) == (1, t) and cw.batch * cw.seq_len == t
            stages = list(blk._stages)
            assert isinstance(blk._sdpa, _SparseSdpa) and blk._sdpa.thd
            assert stages.index(blk._norm_rope) < stages.index(cw) == stages.index(blk._sdpa) - 1
            assert plain._cache_write is None  # the workspace equality is the Rubin cell's (the sparse stage's size reads the arch gate)
        dense = GatedAttentionBlockFwd(**_samples(geom, batch=1, seq_len=t), geometry=geom, paged_kv_page_size=ps)
        assert isinstance(dense._cache_write, _CacheWrite) and dense._cache_write.batch * dense._cache_write.seq_len == t
        geom_i = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_source="indexer", index_band=True))
        with pytest.raises(NotImplementedError, match="packed-sequence form"):
            GatedAttentionBlockFwd(**_samples(geom_i, batch=1, seq_len=t), geometry=geom_i, thd=True, num_sequences=b, max_seq_len=8, paged_kv_page_size=ps)


@_needs_dsl_floor
def test_an_indexer_declaration_builds_the_indexer_stage_and_the_execute_contract_flips():
    """``QsaSpec(index_source="indexer")`` CONSTRUCTS (no host sync): the indexer stage sits right before the sparse SDPA in the
    stage list; below the identity bound it reserves nothing (the carve is the caller-list block's, the selection a plan-time
    identity) and past it the five slots (its queries, the compressed keys, the list, the scores, the scorer's scratch -- the
    scratch's closed form equal to the DSA sizing helper's count); ``execute`` then REQUIRES the two indexer norm weights,
    REFUSES ``block_ids`` / ``block_lens``, form-checks the three optional outputs, and a caller-list or a dense block refuses
    every one of the five; off Rubin ``check_support`` declines ahead of any launch -- below cc 10 from the indexer stage, naming
    the scorer's device family, on a cc 10.x part other than Rubin from the sparse adapter's record, naming the found cc (the
    indexer accepts the family) -- both texts exercised on every GPU host through a simulated capability.  Form only, no plan,
    no launch."""
    from cudnn.deepseek_sparse_attention.indexer_forward import compress_topk_cand_buffer_size_thd
    from cudnn.gated_attention_block.api import _Indexer, _qsa_cand_floats, _qsa_indexer_slots

    spec = QsaSpec(index_source="indexer", index_band=True)
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=spec)
    with _no_host_sync():
        for b, s, lens in ((2, 8, False), (1, spec.identity_bound, False), (1, spec.identity_bound + 1, False), (2, 2300, True)):
            blk = GatedAttentionBlockFwd(**_samples(geom, batch=b, seq_len=s), geometry=geom, return_lse=True, seq_lens_present=lens)
            ix = blk._indexer
            assert isinstance(ix, _Indexer) and blk._stages.index(ix) == blk._stages.index(blk._sdpa) - 1
            assert ix.selects == (s > spec.identity_bound) and ix.n_blocks == s // spec.block_size
            lay = blk._layout()
            caller = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_band=True))
            lay_c = GatedAttentionBlockFwd(**_samples(caller, batch=b, seq_len=s), geometry=caller, return_lse=True, seq_lens_present=lens)._layout()
            if not ix.selects:
                assert ix.cand_floats == 0 and lay == lay_c and lay.ix_q == -1
            else:
                e, t = _itemsize(torch.bfloat16), b * s
                assert ix.cand_floats == _qsa_cand_floats(b, s, spec.block_size) > 0
                # the slots in the stage's own order, each 256-B aligned, appended after the caller-list block's carve; the names and
                # byte counts are the stage's (the layout pins the arithmetic: order, alignment, the total)
                slots = _qsa_indexer_slots(geom, b, s, e, ix.cand_floats)
                assert [nm for nm, _ in slots] == [
                    "ix_q",
                    "ix_kbar",
                    "ix_ids",
                    "ix_scores",
                    "ix_cand",
                ]  # the sort scratch gone (2026-10-29: the canonical kernel runs in place)
                assert (
                    dict(slots)["ix_q"] == t * spec.index_heads * spec.index_head_dim * e and dict(slots)["ix_kbar"] == b * (s // 4) * spec.index_head_dim * e
                )
                assert (
                    dict(slots)["ix_ids"] == t * spec.top_k * 4 and dict(slots)["ix_cand"] == ix.cand_floats * 4 and lay.ix_ids_sorted == lay.ix_sort_idx == -1
                )
                acc = lay_c.total_bytes
                for nm, nbytes in slots:
                    assert getattr(lay, nm) == acc, nm
                    acc += _align_up(nbytes)
                assert lay.total_bytes == acc
    if torch.cuda.is_available():
        # the scratch's closed form against the DSA helper that sizes it on device (a drift would mis-size the workspace)
        for b, s in ((1, 2052), (2, 2300), (3, 4097)):
            cu_q = (torch.arange(b + 1, device="cuda") * s).to(torch.int32)
            cu_k = (torch.arange(b + 1, device="cuda") * (s // 4)).to(torch.int32)
            assert compress_topk_cand_buffer_size_thd(cu_q, cu_k, 4, None)[1] == _qsa_cand_floats(b, s, 4)
    # the execute contract (form only, before any plan): what the indexer block needs, refuses and form-checks
    b, s = 2, 8
    kw = _samples(geom, batch=b, seq_len=s)
    args = tuple(kw[k] for k in ("sample_h", "sample_w_qkvg", "sample_w_q_norm", "sample_w_k_norm", "sample_cos", "sample_sin", "sample_w_o", "sample_out"))
    ws = torch.zeros(16, dtype=torch.uint8, device=_DEV)
    w = torch.ones(spec.index_head_dim, dtype=torch.bfloat16, device=_DEV)
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
    kb = lambda nb: torch.zeros(b, nb, spec.index_head_dim, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    with _no_host_sync():
        blk = GatedAttentionBlockFwd(**kw, geometry=geom)
        for ex_kw, match in (
            (dict(block_ids=i32(b * s, 512), w_iq_norm=w, w_ik_norm=w), "derives the selection"),
            (dict(block_lens=i32(b * s), w_iq_norm=w, w_ik_norm=w), "derives the selection"),
            (dict(w_ik_norm=w), "needs w_iq_norm"),
            (dict(w_iq_norm=w), "needs w_ik_norm"),
            (dict(w_iq_norm=w.half(), w_ik_norm=w), r"w_iq_norm must be a contiguous \[128\] torch.bfloat16"),
            (dict(w_iq_norm=w, w_ik_norm=w[:64]), r"w_ik_norm must be a contiguous \[128\]"),
            (dict(w_iq_norm=w, w_ik_norm=w, index_k_compressed=kb(1)), r"index_k_compressed must be \[B=2, >= 2"),
            (dict(w_iq_norm=w, w_ik_norm=w, index_k_compressed=kb(2).half()), "index_k_compressed must be a contiguous torch.bfloat16"),
            (dict(w_iq_norm=w, w_ik_norm=w, block_ids_out=i32(b * s, 511)), r"block_ids_out must be \[16, 512\] or \[2, 8, 512\]"),
            (dict(w_iq_norm=w, w_ik_norm=w, block_ids_out=i32(b * s, 512).to(torch.int64)), "block_ids_out must be int32"),
            (dict(w_iq_norm=w, w_ik_norm=w, block_lens_out=i32(b * s + 1)), r"block_lens_out must be \[16\] or \[2, 8\]"),
        ):
            with pytest.raises(ValueError, match=match):
                blk.execute(*args, ws, **ex_kw)
        # well-formed (every optional output, the cache wider than the prompt's blocks): the next refusal is the plan's
        with pytest.raises(RuntimeError, match="call compile"):
            blk.execute(*args, ws, w_iq_norm=w, w_ik_norm=w, index_k_compressed=kb(5), block_ids_out=i32(b, s, 512), block_lens_out=i32(b * s))
        with pytest.raises(RuntimeError, match="call compile"):
            blk.execute(*args, ws, w_iq_norm=w, w_ik_norm=w)
        # the mirror images: a caller-list block and a dense block refuse the indexer's tensors rather than ignoring them
        caller = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_band=True))
        blk_c = GatedAttentionBlockFwd(**_samples(caller, batch=b, seq_len=s), geometry=caller)
        for ex_kw in (
            dict(w_iq_norm=w),
            dict(w_ik_norm=w),
            dict(index_k_compressed=kb(2)),
            dict(block_ids_out=i32(b * s, 512)),
            dict(block_lens_out=i32(b * s)),
        ):
            with pytest.raises(ValueError, match="index_source='caller'"):
                blk_c.execute(*args, ws, block_ids=i32(b * s, 512), **ex_kw)
        dense = GatedAttentionBlockGeometry(**_SMALL_D256)
        kw_d = _samples(dense, batch=b, seq_len=s)
        args_d = tuple(
            kw_d[k] for k in ("sample_h", "sample_w_qkvg", "sample_w_q_norm", "sample_w_k_norm", "sample_cos", "sample_sin", "sample_w_o", "sample_out")
        )
        blk_d = GatedAttentionBlockFwd(**kw_d, geometry=dense)
        for ex_kw in (dict(w_iq_norm=w), dict(block_ids_out=i32(b * s, 512)), dict(index_k_compressed=kb(2))):
            with pytest.raises(ValueError, match="declared without geometry.qsa"):
                blk_d.execute(*args_d, ws, **ex_kw)
    if torch.cuda.is_available():
        # the canonical order the stage runs after the top-k (2026-10-29): the FROST row canonicalizer IN PLACE on the list -- the selected ids
        # descending, the -1 padding last, equal to torch's descending sort, -1 past the row's count whatever the slots held -- one launch, no
        # torch op, no allocation (the former torch.sort windows and their int64 index scratch are gone); it runs on any CuTe DSL device
        from cudnn.gated_attention_block.kernels.qsa_canonical_list import compile_qsa_canonical_list, run_qsa_canonical_list

        rows, top_k = 4096 + 7, spec.top_k
        g = torch.Generator(device="cuda").manual_seed(3)
        raw = torch.randperm(4 * top_k, device="cuda", generator=g)[:top_k].to(torch.int32).repeat(rows, 1)
        raw[:, 300:] = -1
        counts = torch.full((rows,), 300, dtype=torch.int32, device="cuda")
        counts[7] = 100  # a count below the valid ids the top-k left: the canonical list is -1 past the COUNT
        want = torch.sort(raw, dim=-1, descending=True).values
        want[7, 100:] = -1
        canon = compile_qsa_canonical_list(with_lens=False)
        dst = raw.clone()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        run_qsa_canonical_list(canon, dst, counts, dst, None, top_k=top_k, stream=torch.cuda.current_stream().cuda_stream)  # in place
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        assert torch.equal(dst, want) and bool((dst[:, 300:] == -1).all()) and bool((dst[7, 100:] == -1).all())
        assert bool((dst[:, :99] > dst[:, 1:100]).all()) and bool((dst[:7, :299] > dst[:7, 1:300]).all())  # strictly descending inside every valid prefix
    if torch.cuda.is_available():
        # Off Rubin the block declines before any launch.  Since the cc 10.0 admission of the dense forms (2026-10-29) the FIRST word
        # on any other part is the block's SPARSE PART STATEMENT (`_check_part_support`, before any stage: the sparse forms -- the
        # core, the lean tile through the block, the indexer, the e4m3 cache -- are served on cc 10.7 only, the lift named), below
        # and beside Rubin alike.  The stages' own gates still speak when asked directly: below cc 10 the indexer stage names the
        # scorer's device family; on a cc 10.x part other than Rubin the sparse stage repeats the part statement AHEAD of the
        # adapter (whose record would admit the explicit lean tile there since #1999).  Every text on every GPU host through a
        # simulated capability (restored on exit), then the real device.
        part = r"sparse forms .* served on cc 10\.7 \(Rubin\) only"
        for sim_cc, stage, stage_pattern in (
            ((9, 0), blk._indexer, r"qsa_indexer: .*\(cc 10\.x\).*found cc 9\.0"),
            ((10, 0), blk._sdpa, part + r".*found SM100"),
        ):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(torch.cuda, "get_device_capability", lambda *a, _cc=sim_cc, **k: _cc)
                with pytest.raises(NotImplementedError, match=part + rf".*found SM{sim_cc[0]}{sim_cc[1]}"):
                    blk.check_support()
                with pytest.raises(NotImplementedError, match=stage_pattern):
                    stage.check_support()
        cc = tuple(torch.cuda.get_device_capability())
        if cc != (10, 7):
            with pytest.raises(NotImplementedError, match=part + rf".*found SM{cc[0]}{cc[1]}"):
                blk.check_support()


def _indexer_stage_probe(monkeypatch, *, seq_len: int, stub_select=None):
    """A declared (never compiled as a block) indexer block at ``_SMALL_D256``, ``B = 1``, with the stage's kernel recipes stubbed out -- the
    compress recipe compiles to a sentinel, the compress launch is a no-op, the scorer + top-k (``_select``) to ``stub_select`` when given --
    so the stage's HOST contract (its plan constants, its stream discipline) runs on any CUDA device without a kernel compile.  Returns the
    stage, its layout, a workspace of the layout's size and the stage's execute operands (zeros; nothing of them is read)."""
    import cudnn.gated_attention_block.kernels.qsa_compress as compress_mod
    from cudnn.gated_attention_block.api import _Indexer

    spec = QsaSpec(index_source="indexer", index_band=True)
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=spec)
    b, s = 1, seq_len
    blk = GatedAttentionBlockFwd(**_samples(geom, batch=b, seq_len=s), geometry=geom, return_lse=True)
    ix, lay = blk._indexer, blk._layout()
    assert isinstance(ix, _Indexer)
    monkeypatch.setattr(compress_mod, "compile_qsa_compress", lambda **kw: object())
    monkeypatch.setattr(compress_mod, "run_qsa_compress", lambda *a, **kw: None)
    if stub_select is not None:
        monkeypatch.setattr(_Indexer, "_select", stub_select)
    ws = torch.empty(lay.total_bytes, dtype=torch.uint8, device="cuda")
    ops = dict(
        proj=torch.zeros(b * s, geom.n_qkvg, dtype=torch.bfloat16, device="cuda"),
        cos=torch.zeros(b, s, geom.rope_dim, dtype=torch.bfloat16, device="cuda"),
        sin=torch.zeros(b, s, geom.rope_dim, dtype=torch.bfloat16, device="cuda"),
        w=torch.ones(spec.index_head_dim, dtype=torch.bfloat16, device="cuda"),
    )
    return ix, lay, ws, ops


def _identity_list(seq_len: int, batch: int, block_size: int, top_k: int) -> tuple:
    """The identity selection and its counts as the stage's contract spells them, derived on the host: ``counts[t] = min(top_k,
    floor((pos + 1) / block_size))``, ``ids[t, j] = counts[t] - 1 - j`` below the count (the block's ONE canonical order, DESCENDING, like every
    scorer-written list; 2026-10-29) and ``-1`` past it."""
    t = batch * seq_len
    pos = torch.arange(seq_len, device="cuda").repeat(batch)
    counts = torch.minimum((pos + 1) // block_size, torch.full_like(pos, top_k)).to(torch.int32)
    j = torch.arange(top_k, device="cuda", dtype=torch.int32)
    ids = torch.where(j[None, :] < counts[:, None], counts[:, None] - 1 - j[None, :], torch.full((t, top_k), -1, dtype=torch.int32, device="cuda"))
    return ids, counts


@_needs_dsl_floor
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stage's row sort and its DtoD copies run on a CUDA device")
def test_indexer_stage_row_sort_runs_on_the_launch_stream_through_as_torch_stream(monkeypatch):
    """Recipe R1 at the stage: past the identity bound ``_Indexer.execute`` puts every row of the top-k's list in canonical order on the LAUNCH
    stream -- since 2026-10-29 the FROST row canonicalizer on the raw launch stream, IN PLACE on ``ix_ids`` (no torch op at all: the former
    ``torch.sort`` windows went through ``as_torch_stream``, never a hand-rolled ``ExternalStream``, which torch before 2.13 made a fresh
    non-blocking pool stream unordered against the top-k and the sparse core).  The compress recipes are stubbed to sentinels and the scorer
    to a known row permutation, so the host contract runs on any CUDA device (the canonical kernel itself runs from sm_80): ``ExternalStream``
    monkeypatched to raise, the list stage (4) consumes IS the ``ix_ids`` slot, equal to torch's descending sort of the stub's list with the
    ``-1`` padding last, the counts copied out."""
    spec = QsaSpec(index_source="indexer", index_band=True)
    s, top_k = spec.identity_bound + 1, spec.top_k  # the first length past the identity: the scorer and the sort run
    _, counts = _identity_list(s, 1, spec.block_size, top_k)
    g = torch.Generator(device="cuda").manual_seed(11)
    perm = torch.argsort(torch.rand(s, top_k, device="cuda", generator=g), dim=-1).to(torch.int32)  # a permutation of [0, top_k) per row
    stub = torch.where(perm < counts[:, None], perm, torch.full_like(perm, -1))  # the row's visible ids once each, in random slots, -1 past the count

    def fake_select(self, qi, kbar, ids, scores, cand, *, stream):
        ids.copy_(stub)  # the "top-k", in the unspecified slot order the real kernel leaves
        return ids

    ix, lay, ws, ops = _indexer_stage_probe(monkeypatch, seq_len=s, stub_select=fake_select)
    assert ix.selects
    ix.compile()  # the real scratch-sizing helper, the stubbed recipes, the stub warm-up
    lens_out = torch.full((s,), -7, dtype=torch.int32, device="cuda")

    def boom(*a, **k):
        raise AssertionError(f"torch.cuda.ExternalStream{a} constructed on the execute path: a raw handle goes through as_torch_stream (R1)")

    monkeypatch.setattr(torch.cuda, "ExternalStream", boom)
    assert torch.cuda.current_stream().cuda_stream == 0, "the probe needs torch on its default stream (the handle 0)"
    got = ix.execute(ops["proj"], ops["cos"], ops["sin"], ops["w"], ops["w"], seq_lens=None, workspace=ws, lay=lay, block_lens_out=lens_out, current_stream=0)
    torch.cuda.synchronize()
    v = ix.views(ws, lay)
    assert got.data_ptr() == v["ids"].data_ptr() and lay.ix_ids_sorted == -1, "the list stage (4) consumes is the ix_ids slot, canonical in place"
    assert torch.equal(got, torch.sort(stub, dim=-1, descending=True).values), "the canonical order differs from torch's descending sort"
    assert torch.equal(got >= 0, torch.arange(top_k, device="cuda")[None, :] < counts[:, None]), "the valid ids are not the count's prefix (-1 last)"
    assert torch.equal(lens_out, counts), "the counts copied out differ from the contract's"


@_needs_dsl_floor
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stage's plan constants and their DtoD copies run on a CUDA device")
def test_indexer_stage_constants_land_on_the_launch_stream_with_the_ambient_stream_parked_past_compile(monkeypatch):
    """The plan constants on the launch stream, the IDENTITY branch (``S <= identity_bound``: the list and the counts are plan constants,
    the stage launches nothing of its own): torch's default (ambient) stream is parked by a long spin from BEFORE ``compile()`` until PAST
    the first execute on a side stream.  Constants FILLED at ``compile()`` on the ambient stream would sit behind the spin while the side
    stream's DtoD copies read them -- the previous tree hands ``block_ids_out`` the poisoned bytes the allocator reused for them; constants
    whose values the first execute uploads on ITS stream behind compile's event read as the identity.  The premise -- the ambient stream
    still busy when ``compile()`` returned and when the execute was issued -- is asserted, never assumed; the caching allocator's blocks of
    the constants' sizes are poisoned first, so an unlanded fill cannot hide behind a stale right value."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is the park; the matmul fallback is too short to outlast compile()")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gated_block_stream_probe import park_the_default_stream

    spec = QsaSpec(index_source="indexer", index_band=True)
    s, top_k = spec.identity_bound, spec.top_k
    ix, lay, ws, ops = _indexer_stage_probe(monkeypatch, seq_len=s)
    assert not ix.selects and lay.ix_q == -1
    identity, counts = _identity_list(s, 1, spec.block_size, top_k)
    ids_out = torch.full((s, top_k), -7, dtype=torch.int32, device="cuda")
    lens_out = torch.full((s,), -7, dtype=torch.int32, device="cuda")
    ambient, side = torch.cuda.default_stream(), torch.cuda.Stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    torch.cuda.synchronize()
    junk = [torch.full((s, top_k), 0x5A5A5A5A, dtype=torch.int32, device="cuda"), torch.full((s,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")]
    torch.cuda.synchronize()
    del junk  # the freed blocks of the constants' exact sizes hold the poison the next allocation of those sizes reuses
    park_the_default_stream(3.0)  # the AMBIENT stream is busy from here, through compile() and past the first execute
    ix.compile()
    parked_after_compile = not ambient.query()
    ix.execute(
        ops["proj"],
        ops["cos"],
        ops["sin"],
        ops["w"],
        ops["w"],
        seq_lens=None,
        workspace=ws,
        lay=lay,
        block_ids_out=ids_out,
        block_lens_out=lens_out,
        current_stream=side.cuda_stream,
    )
    parked_at_issue = not ambient.query()
    side.synchronize()  # the EXECUTION stream only
    assert parked_after_compile and parked_at_issue, "the premise: the park must outlast compile() and the first execute's issue"
    torch.cuda.synchronize()
    assert torch.equal(ids_out, identity), "the identity list the side stream copied out is not the plan's (a constant written on the parked ambient stream)"
    assert torch.equal(lens_out, counts), "the counts the side stream copied out are not the plan's"


def test_a_legal_sparse_declaration_builds_the_sparse_stage_and_the_arch_gate_reads_off_the_record():
    """The block CONSTRUCTS with the sparse stage in stage (4)'s slot (the five-band geometry accepted by every stage ahead
    of it), with no host sync; ``check_support`` then asks the sparse ADAPTER, whose record names the arch range -- so off
    Rubin the decline is the adapter's typed one naming the found cc (the block keeps no ``(10, 7)`` literal of its own for
    the sparse path; below cc 10 the gather4 part floor's decline answers first), and on Rubin the block is supported (the
    end-to-end module runs it)."""
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_band=True))
    kw = _samples(geom)
    with _no_host_sync():
        blk = GatedAttentionBlockFwd(**kw, geometry=geom)
        assert isinstance(blk._sdpa, _SparseSdpa) and blk.qsa is geom.qsa
        i = blk._stages.index(blk._sdpa)  # in pipeline order: the sparse stage, then the gate, then the out projection
        assert blk._stages[i + 1 :] == (blk._gate, blk._out_proj)
        assert blk._sdpa.token_stride == geom.n_qkvg == 5760  # the slab's stride, like the dense stage would read it
        blk._check_declaration()  # the five-band W_qkvg [5760, 512] and every descriptor pass
        if torch.cuda.is_available():
            if _CUTEDSL_REQUIREMENT is not None:
                pytest.skip(str(_CUTEDSL_REQUIREMENT))  # the stages ahead import their CuTe DSL kernels to answer check_support()
            cc = tuple(torch.cuda.get_device_capability())
            if cc != (10, 7):
                # since the cc 10.0 admission of the dense forms (2026-10-29) a part other than Rubin hears the block's SPARSE PART
                # STATEMENT first -- the block-level part row before any stage, then the sparse stage's own gate ahead of the adapter
                # (whose record would admit the explicit lean tile on cc 10.0 / 10.3) -- naming the lift; the record's Rubin-line
                # decline and the gather4 part floor are the standalone adapter's words there
                part = rf"sparse forms .* served on cc 10\.7 \(Rubin\) only.*found SM{cc[0]}{cc[1]}"
                with pytest.raises(NotImplementedError, match=part):
                    blk.check_support()
                assert not blk._is_supported and blk._sdpa._impl is None  # declined before the stage was asked: no adapter, no kernel module
                with pytest.raises(NotImplementedError, match=part):
                    blk._sdpa.check_support()  # the stage on its own: the same statement, before the adapter is built
                assert blk._sdpa._impl is None
            else:
                assert blk.check_support() and blk._is_supported
    # The fused projection fork's own decline is geometry-first (reads the same on every device).
    with pytest.raises(NotImplementedError, match="256-column tiles"):
        _FusedQkvProjection(geom, batch=1, seq_len=8, dtype=torch.bfloat16, want_rstd=False).check_support()
    # ... and the SAME fork accepts the geometry WITHOUT the band as far as its geometry gates go (what it says next
    # is the device's business: the rendering is for sm_107a).
    fork = _FusedQkvProjection(GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec()), batch=1, seq_len=8, dtype=torch.bfloat16, want_rstd=False)
    try:
        fork.check_support()
    except NotImplementedError as e:
        assert "indexer band" not in str(e)


def test_index_k_raw_locates_the_slab_inside_the_workspace():
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_band=True))
    b, s = 1, 8
    blk = GatedAttentionBlockFwd(**_samples(geom, batch=b, seq_len=s), geometry=geom)
    lay = blk._layout()
    ws = torch.zeros(lay.total_bytes, dtype=torch.uint8, device=_DEV)
    t = b * s
    slab = _view(ws, lay.proj, (t, geom.n_qkvg), torch.bfloat16)
    slab.copy_(torch.arange(t * geom.n_qkvg, device=_DEV).view(t, geom.n_qkvg).to(torch.bfloat16))
    k_raw = blk.index_k_raw(ws)
    off = geom.index_k_raw_offset
    assert k_raw.shape == (b, s, 1, 128) and k_raw.data_ptr() == slab.data_ptr() + off * 2
    assert torch.equal(k_raw.reshape(t, 128), slab[:, off : off + 128])
    with pytest.raises(ValueError, match="slab ends at byte"):
        blk.index_k_raw(ws[: lay.proj + 16])
    dense = GatedAttentionBlockFwd(
        **_samples(GatedAttentionBlockGeometry(**_SMALL_D256), batch=b, seq_len=s), geometry=GatedAttentionBlockGeometry(**_SMALL_D256)
    )
    with pytest.raises(ValueError, match="declares no indexer band"):
        dense.index_k_raw(ws)


# ---------------------------------------------------------------------------
# The block's declines, the adapter's declines and the record are ONE fact (any device, no GPU needed)
# ---------------------------------------------------------------------------


def _record():
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SPARSE_CAPABILITIES

    return SPARSE_CAPABILITIES


def test_the_api_constants_and_the_block_declines_read_the_sparse_record():
    """The block transcribes NOTHING of the sparse core's claims: the API's list constants agree with the record (a drift
    is a test failure, not a silent narrowing); the dtype / head-dim / GQA / block-size / top_k declines come from the
    record's fields (a request at each bound is served iff the record says so); the gate / bottom-right declines name the
    record's state and the THD arm constructs because the record claims it (and declines naming the record where it does
    not); and ``_SparseSdpa`` reads the same record object the adapter enforces."""
    from cudnn.gated_attention_block.api import _check_qsa_geometry_against_record, _sparse_record, _sparse_record_torch_dtypes
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    rec = _record()
    assert _sparse_record() is rec and _SparseSdpa._record() is rec and SparseGqaFwdDslSm107.claims() is rec
    assert QSA_TOP_K_MAX == rec.index_top_k_max and QSA_TOP_K_ALIGN == rec.index_top_k_min and QSA_BLOCK_SIZE in rec.index_block_sizes
    assert set(_sparse_record_torch_dtypes(rec)) == {torch.bfloat16, torch.float16}
    assert (256, 256) in rec.d_shapes and rec.index_gqa_group_max == 16 and rec.index_gqa_group_min == 1
    # Every dtype of the record is served, every other activation dtype declined naming the served set.
    for dt in _sparse_record_torch_dtypes(rec):
        _check_qsa_geometry_against_record(GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec()), dt)
    with pytest.raises(NotImplementedError, match="bf16 / f16"):
        _check_qsa_geometry_against_record(GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec()), torch.float32)
    # The GQA bound is the record's: served AT the bound, declined one past it.
    at = GatedAttentionBlockGeometry(**{**_SMALL_D256, "h_q": 2 * rec.index_gqa_group_max, "h_kv": 2}, qsa=QsaSpec())
    _check_qsa_geometry_against_record(at, torch.bfloat16)
    past = GatedAttentionBlockGeometry(**{**_SMALL_D256, "h_q": 2 * (rec.index_gqa_group_max + 1), "h_kv": 2}, qsa=QsaSpec())
    with pytest.raises(NotImplementedError, match=rf"h_q // h_kv <= {rec.index_gqa_group_max}"):
        _check_qsa_geometry_against_record(past, torch.bfloat16)
    # The head dim is the record's d_shapes.
    with pytest.raises(NotImplementedError, match="d_head == 256"):
        _check_qsa_geometry_against_record(GatedAttentionBlockGeometry(**{**_SMALL_D256, "d_head": 128}, qsa=QsaSpec()), torch.bfloat16)
    # top_k: every multiple of 4 in the record's range is served; the API's validator and the record agree on the bounds.
    for top_k in (rec.index_top_k_min, 64, rec.index_top_k_max):
        _check_qsa_geometry_against_record(GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(top_k=top_k)), torch.bfloat16)
    # The arm declines of the declaration follow the record's state AND the block's own binding: an arm the record claims
    # and the block binds CONSTRUCTS the block (the fused epilogue gate and THD, both since 2026-10-08); an arm the record
    # claims but the block does not bind yet is refused naming the BLOCK's state, never the record's; an arm the record
    # declines is refused naming the record.
    block_binds = {"epilogue_gate", "thd"}  # the arms the block binds through its sparse stage
    for blk_kw, geom_kw, field in (
        (dict(thd=True, num_sequences=1, max_seq_len=8), {}, "thd"),
        (dict(fuse_gate=True), {}, "epilogue_gate"),
        ({}, dict(causal_bottom_right=True), "bottom_right"),
    ):
        geom = GatedAttentionBlockGeometry(**{**_SMALL_D256, **geom_kw}, qsa=QsaSpec())
        if getattr(rec, field) and field in block_binds:
            with _no_host_sync():
                blk = GatedAttentionBlockFwd(**_samples(geom), geometry=geom, **blk_kw)
            assert isinstance(blk._sdpa, _SparseSdpa), field
            if field == "epilogue_gate":
                assert (
                    blk.fuse_gate and blk._sdpa.fuse_gate and blk._gate is None
                ), "fuse_gate under QsaSpec binds the gate through the sparse stage; stage (5) is not built"
            if field == "thd":
                assert blk._sdpa.thd, "thd under QsaSpec binds the packed sequences through the sparse stage"
            continue
        with pytest.raises(NotImplementedError) as e:
            GatedAttentionBlockFwd(**_samples(geom), geometry=geom, **blk_kw)
        assert ("the sparse adapter's record declines" in str(e.value)) == (not getattr(rec, field)), (field, str(e.value))


def _adapter_request(**over):
    """A served request through the adapter with storage-free operands at the 24/2 geometry (B=1, S=64), the Rubin cc
    given explicitly; ``over`` overrides the constructor arguments."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107, SparseOperandDesc

    B, S, H, KH, D, top_k = 1, 64, 24, 2, 256, 512
    half = torch.bfloat16
    kw = dict(
        q=SparseOperandDesc((B, S, H, D), (S * H * D, H * D, D, 1), half),
        k=SparseOperandDesc((B, S, KH, D), (S * KH * D, KH * D, D, 1), half),
        v=SparseOperandDesc((B, S, KH, D), (S * KH * D, KH * D, D, 1), half),
        o=SparseOperandDesc((B, S, H, D), (S * H * D, H * D, D, 1), half),
        lse=SparseOperandDesc((B, H, S), (H * S, S, 1), torch.float32),
        block_ids=SparseOperandDesc((B * S, top_k), (top_k, 1), torch.int32),
        block_lens=SparseOperandDesc((B * S,), (1,), torch.int32),
        top_k=top_k,
        device_cc=(10, 7),
    )
    kw.update(over)
    return SparseGqaFwdDslSm107(**kw)


@_needs_sm107_target
def test_adapter_accept_decline_set_equals_the_record():
    """The record-driven consistency check (Form A): for EVERY arm field of the record, a WELL-FORMED request that asks for
    the arm is accepted by the adapter's ``check_support`` iff the record claims it (a claimed arm's request carries the
    arm's own operands -- the paged read's pools, table and lengths; THD's two length descriptors of a one-sequence packing on
    the batch-extent-1 operands, in the ``[B]`` and, for ``cu_seq_len``, the ``[B + 1]`` form; an unclaimed arm's request is a
    typed decline naming the arm); the served baseline (dense BSHD
    bf16 / f16, d 256, caller lists, per-batch KV lengths) passes on any host whose DSL renders ``sm_107a`` (the cc is handed
    in, nothing is launched); where the DSL lacks the target the module skips this test by the adapter's own requirement."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseOperandDesc

    rec = _record()
    assert _adapter_request().check_support()  # the served baseline, bf16
    f16 = {nm: SparseOperandDesc((1, 64, h, 256), (64 * h * 256, h * 256, 256, 1), torch.float16) for nm, h in (("q", 24), ("k", 2), ("v", 2), ("o", 24))}
    assert _adapter_request(**f16).check_support()  # ... and f16: every member of the record's dtypes
    # the paged read's operands at the baseline geometry (B = 1, S = 64 -> 4 pages of 16; HND-compact pools)
    page, max_pages, num_pages = 16, 4, 6
    paged = dict(
        paged_kv=True,
        page_size=page,
        k=SparseOperandDesc((num_pages, 2, page, 256), (2 * page * 256, page * 256, 256, 1), torch.bfloat16),
        v=SparseOperandDesc((num_pages, 2, page, 256), (2 * page * 256, page * 256, 256, 1), torch.bfloat16),
        block_table=SparseOperandDesc((1, max_pages), (max_pages, 1), torch.int32),
        seq_kv_lens=SparseOperandDesc((1,), (1,), torch.int32),
    )
    arms = {
        "thd": dict(thd=True, seq_q_lens=SparseOperandDesc((1,), (1,), torch.int32), seq_kv_lens=SparseOperandDesc((1,), (1,), torch.int32)),
        # the [B + 1] prefix-sum form of the THD lengths (the record's cu_seq_len, reached through thd)
        "cu_seq_len": dict(
            thd=True,
            cu_seq_q_lens=True,
            cu_seq_kv_lens=True,
            seq_q_lens=SparseOperandDesc((2,), (1,), torch.int32),
            seq_kv_lens=SparseOperandDesc((2,), (1,), torch.int32),
        ),
        "paged_kv": paged,
        # the gate is an OPERAND: O-shaped in Q's dtype (here a slab-strided column slice, token stride 13312)
        "epilogue_gate": dict(epilogue_gate=SparseOperandDesc((1, 64, 24, 256), (64 * 13312, 13312, 256, 1), torch.bfloat16)),
        "split_kv": dict(split_kv=2),
        "bottom_right": dict(bottom_right=True),
        "sink": dict(sink=object()),
        "swa": dict(window_left=64),
        "padded": dict(seq_q_lens=SparseOperandDesc((1,), (1,), torch.int32)),
        "stats_log2": dict(stats_log2=True),
    }
    for field, over in arms.items():
        claimed = getattr(rec, field)
        req = _adapter_request(**over)
        if claimed:
            assert req.check_support(), field
        else:
            with pytest.raises(NotImplementedError, match="not served by the kernel body yet"):
                req.check_support()
    # the record's numeric bounds: served at the bound, declined past it
    assert _adapter_request(
        q=SparseOperandDesc((1, 64, 32, 256), (64 * 32 * 256, 32 * 256, 256, 1), torch.bfloat16),
        o=SparseOperandDesc((1, 64, 32, 256), (64 * 32 * 256, 32 * 256, 256, 1), torch.bfloat16),
        lse=SparseOperandDesc((1, 32, 64), (32 * 64, 64, 1), torch.float32),
    ).check_support()
    with pytest.raises(NotImplementedError, match=rf"\[1, {rec.index_gqa_group_max}\]"):
        _adapter_request(
            q=SparseOperandDesc((1, 64, 34, 256), (64 * 34 * 256, 34 * 256, 256, 1), torch.bfloat16),
            o=SparseOperandDesc((1, 64, 34, 256), (64 * 34 * 256, 34 * 256, 256, 1), torch.bfloat16),
            lse=SparseOperandDesc((1, 34, 64), (34 * 64, 64, 1), torch.float32),
        ).check_support()
    with pytest.raises(NotImplementedError, match=rf"\[{rec.index_top_k_min}, {rec.index_top_k_max}\]"):
        _adapter_request(
            top_k=rec.index_top_k_max + 4, block_ids=SparseOperandDesc((64, rec.index_top_k_max + 4), (rec.index_top_k_max + 4, 1), torch.int32)
        ).check_support()
    with pytest.raises(NotImplementedError, match="bf16 or f16"):
        _adapter_request(q=SparseOperandDesc((1, 64, 24, 256), (64 * 24 * 256, 24 * 256, 256, 1), torch.float32)).check_support()
    with pytest.raises(NotImplementedError, match="exactly"):
        _adapter_request(
            q=SparseOperandDesc((1, 64, 24, 128), (64 * 24 * 128, 24 * 128, 128, 1), torch.bfloat16),
            k=SparseOperandDesc((1, 64, 2, 128), (64 * 2 * 128, 2 * 128, 128, 1), torch.bfloat16),
            v=SparseOperandDesc((1, 64, 2, 128), (64 * 2 * 128, 2 * 128, 128, 1), torch.bfloat16),
            o=SparseOperandDesc((1, 64, 24, 128), (64 * 24 * 128, 24 * 128, 128, 1), torch.bfloat16),
        ).check_support()
    # the arch range off the record
    with pytest.raises(NotImplementedError, match="got cc 10.0"):
        _adapter_request(device_cc=(10, 0)).check_support()
    # a descriptor-declared adapter refuses to launch without the tensors (never a dereference of nothing)
    req = _adapter_request()
    req.check_support()
    with pytest.raises(ValueError, match="SparseOperandDesc"):
        req._bind("q", None, req.q, required=True)
    with pytest.raises(ValueError, match="must match its declaration"):
        req._bind("q", SparseOperandDesc((1, 64, 24, 256), (64 * 24 * 256 * 2, 24 * 256, 256, 1), torch.bfloat16), req.q, required=True)
    with pytest.raises(ValueError, match="was not declared"):
        req._bind("seq_kv_lens", SparseOperandDesc((1,), (1,), torch.int32), None, required=False)


@_needs_dsl_floor
def test_the_dsl_floor_declines_through_the_block_before_any_kernel_import(monkeypatch):
    """AGENTS.md Rule 7 through the BLOCK: a too-old DSL is a typed version decline from the sparse stage's ``check_support``
    (the adapter's first check), raised before the kernel module is loaded -- never a ``KeyError('sm_107a')`` from inside the
    DSL.  Both halves: the library floor (the installed version substituted below 4.7.0) and the ``sm_107a`` target (a DSL
    without it on a cc 10.7 device, the device capability substituted so the probe runs on any GPU).  Then the in-block indexer's
    three arms: its floor at DECLARATION (ahead of the head-group read that imports the scorer's package), its floor and its
    ``sm_107a`` target from the stage's own ``check_support``.  No launch, no GPU work."""
    import re

    import cudnn.frost.buffers as buffers

    if not torch.cuda.is_available():
        pytest.skip("the block's check_support reads the device capability")
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec())
    # the block's part rows (its own, then the sparse stage's, both before the adapter) precede the floor check since the cc 10.0
    # admission of the dense forms (2026-10-29): on a part other than Rubin they would answer first, so the probe reads a cc 10.7
    # device throughout (the sm_107a half below needs the same substitution)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))
    blk = GatedAttentionBlockFwd(**_samples(geom), geometry=geom)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(NotImplementedError, match=re.escape("found 4.6.2")):
        blk.check_support()
    assert blk._sdpa._impl is not None and blk._sdpa._impl._module is None, "the kernel module must not be loaded below the floor"
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    blk2 = GatedAttentionBlockFwd(**_samples(geom), geometry=geom)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        blk2.check_support()
    assert blk2._sdpa._impl._module is None
    # The in-block indexer (QsaSpec(index_source="indexer")) carries the floor TWICE: at DECLARATION -- the scorer's packed head
    # groups are read off the DSA tree, whose package imports its kernel modules, so the version decline must come first -- and
    # from the stage's own check_support; the sm_107a target from the stage.  The head-group reader is replaced by a failing
    # stub, so REACHING it below the floor is the failure, never an import error from inside the DSL.
    import cudnn.QSA.selection as qsa_select_forward

    real_groups = qsa_select_forward.indexer_scorer_head_groups
    geom_ix = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_source="indexer", index_band=True))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    blk_ix = GatedAttentionBlockFwd(**_samples(geom_ix), geometry=geom_ix)  # declared AT the floor (4.8.0 above): the stage arms run on it
    monkeypatch.setattr(qsa_select_forward, "indexer_scorer_head_groups", lambda: pytest.fail("the scorer's package was imported below the DSL floor"))
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(NotImplementedError, match=re.escape("found 4.6.2")):
        GatedAttentionBlockFwd(**_samples(geom_ix), geometry=geom_ix)  # the declaration-time arm
    with pytest.raises(NotImplementedError, match=re.escape("found 4.6.2")):
        blk_ix._indexer.check_support()  # the stage on its own
    assert blk_ix._indexer._k_recipe is None and blk_ix._sdpa._impl is None  # nothing compiled; no stage past the indexer was asked
    monkeypatch.setattr(qsa_select_forward, "indexer_scorer_head_groups", real_groups)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)  # the capability still reads (10, 7) from above
    with pytest.raises(NotImplementedError, match="sm_107a"):
        blk_ix._indexer.check_support()
    assert blk_ix._indexer._k_recipe is None


# ---------------------------------------------------------------------------
# execute(): the index tensors' FORM checks (never a value read, never a sync)
# ---------------------------------------------------------------------------


@_needs_dsl_floor
@pytest.mark.parametrize("buffer", ["w_iq_norm", "w_ik_norm", "index_k_compressed", "block_ids_out"])
def test_execute_refuses_a_misaligned_indexer_buffer_before_any_stage_runs(buffer):
    """The indexer form's 16-byte rule -- the caller-list form's ``block_ids`` rule (below) on every buffer the indexer reads or writes in
    16-byte vectors: the two norm weights and the compressed-key cache (the compress kernel's own check would refuse a storage-offset
    view, but only at stage (4i), after the projection, the norm + RoPE and a declared write-through have run) and ``block_ids_out``
    (the list stage (4) binds as a TMA / bulk-copy base).  A view a few bytes off a 16-byte boundary -- contiguous, the right dtype,
    shape and device -- is refused by name HERE, before the first stage runs; the same view at a 16-byte offset is the control and passes
    every form check (the next refusal is the plan's).  Form only, no host sync."""
    spec = QsaSpec(index_source="indexer", index_band=True)
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=spec)
    b, s = 2, 8
    kw = _samples(geom, batch=b, seq_len=s)
    args = tuple(kw[k] for k in ("sample_h", "sample_w_qkvg", "sample_w_q_norm", "sample_w_k_norm", "sample_cos", "sample_sin", "sample_w_o", "sample_out"))
    ws = torch.zeros(16, dtype=torch.uint8, device=_DEV)
    w = torch.ones(spec.index_head_dim, dtype=torch.bfloat16, device=_DEV)
    forms = {
        "w_iq_norm": ((spec.index_head_dim,), torch.bfloat16, 1.0),
        "w_ik_norm": ((spec.index_head_dim,), torch.bfloat16, 1.0),
        "index_k_compressed": ((b, s // spec.block_size, spec.index_head_dim), torch.bfloat16, 0.0),
        "block_ids_out": ((b * s, spec.top_k), torch.int32, 0),
    }
    shape, dtype, fill = forms[buffer]
    e = _itemsize(dtype)

    def offset_view(offset_elems):
        flat = torch.full((torch.Size(shape).numel() + offset_elems,), fill, dtype=dtype, device=_DEV)
        return flat[offset_elems:].view(*shape)

    bad, good = offset_view(1), offset_view(16 // e)  # one element (2 or 4 bytes) off a 16-byte boundary; 16 bytes off = aligned
    assert bad.data_ptr() % 16 == e and good.data_ptr() % 16 == 0 and bad.is_contiguous() and good.is_contiguous()
    base = dict(w_iq_norm=w, w_ik_norm=w)
    with _no_host_sync():
        blk = GatedAttentionBlockFwd(**kw, geometry=geom)
        with pytest.raises(ValueError, match=rf"{buffer} must start at a 16-byte aligned address"):
            blk.execute(*args, ws, **{**base, buffer: bad})
        with pytest.raises(RuntimeError, match="call compile"):  # the control passes every form check: the next refusal is the plan's
            blk.execute(*args, ws, **{**base, buffer: good})


def test_execute_form_checks_on_the_index_tensors():
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec())
    dense_geom = GatedAttentionBlockGeometry(**_SMALL_D256)
    b, s, top_k = 2, 8, geom.qsa.top_k
    t = b * s
    kw = _samples(geom, batch=b, seq_len=s)
    kw_d = _samples(dense_geom, batch=b, seq_len=s)
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
    good, good_dense, lens = i32(t, top_k), i32(b, s, top_k), i32(t)
    bad = {
        "int64": (torch.zeros(t, top_k, dtype=torch.int64, device=_DEV), "int32"),
        "shape": (i32(t, top_k + 1), r"must be \[16, 512\] or \[2, 8, 512\]"),
        "shared_list": (i32(b, top_k), "decode mode"),
        "rank": (i32(t), r"must be \[16, 512\]"),
        "strided": (i32(top_k, t).t(), "contiguous"),
        "not_a_tensor": ([[0] * top_k] * t, "torch.Tensor"),
        # a storage-offset view 4 bytes off a 16-byte boundary: contiguous, int32, the right shape -- and refused by the block HERE, before
        # the first stage runs (the sparse core binds block_ids as a TMA / bulk-copy base; the adapter refuses it too, but at stage (4))
        "misaligned": (i32(t * top_k + 1)[1:].view(t, top_k), "16-byte"),
    }
    good_offset16 = i32(t * top_k + 4)[4:].view(t, top_k)  # the control: a 16-byte storage offset is aligned and well-formed
    if torch.cuda.is_available():
        bad["device"] = (torch.zeros(t, top_k, dtype=torch.int32), "device")
    ws = torch.zeros(16, dtype=torch.uint8, device=_DEV)
    with _no_host_sync():
        blk = GatedAttentionBlockFwd(**kw, geometry=geom)
        dense = GatedAttentionBlockFwd(**kw_d, geometry=dense_geom)
        args = (
            kw["sample_h"],
            kw["sample_w_qkvg"],
            kw["sample_w_q_norm"],
            kw["sample_w_k_norm"],
            kw["sample_cos"],
            kw["sample_sin"],
            kw["sample_w_o"],
            kw["sample_out"],
            ws,
        )
        args_d = (
            kw_d["sample_h"],
            kw_d["sample_w_qkvg"],
            kw_d["sample_w_q_norm"],
            kw_d["sample_w_k_norm"],
            kw_d["sample_cos"],
            kw_d["sample_sin"],
            kw_d["sample_w_o"],
            kw_d["sample_out"],
            ws,
        )
        # A dense block refuses a list (before anything that needs a plan).
        with pytest.raises(ValueError, match="declared without geometry.qsa"):
            dense.execute(*args_d, block_ids=good)
        with pytest.raises(ValueError, match="declared without geometry.qsa"):
            dense.execute(*args_d, block_lens=lens)
        # A sparse block with caller lists needs block_ids.
        with pytest.raises(ValueError, match="needs block_ids"):
            blk.execute(*args)
        for name, (ids, match) in bad.items():
            with pytest.raises(ValueError, match=match):
                blk.execute(*args, block_ids=ids)
        for lens_bad, match in ((lens.to(torch.int64), "int32"), (i32(t + 1), r"must be \[16\] or \[2, 8\]"), (i32(t, 2)[:, 0], "contiguous")):
            with pytest.raises(ValueError, match=match):
                blk.execute(*args, block_ids=good, block_lens=lens_bad)
        # Well-formed lists pass every form check: the next refusal is the plan's (this block compiles only once the core lands).
        for ids, ln in ((good, None), (good_dense, None), (good, lens), (good_dense, i32(b, s)), (good_offset16, None)):
            with pytest.raises(RuntimeError, match="call compile"):
                blk.execute(*args, block_ids=ids, block_lens=ln)


def test_backward_declines_a_qsa_geometry_at_declaration():
    """No sparse training record exists (the forward declines save_for_backward under QsaSpec), so the backward refuses
    the geometry before it reads a shape -- its four-band unpacks never meet a five-band geometry."""
    z = torch.empty(0)
    for qsa in (QsaSpec(), QsaSpec(index_band=True)):
        geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=qsa)
        dy = torch.empty(1, 8, geom.d_model)
        w = torch.ones(geom.d_head)
        saved = SavedForBackward(h=z, gate=z, o=z, lse=z, rstd_q=z, rstd_k=z)
        with pytest.raises(NotImplementedError, match="block-sparse attention"):
            GatedAttentionBlockBwd(dy, saved, z, w, w, z, z, z, geom)


# ---------------------------------------------------------------------------
# Rubin: the unfused projection serves the five-band slab at the 24 / 2 geometry
# ---------------------------------------------------------------------------


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """One bf16 ulp at each element's magnitude: 7 explicit mantissa bits -> ``2^(exponent - 7)`` for ``|x| in [2^e, 2^(e+1))``."""
    _, e = torch.frexp(x.float())  # |x| = m * 2^e, m in [0.5, 1) -> |x| in [2^(e-1), 2^e)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), e - 8)


@_needs_dsl_floor  # the block's stages answer here (the sparse adapter declines by version below the floor): skip, never fail, there
@requires_rubin
def test_unfused_projection_serves_the_indexer_band_at_flash_next():
    """Stage (1) at N = 13952 (the four dense bands + the 640-column indexer band) through the block's own stage objects,
    into the block's own slab slot of a workspace (the whole block, sparse stage included, is supported on Rubin; the
    end-to-end module runs it).  The band equals ``h @ W_i^T`` to the GEMM suite's bf16 bar (one
    rounding of the fp32 accumulation), the four dense bands equal the four-band projection's, norm + RoPE leaves the band
    untouched, and the raw indexer key reads back as a zero-copy view at the right offset."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gated_block_reference import build_rope_tables

    dev = "cuda"
    geom, dense = GEOM_BAND, GEOM_DENSE
    b, s = 1, 512
    t, dm = b * s, geom.d_model
    gen = torch.Generator(device=dev).manual_seed(0)
    rnd = lambda *shape, std=0.02: (torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32) * std).to(torch.bfloat16)  # noqa: E731
    h = rnd(b, s, dm, std=1.0)
    w_dense = rnd(dense.n_qkvg, dm)
    w_i = rnd(geom.qsa.index_band_cols, dm)
    w_qkvg = torch.cat([w_dense, w_i], dim=0).contiguous()
    cos, sin = build_rope_tables(s, geom.rope_dim, batch=b, device=dev, dtype=torch.bfloat16)
    wn = torch.ones(geom.d_head, device=dev, dtype=torch.bfloat16)
    w_o = rnd(dm, geom.h_q * geom.d_head)
    out = torch.empty(b, s, dm, device=dev, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream

    blk = GatedAttentionBlockFwd(h, w_qkvg, wn, wn, cos, sin, w_o, out, geom)
    assert isinstance(blk._sdpa, _SparseSdpa)
    assert blk.check_support()  # the declaration, the projection (with the band), norm + RoPE and the sparse stage all accept on Rubin
    blk4 = GatedAttentionBlockFwd(h, w_dense, wn, wn, cos, sin, w_o, out, dense)

    def project(block, weight):
        block._proj.check_support()
        block._proj.compile()
        block._norm_rope.check_support()
        block._norm_rope.compile()
        lay = block._layout()
        ws = torch.full((lay.total_bytes + _align_up(max(block._proj.workspace_bytes(), 1)),), 0x7F, dtype=torch.uint8, device=dev)
        proj = _view(ws, lay.proj, (t, block.geom.n_qkvg), torch.bfloat16)
        block._proj.execute(h.view(t, dm), weight, proj, ws[lay.engine_scratch :], stream=stream)
        torch.cuda.synchronize()
        return ws, proj

    ws, proj = project(blk, w_qkvg)
    ws4, proj4 = project(blk4, w_dense)
    assert proj.shape == (t, 13952) and proj4.shape == (t, 13312)
    assert torch.isfinite(proj.float()).all()
    ref = h.view(t, dm).double() @ w_qkvg.double().t()  # exact products, fp64 sums: the reference the GEMM's fp32 accumulation rounds toward
    o_i, o_k = geom.qkvg_offsets[ProjBlock.INDEX], geom.index_k_raw_offset
    band, ref_band = proj[:, o_i:].float(), ref[:, o_i:].float()
    # The GEMM suite's bar for a bf16 GEMM output against its fp32 reference: 2 ulps of the max magnitude + reassociation.
    err = (band - ref_band).abs()
    scale = ref_band.abs().max().item()
    cos_band = torch.nn.functional.cosine_similarity(band.flatten(), ref_band.flatten(), dim=0).item()
    ulps = (err / _bf16_ulp(ref_band)).max().item()
    bitwise = (proj[:, o_i:] == ref[:, o_i:].to(torch.bfloat16)).float().mean().item()
    print(
        f"indexer band [{t}, 640]: cos {cos_band:.7f}, max|err| {err.max().item():.3e} (scale {scale:.3e}), max {ulps:.2f} bf16 ulps, bitwise vs rounded fp64 {100 * bitwise:.2f} %"
    )
    assert cos_band > 0.9999
    assert err.max().item() <= 2.0 * scale * 2.0**-8 + 1e-3 * scale
    # The four dense bands of the five-band slab vs the four-band projection of the same h: the same GEMM over a wider N.
    dense_part = proj[:, :o_i]
    err_d = (dense_part.float() - proj4.float()).abs()
    same = (dense_part == proj4).float().mean().item()
    print(f"dense bands [{t}, 13312]: bitwise equal to the four-band projection {100 * same:.2f} %, max|diff| {err_d.max().item():.3e}")
    scale_d = proj4.float().abs().max().item()
    assert err_d.max().item() <= 2.0 * scale_d * 2.0**-8 + 1e-3 * scale_d
    # The raw indexer key: the band's last 128 columns, as a zero-copy view -- through the free function and the block.
    k_raw = index_k_raw_view(proj, geom, b, s)
    assert k_raw.shape == (b, s, 1, 128) and k_raw.data_ptr() == proj.data_ptr() + o_k * 2
    assert torch.equal(k_raw.reshape(t, 128), proj[:, o_k:])
    k_raw_blk = blk.index_k_raw(ws)
    assert k_raw_blk.data_ptr() == k_raw.data_ptr() and torch.equal(k_raw_blk, k_raw)
    torch.testing.assert_close(k_raw.reshape(t, 128).float(), (h.view(t, dm).double() @ w_i[4 * 128 :].double().t()).float(), rtol=2.0**-6, atol=1e-3 * scale)
    # Norm + RoPE over the five-band slab (in place, the block's own call): Q / K change, GATE / V / INDEX stay bitwise.
    before = proj.clone()
    o_q, o_g, o_kk, o_v = geom.qkvg_offsets[:4]
    from cudnn.gated_attention_block.api import _cols

    blk._norm_rope.execute(
        _cols(proj, o_q, geom.h_q, geom.d_head), _cols(proj, o_kk, geom.h_kv, geom.d_head), wn, wn, cos, sin, current_stream=stream, flat=True
    )
    blk4._norm_rope.execute(
        _cols(proj4, o_q, geom.h_q, geom.d_head), _cols(proj4, o_kk, geom.h_kv, geom.d_head), wn, wn, cos, sin, current_stream=stream, flat=True
    )
    torch.cuda.synchronize()
    assert torch.equal(proj[:, o_g:o_kk], before[:, o_g:o_kk]) and torch.equal(proj[:, o_v:o_i], before[:, o_v:o_i])
    assert torch.equal(proj[:, o_i:], before[:, o_i:]), "norm + RoPE must not touch the indexer band"
    assert not torch.equal(proj[:, o_q:o_g], before[:, o_q:o_g])
    normed_same = (proj[:, :o_i] == proj4).float().mean().item()
    print(f"after norm + RoPE: dense bands bitwise equal to the four-band chain {100 * normed_same:.2f} %")
    err_n = (proj[:, :o_i].float() - proj4.float()).abs().max().item()
    scale_n = proj4.float().abs().max().item()
    assert err_n <= 2.0 * scale_n * 2.0**-8 + 1e-3 * scale_n
    # The engine scratch and the workspace beyond the slab were never the GEMM's output: the sentinel survives there.
    lay = blk._layout()
    assert (ws[lay.o : lay.o + 16] == 0x7F).all()


@_needs_dsl_floor
def test_an_e4m3_indexer_cache_declaration_appends_its_slots_and_the_cache_dtype_flips_the_execute_contract():
    """``QsaSpec(index_source="indexer", index_band=True, index_cache_dtype=torch.float8_e4m3fn)`` -- the fp8 indexer cache --
    CONSTRUCTS (no host sync): past the identity bound the carve is the bf16 indexer block's seven slots followed by the two e4m3
    slots ``ix_q8`` / ``ix_kbar8`` (``T x H_i x D_i`` and ``B x floor(S / 4) x D_i`` bytes, each 256-B aligned), below it ONE bf16
    staging slot ``ix_kbar`` for the cache cast (``ix_q`` / ``ix_q8`` / ``ix_kbar8`` unreserved) -- so every layout WITHOUT the
    attribute is byte-identical to before; the launch census is the bf16 indexer's plus two casts (plus one cache cast), below the
    bound the staging compress plus the cache cast; ``execute`` REQUIRES an e4m3 ``index_k_compressed`` on the e4m3 block and a bf16
    one on the bf16 block (the mirror images refused by name, each message naming the declared cache dtype).  Form only, no plan."""
    from cudnn.gated_attention_block.api import _qsa_indexer_slots

    e4 = QsaSpec(index_source="indexer", index_band=True, index_cache_dtype=_E4M3)
    g16 = QsaSpec(index_source="indexer", index_band=True)
    geom, geom16 = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=e4), GatedAttentionBlockGeometry(**_SMALL_D256, qsa=g16)
    with _no_host_sync():
        for b, s, lens in ((1, e4.identity_bound + 1, False), (2, 2300, True), (2, 8, False), (1, e4.identity_bound, False), (1, 3, False)):
            blk = GatedAttentionBlockFwd(**_samples(geom, batch=b, seq_len=s), geometry=geom, return_lse=True, seq_lens_present=lens)
            blk16 = GatedAttentionBlockFwd(**_samples(geom16, batch=b, seq_len=s), geometry=geom16, return_lse=True, seq_lens_present=lens)
            ix, ix16 = blk._indexer, blk16._indexer
            assert ix.fp8 and ix.cache_dtype == _E4M3 and not ix16.fp8 and ix16.cache_dtype == torch.bfloat16
            assert ix.selects == ix16.selects == (s > e4.identity_bound) and ix.n_blocks == s // 4
            lay, lay16 = blk._layout(), blk16._layout()
            t, nb, d_i, h_i = b * s, s // 4, e4.index_head_dim, e4.index_heads
            if ix.selects:
                # the five bf16-indexer slots at the SAME offsets (the sort scratch gone, 2026-10-29), then the two e4m3 slots, then the total
                for nm in ("ix_q", "ix_kbar", "ix_ids", "ix_scores", "ix_cand"):
                    assert getattr(lay, nm) == getattr(lay16, nm) >= 0, nm
                assert lay.ix_ids_sorted == lay16.ix_ids_sorted == lay.ix_sort_idx == lay16.ix_sort_idx == -1
                slots = _qsa_indexer_slots(geom, b, s, 2, ix.cand_floats, e4m3=True)
                assert [nm for nm, _ in slots][-2:] == ["ix_q8", "ix_kbar8"] and len(slots) == 7
                assert dict(slots)["ix_q8"] == t * h_i * d_i and dict(slots)["ix_kbar8"] == b * nb * d_i
                assert lay.ix_q8 == lay16.total_bytes and lay.ix_kbar8 == lay.ix_q8 + _align_up(t * h_i * d_i)
                assert lay.total_bytes == lay.ix_kbar8 + _align_up(b * nb * d_i)
                assert ix.launches_per_execute(False) == 7 == ix16.launches_per_execute(False) + 2  # + the canonical kernel (one launch, no windows)
                assert ix.launches_per_execute(True) == 8 == ix16.launches_per_execute(True) + 2
            else:
                # below the bound: the bf16 indexer block reserves nothing; the e4m3 one reserves the staging slot alone
                assert lay16.ix_kbar == -1 and lay.ix_kbar == lay16.total_bytes and lay.ix_q == lay.ix_q8 == lay.ix_kbar8 == -1
                assert lay.total_bytes == lay16.total_bytes + _align_up(b * nb * d_i * 2)
                assert ix.launches_per_execute(False) == 0 == ix16.launches_per_execute(False)
                assert ix.launches_per_execute(True) == (2 if nb > 0 else 1) and ix16.launches_per_execute(True) == 1
            # a zero-row staging slot (S < 4: no complete block) adds no bytes; get_workspace_size is not called (check_support = the scorer's device gate)
            assert (lay.total_bytes > lay16.total_bytes) == (nb > 0)
        # the execute contract: the cache's dtype is the DECLARED one, both ways, named in the refusal
        b, s = 2, 8
        kw, kw16 = _samples(geom, batch=b, seq_len=s), _samples(geom16, batch=b, seq_len=s)
        names = ("sample_h", "sample_w_qkvg", "sample_w_q_norm", "sample_w_k_norm", "sample_cos", "sample_sin", "sample_w_o", "sample_out")
        args, args16 = tuple(kw[k] for k in names), tuple(kw16[k] for k in names)
        ws = torch.zeros(16, dtype=torch.uint8, device=_DEV)
        w = torch.ones(e4.index_head_dim, dtype=torch.bfloat16, device=_DEV)
        kb = lambda nb, dt: torch.zeros(b, nb, e4.index_head_dim, dtype=dt, device=_DEV)  # noqa: E731
        blk, blk16 = GatedAttentionBlockFwd(**kw, geometry=geom), GatedAttentionBlockFwd(**kw16, geometry=geom16)
        with pytest.raises(
            ValueError, match=r"index_k_compressed must be a contiguous torch.float8_e4m3fn tensor \(QsaSpec.index_cache_dtype=torch.float8_e4m3fn"
        ):
            blk.execute(*args, ws, w_iq_norm=w, w_ik_norm=w, index_k_compressed=kb(2, torch.bfloat16))
        with pytest.raises(ValueError, match=r"index_k_compressed must be a contiguous torch.bfloat16 tensor \(QsaSpec.index_cache_dtype=None"):
            blk16.execute(*args16, ws, w_iq_norm=w, w_ik_norm=w, index_k_compressed=kb(2, _E4M3))
        with pytest.raises(ValueError, match=r"index_k_compressed must be \[B=2, >= 2"):
            blk.execute(*args, ws, w_iq_norm=w, w_ik_norm=w, index_k_compressed=kb(1, _E4M3))
        # well-formed: the next refusal is the plan's
        with pytest.raises(RuntimeError, match="call compile"):
            blk.execute(*args, ws, w_iq_norm=w, w_ik_norm=w, index_k_compressed=kb(5, _E4M3))
        with pytest.raises(RuntimeError, match="call compile"):
            blk.execute(*args, ws, w_iq_norm=w, w_ik_norm=w)


@_needs_dsl_floor  # the block's stage (1) answers here (its CuTe DSL GEMM is compiled): skip, never fail, below the floor
@requires_rubin
@pytest.mark.parametrize("b, s", [(4, 1), (1, 128)], ids=["B4-Sq1-T4", "B1-S128-T128"])
def test_unfused_projection_band_slab_takes_the_small_m_route_capture_safe_and_flag_free_at_decode_row_counts(monkeypatch, b, s):
    """Stage (1) at N = 13952 at DECODE row counts (``T = B x S_q <= 128``): the slab GEMM takes the small-M route's 128-row
    swap-AB tile (13952 = 109 x 128 on the route's overflow branch -- the one N off the 256 grid it names a tile for), so the stage
    compiles with ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` DELETED from the environment (the suite's conftest sets it per test; the
    flag is read live at compile) and with ``capture_safe=True`` on the stage (what the paged serving declaration sets) on the
    ``jit-only`` route at the route's own name -- no ranked list, no cuDNN handle created.  The band equals ``h @ W_i^T`` to the GEMM
    suite's bf16 bar and the four dense bands equal the four-band projection's over the same rows, as in the prefill cell above;
    the raw indexer key reads back as the zero-copy view at the band's last 128 columns."""
    from cudnn.gated_attention_block.api import _qkv_gate_projection
    from cudnn.gated_attention_block.kernels import proj_gemm
    from cudnn.gated_attention_block.kernels.proj_gemm import _FORCED_TILE_NAME, small_m_tile_config

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    dev = "cuda"
    geom, dense = GEOM_BAND, GEOM_DENSE
    t, dm = b * s, geom.d_model
    gen = torch.Generator(device=dev).manual_seed(1)
    rnd = lambda *shape, std=0.02: (torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32) * std).to(torch.bfloat16)  # noqa: E731
    h = rnd(b, s, dm, std=1.0)
    w_dense = rnd(dense.n_qkvg, dm)
    w_i = rnd(geom.qsa.index_band_cols, dm)
    w_qkvg = torch.cat([w_dense, w_i], dim=0).contiguous()
    stream = torch.cuda.current_stream().cuda_stream
    handles_before = dict(proj_gemm._GRAPH_HANDLES)
    expect = small_m_tile_config(t, geom.n_qkvg, dm, dtype=torch.bfloat16)  # the live part's SM count: the overflow branch below 218 SMs
    assert expect is not None and expect != _FORCED_TILE_NAME and expect.startswith("CONFIG_sm100_128x") and expect.endswith("_swapAB"), expect

    def project(block_geom, weight):
        st = _qkv_gate_projection(block_geom, batch=b, seq_len=s, dtype=torch.bfloat16)
        assert st.resolves_a_jit_tile()
        st.capture_safe = True  # the paged serving declaration's flag on the stage
        st.check_support()
        st.compile()
        assert st._plan.capture_safe and st._plan.route == "jit-only" and st._plan.jit is not None, (st._plan.route, st._plan.jit)
        assert st._plan.graph.plans == [], "the capture-safe stage built a ranked list"
        proj = torch.full((t, block_geom.n_qkvg), float("nan"), device=dev, dtype=torch.bfloat16)
        ws = torch.empty(max(st.workspace_bytes(), 16), dtype=torch.uint8, device=dev)
        st.execute(h.view(t, dm), weight, proj, ws, stream=stream)
        torch.cuda.synchronize()
        return st, proj

    st5, proj = project(geom, w_qkvg)
    st4, proj4 = project(dense, w_dense)
    assert st5._plan.tile_config_name == expect, (st5._plan.tile_config_name, expect)
    assert st4._plan.tile_config_name == small_m_tile_config(t, dense.n_qkvg, dm, dtype=torch.bfloat16)
    assert dict(proj_gemm._GRAPH_HANDLES) == handles_before, "the jit-only route created a cuDNN handle"
    assert proj.shape == (t, 13952) and torch.isfinite(proj.float()).all()
    ref = h.view(t, dm).double() @ w_qkvg.double().t()
    o_i = geom.qkvg_offsets[ProjBlock.INDEX]
    band, ref_band = proj[:, o_i:].float(), ref[:, o_i:].float()
    err, scale = (band - ref_band).abs(), ref_band.abs().max().item()
    cos_band = torch.nn.functional.cosine_similarity(band.flatten(), ref_band.flatten(), dim=0).item()
    print(f"\n[T={t}] band slab on {st5._plan.tile_config_name}: band cos {cos_band:.7f}, max|err| {err.max().item():.3e} (scale {scale:.3e})")
    assert cos_band > 0.9999
    assert err.max().item() <= 2.0 * scale * 2.0**-8 + 1e-3 * scale
    err_d, scale_d = (proj[:, :o_i].float() - proj4.float()).abs(), proj4.float().abs().max().item()
    assert err_d.max().item() <= 2.0 * scale_d * 2.0**-8 + 1e-3 * scale_d
    k_raw = index_k_raw_view(proj, geom, b, s)
    assert k_raw.shape == (b, s, 1, 128) and k_raw.data_ptr() == proj.data_ptr() + geom.index_k_raw_offset * 2
