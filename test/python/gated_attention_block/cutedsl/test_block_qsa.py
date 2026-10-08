# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``QsaSpec`` -- the block-sparse attention declaration of the gated attention block -- and the fifth (indexer) band.

Host cells (CPU arithmetic, the declaration-time declines on any GPU, the index-tensor form checks under
``torch.cuda.set_sync_debug_mode("error")``) plus ONE Rubin cell: the unfused projection serving the 13952-column slab at the
24-query-head / 2-KV-head geometry, its indexer band checked against the fp64 matmul and the raw indexer key read back as a
zero-copy view.  The sparse SDPA stage itself is a typed decline until the sparse core lands; its accept matrix is that
change's.
"""

import contextlib
import os
import sys

import pytest
import torch

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
    _FusedQkvProjection,
    _itemsize,
    _plan_workspace,
    _SparseSdpa,
    _view,
)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

_DEV = "cuda" if torch.cuda.is_available() else "cpu"

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
    assert (QSA_BLOCK_SIZE, QSA_TOP_K_MAX, QSA_TOP_K_ALIGN) == (4, 512, 4)
    assert q.index_band_cols == 640
    assert q.identity_bound == 2051  # floor(n / 4) <= 512 <=> n <= 2051
    assert QsaSpec(top_k=256).identity_bound == 1027 and QsaSpec(top_k=4).identity_bound == 19
    q.validate()
    QsaSpec(top_k=4).validate()
    QsaSpec(index_source="indexer", index_band=True).validate()
    assert hash(GEOM_BAND) != hash(GEOM_DENSE)  # frozen dataclasses: a compile key tells the two apart


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
_DECLINES = [
    pytest.param({}, QsaSpec(), dict(thd=True, num_sequences=1, max_seq_len=8), torch.bfloat16, NotImplementedError, "thd=True", id="thd"),
    pytest.param({}, QsaSpec(), dict(save_for_backward=True), torch.bfloat16, NotImplementedError, "save_for_backward=True", id="training"),
    pytest.param({}, QsaSpec(), dict(quant=_UNIT_QUANT), _E4M3, NotImplementedError, "quantized pipelines", id="quant"),
    pytest.param({}, QsaSpec(), dict(fuse_gate=True), torch.bfloat16, NotImplementedError, "fuse_gate=True", id="fuse_gate"),
    pytest.param({}, QsaSpec(index_band=True), dict(fuse_norm_rope=True), torch.bfloat16, NotImplementedError, "256-column tiles", id="fused_proj_x_band"),
    pytest.param(dict(causal_bottom_right=True), QsaSpec(), {}, torch.bfloat16, NotImplementedError, "causal_bottom_right=True", id="bottom_right"),
    pytest.param({}, QsaSpec(), {}, torch.float32, NotImplementedError, "bf16 / f16", id="dtype"),
    pytest.param(dict(h_q=48), QsaSpec(), {}, torch.bfloat16, NotImplementedError, r"h_q // h_kv <= 16", id="gqa_group"),
    pytest.param({}, QsaSpec(index_source="indexer", index_band=True), {}, torch.bfloat16, NotImplementedError, "in-block indexer", id="indexer"),
    pytest.param(dict(d_head=128), QsaSpec(), {}, torch.bfloat16, NotImplementedError, "d_head == 256", id="d_head"),
    pytest.param(dict(is_causal=False), QsaSpec(), {}, torch.bfloat16, ValueError, "is_causal=True", id="bidirectional"),
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


def test_a_legal_sparse_declaration_builds_the_sparse_stage_and_declines_at_check_support():
    """The accept half of the contract today: the block CONSTRUCTS with the sparse stage in stage (4)'s slot, every
    stage ahead of it accepts the five-band geometry, and the sparse stage is the typed decline (the core has not
    landed) -- on every device, with no host sync."""
    geom = GatedAttentionBlockGeometry(**_SMALL_D256, qsa=QsaSpec(index_band=True))
    kw = _samples(geom)
    with _no_host_sync():
        blk = GatedAttentionBlockFwd(**kw, geometry=geom)
        assert isinstance(blk._sdpa, _SparseSdpa) and blk.qsa is geom.qsa
        assert blk._stages[-1 - 2] is blk._sdpa or blk._sdpa in blk._stages  # in pipeline order, before the gate and the out projection
        assert blk._sdpa.token_stride == geom.n_qkvg == 5760  # the slab's stride, like the dense stage would read it
        blk._check_declaration()  # the five-band W_qkvg [5760, 512] and every descriptor pass
        if torch.cuda.is_available():
            with pytest.raises(NotImplementedError, match="sparse attention core"):
                blk.check_support()
            assert not blk._is_supported
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
# execute(): the index tensors' FORM checks (never a value read, never a sync)
# ---------------------------------------------------------------------------


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
    }
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
        for ids, ln in ((good, None), (good_dense, None), (good, lens), (good_dense, i32(b, s))):
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


@requires_rubin
def test_unfused_projection_serves_the_indexer_band_at_flash_next():
    """Stage (1) at N = 13952 (the four dense bands + the 640-column indexer band) through the block's own stage objects,
    into the block's own slab slot of a workspace -- the block itself declines at its sparse stage, AFTER the projection
    and norm + RoPE accepted the five-band geometry.  The band equals ``h @ W_i^T`` to the GEMM suite's bf16 bar (one
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
    with pytest.raises(NotImplementedError, match="sparse attention core"):
        blk.check_support()  # the declaration, the projection and norm + RoPE accepted; the sparse stage is the decline
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
