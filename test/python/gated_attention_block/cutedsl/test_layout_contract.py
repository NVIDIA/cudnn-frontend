# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-(1) layout contract for the gated attention block. CPU-only, no GPU.

These pin the things that are silently wrong rather than loudly failing: a
straddling GEMM output tile, a mis-split double-width ``q_proj``, a block
boundary that stops being tile-aligned when someone changes a head count.
"""

import pytest
import torch

from cudnn.gated_attention_block import (
    QKVG_TILE_ALIGN,
    GatedAttentionBlockGeometry,
    ProjBlock,
    build_fused_qkvg_weight,
)

pytestmark = pytest.mark.L0


# One full-attention layer of Qwen3.5-397B at TP=1.
GEOM_397B = GatedAttentionBlockGeometry(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)
GEOM_SMALL = GatedAttentionBlockGeometry(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16)


def test_n_axis_map_matches_the_documented_397b_numbers():
    g = GEOM_397B
    assert g.n_qkvg == 17408
    assert g.qkvg_block_widths == (8192, 8192, 512, 512)
    assert g.qkvg_offsets == (0, 8192, 16384, 16896)
    assert g.qkvg_heads == (32, 32, 2, 2)
    assert g.gqa_ratio == 16
    assert g.scale == pytest.approx(1.0 / 16.0)


@pytest.mark.parametrize("geom", [GEOM_397B, GEOM_SMALL])
def test_block_for_column_is_exact_at_every_boundary(geom):
    offsets, widths = geom.qkvg_offsets, geom.qkvg_block_widths
    for block, (off, width) in enumerate(zip(offsets, widths)):
        assert geom.block_for_column(off) == (ProjBlock(block), 0, 0)
        last = off + width - 1
        assert geom.block_for_column(last) == (ProjBlock(block), width // geom.d_head - 1, geom.d_head - 1)
    with pytest.raises(ValueError):
        geom.block_for_column(geom.n_qkvg)
    with pytest.raises(ValueError):
        geom.block_for_column(-1)


@pytest.mark.parametrize("tile_n", [64, 128, 256])
def test_no_output_tile_straddles_two_blocks(tile_n):
    """The alignment invariant the epilogue's per-tile specialization rests on."""
    g = GEOM_397B
    plan = g.qkvg_tile_plan(tile_n)
    assert len(plan) == g.n_qkvg // tile_n
    counts = {b: plan.count(b) for b in ProjBlock}
    for block, width in zip(ProjBlock, g.qkvg_block_widths):
        assert counts[block] == width // tile_n
    # Blocks appear as contiguous runs, in ProjBlock order -- not interleaved.
    runs = [b for i, b in enumerate(plan) if i == 0 or plan[i - 1] != b]
    assert runs == list(ProjBlock)


def test_tile_plan_rejects_a_straddling_tile_size():
    """A geometry whose K/V block is not tile-aligned must fail loudly."""
    # widths (96, 96, 32, 32) -> N = 256, so TILE_N=128 divides N but tile 0
    # spans Q columns 0..95 and GATE columns 96..127.
    g = GatedAttentionBlockGeometry(d_model=128, h_q=3, h_kv=1, d_head=32, rope_dim=16)
    assert g.n_qkvg == 256 and g.qkvg_block_widths == (96, 96, 32, 32)
    with pytest.raises(ValueError, match="straddles"):
        g.qkvg_tile_plan(128)
    # ... and the same geometry is rejected outright at build time.
    with pytest.raises(ValueError, match="QKVG_TILE_ALIGN"):
        g.validate()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(h_q=6, h_kv=4), "divisible"),
        (dict(rope_dim=63), "even"),
        (dict(rope_dim=512), r"\[0, d_head"),
        (dict(d_head=96, h_kv=1), "QKVG_TILE_ALIGN"),
        (dict(qk_norm_eps=0.0), "qk_norm_eps"),
        (dict(attn_scale=-1.0), "attn_scale"),
        (dict(window_left=-5), "window_left"),
        (dict(is_causal=False, causal_bottom_right=True), "causal_bottom_right"),
    ],
)
def test_validate_rejects(kwargs, match):
    base = dict(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        GatedAttentionBlockGeometry(**base).validate()


def test_validate_accepts_the_shipped_geometries():
    GEOM_397B.validate()
    GEOM_SMALL.validate()
    assert QKVG_TILE_ALIGN == 64


@pytest.mark.parametrize("layout", ["flat", "per_head"])
def test_fused_weight_reproduces_the_four_projections(layout):
    """One GEMM through ``W_qkvg`` must equal four separate ``nn.Linear`` calls,
    with the blocks landing at exactly ``qkvg_offsets``."""
    torch.manual_seed(0)
    g = GEOM_SMALL
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model

    w_q = torch.randn(hq * d, dm)
    w_gate = torch.randn(hq * d, dm)
    w_k = torch.randn(hkv * d, dm)
    w_v = torch.randn(hkv * d, dm)

    if layout == "flat":
        w_q_gate = torch.cat([w_q, w_gate], dim=0)
    else:  # per_head: [Q_0 | GATE_0 | Q_1 | GATE_1 | ...]
        w_q_gate = torch.cat([torch.cat([w_q[h * d : (h + 1) * d], w_gate[h * d : (h + 1) * d]], dim=0) for h in range(hq)], dim=0)

    fused = build_fused_qkvg_weight(w_q_gate, w_k, w_v, g, q_gate_layout=layout)
    assert fused.shape == (g.n_qkvg, dm)

    h = torch.randn(2, 8, dm)
    proj = h @ fused.t()
    o_q, o_gate, o_k, o_v = g.qkvg_offsets
    torch.testing.assert_close(proj[..., o_q:o_gate], h @ w_q.t())
    torch.testing.assert_close(proj[..., o_gate:o_k], h @ w_gate.t())
    torch.testing.assert_close(proj[..., o_k:o_v], h @ w_k.t())
    torch.testing.assert_close(proj[..., o_v:], h @ w_v.t())


def test_the_two_q_gate_layouts_actually_differ():
    """Guards the parameter's reason for existing: if 'flat' and 'per_head'
    produced the same fused weight, defaulting wrong would be harmless and the
    knob would be noise. They do not, so picking wrong applies every gate to the
    wrong head -- finite, plausible, and wrong."""
    torch.manual_seed(0)
    g = GEOM_SMALL
    w_q_gate = torch.randn(2 * g.h_q * g.d_head, g.d_model)
    w_k = torch.randn(g.h_kv * g.d_head, g.d_model)
    w_v = torch.randn(g.h_kv * g.d_head, g.d_model)
    flat = build_fused_qkvg_weight(w_q_gate, w_k, w_v, g, q_gate_layout="flat")
    per_head = build_fused_qkvg_weight(w_q_gate, w_k, w_v, g, q_gate_layout="per_head")
    assert not torch.equal(flat, per_head)
    # ... and they differ only inside Q|GATE: K and V are untouched by the split.
    o_k = g.qkvg_offsets[ProjBlock.K]
    torch.testing.assert_close(flat[o_k:], per_head[o_k:])


def test_build_fused_qkvg_weight_rejects_wrong_shapes():
    g = GEOM_SMALL
    good_qg = torch.zeros(2 * g.h_q * g.d_head, g.d_model)
    good_kv = torch.zeros(g.h_kv * g.d_head, g.d_model)
    with pytest.raises(ValueError, match="rows"):
        build_fused_qkvg_weight(torch.zeros(7, g.d_model), good_kv, good_kv, g)
    with pytest.raises(ValueError, match="rows"):
        build_fused_qkvg_weight(good_qg, torch.zeros(7, g.d_model), good_kv, g)
    with pytest.raises(ValueError, match="columns"):
        build_fused_qkvg_weight(good_qg, good_kv, torch.zeros(g.h_kv * g.d_head, 7), g)
    with pytest.raises(ValueError, match="q_gate_layout"):
        build_fused_qkvg_weight(good_qg, good_kv, good_kv, g, q_gate_layout="interleaved")


# ---------------------------------------------------------------------------
# The Qwen3.8 family: Flash-Next's QSA layer at TP 1 / 2 / 4 and the two dense
# siblings. All share d_head 256 and rope_dim 64; they differ in d_model and in
# the head counts, which is exactly what the stage-(1) N-axis map, the GQA
# ratio and the norm kernel's tiling depend on. CPU arithmetic only -- the Rubin
# end-to-end cells for these geometries live in test_block_end_to_end.py.
# ---------------------------------------------------------------------------

GEOM_FLASH_NEXT = GatedAttentionBlockGeometry(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64)  # TP 1
GEOM_FLASH_NEXT_TP2 = GatedAttentionBlockGeometry(d_model=2560, h_q=12, h_kv=1, d_head=256, rope_dim=64)
GEOM_FLASH_NEXT_TP4 = GatedAttentionBlockGeometry(d_model=2560, h_q=6, h_kv=1, d_head=256, rope_dim=64)
GEOM_Q38_27B = GatedAttentionBlockGeometry(d_model=5120, h_q=24, h_kv=4, d_head=256, rope_dim=64)
GEOM_Q38_2P4T = GatedAttentionBlockGeometry(d_model=8192, h_q=64, h_kv=4, d_head=256, rope_dim=64)

# The indexer projection of a QSA layer: (index_heads + index_kv_heads) * index_head_dim = (4 + 1) * 128 columns,
# the fifth W_qkvg band a later change appends below V. Its width is pinned here so the alignment arithmetic is
# decided before the band exists.
INDEX_BAND_COLS = (4 + 1) * 128

# (geometry, n_qkvg, block widths, block offsets, gqa_ratio, fitted tile_rows of the TMA norm kernel -- 0 = no
# tile fits and the LDG kernel serves the geometry)
_QWEN38_FAMILY = [
    pytest.param(GEOM_FLASH_NEXT, 13312, (6144, 6144, 512, 512), (0, 6144, 12288, 12800), 12, 12, id="flash_next_tp1_24_2"),
    pytest.param(GEOM_FLASH_NEXT_TP2, 6656, (3072, 3072, 256, 256), (0, 3072, 6144, 6400), 12, 12, id="flash_next_tp2_12_1"),
    pytest.param(GEOM_FLASH_NEXT_TP4, 3584, (1536, 1536, 256, 256), (0, 1536, 3072, 3328), 6, 0, id="flash_next_tp4_6_1"),
    pytest.param(GEOM_Q38_27B, 14336, (6144, 6144, 1024, 1024), (0, 6144, 12288, 13312), 6, 12, id="q38_27b_24_4"),
    pytest.param(GEOM_Q38_2P4T, 34816, (16384, 16384, 1024, 1024), (0, 16384, 32768, 33792), 16, 16, id="q38_2p4t_64_4"),
]


@pytest.mark.parametrize("geom, n_qkvg, widths, offsets, gqa_ratio, _tile_rows", _QWEN38_FAMILY)
def test_qwen38_family_n_axis_map(geom, n_qkvg, widths, offsets, gqa_ratio, _tile_rows):
    """Every number a caller concatenates its checkpoint into, by name, per geometry."""
    geom.validate()
    assert geom.n_qkvg == n_qkvg
    assert geom.qkvg_block_widths == widths
    assert geom.qkvg_offsets == offsets
    assert geom.qkvg_heads == (geom.h_q, geom.h_q, geom.h_kv, geom.h_kv)
    assert geom.gqa_ratio == gqa_ratio
    assert geom.scale == pytest.approx(1.0 / 16.0)  # d_head ** -0.5 at d_head = 256
    assert geom.n_qkv == n_qkvg - geom.h_q * geom.d_head
    # Every band is a whole number of 256-column tiles, so the 256-tile plan below is well defined.
    assert all(w % 256 == 0 for w in widths)


@pytest.mark.parametrize("tile_n", [64, 128, 256])
@pytest.mark.parametrize("geom, n_qkvg, widths, offsets, gqa_ratio, _tile_rows", _QWEN38_FAMILY)
def test_qwen38_family_tile_plans_are_well_defined(geom, n_qkvg, widths, offsets, gqa_ratio, _tile_rows, tile_n):
    """No stage-(1) output tile straddles two of Q / GATE / K / V at any supported TILE_N."""
    plan = geom.qkvg_tile_plan(tile_n)
    assert len(plan) == n_qkvg // tile_n
    for block, width in zip(ProjBlock, widths):
        assert plan.count(block) == width // tile_n
    runs = [b for i, b in enumerate(plan) if i == 0 or plan[i - 1] != b]
    assert runs == list(ProjBlock)


@pytest.mark.parametrize("geom, n_qkvg, widths, offsets, gqa_ratio, tile_rows", _QWEN38_FAMILY)
def test_qwen38_family_fitted_norm_tile_rows(geom, n_qkvg, widths, offsets, gqa_ratio, tile_rows):
    """The TMA norm+RoPE kernel's fitted ``tile_rows`` (the largest r <= 16 dividing h_q, a multiple of h_kv and of
    the CTA's 4 warps): 12 at 24/2, 12/1 and 24/4, 16 at 64/4 -- and NONE at 6/1, where the typed verdict names the
    LDG kernel as the one that serves the geometry (so ``impl="auto"`` resolves to it instead of raising)."""
    from cudnn.gated_attention_block.api import _QK_NORM_ROPE_THREADS, _QkNormRope
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import validate_shape

    stage = _QkNormRope(geom, batch=1, seq_len=2, dtype=torch.bfloat16, want_rstd=False)
    fitted = stage.resolve_tile_rows()
    assert fitted == tile_rows
    if tile_rows == 0:
        with pytest.raises(ValueError, match="the LDG kernel serves this geometry"):
            validate_shape(geom.d_head, geom.rope_dim, geom.h_q, geom.h_kv, fitted, _QK_NORM_ROPE_THREADS)
    else:
        validate_shape(geom.d_head, geom.rope_dim, geom.h_q, geom.h_kv, fitted, _QK_NORM_ROPE_THREADS)


def test_index_band_tile_arithmetic():
    """The fifth band's alignment arithmetic, decided before the band exists.

    640 indexer columns are a whole number of QKVG_TILE_ALIGN tiles, so ``validate()`` admits the band; with it the
    Flash-Next N is 13952 = 218 x 64 = 109 x 128 -- but 13952 = 54.5 x 256, so a 256-wide tile plan is NOT well
    defined there (the band would end mid-tile), and the same holds at TP 2 and TP 4 (7296 = 57 x 128, 4224 = 33 x
    128). For the d_head-256 family the hazard arrives WITH the band: its four bands are multiples of 256 columns, so
    every four-band N of the family is a multiple of 512 (more generally, 128-aligned bands give N % 256 == 0 -- the
    grid below). It is not new to the method, though: bands need only QKVG_TILE_ALIGN = 64 columns, so a four-band
    geometry with 64-wide K / V bands reaches the very same N = 13952 = 2 x 64 x (108 + 1), and the refusal is pinned
    on it at the REAL N: ``qkvg_tile_plan`` refuses a TILE_N that does not divide N before it looks at any band
    boundary, which is the typed error the five-band plan will raise."""
    assert INDEX_BAND_COLS == 640
    assert INDEX_BAND_COLS % QKVG_TILE_ALIGN == 0
    assert INDEX_BAND_COLS % 128 == 0 and INDEX_BAND_COLS % 256 != 0
    for geom, n_with_band, tiles_128 in ((GEOM_FLASH_NEXT, 13952, 109), (GEOM_FLASH_NEXT_TP2, 7296, 57), (GEOM_FLASH_NEXT_TP4, 4224, 33)):
        assert geom.n_qkvg % 512 == 0  # four bands, each a multiple of 256 columns
        n = geom.n_qkvg + INDEX_BAND_COLS
        assert n == n_with_band == tiles_128 * 128
        assert n % 64 == 0
        assert n % 256 == 128  # 54.5 / 28.5 / 16.5 tiles of 256
    assert 13952 == 109 * 128 == 218 * 64
    assert (7296, 4224) == (57 * 128, 33 * 128)
    for d in (32, 64, 128, 256):
        for h_kv in (1, 2, 4):
            for h_q in range(h_kv, 65, h_kv):
                g = GatedAttentionBlockGeometry(d_model=256, h_q=h_q, h_kv=h_kv, d_head=d, rope_dim=16)
                if all(w % 128 == 0 for w in g.qkvg_block_widths):
                    assert g.n_qkvg % 256 == 0
    same_n = GatedAttentionBlockGeometry(d_model=256, h_q=108, h_kv=1, d_head=64, rope_dim=16)
    same_n.validate()
    assert same_n.n_qkvg == 13952 and same_n.qkvg_block_widths == (6912, 6912, 64, 64)
    assert len(same_n.qkvg_tile_plan(64)) == 218
    with pytest.raises(ValueError, match="must be positive and divide N=13952"):
        same_n.qkvg_tile_plan(256)


def test_no_qsa_geometry_snapshot_is_unchanged():
    """A geometry WITHOUT the indexer band keeps its four-band contract bitwise: the 397B numbers of
    ``test_n_axis_map_matches_the_documented_397b_numbers`` plus every derived tuple a caller or a kernel reads.
    Pinned as literals so a later fifth band, declared only on geometries that ask for it, cannot move them."""
    g = GEOM_397B
    snapshot = {
        "n_qkvg": g.n_qkvg,
        "n_qkv": g.n_qkv,
        "widths": g.qkvg_block_widths,
        "offsets": g.qkvg_offsets,
        "heads": g.qkvg_heads,
        "gqa_ratio": g.gqa_ratio,
        "scale": g.scale,
        "plan_64": tuple(int(b) for b in g.qkvg_tile_plan(64)),
        "plan_128": tuple(int(b) for b in g.qkvg_tile_plan(128)),
        "plan_256": tuple(int(b) for b in g.qkvg_tile_plan(256)),
        "first_col": tuple(g.block_for_column(o) for o in g.qkvg_offsets),
        "last_col": tuple(g.block_for_column(o + w - 1) for o, w in zip(g.qkvg_offsets, g.qkvg_block_widths)),
    }
    expected = {
        "n_qkvg": 17408,
        "n_qkv": 9216,
        "widths": (8192, 8192, 512, 512),
        "offsets": (0, 8192, 16384, 16896),
        "heads": (32, 32, 2, 2),
        "gqa_ratio": 16,
        "scale": 0.0625,
        "plan_64": (0,) * 128 + (1,) * 128 + (2,) * 8 + (3,) * 8,
        "plan_128": (0,) * 64 + (1,) * 64 + (2,) * 4 + (3,) * 4,
        "plan_256": (0,) * 32 + (1,) * 32 + (2,) * 2 + (3,) * 2,
        "first_col": ((ProjBlock.Q, 0, 0), (ProjBlock.GATE, 0, 0), (ProjBlock.K, 0, 0), (ProjBlock.V, 0, 0)),
        "last_col": ((ProjBlock.Q, 31, 255), (ProjBlock.GATE, 31, 255), (ProjBlock.K, 1, 255), (ProjBlock.V, 1, 255)),
    }
    assert snapshot == expected
    assert len(g.qkvg_block_widths) == len(g.qkvg_offsets) == len(g.qkvg_heads) == 4
