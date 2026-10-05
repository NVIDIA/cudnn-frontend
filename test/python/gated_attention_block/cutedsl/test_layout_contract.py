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
