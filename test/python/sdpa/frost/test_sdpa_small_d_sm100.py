# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The SM100 (cc 10.0 / 10.3) small head-dim tiles of the f16/bf16 d128 FILE -- the CONFIG half (host-only, no GPU): the shared
d128-family module (``config_d128_family``) at the SM100 budget (227 KiB, 512 TMEM columns, no ``is_exclusive``), the budgets the
engine row / the adapter / the factory read as ONE arithmetic, the typed declines of every Rubin-only geometry, and the pins that the
plain d64 record and the Rubin configs are untouched by the lift.

Every budget number below is DERIVED by the same arithmetic the Rubin budgets run (``bias_resident_tiles(arch, ...)``) at the SM100
budget; the expected values are the as-built audit table of the port (operands at STAGES_KV 4, bf16, TILE_M 128, no Q u O alias), never a
Rubin threshold.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import _SM
from cudnn.sdpa.fwd import config_d128_family as fam
from cudnn.sdpa.fwd import config_sm100 as c0
from cudnn.sdpa.fwd import config_sm107 as c7
from cudnn.sdpa.fwd.config_sm100 import TemplateParams as TP

pytestmark = [pytest.mark.L0]


# --------------------------------------------------------------------------- the shared module: one field block, one map, two budgets


def test_the_two_arch_budgets_are_the_three_numbers_that_differ():
    assert c0.SM100_ARCH == fam.D128FamilyArch(
        name="SM100", smem_usable_bytes=227 * 1024, smem_fixed_overhead=2 * 1024, tmem_total_cols=512, tmem_exclusive_alloc=False
    )
    assert c7.SM107_ARCH == fam.D128FamilyArch(
        name="Rubin", smem_usable_bytes=320 * 1024, smem_fixed_overhead=2 * 1024, tmem_total_cols=576, tmem_exclusive_alloc=True
    )
    assert c7.SM107_ARCH.smem_usable_bytes == c7.SMEM_USABLE_BYTES and c7.SM107_ARCH.tmem_total_cols == c7.TMEM_TOTAL_COLS
    assert c0.SM100_ARCH.smem_usable_bytes == c0._SM100_MAX_DYN_SMEM


def test_both_cfg_lineages_inherit_the_one_small_tile_field_block_at_its_off_defaults():
    names = [f.name for f in dataclasses.fields(fam.D128FamilySmallTileFields)]
    assert names == [
        "PADDED_TAIL_FORM",
        "DEAD_SUBTILE_QUIET",
        "BIAS_MODE",
        "BIAS_RESIDENT_TILES",
        "BIAS_BLOCK_ROWS",
        "BIAS_SWZ_BYTES",
        "BIAS_BOXES_PER_TILE",
        "BIAS_EMPTY_ARRIVERS",
        "KV_MASK_FORM",
        "N_CHUNK",
        "STAGES_BIAS",
        "BIAS_MAX_FORM",
        "S_SLOTS",
        "BIAS_MCAST_CTAS",
        "Q_IN_TMEM",
    ]
    for cls in (c0.CfgD128, c0.CfgD64, c0.CfgD32, c7.CfgD128, c7.CfgD64, c7.CfgD32):
        assert issubclass(cls, fam.D128FamilySmallTileFields), cls
        off = fam.D128FamilySmallTileFields()
        inst = cls()
        for n in names:
            assert getattr(inst, n) == getattr(off, n), (cls, n)


def test_the_rubin_tmem_map_is_the_shared_formula_at_576_and_the_sm100_one_at_512():
    cfg7 = c7.make_cfg_d32(TP(d_flavor=32, cta_mma=1, dtype_qkv=2))[0]
    lay7 = c7.d128_family_tmem_layout(cfg7)
    assert lay7 == fam.d128_family_tmem_layout(cfg7, total_cols=576)
    assert (lay7.S_OFF, lay7.P_OFF, lay7.O_OFF, lay7.STATS_OFF) == ((0, 128), (64, 192), (256, 288), 544)
    cfg0 = c0.make_cfg_d32(TP(d_flavor=32, cta_mma=1, dtype_qkv=2))[0]
    lay0 = fam.d128_family_tmem_layout(cfg0, total_cols=c0.SM100_ARCH.tmem_total_cols)
    assert (lay0.TOTAL_COLS, lay0.S_OFF, lay0.P_OFF, lay0.O_OFF, lay0.STATS_OFF, lay0.Q_OFF, lay0.Q_COLS) == (
        512,
        (0, 128),
        (64, 192),
        (256, 288),
        480,
        (320, 336),
        16,
    )
    # the d128 map of the Rubin kernel is untouched by the lift: 0 / 128 | 64 / 192 | 256 / 384 | 544
    lay128 = c7.d128_family_tmem_layout(c7.make_cfg_d128(TP(d_flavor=128, cta_mma=2, dtype_qkv=2))[0])
    assert (lay128.S_OFF, lay128.P_OFF, lay128.O_OFF, lay128.STATS_OFF) == ((0, 128), (64, 192), (256, 384), 544)


# --------------------------------------------------------------------------- the SM100 budgets = the audit table (bf16, STAGES_KV 4, TILE_M 128)

# (D, TILE_N, pair, cga) -> (resident tiles, resident S_kv max, ring stages) at the 227 KiB budget.
_SM100_BUDGET_ROWS = [
    ((32, 128, False, 1), (2, 256, 2)),
    ((32, 128, True, 1), (4, 512, 3)),
    ((32, 64, False, 1), (5, 320, 3)),
    ((32, 64, True, 1), (10, 640, 3)),
    ((64, 128, False, 1), (0, 0, 0)),  # NO form: 192 KiB of operands leave no 32 KiB tile beside the 2 KiB slack
    ((64, 128, False, 2), (1, 128, 0)),
    ((64, 128, True, 1), (1, 128, 0)),
    ((64, 64, False, 1), (3, 192, 3)),
    ((64, 64, False, 2), (4, 256, 3)),
    ((64, 64, True, 1), (6, 384, 3)),
]


@pytest.mark.parametrize("geom, want", _SM100_BUDGET_ROWS, ids=[f"d{d}_t{tn}_{'pair' if p else 'obe'}_cga{c}" for (d, tn, p, c), _ in _SM100_BUDGET_ROWS])
def test_sm100_resident_and_ring_budgets_are_the_audit_table(geom, want):
    d, tile_n, pair, cga = geom
    tiles, s_kv_max, ring = want
    assert c0.sm100_bias_resident_tiles(d, cga, tile_n=tile_n, pair=pair) == tiles
    assert c0.sm100_bias_resident_s_kv_max(d, cga, tile_n=tile_n, pair=pair) == s_kv_max
    assert c0.sm100_bias_ring_stages(d, cga, tile_n=tile_n, pair=pair) == ring
    # the same arithmetic at the Rubin budget gives the Rubin numbers (the arch is a parameter, never a threshold)
    rubin = fam.bias_resident_tiles(c7.SM107_ARCH, d, cga, stages_kv=4, tile_m=128, tile_n=tile_n, pair=pair)
    assert rubin == c7.sm107_bias_resident_tiles(d, cga, tile_n=tile_n, pair=pair)
    assert rubin >= tiles


def test_the_sm100_budget_is_not_the_rubin_budget():
    """The one row the whole SM100 port turns on: D 64 @128 cga1 holds 128 / 384 bias columns on Rubin and NOTHING on SM100."""
    assert c7.sm107_bias_resident_s_kv_max(64, 1) == 128 and c7.sm107_bias_resident_s_kv_max(64, 1, pair=True) == 384
    assert c0.sm100_bias_resident_s_kv_max(64, 1) == 0 and c0.sm100_bias_resident_s_kv_max(64, 1, pair=True) == 128
    assert c0.sm100_bias_mode_pick(64, 1, 256, pair=True) == fam.BIAS_MODE_NONE
    assert c0.sm100_bias_mode_pick(64, 1, 128, pair=True) == fam.BIAS_MODE_RESIDENT
    # the 64-wide KV tile and cga2 are the D 64 forms that fit at 4 stages
    assert c0.sm100_bias_mode_pick(64, 1, 384, tile_n=64, pair=True) == fam.BIAS_MODE_RESIDENT
    assert c0.sm100_bias_mode_pick(64, 1, 1024, tile_n=64, pair=True) == fam.BIAS_MODE_STREAMED
    assert c0.sm100_bias_auto_cga(64, 128, (1, 2)) == 2 and c0.sm100_bias_auto_cga(64, 256, (1, 2)) is None
    assert sorted(c0.SM100_BIAS_D_SHAPES) == [(32, 32), (64, 64)]


def test_stages_kv_two_is_the_d64_at_128_budget_lever(monkeypatch):
    """The budget lever for D 64 @128 on SM100: at STAGES_KV 2 the operands are 128 KiB, a 3-tile resident block (S_kv 384) and a
    3-stage ring fit (2.5 KiB of slack) -- the sweep seam pins it, the budgets follow, the seam state keys the module."""
    assert c0.sm100_sweep_seam_state() is None
    monkeypatch.setattr(c0, "_STAGES_KV_OVERRIDE_FOR_SWEEPS", 2)
    assert c0.sm100_sweep_seam_state() == ("sm100_sweep_seams", 2)
    assert c0.sm100_small_d_stages_kv(64, 1) == 2
    assert c0.sm100_bias_resident_tiles(64, 1, pair=True) == 3 and c0.sm100_bias_ring_stages(64, 1, pair=True) == 3
    cfg = c0.make_cfg_d64(TP(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=4))[0]
    assert cfg.STAGES_KV == 2 and cfg.BIAS_RESIDENT_TILES == 3 and cfg.BIAS_BLOCK_ROWS == 128
    monkeypatch.setattr(c0, "_STAGES_KV_OVERRIDE_FOR_SWEEPS", 5)
    with pytest.raises(ValueError, match="validated at"):
        c0.sm100_small_d_stages_kv(64, 1)


# --------------------------------------------------------------------------- the small-tile factory: forms built, forms declined


def _cfg(**kw):
    return c0.MAKE_CFG[kw["d_flavor"]](TP(**kw))[0]


def test_the_small_tile_forms_build_with_the_sm100_geometry():
    cfg = _cfg(d_flavor=32, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=4, kv_mask_form=1)
    assert isinstance(cfg, c0.CfgD32) and (cfg.TILE_K, cfg.TILE_O, cfg.TILE_N, cfg.STAGES_KV, cfg.QO_ALIAS) == (32, 32, 128, 4, 0)
    assert (cfg.BIAS_MODE, cfg.BIAS_RESIDENT_TILES, cfg.BIAS_BLOCK_ROWS, cfg.BIAS_BOXES_PER_TILE, cfg.BIAS_EMPTY_ARRIVERS) == (1, 4, 128, 2, 8)
    assert (cfg.N_CHUNK, cfg.KV_MASK_FORM, cfg.SOFTMAX_REGS, cfg.CORRECTION_REGS, cfg.MMA_REGS, cfg.TOTAL_WARPS, cfg.READ_TILE_ARRIVERS) == (
        4,
        1,
        192,
        88,
        40,
        16,
        15,
    )
    assert 8 * cfg.SOFTMAX_REGS + 4 * cfg.CORRECTION_REGS + 4 * cfg.OTHER_REGS == 2048 == 128 * cfg.TOTAL_WARPS
    assert (cfg.Q_SWZ_BYTES, cfg.V_SWZ_BYTES, cfg.O_SWZ_BYTES, cfg.TILE_K_HW_BMM1) == (64, 64, 64, 16)
    ring = _cfg(d_flavor=32, cta_mma=1, dtype_qkv=2, bias_mode=2)
    assert (ring.BIAS_MODE, ring.STAGES_BIAS, ring.BIAS_RESIDENT_TILES, ring.BIAS_BLOCK_ROWS) == (2, 2, 0, 256)
    cga2 = _cfg(d_flavor=64, cta_mma=2, dtype_qkv=2, bias_mode=1)
    assert isinstance(cga2, c0.CfgD64) and (cga2.BIAS_RESIDENT_TILES, cga2.V_SWZ_BYTES, cga2.CGA_M, cga2.QO_ALIAS) == (
        1,
        64,
        2,
        0,
    )  # per-CTA V = 32 cols x 2 B = a 64-B atom
    t64 = _cfg(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=4, tile_n=64)
    assert (t64.TILE_N, t64.N_BMM2_CHUNKS, t64.BIAS_RESIDENT_TILES, t64.BIAS_BOXES_PER_TILE) == (64, 1, 6, 1)
    s2 = _cfg(d_flavor=32, cta_mma=1, dtype_qkv=2, tile_n=64, s_slots=2)
    assert s2.S_SLOTS == 2 and fam.d128_family_tmem_layout(s2, total_cols=512).O_OFF == (256, 288)
    wg3 = _cfg(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=6, tile_n=64, softmax_wgs=3)
    assert (wg3.TOTAL_WARPS, wg3.READ_TILE_ARRIVERS, wg3.SOFTMAX_REGS, wg3.CORRECTION_REGS, wg3.BIAS_EMPTY_ARRIVERS) == (20, 19, 120, 80, 12)
    assert 12 * wg3.SOFTMAX_REGS + 4 * wg3.CORRECTION_REGS + 4 * wg3.OTHER_REGS == 1920 == 96 * wg3.TOTAL_WARPS
    assert fam.d128_family_tmem_layout(wg3, total_cols=512).O_OFF == (192, 256, 320)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=1), "no resident bias block fits"),
        (
            dict(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=4),
            None,
        ),  # the pair item holds ONE 128-row tile at D 64 @128 cga1 (S_kv 128): it BUILDS; the row declines above 128
        (dict(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=2), "streamed bias ring"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=2, s_slots=2), "STATS region"),
        (dict(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=6, softmax_wgs=3), "TILE_N 64"),
        (dict(d_flavor=32, cta_mma=2, dtype_qkv=2), "outside the validated widths"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=0), "FP8/MXFP8"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=2, thd_varlen=True, seq_kv_lens_present=True), "dense, unsplit"),
        (dict(d_flavor=64, cta_mma=1, dtype_qkv=2, bias_mode=1, pack_gqa=True, qh_per_kh=4), "dense, unsplit"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=2, n_chunk=4), "needs bias_mode=1"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=2, bias_mode=1, kv_mask_form=3, bias_max_form=1), "additive kv_mask vector"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=2, padded_tail_form=1), "PLAIN padded body"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=2, bias_mode=1, dead_subtile_quiet=1), "one-batch-entry item without a bias"),
        (dict(d_flavor=32, cta_mma=1, dtype_qkv=2, q_in_tmem=1, cta_mma_=None), None),
    ],
    ids=[
        "d64_bias_cga1_at_128_no_form",
        "d64_pair_cga1_at_128_holds_one_tile",
        "d64_ring_cga1_at_128_no_ring",
        "s_slots_2_at_tile_n_128_overflows_512",
        "wg3_needs_tile_n_64",
        "d32_cga2_unvalidated",
        "fp8_is_not_a_small_tile",
        "thd_rides_the_d128_flavor",
        "pack_gqa_rides_the_d128_flavor",
        "pair_item_needs_a_bias",
        "hw_max_with_an_additive_vector",
        "premasked_tail_needs_the_padded_plain_body",
        "quiet_body_needs_no_bias",
        "q_in_tmem_builds",
    ],
)
def test_rubin_only_and_off_domain_geometries_decline_typed_on_sm100(kw, match):
    kw = {k: v for k, v in kw.items() if k != "cta_mma_"}
    if match is None:
        cfg = _cfg(**kw)
        assert cfg.Q_IN_TMEM == 1 or (cfg.BIAS_MODE == 1 and cfg.BIAS_RESIDENT_TILES == 1 and cfg.N_CHUNK == 4)
        return
    with pytest.raises(ValueError, match=match):
        _cfg(**kw)


def test_a_576_column_or_exclusive_or_oversized_geometry_is_a_message_naming_the_sm100_figure():
    """The acceptance line: the Rubin geometries the SM100 port must refuse raise with the SM100 number in the message."""
    cfg = _cfg(d_flavor=32, cta_mma=1, dtype_qkv=2, tile_n=64, s_slots=2)
    with pytest.raises(ValueError, match="480 of the 512-column allocation"):
        fam.validate_tile_n(dataclasses.replace(cfg, TILE_N=128, N_BMM2_CHUNKS=2), "probe", arch=c0.SM100_ARCH)
    big = dataclasses.replace(
        _cfg(d_flavor=64, cta_mma=1, dtype_qkv=2),
        STAGES_KV=4,
        BIAS_MODE=1,
        BIAS_RESIDENT_TILES=1,
        BIAS_BLOCK_ROWS=256,
        BIAS_BOXES_PER_TILE=2,
        BIAS_EMPTY_ARRIVERS=8,
    )
    with pytest.raises(ValueError, match="227 KiB usable SM100 carveout"):
        fam.validate_smem(big, "probe", arch=c0.SM100_ARCH, qo_alias=False, bias_bytes=fam.bias_resident_block_bytes(big))
    assert not c0.SM100_ARCH.tmem_exclusive_alloc and c0.SM100_ARCH.tmem_total_cols == 512


# --------------------------------------------------------------------------- the plain records and the Rubin configs are untouched


def test_the_plain_d64_record_keeps_its_own_path_and_geometry():
    """No small-tile field set = make_cfg_d64's own path: the aliased cga1 build, 4 stages, the CfgD128 register defaults -- the renderings
    the committed sm_100a records pin.  A small-tile field routes to the factory, whose record is NOT the plain one (no alias)."""
    plain = c0.make_cfg_d64(TP(d_flavor=64, cta_mma=1, dtype_qkv=2))[0]
    assert (plain.QO_ALIAS, plain.STAGES_KV, plain.SOFTMAX_REGS, plain.CORRECTION_REGS, plain.MMA_REGS, plain.BIAS_MODE) == (1, 4, 192, 88, 40, 0)
    assert not c0._small_tile_form_requested(TP(d_flavor=64, cta_mma=1, dtype_qkv=2))
    assert c0._small_tile_form_requested(TP(d_flavor=64, cta_mma=1, dtype_qkv=2, kv_mask_form=1))
    seam = c0.make_cfg_d64(TP(d_flavor=64, cta_mma=1, dtype_qkv=2, kv_mask_form=1))[0]
    assert (seam.QO_ALIAS, seam.KV_MASK_FORM, seam.BIAS_EMPTY_ARRIVERS) == (0, 1, 8)
    # the fp8 d64 flavors (the fp8 / mxfp8 d128 files at d_flavor 64) keep their records: no small-tile field, the plain path
    fp8 = c0.make_cfg_d64(TP(d_flavor=64, cta_mma=1, dtype_qkv=0))[0]
    assert fp8.BIAS_MODE == 0 and fp8.TILE_K_HW_BMM1 == 32
    assert 32 in c0.MAKE_CFG and c0.MAKE_CFG[32] is c0.make_cfg_d32


def test_the_rubin_wrappers_are_the_shared_arithmetic_at_the_rubin_budget():
    for d, cga, pair in ((32, 1, False), (32, 1, True), (64, 1, False), (64, 2, False), (64, 1, True), (128, 2, False)):
        want = fam.bias_resident_tiles(c7.SM107_ARCH, d, cga, stages_kv=c7._stages_kv_d128(2, cga, mxfp8=False, tile_k=d), tile_m=128, pair=pair)
        assert c7.sm107_bias_resident_tiles(d, cga, pair=pair) == want
    assert c7.sm107_bias_resident_tiles(32, 1) == 3 and c7.sm107_bias_resident_tiles(32, 1, pair=True) == 6
    assert c7.sm107_bias_resident_tiles(64, 1) == 1 and c7.sm107_bias_resident_tiles(64, 2) == 2 and c7.sm107_bias_resident_tiles(128, 2) == 0
    for name in (
        "BIAS_MODE_RESIDENT",
        "BIAS_STREAMED_MODES",
        "KV_MASK_BOOL_WORD",
        "BIAS_MAX_HW_UPPER_BOUND",
        "BIAS_RING_STAGES",
        "SMALL_D_STAGES_KV",
        "TCGEN05_V0_ADDR_LIMIT",
        "BIAS_SWZ_BYTES",
    ):
        assert getattr(c7, name) is getattr(fam, name), name


# --------------------------------------------------------------------------- the row and the adapter route the 32-wide tile on its dense leg only

_SM100_ROW = "sdpa_fwd_prefill_sm100"


def _sm100_spec():
    from cudnn.sdpa.fwd import engines

    return next(sp for sp in engines.ENGINE_SPECS if sp.name == _SM100_ROW)


def _sm100_facts(d, *, s=256, b=2, h=4, **kw):
    import cudnn
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    bf16 = cudnn.data_type.BFLOAT16
    base = dict(b=b, h_q=h, h_kv=h, s_q=s, s_kv=s, d_qk=d, d_v=d, dtype=bf16, dtype_o=bf16, device_cc=(10, 0), device_sm_count=148)
    base.update(kw)
    return SdpaGraphFacts(**base)


def test_the_sm100_row_serves_the_32_wide_tile_on_the_dense_leg_and_walks_past_it_otherwise():
    """The row's claim (Rule S2: the tracker's SM100 d32 column): D <= 32 on the dense leg selects (32, 32) at cga1; THD / paged / the score
    outputs / split-KV / PackGQA / a cga2 request walk past it onto the native d64 flavor -- the route a d <= 32 graph took before the tile."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import supported_cgas_for
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _sm100_spec().capabilities
    assert (32, 32) in caps.d_shapes and caps.dense_only_d_shapes == frozenset({(32, 32)}) == c0.SM100_DENSE_ONLY_D_SHAPES
    assert dict(caps.cgas_by_d_shape)[(32, 32)] == frozenset({1}) == c0.SM100_SMALL_D_CGAS[(32, 32)]
    for d in (8, 16, 24, 32):
        assert engines._selected_d_shape(caps, _sm100_facts(d)) == (32, 32), d
    assert engines._selected_d_shape(caps, _sm100_facts(40)) == (64, 64) and engines._selected_d_shape(caps, _sm100_facts(64)) == (64, 64)
    assert engines._selected_d_shape(caps, _sm100_facts(24, thd=True)) == (64, 64)
    assert engines._selected_d_shape(caps, _sm100_facts(24, has_score_max=True)) == (64, 64)
    assert engines._selected_d_shape(caps, _sm100_facts(24), SdpaFwdKnobs(cga=2)) == (64, 64)
    assert engines._selected_d_shape(caps, _sm100_facts(24), SdpaFwdKnobs(split_kv=2)) == (64, 64)
    assert engines._selected_d_shape(caps, _sm100_facts(24), SdpaFwdKnobs(cga=1)) == (32, 32)
    # the adapter's cga domain is the row's
    assert supported_cgas_for((32, 32), device_cc=(10, 0), fp8=False, pertensor=False) == (1,)
    assert supported_cgas_for((64, 64), device_cc=(10, 0), fp8=False, pertensor=False) == (1, 2)


def _desc(shape, dtype, name):
    from cudnn.api_base import TensorDesc

    b, h, s, d = shape
    stride = (s * h * d, d, h * d, 1)
    return TensorDesc(
        dtype=dtype, shape=tuple(shape), stride=stride, stride_order=TensorDesc._compute_stride_order(tuple(shape), stride), device="cuda", name=name
    )


def _api(d, *, s=256, h=4, dtype=torch.bfloat16, score=False, **kw):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    q, o = (_desc((1, h, s, d), dtype, n) for n in ("q", "o"))
    k, v = (_desc((1, h, s, d), dtype, n) for n in ("k", "v"))
    if score:  # the two score side outputs, declared like Stats (b, h, s, 1) fp32
        kw["sample_score_max"] = _desc((1, h, s, 1), torch.float32, "score_max")
        kw["sample_score_sum_exp"] = _desc((1, h, s, 1), torch.float32, "score_sum_exp")
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


def test_the_adapter_pool_is_the_rows_twin_on_sm100(monkeypatch):
    """The standalone adapter on a cc 10.0 device: D 8 / 24 / 32 dense land on the 32-wide tile with the record's d_flavor 32 and cga1; a
    cga2 request, split-KV and the score side outputs land on the native d64 flavor (its own legs, unlike the Rubin d64 tile's); D 40 / 64
    keep the native d64 flavor; the pre-tile declines (D % 8 != 0) are unchanged."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    for d in (8, 24, 32):
        api = _api(d)
        assert api.check_support(), d
        assert api.flavor == (32, 32), (d, api.flavor)
        params = api.template_params()
        assert params.d_flavor == 32 and params.cta_mma == 1, (d, params.d_flavor, params.cta_mma)
    for d in (40, 64):
        api = _api(d)
        assert api.check_support() and api.flavor == (64, 64) and api.template_params().d_flavor == 64, d
    api = _api(24, cga=2)
    assert api.check_support() and api.flavor == (64, 64) and api.template_params().cta_mma == 2 and api.template_params().d_flavor == 64
    api = _api(24, split_kv=2)
    assert api.check_support() and api.flavor == (64, 64) and api.template_params().d_flavor == 64
    # the score side outputs walk past the tile onto the native d64 flavor, which carries no score store on this arch line
    # (config_sm100.SM100_SCORE_STATS_SHAPES): the pre-tile typed decline, unchanged
    with pytest.raises(NotImplementedError, match=r"score_max .* select \(64, 64\)"):
        _api(24, score=True).check_support()
    with pytest.raises((ValueError, NotImplementedError)):
        _api(20).check_support()


# --------------------------------------------------------------------------- the GPU tier (an SM100 / SM103 device; the dev box is SHARED, never a number)

requires_sm100_line = pytest.mark.skipif(_SM not in (100, 103), reason="needs a cc 10.0 / 10.3 device, have " + ("none" if _SM is None else f"sm_{_SM}"))

_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE_IDS = ["fp16", "bf16"]


def _cases():
    from small_tile_cells import CASES

    return list(CASES)


@requires_sm100_line
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("d", [8, 24, 32])
@pytest.mark.parametrize("case", _cases())
def test_the_32_wide_tile_matches_the_fp64_oracle_on_sm100(case, d, dtype):
    """Every head dim the 32-wide tile serves (8 / 24 / 32 on the SM100 file's third geometry) over the degenerate-input matrix, both dtypes:
    the plan is the row's cga1 unsplit plan, keyless rows exactly 0 / -inf, two launches bitwise, live rows against the fp64 oracle."""
    from small_tile_cells import case_inputs, check, check_degenerate_rows, run_graph

    q, k, v, kw, oracle = case_inputs(case, d, dtype)
    out = run_graph(q, k, v, row=_SM100_ROW, **kw)
    assert out["knobs"].cga == 1 and out["knobs"].split_kv == 1, out["knobs"]
    check(out, oracle)
    check_degenerate_rows(out, case)


@requires_sm100_line
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("d", [24, 8])
def test_padded_head_dim_is_bitwise_the_zero_padded_32_wide_tile_on_sm100(d, dtype):
    """D = 24 / 8 on the 32-wide tile is BITWISE the explicitly zero-padded D = 32 run (the TMA zero fill IS the padding)."""
    from small_tile_cells import case_inputs, run_graph

    q, k, v, kw, _ = case_inputs("dense_nkv2", d, dtype)
    out = run_graph(q, k, v, row=_SM100_ROW, **kw)

    def pad(t):  # (b, h, s, d) view of a (b, s, h, d) buffer -> the same with d zero-padded to 32, same layout
        bshd = torch.nn.functional.pad(t.transpose(1, 2).contiguous(), (0, 32 - d))
        return bshd.transpose(1, 2)

    out_w = run_graph(pad(q), pad(k), pad(v), row=_SM100_ROW, **kw)
    assert torch.equal(out["o"], out_w["o"][..., :d]) and torch.equal(out["stats"], out_w["stats"])
