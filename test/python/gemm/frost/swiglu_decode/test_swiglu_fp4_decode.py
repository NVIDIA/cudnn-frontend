# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate / up product of an FP4 MLP for decode rows with SiLU(gate) x up and the FP4 re-quantisation in one launch
(``cudnn.SwigluFp4Decode``).  Host: the config's ledgers and derived constants (the stage rule's mirror, the shared-memory and
tensor-memory ledgers, the barrier counts), the token-tile rule and the form text that heads the kernel symbol.

Marked ``no_workspace_shim``: the engine suite's autouse fixture would otherwise import torch and both architecture compilers for
every test of this direct-call entry."""

from dataclasses import fields, replace

import pytest

cudnn = pytest.importorskip("cudnn")

from cudnn.gemm.frost.swiglu_decode.kernel import swiglu_fp4_decode_config as kcfg  # noqa: E402
from cudnn.gemm.frost.swiglu_decode.kernel.swiglu_fp4_decode_config import Cfg, Geometry  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.no_workspace_shim]

G_DEFAULT = Geometry(5120, 17408)
G_SMALL = Geometry(5120, 128)
TILES = (8, 16, 32, 64)
#: token tile -> (operand stages, staging stages, shared memory B, transaction bytes of one stage, tensor-memory columns used, epilogue tile)
FIGURES = {
    8: (5, 7, 231424, 43008, 80, 8),
    16: (4, 12, 231424, 45056, 96, 16),
    32: (4, 4, 231424, 49152, 128, 32),
    64: (3, 7, 230400, 57344, 192, 32),
}


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_geometry_validates_and_holds_plain_ints():
    g = Geometry(True + 5119, 17408)
    assert (g.HIDDEN, g.INTERMEDIATE, g.WEIGHT_ROWS, g.WEIGHT_TILES, g.K_BLOCKS) == (5120, 17408, 34816, 272, 10) and type(g.HIDDEN) is int
    assert g.describe() == "hidden 5120, intermediate 17408" and G_SMALL.WEIGHT_TILES == 2
    assert kcfg.DEFAULT_GEOMETRY == G_DEFAULT and kcfg.SERVED_GEOMETRIES == (G_DEFAULT, G_SMALL)
    for bad in ((0, 17408), (5120, -1), (5120.5, 17408), (True, 17408)):
        with pytest.raises(ValueError, match="positive int"):
            Geometry(*bad)
    with pytest.raises(TypeError, match="Geometry"):
        kcfg.check_geometry((5120, 17408))
    with pytest.raises(NotImplementedError, match="hidden 5120, intermediate 17408 or hidden 5120, intermediate 128"):
        kcfg.check_geometry(Geometry(4096, 17408))


@pytest.mark.parametrize("readings", [(228 * 1024, 227 * 1024, 148), (233472, 232448, 212)], ids=["sm100-reference", "modelled-212-sm-part"])
def test_config_figures_of_record(readings):
    """The stage rule's mirror per token tile.  Its capacity is a constant, so the figures are the same on every part's readings."""
    assert kcfg.STAGE_RULE_CAPACITY_BYTES == 232448 and kcfg.K_TILE == 512 and kcfg.TMEM_ALLOC_COLS == 512
    part = kcfg.part_readings(*readings)
    for tile, (operand, staging, smem, tx, cols, epi) in FIGURES.items():
        cfg = Cfg(TOKEN_TILE=tile, **part)
        kcfg.validate_cfg(cfg)
        assert kcfg.stages(cfg) == (operand, staging), tile
        assert (kcfg.smem_bytes(cfg), kcfg.tx_bytes(cfg), kcfg.tmem_cols_used(cfg), kcfg.epilogue_tile(cfg)) == (smem, tx, cols, epi), tile
        table = kcfg.smem_table(cfg)
        assert [name for name, _, _ in table] == [
            "ab_full_mbar",
            "ab_empty_mbar",
            "acc_full_mbar",
            "acc_empty_mbar",
            "tmem_dealloc_mbar",
            "tmem_holding_buf",
            "sC",
            "sA",
            "sB",
            "sSFA",
            "sSFB",
        ]
        assert all(offset % 1024 == 0 for name, offset, _ in table if name.startswith("s")), "every ring starts on its alignment"
        assert all(offset + size <= 262144 for _, offset, size in table), "every ring root stays below the 14-bit descriptor start"
        assert dict((name, size) for name, _, size in table)["sA"] == operand * 32768
        assert kcfg.grid(cfg) == (1, 1, 148) and kcfg.grid(replace(cfg, GEOM=G_SMALL)) == (1, 1, 2)
        assert kcfg.grid(replace(cfg, MAX_CLUSTERS=37)) == (1, 1, 37) and kcfg.grid(replace(cfg, MAX_CLUSTERS=1000)) == (1, 1, 272)


def test_barrier_ledger_constants():
    """SUM(issuing lanes) == init for every mbarrier ring; every named barrier counts exactly the threads that reach it."""
    ledger = kcfg.barrier_ledger()
    assert {name: row["init"] for name, row in ledger.items()} == dict(
        ab_full=1, ab_empty=1, acc_full=1, acc_empty=4, cta_sync=192, tmem_ptr_sync=160, epilog_sync=128
    )
    for name, row in ledger.items():
        assert row["issuing_lanes"] == row["init"], name
    assert (kcfg.THREADS_PER_CTA, kcfg.TMEM_PTR_SYNC_THREADS, kcfg.EPILOGUE_THREADS) == (192, 160, 128)
    assert (kcfg.CTA_SYNC_BARRIER, kcfg.EPILOG_SYNC_BARRIER, kcfg.TMEM_PTR_SYNC_BARRIER) == (0, 1, 2)
    assert kcfg.EPILOGUE_WARPS == (0, 1, 2, 3) and (kcfg.MMA_WARP, kcfg.TMA_WARP) == (4, 5)


def test_token_tile_is_the_narrowest_that_holds_the_rows():
    want = {rows: next(t for t in TILES if rows <= t) for rows in range(1, 65)}
    assert {rows: kcfg.token_tile_for(rows) for rows in range(1, 65)} == want
    assert (want[8], want[9], want[16], want[17], want[32], want[33], want[64]) == (8, 16, 16, 32, 32, 64, 64)
    for bad in (0, -1, 1.0, True, "4", None):
        with pytest.raises(ValueError, match="rows must be a positive int"):
            kcfg.token_tile_for(bad)
    with pytest.raises(NotImplementedError, match="at most 64 rows per call"):
        kcfg.token_tile_for(65)


def test_token_tile_floor_and_one_tile_per_call():
    assert [kcfg.token_tile_for(rows, 32) for rows in (1, 8, 9, 32, 33, 64)] == [32, 32, 32, 32, 64, 64]
    assert [kcfg.token_tile_for(rows, 64) for rows in (1, 64)] == [64, 64]
    with pytest.raises(ValueError, match="Cfg.TOKEN_TILE must be one of"):
        kcfg.token_tile_for(3, 12)
    form = kcfg.form_of(Cfg(), 8)
    assert kcfg.check_rows_fit(form, 8) == 8
    with pytest.raises(ValueError, match="rows 9 exceed the token tile 8; the kernel runs one token tile per call"):
        kcfg.check_rows_fit(form, 9)
    assert kcfg.form_of(Cfg(PREWAIT_STAGES=0, MAX_CLUSTERS=37), 32) == Cfg(TOKEN_TILE=32, PREWAIT_STAGES=0, MAX_CLUSTERS=37)


def test_cfg_has_no_numerics_field():
    """Every field of the config is a performance knob, a part reading or the geometry: nothing that could change an output byte."""
    names = [f.name for f in fields(Cfg)]
    assert names == [*kcfg.PERFORMANCE_FIELDS, *kcfg.READING_FIELDS, "GEOM"]
    assert kcfg.PERFORMANCE_FIELDS == ("TOKEN_TILE", "PREWAIT_STAGES", "L2_EVICT_FIRST", "MAX_CLUSTERS", "PDL")
    assert Cfg() == Cfg(8, 2, True, 148, False, 227 * 1024, 228 * 1024, 148, G_DEFAULT)


@pytest.mark.parametrize(
    "bad, exc, match",
    [
        (dict(TOKEN_TILE=12), ValueError, "Cfg.TOKEN_TILE must be one of"),
        (dict(TOKEN_TILE=True), ValueError, "Cfg.TOKEN_TILE must be one of"),
        (dict(PREWAIT_STAGES=1), ValueError, "Cfg.PREWAIT_STAGES must be one of"),
        (dict(PREWAIT_STAGES=4), ValueError, "Cfg.PREWAIT_STAGES must be one of"),
        (dict(L2_EVICT_FIRST=1), ValueError, "Cfg.L2_EVICT_FIRST must be a bool"),
        (dict(PDL=None), ValueError, "Cfg.PDL must be a bool"),
        (dict(MAX_CLUSTERS=0), ValueError, "Cfg.MAX_CLUSTERS must be an int >= 1"),
        (dict(MAX_CLUSTERS=1.5), ValueError, "Cfg.MAX_CLUSTERS must be an int >= 1"),
        (dict(SM_COUNT=0), ValueError, "sm_count must be a positive int"),
        (dict(SMEM_OPTIN_BYTES=300000), ValueError, "cannot exceed the SM's"),
        (dict(SMEM_OPTIN_BYTES=200000), ValueError, "holds 231424 B of shared memory per thread block; the part's per-block maximum is 200000 B"),
        (dict(GEOM=(5120, 17408)), TypeError, "Cfg.GEOM must be"),
        (dict(GEOM=Geometry(5120, 100)), ValueError, "tiles of 128 rows"),
    ],
)
def test_cfg_knobs_validate_by_name(bad, exc, match):
    with pytest.raises(exc, match=match):
        kcfg.validate_cfg(replace(Cfg(), **bad))
    with pytest.raises(TypeError, match="cfg must be"):
        kcfg.validate_cfg(dict(TOKEN_TILE=8))


def test_cfg_accepts_every_knob_value_of_its_domain():
    for tile in TILES:
        for prewait in (0, 2):
            for evict in (True, False):
                for clusters in (1, 37, 148, 212, 424):
                    for pdl in (False, True):
                        kcfg.validate_cfg(Cfg(TOKEN_TILE=tile, PREWAIT_STAGES=prewait, L2_EVICT_FIRST=evict, MAX_CLUSTERS=clusters, PDL=pdl))


def test_form_text_names_every_performance_field():
    """``str(cfg)`` heads the kernel symbol: two forms never share it, and it is short enough to sit in a symbol's first 100 characters
    behind the kernel's name."""
    assert str(Cfg()) == "h5120_i17408_t8_pw2_evf_c148_nopdl"
    assert str(Cfg(TOKEN_TILE=64, PREWAIT_STAGES=0, L2_EVICT_FIRST=False, MAX_CLUSTERS=212, PDL=True, GEOM=G_SMALL)) == "h5120_i128_t64_pw0_dflt_c212_pdl"
    base = Cfg()
    texts = {str(base)}
    for change in (dict(TOKEN_TILE=16), dict(PREWAIT_STAGES=0), dict(L2_EVICT_FIRST=False), dict(MAX_CLUSTERS=147), dict(PDL=True), dict(GEOM=G_SMALL)):
        texts.add(str(replace(base, **change)))
    assert len(texts) == 7, "every performance field and the geometry change the text"
    assert str(replace(base, SM_COUNT=212, SMEM_OPTIN_BYTES=232448, SMEM_PER_SM_BYTES=233472)) == str(base), "a part reading is not a form"
    longest = str(Cfg(TOKEN_TILE=64, MAX_CLUSTERS=99999999, GEOM=G_DEFAULT))
    assert len("cudnn_kernel_frost_swiglu_fp4_decode_") + len(longest) <= 100
