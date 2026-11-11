# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP8 split-K projection for decode rows (``cudnn.Fp8ProjectionDecode``).  Host: the config's ledgers and derived constants (the
shared-memory mirror, the barrier counts, the tensor-memory ledger), the measured pick and its structural separation from the numerics,
the typed declines of the host entry (CPU tensors), the DSL gate, the plan-time compile key and the lazy exports.  On a cc 10.x device:
the entry against an fp64 reference of the E4M3 operands under a per-element bound, exact-integer cells byte for byte against the host,
the edge rows (zeros, NaN, overflow), the output rows it must not touch, every performance arm bitwise with the default, a row alone
against the same row in a batch, CUDA-graph replay, the warm call (a cache hit that reads nothing to the host and allocates nothing),
the launch matrix in a child process with a stage-aware deadline, and a warm stress with byte references.
The operand builders, the reference and the device gate live in ``fp8_projection_decode_utils.py`` beside this module."""

import inspect
import subprocess
import sys
import textwrap
from dataclasses import fields, replace

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from fp8_projection_decode_utils import (  # noqa: E402  (the operand builders, the reference and the device gate)
    CLASSES,
    E4M3,
    G_IN,
    G_OUT,
    MAX_CODE,
    NAN_CODE,
    SM100_READING,
    assert_same_bytes,
    cid,
    device_or_skip,
    error_over_bound,
    from_codes,
    integer_operands,
    make_operands,
    new_out,
    raw,
)

from cudnn.gemm.frost.projection_decode import fp8_projection_decode as entry  # noqa: E402
from cudnn.gemm.frost.projection_decode.kernel import fp8_projection_decode_config as kcfg  # noqa: E402

# ``no_workspace_shim``: this directory's conftest wraps the GEMM engine's MoE plans for every test unless a test opts out, which would
# import the engine's compilers (and the DSL) into tests of an entry that never touches the engine.
pytestmark = [pytest.mark.L0, pytest.mark.no_workspace_shim]

Form, Numerics, Geometry = kcfg.Form, kcfg.Numerics, kcfg.Geometry
C_IN4, C_IN2, C_OUT2 = CLASSES


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_geometry_validates_and_holds_plain_ints():
    g = Geometry(True + 4607, 8192)
    assert (g.OUT_FEATURES, g.IN_FEATURES, g.K_TILES) == (4608, 8192, 64) and type(g.OUT_FEATURES) is int
    assert g.describe() == "out features 4608, in features 8192" and G_OUT.K_TILES == 16
    for bad in ((0, 8192), (4608, -1), (4608.5, 8192), (True, 8192)):
        with pytest.raises(ValueError, match="positive int"):
            Geometry(*bad)


def test_split_k_for_is_a_function_of_the_geometry_alone():
    assert kcfg.split_k_for(G_IN) == 4 and kcfg.split_k_for(G_OUT) == 2
    assert list(inspect.signature(kcfg.split_k_for).parameters) == ["geometry"], "the default split may read nothing but the geometry"
    assert kcfg.split_k_for(Geometry(64, 128)) == 1  # one K tile: no slice of eight


# (class, first row, last row) -> (MMA_M, STAGES, token tile, TMEM_COLS, footprint B, CTAs, CTAs per SM); measured on compute capability 10.3
PICK_TABLE = {
    (C_IN4, 1, 4): (64, 11, 8, 32, 107904, 288, 2),
    (C_IN4, 5, 8): (64, 7, 8, 32, 71040, 288, 3),
    (C_IN4, 9, 16): (64, 7, 16, 32, 84352, 288, 2),
    (C_IN4, 17, 32): (64, 7, 32, 32, 110976, 288, 2),
    (C_IN2, 1, 8): (64, 8, 8, 32, 76160, 144, 3),
    (C_IN2, 9, 16): (64, 8, 16, 32, 86400, 144, 2),
    (C_IN2, 17, 32): (64, 8, 32, 32, 106880, 144, 2),
    (C_OUT2, 1, 8): (64, 8, 8, 32, 76160, 256, 3),
    (C_OUT2, 9, 16): (64, 8, 16, 32, 86400, 256, 2),
    (C_OUT2, 17, 32): (64, 8, 32, 32, 106880, 256, 2),
}


def test_pick_is_the_measured_table_at_one_wave():
    """Every served class x rows 1 .. 32 on the cc 10.3 readings: the form, its footprint in bytes, the grid and the co-residency."""
    seen = 0
    for (numerics, lo, hi), want in PICK_TABLE.items():
        for rows in range(lo, hi + 1):
            form = kcfg.pick(numerics, rows, **SM100_READING)
            cfg = kcfg.resolve_form(
                numerics, rows, form, kcfg.part_readings(SM100_READING["smem_per_sm"], SM100_READING["smem_optin"], SM100_READING["sm_count"])
            )
            kcfg.validate_cfg(cfg)
            got = (cfg.MMA_M, cfg.STAGES, cfg.TOKEN_TILE, cfg.TMEM_COLS, cfg.SMEM_BYTES, cfg.CTAS, cfg.SMEM_RESIDENT)
            assert got == want, (cid(numerics), rows, got)
            assert cfg.RUNNING == cfg.SMEM_RESIDENT and cfg.WAVES == 1 and cfg.SMEM_RESIDENT * cfg.TMEM_COLS <= 512
            seen += 1
    assert seen == 3 * 32


@pytest.mark.parametrize("sm_count", (148, 212))
def test_pick_model_keeps_one_wave_on_other_sm_counts(sm_count):
    """The co-residency MODEL (slots = SMs x CTAs per SM) on the other SM100-class SM counts: the same forms, still one wave.  A slot count,
    not a placement: where a cluster of slices may sit on a part is not modelled here."""
    for numerics in CLASSES:
        for rows in range(1, 33):
            form = kcfg.pick(numerics, rows, sm_count, SM100_READING["smem_per_sm"], SM100_READING["smem_optin"])
            cfg = kcfg.resolve_form(numerics, rows, form, kcfg.part_readings(SM100_READING["smem_per_sm"], SM100_READING["smem_optin"], sm_count))
            assert cfg.WAVES == 1 and form == kcfg.pick(numerics, rows, **SM100_READING), (cid(numerics), rows)


def test_pick_lowers_the_stages_on_a_part_that_cannot_hold_them():
    """A modelled part with 100 KB per SM and per CTA: the pick walks the ring depth down until the grid fits one wave or one stage is
    left; a part whose opt-in cannot hold one stage is the typed decline of a part."""
    small = dict(sm_count=152, smem_per_sm=100 * 1024, smem_optin=99 * 1024)
    form = kcfg.pick(C_IN4, 1, **small)
    cfg = kcfg.resolve_form(C_IN4, 1, form, kcfg.part_readings(small["smem_per_sm"], small["smem_optin"], small["sm_count"]))
    assert form.STAGES < 11 and cfg.SMEM_BYTES <= small["smem_optin"] and cfg.WAVES == 1 and cfg.SMEM_RESIDENT == 2
    form = kcfg.pick(C_IN4, 32, **small)
    cfg = kcfg.resolve_form(C_IN4, 32, form, kcfg.part_readings(small["smem_per_sm"], small["smem_optin"], small["sm_count"]))
    assert form.STAGES < 7 and cfg.WAVES == 1
    narrow = dict(sm_count=152, smem_per_sm=60 * 1024, smem_optin=59 * 1024)  # a 32-row CTA never shares an SM here: one stage, two waves, still served
    form = kcfg.pick(C_IN4, 32, **narrow)
    cfg = kcfg.resolve_form(C_IN4, 32, form, kcfg.part_readings(narrow["smem_per_sm"], narrow["smem_optin"], narrow["sm_count"]))
    assert form.STAGES == 1 and cfg.SMEM_RESIDENT == 1 and cfg.WAVES == 2
    with pytest.raises(NotImplementedError, match="opt-in"):
        kcfg.pick(C_IN4, 32, sm_count=152, smem_per_sm=16 * 1024, smem_optin=15 * 1024)


def test_pick_returns_a_form_and_a_form_holds_no_numerics():
    """The structural separation: the pick's return type has no field of the numerics, so no pick can change the split."""
    form_fields, numerics_fields = {f.name for f in fields(Form)}, {f.name for f in fields(Numerics)}
    assert not form_fields & numerics_fields and "SPLIT_K" not in form_fields and "GEOM" not in form_fields
    assert isinstance(kcfg.pick(C_IN4, 1, **SM100_READING), Form)
    for numerics in CLASSES:
        k = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K)
        assert {k.cfg_for(rows).SPLIT_K for rows in range(1, 33)} == {numerics.SPLIT_K}
        assert {k.cfg_for(rows).NUMERICS for rows in range(1, 33)} == {numerics}
        given = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K, form=Form(MMA_M=128, STAGES=2))
        assert {given.cfg_for(rows).SPLIT_K for rows in range(1, 33)} == {numerics.SPLIT_K}


def test_smem_mirror_and_barrier_constants():
    """The shared-memory mirror in the kernel's allocation order, the barrier init counts (SUM(issuing lanes) == init) and the reduction's
    byte count."""
    assert kcfg.smem_bytes(64, 8, 4, 7) == 71040 and kcfg.smem_bytes(64, 8, 4, 11) == 107904 and kcfg.smem_bytes(64, 16, 4, 11) == 125312
    assert kcfg.smem_bytes(64, 8, 1, 16) == kcfg._align_up(64 * 128 * 16 + 8 * 128 * 16 + 2 * 16 * 8 + 2 * 8 + 4, 128)
    # the walk, spelled out for one form: weight ring | activation ring | 2 x STAGES + 2 barriers | base pointer | mailbox | its barrier
    size = 64 * 128 * 7 + 8 * 128 * 7  # 57,344 + 7,168
    size += 2 * 7 * 8 + 2 * 8  # the two stage arrays and the two single barriers
    size = kcfg._align_up(size + 4, 128)  # the tensor-memory base pointer, then the mailbox's alignment
    size += 3 * 64 * 8 * 4 + 8  # three peer tiles of fp32 and the reduction barrier
    assert kcfg._align_up(size, 128) == 71040
    assert (kcfg.FULL_BAR_ARRIVERS, kcfg.EMPTY_BAR_ARRIVERS, kcfg.MMA_DONE_BAR_ARRIVERS, kcfg.REDUCE_BAR_ARRIVERS) == (2, 1, 1, 1)
    assert kcfg.TMEM_BAR_ARRIVERS == 32 + 128 == kcfg.MMA_WARP_LANES + kcfg.EPILOGUE_THREADS
    assert kcfg.MMAS_PER_K_TILE == 4 and kcfg.K_TILE == 128 and kcfg.THREADS_PER_CTA == 256
    for numerics in CLASSES:
        for tile in kcfg.TOKEN_TILES:
            cfg = kcfg.Cfg(GEOM=numerics.GEOM, SPLIT_K=numerics.SPLIT_K, ROWS=tile, TOKEN_TILE=tile)
            assert cfg.VALUES_PER_THREAD == 64 * tile // 128 and cfg.VALUES_PER_THREAD % 4 == 0
            assert cfg.REDUCE_TX_BYTES == (numerics.SPLIT_K - 1) * 128 * cfg.VALUES_PER_THREAD * 4  # 16 B per remote store, every peer thread
            assert cfg.WEIGHT_STAGE_BYTES + cfg.ACTIVATION_STAGE_BYTES == (64 + tile) * 128  # the two expects of one full barrier
            assert cfg.K_TILES_PER_SLICE * numerics.SPLIT_K == numerics.GEOM.K_TILES


def test_tensor_memory_ledger_declines_a_parked_form_by_name():
    """What the launcher places (shared memory alone) times the columns a CTA allocates must fit an SM's 512: three CTAs of 71,040 B with
    256 columns each do not, and the validator says so; 64 and 128 columns fit beside three, 256 beside two."""
    part = kcfg.part_readings(SM100_READING["smem_per_sm"], SM100_READING["smem_optin"], SM100_READING["sm_count"])

    def cfg(rows, cols, stages=7):
        return kcfg.resolve_form(C_IN4, rows, Form(MMA_M=64, STAGES=stages, TMEM_COLS=cols), part)

    parked = cfg(1, 256)
    assert parked.SMEM_RESIDENT == 3 and parked.RUNNING == 2 and parked.SMEM_RESIDENT * parked.TMEM_COLS == 768
    with pytest.raises(ValueError, match="tensor-memory ledger"):
        kcfg.validate_cfg(parked)
    for cols in (32, 64, 128):
        kcfg.validate_cfg(cfg(1, cols))
    kcfg.validate_cfg(cfg(9, 256))  # token tile 16: 84,352 B, two per SM
    kcfg.validate_cfg(cfg(1, 256, stages=11))  # 107,904 B, two per SM
    with pytest.raises(ValueError, match="the form cannot be served"):
        entry.Fp8ProjectionDecode("cpu", G_IN, form=Form(TMEM_COLS=256))
    assert entry.Fp8ProjectionDecode("cpu", G_IN, form=Form(TMEM_COLS=128)).cfg_for(1).TMEM_COLS == 128
    # the other side of the ledger: a one-stage CTA is so small that shared memory alone would place 18 on an SM at split 2 -- more than
    # the 16 that hold 32 columns each; the count is the shared-memory bound (an upper bound of what the launcher places)
    tiny = kcfg.resolve_form(C_OUT2, 1, Form(MMA_M=64, STAGES=1), part)
    assert (tiny.SMEM_BYTES, tiny.SMEM_RESIDENT) == (11520, 18)
    with pytest.raises(ValueError, match=r"18 CTAs of 11520 B .*576 > 512"):
        kcfg.validate_cfg(tiny)
    kcfg.validate_cfg(kcfg.resolve_form(C_IN4, 1, Form(MMA_M=64, STAGES=1), part))  # 15,616 B: 14 per SM, 448 columns
    kcfg.validate_cfg(kcfg.resolve_form(C_OUT2, 1, Form(MMA_M=64, STAGES=2), part))


def _cfg(**kw):
    base = kcfg.Cfg(GEOM=G_IN, SPLIT_K=4, ROWS=1)
    return replace(base, **kw)


@pytest.mark.parametrize(
    "overrides, token",
    [
        (dict(MMA_M=32), "MMA_M"),
        (dict(GEOM=Geometry(4600, 8192), MMA_M=64), "MMA_M"),
        (dict(GEOM=Geometry(4608, 8200)), "multiple of 128"),
        (dict(SPLIT_K=3), "split_k"),
        (dict(ROWS=0), "rows"),
        (dict(ROWS=33, TOKEN_TILE=32), "rows"),
        (dict(ROWS=9, TOKEN_TILE=8), "token tile"),
        (dict(TOKEN_TILE=12), "token tile"),
        (dict(STAGES=0), "STAGES"),
        (dict(STAGES=17), "17 stages exceed the 16 K tiles"),
        (dict(TMEM_COLS=48), "TMEM_COLS"),
        (dict(TMEM_COLS=16), "TMEM_COLS"),
        (dict(PDL=1), "PDL"),
        (dict(HOLD_TMEM=None), "HOLD_TMEM"),
        (dict(STAGES=16, ROWS=17, TOKEN_TILE=32, SMEM_OPTIN_BYTES=100 * 1024), "opt-in"),
        (dict(SMEM_OPTIN_BYTES=0), "positive int"),
        (dict(SMEM_PER_SM_BYTES=1024, SMEM_OPTIN_BYTES=2048), "cannot exceed"),
    ],
)
def test_cfg_knobs_validate_by_name(overrides, token):
    with pytest.raises(ValueError, match=token):
        kcfg.validate_cfg(_cfg(**overrides))


def test_token_tile_and_the_32_row_ceiling():
    assert [kcfg.token_tile_for(r) for r in (1, 8, 9, 16, 17, 32)] == [8, 8, 16, 16, 32, 32]
    assert kcfg.token_tile_for(3, floor=16) == 16 and kcfg.token_tile_for(20, floor=16) == 32 and kcfg.tmem_cols_for(32) == 32
    for bad in (0, -1, 1.0, True):
        with pytest.raises(ValueError, match="positive int"):
            kcfg.token_tile_for(bad)
    with pytest.raises(NotImplementedError, match="at most 32 rows"):
        kcfg.token_tile_for(33)
    with pytest.raises(ValueError, match="TOKEN_TILE"):
        kcfg.token_tile_for(1, floor=12)


def test_served_classes_decline_every_other_pair_by_name():
    assert kcfg.check_numerics(G_IN) == C_IN4 and kcfg.check_numerics(G_IN, 2) == C_IN2 and kcfg.check_numerics(G_OUT) == C_OUT2
    for geometry, split in ((G_IN, 1), (G_OUT, 1), (G_OUT, 4)):  # legal cuts of the K tiles, never validated
        with pytest.raises(NotImplementedError, match="validated"):
            kcfg.check_numerics(geometry, split)
    with pytest.raises(ValueError, match="split_k must be one of"):
        kcfg.check_numerics(G_IN, 3)
    with pytest.raises(NotImplementedError, match="out features 5120, in features 6144"):
        kcfg.check_numerics(Geometry(5120, 6144))
    with pytest.raises(TypeError, match="Geometry"):
        kcfg.check_numerics((4608, 8192))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the entry's typed declines (CPU tensors)
# ---------------------------------------------------------------------------------------------------------------------------
def _host_ops(geometry=G_OUT, rows=3):
    ops = make_operands(geometry, rows)
    ops["out"] = torch.zeros(rows, geometry.OUT_FEATURES, dtype=torch.bfloat16)
    return ops


def call_check(ops, device_cc=(10, 3), **kw):
    return entry.check_support(ops["weight"], ops["activation"], ops["scale"], ops["out"], device_cc=device_cc, **kw)


def test_served_operands_pass_the_host_checks_at_both_geometries():
    assert call_check(_host_ops(G_OUT, 3)) == 3 and call_check(_host_ops(G_OUT, 32)) == 32
    assert call_check(_host_ops(G_IN, 1), geometry=G_IN) == 1
    zero_dim = _host_ops()
    zero_dim["scale"] = zero_dim["scale"].reshape(())
    assert call_check(zero_dim) == 3


@pytest.mark.parametrize(
    "mutation, exc, token",
    [
        (lambda o: o.update(weight=o["weight"].view(torch.uint8)), TypeError, "weight must be"),
        (lambda o: o.update(activation=o["activation"].view(torch.uint8)), TypeError, "activation must be"),
        (lambda o: o.update(out=o["out"].float()), TypeError, "out must be"),
        (lambda o: o.update(scale=o["scale"].double()), TypeError, "scale must be"),
        (lambda o: o.update(scale=1.0), TypeError, "scale must be a torch.Tensor"),
        (lambda o: o.update(weight=None), TypeError, "weight must be a torch.Tensor"),
        (lambda o: o.update(activation=o["activation"][:, :-128]), ValueError, "activation must be"),
        (lambda o: o.update(activation=o["activation"][0]), ValueError, "activation must be"),
        (lambda o: o.update(activation=o["activation"][:0], out=o["out"][:0]), ValueError, "rows must be"),
        (lambda o: o.update(out=o["out"][:2]), ValueError, "out must be exactly"),
        (lambda o: o.update(out=torch.zeros(4, 8192, dtype=torch.bfloat16)), ValueError, "out must be exactly"),
        (lambda o: o.update(out=torch.zeros(3, 4096, dtype=torch.bfloat16)), ValueError, "out must be exactly"),
        (lambda o: o.update(scale=torch.ones(2)), ValueError, "one fp32 element"),
        (lambda o: o.update(out=torch.zeros(8192, 3, dtype=torch.bfloat16).t()), NotImplementedError, "contiguous"),
        (lambda o: o.update(weight=from_codes(torch.zeros(8192 * 2048 + 1, dtype=torch.uint8))[1:].view(8192, 2048)), ValueError, "32-byte aligned"),
        (lambda o: o.update(out=torch.zeros(3 * 8192 + 1, dtype=torch.bfloat16)[1:].view(3, 8192)), ValueError, "32-byte aligned"),
    ],
)
def test_typed_declines_name_the_operand(mutation, exc, token):
    ops = _host_ops()
    mutation(ops)
    with pytest.raises(exc, match=token):
        call_check(ops)


def test_declines_a_non_contiguous_weight_and_activation_without_a_copy():
    ops = _host_ops()
    wide = from_codes(torch.zeros(3, 2 * 2048, dtype=torch.uint8))
    ops["activation"] = wide[:, ::2]
    with pytest.raises(NotImplementedError, match="contiguous operands; activation"):
        call_check(ops)
    ops = _host_ops()
    ops["weight"] = from_codes(torch.zeros(2048, 8192, dtype=torch.uint8)).t()
    with pytest.raises(NotImplementedError, match="contiguous operands; weight"):
        call_check(ops)


def test_declines_more_than_32_rows_and_an_unserved_geometry_by_name():
    with pytest.raises(NotImplementedError, match="at most 32 rows"):
        call_check(_host_ops(G_OUT, 33))
    ops = _host_ops()
    ops["weight"] = from_codes(torch.zeros(4096, 2048, dtype=torch.uint8))
    with pytest.raises(NotImplementedError, match="out features 4096, in features 2048"):
        call_check(ops)
    with pytest.raises(ValueError, match=r"weight must be \[4608, 8192\]"):
        call_check(_host_ops(), geometry=G_IN)


def test_declines_an_unserved_compute_capability_by_name():
    ops = _host_ops()
    for cc in ((9, 0), (12, 0), (8, 0), (11, 0)):
        with pytest.raises(NotImplementedError, match=rf"10\.x.*{cc[0]}\.{cc[1]}"):
            call_check(ops, device_cc=cc)
    for cc in ((10, 0), (10, 3), (10, 7)):
        assert call_check(ops, device_cc=cc) == 3
    assert not entry.cc_supported(None) and entry.cc_supported((10, 3))


def test_declines_cpu_operands_when_no_capability_is_given():
    with pytest.raises(NotImplementedError, match="CUDA tensors only"):
        call_check(_host_ops(), device_cc=None)


def test_binding_check_requires_every_operand_on_the_given_device():
    ops = _host_ops()
    with pytest.raises(ValueError, match="the entry's"):
        entry.check_binding(ops["weight"], ops["activation"], ops["scale"], ops["out"], device="cuda:0" if torch.cuda.is_available() else "meta")


def test_host_entry_keeps_the_reference_readings_and_a_given_form_overrides_the_pick():
    k = entry.Fp8ProjectionDecode("cpu", G_IN)
    assert k.split_k == 4 and k.numerics == C_IN4 and k.form is None and k.device_cc is None
    assert not any(c.HOLD_TMEM for c in k.cfgs.values()) and entry.Fp8ProjectionDecode("cpu", G_IN, form=Form(HOLD_TMEM=True)).cfg_for(9).HOLD_TMEM
    assert k.cfg_for(9).FORM == Form(MMA_M=64, STAGES=7, TMEM_COLS=32, TOKEN_TILE=16, PREWAIT_WEIGHTS=True, HOLD_TMEM=False)
    assert k.readings == dict(SMEM_PER_SM_BYTES=233472, SMEM_OPTIN_BYTES=232448, SM_COUNT=152)
    assert (k.cfg_for(1).STAGES, k.cfg_for(4).STAGES, k.cfg_for(5).STAGES, k.cfg_for(32).STAGES) == (11, 11, 7, 7)
    assert sorted(k.cfgs) == list(range(1, 33)), "every served row count is resolved and validated at construction"
    given = entry.Fp8ProjectionDecode("cpu", G_IN, split_k=2, form=Form(MMA_M=128, STAGES=3, TOKEN_TILE=16))
    assert given.split_k == 2 and {c.MMA_M for c in given.cfgs.values()} == {128} and {c.STAGES for c in given.cfgs.values()} == {3}
    assert (given.cfg_for(1).TOKEN_TILE, given.cfg_for(16).TOKEN_TILE, given.cfg_for(17).TOKEN_TILE) == (16, 16, 32)  # the tile is a floor
    with pytest.raises(TypeError, match="Form"):
        entry.Fp8ProjectionDecode("cpu", G_IN, form=dict(STAGES=7))
    with pytest.raises(ValueError, match="17 stages exceed"):
        entry.Fp8ProjectionDecode("cpu", G_IN, form=Form(STAGES=17))
    with pytest.raises(NotImplementedError, match="validated"):
        entry.Fp8ProjectionDecode("cpu", G_OUT, split_k=4)
    with pytest.raises(NotImplementedError, match="CUDA devices only"):
        k.compile(1)
    with pytest.raises(NotImplementedError, match="at most 32 rows"):
        k.compile(33)


def test_pdl_launch_knobs_default_off_and_the_release_follows_pdl():
    k = entry.Fp8ProjectionDecode("cpu", G_OUT)
    assert (k.pdl, k.pdl_release, k.cfg_for(1).PDL, k.cfg_for(1).PDL_RELEASE) == (False, False, False, False)
    k = entry.Fp8ProjectionDecode("cpu", G_OUT, pdl=True)
    assert (k.cfg_for(8).PDL, k.cfg_for(8).PDL_RELEASE) == (True, True)
    k = entry.Fp8ProjectionDecode("cpu", G_OUT, pdl=True, pdl_release=False)
    assert (k.cfg_for(8).PDL, k.cfg_for(8).PDL_RELEASE) == (True, False)
    k = entry.Fp8ProjectionDecode("cpu", G_OUT, pdl_release=True)
    assert (k.cfg_for(8).PDL, k.cfg_for(8).PDL_RELEASE) == (False, True)


def test_compile_key_is_plan_time_only():
    """One key per row count (a declared shape), stable across calls, the split and the launch knobs in it; nothing of a call's data."""
    k = entry.Fp8ProjectionDecode("cpu", G_IN)
    assert k.key(3) == k.key(3) and k.key(3) != k.key(4) and len({k.key(r) for r in range(1, 33)}) == 32
    assert k.key(3) == (k.cfg_for(3),) and k.key(3)[0].ROWS == 3 and k.key(3)[0].SPLIT_K == 4
    other = entry.Fp8ProjectionDecode("cpu", G_IN, split_k=2)
    assert other.key(3) != k.key(3) and entry.Fp8ProjectionDecode("cpu", G_IN, pdl=True).key(3) != k.key(3)
    hash(k.key(3))


def test_dsl_gate_names_the_version_and_the_missing_layer(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    assert "4.8.0" in entry.dsl_error() and "4.7.1" in entry.dsl_error()
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    assert "not installed" in entry.dsl_error()
    # a DSL that clears the floor but does not provide the layer the kernel is written on: a decline that names the missing piece
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "9.9.9")))
    monkeypatch.setattr(entry, "DSL_LAYER", (("cutlass_layer_that_does_not_exist", ("x",)),))
    assert "cutlass_layer_that_does_not_exist" in entry.dsl_error() and "9.9.9" in entry.dsl_error()
    monkeypatch.setattr(entry, "DSL_LAYER", (("os", ("path", "no_such_name_in_this_layer")),))
    assert "no_such_name_in_this_layer" in entry.dsl_error() and "does not export" in entry.dsl_error()
    monkeypatch.setattr(entry, "DSL_LAYER", (("os", ("path",)),))
    assert entry.dsl_error() is None


def test_installed_dsl_provides_the_layer_or_declines_by_name():
    """On the installed DSL the gate either passes (every probed name imports) or returns a message; it never raises."""
    msg = entry.dsl_error()
    assert msg is None or isinstance(msg, str)
    names = [n for _, ns in entry.DSL_LAYER for n in ns]
    assert len(names) == len(set(names)) and "kernel" in names and "tma_load" in names


def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.Fp8ProjectionDecode is entry.Fp8ProjectionDecode
    gemm, pkg = importlib.import_module("cudnn.gemm"), importlib.import_module("cudnn.gemm.frost.projection_decode")
    assert gemm.Fp8ProjectionDecode is entry.Fp8ProjectionDecode and gemm.ProjectionGeometry is Geometry and gemm.ProjectionForm is Form
    assert pkg.Fp8ProjectionDecode is entry.Fp8ProjectionDecode and pkg.Geometry is Geometry and pkg.Form is Form
    assert "Fp8ProjectionDecode" in cudnn._LAZY_OPTIONAL_IMPORTS and "Fp8ProjectionDecode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS
    assert {"Fp8ProjectionDecode", "ProjectionGeometry", "ProjectionForm"} <= set(gemm.__all__)


def test_import_cudnn_does_not_import_the_kernel_or_torch():
    """The lazy boundary in fresh interpreters: importing cudnn loads nothing of this package; building a host entry and running its
    declines loads neither the kernel module nor the DSL."""
    code = "import sys, cudnn; print(sorted(m for m in sys.modules if 'projection_decode' in m or m in ('cutlass', 'torch')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
    code = textwrap.dedent("""
        import sys, cudnn
        from cudnn.gemm import ProjectionGeometry
        k = cudnn.Fp8ProjectionDecode("cpu", ProjectionGeometry(8192, 2048))
        try:
            k.compile(1)
        except NotImplementedError:
            pass
        print(sorted(m for m in sys.modules if m == "cutlass" or m.endswith("fp8_projection_decode_e4m3")))
        """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out


# ---------------------------------------------------------------------------------------------------------------------------
# device (compute capability 10.x)
# ---------------------------------------------------------------------------------------------------------------------------
_ENTRIES = {}
_OPS = {}


def _entry(dev, numerics, **kw):
    key = (str(dev), numerics, tuple(sorted(kw.items(), key=lambda kv: kv[0])))
    if key not in _ENTRIES:
        _ENTRIES[key] = entry.Fp8ProjectionDecode(dev, numerics.GEOM, split_k=numerics.SPLIT_K, **kw)
    return _ENTRIES[key]


def _ops(dev, geometry, kind="random", seed=0, integer=False):
    """32-row operands of a geometry on the device, built once per (kind, seed); a call uses the first ``rows`` rows."""
    key = (str(dev), geometry, kind, seed, integer)
    if key not in _OPS:
        _OPS[key] = integer_operands(geometry, 32, dev, seed) if integer else make_operands(geometry, 32, dev, seed, kind)
    return _OPS[key]


def _run(k, ops, rows, out=None):
    out = new_out(k.geometry, rows, ops["weight"].device) if out is None else out
    got = k(ops["weight"], ops["activation"][:rows], ops["scale"], out)
    assert got is out
    return out


@pytest.mark.parametrize("kind", ("random", "wide", "structured"))
@pytest.mark.parametrize("rows", (1, 2, 3, 8, 9, 16, 17, 24, 32))
@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_matches_the_fp64_reference_within_the_bound(numerics, rows, kind):
    dev = device_or_skip()
    ops = _ops(dev, numerics.GEOM, kind)
    out = _run(_entry(dev, numerics), ops, rows)
    torch.cuda.synchronize()
    assert out.shape == (rows, numerics.GEOM.OUT_FEATURES) and out.dtype == torch.bfloat16
    sub = dict(ops, activation=ops["activation"][:rows])
    ratio, bad = error_over_bound(out, sub, numerics.SPLIT_K)
    print(f"{cid(numerics)} rows {rows} {kind}: max |out - ref| / bound = {ratio:.4f}; non-finite mismatches {bad}")
    assert bad == 0, f"{bad} elements are non-finite where the reference is finite (or the reverse)"
    assert ratio <= 1.0, f"max |out - ref| / bound = {ratio:.4f} > 1"


@pytest.mark.parametrize("rows", (1, 8, 9, 17, 32))
@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_exact_integer_operands_write_the_host_bytes(numerics, rows):
    """Integer operands with a power-of-two scale: every partial sum is exact in fp32 in any order, so the bytes are the host's -- one cast,
    round to nearest even, no double rounding, no dropped low-order partial.  Zero tolerance."""
    dev = device_or_skip()
    ops = _ops(dev, numerics.GEOM, integer=True)
    out = _run(_entry(dev, numerics), ops, rows)
    torch.cuda.synchronize()
    x, w = ops["activation"][:rows].float().double(), ops["weight"].float().double()
    exact = x @ w.t()
    assert float(exact.abs().max()) < 2**24
    want = (exact * ops["scale"].double().reshape(())).float().to(torch.bfloat16)
    assert_same_bytes(out, want, f"{cid(numerics)} rows {rows}")


@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_cast_rounds_ties_to_even(numerics):
    """Sums on bf16 ties: 514 lies midway between 512 and 516 and 518 between 516 and 520; nearest-even writes 512 and 520 (truncation
    would write 512 and 516), mirrored for the negative sums."""
    dev = device_or_skip()
    g = numerics.GEOM
    w = torch.zeros(g.OUT_FEATURES, g.IN_FEATURES)
    x = torch.zeros(2, g.IN_FEATURES)
    x[0, :8], x[0, 8] = 8.0, 2.0  # 8 x 8 x 8 = 512, plus 2 x {1, 3}
    w[0, :8], w[0, 8] = 8.0, 1.0  # 514
    w[1, :8], w[1, 8] = 8.0, 3.0  # 518
    w[2], w[3] = -w[0], -w[1]
    x[1] = -x[0]
    ops = dict(weight=w.to(dev).to(E4M3), activation=x.to(dev).to(E4M3), scale=torch.ones(1, dtype=torch.float32, device=dev))
    out = _run(_entry(dev, numerics), ops, 2).float().cpu()
    assert out[0, :4].tolist() == [512.0, 520.0, -512.0, -520.0] and out[1, :4].tolist() == [-512.0, -520.0, 512.0, 520.0]
    assert int((out[:, 4:] != 0).sum()) == 0


@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_edge_rows_zero_nan_and_overflow(numerics):
    """Zero rows write zeros; a NaN code in ONE activation row makes that row NaN and leaves every other row's bytes as in the clean run; a
    NaN code in one weight row makes that output column NaN; a sum past the bf16 range is an infinity of the sum's sign."""
    dev = device_or_skip()
    g = numerics.GEOM
    k = _entry(dev, numerics)
    base = _ops(dev, g, "random")
    clean = _run(k, base, 9).clone()
    codes = base["activation"][:9].view(torch.uint8).clone()
    codes[2] = 0  # +0 everywhere
    codes[5] = 0x80  # -0 everywhere
    codes[7, 4097 % g.IN_FEATURES] = NAN_CODE
    ops = dict(base, activation=from_codes(codes))
    out = _run(k, ops, 9)
    torch.cuda.synchronize()
    assert int((out[2].float() != 0).sum()) == 0 and int((out[5].float() != 0).sum()) == 0, "zero rows"
    assert bool(torch.isnan(out[7].float()).all()), "the row holding a NaN code"
    for t in (0, 1, 3, 4, 6, 8):
        assert_same_bytes(out[t], clean[t], f"row {t} beside a NaN row")
    wcodes = base["weight"].view(torch.uint8).clone()
    wcodes[11, 3] = NAN_CODE | 0x80
    out = _run(k, dict(base, weight=from_codes(wcodes)), 9)
    torch.cuda.synchronize()
    assert bool(torch.isnan(out[:, 11].float()).all()) and int(torch.isnan(out.float()).sum()) == 9, "the output column of a NaN weight row"
    # overflow: the largest code everywhere with one sign pattern, a scale that takes the product past the bf16 range
    big_w = torch.full((g.OUT_FEATURES, g.IN_FEATURES), MAX_CODE, dtype=torch.uint8)
    big_w[1::2] |= 0x80
    big_x = torch.full((1, g.IN_FEATURES), MAX_CODE, dtype=torch.uint8)
    ops = dict(weight=from_codes(big_w).to(dev), activation=from_codes(big_x).to(dev), scale=torch.tensor([1.0e30], dtype=torch.float32, device=dev))
    out = _run(k, ops, 1).float()
    assert bool(torch.isinf(out).all()) and bool((out[0, 0::2] > 0).all()) and bool((out[0, 1::2] < 0).all())


@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_rows_past_the_call_and_the_inputs_stay_untouched(numerics):
    """``out`` as the first rows of a taller buffer: the rows past the call keep their sentinel; the operands hold their bytes."""
    dev = device_or_skip()
    g = numerics.GEOM
    k = _entry(dev, numerics)
    ops = _ops(dev, g, "random")
    before = {name: t.clone() for name, t in ops.items()}
    tall = torch.full((40, g.OUT_FEATURES), 7.0, dtype=torch.bfloat16, device=dev)
    for rows in (1, 5, 9, 17, 32):
        tall.fill_(7.0)
        _run(k, ops, rows, out=tall[:rows])
        torch.cuda.synchronize()
        assert int((tall[rows:] != 7.0).sum()) == 0, f"rows past {rows} were written"
        assert_same_bytes(tall[:rows], _run(k, ops, rows), f"rows {rows} into a taller buffer")
    for name, t in ops.items():
        assert torch.equal(t.view(torch.uint8), before[name].view(torch.uint8)), f"{name} was modified"


FORM_ARMS = [
    dict(form=Form(MMA_M=64, STAGES=2)),
    dict(form=Form(MMA_M=64, STAGES=3)),
    dict(form=Form(MMA_M=64, STAGES=7)),
    dict(form=Form(MMA_M=64, STAGES=8)),
    dict(form=Form(MMA_M=128, STAGES=4)),
    dict(form=Form(MMA_M=64, STAGES=4, TMEM_COLS=64)),
    dict(form=Form(MMA_M=64, STAGES=7, TMEM_COLS=128)),
    dict(form=Form(MMA_M=64, STAGES=8, TOKEN_TILE=16)),
    dict(form=Form(MMA_M=64, STAGES=6, TOKEN_TILE=32)),
    dict(form=Form(MMA_M=64, STAGES=4, PREWAIT_WEIGHTS=False), pdl=True),
    dict(form=Form(MMA_M=64, STAGES=7, HOLD_TMEM=True)),
    dict(form=Form(MMA_M=64, STAGES=7, HOLD_TMEM=True), pdl=True),
    dict(pdl=True),
    dict(pdl=True, pdl_release=False),
    dict(pdl_release=True),
]


@pytest.mark.parametrize("arm", FORM_ARMS, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()).replace("Form", ""))
@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_performance_arms_write_the_default_bytes(numerics, arm):
    """No performance knob changes a byte: ring depths from two stages up (two to nine CTAs per SM), the wider weight tile, more tensor-memory
    columns, a token tile wider than the rows need, the weight loader waiting too, the tensor memory held until its registers are
    consumed, the dependent-launch attribute and the release."""
    dev = device_or_skip()
    ops = _ops(dev, numerics.GEOM, "wide")
    default, other = _entry(dev, numerics), _entry(dev, numerics, **arm)
    for rows in (1, 4, 5, 8, 9, 16, 17, 32):
        want = _run(default, ops, rows)
        torch.cuda.synchronize()
        got = _run(other, ops, rows)
        torch.cuda.synchronize()
        assert_same_bytes(got, want, f"{cid(numerics)} rows {rows} arm {arm}")


@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_a_row_alone_writes_the_bytes_it_writes_in_a_batch(numerics):
    """Batch invariance at a fixed split, for EVERY served row count (each is its own rendering) and so across the three token tiles: row
    ``t`` of an R-row call equals the same row launched alone, R = 2 .. 32."""
    dev = device_or_skip()
    k = _entry(dev, numerics)
    ops = _ops(dev, numerics.GEOM, "wide")
    alone = torch.empty(32, numerics.GEOM.OUT_FEATURES, dtype=torch.bfloat16, device=dev)
    for t in range(32):
        k(ops["weight"], ops["activation"][t : t + 1], ops["scale"], alone[t : t + 1])
    torch.cuda.synchronize()
    for rows in range(2, 33):
        out = _run(k, ops, rows)
        torch.cuda.synchronize()
        assert_same_bytes(out, alone[:rows], f"{cid(numerics)}: {rows} rows against each row alone")
    assert len(k.compiled) == 32, "every served row count was compiled and launched"
    perm = torch.arange(31, -1, -1, device=dev)
    reversed_ops = dict(ops, activation=ops["activation"][perm].contiguous())
    assert_same_bytes(_run(k, reversed_ops, 32), alone[perm], f"{cid(numerics)}: the reversed batch")


def test_split_is_a_byte_class_and_each_class_replays():
    """Another split is another fp32 association: both splits of one geometry stay inside the reference bound and each replays bitwise;
    the number of elements on which they differ is printed, never asserted."""
    dev = device_or_skip()
    ops = _ops(dev, G_IN, "wide")
    outs = {}
    for numerics in (C_IN4, C_IN2):
        k = _entry(dev, numerics)
        outs[numerics] = _run(k, ops, 24)
        assert_same_bytes(_run(k, ops, 24), outs[numerics], f"{cid(numerics)} replay")
        ratio, bad = error_over_bound(outs[numerics], dict(ops, activation=ops["activation"][:24]), numerics.SPLIT_K)
        assert ratio <= 1.0 and bad == 0
    differing = int((raw(outs[C_IN4]) != raw(outs[C_IN2])).sum())
    print(f"split 4 against split 2 at 24 rows: {differing} of {outs[C_IN4].numel()} elements differ")


def test_graph_capture_replays_bitwise_with_eager():
    dev = device_or_skip()
    layers = [(_entry(dev, numerics), _ops(dev, numerics.GEOM, "random", seed=7), rows) for numerics, rows in ((C_IN4, 6), (C_OUT2, 6), (C_IN2, 24))]
    eager = [_run(k, ops, rows).clone() for k, ops, rows in layers]  # compile + warm
    outs = [new_out(k.geometry, rows, dev, fill=0.0) for k, _, rows in layers]
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (k, ops, rows), o in zip(layers, outs):
            _run(k, ops, rows, out=o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for (k, ops, rows), o in zip(layers, outs):
            _run(k, ops, rows, out=o)
    for _ in range(3):
        for o in outs:
            o.zero_()
        g.replay()
        torch.cuda.synchronize()
        for e, o in zip(eager, outs):
            assert_same_bytes(o, e, "graph replay against eager")
    # the graph binds the operands' storage: new bytes in the same activation tensor are what the next replay computes
    k, ops, rows = layers[0]
    saved = ops["activation"][:rows].clone()
    try:
        ops["activation"][:rows].copy_(_ops(dev, k.geometry, "wide")["activation"][:rows])
        want = _run(k, ops, rows).clone()
        g.replay()
        torch.cuda.synchronize()
        assert_same_bytes(outs[0], want, "graph replay after the activation's bytes changed")
    finally:
        ops["activation"][:rows].copy_(saved)


def test_warm_call_is_a_cache_hit_that_reads_nothing_and_allocates_nothing(monkeypatch):
    """After ``compile(rows)`` a launch is a guaranteed hit: with compilation forbidden, device-to-host reads raising and the allocator
    watched, three warm launches of a compiled row count run and write the cold bytes."""
    dev = device_or_skip()
    k = entry.Fp8ProjectionDecode(dev, G_OUT)
    ops = _ops(dev, G_OUT, "random")
    k.compile(5)
    assert len(k.compiled) == 1 and k.cfg_for(5) in k.compiled
    out = new_out(G_OUT, 5, dev)
    want = _run(k, ops, 5).clone()
    torch.cuda.synchronize()
    mod = k._kernel_module

    def forbidden(cfg):
        raise AssertionError(f"compiled on the execute path: {cfg.describe()}")

    monkeypatch.setattr(mod, "compile_kernel", forbidden)
    act = ops["activation"][:5]
    allocated = torch.cuda.memory_allocated(dev)
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(ops["weight"], act, ops["scale"], out)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(dev) == allocated, "a warm launch allocated device memory"
    assert_same_bytes(out, want, "warm launch")
    with pytest.raises(AssertionError, match="compiled on the execute path"):
        _run(k, ops, 6)  # a row count that was never compiled reaches the compiler: the plan-time site is compile()


def test_compile_without_rows_covers_every_served_row_count(monkeypatch):
    dev = device_or_skip()
    k = entry.Fp8ProjectionDecode(dev, G_OUT)
    k.compile()
    assert len(k.compiled) == 32 and all(k.cfg_for(r) in k.compiled for r in range(1, 33))
    monkeypatch.setattr(k._kernel_module, "compile_kernel", lambda cfg: (_ for _ in ()).throw(AssertionError("compiled twice")))
    k.compile()
    ops = _ops(dev, G_OUT, "random")
    for rows in (1, 13, 32):
        _run(k, ops, rows)
    torch.cuda.synchronize()


def test_thirty_three_rows_is_a_typed_decline_never_a_provider_switch():
    dev = device_or_skip()
    k = _entry(dev, C_OUT2)
    ops = make_operands(G_OUT, 33, dev)
    with pytest.raises(NotImplementedError, match="at most 32 rows"):
        k(ops["weight"], ops["activation"], ops["scale"], new_out(G_OUT, 33, dev))
    with pytest.raises(NotImplementedError, match="at most 32 rows"):
        k.compile(33)


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    """The compile key pins the plan-time facts only, so a warm entry would launch on any tensor: the host declines first."""
    dev = device_or_skip()
    k = _entry(dev, C_OUT2)
    ops = _ops(dev, G_OUT, "random")
    out = _run(k, ops, 3)
    torch.cuda.synchronize()
    act = ops["activation"][:3]
    with pytest.raises(TypeError, match="activation must be"):
        k(ops["weight"], act.view(torch.uint8), ops["scale"], out)
    with pytest.raises(TypeError, match="out must be"):
        k(ops["weight"], act, ops["scale"], out.float())
    with pytest.raises(ValueError, match="out must be exactly"):
        k(ops["weight"], act, ops["scale"], new_out(G_OUT, 4, dev))
    with pytest.raises(ValueError, match="one fp32 element"):
        k(ops["weight"], act, torch.ones(2, device=dev), out)
    with pytest.raises(ValueError, match="the entry's"):
        k(ops["weight"].cpu(), act, ops["scale"], out)
    with pytest.raises(ValueError, match=r"weight must be \[8192, 2048\]"):
        k(_ops(dev, G_IN, "random")["weight"], act, ops["scale"], out)
    with pytest.raises(NotImplementedError, match="contiguous"):
        k(ops["weight"], act, ops["scale"], torch.empty(G_OUT.OUT_FEATURES, 3, dtype=torch.bfloat16, device=dev).t())


def test_reservation_covers_the_kernel_allocation(capfd):
    """The launch reserves the config's mirror of the kernel's shared-memory allocation.  The DSL compares that reservation with the
    kernel's own total at every launch and prints when it is short; that message must never appear for a served form."""
    dev = device_or_skip()
    capfd.readouterr()
    for numerics in CLASSES:
        k = _entry(dev, numerics)
        ops = _ops(dev, numerics.GEOM, "random")
        for rows in (1, 4, 5, 8, 9, 16, 17, 32):
            _run(k, ops, rows)
        for arm in (dict(form=Form(MMA_M=128, STAGES=4)), dict(form=Form(MMA_M=64, STAGES=2))):
            _run(_entry(dev, numerics, **arm), ops, 9)
    torch.cuda.synchronize()
    captured = capfd.readouterr()
    assert "may exceed" not in captured.out + captured.err, (captured.out + captured.err)[-2000:]


# The launch matrix in a child interpreter.  The child prints a stage line before every phase and arms its own traceback dump ahead of the
# parent's deadline; the parent reads the LAST stage line to tell a compile stall (a setup budget) from a dispatched kernel that did not
# return (a hang: audit the barrier table of the kernel module -- the limit below is never raised).
CHILD_COMPILE_BUDGET_S = 420  # a cold DSL import plus 54 trace-compiles
CHILD_DISPATCH_LIMIT_S = 60  # 54 forms x 16 warm launches each take milliseconds
_CHILD = """
import faulthandler, sys, time
faulthandler.dump_traceback_later({dump_s}, exit=False)
print("STAGE import", flush=True)
import torch
from cudnn.gemm.frost.projection_decode import fp8_projection_decode as entry
from cudnn.gemm.frost.projection_decode.kernel import fp8_projection_decode_config as kcfg
dev = torch.device("cuda", torch.cuda.current_device())
print("STAGE compile", flush=True)
cells = []
for numerics in kcfg.SERVED_CLASSES:
    g = numerics.GEOM
    w = torch.zeros(g.OUT_FEATURES, g.IN_FEATURES, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    x = torch.zeros(32, g.IN_FEATURES, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    s = torch.ones(1, dtype=torch.float32, device=dev)
    for pdl in (False, True):
        k = entry.Fp8ProjectionDecode(dev, g, split_k=numerics.SPLIT_K, pdl=pdl)
        for rows in (1, 2, 4, 5, 8, 9, 16, 17, 32):
            k.compile(rows)
            cells.append((k, w, x[:rows], s, torch.empty(rows, g.OUT_FEATURES, dtype=torch.bfloat16, device=dev)))
torch.cuda.synchronize()
print("STAGE dispatch", len(cells), "forms", flush=True)
t0 = time.monotonic()
for k, w, x, s, o in cells:
    for _ in range(16):  # back to back: under pdl each launch is the next one's predecessor, released after its last weight issue
        k(w, x, s, o)
torch.cuda.synchronize()
print("STAGE done", round(time.monotonic() - t0, 3), "s", flush=True)
"""


def test_launch_matrix_completes_in_a_child_process():
    device_or_skip()
    limit = CHILD_COMPILE_BUDGET_S + CHILD_DISPATCH_LIMIT_S
    code = _CHILD.format(dump_s=limit - 15)
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        last = [line for line in out.splitlines() if line.startswith("STAGE")][-1:] or ["no stage line"]
        if last[0].startswith("STAGE dispatch"):
            pytest.fail(f"a dispatched kernel did not return: audit the barrier table of the kernel module (never raise this limit)\n{out}\n{err[-4000:]}")
        pytest.fail(f"the child ran out of its setup budget before any kernel was dispatched (last line: {last[0]})\n{out}\n{err[-4000:]}")
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr[-4000:]}"
    assert "STAGE done" in proc.stdout, proc.stdout
    assert "may exceed" not in proc.stdout + proc.stderr, "the launch reserved less shared memory than the kernel allocates"


@pytest.mark.parametrize("rows", (1, 32))
@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_warm_launches_keep_writing_the_cold_bytes(numerics, rows):
    """A stale read across a ring refill shows on WARM launches only, so two launches prove nothing: 2,000 back-to-back launches that
    alternate two operand sets into a ring of output buffers, every output compared byte for byte with its cold reference -- per block,
    after the block's launches, so that no compare sits between two launches; once plainly and once behind another kernel."""
    dev = device_or_skip()
    k = _entry(dev, numerics)
    sets = [_ops(dev, numerics.GEOM, "wide"), _ops(dev, numerics.GEOM, "random")]
    cold = []
    for ops in sets:
        torch.cuda.synchronize()
        cold.append(_run(k, ops, rows).clone())
    torch.cuda.synchronize()
    ring = [new_out(numerics.GEOM, rows, dev) for _ in range(16)]
    filler = torch.ones(1 << 20, device=dev)
    wrong = launches = 0
    for behind_another_kernel in (False, True):
        for block in range(1000 // 16 + 1):
            for i, o in enumerate(ring):
                if behind_another_kernel:
                    filler.mul_(1.0)
                ops = sets[(block + i) % 2]
                k(ops["weight"], ops["activation"][:rows], ops["scale"], o)
            torch.cuda.synchronize()
            for i, o in enumerate(ring):
                wrong += int(bool((raw(o) != raw(cold[(block + i) % 2])).any()))
            launches += len(ring)
    assert launches >= 2000
    assert wrong == 0, f"{wrong} of {launches} warm launches wrote other bytes than the cold launch"
