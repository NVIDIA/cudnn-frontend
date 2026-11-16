# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP8 split-K projection for decode rows (``cudnn.Fp8ProjectionDecode``).  Host: the config's ledgers and derived constants (the
shared-memory mirror, the barrier counts, the tensor-memory ledger), the measured pick and its structural separation from the numerics,
the pick of the classes run as written (their table rows on every part, the late tensor-memory release, the wave counts as numbers),
the splits each geometry is served at, the text a rendering carries in its kernel symbol, the typed declines of the host entry (CPU
tensors), the DSL gate, the plan-time compile key and the lazy exports.  On a cc 10.x device:
the entry against an fp64 reference of the E4M3 operands under a per-element bound, exact-integer cells byte for byte against the host,
the edge rows (zeros, NaN, overflow), the output rows it must not touch, every performance arm bitwise with the default, a row alone
against the same row in a batch, CUDA-graph replay, the warm call (a cache hit that reads nothing to the host and allocates nothing),
the launch matrix in child processes with a stage-aware deadline (one per group of classes; both tensor-memory release orders of the
classes run as written), a warm stress with byte references, and the warm stress behind this GPU's positive control for the classes
run as written.  A child interpreter runs the source overlay of this process and reports the files it loaded; the parent compares
them with its own.
The operand builders, the reference and the device gate live in ``fp8_projection_decode_utils.py`` beside this module; replay, batch
invariance per row count, the pick's byte neutrality and the goldens live in ``test_fp8_projection_decode_determinism.py``."""

import ast
import importlib
import inspect
import json
import subprocess
import sys
import textwrap
import types
import warnings
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from fp8_projection_decode_utils import (  # noqa: E402  (the operand builders, the reference and the device gate)
    CLASSES,
    E4M3,
    G_5K,
    G_14K,
    G_16K,
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
from launch_records import assert_facts_equal_the_launch  # noqa: E402

# ``no_workspace_shim``: this directory's conftest wraps the GEMM engine's MoE plans for every test unless a test opts out, which would
# import the engine's compilers (and the DSL) into tests of an entry that never touches the engine.
pytestmark = [pytest.mark.L0, pytest.mark.no_workspace_shim]

Form, Numerics, Geometry = kcfg.Form, kcfg.Numerics, kcfg.Geometry
C_IN4, C_IN2, C_OUT2 = CLASSES[:3]  # the one-wave classes (a measured pick each)
C_5K4, C_5K2, C_16K1, C_14K1 = CLASSES[3:]  # the classes run as written
AS_WRITTEN = CLASSES[3:]
REPO = Path(__file__).resolve().parents[5]  # <repo>/test/python/gemm/frost/projection_decode/<this file>


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
    # the default follows the K tiles alone, so it is not a served class at every geometry: an explicit split_k builds those entries
    assert [kcfg.split_k_for(g) for g in (G_5K, G_16K, G_14K)] == [4, 4, 4]


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
    for numerics in (c for c in CLASSES if c in kcfg.ONE_WAVE_CLASSES):
        for rows in range(1, 33):
            form = kcfg.pick(numerics, rows, sm_count, SM100_READING["smem_per_sm"], SM100_READING["smem_optin"])
            cfg = kcfg.resolve_form(numerics, rows, form, kcfg.part_readings(SM100_READING["smem_per_sm"], SM100_READING["smem_optin"], sm_count))
            assert cfg.WAVES == 1 and form == kcfg.pick(numerics, rows, **SM100_READING), (cid(numerics), rows)


# (class, first row, last row) -> (MMA_M, STAGES, token tile, TMEM_COLS, footprint B, CTAs, CTAs per SM, waves on 148 / 152 / 212 SMs).
# The rows of the classes run as written: not measured with this entry; the wave counts are the slot-count model of the config.
AS_WRITTEN_TABLE = {
    (C_5K4, 1, 8): (64, 7, 8, 32, 71040, 320, 3, (1, 1, 1)),
    (C_5K4, 9, 16): (64, 7, 16, 32, 84352, 320, 2, (2, 2, 1)),
    (C_5K4, 17, 32): (64, 7, 32, 32, 110976, 320, 2, (2, 2, 1)),
    (C_5K2, 1, 8): (64, 11, 8, 32, 103808, 160, 2, (1, 1, 1)),
    (C_5K2, 9, 16): (64, 11, 16, 32, 117120, 160, 1, (2, 2, 1)),
    (C_5K2, 17, 32): (64, 11, 32, 32, 143744, 160, 1, (2, 2, 1)),
    (C_16K1, 1, 8): (128, 8, 8, 32, 139520, 128, 1, (1, 1, 1)),
    (C_16K1, 9, 16): (128, 8, 16, 32, 147712, 128, 1, (1, 1, 1)),
    (C_16K1, 17, 32): (128, 8, 32, 32, 164096, 128, 1, (1, 1, 1)),
    (C_14K1, 1, 8): (128, 8, 8, 32, 139520, 112, 1, (1, 1, 1)),
    (C_14K1, 9, 16): (128, 8, 16, 32, 147712, 112, 1, (1, 1, 1)),
    (C_14K1, 17, 32): (128, 8, 32, 32, 164096, 112, 1, (1, 1, 1)),
}


def test_pick_of_the_classes_run_as_written_is_the_table_row_on_every_part():
    """Every class outside ``ONE_WAVE_CLASSES`` x rows 1 .. 32: the pick is the table row on 64, 148, 152 and 212 SMs alike (the form
    does not follow the SM count), with 32 tensor-memory columns and the late tensor-memory release; its footprint, grid, co-residency
    and the number of waves the slot-count model gives on three SM counts are pinned as numbers -- two waves where the model says two."""
    assert kcfg.ONE_WAVE_CLASSES == {C_IN4, C_IN2, C_OUT2} and kcfg.HOLD_TMEM_CLASSES == set(AS_WRITTEN)
    assert kcfg.ONE_WAVE_CLASSES | kcfg.HOLD_TMEM_CLASSES == set(kcfg.SERVED_CLASSES) and set(kcfg.PICKS) == set(kcfg.SERVED_CLASSES)
    per_sm, optin = SM100_READING["smem_per_sm"], SM100_READING["smem_optin"]
    seen = 0
    for (numerics, lo, hi), want in AS_WRITTEN_TABLE.items():
        for rows in range(lo, hi + 1):
            forms = {sm_count: kcfg.pick(numerics, rows, sm_count, per_sm, optin) for sm_count in (64, 148, 152, 212)}
            assert len(set(forms.values())) == 1, (cid(numerics), rows, forms)
            form = forms[152]
            assert form == Form(MMA_M=want[0], STAGES=want[1], TMEM_COLS=want[3], HOLD_TMEM=True), (cid(numerics), rows, form)
            waves = []
            for sm_count in (148, 152, 212):
                cfg = kcfg.resolve_form(numerics, rows, form, kcfg.part_readings(per_sm, optin, sm_count))
                kcfg.validate_cfg(cfg)
                waves.append(cfg.WAVES)
            got = (cfg.MMA_M, cfg.STAGES, cfg.TOKEN_TILE, cfg.TMEM_COLS, cfg.SMEM_BYTES, cfg.CTAS, cfg.SMEM_RESIDENT, tuple(waves))
            assert got == want, (cid(numerics), rows, got)
            assert cfg.RUNNING == cfg.SMEM_RESIDENT and cfg.SMEM_RESIDENT * cfg.TMEM_COLS <= 512 and cfg.STAGES < cfg.K_TILES_PER_SLICE
            seen += 1
    assert seen == len(AS_WRITTEN) * 32
    # the one-wave classes keep the early release in their pick: the same Form as before the other classes were served
    assert all(not kcfg.pick(numerics, rows, **SM100_READING).HOLD_TMEM for numerics in kcfg.ONE_WAVE_CLASSES for rows in range(1, 33))


def test_pick_of_a_class_run_as_written_is_lowered_only_to_fit_the_part():
    """A modelled part with 100 KB per SM and a 99 KB per-CTA opt-in: a row that fits is returned as written although its grid takes
    more than one wave; a row that does not fit is lowered until the footprint fits, never further; a part whose opt-in cannot hold
    one stage is the typed decline of a part."""
    small = dict(sm_count=152, smem_per_sm=100 * 1024, smem_optin=99 * 1024)
    part = kcfg.part_readings(small["smem_per_sm"], small["smem_optin"], small["sm_count"])
    form = kcfg.pick(C_5K4, 1, **small)  # 71,040 B fits: seven stages, one CTA per SM here, three waves
    cfg = kcfg.resolve_form(C_5K4, 1, form, part)
    assert (form.STAGES, cfg.SMEM_RESIDENT, cfg.WAVES) == (7, 1, 3) and form.HOLD_TMEM
    for numerics, rows in ((C_5K4, 32), (C_5K2, 1), (C_16K1, 1), (C_14K1, 32)):
        form = kcfg.pick(numerics, rows, **small)
        cfg = kcfg.resolve_form(numerics, rows, form, part)
        deeper = kcfg.resolve_form(numerics, rows, replace(form, STAGES=form.STAGES + 1), part)
        assert form.STAGES < kcfg.pick(numerics, rows, **SM100_READING).STAGES and form.HOLD_TMEM, (cid(numerics), rows)
        assert cfg.SMEM_BYTES <= small["smem_optin"] < deeper.SMEM_BYTES, (cid(numerics), rows, cfg.SMEM_BYTES, deeper.SMEM_BYTES)
        kcfg.validate_cfg(cfg)
    with pytest.raises(NotImplementedError, match="opt-in"):
        kcfg.pick(C_16K1, 1, sm_count=152, smem_per_sm=16 * 1024, smem_optin=15 * 1024)


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


def test_tensor_memory_barrier_phases_are_ordered_by_a_wait():
    """``bar_tmem_alloc`` is ONE barrier with two phases: phase 0 publishes the tensor-memory base pointer, phase 1 releases the columns.
    A parity-1 wait issued while phase 0 is still open returns at once, so nothing but a wait may order the two phases: the MMA warp
    waits for phase 0 in front of its phase-1 arrive (arrive, wait 0, arrive, wait 1, deallocate), and an epilogue thread waits for
    phase 0 behind its entry arrive and in front of its one release arrive (one site per ``HOLD_TMEM`` arm).  Pinned in the source of
    the two roles: a role that arrives twice on this barrier without a phase-0 wait between the two arrives fails here."""
    from cudnn.gemm.frost.projection_decode.kernel import fp8_projection_decode_e4m3 as kernel

    def trace(role):
        fn = next(n for n in ast.walk(ast.parse(Path(kernel.__file__).read_text())) if isinstance(n, ast.FunctionDef) and n.name == role)
        ops = []
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name, args = node.func.attr, node.args
            on_barrier = bool(args) and isinstance(args[0], ast.Name) and args[0].id == "bar_tmem_alloc"
            if name == "mbarrier_arrive" and on_barrier:
                ops.append((node.lineno, "arrive"))
            elif name == "mbarrier_wait" and on_barrier:
                assert isinstance(args[1], ast.Constant), f"{role}: the phase of a wait on this barrier is a literal"
                ops.append((node.lineno, f"wait {args[1].value}"))
            elif name == "dealloc_tmem":
                ops.append((node.lineno, "deallocate"))
        return [op for _, op in sorted(ops)]

    assert trace("mma_warp") == ["arrive", "wait 0", "arrive", "wait 1", "deallocate"]
    # the epilogue: the entry arrive, the phase-0 wait, then its release arrive -- written once per ``HOLD_TMEM`` arm, one compiled
    assert trace("epilog_warp") == ["arrive", "wait 0", "arrive", "arrive"]


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


def test_given_forms_of_the_classes_run_as_written_meet_their_bounds_by_name():
    """The bounds a given ``form`` meets at the four classes run as written, on the reference readings.  The ring holds at most the K
    tiles of ONE slice -- 12 at ``(5120, 6144)`` split 4, 24 at split 2, 40 at the two split-1 classes -- and one stage more is a
    ``ValueError`` naming both counts.  The tensor-memory ledger at split 1, where no reduction mailbox pads a small CTA: a one-stage
    ring of the 64-row weight tile would sit 22 to an SM and is declined, the same ring of the 128-row tile (12 to an SM) and two
    stages are served.  256 tensor-memory columns fit where the table row places at most two CTAs on an SM (the split-1 classes and
    ``(5120, 6144)`` split 2) and are declined where it places three (``(5120, 6144)`` split 4 up to 8 rows).  And a given form is
    taken as given: it releases the tensor memory early unless it says ``HOLD_TMEM``, with which it is the pick's own rendering."""
    part = kcfg.part_readings(SM100_READING["smem_per_sm"], SM100_READING["smem_optin"], SM100_READING["sm_count"])
    for numerics, slice_tiles in zip(AS_WRITTEN, (12, 24, 40, 40)):
        assert numerics.GEOM.K_TILES // numerics.SPLIT_K == slice_tiles
        geometry, split = numerics.GEOM, numerics.SPLIT_K
        with pytest.raises(
            ValueError, match=rf"{slice_tiles + 1} stages exceed the {slice_tiles} K tiles of one slice of K {geometry.IN_FEATURES} at split_k {split}"
        ):
            entry.Fp8ProjectionDecode("cpu", geometry, split_k=split, form=Form(STAGES=slice_tiles + 1))
        mma_m, stages = kcfg.PICKS[numerics][0][2:]
        pick = entry.Fp8ProjectionDecode("cpu", geometry, split_k=split)
        given = entry.Fp8ProjectionDecode("cpu", geometry, split_k=split, form=Form(MMA_M=mma_m, STAGES=stages))
        held = entry.Fp8ProjectionDecode("cpu", geometry, split_k=split, form=Form(MMA_M=mma_m, STAGES=stages, HOLD_TMEM=True))
        for rows in range(1, kcfg.MAX_ROWS + 1):
            assert pick.cfg_for(rows).HOLD_TMEM and not given.cfg_for(rows).HOLD_TMEM, (cid(numerics), rows)
            assert held.key(rows) == pick.key(rows) != given.key(rows), (cid(numerics), rows)
    for numerics in (C_16K1, C_14K1):
        tiny = kcfg.resolve_form(numerics, 1, Form(MMA_M=64, STAGES=1), part)
        assert (tiny.SMEM_BYTES, tiny.SMEM_RESIDENT, tiny.MAILBOX_ELEMENTS) == (9344, 22, 0)
        with pytest.raises(ValueError, match=r"22 CTAs of 9344 B .*704 > 512"):
            kcfg.validate_cfg(tiny)
        with pytest.raises(ValueError, match="the form cannot be served at 1 rows"):
            entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=1, form=Form(MMA_M=64, STAGES=1))
        for form, resident in ((Form(MMA_M=128, STAGES=1), 12), (Form(MMA_M=64, STAGES=2), 11), (Form(MMA_M=128, STAGES=2), 6)):
            cfg = kcfg.resolve_form(numerics, 1, form, part)
            kcfg.validate_cfg(cfg)
            assert cfg.SMEM_RESIDENT == resident, (cid(numerics), form)
    for numerics, resident in ((C_5K2, (2, 1, 1)), (C_16K1, (1, 1, 1)), (C_14K1, (1, 1, 1))):
        mma_m, stages = kcfg.PICKS[numerics][0][2:]
        wide = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K, form=Form(MMA_M=mma_m, STAGES=stages, TMEM_COLS=256))
        assert tuple(wide.cfg_for(rows).SMEM_RESIDENT for rows in (1, 9, 17)) == resident and {c.TMEM_COLS for c in wide.cfgs.values()} == {256}
    with pytest.raises(ValueError, match=r"cannot be served at 1 rows .*3 CTAs of 71040 B .*768 > 512"):
        entry.Fp8ProjectionDecode("cpu", G_5K, form=Form(MMA_M=64, STAGES=7, TMEM_COLS=256))


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
    with pytest.raises(NotImplementedError, match="out features 5120, in features 4096"):
        kcfg.check_numerics(Geometry(5120, 4096))
    with pytest.raises(TypeError, match="Geometry"):
        kcfg.check_numerics((4608, 8192))


def test_classes_run_as_written_are_served_at_their_splits_and_declined_at_every_other():
    """``(5120, 6144)`` at split 4 (the geometry's default) and at split 2; ``(16384, 5120)`` and ``(14336, 5120)`` at split 1 only,
    which is NOT their default: such an entry is built with an explicit ``split_k``, and the decline of the default says so.  Every
    decline of a served geometry names the splits that geometry is served at."""
    assert kcfg.check_numerics(G_5K) == C_5K4 and kcfg.check_numerics(G_5K, 2) == C_5K2
    assert kcfg.check_numerics(G_16K, 1) == C_16K1 and kcfg.check_numerics(G_14K, 1) == C_14K1
    assert [c.SPLIT_K for c in AS_WRITTEN] == [4, 2, 1, 1] and len(set(kcfg.SERVED_CLASSES)) == len(kcfg.SERVED_CLASSES) == 7
    for geometry in (G_16K, G_14K):
        with pytest.raises(
            NotImplementedError, match=rf"serves split_k 1 at {geometry.describe()} \(validated\); got split_k 4 -- the default of the geometry; pass split_k"
        ):
            kcfg.check_numerics(geometry)
        with pytest.raises(NotImplementedError, match=r"got split_k 4 -- the default of the geometry; pass split_k"):
            entry.Fp8ProjectionDecode("cpu", geometry)
        for split in (2, 4):
            with pytest.raises(NotImplementedError, match=rf"serves split_k 1 at {geometry.describe()} \(validated\); got split_k {split}$"):
                kcfg.check_numerics(geometry, split)
    with pytest.raises(NotImplementedError, match=r"serves split_k 4 or 2 at out features 5120, in features 6144 \(validated\); got split_k 1$"):
        kcfg.check_numerics(G_5K, 1)
    with pytest.raises(NotImplementedError, match=r"serves split_k 4 or 2 at out features 4608, in features 8192 \(validated\); got split_k 1$"):
        kcfg.check_numerics(G_IN, 1)
    with pytest.raises(NotImplementedError, match=r"serves split_k 2 at out features 8192, in features 2048 \(validated\); got split_k 4$"):
        entry.Fp8ProjectionDecode("cpu", G_OUT, split_k=4)
    # an explicit split 1: one K slice per weight tile -- no reduction mailbox, a cluster of one, one row of thread blocks
    k = entry.Fp8ProjectionDecode("cpu", G_16K, split_k=1)
    cfg = k.cfg_for(6)
    assert (k.split_k, cfg.MAILBOX_ELEMENTS, cfg.REDUCE_TX_BYTES, cfg.K_TILES_PER_SLICE, cfg.CTAS) == (1, 0, 0, 40, 128)
    assert k.launch_shape_for(6) == ((128, 1, 1), (256, 1, 1), 139520)
    assert cfg.SMEM_BYTES == kcfg._align_up(128 * 128 * 8 + 8 * 128 * 8 + 2 * 8 * 8 + 2 * 8 + 4, 128)  # the rings, the barriers, the base pointer


def test_a_split_that_does_not_cut_the_k_tiles_evenly_is_declined_by_value(monkeypatch):
    """Every served geometry has a K tile count that 1 / 2 / 4 all divide, so the uneven-cut decline is shown on a modelled geometry of
    three K tiles admitted for this test only."""
    odd = Geometry(64, 3 * kcfg.K_TILE)
    monkeypatch.setattr(kcfg, "SERVED_GEOMETRIES", kcfg.SERVED_GEOMETRIES + (odd,))
    with pytest.raises(ValueError, match="split_k 2 does not cut the 3 K tiles"):
        kcfg.check_numerics(odd, 2)
    with pytest.raises(NotImplementedError, match="validated"):
        kcfg.check_numerics(odd)  # split 1 by the geometry rule: a legal cut, not a served class


def test_validator_declines_a_geometry_of_another_type_and_the_pick_an_unmeasured_class():
    with pytest.raises(TypeError, match="Cfg.GEOM must be a Geometry"):
        kcfg.validate_cfg(_cfg(GEOM=(4608, 8192)))
    with pytest.raises(ValueError, match="STAGES must be a positive int"):
        kcfg.validate_cfg(_cfg(STAGES=7.0))
    with pytest.raises(NotImplementedError, match="no measured form for out features 4608, in features 8192, split_k 1"):
        kcfg.pick(Numerics(G_IN, 1), 1, **SM100_READING)
    for bad in (dict(sm_count=0), dict(smem_per_sm=-1), dict(smem_optin=True)):
        with pytest.raises(ValueError, match="positive int"):
            kcfg.pick(C_IN4, 1, **dict(SM100_READING, **bad))


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
    wide = dict(
        weight=from_codes(torch.zeros(14336, 5120, dtype=torch.uint8)),
        activation=from_codes(torch.zeros(6, 5120, dtype=torch.uint8)),
        scale=torch.ones(1),
        out=torch.zeros(6, 14336, dtype=torch.bfloat16),
    )
    assert call_check(wide) == 6 and call_check(wide, geometry=G_14K) == 6
    with pytest.raises(ValueError, match=r"weight must be \[16384, 5120\]"):
        call_check(wide, geometry=G_16K)


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


def test_binding_declines_a_flat_weight_misaligned_operands_and_operands_on_two_devices():
    """The remaining rows of the binding table: a weight that is not 2-D when no geometry is given, an activation and a scale that do
    not start at an aligned address, and -- without an entry's device -- operands that do not share the weight's device."""
    ops = _host_ops()
    with pytest.raises(ValueError, match=r"weight must be \[out features, in features\]"):
        entry.check_binding(ops["weight"].reshape(-1), ops["activation"], ops["scale"], ops["out"])
    odd_activation = from_codes(torch.zeros(3 * 2048 + 1, dtype=torch.uint8))[1:].view(3, 2048)
    with pytest.raises(ValueError, match="activation must start at a 32-byte aligned address"):
        call_check(dict(ops, activation=odd_activation))
    odd_scale = torch.frombuffer(bytearray(16), dtype=torch.float32, count=1, offset=1)
    assert odd_scale.data_ptr() % 4
    with pytest.raises(ValueError, match="scale must start at a 4-byte aligned address"):
        call_check(dict(ops, scale=odd_scale))
    elsewhere = torch.empty(3, 2048, dtype=E4M3, device="meta")
    with pytest.raises(ValueError, match="weight's .*activation is on meta"):
        entry.check_binding(ops["weight"], elsewhere, ops["scale"], ops["out"])


def _modelled_cuda_entry(monkeypatch, readings, cc=(10, 3), **kw):
    """An entry for ``cuda:0`` on MODELLED part facts -- ``readings`` = (shared memory per SM, per-CTA opt-in, multiprocessors) and the
    compute capability -- so the declines of a part are reachable on any host; no device is touched (construction is host work)."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)
    monkeypatch.setattr(entry, "device_readings", lambda device: readings)
    return entry.Fp8ProjectionDecode("cuda:0", **kw)


def test_an_unserved_cuda_part_declines_at_construction_by_name(monkeypatch):
    for cc in ((12, 0), (9, 0), (8, 0)):
        with pytest.raises(NotImplementedError, match=rf"10\.x.*{cc[0]}\.{cc[1]}"):
            _modelled_cuda_entry(monkeypatch, (233472, 232448, 152), cc=cc, geometry=G_IN)
    for cc in ((10, 0), (10, 3), (10, 7)):
        assert _modelled_cuda_entry(monkeypatch, (233472, 232448, 152), cc=cc, geometry=G_IN).device_cc == cc


def test_part_facts_decline_by_type_at_construction_and_the_pick_adapts_where_it_can(monkeypatch):
    """A part whose per-CTA opt-in cannot hold a one-stage CTA is the pick's ``NotImplementedError``; a GIVEN form the part cannot hold
    while the SM100-class reference part can is a ``NotImplementedError`` too (a part fact, not a knob error); on a part with 100 KB per
    SM the pick lowers the ring depth instead of declining, and every row count stays one wave."""
    with pytest.raises(NotImplementedError, match="does not serve this part: a one-stage CTA"):
        _modelled_cuda_entry(monkeypatch, (16 * 1024, 15 * 1024, 152), geometry=G_IN)
    with pytest.raises(NotImplementedError, match="does not serve this part at 1 rows"):
        _modelled_cuda_entry(monkeypatch, (100 * 1024, 99 * 1024, 152), geometry=G_IN, form=Form(STAGES=11))
    k = _modelled_cuda_entry(monkeypatch, (100 * 1024, 99 * 1024, 152), geometry=G_IN)
    assert k.readings == dict(SMEM_PER_SM_BYTES=100 * 1024, SMEM_OPTIN_BYTES=99 * 1024, SM_COUNT=152)
    assert (k.cfg_for(1).STAGES, k.cfg_for(9).STAGES, k.cfg_for(17).STAGES) == (4, 3, 2) and all(c.WAVES == 1 for c in k.cfgs.values())
    assert {c.SPLIT_K for c in k.cfgs.values()} == {4}, "a smaller part changes the form, never the split"


def test_compile_declines_by_name_before_the_kernel_module_is_imported(monkeypatch):
    """The DSL gate at the plan-time compile of a CUDA entry: not installed, below the floor, and a cc 10.7 part with a DSL that has no
    ``sm_107a`` target -- each a ``NotImplementedError`` that names what is missing, raised before the kernel module is imported."""
    from cudnn.frost import buffers

    k = _modelled_cuda_entry(monkeypatch, (233472, 232448, 152), geometry=G_OUT)
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    with pytest.raises(NotImplementedError, match="not installed"):
        k.compile(1)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    with pytest.raises(NotImplementedError, match=r"4\.8\.0.*4\.7\.1|4\.7\.1.*4\.8\.0"):
        k.compile()
    rubin = _modelled_cuda_entry(monkeypatch, (233472, 232448, 212), cc=(10, 7), geometry=G_OUT)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        rubin.compile(1)
    # the DSL knows the target but the compiler it is bound to does not (its CUDA 12 flavor on a cc 10.7 part)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(entry, "dsl_compiler_version", lambda: (12, 9))
    with pytest.raises(NotImplementedError, match=r"compute capability 10\.7 needs .* CUDA 13\.4 or newer.*4\.8\.0.*compiles with CUDA 12\.9"):
        rubin.compile(1)
    assert not k.compiled and not rubin.compiled and k._kernel_module is None and rubin._kernel_module is None


def test_compiler_gate_is_per_part_and_never_declines_on_a_value_it_cannot_read(monkeypatch):
    """The compiler floor belongs to the part: the CUDA 12 flavor is declined on cc 10.7 only (cc 10.0 and cc 10.3 compile with it), a
    compiler at or above the floor passes, and a DSL that does not report its compiler is not declined."""
    assert entry.DSL_COMPILER_FLOORS == {(10, 7): (13, 4)}
    monkeypatch.setattr(entry, "dsl_compiler_version", lambda: (12, 9))
    assert entry.dsl_compiler_error((10, 0)) is None and entry.dsl_compiler_error((10, 3)) is None and entry.dsl_compiler_error(None) is None
    msg = entry.dsl_compiler_error((10, 7))
    assert "10.7" in msg and "13.4" in msg and "12.9" in msg and "nvidia-cutlass-dsl[cu13]" in msg
    for compiler in ((13, 3), (13, 0)):
        monkeypatch.setattr(entry, "dsl_compiler_version", lambda c=compiler: c)
        assert entry.dsl_compiler_error((10, 7)) is not None
    for compiler in ((13, 4), (13, 6), (14, 0), None):
        monkeypatch.setattr(entry, "dsl_compiler_version", lambda c=compiler: c)
        assert entry.dsl_compiler_error((10, 7)) is None, compiler


def test_compile_follows_the_gate_on_this_device():
    """On whatever cc 10.x device and DSL compiler this process has: when the gate passes, the plan-time compile succeeds; when it
    declines (the DSL's CUDA 12 compiler on a cc 10.7 part), ``compile`` raises that message as a ``NotImplementedError`` and nothing
    was traced -- the DSL's own refusal of the target is never what the caller sees."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the FP8 projection decode kernel serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry.dsl_error(cc)
    k = entry.Fp8ProjectionDecode(dev, G_OUT)
    if msg is None:
        k.compile(1)
        assert k.cfg_for(1) in k.compiled
        return
    print(f"declined on this device: {msg}")
    with pytest.raises(NotImplementedError) as declined:
        k.compile(1)
    assert str(declined.value) == msg and not k.compiled and k._kernel_module is None


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


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the launch facts (cudnn.frost.launch_facts) -- the symbol of the compiled rendering, the form as plain data, the launch shape
# ---------------------------------------------------------------------------------------------------------------------------
def _launch_arithmetic(cfg):
    """The launch of ``kernel/fp8_projection_decode_e4m3.py`` restated from the rendering's FIELDS and the shared-memory ledger: one
    thread block per weight tile along x, one per K slice along y for the one token tile that holds the rows; 256 threads; the
    reservation ``smem_bytes`` mirrors."""
    grid = (-(-cfg.GEOM.OUT_FEATURES // cfg.MMA_M), -(-cfg.ROWS // cfg.TOKEN_TILE) * cfg.SPLIT_K, 1)
    return (grid, (256, 1, 1), kcfg.smem_bytes(cfg.MMA_M, cfg.TOKEN_TILE, cfg.SPLIT_K, cfg.STAGES))


LAUNCH_FACT_ARMS = [
    dict(),
    dict(pdl=True),
    dict(form=Form(MMA_M=128, STAGES=4)),
    dict(form=Form(MMA_M=64, STAGES=2, TOKEN_TILE=16), pdl=True, pdl_release=False),
]


@pytest.mark.parametrize("arm", LAUNCH_FACT_ARMS, ids=["pick", "pick-pdl", "form-m128-s4", "form-m64-s2-tile16-pdl"])
@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_launch_facts_equal_the_configuration_arithmetic_at_every_row_count(numerics, arm):
    """EVERY served row count, 1 .. 32, per served class and form arm (the pick, the pick under dependent launch, two given forms): the
    launch shape is the configuration arithmetic of the rendering ``cfg_for`` returns -- its thread blocks are the rendering's ``CTAS``
    -- and the form is that rendering's fields, the geometry nested: the rendering is the whole compile key, so nothing is added."""
    k = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K, **arm)
    for rows in range(1, kcfg.MAX_ROWS + 1):
        cfg = k.cfg_for(rows)
        shape = k.launch_shape_for(rows)
        assert shape == _launch_arithmetic(cfg), (rows, shape)
        assert all(type(v) is int for v in (*shape.grid, *shape.block, shape.shared_memory)), shape
        assert shape.grid[0] * shape.grid[1] == cfg.CTAS and shape.grid[1] == numerics.SPLIT_K and shape.shared_memory <= cfg.SMEM_OPTIN_BYTES
        form = k.form_for(rows)
        assert form == asdict(cfg) and form["ROWS"] == rows and form["SPLIT_K"] == numerics.SPLIT_K, rows
        assert form["GEOM"] == dict(OUT_FEATURES=numerics.GEOM.OUT_FEATURES, IN_FEATURES=numerics.GEOM.IN_FEATURES)
        assert (form["PDL"], form["PDL_RELEASE"]) == (
            bool(arm.get("pdl")),
            bool(arm.get("pdl_release", arm.get("pdl"))),
        ), "the launch knobs are part of the form"
        assert {name: form[name] for name in ("MMA_M", "STAGES", "TOKEN_TILE", "TMEM_COLS", "PREWAIT_WEIGHTS", "HOLD_TMEM")} == asdict(cfg.FORM)


def test_launch_facts_follow_the_part(monkeypatch):
    """The same equality on modelled part facts: a part with 100 KB per SM lowers the ring depth of the pick, and the launch shape and
    the form follow the rendering the entry resolved for THAT part (its readings are fields of the form)."""
    k = _modelled_cuda_entry(monkeypatch, (100 * 1024, 99 * 1024, 152), geometry=G_IN)
    reference = entry.Fp8ProjectionDecode("cpu", G_IN)
    for rows in range(1, kcfg.MAX_ROWS + 1):
        assert k.launch_shape_for(rows) == _launch_arithmetic(k.cfg_for(rows)) and k.form_for(rows) == asdict(k.cfg_for(rows))
        assert k.launch_shape_for(rows).shared_memory < reference.launch_shape_for(rows).shared_memory, "a shallower ring on the smaller part"
    assert (k.form_for(1)["SMEM_PER_SM_BYTES"], k.form_for(1)["SMEM_OPTIN_BYTES"], k.form_for(1)["SM_COUNT"]) == (100 * 1024, 99 * 1024, 152)


def test_launch_shape_of_the_served_classes_at_the_recorded_forms():
    """The shapes a profiler recorded for launches on a cc 10.0 part, each form given explicitly on the reference readings: the three
    served classes at their stage counts (the device cell below repeats the comparison on whatever part runs it)."""
    for numerics, form, rows, want in (
        (C_IN4, Form(MMA_M=64, STAGES=11), 1, ((72, 4, 1), (256, 1, 1), 107904)),
        (C_IN4, Form(MMA_M=64, STAGES=7), 6, ((72, 4, 1), (256, 1, 1), 71040)),
        (C_IN2, Form(MMA_M=64, STAGES=8), 9, ((72, 2, 1), (256, 1, 1), 86400)),
        (C_OUT2, Form(MMA_M=64, STAGES=8), 1, ((128, 2, 1), (256, 1, 1), 76160)),
    ):
        k = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K, form=form)
        assert k.launch_shape_for(rows) == want, (numerics, form, rows)


def test_launch_shape_and_form_of_the_classes_run_as_written():
    """The pick of the four classes run as written on the reference readings: the grid is one row of thread blocks per K slice, the
    shared memory is the table row's footprint at the row count's token tile, and the form states the late tensor-memory release."""
    for numerics, rows, want in (
        (C_5K4, 6, ((80, 4, 1), (256, 1, 1), 71040)),
        (C_5K4, 9, ((80, 4, 1), (256, 1, 1), 84352)),
        (C_5K2, 6, ((80, 2, 1), (256, 1, 1), 103808)),
        (C_16K1, 1, ((128, 1, 1), (256, 1, 1), 139520)),
        (C_16K1, 32, ((128, 1, 1), (256, 1, 1), 164096)),
        (C_14K1, 6, ((112, 1, 1), (256, 1, 1), 139520)),
    ):
        k = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K)
        assert k.launch_shape_for(rows) == want, (cid(numerics), rows)
        form = k.form_for(rows)
        assert form["HOLD_TMEM"] is True and form["TMEM_COLS"] == 32 and form["SPLIT_K"] == numerics.SPLIT_K
        early = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K, form=Form(MMA_M=form["MMA_M"], STAGES=form["STAGES"]))
        assert early.launch_shape_for(rows) == want and early.form_for(rows)["HOLD_TMEM"] is False, "a given form is taken as given"


def test_symbol_text_names_the_rendering_of_a_class_run_as_written():
    """``Cfg.SYMBOL_TEXT``: empty for the classes whose symbol names the weight's shape only; for every other class a text of at most 41
    characters of ``[a-z0-9_]`` (the DSL drops other characters from a symbol) that differs whenever a field the kernel depends on
    differs -- across the classes, the 32 row counts, the dependent-launch knobs and the form arms -- and does not move with the
    part's readings.  The kernel's name is 41 characters, so the text ends inside the first 100 characters of the symbol."""
    assert kcfg.SHAPE_SYMBOL_CLASSES == {C_IN4, C_IN2, C_OUT2}
    for numerics in (C_IN4, C_IN2, C_OUT2):
        for arm in (dict(), dict(pdl=True), dict(form=Form(MMA_M=128, STAGES=4))):
            k = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K, **arm)
            assert {k.cfg_for(rows).SYMBOL_TEXT for rows in range(1, 33)} == {""}
    kernel_name = "cudnn_kernel_frost_fp8_projection_decode_"
    arms = [dict(), dict(pdl=True), dict(pdl=True, pdl_release=False), dict(pdl_release=True)]
    arms += [dict(form=Form(MMA_M=m, STAGES=g)) for m, g in ((64, 7), (64, 8), (128, 4))]
    arms += [
        dict(form=Form(MMA_M=64, STAGES=7, HOLD_TMEM=True)),
        dict(form=Form(MMA_M=64, STAGES=7, TMEM_COLS=64)),
        dict(form=Form(MMA_M=64, STAGES=7, TOKEN_TILE=16)),
    ]
    arms += [dict(form=Form(MMA_M=64, STAGES=7, PREWAIT_WEIGHTS=False), pdl=True)]
    texts = {}
    for numerics in AS_WRITTEN:
        for arm in arms:
            k = entry.Fp8ProjectionDecode("cpu", numerics.GEOM, split_k=numerics.SPLIT_K, **arm)
            for rows in range(1, 33):
                cfg = k.cfg_for(rows)
                text = cfg.SYMBOL_TEXT
                assert text and len(text) <= 41 and len(kernel_name) + len(text) <= 100 and set(text) <= set("abcdefghijklmnopqrstuvwxyz0123456789_"), text
                rendering = replace(cfg, SMEM_PER_SM_BYTES=0, SMEM_OPTIN_BYTES=0, SM_COUNT=0)  # everything of the compile key but the part
                assert texts.setdefault(text, rendering) == rendering, f"two renderings share the text {text}"
                assert replace(cfg, SM_COUNT=64, SMEM_PER_SM_BYTES=1 << 20).SYMBOL_TEXT == text
    assert len(texts) == len({r for r in texts.values()}) > 4 * 32 * 6
    cfg = entry.Fp8ProjectionDecode("cpu", G_16K, split_k=1, pdl=True).cfg_for(6)
    assert cfg.SYMBOL_TEXT == "n16384k5120s1r6_m128g8t8c32_w1h1p1q1"
    assert entry.Fp8ProjectionDecode("cpu", G_5K, split_k=2, form=Form(MMA_M=64, STAGES=11)).cfg_for(8).SYMBOL_TEXT == "n5120k6144s2r8_m64g11t8c32_w1h0p0q0"


def test_launch_facts_are_plain_fresh_and_decline_an_unserved_row_count():
    """The facts are plain data, a fresh object per call (a caller's edit reaches nothing of the entry) and the entry's typed declines
    for a row count outside 1 .. 32."""
    k = entry.Fp8ProjectionDecode("cpu", G_IN, pdl=True)
    form = k.form_for(9)
    assert json.loads(json.dumps(form)) == form and type(form) is dict and type(form["GEOM"]) is dict
    form["PDL"], form["GEOM"]["OUT_FEATURES"] = "edited", 0
    assert k.form_for(9)["PDL"] is True and k.form_for(9)["GEOM"]["OUT_FEATURES"] == 4608 and k.cfg_for(9).GEOM.OUT_FEATURES == 4608
    for fact in (k.kernel_symbols_for, k.form_for, k.launch_shape_for):
        for rows in (0, -1, 1.0, "1", None, True):
            with pytest.raises(ValueError, match="rows must be a positive int"):
                fact(rows)
        with pytest.raises(NotImplementedError, match="at most 32 rows"):
            fact(33)
        fact(32)


class _CompiledStandIn:
    """What the entry keeps per compile key, as far as the launch facts read it: a compiled DSL host function carries ``kernel_info``,
    keyed by the symbols of the kernels it launches."""

    def __init__(self, *symbols):
        self.kernel_info = {symbol: {} for symbol in symbols}


def test_kernel_symbol_is_that_of_the_rendering_compiled_for_the_row_count():
    """Host-only, with stand-ins in the entry's compile cache: nothing compiled -> an empty tuple; a row count names the symbol of ITS
    compiled rendering only (one rendering per row count; two renderings of one geometry share a symbol in practice, which is why the
    form, not the symbol, tells them apart); a compiled object that does not expose its kernels is the typed decline, never a guess."""
    k = entry.Fp8ProjectionDecode("cpu", G_IN)
    assert all(k.kernel_symbols_for(rows) == () for rows in range(1, kcfg.MAX_ROWS + 1))
    shared = "cudnn_kernel_frost_fp8_projection_decode_o460881921819210_0"
    k.compiled[k.cfg_for(3)] = _CompiledStandIn(shared)
    assert k.kernel_symbols_for(3) == (shared,) and k.kernel_symbols_for(4) == () and k.kernel_symbols_for(1) == ()
    k.compiled[k.cfg_for(9)] = _CompiledStandIn(shared)
    assert k.kernel_symbols_for(9) == k.kernel_symbols_for(3) == (shared,) and k.form_for(9) != k.form_for(3), "one symbol, two forms"
    k.compiled[k.cfg_for(3)] = object()
    with pytest.raises(NotImplementedError, match="the FP8 projection decode kernel: the installed CuTe DSL .* does not expose the kernel symbols"):
        k.kernel_symbols_for(3)


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
    assert len(names) == len(set(names)) and "kernel" in names and "tma_load" in names and "LayoutEnum" in names


def test_layer_probe_covers_the_shim_of_the_moved_dsl_names(monkeypatch):
    """The kernel module binds the DSL names that moved between releases through ``cudnn._cutlass_compat``.  The probe imports that shim,
    so a DSL on which it does not import is a decline by name before the kernel module is imported, never an ``ImportError`` out of
    ``compile``."""
    from cudnn.frost import buffers

    shim = [row for row in entry.DSL_LAYER if row[0] == "cudnn._cutlass_compat"]
    assert shim == [("cudnn._cutlass_compat", ("LayoutEnum",))]
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "9.9.9")))
    monkeypatch.setattr(entry, "DSL_LAYER", tuple(shim))
    monkeypatch.setitem(sys.modules, "cudnn._cutlass_compat", None)  # the import system raises for a None entry
    msg = entry.dsl_error()
    assert msg is not None and "cudnn._cutlass_compat" in msg and "9.9.9" in msg and "does not import" in msg
    k = _modelled_cuda_entry(monkeypatch, (233472, 232448, 152), geometry=G_OUT)
    with pytest.raises(NotImplementedError, match=r"cudnn\._cutlass_compat"):
        k.compile(1)
    assert not k.compiled and k._kernel_module is None


def test_kernel_module_reads_no_deprecated_name_off_a_dsl_module():
    """Every name the kernel module imports from a DSL module or reads off one resolves on the installed DSL without a
    ``DeprecationWarning``, unless the layer probe names it.  The DSL keeps a moved name as a warning alias for some releases and then
    drops it; a kernel that still reads the alias raises from inside the trace on that release, at every row count, where the entry
    promises a decline by name."""
    msg = entry.dsl_error()
    if msg is not None:
        pytest.skip(msg)
    from cudnn.gemm.frost.projection_decode.kernel import fp8_projection_decode_e4m3 as kernel

    reads = set()
    for node in ast.walk(ast.parse(Path(kernel.__file__).read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module.startswith("cutlass"):
            reads |= {(node.module, alias.name, node.lineno) for alias in node.names}
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner = vars(kernel).get(node.value.id)
            if isinstance(owner, types.ModuleType) and owner.__name__.startswith("cutlass"):
                reads.add((owner.__name__, node.attr, node.lineno))
    assert len(reads) > 20, "the scan found the kernel module's DSL reads"
    probed = {(module, name) for module, names in entry.DSL_LAYER for name in names}
    flagged = []
    for module, name, line in sorted(reads):
        if (module, name) in probed:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                getattr(importlib.import_module(module), name)
            except (DeprecationWarning, AttributeError) as e:
                flagged.append(f"line {line}: {module}.{name}: {type(e).__name__}: {e}")
    assert not flagged, flagged


def test_cudnn_level_export_resolves_to_the_entry():
    assert cudnn.Fp8ProjectionDecode is entry.Fp8ProjectionDecode
    gemm, pkg = importlib.import_module("cudnn.gemm"), importlib.import_module("cudnn.gemm.frost.projection_decode")
    assert gemm.Fp8ProjectionDecode is entry.Fp8ProjectionDecode and gemm.ProjectionGeometry is Geometry and gemm.ProjectionForm is Form
    assert pkg.Fp8ProjectionDecode is entry.Fp8ProjectionDecode and pkg.Geometry is Geometry and pkg.Form is Form
    assert "Fp8ProjectionDecode" in cudnn._LAZY_OPTIONAL_IMPORTS and "Fp8ProjectionDecode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS
    assert {"Fp8ProjectionDecode", "ProjectionGeometry", "ProjectionForm"} <= set(gemm.__all__)


def test_docs_rows_exist_for_the_api():
    """The determinism status page carries the API's row -- one row for the one scale form (a single fp32 scale); SM90 unsupported
    (declined by type); the evidence differs by class, so each GPU cell states its limit inside the existing column: cc 10.0 ran every
    served class, cc 10.3 and cc 10.7 the classes of ``(4608, 8192)`` and ``(8192, 2048)`` only, and every class a part did not run
    reads Unverified there -- and its required-settings bullet, which names every served geometry; the goldens README carries the
    suite's row and section; the operation page, its navigation entry, the overview line and the ``llms.txt`` line exist; and the
    page's byte-class table names exactly the classes the entry serves."""
    det = (REPO / "docs" / "operations" / "determinism.md").read_text()
    row = [line for line in det.splitlines() if line.startswith("| `Fp8ProjectionDecode`")]
    assert len(row) == 1, "one Fp8ProjectionDecode row on the status page"
    cells = [c.strip() for c in row[0].strip("|").split("|")]
    assert cells[1] == "fprop (inference step)" and cells[3] == "Unsupported", cells
    assert cells[2] == "FP8 E4M3 in, one FP32 scale; BF16 out; FP32 accumulate", cells
    verified_everywhere = "`ProjectionGeometry(4608, 8192)` and `(8192, 2048)`"
    assert cells[4] == f"SM100: Supported; SM103: Supported at {verified_everywhere}, Unverified at the other served geometries", cells
    assert cells[5] == f"Supported at {verified_everywhere}; Unverified at the other served geometries", cells
    assert "FP8 projection for decode rows (`Fp8ProjectionDecode`): no determinism" in det
    bullet = " ".join(det[det.index("FP8 projection for decode rows (`Fp8ProjectionDecode`): no determinism") :].split("\n- ")[0].split())
    for geometry in kcfg.SERVED_GEOMETRIES:
        assert f"`ProjectionGeometry({geometry.OUT_FEATURES}, {geometry.IN_FEATURES})`" in bullet, geometry
    assert "verified on cc 10.0 only" in bullet and "not been run" in bullet
    readme = (REPO / "test" / "python" / "determinism_goldens" / "README.md").read_text()
    assert "`Fp8ProjectionDecode`" in readme and "test_fp8_projection_decode_determinism.py" in readme
    assert "### FP8 projection for decode rows" in readme
    page = REPO / "docs" / "operations" / "gemm_fusions" / "fp8_projection_decode.md"
    assert page.exists()
    text = page.read_text()
    served = [line for line in text.splitlines() if line.startswith("| `ProjectionGeometry(")]
    assert len(served) == len(kcfg.SERVED_CLASSES)
    for numerics, line in zip(kcfg.SERVED_CLASSES, served):
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[0] == f"`ProjectionGeometry({numerics.GEOM.OUT_FEATURES}, {numerics.GEOM.IN_FEATURES})`" and cells[1] == str(numerics.SPLIT_K), cells
    for token in ("rows 1 .. 32", "compute capability 10.x", f"nvidia-cutlass-dsl >= {entry.DSL_FLOOR_TEXT}", "HOLD_TMEM", "FlashInfer"):
        assert token in text, token
    assert "fp8_projection_decode" in (REPO / "docs" / "fern" / "versions" / "latest.yml").read_text()
    assert "fp8_projection_decode" in (REPO / "docs" / "operations" / "overview.md").read_text()
    assert "fp8_projection_decode" in (REPO / "llms.txt").read_text()


# Child interpreters.  A child is a fresh process: nothing of this process's import setup carries over, so ``import cudnn`` there is the
# installed package, while this process may run the entry from a source tree laid over that package (this directory's conftest).  A
# child that imports the entry therefore runs CHILD_TREE first -- the same overlay, with the source directory THIS process loaded the
# entry from -- and every child prints the files it loaded on ``LOADED`` lines, which the parent compares with its own (test/AGENTS.md,
# "Confirm you are testing the code you edited").  Without the overlay a child would test the installed package, whatever tree that is.
SOURCE_CUDNN = str(Path(entry.__file__).parents[3])  # <...>/cudnn of <...>/cudnn/gemm/frost/projection_decode/<the entry module>
KERNEL_FILE = str(Path(kcfg.__file__).with_name("fp8_projection_decode_e4m3.py"))
CHILD_TREE = f"""
import importlib, os
for name in ("cudnn", "cudnn.gemm"):
    package, source = importlib.import_module(name), os.path.join({SOURCE_CUDNN!r}, *name.split(".")[1:])
    if source not in package.__path__:
        package.__path__.insert(0, source)
"""


def loaded_files(stdout):
    """``{name: file}`` of the ``LOADED <name> <file>`` lines a child printed."""
    return dict(line.split(" ", 2)[1:] for line in stdout.splitlines() if line.startswith("LOADED "))


def test_import_cudnn_does_not_import_the_kernel_or_torch():
    """The lazy boundary in fresh interpreters: importing cudnn loads nothing of this package; building a host entry and running its
    declines loads neither the kernel module nor the DSL.  The first child runs the package initializer this process runs; the second
    one the entry module of the tree under test."""
    code = "import sys, cudnn; print('LOADED cudnn', cudnn.__file__)\n"
    code += "print(sorted(m for m in sys.modules if 'projection_decode' in m or m in ('cutlass', 'torch')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded_files(out) == {"cudnn": cudnn.__file__}, f"the child imported another cudnn package than this process:\n{out}"
    assert out.strip().splitlines()[-1] == "[]", out
    code = CHILD_TREE + textwrap.dedent("""
        import sys, cudnn
        from cudnn.gemm import ProjectionGeometry
        k = cudnn.Fp8ProjectionDecode("cpu", ProjectionGeometry(8192, 2048))
        try:
            k.compile(1)
        except NotImplementedError:
            pass
        print("LOADED entry", sys.modules[type(k).__module__].__file__)
        print(sorted(m for m in sys.modules if m == "cutlass" or m.endswith("fp8_projection_decode_e4m3")))
        """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded_files(out) == {"entry": entry.__file__}, f"the child did not load the entry of the tree under test:\n{out}"
    assert out.strip().splitlines()[-1] == "[]", out


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


def test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current():
    """An entry built for a device that is NOT the caller's current one compiles and launches on ITS device and leaves the caller's
    current device as it found it: on the first call (compile + launch), on a warm call, and on a side stream of the entry's device
    entered while the caller's device is current.  The operand views of a launch are built under the entry's device, like the launch
    itself.  The bytes are those of the same call made with the entry's device current.  Needs two CUDA devices the kernel serves."""
    dev = device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    other = torch.device("cuda", (dev.index + 1) % torch.cuda.device_count())
    cc = torch.cuda.get_device_capability(other)
    if not entry.cc_supported(cc) or entry.dsl_error(cc) is not None:
        pytest.skip(f"the second CUDA device ({other}, compute capability {cc[0]}.{cc[1]}) is not one the kernel serves")
    rows = 5
    ops = make_operands(G_OUT, rows, other, seed=77, kind="wide")
    with torch.cuda.device(other):  # the reference: the same call with the entry's device current
        want = _run(entry.Fp8ProjectionDecode(other, G_OUT), ops, rows).clone()
        torch.cuda.synchronize(other)
    assert torch.cuda.current_device() == dev.index
    k = entry.Fp8ProjectionDecode(other, G_OUT)  # built, compiled and launched while the CALLER's device is current
    assert k.device == other
    first = _run(k, ops, rows)  # the first call: compile + launch
    assert torch.cuda.current_device() == dev.index, "the first call (compile + launch) left the entry's device current"
    assert first.device == other
    assert_same_bytes(first, want, "the first call on the entry's device")
    warm = _run(k, ops, rows)  # a warm call: the launch alone
    assert torch.cuda.current_device() == dev.index, "a warm call left the entry's device current"
    assert_same_bytes(warm, want, "a warm call on the entry's device")
    mine = new_out(G_OUT, rows, other)
    side = torch.cuda.Stream(device=other)
    side.wait_stream(torch.cuda.current_stream(other))  # the operands and the fill were produced on the entry's device's current stream
    with torch.cuda.stream(side):  # the entry's device current, ``side`` its current stream ...
        with torch.cuda.device(dev):  # ... then the caller's device current again: ``side`` stays the entry's device's current stream
            _run(k, ops, rows, mine)
            assert torch.cuda.current_device() == dev.index, "a launch on a side stream of the entry's device left that device current"
    torch.cuda.current_stream(other).wait_stream(side)
    assert_same_bytes(mine, want, "the side-stream launch on the entry's device")
    assert torch.cuda.current_device() == dev.index
    assert torch.empty(1, device="cuda").device == dev, "the caller's next default-device allocation landed on another device"


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
def test_extreme_and_subnormal_codes_write_the_host_bytes(numerics):
    """Edge rows whose every partial sum is exact in fp32 in any order, byte for byte against the host: the largest code everywhere
    (the largest accumulator the format can reach), the largest code with alternating signs (exact cancellation: zeros), subnormal
    codes (1 .. 7 units of 2^-9) with and without alternating signs, the smallest magnitude and the largest negative subnormal --
    against weight rows of the largest code (both signs) and of subnormal codes.  Each row holds one family of codes, so every product
    of a (row, weight row) pair is a small integer multiple of one unit and every sum stays below 2^24 units."""
    dev = device_or_skip()
    g = numerics.GEOM
    n, kdim = g.OUT_FEATURES, g.IN_FEATURES
    ki = torch.arange(kdim)
    alternate = ((ki % 2) << 7).to(torch.uint8)  # + - + - along K
    subnormal = (1 + ki % 7).to(torch.uint8)  # codes 1 .. 7
    largest = torch.full((kdim,), MAX_CODE, dtype=torch.uint8)
    x = torch.stack([largest, largest | alternate, subnormal, subnormal | alternate, torch.full_like(largest, 0x01), torch.full_like(largest, 0x87)])
    w = torch.empty(n, kdim, dtype=torch.uint8)
    w[0::4], w[1::4], w[2::4], w[3::4] = largest, largest | 0x80, subnormal.flip(0), subnormal | alternate
    ops = dict(weight=from_codes(w).to(dev), activation=from_codes(x).to(dev), scale=torch.tensor([2.0**-6], dtype=torch.float32, device=dev))
    out = _run(_entry(dev, numerics), ops, x.shape[0])
    torch.cuda.synchronize()
    exact = ops["activation"].float().double() @ ops["weight"].float().double().t()
    want = (exact * ops["scale"].double().reshape(())).float().to(torch.bfloat16)
    assert bool(torch.isfinite(want.float()).all()) and float(exact.abs().max()) == 448.0 * 448.0 * kdim
    assert int((want[2:].float() != 0).sum()) > 0.75 * want[2:].numel(), "the subnormal rows of the host result are not a zero test"
    assert_same_bytes(out, want, f"{cid(numerics)} extreme and subnormal codes")
    assert int((out[1, 0::4].float() != 0).sum()) == 0 and int((out[1, 1::4].float() != 0).sum()) == 0, "exact cancellation"


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


@pytest.mark.parametrize("pair", ((C_IN4, C_IN2), (C_5K4, C_5K2)), ids=lambda pair: cid(pair[0])[:-2])
def test_split_is_a_byte_class_and_each_class_replays(pair):
    """Another split is another fp32 association: both splits of one geometry stay inside the reference bound and each replays bitwise;
    the number of elements on which they differ is printed, never asserted."""
    dev = device_or_skip()
    ops = _ops(dev, pair[0].GEOM, "wide")
    outs = {}
    for numerics in pair:
        k = _entry(dev, numerics)
        outs[numerics] = _run(k, ops, 24)
        assert_same_bytes(_run(k, ops, 24), outs[numerics], f"{cid(numerics)} replay")
        ratio, bad = error_over_bound(outs[numerics], dict(ops, activation=ops["activation"][:24]), numerics.SPLIT_K)
        assert ratio <= 1.0 and bad == 0
    differing = int((raw(outs[pair[0]]) != raw(outs[pair[1]])).sum())
    print(f"{pair[0].GEOM.describe()}: split 4 against split 2 at 24 rows: {differing} of {outs[pair[0]].numel()} elements differ")


def test_graph_capture_replays_bitwise_with_eager():
    dev = device_or_skip()
    cells = ((C_IN4, 6), (C_OUT2, 6), (C_IN2, 24), (C_16K1, 6), (C_5K2, 6), (C_5K4, 12))  # split 4, 2 and 1; one wave and two
    layers = [(_entry(dev, numerics), _ops(dev, numerics.GEOM, "random", seed=7), rows) for numerics, rows in cells]
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


def test_trace_raises_no_deprecation_warning_from_the_kernel_package():
    """A fresh entry's compile -- a real trace on this device's DSL -- attributes no ``DeprecationWarning`` to a file of the entry's
    package: a deprecated DSL accessor in the kernel body shows here on the releases that warn, ahead of the one that drops it."""
    dev = device_or_skip()
    k = entry.Fp8ProjectionDecode(dev, G_IN)  # not one of this module's shared entries: nothing is compiled yet, so compile traces
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        k.compile(1)
        k.compile(17)
    assert len(k.compiled) == 2
    package = str(Path(entry.__file__).parent)
    own = [f"{w.filename}:{w.lineno}: {w.message}" for w in seen if issubclass(w.category, DeprecationWarning) and str(w.filename).startswith(package)]
    assert not own, own


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
CHILD_COMPILE_BUDGET_S = 420  # a cold DSL import plus the child's trace-compiles (54 or 144)
CHILD_DISPATCH_LIMIT_S = 60  # the child's forms x 16 warm launches each take milliseconds
# One child per group of classes: the one-wave classes at their pick; the classes run as written at their pick AND in the given form
# of the same weight tile and ring depth, which releases the tensor memory early -- so both release orders of every class that can
# take more than one wave, and of the split-1 classes, first run where a kernel that does not return is attributed to its stage.
CHILD_GROUPS = {"one-wave": (0, 3, False), "as-written": (3, 7, True)}
_CHILD = """
import faulthandler, sys, time
faulthandler.dump_traceback_later({dump_s}, exit=False)
print("STAGE import", flush=True)
{tree}
import torch
from cudnn.gemm.frost.projection_decode import fp8_projection_decode as entry
from cudnn.gemm.frost.projection_decode.kernel import fp8_projection_decode_config as kcfg
print("LOADED entry", entry.__file__, flush=True)
print("LOADED config", kcfg.__file__, flush=True)
dev = torch.device("cuda", torch.cuda.current_device())
print("STAGE compile", flush=True)
cells = []
for numerics in kcfg.SERVED_CLASSES[{first}:{last}]:
    g = numerics.GEOM
    w = torch.zeros(g.OUT_FEATURES, g.IN_FEATURES, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    x = torch.zeros(32, g.IN_FEATURES, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    s = torch.ones(1, dtype=torch.float32, device=dev)
    for pdl in (False, True):
        entries = [entry.Fp8ProjectionDecode(dev, g, split_k=numerics.SPLIT_K, pdl=pdl)]
        if {given_form}:
            pick = entries[0].cfg_for(1)
            entries.append(entry.Fp8ProjectionDecode(dev, g, split_k=numerics.SPLIT_K, pdl=pdl, form=kcfg.Form(MMA_M=pick.MMA_M, STAGES=pick.STAGES)))
        for k in entries:
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
print("LOADED kernel", cells[0][0]._kernel_module.__file__, flush=True)
print("STAGE done", round(time.monotonic() - t0, 3), "s", flush=True)
"""


@pytest.mark.parametrize("group", CHILD_GROUPS)
def test_launch_matrix_completes_in_a_child_process(group):
    device_or_skip()
    limit = CHILD_COMPILE_BUDGET_S + CHILD_DISPATCH_LIMIT_S
    first, last, given_form = CHILD_GROUPS[group]
    code = _CHILD.format(dump_s=limit - 15, tree=CHILD_TREE, first=first, last=last, given_form=given_form)
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
    tree = dict(entry=entry.__file__, config=kcfg.__file__, kernel=KERNEL_FILE)
    assert loaded_files(proc.stdout) == tree, f"the child did not dispatch the kernel of the tree under test ({tree}):\n{proc.stdout}"
    print(proc.stdout)


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


LAUNCH_RECORD_CELLS = [(C_IN4, 1, dict()), (C_IN4, 6, dict(pdl=True)), (C_IN2, 9, dict()), (C_OUT2, 1, dict())]
LAUNCH_RECORD_IDS = ["in-split4-rows1", "in-split4-rows6-pdl", "in-split2-rows9", "out-split2-rows1"]
LAUNCH_RECORD_CELLS += [(C_5K4, 9, dict()), (C_5K2, 6, dict(pdl=True)), (C_16K1, 6, dict(pdl=True)), (C_14K1, 1, dict())]
LAUNCH_RECORD_CELLS += [(C_16K1, 6, dict(pdl=True, form=Form(MMA_M=128, STAGES=8)))]
LAUNCH_RECORD_IDS += ["n5120k6144-split4-rows9", "n5120k6144-split2-rows6-pdl", "n16384k5120-split1-rows6-pdl", "n14336k5120-split1-rows1"]
LAUNCH_RECORD_IDS += ["n16384k5120-split1-rows6-pdl-given-form"]


@pytest.mark.parametrize("numerics, rows, arm", LAUNCH_RECORD_CELLS, ids=LAUNCH_RECORD_IDS)
def test_launch_facts_equal_the_profilers_record_of_a_launch(numerics, rows, arm):
    """On the device: one launch of the entry's pick for ``rows`` on THIS part (each served class; one under dependent launch) is ONE
    kernel whose profiler record -- name, grid, block, shared memory -- equals the entry's launch facts; nothing is named before the
    first compile, and another row count of the entry is named only once it is compiled.  A fresh entry per cell (not the module's
    shared ones): the cell reads the compile cache.  The helper synchronises between its warm launch and the recorded one, so the
    dependent-launch form finds no kernel in flight."""
    dev = device_or_skip()
    k = entry.Fp8ProjectionDecode(dev, numerics.GEOM, split_k=numerics.SPLIT_K, **arm)
    ops = _ops(dev, numerics.GEOM, "random")
    out = new_out(numerics.GEOM, rows, dev)
    assert k.kernel_symbols_for(rows) == (), "nothing is compiled yet"
    record = assert_facts_equal_the_launch(k, lambda: k(ops["weight"], ops["activation"][:rows], ops["scale"], out), rows, what=entry.NAME)
    assert record.name.startswith("cudnn_kernel_frost_fp8_projection_decode_"), record.name
    text = k.cfg_for(rows).SYMBOL_TEXT
    if numerics in kcfg.SHAPE_SYMBOL_CLASSES:
        assert text == "" and "Fp8ProjectionDecodeKernel" in record.name, record.name
    else:  # the launched kernel's name carries the rendering's text inside its first 100 characters
        assert text and record.name[:100].startswith(f"cudnn_kernel_frost_fp8_projection_decode_{text}_"), (record.name, text)
    other = 32 if rows != 32 else 1
    assert k.kernel_symbols_for(other) == (), "a row count that was not compiled names nothing"
    assert k.form_for(rows) == asdict(k.cfg_for(rows)) and (k.form_for(rows)["PDL"], k.form_for(rows)["ROWS"]) == (bool(arm.get("pdl")), rows)


def test_renderings_of_a_class_run_as_written_have_a_symbol_each():
    """Compiled, not launched: at one class and row count the pick, the pick under dependent launch, the pick without the release and
    the given form of the pick's tile and depth (the early tensor-memory release) are four renderings with four symbols, each
    carrying its ``SYMBOL_TEXT``; two row counts of one entry differ too.  At a one-wave class the same four share one symbol."""
    dev = device_or_skip()

    def symbols(numerics, rows):
        pick = entry.Fp8ProjectionDecode(dev, numerics.GEOM, split_k=numerics.SPLIT_K).cfg_for(rows)
        arms = (dict(), dict(pdl=True), dict(pdl=True, pdl_release=False), dict(form=Form(MMA_M=pick.MMA_M, STAGES=pick.STAGES)))
        found = []
        for arm in arms:
            k = entry.Fp8ProjectionDecode(dev, numerics.GEOM, split_k=numerics.SPLIT_K, **arm)
            k.compile(rows)
            (symbol,) = k.kernel_symbols_for(rows)
            found.append((symbol, k.cfg_for(rows).SYMBOL_TEXT))
        return found

    for numerics in (C_5K4, C_14K1):
        found = symbols(numerics, 6) + symbols(numerics, 7)[:1]
        assert len({symbol for symbol, _ in found}) == len(found) == 5, found
        assert all(text and symbol[:100].startswith(f"cudnn_kernel_frost_fp8_projection_decode_{text}_") for symbol, text in found), found
    shared = symbols(C_OUT2, 6)
    assert len({symbol for symbol, _ in shared}) == 1 and {text for _, text in shared} == {""}, shared


# (class, rows, arm).  The classes run as written, at their pick: the 8- and the 32-row token tile, and nine rows where the grid first
# takes two waves.  Then the cases a pick does not reach: the given form that releases the tensor memory EARLY where the grid takes
# more than one wave (a CTA of the second wave can be given columns a CTA of the first deallocated), and a form of the split-1 classes
# small enough for several thread blocks per SM (their picks sit one per SM, below the geometry the stress's control is run in).
STRESS_CELLS = [(numerics, rows, dict()) for numerics in AS_WRITTEN for rows in (1, 32)] + [(C_5K4, 9, dict()), (C_5K2, 9, dict())]
STRESS_CELLS += [(C_5K4, rows, dict(form=Form(MMA_M=64, STAGES=7))) for rows in (9, 32)]
STRESS_CELLS += [(numerics, 1, dict(form=Form(MMA_M=64, STAGES=4, HOLD_TMEM=hold))) for numerics in (C_16K1, C_14K1) for hold in (True, False)]


def _stress_id(cell):
    numerics, rows, arm = cell
    form = arm.get("form")
    tail = "" if form is None else f"-m{form.MMA_M}g{form.STAGES}-{'late' if form.HOLD_TMEM else 'early'}"
    return f"{cid(numerics)}-rows{rows}{tail}"


@pytest.mark.parametrize("cell", STRESS_CELLS, ids=_stress_id)
def test_warm_stress_of_the_classes_run_as_written(cell):
    """The warm stress with byte references (``warm_stress.py``): a cold reference behind a cache flush that equals the host's bytes
    (exact-integer operands), then warm launches back to back and launches each behind a kernel that overwrites every SM's shared
    memory, every output byte compared after every launch -- behind the positive control of this GPU; where the control does not
    reproduce, the cell is skipped as not validated.  The stage counted as refilled is the operand ring (a K slice is longer than the
    ring at every one of these forms)."""
    from warm_stress import StressCase, assert_warm_stress

    dev = device_or_skip()
    numerics, rows, arm = cell
    k = _entry(dev, numerics, **arm)
    cfg = k.cfg_for(rows)
    ops = _ops(dev, numerics.GEOM, integer=True)
    act = ops["activation"][:rows].contiguous()
    exact = act.float().double() @ ops["weight"].float().double().t()
    want = (exact * ops["scale"].double().reshape(())).float().to(torch.bfloat16).contiguous()
    out = new_out(numerics.GEOM, rows, dev)
    k.compile(rows)
    report = assert_warm_stress(
        StressCase(
            f"{entry.NAME}: {cfg.describe()}, tensor memory released {'late' if cfg.HOLD_TMEM else 'early'}",
            lambda: k(ops["weight"], act, ops["scale"], out),
            {"out": out},
            thread_blocks=cfg.CTAS,
            refills_per_launch=cfg.K_TILES_PER_SLICE - cfg.STAGES,
            reference={"out": want},
        ),
        warm=1024,
        poisoned=1024,
    )
    print(report.text())
