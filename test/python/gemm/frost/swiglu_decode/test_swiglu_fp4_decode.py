# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate / up product of an FP4 MLP for decode rows with SiLU(gate) x up and the FP4 re-quantisation in one launch
(``cudnn.SwigluFp4Decode``).  Host: the config's ledgers and derived constants (the stage rule's mirror, the shared-memory and
tensor-memory ledgers, the barrier counts), the token-tile rule and the form text that heads the kernel symbol; the typed declines of
the host entry (CPU tensors), the DSL gate, the load-time row interleave, the plan-time compile key, the launch facts and the lazy
exports, and the rendering pin (a host trace-compile per claimed target).  On a cc 10.x device: the entry against an fp64 reference
under a derived bound, exact cells byte for byte against a host-built expectation, the edge rows, the bytes it must not touch, every
form and knob arm bitwise with the default, CUDA-graph replay, a dependent launch that waits behind the entry and copies its outputs
(``swiglu_fp4_decode_dependent.py``), the plan-time compile, the warm call (a cache hit that reads nothing to the host and allocates
nothing), the launch facts against the profiler's record, the launch matrix in a child process with a stage-aware deadline, and a
warm stress with byte references.  A child interpreter runs the source overlay of this process and reports the files it loaded; the
parent compares them.  The operand builders, the reference and the device gate live in ``swiglu_fp4_decode_utils.py`` beside this
module; replay, batch invariance per row count and the goldens live in the determinism suite.

Marked ``no_workspace_shim``: the engine suite's autouse fixture would otherwise import torch and both architecture compilers for
every test of this direct-call entry."""

import importlib
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import fields, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

import swiglu_fp4_decode_utils as U  # noqa: E402  (the operand builders, the reference and the device gate)
from launch_records import assert_facts_equal_the_launch  # noqa: E402
from warm_stress import StressCase, assert_warm_stress, build_launchers, stress_validation  # noqa: E402

from cudnn.frost.launch_facts import LaunchShape  # noqa: E402
from cudnn.gemm.frost.swiglu_decode import swiglu_fp4_decode as entry  # noqa: E402
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


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the entry's typed declines (CPU tensors; the compute capability is given)
# ---------------------------------------------------------------------------------------------------------------------------
ARG_ORDER = ("weight", "weight_scale", "activation", "activation_scale", "alpha", "out_global_scale", "out", "out_scale")


def _host_ops(geometry=G_SMALL, rows=3):
    """The eight launch arguments on the host: zero codes, valid byte counts and alignments."""
    h, i = geometry.HIDDEN, geometry.INTERMEDIATE
    return dict(
        weight=torch.zeros(2 * i, h // 2, dtype=torch.uint8),
        weight_scale=torch.zeros(2 * i * h // 16, dtype=torch.uint8),
        activation=torch.zeros(rows, h // 2, dtype=torch.uint8),
        activation_scale=torch.zeros(128 * h // 16, dtype=torch.uint8),
        alpha=torch.ones(1, dtype=torch.float32),
        out_global_scale=torch.ones(1, dtype=torch.float32),
        out=torch.zeros(rows, i // 2, dtype=torch.uint8),
        out_scale=torch.zeros(128 * i // 16, dtype=torch.uint8),
    )


def _check(ops, **kw):
    kw.setdefault("device_cc", (10, 0))
    return entry.check_support(*(ops[name] for name in ARG_ORDER), **kw)


def _misaligned(nbytes, offset=1):
    """A contiguous uint8 tensor whose storage starts one byte off an aligned address."""
    return torch.frombuffer(bytearray(nbytes + 64), dtype=torch.uint8, offset=offset, count=nbytes)


def test_served_operands_pass_the_host_checks():
    for rows in (1, 3, 64):
        assert _check(_host_ops(rows=rows)) == rows
    ops = _host_ops()
    ops["alpha"], ops["out_global_scale"] = torch.tensor(0.5), torch.tensor(3.0)  # 0-d scalars are one element
    assert _check(ops) == 3
    ops = _host_ops()
    for name in ("weight_scale", "activation_scale", "out_scale"):  # the scale blobs as E4M3 tensors of any shape
        ops[name] = ops[name].view(torch.float8_e4m3fn).reshape(-1, 4)
    assert _check(ops, geometry=G_SMALL) == 3
    ops["activation_scale"] = torch.zeros(3 * 128 * 5120 // 16, dtype=torch.uint8)  # more than one atom band is accepted
    assert _check(ops) == 3


DECLINES = [
    (lambda o: o.update(weight=[1, 2]), TypeError, "weight must be a torch.Tensor, got list"),
    (lambda o: o.update(weight=o["weight"].float()), TypeError, "weight must be torch.uint8"),
    (lambda o: o.update(activation=o["activation"].to(torch.int8)), TypeError, "activation must be torch.uint8"),
    (lambda o: o.update(out=o["out"].to(torch.bfloat16)), TypeError, "out must be torch.uint8"),
    (lambda o: o.update(weight_scale=o["weight_scale"].float()), TypeError, "weight_scale must be torch.float8_e4m3fn or torch.uint8"),
    (lambda o: o.update(out_scale=o["out_scale"].to(torch.int8)), TypeError, "out_scale must be torch.float8_e4m3fn or torch.uint8"),
    (lambda o: o.update(alpha=o["alpha"].double()), TypeError, "alpha must be torch.float32"),
    (lambda o: o.update(out_global_scale=o["out_global_scale"].to(torch.bfloat16)), TypeError, "out_global_scale must be torch.float32"),
    (lambda o: o.update(weight=torch.zeros(256, 2559, dtype=torch.uint8)), ValueError, r"weight must be \[256, 2560\] at hidden 5120, intermediate 128"),
    (lambda o: o.update(activation=torch.zeros(3, 2000, dtype=torch.uint8)), ValueError, r"activation must be \[rows, 2560\]"),
    (lambda o: o.update(activation=torch.zeros(3 * 2560, dtype=torch.uint8)), ValueError, r"activation must be \[rows, 2560\]"),
    (lambda o: o.update(activation=torch.zeros(0, 2560, dtype=torch.uint8)), ValueError, "rows must be a positive int"),
    (lambda o: o.update(activation=torch.zeros(65, 2560, dtype=torch.uint8)), NotImplementedError, "at most 64 rows per call"),
    (lambda o: o.update(out=torch.zeros(4, 64, dtype=torch.uint8)), ValueError, r"out must be exactly \[3, 64\]"),
    (lambda o: o.update(out=torch.zeros(3, 128, dtype=torch.uint8)), ValueError, r"out must be exactly \[3, 64\]"),
    (lambda o: o.update(alpha=torch.ones(2)), ValueError, "alpha must hold one fp32 element"),
    (lambda o: o.update(out_global_scale=torch.ones(0)), ValueError, "out_global_scale must hold one fp32 element"),
    (lambda o: o.update(weight_scale=torch.zeros(81919, dtype=torch.uint8)), ValueError, "weight_scale must hold exactly 81920 bytes"),
    (lambda o: o.update(weight_scale=torch.zeros(81921, dtype=torch.uint8)), ValueError, "weight_scale must hold exactly 81920 bytes"),
    (
        lambda o: o.update(activation_scale=torch.zeros(3 * 320, dtype=torch.uint8)),
        ValueError,
        r"activation_scale must hold at least 40960 bytes \(rows padded to 128",
    ),
    (lambda o: o.update(out_scale=torch.zeros(3 * 8, dtype=torch.uint8)), ValueError, r"out_scale must hold at least 1024 bytes \(rows padded to 128"),
    (lambda o: o.update(activation=torch.zeros(3, 5120, dtype=torch.uint8)[:, ::2]), NotImplementedError, "serves contiguous operands; activation has shape"),
    (lambda o: o.update(out=torch.zeros(64, 3, dtype=torch.uint8).t()), NotImplementedError, "serves contiguous operands; out has shape"),
    (lambda o: o.update(weight=_misaligned(256 * 2560).reshape(256, 2560)), ValueError, "weight must start at a 32-byte aligned address"),
    (lambda o: o.update(out=_misaligned(3 * 64).reshape(3, 64)), ValueError, "out must start at a 16-byte aligned address"),
    (lambda o: o.update(activation_scale=_misaligned(40960)), ValueError, "activation_scale must start at a 16-byte aligned address"),
    (lambda o: o.update(alpha=_misaligned(4).view(torch.float32)), ValueError, "alpha must start at a 4-byte aligned address"),
]


@pytest.mark.parametrize("mutation, exc, match", DECLINES, ids=[d[2][:44].replace(" ", "_") for d in DECLINES])
def test_typed_declines_name_the_operand(mutation, exc, match):
    ops = _host_ops()
    mutation(ops)
    before = {name: t.clone() for name, t in ops.items() if isinstance(t, torch.Tensor)}
    with pytest.raises(exc, match=match):
        _check(ops, geometry=G_SMALL)
    for name, t in before.items():  # a decline touches nothing
        assert torch.equal(ops[name], t), name


def test_declines_an_unserved_geometry_and_compute_capability_by_name():
    with pytest.raises(NotImplementedError, match="serves hidden 5120, intermediate 17408 or hidden 5120, intermediate 128"):
        _check(_host_ops(Geometry(4096, 128)))
    with pytest.raises(NotImplementedError, match="got hidden 5120, intermediate 256"):
        _check(_host_ops(Geometry(5120, 256)))
    with pytest.raises(TypeError, match="geometry must be"):
        _check(_host_ops(), geometry=(5120, 128))
    for cc in ((12, 0), (9, 0), (8, 0)):
        with pytest.raises(NotImplementedError, match=rf"compute capability 10\.x \(SM100-class\); this device is {cc[0]}\.{cc[1]}"):
            _check(_host_ops(), device_cc=cc)
    for cc in ((10, 0), (10, 3), (10, 7)):
        assert _check(_host_ops(), device_cc=cc) == 3
    with pytest.raises(NotImplementedError, match="serves CUDA tensors only; the operands are on cpu"):
        _check(_host_ops(), device_cc=None)
    with pytest.raises(ValueError, match="every operand must live on one device, the entry's"):
        _check(_host_ops(), device="cuda:0")


def _modelled_cuda_entry(monkeypatch, readings=(233472, 232448, 148), cc=(10, 0), **kw):
    """An entry for ``cuda:0`` on MODELLED part facts -- ``readings`` = (shared memory per SM, per-CTA opt-in, multiprocessors) and the
    compute capability -- so the declines of a part are reachable on any host; no device is touched (construction is host work)."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)
    monkeypatch.setattr(entry, "device_readings", lambda device: readings)
    return entry.SwigluFp4Decode("cuda:0", **kw)


def test_an_unserved_cuda_part_declines_at_construction_by_name(monkeypatch):
    for cc in ((12, 0), (9, 0), (8, 0)):
        with pytest.raises(NotImplementedError, match=rf"10\.x.*{cc[0]}\.{cc[1]}"):
            _modelled_cuda_entry(monkeypatch, cc=cc)
    for cc in ((10, 0), (10, 3), (10, 7)):
        assert _modelled_cuda_entry(monkeypatch, cc=cc).device_cc == cc


def test_declines_a_part_below_the_footprint(monkeypatch):
    """A part whose per-block shared memory cannot hold a form is a ``NotImplementedError`` at construction (a part fact): on a host
    entry through modelled readings in the config, on a CUDA entry through the part's own."""
    with pytest.raises(NotImplementedError, match="does not serve this part at token tile 8: .* holds 231424 B .* per-block maximum is 200000 B"):
        entry.SwigluFp4Decode("cpu", cfg=Cfg(SMEM_OPTIN_BYTES=200000))
    with pytest.raises(NotImplementedError, match="does not serve this part at token tile 8"):
        _modelled_cuda_entry(monkeypatch, readings=(102400, 101376, 170))
    k = _modelled_cuda_entry(monkeypatch, readings=(233472, 232448, 212), cc=(10, 7))
    assert (k.cfg.SMEM_PER_SM_BYTES, k.cfg.SMEM_OPTIN_BYTES, k.cfg.SM_COUNT) == (233472, 232448, 212), "the part's readings replace the config's"
    assert [kcfg.stages(f) for f in k.forms.values()] == [(5, 7), (4, 12), (4, 4), (3, 7)], "the stage rule does not read the part"


def test_constructor_declines_by_type_and_by_name():
    with pytest.raises(TypeError, match="cfg must be"):
        entry.SwigluFp4Decode("cpu", cfg=dict(TOKEN_TILE=8))
    with pytest.raises(TypeError, match="geometry must be"):
        entry.SwigluFp4Decode("cpu", geometry=(5120, 17408))
    with pytest.raises(NotImplementedError, match="serves hidden 5120, intermediate 17408 or"):
        entry.SwigluFp4Decode("cpu", geometry=Geometry(5120, 4096))
    with pytest.raises(ValueError, match="give one source"):
        entry.SwigluFp4Decode("cpu", geometry=G_SMALL, cfg=Cfg())
    for bad, match in ((dict(TOKEN_TILE=12), "Cfg.TOKEN_TILE"), (dict(PREWAIT_STAGES=1), "Cfg.PREWAIT_STAGES"), (dict(MAX_CLUSTERS=0), "Cfg.MAX_CLUSTERS")):
        with pytest.raises(ValueError, match=match):
            entry.SwigluFp4Decode("cpu", cfg=replace(Cfg(), **bad))
    k = entry.SwigluFp4Decode("cpu", cfg=Cfg(GEOM=G_SMALL))
    assert k.geometry == G_SMALL and k.device.type == "cpu" and k.device_cc is None
    with pytest.raises(NotImplementedError, match="serves CUDA devices only; this entry was built for cpu"):
        k.compile()
    with pytest.raises(NotImplementedError, match="serves CUDA devices only"):
        k.compile(3)


def test_compile_declines_by_name_before_the_kernel_module_is_imported(monkeypatch):
    """The DSL gate at the plan-time compile of a CUDA entry: not installed, below the floor, a missing name, a cc 10.7 part with a DSL
    that has no ``sm_107a`` target, a compiler the kernel is not qualified with -- each a ``NotImplementedError`` that names what is
    missing, raised before the kernel module is imported."""
    from cudnn.frost import buffers

    k = _modelled_cuda_entry(monkeypatch)
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    with pytest.raises(NotImplementedError, match="not installed"):
        k.compile(1)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    with pytest.raises(NotImplementedError, match=r"4\.8\.0.*4\.7\.1|4\.7\.1.*4\.8\.0"):
        k.compile()
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "9.9.9")))
    monkeypatch.setattr(entry, "dsl_compiler_version", lambda: (13, 4))
    monkeypatch.setattr(entry, "DSL_NAMES", (("cutlass_module_that_does_not_exist", ("x",)),))
    with pytest.raises(NotImplementedError, match=r"cutlass_module_that_does_not_exist, which does not import .*9\.9\.9"):
        k.compile(1)
    monkeypatch.setattr(entry, "DSL_NAMES", (("os", ("path", "no_such_name_in_this_module")),))
    with pytest.raises(NotImplementedError, match=r"written on os; .*9\.9\.9.* does not export no_such_name_in_this_module"):
        k.compile(1)
    part107 = _modelled_cuda_entry(monkeypatch, readings=(233472, 232448, 212), cc=(10, 7))
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        part107.compile(1)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(entry, "dsl_compiler_version", lambda: (12, 9))
    for e in (k, part107):  # the compiler floor is the kernel's (its bytes), not a part's: it declines on every class
        with pytest.raises(
            NotImplementedError, match=r"is qualified with the CuTe DSL compiling with CUDA 13\.4 or later; the DSL in this process compiles with CUDA 12\.9"
        ):
            e.compile(1)
    assert not k.compiled and not part107.compiled and k._kernel_module is None and part107._kernel_module is None


def test_compiler_gate_never_declines_on_a_value_it_cannot_read(monkeypatch):
    assert kcfg.QUALIFIED_MIN_COMPILER == (13, 4)
    for compiler in ((13, 3), (13, 0), (12, 9)):
        monkeypatch.setattr(entry, "dsl_compiler_version", lambda c=compiler: c)
        msg = entry.dsl_compiler_error()
        assert "qualified" in msg and "13.4" in msg and f"{compiler[0]}.{compiler[1]}" in msg and "nvidia-cutlass-dsl[cu13]" in msg
    for compiler in ((13, 4), (13, 6), (14, 0), None):
        monkeypatch.setattr(entry, "dsl_compiler_version", lambda c=compiler: c)
        assert entry.dsl_compiler_error() is None, compiler


def test_installed_dsl_provides_the_names_or_declines_by_name():
    """On the installed DSL the gate either passes (every probed name imports) or returns a message; it never raises."""
    msg = entry.dsl_error()
    assert msg is None or isinstance(msg, str)
    names = [(module, n) for module, ns in entry.DSL_NAMES for n in ns]
    assert len(names) == len(set(names)) and ("cudnn._cutlass_compat", "SmemAllocator") in names and ("cutlass.pipeline", "PipelineTmaUmma") in names


_ENGINE_SKIP = re.compile(
    r"runs only on \d+ <= SM < \d+.* but the active GPU is sm_\d+" r"|is served by the sm\d+ arch tree, but this process runs the sm\d+ tree"
)


def test_no_decline_reads_as_an_engine_skip(monkeypatch):
    """This directory's conftest reports two engine messages as skips.  No decline of this entry may match them: a typed decline of a
    direct-call entry is a failure of the test that hit it, never a skip."""
    conftest = (Path(__file__).resolve().parents[1] / "conftest.py").read_text()
    assert "runs only on" in conftest and "arch tree, but this process runs" in conftest, "the engine conftest's skip pattern moved: re-read it"
    messages = []
    for mutation, exc, _ in DECLINES:
        ops = _host_ops()
        mutation(ops)
        with pytest.raises(exc) as info:
            _check(ops, geometry=G_SMALL)
        messages.append(str(info.value))
    for cc in ((12, 0), (8, 0)):
        with pytest.raises(NotImplementedError) as info:
            _check(_host_ops(), device_cc=cc)
        messages.append(str(info.value))
    with pytest.raises(NotImplementedError) as info:
        entry.SwigluFp4Decode("cpu", cfg=Cfg(SMEM_OPTIN_BYTES=200000))
    messages.append(str(info.value))
    monkeypatch.setattr(entry, "dsl_compiler_version", lambda: (12, 9))
    messages.append(entry.dsl_compiler_error())
    assert len(messages) > 30 and not [m for m in messages if _ENGINE_SKIP.search(m)]


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the load-time row interleave
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n, k", [(256, 5120), (384, 128)], ids=["two-tiles-hidden-5120", "three-tiles-hidden-128"])
def test_interleave_gate_up_is_a_byte_permutation(n, k):
    """Against an index-built expectation: interleaved row ``r`` is natural row ``interleave_index(n)[r]`` -- the codes as they are, the
    scale bytes at the atom positions of the new rows -- and an index-built inverse restores the inputs."""
    gen = torch.Generator().manual_seed(n + k)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    linear = torch.randint(0, 256, (n, k // 16), dtype=torch.uint8, generator=gen)
    scales = U.to_atoms(linear)
    for r, b in ((0, 0), (37, 5), (130, k // 16 - 1), (n - 1, 3)):  # the documented byte rule of the atom order
        assert int(scales[U.atom_index(r, b, k // 16)]) == int(linear[r, b])
    keep = (weight.clone(), scales.clone())
    w2, s2 = entry.interleave_gate_up(weight, scales.view(torch.float8_e4m3fn))
    index = U.interleave_index(n)
    assert index[:64].tolist() == list(range(64)) and index[64:128].tolist() == list(range(n // 2, n // 2 + 64)) and sorted(index.tolist()) == list(range(n))
    assert w2.dtype == torch.uint8 and s2.dtype == torch.uint8 and w2.shape == weight.shape and s2.shape == (n * k // 16,)
    assert torch.equal(w2, weight[index]) and torch.equal(U.from_atoms(s2, n, k // 16), linear[index])
    inverse = torch.empty_like(index)
    inverse[index] = torch.arange(n)
    assert torch.equal(w2[inverse], weight) and torch.equal(U.to_atoms(U.from_atoms(s2, n, k // 16)[inverse]), scales)
    assert torch.equal(weight, keep[0]) and torch.equal(scales, keep[1]), "the inputs are not modified"
    assert w2.data_ptr() != weight.data_ptr() and s2.data_ptr() != scales.data_ptr(), "the results are new tensors"


@pytest.mark.parametrize(
    "weight, scale, exc, match",
    [
        (lambda: [1], lambda: torch.zeros(1, dtype=torch.uint8), TypeError, "weight must be a torch.Tensor"),
        (lambda: torch.zeros(256, 64, dtype=torch.uint8), lambda: None, TypeError, "weight_scale must be a torch.Tensor"),
        (lambda: torch.zeros(256, 64), lambda: torch.zeros(2048, dtype=torch.uint8), TypeError, "weight must be torch.uint8"),
        (lambda: torch.zeros(256, 64, dtype=torch.uint8), lambda: torch.zeros(2048), TypeError, "weight_scale must be torch.float8_e4m3fn or torch.uint8"),
        (lambda: torch.zeros(256 * 64, dtype=torch.uint8), lambda: torch.zeros(2048, dtype=torch.uint8), ValueError, "weight must be"),
        (lambda: torch.zeros(200, 64, dtype=torch.uint8), lambda: torch.zeros(1600, dtype=torch.uint8), ValueError, "multiple of 128"),
        (lambda: torch.zeros(256, 40, dtype=torch.uint8), lambda: torch.zeros(1280, dtype=torch.uint8), ValueError, "columns a multiple of 64"),
        (
            lambda: torch.zeros(256, 64, dtype=torch.uint8),
            lambda: torch.zeros(2047, dtype=torch.uint8),
            ValueError,
            "weight_scale must hold exactly 2048 bytes",
        ),
    ],
)
def test_interleave_gate_up_declines(weight, scale, exc, match):
    with pytest.raises(exc, match=match):
        entry.interleave_gate_up(weight(), scale())


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: forms, the compile key, the launch facts, the exports
# ---------------------------------------------------------------------------------------------------------------------------
def test_entry_resolves_the_forms_at_construction():
    k = entry.SwigluFp4Decode("cpu")
    assert k.geometry == G_DEFAULT and k.pdl is False and k.floor == 8 and sorted(k.forms) == [8, 16, 32, 64]
    assert [k.cfg_for(rows).TOKEN_TILE for rows in (1, 8, 9, 16, 17, 32, 33, 64)] == [8, 8, 16, 16, 32, 32, 64, 64]
    assert all(k.cfg_for(rows) is k.forms[kcfg.token_tile_for(rows)] for rows in range(1, 65)), "a served row count never resolves on the execute path"
    wide = entry.SwigluFp4Decode("cpu", cfg=Cfg(TOKEN_TILE=32))
    assert sorted(wide.forms) == [32, 64] and [wide.cfg_for(rows).TOKEN_TILE for rows in (1, 9, 32, 33, 64)] == [32, 32, 32, 64, 64]
    for bad in (0, -3, 2.0, True, None):
        with pytest.raises(ValueError, match="rows must be a positive int"):
            k.cfg_for(bad)
    with pytest.raises(NotImplementedError, match="at most 64 rows per call"):
        k.cfg_for(65)
    with pytest.raises(ValueError, match="one token tile per call"):
        kcfg.check_rows_fit(k.forms[8], 9)


def test_launch_pairs_and_the_pdl_knob():
    """All four (attribute, pre-wait) pairs are admitted; ``pdl`` overrides the config's field; both are part of the compile key."""
    keys = set()
    for pdl in (False, True):
        for prewait in (0, 2):
            k = entry.SwigluFp4Decode("cpu", cfg=Cfg(PREWAIT_STAGES=prewait), pdl=pdl)
            assert (k.pdl, k.cfg_for(1).PDL, k.cfg_for(1).PREWAIT_STAGES) == (pdl, pdl, prewait)
            keys.add(k.key(1))
    assert len(keys) == 4
    assert entry.SwigluFp4Decode("cpu").pdl is False and entry.SwigluFp4Decode("cpu", cfg=Cfg(PDL=True)).pdl is True
    assert entry.SwigluFp4Decode("cpu", cfg=Cfg(PDL=True), pdl=False).cfg_for(64).PDL is False


def test_compile_key_is_plan_time_only():
    k = entry.SwigluFp4Decode("cpu")
    assert len({k.key(rows) for rows in range(1, 9)}) == 1 and len({k.key(rows) for rows in range(1, 65)}) == 4
    assert k.key(3) == (k.cfg_for(3),) and all(isinstance(part, Cfg) for part in k.key(3)), "a config of plan-time facts; no row count, no pointer, no shape"
    assert k.key(3) != entry.SwigluFp4Decode("cpu", cfg=Cfg(L2_EVICT_FIRST=False)).key(3)
    assert k.key(3) != entry.SwigluFp4Decode("cpu", cfg=Cfg(MAX_CLUSTERS=37)).key(3)
    assert k.key(3) != entry.SwigluFp4Decode("cpu", geometry=G_SMALL).key(3)


@pytest.mark.parametrize(
    "arm",
    [dict(), dict(pdl=True), dict(cfg=Cfg(TOKEN_TILE=32, PREWAIT_STAGES=0, MAX_CLUSTERS=37)), dict(geometry=G_SMALL)],
    ids=["default", "pdl", "wide-tile-cap-37", "small"],
)
def test_launch_facts_equal_the_configuration_arithmetic_at_every_row_count(arm):
    k = entry.SwigluFp4Decode("cpu", **arm)
    for rows in range(1, 65):
        cfg = k.cfg_for(rows)
        tile = max(k.floor, next(t for t in TILES if rows <= t))
        assert k.launch_shape_for(rows) == LaunchShape((1, 1, min(cfg.GEOM.WEIGHT_TILES, cfg.MAX_CLUSTERS)), (192, 1, 1), FIGURES[tile][2])
        form = k.form_for(rows)
        assert list(form) == [f.name for f in fields(Cfg)] and form["TOKEN_TILE"] == tile and form["PDL"] is bool(arm.get("pdl", False))
        assert form["GEOM"] == dict(HIDDEN=cfg.GEOM.HIDDEN, INTERMEDIATE=cfg.GEOM.INTERMEDIATE)
        assert k.kernel_symbols_for(rows) == (), "nothing is compiled on a host entry"


def test_launch_facts_are_plain_fresh_and_decline_an_unserved_row_count():
    k = entry.SwigluFp4Decode("cpu")
    a, b = k.form_for(3), k.form_for(3)
    assert a == b and a is not b and a["GEOM"] is not b["GEOM"]
    a["TOKEN_TILE"] = 64
    assert k.form_for(3)["TOKEN_TILE"] == 8 and k.cfg_for(3).TOKEN_TILE == 8, "a form is a copy: editing it changes nothing"
    assert type(k.launch_shape_for(3)) is LaunchShape and k.launch_shape_for(3) == ((1, 1, 148), (192, 1, 1), 231424)
    for fact in (k.kernel_symbols_for, k.form_for, k.launch_shape_for):
        for bad in (0, -1, 1.5, True, "3", None):
            with pytest.raises(ValueError, match="rows must be a positive int"):
                fact(bad)
        with pytest.raises(NotImplementedError, match="at most 64 rows per call"):
            fact(65)
        with pytest.raises(TypeError):
            fact()


def test_cudnn_level_export_resolves_to_the_entry():
    assert cudnn.SwigluFp4Decode is entry.SwigluFp4Decode
    gemm, pkg = importlib.import_module("cudnn.gemm"), importlib.import_module("cudnn.gemm.frost.swiglu_decode")
    assert gemm.SwigluFp4Decode is entry.SwigluFp4Decode and gemm.SwigluGeometry is Geometry
    assert pkg.SwigluFp4Decode is entry.SwigluFp4Decode and pkg.Geometry is Geometry and pkg.Cfg is Cfg and pkg.interleave_gate_up is entry.interleave_gate_up
    assert "SwigluFp4Decode" in cudnn._LAZY_OPTIONAL_IMPORTS and "SwigluFp4Decode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS
    assert {"SwigluFp4Decode", "SwigluGeometry"} <= set(gemm.__all__) and "interleave_gate_up" not in gemm.__all__ and "SwigluCfg" not in gemm.__all__


REPO = Path(__file__).resolve().parents[5]  # <repo>/test/python/gemm/frost/swiglu_decode/<this file>


def test_docs_rows_exist_for_the_api():
    """The determinism status page carries the API's row -- SM90 unsupported (declined by type); SM100/103 verified on cc 10.0 only,
    so the cell names the cc 10.3 gap inside the shared column; SM107 unverified (not run on cc 10.7) -- and its required-settings
    bullet; the goldens README carries the suite's section; the operation page, its navigation entry, the overview line and the
    ``llms.txt`` line exist; the page's tables name exactly the geometries, the token tiles and the config fields the entry has;
    and the package guide's layout names the subpackage."""
    det = (REPO / "docs" / "operations" / "determinism.md").read_text()
    row = [line for line in det.splitlines() if line.startswith("| `SwigluFp4Decode`")]
    assert len(row) == 1, "one SwigluFp4Decode row on the status page"
    cells = [c.strip() for c in row[0].strip("|").split("|")]
    assert cells[1] == "fprop (inference step)" and cells[3] == "Unsupported", cells
    assert cells[4] == "SM100: Supported; SM103: Unverified" and cells[5] == "Unverified", cells
    assert "SwiGLU FP4 gate / up product for decode rows (`SwigluFp4Decode`): no" in det
    readme = (REPO / "test" / "python" / "determinism_goldens" / "README.md").read_text()
    assert "`cudnn.SwigluFp4Decode`" in readme and "test_swiglu_fp4_decode_determinism.py" in readme
    assert "### SwiGLU FP4 gate / up product for decode rows" in readme
    page = REPO / "docs" / "operations" / "gemm_fusions" / "swiglu_fp4_decode.md"
    assert page.exists()
    text = page.read_text()
    for geometry in kcfg.SERVED_GEOMETRIES:
        assert f"`SwigluGeometry({geometry.HIDDEN}, {geometry.INTERMEDIATE})`" in text, geometry
    table = [[c.strip() for c in line.strip("|").split("|")] for line in text.splitlines() if line.startswith("| ")]
    knobs = [cells[0].strip("`") for cells in table if cells[0].strip("`") in kcfg.PERFORMANCE_FIELDS]
    assert tuple(knobs) == kcfg.PERFORMANCE_FIELDS, knobs
    forms = [cells for cells in table if re.fullmatch(r"\d+ \.\. \d+", cells[0])]
    assert len(forms) == len(TILES)
    for tile, cells in zip(TILES, forms):
        operand, staging, smem, _, tmem, _ = FIGURES[tile]
        lo = 1 if tile == TILES[0] else TILES[TILES.index(tile) - 1] + 1
        assert cells == [f"{lo} .. {tile}", str(tile), str(operand), str(staging), f"{smem:,} B", str(tmem)], cells
    for token in ("rows 1 .. 64", "compute capability 10.x", f"nvidia-cutlass-dsl >= {entry.DSL_FLOOR_TEXT}", "interleave_gate_up", "FlashInfer"):
        assert token in text, token
    assert f"CUDA {kcfg.QUALIFIED_MIN_COMPILER[0]}.{kcfg.QUALIFIED_MIN_COMPILER[1]}" in text
    assert "swiglu_fp4_decode" in (REPO / "docs" / "fern" / "versions" / "latest.yml").read_text()
    assert "swiglu_fp4_decode" in (REPO / "docs" / "operations" / "overview.md").read_text()
    assert "swiglu_fp4_decode" in (REPO / "llms.txt").read_text()
    assert "swiglu_decode/" in (REPO / "python" / "cudnn" / "AGENTS.md").read_text()


# Child interpreters.  A child is a fresh process: nothing of this process's import setup carries over, so ``import cudnn`` there is the
# installed package, while this process may run the entry from a source tree laid over that package.  A child that imports the entry
# therefore runs CHILD_TREE first -- the same overlay, with the source directory THIS process loaded the entry from -- and every child
# prints the files it loaded on ``LOADED`` lines, which the parent compares with its own.
SOURCE_CUDNN = str(Path(entry.__file__).parents[3])  # <...>/cudnn of <...>/cudnn/gemm/frost/swiglu_decode/<the entry module>
KERNEL_FILE = str(Path(kcfg.__file__).with_name("swiglu_fp4_decode_nvfp4.py"))
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
    """The lazy boundary in fresh interpreters: importing cudnn loads nothing of this package and neither torch nor the DSL."""
    code = "import sys, cudnn; print('LOADED cudnn', cudnn.__file__)\n"
    code += "print(sorted(m for m in sys.modules if 'swiglu_decode' in m or m in ('cutlass', 'torch')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded_files(out) == {"cudnn": cudnn.__file__}, f"the child imported another cudnn package than this process:\n{out}"
    assert out.strip().splitlines()[-1] == "[]", out


def test_host_entry_and_decline_do_not_import_cutlass():
    """Building a host entry, reading its launch facts and running a decline loads neither the kernel module nor the DSL -- also not
    through the packages above the entry (an eager import anywhere in that chain would show here)."""
    code = CHILD_TREE + textwrap.dedent("""
        import sys, cudnn
        from cudnn.gemm import SwigluGeometry
        k = cudnn.SwigluFp4Decode("cpu", SwigluGeometry(5120, 128))
        k.form_for(3), k.launch_shape_for(64), k.kernel_symbols_for(1)
        try:
            k.compile()
        except NotImplementedError:
            pass
        print("LOADED entry", sys.modules[type(k).__module__].__file__)
        print(sorted(m for m in sys.modules if m == "cutlass" or m.startswith("cutlass.") or m.endswith("swiglu_fp4_decode_nvfp4")))
        """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded_files(out) == {"entry": entry.__file__}, f"the child did not load the entry of the tree under test:\n{out}"
    assert out.strip().splitlines()[-1] == "[]", out


# ---------------------------------------------------------------------------------------------------------------------------
# host: the rendering pin (a trace-compile per claimed target; no device)
# ---------------------------------------------------------------------------------------------------------------------------
_RENDER_PROBE = textwrap.dedent("""
    import json, os, sys
    dump, arch, source = sys.argv[1], sys.argv[2], sys.argv[3]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_ARCH"] = arch              # unconditional: an inherited value would pin another target's text
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import importlib
    for name in ("cudnn", "cudnn.gemm"):
        package, path = importlib.import_module(name), os.path.join(source, *name.split(".")[1:])
        if path not in package.__path__:
            package.__path__.insert(0, path)
    from cutlass.base_dsl.version_info import CUDA_VERSION
    from cudnn.gemm.frost.swiglu_decode.kernel import swiglu_fp4_decode_config as kcfg
    from cudnn.gemm.frost.swiglu_decode.kernel import swiglu_fp4_decode_nvfp4 as kernel
    print("LOADED kernel", kernel.__file__)
    print("COMPILER", int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    for spec in sys.argv[4:]:
        tile, prewait = (int(x) for x in spec.split(":"))
        before = set(os.listdir(dump))
        cfg = kcfg.Cfg(TOKEN_TILE=tile, PREWAIT_STAGES=prewait)
        compiled = kernel.compile_kernel(cfg)        # refuses a trace whose layout differs from the config's mirror
        print("SYMBOLS", spec, json.dumps(list(compiled.kernel_info)))
        for name in sorted(set(os.listdir(dump)) - before):
            print("KEPT", spec, name)
    print("COMPILED", arch)
    """)


def _render(tmp_path, arch, specs):
    """Run the probe for ``arch`` in a fresh process (the DSL reads its dump settings once, at the first import); returns the
    compiler's CUDA version, ``{spec: [kept file paths]}`` and ``{spec: [kernel symbols]}``.  A compile failure FAILS."""
    probe = tmp_path / "render_probe.py"  # a FILE, not ``python -c``: the DSL preprocessor re-reads a kernel's source through inspect
    probe.write_text(_RENDER_PROBE)
    dump = tmp_path / "dump"
    dump.mkdir()
    proc = subprocess.run([sys.executable, str(probe), str(dump), arch, SOURCE_CUDNN, *specs], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"trace-compile for {arch} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED {arch}" in lines, proc.stdout[-2000:]
    assert loaded_files(proc.stdout) == {"kernel": KERNEL_FILE}, f"the child did not render the kernel of the tree under test:\n{proc.stdout[-2000:]}"
    compiler = tuple(int(x) for x in next(ln for ln in lines if ln.startswith("COMPILER")).split()[1:])
    kept, symbols = {}, {}
    for ln in lines:
        if ln.startswith("KEPT "):
            _, spec, name = ln.split(" ", 2)
            kept.setdefault(spec, []).append(str(dump / name))
        elif ln.startswith("SYMBOLS "):
            _, spec, text = ln.split(" ", 2)
            symbols[spec] = __import__("json").loads(text)
    return compiler, kept, symbols


def _sass(cubin):
    """The disassembly of a cubin by the first nvdisasm that decodes it, or None (a toolkit older than the target decodes nothing)."""
    cands = [os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm")] if os.environ.get("CUDA_PATH") else []
    cands += [c for c in [shutil.which("nvdisasm")] if c]
    for nvd in dict.fromkeys(c for c in cands if os.path.isfile(c) and os.access(c, os.X_OK)):
        try:
            proc = subprocess.run([nvd, "-c", cubin], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.splitlines()
    return None


#: PTX statements per epilogue pass: one pass at token tiles 8 and 16, two at 32 and 64 (a 32-token sub-tile is two passes of 16 rows)
_PASSES = {8: 1, 16: 1, 32: 2, 64: 2}
#: the first bf16 rounding: packed multiplies by alpha and packed casts per sub-tile read (4 / 8 / 16 values per thread)
_FIRST_ROUNDING = {8: 4, 16: 8, 32: 16, 64: 16}


@pytest.mark.parametrize("arch, cc", (("sm_100a", (10, 0)), ("sm_103a", (10, 3)), ("sm_107a", (10, 7))))
def test_rendering_pins_the_convention(arch, cc, tmp_path):
    """Trace-compile the four token-tile forms (and the 8-token form without pre-wait stages) for a claimed target on this box -- fake
    tensors, no device -- and pin the statements that ARE the contract.  Per epilogue pass: eight ``ex2.approx.ftz.f32`` and eight
    ``div.approx.f32`` (the SiLU), eight ``cvt.rn.bf16.f32`` (the second rounding), four ``cvt.rn.satfinite.e2m1x2.f32`` (eight codes),
    one ``cvt.rn.satfinite.e4m3x2.f32`` with its exact widening, two ``rcp.approx.ftz.f32`` and the bits of fp32(1/6) once (the block
    scale), one masked ``shfl.sync.bfly``, one 32-bit and one 8-bit global store, sixteen 16-bit shared loads.  Per form: the first
    rounding's packed multiplies and casts, no fused multiply-add, five unaligned ``barrier.sync`` and three ``bar.sync``, three
    ``griddepcontrol.wait`` (one without pre-wait stages: every warp waits at the same statement) and one ``launch_dependents``, nine tensor loads (six without the pre-wait block's three) behind four tensor-map
    prefetches, ``.reqntid 192``, and a symbol that starts with the cuDNN prefix and carries the form's text in its first 100
    characters.  The traced layout equals the config's mirror (``compile_kernel`` refuses otherwise).  With an nvdisasm that decodes
    the cubin: no ``FFMA``, the two reciprocals of each pass predicated (the zero rule is a branch, not a select).  A compile
    failure FAILS; a DSL that does not serve the target SKIPS with the gate's text."""
    msg = entry.dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    specs = ["8:2", "16:2", "32:2", "64:2", "8:0"]
    compiler, kept, symbols = _render(tmp_path, arch, specs)
    seen = set()
    for spec in specs:
        tile, prewait = (int(x) for x in spec.split(":"))
        what = f"{arch} token tile {tile} pre-wait {prewait} (compiler CUDA {compiler[0]}.{compiler[1]})"
        files = kept.get(spec, [])
        ptx_files = [f for f in files if f.endswith(".ptx")]
        assert len(ptx_files) == 1, f"{what}: expected one PTX, kept {files}"
        ptx = Path(ptx_files[0]).read_text()
        count = lambda needle: sum(ln.count(needle) for ln in ptx.splitlines())  # noqa: E731
        passes = _PASSES[tile]
        assert f".target {arch}" in ptx, what
        assert count("ex2.approx.ftz.f32") == 8 * passes and count("div.approx.f32") == 8 * passes, what
        assert count("cvt.rn.bf16.f32") == 8 * passes, what
        assert count("mul.f32x2") == _FIRST_ROUNDING[tile] and count("cvt.rn.bf16x2.f32") == _FIRST_ROUNDING[tile], what
        assert count("cvt.rn.satfinite.e2m1x2.f32") == 4 * passes, what
        assert count("cvt.rn.satfinite.e4m3x2.f32") == passes and count("cvt.rn.f16x2.e4m3x2") == passes, what
        assert count("rcp.approx.ftz.f32") == 2 * passes and count("0x3E2AAAAB") == passes, what
        assert count("shfl.sync.bfly") == passes, what
        assert count("st.global.b32") == passes and count("st.global.b8") == passes, what
        assert count("ld.shared.b16") == 16 * passes, what
        for banned in ("fma.", "div.rn", "div.full", "tanh", "rsqrt", "sqrt"):
            assert count(banned) == 0, f"{what}: {banned} in the rendering"
        assert count("barrier.sync ") == 5 and count("bar.sync") == 3, f"{what}: the five subset barriers are the unaligned form"
        assert count("griddepcontrol.wait") == (3 if prewait else 1) and count("griddepcontrol.launch_dependents") == 1, what
        assert count("cp.async.bulk.tensor") == (9 if prewait else 6) and count("prefetch.tensormap") == 4 and count("createpolicy") == 1, what
        assert count("tcgen05.mma") == 8 and count("tcgen05.cp") == 16 and count("tcgen05.ld") == 2 and count("tcgen05.commit") == 2, what
        operand, _ = kcfg.stages(Cfg(TOKEN_TILE=tile, PREWAIT_STAGES=prewait))
        assert count("mbarrier.init") == 2 * operand + 4, f"{what}: the four rings' barriers and nothing else"
        assert ".reqntid 192" in ptx, what
        (symbol,) = symbols[spec]
        head = f"cudnn_kernel_frost_swiglu_fp4_decode_h5120_i17408_t{tile}_pw{prewait}_evf_c148_nopdl_SwigluFp4DecodeKernel"
        assert symbol.startswith(head) and f".entry {symbol}" in ptx and len(head) <= 100, f"{what}: {symbol}"
        assert symbol not in seen, f"{what}: two forms share the symbol {symbol}"
        seen.add(symbol)
        cubins = [f for f in files if f.endswith(".cubin")]
        sass = _sass(cubins[0]) if cubins else None
        if sass is not None:
            assert not [ln for ln in sass if "FFMA" in ln], f"{what}: a fused multiply-add in the kernel"
            rcp = [ln for ln in sass if "MUFU.RCP" in ln]
            assert len(rcp) == 10 * passes, f"{what}: eight SiLU divisions and two reciprocals per pass, got {len(rcp)}"
            assert (
                sum(1 for ln in rcp if "@" in ln.split("MUFU.RCP")[0]) == 2 * passes
            ), f"{what}: the block scale's reciprocals are predicated by the zero rule"
            assert sum(1 for ln in sass if "MUFU.EX2" in ln) == 8 * passes, what


# ---------------------------------------------------------------------------------------------------------------------------
# device (compute capability 10.x)
# ---------------------------------------------------------------------------------------------------------------------------
_ENTRIES = {}
_WEIGHTS = {}
REFERENCE_TILES = {G_DEFAULT: (0, 135, 271), G_SMALL: (0, 1)}  # the weight tiles the reference is computed for (the channels are independent)
GEOMETRIES = [G_DEFAULT, G_SMALL]


def gid(geometry):
    return f"h{geometry.HIDDEN}i{geometry.INTERMEDIATE}"


def _entry(dev, geometry, **kw):
    key = (str(dev), geometry, tuple(sorted(kw.items())))
    if key not in _ENTRIES:
        _ENTRIES[key] = entry.SwigluFp4Decode(dev, geometry, **kw)
    return _ENTRIES[key]


def _weight(dev, geometry, seed=1, kind="random"):
    """``(weight dict in natural order, its interleaved pair)``, built once per module run."""
    key = (str(dev), geometry, seed, kind)
    if key not in _WEIGHTS:
        w = U.make_weight(geometry, dev, seed=seed, kind=kind)
        _WEIGHTS[key] = (w, U.interleaved(w))
    return _WEIGHTS[key]


def _ops(dev, geometry, rows, kind="random", seed=1, gs=3.0):
    """``(launch operands, activation dict)`` for ``rows`` rows on the module's weight."""
    weight, pair = _weight(dev, geometry)
    act = U.make_activation(geometry, rows, dev, seed=seed, kind=kind)
    return U.launch_operands(weight, act, out_global_scale=gs, pair=pair), act


def _rows_of(ops, act, index):
    """The launch operands of a sub-batch (the rows ``index`` of an activation), sharing the weight."""
    return U.sub_batch(ops, act, index)


def _assert_reference(out, out_scale, weight, act, ops, geometry, tiles=None, label=""):
    chk = U.check_against_reference(out, out_scale, weight, act, float(ops["alpha"]), float(ops["out_global_scale"]), tiles or REFERENCE_TILES[geometry])
    assert chk["scale_outside"] == 0, f"{label}: {chk['scale_outside']} of {chk['blocks']} block scales outside the admissible interval ({chk})"
    assert chk["code_ratio"] <= 1.0, f"{label}: a code is {chk['code_ratio']:.4f} of its bound away from the reference ({chk})"
    return chk


@pytest.mark.parametrize("kind", ("random", "structured"))
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 5, 8, 12, 13, 16, 17, 32, 33, 48, 49, 64))
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=gid)
def test_matches_the_fp64_reference_within_the_bound(geometry, rows, kind):
    """Every block scale inside its admissible interval and every code inside its bound against the fp64 chain on the exactly
    dequantised operands (the utils' docstring derives both); the bytes of rows and blocks the call does not own stay untouched."""
    dev = U.device_or_skip()
    k = _entry(dev, geometry)
    ops, act = _ops(dev, geometry, rows, kind, seed=rows)
    out, out_scale = U.run(k, ops)
    torch.cuda.synchronize()
    chk = _assert_reference(out, out_scale, _weight(dev, geometry)[0], act, ops, geometry, label=f"{gid(geometry)} rows {rows} {kind}")
    assert int((out_scale[~U.live_mask(rows, geometry, dev)] != 0xA5).sum()) == 0, "a scale byte outside the live rows was written"
    if kind == "structured" and rows >= 5:
        assert chk["zero_blocks"] > 0, "the structured operands reach the zero rule"


@pytest.mark.parametrize("rows", (1, 64))
def test_matches_the_fp64_reference_on_every_weight_tile(rows):
    """The reference of the cell above on all 272 weight tiles of the default geometry (sixteen tiles per comparison)."""
    dev = U.device_or_skip()
    k = _entry(dev, G_DEFAULT)
    ops, act = _ops(dev, G_DEFAULT, rows, "random", seed=100 + rows)
    out, out_scale = U.run(k, ops)
    torch.cuda.synchronize()
    for first in range(0, G_DEFAULT.WEIGHT_TILES, 16):
        _assert_reference(out, out_scale, _weight(dev, G_DEFAULT)[0], act, ops, G_DEFAULT, tiles=tuple(range(first, first + 16)), label=f"tiles {first} ..")


@pytest.mark.parametrize("pdl", (False, True))
def test_exact_cells_are_bitwise(pdl):
    """Integer operands whose every intermediate is exact: the kernel's packed codes and scale bytes equal a host-built expectation
    byte for byte -- both bf16 roundings on ties in both directions, the SiLU at 0 and where it is the identity, the sign of a zero
    product, the nibble order, the row interleave, the scale-atom positions and the row / channel map.  An all-zero row writes zero
    codes and zero scale bytes."""
    dev = U.device_or_skip()
    weight, act, alpha, gs = U.exact_case(dev)
    want_out, want_scale = U.exact_expectation(weight, act, alpha, gs)
    rows = act["codes"].shape[0]
    k = _entry(dev, G_SMALL, pdl=pdl)
    ops = U.launch_operands(weight, act, out_global_scale=gs, alpha=alpha)
    out, out_scale = U.run(k, ops)
    torch.cuda.synchronize()
    U.assert_same_bytes(out.cpu(), want_out, "exact cells: packed codes")
    U.assert_same_bytes(U.live_scales(out_scale, rows, G_SMALL).cpu(), want_scale, "exact cells: block scales")
    assert int(want_out[rows - 1].any()) == 0 and int(want_scale[rows - 1].any()) == 0, "the all-zero row"
    assert (
        bool((want_out[0, :4] != 0).all()) and want_scale[: rows - 1].unique().numel() > 8 and want_out.unique().numel() > 32
    ), "the expectation is not trivial"


@pytest.mark.parametrize("pair", ((5, 4), (8, 9), (12, 13), (16, 17), (32, 33), (48, 49), (20, 64)), ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=gid)
def test_edge_rows(geometry, pair):
    """Row counts on both sides of every boundary of the kernel: the second epilogue warp's first token (5), the token-tile forms
    (8 | 9, 16 | 17, 32 | 33), the fourth warp's first token (12 | 13), the pass boundary inside a sub-tile (48 | 49), the two-pass
    sub-tile (20) and the ceiling (64).  The rows the two calls share are byte-identical, and each call is inside the reference
    bound."""
    dev = U.device_or_skip()
    k = _entry(dev, geometry)
    hi = max(pair)
    ops, act = _ops(dev, geometry, hi, "random", seed=7)
    results = {}
    for rows in pair:
        sub_ops, sub_act = _rows_of(ops, act, list(range(rows)))
        out, out_scale = U.run(k, sub_ops)
        torch.cuda.synchronize()
        _assert_reference(out, out_scale, _weight(dev, geometry)[0], sub_act, sub_ops, geometry, label=f"{gid(geometry)} rows {rows}")
        results[rows] = (out, U.live_scales(out_scale, rows, geometry))
    lo = min(pair)
    U.assert_same_bytes(results[hi][0][:lo], results[lo][0], f"rows {lo} against the first {lo} of {hi}: codes")
    U.assert_same_bytes(results[hi][1][:lo], results[lo][1], f"rows {lo} against the first {lo} of {hi}: scales")


def test_sixty_five_rows_is_a_typed_decline_never_a_launch():
    dev = U.device_or_skip()
    k = _entry(dev, G_SMALL)
    ops, _ = _ops(dev, G_SMALL, 64)
    ops["activation"] = torch.zeros(65, G_SMALL.HIDDEN // 2, dtype=torch.uint8, device=dev)
    out = torch.zeros(65, G_SMALL.INTERMEDIATE // 2, dtype=torch.uint8, device=dev)
    with pytest.raises(NotImplementedError, match="at most 64 rows per call"):
        k(
            ops["weight"],
            ops["weight_scale"],
            ops["activation"],
            ops["activation_scale"],
            ops["alpha"],
            ops["out_global_scale"],
            out,
            U.new_outputs(G_SMALL, 64, dev)[1],
        )
    with pytest.raises(NotImplementedError, match="at most 64 rows per call"):
        k.compile(65)
    assert int(out.any()) == 0


@pytest.mark.parametrize("rows", (3, 20, 49))
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=gid)
def test_caller_buffers_past_the_call_stay_untouched(geometry, rows):
    """What is not an input never reaches an output, and what the call does not own is not written: the activation-scale band's rows
    past ``rows`` poisoned (random bytes, then NaN codes) leave every live byte unchanged; a canary row behind ``out`` and the scale
    bytes outside the live rows keep their sentinel; the inputs are not modified."""
    dev = U.device_or_skip()
    k = _entry(dev, geometry)
    ops, _ = _ops(dev, geometry, rows, "random", seed=11)
    want_out, want_scale = U.run(k, ops)
    live = U.live_mask(rows, geometry, dev)
    live_in = U.to_atoms(torch.ones(rows, geometry.HIDDEN // 16, dtype=torch.uint8, device=dev)).bool()
    gen = torch.Generator(device="cpu").manual_seed(5)
    for poison in (torch.randint(0, 256, (int((~live_in).sum()),), dtype=torch.uint8, generator=gen).to(dev), U.E4M3_NAN):
        poisoned = dict(ops, activation_scale=ops["activation_scale"].clone())
        poisoned["activation_scale"][~live_in] = poison
        keep = {name: v.clone() for name, v in poisoned.items()}
        backing = torch.full((rows + 1, geometry.INTERMEDIATE // 2), 0x3C, dtype=torch.uint8, device=dev)
        out_scale = torch.full_like(want_scale, 0x3C)
        k(
            poisoned["weight"],
            poisoned["weight_scale"],
            poisoned["activation"],
            poisoned["activation_scale"],
            poisoned["alpha"],
            poisoned["out_global_scale"],
            backing[:rows],
            out_scale,
        )
        torch.cuda.synchronize()
        U.assert_same_bytes(backing[:rows], want_out, "codes under poisoned padding")
        U.assert_same_bytes(out_scale[live], want_scale[live], "live scale bytes under poisoned padding")
        assert int((backing[rows] != 0x3C).sum()) == 0 and int((out_scale[~live] != 0x3C).sum()) == 0, "a byte the call does not own was written"
        for name, v in keep.items():
            assert torch.equal(poisoned[name], v), f"the input {name} was modified"


@pytest.mark.parametrize("rows", (3, 40))
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=gid)
def test_accepted_operand_forms_write_the_same_bytes(geometry, rows):
    """Every operand form the entry documents as served writes the bytes of the plain form (``uint8`` scale buffers of exactly one
    band, ``[1]`` scalars, whole tensors): the three scale operands as ``torch.float8_e4m3fn`` tensors of another shape; 0-d
    scalars; scale storage longer than one band (the bytes behind the first band are neither read nor written); and operands that
    are aligned views INTO larger storage -- the activation and the outputs starting at a non-zero offset of their buffers, whose
    bytes in front of and behind the view keep their sentinel."""
    dev = U.device_or_skip()
    k = _entry(dev, geometry)
    ops, _ = _ops(dev, geometry, rows, "random", seed=17)
    want_out, want_scale = U.run(k, ops)
    h, i = geometry.HIDDEN, geometry.INTERMEDIATE
    band_in, band_out = 128 * h // 16, 128 * i // 16
    E4M3 = torch.float8_e4m3fn

    def call(o, out, out_scale):
        k(o["weight"], o["weight_scale"], o["activation"], o["activation_scale"], o["alpha"], o["out_global_scale"], out, out_scale)
        torch.cuda.synchronize()

    # (1) E4M3-typed scale tensors of another shape, 0-d scalars
    typed = dict(
        ops,
        weight_scale=ops["weight_scale"].view(E4M3).reshape(2 * i, h // 16),
        activation_scale=ops["activation_scale"].view(E4M3).reshape(128, h // 16),
        alpha=ops["alpha"].reshape(()),
        out_global_scale=ops["out_global_scale"].reshape(()),
    )
    out, out_scale = U.new_outputs(geometry, rows, dev)
    call(typed, out, out_scale.view(E4M3).reshape(-1, i // 16))
    U.assert_same_bytes(out, want_out, "E4M3-typed scales and 0-d scalars: codes")
    U.assert_same_bytes(out_scale, want_scale, "E4M3-typed scales and 0-d scalars: the scale band")

    # (2) scale storage longer than one band: three bands in, two bands out
    gen = torch.Generator(device="cpu").manual_seed(23)
    long_in = torch.randint(0, 256, (3 * band_in,), dtype=torch.uint8, generator=gen).to(dev)
    long_in[:band_in] = ops["activation_scale"]
    long_out = torch.full((2 * band_out,), 0xA5, dtype=torch.uint8, device=dev)
    out = torch.full_like(want_out, 0xA5)
    call(dict(ops, activation_scale=long_in), out, long_out)
    U.assert_same_bytes(out, want_out, "longer scale storage: codes")
    U.assert_same_bytes(long_out[:band_out], want_scale, "longer scale storage: the first band")
    assert int((long_out[band_out:] != 0xA5).sum()) == 0, "a scale byte behind the first band was written"

    # (3) aligned views into larger storage, at a non-zero offset
    act_store = torch.randint(0, 256, (rows + 4, h // 2), dtype=torch.uint8, generator=gen).to(dev)
    act_store[2 : 2 + rows] = ops["activation"]
    out_store = torch.full((rows + 2, i // 2), 0x3C, dtype=torch.uint8, device=dev)
    scale_store = torch.full((3 * band_out,), 0x3C, dtype=torch.uint8, device=dev)
    out_view, scale_view = out_store[1 : 1 + rows], scale_store[band_out : 2 * band_out]
    assert out_view.data_ptr() != out_store.data_ptr() and scale_view.data_ptr() != scale_store.data_ptr()
    call(dict(ops, activation=act_store[2 : 2 + rows]), out_view, scale_view)
    U.assert_same_bytes(out_view, want_out, "views into larger storage: codes")
    live = U.live_mask(rows, geometry, dev)
    U.assert_same_bytes(scale_view[live], want_scale[live], "views into larger storage: live scale bytes")
    assert (
        int((scale_view[~live] != 0x3C).sum()) == 0
        and int((scale_store[:band_out] != 0x3C).sum()) == 0
        and int((scale_store[2 * band_out :] != 0x3C).sum()) == 0
    )
    assert int((out_store[0] != 0x3C).sum()) == 0 and int((out_store[rows + 1] != 0x3C).sum()) == 0, "a byte around the output view was written"


ARMS = [
    dict(cfg=Cfg(TOKEN_TILE=16)),
    dict(cfg=Cfg(TOKEN_TILE=32)),
    dict(cfg=Cfg(TOKEN_TILE=64)),
    dict(cfg=Cfg(PREWAIT_STAGES=0)),
    dict(cfg=Cfg(PREWAIT_STAGES=0), pdl=True),
    dict(cfg=Cfg(L2_EVICT_FIRST=False)),
    dict(cfg=Cfg(MAX_CLUSTERS=1)),
    dict(cfg=Cfg(MAX_CLUSTERS=37)),
    dict(cfg=Cfg(MAX_CLUSTERS=148)),
    dict(cfg="sm-count"),
    dict(cfg="twice-the-sm-count"),
    dict(pdl=True),
]


def _arm_id(arm):
    cfg = arm.get("cfg")
    text = (
        cfg
        if isinstance(cfg, str)
        else ("default" if cfg is None else ",".join(f"{f.name}={getattr(cfg, f.name)}" for f in fields(Cfg) if getattr(cfg, f.name) != getattr(Cfg(), f.name)))
    )
    return text + (",pdl" if arm.get("pdl") else "")


@pytest.mark.parametrize("arm", ARMS, ids=_arm_id)
def test_form_and_knob_arms_write_the_default_bytes(arm):
    """No performance field changes a byte: a wider token tile than the rows need, no pre-wait stages, the default L2 policy, every
    grid cap (one block, a partial wave, the part's SM count, twice the SM count), the launch attribute -- each writes the default
    entry's bytes at rows on both sides of every form boundary."""
    dev = U.device_or_skip()
    geometry = G_DEFAULT
    base = _entry(dev, geometry)
    kw = dict(arm)
    if isinstance(kw.get("cfg"), str):
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        kw["cfg"] = Cfg(MAX_CLUSTERS=sms if kw["cfg"] == "sm-count" else 2 * sms)
    k = _entry(dev, geometry, **kw)
    ops64, act = _ops(dev, geometry, 64, "random", seed=3)
    for rows in (1, 3, 5, 8, 9, 12, 13, 16, 17, 32, 33, 48, 49, 64):
        ops, _ = _rows_of(ops64, act, list(range(rows)))
        want = U.run(base, ops)
        got = U.run(k, ops)
        torch.cuda.synchronize()
        U.assert_same_bytes(got[0], want[0], f"{_arm_id(arm)} rows {rows}: codes")
        U.assert_same_bytes(got[1], want[1], f"{_arm_id(arm)} rows {rows}: the scale band")


def test_plan_time_compile_covers_every_served_row_count(monkeypatch):
    """``compile()`` compiles the four forms; after it no launch of a served row count reaches the compiler."""
    dev = U.device_or_skip()
    k = entry.SwigluFp4Decode(dev, G_SMALL)
    assert not k.compiled and all(k.kernel_symbols_for(rows) == () for rows in (1, 9, 17, 33))
    k.compile()
    assert len(k.compiled) == 4 and all(k.cfg_for(rows) in k.compiled for rows in range(1, 65))
    monkeypatch.setattr(k._kernel_module, "compile_kernel", lambda cfg: (_ for _ in ()).throw(AssertionError(f"compiled on the execute path: {cfg}")))
    k.compile()
    k.compile(40)
    for rows in (1, 9, 17, 33):
        ops, _ = _ops(dev, G_SMALL, rows)
        U.run(k, ops)
    torch.cuda.synchronize()
    symbols = {k.kernel_symbols_for(rows) for rows in (1, 9, 17, 33)}
    assert len(symbols) == 4 and all(len(s) == 1 and s[0].startswith("cudnn_kernel_frost_swiglu_fp4_decode_h5120_i128_t") for s in symbols)


def test_graph_capture_replays_bitwise_with_eager():
    dev = U.device_or_skip()
    cells = []
    for geometry, rows, kw in ((G_DEFAULT, 6, dict(pdl=True)), (G_SMALL, 20, dict()), (G_DEFAULT, 40, dict(pdl=True))):
        k = _entry(dev, geometry, **kw)
        k.compile()
        ops, act = _ops(dev, geometry, rows, "random", seed=21)
        cells.append((k, ops, act, rows))
    eager = [U.run(k, ops) for k, ops, _, _ in cells]  # warm
    outs = [U.new_outputs(k.geometry, rows, dev, fill=0) for k, _, _, rows in cells]

    def launch_all():
        for (k, ops, _, _), (o, s) in zip(cells, outs):
            U.run(k, ops, o, s)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        launch_all()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        launch_all()
    for _ in range(3):
        for o, sc in outs:
            o.zero_()
            sc.fill_(0xA5)
        g.replay()
        torch.cuda.synchronize()
        for (eo, es), (o, sc) in zip(eager, outs):
            U.assert_same_bytes(o, eo, "graph replay against eager: codes")
            U.assert_same_bytes(sc, es, "graph replay against eager: the scale band")
    # the graph binds the operands' storage: new bytes in the same activation tensor are what the next replay computes
    k, ops, act, rows = cells[0]
    saved = ops["activation"].clone()
    try:
        ops["activation"].copy_(U.make_activation(k.geometry, rows, dev, seed=99)["codes"])
        want = U.run(k, ops)
        g.replay()
        torch.cuda.synchronize()
        U.assert_same_bytes(outs[0][0], want[0], "graph replay after the activation's bytes changed")
    finally:
        ops["activation"].copy_(saved)


def test_dependent_that_waits_reads_the_eager_bytes():
    """The dependent-launch chain where the attribute is not inert: a predecessor on the stream that writes the activation (an operand
    the entry reads below its wait), the entry built with ``pdl=True``, and a dependent launched with the attribute that WAITS and then
    copies ``out`` and the scale band (``swiglu_fp4_decode_dependent.py``).  A thread block of the entry signals its dependents when
    its loader warp has finished, before its last output tile is written, so the dependent may start while the entry still writes;
    its wait holds its reads.  Eager and captured in a CUDA graph, at one row count per token tile: the entry's outputs and the
    copies hold the bytes of the entry launched without the attribute.  (A dependent that reads without waiting is the documented
    misuse; it is not run.)"""
    dev = U.device_or_skip()
    from swiglu_fp4_decode_dependent import DependentCopy

    copy = DependentCopy(dev)
    geometry = G_DEFAULT
    plain, k = _entry(dev, geometry), _entry(dev, geometry, pdl=True)
    assert k.cfg.PDL and not plain.cfg.PDL
    k.compile()
    for rows in (1, 12, 20, 64):
        ops, _ = _ops(dev, geometry, rows, "random", seed=31 + rows)
        want_out, want_scale = U.run(plain, ops)
        source = ops["activation"].clone()
        out, out_scale = U.new_outputs(geometry, rows, dev)
        out_copy, scale_copy = torch.zeros_like(out), torch.zeros_like(out_scale)

        def chain():
            ops["activation"].copy_(source)  # the predecessor: a launch that writes what the entry reads below its wait
            k(ops["weight"], ops["weight_scale"], ops["activation"], ops["activation_scale"], ops["alpha"], ops["out_global_scale"], out, out_scale)
            copy(out, out_copy)
            copy(out_scale, scale_copy)

        def reset():
            ops["activation"].zero_()  # the entry computes other bytes if it reads the activation ahead of its predecessor
            out.fill_(0xA5)
            out_scale.fill_(0xA5)
            out_copy.zero_()
            scale_copy.zero_()

        def check(label):
            torch.cuda.synchronize()
            U.assert_same_bytes(out, want_out, f"{label} rows {rows}: the entry's codes")
            U.assert_same_bytes(out_scale, want_scale, f"{label} rows {rows}: the entry's scale band")
            U.assert_same_bytes(out_copy, want_out, f"{label} rows {rows}: the dependent's copy of the codes")
            U.assert_same_bytes(scale_copy, want_scale, f"{label} rows {rows}: the dependent's copy of the scale band")

        reset()
        chain()  # eager (the first call also compiles the dependent)
        check("eager chain")
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain()
        for _ in range(3):
            reset()
            graph.replay()
            check("captured chain")
        assert int((want_out != 0xA5).sum()) > 0 and int((want_scale != 0xA5).sum()) == rows * geometry.INTERMEDIATE // 16


def test_warm_call_allocates_nothing_and_never_synchronizes(monkeypatch):
    """After ``compile()`` a launch is a guaranteed hit: with compilation forbidden, device-to-host reads raising and the allocator
    watched, three warm launches run and write the cold bytes."""
    dev = U.device_or_skip()
    k = entry.SwigluFp4Decode(dev, G_SMALL, pdl=True)
    k.compile()
    ops, _ = _ops(dev, G_SMALL, 5)
    want = U.run(k, ops)
    out, out_scale = U.new_outputs(G_SMALL, 5, dev)
    torch.cuda.synchronize()
    monkeypatch.setattr(k._kernel_module, "compile_kernel", lambda cfg: (_ for _ in ()).throw(AssertionError(f"compiled on the execute path: {cfg}")))
    allocations = torch.cuda.memory_stats(dev)["allocation.all.allocated"]  # the count of allocation events: a free does not move it
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(ops["weight"], ops["weight_scale"], ops["activation"], ops["activation_scale"], ops["alpha"], ops["out_global_scale"], out, out_scale)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocations, "a warm launch allocated device memory"
    U.assert_same_bytes(out, want[0], "warm launch: codes")
    U.assert_same_bytes(out_scale, want[1], "warm launch: the scale band")


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    dev = U.device_or_skip()
    k = _entry(dev, G_SMALL)
    ops, _ = _ops(dev, G_SMALL, 4)
    out, out_scale = U.run(k, ops)
    torch.cuda.synchronize()
    keep = (out.clone(), out_scale.clone())
    args = [ops["weight"], ops["weight_scale"], ops["activation"], ops["activation_scale"], ops["alpha"], ops["out_global_scale"], out, out_scale]
    for index, bad, exc, match in (
        (2, ops["activation"].float(), TypeError, "activation must be torch.uint8"),
        (6, torch.zeros(5, G_SMALL.INTERMEDIATE // 2, dtype=torch.uint8, device=dev), ValueError, r"out must be exactly \[4, 64\]"),
        (0, ops["weight"].cpu(), ValueError, "every operand must live on one device"),
        (4, torch.ones(2, device=dev), ValueError, "alpha must hold one fp32 element"),
        (3, ops["activation_scale"][:100], ValueError, "activation_scale must hold at least 40960 bytes"),
        (2, torch.zeros(4, G_SMALL.HIDDEN, dtype=torch.uint8, device=dev)[:, ::2], NotImplementedError, "serves contiguous operands"),
    ):
        mutated = list(args)
        mutated[index] = bad
        with pytest.raises(exc, match=match):
            k(*mutated)
    torch.cuda.synchronize()
    U.assert_same_bytes(out, keep[0], "a declined launch wrote the output")
    U.assert_same_bytes(out_scale, keep[1], "a declined launch wrote the scales")


LAUNCH_RECORD_CELLS = [(G_DEFAULT, 1, dict()), (G_DEFAULT, 12, dict(pdl=True)), (G_DEFAULT, 40, dict(cfg=Cfg(MAX_CLUSTERS=37))), (G_SMALL, 64, dict())]


@pytest.mark.parametrize("geometry, rows, arm", LAUNCH_RECORD_CELLS, ids=["default-rows1", "default-rows12-pdl", "default-rows40-cap37", "small-rows64"])
def test_launch_facts_equal_the_profilers_record_of_a_launch(geometry, rows, arm):
    """On the device: one call at ``rows`` is ONE kernel whose profiler record -- name, grid, block, shared memory -- equals the entry's
    launch facts; nothing is named before the first compile, and another form of the entry is named only once it is compiled.  A fresh
    entry per cell: the cell reads the compile cache."""
    dev = U.device_or_skip()
    kw = dict(arm)
    if "cfg" in kw:
        kw["cfg"] = replace(kw["cfg"], GEOM=geometry)
    k = entry.SwigluFp4Decode(dev, geometry, **kw)
    ops, _ = _ops(dev, geometry, rows)
    out, out_scale = U.new_outputs(geometry, rows, dev)
    assert k.kernel_symbols_for(rows) == (), "nothing is compiled yet"
    record = assert_facts_equal_the_launch(k, lambda: U.run(k, ops, out, out_scale), rows, what=entry.NAME)
    form = k.cfg_for(rows)
    assert record.name.startswith(f"cudnn_kernel_frost_swiglu_fp4_decode_{form}_"), record.name
    assert len(f"cudnn_kernel_frost_swiglu_fp4_decode_{form}") <= 100, "the form sits in the symbol's first 100 characters"
    other = 64 if kcfg.token_tile_for(rows) != 64 else 1
    assert k.kernel_symbols_for(other) == (), "a form that was not compiled names nothing"
    assert k.form_for(rows)["PDL"] is bool(arm.get("pdl")) and k.form_for(rows)["TOKEN_TILE"] == kcfg.token_tile_for(rows)


# The launch matrix in a child interpreter.  The child prints a stage line before every phase and arms its own traceback dump ahead of
# the parent's deadline; the parent reads the LAST stage line to tell a compile stall (a setup budget) from a dispatched kernel that
# did not return (a hang: audit the barrier table of the kernel module -- the limit below is never raised).
CHILD_COMPILE_BUDGET_S = 600  # a cold DSL import plus sixteen trace-compiles
CHILD_DISPATCH_LIMIT_S = 120
_CHILD = """
import faulthandler, sys, time
faulthandler.dump_traceback_later({dump_s}, exit=False)
print("STAGE import", flush=True)
{tree}
import torch
from cudnn.gemm.frost.swiglu_decode import swiglu_fp4_decode as entry
from cudnn.gemm.frost.swiglu_decode.kernel import swiglu_fp4_decode_config as kcfg
print("LOADED entry", entry.__file__, flush=True)
print("LOADED config", kcfg.__file__, flush=True)
dev = torch.device("cuda", torch.cuda.current_device())
print("STAGE compile", flush=True)
cells = []
for g in kcfg.SERVED_GEOMETRIES:
    h, i = g.HIDDEN, g.INTERMEDIATE
    z = lambda *shape: torch.zeros(*shape, dtype=torch.uint8, device=dev)
    w, ws, sa, one = z(2 * i, h // 2), z(2 * i * h // 16), z(128 * h // 16), torch.ones(1, dtype=torch.float32, device=dev)
    for pdl in (False, True):
        k = entry.SwigluFp4Decode(dev, g, pdl=pdl)
        k.compile()
        for rows in (1, 8, 9, 16, 17, 32, 33, 64):
            cells.append((k, w, ws, z(rows, h // 2), sa, one, one, z(rows, i // 2), z(128 * i // 16)))
torch.cuda.synchronize()
print("STAGE dispatch", len(cells), "cells", flush=True)
t0 = time.monotonic()
for k, *args in cells:
    for _ in range(8):  # back to back: under pdl each launch is the next one's predecessor, released when its loader has finished
        k(*args)
torch.cuda.synchronize()
print("LOADED kernel", cells[0][0]._kernel_module.__file__, flush=True)
print("STAGE done", round(time.monotonic() - t0, 3), "s", flush=True)
"""


def test_dispatch_returns_within_the_guard():
    U.device_or_skip()
    limit = CHILD_COMPILE_BUDGET_S + CHILD_DISPATCH_LIMIT_S
    code = _CHILD.format(dump_s=limit - 15, tree=CHILD_TREE)
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


@pytest.mark.parametrize("rows, cap", ((20, 8), (64, 8), (64, 2)), ids=["tile32-cap8", "tile64-cap8", "tile64-cap2"])
def test_warm_launches_keep_writing_the_reference_bytes(rows, cap):
    """The staging ring is refilled inside a launch only when one thread block runs more sub-tiles than the ring has stages: a small
    grid cap makes every block run tens of weight tiles.  Warm launches of such a shape, back to back and behind a launch that
    overwrites every SM's shared memory, must keep writing the bytes of the default entry -- whose blocks run at most two weight
    tiles each, so that no staging stage is refilled inside its launch (the independent reference)."""
    dev = U.device_or_skip()
    geometry = G_DEFAULT
    build_launchers(dev)
    k = entry.SwigluFp4Decode(dev, geometry, cfg=Cfg(MAX_CLUSTERS=cap), pdl=True)
    k.compile()
    ops, _ = _ops(dev, geometry, rows, "random", seed=31)
    want_out, want_scale = U.run(_entry(dev, geometry), ops)
    out, out_scale = U.new_outputs(geometry, rows, dev)
    form = k.cfg_for(rows)
    sub_tiles = (geometry.WEIGHT_TILES // cap) * (form.TOKEN_TILE // kcfg.epilogue_tile(form))
    refills = sub_tiles // kcfg.stages(form)[1]
    assert refills >= 2, "the shape refills the staging ring inside one launch"
    live = U.live_mask(rows, geometry, dev)
    live_bytes = torch.empty(int(live.sum()), dtype=torch.uint8, device=dev)

    def launch():
        k(ops["weight"], ops["weight_scale"], ops["activation"], ops["activation_scale"], ops["alpha"], ops["out_global_scale"], out, out_scale)
        torch.masked_select(out_scale, live, out=live_bytes)

    case = StressCase(
        name=f"swiglu fp4 decode rows {rows} grid cap {cap}",
        launch=launch,
        outputs=dict(out=out, live_scales=live_bytes),
        thread_blocks=cap,
        refills_per_launch=refills,
        reference=dict(out=want_out, live_scales=want_scale[live].contiguous()),
    )
    assert_warm_stress(case, validation=stress_validation(dev), warm=600, poisoned=600, budget_s=90.0)
