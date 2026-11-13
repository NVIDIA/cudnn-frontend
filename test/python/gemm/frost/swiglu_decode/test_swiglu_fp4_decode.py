# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate / up product of an FP4 MLP for decode rows with SiLU(gate) x up and the FP4 re-quantisation in one launch
(``cudnn.SwigluFp4Decode``).  Host: the config's ledgers and derived constants (the stage rule's mirror, the shared-memory and
tensor-memory ledgers, the barrier counts), the token-tile rule and the form text that heads the kernel symbol; the typed declines of
the host entry (CPU tensors), the DSL gate, the load-time row interleave, the plan-time compile key, the launch facts and the lazy
exports.  A child interpreter runs the source overlay of this process and reports the files it loaded; the parent compares them.
The operand builders live in ``swiglu_fp4_decode_utils.py`` beside this module.

Marked ``no_workspace_shim``: the engine suite's autouse fixture would otherwise import torch and both architecture compilers for
every test of this direct-call entry."""

import importlib
import re
import subprocess
import sys
import textwrap
from dataclasses import fields, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

import swiglu_fp4_decode_utils as U  # noqa: E402  (the operand builders, the reference and the device gate)

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
    rubin = _modelled_cuda_entry(monkeypatch, readings=(233472, 232448, 212), cc=(10, 7))
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        rubin.compile(1)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(entry, "dsl_compiler_version", lambda: (12, 9))
    for e in (k, rubin):  # the compiler floor is the kernel's (its bytes), not a part's: it declines on every class
        with pytest.raises(
            NotImplementedError, match=r"is qualified with the CuTe DSL compiling with CUDA 13\.4 or later; the DSL in this process compiles with CUDA 12\.9"
        ):
            e.compile(1)
    assert not k.compiled and not rubin.compiled and k._kernel_module is None and rubin._kernel_module is None


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
