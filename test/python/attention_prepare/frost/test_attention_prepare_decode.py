# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attention preparation decode step (``cudnn.attention_prepare.frost.prepare_decode``): on the host the config's limits and
tables, the forms and three launch arms, every typed decline of the entry (CPU tensors) with the three construction facts (page
size, position dtype, slot dtype), the overlap rule on views that interleave in one allocation, the DSL gates (version, target,
compiler), the plan-time-only compile key, the launch facts, the lazy export, and the kernel's RENDERING for the claimed targets
(the instructions that define the sum tree and the rounding chain; the position of the dependent-launch wait and release; the form
inside the first 100 characters of the symbol); on a cc 10.x device the bytes against the reference of ``attention_prepare_utils``
(a PyTorch twin with the kernel's rounding points and its band; the gate rows, the value rows and every untouched pool row
bitwise), edge rows, the slot predicate (negative slots, a short and an empty slot vector, a slot at the pool's end and one past
it), output ownership per output, the bit-neutral arms, the wait behind a predecessor that released early, the release beside a
dependent launch, CUDA-graph replay, the warm path's allocation / synchronization discipline and the profiler's record of a launch.

No cell arms a timer of its own: every device cell synchronizes under the suite's per-test watchdog (``CUDNN_TEST_TIMEOUT`` in
``test/python/conftest.py``).  A trip of that watchdog on a launch is a hang of the kernel's one thread-block barrier -- audit the
barrier table in the kernel's docstring; never a larger limit."""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import asdict, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from attention_prepare_utils import (  # noqa: E402  (the form tables, the operand builders, the reference and the device gate)
    BF16,
    CANARY_BITS,
    D,
    DEFAULT_FORM,
    EPS,
    FORMS,
    GEOMETRY,
    HALF,
    HEADS,
    I16,
    INT,
    KV_HEADS,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    ROTARY,
    SECTION,
    SENTINEL_BITS,
    _device_or_skip,
    assert_twin,
    bind,
    bits,
    call_args,
    canary_cache,
    form_kwargs,
    fresh_outputs,
    make_entry,
    make_operands,
)

from cudnn.attention_prepare.frost import prepare_decode as entry  # noqa: E402
from cudnn.attention_prepare.frost.kernel import prepare_decode_config as kcfg  # noqa: E402
from launch_records import assert_facts_equal_the_launch  # noqa: E402

pytestmark = [pytest.mark.L0]

REPO = Path(__file__).resolve().parents[4]
SYMBOL = "cudnn_kernel_frost_attention_prepare_"
TOKENS = (1, 2, 3, 4, 8, 16, 17, 32, 64, 160, 1024)


def _cfg(form=DEFAULT_FORM, arm=(None, None), eps=EPS, geometry=GEOMETRY):
    return entry.make_cfg(geometry, eps, pdl=arm[0], pdl_release=arm[1], **form_kwargs(form))


def _needs_paged_contract():
    """The pools and the slot vector are checked by the paged-pool contract of ``cudnn.gated_attention_block``, whose package imports
    the CuTe DSL: a host cell that reaches those checks needs the DSL importable."""
    pytest.importorskip("cutlass")


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_config_constants_and_tables():
    """The limits with their derivation, the barrier ledger and the shared-memory table as data: one thread block per (token, head)
    of ``HEAD_DIM / 2`` threads in four full warps; ONE thread-block barrier that every thread issues once (``issuing_lanes ==
    participants``); ONE shared-memory buffer of an fp32 slot per warp, written once and never refilled; the token ceiling above one
    scheduling wave of every served part; the position row of every rotary index and the kernel's masks of it."""
    assert (kcfg.PAIR, kcfg.SERVED_GEOMETRIES, kcfg.SERVED_PAGE_SIZES, kcfg.MAX_ROWS, kcfg.SM_THREADS) == (2, ((24, 4, 256, 64),), (64, 16), 1024, 2048)
    assert kcfg.POSITION_FORMS == ("1d", "3d") and kcfg.INT_DTYPES == ("int32", "int64") and kcfg.INT_BYTES == {"int32": 4, "int64": 8}
    cfg = _cfg()
    assert (cfg.HEADS, cfg.KV_HEADS, cfg.HEAD_DIM, cfg.ROTARY_DIM, cfg.HALF, cfg.PROGRAMS) == (24, 4, 256, 64, 32, 28)
    assert (cfg.THREADS, cfg.WARPS, cfg.SMEM_BYTES) == (128, 4, 16) and cfg.THREADS == cfg.HEAD_DIM // kcfg.PAIR == 32 * cfg.WARPS
    assert kcfg.barrier_ledger(cfg) == dict(participants=128, issuing_lanes=128)
    table = kcfg.smem_table(cfg)
    assert len(table) == 1 and table[0]["bytes"] == 4 * cfg.WARPS == 16 and table[0]["elements"] == 4 and table[0]["lane_stride_bytes"] == 4
    assert table[0]["refilled"] is False, "the one buffer is written once and read once: nothing releases it"
    assert cfg.CTAS_PER_SM == 16, "sixteen 128-thread blocks per SM"
    assert kcfg.MAX_ROWS * cfg.PROGRAMS > cfg.CTAS_PER_SM * 212, "the token ceiling crosses a scheduling wave on a 212-SM part"
    assert cfg.POSITION_ROWS == 3 and _cfg("1d-p64-i64-i64").POSITION_ROWS == 1
    # the position row of a rotary index: under (11, 11, 10) row 1 for r % 3 == 1 (r < 33: all of them), row 2 for r % 3 == 2 and r < 30
    rows = [kcfg.position_row_of(cfg, r) for r in range(HALF)]
    assert rows == [(r % 3) if not (r % 3 == 2 and r >= 30) else 0 for r in range(HALF)]
    assert all(kcfg.position_row_of(_cfg("1d-p64-i64-i64"), r) == 0 for r in range(HALF))
    (h0_mask, w0_mask), (h1_mask, w1_mask) = kcfg.position_row_masks(cfg)
    for j in range(HALF // 2):
        assert ((h0_mask >> j) & 1, (w0_mask >> j) & 1) == (int(rows[2 * j] == 1), int(rows[2 * j] == 2)), j
        assert ((h1_mask >> j) & 1, (w1_mask >> j) & 1) == (int(rows[2 * j + 1] == 1), int(rows[2 * j + 1] == 2)), j
    short = replace(cfg, SECTION=(20, 4, 8))  # another section: the masks follow it
    assert [kcfg.position_row_of(short, r) for r in range(HALF)] == [
        1 if (r % 3 == 1 and r < 12) else (2 if (r % 3 == 2 and r < 24) else 0) for r in range(HALF)
    ]


def test_form_text_is_compact_unique_and_inside_the_first_100_characters_of_the_symbol():
    """``str(cfg)`` -- the text the DSL appends to the kernel's name -- states the form (geometry, position form with its section,
    page size, position and slot dtype, launch arm, the fp32 bits of eps), differs between any two forms and arms, and ends inside
    the first 100 characters of the symbol for every served key; ``repr`` stays the field list."""
    texts = {}
    for positions in kcfg.POSITION_FORMS:
        for page in kcfg.SERVED_PAGE_SIZES:
            for pd in kcfg.INT_DTYPES:
                for sd in kcfg.INT_DTYPES:
                    for arm, arm_id in zip(LAUNCH_ARMS, LAUNCH_ARM_IDS):
                        cfg = entry.make_cfg(
                            GEOMETRY, EPS, positions=positions, mrope_section=SECTION if positions == "3d" else None, page_size=page, position_dtype=pd,
                            slot_dtype=sd, pdl=arm[0], pdl_release=arm[1],
                        )  # fmt: skip
                        text = str(cfg)
                        texts[(positions, page, pd, sd, arm_id)] = text
                        assert len(SYMBOL + text) <= 100, text
                        assert re.fullmatch(r"[a-z0-9_]+", text), f"{text!r}: characters the symbol mangling would drop"
                        assert "0x" not in text, "a hex run behind 0x would be dropped from the symbol as an address"
                        assert repr(cfg).startswith("Cfg(GEOMETRY=AttentionPrepareGeometry(HEADS=24, KV_HEADS=4, HEAD_DIM=256, ROTARY_DIM=64), EPS=1e-06, ")
    assert len(set(texts.values())) == len(texts) == 2 * 2 * 2 * 2 * 3
    assert texts[("3d", 64, "int64", "int64", "plain")] == "h24_4_d256_r64_3d_11_11_10_p64_i64_i64_plain_eps358637bd"
    assert texts[("1d", 16, "int32", "int64", "wait")] == "h24_4_d256_r64_1d_p16_i32_i64_wait_eps358637bd"
    assert texts[("3d", 16, "int64", "int32", "release-wait")] == "h24_4_d256_r64_3d_11_11_10_p16_i64_i32_release_wait_eps358637bd"
    assert str(_cfg(eps=1e-5)) != str(_cfg()) and kcfg.eps_bits(1e-6) == 0x358637BD
    assert str(replace(_cfg(), SECTION=(12, 10, 10))) != str(_cfg()), "the section is part of the form"


def _make(**change):
    kw = dict(positions="3d", mrope_section=SECTION, page_size=64, position_dtype="int64", slot_dtype="int64")
    geometry, eps = change.pop("geometry", GEOMETRY), change.pop("eps", EPS)
    kw.update(change)
    return entry.make_cfg(geometry, eps, **kw)


def test_construction_declines_name_the_argument():
    """Every decline of the constructor's arguments, without torch and without the DSL: the geometry (type, sign, the power-of-two
    rule, the served set), eps, the position form and its section, and THE THREE CONSTRUCTION FACTS -- the page size (a positive
    power-of-two int; 64 or 16 served), the position dtype and the slot dtype (the names ``"int32"`` / ``"int64"``) -- then the
    launch knobs."""
    G = kcfg.AttentionPrepareGeometry
    assert _make().PAGE_SIZE == 64 and _make(page_size=16).PAGE_SIZE == 16
    with pytest.raises(TypeError, match="geometry must be an AttentionPrepareGeometry, got tuple"):
        _make(geometry=(24, 4, 256, 64))
    with pytest.raises(TypeError, match="geometry.HEADS must be an int, got float"):
        _make(geometry=G(24.0, 4, 256, 64))
    with pytest.raises(TypeError, match="geometry.KV_HEADS must be an int, got bool"):
        _make(geometry=G(24, True, 256, 64))
    with pytest.raises(ValueError, match="geometry.HEADS must be positive, got 0"):
        _make(geometry=G(0, 4, 256, 64))
    with pytest.raises(ValueError, match="ROTARY_DIM must be even and at most HEAD_DIM"):
        _make(geometry=G(24, 4, 256, 63))
    with pytest.raises(ValueError, match="ROTARY_DIM must be even and at most HEAD_DIM"):
        _make(geometry=G(24, 4, 32, 64))
    for geometry in (G(24, 4, 192, 64), G(24, 4, 256, 48)):
        with pytest.raises(NotImplementedError, match="serves a HEAD_DIM and a ROTARY_DIM / 2 that are powers of two"):
            _make(geometry=geometry)
    for geometry in (G(32, 8, 256, 64), G(24, 4, 128, 64), G(24, 4, 256, 32), G(24, 4, 512, 64)):
        with pytest.raises(NotImplementedError, match=r"serves the geometries \(HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM\) \(24, 4, 256, 64\)"):
            _make(geometry=geometry)
    for eps in (0.0, -1e-6, float("nan"), float("inf"), 1e-60, "1e-6", None, True):
        with pytest.raises(ValueError, match="eps must be a finite positive float"):
            _make(eps=eps)
    for positions in ("2d", "3D", None, 3, ""):
        with pytest.raises(ValueError, match=r"positions must be one of \('1d', '3d'\) \(required: the forms are peers\)"):
            _make(positions=positions)
    with pytest.raises(TypeError):  # the three facts and the position form are REQUIRED keywords: none has a default
        entry.make_cfg(GEOMETRY, EPS)
    for missing in ("positions", "page_size", "position_dtype", "slot_dtype"):
        kw = dict(positions="1d", page_size=64, position_dtype="int64", slot_dtype="int64")
        del kw[missing]
        with pytest.raises(TypeError, match=missing):
            entry.make_cfg(GEOMETRY, EPS, **kw)
    with pytest.raises(ValueError, match="positions='3d' needs mrope_section"):
        _make(mrope_section=None)
    for section in ((11, 11), (11, 11, 10.0), (0, 22, 10), (11, 11, True), "11", (11, -1, 22)):
        with pytest.raises(ValueError, match="positions='3d' needs mrope_section = \\(t, h, w\\), three positive ints"):
            _make(mrope_section=section)
    with pytest.raises(ValueError, match=r"mrope_section must sum to ROTARY_DIM / 2 = 32 .*got \(11, 11, 11\)"):
        _make(mrope_section=(11, 11, 11))
    with pytest.raises(ValueError, match="mrope_section belongs to positions='3d'"):
        _make(positions="1d", mrope_section=SECTION)
    assert _make(mrope_section=[11, 11, 10]).SECTION == (11, 11, 10), "a list is taken as the tuple it spells"
    # the page size: a compile-time constant of the kernel (the divide is a shift)
    for page in (0, -64, 64.0, "64", None, True):
        with pytest.raises(ValueError, match="page_size must be a positive int"):
            _make(page_size=page)
    for page in (48, 100, 3):
        with pytest.raises(ValueError, match="page_size must be a power of two"):
            _make(page_size=page)
    for page in (1, 8, 32, 128, 1024):
        with pytest.raises(NotImplementedError, match=rf"serves page sizes 64, 16 \(validated\); got {page}"):
            _make(page_size=page)
    # the two integer dtypes: names, each a compile-time load width
    for name in ("position_dtype", "slot_dtype"):
        for value in ("int16", "uint32", "long", torch.int64, None, 64):
            with pytest.raises(ValueError, match=rf"{name} must be one of \('int32', 'int64'\)"):
                _make(**{name: value})
    for name in ("pdl", "pdl_release"):
        for value in (1, "yes", 0.0):
            with pytest.raises(TypeError, match=f"{name} must be a bool or None"):
                _make(**{name: value})
    with pytest.raises(ValueError, match="pdl_release releases dependents in front of this launch's own wait and needs pdl=True"):
        _make(pdl_release=True)
    with pytest.raises(ValueError, match="pdl_release .* needs pdl=True"):
        _make(pdl=False, pdl_release=True)
    arms = {
        (c.PDL, c.PDL_RELEASE, c.LAUNCH_ARM)
        for c in (_make(), _make(pdl=False), _make(pdl=True, pdl_release=False), _make(pdl=True), _make(pdl=True, pdl_release=True))
    }
    assert arms == {(False, False, "plain"), (True, False, "wait"), (True, True, "release_wait")}
    with pytest.raises(TypeError, match="cfg must be a prepare_decode_config.Cfg"):
        kcfg.validate_cfg(dict())
    with pytest.raises(TypeError, match="pdl must be a bool"):
        kcfg.validate_cfg(replace(_make(), PDL=1))
    for rows in (0, -3):
        with pytest.raises(ValueError, match="rows must be at least 1"):
            kcfg.validate_rows(rows)
    for rows in (1.0, "1", None, True):
        with pytest.raises(TypeError, match="rows must be an int"):
            kcfg.validate_rows(rows)
    with pytest.raises(NotImplementedError, match="at most 1024 rows"):
        kcfg.validate_rows(1025)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the binding checks (CPU tensors)
# ---------------------------------------------------------------------------------------------------------------------------
def _check(form, ops, **kw):
    kw.setdefault("device_cc", (10, 0))
    return entry.check_support(_cfg(form), *call_args(ops), **kw)


@pytest.mark.parametrize("form", sorted(FORMS))
def test_served_operands_pass_the_host_checks(form):
    """The operands a decode step holds -- four views of one slab, a ``[3, tokens]`` position tensor whose row stride is not its
    length, the two halves of one packed cache -- pass at every token count, with caller-owned outputs and with none, with a full, a
    short and an EMPTY slot vector, and with a ``[3, tokens]`` tensor of three equal rows by a zero row stride."""
    _needs_paged_contract()
    for tokens in (1, 3, 17, 160, kcfg.MAX_ROWS):
        ops = make_operands(tokens, form)
        assert not ops["q"].is_contiguous() and not ops["k_pool"].is_contiguous(), "the cell binds views, not copies"
        assert _check(form, ops) == tokens
        assert _check(form, ops, **fresh_outputs(tokens, "cpu", extra=2)) == tokens
        for n in sorted({0, 1, tokens // 2}):
            short = dict(ops, slot_mapping=ops["slot_mapping"][:n].clone())
            assert _check(form, short) == tokens
    if FORMS[form][0] == "3d":
        ops = make_operands(5, form)
        ops["positions"] = ops["positions"][:1].expand(3, 5)
        assert ops["positions"].stride() == (0, 1) and _check(form, ops) == 5


def _misaligned(shape, dtype=None):
    """A tensor of ``shape`` whose first element sits 2 bytes behind a 16-byte boundary."""
    n = 1
    for s in shape:
        n *= s
    base = torch.zeros(n + 8, dtype=torch.int16)
    offset = next(o for o in range(8) if (base.data_ptr() + 2 * o) % 16 == 2)
    return base[offset : offset + n].view(dtype or BF16).view(*shape)


def _misaligned_int64(shape):
    """An int64 tensor of ``shape`` whose first element sits 4 bytes behind an 8-byte boundary."""
    n = 1
    for s in shape:
        n *= s
    for offset in (4, 12):
        t = torch.frombuffer(bytearray(8 * n + 32), dtype=torch.int64, count=n, offset=offset)
        if t.data_ptr() % 8:
            return t.reshape(*shape)
    raise AssertionError("no misaligned int64 view could be built on this host")


def _mutations():
    """``name -> (form, mutate(ops) -> ops, exception, message pattern)``: ONE change to served operands per decline."""
    T = 5

    def swap(name, value):
        return lambda ops: dict(ops, **{name: value(ops) if callable(value) else value})

    m = {}
    f64, f32 = DEFAULT_FORM, "3d-p16-i32-i32"
    for name in ("q", "gate", "k", "v", "q_weight", "k_weight", "cos_sin", "positions"):
        m[f"{name} is not a tensor"] = (f64, swap(name, None), TypeError, f"{name} must be a torch.Tensor, got NoneType")
    m["q is a list"] = (f64, swap("q", [1.0]), TypeError, "q must be a torch.Tensor, got list")
    for name in ("q", "gate", "k", "v", "q_weight", "k_weight"):
        m[f"{name} is fp16"] = (f64, swap(name, lambda o, n=name: o[n].to(torch.float16)), TypeError, f"{name} must be torch.bfloat16 .*got torch.float16")
    m["cos_sin is fp32"] = (
        f64,
        swap("cos_sin", lambda o: o["cos_sin"].float()),
        NotImplementedError,
        "serves a torch.bfloat16 cos_sin table .*another byte class.*got torch.float32",
    )
    # THE CONSTRUCTION FACTS at the binding: an integer tensor of the other dtype is a TypeError naming the entry's
    m["int32 positions on an int64 entry"] = (
        f64,
        swap("positions", lambda o: o["positions"].to(torch.int32)),
        TypeError,
        r"positions must be torch.int64 \(the entry's position_dtype",
    )
    m["int64 positions on an int32 entry"] = (
        f32,
        swap("positions", lambda o: o["positions"].to(torch.int64)),
        TypeError,
        r"positions must be torch.int32 \(the entry's position_dtype",
    )
    m["float positions"] = (f64, swap("positions", lambda o: o["positions"].float()), TypeError, "positions must be torch.int64")
    m["int32 slots on an int64 entry"] = (
        f64,
        swap("slot_mapping", lambda o: o["slot_mapping"].to(torch.int32)),
        TypeError,
        r"slot_mapping must be torch.int64 \(the entry's slot_dtype",
    )
    m["int64 slots on an int32 entry"] = (
        f32,
        swap("slot_mapping", lambda o: o["slot_mapping"].to(torch.int64)),
        TypeError,
        r"slot_mapping must be torch.int32 \(the entry's slot_dtype",
    )
    # shapes
    m["q of another head count"] = (f64, swap("q", lambda o: o["q"][:, :20]), ValueError, r"q must be \[tokens, 24, 256\]; got \(5, 20, 256\)")
    m["q of another width"] = (f64, swap("q", lambda o: o["q"][..., :128]), ValueError, r"q must be \[tokens, 24, 256\]")
    m["q of rank 2"] = (f64, swap("q", lambda o: o["slab"][:, :6144]), ValueError, r"q must be \[tokens, 24, 256\]")
    m["zero tokens"] = (f64, lambda o: bind({k: (v[:0] if k == "slab" else (v[..., :0] if k == "positions" else (v[:0] if k == "slot_mapping" else v))) for k, v in o.items() if k in ("slab", "q_weight", "k_weight", "cos_sin", "positions", "cache", "slot_mapping")}, f64), ValueError, "a call holds at least one token")  # fmt: skip
    m["gate of fewer tokens"] = (
        f64,
        swap("gate", lambda o: o["gate"][:4]),
        ValueError,
        r"gate must be \[5, 24, 256\] \(q holds 5 tokens\); got \(4, 24, 256\)",
    )
    m["k of the query head count"] = (f64, swap("k", lambda o: o["q"]), ValueError, r"k must be \[5, 4, 256\]")
    m["v of fewer heads"] = (f64, swap("v", lambda o: o["v"][:, :2]), ValueError, r"v must be \[5, 4, 256\]")
    m["q_weight of two rows"] = (f64, swap("q_weight", lambda o: o["q_weight"].reshape(1, D)), ValueError, r"q_weight must be \[256\]; got \(1, 256\)")
    m["k_weight short"] = (f64, swap("k_weight", lambda o: o["k_weight"][:128]), ValueError, r"k_weight must be \[256\]")
    m["cos_sin of half the width"] = (
        f64,
        swap("cos_sin", lambda o: o["cos_sin"][:, :32]),
        ValueError,
        r"cos_sin must be \[positions, 64\] \(32 cosines, then 32 sines per position\)",
    )
    m["one position row on a 3d entry"] = (
        f64,
        swap("positions", lambda o: o["positions"][0]),
        ValueError,
        r"positions must be \[3, 5\] under positions='3d'; got \(5,\)",
    )
    m["three position rows on a 1d entry"] = (
        "1d-p64-i64-i64",
        swap("positions", lambda o: o["positions"].reshape(1, T).expand(3, T)),
        ValueError,
        r"positions must be \[5\] under positions='1d'",
    )
    m["positions of fewer tokens"] = (f64, swap("positions", lambda o: o["positions"][:, :4]), ValueError, r"positions must be \[3, 5\]")
    # layouts
    m["q transposed (last axis not contiguous)"] = (
        f64,
        swap("q", lambda o: torch.zeros(T, D, HEADS, dtype=BF16).transpose(1, 2)),
        NotImplementedError,
        "needs q contiguous along its last axis",
    )
    m["q with a 2-byte token stride"] = (
        f64,
        swap("q", lambda o: torch.zeros(T * 6145, dtype=BF16).as_strided((T, HEADS, D), (6145, D, 1))),
        ValueError,
        r"q stride\[0\] = 6145 elements \(12290 bytes\) must be a multiple of 16 bytes",
    )
    m["k with a head stride of 257 elements"] = (
        f64,
        swap("k", lambda o: torch.zeros(T * 2048, dtype=BF16).as_strided((T, KV_HEADS, D), (2048, 257, 1))),
        ValueError,
        r"k stride\[1\] = 257 elements",
    )
    m["q_weight strided"] = (f64, swap("q_weight", lambda o: torch.zeros(2 * D, dtype=BF16)[::2]), NotImplementedError, "needs q_weight contiguous")
    m["cos_sin strided"] = (f64, swap("cos_sin", lambda o: torch.zeros(8, 2 * ROTARY, dtype=BF16)[:, ::2]), NotImplementedError, "needs cos_sin contiguous")
    m["positions strided along tokens"] = (
        f64,
        swap("positions", lambda o: torch.zeros(3, 2 * T, dtype=torch.int64)[:, ::2]),
        NotImplementedError,
        "needs positions contiguous along its token axis",
    )
    for name, shape in (("q", (T, HEADS, D)), ("k", (T, KV_HEADS, D)), ("q_weight", (D,)), ("cos_sin", (8, ROTARY))):
        m[f"{name} misaligned"] = (
            f64,
            swap(name, lambda o, s=shape: _misaligned(s)),
            ValueError,
            f"{name} must start at a 16-byte aligned address, got data_ptr % 16 = 2",
        )
    m["positions misaligned"] = (f64, swap("positions", lambda o: _misaligned_int64((3, T))), ValueError, "positions must be aligned to its 8-byte element")
    # the outputs
    m["q_out is fp16"] = (f64, lambda o: dict(o, _out=dict(q_out=torch.zeros(T, HEADS, D, dtype=torch.float16))), TypeError, "q_out must be torch.bfloat16")
    m["q_out too short"] = (
        f64,
        lambda o: dict(o, _out=dict(q_out=torch.zeros(T - 1, HEADS, D, dtype=BF16))),
        ValueError,
        r"q_out must be \[>= 5, 24, 256\]; got \(4, 24, 256\)",
    )
    m["gate_out flat"] = (f64, lambda o: dict(o, _out=dict(gate_out=torch.zeros(T, HEADS * D, dtype=BF16))), ValueError, r"gate_out must be \[>= 5, 24, 256\]")
    m["q_out strided"] = (
        f64,
        lambda o: dict(o, _out=dict(q_out=torch.zeros(T, HEADS, 2 * D, dtype=BF16)[..., :D])),
        NotImplementedError,
        "needs q_out contiguous",
    )
    m["gate_out misaligned"] = (
        f64,
        lambda o: dict(o, _out=dict(gate_out=_misaligned((T, HEADS, D)))),
        ValueError,
        "gate_out must start at a 16-byte aligned address",
    )
    m["q_out is not a tensor"] = (f64, lambda o: dict(o, _out=dict(q_out="out")), TypeError, "q_out must be a torch.Tensor, got str")
    # the pools and the slot vector: the paged contract's own declines (ValueError, naming the operand)
    m["k_pool is not a tensor"] = (f64, swap("k_pool", None), ValueError, r"k_pool must be a \[num_pages, 4, 64, 256\] torch.bfloat16 tensor, got NoneType")
    m["v_pool is fp16"] = (f64, swap("v_pool", lambda o: o["v_pool"].to(torch.float16)), ValueError, "v_pool must be torch.bfloat16")
    m["a pool of the other page size on a page-64 entry"] = (
        f64,
        swap("k_pool", lambda o: canary_cache(8, 16)[..., :D]),
        ValueError,
        r"k_pool must be \[num_pages >= 1, H=4, page_size=64, D=256\].*got \(8, 4, 16, 256\)",
    )
    m["a pool of the other page size on a page-16 entry"] = (
        f32,
        swap("v_pool", lambda o: canary_cache(2, 64)[..., D:]),
        ValueError,
        r"v_pool must be \[num_pages >= 1, H=4, page_size=16, D=256\].*got \(2, 4, 64, 256\)",
    )
    m["k_pool of other heads"] = (f64, swap("k_pool", lambda o: o["k_pool"][:, :2]), ValueError, r"k_pool must be \[num_pages >= 1, H=4, page_size=64, D=256\]")
    m["v_pool not contiguous along D"] = (
        f64,
        swap("v_pool", lambda o: torch.zeros(2, KV_HEADS, 64, 2 * D, dtype=BF16)[..., ::2]),
        ValueError,
        "v_pool must be contiguous along D",
    )
    m["k_pool with a row stride of 257 elements"] = (
        f64,
        swap("k_pool", lambda o: torch.zeros(2 * KV_HEADS * 64 * 257, dtype=BF16).as_strided((2, KV_HEADS, 64, D), (KV_HEADS * 64 * 257, 64 * 257, 257, 1))),
        ValueError,
        r"k_pool stride\[2\] = 257 elements \(514 bytes\) must be a multiple of 16 bytes",
    )
    m["slot_mapping is not a tensor"] = (f64, swap("slot_mapping", [0, 1, 2, 3, 4]), ValueError, "slot_mapping must be a .* int32 / int64 tensor")
    m["slot_mapping is float"] = (f64, swap("slot_mapping", lambda o: o["slot_mapping"].float()), ValueError, "slot_mapping must be int32 or int64")
    m["slot_mapping longer than the call"] = (
        f64,
        swap("slot_mapping", lambda o: torch.arange(T + 1)),
        ValueError,
        r"slot_mapping holds at most one slot per token \(5\); got 6 entries",
    )
    m["slot_mapping of rank 2"] = (f64, swap("slot_mapping", lambda o: o["slot_mapping"].reshape(1, T)), ValueError, "slot_mapping must be 1-D")
    m["slot_mapping strided"] = (f64, swap("slot_mapping", lambda o: torch.arange(2 * T)[::2]), ValueError, "slot_mapping must be contiguous")
    # overlap of a written operand with another operand
    m["q_out is the query rows' storage"] = (
        f64,
        lambda o: (lambda t: dict(o, q=t, _out=dict(q_out=t)))(torch.zeros(T, HEADS, D, dtype=BF16)),
        ValueError,
        "q_out must not overlap q$",
    )
    m["q_out and gate_out are one tensor"] = (
        f64,
        lambda o: (lambda t: dict(o, _out=dict(q_out=t, gate_out=t)))(torch.zeros(T, HEADS, D, dtype=BF16)),
        ValueError,
        "q_out must not overlap gate_out",
    )
    m["the two pools are one tensor"] = (f64, swap("v_pool", lambda o: o["k_pool"]), ValueError, "k_pool must not overlap v_pool")
    m["the pools are the cache halves shifted by half a row"] = (
        f64,
        swap("v_pool", lambda o: o["cache"][..., D // 2 : D // 2 + D]),
        ValueError,
        "k_pool must not overlap v_pool",
    )
    m["the weight lives in the pool"] = (f64, swap("q_weight", lambda o: o["cache"][0, 0, 0, :D]), ValueError, "k_pool must not overlap q_weight")
    m["the table lives in an output"] = (
        f64,
        lambda o: (lambda t: dict(o, cos_sin=t.reshape(-1)[: 8 * ROTARY].reshape(8, ROTARY), _out=dict(gate_out=t)))(torch.zeros(T, HEADS, D, dtype=BF16)),
        ValueError,
        "gate_out must not overlap cos_sin",
    )
    return m


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_typed_declines_name_the_operand(name):
    """One mutation of served operands per cell: the decline's type and its text, which names the operand.  The pools and the slot
    vector decline through the paged-pool contract (``ValueError``, also for a non-tensor and a dtype): a pool of another page size
    than the ENTRY's is named with the entry's page size."""
    _needs_paged_contract()
    form, mutate, exc, pattern = _mutations()[name]
    ops = mutate(make_operands(5, form))
    outs = ops.pop("_out", {})
    with pytest.raises(exc, match=pattern):
        _check(form, ops, **outs)


def test_declines_run_in_front_of_the_paged_contract_without_the_dsl(monkeypatch):
    """A host without the CuTe DSL: the checks in front of the pools run and decline by their own type; a call that reaches the
    pools gets the typed decline that names the missing DSL (the paged-pool contract's package imports it), never an import error."""
    from cudnn.frost import buffers

    ops = make_operands(3, DEFAULT_FORM)

    def absent():
        raise NotImplementedError("the attention preparation decode step requires nvidia-cutlass-dsl >= 4.8.0; the CuTe DSL is not installed")

    monkeypatch.setattr(entry, "_paged_contract", absent)
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    with pytest.raises(TypeError, match="q must be torch.bfloat16"):
        _check(DEFAULT_FORM, dict(ops, q=ops["q"].float()))
    with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8.0; the CuTe DSL is not installed"):
        _check(DEFAULT_FORM, ops)


def test_overlap_rule_separates_views_that_interleave_in_one_allocation():
    """The overlap rule on footprints (start, end, contiguous run, period): two views whose bounding ranges intersect are served
    exactly when every byte of one lies in a residue window modulo the common period that the other never reaches -- the two halves
    of a packed cache, at either page size and whatever the allocation's address -- and declined when the windows meet (the halves
    shifted, one pool twice, a contiguous tensor inside the other's rows).  A layout the rule cannot separate is declined."""
    share = entry._share_a_byte
    foot = entry._footprint
    for page in (64, 16):
        for pages in (1, 3):
            cache = canary_cache(pages, page)
            k_pool, v_pool = cache[..., :D], cache[..., D:]
            a, b = foot(k_pool), foot(v_pool)
            assert a[2] == b[2] == 2 * D and a[3] == b[3] == 4 * D, (a, b)  # a run of D elements every 2 D elements
            assert a[0] < b[1] and b[0] < a[1], "the bounding ranges intersect"
            assert not share(a, b) and not share(b, a)
            assert share(a, foot(cache[..., D // 2 : D // 2 + D])) and share(a, a) and share(a, foot(cache))
            assert share(a, foot(cache[0, 0, 0, :D])) and not share(a, foot(cache[0, 0, 0, D:])), "a row of the other half shares nothing"
    whole = torch.zeros(4, HEADS, D, dtype=BF16)
    assert foot(whole)[2:] == (4 * HEADS * D * 2, 0), "a contiguous tensor is ONE run"
    assert foot(whole, (2, HEADS, D))[1] - foot(whole)[0] == 2 * HEADS * D * 2, "the live tokens of an output bound its footprint"
    other = torch.zeros(4, HEADS, D, dtype=BF16)
    assert not share(foot(whole), foot(other)) and share(foot(whole), foot(whole[1:]))
    empty = torch.zeros(0, dtype=torch.int64)
    assert not share(foot(empty), foot(whole)), "an empty slot vector touches nothing"
    slab = torch.zeros(3, 14336, dtype=BF16)
    pairs = slab[:, : HEADS * 2 * D].view(3, HEADS, 2 * D)
    q, gate = pairs[..., :D], pairs[..., D:]
    assert foot(q)[2:] == (2 * D, 4 * D) and not share(foot(q), foot(gate)), "the query rows and the gate rows of one slab share no byte"


def _intercepted_entry(monkeypatch, form=DEFAULT_FORM, tokens=3):
    """A host entry whose compile and launch are stood in for: the binding checks and the entry-owned overlap check run for real."""
    k = make_entry("cpu", form)
    launched = []
    monkeypatch.setattr(k, "compile", lambda rows=None: k.compiled.setdefault(k.key(), object()))
    k.buffers[tokens] = {name: torch.zeros(tokens, HEADS, D, dtype=BF16) for name in entry.OUTPUT_SLOTS}
    return k, launched


def test_declines_an_operand_that_overlaps_an_entry_owned_output(monkeypatch):
    """An output the call does not pass is the entry's own buffer of the token count, an operand of the launch like a passed one: a
    tensor an earlier call returned that comes back as an input (here: as the table's storage and as a weight's) is declined by name
    before the launch; with that output passed by the caller the same operands pass the check."""
    _needs_paged_contract()
    tokens = 3
    k, _ = _intercepted_entry(monkeypatch, tokens=tokens)
    ops = make_operands(tokens, DEFAULT_FORM)
    owned = k.buffers[tokens]
    table = owned["q_out"].reshape(-1)[: 8 * ROTARY].reshape(8, ROTARY)
    with pytest.raises(ValueError, match="the entry-owned q_out must not overlap cos_sin .*pass a caller-owned output"):
        k(*call_args(dict(ops, cos_sin=table, positions=torch.zeros_like(ops["positions"]))))
    weight = owned["gate_out"].reshape(-1)[:D]
    with pytest.raises(ValueError, match="the entry-owned gate_out must not overlap k_weight"):
        k(*call_args(dict(ops, k_weight=weight)))
    mine = fresh_outputs(tokens, "cpu")
    rows = entry.check_binding(k.cfg, *call_args(dict(ops, k_weight=weight)), **mine, device="cpu")
    k._check_entry_owned(rows, dict(zip(entry.OPERAND_NAMES, [*call_args(dict(ops, k_weight=weight)), mine["q_out"], mine["gate_out"]])))


def test_compile_and_outputs_decline_a_bad_row_count():
    k = make_entry("cpu")
    for call in (k.compile, k.outputs):
        with pytest.raises(ValueError, match="rows must be at least 1"):
            call(0)
        with pytest.raises(NotImplementedError, match="at most 1024 rows"):
            call(1025)
        with pytest.raises(TypeError, match="rows must be an int"):
            call(2.0)
    with pytest.raises(NotImplementedError, match="serves CUDA devices only; the entry's device is cpu"):
        k.compile()
    with pytest.raises(NotImplementedError, match="serves CUDA devices only"):
        k.compile(3)
    assert not k.compiled and not k.buffers, "a declined compile allocates nothing"
    assert set(k.outputs(2)) == {"q_out", "gate_out"} and tuple(k.outputs(2)["q_out"].shape) == (2, HEADS, D) and k.outputs(2)["q_out"].dtype == BF16


def test_declines_an_unserved_compute_capability_by_name():
    _needs_paged_contract()
    ops = make_operands(3, DEFAULT_FORM)
    for cc in ((9, 0), (8, 0), (12, 0), (12, 1), (11, 0)):
        with pytest.raises(NotImplementedError, match=rf"serves compute capability 10.x \(SM100-class\); this device is {cc[0]}.{cc[1]}"):
            _check(DEFAULT_FORM, ops, device_cc=cc)
        assert not entry.cc_supported(cc)
    assert all(entry.cc_supported(cc) for cc in ((10, 0), (10, 3), (10, 7))) and not entry.cc_supported(None)
    with pytest.raises(NotImplementedError, match="serves CUDA tensors only; the operands are on cpu"):
        _check(DEFAULT_FORM, ops, device_cc=None)


def test_compile_declines_an_unserved_part_on_the_entrys_own_path(monkeypatch):
    """The capability decline a CALLER reaches: on MODELLED part facts (an entry for ``cuda:0`` with the device's compute capability
    patched: no device is touched) ``compile()`` and ``compile(rows)`` on an unserved part are the typed decline by name, raised in
    front of the DSL gates, and nothing is compiled or allocated.  A served capability passes this gate and reaches the next one (the
    DSL, modelled as missing)."""
    from cudnn.frost import buffers

    for cc in ((12, 0), (9, 0), (8, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        k = make_entry("cuda:0")
        for rows in (None, 3):
            with pytest.raises(NotImplementedError, match=rf"serves compute capability 10\.x \(SM100-class\); this device is {cc[0]}\.{cc[1]}"):
                k.compile(rows)
        assert not k.compiled and not k.buffers, "a declined compile builds and allocates nothing"
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    for cc in ((10, 0), (10, 3), (10, 7)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        k = make_entry("cuda:0")
        with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8.0; the CuTe DSL is not installed"):
            k.compile()
        assert not k.compiled and not k.buffers


def test_compile_key_is_plan_time_only_and_two_keys_are_two_entries():
    """One key per entry whatever the token count; it changes with each field of the config -- with each of the three construction
    facts in particular: a caller that needs two page sizes or two integer dtypes builds two entries."""
    k = make_entry("cpu", arm=(True, None))
    assert k.key() == (k.cfg,) and all(k.cfg_for(rows) is k.cfg for rows in (1, 17, kcfg.MAX_ROWS))
    keys = {k.key()}
    for change in (
        dict(EPS=1e-5),
        dict(POSITIONS="1d", SECTION=None),
        dict(SECTION=(12, 10, 10)),
        dict(PAGE_SIZE=16),
        dict(POSITION_DTYPE="int32"),
        dict(SLOT_DTYPE="int32"),
        dict(PDL_RELEASE=False),
        dict(PDL=False, PDL_RELEASE=False),
    ):
        keys.add((kcfg.validate_cfg(replace(k.cfg, **change)),))
    assert len(keys) == 9
    assert len({make_entry("cpu", form).key() for form in FORMS}) == len(FORMS)
    with pytest.raises(NotImplementedError, match="at most 1024 rows"):
        k.cfg_for(kcfg.MAX_ROWS + 1)


def test_launch_arguments_are_addresses_and_byte_strides():
    """The kernel's runtime arguments of a served call, from the operands' metadata alone: thirteen addresses in call order, the byte
    strides of the four row operands (per token, per head), of the position rows and of the two pools (per page, head, page row), the
    slot vector's length, the capacity of the SMALLER pool in slots and the token count."""
    tokens = 5
    ops = make_operands(tokens, DEFAULT_FORM, pages=3)
    outs = fresh_outputs(tokens, "cpu")
    cfg = _cfg()
    args = entry.launch_arguments(cfg, tokens, *call_args(ops), outs["q_out"], outs["gate_out"])
    assert all(type(v) is int for v in args) and len(args) == 13 + 15 + 3
    assert list(args[:13]) == [t.data_ptr() for t in (*call_args(ops), outs["q_out"], outs["gate_out"])]
    assert args[13:21] == (28672, 1024, 28672, 1024, 28672, 512, 28672, 512), "the slab's token stride; a query head's [q | gate] pair; a key / value head row"
    assert args[21] == (tokens + 17) * 8, "the byte stride between the position rows (the tensor is cut out of a wider one)"
    assert args[22:28] == (4 * 64 * 1024, 64 * 1024, 1024) * 2 and args[28:] == (tokens, 3 * 64, tokens)
    one = make_operands(tokens, "1d-p16-i64-i32", pages=3)
    small = dict(one, k_pool=one["cache"][:2, ..., :D])
    args = entry.launch_arguments(_cfg("1d-p16-i64-i32"), tokens, *call_args(small), outs["q_out"], outs["gate_out"])
    assert (
        args[21] == 0 and args[22:25] == (4 * 16 * 1024, 16 * 1024, 1024) and args[29] == 2 * 16
    ), "one position row: no row stride; the smaller pool bounds the slots"
    empty = dict(ops, slot_mapping=torch.zeros(0, dtype=torch.int64))
    assert entry.launch_arguments(cfg, tokens, *call_args(empty), outs["q_out"], outs["gate_out"])[28] == 0


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the launch facts (cudnn.frost.launch_facts) -- the symbol of the compiled form, the form as plain data, the launch shape
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_facts_equal_the_configuration_arithmetic_at_every_row_count(arm, arm_id):
    """EVERY served token count, 1 .. 1024, per form and launch arm: the launch shape is one thread block per (token, head) over the
    24 query and the 4 key / value heads, 128 threads, the kernel's one shared-memory buffer of 16 bytes; the form is the config's
    fields with the geometry as a nested dict -- nothing else, because the compile key is the config alone."""
    for form in FORMS:
        k = make_entry("cpu", form, arm)
        want_form = asdict(k.cfg)
        for rows in range(1, kcfg.MAX_ROWS + 1):
            shape = k.launch_shape_for(rows)
            assert shape == ((rows, 28, 1), (128, 1, 1), 16), (form, rows, shape)
            assert all(type(v) is int for v in (*shape.grid, *shape.block, shape.shared_memory)), shape
            assert k.form_for(rows) == want_form, (form, rows)
        positions, page, pd, sd = FORMS[form]
        assert want_form == dict(
            GEOMETRY=dict(HEADS=24, KV_HEADS=4, HEAD_DIM=256, ROTARY_DIM=64),
            EPS=1e-6,
            POSITIONS=positions,
            SECTION=SECTION if positions == "3d" else None,
            PAGE_SIZE=page,
            POSITION_DTYPE=pd,
            SLOT_DTYPE=sd,
            PDL=bool(arm[0]),
            PDL_RELEASE=arm_id == "release-wait",
        )


def test_launch_facts_are_plain_fresh_and_decline_an_unserved_row_count():
    k = make_entry("cpu", arm=(True, None))
    form = k.form_for(7)
    assert json.loads(json.dumps(form)) == json.loads(json.dumps(k.form_for(7))) and type(form) is dict
    form["PAGE_SIZE"] = 16
    form["GEOMETRY"]["HEADS"] = 1
    assert k.form_for(7)["PAGE_SIZE"] == 64 and k.form_for(7)["GEOMETRY"]["HEADS"] == 24 and k.cfg.PAGE_SIZE == 64
    for fact in (k.kernel_symbols_for, k.form_for, k.launch_shape_for):
        for rows in (0, -1):
            with pytest.raises(ValueError, match="rows must be at least 1"):
                fact(rows)
        with pytest.raises(NotImplementedError, match="at most 1024 rows.*got 1025"):
            fact(1025)
        for rows in (1.0, "1", None, True):
            with pytest.raises(ValueError, match="rows must be an int"):
                fact(rows)
        fact(1024)


class _CompiledStandIn:
    """What the entry keeps per compile key, as far as the launch facts read it: a compiled DSL host function carries ``kernel_info``,
    keyed by the symbols of the kernels it launches."""

    def __init__(self, *symbols):
        self.kernel_info = {symbol: {} for symbol in symbols}


def test_kernel_symbol_is_that_of_the_compiled_form_at_every_row_count():
    k = make_entry("cpu")
    assert k.kernel_symbols_for(1) == () == k.kernel_symbols_for(kcfg.MAX_ROWS)
    k.compiled[(replace(k.cfg, PAGE_SIZE=16),)] = _CompiledStandIn(SYMBOL + "other_0")
    assert k.kernel_symbols_for(1) == (), "a form the entry does not run is not named"
    k.compiled[k.key()] = _CompiledStandIn(SYMBOL + "A_0")
    assert all(k.kernel_symbols_for(rows) == (SYMBOL + "A_0",) for rows in (1, 2, 160, kcfg.MAX_ROWS))
    k.compiled[k.key()] = object()
    with pytest.raises(NotImplementedError, match=r"the attention preparation decode step: the installed CuTe DSL .* does not expose the kernel symbols"):
        k.kernel_symbols_for(1)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the DSL gates (installed -> version floor -> the part's target -> the compiler that assembles it)
# ---------------------------------------------------------------------------------------------------------------------------
def test_declines_a_dsl_below_the_floor_by_version(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    for msg in (entry._dsl_version_error(), entry._dsl_error((10, 3)), entry._dsl_error((10, 7))):
        assert "requires nvidia-cutlass-dsl >= 4.8.0" in msg and "found 4.7.1" in msg
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    assert entry._dsl_version_error() is None


def test_declines_a_missing_dsl(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    for msg in (entry._dsl_version_error(), entry._dsl_error((10, 3))):
        assert "requires nvidia-cutlass-dsl >= 4.8.0" in msg and "not installed" in msg


def test_declines_a_dsl_without_the_parts_target_by_name(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    msg = entry._dsl_error((10, 7))
    assert msg is not None and "sm_107a" in msg and "nvidia-cutlass-dsl 4.8.0" in msg and msg.startswith(kcfg.NAME)
    assert entry._dsl_target_error((10, 3)) is None and entry._dsl_target_error((10, 0)) is None
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))

    def unreachable():
        raise AssertionError("the target probe ran although the version gate declined")

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", unreachable)
    assert "found 4.7.0" in entry._dsl_error((10, 7))


def test_declines_a_compiler_that_rejects_the_parts_target(monkeypatch):
    """cc 10.7 under a DSL whose compiler is the CUDA 12 toolchain flavour: declined BEFORE JIT, naming the compiler found and the one
    required.  cc 10.0 / 10.3 are served by both flavours."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (12, 9))
    msg = entry._dsl_error((10, 7))
    assert msg is not None and "compute capability 10.7" in msg and "CUDA 13.4 or later" in msg and "compiles with CUDA 12.9" in msg and "cu13" in msg
    assert entry._dsl_error((10, 3)) is None and entry._dsl_error((10, 0)) is None
    for served in ((13, 4), (13, 6), (14, 0)):
        monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda v=served: v)
        assert entry._dsl_error((10, 7)) is None


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: exports, the import boundary, the kernel's source and the docs rows
# ---------------------------------------------------------------------------------------------------------------------------
def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.AttentionPrepareDecode is entry.AttentionPrepareDecode
    frost = importlib.import_module("cudnn.attention_prepare.frost")
    family = importlib.import_module("cudnn.attention_prepare")
    assert frost.AttentionPrepareDecode is entry.AttentionPrepareDecode and frost.Cfg is kcfg.Cfg
    assert frost.AttentionPrepareGeometry is kcfg.AttentionPrepareGeometry is family.AttentionPrepareGeometry is entry.AttentionPrepareGeometry
    assert family.AttentionPrepareDecode is entry.AttentionPrepareDecode
    assert "AttentionPrepareDecode" in cudnn._LAZY_OPTIONAL_IMPORTS and "AttentionPrepareDecode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS
    assert entry.SERVED_GEOMETRIES is kcfg.SERVED_GEOMETRIES and entry.SERVED_PAGE_SIZES is kcfg.SERVED_PAGE_SIZES
    with pytest.raises(AttributeError):
        frost.NoSuchName
    with pytest.raises(AttributeError):
        family.NoSuchName


def test_import_cudnn_does_not_import_the_kernel_or_torch():
    """The lazy boundary, in fresh interpreters: importing cudnn loads neither the step's modules nor the DSL nor torch; the config
    module loads neither the DSL nor torch; importing the host entry and constructing an entry (its construction declines included)
    still loads neither the DSL nor the kernel module."""

    def fresh(code):
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()

    seen = "print(sorted(m for m in sys.modules if m.startswith('cudnn.attention_prepare') or m in ('cutlass', 'torch')))"
    assert fresh(f"import sys, cudnn; {seen}") == "[]"
    out = fresh(
        f"import sys, cudnn.attention_prepare.frost, cudnn.attention_prepare.frost.kernel.prepare_decode_config, cudnn.attention_prepare.frost.prepare_decode; {seen}"
    )
    assert "cutlass" not in out and "torch" not in out and "prepare_decode_f16" not in out, out
    code = (
        "import sys, cudnn\n"
        "from cudnn.attention_prepare import AttentionPrepareGeometry as G\n"
        "k = cudnn.AttentionPrepareDecode('cpu', G(24, 4, 256, 64), 1e-6, positions='3d', mrope_section=(11, 11, 10), page_size=64, position_dtype='int64',\n"
        "                                 slot_dtype='int64', pdl=True)\n"
        "try:\n"
        "    cudnn.AttentionPrepareDecode('cpu', G(24, 4, 256, 64), 1e-6, positions='1d', page_size=48, position_dtype='int64', slot_dtype='int64')\n"
        "except ValueError:\n"
        "    pass\n"
        "print(k.launch_shape_for(3).grid, sorted(m for m in sys.modules if m == 'cutlass' or m.endswith('prepare_decode_f16')))\n"
    )
    assert fresh(code) == "(3, 28, 1) []"


def test_kernel_takes_its_instructions_from_the_tile_library_and_names_its_source():
    """One definition per instruction: the kernel module imports the approximate division, the approximate root, the bf16 converts and
    the 32-bit global accesses from ``cudnn.frost.tile_dsl`` and carries no inline PTX of its own; its row reduction is ONE call of
    ``cudnn.frost.tile_dsl.rowwise.block_reduce`` with ``kind="sum", order="descending"`` -- an order the tile library defines -- so the
    module holds no thread-block barrier of its own.  Its header names the source the kernel follows, the commit and the licence, and
    the third-party notices carry the matching entry.  Read from the source: no DSL import."""
    source = (REPO / "python" / "cudnn" / "attention_prepare" / "frost" / "kernel" / "prepare_decode_f16.py").read_text()
    tree = ast.parse(source)
    imported = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cudnn.frost.tile_dsl")
        for alias in node.names
    }
    assert {
        ("cudnn.frost.tile_dsl.pointwise", n) for n in ("div_full_f32", "rsqrt_approx_ftz_f32", "opaque_f32_bits", "f16x2_to_f32", "fp32_to_fp16")
    } <= imported
    assert {("cudnn.frost.tile_dsl.tma", n) for n in ("ld_global", "ld_global_v2", "st_global")} <= imported
    assert {("cudnn.frost.tile_dsl.barrier", n) for n in ("launch_dependent_grids", "wait_on_dependent_grids")} <= imported
    assert ("cudnn.frost.tile_dsl.rowwise", "block_reduce") in imported
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert defined == {"_load_int", "frost_attention_prepare", "launch"}, defined
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "inline_ptx" not in names, "the kernel spells an instruction inline instead of through the tile library"
    assert "barrier" not in names, "the kernel spells the thread-block barrier of its reduction itself instead of through block_reduce"
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "block_reduce"]
    assert len(calls) == 1 and {kw.arg: getattr(kw.value, "value", None) for kw in calls[0].keywords} == {"kind": "sum", "order": "descending"}
    library = ast.parse((REPO / "python" / "cudnn" / "frost" / "tile_dsl" / "rowwise.py").read_text())
    orders = [
        ast.literal_eval(n.value) for n in library.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "SUM_ORDERS"
    ]
    named = set(re.findall(r'order="(\w+)"', source))
    assert (
        len(orders) == 1 and "descending" in named and named <= set(orders[0])
    ), f"the kernel module names a sum order the tile library does not define: {sorted(named)}"
    head = source[: source.index('"""')]
    for needle in ("vllm/model_executor/layers/fused_qk_norm_rope.py", "2a777ce26e8315fa503ac5e88b595235d1c2b928", "Apache-2.0", "has\n# been modified"):
        assert needle in head, f"the kernel module's header does not carry {needle!r}"
    notices = (REPO / "THIRD_PARTY_LICENSES.txt").read_text()
    assert "python/cudnn/attention_prepare/frost/kernel/prepare_decode_f16.py" in notices and "2a777ce26e8315fa503ac5e88b595235d1c2b928" in notices


GOLDEN_CASES = ("3d-p64-i64-i64-full-rows9", "3d-p64-i64-i64-short-rows9", "1d-p16-i64-i32-full-rows9", "1d-p16-i64-i32-short-rows9")


def test_docs_rows_exist_for_the_api():
    """The determinism status page carries the API's ONE row -- SM90 unsupported (declined by type); SM100/103 split by the evidence;
    SM107 unverified -- and its bullet under the required settings; the goldens README its suite row and section; the operation page,
    its navigation entry and the overview and llms.txt links exist; and the four golden cases are committed for at least one runtime."""
    det = (REPO / "docs" / "operations" / "determinism.md").read_text()
    row = [ln for ln in det.splitlines() if ln.startswith("| `AttentionPrepareDecode`")]
    assert len(row) == 1, "one AttentionPrepareDecode row on the status page"
    cells = [c.strip() for c in row[0].strip("|").split("|")]
    assert cells[1] == "fprop (inference step)" and cells[2] == "BF16 in and out; FP32 accumulate", cells
    assert cells[3] == "Unsupported" and cells[4] == "SM100: Supported; SM103: Unverified" and cells[5] == "Unverified", cells
    assert "- Attention preparation decode step (`AttentionPrepareDecode`):" in det
    readme = (REPO / "test" / "python" / "determinism_goldens" / "README.md").read_text()
    assert "AttentionPrepareDecode" in readme and "attention_prepare/frost/test_attention_prepare_decode_determinism.py" in readme
    assert "### Attention preparation decode step:" in readme
    page = REPO / "docs" / "operations" / "attention" / "attention_prepare_decode.md"
    assert page.exists() and "cudnn.AttentionPrepareDecode" in page.read_text()
    assert "../../operations/attention/attention_prepare_decode.md" in (REPO / "docs" / "fern" / "versions" / "latest.yml").read_text()
    assert "/operations/Attention/attention_prepare_decode" in (REPO / "docs" / "operations" / "overview.md").read_text()
    assert "docs/operations/attention/attention_prepare_decode.md" in (REPO / "llms.txt").read_text()
    goldens = {p.name for p in (REPO / "test" / "python" / "determinism_goldens").glob("*/attention-prepare-decode-*.json")}
    assert goldens == {f"attention-prepare-decode-{case}.json" for case in GOLDEN_CASES}, goldens


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the rendering (the instructions that define the sum tree and the rounding chain), per claimed target
# ---------------------------------------------------------------------------------------------------------------------------
_RENDER_PROBE = textwrap.dedent("""
    import os, sys
    dump, arch = sys.argv[1], sys.argv[2]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_ARCH"] = arch  # unconditional: an inherited value would render another target
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.version_info import CUDA_VERSION
    from cutlass.cute.runtime import make_fake_stream
    from cudnn.attention_prepare.frost import prepare_decode as entry
    from cudnn.attention_prepare.frost.kernel import prepare_decode_config as kcfg
    from cudnn.attention_prepare.frost.kernel import prepare_decode_f16 as mod

    print("COMPILER", int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    for spec in sys.argv[3:]:
        before = set(os.listdir(dump))
        positions, page, pd, sd, pdl, release = (None if p == "None" else (p == "True" if p in ("True", "False") else p) for p in spec.split("|"))
        cfg = entry.make_cfg(kcfg.AttentionPrepareGeometry(24, 4, 256, 64), 1e-6, positions=positions, mrope_section=(11, 11, 10) if positions == "3d" else None,
                             page_size=int(page), position_dtype=pd, slot_dtype=sd, pdl=pdl, pdl_release=release)
        # the plan-time arguments of the ENTRY's own compile: addresses and strides are runtime values, the zeros pin their types
        args = [cutlass.Int64(0)] * (len(mod.OPERANDS) + len(mod.STRIDES)) + [cutlass.Int32(0), cutlass.Int64(0), cutlass.Int32(1)]
        cute.compile(mod.launch, cfg, *args, make_fake_stream())
        for name in sorted(set(os.listdir(dump)) - before):
            print("KEPT", spec, name)
    print("COMPILED", arch)
    """)


def _spec(form, arm):
    positions, page, pd, sd = FORMS[form]
    return "|".join(str(v) for v in (positions, page, pd, sd, arm[0], arm[1]))


def _render(tmp_path, arch, specs):
    """Run the probe for ``arch`` in a fresh process (the DSL reads its dump settings once, at the first import); returns the
    compiler's CUDA major and ``{spec: [kept file paths]}``.  A compile failure FAILS."""
    probe = tmp_path / "render_probe.py"  # a FILE, not ``python -c``: the DSL preprocessor re-reads a kernel's source through inspect
    probe.write_text(_RENDER_PROBE)
    dump = tmp_path / "dump"
    dump.mkdir()
    proc = subprocess.run([sys.executable, str(probe), str(dump), arch, *specs], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"trace-compile for {arch} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED {arch}" in lines, proc.stdout[-2000:]
    major = int(next(ln for ln in lines if ln.startswith("COMPILER")).split()[1])
    kept = {}
    for ln in lines:
        if ln.startswith("KEPT "):
            _, spec, name = ln.split(" ", 2)
            kept.setdefault(spec, []).append(str(dump / name))
    return major, kept


def _nvdisasm_candidates():
    import shutil

    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sass(cubin):
    """The disassembly of a cubin by the first nvdisasm that decodes it, or None (a toolkit older than the target decodes nothing)."""
    for nvd in _nvdisasm_candidates():
        try:
            proc = subprocess.run([nvd, "-c", cubin], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.splitlines()
    return None


def _instructions(ptx):
    """The instruction lines of a rendering's entry, in text order: no comments, no blank lines, no register declarations, with
    every inline block's braces kept on its own line as rendered."""
    body = ptx[ptx.index("{", ptx.index(".entry")) :]
    return [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("//") and not ln.strip().startswith(".reg")]


def _arm_neutral(lines):
    """A rendering's instruction lines without its ``griddepcontrol`` instructions and with register numbers masked: what two launch
    arms of one form must share, as a sorted multiset (the arms place the same instructions; an arm may order address arithmetic
    differently around its wait)."""
    return sorted(re.sub(r"%(r|rd|p|rs)\d+", r"%\1", re.sub(SYMBOL + r"\w+", "ENTRY", ln)) for ln in lines if "griddepcontrol" not in ln)


@pytest.mark.parametrize("arch, cc", (("sm_100a", (10, 0)), ("sm_103a", (10, 3)), ("sm_107a", (10, 7))))
def test_rendering_pins_the_convention(arch, cc, tmp_path):
    """Trace-compile the kernel for a claimed target on this box (no device needed: the kernel takes addresses) in every form of the
    suite x three launch arms and pin the instructions that ARE the contract.  In EVERY rendering: ONE ``div.full.f32`` with the width
    256 in a register (the bits ``0x43800000``) and ONE ``rsqrt.approx.ftz.f32`` (never ``div.rn`` / ``div.approx`` /
    ``rsqrt.approx.f32``), eps as ``0f358637BD``; three ``fma.rn.f32`` (the fused leaf and the two rotated values, never a separate
    multiply and add) and seven ``mul.f32``; NINE ``shfl.sync.bfly`` whose offsets read, in program order, 16, 8, 4, 2, 1 (inside a
    warp), 2, 1 (over the four warp sums: the SEVEN shuffles of the reduction) and 16, 16 (the two rotary partners); ONE ``bar.sync``
    with one ``st.shared`` in front of it and one ``ld.shared`` behind it that is NOT under a predicate (every lane loads); two
    ``cvt.rn.bf16x2.f32`` (the rounding of record and the rotated pair); two 32-bit global stores; ``.reqntid 128``; no ``mbarrier`` /
    ``cp.async`` / ``tcgen05``; NO integer ``div`` / ``rem`` (the page size is a compile-time power of two: a shift and a mask); the
    symbol = the kernel's name + the form text inside the first 100 characters; thirty kernel parameters whatever the form.  THE
    LOAD WIDTHS follow the form's integer dtypes: an int64 ``positions`` / ``slot_mapping`` value is ONE ``ld.global.v2.b32`` (three
    position rows: three), an int32 one ONE ``ld.global.b32`` beside the form's 32-bit data loads (weights, row pair, copied pair;
    the table words: four with three position rows, two with one).  THE LAUNCH ARMS: ``griddepcontrol.wait`` exactly under ``pdl``
    and ``launch_dependents`` exactly under ``pdl_release``; the release is in front of every global access; exactly ONE global load
    (the weights) sits in front of the wait and every other global load and both stores behind it; within a form the three arms hold
    the same instructions.  With an nvdisasm that decodes the cubin, also the SASS: one ``BAR.SYNC``, one shared store and load, no
    spill, ONE ``MUFU.RSQ`` directly behind the ``FFMA`` that folds the division by 256 and the eps (a register multiplier and the
    eps as its immediate addend), ``ACQBULK`` / ``PREEXIT`` as the arm says and in that order around the global accesses.  A compile
    failure FAILS; a DSL that does not serve the target (version, target list, compiler) SKIPS with the gate's text."""
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    cells = [(form, arm, arm_id) for form in FORMS for arm, arm_id in zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)]
    major, kept = _render(tmp_path, arch, [_spec(form, arm) for form, arm, _ in cells])
    neutral, symbols = {}, set()
    for form, arm, arm_id in cells:
        files = kept.get(_spec(form, arm), [])
        ptx_files = [f for f in files if f.endswith(".ptx")]
        assert len(ptx_files) == 1, f"{arch} {form} {arm_id}: expected one PTX, kept {files}"
        ptx = Path(ptx_files[0]).read_text()
        lines = _instructions(ptx)
        count = lambda needle: sum(1 for ln in lines if needle in ln)  # noqa: E731
        positions, page, pd, sd = FORMS[form]
        rows = 3 if positions == "3d" else 1
        wide = rows * int(pd == "int64") + int(sd == "int64")
        narrow = 3 + (4 if positions == "3d" else 2) + rows * int(pd == "int32") + int(sd == "int32")
        wait, release = arm_id != "plain", arm_id == "release-wait"
        want = {
            "div.full.f32": 1,
            "rsqrt.approx.ftz.f32": 1,
            "0x43800000": 1,
            "0f358637BD": 1,
            "fma.rn.f32": 3,
            "mul.f32": 7,
            "shfl.sync.bfly": 9,
            "bar.sync": 1,
            "st.shared": 1,
            "ld.shared": 1,
            "cvt.rn.bf16x2.f32": 2,
            "st.global.b32": 2,
            "st.global": 2,
            "ld.global.v2.b32": wide,
            "ld.global.b32": narrow,
            "ld.global": wide + narrow,
            "div.rn": 0,
            "div.approx": 0,
            "rsqrt.approx.f32": 0,
            "mbarrier": 0,
            "cp.async": 0,
            "tcgen05": 0,
            "griddepcontrol.wait": int(wait),
            "griddepcontrol.launch_dependents": int(release),
        }
        what = f"{arch} {form} {arm_id} (compiler CUDA {major})"
        got = {needle: count(needle) for needle in want}
        assert got == want, f"{what}: " + ", ".join(f"{k} {got[k]} (want {want[k]})" for k in want if got[k] != want[k])
        assert not [
            ln for ln in lines if re.match(r"(div|rem)\.[su](16|32|64)\b", ln)
        ], f"{what}: an integer division in the kernel (the page size is a compile-time power of two)"
        assert f".target {arch}" in ptx and ".reqntid 128, 1, 1" in ptx, what
        offsets = [int(re.search(r"shfl\.sync\.bfly\.b32\s+%r\d+, %r\d+, (\d+),", ln).group(1)) for ln in lines if "shfl.sync.bfly" in ln]
        assert offsets == [16, 8, 4, 2, 1, 2, 1, 16, 16], f"{what}: the shuffle offsets in program order are {offsets}"
        at = {needle: next(i for i, ln in enumerate(lines) if needle in ln) for needle in ("st.shared", "bar.sync", "ld.shared")}
        shuffles = [i for i, ln in enumerate(lines) if "shfl.sync.bfly" in ln]
        assert shuffles[4] < at["st.shared"] < at["bar.sync"] < at["ld.shared"] < shuffles[5], f"{what}: the two levels do not bracket the barrier"
        assert lines[at["ld.shared"]].startswith("ld.shared"), f"{what}: the slot load is under a predicate: {lines[at['ld.shared']]}"
        params = len(re.findall(r"^\s*\.param", ptx, re.M))
        assert params == 30, f"{what}: {params} kernel parameters"
        symbol = re.search(r"\.entry\s+(\w+)", ptx).group(1)
        head = SYMBOL + str(_cfg(form, arm))
        assert symbol.startswith(head + "_") and len(head) <= 100, f"{what}: the symbol {symbol[:120]} does not carry the form in its first 100 characters"
        symbols.add(symbol)
        loads = [i for i, ln in enumerate(lines) if "ld.global" in ln]
        stores = [i for i, ln in enumerate(lines) if "st.global" in ln]
        if release:
            first = next(i for i, ln in enumerate(lines) if "griddepcontrol.launch_dependents" in ln)
            assert first < min(loads) and first < min(i for i, ln in enumerate(lines) if "griddepcontrol.wait" in ln), f"{what}: the release is not the first"
            assert not any(re.match(r"(\$\w+:|bra\b)", ln) for ln in lines[:first]), f"{what}: the release is not in the kernel's entry block"
        if wait:
            w = next(i for i, ln in enumerate(lines) if "griddepcontrol.wait" in ln)
            assert not any(re.match(r"(\$\w+:|bra\b)", ln) for ln in lines[:w]), f"{what}: the wait is not in the kernel's entry block"
            assert sum(i < w for i in loads) == 1 and lines[min(loads)].startswith("ld.global.b32"), f"{what}: the global loads in front of the wait"
            assert all(i > w for i in stores), f"{what}: a global store in front of the wait"
        neutral[(form, arm_id)] = _arm_neutral(lines)
        cubins = [f for f in files if f.endswith(".cubin")]
        sass = _sass(cubins[0]) if cubins else None
        if sass is not None:
            ops = [m.group(1) for m in (re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z0-9_.]+)", ln) for ln in sass) if m]
            n = lambda needle: sum(1 for op in ops if op.startswith(needle))  # noqa: E731
            assert n("BAR.SYNC") == 1 and n("STS") == 1 and n("LDS") == 1, f"{what}: the barrier and the warp-sum slot in the SASS"
            assert n("STL") == 0 and n("LDL") == 0, f"{what}: a spill in the SASS"
            assert n("MUFU") == 1 and n("MUFU.RSQ") == 1 and n("FCHK") == 0, f"{what}: the root and no IEEE division in the SASS"
            coded = [ln for ln in sass if re.match(r"\s*/\*[0-9a-f]+\*/", ln)]
            root = next(i for i, ln in enumerate(coded) if "MUFU.RSQ" in ln)
            fused = next(ln for ln in reversed(coded[:root]) if re.search(r"\sF[A-Z0-9_.]+\s", ln))  # the float instruction in front of the root
            assert re.search(
                r"\sFFMA R\d+, R\d+, R\d+, [-+0-9.e]+ ;", fused
            ), f"{what}: in front of the root: {fused.strip()} (want the division by 256 and eps as ONE FFMA)"
            assert n("ACQBULK") == int(wait) and n("PREEXIT") == int(release), what
            if wait:
                acq = ops.index("ACQBULK")
                assert sum(op.startswith("LDG") for op in ops[:acq]) == 1 and not any(
                    op.startswith("STG") for op in ops[:acq]
                ), f"{what}: the SASS in front of the wait"
            if release:
                assert not any(op.startswith(("LDG", "STG")) for op in ops[: ops.index("PREEXIT")]), f"{what}: a global access in front of the release"
    assert len(symbols) == len(cells), "two forms or arms share a kernel symbol"
    for form in FORMS:
        for arm_id in ("wait", "release-wait"):
            assert neutral[(form, arm_id)] == neutral[(form, "plain")], f"{arch} {form}: the {arm_id} arm holds other instructions than the plain launch"


# ---------------------------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------------------------
_ENTRIES = {}


def _entry(dev, form=DEFAULT_FORM, arm=(None, None)):
    """One entry per (device, form, launch arm) for the cells that only launch (an entry is one compiled kernel for every token count)."""
    key = (str(dev), form, arm)
    if key not in _ENTRIES:
        _ENTRIES[key] = make_entry(dev, form, arm)
    return _ENTRIES[key]


def _run(k, ops, extra=0):
    """One launch into fresh caller-owned outputs and a fresh canary cache; returns ``(outs, ops)`` with the written cache bound."""
    tokens = ops["q"].shape[0]
    dev = ops["q"].device
    ops = bind(
        dict(
            {name: ops[name] for name in ("slab", "q_weight", "k_weight", "cos_sin", "positions", "slot_mapping")},
            cache=canary_cache(*ops["cache"].shape[0:3:2]).to(dev),
        ),
        ops["form"],
    )
    outs = fresh_outputs(tokens, dev, extra=extra)
    k(*call_args(ops), **outs)
    torch.cuda.synchronize()
    return outs, ops


def _snap(k, ops):
    outs, ops = _run(k, ops)
    return dict(q_out=outs["q_out"].clone(), gate_out=outs["gate_out"].clone(), cache=ops["cache"].clone())


def _assert_same(got, want, label):
    for name in want:
        a, b = bits(got[name]), bits(want[name])
        assert a.shape == b.shape, f"{label}: {name} shape"
        n = int((a != b).sum())
        assert n == 0, f"{label}: {name} differs in {n} of {a.numel()} bf16 elements"


@pytest.mark.parametrize("kind", ("random", "structured"))
@pytest.mark.parametrize("form", sorted(FORMS))
def test_matches_the_twin(form, kind):
    """Every form at eleven token counts against the PyTorch twin with the kernel's rounding points: pass-through elements within one
    bf16 unit at the larger magnitude, rotated elements inside their band, the gate rows, the value rows and every pool row no slot
    names bitwise; rows behind the call's tokens in a taller caller-owned output hold the sentinel."""
    dev = _device_or_skip()
    k = _entry(dev, form)
    moved = total = 0
    for tokens in TOKENS:
        ops = make_operands(tokens, form, dev, seed=10 * tokens + len(kind), kind=kind)
        outs, ops = _run(k, ops, extra=2)
        moved += assert_twin(ops, outs, k.cfg, label=f"{form} {kind} tokens {tokens}")
        total += tokens * (HEADS + KV_HEADS) * D
    assert moved <= total // 2000, f"{form} {kind}: {moved} of {total} elements are a unit off the twin: more than an approximate root explains"


def test_edge_rows():
    """Finite edge tokens beside random ones: an all-zero token (the root of eps: zero outputs with the operands' signs), negative
    zeros, one dominant element per head row, values whose squares underflow, large magnitudes.  The twin and its band hold; a zero
    row writes zeros."""
    dev = _device_or_skip()
    for form in (DEFAULT_FORM, "1d-p16-i64-i32"):
        k = _entry(dev, form)
        ops = make_operands(9, form, dev, seed=5, kind="edge")
        outs, ops = _run(k, ops)
        assert_twin(ops, outs, k.cfg, label=f"{form} edge")
        assert bool((outs["q_out"][0].float() == 0).all()) and bool((outs["q_out"][1].float() == 0).all()), "a zero row normalizes to zeros"


def test_slot_predicate_decides_which_pool_rows_are_written():
    """The pool store is predicated by ``0 <= slot < pages x page_size`` and by the slot vector's length: negative slots (the padding
    convention), a slot AT the pool's end and one PAST it, a slot vector shorter than the call and an EMPTY one write no pool row --
    every other pool row holds its canary -- while every token's query and gate rows are written; the last slot of the pool is
    written.  With the pools given as two tensors of different page counts the smaller one bounds the slots of both."""
    dev = _device_or_skip()
    for form in (DEFAULT_FORM, "3d-p16-i32-i32"):
        k = _entry(dev, form)
        page = k.cfg.PAGE_SIZE
        pages, tokens = 3, 9
        last = pages * page - 1
        slots = [5, -1, last, last + 1, 0, -(2**31) + 7, last + 4 * page, page, 2 * page - 1]
        ops = make_operands(tokens, form, dev, seed=3, pages=pages, slots=slots)
        outs, ops = _run(k, ops)
        assert_twin(ops, outs, k.cfg, label=f"{form}: negative slots, the pool's end and past it")
        for n in (4, 1, 0):
            short = make_operands(tokens, form, dev, seed=3, pages=pages, slots=slots[:n])
            outs, short = _run(k, short)
            assert_twin(short, outs, k.cfg, label=f"{form}: a slot vector of {n} entries")
            written = int((bits(short["cache"]) != CANARY_BITS).any(dim=-1).sum())
            assert written == KV_HEADS * sum(1 for s in slots[:n] if 0 <= s <= last), f"{form}: {written} pool rows written with {n} slots"
        # two pools of different page counts: a slot inside the taller pool and past the smaller one writes NEITHER
        ops = make_operands(tokens, form, dev, seed=4, pages=pages, slots=[0, page * 2 + 3, 7])
        small = canary_cache(2, page).to(dev)
        split = dict(ops, v_pool=small[..., D:])
        outs = fresh_outputs(tokens, dev)
        k(*call_args(split), **outs)
        torch.cuda.synchronize()
        k_rows = (bits(ops["cache"][..., :D]) != CANARY_BITS).any(dim=-1)  # [pages, KV_HEADS, page]
        v_rows = (bits(small[..., D:]) != CANARY_BITS).any(dim=-1)
        assert int(k_rows.sum()) == 2 * KV_HEADS == int(v_rows.sum()) and not bool(k_rows[2].any()), f"{form}: the smaller pool bounds the slots of both"
        assert bool((bits(ops["cache"][..., D:]) == CANARY_BITS).all()) and bool(
            (bits(small[..., :D]) == CANARY_BITS).all()
        ), "each pool is written through its own view only"


def test_output_ownership_per_output():
    """An output the caller passes is returned as given and written in the call's tokens only; an output the caller does not pass is
    the entry's buffer of that token count -- the same tensor at the next call of that count, another one at another count; the bytes
    are the same whoever owns the storage."""
    dev = _device_or_skip()
    k = make_entry(dev)
    ops = make_operands(6, DEFAULT_FORM, dev, seed=21)
    want = _snap(k, ops)
    q_out, gate_out = k(*call_args(ops))
    torch.cuda.synchronize()
    assert q_out is k.outputs(6)["q_out"] and gate_out is k.outputs(6)["gate_out"] and tuple(q_out.shape) == (6, HEADS, D)
    _assert_same(dict(q_out=q_out, gate_out=gate_out), dict(q_out=want["q_out"], gate_out=want["gate_out"]), "entry-owned outputs")
    mine = fresh_outputs(6, dev, extra=3)
    got_q, got_gate = k(*call_args(ops), q_out=mine["q_out"])
    torch.cuda.synchronize()
    assert got_q is mine["q_out"] and got_gate is k.outputs(6)["gate_out"], "ownership is per output"
    _assert_same(dict(q_out=got_q[:6]), dict(q_out=want["q_out"]), "a caller-owned q_out beside an entry-owned gate_out")
    assert bool((bits(mine["q_out"][6:]) == SENTINEL_BITS).all()), "rows past the call's tokens are untouched"
    two = make_operands(2, DEFAULT_FORM, dev, seed=22)
    other_q, _ = k(*call_args(two))
    assert other_q is not q_out and set(k.buffers) == {2, 6}
    with pytest.raises(ValueError, match="the entry-owned q_out must not overlap q_weight"):
        k(*call_args(dict(ops, q_weight=q_out.reshape(-1)[:D])))


@pytest.mark.parametrize("tokens", (1, 17, 160, 1024))
def test_bit_neutral_arms_write_the_default_bytes(tokens):
    """The launch arms (wait; release + wait) against the plain launch, for every form: the same bytes in ``q_out``, ``gate_out`` and
    the whole cache.  The arms run behind torch copies on the same stream that rewrite the slab right before the launch (such a copy
    never releases its dependents: this cell shows the arms' bytes, the wait's position is pinned by the rendering cell and by
    ``test_wait_holds_every_read_behind_a_predecessor_that_released_early``)."""
    dev = _device_or_skip()
    for form in FORMS:
        ops = make_operands(tokens, form, dev, seed=40 + tokens)
        base = _snap(_entry(dev, form), ops)
        stale = make_operands(tokens, form, dev, seed=41 + tokens)["slab"]
        for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
            k = _entry(dev, form, arm)
            k.compile(tokens)
            work = bind(
                dict(
                    {n: ops[n] for n in ("q_weight", "k_weight", "cos_sin", "positions", "slot_mapping")},
                    slab=stale.clone(),
                    cache=canary_cache(*ops["cache"].shape[0:3:2]).to(dev),
                ),
                form,
            )
            outs = fresh_outputs(tokens, dev)
            work["slab"].copy_(ops["slab"])  # a stream predecessor that writes the operands: complete before this launch starts
            k(*call_args(work), **outs)
            torch.cuda.synchronize()
            _assert_same(dict(outs, cache=work["cache"]), base, f"{form} {arm_id} tokens {tokens}")


def _packed_step(dev, form, tokens, seed):
    """The operands of a call whose row data -- the slab, the positions and the slot vector -- live in ONE byte buffer, so that ONE
    predecessor launch can write all of them: ``(buffer, operands, new bytes)`` with the buffer holding STALE values of the same
    shapes (other valid rows, other valid positions and slots)."""
    new = make_operands(tokens, form, "cpu", seed=seed)
    old = make_operands(tokens, form, "cpu", seed=seed + 1)
    pd, sd = INT[FORMS[form][2]], INT[FORMS[form][3]]
    rows = 3 if FORMS[form][0] == "3d" else 1

    def pack(o):
        parts = [
            o["slab"].contiguous().view(torch.uint8).reshape(-1),
            o["positions"].contiguous().view(torch.uint8).reshape(-1),
            o["slot_mapping"].contiguous().view(torch.uint8).reshape(-1),
        ]
        sizes = [p.numel() for p in parts]
        offsets = [0, (sizes[0] + 15) // 16 * 16]
        offsets.append((offsets[1] + sizes[1] + 15) // 16 * 16)
        buf = torch.zeros((offsets[2] + sizes[2] + 15) // 16 * 16, dtype=torch.uint8)
        for off, p in zip(offsets, parts):
            buf[off : off + p.numel()] = p
        return buf, offsets, sizes

    new_bytes, offsets, sizes = pack(new)
    buf = pack(old)[0].to(dev)
    slab = buf[: sizes[0]].view(BF16).view(tokens, -1)
    positions = buf[offsets[1] : offsets[1] + sizes[1]].view(pd)
    positions = positions.view(3, tokens) if rows == 3 else positions
    slot_mapping = buf[offsets[2] : offsets[2] + sizes[2]].view(sd)
    ops = bind(
        dict(
            slab=slab,
            q_weight=new["q_weight"].to(dev),
            k_weight=new["k_weight"].to(dev),
            cos_sin=new["cos_sin"].to(dev),
            positions=positions,
            cache=new["cache"].to(dev),
            slot_mapping=slot_mapping,
        ),
        form,
    )
    return buf, ops, new_bytes.to(dev), new


def test_wait_holds_every_read_behind_a_predecessor_that_released_early():
    """The POSITION of the dependent-launch wait, on a device.  The stream predecessor is a launch that releases its dependents as
    its FIRST statement and only then writes the step's row data -- the slab, the positions and the slot vector, in ONE buffer
    (``pdl_dependent_copy.EarlyReleaseWriter``: one thread block walks the buffer in many passes, of which only the last leaves the
    written values).  The step's kernel therefore STARTS while its operands hold other bytes, and its wait is the only thing that
    holds its reads.

    The positive control comes first: a dependent WITHOUT a wait copies the buffer's tail behind the same writer and must read other
    bytes than the written ones, eagerly and replayed from a CUDA graph.  Where the early start does not reproduce the cell SKIPS (a
    coverage gap, not a pass); either outcome is one line of the run's "measured" section.  Then the step, in the wait and the
    release + wait arm, eager (warm launches) and captured: ``q_out``, ``gate_out`` and the whole cache are bitwise those of the
    plain step on the written values.  A kernel that read a row, a position or a slot in front of its wait fails here."""
    dev = _device_or_skip()
    import frost_routing
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter

    form, tokens = DEFAULT_FORM, kcfg.MAX_ROWS
    buf, ops, new_bytes, new = _packed_step(dev, form, tokens, seed=67)
    stale = buf.clone()
    writer, unwaited = EarlyReleaseWriter(dev), DependentCopy(dev, wait=False)
    tail = 4096
    seen = torch.zeros(tail, dtype=torch.uint8, device=dev)

    def control():
        writer(new_bytes, buf)
        unwaited(buf[-tail:], seen)  # no wait: it reads whatever the buffer's tail holds when it starts

    def early_starts(run):
        count = 0
        for _ in range(4):
            buf.copy_(stale)
            seen.zero_()
            torch.cuda.synchronize()  # an idle stream: the writer is the dependent's only predecessor in flight
            run()
            torch.cuda.synchronize()
            assert torch.equal(buf, new_bytes), "the writer leaves the written bytes"
            count += int(not torch.equal(seen, new_bytes[-tail:]))
        return count

    control()  # compiles both launches
    torch.cuda.synchronize()
    eager = early_starts(control)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            control()
        captured = early_starts(graph.replay)
    finally:
        graph.reset()
    key = "attention preparation decode step, the wait behind an early-releasing predecessor"
    seen_text = f"the control (a dependent without a wait) read other bytes than the written ones in {eager} of 4 eager launches and {captured} of 4 replays"
    if eager + captured == 0:
        frost_routing.measured(key, f"NOT COMPARED: {seen_text} -- the early start did not reproduce on this device")
        pytest.skip(f"{seen_text}: the early start did not reproduce on this device (a coverage gap, not a pass)")
    want = _snap(_entry(dev, form), bind({n: new[n].to(dev) for n in ("slab", "q_weight", "k_weight", "cos_sin", "positions", "cache", "slot_mapping")}, form))
    compared, differing = 0, []
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        k = _entry(dev, form, arm)
        k.compile(tokens)
        outs = fresh_outputs(tokens, dev)
        got = dict(outs, cache=ops["cache"])

        def chain():
            writer(new_bytes, buf)  # releases at its entry, then writes the slab, the positions and the slots
            k(*call_args(ops), **outs)  # may start at once; every read of row data is behind its wait

        def restore():
            buf.copy_(stale)
            ops["cache"].view(I16).fill_(CANARY_BITS)
            for t in outs.values():
                t.view(I16).fill_(SENTINEL_BITS)
            torch.cuda.synchronize()

        def compare(label):
            nonlocal compared
            torch.cuda.synchronize()
            compared += 1
            try:
                _assert_same(got, want, f"{arm_id}, {label}")
            except AssertionError as err:
                differing.append(str(err))

        restore()
        chain()  # the kernel's first launch cannot start early: warm-up only
        for _ in range(2):
            restore()
            chain()
            compare("eager")
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                chain()
            for _ in range(2):
                restore()
                graph.replay()
                compare("captured")
        finally:
            graph.reset()
    text = f"{compared - len(differing)} of {compared} launches of the step equal the plain step; {seen_text}"
    frost_routing.measured(key, ("COMPARED, equal: " if not differing else "FAILED: ") + text)
    assert not differing, f"behind the early-releasing writer {len(differing)} of {compared} launches differ from the plain step:\n" + "\n".join(differing)


def test_release_overlaps_a_dependent_without_changing_bytes():
    """The release with a dependent on the stream: the step built with ``pdl=True`` (release + wait) followed by launches that carry
    the programmatic-stream-serialization attribute, wait, and copy the step's ``q_out`` and its cache (``pdl_dependent_copy.py``).
    The dependents may start while the step runs; their wait holds their reads.  Eager and captured in a CUDA graph, at 1 and 32
    tokens: the copied bytes are those of the plain step."""
    dev = _device_or_skip()
    from pdl_dependent_copy import DependentCopy

    copy = DependentCopy(dev)
    k = make_entry(dev, DEFAULT_FORM, (True, None))
    assert k.cfg.PDL and k.cfg.PDL_RELEASE and k.cfg.LAUNCH_ARM == "release_wait"
    for tokens in (1, 32):
        ops = make_operands(tokens, DEFAULT_FORM, dev, seed=61 + tokens)
        want = _snap(_entry(dev), ops)
        k.compile(tokens)
        q_copy = torch.zeros(tokens, HEADS, D, dtype=BF16, device=dev)
        cache_copy = torch.zeros_like(ops["cache"])

        def chain():
            q_out, _ = k(*call_args(ops))
            copy(q_out, q_copy)
            copy(ops["cache"], cache_copy)

        def restore():
            ops["cache"].view(I16).fill_(CANARY_BITS)
            for t in (q_copy, cache_copy):
                t.view(I16).fill_(0)

        restore()
        chain()  # eager (also compiles the dependent)
        torch.cuda.synchronize()
        got = dict(q_out=q_copy, cache=cache_copy)
        _assert_same(got, dict(q_out=want["q_out"], cache=want["cache"]), f"eager chain tokens {tokens}")
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                chain()
            for _ in range(3):
                restore()
                graph.replay()
                torch.cuda.synchronize()
                _assert_same(got, dict(q_out=want["q_out"], cache=want["cache"]), f"captured chain tokens {tokens}")
        finally:
            graph.reset()


def test_one_compile_serves_every_row_count(monkeypatch):
    """Compiled once without an operand, the entry launches every token count -- entry-owned and caller-owned outputs, full, short
    and empty slot vectors, pools of other page counts -- with compilation FORBIDDEN (Rule 4: the token count, every address and
    every stride are runtime arguments, never a key)."""
    dev = _device_or_skip()
    k = make_entry(dev)
    k.compile()
    assert len(k.compiled) == 1 and not k.buffers
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("a warm call compiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    for tokens in TOKENS:
        for pages, n_slots in ((None, tokens), (tokens // 64 + 5, tokens // 2), (2 * (tokens // 64 + 1), 0)):
            ops = make_operands(tokens, DEFAULT_FORM, dev, seed=70 + tokens, pages=pages)
            ops["slot_mapping"] = ops["slot_mapping"][:n_slots].clone()
            outs, ops = _run(k, ops)
            assert_twin(ops, outs, k.cfg, label=f"tokens {tokens}, {n_slots} slots after one compile")
        q_out, gate_out = k(*call_args(ops))
        torch.cuda.synchronize()
        _assert_same(dict(q_out=q_out, gate_out=gate_out), outs, f"tokens {tokens}: entry-owned outputs")
    k.compile(3)  # a second compile() is a no-op for the kernel
    assert len(k.compiled) == 1 and set(k.buffers) == set(TOKENS) | {3}


def test_compile_declines_a_dsl_below_the_floor_before_the_kernel_is_built(monkeypatch):
    dev = _device_or_skip()
    from cudnn.frost import buffers

    k = make_entry(dev)
    ops = make_operands(1, DEFAULT_FORM, dev)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.1"):
        k.compile(1)
    with pytest.raises(NotImplementedError, match="found 4.7.1"):
        k(*call_args(ops))
    assert not k.compiled


def test_graph_capture_replays_bitwise_with_eager():
    """Several layers' launches of two forms captured into one CUDA graph after ``compile(rows)``: the replay is bitwise the eager
    run, and after the slab, the positions, the slots, the weights and the table change IN PLACE the replay is bitwise a fresh eager
    run on the new values (every tensor is a launch argument by its address: the graph holds pointers, never values)."""
    dev = _device_or_skip()
    tokens = 6
    forms = (DEFAULT_FORM, "1d-p16-i64-i32", DEFAULT_FORM)
    entries = [_entry(dev, form, (True, None)) for form in forms]
    for k in entries:
        k.compile(tokens)
    layers = [make_operands(tokens, form, dev, seed=100 + i, kind="structured" if i == 1 else "random") for i, form in enumerate(forms)]
    outs = [fresh_outputs(tokens, dev) for _ in forms]

    def eager():
        return [_snap(_entry(dev, form), ops) for form, ops in zip(forms, layers)]

    want = eager()
    graph = torch.cuda.CUDAGraph()

    def replay():
        for ops, o in zip(layers, outs):
            ops["cache"].view(I16).fill_(CANARY_BITS)
            for t in o.values():
                t.view(I16).fill_(SENTINEL_BITS)
        graph.replay()
        torch.cuda.synchronize()
        return [dict(o, cache=ops["cache"]) for ops, o in zip(layers, outs)]

    try:
        with torch.cuda.graph(graph):
            for k, ops, o in zip(entries, layers, outs):
                k(*call_args(ops), **o)
        for _ in range(3):
            for got, ref in zip(replay(), want):
                _assert_same(got, ref, "graph replay bitwise with eager")
        for i, (form, ops) in enumerate(zip(forms, layers)):  # new values in the same storage
            fresh = make_operands(tokens, form, dev, seed=500 + i)
            for name in ("slab", "q_weight", "k_weight", "cos_sin", "slot_mapping"):
                ops[name].copy_(fresh[name])
            ops["positions"].copy_(fresh["positions"])
        got_all = [{n: t.clone() for n, t in got.items()} for got in replay()]
        for got, ref in zip(got_all, eager()):
            _assert_same(got, ref, "the replay reads the operands' current values")
    finally:
        graph.reset()


def test_warm_call_allocates_nothing_and_never_synchronizes():
    """Rules 1 and 3 on the warm path: with caller-owned outputs and with entry-owned outputs of a token count that was already
    called, three calls allocate no device memory, and a call under ``set_sync_debug_mode('error')`` raises nothing (no device read
    reaches the host: positions, slots and every other value stay on the device)."""
    dev = _device_or_skip()
    tokens = 8
    k = make_entry(dev, DEFAULT_FORM, (True, None))
    ops = make_operands(tokens, DEFAULT_FORM, dev, seed=81)
    want = _snap(_entry(dev), ops)
    mine = fresh_outputs(tokens, dev)
    k(*call_args(ops), **mine)
    k(*call_args(ops))
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(*call_args(ops), **mine)
            k(*call_args(ops))
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    _assert_same(dict(mine, cache=ops["cache"]), want, "the warm calls' bytes")


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    dev = _device_or_skip()
    k = make_entry(dev)
    ops = make_operands(2, DEFAULT_FORM, dev)
    want = _snap(k, ops)
    mine = fresh_outputs(2, dev)
    with pytest.raises(TypeError, match="q must be torch.bfloat16"):
        k(*call_args(dict(ops, q=ops["q"].float())), **mine)
    with pytest.raises(ValueError, match="every operand must live on one device, the entry's"):
        k(*call_args(dict(ops, q_weight=ops["q_weight"].cpu())), **mine)
    with pytest.raises(TypeError, match=r"slot_mapping must be torch.int64 \(the entry's slot_dtype"):
        k(*call_args(dict(ops, slot_mapping=ops["slot_mapping"].to(torch.int32))), **mine)
    with pytest.raises(ValueError, match=r"k_pool must be \[num_pages >= 1, H=4, page_size=64, D=256\]"):
        k(*call_args(dict(ops, k_pool=canary_cache(8, 16).to(dev)[..., :D])), **mine)
    with pytest.raises(ValueError, match="k_pool must not overlap v_pool"):
        k(*call_args(dict(ops, v_pool=ops["k_pool"])), **mine)
    torch.cuda.synchronize()
    assert bool((bits(mine["q_out"]) == SENTINEL_BITS).all()) and bool((bits(mine["gate_out"]) == SENTINEL_BITS).all()), "a declined warm call launches nothing"
    assert bool((bits(ops["cache"]) == CANARY_BITS).all()), "a declined warm call writes no pool row"
    assert want["q_out"].shape == (2, HEADS, D)


def test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current():
    """Rule 5 (the device is implicit state exactly like the stream): an entry built for a device that is NOT the caller's current one
    compiles and launches there, returns that device's bytes, and leaves the caller's current device as it was."""
    _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    other = torch.device("cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count())
    cc = torch.cuda.get_device_capability(other)
    if not entry.cc_supported(cc) or entry._dsl_error(cc) is not None:
        pytest.skip(f"the second device (cc {cc[0]}.{cc[1]}) is not served")
    current = torch.cuda.current_device()
    k = make_entry(other)
    ops = make_operands(3, DEFAULT_FORM, other, seed=91)
    outs = fresh_outputs(3, other)
    q_out, gate_out = k(*call_args(ops), **outs)
    torch.cuda.synchronize(other)
    assert torch.cuda.current_device() == current and q_out.device == other and gate_out.device == other
    assert_twin(ops, outs, k.cfg, label="an entry on a non-current device")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
@pytest.mark.parametrize("form, tokens", ((DEFAULT_FORM, 1), ("1d-p64-i64-i64", 3), ("3d-p16-i32-i32", 17), ("1d-p16-i64-i32", 160)))
def test_launch_facts_equal_the_profilers_record_of_a_launch(form, tokens, arm, arm_id):
    """One call is ONE kernel launch whose profiler record equals the entry's launch facts: the kernel's name is the entry's symbol of
    the compiled form -- which carries the form's text directly behind the kernel's name -- and grid, block and shared memory are the
    launch shape.  Before the first compile the entry names no symbol."""
    dev = _device_or_skip()
    k = make_entry(dev, form, arm)
    assert k.kernel_symbols_for(tokens) == (), "nothing is compiled yet"
    ops = make_operands(tokens, form, dev, seed=95)
    outs = fresh_outputs(tokens, dev)
    record = assert_facts_equal_the_launch(k, lambda: k(*call_args(ops), **outs), tokens)
    head = SYMBOL + str(k.cfg)
    assert record.name.startswith(head + "_") and len(head) <= 100, record.name
    assert k.kernel_symbols_for(tokens) == (record.name,) == k.kernel_symbols_for(kcfg.MAX_ROWS), "one compiled kernel answers every token count"
    assert k.form_for(tokens) == asdict(k.cfg)
