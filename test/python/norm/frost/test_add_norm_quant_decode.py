# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The add + RMSNorm + quantize decode step (``cudnn.norm.frost.add_norm_quant_decode``): on the host the config's limits and tables,
the fifteen forms and three launch arms, every typed decline of the entry (CPU tensors), the DSL gates (version, target, compiler), the
plan-time-only compile key, the launch facts, the lazy export, and the kernel's RENDERING for the claimed targets (the instructions
that define the sum tree and the rounding chain; the form inside the first 100 characters of the symbol); on a cc 10.x device the
bytes against the independent references of ``add_norm_quant_utils`` (the fp64 bracket of the normalized value, the residual, the
E4M3 codes, the NVFP4 block scales and code bracket), the exact cells, the edge rows, output ownership per output, the bit-neutral
arms, the release beside a dependent launch, CUDA-graph replay, the warm path's allocation / synchronization discipline and the
profiler's record of a launch.

No cell arms a timer of its own: every device cell synchronizes under the suite's per-test watchdog (``CUDNN_TEST_TIMEOUT`` in
``test/python/conftest.py``), which a second in-process timer would replace and then disarm.  A trip of that watchdog on a launch is
a hang of the kernel's one thread-block barrier -- audit the barrier table in the kernel's docstring; never a larger limit."""

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

from add_norm_quant_utils import (  # noqa: E402  (the form tables, the operand builders, the references and the device gate)
    BF16,
    DETERMINISM_FORMS,
    E2M1_MIDPOINTS,
    E4M3,
    E4M3_SCALES,
    EPS,
    FORMS,
    HIDDEN,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    NVFP4_SCALES,
    U8,
    _device_or_skip,
    assert_nvfp4_codes,
    assert_y,
    e2m1_codes,
    e4m3_codes,
    edge_operands,
    exact_operands,
    form_kwargs,
    make_entry,
    make_operands,
    nvfp4_scale_bytes,
    pack_e2m1,
    residual_reference,
    scale_bytes_layout_mask,
    scale_bytes_of_rows,
    scale_tensor,
    unpack_e2m1,
)

from cudnn.norm.frost import add_norm_quant_decode as entry  # noqa: E402
from cudnn.norm.frost.kernel import add_norm_quant_decode_config as kcfg  # noqa: E402
from launch_records import assert_facts_equal_the_launch  # noqa: E402

pytestmark = [pytest.mark.L0]

REPO = Path(__file__).resolve().parents[4]
I16 = torch.int16
ROWS = (1, 2, 3, 4, 8, 16, 17, 24, 32, 64, 128, 129, 160, 1024)
SYMBOL = "cudnn_kernel_frost_add_rms_norm_quant_"


def _cfg(form="add-bf16-e4m3", arm=(None, None), hidden=HIDDEN, eps=EPS):
    return entry.make_cfg(hidden, eps, pdl=arm[0], pdl_release=arm[1], **form_kwargs(form))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_config_constants_and_tables():
    """The limits with their derivation, the barrier ledger and the shared-memory table as data: one thread block per row of
    ``hidden / 8`` threads in full warps; ONE thread-block barrier that every thread issues once (``issuing_lanes == participants``);
    ONE shared-memory buffer of an fp32 slot per warp, written once and never refilled; the NVFP4 scale geometry; the row ceiling
    inside the signed 32-bit element index and above one scheduling wave of every served part."""
    assert (kcfg.VEC, kcfg.SERVED_HIDDEN, kcfg.MAX_ROWS, kcfg.SM_THREADS, kcfg.ROW_ADDRESS_LIMIT) == (8, (5120,), 1024, 2048, 2**31 - 1)
    assert kcfg.QUANT_FORMS == ("e4m3_static", "nvfp4"), "the quant vocabulary is a literal: an appended form is a visible diff"
    cfg = _cfg()
    assert (cfg.THREADS, cfg.WARPS, cfg.SMEM_BYTES) == (640, 20, 80) and cfg.THREADS == cfg.HIDDEN // kcfg.VEC == 32 * cfg.WARPS
    assert kcfg.barrier_ledger(cfg) == dict(participants=640, issuing_lanes=640)
    table = kcfg.smem_table(cfg)
    assert len(table) == 1 and table[0]["bytes"] == 4 * cfg.WARPS == 80 and table[0]["elements"] == 20 and table[0]["lane_stride_bytes"] == 4
    assert table[0]["refilled"] is False, "the one buffer is written once and read once: nothing releases it"
    assert (cfg.SF_BLOCKS, cfg.SF_K_TILES) == (320, 80)
    assert [kcfg.scale_rows(r) for r in (1, 128, 129, 160, 1024)] == [128, 128, 256, 256, 1024]
    assert kcfg.MAX_ROWS * max(kcfg.SERVED_HIDDEN) <= kcfg.ROW_ADDRESS_LIMIT
    assert cfg.CTAS_PER_SM == 3, "three 640-thread blocks per SM: one scheduling wave holds 3 x SMs rows"
    assert kcfg.MAX_ROWS > cfg.CTAS_PER_SM * 212, "the row ceiling crosses a scheduling wave on a 212-SM part"
    for hidden in (256, 2048, 8192):
        wide = replace(cfg, HIDDEN=hidden)
        assert kcfg.validate_cfg(wide).THREADS == hidden // 8 and kcfg.barrier_ledger(wide)["participants"] == 32 * wide.WARPS == wide.THREADS
    # the scale byte position: tiles of 128 rows one after the other, the 128 x 4 atom rule inside a tile
    for row, block, want in (
        (0, 0, 0),
        (0, 5, 513),
        (1, 0, 16),
        (32, 0, 4),
        (127, 3, 31 * 16 + 3 * 4 + 3),
        (128, 0, 80 * 512),
        (129, 7, 80 * 512 + 512 + 16 + 3),
    ):
        assert kcfg.scale_byte_offset(cfg, row, block) == want, (row, block)
    offsets = {kcfg.scale_byte_offset(cfg, r, b) for r in range(256) for b in range(320)}
    assert len(offsets) == 256 * 320 and max(offsets) == 256 * 320 - 1, "the layout is a bijection of (row, block) onto the scale buffer's bytes"


def test_form_text_is_compact_unique_and_inside_the_first_100_characters_of_the_symbol():
    """``str(cfg)`` -- the text the DSL appends to the kernel's name -- states the form (width, residual mode, outputs, launch arm,
    the fp32 bits of eps), differs between any two forms and arms, and ends inside the first 100 characters of the symbol; ``repr``
    stays the field list."""
    texts = {}
    for form in FORMS:
        for arm, arm_id in zip(LAUNCH_ARMS, LAUNCH_ARM_IDS):
            cfg = _cfg(form, arm)
            text = str(cfg)
            texts[(form, arm_id)] = text
            assert len(SYMBOL + text) <= 100, text
            assert re.fullmatch(r"[a-z0-9_]+", text), f"{text!r}: characters the symbol mangling would drop"
            assert "0x" not in text, "a hex run behind 0x would be dropped from the symbol as an address"
            assert repr(cfg).startswith("Cfg(HIDDEN=5120, EPS=1e-06, ")
    assert len(set(texts.values())) == len(texts) == 45
    assert texts[("add-bf16-e4m3", "wait-release")] == "h5120_add_bf16_e4m3_static_wait_release_eps358637bd"
    assert texts[("store-e4m3", "plain")] == "h5120_store_e4m3_static_plain_eps358637bd"
    assert texts[("none-bf16-nvfp4", "wait")] == "h5120_none_bf16_nvfp4_wait_eps358637bd"
    assert str(_cfg(eps=1e-5)) != str(_cfg()) and str(replace(_cfg(), HIDDEN=2048)).startswith("h2048_")
    assert kcfg.eps_bits(1e-6) == 0x358637BD


def test_cfg_knobs_validate_by_name():
    """The construction declines, each by name and type, and the fifteen forms x three launch arms accepted."""
    for form in FORMS:
        for arm in LAUNCH_ARMS:
            cfg = _cfg(form, arm)
            add, store, bf16, quant = FORMS[form]
            want = (add, store, bf16, quant, bool(arm[0]), bool(arm[0]) if arm[1] is None else arm[1])
            assert (cfg.ADD, cfg.STORE_RESIDUAL, cfg.BF16_OUT, cfg.QUANT, cfg.PDL, cfg.PDL_RELEASE) == want
            assert (
                kcfg.validate_cfg(cfg) is cfg and entry.AddNormQuantDecode("cpu", HIDDEN, EPS, pdl=arm[0], pdl_release=arm[1], **form_kwargs(form)).cfg == cfg
            )
    ok = dict(add=True, bf16=True, quant="e4m3_static")
    for hidden in (5120.0, "5120", None, True):
        with pytest.raises(TypeError, match="hidden must be an int"):
            entry.make_cfg(hidden, EPS, **ok)
    for hidden in (0, -256, 100, 5121, 8192 + 256):
        with pytest.raises(ValueError, match="hidden must be a multiple of 256 in 256 .. 8192"):
            entry.make_cfg(hidden, EPS, **ok)
    for hidden in (256, 2048, 4096, 8192):
        with pytest.raises(NotImplementedError, match=f"serves hidden 5120 .validated.; got hidden {hidden}"):
            entry.make_cfg(hidden, EPS, **ok)
    for eps in (0.0, -1e-6, float("nan"), float("inf"), "1e-6", None, True, 1e-50, 1e39):
        with pytest.raises(ValueError, match="eps must be a finite positive float"):
            entry.make_cfg(HIDDEN, eps, **ok)
    with pytest.raises(ValueError, match="store_residual initializes the residual from x and cannot be combined with add"):
        entry.make_cfg(HIDDEN, EPS, add=True, bf16=True, quant=None, store_residual=True)
    for quant in ("fp8", "e4m3", "e4m3_row", "NVFP4", 8, ("nvfp4",)):
        with pytest.raises(ValueError, match="quant must be None or one of"):
            entry.make_cfg(HIDDEN, EPS, add=True, bf16=True, quant=quant)
    with pytest.raises(ValueError, match="the form writes no output"):
        entry.make_cfg(HIDDEN, EPS, add=True, bf16=False, quant=None)
    for pdl in (None, False):
        with pytest.raises(ValueError, match="pdl_release releases dependents after this launch's own wait and needs pdl=True"):
            entry.make_cfg(HIDDEN, EPS, pdl=pdl, pdl_release=True, **ok)
    for name in ("add", "bf16", "store_residual"):
        for bad in (1, 0, None, "True"):
            with pytest.raises(TypeError, match=f"{name} must be a bool"):
                entry.make_cfg(HIDDEN, EPS, **{**dict(ok, store_residual=False), name: bad})
    for name in ("pdl", "pdl_release"):
        for bad in (1, 0, "True"):
            with pytest.raises(TypeError, match=f"{name} must be a bool or None"):
                entry.make_cfg(HIDDEN, EPS, **{**ok, name: bad})
    with pytest.raises(TypeError, match="cfg must be an add_norm_quant_decode_config.Cfg"):
        kcfg.validate_cfg(dict(HIDDEN=5120))
    # a config built by hand passes the same gates as the constructor's (the kernel is traced from a validated config only)
    for field, name in (("ADD", "add"), ("STORE_RESIDUAL", "store_residual"), ("BF16_OUT", "bf16"), ("PDL", "pdl"), ("PDL_RELEASE", "pdl_release")):
        with pytest.raises(TypeError, match=f"{name} must be a bool, got int"):
            kcfg.validate_cfg(replace(_cfg(), **{field: 1}))
    with pytest.raises(ValueError, match="pdl_release releases dependents"):
        kcfg.validate_cfg(replace(_cfg(), PDL_RELEASE=True))
    # add, bf16 and quant are REQUIRED keywords: the forms are peers, none is a default
    for missing in ("add", "bf16", "quant"):
        with pytest.raises(TypeError, match=missing):
            entry.AddNormQuantDecode("cpu", HIDDEN, EPS, **{k: v for k, v in ok.items() if k != missing})
    with pytest.raises(TypeError):
        entry.AddNormQuantDecode("cpu", HIDDEN, EPS, True, True, None)  # everything behind eps is keyword-only


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the binding checks (CPU tensors)
# ---------------------------------------------------------------------------------------------------------------------------
def _host_operands(form, rows=3, hidden=HIDDEN, own=True, extra_rows=0):
    """The served operand set of a form on the host: ``dict(x, residual, weight, scale[, out, codes, scales])`` -- ``None`` exactly
    where the form has no such operand; with ``own`` the caller-owned outputs too."""
    add, store, bf16, quant = FORMS[form]
    ops = dict(
        x=torch.zeros(rows, hidden, dtype=BF16),
        residual=torch.zeros(rows + extra_rows, hidden, dtype=BF16) if (add or store) else None,
        weight=torch.zeros(hidden, dtype=BF16),
        scale=torch.ones(1, dtype=torch.float32) if quant else None,
    )
    if own:
        ops["out"] = torch.zeros(rows + extra_rows, hidden, dtype=BF16) if bf16 else None
        ops["codes"] = (
            None
            if quant is None
            else (torch.zeros(rows + extra_rows, hidden, dtype=E4M3) if quant == "e4m3_static" else torch.zeros(rows + extra_rows, hidden // 2, dtype=U8))
        )
        ops["scales"] = torch.zeros(kcfg.scale_rows(rows) + (128 if extra_rows else 0), hidden // 16, dtype=E4M3) if quant == "nvfp4" else None
    return ops


def _check(form, ops, **kw):
    return entry.check_support(_cfg(form), ops["x"], ops["residual"], ops["weight"], ops["scale"], ops.get("out"), ops.get("codes"), ops.get("scales"), **kw)


@pytest.mark.parametrize("form", sorted(FORMS))
def test_served_operands_pass_the_host_checks(form):
    """Every form at 1 row and at the row ceiling, entry-owned-shaped (no output given) and caller-owned operands (exact and taller),
    a 0-d and a ``[1]`` scale, a ``[hidden]`` and a ``[1, hidden]`` weight: the host checks return the row count."""
    for rows in (1, kcfg.MAX_ROWS):
        for own, extra in ((False, 0), (True, 0), (True, 5)):
            ops = _host_operands(form, rows, own=own, extra_rows=extra)
            for cc in ((10, 0), (10, 3), (10, 7)):
                assert _check(form, ops, device_cc=cc) == rows
            assert (
                entry.check_binding(_cfg(form), ops["x"], ops["residual"], ops["weight"], ops["scale"], ops.get("out"), ops.get("codes"), ops.get("scales"))
                == rows
            )
    ops = _host_operands(form, 2)
    ops["weight"] = ops["weight"].reshape(1, HIDDEN)
    if ops["scale"] is not None:
        ops["scale"] = torch.tensor(1.0)
    assert _check(form, ops, device_cc=(10, 0)) == 2


def _misaligned(dtype, shape):
    """A contiguous CPU tensor of ``shape`` whose first byte sits one ELEMENT (one byte for uint8-backed E4M3, two for bf16) past a
    16-byte aligned address."""
    n = 1
    for d in shape:
        n *= d
    if dtype == E4M3:
        return torch.zeros(n + 16, dtype=U8)[1 : 1 + n].view(E4M3).view(*shape)
    return torch.zeros(n + 16, dtype=dtype)[1 : 1 + n].view(*shape)


def _misaligned_scale():
    """One fp32 element at an address that is not a multiple of 4 (a byte buffer read at an odd offset)."""
    buf = bytearray(64)
    base = torch.frombuffer(buf, dtype=U8).data_ptr()
    return torch.frombuffer(buf, dtype=torch.float32, count=1, offset=1 + (-base) % 4)  # address % 4 == 1


def _mutations():
    """``name -> (form, mutate(ops), exception, message token)``: one served operand set with ONE thing wrong."""

    def put(key, make):
        def mutate(ops):
            ops[key] = make(ops)

        return mutate

    rows = 3
    cases = {}
    for key in ("x", "residual", "weight", "scale", "out", "codes"):
        cases[f"{key}-not-a-tensor"] = ("add-bf16-e4m3", put(key, lambda ops: [1.0]), TypeError, f"{key} must be a torch.Tensor, got list")
    cases["scales-not-a-tensor"] = ("add-nvfp4", put("scales", lambda ops: 3), TypeError, "scales must be a torch.Tensor, got int")
    cases["x-missing"] = ("add-bf16", put("x", lambda ops: None), TypeError, "x must be a torch.Tensor, got NoneType")
    cases["weight-missing"] = ("add-bf16", put("weight", lambda ops: None), TypeError, "weight must be a torch.Tensor, got NoneType")
    for key in ("x", "residual", "weight", "out"):
        cases[f"{key}-dtype"] = (
            "add-bf16-e4m3",
            put(key, lambda ops, key=key: ops[key].float()),
            TypeError,
            f"{key} must be torch.bfloat16 .kernel-native; callers convert., got torch.float32",
        )
    cases["scale-dtype"] = ("add-bf16-e4m3", put("scale", lambda ops: ops["scale"].double()), TypeError, "scale must be torch.float32")
    cases["codes-dtype-e4m3"] = ("add-e4m3", put("codes", lambda ops: ops["codes"].view(U8)), TypeError, "codes must be torch.float8_e4m3fn")
    cases["codes-dtype-nvfp4"] = ("add-nvfp4", put("codes", lambda ops: ops["codes"].view(E4M3)), TypeError, "codes must be torch.uint8")
    cases["scales-dtype"] = ("add-nvfp4", put("scales", lambda ops: ops["scales"].view(U8)), TypeError, "scales must be torch.float8_e4m3fn")
    cases["x-3d"] = (
        "add-bf16",
        put("x", lambda ops: ops["x"].reshape(1, rows, HIDDEN)),
        ValueError,
        r"x must be \[rows, 5120\] with at least one row; got \(1, 3, 5120\)",
    )
    cases["x-width"] = ("none-bf16", put("x", lambda ops: torch.zeros(rows, 2048, dtype=BF16)), ValueError, r"x must be \[rows, 5120\]")
    cases["x-zero-rows"] = ("none-bf16", put("x", lambda ops: torch.zeros(0, HIDDEN, dtype=BF16)), ValueError, "with at least one row")
    cases["rows-above-the-ceiling"] = (
        "none-bf16",
        lambda ops: ops.update(x=torch.zeros(kcfg.MAX_ROWS + 1, HIDDEN, dtype=BF16), out=torch.zeros(kcfg.MAX_ROWS + 1, HIDDEN, dtype=BF16)),
        NotImplementedError,
        "serves at most 1024 rows per call; got 1025",
    )
    cases["residual-missing-add"] = ("add-bf16", put("residual", lambda ops: None), ValueError, "residual is required by this form")
    cases["residual-missing-store"] = ("store-e4m3", put("residual", lambda ops: None), ValueError, "residual is required by this form")
    cases["residual-given-none-mode"] = (
        "none-bf16",
        put("residual", lambda ops: torch.zeros(rows, HIDDEN, dtype=BF16)),
        ValueError,
        "residual was given but the entry was built without add or store_residual; pass None",
    )
    cases["scale-missing"] = ("add-e4m3", put("scale", lambda ops: None), ValueError, "scale is required by quant='e4m3_static'")
    cases["scale-missing-nvfp4"] = ("add-nvfp4", put("scale", lambda ops: None), ValueError, "scale is required by quant='nvfp4'")
    cases["scale-given-bf16-only"] = (
        "add-bf16",
        put("scale", lambda ops: torch.ones(1)),
        ValueError,
        "scale was given but the entry writes no codes; pass None",
    )
    cases["out-given-no-bf16"] = (
        "add-e4m3",
        put("out", lambda ops: torch.zeros(rows, HIDDEN, dtype=BF16)),
        ValueError,
        "out was given but the form does not write it",
    )
    cases["codes-given-bf16-only"] = (
        "add-bf16",
        put("codes", lambda ops: torch.zeros(rows, HIDDEN, dtype=E4M3)),
        ValueError,
        "codes was given but the form does not write it",
    )
    cases["scales-given-e4m3"] = (
        "add-e4m3",
        put("scales", lambda ops: torch.zeros(128, 320, dtype=E4M3)),
        ValueError,
        "scales was given but the form does not write it",
    )
    cases["residual-short"] = (
        "add-bf16",
        put("residual", lambda ops: torch.zeros(rows - 1, HIDDEN, dtype=BF16)),
        ValueError,
        r"residual must be \[>= 3, 5120\]; got \(2, 5120\)",
    )
    cases["weight-shape"] = (
        "add-bf16",
        put("weight", lambda ops: torch.zeros(2, HIDDEN, dtype=BF16)),
        ValueError,
        r"weight must be \[5120\] or \[1, 5120\]; got \(2, 5120\)",
    )
    cases["weight-short"] = ("add-bf16", put("weight", lambda ops: torch.zeros(HIDDEN - 8, dtype=BF16)), ValueError, r"weight must be \[5120\] or \[1, 5120\]")
    cases["scale-two-elements"] = (
        "add-e4m3",
        put("scale", lambda ops: torch.ones(2)),
        ValueError,
        r"scale must be one fp32 element .0-d or \[1\].; got \(2,\)",
    )
    cases["out-short"] = ("add-bf16", put("out", lambda ops: torch.zeros(rows - 1, HIDDEN, dtype=BF16)), ValueError, r"out must be \[>= 3, 5120\]")
    cases["codes-width-nvfp4"] = (
        "add-nvfp4",
        put("codes", lambda ops: torch.zeros(rows, HIDDEN, dtype=U8)),
        ValueError,
        r"codes must be \[>= 3, 2560\]; got \(3, 5120\)",
    )
    cases["codes-width-e4m3"] = ("add-e4m3", put("codes", lambda ops: torch.zeros(rows, HIDDEN // 2, dtype=E4M3)), ValueError, r"codes must be \[>= 3, 5120\]")
    cases["scales-short"] = (
        "add-nvfp4",
        put("scales", lambda ops: torch.zeros(rows, 320, dtype=E4M3)),
        ValueError,
        r"scales must be \[>= 128, 320\]; got \(3, 320\)",
    )
    cases["scales-width"] = ("add-nvfp4", put("scales", lambda ops: torch.zeros(128, 640, dtype=E4M3)), ValueError, r"scales must be \[>= 128, 320\]")
    cases["x-not-contiguous"] = (
        "none-bf16",
        put("x", lambda ops: torch.zeros(rows, 2 * HIDDEN, dtype=BF16)[:, ::2]),
        NotImplementedError,
        r"needs x contiguous .*got strides \(10240, 2\)",
    )
    cases["out-not-contiguous"] = (
        "none-bf16",
        put("out", lambda ops: torch.zeros(2 * rows, HIDDEN, dtype=BF16)[::2]),
        NotImplementedError,
        "needs out contiguous",
    )
    cases["residual-not-contiguous"] = (
        "add-bf16",
        put("residual", lambda ops: torch.zeros(HIDDEN, rows, dtype=BF16).t()),
        NotImplementedError,
        "needs residual contiguous",
    )
    for key in ("x", "residual", "weight", "out"):
        shape = (HIDDEN,) if key == "weight" else (rows, HIDDEN)
        cases[f"{key}-misaligned"] = (
            "add-bf16",
            put(key, lambda ops, shape=shape: _misaligned(BF16, shape)),
            ValueError,
            f"{key} must start at a 16-byte aligned address for its 128-bit loads / stores, got data_ptr % 16 = 2",
        )
    cases["codes-misaligned"] = (
        "add-e4m3",
        put("codes", lambda ops: _misaligned(E4M3, (rows, HIDDEN))),
        ValueError,
        "codes must start at a 16-byte aligned address",
    )
    cases["scales-misaligned"] = (
        "add-nvfp4",
        put("scales", lambda ops: _misaligned(E4M3, (128, 320))),
        ValueError,
        "scales must start at a 16-byte aligned address",
    )
    cases["scale-misaligned"] = (
        "add-e4m3",
        put("scale", lambda ops: _misaligned_scale()),
        ValueError,
        "scale must start at a 4-byte aligned address for its 32-bit load, got data_ptr % 4 = 1",
    )
    cases["residual-is-x"] = ("add-bf16", put("residual", lambda ops: ops["x"]), ValueError, "residual must not overlap x")
    cases["out-is-x"] = ("none-bf16", put("out", lambda ops: ops["x"]), ValueError, "out must not overlap x")
    cases["out-is-residual"] = ("add-bf16", put("out", lambda ops: ops["residual"]), ValueError, "residual must not overlap out")
    cases["out-holds-weight"] = ("none-bf16", put("weight", lambda ops: ops["out"][1]), ValueError, "out must not overlap weight")
    cases["codes-hold-scale"] = (
        "none-e4m3",
        put("scale", lambda ops: ops["codes"].view(U8).reshape(-1)[16:20].view(torch.float32)),
        ValueError,
        "codes must not overlap scale",
    )
    cases["codes-not-contiguous"] = ("none-nvfp4", put("codes", lambda ops: ops["x"].view(U8)[:, : HIDDEN // 2]), NotImplementedError, "needs codes contiguous")
    cases["codes-alias-x-bytes"] = (
        "none-e4m3",
        put("codes", lambda ops: ops["x"].view(U8).reshape(-1)[: rows * HIDDEN].view(E4M3).view(rows, HIDDEN)),
        ValueError,
        "codes must not overlap x",
    )
    cases["control-a-fresh-disjoint-out-stays-served"] = (
        "none-bf16-nvfp4",
        put("out", lambda ops: torch.zeros(128 * 320 // 2 + rows * HIDDEN, dtype=BF16)[: rows * HIDDEN].view(rows, HIDDEN)),
        None,
        None,
    )
    return cases


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_typed_declines_name_the_operand(name):
    """Every binding decline, by exception type and by the operand's name: a missing or an unserved operand (both directions), a
    non-tensor, a dtype, a shape, a row count above the ceiling, a non-contiguous operand, a misaligned address (16 bytes; ``scale``
    4), and a written operand that overlaps another operand (``x``, ``weight`` and ``scale`` included)."""
    form, mutate, exc, token = _mutations()[name]
    ops = _host_operands(form)
    assert _check(form, ops, device_cc=(10, 0)) == 3, "the unmutated set is served"
    mutate(ops)
    if exc is None:  # the control: a fresh, disjoint operand of the right shape stays served
        assert _check(form, ops, device_cc=(10, 0)) == 3
        return
    with pytest.raises(exc, match=token):
        _check(form, ops, device_cc=(10, 0))


def test_binding_check_requires_every_operand_on_the_given_device():
    ops = _host_operands("add-bf16-e4m3")
    cfg = _cfg("add-bf16-e4m3")
    with pytest.raises(ValueError, match=r"every operand must live on one device, the entry's \(cuda:1\); x is on cpu"):
        entry.check_binding(cfg, ops["x"], ops["residual"], ops["weight"], ops["scale"], device="cuda:1")
    assert entry.check_binding(cfg, ops["x"], ops["residual"], ops["weight"], ops["scale"], device="cpu") == 3


def test_compile_and_outputs_decline_a_bad_row_count():
    """``compile(rows)`` and ``outputs(rows)``: a row count that is not an ``int`` (``TypeError``), below one (``ValueError``), above
    the ceiling (``NotImplementedError``) -- before any device work; ``compile()`` on an entry whose device is not CUDA is the typed
    decline of the device."""
    k = make_entry("cpu", "add-bf16-nvfp4")
    for method in (k.compile, k.outputs):
        for rows in (1.0, "1", True):
            with pytest.raises(TypeError, match="rows must be an int"):
                method(rows)
        for rows in (0, -3):
            with pytest.raises(ValueError, match="rows must be at least 1"):
                method(rows)
        with pytest.raises(NotImplementedError, match="serves at most 1024 rows per call; got 1025"):
            method(kcfg.MAX_ROWS + 1)
    with pytest.raises(TypeError, match="rows must be an int"):
        k.outputs(None)
    for rows in (None, 3):
        with pytest.raises(NotImplementedError, match="serves CUDA devices only; the entry's device is cpu"):
            k.compile(rows)
    assert not k.compiled and not k.buffers, "a declined compile allocates nothing"
    ops = _host_operands("add-bf16-nvfp4", own=False)
    with pytest.raises(NotImplementedError, match="serves CUDA devices only"):
        k(ops["x"], ops["residual"], ops["weight"], ops["scale"])


def test_declines_an_unserved_compute_capability_by_name():
    ops = _host_operands("add-bf16-e4m3")
    for cc in ((9, 0), (8, 0), (12, 0), (12, 1), (11, 0)):
        with pytest.raises(NotImplementedError, match=rf"serves compute capability 10.x \(SM100-class\); this device is {cc[0]}.{cc[1]}"):
            _check("add-bf16-e4m3", ops, device_cc=cc)
        assert not entry.cc_supported(cc)
    assert all(entry.cc_supported(cc) for cc in ((10, 0), (10, 3), (10, 7))) and not entry.cc_supported(None)


def test_declines_cpu_operands_when_no_capability_is_given():
    ops = _host_operands("add-bf16-e4m3")
    with pytest.raises(NotImplementedError, match="serves CUDA tensors only; the operands are on cpu"):
        _check("add-bf16-e4m3", ops)


def test_compile_key_is_plan_time_only():
    """One key per entry whatever the row count; it changes with each field of the config."""
    k = make_entry("cpu", "add-bf16-e4m3", (True, None))
    assert k.key() == (k.cfg,) and all(k.cfg_for(rows) is k.cfg for rows in (1, 17, kcfg.MAX_ROWS))
    keys = {k.key()}
    for change in (
        dict(EPS=1e-5),
        dict(ADD=False),
        dict(ADD=False, STORE_RESIDUAL=True),
        dict(BF16_OUT=False),
        dict(QUANT="nvfp4"),
        dict(PDL_RELEASE=False),
        dict(PDL=False, PDL_RELEASE=False),
        dict(HIDDEN=2048),
    ):
        keys.add((replace(k.cfg, **change),))
    assert len(keys) == 9
    with pytest.raises(NotImplementedError, match="at most 1024 rows"):
        k.cfg_for(kcfg.MAX_ROWS + 1)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the launch facts (cudnn.frost.launch_facts) -- the symbol of the compiled form, the form as plain data, the launch shape
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_facts_equal_the_configuration_arithmetic_at_every_row_count(arm, arm_id):
    """EVERY served row count, 1 .. 1024, per form and launch arm: the launch shape is one thread block per row of ``hidden / 8``
    threads with the kernel's one shared-memory buffer (an fp32 slot per warp), and the form is the config's fields -- nothing else,
    because the compile key is the config alone."""
    for form in FORMS:
        k = make_entry("cpu", form, arm)
        want_form = asdict(k.cfg)
        for rows in range(1, kcfg.MAX_ROWS + 1):
            shape = k.launch_shape_for(rows)
            assert shape == ((rows, 1, 1), (640, 1, 1), 80), (form, rows, shape)
            assert all(type(v) is int for v in (*shape.grid, *shape.block, shape.shared_memory)), shape
            assert k.form_for(rows) == want_form, (form, rows)
        add, store, bf16, quant = FORMS[form]
        assert want_form == dict(
            HIDDEN=5120, EPS=1e-6, ADD=add, STORE_RESIDUAL=store, BF16_OUT=bf16, QUANT=quant, PDL=bool(arm[0]), PDL_RELEASE=arm_id == "wait-release"
        )


def test_launch_facts_are_plain_fresh_and_decline_an_unserved_row_count():
    """The facts are plain data, a fresh object per call (a caller's edit reaches nothing of the entry) and the entry's typed declines
    for a row count outside 1 .. 1024; a row count that is not an ``int`` is a ``ValueError`` that says so, as on every entry's
    launch facts."""
    k = make_entry("cpu", "add-nvfp4", (True, None))
    form = k.form_for(7)
    assert json.loads(json.dumps(form)) == form and type(form) is dict
    form["QUANT"] = "edited"
    assert k.form_for(7)["QUANT"] == "nvfp4" and k.cfg.QUANT == "nvfp4"
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
    """Host-only, with a stand-in in the entry's compile cache: nothing compiled -> an empty tuple; once the form is compiled its ONE
    symbol answers every row count; a compiled kernel of another form is not named; a compiled object that does not expose its kernels
    is the typed decline, never a guess."""
    k = make_entry("cpu", "add-bf16-e4m3")
    assert k.kernel_symbols_for(1) == () == k.kernel_symbols_for(kcfg.MAX_ROWS)
    k.compiled[(replace(k.cfg, QUANT="nvfp4"),)] = _CompiledStandIn(SYMBOL + "other_0")
    assert k.kernel_symbols_for(1) == (), "a form the entry does not run is not named"
    k.compiled[k.key()] = _CompiledStandIn(SYMBOL + "A_0")
    assert all(k.kernel_symbols_for(rows) == (SYMBOL + "A_0",) for rows in (1, 2, 160, kcfg.MAX_ROWS))
    k.compiled[k.key()] = object()
    with pytest.raises(NotImplementedError, match=r"the add \+ RMSNorm \+ quantize decode step: the installed CuTe DSL .* does not expose the kernel symbols"):
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
    """A version floor does not guarantee the part's target: with the version gate passed and a DSL whose target list lacks ``sm_107a``
    (a controlled missing-target probe), cc 10.7 is declined naming the target and the DSL found, cc 10.3 is admitted.  The version
    gate runs first: a too-old DSL is reported as too old, and the target probe (the first gate that imports the DSL) is never reached."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    msg = entry._dsl_error((10, 7))
    assert msg is not None and "sm_107a" in msg and "nvidia-cutlass-dsl 4.8.0" in msg and msg.startswith(entry.NAME)
    assert entry._dsl_target_error((10, 3)) is None and entry._dsl_target_error((10, 0)) is None
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))

    def unreachable():
        raise AssertionError("the target probe ran although the version gate declined")

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", unreachable)
    assert "found 4.7.0" in entry._dsl_error((10, 7))


def test_declines_a_compiler_that_rejects_the_parts_target(monkeypatch):
    """cc 10.7 under a DSL whose compiler is the CUDA 12 toolchain flavour: declined BEFORE JIT, naming the compiler found and the one
    required (the flavour rejects the ``sm_107a`` target at compile time although the DSL lists it).  cc 10.0 / 10.3 are served by
    both flavours."""
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
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (13, 3))
    assert "compiles with CUDA 13.3" in entry._dsl_error((10, 7))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: exports and the import boundary
# ---------------------------------------------------------------------------------------------------------------------------
def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.AddNormQuantDecode is entry.AddNormQuantDecode
    frost = importlib.import_module("cudnn.norm.frost")
    assert frost.AddNormQuantDecode is entry.AddNormQuantDecode and frost.Cfg is kcfg.Cfg
    assert "AddNormQuantDecode" in cudnn._LAZY_OPTIONAL_IMPORTS and "AddNormQuantDecode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS
    assert entry.scale_rows is kcfg.scale_rows and entry.QUANT_FORMS is kcfg.QUANT_FORMS and entry.NAME == kcfg.NAME
    with pytest.raises(AttributeError):
        frost.NoSuchName


def test_import_cudnn_does_not_import_the_kernel_or_torch():
    """The lazy boundary, in fresh interpreters: importing cudnn loads neither the step's modules nor the DSL nor torch; the config
    module loads neither the DSL nor torch; importing the host entry and constructing a host entry (its declines included) still
    loads neither the DSL nor the kernel module."""

    def fresh(code):
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()

    seen = "print(sorted(m for m in sys.modules if m.startswith('cudnn.norm') or m in ('cutlass', 'torch')))"
    assert fresh(f"import sys, cudnn; {seen}") == "[]"
    out = fresh(f"import sys, cudnn.norm.frost, cudnn.norm.frost.kernel.add_norm_quant_decode_config, cudnn.norm.frost.add_norm_quant_decode; {seen}")
    assert "cutlass" not in out and "torch" not in out and "add_norm_quant_decode_f16" not in out, out
    code = (
        "import sys, cudnn\n"
        "k = cudnn.AddNormQuantDecode('cpu', 5120, 1e-6, add=True, bf16=True, quant='nvfp4', pdl=True)\n"
        "try:\n"
        "    cudnn.AddNormQuantDecode('cpu', 5120, 1e-6, add=True, bf16=False, quant=None)\n"
        "except ValueError:\n"
        "    pass\n"
        "print(sorted(m for m in sys.modules if m == 'cutlass' or m.endswith('add_norm_quant_decode_f16')))\n"
    )
    assert fresh(code) == "[]"


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
    from cudnn.norm.frost import add_norm_quant_decode as entry
    from cudnn.norm.frost.kernel import add_norm_quant_decode_f16 as mod

    print("COMPILER", int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    half = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.BFloat16, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    byte = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.Uint8, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    one = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(1,), stride=(1,), assumed_align=4)
    for spec in sys.argv[3:]:
        before = set(os.listdir(dump))
        parts = spec.split("|")
        add, store, bf16, quant, pdl, release = (None if p == "None" else (p == "True" if p in ("True", "False") else p) for p in parts[1:7])
        cfg = entry.make_cfg(5120, 1e-6, add=add, store_residual=store, bf16=bf16, quant=quant, pdl=pdl, pdl_release=release)
        if parts[0] == "bind":  # the launch arguments the ENTRY builds from CUDA tensors of that many rows
            import torch

            rows = int(parts[7])
            dev = torch.device("cuda", torch.cuda.current_device())
            k = entry.AddNormQuantDecode(dev, 5120, 1e-6, add=add, store_residual=store, bf16=bf16, quant=quant, pdl=pdl, pdl_release=release)
            slab = lambda: torch.zeros(rows, 5120, dtype=torch.bfloat16, device=dev)
            args = k._args(slab(), slab() if (add or store) else None, torch.zeros(5120, dtype=torch.bfloat16, device=dev),
                           torch.ones(1, dtype=torch.float32, device=dev) if quant else None, k._owned_views(rows))
            cute.compile(mod.launch, cfg, *args, cutlass.Int32(rows), make_fake_stream())
        else:
            args = [half(), half() if (add or store) else None, half(), one() if quant else None, half() if bf16 else None,
                    byte() if quant else None, byte() if quant == "nvfp4" else None]
            cute.compile(mod.launch, cfg, *args, cutlass.Int32(2), make_fake_stream())
        for name in sorted(set(os.listdir(dump)) - before):
            print("KEPT", spec, name)
    print("COMPILED", arch)
    """)


def _spec(form, arm, kind="fake", rows=None):
    add, store, bf16, quant = FORMS[form]
    return "|".join(str(v) for v in (kind, add, store, bf16, quant, arm[0], arm[1]) + (() if rows is None else (rows,)))


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


def _body(text):
    """A rendering from its entry on, with the symbol normalized."""
    return re.sub(SYMBOL + r"\w+", "ENTRY", text[text.index(".entry") :])


def _without_pdl_blocks(text):
    """A rendering without its ``griddepcontrol`` inline blocks (the three lines of each)."""
    lines, keep, i = text.splitlines(), [], 0
    while i < len(lines):
        if "// begin inline asm" in lines[i] and i + 2 < len(lines) and "griddepcontrol" in lines[i + 1]:
            i += 3
            continue
        keep.append(lines[i])
        i += 1
    return "\n".join(keep)


@pytest.mark.parametrize("arch, cc", (("sm_100a", (10, 0)), ("sm_103a", (10, 3)), ("sm_107a", (10, 7))))
def test_rendering_pins_the_convention(arch, cc, tmp_path):
    """Trace-compile the kernel for a claimed target on this box (fake tensors with symbolic extents: no device needed) in all fifteen
    forms x three launch arms and pin the instructions that ARE the contract.  In EVERY rendering: eight ``fma.rn.f32`` (the fused sum
    of squares, never a separate multiply and add), one ``bar.sync``, one ``st.shared`` and one ``ld.shared`` (the warp sums), one
    ``rsqrt.approx.f32`` and one ``div.rn.f32`` by the width (the immediate ``0f45A00000`` = 5120; never ``div.approx`` / ``div.full`` /
    ``rsqrt.approx.ftz``), eps as ``0f358637BD``, eight ``cvt.rn.bf16.f32`` (the rounding of record), no ``mbarrier`` / ``cp.async`` /
    ``tcgen05``, ``.reqntid 640``, and the symbol = the kernel's name + the form text inside the first 100 characters.  Per form: ten
    ``shfl.sync.bfly`` (eleven with NVFP4: the block maximum); a 128-bit global load of ``x`` and of ``weight`` (one more with the add); a
    128-bit bf16 store per written bf16 operand; static E4M3: the reciprocal ``div.rn.f32`` with the immediate numerator 1.0, eight
    ``cvt.rn.satfinite.e4m3x2.f32`` and eight one-byte stores; NVFP4: ONE E4M3 convert (the block scale), two ``rcp.approx.ftz.f32``,
    fp32(1/6) as the bits ``0x3E2AAAAB``, four ``cvt.rn.satfinite.e2m1x2.f32``, one 32-bit store of the codes and one one-byte store of
    the scale; ``griddepcontrol.wait`` exactly under ``pdl`` and ``launch_dependents`` exactly under ``pdl_release``; the kernel's
    parameters = ``x`` and ``weight`` plus the form's live optional operands (a dead operand is dropped, never bound).  Within a form
    the three arms differ ONLY in the ``griddepcontrol`` blocks (the launch knobs move no arithmetic).  With an nvdisasm that decodes
    the cubin, also the SASS: one ``BAR.SYNC``, one shared store and load, no spill, ``ACQBULK`` / ``PREEXIT`` as the arm says.  A
    compile failure FAILS; a DSL that does not serve the target (version, target list, compiler) SKIPS with the gate's text."""
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    cells = [(form, arm, arm_id) for form in FORMS for arm, arm_id in zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)]
    major, kept = _render(tmp_path, arch, [_spec(form, arm) for form, arm, _ in cells])
    bodies = {}
    for form, arm, arm_id in cells:
        files = kept.get(_spec(form, arm), [])
        ptx_files = [f for f in files if f.endswith(".ptx")]
        assert len(ptx_files) == 1, f"{arch} {form} {arm_id}: expected one PTX, kept {files}"
        ptx = Path(ptx_files[0]).read_text()
        lines = ptx.splitlines()
        count = lambda needle: sum(1 for ln in lines if needle in ln)  # noqa: E731
        add, store, bf16, quant = FORMS[form]
        e4, fp4 = quant == "e4m3_static", quant == "nvfp4"
        stores128 = int(bf16) + int(add or store)
        wait, release = arm_id != "plain", arm_id == "wait-release"
        want = {
            "fma.rn.f32": 8,
            "bar.sync": 1,
            "st.shared": 1,
            "ld.shared": 1,
            "rsqrt.approx.f32": 1,
            "0f45A00000": 1,
            "0f358637BD": 1,
            "cvt.rn.bf16.f32": 8,
            "mbarrier": 0,
            "cp.async": 0,
            "tcgen05": 0,
            "div.approx": 0,
            "div.full": 0,
            "rsqrt.approx.ftz": 0,
            "shfl.sync.bfly": 11 if fp4 else 10,
            "div.rn.f32": 2 if e4 else 1,
            "ld.global.v4": 2 + int(add),
            "st.global.v4": stores128,
            "cvt.rn.bf16x2.f32": 4 * stores128,
            "cvt.rn.satfinite.e4m3x2.f32": 8 if e4 else (1 if fp4 else 0),
            "cvt.rn.satfinite.e2m1x2.f32": 4 if fp4 else 0,
            "rcp.approx.ftz.f32": 2 if fp4 else 0,
            "st.global.b8": 8 if e4 else (1 if fp4 else 0),
            "griddepcontrol.wait": int(wait),
            "griddepcontrol.launch_dependents": int(release),
        }
        what = f"{arch} {form} {arm_id} (compiler CUDA {major})"
        got = {needle: count(needle) for needle in want}
        assert got == want, f"{what}: " + ", ".join(f"{k} {got[k]} (want {want[k]})" for k in want if got[k] != want[k])
        assert f".target {arch}" in ptx and ".reqntid 640, 1, 1" in ptx, what
        assert sum(1 for ln in lines if "div.rn.f32" in ln and "0f3F800000," in ln) == int(e4), f"{what}: the reciprocal's immediate numerator"
        assert sum(1 for ln in lines if "1042983595" in ln or "3E2AAAAB" in ln) == int(fp4), f"{what}: fp32(1/6) as bits"
        params = len(re.findall(r"^\s*\.param", ptx, re.M))
        assert params == 2 + int(add or store) + 2 * int(quant is not None) + int(bf16) + int(fp4), f"{what}: {params} kernel parameters"
        symbol = re.search(r"\.entry\s+(\w+)", ptx).group(1)
        head = SYMBOL + str(_cfg(form, arm))
        assert symbol.startswith(head + "_") and len(head) <= 100, f"{what}: the symbol {symbol[:120]} does not carry the form in its first 100 characters"
        bodies[(form, arm_id)] = _body(ptx)
        cubins = [f for f in files if f.endswith(".cubin")]
        sass = _sass(cubins[0]) if cubins else None
        if sass is not None:
            n = lambda needle: sum(1 for ln in sass if needle in ln)  # noqa: E731
            assert n("BAR.SYNC") == 1 and n("STS") == 1 and n("LDS") == 1, f"{what}: the barrier and the warp-sum slot in the SASS"
            assert n(" STL") == 0 and n(" LDL") == 0, f"{what}: a spill in the SASS"
            assert n("ACQBULK") == int(wait) and n("PREEXIT") == int(release), what
            assert n("STG.E.128") == stores128 and n("STG.E.U8") == (8 if e4 else (1 if fp4 else 0)), f"{what}: the store forms in the SASS"
            assert n("FCHK") == (2 if e4 else 1) and n("MUFU.RSQ") >= 1, f"{what}: the IEEE divisions and the reciprocal root in the SASS"
    for form in FORMS:
        plain = _without_pdl_blocks(bodies[(form, "plain")])
        for arm_id in ("wait", "wait-release"):
            assert _without_pdl_blocks(bodies[(form, arm_id)]) == plain, f"{arch} {form}: the {arm_id} arm changes more than its griddepcontrol blocks"


def test_entry_binding_compiles_the_pinned_rendering(tmp_path):
    """Rule 4 on the compiled kernel: the launch arguments the ENTRY builds from CUDA tensors -- of 1 row and of 1024 rows -- compile
    to the PTX text of the fake-tensor rendering pinned above, byte for byte, the kernel symbol and its parameter block included.  So
    the kernel the entry ships is the pinned one whatever row count it is compiled at: no extent of an operand enters it.  Compile-only,
    for ``sm_100a`` (a target both compiler flavours assemble): any CUDA device serves, whatever its compute capability; nothing is
    launched."""
    if not torch.cuda.is_available():
        pytest.skip("a CUDA device is required to build the entry's launch arguments (any compute capability: the cell compiles, never launches)")
    msg = entry._dsl_error((10, 0))
    if msg is not None:
        pytest.skip(msg)
    arm = (True, None)
    for form in ("add-bf16-e4m3", "none-bf16-nvfp4"):
        texts = []
        # one fresh process per rendering: inside one process the DSL serves the second binding from its own compile cache and keeps no file
        for tag, spec in (("fake", _spec(form, arm)), ("rows1", _spec(form, arm, "bind", 1)), ("rows1024", _spec(form, arm, "bind", kcfg.MAX_ROWS))):
            sub = tmp_path / f"{form}-{tag}"
            sub.mkdir()
            _, kept = _render(sub, "sm_100a", [spec])
            ptx_files = [f for f in kept.get(spec, []) if f.endswith(".ptx")]
            assert len(ptx_files) == 1, f"{spec}: expected one PTX, kept {kept.get(spec)}"
            texts.append(Path(ptx_files[0]).read_text())
        entry_line = lambda text: next(ln for ln in text.splitlines() if ".entry" in ln)  # noqa: E731
        for text, rows in zip(texts[1:], (1, kcfg.MAX_ROWS)):
            assert (
                text == texts[0]
            ), f"{form}: the entry's launch arguments ({rows} rows) compile to another kernel than the pinned rendering:\n  pinned: {entry_line(texts[0])}\n  entry:  {entry_line(text)}"


# ---------------------------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------------------------
_ENTRIES = {}


def _entry(dev, form, arm=(None, None)):
    """One entry per (device, form, launch arm) for the cells that only launch (an entry is one compiled kernel for every row count)."""
    key = (str(dev), form, arm)
    if key not in _ENTRIES:
        _ENTRIES[key] = make_entry(dev, form, arm)
    return _ENTRIES[key]


def _bits(t):
    """A tensor as its bit pattern: bf16 through int16 (a signed zero and a NaN payload are their own values), codes through uint8."""
    return t.view(I16) if t.dtype == BF16 else t.view(U8)


def _launch(k, x, residual, weight, scale, **outs):
    """One call of ``k`` with the operands its form takes (``None`` for the others)."""
    cfg = k.cfg
    return k(x, residual if (cfg.ADD or cfg.STORE_RESIDUAL) else None, weight, scale if cfg.QUANT is not None else None, **outs)


def _snap(k, x, res0, weight, scale, **outs):
    """One call on a FRESH copy of the residual (it is in / out); ``{name: cloned tensor}`` of the residual after the call and of
    every output the form produces."""
    cfg = k.cfg
    residual = res0.clone() if (cfg.ADD or cfg.STORE_RESIDUAL) else None
    out, codes, scales = _launch(k, x, residual, weight, scale, **outs)
    torch.cuda.synchronize()
    got = dict(residual=residual, out=out, codes=codes, scales=scales)
    return {name: t.clone() for name, t in got.items() if t is not None}


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(_bits(a), _bits(b)))


def _assert_same(got, want, label, names=None):
    for name in want if names is None else names:
        assert _same(got[name], want[name]), f"{label}: {name} differs in {int((_bits(got[name]) != _bits(want[name])).sum())} of {got[name].numel()} elements"


def _assert_reference(form, got, x, res0, weight, scale, y=None, share_bound=False, label="", bracket=None):
    """The references of ``add_norm_quant_utils`` on one launch's bytes: the residual, the normalized value (where the form writes it;
    otherwise ``y`` is the bf16 twin form's output, already checked), the E4M3 codes, the NVFP4 block scales and the code bracket.
    ``bracket`` is the precomputed fp64 bracket of the same operands."""
    add, store, bf16, quant = FORMS[form]
    rows = x.shape[0]
    if add or store:
        assert torch.equal(_bits(got["residual"].cpu()), _bits(residual_reference(x, res0, add))), f"{label}: the residual"
    else:
        assert "residual" not in got
    if bf16:
        assert_y(got["out"], x, res0, weight, add, share_bound=share_bound, label=label, bracket=bracket)
        y = got["out"]
    else:
        assert "out" not in got and y is not None
    if quant == "e4m3_static":
        assert got["codes"].dtype == E4M3 and tuple(got["codes"].shape) == (rows, HIDDEN)
        want = e4m3_codes(y, scale)
        n = int((got["codes"].view(U8).cpu() != want).sum())
        assert n == 0, f"{label}: {n} E4M3 codes differ from the static quantization of the kernel's own bf16 value"
    if quant == "nvfp4":
        assert got["codes"].dtype == U8 and tuple(got["codes"].shape) == (rows, HIDDEN // 2)
        assert got["scales"].dtype == E4M3 and tuple(got["scales"].shape) == (kcfg.scale_rows(rows), HIDDEN // 16)
        stored = scale_bytes_of_rows(got["scales"], rows)
        n = int((stored != nvfp4_scale_bytes(y, scale)).sum())
        assert n == 0, f"{label}: {n} NVFP4 block scales differ from e4m3(gs x (amax x fp32(1/6))) of the kernel's own bf16 value"
        assert_nvfp4_codes(got["codes"], y, stored, scale, label)
        pad = ~scale_bytes_layout_mask(kcfg.scale_rows(rows), rows)
        assert bool((got["scales"].view(U8).cpu().reshape(-1)[pad] == 0).all()), f"{label}: the entry-owned scale buffer's pad rows are zero"
    if quant is None:
        assert "codes" not in got and "scales" not in got


def _twin(form):
    """The form with the same residual mode and quant that also writes the bf16 value."""
    mode, _, rest = form.partition("-")
    return form if rest.startswith("bf16") else f"{mode}-bf16-{rest}"


_OPERANDS, _BRACKETS = {}, {}


def _reference_operands(dev, kind):
    """The operand set of a kind for the row ceiling (one draw per kind: the cells slice its first rows)."""
    key = (str(dev), kind)
    if key not in _OPERANDS:
        _OPERANDS[key] = make_operands(max(ROWS), HIDDEN, dev, seed=17 if kind == "random" else 23, kind=kind)
    return _OPERANDS[key]


def _reference_bracket(dev, kind, add, rows):
    """The fp64 bracket of the first ``rows`` rows of a kind's operands (it depends on the operands and the residual mode's function
    only: every form and every scale of one function shares it)."""
    from add_norm_quant_utils import y_bracket

    key = (str(dev), kind, add, rows)
    if key not in _BRACKETS:
        xs, rs, w = _reference_operands(dev, kind)
        _BRACKETS[key] = y_bracket(xs[:rows], rs[:rows], w, add)
    return _BRACKETS[key]


@pytest.mark.parametrize("kind", ("random", "structured"))
@pytest.mark.parametrize("form", sorted(FORMS))
def test_matches_the_reference(form, kind):
    """Every form at fourteen row counts (1 .. 1024: one thread block, a partial and a full scale tile, more rows than one scheduling
    wave holds) on random and structured operands and several scales: the residual bitwise, the normalized value inside the fp64
    bracket (and, on random operands, off the true nearest no more often than the arithmetic allows), the E4M3 codes and the NVFP4
    block scales bitwise the model on the kernel's own bf16 value, the NVFP4 codes inside the reciprocal bracket.  A form that does
    not write the bf16 value is checked against its twin form's (and writes the twin's code bytes).  The scale moves the codes only:
    the normalized value and the residual of every scale are the bytes of the first."""
    dev = _device_or_skip()
    add, store, bf16, quant = FORMS[form]
    k = _entry(dev, form)
    twin = None if bf16 else _entry(dev, _twin(form))
    xs, rs, w = _reference_operands(dev, kind)
    scales = (None,) if quant is None else (E4M3_SCALES if quant == "e4m3_static" else NVFP4_SCALES)
    for rows in ROWS:
        x, res0 = xs[:rows], rs[:rows]
        bracket = _reference_bracket(dev, kind, add, rows)
        first = None
        for value in scales if rows in (1, 17, 160) else scales[:2]:
            scale = None if value is None else scale_tensor(value, dev)
            label = f"{form} rows {rows} {kind} scale {value}"
            got = _snap(k, x, res0, w, scale)
            y = None
            if twin is not None:
                ref = _snap(twin, x, res0, w, scale)
                if first is None:
                    assert_y(ref["out"], x, res0, w, add, share_bound=kind == "random", label=label + " (twin)", bracket=bracket)
                _assert_same(got, ref, label + ": against the form that also writes bf16", names=list(got))
                y = ref["out"]
            if first is None:
                _assert_reference(form, got, x, res0, w, scale, y=y, share_bound=kind == "random", label=label, bracket=bracket)
                first = dict(got, y=y)
            else:
                _assert_same(got, first, label + ": the scale moves codes only", names=[n for n in ("out", "residual") if n in got])
                if twin is not None:
                    assert _same(y, first["y"]), label + ": the twin's bf16 value does not depend on the scale"
                _assert_reference(form, dict(got, **({"out": first["out"]} if bf16 else {})), x, res0, w, scale, y=y, label=label, bracket=bracket)
    assert len(k.compiled) == 1, "one compiled kernel served every row count"


def _exact_targets():
    """``[hidden]`` targets of the exact cells, per 16-element block: blocks 0-3 a block maximum of ``6 x 2^m`` with the seven E2M1
    midpoints times ``2^m`` twice and a zero (power-of-two block scales: exact codes); blocks 8-9 a maximum of 9.0 and 10.5 with the
    midpoints times 1.5 and 1.75 (block scales 1.5 and 1.75 at global scale 1: every midpoint element is an EXACT tie decided by the
    approximate reciprocal); block 20 all zero; block 21 all at the maximum; the rest 1.0."""
    mids = torch.tensor(E2M1_MIDPOINTS)
    t = torch.ones(HIDDEN)
    for i, (peak, unit) in enumerate(((6.0, 1.0), (12.0, 2.0), (0.75, 0.125), (96.0, 16.0))):
        t[16 * i : 16 * i + 16] = torch.cat([torch.tensor([peak]), mids * unit, mids * unit, torch.tensor([0.0])])
    for i, (peak, unit) in ((8, (9.0, 1.5)), (9, (10.5, 1.75))):
        t[16 * i : 16 * i + 16] = torch.cat([torch.tensor([peak]), mids * unit, mids * unit, torch.tensor([0.0])])
    t[320:336] = 0.0
    t[336:352] = 6.0
    return t


def _exact_signs(rows):
    gen = torch.Generator(device="cpu").manual_seed(9)
    signs = torch.where(torch.rand(rows, HIDDEN, generator=gen) < 0.5, -1.0, 1.0)
    signs[0] = 1.0
    signs[1] = -1.0
    return signs


def _assert_exact_nvfp4_codes(codes, y, block_scale, gs, rows):
    """The exact cells' codes against exact arithmetic: ``a = y x (gs / F)`` in fp64 (``F`` the stored block scale; a zero ``F`` encodes
    the signed zero of ``y``).  Blocks 0-3 and 21 (power-of-two ``F``) and the all-zero block 20: the code is EXACTLY the nearest,
    ties to the even code.  Blocks 8-9 (``F`` = 1.5, 1.75 times the global scale): fourteen elements each sit exactly on an E2M1
    midpoint and take one of the midpoint's two neighbours; the other two (the maximum, the zero) are exact."""
    a = torch.where(block_scale != 0, y * gs / block_scale.clamp(min=1e-30), y * 0.0)
    exact = e2m1_codes(a)
    for lo, hi, what in ((0, 64, "power-of-two block scales"), (320, 336, "the all-zero block"), (336, 352, "a block at its maximum")):
        assert torch.equal(codes[:, lo:hi], exact[:, lo:hi]), f"global scale {gs}: {what}: the codes are exact"
    assert torch.equal(codes[:, 320:336], torch.signbit(y[:, 320:336]).to(U8) << 3), "an all-zero block: the signed zero's code (0 or 8)"
    mids = torch.tensor(E2M1_MIDPOINTS, dtype=torch.float64)
    mag, sign = a[:, 128:160].abs(), torch.signbit(a[:, 128:160]).to(U8) << 3
    tie = torch.isin(mag, mids)
    assert int(tie.sum()) == rows * 2 * 14, "blocks 8 and 9: fourteen exact ties each"
    got = codes[:, 128:160]
    lower, upper = torch.bucketize(mag, mids, right=False).to(U8) | sign, torch.bucketize(mag, mids, right=True).to(U8) | sign
    assert bool(((got == lower) | (got == upper))[tie].all()), f"global scale {gs}: a tie element's code is one of the tie's two neighbours"
    assert torch.equal(got[~tie], exact[:, 128:160][~tie]), f"global scale {gs}: the elements beside the ties are exact"


@pytest.mark.parametrize("form", ("add-bf16-nvfp4", "store-bf16-nvfp4", "none-nvfp4"))
def test_exact_cells_nvfp4(form):
    """The exact construction (``x = +-1``, the weight places ``y``): the bf16 output is bitwise the targets; the block scales are
    bitwise ``e4m3(gs x (max x fp32(1/6)))`` -- 1.0, 2.0, 0.125, 16.0, 1.5, 1.75, zero for the all-zero block -- under global scales 1
    and 0.25.  Power-of-two block scales: the codes are EXACT (the encode scale and its reciprocals are powers of two), every E2M1
    midpoint rounds to its even neighbour, the maximum is code 7, a zero is the signed zero of ``x`` (code 0 or 8).  Block scales 1.5
    and 1.75: every midpoint element is an exact tie in exact arithmetic and its code is one of the tie's two neighbours (which one is
    the approximate reciprocal's; not claimed equal across GPU architectures); every other element is exact.  Replayed eight times:
    one set of bytes."""
    dev = _device_or_skip()
    add, store, bf16, quant = FORMS[form]
    rows = 4
    x, res0, w, y = exact_operands(_exact_targets(), _exact_signs(rows), dev)
    k = _entry(dev, form)
    for gs in (1.0, 0.25):
        scale = scale_tensor(gs, dev)
        got = _snap(k, x, res0, w, scale)
        if bf16:
            assert _same(got["out"], y), "the exact construction: out is bitwise the targets (a failure here is the construction's, not a code's)"
        stored = scale_bytes_of_rows(got["scales"], rows)
        assert torch.equal(stored, nvfp4_scale_bytes(y, scale)), f"global scale {gs}: the block scales"
        value = stored.view(E4M3).float()
        assert torch.equal(value[:, :4], torch.tensor([1.0, 2.0, 0.125, 16.0]).mul(gs).expand(rows, 4)) and torch.equal(
            value[:, 8:10], torch.tensor([1.5, 1.75]).mul(gs).expand(rows, 2)
        )
        assert bool((stored[:, 20] == 0).all()) and bool((value[:, 21] == gs).all())
        codes = unpack_e2m1(got["codes"].cpu())
        _assert_exact_nvfp4_codes(codes, y.cpu().double(), value.double().repeat_interleave(16, dim=1), gs, rows)
        ties = assert_nvfp4_codes(got["codes"], y, stored, scale, f"{form} global scale {gs}")
        assert ties >= rows * 2 * 14, f"the tie cells were not ties ({ties} two-code elements): the construction did not place y on the midpoints"
        for _ in range(8):
            _assert_same(_snap(k, x, res0, w, scale), got, f"{form} global scale {gs}: replay")


def test_exact_cells_e4m3():
    """The exact construction under the static E4M3 form: ``out`` bitwise the targets and the codes bitwise ``e4m3(y / scale)`` --
    exact at scale 1 for E4M3 values, saturated at +-448 (``0x7E`` / ``0xFE``) at scale 2^-7, the zero's code by sign (``0x00`` /
    ``0x80``)."""
    dev = _device_or_skip()
    rows = 4
    x, res0, w, y = exact_operands(_exact_targets(), _exact_signs(rows), dev)
    for form in ("store-bf16-e4m3", "none-e4m3"):
        k = _entry(dev, form)
        for value in (1.0, 2.0**-7, 0.25):
            scale = scale_tensor(value, dev)
            got = _snap(k, x, res0, w, scale)
            if "out" in got:
                assert _same(got["out"], y)
            if "residual" in got:
                assert _same(got["residual"], x), "store_residual writes x"
            codes = got["codes"].view(U8).cpu()
            assert torch.equal(codes, e4m3_codes(y, scale)), f"{form} scale {value}"
            if value == 2.0**-7:
                assert bool((codes[0, 0:1] == 0x7E).all()) and bool((codes[1, 0:1] == 0xFE).all()), "6 / 2^-7 = 768 saturates at +-448"
            assert bool((codes[0, 320:336] == 0x00).all()) and bool((codes[1, 320:336] == 0x80).all()), "the zero's code carries the sign"


def test_edge_rows():
    """Edge rows beside a plain one: an all-zero row (``y`` a signed zero whose sign is ``1 + w``'s, zero block scales), negative zeros,
    one dominant element, values whose squares underflow, a large in-domain row, a residual that cancels ``x`` -- each inside the same
    references.  Then a NaN and an infinity in two rows: those rows' ``out`` is non-finite, every OTHER row is byte-identical to the
    clean run (the non-finite rows' codes are out of contract and not asserted)."""
    dev = _device_or_skip()
    x, res0, w = edge_operands(dev)
    rows = x.shape[0]
    for form in ("add-bf16-e4m3", "add-bf16-nvfp4", "none-bf16-e4m3", "store-bf16-nvfp4", "add-bf16"):
        add, store, bf16, quant = FORMS[form]
        k = _entry(dev, form)
        scale = None if quant is None else scale_tensor(0.02 if quant == "e4m3_static" else 37.0, dev)
        got = _snap(k, x, res0, w, scale)
        _assert_reference(form, got, x, res0, w, scale, label=f"{form} edge rows")
        out = got["out"].float().cpu()
        assert bool((out[0] == 0).all()) and torch.equal(
            torch.signbit(out[0]), torch.signbit(1.0 + w.float().cpu())
        ), "an all-zero row: signed zeros with the sign of 1 + w"
        if quant == "nvfp4":
            assert bool((scale_bytes_of_rows(got["scales"], 1) == 0).all()), "an all-zero row: zero block scales"
        bad_x = x.clone()
        bad_x[2, 77] = float("nan")
        bad_x[4, 4000] = float("inf")
        bad = _snap(k, bad_x, res0, w, scale)
        finite = torch.isfinite(bad["out"].float()).all(dim=1).cpu()
        assert finite.tolist() == [True, True, False, True, False, True, True], "a NaN or an infinity makes its own row non-finite"
        clean = [r for r in range(rows) if r not in (2, 4)]
        for name in ("out", "codes", "residual"):
            if name in got:
                assert _same(bad[name][clean], got[name][clean]), f"{form}: {name} of the rows beside a non-finite row"
        if quant == "nvfp4":
            assert torch.equal(scale_bytes_of_rows(bad["scales"], rows)[clean], scale_bytes_of_rows(got["scales"], rows)[clean])


def _outs(cfg, rows, dev, extra=3):
    """Caller-owned outputs of a form filled with sentinel bytes: ``extra`` rows taller than the call (and one more scale tile) when
    ``extra`` is not zero, exactly the call's shapes otherwise."""
    outs = {}
    if cfg.BF16_OUT:
        outs["out"] = torch.full((rows + extra, cfg.HIDDEN), 0x5A5A, dtype=I16, device=dev).view(BF16)
    if cfg.QUANT == "e4m3_static":
        outs["codes"] = torch.full((rows + extra, cfg.HIDDEN), 0xA5, dtype=U8, device=dev).view(E4M3)
    if cfg.QUANT == "nvfp4":
        outs["codes"] = torch.full((rows + extra, cfg.HIDDEN // 2), 0xA5, dtype=U8, device=dev)
        outs["scales"] = torch.full((kcfg.scale_rows(rows) + (128 if extra else 0), cfg.HIDDEN // 16), 0xA5, dtype=U8, device=dev).view(E4M3)
    return outs


@pytest.mark.parametrize("form", ("add-bf16-nvfp4", "store-bf16-e4m3", "none-bf16", "add-nvfp4"))
def test_output_ownership_per_output(form):
    """Ownership is per output: for EVERY non-empty subset of the form's outputs given by the caller, the returned tuple holds the
    caller's tensors (as given: the same object, never re-sliced) in those slots and the entry-owned buffers of this row count in the
    others; the bytes are those of the all-entry-owned run; a caller-owned output's rows past the call and the pad rows of a
    caller-owned scale buffer keep the caller's bytes; an entry-owned buffer is not touched by a call that owns that slot; ``x``,
    ``weight`` and ``scale`` are unmodified."""
    import itertools

    dev = _device_or_skip()
    rows = 5
    k = make_entry(dev, form)
    cfg = k.cfg
    x, res0, w = make_operands(rows, HIDDEN, dev, seed=31)
    scale = None if cfg.QUANT is None else scale_tensor(0.02 if cfg.QUANT == "e4m3_static" else 37.0, dev)
    keep = (x.clone(), w.clone(), None if scale is None else scale.clone())
    base = _snap(k, x, res0, w, scale)
    owned = k.outputs(rows)
    names = entry.produced_slots(cfg)
    assert set(owned) == set(names) and tuple(n for n in entry.OUTPUT_SLOTS if n in owned) == names
    other_x, other_res, _ = make_operands(rows, HIDDEN, dev, seed=77)
    live = None if cfg.QUANT != "nvfp4" else scale_bytes_layout_mask(kcfg.scale_rows(rows) + 128, rows).to(dev)
    for count in range(1, len(names) + 1):
        for subset in itertools.combinations(names, count):
            mine = {name: t for name, t in _outs(cfg, rows, dev).items() if name in subset}
            before = {name: _bits(t).clone() for name, t in owned.items()}
            residual = other_res.clone() if (cfg.ADD or cfg.STORE_RESIDUAL) else None
            _launch(k, other_x, residual, w, scale, **mine)  # other operands into the caller's buffers: an entry-owned slot it OWNS stays as it was
            torch.cuda.synchronize()
            for name in subset:
                assert torch.equal(_bits(owned[name]), before[name]), f"{subset}: the call wrote the entry-owned {name} although the caller owns that slot"
            residual = res0.clone() if (cfg.ADD or cfg.STORE_RESIDUAL) else None
            result = dict(zip(entry.OUTPUT_SLOTS, _launch(k, x, residual, w, scale, **mine)))
            torch.cuda.synchronize()
            for name in entry.OUTPUT_SLOTS:
                if name not in names:
                    assert result[name] is None, f"{name}: a slot the form does not produce is None"
                elif name in subset:
                    assert result[name] is mine[name], f"{name}: the caller's tensor comes back as given"
                else:
                    assert result[name] is owned[name]
            for name in names:
                if name == "scales":
                    got = scale_bytes_of_rows(result[name], rows)
                    assert torch.equal(got, scale_bytes_of_rows(base[name], rows)), f"{subset}: the live rows' scale bytes"
                    if name in subset:
                        flat = result[name].view(U8).reshape(-1)
                        assert bool((flat[~live] == 0xA5).all()), "the kernel writes the LIVE rows' scale bytes only: pad rows keep the caller's bytes"
                else:
                    assert _same(result[name][:rows], base[name]), f"{subset}: {name}"
                    if name in subset:
                        tail = _bits(result[name])[rows:]
                        assert bool((tail == (0x5A5A if result[name].dtype == BF16 else 0xA5)).all()), f"{name}: rows past the call are never written"
            if residual is not None:
                assert _same(residual, base["residual"])
    for now, was in zip((x, w, scale), keep):
        if now is not None:
            assert torch.equal(
                now.view(I16 if now.dtype == BF16 else torch.int32), was.view(I16 if was.dtype == BF16 else torch.int32)
            ), "the inputs are unmodified"


def test_caller_owned_scales_keep_the_owners_bytes_past_the_live_rows():
    """The documented contract of a caller-owned ``scales`` buffer across row counts: after a call at 130 rows, a call at 3 rows into
    the SAME buffer rewrites the bytes of rows 0 .. 2 only -- rows 3 .. 129 still hold the first call's bytes (the kernel writes live
    rows only; the owner clears the buffer when the row count shrinks).  The entry-owned buffers are per row count and show zeros."""
    dev = _device_or_skip()
    k = make_entry(dev, "none-bf16-nvfp4")
    x, _, w = make_operands(130, HIDDEN, dev, seed=41)
    scale = scale_tensor(37.0, dev)
    mine = torch.zeros(256, HIDDEN // 16, dtype=E4M3, device=dev)
    k(x, None, w, scale, scales=mine)
    torch.cuda.synchronize()
    first = scale_bytes_of_rows(mine, 130).clone()
    assert int((first != 0).sum()) > 100 * 320, "the first call wrote scales for 130 rows"
    other_x, _, _ = make_operands(3, HIDDEN, dev, seed=42)
    _, _, returned = k(other_x, None, w, scale, scales=mine)
    torch.cuda.synchronize()
    assert returned is mine
    after = scale_bytes_of_rows(mine, 130)
    assert torch.equal(after[3:], first[3:]), "rows past the second call's live rows keep the first call's bytes"
    assert not torch.equal(after[:3], first[:3]), "the live rows were rewritten"
    _, _, owned = k(other_x, None, w, scale)
    torch.cuda.synchronize()
    assert tuple(owned.shape) == (128, HIDDEN // 16) and torch.equal(scale_bytes_of_rows(owned, 3), after[:3])
    pad = ~scale_bytes_layout_mask(128, 3)
    assert bool((owned.view(U8).cpu().reshape(-1)[pad] == 0).all()), "the entry-owned buffer of 3 rows is zero past its live rows"


@pytest.mark.parametrize("rows", (1, 17, 160, 1024))
@pytest.mark.parametrize("quant", (None, "e4m3_static", "nvfp4"))
def test_bit_neutral_arms_write_the_default_bytes(quant, rows):
    """What never changes a byte: the launch arms (wait; wait + release) against the plain launch; writing the bf16 value or not;
    storing ``x`` to the residual or not.  The launch arms run BEHIND a real predecessor on the same stream -- a torch copy that
    rewrites ``x`` and the residual right before the launch -- so the wait has something to wait for: a kernel that read ``x`` before
    its wait would see the previous values.  The release has no dependent in this cell (inert here; the cell below gives it one)."""
    dev = _device_or_skip()
    tail = {None: "", "e4m3_static": "-e4m3", "nvfp4": "-nvfp4"}[quant]
    scale = None if quant is None else scale_tensor(0.02 if quant == "e4m3_static" else 37.0, dev)
    xs, rs, w = make_operands(rows, HIDDEN, dev, seed=51 + rows)
    stale_x, stale_r, _ = make_operands(rows, HIDDEN, dev, seed=52 + rows)
    for mode in ("add", "store", "none"):
        base_form = f"{mode}-bf16{tail}"
        base = _snap(_entry(dev, base_form), xs, rs, w, scale)
        for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
            k = _entry(dev, base_form, arm)
            x, residual = stale_x.clone(), stale_r.clone()
            _launch(k, x, residual, w, scale)  # warm (compile outside the measured sequence), on the stale values
            x.copy_(xs)  # the predecessors on the stream: they WRITE the operands this launch reads behind its wait
            residual.copy_(rs)
            out, codes, scales = _launch(k, x, residual, w, scale)
            torch.cuda.synchronize()
            got = {n: t for n, t in dict(out=out, codes=codes, scales=scales).items() if t is not None}
            if mode != "none":
                got["residual"] = residual
            _assert_same(got, base, f"{base_form} {arm_id} rows {rows}")
        if quant is not None:  # the form without the bf16 value writes the same residual, codes and scales
            lean = _snap(_entry(dev, f"{mode}{tail}"), xs, rs, w, scale)
            _assert_same(lean, base, f"{mode}{tail} against {base_form} rows {rows}", names=list(lean))
    stored = _snap(_entry(dev, f"store-bf16{tail}"), xs, rs, w, scale)
    plain = _snap(_entry(dev, f"none-bf16{tail}"), xs, rs, w, scale)
    _assert_same(stored, plain, f"store against none rows {rows}", names=list(plain))
    assert _same(stored["residual"], xs), "store_residual writes x bitwise"


def test_release_overlaps_a_dependent_without_changing_bytes():
    """The release with a dependent on the stream: the step built with ``pdl=True`` (wait + release) followed by a launch that carries
    the programmatic-stream-serialization attribute, waits, and copies the step's ``out`` and ``codes`` (``pdl_dependent_copy.py``).
    The dependent may start while the step runs; its wait holds its reads.  Eager and captured in a CUDA graph, at 1 and 32 rows: the
    copied bytes are those of the plain step."""
    dev = _device_or_skip()
    from pdl_dependent_copy import DependentCopy

    copy = DependentCopy(dev)
    form = "add-bf16-e4m3"
    k = make_entry(dev, form, (True, None))
    assert k.cfg.PDL and k.cfg.PDL_RELEASE
    scale = scale_tensor(0.02, dev)
    for rows in (1, 32):
        x, res0, w = make_operands(rows, HIDDEN, dev, seed=61 + rows)
        want = _snap(_entry(dev, form), x, res0, w, scale)
        k.compile(rows)
        out_copy = torch.zeros(rows, HIDDEN, dtype=BF16, device=dev)
        codes_copy = torch.zeros(rows, HIDDEN, dtype=E4M3, device=dev)
        residual = res0.clone()

        def chain():
            out, codes, _ = k(x, residual, w, scale)
            copy(out, out_copy)
            copy(codes, codes_copy)

        chain()  # eager (also compiles the dependent)
        torch.cuda.synchronize()
        got = dict(out=out_copy, codes=codes_copy, residual=residual)
        _assert_same(got, want, f"eager chain rows {rows}")
        for t in (out_copy, codes_copy):
            _bits(t).fill_(0)
        residual.copy_(res0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain()
        for _ in range(3):
            for t in (out_copy, codes_copy):
                _bits(t).fill_(0)
            residual.copy_(res0)
            graph.replay()
            torch.cuda.synchronize()
            _assert_same(got, want, f"captured chain rows {rows}")


def test_one_compile_serves_every_row_count(monkeypatch):
    """Compiled once without an operand, the entry launches every row count -- entry-owned and caller-owned outputs, a 0-d and a
    ``[1]`` scale, a ``[hidden]`` and a ``[1, hidden]`` weight -- with compilation FORBIDDEN (Rule 4: the row count is a runtime
    argument, never a key)."""
    dev = _device_or_skip()
    form = "add-bf16-nvfp4"
    k = make_entry(dev, form)
    k.compile()
    assert len(k.compiled) == 1 and set(k.buffers) == {1}
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("a warm call compiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    xs, rs, w = make_operands(max(ROWS), HIDDEN, dev, seed=71)
    scale = scale_tensor(37.0, dev)
    for rows in ROWS:
        x, res0 = xs[:rows], rs[:rows]
        got = _snap(k, x, res0, w, scale)
        _assert_reference(form, got, x, res0, w, scale, label=f"rows {rows} after one compile")
        _assert_same(_snap(k, x, res0, w.reshape(1, HIDDEN), scale_tensor(37.0, dev, zero_d=True)), got, f"rows {rows}: a 0-d scale and a [1, hidden] weight")
        mine = {name: t for name, t in _outs(k.cfg, rows, dev, extra=0).items()}
        mine["scales"].view(U8).zero_()
        theirs = _snap(k, x, res0, w, scale, **mine)
        _assert_same({n: (t[:rows] if n != "scales" else t[: kcfg.scale_rows(rows)]) for n, t in theirs.items()}, got, f"rows {rows}: caller-owned outputs")
    k.compile(3)  # a second compile() is a no-op for the kernel
    assert len(k.compiled) == 1 and set(k.buffers) == set(ROWS)


def test_compile_declines_a_dsl_below_the_floor_before_the_kernel_is_built(monkeypatch):
    dev = _device_or_skip()
    from cudnn.frost import buffers

    k = make_entry(dev, "add-bf16")
    x, res0, w = make_operands(1, HIDDEN, dev)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.1"):
        k.compile(1)
    with pytest.raises(NotImplementedError, match="found 4.7.1"):
        k(x, res0.clone(), w, None)
    assert not k.compiled


def test_graph_capture_replays_bitwise_with_eager():
    """Several layers' launches of two forms captured into one CUDA graph after ``compile(rows)``: the replay is bitwise the eager
    run, and after the operands and the scale change IN PLACE the replay is bitwise a fresh eager run on the new values (every tensor
    is a launch argument: the graph holds pointers, never values).  The residual is in / out, so it is restored before every run."""
    dev = _device_or_skip()
    rows = 6
    forms = ("add-bf16-e4m3", "add-nvfp4", "add-bf16-e4m3", "store-bf16-nvfp4")
    entries = [_entry(dev, form, (True, None)) for form in forms]
    for k in entries:
        k.compile(rows)
    layers = [make_operands(rows, HIDDEN, dev, seed=100 + i, kind="structured" if i == 1 else "random") for i in range(len(forms))]
    scales = [scale_tensor(v, dev) for v in (0.02, 37.0, 1.0, 0.25)]
    residuals = [r.clone() for _, r, _ in layers]
    outs = [_outs(k.cfg, rows, dev, extra=0) for k in entries]

    def eager():
        results = []
        for k, (x, r, w), s in zip(entries, layers, scales):
            results.append(_snap(k, x, r, w, s))
        return results

    want = eager()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for k, (x, _, w), s, res, o in zip(entries, layers, scales, residuals, outs):
            _launch(k, x, res, w, s, **o)

    def replay():
        for (_, r, _), res, o in zip(layers, residuals, outs):
            res.copy_(r)
            for name, t in o.items():
                _bits(t).fill_(0 if name == "scales" else 0x55)
        graph.replay()
        torch.cuda.synchronize()
        return [dict(o, **({"residual": res} if (k.cfg.ADD or k.cfg.STORE_RESIDUAL) else {})) for k, res, o in zip(entries, residuals, outs)]

    for _ in range(3):
        for got, ref in zip(replay(), want):
            _assert_same(got, ref, "graph replay bitwise with eager")
    for i, (x, r, w) in enumerate(layers):  # new values in the same storage
        nx, nr, nw = make_operands(rows, HIDDEN, dev, seed=500 + i)
        x.copy_(nx)
        r.copy_(nr)
        w.copy_(nw)
        scales[i].fill_((1.0, 0.25, 0.02, 37.0)[i])
    got_all = [{n: t.clone() for n, t in got.items()} for got in replay()]
    for got, ref in zip(got_all, eager()):
        _assert_same(got, ref, "the replay reads the operands' current values")


def test_warm_call_allocates_nothing_and_never_synchronizes():
    """Rules 1 and 3 on the warm path: with caller-owned outputs and with entry-owned outputs of a row count that was already called,
    three calls allocate no device memory, and a call under ``set_sync_debug_mode('error')`` raises nothing (no device read reaches
    the host; ``scale``'s value is never fetched)."""
    dev = _device_or_skip()
    rows = 8
    k = make_entry(dev, "add-bf16-nvfp4", (True, None))
    x, res0, w = make_operands(rows, HIDDEN, dev, seed=81)
    scale = scale_tensor(37.0, dev)
    mine = _outs(k.cfg, rows, dev, extra=0)
    mine["scales"].view(U8).zero_()
    residual = res0.clone()
    want = _snap(k, x, res0, w, scale)
    _launch(k, x, residual, w, scale, **mine)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            residual.copy_(res0)
            _launch(k, x, residual, w, scale, **mine)
            residual.copy_(res0)
            _launch(k, x, residual, w, scale)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    _assert_same(dict(mine, residual=residual), want, "the warm calls' bytes")


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    dev = _device_or_skip()
    k = make_entry(dev, "add-bf16-e4m3")
    x, res0, w = make_operands(2, HIDDEN, dev)
    scale = scale_tensor(0.02, dev)
    want = _snap(k, x, res0, w, scale)
    residual = res0.clone()
    with pytest.raises(TypeError, match="x must be torch.bfloat16"):
        k(x.float(), residual, w, scale)
    with pytest.raises(ValueError, match="every operand must live on one device"):
        k(x.cpu(), residual, w, scale)
    with pytest.raises(ValueError, match=r"x must be \[rows, 5120\]"):
        k(x[:, :2048].contiguous(), residual, w, scale)
    with pytest.raises(ValueError, match="residual is required by this form"):
        k(x, None, w, scale)
    with pytest.raises(ValueError, match="scale must be one fp32 element"):
        k(x, residual, w, torch.tensor([0.01, 0.02], device=dev))
    with pytest.raises(ValueError, match="residual must not overlap x"):
        k(x, x, w, scale)
    with pytest.raises(ValueError, match="scales was given but the form does not write it"):
        k(x, residual, w, scale, scales=torch.zeros(128, 320, dtype=E4M3, device=dev))
    torch.cuda.synchronize()
    assert _same(residual, res0), "a declined warm call launches nothing: the residual is untouched"
    owned = k.outputs(2)
    assert _same(owned["out"], want["out"]) and _same(owned["codes"], want["codes"])


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
    form = "add-bf16-e4m3"
    k = make_entry(other, form)
    x, res0, w = make_operands(3, HIDDEN, other, seed=91)
    scale = scale_tensor(0.02, other)
    residual = res0.clone()
    out, codes, _ = k(x, residual, w, scale)
    torch.cuda.synchronize(other)
    assert torch.cuda.current_device() == current and out.device == other and codes.device == other
    with torch.cuda.device(other):  # the references are plain torch math: run them with the operands' device current
        _assert_reference(form, dict(residual=residual, out=out, codes=codes), x, res0, w, scale, label="an entry on a non-current device")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
@pytest.mark.parametrize("form, rows", (("add-bf16-e4m3", 1), ("add-nvfp4", 3), ("none-bf16", 17), ("store-bf16-nvfp4", 160)))
def test_launch_facts_equal_the_profilers_record_of_a_launch(form, rows, arm, arm_id):
    """One call is ONE kernel launch whose profiler record equals the entry's launch facts: the kernel's name is the entry's symbol of
    the compiled form -- which carries the form's text directly behind the kernel's name -- and grid, block and shared memory are the
    launch shape.  Before the first compile the entry names no symbol."""
    dev = _device_or_skip()
    k = make_entry(dev, form, arm)
    assert k.kernel_symbols_for(rows) == (), "nothing is compiled yet"
    x, res0, w = make_operands(rows, HIDDEN, dev, seed=95)
    scale = None if k.cfg.QUANT is None else scale_tensor(1.0, dev)
    residual = res0.clone()
    record = assert_facts_equal_the_launch(k, lambda: _launch(k, x, residual, w, scale), rows)
    head = SYMBOL + str(k.cfg)
    assert record.name.startswith(head + "_") and len(head) <= 100, record.name
    assert k.kernel_symbols_for(rows) == (record.name,) == k.kernel_symbols_for(kcfg.MAX_ROWS), "one compiled kernel answers every row count"
    assert k.form_for(rows) == asdict(k.cfg)
