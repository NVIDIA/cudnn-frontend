# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP4 row quantization decode step (``cudnn.row_quantize.frost.row_quantize_fp4_decode``): on the host the config's limits and
tables, the three launch arms, every typed decline of the entry (CPU tensors), the DSL gates (version, target, compiler), the
plan-time-only compile key, the launch facts, the lazy export, why the reference's global scales are what they are, and the kernel's
RENDERING for the claimed targets (the instructions that define the rounding chain, the position of the dependent-launch wait, the
byte every global load and store addresses, one text for every claimed target, the operand wiring of the compiled kernel, the
form inside the first 100 characters of the symbol) and the scale positions against the layout a block-scaled FP4 multiply
reads; on a cc 10.x device the bytes
against the independent references of ``row_quantize_fp4_decode_utils`` (the block scales bitwise, the two-code bracket), the scale
byte of EVERY finite bf16 block maximum under eleven global scales, the exact cells (codes bitwise), the equality with the NVFP4 form
of the add + RMSNorm + quantize step, blocks with a NaN or an infinity, output ownership per output, the bit-neutral arms, the wait
behind a predecessor that released early, the release beside a dependent launch, CUDA-graph replay, the warm path's allocation /
synchronization discipline, the ordering of a launch behind an entry-owned pair's initialization that is pending on another stream
(eager, captured, per row count, on a recycled block; its decision table on the host) and the profiler's record of a launch.

No cell arms a timer of its own: every device cell synchronizes under the suite's per-test watchdog (``CUDNN_TEST_TIMEOUT`` in
``test/python/conftest.py``).  The kernel has no barrier and no shared memory, so a trip of that watchdog on a launch is a hang of the
dependent-launch pair or of the caller's chain -- audit those; never a larger limit."""

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

from row_quantize_fp4_decode_utils import (  # noqa: E402  (the launch arms, the global scales, the operand builders, the references, the device gate)
    BF16,
    BF16_FINITE,
    BLOCK,
    BLOCKS,
    CONSTRUCTED_SCALE_BITS,
    E4M3,
    EXACT_ROWS,
    FP4_MAX_RCP_BITS,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    ODD_SCALE_BITS,
    PLAIN_SCALE_BITS,
    POWER_OF_TWO_SCALE_BITS,
    SCALE_BITS,
    U8,
    WIDTH,
    _device_or_skip,
    all_maxima_operand,
    assert_codes,
    assert_reference,
    code_domain_operand,
    exact_codes,
    exact_operands,
    make_entry,
    make_operands,
    maxima_scale_bytes,
    ptx_global_accesses,
    sass_wiring_facts,
    scale_byte_index,
    scale_bytes,
    scale_bytes_of_rows,
    scale_id,
    scale_layout_mask,
    scale_rows,
    scale_tensor,
    unpack_e2m1,
)

from cudnn.row_quantize.frost import row_quantize_fp4_decode as entry  # noqa: E402
from cudnn.row_quantize.frost.kernel import row_quantize_fp4_decode_config as kcfg  # noqa: E402
from launch_records import assert_facts_equal_the_launch  # noqa: E402

pytestmark = [pytest.mark.L0]

REPO = Path(__file__).resolve().parents[4]
ROWS = (1, 2, 3, 6, 8, 64, 129, 1024)
SYMBOL = "cudnn_kernel_frost_row_quantize_fp4_"
THREADS = WIDTH // BLOCK


def _cfg(arm=(None, None), width=WIDTH):
    return entry.make_cfg(width, pdl=arm[0], pdl_release=arm[1])


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_config_constants_and_tables():
    """The limits, the derived launch quantities, the layout arithmetic and the two EMPTY tables (no shared-memory buffer, no
    synchronization object).  ``MAX_ROWS x MAX_WIDTH`` stays inside the signed 32-bit element index: an invariant of the constants,
    which is why no call declines on it."""
    assert (kcfg.BLOCK, kcfg.WARP, kcfg.WIDTH_STEP, kcfg.MAX_WIDTH, kcfg.MAX_ROWS) == (16, 32, 512, 16384, 1024)
    assert kcfg.SERVED_WIDTH == (WIDTH,) and all(kcfg.validate_width(w) == w for w in kcfg.SERVED_WIDTH)
    assert kcfg.MAX_ROWS * kcfg.MAX_WIDTH <= kcfg.ROW_ADDRESS_LIMIT == 2**31 - 1
    assert (kcfg.SF_TILE_ROWS, kcfg.SF_ATOM_COLS, kcfg.SF_ATOM_BYTES, kcfg.SM_THREADS) == (128, 4, 512, 2048)
    cfg = _cfg()
    assert (cfg.THREADS, cfg.WARPS, cfg.SMEM_BYTES, cfg.SF_BLOCKS, cfg.SF_ATOMS, cfg.CTAS_PER_SM) == (320, 10, 0, 320, 80, 6)
    assert cfg.THREADS % kcfg.WARP == 0 and cfg.SF_BLOCKS % kcfg.SF_ATOM_COLS == 0, "full warps; whole scale atoms (no padded scale column)"
    for width in range(kcfg.WIDTH_STEP, kcfg.MAX_WIDTH + 1, kcfg.WIDTH_STEP):
        wide = kcfg.Cfg(WIDTH=width, PDL=False, PDL_RELEASE=False)
        assert wide.THREADS * kcfg.BLOCK == width and wide.THREADS <= 1024 and wide.THREADS % kcfg.WARP == 0 and wide.SF_ATOMS * 4 == wide.SF_BLOCKS
    assert kcfg.smem_table(cfg) == () and kcfg.barrier_ledger(cfg) == {}
    assert sum(kcfg.barrier_ledger(cfg).values()) == 0, "SUM(issuing lanes) == the count of every barrier: there is none"
    assert [kcfg.scale_rows(r) for r in (1, 127, 128, 129, 256, 257, 1024)] == [128, 128, 128, 256, 256, 384, 1024]
    assert entry.scale_rows is kcfg.scale_rows and all(scale_rows(r) == kcfg.scale_rows(r) for r in range(1, 1025))
    index = scale_byte_index(300)  # the layout written out in the test helpers, not through the package
    for row in (0, 1, 31, 32, 63, 64, 95, 96, 127, 128, 129, 255, 256, 299):
        assert [kcfg.scale_byte_offset(cfg, row, block) for block in range(BLOCKS)] == index[row].tolist(), row
    whole = sorted(kcfg.scale_byte_offset(cfg, row, block) for row in range(256) for block in range(BLOCKS))
    assert whole == list(range(256 * BLOCKS)), "two whole bands: the layout is a bijection onto the scale buffer's bytes"


def test_form_text_is_compact_unique_and_inside_the_first_100_characters_of_the_symbol():
    texts = {str(_cfg(arm)) for arm in LAUNCH_ARMS}
    assert texts == {"w5120_plain", "w5120_wait", "w5120_wait_release"}
    assert str(_cfg((True, True))) == "w5120_wait_release" and _cfg((True, True)) == _cfg((True, None))
    assert all(len(SYMBOL + text) <= 100 for text in texts)
    assert repr(_cfg()) == "Cfg(WIDTH=5120, PDL=False, PDL_RELEASE=False)"
    assert _cfg((True, False)).LAUNCH_ARM == "wait" and _cfg().LAUNCH_ARM == "plain"


def test_cfg_knobs_validate_by_name():
    """Construction declines, by type and by name, without torch: the width's structure, the served set, the launch knobs."""
    for width in ("5120", 5120.0, None, True):
        with pytest.raises(TypeError, match="width must be an int"):
            entry.make_cfg(width)
    for width in (0, -512, 511, 5121, 5136, 256, 16384 + 512):
        with pytest.raises(ValueError, match="width must be a multiple of 512 in 512 .. 16384"):
            entry.make_cfg(width)
    for width in (512, 2048, 4096, 5632, 16384):
        with pytest.raises(NotImplementedError, match=rf"serves width 5120 \(validated\); got width {width}"):
            entry.make_cfg(width)
    for knob in ("pdl", "pdl_release"):
        for bad in (1, 0, "yes", 1.0):
            with pytest.raises(TypeError, match=f"{knob} must be a bool or None"):
                entry.make_cfg(WIDTH, **{knob: bad})
    for pdl in (None, False):
        with pytest.raises(ValueError, match="pdl_release releases dependents after this launch's own wait and needs pdl=True"):
            entry.make_cfg(WIDTH, pdl=pdl, pdl_release=True)
    assert entry.make_cfg(WIDTH, pdl=False, pdl_release=False) == entry.make_cfg(WIDTH) == kcfg.Cfg(WIDTH, False, False)
    assert entry.make_cfg(WIDTH, pdl=True, pdl_release=False) == kcfg.Cfg(WIDTH, True, False)
    with pytest.raises(TypeError, match="cfg must be a row_quantize_fp4_decode_config.Cfg"):
        kcfg.validate_cfg(dict(WIDTH=WIDTH))
    for field in ("PDL", "PDL_RELEASE"):
        with pytest.raises(TypeError, match=f"{field.lower()} must be a bool"):
            kcfg.validate_cfg(replace(kcfg.Cfg(WIDTH, True, True), **{field: 1}))
    with pytest.raises(ValueError, match="needs pdl=True"):
        kcfg.validate_cfg(kcfg.Cfg(WIDTH, False, True))
    for rows in (1.0, "1", None, True):
        with pytest.raises(TypeError, match="rows must be an int"):
            kcfg.validate_rows(rows)
    with pytest.raises(ValueError, match="rows must be at least 1, got 0"):
        kcfg.validate_rows(0)
    with pytest.raises(NotImplementedError, match="serves at most 1024 rows per call; got 1025"):
        kcfg.validate_rows(1025)
    assert kcfg.validate_rows(1) == 1 and kcfg.validate_rows(1024) == 1024


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the typed declines of a call (CPU tensors; nothing is compiled or launched)
# ---------------------------------------------------------------------------------------------------------------------------
def _host_operands(rows=3, own=True, extra_rows=0):
    """Served CPU operands of ``rows`` rows: ``x``, ``scale`` and (``own``) caller-owned ``codes`` and ``scales``."""
    ops = dict(x=torch.zeros(rows, WIDTH, dtype=BF16), scale=torch.ones(1, dtype=torch.float32))
    if own:
        ops["codes"] = torch.zeros(rows + extra_rows, WIDTH // 2, dtype=U8)
        ops["scales"] = torch.zeros(scale_rows(rows) + (128 if extra_rows else 0), BLOCKS, dtype=E4M3)
    return ops


def _check(ops, **kw):
    return entry.check_support(_cfg(), ops["x"], ops["scale"], ops.get("codes"), ops.get("scales"), **kw)


def test_served_operands_pass_the_host_checks():
    for rows in (1, 3, 128, 129, 1024):
        for own in (True, False):
            for extra in (0, 5):
                assert _check(_host_operands(rows, own, extra), device_cc=(10, 0)) == rows
    ops = _host_operands()
    assert _check(dict(ops, scale=torch.ones((), dtype=torch.float32)), device_cc=(10, 3)) == 3, "a 0-d scale"
    assert entry.check_binding(_cfg((True, None)), ops["x"], ops["scale"]) == 3


def _misaligned(dtype, shape, elements=1):
    """A contiguous CPU tensor of ``shape`` whose first byte sits ``elements`` ELEMENTS past a 16-byte aligned address."""
    n = 1
    for d in shape:
        n *= d
    backing = U8 if dtype == E4M3 else dtype
    t = torch.zeros(n + 64, dtype=backing)
    start = ((-t.data_ptr()) % 16) // t.element_size() + elements
    t = t[start : start + n].view(dtype).view(*shape)
    assert t.data_ptr() % 16 == elements * t.element_size() and t.is_contiguous()
    return t


def _misaligned_scale():
    """One fp32 element at an address that is not a multiple of 4 (a byte buffer read at an odd offset)."""
    buf = bytearray(64)
    base = torch.frombuffer(buf, dtype=U8).data_ptr()
    return torch.frombuffer(buf, dtype=torch.float32, count=1, offset=1 + (-base) % 4)  # address % 4 == 1


def _overlapping():
    """``(x, codes)`` on ONE byte buffer: ``codes`` starts inside the bytes of ``x``."""
    rows = 3
    raw = torch.zeros(rows * WIDTH * 2 + rows * (WIDTH // 2), dtype=U8)
    x = raw[: rows * WIDTH * 2].view(BF16).reshape(rows, WIDTH)
    codes = raw[1024 : 1024 + rows * (WIDTH // 2)].reshape(rows, WIDTH // 2)
    return x, codes


def _mutations():
    """``{name: (operands, exception type, message pattern)}``: ONE change per decline of a call."""
    base = _host_operands()
    wide = torch.zeros(3, 2 * WIDTH, dtype=BF16)
    over_x, over_codes = _overlapping()
    raw = torch.zeros(128 * BLOCKS + 3 * (WIDTH // 2), dtype=U8)  # ONE byte buffer under a scale band and, from byte 160 on, three code rows
    band = raw[: 128 * BLOCKS].view(E4M3).reshape(128, BLOCKS)
    cases = {
        "x-missing": (dict(base, x=None), TypeError, "x must be a torch.Tensor, got NoneType"),
        "scale-missing": (dict(base, scale=None), TypeError, "scale must be a torch.Tensor, got NoneType"),
        "x-not-a-tensor": (dict(base, x=[[0.0] * WIDTH]), TypeError, "x must be a torch.Tensor, got list"),
        "scale-not-a-tensor": (dict(base, scale=1.0), TypeError, "scale must be a torch.Tensor, got float"),
        "codes-not-a-tensor": (dict(base, codes=bytearray(8)), TypeError, "codes must be a torch.Tensor, got bytearray"),
        "x-fp32": (dict(base, x=base["x"].float()), TypeError, r"x must be torch.bfloat16 \(kernel-native; callers convert\), got torch.float32"),
        "x-fp16": (dict(base, x=base["x"].half()), TypeError, "x must be torch.bfloat16"),
        "scale-fp64": (dict(base, scale=base["scale"].double()), TypeError, "scale must be torch.float32"),
        "scale-bf16": (dict(base, scale=base["scale"].bfloat16()), TypeError, "scale must be torch.float32"),
        "codes-int8": (dict(base, codes=base["codes"].view(torch.int8)), TypeError, "codes must be torch.uint8"),
        "codes-e4m3": (dict(base, codes=base["codes"].view(E4M3)), TypeError, "codes must be torch.uint8"),
        "scales-uint8": (dict(base, scales=base["scales"].view(U8)), TypeError, "scales must be torch.float8_e4m3fn"),
        "scales-e5m2": (dict(base, scales=base["scales"].view(torch.float8_e5m2)), TypeError, "scales must be torch.float8_e4m3fn"),
        "x-1d": (dict(base, x=base["x"].reshape(-1)), ValueError, r"x must be \[rows, 5120\] with at least one row"),
        "x-3d": (dict(base, x=base["x"].reshape(1, 3, WIDTH)), ValueError, r"x must be \[rows, 5120\]"),
        "x-other-width": (dict(base, x=torch.zeros(3, 4096, dtype=BF16)), ValueError, r"x must be \[rows, 5120\].*got \(3, 4096\)"),
        "zero-rows": (dict(base, x=base["x"][:0]), ValueError, "at least one row"),
        "rows-above-the-ceiling": (_host_operands(kcfg.MAX_ROWS + 1, own=False), NotImplementedError, "serves at most 1024 rows per call; got 1025"),
        "scale-two-elements": (dict(base, scale=torch.ones(2)), ValueError, r"scale must be one fp32 element \(0-d or \[1\]\); got \(2,\)"),
        "scale-empty": (dict(base, scale=torch.ones(0)), ValueError, "scale must be one fp32 element"),
        "codes-too-few-rows": (dict(base, codes=base["codes"][:2]), ValueError, r"codes must be \[>= 3, 2560\]; got \(2, 2560\)"),
        "codes-full-width": (dict(base, codes=torch.zeros(3, WIDTH, dtype=U8)), ValueError, r"codes must be \[>= 3, 2560\]"),
        "codes-1d": (dict(base, codes=base["codes"].reshape(-1)), ValueError, r"codes must be \[>= 3, 2560\]"),
        "scales-row-major-height": (dict(base, scales=torch.zeros(3, BLOCKS, dtype=E4M3)), ValueError, r"scales must be \[>= 128, 320\]; got \(3, 320\)"),
        "scales-other-width": (dict(base, scales=torch.zeros(128, BLOCKS // 2, dtype=E4M3)), ValueError, r"scales must be \[>= 128, 320\]"),
        "scales-one-band-for-129-rows": (
            dict(_host_operands(129, own=False), scales=torch.zeros(128, BLOCKS, dtype=E4M3)),
            ValueError,
            r"scales must be \[>= 256, 320\]",
        ),
        "x-not-contiguous": (dict(base, x=wide[:, ::2]), NotImplementedError, r"needs x contiguous .*got strides \(10240, 2\)"),
        "codes-not-contiguous": (dict(base, codes=torch.zeros(3, WIDTH, dtype=U8)[:, ::2]), NotImplementedError, "needs codes contiguous"),
        "scales-not-contiguous": (dict(base, scales=torch.zeros(128, 2 * BLOCKS, dtype=E4M3)[:, ::2]), NotImplementedError, "needs scales contiguous"),
        "x-misaligned": (
            dict(base, x=_misaligned(BF16, (3, WIDTH), 4)),
            ValueError,
            r"x must start at a 16-byte aligned address for its 128-bit loads, got data_ptr % 16 = 8",
        ),
        "codes-misaligned": (dict(base, codes=_misaligned(U8, (3, WIDTH // 2), 8)), ValueError, "codes must start at a 16-byte aligned address"),
        "scales-misaligned": (dict(base, scales=_misaligned(E4M3, (128, BLOCKS), 4)), ValueError, "scales must start at a 16-byte aligned address"),
        "scale-misaligned": (dict(base, scale=_misaligned_scale()), ValueError, r"scale must start at a 4-byte aligned address for its 32-bit load"),
        "codes-overlaps-x": (dict(base, x=over_x, codes=over_codes), ValueError, "codes must not overlap x"),
        "scales-overlaps-codes": (
            dict(base, codes=raw[160 : 160 + 3 * (WIDTH // 2)].reshape(3, WIDTH // 2), scales=band),
            ValueError,
            "codes must not overlap scales",
        ),
        "scales-overlaps-scale": (
            dict(base, scale=raw[64:68].view(torch.float32), scales=band),
            ValueError,
            "scales must not overlap scale",
        ),
    }
    return cases


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_typed_declines_name_the_operand(name):
    """One mutation per decline: the type, and a message that names the operand and what it must be.  Every decline is raised by
    host metadata arithmetic, before any DSL import, compile or launch."""
    ops, error, pattern = _mutations()[name]
    with pytest.raises(error, match=pattern):
        _check(ops, device_cc=(10, 0))
    with pytest.raises(error, match=pattern):
        entry.check_binding(_cfg(), ops["x"], ops["scale"], ops.get("codes"), ops.get("scales"))


def test_binding_check_requires_every_operand_on_the_given_device():
    ops = _host_operands()
    with pytest.raises(ValueError, match=r"every operand must live on one device, the entry's \(cuda:0\); x is on cpu"):
        entry.check_binding(_cfg(), ops["x"], ops["scale"], device=torch.device("cuda", 0))
    with pytest.raises(ValueError, match=r"every operand must live on one device, x's \(cpu\); scale is on meta"):
        entry.check_binding(_cfg(), ops["x"], torch.ones(1, device="meta"))  # no device given: the operands share x's
    assert entry.check_binding(_cfg(), ops["x"], ops["scale"], ops["codes"], ops["scales"], device="cpu") == 3


def test_declines_an_operand_that_overlaps_an_entry_owned_output():
    """An output the call does not pass is the entry's own buffer of the row count, and a tensor an earlier call returned can come
    back as an operand: a ``scale`` that lives inside the entry-owned ``codes`` (or ``scales``) is declined by name BEFORE the compile
    and the launch; with that output passed by the caller the same binding is checked like any other.  A CPU entry: nothing compiles."""
    k = make_entry("cpu")
    ops = _host_operands(own=False)
    owned = k.outputs(3)
    assert tuple(owned["codes"].shape) == (3, WIDTH // 2) and tuple(owned["scales"].shape) == (128, BLOCKS)
    assert bool((owned["scales"].view(U8) == 0).all()), "an entry-owned scale buffer is zero-filled"
    inside_codes = owned["codes"].reshape(-1)[:4].view(torch.float32)
    with pytest.raises(ValueError, match="the entry-owned codes must not overlap scale .*pass a caller-owned codes"):
        k(ops["x"], inside_codes)
    inside_scales = owned["scales"].view(U8).reshape(-1)[512:516].view(torch.float32)
    with pytest.raises(ValueError, match="the entry-owned scales must not overlap scale"):
        k(ops["x"], inside_scales)
    with pytest.raises(ValueError, match="the entry-owned scales must not overlap codes"):
        k(ops["x"], ops["scale"], codes=owned["scales"].view(U8).reshape(-1)[: 3 * (WIDTH // 2)].reshape(3, WIDTH // 2))
    with pytest.raises(NotImplementedError, match="serves CUDA devices only"):  # the checks passed: the next gate is the device's
        k(ops["x"], ops["scale"])
    assert not k.compiled


def test_compile_and_outputs_decline_a_bad_row_count():
    k = make_entry("cpu")
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


def test_initialization_order_is_decided_per_pair_and_waits_for_one_event_at_most(monkeypatch):
    """What a call does for an entry-owned pair whose initialization may be pending, on stand-ins for the event, the streams and
    the driver (no device; the device cells run the schedules).  Complete: the record is dropped.  Pending on the launch stream:
    nothing is enqueued and the record stays.  Pending on another stream: THAT stream waits for the event and the record stays
    for the next stream.  Pending under a capture: the host waits for this one event, without querying it first, and the record
    is dropped.  No branch synchronizes a stream or the device; every read of the event runs with the thread's capture check
    relaxed, and the check is restored also when the read raises.  A pair of another row count is never touched, and an entry
    on a device without streams records nothing."""
    calls, capturing = [], [False]

    class Event:
        def __init__(self, complete, fails=False):
            self.complete, self.fails = complete, fails

        def query(self):
            calls.append("query")
            if self.fails:
                raise RuntimeError("the event's query failed")
            return self.complete

        def synchronize(self):
            calls.append("synchronize")
            self.complete = True

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait_event", self.name, event))

        def synchronize(self):
            raise AssertionError("a stream was synchronized")

        wait_stream = synchronize

    class Driver:
        class CUstreamCaptureMode:
            CU_STREAM_CAPTURE_MODE_RELAXED = "relaxed"

        mode, fails = "global", False

        @classmethod
        def cuThreadExchangeStreamCaptureMode(cls, mode):
            if cls.fails:
                return 1, cls.mode
            previous, cls.mode = cls.mode, mode
            calls.append(("capture mode", mode))
            return 0, previous

    def no_device_synchronization(*args, **kwargs):
        raise AssertionError("the device was synchronized")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    monkeypatch.setattr(torch.cuda, "synchronize", no_device_synchronization)
    k = make_entry("cpu")
    k.outputs(3), k.outputs(4)
    assert k._init == {}, "an entry on a device without streams records no initialization"
    init, other, untouched = Stream("init"), Stream("other"), (Event(False), Stream("of another row count"))

    def decide(event, stream, captured=False):
        calls.clear()
        capturing[0] = captured
        k._init = {3: (event, init), 4: untouched}
        k._order_behind_init(3, stream, Driver)
        assert calls[0] == ("capture mode", "relaxed") and calls[-1] == ("capture mode", "global") and Driver.mode == "global"
        assert k._init[4] is untouched
        return calls[1:-1], 3 in k._init

    pending = Event(False)
    assert decide(pending, init) == (["query"], True)
    assert decide(pending, other) == (["query", ("wait_event", "other", pending)], True)
    assert decide(pending, other, captured=True) == (["synchronize"], False)
    assert decide(Event(False), init, captured=True) == (["synchronize"], False)
    assert decide(Event(True), other) == (["query"], False)
    k._init = {3: (Event(False, fails=True), init)}
    with pytest.raises(RuntimeError, match="the event's query failed"):
        k._order_behind_init(3, other, Driver)
    assert Driver.mode == "global", "the capture check is restored behind a failed read"
    Driver.fails = True
    calls.clear()
    with pytest.raises(RuntimeError, match="cuThreadExchangeStreamCaptureMode failed"):
        k._order_behind_init(3, other, Driver)
    assert not calls and 3 in k._init, "nothing is read and nothing is dropped when the capture check cannot be relaxed"


def test_declines_an_unserved_compute_capability_by_name():
    ops = _host_operands()
    for cc in ((9, 0), (8, 0), (8, 9), (12, 0), (12, 1), (11, 0)):
        with pytest.raises(NotImplementedError, match=rf"serves compute capability 10.x \(SM100-class\); this device is {cc[0]}.{cc[1]}"):
            _check(ops, device_cc=cc)
        assert not entry.cc_supported(cc)
    assert all(entry.cc_supported(cc) for cc in ((10, 0), (10, 3), (10, 7))) and not entry.cc_supported(None)
    assert entry.SUPPORTED_CC_RANGES == ((100, 109),) and entry.DSL_FLOOR == (4, 8, 0) and entry.SM107_MIN_COMPILER == (13, 4)


def test_declines_cpu_operands_when_no_capability_is_given():
    with pytest.raises(NotImplementedError, match="serves CUDA tensors only; the operands are on cpu"):
        _check(_host_operands())


def test_compile_declines_an_unserved_part_on_the_entrys_own_path(monkeypatch):
    """The capability decline a CALLER reaches: an entry's calls run the binding checks, and the part is gated in ``compile()``.  On
    MODELLED part facts (an entry for ``cuda:0`` with the device's compute capability patched: no device is touched) ``compile()``,
    ``compile(rows)`` and the first call on an unserved part are the typed decline by name, in front of the DSL gates, and nothing is
    compiled or allocated.  A served capability passes this gate and reaches the next one (the DSL, modelled as missing)."""
    from cudnn.frost import buffers

    ops = _host_operands(own=False)
    for cc in ((12, 0), (9, 0), (8, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        k = make_entry("cuda:0")
        text = rf"serves compute capability 10\.x \(SM100-class\); this device is {cc[0]}\.{cc[1]}"
        for rows in (None, 3):
            with pytest.raises(NotImplementedError, match=text):
                k.compile(rows)
        with monkeypatch.context() as bound:  # the first call compiles: with the binding checks stood in for, it reaches the same gate
            bound.setattr(entry, "check_binding", lambda *args, **kwargs: 3)
            with pytest.raises(NotImplementedError, match=text):
                k(ops["x"], ops["scale"])
        assert not k.compiled and not k.buffers and not k._buffer_views, "a declined compile builds and allocates nothing"
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    for cc in ((10, 0), (10, 3), (10, 7)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        k = make_entry("cuda:0")
        with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8.0; the CuTe DSL is not installed"):
            k.compile()
        assert not k.compiled and not k.buffers


def test_compile_key_is_plan_time_only():
    """One key per entry whatever the row count; it changes with each field of the config."""
    k = make_entry("cpu", (True, None))
    assert k.key() == (k.cfg,) and all(k.cfg_for(rows) is k.cfg for rows in (1, 17, kcfg.MAX_ROWS))
    keys = {k.key()}
    for change in (dict(PDL_RELEASE=False), dict(PDL=False, PDL_RELEASE=False), dict(WIDTH=2048)):
        keys.add((replace(k.cfg, **change),))
    assert len(keys) == 4
    with pytest.raises(NotImplementedError, match="at most 1024 rows"):
        k.cfg_for(kcfg.MAX_ROWS + 1)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the launch facts (cudnn.frost.launch_facts) -- the symbol of the compiled form, the form as plain data, the launch shape
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_facts_equal_the_configuration_arithmetic_at_every_row_count(arm, arm_id):
    """EVERY served row count, 1 .. 1024, per launch arm: one thread block per row of ``width / 16`` threads and NO shared memory;
    the form is the config's fields -- nothing else, because the compile key is the config alone."""
    k = make_entry("cpu", arm)
    want_form = asdict(k.cfg)
    assert want_form == dict(WIDTH=WIDTH, PDL=bool(arm[0]), PDL_RELEASE=arm_id == "wait-release")
    for rows in range(1, kcfg.MAX_ROWS + 1):
        shape = k.launch_shape_for(rows)
        assert shape == ((rows, 1, 1), (THREADS, 1, 1), 0), (rows, shape)
        assert all(type(v) is int for v in (*shape.grid, *shape.block, shape.shared_memory)), shape
        assert k.form_for(rows) == want_form, rows
    assert k.kernel_symbols_for(1) == () == k.kernel_symbols_for(kcfg.MAX_ROWS), "nothing is compiled: no symbol"


def test_launch_facts_are_plain_fresh_and_decline_an_unserved_row_count():
    k = make_entry("cpu", (True, None))
    form = k.form_for(7)
    assert json.loads(json.dumps(form)) == form and type(form) is dict
    form["WIDTH"] = "edited"
    assert k.form_for(7)["WIDTH"] == WIDTH and k.cfg.WIDTH == WIDTH
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
    k.compiled[(replace(k.cfg, PDL=True),)] = _CompiledStandIn(SYMBOL + "other_0")
    assert k.kernel_symbols_for(1) == (), "a form the entry does not run is not named"
    k.compiled[k.key()] = _CompiledStandIn(SYMBOL + "A_0")
    assert all(k.kernel_symbols_for(rows) == (SYMBOL + "A_0",) for rows in (1, 2, 160, kcfg.MAX_ROWS))
    k.compiled[k.key()] = object()
    with pytest.raises(NotImplementedError, match=r"the FP4 row quantization decode step: the installed CuTe DSL .* does not expose the kernel symbols"):
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
    cc 10.7 is declined naming the target and the DSL found, cc 10.3 is admitted.  The version gate runs first: the target probe
    (the first gate that imports the DSL) is never reached under a too-old DSL."""
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
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (13, 3))
    assert "compiles with CUDA 13.3" in entry._dsl_error((10, 7))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: exports, the import boundary, where the kernel's instructions come from
# ---------------------------------------------------------------------------------------------------------------------------
def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.RowQuantizeFp4Decode is entry.RowQuantizeFp4Decode
    family = importlib.import_module("cudnn.row_quantize")
    frost = importlib.import_module("cudnn.row_quantize.frost")
    assert family.RowQuantizeFp4Decode is entry.RowQuantizeFp4Decode
    assert frost.RowQuantizeFp4Decode is entry.RowQuantizeFp4Decode and frost.Cfg is kcfg.Cfg
    assert "RowQuantizeFp4Decode" in cudnn._LAZY_OPTIONAL_IMPORTS and "RowQuantizeFp4Decode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS
    assert entry.NAME == kcfg.NAME and set(entry.__all__) >= {"RowQuantizeFp4Decode", "check_binding", "check_support", "cc_supported", "make_cfg"}
    for module in (family, frost):
        with pytest.raises(AttributeError):
            module.NoSuchName


def test_import_cudnn_does_not_import_the_kernel_or_a_heavy_module():
    """The lazy boundary, in fresh interpreters.  The heavy-module list is the one the package's own boundary test holds
    (``core/test_import_boundaries.py::_HEAVY``), read from that module's source, and what a statement imports is a DELTA against an
    empty interpreter (a distribution can inject a package at startup through a ``.pth`` file).  Importing cudnn loads none of the
    family's modules and no heavy module; the family packages, the config module and the host entry load no heavy module and not the
    kernel module; constructing a host entry (its declines included) still loads neither the DSL nor the kernel module."""
    boundary = ast.parse((REPO / "test" / "python" / "core" / "test_import_boundaries.py").read_text())
    heavy = [ast.literal_eval(n.value) for n in boundary.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_HEAVY"]
    assert len(heavy) == 1 and {"torch", "cutlass"} <= set(heavy[0]), heavy
    heavy = tuple(heavy[0])

    def modules(code):
        probe = code + "\nimport json, sys; print('@@' + json.dumps(sorted(sys.modules)))"
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
        return set(json.loads(next(ln for ln in out.splitlines() if ln.startswith("@@"))[2:]))

    def added(code):
        return modules(code) - modules("")

    def unwanted(names):
        return sorted(m for m in names if m.split(".")[0] in heavy or m.endswith("row_quantize_fp4_decode_f16"))

    by_cudnn = added("import cudnn")
    assert not unwanted(by_cudnn) and not [m for m in by_cudnn if m.startswith("cudnn.row_quantize")], sorted(by_cudnn)
    family = "cudnn.row_quantize, cudnn.row_quantize.frost, cudnn.row_quantize.frost.kernel.row_quantize_fp4_decode_config, cudnn.row_quantize.frost.row_quantize_fp4_decode"
    assert not unwanted(added(f"import {family}"))
    construct = (
        "import cudnn\n"
        "k = cudnn.RowQuantizeFp4Decode('cpu', 5120, pdl=True)\n"
        "try:\n"
        "    cudnn.RowQuantizeFp4Decode('cpu', 5120, pdl_release=True)\n"
        "except ValueError:\n"
        "    pass\n"
    )
    assert not [m for m in added(construct) if m.split(".")[0] == "cutlass" or m.endswith("row_quantize_fp4_decode_f16")]


def test_kernel_takes_its_instructions_from_the_tile_library():
    """One definition per helper, read from the source (no DSL import): the kernel module takes the block maximum and the
    block-scale chain from ``cudnn.frost.tile_dsl.pointwise``, the row load and the 4-bit store from ``tile_dsl.rowwise``, the
    layout rule from ``tile_dsl.sf_layout`` and the dependent-launch pair from ``tile_dsl.barrier``; it spells no instruction inline,
    allocates no shared memory, holds no barrier and no shuffle, and names no address space (a kernel module takes that name from
    ``cudnn._cutlass_compat``, never off ``cute``).  Its ONE subscript store is the scale byte, indexed by the layout rule of
    ``(row, block)`` with the config's atoms per band (the rendering cells evaluate the compiled address)."""
    source = (REPO / "python" / "cudnn" / "row_quantize" / "frost" / "kernel" / "row_quantize_fp4_decode_f16.py").read_text()
    tree = ast.parse(source)

    def imported(module):
        return {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == module for alias in node.names}

    assert imported("cudnn.frost.tile_dsl.pointwise") == {"abs_max_tree", "nvfp4_block_scale_global"}
    assert imported("cudnn.frost.tile_dsl.rowwise") == {"ROW_VEC", "row_load8", "row_store16_e2m1"}
    assert imported("cudnn.frost.tile_dsl.sf_layout") == {"sf_band_offset"}
    assert imported("cudnn.frost.tile_dsl.barrier") == {"launch_dependent_grids", "wait_on_dependent_grids"}
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert defined == {"frost_row_quantize_fp4", "launch"}, defined
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for absent in ("inline_ptx", "SmemAllocator", "barrier", "shuffle_sync_bfly", "AddressSpace", "block_reduce"):
        assert absent not in names, f"the kernel module names {absent}"
    stores = [ast.unparse(t) for node in ast.walk(tree) if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Subscript)]
    assert stores == ["sf[sf_band_offset(row, tid, cfg.SF_ATOMS)]"], f"the kernel's subscript stores: {stores}"
    code = "\n".join(ln for ln in source.splitlines() if not ln.lstrip().startswith("#"))
    assert "cute.AddressSpace" not in code.split('"""')[-1], "the kernel's code reads AddressSpace off cute"


def test_reference_separates_the_scale_associations():
    """WHY the reference's scale list holds three constructed scales.  The scale byte is a function of (block maximum, global scale)
    alone, and a bf16 block maximum has 32,640 finite non-negative patterns.  Counted over all of them, per global scale: how many
    maxima get ANOTHER scale byte when the product is formed as ``(amax x gs) x fp32(1/6)``, as ``amax x (gs x fp32(1/6))`` or as
    ``(amax / 6) x gs`` instead of the definition's ``gs x (amax x fp32(1/6))``, and when the constant is one fp32 unit lower or
    higher.  Under each of the eight plain scales: NONE for the three associations -- reference cells under those scales cannot fail
    for a kernel that associates differently -- and under the two that are not powers of two none for the constant either.  Under the
    three constructed scales tens of maxima each.  Whoever removes those three from the list is told here what the reference cells
    then no longer see."""
    want_plain = {bits: ([0, 0, 0], [63, 63] if bits in POWER_OF_TWO_SCALE_BITS else [0, 0]) for bits in PLAIN_SCALE_BITS}
    want_constructed = {0x41BFFFFF: ([45, 30, 30], [30, 79]), 0x4219999A: ([47, 16, 17], [64, 16]), 0x41DB6DB6: ([31, 46, 16], [46, 64])}
    assert set(want_constructed) == set(CONSTRUCTED_SCALE_BITS) and SCALE_BITS == PLAIN_SCALE_BITS + CONSTRUCTED_SCALE_BITS
    assert set(POWER_OF_TWO_SCALE_BITS) < set(PLAIN_SCALE_BITS) and ODD_SCALE_BITS not in POWER_OF_TWO_SCALE_BITS
    for bits, (associations, constant) in {**want_plain, **want_constructed}.items():
        definition = maxima_scale_bytes(bits)
        assert definition.numel() == BF16_FINITE == 32640 and int(definition[0]) == 0
        got = [int((maxima_scale_bytes(bits, chain) != definition).sum()) for chain in ("amax_gs_first", "gs_sixth_first", "divide_by_six")]
        off = [int((maxima_scale_bytes(bits, sixth_bits=FP4_MAX_RCP_BITS + step) != definition).sum()) for step in (-1, 1)]
        assert got == associations and off == constant, f"{scale_id(bits)}: associations {got} (want {associations}), constant {off} (want {constant})"
        assert bool((definition[1:] >= definition[:-1]).all()), "the scale byte is monotone in the block maximum"


def test_operand_builders_enumerate_what_they_state():
    """The two enumerating operands, on the host.  ``all_maxima_operand``: 102 rows, block ``k`` has the bf16 pattern ``k`` as its
    maximum magnitude, at element ``k % 16``, for every finite pattern.  ``code_domain_operand`` under the non-power-of-two plain
    scale: every scale byte 0 .. 0x7E occurs; every block's maximum is its element 0; per scale byte the elements 1 .. 15 run
    through every positive bf16 pattern up to that maximum; the second half of the rows is the first with those elements negated."""
    x = all_maxima_operand()
    assert tuple(x.shape) == (102, WIDTH) and x.dtype == BF16
    magnitude = (x.view(torch.int16).to(torch.int32) & 0x7FFF).reshape(-1, BLOCK)
    k = torch.arange(BF16_FINITE)
    assert torch.equal(magnitude.amax(dim=1), k.to(torch.int32)) and torch.equal(magnitude[k, k % BLOCK], k.to(torch.int32))
    assert bool(torch.isfinite(x.float()).all())
    scale = scale_tensor(ODD_SCALE_BITS, "cpu")
    assert torch.equal(scale_bytes(x, scale).reshape(-1), maxima_scale_bytes(ODD_SCALE_BITS))
    domain = code_domain_operand(ODD_SCALE_BITS)
    rows = domain.shape[0]
    assert rows == 854 and rows % 2 == 0 and rows <= kcfg.MAX_ROWS, rows
    patterns = (domain.view(torch.int16).to(torch.int32) & 0xFFFF).reshape(rows, BLOCKS, BLOCK)
    half = rows // 2
    assert torch.equal(patterns[:half, :, 0], patterns[half:, :, 0]) and bool((patterns[:half] < 0x8000).all())
    live = patterns[:half, :, 1:] != 0
    assert torch.equal(patterns[half:, :, 1:], patterns[:half, :, 1:] | 0x8000)
    assert bool((patterns[:half, :, 1:] <= patterns[:half, :, :1]).all()), "element 0 is the block's maximum"
    block_bytes = scale_bytes(domain[:half], scale)
    assert set(block_bytes.unique().tolist()) == set(range(0x7F)), "every scale byte occurs"
    by_pattern = maxima_scale_bytes(ODD_SCALE_BITS)
    for byte in (0, 1, 0x40, 0x7E):
        hi = int((by_pattern == byte).nonzero().max())
        of_byte = patterns[:half][block_bytes == byte]
        of_byte = of_byte[of_byte[:, 0] != 0]  # without the zero blocks that fill the last row
        assert bool((of_byte[:, 0] == hi).all())
        assert set(of_byte[:, 1:].reshape(-1).tolist()) == set(range(1, hi + 1)), f"scale byte {byte:#x}: every positive pattern up to {hi:#x}"
    assert int(live.sum()) >= 2_045_270, "the enumerated (scale byte, element) pairs of one sign"


GOLDEN_CASES = ("random-rows1", "random-rows8", "structured-rows130", "code-domain-rows854")


def test_docs_rows_exist_for_the_api():
    """The determinism status page carries the API's ONE row -- SM90 unsupported (declined by type); SM100/103 split by the evidence;
    SM107 unverified -- and its bullet; the goldens README its suite row and section; the operation page, its navigation entry, the
    overview and llms.txt links and the pointer from the block-scale conversion page exist; the golden cases are committed for at
    least one runtime."""
    det = (REPO / "docs" / "operations" / "determinism.md").read_text()
    row = [ln for ln in det.splitlines() if ln.startswith("| `RowQuantizeFp4Decode`")]
    assert len(row) == 1, "one RowQuantizeFp4Decode row on the status page"
    cells = [c.strip() for c in row[0].strip("|").split("|")]
    assert cells[1] == "fprop (inference step)" and cells[2] == "BF16 in; FP4 E2M1 codes with E4M3 block scales out", cells
    assert cells[3] == "Unsupported" and cells[4] == "SM100: Supported; SM103: Unverified" and cells[5] == "Unverified", cells
    assert "- FP4 row quantization decode step (`RowQuantizeFp4Decode`):" in det and "No backward entry point exists." in det
    readme = (REPO / "test" / "python" / "determinism_goldens" / "README.md").read_text()
    assert "RowQuantizeFp4Decode" in readme and "row_quantize/frost/test_row_quantize_fp4_decode_determinism.py" in readme
    assert "### FP4 row quantization decode step:" in readme
    page = REPO / "docs" / "operations" / "block_scaling" / "row_quantize_fp4_decode.md"
    assert page.exists() and "cudnn.RowQuantizeFp4Decode" in page.read_text()
    assert "../../operations/block_scaling/row_quantize_fp4_decode.md" in (REPO / "docs" / "fern" / "versions" / "latest.yml").read_text()
    overview = (REPO / "docs" / "operations" / "overview.md").read_text()
    assert overview.count("/operations/BlockScaling/row_quantize_fp4_decode") == 2, "the decode-step list and the block-scaling list"
    assert "docs/operations/block_scaling/row_quantize_fp4_decode.md" in (REPO / "llms.txt").read_text()
    assert "row_quantize_fp4_decode" in (REPO / "docs" / "operations" / "block_scaling" / "nvfp4_block_scale_conversion.md").read_text()
    goldens = {p.name for p in (REPO / "test" / "python" / "determinism_goldens").glob("*/row-quantize-fp4-decode-*.json")}
    assert goldens == {f"row-quantize-fp4-decode-w{WIDTH}-{case}.json" for case in GOLDEN_CASES}, goldens


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the rendering (the instructions that define the rounding chain), per target
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
    from cudnn.row_quantize.frost import row_quantize_fp4_decode as entry
    from cudnn.row_quantize.frost.kernel import row_quantize_fp4_decode_f16 as mod

    print("COMPILER", int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    half = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.BFloat16, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    byte = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.Uint8, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    one = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(1,), stride=(1,), assumed_align=4)
    for spec in sys.argv[3:]:
        before = set(os.listdir(dump))
        parts = spec.split("|")
        pdl, release = (None if p == "None" else p == "True" for p in parts[1:3])
        cfg = entry.make_cfg(5120, pdl=pdl, pdl_release=release)
        try:
            if parts[0] == "bind":  # the launch arguments the ENTRY builds from CUDA tensors of that many rows
                import torch

                rows = int(parts[3])
                dev = torch.device("cuda", torch.cuda.current_device())
                k = entry.RowQuantizeFp4Decode(dev, 5120, pdl=pdl, pdl_release=release)
                views = k._owned_views(rows)
                args = [entry._flat(torch.zeros(rows, 5120, dtype=torch.bfloat16, device=dev)), entry._flat(torch.ones(1, dtype=torch.float32, device=dev)),
                        views["codes"], views["scales"]]
                cute.compile(mod.launch, cfg, *args, cutlass.Int32(rows), make_fake_stream())
            else:
                cute.compile(mod.launch, cfg, half(), one(), byte(), byte(), cutlass.Int32(2), make_fake_stream())
        except Exception as error:  # a target the installed compiler flavour does not assemble: reported, the caller decides
            print("REJECTED", spec, type(error).__name__, " ".join(str(error).split())[:400])
            continue
        for name in sorted(set(os.listdir(dump)) - before):
            print("KEPT", spec, name)
    print("COMPILED", arch)
    """)

_UNSERVED = "Unsupported GPU architecture"  # the DSL's message when its compiler flavour has no such target


def _spec(arm, kind="fake", rows=None):
    return "|".join(str(v) for v in (kind, arm[0], arm[1]) + (() if rows is None else (rows,)))


def _render(tmp_path, arch, specs):
    """Run the probe for ``arch`` in a fresh process (the DSL reads its dump settings once, at the first import); returns the
    compiler's ``(major, minor)``, ``{spec: [kept file paths]}`` and ``{spec: rejection text}``."""
    probe = tmp_path / "render_probe.py"  # a FILE, not ``python -c``: the DSL preprocessor re-reads a kernel's source through inspect
    probe.write_text(_RENDER_PROBE)
    dump = tmp_path / "dump"
    dump.mkdir()
    proc = subprocess.run([sys.executable, str(probe), str(dump), arch, *specs], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"trace-compile for {arch} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED {arch}" in lines, proc.stdout[-2000:]
    compiler = tuple(int(v) for v in next(ln for ln in lines if ln.startswith("COMPILER")).split()[1:3])
    kept, rejected = {}, {}
    for ln in lines:
        if ln.startswith("KEPT "):
            _, spec, name = ln.split(" ", 2)
            kept.setdefault(spec, []).append(str(dump / name))
        if ln.startswith("REJECTED "):
            _, spec, text = ln.split(" ", 2)
            rejected[spec] = text
    return compiler, kept, rejected


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


def _statements(ptx):
    """The statements of a PTX text from its entry on as token lists: comments dropped, a leading predicate (``@%p1``) removed.
    Inline blocks keep several statements on one line, so the split is on ``;`` and on line ends; braces are separators."""
    out = []
    for line in ptx[ptx.index(".entry") :].splitlines():
        line = line.split("//")[0]
        for stmt in line.replace("{", ";").replace("}", ";").split(";"):
            tokens = stmt.split()
            if tokens and tokens[0].startswith("@"):
                tokens = tokens[1:]
            if tokens:
                out.append(tokens)
    return out


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


#: the opcode counts of EVERY rendering (PTX statements by their first token); the two dependent-launch lines follow the arm
PTX_PINS = {
    "ld.global.b32": 1,  # the global scale: the ONE 32-bit global load
    "ld.global.v4.b32": 2,  # the block's sixteen bf16 values
    "st.global.v2.b32": 1,  # the block's eight code bytes
    "st.global.b8": 1,  # the block's scale byte
    "max.f32": 15,  # the block maximum of sixteen magnitudes
    "abs.f32": 16,
    "mul.f32": 19,  # amax x 1/6, gs x that, value x rcp(gs), sixteen elements x the encode scale
    "cvt.rn.satfinite.e4m3x2.f32": 1,  # the block scale
    "setp.eq.f32": 1,  # the zero rule
    "rcp.approx.ftz.f32": 2,
    "cvt.rn.satfinite.e2m1x2.f32": 8,
}
#: opcode prefixes no rendering holds: a division of any kind, a fused multiply-add, any synchronization or shared-memory access
PTX_ABSENT = ("div.", "fma.", "shfl.", "bar.", "mbarrier", "cp.async", "tcgen05", "ld.shared", "st.shared", "redux.", "atom.", "rsqrt.", "sqrt.")


#: arch -> {arm id: the rendering from its entry on}: this session's renderings (a cache of what a cell rendered, never an input)
_RENDERED_BODIES = {}


def _assert_addresses(ptx, what):
    """WHERE the kernel reads and writes, from the PTX of a rendering (``ptx_global_accesses``: the integer address arithmetic
    evaluated for every thread of every thread block of a ``MAX_ROWS`` call).  Five global accesses on four distinct kernel
    parameters: the scale's 32-bit load at byte 0 of its own; the two 128-bit loads at the bf16 elements ``row x width + 16 t``
    and eight elements on; the 64-bit store at byte ``row x width / 2 + 8 t`` of the codes; the byte store at the position of
    block scale ``(row, t)`` in the 128 x 4 layout WRITTEN OUT in the test helpers (``scale_byte_index``) -- and what that store
    writes is the E4M3 convert's code.  A row-major scale index, a dropped band term or another quarter of the atom FAILS here."""
    rows = kcfg.MAX_ROWS
    accesses = ptx_global_accesses(ptx, rows, THREADS)
    assert sorted(a["op"] for a in accesses) == ["ld.global.b32", "ld.global.v4.b32", "ld.global.v4.b32", "st.global.b8", "st.global.v2.b32"], what
    lost = [a["op"] for a in accesses if a["param"] is None]
    assert not lost, f"{what}: the address of {lost} is not followed by the reader (an instruction it does not evaluate feeds it)"
    by_op = {a["op"]: a for a in accesses}
    row = torch.arange(rows, dtype=torch.int64).reshape(-1, 1)
    block = torch.arange(THREADS, dtype=torch.int64).reshape(1, -1)
    element = row * WIDTH + block * BLOCK

    def at(access, want, name):
        got = access["offset"].expand(rows, THREADS)
        differing = got != want.expand(rows, THREADS)
        first = tuple(int(v) for v in differing.nonzero()[0]) if bool(differing.any()) else None
        assert first is None, (
            f"{what}: {name} addresses another byte for {int(differing.sum())} of {rows * THREADS} (row, block) pairs "
            f"(first at {first}: byte {int(got[first])}, want {int(want.expand(rows, THREADS)[first])})"
        )

    row_loads = [a for a in accesses if a["op"] == "ld.global.v4.b32"]
    at(by_op["ld.global.b32"], torch.zeros((), dtype=torch.int64), "the scale's load")
    at(row_loads[0], 2 * element, "the first row load")
    at(row_loads[1], 2 * (element + BLOCK // 2), "the second row load")
    at(by_op["st.global.v2.b32"], row * (WIDTH // 2) + block * (BLOCK // 2), "the code store")
    at(by_op["st.global.b8"], scale_byte_index(rows), "the scale byte's store")
    operands = [by_op["ld.global.b32"]["param"], row_loads[0]["param"], by_op["st.global.v2.b32"]["param"], by_op["st.global.b8"]["param"]]
    assert row_loads[1]["param"] == row_loads[0]["param"] and len(set(operands)) == 4, f"{what}: the accesses are not on four distinct operands: {operands}"
    convert = re.search(r"cvt\.rn\.satfinite\.e4m3x2\.f32\s+(\w+)\s*,[^;]*;\s*cvt\.u32\.u16\s+(%\w+)\s*,\s*\1\s*;", ptx)
    assert convert is not None, f"{what}: the E4M3 convert's code register"
    code = convert.group(2)
    assert (
        by_op["st.global.b8"]["value"] == code and len(re.findall(r"[\w.]+\s+" + re.escape(code) + r"\s*,", ptx)) == 1
    ), f"{what}: the scale byte's store writes {by_op['st.global.b8']['value']}, not the E4M3 convert's code {code} (written once)"


def _assert_arm_sites(ops, lines, wait, release, what):
    """The dependent-launch sites, read from the PTX by INSTRUCTION FORM (never by a kernel parameter's number): the one 32-bit
    global load is the scale's; the wait sits in the entry block behind it and in front of both 128-bit loads and both stores; the
    release, where the arm has one, is the very next instruction."""
    scale_load = ops.index("ld.global.b32")
    row_loads = [i for i, op in enumerate(ops) if op == "ld.global.v4.b32"]
    stores = [i for i, op in enumerate(ops) if op.startswith("st.global")]
    assert scale_load < min(row_loads) < min(stores), f"{what}: the scale load, the row loads and the stores are not in that order"
    if not wait:
        assert "griddepcontrol.wait" not in ops and "griddepcontrol.launch_dependents" not in ops, what
        return
    at = ops.index("griddepcontrol.wait")
    assert scale_load < at < min(row_loads), f"{what}: the wait does not sit between the scale load and the first row load"
    text_at = next(i for i, ln in enumerate(lines) if "griddepcontrol.wait" in ln)
    head = lines[next(i for i, ln in enumerate(lines) if ln.strip() == "{") + 1 : text_at]
    assert not any(re.match(r"\s*(\$\w+:|bra\b|call\b|ret\b)", ln) for ln in head), f"{what}: the wait is not in the kernel's entry block"
    assert (ops[at + 1] == "griddepcontrol.launch_dependents") == release, f"{what}: the instruction behind the wait is {ops[at + 1]!r}"
    assert ("griddepcontrol.launch_dependents" in ops) == release, what


@pytest.mark.parametrize("arch, cc", (("sm_100a", (10, 0)), ("sm_103a", (10, 3)), ("sm_107a", (10, 7)), ("sm_120a", None)))
def test_rendering_pins_the_convention(arch, cc, tmp_path):
    """Trace-compile the kernel for a target on this box (fake tensors with symbolic extents: no device needed) under the three
    launch arms and pin the instructions that ARE the contract.  ``sm_100a`` / ``sm_103a`` / ``sm_107a`` are the claimed targets;
    ``sm_120a`` is rendered to show that the kernel ASSEMBLES there (the entry declines that class).

    In EVERY rendering, exactly (``PTX_PINS``): ONE 32-bit global load (the scale), two 128-bit global loads (the block), one 64-bit
    store (the codes), one byte store (the scale); fifteen ``max.f32`` on sixteen ``abs.f32``; nineteen ``mul.f32``; ONE E4M3 convert,
    the zero rule's one compare, two ``rcp.approx.ftz.f32``, eight E2M1 pair converts; fp32(1/6) once, as the bits ``0x3E2AAAAB``, and
    no reciprocal of a constant; NO division, no fused multiply-add, no shuffle, barrier or shared-memory access of any kind
    (``PTX_ABSENT``), no ``.shared`` declaration; ``.reqntid 320``; the symbol = the kernel's name + the form text inside the first
    100 characters; four kernel parameters.  THE DEPENDENT-LAUNCH SITES (``_assert_arm_sites``): the wait sits in the entry block,
    behind the scale's load and in front of the first read of the row and of every store; the release is the very next instruction.
    THE ADDRESSES (``_assert_addresses``): the byte every global load and store addresses, evaluated from the PTX for every
    (row, block) of a 1024-row call -- the scale byte at its position in the 128 x 4 layout written out in the test helpers.
    The three arms differ ONLY in the ``griddepcontrol`` blocks.

    With an nvdisasm that decodes the cubin, also the compiled kernel: two ``MUFU.RCP`` and one multiply between them, all three
    PREDICATED (the reciprocals stay inside the zero rule's guard: nothing is speculated across it); nineteen multiplies, none
    ``.FTZ``, one of them by the immediate fp32(1/6); one E4M3 and eight E2M1 converts; one 64-bit and one byte store; no barrier, no
    shuffle, no shared-memory access, no spill; ``ACQBULK`` / ``PREEXIT`` as the arm says, the scale's load in front of ``ACQBULK``
    and every other global access behind it (in a launch arm that waits); and the operand WIRING with the registers kept
    (``row_quantize_fp4_decode_utils.sass_wiring_facts``): which value reaches the scale convert's operands, the association of the
    scale product, the guarded reciprocal chain, the ``+0`` of the zero rule, the element order of the code converts and of the
    store.  A compile failure of a claimed target FAILS; a DSL that does not serve the target (version, target list, compiler)
    SKIPS with the gate's text."""
    if cc is not None:
        msg = entry._dsl_error(cc)
        if msg is not None:
            pytest.skip(msg)
    else:
        msg = entry._dsl_version_error()
        if msg is not None:
            pytest.skip(msg)
        from cutlass.base_dsl.enums import Arch

        try:
            Arch.from_string(arch)
        except Exception:
            pytest.skip(f"this cutlass-dsl has no {arch}")
    cells = list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS))
    compiler, kept, rejected = _render(tmp_path, arch, [_spec(arm) for arm, _ in cells])
    if rejected:
        if cc is None and all(_UNSERVED in text for text in rejected.values()):
            pytest.skip(f"the installed compiler flavour of the DSL (CUDA {compiler[0]}.{compiler[1]}) does not assemble {arch}")
        pytest.fail(f"{arch}: the trace-compile was rejected: {rejected}")
    bodies, decoded = {}, 0
    for arm, arm_id in cells:
        files = kept.get(_spec(arm), [])
        ptx_files = [f for f in files if f.endswith(".ptx")]
        assert len(ptx_files) == 1, f"{arch} {arm_id}: expected one PTX, kept {files}"
        ptx = Path(ptx_files[0]).read_text()
        lines = ptx.splitlines()
        ops = [tokens[0] for tokens in _statements(ptx)]
        wait, release = arm_id != "plain", arm_id == "wait-release"
        what = f"{arch} {arm_id} (compiler CUDA {compiler[0]}.{compiler[1]})"
        want = dict(PTX_PINS, **{"griddepcontrol.wait": int(wait), "griddepcontrol.launch_dependents": int(release)})
        got = {op: ops.count(op) for op in want}
        assert got == want, f"{what}: " + ", ".join(f"{k} {got[k]} (want {want[k]})" for k in want if got[k] != want[k])
        stray = sorted({op for op in ops if op.startswith(PTX_ABSENT)})
        assert not stray, f"{what}: {stray}"
        assert sum(op.startswith("ld.global") for op in ops) == 3 and sum(op.startswith("st.global") for op in ops) == 2, f"{what}: another global access"
        assert ".shared" not in ptx and f".target {arch}" in ptx and f".reqntid {THREADS}, 1, 1" in ptx, what
        sixth = [tokens for tokens in _statements(ptx) if any("3E2AAAAB" in t.upper() or "1042983595" in t for t in tokens)]
        assert len(sixth) == 1 and sixth[0][0] == "mov.b32", f"{what}: fp32(1/6) as bits, once: {sixth}"
        constant_rcp = [tokens for tokens in _statements(ptx) if tokens[0].startswith("rcp.") and not tokens[-1].startswith("%")]
        assert not constant_rcp, f"{what}: a reciprocal of a constant: {constant_rcp}"
        assert len(re.findall(r"^\s*\.param", ptx, re.M)) == 4, f"{what}: the kernel's parameters"
        symbol = re.search(r"\.entry\s+(\w+)", ptx).group(1)
        head = SYMBOL + str(_cfg(arm))
        assert symbol.startswith(head + "_") and len(head) <= 100, f"{what}: the symbol {symbol[:120]} does not carry the form in its first 100 characters"
        _assert_arm_sites(ops, lines, wait, release, what)
        _assert_addresses(ptx, what)
        bodies[arm_id] = _body(ptx)
        cubins = [f for f in files if f.endswith(".cubin")]
        sass = _sass(cubins[0]) if cubins else None
        if sass is None:
            continue
        decoded += 1
        code = [re.sub(r"/\*[0-9a-f]+\*/", "", ln).strip() for ln in sass if re.search(r"/\*[0-9a-f]{4}\*/", ln)]
        n = lambda needle: sum(1 for ln in code if needle in ln)  # noqa: E731
        opcode = [next(t for t in ln.split() if not t.startswith("@")) for ln in code]
        assert n("MUFU.RCP") == 2 and sum(ln.startswith("@") and "MUFU.RCP" in ln for ln in code) == 2, f"{what}: two PREDICATED reciprocals in the SASS"
        assert n("FMUL") == 19 and sum(ln.startswith("@") and "FMUL" in ln for ln in code) == 1 and n("FMUL.FTZ") == 0, f"{what}: the multiplies in the SASS"
        assert (
            n("0.16666667163372039795") == 1 and n("MUFU") == 2 and n("FCHK") == 0 and n("FFMA") == 0
        ), f"{what}: the constant; no division, no fused multiply-add"
        assert n("F2FP.SATFINITE.E4M3") == 1 and n("F2FP.SATFINITE.E2M1") == 8, f"{what}: the converts in the SASS"
        assert n("STG.E.64") == 1 and n("STG.E.U8") == 1 and n("STG") == 2 and n("LDG.E.128") == 2 and n("LDG") == 3, f"{what}: the global accesses in the SASS"
        stray = sorted({op for op in opcode if op.startswith(("BAR", "SHFL", "STS", "LDS", "STL", "LDL", "ATOM", "RED"))})
        assert not stray, f"{what}: a barrier, shuffle, shared access or spill in the SASS: {stray}"
        assert n("ACQBULK") == int(wait) and n("PREEXIT") == int(release), what
        order = [op for op in opcode if op.startswith(("LDG", "STG", "ACQBULK", "PREEXIT"))]
        if wait:  # the compiled order: the scale's load, the wait, the release, then every other global access
            want_order = ["LDG.E", "ACQBULK"] + ["PREEXIT"] * int(release) + ["LDG.E.128", "LDG.E.128", "STG.E.64", "STG.E.U8"]
            assert order == want_order, f"{what}: the order of the global accesses and the dependent-launch pair in the SASS: {order}"
        else:  # without a wait the three loads may come in any order, in front of both stores
            assert sorted(order[:3]) == ["LDG.E", "LDG.E.128", "LDG.E.128"] and order[3:] == ["STG.E.64", "STG.E.U8"], f"{what}: {order}"
        facts = sass_wiring_facts(sass)
        assert facts is not None and all(facts.values()), f"{what}: the operand wiring of the compiled kernel: {facts}"
    for arm_id in ("wait", "wait-release"):
        assert _without_pdl_blocks(bodies[arm_id]) == _without_pdl_blocks(
            bodies["plain"]
        ), f"{arch}: the {arm_id} arm changes more than its griddepcontrol blocks"
    _RENDERED_BODIES[arch] = bodies
    if not decoded:
        pytest.skip(f"compiled + PTX pinned for {arch}; SASS half skipped: no nvdisasm candidate decodes the target")


def _arm_bodies(tmp_path, arch):
    """The three launch arms' renderings of ``arch`` from their entry on (the symbol normalized): rendered once per session."""
    if arch not in _RENDERED_BODIES:
        cells = list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS))
        where = tmp_path / arch
        where.mkdir()
        compiler, kept, rejected = _render(where, arch, [_spec(arm) for arm, _ in cells])
        assert not rejected, f"{arch} (compiler CUDA {compiler[0]}.{compiler[1]}): the trace-compile was rejected: {rejected}"
        _RENDERED_BODIES[arch] = {arm_id: _body(Path(next(f for f in kept[_spec(arm)] if f.endswith(".ptx"))).read_text()) for arm, arm_id in cells}
    return _RENDERED_BODIES[arch]


@pytest.mark.parametrize("arch, cc", (("sm_103a", (10, 3)), ("sm_107a", (10, 7))))
def test_rendering_of_a_claimed_target_is_the_sm_100a_text(arch, cc, tmp_path):
    """The kernel is arch-portable PTX: from its entry on (the ``.target`` line sits in the header in front of it) the rendering
    for ``sm_103a`` and for ``sm_107a`` is, line for line, the ``sm_100a`` rendering of the same launch arm.  So the text whose
    bytes the device cells check on a cc 10.0 part is the text the entry admits the other compute capability 10.x parts on; what
    a part of another class adds is its own assembler and its own approximate reciprocal -- the device run that class is owed,
    not another kernel.  A DSL that does not serve one of the two targets SKIPS with the gate's text."""
    for target_cc in ((10, 0), cc):
        msg = entry._dsl_error(target_cc)
        if msg is not None:
            pytest.skip(msg)
    reference, other = _arm_bodies(tmp_path, "sm_100a"), _arm_bodies(tmp_path, arch)
    for arm_id in LAUNCH_ARM_IDS:
        want, got = reference[arm_id].splitlines(), other[arm_id].splitlines()
        differing = [i for i, (a, b) in enumerate(zip(want, got)) if a != b]
        assert (
            len(want) == len(got) and not differing
        ), f"{arch} {arm_id}: the rendering is not the sm_100a text ({len(got)} lines against {len(want)}; {len(differing)} differing" + (
            f", first at line {differing[0]}: {got[differing[0]]!r} against {want[differing[0]]!r})" if differing else ")"
        )


_LAYOUT_PROBE = textwrap.dedent("""
    import json, os, sys
    os.environ["CUTE_DSL_ARCH"] = "sm_100a"  # unconditional: a host function is traced, nothing is assembled or launched
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blockscaled_layout as blockscaled_layout

    width, block = int(sys.argv[1]), int(sys.argv[2])
    row_counts = tuple(int(v) for v in sys.argv[3:])
    out = dict(reads={}, one_block=[])

    def evaluate():  # plain Python, run while the host function below is traced: the layout algebra needs a trace
        for rows in row_counts:
            layout = blockscaled_layout.tile_atom_to_shape_SF((rows, width, 1), block)
            out["reads"][str(rows)] = [[int(cute.crd2idx((r, c * block, 0), layout)) for c in range(width // block)] for r in range(rows)]
        layout = blockscaled_layout.tile_atom_to_shape_SF((2, width, 1), block)
        out["one_block"] = [int(cute.crd2idx((1, 7 * block + j, 0), layout)) for j in range(block)]

    @cute.jit
    def trace(rows: cutlass.Constexpr):
        evaluate()

    cute.compile(trace, row_counts)
    print("LAYOUT", json.dumps(out))
    """)


def test_scale_positions_are_those_a_block_scaled_fp4_multiply_reads(tmp_path):
    """THE LAYOUT, position by position, on the host (the DSL's layout algebra at trace time of a host function, in a fresh
    process: nothing is assembled or launched and no device is needed).  A block-scaled FP4 multiply of this library builds the
    scale layout of an operand of shape ``(rows, width, 1)`` with ``cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF(shape,
    16)`` (``cudnn.SwigluFp4Decode`` binds the scales of both of its operands so).  For EVERY (row, block) of 1, 2 and 129 rows --
    one band and two -- the byte that layout addresses for the block's elements is the byte the entry writes the block's scale to
    (the config's ``scale_byte_offset``: the kernel's index) and the byte the suite's reference reads (``scale_byte_index``: the
    layout written out); the sixteen elements of a block address ONE byte; and of the 640 (row, block) pairs of two rows a
    row-major index names the layout's byte for the first four blocks of row 0 only.  Needs the CuTe DSL; SKIPS with the version
    gate's text without it."""
    msg = entry._dsl_version_error()
    if msg is not None:
        pytest.skip(msg)
    row_counts = (1, 2, 129)
    probe = tmp_path / "layout_probe.py"  # a FILE: the DSL preprocessor re-reads a traced function's source through inspect
    probe.write_text(_LAYOUT_PROBE)
    proc = subprocess.run([sys.executable, str(probe), str(WIDTH), str(BLOCK), *(str(r) for r in row_counts)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"the layout probe failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    out = json.loads(next(ln for ln in proc.stdout.splitlines() if ln.startswith("LAYOUT ")).split(" ", 1)[1])
    cfg = _cfg()
    for rows in row_counts:
        reads = torch.tensor(out["reads"][str(rows)])
        assert tuple(reads.shape) == (rows, BLOCKS)
        writes = torch.tensor([[kcfg.scale_byte_offset(cfg, row, block) for block in range(BLOCKS)] for row in range(rows)])
        differing = reads != writes
        assert not bool(differing.any()), (
            f"{rows} rows: the multiply's layout reads another byte than the entry writes for {int(differing.sum())} of {rows * BLOCKS} "
            f"(row, block) pairs (first at {tuple(int(v) for v in differing.nonzero()[0]) if bool(differing.any()) else None})"
        )
        assert torch.equal(writes, scale_byte_index(rows)), f"{rows} rows: the entry's index is not the layout written out in the test helpers"
        assert int(writes.max()) < scale_rows(rows) * BLOCKS and writes.unique().numel() == writes.numel(), f"{rows} rows: distinct bytes inside the buffer"
    two_rows = torch.tensor(out["reads"]["2"])
    assert set(out["one_block"]) == {int(two_rows[1, 7])}, "the sixteen elements of a block address one scale byte"
    row_major = torch.arange(2 * BLOCKS).reshape(2, BLOCKS)
    same = (two_rows == row_major).nonzero().tolist()
    assert same == [[0, 0], [0, 1], [0, 2], [0, 3]], f"a row-major index names the layout's byte at {same}"


def test_wiring_reader_refuses_a_miswired_kernel():
    """The reader of the compiled kernel's wiring can FAIL: on a correct disassembly of the kernel's shape every fact holds, and each
    of ten one-line changes -- swapped nibbles of one code byte, the two stored words exchanged, the first reciprocal taken of the
    scale value instead of the global scale, an unpredicated reciprocal, an unpredicated multiply between the reciprocals, the
    product in the scale convert's upper slot, the constant multiplied into the global scale first, the constant one unit lower, a
    ``-0`` else value, a ``.FTZ`` multiply -- turns its own fact false.  Host-only, on a fixed text: no compiler."""
    good = textwrap.dedent("""
        /*0050*/ LDG.E R14, desc[UR4][R14.64] ;
        /*0060*/ ACQBULK ;
        /*00c0*/ PREEXIT ;
        /*00e0*/ LDG.E.128 R8, desc[UR4][R16.64] ;
        /*00f0*/ LDG.E.128 R4, desc[UR4][R16.64+0x10] ;
        /*0110*/ PRMT R16, RZ, 0x5410, R10.reuse ;
        /*0120*/ PRMT R15, RZ, 0x7610, R10 ;
        /*0130*/ PRMT R17, RZ, 0x5410, R11 ;
        /*0140*/ PRMT R18, RZ, 0x7610, R11 ;
        /*0150*/ PRMT R10, RZ, 0x5410, R4.reuse ;
        /*0160*/ PRMT R11, RZ, 0x7610, R4 ;
        /*0170*/ PRMT R2, RZ, 0x5410, R8 ;
        /*0180*/ PRMT R13, RZ, 0x5410, R9 ;
        /*0190*/ PRMT R4, RZ, 0x5410, R5 ;
        /*01a0*/ PRMT R19, RZ, 0x5410, R6 ;
        /*01b0*/ PRMT R8, RZ, 0x7610, R8 ;
        /*01c0*/ PRMT R9, RZ, 0x7610, R9 ;
        /*01d0*/ PRMT R5, RZ, 0x7610, R5 ;
        /*01e0*/ PRMT R6, RZ, 0x7610, R6 ;
        /*01f0*/ PRMT R20, RZ, 0x5410, R7 ;
        /*0200*/ FMNMX3 R21, |R2|, |R8|, |R13|, !PT ;
        /*0210*/ FMNMX3 R22, |R9|, |R16|, |R15|, !PT ;
        /*0220*/ FMNMX3 R23, |R17|, |R18|, |R10|, !PT ;
        /*0230*/ FMNMX3 R24, |R11|, |R4|, |R5|, !PT ;
        /*0240*/ FMNMX3 R25, |R19|, |R6|, |R20|, !PT ;
        /*0250*/ PRMT R7, RZ, 0x7610, R7 ;
        /*0260*/ FMNMX3 R21, R21, R22, R23, !PT ;
        /*0270*/ FMNMX3 R24, R24, R25, |R7|, !PT ;
        /*0280*/ FMNMX R21, R21, R24, !PT ;
        /*02a0*/ FMUL R21, R21, 0.16666667163372039795 ;
        /*02d0*/ FMUL R21, R14, R21 ;
        /*0300*/ F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R21, RZ, R21, RZ ;
        /*0320*/ LOP3.LUT R21, R21, 0xffff, RZ, 0xc0, !PT ;
        /*0330*/ F2FP.F16.E4M3.UNPACK_B R22, R21 ;
        /*0340*/ HADD2.F32 R23, -RZ, R22.H0_H0 ;
        /*0350*/ IMAD.MOV.U32 R22, RZ, RZ, RZ ;
        /*0360*/ FSETP.NEU.AND P0, PT, R23, RZ, PT ;
        /*0370*/ @P0 MUFU.RCP R14, R14 ;
        /*0380*/ @P0 FMUL R23, R23, R14 ;
        /*03b0*/ @P0 MUFU.RCP R22, R23 ;
        /*03e0*/ FMUL R20, R20, R22.reuse ;
        /*03f0*/ FMUL R7, R7, R22.reuse ;
        /*0400*/ FMUL R17, R17, R22.reuse ;
        /*0410*/ FMUL R18, R18, R22.reuse ;
        /*0420*/ FMUL R19, R19, R22.reuse ;
        /*0430*/ FMUL R6, R6, R22 ;
        /*0440*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R7, R7, R20, RZ ;
        /*0450*/ FMUL R16, R16, R22.reuse ;
        /*0460*/ FMUL R15, R15, R22.reuse ;
        /*0470*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R17, R18, R17, RZ ;
        /*0480*/ FMUL R13, R13, R22.reuse ;
        /*0490*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R19, R6, R19, R7 ;
        /*04a0*/ FMUL R6, R2, R22.reuse ;
        /*04b0*/ FMUL R2, R9, R22.reuse ;
        /*04c0*/ FMUL R4, R4, R22.reuse ;
        /*04d0*/ FMUL R7, R5, R22 ;
        /*04e0*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R15, R15, R16, R17 ;
        /*04f0*/ FMUL R5, R8, R22.reuse ;
        /*0500*/ FMUL R10, R10, R22.reuse ;
        /*0510*/ FMUL R11, R11, R22 ;
        /*0520*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R13, R2, R13, R15 ;
        /*0530*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R7, R7, R4, R19 ;
        /*0560*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R6, R5, R6, R13 ;
        /*0570*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R7, R11, R10, R7 ;
        /*05a0*/ STG.E.64 desc[UR4][R4.64], R6 ;
        /*05b0*/ STG.E.U8 desc[UR4][R2.64], R21 ;
        /*05c0*/ EXIT ;
        """)
    facts = sass_wiring_facts(good.splitlines())
    assert facts is not None and all(facts.values()), facts
    assert sass_wiring_facts(["/*0000*/ EXIT ;"]) is None
    mutants = (
        ("code_order", "PACK_AB_MERGE_C R6, R5, R6, R13", "PACK_AB_MERGE_C R6, R6, R5, R13"),
        ("code_order", "STG.E.64 desc[UR4][R4.64], R6", "STG.E.64 desc[UR4][R4.64], R7"),
        ("guarded_reciprocals", "@P0 MUFU.RCP R14, R14", "@P0 MUFU.RCP R14, R23"),
        ("guarded_reciprocals", "@P0 MUFU.RCP R14, R14", "MUFU.RCP R14, R14"),
        ("guarded_reciprocals", "@P0 FMUL R23, R23, R14", "FMUL R23, R23, R14"),
        ("scale_convert", "PACK_AB_MERGE_C R21, RZ, R21, RZ", "PACK_AB_MERGE_C R21, R21, RZ, RZ"),
        ("scale_product", "FMUL R21, R21, 0.16666667163372039795", "FMUL R14, R14, 0.16666667163372039795"),
        ("scale_product", "0.16666667163372039795", "0.16666665673255920410"),
        ("zero_else", "IMAD.MOV.U32 R22, RZ, RZ, RZ", "IMAD.MOV.U32 R22, RZ, RZ, 0x80000000"),
        ("no_ftz_multiply", "FMUL R21, R14, R21", "FMUL.FTZ R21, R14, R21"),
    )
    for fact, old, new in mutants:
        assert good.count(old) == 1
        got = sass_wiring_facts(good.replace(old, new).splitlines())
        assert got is not None and not got[fact], f"the change {old!r} -> {new!r} leaves {fact} true: {got}"
    for other_zero in ("HFMA2 R22, -RZ, RZ, 0, 0",):  # another assembler's zero move is the same fact
        got = sass_wiring_facts(good.replace("IMAD.MOV.U32 R22, RZ, RZ, RZ", other_zero).splitlines())
        assert got is not None and all(got.values()), got


def test_entry_binding_compiles_the_pinned_rendering(tmp_path):
    """The launch arguments the ENTRY builds from CUDA tensors -- of 1 row and of 1024 rows -- compile to the PTX text of the
    fake-tensor rendering pinned above, byte for byte, the kernel symbol and its parameter block included: the kernel the entry
    ships is the pinned one whatever row count it is compiled at (no extent of an operand enters it).  Compile-only, for
    ``sm_100a`` (a target both compiler flavours assemble): any CUDA device serves; nothing is launched."""
    if not torch.cuda.is_available():
        pytest.skip("a CUDA device is required to build the entry's launch arguments (any compute capability: the cell compiles, never launches)")
    msg = entry._dsl_error((10, 0))
    if msg is not None:
        pytest.skip(msg)
    arm = (True, None)
    texts = []
    # one fresh process per rendering: inside one process the DSL serves the second binding from its own compile cache and keeps no file
    for tag, spec in (("fake", _spec(arm)), ("rows1", _spec(arm, "bind", 1)), ("rows1024", _spec(arm, "bind", kcfg.MAX_ROWS))):
        sub = tmp_path / tag
        sub.mkdir()
        _, kept, rejected = _render(sub, "sm_100a", [spec])
        assert not rejected, rejected
        ptx_files = [f for f in kept.get(spec, []) if f.endswith(".ptx")]
        assert len(ptx_files) == 1, f"{spec}: expected one PTX, kept {kept.get(spec)}"
        texts.append(Path(ptx_files[0]).read_text())
    entry_line = lambda text: next(ln for ln in text.splitlines() if ".entry" in ln)  # noqa: E731
    for text, rows in zip(texts[1:], (1, kcfg.MAX_ROWS)):
        assert text == texts[0], (
            f"the entry's launch arguments ({rows} rows) compile to another kernel than the pinned rendering:\n"
            f"  pinned: {entry_line(texts[0])}\n  entry:  {entry_line(text)}"
        )


# ---------------------------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------------------------
_ENTRIES = {}


def _entry(dev, arm=(None, None)):
    """One entry per (device, launch arm) for the cells that only launch (an entry is one compiled kernel for every row count)."""
    key = (str(dev), arm)
    if key not in _ENTRIES:
        _ENTRIES[key] = make_entry(dev, arm)
    return _ENTRIES[key]


def _snap(k, x, scale, **outs):
    """One call; ``{codes, scales}`` as CLONES taken behind a synchronize."""
    codes, scales = k(x, scale, **outs)
    torch.cuda.synchronize()
    return dict(codes=codes.clone(), scales=scales.clone())


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.view(U8), b.view(U8)))


def _assert_same(got, want, label, names=("codes", "scales")):
    for name in names:
        differing = int((got[name].view(U8) != want[name].view(U8)).sum()) if got[name].shape == want[name].shape else -1
        assert _same(got[name], want[name]), f"{label}: {name} differs in {differing} of {got[name].numel()} bytes"


def _outs(rows, dev, extra=0, codes_fill=0xA5, scales_fill=0):
    """Fresh caller-owned outputs of ``rows`` rows (``extra`` more code rows and, with them, one more scale band)."""
    return dict(
        codes=torch.full((rows + extra, WIDTH // 2), codes_fill, dtype=U8, device=dev),
        scales=torch.full((scale_rows(rows) + (128 if extra else 0), BLOCKS), scales_fill, dtype=U8, device=dev).view(E4M3),
    )


@pytest.mark.parametrize("bits", SCALE_BITS, ids=[scale_id(b) for b in SCALE_BITS])
def test_matches_the_reference(bits):
    """The entry's bytes against the reference written from the format's definition, per global scale, at 1, 2, 3, 6, 8, 64, 129 and
    1024 rows of random and structured operands, in the wait + release arm: every block scale BITWISE, read through the layout
    written out in the test helpers; every code inside the two-code bracket of the approximate reciprocals; the padded rows of the
    entry-owned scale buffer zero.  The count of elements whose bracket holds two codes is one line of the run's "measured" section."""
    dev = _device_or_skip()
    import frost_routing

    k = _entry(dev, (True, None))
    scale = scale_tensor(bits, dev)
    two, elements = 0, 0
    for rows in ROWS:
        x = make_operands(rows, dev, seed=11 + rows, kind="structured" if rows in (3, 129) else "random")
        got = _snap(k, x, scale)
        assert tuple(got["codes"].shape) == (rows, WIDTH // 2) and tuple(got["scales"].shape) == (scale_rows(rows), BLOCKS)
        two += assert_reference(got["codes"], got["scales"], x, scale, label=f"{scale_id(bits)} rows {rows}")
        elements += rows * WIDTH
    frost_routing.measured(
        f"FP4 row quantization, reference cells under {scale_id(bits)}", f"{two} of {elements} elements have a two-code bracket; 0 outside it"
    )


@pytest.mark.parametrize("bits", SCALE_BITS, ids=[scale_id(b) for b in SCALE_BITS])
def test_scale_byte_of_every_block_maximum(bits):
    """EVERY finite bf16 magnitude as the maximum of one block (32,640 blocks, one call of 102 rows), per global scale: the stored
    scale byte equals the definition's ``e4m3(gs x (amax x fp32(1/6)))`` for every one of them -- a proof per global scale, not a
    sample -- and every code is inside the bracket.  Under the three constructed scales this cell FAILS for a kernel that forms the
    product in another association (``test_reference_separates_the_scale_associations`` pins on how many blocks)."""
    dev = _device_or_skip()
    x = all_maxima_operand()
    scale = scale_tensor(bits, dev)
    got = _snap(_entry(dev, (True, None)), x.to(dev), scale)
    stored = scale_bytes_of_rows(got["scales"], x.shape[0]).reshape(-1)
    want = maxima_scale_bytes(bits)
    differing = (stored != want).nonzero().reshape(-1)
    assert not differing.numel(), (
        f"{scale_id(bits)}: {differing.numel()} of {want.numel()} block maxima get another scale byte than the definition "
        f"(first: bf16 pattern {int(differing[0]):#06x}, stored {int(stored[differing[0]]):#04x}, definition {int(want[differing[0]]):#04x})"
    )
    assert_reference(got["codes"], got["scales"], x, scale, label=f"{scale_id(bits)} every block maximum")


@pytest.mark.parametrize("bits", POWER_OF_TWO_SCALE_BITS, ids=[scale_id(b) for b in POWER_OF_TWO_SCALE_BITS])
def test_exact_cells(bits):
    """Power-of-two global scales and block scales: both reciprocals are exact, so EVERY code is pinned BITWISE.  Row by row
    (``exact_operands``): every E2M1 rounding tie (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5 times the step, both signs) goes to the even
    code, under every power-of-two block scale E4M3 holds and with the block maximum at each of the sixteen positions; zeros of both
    signs encode as 0 and 8 under scale byte 0; a block whose scale underflows to the zero code encodes as zeros with the signs
    kept; a block of the largest finite bf16 magnitude saturates its scale at ``0x7E`` and its large elements at +-6; bf16
    subnormals have scale byte 0."""
    dev = _device_or_skip()
    x = exact_operands(bits)
    scale = scale_tensor(bits, dev)
    for arm in LAUNCH_ARMS:
        got = _snap(_entry(dev, arm), x.to(dev), scale)
        assert_reference(got["codes"], got["scales"], x, scale, label=f"exact {scale_id(bits)}")
        stored = scale_bytes_of_rows(got["scales"], x.shape[0])
        codes = unpack_e2m1(got["codes"].cpu())
        want = exact_codes(x, stored, scale)
        rows = [r for name, r in EXACT_ROWS.items() if name != "saturating"]
        bad = codes[rows] != want[rows]
        assert not bool(
            bad.any()
        ), f"exact {scale_id(bits)}: {int(bad.sum())} codes differ from e2m1(x x gs / F) (first at {tuple(int(v) for v in bad.nonzero()[0])})"
        ties = stored[EXACT_ROWS["ties"]]
        assert set(ties.tolist()) == {int(v) for v in torch.exp2(torch.arange(-9.0, 9.0)).to(E4M3).view(U8)}, "every power-of-two block scale"
        tie_row = x[EXACT_ROWS["ties"]].float().reshape(BLOCKS, BLOCK)
        assert set(tie_row.abs().argmax(dim=1).tolist()) == set(range(BLOCK)), "the block maximum sits at each of the sixteen positions"
        assert bool((stored[EXACT_ROWS["zeros"]] == 0).all()) and set(codes[EXACT_ROWS["zeros"]].tolist()) == {0, 8}
        assert bool((stored[EXACT_ROWS["underflow"]] == 0).all()) and set(codes[EXACT_ROWS["underflow"]].tolist()) == {0, 8}
        assert bool((stored[EXACT_ROWS["subnormal"]] == 0).all()) and set(codes[EXACT_ROWS["subnormal"]].tolist()) == {0, 8}
        sat = EXACT_ROWS["saturating"]
        assert bool((stored[sat] == 0x7E).all()), "the scale saturates at 448"
        assert codes[sat].reshape(-1, 4).unique(dim=0).tolist() == [[7, 15, 0, 8]], "the large elements saturate at +-6, the zeros keep their sign"


@pytest.mark.parametrize("rows", (1, 8, 129))
def test_codes_equal_the_norm_entrys_nvfp4_form(rows):
    """Two kernels of one convention inside the repository: ``cudnn.AddNormQuantDecode(add=False, bf16=True, quant="nvfp4")``
    returns the normalized bf16 rows, their codes and their block scales in one launch; this entry on those bf16 rows with the same
    global scale returns the SAME codes and the SAME scale buffer, raw bytes."""
    dev = _device_or_skip()
    norm = cudnn.AddNormQuantDecode(dev, WIDTH, 1e-6, add=False, bf16=True, quant="nvfp4")
    gen = torch.Generator(device="cpu").manual_seed(300 + rows)
    x = (torch.randn(rows, WIDTH, generator=gen) * 3).bfloat16().to(dev)
    weight = (torch.randn(WIDTH, generator=gen) * 0.3).bfloat16().to(dev)
    for bits in (ODD_SCALE_BITS, 0x42140000, CONSTRUCTED_SCALE_BITS[0]):
        scale = scale_tensor(bits, dev)
        out, codes, scales = norm(x, None, weight, scale)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out.float()).all())
        want = dict(codes=codes.clone(), scales=scales.clone())
        got = _snap(_entry(dev), out.clone(), scale)
        _assert_same(got, want, f"rows {rows} {scale_id(bits)}: this entry on the norm entry's bf16 rows")


def test_nan_and_infinity_blocks_touch_only_their_block():
    """A block that holds a NaN or an infinity is outside the contract: that block's codes and scale are unspecified.  EVERY other
    block of the row and every other row writes the bytes of a run without them.  What the affected blocks hold is one line of the
    run's "measured" section, never an assertion."""
    dev = _device_or_skip()
    import frost_routing

    rows = 4
    clean = make_operands(rows, "cpu", seed=77)
    dirty = clean.clone()
    marks = {(0, 319): float("nan"), (1, 7): float("nan"), (2, 100): float("inf"), (3, 0): -float("inf")}
    dirty[0, 319 * BLOCK : 320 * BLOCK] = float("nan")  # a block of NaN only
    dirty[1, 7 * BLOCK + 3] = float("nan")
    dirty[2, 100 * BLOCK + 15] = float("inf")
    dirty[3, 0] = -float("inf")
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    k = _entry(dev)
    want = _snap(k, clean.to(dev), scale)
    got = _snap(k, dirty.to(dev), scale)
    affected = torch.zeros(rows, BLOCKS, dtype=torch.bool)
    for row, block in marks:
        affected[row, block] = True
    code_blocks = lambda t: t["codes"].cpu().reshape(rows, BLOCKS, BLOCK // 2)  # noqa: E731
    assert torch.equal(code_blocks(got)[~affected], code_blocks(want)[~affected]), "a block without a NaN or an infinity changed its codes"
    got_scales, want_scales = scale_bytes_of_rows(got["scales"], rows), scale_bytes_of_rows(want["scales"], rows)
    assert torch.equal(got_scales[~affected], want_scales[~affected]), "a block without a NaN or an infinity changed its scale"
    seen = "; ".join(
        f"({row}, {block}) {'NaN' if value != value else value}: scale {int(got_scales[row, block]):#04x}, codes {code_blocks(got)[row, block].tolist()}"
        for (row, block), value in marks.items()
    )
    frost_routing.measured("FP4 row quantization, blocks with a NaN or an infinity (unspecified by contract; reported, not asserted)", seen)


def test_output_ownership_per_output():
    """Per output: a passed output is returned AS GIVEN and written in its live rows only (the code rows past ``rows`` and every
    scale byte that is not a live row's keep a sentinel); an output that is not passed is the entry's buffer of the row count, the
    same object on the next call, its scale buffer zero in the padded rows; both ownerships write the same bytes, in every mix."""
    dev = _device_or_skip()
    k = make_entry(dev, (True, None))
    for rows in (1, 3, 129):
        x = make_operands(rows, dev, seed=40 + rows)
        scale = scale_tensor(ODD_SCALE_BITS, dev)
        codes, scales = k(x, scale)
        torch.cuda.synchronize()
        assert codes is k.outputs(rows)["codes"] and scales is k.outputs(rows)["scales"]
        assert tuple(codes.shape) == (rows, WIDTH // 2) and tuple(scales.shape) == (scale_rows(rows), BLOCKS) and scales.dtype == E4M3
        owned = dict(codes=codes.clone(), scales=scales.clone())
        assert_reference(owned["codes"], owned["scales"], x, scale, label=f"entry-owned rows {rows}")
        mine = _outs(rows, dev, extra=3, codes_fill=0xA5, scales_fill=0x55)
        got_codes, got_scales = k(x, scale, **mine)
        torch.cuda.synchronize()
        assert got_codes is mine["codes"] and got_scales is mine["scales"]
        assert _same(mine["codes"][:rows], owned["codes"]) and bool((mine["codes"][rows:] == 0xA5).all()), f"rows {rows}: caller-owned codes"
        flat = mine["scales"].view(U8).cpu().reshape(-1)
        live = scale_layout_mask(mine["scales"].shape[0], rows)
        assert torch.equal(flat[live], owned["scales"].view(U8).cpu().reshape(-1)[scale_layout_mask(scale_rows(rows), rows)]), f"rows {rows}: live scale bytes"
        assert bool((flat[~live] == 0x55).all()), f"rows {rows}: the kernel wrote a scale byte that is not a live row's"
        for passed in ("codes", "scales"):
            fresh = _outs(rows, dev)
            pair = k(x, scale, **{passed: fresh[passed]})
            torch.cuda.synchronize()
            assert pair[("codes", "scales").index(passed)] is fresh[passed]
            _assert_same(dict(codes=pair[0], scales=pair[1]), owned, f"rows {rows}: only {passed} passed")
        again = k(x, scale)
        assert again[0] is codes and again[1] is scales, "the entry-owned buffers of a row count are one pair"


@pytest.mark.parametrize("rows", (1, 17, 160, 1024))
def test_bit_neutral_arms_write_the_default_bytes(rows):
    """The launch arms (wait; wait + release) against the plain launch: identical bytes.  The arms run behind a torch copy on the
    same stream that rewrites ``x`` right before the launch; such a copy never releases its dependents, so the kernel starts only
    after it has completed -- this cell shows that an arm's bytes are the plain launch's, and nothing about WHERE the wait sits (the
    rendering cell pins that in the kernel's text, a later cell on the device)."""
    dev = _device_or_skip()
    xs = make_operands(rows, dev, seed=51 + rows, kind="structured")
    stale = make_operands(rows, dev, seed=52 + rows)
    for bits in (ODD_SCALE_BITS, CONSTRUCTED_SCALE_BITS[1]):
        scale = scale_tensor(bits, dev)
        base = _snap(_entry(dev), xs, scale)
        for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
            k = _entry(dev, arm)
            x = stale.clone()
            k(x, scale)  # warm (compile outside the compared sequence), on the stale values
            x.copy_(xs)  # a stream predecessor that writes the operand: an ordinary copy, complete before this launch starts
            _assert_same(_snap(k, x, scale), base, f"{arm_id} rows {rows} {scale_id(bits)}")


def test_release_overlaps_a_dependent_without_changing_bytes():
    """The release with a dependent on the stream: the step built with ``pdl=True`` (wait + release) followed by launches that carry
    the programmatic-stream-serialization attribute, wait, and copy the step's ``codes`` and ``scales``
    (``pdl_dependent_copy.DependentCopy``).  The dependent may start while the step runs; its wait holds its reads.  Eager and
    captured in a CUDA graph, at 1 and 32 rows: the copied bytes are those of the plain step."""
    dev = _device_or_skip()
    from pdl_dependent_copy import DependentCopy

    copy = DependentCopy(dev)
    k = make_entry(dev, (True, None))
    assert k.cfg.PDL and k.cfg.PDL_RELEASE
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    for rows in (1, 32):
        x = make_operands(rows, dev, seed=61 + rows)
        want = _snap(_entry(dev), x, scale)
        k.compile(rows)
        copies = dict(codes=torch.zeros(rows, WIDTH // 2, dtype=U8, device=dev), scales=torch.zeros(scale_rows(rows), BLOCKS, dtype=E4M3, device=dev))

        def chain():
            codes, scales = k(x, scale)
            copy(codes, copies["codes"])
            copy(scales, copies["scales"])

        chain()  # eager (also compiles the dependent)
        torch.cuda.synchronize()
        _assert_same(copies, want, f"eager chain rows {rows}")
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                chain()
            for _ in range(3):
                for t in copies.values():
                    t.view(U8).fill_(0x33)
                graph.replay()
                torch.cuda.synchronize()
                _assert_same(copies, want, f"captured chain rows {rows}")
        finally:
            graph.reset()  # a test-owned graph is reset before the cell ends, also when it fails


def test_wait_holds_every_read_behind_a_predecessor_that_released_early():
    """The POSITION of the dependent-launch wait, on a device.  The stream predecessor is a launch that releases its dependents as
    its FIRST statement and only then writes ``x`` (``pdl_dependent_copy.EarlyReleaseWriter``: one thread block walks 1024 rows in
    many passes, of which only the last leaves the written values).  The step's kernel therefore STARTS while ``x`` holds other
    bytes, and its wait is the only thing that holds its reads of the row.

    The positive control comes first: a dependent WITHOUT a wait copies the buffer's last row behind the same writer and must read
    other bytes than the written ones, launched eagerly and replayed from a CUDA graph.  Only behind that observation does a pass
    of the step show anything; where the early start does not reproduce at all, the cell SKIPS (a coverage gap, not a pass), and
    either outcome is one line of the run's "measured" section.

    Then the step, in the wait and the wait + release arm, eager and captured: codes and scales are bitwise those of the plain step
    on the written values.  A kernel that read the row in front of its wait fails here in every row of its first scheduling wave.
    The eager launches are WARM ones: the first launch of a freshly compiled kernel returns to the host only when the work in
    flight on the stream has completed, so it never starts early and would show nothing."""
    dev = _device_or_skip()
    import frost_routing
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter

    rows = kcfg.MAX_ROWS
    new = make_operands(rows, dev, seed=67)
    stale = make_operands(rows, dev, seed=68)
    x = stale.clone()
    writer, unwaited = EarlyReleaseWriter(dev), DependentCopy(dev, wait=False)
    seen = torch.zeros(WIDTH, dtype=BF16, device=dev)

    def control():
        writer(new, x)
        unwaited(x[-1], seen)  # no wait: it reads whatever the last row holds when it starts

    def early_starts(run):
        """How many of four runs of the control read other bytes than the written ones (the dependent started in front of the write)."""
        count = 0
        for _ in range(4):
            x.copy_(stale)
            seen.view(torch.int16).fill_(0)
            torch.cuda.synchronize()  # an idle stream: the writer is the dependent's only predecessor in flight
            run()
            torch.cuda.synchronize()
            assert torch.equal(x.view(torch.int16), new.view(torch.int16)), "the writer leaves the written bytes"
            count += int(not torch.equal(seen.view(torch.int16), new[-1].view(torch.int16)))
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
    key = "FP4 row quantization decode step, the wait behind an early-releasing predecessor"
    seen_text = f"the control (a dependent without a wait) read other bytes than the written ones in {eager} of 4 eager launches and {captured} of 4 replays"
    if eager + captured == 0:
        frost_routing.measured(key, f"NOT COMPARED: {seen_text} -- the early start did not reproduce on this device")
        pytest.skip(f"{seen_text}: the early start did not reproduce on this device (a coverage gap, not a pass)")
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    want = _snap(_entry(dev), new, scale)
    compared, differing = 0, []
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        k = _entry(dev, arm)
        k.compile(rows)
        outs = _outs(rows, dev)

        def chain():
            writer(new, x)  # releases at its entry, then writes x
            k(x, scale, **outs)  # may start at once; every read of x is behind its wait

        def restore():
            x.copy_(stale)
            outs["codes"].fill_(0xA5)
            outs["scales"].view(U8).fill_(0)
            torch.cuda.synchronize()  # an idle stream: the writer is the step's only predecessor in flight

        def compare(label):
            nonlocal compared
            torch.cuda.synchronize()
            compared += 1
            try:
                _assert_same(outs, want, f"{arm_id}, {label}")
            except AssertionError as err:
                differing.append(str(err))

        restore()
        chain()  # the kernel's first launch cannot start early (the docstring): warm-up only
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


def test_one_compile_serves_every_row_count(monkeypatch):
    """Compiled once without an operand, the entry launches every row count -- entry-owned and caller-owned outputs, a 0-d and a
    ``[1]`` scale -- with compilation FORBIDDEN (the row count is a runtime argument, never a key)."""
    dev = _device_or_skip()
    k = make_entry(dev)
    k.compile()
    assert len(k.compiled) == 1 and set(k.buffers) == {1}
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("a warm call compiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    xs = make_operands(max(ROWS), dev, seed=71)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    for rows in ROWS:
        x = xs[:rows]
        got = _snap(k, x, scale)
        assert_reference(got["codes"], got["scales"], x, scale, label=f"rows {rows} after one compile")
        _assert_same(_snap(k, x, scale_tensor(ODD_SCALE_BITS, dev, zero_d=True)), got, f"rows {rows}: a 0-d scale")
        _assert_same(_snap(k, x, scale, **_outs(rows, dev)), got, f"rows {rows}: caller-owned outputs")
    k.compile(3)  # a second compile() is a no-op for the kernel
    assert len(k.compiled) == 1 and set(k.buffers) == set(ROWS)


def test_compile_declines_a_dsl_below_the_floor_before_the_kernel_is_built(monkeypatch):
    dev = _device_or_skip()
    from cudnn.frost import buffers

    k = make_entry(dev)
    x = make_operands(1, dev)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.1"):
        k.compile(1)
    with pytest.raises(NotImplementedError, match="found 4.7.1"):
        k(x, scale_tensor(ODD_SCALE_BITS, dev))
    assert not k.compiled


def test_graph_capture_replays_bitwise_with_eager():
    """Several launches captured into one CUDA graph after ``compile(rows)``: the replay is bitwise the eager run, and after the
    operands and the scales change IN PLACE the replay is bitwise a fresh eager run on the new values (every tensor is a launch
    argument: the graph holds pointers, never values)."""
    dev = _device_or_skip()
    rows = 6
    arms = ((True, None), (True, False), (True, None), (None, None))
    entries = [_entry(dev, arm) for arm in arms]
    for k in entries:
        k.compile(rows)
    xs = [make_operands(rows, dev, seed=100 + i, kind="structured" if i == 1 else "random") for i in range(len(arms))]
    scales = [scale_tensor(bits, dev) for bits in (ODD_SCALE_BITS, 0x3F800000, CONSTRUCTED_SCALE_BITS[0], 0x42140000)]
    outs = [_outs(rows, dev) for _ in arms]

    def eager():
        return [_snap(k, x, s) for k, x, s in zip(entries, xs, scales)]

    want = eager()
    graph = torch.cuda.CUDAGraph()

    def replay():
        for o in outs:
            o["codes"].fill_(0x55)
            o["scales"].view(U8).fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        return outs

    try:
        with torch.cuda.graph(graph):
            for k, x, s, o in zip(entries, xs, scales, outs):
                k(x, s, **o)
        for _ in range(3):
            for got, ref in zip(replay(), want):
                _assert_same(got, ref, "graph replay bitwise with eager")
        for i, x in enumerate(xs):  # new values in the same storage
            x.copy_(make_operands(rows, dev, seed=500 + i))
            scales[i].copy_(scale_tensor((0x3E800000, CONSTRUCTED_SCALE_BITS[2], ODD_SCALE_BITS, 0x44800000)[i], dev))
        got_all = [{n: t.clone() for n, t in got.items()} for got in replay()]
        for got, ref in zip(got_all, eager()):
            _assert_same(got, ref, "the replay reads the operands' current values")
    finally:
        graph.reset()


def test_warm_call_allocates_nothing_and_never_synchronizes():
    """The warm path: with caller-owned outputs and with entry-owned outputs of a row count that was already called, three calls
    allocate no device memory, and a call under ``set_sync_debug_mode('error')`` raises nothing (no device read reaches the host;
    ``scale``'s value is never fetched)."""
    dev = _device_or_skip()
    rows = 8
    k = make_entry(dev, (True, None))
    x = make_operands(rows, dev, seed=81)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    mine = _outs(rows, dev)
    want = _snap(k, x, scale)
    k(x, scale, **mine)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(x, scale, **mine)
            k(x, scale)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    _assert_same(mine, want, "the warm calls' bytes")


# The initialization of an entry-owned pair is stream work of the stream that allocated it (Rule 5).  The cells below park a stream
# behind a delay kernel, allocate a row count's pair on it (``compile(rows)``) and launch from ANOTHER stream while that work is
# still queued.  Every cell first warms the entry with a real launch: a cold first launch synchronizes while the kernel's module
# loads, which hides the overlap.
DELAY = 1_000_000_000  # cycles of the delay kernel: about half a second on the validated part


def _park(stream, work=None, cycles=DELAY):
    """Park ``stream`` behind a delay kernel, run ``work()`` on it, and return an event recorded behind both: pending until the
    delay has run.  A cell queries it to PROVE that its call overlapped the parked work, and -- where the call must not wait on
    the host -- that it did not."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")
    with torch.cuda.stream(stream):
        torch.cuda._sleep(cycles)
        if work is not None:
            work()
        marker = torch.cuda.Event()
        marker.record(stream)
    return marker


def _pooled(dev, rows, stream=None):
    """``stream`` (a fresh side stream when none is given) with blocks of the two output sizes of ``rows`` in its allocator pool,
    so the allocation inside ``compile(rows)`` on it is a pool hit and no allocation can synchronize the schedule away."""
    stream = torch.cuda.Stream(device=dev) if stream is None else stream
    with torch.cuda.stream(stream):
        blocks = _outs(rows, dev)
        del blocks
    torch.cuda.synchronize()
    return stream


def _warm_entry_and_reference(dev, arm, rows, seed):
    """A fresh entry warmed by a real one-row launch, operands of ``rows`` rows, and their bytes written into READY caller-owned
    outputs, held to the suite's reference: ``(entry, x, scale, want)``."""
    k = make_entry(dev, arm)
    x = make_operands(rows, dev, seed=seed)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    _snap(k, x[:1], scale)
    want = _snap(k, x, scale, **_outs(rows, dev))
    assert_reference(want["codes"], want["scales"], x, scale, label=f"rows {rows}: caller-owned outputs")
    return k, x, scale, want


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_warm_launch_waits_for_an_initialization_pending_on_another_stream(arm, arm_id):
    """``compile(rows)`` on a side stream behind a delay, the call on the original stream: the launch is ordered behind the
    pair's zero-fill although the host does not wait (the side stream's work is still pending when the call returns).  The codes
    AND every byte of the scale buffer are those of ready caller-owned outputs; without the ordering the kernel's block scales
    are written first and the delayed zero-fill erases every live scale byte."""
    dev = _device_or_skip()
    rows = 7
    k, x, scale, want = _warm_entry_and_reference(dev, arm, rows, seed=131)
    pending = _park(_pooled(dev, rows), lambda: k.compile(rows))
    codes, scales = k(x, scale)
    assert not pending.query(), "the call did not overlap the pending initialization (or it waited on the host)"
    torch.cuda.synchronize()
    assert codes is k.outputs(rows)["codes"] and scales is k.outputs(rows)["scales"]
    _assert_same(dict(codes=codes, scales=scales), want, f"{arm_id}: entry-owned outputs initialized on another stream")


def test_warm_launch_on_a_side_stream_waits_for_an_initialization_pending_on_the_original_stream():
    """The roles of the streams exchanged: ``compile(rows)`` on the original stream behind a delay, the call on a side stream."""
    dev = _device_or_skip()
    rows = 7
    k, x, scale, want = _warm_entry_and_reference(dev, (True, None), rows, seed=132)
    original = _pooled(dev, rows, torch.cuda.current_stream(dev))
    side = torch.cuda.Stream(device=dev)
    pending = _park(original, lambda: k.compile(rows))
    with torch.cuda.stream(side):
        codes, scales = k(x, scale)
    assert not pending.query(), "the call did not overlap the pending initialization (or it waited on the host)"
    torch.cuda.synchronize()
    _assert_same(dict(codes=codes, scales=scales), want, "entry-owned outputs initialized on the original stream, written on a side stream")


def test_captured_call_waits_on_the_host_for_a_pending_initialization_and_for_nothing_else():
    """The documented setup of a capture -- ``compile(rows)`` first -- on a side stream behind a delay, then a RAW capture
    (``torch.cuda.graph`` synchronizes the device when it is entered, which hides the schedule) and a replay at once.  A capturing
    stream cannot wait for an event recorded outside the capture, so the captured call waits for that ONE event on the host: the
    initialization is complete when the capture ends, unrelated work on a third stream is still pending (no stream or device
    synchronization), and both replays -- the second on new operand values -- write the bytes of ready caller-owned outputs."""
    dev = _device_or_skip()
    rows = 6
    k, x, scale, want = _warm_entry_and_reference(dev, (True, None), rows, seed=133)
    side = _pooled(dev, rows)
    unrelated = _park(torch.cuda.Stream(device=dev), cycles=4 * DELAY)
    pending = _park(side, lambda: k.compile(rows))
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(torch.cuda.Stream(device=dev)):
            graph.capture_begin()
            try:
                codes, scales = k(x, scale)
            finally:
                graph.capture_end()
        left_pending, synchronized_more = not pending.query(), unrelated.query()
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(dict(codes=codes, scales=scales), want, "replay 1, launched as soon as the capture ended")
        x.copy_(make_operands(rows, dev, seed=134))
        want = _snap(k, x, scale, **_outs(rows, dev))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(dict(codes=codes, scales=scales), want, "replay 2, on new operand values")
        assert not left_pending, "the captured call left the initialization pending"
        assert not synchronized_more, "the captured call waited for more than the initialization's event"
    finally:
        graph.reset()


def test_warm_call_does_nothing_for_an_initialization_that_is_complete(monkeypatch):
    """Once a call has seen the initialization complete, the warm path is what it was: three calls on the original stream and
    three on the stream that initialized the pair allocate no device memory, raise nothing under
    ``set_sync_debug_mode('error')``, and read or wait for no event and no stream."""
    dev = _device_or_skip()
    rows = 8
    k, x, scale, want = _warm_entry_and_reference(dev, (True, None), rows, seed=135)
    side = _pooled(dev, rows)
    pending = _park(side, lambda: k.compile(rows))
    codes, scales = k(x, scale)
    assert not pending.query(), "the call did not overlap the pending initialization (or it waited on the host)"
    torch.cuda.synchronize()
    _assert_same(dict(codes=codes, scales=scales), want, "the call that met the pending initialization")
    k(x, scale)  # the call that sees it complete
    torch.cuda.synchronize()
    touched = []
    for name in ("query", "synchronize", "wait"):
        monkeypatch.setattr(torch.cuda.Event, name, lambda *a, _name=name, **kw: touched.append(f"Event.{_name}"))
    for name in ("wait_event", "wait_stream", "synchronize"):
        monkeypatch.setattr(torch.cuda.Stream, name, lambda *a, _name=name, **kw: touched.append(f"Stream.{_name}"))
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(x, scale)
            with torch.cuda.stream(side):
                k(x, scale)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    monkeypatch.undo()
    torch.cuda.synchronize()
    assert not touched, f"a warm call behind a complete initialization touched {sorted(set(touched))}"
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    _assert_same(dict(codes=codes, scales=scales), want, "the warm calls' bytes")


def test_row_counts_initialized_on_two_streams_are_ordered_one_by_one():
    """Two row counts, each pair initialized on its own parked stream; the second stream is parked for longer.  The first row
    count is launched while both are pending and again after ITS initialization completed; the second row count is launched
    afterwards, while its own initialization is still pending: its launch still waits for it (the completion of one pair says
    nothing about another), and neither pair's bytes are disturbed."""
    dev = _device_or_skip()
    first, second = 5, 9
    k, x, scale, want_second = _warm_entry_and_reference(dev, (True, None), second, seed=136)
    want_first = _snap(k, x[:first], scale, **_outs(first, dev))
    streams = _pooled(dev, first), _pooled(dev, second)
    early = _park(streams[0], lambda: k.compile(first))
    late = _park(streams[1], lambda: k.compile(second), cycles=4 * DELAY)
    got_first = k(x[:first], scale)
    assert not early.query() and not late.query(), "the first call did not overlap both pending initializations"
    early.synchronize()  # this row count's initialization and nothing else: the other stream stays parked
    k(x[:first], scale)  # the call that sees the first pair's initialization complete
    got_second = k(x, scale)
    assert not late.query(), "the second row count's call did not overlap its pending initialization (or it waited on the host)"
    torch.cuda.synchronize()
    _assert_same(dict(zip(("codes", "scales"), got_first)), want_first, f"rows {first}, initialized on the first stream")
    _assert_same(dict(zip(("codes", "scales"), got_second)), want_second, f"rows {second}, initialized on the second stream")


def test_entry_owned_codes_on_a_recycled_block_wait_for_what_its_stream_still_owes_it():
    """The entry-owned ``codes`` are not initialized, but their block can be a recycled one: a tensor filled and freed on the
    parked stream right before ``compile(rows)`` hands its block to the entry while its fill is still queued.  A call on the
    original stream that uses the entry's ``codes`` (its ``scales`` are the caller's here) is ordered behind that fill too: the
    event is recorded behind everything the allocating stream had queued."""
    dev = _device_or_skip()
    rows = 7
    k, x, scale, want = _warm_entry_and_reference(dev, (True, None), rows, seed=137)
    side = _pooled(dev, rows)
    recycled = []

    def previous_owner_then_compile():
        old = torch.empty(rows, WIDTH // 2, dtype=U8, device=dev)
        old.fill_(0xEE)
        recycled.append(old.data_ptr())
        del old
        k.compile(rows)

    pending = _park(side, previous_owner_then_compile)
    if k.outputs(rows)["codes"].data_ptr() != recycled[0]:
        pytest.skip("the allocator did not hand the freed block to the entry: the schedule shows nothing")
    mine = _outs(rows, dev)
    codes, scales = k(x, scale, scales=mine["scales"])
    assert not pending.query(), "the call did not overlap the pending fill (or it waited on the host)"
    torch.cuda.synchronize()
    assert codes is k.outputs(rows)["codes"] and scales is mine["scales"]
    _assert_same(dict(codes=codes, scales=scales), want, "entry-owned codes on a recycled block")


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(2, dev)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    want = _snap(k, x, scale)
    sentinel = _outs(2, dev, codes_fill=0x77, scales_fill=0x77)
    with pytest.raises(TypeError, match="x must be torch.bfloat16"):
        k(x.float(), scale, **sentinel)
    with pytest.raises(ValueError, match="every operand must live on one device"):
        k(x.cpu(), scale, **sentinel)
    with pytest.raises(ValueError, match=r"x must be \[rows, 5120\]"):
        k(x[:, :2048].contiguous(), scale, **sentinel)
    with pytest.raises(ValueError, match="scale must be one fp32 element"):
        k(x, torch.tensor([0.01, 0.02], device=dev), **sentinel)
    with pytest.raises(TypeError, match="scales must be torch.float8_e4m3fn"):
        k(x, scale, codes=sentinel["codes"], scales=sentinel["scales"].view(U8))
    with pytest.raises(ValueError, match="at least one row"):
        k(x[:0], scale, **sentinel)
    with pytest.raises(ValueError, match="the entry-owned codes must not overlap scale"):
        k(x, k.outputs(2)["codes"].reshape(-1)[:4].view(torch.float32))
    torch.cuda.synchronize()
    assert bool((sentinel["codes"] == 0x77).all()) and bool((sentinel["scales"].view(U8) == 0x77).all()), "a declined warm call launches nothing"
    owned = k.outputs(2)
    assert _same(owned["codes"], want["codes"]) and _same(owned["scales"], want["scales"])


def test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current():
    """The device is implicit state exactly like the stream: an entry built for a device that is NOT the caller's current one
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
    x = make_operands(3, other, seed=91)
    scale = scale_tensor(ODD_SCALE_BITS, other)
    codes, scales = k(x, scale)
    torch.cuda.synchronize(other)
    assert torch.cuda.current_device() == current and codes.device == other and scales.device == other
    assert_reference(codes, scales, x, scale, label="an entry on a non-current device")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
@pytest.mark.parametrize("rows", (1, 6, 160))
def test_launch_facts_equal_the_profilers_record_of_a_launch(rows, arm, arm_id):
    """One call is ONE kernel launch whose profiler record equals the entry's launch facts: the kernel's name is the entry's symbol
    of the compiled form -- which carries the form's text directly behind the kernel's name -- and grid, block and shared memory
    (zero) are the launch shape.  Before the first compile the entry names no symbol."""
    dev = _device_or_skip()
    k = make_entry(dev, arm)
    assert k.kernel_symbols_for(rows) == (), "nothing is compiled yet"
    x = make_operands(rows, dev, seed=95)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    record = assert_facts_equal_the_launch(k, lambda: k(x, scale), rows)
    head = SYMBOL + str(k.cfg)
    assert record.name.startswith(head + "_") and len(head) <= 100, record.name
    assert k.kernel_symbols_for(rows) == (record.name,) == k.kernel_symbols_for(kcfg.MAX_ROWS), "one compiled kernel answers every row count"
    assert k.form_for(rows) == asdict(k.cfg)


def test_code_domain_is_inside_the_bracket():
    """The enumerated code domain of the non-power-of-two plain scale (854 rows, one call): every positive and every negative bf16
    element under every scale byte.  Scales bitwise; every code inside the two-code bracket; the count of two-code elements is one
    line of the run's "measured" section (they are the elements that can see the last bit of the part's approximate reciprocals).
    No block is exact under this scale: the one-code statement of the exact blocks is ``test_matches_the_reference`` under the six
    power-of-two scales and ``test_exact_cells``."""
    dev = _device_or_skip()
    import frost_routing

    x = code_domain_operand(ODD_SCALE_BITS)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    got = _snap(_entry(dev, (True, None)), x.to(dev), scale)
    two = assert_reference(got["codes"], got["scales"], x, scale, label="the code domain")
    stored = scale_bytes_of_rows(got["scales"], x.shape[0])
    assert set(stored.unique().tolist()) == set(range(0x7F))
    zero_blocks = stored == 0
    codes = unpack_e2m1(got["codes"].cpu()).reshape(x.shape[0], BLOCKS, BLOCK)
    assert set(codes[zero_blocks].reshape(-1).tolist()) <= {0, 8}, "a zero scale byte encodes signed zeros"
    assert_codes(got["codes"], x, stored, scale, label="the code domain, blocks with a non-zero scale", blocks=~zero_blocks)
    frost_routing.measured("FP4 row quantization, the enumerated code domain", f"{two} of {x.numel()} elements have a two-code bracket; 0 outside it")
