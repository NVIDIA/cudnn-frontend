# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP8 row quantization decode step (``cudnn.RowQuantizeFp8Decode``): one launch that quantizes every row of a bf16 slab to
E4M3 under its own scale, the row maximum over 448 with a floor.

HOST cells (no device of the served class is needed): the width's structure and the thread plan it decides, the typed declines of
the constructor and of every operand, the order of a launch behind an entry-owned allocation on stand-ins, the compute-capability
and CuTe DSL gates, the launch facts against the configuration arithmetic, the exports and the import boundary, and the kernel's
rendering for every claimed target (what synchronizes, where the dependent-launch wait and release sit, which divisions the
rounding chain uses).

DEVICE cells (a compute capability 10.x part; skipped elsewhere, which is a coverage gap and never a pass): every byte of the codes
and every bit of the scales against the independent reference of ``row_quantize_fp8_decode_utils`` -- random and structured rows
at 1 .. 1024 rows, EVERY finite bf16 magnitude as a row maximum, rows whose bytes are stated here as literals (zeros of both
signs, the floor on both sides of its knee, exact rounding ties, the saturated codes, a maximum in the first and in the last
column) -- rows of a wider slab and a repeated row, rows that hold a NaN or an infinity (only their own row is affected), output
ownership, the launch arms (bit-neutral; the wait behind a predecessor that released early; the release beside a dependent),
one compile for every row count, CUDA-graph replay, the warm path, the order behind a pending allocation on another stream, and
the launch facts against the profiler's record of a launch.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from row_quantize_fp8_decode_utils import (  # noqa: E402  (the launch arms, the operand builders, the reference, the device gate)
    BF16,
    BF16_FINITE,
    E4M3,
    FP32,
    FLOOR_BITS,
    KNEE_PATTERN,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    U8,
    VEC,
    WIDTH,
    _device_or_skip,
    assert_reference,
    bf16_from_patterns,
    e4m3_encode,
    e4m3_values,
    f32,
    make_entry,
    make_operands,
    row_scales,
    spike_reference,
    spike_rows,
)

from cudnn.frost import init_order  # noqa: E402
from cudnn.row_quantize.frost import row_quantize_fp8_decode as entry  # noqa: E402
from cudnn.row_quantize.frost.kernel import row_quantize_fp8_decode_config as kcfg  # noqa: E402
from init_order_schedules import (  # noqa: E402, F401  (settled_records is a fixture)
    deterministic_fill,
    host_allocations,
    on_device,
    park,
    pooled,
    previous_owner,
    settled_records,
    spans_of,
    stand_ins,
    warm_fills,
)
from launch_records import assert_facts_equal_the_launch  # noqa: E402

pytestmark = [pytest.mark.L0]

MAX_ROWS = kcfg.MAX_ROWS
SYMBOL = "cudnn_kernel_frost_row_quantize_fp8_"
#: structurally valid widths the entry does not serve: one vector, one warp, one full pass, two passes with a partial second, the limit
UNSERVED_WIDTHS = (8, 256, 8192, 8200, 16384, kcfg.MAX_WIDTH)


def _cfg(arm=(None, None), width=WIDTH):
    return entry.make_cfg(width, pdl=arm[0], pdl_release=arm[1])


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the width's structure, the thread plan, the launch knobs
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", (WIDTH,) + UNSERVED_WIDTHS)
def test_width_decides_a_thread_plan_that_covers_every_vector_once(width):
    """Whatever the structurally valid width: full warps, at most 1024 threads and three passes, every pass but the last owned by
    every thread, and the owners of all passes together are exactly the row's vectors -- no vector twice, none past the row."""
    cfg = kcfg.validate_cfg(kcfg.Cfg(WIDTH=width, PDL=False, PDL_RELEASE=False))
    assert cfg.THREADS % kcfg.WARP == 0 and 1 <= cfg.WARPS <= 32 and 1 <= cfg.PASSES <= kcfg.MAX_PASSES
    owners = [cfg.live_threads(p) for p in range(cfg.PASSES)]
    assert all(n == cfg.THREADS for n in owners[:-1]) and 1 <= owners[-1] <= cfg.THREADS
    assert sum(owners) * VEC == width and cfg.live_threads(cfg.PASSES) == 0
    assert cfg.SMEM_BYTES == 4 * cfg.WARPS and cfg.CTAS_PER_SM * cfg.THREADS <= kcfg.SM_THREADS


def test_served_width_and_its_plan():
    cfg = _cfg()
    assert kcfg.SERVED_WIDTH == (WIDTH,) and (cfg.THREADS, cfg.PASSES, cfg.live_threads(2), cfg.SMEM_BYTES) == (1024, 3, 128, 128)
    assert MAX_ROWS > cfg.CTAS_PER_SM * 212, "the row ceiling crosses a scheduling wave on every served part"


@pytest.mark.parametrize("width", (4, 0, -8, 12, WIDTH + 1, WIDTH + 4, kcfg.MAX_WIDTH + 8))
def test_declines_a_width_outside_the_structure(width):
    with pytest.raises(ValueError, match="width must be a multiple of 8"):
        make_entry("cpu", width=width)


@pytest.mark.parametrize("width", (float(WIDTH), str(WIDTH), None, True))
def test_declines_a_width_that_is_not_an_int(width):
    with pytest.raises(TypeError, match="width must be an int"):
        make_entry("cpu", width=width)


@pytest.mark.parametrize("width", UNSERVED_WIDTHS)
def test_declines_a_structurally_valid_width_it_does_not_serve(width):
    with pytest.raises(NotImplementedError, match=f"serves width {WIDTH}.*got width {width}"):
        make_entry("cpu", width=width)


def test_launch_knobs_validate_by_name():
    """``pdl`` ``None`` is off; ``pdl_release`` ``None`` follows ``pdl``; the release needs the wait; a knob is a ``bool`` or
    ``None``.  The form text names the width and the launch arm, one text per arm."""
    arms = {(None, None): (False, False), (False, None): (False, False), (True, None): (True, True), (True, True): (True, True), (True, False): (True, False)}
    for (pdl, release), (wait, released) in arms.items():
        cfg = _cfg((pdl, release))
        assert (cfg.PDL, cfg.PDL_RELEASE) == (wait, released) and make_entry("cpu", (pdl, release)).cfg == cfg
    assert {str(_cfg(arm)) for arm in LAUNCH_ARMS} == {f"w{WIDTH}_plain", f"w{WIDTH}_wait", f"w{WIDTH}_release_wait"}
    for release in (True,):
        for pdl in (None, False):
            with pytest.raises(ValueError, match="pdl_release.*needs pdl=True"):
                make_entry("cpu", (pdl, release))
    for bad in (1, 0, "yes", 1.0):
        with pytest.raises(TypeError, match="pdl must be a bool or None"):
            make_entry("cpu", (bad, None))
        with pytest.raises(TypeError, match="pdl_release must be a bool or None"):
            make_entry("cpu", (True, bad))
    with pytest.raises(TypeError, match="cfg must be"):
        kcfg.validate_cfg(dict(WIDTH=WIDTH))
    with pytest.raises(TypeError, match="pdl must be a bool"):
        kcfg.validate_cfg(kcfg.Cfg(WIDTH=WIDTH, PDL=1, PDL_RELEASE=False))
    with pytest.raises(ValueError, match="pdl_release"):
        kcfg.validate_cfg(kcfg.Cfg(WIDTH=WIDTH, PDL=False, PDL_RELEASE=True))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the typed declines of a call (CPU tensors; nothing is compiled or launched)
# ---------------------------------------------------------------------------------------------------------------------------
def _e4m3(*shape):
    return torch.zeros(*shape, dtype=U8).view(E4M3)


def _host_operands(rows=3, own=True, extra_rows=0):
    """Served CPU operands of ``rows`` rows: ``x`` and (``own``) caller-owned ``codes`` and ``scales``."""
    ops = dict(x=torch.zeros(rows, WIDTH, dtype=BF16))
    if own:
        ops["codes"] = _e4m3(rows + extra_rows, WIDTH)
        ops["scales"] = torch.zeros(rows + extra_rows, 1, dtype=FP32)
    return ops


def _check(ops, **kw):
    return entry.check_support(_cfg(), ops["x"], ops.get("codes"), ops.get("scales"), **kw)


def _at_offset(dtype, shape, offset):
    """A contiguous CPU tensor of ``shape`` whose first byte sits ``offset`` BYTES past a 16-byte aligned address."""
    n = 1
    for d in shape:
        n *= d
    size = torch.empty(0, dtype=dtype).element_size()
    raw = bytearray(n * size + 64)
    base = torch.frombuffer(raw, dtype=U8).data_ptr()
    t = torch.frombuffer(raw, dtype=U8, count=n * size, offset=(-base) % 16 + offset).view(dtype).view(*shape)
    assert t.data_ptr() % 16 == offset and t.is_contiguous()
    return t


def test_served_operands_pass_the_host_checks():
    """What the entry serves, on the host: every row count band, caller-owned or entry-owned outputs, taller outputs, the rows of a
    wider slab (a row stride that is whole 16-byte vectors), a repeated row (stride 0), ONE row of a slab of any row stride, and
    every operand at the smallest alignment its accesses need."""
    for rows in (1, 3, 1024):
        for own in (True, False):
            for extra in (0, 5):
                for cc in ((10, 0), (10, 3), (10, 7)):
                    assert _check(_host_operands(rows, own, extra), device_cc=cc) == rows
    ops = _host_operands()
    slab = torch.zeros(3, WIDTH + 24, dtype=BF16)
    assert _check(dict(ops, x=slab[:, :WIDTH]), device_cc=(10, 0)) == 3 and slab[:, :WIDTH].stride(0) == WIDTH + 24
    assert _check(dict(ops, x=torch.zeros(1, WIDTH, dtype=BF16).expand(3, WIDTH)), device_cc=(10, 0)) == 3, "a repeated row"
    odd = torch.zeros(4, WIDTH + 3, dtype=BF16)[:1, :WIDTH]
    assert _check(dict(_host_operands(1), x=odd), device_cc=(10, 0)) == 1, "one row has no row stride"
    least = dict(x=_at_offset(BF16, (3, WIDTH), 0), codes=_at_offset(E4M3, (3, WIDTH), 8), scales=_at_offset(FP32, (3, 1), 4))
    assert _check(least, device_cc=(10, 0)) == 3
    assert entry.check_binding(_cfg((True, None)), ops["x"]) == 3


def _mutations():
    """``{name: (operands, exception type, message pattern)}``: ONE change per decline of a call."""
    base = _host_operands()
    wide = torch.zeros(3, 2 * WIDTH, dtype=BF16)
    raw = torch.zeros(3 * WIDTH * 2 + 3 * WIDTH + 64, dtype=U8)  # ONE byte buffer under x, codes and scales
    x_on_raw = raw[: 3 * WIDTH * 2].view(BF16).reshape(3, WIDTH)
    return {
        "x-missing": (dict(base, x=None), TypeError, "x must be a torch.Tensor"),
        "x-not-a-tensor": (dict(base, x=[[0.0] * WIDTH]), TypeError, "x must be a torch.Tensor"),
        "codes-not-a-tensor": (dict(base, codes=bytearray(8)), TypeError, "codes must be a torch.Tensor"),
        "scales-not-a-tensor": (dict(base, scales=1.0), TypeError, "scales must be a torch.Tensor"),
        "x-fp32": (dict(base, x=base["x"].float()), TypeError, "x must be torch.bfloat16"),
        "x-fp16": (dict(base, x=base["x"].half()), TypeError, "x must be torch.bfloat16"),
        "codes-uint8": (dict(base, codes=base["codes"].view(U8)), TypeError, "codes must be torch.float8_e4m3fn"),
        "codes-e5m2": (dict(base, codes=base["codes"].view(torch.float8_e5m2)), TypeError, "codes must be torch.float8_e4m3fn"),
        "scales-fp64": (dict(base, scales=base["scales"].double()), TypeError, "scales must be torch.float32"),
        "scales-bf16": (dict(base, scales=base["scales"].to(BF16)), TypeError, "scales must be torch.float32"),
        "x-1d": (dict(base, x=base["x"].reshape(-1)), ValueError, "x must be"),
        "x-3d": (dict(base, x=base["x"].reshape(1, 3, WIDTH)), ValueError, "x must be"),
        "x-other-width": (dict(base, x=torch.zeros(3, 8192, dtype=BF16)), ValueError, "x must be"),
        "zero-rows": (dict(base, x=base["x"][:0]), ValueError, "at least one row"),
        "rows-above-the-ceiling": (_host_operands(MAX_ROWS + 1, own=False), NotImplementedError, f"at most {MAX_ROWS} rows"),
        "codes-too-few-rows": (dict(base, codes=base["codes"][:2]), ValueError, "codes must be"),
        "codes-other-width": (dict(base, codes=_e4m3(3, WIDTH // 2)), ValueError, "codes must be"),
        "codes-1d": (dict(base, codes=base["codes"].reshape(-1)), ValueError, "codes must be"),
        "scales-too-few-rows": (dict(base, scales=base["scales"][:2]), ValueError, "scales must be"),
        "scales-1d": (dict(base, scales=base["scales"].reshape(-1)), ValueError, "scales must be"),
        "scales-two-columns": (dict(base, scales=torch.zeros(3, 2, dtype=FP32)), ValueError, "scales must be"),
        "x-last-axis-strided": (dict(base, x=wide[:, ::2]), NotImplementedError, "x contiguous along its last axis"),
        "x-row-stride-not-whole-vectors": (dict(base, x=torch.zeros(3, WIDTH + 4, dtype=BF16)[:, :WIDTH]), ValueError, r"x stride\[0\]"),
        "codes-not-contiguous": (dict(base, codes=_e4m3(3, 2 * WIDTH)[:, ::2]), NotImplementedError, "codes contiguous"),
        "scales-not-contiguous": (dict(base, scales=torch.zeros(3, 2, dtype=FP32)[:, ::2]), NotImplementedError, "scales contiguous"),
        "x-misaligned": (dict(base, x=_at_offset(BF16, (3, WIDTH), 8)), ValueError, "x must start at a 16-byte aligned address"),
        "codes-misaligned": (dict(base, codes=_at_offset(E4M3, (3, WIDTH), 4)), ValueError, "codes must start at a 8-byte aligned address"),
        "scales-misaligned": (dict(base, scales=_at_offset(FP32, (3, 1), 2)), ValueError, "scales must start at a 4-byte aligned address"),
        "codes-overlaps-x": (
            dict(base, x=x_on_raw, codes=raw[1024 : 1024 + 3 * WIDTH].view(E4M3).reshape(3, WIDTH)),
            ValueError,
            "codes must not overlap x",
        ),
        "scales-overlaps-x": (dict(base, x=x_on_raw, scales=raw[64:76].view(FP32).reshape(3, 1)), ValueError, "scales must not overlap x"),
        "scales-overlaps-codes": (
            dict(base, codes=raw[: 3 * WIDTH].view(E4M3).reshape(3, WIDTH), scales=raw[2 * WIDTH : 2 * WIDTH + 12].view(FP32).reshape(3, 1)),
            ValueError,
            "codes must not overlap scales",
        ),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_typed_declines_name_the_operand(name):
    """One mutation per decline: the type, and a message that names the operand.  Every decline is raised by host metadata
    arithmetic, before any DSL import, compile or launch."""
    ops, error, pattern = _mutations()[name]
    with pytest.raises(error, match=pattern):
        _check(ops, device_cc=(10, 0))
    with pytest.raises(error, match=pattern):
        entry.check_binding(_cfg(), ops["x"], ops.get("codes"), ops.get("scales"))


def test_only_the_live_rows_of_an_output_count_as_written():
    """The overlap rule follows what a launch touches: the rows of a taller output past the call's rows may share bytes with
    another operand; its live rows may not."""
    raw = torch.zeros(8 * WIDTH, dtype=U8)
    codes = raw.view(E4M3).reshape(8, WIDTH)
    scales_past = raw[5 * WIDTH : 5 * WIDTH + 12].view(FP32).reshape(3, 1)
    scales_live = raw[2 * WIDTH : 2 * WIDTH + 12].view(FP32).reshape(3, 1)
    x = torch.zeros(3, WIDTH, dtype=BF16)
    assert entry.check_binding(_cfg(), x, codes, scales_past) == 3
    with pytest.raises(ValueError, match="must not overlap"):
        entry.check_binding(_cfg(), x, codes, scales_live)


def test_operands_that_only_touch_are_served():
    """The overlap rule is one of half-open byte ranges: an output that starts at the byte where another operand ends shares no
    byte with it.  ``x``, ``codes`` and ``scales`` laid back to back in ONE buffer bind; the same ``scales`` moved one row down
    into the live codes is declined."""
    rows = 3
    raw = torch.zeros(rows * WIDTH * 2 + rows * WIDTH + 4 * rows, dtype=U8)
    x = raw[: rows * WIDTH * 2].view(BF16).reshape(rows, WIDTH)
    codes = raw[rows * WIDTH * 2 : rows * WIDTH * 3].view(E4M3).reshape(rows, WIDTH)
    scales = raw[rows * WIDTH * 3 :].view(FP32).reshape(rows, 1)
    assert x.data_ptr() + 2 * x.numel() == codes.data_ptr() and codes.data_ptr() + codes.numel() == scales.data_ptr()
    assert entry.check_binding(_cfg(), x, codes, scales) == rows
    inside = raw[rows * WIDTH * 3 - 4 : rows * WIDTH * 3 + 8].view(FP32).reshape(rows, 1)
    with pytest.raises(ValueError, match="must not overlap"):
        entry.check_binding(_cfg(), x, codes, inside)


def test_binding_check_requires_every_operand_on_the_given_device():
    ops = _host_operands()
    with pytest.raises(ValueError, match="every operand must live on one device.*x is on cpu"):
        entry.check_binding(_cfg(), ops["x"], device=torch.device("cuda", 0))
    with pytest.raises(ValueError, match="every operand must live on one device.*scales is on meta"):
        entry.check_binding(_cfg(), ops["x"], ops["codes"], torch.zeros(3, 1, device="meta"))  # no device given: the operands share x's
    assert entry.check_binding(_cfg(), ops["x"], ops["codes"], ops["scales"], device="cpu") == 3


def test_declines_an_operand_that_overlaps_an_entry_owned_output():
    """An output the call does not pass is the entry's own buffer of the row count, and a tensor an earlier call returned can come
    back as an operand: a ``scales`` laid inside the entry-owned ``codes`` is declined by name BEFORE the compile and the launch;
    with both outputs passed the same binding is checked like any other.  A CPU entry: nothing compiles."""
    k = make_entry("cpu")
    ops = _host_operands(own=False)
    owned = k.outputs(3)
    assert tuple(owned["codes"].shape) == (3, WIDTH) and owned["codes"].dtype == E4M3
    assert tuple(owned["scales"].shape) == (3, 1) and owned["scales"].dtype == FP32
    assert owned is not k.outputs(3) and owned["codes"] is k.outputs(3)["codes"], "a copy of the entry's table, the same tensors"
    inside_codes = owned["codes"].view(U8).reshape(-1)[4 * WIDTH // 2 : 4 * WIDTH // 2 + 12].view(FP32).reshape(3, 1)
    with pytest.raises(ValueError, match="the entry-owned codes must not overlap scales.*pass a caller-owned output"):
        k(ops["x"], scales=inside_codes)
    with pytest.raises(ValueError, match="codes must not overlap scales"):
        k(ops["x"], owned["codes"], inside_codes)
    with pytest.raises(NotImplementedError, match="CUDA devices only"):  # the checks passed: the next gate is the device's
        k(ops["x"])
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
        with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows"):
            method(MAX_ROWS + 1)
    with pytest.raises(TypeError, match="rows must be an int"):
        k.outputs(None)
    for rows in (None, 3):
        with pytest.raises(NotImplementedError, match="CUDA devices only"):
            k.compile(rows)
    assert not k.compiled and not k.buffers, "a declined compile allocates nothing"


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the order of a launch behind an entry-owned allocation (cudnn.frost.init_order), on stand-ins
# ---------------------------------------------------------------------------------------------------------------------------
def test_the_entry_owned_pair_is_recorded_before_a_launch_can_find_it(monkeypatch):
    """The record of a pair's allocation, on stand-ins (no device): both tensors are allocated and only THEN one event is
    recorded on the stream that is current at that moment, with the whole byte range of each, BEFORE the entry holds the pair.
    A second ``outputs`` of the row count allocates and records nothing; another row count is another record.  The records are
    the device's, one object for every entry.  What a launch is ordered by is the live rows of its resolved outputs."""
    s = stand_ins(monkeypatch)
    dev = torch.device("cuda", 5)
    host_allocations(monkeypatch, dev, s.calls)
    k = make_entry(dev)
    assert k._pending is init_order.pending_of(dev) and make_entry(dev, (True, None))._pending is k._pending
    held, real = [], init_order.record
    monkeypatch.setattr(init_order, "record", lambda pending, tensors: (held.append(dict(k.buffers)), real(pending, tensors)))
    pair = k.outputs(3)
    assert s.calls == [("empty", (3, WIDTH)), ("empty", (3, 1)), ("enter", dev), ("record", 100), ("leave", dev)]
    assert held == [{}], "the record is made before the entry holds the pair"
    ((_, handle, spans),) = k._pending.values()
    assert handle == 100 and spans == spans_of(pair.values())
    s.calls.clear()
    assert k.outputs(3)["codes"] is pair["codes"] and not s.calls and len(k._pending) == 1, "allocated and recorded once"
    k.outputs(5)
    assert [c[0] for c in s.calls] == ["empty", "empty", "enter", "record", "leave"] and len(k._pending) == 2
    codes, scales = _e4m3(9, WIDTH), torch.zeros(9, 1, dtype=FP32)
    assert k._launch_writes(3, dict(codes=codes, scales=scales)) == (
        (codes.data_ptr(), codes.data_ptr() + 3 * WIDTH),
        (scales.data_ptr(), scales.data_ptr() + 12),
    )


def test_a_call_orders_its_launch_behind_a_pending_record_over_the_outputs_it_writes(monkeypatch):
    """The wiring, on stand-ins with the launch intercepted (no device): IN FRONT of the launch and on the LAUNCH stream, a call
    is ordered behind a pending record over the bytes of an output it writes -- passed, or the entry's own -- and behind none
    over ``x``, which it only reads; without a record nothing but the launch happens."""
    s = stand_ins(monkeypatch)
    k = make_entry("cpu")
    k.compiled[k.key()] = lambda *args: s.calls.append(("launch", int(args[-1])))
    ops = _host_operands()

    def ordering_of_a_call(over=None, **outs):
        k._pending.clear()
        event = s.Event(False)
        if over is not None:
            k._pending[0] = (event, 7, spans_of((over,)))
        s.calls.clear()
        k(ops["x"], **outs)
        return [c for c in s.calls if c[0] in ("wait_event", "launch")], [("wait_event", 100, event), ("launch", 100)]

    for name in ("codes", "scales"):
        got, ordered = ordering_of_a_call(ops[name], codes=ops["codes"], scales=ops["scales"])
        assert got == ordered, f"a record over the passed {name}"
        got, ordered = ordering_of_a_call(k.outputs(3)[name])
        assert got == ordered, f"a record over the entry-owned {name}"
        got, ordered = ordering_of_a_call(k.outputs(3)[name], **{name: k.outputs(3)[name]})
        assert got == ordered, f"a record over the entry-owned {name} passed back"
    assert ordering_of_a_call(ops["x"])[0] == [("launch", 100)], "x is only read"
    assert ordering_of_a_call()[0] == [("launch", 100)] and s.calls == [("launch", 100)], "without a record: the launch alone"


def test_a_declined_call_compile_or_outputs_allocates_nothing_and_leaves_no_record(monkeypatch):
    """A decline touches nothing, on stand-ins (no device).  Whatever declines a call or a ``compile(rows)`` that would use an
    entry-owned pair -- a part the kernel does not serve, a DSL gate, a binding check, the row count, a host device -- runs IN
    FRONT of the pair's allocation: the entry holds no pair, nothing was allocated and the device has no record.  A
    ``compile(rows)`` that is not declined allocates the pair of that row count and records it; ``compile()`` allocates nothing."""
    from cudnn.frost import buffers

    s = stand_ins(monkeypatch)
    dev, part = torch.device("cuda", 5), [(10, 0)]
    host_allocations(monkeypatch, dev, s.calls)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: part[0])
    plain = _host_operands(own=False)["x"]
    (x,) = on_device(dev, plain)
    tall, fp32 = on_device(dev, torch.zeros(MAX_ROWS + 1, WIDTH, dtype=BF16), torch.zeros(3, WIDTH))
    missing, old = (False, None), (True, ("nvidia-cutlass-dsl", "4.7.1"))
    declines = (  # what declines: the entry's device, the part, the DSL state, x, the decline
        ("a part the kernel does not serve", dev, (8, 0), missing, x, NotImplementedError, r"this device is 8\.0"),
        ("another line's part", dev, (12, 0), missing, x, NotImplementedError, r"this device is 12\.0"),
        ("a missing DSL", dev, (10, 0), missing, x, NotImplementedError, "the CuTe DSL is not installed"),
        ("a DSL below the floor", dev, (10, 3), old, x, NotImplementedError, r"found 4\.7\.1"),
        ("the row count", dev, (10, 0), missing, tall, NotImplementedError, f"at most {MAX_ROWS} rows"),
        ("a dtype", dev, (10, 0), missing, fp32, TypeError, "x must be torch.bfloat16"),
        ("a host device", "cpu", (10, 0), missing, plain, NotImplementedError, "CUDA devices only"),
    )
    for what, device, cc, dsl, operand, error, match in declines:
        part[0] = cc
        monkeypatch.setattr(buffers, "_DSL_STATE", dsl)
        k = make_entry(device)
        s.calls.clear()
        with pytest.raises(error, match=match):
            k(operand)
        assert not k.buffers and not k.compiled and not s.calls and not init_order.pending_of(dev), f"a call declined by {what} allocated or recorded"
        if what not in ("the row count", "a dtype"):  # those two are declines of a call's operands
            with pytest.raises(error, match=match):
                k.compile(3)
            assert not k.buffers and not k.compiled and not s.calls and not init_order.pending_of(dev), f"a compile declined by {what} allocated or recorded"
    k = make_entry(dev)
    with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows"):
        k.outputs(MAX_ROWS + 1)
    assert not k.buffers and not s.calls and not k._pending, "an outputs declined by the row count allocated or recorded"
    k.compiled[k.key()] = None  # a compiled entry: no gate is asked again
    k.compile()
    assert not k.buffers and not s.calls and not k._pending, "a compile without a row count allocates nothing"
    k.compile(3)
    assert [c[0] for c in s.calls] == ["empty", "empty", "enter", "record", "leave"] and len(k._pending) == 1 and set(k.buffers) == {3}


def test_compile_and_launch_run_under_the_entrys_device(monkeypatch):
    """The device is implicit state like the stream: the trace-compile, and a call's whole launch step -- the entry-owned pair, the
    launch stream, the order behind a pending allocation, the launch -- run inside the scope of the ENTRY's device and leave it
    again.  On stand-ins, with the compile and the launch intercepted (no device): an entry for a CUDA ordinal nobody made
    current."""
    cute = pytest.importorskip("cutlass.cute")
    s = stand_ins(monkeypatch)
    dev = torch.device("cuda", 5)
    host_allocations(monkeypatch, dev, s.calls)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    monkeypatch.setattr(entry, "_dsl_error", lambda cc: None)

    @contextmanager
    def scope(device):
        s.calls.append(("scope in", device))
        try:
            yield
        finally:
            s.calls.append(("scope out", device))

    monkeypatch.setattr(entry, "device_context", scope)
    monkeypatch.setattr(cute, "compile", lambda *args, **kwargs: s.calls.append(("compile",)) or (lambda *launch: s.calls.append(("launch",))))
    k = make_entry(dev)
    (x,) = on_device(dev, _host_operands(own=False)["x"])
    steps = ("scope in", "compile", "empty", "wait_event", "launch", "scope out")
    k.compile()
    assert [c for c in s.calls if c[0] in steps] == [("scope in", dev), ("compile",), ("scope out", dev)]
    s.calls.clear()
    k(x)
    assert [c[0] for c in s.calls if c[0] in steps] == ["scope in", "empty", "empty", "wait_event", "launch", "scope out"]
    assert {c[1] for c in s.calls if c[0] in ("scope in", "scope out")} == {dev}


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the compute-capability gate, the compile key, the launch facts
# ---------------------------------------------------------------------------------------------------------------------------
def test_declines_an_unserved_compute_capability_by_name():
    ops = _host_operands()
    for cc in ((9, 0), (9, 9), (8, 0), (8, 9), (12, 0), (12, 1), (11, 0)):
        with pytest.raises(NotImplementedError, match=rf"compute capability 10\.x.*this device is {cc[0]}\.{cc[1]}"):
            _check(ops, device_cc=cc)
        assert not entry.cc_supported(cc)
    assert all(entry.cc_supported(cc) for cc in ((10, 0), (10, 3), (10, 7), (10, 9))) and not entry.cc_supported(None)


def test_declines_cpu_operands_when_no_capability_is_given():
    with pytest.raises(NotImplementedError, match="CUDA tensors only"):
        _check(_host_operands())


def test_compile_declines_an_unserved_part_on_the_entrys_own_path(monkeypatch):
    """The capability decline a CALLER reaches: an entry's calls run the binding checks, and the part is gated in ``compile()``.  On
    MODELLED part facts (an entry for ``cuda:0`` with the device's compute capability patched: no device is touched)
    ``compile()``, ``compile(rows)`` and the first call on an unserved part are the typed decline by name, in front of the DSL
    gates, and nothing is compiled or allocated.  A served capability passes this gate and reaches the next one (the DSL,
    modelled as missing)."""
    from cudnn.frost import buffers

    x = _host_operands(own=False)["x"]
    for cc in ((12, 0), (12, 1), (9, 0), (8, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        k = make_entry("cuda:0")
        text = rf"compute capability 10\.x.*this device is {cc[0]}\.{cc[1]}"
        for rows in (None, 3):
            with pytest.raises(NotImplementedError, match=text):
                k.compile(rows)
        with monkeypatch.context() as bound:  # the first call compiles: with the binding checks stood in for, it reaches the same gate
            bound.setattr(entry, "check_binding", lambda *args, **kwargs: 3)
            with pytest.raises(NotImplementedError, match=text):
                k(x)
        assert not k.compiled and not k.buffers, "a declined compile builds and allocates nothing"
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    for cc in ((10, 0), (10, 3), (10, 7)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        k = make_entry("cuda:0")
        with pytest.raises(NotImplementedError, match="the CuTe DSL is not installed"):
            k.compile()
        assert not k.compiled and not k.buffers


def test_compile_key_is_plan_time_only():
    """One key per entry whatever the row count; it changes with each field of the config."""
    k = make_entry("cpu", (True, None))
    assert k.key() == (k.cfg,) and all(k.cfg_for(rows) is k.cfg for rows in (1, 17, MAX_ROWS))
    keys = {k.key()}
    for change in (dict(PDL_RELEASE=False), dict(PDL=False, PDL_RELEASE=False), dict(WIDTH=8192)):
        keys.add((replace(k.cfg, **change),))
    assert len(keys) == 4
    with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows"):
        k.cfg_for(MAX_ROWS + 1)


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_facts_equal_the_configuration_arithmetic_at_every_row_count(arm, arm_id):
    """EVERY served row count, 1 .. 1024, per launch arm: one thread block per row, the config's threads and shared memory; the
    form is the config's fields -- nothing else, because the compile key is the config alone."""
    k = make_entry("cpu", arm)
    want_form = asdict(k.cfg)
    assert want_form == dict(WIDTH=WIDTH, PDL=bool(arm[0]), PDL_RELEASE=arm_id == "release-wait")
    for rows in range(1, MAX_ROWS + 1):
        shape = k.launch_shape_for(rows)
        assert shape == ((rows, 1, 1), (k.cfg.THREADS, 1, 1), k.cfg.SMEM_BYTES), (rows, shape)
        assert all(type(v) is int for v in (*shape.grid, *shape.block, shape.shared_memory)), shape
        assert k.form_for(rows) == want_form, rows
    assert k.kernel_symbols_for(1) == () == k.kernel_symbols_for(MAX_ROWS), "nothing is compiled: no symbol"


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
        with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows"):
            fact(MAX_ROWS + 1)
        for rows in (1.0, "1", None, True):
            with pytest.raises(ValueError, match="rows must be an int"):
                fact(rows)
        fact(MAX_ROWS)


class _CompiledStandIn:
    """What the entry keeps per compile key, as far as the launch facts read it: a compiled DSL host function carries
    ``kernel_info``, keyed by the symbols of the kernels it launches."""

    def __init__(self, *symbols):
        self.kernel_info = {symbol: {} for symbol in symbols}


def test_kernel_symbol_is_that_of_the_compiled_form_at_every_row_count():
    k = make_entry("cpu")
    assert k.kernel_symbols_for(1) == () == k.kernel_symbols_for(MAX_ROWS)
    k.compiled[(replace(k.cfg, PDL=True),)] = _CompiledStandIn(SYMBOL + "other_0")
    assert k.kernel_symbols_for(1) == (), "a form the entry does not run is not named"
    k.compiled[k.key()] = _CompiledStandIn(SYMBOL + "A_0")
    assert all(k.kernel_symbols_for(rows) == (SYMBOL + "A_0",) for rows in (1, 2, 160, MAX_ROWS))
    k.compiled[k.key()] = object()
    with pytest.raises(NotImplementedError, match="does not expose the kernel symbols"):
        k.kernel_symbols_for(1)


def test_launch_arguments_are_the_addresses_and_the_row_stride_in_bytes():
    ops = _host_operands(5)
    slab = torch.zeros(5, WIDTH + 8, dtype=BF16)
    x = slab[:, :WIDTH]
    got = entry.launch_arguments(_cfg(), 5, x, ops["codes"], ops["scales"])
    assert got == (x.data_ptr(), 2 * (WIDTH + 8), ops["codes"].data_ptr(), ops["scales"].data_ptr(), 5) and all(type(v) is int for v in got)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the DSL gates (installed -> version floor -> the part's target -> the compiler that assembles it)
# ---------------------------------------------------------------------------------------------------------------------------
def test_declines_a_dsl_below_the_floor_by_version(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    for msg in (entry._dsl_version_error(), entry._dsl_error((10, 3)), entry._dsl_error((10, 7))):
        assert "nvidia-cutlass-dsl >= 4.8.0" in msg and "found 4.7.1" in msg
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    assert entry._dsl_version_error() is None


def test_declines_a_missing_dsl(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    for msg in (entry._dsl_version_error(), entry._dsl_error((10, 3))):
        assert "nvidia-cutlass-dsl >= 4.8.0" in msg and "not installed" in msg


def test_declines_a_dsl_without_the_parts_target_by_name(monkeypatch):
    """A version floor does not guarantee the part's target: with the version gate passed and a DSL whose target list lacks
    ``sm_107a`` cc 10.7 is declined naming the target, cc 10.0 / 10.3 are admitted.  The version gate runs first: the target probe
    (the first gate that imports the DSL) is never reached under a too-old DSL."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    msg = entry._dsl_error((10, 7))
    assert msg is not None and "sm_107a" in msg and msg.startswith(entry.NAME)
    assert entry._dsl_target_error((10, 3)) is None and entry._dsl_target_error((10, 0)) is None
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))

    def unreachable():
        raise AssertionError("the target probe ran although the version gate declined")

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", unreachable)
    assert "found 4.7.0" in entry._dsl_error((10, 7))


def test_declines_a_compiler_that_rejects_the_parts_target(monkeypatch):
    """cc 10.7 under a DSL whose compiler is the CUDA 12 toolchain flavour: declined BEFORE JIT, naming the compiler found and the
    one required.  cc 10.0 / 10.3 are served by both flavours."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (12, 9))
    msg = entry._dsl_error((10, 7))
    assert msg is not None and "compute capability 10.7" in msg and "CUDA 12.9" in msg and "cu13" in msg
    assert entry._dsl_error((10, 3)) is None and entry._dsl_error((10, 0)) is None
    for served in ((13, 4), (13, 6), (14, 0)):
        monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda v=served: v)
        assert entry._dsl_error((10, 7)) is None
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (13, 3))
    assert "CUDA 13.3" in entry._dsl_error((10, 7))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: exports and the import boundary
# ---------------------------------------------------------------------------------------------------------------------------
def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.RowQuantizeFp8Decode is entry.RowQuantizeFp8Decode
    family = importlib.import_module("cudnn.row_quantize")
    frost = importlib.import_module("cudnn.row_quantize.frost")
    assert family.RowQuantizeFp8Decode is entry.RowQuantizeFp8Decode and frost.RowQuantizeFp8Decode is entry.RowQuantizeFp8Decode
    assert entry.NAME == kcfg.NAME and set(entry.__all__) >= {"RowQuantizeFp8Decode", "check_binding", "check_support", "cc_supported", "make_cfg"}
    for module in (family, frost):
        with pytest.raises(AttributeError):
            module.NoSuchName


def test_constructing_an_entry_imports_neither_the_dsl_nor_the_kernel():
    """The lazy boundary, in fresh interpreters: what a statement imports is a DELTA against an empty interpreter.  Importing
    cudnn loads none of the family's modules; the family packages, the config module and the host entry load neither torch nor
    the CuTe DSL nor the kernel module; constructing an entry (which resolves its device with torch) and running its
    construction declines still loads neither the DSL nor the kernel module."""

    def modules(code):
        probe = code + "\nimport json, sys; print('@@' + json.dumps(sorted(sys.modules)))"
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
        return set(json.loads(next(ln for ln in out.splitlines() if ln.startswith("@@"))[2:]))

    def added(code):
        return modules(code) - modules("")

    def unwanted(names):
        return sorted(m for m in names if m.split(".")[0] in ("torch", "cutlass", "jax") or m.endswith("row_quantize_fp8_decode_f16"))

    by_cudnn = added("import cudnn")
    assert not unwanted(by_cudnn) and not [m for m in by_cudnn if m.startswith("cudnn.row_quantize")], sorted(by_cudnn)
    family = "cudnn.row_quantize, cudnn.row_quantize.frost, cudnn.row_quantize.frost.kernel.row_quantize_fp8_decode_config, cudnn.row_quantize.frost.row_quantize_fp8_decode"
    assert not unwanted(added(f"import {family}"))
    construct = (
        "import cudnn\n"
        f"k = cudnn.RowQuantizeFp8Decode('cpu', {WIDTH}, pdl=True)\n"
        "for bad in (dict(width=12), dict(width=8192), dict(width='x'), dict(width=%d, pdl_release=True)):\n"
        "    try:\n"
        "        cudnn.RowQuantizeFp8Decode('cpu', **bad)\n"
        "    except (ValueError, TypeError, NotImplementedError):\n"
        "        pass\n"
        "    else:\n"
        "        raise SystemExit('a construction decline did not raise')\n" % WIDTH
    )
    assert not [m for m in added(construct) if m.split(".")[0] == "cutlass" or m.endswith("row_quantize_fp8_decode_f16")]


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the kernel's rendering for every claimed target (a trace-compile in a fresh process; nothing is launched)
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
    from cudnn.row_quantize.frost import row_quantize_fp8_decode as entry
    from cudnn.row_quantize.frost.kernel import row_quantize_fp8_decode_f16 as mod

    print("COMPILER", int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    for spec in sys.argv[3:]:
        before = set(os.listdir(dump))
        width, pdl, release = spec.split("|")
        cfg = entry.make_cfg(int(width), pdl=None if pdl == "None" else pdl == "True", pdl_release=None if release == "None" else release == "True")
        args = [cutlass.Int64(0)] * len(mod.OPERANDS) + [cutlass.Int32(1)]  # the argument types the entry compiles with
        try:
            cute.compile(mod.launch, cfg, *args, make_fake_stream())
        except Exception as error:  # a target the installed compiler flavour does not assemble: reported, the caller decides
            print("REJECTED", spec, type(error).__name__, " ".join(str(error).split())[:400])
            continue
        for name in sorted(set(os.listdir(dump)) - before):
            print("KEPT", spec, name)
    print("COMPILED", arch)
    """)

_UNSERVED = "Unsupported GPU architecture"  # the DSL's message when its compiler flavour has no such target


def _spec(arm):
    return "|".join(str(v) for v in (WIDTH, arm[0], arm[1]))


def _render(tmp_path, arch, specs):
    """Run the probe for ``arch`` in a fresh process (the DSL reads its dump settings once, at the first import); returns the
    compiler's ``(major, minor)``, ``{spec: {suffix: path}}`` and ``{spec: rejection text}``."""
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
            kept.setdefault(spec, {})[name.rsplit(".", 1)[-1]] = str(dump / name)
        if ln.startswith("REJECTED "):
            _, spec, text = ln.split(" ", 2)
            rejected[spec] = text
    return compiler, kept, rejected


def _sass(cubin):
    """The disassembly of a cubin by the first ``nvdisasm`` (``$CUDA_PATH/bin``, then ``PATH``) that decodes it, or None."""
    candidates = [os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm")] if os.environ.get("CUDA_PATH") else []
    candidates.append(shutil.which("nvdisasm"))
    for tool in dict.fromkeys(c for c in candidates if c and os.path.isfile(c) and os.access(c, os.X_OK)):
        try:
            proc = subprocess.run([tool, "-c", cubin], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
    return None


def _first(body, token):
    found = re.search(re.escape(token), body)
    return None if found is None else found.start()


@pytest.mark.parametrize("arch, cc", (("sm_100a", (10, 0)), ("sm_103a", (10, 3)), ("sm_107a", (10, 7))))
def test_rendering_synchronizes_once_and_waits_before_every_global_access(arch, cc, tmp_path):
    """The kernel compiles for every claimed target under the installed DSL, in every launch arm, with the properties its text
    states: ONE thread-block barrier and no other synchronization object (no mbarrier, no asynchronous copy, no tensor memory, no
    atomic); under ``pdl`` the dependent-launch wait in front of EVERY global load and store, under ``pdl_release`` the release
    in front of the wait, neither in the plain arm; the two divisions of the scale are the correctly rounded ones and no
    approximate reciprocal or division appears; the floor is the stated bit pattern; the clamp is in the stream.  Where a
    disassembler of the toolkit decodes the image: no local-memory access (nothing is spilled).  A target the installed
    compiler flavour does not assemble is the entry's own typed decline, and the cell says so instead of passing."""
    compiler, kept, rejected = _render(tmp_path, arch, [_spec(arm) for arm in LAUNCH_ARMS])
    if rejected:
        assert all(_UNSERVED in text for text in rejected.values()) and cc == (10, 7) and compiler < entry.SM107_MIN_COMPILER, rejected
        pytest.skip(f"the installed DSL compiles with CUDA {compiler[0]}.{compiler[1]}, which does not assemble {arch}: the entry declines it by name")
    for arm, arm_id in zip(LAUNCH_ARMS, LAUNCH_ARM_IDS):
        cfg = _cfg(arm)
        what = f"{arch} {arm_id}"
        ptx = open(kept[_spec(arm)]["ptx"]).read()
        body = re.sub(r"//[^\n]*", "", ptx[ptx.index(".entry") :])
        assert len(re.findall(r"\bbar\.sync\b", body)) == 1, f"{what}: the one thread-block barrier"
        assert len(re.findall(r"\bst\.shared\b", body)) == 1 and len(re.findall(r"\bld\.shared\b", body)) == 1, f"{what}: one slot store, one slot load"
        for absent in ("mbarrier", "cp.async", "tcgen05", "atom.", "rcp.approx", "div.full", "div.approx", "sqrt"):
            assert absent not in body, f"{what}: {absent}"
        assert len(re.findall(r"\bdiv\.rn\.f32\b", body)) == 2 and f"0x{FLOOR_BITS:08X}" in body.upper().replace("0X", "0x"), f"{what}: the scale's chain"
        assert "min.xorsign.abs.f32" in body and "cvt.rn.satfinite.e4m3x2.f32" in body, f"{what}: the clamp and the saturating convert"
        assert f".reqntid {cfg.THREADS}" in body and re.search(r"\.entry\s+(\w+)", ptx).group(1).startswith(SYMBOL + str(cfg) + "_"), what
        wait, release = _first(body, "griddepcontrol.wait"), _first(body, "griddepcontrol.launch_dependents")
        accesses = [m.start() for m in re.finditer(r"\b(?:ld|st)\.global\b", body)]
        assert accesses, what
        assert (wait is not None) == cfg.PDL and (release is not None) == cfg.PDL_RELEASE, f"{what}: the arm's control instructions"
        if cfg.PDL:
            assert wait < min(accesses), f"{what}: a global access in front of the wait"
        if cfg.PDL_RELEASE:
            assert release < wait, f"{what}: the release behind the wait"
        sass = _sass(kept[_spec(arm)]["cubin"])
        if sass is not None:
            assert not re.search(r"\b(?:STL|LDL)\b", sass), f"{what}: a local-memory access"


# ---------------------------------------------------------------------------------------------------------------------------
# device: the bytes
# ---------------------------------------------------------------------------------------------------------------------------
_ENTRIES = {}


def _entry(dev, arm=(None, None)):
    """One compiled entry per (device, launch arm) for the cells that only launch."""
    key = (str(dev), arm)
    if key not in _ENTRIES:
        _ENTRIES[key] = make_entry(dev, arm)
    return _ENTRIES[key]


def _outs(rows, dev, extra=0, codes_fill=0xA5, scales_fill=-7.0):
    """Fresh caller-owned outputs of ``rows + extra`` rows, sentinel-filled."""
    return dict(
        codes=torch.full((rows + extra, WIDTH), codes_fill, dtype=U8, device=dev).view(E4M3),
        scales=torch.full((rows + extra, 1), scales_fill, dtype=FP32, device=dev),
    )


def _snap(k, x):
    """One launch into fresh caller-owned buffers: ``{"codes": uint8, "scales": fp32}`` clones."""
    outs = _outs(x.shape[0], x.device)
    k(x, **outs)
    torch.cuda.synchronize()
    return dict(codes=outs["codes"].view(U8).clone(), scales=outs["scales"].clone())


def _same(a, b):
    return a.shape == b.shape and bool((a.reshape(-1).view(U8) == b.reshape(-1).view(U8)).all())


def _assert_same(got, want, label):
    for name in ("codes", "scales"):
        a, b = got[name].contiguous().view(U8).reshape(-1), want[name].contiguous().view(U8).reshape(-1)
        assert a.shape == b.shape, f"{label}: {name} is {tuple(got[name].shape)}, want {tuple(want[name].shape)}"
        differing = int((a != b).sum())
        assert differing == 0, f"{label}: {differing} of {a.numel()} bytes of {name} differ"


def _assert_untouched(outs, rows, label, codes_fill=0xA5, scales_fill=-7.0):
    assert bool((outs["codes"][rows:].view(U8) == codes_fill).all()), f"{label}: a code byte past the call's rows was written"
    assert bool((outs["scales"][rows:] == scales_fill).all()), f"{label}: a scale past the call's rows was written"


@pytest.mark.parametrize("kind", ("random", "structured"))
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 6, 8, 16, 64, MAX_ROWS))
def test_matches_the_reference(rows, kind):
    """Every code byte and every scale bit against the independent reference, into outputs one row taller than the call (the row
    past the call keeps its sentinel: a thread without a vector in the last pass stores nothing behind the last row)."""
    dev = _device_or_skip()
    x = make_operands(rows, dev, seed=11 * rows + (kind == "structured"), kind=kind)
    outs = _outs(rows, dev, extra=1)
    codes, scales = _entry(dev)(x, **outs)
    torch.cuda.synchronize()
    assert codes is outs["codes"] and scales is outs["scales"]
    assert_reference(codes, scales, x, label=f"{kind} rows {rows}")
    _assert_untouched(outs, rows, f"{kind} rows {rows}")


def test_scale_and_code_of_every_finite_row_maximum():
    """EVERY finite bf16 magnitude as the maximum of a row -- one non-zero element per row, of either sign, in a column that walks
    the row's first and last columns, the edges of the three passes and the last thread that owns a vector of the third.  The
    scale is a function of the row maximum alone: its bits are the reference's for all 32,640 magnitudes -- the floor for every
    magnitude up to the knee ``2**-9`` and the correctly rounded quotient above it -- and the row's codes are the code of ``+0``
    everywhere but at the element, whose code is the reference's (``0x7E`` / ``0xFE`` above the knee)."""
    dev = _device_or_skip()
    k = _entry(dev)
    patterns = torch.arange(BF16_FINITE, dtype=torch.int64)
    patterns = patterns | ((patterns % 3 == 1).to(torch.int64) << 15)  # a third of the rows hold a negative maximum
    edges = torch.tensor([0, 7, 8, 8191, 8192, 16383, 16384, 16384 + 127 * VEC, WIDTH - 1 - VEC, WIDTH - 1], dtype=torch.int64)
    index = torch.arange(BF16_FINITE, dtype=torch.int64)
    columns = torch.where(index % 2 == 0, edges[(index // 2) % len(edges)], (index * 7919) % WIDTH)
    want_code, want_scale = spike_reference(patterns)
    rows = 1020
    assert BF16_FINITE % rows == 0
    wrong = []
    for start in range(0, BF16_FINITE, rows):
        part = slice(start, start + rows)
        x = spike_rows(patterns[part], columns[part]).to(dev)
        outs = _outs(rows, dev)
        k(x, **outs)
        expected = torch.zeros(rows, WIDTH, dtype=U8)
        expected[torch.arange(rows), columns[part]] = want_code[part]
        codes_differ = int((outs["codes"].view(U8) != expected.to(dev)).sum())
        scales_differ = int((outs["scales"].view(torch.int32).cpu().reshape(-1) != want_scale[part].view(torch.int32)).sum())
        if codes_differ or scales_differ:
            wrong.append((start, codes_differ, scales_differ))
    assert not wrong, f"(first pattern, differing codes, differing scales) per call of {rows} rows: {wrong}"
    knee = int((want_scale.view(torch.int32) == FLOOR_BITS).sum())
    assert knee == KNEE_PATTERN + 1, "the floor holds for every magnitude up to the knee and for no other"
    above = want_code[(patterns & 0x7FFF) > KNEE_PATTERN] & 0x7F
    assert bool((above == 0x7E).all()), "above the knee the element that holds the row maximum has the largest code"


#: E4M3 codes of exact values under scale 1 (a row maximum of 448), stated as literals: the saturated codes, values on the grid,
#: exact rounding ties (to the EVEN code), the subnormal codes, signed zeros
_LITERAL_CODES = (
    (448.0, 0x7E),
    (-448.0, 0xFE),
    (432.0, 0x7E),  # the tie of 416 (0x7D) and 448 (0x7E)
    (400.0, 0x7C),  # the tie of 384 (0x7C) and 416 (0x7D)
    (-400.0, 0xFC),
    (1.0, 0x38),
    (1.125, 0x39),
    (1.0625, 0x38),  # the tie of 1 (0x38) and 1.125 (0x39)
    (1.1875, 0x3A),  # the tie of 1.125 (0x39) and 1.25 (0x3A)
    (-1.0625, 0xB8),
    (2.0**-9, 0x01),  # the smallest subnormal code
    (2.0**-10, 0x00),  # the tie of 0 and 2**-9
    (-(2.0**-10), 0x80),  # a negative value that rounds to zero keeps its sign
    (1.5 * 2.0**-9, 0x02),  # the tie of 2**-9 (0x01) and 2**-8 (0x02)
    (2.0**-11, 0x00),
    (0.0, 0x00),
    (-0.0, 0x80),
)


def test_rows_whose_bytes_are_stated_as_literals():
    """Rows with bytes written out here, not computed: a zero row and a row of negative zeros (the floor scale; the codes of the
    signed zeros); the floor on both sides of its knee; a row under scale 1 that holds exact rounding ties (to the even code),
    the saturated codes, the subnormal codes and signed zeros; a row whose maximum is its LAST element and one whose maximum is
    its first; and each of them between a row of tiny and a row of huge values (a row's maximum never crosses into a
    neighbour).  The whole call is also held to the reference."""
    dev = _device_or_skip()
    x = torch.zeros(12, WIDTH, dtype=BF16)
    x[1] = -0.0
    below, knee, above = (bf16_from_patterns(torch.tensor([p])) for p in (KNEE_PATTERN - 1, KNEE_PATTERN, KNEE_PATTERN + 1))
    x[2, 5], x[3, 5], x[4, 5] = below, knee, above
    x[2, 9] = 2.0**-10  # half the knee, under the floor scale
    values = torch.tensor([v for v, _ in _LITERAL_CODES]).to(BF16)
    assert torch.equal(values.float(), torch.tensor([v for v, _ in _LITERAL_CODES])), "every literal is a bf16 value"
    x[5, 100 : 100 + len(values)] = values
    x[5, WIDTH - len(values) :] = values  # the same values in the vectors of the last pass
    below_448 = values[2:]  # without the two values that hold the maximum themselves
    x[6, WIDTH - 1] = -448.0  # the row's ONLY maximum is its last element
    x[6, : len(below_448)] = below_448
    x[7, 0] = 448.0  # the row's ONLY maximum is its first element
    x[7, WIDTH - len(below_448) :] = below_448
    x[8] = 2.0**-40  # a row of tiny values ...
    x[9] = 3.0e38  # ... beside a row of huge ones
    x[10] = 2.0**-40
    x[11, ::2], x[11, 1::2] = 448.0, -448.0
    x = x.to(dev)
    got = _snap(_entry(dev), x)
    assert_reference(got["codes"], got["scales"], x, label="the literal rows")
    codes, scales = got["codes"].cpu(), got["scales"].view(torch.int32).cpu().reshape(-1)
    one = 0x3F800000
    assert int(scales[0]) == FLOOR_BITS and bool((codes[0] == 0x00).all()), "a zero row"
    assert int(scales[1]) == FLOOR_BITS and bool((codes[1] == 0x80).all()), "a row of negative zeros"
    assert int(scales[2]) == FLOOR_BITS and int(scales[3]) == FLOOR_BITS, "the floor up to its knee"
    assert int(scales[4]) > FLOOR_BITS and [int(codes[r, 5]) for r in (2, 3, 4)] == [0x7E] * 3, "around the knee"
    assert int(codes[2, 9]) == 0x76, "under the floor scale half the knee is the code of 224"
    want = torch.tensor([c for _, c in _LITERAL_CODES], dtype=U8)
    assert int(scales[5]) == one and torch.equal(codes[5, 100 : 100 + len(want)], want) and torch.equal(codes[5, WIDTH - len(want) :], want)
    assert int(scales[6]) == one and int(codes[6, WIDTH - 1]) == 0xFE and torch.equal(codes[6, : len(want) - 2], want[2:]), "a maximum in the last column"
    assert int(scales[7]) == one and int(codes[7, 0]) == 0x7E and torch.equal(codes[7, WIDTH - len(want) + 2 :], want[2:]), "a maximum in the first column"
    assert int(scales[8]) == FLOOR_BITS and int(scales[10]) == FLOOR_BITS and bool((codes[9] == 0x7E).all()), "tiny rows beside a huge row"
    assert int(scales[11]) == one and bool((codes[11, ::2] == 0x7E).all()) and bool((codes[11, 1::2] == 0xFE).all())


def test_reference_encoder_is_the_format():
    """The reference's own parts, on the host: its E4M3 table is the format's (every code decodes to its table value in torch's
    E4M3 dtype), its encoder inverts the table, sends every exact midpoint to the even code and agrees with torch's own
    conversion on a spread of values of both signs, and its divisions are the correctly rounded fp32 ones for every finite bf16
    row maximum (the fp32 quotient of torch on the host, where a division is IEEE)."""
    values = e4m3_values()
    decoded = torch.arange(0x7F, dtype=U8).view(E4M3).to(FP32).to(torch.float64)
    assert torch.equal(decoded, values) and float(values[-1]) == 448.0 and bool((values[1:] > values[:-1]).all())
    assert torch.equal(e4m3_encode(values.to(FP32)), torch.arange(0x7F, dtype=U8))
    assert torch.equal(e4m3_encode(-values.to(FP32)), torch.arange(0x7F, dtype=U8) | 0x80)
    assert e4m3_encode(torch.tensor([1e30, -1e30, float("inf")])).tolist() == [0x7E, 0xFE, 0x7E]
    ties = ((values[1:] + values[:-1]) / 2).to(FP32)  # every exact midpoint of two neighbouring codes: to the even one
    assert torch.equal(e4m3_encode(ties) % 2, torch.zeros(len(ties), dtype=U8)) and bool((e4m3_encode(ties) - torch.arange(len(ties)) <= 1).all())
    spread = torch.cat([ties, -ties, torch.randn(1 << 16, generator=torch.Generator().manual_seed(3)) * torch.logspace(-4, 3, 1 << 16)])
    assert torch.equal(e4m3_encode(spread), spread.to(E4M3).view(U8)), "the encoder against torch's own E4M3 conversion on the host"
    amax = bf16_from_patterns(torch.arange(BF16_FINITE)).to(FP32)
    scale, inv = row_scales(amax)
    assert torch.equal(scale, torch.maximum(amax / 448.0, f32(FLOOR_BITS))) and torch.equal(inv, 1.0 / scale)


def test_rows_of_a_wider_slab_and_a_repeated_row():
    """``x`` binds as the rows of a WIDER slab (a row stride that is whole 16-byte vectors): the columns between two rows hold
    huge values that no row may read.  ``x`` binds as ONE row repeated (stride 0): every row writes that row's bytes.  The
    outputs bind at the smallest alignment their stores need: ``codes`` at an address that is 8 and not 16 bytes aligned,
    ``scales`` at one that is 4 and not 8."""
    dev = _device_or_skip()
    k = _entry(dev)
    rows = 5
    slab = torch.full((rows, WIDTH + 24), 3.0e38, dtype=BF16)
    slab[:, :WIDTH] = make_operands(rows, seed=31, kind="structured")
    x = slab.to(dev)[:, :WIDTH]
    assert not x.is_contiguous() and x.stride(0) == WIDTH + 24
    raw_codes = torch.full((rows * WIDTH + 64,), 0xA5, dtype=U8, device=dev)
    raw_scales = torch.full((4 * rows + 64,), 0xA5, dtype=U8, device=dev)
    codes = raw_codes[8 : 8 + rows * WIDTH].view(E4M3).reshape(rows, WIDTH)
    scales = raw_scales[4 : 4 + 4 * rows].view(FP32).reshape(rows, 1)
    assert codes.data_ptr() % 16 == 8 and scales.data_ptr() % 8 == 4
    k(x, codes, scales)
    torch.cuda.synchronize()
    assert_reference(codes, scales, x, label="the rows of a wider slab")
    assert bool((raw_codes[:8] == 0xA5).all()) and bool((raw_codes[8 + rows * WIDTH :] == 0xA5).all()), "a code byte outside the operand"
    assert bool((raw_scales[:4] == 0xA5).all()) and bool((raw_scales[4 + 4 * rows :] == 0xA5).all()), "a scale byte outside the operand"
    one = make_operands(1, dev, seed=32)
    alone = _snap(k, one)
    repeated = _snap(k, one.expand(rows, WIDTH))
    _assert_same(repeated, {name: t.expand(rows, -1) for name, t in alone.items()}, "a repeated row")


def test_nan_and_infinity_rows_touch_only_their_row():
    """A row that holds a NaN or an infinity leaves its OWN codes and scale unspecified.  Every other row of the call writes the
    bytes it writes without such a neighbour.  What the affected rows hold is one line of the run's "measured" section, never an
    assertion."""
    dev = _device_or_skip()
    import frost_routing

    k = _entry(dev)
    rows = 9
    clean = make_operands(rows, seed=41, kind="structured")
    x = clean.clone()
    affected = {2: ("one NaN", float("nan")), 4: ("+inf", float("inf")), 6: ("-inf", float("-inf")), 8: ("a row of NaN", float("nan"))}
    for row, (what, value) in affected.items():
        if what == "a row of NaN":
            x[row] = value
        else:
            x[row, 777 + row] = value
    got = _snap(k, x.to(dev))
    want = _snap(k, clean.to(dev))
    others = [r for r in range(rows) if r not in affected]
    _assert_same({n: t[others] for n, t in got.items()}, {n: t[others] for n, t in want.items()}, "the rows beside a non-finite row")
    seen = []
    for row, (what, _) in affected.items():
        bits = int(got["scales"].view(torch.int32)[row, 0]) & 0xFFFFFFFF
        codes = sorted(set(got["codes"][row].tolist()))
        seen.append(f"{what}: scale bits 0x{bits:08x}, {len(codes)} distinct codes {[hex(c) for c in codes[:6]]}")
    frost_routing.measured("FP8 row quantization decode step, rows with a NaN or an infinity (reported, not asserted)", "; ".join(seen))


# ---------------------------------------------------------------------------------------------------------------------------
# device: ownership, the launch arms, the compile, capture, the warm path
# ---------------------------------------------------------------------------------------------------------------------------
def test_output_ownership_per_output():
    """An output the caller passes is returned as given; one it does not pass is the entry's own of that row count -- each slot
    alone.  The entry-owned pair has the call's rows; another row count has another pair; a taller caller-owned output keeps
    its rows past the call."""
    dev = _device_or_skip()
    k = make_entry(dev)
    rows = 5
    x = make_operands(rows, dev, seed=51)
    want = _snap(k, x)
    codes, scales = k(x)
    torch.cuda.synchronize()
    owned = k.outputs(rows)
    assert codes is owned["codes"] and scales is owned["scales"] and tuple(codes.shape) == (rows, WIDTH) and tuple(scales.shape) == (rows, 1)
    assert codes.dtype == E4M3 and scales.dtype == FP32
    _assert_same(dict(codes=codes, scales=scales), want, "entry-owned outputs")
    mine = _outs(rows, dev, extra=3)
    got_codes, got_scales = k(x, codes=mine["codes"])
    torch.cuda.synchronize()
    assert got_codes is mine["codes"] and got_scales is owned["scales"]
    _assert_same(dict(codes=got_codes[:rows], scales=got_scales), want, "caller-owned codes, entry-owned scales")
    owned["codes"].view(U8).fill_(0)
    got_codes, got_scales = k(x, scales=mine["scales"])
    torch.cuda.synchronize()
    assert got_codes is owned["codes"] and got_scales is mine["scales"]
    _assert_same(dict(codes=got_codes, scales=got_scales[:rows]), want, "entry-owned codes, caller-owned scales")
    _assert_untouched(mine, rows, "a taller caller-owned output")
    assert k(x[:2])[0] is k.outputs(2)["codes"] and k.outputs(2)["codes"] is not owned["codes"] and set(k.buffers) == {2, rows}
    torch.cuda.synchronize()


@pytest.mark.parametrize("rows", (1, 17, MAX_ROWS))
def test_bit_neutral_arms_write_the_default_bytes(rows):
    """The wait arm and the release + wait arm write the bytes of the plain arm (an idle stream: the wait returns at once)."""
    dev = _device_or_skip()
    x = make_operands(rows, dev, seed=60 + rows, kind="structured")
    want = _snap(_entry(dev), x)
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        _assert_same(_snap(_entry(dev, arm), x), want, f"{arm_id} rows {rows}")


def test_release_overlaps_a_dependent_without_changing_bytes():
    """The release with a dependent on the stream: the step built with ``pdl=True`` (release + wait) followed by launches that
    carry the programmatic-stream-serialization attribute, wait, and copy the step's ``codes`` and ``scales``
    (``pdl_dependent_copy.DependentCopy``).  The dependent may start while the step runs; its wait holds its reads.  Eager and
    captured in a CUDA graph, at 2 and 32 rows: the copied bytes are those of the plain step."""
    dev = _device_or_skip()
    from pdl_dependent_copy import DependentCopy

    copy = DependentCopy(dev)
    k = make_entry(dev, (True, None))
    assert k.cfg.PDL and k.cfg.PDL_RELEASE
    for rows in (2, 32):
        x = make_operands(rows, dev, seed=61 + rows)
        want = _snap(_entry(dev), x)
        k.compile(rows)
        copies = dict(codes=torch.zeros(rows, WIDTH, dtype=U8, device=dev), scales=torch.zeros(rows, 1, dtype=FP32, device=dev))

        def chain():
            codes, scales = k(x)
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
                for t in (*copies.values(), *k.outputs(rows).values()):  # the step's own outputs too: the replay must rewrite them
                    t.view(U8).fill_(0x33)
                graph.replay()
                torch.cuda.synchronize()
                _assert_same(copies, want, f"captured chain rows {rows}")
        finally:
            graph.reset()  # a test-owned graph is reset before the cell ends, also when it fails


def test_wait_holds_every_read_behind_a_predecessor_that_released_early():
    """The POSITION of the dependent-launch wait, on a device.  The stream predecessor is a launch that releases its dependents
    as its FIRST statement and only then writes ``x`` (``pdl_dependent_copy.EarlyReleaseWriter``: one thread block walks the
    slab in many passes, of which only the last leaves the written values).  The step's kernel therefore STARTS while ``x`` holds
    other bytes, and its wait is the only thing that holds its reads.

    The positive control comes first: a dependent WITHOUT a wait copies the slab's last row behind the same writer and must read
    other bytes than the written ones, launched eagerly and replayed from a CUDA graph.  Only behind that observation does a
    pass of the step show anything; where the early start does not reproduce at all, the cell SKIPS (a coverage gap, not a
    pass), and either outcome is one line of the run's "measured" section.

    Then the step, in the wait and the release + wait arm, eager and captured: codes and scales are bitwise those of the plain
    step on the written values.  A kernel that read a row in front of its wait fails here.  The eager launches are WARM ones: the
    first launch of a freshly compiled kernel has been observed to wait for the work in flight and would show nothing."""
    dev = _device_or_skip()
    import frost_routing
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter

    rows = 296
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
    key = "FP8 row quantization decode step, the wait behind an early-releasing predecessor"
    seen_text = f"the control (a dependent without a wait) read other bytes than the written ones in {eager} of 4 eager launches and {captured} of 4 replays"
    if eager + captured == 0:
        frost_routing.measured(key, f"NOT COMPARED: {seen_text} -- the early start did not reproduce on this device")
        pytest.skip(f"{seen_text}: the early start did not reproduce on this device (a coverage gap, not a pass)")
    want = _snap(_entry(dev), new)
    compared, differing = 0, []
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        k = _entry(dev, arm)
        k.compile()
        outs = _outs(rows, dev)

        def chain():
            writer(new, x)  # releases at its entry, then writes x
            k(x, **outs)  # may start at once; every read of x is behind its wait

        def restore():
            x.copy_(stale)
            outs["codes"].view(U8).fill_(0xA5)
            outs["scales"].fill_(-7.0)
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


def test_release_reaches_a_dependent_while_the_predecessor_still_runs():
    """``pdl=True`` is a fact of the LAUNCH as well as of the kernel: the launch carries the programmatic-stream-serialization
    attribute, so the step starts when its stream predecessor releases, and the step's own release -- its first statement -- lets
    ITS dependent start while that predecessor still runs.  On a device: a predecessor that releases first and only then writes a
    slab (``pdl_dependent_copy.EarlyReleaseWriter``), the release + wait step on another operand, then a dependent WITHOUT a
    wait that copies the slab's last row.  That copy reads other bytes than the written ones only when the step started before
    the predecessor finished and released in front of its own wait; behind a step launched without the attribute it starts
    after the predecessor and reads the written bytes every time.

    The positive control comes first (the same copy directly behind the writer); where the early start does not reproduce at
    all the cell SKIPS -- a coverage gap, not a pass.  Eager launches are warm ones, and the chain is also replayed from a CUDA
    graph.  The step's own bytes are those of the plain step throughout."""
    dev = _device_or_skip()
    import frost_routing
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter

    rows = 296
    new = make_operands(rows, dev, seed=77)
    stale = make_operands(rows, dev, seed=78)
    slab = stale.clone()
    x = make_operands(8, dev, seed=79)  # the step's own operand: nobody writes it
    want = _snap(_entry(dev), x)
    writer, unwaited = EarlyReleaseWriter(dev), DependentCopy(dev, wait=False)
    seen = torch.zeros(WIDTH, dtype=BF16, device=dev)
    k = _entry(dev, (True, None))
    k.compile()
    outs = _outs(8, dev)

    def control():
        writer(new, slab)
        unwaited(slab[-1], seen)  # no wait: it reads whatever the last row holds when it starts

    def chain():
        writer(new, slab)  # releases at its entry, then writes the slab
        k(x, **outs)  # starts at that release, releases as its first statement, then waits for the writer
        unwaited(slab[-1], seen)  # the STEP's dependent: it starts at the step's release

    def early_starts(run, runs=4):
        """How many runs copied other bytes than the written ones (the copy started in front of the writer's last pass)."""
        count = 0
        for _ in range(runs):
            slab.copy_(stale)
            seen.view(torch.int16).fill_(0)
            torch.cuda.synchronize()  # an idle stream: the writer is the only work in flight
            run()
            torch.cuda.synchronize()
            assert torch.equal(slab.view(torch.int16), new.view(torch.int16)), "the writer leaves the written bytes"
            count += int(not torch.equal(seen.view(torch.int16), new[-1].view(torch.int16)))
        return count

    def eager_and_captured(run):
        run()  # compiles on first use; the first launch of a compiled kernel does not start early
        torch.cuda.synchronize()
        eager = early_starts(run)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                run()
            captured = early_starts(graph.replay)
        finally:
            graph.reset()
        return eager, captured

    key = "FP8 row quantization decode step, the release in front of the wait reaches a dependent"
    controls = eager_and_captured(control)
    if sum(controls) == 0:
        frost_routing.measured(key, "NOT COMPARED: the control (a dependent without a wait behind the writer) never started early on this device")
        pytest.skip("the early start did not reproduce on this device (a coverage gap, not a pass)")
    through = eager_and_captured(chain)
    torch.cuda.synchronize()
    _assert_same(outs, want, "the step between the writer and the dependent")
    text = f"behind the step the dependent started early in {through[0]} of 4 eager launches and {through[1]} of 4 replays"
    text += f"; the control in {controls[0]} and {controls[1]}"
    frost_routing.measured(key, ("COMPARED: " if sum(through) else "FAILED: ") + text)
    assert sum(through) > 0, f"the step's release never reached its dependent while the predecessor ran ({text})"


def test_one_compile_serves_every_row_count_and_binds_no_operand(monkeypatch):
    """``compile()`` needs no operand and allocates no output; behind it every row count, every row stride and every output is a
    launch of the ONE compiled kernel (a second trace-compile would raise here)."""
    dev = _device_or_skip()
    import cutlass.cute as cute

    k = make_entry(dev, (True, None))
    k.compile()
    assert len(k.compiled) == 1 and not k.buffers, "compile() allocates no output"

    def forbidden(*args, **kwargs):
        raise AssertionError("a second trace-compile")

    monkeypatch.setattr(cute, "compile", forbidden)
    k.compile()
    k.compile(3)
    assert set(k.buffers) == {3}
    for rows in (1, 2, 3, 17, 296, 297, MAX_ROWS):
        x = make_operands(rows, dev, seed=70 + rows)
        got = _snap(k, x)
        assert_reference(got["codes"], got["scales"], x, label=f"one compile, rows {rows}")
    assert len(k.compiled) == 1 and len(k.kernel_symbols_for(1)) == 1


def test_compile_declines_a_dsl_below_the_floor_before_the_kernel_is_built(monkeypatch):
    dev = _device_or_skip()
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    k = make_entry(dev)
    with pytest.raises(NotImplementedError, match="found 4.7.1"):
        k.compile()
    with pytest.raises(NotImplementedError, match="found 4.7.1"):
        k(make_operands(1, dev))
    assert not k.compiled and not k.buffers


def test_graph_capture_replays_bitwise_with_eager():
    """Several launches captured into one CUDA graph after ``compile(rows)``: the replay is bitwise the eager run, and after the
    operands change IN PLACE the replay is bitwise a fresh eager run on the new values (every operand is a launch argument: the
    graph holds addresses, never values)."""
    dev = _device_or_skip()
    rows = 6
    arms = ((True, None), (True, False), (True, None), (None, None))
    entries = [_entry(dev, arm) for arm in arms]
    for k in entries:
        k.compile(rows)
    xs = [make_operands(rows, dev, seed=100 + i, kind="structured" if i == 1 else "random") for i in range(len(arms))]
    outs = [_outs(rows, dev) for _ in arms]

    def eager():
        return [_snap(k, x) for k, x in zip(entries, xs)]

    want = eager()
    graph = torch.cuda.CUDAGraph()

    def replay():
        for o in outs:
            o["codes"].view(U8).fill_(0x55)
            o["scales"].fill_(0.0)
        graph.replay()
        torch.cuda.synchronize()
        return outs

    try:
        with torch.cuda.graph(graph):
            for k, x, o in zip(entries, xs, outs):
                k(x, **o)
        for _ in range(3):
            for got, ref in zip(replay(), want):
                _assert_same(got, ref, "graph replay bitwise with eager")
        for i, x in enumerate(xs):  # new values in the same storage
            x.copy_(make_operands(rows, dev, seed=500 + i))
        got_all = [{n: t.clone() for n, t in got.items()} for got in replay()]
        for got, ref in zip(got_all, eager()):
            _assert_same(got, ref, "the replay reads the operands' current values")
    finally:
        graph.reset()


def test_warm_call_allocates_nothing_and_never_synchronizes():
    """The warm path: with caller-owned outputs and with entry-owned outputs of a row count that was already called, three calls
    allocate no device memory, and a call under ``set_sync_debug_mode('error')`` raises nothing (no device read reaches the
    host)."""
    dev = _device_or_skip()
    rows = 8
    k = make_entry(dev, (True, None))
    x = make_operands(rows, dev, seed=81)
    mine = _outs(rows, dev)
    want = _snap(k, x)
    k(x, **mine)
    k(x)
    torch.cuda.synchronize()
    k(x)  # the call that sees the pair's allocation complete drops its record
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(x, **mine)
            k(x)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    _assert_same(mine, want, "the warm calls' bytes")
    _assert_same(k.outputs(rows), want, "the warm calls' entry-owned bytes")


# The allocation of an entry-owned pair is stream work of the stream that allocated it (``cudnn.frost.init_order``): the entry
# does not initialize the pair, but torch fills it under deterministic algorithms and a recycled block can carry a queued write.
# The cells below warm an entry with CALLER-OWNED outputs (it has no pair yet), park a side stream behind a delay kernel, let
# ``outputs(rows)`` / ``compile(rows)`` allocate the pair on it and launch from the original stream while that work is still
# queued (``init_order_schedules`` says what every such cell must do).
def _warm_entry_without_a_pair(dev, rows, seed):
    """``(entry, x, want)``: the entry warmed by a real launch into caller-owned outputs, whose bytes -- held to the reference --
    are ``want``."""
    k = make_entry(dev)
    x = make_operands(rows, dev, seed=seed)
    want = _snap(k, x)
    assert_reference(want["codes"], want["scales"], x, label=f"rows {rows}: caller-owned outputs")
    assert not k.buffers, "the warm-up allocated an entry-owned pair"
    warm_fills(dev, E4M3, U8, FP32)
    return k, x, want


def _pair_blocks(dev, rows):
    return [torch.empty(rows, WIDTH, dtype=U8, device=dev), torch.empty(rows, 1, dtype=FP32, device=dev)]


def _park_the_pairs_allocation(dev, k, rows, fill, allocate=None):
    """``outputs(rows)`` (or ``allocate()``) on a side stream parked behind a delay, with stream work queued on the pair's blocks:
    the event that is pending while that work is."""
    recycled = []

    def work():
        if fill.startswith("a previous owner"):
            recycled.append(previous_owner(lambda: torch.empty(rows, WIDTH, dtype=U8, device=dev), 0xEE))
            k.outputs(rows) if allocate is None else allocate()
        else:
            with deterministic_fill():
                k.outputs(rows) if allocate is None else allocate()

    marker = park(pooled(dev, lambda: _pair_blocks(dev, rows)), work)
    if recycled and k.outputs(rows)["codes"].data_ptr() != recycled[0]:
        pytest.skip("the allocator did not hand the freed block to the entry: the schedule shows nothing")
    return marker


@pytest.mark.usefixtures("settled_records")
@pytest.mark.parametrize("outputs_reach_the_call", ("by default", "the pair passed back", "the codes passed back"))
@pytest.mark.parametrize("fill", ("the fill torch adds under deterministic algorithms", "a previous owner's fill of a recycled block"))
def test_warm_launch_waits_for_what_the_allocating_stream_still_owes_the_entry_owned_pair(fill, outputs_reach_the_call):
    """The launch that writes the entry-owned pair -- by default, or as the tensors ``outputs(rows)`` returned, passed back -- is
    ordered behind the stream work queued on the pair's blocks although the host does not wait; unordered, that work lands on
    the kernel's bytes."""
    dev = _device_or_skip()
    rows = 7
    k, x, want = _warm_entry_without_a_pair(dev, rows, 131)
    pending = _park_the_pairs_allocation(dev, k, rows, fill)
    pair = k.outputs(rows)
    if outputs_reach_the_call == "by default":
        codes, scales = k(x)
    elif outputs_reach_the_call == "the pair passed back":
        codes, scales = k(x, pair["codes"], pair["scales"])
    else:
        codes, scales = k(x, codes=pair["codes"])
    assert not pending.query(), "the call did not overlap the pending work (or it waited on the host)"
    torch.cuda.synchronize()
    assert codes is pair["codes"] and scales is pair["scales"]
    _assert_same(dict(codes=codes, scales=scales), want, f"the entry-owned pair behind {fill}, {outputs_reach_the_call}")


@pytest.mark.usefixtures("settled_records")
def test_pair_allocated_by_compile_is_ordered_like_one_allocated_by_outputs():
    """The entry's OTHER allocating route: ``compile(rows)`` on the parked side stream (under deterministic algorithms), then a
    call by default on the original stream."""
    dev = _device_or_skip()
    rows = 7
    k, x, want = _warm_entry_without_a_pair(dev, rows, 136)
    pending = _park_the_pairs_allocation(dev, k, rows, "the fill torch adds under deterministic algorithms", allocate=lambda: k.compile(rows))
    assert set(k.buffers) == {rows}, "compile(rows) did not allocate the pair"
    codes, scales = k(x)
    assert not pending.query(), "the call did not overlap the pending work (or it waited on the host)"
    torch.cuda.synchronize()
    _assert_same(dict(codes=codes, scales=scales), want, "the pair allocated by compile on another stream")


@pytest.mark.usefixtures("settled_records")
def test_warm_call_does_nothing_for_an_allocation_whose_stream_work_is_complete(monkeypatch):
    """A record is dropped by the first call that sees its event complete -- one with caller-owned outputs too -- and then the
    warm path is what it was: calls with the entry-owned pair and with caller-owned outputs allocate no device memory, raise
    nothing under ``set_sync_debug_mode('error')``, read or wait for no event and no stream, and do not enter the ordering step."""
    dev = _device_or_skip()
    rows = 8
    k, x, want = _warm_entry_without_a_pair(dev, rows, 135)
    pending = _park_the_pairs_allocation(dev, k, rows, "the fill torch adds under deterministic algorithms")
    codes, scales = k(x)
    assert not pending.query(), "the call did not overlap the pending work (or it waited on the host)"
    torch.cuda.synchronize()
    _assert_same(dict(codes=codes, scales=scales), want, "the call that met the pending fill")
    mine = _outs(rows, dev)
    k(x, **mine)  # caller-owned outputs: the call writes no entry-owned byte and still drops what is complete
    assert not k._pending, "a complete record was kept by a call that does not write its allocation"
    touched = []
    monkeypatch.setattr(init_order, "order_behind", lambda *a, **kw: touched.append("the ordering step"))
    for name in ("query", "synchronize", "wait"):
        monkeypatch.setattr(torch.cuda.Event, name, lambda *a, _name=name, **kw: touched.append(f"Event.{_name}"))
    for name in ("wait_event", "wait_stream", "synchronize"):
        monkeypatch.setattr(torch.cuda.Stream, name, lambda *a, _name=name, **kw: touched.append(f"Stream.{_name}"))
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(x)
            k(x, **mine)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    monkeypatch.undo()
    torch.cuda.synchronize()
    assert not touched, f"a warm call behind a complete allocation touched {sorted(set(touched))}"
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    _assert_same(dict(codes=codes, scales=scales), want, "the warm calls' entry-owned bytes")
    _assert_same(mine, want, "the warm calls' caller-owned bytes")


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(2, dev)
    want = _snap(k, x)
    sentinel = _outs(2, dev, codes_fill=0x77, scales_fill=-3.0)
    with pytest.raises(TypeError, match="x must be torch.bfloat16"):
        k(x.float(), **sentinel)
    with pytest.raises(ValueError, match="every operand must live on one device"):
        k(x.cpu(), **sentinel)
    with pytest.raises(ValueError, match="x must be"):
        k(x[:, :8192].contiguous(), **sentinel)
    with pytest.raises(TypeError, match="codes must be torch.float8_e4m3fn"):
        k(x, codes=sentinel["codes"].view(U8), scales=sentinel["scales"])
    with pytest.raises(ValueError, match="at least one row"):
        k(x[:0], **sentinel)
    with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows"):
        k(torch.zeros(1, WIDTH, dtype=BF16, device=dev).expand(MAX_ROWS + 1, WIDTH), **sentinel)
    torch.cuda.synchronize()
    _assert_untouched(sentinel, 0, "a declined warm call", codes_fill=0x77, scales_fill=-3.0)
    _assert_same(_snap(k, x), want, "the entry after the declines")


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
    codes, scales = k(x)
    torch.cuda.synchronize(other)
    assert torch.cuda.current_device() == current and codes.device == other and scales.device == other
    assert_reference(codes, scales, x, label="an entry on a non-current device")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
@pytest.mark.parametrize("rows", (1, 6, 300))
def test_launch_facts_equal_the_profilers_record_of_a_launch(rows, arm, arm_id):
    """One call is ONE kernel launch whose profiler record equals the entry's launch facts: the kernel's name is the entry's
    symbol of the compiled form -- which carries the form's text directly behind the kernel's name -- and grid, block and shared
    memory are the launch shape.  Before the first compile the entry names no symbol."""
    dev = _device_or_skip()
    k = make_entry(dev, arm)
    assert k.kernel_symbols_for(rows) == (), "nothing is compiled yet"
    x = make_operands(rows, dev, seed=95)
    outs = _outs(rows, dev)
    record = assert_facts_equal_the_launch(k, lambda: k(x, **outs), rows)
    head = SYMBOL + str(k.cfg)
    assert record.name.startswith(head + "_") and len(head) <= 100, record.name
    assert k.kernel_symbols_for(rows) == (record.name,) == k.kernel_symbols_for(MAX_ROWS), "one compiled kernel answers every row count"
    assert k.form_for(rows) == asdict(k.cfg)
