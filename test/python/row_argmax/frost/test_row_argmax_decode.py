# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The row argmax decode step (``cudnn.row_argmax.frost.row_argmax_decode``): on the host the config's limits and tables, every
typed decline of the entry (CPU tensors), the DSL gates, the compile key, the launch facts, the lazy export, the reference held
to the hand-written rule rows, the order of a launch behind an entry-owned allocation on stand-ins, and the kernel's RENDERING
for the claimed targets (the instructions the rule rests on); on a cc 10.x device every byte of the result against the reference
of ``row_argmax_decode_utils`` for the three table kinds, the rule rows, ``torch.argmax`` on rows without a NaN, strided slab
views, output ownership, one compile for every row count, CUDA-graph replay, the warm path's allocation / synchronization
discipline, a warm launch behind an allocation that is pending on another stream, and the profiler's record of a launch.

No cell arms a timer of its own: every device cell synchronizes under the suite's per-test watchdog (``CUDNN_TEST_TIMEOUT`` in
``test/python/conftest.py``).  A trip of that watchdog on a launch is a hang of the kernel's one thread-block barrier (a thread
that does not reach it) -- audit the barrier table in the kernel's docstring; never a larger limit."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from row_argmax_decode_utils import (  # noqa: E402  (the reference, the operand builders, the rule rows, the device gate)
    BF16,
    I32,
    I64,
    RESULT,
    TABLES,
    WIDTH,
    _device_or_skip,
    assert_rule,
    make_entry,
    make_rows,
    make_table,
    reference,
    reference_columns,
    result_bytes,
    rule_rows,
)

from cudnn.frost import init_order  # noqa: E402
from cudnn.row_argmax.frost import row_argmax_decode as entry  # noqa: E402
from cudnn.row_argmax.frost.kernel import row_argmax_decode_config as kcfg  # noqa: E402
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

ROWS = (1, 2, 3, 8, 17, 64, 257)
SYMBOL = "cudnn_kernel_frost_row_argmax_"
THREADS = 1024
SHARED_MEMORY = 256


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_config_constants_and_tables():
    """The limits, the derived launch quantities and the two tables: ONE thread-block barrier that every thread issues once, ONE
    shared-memory buffer that is written once."""
    assert (kcfg.VEC, kcfg.WARP, kcfg.THREADS, kcfg.MAX_PASSES, kcfg.MAX_WIDTH, kcfg.PASS_WIDTH, kcfg.MAX_ROWS) == (8, 32, 1024, 8, 65536, 8192, 1024)
    assert kcfg.SERVED_WIDTH == (WIDTH,) and kcfg.TABLE_KINDS == TABLES
    assert all(w % kcfg.PASS_WIDTH == 0 and kcfg.validate_width(w) == w for w in kcfg.SERVED_WIDTH), "a served width is whole passes"
    assert kcfg.RESULT_DTYPE == dict(none="int32", int32="int32", int64="int64") and kcfg.INT_BYTES == dict(int32=4, int64=8)
    cfg = entry.make_cfg(WIDTH)
    assert (cfg.THREADS, cfg.WARPS, cfg.PASSES, cfg.SMEM_WORDS, cfg.SMEM_BYTES, cfg.CTAS_PER_SM) == (THREADS, 32, 8, 64, SHARED_MEMORY, 2)
    assert cfg.THREADS == kcfg.WARP * cfg.WARPS and cfg.PASSES * cfg.THREADS * kcfg.VEC == WIDTH, "full warps; every column is one thread's"
    ledger = kcfg.barrier_ledger(cfg)
    assert ledger == dict(participants=THREADS, issuing_lanes=THREADS), "SUM(issuing lanes) == the barrier's participants"
    (buffer,) = kcfg.smem_table(cfg)
    assert buffer["bytes"] == SHARED_MEMORY and buffer["elements"] == 64 and buffer["lane_stride_bytes"] == 4 and buffer["refilled"] is False
    assert str(cfg) == "w65536" and repr(cfg) == "Cfg(WIDTH=65536)" and len(SYMBOL + str(cfg) + "_int64") <= 100


@pytest.mark.parametrize(
    "width, error, token",
    [
        ("65536", TypeError, "width"),
        (65536.0, TypeError, "width"),
        (None, TypeError, "width"),
        (True, TypeError, "width"),
        (0, ValueError, "width"),
        (-8, ValueError, "width"),
        (4, ValueError, "width"),
        (65535, ValueError, "width"),
        (65544, ValueError, "width"),
        (8, NotImplementedError, "width"),
        (8192, NotImplementedError, "width"),
        (32768, NotImplementedError, "width"),
        (65528, NotImplementedError, "width"),
    ],
)
def test_construction_declines_a_width_by_type(width, error, token):
    """A width that is not an ``int``; one outside the kernel's structure (a multiple of 8 in 8 .. 65536); a structurally valid
    one the entry does not serve -- from the constructor and from ``make_cfg``, without a device."""
    with pytest.raises(error, match=token):
        entry.make_cfg(width)
    with pytest.raises(error, match=token):
        entry.RowArgmaxDecode("cpu", width)


def test_the_kernel_body_refuses_a_config_it_does_not_have():
    """A config of a structurally valid width that is not whole passes of the thread block never reaches a kernel: the config's
    validator declines it, as does an unknown table kind."""
    for width in (8, 4096, 8200, 65528):
        with pytest.raises(NotImplementedError, match="whole passes"):
            kcfg.validate_cfg(kcfg.Cfg(WIDTH=width))
    with pytest.raises(TypeError, match="cfg"):
        kcfg.validate_cfg(dict(WIDTH=WIDTH))
    for kind in ("uint32", "INT64", None, 64):
        with pytest.raises(ValueError, match="table kind"):
            kcfg.validate_table_kind(kind)
    assert [kcfg.validate_table_kind(kind) for kind in TABLES] == list(TABLES)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the binding checks (CPU tensors; no device, no DSL)
# ---------------------------------------------------------------------------------------------------------------------------
def _host_operands(rows=3, table="int64", with_out=True):
    x = torch.zeros(rows, WIDTH, dtype=BF16)
    ids = None if table == "none" else torch.zeros(WIDTH, dtype=RESULT[table])
    out = torch.zeros(rows + 2, dtype=RESULT[table]) if with_out else None
    return dict(x=x, ids=ids, out=out)


def _check(ops, **kw):
    return entry.check_support(entry.make_cfg(WIDTH), ops["x"], ops["ids"], ops["out"], **kw)


@pytest.mark.parametrize("table", TABLES)
def test_served_operands_pass_the_host_checks(table):
    """Every table kind, with and without ``out``, at the row limits, as a strided view of a wider slab and as an expanded row:
    the checks return the row count and name the call's table kind."""
    for rows in (1, 3, kcfg.MAX_ROWS):
        x = torch.zeros(1, WIDTH, dtype=BF16).expand(rows, WIDTH)  # one stored row: a host tensor of the ceiling's shape for free
        ids = _host_operands(1, table)["ids"]
        for out in (None, torch.zeros(rows, dtype=RESULT[table]), torch.zeros(rows + 5, dtype=RESULT[table])):
            assert entry.check_support(entry.make_cfg(WIDTH), x, ids, out, device_cc=(10, 0)) == rows
        assert entry.table_kind(ids) == table
    slab = torch.zeros(4, WIDTH + 24, dtype=BF16)
    assert _check(dict(x=slab[:, :WIDTH], ids=None, out=None), device_cc=(10, 3)) == 4
    assert _check(dict(x=slab[1:3, 16 : WIDTH + 16], ids=None, out=None), device_cc=(10, 7)) == 2


def _misaligned(dtype, elements):
    """A 1-D tensor of ``dtype`` whose first element sits half an element behind an address aligned to the element."""
    size = torch.empty((), dtype=dtype).element_size()
    for offset in (size // 2, size // 2 + size):
        t = torch.frombuffer(bytearray(size * (elements + 4)), dtype=dtype, count=elements, offset=offset)
        if t.data_ptr() % size:
            return t
    raise AssertionError("no misaligned view could be built on this host")


def _mutations():
    """One mutation per decline of the binding checks: ``name -> (operands, error, the operand the message names)``."""
    cases = {}

    def add(name, error, token, **changed):
        cases[name] = (dict(_host_operands(), **changed), error, token)

    wide = torch.zeros(3, 2 * WIDTH, dtype=BF16)
    slab = torch.zeros(3, WIDTH + 8, dtype=BF16)
    add("x is not a tensor", TypeError, "x", x=[[0.0] * 8])
    add("ids is not a tensor", TypeError, "ids", ids=list(range(8)))
    add("out is not a tensor", TypeError, "out", out=[0, 0, 0])
    add("x is fp16", TypeError, "x", x=torch.zeros(3, WIDTH, dtype=torch.float16))
    add("x is fp32", TypeError, "x", x=torch.zeros(3, WIDTH))
    add("ids is int16", TypeError, "ids", ids=torch.zeros(WIDTH, dtype=torch.int16))
    add("ids is fp32", TypeError, "ids", ids=torch.zeros(WIDTH))
    add("ids is uint8", TypeError, "ids", ids=torch.zeros(WIDTH, dtype=torch.uint8))
    add("out is int32 beside an int64 table", TypeError, "out", out=torch.zeros(5, dtype=I32))
    add("out is int64 without a table", TypeError, "out", ids=None, out=torch.zeros(5, dtype=I64))
    add("out is int64 beside an int32 table", TypeError, "out", ids=torch.zeros(WIDTH, dtype=I32))
    add("x has three axes", ValueError, "x", x=torch.zeros(1, 3, WIDTH, dtype=BF16))
    add("x has one axis", ValueError, "x", x=torch.zeros(WIDTH, dtype=BF16))
    add("x has another width", ValueError, "x", x=torch.zeros(3, WIDTH // 2, dtype=BF16))
    add("an empty call", ValueError, "x", x=torch.zeros(0, WIDTH, dtype=BF16))
    add("ids holds another count", ValueError, "ids", ids=torch.zeros(WIDTH - 1, dtype=I64))
    add("ids has two axes", ValueError, "ids", ids=torch.zeros(1, WIDTH, dtype=I64))
    add("out is shorter than the call", ValueError, "out", out=torch.zeros(2, dtype=I64))
    add("out has two axes", ValueError, "out", out=torch.zeros(3, 1, dtype=I64))
    add("more rows than the ceiling", NotImplementedError, "rows", x=torch.zeros(1, WIDTH, dtype=BF16).expand(kcfg.MAX_ROWS + 1, WIDTH), out=None)
    add("x steps along a row", NotImplementedError, "x", x=wide[:, ::2])
    add("ids steps", NotImplementedError, "ids", ids=torch.zeros(2 * WIDTH, dtype=I64)[::2])
    add("out steps", NotImplementedError, "out", out=torch.zeros(10, dtype=I64)[::2])
    add("a row stride that is not whole vectors", ValueError, "x", x=torch.zeros(3, WIDTH + 4, dtype=BF16)[:, :WIDTH])
    add("x off a 16-byte address", ValueError, "x", x=slab[:, 4 : WIDTH + 4])
    add("ids off its element's alignment", ValueError, "ids", ids=_misaligned(I64, WIDTH))
    add("out off its element's alignment", ValueError, "out", out=_misaligned(I64, 5))
    table = torch.zeros(WIDTH, dtype=I64)
    add("out over the table", ValueError, "out", ids=table, out=table[100:110])
    x = torch.zeros(3, WIDTH, dtype=BF16)
    add("out over x", ValueError, "out", x=x, out=x.view(torch.uint8).reshape(-1)[4096 : 4096 + 40].view(I64))
    add("ids on another device", ValueError, "ids", ids=torch.zeros(WIDTH, dtype=I64, device="meta"))
    add("out on another device", ValueError, "out", out=torch.zeros(5, dtype=I64, device="meta"))
    return cases


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_typed_declines_name_the_operand(name):
    """One mutation of a served call per decline: the type of the decline and the operand it names, from the support check and
    from the binding check a warm launch re-runs."""
    ops, error, token = _mutations()[name]
    with pytest.raises(error, match=rf"\b{token}\b"):
        _check(ops, device_cc=(10, 0))
    with pytest.raises(error, match=rf"\b{token}\b"):
        entry.check_binding(entry.make_cfg(WIDTH), ops["x"], ops["ids"], ops["out"])


def test_binding_check_requires_every_operand_on_the_given_device():
    ops = _host_operands()
    with pytest.raises(ValueError, match="one device"):
        entry.check_binding(entry.make_cfg(WIDTH), ops["x"], ops["ids"], ops["out"], device="meta")
    assert entry.check_binding(entry.make_cfg(WIDTH), ops["x"], ops["ids"], ops["out"], device="cpu") == 3


def test_declines_an_unserved_compute_capability_and_host_operands():
    ops = _host_operands()
    for cc in ((9, 0), (8, 0), (8, 9), (11, 0), (12, 0), (12, 1)):
        with pytest.raises(NotImplementedError, match="compute capability"):
            _check(ops, device_cc=cc)
        assert not entry.cc_supported(cc)
    assert all(entry.cc_supported(cc) for cc in ((10, 0), (10, 3), (10, 7))) and not entry.cc_supported(None)
    with pytest.raises(NotImplementedError, match="CUDA tensors"):
        _check(ops)


def test_compile_declines_an_unserved_part_and_a_missing_dsl_before_anything_is_built(monkeypatch):
    """The declines a CALLER reaches, on MODELLED part facts (an entry for ``cuda:0`` with the device's compute capability
    patched: no device is touched): ``compile()``, ``compile(rows)`` and a first call on an unserved part decline by type in
    front of the DSL gates; a served part reaches the DSL gate (modelled as missing).  Nothing is compiled or allocated."""
    from cudnn.frost import buffers

    x = on_device(torch.device("cuda", 0), torch.zeros(3, WIDTH, dtype=BF16))[0]
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    for cc, token in (
        ((12, 0), "compute capability"),
        ((9, 0), "compute capability"),
        ((8, 0), "compute capability"),
        ((10, 0), "not installed"),
        ((10, 7), "not installed"),
    ):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        k = make_entry("cuda:0")
        for attempt in (lambda: k.compile(), lambda: k.compile(3), lambda: k(x)):
            with pytest.raises(NotImplementedError, match=token):
                attempt()
        assert not k.compiled and not k.buffers, "a declined compile builds and allocates nothing"
    host = make_entry("cpu")
    with pytest.raises(NotImplementedError, match="CUDA devices"):
        host.compile()
    for rows in (0, kcfg.MAX_ROWS + 1):
        with pytest.raises((ValueError, NotImplementedError), match="rows"):
            host.compile(rows)


def test_compile_key_holds_the_config_and_the_table_kind():
    """Plan-time facts only: one key per table kind whatever the row count."""
    k = make_entry("cpu")
    assert [k.key(table) for table in TABLES] == [(k.cfg, table) for table in TABLES] and k.key() == (k.cfg, "none")
    assert all(k.cfg_for(rows) is k.cfg for rows in (1, 17, kcfg.MAX_ROWS))
    with pytest.raises(ValueError, match="table kind"):
        k.key("int16")
    with pytest.raises(NotImplementedError, match="rows"):
        k.cfg_for(kcfg.MAX_ROWS + 1)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the launch facts (cudnn.frost.launch_facts)
# ---------------------------------------------------------------------------------------------------------------------------
class _CompiledStandIn:
    """What the entry keeps per compile key, as far as the launch facts read it: a compiled DSL host function carries
    ``kernel_info``, keyed by the symbols of the kernels it launches."""

    def __init__(self, *symbols):
        self.kernel_info = {symbol: {} for symbol in symbols}


def test_launch_facts_equal_the_configuration_arithmetic_at_every_row_count():
    """EVERY served row count: one thread block per row, 1024 threads, the one 256-byte buffer; the form is the config's field."""
    k = make_entry("cpu")
    for rows in range(1, kcfg.MAX_ROWS + 1):
        shape = k.launch_shape_for(rows)
        assert shape == ((rows, 1, 1), (THREADS, 1, 1), SHARED_MEMORY), (rows, shape)
        assert all(type(v) is int for v in (*shape.grid, *shape.block, shape.shared_memory)), shape
        assert k.form_for(rows) == asdict(k.cfg) == dict(WIDTH=WIDTH), rows
    assert k.kernel_symbols_for(1) == () == k.kernel_symbols_for(kcfg.MAX_ROWS), "nothing is compiled: no symbol"


def test_launch_facts_are_plain_fresh_and_decline_an_unserved_row_count():
    k = make_entry("cpu")
    form = k.form_for(7)
    assert json.loads(json.dumps(form)) == form and type(form) is dict
    form["WIDTH"] = "edited"
    assert k.form_for(7)["WIDTH"] == WIDTH and k.cfg.WIDTH == WIDTH
    for fact in (k.kernel_symbols_for, k.form_for, k.launch_shape_for):
        for rows in (0, -1):
            with pytest.raises(ValueError, match="rows"):
                fact(rows)
        with pytest.raises(NotImplementedError, match="rows"):
            fact(kcfg.MAX_ROWS + 1)
        for rows in (1.0, "1", None, True):
            with pytest.raises(ValueError, match="rows"):
                fact(rows)
        fact(kcfg.MAX_ROWS)


def test_kernel_symbols_are_those_of_the_compiled_table_kinds_in_compile_order():
    k = make_entry("cpu")
    k.compiled[k.key("int64")] = _CompiledStandIn(SYMBOL + "w65536_int64_0")
    assert k.kernel_symbols_for(1) == (SYMBOL + "w65536_int64_0",) == k.kernel_symbols_for(kcfg.MAX_ROWS)
    k.compiled[k.key("none")] = _CompiledStandIn(SYMBOL + "w65536_none_0")
    assert k.kernel_symbols_for(5) == (SYMBOL + "w65536_int64_0", SYMBOL + "w65536_none_0")
    k.compiled[k.key("int32")] = object()
    with pytest.raises(NotImplementedError, match="kernel symbols"):
        k.kernel_symbols_for(1)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the DSL gates (installed -> version floor -> the part's target -> the compiler that assembles it)
# ---------------------------------------------------------------------------------------------------------------------------
def test_dsl_gates_decline_by_version_by_target_and_by_compiler(monkeypatch):
    """A missing DSL and one below the floor name the floor; with the version gate passed a DSL without the part's target
    declines cc 10.7 and serves cc 10.0 / 10.3; with the target present a compiler of the CUDA 12 toolchain flavour declines
    cc 10.7 before JIT, naming the compiler found.  The target probe (the first gate that imports the DSL) is not reached under
    a DSL that the version gate declines."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    assert "not installed" in entry._dsl_error((10, 3)) and "4.8.0" in entry._dsl_version_error()

    def unreachable():
        raise AssertionError("the target probe ran although the version gate declined")

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", unreachable)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    for cc in ((10, 0), (10, 3), (10, 7)):
        assert "4.8.0" in entry._dsl_error(cc) and "4.7.1" in entry._dsl_error(cc)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    assert "sm_107a" in entry._dsl_error((10, 7)) and entry._dsl_error((10, 7)).startswith(entry.NAME)
    assert entry._dsl_error((10, 3)) is None and entry._dsl_error((10, 0)) is None
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda: (12, 9))
    assert "12.9" in entry._dsl_error((10, 7)) and "cu13" in entry._dsl_error((10, 7))
    assert entry._dsl_error((10, 3)) is None and entry._dsl_error((10, 0)) is None
    for served in (entry.SM107_MIN_COMPILER, (13, 6), (14, 0)):
        monkeypatch.setattr(entry, "dsl_compiler_cuda_version", lambda v=served: v)
        assert entry._dsl_error((10, 7)) is None


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: exports and the import boundary
# ---------------------------------------------------------------------------------------------------------------------------
def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.RowArgmaxDecode is entry.RowArgmaxDecode
    family, frost = importlib.import_module("cudnn.row_argmax"), importlib.import_module("cudnn.row_argmax.frost")
    assert family.RowArgmaxDecode is entry.RowArgmaxDecode and frost.RowArgmaxDecode is entry.RowArgmaxDecode
    assert entry.NAME == kcfg.NAME and set(entry.__all__) >= {"RowArgmaxDecode", "check_binding", "check_support", "cc_supported", "make_cfg", "table_kind"}
    for module in (family, frost):
        with pytest.raises(AttributeError):
            module.NoSuchName


def test_import_cudnn_does_not_import_the_kernel_or_a_heavy_module():
    """The lazy boundary, in fresh interpreters, as a DELTA against an empty interpreter (a distribution can load a package at
    start-up): importing cudnn loads none of the family's modules, no tensor framework and no DSL; the family packages, the
    config and the host entry load neither a framework nor the kernel module; constructing a host entry (its declines included)
    still loads neither the DSL nor the kernel module."""
    heavy = ("torch", "jax", "jaxlib", "cutlass", "nvidia_cutlass_dsl")

    def modules(code):
        probe = code + "\nimport json, sys; print('@@' + json.dumps(sorted(sys.modules)))"
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
        return set(json.loads(next(ln for ln in out.splitlines() if ln.startswith("@@"))[2:]))

    def added(code):
        return modules(code) - modules("")

    def unwanted(names, frameworks=heavy):
        return sorted(m for m in names if m.split(".")[0] in frameworks or m.endswith("row_argmax_decode_f16"))

    by_cudnn = added("import cudnn")
    assert not unwanted(by_cudnn) and not [m for m in by_cudnn if m.startswith("cudnn.row_argmax")], sorted(by_cudnn)
    family = "cudnn.row_argmax, cudnn.row_argmax.frost, cudnn.row_argmax.frost.kernel.row_argmax_decode_config, cudnn.row_argmax.frost.row_argmax_decode"
    assert not unwanted(added(f"import {family}"))
    construct = (
        "import cudnn\n"
        "k = cudnn.RowArgmaxDecode('cpu', 65536)\n"
        "try:\n"
        "    cudnn.RowArgmaxDecode('cpu', 8192)\n"
        "except NotImplementedError:\n"
        "    pass\n"
        "k.form_for(3), k.launch_shape_for(3), k.kernel_symbols_for(3)\n"
    )
    assert not unwanted(added(construct), frameworks=("cutlass",))


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the reference is the rule
# ---------------------------------------------------------------------------------------------------------------------------
def test_reference_gives_the_hand_written_column_of_every_rule_row():
    """The reference of the helpers against the columns written by hand beside each sentence of the rule, and against
    ``torch.argmax`` on the rows that hold no NaN; with a table the result is the table's element at that column."""
    names, x, columns = rule_rows()
    got = reference_columns(x)
    wrong = [f"{name}: {int(a)} != {int(b)}" for name, a, b in zip(names, got, columns) if int(a) != int(b)]
    assert not wrong, wrong
    values = x.float()
    plain = ~torch.isnan(values).any(dim=1)
    assert int(plain.sum()) >= 15 and torch.equal(values[plain].argmax(dim=1), columns[plain]), "torch.argmax on the rows without a NaN"
    for table in TABLES:
        ids = make_table(table)
        want = columns.to(I32) if ids is None else ids[columns]
        assert torch.equal(reference(x, ids), want) and reference(x, ids).dtype == RESULT[table]
    for kind in ("random", "ties", "special"):
        x = make_rows(24, seed=3, kind=kind)
        values = x.float()
        plain = ~torch.isnan(values).any(dim=1)
        assert torch.equal(values[plain].argmax(dim=1), reference_columns(x)[plain]), kind


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the order of a launch behind an entry-owned allocation, on stand-ins (cudnn.frost.init_order)
# ---------------------------------------------------------------------------------------------------------------------------
def test_the_entry_owned_pair_is_recorded_before_a_launch_can_find_it(monkeypatch):
    """The record of a pair's allocation, on stand-ins (no device): the two vectors are allocated and only THEN an event is
    recorded on the stream that is current at that moment, with both vectors' byte ranges, BEFORE the entry holds the pair.  A
    second ``outputs`` allocates and records nothing and returns a copy of the entry's table.  The records are the device's,
    one object for every entry.  What a launch is ordered by is the first ``rows`` results of its resolved ``out``."""
    s = stand_ins(monkeypatch)
    dev = torch.device("cuda", 5)
    host_allocations(monkeypatch, dev, s.calls)
    k = make_entry(dev)
    assert k._pending is init_order.pending_of(dev) and make_entry(dev)._pending is k._pending
    held, real = [], init_order.record
    monkeypatch.setattr(init_order, "record", lambda pending, tensors: (held.append(dict(k.buffers)), real(pending, tensors)))
    pair = k.outputs(3)
    assert s.calls == [("empty", (3,)), ("empty", (3,)), ("enter", dev), ("record", 100), ("leave", dev)]
    assert held == [{}], "the record is made before the entry holds the pair"
    ((_, handle, spans),) = k._pending.values()
    assert handle == 100 and spans == spans_of(pair.values()) and {name: t.dtype for name, t in pair.items()} == dict(int32=I32, int64=I64)
    s.calls.clear()
    again = k.outputs(3)
    assert again is not pair and all(again[name] is pair[name] for name in pair) and not s.calls and len(k._pending) == 1
    mine = torch.zeros(9, dtype=I64)
    assert k._launch_writes(3, mine) == ((mine.data_ptr(), mine.data_ptr() + 3 * 8),)


@pytest.mark.parametrize("table", TABLES)
def test_a_call_orders_its_launch_behind_a_pending_record_over_the_out_it_writes(monkeypatch, table):
    """The wiring, on stand-ins with the launch intercepted (no device): IN FRONT of the launch and on the LAUNCH stream, a
    call is ordered behind a pending record over the bytes of the ``out`` it writes -- passed, or the entry's vector of the
    result's dtype -- and behind none over an operand it only reads or over the entry's vector of the OTHER dtype; without a
    record nothing but the launch happens.  The launch gets the operands' addresses, the row stride in bytes and the row count."""
    s = stand_ins(monkeypatch)
    k = make_entry("cpu")
    k.compiled[k.key(table)] = lambda *args: s.calls.append(("launch", tuple(int(a) for a in args)))
    ops = _host_operands(table=table)
    x, ids = ops["x"], ops["ids"]

    def ordering_of_a_call(over=None, out=None):
        k._pending.clear()
        event = s.Event(False)
        if over is not None:
            k._pending[0] = (event, 7, spans_of((over,)))
        s.calls.clear()
        got = k(x, ids, out)
        written = k.outputs(3)[kcfg.RESULT_DTYPE[table]] if out is None else out
        assert got is written
        launch = ("launch", (x.data_ptr(), 2 * WIDTH, 0 if ids is None else ids.data_ptr(), written.data_ptr(), 3, 100))
        return [c for c in s.calls if c[0] in ("wait_event", "launch")], [("wait_event", 100, event), launch], [launch]

    got, ordered, _ = ordering_of_a_call(ops["out"], ops["out"])
    assert got == ordered, "a record over the passed out"
    got, ordered, _ = ordering_of_a_call(k.outputs(3)[kcfg.RESULT_DTYPE[table]])
    assert got == ordered, "a record over the entry-owned vector the call writes"
    other = "int64" if kcfg.RESULT_DTYPE[table] == "int32" else "int32"
    for name, read_only in (("x", x), ("ids", ids), ("the entry's vector of the other dtype", k.outputs(3)[other])):
        if read_only is not None:
            got, _, alone = ordering_of_a_call(read_only)
            assert got == alone, f"{name} is not written by the call"
    got, _, alone = ordering_of_a_call()
    assert got == alone == s.calls, "without a record: the launch alone"


def test_a_declined_call_compile_or_outputs_allocates_nothing_and_leaves_no_record(monkeypatch):
    """A decline touches nothing, on stand-ins (no device).  Whatever declines a call, a ``compile`` or an ``outputs`` -- a part
    the kernel does not serve, a DSL gate, a binding check, the row count, a host device -- runs IN FRONT of the pair's
    allocation: the entry holds no pair, nothing was allocated and the device has no record.  A ``compile(rows)`` that is not
    declined allocates the pair and records it."""
    from cudnn.frost import buffers

    s = stand_ins(monkeypatch)
    dev, part = torch.device("cuda", 5), [(10, 0)]
    host_allocations(monkeypatch, dev, s.calls)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: part[0])
    row = torch.zeros(1, WIDTH, dtype=BF16)
    x, tall, fp32 = on_device(dev, row.expand(3, WIDTH), row.expand(kcfg.MAX_ROWS + 1, WIDTH), torch.zeros(1, WIDTH).expand(3, WIDTH))
    missing, old = (False, None), (True, ("nvidia-cutlass-dsl", "4.7.1"))
    declines = (  # what declines: the entry's device, the part, the DSL state, the operand, the decline
        ("a part the kernel does not serve", dev, (8, 0), missing, x, NotImplementedError, "compute capability"),
        ("a missing DSL", dev, (10, 0), missing, x, NotImplementedError, "not installed"),
        ("a DSL below the floor", dev, (10, 3), old, x, NotImplementedError, "4.7.1"),
        ("the row count", dev, (10, 0), missing, tall, NotImplementedError, "rows"),
        ("a dtype", dev, (10, 0), missing, fp32, TypeError, r"\bx\b"),
        ("a host device", "cpu", (10, 0), missing, row, NotImplementedError, "CUDA devices"),
    )
    for what, device, cc, dsl, operand, error, token in declines:
        part[0] = cc
        monkeypatch.setattr(buffers, "_DSL_STATE", dsl)
        for route in ("a call", "a compile"):
            if route == "a compile" and what in ("the row count", "a dtype"):
                continue  # a compile takes no operand
            k = make_entry(device)
            s.calls.clear()
            with pytest.raises(error, match=token):
                k(operand) if route == "a call" else k.compile(3)
            assert not k.buffers and not s.calls and not init_order.pending_of(dev), f"{route} declined by {what} allocated or recorded"
    k = make_entry(dev)
    for attempt in (k.outputs, k.compile):
        with pytest.raises(NotImplementedError, match="rows"):
            attempt(kcfg.MAX_ROWS + 1)
    assert not k.buffers and not s.calls and not k._pending, "a row count that is declined allocated or recorded"
    k.compiled.update({k.key(table): None for table in TABLES})  # a compiled entry: nothing is missing, so no DSL gate is asked
    k.compile(3)
    assert [c[0] for c in s.calls] == ["empty", "empty", "enter", "record", "leave"] and len(k._pending) == 1 and set(k.buffers) == {3}


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the rendering (the instructions the rule rests on), per claimed target
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
    from cudnn.row_argmax.frost import row_argmax_decode as entry
    from cudnn.row_argmax.frost.kernel import row_argmax_decode_f16 as mod

    print("COMPILER", int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    for table in sys.argv[3:]:
        before = set(os.listdir(dump))
        try:
            cute.compile(mod.launch, entry.make_cfg(65536), table, *[cutlass.Int64(0)] * 4, cutlass.Int32(1), make_fake_stream())
        except Exception as err:  # a compiler flavour without the target: reported, the cell decides
            print("REJECTED", table, " ".join(str(err).split())[:300])
            continue
        for name in sorted(set(os.listdir(dump)) - before):
            print("KEPT", table, name)
    print("COMPILED", arch)
    """)
_UNSERVED = "Unsupported GPU architecture"  # the DSL's message when its compiler flavour has no such target


def _render(tmp_path, arch):
    """Trace-compile the three table kinds for ``arch`` in a fresh process (the DSL reads its dump settings once, at the first
    import; no device is needed: the kernel's operands are addresses).  Returns the compiler's ``(major, minor)``,
    ``{table: (ptx text, cubin path)}`` and ``{table: rejection text}``."""
    probe = tmp_path / f"render_probe_{arch}.py"  # a FILE, not ``python -c``: the DSL preprocessor re-reads a kernel's source through inspect
    probe.write_text(_RENDER_PROBE)
    dump = tmp_path / f"dump_{arch}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, str(probe), str(dump), arch, *TABLES], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"trace-compile for {arch} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED {arch}" in lines, proc.stdout[-2000:]
    compiler = tuple(int(v) for v in next(ln for ln in lines if ln.startswith("COMPILER")).split()[1:3])
    kept, rejected = {}, {}
    for ln in lines:
        if ln.startswith("KEPT "):
            _, table, name = ln.split(" ", 2)
            kept.setdefault(table, {})[name.rsplit(".", 1)[-1]] = str(dump / name)
        if ln.startswith("REJECTED "):
            _, table, text = ln.split(" ", 2)
            rejected[table] = text
    return compiler, {table: (open(files["ptx"]).read(), files["cubin"]) for table, files in kept.items()}, rejected


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


def _sass(cubin):
    """The disassembly of a cubin by the first nvdisasm that decodes it, or None (a toolkit older than the target decodes nothing)."""
    candidates = [os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm")] if os.environ.get("CUDA_PATH") else []
    candidates.append(shutil.which("nvdisasm") or "")
    for nvdisasm in dict.fromkeys(c for c in candidates if c and os.access(c, os.X_OK)):
        try:
            proc = subprocess.run([nvdisasm, "-c", cubin], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.splitlines()
    return None


@pytest.mark.parametrize("arch, cc", (("sm_100a", (10, 0)), ("sm_103a", (10, 3)), ("sm_107a", (10, 7)), ("sm_120a", None)))
def test_rendering_holds_the_rule(arch, cc, tmp_path):
    """Trace-compile the kernel for a target and hold, per table kind, the instructions the rule rests on.  The scan: eight
    128-bit global loads and 64 ORDERED compares that are strict or the compiler's fold of the first one against -inf (``gt``,
    ``ne``) -- never an unordered or a non-strict compare, never a floating-point maximum in place of a compare and a select on
    a target whose warp maximum is one reduction instruction.  The reduction: ONE thread-block barrier, two shared stores in
    front of it and two shared loads behind it, two integer warp reductions; the warp maximum as two reduction instructions on a
    compute-capability 10.x target and as butterfly shuffles elsewhere.  The result: the table kind's one load and one store.
    No dependent-launch instruction, no mbarrier, no copy group; every branch goes forward (no loop).  1024 threads; the form
    inside the first 100 characters of a cuDNN symbol.  Where a disassembler decodes the cubin: no access to local memory (every
    value stays in a register).  ``sm_120a`` is rendered although the entry declines that class: the kernel assembles for it."""
    compiler, kept, rejected = _render(tmp_path, arch)
    if rejected:
        assert cc == (10, 7) and compiler < entry.SM107_MIN_COMPILER and all(_UNSERVED in text for text in rejected.values()), rejected
        monkey = pytest.MonkeyPatch()
        with monkey.context() as patched:
            patched.setattr(entry, "dsl_compiler_cuda_version", lambda: compiler)
            assert entry._dsl_compiler_error(cc) is not None, "a target the compiler rejects is one the entry declines"
        pytest.skip(f"the DSL's compiler (CUDA {compiler[0]}.{compiler[1]}) rejects {arch}; the entry declines the part by name")
    assert set(kept) == set(TABLES)
    redux = cc is not None
    for table in TABLES:
        ptx, cubin = kept[table]
        ops = [tokens[0] for tokens in _statements(ptx)]
        count = ops.count
        what = f"{arch}, table kind {table}"
        symbol = ptx[ptx.index(".entry") :].split("(")[0].split()[-1]
        assert symbol.startswith(f"{SYMBOL}w{WIDTH}_{table}") and len(f"{SYMBOL}w{WIDTH}_{table}") <= 100, symbol
        assert ".reqntid 1024" in ptx, what
        assert count("ld.global.v4.b32") == 8, what
        compares = [op for op in ops if op.startswith("setp.") and op.endswith(".f32")]
        assert compares.count("setp.eq.f32") == 2 and len(compares) == 66 and set(compares) <= {"setp.gt.f32", "setp.ne.f32", "setp.eq.f32"}, (
            what,
            sorted(set(compares)),
        )
        assert compares.count("setp.ne.f32") <= 1, what
        assert count("bar.sync") == 1 and count("st.shared.b32") == 2 and count("ld.shared.b32") == 2, what
        assert count("redux.sync.min.s32") == 2, what
        if redux:
            assert count("redux.sync.max.f32") == 2 and count("max.f32") == 0 and count("shfl.sync.bfly.b32") == 0, what
        else:
            assert count("redux.sync.max.f32") == 0 and count("max.f32") == 10 and count("shfl.sync.bfly.b32") == 10, what
        loads = dict(none=(0, 0), int32=(1, 0), int64=(0, 1))[table]
        stores = dict(none=(1, 0), int32=(1, 0), int64=(0, 1))[table]
        assert (count("ld.global.b32"), count("ld.global.v2.b32")) == loads and (count("st.global.b32"), count("st.global.v2.b32")) == stores, what
        assert not [op for op in ops if op.startswith(("griddepcontrol", "mbarrier", "cp.async", "tcgen05"))], what
        statements = _statements(ptx)
        labels = {tokens[0][:-1]: position for position, tokens in enumerate(statements) if tokens[0].endswith(":")}
        branches = [(position, tokens[-1]) for position, tokens in enumerate(statements) if tokens[0].startswith("bra")]
        assert branches and all(labels[target] > position for position, target in branches), f"{what}: a backward branch (a loop)"
        sass = _sass(cubin)
        if sass is not None:
            assert not [ln for ln in sass if " STL" in ln or " LDL" in ln], f"{what}: the compiled kernel accesses local memory"
    if cc is not None and arch != "sm_100a":
        _, base, _ = _render(tmp_path, "sm_100a")

        def body(text):
            return text[text.index(".entry") :]

        assert all(body(kept[table][0]) == body(base[table][0]) for table in TABLES), f"from its entry on the {arch} rendering is not the sm_100a text"


# ---------------------------------------------------------------------------------------------------------------------------
# device: the rule
# ---------------------------------------------------------------------------------------------------------------------------
def _fresh_out(rows, table, dev, extra=0):
    """A caller-owned result vector filled with a sentinel no id and no column equals."""
    return torch.full((rows + extra,), -0x5A5A5A5A, dtype=RESULT[table], device=dev)


@pytest.mark.parametrize("kind", ("random", "ties", "special"))
@pytest.mark.parametrize("table", TABLES)
def test_matches_the_rule(table, kind):
    """Every byte of the result is the reference's at 1 .. 257 rows, for every table kind, on random rows, on rows that hold
    their maximum thousands of times and on rows with NaN, infinities and zeros of both signs -- with a caller-owned and with
    the entry-owned output."""
    dev = _device_or_skip()
    k = make_entry(dev)
    ids = make_table(table, dev, seed=7)
    x = make_rows(max(ROWS), dev, seed=40 + len(kind), kind=kind)
    for rows in ROWS:
        mine = _fresh_out(rows, table, dev)
        assert k(x[:rows], ids, mine) is mine
        assert_rule(mine, x[:rows], ids, f"{kind} rows {rows}, table {table}, caller-owned")
        assert_rule(k(x[:rows], ids), x[:rows], ids, f"{kind} rows {rows}, table {table}, entry-owned")


@pytest.mark.parametrize("table", TABLES)
def test_rule_rows(table):
    """The rows that state the rule one sentence at a time -- the maximum in the first and in the last column, ties inside a
    vector, across a thread's passes, across threads and across warps (the lower COLUMN wins also where it sits in the higher
    thread or warp), zeros of both signs, infinities, the most negative and the largest finite value, a subnormal, NaN beside
    values, rows of NaN and of -inf -- as ONE call and each row alone: the hand-written column, through the table."""
    dev = _device_or_skip()
    k = make_entry(dev)
    names, x, columns = rule_rows()
    ids = make_table(table, seed=11)
    want = columns.to(I32) if ids is None else ids[columns]
    x, ids = x.to(dev), None if ids is None else ids.to(dev)
    got = k(x, ids, _fresh_out(len(names), table, dev)).cpu()
    wrong = [f"{name}: got {int(a)}, want {int(b)}" for name, a, b in zip(names, got, want) if int(a) != int(b)]
    assert not wrong, wrong
    assert got.dtype == RESULT[table] and torch.equal(result_bytes(got), result_bytes(want))
    alone = torch.cat([k(x[r : r + 1], ids, _fresh_out(1, table, dev)) for r in range(len(names))]).cpu()
    assert torch.equal(alone, want), [name for name, a, b in zip(names, alone, want) if int(a) != int(b)]


def test_columns_are_those_of_torch_argmax_on_rows_without_a_nan():
    dev = _device_or_skip()
    k = make_entry(dev)
    for kind in ("random", "ties"):
        x = make_rows(64, dev, seed=17, kind=kind)
        assert not bool(torch.isnan(x.float()).any())
        got = k(x)
        assert got.dtype == I32 and torch.equal(got.to(I64), torch.argmax(x.float(), dim=1)), kind


def test_slab_views_bind_as_they_are():
    """``x`` as a view of a wider slab (a row stride of whole vectors that is not the width; a column offset of whole vectors),
    as every second row of a slab, and as one stored row expanded to several (a row stride of zero): the rule on each view's own
    rows, with nothing copied."""
    dev = _device_or_skip()
    k = make_entry(dev)
    ids = make_table("int64", dev, seed=5)
    slab = torch.zeros(6, WIDTH + 64, dtype=BF16, device=dev)
    slab[:, 16 : WIDTH + 16] = make_rows(6, dev, seed=77, kind="special")
    slab[:, :16], slab[:, WIDTH + 16 :] = 3.0e38, 3.0e38  # larger than every value of a view: a launch that read past a row would choose them
    views = {"a column window": slab[:, 16 : WIDTH + 16], "rows 1 .. 3": slab[1:4, 16 : WIDTH + 16], "every second row": slab[::2, 16 : WIDTH + 16]}
    views["one row expanded"] = slab[2:3, 16 : WIDTH + 16].expand(5, WIDTH)
    for name, view in views.items():
        assert not view.is_contiguous(), name
        assert_rule(k(view, ids, _fresh_out(view.shape[0], "int64", dev)), view, ids, name)


def test_output_ownership():
    """A caller-owned ``out`` is returned as given and its elements past the call's rows keep their bytes.  Without ``out`` the
    entry returns ITS vector of the row count and the result's dtype -- the same tensor on every call of that row count, one per
    dtype, the pair ``outputs(rows)`` names -- and such a vector passed back as ``out`` is written like any other."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(9, dev, seed=23)
    for table in TABLES:
        ids = make_table(table, dev, seed=2)
        mine = _fresh_out(4, table, dev, extra=3)
        assert k(x[:4], ids, mine) is mine
        assert_rule(mine[:4], x[:4], ids, f"caller-owned, table {table}")
        assert bool((mine[4:] == -0x5A5A5A5A).all()), "elements past the call's rows were written"
        owned = k(x[:4], ids)
        assert owned is k.outputs(4)[kcfg.RESULT_DTYPE[table]] and owned is k(x[:4], ids) and owned.shape == (4,)
        assert_rule(owned, x[:4], ids, f"entry-owned, table {table}")
    pair = k.outputs(4)
    assert set(pair) == {"int32", "int64"} and pair["int32"].dtype == I32 and pair["int64"].dtype == I64 and set(k.buffers) == {4}
    other = k(x[4:8], make_table("int64", dev, seed=2), pair["int64"])
    assert other is pair["int64"]
    assert_rule(other, x[4:8], make_table("int64", dev, seed=2), "the entry-owned vector passed back as out")
    assert k(x).shape == (9,) and set(k.buffers) == {4, 9}


def test_compile_builds_every_table_kind_and_no_later_call_compiles(monkeypatch):
    """A call on a fresh entry compiles ITS table kind and no other.  ``compile()`` -- which takes no operand -- builds the
    three kinds, after which every row count and every table kind launches with compilation FORBIDDEN (the row count, the
    addresses and the row stride are runtime arguments, never a key); a second ``compile()`` is a no-op."""
    dev = _device_or_skip()
    x = make_rows(64, dev, seed=31)
    first = make_entry(dev)
    assert_rule(first(x[:2], make_table("int32", dev)), x[:2], make_table("int32", dev), "first use")
    assert list(first.compiled) == [first.key("int32")] and len(first.kernel_symbols_for(2)) == 1
    k = make_entry(dev)
    k.compile()
    assert list(k.compiled) == [k.key(table) for table in TABLES] and not k.buffers
    symbols = k.kernel_symbols_for(1)
    assert len(set(symbols)) == 3 and all(symbol.startswith(f"{SYMBOL}w{WIDTH}_{table}") for symbol, table in zip(symbols, TABLES)), symbols
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("a call behind compile() compiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    for table in TABLES:
        ids = make_table(table, dev, seed=3)
        for rows in (1, 5, 64):
            assert_rule(k(x[:rows], ids), x[:rows], ids, f"rows {rows}, table {table}, behind one compile")
    k.compile()
    k.compile(5)
    assert len(k.compiled) == 3


def test_graph_capture_replays_bitwise_with_eager():
    """The three table kinds captured into one CUDA graph behind ``compile(rows)``: the replay is the eager result, and after the
    rows and a table change IN PLACE the replay is a fresh eager result on the new values (every tensor is a launch argument:
    the graph holds addresses, never values)."""
    dev = _device_or_skip()
    rows = 6
    k = make_entry(dev)
    k.compile(rows)
    x = make_rows(rows, dev, seed=51, kind="special")
    tables = {table: make_table(table, dev, seed=9) for table in TABLES}
    outs = {table: _fresh_out(rows, table, dev) for table in TABLES}
    eager = {table: k(x, tables[table], _fresh_out(rows, table, dev)).clone() for table in TABLES}
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for table in TABLES:
                k(x, tables[table], outs[table])
        for _ in range(3):
            for out in outs.values():
                out.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            for table in TABLES:
                assert torch.equal(outs[table], eager[table]), f"graph replay, table {table}"
        x.copy_(make_rows(rows, dev, seed=52, kind="ties"))
        tables["int64"].copy_(make_table("int64", dev, seed=10))
        graph.replay()
        torch.cuda.synchronize()
        for table in TABLES:
            assert_rule(outs[table], x, tables[table], f"the replay reads the operands' current values, table {table}")
    finally:
        graph.reset()


def test_warm_call_allocates_nothing_and_never_synchronizes():
    """Rules 1 and 3 on the warm path: with a caller-owned and with the entry-owned output, calls allocate no device memory and
    raise nothing under ``set_sync_debug_mode('error')`` (no device read reaches the host)."""
    dev = _device_or_skip()
    rows = 8
    k = make_entry(dev)
    x = make_rows(rows, dev, seed=41)
    ids = make_table("int64", dev, seed=4)
    mine = _fresh_out(rows, "int64", dev)
    k(x, ids, mine)
    k(x, ids)
    k(x)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            k(x, ids, mine)
            k(x, ids)
            k(x)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    assert_rule(mine, x, ids, "the warm calls' result")


# The allocation of an entry-owned pair is stream work of the stream that allocated it (``cudnn.frost.init_order``): the entry
# does not initialize the pair, but torch fills it under deterministic algorithms and a recycled block can carry a queued
# write.  The cells below warm an entry with a CALLER-OWNED ``out`` (it holds no pair yet), park a side stream behind a delay
# kernel, let the entry allocate the pair on it and launch from the original stream while that work is still queued
# (``init_order_schedules`` says what every such cell must do).
def _warm_entry_without_a_pair(dev, rows, seed, tables=("int64",)):
    """``(entry, x, ids, want)``: the entry with every table kind compiled and those of ``tables`` warmed by a real launch into
    a caller-owned ``out``; ``ids`` and ``want`` (the result, held to the rule) are those of the int64 table."""
    k = make_entry(dev)
    k.compile()  # every kind is built here, so that an allocating compile(rows) on a parked stream compiles nothing
    x = make_rows(rows, dev, seed=seed, kind="ties")
    ids = make_table("int64", dev, seed=seed)
    for table in tables:
        mine = k(x, ids if table == "int64" else None, _fresh_out(rows, table, dev))
        assert_rule(mine, x, ids if table == "int64" else None, f"rows {rows}: a caller-owned out, table {table}")
    assert not k.buffers, "the warm-up allocated the entry-owned pair"
    warm_fills(dev, I32, I64, torch.uint8)
    return k, x, ids, reference(x, ids).to(dev)


def _pool_of_a_pair(dev, rows):
    return pooled(dev, lambda: [torch.empty(rows, dtype=dtype, device=dev) for dtype in (I32, I64)])


@pytest.mark.usefixtures("settled_records")
@pytest.mark.parametrize("out_reaches_the_call", ("by default", "the vector passed back"))
@pytest.mark.parametrize("allocator", ("outputs", "compile"))
def test_warm_launch_waits_for_the_fill_the_allocating_stream_still_owes_the_entry_owned_pair(allocator, out_reaches_the_call):
    """The launch that writes the entry-owned vector -- by default, or as the vector ``outputs(rows)`` returned, passed back as
    ``out`` -- is ordered behind the fill torch adds to the pair's allocation under deterministic algorithms, although that fill
    is queued on another stream and the host does not wait; unordered, the fill lands on the kernel's result.  Both allocating
    routes: ``outputs(rows)`` and ``compile(rows)``."""
    dev = _device_or_skip()
    rows = 7
    k, x, ids, want = _warm_entry_without_a_pair(dev, rows, 131)

    def allocate():
        with deterministic_fill():
            k.outputs(rows) if allocator == "outputs" else k.compile(rows)

    pending = park(_pool_of_a_pair(dev, rows), allocate)
    assert set(k.buffers) == {rows}, "the pair was not allocated on the parked stream"
    vector = k.outputs(rows)["int64"]
    out = k(x, ids) if out_reaches_the_call == "by default" else k(x, ids, vector)
    assert not pending.query(), "the call did not overlap the pending work (or it waited on the host)"
    torch.cuda.synchronize()
    assert out is vector
    differing = int((out != want).sum())
    assert (
        differing == 0
    ), f"the entry-owned vector behind a pending fill, allocated by {allocator}, {out_reaches_the_call}: {differing} of {rows} results differ"


@pytest.mark.usefixtures("settled_records")
def test_warm_launch_waits_for_a_previous_owners_write_of_a_recycled_block():
    """A pair allocated on a parked stream into a block whose previous owner still has a fill queued there: the launch from
    the original stream is ordered behind that fill for whichever vector of the pair got the block."""
    dev = _device_or_skip()
    rows = 7
    k, x, ids, _ = _warm_entry_without_a_pair(dev, rows, 133, tables=("none", "int64"))
    recycled = []

    def allocate():
        recycled.append(previous_owner(lambda: torch.empty(rows * 8, dtype=torch.uint8, device=dev), 0xEE))
        k.outputs(rows)

    pending = park(_pool_of_a_pair(dev, rows), allocate)
    pair = k.outputs(rows)
    hit = [name for name, vector in pair.items() if vector.data_ptr() == recycled[0]]
    if not hit:
        pytest.skip("the allocator did not hand the freed block to the entry: the schedule shows nothing")
    table = dict(int32=None, int64=ids)[hit[0]]
    out = k(x, table)
    assert not pending.query(), "the call did not overlap the pending work (or it waited on the host)"
    torch.cuda.synchronize()
    assert out is pair[hit[0]]
    assert_rule(out, x, table, f"the {hit[0]} vector on a recycled block")


@pytest.mark.usefixtures("settled_records")
def test_warm_call_does_nothing_for_an_allocation_whose_stream_work_is_complete(monkeypatch):
    """A record is dropped by the first call that sees its event complete -- one with a caller-owned ``out`` too -- and then the
    warm path is what it was: calls allocate no device memory, raise nothing under ``set_sync_debug_mode('error')``, read or
    wait for no event and no stream, and do not enter the ordering step."""
    dev = _device_or_skip()
    rows, table = 8, "int64"
    k, x, ids, want = _warm_entry_without_a_pair(dev, rows, 135)

    def allocate():
        with deterministic_fill():
            k.outputs(rows)

    pending = park(_pool_of_a_pair(dev, rows), allocate)
    out = k(x, ids)
    assert not pending.query(), "the call did not overlap the pending work (or it waited on the host)"
    torch.cuda.synchronize()
    assert torch.equal(out, want), "the call that met the pending fill"
    mine = _fresh_out(rows, table, dev)
    k(x, ids, mine)  # a caller-owned out: the call writes no entry-owned byte and still drops what is complete
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
            k(x, ids)
            k(x, ids, mine)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    monkeypatch.undo()
    torch.cuda.synchronize()
    assert not touched, f"a warm call behind a complete allocation touched {sorted(set(touched))}"
    assert torch.cuda.memory_stats(dev)["allocation.all.allocated"] == allocated, "a warm call allocated device memory"
    assert torch.equal(out, want) and torch.equal(mine, want), "the warm calls' results"


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_kernel_is_looked_up():
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(2, dev, seed=5)
    ids = make_table("int64", dev)
    want = k(x, ids).clone()
    with pytest.raises(TypeError, match="x"):
        k(x.float(), ids)
    with pytest.raises(ValueError, match="one device"):
        k(x.cpu(), ids)
    with pytest.raises(ValueError, match="x"):
        k(x[:, : WIDTH // 2], ids)
    with pytest.raises(TypeError, match="ids"):
        k(x, ids.to(torch.int16))
    with pytest.raises(TypeError, match="out"):
        k(x, ids, torch.zeros(2, dtype=I32, device=dev))
    with pytest.raises(ValueError, match="out"):
        k(x, ids, ids[:2])
    torch.cuda.synchronize()
    assert torch.equal(k.outputs(2)["int64"], want), "a declined warm call launches nothing"


def test_entry_on_another_device_launches_there_and_keeps_the_callers_device_current():
    """The device is implicit state like the stream: an entry built for a device that is NOT the caller's current one compiles
    and launches on ITS device and leaves the caller's current device as it found it, on the first call and on a warm call.
    Needs two CUDA devices the step serves."""
    dev = _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    other = torch.device("cuda", (dev.index + 1) % torch.cuda.device_count())
    cc = torch.cuda.get_device_capability(other)
    if not entry.cc_supported(cc) or entry._dsl_error(cc) is not None:
        pytest.skip(f"the second CUDA device ({other}, compute capability {cc[0]}.{cc[1]}) is not one the step serves")
    x = make_rows(17, other, seed=77, kind="special")
    ids = make_table("int64", other, seed=1)
    k = make_entry(other)
    assert k.device == other
    for label, out in (("the first call", None), ("a warm call, entry-owned", None), ("a warm call, caller-owned", _fresh_out(17, "int64", other))):
        got = k(x, ids, out)
        assert torch.cuda.current_device() == dev.index, f"{label} left the entry's device current"
        assert got.device == other
        assert_rule(got, x, ids, label)
    assert torch.empty(1, device="cuda").device == dev, "the caller's next default-device allocation landed on another device"


@pytest.mark.parametrize("table", TABLES)
def test_launch_facts_equal_the_profilers_record_of_a_launch(table):
    """On the device: one launch of the entry is ONE kernel whose profiler record -- name, grid, block, shared memory -- equals
    the entry's launch facts; nothing is named before the first compile; the SAME compiled kernel serves another row count,
    whose launch record follows its own row count's grid; and the symbol carries the form."""
    dev = _device_or_skip()
    k = make_entry(dev)
    ids = make_table(table, dev)
    assert k.kernel_symbols_for(3) == (), "nothing is compiled yet"
    records = []
    for rows in (3, 17):
        x = make_rows(rows, dev, seed=3)
        out = _fresh_out(rows, table, dev)
        records.append(assert_facts_equal_the_launch(k, lambda: k(x, ids, out), rows, what=entry.NAME))
    assert records[0].name == records[1].name and records[0].name.startswith(f"{SYMBOL}w{WIDTH}_{table}"), records
    assert records[0].grid != records[1].grid and records[0].block == records[1].block == (THREADS, 1, 1), records
    assert k.form_for(3) == asdict(k.cfg) and len(k.compiled) == 1
