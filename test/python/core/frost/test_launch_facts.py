# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared pieces of the entries' launch facts (``cudnn.frost.launch_facts``), host-only: the launch shape is a plain tuple, a form is
plain data built from a config's fields (never its derived properties), the symbols are read from compiled objects and never guessed,
and the module needs neither torch nor the CuTe DSL.  Also the device comparison every entry suite shares
(``launch_records.assert_facts_equal_the_launch``), on stubbed profiler records: it fails on each field it compares and skips, naming
the reason, where the profiler gives nothing to compare.  What each entry states with these pieces is pinned in the entry's own suite."""

import json
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass

import pytest

import launch_records
from cudnn.frost import launch_facts as lf

pytestmark = pytest.mark.L0


@dataclass(frozen=True)
class _Inner:
    HEADS: int = 3
    WIDTH: int = 128


@dataclass(frozen=True)
class _Cfg:
    THREADS: int = 256
    PATH: str = "bulk"
    RELEASE: bool = False
    NORM: object = None
    SPLITS: tuple = (1, 2, 4)
    GEOM: _Inner = _Inner()

    @property
    def DERIVED(self) -> int:
        return self.THREADS // 32


class _Compiled:
    """A stand-in for a compiled DSL host function: ``kernel_info`` keyed by the symbols of the kernels it launches, in launch order."""

    def __init__(self, *symbols):
        self.kernel_info = OrderedDict((symbol, {}) for symbol in symbols)


def test_a_launch_shape_is_a_plain_tuple_of_plain_values():
    shape = lf.LaunchShape((96, 1, 1), (256, 1, 1), 42112)
    assert shape == ((96, 1, 1), (256, 1, 1), 42112) and (shape.grid, shape.block, shape.shared_memory) == ((96, 1, 1), (256, 1, 1), 42112)
    assert json.loads(json.dumps(shape)) == [[96, 1, 1], [256, 1, 1], 42112]
    grid, block, shared_memory = shape
    assert (grid, block, shared_memory) == shape


def test_a_form_is_the_configs_fields_as_plain_data_with_the_entry_facts_appended():
    form = lf.plain_form(_Cfg(RELEASE=True), ROWS=4, EPS=1e-6)
    assert form == dict(THREADS=256, PATH="bulk", RELEASE=True, NORM=None, SPLITS=(1, 2, 4), GEOM=dict(HEADS=3, WIDTH=128), ROWS=4, EPS=1e-6)
    assert list(form) == ["THREADS", "PATH", "RELEASE", "NORM", "SPLITS", "GEOM", "ROWS", "EPS"], "the config's field order, then the entry facts"
    assert "DERIVED" not in form, "a derived property is not a field"
    assert type(form) is dict and type(form["GEOM"]) is dict
    assert json.loads(json.dumps(form))["GEOM"] == {"HEADS": 3, "WIDTH": 128}
    again = lf.plain_form(_Cfg(RELEASE=True), ROWS=4, EPS=1e-6)
    assert again == form and again is not form and again["GEOM"] is not form["GEOM"], "a fresh dict per call: a caller's edit reaches nothing"


def test_a_form_refuses_what_is_not_a_config_or_not_plain_data():
    with pytest.raises(TypeError, match="config dataclass instance, got dict"):
        lf.plain_form(dict(THREADS=256))
    with pytest.raises(TypeError, match="config dataclass instance, got type"):
        lf.plain_form(_Cfg)
    with pytest.raises(TypeError, match=r"_Cfg\.NORM is a set"):
        lf.plain_form(_Cfg(NORM={1, 2}))
    with pytest.raises(TypeError, match=r"_Cfg\.SPLITS\[1\] is a bytes"):
        lf.plain_form(_Cfg(SPLITS=(1, b"x")))
    with pytest.raises(TypeError, match=r"ROWS\[0\] is a object"):
        lf.plain_form(_Cfg(), ROWS=[object()])
    with pytest.raises(ValueError, match="entry fact THREADS would replace the config field"):
        lf.plain_form(_Cfg(), THREADS=128)


def test_kernel_symbols_are_read_from_the_compiled_objects_in_order_without_repeats():
    assert lf.kernel_symbols((), "an entry") == ()
    assert lf.kernel_symbols([_Compiled("k_a")], "an entry") == ("k_a",)
    assert lf.kernel_symbols(iter([_Compiled("k_b"), _Compiled("k_a"), _Compiled("k_b")]), "an entry") == ("k_b", "k_a")
    assert lf.kernel_symbols([_Compiled("k_main", "k_aux")], "an entry") == ("k_main", "k_aux"), "a function's kernels in launch order"


@pytest.mark.parametrize("compiled", [object(), type("NoMapping", (), {"kernel_info": ["k_a"]})(), type("IntKeys", (), {"kernel_info": {1: {}}})()])
def test_kernel_symbols_are_never_guessed(compiled):
    """A compiled object that does not expose its kernels is a typed decline naming the entry and the installed DSL, not an empty answer."""
    with pytest.raises(NotImplementedError, match="an entry: the installed CuTe DSL .* does not expose the kernel symbols of a compiled function"):
        lf.kernel_symbols([_Compiled("k_a"), compiled], "an entry")


def test_the_module_imports_without_a_framework_or_the_dsl():
    """A fresh interpreter: the launch-facts module adds neither torch nor the CuTe DSL (a delta against an empty interpreter, because
    a distribution's ``.pth`` file can import its package before the first line of any program)."""
    probe = "import json, sys; print('@@' + json.dumps(sorted(sys.modules)))"

    def modules(code):
        run = subprocess.run([sys.executable, "-c", code + "\n" + probe], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-400:]
        return set(json.loads(next(ln for ln in run.stdout.splitlines() if ln.startswith("@@"))[2:]))

    added = modules("import cudnn.frost.launch_facts") - modules("")
    assert "cudnn.frost.launch_facts" in added
    heavy = sorted(m for m in added if m.split(".")[0] in ("torch", "cutlass", "nvidia_cutlass_dsl", "jax"))
    assert not heavy, f"cudnn.frost.launch_facts imported {heavy}"


class _Entry:
    """The two facts the device comparison reads, as an entry states them."""

    def __init__(self, symbols, shape):
        self.symbols, self.shape = symbols, shape

    def kernel_symbols_for(self, rows):
        return self.symbols

    def launch_shape_for(self, rows):
        return self.shape


def test_the_device_comparison_fails_on_each_field_and_skips_where_nothing_is_recorded(monkeypatch):
    """``assert_facts_equal_the_launch`` on stubbed profiler records: equal facts pass and return the record; a wrong symbol, grid,
    block or shared memory and a call that launched two kernels each FAIL with the sentence naming both sides; no kernel record (a
    node without CUPTI) and a record without launch arguments are SKIPS that say so -- a coverage gap, never a pass."""
    record = launch_records.KernelRecord("k_a", (96, 1, 1), (256, 1, 1), 42112)
    shape = lf.LaunchShape((96, 1, 1), (256, 1, 1), 42112)
    recorded = [record]
    monkeypatch.setattr(launch_records, "kernel_records", lambda launch: recorded)
    check = launch_records.assert_facts_equal_the_launch
    assert check(_Entry(("k_a",), shape), None, 1) == record
    for symbols in ((), ("k_b",), ("k_a", "k_b")):
        with pytest.raises(AssertionError, match="the entry names .* the profiler recorded 'k_a'"):
            check(_Entry(symbols, shape), None, 1)
    for wrong in (shape._replace(grid=(48, 2, 1)), shape._replace(block=(128, 1, 1)), shape._replace(shared_memory=42128)):
        with pytest.raises(AssertionError, match="the entry states .* the profiler recorded"):
            check(_Entry(("k_a",), wrong), None, 1)
    recorded = [record, record._replace(name="k_b")]
    with pytest.raises(AssertionError, match="one call launched 2 kernels"):
        check(_Entry(("k_a",), shape), None, 1)
    recorded = []
    with pytest.raises(pytest.skip.Exception, match="recorded no kernel launch"):
        check(_Entry(("k_a",), shape), None, 1)
    recorded = [launch_records.KernelRecord("k_a", None, None, None)]
    with pytest.raises(pytest.skip.Exception, match="carries no grid / block / shared memory here .* the symbol matched"):
        check(_Entry(("k_a",), shape), None, 1)
    with pytest.raises(AssertionError, match="the entry names"):
        check(_Entry(("k_b",), shape), None, 1)
