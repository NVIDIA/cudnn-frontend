# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compiled-plan cache's rules, without a GPU or the DSL: identity is the
whole manifest, an entry is only ever reused under its own embedded key,
anything doubtful is a miss, and the in-process memo in front of the files
hands one object per kernel per process."""

import json
import pathlib

import pytest

import cudnn
from cudnn.frost import compiled_cache as cc

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def _fresh_memo():
    """Every test starts from an empty in-process memo: the fakes below compile under short
    literal keys, and an object memoised by one test must not be served to the next."""
    cc.clear_memo()
    yield
    cc.clear_memo()


def test_manifest_names_every_dependency_and_is_deterministic():
    m = cc.environment_manifest()
    for field in (
        "schema",
        "cudnn_frontend",
        "cudnn_source",
        "cutlass_dsl",
        "tvm_ffi",
        "cuda_driver",
        "device_name",
        "compute_capability",
        "sm_count",
        "l2_bytes",
    ):
        assert field in m and isinstance(m[field], str) and m[field], field
    assert m["schema"] == cc._SCHEMA and m["cudnn_frontend"] == cudnn.__version__
    assert cc.environment_manifest() == m  # same process, same answer


def test_entry_location_is_the_content_hash_of_manifest_and_key(tmp_path):
    m = {"schema": cc._SCHEMA, "cudnn_frontend": "1.30.0", "cutlass_dsl": "4.7.1"}
    a = cc._entry_dir(tmp_path, m, "k1")
    b = cc._entry_dir(tmp_path, m, "k2")
    c = cc._entry_dir(tmp_path, dict(m, cutlass_dsl="4.8.0"), "k1")
    assert a.parent == b.parent and a != b  # same environment, different kernels
    assert c.parent != a.parent  # a dependency moved: a different directory outright
    assert a.parts[-3] == cc._SCHEMA


def test_cache_dir_override_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "from_env"))
    assert cc.get_cache_dir() == tmp_path / "from_env"
    cc.set_cache_dir(tmp_path / "explicit")
    try:
        assert cc.get_cache_dir() == tmp_path / "explicit"
    finally:
        cc.set_cache_dir(None)
    assert cc.get_cache_dir() == tmp_path / "from_env"


def test_disabled_or_uncacheable_compiles_as_before(monkeypatch):
    import cutlass.cute as cute

    calls = []
    monkeypatch.setattr(cute, "compile", lambda fn, *a, **k: calls.append((fn, a, k)) or "compiled")
    cc.reset_stats()
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    assert cc.compile_cached(lambda: None, 1, cache_key="k", options="--enable-tvm-ffi") == "compiled"
    monkeypatch.delenv(cc._ENV_DISABLE)
    assert cc.compile_cached(lambda: None, 1, cache_key=None, options="--enable-tvm-ffi") == "compiled"  # no key: not cacheable
    assert cc.compile_cached(lambda: None, 1, cache_key="k", options="") == "compiled"  # no tvm-ffi: not exportable
    assert len(calls) == 3 and cc.stats()["bypassed"] == 3


def test_an_environment_that_cannot_identify_itself_persists_nothing(monkeypatch, tmp_path):
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", lambda fn, *a, **k: "compiled")
    monkeypatch.setattr(cc, "environment_manifest", lambda device=None: {"schema": cc._SCHEMA, "cudnn_frontend": "1.30.0", "cuda_driver": "unknown"})
    cc.set_cache_dir(tmp_path)
    try:
        cc.reset_stats()
        assert cc.compile_cached(lambda x: x, 1, cache_key="k", options="--enable-tvm-ffi") == "compiled"
    finally:
        cc.set_cache_dir(None)
    assert cc.stats()["bypassed"] == 1 and not list(tmp_path.rglob("*"))


def test_a_foreign_or_corrupt_entry_is_a_miss(tmp_path):
    cc.reset_stats()
    entry = tmp_path / "e"
    entry.mkdir()
    (entry / cc._OBJECT).write_bytes(b"not an object")
    (entry / cc._ENTRY).write_text(json.dumps({"schema": cc._SCHEMA, "key": "other", "symbol": "frost_gemm"}))
    assert cc._try_load(entry, "mine", "frost_gemm") is None  # embedded key differs
    (entry / cc._ENTRY).write_text("{not json")
    assert cc._try_load(entry, "mine", "frost_gemm") is None  # unreadable record
    (entry / cc._ENTRY).unlink()
    assert cc._try_load(entry, "mine", "frost_gemm") is None  # no record at all
    assert cc.stats()["invalid"] == 1  # only the key mismatch is worth a warning; the others are plain misses


def test_the_record_carries_the_in_process_calling_convention():
    """A kernel's runtime signature is its Python signature MINUS constexpr and
    env-stream parameters; the record must say what the DSL's wrapper said."""
    import inspect
    from collections import namedtuple

    Spec = namedtuple("Spec", "arg_names arg_defaults kwonly_names kwonly_defaults")

    class ExecutionArgs:  # what jit_executor exposes
        def get_kwargs_wrapper_spec(self, exclude=()):
            names = [n for n in ("q", "flag", "count", "tail") if n not in exclude]
            defaults = {"count": 0, "tail": None}
            return Spec(names, tuple(defaults[n] for n in names if n in defaults), [n for n in ("stream",) if n not in exclude], {})

    def in_process_wrapper(q, count=0, tail=None, *, stream):  # 'flag' was a Constexpr: gone at run time
        return None

    class Compiled:
        execution_args = ExecutionArgs()
        _kwargs_wrapper = staticmethod(in_process_wrapper)

    spec = cc._wrapper_spec_of(Compiled(), (1, True, 0, None), {})
    assert spec == {"arg_names": ["q", "count", "tail"], "arg_defaults": [0, None], "kwonly_names": ["stream"], "kwonly_defaults": {}}
    calls = []
    rebuilt = cc._rebuild_wrapper(lambda *a: calls.append(a), spec)
    rebuilt("Q", stream="S")
    rebuilt("Q", 7, tail="T", stream="S")
    assert calls == [("Q", 0, None, "S"), ("Q", 7, "T", "S")]  # defaults filled, kwargs placed, positional-only underneath
    assert list(inspect.signature(rebuilt).parameters) == ["q", "count", "tail", "stream"]

    # not reproducible exactly -> no record
    class NoWrapper:
        execution_args = ExecutionArgs()

    assert cc._wrapper_spec_of(NoWrapper(), (), {}) is None
    from dataclasses import dataclass

    @dataclass
    class Op:
        k: int = 1

    assert cc._wrapper_spec_of(Compiled(), (Op(),), {}) is None  # a dataclass argument goes through a hook the record cannot describe
    assert cc._rebuild_wrapper(lambda *a: None, None) is None  # an entry from before the record carried a spec is a miss


def test_template_key_joins_the_digest_with_plain_arguments_only():
    g = {"FROST_SOURCE_DIGEST": "abc123"}
    key = cc.template_key(g, {"b": 2, "qh": 8, "lse_stride": (1, 2, 3), "flag": True, "opt": None}, "compile")
    assert key.startswith("abc123|compile|") and "('b', 2)" in key and "('lse_stride', (1, 2, 3))" in key
    assert cc.template_key(g, {"qh": 8, "b": 2}, "compile") == cc.template_key(g, {"b": 2, "qh": 8}, "compile")  # order-free
    assert cc.template_key(g, {"b": 2}, "compile") != cc.template_key(g, {"b": 2}, "other")  # the function is named
    assert cc.template_key({}, {"b": 2}) is None  # no digest: the loader did not produce this module
    assert cc.template_key(g, {"b": 2, "device": object()}) is None  # a repr that need not survive a process

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class P:
        d: int = 128
        thd: bool = False

    assert "P(d=128, thd=False)" in cc.template_key(g, {"p": P()}, "compile")  # a params record of plain fields is plain
    assert cc.template_key(g, {"p": P(d=object())}, "compile") is None  # unless one of its fields is not
    import cutlass

    assert "cutlass" in cc.template_key(g, {"io_dtype": cutlass.BFloat16}, "compile")  # a dtype CLASS names itself


def test_the_loader_digests_the_file_and_the_params(tmp_path):
    from dataclasses import dataclass

    from cudnn.frost import template_loader

    @dataclass(frozen=True)
    class Params:
        causal: bool = False

    src = tmp_path / "tiny_template.py"
    src.write_text("CFG = FROST_TEMPLATE_PARAMS\n")
    a = template_loader.load_template(str(src), Params(False), tag="tiny")
    b = template_loader.load_template(str(src), Params(True), tag="tiny")
    assert a.FROST_SOURCE_DIGEST and len(a.FROST_SOURCE_DIGEST) == 16
    assert a.FROST_SOURCE_DIGEST != b.FROST_SOURCE_DIGEST  # the params are part of it
    assert template_loader.load_template(str(src), Params(False), tag="tiny") is a  # the module cache still keys on (path, params)
    src.write_text("CFG = FROST_TEMPLATE_PARAMS  # edited\n")
    c = template_loader.load_template(str(src), Params(None), tag="tiny")  # a new params value forces a re-read
    assert c.FROST_SOURCE_DIGEST not in (a.FROST_SOURCE_DIGEST, b.FROST_SOURCE_DIGEST)


def test_the_manifest_carries_the_source_tree_not_a_commit(tmp_path):
    """An edit anywhere in the package -- committed or not -- lands in another
    environment directory; a wheel hashes the same every process."""
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    (tmp_path / "sub" / "__pycache__").mkdir()
    (tmp_path / "sub" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"ignored")
    before = cc.source_tree_digest(tmp_path)
    assert before == cc.source_tree_digest(tmp_path) and len(before) == 16
    (tmp_path / "sub" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"still ignored")
    assert cc.source_tree_digest(tmp_path) == before
    (tmp_path / "sub" / "b.py").write_text("y = 3\n")  # a shared helper edited, no version bump
    assert cc.source_tree_digest(tmp_path) != before
    import cudnn

    assert cc.environment_manifest()["cudnn_source"] == cc.source_tree_digest(pathlib.Path(cudnn.__file__).resolve().parent)


def test_prune_retires_dead_environments_oldest_first_and_keeps_the_current_one(tmp_path, monkeypatch):
    """Every edited checkout (and every CI commit) mints an environment that is
    never hit again; a persistent home would grow by hundreds of MB per commit.
    Whole environment directories go, dead schemas first, then oldest first,
    until the root fits; the current environment is never a candidate."""
    import os
    import time

    def env(schema, name, size, age_s, manifest=None):
        d = tmp_path / schema / (cc._digest(name) if len(name) != 24 else name)
        d.mkdir(parents=True)
        if manifest is None:
            manifest = json.dumps({"schema": schema, "cudnn_frontend": "1.30.0", "cutlass_dsl": "4.7.1"})
        if manifest:
            (d / cc._MANIFEST).write_text(manifest)
        e = d / "entry_x"
        e.mkdir()
        (e / cc._OBJECT).write_bytes(b"x" * size)
        (e / cc._ENTRY).write_text("{}")
        for f in d.rglob("*"):
            os.utime(f, (time.time() - age_s, time.time() - age_s))
        return d

    old = env(cc._SCHEMA, "old", 1000, 3000)
    mid = env(cc._SCHEMA, "mid", 1000, 2000)
    cur = env(cc._SCHEMA, "cur", 1000, 4000)  # the oldest by mtime, but this process's own
    dead = env("v0", "ancient", 100, 10)  # a dead schema goes first whatever its age
    # not ours, whatever the size or age: a caller's artifacts under a shared root, a directory without
    # our manifest, a schema-like directory with a foreign name -- never deleted, never counted
    foreign = tmp_path / "flashinfer" / "existing_artifact"
    foreign.mkdir(parents=True)
    (foreign / "caller_owned.bin").write_bytes(b"z" * 100_000)
    no_manifest = env(cc._SCHEMA, "half_written", 100_000, 9000, manifest="")
    odd_name = env(cc._SCHEMA, "x" * 24, 100_000, 9000)  # 24 chars but not a hex digest
    other_tool = env(cc._SCHEMA, "b" * 24, 100_000, 9000, manifest='{"producer": "another-tool"}')  # right shape, not our manifest
    empty_manifest = env(cc._SCHEMA, "c" * 24, 100_000, 9000, manifest="{}")
    wrong_schema = env(cc._SCHEMA, "d" * 24, 100_000, 9000, manifest=json.dumps({"schema": "v9", "cudnn_frontend": "1.30.0"}))
    corrupt = env(cc._SCHEMA, "e" * 24, 100_000, 9000, manifest="{not json")
    empty_version = env(cc._SCHEMA, "f" * 24, 100_000, 9000, manifest=json.dumps({"schema": cc._SCHEMA, "cudnn_frontend": ""}))
    cc.reset_stats()
    assert cc.prune(tmp_path, limit=0) == 0  # 0 = never prune
    assert cc.prune(tmp_path, limit=2500, keep=cur) == 2
    assert not dead.exists() and not old.exists() and mid.exists() and cur.exists()
    assert (foreign / "caller_owned.bin").exists() and no_manifest.exists() and odd_name.exists()
    assert other_tool.exists() and empty_manifest.exists() and wrong_schema.exists() and corrupt.exists() and empty_version.exists()
    assert cc.stats()["pruned"] == 2
    assert cc.prune(tmp_path, limit=2500, keep=cur) == 0  # under the cap: nothing to do
    # a symlink planted in the root is neither followed nor a deletion target
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    (outside / "victim").write_bytes(b"y" * 10_000)
    (tmp_path / cc._SCHEMA / "link").symlink_to(outside, target_is_directory=True)
    assert cc.prune(tmp_path, limit=1, keep=cur) == 1 and not mid.exists() and cur.exists()
    assert (outside / "victim").exists() and (tmp_path / cc._SCHEMA / "link").is_symlink()
    # a symlinked FILE inside an environment counts nothing: the target's size must not push a live environment out
    before = cc._dir_bytes_and_mtime(cur)[0]
    (cur / "entry_x" / "planted").symlink_to(outside / "victim")
    assert cc._dir_bytes_and_mtime(cur)[0] == before
    monkeypatch.setenv(cc._ENV_MAX_BYTES, "0")
    assert cc.max_bytes() == 0
    monkeypatch.setenv(cc._ENV_MAX_BYTES, "not-a-number")
    assert cc.max_bytes() == cc._DEFAULT_MAX_BYTES


# ---------------------------------------------------------------------------
# The in-process memo
# ---------------------------------------------------------------------------


def _fake_compile(monkeypatch, calls):
    """``cute.compile`` replaced by a counter that mints a fresh object per call."""
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", lambda fn, *a, **k: calls.append((fn, a, k)) or object())


def test_the_memo_hands_out_the_same_object_without_a_second_compile(monkeypatch):
    """A second compile of the same (cache_key, symbol, options) in one process returns the
    very object the first produced -- no trace, no compile -- with the on-disk cache OFF."""
    calls = []
    _fake_compile(monkeypatch, calls)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    cc.reset_stats()
    first = cc.compile_cached(lambda: None, 1, cache_key="k", options="--enable-tvm-ffi")
    second = cc.compile_cached(lambda: None, 2, cache_key="k", options="--enable-tvm-ffi")
    assert second is first and len(calls) == 1
    assert cc.stats()["memo_hits"] == 1 and cc.stats()["bypassed"] == 1
    assert cc.memo_enabled()


def test_the_memo_sits_in_front_of_an_enabled_on_disk_cache(monkeypatch, tmp_path):
    """With the files ON, the second call never reaches them: one miss, one load attempt."""
    calls, loads = [], []
    _fake_compile(monkeypatch, calls)  # the fake object has no export_to_c: a miss stores nothing and returns it
    monkeypatch.setattr(cc, "environment_manifest", lambda device=None: {"schema": cc._SCHEMA, "cudnn_frontend": "1.30.0", "cutlass_dsl": "4.8.0"})
    monkeypatch.setattr(cc, "_try_load", lambda entry, key, symbol: loads.append(entry) or None)
    monkeypatch.delenv(cc._ENV_DISABLE, raising=False)
    cc.set_cache_dir(tmp_path)
    try:
        cc.reset_stats()
        first = cc.compile_cached(lambda: None, 1, cache_key="k", options="--enable-tvm-ffi")
        second = cc.compile_cached(lambda: None, 1, cache_key="k", options="--enable-tvm-ffi")
    finally:
        cc.set_cache_dir(None)
    assert second is first and len(calls) == 1 and len(loads) == 1
    assert cc.stats()["misses"] == 1 and cc.stats()["hits"] == 0 and cc.stats()["memo_hits"] == 1


def test_the_memo_keys_on_cache_key_symbol_and_options(monkeypatch):
    calls = []
    _fake_compile(monkeypatch, calls)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    cc.reset_stats()
    f = lambda: None  # noqa: E731
    cc.compile_cached(f, cache_key="k", options="--enable-tvm-ffi")
    cc.compile_cached(f, cache_key="k2", options="--enable-tvm-ffi")  # another kernel
    cc.compile_cached(f, cache_key="k", symbol="other", options="--enable-tvm-ffi")  # another exported symbol
    cc.compile_cached(f, cache_key="k", options="--enable-tvm-ffi --opt-level 1")  # other compile options
    cc.compile_cached(f, cache_key="k", options="")  # the options are part of the key even without tvm-ffi
    assert len(calls) == 5 and cc.stats()["memo_hits"] == 0 and len(cc._MEMO) == 5
    cc.compile_cached(f, cache_key="k", options="")  # memoised too: the memo needs no exportable object
    assert len(calls) == 5 and cc.stats()["memo_hits"] == 1


def test_the_memo_keys_on_the_device_a_build_targets(monkeypatch):
    """A compiled object bakes device-derived constants (SM count, L2, SMEM budget): a build
    scoped to another device is another kernel, exactly as the manifest names the device."""
    from cudnn.frost import device as frost_device

    calls = []
    _fake_compile(monkeypatch, calls)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    with frost_device.build_device(0):
        a = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    with frost_device.build_device(1):
        b = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    assert b is not a and len(calls) == 2
    with frost_device.build_device(0):
        assert cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi") is a
    assert len(calls) == 2 and {k[0] for k in cc._MEMO} == {0, 1}


def test_a_kernel_without_a_key_is_never_memoised(monkeypatch):
    calls = []
    _fake_compile(monkeypatch, calls)
    cc.reset_stats()
    a = cc.compile_cached(lambda: None, cache_key=None, options="--enable-tvm-ffi")
    b = cc.compile_cached(lambda: None, cache_key=None, options="--enable-tvm-ffi")
    assert b is not a and len(calls) == 2 and not cc._MEMO
    assert cc.stats()["bypassed"] == 2 and cc.stats()["memo_hits"] == 0


def test_the_memo_switch_turns_it_off(monkeypatch):
    calls = []
    _fake_compile(monkeypatch, calls)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    monkeypatch.setenv(cc._ENV_MEMO, "0")
    assert not cc.memo_enabled()
    cc.reset_stats()
    a = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    b = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    assert b is not a and len(calls) == 2 and not cc._MEMO and cc.stats()["memo_hits"] == 0
    for value in ("false", "no", " OFF "):
        monkeypatch.setenv(cc._ENV_MEMO, value)
        assert not cc.memo_enabled(), value
    for value in ("1", "true", "yes", "on", ""):
        monkeypatch.setenv(cc._ENV_MEMO, value)
        assert cc.memo_enabled(), value
    monkeypatch.delenv(cc._ENV_MEMO)
    assert cc.memo_enabled()  # the default is on
    c = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    assert c is cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi") and len(calls) == 3


def test_a_failed_compile_is_not_memoised(monkeypatch):
    import cutlass.cute as cute

    attempts = []

    def flaky(fn, *a, **k):
        attempts.append(fn)
        if len(attempts) == 1:
            raise RuntimeError("ptxas died")
        return object()

    monkeypatch.setattr(cute, "compile", flaky)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    with pytest.raises(RuntimeError, match="ptxas died"):
        cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    assert not cc._MEMO
    obj = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    assert len(attempts) == 2 and cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi") is obj and len(attempts) == 2


def test_clear_memo_forgets_without_invalidating_what_callers_hold(monkeypatch):
    calls = []
    _fake_compile(monkeypatch, calls)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    a = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    cc.clear_memo()
    assert not cc._MEMO
    b = cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi")
    assert b is not a and len(calls) == 2  # a is still the object its caller holds; b is the memo's from now on
    assert cc.compile_cached(lambda: None, cache_key="k", options="--enable-tvm-ffi") is b


def test_the_memo_is_safe_under_concurrent_compiles(monkeypatch):
    """Threads racing on an empty slot may each compile once, but every one of them leaves
    with the one object the memo kept, and the memo holds exactly one object per key."""
    import threading
    import time

    import cutlass.cute as cute

    compiles = []

    def slow_compile(fn, *a, **k):
        time.sleep(0.01)  # long enough for the racers to overlap
        compiles.append(fn)
        return object()

    monkeypatch.setattr(cute, "compile", slow_compile)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    keys = ["ka", "kb", "kc"]
    results = {key: [] for key in keys}
    errors = []
    n_threads = 12
    barrier = threading.Barrier(n_threads)

    def worker(key):
        try:
            barrier.wait(timeout=30)
            for _ in range(5):
                results[key].append(cc.compile_cached(lambda: None, cache_key=key, options="--enable-tvm-ffi"))
        except Exception as exc:  # noqa: BLE001 -- reported below, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(keys[i % len(keys)],)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors and not any(t.is_alive() for t in threads)
    for key in keys:
        assert len(results[key]) == 5 * (n_threads // len(keys)) and len({id(obj) for obj in results[key]}) == 1, key
    assert len(cc._MEMO) == len(keys) and len(compiles) >= len(keys)
    assert cc.stats()["memo_hits"] >= 5 * n_threads - len(compiles)
