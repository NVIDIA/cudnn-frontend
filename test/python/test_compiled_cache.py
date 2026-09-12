# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compiled-plan cache's rules, without a GPU or the DSL: identity is the
whole manifest, an entry is only ever reused under its own embedded key, and
anything doubtful is a miss."""

import json

import pytest

import cudnn
from cudnn.frost import compiled_cache as cc

pytestmark = pytest.mark.L0


def test_manifest_names_every_dependency_and_is_deterministic():
    m = cc.environment_manifest()
    for field in ("schema", "cudnn_frontend", "cutlass_dsl", "tvm_ffi", "cuda_driver", "device_name", "compute_capability", "sm_count", "l2_bytes"):
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
    assert cc._try_load(entry, "mine", "frost_gemm", lambda x: x) is None  # embedded key differs
    (entry / cc._ENTRY).write_text("{not json")
    assert cc._try_load(entry, "mine", "frost_gemm", lambda x: x) is None  # unreadable record
    (entry / cc._ENTRY).unlink()
    assert cc._try_load(entry, "mine", "frost_gemm", lambda x: x) is None  # no record at all
    assert cc.stats()["invalid"] == 1  # only the key mismatch is worth a warning; the others are plain misses


def test_signature_of_rejects_varargs():
    def ok(a, b, *, stream=None):
        return None

    def bad(*args, **kwargs):
        return None

    assert [p for p in cc._signature_of(ok).parameters] == ["a", "b", "stream"]
    assert cc._signature_of(bad) is None
