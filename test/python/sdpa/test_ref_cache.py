# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa.ref_cache``: off without the variable, a miss computes and stores, a hit is bitwise the stored value without a compute, VERIFY
recomputes and fails on a drift, and the key covers what the reference depends on."""

import pytest
import torch

from sdpa import ref_cache

pytestmark = pytest.mark.L0


def _oracle(calls, value=1.0):
    def compute():
        calls.append(1)
        return dict(o=torch.full((4, 8), value, device="cuda"), stats=torch.arange(4, dtype=torch.float32, device="cuda"), amax=value)

    return compute


def test_off_without_the_variable_computes_every_time(monkeypatch):
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE", raising=False)
    calls = []
    for _ in range(2):
        ref_cache.cached_reference("unit", dict(seed=0), _oracle(calls))
    assert len(calls) == 2 and ref_cache.cache_dir() is None


def test_a_miss_stores_and_a_hit_returns_the_stored_value_without_a_compute(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    calls = []
    first = ref_cache.cached_reference("unit", dict(seed=1, s=512), _oracle(calls))
    assert len(calls) == 1 and len(list(tmp_path.glob("unit-*.pt"))) == 1
    second = ref_cache.cached_reference("unit", dict(seed=1, s=512), _oracle(calls, value=7.0))  # a hit never runs compute
    assert len(calls) == 1
    assert second["o"].device.type == "cuda" and torch.equal(second["o"], first["o"]) and torch.equal(second["stats"], first["stats"])
    assert second["amax"] == first["amax"] == 1.0


def test_the_key_covers_the_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    calls = []
    ref_cache.cached_reference("unit", dict(seed=1, s=512), _oracle(calls))
    ref_cache.cached_reference("unit", dict(seed=2, s=512), _oracle(calls))  # another seed: another entry
    ref_cache.cached_reference("unit", dict(seed=1, s=1024), _oracle(calls))  # another shape: another entry
    assert len(calls) == 3 and len(list(tmp_path.glob("unit-*.pt"))) == 3


def test_verify_mode_recomputes_and_fails_on_a_drift(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", "1")
    calls = []
    ref_cache.cached_reference("unit", dict(seed=3), _oracle(calls))
    before = ref_cache.stats()["verified"]
    ref_cache.cached_reference("unit", dict(seed=3), _oracle(calls))  # a hit: recomputed and compared bitwise
    assert len(calls) == 2 and ref_cache.stats()["verified"] == before + 1
    with pytest.raises(AssertionError, match="differs from a fresh compute"):
        ref_cache.cached_reference("unit", dict(seed=3), _oracle(calls, value=1.0 + 2**-20))


def test_a_torn_file_is_recomputed_and_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    calls = []
    ref_cache.cached_reference("unit", dict(seed=4), _oracle(calls))
    (path,) = tmp_path.glob("unit-*.pt")
    path.write_bytes(b"not a torch file")
    value = ref_cache.cached_reference("unit", dict(seed=4), _oracle(calls))
    assert len(calls) == 2 and value["amax"] == 1.0 and path.stat().st_size > 16
