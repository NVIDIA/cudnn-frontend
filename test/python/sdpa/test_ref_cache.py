# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa.ref_cache``: off without the variable, a miss computes and stores, a hit is bitwise the stored value without a compute, VERIFY
recomputes and fails on a drift (``=0`` is off), the key covers what the reference depends on, and an entry is read with the weights-only
unpickler: a pickle payload or a foreign file planted at an entry path is refused as a miss, never executed."""

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


def test_verify_is_an_on_off_switch(monkeypatch):
    for off in ("", "0", "false", "No", "OFF", " 0 "):
        monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", off)
        assert not ref_cache.verify(), repr(off)
    for on in ("1", "true", "YES", "on", "2"):
        monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", on)
        assert ref_cache.verify(), repr(on)
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY")
    assert not ref_cache.verify()


def test_verify_0_leaves_a_hit_unrecomputed(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", "0")
    calls = []
    ref_cache.cached_reference("unit", dict(seed=5), _oracle(calls))
    before = ref_cache.stats()["verified"]
    ref_cache.cached_reference("unit", dict(seed=5), _oracle(calls))
    assert len(calls) == 1 and ref_cache.stats()["verified"] == before, "=0 is OFF: the hit is served, not recomputed"


def _entry_digest(path):
    return path.stem.rsplit("-", 1)[1]


def test_a_pickle_payload_planted_at_the_entry_path_is_refused_not_executed(monkeypatch, tmp_path):
    """The directory is shared and an entry's name is predictable, so an entry is read with torch's weights-only unpickler -- tensors
    and plain Python containers / scalars only.  A checkpoint carrying a pickle payload under this key's own digest is refused: a miss
    that recomputes and overwrites it, never an executed payload."""
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    marker = tmp_path / "payload-executed.marker"

    class _Payload:
        def __reduce__(self):  # unpickling this object calls open(marker, "w"): the marker exists iff the payload ran
            return (open, (str(marker), "w"))

    calls = []
    ref_cache.cached_reference("unit", dict(seed=6), _oracle(calls))
    (path,) = tmp_path.glob("unit-*.pt")
    torch.save(dict(digest=_entry_digest(path), key="{}", value=_Payload()), path)
    before = ref_cache.stats().get("refused", 0)
    value = ref_cache.cached_reference("unit", dict(seed=6), _oracle(calls))
    assert not marker.exists(), "the planted payload was executed"
    assert len(calls) == 2 and value["amax"] == 1.0 and ref_cache.stats()["refused"] == before + 1
    ref_cache.cached_reference("unit", dict(seed=6), _oracle(calls))  # the refused file was overwritten by a valid entry: a hit again
    assert len(calls) == 2


def test_a_foreign_file_at_the_entry_path_is_a_miss_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    calls = []
    ref_cache.cached_reference("unit", dict(seed=7), _oracle(calls))
    (path,) = tmp_path.glob("unit-*.pt")
    # not a dict at all / another key's entry / this key's digest without a value
    for foreign in (torch.zeros(3), dict(digest="another-key", value=1.0), dict(digest=_entry_digest(path))):
        torch.save(foreign, path)
        assert ref_cache.cached_reference("unit", dict(seed=7), _oracle(calls))["amax"] == 1.0
    assert len(calls) == 4


def test_a_nested_entry_round_trips_through_the_safe_loader(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDNN_TEST_REF_CACHE_VERIFY", raising=False)
    nested = (
        torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3),
        dict(n=1, x=2.5, none=None, s="s", b=True, l=[torch.zeros(2, device="cuda"), (3, 4.0)]),
        7,
    )
    calls = []

    def compute():
        calls.append(1)
        return nested

    first = ref_cache.cached_reference("nested", dict(seed=8), compute)
    second = ref_cache.cached_reference("nested", dict(seed=8), compute)
    assert len(calls) == 1 and ref_cache._same(first, second)
    assert isinstance(second, tuple) and isinstance(second[1]["l"], list) and isinstance(second[1]["l"][1], tuple) and second[1]["none"] is None
    assert second[0].device.type == "cuda" and second[1]["l"][0].device.type == "cuda"


def test_a_result_the_cache_cannot_represent_is_refused_at_store_time(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    with pytest.raises(TypeError, match=r"tensors and plain Python .* value\['obj'\] is a object"):
        ref_cache.cached_reference("unit", dict(seed=9), lambda: dict(o=torch.zeros(2), obj=object()))
    assert not list(tmp_path.glob("unit-*.pt")), "nothing the loader would refuse is ever written"
