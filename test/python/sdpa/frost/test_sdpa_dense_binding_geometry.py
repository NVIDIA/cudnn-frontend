# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Geometry memoization must never reuse storage observations or launch bindings."""

from types import SimpleNamespace

import pytest

from cudnn.sdpa.fwd import config_sm100, prepared
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


def _facts(**changes):
    return prepared.BufferFacts(4096, "float16", (2, 0), 128, (2, 2, 4, 8), (64, 8, 16, 1))._replace(**changes)


def _role(facts, *, b=2, heads=2, d=8, s_max=4, tma=True):
    return prepared._dense_role(SimpleNamespace(b=b, device_index=0), {"q": facts}, "q", heads, d, s_max, "float16", tma=tma)


def test_dense_geometry_reuses_validation_and_rebinds_storage(monkeypatch):
    cache = getattr(prepared, "_dense_role_layout", None)
    if cache is not None:
        cache.cache_clear()
    original = config_sm100.dense_bind_strides
    calls = []

    def validate(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(config_sm100, "dense_bind_strides", validate)
    assert _role(_facts()).ptr == 4096
    assert _role(_facts(ptr=8192)).ptr == 8192
    assert len(calls) == 1
    for changed, error in (({"ptr": 4097}, "aligned"), ({"span": 127}, "spans"), ({"device": (2, 1)}, "device"), ({"dtype": "float32"}, "dtype")):
        with pytest.raises(ValueError, match=error):
            _role(_facts(**changed))
    assert len(calls) == 1
    assert _role(_facts(shape=(2, 2, 3, 8))).s == 3
    assert len(calls) == 2


@pytest.mark.parametrize("change", [dict(b=1), dict(heads=3), dict(d=16), dict(s_max=3)])
def test_dense_geometry_keeps_plan_envelope_in_key(change):
    _role(_facts())
    with pytest.raises(ValueError):
        _role(_facts(), **change)


def test_dense_geometry_preserves_wide_strides_and_span():
    stride = 2**32 + 64
    f = _facts(strides=(stride, 8, 16, 1), span=stride + 64)
    assert _role(f).strides[0] == stride
    with pytest.raises(ValueError, match="spans"):
        _role(f._replace(span=stride + 63))


def test_dense_geometry_distinguishes_tma_from_plain_output():
    f = _facts(strides=(64, 32, 8, 1))
    assert _role(f, tma=False).strides == (64, 8, 32)
    with pytest.raises(ValueError, match="zero-copy"):
        _role(f, tma=True)


def test_stats_geometry_keeps_current_pointer_device_span_and_alias_checks():
    spec = SimpleNamespace(qh=2, device_index=0)
    f = prepared.BufferFacts(4096, "float32", (2, 0), 16, (2, 2, 4), (8, 4, 1))
    bind = lambda fact, b=2, sq=4: prepared._dense_lse(spec, fact, b, sq, required=True)
    assert bind(f)[0] == 4096
    assert bind(f._replace(ptr=8192))[0] == 8192
    for changed, error in (({"ptr": 4097}, "aligned"), ({"span": 15}, "spans"), ({"device": (2, 1)}, "device"), ({"strides": (8, 1, 1)}, "alias")):
        with pytest.raises(ValueError, match=error):
            bind(f._replace(**changed))
    with pytest.raises(ValueError, match="must be"):
        bind(f, sq=5)
    with pytest.raises(ValueError, match="required"):
        bind(None)
