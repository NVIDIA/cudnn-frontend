# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Host-only contracts for reuse of a prepared THD plan's pure geometry calculation."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cudnn.sdpa.fwd import prepared as prep

pytestmark = [pytest.mark.L0]


def _fixture(*, padded_stats=False):
    b, h, hk, sq, sk, d = 4, 8, 2, 4, 128, 128
    spec = prep.ThdLaunchSpec()
    spec.b, spec.qh, spec.kh, spec.d_qk, spec.d_v = b, h, hk, d, d
    spec.paged, spec.lens_form = False, 3
    spec.has_lse, spec.has_sink = padded_stats, False
    spec.lse_padded, spec.lse_head_major = padded_stats, False
    spec.lse_head_stride, spec.lse_stride = 0, (h * sq, sq, 1)
    spec.s_q_max, spec.total_q, spec.total_kv = sq, b * sq, b * sk
    spec.device_index, spec.off_o_desc, spec.neg_inf = 0, 4096, 0xFF800000
    spec.expect = dict.fromkeys(("q", "k", "v", "o"), "bfloat16")
    spec.decl = {name: (hh, d, hh * d, d, 1, hh * d) for name, hh in (("q", h), ("k", hk), ("v", hk), ("o", h))}
    spec._geometry_cache = None
    spec._dummies = {"sinks": SimpleNamespace(data_ptr=lambda: 0x100000), "v_stub": SimpleNamespace(data_ptr=lambda: 0x200000)}
    spec.order = sorted(prep._FILLED_AT_BUILD | prep._FILLED_PER_CALL)
    spec.index = {name: i for i, name in enumerate(spec.order)}
    spec.template = [None] * len(spec.order)
    facts = {}
    for i, (name, heads, seq) in enumerate((("q", h, sq), ("k", hk, sk), ("v", hk, sk), ("o", h, sq))):
        facts[name] = prep.BufferFacts(0x1000 * (i + 1), "bfloat16", (2, 0), b * heads * seq * d, (b, heads, seq, d), (seq * heads * d, d, heads * d, 1))
    for i, name in enumerate(("q_lens", "kv_lens")):
        facts[name] = prep.BufferFacts(0x10000 + i * 0x1000, "int32", (2, 0), b + 1, (b + 1,), (1,))
    if padded_stats:
        facts["lse"] = prep.BufferFacts(0x20000, "float32", (2, 0), b * h * sq, (b, h, sq), spec.lse_stride)
    return spec, facts


def _bind(spec, facts, workspace=0x30000, stream=17):
    return prep.bind_thd(spec, facts, workspace, stream, stream)


def test_thd_geometry_reuse_rebinds_addresses_workspace_stream_and_storage_span():
    spec, facts = _fixture()
    before = _bind(spec, facts)
    geometry = spec._geometry_cache[1]
    rebound = {name: value._replace(ptr=value.ptr + 0x100000) for name, value in facts.items()}
    rebound["q"] = rebound["q"]._replace(span=8 * spec.qh * spec.d_qk)
    after = _bind(spec, rebound, workspace=0x40000, stream=23)
    assert spec._geometry_cache[1] is geometry, "pure geometry can be reused even when allocations change"
    ix = spec.index
    for role, slot in (("q", "q_ptr"), ("k", "k_ptr"), ("v", "v_ptr"), ("o", "o_ptr"), ("q_lens", "thd_q_lens_ptr"), ("kv_lens", "thd_kv_lens_ptr")):
        assert after[ix[slot]] == rebound[role].ptr
        assert before[ix[slot]] == facts[role].ptr
    assert before[ix["problem_size"]][3] == 16
    assert after[ix["problem_size"]][3] == 8, "the current observed span still bounds packed capacity"
    assert after[ix["meta_ptr"]] == 0x40000 and after[ix["o_desc_ptr"]] == 0x40000 + spec.off_o_desc
    assert after[ix["stream"]] == 23 and before[ix["stream"]] == 17
    with pytest.raises(TypeError):
        geometry.roles["q"] = (1, 2, 3, 4)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"device": (1, 0)}, "CUDA device"),
        ({"device": (2, 1)}, "CUDA device"),
        ({"dtype": "float32"}, "dtype"),
        ({"ptr": 4097}, "16-byte aligned"),
        ({"span": -1}, "sized buffer"),
        ({"shape": (4, 7, 4, 128)}, "effective shape"),
        ({"strides": (4096, 128, 1024, 2)}, "elem stride 1"),
    ],
)
def test_thd_geometry_warm_cache_does_not_skip_buffer_or_layout_validation(updates, match):
    spec, facts = _fixture()
    _bind(spec, facts)
    with pytest.raises(ValueError, match=match):
        _bind(spec, dict(facts, q=facts["q"]._replace(**updates)))
    with pytest.raises(ValueError, match="workspace must be 16-byte aligned"):
        _bind(spec, facts, workspace=0x30001)


def test_thd_geometry_batch_override_small_large_small_and_rejection():
    spec, facts = _fixture()

    def run(batch):
        changed = dict(facts)
        for role in ("q", "k", "v", "o"):
            f = facts[role]
            changed[role] = f._replace(shape=(batch, *f.shape[1:]))
        for role in ("q_lens", "kv_lens"):
            changed[role] = facts[role]._replace(shape=(batch + 1,), span=batch + 1)
        return _bind(spec, changed)[spec.index["problem_size"]][0]

    assert [run(b) for b in (2, 4, 2)] == [2, 4, 2]
    with pytest.raises(ValueError, match="prepared for 1..4"):
        run(5)
    with ThreadPoolExecutor(2) as pool:
        batches = [2, 4] * 20
        assert list(pool.map(run, batches)) == batches


def test_thd_geometry_reuse_keeps_padded_stats_seed_per_call(monkeypatch):
    spec, facts = _fixture(padded_stats=True)
    seeds = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *args: seeds.append(args))
    _bind(spec, facts, stream=17)
    geometry = spec._geometry_cache[1]
    replacement = dict(facts, lse=facts["lse"]._replace(ptr=0x50000))
    _bind(spec, replacement, stream=23)
    assert spec._geometry_cache[1] is geometry
    assert [(ptr, stream) for ptr, _, _, stream in seeds] == [(0x20000, 17), (0x50000, 23)]
