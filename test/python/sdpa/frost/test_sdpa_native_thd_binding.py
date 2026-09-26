# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Metadata-only differential contracts for the native f16 THD binder.

The Python binder is an explicit reference here, never an exception-driven
production fallback for native plans. No test pins a heuristic or timing.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep

pytestmark = [pytest.mark.L0]


def _fixture(dtype="bfloat16", layout="NH", rank=4):
    s = prep.ThdLaunchSpec()
    s.b, s.qh, s.kh, s.d_qk, s.d_v = 4, 8, 2, 128, 128
    s.paged, s.has_sink, s.lse_padded = False, False, False
    s.has_lse, s.lse_head_major = layout is not None, layout == "HN"
    s.lse_head_stride, s.lse_stride = (16 if layout == "HN" else 0), None
    s.device_index, s.lens_form = 0, 3
    s.total_q, s.total_kv, s.off_o_desc = 16, 512, 4096
    s.expect = dict.fromkeys(("q", "k", "v", "o"), dtype)
    s.decl = {name: (h, 128, h * 128, 128, 1, h * 128) for name, h in (("q", 8), ("k", 2), ("v", 2), ("o", 8))}
    s.order = sorted(prep._FILLED_AT_BUILD | prep._FILLED_PER_CALL)
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [None] * len(s.order)
    s.template[s.index["lse_ext"]] = s.lse_head_stride
    s.template[s.index["scale_softmax_log2"]] = 0.125
    s._geometry_cache, s.native = None, None
    s.owner = object()
    frames = []
    s.fn = lambda *frame: frames.append(frame)
    facts = {}
    for i, (name, h, sq) in enumerate((("q", 8, 4), ("k", 2, 128), ("v", 2, 128), ("o", 8, 4))):
        shape, strides = ((4, h, sq, 128), (sq * h * 128, 128, h * 128, 1)) if rank == 4 else ((4 * sq, h, 128), (h * 128, 128, 1))
        facts[name] = prep.BufferFacts(0x1000 * (i + 1), dtype, (2, 0), 4 * h * sq * 128, shape, strides)
    for i, role in enumerate(("q_lens", "kv_lens")):
        facts[role] = prep.BufferFacts(0x10000 + i * 0x1000, "int32", (2, 0), 5, (5,), (1,))
    if layout:
        shape, strides = ((1, 8, 16), (128, 16, 1)) if layout == "HN" else ((4, 8, 4, 1), (32, 1, 8, 1))
        facts["lse"] = prep.BufferFacts(0x20000, "float32", (2, 0), 128, shape, strides)
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    return s, facts, frames


def _native(s, facts, workspace=0x30000, stream=17):
    return s.native.bind(prep._native_pack_from_facts(facts), prep._NATIVE_THD_INDICES, workspace, stream)


def _reference(s, facts, workspace=0x30000, stream=17):
    return prep._bind_thd_python(s, facts, workspace, stream, stream)


def _equal(s, facts, **kwargs):
    actual, reference = _native(s, facts, **kwargs), _reference(s, facts, **kwargs)
    assert (actual is None) == (reference is None)
    if actual is not None:
        assert list(actual) == reference
    return actual


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("layout", [None, "NH", "HN"])
@pytest.mark.parametrize("rank", [3, 4])
def test_native_matches_python_contract(dtype, layout, rank):
    s, facts, _ = _fixture(dtype, layout, rank)
    _equal(s, facts)
    replacement = {name: f._replace(ptr=f.ptr + 0x100000) for name, f in facts.items()}
    replacement["q"] = replacement["q"]._replace(span=8 * s.qh * s.d_qk)
    first = _equal(s, facts)
    second = _equal(s, replacement, workspace=0x40000, stream=23)
    assert first[s.index["problem_size"]][3] == 16
    assert second[s.index["problem_size"]][3] == 8
    assert first[s.index["meta_ptr"]] == 0x30000
    assert second[s.index["meta_ptr"]] == 0x40000
    assert first[s.index["stream"]] == 17 and second[s.index["stream"]] == 23


@pytest.mark.parametrize(
    "role,updates",
    [
        ("q", {"device": (1, 0)}),
        ("q", {"device": (2, 1)}),
        ("q", {"dtype": "float32"}),
        ("q", {"ptr": 4097}),
        ("q", {"span": -1}),
        ("q", {"shape": (4, 7, 4, 128)}),
        ("q", {"shape": (4, 8, 4, 127)}),
        ("q", {"strides": (4096, 128, 1024, 2)}),
        ("q", {"strides": (4096, 129, 1032, 1)}),
        ("q", {"strides": (4096, 128, 1016, 1)}),
        ("q_lens", {"dtype": "int64"}),
        ("q_lens", {"device": (1, 0)}),
        ("q_lens", {"ptr": 0x10001}),
        ("q_lens", {"span": 4}),
        ("q_lens", {"shape": (6,), "span": 6}),
        ("q_lens", {"strides": (2,), "span": 9}),
        ("kv_lens", {"shape": (4,), "span": 4}),
        ("lse", {"dtype": "bfloat16"}),
        ("lse", {"ptr": 0x20001}),
        ("lse", {"device": (2, 1)}),
        ("lse", {"span": -1}),
        ("lse", {"strides": (32, 4, 1, 1)}),
    ],
)
def test_native_rejects_invalid_runtime_contract_without_fallback(role, updates, monkeypatch):
    s, facts, frames = _fixture()
    _equal(s, facts)  # rejection must still happen after a valid warm binding
    changed = dict(facts, **{role: facts[role]._replace(**updates)})
    with pytest.raises(ValueError):
        _reference(s, changed)
    monkeypatch.setattr(prep, "_bind_thd_python", lambda *args: pytest.fail("native plans must not fall back after a validation error"))
    with pytest.raises(ValueError):
        prep.bind_thd(s, changed, 0x30000, 17, 17)
    assert frames == []


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "q_lens", "kv_lens", "lse"])
def test_native_rejects_missing_roles(role):
    s, facts, _ = _fixture()
    del facts[role]
    with pytest.raises(ValueError):
        _native(s, facts)
    with pytest.raises(ValueError):
        _reference(s, facts)


def test_native_stats_and_workspace_contracts():
    s, facts, _ = _fixture(layout="HN")
    for updates in ({"span": 127}, {"shape": (1, 8, 15)}, {"strides": (128, 17, 1)}, {"strides": (128, 16, 2)}):
        changed = dict(facts, lse=facts["lse"]._replace(**updates))
        with pytest.raises(ValueError):
            _native(s, changed)
        with pytest.raises(ValueError):
            _reference(s, changed)
    with pytest.raises(ValueError, match="workspace must be 16-byte aligned"):
        _native(s, facts, workspace=0x30001)
    s0, facts0, _ = _fixture(layout=None)
    with pytest.raises(ValueError, match="without a Stats output"):
        _native(s0, dict(facts0, lse=facts["lse"]))
    with pytest.raises(ValueError, match="without a sink"):
        _native(s, dict(facts, sinks=facts["lse"]))


def test_native_dynamic_batch_stride_capacity_and_empty_semantics():
    s, facts, _ = _fixture()
    for batch in (2, 4, 1, 4):
        changed = dict(facts)
        for role in ("q", "k", "v", "o"):
            f = facts[role]
            # Huge singleton/unused batch stride is legal and never narrowed.
            changed[role] = f._replace(shape=(batch, *f.shape[1:]), strides=(2**35, *f.strides[1:]))
        for role in ("q_lens", "kv_lens"):
            changed[role] = facts[role]._replace(shape=(batch + 1,), span=batch + 1)
        frame = _equal(s, changed)
        assert frame[s.index["problem_size"]][0] == batch
    for role in ("q", "o"):
        changed = dict(facts, **{role: facts[role]._replace(shape=(0,), strides=(1,), span=0)})
        assert _equal(s, changed) is None
    changed = dict(facts, k=facts["k"]._replace(shape=(0,), strides=(1,), span=0), v=facts["v"]._replace(shape=(0,), strides=(1,), span=0))
    frame = _equal(s, changed)
    assert frame[s.index["problem_size"]][4] == 1
    assert frame[s.index["k_ptr"]] == facts["q"].ptr
    assert frame[s.index["v_ptr"]] == facts["o"].ptr
    interleaved = dict(facts)
    for role in ("k", "v"):
        f = facts[role]
        interleaved[role] = f._replace(strides=(f.strides[0] * 2, f.strides[1], f.strides[2] * 2, 1), span=f.span * 2 - 256)
    _equal(s, interleaved)


def test_native_pack_uses_effective_dtype_but_observed_byte_capacity():
    s, facts, _ = _fixture(rank=3)
    pack = prep._native_pack_from_facts(facts)
    q = facts["q"]
    # An opaque uint8 producer is described as BF16, preserving its byte span.
    pack.set_operand(0, q.ptr, (q.span * 2,), (), 1, 8, 1, q.span * 2, 2, 0)
    declared = cudnn._pybind_module.DeclaredLayout(len(prep._NATIVE_THD_ROLES))
    declared.set(0, q.shape, q.strides, 2, 4, 16)
    pack.describe_from(declared, [])
    actual = s.native.bind(pack, prep._NATIVE_THD_INDICES, 0x30000, 17)
    assert list(actual) == _reference(s, facts)
    # A compact DLPack producer may spell strides as null/empty.
    pack.set_operand(0, q.ptr, q.shape, (), 4, 16, 1, q.span * 2, 2, 0)
    assert list(s.native.bind(pack, prep._NATIVE_THD_INDICES, 0x30000, 17)) == _reference(s, facts)


def test_native_launch_retains_owner_but_never_reuses_an_invocation_frame():
    s, facts, recorded = _fixture()

    def run(i):
        changed = {name: f._replace(ptr=f.ptr + i * 0x100000) for name, f in facts.items()}
        pack = prep._native_pack_from_facts(changed)
        assert s.native.execute(pack, prep._NATIVE_THD_INDICES, 0x30000 + i * 0x10000, 17 + i, 0.5 + i)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, range(8)))
    assert len(recorded) == 8
    for frame in recorded:
        i = frame[s.index["stream"]] - 17
        assert frame[s.index["q_ptr"]] == facts["q"].ptr + i * 0x100000
        assert frame[s.index["meta_ptr"]] == 0x30000 + i * 0x10000
        assert frame[s.index["scale_softmax_log2"]] == 0.5 + i
    # A native plan holds its original callable even if the mutable Python spec
    # is later reused/replaced; the prepared artifact is immutable.
    s.fn = lambda *args: pytest.fail("a prepared binder must retain its original entry")
    run(9)
    assert len(recorded) == 9


def test_native_unknown_metadata_span_keeps_the_bare_pointer_contract():
    s, facts, _ = _fixture(layout="HN")
    changed = dict(facts)
    for role in ("q_lens", "kv_lens", "lse"):
        changed[role] = facts[role]._replace(span=-1, device=(-1, -1))
    _equal(s, changed)


@pytest.mark.parametrize("field,value", [("b", 0), ("qh", 0), ("kh", 0), ("device_index", -1), ("lens_form", 4), ("total_q", -2), ("total_kv", -2)])
def test_native_rejects_malformed_plan(field, value):
    s, _, _ = _fixture()
    setattr(s, field, value)
    with pytest.raises(ValueError):
        cudnn._pybind_module._SdpaThdBinder(s)


def test_native_rejects_zero_divisor_declaration_and_overflow():
    s, facts, _ = _fixture()
    s.decl["q"] = (1, 0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="declaration"):
        cudnn._pybind_module._SdpaThdBinder(s)
    s, facts, _ = _fixture()
    facts["q"] = facts["q"]._replace(shape=(2**62, 8, 4, 128))
    with pytest.raises(ValueError, match="int64"):
        _native(s, facts)


def test_native_asymmetric_head_dims_and_int64_runtime_token_strides():
    s, facts, _ = _fixture()
    s.d_qk = 192
    for role in ("q", "k"):
        f = facts[role]
        h = f.shape[1]
        sq = f.shape[2]
        s.decl[role] = (h, 192, h * 192, 192, 1, h * 192)
        facts[role] = f._replace(shape=(4, h, sq, 192), strides=(sq * h * 192, 192, h * 192, 1), span=4 * h * sq * 192)
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    _equal(s, facts)
    q = facts["q"]
    token_stride = 2**32
    facts["q"] = q._replace(strides=(2**36, 192, token_stride, 1), span=15 * token_stride + 8 * 192)
    frame = _equal(s, facts)
    assert frame[s.index["q_strides"]] == (token_stride, token_stride, 192)
    assert frame[s.index["problem_size"]][3] == 16
