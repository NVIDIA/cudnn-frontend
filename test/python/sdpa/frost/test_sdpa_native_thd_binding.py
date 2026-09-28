# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Metadata-only differential contracts for the native f16 THD binder.

The Python binder is an explicit reference here, never an exception-driven
production fallback for native plans. No test pins a heuristic or timing.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep

pytestmark = [pytest.mark.L0]


def _fixture(dtype="bfloat16", layout="NH", rank=4):
    s = prep.ThdLaunchSpec()
    s.b, s.qh, s.kh, s.d_qk, s.d_v = 4, 8, 2, 128, 128
    s.cga_tile_m = 512
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
    s.template[s.index["n_thd_units"]] = 32
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


def _choice_fixture(dtype="bfloat16", layout="HN", qh=16, kh=2):
    base, _, _ = _fixture(dtype, layout)
    base.b, base.qh, base.kh, base.d_qk, base.d_v = 8, qh, kh, 192, 128
    base.total_q, base.total_kv, base.quant = None, None, None
    base.s_q_max, base.lse_head_stride, base.lse_stride_override = 65536, 0, True
    base.decl = {role: (h, d, h * d, d, 1, h * d) for role, h, d in (("q", qh, 192), ("k", kh, 192), ("v", kh, 128), ("o", qh, 128))}
    variants, launches = [], []
    for i, tile in enumerate((128, 512)):
        spec = copy(base)
        spec.cga_tile_m = tile
        spec.template = list(base.template)
        spec.template[spec.index["n_thd_units"]] = 1024
        spec.fn = lambda *frame, index=i: launches.append((index, frame))
        spec.native = cudnn._pybind_module._SdpaThdBinder(spec)
        variants.append(spec)
    return variants, launches


def _choice_facts(spec, batch, max_q, total, max_kv=64):
    facts = {}
    for i, role in enumerate(("q", "k", "v", "o")):
        h, d = spec.decl[role][:2]
        sq = max_q if role in ("q", "o") else max_kv
        tokens = total if role in ("q", "o") else batch * sq
        facts[role] = prep.BufferFacts(0x1000 * (i + 1), spec.expect[role], (2, 0), tokens * h * d, (batch, h, sq, d), (sq * h * d, d, h * d, 1))
    for i, role in enumerate(("q_lens", "kv_lens")):
        facts[role] = prep.BufferFacts(0x10000 + i * 0x1000, "int32", (2, 0), batch + 1, (batch + 1,), (1,))
    if spec.has_lse:
        strides = (spec.qh * total, total, 1, 1) if spec.lse_head_major else (spec.qh * max_q, 1, spec.qh, 1)
        facts["lse"] = prep.BufferFacts(0x20000, "float32", (2, 0), spec.qh * total, (batch, spec.qh, max_q, 1), strides)
    return facts


def _packed_split_fixture(layout="HN", splits=8):
    variants, launches = _choice_fixture(layout=layout)
    s = variants[0]
    s.order = list(s.order) + ["lse_partial_ptr", "partial_o_strides"]
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = list(s.template) + [None, None]
    s.template[s.index["n_thd_units"]] = 148
    off_o, capacity = 8192, 512
    off_lse = off_o + splits * capacity * s.qh * 128 * 4
    s.split_workspace = prep.ThdSplitWorkspace(splits, capacity, off_o, off_lse)
    s.scratch_bytes = off_lse + splits * capacity * s.qh * 4
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    return s, launches


@pytest.mark.parametrize("layout", [None, "NH", "HN"])
@pytest.mark.parametrize("splits", [2, 4, 8])
def test_packed_split_binds_fixed_regions_and_independent_frames(layout, splits):
    s, launches = _packed_split_fixture(layout, splits)
    saved = []
    for batch, total in ((1, 64), (1, 129), (4, 257), (2, 512)):
        facts = _choice_facts(s, batch, total, total)
        workspace = 0x4000000 * (1 + len(saved))
        frame = _equal(s, facts, workspace=workspace, stream=29 + len(saved))
        assert frame[s.index["o_partial_ptr"]] == workspace + s.split_workspace.off_o
        assert frame[s.index["lse_partial_ptr"]] == workspace + s.split_workspace.off_lse
        assert frame[s.index["partial_o_strides"]] == (total * s.qh * 128, s.qh * 128, 128)
        assert frame[s.index["n_thd_units"]] == min(148, ((total - 1) // 128 + batch) * s.qh * splits)
        saved.append((frame, tuple(frame)))
        s.native.execute(prep._native_pack_from_facts(facts), prep._NATIVE_THD_INDICES, workspace, 29 + len(saved) - 1)
        assert launches[-1][1] == tuple(frame)
        assert all(tuple(old) == unchanged for old, unchanged in saved)
    assert s.template[s.index["o_partial_ptr"]] is None
    assert s.template[s.index["lse_partial_ptr"]] is None


def test_packed_split_rejects_capacity_before_launch_and_keeps_empty_semantics():
    s, launches = _packed_split_fixture()
    facts = _choice_facts(s, 1, 513, 513)
    for binder in (_native, _reference):
        with pytest.raises(ValueError, match="packed Q capacity"):
            binder(s, facts)
        empty = dict(facts, q=facts["q"]._replace(span=0))
        assert binder(s, empty) is None
        valid = _choice_facts(s, 1, 64, 64)
        with pytest.raises(ValueError, match="non-null workspace"):
            binder(s, valid, workspace=0)
    assert not launches


@pytest.mark.parametrize("bad", ["capacity", "offset", "overlap", "reservation", "overflow", "tile"])
def test_packed_split_rejects_invalid_workspace_contract(bad):
    s, _ = _packed_split_fixture()
    if bad == "capacity":
        s.split_workspace = s.split_workspace._replace(capacity=0)
    elif bad == "offset":
        s.split_workspace = s.split_workspace._replace(off_o=4096)
    elif bad == "overlap":
        s.split_workspace = s.split_workspace._replace(off_lse=s.split_workspace.off_o)
    elif bad == "reservation":
        s.scratch_bytes -= 1
    elif bad == "overflow":
        s.split_workspace = s.split_workspace._replace(splits=2**62)
    else:
        s.cga_tile_m = 512
    with pytest.raises(ValueError, match="packed split|int64"):
        cudnn._pybind_module._SdpaThdBinder(s)


def test_cga_policy_cannot_silently_acquire_split_semantics():
    split, _ = _packed_split_fixture()
    variants, _ = _choice_fixture()
    with pytest.raises(ValueError, match="unsplit"):
        cudnn._pybind_module._SdpaThdPlanChoices(split, variants[1], 148, 2)


def _split_member(base, launches, splits, capacity, index):
    split = copy(base)
    split.order = list(split.order) + ["lse_partial_ptr", "partial_o_strides"]
    split.index = {name: i for i, name in enumerate(split.order)}
    split.template = list(split.template) + [None, None]
    split.template[split.index["n_thd_units"]] = 148
    off_lse = 8192 + splits * capacity * split.qh * 128 * 4
    split.split_workspace = prep.ThdSplitWorkspace(splits, capacity, 8192, off_lse)
    split.scratch_bytes = off_lse + splits * capacity * split.qh * 4
    split.fn = lambda *frame: launches.append((index, frame))
    split.native = cudnn._pybind_module._SdpaThdBinder(split)
    return split


def _split_policy_fixture(dtype="bfloat16", layout="HN", capacity=128):
    variants, launches = _choice_fixture(dtype, layout, kh=16)
    split = _split_member(variants[0], launches, 8, capacity, 2)
    return [*variants, split], launches


@pytest.mark.parametrize("heads,short_splits,long_splits", [(4, 32, 16), (8, 16, 8), (16, 8, 4)])
@pytest.mark.parametrize("dtype,layout", [("bfloat16", "HN"), ("float16", "NH"), ("bfloat16", None)])
def test_balanced_split_policy_binds_four_artifacts_and_keeps_old_frames(heads, short_splits, long_splits, dtype, layout):
    variants, launches = _choice_fixture(dtype, layout, qh=heads, kh=heads)
    variants += [_split_member(variants[0], launches, short_splits, 128, 2), _split_member(variants[0], launches, long_splits, 256, 3)]
    assert variants[2].scratch_bytes == variants[3].scratch_bytes == 8192 + 16384 * (128 + 1) * 4
    choices = cudnn._pybind_module._SdpaThdPlanChoices(*variants[:2], 148, 2, variants[2], 3, variants[3])
    cases = [(1, q, 32768, selected) for q, selected in ((1, 2), (64, 2), (127, 2), (128, 2), (129, 3), (255, 3), (256, 3), (257, 0))]
    cases += [(1, 128, 32767, 0), (1, 129, 131072, 3), (2, 64, 32768, 0), (1, 32768 // heads, 32768, 1)]
    saved = []
    for batch, total, kv, expected in cases:
        facts = _choice_facts(variants[0], batch, total, total, kv)
        workspace, stream = 0x4000000 * (len(saved) + 1), 31 + len(saved)
        pack = prep._native_pack_from_facts(facts)
        index, frame = choices.bind(pack, prep._NATIVE_THD_INDICES, workspace, stream)
        assert index == choices.select_index(pack, prep._NATIVE_THD_INDICES) == expected
        assert list(frame) == _reference(variants[index], facts, workspace, stream)
        assert choices.execute(pack, prep._NATIVE_THD_INDICES, workspace, stream)
        assert launches[-1] == (index, tuple(frame))
        saved.append((frame, tuple(frame)))
        assert all(tuple(old) == original for old, original in saved)


@pytest.mark.parametrize(
    "bad", ["missing", "unrecorded", "small_capacity", "large_capacity", "small_splits", "large_splits", "contract", "heads", "same_member"]
)
def test_balanced_split_policy_rejects_incomplete_or_incompatible_members(bad):
    variants, launches = _choice_fixture(qh=8, kh=8)
    small = _split_member(variants[0], launches, 16, 128, 2)
    large = _split_member(variants[0], launches, 8, 256, 3)
    policy = 3
    if bad == "missing":
        large = None
    elif bad == "same_member":
        large = small
    elif bad == "unrecorded":
        policy = 2
    elif bad == "small_capacity":
        small.split_workspace = small.split_workspace._replace(capacity=256)
    elif bad == "large_capacity":
        large.split_workspace = large.split_workspace._replace(capacity=128)
    elif bad == "small_splits":
        small.split_workspace = small.split_workspace._replace(splits=8)
    elif bad == "large_splits":
        large.split_workspace = large.split_workspace._replace(splits=16)
    elif bad == "contract":
        large.total_q = 64
    else:
        for member in (*variants, small, large):
            member.qh = member.kh = 32
    with pytest.raises(ValueError, match="split policy|plan choices must share"):
        cudnn._pybind_module._SdpaThdPlanChoices(*variants, 148, 2, small, policy, large)


def test_balanced_split_requires_matching_native_support_without_breaking_older_policies(monkeypatch):
    variants, launches = _choice_fixture(kh=16)
    variants += [_split_member(variants[0], launches, 8, 128, 2), _split_member(variants[0], launches, 4, 256, 3)]
    members = [SimpleNamespace(spec=s, _roles=("q",), _uids=(1,)) for s in variants]
    calls = []

    def old_factory(*args):
        calls.append(args)
        assert len(args) == 6
        return object()

    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdPlanChoices", old_factory)
    prep.PreparedThdChoices(members[:3], 148, policy=2, split_policy=1)
    with pytest.raises(NotImplementedError, match="matching native"):
        prep.PreparedThdChoices(members, 148, policy=2, split_policy=3)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype,layout", [("bfloat16", "HN"), ("float16", "NH"), ("bfloat16", None)])
@pytest.mark.parametrize("cga_policy", [1, 2])
@pytest.mark.parametrize("split_policy", [1, 2])
def test_split_policy_binds_selected_artifact_without_mutating_old_frames(dtype, layout, cga_policy, split_policy):
    capacity = 128 if split_policy == 1 else 256
    variants, launches = _split_policy_fixture(dtype, layout, capacity)
    choices = cudnn._pybind_module._SdpaThdPlanChoices(*variants[:2], 148, cga_policy, variants[2], split_policy)
    saved = []
    # This pins the explicitly recorded policy contract, not heuristic ranking.
    cases = [(1, total, 32768, 2 if total <= capacity else 0) for total in (64, 128, 129, 255, 256, 257)]
    cases += [(1, capacity, 131072, 2), (1, 64, 32767, 0), (2, 64, 32768, 0), (1, 2048, 32768, 1)]
    for batch, total, kv, expected in cases:
        facts = _choice_facts(variants[0], batch, total, total, kv)
        workspace, stream = 0x4000000 * (len(saved) + 1), 31 + len(saved)
        pack = prep._native_pack_from_facts(facts)
        index, frame = choices.bind(pack, prep._NATIVE_THD_INDICES, workspace, stream)
        assert index == choices.select_index(pack, prep._NATIVE_THD_INDICES) == expected
        assert list(frame) == _reference(variants[index], facts, workspace, stream)
        assert choices.execute(pack, prep._NATIVE_THD_INDICES, workspace, stream)
        assert launches[-1] == (index, tuple(frame))
        saved.append((frame, tuple(frame)))
        assert all(tuple(old) == original for old, original in saved)
    empty = dict(facts, q=facts["q"]._replace(span=0))
    before = len(launches)
    assert not choices.execute(prep._native_pack_from_facts(empty), prep._NATIVE_THD_INDICES, workspace, stream)
    assert len(launches) == before


@pytest.mark.parametrize("role,updates", [("q", {"device": (1, 0)}), ("kv_lens", {"span": 1}), ("o", {"ptr": 0x4001}), ("lse", {"dtype": "bfloat16"})])
def test_split_policy_preserves_selected_binder_validation(role, updates):
    variants, launches = _split_policy_fixture()
    choices = cudnn._pybind_module._SdpaThdPlanChoices(*variants[:2], 148, 2, variants[2], 1)
    facts = _choice_facts(variants[0], 1, 64, 64, 32768)
    facts[role] = facts[role]._replace(**updates)
    with pytest.raises(ValueError):
        choices.execute(prep._native_pack_from_facts(facts), prep._NATIVE_THD_INDICES, 0x4000000, 31)
    assert not launches


@pytest.mark.parametrize("bad", ["unrecorded", "unknown", "missing", "capacity", "splits", "contract"])
@pytest.mark.parametrize("split_policy", [1, 2])
def test_split_policy_rejects_incompatible_or_unrecorded_members(bad, split_policy):
    variants, _ = _split_policy_fixture(capacity=128 if split_policy == 1 else 256)
    single, pair, split = variants
    policy = split_policy
    if bad == "unrecorded":
        policy = 0
    elif bad == "unknown":
        policy = 4
    elif bad == "missing":
        split = None
    elif bad == "capacity":
        split.split_workspace = split.split_workspace._replace(capacity=256 if split_policy == 1 else 128)
    elif bad == "splits":
        split.split_workspace = split.split_workspace._replace(splits=4)
    else:
        split.total_q = 64
    with pytest.raises(ValueError, match="split policy|plan choices must share|split member"):
        cudnn._pybind_module._SdpaThdPlanChoices(single, pair, 148, 2, split, policy)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("layout", [None, "NH", "HN"])
@pytest.mark.parametrize("policy", [1, 2])
def test_native_plan_choices_keep_frames_and_explicit_artifacts_independent(dtype, layout, policy):
    variants, launches = _choice_fixture(dtype, layout)
    choices = cudnn._pybind_module._SdpaThdPlanChoices(*variants, 148, policy)
    saved = []
    for batch, maximum, total, expected in ((1, 1024, 1024, 0), (2, 513, 640, 0), (1, 2048, 2048, 1), (4, 128, 319, 0), (8, 512, 4096, 1)):
        facts = _choice_facts(variants[0], batch, maximum, total)
        workspace, stream = 0x30000 + len(saved) * 0x1000, 17 + len(saved)
        pack = prep._native_pack_from_facts(facts)
        index, frame = choices.bind(pack, prep._NATIVE_THD_INDICES, workspace, stream)
        assert index == choices.select_index(pack, prep._NATIVE_THD_INDICES) == expected
        reference = _reference(variants[index], facts, workspace, stream)
        assert list(frame) == reference
        saved.append((frame, tuple(frame)))
        choices.execute(pack, prep._NATIVE_THD_INDICES, workspace, stream)
        assert launches[-1] == (index, tuple(reference))
        assert all(tuple(old) == unchanged for old, unchanged in saved)
    empty = dict(facts, q=facts["q"]._replace(span=0))
    before = len(launches)
    assert not choices.execute(prep._native_pack_from_facts(empty), prep._NATIVE_THD_INDICES, workspace, stream)
    assert len(launches) == before


@pytest.mark.parametrize(
    "role,updates",
    [
        ("q", {"device": (1, 0)}),
        ("q", {"dtype": "float32"}),
        ("o", {"ptr": 0x4001}),
        ("q_lens", {"span": 0}),
        ("kv_lens", {"shape": (3,)}),
        ("lse", {"dtype": "bfloat16"}),
        ("lse", {"span": 3}),
        ("k", {"ptr": 0x2001}),
    ],
)
@pytest.mark.parametrize("policy", [1, 2])
def test_native_plan_choices_preserve_runtime_rejections(role, updates, policy):
    variants, launches = _choice_fixture()
    choices = cudnn._pybind_module._SdpaThdPlanChoices(*variants, 148, policy)
    for total in (1024, 2048):
        facts = _choice_facts(variants[0], 1, total, total)
        facts[role] = facts[role]._replace(**updates)
        for variant in variants:
            with pytest.raises(ValueError):
                _native(variant, facts)
        with pytest.raises(ValueError):
            choices.execute(prep._native_pack_from_facts(facts), prep._NATIVE_THD_INDICES, 0x30000, 17)
    assert launches == []


@pytest.mark.parametrize("policy", [1, 2])
def test_native_cga_policy_preserves_one_wave_and_equal_wave_rules(policy):
    variants, _ = _choice_fixture(qh=32)
    choices = cudnn._pybind_module._SdpaThdPlanChoices(*variants, 148, policy)
    # Two packed sequences fit in two waves of either geometry. Policy1
    # preserves its original pair choice; policy2 selects single on this tie.
    facts = _choice_facts(variants[0], 2, 513, 640)
    assert choices.select_index(prep._native_pack_from_facts(facts), prep._NATIVE_THD_INDICES) == (1 if policy == 1 else 0)
    # One sequence with 1024 rows needs two single waves but one pair wave.
    facts = _choice_facts(variants[0], 1, 1024, 1024)
    assert choices.select_index(prep._native_pack_from_facts(facts), prep._NATIVE_THD_INDICES) == 1


@pytest.mark.parametrize("policy", [0, 3])
def test_native_cga_policy_rejects_unknown_values(policy):
    variants, _ = _choice_fixture()
    with pytest.raises(ValueError, match="unknown native THD CGA policy"):
        cudnn._pybind_module._SdpaThdPlanChoices(*variants, 148, policy)


def test_native_plan_choices_reject_incompatible_specializations():
    variants, _ = _choice_fixture()
    for name, value in (("qh", 32), ("total_q", 2048), ("device_index", 1), ("lse_stride_override", False)):
        changed = copy(variants[1])
        setattr(changed, name, value)
        with pytest.raises(ValueError, match="plan choices must share"):
            cudnn._pybind_module._SdpaThdPlanChoices(variants[0], changed, 148)


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


@pytest.mark.parametrize(
    "field,value", [("b", 0), ("qh", 0), ("kh", 0), ("device_index", -1), ("lens_form", 4), ("total_q", -2), ("total_kv", -2), ("cga_tile_m", 0)]
)
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


@pytest.mark.parametrize("batch,tokens,expected_limit", [(1, 8192, 128), (4, 512, 32), (4, 1, 32)])
def test_native_thd_launch_bound_uses_current_capacity(batch, tokens, expected_limit):
    """A cache-shape envelope must not launch millions of dead THD units."""
    s, facts, _ = _fixture(layout=None, rank=3)
    s.b, s.s_q_max, s.total_q = 4096, 65536, None
    declared_units = 4096 * 128 * s.qh
    s.template[s.index["n_thd_units"]] = declared_units
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    for role in ("q", "o"):
        facts[role] = facts[role]._replace(shape=(tokens, s.qh, 128), span=tokens * s.qh * 128)
    for role in ("q_lens", "kv_lens"):
        facts[role] = facts[role]._replace(shape=(batch + 1,), span=batch + 1)
    before = list(s.template)
    frame = _equal(s, facts)
    assert frame[s.index["n_thd_units"]] <= expected_limit
    assert frame[s.index["n_thd_units"]] >= s.qh
    assert s.template == before, "launch bounds belong to the invocation, not the shared plan"


def test_native_thd_launch_bound_retains_persistent_cap():
    s, facts, _ = _fixture(layout=None)
    s.template[s.index["n_thd_units"]] = 7
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    assert _equal(s, facts)[s.index["n_thd_units"]] == 7
    assert s.template[s.index["n_thd_units"]] == 7


def test_native_dynamic_hn_stride_keeps_invocation_frames_independent():
    s, facts, recorded = _fixture(layout="HN")
    s.lse_stride_override = True
    s.native = cudnn._pybind_module._SdpaThdBinder(s)

    def run(i):
        stride = 16 + i
        changed = dict(facts)
        changed["lse"] = facts["lse"]._replace(ptr=0x20000 + 0x1000 * i, shape=(1, 8, stride), strides=(8 * stride, stride, 1), span=8 * stride)
        frame = _equal(s, changed, stream=17 + i)
        assert frame[s.index["lse_ext"]] == stride
        assert s.native.execute(prep._native_pack_from_facts(changed), prep._NATIVE_THD_INDICES, 0x30000, 17 + i)
        return frame

    with ThreadPoolExecutor(2) as pool:
        frames = list(pool.map(run, range(8)))
    assert len(recorded) == 8
    for i, frame in enumerate(frames):
        assert frame[s.index["lse_ext"]] == 16 + i
        assert frame[s.index["lse_ptr"]] == 0x20000 + 0x1000 * i
    assert s.template[s.index["lse_ext"]] == s.lse_head_stride == 16
    for updates in ({"span": 127}, {"strides": (128, 17, 1)}, {"strides": (128, 0, 1)}, {"strides": (128, 16, 2)}):
        changed = dict(facts, lse=facts["lse"]._replace(**updates))
        with pytest.raises(ValueError):
            _native(s, changed)
        with pytest.raises(ValueError):
            _reference(s, changed)


def _paged_fixture(hnd=False, layout="NH", table_rank=4, dtype="bfloat16", kh=2):
    s, facts, frames = _fixture(dtype=dtype, layout=layout)
    s.paged, s.paged_hnd, s.page_size, s.kh = True, hnd, 16, kh
    s.lens_form = 1  # cumulative Q, direct per-sequence KV lengths
    facts["kv_lens"] = facts["kv_lens"]._replace(shape=(4,), strides=(1,), span=4)
    for role in ("k", "v"):
        s.decl[role] = (kh, 128, kh * 128, 128, 1, kh * 128)
        strides = (16 * kh * 128, 16 * 128, 128, 1) if hnd else (16 * kh * 128, 128, kh * 128, 1)
        facts[role] = facts[role]._replace(shape=(8, kh, 16, 128), strides=strides, span=8 * kh * 16 * 128)
    shape, strides = ((4, 1, 4, 1), (4, 4, 1, 1)) if table_rank == 4 else ((4, 4), (4, 1))
    for i, role in enumerate(("block_table", "block_table_v")):
        facts[role] = prep.BufferFacts(0x40000 + i * 0x1000, "int32", (2, 0), 16, shape, strides)
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    return s, facts, frames


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("layout", [None, "NH", "HN"])
@pytest.mark.parametrize("table_rank", [2, 4])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_native_paged_matches_python_and_rebinds(hnd, layout, table_rank, dtype):
    s, facts, _ = _paged_fixture(hnd, layout, table_rank, dtype)
    original = tuple(s.template)
    first = _equal(s, facts)
    changed = {name: f._replace(ptr=f.ptr + 0x100000) for name, f in facts.items()}
    for role in ("k", "v"):
        f = changed[role]
        bs = 2**32 + f.strides[0]
        changed[role] = f._replace(strides=(bs, *f.strides[1:]), span=7 * bs + f.strides[0])
    for role in ("block_table", "block_table_v"):
        f = changed[role]
        changed[role] = f._replace(shape=(4, 4), strides=(0, 2**33), span=-1)
    second = _equal(s, changed, workspace=0x50000, stream=29)
    assert first[s.index["problem_size"]][4] == 64
    assert second[s.index["table_strides"]] == (0, 2**33)
    assert second[s.index["block_table_ptr"]] == changed["block_table"].ptr
    assert second[s.index["k_strides"]][0] > 2**32
    assert tuple(s.template) == original


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("role", ["k", "v", "block_table", "block_table_v"])
@pytest.mark.parametrize("defect", ["span", "dtype", "device", "pointer", "stride", "shape", "missing"])
def test_native_paged_revalidates_each_call(hnd, role, defect, monkeypatch):
    s, facts, frames = _paged_fixture(hnd)
    _equal(s, facts)
    f = facts[role]
    if defect == "missing":
        changed = dict(facts)
        del changed[role]
    else:
        is_pool = role in ("k", "v")
        update = {
            "span": dict(span=f.span - 1),
            "dtype": dict(dtype="float32"),
            "device": dict(device=(2, 1)),
            "pointer": dict(ptr=f.ptr + 1),
            "stride": dict(strides=(f.strides[0] + 1, *f.strides[1:])) if is_pool else dict(strides=(4, 4, -1, 1)),
            "shape": dict(shape=(0, *f.shape[1:])),
        }[defect]
        changed = dict(facts, **{role: f._replace(**update)})
    with pytest.raises(ValueError):
        _reference(s, changed)
    monkeypatch.setattr(prep, "_bind_thd_python", lambda *args: pytest.fail("native paged plans must not fall back"))
    with pytest.raises(ValueError):
        prep.bind_thd(s, changed, 0x30000, 17, 17)
    assert frames == []


@pytest.mark.parametrize("hnd", [False, True])
def test_native_paged_singleton_strides_and_empty_q(hnd):
    s, facts, _ = _paged_fixture(hnd=hnd, kh=1)
    for role in ("k", "v"):
        f = facts[role]
        # The head axis is not stepped. Preserve the compiled ordering kind.
        hs = 2**35 if hnd else 128
        facts[role] = f._replace(shape=(1, 1, 16, 128), strides=(2**37, hs, 128, 1), span=2048)
    _equal(s, facts)
    empty = dict(facts, q=facts["q"]._replace(shape=(0,), strides=(1,), span=0))
    del empty["block_table"]
    del empty["block_table_v"]
    assert _equal(s, empty) is None


def test_native_nonpaged_host_does_not_require_paged_slots():
    """SM120's nonpaged host omits page tables entirely."""
    s, facts, _ = _fixture()
    slots = {"block_table_ptr", "block_table_v_ptr", "table_strides", "n_pages"}
    kept = [(n, value) for n, value in zip(s.order, s.template) if n not in slots]
    s.order = [n for n, _ in kept]
    s.template = [value for _, value in kept]
    s.index = {n: i for i, n in enumerate(s.order)}
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    _equal(s, facts)


@pytest.mark.parametrize("separate_v_strides", [False, True])
def test_native_paged_table_stride_slots_match_the_compiled_host(separate_v_strides):
    s, facts, _ = _paged_fixture()
    if not separate_v_strides:
        kept = [(name, value) for name, value in zip(s.order, s.template) if name != "table_v_strides"]
        s.order = [name for name, _ in kept]
        s.template = [value for _, value in kept]
        s.index = {name: i for i, name in enumerate(s.order)}
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    _equal(s, facts)
    changed = dict(facts, block_table_v=facts["block_table_v"]._replace(strides=(8, 8, 2, 1), span=32))
    if separate_v_strides:
        frame = _equal(s, changed)
        assert frame[s.index["table_strides"]] == (4, 1)
        assert frame[s.index["table_v_strides"]] == (8, 2)
        short = dict(changed, block_table_v=changed["block_table_v"]._replace(span=30))
        for fn in (_native, _reference):
            with pytest.raises(ValueError, match="storage|spans"):
                fn(s, short)
    else:
        for fn in (_native, _reference):
            with pytest.raises(ValueError, match="matching K/V table strides"):
                fn(s, changed)
