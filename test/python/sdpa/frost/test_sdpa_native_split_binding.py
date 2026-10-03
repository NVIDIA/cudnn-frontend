# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native split binding preserves both launch frames and final storage contracts."""

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_dense_binding import _fixture, _pack

pytestmark = [pytest.mark.L0]


def _split_fixture(dtype="bfloat16", paged=False, hnd=False, lse=True, lengths=True, d=128, sq=1, splits=4):
    s, facts, frames = _fixture(dtype, paged, hnd, lse, lengths, d, sq)
    s.split, s.fp32_partial, s.has_lse = splits, True, True
    s.expect["o"] = "float32"
    rows, h = splits * s.b, s.qh
    o_size, lse_size = rows * h * sq * d, rows * h * sq
    combined = []
    s.combine = prep.SplitCombineSpec(
        lambda *args: combined.append(args),
        object(),
        prep.BufferFacts(0, "float32", (2, 0), o_size, (rows, h, sq, d), (sq * h * d, d, h * d, 1)),
        prep.BufferFacts(0, "float32", (2, 0), lse_size, (rows, h, sq), (h * sq, sq, 1)),
        o_size * 4,
        dtype,
        lse,
    )
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    return s, facts, frames, combined


def _equal(s, facts, workspace=0x100000, stream=17):
    expected = prep.bind_dense_split(s, facts, workspace, stream, stream)
    actual = s.native.bind_split(_pack(facts), prep._NATIVE_DENSE_INDICES, workspace, stream)
    assert list(actual[0]) == expected[0]
    assert actual[1] == expected[1]
    return actual


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("paged,hnd", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("lse,lengths", [(False, False), (True, True)])
@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 1), (256, 4)])
@pytest.mark.parametrize("splits", [2, 8])
def test_native_split_frames_match_python_and_rebind(dtype, paged, hnd, lse, lengths, d, sq, splits):
    s, facts, frames, combined = _split_fixture(dtype, paged, hnd, lse, lengths, d, sq, splits)
    first = _equal(s, facts)
    fresh = {role: f._replace(ptr=f.ptr + 0x200000) for role, f in facts.items()}
    second = _equal(s, fresh, workspace=0x400000, stream=29)
    assert first[1][0] == 0x100000 and second[1][0] == 0x400000
    s.native.execute(_pack(fresh), prep._NATIVE_DENSE_INDICES, 29, workspace=0x400000)
    assert frames == [second[0]] and combined == [second[1]]


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "lse", "seq_q_lens", "seq_kv_lens", "block_table", "block_table_v"])
@pytest.mark.parametrize("change", ["short", "misaligned", "wrong_dtype", "wrong_device", "missing"])
def test_native_split_rejects_before_either_launch_after_warmup(role, change):
    s, facts, frames, combined = _split_fixture(paged=True)
    _equal(s, facts)
    f = facts[role]
    updates = {
        "short": {"span": 0},
        "misaligned": {"ptr": f.ptr + 1},
        "wrong_dtype": {"dtype": "float32" if f.dtype != "float32" else "bfloat16"},
        "wrong_device": {"device": (2, 1)},
    }
    changed = dict(facts)
    if change == "missing":
        del changed[role]
    else:
        changed[role] = f._replace(**updates[change])
    with pytest.raises(ValueError):
        prep.bind_dense_split(s, changed, 0x100000, 17, 17)
    with pytest.raises(ValueError):
        s.native.execute(_pack(changed), prep._NATIVE_DENSE_INDICES, 17, workspace=0x100000)
    assert frames == combined == []


def test_native_split_strided_outputs_wide_tables_and_fixed_query_geometry():
    s, facts, frames, combined = _split_fixture(paged=True, d=256, sq=4)
    # The scalar combine accepts a half-aligned base and covering D-contiguous
    # final output even when its head stride is not TMA aligned.
    os = (2**33, 257, 8 * 257, 1)
    need = sum((n - 1) * st for n, st in zip(facts["o"].shape, os)) + 1
    facts["o"] = facts["o"]._replace(ptr=0x4002, strides=os, span=need)
    for role in ("block_table", "block_table_v"):
        f = facts[role]
        facts[role] = f._replace(strides=(2**33, 3), span=(s.b - 1) * 2**33 + 22)
    _equal(s, facts)
    for role in ("q", "o"):
        f = facts[role]
        changed = dict(facts, **{role: f._replace(shape=(s.b - 1, *f.shape[1:]))})
        with pytest.raises(ValueError):
            s.native.execute(_pack(changed), prep._NATIVE_DENSE_INDICES, 17, workspace=0x100000)
    for workspace in (0, 0x100001, -16, 2**63 - 16):
        with pytest.raises(ValueError):
            s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, workspace=workspace)
    assert frames == combined == []


def test_native_split_scale_override_is_call_local():
    s, facts, frames, combined = _split_fixture()
    original = tuple(s.template)
    for scale in (0.1, 0.25):
        s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, scale, 0x100000)
        assert frames[-1][s.index["scale_softmax_log2"]] == scale
    assert tuple(s.template) == original
    assert len(frames) == len(combined) == 2


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 4)])
def test_native_split_graph_rebinds_workspace_and_replays(paged, dtype, d, sq, monkeypatch, request):
    from test_sdpa_native_dense_binding import test_native_dense_graph_fresh_bindings_and_changed_replay

    test_native_dense_graph_fresh_bindings_and_changed_replay(paged, d, sq, dtype, monkeypatch, request, splits=4)


@pytest.mark.parametrize("d", [64, 256])
def test_native_split_wide_live_tables(d, monkeypatch, request):
    from test_sdpa_native_dense_binding import test_native_dense_graph_fresh_bindings_and_changed_replay

    test_native_dense_graph_fresh_bindings_and_changed_replay(True, d, 1, "bfloat16", monkeypatch, request, wide_tables=True, splits=4)


@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 4)])
@pytest.mark.parametrize("stats_mode", ["none", "ln", "log2"])
def test_native_split_standalone_workspace_scale_and_stats(d, sq, stats_mode, monkeypatch):
    from test_sdpa_native_dense_binding import test_native_decode_standalone_rebinds_scale_and_capture

    test_native_decode_standalone_rebinds_scale_and_capture(d, sq, monkeypatch, splits=4, stats_mode=stats_mode)
