# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared binding preserves SM120 envelope widths and half split partials."""

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_dense_binding import (
    _pack,
    test_native_decode_standalone_rebinds_scale_and_capture as _standalone,
    test_native_dense_graph_fresh_bindings_and_changed_replay as _graph,
)
from test_sdpa_native_prefill_binding import _prefill_fixture

pytestmark = [pytest.mark.L0]
_WIDTHS = [(8, 16), (48, 80), (128, 128), (192, 128), (248, 256), (256, 256), (264, 384), (512, 512)]


@pytest.fixture
def sm120_cc():
    import torch

    cc = torch.cuda.get_device_capability()
    if cc not in ((12, 0), (12, 1)):
        pytest.skip("native SM120/SM121 GPU validation")
    return cc


@pytest.mark.parametrize("dq,dv", _WIDTHS)
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sm120_actual_host_frames_and_half_partial_offsets(dq, dv, split, dtype):
    s, facts, frames, combined = _prefill_fixture(dq, dv, False, split, dtype, arch="sm120")
    for offset in (0, 0x100000):
        fresh = {role: f._replace(ptr=f.ptr + offset) for role, f in facts.items()}
        if split == 1:
            expected = prep.bind_dense(s, fresh, 17, 17)
            actual = s.native.bind(_pack(fresh), prep._NATIVE_DENSE_INDICES, 17)
            assert list(actual) == expected
        else:
            workspace = 0x400000 + offset
            expected = prep.bind_dense_split(s, fresh, workspace, 17, 17)
            actual = s.native.bind_split(_pack(fresh), prep._NATIVE_DENSE_INDICES, workspace, 17)
            assert list(actual[0]) == expected[0]
            assert actual[1] == expected[1]
            assert s.combine.lse_offset == split * s.b * s.qh * s.s_q_max * dv * 2
            assert actual[1][1] == workspace + s.combine.lse_offset
        s.native.execute(_pack(fresh), prep._NATIVE_DENSE_INDICES, 17, workspace=0x400000 + offset if split > 1 else 0)
    assert len(frames) == 2 and len(combined) == (2 if split > 1 else 0)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("bad", ["dtype", "offset", "overflow", "missing_fp32_slot"])
def test_sm120_bad_partial_contract_rejects(dtype, bad):
    s, _, _, _ = _prefill_fixture(248, 256, False, 4, dtype, arch="sm120")
    if bad == "dtype":
        s.expect["o"] = "float32"
    elif bad == "offset":
        s.combine = s.combine._replace(lse_offset=s.combine.lse_offset - 16)
    elif bad == "overflow":
        s.b = 2**62
    else:
        s.fp32_partial = True
        s.expect["o"] = "float32"
        s.combine = s.combine._replace(lse_offset=s.combine.lse_offset * 2)
    with pytest.raises(ValueError):
        cudnn._pybind_module._SdpaDenseBinder(s)


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "lse"])
@pytest.mark.parametrize("split", [1, 4])
def test_sm120_revalidates_current_storage_after_geometry_hit(role, split):
    s, facts, frames, combined = _prefill_fixture(48, 80, False, split, "bfloat16", arch="sm120")
    workspace = 0x400000 if split > 1 else 0
    s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, workspace=workspace)
    frames.clear()
    combined.clear()
    f = facts[role]
    for changed in (f._replace(span=1), f._replace(ptr=f.ptr + 1), f._replace(device=(2, 1)), f._replace(dtype="int32")):
        with pytest.raises(ValueError):
            s.native.execute(_pack(dict(facts, **{role: changed})), prep._NATIVE_DENSE_INDICES, 17, workspace=workspace)
        assert frames == combined == []


@pytest.mark.parametrize("dq,dv", _WIDTHS)
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sm120_native_graph_envelopes_fresh_storage_and_replay(dq, dv, split, dtype, monkeypatch, request, sm120_cc):
    _graph(False, dq, 65, dtype, monkeypatch, request, splits=split, d_v=dv, prefill=True, cc=sm120_cc)


@pytest.mark.parametrize("dq,dv", [(48, 80), (248, 256), (264, 384)])
@pytest.mark.parametrize("split", [1, 4])
def test_sm120_native_physical_output_stride_above_int32(dq, dv, split, monkeypatch, request, sm120_cc):
    _graph(False, dq, 65, "bfloat16", monkeypatch, request, splits=split, d_v=dv, prefill=True, wide_output=True, cc=sm120_cc)


@pytest.mark.parametrize("dq,dv", _WIDTHS)
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
def test_sm120_native_standalone_envelopes_stats_scale_and_replay(dq, dv, split, stats, monkeypatch, sm120_cc):
    _standalone(dq, 65, monkeypatch, splits=split, stats_mode=stats, d_v=dv, prefill=True, cc=sm120_cc)


@pytest.mark.parametrize("dq,dv", [(48, 80), (248, 256), (264, 384)])
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
def test_sm120_native_standalone_sinks(dq, dv, stats, monkeypatch, sm120_cc):
    _standalone(dq, 65, monkeypatch, stats_mode=stats, d_v=dv, prefill=True, has_sink=True, cc=sm120_cc)


@pytest.mark.parametrize("dq,dv", [(48, 80), (248, 256), (264, 384)])
def test_sm120_native_graph_sinks(dq, dv, monkeypatch, request, sm120_cc):
    _graph(False, dq, 65, "bfloat16", monkeypatch, request, d_v=dv, prefill=True, has_sink=True, cc=sm120_cc)
