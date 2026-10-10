# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native binding preserves the SM103 and SM107 half attention host contracts."""

import sdpa_binding_reference as binding_reference

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_dense_binding import (
    _pack,
    test_native_decode_standalone_rebinds_scale_and_capture as _standalone,
    test_native_dense_graph_fresh_bindings_and_changed_replay as _graph,
)
from test_sdpa_native_prefill_binding import _prefill_fixture

pytestmark = [pytest.mark.L1]
_WIDTHS = [(128, 128), (192, 128), (256, 256), (512, 512)]
_ARCH_WIDTHS = [(cc, dq, dv) for cc in ((10, 3), (10, 7)) for dq, dv in _WIDTHS] + [((10, 3), 64, 64)]


@pytest.mark.parametrize("dq,dv", _WIDTHS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sm107_native_matches_actual_host_abi_and_rebinds(dq, dv, dtype):
    s, facts, frames, _ = _prefill_fixture(dq, dv, False, 1, dtype, arch="sm107")
    for offset in (0, 0x100000):
        fresh = {role: f._replace(ptr=f.ptr + offset) for role, f in facts.items()}
        expected = binding_reference.bind_dense(s, fresh, 17, 17)
        actual = s.native.bind(_pack(fresh), prep._NATIVE_DENSE_INDICES, 17)
        assert list(actual) == expected
        s.native.execute(_pack(fresh), prep._NATIVE_DENSE_INDICES, 17)
    assert len(frames) == 2


@pytest.mark.parametrize("slot", ["q_ptr", "lse_ptr", "seq_q_lens_addr", "problem_size", "stream"])
def test_missing_required_host_argument_still_rejects(slot):
    s, _, _, _ = _prefill_fixture(256, 256, False, 1, "bfloat16", arch="sm107")
    pos = s.order.index(slot)
    s.order.pop(pos)
    s.template.pop(pos)
    with pytest.raises(ValueError, match="no argument " + slot):
        cudnn._pybind_module._SdpaDenseBinder(s)


@pytest.mark.L0
def test_sm107_dense_only_host_cannot_bind_paged_plan():
    s, _, _, _ = _prefill_fixture(256, 256, False, 1, "bfloat16", arch="sm107")
    s.paged = True
    with pytest.raises(ValueError, match="no argument block_table_ptr"):
        cudnn._pybind_module._SdpaDenseBinder(s)


@pytest.mark.parametrize(
    "cc,dq,dv,sq,dtype",
    [
        pytest.param(cc, dq, dv, sq, dtype, marks=pytest.mark.L0 if (dq, dv, sq, dtype) == (128, 128, 65, "bfloat16") else ())
        for cc, dq, dv in _ARCH_WIDTHS
        for sq in (1, 65)
        for dtype in ("bfloat16", "float16")
    ],
)
def test_arch_native_graph_fresh_storage_overrides_and_replay(cc, dq, dv, sq, dtype, monkeypatch, request):
    _graph(False, dq, sq, dtype, monkeypatch, request, d_v=dv, prefill=True, cc=cc)


@pytest.mark.parametrize("cc,dq,dv", _ARCH_WIDTHS)
def test_arch_native_physical_output_stride_above_int32(cc, dq, dv, monkeypatch, request):
    _graph(False, dq, 65, "bfloat16", monkeypatch, request, d_v=dv, prefill=True, wide_output=True, cc=cc)


@pytest.mark.parametrize("cc,dq,dv", _ARCH_WIDTHS)
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
@pytest.mark.parametrize("sink", [False, True])
def test_arch_native_standalone_stats_sinks_and_replay(cc, dq, dv, stats, sink, monkeypatch):
    _standalone(dq, 65, monkeypatch, d_v=dv, prefill=True, stats_mode=stats, has_sink=sink, cc=cc)


@pytest.mark.parametrize("cc,dq,dv", _ARCH_WIDTHS)
def test_arch_native_graph_sinks(cc, dq, dv, monkeypatch, request):
    _graph(False, dq, 65, "bfloat16", monkeypatch, request, d_v=dv, prefill=True, has_sink=True, cc=cc)


@pytest.mark.parametrize("dq,dv", [(64, 64), *_WIDTHS])
@pytest.mark.parametrize("prefill", [False, True])
def test_sm103_native_split_dense(dq, dv, prefill, monkeypatch, request):
    # D192 and D512 have no separate decode template.
    prefill = prefill or dq in (192, 512)
    _graph(False, dq, 65 if prefill else 1, "bfloat16", monkeypatch, request, splits=4, d_v=dv, prefill=prefill, cc=(10, 3))


@pytest.mark.parametrize("dq,dv", [(128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("hnd", [False, True])
def test_sm103_native_paged_prefill(dq, dv, split, hnd, monkeypatch, request):
    _graph(True, dq, 65, "bfloat16", monkeypatch, request, splits=split, d_v=dv, prefill=True, hnd=hnd, cc=(10, 3))


@pytest.mark.parametrize("d", [64, 128, 256])
def test_sm103_native_decode_physical_wide_tables(d, monkeypatch, request):
    _graph(True, d, 1, "bfloat16", monkeypatch, request, splits=4, wide_tables=True, cc=(10, 3))


@pytest.mark.parametrize("dq,dv", [(128, 128), (192, 128)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sm107_native_split_matches_new_host_abi(dq, dv, dtype):
    s, facts, frames, combined = _prefill_fixture(dq, dv, False, 4, dtype, arch="sm107")
    for workspace in (0x400000, 0x800000):
        expected = binding_reference.bind_dense_split(s, facts, workspace, 17, 17)
        actual = s.native.bind_split(_pack(facts), prep._NATIVE_DENSE_INDICES, workspace, 17)
        assert list(actual[0]) == expected[0] and actual[1] == expected[1]
        s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, workspace=workspace)
    assert len(frames) == len(combined) == 2


@pytest.mark.parametrize(
    "dq,dv,dtype",
    [
        pytest.param(dq, dv, dtype, marks=pytest.mark.L0 if (dq, dv, dtype) == (128, 128, "bfloat16") else ())
        for dq, dv in ((64, 64), (128, 128), (192, 128))
        for dtype in ("float16", "bfloat16")
    ],
)
def test_sm107_native_split_graph_rebinds_and_replays(dq, dv, dtype, monkeypatch, request):
    _graph(False, dq, 65, dtype, monkeypatch, request, splits=4, d_v=dv, prefill=True, cc=(10, 7))


@pytest.mark.parametrize("dq,dv", [(64, 64), (128, 128), (192, 128)])
def test_sm107_native_split_physical_output_stride_above_int32(dq, dv, monkeypatch, request):
    _graph(False, dq, 65, "bfloat16", monkeypatch, request, splits=4, d_v=dv, prefill=True, wide_output=True, cc=(10, 7))
