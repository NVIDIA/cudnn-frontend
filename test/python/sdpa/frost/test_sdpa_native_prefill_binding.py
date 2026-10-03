# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native dense prefill preserves each host ABI and independent QK/V widths."""

import ast
from pathlib import Path

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_dense_binding import (
    _fixture,
    _pack,
    test_native_decode_standalone_rebinds_scale_and_capture as _standalone,
    test_native_dense_graph_fresh_bindings_and_changed_replay as _graph,
)

pytestmark = [pytest.mark.L0]
_WIDTHS = [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)]


def _prefill_fixture(dq, dv, paged, split, dtype):
    s, facts, frames = _fixture(dtype=dtype, paged=paged, sq=65)
    s.d_qk, s.d_v = dq, dv
    for role in ("q", "k", "v", "o"):
        f = facts[role]
        dim = dq if role in ("q", "k") else dv
        shape = (*f.shape[:-1], dim)
        strides = tuple(st // 128 * dim for st in f.strides[:-1]) + (1,)
        span = sum((n - 1) * st for n, st in zip(shape, strides)) + 1
        facts[role] = f._replace(shape=shape, strides=strides, span=span)
    stem = "prefill_d192_d128_f16" if dq == 192 else f"prefill_d{max(128, dq)}_f16"
    path = Path(prep.__file__).parent / "kernels/sm100" / (stem + ".py")
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_host")
    s.order = [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [None] * len(s.order)
    for name in ("seq_q_lens_addr", "lse_ext", "n_thd_units"):
        if name in s.index:
            s.template[s.index[name]] = 0
    combined = []
    if split > 1:
        s.split, s.fp32_partial = split, True
        s.expect["o"] = "float32"
        rows, h, sq = split * s.b, s.qh, s.s_q_max
        o_size, lse_size = rows * sq * h * dv, rows * h * sq
        s.combine = prep.SplitCombineSpec(
            lambda *args: combined.append(args),
            object(),
            prep.BufferFacts(0, "float32", (2, 0), o_size, (rows, h, sq, dv), (sq * h * dv, dv, h * dv, 1)),
            prep.BufferFacts(0, "float32", (2, 0), lse_size, (rows, h, sq), (h * sq, sq, 1)),
            o_size * 4,
            dtype,
            True,
        )
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    return s, facts, frames, combined


@pytest.mark.parametrize("dq,dv", _WIDTHS)
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_prefill_native_frames_match_actual_host_and_python(dq, dv, paged, split, dtype):
    # Paged metadata parity is useful for every ABI even where graph admission
    # does not expose that flavor. GPU cases below use admitted combinations.
    s, facts, frames, combined = _prefill_fixture(dq, dv, paged, split, dtype)
    for offset in (0, 0x100000):
        fresh = {role: f._replace(ptr=f.ptr + offset) for role, f in facts.items()}
        if split == 1:
            expected = prep.bind_dense(s, fresh, 17, 17)
            actual = s.native.bind(_pack(fresh), prep._NATIVE_DENSE_INDICES, 17)
            assert list(actual) == expected
        else:
            expected = prep.bind_dense_split(s, fresh, 0x400000, 17, 17)
            actual = s.native.bind_split(_pack(fresh), prep._NATIVE_DENSE_INDICES, 0x400000, 17)
            assert list(actual[0]) == expected[0]
            assert actual[1] == expected[1]
            assert actual[1][4][-1] == dv
        s.native.execute(_pack(fresh), prep._NATIVE_DENSE_INDICES, 17, workspace=0x400000 if split > 1 else 0)
    assert len(frames) == 2 and len(combined) == (2 if split > 1 else 0)


@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
@pytest.mark.parametrize("split", [1, 4])
def test_prefill_asymmetric_width_and_current_span_reject_before_launch(role, split):
    s, facts, frames, combined = _prefill_fixture(192, 128, True, split, "bfloat16")
    s.native.execute(_pack(facts), prep._NATIVE_DENSE_INDICES, 17, workspace=0x400000 if split > 1 else 0)
    frames.clear()
    combined.clear()
    f = facts[role]
    for value in (f._replace(shape=(*f.shape[:-1], 128 if role in ("q", "k") else 192)), f._replace(span=1)):
        bad = dict(facts, **{role: value})
        with pytest.raises(ValueError):
            s.native.execute(_pack(bad), prep._NATIVE_DENSE_INDICES, 17, workspace=0x400000 if split > 1 else 0)
        assert frames == combined == []


@pytest.mark.parametrize("dq,dv,paged", [(dq, dv, False) for dq, dv in _WIDTHS] + [(128, 128, True), (192, 128, True), (256, 256, True)])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_prefill_graph_fresh_storage_padding_overrides_and_replay(dq, dv, paged, split, dtype, monkeypatch, request):
    _graph(paged, dq, 65, dtype, monkeypatch, request, splits=split, d_v=dv, prefill=True)


@pytest.mark.parametrize("dq,dv", _WIDTHS)
@pytest.mark.parametrize("split", [1, 4])
def test_prefill_graph_physical_output_stride_above_int32(dq, dv, split, monkeypatch, request):
    _graph(False, dq, 65, "bfloat16", monkeypatch, request, splits=split, d_v=dv, prefill=True, wide_output=True)


@pytest.mark.parametrize("dq,dv", _WIDTHS)
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
def test_prefill_standalone_fresh_storage_scale_and_replay(dq, dv, split, stats, monkeypatch):
    _standalone(dq, 65, monkeypatch, splits=split, stats_mode=stats, d_v=dv, prefill=True)
