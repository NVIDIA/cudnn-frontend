# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM90 native frames preserve natural scales, layouts and fixed THD metadata."""

import ast
import itertools
from pathlib import Path

import cudnn
import pytest
import torch

from cudnn.sdpa.fwd import prepared as prep
from frost_test_utils import requires_dsl
from test_sdpa_native_dense_binding import _fixture as _dense_fixture, _pack
from test_sdpa_native_thd_binding import _fixture as _thd_fixture

pytestmark = [pytest.mark.L0]


def _host_order():
    path = Path(prep.__file__).parent / "kernels/sm90/prepared_host.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "host")
    return [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]


def _spec(thd, dtype="bfloat16", layout="NH"):
    s, facts, frames = _thd_fixture(dtype, layout) if thd else _dense_fixture(dtype, sq=65)
    values = dict(zip(s.order, s.template))
    s.order = _host_order()
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [values.get(name) for name in s.order]
    s.template[s.index["scale_softmax"]] = 0.125
    if thd:
        s.fixed_batch, s.workspace_alignment, s.cga_tile_m = True, 128, 64
        s.native = cudnn._pybind_module._SdpaThdBinder(s)
    else:
        s.dense_flex = s.shape_fixed = s.lpt_grid_fixed = s.kv_tail_native = True
        s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    return s, facts, frames


def _bind(s, facts, thd, workspace=0x30000, stream=17):
    if thd:
        return s.native.bind(prep._native_pack_from_facts(facts), prep._NATIVE_THD_INDICES, workspace, stream)
    return s.native.bind(_pack(facts), prep._NATIVE_DENSE_INDICES, stream)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("permutation", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("wide", [False, True])
def test_dense_permutation_frames_match_python(dtype, permutation, wide):
    s, facts, frames = _spec(False, dtype)
    for role in ("q", "k", "v", "o"):
        f = facts[role]
        strides, pitch = [0, 0, 0, 1], f.shape[3]
        for axis in permutation:
            strides[axis] = pitch
            pitch *= f.shape[axis]
        if wide:
            strides[permutation[-1]] += 2**32
        span = 1 + sum((n - 1) * st for n, st in zip(f.shape, strides))
        facts[role] = f._replace(strides=tuple(strides), span=span)
    for offset in (0, 0x100000):
        fresh = {role: f._replace(ptr=f.ptr + offset) for role, f in facts.items()}
        assert list(_bind(s, fresh, False)) == prep.bind_dense(s, fresh, 17, 17)
        s.native.execute(_pack(fresh), prep._NATIVE_DENSE_INDICES, 17, -0.25)
        assert frames[-1][s.index["scale_softmax"]] == -0.25


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("layout", [None, "NH", "HN"])
@pytest.mark.parametrize("lens_form", [0, 1, 2, 3])
def test_thd_frames_preserve_lengths_and_natural_scale(dtype, layout, lens_form):
    s, facts, frames = _spec(True, dtype, layout)
    s.lens_form = lens_form
    s.template[s.index["thd_lens_form"]] = lens_form
    for bit, role in enumerate(("q_lens", "kv_lens")):
        n = s.b + ((lens_form >> bit) & 1)
        facts[role] = facts[role]._replace(shape=(n,), span=n)
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    for scale, offset in ((0.0, 0), (-0.5, 0x100000), (0.25, 0x200000)):
        fresh = {role: f._replace(ptr=f.ptr + offset) for role, f in facts.items()}
        actual = _bind(s, fresh, True, stream=23)
        assert list(actual) == prep._bind_thd_python(s, fresh, 0x30000, 23, 23)
        s.native.execute(prep._native_pack_from_facts(fresh), prep._NATIVE_THD_INDICES, 0x30000, 23, scale)
        assert frames[-1][s.index["scale_softmax"]] == scale


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("role", ["q", "k", "v", "o", "lse"])
@pytest.mark.parametrize("bad", ["device", "dtype", "pointer", "span"])
def test_current_storage_still_rejects_after_warmup(thd, role, bad):
    s, facts, frames = _spec(thd)
    _bind(s, facts, thd)
    f = facts[role]
    update = {"device": {"device": (2, 1)}, "dtype": {"dtype": "int32"}, "pointer": {"ptr": f.ptr + 1}, "span": {"span": -1 if thd else 1}}[bad]
    changed = dict(facts, **{role: f._replace(**update)})
    with pytest.raises(ValueError):
        _bind(s, changed, thd)
    assert not frames


def test_thd_fixed_batch_and_tensor_map_alignment():
    s, facts, frames = _spec(True)
    _bind(s, facts, True)
    shorter = dict(facts)
    for role in ("q_lens", "kv_lens"):
        shorter[role] = facts[role]._replace(shape=(4,), span=4)
    with pytest.raises(ValueError, match="exactly 4 sequences"):
        _bind(s, shorter, True)
    for ptr in (0, 0x30010, 0x30040):
        with pytest.raises(ValueError, match="non-null|128-byte"):
            _bind(s, facts, True, workspace=ptr)
    assert not frames


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("slot", ["scale_softmax", "q_strides", "stream"])
def test_missing_actual_host_slot_rejects(thd, slot):
    s, _, _ = _spec(thd)
    index = s.order.index(slot)
    s.order.pop(index)
    s.template.pop(index)
    binder = cudnn._pybind_module._SdpaThdBinder if thd else cudnn._pybind_module._SdpaDenseBinder
    with pytest.raises(ValueError, match="no argument"):
        binder(s)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires SM90")
@requires_dsl
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125])
@pytest.mark.parametrize("dq,dv", [(96, 128), (512, 512)])
def test_standalone_native_route_fresh_storage_scale_and_replay(thd, scale, dq, dv, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    from test_sdpa_fwd_dsl_sm100 import _bhsd, _ref_sdpa_full

    b, h, sq, skv = 2, 2, 33, 65
    q, k, v, o = (_bhsd(b, h, s, d, torch.bfloat16) for s, d in ((sq, dq), (skv, dq), (skv, dv), (sq, dv)))
    stats = torch.empty((b, sq, h), device="cuda").transpose(1, 2)
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=stats, thd=thd, scale_softmax=scale)
    api.compile()
    spec = api._thd_spec if thd else api._dense_spec
    assert spec.native is not None
    kwargs = {}
    if thd:
        kwargs = dict(
            seq_q_lens=torch.full((b,), sq, dtype=torch.int32, device="cuda"),
            seq_kv_lens=torch.full((b,), skv, dtype=torch.int32, device="cuda"),
            workspace=torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda"),
        )
    run = lambda: api.execute(q, k, v, o, lse_tensor=stats, scale_softmax=scale * 2, **kwargs)
    run()

    def forbidden(*args, **kwargs):
        raise AssertionError("native SM90 returned to Python operand binding")

    with monkeypatch.context() as guard:
        guard.setattr(prep, "facts_of_tensor", forbidden)
        guard.setattr(prep, "bind_dense", forbidden)
        guard.setattr(prep, "_bind_thd_python", forbidden)
        q, k, v = (torch.randn_like(t) for t in (q, k, v))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            run()
            with torch.cuda.graph(captured, stream=stream):
                run()
        stream.synchronize()
        q.add_(0.25)
        k.mul_(0.5)
        v.add_(0.125)
        captured.replay()
        torch.cuda.synchronize()
    expected_o, expected_lse = _ref_sdpa_full(q, k, v, scale=scale * 2, return_stats=True)
    torch.testing.assert_close(o, expected_o, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(stats, expected_lse, atol=2e-2, rtol=2e-2)
    assert spec.native is not None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires SM90")
@requires_dsl
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_graph_native_route_without_python_facts(monkeypatch, thd):
    import test_sdpa_fwd_dsl_sm100 as graph_tests

    def forbidden(*args, **kwargs):
        raise AssertionError("native graph launch reconstructed Python facts or bindings")

    monkeypatch.setattr(prep, "facts_of_roles", forbidden)
    monkeypatch.setattr(prep, "bind_dense", forbidden)
    monkeypatch.setattr(prep, "_bind_thd_python", forbidden)
    monkeypatch.setattr(graph_tests, "_ARCH", "sm90")
    if thd:
        graph_tests._run_thd_stats_case(seq_lens_q=[33, 0, 7], seq_lens_kv=[65, 5, 7], d=512, H_q=4, H_kv=2, mask="causal_br", cu_lens=True)
    else:
        graph_tests.test_sdpa_fwd_dsl_sm100_graph_api(torch.bfloat16, True, 512)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires SM90")
@requires_dsl
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("thd,role", [(False, role) for role in ("q", "k", "v", "o", "stats")] + [(True, role) for role in ("q", "k", "v", "o")])
@pytest.mark.parametrize("product", [False, True], ids=["wide-stride", "wide-product"])
def test_physical_wide_strides_and_products(thd, role, product):
    import ctypes
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    from test_sdpa_fwd_dsl_sm100 import _bhsd, _ref_sdpa_full

    count, h, d = (5 if product else 2), 2, 128
    b, sq, skv = (1, count, count) if thd else (count, 2, 3)
    tensors = {name: _bhsd(b, h, seq, d, torch.bfloat16) for name, seq in (("q", sq), ("k", skv), ("v", skv), ("o", sq))}
    tensors["stats"] = torch.empty((b, sq, h), device="cuda").transpose(1, 2)
    expected_o, expected_stats = _ref_sdpa_full(tensors["q"], tensors["k"], tensors["v"], scale=d**-0.5, return_stats=True)
    source = tensors[role]
    axis = 2 if thd else 0
    strides = list(source.stride())
    strides[axis] += 2**30 if product else 2**32
    origin = 2**31 if product else 0
    span = 1 + sum((n - 1) * st for n, st in zip(source.shape, strides))
    torch.cuda.empty_cache()
    if (origin + span) * source.element_size() + 1024**3 > torch.cuda.mem_get_info()[0]:
        pytest.skip("physical Int64 probe requires room for guarded wide storage")
    try:
        backing = torch.empty(origin + span, device="cuda", dtype=source.dtype)
    except torch.OutOfMemoryError:
        pytest.skip("another allocation consumed the guarded probe capacity")
    shape = list(source.shape)
    shape[axis] = 1
    guards = []
    for index in range(1, source.shape[axis]):
        offset = index * strides[axis]
        narrowed = ctypes.c_int32(offset).value
        if offset != narrowed:
            guard = backing.as_strided(shape, source.stride(), origin + narrowed)
            guard.fill_(float("nan"))
            guards.append(guard)
    wide = backing.as_strided(source.shape, strides, origin)
    wide.copy_(source) if role not in ("o", "stats") else wide.fill_(float("nan"))
    tensors[role] = wide
    api = SdpaFwdDslSm90(*(tensors[name] for name in ("q", "k", "v", "o")), sample_lse=tensors["stats"], thd=thd)
    api.compile()
    assert (api._thd_spec if thd else api._dense_spec).native is not None
    kwargs = {}
    if thd:
        kwargs = dict(
            seq_q_lens=torch.tensor([sq], device="cuda", dtype=torch.int32),
            seq_kv_lens=torch.tensor([skv], device="cuda", dtype=torch.int32),
            workspace=torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8),
        )
    api.execute(*(tensors[name] for name in ("q", "k", "v", "o")), lse_tensor=tensors["stats"], **kwargs)
    torch.testing.assert_close(tensors["o"], expected_o, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(tensors["stats"], expected_stats, atol=2e-2, rtol=2e-2)
    assert guards and all(torch.isnan(guard).all().item() for guard in guards)
