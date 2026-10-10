# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Opaque MXFP8 scale storage has one native contract for dense and THD."""

import sdpa_binding_reference as binding_reference

import ast
from pathlib import Path

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_fp8_binding import _fixture as _dense_fixture
from test_sdpa_native_thd_fp8_binding import _fixture as _thd_fixture

pytestmark = [pytest.mark.L1]


def _fixture(thd=False, d=128, dv=128, split=1, output="bfloat16", carrier="uint8"):
    if thd:
        s, facts, frames = _thd_fixture(d, dv, output=output)
        combined = []
    else:
        s, facts, frames, combined = _dense_fixture(d, dv, split, "float8_e4m3fn", output)
    if not thd and d == 512 and split > 1:
        # MX D512 writes half partials even on SM100; the final FP8 O is separate.
        partial = "bfloat16" if output == "bfloat16" else "float16"
        s.fp32_partial = False
        s.expect["o"] = partial
        s.elem_bytes["o"] = 2
        s.combine = s.combine._replace(o=s.combine.o._replace(dtype=partial), lse_offset=s.combine.o.numel * 2)
        s.quant = s.quant._replace(scratch_offset=s.combine.lse_offset + s.combine.lse.numel * 4)
    old = dict(zip(s.order, s.template))
    path = Path(prep.__file__).parent / "kernels/_mxfp8_host.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "host")
    s.order = [a.arg for a in host.args.args if "Constexpr" not in ast.unparse(a.annotation)]
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [old.get(name) for name in s.order]
    sizes = ((d + 127) // 128 * 512,) * 2 + ((dv + 127) // 128 * 512,)
    s.quant = s.quant._replace(sf_sizes=sizes)
    for role in prep._QUANT_ROLES[:-1]:
        facts.pop(role, None)
    for i, (role, h, size) in enumerate(zip(("sf_q", "sf_k", "sf_v"), (s.qh, s.kh, s.kh), sizes)):
        count = (2 if i == 0 else 4) if thd else ((s.s_q_max if i == 0 else s.s_k_max) + 127) // 128
        b = 1 if thd else s.b
        width = prep._buffers.DTYPE_ITEMSIZE[carrier]
        size //= width
        facts[role] = prep.BufferFacts(
            0x60000000 + i * 0x1000000, carrier, (2, 0), b * h * count * size, (b, h, count, size), (h * count * size, count * size, size, 1)
        )
    roles = (prep._NATIVE_THD_ROLES if thd else prep._NATIVE_DENSE_ROLES) + prep._native_quant_roles(s.quant)
    s.native = (cudnn._pybind_module._SdpaThdBinder if thd else cudnn._pybind_module._SdpaDenseBinder)(s)
    return s, facts, frames, combined, roles


def _execute(s, facts, roles, thd, workspace=0x50000000):
    pack = prep._native_pack_from_facts(facts, roles)
    indices = tuple(range(len(roles)))
    if thd:
        return s.native.execute(pack, indices, workspace, 17)
    return s.native.execute(pack, indices, 17, workspace=workspace)


@pytest.mark.parametrize("thd,split", [(False, 1), (False, 4), (True, 1)])
@pytest.mark.parametrize("d,dv", [pytest.param(128, 128, marks=pytest.mark.L0), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output", ["bfloat16", "float8_e4m3fn"])
@pytest.mark.parametrize("carrier", ["uint8", "int32"])
def test_mxfp8_native_actual_host_frames(thd, split, d, dv, output, carrier):
    s, facts, frames, combined, roles = _fixture(thd, d, dv, split, output, carrier)
    for offset in (0, 2**33):
        fresh = {role: f._replace(ptr=f.ptr + offset) if f is not None else None for role, f in facts.items()}
        native, s.native = s.native, None
        try:
            binding_reference.execute_quantized(s, fresh, 0x50000000 + offset, 17, 17)
        finally:
            s.native = native
        expected, tail = frames.pop(), combined.pop() if split > 1 else None
        _execute(s, fresh, roles, thd, 0x50000000 + offset)
        assert frames.pop() == expected
        if tail is not None:
            assert combined.pop() == tail


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("role", ["sf_q", "sf_k", "sf_v"])
@pytest.mark.parametrize("kind", ["missing", "device", "alignment", "null", "span", "hole", "overlap", "negative", "workspace", "amax"])
def test_mxfp8_current_scale_rejection_precedes_writes(thd, role, kind, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: writes.append(a))
    monkeypatch.setattr(prep._buffers, "memset_zero_async", lambda *a: writes.append(a))
    s, facts, frames, combined, roles = _fixture(thd)
    _execute(s, facts, roles, thd)
    frames.clear()
    f = facts[role]
    changed = dict(
        missing=None,
        device=f._replace(device=(1, 0)),
        alignment=f._replace(ptr=f.ptr + 1),
        null=f._replace(ptr=0),
        span=f._replace(span=f.span - 1),
        hole=f._replace(strides=(*f.strides[:-1], 2)),
        overlap=f._replace(strides=(f.strides[0], 0, *f.strides[2:])),
        negative=f._replace(strides=(*f.strides[:-1], -1)),
        workspace=f._replace(ptr=0x50000000),
        amax=f._replace(ptr=facts["amax_o"].ptr),
    )[kind]
    with pytest.raises(ValueError):
        _execute(s, dict(facts, **{role: changed}), roles, thd)
    assert frames == combined == writes == []


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("kind", ["int64_product", "int64_address", "tile_count"])
def test_mxfp8_wide_geometry_rejects_without_launch(thd, kind):
    s, facts, frames, combined, roles = _fixture(thd)
    f = facts["sf_q"]
    if kind == "int64_product":
        f = f._replace(shape=(2**62, 8), strides=(8, 1), span=-1)
    elif kind == "int64_address":
        f = f._replace(ptr=2**63 - 16)
    else:
        size = s.qh * s.quant.sf_sizes[0]
        f = f._replace(shape=(2**31, size), strides=(size, 1), span=-1)
    with pytest.raises(ValueError):
        _execute(s, dict(facts, sf_q=f), roles, thd)
    assert frames == combined == []


@pytest.mark.parametrize("thd,split", [(False, 1), (False, 4), (True, 1)])
@pytest.mark.parametrize("d,dv", [pytest.param(128, 128, marks=pytest.mark.L0), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_mxfp8_native_graph_rebind_and_capture(thd, split, d, dv, output, monkeypatch):
    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_mxfp8 import _case, _check, _change_scales

    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)) or not _dsl_installed():
        pytest.skip("requires an existing MXFP8 architecture")
    if cc == (10, 7) and (thd or split > 1):
        pytest.skip("SM107 MXFP8 THD and split are not existing graph rows")
    arch = "sm107" if cc == (10, 7) else "sm100"
    g, vp, ws, bufs, ts = _case(thd=thd, split_kv=split, d=d, dv=dv, output_dtype=output, arch=arch, explicit_plan=True)
    assert g._compiled_plans[g._plan_index]._prepared.spec.native is not None
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native MXFP8 rebuilt Python facts"))
    monkeypatch.setattr(prep, "execute_quantized", lambda *a, **k: pytest.fail("native MXFP8 entered Python binding"), raising=False)
    g.execute(vp, ws)
    _check(bufs, thd=thd)
    for name in ("q", "k", "v", "o", "sf_q", "sf_k", "sf_v", "lse", "amax_o"):
        bufs[name] = bufs[name].clone()
        vp[ts[name]] = bufs[name]
    ws = torch.empty_like(ws)
    g.execute(vp, ws)
    _check(bufs, thd=thd)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        _change_scales(bufs)
        graph.replay()
        torch.cuda.synchronize()
        _check(bufs, thd=thd)
    finally:
        graph.reset()


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_mxfp8_native_paged_scales_and_replay(d, split, output, monkeypatch):
    from frost_test_utils import _dsl_installed
    from test_sdpa_fwd_paged_sm100 import _build_mxfp8, _ref_mxfp8, _check_o
    from cudnn.engines import MANIFEST
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, engine_name

    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)) or not _dsl_installed():
        pytest.skip("requires an existing paged MXFP8 graph row")
    rubin = cc == (10, 7)
    if rubin and split > 1:
        pytest.skip("the cc 10.7 MXFP8 row serves paged pools unsplit")
    g, vp, out, stats, amax, reference = _build_mxfp8(
        2,
        4,
        2,
        128,
        4,
        [512, 512],
        False,
        s_q=128,
        stats=True,
        separate_v=True,
        batch_inner=True,
        d_qk=d,
        d_v=d,
        out_dt=output,
    )
    g.validate()
    g.build_operation_graph()
    # The cc 10.7 row serves the same pool contract on d128 (cga2) / d256 (cga1): separate V table, batch-innermost
    # tables, e4m3 O, Stats, CUDA-graph replay after mutating the SF bytes, native-binder-only execution.
    name = engine_name(arch="sm107" if rubin else "sm100", mxfp8=True)
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == name)
    cga = max(dict(caps.cgas_by_d_shape).get((d, d), caps.cgas))
    knobs = SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, split_kv=split, sched_policy=0, pack_gqa=False)
    g.create_execution_plan(family.offered_ids()[name], knobs)
    g.check_support()
    g.build_plans()
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None and spec.paged
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    output_ports = [next(t for t, x in vp.items() if x is target) for target in (out, stats, amax)]
    scale_ports = [t for t in vp if t.get_data_type() == cudnn.data_type.FP8_E8M0]
    assert len(scale_ports) == 3
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native paged MXFP8 rebuilt Python facts"))
    monkeypatch.setattr(prep, "execute_quantized", lambda *a, **k: pytest.fail("native paged MXFP8 entered Python binding"), raising=False)

    def check():
        expected, lse = _ref_mxfp8(**reference)
        out, stats, amax = (vp[t] for t in output_ports)
        atol = _check_o(out.float(), expected, output, "e4m3")
        torch.testing.assert_close(stats.view(2, 4, 128), lse, atol=atol, rtol=0.03)
        torch.testing.assert_close(amax.flatten()[0], expected.abs().amax(), atol=atol, rtol=0.03)

    g.execute(vp, ws)
    check()
    vp = {t: x.clone() for t, x in vp.items()}
    ws = torch.empty_like(ws)
    g.execute(vp, ws)
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        for t in scale_ports:
            vp[t].add_(-1)
        for role in ("qd", "kd_pool", "vd_pool"):
            reference[role] = reference[role] * 0.5
        for t in output_ports:
            vp[t].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()
    finally:
        graph.reset()
