# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native block-output binding preserves packed storage and complete SF atoms."""

import sdpa_binding_reference as binding_reference

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_fp8_binding import _fixture as _fp8_fixture
from test_sdpa_native_mxfp8_binding import _fixture as _mx_fixture

pytestmark = [pytest.mark.L1]
_CASES = [(16, False, True), (32, False, True), (16, True, True), (32, True, True), (32, True, False)]


def _fixture(block=16, mx=False, has_scale=True, d=128, arch="sm100"):
    if mx:
        s, facts, frames, combined, _ = _mx_fixture(d=d, output="float8_e4m3fn")
    else:
        s, facts, frames, combined = _fp8_fixture(d, 128, 1, "float8_e4m3fn", "float8_e4m3fn", arch=arch)
    pack = 2 if block == 16 else 1
    rows, cols = (s.s_q_max + 127) // 128 * 128, s.d_v // block
    plane = rows * cols
    s.quant = s.quant._replace(block_output=prep.BlockOutputSpec((plane, 0, 0, cols), s.b * s.qh * plane, pack, has_scale))
    if mx and has_scale:
        facts["scale_o"] = prep.BufferFacts(0x41000000, "float32", (2, 0), 1, (1,), (1,))
    f = facts["o"]
    if pack == 2:
        s.expect["o"], s.elem_bytes["o"] = "uint8", 1
        facts["o"] = f._replace(dtype="uint8", shape=(*f.shape[:-1], f.shape[-1] // 2), strides=(*(st // 2 for st in f.strides[:-1]), 1), span=f.span // 2)
    facts["sf_o"] = prep.BufferFacts(0x70000000, "uint8", (2, 0), s.b * s.qh * plane, (s.b, s.qh, rows, cols), (s.qh * plane, plane, cols, 1))
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    roles = prep._NATIVE_DENSE_ROLES + prep._native_quant_roles(s.quant)
    return s, facts, frames, roles


def _execute(s, facts, roles, workspace=0x50000000, stream=17):
    return s.native.execute(prep._native_pack_from_facts(facts, roles), tuple(range(len(roles))), stream, workspace=workspace)


@pytest.mark.L0
@pytest.mark.parametrize("block,mx,has_scale", _CASES)
@pytest.mark.parametrize("d", [128, 192])
def test_block_output_native_actual_host_frame(block, mx, has_scale, d):
    s, facts, frames, roles = _fixture(block, mx, has_scale, d)
    for offset, stream in ((0, 17), (2**33, 23)):
        fresh = {name: f._replace(ptr=f.ptr + offset) if f is not None else None for name, f in facts.items()}
        native, s.native = s.native, None
        try:
            binding_reference.execute_quantized(s, fresh, 0x50000000 + offset, stream, stream)
        finally:
            s.native = native
        expected = frames.pop()
        _execute(s, fresh, roles, 0x50000000 + offset, stream)
        assert frames.pop() == expected


@pytest.mark.L0
@pytest.mark.parametrize("block", [16, 32])
def test_block_output_sm120_host_frame(block):
    s, facts, frames, roles = _fixture(block, arch="sm120")
    binding_reference.execute_quantized(s, facts, 0x50000000, 17, 17)
    expected = frames.pop()
    _execute(s, facts, roles)
    assert frames.pop() == expected


@pytest.mark.L0
@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize(
    "kind", ["missing", "device", "dtype", "alignment", "null", "span", "gap", "overlap", "input", "output", "workspace", "amax", "address"]
)
def test_block_output_bad_current_sf_rejected_before_writes(mx, kind, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: writes.append(a))
    s, facts, frames, roles = _fixture(mx=mx)
    _execute(s, facts, roles)
    frames.clear()
    sf = facts["sf_o"]
    bad = dict(
        missing=None,
        device=sf._replace(device=(2, 1)),
        dtype=sf._replace(dtype="int32"),
        alignment=sf._replace(ptr=sf.ptr + 1),
        null=sf._replace(ptr=0),
        span=sf._replace(span=sf.span - 1),
        gap=sf._replace(strides=(*sf.strides[:-1], 2)),
        overlap=sf._replace(strides=(sf.strides[0], 0, *sf.strides[2:])),
        input=sf._replace(ptr=facts["q"].ptr),
        output=sf._replace(ptr=facts["o"].ptr),
        workspace=sf._replace(ptr=0x50000000),
        amax=sf._replace(ptr=facts["amax_o"].ptr),
        address=sf._replace(ptr=2**63 - 16),
    )[kind]
    if not mx:
        facts.pop("descale_q")  # an invalid output must reject before even the identity write
    with pytest.raises(ValueError):
        _execute(s, dict(facts, sf_o=bad), roles)
    assert frames == writes == []


@pytest.mark.L0
@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize("span", [3072, -1])
def test_block_output_native_token_major_padding_and_alias(mx, span):
    s, facts, frames, roles = _fixture(32, mx)
    s.quant = s.quant._replace(block_output=prep.BlockOutputSpec((0, 150, 4, 8), 3072, 1, True))
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    sf = prep.BufferFacts(0x70000000, "uint8", (2, 0), span, (2, 2, 150, 4), (1200, 4, 8, 1))
    facts["sf_o"] = sf
    _execute(s, facts, roles)
    frames.clear()
    for ptr in (sf.ptr + 2400, sf.ptr + 3072 - 16):
        # Logical bytes stop at 2400; the compiled atom extends through 3072.
        with pytest.raises(ValueError):
            _execute(s, dict(facts, q=facts["q"]._replace(ptr=ptr)), roles)
        assert frames == []
    _execute(s, dict(facts, q=facts["q"]._replace(ptr=sf.ptr + 3072)), roles)
    assert len(frames) == 1


@pytest.mark.L0
@pytest.mark.parametrize("has_scale", [False, True])
def test_block_output_native_mx_scale_presence(has_scale):
    s, facts, frames, roles = _fixture(32, True, has_scale)
    facts["scale_o"] = None if has_scale else prep.BufferFacts(0x41000000, "float32", (2, 0), 1, (1,), (1,))
    with pytest.raises(ValueError, match="scale_o presence"):
        _execute(s, facts, roles)
    assert frames == []


def _assert_bytes(case, other, tensors, other_tensors):
    for name in ("o", "sf_o", "amax_o", "lse"):
        if name not in tensors:
            continue
        a, b = case[tensors[name]], other[other_tensors[name]]
        torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize("block,mx,has_scale", _CASES)
@pytest.mark.parametrize("ordered", [False, True])
def test_block_output_native_graph_fresh_buffers_and_replay(block, mx, has_scale, ordered, monkeypatch):
    from test_sdpa_prepared_block_output import _fp8_case

    g, vp, ws, _, _, _, ts = _fp8_case(block, mxfp8=mx, scale_o=has_scale, stats=True)
    ref, rvp, rws, _, _, _, rts = _fp8_case(block, mxfp8=mx, scale_o=has_scale, stats=True)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None, "parent still binds block output in Python"
    binding_reference.use_reference(ref._compiled_plans[ref._plan_index]._prepared.spec)
    ref.execute(rvp, rws)
    g.execute(vp, ws)
    _assert_bytes(vp, rvp, ts, rts)
    vp = {t: x.clone() for t, x in vp.items()}
    ws = torch.empty_like(ws)
    uids, bufs = tuple(t.get_uid() for t in vp), tuple(vp.values())

    def execute():
        if ordered:
            g.execute(bufs, ws, tensor_uids=uids)
        else:
            g.execute(vp, ws)

    graph = torch.cuda.CUDAGraph()
    try:
        with monkeypatch.context() as guard:
            guard.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native block output rebuilt Python facts"))
            guard.setattr(prep, "execute_quantized", lambda *a, **kw: pytest.fail("native block output entered Python execution"), raising=False)
            execute()
            with torch.cuda.graph(graph):
                execute()
        for name, rt in rts.items():
            if name == "v":
                value = rvp[rt].float().mul(0.5).to(rvp[rt].dtype)
                rvp[rt].copy_(value)
                vp[ts[name]].copy_(value)
        if has_scale:
            rvp[rts["scale_o"]].fill_(2)
            vp[ts["scale_o"]].fill_(2)
        ref.execute(rvp, rws)
        graph.replay()
        _assert_bytes(vp, rvp, ts, rts)
    finally:
        graph.reset()


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("block", [16, 32])
@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_block_output_physical_sf_int64_address(block, mx, product, native):
    from test_sdpa_prepared_block_output import _fp8_case

    batch, plane = (5, 2**30 + 1024) if product else (2, 2**32 + 1024)
    ref = _fp8_case(block, mxfp8=mx, b=batch)
    wide = _fp8_case(block, mxfp8=mx, b=batch, plane_stride=plane)
    g, vp, ws, out, sf, amax, ts = wide
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None
    if not native:
        binding_reference.use_reference(spec)
    binding_reference.use_reference(ref[0]._compiled_plans[ref[0]._plan_index]._prepared.spec)
    guard_offset = ((batch - 1) * plane) % 2**32
    guard = torch.as_strided(sf, (128 * (128 // block),), (1,), storage_offset=guard_offset)
    guard.fill_(0xAD)

    def check():
        torch.testing.assert_close(out, ref[3], rtol=0, atol=0)
        torch.testing.assert_close(sf[:, :, :128], ref[4], rtol=0, atol=0)
        torch.testing.assert_close(amax, ref[5], rtol=0, atol=0)
        assert torch.all(guard == 0xAD)

    ref[0].execute(ref[1], ref[2])
    g.execute(vp, ws)
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        for case in (ref, wide):
            value = case[1][case[6]["v"]]
            value.copy_((value.float() * 0.5).to(value.dtype))
        ref[0].execute(ref[1], ref[2])
        out.fill_(0xAA)
        sf[:, :, :128].fill_(0xAA)
        amax.fill_(float("nan"))
        graph.replay()
        check()
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
def test_block_output_native_typed_nvfp4_storage(mx, ordered):
    from test_sdpa_prepared_block_output import _fp8_case

    g, vp, ws, _, _, _, ts = _fp8_case(16, mxfp8=mx, stats=True)
    assert g._compiled_plans[g._plan_index]._prepared.spec.native is not None
    g.execute(vp, ws)
    expected = {name: vp[ts[name]].view(torch.uint8).clone() for name in ("o", "sf_o", "amax_o", "lse")}
    packed = vp[ts["o"]].view(torch.float4_e2m1fn_x2)
    assert packed.shape == vp[ts["o"]].shape and packed.stride() == vp[ts["o"]].stride()
    vp[ts["o"]] = packed
    uids, bufs = tuple(t.get_uid() for t in vp), tuple(vp.values())

    def call():
        if ordered:
            g.execute(bufs, ws, tensor_uids=uids)
        else:
            g.execute(vp, ws)

    graph = torch.cuda.CUDAGraph()
    try:
        call()
        with torch.cuda.graph(graph):
            call()
        for name in expected:
            vp[ts[name]].view(torch.uint8).fill_(0xAA)
        graph.replay()
        for name, target in expected.items():
            torch.testing.assert_close(vp[ts[name]].view(torch.uint8), target, rtol=0, atol=0)
    finally:
        graph.reset()
