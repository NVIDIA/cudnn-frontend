# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native output gates retain current storage, full-width strides and ungated statistics."""

import sdpa_binding_reference as binding_reference

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_prefill_binding import _prefill_fixture
from test_sdpa_native_fp8_binding import _fixture as _fp8_fixture
from test_sdpa_native_mxfp8_binding import _fixture as _mx_fixture

pytestmark = [pytest.mark.L1]
_KINDS = ("float16", "bfloat16", "fp8", "mx")


def _fixture(kind):
    if kind == "mx":
        s, facts, frames, _, _ = _mx_fixture(d=256, dv=256)
    elif kind == "fp8":
        s, facts, frames, _ = _fp8_fixture(256, 256, 1, "float8_e4m3fn", "bfloat16", arch="sm107")
    else:
        s, facts, frames, _ = _prefill_fixture(256, 256, False, 1, kind, arch="sm107")
    s.gate_expect = kind if kind in ("float16", "bfloat16") else "bfloat16"
    facts["gate"] = facts["q"]._replace(ptr=0x70000000, dtype=s.gate_expect)
    s.native = cudnn._pybind_module._SdpaDenseBinder(s)
    roles = prep._NATIVE_DENSE_ROLES + prep._native_quant_roles(s.quant)
    return s, facts, frames, roles


def _execute(s, facts, roles, workspace=0x50000000, stream=17):
    return s.native.execute(prep._native_pack_from_facts(facts, roles), tuple(range(len(roles))), stream, workspace=workspace)


@pytest.mark.L0
@pytest.mark.parametrize("kind", _KINDS)
def test_native_gate_actual_host_frames_and_changed_pointers(kind):
    s, facts, frames, roles = _fixture(kind)
    for offset, stream in ((0, 17), (2**33, 23)):
        fresh = {name: f._replace(ptr=f.ptr + offset) if f is not None else None for name, f in facts.items()}
        if s.quant is None:
            expected = tuple(binding_reference.bind_dense(s, fresh, stream, stream))
        else:
            native, s.native = s.native, None
            try:
                binding_reference.execute_quantized(s, fresh, 0x50000000 + offset, stream, stream)
            finally:
                s.native = native
            expected = frames.pop()
        _execute(s, fresh, roles, 0x50000000 + offset, stream)
        assert frames.pop() == expected


@pytest.mark.L0
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("bad", ["missing", "device", "dtype", "null", "alignment", "batch", "seq", "stride", "span", "address"])
def test_native_gate_rejects_current_storage_before_writes(kind, bad, monkeypatch):
    writes = []
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *args: writes.append(args))
    s, facts, frames, roles = _fixture(kind)
    _execute(s, facts, roles)
    frames.clear()
    writes.clear()
    f = facts["gate"]
    invalid = dict(
        missing=None,
        device=f._replace(device=(2, 1)),
        dtype=f._replace(dtype="float32"),
        null=f._replace(ptr=0),
        alignment=f._replace(ptr=f.ptr + 2),
        batch=f._replace(shape=(f.shape[0] - 1, *f.shape[1:])),
        seq=f._replace(shape=(*f.shape[:2], f.shape[2] - 1, f.shape[3])),
        stride=f._replace(strides=(f.strides[0] + 1, *f.strides[1:])),
        span=f._replace(span=f.span - 1),
        address=f._replace(ptr=2**63 - 16),
    )[bad]
    if kind == "fp8":
        facts.pop("descale_q")
    with pytest.raises((ValueError, OverflowError)):
        _execute(s, dict(facts, gate=invalid), roles)
    assert frames == writes == []


def _case(kind, gate, *, b=2):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("existing fused SDPA output gate requires SM107")
    if kind in ("fp8", "mx"):
        from test_sdpa_prepared_quantized_gate import _helper

        return _helper(kind == "mx")._case(d=256, arch="sm107", gate=gate, b=b)
    from frost_test_utils import select_engine

    torch.manual_seed(623)
    dtype = getattr(torch, kind)
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF if kind == "float16" else cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    bufs, ts, vp = {}, {}, {}
    for name, heads in (("q", 4), ("k", 2), ("v", 2)):
        bufs[name] = (torch.randn(b, 128, heads, 256, device="cuda") * 0.4).to(dtype).transpose(1, 2)
        ts[name] = g.tensor_like(bufs[name])
        vp[ts[name]] = bufs[name]
    bufs["gate"], ts["gate"] = gate, g.tensor_like(gate)
    vp[ts["gate"]] = gate
    o, lse = g.sdpa(q=ts["q"], k=ts["k"], v=ts["v"], attn_scale=256**-0.5, generate_stats=True)
    shape, strides = [b, 4, 128, 256], [128 * 4 * 256, 256, 4 * 256, 1]
    o.set_dim(shape).set_stride(strides)
    o = g.mul(a=o, b=g.sigmoid(input=ts["gate"]))
    o.set_output(True).set_dim(shape).set_stride(strides)
    lse.set_output(True).set_dim([b, 4, 128, 1]).set_stride([4 * 128, 128, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    bufs["o"] = torch.empty((b, 128, 4, 256), device="cuda", dtype=dtype).transpose(1, 2)
    bufs["lse"] = torch.empty((b, 4, 128), device="cuda")
    ts.update(o=o, lse=lse)
    vp.update({o: bufs["o"], lse: bufs["lse"]})
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, "sdpa_fwd_prefill_sm107", pack_gqa=False)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, vp, ws, bufs, ts


def _same_outputs(bufs, expected):
    for name in ("o", "lse", "amax_o"):
        if name in bufs:
            torch.testing.assert_close(bufs[name], expected[name], rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("ordered", [False, True])
def test_native_gate_fresh_buffers_and_changed_replay(kind, ordered, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("existing fused output gate requires SM107")
    dtype = getattr(torch, kind) if kind in ("float16", "bfloat16") else torch.bfloat16
    gate = torch.full((2, 128, 4, 256), 1, device="cuda", dtype=dtype).transpose(1, 2)
    g, vp, ws, bufs, ts = _case(kind, gate)
    ref, rvp, rws, rb, rt = _case(kind, gate.clone())
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None, "parent still binds the epilogue gate in Python"
    binding_reference.use_reference(ref._compiled_plans[ref._plan_index]._prepared.spec)
    for name, t in ts.items():
        if name in bufs:
            bufs[name] = bufs[name].clone()
            vp[t] = bufs[name]
    ws = torch.empty_like(ws)
    uids, buffers = tuple(t.get_uid() for t in vp), tuple(vp.values())

    def call():
        if ordered:
            g.execute(buffers, ws, tensor_uids=uids)
        else:
            g.execute(vp, ws)

    ref.execute(rvp, rws)
    capture = torch.cuda.CUDAGraph()
    try:
        with monkeypatch.context() as guard:
            guard.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native gate rebuilt Python facts"))
            guard.setattr(prep, "execute_quantized", lambda *a, **kw: pytest.fail("native gate entered Python binding"), raising=False)
            call()
            _same_outputs(bufs, rb)
            with torch.cuda.graph(capture):
                call()
        stats = {name: rb[name].clone() for name in ("lse", "amax_o") if name in rb}
        for value in (-1e4, 1e4):
            bufs["gate"].fill_(value)
            rb["gate"].fill_(value)
            ref.execute(rvp, rws)
            capture.replay()
            _same_outputs(bufs, rb)
            for name, expected in stats.items():
                torch.testing.assert_close(bufs[name], expected, rtol=0, atol=0)
    finally:
        capture.reset()


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("kind", ["bfloat16", "fp8", "mx"])
@pytest.mark.parametrize("product", [False, True])
def test_native_gate_physical_int64_stride_and_product(kind, product):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("existing fused output gate requires SM107")
    compact = 4 * 128 * 256
    batch, plane = (5, 2**30 + compact) if product else (2, 2**32 + compact)
    try:
        gate = torch.empty_strided((batch, 4, 128, 256), (plane, 256, 4 * 256, 1), device="cuda", dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        pytest.skip("physical gate address test requires over 8 GiB")
    gate.fill_(1e4)
    wrapped = ((batch - 1) * plane) % 2**32
    decoy = torch.as_strided(gate, (4, 128, 256), (256, 4 * 256, 1), storage_offset=wrapped)
    decoy.fill_(-1e4)
    g, vp, ws, bufs, ts = _case(kind, gate, b=batch)
    ref, rvp, rws, rb, _ = _case(kind, gate.transpose(1, 2).contiguous().transpose(1, 2), b=batch)
    assert g._compiled_plans[g._plan_index]._prepared.spec.native is not None
    binding_reference.use_reference(ref._compiled_plans[ref._plan_index]._prepared.spec)
    ref.execute(rvp, rws)
    g.execute(vp, ws)
    _same_outputs(bufs, rb)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            g.execute(vp, ws)
        gate.fill_(-1e4)
        rb["gate"].fill_(-1e4)
        decoy.fill_(1e4)
        ref.execute(rvp, rws)
        capture.replay()
        _same_outputs(bufs, rb)
        assert torch.all(decoy == 1e4)
    finally:
        capture.reset()
