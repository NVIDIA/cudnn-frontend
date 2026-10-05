# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Declared padded Stats: pure binding, ordered initialization, and actual stores."""

import math

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep
from test_sdpa_native_thd_binding import _fixture as _half_fixture
from test_sdpa_native_thd_binding import _paged_fixture
from test_sdpa_native_thd_fp8_binding import _fixture as _fp8_fixture

pytestmark = [pytest.mark.L1]


def _fixture(monkeypatch, *, quantized=False, strides=(32, 1, 8), empty=False, paged=False):
    writes = []
    monkeypatch.setattr(prep._buffers, "apply_fill_plan", lambda *a: writes.append(("stats", *a)))
    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: writes.append(("identity", *a)))
    monkeypatch.setattr(prep._buffers, "memset_zero_async", lambda *a: writes.append(("zero", *a)))
    s, facts, frames = _paged_fixture() if paged else _fp8_fixture() if quantized else _half_fixture()
    s.lse_padded, s.s_q_max, s.lse_stride = True, 4, strides
    s.neg_inf = 0xFF800000
    s.lse_fill_plan = tuple(prep._buffers.strided_fill_plan((4, 8, 4), strides))
    s.template[s.index["lse_strides"]] = strides
    s.template[s.index["lse_ext"]] = s.s_q_max
    span = 1 + sum((n - 1) * st for n, st in zip((4, 8, 4), strides))
    facts["lse"] = facts["lse"]._replace(strides=(*strides, 1), span=span, ptr=2**45)
    if empty:
        facts["q"] = facts["q"]._replace(span=0)
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    return s, facts, frames, writes


def _execute(s, facts, native, stream=17, workspace=0x50000000):
    roles = prep._NATIVE_THD_ROLES + (prep._QUANT_ROLES if getattr(s, "quant", None) is not None else ())
    if native:
        return s.native.execute(prep._native_pack_from_facts(facts, roles), tuple(range(len(roles))), workspace, stream)
    saved, s.native = s.native, None
    try:
        fn = prep.execute_quantized if getattr(s, "quant", None) is not None else prep.execute_thd
        return fn(s, facts, workspace, stream, stream)
    finally:
        s.native = saved


@pytest.mark.L0
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("strides", [(32, 1, 8), (32, 4, 1), (1, 16, 4), (80, 1, 16), (2**33, 1, 8)])
def test_padded_bind_is_pure_and_execute_seeds_once(monkeypatch, quantized, empty, strides):
    s, facts, frames, writes = _fixture(monkeypatch, quantized=quantized, empty=empty, strides=strides)
    # Both low-level binders leave storage untouched, even for empty Q.
    roles = prep._NATIVE_THD_ROLES + (prep._QUANT_ROLES if quantized else ())
    native_frame = s.native.bind(prep._native_pack_from_facts(facts, roles), tuple(range(len(roles))), 0x50000000, 17)
    python_frame = prep._bind_thd_python(s, facts, 0x50000000, 17, 17)
    if not quantized and not empty:
        assert tuple(native_frame) == tuple(python_frame)
    assert frames == writes == []
    for offset, stream in ((0, 17), (2**34, 23)):
        changed = {name: f._replace(ptr=f.ptr + offset) for name, f in facts.items()}
        assert _execute(s, changed, False, stream, 0x50000000 + offset) == (not empty)
        expected_frames, expected_writes = frames[:], writes[:]
        frames.clear()
        writes.clear()
        assert _execute(s, changed, True, stream, 0x50000000 + offset) == (not empty)
        assert frames == expected_frames and writes == expected_writes
        assert sum(w[0] == "stats" for w in writes) == 1
        assert writes[0] == ("stats", changed["lse"].ptr, s.lse_fill_plan, s.neg_inf, stream)
        frames.clear()
        writes.clear()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("bad", ["size", "span", "dtype", "device", "alignment", "null", "overflow", "strides", "scalar"])
def test_padded_rejects_before_any_write(monkeypatch, native, empty, bad):
    s, facts, frames, writes = _fixture(monkeypatch, quantized=True, empty=empty)
    f = facts["lse"]
    invalid = dict(
        size=f._replace(shape=(127,)),
        span=f._replace(span=127),
        dtype=f._replace(dtype="bfloat16"),
        device=f._replace(device=(2, 1)),
        alignment=f._replace(ptr=f.ptr + 1),
        null=f._replace(ptr=0),
        overflow=f._replace(ptr=2**63 - 128),
        strides=f._replace(strides=(32, 3, 8, 1)),
    )
    if bad == "scalar":
        facts["descale_q"] = facts["descale_q"]._replace(dtype="int32")
    else:
        facts["lse"] = invalid[bad]
    with pytest.raises(ValueError):
        _execute(s, facts, native)
    assert frames == writes == []


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("bad", ["cpu_sink", "missing_sink", "unexpected_sink", "null_workspace", "unaligned_workspace"])
def test_empty_padded_stats_revalidates_before_initializing(monkeypatch, native, quantized, bad):
    s, facts, frames, writes = _fixture(monkeypatch, quantized=quantized, empty=True)
    s.has_sink = bad != "unexpected_sink"
    sink = prep.BufferFacts(2**44, "float32", (2, 0), s.qh, (s.qh,), (1,))
    if s.has_sink:
        facts["sinks"] = sink
    s.native = cudnn._pybind_module._SdpaThdBinder(s)
    assert not _execute(s, facts, native)
    assert any(w[0] == "stats" for w in writes)
    writes.clear()

    changed, workspace = dict(facts), 0x50000000
    if bad == "cpu_sink":
        changed["sinks"] = sink._replace(device=(1, 0))
    elif bad == "missing_sink":
        del changed["sinks"]
    elif bad == "unexpected_sink":
        changed["sinks"] = sink
    else:
        workspace = 0 if bad == "null_workspace" else workspace + 1
    with pytest.raises(ValueError):
        _execute(s, changed, native, workspace=workspace)
    assert frames == writes == []
    assert not _execute(s, facts, native)
    assert any(w[0] == "stats" for w in writes)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bad", ["missing", "cpu", "span"])
def test_empty_padded_stats_validates_paged_tables(monkeypatch, native, bad):
    s, facts, frames, writes = _fixture(monkeypatch, empty=True, paged=True)
    assert not _execute(s, facts, native)
    writes.clear()
    changed = dict(facts)
    if bad == "missing":
        del changed["block_table"]
    else:
        table = changed["block_table"]
        changed["block_table"] = table._replace(device=(1, 0)) if bad == "cpu" else table._replace(span=1)
    with pytest.raises(ValueError):
        _execute(s, changed, native)
    assert frames == writes == []
    assert not _execute(s, facts, native)


def _gpu_arch():
    from frost_test_utils import _dsl_installed

    cc = torch.cuda.get_device_capability()
    if not _dsl_installed() or cc not in ((10, 0), (10, 3), (10, 7), (12, 0)):
        pytest.skip("requires an existing Blackwell/Rubin THD row and CuTe DSL")
    return "sm107" if cc == (10, 7) else "sm120" if cc == (12, 0) else "sm100"


def _case(*, fp8=False, layout="bsh", arch=None):
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, engine_name
    from frost_test_utils import select_engine

    arch = arch or _gpu_arch()
    row = next(row for row in ENGINE_SPECS if row.name == engine_name(arch=arch, fp8=fp8))
    if not row.capabilities.thd_padded_stats:
        pytest.skip(f"{row.name} currently requires packed THD Stats")
    b, h, hk, sq, sk, d = 2, 8 if layout == "product" else 4, 2, 64, 96, 128
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    cdtype = cudnn.data_type.FP8_E4M3 if fp8 else cudnn.data_type.BFLOAT16
    rng = torch.Generator(device="cuda").manual_seed(913)
    g = cudnn.pygraph(io_data_type=cdtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    buf, t, vp = {}, {}, {}
    for role, heads, seq in (("q", h, sq), ("k", hk, sk), ("v", hk, sk)):
        buf[role] = (torch.randn((b * seq, heads, d), generator=rng, device="cuda") * 0.2).to(dtype)
        t[role] = g.tensor(dim=[b, heads, seq, d], stride=[seq * heads * d, d, heads * d, 1], data_type=cdtype, name=role)
        vp[t[role]] = buf[role]
    kwargs = dict(generate_stats=True, attn_scale=1 / math.sqrt(d), use_padding_mask=True)
    for name, seq, heads in (("q", sq, h), ("kv", sk, hk)):
        cu = torch.arange(b + 1, device="cuda", dtype=torch.int32) * seq
        buf["cu_" + name] = cu
        buf["off_" + name] = cu * heads * d
        for prefix in ("cu_", "off_"):
            t[prefix + name] = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name=prefix + name)
            vp[t[prefix + name]] = buf[prefix + name]
        for role in (("q",) if name == "q" else ("k", "v")):
            t[role].set_ragged_offset(t["off_" + name])
        if arch == "sm120":
            buf["lens_" + name] = torch.full((b,), seq, device="cuda", dtype=torch.int32)
            t["lens_" + name] = g.tensor(dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
            vp[t["lens_" + name]] = buf["lens_" + name]
            del vp[t["cu_" + name]]
            kwargs["seq_len_" + name] = t["lens_" + name]
        else:
            kwargs["cu_seq_len_" + name] = t["cu_" + name]
    if fp8:
        for name in ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"):
            t[name] = g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, name=name)
            buf[name] = torch.ones(1, device="cuda")
            vp[t[name]], kwargs[name] = buf[name], t[name]
        to, ts, _, _ = g.sdpa_fp8(q=t["q"], k=t["k"], v=t["v"], **kwargs)
    else:
        to, ts = g.sdpa(q=t["q"], k=t["k"], v=t["v"], **kwargs)
    buf["o"] = torch.empty((b * sq, h, d), device="cuda", dtype=torch.bfloat16)
    to.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim([b, h, sq, d]).set_stride([sq * h * d, d, h * d, 1]).set_ragged_offset(t["off_q"])
    strides = {
        "bsh": (sq * h, 1, h),
        "bhs": (sq * h, sq, 1),
        "hsb": (1, sq * b, b),
        "gapped": (sq * h * 2, sq * 2, 2),
        "wide": (2**32 + sq * h, sq, 1),
        "product": (sq * h, 2**30 + sq, 1),
    }[layout]
    try:
        buf["lse"] = torch.empty_strided((b, h, sq), strides, device="cuda", dtype=torch.float32)
    except torch.OutOfMemoryError:
        if layout not in ("wide", "product"):
            raise
        pytest.skip("physical Int64 Stats test requires large sparse storage")
    ts.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, sq, 1]).set_stride([*strides, 1])
    vp[to], vp[ts], t["o"], t["lse"] = buf["o"], buf["lse"], to, ts
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine_name(arch=arch, fp8=fp8), pack_gqa=False)
    g.check_support()
    g.build_plans()
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    return g, vp, ws, buf, t


def _check(buf, qlens=(64, 64), klens=(96, 96)):
    qo = ko = 0
    for batch, (nq, nk) in enumerate(zip(qlens, klens)):
        q = buf["q"][qo : qo + nq].float().transpose(0, 1)
        k = buf["k"][ko : ko + nk].float().transpose(0, 1).repeat_interleave(buf["q"].shape[1] // buf["k"].shape[1], 0)
        v = buf["v"][ko : ko + nk].float().transpose(0, 1).repeat_interleave(buf["q"].shape[1] // buf["k"].shape[1], 0)
        scores = (q @ k.transpose(1, 2)) / math.sqrt(128)
        ref = (torch.softmax(scores, dim=-1) @ v).transpose(0, 1)
        torch.testing.assert_close(buf["o"][qo : qo + nq].float(), ref, atol=0.007, rtol=0.04)
        torch.testing.assert_close(buf["lse"][batch, :, :nq], torch.logsumexp(scores, dim=-1), atol=0.006, rtol=0.002)
        assert torch.isneginf(buf["lse"][batch, :, nq:]).all()
        qo += nq
        ko += nk


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bad", ["cpu_sink", "missing_sink"])
def test_empty_padded_standalone_rejection_does_not_write(native, bad):
    if _gpu_arch() != "sm100":
        pytest.skip("standalone SM100/SM103 padded Stats contract")
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 2, 4, 64, 128
    q, k, v = (torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3))
    o = torch.empty_like(q)
    lse = torch.empty((b, h, s), device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, thd=True, thd_stats_padded=True, has_sink=True)
    assert api.check_support()
    api.compile()
    assert api._thd_spec.native is not None
    if not native:
        api._thd_spec.native = None
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    lens = torch.zeros(b, device="cuda", dtype=torch.int32)
    sink = torch.ones(h, device="cuda")

    def call(sinks):
        api.execute(
            q_tensor=q[:, :, :0], k_tensor=k, v_tensor=v, o_tensor=o[:, :, :0], seq_q_lens=lens, seq_kv_lens=lens, lse_tensor=lse, sinks=sinks, workspace=ws
        )

    call(sink)
    assert torch.isneginf(lse).all()
    lse.fill_(12345)
    o.fill_(97)
    ws.fill_(165)
    invalid = torch.ones(h, device="cpu") if bad == "cpu_sink" else None
    with pytest.raises(ValueError, match="sink"):
        call(invalid)
    assert torch.all(lse == 12345) and torch.all(o == 97) and torch.all(ws == 165)
    call(sink)
    assert torch.isneginf(lse).all()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rank", [3, 4])
def test_padded_graph_contiguous_storage_uses_declared_strides(native, fp8, rank):
    _gpu_arch()
    g, vp, ws, buf, t = _case(fp8=fp8, layout="bsh")
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.lse_padded and spec.native is not None
    if not native:
        spec.native = None
    b, h, s = buf["lse"].shape
    for _ in range(2):
        # The graph declares BHS axes with BSH strides; the bound tensor is
        # contiguous storage, exactly as for a flat carrier. Only the oracle
        # uses a logical BHS view. No output copy or adapter kernel is needed.
        storage = torch.full((b, s, h) if rank == 3 else (b, s, h, 1), float("nan"), device="cuda")
        assert storage.is_contiguous() and tuple(storage.stride()[:3]) != tuple(spec.lse_stride)
        vp[t["lse"]] = storage
        buf["lse"] = storage.view(b, s, h).transpose(1, 2)
        g.execute(vp, ws)
        _check(buf)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g.execute(vp, ws)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            g.execute(vp, ws)
        buf["q"].copy_((buf["q"].float() * 0.5).to(buf["q"].dtype))
        storage.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(buf)
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("layout", ["bsh", "bhs", "hsb", "gapped"])
def test_padded_graph_native_rebind_and_replay(fp8, ordered, layout, monkeypatch):
    _gpu_arch()
    g, vp, ws, buf, t = _case(fp8=fp8, layout=layout)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.lse_padded and spec.native is not None
    uids, values = tuple(x.get_uid() for x in vp), tuple(vp.values())

    def call():
        return g.execute(values, ws, tensor_uids=uids) if ordered else g.execute(vp, ws)

    call()
    _check(buf)
    # A geometry cache must never trigger another plan calculation in execute.
    monkeypatch.setattr(prep._buffers, "strided_fill_plan", lambda *a: pytest.fail("fill geometry belongs to prepare"))
    for role in ("q", "k", "v", "o", "lse"):
        old = buf[role]
        new = torch.empty_strided(old.shape, old.stride(), device=old.device, dtype=old.dtype)
        new.copy_(old)
        buf[role] = vp[t[role]] = new
    ws = torch.empty_like(ws)
    values = tuple(vp.values())
    call()
    _check(buf)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            call()
        buf["cu_q"].copy_(torch.tensor([0, 41, 64], device="cuda", dtype=torch.int32))
        buf["cu_kv"].copy_(torch.tensor([0, 79, 130], device="cuda", dtype=torch.int32))
        if "lens_q" in buf:
            buf["lens_q"].copy_(torch.tensor([41, 23], device="cuda", dtype=torch.int32))
            buf["lens_kv"].copy_(torch.tensor([79, 51], device="cuda", dtype=torch.int32))
        buf["off_q"].copy_(buf["cu_q"] * 4 * 128)
        buf["off_kv"].copy_(buf["cu_kv"] * 2 * 128)
        buf["lse"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(buf, (41, 23), (79, 51))
    finally:
        graph.reset()


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("layout", ["wide", "product"])
def test_padded_stats_physical_int64(fp8, native, layout):
    _gpu_arch()
    g, vp, ws, buf, t = _case(fp8=fp8, layout=layout)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.native is not None
    if not native:
        spec.native = None
    # Poison unused low addresses reached by byte-offset narrowing, without
    # initializing the multi-GB holes in the true sparse layout.
    guard = buf["lse"].as_strided((64,), (1,), storage_offset=256)
    guard.fill_(12345)
    buf["lse"].fill_(float("nan"))
    g.execute(vp, ws)
    _check(buf)
    assert torch.all(guard == 12345)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g.execute(vp, ws)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            g.execute(vp, ws)
        buf["q"].copy_((buf["q"].float() * 0.5).to(buf["q"].dtype))
        buf["lse"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check(buf)
        assert torch.all(guard == 12345)
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_padded_mxfp8_graph_current_scales(ordered, native):
    arch = _gpu_arch()
    if arch != "sm100":
        pytest.skip("MXFP8 THD currently has an SM100/SM103 graph row")
    from test_sdpa_prepared_mxfp8 import _case as mx_case, _check as mx_check, _change_scales

    g, vp, ws, buf, tensors = mx_case(thd=True, padded_stats=True)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.lse_padded and spec.native is not None
    if not native:
        spec.native = None
    uids, values = tuple(t.get_uid() for t in vp), tuple(vp.values())

    def call():
        return g.execute(values, ws, tensor_uids=uids) if ordered else g.execute(vp, ws)

    def check():
        # The existing reference emits packed Stats; only the explicit reference
        # view is flattened, never a production execution operand.
        mx_check(dict(buf, lse=buf["lse"].transpose(1, 2).reshape(-1, 4)), thd=True)

    call()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            call()
        _change_scales(buf)
        buf["lse"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()
    finally:
        graph.reset()
