# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepared block-scaled outputs retain physical byte addressing."""

import math

import pytest
import torch

import cudnn
from cudnn.engines import MANIFEST
from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, engine_name
from frost_test_utils import requires_dsl

pytestmark = [requires_dsl]


def _fp8_case(block, *, plane_stride=None, mxfp8=False, stats=False, dtype=torch.float8_e4m3fn, scale_o=True, amax=True, b=2, h=1, sq=128, skv=128):
    cc = torch.cuda.get_device_capability()
    arch = "sm107" if cc == (10, 7) else "sm100" if cc in ((10, 0), (10, 3)) else "sm120"
    if cc not in ((10, 0), (10, 3), (10, 7), (12, 0), (12, 1)):
        pytest.skip("block-scaled output needs a supported Blackwell or Rubin GPU")
    if mxfp8 and cc[0] == 12:
        pytest.skip("SM120 has no MXFP8 input engine")
    generator = torch.Generator(device="cuda").manual_seed(397)
    d = 128
    cols = d // block
    rows = ((sq + 127) // 128 * 128) if plane_stride is None else plane_stride // cols
    assert rows % 128 == 0 and (plane_stride is None or rows * cols == plane_stride)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    vp, ts = {}, {}
    for name, seq in (("q", sq), ("k", skv), ("v", skv)):
        buf = (torch.randn(b, seq, h, d, device="cuda", generator=generator) * 0.4).to(dtype).transpose(1, 2)
        ts[name] = g.tensor_like(buf)
        vp[ts[name]] = buf
    kwargs = {}
    names = ("scale_o",) if mxfp8 else ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o")
    for name in names:
        if name == "scale_o" and not scale_o:
            continue
        t = g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, name=name)
        vp[t] = torch.ones(1, device="cuda")
        kwargs[name] = ts[name] = t
    if mxfp8:
        for name in ("q", "k", "v"):
            tiles = ((sq if name == "q" else skv) + 127) // 128
            sf_rows, columns = (tiles * 4, 128) if name == "v" else (tiles * 128, 4)
            t = g.tensor(
                dim=[b, h, sf_rows, columns],
                stride=[h * tiles * 512, tiles * 512, columns, 1],
                data_type=cudnn.data_type.FP8_E8M0,
                reordering_type=cudnn.tensor_reordering.F8_128x4,
                name="sf_" + name,
            )
            vp[t] = torch.full((b, h, tiles * 512), 127, device="cuda", dtype=torch.uint8)
            kwargs["descale_" + name] = t
            ts["sf_" + name] = t
    sf = torch.empty((b, h, rows, cols), device="cuda", dtype=torch.uint8)
    # Initialize only the islands the kernel must write; untouched capacity
    # between them is deliberately not part of the numerical oracle.
    sf[:, :, :128].fill_(0xAA)
    sf_t = g.tensor(
        dim=[b, h, rows, cols], stride=[h * rows * cols, rows * cols, cols, 1], data_type=cudnn.data_type.FP8_E4M3 if block == 16 else cudnn.data_type.FP8_E8M0
    )
    outputs = (g.sdpa_mxfp8 if mxfp8 else g.sdpa_fp8)(q=ts["q"], k=ts["k"], v=ts["v"], sf_o=sf_t, attn_scale=1 / math.sqrt(d), generate_stats=stats, **kwargs)
    o, lse, am = (outputs[0], outputs[1], outputs[-1])
    if stats:
        lse.set_output(True).set_dim([b, h, sq, 1]).set_stride([h * sq, sq, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        vp[lse] = torch.empty((b, h, sq), device="cuda")
        ts["lse"] = lse
    o.set_output(True).set_dim([b, h, sq, d]).set_stride([sq * h * d, d, h * d, 1])
    o.set_data_type(cudnn.data_type.FP4_E2M1 if block == 16 else cudnn.data_type.FP8_E4M3)
    output = torch.empty((b, sq, h, d // (2 if block == 16 else 1)), device="cuda", dtype=torch.uint8 if block == 16 else torch.float8_e4m3fn).transpose(1, 2)
    amax_buf = None
    vp[o], vp[sf_t] = output, sf
    if amax:
        am.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        amax_buf = torch.full((1,), float("nan"), device="cuda")
        vp[am] = amax_buf
        ts["amax_o"] = am
    g.validate()
    g.build_operation_graph()
    name = engine_name(arch=arch, fp8=not mxfp8, mxfp8=mxfp8)
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == name)
    knobs = SdpaFwdKnobs(tile_m=max(caps.tile_ms), tile_n=max(caps.tile_ns), cga=max(caps.cgas), sched_policy=0, pack_gqa=False, split_kv=1)
    g.create_execution_plan(family.offered_ids()[name], knobs)
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    assert g._compiled_plans[g._plan_index]._prepared is not None
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    ts.update(o=o, sf_o=sf_t)
    return g, vp, ws, output.view(torch.uint8), sf, amax_buf, ts


@pytest.mark.L1
@pytest.mark.parametrize("block", [16, 32])
@pytest.mark.parametrize("mxfp8", [False, True])
def test_block_scaled_sf_plane_stride_above_int32(block, mxfp8):
    compact = _fp8_case(block, mxfp8=mxfp8)
    compact[0].execute(compact[1], compact[2])
    reference_o, reference_sf, reference_amax = (x.clone() for x in compact[3:6])
    assert torch.isfinite(reference_amax).all() and (reference_sf != 0).any()
    wide = _fp8_case(block, plane_stride=2**32 + 1024, mxfp8=mxfp8)

    def check():
        torch.testing.assert_close(wide[3], reference_o, rtol=0, atol=0)
        torch.testing.assert_close(wide[4][:, :, :128], reference_sf, rtol=0, atol=0)
        torch.testing.assert_close(wide[5], reference_amax, rtol=0, atol=0)

    wide[0].execute(wide[1], wide[2])
    check()
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            wide[0].execute(wide[1], wide[2])
        wide[3].fill_(0xAA)
        wide[4][:, :, :128].fill_(0xAA)
        wide[5].fill_(float("nan"))
        captured.replay()
        check()
    finally:
        captured.reset()


@pytest.mark.L0
@pytest.mark.parametrize("block,mxfp8,has_scale", [(16, False, True), (32, False, True), (16, True, True), (32, True, True), (32, True, False)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("amax", [False, True])
def test_block_scaled_prepared_rebind_and_replay(block, mxfp8, has_scale, dtype, stats, amax, monkeypatch):
    reference = _fp8_case(block, mxfp8=mxfp8, dtype=dtype, stats=stats, scale_o=has_scale, amax=amax)
    case = _fp8_case(block, mxfp8=mxfp8, dtype=dtype, stats=stats, scale_o=has_scale, amax=amax)
    g, vp, ws, _, _, _, tensors = case
    ref_g, ref_vp, ref_ws, _, _, _, ref_tensors = reference
    g.execute(vp, ws)
    # Rebind every input/output address before capture; metadata and artifact
    # stay unchanged. Different scale values expose stale scalar/SF pointers.
    rebound = {t: buf.clone() for t, buf in vp.items()}
    scale_values = {"sf_q": 126, "sf_k": 128, "sf_v": 126} if mxfp8 else {"descale_q": 0.75, "descale_k": 1.25, "descale_v": 0.5}
    if has_scale:
        scale_values["scale_o"] = 2.5

    def update(values):
        for name, value in values.items():
            rebound[tensors[name]].fill_(value)
            ref_vp[ref_tensors[name]].fill_(value)

    def check():
        for name in ("o", "sf_o") + (("amax_o",) if amax else ()) + (("lse",) if stats else ()):
            actual, expected = rebound[tensors[name]], ref_vp[ref_tensors[name]]
            if name in ("o", "sf_o"):
                actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    update(scale_values)
    ref_g.execute(ref_vp, ref_ws)
    g.execute(rebound, ws)
    check()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream), monkeypatch.context() as guards:

            def forbidden(*args, **kwargs):
                pytest.fail("prepared execution allocated a tensor")

            for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full"):
                guards.setattr(torch, name, forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                g.execute(rebound, ws)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph, stream=stream):
                g.execute(rebound, ws)
        torch.cuda.current_stream().wait_stream(stream)
        update({"sf_v": 127} if mxfp8 else {"descale_v": 1.5})
        if has_scale:
            update({"scale_o": 0.75})
        ref_g.execute(ref_vp, ref_ws)
        for name in ("o", "sf_o"):
            rebound[tensors[name]].view(torch.uint8).fill_(0xAA)
        if amax:
            rebound[tensors["amax_o"]].fill_(float("nan"))
        if stats:
            rebound[tensors["lse"]].fill_(float("nan"))
        graph.replay()
        check()
    finally:
        graph.reset()


def _sf_output_facts():
    from types import SimpleNamespace
    from cudnn.sdpa.fwd.prepared import BlockOutputSpec, BufferFacts, QuantizedLaunchSpec

    # B=2, H=2, R=150, C=4: declared token-major bytes=2400;
    # the atom layout needs 384 padded rows x 8 columns = 3072 bytes.
    spec = SimpleNamespace(device_index=0, quant=QuantizedLaunchSpec(True, 0, (), BlockOutputSpec((0, 150, 4, 8), 3072, 1, True)))
    sf = BufferFacts(0x10000, "uint8", (2, 0), 3072, (2, 2, 150, 4), (1200, 4, 8, 1))
    return spec, {"sf_o": sf}


@pytest.mark.L0
@pytest.mark.parametrize("bad", ["short", "logical_only", "alignment", "null", "device", "width", "gap", "overlap", "missing", "alias"])
def test_block_scaled_sf_rejects_bad_runtime_facts_after_cache_warmup(bad):
    from cudnn.sdpa.fwd.prepared import _bind_block_output

    spec, facts = _sf_output_facts()
    assert _bind_block_output(spec, facts) == facts["sf_o"].ptr
    sf = facts["sf_o"]
    changes = {
        "short": dict(span=3071),
        "logical_only": dict(span=2400),
        "alignment": dict(ptr=sf.ptr + 1),
        "null": dict(ptr=0),
        "device": dict(device=(2, 1)),
        "width": dict(dtype="float32"),
        "gap": dict(strides=(1208, 4, 8, 1)),
        "overlap": dict(strides=(1200, 4, 4, 1)),
    }
    if bad in changes:
        facts["sf_o"] = sf._replace(**changes[bad])
    elif bad == "missing":
        facts["sf_o"] = None
    else:
        facts["q"] = sf._replace(ptr=sf.ptr + 2560, span=1024, shape=(1024,), strides=(1,))
    with pytest.raises(ValueError):
        _bind_block_output(spec, facts)


@pytest.mark.L0
@pytest.mark.parametrize("span", [3072, -1])
def test_block_scaled_sf_token_major_uses_observed_capacity(span):
    from cudnn.sdpa.fwd.prepared import _bind_block_output

    spec, facts = _sf_output_facts()
    sf = facts["sf_o"]
    facts["sf_o"] = sf._replace(span=span)
    assert _bind_block_output(spec, facts) == sf.ptr
    facts["q"] = sf._replace(ptr=sf.ptr + 3072, span=16, shape=(16,), strides=(1,))
    assert _bind_block_output(spec, facts) == sf.ptr  # adjacent storage is disjoint


@pytest.mark.L0
@pytest.mark.parametrize("block,mxfp8,has_scale", [(16, False, True), (32, False, True), (16, True, True), (32, True, True), (32, True, False)])
def test_block_scaled_graph_and_adapter_bind_same_frame(block, mxfp8, has_scale, monkeypatch):
    g, vp, ws, _, _, _, _ = _fp8_case(block, mxfp8=mxfp8, scale_o=has_scale, stats=True)
    plan = g._compiled_plans[g._plan_index]
    prepared = plan._prepared
    frames = []
    monkeypatch.setattr(prepared.spec, "fn", lambda *args: frames.append(args))
    g.execute(vp, ws)
    plan._prepared, plan.takes_variant_pack = None, False
    try:
        g.execute(vp, ws)
    finally:
        plan._prepared, plan.takes_variant_pack = prepared, True
    assert len(frames) == 2
    for i, name in enumerate(prepared.spec.order):
        if name != "stream":
            assert str(frames[0][i]) == str(frames[1][i]), name


@pytest.mark.L0
@pytest.mark.parametrize("has_scale", [False, True])
def test_block_scaled_mxfp8_scale_presence_matches_compilation(has_scale, monkeypatch):
    from cudnn.sdpa.fwd.prepared import execute_quantized, facts_of_tensor

    g, vp, ws, _, _, _, tensors = _fp8_case(32, mxfp8=True, scale_o=has_scale)
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    facts = {name: facts_of_tensor(vp[t]) for name, t in tensors.items()}
    scale = torch.ones(1, device="cuda")
    facts["scale_o"] = None if has_scale else facts_of_tensor(scale)
    monkeypatch.setattr(spec, "fn", lambda *args: pytest.fail("mismatched scale presence must not launch"))
    with pytest.raises(ValueError, match="scale_o presence"):
        execute_quantized(spec, facts, ws.data_ptr(), None, 0)
