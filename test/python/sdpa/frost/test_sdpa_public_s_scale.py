# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Public per-tensor FP8 S operands affect conversion, binding and replay."""

import math

import pytest
import torch

import cudnn
from cudnn.engines.manifest import MANIFEST
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_prepared_fp8 import _case, _cuda_graph

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _graph(owner, cga, thd, skv, host=(False, False), source="buffer", pair=(1.0, 1.0)):
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    name = "sdpa_fwd_sm100_d128_fp8_" + owner
    knobs = SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0)
    record = (
        (family.offered_ids()[name], knobs.to_public())
        if owner != "backend"
        else (
            3,
            {cudnn.knob_type.STREAM_K: 0, cudnn.knob_type.KERNEL_CFG: 1, cudnn.knob_type.TILE_CGA_M: 0, cudnn.knob_type.TILE_N: 2, cudnn.knob_type.TILE_M: 3},
        )
    )
    result = _case(b=1, sq=128, skv=skv, thd=thd, heads=(4, 2), record=record, s_scale_host=host, s_scale_source=source, s_scale_pair=pair)
    g, vp, ws, bufs, tensors = result
    if owner != "backend":
        assert g.selected_engine.name == name
    bufs["q"].zero_()
    bufs["k"].zero_()
    bufs["v"].fill_(1)
    for key in ("descale_q", "descale_k", "descale_v", "scale_o"):
        bufs[key].fill_(1)
    return result


def _check(bufs, skv, value, s, ds):
    # Q=K=0 and constant V give an independent attention oracle. The
    # nonbinary pair deliberately exercises intermediate FP8 rounding.
    expected = value * s * ds
    tol = 0.03 * abs(expected) + 1e-5
    for name in ("o", "lse", "amax_o"):
        assert torch.isfinite(bufs[name]).all(), name
    torch.testing.assert_close(bufs["o"].float(), torch.full_like(bufs["o"], expected).float(), rtol=0, atol=tol)
    torch.testing.assert_close(bufs["lse"], torch.full_like(bufs["lse"], math.log(skv)), rtol=0, atol=1e-3)
    torch.testing.assert_close(bufs["amax_o"], torch.full_like(bufs["amax_o"], abs(expected)), rtol=0, atol=tol)


def _poison(bufs):
    for name in ("o", "lse", "amax_o"):
        bufs[name].fill_(float("nan"))


@pytest.mark.parametrize(
    "owner,cga,thd",
    [("multi_q", 1, False), ("multi_q", 2, False), ("dual_kv", 1, False), ("multi_q", 2, True)],
    ids=["dense-multi-cga1", "dense-multi-cga2", "dense-dual-cga1", "thd-multi-cga2"],
)
@pytest.mark.parametrize("skv", [128, 8192], ids=["K128", "K8192-refill"])
@pytest.mark.parametrize("host", [(False, False), (True, True), (True, False), (False, True)], ids=["device", "host", "host-device", "device-host"])
def test_public_s_conversion_rebind_and_replay(owner, cga, thd, skv, host):
    g, vp, ws, bufs, tensors = _graph(owner, cga, thd, skv, host)
    pairs = ((1, 1), (256, 1 / 256), (448, 1 / 448), (2, 1), (1, 2), (1.1, 1 / 1.1), (0, 1))
    snapshots = {}
    for s, ds in pairs:
        # New allocations must be rebound at execute/capture time. Captured
        # host values are fixed launch parameters; device values remain live.
        for name, value in (("scale_s", s), ("descale_s", ds)):
            bufs[name] = torch.full_like(bufs[name], value)
            vp[tensors[name]] = bufs[name]
        g.execute(vp, ws)
        _check(bufs, skv, 1, s, ds)
        with _cuda_graph() as replay:
            with torch.cuda.graph(replay):
                g.execute(vp, ws)
            for value in (1, 2):
                bufs["v"].fill_(value)
                _poison(bufs)
                replay.replay()
                _check(bufs, skv, value, s, ds)
                if value == 1:
                    snapshots[s] = bufs["o"].clone()
                # Snapshot before relaunch so shared output storage cannot
                # make repeatability pass by comparing a buffer with itself.
                expected = {name: bufs[name].clone().view(torch.uint8) for name in ("o", "lse", "amax_o")}
                _poison(bufs)
                replay.replay()
                for name, result in expected.items():
                    assert torch.equal(bufs[name].view(torch.uint8), result), name
        bufs["v"].fill_(1)
    assert not torch.equal(snapshots[1], snapshots[1.1]), "scale_s must affect P conversion, not just the final product"


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_public_device_s_values_change_inside_retained_capture(thd):
    g, vp, ws, bufs, _ = _graph("multi_q", 2, thd, 8192)
    g.execute(vp, ws)
    with _cuda_graph() as replay:
        with torch.cuda.graph(replay):
            g.execute(vp, ws)
        for s, ds in ((1, 1), (256, 1 / 256), (2, 1), (1, 2), (0, 1)):
            bufs["scale_s"].fill_(s)
            bufs["descale_s"].fill_(ds)
            _poison(bufs)
            replay.replay()
            _check(bufs, 8192, 1, s, ds)


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@pytest.mark.parametrize("role", ["scale_s", "descale_s"])
@pytest.mark.parametrize("bad", ["missing", "wrong-device", "too-short"])
def test_public_s_rejects_bad_binding_before_writes(thd, role, bad):
    g, vp, ws, bufs, tensors = _graph("multi_q", 2, thd, 128)
    g.execute(vp, ws)
    buf = bufs[role]
    if bad == "missing":
        vp.pop(tensors[role])
    elif bad == "wrong-device":
        vp[tensors[role]] = buf.cpu()
    else:
        vp[tensors[role]] = buf[:0]
    for name in ("o", "lse", "amax_o"):
        bufs[name].fill_(11)
    ws.fill_(0xA5)
    with pytest.raises((ValueError, RuntimeError), match="scale_s|descale_s|missing a buffer"):
        g.execute(vp, ws)
    for name in ("o", "lse", "amax_o"):
        assert (bufs[name] == 11).all(), name
    assert (ws == 0xA5).all()
    vp[tensors[role]] = buf
    g.execute(vp, ws)
    _check(bufs, 128, 1, 1, 1)


@pytest.mark.parametrize("owner,cga,thd", [("multi_q", 1, False), ("multi_q", 2, True), ("dual_kv", 1, False), ("backend", 1, False)])
@pytest.mark.parametrize("skv", [128, 8192])
@pytest.mark.parametrize("source,pair", [("omitted", (256.0, 1 / 256.0)), ("embedded", (1.0, 1.0)), ("embedded", (2.0, 1.0)), ("embedded", (0.0, 1.0))])
def test_graph_owned_s_scales(owner, cga, thd, skv, source, pair):
    if owner == "backend" and cudnn.backend_version() != 92400:
        pytest.skip("native engine-3 control is pinned to cuDNN 9.24")
    g, vp, ws, bufs, _ = _graph(owner, cga, thd, skv, source=source, pair=pair)
    assert "scale_s" not in bufs and "descale_s" not in bufs
    node = next(n for n in g._nodes if "scale_s" in n.inputs)
    for role, value in zip(("scale_s", "descale_s"), pair):
        tensor = node.inputs[role]
        assert tensor.get_is_pass_by_value() and tensor.pass_by_value == value
        assert tensor not in vp
    # No caller-created scalar buffers, even on 9.24. Both routes consume
    # the same graph declarations, workspace contract and capture API.
    handle = cudnn.create_handle()
    try:

        def execute():
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            g.execute(vp, ws, handle=handle)

        execute()
        _check(bufs, skv, 1, *pair)
        before = torch.cuda.memory_allocated()
        mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            execute()
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        assert torch.cuda.memory_allocated() == before
        with _cuda_graph() as replay:
            with torch.cuda.graph(replay):
                execute()
            for value in (1, 2):
                bufs["v"].fill_(value)
                _poison(bufs)
                replay.replay()
                _check(bufs, skv, value, *pair)
    finally:
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("present", ["scale_s", "descale_s"])
def test_graph_s_scales_require_a_complete_pair(present):
    g = cudnn.pygraph()
    scale = g.tensor(dim=[1] * 4, stride=[1] * 4, data_type=cudnn.data_type.FLOAT)
    with pytest.raises(ValueError, match="supply scale_s and descale_s together"):
        g.sdpa_fp8(**{present: scale})


@pytest.mark.parametrize("dtype,dropout", [(cudnn.data_type.FP8_E5M2, None), (cudnn.data_type.FP8_E4M3, (0.1, 1, 0))])
def test_omitted_s_scales_reject_unqualified_defaults(dtype, dropout):
    g = cudnn.pygraph(io_data_type=dtype)
    with pytest.raises(ValueError, match="E4M3 input without dropout"):
        g.sdpa_fp8(dropout=dropout)


@pytest.mark.parametrize(
    "owner,d,dv,cga,thd,sq",
    [
        ("d128_fp8_swap_ab", 128, 128, 1, False, 4),
        ("d192_fp8_multi_q", 192, 128, 1, False, 128),
        ("d192_fp8_multi_q", 192, 128, 2, False, 128),
        ("d192_fp8_multi_q", 192, 128, 2, True, 128),
        ("d256_fp8_single_q", 256, 256, 1, False, 128),
        ("d256_fp8_single_q", 256, 256, 1, True, 128),
        ("d512_fp8_role_split", 512, 512, 4, False, 128),
        ("d512_fp8_role_split", 512, 512, 4, True, 128),
    ],
)
@pytest.mark.parametrize("host", [(False, False), (True, True)], ids=["device", "host"])
def test_public_s_other_sm100_main_bodies(owner, d, dv, cga, thd, sq, host):
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    name = "sdpa_fwd_sm100_" + owner
    knobs = SdpaFwdKnobs(tile_m=16 if "swap_ab" in owner else 128, tile_n=128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0)
    g, vp, ws, bufs, tensors = _case(
        b=1,
        sq=sq,
        skv=1024,
        d=d,
        dv=dv,
        thd=thd,
        record=(family.offered_ids()[name], knobs.to_public()),
        s_scale_host=host,
    )
    assert g.selected_engine.name == name
    bufs["q"].zero_()
    bufs["k"].zero_()
    for role in ("descale_q", "descale_k", "descale_v", "scale_o"):
        bufs[role].fill_(1)
    for s, ds in ((1, 1), (256, 1 / 256), (2, 1), (1, 2), (1.1, 1 / 1.1), (0, 1)):
        for role, value in (("scale_s", s), ("descale_s", ds)):
            bufs[role] = torch.full_like(bufs[role], value)
            vp[tensors[role]] = bufs[role]
        bufs["v"].fill_(1)
        g.execute(vp, ws)
        _check(bufs, 1024, 1, s, ds)
        with _cuda_graph() as replay:
            with torch.cuda.graph(replay):
                g.execute(vp, ws)
            for value in (1, 2):
                bufs["v"].fill_(value)
                _poison(bufs)
                replay.replay()
                _check(bufs, 1024, value, s, ds)
                snapshot = {role: bufs[role].clone().view(torch.uint8) for role in ("o", "lse", "amax_o")}
                _poison(bufs)
                replay.replay()
                for role, expected in snapshot.items():
                    assert torch.equal(bufs[role].view(torch.uint8), expected), role


@pytest.mark.parametrize("owner,cga,thd", [("multi_q", 1, False), ("multi_q", 2, True), ("backend", 1, False)])
@pytest.mark.parametrize("source", ["omitted", "embedded"])
@pytest.mark.parametrize("ordered", [False, True], ids=["mapping", "ordered"])
def test_graph_owned_runtime_s_override_and_capture(owner, cga, thd, source, ordered):
    if owner == "backend" and cudnn.backend_version() != 92400:
        pytest.skip("native engine-3 control is pinned to cuDNN 9.24")
    g, vp, ws, bufs, _ = _graph(owner, cga, thd, 128, source=source)
    node = next(n for n in g._nodes if "scale_s" in n.inputs)
    scalars = tuple(node.inputs[role] for role in ("scale_s", "descale_s"))
    handle = cudnn.create_handle()
    try:

        def execute():
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            if ordered:
                g.execute(tuple(vp.values()), ws, tensor_uids=tuple(t.get_uid() for t in vp), handle=handle)
            else:
                g.execute(vp, ws, handle=handle)

        for pair in (None, (2.0, 1.0), (1.0, 3.0), None):
            for tensor in scalars:
                vp.pop(tensor, None)
            if pair is not None:
                for tensor, value in zip(scalars, pair):
                    vp[tensor] = torch.tensor([value], dtype=torch.float32)
            _poison(bufs)
            execute()
            _check(bufs, 128, 1, *(pair or (1, 1)))

        scale = torch.tensor([2.0], dtype=torch.float32)
        vp[scalars[0]] = scale
        vp[scalars[1]] = torch.tensor([1.0], dtype=torch.float32)
        execute()
        with _cuda_graph() as replay:
            with torch.cuda.graph(replay):
                execute()
            scale.fill_(3)
            bufs["v"].fill_(2)
            _poison(bufs)
            replay.replay()
            _check(bufs, 128, 2, 2, 1)
            execute()
            _check(bufs, 128, 2, 3, 1)
    finally:
        cudnn.destroy_handle(handle)


def check_graph_and_adapter_defaults(owner, cga, thd, source, pair):
    """Graph-owned S values also reach the standalone adapter's prepared frame."""
    g, vp, ws, bufs, _ = _graph(owner, cga, thd, 128, source=source, pair=pair)
    node = next(node for node in g._nodes if "scale_s" in node.inputs)
    scalars = tuple(node.inputs[role] for role in ("scale_s", "descale_s"))
    plan = g._compiled_plans[g._plan_index]
    prepared, takes_variant_pack = plan._prepared, plan.takes_variant_pack
    assert prepared is not None
    for override in (None, (3.0, 1.0), None):
        for tensor in scalars:
            vp.pop(tensor, None)
        if override is not None:
            for tensor, value in zip(scalars, override):
                vp[tensor] = torch.tensor([value], dtype=torch.float32)
        effective_pair = pair if override is None else override
        _poison(bufs)
        g.execute(vp, ws)
        _check(bufs, 128, 1, *effective_pair)
        expected = {name: bufs[name].clone() for name in ("o", "lse", "amax_o")}
        plan._prepared, plan.takes_variant_pack = None, False
        try:
            _poison(bufs)
            g.execute(vp, ws)
            _check(bufs, 128, 1, *effective_pair)
            for name, value in expected.items():
                assert torch.equal(bufs[name].contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)), name
        finally:
            plan._prepared, plan.takes_variant_pack = prepared, takes_variant_pack


@pytest.mark.parametrize("owner,cga,thd", [("multi_q", 1, False), ("multi_q", 2, True), ("dual_kv", 1, False)])
@pytest.mark.parametrize("source,pair", [("omitted", (256.0, 1 / 256)), ("embedded", (2.0, 1.0))])
def test_graph_and_adapter_defaults(owner, cga, thd, source, pair):
    check_graph_and_adapter_defaults(owner, cga, thd, source, pair)
