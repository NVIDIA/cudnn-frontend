# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native public-S cases collected through the selected SM107 CI entry.

Conversion, binding, replay, refill stress and generated-PTX checks.
Batch/golden qualification and backward public-S support are separate work.
"""

import inspect
import math

import pytest
import torch

import cudnn
from cudnn.engines.manifest import MANIFEST
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from test_sdpa_prepared_fp8 import _case, _cuda_graph, _check as _check_random
import test_sdpa_public_s_scale as shared

_SHAPES = {
    128: (128, 2, "d128_fp8_multi_q"),
    192: (128, 2, "d192_fp8_multi_q"),
    256: (256, 1, "d256_fp8_single_q"),
    512: (512, 4, "d512_fp8_role_split"),
}


def _graph(d, half, *, stats=True, thd=False, dtype=torch.float8_e4m3fn, source="buffer", pair=(1.0, 1.0), b=1, sq=128, skv=128, uniform=True):
    dv, cga, owner = _SHAPES[d]
    name = "sdpa_fwd_sm107_" + owner
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    knobs = SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0)
    result = _case(
        arch="sm107",
        b=b,
        sq=sq,
        skv=skv,
        d=d,
        dv=dv,
        thd=thd,
        stats=stats,
        dtype=dtype,
        heads=(4, 2),
        record=(family.offered_ids()[name], knobs.to_public()),
        sdpa_kwargs={"softmax_precision": cudnn.data_type.HALF if half else cudnn.data_type.FLOAT},
        s_scale_source=source,
        s_scale_pair=pair,
    )
    g, _, _, bufs, _ = result
    assert g.selected_engine.name == name
    if uniform:
        bufs["q"].zero_()
        bufs["k"].zero_()
        bufs["v"].fill_(1)
        for name in ("descale_q", "descale_k", "descale_v", "scale_o"):
            bufs[name].fill_(1)
    return result


def _check(bufs, value, s, ds):
    expected = value * s * ds
    tol = 0.03 * abs(expected) + 1e-5
    for name in ("o", "amax_o"):
        assert torch.isfinite(bufs[name]).all(), name
    torch.testing.assert_close(bufs["o"].float(), torch.full_like(bufs["o"], expected).float(), rtol=0, atol=tol)
    torch.testing.assert_close(bufs["amax_o"], torch.full_like(bufs["amax_o"], abs(expected)), rtol=0, atol=tol)
    if "lse" in bufs:
        torch.testing.assert_close(bufs["lse"], torch.full_like(bufs["lse"], math.log(128)), rtol=0, atol=1e-3)


@pytest.mark.parametrize("d", [128, 192, 256, 512])
@pytest.mark.parametrize("half", [False, True], ids=["f32-exp", "f16-exp"])
@pytest.mark.parametrize("stats", [False, True], ids=["no-stats", "stats"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"])
def conversion_and_replay(d, half, stats, dtype):
    g, vp, ws, bufs, tensors = _graph(d, half, stats=stats, dtype=dtype)
    pairs = [(1.0, 1.0), (256.0, 1 / 256), (2.0, 1.0), (1.0, 2.0), (1.5, 2 / 3), (0.0, 1.0)]
    if dtype == torch.float8_e4m3fn:
        pairs.append((1.1, 1 / 1.1))
    outputs = [name for name in ("o", "lse", "amax_o") if name in bufs]
    for s, ds in pairs:
        for name, value in (("scale_s", s), ("descale_s", ds)):
            bufs[name] = torch.full_like(bufs[name], value)
            vp[tensors[name]] = bufs[name]
        bufs["v"].fill_(1)
        g.execute(vp, ws)
        _check(bufs, 1, s, ds)
        with _cuda_graph() as replay:
            with torch.cuda.graph(replay):
                g.execute(vp, ws)
            for value in (1, 2):
                bufs["v"].fill_(value)
                for name in outputs:
                    bufs[name].fill_(float("nan"))
                replay.replay()
                _check(bufs, value, s, ds)
                expected = {name: bufs[name].contiguous().view(torch.uint8).clone() for name in outputs}
                for name in outputs:
                    bufs[name].fill_(float("nan"))
                replay.replay()
                for name, reference in expected.items():
                    assert torch.equal(bufs[name].contiguous().view(torch.uint8), reference), name


@pytest.mark.parametrize("d", [128, 192, 256, 512])
@pytest.mark.parametrize("half", [False, True], ids=["f32-exp", "f16-exp"])
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def defaults_and_adapter(d, half, thd, monkeypatch):
    def graph(owner, cga, packed, skv, *, source="buffer", pair=(1.0, 1.0)):
        assert skv == 128
        return _graph(owner, half, thd=packed, source=source, pair=pair)

    monkeypatch.setattr(shared, "_graph", graph)
    shared.check_graph_and_adapter_defaults(d, _SHAPES[d][1], thd, "omitted", (256.0, 1 / 256))


@pytest.mark.parametrize("d", [128, 192, 256, 512])
@pytest.mark.parametrize("half", [False, True], ids=["f32-exp", "f16-exp"])
@pytest.mark.parametrize("host", [False, True], ids=["device-pair", "host-pair"])
def public_s_register_release(d, half, host, tmp_path):
    """Both precompiled host branches must retain the common service release."""
    from service_register_release_checks import check_service_release

    _, cga, owner = _SHAPES[d]
    params = dict(dtype_qkv=0, dtype_o=2, cta_mma=min(cga, 2), softmax_f16=int(half))
    check_service_release(
        tmp_path,
        f"sm107/{owner}/kernel.py",
        params,
        compile_kwargs={"s_scales": (host, host)},
        expected_entries=2 if host else 1,
    )


@pytest.mark.parametrize("d", [128, 192, 256, 512])
@pytest.mark.parametrize("half", [False, True], ids=["f32-exp", "f16-exp"])
@pytest.mark.parametrize("source", ["buffer", "omitted"], ids=["device-pair", "default-pair"])
def random_refill_and_replay(d, half, source):
    """Random quantized data across twelve KV tiles, fresh pointers and retained capture."""
    b, sq, skv = 2, 384, 1536
    g, vp, ws, bufs, tensors = _graph(d, half, source=source, pair=(256.0, 1 / 256), b=b, sq=sq, skv=skv, uniform=False)
    outputs = ("o", "lse", "amax_o")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for iteration in range(2):
            if iteration:
                for name in ("q", "k", "v", *outputs, "descale_q", "descale_k", "descale_v", "scale_o"):
                    bufs[name] = bufs[name].clone()
                    vp[tensors[name]] = bufs[name]
                if source == "buffer":
                    for name in ("scale_s", "descale_s"):
                        bufs[name] = bufs[name].clone()
                        vp[tensors[name]] = bufs[name]
                bufs["v"].copy_((-bufs["v"].float()).to(bufs["v"].dtype))
            g.execute(vp, ws)
            _check_random(bufs, thd=False, b=b, sq=sq, skv=skv)
            with _cuda_graph() as replay:
                with torch.cuda.graph(replay):
                    g.execute(vp, ws)
                for sign in (1, -1):
                    bufs["v"].copy_((bufs["v"].float() * sign).to(bufs["v"].dtype))
                    for name in outputs:
                        bufs[name].fill_(float("nan"))
                    replay.replay()
                    _check_random(bufs, thd=False, b=b, sq=sq, skv=skv)
                    expected = {name: bufs[name].contiguous().view(torch.uint8).clone() for name in outputs}
                    for name in outputs:
                        bufs[name].fill_(float("nan"))
                    replay.replay()
                    for name, reference in expected.items():
                        assert torch.equal(bufs[name].contiguous().view(torch.uint8), reference), name
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("d", [128, 192, 256, 512])
@pytest.mark.parametrize("half", [False, True], ids=["f32-exp", "f16-exp"])
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def public_s_refill_warm_stress(d, half):
    """Every KV ring refills within a launch; byte checks retain all output buffers."""
    from warm_stress import StressCase, assert_warm_stress

    b, sq, skv = 64, 384, 1536
    g, vp, ws, bufs, _ = _graph(d, half, source="omitted", pair=(256.0, 1 / 256), b=b, sq=sq, skv=skv, uniform=False)
    g.execute(vp, ws)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _check_random(bufs, thd=False, b=b, sq=sq, skv=skv)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    api = inspect.getclosurevars(g._compiled_plans[g._plan_index]._compiled.default_stream).nonlocals["api"]
    cfg = api._k_mod.CFG
    assert cfg.SCHEDULER_POLICY == 0 and not cfg.PACK_GQA
    rows_per_cluster = cfg.TILES_Q * cfg.TILE_M * cfg.CTA_MMA
    blocks = math.ceil(sq / rows_per_cluster) * cfg.CGA_M * 4 * b
    refills = math.ceil(skv / cfg.TILE_N) - cfg.STAGES_KV
    assert refills > 0
    assert_warm_stress(
        StressCase(
            f"SM107 D{d} public-S FP{'16' if half else '32'} softmax",
            lambda: g.execute(vp, ws),
            {"o": bufs["o"].transpose(1, 2), "stats": bufs["lse"], "amax_o": bufs["amax_o"]},
            thread_blocks=blocks,
            refills_per_launch=refills,
        ),
        warm=256,
        poisoned=256,
    )


@pytest.mark.parametrize("d", [128, 192, 256, 512])
@pytest.mark.parametrize("half", [False, True], ids=["f32-exp", "f16-exp"])
def refill_sanitizer(d, half):
    """Small grid for instrumented runs, retaining every body's within-launch KV-ring refill."""
    b, sq, skv = 1, 128, 1536
    g, vp, ws, bufs, _ = _graph(d, half, source="buffer", pair=(256.0, 1 / 256), b=b, sq=sq, skv=skv, uniform=False)
    api = inspect.getclosurevars(g._compiled_plans[g._plan_index]._compiled.default_stream).nonlocals["api"]
    assert math.ceil(skv / api._k_mod.CFG.TILE_N) > api._k_mod.CFG.STAGES_KV
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        g.execute(vp, ws)
        _check_random(bufs, thd=False, b=b, sq=sq, skv=skv)
        with _cuda_graph() as replay:
            with torch.cuda.graph(replay):
                g.execute(vp, ws)
            bufs["v"].copy_((-bufs["v"].float()).to(bufs["v"].dtype))
            for name in ("o", "lse", "amax_o"):
                bufs[name].fill_(float("nan"))
            replay.replay()
            _check_random(bufs, thd=False, b=b, sq=sq, skv=skv)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
