# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 FP8 pointer launches preserve scale, geometry and wide-address contracts."""

from functools import partial

import pytest
import torch

import test_sdpa_prepared_fp8 as shared
from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d", [128, 512])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm120_fp8_workspace_sized_before_compile(thd, d, output_dtype):
    """Standalone callers may allocate the final scratch size after check_support."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    torch.manual_seed(827)
    b, sq, skv, hq, hk = 2, 16, 128, 4, 2
    q = (torch.randn(b, sq, hq, d, device="cuda") * 0.4).to(torch.float8_e4m3fn).transpose(1, 2)
    k = (torch.randn(b, skv, hk, d, device="cuda") * 0.4).to(torch.float8_e4m3fn).transpose(1, 2)
    v = torch.randn_like(k.float()).mul_(0.4).to(torch.float8_e4m3fn)
    o = torch.empty((b, sq, hq, d), device="cuda", dtype=output_dtype).transpose(1, 2)
    api = SdpaFwdDslSm120(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        pertensor_fp8=True,
        thd=thd,
        seq_q_lens_present=False,
        seq_kv_lens_present=thd,
        pack_gqa=False,
        split_kv=1,
        max_total_seq_len_q=b * sq if thd else None,
        max_total_seq_len_kv=b * skv if thd else None,
    )
    assert api.check_support()
    before = api.scratch_workspace_bytes()
    workspace = torch.empty(before, device="cuda", dtype=torch.uint8)
    api.compile()
    assert before == api.scratch_workspace_bytes(), "compilation must not increase caller workspace requirements"
    bufs = {"q": q, "k": k, "v": v, "o": o}
    if thd:
        for name, heads in (("q", hq), ("k", hk), ("v", hk), ("o", hq)):
            bufs[name] = bufs[name].transpose(1, 2).reshape(-1, heads, d)
    for name in ("descale_q", "descale_k", "descale_v", "scale_o"):
        bufs[name] = torch.ones(1, device="cuda")
    bufs["amax_o"] = torch.full((1,), 999.0, device="cuda")
    lens = (
        dict(
            seq_q_lens=torch.full((b,), sq, device="cuda", dtype=torch.int32),
            seq_kv_lens=torch.full((b,), skv, device="cuda", dtype=torch.int32),
        )
        if thd
        else {}
    )
    api.execute(
        q_tensor=bufs["q"],
        k_tensor=bufs["k"],
        v_tensor=bufs["v"],
        o_tensor=bufs["o"],
        workspace=workspace,
        **{name: bufs[name] for name in ("descale_q", "descale_k", "descale_v", "scale_o", "amax_o")},
        **lens,
    )
    shared._check(bufs, thd=thd, sq=sq, skv=skv)


@pytest.fixture(autouse=True)
def sm120_cases(monkeypatch):
    monkeypatch.setattr(shared, "_case", partial(shared._case, arch="sm120"))


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(96, 80), (128, 128), (192, 128), (256, 256), (384, 320), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_sm120_fp8_rebind_scales_and_buffers(thd, d, dv, dtype, output_dtype, monkeypatch):
    import cutlass.cute as cute

    monkeypatch.setattr(cute.runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("prepared plan constructed a tensor fake"))
    monkeypatch.setattr(cute.runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("prepared plan constructed a compact tensor fake"))
    shared.test_prepared_fp8_rebind_scales_and_buffers(thd, True, True, dtype, d, dv, output_dtype, monkeypatch)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (384, 320), (512, 512)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e5m2])
def test_sm120_fp8_capture_replay_reads_current_scales(thd, d, dv, output_dtype, monkeypatch):
    shared.test_prepared_fp8_capture_replay_reads_current_scales(thd, monkeypatch, d, dv, output_dtype)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (512, 512)])
def test_sm120_fp8_without_optional_outputs(thd, d, dv, monkeypatch):
    shared.test_prepared_fp8_rebind_scales_and_buffers(thd, False, False, torch.float8_e4m3fn, d, dv, torch.bfloat16, monkeypatch)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (384, 320), (512, 512)])
def test_sm120_fp8_graph_and_adapter_bind_the_same_frame(thd, d, dv, monkeypatch):
    shared.test_prepared_fp8_graph_and_adapter_bind_the_same_frame(thd, monkeypatch, d, dv)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (512, 512)])
def test_sm120_fp8_output_row_stride_above_int32(dtype, d, dv, thd):
    shared.test_prepared_fp8_thd_output_row_stride_above_int32(dtype, d, dv, thd=thd)


@pytest.mark.parametrize("d,dv", [(128, 128), (384, 320), (512, 512)])
def test_sm120_fp8_empty_thd_resets_amax_without_attention(d, dv, monkeypatch):
    shared.test_prepared_fp8_empty_thd_resets_amax_without_attention(d, dv, monkeypatch)


@pytest.mark.parametrize("d,dv", [(96, 80), (128, 128), (384, 320), (512, 512)])
def test_sm120_fp8_dense_input_strides(d, dv):
    shared.test_prepared_fp8_dense_input_strides(d, dv)


@pytest.mark.parametrize("output_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("padding", [8, 16])
def test_sm120_fp8_output_pitch_preserves_conversion_route(output_dtype, padding, monkeypatch):
    shared.test_fp8_output_pitch_preserves_prepared_and_conversion_routes(output_dtype, padding, monkeypatch)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (384, 320), (512, 512)])
def test_sm120_fp8_bounded_geometry_override(thd, d, dv, monkeypatch):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors = shared._case(d=d, dv=dv, thd=thd, override=True, skv=256)
    plan = g._compiled_plans[g._plan_index]
    assert plan._prepared is not None
    owner = plan._prepared.spec.owner
    uids, shapes, strides = [], [], []
    for name in ("q", "k", "v", "o", "lse"):
        t = tensors[name]
        seq = 113 if name in ("k", "v") else 128
        bufs[name] = bufs[name][:seq].clone() if thd else bufs[name][:1, :, :seq].clone()
        vp[t] = bufs[name]
        shape = list(t.get_dim())
        shape[0], shape[2] = 1, seq
        uids.append(t.get_uid())
        shapes.append(shape)
        strides.append(list(t.get_stride()) if thd else list(bufs[name].stride()) + ([1] if name == "lse" else []))
    if thd:
        for name in ("cu_q", "cu_kv", "off_q", "off_kv", "off_lse") + (("off_v", "off_o") if d != dv else ()):
            t = tensors[name]
            n = 1 if name.startswith("cu_") else 2
            vp[t] = vp[t][:n].clone()
            if name == "cu_kv":
                vp[t].fill_(113)
            uids.append(t.get_uid())
            shapes.append([n, 1, 1, 1])
            strides.append([1, 1, 1, 1])
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("override must reuse the artifact"))
    g.execute(vp, ws, override_uids=uids, override_shapes=shapes, override_strides=strides)
    assert plan._prepared.spec.owner is owner
    shared._check(bufs, thd=thd, b=1, skv=113)
