# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""FP8 split launches bind partial slabs and recombined output scalars without tensor fakes."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, requires_blackwell_geforce
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl]

CASES = [
    pytest.param("sm100", 128, 128, marks=requires_pre_rubin_blackwell),
    pytest.param("sm100", 192, 128, marks=requires_pre_rubin_blackwell),
    pytest.param("sm100", 256, 256, marks=requires_pre_rubin_blackwell),
    pytest.param("sm120", 96, 80, marks=requires_blackwell_geforce),
    pytest.param("sm120", 128, 128, marks=requires_blackwell_geforce),
    pytest.param("sm120", 384, 320, marks=requires_blackwell_geforce),
    pytest.param("sm120", 512, 512, marks=requires_blackwell_geforce),
]


@pytest.mark.parametrize("arch,d,dv", CASES)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("split_kv", [2, 4])
def test_prepared_fp8_split_rebind(arch, d, dv, dtype, output_dtype, split_kv, monkeypatch):
    import cutlass.cute as cute

    def tensor_fake(*args, **kwargs):
        pytest.fail("prepared split launch constructed a tensor fake")

    monkeypatch.setattr(cute.runtime, "make_fake_tensor", tensor_fake)
    monkeypatch.setattr(cute.runtime, "make_fake_compact_tensor", tensor_fake)
    g, vp, ws, bufs, tensors = shared._case(arch=arch, d=d, dv=dv, sq=16, skv=256, dtype=dtype, output_dtype=output_dtype, split_kv=split_kv)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None
    assert prepared.spec.combine is not None
    # Explicit graph scales need no identity initialization. A driver memset
    # here adds a captured engine dependency before attention and regresses replay.
    from cudnn.sdpa.fwd import prepared as prep

    monkeypatch.setattr(prep._buffers, "fill_word_async", lambda *a: pytest.fail("split main must specialize the combine-owned scale to one"))
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o", "lse", "amax_o", "descale_q", "descale_k", "descale_v", "scale_o"):
                bufs[name] = bufs[name].clone()
                vp[tensors[name]] = bufs[name]
            bufs["descale_v"].fill_(0.3)
            bufs["scale_o"].fill_(1.9)
            ws = torch.empty_like(ws)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        g.execute(vp, ws)
        shared._check(bufs, thd=False, sq=16, skv=256)


@pytest.mark.parametrize("arch,d,dv", CASES)
@pytest.mark.parametrize("stats,amax", [(True, True), (False, False)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_fp8_split_capture(arch, d, dv, stats, amax, output_dtype):
    g, vp, ws, bufs, _ = shared._case(arch=arch, d=d, dv=dv, sq=16, skv=256, stats=stats, amax=amax, output_dtype=output_dtype, split_kv=4)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=16, skv=256)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with shared._cuda_graph() as graph:
        with torch.cuda.stream(stream):
            torch.cuda.set_sync_debug_mode("error")
            try:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                g.execute(vp, ws)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph, stream=stream):
                g.execute(vp, ws)
        torch.cuda.current_stream().wait_stream(stream)
        bufs["descale_v"].fill_(0.3)
        bufs["scale_o"].fill_(1.9)
        bufs["o"].fill_(float("nan"))
        if amax:
            bufs["amax_o"].fill_(999)
        graph.replay()
        shared._check(bufs, thd=False, sq=16, skv=256)


@pytest.mark.parametrize("arch,d,dv", CASES)
def test_prepared_fp8_split_runtime_kv_and_strides(arch, d, dv, monkeypatch):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors = shared._case(arch=arch, d=d, dv=dv, sq=16, skv=256, split_kv=4, override=True, padded=True, output_dtype=torch.float8_e4m3fn)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None
    owner, combine_owner = prepared.spec.owner, prepared.spec.combine.owner
    for name in ("k", "v"):
        bufs[name] = bufs[name][:, :, :128]
        vp[tensors[name]] = bufs[name]
    # Final Stats are strided independently of the compact partial slab.
    stats_storage = torch.full((2, 4, 32), float("nan"), device="cuda")
    bufs["lse"] = stats_storage[..., ::2]
    vp[tensors["lse"]] = bufs["lse"]
    output_storage = torch.full((2, 16, 4, dv + 8), 12, device="cuda", dtype=torch.float8_e4m3fn)
    bufs["o"] = output_storage[..., :dv].transpose(1, 2)
    bufs["o_storage"] = output_storage
    vp[tensors["o"]] = bufs["o"]
    names = ("k", "v", "o", "lse")
    overrides = dict(
        override_uids=[tensors[n].get_uid() for n in names],
        override_shapes=[list(bufs[n].shape) + ([1] if n == "lse" else []) for n in names],
        override_strides=[list(bufs[n].stride()) + ([1] if n == "lse" else []) for n in names],
    )
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("override must reuse both artifacts"))
    g.execute(vp, ws, **overrides)
    shared._check(bufs, thd=False, sq=16, skv=128)
    with shared._cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws, **overrides)
        bufs["scale_o"].fill_(1.9)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        shared._check(bufs, thd=False, sq=16, skv=128)
    assert torch.all(bufs["o_storage"][..., dv:] == 12)
    assert torch.isnan(stats_storage[..., 1::2]).all()
    assert prepared.spec.owner is owner and prepared.spec.combine.owner is combine_owner


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("arch,d,dv", [CASES[0], CASES[2], CASES[4], CASES[6]])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_fp8_split_output_stride_int64(arch, d, dv, output_dtype):
    row_stride = 2**32 + 4 * dv
    if torch.cuda.mem_get_info()[0] < row_stride * output_dtype.itemsize + 2**30:
        pytest.skip("wide physical row-stride storage needs additional GPU memory")
    g, vp, ws, bufs, tensors = shared._case(arch=arch, d=d, dv=dv, sq=1, skv=128, split_kv=4, override=True, output_dtype=output_dtype)
    try:
        bufs["o"] = torch.empty_strided((2, 4, 1, dv), (row_stride, dv, 4 * dv, 1), device="cuda", dtype=output_dtype)
    except torch.OutOfMemoryError:
        pytest.skip("wide physical row-stride storage unavailable under current GPU memory pressure")
    vp[tensors["o"]] = bufs["o"]
    overrides = dict(override_uids=[tensors["o"].get_uid()], override_shapes=[list(bufs["o"].shape)], override_strides=[list(bufs["o"].stride())])
    bufs["o"].fill_(float("nan"))
    g.execute(vp, ws, **overrides)
    shared._check(bufs, thd=False, sq=1, skv=128)
    with shared._cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws, **overrides)
        bufs["descale_v"].fill_(0.3)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        shared._check(bufs, thd=False, sq=1, skv=128)


@pytest.mark.parametrize("arch,d,dv", CASES + [pytest.param("sm100", 512, 512, marks=requires_pre_rubin_blackwell)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_fp8_split_standalone_default_scales(arch, d, dv, output_dtype):
    """Omitted scales use caller scratch, including a None-specialized main scale."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100, SdpaFwdDslSm120

    torch.manual_seed(827)
    b, sq, skv, hq, hk = 2, 16, 256, 4, 2
    bufs = {}
    for name, seq, heads, dim in (("q", sq, hq, d), ("k", skv, hk, d), ("v", skv, hk, dv)):
        bufs[name] = (torch.randn(b, seq, heads, dim, device="cuda") * 0.4).to(torch.float8_e4m3fn).transpose(1, 2)
    bufs["o"] = torch.empty((b, sq, hq, dv), device="cuda", dtype=output_dtype).transpose(1, 2)
    bufs["amax_o"] = torch.empty(1, device="cuda")
    api_type = SdpaFwdDslSm100 if arch == "sm100" else SdpaFwdDslSm120
    api = api_type(
        **{"sample_" + name: bufs[name] for name in ("q", "k", "v", "o")},
        pertensor_fp8=True,
        pack_gqa=False,
        split_kv=4,
        **({"cga": 2} if arch == "sm100" and d == 192 else {}),
    )
    assert api.check_support()
    size = api.scratch_workspace_bytes()
    workspace = torch.empty(size, device="cuda", dtype=torch.uint8)
    api.compile()
    assert api.scratch_workspace_bytes() == size
    assert api._prepared_fp8 and api._dense_spec.combine is not None
    for name in ("descale_q", "descale_k", "descale_v", "scale_o"):
        bufs[name] = torch.ones(1, device="cuda")
    for explicit in (True, False):
        bufs["descale_q"].fill_(0.8 if explicit else 1.0)
        bufs["scale_o"].fill_(1.9 if explicit else 1.0)
        scalars = {name: bufs[name] for name in ("descale_q", "scale_o")} if explicit else {}
        workspace.fill_(255)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        with shared._cuda_graph() as graph:
            with torch.cuda.graph(graph):
                api.execute(
                    **{name + "_tensor": bufs[name] for name in ("q", "k", "v", "o")},
                    workspace=workspace,
                    amax_o=bufs["amax_o"],
                    **scalars,
                )
            graph.replay()
            shared._check(bufs, thd=False, sq=sq, skv=skv)
