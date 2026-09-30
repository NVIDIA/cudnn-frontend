# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepared SM100 FP8 page pools and existing dense head envelopes."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_prepared_fp8 as shared
import test_sdpa_fwd_paged_sm100 as paged

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _case(*, d=128, hnd=False, split=1, dtype=torch.bfloat16, in_key="e4m3", b=2, pages=16):
    return paged._run_graph_fp8(
        b, 4, 2, d, 16, pages, [16 * pages] * b, hnd, in_key=in_key, out_dt=dtype, s_q=16, explicit_split=split, return_case=True, override=True
    )


def _check(bufs):
    dense = dict(bufs)
    b, _, sq, _ = bufs["q"].shape
    for name in ("k", "v"):
        pool = bufs[name]
        table = bufs[name + "_table"][:, 0, :, 0].long()
        # Gather K and V independently; a shared-table bug must change the answer.
        dense[name] = pool[table].permute(0, 2, 1, 3, 4).reshape(b, 2, -1, pool.shape[-1])
    dense["lse"] = bufs["lse"].squeeze(-1)
    dense["amax_o"] = bufs["amax_o"].reshape(-1)
    shared._check(dense, thd=False, b=b, sq=sq, skv=dense["k"].shape[2])


def _overrides(bufs, tensors, names):
    return dict(
        override_uids=[tensors[n].get_uid() for n in names],
        override_shapes=[list(bufs[n].shape) for n in names],
        override_strides=[list(bufs[n].stride()) for n in names],
    )


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_prepared_fp8_paged_rebind(d, hnd, split, in_key, dtype, monkeypatch):
    import cutlass.cute.runtime as runtime

    monkeypatch.setattr(runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("paged FP8 reentered tensor compilation"))
    monkeypatch.setattr(runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("paged FP8 reentered tensor compilation"))
    g, vp, ws, bufs, tensors = _case(d=d, hnd=hnd, split=split, dtype=dtype, in_key=in_key)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None and prepared.spec.paged
    owner = prepared.spec.owner
    for name in bufs:
        bufs[name] = bufs[name].clone()
        vp[tensors[name]] = bufs[name]
    bufs["v_table"].copy_(bufs["k_table"].flip(2))
    bufs["descale_v"].mul_(0.5)
    bufs["scale_o"].fill_(1.9)
    bufs["o"].fill_(float("nan"))
    bufs["amax_o"].fill_(999)
    g.execute(vp, torch.empty_like(ws))
    _check(bufs)
    assert prepared.spec.owner is owner


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_fp8_paged_strides_and_capture(hnd, split, dtype, monkeypatch):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors = _case(hnd=hnd, split=split, dtype=dtype)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None
    owner = prepared.spec.owner
    # Padded pools keep their in-page layout while changing each runtime stride.
    for name in ("k", "v"):
        old = bufs[name]
        n, h, p, d = old.shape
        raw = torch.empty((n, h, p, d + 16) if hnd else (n, p, h, d + 16), device="cuda", dtype=old.dtype)
        bufs[name] = (raw if hnd else raw.transpose(1, 2))[..., :d]
        bufs[name].copy_(old)
    # Rebind nonunit column strides and independent V page order on the same artifact.
    for name in ("k_table", "v_table"):
        old = bufs[name]
        raw = torch.full((2, 1, old.shape[2] * 2 + 1, 1), -1, device="cuda", dtype=torch.int32)
        bufs[name] = raw[:, :, 1::2]
        bufs[name].copy_(old if name == "k_table" else old.flip(2))
    for name in ("k", "v", "k_table", "v_table"):
        vp[tensors[name]] = bufs[name]
    overrides = _overrides(bufs, tensors, ("k", "v", "k_table", "v_table"))
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("paged rebinding must reuse its artifact"))
    g.execute(vp, ws, **overrides)
    _check(bufs)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with shared._cuda_graph() as graph:
        with torch.cuda.stream(stream):
            torch.cuda.set_sync_debug_mode("error")
            try:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                g.execute(vp, ws, **overrides)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph, stream=stream):
                g.execute(vp, ws, **overrides)
        torch.cuda.current_stream().wait_stream(stream)
        bufs["descale_v"].mul_(0.5)
        bufs["scale_o"].fill_(1.9)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        _check(bufs)
    assert prepared.spec.owner is owner


@pytest.mark.parametrize("d,dv,split", [(64, 64, 1), (64, 64, 4), (112, 96, 1), (112, 96, 4), (384, 320, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_fp8_dense_head_envelopes(d, dv, split, dtype, monkeypatch):
    import cutlass.cute.runtime as runtime

    monkeypatch.setattr(runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("envelope reentered tensor compilation"))
    monkeypatch.setattr(runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("envelope reentered tensor compilation"))
    g, vp, ws, bufs, _ = shared._case(d=d, dv=dv, sq=16, skv=256, output_dtype=dtype, split_kv=split, override=True)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=16, skv=256)
    with shared._cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        bufs["descale_v"].fill_(0.3)
        bufs["scale_o"].fill_(1.9)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        shared._check(bufs, thd=False, sq=16, skv=256)


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
def test_prepared_fp8_paged_table_batch_stride_int64(hnd, split):
    g, vp, ws, bufs, tensors = _case(hnd=hnd, split=split, b=1)
    for name in ("k_table", "v_table"):
        old = bufs[name]
        bufs[name] = old.as_strided(old.shape, (2**32 + old.shape[2], *old.stride()[1:]))
        vp[tensors[name]] = bufs[name]
    g.execute(vp, ws, **_overrides(bufs, tensors, ("k_table", "v_table")))
    _check(bufs)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_fp8_paged_physical_pool_stride_int64(hnd, split, dtype):
    page_stride = 2**32 + 2 * 16 * 128
    if torch.cuda.mem_get_info()[0] < 2 * page_stride + 2**30:
        pytest.skip("two wide FP8 pools require additional GPU memory")
    g, vp, ws, bufs, tensors = _case(hnd=hnd, split=split, dtype=dtype, pages=1)
    inner = (16 * 128, 128, 1) if hnd else (128, 2 * 128, 1)
    try:
        pools = [torch.empty_strided((2, 2, 16, 128), (page_stride, *inner), device="cuda", dtype=bufs["k"].dtype) for _ in range(2)]
    except torch.OutOfMemoryError:
        pytest.skip("wide physical pool allocation unavailable under current memory pressure")
    for name, pool in zip(("k", "v"), pools):
        # A deliberately narrowed page stride reads this NaN-poisoned low page.
        pool.as_strided(pool.shape, (2 * 16 * 128, *inner)).fill_(float("nan"))
        table = bufs[name + "_table"][:, 0, 0, 0].long()
        pool.copy_(bufs[name][table])
        bufs[name] = pool
        bufs[name + "_table"] = torch.arange(2, device="cuda", dtype=torch.int32).reshape(2, 1, 1, 1)
    for name in ("k", "v", "k_table", "v_table"):
        vp[tensors[name]] = bufs[name]
    overrides = _overrides(bufs, tensors, ("k", "v", "k_table", "v_table"))
    bufs["o"].fill_(float("nan"))
    g.execute(vp, ws, **overrides)
    _check(bufs)
    with shared._cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws, **overrides)
        bufs["descale_v"].mul_(0.5)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        _check(bufs)


@pytest.mark.parametrize("role", ["k", "v"])
@pytest.mark.parametrize("defect", ["short_pool", "pool_stride", "table_pointer"])
def test_prepared_fp8_paged_rejects_invalid_overrides_before_launch(role, defect, monkeypatch):
    g, vp, ws, bufs, tensors = _case()
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None
    monkeypatch.setattr(prepared.spec, "fn", lambda *a: pytest.fail("invalid paged metadata reached attention"))
    overrides = {}
    if defect == "table_pointer":
        name = role + "_table"
        vp[tensors[name]] = bufs[name].data_ptr() + 1
        message = "4-byte-aligned"
    else:
        shape, strides = list(bufs[role].shape), list(bufs[role].stride())
        if defect == "short_pool":
            shape[0] += 1
            other = "v" if role == "k" else "k"
            # Keep equal effective pool counts, with actual extra storage only
            # for the other pool. A count mismatch would mask the missing bound.
            n, h, p, d = shape
            bufs[other] = torch.empty((n, p, h, d), device="cuda", dtype=bufs[other].dtype).transpose(1, 2)
            vp[tensors[other]] = bufs[other]
            overrides = dict(
                override_uids=[tensors[role].get_uid(), tensors[other].get_uid()],
                override_shapes=[shape, list(bufs[other].shape)],
                override_strides=[strides, list(bufs[other].stride())],
            )
            message = role + ": page pool spans"
        else:
            strides[0] += 1
            overrides = dict(override_uids=[tensors[role].get_uid()], override_shapes=[shape], override_strides=[strides])
            message = "16-byte aligned"
    with pytest.raises(ValueError, match=message):
        g.execute(vp, ws, **overrides)


@pytest.mark.parametrize("v_table_layout", ["strided", "batch_inner"])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fp8_paged_distinct_table_strides_prepared(v_table_layout, split, hnd, override, dtype, monkeypatch):
    import cutlass.cute as cute
    import cutlass.cute.runtime as runtime

    monkeypatch.setattr(runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("distinct tables reentered tensor compilation"))
    monkeypatch.setattr(runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("distinct tables reentered tensor compilation"))
    g, vp, ws, bufs, tensors = paged._run_graph_fp8(
        2,
        4,
        2,
        128,
        16,
        16,
        [256, 256],
        hnd,
        out_dt=dtype,
        s_q=16,
        explicit_split=split,
        return_case=True,
        v_table_layout=v_table_layout,
        override=override,
    )
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None
    owner = prepared.spec.owner
    assert bufs["k_table"].stride() != bufs["v_table"].stride()
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("table rebinding must reuse its artifact"))
    # New allocations preserve the declared geometry; K/V page values differ.
    for name in ("k_table", "v_table"):
        old = bufs[name]
        bufs[name] = torch.empty_strided(old.shape, old.stride(), dtype=old.dtype, device=old.device)
        bufs[name].copy_(old if name == "k_table" else old.flip(2))
        vp[tensors[name]] = bufs[name]
    overrides = {}
    if override:
        old = bufs["k_table"]
        raw = torch.full((2, 1, old.shape[2] * 3 + 1, 1), -1, device="cuda", dtype=torch.int32)
        bufs["k_table"] = raw[:, :, 1::3, :]
        bufs["k_table"].copy_(old)
        vp[tensors["k_table"]] = bufs["k_table"]
        overrides = _overrides(bufs, tensors, ("k_table", "v_table"))
    g.execute(vp, torch.empty_like(ws), **overrides)
    _check(bufs)
    with shared._cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws, **overrides)
        bufs["k_table"].copy_(bufs["k_table"].flip(2))
        bufs["descale_v"].mul_(0.5)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        _check(bufs)
    assert prepared.spec.owner is owner


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["k", "v"])
@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize("split", [1, 4])
def test_prepared_fp8_independent_table_physical_stride_int64(role, product, split):
    batch, stride, origin = (5, 2**30 + 16, 2**31) if product else (2, 2**32 + 16, 0)
    pages = 2
    extent = origin + (batch - 1) * stride + pages
    if torch.cuda.mem_get_info()[0] < extent * 4 + 2**30:
        pytest.skip("wide physical page table requires additional GPU memory")
    g, vp, ws, bufs, tensors = _case(split=split, b=batch, pages=pages)
    try:
        storage = torch.empty(extent, device="cuda", dtype=torch.int32)
    except torch.OutOfMemoryError:
        pytest.skip("wide physical page-table allocation unavailable")
    name = role + "_table"
    old = bufs[name]
    # Signed Int32 address products stay inside allocated prefix guard storage.
    # An intentionally narrowed control therefore reads valid wrong page IDs.
    decoys = []
    for bi in range(batch):
        offset = bi * stride
        wrapped = ((offset + 2**31) % 2**32) - 2**31
        if wrapped != offset:
            decoy = storage[origin + wrapped : origin + wrapped + pages]
            decoy.fill_(0)
            decoys.append(decoy)
    table = storage.as_strided(old.shape, (stride, stride, 1, 1), origin)
    table.copy_(old)
    bufs[name] = table
    vp[tensors[name]] = table
    overrides = _overrides(bufs, tensors, (name,))
    g.execute(vp, ws, **overrides)
    _check(bufs)
    with shared._cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws, **overrides)
        table.copy_(table.flip(2))
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        _check(bufs)
    for decoy in decoys:
        assert torch.count_nonzero(decoy) == 0
