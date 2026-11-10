# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM120 paged pools, independent tables and device-length replay."""

import math
from dataclasses import replace

import cudnn
import pytest
import torch

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_blackwell_geforce, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


def _case(
    dtype,
    kh,
    page,
    k_hnd,
    v_hnd,
    *,
    mask="none",
    sink=False,
    packed=False,
    stats_log2=False,
    tile_m=64,
    tile_n=64,
    sq=67,
    scheduler=0,
    sink_value=None,
    pool_capacity=9,
    dims=(128, 128),
    split_kv=1,
    h=4,
    explicit_plan=False,
):
    torch.manual_seed(381)
    b, pages = 2, 3
    d_qk, d_v = dims
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bufs = {"q": torch.randn(b, sq, h, d_qk, device="cuda", dtype=dtype).transpose(1, 2) * 0.3}
    for name, hnd, capacity in [("k", k_hnd, pool_capacity), ("v", v_hnd, pool_capacity)]:
        d = d_qk if name == "k" else d_v
        width = d if dims == (128, 128) else d + 8
        shape = (capacity, kh, page, width) if hnd else (capacity, page, kh, width)
        if pool_capacity > 9:
            raw = torch.empty(shape, device="cuda", dtype=dtype)
            raw[:8].normal_(std=0.3)
            raw[-1].fill_(1.0 if name == "k" else 0.75)
            raw[1].fill_(-3.0 if name == "k" else -2.0)
        else:
            raw = torch.randn(shape, device="cuda", dtype=dtype) * 0.3
        raw[..., d:].fill_(float("nan"))
        raw = raw[..., :d]
        bufs[name] = raw if hnd else raw.transpose(1, 2)
        bufs[name][8 if pool_capacity > 9 else -1].fill_(float("nan"))
    bufs["kt"] = torch.tensor([[2, 5, -1], [1, 3, -1]], device="cuda", dtype=torch.int32).view(b, 1, pages, 1)
    bufs["vt"] = torch.tensor([[4, 6, -1], [0, 7, -1]], device="cuda", dtype=torch.int32).view(b, 1, pages, 1)
    if pool_capacity > 9:
        bufs["kt"][0, 0, 0, 0] = pool_capacity - 1
        bufs["vt"][0, 0, 0, 0] = pool_capacity - 1
    # Only three rows of the final live page belong to request zero.
    bufs["k"][5, :, 3:].fill_(float("nan"))
    bufs["v"][6, :, 3:].fill_(float("nan"))
    bufs["ql"] = torch.full((b, 1, 1, 1), sq, device="cuda", dtype=torch.int32)
    bufs["ql"][1] = max(1, sq // 2)
    bufs["q"][1, :, max(1, sq // 2) :].fill_(float("nan"))
    bufs["kl"] = torch.tensor([page + 3, 0], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
    storage = torch.full((b, sq, h, d_v + 8), 123.0, device="cuda", dtype=dtype)
    bufs["o"] = storage[..., :d_v].transpose(1, 2)
    stats_storage = torch.full((b, h, sq * 2), 777.0, device="cuda")
    bufs["stats"] = stats_storage[..., ::2]
    if sink:
        bufs["sink"] = torch.randn(1, h, 1, 1, device="cuda")
        if sink_value is not None:
            bufs["sink"].fill_(sink_value)
    nodes = {n: g.tensor_like(t, name=n) for n, t in bufs.items() if n not in ("o", "stats")}
    extra = {}
    if mask != "none":
        extra["diagonal_band_right_bound"] = 3 if mask == "band" else 0
        if mask == "br_swa":
            extra.update(diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT, diagonal_band_left_bound=20)
    if sink:
        extra["sink_token"] = nodes["sink"]
    o, stats = g.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        generate_stats=True,
        attn_scale=d_qk**-0.5,
        use_padding_mask=True,
        seq_len_q=nodes["ql"],
        seq_len_kv=nodes["kl"],
        paged_attention_k_table=nodes["kt"],
        paged_attention_v_table=nodes["vt"],
        paged_attention_max_seq_len_kv=pages * page,
        stats_use_log2=stats_log2,
        **extra,
    )
    o.set_output(True).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, sq, 1]).set_stride([h * sq * 2, sq * 2, 2, 1])
    nodes.update(o=o, stats=stats)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    if explicit_plan:
        from cudnn.engines.manifest import engines_for
        from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

        # Native qualification may request a legal tile that the ranking does
        # not propose for this Q length. Keep ordinary-candidate tests separate.
        selected = next(engine for engine in engines_for(g) if engine.name == engine_name(arch="sm120"))
        knobs = SdpaFwdKnobs(cga=1, tile_m=tile_m, tile_n=tile_n, split_kv=split_kv, sched_policy=scheduler, pack_gqa=packed)
    else:
        selected = select_engine(g, engine_name(arch="sm120"))
        knobs = replace(selected.knobs, tile_m=tile_m, tile_n=tile_n, split_kv=split_kv, sched_policy=scheduler, pack_gqa=packed)
    g.create_execution_plan(selected.engine_id, knobs)
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    assert g.selected_engine is not None
    assert g._compiled_plans[g._plan_index]._prepared is not None
    if split_kv == 1:
        assert g.get_workspace_size() == 0
    ws = torch.full((g.get_workspace_size() + 16,), 113, device="cuda", dtype=torch.uint8)
    return g, nodes, bufs, ws, storage, stats_storage


def _check(bufs, page, *, mask="none", stats_log2=False):
    q = bufs["q"].double()
    h, kh = q.shape[1], bufs["k"].shape[1]
    for batch, length in enumerate(bufs["kl"].flatten().tolist()):
        qlength = int(bufs["ql"].flatten()[batch])
        pools = []
        for name, table in [("k", "kt"), ("v", "vt")]:
            rows = [bufs[name][int(bufs[table][batch, 0, i // page, 0]), :, i % page] for i in range(length)]
            tensor = torch.stack(rows, dim=1).double() if rows else q.new_empty((kh, 0, bufs[name].shape[-1]))
            pools.append(tensor.repeat_interleave(h // kh, dim=0))
        scores = q[batch] @ pools[0].transpose(-1, -2) / math.sqrt(q.shape[-1])
        qr = torch.arange(q.shape[2], device=q.device)[:, None]
        kr = torch.arange(length, device=q.device)[None, :]
        visible = qr < qlength
        if mask != "none":
            diagonal = qr + (length - qlength if mask == "br_swa" else 0)
            visible = visible & (kr <= diagonal + (3 if mask == "band" else 0))
            if mask == "br_swa":
                visible = visible & (kr >= diagonal - 19)
        scores.masked_fill_(~visible, -float("inf"))
        values = pools[1]
        if "sink" in bufs:
            sink = bufs["sink"].flatten().double()[:, None, None].expand(h, q.shape[2], 1).clone()
            sink.masked_fill_((qr >= qlength)[None], -float("inf"))
            scores = torch.cat((scores, sink), dim=-1)
            values = torch.cat((values, values.new_zeros(h, 1, bufs["v"].shape[-1])), dim=1)
        ref = torch.nan_to_num(scores.softmax(-1), nan=0.0) @ values
        torch.testing.assert_close(bufs["o"][batch].double(), ref, atol=2e-3, rtol=2e-2)
        lse = scores.logsumexp(-1) * (math.log2(math.e) if stats_log2 else 1.0)
        torch.testing.assert_close(bufs["stats"][batch].double(), lse, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kh,k_hnd,v_hnd,page", [(2, True, False, 8), (2, False, True, 128), (1, True, False, 16), (1, False, True, 256)])
def test_sm120_frost_paged_native_rebind_replay(
    dtype,
    kh,
    k_hnd,
    v_hnd,
    page,
    monkeypatch,
    mask="none",
    sink=False,
    packed=False,
    stats_log2=False,
    tile_m=64,
    tile_n=64,
    sq=67,
    scheduler=0,
    replays=1,
    sink_value=None,
    endpoints=False,
    capture_adapter=False,
    pool_capacity=9,
    dims=(128, 128),
    split_kv=1,
    h=4,
    explicit_plan=False,
):
    import cutlass.cute as cute

    g, nodes, bufs, ws, storage, stats_storage = _case(
        dtype,
        kh,
        page,
        k_hnd,
        v_hnd,
        mask=mask,
        sink=sink,
        packed=packed,
        stats_log2=stats_log2,
        tile_m=tile_m,
        tile_n=tile_n,
        sq=sq,
        scheduler=scheduler,
        sink_value=sink_value,
        pool_capacity=pool_capacity,
        dims=dims,
        split_kv=split_kv,
        h=h,
        explicit_plan=explicit_plan,
    )
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    api = SdpaFwdDslSm120(
        *(bufs[n] for n in ("q", "k", "v", "o")),
        sample_lse=bufs["stats"],
        tile_m=tile_m,
        tile_n=tile_n,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
        has_sink=sink,
        pack_gqa=packed,
        stats_log2=stats_log2,
        sched_policy=scheduler,
        split_kv=split_kv,
        is_causal=mask != "none",
        causal_bottom_right=mask == "br_swa",
        window_size_left=19 if mask == "br_swa" else None,
        window_size_right=3 if mask == "band" else None,
        paged_page_size=page,
        paged_max_seq_len_kv=3 * page,
    )
    api.check_support()
    api.compile()
    assert api.compute_capability in ((12, 0), (12, 1))
    kernel_file = "decode_f16.py" if tile_m in (16, 32) else ("prefill_d512_f16.py" if min(dims) > 256 else "prefill_f16.py")
    assert api._k_mod.__file__.endswith("/sm120/" + kernel_file)
    assert api._k_mod.PARAMS.split_kv == split_kv
    assert api._k_mod.PARAMS.page_size == page
    assert api._k_mod._host.__name__ == "paged_host"
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    assert spec.paged and spec.page_size == page
    assert {"block_table_ptr", "block_table_v_ptr", "table_strides", "table_v_strides"} <= set(spec.order)
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))
    for iteration in range(2):
        if iteration:
            storage = storage.clone()
            stats_storage = stats_storage.clone()
            bufs["o"] = storage[..., : dims[1]].transpose(1, 2)
            bufs["stats"] = stats_storage[..., ::2]
            ws = ws.clone()
            for name in ("q", "k", "v", "kt", "vt", "ql", "kl") + (("sink",) if sink else ()):
                if name in ("k", "v") and dims != (128, 128):
                    old = bufs[name]
                    fresh = torch.empty_strided(old.shape, old.stride(), dtype=old.dtype, device=old.device)
                    span = sum((n - 1) * st for n, st in zip(old.shape, old.stride())) + 1
                    fresh.as_strided((span,), (1,)).fill_(float("nan"))
                    fresh.copy_(old)
                    bufs[name] = fresh
                else:
                    bufs[name] = bufs[name].clone(memory_format=torch.preserve_format)
            bufs["q"].mul_(0.7)
        vp = {nodes[n]: t for n, t in bufs.items()}
        g.execute(vp, ws)
        _check(bufs, page, mask=mask, stats_log2=stats_log2)

        def execute_adapter():
            api.execute(
                bufs["q"],
                bufs["k"],
                bufs["v"],
                bufs["o"],
                lse_tensor=bufs["stats"],
                seq_q_lens=bufs["ql"],
                seq_kv_lens=bufs["kl"],
                workspace=ws,
                block_table=bufs["kt"],
                block_table_v=bufs["vt"],
                sinks=bufs.get("sink"),
            )

        execute_adapter()
        _check(bufs, page, mask=mask, stats_log2=stats_log2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                previous = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    if capture_adapter:
                        execute_adapter()
                    else:
                        g.execute(vp, ws)
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                finally:
                    torch.cuda.set_sync_debug_mode(previous)
            torch.cuda.current_stream().wait_stream(stream)
            for replay in range(replays):
                bufs["kt"][:, :, 0].copy_(
                    torch.tensor([5, 3] if replay % 2 == 0 else [pool_capacity - 1 if pool_capacity > 9 else 2, 1], device="cuda", dtype=torch.int32).view(
                        2, 1, 1
                    )
                )
                bufs["vt"][:, :, 0].copy_(
                    torch.tensor([6, 7] if replay % 2 == 0 else [pool_capacity - 1 if pool_capacity > 9 else 4, 0], device="cuda", dtype=torch.int32).view(
                        2, 1, 1
                    )
                )
                bufs["kl"].copy_(torch.tensor([0, page - 1] if replay % 2 == 0 else [page + 3, 0], device="cuda", dtype=torch.int32).view(2, 1, 1, 1))
                if endpoints:
                    bufs["sink"].fill_((-float("inf"), float("inf"), 0.5, -float("inf"))[replay % 4])
                graph.replay()
                _check(bufs, page, mask=mask, stats_log2=stats_log2)
        finally:
            graph.reset()
        assert (ws[g.get_workspace_size() :] == 113).all()
        assert (storage[..., dims[1] :] == 123).all()
        assert (stats_storage[..., 1::2] == 777).all()


@pytest.mark.parametrize("scheduler", [0, 1, 2])
@pytest.mark.parametrize("mask", ["none", "causal", "br_swa", "band"])
@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_sm120_frost_paged_mask_sink_pack_stats(mask, sink, packed, scheduler, monkeypatch):
    test_sm120_frost_paged_native_rebind_replay(
        torch.bfloat16,
        2,
        False,
        True,
        128,
        monkeypatch,
        mask=mask,
        sink=sink,
        packed=packed,
        stats_log2=True,
        tile_m=128 if packed else 64,
        tile_n=128 if sink else 64,
        scheduler=scheduler,
    )


def test_sm120_frost_paged_multiwave_replay(monkeypatch):
    test_sm120_frost_paged_native_rebind_replay(torch.float16, 2, True, False, 16, monkeypatch, sq=4097, replays=8)


def test_sm120_frost_paged_scope_declines_before_compile(monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, _sm120_spec, mismatch
    from cudnn.sdpa.graph_analyzer import analyze
    from cudnn.sdpa.fwd.config_sm120 import TemplateParams, validate_params

    g, _, bufs, _, _, _ = _case(torch.float16, 2, 16, True, False)
    facts = analyze(g)
    caps = _sm120_spec().capabilities
    assert mismatch(caps, facts, SdpaFwdKnobs(tile_m=64, tile_n=64, split_kv=1)) is None
    for changes in ({"thd": True}, {"d_qk": 136}, {"d_v": 136}, {"d_qk": 65}, {"is_fp8": True}, {"is_mxfp8": True}):
        assert mismatch(caps, replace(facts, **changes), SdpaFwdKnobs(tile_m=64, tile_n=64, split_kv=1)) is not None
    for knobs in (SdpaFwdKnobs(tile_m=64, split_kv=2), SdpaFwdKnobs(tile_m=128, split_kv=2)):
        assert mismatch(caps, facts, knobs) is not None
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("unsupported plans must not compile"))
    for changes in ({"thd": True}, {"split_kv": 2}, {"seq_kv_lens_present": False}, {"paged_page_size": 24}, {"sample_bias": bufs["q"]}):
        kwargs = dict(tile_m=64, tile_n=64, seq_q_lens_present=True, seq_kv_lens_present=True, paged_page_size=16, paged_max_seq_len_kv=48)
        kwargs.update(changes)
        api = SdpaFwdDslSm120(*(bufs[n] for n in ("q", "k", "v", "o")), **kwargs)
        with pytest.raises(NotImplementedError, match="SM120 paged KV"):
            api.check_support()
    for changes in ({"thd_varlen": True}, {"split_kv": 2}):
        with pytest.raises(NotImplementedError, match="SM120 paged KV"):
            validate_params(TemplateParams(page_size=16, **changes))


def test_sm120_frost_paged_single_query(monkeypatch):
    test_sm120_frost_paged_native_rebind_replay(torch.bfloat16, 1, False, True, 16, monkeypatch, sq=1)


@pytest.mark.parametrize("sink_value", [-float("inf"), float("inf")])
@pytest.mark.parametrize("capture_adapter", [False, True])
def test_sm120_frost_paged_sink_endpoints(sink_value, capture_adapter, monkeypatch):
    test_sm120_frost_paged_native_rebind_replay(
        torch.bfloat16,
        2,
        True,
        False,
        16,
        monkeypatch,
        mask="br_swa",
        sink=True,
        packed=True,
        sink_value=sink_value,
        endpoints=True,
        replays=4,
        capture_adapter=capture_adapter,
    )


def test_sm120_frost_paged_wide_page_product(monkeypatch):
    # 2 heads * 128 rows * 128 elements/page. Index 131073 addresses
    # 2**32 + 32768 ELEMENTS; a narrowed product reads poisoned page 1.
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        pytest.skip("physical wide-pool/rebinding proof needs 40 GiB free")
    test_sm120_frost_paged_native_rebind_replay(
        torch.float16,
        2,
        True,
        False,
        128,
        monkeypatch,
        pool_capacity=131074,
        replays=4,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(8, 8), (16, 32), (24, 40), (32, 24), (48, 56), (56, 64), (64, 64), (64, 128), (72, 120), (96, 88), (120, 128), (128, 120)])
@pytest.mark.parametrize("mode", ["decode", "packed_sink", "band"])
def test_sm120_frost_paged_widths(dtype, dims, mode, monkeypatch):
    test_sm120_frost_paged_native_rebind_replay(
        dtype,
        2,
        mode != "band",
        mode == "band",
        16 if mode != "band" else 128,
        monkeypatch,
        dims=dims,
        sq=1 if mode == "decode" else 67,
        mask="causal" if mode == "decode" else "br_swa" if mode == "packed_sink" else "band",
        sink=mode == "packed_sink",
        packed=mode == "packed_sink",
        stats_log2=mode == "band",
        tile_m=128 if mode == "packed_sink" else 64,
        tile_n=128 if mode == "packed_sink" else 64,
        scheduler=0 if mode == "decode" else 1 if mode == "packed_sink" else 2,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(64, 64), (120, 88)])
@pytest.mark.parametrize("packed", [False, True])
def test_sm120_frost_paged_width_determinism(dtype, dims, packed, *, tile_m=64, tile_n=64, split_kv=1):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
    from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

    d, dv = dims
    h, kh, sq, page, blocks = 4, 2, 17, 16, 5
    rng = torch.Generator().manual_seed(13631)
    q = (torch.randn(5, h, sq, d, generator=rng) * 0.3).to(dtype).cuda()
    k = (torch.randn(37, kh, page, d, generator=rng) * 0.3).to(dtype).cuda()
    v = (torch.randn(37, page, kh, dv, generator=rng) * 0.3).to(dtype).cuda().transpose(1, 2)
    kt = torch.stack([torch.randperm(37, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    vt = torch.stack([torch.randperm(37, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    ql = torch.tensor([17, 1, 0, 9, 16], device="cuda", dtype=torch.int32)
    kl = torch.tensor([67, 33, 0, 49, 1], device="cuda", dtype=torch.int32)
    sinks = torch.linspace(-0.5, 0.5, h).cuda()
    source = dict(q=q, k=k, v=v, kt=kt, vt=vt, ql=ql, kl=kl, sinks=sinks)
    before = {n: t.clone() for n, t in source.items()}
    plans = {}

    def buffers(order):
        index = torch.tensor(order, device="cuda")
        b = len(order)
        qb = torch.full((b, sq, h, d + 8), float("nan"), device="cuda", dtype=dtype)[..., :d].transpose(1, 2)
        qb.copy_(q.index_select(0, index))
        storage = torch.full((b, sq, h, dv + 8), 123.0, device="cuda", dtype=dtype)
        return dict(
            q=qb,
            o=storage[..., :dv].transpose(1, 2),
            storage=storage,
            lse=torch.empty(b, h, sq, device="cuda"),
            kt=kt.index_select(0, index).t().contiguous().t(),
            vt=vt.index_select(0, index).t().contiguous().t(),
            ql=ql.index_select(0, index),
            kl=kl.index_select(0, index),
        )

    def check(t):
        assert (t["storage"][..., dv:] == 123).all()
        want_o = torch.zeros(t["o"].shape, dtype=torch.float64)
        want_lse = torch.full(t["lse"].shape, -torch.inf, dtype=torch.float64)
        for b, (nq, nk) in enumerate(zip(t["ql"].tolist(), t["kl"].tolist())):
            if not nq:
                continue
            qq = t["q"][b, :, :nq].cpu().double()
            kk = k[t["kt"][b].long()].cpu().transpose(0, 1).reshape(kh, blocks * page, d)[:, :nk].double().repeat_interleave(h // kh, 0)
            vv = v[t["vt"][b].long()].cpu().transpose(0, 1).reshape(kh, blocks * page, dv)[:, :nk].double().repeat_interleave(h // kh, 0)
            scores = qq @ kk.transpose(-1, -2) * d**-0.5
            diagonal = torch.arange(nq)[:, None] + nk - nq
            keys = torch.arange(nk)[None, :]
            scores.masked_fill_((keys > diagonal) | (keys < diagonal - 21), -torch.inf)
            scores = torch.cat((scores, sinks.cpu().double().view(h, 1, 1).expand(h, nq, 1)), -1)
            want_o[b, :, :nq] = scores.softmax(-1)[..., :nk] @ vv
            want_lse[b, :, :nq] = scores.logsumexp(-1)
        torch.testing.assert_close(t["o"].cpu().double(), want_o, atol=0.003, rtol=0.02)
        torch.testing.assert_close(t["lse"].cpu().double(), want_lse, atol=0.0006, rtol=0)
        return {n: t[n] for n in ("o", "lse")}

    def run(order, capture=False):
        t = buffers(order)
        b = len(order)
        if b not in plans:
            api = SdpaFwdDslSm120(
                t["q"],
                k,
                v,
                t["o"],
                t["lse"],
                is_causal=True,
                causal_bottom_right=True,
                window_size_left=21,
                has_sink=True,
                seq_q_lens_present=True,
                seq_kv_lens_present=True,
                paged_page_size=page,
                paged_max_seq_len_kv=blocks * page,
                paged_table_stride=t["kt"].stride(),
                paged_table_v_stride=t["vt"].stride(),
                tile_m=tile_m,
                tile_n=tile_n,
                split_kv=split_kv,
                pack_gqa=packed,
                sched_policy=0 if tile_m in (16, 32) or split_kv > 1 else 1,
            )
            api.check_support()
            api.compile()
            kernel_file = "decode_f16.py" if tile_m in (16, 32) else ("prefill_d512_f16.py" if min(dims) > 256 else "prefill_f16.py")
            assert api._k_mod.__file__.endswith("sm120/" + kernel_file)
            assert api._k_mod.PARAMS.split_kv == split_kv
            assert api._k_mod.PARAMS.page_size == page
            plans[b] = api
        api = plans[b]
        workspace = torch.full((api.scratch_workspace_bytes() + 16,), 113, dtype=torch.uint8, device="cuda")

        def execute():
            api.execute(
                t["q"],
                k,
                v,
                t["o"],
                t["lse"],
                sinks=sinks,
                seq_q_lens=t["ql"],
                seq_kv_lens=t["kl"],
                block_table=t["kt"],
                block_table_v=t["vt"],
                workspace=workspace,
            )

        execute()
        result = check(t)
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.cuda.set_sync_debug_mode("error")
                try:
                    execute()
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            try:
                target = snapshot({n: result[n][:1] for n in result})
                t["q"][1:].mul_(0.7)
                t["ql"][1:] = 1
                t["kl"][1:] = 17
                t["kt"][1:].copy_(t["kt"][1:].flip(1))
                t["vt"][1:].copy_(t["vt"][1:].flip(1))
                t["o"].fill_(torch.nan)
                t["lse"].fill_(torch.nan)
                graph.replay()
                result = check(t)
                assert_bitwise({n: result[n][:1] for n in result}, target, "changed neighbors")
            finally:
                torch.cuda.synchronize()
                graph.reset()
        assert (workspace[api.scratch_workspace_bytes() :] == 113).all()
        assert_bitwise(source, before, "unchanged inputs")
        return result

    target = assert_replays(lambda: run([0]), repeats=4)
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [4, 0, 3], [1] * 64 + [0] + [2] * 64):
        actual = snapshot(run(order))
        for row, sequence in enumerate(order):
            if sequence == 0:
                assert_bitwise({n: actual[n][row : row + 1] for n in target}, target, f"batch row {row}")
    run([0, 1, 2, 3, 4], capture=True)
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    suffix = f"-decode{tile_m}-split{split_kv}" if tile_m in (16, 32) else (f"-d512-split{split_kv}" if min(dims) > 256 else "")
    assert_golden(f"sdpa-sm120-paged-d{d}v{dv}-{dtype}-pack{int(packed)}{suffix}", before, actual)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(24, 40), (64, 64), (120, 88), (128, 128), (8, 8), (16, 32)])
@pytest.mark.parametrize("tile_m", [16, 32])
@pytest.mark.parametrize("split_kv", [1, 3])
def test_sm120_frost_paged_decode(dtype, dims, tile_m, split_kv, monkeypatch, *, explicit_plan=False):
    from cudnn.sdpa.fwd.config_sm120 import decode_kv_stage, head_tile

    test_sm120_frost_paged_native_rebind_replay(
        dtype,
        2,
        tile_m == 16,
        tile_m == 32,
        128 if tile_m == 32 else 16,
        monkeypatch,
        dims=dims,
        tile_m=tile_m,
        tile_n=decode_kv_stage(head_tile(dims[0], 16), head_tile(dims[1], 16)),
        split_kv=split_kv,
        sq=1 if tile_m == 16 else 37,
        mask="br_swa" if tile_m == 16 else "band",
        sink=True,
        packed=tile_m == 16,
        stats_log2=tile_m == 32,
        replays=4,
        endpoints=True,
        capture_adapter=split_kv > 1,
        explicit_plan=explicit_plan,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile_m,dims", [(16, (64, 64)), (32, (120, 88))])
@pytest.mark.parametrize("split_kv", [1, 3])
def test_sm120_frost_paged_decode_determinism(dtype, tile_m, dims, split_kv):
    from cudnn.sdpa.fwd.config_sm120 import decode_kv_stage, head_tile

    test_sm120_frost_paged_width_determinism(
        dtype, dims, True, tile_m=tile_m, tile_n=decode_kv_stage(head_tile(dims[0], 16), head_tile(dims[1], 16)), split_kv=split_kv
    )


@pytest.mark.parametrize("page", [8, 64, 256])
@pytest.mark.parametrize("split_kv", [1, 4])
def test_sm120_frost_paged_decode_partial_gqa(page, split_kv, monkeypatch):
    test_sm120_frost_paged_native_rebind_replay(
        torch.bfloat16,
        2,
        False,
        True,
        page,
        monkeypatch,
        dims=(64, 64),
        h=12,
        tile_m=16,
        tile_n=32,
        split_kv=split_kv,
        sq=7,
        mask="none",
        sink=False,
        packed=True,
        replays=4,
        capture_adapter=split_kv > 1,
    )


@pytest.mark.parametrize("tile_m,dims", [(16, (64, 64)), (32, (120, 88))])
@pytest.mark.parametrize("split_kv", [1, 3])
def test_sm120_frost_paged_decode_ring_reuse(tile_m, dims, split_kv, monkeypatch):
    from cudnn.sdpa.fwd.config_sm120 import decode_kv_stage, head_tile

    # More stages per split than ring slots; each physical slot changes phase
    # repeatedly. Short page-boundary cases alone never exercise this reuse.
    test_sm120_frost_paged_native_rebind_replay(
        torch.bfloat16,
        2,
        True,
        False,
        4096,
        monkeypatch,
        dims=dims,
        h=12,
        tile_m=tile_m,
        tile_n=decode_kv_stage(head_tile(dims[0], 16), head_tile(dims[1], 16)),
        split_kv=split_kv,
        sq=7,
        mask="none",
        sink=True,
        packed=True,
        replays=4,
        capture_adapter=split_kv > 1,
    )


@pytest.mark.parametrize("tile_m,dims,split_kv,page", [(16, (64, 64), 3, 2048), (32, (120, 88), 1, 1024)])
def test_sm120_frost_paged_decode_ring_pipeline(tile_m, dims, split_kv, page, *, explicit_plan=False):
    from cudnn.sdpa.fwd.config_sm120 import decode_kv_groups, decode_kv_stage, decode_slots_per_group, head_tile

    dq, dv = (head_tile(d, 16) for d in dims)
    stage = decode_kv_stage(dq, dv)
    assert math.ceil((page + 3) / stage) // split_kv > decode_kv_groups(tile_m) * decode_slots_per_group(dq, dv, tile_m)
    g, nodes, bufs, ws, storage, stats_storage = _case(
        torch.bfloat16, 2, page, True, False, dims=dims, tile_m=tile_m, tile_n=stage, split_kv=split_kv, sq=1, packed=True, explicit_plan=explicit_plan
    )
    # A compact grid keeps sanitizer instrumentation bounded while every
    # producer still reuses a slot after its prior consumer releases it.
    vp = {nodes[n]: t for n, t in bufs.items()}
    for _ in range(2):
        g.execute(vp, ws)
        _check(bufs, page)
    assert (ws[g.get_workspace_size() :] == 113).all()
    assert (storage[..., dims[1] :] == 123).all()
    assert (stats_storage[..., 1::2] == 777).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(136, 8), (8, 192), (192, 128), (128, 192), (248, 256), (256, 248), (256, 256)])
@pytest.mark.parametrize("tile_m", [16, 32])
@pytest.mark.parametrize("split_kv", [1, 3])
def test_sm120_frost_paged_wide_decode(dtype, dims, tile_m, split_kv, monkeypatch):
    test_sm120_frost_paged_decode(dtype, dims, tile_m, split_kv, monkeypatch, explicit_plan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile_m,dims", [(16, (256, 256)), (32, (192, 128))])
@pytest.mark.parametrize("split_kv", [1, 3])
def test_sm120_frost_paged_wide_decode_determinism(dtype, tile_m, dims, split_kv):
    test_sm120_frost_paged_decode_determinism(dtype, tile_m, dims, split_kv)


@pytest.mark.parametrize("tile_m,dims,split_kv,page", [(16, (256, 256), 3, 1024), (32, (248, 192), 1, 512)])
def test_sm120_frost_paged_wide_decode_ring_pipeline(tile_m, dims, split_kv, page):
    test_sm120_frost_paged_decode_ring_pipeline(tile_m, dims, split_kv, page, explicit_plan=True)


def test_sm120_frost_paged_wide_decode_admission():
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, _sm120_spec, mismatch
    from cudnn.sdpa.graph_analyzer import analyze

    g, _, _, _, _, _ = _case(torch.float16, 2, 16, True, False, dims=(256, 256), tile_m=16, tile_n=16, sq=1, explicit_plan=True)
    facts = analyze(g)
    caps = _sm120_spec().capabilities
    assert mismatch(caps, facts) is None
    for tile_m in (16, 32):
        for split in (1, 3):
            assert mismatch(caps, facts, SdpaFwdKnobs(tile_m=tile_m, tile_n=16, split_kv=split)) is None
    for tile_m in (64, 128):
        reason = mismatch(caps, facts, SdpaFwdKnobs(tile_m=tile_m, tile_n=64, split_kv=1))
        assert reason is not None and "decode tile" in reason
    for dims in ((264, 256), (256, 264), (249, 256)):
        assert mismatch(caps, replace(facts, d_qk=dims[0], d_v=dims[1]), SdpaFwdKnobs(tile_m=16, tile_n=16)) is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dims,mask,packed,splits,page,sq,sink",
    [
        ((512, 512), "none", False, 1, 16, 1, False),
        ((512, 512), "none", True, 3, 128, 7, True),
        ((264, 504), "band", False, 1, 8, 67, False),
        ((504, 264), "br_swa", True, 3, 256, 37, True),
        ((512, 512), "none", False, 1, 128, 257, True),
        ((512, 512), "br_swa", False, 3, 16, 129, True),
    ],
    ids=["plain", "split_pack", "q_envelope", "v_envelope", "reuse", "empty_splits"],
)
def test_sm120_frost_paged_d512(dtype, dims, mask, packed, splits, page, sq, sink, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    # A small persistent grid forces K/V barrier reinitialization and Q-slab
    # reuse even in the compact sanitizer geometry.
    monkeypatch.setattr(SdpaFwdDslSm120, "_persistent_ctas", lambda self, device: 2)
    test_sm120_frost_paged_native_rebind_replay(
        dtype,
        2,
        True,
        False,
        page,
        monkeypatch,
        dims=dims,
        mask=mask,
        sink=sink,
        packed=packed,
        stats_log2=splits > 1,
        tile_m=64,
        tile_n=32,
        sq=sq,
        h=12,
        split_kv=splits,
        replays=4,
        capture_adapter=splits > 1,
        explicit_plan=True,
    )


@pytest.mark.parametrize("sink_value", [-float("inf"), float("inf")])
@pytest.mark.parametrize("split_kv", [1, 3])
def test_sm120_frost_paged_d512_sink_limits(sink_value, split_kv, monkeypatch):
    test_sm120_frost_paged_native_rebind_replay(
        torch.bfloat16,
        2,
        False,
        True,
        16,
        monkeypatch,
        dims=(512, 512),
        tile_m=64,
        tile_n=32,
        split_kv=split_kv,
        sq=7,
        sink=True,
        packed=True,
        sink_value=sink_value,
        endpoints=True,
        replays=4,
        capture_adapter=split_kv > 1,
        explicit_plan=True,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("split_kv", [1, 3])
def test_sm120_frost_paged_d512_determinism(dtype, packed, split_kv):
    test_sm120_frost_paged_width_determinism(dtype, (512, 512), packed, tile_m=64, tile_n=32, split_kv=split_kv)


def test_sm120_frost_paged_d512_admission():
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, _sm120_spec, mismatch
    from cudnn.sdpa.graph_analyzer import analyze

    # Ordinary graph candidates must expose the newly legal native path.
    g, _, _, _, _, _ = _case(torch.float16, 2, 16, True, False, dims=(512, 512), tile_m=64, tile_n=32, sq=1)
    facts, caps = analyze(g), _sm120_spec().capabilities
    for dims in ((512, 512), (264, 504), (504, 264)):
        for split in (1, 3):
            assert mismatch(caps, replace(facts, d_qk=dims[0], d_v=dims[1]), SdpaFwdKnobs(tile_m=64, tile_n=32, split_kv=split)) is None
    for dims in ((264, 256), (256, 512), (520, 512), (257, 512)):
        assert mismatch(caps, replace(facts, d_qk=dims[0], d_v=dims[1]), SdpaFwdKnobs(tile_m=64, tile_n=32)) is not None
    for changes in ({"thd": True}, {"is_fp8": True}, {"is_mxfp8": True}):
        assert mismatch(caps, replace(facts, **changes), SdpaFwdKnobs(tile_m=64, tile_n=32)) is not None
    for tile_m, tile_n in ((16, 16), (32, 16), (64, 64), (128, 32)):
        assert mismatch(caps, facts, SdpaFwdKnobs(tile_m=tile_m, tile_n=tile_n)) is not None
