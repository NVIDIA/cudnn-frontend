# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM120 D128 paged pools, independent tables and device-length replay."""

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
):
    torch.manual_seed(381)
    b, h, pages = 2, 4, 3
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bufs = {"q": torch.randn(b, sq, h, 128, device="cuda", dtype=dtype).transpose(1, 2) * 0.3}
    for name, hnd, capacity in [("k", k_hnd, pool_capacity), ("v", v_hnd, pool_capacity)]:
        shape = (capacity, kh, page, 128) if hnd else (capacity, page, kh, 128)
        if pool_capacity > 9:
            raw = torch.empty(shape, device="cuda", dtype=dtype)
            raw[:8].normal_(std=0.3)
            raw[-1].fill_(1.0 if name == "k" else 0.75)
            raw[1].fill_(-3.0 if name == "k" else -2.0)
        else:
            raw = torch.randn(shape, device="cuda", dtype=dtype) * 0.3
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
    storage = torch.full((b, sq, h, 136), 123.0, device="cuda", dtype=dtype)
    bufs["o"] = storage[..., :128].transpose(1, 2)
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
        attn_scale=128**-0.5,
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
    selected = select_engine(g, engine_name(arch="sm120"))
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, tile_m=tile_m, tile_n=tile_n, split_kv=1, sched_policy=scheduler, pack_gqa=packed))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    assert g.selected_engine is not None
    assert g._compiled_plans[g._plan_index]._prepared is not None
    assert g.get_workspace_size() == 0
    ws = torch.full((1,), 113, device="cuda", dtype=torch.uint8)
    return g, nodes, bufs, ws, storage, stats_storage


def _check(bufs, page, *, mask="none", stats_log2=False):
    q = bufs["q"].double()
    h, kh = q.shape[1], bufs["k"].shape[1]
    for batch, length in enumerate(bufs["kl"].flatten().tolist()):
        qlength = int(bufs["ql"].flatten()[batch])
        pools = []
        for name, table in [("k", "kt"), ("v", "vt")]:
            rows = [bufs[name][int(bufs[table][batch, 0, i // page, 0]), :, i % page] for i in range(length)]
            tensor = torch.stack(rows, dim=1).double() if rows else q.new_empty((kh, 0, 128))
            pools.append(tensor.repeat_interleave(h // kh, dim=0))
        scores = q[batch] @ pools[0].transpose(-1, -2) / math.sqrt(128)
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
            values = torch.cat((values, values.new_zeros(h, 1, 128)), dim=1)
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
    assert api._k_mod.__file__.endswith("/sm120/prefill_f16.py")
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
            bufs["o"] = storage[..., :128].transpose(1, 2)
            bufs["stats"] = stats_storage[..., ::2]
            ws = ws.clone()
            for name in ("q", "k", "v", "kt", "vt", "ql", "kl") + (("sink",) if sink else ()):
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
        assert (ws == 113).all()
        assert (storage[..., 128:] == 123).all()
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
    for changes in ({"thd": True}, {"d_qk": 120}, {"d_v": 120}, {"is_fp8": True}, {"is_mxfp8": True}):
        assert mismatch(caps, replace(facts, **changes), SdpaFwdKnobs(tile_m=64, tile_n=64, split_kv=1)) is not None
    for knobs in (SdpaFwdKnobs(tile_m=16), SdpaFwdKnobs(tile_m=32), SdpaFwdKnobs(split_kv=2)):
        assert mismatch(caps, facts, knobs) is not None
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("unsupported plans must not compile"))
    for changes in ({"thd": True}, {"split_kv": 2}, {"tile_m": 16}, {"seq_kv_lens_present": False}, {"paged_page_size": 24}, {"sample_bias": bufs["q"]}):
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
