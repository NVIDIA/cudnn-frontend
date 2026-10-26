# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native D576x512 through the public prepared half API, with real cache storage."""

import math

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell


@pytest.mark.L0
@requires_dsl
@requires_pre_rubin_blackwell
@pytest.mark.parametrize(
    "heads,kh,page,hnd,v_hnd,split,dtype,mask,sink,log2,packed,optional_stats",
    [
        (96, 1, 32, False, False, 1, torch.bfloat16, "br", False, False, True, False),
        (96, 1, 32, False, True, 4, torch.bfloat16, "br", False, True, True, False),
        (48, 2, 8, True, False, 4, torch.float16, "br", False, False, True, False),
        (24, 2, 16, False, False, 1, torch.float16, "br_window", True, False, True, False),
        (12, 1, 128, True, True, 4, torch.bfloat16, "dense", True, False, True, False),
        (48, 2, 256, True, False, 1, torch.bfloat16, "tl", False, False, True, False),
        (96, 1, 0, False, False, 4, torch.bfloat16, "br", False, False, True, False),
        (128, 1, 0, False, False, 1, torch.float16, "dense", False, False, True, True),
        (3, 3, 32, False, True, 1, torch.float16, "br", False, False, False, False),
        (12, 1, 32, True, False, 1, torch.bfloat16, "multiwave", False, False, True, False),
    ],
)
def test_native_d576_prepared_replay(heads, kh, page, hnd, v_hnd, split, dtype, mask, sink, log2, packed, optional_stats):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    torch.manual_seed(20261026 + heads + page)
    b, sq, skv = (9, 65, 513) if mask == "multiwave" else (3, 4, 257)
    group = heads // kh
    qbase = torch.randn(b, sq, heads, 640, device="cuda", dtype=dtype)
    q = qbase[..., :576]
    pages = (skv + page - 1) // page if page else 0
    npool = b * pages + 3
    if page:
        k = torch.randn((npool, kh, page, 576) if hnd else (npool, page, kh, 576), device="cuda", dtype=dtype)
        if hnd:
            k = k.permute(0, 2, 1, 3)
        if hnd == v_hnd:
            v = k[..., :512]  # Actual absorbed-MLA shared record, with a 576-element row stride.
        else:
            v = torch.randn((npool, kh, page, 576) if v_hnd else (npool, page, kh, 576), device="cuda", dtype=dtype)[..., :512]
            if v_hnd:
                v = v.permute(0, 2, 1, 3)
        table = torch.arange(b * pages, device="cuda", dtype=torch.int32).reshape(b, pages).contiguous()
        table_base_v = torch.full((b, pages * 2 + 3), -1, device="cuda", dtype=torch.int32)
        table_v = table_base_v[:, 1 : 1 + pages * 2 : 2]
        table_v.copy_((table + 2) % npool)
    else:
        k = torch.randn(b, skv, kh, 576, device="cuda", dtype=dtype)
        v = k[..., :512]
        table = table_v = None
    obase = torch.full((b, sq, heads, 576), float("nan"), device="cuda", dtype=dtype)
    o = obase[..., :512]
    lse = None if optional_stats else torch.empty(b, heads, sq, device="cuda", dtype=torch.float32)
    qlens = torch.tensor(([sq, sq - 1, 1] * ((b + 2) // 3))[:b], device="cuda", dtype=torch.int32)
    klens = torch.tensor(([skv, skv // 2, 0] * ((b + 2) // 3))[:b], device="cuda", dtype=torch.int32)
    sinks = torch.linspace(-0.5, 0.5, heads, device="cuda") if sink else None
    causal = mask in ("br", "br_window", "tl")
    br = mask in ("br", "br_window")
    left = 17 if mask == "br_window" else None
    scale = 1 / math.sqrt(192)
    qh, khv, vh, oh = (t.transpose(1, 2) for t in (q, k, v, o))
    api = SdpaFwdDslSm100(
        sample_q=qh,
        sample_k=khv,
        sample_v=vh,
        sample_o=oh,
        sample_lse=lse,
        is_causal=causal,
        causal_bottom_right=br,
        window_size_left=left,
        scale_softmax=scale,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        has_sink=sink,
        stats_log2=log2,
        split_kv=split,
        pack_gqa=packed,
        paged_page_size=page,
        paged_max_seq_len_kv=pages * page if page else None,
        paged_table_stride=tuple(table.stride()) if page else None,
        paged_table_v_stride=tuple(table_v.stride()) if page else None,
    )
    api.check_support()
    api.compile()
    assert api.flavor == (576, 512)
    assert api._k_mod.CFG.TILE_K == 576 and api._k_mod.CFG.N_BMM1_CHUNKS == 3
    assert api._k_mod.CFG.PACK_G == (math.gcd(group, 64) if packed else 1)
    assert api._dense_spec.native is not None
    workspace = torch.empty(max(1, api.scratch_workspace_bytes()), device="cuda", dtype=torch.uint8)

    def launch():
        api.execute(
            qh, khv, vh, oh, lse_tensor=lse, sinks=sinks, seq_q_lens=qlens, seq_kv_lens=klens, block_table=table, block_table_v=table_v, workspace=workspace
        )

    def check():
        torch.cuda.synchronize()
        assert torch.isnan(obase[..., 512:]).all(), "output head padding overwritten"
        for bi, (nq, nk) in enumerate(zip(qlens.cpu().tolist(), klens.cpu().tolist())):
            if not nq:
                continue
            if page:
                rk = k.index_select(0, table[bi].long()).reshape(-1, kh, 576)[:nk]
                rv = v.index_select(0, table_v[bi].long()).reshape(-1, kh, 512)[:nk]
            else:
                rk, rv = k[bi, :nk], v[bi, :nk]
            rk = rk.double().repeat_interleave(group, dim=1)
            rv = rv.double().repeat_interleave(group, dim=1)
            score = torch.einsum("qhd,khd->hqk", q[bi, :nq].double(), rk) * scale
            row = torch.arange(nq, device="cuda")[:, None]
            col = torch.arange(nk, device="cuda")[None, :]
            diag = row + (nk - nq if br else 0)
            if causal:
                score = score.masked_fill((col > diag)[None], float("-inf"))
            if left is not None:
                # The standalone API takes an inclusive left offset, not a window length.
                score = score.masked_fill((col < diag - left)[None], float("-inf"))
            if sink:
                score = torch.cat((score, sinks.double()[:, None, None].expand(heads, nq, 1)), dim=-1)
            weights = score.softmax(-1).nan_to_num()[..., :nk]
            ref = torch.einsum("hqk,khd->qhd", weights, rv)
            ref_s = score.logsumexp(-1) * (math.log2(math.e) if log2 else 1)
            torch.testing.assert_close(
                o[bi, :nq].double(), ref, atol=0.005 if dtype == torch.bfloat16 else 0.002, rtol=0.02 if dtype == torch.bfloat16 else 0.005
            )
            if lse is not None:
                torch.testing.assert_close(lse[bi, :, :nq].double(), ref_s, atol=0.003, rtol=0)

    launch()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        old_debug = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            launch()
        finally:
            torch.cuda.set_sync_debug_mode(old_debug)
    graph.replay()
    check()
    q.mul_(0.5)
    k.mul_(0.75)
    if page and hnd != v_hnd:
        v.mul_(0.6)
    qlens.copy_(torch.tensor(([sq - 1, sq, 1] * ((b + 2) // 3))[:b], device="cuda", dtype=torch.int32))
    klens.copy_(torch.tensor(([0, skv, 65] * ((b + 2) // 3))[:b], device="cuda", dtype=torch.int32))
    if page:
        table.copy_((table + 1) % npool)
        table_v.copy_((table_v + 3) % npool)
    o.fill_(float("nan"))
    if lse is not None:
        lse.fill_(float("nan"))
    graph.replay()
    check()


@pytest.mark.L0
@requires_dsl
@requires_pre_rubin_blackwell
@pytest.mark.parametrize("split,hnd,v_hnd", [(1, False, True), (4, True, False)])
def test_native_d576_graph(split, hnd, v_hnd):
    import cudnn
    from test_sdpa_fwd_paged_sm100 import _run_graph

    _run_graph(
        3,
        96,
        1,
        576,
        32,
        9,
        [257, 129, 0],
        hnd,
        dtype=torch.bfloat16,
        d_v=512,
        stats=True,
        s_q=4,
        causal_br=True,
        pack_gqa=True,
        want_split=split,
        want_cga=2,
        v_hnd=v_hnd,
        replay=True,
        explicit_knobs={cudnn.knob_type.TILE_CGA_M: 2, cudnn.knob_type.SPLIT_KV: split, cudnn.knob_type.PACK_GQA: 1},
    )


@pytest.mark.L1
@requires_dsl
@requires_pre_rubin_blackwell
def test_native_d576_long_cache_split_replay():
    """The absorbed-MLA generation contract: 96 heads, four Q tokens, 128K cache."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    torch.manual_seed(20261026)
    b, heads, sq, skv, page = 1, 96, 4, 131072, 32
    q = torch.randn(b, sq, heads, 576, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    pool = torch.randn(skv // page, page, 1, 576, device="cuda", dtype=torch.bfloat16)
    k, v = pool.transpose(1, 2), pool[..., :512].transpose(1, 2)
    o = torch.empty(b, sq, heads, 512, device="cuda", dtype=q.dtype).transpose(1, 2)
    lse = torch.empty(b, heads, sq, device="cuda", dtype=torch.float32)
    table = torch.arange(skv // page, device="cuda", dtype=torch.int32).view(1, -1)
    qlens = torch.tensor([sq], device="cuda", dtype=torch.int32)
    klens = torch.tensor([skv], device="cuda", dtype=torch.int32)
    scale = 1 / math.sqrt(192)
    api = SdpaFwdDslSm100(
        q,
        k,
        v,
        o,
        lse,
        scale_softmax=scale,
        is_causal=True,
        causal_bottom_right=True,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
        paged_page_size=page,
        paged_max_seq_len_kv=skv,
        pack_gqa=True,
        split_kv=32,
    )
    api.check_support()
    api.compile()
    assert api.flavor == (576, 512) and api._k_mod.CFG.PACK_G == 32
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)

    def launch():
        api.execute(q, k, v, o, lse_tensor=lse, seq_q_lens=qlens, seq_kv_lens=klens, block_table=table, block_table_v=table, workspace=workspace)

    def check(nq, nk):
        torch.cuda.synchronize()
        # MQA shares K/V across heads: do not materialize 96 copies in the reference.
        dense = pool.reshape(skv, 576)[:nk].double()
        score = torch.einsum("hqd,kd->hqk", q[0, :, :nq].double(), dense) * scale
        mask = torch.arange(nk, device="cuda")[None, :] > nk - nq + torch.arange(nq, device="cuda")[:, None]
        score.masked_fill_(mask[None], float("-inf"))
        ref = torch.einsum("hqk,kd->hqd", score.softmax(-1), dense[:, :512])
        torch.testing.assert_close(o[0, :, :nq].double(), ref, atol=0.005, rtol=0.02)
        torch.testing.assert_close(lse[0, :, :nq].double(), score.logsumexp(-1), atol=0.003, rtol=0)

    launch()
    check(sq, skv)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    qlens.fill_(1)
    klens.fill_(32769)
    q.mul_(0.5)
    pool.mul_(0.75)
    graph.replay()
    check(1, 32769)


@pytest.mark.L0
@requires_dsl
@requires_pre_rubin_blackwell
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_native_d576_register_split_sass(tmp_path, arch, monkeypatch):
    import os
    import sys
    from frost_test_utils import assert_no_new_spills, run_sass_probe, sass_probe_source

    # Preserve the tested source and DSL paths in the fresh compiler process.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
    probe = sass_probe_source(
        """
        params = TemplateParams(dtype_qkv=2, d_flavor=576, mma_2x2=True,
            pack_gqa=True, qh_per_kh=96, split_kv=4, seq_q_lens_present=True,
            seq_kv_lens_present=True, paged_kv=True, page_size=32,
            window_right=0, bottom_right=True)
        mod = _load_sm100_kernel_module((576, 512), params)
        print("SMEM_TOTAL", mod._SMEM["total"])
        mod.compile(d_qk=576, d_v=512, paged_hnd=False, paged_v_hnd=True)
        """,
        counts={
            "ALLOC_192": ("USETMAXREG.TRY_ALLOC", "0xc0"),
            "ALLOC_208": ("USETMAXREG.TRY_ALLOC", "0xd0"),
            "DEALLOC_40": ("USETMAXREG.DEALLOC", "0x28"),
            "LDL": ("LDL",),
            "STL": ("STL",),
        },
    )
    result = run_sass_probe(tmp_path, probe_src=probe, arch=arch, params={}, tag="native_d576", timeout=180)
    assert result.expect["SMEM_TOTAL"] == 224120
    assert result.stats["ALLOC_192"] == result.stats["ALLOC_208"] == 1
    assert result.stats["DEALLOC_40"] == 4
    assert_no_new_spills(result.stats, {"LDL": 0, "STL": 0})
