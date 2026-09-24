# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged MXFP8 KV caches on the FROST SM100 d256 MXFP8 engine.

``sdpa_mxfp8`` over cuDNN's paged-cache contract, with the descale tensors as page
pools sharing the K/V block tables: ``descale_k [num_pages, H_kv, page_size, D/32]``,
``descale_v [num_pages, H_kv, page_size/32, D]``, both F8_128x4-reordered, so ``page_size % 128 == 0``
(a page holds whole SF atoms). Reference: dequantize the pools, gather the live tokens, fp32 attention.
"""

import math

import pytest
import torch

from frost_test_utils import _SM, offers_engine, requires_blackwell, requires_dsl, select_engine

pytestmark = [requires_blackwell, requires_dsl, pytest.mark.skipif(_SM == 107, reason="paged MXFP8 KV is not wired on the Rubin (SM107) kernels")]

D = 256
_BLOCK = 32
_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_CUDNN_OTYPE = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float8_e4m3fn: "FP8_E4M3", torch.float8_e5m2: "FP8_E5M2"}


def _quantize(t, b, h, s, d, fp8, *, columnwise):
    """-> (fp8 data, SF bytes, per-element dequant scale, SF graph dims); ``columnwise`` scales along S (V), else D (Q/K)."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d_scale_pad = -(-(-(-d // _BLOCK)) // 4) * 4
    s_scale_pad = -(-(-(-s // _BLOCK)) // 4) * 4
    data_d, dq_d, sf_d, data_s, dq_s, sf_s = quantize_to_mxfp8(t, b, h, s, d, _BLOCK, fp8, with_ref=True)
    if columnwise:
        return data_s, sf_s, dq_s.reshape(b, h, s, d), (b, h, s_scale_pad, d_scale_pad * _BLOCK)
    return data_d, sf_d, dq_d.reshape(b, h, s, d), (b, h, s_scale_pad * _BLOCK, d_scale_pad)


def _ref(qd, kd_pool, vd_pool, bt_k, bt_v, seq_lens, scale, *, causal_br=False, window_left=None, sinks=None):
    """fp32 attention over each sequence's live tokens -> (O, LSE); an empty sequence gives O := 0 / LSE := -inf."""
    B, H, s_q, d_qk = qd.shape
    KH, P = kd_pool.shape[1], kd_pool.shape[2]
    d_v = vd_pool.shape[-1]
    dev = qd.device
    out = torch.zeros(B, H, s_q, d_v, device=dev, dtype=torch.float32)
    lse = torch.full((B, H, s_q), float("-inf"), device=dev, dtype=torch.float32)
    for b, L in enumerate(seq_lens.tolist()):
        if L == 0:
            continue
        n = (L + P - 1) // P
        k = kd_pool[bt_k[b, :n].long()].permute(1, 0, 2, 3).reshape(KH, -1, d_qk)[:, :L].repeat_interleave(H // KH, dim=0)
        v = vd_pool[bt_v[b, :n].long()].permute(1, 0, 2, 3).reshape(KH, -1, d_v)[:, :L].repeat_interleave(H // KH, dim=0)
        s = torch.einsum("hqd,hld->hql", qd[b], k) * scale
        i = torch.arange(s_q, device=dev).view(s_q, 1)
        j = torch.arange(L, device=dev).view(1, L)
        diag = L - s_q if causal_br else 0
        if causal_br:
            s = s.masked_fill(j > i + diag, float("-inf"))
        if window_left is not None:  # cuDNN left bound: keys at rel <= diag - W are out
            s = s.masked_fill(j - i <= diag - window_left, float("-inf"))
        if sinks is not None:  # a virtual column with no V row
            s = torch.cat([s, sinks.view(H, 1, 1).expand(H, s_q, 1)], -1)
            p = torch.softmax(s, -1)[..., :L]
        else:
            p = torch.softmax(s, -1)
        out[b] = torch.einsum("hql,hld->hqd", p, v)
        lse[b] = torch.logsumexp(s, -1)
    return out, lse


def _table(num_pages, B, max_pages, *, batch_inner=False):
    perm = torch.randperm(num_pages, device="cuda")[: B * max_pages].to(torch.int32)
    if batch_inner:  # strides (1, ., B, .): the layout test_mhas_v2's harness builds
        return perm.reshape(max_pages, 1, B, 1).transpose(0, 2)
    return perm.view(B, 1, max_pages, 1).contiguous()


def _pools(B, KH, P, max_pages, hnd, fp8, *, d_qk=D, d_v=D, seed=0, separate_v=False, batch_inner=False):
    """Quantized page pools (container views), SF bytes, dequantized pools and block tables (V's own when ``separate_v``)."""
    torch.manual_seed(seed)
    dev = "cuda"
    num_pages = B * max_pages + 5
    Kf = torch.randn(num_pages, KH, P, d_qk, device=dev) * 0.5
    Vf = torch.randn(num_pages, KH, P, d_v, device=dev) * 0.5
    K8, sfk, dqk, sfk_dims = _quantize(Kf, num_pages, KH, P, d_qk, fp8, columnwise=False)
    V8, sfv, dqv, sfv_dims = _quantize(Vf, num_pages, KH, P, d_v, fp8, columnwise=True)
    if hnd:
        k_c, v_c = K8, V8
    else:
        k_c = K8.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        v_c = V8.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    bt_k = _table(num_pages, B, max_pages, batch_inner=batch_inner)
    bt_v = _table(num_pages, B, max_pages, batch_inner=batch_inner) if separate_v else bt_k
    return dict(
        k_c=k_c,
        v_c=v_c,
        sfk_g=sfk.view(torch.uint8).reshape(sfk_dims),
        sfv_g=sfv.view(torch.uint8).reshape(sfv_dims),
        kd=K8.float() * dqk,
        vd=V8.float() * dqv,
        bt_k=bt_k,
        bt_v=bt_v,
        num_pages=num_pages,
    )


def _poison_dead_pages(pools, lens, P, d_v):
    """NaN every page a block table names past its sequence's live pages (data 0x7F/0xFF, SF 0xFF = E8M0 NaN):
    the kernel promises TMA-OOB page -1 there, so dereferencing a dead slot poisons O through 0 * NaN."""
    num_pages, KH = pools["num_pages"], pools["k_c"].shape[1]
    m = P // 128
    planes = -(-d_v // 128)
    sfv_flat = pools["sfv_g"].view(-1)
    groups = num_pages * KH * m
    for tbl, data, sf, is_v in ((pools["bt_k"], pools["k_c"], pools["sfk_g"], False), (pools["bt_v"], pools["v_c"], pools["sfv_g"], True)):
        tbl2 = tbl.squeeze(1).squeeze(-1)
        for b, L in enumerate(lens):
            for page in tbl2[b, -(-L // P) :].tolist():
                data[page] = float("nan")
                if is_v:  # V SF is plane-major across the pool: group = (page*KH + head)*m + tile
                    for plane in range(planes):
                        g0 = (page * KH) * m
                        sfv_flat[(plane * groups + g0) * 512 : (plane * groups + g0 + KH * m) * 512] = 0xFF
                else:
                    sf[page] = 0xFF


def _build(
    B,
    H,
    KH,
    P,
    max_pages,
    lens,
    hnd,
    *,
    in_key="e4m3",
    out_dt=torch.bfloat16,
    stats=False,
    s_q=1,
    causal_br=False,
    window_left=None,
    sink=False,
    separate_v=False,
    batch_inner=False,
    max_seq_len="default",
    d_qk=D,
    d_v=D,
    poison_dead_pages=True,
    q_bhsd=False,
    mixed_layout=False,
):
    """-> (graph, variant pack, O, Stats, amax, reference inputs)."""
    import cudnn

    dev = "cuda"
    fp8 = _FP8[in_key]
    scale = 1.0 / math.sqrt(d_qk)
    pools = _pools(B, KH, P, max_pages, hnd, fp8, d_qk=d_qk, d_v=d_v, separate_v=separate_v, batch_inner=batch_inner)
    if mixed_layout:  # HND K pool next to an NHD V pool
        pools["v_c"] = pools["v_c"].permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3) if hnd else pools["v_c"].contiguous()
    if poison_dead_pages:
        _poison_dead_pages(pools, lens, P, d_v)
    Qf = torch.randn(B, H, s_q, d_qk, device=dev) * 0.5
    Q8, sfq, dqq, sfq_dims = _quantize(Qf, B, H, s_q, d_qk, fp8, columnwise=False)
    Qb = Q8.contiguous() if q_bhsd else Q8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)
    Ob = torch.empty(B, s_q, H, d_v, device=dev, dtype=out_dt).transpose(1, 2)
    sfq_g = sfq.view(torch.uint8).reshape(sfq_dims)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    slk = seq_lens.view(B, 1, 1, 1)
    slq = torch.full((B, 1, 1, 1), s_q, dtype=torch.int32, device=dev)
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    itype = getattr(cudnn.data_type, _CUDNN_ITYPE[in_key])
    g = cudnn.pygraph(io_data_type=itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(Qb), g.tensor_like(pools["k_c"]), g.tensor_like(pools["v_c"])

    def _sf(dims):
        return g.tensor(
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

    dq, dk, dv = _sf(sfq_dims), _sf(pools["sfk_g"].shape), _sf(pools["sfv_g"].shape)
    tk, tv = g.tensor_like(pools["bt_k"]), g.tensor_like(pools["bt_v"])
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    sinks = torch.randn(1, H, 1, 1, device=dev) if sink else None
    kw = {}
    if sink:
        kw["sink_token"] = g.tensor_like(sinks)
    if window_left is not None:
        kw["diagonal_band_left_bound"] = window_left
    if max_seq_len is not None:
        kw["paged_attention_max_seq_len_kv"] = max_pages * P if max_seq_len == "default" else max_seq_len
    kw.update({"seq_len_q": sq_t, "seq_len_kv": sk_t})
    o, st, amax_o = g.sdpa_mxfp8(
        q=q,
        k=k,
        v=v,
        descale_q=dq,
        descale_k=dk,
        descale_v=dv,
        attn_scale=scale,
        generate_stats=stats,
        use_causal_mask_bottom_right=causal_br,
        use_padding_mask=True,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        **kw,
    )
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(list(stats_gpu.shape)).set_stride(list(stats_gpu.stride())).set_data_type(cudnn.data_type.FLOAT)
    vp = {
        q: Qb,
        k: pools["k_c"],
        v: pools["v_c"],
        dq: sfq_g,
        dk: pools["sfk_g"],
        dv: pools["sfv_g"],
        tk: pools["bt_k"],
        tv: pools["bt_v"],
        sq_t: slq,
        sk_t: slk,
        o: Ob,
        amax_o: amax,
    }
    if stats:
        vp[st] = stats_gpu
    if sink:
        vp[kw["sink_token"]] = sinks
    ref_in = dict(
        qd=Q8.float() * dqq,
        kd_pool=pools["kd"],
        vd_pool=pools["vd"],
        bt_k=pools["bt_k"].squeeze(1).squeeze(-1),
        bt_v=pools["bt_v"].squeeze(1).squeeze(-1),
        seq_lens=seq_lens,
        scale=scale,
        causal_br=causal_br,
        window_left=window_left,
        sinks=sinks,
    )
    return g, vp, Ob, stats_gpu, amax, ref_in


def _check_o(out, ref_o, out_dt, in_key):
    atol = 7e-2 if in_key == "e5m2" else 5e-2
    if out_dt in (torch.float8_e4m3fn, torch.float8_e5m2):  # gauge against the fp8 output-quantization floor
        floor = (ref_o - ref_o.to(out_dt).float()).abs().max().item()
        atol = max(atol, 3.0 * floor)
    torch.testing.assert_close(out, ref_o, atol=atol, rtol=0)
    return atol


def _run_graph(B, H, KH, P, max_pages, lens, hnd, *, want_split=None, **build_kw):
    import cudnn
    import cudnn.sdpa  # noqa: F401 — registers the FROST engines
    from cudnn.sdpa.fwd.engines import engine_name

    g, vp, Ob, stats_gpu, amax, ref_in = _build(B, H, KH, P, max_pages, lens, hnd, **build_kw)
    name = engine_name(mxfp8=True)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, name)
    if want_split is not None:  # select a split plan explicitly: correctness does not ride the heuristic ranking
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        wanted = (lambda s: s > 1) if want_split is True else (lambda s: s == want_split)
        idx = next((i for i, n in enumerate(names) if n.startswith(name) and wanted(g.plans[i].knobs.split_kv)), None)
        assert idx is not None, f"no {name} plan with split_kv={want_split}; knobs={[p.knobs for p in g.plans]}"
        g.select_plan(idx)
        plan = g.plans[idx]
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    # Rule 3: any blocking D2H in the FROST execute path is a bug.
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    ref_o, ref_lse = _ref(**ref_in)
    out = Ob.float()
    assert not torch.isnan(out).any(), "NaN in O"
    atol = _check_o(out, ref_o, build_kw.get("out_dt", torch.bfloat16), build_kw.get("in_key", "e4m3"))
    assert abs(amax.item() - ref_o.abs().max().item()) <= atol, f"amax {amax.item()} vs ref {ref_o.abs().max().item()}"
    live = ref_in["seq_lens"] > 0
    if (~live).any():
        assert out[~live].abs().max().item() == 0.0, "empty sequence must write O := 0"
    if stats_gpu is not None:
        # Same Stats tolerance as the dense MXFP8 suite (fp8 P / exp2 emulation).
        got_lse = stats_gpu.view(B, H, -1)
        torch.testing.assert_close(got_lse[live], ref_lse[live], atol=atol, rtol=3e-2)
        if (~live).any():
            assert torch.isinf(got_lse[~live]).all() and (got_lse[~live] < 0).all(), "empty sequence must write LSE := -inf"
    return plan


# --- decode ------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page_size", [128, 256])
def test_paged_mxfp8_graph_matches_reference(hnd, page_size):
    """Decode: mixed lengths incl. 0 and 1, GQA 8:2, tile-unaligned tails, Stats."""
    _run_graph(5, 8, 2, page_size, -(-1100 // page_size), [300, 77, 0, 1, 1024], hnd, stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("window_left", [128, 33], ids=["w128", "w33"])
def test_paged_mxfp8_graph_sliding_window_causal_br(window_left):
    _run_graph(3, 8, 2, 128, 8, [1000, 40, 513], False, s_q=4, causal_br=True, window_left=window_left, stats=True)


@pytest.mark.L0
def test_paged_mxfp8_graph_separate_v_table():
    """K and V pools behind different block tables (the graph's two-table contract)."""
    _run_graph(3, 8, 2, 128, 6, [500, 130, 700], False, stats=True, separate_v=True)


@pytest.mark.L0
def test_paged_mxfp8_graph_batch_innermost_block_table():
    """Block tables declared batch-innermost (strides (1, ., B, .)) bind as views."""
    _run_graph(3, 8, 2, 128, 6, [500, 130, 700], False, batch_inner=True)


@pytest.mark.L0
def test_paged_mxfp8_graph_all_empty_batch():
    """Every sequence empty: O := 0 and LSE := -inf, nothing is loaded."""
    _run_graph(3, 8, 2, 128, 4, [0, 0, 0], False, stats=True)


@pytest.mark.L0
def test_paged_mxfp8_graph_declared_max_seq_len_below_table_reach():
    _run_graph(2, 8, 2, 128, 16, [1000, 1500], False, max_seq_len=1536)


# --- prefill-shaped -----------------------------------------------------------


@pytest.mark.L0
def test_paged_mxfp8_graph_prefill_shaped_s_q():
    """Chunked prefill: S_q > 1 with the chunk at the end of each sequence (bottom-right causal)."""
    _run_graph(2, 4, 2, 128, 8, [700, 300], False, s_q=64, causal_br=True, stats=True)


# --- split-KV / scale ---------------------------------------------------------


@pytest.mark.L0
def test_paged_mxfp8_graph_split_kv():
    """A KV split over pools (8k tokens, one KV head), selected explicitly: the fp32 partials recombine through the shared combine."""
    plan = _run_graph(1, 4, 1, 128, 64, [8000], False, stats=True, want_split=True)
    assert plan.knobs.split_kv > 1


# --- direct adapter: CUDA-graph replay ---------------------------------------


@pytest.mark.L0
def test_paged_mxfp8_adapter_cuda_graph_replay_no_host_sync():
    """Captured once at fixed B; seq_lens CONTENT changes between replays. The
    sync-debug mode is armed INSIDE the capture (torch's own capture_begin/end synchronize)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, max_pages = 8, 16, 4, 128, 16
    dev = "cuda"
    fp8 = _FP8["e4m3"]
    pools = _pools(B, KH, P, max_pages, False, fp8, seed=1)
    Qf = torch.randn(B, H, 1, D, device=dev) * 0.5
    Q8, sfq, dqq, sfq_dims = _quantize(Qf, B, H, 1, D, fp8, columnwise=False)
    Qb = Q8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)
    sfq_g = sfq.view(torch.uint8).reshape(sfq_dims)
    Ob = torch.empty(B, 1, H, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    lse = torch.empty(B, H, 1, device=dev, dtype=torch.float32)
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
    seq_lens = torch.full((B,), 1000, dtype=torch.int32, device=dev)
    seq_q = torch.ones(B, dtype=torch.int32, device=dev)
    bt = pools["bt_k"].view(B, max_pages)
    api = SdpaFwdDslSm100(
        sample_q=Qb,
        sample_k=pools["k_c"],
        sample_v=pools["v_c"],
        sample_o=Ob,
        sample_lse=lse,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        dtype_o=torch.bfloat16,
        paged_page_size=P,
        paged_max_seq_len_kv=max_pages * P,
        split_kv=4,
        sample_amax_o=amax,
    )
    api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)

    def run():
        api.execute(
            Qb,
            pools["k_c"],
            pools["v_c"],
            Ob,
            lse_tensor=lse,
            seq_kv_lens=seq_lens,
            seq_q_lens=seq_q,
            sf_q=sfq_g,
            sf_k=pools["sfk_g"],
            sf_v=pools["sfv_g"],
            amax_o=amax,
            block_table=bt,
            workspace=ws,
        )

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run()
    torch.cuda.synchronize()
    cg = torch.cuda.CUDAGraph()
    prev = torch.cuda.get_sync_debug_mode()
    with torch.cuda.graph(cg, stream=s):
        torch.cuda.set_sync_debug_mode("error")
        try:
            run()
        finally:
            torch.cuda.set_sync_debug_mode(prev)
    scale = 1.0 / math.sqrt(D)
    for new_lens in ([5, 1024, 77, 128, 129, 1, 512, 1000], [2048] * B, [0, 1, 2, 3, 4, 5, 6, 7]):
        seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        cg.replay()
        torch.cuda.synchronize()
        ref_o, ref_lse = _ref(Q8.float() * dqq, pools["kd"], pools["vd"], bt, bt, seq_lens, scale)
        live = seq_lens > 0
        torch.testing.assert_close(Ob.float(), ref_o, atol=5e-2, rtol=0)
        torch.testing.assert_close(lse.view(B, H, 1)[live], ref_lse[live], atol=5e-2, rtol=3e-2)


@pytest.mark.L0
def test_paged_mxfp8_graph_e5m2_sink_swa():
    """The paged loader composes with the loader-independent features (E5M2 in, fp8 out, sink, sliding window)."""
    _run_graph(3, 8, 2, 128, 9, [1000, 1, 513], False, in_key="e5m2", out_dt=torch.float8_e4m3fn, stats=True, sink=True, window_left=300)


@pytest.mark.L0
@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((128, 128), id="d128"),
        pytest.param((192, 128), id="d192x128"),
        pytest.param((512, 512), id="d512"),
    ],
)
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
def test_paged_mxfp8_graph_other_flavors(dims, hnd):
    d_qk, d_v = dims
    _run_graph(3, 8, 2, 128, 6, [700, 2, 129], hnd, d_qk=d_qk, d_v=d_v, stats=True, s_q=2, causal_br=True)  # L >= s_q: no keyless rows


@pytest.mark.L0
def test_paged_mxfp8_adapter_declines_sm107_device(monkeypatch):
    """check_support declines paged MXFP8 KV on a cc10.7 device (no SM107 kernel carries
    PAGED_KV); the same adapter accepts the graph on the real SM100 device."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, max_pages = 2, 8, 2, 128, 4
    dev = "cuda"
    pools = _pools(B, KH, P, max_pages, False, _FP8["e4m3"])
    Qb = torch.zeros(B, 1, H, D, device=dev, dtype=torch.float8_e4m3fn).transpose(1, 2)
    Ob = torch.empty(B, 1, H, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    lse = torch.empty(B, H, 1, device=dev, dtype=torch.float32)

    def _api():
        return SdpaFwdDslSm100(
            sample_q=Qb,
            sample_k=pools["k_c"],
            sample_v=pools["v_c"],
            sample_o=Ob,
            sample_lse=lse,
            seq_kv_lens_present=True,
            seq_q_lens_present=True,
            dtype_o=torch.bfloat16,
            paged_page_size=P,
            paged_max_seq_len_kv=max_pages * P,
        )

    _api().check_support()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 7))
    with pytest.raises(NotImplementedError, match="SM107"):
        _api().check_support()


@pytest.mark.L0
def test_paged_mxfp8_adapter_rejects_unequal_table_extents():
    """Direct API: K and V block tables of different page-axis extents are declined by name in
    execute() (the kernel compiles both tables on one dynamic extent and reads its KV maximum
    from the K table); equal extents execute. The graph path declines the same mismatch in
    graph_analyzer."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, max_pages = 2, 8, 2, 128, 4
    dev = "cuda"
    fp8 = _FP8["e4m3"]
    pools = _pools(B, KH, P, max_pages, False, fp8, seed=1)
    Q8, sfq, _, sfq_dims = _quantize(torch.randn(B, H, 1, D, device=dev) * 0.5, B, H, 1, D, fp8, columnwise=False)
    Qb = Q8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)
    Ob = torch.empty(B, 1, H, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    lse = torch.empty(B, H, 1, device=dev, dtype=torch.float32)
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
    bt = pools["bt_k"].view(B, max_pages)
    api = SdpaFwdDslSm100(
        sample_q=Qb,
        sample_k=pools["k_c"],
        sample_v=pools["v_c"],
        sample_o=Ob,
        sample_lse=lse,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        dtype_o=torch.bfloat16,
        paged_page_size=P,
        paged_max_seq_len_kv=max_pages * P,
    )
    api.check_support()
    api.compile()
    ex = dict(
        lse_tensor=lse,
        seq_kv_lens=torch.full((B,), 300, dtype=torch.int32, device=dev),
        seq_q_lens=torch.ones(B, dtype=torch.int32, device=dev),
        sf_q=sfq.view(torch.uint8).reshape(sfq_dims),
        sf_k=pools["sfk_g"],
        sf_v=pools["sfv_g"],
        amax_o=amax,
        workspace=torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8),
    )
    wider = torch.cat([bt, bt[:, :2]], dim=1).contiguous()  # a V table two pages wider, otherwise valid
    with pytest.raises(ValueError, match="same page-axis extent"):
        api.execute(Qb, pools["k_c"], pools["v_c"], Ob, block_table=bt, block_table_v=wider, **ex)
    api.execute(Qb, pools["k_c"], pools["v_c"], Ob, block_table=bt, block_table_v=bt.clone(), **ex)
    torch.cuda.synchronize()
    assert not torch.isnan(Ob.float()).any()


# --- declines -----------------------------------------------------------------


@pytest.mark.L0
def test_paged_mxfp8_graph_declines_off_contract():
    """Plan-time declines stay plan-time: no MXFP8 engine plan is offered, nothing compiles."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    def _plans(P, **kw):
        g, *_ = _build(2, 8, 2, P, 4, [100, 200], False, **kw)
        try:
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A])
        except (cudnn.cudnnGraphNotSupportedError, ValueError):
            return None
        return g

    def _offers(g):
        return g is not None and offers_engine(g, engine_name(mxfp8=True))

    assert _offers(_plans(128))
    assert not _offers(_plans(64)), "F8_128x4 pools hold whole 128-row SF atoms: page_size 64 is off-contract"
    assert not _offers(_plans(128, d_qk=64, d_v=64)), "MXFP8 is exact-native: no d64 envelope"
    assert not _offers(_plans(128, q_bhsd=True, s_q=4)), "the MXFP8 row serves BSHD-physical Q/O only (at s_q == 1 the layouts coincide)"
    assert not _offers(_plans(128, mixed_layout=True)), "K and V pools must share an in-page layout"
