# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged KV caches on the FROST SM100 f16/bf16 engine (issue #920).

The graph is cuDNN's own paged-cache contract — ``graph.sdpa(..., k=<page pool>,
v=<page pool>, use_padding_mask=True, seq_len_q=, seq_len_kv=,
paged_attention_k_table=, paged_attention_v_table=,
paged_attention_max_seq_len_kv=)`` — served by ``sdpa_fwd_prefill_sm100``
through the d128 kernel's ``PAGED_KV`` specialization: block-table indirection on
the K/V TMA loads, HND and NHD page layouts, per-batch lengths read on device,
KV split + combine.  The reference gathers each sequence's pages in torch and
runs fp32 attention over the live tokens.

The second half drives the kernel template directly (like the split-KV suite)
for the geometry the heuristic would not pick on its own: forced splits with
empty ranges, cga1, page sizes across the 128-row tile.
"""

import math
import os

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]

D = 128


def _ref_rows(q, k_pool, v_pool, block_table, seq_lens, hnd, scale, *, causal_br=False, window=None):
    """q [B, H, S_q, D], every row live (seq_len_q == S_q); returns (O [B, H, S_q, D_v],
    LSE [B, H, S_q]) fp32. Bottom-right causal anchors row r of batch b at key
    L_b - S_q + r; ``window`` is cuDNN's ``sliding_window_length``: the ``window``
    keys ending at the diagonal. A row without a live key is O := 0 / LSE := -inf."""
    B, H, S, d = q.shape
    KH = k_pool.shape[1] if hnd else k_pool.shape[2]
    P = k_pool.shape[2] if hnd else k_pool.shape[1]
    Dv = v_pool.shape[-1]
    out = torch.zeros(B, H, S, Dv, device=q.device, dtype=torch.float32)
    lse = torch.full((B, H, S), float("-inf"), device=q.device, dtype=torch.float32)
    for b, L in enumerate(seq_lens.tolist()):
        if L == 0:
            continue
        pages = block_table[b, : (L + P - 1) // P].long()
        k, v = k_pool[pages], v_pool[pages]
        if hnd:
            k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        k = k.reshape(-1, KH, d)[:L].repeat_interleave(H // KH, dim=1).float()
        v = v.reshape(-1, KH, Dv)[:L].repeat_interleave(H // KH, dim=1).float()
        s = torch.einsum("hrd,lhd->hrl", q[b].float(), k) * scale
        rows = torch.arange(S, device=q.device).view(S, 1)
        cols = torch.arange(L, device=q.device).view(1, L)
        diag = rows + (L - S)
        masked = torch.zeros(S, L, dtype=torch.bool, device=q.device)
        if causal_br:
            masked |= cols > diag
        if window is not None:
            masked |= cols < diag - (window - 1)
        s = s.masked_fill(masked.view(1, S, L), float("-inf"))
        out[b] = torch.einsum("hrl,lhd->hrd", torch.softmax(s, -1).nan_to_num(0.0), v)
        lse[b] = torch.logsumexp(s, -1)  # keyless rows -> -inf
    return out, lse


def _ref(q, k_pool, v_pool, block_table, seq_lens, hnd, scale):
    """q [B, H, D] (one query row); returns (O [B, H, D_v], LSE [B, H]) fp32."""
    out, lse = _ref_rows(q.unsqueeze(2), k_pool, v_pool, block_table, seq_lens, hnd, scale)
    return out[:, :, 0], lse[:, :, 0]


def _pools(B, KH, d, P, max_pages, hnd, dtype, seed=0):
    """Page pools + a scattered block table.  Returns (k_pool, v_pool, k_container,
    v_container, block_table[B, 1, max_pages, 1]) — the containers are the
    [num_pages, H_kv, page_size, D]-dim views the graph declares."""
    torch.manual_seed(seed)
    dev = "cuda"
    num_pages = B * max_pages + 5
    if hnd:
        k_pool = torch.randn(num_pages, KH, P, d, device=dev, dtype=dtype)
        v_pool = torch.randn(num_pages, KH, P, d, device=dev, dtype=dtype)
        k_c, v_c = k_pool, v_pool
    else:
        k_pool = torch.randn(num_pages, P, KH, d, device=dev, dtype=dtype)
        v_pool = torch.randn(num_pages, P, KH, d, device=dev, dtype=dtype)
        k_c, v_c = k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3)
    bt = torch.randperm(num_pages, device=dev)[: B * max_pages].to(torch.int32).view(B, 1, max_pages, 1).contiguous()
    return k_pool, v_pool, k_c, v_c, bt


def _run_graph(
    B,
    H,
    KH,
    d,
    P,
    max_pages,
    lens,
    hnd,
    *,
    dtype=torch.float16,
    stats=False,
    s_q=1,
    max_seq_len=None,
    want_split=None,
    causal_br=False,
    window=None,
    want_cga=None,
    pack_gqa=None,
    lead_split=None,
):
    """Build, pin the FROST engine (``pack_gqa`` pins that leg), run under the
    sync-debug guard and check every Q row against the fp32 gather reference.
    ``causal_br`` / ``window`` add the bottom-right causal band and cuDNN's
    ``sliding_window_length``; ``want_cga`` asserts the cluster width of the
    plan run and ``lead_split`` the KV split the heuristics LED with (checked
    before any ``want_split`` re-selection). Returns the pinned plan."""
    import cudnn
    import cudnn.sdpa  # noqa: F401 — registers the FROST engines
    from cudnn.sdpa.fwd.engines import engine_name

    dev = "cuda"
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, k_c, v_c, bt = _pools(B, KH, d, P, max_pages, hnd, dtype)
    q_gpu = torch.randn(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    slk = seq_lens.view(B, 1, 1, 1)
    slq = torch.full((B, 1, 1, 1), s_q, dtype=torch.int32, device=dev)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(q_gpu), g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    mask_kw = {}
    if causal_br:
        mask_kw["use_causal_mask_bottom_right"] = True
    if window is not None:
        mask_kw["sliding_window_length"] = window
    o, st = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=stats,
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=max_seq_len if max_seq_len is not None else max_pages * P,
        **mask_kw,
    )
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, engine_name(), pack_gqa=pack_gqa)
    if lead_split is not None:
        assert plan.knobs.split_kv == lead_split, f"expected the heuristics to lead with split_kv={lead_split}; got {plan.knobs}"
    if want_split is not None:
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        idx = next((i for i, n in enumerate(names) if n.startswith(engine_name()) and g.plans[i].knobs.split_kv == want_split), None)
        assert idx is not None, f"no {engine_name()} plan with split_kv={want_split}; knobs={[p.knobs for p in g.plans]}"
        g.select_plan(idx)
        plan = g.plans[idx]
    if want_cga is not None:
        assert plan.knobs.cga == want_cga, f"expected the heuristics to lead with cga={want_cga}; got {plan.knobs}"
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    vp = {q: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: slk, o: o_gpu}
    if stats:
        vp[st] = stats_gpu
    # Rule 3: the FROST execute path reads the per-batch lengths on device;
    # any blocking D2H here is a bug, not a slow path.
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    ref_o, ref_lse = _ref_rows(q_gpu, k_pool, v_pool, bt.view(B, max_pages), seq_lens, hnd, scale, causal_br=causal_br, window=window)
    out = o_gpu.float()
    assert not torch.isnan(out).any(), "NaN in O"
    torch.testing.assert_close(out, ref_o, atol=2e-2 if dtype == torch.float16 else 1e-1, rtol=0)
    # Rows without a live key (empty sequence, or a bottom-right row whose whole
    # band falls before key 0): O := 0 / LSE := -inf.
    live = ~torch.isinf(ref_lse)
    if (~live).any():
        assert out[~live].abs().max().item() == 0.0, "a keyless row must write O := 0"
    if stats:
        got_lse = stats_gpu.view(B, H, s_q)
        torch.testing.assert_close(got_lse[live], ref_lse[live], atol=5e-3, rtol=0)
        if (~live).any():
            assert torch.isinf(got_lse[~live]).all() and (got_lse[~live] < 0).all(), "a keyless row must write LSE := -inf"
    return plan


# --- graph API ---------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 128])
def test_paged_graph_matches_reference(hnd, page_size):
    """Mixed lengths incl. 0 and 1, GQA 8:2 (PackGQA), a tile-unaligned tail and a
    length ending exactly on a page/tile boundary; Stats requested."""
    _run_graph(5, 8, 2, D, page_size, -(-1100 // page_size), [300, 77, 0, 1, 1024], hnd, stats=True)


@pytest.mark.L0
def test_paged_graph_bf16_mha_no_stats():
    _run_graph(3, 8, 8, D, 32, 40, [1000, 1279, 33], hnd=False, dtype=torch.bfloat16)


@pytest.mark.L0
def test_paged_graph_long_kv_heuristic_splits():
    """32k tokens at B=1: a 1-CTA-per-KV-head launch — the split heuristic must
    engage (the padded exclusion is lifted for paged graphs) and the recombined
    LSE must match."""
    plan = _run_graph(1, 8, 1, D, 16, 2048, [32000], hnd=True, stats=True)
    assert plan.knobs.split_kv > 1, plan.knobs


@pytest.mark.L0
@pytest.mark.parametrize("page_size", [16, 128])
def test_paged_graph_ragged_declared_max_heuristic_splits(page_size):
    """FlashInfer's decode spelling: no mask, ``paged_attention_max_seq_len_kv``
    = the caller's TRUE max (4000, not a multiple of the 128-row KV tile),
    tables padded past it. B=2, H_kv=1 is a 4-CTA launch — the split heuristic
    must engage exactly as it does for a 128-multiple max (a paged graph never
    rides the synthesized KV-tail padding that excludes the split on a dense
    mask-free graph), and the recombined O / LSE must match the reference."""
    max_pages = -(-4096 // page_size)
    plan = _run_graph(2, 8, 1, D, page_size, max_pages, [4000, 3000], hnd=True, stats=True, max_seq_len=4000)
    assert plan.knobs.split_kv > 1, plan.knobs


@pytest.mark.L0
def test_paged_graph_declared_max_seq_len_below_table_reach():
    """paged_attention_max_seq_len_kv smaller than max_pages * page_size is the
    common framework case (tables padded to the model's max length)."""
    _run_graph(2, 8, 2, D, 16, 20, [300, 77], hnd=False, max_seq_len=300, stats=True)


@pytest.mark.L0
def test_paged_graph_d64_envelope_large_batch():
    """d=64 rides the d128 kernel zero-padded; B=256 GQA 4:4."""
    torch.manual_seed(3)
    lens = torch.randint(1, 257, (256,)).tolist()
    _run_graph(256, 4, 4, 64, 64, 4, lens, hnd=False)


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
def test_paged_graph_d256(hnd):
    """d=256 selects the d256 f16 flavor; same graph contract, Stats out."""
    _run_graph(3, 8, 2, 256, 32, 40, [1000, 1, 1279], hnd, stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("h,kh", [(96, 8), (48, 8)], ids=["g12_packs4", "g6_packs2"])
@pytest.mark.parametrize("s_q", [1, 2, 4, 8])
def test_paged_graph_partial_pack_gqa(hnd, h, kh, s_q):
    """Partial PackGQA: a GQA group that does not divide the 128-row Q tile packs
    its largest divisor that does (G=12 -> 4 heads per token row-group, three
    packed heads per KV head; G=6 -> 2), and the heuristics rank that packed plan
    first on these decode / MTP shapes -- the 96/8 d128 paged decode that ran
    unpacked before (one live row per 512-row cluster, 12x KV re-read).  Bottom-
    right causal at S_q > 1 with Stats out: the packed epilogue scatters LSE
    over the p head rows, the mask predicate is per token, and the KV head is
    packed head // (G / p); a slip in any of the three shows up here.  Lengths
    include a 1-token sequence, so at S_q = 8 seven of its rows are dead."""
    plan = _run_graph(4, h, kh, D, 16, -(-1100 // 16), [300, 77, 1, 1100], hnd, s_q=s_q, causal_br=s_q > 1, stats=True)
    assert plan.knobs.pack_gqa is True, plan.knobs


@pytest.mark.L0
def test_paged_graph_prefill_shaped_s_q():
    """The same engine serves S_q > 1 over a paged cache (paged prefill / chunked
    prefill); all q rows attend the whole live KV (no causal mask)."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    B, H, KH, P, max_pages, s_q = 2, 4, 4, 16, 8, 3
    dev, dtype = "cuda", torch.float16
    scale = 1.0 / math.sqrt(D)
    k_pool, v_pool, k_c, v_c, bt = _pools(B, KH, D, P, max_pages, False, dtype)
    q_gpu = torch.randn(B, s_q, H, D, device=dev, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(B, s_q, H, D, device=dev, dtype=dtype).transpose(1, 2)
    lens = torch.tensor([100, 128], dtype=torch.int32, device=dev)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v, tk, tv = g.tensor_like(q_gpu), g.tensor_like(k_c), g.tensor_like(v_c), g.tensor_like(bt), g.tensor_like(bt)
    slq = torch.full((B, 1, 1, 1), s_q, dtype=torch.int32, device=dev)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(lens.view(B, 1, 1, 1))
    o, _ = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=max_pages * P,
    )
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine_name())
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: lens.view(B, 1, 1, 1), o: o_gpu}, ws)
    torch.cuda.synchronize()
    for r in range(s_q):
        ref_o, _ = _ref(q_gpu[:, :, r, :], k_pool, v_pool, bt.view(B, max_pages), lens, False, scale)
        torch.testing.assert_close(o_gpu[:, :, r, :].float(), ref_o, atol=2e-2, rtol=0)


@pytest.mark.L0
def test_paged_graph_declines_off_contract():
    """Plan-time declines stay plan-time: no engine plan is offered, nothing compiles."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name
    from frost_test_utils import offers_engine

    def _build(P, d=D, H=8, KH=2, hnd=False, padding=True):
        B, max_pages = 2, 8
        dev = "cuda"
        _, _, k_c, v_c, bt = _pools(B, KH, d, P, max_pages, hnd, torch.float16)
        q_gpu = torch.randn(B, 1, H, d, device=dev, dtype=torch.float16).transpose(1, 2)
        g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        q, k, v, tk, tv = g.tensor_like(q_gpu), g.tensor_like(k_c), g.tensor_like(v_c), g.tensor_like(bt), g.tensor_like(bt)
        lens = torch.full((B, 1, 1, 1), 10, dtype=torch.int32, device=dev)
        sq_t, sk_t = g.tensor_like(lens), g.tensor_like(lens)
        o, _ = g.sdpa(
            name="sdpa",
            q=q,
            k=k,
            v=v,
            generate_stats=False,
            attn_scale=0.1,
            use_padding_mask=padding,
            seq_len_q=sq_t,
            seq_len_kv=sk_t,
            paged_attention_k_table=tk,
            paged_attention_v_table=tv,
            paged_attention_max_seq_len_kv=max_pages * P,
        )
        o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
        try:
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A])
        except (cudnn.cudnnGraphNotSupportedError, ValueError):
            # Rejected upstream of any engine (the validator: paged needs the
            # padding mask; the backend: page size must be a power of two), so
            # no plan of any kind exists — the FROST row certainly offered none.
            return None
        return g

    def _offers(g):
        return g is not None and offers_engine(g, engine_name())

    assert _offers(_build(16))
    assert not _offers(_build(48)), "page_size 48 neither divides nor is a multiple of the 128-row tile"
    assert not _offers(_build(16, d=512)), "paged KV is wired on the d128 / d256 flavors only"
    assert not _offers(_build(16, padding=False)), "paged KV requires the padding mask"


# --- decode-shaped launches: cga1 on the graph path ---------------------------
#
# S_q * G <= 256 live Q rows per (batch, packed head) unit: the heuristics lead
# with cga1 -- one CTA per unit, the kernel's QO-alias configuration -- on the
# plain scheduler (heuristics._auto_sched_cga / _sched_points).  These are the
# graph-path runs of d128 cga1 under paged loads, PackGQA and the bottom-right /
# sliding-window masks; the kernel-level tests below drove cga1 with splits only.


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_paged_graph_decode_shaped_leads_with_cga1(hnd, page_size):
    """FlashInfer-shaped decode (GQA 32:2, S_q=1, mixed lengths incl. 0 / 1 / a
    page boundary / a tile boundary + 1): the lead plan is cga1 + PackGQA on the
    plain scheduler, Stats out."""
    plan = _run_graph(8, 32, 2, D, page_size, -(-4096 // page_size), [4096, 77, 0, 1, 1024, 2000, 4095, 129], hnd, stats=True, want_cga=1)
    assert plan.knobs.pack_gqa is True and plan.knobs.sched_policy == 0, plan.knobs


@pytest.mark.L0
@pytest.mark.parametrize("pack_gqa", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize("s_q,window", [(2, None), (4, None), (8, 300), (8, 1)], ids=["mtp2", "mtp4", "mtp8_swa300", "mtp8_swa1"])
def test_paged_graph_mtp_bottom_right_cga1(s_q, window, pack_gqa):
    """MTP: S_q in [2, 8] bottom-right causal (+ sliding window) over the paged
    cache, GQA 32:4, lengths below S_q included (keyless rows are O := 0 /
    LSE := -inf). Both PackGQA legs run at cga1 on the plain scheduler."""
    plan = _run_graph(
        6, 32, 4, D, 16, 256, [4096, 3, 0, 1, 1000, 4095], hnd=False, stats=True, s_q=s_q, causal_br=True, window=window, want_cga=1, pack_gqa=pack_gqa
    )
    assert plan.knobs.pack_gqa is pack_gqa and plan.knobs.sched_policy == 0, plan.knobs


@pytest.mark.L0
def test_paged_graph_group_not_dividing_tile_runs_unpacked_cga1():
    """H/H_kv = 12 (a 96/8-style group) cannot pack a 128-row tile: each head is
    its own one-live-row unit -- still one CTA's worth, so cga1 leads unpacked."""
    plan = _run_graph(4, 24, 2, D, 16, 64, [1000, 1, 0, 1024], hnd=True, dtype=torch.bfloat16, stats=True, want_cga=1)
    assert plan.knobs.pack_gqa is False, plan.knobs


@pytest.mark.L0
def test_paged_graph_prefill_shaped_keeps_cga2():
    """Past one CTA's 256 rows (S_q=300, MHA) the paged prefill stays on cga2."""
    _run_graph(2, 4, 4, D, 16, 32, [500, 128], hnd=False, s_q=300, want_cga=2)


def _sm_count():
    """SM count of the current device, from the owner the heuristics GPU test
    reads too (test_sdpa_fwd_heuristics's split_kv-by-name case)."""
    from cudnn._device import device_info

    return device_info(torch.cuda.current_device()).sm_count


_B200_SMS = 148  # the part the split choices below were measured on


def _device_lead_split(B, H, KH, s_kv, *, s_q=1, causal_br=False, dtype=torch.float16, page_size=16):
    """The KV split the heuristics lead with for this paged graph ON THIS DEVICE:
    ``recommend`` fed the graph's facts and the device's SM count. A split is a
    wave-count decision -- the 64-CTA launch that idles half of a 148-SM part
    and splits in two fills a 68-SM part and stays unsplit -- so the graph
    tests below assert the lead against the model rather than against one
    part's number (the fixed 148-SM choices stay pinned in
    test_sdpa_fwd_heuristics) and run whatever the device leads with."""
    import cudnn
    from cudnn.sdpa.fwd.engines import engine_name
    from cudnn.sdpa.fwd.heuristics import recommend
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    facts = SdpaGraphFacts(
        b=B,
        h_q=H,
        h_kv=KH,
        s_q=s_q,
        s_kv=s_kv,
        d_qk=D,
        d_v=D,
        dtype=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16,
        causal=causal_br,
        bottom_right=causal_br,
        padded=True,
        has_paged_kv=True,
        page_size=page_size,
        wants_stats=True,
        device_cc=torch.cuda.get_device_capability(),
        device_sm_count=_sm_count(),
    )
    plans = recommend("A", facts, {engine_name(): 0})
    assert plans, "the f16 row must serve this graph"
    return plans[0].knobs.split_kv


@pytest.mark.L0
@pytest.mark.parametrize("s_q,b200_split", [(64, 1), (16, 2)], ids=["chunk64_unsplit", "chunk16_split2"])
def test_paged_graph_small_batch_chunk_splits_only_where_the_combine_is_cheap(s_q, b200_split):
    """b=8, GQA 32:8, bottom-right causal over a 4k paged cache: one CTA's rows
    per unit at cga1 and a launch that leaves half of a B200 idle. There the
    wave model splits the S_q=16 chunk (4096 combine rows; 63.8 -> 49.4 us
    against the pre-change cga2 plan) and leaves the S_q=64 chunk unsplit
    (16384 combine rows; its split 2 measured 75.0 us against 67.6 us) -- the
    ids name those 148-SM choices, pinned when the device is that part. On
    any other part the lead is the model's own for its SM count (a 68-SM part
    fills with the 64 CTAs and leaves both chunks unsplit). Both run under the
    sync-debug guard and check every row, keyless ones included."""
    lead_split = _device_lead_split(8, 32, 8, 4096, s_q=s_q, causal_br=True)
    if _sm_count() == _B200_SMS:
        assert lead_split == b200_split, f"the measured 148-SM choice moved: split_kv={lead_split}"
    plan = _run_graph(
        8, 32, 8, D, 16, 256, [4096, 3000, 77, 0, 1, 4095, 129, 2048], hnd=False, stats=True, s_q=s_q, causal_br=True, want_cga=1, lead_split=lead_split
    )
    assert plan.knobs.pack_gqa is True and plan.knobs.sched_policy == 0, plan.knobs


@pytest.mark.L0
@pytest.mark.parametrize("H,KH,s_kv,b200_split", [(16, 2, 32768, 32), (64, 4, 4096, 8)], ids=["b1_16_2_32k_split32", "b1_64_4_4k_split8"])
def test_paged_graph_few_unit_long_kv_splits_to_the_latency_floor(H, KH, s_kv, b200_split):
    """b=1 decode with two or four KV heads: the whole launch is a handful of
    cga1 CTAs, so the wave model splits the KV loop across the idle SMs -- and
    stops where a finer split's partials cost the lone combine block more than
    the loop saves (choose_split_kv's COMBINE_FLOOR): on a B200, 32 splits over
    the 32k cache (39.7 us; 64 measured 50.8) and 8 over the 4k one (20.3 us;
    16 measured 22.3) -- the ids name those 148-SM choices, pinned when the
    device is that part; on any other part the lead is the model's own for its
    SM count. Runs the lead under the sync-debug guard and checks O and Stats
    against the fp32 gather reference."""
    lead_split = _device_lead_split(1, H, KH, s_kv, dtype=torch.bfloat16)
    if _sm_count() == _B200_SMS:
        assert lead_split == b200_split, f"the measured 148-SM choice moved: split_kv={lead_split}"
    plan = _run_graph(1, H, KH, D, 16, s_kv // 16, [s_kv], hnd=False, dtype=torch.bfloat16, stats=True, want_cga=1, lead_split=lead_split)
    assert plan.knobs.pack_gqa is True and plan.knobs.sched_policy == 0, plan.knobs


# --- kernel template, direct -----------------------------------------------
#
# What the graph path's heuristic would not choose on its own: forced split
# counts with EMPTY ranges (more splits than a short batch has tiles), cga1,
# and page sizes on both sides of the tile.  Mirrors test_sdpa_fwd_split_kv_sm100.


def _run_kernel(B, H, KH, P, max_pages, lens, hnd, splits, *, cta_mma=1, dtype=torch.float16, d=128, expect_pack_g=None):
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import MAKE_CFG, TemplateParams, pack_gqa_group_size
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    dev = "cuda"
    G = H // KH
    # The d128 / d256 f16 kernels pack the largest divisor of G that divides the
    # 128-row tile (partial PackGQA); 1 means nothing packs and the unpacked path serves.
    pack = pack_gqa_group_size(G, 128, partial=True) > 1
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, k_c, v_c, bt4 = _pools(B, KH, d, P, max_pages, hnd, dtype)
    bt = bt4.view(B, max_pages)
    q = torch.randn(B, 1, H, d, device=dev, dtype=dtype)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    # kernel view: [num_pages, page_size, H_kv, d] (a permutation of the container)
    k_view, v_view = k_c.permute(0, 2, 1, 3), v_c.permute(0, 2, 1, 3)

    path = os.path.join(os.path.dirname(os.path.abspath(api_dsl.__file__)), "kernels", "sm100", f"prefill_d{d}_f16.py")
    params = TemplateParams(
        dtype_qkv=3 if dtype == torch.float16 else 2,
        seq_kv_lens_present=True,
        paged_kv=True,
        page_size=P,
        split_kv=splits,
        cta_mma=cta_mma,
        pack_gqa=pack,
        qh_per_kh=G if pack else 1,
    )
    if expect_pack_g is not None:
        cfg, _ = MAKE_CFG[d](params)
        assert cfg.PACK_G == expect_pack_g, f"PACK_G={cfg.PACK_G} for G={G}, expected {expect_pack_g}"
    mod = load_template(path, params, tag=f"paged_p{P}_s{splits}_c{cta_mma}_g{G if pack else 1}_{dtype}")
    # The pools' strides in the kernel's [num_pages, page_size, H_kv, d] order carry the layout (HND vs NHD).
    fn = mod.compile(b=B, qh=H, kh=KH, sq=1, skv=0, d_qk=d, d_v=d, has_lse=True, k_stride=tuple(k_view.stride()), v_stride=tuple(v_view.stride()))
    # Split partials are fp32 on SM100 (#891); the combine must be compiled for that width.
    from test_sdpa_fwd_split_kv_sm100 import _partial_kwargs, _partial_o_dtype, _partial_tag

    o_p = torch.zeros(splits * B, 1, H, d, device=dev, dtype=_partial_o_dtype(splits, dtype))
    lse_p = torch.zeros(splits * B, H, 1, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q,
        k_view,
        v_view,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),
        seq_lens,
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, KH, 1, 0, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        0,  # seq_q_lens_addr: no per-batch Q lengths
        **_partial_kwargs(splits, o_p),
        block_table_tensor=bt,
        block_table_v_tensor=bt,
        stream=stream,
    )
    if splits == 1:
        o_out, lse_out = o_p, lse_p
    else:
        o_out = torch.zeros(B, 1, H, d, device=dev, dtype=dtype)
        lse_out = torch.zeros(B, H, 1, device=dev, dtype=torch.float32)
        cfn = comb.compile(
            b=B, h=H, sq=1, d_v=d, splits=splits, dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=True, dtype_partial=_partial_tag(splits, dtype)
        )
        cfn(o_p, lse_p, o_out, lse_out, None, None, (B, H, 1, d), cutlass.Int32(splits), stream=stream)
    torch.cuda.synchronize()
    ref_o, ref_lse = _ref(q[:, 0], k_pool, v_pool, bt, seq_lens, hnd, scale)
    live = seq_lens > 0
    torch.testing.assert_close(o_out[:, 0].float(), ref_o, atol=2e-2 if dtype == torch.float16 else 1e-1, rtol=0)
    torch.testing.assert_close(lse_out.view(B, H)[live], ref_lse[live], atol=5e-3, rtol=0)


@pytest.mark.L0
@pytest.mark.parametrize("page_size", [32, 64, 256])
def test_paged_kernel_page_sizes(page_size):
    """Pages narrower than the tile (several row boxes per tile) and wider than it
    (one box inside the page, runtime row offset)."""
    _run_kernel(2, 8, 2, page_size, -(-1100 // page_size), [1000, 77], hnd=page_size == 256, splits=2, cta_mma=2)


@pytest.mark.L0
@pytest.mark.parametrize("splits", [2, 8])
def test_paged_kernel_forced_splits_empty_ranges_cga1(splits):
    """3 batches x 8 heads x 8 splits = 192 CTAs > SM count with [4000, 1, 129]:
    empty split ranges interleaved with live tiles under the persistent scheduler
    at cga1 — the Q/O-alias parity regression this PR fixed (see
    test_split_kv_cga1_empty_splits_multiwave for the dense twin)."""
    _run_kernel(3, 8, 8, 128, 33, [4000, 1, 129], hnd=True, splits=splits, cta_mma=1)


@pytest.mark.L0
@pytest.mark.parametrize("page_size", [16, 128])
def test_paged_kernel_d256(page_size):
    """The d256 (Qwen) f16 flavor carries the same PAGED_KV specialization: cga2
    (K box = 64 rows per CTA), forced 4 splits with one empty range."""
    _run_kernel(2, 8, 2, page_size, -(-1100 // page_size), [1000, 77], hnd=page_size == 16, splits=4, cta_mma=2, d=256)


@pytest.mark.L0
@pytest.mark.parametrize(
    "h,kh,pack_g,d",
    [(6, 2, 1, 128), (10, 2, 1, 128), (24, 2, 4, 128), (12, 2, 2, 128), (24, 2, 4, 256), (256, 1, 128, 128)],
    ids=["g3_unpacked", "g5_unpacked", "g12_packs4", "g6_packs2", "g12_packs4_d256", "g256_packs128"],
)
def test_paged_kernel_gqa_group_not_dividing_tile(h, kh, pack_g, d):
    """A GQA group with no factor in common with the 128-row tile (G=3, G=5) runs
    unpacked (PACK_G = 1); one that shares a factor packs its largest divisor of
    the tile (G=12 -> 4 heads per token row-group, G=6 -> 2) on the d128 and d256
    f16 flavors -- partial PackGQA, ``CfgD128.PACK_G`` / ``CfgD256.PACK_G``.  A
    group LARGER than the tile (256/1 MQA) packs the whole tile, 128 heads of one
    token, two packed heads per KV head -- the G=128 geometry plus the
    PACKED_HEADS_PER_KV division; declined before partial packing."""
    _run_kernel(2, h, kh, 16, 8, [50, 128], hnd=False, splits=1, cta_mma=2, d=d, expect_pack_g=pack_g)


@pytest.mark.L0
def test_paged_adapter_cuda_graph_replay_no_host_sync():
    """The adapter's execute path captured once at fixed B; seq_lens CONTENT
    changes between replays.  ``set_sync_debug_mode("error")`` around the
    captured execute makes any blocking D2H raise (python/cudnn/AGENTS.md
    Rule 3).  The mode is armed INSIDE the capture context: torch's own
    ``CUDAGraph.capture_begin`` / ``capture_end`` synchronize the device, and
    arming it outside them raises on torch's sync, not on the adapter's, and
    leaves the stream capturing with no way to end it (an unusable context)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, max_pages = 8, 16, 4, 16, 64
    dev, dtype = "cuda", torch.float16
    k_pool, v_pool, k_c, v_c, bt4 = _pools(B, KH, D, P, max_pages, False, dtype, seed=1)
    bt = bt4.view(B, max_pages)
    q_gpu = torch.randn(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    lse = torch.empty(B, H, 1, device=dev, dtype=torch.float32)
    seq_lens = torch.full((B,), 1000, dtype=torch.int32, device=dev)
    seq_q = torch.ones(B, dtype=torch.int32, device=dev)
    api = SdpaFwdDslSm100(
        sample_q=q_gpu,
        sample_k=k_c,
        sample_v=v_c,
        sample_o=o_gpu,
        sample_lse=lse,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        paged_page_size=P,
        paged_max_seq_len_kv=max_pages * P,
        split_kv=4,
        pack_gqa=True,
    )
    api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    prev_sync_mode = torch.cuda.get_sync_debug_mode()
    with torch.cuda.graph(g, stream=s):
        torch.cuda.set_sync_debug_mode("error")
        try:
            api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
        finally:
            torch.cuda.set_sync_debug_mode(prev_sync_mode)
    scale = 1.0 / math.sqrt(D)
    for new_lens in ([5, 1024, 77, 128, 129, 1, 512, 1000], [1024] * B, [0, 1, 2, 3, 4, 5, 6, 7]):
        seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        ref_o, ref_lse = _ref(q_gpu[:, :, 0, :], k_pool, v_pool, bt, seq_lens, False, scale)
        live = seq_lens > 0
        torch.testing.assert_close(o_gpu[:, :, 0, :].float(), ref_o, atol=2e-2, rtol=0)
        torch.testing.assert_close(lse.view(B, H)[live], ref_lse[live], atol=5e-3, rtol=0)


# --- THD (ragged) queries over a paged cache: chunked prefill -----------------


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
def test_paged_graph_thd_queries(hnd):
    """Ragged Q/O (packed [T, H, D] storage + ragged offsets, per-sequence
    ``seq_len_q``) attending K/V page pools through block tables: each sequence's
    q tokens see its own live KV pages. Only Q/O are ragged — the pools are
    ordinary dense tensors — and the THD scheduler walks the Q units while the
    KV side comes from ``seq_len_kv`` + the tables."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    dev, dtype = "cuda", torch.float16
    H, KH, d, P, max_pages = 8, 2, D, 16, 20
    q_lens = [37, 130, 5]
    kv_lens = [300, 77, 129]
    B, T, S_max = len(q_lens), sum(q_lens), max(q_lens)
    cu = [0]
    for s in q_lens:
        cu.append(cu[-1] + s)
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, k_c, v_c, bt = _pools(B, KH, d, P, max_pages, hnd, dtype)
    torch.manual_seed(7)
    q_pk = torch.randn(T, H, d, device=dev, dtype=dtype)
    stride = (S_max * H * d, d, H * d, 1)
    q_stor = torch.zeros(B * S_max * H * d, device=dev, dtype=dtype)
    q_stor[: T * H * d] = q_pk.reshape(-1)
    q_gpu = q_stor.as_strided((B, H, S_max, d), stride)
    o_stor = torch.zeros(B * S_max * H * d, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H, S_max, d), stride)
    slq = torch.tensor(q_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(kv_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro = (torch.tensor(cu, dtype=torch.int64, device=dev) * H * d).view(B + 1, 1, 1, 1)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, S_max, d], stride=list(stride), data_type=cudnn.data_type.HALF, name="q")
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    qro, oro = g.tensor_like(ro), g.tensor_like(ro)
    tq.set_ragged_offset(qro)
    o, _ = g.sdpa(
        name="sdpa",
        q=tq,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=max_pages * P,
    )
    o.set_output(True).set_dim([B, H, S_max, d]).set_stride(list(stride))
    o.set_ragged_offset(oro)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine_name())
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute({tq: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: slk, qro: ro, oro: ro, o: o_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    o_out = o_stor[: T * H * d].reshape(T, H, d).float()
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    for b in range(B):
        for r in range(q_lens[b]):
            q_row = q_pk[cu[b] + r].unsqueeze(0)  # [1, H, d] -> batch of one
            ref_o, _ = _ref(q_row, k_pool, v_pool, bt.view(B, max_pages)[b : b + 1], kv_lens_t[b : b + 1], hnd, scale)
            torch.testing.assert_close(o_out[cu[b] + r], ref_o[0], atol=2e-2, rtol=0)


@pytest.mark.L0
def test_paged_graph_batch_innermost_block_table():
    """Block tables declared batch-innermost — strides (1, ., B, .) on the
    (B, 1, max_pages, 1) tensor, the layout test_mhas_v2's harness builds — bind
    as views: the table strides are part of the compile key, not a contract."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    dev, dtype = "cuda", torch.float16
    B, H, KH, P, max_pages = 3, 8, 2, 32, 12
    k_pool, v_pool, k_c, v_c, _ = _pools(B, KH, D, P, max_pages, False, dtype)
    num_pages = k_pool.shape[0]
    # linspace(...).reshape(max_pages, 1, B, 1).transpose(0, 2): batch stride 1, page stride B.
    bt = torch.randperm(num_pages, device=dev)[: B * max_pages].to(torch.int32).reshape(max_pages, 1, B, 1).transpose(0, 2)
    assert tuple(bt.stride()) == (1, B, B, 1) or bt.stride()[0] == 1
    q_gpu = torch.randn(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    lens = torch.tensor([300, 1, 384], dtype=torch.int32, device=dev)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v, tk, tv = g.tensor_like(q_gpu), g.tensor_like(k_c), g.tensor_like(v_c), g.tensor_like(bt), g.tensor_like(bt)
    slq = torch.ones(B, 1, 1, 1, dtype=torch.int32, device=dev)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(lens.view(B, 1, 1, 1))
    o, _ = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / math.sqrt(D),
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=max_pages * P,
    )
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine_name())
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: lens.view(B, 1, 1, 1), o: o_gpu}, ws)
    torch.cuda.synchronize()
    # The reference takes a row-major (B, max_pages) table: materialize the same ids.
    ref_o, _ = _ref(q_gpu[:, :, 0, :], k_pool, v_pool, bt.reshape(B, max_pages).contiguous(), lens, False, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o_gpu[:, :, 0, :].float(), ref_o, atol=2e-2, rtol=0)
