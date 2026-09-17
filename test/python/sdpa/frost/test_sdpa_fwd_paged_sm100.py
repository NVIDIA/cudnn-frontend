# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged KV caches on the FROST SM100 f16/bf16 engine (issue #920).

The graph is cuDNN's own paged-cache contract — ``graph.sdpa(..., k=<page pool>,
v=<page pool>, use_padding_mask=True, seq_len_q=, seq_len_kv=,
paged_attention_k_table=, paged_attention_v_table=,
paged_attention_max_seq_len_kv=)`` — served by ``sdpa_fwd_prefill_sm100``
through the d128 / d192x128 / d256 kernels' ``PAGED_KV`` specialization:
block-table indirection on the K/V TMA loads, HND and NHD page layouts,
per-batch lengths read on device, KV split + combine.  The K and V pools may
differ in row width (d192x128: a 192-wide K pool and a 128-wide V pool); mixed
head dims land on the smallest covering flavor.  The reference gathers each
sequence's pages in torch and runs fp32 attention over the live tokens.
Attention sinks and left sliding windows (bottom-right causal, the decode
spelling) ride the same graph: the sink is an epilogue fold and the window only
moves the first KV tile, so neither touches the paged loader -- but the pair had
never been compiled together.

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


def _ref_rows(q, k_pool, v_pool, block_table, seq_lens, hnd, scale, *, sink=None, window_left=None, causal_br=False):
    """q [B, H, S_q, D]; pools in their storage layout; returns (O [B, H, S_q, D_v],
    LSE [B, H, S_q]) fp32 with the kernel's semantics. ``causal_br``: bottom-right
    causal -- row r of batch b has its diagonal at key ``seq_lens[b] - S_q + r``.
    ``window_left``: cuDNN ``diagonal_band_left_bound`` -- visible keys INCLUDING the
    diagonal (FA-style w maps to w + 1). ``sink`` [H] fp32: one extra softmax column
    with V = 0, so a keyless row is O = 0 / LSE = sink (O = 0 / LSE = -inf without one)."""
    B, H, S_q, d = q.shape
    KH = k_pool.shape[1] if hnd else k_pool.shape[2]
    P = k_pool.shape[2] if hnd else k_pool.shape[1]
    Dv = v_pool.shape[-1]
    out = torch.zeros(B, H, S_q, Dv, device=q.device, dtype=torch.float32)
    lse = torch.full((B, H, S_q), float("-inf"), device=q.device, dtype=torch.float32)
    if sink is not None:
        lse[:] = sink.float().view(1, H, 1)
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
        if causal_br or window_left is not None:
            r = torch.arange(S_q, device=q.device).view(S_q, 1)
            c = torch.arange(L, device=q.device).view(1, L)
            diag = L - S_q + r
            masked = torch.zeros(S_q, L, dtype=torch.bool, device=q.device)
            if causal_br:
                masked |= c > diag
            if window_left is not None:
                masked |= c <= diag - window_left
            s = s.masked_fill(masked.unsqueeze(0), float("-inf"))
        if sink is not None:
            s = torch.cat([s, sink.float().view(H, 1, 1).expand(H, S_q, 1)], dim=-1)
            probs = torch.softmax(s, -1)[..., :L]
        else:
            probs = torch.softmax(s, -1).nan_to_num(0.0)
        out[b] = torch.einsum("hrl,lhd->hrd", probs, v)
        lse[b] = torch.logsumexp(s, -1)
    return out, lse


def _ref(q, k_pool, v_pool, block_table, seq_lens, hnd, scale):
    """Decode form of ``_ref_rows``: q [B, H, D] -> (O [B, H, D_v], LSE [B, H]) fp32."""
    out, lse = _ref_rows(q.unsqueeze(2), k_pool, v_pool, block_table, seq_lens, hnd, scale)
    return out[:, :, 0], lse[:, :, 0]


def _pools(B, KH, d, P, max_pages, hnd, dtype, seed=0, d_v=None):
    """Page pools + a scattered block table.  Returns (k_pool, v_pool, k_container,
    v_container, block_table[B, 1, max_pages, 1]) — the containers are the
    [num_pages, H_kv, page_size, D]-dim views the graph declares.  ``d_v``
    (default ``d``) is the V pool's row width: the K and V pools may differ
    (MLA-style d_qk != d_v — the d192x128 flavor's contract)."""
    torch.manual_seed(seed)
    dev = "cuda"
    d_v = d if d_v is None else d_v
    num_pages = B * max_pages + 5
    if hnd:
        k_pool = torch.randn(num_pages, KH, P, d, device=dev, dtype=dtype)
        v_pool = torch.randn(num_pages, KH, P, d_v, device=dev, dtype=dtype)
        k_c, v_c = k_pool, v_pool
    else:
        k_pool = torch.randn(num_pages, P, KH, d, device=dev, dtype=dtype)
        v_pool = torch.randn(num_pages, P, KH, d_v, device=dev, dtype=dtype)
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
    sink=False,
    window_left=None,
    causal_br=False,
    pack_gqa=None,
    d_v=None,
):
    """Build, pin the FROST engine, execute, compare every Q row against ``_ref_rows``.

    ``sink`` binds a (1, H, 1, 1) fp32 ``sink_token``: ``True`` draws one logit per head
    from randn, a number pins every head to that logit (the underflow regression needs
    -120, far below anything randn draws). ``causal_br`` spells the bottom-right causal
    diagonal (``diagonal_band_right_bound=0``, BOTTOM_RIGHT) and ``window_left`` adds
    ``diagonal_band_left_bound`` to it -- the decode spelling FlashInfer uses (a left
    window alone with BOTTOM_RIGHT has no right bound and the row declines it).
    ``pack_gqa`` pins the packed / unpacked plan. ``d_v`` (default ``d``) is the V
    pool's row width: the K and V pools may differ (MLA-style d_qk != d_v -- the
    d192x128 flavor's contract). ``want_split`` pins the split / unsplit leg."""
    import cudnn
    import cudnn.sdpa  # noqa: F401 — registers the FROST engines
    from cudnn.sdpa.fwd.engines import engine_name

    dev = "cuda"
    d_v = d if d_v is None else d_v
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, k_c, v_c, bt = _pools(B, KH, d, P, max_pages, hnd, dtype, d_v=d_v)
    q_gpu = torch.randn(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(B, s_q, H, d_v, device=dev, dtype=dtype).transpose(1, 2)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    slk = seq_lens.view(B, 1, 1, 1)
    slq = torch.full((B, 1, 1, 1), s_q, dtype=torch.int32, device=dev)
    if sink is True:
        sink_gpu = torch.randn(1, H, 1, 1, device=dev, dtype=torch.float32)
    elif sink is False or sink is None:
        sink_gpu = None
    else:
        sink_gpu = torch.full((1, H, 1, 1), float(sink), device=dev, dtype=torch.float32)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(q_gpu), g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    kw = dict(
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
    )
    vp = {q: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: slk}
    if causal_br:
        kw.update(diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT, diagonal_band_right_bound=0)
    if window_left is not None:
        assert causal_br, "a left window at decode rides the bottom-right causal diagonal (right bound 0)"
        kw["diagonal_band_left_bound"] = window_left
    if sink_gpu is not None:
        snk = g.tensor_like(sink_gpu)
        kw["sink_token"] = snk
        vp[snk] = sink_gpu
    o, st = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, engine_name(), pack_gqa=pack_gqa)
    if want_split is not None:
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        idx = next((i for i, n in enumerate(names) if n.startswith(engine_name()) and g.plans[i].knobs.split_kv == want_split), None)
        assert idx is not None, f"no {engine_name()} plan with split_kv={want_split}; knobs={[p.knobs for p in g.plans]}"
        g.select_plan(idx)
        plan = g.plans[idx]
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    vp[o] = o_gpu
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

    sink_ref = sink_gpu.flatten() if sink_gpu is not None else None
    ref_o, ref_lse = _ref_rows(q_gpu, k_pool, v_pool, bt.view(B, max_pages), seq_lens, hnd, scale, sink=sink_ref, window_left=window_left, causal_br=causal_br)
    out = o_gpu.float()
    assert not torch.isnan(out).any(), "NaN in O"
    torch.testing.assert_close(out, ref_o, atol=2e-2 if dtype == torch.float16 else 1e-1, rtol=0)
    # A keyless row -- every row of an empty sequence, or a row above the bottom-right
    # diagonal (its diagonal key ``L - S_q + r`` is negative; a left window never empties
    # a row whose diagonal key exists) -- has no KV mass: O := 0 exactly, and LSE := -inf
    # without a sink, := sink with one (the sink is then the row's whole mass, whatever
    # its magnitude -- exp(sink - max) underflowing in fp32 must not turn the row into
    # O = NaN / LSE = -inf).
    rows = torch.arange(s_q, device=dev).view(1, s_q)
    keyless = (seq_lens.view(B, 1) - s_q + rows < 0) if causal_br else (seq_lens.view(B, 1) == 0).expand(B, s_q)
    keyless = keyless.view(B, 1, s_q).expand(B, H, s_q)
    if keyless.any():
        assert out[keyless].abs().max().item() == 0.0, "a keyless row must write O := 0"
    if stats:
        got_lse = stats_gpu.view(B, H, s_q)
        torch.testing.assert_close(got_lse[~keyless], ref_lse[~keyless], atol=5e-3, rtol=0)
        if keyless.any():
            if sink_gpu is not None:
                want = sink_gpu.view(1, H, 1).expand(B, H, s_q)
                torch.testing.assert_close(got_lse[keyless], want[keyless], atol=1e-4, rtol=0)
            else:
                assert torch.isneginf(got_lse[keyless]).all(), "a keyless row must write LSE := -inf"
    return plan


# --- graph API ---------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 128])
@pytest.mark.parametrize("dims", [(128, 128), (192, 128)], ids=["d128", "d192x128"])
def test_paged_graph_matches_reference(dims, hnd, page_size):
    """Mixed lengths incl. 0 and 1, GQA 8:2 (PackGQA), a tile-unaligned tail and a
    length ending exactly on a page/tile boundary; Stats requested.  (192, 128)
    is the native d192x128 flavor: a 192-wide K pool and a 128-wide V pool
    behind separate block tables."""
    d_qk, d_v = dims
    _run_graph(5, 8, 2, d_qk, page_size, -(-1100 // page_size), [300, 77, 0, 1, 1024], hnd, stats=True, d_v=d_v)


@pytest.mark.L0
@pytest.mark.parametrize("dims", [(256, 128), (64, 192), (136, 72)], ids=["d256x128", "d64x192", "d136x72"])
def test_paged_graph_mixed_head_dims_ride_envelopes(dims):
    """Mixed head dims land on the smallest covering flavor — (256, 128) and
    (64, 192) on d256, (136, 72) on d192x128 — with the pools zero-padded by TMA
    past their real widths.  The earlier gate declined every graph with exactly
    one dim above 128 (INVERTED from a decline)."""
    d_qk, d_v = dims
    _run_graph(3, 8, 2, d_qk, 32, 40, [1000, 1, 1279], hnd=d_qk > d_v, stats=True, d_v=d_v)


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


# --- attention sink and sliding window at decode ----------------------------
#
# The sink is a per-Q-row epilogue fold (max lifted to max(m, sink), exp(sink -
# max) added to the denominator) and a left window only moves the first KV tile
# the loop visits; neither touches the block-table loader.  PAGED_KV + HAS_SINK
# had never been compiled together before these tests, and the python validator
# used to reject sink_token at s_q == 1 outright (the backend engines' rule).


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 128])
def test_paged_graph_sink(hnd, page_size):
    """Decode with an attention sink over a paged cache: GQA 8:2 (PackGQA), mixed
    lengths incl. 0 and 1 -- a keyless row is O = 0 / LSE = sink -- Stats out."""
    _run_graph(5, 8, 2, D, page_size, -(-1100 // page_size), [300, 77, 0, 1, 1024], hnd, sink=True, stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("pack_gqa", [False, True], ids=["unpacked", "packed"])
def test_paged_graph_sink_pack_gqa(pack_gqa):
    """Both GQA row mappings read the per-row sink logit (row_head_idx): pinned
    explicitly rather than left to whichever plan the heuristic ranks first."""
    _run_graph(3, 8, 2, D, 32, 40, [1000, 1279, 33], hnd=True, dtype=torch.bfloat16, sink=True, stats=True, pack_gqa=pack_gqa)


@pytest.mark.L0
@pytest.mark.parametrize("h,kh", [(96, 8), (48, 8)], ids=["g12_packs4", "g6_packs2"])
@pytest.mark.parametrize("s_q", [1, 4])
def test_paged_graph_sink_partial_pack_gqa(h, kh, s_q):
    """Sink + partial PackGQA (#1104): a GQA group that does not divide the 128-row
    tile packs its largest divisor that does (96/8: 4 of the 12 heads per token
    row-group, three packed heads per KV head; 48/8: 2 of 6).  The sink fold reads
    ``sinks[row_head_idx]`` with ``row_head_idx = packed_head * PACK_G + row % PACK_G``
    -- the Q head, not the KV head or the packed head -- and every head draws its own
    logit from randn, so a slip in that mapping moves the live rows' LSE and the
    keyless rows' LSE (= sink) alike.  bf16, page 16, left window 128 under the
    bottom-right diagonal at S_q = 4 (the 1-token sequence leaves three keyless
    rows), the packed plan pinned."""
    plan = _run_graph(
        4,
        h,
        kh,
        D,
        16,
        -(-1100 // 16),
        [300, 77, 1, 1100],
        hnd=True,
        dtype=torch.bfloat16,
        s_q=s_q,
        sink=True,
        window_left=128 if s_q > 1 else None,
        causal_br=s_q > 1,
        stats=True,
        pack_gqa=True,
    )
    assert plan.knobs.pack_gqa is True, plan.knobs


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("s_q", [1, 2])
def test_paged_graph_d256_sink(hnd, s_q):
    """The d256 flavor's PAGED_KV specialization with the sink fold, at S_q = 1 and
    as two-token decode under the bottom-right causal diagonal (a batch with one
    key leaves its first row keyless: LSE = sink)."""
    _run_graph(3, 8, 2, 256, 32, 40, [1000, 1, 1279], hnd, s_q=s_q, sink=True, causal_br=s_q > 1, stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [2, 4])
def test_paged_graph_sink_multi_token_causal_br(s_q):
    """S_q in {2, 4} (speculative / multi-token decode) with the bottom-right causal
    diagonal and a sink; a batch shorter than S_q leaves keyless rows (LSE = sink)."""
    _run_graph(4, 8, 2, D, 16, 70, [1000, 1, 129, 0], hnd=False, s_q=s_q, sink=True, causal_br=True, stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("window_left", [128, 33], ids=["w128", "w33"])
def test_paged_graph_sliding_window_causal_br(window_left):
    """Left window at decode over a paged cache: the window start lands mid-cache on
    page- and tile-unaligned keys, so the first KV tile the loop visits is read
    through the block table, not from page 0."""
    _run_graph(4, 8, 2, D, 16, 70, [1000, 129, 17, 1100], hnd=True, window_left=window_left, causal_br=True, stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [1, 4])
def test_paged_graph_d64_gqa8_sink_sliding_window(s_q):
    """d=64 on the d128 envelope, 64/8 heads (PackGQA, group 8), bf16, page 16,
    sink + left window 128 + bottom-right causal: the decode graph FlashInfer
    builds for a GPT-OSS-class model, at S_q = 1 and as multi-token decode."""
    _run_graph(4, 64, 8, 64, 16, 128, [2048, 1337, 129, 16], hnd=True, s_q=s_q, dtype=torch.bfloat16, sink=True, window_left=128, causal_br=True, stats=True)


@pytest.mark.L0
@pytest.mark.parametrize("sink", [-120.0, -5.0, 3.0], ids=["sink_m120", "sink_m5", "sink_p3"])
@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
def test_paged_graph_keyless_rows_sink_magnitude(d, sink):
    """Review regression (PR #1095): bf16, B = 1, 4/1 heads, paged page 16 HND, a
    128-key cache with ONE live key, S_q = 4 under the bottom-right causal diagonal,
    Stats on, the sink pinned per head.  Three of the four rows have no key, so the
    sink is their whole mass: O := 0, LSE := sink.  The softmax publishes a
    0-substituted row max with total_sum = 0 for such a row; a sink fold that
    COMPUTES the denominator from it gets exp(-120 - 0) = 0 in fp32, i.e. a zero
    denominator -> O = 0 * inf = NaN and LSE = 0 + log(0) = -inf.  -5 does not
    underflow (the fold happens to be right), +3 sits above the substituted max
    (the sink dominates; also right) -- the three pin the row's contract, not one
    arithmetic accident."""
    _run_graph(1, 4, 1, d, 16, 8, [1], hnd=True, dtype=torch.bfloat16, s_q=4, sink=sink, causal_br=True, stats=True)


@pytest.mark.L0
def test_paged_graph_declines_off_contract():
    """Plan-time declines stay plan-time: no engine plan is offered, nothing compiles."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name
    from frost_test_utils import offers_engine

    def _build(P, d=D, H=8, KH=2, hnd=False, padding=True, sink=False, d_v=None):
        B, max_pages = 2, 8
        dev = "cuda"
        d_v = d if d_v is None else d_v
        _, _, k_c, v_c, bt = _pools(B, KH, d, P, max_pages, hnd, torch.float16, d_v=d_v)
        q_gpu = torch.randn(B, 1, H, d, device=dev, dtype=torch.float16).transpose(1, 2)
        o_gpu = torch.empty(B, 1, H, d_v, device=dev, dtype=torch.float16).transpose(1, 2)
        g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        q, k, v, tk, tv = g.tensor_like(q_gpu), g.tensor_like(k_c), g.tensor_like(v_c), g.tensor_like(bt), g.tensor_like(bt)
        lens = torch.full((B, 1, 1, 1), 10, dtype=torch.int32, device=dev)
        sq_t, sk_t = g.tensor_like(lens), g.tensor_like(lens)
        kw = {}
        if sink:
            kw["sink_token"] = g.tensor(name="sink", dim=(1, H, 1, 1), stride=(H, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
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
            **kw,
        )
        o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
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
    # Lifted: paged KV + attention sink at s_q == 1 is offered (the decline
    # "paged KV with an attention sink is not validated" is gone, and the python
    # validator no longer rejects sink_token at s_q == 1).
    assert _offers(_build(16, sink=True)), "paged KV with an attention sink at decode must be offered"
    assert not _offers(_build(48)), "page_size 48 neither divides nor is a multiple of the 128-row tile"
    # The head-dim gate is the SELECTED flavor (Capabilities.paged_d_shapes):
    # d192x128 is wired, and mixed dims that ride the d256 envelope are served
    # (all three INVERTED from declines); a d512-envelope selection stays out.
    assert _offers(_build(16, d=192, d_v=128)), "paged KV is wired on the d192x128 flavor"
    assert _offers(_build(16, d=256, d_v=128)), "(256, 128) rides the d256 envelope"
    assert _offers(_build(16, d=64, d_v=192)), "(64, 192) rides the d256 envelope"
    assert not _offers(_build(16, d=512)), "paged KV is wired on the d128 / d192x128 / d256 flavors only"
    assert not _offers(_build(16, d=512, d_v=128)), "(512, 128) selects the d512 flavor, which is not wired"
    assert not _offers(_build(16, padding=False)), "paged KV requires the padding mask"


# --- kernel template, direct -----------------------------------------------
#
# What the graph path's heuristic would not choose on its own: forced split
# counts with EMPTY ranges (more splits than a short batch has tiles), cga1,
# and page sizes on both sides of the tile.  Mirrors test_sdpa_fwd_split_kv_sm100.


_KERNEL_FILES = {128: "prefill_d128_f16.py", 192: "prefill_d192_d128_f16.py", 256: "prefill_d256_f16.py"}


def _run_kernel(B, H, KH, P, max_pages, lens, hnd, splits, *, cta_mma=1, dtype=torch.float16, d=128, d_v=None, expect_pack_g=None):
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import MAKE_CFG, TemplateParams, pack_gqa_group_size
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    dev = "cuda"
    d_v = d if d_v is None else d_v
    G = H // KH
    # The d128 / d256 f16 kernels pack the largest divisor of G that divides the
    # 128-row tile (partial PackGQA); 1 means nothing packs and the unpacked path serves.
    pack = pack_gqa_group_size(G, 128, partial=True) > 1
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, k_c, v_c, bt4 = _pools(B, KH, d, P, max_pages, hnd, dtype, d_v=d_v)
    bt = bt4.view(B, max_pages)
    q = torch.randn(B, 1, H, d, device=dev, dtype=dtype)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    # kernel view: [num_pages, page_size, H_kv, d] (a permutation of the container)
    k_view, v_view = k_c.permute(0, 2, 1, 3), v_c.permute(0, 2, 1, 3)

    path = os.path.join(os.path.dirname(os.path.abspath(api_dsl.__file__)), "kernels", "sm100", _KERNEL_FILES[d])
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
    fn = mod.compile(b=B, qh=H, kh=KH, sq=1, skv=0, d_qk=d, d_v=d_v, has_lse=True, k_stride=tuple(k_view.stride()), v_stride=tuple(v_view.stride()))
    # Split partials are fp32 on SM100 (#891); the combine must be compiled for that width.
    from test_sdpa_fwd_split_kv_sm100 import _partial_kwargs, _partial_o_dtype, _partial_tag

    o_p = torch.zeros(splits * B, 1, H, d_v, device=dev, dtype=_partial_o_dtype(splits, dtype))
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
        o_out = torch.zeros(B, 1, H, d_v, device=dev, dtype=dtype)
        lse_out = torch.zeros(B, H, 1, device=dev, dtype=torch.float32)
        cfn = comb.compile(
            b=B, h=H, sq=1, d_v=d_v, splits=splits, dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=True, dtype_partial=_partial_tag(splits, dtype)
        )
        cfn(o_p, lse_p, o_out, lse_out, None, None, (B, H, 1, d_v), cutlass.Int32(splits), stream=stream)
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
@pytest.mark.parametrize("page_size,cta_mma,splits", [(16, 2, 4), (128, 2, 4), (32, 1, 1)], ids=["p16_cga2_split4", "p128_cga2_split4", "p32_cga1"])
def test_paged_kernel_d192_d128(page_size, cta_mma, splits):
    """The d192x128 f16 flavor's PAGED_KV specialization: a 192-wide K pool (three
    128 B D-subtiles per box) and a 128-wide V pool (two) behind separate block
    tables; cga2 (K box = 64 rows per CTA) with 4 forced splits and one empty
    range, and cga1 (K box = 128 rows, STAGES_KV=1) unsplit — d192 split-KV is
    cga2-only by its config rule."""
    _run_kernel(2, 8, 2, page_size, -(-1100 // page_size), [1000, 77], hnd=page_size == 16, splits=splits, cta_mma=cta_mma, d=192, d_v=128)


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
@pytest.mark.parametrize("dims", [(128, 128), (192, 128)], ids=["d128", "d192x128"])
def test_paged_graph_thd_queries(dims, hnd):
    """Ragged Q/O (packed [T, H, D] storage + ragged offsets, per-sequence
    ``seq_len_q``) attending K/V page pools through block tables: each sequence's
    q tokens see its own live KV pages. Only Q/O are ragged — the pools are
    ordinary dense tensors — and the THD scheduler walks the Q units while the
    KV side comes from ``seq_len_kv`` + the tables. On d192x128 the THD setup
    kernel skips the packed-total K/V descriptor clamp (pool-shaped descriptors)."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    dev, dtype = "cuda", torch.float16
    d, d_v = dims
    H, KH, P, max_pages = 8, 2, 16, 20
    q_lens = [37, 130, 5]
    kv_lens = [300, 77, 129]
    B, T, S_max = len(q_lens), sum(q_lens), max(q_lens)
    cu = [0]
    for s in q_lens:
        cu.append(cu[-1] + s)
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, k_c, v_c, bt = _pools(B, KH, d, P, max_pages, hnd, dtype, d_v=d_v)
    torch.manual_seed(7)
    q_pk = torch.randn(T, H, d, device=dev, dtype=dtype)
    stride = (S_max * H * d, d, H * d, 1)
    o_stride = (S_max * H * d_v, d_v, H * d_v, 1)
    q_stor = torch.zeros(B * S_max * H * d, device=dev, dtype=dtype)
    q_stor[: T * H * d] = q_pk.reshape(-1)
    q_gpu = q_stor.as_strided((B, H, S_max, d), stride)
    o_stor = torch.zeros(B * S_max * H * d_v, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H, S_max, d_v), o_stride)
    slq = torch.tensor(q_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(kv_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro = (torch.tensor(cu, dtype=torch.int64, device=dev) * H * d).view(B + 1, 1, 1, 1)
    o_ro = (torch.tensor(cu, dtype=torch.int64, device=dev) * H * d_v).view(B + 1, 1, 1, 1)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, S_max, d], stride=list(stride), data_type=cudnn.data_type.HALF, name="q")
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    qro, oro = g.tensor_like(ro), g.tensor_like(o_ro)
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
    o.set_output(True).set_dim([B, H, S_max, d_v]).set_stride(list(o_stride))
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
        g.execute({tq: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: slk, qro: ro, oro: o_ro, o: o_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    o_out = o_stor[: T * H * d_v].reshape(T, H, d_v).float()
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
