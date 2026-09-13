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


def _ref(q, k_pool, v_pool, block_table, seq_lens, hnd, scale):
    """q [B, H, D]; pools in their storage layout; returns (O [B, H, D_v], LSE [B, H]) fp32."""
    B, H, d = q.shape
    KH = k_pool.shape[1] if hnd else k_pool.shape[2]
    P = k_pool.shape[2] if hnd else k_pool.shape[1]
    Dv = v_pool.shape[-1]
    out = torch.zeros(B, H, Dv, device=q.device, dtype=torch.float32)
    lse = torch.full((B, H), float("-inf"), device=q.device, dtype=torch.float32)
    for b, L in enumerate(seq_lens.tolist()):
        if L == 0:
            continue
        pages = block_table[b, : (L + P - 1) // P].long()
        k, v = k_pool[pages], v_pool[pages]
        if hnd:
            k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        k = k.reshape(-1, KH, d)[:L].repeat_interleave(H // KH, dim=1).float()
        v = v.reshape(-1, KH, Dv)[:L].repeat_interleave(H // KH, dim=1).float()
        s = torch.einsum("hd,lhd->hl", q[b].float(), k) * scale
        out[b] = torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v)
        lse[b] = torch.logsumexp(s, -1)
    return out, lse


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


def _run_graph(B, H, KH, d, P, max_pages, lens, hnd, *, dtype=torch.float16, stats=False, s_q=1, max_seq_len=None, want_split=None):
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
    )
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, engine_name())
    if want_split is not None:
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        idx = next((i for i, n in enumerate(names) if n.startswith(engine_name()) and g.plans[i].knobs.split_kv == want_split), None)
        assert idx is not None, f"no {engine_name()} plan with split_kv={want_split}; knobs={[p.knobs for p in g.plans]}"
        g.select_plan(idx)
        plan = g.plans[idx]
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

    ref_o, ref_lse = _ref(q_gpu[:, :, 0, :], k_pool, v_pool, bt.view(B, max_pages), seq_lens, hnd, scale)
    out = o_gpu[:, :, 0, :].float()
    assert not torch.isnan(out).any(), "NaN in O"
    torch.testing.assert_close(out, ref_o, atol=2e-2 if dtype == torch.float16 else 1e-1, rtol=0)
    live = seq_lens > 0
    if (~live).any():
        assert out[~live].abs().max().item() == 0.0, "empty sequence must write O := 0"
    if stats:
        got_lse = stats_gpu.view(B, H)
        torch.testing.assert_close(got_lse[live], ref_lse[live], atol=5e-3, rtol=0)
        if (~live).any():
            assert torch.isinf(got_lse[~live]).all() and (got_lse[~live] < 0).all(), "empty sequence must write LSE := -inf"
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


# --- kernel template, direct -----------------------------------------------
#
# What the graph path's heuristic would not choose on its own: forced split
# counts with EMPTY ranges (more splits than a short batch has tiles), cga1,
# and page sizes on both sides of the tile.  Mirrors test_sdpa_fwd_split_kv_sm100.


def _run_kernel(B, H, KH, P, max_pages, lens, hnd, splits, *, cta_mma=1, dtype=torch.float16, d=128):
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    dev = "cuda"
    G = H // KH
    pack = G > 1 and 128 % G == 0
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
        None,
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
def test_paged_kernel_gqa_group_not_dividing_tile():
    """H/H_kv = 3 cannot pack a 128-row tile; the unpacked path serves it."""
    _run_kernel(2, 6, 2, 16, 8, [50, 128], hnd=False, splits=1, cta_mma=2)


@pytest.mark.L0
def test_paged_adapter_cuda_graph_replay_no_host_sync():
    """The adapter's execute path captured once at fixed B; seq_lens CONTENT
    changes between replays.  ``set_sync_debug_mode("error")`` during capture
    makes any blocking D2H raise (python/cudnn/AGENTS.md Rule 3)."""
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
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.graph(g, stream=s):
            api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
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
