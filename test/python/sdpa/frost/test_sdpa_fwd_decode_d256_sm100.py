# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The d256 DECODE tile of the FROST SM100 f16/bf16 engine (sm100/decode_d256_f16.py).

Decode-shaped d256 graphs (S_q x packed heads <= 32 rows) lower onto the
swap-AB tile -- KV tokens on the MMA M axis, the packed Q rows on N -- instead
of the 256-row prefill tile.  Same graph contract, same engine row
(``sdpa_fwd_prefill_sm100``): every test here pins the engine with
``select_engine`` and additionally asserts WHICH template served the plan
through the executor's ``kernel_template`` (a decline or a fallback to the
prefill tile fails instead of passing on the wrong kernel).  The reference is
an fp32 torch softmax over the dense K/V the pools were built from.

Coverage: paged (page 16/32/64/128, NHD/HND) and dense padded caches, mixed
lengths incl. 0 and 1, PackGQA 8:1 / 16:1 / 32:2 and MHA, MTP bottom-right
causal at S_q 2 and 4 with per-batch Q lengths (dense padded-Q trim), sliding
window, right band, sink, Stats natural and base-2, fp16 and bf16, the
decode split policy (the serving shape leads unsplit with the split as the
runner-up plan; a small batch splits) plus forced splits with empty ranges,
CUDA-graph replay under ``set_sync_debug_mode("error")`` (Rule 3), and the
routing boundary: larger S_q x G, THD queries and d128 stay on the prefill
tiles.
"""

import math

import pytest
import torch

from frost_test_utils import _is_plan_for, requires_dsl, requires_pre_rubin_blackwell, select_engine

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]

D = 256
DECODE = "decode_d256_f16"
PREFILL = "prefill_d256_f16"


def _ref(q, k, v, lens, q_lens, scale, *, causal_br=False, window_left=None, window_right=None, sinks=None):
    """fp32 reference.  q [B, S_q, H, d]; k/v [B, S, KH, d] dense; per-batch KV
    lengths ``lens`` and Q lengths ``q_lens`` (rows past them: O := 0, LSE := -inf).
    Returns O [B, S_q, H, d_v] and the natural-log LSE [B, H, S_q]."""
    B, Sq, H, _ = q.shape
    KH = k.shape[2]
    G = H // KH
    dev = q.device
    if causal_br and window_right is None:
        window_right = 0  # bottom-right causal: the band's right edge is the diagonal itself
    O = torch.zeros(B, Sq, H, v.shape[3], dtype=torch.float32, device=dev)
    LSE = torch.full((B, H, Sq), float("-inf"), dtype=torch.float32, device=dev)
    for b in range(B):
        L = int(lens[b])
        Lq = int(q_lens[b]) if q_lens is not None else Sq
        if L == 0:
            if sinks is not None:
                LSE[b] = sinks.view(H, 1).expand(H, Sq)
                LSE[b, :, Lq:] = float("-inf")
            continue
        kk = k[b, :L].float().repeat_interleave(G, dim=1)
        vv = v[b, :L].float().repeat_interleave(G, dim=1)
        s = torch.einsum("qhd,khd->hqk", q[b].float(), kk) * scale
        key = torch.arange(L, device=dev).view(1, 1, L)
        qi = torch.arange(Sq, device=dev).view(1, Sq, 1)
        diag = (L - Lq) if causal_br else 0
        mask = torch.zeros(1, Sq, L, dtype=torch.bool, device=dev)
        if window_right is not None:
            mask |= key > qi + diag + window_right
        if window_left is not None:
            mask |= key < qi + diag - window_left
        s = s.masked_fill(mask, float("-inf"))
        s_ext = torch.cat([s, sinks.view(H, 1, 1).expand(H, Sq, 1)], dim=-1) if sinks is not None else s
        m = s_ext.amax(-1, keepdim=True)
        dead = torch.isinf(m) & (m < 0)
        m = torch.where(dead, torch.zeros_like(m), m)
        p = torch.exp(s_ext - m)
        l = p.sum(-1, keepdim=True)
        o = torch.einsum("hqk,khd->qhd", p[..., :L] / l.clamp_min(1e-30), vv)
        lse = (m + torch.log(l.clamp_min(1e-30))).squeeze(-1)
        lse = torch.where(dead.squeeze(-1), torch.full_like(lse, float("-inf")), lse)
        o = torch.where(dead.squeeze(-1).t().unsqueeze(-1), torch.zeros_like(o), o)
        o[Lq:] = 0
        lse[:, Lq:] = float("-inf")
        O[b] = o
        LSE[b] = lse
    return O, LSE


def _pools(k_dense, v_dense, P, hnd, seed=0):
    """Scatter dense [B, S, KH, d] K/V into page pools (scattered block table).
    Returns the containers the graph declares ([num_pages, KH, P, d] dims; HND
    compact or NHD storage through the strides) and the (B, 1, max_pages, 1) table."""
    B, S, KH, d = k_dense.shape
    dev, dtype = k_dense.device, k_dense.dtype
    max_pages = -(-S // P)
    num_pages = B * max_pages + 5
    g = torch.Generator(device="cpu").manual_seed(seed)
    bt = torch.randperm(num_pages, generator=g)[: B * max_pages].to(torch.int32).view(B, max_pages).to(dev)
    pad = max_pages * P - S
    kp = torch.nn.functional.pad(k_dense, (0, 0, 0, 0, 0, pad)).view(B, max_pages, P, KH, d)
    vp = torch.nn.functional.pad(v_dense, (0, 0, 0, 0, 0, pad)).view(B, max_pages, P, KH, d)
    if hnd:
        k_pool = torch.zeros(num_pages, KH, P, d, device=dev, dtype=dtype)
        v_pool = torch.zeros(num_pages, KH, P, d, device=dev, dtype=dtype)
        k_pool[bt.view(-1).long()] = kp.view(-1, P, KH, d).permute(0, 2, 1, 3)
        v_pool[bt.view(-1).long()] = vp.view(-1, P, KH, d).permute(0, 2, 1, 3)
        k_c, v_c = k_pool, v_pool
    else:
        k_pool = torch.zeros(num_pages, P, KH, d, device=dev, dtype=dtype)
        v_pool = torch.zeros(num_pages, P, KH, d, device=dev, dtype=dtype)
        k_pool[bt.view(-1).long()] = kp.view(-1, P, KH, d)
        v_pool[bt.view(-1).long()] = vp.view(-1, P, KH, d)
        k_c, v_c = k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3)
    return k_c, v_c, bt.view(B, 1, max_pages, 1).contiguous()


def _served_by(graph, plan_index):
    """The template file stem that serves the built plan (``kernel_template``
    on the FROST executor, see engines.lower_dsl_prefill)."""
    return graph._compiled_plans[plan_index]._compiled.kernel_template


def _run_graph(
    *,
    B,
    H,
    KH,
    s_q,
    lens,
    d=D,
    page=16,
    hnd=True,
    dtype=torch.float16,
    stats=True,
    stats_log2=False,
    causal_br=False,
    window_left=None,
    window_right=None,
    sink=False,
    q_lens=None,
    expect=DECODE,
    seed=0,
    split_kv=None,
):
    """Build cuDNN's paged (``page`` > 0) or dense padded SDPA graph, pin the
    FROST engine (its first entry: the heuristics' own choice, split included --
    or, with ``split_kv``, its ranked entry carrying that split, a runner-up
    when the heuristics lead elsewhere), assert the serving template, execute
    under the D2H detector and compare O / Stats against :func:`_ref`.
    Returns the pinned plan."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    torch.manual_seed(seed)
    dev = "cuda"
    S = max(max(lens), 1)
    scale = 1.0 / math.sqrt(d)
    q_gpu = torch.randn(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)  # BHSD strides over a BSHD buffer
    k_dense = torch.randn(B, S, KH, d, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, d, device=dev, dtype=dtype)
    o_gpu = torch.empty(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)
    seq_kv = torch.tensor(lens, dtype=torch.int32, device=dev)
    seq_q = torch.tensor(q_lens if q_lens is not None else [s_q] * B, dtype=torch.int32, device=dev)
    sinks = (torch.randn(H, device=dev) * 2.0) if sink else None

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor_like(q_gpu)
    kw = dict(name="sdpa", generate_stats=stats, attn_scale=scale, use_padding_mask=True, stats_use_log2=stats_log2)
    if page:
        k_c, v_c, bt = _pools(k_dense, v_dense, page, hnd, seed)
        k, v = g.tensor_like(k_c), g.tensor_like(v_c)
        tk, tv = g.tensor_like(bt), g.tensor_like(bt)
        kw.update(paged_attention_k_table=tk, paged_attention_v_table=tv, paged_attention_max_seq_len_kv=S)
    else:
        k_c, v_c = k_dense.transpose(1, 2), v_dense.transpose(1, 2)  # BHSD view of the BSHD buffers
        k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    slq, slk = seq_q.view(B, 1, 1, 1), seq_kv.view(B, 1, 1, 1)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    kw.update(q=q, k=k, v=v, seq_len_q=sq_t, seq_len_kv=sk_t)
    if causal_br:
        kw["use_causal_mask_bottom_right"] = True
    elif window_right is not None:
        kw["use_causal_mask"] = window_right == 0
        if window_right > 0:
            kw["diagonal_band_right_bound"] = window_right
    if window_left is not None:
        kw["sliding_window_length"] = window_left + 1
    sink_t = None
    if sink:
        sink_gpu = sinks.view(1, H, 1, 1).contiguous()
        sink_t = g.tensor_like(sink_gpu)
        kw["sink_token"] = sink_t
    o, st = g.sdpa(**kw)
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, engine_name())
    if split_kv is not None:
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        idx = next((i for i, n in enumerate(names) if _is_plan_for(n, engine_name()) and g.plans[i].knobs.split_kv == split_kv), None)
        assert idx is not None, f"no {engine_name()} entry with split_kv={split_kv}; plans={names}"
        g.select_plan(idx)
        plan = g.plans[idx]
    idx = g._plan_index
    g.check_support()
    g.build_plans()
    assert _served_by(g, idx) == expect, f"plan {g.get_plan_name_at_index(idx)} served by {_served_by(g, idx)}, expected {expect}"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    vp = {q: q_gpu, k: k_c, v: v_c, sq_t: slq, sk_t: slk, o: o_gpu}
    if page:
        vp.update({tk: bt, tv: bt})
    if stats:
        vp[st] = stats_gpu
    if sink:
        vp[sink_t] = sink_gpu
    # Rule 3: the execute path reads the per-batch lengths and page tables on
    # device; any blocking D2H here is a bug, not a slow path.
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    ref_o, ref_lse = _ref(
        q_gpu.transpose(1, 2), k_dense, v_dense, lens, q_lens, scale, causal_br=causal_br, window_left=window_left, window_right=window_right, sinks=sinks
    )
    out = o_gpu.transpose(1, 2).float()
    assert not torch.isnan(out).any(), "NaN in O"
    torch.testing.assert_close(out, ref_o, atol=2e-2 if dtype == torch.float16 else 5e-2, rtol=0)
    dead = torch.isinf(ref_lse)
    if dead.any():
        assert out.transpose(1, 2)[dead].abs().max().item() == 0.0, "dead rows must write O := 0"
    if stats:
        got_lse = stats_gpu.view(B, H, s_q)
        exp_lse = ref_lse * (1.4426950408889634 if stats_log2 else 1.0)
        torch.testing.assert_close(got_lse[~dead], exp_lse[~dead], atol=5e-3, rtol=0)
        if dead.any():
            assert torch.isinf(got_lse[dead]).all() and (got_lse[dead] < 0).all(), "dead rows must write LSE := -inf"
    return plan


# --- accept: the decode tile serves the contract ------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page", [16, 32, 64, 128])
def test_decode_graph_paged_page_sizes(page, hnd):
    """Qwen3.5-shaped 32/2 (PackGQA 16:1) decode over pages of every admitted
    size: mixed lengths incl. 0 and 1, a tile-unaligned tail, a length ending on
    a page/tile boundary; Stats out."""
    _run_graph(B=5, H=32, KH=2, s_q=1, lens=[300, 77, 0, 1, 1024], page=page, hnd=hnd)


@pytest.mark.L0
@pytest.mark.parametrize(("H", "KH"), [(8, 1), (16, 1), (32, 2), (16, 4), (4, 4)], ids=["8to1", "16to1", "32to2", "16to4", "mha"])
def test_decode_graph_head_groups(H, KH):
    """PackGQA 8:1 and 16:1 (one and two tokens per 16-row tile at S_q = 1),
    32:2, a 4-wide group, and MHA (one row per unit)."""
    _run_graph(B=3, H=H, KH=KH, s_q=1, lens=[1000, 129, 640], page=32, dtype=torch.bfloat16)


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [2, 4])
def test_decode_graph_mtp_bottom_right(s_q):
    """MTP: S_q in {2, 4} bottom-right causal over a paged cache (8:1 packing
    -> 16 and 32 rows: one and two column groups, with two tokens per group at
    S_q = 4), with per-batch Q lengths below S_q (dense padded-Q trim: O := 0 /
    LSE := -inf past them, and the diagonal anchored at seq_len_kv[b] -
    seq_len_q[b])."""
    _run_graph(B=3, H=16, KH=2, s_q=s_q, lens=[700, 130, 5], q_lens=[s_q, max(1, s_q - 1), 0], page=16, causal_br=True)


@pytest.mark.L0
def test_decode_graph_sliding_window_bottom_right():
    """Sliding window (left bound) on the bottom-right diagonal, S_q = 1."""
    _run_graph(B=2, H=32, KH=2, s_q=1, lens=[700, 130], page=16, causal_br=True, window_left=200)


@pytest.mark.L0
def test_decode_graph_right_band_dense():
    """Top-left causal widened by a right band on a dense padded cache, S_q = 2."""
    _run_graph(B=2, H=8, KH=2, s_q=2, lens=[300, 129], page=0, window_right=100)


@pytest.mark.L0
def test_decode_graph_sink_dense():
    """Attention sink folded once per Q row (dense cache: the engine row keeps
    paged + sink declined, see engines.mismatch; S_q = 2 because cuDNN's
    validator rejects a sink at S_q == 1); a keyless batch keeps the sink's
    finite LSE and O := 0."""
    _run_graph(B=3, H=32, KH=2, s_q=2, lens=[700, 0, 300], page=0, sink=True)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_decode_graph_stats_log2(dtype):
    """Base-2 Stats (stats_use_log2) on the decode tile, both half dtypes."""
    _run_graph(B=2, H=32, KH=2, s_q=1, lens=[700, 130], page=16, dtype=dtype, stats_log2=True)


@pytest.mark.L0
def test_decode_graph_dense_padded_no_stats():
    _run_graph(B=2, H=32, KH=2, s_q=1, lens=[1000, 77], page=0, stats=False, dtype=torch.bfloat16)


@pytest.mark.L0
def test_decode_graph_serving_shape_leads_unsplit_with_the_split_as_runner_up():
    """b=32 x 2 KV heads over 4096 keys (Qwen3.5 serving) at S_q=1: the LEADING
    plan does not split -- split 2 saves ~6 us of GPU time but costs an eager
    caller ~30 us of host time per execute -- and the captured caller's split-2
    plan is the runner-up, reachable by select_plan.  Both run and match the
    reference.
    The literal splits are the 148-SM fit (test_split_kv_heuristic pins the
    policy); on another part the plans must still agree with the model."""
    from cudnn.sdpa.fwd.heuristics import choose_decode_tile_split_kv

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    shape = dict(B=32, H=32, KH=2, s_q=1, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16)
    lead = _run_graph(**shape)
    assert lead.knobs.split_kv == choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=sm), lead.knobs
    captured = choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=sm, launch_cost=0.0)
    runner = _run_graph(**shape, split_kv=captured)
    assert runner.knobs.split_kv == captured, runner.knobs
    if sm == 148:
        assert (lead.knobs.split_kv, runner.knobs.split_kv) == (1, 2), (lead.knobs, runner.knobs)
    # The MTP step (S_q=2 bottom-right: 32 packed rows on the 32-column tile,
    # 90 us unsplit vs 58 us split on B200) leads WITH the split.
    mtp = _run_graph(B=32, H=32, KH=2, s_q=2, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16, causal_br=True)
    assert mtp.knobs.split_kv == choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=sm, q_tile=32), mtp.knobs
    if sm == 148:
        assert mtp.knobs.split_kv == 2, mtp.knobs


@pytest.mark.L0
def test_decode_graph_small_batch_splits_and_recombines():
    """b=8 x 2 KV heads is 16 units: unsplit they would stream 32 tiles each on
    16 of the SMs, so the decode model splits (8 ways on a 148-SM part: 128
    CTAs, and the GPU saving covers the second launch) and the recombined
    O / LSE over mixed lengths match."""
    from cudnn.sdpa.fwd.heuristics import choose_decode_tile_split_kv

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    plan = _run_graph(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 4096, 300, 77], page=16, dtype=torch.bfloat16)
    assert plan.knobs.split_kv == choose_decode_tile_split_kv(units=16, kv_tiles=32, sm_count=sm), plan.knobs
    if sm == 148:
        assert plan.knobs.split_kv == 8, plan.knobs


@pytest.mark.L0
def test_decode_graph_deep_split_empty_ranges():
    """Six units over a 4096-key table: the decode rule splits 16 ways (96 CTAs on
    a 148-SM part), so the 1- and 129-key batches leave most split ranges empty
    next to live ones; those must yield -inf / 0 partials the combine ignores."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    plan = _run_graph(B=3, H=32, KH=2, s_q=1, lens=[4096, 1, 129], page=16)
    assert plan.knobs.split_kv >= (8 if sm >= 96 else 2), plan.knobs


# --- reject / routing boundary: everything else stays on the prefill tiles ------


@pytest.mark.L0
def test_decode_q_tile_rule():
    from cudnn.sdpa.fwd.config_sm100 import D256_DECODE_MAX_Q_ROWS, decode_d256_q_tile

    assert D256_DECODE_MAX_Q_ROWS == 32
    assert decode_d256_q_tile(1, 16) == 16
    assert decode_d256_q_tile(1, 12) == 16  # a group that does not divide the tile still fits it
    assert decode_d256_q_tile(2, 16) == 32
    assert decode_d256_q_tile(1, 32) == 32
    assert decode_d256_q_tile(32, 1) == 32
    assert decode_d256_q_tile(3, 16) == 0
    assert decode_d256_q_tile(33, 1) == 0
    assert decode_d256_q_tile(1, 64) == 0


@pytest.mark.L0
def test_decode_routing_boundary():
    """S_q x G past the tile stays on the prefill d256 tile; d128 stays on its
    own (THD queries: test_decode_routing_boundary_thd_queries)."""
    _run_graph(B=2, H=32, KH=2, s_q=3, lens=[700, 130], page=16, expect=PREFILL)  # 48 packed rows
    _run_graph(B=2, H=8, KH=2, s_q=1, lens=[700, 130], d=128, page=16, expect="prefill_d128_f16")
    _run_graph(B=2, H=64, KH=1, s_q=1, lens=[300, 130], page=16, expect=PREFILL)  # 64 packed rows


@pytest.mark.L0
def test_decode_routing_boundary_thd_queries():
    """Ragged (THD) queries -- packed [T, H, d] Q/O storage with ragged offsets,
    here one token per sequence -- over a paged cache stay on the prefill d256
    tile (the decode tile has no THD scheduler); the graph still runs under the
    D2H detector and matches the reference."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    dev, dtype = "cuda", torch.float16
    B, H, KH, P = 3, 32, 2, 16
    lens = [700, 130, 5]
    S = max(lens)
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(3)
    k_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    k_c, v_c, bt = _pools(k_dense, v_dense, P, True, seed=3)
    q_pk = torch.randn(B, H, D, device=dev, dtype=dtype)  # T = B tokens, one per sequence
    stride = (H * D, D, H * D, 1)  # (B, H, S_max=1, d) over the packed [T, H, d] storage
    q_gpu = q_pk.view(-1).as_strided((B, H, 1, D), stride)
    o_stor = torch.zeros(B * H * D, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H, 1, D), stride)
    slq = torch.ones(B, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro = (torch.arange(B + 1, dtype=torch.int64, device=dev) * H * D).view(B + 1, 1, 1, 1)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, 1, D], stride=list(stride), data_type=cudnn.data_type.HALF, name="q")
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
        paged_attention_max_seq_len_kv=S,
    )
    o.set_output(True).set_dim([B, H, 1, D]).set_stride(list(stride))
    o.set_ragged_offset(oro)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine_name())
    idx = g._plan_index
    g.check_support()
    g.build_plans()
    assert _served_by(g, idx) == PREFILL, f"plan {g.get_plan_name_at_index(idx)} served by {_served_by(g, idx)}, expected {PREFILL}"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute({tq: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: slk, qro: ro, oro: ro, o: o_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    ref_o, _ = _ref(q_pk.view(B, 1, H, D), k_dense, v_dense, lens, None, scale)
    torch.testing.assert_close(o_stor.view(B, 1, H, D).float(), ref_o, atol=2e-2, rtol=0)


@pytest.mark.L0
def test_decode_prefill_backstop_rejects_decode_record():
    """A TemplateParams carrying decode_q_tile must never reach a prefill template."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256, make_cfg_d256_decode

    params = TemplateParams(decode_q_tile=16, seq_kv_lens_present=True)
    with pytest.raises(ValueError, match="decode_q_tile"):
        make_cfg_d256(params)
    with pytest.raises(ValueError, match="decode_q_tile"):
        make_cfg_d256_decode(TemplateParams())
    with pytest.raises(ValueError, match="THD"):
        make_cfg_d256_decode(TemplateParams(decode_q_tile=16, thd_varlen=True, seq_kv_lens_present=True))
    with pytest.raises(ValueError, match="fit the Q tile"):
        make_cfg_d256_decode(TemplateParams(decode_q_tile=16, pack_gqa=True, qh_per_kh=32))
    cfg, _ = make_cfg_d256_decode(params)
    assert cfg.N_Q == 16 and cfg.TILE_N == 128 and cfg.STAGES_KV == 3


@pytest.mark.L0
def test_decode_adapter_cuda_graph_replay_no_host_sync():
    """The adapter's execute path captured once at fixed B; seq_lens CONTENT
    changes between replays (Rule 3, same protocol as the paged suite's twin)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, S = 8, 32, 2, 16, 1024
    dev, dtype = "cuda", torch.float16
    torch.manual_seed(2)
    k_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, False, seed=2)
    bt = bt4.view(B, -1)
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
        paged_max_seq_len_kv=S,
        split_kv=4,
        pack_gqa=True,
    )
    api.check_support()
    api.compile()
    assert api.kernel_template == DECODE
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    prev = torch.cuda.get_sync_debug_mode()
    with torch.cuda.graph(g, stream=s):
        torch.cuda.set_sync_debug_mode("error")
        try:
            api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
        finally:
            torch.cuda.set_sync_debug_mode(prev)
    scale = 1.0 / math.sqrt(D)
    for new_lens in ([5, 1024, 77, 128, 129, 1, 512, 1000], [1024] * B, [0, 1, 2, 3, 4, 5, 6, 7]):
        seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        ref_o, ref_lse = _ref(q_gpu.transpose(1, 2), k_dense, v_dense, new_lens, None, scale)
        live = ~torch.isinf(ref_lse)
        torch.testing.assert_close(o_gpu.transpose(1, 2).float(), ref_o, atol=2e-2, rtol=0)
        torch.testing.assert_close(lse.view(B, H, 1)[live], ref_lse[live], atol=5e-3, rtol=0)
