# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST SM100 d128 DECODE tile (``sm100/decode_d128_f16.py``).

``sdpa_fwd_prefill_sm100``'s (128, 128) f16/bf16 flavor has two tiles behind
one knob: ``TILE_CGA_M=2`` is the prefill pipeline (512 Q rows per cga2
cluster) and ``TILE_CGA_M=1`` the decode tile (128 rows per independent CTA,
one softmax warpgroup, three KV stages).  The heuristics propose cga=1 exactly
when one 128-row tile covers a KV head's Q rows -- ``S_q * pack_g <= 128``:
S_q = 1 decode and MTP -- and cga=2 otherwise.

Three tiers:

- GPU-free: the config backstop (accept / reject), the standalone cga domain,
  the heuristics' cga and scheduler rules, and the ``mismatch`` gates (a THD
  graph never gets the decode tile).
- Kernel template, direct: paged pools at every page geometry, mixed lengths
  incl. 0 and 1, PackGQA on / off / non-dividing group, forced splits with
  empty ranges, bottom-right causal MTP with per-batch Q lengths (the trim),
  sliding window, sink, base-2 stats, the d64 envelope, the LPT schedulers.
- Graph API: decode / MTP shapes select the decode tile, a prefill shape keeps
  the prefill tile, a pinned cga=2 on a decode shape is honored, THD declines
  cga=1, dense (non-paged) padded decode, a small-batch split, and CUDA-graph
  replay under ``set_sync_debug_mode("error")`` (Rule 3).
"""

import math
import os

import pytest
import torch

from frost_test_utils import offers_engine, requires_dsl, requires_pre_rubin_blackwell, select_engine

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]

D = 128
ENGINE = "sdpa_fwd_prefill_sm100"
_SM100_ID = 20511  # FROST_SDPA_FWD_ID_BASE + manifest slot 11 (sdpa_fwd_prefill_sm100)


# --- reference ----------------------------------------------------------------


def _gather_kv(pool, table_row, n_tokens, hnd):
    """One batch's live K or V rows [L, KH, d] from its page pool through its table row."""
    P = pool.shape[2] if hnd else pool.shape[1]
    pages = table_row[: (n_tokens + P - 1) // P].long()
    t = pool[pages]
    if hnd:
        t = t.permute(0, 2, 1, 3)
    return t.reshape(-1, t.shape[2], t.shape[3])[:n_tokens]


def _ref(q, k_rows, v_rows, q_len, scale, *, causal_br=False, window_left=None, sink=None):
    """fp32 attention for ONE batch: q [S_q, H, d], k_rows/v_rows [L, KH, d].

    Bottom-right causal anchors the diagonal at (q_len, L): row r (< q_len) sees
    columns <= L - q_len + r; a sliding window keeps columns >= diag - window_left;
    the sink is a virtual column with no V row.  Rows >= q_len are dead (O := 0,
    LSE := -inf), as is every row of a batch with L == 0 and no sink.
    Returns (O [S_q, H, d_v] fp32, LSE [H, S_q] fp32).
    """
    s_q, H, d = q.shape
    L = k_rows.shape[0]
    KH = k_rows.shape[1]
    G = H // KH
    out = torch.zeros(s_q, H, v_rows.shape[2], dtype=torch.float32, device=q.device)
    lse = torch.full((H, s_q), float("-inf"), dtype=torch.float32, device=q.device)
    if q_len == 0 or (L == 0 and sink is None):
        return out, lse
    k = k_rows.float().repeat_interleave(G, dim=1)  # [L, H, d]
    v = v_rows.float().repeat_interleave(G, dim=1)
    s = torch.einsum("qhd,lhd->hql", q[:q_len].float(), k) * scale  # [H, q_len, L]
    if causal_br or window_left is not None:
        rows = torch.arange(q_len, device=q.device)[:, None]
        cols = torch.arange(L, device=q.device)[None, :]
        diag = L - q_len + rows
        mask = torch.zeros(q_len, L, dtype=torch.bool, device=q.device)
        if causal_br:
            mask |= cols > diag
        if window_left is not None:
            mask |= cols < diag - window_left
        s = s.masked_fill(mask[None], float("-inf"))
    if sink is not None:
        s = torch.cat([s, sink.view(H, 1, 1).expand(H, q_len, 1).float()], dim=-1)
        p = torch.softmax(s, -1)[..., :L]
    else:
        p = torch.softmax(s, -1)
    p = torch.nan_to_num(p, nan=0.0)  # a fully-masked row (no sink): dead
    out[:q_len] = torch.einsum("hql,lhd->qhd", p, v)
    lse_live = torch.logsumexp(s, -1)  # [H, q_len]; -inf for a fully-masked row
    lse[:, :q_len] = lse_live
    dead_rows = torch.isinf(lse_live).T  # [q_len, H]
    out[:q_len][dead_rows] = 0.0
    return out, lse


def _pools(B, KH, d, P, max_pages, hnd, dtype, seed=0):
    """Page pools + a scattered block table [B, max_pages] (and its (B,1,max_pages,1) graph view)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    num_pages = B * max_pages + 3
    shape = (num_pages, KH, P, d) if hnd else (num_pages, P, KH, d)
    k_pool = torch.randn(shape, device=dev, dtype=torch.float32, generator=g).to(dtype)
    v_pool = torch.randn(shape, device=dev, dtype=torch.float32, generator=g).to(dtype)
    bt = torch.randperm(num_pages, device=dev, generator=g)[: B * max_pages].to(torch.int32).view(B, max_pages).contiguous()
    return k_pool, v_pool, bt


def _tol(dtype):
    return 2e-2 if dtype == torch.float16 else 1e-1


# --- GPU-free: config, standalone domain, heuristics, mismatch ------------------


@pytest.mark.L0
def test_decode_cfg_accepts_the_decode_geometry_and_rejects_the_rest():
    """The config backstop: the tile is cga1 / TILES_Q=1 / one softmax warpgroup /
    three KV stages / 12 warps and fits SMEM; it refuses cga2, THD, fp8 and the
    MXFP8 experiment axes (each a constraint the engine gates uphold upstream)."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
    from cudnn.sdpa.fwd.config_sm100 import CfgD128Decode, TemplateParams, _SM100_MAX_DYN_SMEM, _d128_smem_bytes, cga_tile_m, make_cfg_d128_decode

    for dtype in (DTYPE_FP16, DTYPE_BF16):
        cfg, tma = make_cfg_d128_decode(
            TemplateParams(dtype_qkv=dtype, cta_mma=1, seq_kv_lens_present=True, paged_kv=True, page_size=16, pack_gqa=True, qh_per_kh=16)
        )
        assert isinstance(cfg, CfgD128Decode)
        assert (cfg.CTA_MMA, cfg.CGA_M, cfg.TILES_Q, cfg.STAGES_KV, cfg.SOFTMAX_WARPGROUPS, cfg.TOTAL_WARPS, cfg.QO_ALIAS) == (1, 1, 1, 3, 1, 12, 1)
        assert cfg.READ_TILE_ARRIVERS == 11 and cfg.PAGED_KV == 1 and cfg.PAGE_SIZE == 16 and cfg.PACK_GQA == 1
        assert _d128_smem_bytes(cfg) == 224 * 1024 <= _SM100_MAX_DYN_SMEM
        assert (tma.QK_ITERS, tma.VO_ITERS) == (2, 2)
    # cga1 on the d128 flavor IS the decode tile: 128 Q rows per CTA, not the prefill's 512 per cluster.
    assert cga_tile_m(128, 1) == 128 and cga_tile_m(128, 2) == 512 and cga_tile_m(128) == 512
    with pytest.raises(ValueError, match="cga1 only"):
        make_cfg_d128_decode(TemplateParams(cta_mma=2))
    with pytest.raises(ValueError, match="THD"):
        make_cfg_d128_decode(TemplateParams(cta_mma=1, thd_varlen=True, seq_kv_lens_present=True))
    with pytest.raises(ValueError, match="f16/bf16"):
        make_cfg_d128_decode(TemplateParams(cta_mma=1, dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_BF16))
    with pytest.raises(ValueError, match="PackGQA|divide"):
        make_cfg_d128_decode(TemplateParams(cta_mma=1, pack_gqa=True, qh_per_kh=12))


@pytest.mark.L0
def test_standalone_cga_domain_admits_cga1_on_d128_f16_only():
    """The adapter's twin of the engine row's domain (keep the three in lockstep)."""
    from cudnn.sdpa.fwd.api_dsl import supported_cgas_for

    assert supported_cgas_for((128, 128), fp8=False, device_cc=(10, 0)) == (1, 2)
    assert supported_cgas_for((128, 128), fp8=False, device_cc=(10, 3)) == (1, 2)
    assert supported_cgas_for((128, 128), fp8=True, device_cc=(10, 0)) == (2,), "the fp8 families keep the prefill tile"
    assert supported_cgas_for((128, 128), fp8=False, device_cc=(10, 7)) == (2,), "no Rubin sibling of the decode tile"
    assert supported_cgas_for((256, 256), fp8=False, device_cc=(10, 0)) == (2,)


def _facts(**over):
    import cudnn
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    base = dict(
        b=32,
        h_q=64,
        h_kv=4,
        s_q=1,
        s_kv=4096,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=False,
        padded=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    base.update(over)
    return SdpaGraphFacts(**base)


def _plans(facts):
    from cudnn.sdpa.fwd.heuristics import recommend

    plans = recommend("A", facts, {ENGINE: _SM100_ID})
    assert plans, "the SM100 f16 row must be eligible for these facts"
    return plans


@pytest.mark.L0
def test_heuristics_propose_the_decode_tile_for_decode_and_mtp_shapes():
    """cga=1 leads (and is the only width) when S_q * pack_g <= 128; the packed
    leg leads; a bottom-right causal MTP graph walks NATURAL first (every unit
    has the same work), with the LPT variants behind for autotune."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

    decode = _plans(_facts())
    assert [(p.knobs.cga, p.knobs.pack_gqa, p.knobs.split_kv) for p in decode] == [(1, True, 1), (1, False, 1)]
    mtp = _plans(_facts(s_q=4, causal=True, bottom_right=True))
    assert all(p.knobs.cga == 1 for p in mtp)
    assert (mtp[0].knobs.pack_gqa, mtp[0].knobs.sched_policy) == (True, SCHED_NATURAL)
    assert [p.knobs.sched_policy for p in mtp if p.knobs.pack_gqa] == [SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2]
    # 96/8: G=12 does not divide the tile, so the group runs unpacked -- S_q=8 rows still fit one tile.
    glm = _plans(_facts(s_q=8, h_q=96, h_kv=8, causal=True, bottom_right=True))
    assert all((p.knobs.cga, p.knobs.pack_gqa) == (1, False) for p in glm)
    # d64 rides the d128 envelope, decode tile included.
    assert all(p.knobs.cga == 1 for p in _plans(_facts(h_kv=8, d_qk=64, d_v=64)))
    # MHA decode (no group to pack).
    assert all(p.knobs.cga == 1 for p in _plans(_facts(h_q=8, h_kv=8)))


@pytest.mark.L0
def test_heuristics_keep_the_prefill_tile_when_the_rows_overflow_one_tile():
    """S_q * G > 128, prefill shapes, THD and the other flavors stay on cga2."""
    assert all(p.knobs.cga == 2 for p in _plans(_facts(s_q=9)))  # 9 * 16 = 144 rows
    assert all(p.knobs.cga == 2 for p in _plans(_facts(s_q=512, h_q=8, h_kv=8, s_kv=512, padded=False)))
    assert all(p.knobs.cga == 2 for p in _plans(_facts(s_q=2048, causal=True, padded=False)))
    assert all(p.knobs.cga == 2 for p in _plans(_facts(thd=True)))
    assert all(p.knobs.cga == 2 for p in _plans(_facts(h_q=32, h_kv=2, d_qk=256, d_v=256)))
    # A causal prefill keeps its measured LPT_L2 primary: the decode NATURAL rule is decode-shaped only.
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    assert _plans(_facts(s_q=2048, causal=True, padded=False))[0].knobs.sched_policy == SCHED_LPT_L2


@pytest.mark.L0
def test_heuristics_split_the_decode_tile_on_an_underfilled_launch():
    """b=1, H_kv=1 over 32k tokens (dense, unpadded): one decode CTA per KV head;
    the wave model splits.  (A dense PADDED graph never splits -- the padded
    exclusion -- while a paged one does: test_graph_small_batch_decode_splits_kv.)"""
    plans = _plans(_facts(b=1, h_q=8, h_kv=1, s_kv=32768, padded=False))
    assert all(
        p.knobs.split_kv == 1 for p in _plans(_facts(b=1, h_q=8, h_kv=1, s_kv=32768, padded=True))
    ), "dense padded graphs keep the padded split exclusion"
    assert plans[0].knobs.cga == 1 and plans[0].knobs.split_kv > 1, plans[0].knobs
    assert any(p.knobs.split_kv == 1 for p in plans), "no-split stays reachable"


@pytest.mark.L0
def test_mismatch_admits_cga1_on_dense_d128_and_declines_it_for_thd():
    from cudnn.sdpa.fwd import engines

    spec = next(s for s in engines.ENGINE_SPECS if s.name == ENGINE)
    caps = spec.capabilities
    assert engines.mismatch(caps, _facts(), engines.SdpaFwdKnobs(cga=1)) is None
    assert engines.mismatch(caps, _facts(s_q=512, padded=False), engines.SdpaFwdKnobs(cga=1)) is None, "a pinned cga=1 is honored on any dense S_q"
    assert engines.mismatch(caps, _facts(), engines.SdpaFwdKnobs(cga=2)) is None
    thd = engines.mismatch(caps, _facts(thd=True), engines.SdpaFwdKnobs(cga=1))
    assert thd is not None and "decode tile" in thd and "THD" in thd
    assert engines.mismatch(caps, _facts(thd=True), engines.SdpaFwdKnobs(cga=2)) is None
    d256 = engines.mismatch(caps, _facts(h_q=32, h_kv=2, d_qk=256, d_v=256), engines.SdpaFwdKnobs(cga=1))
    assert d256 is not None and "outside this engine's domain" in d256
    # Every proposed set re-validates (honored or never listed).
    for f in (_facts(), _facts(s_q=4, causal=True, bottom_right=True), _facts(thd=True), _facts(s_q=9)):
        for p in _plans(f):
            assert engines.mismatch(caps, f, p.knobs) is None


# --- kernel template, direct -----------------------------------------------------


def _load_decode(params, tag):
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl

    path = os.path.join(os.path.dirname(os.path.abspath(api_dsl.__file__)), "kernels", "sm100", "decode_d128_f16.py")
    return load_template(path, params, tag=tag)


def _run_kernel(
    B,
    H,
    KH,
    P,
    max_pages,
    lens,
    hnd,
    splits,
    *,
    s_q=1,
    q_lens=None,
    causal_br=False,
    window_left=None,
    sink=False,
    stats_log2=False,
    pack=None,
    sched=0,
    dtype=torch.float16,
    d=128,
    paged=True,
    seed=0,
):
    """Drive sm100/decode_d128_f16.py directly (the split-KV suite's idiom).

    ``paged=False`` binds dense [B, S_kv, KH, d] K/V with per-batch KV lengths
    (the padded path) instead of pools + tables.  ``q_lens`` (per batch, <= s_q)
    compiles the dense padded-Q trim; None binds s_q for every batch.
    """
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    dev = "cuda"
    G = H // KH
    if pack is None:
        pack = G > 1 and 128 % G == 0
    scale = 1.0 / math.sqrt(d)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    q = torch.randn(B, s_q, H, d, device=dev, dtype=torch.float32, generator=gen).to(dtype)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    q_len_list = [s_q] * B if q_lens is None else list(q_lens)
    seq_q = torch.tensor(q_len_list, dtype=torch.int32, device=dev)
    sinks = torch.randn(H, device=dev, dtype=torch.float32, generator=gen) if sink else torch.zeros(H, dtype=torch.float32, device=dev)
    if paged:
        k_pool, v_pool, bt = _pools(B, KH, d, P, max_pages, hnd, dtype, seed)
        # kernel view: [num_pages, page_size, KH, d] (a permutation of the container)
        k_view, v_view = (k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3)) if hnd else (k_pool, v_pool)
        k_rows = [_gather_kv(k_pool, bt[b], lens[b], hnd) for b in range(B)]
        v_rows = [_gather_kv(v_pool, bt[b], lens[b], hnd) for b in range(B)]
        paged_kw = dict(paged_kv=True, page_size=P)
        compile_kw = dict(k_stride=tuple(k_view.stride()), v_stride=tuple(v_view.stride()))
        skv = 0
    else:
        skv = max_pages * P
        k_view = torch.randn(B, skv, KH, d, device=dev, dtype=torch.float32, generator=gen).to(dtype)
        v_view = torch.randn(B, skv, KH, d, device=dev, dtype=torch.float32, generator=gen).to(dtype)
        k_rows = [k_view[b, : lens[b]] for b in range(B)]
        v_rows = [v_view[b, : lens[b]] for b in range(B)]
        bt = None
        paged_kw = {}
        compile_kw = {}
    params = TemplateParams(
        dtype_qkv=3 if dtype == torch.float16 else 2,
        window_left=window_left,
        window_right=0 if causal_br else None,
        bottom_right=causal_br,
        has_sink=sink,
        stats_log2=stats_log2 and splits == 1,
        seq_kv_lens_present=True,
        seq_q_lens_present=q_lens is not None,
        sched_policy=sched,
        split_kv=splits,
        cta_mma=1,
        pack_gqa=pack,
        qh_per_kh=G if pack else 1,
        **paged_kw,
    )
    mod = _load_decode(
        params,
        tag=f"decode_test_{'p' + str(P) if paged else 'dense'}_s{splits}_g{G if pack else 1}_{dtype}_br{int(causal_br)}_w{window_left}_sk{int(sink)}_l2{int(stats_log2)}_q{int(q_lens is not None)}_sc{sched}",
    )
    assert mod.CGA_TILE_M == 128 and mod.CFG.STAGES_KV == 3 and mod.CFG.TOTAL_WARPS == 12
    fn = mod.compile(b=B, qh=H, kh=KH, sq=s_q, skv=skv, d_qk=d, d_v=d, has_lse=True, **compile_kw)
    o_p = torch.zeros(splits * B, s_q, H, d, device=dev, dtype=torch.float32 if splits > 1 else dtype)
    lse_p = torch.zeros(splits * B, H, s_q, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    kwargs = {}
    if splits > 1:
        kwargs["o_partial_f32"] = o_p
    if paged:
        kwargs.update(block_table_tensor=bt, block_table_v_tensor=bt)
    fn(
        q,
        k_view,
        v_view,
        o_p,
        lse_p,
        sinks,
        seq_lens,
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, KH, s_q, skv, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        int(seq_q.data_ptr()) if q_lens is not None else 0,
        **kwargs,
        stream=stream,
    )
    if splits == 1:
        o_out, lse_out = o_p, lse_p
    else:
        o_out = torch.zeros(B, s_q, H, d, device=dev, dtype=dtype)
        lse_out = torch.zeros(B, H, s_q, device=dev, dtype=torch.float32)
        cfn = comb.compile(
            b=B,
            h=H,
            sq=s_q,
            d_v=d,
            splits=splits,
            dtype_o="f16" if dtype == torch.float16 else "bf16",
            has_lse=True,
            dtype_partial="f32",
            stats_log2=stats_log2,
        )
        cfn(o_p, lse_p, o_out, lse_out, None, None, (B, H, s_q, d), cutlass.Int32(splits), stream=stream)
    torch.cuda.synchronize()
    assert not torch.isnan(o_out).any(), "NaN in O"
    for b in range(B):
        ref_o, ref_lse = _ref(q[b], k_rows[b], v_rows[b], q_len_list[b], scale, causal_br=causal_br, window_left=window_left, sink=sinks if sink else None)
        if stats_log2:
            ref_lse = ref_lse * math.log2(math.e)
        torch.testing.assert_close(o_out[b].float(), ref_o, atol=_tol(dtype), rtol=0, msg=f"O mismatch in batch {b} (L={lens[b]}, q_len={q_len_list[b]})")
        got = lse_out[b]
        live = ~torch.isinf(ref_lse)
        torch.testing.assert_close(got[live], ref_lse[live], atol=5e-3, rtol=0, msg=f"LSE mismatch in batch {b}")
        assert torch.isinf(got[~live]).all() and (got[~live] < 0).all(), f"dead rows must carry LSE = -inf (batch {b})"
        assert o_out[b].float()[~live.T].abs().max().item() == 0.0 if (~live).any() else True, f"dead rows must write O = 0 (batch {b})"
    assert SCHED_NATURAL == 0
    return mod


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128, 256])
def test_decode_kernel_paged_mixed_lengths(hnd, page_size):
    """Every page geometry the contract admits (several boxes per tile, one box
    per tile, one box inside a taller page): GQA 8:2 packed, lengths incl. 0, 1,
    a tile-unaligned tail and a length on a page/tile boundary."""
    _run_kernel(5, 8, 2, page_size, -(-1100 // page_size), [300, 77, 0, 1, 1024], hnd, splits=1)


@pytest.mark.L0
@pytest.mark.parametrize("group", [4, 16, 128], ids=["G4", "G16", "G128"])
def test_decode_kernel_pack_gqa_groups(group):
    """Packed groups 4 (32 tokens per tile), 16 (8) and 128 (one token: MQA at
    H=128); bf16; lengths across the batch differ so every packed row's mask
    is exercised."""
    _run_kernel(4, group, 1, 16, 20, [17, 320, 1, 129], hnd=True, splits=1, dtype=torch.bfloat16)


@pytest.mark.L0
def test_decode_kernel_unpacked_paths():
    """PackGQA off on a packable group (the heuristics' runner-up) and a group
    that does not divide the tile (H/H_kv = 12: the GLM 96/8 shape, served unpacked)."""
    _run_kernel(2, 8, 2, 16, 8, [50, 128], hnd=False, splits=1, pack=False)
    _run_kernel(2, 24, 2, 32, 8, [100, 256], hnd=True, splits=1)  # G=12 -> unpacked by the pack rule


@pytest.mark.L0
@pytest.mark.parametrize("splits", [2, 8])
def test_decode_kernel_forced_splits_empty_ranges(splits):
    """3 batches x 8 heads x 8 splits = 192 CTAs > SM count with [4000, 1, 129]:
    empty split ranges interleaved with live tiles under the persistent
    scheduler (the Q/O-alias parity hazard the prefill kernel's cga1 arm fixed)."""
    _run_kernel(3, 8, 8, 128, 33, [4000, 1, 129], hnd=True, splits=splits)


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [2, 4, 8])
def test_decode_kernel_mtp_bottom_right_with_q_trim(s_q):
    """MTP: s_q query tokens per sequence under bottom-right causal; per-batch Q
    lengths below s_q (the dense padded-Q trim: rows past the length write
    O := 0 / LSE := -inf, and the diagonal anchors at (q_len_b, L_b))."""
    q_lens = [s_q, max(1, s_q // 2), s_q, 1]
    _run_kernel(4, 16, 4, 16, 30, [400, 77, 1, 129], hnd=True, splits=1, s_q=s_q, q_lens=q_lens, causal_br=True)


@pytest.mark.L0
def test_decode_kernel_mtp_split_kv():
    """MTP s_q=4 bottom-right causal with a forced 4-way split (the small-batch lever)."""
    _run_kernel(2, 8, 2, 16, 64, [1000, 513], hnd=False, splits=4, s_q=4, causal_br=True)


@pytest.mark.L0
@pytest.mark.parametrize("window_left", [63, 200])
def test_decode_kernel_sliding_window(window_left):
    """Bottom-right causal + sliding window (gpt-oss style), window shorter and
    longer than a KV tile; lengths shorter than the window too."""
    _run_kernel(3, 8, 2, 16, 40, [600, 30, 129], hnd=True, splits=1, s_q=2, causal_br=True, window_left=window_left)


@pytest.mark.L0
def test_decode_kernel_dense_padded_sink_and_base2_stats():
    """The dense (non-paged) padded path with an attention sink (the paged row
    declines sink today, so the sink fold is exercised densely), and base-2
    Stats on the same shape without the sink."""
    _run_kernel(3, 8, 2, 128, 4, [300, 77, 0], hnd=False, splits=1, paged=False, sink=True)
    _run_kernel(3, 8, 2, 128, 4, [300, 77, 512], hnd=False, splits=1, paged=False, stats_log2=True)
    _run_kernel(2, 8, 2, 16, 20, [300, 77], hnd=True, splits=1, stats_log2=True)


@pytest.mark.L0
def test_decode_kernel_d64_envelope():
    """d=64 (gpt-oss) rides the decode tile zero-padded: real extents on the descriptors."""
    _run_kernel(3, 8, 1, 16, 20, [300, 77, 1], hnd=False, splits=1, d=64, dtype=torch.bfloat16)


@pytest.mark.L0
@pytest.mark.parametrize("sched", [1, 2], ids=["LPT", "LPT_L2"])
def test_decode_kernel_lpt_schedulers(sched):
    """The flattened LPT grids decode correctly at cga1 (the heuristics offer
    them as autotune runners for causal MTP)."""
    _run_kernel(3, 16, 4, 16, 30, [400, 77, 129], hnd=True, splits=1, s_q=4, causal_br=True, sched=sched)


@pytest.mark.L0
def test_decode_kernel_template_rejects_off_contract_params():
    """The template's config backstop fires at load for what the gates upstream
    decline: cga2 and THD have no decode-tile arm."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    with pytest.raises(ValueError, match="cga1 only"):
        _load_decode(TemplateParams(dtype_qkv=3, cta_mma=2), tag="decode_reject_cga2")
    with pytest.raises(ValueError, match="THD"):
        _load_decode(TemplateParams(dtype_qkv=3, cta_mma=1, thd_varlen=True, seq_kv_lens_present=True), tag="decode_reject_thd")


# --- graph API ----------------------------------------------------------------------


def _paged_graph(B, H, KH, d, P, max_pages, lens, hnd, *, s_q=1, causal_br=False, stats=False, dtype=torch.float16, pin_cga=None, seed=0):
    """Build + run cuDNN's paged-cache graph; return (plan, o [B,H,s_q,d], stats or None, inputs)."""
    import cudnn
    import cudnn.sdpa  # noqa: F401 — registers the FROST engines

    dev = "cuda"
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, bt = _pools(B, KH, d, P, max_pages, hnd, dtype, seed)
    k_c, v_c = (k_pool, v_pool) if hnd else (k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3))
    bt4 = bt.view(B, 1, max_pages, 1)
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    q_gpu = torch.randn(B, s_q, H, d, device=dev, dtype=torch.float32, generator=gen).to(dtype).transpose(1, 2)
    o_gpu = torch.empty(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    slk = seq_lens.view(B, 1, 1, 1)
    slq = torch.full((B, 1, 1, 1), s_q, dtype=torch.int32, device=dev)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(q_gpu), g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt4), g.tensor_like(bt4)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    kw = dict(use_causal_mask_bottom_right=True) if causal_br else {}
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
        paged_attention_max_seq_len_kv=max_pages * P,
        **kw,
    )
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, ENGINE)
    if pin_cga is not None:
        # The heuristics list ONE width for a shape (cga=1 for decode shapes), so
        # the other width is pinned the way an autotuner replays a knob record:
        # an explicit (engine_id, knobs) plan, strict on decline.
        from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

        assert all(
            g.plans[i].knobs.cga != pin_cga for i in range(len(g.plans)) if g.get_plan_name_at_index(i).startswith(ENGINE)
        ), "cga already proposed; pin is moot"
        g.create_execution_plan(_SM100_ID, SdpaFwdKnobs(sched_policy=0, tile_m=128, tile_n=128, cga=pin_cga, pack_gqa=plan.knobs.pack_gqa, split_kv=1))
        g.select_plan(len(g.plans) - 1)
        plan = g.plans[len(g.plans) - 1]
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    vp = {q: q_gpu, k: k_c, v: v_c, tk: bt4, tv: bt4, sq_t: slq, sk_t: slk, o: o_gpu}
    if stats:
        vp[st] = stats_gpu
    # Rule 3: the FROST execute path reads the per-batch lengths on device.
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for b in range(B):
        ref_o, ref_lse = _ref(
            q_gpu[b].transpose(0, 1), _gather_kv(k_pool, bt[b], lens[b], hnd), _gather_kv(v_pool, bt[b], lens[b], hnd), s_q, scale, causal_br=causal_br
        )
        got = o_gpu[b].transpose(0, 1).float()  # [s_q, H, d]
        assert not torch.isnan(got).any()
        torch.testing.assert_close(got, ref_o, atol=_tol(dtype), rtol=0, msg=f"batch {b}")
        if stats:
            got_lse = stats_gpu[b, :, :, 0]
            live = ~torch.isinf(ref_lse)
            torch.testing.assert_close(got_lse[live], ref_lse[live], atol=5e-3, rtol=0)
            assert torch.isinf(got_lse[~live]).all()
    return plan


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
def test_graph_decode_selects_the_decode_tile(hnd):
    """S_q=1 GQA 8:2 over a paged cache: the heuristics' first plan is the
    decode tile (TILE_CGA_M=1, packed), Stats out, lengths incl. 0 and 1."""
    plan = _paged_graph(5, 8, 2, D, 16, 70, [300, 77, 0, 1, 1024], hnd, stats=True)
    assert (plan.knobs.cga, plan.knobs.pack_gqa, plan.knobs.split_kv) == (1, True, 1), plan.knobs


@pytest.mark.L0
def test_graph_mtp_bottom_right_selects_the_decode_tile_natural():
    """S_q=4 bottom-right causal (MTP) at 64/4: decode tile, NATURAL scheduler, packed."""
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL

    plan = _paged_graph(4, 64, 4, D, 16, 40, [600, 4, 129, 640], hnd=True, s_q=4, causal_br=True, stats=True, dtype=torch.bfloat16)
    assert (plan.knobs.cga, plan.knobs.pack_gqa, plan.knobs.sched_policy) == (1, True, SCHED_NATURAL), plan.knobs


@pytest.mark.L0
def test_graph_prefill_shape_keeps_the_prefill_tile():
    """S_q=64 with GQA 8:2 (256 packed rows) over a paged cache is not
    decode-shaped: the prefill pipeline (cga2) is proposed and serves it."""
    plan = _paged_graph(2, 8, 2, D, 16, 8, [100, 128], hnd=False, s_q=64)
    assert plan.knobs.cga == 2, plan.knobs


@pytest.mark.L0
def test_graph_pinned_cga2_on_a_decode_shape_is_honored():
    """A caller pinning TILE_CGA_M=2 on a decode shape gets the prefill tile
    (a knob is honored, never substituted), and it matches the reference."""
    plan = _paged_graph(3, 8, 2, D, 32, 40, [1000, 1, 1279], hnd=True, pin_cga=2)
    assert plan.knobs.cga == 2


@pytest.mark.L0
def test_graph_decode_d64_envelope_bf16():
    """d=64 GQA 8:2 decode rides the decode tile zero-padded."""
    plan = _paged_graph(6, 8, 2, 64, 64, 4, [200, 3, 256, 1, 100, 255], hnd=False, dtype=torch.bfloat16)
    assert plan.knobs.cga == 1


@pytest.mark.L0
def test_graph_small_batch_decode_splits_kv():
    """B=1, H_kv=1, 32k tokens: one decode CTA per KV head — the wave model
    splits, the partials recombine (Stats checked)."""
    plan = _paged_graph(1, 8, 1, D, 16, 2048, [32000], hnd=True, stats=True)
    assert plan.knobs.cga == 1 and plan.knobs.split_kv > 1, plan.knobs


@pytest.mark.L0
def test_graph_thd_queries_never_get_the_decode_tile():
    """Ragged (THD) queries over a paged cache: no cga=1 plan is offered for the
    engine (the decode tile has no THD leg), and a pinned cga=1 declines."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    dev, dtype = "cuda", torch.float16
    H, KH, d, P, max_pages = 8, 2, D, 16, 20
    q_lens, kv_lens = [37, 130, 5], [300, 77, 129]
    B, T, S_max = len(q_lens), sum(q_lens), max(q_lens)
    cu = [0]
    for s in q_lens:
        cu.append(cu[-1] + s)
    k_pool, v_pool, bt = _pools(B, KH, d, P, max_pages, True, dtype)
    bt4 = bt.view(B, 1, max_pages, 1)
    stride = (S_max * H * d, d, H * d, 1)
    q_stor = torch.randn(B * S_max * H * d, device=dev, dtype=dtype)
    q_gpu = q_stor.as_strided((B, H, S_max, d), stride)
    slq = torch.tensor(q_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(kv_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro = (torch.tensor(cu, dtype=torch.int64, device=dev) * H * d).view(B + 1, 1, 1, 1)

    def _build():
        g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tq = g.tensor(dim=[B, H, S_max, d], stride=list(stride), data_type=cudnn.data_type.HALF, name="q")
        k, v = g.tensor_like(k_pool), g.tensor_like(v_pool)
        tk, tv = g.tensor_like(bt4), g.tensor_like(bt4)
        sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
        qro, oro = g.tensor_like(ro), g.tensor_like(ro)
        tq.set_ragged_offset(qro)
        o, _ = g.sdpa(
            name="sdpa",
            q=tq,
            k=k,
            v=v,
            generate_stats=False,
            attn_scale=1.0 / math.sqrt(d),
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
        return g

    g = _build()
    g.create_execution_plans([cudnn.heur_mode.A])
    assert offers_engine(g, ENGINE)
    cgas = {g.plans[i].knobs.cga for i in range(len(g.plans)) if g.get_plan_name_at_index(i).startswith(ENGINE)}
    assert cgas == {2}, f"THD must stay on the prefill tile; offered cga domain {cgas}"
    # Pinned cga=1 on the THD graph is a typed decline, not a silent substitution.
    g2 = _build()
    g2.create_execution_plan(_SM100_ID, SdpaFwdKnobs(sched_policy=0, tile_m=128, tile_n=128, cga=1, pack_gqa=False, split_kv=1))
    g2.select_plan(0)
    with pytest.raises((cudnn.cudnnGraphNotSupportedError, NotImplementedError, ValueError), match="decode tile|THD"):
        g2.check_support()
        g2.build_plans()


@pytest.mark.L0
def test_graph_dense_padded_decode_bf16():
    """A non-paged padded decode graph (dense K/V with per-batch KV lengths,
    S_q=1, GQA 16:2, bf16): the decode tile serves the dense contract too."""
    import cudnn
    import cudnn.sdpa  # noqa: F401

    dev, dtype = "cuda", torch.bfloat16
    B, H, KH, S_kv = 4, 16, 2, 1024
    lens = [1024, 1, 700, 129]
    gen = torch.Generator(device=dev).manual_seed(11)
    q_gpu = torch.randn(B, 1, H, D, device=dev, dtype=torch.float32, generator=gen).to(dtype).transpose(1, 2)
    k_gpu = torch.randn(B, S_kv, KH, D, device=dev, dtype=torch.float32, generator=gen).to(dtype).transpose(1, 2)
    v_gpu = torch.randn(B, S_kv, KH, D, device=dev, dtype=torch.float32, generator=gen).to(dtype).transpose(1, 2)
    o_gpu = torch.empty(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    slk = torch.tensor(lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slq = torch.ones(B, 1, 1, 1, dtype=torch.int32, device=dev)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(q_gpu), g.tensor_like(k_gpu), g.tensor_like(v_gpu)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, generate_stats=False, attn_scale=1.0 / math.sqrt(D), use_padding_mask=True, seq_len_q=sq_t, seq_len_kv=sk_t)
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, ENGINE)
    assert (plan.knobs.cga, plan.knobs.pack_gqa) == (1, True), plan.knobs
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute({q: q_gpu, k: k_gpu, v: v_gpu, sq_t: slq, sk_t: slk, o: o_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for b in range(B):
        ref_o, _ = _ref(q_gpu[b].transpose(0, 1), k_gpu[b].transpose(0, 1)[: lens[b]], v_gpu[b].transpose(0, 1)[: lens[b]], 1, 1.0 / math.sqrt(D))
        torch.testing.assert_close(o_gpu[b].transpose(0, 1).float(), ref_o, atol=_tol(dtype), rtol=0)


@pytest.mark.L0
def test_adapter_decode_tile_cuda_graph_replay_no_host_sync():
    """The adapter at cga=1 (the decode tile) with a 2-way split, captured once
    and replayed with different per-batch lengths under
    ``set_sync_debug_mode("error")`` (Rule 3): no D2H, correct on every replay."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, max_pages = 8, 16, 4, 16, 64
    dev, dtype = "cuda", torch.float16
    k_pool, v_pool, bt = _pools(B, KH, D, P, max_pages, False, dtype, seed=1)
    k_c, v_c = k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3)
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
        split_kv=2,
        cga=1,
        pack_gqa=True,
    )
    api.check_support()
    api.compile()
    assert api._k_mod.CGA_TILE_M == 128 and api._k_mod.CFG.TILES_Q == 1, "cga=1 must have loaded the decode tile"
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
        for b in range(B):
            ref_o, ref_lse = _ref(
                q_gpu[b].transpose(0, 1), _gather_kv(k_pool, bt[b], new_lens[b], False), _gather_kv(v_pool, bt[b], new_lens[b], False), 1, scale
            )
            torch.testing.assert_close(o_gpu[b].transpose(0, 1).float(), ref_o, atol=2e-2, rtol=0)
            live = ~torch.isinf(ref_lse)
            torch.testing.assert_close(lse[b][live], ref_lse[live], atol=5e-3, rtol=0)
