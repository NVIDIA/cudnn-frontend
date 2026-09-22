# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST SM100 d128 DECODE tile (``sm100/decode_d128_f16.py``).

``sdpa_fwd_prefill_sm100``'s (128, 128) f16/bf16 flavor has two tiles behind
one knob: ``TILE_CGA_M=2`` is the prefill pipeline (512 Q rows per cga2
cluster) and ``TILE_CGA_M=1`` the decode tile (128 rows per independent CTA,
one softmax warpgroup, three KV stages).  The heuristics propose cga=1 exactly
when one 128-row tile covers a packed head's Q rows -- ``S_q * pack_g <= 128``
with ``pack_g`` the CANDIDATE's own packing (the packed group ``Cfg.PACK_G`` on
the packed leg: G, or its largest divisor of 128 under partial PackGQA; 1 on
the unpacked one): S_q = 1 decode and MTP -- and cga=2 otherwise.

Three tiers:

- GPU-free: the config backstop (accept / reject), the standalone cga domain,
  the heuristics' cga and scheduler rules, and the ``mismatch`` gates (a THD
  graph gets the decode tile only as the ragged-Q-over-paged-KV leg at
  S_q(max) == 1 -- ``engines._thd_decode_leg`` -- and keeps the prefill tile
  otherwise).
- Kernel template, direct: paged pools at every page geometry, mixed lengths
  incl. 0 and 1, PackGQA on / off / partial (a group that does not divide the
  tile packs its largest divisor that does; one sharing no factor with it runs
  unpacked), forced splits with
  empty ranges, bottom-right causal MTP with per-batch Q lengths (the trim),
  sliding window, sink, base-2 stats, the d64 envelope, the LPT schedulers, and
  the keyless-row sink contract (O := 0, LSE := sink at any sink magnitude).
- Graph API: decode / MTP shapes select the decode tile (partially packed GQA
  groups included), a prefill shape keeps
  the prefill tile, a pinned cga=2 on a decode shape is honored, a chunked-prefill
  THD graph declines cga=1 while FlashInfer's ragged paged graph at one token per
  sequence rides the ragged-Q leg (int32 / int64 offsets, an empty sequence, no
  Stats), dense (non-paged) padded decode, a dense MTP graph whose keyless rows
  carry a sink, a small-batch split, and CUDA-graph replay under
  ``set_sync_debug_mode("error")`` (Rule 3).
"""

import math
import os

import pytest
import torch

from frost_test_utils import launch_f16, offers_engine, requires_dsl, requires_pre_rubin_blackwell, select_engine

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
    MXFP8 experiment axes (each a constraint the engine gates uphold upstream).
    PackGQA follows the d128 prefill tile's partial contract: a group that does
    not divide the tile packs its largest divisor that does (``PACK_G``); one
    sharing no factor with it is declined (inverted from the whole-group-only
    pin this tile carried before partial PackGQA landed)."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
    from cudnn.sdpa.fwd.config_sm100 import CfgD128Decode, TemplateParams, _SM100_MAX_DYN_SMEM, _d128_smem_bytes, cga_tile_m, make_cfg_d128_decode

    for dtype in (DTYPE_FP16, DTYPE_BF16):
        cfg, tma = make_cfg_d128_decode(
            TemplateParams(dtype_qkv=dtype, cta_mma=1, seq_kv_lens_present=True, paged_kv=True, page_size=16, pack_gqa=True, qh_per_kh=16)
        )
        assert isinstance(cfg, CfgD128Decode)
        assert (cfg.CTA_MMA, cfg.CGA_M, cfg.TILES_Q, cfg.STAGES_KV, cfg.SOFTMAX_WARPGROUPS, cfg.TOTAL_WARPS, cfg.QO_ALIAS) == (1, 1, 1, 3, 1, 12, 1)
        assert cfg.READ_TILE_ARRIVERS == 11 and cfg.PAGED_KV == 1 and cfg.PAGE_SIZE == 16 and cfg.PACK_GQA == 1 and cfg.PACK_G == 16
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
    # Partial PackGQA: 96/8 (G=12) packs 4 heads per token row-group, 48/8 (G=6)
    # packs 2, MQA 256/1 packs a whole 128-row tile; G=3 shares no factor and is declined.
    for g, pack_g in ((12, 4), (6, 2), (256, 128), (16, 16), (1, 1)):
        cfg, _ = make_cfg_d128_decode(TemplateParams(cta_mma=1, pack_gqa=True, qh_per_kh=g))
        assert (cfg.PACK_GQA, cfg.QH_PER_KH, cfg.PACK_G) == (1, g, pack_g), (g, cfg.PACK_G)
    assert make_cfg_d128_decode(TemplateParams(cta_mma=1, pack_gqa=False, qh_per_kh=12))[0].PACK_G == 1
    with pytest.raises(ValueError, match="PackGQA|share a factor"):
        make_cfg_d128_decode(TemplateParams(cta_mma=1, pack_gqa=True, qh_per_kh=3))


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
    """cga=1 leads (and is the only width) when S_q * pack_g <= 128 on every
    leg; the packed leg leads; a bottom-right causal MTP graph walks NATURAL
    first (every unit has the same work), with the LPT variants behind for
    autotune."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

    decode = _plans(_facts())
    assert [(p.knobs.cga, p.knobs.pack_gqa, p.knobs.split_kv) for p in decode] == [(1, True, 1), (1, False, 1)]
    mtp = _plans(_facts(s_q=4, causal=True, bottom_right=True))
    assert all(p.knobs.cga == 1 for p in mtp)
    assert (mtp[0].knobs.pack_gqa, mtp[0].knobs.sched_policy) == (True, SCHED_NATURAL)
    assert [p.knobs.sched_policy for p in mtp if p.knobs.pack_gqa] == [SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2]
    # 96/8: G=12 does not divide the tile, so the packed leg carries PACK_G=4 heads
    # per token (partial PackGQA) -- 8 * 4 = 32 rows fit one tile and the packed leg
    # leads; the unpacked runner-up (8 rows) fits too.  (Before partial PackGQA the
    # group ran unpacked on the decode tile; inverted, not deleted.)
    glm = _plans(_facts(s_q=8, h_q=96, h_kv=8, causal=True, bottom_right=True))
    assert (glm[0].knobs.cga, glm[0].knobs.pack_gqa) == (1, True), glm[0].knobs
    assert all(p.knobs.cga == 1 for p in glm) and False in {p.knobs.pack_gqa for p in glm}, [p.knobs for p in glm]
    # The fit is judged on PACK_G, not G: 96/8 at S_q=33 overflows the packed tile
    # (33 * 4 = 132 > 128) while 33 unpacked rows fit; at S_q=32 both legs fit.
    glm33 = _plans(_facts(s_q=33, h_q=96, h_kv=8))
    assert all(p.knobs.cga == (2 if p.knobs.pack_gqa else 1) for p in glm33), [p.knobs for p in glm33]
    assert all(p.knobs.cga == 1 for p in _plans(_facts(s_q=32, h_q=96, h_kv=8)))
    # 24/8: G=3 shares no factor with the tile -- unpacked only, decode tile.
    assert all((p.knobs.cga, p.knobs.pack_gqa) == (1, False) for p in _plans(_facts(h_q=24, h_kv=8)))
    # d64 rides the d128 envelope, decode tile included.
    assert all(p.knobs.cga == 1 for p in _plans(_facts(h_kv=8, d_qk=64, d_v=64)))
    # MHA decode (no group to pack).
    assert all(p.knobs.cga == 1 for p in _plans(_facts(h_q=8, h_kv=8)))


@pytest.mark.L0
def test_heuristics_keep_the_prefill_tile_when_the_rows_overflow_one_tile():
    """S_q * G > 128 on the packed leg, prefill shapes, THD and the other
    flavors stay on cga2.  The unpacked runner-up of an overflowing packed leg
    is judged on ITS rows (S_q per head): see
    test_heuristics_decode_geometry_follows_the_selected_packing."""
    nine = _plans(_facts(s_q=9))  # 9 * 16 = 144 packed rows; 9 unpacked
    assert all(p.knobs.cga == 2 for p in nine if p.knobs.pack_gqa), "the packed legs overflow one tile"
    assert all(p.knobs.cga == 1 for p in nine if not p.knobs.pack_gqa), "the unpacked runner-up fits one tile"
    assert all(p.knobs.cga == 2 for p in _plans(_facts(s_q=512, h_q=8, h_kv=8, s_kv=512, padded=False)))
    assert all(p.knobs.cga == 2 for p in _plans(_facts(s_q=2048, causal=True, padded=False)))
    assert all(p.knobs.cga == 2 for p in _plans(_facts(thd=True)))
    assert all(p.knobs.cga == 2 for p in _plans(_facts(h_q=32, h_kv=2, d_qk=256, d_v=256)))
    # A causal prefill keeps its measured LPT_L2 primary: the decode NATURAL rule is decode-shaped only.
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    assert _plans(_facts(s_q=2048, causal=True, padded=False))[0].knobs.sched_policy == SCHED_LPT_L2


@pytest.mark.L0
def test_heuristics_decode_geometry_follows_the_selected_packing():
    """Review (PR #1094): the decode-tile fit is decided per CANDIDATE from its
    own packing, not once per graph from PackGQA eligibility.  G=16, S_q=16:
    the packed leg carries 256 rows (prefill tile, and it still leads), the
    unpacked runner-up 16 rows per head (decode tile) -- before the fix both
    rode cga2 because the fit was computed as if every candidate were packed.
    At S_q > 128 no leg fits; the heur_mode B fallback (unpacked, unsplit)
    follows its own rows too; every emitted set re-validates."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.heuristics import recommend

    plans = _plans(_facts(s_q=16))
    legs = [(p.knobs.pack_gqa, p.knobs.cga) for p in plans]
    assert legs[0] == (True, 2), legs
    assert (False, 1) in legs and (False, 2) not in legs and (True, 1) not in legs, legs
    assert all(p.knobs.cga == 2 for p in _plans(_facts(s_q=129))), "129 rows overflow one tile unpacked too"
    fallback = recommend("B", _facts(s_q=16), {ENGINE: _SM100_ID})
    assert [(p.knobs.pack_gqa, p.knobs.cga) for p in fallback] == [(False, 1)], fallback
    # MHA has no group: one rule for the only leg.
    assert all(p.knobs.cga == 1 for p in _plans(_facts(s_q=64, h_q=8, h_kv=8)))
    assert all(p.knobs.cga == 2 for p in _plans(_facts(s_q=129, h_q=8, h_kv=8)))
    caps = next(spec for spec in engines.ENGINE_SPECS if spec.name == ENGINE).capabilities
    for f in (_facts(s_q=16), _facts(s_q=9, causal=True, bottom_right=True)):
        for p in _plans(f):
            assert engines.mismatch(caps, f, p.knobs) is None, p.knobs


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
    # Partial PackGQA rides the decode tile: a pinned cga=1 + PACK_GQA=1 on 96/8
    # (G=12 packs 4) is honored; on 24/8 (G=3, no common factor) it is declined.
    assert engines.mismatch(caps, _facts(h_q=96, h_kv=8), engines.SdpaFwdKnobs(cga=1, pack_gqa=True)) is None
    g3 = engines.mismatch(caps, _facts(h_q=24, h_kv=8), engines.SdpaFwdKnobs(cga=1, pack_gqa=True))
    assert g3 is not None and "share a factor" in g3, g3
    # Every proposed set re-validates (honored or never listed).
    for f in (_facts(), _facts(s_q=4, causal=True, bottom_right=True), _facts(thd=True), _facts(s_q=9), _facts(s_q=8, h_q=96, h_kv=8)):
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
    has_lse=True,
    pack=None,
    sched=0,
    dtype=torch.float16,
    d=128,
    paged=True,
    seed=0,
    expect_pack_g=None,
):
    """Drive sm100/decode_d128_f16.py directly (the split-KV suite's idiom).

    ``paged=False`` binds dense [B, S_kv, KH, d] K/V with per-batch KV lengths
    (the padded path) instead of pools + tables.  ``q_lens`` (per batch, <= s_q)
    compiles the dense padded-Q trim; None binds s_q for every batch.  ``sink``
    is ``True`` for one randn logit per head or a number that pins every head to
    that logit (the underflow regression needs -120, far below anything randn
    draws).  ``has_lse=False`` compiles the Stats store out (unsplit only).
    ``pack=None`` packs whenever the group shares a factor with the 128-row
    tile (the partial-PackGQA contract the kernel carries); ``expect_pack_g``
    pins the config's packed group.
    """
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128_decode, pack_gqa_group_size
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    dev = "cuda"
    G = H // KH
    if pack is None:
        pack = pack_gqa_group_size(G, 128, partial=True) > 1
    scale = 1.0 / math.sqrt(d)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    q = torch.randn(B, s_q, H, d, device=dev, dtype=torch.float32, generator=gen).to(dtype)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    q_len_list = [s_q] * B if q_lens is None else list(q_lens)
    seq_q = torch.tensor(q_len_list, dtype=torch.int32, device=dev)
    if sink is True:
        sinks = torch.randn(H, device=dev, dtype=torch.float32, generator=gen)
    elif sink is False or sink is None:
        sinks = torch.zeros(H, dtype=torch.float32, device=dev)
    else:
        sinks = torch.full((H,), float(sink), dtype=torch.float32, device=dev)
    has_sink = sink is not False and sink is not None
    assert has_lse or (splits == 1 and not stats_log2), "the Stats store can only be compiled out of an unsplit, natural-base run"
    if paged:
        k_pool, v_pool, bt = _pools(B, KH, d, P, max_pages, hnd, dtype, seed)
        # kernel view: [num_pages, page_size, KH, d] (a permutation of the container)
        k_view, v_view = (k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3)) if hnd else (k_pool, v_pool)
        k_rows = [_gather_kv(k_pool, bt[b], lens[b], hnd) for b in range(B)]
        v_rows = [_gather_kv(v_pool, bt[b], lens[b], hnd) for b in range(B)]
        paged_kw = dict(paged_kv=True, page_size=P)
        skv = 0
    else:
        skv = max_pages * P
        k_view = torch.randn(B, skv, KH, d, device=dev, dtype=torch.float32, generator=gen).to(dtype)
        v_view = torch.randn(B, skv, KH, d, device=dev, dtype=torch.float32, generator=gen).to(dtype)
        k_rows = [k_view[b, : lens[b]] for b in range(B)]
        v_rows = [v_view[b, : lens[b]] for b in range(B)]
        bt = None
        paged_kw = {}
    params = TemplateParams(
        dtype_qkv=3 if dtype == torch.float16 else 2,
        window_left=window_left,
        window_right=0 if causal_br else None,
        bottom_right=causal_br,
        has_sink=has_sink,
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
    if expect_pack_g is not None:
        cfg, _ = make_cfg_d128_decode(params)
        assert cfg.PACK_G == expect_pack_g, f"PACK_G={cfg.PACK_G} for G={G}, expected {expect_pack_g}"
    mod = _load_decode(
        params,
        tag=f"decode_test_{'p' + str(P) if paged else 'dense'}_s{splits}_g{G if pack else 1}_{dtype}_br{int(causal_br)}_w{window_left}_sk{int(has_sink)}_l2{int(stats_log2)}_q{int(q_lens is not None)}_sc{sched}_lse{int(has_lse)}",
    )
    assert mod.CGA_TILE_M == 128 and mod.CFG.STAGES_KV == 3 and mod.CFG.TOTAL_WARPS == 12
    fn = mod.compile(d_qk=d, d_v=d, has_lse=has_lse, paged_hnd=paged and hnd)
    o_p = torch.zeros(splits * B, s_q, H, d, device=dev, dtype=torch.float32 if splits > 1 else dtype)
    lse_p = torch.zeros(splits * B, H, s_q, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    kwargs = {}
    if splits > 1:
        kwargs["o_partial_f32"] = o_p
    if paged:
        kwargs.update(block_table_tensor=bt, block_table_v_tensor=bt)
    launch_f16(
        fn,
        q,
        k_view,
        v_view,
        o_p,
        lse_p if has_lse else None,
        sinks,
        seq_lens,
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, KH, s_q, skv, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        int(seq_q.data_ptr()) if q_lens is not None else 0,
        **kwargs,
        page_size=P if paged else 0,
        stream=stream,
        host=mod._host,
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
        ref_o, ref_lse = _ref(q[b], k_rows[b], v_rows[b], q_len_list[b], scale, causal_br=causal_br, window_left=window_left, sink=sinks if has_sink else None)
        if stats_log2:
            ref_lse = ref_lse * math.log2(math.e)
        torch.testing.assert_close(o_out[b].float(), ref_o, atol=_tol(dtype), rtol=0, msg=f"O mismatch in batch {b} (L={lens[b]}, q_len={q_len_list[b]})")
        if not has_lse:
            continue
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
@pytest.mark.parametrize(
    "h,kh,pack_g,s_q,splits",
    [(24, 2, 4, 1, 1), (24, 2, 4, 4, 1), (24, 2, 4, 2, 2), (12, 2, 2, 1, 1), (6, 2, 1, 1, 1), (10, 2, 1, 1, 1), (256, 1, 128, 1, 1)],
    ids=["g12_packs4", "g12_packs4_mtp4_br", "g12_packs4_mtp2_split2", "g6_packs2", "g3_unpacked", "g5_unpacked", "g256_packs128"],
)
def test_decode_kernel_partial_pack_gqa_groups(h, kh, pack_g, s_q, splits):
    """Partial PackGQA on the decode tile (``CfgD128Decode.PACK_G``, the d128
    prefill tile's contract): a group that does not divide the 128-row tile packs
    its largest divisor that does -- G=12 (the GLM 96/8 shape) packs 4 heads per
    token row-group with three packed heads per KV head, G=6 packs 2 -- while a
    group sharing no factor with the tile (G=3, G=5) runs unpacked (PACK_G=1), and
    a group LARGER than the tile (256/1 MQA) packs a whole tile of 128 heads with
    two packed heads per KV head.  Bottom-right causal MTP keeps the diagonal on
    QH_PER_KH (the GQA ratio) while the bounds and the KV-head index run on
    PACK_G; a forced split chunks the PACKED token-span bounds.  Lengths include a
    1-token sequence.  (Before partial PackGQA this tile ran G=12 unpacked.)"""
    _run_kernel(
        2, h, kh, 16, 20, [100, 1] if s_q == 1 else [300, 1], hnd=True, splits=splits, s_q=s_q, causal_br=s_q > 1, expect_pack_g=pack_g, dtype=torch.bfloat16
    )


@pytest.mark.L0
def test_decode_kernel_unpacked_paths():
    """PackGQA off on a packable group (the heuristics' runner-up), and off on a
    partially packable one (H/H_kv = 12: the GLM 96/8 shape, whose packed leg now
    packs 4 -- see test_decode_kernel_partial_pack_gqa_groups -- and whose
    unpacked runner-up is this)."""
    _run_kernel(2, 8, 2, 16, 8, [50, 128], hnd=False, splits=1, pack=False)
    _run_kernel(2, 24, 2, 32, 8, [100, 256], hnd=True, splits=1, pack=False, expect_pack_g=1)  # G=12 unpacked runner-up
    # G=16 at S_q=16 unpacked: 16 rows per head in one tile -- the runner-up the
    # heuristics emit for a packed leg that overflows the tile (256 rows).
    _run_kernel(2, 16, 1, 16, 20, [17, 300], hnd=True, splits=1, s_q=16, pack=False, causal_br=True, dtype=torch.bfloat16)


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
@pytest.mark.parametrize("stats", ["stats", "stats_log2", "no_stats"])
@pytest.mark.parametrize("sink", [-120.0, -5.0, 3.0], ids=["sink_m120", "sink_m5", "sink_p3"])
def test_decode_kernel_keyless_rows_sink_magnitude(sink, stats):
    """Review regression (PR #1094, inherited from the prefill tile and fixed
    there by PR #1095): bf16, B = 1, 4/1 heads packed, paged page 16 HND, a
    128-key cache with ONE live key, S_q = 4 under the bottom-right causal
    diagonal, the sink pinned per head.  Three of the four rows have no key, so
    the sink is their whole mass: O := 0, LSE := sink (times log2(e) in base 2).
    The softmax publishes a 0-substituted row max with total_sum = 0 for such a
    row; a fold that COMPUTES the denominator from it gets exp(-120 - 0) = 0 in
    fp32 -> O = 0 * inf = NaN, LSE = 0 + log(0) = -inf.  -5 does not underflow
    (the fold happens to be right), +3 sits above the substituted max (the sink
    dominates; also right) -- the three pin the row's contract, not one
    arithmetic accident.  Stats on, in base 2, and compiled out (O alone)."""
    _run_kernel(
        1,
        4,
        1,
        16,
        8,
        [1],
        hnd=True,
        splits=1,
        s_q=4,
        causal_br=True,
        sink=sink,
        stats_log2=stats == "stats_log2",
        has_lse=stats != "no_stats",
        dtype=torch.bfloat16,
    )


@pytest.mark.L0
@pytest.mark.parametrize("paged", [True, False], ids=["paged", "dense"])
def test_decode_kernel_keyless_rows_very_negative_sink_with_empty_sequence(paged):
    """The other keyless shape: seq_len_kv = 0 (every row keyless, the tile's
    empty-range arm) next to the one-key and a full batch, sink -120, GQA 8:2
    packed, paged NHD and dense padded -- the empty batch is keyless on all four
    rows, the one-key batch on three, the full one on none."""
    _run_kernel(3, 8, 2, 16, 8, [1, 0, 128], hnd=False, splits=1, s_q=4, causal_br=True, sink=-120.0, paged=paged, dtype=torch.bfloat16)


@pytest.mark.L0
def test_decode_kernel_d64_envelope():
    """d=64 (gpt-oss) rides the decode tile zero-padded: real extents on the descriptors."""
    _run_kernel(3, 8, 1, 16, 20, [300, 77, 1], hnd=False, splits=1, d=64, dtype=torch.bfloat16)


@pytest.mark.L0
@pytest.mark.parametrize("sched", [1, 2], ids=["LPT", "LPT_L2"])
def test_decode_kernel_lpt_schedulers(sched):
    """The flattened LPT grids decode correctly at cga1 (the heuristics offer
    them as autotune runners for causal MTP) -- for a whole-group packing (16/4)
    and a partial one (24/2: G=12 packs 4, so LPT_L2 groups the three packed
    heads that read one KV head, a grouping the whole-group case never has)."""
    _run_kernel(3, 16, 4, 16, 30, [400, 77, 129], hnd=True, splits=1, s_q=4, causal_br=True, sched=sched)
    _run_kernel(3, 24, 2, 16, 30, [400, 77, 129], hnd=True, splits=1, s_q=4, causal_br=True, sched=sched, expect_pack_g=4)


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


def _paged_graph(
    B,
    H,
    KH,
    d,
    P,
    max_pages,
    lens,
    hnd,
    *,
    s_q=1,
    causal_br=False,
    stats=False,
    dtype=torch.float16,
    pin_cga=None,
    seed=0,
    table_batch_stride=None,
    require_prepared=False,
):
    """Build + run cuDNN's paged-cache graph; return (plan, o [B,H,s_q,d], stats or None, inputs)."""
    import cudnn
    import cudnn.sdpa  # noqa: F401 — registers the FROST engines

    dev = "cuda"
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, bt = _pools(B, KH, d, P, max_pages, hnd, dtype, seed)
    k_c, v_c = (k_pool, v_pool) if hnd else (k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3))
    bt4 = bt.view(B, 1, max_pages, 1)
    if table_batch_stride is not None:
        assert B == 1, "the large-stride probe traverses no batch stride and needs only one row of storage"
        bt4 = torch.as_strided(bt4, bt4.shape, (table_batch_stride, max_pages, 1, 1))
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
    if require_prepared:
        from cudnn.sdpa.fwd.prepared import PreparedDenseLaunch

        compiled = g._compiled_plans[g._plan_index]
        assert compiled._compiled.kernel_template == "decode_d128_f16"
        assert isinstance(compiled._prepared, PreparedDenseLaunch)
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
def test_graph_decode_prepared_keeps_int64_page_table_batch_stride():
    """A singleton table's unused batch stride must not narrow at the pointer ABI.
    B=1 keeps the valid allocation small while a stride over 2**31 catches int32 fakes."""
    _paged_graph(1, 8, 2, D, 16, 8, [128], False, stats=True, table_batch_stride=2**31 + 32, require_prepared=True)


@pytest.mark.L0
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
def test_graph_decode_selects_the_decode_tile(hnd):
    """S_q=1 GQA 8:2 over a paged cache: the heuristics' first plan is the
    decode tile (TILE_CGA_M=1, packed), Stats out, lengths incl. 0 and 1.  The
    split follows the wave-cost model: b=5 x 2 KV heads underfills the SMs and
    the declared KV max (70 pages x 16 = 1120, not a 128-multiple) no longer
    withholds the split since #1092, so the decode tile runs split + combine
    here (Stats recombined too)."""
    plan = _paged_graph(5, 8, 2, D, 16, 70, [300, 77, 0, 1, 1024], hnd, stats=True)
    assert (plan.knobs.cga, plan.knobs.pack_gqa) == (1, True), plan.knobs
    assert plan.knobs.split_kv > 1, plan.knobs


@pytest.mark.L0
def test_graph_mtp_bottom_right_selects_the_decode_tile_natural():
    """S_q=4 bottom-right causal (MTP) at 64/4: decode tile, NATURAL scheduler, packed."""
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL

    plan = _paged_graph(4, 64, 4, D, 16, 40, [600, 4, 129, 640], hnd=True, s_q=4, causal_br=True, stats=True, dtype=torch.bfloat16)
    assert (plan.knobs.cga, plan.knobs.pack_gqa, plan.knobs.sched_policy) == (1, True, SCHED_NATURAL), plan.knobs


@pytest.mark.L0
@pytest.mark.parametrize("h,kh,packed", [(96, 8, True), (48, 8, True), (24, 8, False)], ids=["g12_packs4", "g6_packs2", "g3_unpacked"])
@pytest.mark.parametrize("s_q", [1, 4])
def test_graph_partial_pack_gqa_selects_the_decode_tile(h, kh, packed, s_q):
    """GQA groups that do not divide the 128-row tile over a paged cache (the GLM
    96/8 decode shape): the heuristics' first plan is the decode tile
    (TILE_CGA_M=1) with partial PackGQA (G=12 packs 4, G=6 packs 2; S_q * PACK_G
    <= 128 at S_q = 1 and MTP S_q = 4 bottom-right), Stats out; a group sharing
    no factor with the tile (24/8, G=3) rides the decode tile unpacked.  The
    reference is per Q head, so a slip in the packed KV-head index
    (packed head // (G / PACK_G)) or the LSE scatter shows up here."""
    plan = _paged_graph(4, h, kh, D, 16, -(-1100 // 16), [300, 77, 1, 1100], hnd=True, s_q=s_q, causal_br=s_q > 1, stats=True, dtype=torch.bfloat16)
    assert (plan.knobs.cga, plan.knobs.pack_gqa) == (1, packed), plan.knobs


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


# --- the ragged-Q leg: ragged Q/O/Stats over paged K/V at S_q(max) == 1 (nvbug 6607857) ---


class _RaggedStub:
    """A graph tensor carrying ragged offsets, for the GPU-free facts."""

    class _Off:
        def __init__(self, dtype):
            self._dtype = dtype

        def get_data_type(self):
            return self._dtype

    def __init__(self, dtype, mult=1, stride=(64 * 128, 128, 64 * 128, 1)):
        self.ragged_offset = _RaggedStub._Off(dtype)
        self._mult = mult
        self._stride = tuple(stride)

    def get_ragged_offset_multiplier(self):
        return self._mult

    def get_stride(self):
        return self._stride


def _ragged_paged_facts(*, dtype=None, s_q=1, stats=True, mult=(1, 1, 1), o_token_stride=64 * 128, **over):
    """FlashInfer's prefill-style paged graph at one token per sequence: ragged
    Q/O(/Stats) over page pools, per-batch lengths, GQA 64/8 (packed (T, H, D)
    Q; O's token stride may carry a gap; token-major (T, H) Stats)."""
    import cudnn

    dtype = cudnn.data_type.INT32 if dtype is None else dtype
    q_t = _RaggedStub(dtype, mult[0])
    o_t = _RaggedStub(dtype, mult[1], stride=(o_token_stride, 128, o_token_stride, 1))
    st_t = _RaggedStub(dtype, mult[2], stride=(64, 1, 64, 1))
    base = dict(
        b=4, h_q=64, h_kv=8, s_q=s_q, s_kv=2048, thd=True, padded=True, has_paged_kv=True, page_size=16, q_t=q_t, o_t=o_t, stats_t=st_t if stats else None
    )
    base.update(over)
    return _facts(**base)


@pytest.mark.L0
def test_decode_cfg_ragged_q_leg_contract():
    """The config twin of the ragged-Q leg: ragged_q needs the decode tile, a split
    (>= 2) and paged K/V, excludes the prefill THD leg and the dense Q trim, and
    surfaces as CfgD128Decode.RAGGED_Q."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128, make_cfg_d128_decode

    ok = dict(cta_mma=1, ragged_q=True, paged_kv=True, page_size=16, seq_kv_lens_present=True, split_kv=2)
    cfg, _ = make_cfg_d128_decode(TemplateParams(**ok))
    assert cfg.RAGGED_Q == 1 and cfg.THD_VARLEN == 0 and cfg.SPLIT_KV == 2 and cfg.PAGED_KV == 1
    cfg_packed, _ = make_cfg_d128_decode(TemplateParams(**ok, pack_gqa=True, qh_per_kh=8))
    assert cfg_packed.RAGGED_Q == 1 and cfg_packed.PACK_G == 8
    with pytest.raises(ValueError, match="split_kv must be >= 2"):
        make_cfg_d128_decode(TemplateParams(**{**ok, "split_kv": 1}))
    with pytest.raises(ValueError, match="PAGED"):
        make_cfg_d128_decode(TemplateParams(cta_mma=1, ragged_q=True, seq_kv_lens_present=True, split_kv=2))
    with pytest.raises(ValueError, match="mutually exclusive|THD"):
        make_cfg_d128_decode(TemplateParams(**ok, thd_varlen=True))
    with pytest.raises(ValueError, match="seq_q_lens_present"):
        make_cfg_d128_decode(TemplateParams(**ok, seq_q_lens_present=True))
    with pytest.raises(ValueError, match="decode tile only"):
        make_cfg_d128(TemplateParams(**{**ok, "cta_mma": 2}))


@pytest.mark.L0
def test_ragged_q_leg_predicate_and_heuristics():
    """engines._thd_decode_leg admits exactly FlashInfer's shape (ragged Q/O/Stats,
    paged, S_q(max) == 1, d128 half, one offset width whose multiplier divides the
    row) and the heuristics then propose the decode tile with PackGQA and a split
    of at least 2 -- no unsplit runner-up; every other THD graph keeps cga=2."""
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import _thd_decode_leg, _thd_decode_leg_divisors, _thd_decode_leg_int64

    specs = list(engines.ENGINE_SPECS.values()) if isinstance(engines.ENGINE_SPECS, dict) else list(engines.ENGINE_SPECS)
    caps = next(s for s in specs if s.name == ENGINE).capabilities
    leg = _ragged_paged_facts()
    assert _thd_decode_leg(caps, leg)
    assert _thd_decode_leg_divisors(leg) == (64 * 128, 64 * 128, 64) and not _thd_decode_leg_int64(leg)
    # cuDNN's multiplier form: offsets in tokens (Q/O: H*D per token; Stats: H per token).
    tok = _ragged_paged_facts(mult=(64 * 128, 64 * 128, 64))
    assert _thd_decode_leg(caps, tok) and _thd_decode_leg_divisors(tok) == (1, 1, 1)
    # The offset unit is the DECLARED token stride, not H*D: a token-stride gap on O
    # (test_mhas_v2's with_ragged_token_gap) makes its offsets 4x larger per token.
    gap = _ragged_paged_facts(o_token_stride=4 * 64 * 128)
    assert _thd_decode_leg(caps, gap) and _thd_decode_leg_divisors(gap) == (64 * 128, 4 * 64 * 128, 64)
    i64 = _ragged_paged_facts(dtype=cudnn.data_type.INT64)
    assert _thd_decode_leg(caps, i64) and _thd_decode_leg_int64(i64)
    assert _thd_decode_leg(caps, _ragged_paged_facts(stats=False))
    for off in (
        _ragged_paged_facts(s_q=2),  # MTP-THD keeps the prefill THD leg
        _ragged_paged_facts(has_paged_kv=False, page_size=0),  # ragged K/V: the THD leg's clamped descriptors
        _ragged_paged_facts(mult=(3, 1, 1)),  # a multiplier that does not divide the row
        _ragged_paged_facts(d_qk=64, d_v=64),  # the d128 envelope, not the native flavor
        _ragged_paged_facts(has_sink=True),
        _ragged_paged_facts(stats_t=_RaggedStub(cudnn.data_type.INT64)),  # mixed offset widths
        _ragged_paged_facts(cu_seq_kv_t=object()),  # the cu form is not plumbed on the dense kernel
    ):
        assert not _thd_decode_leg(caps, off), off
    # Rubin has no decode tile: its row's capabilities (sm 107) never admit the leg.
    rubin = [s for s in specs if s.capabilities.sm_lo == 107 and not (s.capabilities.is_fp8 or s.capabilities.is_mxfp8)]
    assert rubin and all(not _thd_decode_leg(s.capabilities, leg) for s in rubin)
    plans = _plans(leg)
    assert plans and all(p.knobs.cga == 1 for p in plans), [p.knobs for p in plans]
    assert all(p.knobs.split_kv >= 2 for p in plans), "the ragged final rows exist only through the combine"
    assert plans[0].knobs.pack_gqa is True
    assert all(p.knobs.cga == 2 and p.knobs.split_kv == 1 for p in _plans(_ragged_paged_facts(s_q=2)))
    # mismatch: cga=1 needs the split; cga=2 is the prefill THD leg (unsplit, unpacked).
    K = engines.SdpaFwdKnobs
    assert "split_kv >= 2" in (engines.mismatch(caps, leg, K(cga=1, split_kv=1)) or "")
    assert engines.mismatch(caps, leg, K(cga=1, split_kv=4, pack_gqa=True)) is None
    assert engines.mismatch(caps, leg, K(cga=2, split_kv=1)) is None
    assert engines.mismatch(caps, leg, K(cga=2, split_kv=2)) is not None, "the prefill THD leg cannot split"
    assert "decode tile" in (engines.mismatch(caps, _ragged_paged_facts(s_q=2), K(cga=1, split_kv=2)) or ""), "MTP-THD keeps the prefill tile"


def _thd_decode_graph(*, dtype, offset_dtype, hnd, q_lens, kv_lens, H=16, KH=2, P=16, max_pages=20, stats=True, out_cap=None, empty_outputs=False):
    """Build + run the FlashInfer-shaped ragged paged graph at S_q(max) == 1 and check
    it end to end: routing (decode tile, split, PackGQA, prepared launch), a
    zero-length sequence whose rows are never written, per-sequence numerics
    (O and packed token-major Stats at the ragged offsets), Rule 3 (no host read).

    ``out_cap``: the O / Stats buffers hold this many packed tokens (default T + 3,
    a poisoned tail past the packed total); a capacity BELOW T checks that the
    combine's stores are bounded by it.  ``empty_outputs``: zero-element O / Stats
    producers -- the THD contract launches nothing for them."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.prepared import PreparedDenseLaunch

    dev, d = "cuda", D
    B, T = len(q_lens), sum(q_lens)
    assert max(q_lens) == 1 and T >= 1
    cu = [0]
    for s in q_lens:
        cu.append(cu[-1] + s)
    scale = 1.0 / math.sqrt(d)
    k_pool, v_pool, bt = _pools(B, KH, d, P, max_pages, hnd, dtype, seed=3)
    k_c, v_c = (k_pool, v_pool) if hnd else (k_pool.permute(0, 2, 1, 3), v_pool.permute(0, 2, 1, 3))
    bt4 = bt.view(B, 1, max_pages, 1)
    gen = torch.Generator(device=dev).manual_seed(11)
    # Packed (T, H, d) storage with a poisoned tail past the last token: the leg must
    # neither read it into a live row nor write past T (the empty sequence has no row).
    cap = T + 3
    q_pk = torch.randn(cap, H, d, device=dev, dtype=torch.float32, generator=gen).to(dtype)
    out_rows = 0 if empty_outputs else (cap if out_cap is None else out_cap)
    o_pk = torch.full((out_rows, H, d), float("nan"), device=dev, dtype=dtype)
    st_pk = torch.full((out_rows, H), float("nan"), device=dev, dtype=torch.float32)
    stride_q = (H * d, d, H * d, 1)  # the graph's (B, H, S_max=1, d) declaration; batch stride == token stride (never stepped)
    tdt = torch.int32 if offset_dtype == cudnn.data_type.INT32 else torch.int64
    ro_q = (torch.tensor(cu, dtype=torch.int64, device=dev) * H * d).to(tdt).view(B + 1, 1, 1, 1)
    ro_st = (torch.tensor(cu, dtype=torch.int64, device=dev) * H).to(tdt).view(B + 1, 1, 1, 1)
    slq = torch.tensor(q_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(kv_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, 1, d], stride=list(stride_q), data_type=io, name="q")
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt4), g.tensor_like(bt4)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    qro, oro, sro = g.tensor_like(ro_q), g.tensor_like(ro_q), g.tensor_like(ro_st)
    tq.set_ragged_offset(qro)
    o, st = g.sdpa(
        name="sdpa",
        q=tq,
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
    )
    o.set_output(True).set_dim([B, H, 1, d]).set_stride(list(stride_q))
    o.set_ragged_offset(oro)
    if stats:
        st.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([B, H, 1, 1]).set_stride([H, 1, H, 1])  # token-major (T, H)
        st.set_ragged_offset(sro)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, ENGINE)
    assert plan.knobs.cga == 1 and plan.knobs.split_kv >= 2 and plan.knobs.pack_gqa is True, plan.knobs
    g.check_support()
    g.build_plans()
    compiled = g._compiled_plans[g._plan_index]
    assert compiled._compiled.kernel_template == "decode_d128_f16"
    assert isinstance(compiled._prepared, PreparedDenseLaunch)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    vp = {tq: q_pk, k: k_c, v: v_c, tk: bt4, tv: bt4, sq_t: slq, sk_t: slk, qro: ro_q, oro: ro_q, o: o_pk}
    if stats:
        vp[st] = st_pk
        vp[sro] = ro_st
    # Rule 3: the ragged offsets and the per-batch lengths are read on device.
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    if empty_outputs:
        return plan  # nothing addressable: the binder skipped both launches (a null-pointer store would have faulted)
    o_out = o_pk.float()
    for b in range(B):
        if q_lens[b] == 0 or cu[b] >= out_rows:
            continue  # no row, or a row past the short buffer's capacity (never written)
        ref_o, ref_lse = _ref(q_pk[cu[b] : cu[b] + 1], _gather_kv(k_pool, bt[b], kv_lens[b], hnd), _gather_kv(v_pool, bt[b], kv_lens[b], hnd), 1, scale)
        got = o_out[cu[b] : cu[b] + 1]
        assert not torch.isnan(got).any(), f"batch {b}: NaN in O"
        torch.testing.assert_close(got, ref_o, atol=_tol(dtype), rtol=0, msg=f"batch {b}")
        if stats:
            got_lse = st_pk[cu[b], :]  # (H,)
            ref = ref_lse[:, 0]
            live = ~torch.isinf(ref)
            torch.testing.assert_close(got_lse[live], ref[live], atol=5e-3, rtol=0, msg=f"batch {b} Stats")
            assert torch.isinf(got_lse[~live]).all()
    # Nothing written past the packed total (the empty sequence owns no row).
    assert torch.isnan(o_pk[T:].float()).all() and (not stats or torch.isnan(st_pk[T:]).all())
    return plan


@pytest.mark.L0
@pytest.mark.parametrize("stats", [True, False], ids=["stats", "no_stats"])
def test_graph_thd_paged_decode_empty_outputs_launch_nothing(stats):
    """The THD contract for a producer with no addressable token: zero-element O
    (and Stats) buffers launch neither the decode kernel nor the combine -- the
    binder returns no frame instead of scheduling a store through a null pointer
    (codex review on #1191)."""
    import cudnn

    _thd_decode_graph(
        dtype=torch.bfloat16, offset_dtype=cudnn.data_type.INT32, hnd=False, q_lens=[1, 1, 1], kv_lens=[64, 300, 17], stats=stats, empty_outputs=True
    )


@pytest.mark.L0
def test_graph_thd_paged_decode_short_outputs_are_bounded():
    """O / Stats buffers holding fewer packed tokens than the sequences address:
    the combine skips every row at or past the buffer's capacity (rows below it
    are correct), so a short buffer or an out-of-range offset never writes
    outside the caller's bytes."""
    import cudnn

    _thd_decode_graph(dtype=torch.bfloat16, offset_dtype=cudnn.data_type.INT64, hnd=True, q_lens=[1, 1, 1, 1, 1], kv_lens=[40, 300, 17, 129, 8], out_cap=2)


@pytest.mark.L0
@pytest.mark.parametrize("offsets", ["int32", "int64"])
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
def test_graph_thd_paged_decode_rides_the_ragged_q_leg(hnd, offsets):
    """FlashInfer's prefill-style paged graph at one token per sequence (ragged
    Q/O/Stats + page pools, S_q(max) == 1, GQA 16/2) rides the decode tile's
    ragged-Q leg: cga=1, PackGQA, split + combine, prepared launch; one sequence
    is empty (no row written), KV lengths span short / page-unaligned / one
    page, int32 (FlashInfer) and int64 (cuDNN's default) offsets."""
    import cudnn

    dt = cudnn.data_type.INT32 if offsets == "int32" else cudnn.data_type.INT64
    _thd_decode_graph(dtype=torch.bfloat16, offset_dtype=dt, hnd=hnd, q_lens=[1, 1, 0, 1, 1], kv_lens=[300, 77, 129, 16, 1])


@pytest.mark.L0
def test_graph_thd_paged_decode_without_stats_fp16():
    """The same leg with no Stats output (has_lse folds the combine's LSE store out)."""
    import cudnn

    _thd_decode_graph(dtype=torch.float16, offset_dtype=cudnn.data_type.INT32, hnd=True, q_lens=[1, 1, 1], kv_lens=[320, 5, 64], stats=False)


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
@pytest.mark.parametrize("sink", [-120.0, -5.0, 3.0], ids=["sink_m120", "sink_m5", "sink_p3"])
def test_graph_dense_mtp_keyless_rows_sink_magnitude(sink):
    """Through the graph API on the decode tile: bf16 dense padded MTP (S_q = 4,
    bottom-right causal), 4/1 heads, a one-key batch and a full 128-key batch,
    sink_token pinned per head, Stats out.  The one-key batch's three keyless
    rows write O := 0 / LSE := sink at every magnitude (the -120 case used to
    be O = NaN / LSE = -inf).  Dense because the row declines paged + sink
    today (a separate change lifts that; its module carries the paged twin)."""
    import cudnn
    import cudnn.sdpa  # noqa: F401

    dev, dtype = "cuda", torch.bfloat16
    B, H, KH, S_q, S_kv = 2, 4, 1, 4, 128
    lens = [1, 128]
    scale = 1.0 / math.sqrt(D)
    gen = torch.Generator(device=dev).manual_seed(1095)
    q_gpu = torch.randn(B, S_q, H, D, device=dev, dtype=torch.float32, generator=gen).to(dtype).transpose(1, 2)
    k_gpu = torch.randn(B, S_kv, KH, D, device=dev, dtype=torch.float32, generator=gen).to(dtype).transpose(1, 2)
    v_gpu = torch.randn(B, S_kv, KH, D, device=dev, dtype=torch.float32, generator=gen).to(dtype).transpose(1, 2)
    o_gpu = torch.empty(B, S_q, H, D, device=dev, dtype=dtype).transpose(1, 2)
    stats_gpu = torch.empty(B, H, S_q, 1, device=dev, dtype=torch.float32)
    sink_gpu = torch.full((1, H, 1, 1), sink, device=dev, dtype=torch.float32)
    slk = torch.tensor(lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slq = torch.full((B, 1, 1, 1), S_q, dtype=torch.int32, device=dev)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(q_gpu), g.tensor_like(k_gpu), g.tensor_like(v_gpu)
    sq_t, sk_t, sink_t = g.tensor_like(slq), g.tensor_like(slk), g.tensor_like(sink_gpu)
    o, st = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        use_causal_mask_bottom_right=True,
        sink_token=sink_t,
    )
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, ENGINE)
    assert (plan.knobs.cga, plan.knobs.pack_gqa, plan.knobs.split_kv) == (1, True, 1), plan.knobs
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute({q: q_gpu, k: k_gpu, v: v_gpu, sq_t: slq, sk_t: slk, sink_t: sink_gpu, o: o_gpu, st: stats_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    out = o_gpu.transpose(1, 2).float()  # [B, S_q, H, D]
    assert torch.isfinite(out).all(), "keyless rows must not NaN the output"
    rows = torch.arange(S_q, device=dev).view(1, S_q)
    keyless = (torch.tensor(lens, device=dev).view(B, 1) - S_q + rows < 0).view(B, S_q, 1).expand(B, S_q, H)
    assert keyless.sum().item() == 3 * H, "the one-key batch is keyless on its three rows above the diagonal"
    assert (out[keyless] == 0).all(), "a keyless row writes O := 0 even with a sink"
    got_lse = stats_gpu[:, :, :, 0].transpose(1, 2)  # [B, S_q, H]
    torch.testing.assert_close(got_lse[keyless], torch.full_like(got_lse[keyless], sink), atol=1e-4, rtol=0)
    for b in range(B):
        ref_o, ref_lse = _ref(
            q_gpu[b].transpose(0, 1),
            k_gpu[b].transpose(0, 1)[: lens[b]],
            v_gpu[b].transpose(0, 1)[: lens[b]],
            S_q,
            scale,
            causal_br=True,
            sink=sink_gpu.flatten(),
        )
        torch.testing.assert_close(out[b], ref_o, atol=_tol(dtype), rtol=0, msg=f"batch {b}")
        torch.testing.assert_close(stats_gpu[b, :, :, 0], ref_lse, atol=5e-3, rtol=0, msg=f"LSE batch {b}")


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
