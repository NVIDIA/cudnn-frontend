# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.bwd.config_d256_2x2``: the 2x2-datapath d256 backward config, host-only.

Every layout fact the body and its adapters rely on is pinned as a VALUE per profile (TMEM map, SMEM slab table and
descriptor roots, descriptor version, every mbarrier stage / init count and TMA transaction size, the scheduler
arriver count, the register pool, the stage-3 granularity), and every validator predicate gets one raising case.  The
4x1 config is pinned UNCHANGED: ``make_cfg_d256_bwd(TemplateParams())`` builds the same record as before the appended
``datapath_2x2_profile`` field, the 4x1 factory rejects a non-zero profile and the 2x2 factory rejects profile 0.
Device-independent: nothing here compiles or launches."""

from __future__ import annotations

import dataclasses

import pytest

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16, MASK_CAUSAL, MASK_PADDED, MASK_SWA, SCHED_LPT, SCHED_NATURAL
from cudnn.sdpa.bwd import config_d256_2x2 as c2
from cudnn.sdpa.bwd import config_sm107 as c4
from cudnn.sdpa.bwd.config_sm107 import FAMILY_F16, FAMILY_FP8, FAMILY_MXFP8, TemplateParams

pytestmark = [pytest.mark.L0]

P1, P2 = c2.PROFILE_SM100, c2.PROFILE_SM107_INTERLEAVED


def _cfg(profile=P1, **kw):
    kw.setdefault("dtype_qkv", DTYPE_BF16)
    return c2.make_cfg_d256_2x2(TemplateParams(datapath_2x2_profile=profile, **kw), FAMILY_F16)


# --------------------------------------------------------------------------- the profile axis and the 4x1 record


def test_appended_template_params_field_defaults_to_the_4x1_body():
    fields = [f.name for f in dataclasses.fields(TemplateParams)]
    assert fields[-1] == "datapath_2x2_profile" and TemplateParams().datapath_2x2_profile == 0
    assert c2.PROFILES == (1, 2) and c2.PROFILE_OFF == 0


@pytest.mark.parametrize(
    "family, kw", [(FAMILY_F16, dict(dtype_qkv=DTYPE_BF16)), (FAMILY_FP8, dict(dtype_qkv=DTYPE_E4M3)), (FAMILY_MXFP8, dict(dtype_qkv=DTYPE_E4M3))]
)
@pytest.mark.parametrize("profile", [1, 2, 7])
def test_4x1_factory_rejects_a_2x2_profile(family, kw, profile):
    with pytest.raises(ValueError, match=r"datapath_2x2_profile=\d+ selects the 2x2-datapath body"):
        c4.make_cfg_d256_bwd(TemplateParams(datapath_2x2_profile=profile, **kw), family)


def test_4x1_record_is_unchanged_by_the_appended_field():
    """``make_cfg_d256_bwd`` never reads the new field: the record at profile 0 equals the one a pre-field caller built
    (the dd3235c3 byte-for-byte pins in test_sdpa_bwd_config_sm107.py hold its values)."""
    base = c4.make_cfg_d256_bwd(TemplateParams(dtype_qkv=DTYPE_BF16), FAMILY_F16)
    explicit = c4.make_cfg_d256_bwd(TemplateParams(dtype_qkv=DTYPE_BF16, datapath_2x2_profile=0), FAMILY_F16)
    assert base == explicit
    assert not any(f.name.startswith("DATAPATH") or f.name in ("KV_SUBBLOCKS", "ROWS_PER_CTA", "L_CNT") for f in dataclasses.fields(c4.CfgBwdD256))


@pytest.mark.parametrize("profile", [0, 3, -1])
def test_2x2_factory_rejects_profile_off_and_unknown(profile):
    with pytest.raises(ValueError, match=r"datapath_2x2_profile must be 1 .* or 2"):
        c2.make_cfg_d256_2x2(TemplateParams(dtype_qkv=DTYPE_BF16, datapath_2x2_profile=profile), FAMILY_F16)


def test_2x2_factory_rejects_other_families():
    with pytest.raises(ValueError, match=r"dtype_family must be FAMILY_F16"):
        c2.make_cfg_d256_2x2(TemplateParams(dtype_qkv=DTYPE_E4M3, datapath_2x2_profile=1), FAMILY_FP8)
    with pytest.raises(ValueError, match=r"f16 body takes DTYPE_BF16"):
        _cfg(dtype_qkv=DTYPE_E4M3)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(window_left=0), r"SWA requires window_left > 0"),
        (dict(window_right=64), r"window_right must be 0 when set"),
        (dict(bottom_right=True), r"bottom_right alignment requires a causal band"),
        (dict(seq_q_lens_present=True), r"seq_q_lens_present is not implemented"),
        (dict(thd_varlen=True), r"thd_varlen is not implemented"),
        (dict(dtype_o=DTYPE_E4M3), r"dtype_o must be -1"),
        (dict(dtype_ds=DTYPE_FP16), r"dtype_ds must be -1 or the io dtype"),
        (dict(ds_sf_policy=1), r"ds_sf_policy is the MXFP8 family's"),
    ],
)
def test_2x2_factory_shares_the_record_guards(kw, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**kw)


# --------------------------------------------------------------------------- the two profiles, pinned as values


def test_profile_1_geometry():
    cfg = _cfg(P1)
    assert (cfg.DATAPATH_2X2_PROFILE, cfg.KV_SUBBLOCKS, cfg.TILE_M, cfg.ROWS_PER_CTA, cfg.KV_BLOCK_ROWS, cfg.MMA_M) == (1, 1, 64, 64, 128, 128)
    assert (cfg.STAGE3_GRAN_ROWS, c2.kv_pad_rows_2x2(cfg), c2.q_pad_rows_2x2(cfg), c2.q_write_tiles_2x2(cfg)) == (256, 256, 128, 2)
    assert (cfg.SUBBLOCK_WGS, cfg.COLS_PER_LANE, cfg.L_CNT) == (2, 32, 512)
    assert (cfg.STAGES_Q, cfg.STAGES_dO, cfg.STAGES_dO_DV, cfg.STAGES_KV, cfg.STAGES_TMEM_S, cfg.STAGES_SMEM_P, cfg.STAGES_TMEM_P, cfg.XFER_STAGES) == (
        1,
        1,
        1,
        1,
        2,
        2,
        0,
        1,
    )
    # MMA_LOOKAHEAD 0: profile 1 ships the NATURAL MMA order (B200 A/B 2026-10-01: stage 2 -16% dense / +2% causal vs lookahead).
    assert (cfg.K_SPLIT_UTCCP, cfg.IS_FP8, cfg.IS_MXFP8, cfg.MMA_LOOKAHEAD, cfg.SUBBLOCK_STRIDE_COLS) == (0, 0, 0, 0, 0)
    assert cfg.SMEM_CAP == 227 * 1024 and cfg.TMEM_ALLOC_COLS == 512
    assert (cfg.TILE_K_HW_BMM1, cfg.TILE_K_HW_BMM2, cfg.IDESC_K_DIM) == (16, 16, 0)
    assert (cfg.CGA_M, cfg.CGA_N, cfg.CTA_MMA) == (2, 1, 2)


def test_profile_2_geometry():
    cfg = _cfg(P2)
    assert (cfg.DATAPATH_2X2_PROFILE, cfg.KV_SUBBLOCKS, cfg.TILE_M, cfg.ROWS_PER_CTA, cfg.KV_BLOCK_ROWS, cfg.MMA_M) == (2, 2, 64, 128, 256, 128)
    assert (cfg.STAGE3_GRAN_ROWS, c2.kv_pad_rows_2x2(cfg), c2.q_write_tiles_2x2(cfg)) == (256, 256, 2)
    assert (cfg.SUBBLOCK_WGS, cfg.COLS_PER_LANE, cfg.L_CNT) == (1, 64, 256)
    assert (cfg.STAGES_Q, cfg.STAGES_dO, cfg.STAGES_dO_DV, cfg.STAGES_TMEM_S, cfg.STAGES_SMEM_P, cfg.XFER_STAGES) == (2, 1, 1, 1, 1, 1)
    assert cfg.SMEM_CAP == 327 * 1024 and cfg.SUBBLOCK_STRIDE_COLS == 256


@pytest.mark.parametrize("profile", [P1, P2])
def test_dtype_members(profile):
    for dt in (DTYPE_BF16, DTYPE_FP16):
        cfg = _cfg(profile, dtype_qkv=dt)
        assert (cfg.DTYPE_QKV, cfg.DTYPE_O, cfg.DTYPE_DS, cfg.BPE, cfg.BPE_O, cfg.BPE_DS) == (dt, dt, dt, 2, 2, 2)
    cfg = _cfg(profile, dtype_qkv=DTYPE_BF16, dtype_o=DTYPE_FP16)
    assert cfg.DTYPE_O == DTYPE_FP16 and cfg.DTYPE_DS == DTYPE_BF16


@pytest.mark.parametrize(
    "params, flags",
    [
        ({}, 0),
        ({"window_right": 0}, MASK_CAUSAL),
        ({"window_right": 0, "bottom_right": True}, MASK_CAUSAL),
        ({"window_left": 64}, MASK_SWA),
        ({"window_right": 0, "window_left": 640}, MASK_CAUSAL | MASK_SWA),
        ({"seq_kv_lens_present": True}, MASK_PADDED),
    ],
)
@pytest.mark.parametrize("profile", [P1, P2])
def test_mask_arms_and_schedule(profile, params, flags):
    cfg = _cfg(profile, sched_policy=SCHED_LPT, **params)
    assert cfg.MASK_FLAGS == flags and cfg.SWA_WINDOW == params.get("window_left", 0) and cfg.CAUSAL_BOTTOM_RIGHT == int(params.get("bottom_right", False))
    assert cfg.SCHEDULER_POLICY == SCHED_LPT
    assert c2.smem_layout_2x2(cfg) == c2.smem_layout_2x2(_cfg(profile)), "masks move no bytes"


def test_tmem_maps():
    t1 = c2.tmem_layout_2x2(_cfg(P1))
    assert dataclasses.asdict(t1) == dict(
        TOTAL_COLS=512, S_COLS=64, DV_COLS=128, STAGES_TMEM_S=2, S_BASE=0, dP_BASE=128, dV_BASE=256, SUB_USED=384, SUB_STRIDE=0, USED_COLS=384, FREE_COLS=128
    )
    t2 = c2.tmem_layout_2x2(_cfg(P2))
    assert dataclasses.asdict(t2) == dict(
        TOTAL_COLS=512, S_COLS=64, DV_COLS=128, STAGES_TMEM_S=1, S_BASE=0, dP_BASE=64, dV_BASE=128, SUB_USED=256, SUB_STRIDE=256, USED_COLS=512, FREE_COLS=0
    )


_SLABS_P1 = (
    ("sQ", 0, 32768, (("sQ[0]", 0),)),
    ("sdO", 32768, 32768, (("sdO[0]", 32768),)),
    ("sdOdv", 65536, 32768, (("sdO_dv[0]", 65536),)),
    ("sK(+sdV alias)", 98304, 32768, (("sK[0]", 98304),)),
    ("sV", 131072, 32768, (("sV[0]", 131072),)),
    ("sP", 163840, 32768, (("sP[0][0]", 163840), ("sP[0][1]", 180224))),
    ("sStats", 196608, 2048, ()),
    ("sdS", 198656, 16384, ()),
)
_SLABS_P2 = (
    ("sQ", 0, 65536, (("sQ[0]", 0), ("sQ[1]", 32768))),
    ("sdO", 65536, 32768, (("sdO[0]", 65536),)),
    ("sdOdv", 98304, 32768, (("sdO_dv[0]", 98304),)),
    ("sK(+sdV alias)", 131072, 65536, (("sK[0]", 131072), ("sK[1]", 163840))),
    ("sV", 196608, 65536, (("sV[0]", 196608), ("sV[1]", 229376))),
    ("sP", 262144, 32768, (("sP[0][0]", 262144), ("sP[1][0]", 278528))),
    ("sStats", 294912, 2048, ()),
    ("sdS", 296960, 32768, ()),
)


@pytest.mark.parametrize("profile, slabs, total, version", [(P1, _SLABS_P1, 215040, 0), (P2, _SLABS_P2, 329728, 1)], ids=["sm100", "sm107"])
def test_smem_tables_roots_and_descriptor_version(profile, slabs, total, version):
    cfg = _cfg(profile)
    assert tuple((s.name, s.offset, s.nbytes, s.roots) for s in c2.smem_layout_2x2(cfg)) == slabs
    assert c2.smem_bytes_2x2(cfg) == total and c2.kernel_smem_bytes_2x2(cfg) == total + c2.SMEM_SCAFFOLD_BYTES
    assert c2.kernel_smem_bytes_2x2(cfg) <= cfg.SMEM_CAP
    assert c2.desc_version_2x2(cfg) == version
    roots = dict(c2.desc_roots_2x2(cfg))
    assert (max(roots.values()) >= 1 << 18) == bool(version)
    assert c2.scaffold_bytes_declared_2x2(cfg) <= c2.SMEM_SCAFFOLD_BYTES


def test_desc_version_flips_when_a_root_crosses_the_line(monkeypatch):
    cfg = _cfg(P1)
    monkeypatch.setattr(c2, "TCGEN05_V0_ADDR_LIMIT", 1 << 18, raising=False)
    monkeypatch.setattr(c4, "TCGEN05_V0_ADDR_LIMIT", 180224)
    assert c2.desc_version_2x2(cfg) == 1
    with pytest.raises(ValueError, match=r"renders descriptor version 0"):
        c2._validate_cfg_d256_2x2(cfg)


@pytest.mark.parametrize("profile", [P1, P2])
def test_buffer_elems_and_tx_bytes(profile):
    cfg = _cfg(profile)
    b = c2.buffer_elems_2x2(cfg)
    n = cfg.KV_SUBBLOCKS
    assert (b._M_PER_CTA, b.qBufferElems, b.dOBufferElems, b.dOdvBufferElems) == (64, 64 * 256, 64 * 256, 128 * 128)
    assert (b.kSubElems, b.kBufferElems, b.vSubElems, b.vBufferElems) == (64 * 256, n * 64 * 256, 64 * 256, n * 64 * 256)
    assert (b.pSlabElems, b.dSBufferElems, b.dVBufferElems) == (64 * 128, 64 * n * 128, 64 * n * 256)
    assert (b.qTmaTransactionBytes, b.dOTmaTransactionBytes, b.dOdvTmaTransactionBytes) == (65536, 65536, 65536)
    assert (b.kTmaTransactionBytes, b.vTmaTransactionBytes) == (65536 * n, 65536 * n)
    assert (b.TMA_QK_ITERS, b.TMA_VO_ITERS, b.TMA_VO_SG1_ITERS, b.TMA_DV_ITERS, b.P_TMA_ITERS) == (4, 4, 2, 4, 2)
    assert (b.TMA_QK_GRANU_ELEMS, b.DV_D_BLOCK, b.P_D_BLOCK, b.P_SUB_SLAB) == (64, 64, 64, 4096)
    assert (b.DV_BLOCK_SLAB, b.DS_BLOCK_SLAB) == (64 * n * 64, 64 * n * 64)
    assert (b.LEADING_BYTE_OFFSET_QK, b.STRIDE_BYTE_OFFSET_QK, b.LEADING_BYTE_OFFSET_dO_SG1, b.STRIDE_BYTE_OFFSET_dO_SG1) == (0, 1024, 16384, 1024)
    assert (b.LEADING_BYTE_OFFSET_P, b.STRIDE_BYTE_OFFSET_P, b.SMEM_LAYOUT_SW128) == (0, 1024, 2)
    assert b.P_STORE_ALIGN == (64 if profile == P1 else 128)
    assert (b._KV_BLOCK_ROWS, b._KV_WRITE_ROWS, b._Q_WRITE_TILES, b.CGA_SIZE) == (128 * n, 256, 2, 2)
    assert (b.STATS_SLOT_ELEMS, b.STATS_LSE_OFF, b.STATS_DOT_OFF) == (256, 0, 128)


def test_mbarrier_ledger_profile_1():
    cfg = _cfg(P1)
    assert c2.mbar_stage_counts_2x2(cfg) == {
        "mb_q_full": 1,
        "mb_q_empty": 1,
        "mb_do_full": 1,
        "mb_do_empty": 1,
        "mb_dodv_full": 1,
        "mb_dodv_empty": 1,
        "mb_k_full": 1,
        "mb_k_empty": 1,
        "mb_v_full": 1,
        "mb_v_empty": 1,
        "mb_s_full": 2,
        "mb_s_acc_empty": 2,
        "mb_dp_full": 2,
        "mb_dp_empty": 2,
        "mb_p_full": 2,
        "mb_p_empty": 2,
        "mb_stats_full": 2,
        "mb_stats_empty": 2,
        "mb_ds_smem_full": 1,
        "mb_ds_smem_empty": 1,
        "mb_dv_ready": 1,
        "mb_dv_acc_empty": 1,
        "mb_dv_stg_full": 1,
        "mb_dv_stg_empty": 1,
        "mb_tmem_dealloc": 1,
    }
    counts = c2.mbar_init_counts_2x2(cfg)
    assert {k: v for k, v in counts.items() if v != 1} == {
        "mb_s_acc_empty": 512,
        "mb_dp_empty": 512,
        "mb_p_full": 512,
        "mb_stats_full": 32,
        "mb_stats_empty": 256,
        "mb_ds_smem_full": 256,
        "mb_dv_acc_empty": 512,
        "mb_dv_stg_full": 256,
        "mb_tmem_dealloc": 512,
    }
    assert set(counts) == set(c2.mbar_stage_counts_2x2(cfg))
    assert c2.read_tile_arrivers_tot_2x2(cfg) == cfg.READ_TILE_ARRIVERS_TOT == 21


def test_mbarrier_ledger_profile_2():
    cfg = _cfg(P2)
    stages = c2.mbar_stage_counts_2x2(cfg)
    assert (stages["mb_q_full"], stages["mb_s_full"], stages["mb_s_acc_empty"], stages["mb_dp_full"], stages["mb_p_full"], stages["mb_p_empty"]) == (
        2,
        2,
        2,
        2,
        2,
        2,
    )
    counts = c2.mbar_init_counts_2x2(cfg)
    assert (counts["mb_s_acc_empty"], counts["mb_dp_empty"], counts["mb_p_full"], counts["mb_dv_acc_empty"], counts["mb_tmem_dealloc"]) == (
        256,
        256,
        256,
        512,
        512,
    )


def test_removed_and_added_bars_vs_the_4x1_body():
    f16 = c4.mbar_stage_counts(c4.make_cfg_d256_bwd(TemplateParams(dtype_qkv=DTYPE_BF16), FAMILY_F16))
    two = c2.mbar_stage_counts_2x2(_cfg(P1))
    assert set(f16) - set(two) == {"mb_k_utccp_done", "mb_p_ready", "mb_s_acc_full"}
    assert set(two) - set(f16) == {"mb_s_full", "mb_s_acc_empty", "mb_p_full", "mb_p_empty"}


@pytest.mark.parametrize("profile", [P1, P2])
def test_register_pool_and_warp_roster(profile):
    cfg = _cfg(profile)
    assert (cfg.TOTAL_WARPS, cfg.THREADS_PER_CTA, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == (12, 384, 8, 9, 10, 11)
    # Per profile: 176 / 152 on profile 1 (ptxas hoists the static 1-stage B descriptors into the MMA warp: 70 / 79 STL / LDL
    # at 56 on sm_100a), 224 / 56 on profile 2 (64 q columns per compute lane: 91 / 129 STL / LDL at 176 on sm_107a, 0 / 0 at
    # 224 -- the Rubin board register-split sweep of 2026-10-01).  Both fill the 12-warp ENTRY pool exactly.
    soft, svc = (176, 152) if profile == P1 else (224, 56)
    assert (cfg.SOFTMAX_REGS, cfg.MMA_REGS, cfg.TMALDG_REGS, cfg.TMASTG_REGS, cfg.SCHEDULER_REGS, cfg.OTHER_REGS) == (soft, svc, svc, svc, svc, svc)
    assert 8 * soft + 4 * svc == c4.reg_entry_pool(12) == 2016
    assert (cfg.SOFTMAX_LANES, cfg.SOFT_X_CTA_MMA, cfg.MMA_COMMIT_ARRIVES) == (256, 512, 1)


# --------------------------------------------------------------------------- the validator, one raising case per predicate


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(KV_SUBBLOCKS=2), r"KV_SUBBLOCKS is the profile number"),
        (dict(TILE_M=128, MMA_M=256), r"TILE_M = 64 kv rows per CTA per sub-block"),
        (dict(ROWS_PER_CTA=128), r"ROWS_PER_CTA must be TILE_M \* KV_SUBBLOCKS"),
        (dict(KV_BLOCK_ROWS=256), r"KV_BLOCK_ROWS must be ROWS_PER_CTA \* CTA_MMA"),
        (dict(STAGE3_GRAN_ROWS=128), r"STAGE3_GRAN_ROWS is the 256-row kv write pair"),
        (dict(COLS_PER_LANE=64), r"COLS_PER_LANE = 64 // SUBBLOCK_WGS"),
        (dict(L_CNT=256), r"L_CNT .* must be the sub-block's compute lanes x CTA_MMA = 512"),
        (dict(MMA_LOOKAHEAD=2), r"MMA_LOOKAHEAD is 0 .* or 1"),
        (dict(DTYPE_DS=DTYPE_FP16), r"dS workspace IS the io dtype"),
        (dict(TILE_K_HW_BMM1=64), r"k-steps are K = 16"),
        (dict(K_SPLIT_UTCCP=1), r"no UTCCP K-split and no TMEM P"),
        (dict(STAGES_KV=2), r"loaded ONCE per kv block"),
        (dict(STAGES_TMEM_S=3), r"1- or 2-deep TMEM parity rings"),
        (dict(STAGES_SMEM_P=0), r"SMEM P ring is at least 1 deep"),
        (dict(TMEM_ALLOC_COLS=576), r"512-column non-exclusive"),
        (dict(SUBBLOCK_STRIDE_COLS=256), r"SUBBLOCK_STRIDE_COLS must be the derived sub-block stride 0"),
        (dict(SOFTMAX_REGS=184), r"exceeds the 12-warp ENTRY pool"),
        (dict(READ_TILE_ARRIVERS_TOT=20), r"READ_TILE_ARRIVERS_TOT must be 21"),
        (dict(MASK_FLAGS=MASK_SWA), r"MASK_SWA <=> SWA_WINDOW > 0"),
        (dict(Q_SWZ_BYTES=64), r"every slab is 128-B swizzled"),
        (dict(SMEM_CAP=327 * 1024), r"validated against the 227 KiB cap"),
        (dict(STAGES_Q=2), r"exceed the 227 KiB per-CTA cap"),
    ],
)
def test_rejects_every_cfg_predicate(overrides, match):
    cfg = dataclasses.replace(_cfg(P1), **overrides)
    with pytest.raises(ValueError, match=match):
        c2._validate_cfg_d256_2x2(cfg)


def test_profile_2_tmem_is_exactly_full_and_a_third_parity_overflows():
    cfg = _cfg(P2)
    assert c2.tmem_layout_2x2(cfg).FREE_COLS == 0
    with pytest.raises(ValueError, match=r"must fit the 512-column alloc"):
        c2._validate_cfg_d256_2x2(dataclasses.replace(cfg, STAGES_TMEM_S=2, SUBBLOCK_STRIDE_COLS=384))


# --------------------------------------------------------------------------- shape-time helpers


def test_workspace_and_grid_helpers():
    cfg = _cfg(P1)
    assert c2.ds_workspace_bytes_2x2(cfg, 1, 32, 8192, 8192) == 32 * 8192 * 8192 * 2
    with pytest.raises(ValueError, match=r"padded to q 128 / kv 256 rows"):
        c2.ds_workspace_bytes_2x2(cfg, 1, 1, 8192, 8192 - 128)
    # profile 1: 128-row blocks -> 64 blocks x 2 CTAs along x at S_kv = 8192; profile 2: 32 blocks
    assert c2.launch_grid_2x2(cfg, 2, 8, 8192) == ((64 * 2, 8, 2), (2, 1, 1))
    assert c2.launch_grid_2x2(_cfg(P2), 2, 8, 8192) == ((32 * 2, 8, 2), (2, 1, 1))
    assert c2.launch_grid_2x2(_cfg(P1, sched_policy=SCHED_LPT), 2, 8, 8192) == ((64 * 8 * 2 * 2, 1, 1), (2, 1, 1))
    assert c2.launch_grid_2x2(_cfg(P1, sched_policy=SCHED_NATURAL), 1, 1, 256) == ((4, 1, 1), (2, 1, 1))
    c2.validate_head_chunk(32, 2, 32)


def test_mma_order_default_per_profile():
    """Profile 1 ships the NATURAL MMA order (MMA_LOOKAHEAD 0: B200 A/B 2026-10-01, stage 2 3781 us vs the lookahead's 4525 us
    dense, 2020 vs 1974 us causal); profile 2 keeps the design's lookahead order until the Rubin lane measures it."""
    assert _cfg(P1).MMA_LOOKAHEAD == 0
    assert _cfg(P2).MMA_LOOKAHEAD == 1
