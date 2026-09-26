# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.bwd.config_sm107``: the Rubin d256 backward configs, host-only.

Every predicate of ``_validate_params`` / ``_validate_cfg_d256_bwd`` gets ONE
raising case (``pytest.raises(ValueError, match=...)`` on the failure signature
the message carries), and every dtype the two bodies serve gets an accepting
case.  The layout facts the kernel ports and the adapter rely on (SMEM tally,
descriptor roots, TMEM map, scheduler arrivers, scaffolding) are pinned as
values, not just as "does not raise" -- a config that silently moved one of
them is exactly the class of bug the module exists to make impossible.

Device-independent: nothing here compiles or launches.  The TMEM predicates
that are tautological on a derived ``tmem_layout`` (sum to 576, P width) are
pinned as layout VALUES rather than provoked, since no field flip reaches them
before an earlier pin raises.
"""

from __future__ import annotations

import dataclasses

import pytest

from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    MASK_CAUSAL,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)
from cudnn.sdpa.bwd import config_sm107 as cfgmod
from cudnn.sdpa.bwd.config_sm100 import TemplateParams as BaseTemplateParams
from cudnn.sdpa.bwd.config_sm107 import (
    FAMILY_F16,
    FAMILY_FP8,
    SMEM_CAP_BYTES,
    SMEM_SCAFFOLD_BYTES,
    TCGEN05_V0_ADDR_LIMIT,
    TMEM_TOTAL_COLS,
    TemplateParams,
    buffer_elems,
    desc_roots,
    desc_version,
    ds_workspace_bytes,
    kernel_smem_bytes,
    launch_grid,
    make_cfg_d256_bwd,
    mbar_stage_counts,
    read_tile_arrivers_tot,
    scaffold_bytes_declared,
    smem_bytes,
    smem_layout,
    tmem_layout,
    validate_head_chunk,
)

pytestmark = [pytest.mark.L0]

_KiB = 1024


def _cfg(family, **params):
    dq = {FAMILY_F16: DTYPE_BF16, FAMILY_FP8: DTYPE_E4M3}[family]
    return make_cfg_d256_bwd(TemplateParams(dtype_qkv=params.pop("dtype_qkv", dq), **params), family)


def _validate(family, **overrides):
    """Flip fields on a VALID cfg and re-run the cfg validator."""
    cfg = dataclasses.replace(_cfg(family), **overrides)
    cfgmod._validate_cfg_d256_bwd(cfg, cfgmod._FLAVOR[family])


# ---------------------------------------------------------------------------
# Accept: every dtype member of both bodies, and the record shapes the adapter builds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, dtype_qkv, dtype_o, want_o, want_ds",
    [
        (FAMILY_F16, DTYPE_BF16, -1, DTYPE_BF16, DTYPE_BF16),
        (FAMILY_F16, DTYPE_FP16, -1, DTYPE_FP16, DTYPE_FP16),
        (FAMILY_F16, DTYPE_BF16, DTYPE_FP16, DTYPE_FP16, DTYPE_BF16),
        # fp8: E4M3 io; grads default to E4M3 (the fp8 graph contract), dS is BF16 for the bf16 GEMMs
        (FAMILY_FP8, DTYPE_E4M3, -1, DTYPE_E4M3, DTYPE_BF16),
        (FAMILY_FP8, DTYPE_E4M3, DTYPE_BF16, DTYPE_BF16, DTYPE_BF16),
        (FAMILY_FP8, DTYPE_E4M3, DTYPE_FP16, DTYPE_FP16, DTYPE_BF16),
    ],
)
def test_accepts_every_dtype_member(family, dtype_qkv, dtype_o, want_o, want_ds):
    cfg = _cfg(family, dtype_qkv=dtype_qkv, dtype_o=dtype_o)
    assert (cfg.DTYPE_QKV, cfg.DTYPE_O, cfg.DTYPE_DS) == (dtype_qkv, want_o, want_ds)
    assert (cfg.BPE, cfg.BPE_O, cfg.BPE_DS) == (cfgmod.bpe(dtype_qkv), cfgmod.bpe(want_o), cfgmod.bpe(want_ds))
    assert cfg.IS_FP8 == (family == FAMILY_FP8)


def test_accepts_the_shared_sm100_record_unextended():
    """A plain bwd TemplateParams (no dtype_o / has_sink) is shape-compatible."""
    cfg = make_cfg_d256_bwd(
        BaseTemplateParams(dtype_qkv=DTYPE_FP16, window_right=0, bottom_right=True, window_left=256, seq_kv_lens_present=True, sched_policy=SCHED_LPT),
        FAMILY_F16,
    )
    assert cfg.MASK_FLAGS == MASK_CAUSAL | MASK_SWA | MASK_PADDED
    assert (cfg.SWA_WINDOW, cfg.CAUSAL_BOTTOM_RIGHT, cfg.SEQ_KV_LENS_PRESENT, cfg.HAS_SINK, cfg.SCHEDULER_POLICY) == (256, 1, 1, 0, SCHED_LPT)
    assert cfg.DTYPE_O == DTYPE_FP16  # inherits the io dtype


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
@pytest.mark.parametrize(
    "params, flags",
    [
        ({}, 0),
        ({"window_right": 0}, MASK_CAUSAL),
        ({"window_right": 0, "bottom_right": True}, MASK_CAUSAL),
        ({"window_left": 64}, MASK_SWA),
        ({"window_right": 0, "window_left": 640}, MASK_CAUSAL | MASK_SWA),
        ({"seq_kv_lens_present": True}, MASK_PADDED),
        ({"window_right": 0, "window_left": 128, "seq_kv_lens_present": True, "bottom_right": True}, MASK_CAUSAL | MASK_SWA | MASK_PADDED),
    ],
)
def test_accepts_every_mask_arm(family, params, flags):
    cfg = _cfg(family, **params)
    assert cfg.MASK_FLAGS == flags
    assert cfg.SWA_WINDOW == params.get("window_left", 0)
    assert cfg.CAUSAL_BOTTOM_RIGHT == int(params.get("bottom_right", False))


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
@pytest.mark.parametrize("policy", [SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2])
def test_accepts_every_scheduler_policy_and_has_sink(family, policy):
    cfg = _cfg(family, sched_policy=policy, has_sink=True)
    assert cfg.SCHEDULER_POLICY == policy and cfg.HAS_SINK == 1


# ---------------------------------------------------------------------------
# Reject: the per-graph record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, params, match",
    [
        (FAMILY_F16, dict(dtype_qkv=7), r"tile_dsl DTYPE_\* code"),
        (FAMILY_FP8, dict(dtype_qkv=DTYPE_E5M2), r"E4M3-only.*E5M2 is not implemented"),
        (FAMILY_FP8, dict(dtype_qkv=DTYPE_BF16), r"E4M3-only.*belongs to the f16 body"),
        (FAMILY_FP8, dict(dtype_o=9), r"dtype_o must be -1 \(inherit -> E4M3"),
        (FAMILY_F16, dict(dtype_qkv=DTYPE_E4M3), r"f16 body takes DTYPE_BF16.*belongs to the fp8 body"),
        (FAMILY_F16, dict(dtype_o=DTYPE_E4M3), r"dtype_o must be -1 \(inherit the io dtype\)"),
        (FAMILY_F16, dict(window_left=0), r"SWA requires window_left > 0"),
        (FAMILY_FP8, dict(window_left=-5), r"SWA requires window_left > 0"),
        (FAMILY_F16, dict(window_right=64), r"window_right must be 0 when set.*Right-band widening"),
        (FAMILY_FP8, dict(bottom_right=True), r"bottom_right alignment requires a causal band"),
        (FAMILY_F16, dict(seq_q_lens_present=True, seq_kv_lens_present=True), r"seq_q_lens_present is not implemented"),
        (FAMILY_FP8, dict(thd_varlen=True), r"thd_varlen is not implemented"),
        (FAMILY_F16, dict(sched_policy=3), r"sched_policy must be one of NATURAL/LPT/LPT_L2"),
    ],
)
def test_rejects_a_record_the_body_cannot_express(family, params, match):
    with pytest.raises(ValueError, match=match):
        _cfg(family, **params)


def test_rejects_an_unknown_family():
    with pytest.raises(ValueError, match=r"dtype_family must be one of"):
        make_cfg_d256_bwd(TemplateParams(dtype_qkv=DTYPE_BF16), "mxfp8")


# ---------------------------------------------------------------------------
# Reject: every cfg predicate, one flipped field each (both families where the
# predicate is shared, the owning family where it is not)
# ---------------------------------------------------------------------------

_SHARED_CFG_REJECTS = [
    # register split
    # 32: distinct from BOTH families' service count (f16 56, fp8 40) and keeps the sum under the pool, so only the equality predicate fires
    (dict(MMA_REGS=32), r"MMA/TMALDG/TMASTG/SCHEDULER regs must be equal"),
    (dict(SOFTMAX_REGS=240), r"register split (2144|2080) over the 12-warp ENTRY pool 2016"),  # f16 8x240+4x56 / fp8 8x240+4x40
    (dict(SOFTMAX_REGS=220), r"multiple of 8"),  # under the pool on both families, so only the granule predicate fires
    (dict(MMA_REGS=16, TMALDG_REGS=16, TMASTG_REGS=16, SCHEDULER_REGS=16, OTHER_REGS=16), r"within 24\.\.256"),
    (dict(OTHER_REGS=32), r"OTHER_REGS.*must equal MMA_REGS"),
    # warp population + scheduler ring
    (dict(SOFTMAX_WARPGROUPS=1), r"8 compute warps in 2 warpgroups of 4"),
    (dict(TOTAL_WARPS=16), r"TOTAL_WARPS must be compute \+ correction \+ 4"),
    (dict(THREADS_PER_CTA=256), r"THREADS_PER_CTA must be TOTAL_WARPS\*32"),
    (dict(MMA_WARP_ID=12), r"warp ids must be"),
    (dict(READ_TILE_ARRIVERS_TOT=22), r"READ_TILE_ARRIVERS_TOT must be 21.*hang at EVERY shape"),
    (dict(READ_TILE_ARRIVERS_TOT=20), r"READ_TILE_ARRIVERS_TOT must be 21.*advances a tile early"),
    (dict(CGA_M=4), r"the cluster IS the MMA pair"),
    # mbarrier lane constants
    (dict(SOFTMAX_WG_LANES=64), r"SOFTMAX_WG_LANES must be SOFTMAX_WG_WARPS\*32"),
    # the pre-port comment's "128" -- SOFTMAX_LANES is BOTH warpgroups (256)
    (dict(SOFTMAX_LANES=128), r"SOFTMAX_LANES must be every compute lane of one CTA \(8\*32\)"),
    (dict(SOFT_X_CTA_MMA=256), r"SOFT_X_CTA_MMA.*SOFTMAX_LANES\*CTA_MMA"),
    (dict(MMA_COMMIT_ARRIVES=32), r"ONE arrive per target CTA"),
    # geometry the bodies hardcode
    (dict(TILE_M=64), r"TILE_M = TILE_N = 128"),
    (dict(TILE_K=512), r"d_qk = d_v = 256"),
    (dict(CGA_M=1, CTA_MMA=1, READ_TILE_ARRIVERS_TOT=11, SOFT_X_CTA_MMA=256), r"single cga2 sub-group \(CGA_M=2, CGA_N=1, CTA_MMA=2\)"),
    (dict(TILES_Q=2), r"TILES_Q == 1"),
    (dict(SCHEDULER_STAGES=3), r"SCHEDULER_STAGES == 2"),
    (dict(N_BMM2_CHUNKS=4), r"N_BMM2_CHUNKS\*BMM2_CHUNK_SIZE must equal TILE_N"),
    # dtype bookkeeping
    (dict(IS_FP8=2), r"IS_FP8 must follow DTYPE_QKV"),
    (dict(BPE_O=3), r"BPE/BPE_O/BPE_DS must match their dtypes"),
    # rings common to both
    (dict(STAGES_KV=2), r"loaded ONCE per kv-block"),
    (dict(STAGES_TMEM_S=2), r"single-buffered"),
    (dict(STATS_STAGES=3), r"prefetch ring is 2-deep"),
    # swizzles
    (dict(Q_SWZ_BYTES=64), r"Q/K/dO/V swizzle must be 128 B"),
    (dict(P_SWZ_BYTES=64), r"ONE unit.*cos ~ 0.006"),
    (dict(dV_SWZ_BYTES=64), r"dV staging store_swizzled"),
    # masks
    (dict(CAUSAL_BOTTOM_RIGHT=1), r"bottom-right alignment requires a causal band"),
    (dict(SWA_WINDOW=64), r"MASK_SWA <=> SWA_WINDOW > 0"),
    (dict(SEQ_KV_LENS_PRESENT=1), r"MASK_PADDED <=> SEQ_KV_LENS_PRESENT.*attends the whole pad"),
    (dict(SCHEDULER_POLICY=5), r"SCHEDULER_POLICY must be 0/1/2"),
]


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
@pytest.mark.parametrize("overrides, match", _SHARED_CFG_REJECTS, ids=[",".join(o) for o, _ in _SHARED_CFG_REJECTS])
def test_rejects_every_shared_cfg_predicate(family, overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate(family, **overrides)


_FP8_CFG_REJECTS = [
    (dict(TILE_K_HW_BMM1=32, TILE_K_HW_BMM2=32), r"K=64 path and EVERY idesc must pass k_dim=1.*scrambles accumulator ROWS"),
    (dict(IDESC_K_DIM=0), r"EVERY idesc must pass k_dim=1"),
    (dict(N_BMM2_CHUNKS=4, BMM2_CHUNK_SIZE=32), r"BMM2_CHUNK_SIZE == TILE_K_HW_BMM2"),
    (dict(DTYPE_DS=DTYPE_E4M3, BPE_DS=1), r"dS workspace as BF16.*fp8 GEMM arm"),
    (dict(K_SPLIT_UTCCP=1), r"fp8 body has no BMM1 K-split"),
    (dict(STAGES_Q=2), r"Q / dO rings are 3-deep in this body"),
    (dict(STAGES_dO_DV=2), r"drives the dO_dv ring at STAGES_dO"),
    (dict(STAGES_TMEM_P=1), r"fp8 P ring is exactly 2 stages"),
    (dict(XFER_STAGES=2), r"fp8 dS SMEM ring is 3-deep"),
]


@pytest.mark.parametrize("overrides, match", _FP8_CFG_REJECTS, ids=[",".join(o) for o, _ in _FP8_CFG_REJECTS])
def test_rejects_every_fp8_cfg_predicate(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate(FAMILY_FP8, **overrides)


_F16_CFG_REJECTS = [
    (dict(TILE_K_HW_BMM1=32, TILE_K_HW_BMM2=32), r"TILE_K_HW must be 16.*silently wrong on SM10x"),
    (dict(IDESC_K_DIM=1), r"default idesc k_dim=0"),
    (dict(DTYPE_DS=DTYPE_FP16), r"dS workspace dtype IS the io dtype"),
    (dict(K_SPLIT_UTCCP=0), r"K_SPLIT_UTCCP must be 1"),
    (dict(STAGES_Q=3), r"Q / dO rings are 2-deep in this body"),
    (dict(STAGES_dO_DV=1), r"2-deep with stage 1 ALIASING the K back-half"),
    (dict(STAGES_TMEM_P=2), r"single buffer INSIDE S_acc"),
    (dict(XFER_STAGES=2), r"f16 dS SMEM ring is 1-deep"),
]


@pytest.mark.parametrize("overrides, match", _F16_CFG_REJECTS, ids=[",".join(o) for o, _ in _F16_CFG_REJECTS])
def test_rejects_every_f16_cfg_predicate(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate(FAMILY_F16, **overrides)


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
def test_rejects_a_slab_layout_over_the_rubin_cap(family, monkeypatch):
    """No field flip reaches the cap on a valid cfg (every depth is pinned), so
    shrink the budget: the message must carry the per-slab tally."""
    monkeypatch.setattr(cfgmod, "SMEM_USABLE_BYTES", 300 * _KiB)
    with pytest.raises(ValueError, match=r"exceed the 327 KiB Rubin oversized per-CTA cap \(sQ .* \| sdS .*\)"):
        _cfg(family)


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
def test_rejects_scaffolding_over_its_budget(family, monkeypatch):
    monkeypatch.setattr(cfgmod, "SMEM_SCAFFOLD_BYTES", 256)
    with pytest.raises(ValueError, match=r"declared scaffolding \(\d+ B of mbarriers \+ scheduler \+ tmem ptr\) exceeds the 256 B budget"):
        _cfg(family)


# ---------------------------------------------------------------------------
# Pins: the facts the kernel ports and the adapter read off this module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, slab_kib, names",
    [
        (FAMILY_F16, 322, ["sQ", "sdO", "sCombined[sdOdv_s0|K|V](+sdV alias)", "sStats", "sdS"]),
        (FAMILY_FP8, 306, ["sQ", "sdO", "sdOdv", "sExcl[K|V](+sdV alias)", "sStats", "sdS"]),
    ],
)
def test_smem_tally_and_declaration_order(family, slab_kib, names):
    cfg = _cfg(family)
    slabs = smem_layout(cfg)
    assert [s.name for s in slabs] == names
    assert smem_bytes(cfg) == slab_kib * _KiB
    assert kernel_smem_bytes(cfg) == slab_kib * _KiB + SMEM_SCAFFOLD_BYTES
    assert kernel_smem_bytes(cfg) <= SMEM_CAP_BYTES
    # contiguous, 1024-aligned, in declaration order
    off = 0
    for s in slabs:
        assert s.offset == off and s.nbytes % _KiB == 0
        off += s.nbytes
    assert scaffold_bytes_declared(cfg) < SMEM_SCAFFOLD_BYTES


def test_f16_layout_offsets_and_the_k_back_alias():
    cfg = _cfg(FAMILY_F16)
    roots = dict(desc_roots(cfg))
    assert roots["sQ[0]"] == 0 and roots["sQ[1]"] == 32 * _KiB
    assert roots["sdO[0]"] == 64 * _KiB and roots["sdO[1]"] == 96 * _KiB
    assert roots["sdO_dv[0]"] == 128 * _KiB
    assert roots["sK"] == 160 * _KiB and roots["sV"] == 224 * _KiB
    # the K-split alias is EXACT: dO_dv stage 1 == the UTCCP'd K back-half
    assert roots["sdO_dv[1](=sK_back alias)"] == roots["sK_back(UTCCP src)"] == 192 * _KiB
    b = buffer_elems(cfg)
    assert b._DODV_STAGE_ELEMS == b.dOBufferElems + b.K_BACK_OFF_ELEMS and b.dOBufferElems == b.K_BACK_OFF_ELEMS
    # sStats / sdS sit past the 256 KiB line but nothing descriptor-reads them
    by_name = {s.name: s for s in smem_layout(cfg)}
    assert by_name["sStats"].offset == 288 * _KiB and by_name["sdS"].offset == 290 * _KiB
    assert by_name["sStats"].roots == () and by_name["sdS"].roots == ()


def test_fp8_layout_offsets():
    cfg = _cfg(FAMILY_FP8)
    roots = dict(desc_roots(cfg))
    assert [roots[f"sQ[{s}]"] for s in range(3)] == [0, 16 * _KiB, 32 * _KiB]
    assert [roots[f"sdO[{s}]"] for s in range(3)] == [48 * _KiB, 64 * _KiB, 80 * _KiB]
    assert [roots[f"sdO_dv[{s}]"] for s in range(3)] == [96 * _KiB, 112 * _KiB, 128 * _KiB]
    assert roots["sK"] == 144 * _KiB and roots["sV"] == 176 * _KiB
    by_name = {s.name: s for s in smem_layout(cfg)}
    # bf16 dS: 3 x 32 KiB, and every dS figure follows BPE_DS, not BPE
    assert by_name["sdS"].nbytes == 96 * _KiB
    b = buffer_elems(cfg)
    assert (b.P_TMA_ITERS, b.P_D_BLOCK, b.pXferBytes) == (2, 64, 32 * _KiB)
    # e4m3 dV: 2 store subtiles of 128 d cols; a bf16 dV: 4 of 64
    assert (b.TMA_DV_ITERS, b.DV_D_BLOCK) == (2, 128)
    b2 = buffer_elems(_cfg(FAMILY_FP8, dtype_o=DTYPE_BF16))
    assert (b2.TMA_DV_ITERS, b2.DV_D_BLOCK) == (4, 64)


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
def test_every_descriptor_root_is_under_the_v0_window_so_desc_version_is_0(family):
    cfg = _cfg(family)
    for label, off in desc_roots(cfg):
        assert off < TCGEN05_V0_ADDR_LIMIT, f"{label} at {off}"
    assert desc_version(cfg) == 0 and cfgmod._needs_desc_v1(cfg) is False


def test_desc_version_flips_when_a_root_crosses_the_line():
    """The derivation is live, not a literal: a layout with a root past 256 KiB
    selects version 1.  (The f16 body at a third Q stage would; its depth pin
    rejects that cfg, so exercise the tally directly.)"""
    cfg = dataclasses.replace(_cfg(FAMILY_F16), STAGES_Q=3)
    assert dict(desc_roots(cfg))["sV"] == 256 * _KiB
    assert desc_version(cfg) == 1


def test_tmem_maps():
    f16 = tmem_layout(_cfg(FAMILY_F16))
    assert (f16.S_OFF, f16.S_COLS, f16.dP_OFF, f16.dP_COLS, f16.dV_OFF, f16.dV_COLS) == (0, 128, 128, 128, 256, 256)
    assert (f16.P_OFF, f16.P_COLS, f16.RSVD_OFF, f16.RSVD_COLS, f16.TOTAL_COLS) == (32, 64, 512, 64, TMEM_TOTAL_COLS)
    assert f16.S_COLS + f16.dP_COLS + f16.dV_COLS + f16.RSVD_COLS == TMEM_TOTAL_COLS
    fp8_cfg = _cfg(FAMILY_FP8)
    fp8 = tmem_layout(fp8_cfg)
    assert (fp8.P_OFF, fp8.P_COLS, fp8.RSVD_COLS) == (512, 32, 0)
    assert fp8.P_OFF + fp8_cfg.STAGES_TMEM_P * fp8.P_COLS == TMEM_TOTAL_COLS
    assert cfgmod.TMEM_IS_EXCLUSIVE is True


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
def test_scheduler_arrivers_and_warp_layout(family):
    cfg = _cfg(family)
    assert cfg.READ_TILE_ARRIVERS_TOT == read_tile_arrivers_tot(cfg) == 21 == 2 * (8 + 2) + 1
    assert (cfg.TOTAL_WARPS, cfg.THREADS_PER_CTA) == (12, 384)
    assert (cfg.SOFTMAX_WG0_BASE, cfg.SOFTMAX_WG1_BASE, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == (0, 4, 8, 9, 10, 11)
    # Register split: the per-warp sum must not exceed the 12-warp ENTRY pool 12 x 168 = 2016 -- the pool setmaxnreg
    # redistributes is the LAUNCH allocation, not the 2048-register file (8 x 232 + 4 x 48 = 2048 HUNG the first Rubin
    # launch, 2026-09-23: the last softmax INCREASE parks forever).  f16 224 / 56 (the 40-register MMA warp spilled one
    # slot on the bf16 sm_107a builds, so 8 registers move from the softmax warps to the service warps), fp8 the
    # pre-port 232 / 40.  Both balance the pool exactly.
    softmax, service = (224, 56) if family == FAMILY_F16 else (232, 40)
    assert (cfg.SOFTMAX_REGS, cfg.MMA_REGS, cfg.OTHER_REGS) == (softmax, service, service)
    assert 8 * cfg.SOFTMAX_REGS + 4 * cfg.MMA_REGS == 8 * softmax + 4 * service == cfgmod.reg_entry_pool(cfg.TOTAL_WARPS) == 2016 < cfgmod.REG_BUDGET_PER_CTA
    assert (cfg.SOFTMAX_WG_LANES, cfg.SOFTMAX_LANES, cfg.SOFT_X_CTA_MMA, cfg.MMA_COMMIT_ARRIVES) == (128, 256, 512, 1)


def test_mbar_inventory_differs_between_the_bodies_exactly_where_the_pipelines_do():
    f16, fp8 = mbar_stage_counts(_cfg(FAMILY_F16)), mbar_stage_counts(_cfg(FAMILY_FP8))
    assert set(f16) - set(fp8) == {"mb_k_utccp_done"}  # the K-split alias seam
    assert set(fp8) - set(f16) == {"mb_s_acc_empty"}  # the lookahead's S WAR handshake
    assert (fp8["mb_q_full"], fp8["mb_p_ready"], fp8["mb_ds_smem_full"]) == (3, 2, 3)
    assert (f16["mb_q_full"], f16["mb_p_ready"], f16["mb_ds_smem_full"], f16["mb_dodv_full"]) == (2, 1, 1, 2)


# ---------------------------------------------------------------------------
# Shape-time helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_q, h_kv, chunk", [(8, 8, 1), (8, 8, 8), (8, 2, 4), (8, 2, 8), (128, 8, 16), (64, 8, 64)])
def test_head_chunk_accepts_whole_gqa_groups(h_q, h_kv, chunk):
    validate_head_chunk(h_q, h_kv, chunk)


@pytest.mark.parametrize(
    "h_q, h_kv, chunk, match",
    [
        (8, 3, 1, r"positive multiple of H_kv"),
        (8, 2, 3, r"positive divisor of H_q"),
        (8, 2, 0, r"positive divisor of H_q"),
        (8, 2, 2, r"multiple of the GQA group H_q/H_kv = 4.*fold over a chunk is partial"),
    ],
)
def test_head_chunk_rejects(h_q, h_kv, chunk, match):
    with pytest.raises(ValueError, match=match):
        validate_head_chunk(h_q, h_kv, chunk)


def test_workspace_and_grid_helpers():
    cfg = _cfg(FAMILY_FP8)
    assert (cfgmod.q_pad_rows(cfg), cfgmod.kv_pad_rows(cfg)) == (128, 256)
    # bf16 dS on the fp8 chain: 2 B per cell
    assert ds_workspace_bytes(cfg, 1, 128, 8192, 8192) == 128 * 8192 * 8192 * 2
    with pytest.raises(ValueError, match=r"padded to q 128 / kv 256 rows"):
        ds_workspace_bytes(cfg, 1, 1, 8000, 8192)
    assert launch_grid(cfg, 2, 8, 8192) == ((32 * 2, 8, 2), (2, 1, 1))
    lpt = _cfg(FAMILY_FP8, sched_policy=SCHED_LPT)
    assert launch_grid(lpt, 2, 8, 8192) == ((32 * 8 * 2 * 2, 1, 1), (2, 1, 1))
