# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.bwd.config_sm107``: the Rubin d256 backward configs, host-only.

Every predicate of ``_validate_params`` / ``_validate_cfg_d256_bwd`` gets ONE
raising case (``pytest.raises(ValueError, match=...)`` on the failure signature
the message carries), and every dtype the three bodies serve gets an accepting
case.  The layout facts the kernel ports and the adapter rely on (SMEM tally,
descriptor roots, TMEM map, scheduler arrivers, scaffolding) are pinned as
values, not just as "does not raise" -- a config that silently moved one of
them is exactly the class of bug the module exists to make impossible.

The MXFP8 family (``FAMILY_MXFP8``) is pinned as a TALLY: every SMEM offset and
descriptor root of its 13-slab table (the six SF slabs before sdS, no SMEM P
ring), the fp8 TMEM map with the SLOT-RELATIVE scale-factor alias offsets, the
fp8 barrier inventory plus the one added commit ring ``mb_p_sf_consumed``, and
the register split -- the numbers a kernel body reads off this module.  The f16 and fp8 families are pinned byte-for-byte to
their values at develop dd3235c3 (PR #1212), taken from the code BEFORE the
third family landed, so the family cannot have moved either sibling.

Device-independent: nothing here compiles or launches.  The TMEM predicates
that are tautological on a derived ``tmem_layout`` (sum to 576, P width, the
two alias bands inside one P slot) are pinned as layout VALUES rather than
provoked, since no field flip reaches them before an earlier pin raises.

Also pinned: the -1 policy inherit is ``DS_SF_POLICY_DEFAULT`` -- P-b, the
shipped block-scaled chain, since the Rubin A/B (P-c is the bf16-dS oracle
twin; P-a is optional validation only), ``ds_workspace_bytes`` counts
``DS_PAYLOADS`` rings, P-a stages TWO dS SF atoms per stage (one per GEMM
orientation), the zero-field pin is literal, and the SF atom helpers are
pinned against the forward config's.
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
    DS_SF_NONE,
    DS_SF_P_A,
    DS_SF_P_B,
    DS_SF_P_C,
    DS_SF_POLICY_DEFAULT,
    FAMILY_F16,
    FAMILY_FP8,
    FAMILY_MXFP8,
    MXFP8_P_SCALE_LOG2,
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
    sf_smem_bytes,
    sf_tmem_cols,
    smem_bytes,
    smem_layout,
    tmem_layout,
    validate_head_chunk,
)
from cudnn.sdpa.fwd.config_sm107 import _d256_mxfp8_sf_sizes

pytestmark = [pytest.mark.L0]

_KiB = 1024
_FAMILIES = [FAMILY_F16, FAMILY_FP8, FAMILY_MXFP8]
_POLICIES = [DS_SF_P_A, DS_SF_P_B, DS_SF_P_C]
_POLICY_IDS = {DS_SF_P_A: "P-a", DS_SF_P_B: "P-b", DS_SF_P_C: "P-c"}


def _cfg(family, **params):
    dq = {FAMILY_F16: DTYPE_BF16, FAMILY_FP8: DTYPE_E4M3, FAMILY_MXFP8: DTYPE_E4M3}[family]
    return make_cfg_d256_bwd(TemplateParams(dtype_qkv=params.pop("dtype_qkv", dq), **params), family)


def _validate(family, **overrides):
    """Flip fields on a VALID cfg and re-run the cfg validator."""
    params = overrides.pop("_params", {})
    cfg = dataclasses.replace(_cfg(family, **params), **overrides)
    cfgmod._validate_cfg_d256_bwd(cfg, cfgmod._FLAVOR[family])


# ---------------------------------------------------------------------------
# Accept: every dtype member of the three bodies, and the record shapes the adapter builds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, dtype_qkv, dtype_o, dtype_ds, want_o, want_ds",
    [
        (FAMILY_F16, DTYPE_BF16, -1, -1, DTYPE_BF16, DTYPE_BF16),
        (FAMILY_F16, DTYPE_FP16, -1, -1, DTYPE_FP16, DTYPE_FP16),
        (FAMILY_F16, DTYPE_BF16, DTYPE_FP16, -1, DTYPE_FP16, DTYPE_BF16),
        (FAMILY_F16, DTYPE_FP16, -1, DTYPE_FP16, DTYPE_FP16, DTYPE_FP16),
        # fp8: E4M3 io; grads default to E4M3 (the fp8 graph contract); dS defaults to E4M3 (dS_q = e4m3(dS * scale_dP) for the
        # fp8 GEMM arm) and takes BF16 explicitly (the pre-quantized twin the bf16 GEMM renderings read)
        (FAMILY_FP8, DTYPE_E4M3, -1, -1, DTYPE_E4M3, DTYPE_E4M3),
        (FAMILY_FP8, DTYPE_E4M3, DTYPE_BF16, -1, DTYPE_BF16, DTYPE_E4M3),
        (FAMILY_FP8, DTYPE_E4M3, DTYPE_FP16, -1, DTYPE_FP16, DTYPE_E4M3),
        (FAMILY_FP8, DTYPE_E4M3, -1, DTYPE_E4M3, DTYPE_E4M3, DTYPE_E4M3),
        (FAMILY_FP8, DTYPE_E4M3, -1, DTYPE_BF16, DTYPE_E4M3, DTYPE_BF16),
        (FAMILY_FP8, DTYPE_E4M3, DTYPE_BF16, DTYPE_BF16, DTYPE_BF16, DTYPE_BF16),
        # mxfp8: E4M3 payloads; grads are the graph's half dtype (inherit -> BF16); dS follows the policy, and the -1 inherit is
        # DS_SF_POLICY_DEFAULT = P-b (the shipped block-scaled chain), so the default record reads an e4m3 dS (two payloads)
        (FAMILY_MXFP8, DTYPE_E4M3, -1, -1, DTYPE_BF16, DTYPE_E4M3),
        (FAMILY_MXFP8, DTYPE_E4M3, DTYPE_FP16, -1, DTYPE_FP16, DTYPE_E4M3),
        (FAMILY_MXFP8, DTYPE_E4M3, DTYPE_BF16, DTYPE_E4M3, DTYPE_BF16, DTYPE_E4M3),
    ],
)
def test_accepts_every_dtype_member(family, dtype_qkv, dtype_o, dtype_ds, want_o, want_ds):
    cfg = _cfg(family, dtype_qkv=dtype_qkv, dtype_o=dtype_o, dtype_ds=dtype_ds)
    assert (cfg.DTYPE_QKV, cfg.DTYPE_O, cfg.DTYPE_DS) == (dtype_qkv, want_o, want_ds)
    assert (cfg.BPE, cfg.BPE_O, cfg.BPE_DS) == (cfgmod.bpe(dtype_qkv), cfgmod.bpe(want_o), cfgmod.bpe(want_ds))
    # IS_FP8 = E4M3 payloads (the fp8-CLASS bodies: per-tensor fp8 AND MXFP8); IS_MXFP8 = the block-scaled one
    assert cfg.IS_FP8 == (family != FAMILY_F16) and cfg.IS_MXFP8 == (family == FAMILY_MXFP8)


@pytest.mark.parametrize("dtype_o", [DTYPE_BF16, DTYPE_FP16], ids=["o_bf16", "o_fp16"])
@pytest.mark.parametrize("policy", _POLICIES, ids=[_POLICY_IDS[p] for p in _POLICIES])
def test_mxfp8_accepts_every_policy_arm_per_grad_dtype(policy, dtype_o):
    """One accepting case per dS scale-factor policy x gradient dtype, with the policy's derived facts:
    P-a = one e4m3 payload + TWO SF atoms per stage (the one tile byte expanded once per GEMM orientation, sf_ds_dk and
    sf_ds_dq) at the fp8 body's 3-deep ring; P-b = two e4m3 payloads + two SF atoms (one per payload) at a
    2-deep ring (no rcp gather slab: the kernel keeps the warp-uniform column scales in registers); P-c = a bf16 dS at a 2-deep
    ring and no scale factors."""
    cfg = _cfg(FAMILY_MXFP8, dtype_o=dtype_o, ds_sf_policy=policy)
    want_ds = DTYPE_BF16 if policy == DS_SF_P_C else DTYPE_E4M3
    assert (cfg.DTYPE_O, cfg.BPE_O, cfg.DTYPE_DS, cfg.BPE_DS) == (dtype_o, 2, want_ds, cfgmod.bpe(want_ds))
    assert cfg.DS_SF_POLICY == policy
    assert cfg.XFER_STAGES == (3 if policy == DS_SF_P_A else 2)
    assert cfg.DS_PAYLOADS == (2 if policy == DS_SF_P_B else 1)
    assert cfg.DS_SF_ATOMS == {DS_SF_P_A: 2, DS_SF_P_B: 2, DS_SF_P_C: 0}[policy]
    # an explicit dS dtype that agrees with the policy is accepted too
    assert _cfg(FAMILY_MXFP8, dtype_o=dtype_o, ds_sf_policy=policy, dtype_ds=want_ds) == cfg
    # the two adapter-set trace-time flags ride into the cfg as 0/1
    flagged = _cfg(FAMILY_MXFP8, dtype_o=dtype_o, ds_sf_policy=policy, scaled_fp8_pack=True, mask_q_pad=True)
    assert (flagged.SCALED_FP8_PACK, flagged.MASK_Q_PAD) == (1, 1) and (cfg.SCALED_FP8_PACK, cfg.MASK_Q_PAD) == (0, 0)
    assert smem_layout(flagged) == smem_layout(cfg) and tmem_layout(flagged) == tmem_layout(cfg), "the flags move no bytes"


def test_mxfp8_family_default_policy_is_the_shipped_p_b():
    """The SHIPPED policy is P-b (exact 1x32 block-scaled e4m3 dS both ways: two payloads + two staged atoms at a 2-deep ring), flipped
    from the bring-up twin P-c (bf16 dS, the oracle twin, kept as a built arm) once it beat the bf16-dS chain on Rubin; P-a is
    optional validation only.  The -1 inherit is the ONE module constant DS_SF_POLICY_DEFAULT -- P-b, never P-a (the retired arm an
    earlier revision of this module defaulted to), and the P-c record is still reachable explicitly."""
    assert DS_SF_POLICY_DEFAULT == DS_SF_P_B and DS_SF_POLICY_DEFAULT != DS_SF_P_A
    cfg = _cfg(FAMILY_MXFP8)
    assert cfg == _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_POLICY_DEFAULT) == _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_B)
    assert cfg.DS_SF_POLICY == DS_SF_P_B and cfg.DTYPE_DS == DTYPE_E4M3 and cfg.BPE_DS == 1 and cfg.DTYPE_O == DTYPE_BF16
    assert (cfg.XFER_STAGES, cfg.DS_PAYLOADS, cfg.DS_SF_ATOMS) == (2, 2, 2)
    assert (cfg.SCALED_FP8_PACK, cfg.MASK_Q_PAD) == (0, 0)
    pc = _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_C)
    assert pc.DS_SF_POLICY == DS_SF_P_C and pc.DTYPE_DS == DTYPE_BF16 and (pc.XFER_STAGES, pc.DS_PAYLOADS, pc.DS_SF_ATOMS) == (2, 1, 0)
    # the fp8-class pipeline facts the MXFP8 body inherits from the per-tensor fp8 one (the dS ring depth is the policy's:
    # only the P-a arm keeps the fp8 body's 3-deep ring)
    fp8 = _cfg(FAMILY_FP8)
    for f in ("TILE_K_HW_BMM1", "TILE_K_HW_BMM2", "IDESC_K_DIM", "K_SPLIT_UTCCP", "STAGES_Q", "STAGES_dO", "STAGES_dO_DV", "BMM2_CHUNK_SIZE"):
        assert getattr(cfg, f) == getattr(fp8, f), f
    assert _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_A).XFER_STAGES == fp8.XFER_STAGES == 3


def test_accepts_the_shared_sm100_record_unextended():
    """A plain bwd TemplateParams (no dtype_o / has_sink / MXFP8 fields) is shape-compatible."""
    cfg = make_cfg_d256_bwd(
        BaseTemplateParams(dtype_qkv=DTYPE_FP16, window_right=0, bottom_right=True, window_left=256, seq_kv_lens_present=True, sched_policy=SCHED_LPT),
        FAMILY_F16,
    )
    assert cfg.MASK_FLAGS == MASK_CAUSAL | MASK_SWA | MASK_PADDED
    assert (cfg.SWA_WINDOW, cfg.CAUSAL_BOTTOM_RIGHT, cfg.SEQ_KV_LENS_PRESENT, cfg.HAS_SINK, cfg.SCHEDULER_POLICY) == (256, 1, 1, 0, SCHED_LPT)
    assert cfg.DTYPE_O == DTYPE_FP16  # inherits the io dtype
    # and on the MXFP8 body the missing extras take the family defaults (DS_SF_POLICY_DEFAULT = P-b, bf16 grads, e4m3 dS payloads)
    mx = make_cfg_d256_bwd(BaseTemplateParams(dtype_qkv=DTYPE_E4M3, window_right=0), FAMILY_MXFP8)
    assert (mx.DS_SF_POLICY, mx.DTYPE_O, mx.DTYPE_DS, mx.MASK_FLAGS) == (DS_SF_POLICY_DEFAULT, DTYPE_BF16, DTYPE_E4M3, MASK_CAUSAL)


def test_template_params_extend_append_only():
    """The Rubin record appends to the shared SM100 one; new fields go LAST with defaults (positional callers across the
    adapters must keep working)."""
    base = [f.name for f in dataclasses.fields(BaseTemplateParams)]
    ours = [f.name for f in dataclasses.fields(TemplateParams)]
    assert ours[: len(base)] == base
    assert ours[len(base) :] == ["dtype_o", "dtype_ds", "has_sink", "scaled_fp8_pack", "mask_q_pad", "ds_sf_policy"]
    defaults = {f.name: f.default for f in dataclasses.fields(TemplateParams)}
    assert (defaults["scaled_fp8_pack"], defaults["mask_q_pad"], defaults["ds_sf_policy"]) == (False, False, -1)


@pytest.mark.parametrize("family", _FAMILIES)
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


@pytest.mark.parametrize("family", _FAMILIES)
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
        (FAMILY_FP8, dict(dtype_ds=DTYPE_FP16), r"dtype_ds must be -1 \(inherit -> E4M3.*DTYPE_BF16 \(the pre-quantized dS"),
        (FAMILY_FP8, dict(dtype_ds=DTYPE_E5M2), r"dtype_ds must be -1 \(inherit -> E4M3"),
        (FAMILY_F16, dict(dtype_ds=DTYPE_E4M3), r"dtype_ds must be -1 or the io dtype"),
        (FAMILY_F16, dict(dtype_qkv=DTYPE_BF16, dtype_ds=DTYPE_FP16), r"dtype_ds must be -1 or the io dtype \(2\)"),
        (FAMILY_F16, dict(window_left=0), r"SWA requires window_left > 0"),
        (FAMILY_FP8, dict(window_left=-5), r"SWA requires window_left > 0"),
        (FAMILY_F16, dict(window_right=64), r"window_right must be 0 when set.*Right-band widening"),
        (FAMILY_FP8, dict(bottom_right=True), r"bottom_right alignment requires a causal band"),
        (FAMILY_F16, dict(seq_q_lens_present=True, seq_kv_lens_present=True), r"seq_q_lens_present is not implemented"),
        (FAMILY_FP8, dict(thd_varlen=True), r"thd_varlen is not implemented"),
        (FAMILY_F16, dict(sched_policy=3), r"sched_policy must be one of NATURAL/LPT/LPT_L2"),
        # the MXFP8 record: E4M3 payloads, half gradients, a policy-consistent dS dtype
        (FAMILY_MXFP8, dict(dtype_qkv=DTYPE_E5M2), r"MXFP8 body takes E4M3 payloads.*E5M2 payloads are not implemented"),
        (FAMILY_MXFP8, dict(dtype_qkv=DTYPE_BF16), r"MXFP8 body takes E4M3 payloads.*belongs to the f16 body"),
        (FAMILY_MXFP8, dict(dtype_o=DTYPE_E4M3), r"dtype_o must be -1 \(inherit -> BF16\).*gradients are half precision"),
        (FAMILY_MXFP8, dict(dtype_o=9), r"dtype_o must be -1 \(inherit -> BF16\)"),
        (
            FAMILY_MXFP8,
            dict(ds_sf_policy=7),
            r"ds_sf_policy must be -1 \(DS_SF_POLICY_DEFAULT = P-b, the shipped block-scaled chain.*or one of DS_SF_P_A/P_B/P_C",
        ),
        (FAMILY_MXFP8, dict(ds_sf_policy=DS_SF_NONE), r"ds_sf_policy must be -1 \(DS_SF_POLICY_DEFAULT = P-b.*P-c is the bf16-dS oracle twin"),
        (
            FAMILY_MXFP8,
            dict(ds_sf_policy=DS_SF_P_A, dtype_ds=DTYPE_BF16),
            r"dtype_ds must be -1 or DTYPE_E4M3 \(0\) under P-a.*e4m3 dS payload \+ E8M0 scale factors",
        ),
        (FAMILY_MXFP8, dict(ds_sf_policy=DS_SF_P_B, dtype_ds=DTYPE_BF16), r"dtype_ds must be -1 or DTYPE_E4M3 \(0\) under P-b"),
        (
            FAMILY_MXFP8,
            dict(ds_sf_policy=DS_SF_P_C, dtype_ds=DTYPE_E4M3),
            r"dtype_ds must be -1 or DTYPE_BF16 \(2\) under P-c.*bf16 dS the bf16 renderings read",
        ),
        # the default arm IS P-b: an FP16 dS on the bare record is refused with P-b's message
        (FAMILY_MXFP8, dict(dtype_ds=DTYPE_FP16), r"dtype_ds must be -1 or DTYPE_E4M3 \(0\) under P-b.*e4m3 dS payload \+ E8M0 scale factors"),
        # the shared record predicates hold on the third body too (the code path is shared today; this keeps it so)
        (FAMILY_MXFP8, dict(window_left=0), r"SWA requires window_left > 0"),
        (FAMILY_MXFP8, dict(window_right=64), r"window_right must be 0 when set.*Right-band widening"),
        (FAMILY_MXFP8, dict(bottom_right=True), r"bottom_right alignment requires a causal band"),
        (FAMILY_MXFP8, dict(seq_q_lens_present=True, seq_kv_lens_present=True), r"seq_q_lens_present is not implemented"),
        (FAMILY_MXFP8, dict(thd_varlen=True), r"thd_varlen is not implemented"),
        (FAMILY_MXFP8, dict(sched_policy=3), r"sched_policy must be one of NATURAL/LPT/LPT_L2"),
        # the MXFP8-only record fields are rejected, not ignored, on the other two bodies
        (FAMILY_FP8, dict(ds_sf_policy=DS_SF_P_A), r"ds_sf_policy is the MXFP8 family's.*write no dS scale factors"),
        (FAMILY_F16, dict(ds_sf_policy=DS_SF_P_C), r"ds_sf_policy is the MXFP8 family's"),
        (FAMILY_FP8, dict(scaled_fp8_pack=True), r"scaled_fp8_pack is the MXFP8 body's Rubin fused cvt arm"),
        (FAMILY_F16, dict(scaled_fp8_pack=True), r"scaled_fp8_pack is the MXFP8 body's Rubin fused cvt arm"),
        (FAMILY_FP8, dict(mask_q_pad=True), r"mask_q_pad is the MXFP8 body's q-pad band"),
        (FAMILY_F16, dict(mask_q_pad=True), r"mask_q_pad is the MXFP8 body's q-pad band"),
    ],
)
def test_rejects_a_record_the_body_cannot_express(family, params, match):
    with pytest.raises(ValueError, match=match):
        _cfg(family, **params)


def test_rejects_an_unknown_family():
    with pytest.raises(ValueError, match=r"dtype_family must be one of"):
        make_cfg_d256_bwd(TemplateParams(dtype_qkv=DTYPE_BF16), "nvfp4")


# ---------------------------------------------------------------------------
# Reject: every cfg predicate, one flipped field each (every family where the
# predicate is shared, the owning family where it is not)
# ---------------------------------------------------------------------------

_SHARED_CFG_REJECTS = [
    # register split
    # 32: distinct from every family's service count (f16 56, fp8 / mxfp8 40) and keeps the sum under the pool, so only the equality predicate fires
    (dict(MMA_REGS=32), r"MMA/TMALDG/TMASTG/SCHEDULER regs must be equal"),
    (dict(SOFTMAX_REGS=240), r"register split (2144|2080) over the 12-warp ENTRY pool 2016"),  # f16 8x240+4x56 / fp8, mxfp8 8x240+4x40
    (dict(SOFTMAX_REGS=220), r"multiple of 8"),  # under the pool on every family, so only the granule predicate fires
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
    # rings common to all
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


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("overrides, match", _SHARED_CFG_REJECTS, ids=[",".join(o) for o, _ in _SHARED_CFG_REJECTS])
def test_rejects_every_shared_cfg_predicate(family, overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate(family, **overrides)


# the fp8-CLASS predicates (E4M3 payloads): the per-tensor fp8 body and the MXFP8 body share the pipeline they pin
_FP8_CLASS_CFG_REJECTS = [
    (dict(TILE_K_HW_BMM1=32, TILE_K_HW_BMM2=32), r"K=64 path and EVERY idesc must pass k_dim=1.*scrambles accumulator ROWS"),
    (dict(IDESC_K_DIM=0), r"EVERY idesc must pass k_dim=1"),
    (dict(N_BMM2_CHUNKS=4, BMM2_CHUNK_SIZE=32), r"BMM2_CHUNK_SIZE == TILE_K_HW_BMM2"),
    # a dS subtile narrower than a warpgroup's q half: P_D_BLOCK = TILE_N * BPE_DS / dS_SWZ_BYTES = 32 at a 32-B swizzle row
    (dict(dS_SWZ_BYTES=32, P_SWZ_BYTES=32), r"whole warpgroup q halves|ONE unit and must both be 128 B"),
    (dict(K_SPLIT_UTCCP=1), r"has no BMM1 K-split"),
    (dict(STAGES_Q=2), r"Q / dO rings are 3-deep in this body"),
    (dict(STAGES_dO_DV=2), r"drives the dO_dv ring at STAGES_dO"),
]


@pytest.mark.parametrize("family", [FAMILY_FP8, FAMILY_MXFP8])
@pytest.mark.parametrize("overrides, match", _FP8_CLASS_CFG_REJECTS, ids=[",".join(o) for o, _ in _FP8_CLASS_CFG_REJECTS])
def test_rejects_every_fp8_class_cfg_predicate(family, overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate(family, **overrides)


_FP8_CFG_REJECTS = [
    (dict(DTYPE_DS=DTYPE_FP16, BPE_DS=2), r"dS workspace is E4M3 .*or BF16 .*silent garbage gradients"),
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


# One raising case per MXFP8 predicate; ``_params`` selects the policy arm the flip
# is applied to (the bare record is DS_SF_POLICY_DEFAULT = P-b; the P-a / P-c rows say so).
_MXFP8_CFG_REJECTS = [
    # P is NOT in TMEM: the scale-factor columns take the tail the fp8 body's P ring occupied
    (dict(STAGES_TMEM_P=1), r"MXFP8 P ring is the fp8 body's: exactly 2 stages.*NO dedicated scale-factor columns.*ONE dead slot per q iteration"),
    (dict(STAGES_TMEM_P=3), r"MXFP8 P ring is the fp8 body's: exactly 2 stages.*a deeper ring puts the slot rule"),
    # SF geometry in the general rows x K-chunks form (R-19)
    (dict(SF_TMEM_COLS_K=9), r"every SF TMEM count is 4 x ceil\(rows/128\) x ceil\(K/128\).*got K 9"),
    (dict(SF_TMEM_COLS_P=8), r"SF_TMEM_COLS_P is 4 x ceil\(TILE_M/128\) x ceil\(TILE_N/128\).*NOT 4 x ceil\(TILE_N/128\)"),
    (dict(SF_SMEM_K=512), r"every SF slab is the operand's F8_128x4 atom count x 512 B.*got K 512"),
    (dict(SF_SMEM_dOT=512), r"every SF slab is the operand's F8_128x4 atom count x 512 B.*dO_T 512"),
    (dict(SF_BLOCK=16), r"32 elements share one E8M0 byte"),
    (dict(SF_BLOCKS_PER_STEP=1), r"sf_blocks_per_step=\) is the K64 hardware k-step over 32-element blocks = 2"),
    # P's fixed scale and its descale byte are ONE fact
    (dict(P_SCALE_LOG2=127, P_SF_BYTE=0), r"P_q = e4m3\(P \* 2\^P_SCALE_LOG2\).*127 - P_SCALE_LOG2"),
    (dict(P_SF_BYTE=127), r"a byte that disagrees with the scale rescales dV by 2\^k silently"),
    # expect_tx grows by the SF bytes of BOTH CTAs
    (dict(K_SF_TX=1024), r"grows each _full expect_tx by the SF bytes of BOTH CTAs.*got K 1024"),
    (dict(Q_SF_TX=4096), r"grows each _full expect_tx by the SF bytes of BOTH CTAs.*Q 4096"),
    # the policy couplings: dS dtype, ring depth, payload rings, staged SF atoms
    (dict(DS_SF_POLICY=DS_SF_NONE), r"DS_SF_POLICY must be DS_SF_P_A / P_B / P_C \(1/2/3\) on the MXFP8 body"),
    (dict(_params=dict(ds_sf_policy=DS_SF_P_A), DTYPE_DS=DTYPE_BF16, BPE_DS=2), r"under P-a the dS workspace is e4m3 \+ E8M0 scale factors.*got DTYPE_DS=2"),
    (dict(_params=dict(ds_sf_policy=DS_SF_P_C), DTYPE_DS=DTYPE_E4M3, BPE_DS=1), r"under P-c the dS workspace is bf16.*got DTYPE_DS=0"),
    (dict(_params=dict(ds_sf_policy=DS_SF_P_A), XFER_STAGES=2), r"the P-a dS ring is 3-deep \(the fp8 body's validated depth"),
    # The 2-deep ring was MEASURED on the MXFP8 body 2026-09-30 (48 / 0 / 0 fresh processes at n_q_tiles 1-4): the message carries the date, not PENDING.
    (
        dict(_params=dict(ds_sf_policy=DS_SF_P_C), XFER_STAGES=3),
        r"a 2-deep dS ring: the bf16 dS ring does not fit at 3 stages.*validated on the MXFP8 body at n_q_tiles 1/2/3/4 on 2026-09-30",
    ),
    (dict(_params=dict(ds_sf_policy=DS_SF_P_B), XFER_STAGES=3), r"a 2-deep dS ring: P-b's second e4m3 payload ring does not fit at 3 stages"),
    (dict(DS_PAYLOADS=1), r"P-b writes TWO dS payload rings.*every other policy one; got DS_PAYLOADS=1 under P-b"),
    (dict(_params=dict(ds_sf_policy=DS_SF_P_C), DS_PAYLOADS=2), r"P-b writes TWO dS payload rings.*every other policy one; got DS_PAYLOADS=2 under P-c"),
    (
        dict(_params=dict(ds_sf_policy=DS_SF_P_A), DS_SF_ATOMS=1),
        r"dS SF atoms staged per ring stage: P-a 2 \(the ONE 32x32 tile byte.*P-b 2.*P-c 0.*got 1 under P-a",
    ),
    (dict(_params=dict(ds_sf_policy=DS_SF_P_C), DS_SF_ATOMS=1), r"dS SF atoms staged per ring stage.*got 1 under P-c"),
    # P-b body geometry: both payload halves in ONE 128-B ring row (a 64-B store swizzle halves P_D_BLOCK -> the atom offsets move)
    (
        dict(_params=dict(ds_sf_policy=DS_SF_P_B), dS_SWZ_BYTES=64),
        r"P-b stages ONE 512-B atom per orientation per dS ring stage \(sf_ds_dk at \+0, sf_ds_dq at \+512\).*P_D_BLOCK == TILE_N at BPE_DS 1.*got .*P_D_BLOCK=64",
    ),
    # the gradient dtype is the graph's half dtype
    (dict(DTYPE_O=DTYPE_E4M3, BPE_O=1), r"MXFP8 backward's dV is half precision.*no quantized gradient and no amax"),
    # the two trace-time flags
    (dict(SCALED_FP8_PACK=2), r"SCALED_FP8_PACK / MASK_Q_PAD are 0/1 trace-time constants"),
    (dict(MASK_Q_PAD=2), r"SCALED_FP8_PACK / MASK_Q_PAD are 0/1 trace-time constants"),
    # (the SMEM-P-ring store-swizzle coupling retired with the ring: a Q_SWZ / P_SWZ flip now fires the shared swizzle predicates)
]


@pytest.mark.parametrize("overrides, match", _MXFP8_CFG_REJECTS, ids=[",".join(k for k in o if k != "_params") for o, _ in _MXFP8_CFG_REJECTS])
def test_rejects_every_mxfp8_cfg_predicate(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate(FAMILY_MXFP8, **dict(overrides))


# Every MXFP8-only CfgBwdD256 field -- the fields declared after BMM2_CHUNK_SIZE (the last shared one) -- LITERALLY.  A pin
# that iterated the module's own _MXFP8_ZERO_FIELDS could not fail when a field was dropped from it (found by mutation testing).
_MXFP8_ONLY_CFG_FIELDS = (
    "IS_MXFP8",
    "SF_BLOCK",
    "SF_BLOCKS_PER_STEP",
    "P_SCALE_LOG2",
    "P_SF_BYTE",
    "SF_SMEM_K",
    "SF_SMEM_V",
    "SF_SMEM_Q",
    "SF_SMEM_dO",
    "SF_SMEM_P",
    "SF_SMEM_dOT",
    "SF_SMEM_dS",
    "SF_TMEM_COLS_K",
    "SF_TMEM_COLS_V",
    "SF_TMEM_COLS_Q",
    "SF_TMEM_COLS_dO",
    "SF_TMEM_COLS_P",
    "SF_TMEM_COLS_dOT",
    "K_SF_TX",
    "V_SF_TX",
    "Q_SF_TX",
    "dO_SF_TX",
    "dOT_SF_TX",
    "DS_SF_POLICY",
    "DS_PAYLOADS",
    "DS_SF_ATOMS",
    "SCALED_FP8_PACK",
    "MASK_Q_PAD",
)
# the neutral (f16 / fp8) value of each: 0, except the payload ring count
_MXFP8_NEUTRAL = {name: (1 if name == "DS_PAYLOADS" else 0) for name in _MXFP8_ONLY_CFG_FIELDS}


def test_the_mxfp8_zero_field_tuple_is_every_mxfp8_only_cfg_field():
    """The validator's _MXFP8_ZERO_FIELDS must name every MXFP8-only field but DS_PAYLOADS (neutral value 1, pinned
    separately): checked against the dataclass (a field appended after BMM2_CHUNK_SIZE without a zero pin turns this red)
    AND against the literal tuple above (a field dropped from the module's tuple turns it red)."""
    names = [f.name for f in dataclasses.fields(cfgmod.CfgBwdD256)]
    assert tuple(names[names.index("BMM2_CHUNK_SIZE") + 1 :]) == _MXFP8_ONLY_CFG_FIELDS
    assert set(cfgmod._MXFP8_ZERO_FIELDS) == set(_MXFP8_ONLY_CFG_FIELDS) - {"DS_PAYLOADS"}
    assert len(set(cfgmod._MXFP8_ZERO_FIELDS)) == len(cfgmod._MXFP8_ZERO_FIELDS) == 27
    assert cfgmod._MXFP8_ZERO_FIELDS[0] == "IS_MXFP8", "first, so the message names the family flag before the SF fields"
    # the neutral values ARE the dataclass defaults (a bare CfgBwdD256 is SF-free)
    assert {n: getattr(cfgmod.CfgBwdD256(), n) for n in _MXFP8_ONLY_CFG_FIELDS} == _MXFP8_NEUTRAL


@pytest.mark.parametrize("family", [FAMILY_F16, FAMILY_FP8])
@pytest.mark.parametrize("name", [n for n in _MXFP8_ONLY_CFG_FIELDS if n != "IS_MXFP8"])
def test_rejects_mxfp8_fields_on_the_other_bodies(family, name):
    """The f16 / per-tensor fp8 bodies load no scale factors: EVERY MXFP8 field left non-neutral on them (non-zero;
    DS_PAYLOADS != 1) is a claim the body cannot honour, and the validator names the field.  (IS_MXFP8 itself re-routes the
    validator arm -- test_the_is_mxfp8_flag_and_the_sf_fields_are_one_fact.)"""
    with pytest.raises(ValueError, match=r"every MXFP8 field must be 0.*" + name):
        _validate(family, **{name: _MXFP8_NEUTRAL[name] + 1})


def test_the_is_mxfp8_flag_and_the_sf_fields_are_one_fact():
    """IS_MXFP8 selects the validator arm.  Dropped on the MXFP8 cfg, the per-tensor arm sees SF fields the fp8 body cannot
    honour; raised on the fp8 cfg, the MXFP8 arm's first structural predicate sees the zero-sized scale factors."""
    with pytest.raises(ValueError, match=r"every MXFP8 field must be 0"):
        _validate(FAMILY_MXFP8, IS_MXFP8=0)
    with pytest.raises(ValueError, match=r"32 elements share one E8M0 byte.*got 0"):
        _validate(FAMILY_FP8, IS_MXFP8=1)


def test_mxfp8_rejects_a_descriptor_root_past_the_v0_window(monkeypatch):
    """The ordering rule as a raise: every UTCCP / MMA root must stay under the version-0 window.  No field flip reaches it on
    the shipped order (the depth pins fire first), so shrink the window under the highest root (sdOT_SF[2] at 221 KiB) and check
    the message names the signature."""
    monkeypatch.setattr(cfgmod, "TCGEN05_V0_ADDR_LIMIT", 221 * _KiB)
    with pytest.raises(ValueError, match=r"version-0 window \(highest: sdOT_SF\[2\] at 226304 B\).*UTCCP copy P/dS DATA into the SF columns"):
        _cfg(FAMILY_MXFP8)
    # the same layout, one byte of window more, is the shipped one
    monkeypatch.setattr(cfgmod, "TCGEN05_V0_ADDR_LIMIT", 226304 + 1)
    assert desc_version(_cfg(FAMILY_MXFP8)) == 0


@pytest.mark.parametrize("family", _FAMILIES)
def test_rejects_a_slab_layout_over_the_rubin_cap(family, monkeypatch):
    """No field flip reaches the cap on a valid cfg (every depth is pinned), so
    shrink the budget below every body's slabs (f16 322 KiB, fp8 258 KiB at the e4m3 dS, mxfp8 288 KiB at the default P-b):
    the message must carry the per-slab tally."""
    monkeypatch.setattr(cfgmod, "SMEM_USABLE_BYTES", 250 * _KiB)
    with pytest.raises(ValueError, match=r"exceed the 327 KiB Rubin oversized per-CTA cap \(sQ .* \| sdS .*\)"):
        _cfg(family)


@pytest.mark.parametrize("family", _FAMILIES)
def test_rejects_scaffolding_over_its_budget(family, monkeypatch):
    monkeypatch.setattr(cfgmod, "SMEM_SCAFFOLD_BYTES", 256)
    with pytest.raises(ValueError, match=r"declared scaffolding \(\d+ B of mbarriers \+ scheduler \+ tmem ptr\) exceeds the 256 B budget"):
        _cfg(family)


# ---------------------------------------------------------------------------
# Pins: the facts the kernel ports and the adapter read off this module
# ---------------------------------------------------------------------------

_MXFP8_SLABS_P_A = ["sQ", "sdO", "sdOdv", "sExcl[K|V](+sdV alias)", "sStats", "sK_SF", "sV_SF", "sP_SF", "sQ_SF", "sdO_SF", "sdOT_SF", "sdS_SF", "sdS"]


@pytest.mark.parametrize(
    "family, params, slab_kib, names",
    [
        (FAMILY_F16, {}, 322, ["sQ", "sdO", "sCombined[sdOdv_s0|K|V](+sdV alias)", "sStats", "sdS"]),
        (FAMILY_FP8, {}, 258, ["sQ", "sdO", "sdOdv", "sExcl[K|V](+sdV alias)", "sStats", "sdS"]),
        (FAMILY_MXFP8, dict(ds_sf_policy=DS_SF_P_A), 273, _MXFP8_SLABS_P_A),
        (FAMILY_MXFP8, dict(ds_sf_policy=DS_SF_P_B), 288, _MXFP8_SLABS_P_A + ["sdS_kv"]),
        (FAMILY_MXFP8, dict(ds_sf_policy=DS_SF_P_C), 286, _MXFP8_SLABS_P_A[:-2] + ["sdS"]),
        (FAMILY_MXFP8, {}, 288, _MXFP8_SLABS_P_A + ["sdS_kv"]),  # the bare record = DS_SF_POLICY_DEFAULT = P-b
    ],
    ids=["f16", "fp8", "mxfp8-P-a", "mxfp8-P-b", "mxfp8-P-c", "mxfp8-default"],
)
def test_smem_tally_and_declaration_order(family, params, slab_kib, names):
    cfg = _cfg(family, **params)
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
    # e4m3 dS (the default): 3 x 16 KiB, ONE 128-col store subtile per 128-B row (both warpgroups' halves in it); every dS
    # figure follows BPE_DS, not BPE
    assert by_name["sdS"].nbytes == 48 * _KiB and by_name["sdS"].offset == 210 * _KiB
    b = buffer_elems(cfg)
    assert (b.P_TMA_ITERS, b.P_D_BLOCK, b.pXferBytes) == (1, 128, 16 * _KiB)
    assert b.P_D_BLOCK % b._SMX_CHUNK == 0
    # the bf16-dS twin: 3 x 32 KiB, two 64-col subtiles per row (one per warpgroup), 306 KiB of slabs
    cfg_bf16 = _cfg(FAMILY_FP8, dtype_ds=DTYPE_BF16)
    b_bf16 = buffer_elems(cfg_bf16)
    assert {s.name: s for s in smem_layout(cfg_bf16)}["sdS"].nbytes == 96 * _KiB and smem_bytes(cfg_bf16) == 306 * _KiB
    assert (b_bf16.P_TMA_ITERS, b_bf16.P_D_BLOCK, b_bf16.pXferBytes) == (2, 64, 32 * _KiB)
    assert dict(desc_roots(cfg_bf16)) == roots, "the dS dtype moves nothing a descriptor reads (sdS is TMA-only and last)"
    # e4m3 dV: 2 store subtiles of 128 d cols; a bf16 dV: 4 of 64
    assert (b.TMA_DV_ITERS, b.DV_D_BLOCK) == (2, 128)
    b2 = buffer_elems(_cfg(FAMILY_FP8, dtype_o=DTYPE_BF16))
    assert (b2.TMA_DV_ITERS, b2.DV_D_BLOCK) == (4, 64)


# --- the MXFP8 SMEM tally at P-a (the SMEM-P-ring predecessor's table minus its sP row) ---------------------------------------
# P-a is optional validation only, but the table is the primary tally and its 23 roots are EVERY policy's, so it stays pinned --
# built with an explicit ds_sf_policy=DS_SF_P_A, never from the bare record.

# (name, KiB @ offset, KiB size) -- the 13 rows of the table, byte offsets as the allocator lays them out.  P-a writes TWO SF
# tensors per tile (sf_ds_dk AND sf_ds_dq -- distinct F8_128x4 atoms of the same tile byte), so the staging is 3 x 2 x 512 B = 3 KiB.
# The P ring is in TMEM (the fp8 body's), so no slab follows the SF slabs before the dS staging.
_P_A_SMEM_TABLE = [
    ("sQ", 0, 48),
    ("sdO", 48, 48),
    ("sdOdv", 96, 48),
    ("sExcl[K|V](+sdV alias)", 144, 64),
    ("sStats", 208, 2),
    ("sK_SF", 210, 1),
    ("sV_SF", 211, 1),
    ("sP_SF", 212, 1),  # 512 B padded to the 1 KiB slab
    ("sQ_SF", 213, 3),
    ("sdO_SF", 216, 3),
    ("sdOT_SF", 219, 3),
    ("sdS_SF", 222, 3),  # 3 stages x 2 atoms x 512 B
    ("sdS", 225, 48),
]
# every tcgen05 descriptor root of the table, absolute bytes
_P_A_DESC_ROOTS = {
    "sQ[0]": 0,
    "sQ[1]": 16384,
    "sQ[2]": 32768,
    "sdO[0]": 49152,
    "sdO[1]": 65536,
    "sdO[2]": 81920,
    "sdO_dv[0]": 98304,
    "sdO_dv[1]": 114688,
    "sdO_dv[2]": 131072,
    "sK": 147456,
    "sV": 180224,
    "sK_SF": 215040,
    "sV_SF": 216064,
    "sP_SF": 217088,
    "sQ_SF[0]": 218112,
    "sQ_SF[1]": 219136,
    "sQ_SF[2]": 220160,
    "sdO_SF[0]": 221184,
    "sdO_SF[1]": 222208,
    "sdO_SF[2]": 223232,
    "sdOT_SF[0]": 224256,
    "sdOT_SF[1]": 225280,
    "sdOT_SF[2]": 226304,
}


def test_mxfp8_p_a_smem_table_offsets_roots_and_desc_version():
    """The P-a table as MEASURED values: 13 slabs at the tabled offsets, 23 descriptor roots with sdOT_SF[2] the highest at
    226304 B (221 KiB) < 262144 -> desc_version 0; with the two staged dS SF atoms per stage the slabs are 279552 B = 273 KiB,
    275 KiB with the scaffold, 52 KiB free.  (The SMEM-P-ring predecessor had 14 slabs / 25 roots / 305 KiB with sP[1] at 243712.)"""
    cfg = _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_A)
    slabs = smem_layout(cfg)
    assert [(s.name, s.offset // _KiB, s.nbytes // _KiB) for s in slabs] == _P_A_SMEM_TABLE
    assert all(s.offset % _KiB == 0 for s in slabs)
    roots = dict(desc_roots(cfg))
    assert roots == _P_A_DESC_ROOTS
    assert len(desc_roots(cfg)) == len(_P_A_DESC_ROOTS) == 23, "no duplicate root labels"
    assert max(roots.values()) == roots["sdOT_SF[2]"] == 226304 == 221 * _KiB < TCGEN05_V0_ADDR_LIMIT
    assert desc_version(cfg) == 0 and cfgmod._needs_desc_v1(cfg) is False
    assert smem_bytes(cfg) == 279552 == 273 * _KiB
    assert kernel_smem_bytes(cfg) == 275 * _KiB == 281600
    assert cfgmod.SMEM_USABLE_BYTES - smem_bytes(cfg) == 52 * _KiB, "52 KiB left vs the 325 KiB usable"
    # every descriptor-fed slab is declared BEFORE sdS (the rule that keeps the roots under the line); sStats, the dS SF
    # staging and sdS carry no roots (they may sit past the 256 KiB line; at this tally they do not)
    names = [s.name for s in slabs]
    rooted = [s.name for s in slabs if s.roots]
    assert all(names.index(r) < names.index("sdS") for r in rooted)
    assert rooted == [n for n in names if n not in ("sStats", "sdS_SF", "sdS")], "every slab but the three TMA-only ones carries a root"
    by_name = {s.name: s for s in slabs}
    assert by_name["sStats"].roots == () and by_name["sdS_SF"].roots == () and by_name["sdS"].roots == ()
    assert by_name["sdS"].offset == 225 * _KiB < TCGEN05_V0_ADDR_LIMIT, "the e4m3 dS ring follows the SF staging directly (no sP slab)"
    # no SMEM P ring: the e4m3 P lives in the TMEM ring (the fp8 body's map); the retired SMEM-ring size reads 0
    b = buffer_elems(cfg)
    assert "sP" not in by_name and b.pRingStageBytes == 0 and cfg.STAGES_TMEM_P == 2
    assert cfg.TILE_N * cfg.BPE == cfg.P_SWZ_BYTES == cfg.Q_SWZ_BYTES == 128
    assert (
        (b.LEADING_BYTE_OFFSET_P, b.STRIDE_BYTE_OFFSET_P, b.SMEM_LAYOUT_P)
        == (b.LEADING_BYTE_OFFSET_QK, b.STRIDE_BYTE_OFFSET_QK, b.SMEM_LAYOUT_Q)
        == (0, 1024, 2)
    )
    # the dS SF staging: TWO 512-B atoms per dS ring stage under P-a (the dK-orientation atom and the dQ-orientation one)
    assert (cfg.DS_SF_ATOMS, cfg.SF_SMEM_dS, b.dSSfStagingBytes) == (2, 512, 1024) and by_name["sdS_SF"].nbytes == 3 * _KiB
    assert b.rcpGatherElems == 0


def test_mxfp8_sf_slab_sizes_and_tx_bytes():
    """The six SF slabs (the forward's formulas, full-size regardless of CTA_MMA) and the expect_tx growth per _full
    mbarrier (SF_SMEM x CTA_MMA)."""
    cfg = _cfg(FAMILY_MXFP8)
    b = buffer_elems(cfg)
    assert (cfg.SF_SMEM_K, cfg.SF_SMEM_V, cfg.SF_SMEM_Q, cfg.SF_SMEM_dO, cfg.SF_SMEM_P, cfg.SF_SMEM_dOT, cfg.SF_SMEM_dS) == (
        1024,
        1024,
        1024,
        1024,
        512,
        1024,
        512,
    )
    assert (cfg.K_SF_TX, cfg.V_SF_TX, cfg.Q_SF_TX, cfg.dO_SF_TX, cfg.dOT_SF_TX) == (2048, 2048, 2048, 2048, 2048)
    # the barrier table's tx column: payload (unchanged from the fp8 body) + SF
    assert (b.qTmaTransactionBytes, b.dOTmaTransactionBytes, b.kTmaTransactionBytes, b.vTmaTransactionBytes) == (32768, 32768, 65536, 65536)
    assert b.qTmaTransactionBytes + cfg.Q_SF_TX == 34816  # mb_q_full
    assert b.dOTmaTransactionBytes + cfg.dO_SF_TX == 34816  # mb_do_full
    assert b.dOTmaTransactionBytes + cfg.dOT_SF_TX == 34816  # mb_dodv_full (the payload is dO_T through the same BT box)
    assert b.kTmaTransactionBytes + cfg.K_SF_TX == b.vTmaTransactionBytes + cfg.V_SF_TX == 67584  # mb_k_full / mb_v_full
    assert b.dVTmaTransactionBytes == 128 * 256 * 2, "a bf16 dV store per CTA"
    # block-scale MMA constants
    assert (cfg.SF_BLOCK, cfg.SF_BLOCKS_PER_STEP, cfg.TILE_K_HW_BMM1, cfg.IDESC_K_DIM) == (32, 2, 64, 1)
    assert (cfg.P_SCALE_LOG2, cfg.P_SF_BYTE, MXFP8_P_SCALE_LOG2) == (8, 119, 8) and cfg.P_SF_BYTE == 127 - 8


def test_sf_helpers_are_the_general_rows_x_k_chunks_form():
    """Both extents count.  ``4 * ceil(TILE_N/128)`` for P coincides with the general form only while TILE_M ==
    TILE_N; a 256-row operand at K=128 takes 8 columns / 1 KiB, not 4 / 512 B."""
    assert sf_tmem_cols(128, 256) == sf_tmem_cols(256, 128) == 8 and sf_smem_bytes(128, 256) == sf_smem_bytes(256, 128) == 1024
    assert sf_tmem_cols(128, 128) == 4 and sf_smem_bytes(128, 128) == 512
    assert sf_tmem_cols(256, 256) == 16 and sf_smem_bytes(256, 256) == 2048
    # partial atoms round UP (a 192-wide K is two K-chunks; 64 rows are one atom row)
    assert sf_tmem_cols(64, 192) == 8 and sf_smem_bytes(64, 192) == 1024
    assert (cfgmod.MX_BLOCK, cfgmod.SF_ATOM_BYTES, cfgmod.SF_TMEM_COLS_PER_ATOM) == (32, 512, 4)


def test_sf_helpers_agree_with_the_forward_config_at_the_d256_tile():
    """``fwd/config_sm107._d256_mxfp8_sf_sizes`` (Q, K-per-stage, P, V-per-stage) and this module's ``sf_smem_bytes`` are two
    spellings of the F8_128x4 atom arithmetic; a fix landing in one and not the other would size the forward's and the
    backward's SF slabs differently for the same operand shape, visible only on a Rubin launch of BOTH.  The forward's
    spelling rounds K only (not the general rows x K-chunks form), so the two agree at whole 128-row tiles -- every tile
    either family builds."""
    tile_m, tile_n, tile_k, tile_o = 128, 128, 256, 256
    cfg = _cfg(FAMILY_MXFP8)
    assert (cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O) == (tile_m, tile_n, tile_k, tile_o)
    assert (
        _d256_mxfp8_sf_sizes(tile_m, tile_n, tile_k, tile_o)
        == (
            sf_smem_bytes(tile_m, tile_k),  # Q: rowwise [128 x 256]
            sf_smem_bytes(tile_n, tile_k),  # K per stage
            sf_smem_bytes(tile_m, tile_n),  # P: [128 x 128], one atom
            sf_smem_bytes(tile_o, tile_n),  # V per stage: columnwise, the 128-padded non-K extent
        )
        == (1024, 1024, 512, 1024)
    )
    # the backward's own slabs for the operands that share those shapes
    assert (cfg.SF_SMEM_Q, cfg.SF_SMEM_K, cfg.SF_SMEM_P, cfg.SF_SMEM_dOT) == (1024, 1024, 512, 1024)
    # and at the 256-row / 192-K shapes the forward's K-rounding and the general form still coincide
    for rows, k in ((256, 256), (128, 192), (256, 128)):
        assert _d256_mxfp8_sf_sizes(rows, rows, k, k)[0] == sf_smem_bytes(rows, k)


@pytest.mark.parametrize(
    "policy, slab_kib, extra",
    [
        (DS_SF_P_B, 288, {"sdS_SF": (222, 2), "sdS": (224, 32), "sdS_kv": (256, 32)}),
        (DS_SF_P_C, 286, {"sdS": (222, 64)}),
    ],
    ids=["P-b", "P-c"],
)
def test_mxfp8_policy_b_and_c_layouts(policy, slab_kib, extra):
    """The P-b / P-c totals: P-b 288 KiB slabs / 290 with scaffold (37 KiB free -- the SMEM-P-ring predecessor's 320 / 322 / 5 less
    the 32 KiB sP ring; no rcp gather slab is declared: the along-kv column scales stay in registers) and P-c 286 /
    288 (39 KiB free; the predecessor's 318 / 320 / 7); the descriptor roots are the P-a table's exactly (the dS rows carry none)."""
    cfg = _cfg(FAMILY_MXFP8, ds_sf_policy=policy)
    by_name = {s.name: s for s in smem_layout(cfg)}
    for name, (off_kib, kib) in extra.items():
        assert (by_name[name].offset, by_name[name].nbytes) == (off_kib * _KiB, kib * _KiB), name
    assert smem_bytes(cfg) == slab_kib * _KiB and kernel_smem_bytes(cfg) == (slab_kib + 2) * _KiB
    assert cfgmod.SMEM_USABLE_BYTES - smem_bytes(cfg) == (37 if policy == DS_SF_P_B else 39) * _KiB
    assert dict(desc_roots(cfg)) == _P_A_DESC_ROOTS and desc_version(cfg) == 0
    b = buffer_elems(cfg)
    if policy == DS_SF_P_B:
        assert (cfg.XFER_STAGES, cfg.DS_PAYLOADS, cfg.DS_SF_ATOMS, b.dSSfStagingBytes, b.rcpGatherElems) == (2, 2, 2, 1024, 0)
        assert by_name["sdS_kv"].roots == () and "sRcp" not in by_name, "no rcp gather slab: the column scales stay in registers"
        assert (by_name["sdS"].nbytes, by_name["sdS_kv"].nbytes) == (2 * 16 * _KiB,) * 2 and (b.P_TMA_ITERS, b.P_D_BLOCK, b.pXferBytes) == (1, 128, 16 * _KiB)
    else:
        assert (cfg.XFER_STAGES, cfg.DS_PAYLOADS, cfg.DS_SF_ATOMS, b.dSSfStagingBytes, b.rcpGatherElems) == (2, 1, 0, 0, 0)
        assert "sdS_SF" not in by_name and (b.P_TMA_ITERS, b.P_D_BLOCK, b.pXferBytes) == (2, 64, 32 * _KiB)
    # the P ring is the TMEM one under every policy (only dS changes dtype); no SMEM slab for it
    assert "sP" not in by_name and b.pRingStageBytes == 0


def test_mxfp8_p_b_atom_column_coupling_names_its_failure(monkeypatch):
    """The P-b body derives its sf_ds_dk / sf_ds_dq byte offsets from `TILE_N / SF_BLOCK == 4 atom columns` and `a warpgroup half = two
    blocks`; every single-field flip of those trips an earlier structural predicate first, so the coupling is provoked through the
    atom-column constant itself -- the message must name the wrong-scale signature.  P-c / P-a never evaluate it."""
    monkeypatch.setattr(cfgmod, "SF_ATOM_COLS", 8)
    with pytest.raises(ValueError, match=r"P-b's sf_ds_dk atom columns are the q-blocks of the 128-q tile.*a neighbour row's scale \(wrong dK, no crash\)"):
        _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_B)
    for policy in (DS_SF_P_A, DS_SF_P_C):
        _cfg(FAMILY_MXFP8, ds_sf_policy=policy)


@pytest.mark.parametrize("policy", _POLICIES, ids=[_POLICY_IDS[p] for p in _POLICIES])
def test_sf_workspace_bytes_is_one_atom_per_tile_pair_per_orientation(policy):
    """``sf_workspace_bytes`` = bytes of ONE dS scale-factor tensor: B x qh_chunk x (S_kv/128) x (S_q/128) x 512 = the payload's bytes
    / 32 (one E8M0 per 32 e4m3), the SAME for sf_ds_dk and sf_ds_dq (their tile axes are transposed, the count is not); 0 under P-c
    (no atoms); the padding contract is ``ds_workspace_bytes``'s."""
    cfg = _cfg(FAMILY_MXFP8, ds_sf_policy=policy)
    one_payload = cfgmod.ds_workspace_bytes(cfg, 2, 4, 1024, 1536) // cfg.DS_PAYLOADS
    got = cfgmod.sf_workspace_bytes(cfg, 2, 4, 1024, 1536)
    if policy == DS_SF_P_C:
        assert got == 0 and cfg.DS_SF_ATOMS == 0
    else:
        assert got == 2 * 4 * (1536 // 128) * (1024 // 128) * cfgmod.SF_ATOM_BYTES == one_payload // 32 and cfg.DS_SF_ATOMS == 2
    with pytest.raises(ValueError, match=r"workspace extents must be padded"):
        cfgmod.sf_workspace_bytes(cfg, 1, 1, 1000, 1024)
    with pytest.raises(ValueError, match=r"workspace extents must be padded"):
        cfgmod.sf_workspace_bytes(cfg, 1, 1, 1024, 1152)


@pytest.mark.parametrize("family", _FAMILIES)
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
    # neither sibling has SF columns
    for tm in (f16, fp8):
        assert all(getattr(tm, f"SF_{n}_{k}") == 0 for n in ("K", "V", "Q", "dO", "P", "dOT") for k in ("OFF", "COLS"))


def test_mxfp8_tmem_map_is_the_fp8_ring_with_the_sf_alias_bands():
    """S 128 | dP 128 | dV 256 | P ring 2 x 32 = 576, RSVD 0 -- the fp8 body's map -- and the SLOT-RELATIVE alias offsets: BMM1 band
    K 0 | V 8 | Q 16 | dO 24 (32 = one slot exactly), BMM2 band P 0 | dOT 4 (12).  Every SF atom lives in the P slot the softmax is not
    writing; the kernel adds P_OFF + sf_slot * P_COLS."""
    cfg = _cfg(FAMILY_MXFP8)
    tm = tmem_layout(cfg)
    fp8 = tmem_layout(_cfg(FAMILY_FP8))
    assert (tm.S_OFF, tm.S_COLS, tm.dP_OFF, tm.dP_COLS, tm.dV_OFF, tm.dV_COLS) == (0, 128, 128, 128, 256, 256)
    assert (
        (tm.P_OFF, tm.P_COLS, tm.RSVD_OFF, tm.RSVD_COLS, cfg.STAGES_TMEM_P)
        == (512, 32, TMEM_TOTAL_COLS, 0, 2)
        == (fp8.P_OFF, fp8.P_COLS, fp8.RSVD_OFF, fp8.RSVD_COLS, 2)
    )
    assert tm.P_OFF + cfg.STAGES_TMEM_P * tm.P_COLS == TMEM_TOTAL_COLS and not hasattr(cfg, "STAGES_SMEM_P")
    assert [(getattr(tm, f"SF_{n}_OFF"), getattr(tm, f"SF_{n}_COLS")) for n in ("K", "V", "Q", "dO", "P", "dOT")] == [
        (0, 8),
        (8, 8),
        (16, 8),
        (24, 8),
        (0, 4),
        (4, 8),
    ]
    assert (cfg.SF_TMEM_COLS_K, cfg.SF_TMEM_COLS_V, cfg.SF_TMEM_COLS_Q, cfg.SF_TMEM_COLS_dO, cfg.SF_TMEM_COLS_P, cfg.SF_TMEM_COLS_dOT) == (8, 8, 8, 8, 4, 8)
    assert (tm.SF_BMM1_COLS, tm.SF_BMM2_COLS) == (32, 12) and tm.SF_BMM1_COLS == tm.P_COLS and tm.SF_BMM2_COLS <= tm.P_COLS
    # the two bands share the slot in TIME, never in columns with the live P slot: the fp8 map has no SF fields at all
    assert (fp8.SF_BMM1_COLS, fp8.SF_BMM2_COLS) == (0, 0)
    assert tm.S_COLS + tm.dP_COLS + tm.dV_COLS + cfg.STAGES_TMEM_P * tm.P_COLS == TMEM_TOTAL_COLS
    # the map does not depend on the dS policy or the gradient dtype
    for kw in (dict(ds_sf_policy=DS_SF_P_A), dict(ds_sf_policy=DS_SF_P_B), dict(ds_sf_policy=DS_SF_P_C), dict(dtype_o=DTYPE_FP16)):
        assert tmem_layout(_cfg(FAMILY_MXFP8, **kw)) == tm


@pytest.mark.parametrize("family", _FAMILIES)
def test_scheduler_arrivers_and_warp_layout(family):
    cfg = _cfg(family)
    assert cfg.READ_TILE_ARRIVERS_TOT == read_tile_arrivers_tot(cfg) == 21 == 2 * (8 + 2) + 1
    assert (cfg.TOTAL_WARPS, cfg.THREADS_PER_CTA) == (12, 384)
    assert (cfg.SOFTMAX_WG0_BASE, cfg.SOFTMAX_WG1_BASE, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == (0, 4, 8, 9, 10, 11)
    # Register split: the per-warp sum must not exceed the 12-warp ENTRY pool 12 x 168 = 2016 -- the pool setmaxnreg
    # redistributes is the LAUNCH allocation, not the 2048-register file (8 x 232 + 4 x 48 = 2048 HUNG the first Rubin
    # launch, 2026-09-23: the last softmax INCREASE parks forever).  f16 224 / 56 (the 40-register MMA warp spilled one
    # slot on the bf16 sm_107a builds, so 8 registers move from the softmax warps to the service warps), fp8 the
    # pre-port 232 / 40, and the MXFP8 body starts from the fp8 split unchanged (the f16 split is its
    # named fallback if the MMA warp's six SF descriptors spill).  All balance the pool exactly.
    softmax, service = (224, 56) if family == FAMILY_F16 else (232, 40)
    assert (cfg.SOFTMAX_REGS, cfg.MMA_REGS, cfg.OTHER_REGS) == (softmax, service, service)
    assert 8 * cfg.SOFTMAX_REGS + 4 * cfg.MMA_REGS == 8 * softmax + 4 * service == cfgmod.reg_entry_pool(cfg.TOTAL_WARPS) == 2016 < cfgmod.REG_BUDGET_PER_CTA
    assert (cfg.SOFTMAX_WG_LANES, cfg.SOFTMAX_LANES, cfg.SOFT_X_CTA_MMA, cfg.MMA_COMMIT_ARRIVES) == (128, 256, 512, 1)


def test_mbar_inventory_differs_between_the_bodies_exactly_where_the_pipelines_do():
    f16, fp8 = mbar_stage_counts(_cfg(FAMILY_F16)), mbar_stage_counts(_cfg(FAMILY_FP8))
    mx = mbar_stage_counts(_cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_A))
    assert set(f16) - set(fp8) == {"mb_k_utccp_done"}  # the K-split alias seam
    assert set(fp8) - set(f16) == {"mb_s_acc_empty"}  # the lookahead's S WAR handshake
    assert (fp8["mb_q_full"], fp8["mb_p_ready"], fp8["mb_ds_smem_full"]) == (3, 2, 3)
    assert (f16["mb_q_full"], f16["mb_p_ready"], f16["mb_ds_smem_full"], f16["mb_dodv_full"]) == (2, 1, 1, 2)
    # ONE new mbarrier on the MXFP8 body -- mb_p_sf_consumed (1 stage, the SF-alias release), right after mb_p_ready; the SF loads
    # ride the _full bars, the TMEM P ring keeps mb_p_ready at 2 stages (STAGES_TMEM_P, no p_empty); 25 rings + the scheduler's two
    assert set(mx) - set(fp8) == {"mb_p_sf_consumed"} and mx["mb_p_sf_consumed"] == 1 and len(mx) == 25
    assert {k: v for k, v in mx.items() if k != "mb_p_sf_consumed"} == fp8
    assert list(mx).index("mb_p_sf_consumed") == list(mx).index("mb_p_ready") + 1
    assert scaffold_bytes_declared(_cfg(FAMILY_FP8)) == 624
    assert scaffold_bytes_declared(_cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_A)) == 624 + 16  # one 1-slot (16 B) array more
    # P-b / P-c (and so the bare record, DS_SF_POLICY_DEFAULT = P-b): the same 25 names; only the dS ring depth (and so its
    # two bars) changes
    assert mbar_stage_counts(_cfg(FAMILY_MXFP8)) == mbar_stage_counts(_cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_B))
    for policy in (DS_SF_P_B, DS_SF_P_C):
        two_deep = mbar_stage_counts(_cfg(FAMILY_MXFP8, ds_sf_policy=policy))
        assert set(two_deep) == set(fp8) | {"mb_p_sf_consumed"}
        assert {k for k in fp8 if two_deep[k] != fp8[k]} == {"mb_ds_smem_full", "mb_ds_smem_empty"} and two_deep["mb_ds_smem_full"] == 2
        assert scaffold_bytes_declared(_cfg(FAMILY_MXFP8, ds_sf_policy=policy)) == 624 + 16 - 2 * 16  # two 3-slot (32 B) arrays -> 2-slot (16 B)


# ---------------------------------------------------------------------------
# Regression pin: the f16 and fp8 families at develop dd3235c3 (PR #1212), literal.  Taken from the code BEFORE the MXFP8
# family landed.  A third
# family must not move a byte of either sibling's SMEM table, TMEM map, barrier inventory or launch bytes.
# ---------------------------------------------------------------------------

_TMEM_FIELDS_DD3235C3 = ("TOTAL_COLS", "S_OFF", "S_COLS", "dP_OFF", "dP_COLS", "dV_OFF", "dV_COLS", "P_OFF", "P_COLS", "RSVD_OFF", "RSVD_COLS")

_F16_MBARS_DD3235C3 = {
    "mb_q_full": 2,
    "mb_q_empty": 2,
    "mb_do_full": 2,
    "mb_do_empty": 2,
    "mb_dodv_full": 2,
    "mb_dodv_empty": 2,
    "mb_k_full": 1,
    "mb_k_empty": 1,
    "mb_v_full": 1,
    "mb_v_empty": 1,
    "mb_k_utccp_done": 1,
    "mb_s_acc_full": 1,
    "mb_dp_full": 1,
    "mb_dp_empty": 1,
    "mb_p_ready": 1,
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
_FP8_MBARS_DD3235C3 = {
    "mb_q_full": 3,
    "mb_q_empty": 3,
    "mb_do_full": 3,
    "mb_do_empty": 3,
    "mb_dodv_full": 3,
    "mb_dodv_empty": 3,
    "mb_k_full": 1,
    "mb_k_empty": 1,
    "mb_v_full": 1,
    "mb_v_empty": 1,
    "mb_s_acc_full": 1,
    "mb_s_acc_empty": 1,
    "mb_dp_full": 1,
    "mb_dp_empty": 1,
    "mb_p_ready": 2,
    "mb_stats_full": 2,
    "mb_stats_empty": 2,
    "mb_ds_smem_full": 3,
    "mb_ds_smem_empty": 3,
    "mb_dv_ready": 1,
    "mb_dv_acc_empty": 1,
    "mb_dv_stg_full": 1,
    "mb_dv_stg_empty": 1,
    "mb_tmem_dealloc": 1,
}

_F16_SLABS_DD3235C3 = (
    ("sQ", 0, 65536, (("sQ[0]", 0), ("sQ[1]", 32768))),
    ("sdO", 65536, 65536, (("sdO[0]", 65536), ("sdO[1]", 98304))),
    (
        "sCombined[sdOdv_s0|K|V](+sdV alias)",
        131072,
        163840,
        (("sdO_dv[0]", 131072), ("sK", 163840), ("sK_back(UTCCP src)", 196608), ("sV", 229376), ("sdO_dv[1](=sK_back alias)", 196608)),
    ),
    ("sStats", 294912, 2048, ()),
    ("sdS", 296960, 32768, ()),
)
_FP8_SLABS_DD3235C3 = (
    ("sQ", 0, 49152, (("sQ[0]", 0), ("sQ[1]", 16384), ("sQ[2]", 32768))),
    ("sdO", 49152, 49152, (("sdO[0]", 49152), ("sdO[1]", 65536), ("sdO[2]", 81920))),
    ("sdOdv", 98304, 49152, (("sdO_dv[0]", 98304), ("sdO_dv[1]", 114688), ("sdO_dv[2]", 131072))),
    ("sExcl[K|V](+sdV alias)", 147456, 65536, (("sK", 147456), ("sV", 180224))),
    ("sStats", 212992, 2048, ()),
    ("sdS", 215040, 49152, ()),
)
_FP8_SLABS_BF16_DS_DD3235C3 = _FP8_SLABS_DD3235C3[:-1] + (("sdS", 215040, 98304, ()),)

_F16_TMEM_DD3235C3 = dict(
    TOTAL_COLS=576, S_OFF=0, S_COLS=128, dP_OFF=128, dP_COLS=128, dV_OFF=256, dV_COLS=256, P_OFF=32, P_COLS=64, RSVD_OFF=512, RSVD_COLS=64
)
_FP8_TMEM_DD3235C3 = dict(
    TOTAL_COLS=576, S_OFF=0, S_COLS=128, dP_OFF=128, dP_COLS=128, dV_OFF=256, dV_COLS=256, P_OFF=512, P_COLS=32, RSVD_OFF=576, RSVD_COLS=0
)


@pytest.mark.parametrize(
    "family, params, slabs, tmem, kernel_bytes, mbars, scaffold",
    [
        (FAMILY_F16, dict(dtype_qkv=DTYPE_BF16), _F16_SLABS_DD3235C3, _F16_TMEM_DD3235C3, 331776, _F16_MBARS_DD3235C3, 496),
        (FAMILY_F16, dict(dtype_qkv=DTYPE_FP16), _F16_SLABS_DD3235C3, _F16_TMEM_DD3235C3, 331776, _F16_MBARS_DD3235C3, 496),
        (FAMILY_F16, dict(dtype_qkv=DTYPE_BF16, dtype_o=DTYPE_FP16), _F16_SLABS_DD3235C3, _F16_TMEM_DD3235C3, 331776, _F16_MBARS_DD3235C3, 496),
        (FAMILY_FP8, dict(), _FP8_SLABS_DD3235C3, _FP8_TMEM_DD3235C3, 266240, _FP8_MBARS_DD3235C3, 624),
        (FAMILY_FP8, dict(dtype_o=DTYPE_BF16), _FP8_SLABS_DD3235C3, _FP8_TMEM_DD3235C3, 266240, _FP8_MBARS_DD3235C3, 624),
        (FAMILY_FP8, dict(dtype_ds=DTYPE_BF16), _FP8_SLABS_BF16_DS_DD3235C3, _FP8_TMEM_DD3235C3, 315392, _FP8_MBARS_DD3235C3, 624),
        (FAMILY_FP8, dict(dtype_o=DTYPE_FP16, dtype_ds=DTYPE_BF16), _FP8_SLABS_BF16_DS_DD3235C3, _FP8_TMEM_DD3235C3, 315392, _FP8_MBARS_DD3235C3, 624),
    ],
    ids=["f16-bf16", "f16-fp16", "f16-bf16-o_fp16", "fp8", "fp8-o_bf16", "fp8-ds_bf16", "fp8-o_fp16-ds_bf16"],
)
def test_f16_and_fp8_layouts_are_byte_identical_to_dd3235c3(family, params, slabs, tmem, kernel_bytes, mbars, scaffold):
    cfg = _cfg(family, **params)
    assert tuple((s.name, s.offset, s.nbytes, s.roots) for s in smem_layout(cfg)) == slabs
    assert kernel_smem_bytes(cfg) == kernel_bytes and smem_bytes(cfg) == kernel_bytes - SMEM_SCAFFOLD_BYTES
    assert desc_version(cfg) == 0
    tm = tmem_layout(cfg)
    assert {f: getattr(tm, f) for f in _TMEM_FIELDS_DD3235C3} == tmem
    assert mbar_stage_counts(cfg) == mbars
    assert scaffold_bytes_declared(cfg) == scaffold
    assert read_tile_arrivers_tot(cfg) == cfg.READ_TILE_ARRIVERS_TOT == 21
    # and the new MXFP8 fields are all zero on both siblings (the SF-free claim the validator also pins)
    assert cfg.IS_MXFP8 == 0 and cfg.DS_SF_POLICY == DS_SF_NONE and cfg.DS_PAYLOADS == 1
    assert {n: getattr(cfg, n) for n in _MXFP8_ONLY_CFG_FIELDS} == _MXFP8_NEUTRAL
    b = buffer_elems(cfg)
    assert (b.pRingStageBytes, b.dSSfStagingBytes, b.rcpGatherElems) == (0, 0, 0)


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
    # e4m3 dS on the fp8 chain (the default): 1 B per cell; the bf16 twin: 2
    assert ds_workspace_bytes(cfg, 1, 128, 8192, 8192) == 128 * 8192 * 8192
    assert ds_workspace_bytes(_cfg(FAMILY_FP8, dtype_ds=DTYPE_BF16), 1, 128, 8192, 8192) == 128 * 8192 * 8192 * 2
    assert ds_workspace_bytes(_cfg(FAMILY_F16), 1, 128, 8192, 8192) == 128 * 8192 * 8192 * 2
    # MXFP8: the PAYLOAD rings one launch writes -- P-a one e4m3,
    # P-b TWO e4m3 (dS along q for dK + along kv for dQ: DS_PAYLOADS), P-c one bf16; the SF tensors are separate workspace rows
    assert ds_workspace_bytes(_cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_A), 1, 128, 8192, 8192) == 128 * 8192 * 8192
    assert ds_workspace_bytes(_cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_B), 1, 128, 8192, 8192) == 2 * 128 * 8192 * 8192
    assert ds_workspace_bytes(_cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_C), 1, 128, 8192, 8192) == 128 * 8192 * 8192 * 2
    assert ds_workspace_bytes(_cfg(FAMILY_MXFP8), 1, 128, 8192, 8192) == ds_workspace_bytes(
        _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_POLICY_DEFAULT), 1, 128, 8192, 8192
    )
    assert _cfg(FAMILY_MXFP8, ds_sf_policy=DS_SF_P_B).DS_PAYLOADS == 2 and _cfg(FAMILY_FP8).DS_PAYLOADS == 1
    with pytest.raises(ValueError, match=r"padded to q 128 / kv 256 rows"):
        ds_workspace_bytes(cfg, 1, 1, 8000, 8192)
    assert launch_grid(cfg, 2, 8, 8192) == ((32 * 2, 8, 2), (2, 1, 1))
    lpt = _cfg(FAMILY_FP8, sched_policy=SCHED_LPT)
    assert launch_grid(lpt, 2, 8, 8192) == ((32 * 8 * 2 * 2, 1, 1), (2, 1, 1))
    assert (cfgmod.q_pad_rows(_cfg(FAMILY_MXFP8)), cfgmod.kv_pad_rows(_cfg(FAMILY_MXFP8)), cfgmod.q_write_tiles(_cfg(FAMILY_MXFP8))) == (128, 256, 2)
