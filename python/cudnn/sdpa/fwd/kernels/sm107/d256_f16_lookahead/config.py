# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent SM107 half configuration and local immutable record."""

from dataclasses import dataclass
from cudnn.sdpa.fwd.config_sm107 import (
    SCHED_NATURAL,
    SMEM_USABLE_BYTES,
    TCGEN05_V0_ADDR_LIMIT,
    TemplateParams,
    TmaIters,
    Tuple,
    _D256_TOTAL_WARPS,
    _DTYPE_BF16,
    _DTYPE_E4M3,
    _DTYPE_E5M2,
    _DTYPE_FP16,
    _band_fields,
    _check,
    _tma_iters,
    _validate_dtype_k_step,
    _validate_masks,
    _validate_params,
    _validate_regs,
    _validate_smem,
    _validate_swizzles,
    _validate_warp_layout,
    bpe,
    d256_mxfp8_last_sf_tile_start,
    d256_mxfp8_sf_smem_bytes,
    o_swz_bytes,
    q_swz_bytes,
    rescale_threshold,
    resolve_dtype_o,
    tile_k_hw,
    v_swz_bytes,
)


@dataclass(frozen=True)
class CfgD256:
    PADDED_TAIL_FORM: int = 0
    DEAD_SUBTILE_QUIET: int = 0
    BIAS_MODE: int = 0
    BIAS_RESIDENT_TILES: int = 0
    BIAS_BLOCK_ROWS: int = 0
    BIAS_SWZ_BYTES: int = 128
    BIAS_BOXES_PER_TILE: int = 0
    BIAS_EMPTY_ARRIVERS: int = 0
    KV_MASK_FORM: int = 0
    N_CHUNK: int = 0
    STAGES_BIAS: int = 0
    BIAS_MAX_FORM: int = 0
    S_SLOTS: int = 1
    BIAS_MCAST_CTAS: int = 1
    Q_IN_TMEM: int = 0
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128
    TILE_O: int = 128
    DTYPE_QKV: int = 3
    DTYPE_O: int = 3
    BPE: int = 2
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    SPLIT_PIPELINE: int = 0
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    TILES_Q: int = 2
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 4
    SOFTMAX_WARPGROUPS: int = 2
    CORRECTION_WARPS: int = 4
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40
    RESCALE_THRESHOLD: float = 8.0
    MASK_FLAGS: int = 0
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    PACK_GQA: int = 0
    PACK_G: int = 0
    QH_PER_KH: int = 1
    SPLIT_KV: int = 1
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = 0
    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 64
    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 512
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 8
    MMA_WARP_ID: int = 12
    TMALDG_WARP_ID: int = 13
    TMASTG_WARP_ID: int = 14
    SCHED_WARP_ID: int = 15
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256
    READ_TILE_ARRIVERS: int = 15
    EPILOGUE_GATE: int = 0
    GATE_BPE: int = 2
    DTYPE_KV: int = -1


def _d256_read_tile_arrivers(cta_mma: int) -> int:
    return ((1 * 4) + 4 + 2) * (cta_mma * 1) + (cta_mma // cta_mma)


def _make_cfg_d256_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    """The cc 10.7 d256 config (per-tensor FP8 or MXFP8): tile geometry, ring depths, warp layout and the SMEM / descriptor-window
    checks for the one-CTA and the 2-CTA-pair builds (``params.cta_mma``); raises with the failing tally on an impossible layout."""
    split_wired = not mxfp8 and params.dtype_qkv in (_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16)
    _validate_params(flavor, params, split_wired=split_wired)
    if (params.split_kv or 1) > 1 and params.thd_varlen:
        raise ValueError(f"{flavor}: native split-KV is dense-only")
    if (params.split_kv or 1) > 1 and params.pack_gqa and params.dtype_qkv in (_DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: native half prefill split is unpacked")
    if params.pack_gqa and not mxfp8:
        if params.thd_varlen:
            raise ValueError(f"{flavor}: per-tensor FP8 PackGQA is dense-only")
        if 128 % params.qh_per_kh:
            raise ValueError(f"{flavor}: PackGQA requires a whole head group dividing TILE_M=128; got {params.qh_per_kh}")
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    # The explicit one-Q D128 topology reuses this family's parity pipeline.
    single_q_d128 = params.softmax_wgs == 1
    if single_q_d128 and not (
        not mxfp8
        and params.dtype_qkv in (_DTYPE_BF16, _DTYPE_FP16)
        and params.d_flavor == 128
        and params.cta_mma == 1
        and params.thd_varlen
        and not params.pack_gqa
        and params.split_kv == 1
        and params.tile_n == 128
        and params.s_slots == 2
        and not params.q_in_tmem
        and not params.epilogue_gate
        and not params.bias_mode
        and not params.kv_mask_form
        and params.dtype_kv == -1
    ):
        raise ValueError("single-Q D128 requires half THD CTA1 N128, unsplit/unpacked and two S slots")
    tile_k = tile_o = 128 if single_q_d128 else 256
    tile_n = 128
    # STAGES_KV is a TUNING KNOB again (2..4), and the note that said otherwise
    # was WRONG about why.
    #
    # It used to be pinned to 2, blaming a body that "conflates the KV ring
    # index with the 2-slot S_acc parity". That diagnosis does not survive
    # reading the body: every parity site derives parity from the ABSOLUTE
    # `kv_loop & 1`, and K/V ride separate PipelineStates advanced in lockstep
    # with the TMA warp, so the body is depth-agnostic.
    #
    # The real cause was the >256 KiB tcgen05 descriptor wrap. Buffers are laid
    # out sQO | sK[stages] | sV[stages]; at depth 4 the last two V stages start
    # at 256 KiB and 288 KiB, and a VERSION-0 descriptor truncates
    # `start_address` to 14 bits, so those stages alias the bottom of SMEM. The
    # answer goes wrong from the KV iteration that first touches a wrapped
    # stage -- which is why it looked like a ring bug and why it only shows up
    # at S_kv > 256.
    #
    # Measured on Rubin, d256 f16, cos vs an fp32 reference, n_kv = 2/3/4/8:
    #   depth 4, desc v0:  1.0000 / 0.6806 / 0.4980 / 0.4640   <- the old data
    #   depth 4, desc v1:  1.0000 / 1.0000 / 1.0000 / 1.0000   dense AND causal
    #   depth 3, desc v0:  1.0000 / 1.0000 / 1.0000 / 1.0000   (stays under)
    # `prefill_d256_f16.DESC_VERSION` is now DERIVED from the layout, so any
    # depth that fits SMEM is correct by construction. Do not re-literal it.
    # (The per-tensor FP8 sibling `prefill_d256_fp8.py` carried a literal 0
    # until the epilogue-gate PR ported the same `_needs_desc_v1` derivation;
    # at e4m3-O depth 4 its last V stage starts at exactly 262144, so the
    # literal was one depth away from the same wrap.)
    # `is not None`, NOT a truth test: a truthiness check maps an explicit
    # stages_kv=0 onto the default 2, which is a knob SUBSTITUTION -- the one
    # thing the engine contract forbids (honored or ineligible). Out-of-domain
    # values must reach the 2..4 check below and raise there.
    stages_kv = params.stages_kv if getattr(params, "stages_kv", None) is not None else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    arrivers = _d256_read_tile_arrivers(cta_mma)
    # The four single-warp roles (MMA / TMA-LDG / TMA-STG / scheduler; the HW
    # equality constraint keeps them one number) run at 40 on every d256
    # module.  MEASURED DEAD END (2026-09-15, gated MXFP8): its sm_107a SASS
    # carries 25 STL/LDL against the ungated 6 (the SR_CTAID.{x,y,z} tile
    # coordinates, spilled at kernel entry and refilled in the TMA-LDG /
    # TMA-STG / scheduler regions), and raising these four roles to 48 left the
    # count at 25.  The refill sites carry MOV.SPILL UR -> R (5 vs 0 ungated):
    # the pressure starts in the UNIFORM register file, which setmaxnreg does
    # not govern, so this vector budget is not the lever.  The split itself IS
    # applied (USETMAXREG.TRY_ALLOC / DEALLOC present in the sm_107a SASS).  Do
    # not retry the bump; see the REGISTER NOTE in
    # sm107/prefill_d256_mxfp8.py's TMA-LDG warp for the levers not yet tried.
    other_regs = 40

    cfg = CfgD256(
        DTYPE_KV=params.dtype_kv,
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        # The body is a SINGLE-warpgroup pipeline. Two warpgroups is the
        # Blackwell d256 split-P design and does not exist here.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=4,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=96,
        MMA_REGS=other_regs,
        TMALDG_REGS=other_regs,
        TMASTG_REGS=other_regs,
        SCHEDULER_REGS=other_regs,
        OTHER_REGS=other_regs,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        PAGED_KV=int(bool(params.paged_kv)),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=_D256_TOTAL_WARPS,
        THREADS_PER_CTA=_D256_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused at SOFTMAX_WARPGROUPS=1
        CORR_WARP_BASE=4,
        MMA_WARP_ID=8,
        TMALDG_WARP_ID=9,
        TMASTG_WARP_ID=10,
        SCHED_WARP_ID=11,
        READ_TILE_ARRIVERS=arrivers,
        # Fused epilogue gate (validated per flavor in _validate_params). The gate
        # is 2 B/elem on BOTH the half kernel (Q dtype) and the FP8 kernel (bf16).
        EPILOGUE_GATE=int(params.epilogue_gate),
        GATE_BPE=2,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D256_TOTAL_WARPS, expect_arrivers=arrivers)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    # The STAGES_KV upper bound differs per dtype family, and the message says
    # which: the f16 and per-tensor FP8 bodies DERIVE their tcgen05 descriptor
    # version from the layout (`_needs_desc_v1`), so for them every depth that
    # fits SMEM is correct; the MXFP8 body PINS DESC_VERSION=0 (version 1
    # regressed 21 green d128/d256 MXFP8 tests, rules/mma-tma-matrix.md S6), so
    # for it the bound is the 256 KiB version-0 window on the LAST scale-factor
    # slab -- checked below with its own diagnostic.
    _depth_bound = (
        f"the {SMEM_USABLE_BYTES // 1024} KiB Rubin carveout, and the kernel derives its tcgen05 descriptor version from the resulting "
        f"layout so every depth in range is correct"
        if not mxfp8
        else f"the {SMEM_USABLE_BYTES // 1024} KiB Rubin carveout AND the {TCGEN05_V0_ADDR_LIMIT // 1024} KiB version-0 tcgen05 descriptor "
        f"window on the last scale-factor slab (this kernel pins DESC_VERSION=0; see d256_mxfp8_last_sf_tile_start)"
    )
    _check(
        [
            (cfg.TILES_Q == 1, f"{flavor}: the d256 pipeline mandates TILES_Q == 1"),
            (cfg.SOFTMAX_WARPGROUPS == 1, f"{flavor}: the d256 pipeline mandates SOFTMAX_WARPGROUPS == 1"),
            (
                2 <= cfg.STAGES_KV <= 4,
                f"{flavor}: STAGES_KV must be in 2..4 (got {cfg.STAGES_KV}); the upper bound is {_depth_bound}",
            ),
        ]
    )
    sf_bytes = 0
    if mxfp8:
        # MXFP8: the four scale-factor slabs are declared after sQO | sK | sV
        # and read through VERSION-0 UTCCP descriptors, so their START must sit
        # under the 14-bit window.  A bf16 O at STAGES_KV=3 (or an e4m3 O at 4)
        # puts sQ_SF at or past 262144: the descriptor wraps to offset 0 and the
        # UTCCP copies Q DATA bytes into the SF TMEM columns -- LSE = +inf and
        # O = NaN on 100 % of cells at every shape, the SF values provably
        # irrelevant (the d512 MXFP8 failure).  Latent until 2026-09-15 (depth 3
        # fit the SMEM check); raised HERE, before the kernel's own backstop.
        # The epilogue gate tile is declared AFTER the SF slabs (PR-B D7) and is
        # not in this sum.
        last_sf = d256_mxfp8_last_sf_tile_start(cfg)
        _check(
            [
                # The gated MXFP8 module runs at both widths: the dense unsplit legs take the 2-CTA pair since
                # d256_mxfp8_pair_leg (K/V rings halved, SF slabs full-size, one gate tile per CTA -- the tallies
                # below bound that layout too), validated by the MXFP8 gate tests at the pair's default width.
                (
                    last_sf < TCGEN05_V0_ADDR_LIMIT,
                    f"{flavor}: the last scale-factor slab starts at {last_sf} B, at or past the {TCGEN05_V0_ADDR_LIMIT} B version-0 tcgen05 "
                    f"descriptor window (STAGES_KV={cfg.STAGES_KV}, BPE_O={cfg.BPE_O}, CTA_MMA={cfg.CTA_MMA}); a version-0 SF descriptor there "
                    f"wraps to offset 0 and the UTCCP copies Q data as scale factors (LSE = +inf, O = NaN on every cell).  Reduce STAGES_KV or "
                    f"write an e4m3 O -- the kernel pins DESC_VERSION=0 (version 1 is not a transparent widening, rules/mma-tma-matrix.md S6)",
                ),
            ]
        )
        sf_bytes = d256_mxfp8_sf_smem_bytes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.STAGES_KV)
    # d256 always Q-union-O aliases.  With the epilogue gate on, the 64 KiB
    # staging tile is added here: depth 2 fits at both CTA_MMA widths (258 KiB
    # f16 cga2 / 226 KiB fp8 e4m3-O / 258 KiB fp8 bf16-O; MXFP8 adds its 6 KiB of
    # scale-factor slabs: 232 KiB e4m3-O / 264 KiB bf16-O), depth 3 does not
    # (322 KiB) -- the raise names every term so the reader sees why a depth
    # that fits ungated no longer does.
    _validate_smem(cfg, flavor, qo_alias=True, sf_bytes=sf_bytes)
    return cfg, _tma_iters(cfg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    flavor = "sm107 d128" if params.softmax_wgs == 1 and params.paged_kv else "sm107 d256"
    return _make_cfg_d256_family(params, flavor=flavor, mxfp8=False)
