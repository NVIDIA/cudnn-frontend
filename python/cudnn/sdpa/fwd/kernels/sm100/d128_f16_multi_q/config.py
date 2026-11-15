# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent half multi-Q configuration; D32/D64/D128 are local variants."""

from dataclasses import dataclass


from typing import Optional, Tuple


from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    DTYPE_O_MXFP8,
    DTYPE_O_NVFP4,
    O_BLOCK_SCALE_BY_DTYPE,
    MASK_PADDED,
)


from cudnn.sdpa.fwd.config_common import (
    TemplateParams,
    _validate_params,
    _mask_flags_from,
    bpe,
    o_pack_div,
    tile_k_hw,
    q_swz_bytes,
    v_swz_bytes,
    o_swz_bytes,
    rescale_threshold,
    _pack_g,
    TmaIters,
    _tma_iters,
)


from cudnn.sdpa.fwd.config_d128_family import (
    BIAS_MAX_FORMS,
    BIAS_MAX_HW_UPPER_BOUND,
    BIAS_MCAST_CTAS,
    BIAS_MODES,
    BIAS_MODE_NONE,
    BIAS_MODE_RESIDENT,
    BIAS_MODE_STREAMED_MULTICAST,
    BIAS_RING_STAGES,
    BIAS_STREAMED_MODES,
    BIAS_SWZ_BYTES,
    D128FamilyArch,
    D128_CORRECTION_WARPS,
    D128_OTHER_WARPS,
    D128_REG_SPLIT,
    D128_SOFTMAX_WG_WARPS,
    DEAD_SUBTILE_QUIET_FORMS,
    KV_MASK_ADDITIVE_16,
    KV_MASK_ADDITIVE_F32,
    KV_MASK_FORMS,
    PADDED_TAIL_FORMS,
    SMALL_D_Q_IN_TMEM,
    SMALL_D_SOFTMAX_WGS,
    SMALL_D_STAGES_KV,
    bias_block_rows,
    bias_resident_block_bytes,
    bias_resident_tiles,
    bias_ring_stages,
    d128_read_tile_arrivers,
    d128_total_warps,
    validate_bias,
    validate_head_dim_tile,
    validate_q_in_tmem,
    validate_regs,
    validate_small_tile_masks,
    validate_smem,
    validate_softmax_wgs,
    validate_swizzles,
    validate_tile_n,
    validate_warp_layout,
)


@dataclass(frozen=True)
class CfgD128:
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
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    QO_ALIAS: int = 0
    SPLIT_PIPELINE: int = 0
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    TILES_Q: int = 2
    Q_TILES_LIVE: int = 2
    DUAL_KV_STREAM: int = 0
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2
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
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0
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
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    SPLIT_KV: int = 1
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    PACK_G: int = 1
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    DTYPE_KV: int = -1


_SM100_MAX_DYN_SMEM = 227 * 1024


def _d128_smem_bytes(cfg) -> int:
    """Data-buffer SMEM for the d128 pipeline (barriers/TMEM ptr are noise).

    Q and O are TILES_Q slabs each; under QO_ALIAS they share one slab sized to
    the larger.  K/V are STAGES_KV buffers each, and their PER-CTA size is
    divided by CTA_MMA because the cga2 collective MMA lets a CTA pair hold half
    of every tile.  That divisor is exactly what cga1 gives up, which is why
    cga1 needs the Q/O alias to break even:

        cga2, no alias : 64(Q) + 64(O) + 32(K) + 32(V) = 192 KiB
        cga1, no alias : 64(Q) + 64(O) + 64(K) + 64(V) = 256 KiB  (over cap)
        cga1, alias    : 64(Q u O)     + 64(K) + 64(V) = 192 KiB
    """
    q_slab = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    o_slab = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O // cfg.O_PACK_DIV
    # Two KV streams over one Q tile: one Q slab and one O slab (the fp8 kernel
    # never aliases them), while P stays one tile per stream.
    q_tiles = getattr(cfg, "Q_TILES_LIVE", cfg.TILES_Q) if getattr(cfg, "DUAL_KV_STREAM", 0) else cfg.TILES_Q
    qo = q_tiles * (max(q_slab, o_slab) if cfg.QO_ALIAS else q_slab + o_slab)
    k = cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K * cfg.BPE // cfg.CTA_MMA)
    v = cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N * cfg.BPE_V // cfg.CTA_MMA)
    # The per-tensor FP8 kernel stages P (TILE_M x TILE_N fp8 per Q sub-tile) in SMEM so the next
    # step's QK^T can overlap the softmax; counted for every quantized row (pessimistic for MXFP8).
    p = cfg.TILES_Q * cfg.TILE_M * cfg.TILE_N * cfg.BPE if cfg.DTYPE_QKV <= 1 else 0
    raw_kv = max(cfg.TILE_N * cfg.TILE_K, cfg.TILE_O * cfg.TILE_N) // cfg.CTA_MMA if cfg.DTYPE_KV in (DTYPE_E4M3, DTYPE_E5M2) and cfg.DTYPE_QKV > 1 else 0
    return qo + k + v + p + raw_kv


def _validate_cfg_d128(cfg: CfgD128) -> None:
    """Consistency checks on the (mostly hardcoded) d128 (llama) geometry."""
    _fp8 = cfg.DTYPE_QKV <= 1  # E4M3/E5M2 inputs (MXFP8): STAGES_KV=4, TILE_K_HW=32, independent DTYPE_O
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d128: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d128: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d128: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA and cfg.CTA_MMA in (1, 2), "d128 SM100: CGA_M must equal CTA_MMA, and CTA_MMA must be 1 (cga1) or 2 (cga2)"),
        (
            cfg.QO_ALIAS == 1 if cfg.CTA_MMA == 1 and not cfg.DUAL_KV_STREAM else True,
            "d128 cga1: QO_ALIAS is mandatory — cga1 doubles per-CTA K/V (no collective MMA to halve it), "
            "so Q and O must share one slab to stay inside the SMEM cap (the two-stream fp8 decode keeps "
            "one unaliased Q and O slab and fits through its 1-byte K/V stages)",
        ),
        (
            cfg.Q_TILES_LIVE == (1 if cfg.DUAL_KV_STREAM else cfg.TILES_Q),
            "d128: Q_TILES_LIVE is TILES_Q, or 1 under DUAL_KV_STREAM (sub-tile 1 is a KV stream, not Q rows)",
        ),
        (
            not cfg.DUAL_KV_STREAM or (_fp8 and cfg.CTA_MMA == 1 and not cfg.THD_VARLEN and not cfg.PV_BF16),
            "d128: DUAL_KV_STREAM is the per-tensor fp8 cga1 dense/paged decode specialization",
        ),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d128: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.TILE_K == 128 and cfg.TILE_O == 128, "d128: d_qk = d_v = 128"),
        (cfg.TILES_Q == 2, "d128 (llama): TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d128 (llama): SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d128 (llama): CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d128 (llama): 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d128 llama: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (
            (cfg.STAGES_KV == 4 if (cfg.CTA_MMA == 2 or cfg.DUAL_KV_STREAM) else cfg.STAGES_KV in (2, 3)) if _fp8 else cfg.STAGES_KV == 2,
            "d128 SM100: STAGES_KV must be 2 (f16/bf16) or, for fp8/mxfp8, 4 at cga2 and 2 or 3 at cga1 — "
            "the stage depth scales with the cluster width so stages x per-CTA-buffer stays constant; "
            "the two-stream fp8 decode holds a K/V pair per step and runs 4 at cga1",
        ),
        (
            cfg.TILE_K_HW_BMM1 == (32 if _fp8 else 16) and cfg.TILE_K_HW_BMM2 == (16 if cfg.PV_BF16 else (32 if _fp8 else 16)),
            "d128: BMM1 uses K=32 for MXFP8; experimental BF16 PV uses K=16",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d128: Q/K swizzle must be 64/128B"),
        (cfg.PV_BF16 in (0, 1) and (not cfg.PV_BF16 or _fp8), "d128: pv_bf16 requires MXFP8 Q/K"),
        (cfg.BPE_V == (2 if cfg.PV_BF16 else cfg.BPE), "d128: V bytes/element must match the PV specialization"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d128: V/O swizzle out of range"),
        (
            cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16, DTYPE_O_NVFP4, DTYPE_O_MXFP8) if _fp8 else cfg.DTYPE_O == cfg.DTYPE_QKV,
            "d128: DTYPE_O must equal DTYPE_QKV for half input; fp8/mxfp8 allows an independent output dtype",
        ),
        (
            cfg.O_BLOCK_SCALE == O_BLOCK_SCALE_BY_DTYPE[cfg.DTYPE_O] and cfg.O_PACK_DIV == o_pack_div(cfg.DTYPE_O),
            "d128: O_BLOCK_SCALE / O_PACK_DIV must follow DTYPE_O",
        ),
        (
            cfg.O_BLOCK_SCALE == 0 or not (cfg.THD_VARLEN or cfg.SEQ_Q_LENS_PRESENT or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or cfg.PAGED_KV),
            "d128: block-scaled O serves dense (unpaged), unsplit, unpacked graphs only",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d128(params: TemplateParams, *, cga1_kv_stages: int = 2) -> Tuple[CfgD128, TmaIters]:
    """``cga1_kv_stages``: K/V ring depth of the fp8 family's single-CTA (cga1) configuration -- 2 by default (the
    per-tensor fp8 kernel's verified geometry); the MXFP8 kernel asks for 3 (see STAGES_KV below)."""
    if params.dtype_qkv not in (DTYPE_FP16, DTYPE_BF16):
        raise ValueError("half multi-Q requires FP16/BF16 query storage")
    if params.decode_tile or params.decode_quad or params.decode_q_tile or params.dual_kv_stream:
        raise ValueError("parameters select a different half main implementation")
    if params.thd_varlen and params.split_kv > 1:
        raise ValueError("THD split-KV requires the single-Q implementation")
    _validate_params("d128", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv <= 1  # E4M3/E5M2 inputs → MXFP8 kernel
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    o_div = o_pack_div(dtype_o)
    b_v = 2 if params.pv_bf16 else b
    # FP8/MXFP8 pins the Blackwell K=32 QMMA path (TILE_K_HW=32) and STAGES_KV=4
    # (BPE=1 → 8 KiB/stage, fits 4); f16/bf16 keep 16 / 2.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    cfg = CfgD128(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_KV=params.dtype_kv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b_v,
        PV_BF16=int(params.pv_bf16),
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        O_BLOCK_SCALE=O_BLOCK_SCALE_BY_DTYPE[dtype_o],
        O_PACK_DIV=o_div,
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        # cga1 has no collective MMA to halve per-CTA K/V, so Q and O must share
        # one slab to stay under the SMEM cap (_validate_cfg_d128 enforces it).
        QO_ALIAS=1 if params.cta_mma == 1 and not params.dual_kv_stream else 0,
        Q_TILES_LIVE=1 if params.dual_kv_stream else 2,
        DUAL_KV_STREAM=int(params.dual_kv_stream),
        Q_SWZ_BYTES=q_swz_bytes(128, b),
        K_SWZ_BYTES=q_swz_bytes(128, b),
        V_SWZ_BYTES=v_swz_bytes(128, params.cta_mma, b_v),
        O_SWZ_BYTES=o_swz_bytes(128, b_o, o_div),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=16 if params.pv_bf16 else tile_k_hw_fp8,
        # KV stage depth scales with the cluster width, as in cuDNN's own
        # kernels (stages_kv = N * CTA_MMA): cga1 has no collective MMA to halve
        # per-CTA K/V, so the stage count halves instead to keep the product --
        # and hence the SMEM -- constant.  Only the fp8 family needs this here:
        # f16/bf16 already fit at cga1 by aliasing Q and O, and their verified
        # cga1 configuration keeps STAGES_KV=2.  mxfp8 additionally stages E8M0
        # scale factors that the SMEM model cannot see, and at STAGES_KV=4 that
        # pushed a cga1 CTA to 237024 B against the 232448 B cap.
        # The MXFP8 kernel takes a THIRD full-width stage at cga1 (cga1_kv_stages=3): Q and O already share a slab
        # there, so 3 x (16 K + 16 V) KiB plus the scale factors still fit, and the deeper prefetch covers the
        # tile boundary of the short packed GQA legs that run cga1.  The two-stream per-tensor fp8 decode holds a
        # K/V pair per step and runs 4 at cga1.
        STAGES_KV=(4 if (params.cta_mma == 2 or params.dual_kv_stream) else cga1_kv_stages) if fp8 else 2,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # Partial PackGQA is wired in the f16/bf16 d128 kernel; the fp8 / mxfp8
        # siblings pack HEADS_PER_TILE = QH_PER_KH and keep the full-ratio contract.
        PACK_G=_pack_g(params, CfgD128.TILE_M, partial=not fp8),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
        # f16 / bf16 register split: 8 more per lane to the MMA / TMA / scheduler warpgroup, taken from the two
        # softmax warpgroups (2 x 184 + 88 + 56 = 512).  At 40 the f16 kernel's MMA warp held the 16 per-k-step Q
        # descriptors of both sub-tiles through local memory (7 STL / 14 LDL pairs per KV step on the path from the
        # K-full / P-ready waits to the tcgen05.mma issue, 112 B stack); at 56 it is spill-free and the softmax still
        # is at 184.  MEASURED (same-node A/B, cuDNN 9.28 control): B300 llama bf16 causal -2 % time (0.92x -> 0.90x
        # of cuDNN @2K, 0.95x -> 0.93x @8K), B200 and dense within noise.  The fp8 / mxfp8 siblings keep the defaults
        # (their MMA warp already fits 40; not re-measured).
        SOFTMAX_REGS=CfgD128.SOFTMAX_REGS if fp8 else 184,
        MMA_REGS=CfgD128.MMA_REGS if fp8 else 56,
        TMALDG_REGS=CfgD128.TMALDG_REGS if fp8 else 56,
        TMASTG_REGS=CfgD128.TMASTG_REGS if fp8 else 56,
        SCHEDULER_REGS=CfgD128.SCHEDULER_REGS if fp8 else 56,
        OTHER_REGS=CfgD128.OTHER_REGS if fp8 else 56,
    )
    _validate_cfg_d128(cfg)
    return cfg, _tma_iters(cfg)


@dataclass(frozen=True)
class CfgD64:
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
    TILE_K: int = 64
    TILE_O: int = 64
    DTYPE_QKV: int = 3
    DTYPE_O: int = 3
    BPE: int = 2
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    QO_ALIAS: int = 0
    SPLIT_PIPELINE: int = 0
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    TILES_Q: int = 2
    Q_TILES_LIVE: int = 2
    DUAL_KV_STREAM: int = 0
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2
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
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0
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
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    SPLIT_KV: int = 1
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    PACK_G: int = 1
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    DTYPE_KV: int = -1


def _validate_cfg_d64(cfg: CfgD64) -> None:
    """Consistency checks on the native d64 geometry."""
    checks = (
        (cfg.TILE_K == 64 and cfg.TILE_O == 64, "d64: d_qk = d_v = 64"),
        (cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16), "d64: DTYPE_QKV must be E4M3/E5M2/BF16/FP16"),
        (cfg.DTYPE_QKV <= 1 or cfg.DTYPE_O == cfg.DTYPE_QKV, "d64: DTYPE_O must equal DTYPE_QKV for half input"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d64: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d64: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d64: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA and cfg.CTA_MMA in (1, 2), "d64 SM100: CGA_M must equal CTA_MMA, and CTA_MMA must be 1 (cga1) or 2 (cga2)"),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d64: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.TILES_Q == 2, "d64: TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d64: SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d64: CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d64: 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d64: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (
            cfg.TILE_K_HW_BMM1 == (32 if cfg.DTYPE_QKV <= 1 else 16) and cfg.TILE_K_HW_BMM2 == (16 if cfg.PV_BF16 else (32 if cfg.DTYPE_QKV <= 1 else 16)),
            "d64: TILE_K_HW must be 32 for FP8/MXFP8 inputs and 16 for f16/bf16",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d64: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d64: V/O swizzle out of range"),
        (cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N, "d64: N_BMM2_CHUNKS * BMM2_CHUNK_SIZE must equal TILE_N (KV axis), not TILE_O"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d64(params: TemplateParams) -> Tuple[CfgD64, TmaIters]:
    if params.dtype_qkv not in (DTYPE_FP16, DTYPE_BF16):
        raise ValueError("half multi-Q requires FP16/BF16 query storage")
    if params.decode_tile or params.decode_quad or params.decode_q_tile or params.dual_kv_stream:
        raise ValueError("parameters select a different half main implementation")
    if params.thd_varlen and params.split_kv > 1:
        raise ValueError("THD split-KV requires the single-Q implementation")
    if _small_tile_form_requested(params):
        # A small-tile SEAM on the d64 record (a bias / kv_mask form, the pair item, a ring depth, a tail / quiet / max form, Q in TMEM, a
        # second S slot, a third softmax warp group, the 64-wide KV tile): the shared small-tile factory at SM100_ARCH.  The plain d64
        # record below is untouched -- its renderings (and the fp8 / mxfp8 d64 files') stay byte-identical.
        return _make_cfg_small_d_sm100(params, tile_k=64)
    _validate_params("d64", params)
    if params.decode_tile:
        raise ValueError("d64: decode_tile selects the decode tile (make_cfg_d64_decode), not the prefill one")
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv <= 1  # E4M3/E5M2 inputs: the FP8 / MXFP8 d128 kernel files at d_flavor=64
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    if fp8 and params.decode_tile:
        raise ValueError("d64: the decode tile is f16/bf16 only")
    # Same K-path rule as d128: FP8/MXFP8 pin the Blackwell K=32 QMMA path.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    b_v = 2 if params.pv_bf16 else b
    cfg = CfgD64(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b_v,
        PV_BF16=int(params.pv_bf16),
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        O_BLOCK_SCALE=O_BLOCK_SCALE_BY_DTYPE[dtype_o],
        O_PACK_DIV=o_pack_div(dtype_o),
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        # Mirror d128's rule rather than exploiting d64's smaller slabs: the
        # kernel body reads QO_ALIAS, so it is not a free SMEM knob, and only
        # the aliased arrangement is validated at cga1.
        QO_ALIAS=1 if params.cta_mma == 1 else 0,
        Q_SWZ_BYTES=q_swz_bytes(64, b),
        K_SWZ_BYTES=q_swz_bytes(64, b),
        V_SWZ_BYTES=v_swz_bytes(64, params.cta_mma, b_v),
        O_SWZ_BYTES=o_swz_bytes(64, b_o, o_pack_div(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=16 if params.pv_bf16 else tile_k_hw_fp8,
        # Deeper KV pipeline than d128's 2. The halved d64 slabs pay for it
        # (cga1: 32 KiB Q u O + 64 K + 64 V = 160 KiB against the 227 KiB cap;
        # cga2: 128 KiB), and ncu says this is where the room is worth spending:
        # `long_scoreboard` -- warps blocked on a memory dependency -- is by far
        # the dominant stall for this kernel (56.6% at SWA=128, 58.4% at full
        # causal), against ~2% for barrier stalls.
        STAGES_KV=4,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # partial=False: the capability row leaves (64, 64) out of
        # pack_gqa_partial_d_shapes, so only a group that FULLY divides TILE_M
        # is admitted here. _pack_g raises on a group the gate would not pass,
        # which is the check this used to open-code after _validate_cfg_d64.
        PACK_G=_pack_g(params, CfgD64.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d64(cfg)
    return cfg, _tma_iters(cfg)


SM100_ARCH = D128FamilyArch(name="SM100", smem_usable_bytes=_SM100_MAX_DYN_SMEM, smem_fixed_overhead=2 * 1024, tmem_total_cols=512, tmem_exclusive_alloc=False)


SM100_F16_SMALL_D_SHAPES = frozenset({(32, 32), (64, 64)})


SM100_SMALL_D_CGAS = {(32, 32): frozenset({1}), (64, 64): frozenset({1, 2})}


SM100_TILE_NS_BY_D_SHAPE = (((32, 32), frozenset({64, 128})), ((64, 64), frozenset({64, 128})))


SM100_SMALL_D_STAGES_KV_DEFAULT = 4


_STAGES_KV_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm100_small_d_stages_kv(tile_k: int, cta_mma: int) -> int:
    """The K / V ring depth of an SM100 small-tile form: the seam when set (validated to SMALL_D_STAGES_KV), else
    :data:`SM100_SMALL_D_STAGES_KV_DEFAULT`.  ONE helper for the budgets and the factory (rule 8b')."""
    if tile_k > 64:
        return CfgD128.STAGES_KV
    if _STAGES_KV_OVERRIDE_FOR_SWEEPS is not None:
        want = int(_STAGES_KV_OVERRIDE_FOR_SWEEPS)
        if want not in SMALL_D_STAGES_KV:
            raise ValueError(
                f"_STAGES_KV_OVERRIDE_FOR_SWEEPS={want}: the SM100 32- / 64-wide tiles' K / V ring is validated at {sorted(SMALL_D_STAGES_KV)} stages "
                f"(SMEM fit + the ring-wrap axis of the fresh-process census); a depth outside it is unvalidated, not merely slow"
            )
        return want
    return SM100_SMALL_D_STAGES_KV_DEFAULT


def sm100_bias_resident_tiles(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """How many ``tile_n``-column bias tiles the SM100 carveout holds beside the d128 family's Q / O slabs and K / V rings at
    ``tile_k = tile_o`` and the small tiles' ring depth (``config_d128_family.bias_resident_tiles`` at ``SM100_ARCH``; the Q / O slabs
    are NOT aliased on the small-tile forms).  0 = no resident block fits.  Derived at 4 stages: 2 (one-batch-entry) / 4 (pair) at
    the 32-wide tile, 5 / 10 at its 64-wide KV tile; **0 at the 64-wide tile's cga1 on either item**, 1 at its cga2 one-batch-entry
    item, 3 / 6 at its 64-wide KV tile (cga1, one-batch-entry / pair).  Never a Rubin threshold."""
    stages = sm100_small_d_stages_kv(tile_k, cta_mma)
    tile_m = 128 if tile_m is None else int(tile_m)
    return bias_resident_tiles(SM100_ARCH, tile_k, cta_mma, stages_kv=stages, tile_m=tile_m, bpe_val=bpe(dtype_qkv), tile_n=tile_n, tiles_q=tiles_q, pair=pair)


def sm100_bias_ring_stages(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """The deepest STREAMED bias ring the SM100 carveout holds beside the operands at this width (``config_d128_family.bias_ring_stages``
    at ``SM100_ARCH``): 3 / 2 stages, 0 when fewer than 2 fit.  Derived at 4 stages: 2 at the 32-wide tile's one-batch-entry item
    (its 64 KiB stages), 3 on its pair item; 0 at the 64-wide tile's 128-wide forms at cga1; 3 at its 64-wide KV tile."""
    stages = sm100_small_d_stages_kv(tile_k, cta_mma)
    tile_m = 128 if tile_m is None else int(tile_m)
    return bias_ring_stages(SM100_ARCH, tile_k, cta_mma, stages_kv=stages, tile_m=tile_m, bpe_val=bpe(dtype_qkv), tile_n=tile_n, tiles_q=tiles_q, pair=pair)


def sm100_bias_stages_pick(
    d_flavor: int, cta_mma: int, *, dtype_qkv: int = DTYPE_BF16, tile_n: int = 128, pair: bool = False, request: int = 0, tiles_q: int = 2
) -> int:
    """The ``STAGES_BIAS`` of a STREAMED plan on the SM100 small tiles: the request when it is in ``BIAS_RING_STAGES`` and fits, else the
    derived deepest ring (both item forms -- the Rubin one-batch-entry item's measured 2-stage preference is a prior the sweep decides);
    0 = no ring fits (the factory raises, the row / adapter decline)."""
    deepest = sm100_bias_ring_stages(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, pair=pair)
    want = int(request) if request else 0
    if want:
        return want if (want in BIAS_RING_STAGES and want <= deepest) else 0
    return deepest


SM100_BIAS_D_SHAPES = frozenset(
    shape for shape in SM100_F16_SMALL_D_SHAPES if any(sm100_bias_resident_tiles(shape[0], cga) > 0 for cga in SM100_SMALL_D_CGAS[shape])
)


@dataclass(frozen=True)
class CfgD32:
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
    TILE_K: int = 32
    TILE_O: int = 32
    DTYPE_QKV: int = 3
    DTYPE_O: int = 3
    BPE: int = 2
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    QO_ALIAS: int = 0
    SPLIT_PIPELINE: int = 0
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    TILES_Q: int = 2
    Q_TILES_LIVE: int = 2
    DUAL_KV_STREAM: int = 0
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2
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
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0
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
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    SPLIT_KV: int = 1
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    PACK_G: int = 1
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    DTYPE_KV: int = -1


_SMALL_TILE_RECORD_FIELDS = ("bias_mode", "kv_mask_form", "n_chunk", "bias_stages", "padded_tail_form", "dead_subtile_quiet", "bias_max_form", "q_in_tmem")


def _small_tile_form_requested(params: TemplateParams) -> bool:
    """Whether a d64 record asks for a small-tile SEAM (any of the appended small-tile fields off its default: a bias / kv_mask form, the pair
    item, a ring depth, a tail / quiet form, the row-max form, Q in TMEM, a second S slot, a third softmax warp group, the 64-wide KV tile).
    False = the plain d64 record every existing rendering is built from (``make_cfg_d64``'s own path, byte-identical)."""
    return (
        any(int(getattr(params, f, 0)) for f in _SMALL_TILE_RECORD_FIELDS)
        or int(getattr(params, "s_slots", 1)) != 1
        or int(getattr(params, "softmax_wgs", 2)) != 2
        or int(getattr(params, "tile_n", 128)) != 128
    )


def _make_cfg_small_d_sm100(params: TemplateParams, *, tile_k: int) -> Tuple[CfgD128, TmaIters]:
    """The f16/bf16 SMALL-tile factory of the SM100 d128 file (``TILE_K = TILE_O = tile_k`` in {32, 64}): the record's small-tile fields
    (bias / kv_mask / pair / ring / max form / tail / quiet / S_SLOTS / Q_IN_TMEM / softmax WGs / TILE_N) resolved against the shared
    validators at ``SM100_ARCH`` -- the sm107 factory's resolution with the SM100 budget, the SM100 ring depth, the 128-row tile and NO Q u O
    alias (the resident block needs the separate ``sO``; the plain d64 cga1 record keeps its alias on ``make_cfg_d64``'s path).  Every
    structural coupling raises here with the failure it would otherwise produce (frost-kernels.md s 5 / s 6); every rejection must also be
    a Capabilities / check_support decline."""
    flavor = f"d{tile_k}"
    tag = f"sm100 {flavor}"
    _validate_params(flavor, params)
    if params.dtype_qkv <= 1:
        raise ValueError(
            f"{tag}: the 32- / 64-wide small-tile forms are wired on the f16/bf16 d128 file only (dtype_qkv={params.dtype_qkv}); the fp8 / mxfp8 d64 flavors keep their own records"
        )
    if params.decode_tile or params.decode_q_tile:
        raise ValueError(f"{tag}: the decode tiles do not consume the small-tile record fields")
    if params.thd_varlen or params.split_kv != 1 or params.pack_gqa or params.paged_kv or params.epilogue_gate:
        raise ValueError(
            f"{tag}: the small-tile forms serve dense, unsplit, unpaged, non-PackGQA, ungated graphs (global or per-batch Q / KV lengths, the sm107 "
            f"factory's envelope) only -- THD / paged / split-KV / PackGQA ride the native d64 / d128 flavors, where those legs are wired"
        )
    cta_mma = params.cta_mma
    if cta_mma not in SM100_SMALL_D_CGAS[(tile_k, tile_k)]:
        raise ValueError(
            f"{tag}: cta_mma={cta_mma} is outside the validated widths {sorted(SM100_SMALL_D_CGAS[(tile_k, tile_k)])} for TILE_O={tile_k}; "
            f"at cga2 the per-CTA V operand is {tile_k // 2} columns = a {tile_k // 2 * 2}-byte MN-major swizzle atom"
        )
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_n = int(getattr(params, "tile_n", 128))
    if tile_n not in dict(SM100_TILE_NS_BY_D_SHAPE)[(tile_k, tile_k)]:
        raise ValueError(f"{tag}: TILE_N must be one of {sorted(dict(SM100_TILE_NS_BY_D_SHAPE)[(tile_k, tile_k)])} (got {tile_n})")
    s_slots = int(getattr(params, "s_slots", 1))
    q_in_tmem = int(getattr(params, "q_in_tmem", 0))
    if q_in_tmem not in SMALL_D_Q_IN_TMEM:
        raise ValueError(f"{tag}: q_in_tmem must be 0 (Q from SMEM) or 1 (Q resident in TMEM); got {q_in_tmem}")
    bias_mode = int(getattr(params, "bias_mode", 0))
    kv_mask_form = int(getattr(params, "kv_mask_form", 0))
    if (bias_mode or kv_mask_form) and (bias_mode not in BIAS_MODES or kv_mask_form not in KV_MASK_FORMS):
        raise ValueError(
            f"{tag}: bias_mode must be 0 / 1 (resident row-block) / 2 (streamed ring) / 3 (streamed ring, cluster multicast) and kv_mask_form "
            f"0 / 1 (bit word) / 2 / 3 (additive io-dtype / fp32 vector); got {bias_mode} / {kv_mask_form}"
        )
    n_chunk = int(getattr(params, "n_chunk", 0))
    pair = n_chunk > 0
    if pair and bias_mode == BIAS_MODE_NONE:
        raise ValueError(f"{tag}: n_chunk={n_chunk} is the (n, n+1) pair item of the BIAS arm; it needs bias_mode=1 / 2 (got {bias_mode})")
    bias_stages_req = int(getattr(params, "bias_stages", 0))
    if bias_stages_req and bias_mode not in BIAS_STREAMED_MODES:
        raise ValueError(f"{tag}: bias_stages={bias_stages_req} is the STREAMED ring's depth; it needs bias_mode=2 / 3 (got {bias_mode})")
    bias_max_form = int(getattr(params, "bias_max_form", 0))
    if bias_max_form not in BIAS_MAX_FORMS:
        raise ValueError(f"{tag}: bias_max_form must be 0 (the software row max) or 1 (the hardware upper bound); got {bias_max_form}")
    if bias_max_form == BIAS_MAX_HW_UPPER_BOUND:
        if not (bias_mode or kv_mask_form):
            raise ValueError(
                f"{tag}: bias_max_form=1 is the row-max form of the bias / kv_mask body; this record has neither arm (bias_mode=0, kv_mask_form=0)"
            )
        if kv_mask_form in (KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32):
            raise ValueError(
                f"{tag}: bias_max_form=1 (the hardware upper-bound row max) is served with the bit-word kv_mask or none; the additive kv_mask vector "
                f"(kv_mask_form={kv_mask_form}) has no per-tile max to bound -- keep bias_max_form=0 with it"
            )
    softmax_wgs = int(getattr(params, "softmax_wgs", 2))
    if softmax_wgs not in SMALL_D_SOFTMAX_WGS:
        raise ValueError(f"{tag}: softmax_wgs must be one of {sorted(SMALL_D_SOFTMAX_WGS)}; got {softmax_wgs}")
    if softmax_wgs == 3 and (tile_n != 64 or not pair or cta_mma != 1):
        raise ValueError(
            f"{tag}: softmax_wgs=3 serves the (n, n+1) pair item at TILE_N 64 and cga1 only (got tile_n={tile_n}, n_chunk={n_chunk}, cta_mma={cta_mma}): "
            f"its 128-register softmax holds one 64-column chunk and the one-batch-entry paths hand-unroll two sub-tiles"
        )
    stages_kv = sm100_small_d_stages_kv(tile_k, cta_mma)
    tile_m = 128
    softmax_regs, correction_regs = D128_REG_SPLIT[2 if softmax_wgs == 2 else (softmax_wgs, tile_k)]
    total_warps = d128_total_warps(softmax_wgs)
    corr_warp_base = softmax_wgs * D128_SOFTMAX_WG_WARPS
    mma_warp_id = corr_warp_base + D128_CORRECTION_WARPS
    bias_tiles = 0
    stages_bias = 0
    if bias_mode == BIAS_MODE_RESIDENT:
        bias_tiles = sm100_bias_resident_tiles(tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, tiles_q=softmax_wgs, pair=pair)
        if bias_tiles == 0:
            raise ValueError(
                f"{tag}: no resident bias block fits beside the Q / O slabs and the {stages_kv}-deep K / V ring at TILE_K={tile_k}, TILE_N={tile_n}, cta_mma={cta_mma} "
                f"({SM100_ARCH.smem_usable_bytes // 1024} KiB usable on SM100: the 64-wide tile's 128-wide forms hold no block at cga1 at 4 stages); the bias is "
                f"served on the widths of SM100_BIAS_D_SHAPES={sorted(SM100_BIAS_D_SHAPES)} at the cga / KV width / ring depth whose budget holds a tile"
            )
    elif bias_mode in BIAS_STREAMED_MODES:
        stages_bias = sm100_bias_stages_pick(
            tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, pair=pair, request=bias_stages_req, tiles_q=softmax_wgs
        )
        if stages_bias == 0:
            raise ValueError(
                f"{tag}: no {sorted(BIAS_RING_STAGES)}-stage streamed bias ring of {bias_block_rows(softmax_wgs, tile_m, pair=pair)} x {tile_n} tiles "
                f"{'(requested bias_stages=' + str(bias_stages_req) + ') ' if bias_stages_req else ''}fits beside the Q / O slabs and the {stages_kv}-deep "
                f"K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} ({SM100_ARCH.smem_usable_bytes // 1024} KiB usable on SM100)"
            )
    mask_flags = _mask_flags_from(params)
    padded_tail_form = int(getattr(params, "padded_tail_form", 0))
    if padded_tail_form not in PADDED_TAIL_FORMS:
        raise ValueError(f"{tag}: padded_tail_form must be 0 (the legacy tail body) or 1 (the pre-masked tail); got {padded_tail_form}")
    if padded_tail_form == 1 and (mask_flags != MASK_PADDED or kv_mask_form):
        raise ValueError(
            f"{tag}: padded_tail_form=1 (the pre-masked tail) is wired on the f16/bf16 32- / 64-wide tiles' padded body without a kv_mask only "
            f"(MASK_FLAGS == MASK_PADDED alone; the plain or the bias body); got MASK_FLAGS={mask_flags}, kv_mask_form={kv_mask_form}"
        )
    dead_subtile_quiet = int(getattr(params, "dead_subtile_quiet", 0))
    if dead_subtile_quiet not in DEAD_SUBTILE_QUIET_FORMS:
        raise ValueError(f"{tag}: dead_subtile_quiet must be 0 (the full body on a dead sub-tile) or 1 (the quiet body); got {dead_subtile_quiet}")
    if dead_subtile_quiet == 1 and (pair or not (bias_mode or kv_mask_form)):
        raise ValueError(
            f"{tag}: dead_subtile_quiet=1 (the dead-sub-tile quiet body) is wired on the -inf body (a bias or a kv_mask) of the one-batch-entry item only "
            f"on this line; got bias_mode={bias_mode}, kv_mask_form={kv_mask_form}, n_chunk={n_chunk}"
        )
    cls = CfgD64 if tile_k == 64 else CfgD32
    cfg = cls(
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_k,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b,
        PV_BF16=0,
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        O_BLOCK_SCALE=0,
        O_PACK_DIV=1,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        # No Q u O alias on the small-tile forms: the resident block / the pair item's Q refill need the separate sO (the sm107 geometry every
        # seam's barrier analysis assumed); the plain d64 cga1 record keeps its alias on make_cfg_d64's own path.
        QO_ALIAS=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_k, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_k, b_o, 1),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=softmax_wgs,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        SOFTMAX_WARPGROUPS=softmax_wgs,
        CORRECTION_WARPS=D128_CORRECTION_WARPS,
        SOFTMAX_REGS=softmax_regs,
        CORRECTION_REGS=correction_regs,
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=total_warps,
        THREADS_PER_CTA=total_warps * 32,
        SOFTMAX_WG_WARPS=D128_SOFTMAX_WG_WARPS,
        OTHER_WARPS=D128_OTHER_WARPS,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=D128_SOFTMAX_WG_WARPS,
        CORR_WARP_BASE=corr_warp_base,
        MMA_WARP_ID=mma_warp_id,
        TMALDG_WARP_ID=mma_warp_id + 1,
        TMASTG_WARP_ID=mma_warp_id + 2,
        SCHED_WARP_ID=mma_warp_id + 3,
        READ_TILE_ARRIVERS=d128_read_tile_arrivers(softmax_wgs),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        # The per-batch Q-length trim (the graph's seq_len_q under the padding mask) is the d128 body's own leg on every flavor of this file
        # (`SEQ_Q_LENS_PRESENT` folds the OOB-row store guard + the Q trim); a padded graph always binds it (engines: a dense padded graph
        # with seq_q_t compiles the trim specialization), so the small tiles take it like the sm107 factory does -- a refusal here was a
        # whole-row decline of every padded graph at D <= 32 (12 / 117 cells of the dev-box tier, 2026-10-13).
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=0,
        SPLIT_KV=1,
        PACK_GQA=0,
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=1,
        PAGED_KV=0,
        PAGE_SIZE=0,
        PADDED_TAIL_FORM=padded_tail_form,
        DEAD_SUBTILE_QUIET=dead_subtile_quiet,
        BIAS_MODE=bias_mode,
        BIAS_RESIDENT_TILES=bias_tiles,
        BIAS_BLOCK_ROWS=bias_block_rows(softmax_wgs, tile_m, pair=pair) if bias_mode else 0,
        BIAS_SWZ_BYTES=BIAS_SWZ_BYTES,
        BIAS_BOXES_PER_TILE=(tile_n * b) // BIAS_SWZ_BYTES if bias_mode else 0,
        BIAS_EMPTY_ARRIVERS=softmax_wgs * D128_SOFTMAX_WG_WARPS if (bias_mode or kv_mask_form) else 0,
        KV_MASK_FORM=kv_mask_form,
        N_CHUNK=n_chunk,
        STAGES_BIAS=stages_bias,
        BIAS_MAX_FORM=bias_max_form,
        S_SLOTS=s_slots,
        Q_IN_TMEM=q_in_tmem,
        BIAS_MCAST_CTAS=BIAS_MCAST_CTAS if bias_mode == BIAS_MODE_STREAMED_MULTICAST else 1,
    )
    # The SM100 facts the shared validators do not know: the f16 k-step and the half-precision O.
    if cfg.TILE_K_HW_BMM1 != 16 or cfg.TILE_K_HW_BMM2 != 16:
        raise ValueError(
            f"{tag}: the f16/bf16 small tiles run the 16-wide k-step (got {cfg.TILE_K_HW_BMM1} / {cfg.TILE_K_HW_BMM2}); the 2-chunk path is silently wrong on SM10x"
        )
    if cfg.DTYPE_O != cfg.DTYPE_QKV:
        raise ValueError(f"{tag}: DTYPE_O must equal DTYPE_QKV for half input")
    validate_regs(cfg, tag)
    validate_warp_layout(cfg, tag, expect_total=d128_total_warps(softmax_wgs), expect_arrivers=d128_read_tile_arrivers(softmax_wgs))
    validate_softmax_wgs(cfg, tag)
    validate_swizzles(cfg, tag)
    validate_head_dim_tile(cfg, tag, arch=SM100_ARCH)
    validate_tile_n(cfg, tag, arch=SM100_ARCH)
    validate_q_in_tmem(cfg, tag, arch=SM100_ARCH)
    validate_small_tile_masks(cfg, tag)
    validate_bias(cfg, tag, arch=SM100_ARCH)
    validate_smem(cfg, tag, arch=SM100_ARCH, qo_alias=False, bias_bytes=bias_resident_block_bytes(cfg))
    return cfg, _tma_iters(cfg)


def make_cfg_d32(params: TemplateParams) -> Tuple[CfgD32, TmaIters]:
    """The 32-wide head-dim tile of ``sm100/prefill_d128_f16.py`` (``TemplateParams.d_flavor=32``): every f16/bf16 head dim <= 32 with
    D % 8 == 0 by TMA zero fill; cga1 (SM100_SMALL_D_CGAS); dense / causal / bottom-right / SWA / padded / sink / GQA like d128 plus the
    small-tile seams; THD / split-KV / paged KV / PackGQA / the score side outputs ride the d128 flavor (the row's ``dense_only_d_shapes``)."""
    if params.dtype_qkv not in (DTYPE_FP16, DTYPE_BF16):
        raise ValueError("half multi-Q requires FP16/BF16 query storage")
    if params.decode_tile or params.decode_quad or params.decode_q_tile or params.dual_kv_stream:
        raise ValueError("parameters select a different half main implementation")
    if params.thd_varlen and params.split_kv > 1:
        raise ValueError("THD split-KV requires the single-Q implementation")
    return _make_cfg_small_d_sm100(params, tile_k=32)
