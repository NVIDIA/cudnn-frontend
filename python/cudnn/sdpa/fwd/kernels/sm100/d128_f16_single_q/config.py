# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent records and factories for the shared half single-Q main implementation.

D64/D128/D192 are geometry variants; quadrant cooperation is a different body.
Graph owners separately bound which geometries are admitted in this migration.
"""

from dataclasses import dataclass
from typing import Tuple
from cudnn.frost.tile_dsl.constants import DTYPE_FP16, DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, MASK_NONE, SCHED_NATURAL
from cudnn.sdpa.fwd.config_common import (
    TemplateParams,
    TmaIters,
    _validate_params,
    _mask_flags_from,
    _pack_g,
    _tma_iters,
    bpe,
    q_swz_bytes,
    v_swz_bytes,
    o_swz_bytes,
    rescale_threshold,
    tile_k_hw,
)

_SM100_MAX_DYN_SMEM = 227 * 1024


@dataclass(frozen=True)
class CfgD128Decode:
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
    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
    CGA_M: int = 1
    CGA_N: int = 1
    CTA_MMA: int = 1
    QO_ALIAS: int = 1
    SPLIT_PIPELINE: int = 0
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    TILES_Q: int = 1
    Q_TILES_LIVE: int = 1
    DUAL_KV_STREAM: int = 0
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 3
    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4
    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 96
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40
    RESCALE_THRESHOLD: float = 8.0
    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0
    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL
    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64
    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256
    READ_TILE_ARRIVERS: int = 11
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
    RAGGED_Q: int = 0


@dataclass(frozen=True)
class CfgD64Decode:
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
    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
    CGA_M: int = 1
    CGA_N: int = 1
    CTA_MMA: int = 1
    QO_ALIAS: int = 1
    SPLIT_PIPELINE: int = 0
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    TILES_Q: int = 1
    Q_TILES_LIVE: int = 1
    DUAL_KV_STREAM: int = 0
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 3
    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4
    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 96
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40
    RESCALE_THRESHOLD: float = 8.0
    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0
    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL
    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64
    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256
    READ_TILE_ARRIVERS: int = 11
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


def _validate_cfg_d128_decode(cfg: CfgD128Decode) -> None:
    """Consistency checks on the d128 decode-tile geometry."""
    checks = (
        (
            cfg.Q_TILES_LIVE == cfg.TILES_Q == 1 and cfg.DUAL_KV_STREAM == 0,
            "d128 decode: one live Q tile, no second KV stream (Q_TILES_LIVE / DUAL_KV_STREAM must not inherit CfgD128's defaults)",
        ),
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "d128 decode: f16/bf16 only (the fp8 families keep the prefill tile)"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d128 decode: DTYPE_O must equal DTYPE_QKV"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d128 decode: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d128 decode: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d128 decode: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 1 and cfg.CTA_MMA == 1, "d128 decode: one independent CTA per tile (cga1)"),
        (cfg.QO_ALIAS == 1, "d128 decode: Q and O share one SMEM slab (pays for the third KV stage)"),
        (cfg.TILES_Q == 1, "d128 decode: TILES_Q must be 1 (one 128-row Q tile per CTA)"),
        (cfg.TILE_M == 128 and cfg.TILE_N == 128 and cfg.TILE_K in (128, 192) and cfg.TILE_O == 128, "single-Q tile: QK width 128 or 192, V width 128"),
        (cfg.STAGES_KV == (3 if cfg.TILE_K == 128 and cfg.DTYPE_KV == -1 else 2), "single-Q tile: KV stages must fit the QK width and storage conversion"),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d128 decode: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 4, "d128 decode: one softmax warpgroup, four correction warps"),
        (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, "d128 decode: 12 warps / 384 threads"),
        (
            cfg.SOFTMAX_WG0_BASE == 0
            and cfg.CORR_WARP_BASE == 4
            and cfg.MMA_WARP_ID == 8
            and cfg.TMALDG_WARP_ID == 9
            and cfg.TMASTG_WARP_ID == 10
            and cfg.SCHED_WARP_ID == 11,
            "d128 decode: role layout and warp count disagree",
        ),
        (cfg.READ_TILE_ARRIVERS == 11, f"d128 decode: expected READ_TILE_ARRIVERS=11, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d128 decode: f16 K=16 MMA phases"),
        (
            not cfg.THD_VARLEN or ((cfg.TILE_K == 128 and cfg.SPLIT_KV > 1) or (cfg.TILE_K == 192 and not cfg.PAGED_KV and not cfg.PACK_GQA)),
            "single-Q THD: D128 split or unpacked nonpaged D192",
        ),
        (
            cfg.RAGGED_Q == 0 or (cfg.SPLIT_KV >= 2 and cfg.PAGED_KV == 1 and cfg.SEQ_Q_LENS_PRESENT == 0),
            "d128 decode: RAGGED_Q rides the split path over paged K/V (SPLIT_KV >= 2, PAGED_KV, no dense Q-length trim)",
        ),
        (
            cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.O_SWZ_BYTES == 128,
            "d128 decode: 128 B swizzle on every operand",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d128_decode(params: TemplateParams) -> Tuple[CfgD128Decode, TmaIters]:
    """Config for the shared single-Q half pipeline at ``cta_mma=1``.

    Backstop, like every ``make_cfg_*``: the (128, 128) f16/bf16 row admits
    cga=1 on dense graphs and on the ragged-Q-over-paged-KV leg (``ragged_q``,
    S_q(max) == 1), and the adapter routes exactly those combinations here;
    D192/V128 reuses it for unpacked nonpaged THD, with two KV stages to
    accommodate the wider Q/K slabs. Anything else raising below is a gap
    in those gates.
    """
    if params.decode_quad:
        raise ValueError("single-Q configuration cannot select the quadrant implementation")
    _validate_params("d128", params)
    if params.cta_mma != 1:
        raise ValueError(f"d128 decode: the decode tile is cga1 only (cta_mma=1); got cta_mma={params.cta_mma}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"d128 decode: f16/bf16 inputs only (DTYPE_QKV 2/3); got {params.dtype_qkv}")
    d_qk = params.single_q_head_dim
    if params.dtype_kv != -1 and d_qk != 128:
        raise ValueError("independent FP8 KV storage requires native D128")
    if d_qk not in (128, 192):
        raise ValueError("single-Q tile: QK width must be 128 or 192")
    if d_qk == 192 and not (params.thd_varlen and not params.paged_kv and not params.pack_gqa):
        raise ValueError("D192 single-Q tile requires unpacked nonpaged THD")
    if params.thd_varlen and not ((d_qk == 192 and not params.paged_kv) or (d_qk == 128 and params.split_kv > 1)):
        raise ValueError("single-Q THD: D128 split or unpacked nonpaged D192")
    if params.pv_bf16 or not params.emit_amax_o:
        raise ValueError("d128 decode: pv_bf16 / emit_amax_o are MXFP8-only experiment axes")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    cfg = CfgD128Decode(
        TILE_K=d_qk,
        # A raw FP8 tile shares the local load warp between K and V. Two
        # converted stages leave room for that16KiB slab; uniform half keeps3.
        STAGES_KV=3 if d_qk == 128 and params.dtype_kv == -1 else 2,
        DTYPE_KV=params.dtype_kv,
        THD_VARLEN=int(params.thd_varlen),
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b,
        BPE_O=bpe(dtype_o),
        Q_SWZ_BYTES=q_swz_bytes(d_qk, b),
        K_SWZ_BYTES=q_swz_bytes(d_qk, b),
        V_SWZ_BYTES=v_swz_bytes(128, 1, b),
        O_SWZ_BYTES=o_swz_bytes(128, bpe(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # Partial PackGQA, as on the d128 prefill tile: the kernel packs the
        # largest divisor of the group that divides the 128-row tile
        # (HEADS_PER_TILE = PACK_G; a group sharing no factor with it raises here).
        PACK_G=_pack_g(params, CfgD128Decode.TILE_M, partial=True),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
        RAGGED_Q=int(params.ragged_q),
    )
    _validate_cfg_d128_decode(cfg)
    return cfg, _tma_iters(cfg)


def _validate_cfg_d64_decode(cfg: CfgD64Decode) -> None:
    """Consistency checks on the d64 decode-tile geometry.

    The d128 decode checks with the narrow extents: same reshape, same role
    layout, only TILE_K / TILE_O and the V swizzle differ (a 64-wide V row is
    128 B at cga1, so every operand still lands on the 128 B atom).
    """
    checks = (
        (
            cfg.Q_TILES_LIVE == cfg.TILES_Q == 1 and cfg.DUAL_KV_STREAM == 0,
            "d64 decode: one live Q tile, no second KV stream (Q_TILES_LIVE / DUAL_KV_STREAM must not inherit CfgD128's defaults)",
        ),
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "d64 decode: f16/bf16 only"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d64 decode: DTYPE_O must equal DTYPE_QKV"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d64 decode: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d64 decode: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d64 decode: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 1 and cfg.CTA_MMA == 1, "d64 decode: one independent CTA per tile (cga1)"),
        (cfg.QO_ALIAS == 1, "d64 decode: Q and O share one SMEM slab"),
        (cfg.TILES_Q == 1, "d64 decode: TILES_Q must be 1 (one 128-row Q tile per CTA)"),
        (cfg.TILE_M == 128 and cfg.TILE_N == 128 and cfg.TILE_K == 64 and cfg.TILE_O == 64, "d64 decode: 128x128 tiles, d_qk = d_v = 64"),
        (cfg.STAGES_KV == 3, "d64 decode: STAGES_KV must be 3"),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d64 decode: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 4, "d64 decode: one softmax warpgroup, four correction warps"),
        (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, "d64 decode: 12 warps / 384 threads"),
        (
            cfg.SOFTMAX_WG0_BASE == 0
            and cfg.CORR_WARP_BASE == 4
            and cfg.MMA_WARP_ID == 8
            and cfg.TMALDG_WARP_ID == 9
            and cfg.TMASTG_WARP_ID == 10
            and cfg.SCHED_WARP_ID == 11,
            "d64 decode: role layout and warp count disagree",
        ),
        (cfg.READ_TILE_ARRIVERS == 11, f"d64 decode: expected READ_TILE_ARRIVERS=11, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d64 decode: f16 K=16 MMA phases"),
        (not cfg.THD_VARLEN or cfg.PAGED_KV or cfg.SPLIT_KV > 1, "d64 decode: unsplit THD requires paged KV"),
        (cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N, "d64 decode: BMM2 chunking is TILE_N-derived"),
        (
            cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.O_SWZ_BYTES == 128,
            "d64 decode: 128 B swizzle on every operand",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d64_decode(params: TemplateParams) -> Tuple[CfgD64Decode, TmaIters]:
    """Config for ``sm100/decode_d128_f16.py`` at ``d_flavor=64`` -- the d64
    flavor at ``cta_mma=1``.  Backstop, like every ``make_cfg_*``."""
    if params.decode_quad:
        raise ValueError("single-Q configuration cannot select the quadrant implementation")
    _validate_params("d64", params)
    if not params.decode_tile:
        raise ValueError("d64 decode: this tile is selected by decode_tile=True (the d64 prefill leg also runs cga1)")
    if params.cta_mma != 1:
        raise ValueError(f"d64 decode: the decode tile is cga1 only (cta_mma=1); got cta_mma={params.cta_mma}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"d64 decode: f16/bf16 inputs only (DTYPE_QKV 2/3); got {params.dtype_qkv}")
    if params.thd_varlen and not params.paged_kv and params.split_kv == 1:
        # Nonpaged split loads use compact or per-sequence input descriptors;
        # the shared combine owns final O/Stats origins. Unsplit stays on prefill.
        raise ValueError("d64 decode: unsplit THD requires paged KV")
    if params.pv_bf16 or not params.emit_amax_o:
        raise ValueError("d64 decode: pv_bf16 / emit_amax_o are MXFP8-only experiment axes")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    cfg = CfgD64Decode(
        THD_VARLEN=int(params.thd_varlen),
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b,
        BPE_O=bpe(dtype_o),
        Q_SWZ_BYTES=q_swz_bytes(64, b),
        K_SWZ_BYTES=q_swz_bytes(64, b),
        V_SWZ_BYTES=v_swz_bytes(64, 1, b),
        O_SWZ_BYTES=o_swz_bytes(64, bpe(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # The shared single-Q body addresses packed rows through PACK_G,
        # including a proper divisor of G on the native-width SM107 split leg.
        # SM100 graph admission retains its existing whole-group domain.
        PACK_G=_pack_g(params, CfgD64Decode.TILE_M, partial=True),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d64_decode(cfg)
    return cfg, _tma_iters(cfg)
