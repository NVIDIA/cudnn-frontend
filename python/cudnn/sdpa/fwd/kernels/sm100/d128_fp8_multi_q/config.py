# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent record and factory for per-tensor FP8 multi_q.

Legacy field defaults are retained; no half/MXFP8 or sibling factory is imported.
"""

from dataclasses import dataclass
from typing import Tuple
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, DTYPE_FP16, DTYPE_O_MXFP8, DTYPE_O_NVFP4, O_BLOCK_SCALE_BY_DTYPE
from cudnn.sdpa.fwd.config_common import (
    TemplateParams,
    TmaIters,
    _mask_flags_from,
    _pack_g,
    _tma_iters,
    _validate_params,
    bpe,
    o_pack_div,
    o_swz_bytes,
    q_swz_bytes,
    rescale_threshold,
    tile_k_hw,
    v_swz_bytes,
)

_SM100_MAX_DYN_SMEM = 227 * 1024


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
    if params.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2):
        raise ValueError("per-tensor FP8 implementation requires FP8 input")
    if bool(params.dual_kv_stream) != False:
        raise ValueError("configuration selects a different FP8 main implementation")
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
    if params.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2):
        raise ValueError("per-tensor FP8 implementation requires FP8 input")
    if bool(params.dual_kv_stream) != False:
        raise ValueError("configuration selects a different FP8 main implementation")
    if _small_tile_form_requested(params):
        # Preserve the legacy FP8 decline without importing the half factory.
        _validate_params("d64", params)
        raise ValueError(
            f"sm100 d64: the 32- / 64-wide small-tile forms are wired on the f16/bf16 d128 file only (dtype_qkv={params.dtype_qkv}); the fp8 / mxfp8 d64 flavors keep their own records"
        )
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
