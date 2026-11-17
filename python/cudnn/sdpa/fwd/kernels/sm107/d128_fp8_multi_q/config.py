# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Owned D64/D128 per-tensor FP8 configuration and native pipeline geometry."""

from dataclasses import dataclass
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.config_sm107 import (
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
    DEAD_SUBTILE_QUIET_FORMS,
    KV_MASK_ADDITIVE_16,
    KV_MASK_ADDITIVE_F32,
    KV_MASK_FORMS,
    MASK_NONE,
    MASK_PADDED,
    PADDED_TAIL_FORMS,
    SCHED_NATURAL,
    SM107_BIAS_D_SHAPES,
    SM107_FP8_SPLIT_SHAPES,
    SM107_FP8_TILE_NS_BY_D_SHAPE,
    SM107_MXFP8_SPLIT_SHAPES,
    SM107_SMALL_D_CGAS,
    SMALL_D_Q_IN_TMEM,
    SMALL_D_SOFTMAX_WGS,
    SMEM_USABLE_BYTES,
    _D128_CORRECTION_WARPS,
    _D128_OTHER_WARPS,
    _D128_REG_SPLIT,
    _D128_SOFTMAX_WG_WARPS,
    _DTYPE_E5M2,
    _DTYPE_FP16,
    _O_BLOCK_SCALE_BY_DTYPE,
    _band_fields,
    _d128_read_tile_arrivers,
    _d128_total_warps,
    _pack_g,
    _stages_kv_d128,
    _tile_m_d128,
    _tma_iters,
    _validate_bias,
    _validate_dtype_k_step,
    _validate_head_dim_tile,
    _validate_masks,
    _validate_params,
    _validate_q_in_tmem,
    _validate_regs,
    _validate_smem,
    _validate_softmax_wgs,
    _validate_swizzles,
    _validate_tile_n,
    _validate_warp_layout,
    bias_resident_block_bytes,
    bpe,
    o_pack_div,
    o_swz_bytes,
    q_swz_bytes,
    rescale_threshold,
    resolve_dtype_o,
    sm107_bias_block_rows,
    sm107_bias_resident_tiles,
    sm107_bias_stages_pick,
    tile_k_hw,
    v_swz_bytes,
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
    DTYPE_QKV: int = _DTYPE_FP16
    DTYPE_O: int = _DTYPE_FP16
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
    MASK_FLAGS: int = MASK_NONE
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
    SCHEDULER_POLICY: int = SCHED_NATURAL
    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 64
    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
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
    QO_ALIAS: int = 0
    DTYPE_KV: int = -1


@dataclass(frozen=True)
class CfgD64(CfgD128):
    TILE_K: int = 64
    TILE_O: int = 64


def _make_cfg_d128_family(params: TemplateParams, *, flavor: str, tile_k: int, tile_o: int, mxfp8: bool):
    # Per-tensor FP8 D64/D128/D192, half D64/D128/D192 and MXFP8 D128 share FP32 split partials.
    if mxfp8:
        split_shapes = SM107_MXFP8_SPLIT_SHAPES
    else:
        split_shapes = SM107_FP8_SPLIT_SHAPES if params.dtype_qkv <= _DTYPE_E5M2 else {(64, 64), (128, 128), (192, 128)}
    split_wired = (tile_k, tile_o) in split_shapes
    # The block-scaled O epilogue lives in the two d128 kernels (per-tensor fp8 and
    # mxfp8); the d192xd128 siblings share this config family but not the epilogue.
    block_scaled_o_wired = tile_k == 128 and tile_o == 128
    _validate_params(flavor, params, split_wired=split_wired, block_scaled_o_wired=block_scaled_o_wired)
    # softmax_f16 / softmax_scale_prefolded: per-flavor backstops in _validate_params (SM107_*_FLAVORS).
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    o_div = o_pack_div(dtype_o)
    # TILE_N is a Cfg field (TemplateParams.tile_n): the block-scaled siblings keep 128 (their SF geometry is tied to the 128-column
    # tile), the f16/bf16 small tiles take 64 / 128 (SM107_TILE_NS_BY_D_SHAPE; the 64-wide rendering is trace-compiled and served
    # through the TILE_N knob), and the half D128 rendering takes 64 / 128; 256 is refused by _validate_tile_n.
    tile_n = int(getattr(params, "tile_n", 128))
    pertensor_tile_n = not mxfp8 and params.dtype_qkv <= _DTYPE_E5M2 and tile_n in dict(SM107_FP8_TILE_NS_BY_D_SHAPE).get((tile_k, tile_o), ())
    if tile_n != 128 and (params.dtype_qkv <= _DTYPE_E5M2 or tile_k > 128 or (tile_k == 128 and tile_n != 64)) and not pertensor_tile_n:
        raise ValueError(
            f"{flavor}: TILE_N = {tile_n} requires a supported half-precision width: D128 half and per-tensor FP8 D64 permit 64/128; other quantized renderings keep 128"
        )
    # The S / P slot count (TemplateParams.s_slots, appended; getattr so a record without the field renders one slot): 2 = the
    # double-buffered S of the f16/bf16 small tiles at TILE_N 64 -- _validate_tile_n pins the couplings (the TMEM fit, the width, the dtype).
    s_slots = int(getattr(params, "s_slots", 1))
    if s_slots != 1 and (params.dtype_qkv <= _DTYPE_E5M2 or mxfp8 or tile_k > 128):
        raise ValueError(f"{flavor}: s_slots = {s_slots} requires f16/bf16 TILE_K <= 128; quantized and wider renderings keep one slot")
    # Q resident in TMEM (TemplateParams.q_in_tmem, appended; getattr so a record without the field renders Q from SMEM): the f16/bf16
    # small tiles at cga1 / TILE_M 128 / two sub-tiles -- _validate_q_in_tmem pins every coupling (the TMEM fit, the UTCCP geometry, the dtype).
    q_in_tmem = int(getattr(params, "q_in_tmem", 0))
    if q_in_tmem not in SMALL_D_Q_IN_TMEM:
        raise ValueError(f"{flavor}: q_in_tmem must be 0 (Q from SMEM) or 1 (Q resident in TMEM); got {q_in_tmem}")
    if q_in_tmem and (params.dtype_qkv <= _DTYPE_E5M2 or mxfp8 or tile_k > 64):
        raise ValueError(
            f"{flavor}: q_in_tmem=1 (Q resident in TMEM) is wired on the f16/bf16 32- / 64-wide tiles only (the quantized and d128 renderings keep Q in SMEM: "
            f"at TILE_K 128 the Q slots would end at column {2 * 128 + 2 * 128 + 2 * 64} of the 576, past the STATS region)"
        )
    bias_mode = int(getattr(params, "bias_mode", 0))
    kv_mask_form = int(getattr(params, "kv_mask_form", 0))
    if bias_mode or kv_mask_form:
        if params.dtype_qkv <= _DTYPE_E5M2 or mxfp8:
            raise ValueError(f"{flavor}: the additive bias / kv_mask are wired on the f16/bf16 prefill kernel only (dtype_qkv={params.dtype_qkv})")
        if params.thd_varlen or (params.split_kv or 1) > 1 or params.pack_gqa or params.paged_kv or params.epilogue_gate:
            raise ValueError(f"{flavor}: the additive bias / kv_mask serve dense, unsplit, unpaged, non-PackGQA, ungated graphs only")
        if bias_mode not in BIAS_MODES or kv_mask_form not in KV_MASK_FORMS:
            raise ValueError(
                f"{flavor}: bias_mode must be 0 / 1 (resident row-block) / 2 (streamed ring) / 3 (streamed ring, cluster multicast) and kv_mask_form "
                f"0 / 1 (bit word) / 2 / 3 (additive io-dtype / fp32 vector); got {bias_mode} / {kv_mask_form}"
            )
    stages_kv = _stages_kv_d128(params.dtype_qkv, cta_mma, mxfp8=mxfp8, tile_k=tile_k)
    # The (n, n+1) pair item (TemplateParams.n_chunk, appended; read through getattr so a record without the field renders
    # the one-batch-entry item): a 128-row block, cga1, the bias arm only -- _validate_bias pins the couplings.
    n_chunk = int(getattr(params, "n_chunk", 0))
    pair = n_chunk > 0
    if pair and bias_mode == BIAS_MODE_NONE:
        raise ValueError(f"{flavor}: n_chunk={n_chunk} is the (n, n+1) pair item of the BIAS arm; it needs bias_mode=1 / 2 (got {bias_mode})")
    bias_stages_req = int(getattr(params, "bias_stages", 0))
    if bias_stages_req and bias_mode not in BIAS_STREAMED_MODES:
        raise ValueError(f"{flavor}: bias_stages={bias_stages_req} is the STREAMED ring's depth; it needs bias_mode=2 / 3 (got {bias_mode})")
    # The Q tile height (TILE_M): 128 on every flavor but the f16/bf16 small tiles, where the private seam may pin 64 (the TILE_M = 64
    # lever; ``_tile_m_d128`` is the ONE helper the budgets below read too, so the Q / O slabs, the block rows and the budgets agree).
    tile_m = _tile_m_d128(tile_k) if (tile_k <= 64 and params.dtype_qkv > _DTYPE_E5M2 and not mxfp8) else 128
    # The row-max form (appended; a record without the field renders the software tree): the hardware upper bound needs an arm whose body it
    # replaces and a kv_mask it can bound (none or the bit word) -- the additive vector is a typed decline here and in the adapter.
    bias_max_form = int(getattr(params, "bias_max_form", 0))
    if bias_max_form not in BIAS_MAX_FORMS:
        raise ValueError(f"{flavor}: bias_max_form must be 0 (the software row max) or 1 (the hardware upper bound); got {bias_max_form}")
    if bias_max_form == BIAS_MAX_HW_UPPER_BOUND:
        if not (bias_mode or kv_mask_form):
            raise ValueError(
                f"{flavor}: bias_max_form=1 is the row-max form of the bias / kv_mask body; this record has neither arm (bias_mode=0, kv_mask_form=0)"
            )
        if kv_mask_form in (KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32):
            raise ValueError(
                f"{flavor}: bias_max_form=1 (the hardware upper-bound row max) is served with the bit-word kv_mask or none; the additive kv_mask vector "
                f"(kv_mask_form={kv_mask_form}) has no per-tile max to bound -- keep bias_max_form=0 with it"
            )
    # The softmax warp-group count (TemplateParams.softmax_wgs, appended; read through getattr so a record without the field renders the
    # classic two-group body): 3 = the third softmax warp group of the TILE_N-64 pair item -- the backstops below mirror
    # _validate_softmax_wgs (the adapter's check_support declines typed first); the warp table, the split and the arriver counts derive
    # from it so no two of them can disagree.
    softmax_wgs = int(getattr(params, "softmax_wgs", 2))
    if softmax_wgs not in SMALL_D_SOFTMAX_WGS:
        raise ValueError(f"{flavor}: softmax_wgs must be one of {sorted(SMALL_D_SOFTMAX_WGS)}; got {softmax_wgs}")
    if softmax_wgs == 3:
        if params.dtype_qkv <= _DTYPE_E5M2 or mxfp8 or tile_k > 64:
            raise ValueError(f"{flavor}: softmax_wgs=3 (the third softmax warp group) is wired on the f16/bf16 32- / 64-wide tiles only")
        if tile_n != 64 or not pair or cta_mma != 1 or tile_m != 128:
            raise ValueError(
                f"{flavor}: softmax_wgs=3 serves the (n, n+1) pair item at TILE_N 64, cga1 and the 128-row Q tile only (got tile_n={tile_n}, n_chunk={n_chunk}, cta_mma={cta_mma}, tile_m={tile_m}): "
                f"its 128-register softmax holds one 64-column chunk and the one-batch-entry paths hand-unroll two sub-tiles"
            )
    softmax_regs, correction_regs = _D128_REG_SPLIT[2 if softmax_wgs == 2 else (softmax_wgs, tile_k)]
    if mxfp8:
        softmax_regs, correction_regs = 184, 104
    total_warps = _d128_total_warps(softmax_wgs)
    corr_warp_base = softmax_wgs * _D128_SOFTMAX_WG_WARPS
    mma_warp_id = corr_warp_base + _D128_CORRECTION_WARPS
    bias_tiles = 0
    stages_bias = 0
    if bias_mode == BIAS_MODE_RESIDENT:
        bias_tiles = sm107_bias_resident_tiles(tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, tile_m=tile_m, tiles_q=softmax_wgs, pair=pair)
        if bias_tiles == 0:
            raise ValueError(
                f"{flavor}: no resident bias block fits beside the Q / O slabs and the {stages_kv}-deep K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} "
                f"({SMEM_USABLE_BYTES // 1024} KiB usable); the bias is served on the widths of SM107_BIAS_D_SHAPES={sorted(SM107_BIAS_D_SHAPES)}"
            )
    elif bias_mode in BIAS_STREAMED_MODES:
        stages_bias = sm107_bias_stages_pick(
            tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, pair=pair, request=bias_stages_req, tiles_q=softmax_wgs
        )
        if stages_bias == 0:
            raise ValueError(
                f"{flavor}: no {sorted(BIAS_RING_STAGES)}-stage streamed bias ring of {sm107_bias_block_rows(softmax_wgs, tile_m, pair=pair)} x {tile_n} tiles "
                f"{'(requested bias_stages=' + str(bias_stages_req) + ') ' if bias_stages_req else ''}fits beside the Q / O slabs and the {stages_kv}-deep "
                f"K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} ({SMEM_USABLE_BYTES // 1024} KiB usable)"
            )
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    # The padded tail form (TemplateParams.padded_tail_form, appended; read through getattr so a record without the field renders the
    # legacy tail body): the pre-masked tail serves the f16/bf16 small tiles' PLAIN padded body only -- a band has masked segments on
    # both sides, the bias / kv_mask arms run the -inf body (one body already), the d128 flavor keeps its own rendering.
    padded_tail_form = int(getattr(params, "padded_tail_form", 0))
    if padded_tail_form not in PADDED_TAIL_FORMS:
        raise ValueError(f"{flavor}: padded_tail_form must be 0 (the legacy tail body) or 1 (the pre-masked tail); got {padded_tail_form}")
    if padded_tail_form == 1 and (mask_flags != MASK_PADDED or bias_mode or kv_mask_form or tile_k > 64 or mxfp8 or params.dtype_qkv <= _DTYPE_E5M2):
        raise ValueError(
            f"{flavor}: padded_tail_form=1 (the pre-masked tail) is wired on the f16/bf16 32- / 64-wide tiles' PLAIN padded body only "
            f"(MASK_FLAGS == MASK_PADDED alone, no bias / kv_mask); got MASK_FLAGS={mask_flags}, bias_mode={bias_mode}, kv_mask_form={kv_mask_form}, "
            f"TILE_K={tile_k}, dtype_qkv={params.dtype_qkv}"
        )
    # The dead-sub-tile quiet body (TemplateParams.dead_subtile_quiet, appended; read through getattr so a record without the field
    # renders the full body): the one-batch-entry item of the f16/bf16 small tiles without a bias -- the bias arms also wait and
    # release the bias barriers per item, the pair item's dead sub-tile is the odd chunk tail, the d128 flavor keeps its rendering.
    dead_subtile_quiet = int(getattr(params, "dead_subtile_quiet", 0))
    if dead_subtile_quiet not in DEAD_SUBTILE_QUIET_FORMS:
        raise ValueError(f"{flavor}: dead_subtile_quiet must be 0 (the full body on a dead sub-tile) or 1 (the quiet body); got {dead_subtile_quiet}")
    if dead_subtile_quiet == 1 and (
        bias_mode
        or pair
        or tile_k > 64
        or mxfp8
        or params.dtype_qkv <= _DTYPE_E5M2
        or params.pack_gqa
        or (params.thd_varlen and (tile_k != 64 or params.paged_kv or (params.split_kv or 1) > 1))
        or params.epilogue_gate
    ):
        raise ValueError(
            f"{flavor}: dead_subtile_quiet=1 (the dead-sub-tile quiet body) is wired on the f16/bf16 32- / 64-wide tiles' one-batch-entry item "
            f"without a bias, PackGQA or the gated epilogue; THD requires non-paged, unsplit D64; got bias_mode={bias_mode}, n_chunk={n_chunk}, TILE_K={tile_k}, "
            f"dtype_qkv={params.dtype_qkv}, pack_gqa={params.pack_gqa}, thd_varlen={params.thd_varlen}, epilogue_gate={params.epilogue_gate}"
        )
    cls = {128: CfgD128, 64: CfgD64}[tile_k]
    if tile_k <= 64:
        # Per-tensor FP8 can issue one Rubin K=64 MMA at the native D64 width.
        # Block-scaled FP8 and D32 retain their existing larger envelope.
        if params.dtype_qkv <= _DTYPE_E5M2 and (mxfp8 or tile_k != 64):
            raise ValueError(f"{flavor}: quantized small head-dim tiles require per-tensor FP8 D64")
        if cta_mma not in SM107_SMALL_D_CGAS[(tile_k, tile_o)]:
            raise ValueError(
                f"{flavor}: cta_mma={cta_mma} is outside the validated widths {sorted(SM107_SMALL_D_CGAS[(tile_k, tile_o)])} for TILE_O={tile_o}; "
                f"at cga2 the per-CTA V operand is {tile_o // 2} columns = a {tile_o // 2 * b}-byte MN-major swizzle atom"
            )

    cfg = cls(
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_KV=params.dtype_kv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        QO_ALIAS=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o, o_div),
        O_BLOCK_SCALE=_O_BLOCK_SCALE_BY_DTYPE.get(dtype_o, 0),
        O_PACK_DIV=o_div,
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=softmax_wgs,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        SOFTMAX_WARPGROUPS=softmax_wgs,
        CORRECTION_WARPS=_D128_CORRECTION_WARPS,
        SOFTMAX_REGS=softmax_regs,
        CORRECTION_REGS=correction_regs,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        PADDED_TAIL_FORM=padded_tail_form,
        DEAD_SUBTILE_QUIET=dead_subtile_quiet,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        PACK_G=_pack_g(params, tile_m, partial=params.dtype_qkv > _DTYPE_E5M2),
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
        TOTAL_WARPS=total_warps,
        THREADS_PER_CTA=total_warps * 32,
        SOFTMAX_WG_WARPS=_D128_SOFTMAX_WG_WARPS,
        OTHER_WARPS=_D128_OTHER_WARPS,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=_D128_SOFTMAX_WG_WARPS,
        CORR_WARP_BASE=corr_warp_base,
        MMA_WARP_ID=mma_warp_id,
        TMALDG_WARP_ID=mma_warp_id + 1,
        TMASTG_WARP_ID=mma_warp_id + 2,
        SCHED_WARP_ID=mma_warp_id + 3,
        READ_TILE_ARRIVERS=_d128_read_tile_arrivers(softmax_wgs),
        BIAS_MODE=bias_mode,
        BIAS_RESIDENT_TILES=bias_tiles,
        BIAS_BLOCK_ROWS=sm107_bias_block_rows(softmax_wgs, tile_m, pair=pair) if bias_mode else 0,
        BIAS_SWZ_BYTES=BIAS_SWZ_BYTES,
        BIAS_BOXES_PER_TILE=(tile_n * b) // BIAS_SWZ_BYTES if bias_mode else 0,
        BIAS_EMPTY_ARRIVERS=softmax_wgs * _D128_SOFTMAX_WG_WARPS if (bias_mode or kv_mask_form) else 0,
        KV_MASK_FORM=kv_mask_form,
        N_CHUNK=n_chunk,
        STAGES_BIAS=stages_bias,
        BIAS_MAX_FORM=bias_max_form,
        S_SLOTS=s_slots,
        Q_IN_TMEM=q_in_tmem,
        # The multicast ring's cluster extent along the chunk axis (its fan-out); 1 everywhere else -- _validate_bias pins the coupling.
        BIAS_MCAST_CTAS=BIAS_MCAST_CTAS if bias_mode == BIAS_MODE_STREAMED_MULTICAST else 1,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_d128_total_warps(softmax_wgs), expect_arrivers=_d128_read_tile_arrivers(softmax_wgs))
    _validate_softmax_wgs(cfg, flavor)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_head_dim_tile(cfg, flavor)
    _validate_tile_n(cfg, flavor)
    _validate_q_in_tmem(cfg, flavor)
    _validate_masks(cfg, flavor)
    _validate_bias(cfg, flavor)
    _validate_smem(cfg, flavor, qo_alias=bool(cfg.QO_ALIAS), bias_bytes=bias_resident_block_bytes(cfg))
    return cfg, _tma_iters(cfg)


def _check_owned_inputs(params):
    if params.dtype_qkv not in (0, 1):
        raise ValueError("SM107 D64/D128 per-tensor FP8 requires E4M3/E5M2 inputs")
    if params.mma_2x2:
        raise ValueError("SM107 D64/D128 FP8 does not select a 2x2 implementation")


def make_cfg_d64(params: TemplateParams):
    _check_owned_inputs(params)
    if params.thd_varlen:
        raise ValueError("sm107 d64: per-tensor FP8 THD is not wired at the native D64 width")
    return _make_cfg_d128_family(params, flavor="sm107 d64", tile_k=64, tile_o=64, mxfp8=False)


def make_cfg_d128(params: TemplateParams):
    _check_owned_inputs(params)
    return _make_cfg_d128_family(params, flavor="sm107 d128", tile_k=128, tile_o=128, mxfp8=False)
