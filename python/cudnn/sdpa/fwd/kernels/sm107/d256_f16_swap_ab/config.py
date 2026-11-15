# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent SM107 half configuration and local immutable record."""

from dataclasses import dataclass
from cudnn.sdpa.fwd.config_sm107 import (
    MASK_CAUSAL,
    MASK_SWA,
    SMEM_STANDARD_CARVEOUT_BYTES,
    SMEM_USABLE_BYTES,
    TCGEN05_V0_ADDR_LIMIT,
    TemplateParams,
    Tuple,
    _D256_DECODE_BARRIER_WORDS,
    _D256_DECODE_COLS,
    _D256_DECODE_FLAVOR,
    _D256_DECODE_MAX_RED_SLOTS,
    _D256_DECODE_OVERSIZED_Q_TILES,
    _D256_DECODE_Q_TILES_SM107,
    _D256_DECODE_WIDE_PACK_MIN,
    _align_slab,
    _check,
    tile_k_hw,
)
from cudnn.sdpa.fwd.config_swap_ab import (
    DTYPE_BF16,
    DTYPE_FP16,
    TmaIters,
    _D256_DECODE_Q_TILES,
    _mask_flags_from,
    _tma_iters,
    _validate_cfg_d256_decode,
    bpe,
)


@dataclass(frozen=True)
class CfgD256Decode:
    TILE_N: int = 128
    TILE_K: int = 256
    TILE_O: int = 256
    N_Q: int = 16
    DTYPE_QKV: int = 3
    DTYPE_O: int = 3
    BPE: int = 2
    BPE_O: int = 2
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    P_SWZ_BYTES: int = 32
    TILE_K_HW: int = 16
    STAGES_KV: int = 3
    MASK_FLAGS: int = 0
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    SPLIT_KV: int = 1
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    SOFTMAX_WARPS: int = 4
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TOTAL_WARPS: int = 6
    THREADS_PER_CTA: int = 192


def _make_cfg_d256_decode_record(params: TemplateParams, q_tiles: Tuple[int, ...] = _D256_DECODE_Q_TILES) -> Tuple[CfgD256Decode, TmaIters]:
    """Config for sm100/d256_f16_swap_ab/kernel.py from the adapter's TemplateParams.

    Backstop only (see TemplateParams): the adapter selects this tile for
    decode-shaped f16/bf16 d256 graphs and keeps every other graph on the
    prefill tile, so a ValueError here is an adapter gap, not a user error.
    ``cta_mma`` / ``sched_policy`` are accepted and unused — the decode tile
    is one cta_group::1 CTA per unit with nothing to schedule.

    ``q_tiles`` (appended): the tile widths the calling arch line compiles.
    The default is this line's ``_D256_DECODE_Q_TILES`` (16, 32: what 227 KiB
    holds); ``config_sm107.make_cfg_d256_decode`` passes the Rubin set, which
    adds the 64-column tile (four softmax groups, 258 KiB of SMEM under the
    320 KiB Rubin carveout) and re-validates the record against its own facts.
    """
    if params.qsa_block_topk or params.qsa_block_size or params.qsa_softmax_groups:
        # Same decline as the prefill templates' _validate_params: an index-list record belongs to the cc 10.7 sparse
        # kernel (its own swap-AB body over gathered K/V), never to this dense ring.
        raise ValueError(
            f"d256 decode: qsa_block_topk={params.qsa_block_topk} / qsa_block_size={params.qsa_block_size} / qsa_softmax_groups={params.qsa_softmax_groups} "
            f"select the index-list sparse kernel; the decode tile does not consume them"
        )
    if params.decode_q_tile not in q_tiles:
        raise ValueError(f"d256 decode: decode_q_tile must be one of {q_tiles}; got {params.decode_q_tile}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError("d256 decode: f16/bf16 inputs only")
    if params.thd_varlen:
        raise ValueError("d256 decode: THD/varlen queries ride the prefill tile")
    if params.pv_bf16 or params.softmax_f16:
        raise ValueError("d256 decode: pv_bf16 / softmax_f16 are quantized-kernel specializations")
    if params.window_left is not None and params.window_left < 0:
        raise ValueError(f"d256 decode: window_left must be >= 0 (or None); got {params.window_left}")
    if params.window_right is not None and params.window_right < 0:
        raise ValueError(f"d256 decode: window_right must be >= 0 (or None); got {params.window_right}")
    if params.bottom_right and params.window_right is None:
        raise ValueError("d256 decode: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if params.seq_q_lens_present and not params.seq_kv_lens_present:
        raise ValueError("d256 decode: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b = bpe(params.dtype_qkv)
    softmax_warps = 4 * (int(params.decode_q_tile) // 16)
    decode_head_dim = getattr(params, "decode_head_dim", 256)
    if decode_head_dim not in (128, 256):
        raise ValueError("swap-AB decode: head-dimension envelope must be 128 or 256")
    cfg = CfgD256Decode(
        TILE_K=decode_head_dim,
        TILE_O=decode_head_dim,
        N_Q=int(params.decode_q_tile),
        SOFTMAX_WARPS=softmax_warps,
        MMA_WARP_ID=softmax_warps,
        TMALDG_WARP_ID=softmax_warps + 1,
        TOTAL_WARPS=softmax_warps + 2,
        THREADS_PER_CTA=(softmax_warps + 2) * 32,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=bpe(dtype_o),
        P_SWZ_BYTES=int(params.decode_q_tile) * b,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d256_decode(cfg, q_tiles)
    return cfg, _tma_iters(cfg)


def d256_decode_smem_layout(cfg: CfgD256Decode) -> dict:
    """Byte offsets of the decode tile's SMEM slabs in the kernel's DECLARATION order
    (``sQ_raw | sKV_raw | sP_raw | mred_smem | red_smem | tmem_ptr | mbarriers``), every slab
    1024-aligned like the kernel's ``cutlass.Array(..., alignment=1024)`` (the small
    trailing arrays at their 16-B alignment), plus the total, the standard-carveout
    verdict and the carveout the launch needs (227 KiB standard, else the 320 KiB
    oversized one).  The MMA-operand slabs are the first three; ``p_start`` is the
    highest operand start and is what the descriptor-version derivation reads;
    ``p_end`` is the last operand byte + 1 (the version-0 window check reads both)."""
    q_bytes = cfg.N_Q * cfg.TILE_K * cfg.BPE
    kv_bytes = cfg.STAGES_KV * cfg.TILE_N * cfg.TILE_K * cfg.BPE
    p_bytes = 2 * (cfg.TILE_N * cfg.N_Q // 2) * 4  # two P^T tiles of TILE_N key rows x N_Q/2 packed-half Int32 words
    mred_bytes = _D256_DECODE_MAX_RED_SLOTS * (cfg.N_Q // _D256_DECODE_COLS) * _D256_DECODE_COLS * 4  # the max-key slots, one per group per rotating slot
    red_bytes = cfg.SOFTMAX_WARPS * _D256_DECODE_COLS * 4  # the epilogue's per-warp column sums
    q_start = 0
    kv_start = _align_slab(q_start + q_bytes)
    p_start = _align_slab(kv_start + kv_bytes)
    mred_start = _align_slab(p_start + p_bytes)
    red_start = -(-(mred_start + mred_bytes) // 16) * 16
    tmem_ptr_start = -(-(red_start + red_bytes) // 16) * 16
    bars_start = -(-(tmem_ptr_start + 4) // 16) * 16
    total = bars_start + 8 * _D256_DECODE_BARRIER_WORDS
    return {
        "q_start": q_start,
        "kv_start": kv_start,
        "p_start": p_start,
        "mred_start": mred_start,
        "red_start": red_start,
        "tmem_ptr_start": tmem_ptr_start,
        "bars_start": bars_start,
        "p_end": p_start + p_bytes,
        "total": total,
        "fits_standard_carveout": total <= SMEM_STANDARD_CARVEOUT_BYTES,
        "carveout": SMEM_STANDARD_CARVEOUT_BYTES if total <= SMEM_STANDARD_CARVEOUT_BYTES else SMEM_USABLE_BYTES,
    }


def d256_decode_desc_version(cfg: CfgD256Decode) -> int:
    """The tcgen05 SMEM-descriptor version the decode tile's layout NEEDS: 1 as soon as an
    MMA-operand slab (the last one declared is P^T) starts at or past the 256 KiB window of a
    version-0 descriptor, else 0.  The kernel module reads this ONCE into its DESC_VERSION
    constant and wires it into every SmemTile (test_sm107_decode_descriptor_version_*)."""
    return 1 if d256_decode_smem_layout(cfg)["p_start"] >= TCGEN05_V0_ADDR_LIMIT else 0


def _validate_cfg_d256_decode_sm107(cfg: CfgD256Decode, flavor: str) -> None:
    """The Rubin arch facts on top of config_sm100._validate_cfg_d256_decode (which the SM100
    factory already ran on this record)."""
    smem = d256_decode_smem_layout(cfg)
    want_k = tile_k_hw(cfg.DTYPE_QKV)
    _check(
        [
            (
                cfg.TILE_K_HW == want_k,
                f"{flavor}: TILE_K_HW must be {want_k} at DTYPE_QKV={cfg.DTYPE_QKV} on Rubin (got {cfg.TILE_K_HW}); the body's "
                f"f16 idesc leaves k_dim at its default, so a wider k-step silently scrambles accumulator rows",
            ),
            (
                smem["total"] <= SMEM_USABLE_BYTES,
                f"{flavor}: SMEM {smem['total']} B (Q^T + {cfg.STAGES_KV}-slot K/V ring + 2 P^T + reduction scratch + barriers) exceeds the "
                f"{SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout at N_Q={cfg.N_Q}, STAGES_KV={cfg.STAGES_KV}: {smem}.  Overflowing it "
                f"does NOT fail the launch -- it clobbers the last buffer allocated (measured on the prefill line: O came back 50 % zeros "
                f"with an exact LSE), so this must raise here",
            ),
            (
                cfg.N_Q in _D256_DECODE_Q_TILES_SM107,
                f"{flavor}: N_Q must be one of {_D256_DECODE_Q_TILES_SM107} (the widths this line compiles); got {cfg.N_Q}",
            ),
            # The two FORM checks below are decode_d256_wide_tile_admits read on the compiled record (the route reads it on the
            # declaration); their wording is pinned by the config backstop test, their agreement with the predicate by the form-matrix test.
            (
                cfg.N_Q not in _D256_DECODE_OVERSIZED_Q_TILES or (cfg.PACK_GQA == 1 and cfg.QH_PER_KH >= _D256_DECODE_WIDE_PACK_MIN),
                f"{flavor}: the {cfg.N_Q}-column tile serves PACKED groups of at least {_D256_DECODE_WIDE_PACK_MIN} heads only (got pack_gqa="
                f"{cfg.PACK_GQA}, qh_per_kh={cfg.QH_PER_KH}): its unpacked rendering spills at ptxas's register cap for 18 warps; an unpacked"
                f" step of more than 32 tokens rides the 32-column tile or the prefill kernel",
            ),
            (
                cfg.N_Q not in _D256_DECODE_OVERSIZED_Q_TILES or not cfg.PAGED_KV or (cfg.MASK_FLAGS & (MASK_CAUSAL | MASK_SWA)) != 0,
                f"{flavor}: the {cfg.N_Q}-column tile over a PAGED cache needs the causal / window mask (the MTP step's bottom-right causal"
                f" mask; got MASK_FLAGS={cfg.MASK_FLAGS}): its padded-only paged rendering spills at ptxas's register cap for 18 warps; a"
                f" mask-free paged step of that width rides the 32-column tile in two token units",
            ),
            (
                smem["fits_standard_carveout"] or cfg.N_Q in _D256_DECODE_OVERSIZED_Q_TILES,
                f"{flavor}: SMEM {smem['total']} B no longer fits the standard 227 KiB per-CTA carveout at N_Q={cfg.N_Q}, "
                f"STAGES_KV={cfg.STAGES_KV}, and this width is not in _D256_DECODE_OVERSIZED_Q_TILES ({_D256_DECODE_OVERSIZED_Q_TILES}: the "
                f"widths whose oversized-carveout launch (L1 -> 8 kB) was verified on cc 10.7); verify the launch before admitting it",
            ),
            (
                smem["p_start"] < TCGEN05_V0_ADDR_LIMIT and smem["p_end"] <= TCGEN05_V0_ADDR_LIMIT,
                f"{flavor}: the P^T operand slab spans [{smem['p_start']}, {smem['p_end']}) B, reaching past the {TCGEN05_V0_ADDR_LIMIT} B version-0 "
                f"tcgen05 descriptor window (N_Q={cfg.N_Q}, STAGES_KV={cfg.STAGES_KV}); the kernel derives DESC_VERSION from the slab START "
                f"(d256_decode_desc_version) and the descriptor's per-k-step advances must stay inside the window too, so an operand past "
                f"the line must be validated under version 1 before the record is admitted",
            ),
        ]
    )


def make_cfg_d256_decode(params: TemplateParams) -> Tuple[CfgD256Decode, TmaIters]:
    """Config for sm107/decode_d256_f16.py: the SM100 decode record (config_sm100.make_cfg_d256_decode,
    whose predicates are arch-neutral, built here over the Rubin width set _D256_DECODE_Q_TILES_SM107 --
    the SM100 pair plus the 64-column tile) re-validated against the Rubin SMEM carveout (standard or
    oversized) and the version-0 tcgen05 descriptor window.

    Backstop only (see TemplateParams): the adapter routes decode-shaped f16/bf16 d256 graphs here
    (``_decode_q_tile``) and keeps every other graph on the prefill tile, so a ValueError is an adapter
    gap, not a user error.  ``cta_mma`` / ``sched_policy`` are accepted and unused (one cta_group::1
    CTA per unit, nothing to schedule).  Not served by this tile, declined here so a mis-routed record
    cannot trace a silently incomplete kernel: the fused epilogue gate (no gate seams on the decode
    tile; the Rubin d256 PREFILL kernel serves gated graphs), the pre-folded softmax scale (the tile
    applies the scale in-kernel), the d512 2x2 / d64 decode / ragged-Q records of other templates.
    """
    flavor = _D256_DECODE_FLAVOR
    if params.epilogue_gate:
        raise ValueError(f"{flavor}: the fused epilogue gate is not wired on the decode tile (the Rubin d256 prefill kernel serves gated graphs)")
    if params.softmax_scale_prefolded:
        raise ValueError(f"{flavor}: softmax_scale_prefolded is not wired on the decode tile (it applies the scale in-kernel)")
    if getattr(params, "mma_2x2", False) or params.decode_tile or params.ragged_q:
        raise ValueError(f"{flavor}: mma_2x2 / decode_tile / ragged_q select other templates; the d256 decode tile does not consume them")
    cfg, tma = _make_cfg_d256_decode_record(params, _D256_DECODE_Q_TILES_SM107)
    _validate_cfg_d256_decode_sm107(cfg, flavor)
    return cfg, tma
