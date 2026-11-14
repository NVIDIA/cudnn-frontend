# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared swap-AB geometry construction, with an explicit query-tile domain.

SM100 and SM107 use the same field/operand geometry. Each target's caller
supplies its admitted widths and checks its resource budget. This module
does not decide which implementation or query tile a graph should use.
"""

from dataclasses import dataclass
from typing import Tuple

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, DTYPE_E4M3, DTYPE_E5M2, MASK_NONE
from cudnn.sdpa.fwd.config_common import TemplateParams, TmaIters, bpe, _mask_flags_from, _tma_iters

_D256_DECODE_Q_TILES = (16, 32)


@dataclass(frozen=True)
class CfgD256Decode:
    # KV tokens per MMA (the M axis); TILE_K / TILE_O are the head-dim envelopes.
    TILE_N: int = 128
    TILE_K: int = 256
    TILE_O: int = 256
    # Packed Q rows per unit (the MMA N axis).
    N_Q: int = 16

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    # P^T rows are N_Q half-precision values: 32 B (N_Q = 16) or 64 B (N_Q = 32).
    P_SWZ_BYTES: int = 32
    TILE_K_HW: int = 16

    # K and V tiles share one ring of 64 KiB slots (K(t), V(t), K(t+1), ...).
    STAGES_KV: int = 3

    MASK_FLAGS: int = MASK_NONE
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

    # 4 softmax warps per 16 S^T columns (one group at N_Q = 16, two at 32),
    # then the MMA warp and the TMA warp.
    SOFTMAX_WARPS: int = 4
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TOTAL_WARPS: int = 6
    THREADS_PER_CTA: int = 6 * 32


def _validate_cfg_d256_decode(cfg: CfgD256Decode, q_tiles: Tuple[int, ...] = _D256_DECODE_Q_TILES) -> None:
    """``q_tiles`` (appended): the tile widths the CALLING arch line compiles -- the Blackwell set by default; the
    cc 10.7 factory passes its own (config_sm107._D256_DECODE_Q_TILES_SM107, which adds the 64-column tile the 320 KiB
    Rubin carveout holds).  Every other predicate is arch-neutral."""
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    checks = (
        (not fp8, "d256 decode: f16/bf16 inputs only"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d256 decode: half input requires DTYPE_O == DTYPE_QKV"),
        (cfg.N_Q in q_tiles, f"d256 decode: N_Q must be one of {q_tiles}"),
        (cfg.SOFTMAX_WARPS == 4 * (cfg.N_Q // 16), "d256 decode: 4 softmax warps per 16 Q columns"),
        (
            cfg.MMA_WARP_ID == cfg.SOFTMAX_WARPS and cfg.TMALDG_WARP_ID == cfg.SOFTMAX_WARPS + 1 and cfg.TOTAL_WARPS == cfg.SOFTMAX_WARPS + 2,
            "d256 decode: role layout is softmax warps, then the MMA warp, then the TMA warp",
        ),
        (cfg.THREADS_PER_CTA == 32 * cfg.TOTAL_WARPS, "d256 decode: THREADS_PER_CTA must be 32 * TOTAL_WARPS"),
        (cfg.TILE_N == 128, "d256 decode: the KV tile is the 128-lane TMEM layout"),
        (cfg.STAGES_KV >= 2, "d256 decode: the K/V ring needs at least K(t) and V(t) resident"),
        (
            cfg.P_SWZ_BYTES == cfg.N_Q * cfg.BPE and cfg.P_SWZ_BYTES in (32, 64, 128),
            "d256 decode: P^T row bytes must be the 32 B, 64 B or 128 B swizzle atom (N_Q x BPE)",
        ),
        (not cfg.PACK_GQA or cfg.QH_PER_KH <= cfg.N_Q, "d256 decode: a packed head group must fit the Q tile"),
        (cfg.QH_PER_KH >= 1, "d256 decode: qh_per_kh must be >= 1"),
        (cfg.SPLIT_KV >= 1, "d256 decode: split_kv must be >= 1"),
        (not cfg.PAGED_KV or cfg.SEQ_KV_LENS_PRESENT == 1, "d256 decode: paged KV requires per-batch KV lengths"),
        (
            not cfg.PAGED_KV or (cfg.PAGE_SIZE >= 8 and cfg.PAGE_SIZE % 8 == 0 and (128 % cfg.PAGE_SIZE == 0 or cfg.PAGE_SIZE % 128 == 0)),
            "d256 decode: page_size must be a positive multiple of 8 that divides the 128-row tile or is a multiple of it",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d256_decode(params: TemplateParams, q_tiles: Tuple[int, ...] = _D256_DECODE_Q_TILES) -> Tuple[CfgD256Decode, TmaIters]:
    """Config for sm100/decode_d256_f16.py from the adapter's TemplateParams.

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
