# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent record and factory for per-tensor FP8 swap_ab.

Legacy field defaults are retained; no half/MXFP8 or sibling factory is imported.
"""

from dataclasses import dataclass, replace
from typing import Tuple
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, DTYPE_FP16
from cudnn.sdpa.fwd.config_common import TemplateParams, TmaIters, _mask_flags_from, _tma_iters, _validate_params, bpe


@dataclass(frozen=True)
class CfgD128Fp8Decode:
    TILE_N: int = 128
    TILE_K: int = 128
    TILE_O: int = 128
    N_Q: int = 16
    DTYPE_QKV: int = 0
    DTYPE_O: int = 0
    BPE: int = 1
    BPE_O: int = 1
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    P_SWZ_BYTES: int = 32
    TILE_K_HW: int = 32
    STAGES_KV: int = 5
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
    THD_VARLEN: int = 0


def make_cfg_d128_fp8_decode(params: TemplateParams) -> Tuple[CfgD128Fp8Decode, TmaIters]:
    """Native per-tensor FP8 dense or paged decode, retaining common graph-semantic checks."""
    # decode_q_tile selects this body instead of the prefill body. Validate
    # the remaining public semantics with the common D128 predicates.
    _validate_params("d128", replace(params, decode_q_tile=0))
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    checks = (
        (params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2), "FP8 inputs required"),
        (dtype_o in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16), "FP8 or half output required"),
        (params.cta_mma == 1, "one CTA required"),
        (not params.thd_varlen and not params.ragged_q, "THD queries use the prefill tile"),
        (not params.pv_bf16 and not params.softmax_f16, "FP8 P and FP32 softmax required"),
        (not params.dual_kv_stream, "dual_kv_stream selects the prefill pipeline"),
        (params.decode_q_tile == 16, "a 16-column MMA tile is required"),
        (getattr(params, "decode_head_dim", 0) == 128, "a physical D128 envelope is required"),
        (getattr(params, "decode_live_columns", 0) == 8, "eight live softmax columns are required"),
        (not params.pack_gqa or params.qh_per_kh <= 8, "the full packed head group must fit eight columns"),
    )
    for ok, message in checks:
        if not ok:
            raise ValueError(f"d128 FP8 decode: {message}")
    cfg = CfgD128Fp8Decode(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE_O=bpe(dtype_o),
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
    return cfg, _tma_iters(cfg)
