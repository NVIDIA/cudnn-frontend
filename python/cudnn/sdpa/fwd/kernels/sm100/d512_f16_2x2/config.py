# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 wide half 2x2 configuration; native SM107 uses its own factory."""

from typing import Tuple
from cudnn.frost.tile_dsl.constants import DTYPE_FP16, DTYPE_BF16
from cudnn.sdpa.fwd.config_common import (
    TemplateParams,
    TmaIters,
    _validate_params,
    _mask_flags_from,
    _pack_g,
    _tma_iters,
    bpe,
    rescale_threshold,
)
from cudnn.sdpa.fwd.config_d512_2x2_common import CfgD512X2, d512_2x2_p_ring_start_bytes, d512_2x2_geometry_checks


def _validate_cfg_d512_2x2(cfg: CfgD512X2) -> None:
    """The SM100 2x2-datapath d512 configuration: the arch-neutral geometry checks plus the SM100 arch facts
    (227 KiB opt-in cap, every operand below the 256 KiB version-0 descriptor window)."""
    checks = d512_2x2_geometry_checks(cfg) + (
        (cfg.SMEM_CAP_BYTES == 227 * 1024, f"d512 2x2 (SM100): the budget is checked against the 227 KiB opt-in cap, got {cfg.SMEM_CAP_BYTES}"),
        (
            cfg.DESC_VERSION == 0 and d512_2x2_p_ring_start_bytes(cfg) + cfg.XFER_STAGES * cfg.TILE_M * cfg.TILE_N * cfg.BPE <= 256 * 1024,
            "d512 2x2 (SM100): every operand sits below 256 KiB -> tcgen05 SMEM descriptor version 0",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512_2x2(params: TemplateParams, *, cga_m: int = 4) -> Tuple[CfgD512X2, TmaIters]:
    """The 2x2-datapath d512 configuration (``TemplateParams.mma_2x2``).

    ``cga_m`` = 4 (two twin pairs sharing K/V by multicast, the shipped arm) or 2
    (the bring-up / bitwise-twin arm: one pair, own-bit loads).  Not a knob:
    tests load the template with the arm they want through the module constant
    the kernel file reads."""
    if not getattr(params, "mma_2x2", False):
        raise ValueError("d512 2x2: make_cfg_d512_2x2 needs a TemplateParams record with mma_2x2=True (make_cfg_d512 dispatches on it)")
    native_d576 = params.d_flavor == 576
    _validate_params("d512", params, d512_partial_pack_wired=native_d576)
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError("d512 2x2: BF16/FP16 inputs only")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    n_pairs = cga_m // CfgD512X2.CTA_MMA
    cfg = CfgD512X2(
        TILE_K=576 if native_d576 else 512,
        N_BMM1_CHUNKS=3 if native_d576 else 2,
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=bpe(dtype_o),
        CGA_M=cga_m,
        KV_SHARE=n_pairs,
        Q_SUPERS_PER_CLUSTER=cga_m,
        ROWS_PER_CLUSTER=cga_m * CfgD512X2.TILE_M,
        READ_TILE_ARRIVERS=10 * cga_m + n_pairs,
        KV_EMPTY_ARRIVERS=n_pairs,
        O_EMPTY_ARRIVERS=CfgD512X2.ONE_WARP * n_pairs,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
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
        PACK_G=_pack_g(params, CfgD512X2.TILE_M, partial=native_d576),
    )
    _validate_cfg_d512_2x2(cfg)
    return cfg, _tma_iters(cfg)
