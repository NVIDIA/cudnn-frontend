# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.fwd.config_sm107.make_cfg_d256_sparse`` -- the Rubin index-list sparse d256 forward config, host-only.

The config is a set of CLAIMS about a kernel body (warp count, arriver counts, ring depths, SMEM starts, the register split);
the body cannot check them at run time, so a disagreement is undefined behaviour: an unreachable mbarrier init HANGS at every
shape, a stale ring slot is a SILENT wrong answer, an SMEM overflow clobbers the last buffer with an exact LSE.  Every
REACHABLE predicate of ``_validate_cfg_d256_sparse`` therefore gets ONE raising case here (``pytest.raises(ValueError,
match=...)`` on the failure signature its message carries), the three predicates that are unreachable by construction get a
simulated one (a misaligned layout; the whole predicate table on an oversized box), an in-module coverage check pins that the
rows reach EVERY predicate (a row pre-empted by an earlier predicate is not coverage), and the accepting record is pinned
VALUE by VALUE against the kernel header's tables -- the issuing-lane ledger (``SUM(issuing lanes) == init``), the byte-exact SMEM table, the box-bytes rule for every
``expect_tx``, the register arithmetic ``SUM(role regs x warps) == entry x warps``, the descriptor version derived from the
last K/V stage's start.

Also pinned: the four index-list ``TemplateParams`` fields are appended with inert defaults (every existing forward config
renders identically whether or not they are passed), every DENSE factory declines a record that sets them, the kernel
module's prologue takes its geometry from the config (DESC_VERSION derived, the spin constant a module literal), and
``_item_bounds`` is the kernel file's only source of ``n_tiles`` (a pin that ARMS with the body: a module with any function
definition must define the helper and bind ``n_tiles`` from a call to it).

Device-independent: nothing here compiles or launches.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
import re

import pytest

from cudnn.sdpa.fwd import config_sm100 as c100
from cudnn.sdpa.fwd import config_sm107 as c107
from cudnn.sdpa.fwd.config_sm100 import TemplateParams
from cudnn.sdpa.fwd.config_sm107 import (
    SMEM_STANDARD_CARVEOUT_BYTES,
    SMEM_USABLE_BYTES,
    SPARSE_D256_WIRED_ARMS,
    TCGEN05_V0_ADDR_LIMIT,
    CfgD256Sparse,
    _validate_cfg_d256_sparse,
    d256_sparse_smem_layout,
    make_cfg_d256_sparse,
    sparse_entry_regs,
)

pytestmark = [pytest.mark.L0]

BF16, FP16, E4M3 = 2, 3, 0
# The 24/2 geometry (12 query heads per KV head) at top_k = 512 -- the record the kernel's plain import builds.
_BASE = dict(dtype_qkv=BF16, cta_mma=1, pack_gqa=True, qh_per_kh=12, qsa_block_topk=512, qsa_block_size=4)
_KERNEL = pathlib.Path(c107.__file__).resolve().parent / "kernels" / "sm107" / "sparse_d256_f16.py"


def _sparse(**over):
    return TemplateParams(**{**_BASE, **over})


# ---------------------------------------------------------------------------- the accepting record, value by value


def test_sparse_record_pins_the_header_tables():
    """The 16-warp record at 24/2, top_k 512: every number the kernel header's tables quote."""
    cfg, tma = make_cfg_d256_sparse(_sparse())
    assert isinstance(cfg, CfgD256Sparse)
    # tile geometry
    assert (cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.N_Q, cfg.BLOCK_SIZE, cfg.BLOCKS_PER_TILE) == (128, 256, 256, 16, 4, 32)
    assert (cfg.BLOCK_TOPK, cfg.MAX_TILES_PER_ITEM) == (512, 17)
    assert (cfg.DTYPE_QKV, cfg.DTYPE_O, cfg.BPE, cfg.BPE_O, cfg.GATE_BPE) == (BF16, BF16, 2, 2, 2)
    assert (cfg.Q_SWZ_BYTES, cfg.K_SWZ_BYTES, cfg.V_SWZ_BYTES, cfg.P_SWZ_BYTES, cfg.TILE_K_HW) == (128, 128, 128, 32, 16)
    assert (cfg.STAGES_KV, cfg.STAGES_Q, cfg.STAGES_IDS, cfg.STAGES_GATE, cfg.SCHEDULER_STAGES) == (3, 2, 2, 1, 2)
    # warp map: role-homogeneous warpgroups
    assert (cfg.SOFTMAX_WARPS, cfg.GATHER_WARPS, cfg.AUX_WARPS, cfg.TOTAL_WARPS, cfg.THREADS_PER_CTA) == (4, 8, 4, 16, 512)
    assert (cfg.SOFTMAX_WARP_BASE, cfg.GATHER_WARP_BASE, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.SCHED_WARP_ID, cfg.SPARE_WARP_ID) == (0, 4, 12, 13, 14, 15)
    # the issuing-lane ledger: SUM(issuing lanes) == init per barrier
    assert (cfg.ONE_LANE, cfg.SOFTMAX_LANES) == (1, 128)
    assert cfg.KV_FULL_ARRIVERS == cfg.GATHER_WARPS == 8, "mb_kv_full: one expect_tx per gather warp"
    assert cfg.IDS_EMPTY_ARRIVERS == cfg.GATHER_WARPS + cfg.SOFTMAX_WARPS == 12, "mb_ids_empty: one elected lane per list-reading warp"
    assert cfg.READ_TILE_ARRIVERS == cfg.SOFTMAX_WARPS + 1 + 1 + cfg.GATHER_WARPS == 14, "the scheduler credit: every warp but the scheduler and the spare"
    assert (cfg.BAR_TMEM_THREADS, cfg.BAR_SOFTMAX_THREADS) == (160, 128)
    # the gather geometry: the per-warp expect_tx equals its gather4 bytes and the warps sum to the stage
    assert (cfg.BLOCKS_PER_WARP, cfg.GATHER_BOXES, cfg.GATHER_BOX_ELEMS, cfg.GATHER_ISSUE_BYTES, cfg.GATHERS_PER_WARP_PER_STAGE) == (4, 4, 64, 512, 16)
    assert cfg.KV_TX_BYTES_PER_WARP == 16 * 512 == 8192 and cfg.KV_TX_BYTES_PER_WARP * cfg.GATHER_WARPS == cfg.KV_STAGE_BYTES == 65536
    # the box-bytes rule: Q_TX / GATE_TX are the BOX (G rows x 128 B x 4 subtiles), never the 8 KiB slot
    assert (cfg.Q_BOX_TOKENS, cfg.Q_BOX_ROWS, cfg.Q_SLOT_BYTES) == (1, 12, 8192)
    assert cfg.Q_TX_BYTES == 12 * 256 * 2 == 6144 and cfg.GATE_TX_BYTES == 12 * 256 * 2 == 6144
    assert (cfg.IDS_TX_BYTES, cfg.IDS_SLOT_BYTES) == (2048, 2112)
    # TMEM
    assert (cfg.TMEM_COLS, cfg.S_ACC_OFF, cfg.O_OFF) == (64, (0, 16), (32, 48))
    # registers: the entry pool and the declared split
    assert (cfg.ENTRY_REGS, cfg.SOFTMAX_REGS, cfg.GATHER_REGS, cfg.AUX_REGS, cfg.REG_SPLIT_DECLARED) == (128, 240, 96, 80, 1)
    assert 4 * 240 + 8 * 96 + 4 * 80 == 2048 == 128 * 16
    # SMEM: the byte table at 1024-B alignment
    assert (cfg.SMEM_TOTAL_BYTES, cfg.SMEM_CARVEOUT_BYTES, cfg.DESC_VERSION) == (226688, SMEM_STANDARD_CARVEOUT_BYTES, 0)
    assert SMEM_STANDARD_CARVEOUT_BYTES - cfg.SMEM_TOTAL_BYTES == 5760
    # features of the base arm
    assert (cfg.INCLUDE_OPEN_BLOCK, cfg.EPILOGUE_GATE, cfg.PAGED_KV, cfg.PAGE_SIZE, cfg.THD_VARLEN, cfg.BOTTOM_RIGHT, cfg.SPLIT_KV, cfg.LIST_PER_SEQUENCE) == (
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
    )
    assert (cfg.PACK_GQA, cfg.QH_PER_KH, cfg.MASK_FLAGS, cfg.HAS_SINK, cfg.CTA_MMA, cfg.CGA_M, cfg.CGA_N, cfg.SCHEDULER_POLICY) == (1, 12, 0, 0, 1, 1, 1, 0)
    # the Q^T / gate TMA walk: 4 subtiles of 64 elements (one 128-B span each)
    assert (tma.QK_ITERS, tma.VO_ITERS, tma.QK_GRANU_ELEMS, tma.VO_GRANU_ELEMS) == (4, 4, 64, 64)


def test_sparse_smem_layout_is_the_header_table():
    """Declaration order = address order, descriptor-read operands first; sKV[2] at 152 KiB keeps the version-0 descriptor."""
    cfg, _ = make_cfg_d256_sparse(_sparse())
    lay = d256_sparse_smem_layout(cfg)
    assert lay["starts"] == {"sQ": 0, "sP": 16384, "sKV": 24576, "sKV_last": 155648, "sIds": 221184, "red": 225408, "misc": 226176}
    assert lay["total"] == 226688 and lay["carveout"] == SMEM_STANDARD_CARVEOUT_BYTES and lay["desc_version"] == 0
    assert all(lay["starts"][k] % 1024 == 0 for k in ("sQ", "sP", "sKV")) and lay["starts"]["sIds"] % 16 == 0
    assert lay["starts"]["sKV_last"] < TCGEN05_V0_ADDR_LIMIT
    # depth 4 and depth 5: the layout grows by one 64 KiB stage each, into the oversized carveout, and the last stage crosses the
    # version-0 window only at depth 5 (221,184 at depth 4 is still under it)
    d4 = d256_sparse_smem_layout(dataclasses.replace(cfg, STAGES_KV=4))
    assert d4["total"] == 226688 + 65536 and d4["carveout"] == SMEM_USABLE_BYTES and d4["desc_version"] == 0 and d4["starts"]["sKV_last"] == 221184
    d5 = d256_sparse_smem_layout(dataclasses.replace(cfg, STAGES_KV=5))
    assert d5["starts"]["sKV_last"] == 286720 >= TCGEN05_V0_ADDR_LIMIT and d5["desc_version"] == 1
    # ... but depth 5 is 357,760 B > the 320 KiB usable carveout: no K/V depth that FITS crosses the version-0 window, so every
    # expressible record of this kernel is DESC_VERSION 0 (the validator's version predicate is a tripwire for a layout change,
    # not an arm the body carries)
    assert d5["total"] == 357760 > SMEM_USABLE_BYTES
    assert all(d256_sparse_smem_layout(dataclasses.replace(cfg, STAGES_KV=d))["desc_version"] == 0 for d in (2, 3, 4))
    # the misc reserve covers what the body declares: tmem_ptr 16 B + 15 mbarrier arrays (29 stages, 16-B padded) + the payload ring
    n_bar_arrays = 15
    stage_counts = (2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2)
    assert len(stage_counts) == n_bar_arrays and sum(stage_counts) == 29
    padded = sum(-(-(n * 8) // 16) * 16 for n in stage_counts)
    assert padded == 272 and 16 + padded + 2 * 8 * 4 == 352 <= 512


@pytest.mark.parametrize("g, q_tx", [(6, 3072), (12, 6144), (16, 8192), (1, 512)])
def test_sparse_q_and_gate_tx_are_the_box_bytes(g, q_tx):
    """One token per item: Q_BOX_ROWS = G, and the two expect_tx values are G x 512 B -- 8192 only when G fills the 16-row tile.
    An expect_tx of the slot against a G-row delivery would never complete (a hang at every shape)."""
    cfg, _ = make_cfg_d256_sparse(_sparse(qh_per_kh=g))
    assert (cfg.Q_BOX_TOKENS, cfg.Q_BOX_ROWS, cfg.QH_PER_KH) == (1, g, g)
    assert cfg.Q_TX_BYTES == q_tx == cfg.Q_BOX_ROWS * cfg.TILE_K * cfg.BPE
    assert cfg.GATE_TX_BYTES == q_tx == cfg.Q_BOX_ROWS * cfg.TILE_O * cfg.GATE_BPE
    assert cfg.Q_TX_BYTES <= cfg.Q_SLOT_BYTES == 8192


@pytest.mark.parametrize(
    "top_k, ids_tx, slot, tiles, total", [(512, 2048, 2112, 17, 226688), (256, 1024, 1088, 9, 224640), (64, 256, 320, 3, 223104), (4, 16, 80, 1, 222624)]
)
def test_sparse_top_k_drives_every_ids_byte_count(top_k, ids_tx, slot, tiles, total):
    """CFG.BLOCK_TOPK = the record's top_k; the bulk-copy length, the expect_tx and the staging slot all follow it (a fixed
    2048-B row would over-read the next token's list at top_k < 512)."""
    cfg, _ = make_cfg_d256_sparse(_sparse(qsa_block_topk=top_k))
    assert (cfg.BLOCK_TOPK, cfg.IDS_TX_BYTES, cfg.IDS_SLOT_BYTES, cfg.MAX_TILES_PER_ITEM) == (top_k, ids_tx, slot, tiles)
    assert cfg.IDS_TX_BYTES == top_k * 4 and cfg.IDS_TX_BYTES % 16 == 0 and cfg.IDS_SLOT_BYTES % 16 == 0
    assert cfg.SMEM_TOTAL_BYTES == total and cfg.DESC_VERSION == 0


def test_sparse_fallback_config_is_twelve_warps():
    """``gather_warps=4``: the named fallback -- 12 warps, 168 entry registers, every arriver count re-derived, the split balanced
    on ITS pool (2016), the same SMEM table."""
    cfg, _ = make_cfg_d256_sparse(_sparse(), gather_warps=4)
    assert (cfg.GATHER_WARPS, cfg.TOTAL_WARPS, cfg.THREADS_PER_CTA, cfg.ENTRY_REGS) == (4, 12, 384, 168)
    assert (cfg.GATHER_WARP_BASE, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.SCHED_WARP_ID, cfg.SPARE_WARP_ID) == (4, 8, 9, 10, 11)
    assert (cfg.KV_FULL_ARRIVERS, cfg.IDS_EMPTY_ARRIVERS, cfg.READ_TILE_ARRIVERS) == (4, 8, 10)
    assert (cfg.BLOCKS_PER_WARP, cfg.GATHERS_PER_WARP_PER_STAGE, cfg.KV_TX_BYTES_PER_WARP) == (8, 32, 16384)
    assert cfg.KV_TX_BYTES_PER_WARP * cfg.GATHER_WARPS == 65536
    assert (cfg.SOFTMAX_REGS, cfg.GATHER_REGS, cfg.AUX_REGS) == (240, 96, 168) and 4 * 240 + 4 * 96 + 4 * 168 == 2016 == 168 * 12
    assert cfg.SMEM_TOTAL_BYTES == 226688 and cfg.DESC_VERSION == 0
    for bad in (2, 6, 16, 0):
        with pytest.raises(ValueError, match="gather_warps must be 8"):
            make_cfg_d256_sparse(_sparse(), gather_warps=bad)


def test_sparse_entry_regs_is_the_launch_pool():
    assert (sparse_entry_regs(16), sparse_entry_regs(12), sparse_entry_regs(8), sparse_entry_regs(6)) == (128, 168, 255, 255)


@pytest.mark.parametrize("dtype", [BF16, FP16])
def test_sparse_accepts_both_half_dtypes(dtype):
    cfg, _ = make_cfg_d256_sparse(_sparse(dtype_qkv=dtype))
    assert cfg.DTYPE_QKV == cfg.DTYPE_O == dtype and cfg.BPE == 2


def test_sparse_accepts_the_padding_and_stats_fields():
    cfg, _ = make_cfg_d256_sparse(_sparse(seq_kv_lens_present=True, seq_q_lens_present=False, stats_log2=True))
    assert (cfg.SEQ_KV_LENS_PRESENT, cfg.SEQ_Q_LENS_PRESENT, cfg.STATS_LOG2, cfg.MASK_FLAGS) == (
        1,
        0,
        1,
        0,
    ), "per-batch lengths are not a mask bit on this kernel"


# ---------------------------------------------------------------------------- the TemplateParams-level declines (RED)


@pytest.mark.parametrize(
    "over, pattern",
    [
        (dict(qsa_block_topk=0), "not an index-list record"),
        (dict(qsa_block_size=8), "qsa_block_size must be 4"),
        (dict(qsa_block_size=0), "qsa_block_size must be 4"),
        (dict(dtype_qkv=E4M3, dtype_o=E4M3), "f16/bf16 inputs only"),
        (dict(dtype_o=FP16), "dtype_o == dtype_qkv"),
        (dict(cta_mma=2), "cta_mma must be 1"),
        (dict(pack_gqa=False), "pack_gqa must be True"),
        (dict(qh_per_kh=17), "qh_per_kh must be in 1..16"),
        (dict(qh_per_kh=0), "qh_per_kh must be in 1..16"),
        (dict(window_right=0), "the selection IS the mask"),
        (dict(window_left=640), "the selection IS the mask"),
        (dict(has_sink=True), "no sink"),
        (dict(decode_q_tile=16), "other kernels' specializations"),
        (dict(mma_2x2=True), "other kernels' specializations"),
        (dict(softmax_f16=True), "other kernels' specializations"),
        (dict(sched_policy=1), "sched_policy must be NATURAL"),
        (dict(paged_kv=True, page_size=6, seq_kv_lens_present=True), "multiple of 4"),
        (dict(seq_q_lens_present=True), "SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT"),
        (dict(qsa_block_topk=2), "multiple of 4 in"),
        (dict(qsa_block_topk=516), "multiple of 4 in"),
        (dict(qsa_block_topk=510), "multiple of 4 in"),
    ],
)
def test_sparse_declines_at_the_record(over, pattern):
    with pytest.raises(ValueError, match=pattern):
        make_cfg_d256_sparse(_sparse(**over))


@pytest.mark.parametrize(
    "over, arm",
    [
        (dict(epilogue_gate=True), "epilogue_gate"),
        (dict(thd_varlen=True), "thd_varlen"),
        (dict(paged_kv=True, page_size=16, seq_kv_lens_present=True), "paged_kv"),
        (dict(split_kv=2), "split_kv"),
        (dict(qsa_list_per_sequence=True), "list_per_sequence"),
        (dict(bottom_right=True), "bottom_right"),
        (dict(qsa_include_open_block=False), "pure_list"),
        (dict(seq_q_lens_present=True, seq_kv_lens_present=True), "seq_q_lens"),
    ],
)
def test_sparse_declines_the_arms_the_body_does_not_carry(over, arm):
    """Each arm is admitted by adding its name to SPARSE_D256_WIRED_ARMS in the commit that lands its body (and flips the adapter's
    claims record); until then the config declines it by NAME.  When an arm lands, move its row to an accept test -- do not delete it."""
    if arm in SPARSE_D256_WIRED_ARMS:
        pytest.skip(f"the {arm} arm is wired now: its accept test replaces this row")
    with pytest.raises(ValueError, match=f"does not carry the .*{arm}"):
        make_cfg_d256_sparse(_sparse(**over))


# ---------------------------------------------------------------------------- the validator, predicate by predicate (RED on a replaced field)

# One row per REACHABLE predicate of _validate_cfg_d256_sparse: the replaced field(s) make THAT predicate the first to fail and the
# pattern is a fragment of ITS message.  A row whose failure an earlier predicate reports is not coverage (the coverage test below
# counts first-failing predicates, not rows); the predicates no replaced field can reach are the enumerated _TRIPWIRES.
_RED_ROWS = [
    (dict(DTYPE_QKV=E4M3, DTYPE_O=E4M3), "f16/bf16 inputs only"),
    (dict(DTYPE_O=FP16), "DTYPE_O == DTYPE_QKV"),
    (dict(GATE_BPE=1), "GATE_BPE are 2"),
    (dict(TILE_N=64), "128-lane TMEM layout"),
    (dict(BLOCKS_PER_TILE=16, TILE_N=64), "128-lane TMEM layout"),  # the TILE_N == 128 clause alone: 64 == 16 x 4 keeps the product clause true
    (dict(BLOCK_SIZE=2, BLOCKS_PER_TILE=64), "four rows of one gather4"),  # 128 == 64 x 2 keeps the tile predicate true; the block predicate fires
    (dict(TILE_K=128), "d_qk = d_v = 256 only"),
    (dict(N_Q=32), "N_Q must be 16"),
    (dict(SOFTMAX_WARPS=8), "4 softmax warps"),
    (dict(GATHER_WARPS=6, TOTAL_WARPS=14, THREADS_PER_CTA=448), "must divide across the gather warps"),
    (dict(GATHER_WARPS=2, TOTAL_WARPS=10, THREADS_PER_CTA=320), "whole warpgroups"),
    (dict(TOTAL_WARPS=15, THREADS_PER_CTA=480), "TOTAL_WARPS must be 4 \\+ GATHER_WARPS \\+ 4"),
    (dict(THREADS_PER_CTA=256), "32 x TOTAL_WARPS"),
    (dict(GATHER_WARP_BASE=8), "gather warps follow the softmax warpgroup"),
    (dict(MMA_WARP_ID=13, TMALDG_WARP_ID=12), "MMA / TMA-LDG / scheduler / spare in that order"),
    (dict(SPARE_WARP_ID=14), "in that order"),
    (dict(SOFTMAX_LANES=96), "SOFTMAX_LANES is 32 x SOFTMAX_WARPS"),
    (dict(KV_FULL_ARRIVERS=1), "one expect_tx per gather warp"),
    (dict(IDS_EMPTY_ARRIVERS=8), "one elected lane per list-reading warp"),
    (dict(READ_TILE_ARRIVERS=15), "hang at EVERY shape"),
    (dict(BAR_TMEM_THREADS=128), "named barriers count"),
    (dict(BLOCKS_PER_WARP=8), "32 / GATHER_WARPS"),
    (dict(GATHER_BOXES=2), "4 boxes of 64 elements"),
    (dict(GATHER_ISSUE_BYTES=256), "4 rows x 128 B"),
    (dict(GATHERS_PER_WARP_PER_STAGE=8), "blocks per warp x boxes per row"),
    (dict(KV_TX_BYTES_PER_WARP=65536), "per-warp expect_tx must equal its gather4 bytes"),
    (dict(KV_STAGE_BYTES=32768), "must sum to the 64 KiB stage"),
    (dict(PACK_GQA=0), "PACK_GQA must be 1"),
    (dict(Q_BOX_TOKENS=2, Q_BOX_ROWS=24), "Q_BOX_TOKENS must be 1|must fit the N_Q"),
    (dict(QH_PER_KH=17, Q_BOX_ROWS=17), "must fit the N_Q"),
    (dict(Q_BOX_ROWS=16), "Q_BOX_ROWS must be Q_BOX_TOKENS x QH_PER_KH"),
    (dict(Q_SLOT_BYTES=4096), "8 KiB"),
    (dict(Q_TX_BYTES=8192), "hang at every shape"),
    (dict(GATE_TX_BYTES=8192), "GATE_TX_BYTES must be the gate BOX's bytes"),
    (dict(Q_SWZ_BYTES=64), "SW128"),
    (dict(P_SWZ_BYTES=64), "P\\^T row bytes"),
    (dict(TILE_K_HW=32), "2-chunk is silently wrong"),
    (dict(BLOCK_TOPK=1024, IDS_TX_BYTES=4096, IDS_SLOT_BYTES=4160, MAX_TILES_PER_ITEM=33), "multiple of 4 in"),
    (dict(IDS_TX_BYTES=2056), "IDS_TX_BYTES must be BLOCK_TOPK x 4"),
    (dict(IDS_SLOT_BYTES=2048), "64-B reserve"),
    (dict(MAX_TILES_PER_ITEM=16), "ceil"),
    (dict(STAGES_KV=1), "at least K\\(t\\) and V\\(t\\)"),
    (dict(STAGES_IDS=1), "one-item-ahead prefetch"),
    (dict(SMEM_TOTAL_BYTES=1), "disagrees with the layout"),
    (dict(SMEM_CARVEOUT_BYTES=327 * 1024), "227 KiB standard carveout or the 320 KiB"),
    (dict(STAGES_KV=4), "disagrees with the layout"),  # a moved ring with a stale total
    (
        dict(STAGES_KV=4, SMEM_TOTAL_BYTES=226688 + 65536),
        "needs the 320 KiB carveout",
    ),  # the layout moved to the oversized mode but the record still claims standard
    (dict(STAGES_KV=6, SMEM_TOTAL_BYTES=226688 + 3 * 65536, SMEM_CARVEOUT_BYTES=SMEM_USABLE_BYTES), "exceeds the 320 KiB usable Rubin carveout"),
    (dict(DESC_VERSION=1), "descriptor version must be 0"),
    (dict(TMEM_COLS=128), "4 x N_Q fp32 columns"),
    (dict(O_OFF=(16, 32)), "TMEM map"),
    (dict(ENTRY_REGS=136), "ENTRY_REGS must be the launch"),
    (dict(SOFTMAX_REGS=244), "multiple of 8 in 24..256"),
    (dict(SOFTMAX_REGS=248), "parks the last INCREASE warp forever"),
    (dict(AUX_REGS=72), "parks the last INCREASE warp forever"),
    (dict(MASK_FLAGS=1), "the selection IS the mask"),
    (dict(HAS_SINK=1), "no sink"),
    (dict(CTA_MMA=2), "one cga1 CTA per SM"),
    (dict(PAGED_KV=1, PAGE_SIZE=6), "straddles two pages"),
    (dict(PAGED_KV=1, PAGE_SIZE=0), "straddles two pages"),
    (dict(SPLIT_KV=0), "split_kv must be >= 1"),
    (dict(THD_VARLEN=1), "force SEQ_KV_LENS_PRESENT=1"),
    (dict(SEQ_Q_LENS_PRESENT=1), "SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT"),
    (dict(SCHEDULER_POLICY=1), "NATURAL in v1"),
]


@pytest.mark.parametrize("bad, pattern", _RED_ROWS)
def test_sparse_validator_raises_on_each_claim(bad, pattern):
    """RED side of every predicate: a config whose claim no longer holds raises with the message naming the failure's look."""
    cfg, _ = make_cfg_d256_sparse(_sparse())
    with pytest.raises(ValueError, match=pattern):
        _validate_cfg_d256_sparse(dataclasses.replace(cfg, **bad))


def test_sparse_validator_green_on_the_record_and_on_a_consistent_depth_4():
    """GREEN side: the factory's own records pass, and a depth-4 record whose SMEM claims were re-derived (oversized carveout,
    desc version from the layout) passes too -- the levers are expressible, just not the default."""
    cfg, _ = make_cfg_d256_sparse(_sparse())
    _validate_cfg_d256_sparse(cfg)
    deep = dataclasses.replace(cfg, STAGES_KV=4)
    lay = d256_sparse_smem_layout(deep)
    _validate_cfg_d256_sparse(dataclasses.replace(deep, SMEM_TOTAL_BYTES=lay["total"], SMEM_CARVEOUT_BYTES=lay["carveout"], DESC_VERSION=lay["desc_version"]))
    assert lay["carveout"] == SMEM_USABLE_BYTES and lay["total"] == 292224


def _predicate_table(monkeypatch, cfg):
    """The validator's WHOLE predicate table on ``cfg`` as ``[(ok, message)]``: ``_check`` is captured instead of raising at the
    first failure, so a test sees which predicate reports first AND whether a later one would have fired too."""
    seen = []
    monkeypatch.setattr(c107, "_check", lambda preds: seen.extend(preds))
    _validate_cfg_d256_sparse(cfg)
    return seen


# Predicates no replaced field can make the FIRST failure -- each has a simulated RED test below instead of a row in _RED_ROWS:
_TRIPWIRES = (
    "must fit the aliased slot",  # Q_TX / GATE_TX <= Q_SLOT: implied by Q_BOX_ROWS == QH_PER_KH <= N_Q, and the box-bytes rules report first
    "must start 1024-B aligned",  # d256_sparse_smem_layout aligns every slab start by construction (_align_slab)
    "sIds must start 16-B aligned",  # idem (_align16)
)


def test_every_validator_predicate_is_reached_by_a_red_row_or_is_an_enumerated_tripwire(monkeypatch):
    """The "one RED row per predicate" claim, enforced in-module: for every row of _RED_ROWS the FIRST failing predicate's message
    matches the row's pattern, the first-failing predicates cover every predicate of the table except the enumerated tripwires, and
    every tripwire is indeed unreached (a tripwire a row reaches must leave the list).  A new predicate without a row, or a row an
    earlier predicate pre-empts, fails here -- the counts never have to be quoted by hand."""
    cfg, _ = make_cfg_d256_sparse(_sparse())
    table = _predicate_table(monkeypatch, cfg)
    assert table and all(ok for ok, _ in table), "the factory's record must be green on every predicate"
    reached = {}
    for bad, pattern in _RED_ROWS:
        row = _predicate_table(monkeypatch, dataclasses.replace(cfg, **bad))
        first = next((i for i, (ok, _) in enumerate(row) if not ok), None)
        assert first is not None, f"{bad}: no predicate fails"
        assert re.search(pattern, row[first][1]), f"{bad}: the first failing predicate says {row[first][1]!r}, the row expects {pattern!r}"
        reached.setdefault(first, []).append(bad)
    tripwires = {i for i, (_, msg) in enumerate(table) if any(t in msg for t in _TRIPWIRES)}
    assert len(tripwires) == len(_TRIPWIRES), "every enumerated tripwire names exactly one predicate"
    unreached = set(range(len(table))) - set(reached)
    assert unreached == tripwires, (
        f"predicates no RED row reaches: {[table[i][1] for i in sorted(unreached - tripwires)]}; "
        f"tripwires a row reaches (drop them from _TRIPWIRES): {[table[i][1] for i in sorted(tripwires & set(reached))]}"
    )


def _layout_with(moved_starts):
    """``d256_sparse_smem_layout`` with the named starts moved -- the only way to reach the alignment predicates, which the real
    layout satisfies by construction."""
    real = d256_sparse_smem_layout

    def moved(cfg):
        lay = real(cfg)
        return {**lay, "starts": {**lay["starts"], **moved_starts}}

    return moved


@pytest.mark.parametrize(
    "move, pattern",
    [
        (dict(sP=16384 + 512), "SW128 buffer must start 1024-B aligned .*sP 16896"),
        (dict(sKV=24576 + 128), "SW128 buffer must start 1024-B aligned .*sKV 24704"),
        (dict(sIds=221184 + 8), "sIds must start 16-B aligned .*got 221192"),
    ],
)
def test_sparse_validator_alignment_tripwires_fire_on_a_misaligned_layout(monkeypatch, move, pattern):
    """RED side of the two alignment tripwires: the layout function is replaced by one that moves a start, and the validator (which
    reads the layout through the module, not a cached copy) names the misaligned buffer and its offset."""
    cfg, _ = make_cfg_d256_sparse(_sparse())
    monkeypatch.setattr(c107, "d256_sparse_smem_layout", _layout_with(move))
    with pytest.raises(ValueError, match=pattern):
        _validate_cfg_d256_sparse(cfg)


def test_sparse_validator_box_fits_the_slot_is_implied_and_live(monkeypatch):
    """The third tripwire: Q_TX / GATE_TX <= Q_SLOT can never be the FIRST failure (Q_BOX_ROWS == QH_PER_KH <= N_Q pins the box at or
    under the 8 KiB slot, and an oversized Q_TX_BYTES is reported by the box-bytes rule first).  Pinned two ways: the implication over
    the whole admitted domain (every G in 1..16, both half dtypes; the slot is exactly full at G = 16), and the predicate itself reading
    False on an oversized box in the whole table while the box-bytes rule reports -- live code, not dead."""
    for dtype in (BF16, FP16):
        for g in range(1, 17):
            cfg, _ = make_cfg_d256_sparse(_sparse(dtype_qkv=dtype, qh_per_kh=g))
            assert cfg.Q_TX_BYTES == cfg.GATE_TX_BYTES == g * 512 <= cfg.Q_SLOT_BYTES == 8192
    cfg, _ = make_cfg_d256_sparse(_sparse())
    table = _predicate_table(monkeypatch, dataclasses.replace(cfg, Q_TX_BYTES=2 * cfg.Q_SLOT_BYTES))
    assert [ok for ok, msg in table if "must fit the aliased slot" in msg] == [False]
    assert "hang at every shape" in next(msg for ok, msg in table if not ok), "the box-bytes rule reports first"


# ---------------------------------------------------------------------------- TemplateParams: appended, inert, declined by the dense factories


def test_index_list_fields_are_appended_with_inert_defaults():
    names = [f.name for f in dataclasses.fields(TemplateParams)]
    assert names[-4:] == [
        "qsa_block_topk",
        "qsa_block_size",
        "qsa_include_open_block",
        "qsa_list_per_sequence",
    ], "append-only: the four index-list fields are the LAST four"
    assert TemplateParams() == TemplateParams(qsa_block_topk=0, qsa_block_size=0, qsa_include_open_block=True, qsa_list_per_sequence=False)
    d = TemplateParams()
    assert (d.qsa_block_topk, d.qsa_block_size, d.qsa_include_open_block, d.qsa_list_per_sequence) == (0, 0, True, False)


_DENSE_CASES = [
    ("sm107.d128", c107.make_cfg_d128, {}),
    ("sm107.d128.mxfp8", c107.make_cfg_d128_mxfp8, dict(dtype_qkv=E4M3, dtype_o=BF16)),
    ("sm107.d192", c107.make_cfg_d192, {}),
    ("sm107.d256", c107.make_cfg_d256, {}),
    ("sm107.d256.gate", c107.make_cfg_d256, dict(epilogue_gate=True)),
    ("sm107.d256.mxfp8", c107.make_cfg_d256_mxfp8, dict(dtype_qkv=E4M3, dtype_o=BF16, cta_mma=1)),
    ("sm107.d512", c107.make_cfg_d512, {}),
    ("sm107.d512.2x2", c107.make_cfg_d512_2x2, dict(mma_2x2=True)),
    ("sm100.d128", c100.make_cfg_d128, {}),
    ("sm100.d64", c100.make_cfg_d64, dict(d_flavor=64, cta_mma=1)),
    ("sm100.d192", c100.make_cfg_d192, {}),
    ("sm100.d256", c100.make_cfg_d256, {}),
    ("sm100.d256.decode", c100.make_cfg_d256_decode, dict(decode_q_tile=16, seq_kv_lens_present=True, pack_gqa=True, qh_per_kh=12)),
    ("sm100.d512", c100.make_cfg_d512, {}),
    ("sm100.d512.2x2", c100.make_cfg_d512_2x2, dict(mma_2x2=True)),
]


@pytest.mark.parametrize("name, factory, kw", _DENSE_CASES, ids=[c[0] for c in _DENSE_CASES])
def test_every_dense_config_is_unchanged_by_the_inert_defaults(name, factory, kw):
    """Every existing forward factory renders the SAME record whether the four fields are left to their defaults or passed
    explicitly at them: the append is invisible to the dense line (the base-vs-branch value snapshot is kept beside the
    results; this is its in-tree twin)."""
    plain = factory(TemplateParams(**kw))
    explicit = factory(TemplateParams(**kw, qsa_block_topk=0, qsa_block_size=0, qsa_include_open_block=True, qsa_list_per_sequence=False))
    assert plain == explicit


@pytest.mark.parametrize("name, factory, kw", _DENSE_CASES, ids=[c[0] for c in _DENSE_CASES])
def test_every_dense_config_declines_an_index_list_record(name, factory, kw):
    """A dense template must never consume an index-list record -- it would read K/V densely and silently ignore the list."""
    with pytest.raises(ValueError, match="select the index-list sparse kernel"):
        factory(TemplateParams(**kw, qsa_block_topk=512, qsa_block_size=4))
    with pytest.raises(ValueError, match="select the index-list sparse kernel"):
        factory(TemplateParams(**kw, qsa_block_size=4))


# ---------------------------------------------------------------------------- the kernel file: prologue pins and the one-helper invariant


def test_sparse_kernel_prologue_takes_its_geometry_from_the_config():
    """The module's CFG comes from make_cfg_d256_sparse, DESC_VERSION is the config's derived value (never a literal) and the ring
    spin constant is a module literal -- the conventions every Rubin kernel's source pins rely on."""
    src = _KERNEL.read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "CFG, _TMA = make_cfg_d256_sparse(PARAMS)" in code
    assert "DESC_VERSION: int = CFG.DESC_VERSION" in code
    assert "SPIN_RING_WAITS: bool = False" in code
    assert "desc_version=0" not in code and "desc_version=1" not in code, "a re-literalled descriptor version"
    try:
        import cudnn.sdpa.fwd.kernels.sm107.sparse_d256_f16 as kern
    except NotImplementedError as exc:
        # The body's Rule-7 gate: a DSL without the sm_107a target declines by name at import -- a skip here, never a failure.
        pytest.skip(f"the sparse kernel module declines this DSL: {exc}")

    assert kern.DESC_VERSION == kern.CFG.DESC_VERSION == 0 and kern.CFG.TOTAL_WARPS == 16 and kern.Cfg is CfgD256Sparse


def _n_tiles_bindings(tree: ast.AST):
    """Every binding of a name ``n_tiles`` in the module: (line, inside a def named _item_bounds?, bound from a call to _item_bounds?)."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                names = []
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name) and n.id == "n_tiles":
                            names.append(n)
                if not names:
                    continue
                value = sub.value
                from_helper = isinstance(value, ast.Call) and getattr(value.func, "id", None) == "_item_bounds"
                out.append((sub.lineno, node.name == "_item_bounds", from_helper))
    return out


def test_item_bounds_is_the_only_n_tiles_source_in_the_kernel():
    """``n_tiles`` (with ``count`` / ``has_open`` / ``dead``) comes from ONE pure helper called by every role over the same inputs; a
    second source (a count word staged through SMEM, a role-local recomputation) is a disagreement between roles = an unreachable or
    over-run barrier.  Every binding of ``n_tiles`` in the file is either inside ``_item_bounds`` itself or unpacked from a call to it
    -- and the pin ARMS with the body: a module that defines any function must define ``_item_bounds`` and bind ``n_tiles`` from a
    call to it at least once (a body that never calls the helper, or names the variable differently, fails); the header-only module
    is asserted as such (no function, no binding), never read as a vacuous pass."""
    src = _KERNEL.read_text()
    assert "_item_bounds" in src, "the kernel header names the one helper"
    tree = ast.parse(src)
    bindings = _n_tiles_bindings(tree)
    offenders = [(ln, in_helper, from_helper) for ln, in_helper, from_helper in bindings if not (in_helper or from_helper)]
    assert not offenders, f"n_tiles bound outside _item_bounds and not from a call to it: {offenders}"
    defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if defs:
        assert "_item_bounds" in defs, f"the body defines {len(defs)} functions but no _item_bounds"
        assert any(from_helper for _, _, from_helper in bindings), "no role binds n_tiles from a call to _item_bounds"
    else:
        assert not bindings, "a header-only module binds n_tiles"
    # the detector itself is live: a synthetic module with a role-local recomputation is caught
    probe = ast.parse("def _mma_warp_group(hi, lo):\n    n_tiles = hi - lo\n    return n_tiles\n")
    assert any(not (a or b) for _, a, b in _n_tiles_bindings(probe))
    ok = ast.parse("def _item_bounds(p):\n    n_tiles = 1\n    return n_tiles\n\ndef _mma(p):\n    lo, n_tiles, dead = _item_bounds(p)\n    return n_tiles\n")
    assert all(a or b for _, a, b in _n_tiles_bindings(ok))


def test_sparse_factory_signature_is_append_only():
    """``make_cfg_d256_sparse(params, *, gather_warps=8)``: the positional record first, the warp-population switch keyword-only."""
    sig = inspect.signature(make_cfg_d256_sparse)
    params = list(sig.parameters.values())
    assert params[0].name == "params" and params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[1].name == "gather_warps" and params[1].kind is inspect.Parameter.KEYWORD_ONLY and params[1].default == 8
