# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Decode-shaped SDPA forward for the SM100 d256 f16/bf16 flavor: the swap-AB tile.

Why a second template.  The prefill tile (sm100/prefill_d256_f16.py) issues
BMM1 / BMM2 over a 256-row collective Q tile per 128-key KV tile whatever the
number of LIVE Q rows, and its softmax warpgroup exponentiates the full
128 x 128 score tile.  A decode step of a 32/2-head d256 model has 16 live
rows per (batch, KV head), so 15/16 of that work is spent on TMA-OOB zero rows
and the kernel runs at half the HBM rate (67 us for b=32, s_kv=4096 on a B200;
the KV bytes alone take 34 us at 8 TB/s).  Shrinking the Q tile does not help
on this hardware: a 64-row UMMA runs at half rate per SM, so the MMA time per
key is unchanged, and the 64-row TMEM layouts scatter the rows across the
sub-partitions.

This tile transposes the problem instead (the "swap-AB" decode form):

    S^T[128 keys, N_Q] = K[128 keys, d_qk] . Q^T[d_qk, N_Q]      (BMM1, M = keys)
    O^T[d_v, N_Q]     += V^T[d_v, 128 keys] . P^T[128 keys, N_Q]  (BMM2, M = d_v)

The KV tokens are the MMA M axis (M = 128, the standard TMEM lane == row
layout) and the packed Q rows are the N axis (N_Q = 16 or 32, a multiple of
the f16 UMMA N granule), so the tensor-core and exp work scale with the live
rows: 8x less MMA and 8x fewer exponentials per key than the prefill tile at
N_Q = 16.  The online softmax reduces over LANES (a column of S^T is one Q
row's scores for the tile's 128 keys): a 5-step butterfly inside each warp
and a 4-way exchange through shared memory across a 4-warp group, once per KV
tile.  A group owns 16 columns; N_Q = 32 runs two groups (8 softmax warps)
over the same 128 TMEM lanes, so the per-thread exp / shuffle / register load
is the same at either tile width (one 4-warp group over 32 columns was
issue-bound and lost to the prefill tile on the S_q = 2 MTP shape).
P^T is stored to a small swizzled SMEM tile (the B operand of BMM2 is
MN-major: the N_Q values of one key are contiguous), V is consumed in place
as an MN-major A operand (d_v contiguous), and O^T accumulates in TMEM with
lane = d_v (two 128-lane blocks for d_v = 256).

Roles (6 warps at N_Q = 16, 10 at N_Q = 32; cta_group::1, one CTA per
(KV-head group, batch, split) unit; W = 4 * N_Q / 16 softmax warps):

  * warps 0..W-1  softmax + epilogue, 4 warps per 16-column group: thread ==
                  key lane (tidx % 128) for S^T / P^T, == d lane for the O^T
                  rescale and the final store;
  * warp W        MMA issue + TMEM alloc/dealloc;
  * warp W+1      TMA loads: Q^T once, then K/V through a 3-slot 64 KiB ring
                  (K(t), V(t), K(t+1), ...), block-table indirection under PAGED_KV.

SMEM: Q^T N_Q x 512 B (8 / 16 KiB) + 3 x 64 KiB ring + 2 x P^T (4 / 8 KiB) =
208 / 224 KiB at N_Q = 16 / 32.  TMEM: 2 S^T slots + 2 O^T blocks = 4 * N_Q
columns.  The KV loop, per-batch lengths, masks (padding / bottom-right causal
/ SWA / right band), sink, Stats (natural or base-2), dense padded-Q trim and
split-KV partials follow the prefill tile's contract exactly: a split writes
fp32 O / natural-log LSE partials into the split-major workspace that
sm100/split_combine.py reduces, and an empty range yields O := 0 / LSE := -inf.
Nothing reaches the host (python/cudnn/AGENTS.md Rule 3): lengths, page
tables and the split chunking are all resolved on device.

Not served here (the adapter keeps these on the prefill tile): THD queries,
S_q * pack_g > 32 rows, FP8/MXFP8.
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256_decode

# The per-graph params are injected as a module global by the loader
# (api._load_kernel_module) before this body executes; a plain import gets a
# padded fp16 decode config for N_Q = 16.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(decode_q_tile=16, seq_kv_lens_present=True))
CFG, _TMA = make_cfg_d256_decode(PARAMS)

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.tma import tma_load_tile
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_PADDED, MASK_SWA, compute_kv_loop_bounds
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.sdpa.fwd.kernels._common_blackwell import row_max_for_exp2

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
else:
    STORAGE_DTYPE = cutlass.Float16
OUT_STORAGE_DTYPE = STORAGE_DTYPE
MMA_KIND = nvvm.Tcgen05MMAKind.F16
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1

N_Q = CFG.N_Q
TILE_N = CFG.TILE_N
TILE_K = CFG.TILE_K
TILE_O = CFG.TILE_O
BPE = CFG.BPE
STAGES = CFG.STAGES_KV
SPLIT_KV = CFG.SPLIT_KV
_FP32_PARTIALS = SPLIT_KV > 1

# === PackGQA === row r of the Q tile is token r // G, head r % G (the prefill
# tile's convention).  The Q TMA box covers Q_BOX_TOKENS tokens x G heads; a
# group that does not divide N_Q leaves N_Q - Q_BOX_ROWS tail rows, zero-filled
# once at kernel start (their S^T / O^T columns are computed and never stored).
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
Q_BOX_TOKENS = N_Q // HEADS_PER_TILE
Q_BOX_ROWS = Q_BOX_TOKENS * HEADS_PER_TILE
if Q_BOX_TOKENS < 1:
    raise ValueError(f"decode_d256_f16: the packed head group ({HEADS_PER_TILE}) does not fit the {N_Q}-row Q tile")

# === Paged KV (same contract as prefill_d256_f16.py) ===
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
KV_BOX_ROWS = min(PAGE_SIZE, TILE_N) if PAGED_KV else TILE_N
KV_BOXES = TILE_N // KV_BOX_ROWS

# TMA granule: one 128 B swizzle atom of the head dim (64 half elements).
GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
QK_ITERS = _TMA.QK_ITERS
VO_ITERS = _TMA.VO_ITERS
if _TMA.VO_GRANU_ELEMS != GRANU_ELEMS:
    raise ValueError("decode_d256_f16: Q/K and V/O must share the 128 B swizzle granule")

qBufferElems = N_Q * TILE_K
kvBufferElems = TILE_N * TILE_K
# P^T is stored as packed half pairs: N_Q / 2 words per key row.
pBufferWords = TILE_N * N_Q // 2

Q_TX_BYTES = Q_BOX_ROWS * TILE_K * BPE
KV_TX_BYTES = kvBufferElems * BPE

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_BITS = {128: 3, 64: 2, 32: 1}
SMEM_LAYOUT_QK = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_P = _SWZ_ENUM[CFG.P_SWZ_BYTES]
# The generic-proxy P^T stores apply the same XOR pattern the UMMA descriptor
# decodes (Swizzle<B, 4, 3>: 16 B chunk index ^= 128 B row-group index).
_P_SMEM_SWIZZLE = cutlass.Swizzle(_SWZ_BITS[CFG.P_SWZ_BYTES], 4, 3)
P_ROW_WORDS = N_Q // 2

# K-major operands (K as A of BMM1, Q^T as B of BMM1): SBO = 8 rows x 128 B.
STRIDE_BYTE_OFFSET_QK = 8 * CFG.K_SWZ_BYTES
# V^T as the MN-major A of BMM2: 128 d columns span two 64-wide swizzle atoms,
# LBO steps between them (TILE_N rows x 128 B each); SBO = 8 key rows.
LEADING_BYTE_OFFSET_VT = TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_VT = 8 * CFG.V_SWZ_BYTES
# P^T as the MN-major B of BMM2: one atom wide (N_Q * 2 B), SBO = 8 key rows.
STRIDE_BYTE_OFFSET_P = 8 * CFG.P_SWZ_BYTES

# TMEM: two S^T slots then the two O^T d-blocks, N_Q fp32 columns each.
S_ACC_OFF = (0, N_Q)
O_OFF = (2 * N_Q, 3 * N_Q)
TMEM_COLS = 32
while TMEM_COLS < 4 * N_Q:
    TMEM_COLS *= 2
O_BLOCKS = TILE_O // 128

# Softmax column groups: a 4-warp group owns COLS = 16 S^T / O^T columns (the
# f16 UMMA N granule), so N_Q = 32 runs two groups that read the same 128 TMEM
# lanes (tcgen05.ld lane access is by warp_id % 4) and the per-thread work is
# independent of the tile width.  Group g = tidx // 128 owns columns
# [16 g, 16 g + 16) and the matching P^T words / O^T columns / LSE rows.
COLS = 16
COL_GROUPS = N_Q // COLS
# This group's slice of a P^T key row: COLS half values = COLS / 2 words.
P_GROUP_WORDS = COLS // 2
P_CHUNKS_PER_GROUP = (COLS * BPE) // 16
SOFTMAX_WARPS = CFG.SOFTMAX_WARPS
SOFTMAX_LANES = SOFTMAX_WARPS * 32
if N_Q % COLS != 0 or SOFTMAX_WARPS != 4 * COL_GROUPS:
    raise ValueError(f"decode_d256_f16: N_Q={N_Q} needs {4 * (N_Q // COLS)} softmax warps (4 per 16 columns); config says {SOFTMAX_WARPS}")
MMA_WARP_ID = CFG.MMA_WARP_ID
TMALDG_WARP_ID = CFG.TMALDG_WARP_ID
# Named barriers: 1 = TMEM base hand-off (softmax warps + MMA warp), 2 = the
# softmax warps' own cross-warp reductions (all groups, they run in lockstep).
_BAR_TMEM = 1
_BAR_SOFTMAX = 2
# Cross-warp reduction scratch: [2 tile parities + 1 epilogue][softmax warp][COLS];
# a group reads the 4 entries of its own warps.
RED_SLOTS = 3
RED_WORDS = RED_SLOTS * SOFTMAX_WARPS * COLS

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


class DecodeBars(NamedTuple):
    mb_q_full: object
    mb_kv_full: object
    mb_kv_empty: object
    mb_s_full: object
    mb_s_empty: object
    mb_p_full: object
    mb_bmm2_done: object
    mb_tmem_dealloc: object


def make_decode_bars() -> DecodeBars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return DecodeBars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=1, producer=Producer.TMA_LOAD),
        mb_kv_full=MBarrier(_alloc(STAGES), stages=STAGES, init_count=1, producer=Producer.TMA_LOAD),
        mb_kv_empty=MBarrier(_alloc(STAGES), stages=STAGES, init_count=1, producer=Producer.MMA_COMMIT),
        mb_s_full=MBarrier(_alloc(2), stages=2, init_count=1, producer=Producer.MMA_COMMIT),
        mb_s_empty=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_p_full=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_bmm2_done=MBarrier(_alloc(2), stages=2, init_count=1, producer=Producer.MMA_COMMIT),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
    )


@cute.jit
def _select_i32(cond, a, b):
    return cutlass.Int32(arith.select(cond.ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()))


@cute.jit
def _select_f32(cond, a, b):
    return cutlass.Float32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv):
    """This unit's KV-tile range ``[lo, hi)`` plus the per-batch lengths.

    Every role computes it identically (pure function of the same inputs), so
    the barrier handshakes stay in lockstep.  The masks narrow the range the
    way the prefill tile does (compute_kv_loop_bounds over the tile's token
    span), the dense padded-Q trim collapses a dead batch to an empty range,
    and the split chunking mirrors _common_blackwell's _split_chunk.
    """
    eff_seqlen_kv = seqlen_kv
    if cutlass.const_expr(CFG.SEQ_KV_LENS_PRESENT == 1):
        eff_seqlen_kv = cutlass.Int32(cutlass.make_array_view(seq_kv_lens_tensor)[batch_idx])
    q_len_b = seqlen_q
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1):
        _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
        q_len_b = cutlass.Int32(_sq_arr[batch_idx])
    eff_seqlen_q = seqlen_q
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1 and CFG.BOTTOM_RIGHT == 1):
        eff_seqlen_q = cute.math.max(cutlass.Int32(0), cute.math.min(q_len_b, seqlen_q))
    bounds = compute_kv_loop_bounds(
        cutlass.Int32(0),
        eff_seqlen_q,
        eff_seqlen_kv,
        CFG.WINDOW_LEFT,
        CFG.MASK_FLAGS,
        TILE_N,
        Q_BOX_TOKENS,
        bottom_right=bool(CFG.BOTTOM_RIGHT),
        window_right=int(CFG.WINDOW_RIGHT),
    )
    left = bounds.left
    right = cute.math.max(bounds.right, left)
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1):
        right = _select_i32(q_len_b <= cutlass.Int32(0), left, right)
    n_tiles = right - left
    per = n_tiles // cutlass.Int32(SPLIT_KV)
    rem = n_tiles % cutlass.Int32(SPLIT_KV)
    lo = left + split_idx * per + cute.math.min(split_idx, rem)
    hi = lo + per + _select_i32(split_idx < rem, 1, 0)
    return lo, hi, eff_seqlen_kv, eff_seqlen_q, q_len_b


@cute.jit
def _paged_load_tile(
    smem_tile,
    tma,
    block_table_tensor,
    batch_idx,
    kv_head_idx,
    n_pages_b,
    kv_tile,
    mbar,
    box_rows: cutlass.Constexpr[int],
    n_boxes: cutlass.Constexpr[int],
    granu_elems: cutlass.Constexpr[int],
):
    """One paged K or V tile as ``n_boxes`` row boxes through the block table
    (same contract as prefill_d256_f16._paged_load_tile: boxes past the batch's
    live pages take page -1 = TMA-OOB zeros, bytes still credited)."""
    bt = cutlass.make_array_view(block_table_tensor)
    last_live = cute.math.max(n_pages_b - cutlass.Int32(1), cutlass.Int32(0))
    for j in cutlass.range_constexpr(n_boxes):
        g = kv_tile * cutlass.Int32(TILE_N) + cutlass.Int32(j * box_rows)
        slot = g // cutlass.Int32(PAGE_SIZE)
        row_in_page = g % cutlass.Int32(PAGE_SIZE)
        page_live = cutlass.Int32(bt[batch_idx, cute.math.min(slot, last_live)])
        in_range = slot < n_pages_b
        page = _select_i32(in_range, page_live, cutlass.Int32(-1))
        tma_load_tile(
            smem_tile.shifted(j * box_rows * granu_elems),
            tma(cutlass.Int32(0), kv_head_idx, row_in_page, page),
            mbar,
            cta_group=1,
        )


@cute.jit
def _load_kv(smem_tile, tma, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, mbar):
    """One K or V tile (TILE_N keys x TILE_K) into ``smem_tile``: dense TMA box or
    the paged block-table walk."""
    if cutlass.const_expr(PAGED_KV):
        _paged_load_tile(smem_tile, tma, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, mbar, KV_BOX_ROWS, KV_BOXES, GRANU_ELEMS)
    else:
        tma_load_tile(smem_tile, tma(cutlass.Int32(0), kv_head_idx, kv_tile * cutlass.Int32(TILE_N), batch_idx), mbar, cta_group=1)


@cute.jit
def _kv_slot(seq, n_tiles):
    """Ring slot and full-barrier parity of the ``seq``-th load of the unit's
    K(0), K(1), V(0), K(2), V(1), ... sequence (V(n-1) is the 2n-1-th load: no
    K(n) precedes it)."""
    slot = seq % cutlass.Int32(STAGES)
    phase = (seq // cutlass.Int32(STAGES)) & cutlass.Int32(1)
    return slot, phase


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sKV,
    bars,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    head_idx,
    batch_idx,
    split_idx,
    block_table_tensor=None,
    block_table_v_tensor=None,
    paged_hnd: cutlass.Constexpr[bool] = False,
):
    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(PAGED_KV and paged_hnd):
        # HND page pools: descriptor dims are (D, row, H_kv, page); every load
        # site keeps the (d, head, row, page) vocabulary, the swap lives here.
        _tk, _tv = GmemTileTma(tma_k_desc), GmemTileTma(tma_v_desc)
        tma_k = lambda d, h, r, p: _tk(d, r, h, p)  # noqa: E731
        tma_v = lambda d, h, r, p: _tv(d, r, h, p)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    lo, hi, eff_seqlen_kv, _eq, _ql = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv)
    n_tiles = hi - lo
    q_head_base = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // cutlass.Int32(CFG.QH_PER_KH)
    n_pages_b = cutlass.Int32(0)
    if cutlass.const_expr(PAGED_KV):
        n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

    if n_tiles > cutlass.Int32(0):
        bars.mb_q_full.arrive(n_bytes=Q_TX_BYTES, pred=nvvm.elect_sync())
        tma_load_tile(sQ[0], tma_q(cutlass.Int32(0), q_head_base, cutlass.Int32(0), batch_idx), bars.mb_q_full.smem_ptr, cta_group=1)

        # Ring order K(0), K(1), V(0), K(2), V(1), ...: K runs one tile ahead of V, so
        # K(t+2) is issued when V(t-1)'s slot frees (end of iteration t-1) and lands
        # a whole iteration before BMM1(t+2) needs it, while V(t+1) is issued when
        # K(t+1)'s slot frees (start of iteration t).  The MMA warp consumes in the
        # same order, so one 3-slot ring serves both operands (see _kv_slot).
        kv_state = PipelineState.start(phase=1)
        bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
        bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
        _load_kv(sKV[kv_state.idx], tma_k, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, lo, bars.mb_kv_full[kv_state.idx].smem_ptr)
        kv_state = advance(kv_state, STAGES)
        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            kv_tile = lo + i
            if i + cutlass.Int32(1) < n_tiles:
                bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
                bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
                _load_kv(
                    sKV[kv_state.idx],
                    tma_k,
                    block_table_tensor,
                    batch_idx,
                    kv_head_idx,
                    n_pages_b,
                    kv_tile + cutlass.Int32(1),
                    bars.mb_kv_full[kv_state.idx].smem_ptr,
                )
                kv_state = advance(kv_state, STAGES)
            bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
            _load_kv(sKV[kv_state.idx], tma_v, block_table_v_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, bars.mb_kv_full[kv_state.idx].smem_ptr)
            kv_state = advance(kv_state, STAGES)


@cute.jit
def _mma_warp_group(
    sQ,
    sK,
    sVt,
    sP,
    tmem_ptr_i32,
    bars,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    batch_idx,
    split_idx,
):
    tmem_alloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(_BAR_TMEM, 32 * (SOFTMAX_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=TILE_N,
        k_dim=0,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=128,
        a_major=1,
        b_major=1,
        k_dim=0,
    )
    # BMM1: S^T = K . Q^T -- K [128 keys x TILE_K] K-major, Q^T [N_Q x TILE_K] K-major.
    bmm1_desc = MmaDesc(
        M=TILE_N,
        N=N_Q,
        K=TILE_K,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2: O^T = V^T . P^T -- V^T [128 d x 128 keys] MN-major (d contiguous),
    # P^T [N_Q x 128 keys] MN-major (the N_Q values of one key contiguous).
    bmm2_desc = MmaDesc(
        M=128,
        N=N_Q,
        K=TILE_N,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        atranspose=True,
        btranspose=True,
        cta_group=1,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    lo, hi, _ek, _eq, _ql = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv)
    n_tiles = hi - lo

    # Barrier bookkeeping is arithmetic in the tile index (no loop-carried
    # pipeline state inside a data-dependent branch):
    #   ring load seq (K(0), K(1), V(0), K(2), V(1), ...): slot seq % STAGES, phase (seq // STAGES) & 1;
    #   S slot t % 2: the softmax releases it once per tile that used it, so
    #   BMM1(t) waits completion #(t // 2) of s_empty[t % 2], i.e. parity
    #   ((t // 2) - 1) & 1 -- which is 1 for t < 2 and passes a fresh barrier;
    #   p_full[t % 2] / bmm2_done[t % 2]: parity (t // 2) & 1.
    if n_tiles > cutlass.Int32(0):
        bars.mb_q_full.wait(cutlass.Int32(0))
        desc_Q = sQ[0].desc()

        # Prologue: BMM1(0).
        slot0, phase0 = _kv_slot(cutlass.Int32(0), n_tiles)
        bars.mb_kv_full[slot0].wait(phase0)
        bars.mb_s_empty[0].wait(cutlass.Int32(1))
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(bmm1_desc, sK[slot0].desc(), desc_Q, tmem_raw.subview(cutlass.Int32(S_ACC_OFF[0])))
        elect_p = nvvm.elect_sync()
        bars.mb_s_full[0].arrive(cta_group=1, pred=elect_p)
        bars.mb_kv_empty[slot0].arrive(cta_group=1, pred=elect_p)

        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            # BMM1(i + 1) ahead of BMM2(i): the softmax of tile i overlaps the next
            # score tile, and the K slot is released as early as possible.
            if i + cutlass.Int32(1) < n_tiles:
                t_next = i + cutlass.Int32(1)
                slot_k, phase_k = _kv_slot(cutlass.Int32(2) * t_next - cutlass.Int32(1), n_tiles)
                par_next = t_next & cutlass.Int32(1)
                phase_s_empty = ((t_next // cutlass.Int32(2)) - cutlass.Int32(1)) & cutlass.Int32(1)
                bars.mb_kv_full[slot_k].wait(phase_k)
                bars.mb_s_empty[par_next].wait(phase_s_empty)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                s_off_next = _select_i32(par_next == cutlass.Int32(0), S_ACC_OFF[0], S_ACC_OFF[1])
                mma_ss(bmm1_desc, sK[slot_k].desc(), desc_Q, tmem_raw.subview(s_off_next))
                elect_p1 = nvvm.elect_sync()
                bars.mb_s_full[par_next].arrive(cta_group=1, pred=elect_p1)
                bars.mb_kv_empty[slot_k].arrive(cta_group=1, pred=elect_p1)

            seq_v = _select_i32(i + cutlass.Int32(1) < n_tiles, cutlass.Int32(2) * i + cutlass.Int32(2), cutlass.Int32(2) * n_tiles - cutlass.Int32(1))
            slot_v, phase_v = _kv_slot(seq_v, n_tiles)
            par_cur = i & cutlass.Int32(1)
            phase_p = (i // cutlass.Int32(2)) & cutlass.Int32(1)
            bars.mb_kv_full[slot_v].wait(phase_v)
            bars.mb_p_full[par_cur].wait(phase_p)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            desc_P = sP[par_cur].desc()
            accum = cutlass.Boolean(i > cutlass.Int32(0))
            for blk in cutlass.range_constexpr(O_BLOCKS):
                mma_ss(
                    bmm2_desc,
                    sVt[slot_v].shifted(blk * 2 * TILE_N * GRANU_ELEMS).desc(),
                    desc_P,
                    tmem_raw.subview(cutlass.Int32(O_OFF[blk])),
                    accumulate=accum,
                )
            elect_p2 = nvvm.elect_sync()
            bars.mb_bmm2_done[par_cur].arrive(cta_group=1, pred=elect_p2)
            bars.mb_kv_empty[slot_v].arrive(cta_group=1, pred=elect_p2)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)


@cute.jit
def _warp_reduce_max(x):
    for off in cutlass.range_constexpr(5):
        x = cute.math.max(x, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY)))
    return x


@cute.jit
def _warp_reduce_sum(x):
    for off in cutlass.range_constexpr(5):
        x = x + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY))
    return x


@cute.jit
def _softmax_warp_group(
    tmem_ptr_i32,
    bars,
    red_smem,
    sP_raw,
    o_tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    n_batch,
    head_idx,
    batch_idx,
    split_idx,
    scale_log2: cutlass.Float32,
):
    nvvm.barrier_cta_sync(barrier_id=_BAR_TMEM, thread_count=32 * (SOFTMAX_WARPS + 1))
    tidx = cute.arch.thread_idx()[0]
    sm_warp = tidx // cutlass.Int32(32)  # 0 .. SOFTMAX_WARPS - 1
    if cutlass.const_expr(COL_GROUPS == 1):
        lane = tidx  # key row of S^T / P^T, d row of O^T
        col0 = cutlass.Int32(0)
    else:
        lane = tidx % cutlass.Int32(128)
        # Column group (warp-uniform): the first S^T / O^T column this thread owns.
        col0 = cute.arch.make_warp_uniform(tidx // cutlass.Int32(128)) * cutlass.Int32(COLS)
    G = cutlass.Int32(HEADS_PER_TILE)
    tmem_base = tmem_ptr_i32.load()

    lo, hi, eff_seqlen_kv, eff_seqlen_q, q_len_b = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv)
    n_tiles = hi - lo
    causal_diag = cutlass.Int32(0)
    if cutlass.const_expr(CFG.BOTTOM_RIGHT):
        causal_diag = eff_seqlen_kv - eff_seqlen_q

    NEG_INF = cutlass.Float32(float("-inf"))
    ZERO = cutlass.Float32(0.0)
    ONE = cutlass.Float32(1.0)
    # Running column state: max in the scaled log2 domain (-inf = no live key
    # yet), this lane's partial sum (the lane sum is reduced once, at the end).
    m_vec = cutlass.Vector.from_elements(tuple(NEG_INF for _ in range(COLS)), cutlass.Float32)
    l_vec = cutlass.Vector.from_elements(tuple(ZERO for _ in range(COLS)), cutlass.Float32)

    red_ptr = Pointer(red_smem.data_ptr(), dtype=cutlass.Float32)

    for i in cutlass.range(0, n_tiles, 1, unroll=1):
        kv_tile = lo + i
        par = i & cutlass.Int32(1)
        phase_i = (i // cutlass.Int32(2)) & cutlass.Int32(1)
        s_off = _select_i32(par == cutlass.Int32(0), S_ACC_OFF[0], S_ACC_OFF[1])

        bars.mb_s_full[par].wait(phase_i)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        s_raw = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + s_off + col0, cutlass.Float32), num=COLS)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.mb_s_empty[par].arrive()

        key_abs = kv_tile * cutlass.Int32(TILE_N) + lane
        # Per-column (= per Q row) mask in the transposed orientation: the key
        # is this lane, the Q token is the column's row // G.
        s_cols = []
        for j in cutlass.range_constexpr(COLS):
            v = cutlass.Float32(s_raw[j]) * scale_log2
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                q_abs = (col0 + cutlass.Int32(j)) // G
                masked = None
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
                    masked = key_abs >= eff_seqlen_kv
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
                    term = key_abs > (q_abs + causal_diag + cutlass.Int32(CFG.WINDOW_RIGHT))
                    masked = term if masked is None else (masked | term)
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
                    term = key_abs < (q_abs + causal_diag - cutlass.Int32(CFG.WINDOW_LEFT))
                    masked = term if masked is None else (masked | term)
                v = _select_f32(masked, NEG_INF, v)
            s_cols.append(v)

        # Column max over the tile's 128 keys: butterfly inside the warp, then
        # the group's four warps exchange through SMEM (slot = tile parity, so
        # the single barrier per tile also orders the next reuse of the slot).
        red_base = par * cutlass.Int32(SOFTMAX_WARPS * COLS) + sm_warp * cutlass.Int32(COLS)
        col_max = []
        for j in cutlass.range_constexpr(COLS):
            cm = _warp_reduce_max(s_cols[j])
            (red_ptr + (red_base + cutlass.Int32(j))).store(cm)
            col_max.append(cm)
        nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=SOFTMAX_LANES)
        red_read = par * cutlass.Int32(SOFTMAX_WARPS * COLS) + (sm_warp - (sm_warp % cutlass.Int32(4))) * cutlass.Int32(COLS)
        tile_max = []
        for j in cutlass.range_constexpr(COLS):
            tm = col_max[j]
            for w in cutlass.range_constexpr(4):
                tm = cute.math.max(tm, cutlass.Float32((red_ptr + (red_read + cutlass.Int32(w * COLS + j))).load()))
            tile_max.append(tm)

        # Online update per column (identical in every lane: the max is the
        # tile-wide column max, so alpha is uniform and O^T's rescale is exact).
        m_new = []
        l_new = []
        alpha = []
        all_one = None
        p_cols = []
        for j in cutlass.range_constexpr(COLS):
            m_old_j = cutlass.Float32(m_vec[j])
            m_new_j = cute.math.max(m_old_j, tile_max[j])
            ms_old = row_max_for_exp2(m_old_j)
            ms_new = row_max_for_exp2(m_new_j)
            alpha_j = cute.math.exp2(cute.math.min(ms_old - ms_new, ZERO), fastmath=True)
            p_j = cute.math.exp2(s_cols[j] - ms_new, fastmath=True)
            l_new.append(cutlass.Float32(l_vec[j]) * alpha_j + p_j)
            m_new.append(m_new_j)
            alpha.append(alpha_j)
            p_cols.append(p_j)
            is_one = alpha_j == ONE
            all_one = is_one if all_one is None else (all_one & is_one)
        m_vec = cutlass.Vector.from_elements(tuple(m_new), cutlass.Float32)
        l_vec = cutlass.Vector.from_elements(tuple(l_new), cutlass.Float32)

        # P^T(t) -> SMEM slot t % 2 as packed half pairs, row = this lane's key,
        # this group's COLS-wide slice of it, swizzled the way the BMM2 B
        # descriptor decodes it.
        p_words = []
        for w in cutlass.range_constexpr(P_GROUP_WORDS):
            p_words.append(fp32_to_fp16(p_cols[2 * w], p_cols[2 * w + 1], dtype=STORAGE_DTYPE))
        p_row_base = par * cutlass.Int32(pBufferWords) + lane * cutlass.Int32(P_ROW_WORDS) + (col0 // cutlass.Int32(2))
        for c in cutlass.range_constexpr(P_CHUNKS_PER_GROUP):
            chunk = cutlass.Vector.from_elements(tuple(p_words[4 * c + k] for k in range(4)), cutlass.Int32)
            p_ptr = Pointer(sP_raw.subview(p_row_base + cutlass.Int32(4 * c)).data_ptr(), dtype=cutlass.Int32)
            p_ptr.store_swizzled(chunk, _P_SMEM_SWIZZLE, alignment=16)
        nvvm.fence_proxy("async.shared", space="cta")

        # O^T(t-1) is complete once BMM2(t-1) commits; rescale this group's
        # columns by alpha before BMM2(t) accumulates on top.  Skipped when no
        # column's max moved.
        if i > cutlass.Int32(0):
            t_prev = i - cutlass.Int32(1)
            bars.mb_bmm2_done[t_prev & cutlass.Int32(1)].wait((t_prev // cutlass.Int32(2)) & cutlass.Int32(1))
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            if ~all_one:
                for blk in cutlass.range_constexpr(O_BLOCKS):
                    o_ptr = nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk]) + col0, cutlass.Float32)
                    o_vals = nvvm.tcgen05_ld("32x32b", o_ptr, num=COLS)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    o_scaled = cutlass.Vector.from_elements(tuple(cutlass.Float32(o_vals[j]) * alpha[j] for j in range(COLS)), cutlass.Float32)
                    nvvm.tcgen05_st("32x32b", o_ptr, o_scaled)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        # Publish P^T(t) (and the rescaled O^T) to the MMA warp: the tcgen05 stores
        # are ordered before the arrive, and the MMA fences after its wait.
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.mb_p_full[par].arrive()

    # --- epilogue -----------------------------------------------------------
    if n_tiles > cutlass.Int32(0):
        # The last BMM2 must land before O^T is read.
        t_last = n_tiles - cutlass.Int32(1)
        bars.mb_bmm2_done[t_last & cutlass.Int32(1)].wait((t_last // cutlass.Int32(2)) & cutlass.Int32(1))
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)

    # Column sums over the 128 lanes (once per unit), via the epilogue slot.
    red_epi = cutlass.Int32(2 * SOFTMAX_WARPS * COLS) + sm_warp * cutlass.Int32(COLS)
    for j in cutlass.range_constexpr(COLS):
        (red_ptr + (red_epi + cutlass.Int32(j))).store(_warp_reduce_sum(cutlass.Float32(l_vec[j])))
    nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=SOFTMAX_LANES)
    red_epi_read = cutlass.Int32(2 * SOFTMAX_WARPS * COLS) + (sm_warp - (sm_warp % cutlass.Int32(4))) * cutlass.Int32(COLS)

    lse_cols = []
    inv_cols = []
    dead_cols = []
    live_cols = []
    LN2_F = cutlass.Float32(LN2)
    TINY = cutlass.Float32(1e-30)
    if cutlass.const_expr(CFG.HAS_SINK):
        sinks_arr = cutlass.make_array_view(sinks_tensor)
    tok_cols = []
    head_cols = []
    for j in cutlass.range_constexpr(COLS):
        jg = col0 + cutlass.Int32(j)  # the packed Q row this column is
        tok_j = jg // G
        head_j = head_idx * G + (jg % G)
        tok_cols.append(tok_j)
        head_cols.append(head_j)
        # Every warp's partial comes back from SMEM (this warp's included), so
        # the group's four warps see the same total.
        l_tot = ZERO
        for w in cutlass.range_constexpr(4):
            l_tot = l_tot + cutlass.Float32((red_ptr + (red_epi_read + cutlass.Int32(w * COLS + j))).load())
        m_nat = row_max_for_exp2(cutlass.Float32(m_vec[j])) * LN2_F
        row_dead = l_tot <= ZERO
        if cutlass.const_expr(CFG.HAS_SINK):
            sink_logit = cutlass.Float32(sinks_arr[head_j])
            new_max = cute.math.max(m_nat, sink_logit)
            scale = cute.math.exp(m_nat - new_max, fastmath=True)
            new_sum = l_tot * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_j = new_max + cute.math.log(new_sum, fastmath=True)
            inv_j = scale / new_sum
            # A keyless row keeps a finite LSE (the sink is its only column) and
            # O := 0; the select below only has to keep NaN out of 0 * garbage.
        else:
            lse_j = m_nat + cute.math.log(cute.math.max(l_tot, TINY), fastmath=True)
            inv_j = ONE / cute.math.max(l_tot, TINY)
            lse_j = _select_f32(row_dead, NEG_INF, lse_j)
            inv_j = _select_f32(row_dead, ZERO, inv_j)
        live_j = tok_j < seqlen_q
        if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
            # Dense padded-Q trim: rows past seq_len_q[b] write O := 0 / LSE := -inf,
            # applied AFTER the sink fold (a trimmed row is dead even with a sink).
            row_trim = tok_j >= q_len_b
            lse_j = _select_f32(row_trim, NEG_INF, lse_j)
            inv_j = _select_f32(row_trim, ZERO, inv_j)
            row_dead = row_dead | row_trim
        if cutlass.const_expr(CFG.STATS_LOG2):
            lse_j = lse_j * cutlass.Float32(LOG2E)
        lse_cols.append(lse_j)
        inv_cols.append(inv_j)
        dead_cols.append(row_dead)
        live_cols.append(live_j)

    # Split partials stack split-major on the workspace batch axis (b + s*B).
    o_batch = batch_idx + split_idx * n_batch

    if cutlass.const_expr(lse_tensor is not None):
        lse_arr = cutlass.make_array_view(lse_tensor)
        for j in cutlass.range_constexpr(COLS):
            if (lane == cutlass.Int32(j)) & live_cols[j]:
                lse_arr[o_batch, head_cols[j], tok_cols[j]] = lse_cols[j]

    # O[q, d] = O^T[d, q] / l[q]: this lane is d (and d + 128); the store
    # writes the graph's actual d_v columns only (the tile is an envelope).
    oo = cutlass.make_array_view(o_tensor)
    D_V = cutlass.const_expr(o_tensor.shape[3])
    for blk in cutlass.range_constexpr(O_BLOCKS):
        if cutlass.const_expr(blk * 128 < D_V):
            d_idx = lane + cutlass.Int32(blk * 128)
            o_vals = cutlass.Vector.from_elements(tuple(ZERO for _ in range(COLS)), cutlass.Float32)
            if n_tiles > cutlass.Int32(0):
                # Empty range: O^T TMEM was never written, so the load is skipped
                # rather than multiplied by 0 (NaN * 0 is NaN).
                o_vals = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk]) + col0, cutlass.Float32), num=COLS)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            if d_idx < cutlass.Int32(D_V):
                for j in cutlass.range_constexpr(COLS):
                    val = _select_f32(dead_cols[j], ZERO, cutlass.Float32(o_vals[j]) * inv_cols[j])
                    if live_cols[j]:
                        o_row = oo[o_batch, tok_cols[j], head_cols[j], :]
                        o_row[d_idx] = val.to(o_tensor.element_type)

    bars.mb_tmem_dealloc.arrive()


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_batch: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    # Dense padded-Q trim: (B,)-int32 per-batch Q lengths by address; 0 (unread)
    # unless CFG.SEQ_Q_LENS_PRESENT.
    seq_q_lens_addr: cutlass.Int64 = 0,
    # Paged KV: [B, max_pages] int32 page ids for K and V; None unless CFG.PAGED_KV.
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    paged_hnd: cutlass.Constexpr[bool] = False,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    # One CTA per unit: x = KV-head group (packed) or Q head (unpacked), y = batch, z = split.
    head_idx = cute.arch.block_idx()[0]
    batch_idx = cute.arch.block_idx()[1]
    split_idx = cute.arch.block_idx()[2]

    sQ_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sKV_raw = cutlass.Array(STORAGE_DTYPE, STAGES * kvBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_raw = cutlass.Array(cutlass.Int32, 2 * pBufferWords, alignment=1024, space=cutlass.AddressSpace.smem)
    red_smem = cutlass.Array(cutlass.Float32, RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # Q^T: the K-major B operand of BMM1 (N_Q rows x TILE_K), one TMA sub-tile
    # (N_Q rows x 128 B) per 64-wide head-dim granule.
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=N_Q * GRANU_ELEMS,
    )
    # The K/V ring, seen as the K-major A operand of BMM1 (TILE_N rows x TILE_K) ...
    sK = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
    )
    # ... and as the MN-major A operand of BMM2 (V^T: d contiguous, LBO steps
    # the 64-wide d atoms).  Same storage, different descriptor.
    sVt = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_VT,
        stride_byte_offset=STRIDE_BYTE_OFFSET_VT,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=VO_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
    )
    # P^T: the MN-major B operand of BMM2, two buffers of TILE_N key rows.
    sP = SmemTile(
        base=sP_raw,
        elems_per_stage=pBufferWords,
        stages=2,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_P,
        layout=SMEM_LAYOUT_P,
    )

    bars = make_decode_bars()

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_q_full.init()
            for s in cutlass.range_constexpr(STAGES):
                bars.mb_kv_full[s].init()
                bars.mb_kv_empty[s].init()
            for p in cutlass.range_constexpr(2):
                bars.mb_s_full[p].init()
                bars.mb_s_empty[p].init()
                bars.mb_p_full[p].init()
                bars.mb_bmm2_done[p].init()
            bars.mb_tmem_dealloc.init()

    if cutlass.const_expr(Q_BOX_ROWS < N_Q):
        # Zero the Q^T tail rows the TMA box does not cover (a head group that
        # does not divide N_Q), so their never-stored columns stay finite.
        if warp_idx < SOFTMAX_WARPS:
            zero4 = cutlass.Vector.from_elements(tuple(cutlass.Int32(0) for _ in range(4)), cutlass.Int32)
            tail_chunks = (N_Q - Q_BOX_ROWS) * 8  # 16 B chunks per sub-tile
            for it in cutlass.range_constexpr((QK_ITERS * tail_chunks + SOFTMAX_LANES - 1) // SOFTMAX_LANES):
                chunk = cutlass.Int32(it * SOFTMAX_LANES) + tidx
                if chunk < cutlass.Int32(QK_ITERS * tail_chunks):
                    sub = chunk // cutlass.Int32(tail_chunks)
                    within = chunk % cutlass.Int32(tail_chunks)
                    elem_off = sub * cutlass.Int32(N_Q * GRANU_ELEMS) + cutlass.Int32(Q_BOX_ROWS * GRANU_ELEMS) + within * cutlass.Int32(8)
                    Pointer(sQ_raw.subview(elem_off).data_ptr(), dtype=cutlass.Int32).store(zero4, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if warp_idx < SOFTMAX_WARPS:
        _softmax_warp_group(
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            red_smem=red_smem,
            sP_raw=sP_raw,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            n_batch=n_batch,
            head_idx=head_idx,
            batch_idx=batch_idx,
            split_idx=split_idx,
            scale_log2=scale_softmax_log2,
        )
    elif warp_idx == MMA_WARP_ID:
        _mma_warp_group(
            sQ=sQ,
            sK=sK,
            sVt=sVt,
            sP=sP,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            batch_idx=batch_idx,
            split_idx=split_idx,
        )
    elif warp_idx == TMALDG_WARP_ID:
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sKV=sK,
            bars=bars,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_idx=head_idx,
            batch_idx=batch_idx,
            split_idx=split_idx,
            block_table_tensor=block_table_tensor,
            block_table_v_tensor=block_table_v_tensor,
            paged_hnd=paged_hnd,
        )


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    seq_q_lens_addr: cutlass.Int64 = 0,
    # THD-only slots of the shared launcher ABI (this tile never compiles THD).
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    # Under a split the positional o_tensor IS the fp32 partial slab; this slot
    # only keeps the ABI in step with the prefill launcher.
    o_partial_f32: Optional[cute.Tensor] = None,
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(PAGED_KV):
        SKV = block_table_tensor.shape[1] * cutlass.Int32(PAGE_SIZE)

    # Q^T box: Q_BOX_TOKENS tokens x HEADS_PER_TILE heads of one KV group, row =
    # token * G + head (the packed-row convention); rows past S_q zero-fill.
    box_q = (1, Q_BOX_TOKENS, HEADS_PER_TILE, GRANU_ELEMS)
    box_kv = (1, KV_BOX_ROWS, 1, GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)
    paged_hnd = bool(PAGED_KV) and k_tensor.stride[1] < k_tensor.stride[2]
    if cutlass.const_expr(PAGED_KV and (v_tensor.stride[1] < v_tensor.stride[2]) != paged_hnd):
        raise ValueError("paged K and V pools must share an in-page layout (both HND or both NHD)")
    kv_stride_order = (3, 1, 2, 0) if paged_hnd else stride_order

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    def _create_tma_desc(tensor: cute.Tensor, box_dims, swizzle, order=stride_order):
        return tmap.create_tensor_map_tiled(
            global_address=tensor.iterator.toint(),
            dtype=tensor.element_type,
            global_dims=tuple(tensor.shape[i] for i in order),
            global_strides=tuple(cutlass.Int64(tensor.stride[i]) * tensor.element_type.width // 128 for i in order[1:]),
            box_dims=tuple(box_dims[i] for i in order),
            swizzle=swizzle,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_q_desc = _create_tma_desc(q_tensor, box_q, _tma_swz(CFG.Q_SWZ_BYTES))
    tma_k_desc = _create_tma_desc(k_tensor, box_kv, _tma_swz(CFG.K_SWZ_BYTES), kv_stride_order)
    tma_v_desc = _create_tma_desc(v_tensor, box_kv, _tma_swz(CFG.V_SWZ_BYTES), kv_stride_order)

    grid_shape = (QH // HEADS_PER_TILE, B, SPLIT_KV)
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(B),
        scale_softmax_log2,
        seq_q_lens_addr,
        block_table_tensor,
        block_table_v_tensor,
        paged_hnd,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 1,
    skv: int = 128,
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_stride: Optional[tuple[int, int, int]] = None,
    q_stride: Optional[tuple] = None,
    k_stride: Optional[tuple] = None,
    v_stride: Optional[tuple] = None,
    o_stride: Optional[tuple] = None,
    block_table_stride: Optional[tuple[int, int]] = None,
    block_table_v_stride: Optional[tuple[int, int]] = None,
) -> Callable:
    """Same argument contract as prefill_d256_f16.compile (dense and paged): the
    adapter calls both alike.  ENVELOPE: ``d_qk`` / ``d_v`` are the ACTUAL head
    dims; the TMA descriptors carry them (loads past them zero-fill) and the O
    store writes ``d_v`` columns only.  ``sq * pack_g`` rows must fit N_Q."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d256 decode envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * BPE) % 16 != 0 or (d_v * BPE) % 16 != 0:
        raise ValueError(f"d256 decode envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v})")
    if sq * HEADS_PER_TILE > N_Q:
        raise ValueError(f"d256 decode: S_q ({sq}) x packed heads ({HEADS_PER_TILE}) exceeds the {N_Q}-row Q tile")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_stride is not None and SPLIT_KV > 1:
        raise ValueError("dense LSE strides are not valid for split-KV workspaces")
    if PAGED_KV and (k_stride is None or v_stride is None):
        raise ValueError("PAGED_KV: k_stride / v_stride (the pools' strides in [num_pages, page_size, H_kv, D] order) are required")

    def _fake_bshd(shape, stride, dtype=STORAGE_DTYPE, bpe=BPE):
        if stride is None:
            return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)
        if stride[3] != 1:
            raise ValueError(f"declared stride {stride}: the head dim must be innermost-contiguous (stride[3] == 1)")
        for axis in (1, 2):
            if (stride[axis] * bpe) % 16 != 0:
                raise ValueError(f"declared stride {stride} axis {axis} must be a 16-byte multiple at BPE={bpe} (TMA global-stride rule)")
        return cute.runtime.make_fake_tensor(dtype, shape, tuple(stride), assumed_align=16)

    fake_q = _fake_bshd((b, sq, qh, d_qk), q_stride)
    if PAGED_KV:
        n_pages = cute.sym_int(divisibility=1)
        fake_k = _fake_bshd((n_pages, PAGE_SIZE, kh, d_qk), k_stride)
        fake_v = _fake_bshd((n_pages, PAGE_SIZE, kh, d_v), v_stride)
        _max_pages = cute.sym_int(divisibility=1)

        def _fake_table(stride):
            if stride is None:
                return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b, _max_pages), stride_order=(1, 0), assumed_align=4)
            return cute.runtime.make_fake_tensor(cutlass.Int32, (b, _max_pages), tuple(stride), assumed_align=4)

        fake_block_table = _fake_table(block_table_stride)
        fake_block_table_v = _fake_table(block_table_v_stride)
    else:
        fake_k = _fake_bshd((b, skv, kh, d_qk), k_stride)
        fake_v = _fake_bshd((b, skv, kh, d_v), v_stride)
        fake_block_table = None
        fake_block_table_v = None
    _o_batch = b * SPLIT_KV
    fake_o = _fake_bshd(
        (_o_batch, sq, qh, d_v),
        o_stride,
        dtype=cutlass.Float32 if _FP32_PARTIALS else OUT_STORAGE_DTYPE,
        bpe=4 if _FP32_PARTIALS else BPE,
    )
    if not has_lse:
        fake_lse = None
    else:
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, (_o_batch, qh, sq), lse_stride, assumed_align=4)
            if lse_stride is not None
            else cute.runtime.make_fake_compact_tensor(cutlass.Float32, (_o_batch, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
        )
    fake_sinks = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (qh,), stride_order=(0,), assumed_align=16)
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b,), stride_order=(0,), assumed_align=16)
    fake_o_desc = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (1,), stride_order=(0,), assumed_align=16)
    return _compile_cached(
        _host,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        fake_sinks,
        fake_seq_kv_lens,
        fake_o_desc,
        (b, qh, kh, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        cutlass.Int64(0),
        None,
        None,
        None,
        # o_partial_f32 slot: the fp32 partial O under a split, else -- only when
        # the paged tables follow it positionally -- an explicit None.
        *((fake_o,) if _FP32_PARTIALS else ((None,) if PAGED_KV else ())),
        *((fake_block_table, fake_block_table_v) if PAGED_KV else ()),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )
