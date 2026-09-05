# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 block-scale MXFP8 prefill SDPA for exact d_qk=d_v=512.

The production path reuses the D256 one-CTA M128 pipeline with K=512 and
computes two 256-column V/O slices. This geometry stays within the SM100 SMEM
and 512-column TMEM limits and outperforms the validated two-CTA M128 design.
"""

from functools import lru_cache
from typing import Callable, Optional, Tuple

from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync  # noqa: F401
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from dataclasses import dataclass

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d512_mxfp8

# The template loader injects the graph's MXFP8 parameters before executing
# this module. The default record only keeps import-time tooling well-defined;
# it is not a supported standalone execution configuration for this kernel.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1))
CFG, _TMA = make_cfg_d512_mxfp8(PARAMS)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

FULL_DV = 512
OUTPUT_SLICE_D = CFG.TILE_O
OUTPUT_SLICES = FULL_DV // OUTPUT_SLICE_D
if FULL_DV % OUTPUT_SLICE_D:
    raise ValueError("d512 MXFP8 output dimension must divide evenly into output slices")

TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    wait,
    arrive_expect_tx,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    read_tile_id_arrive,
    SCHED_NATURAL,
)
from cudnn.frost.tile_dsl.pointwise import (
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_wait, tma_tensormap_acquire
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import (
    apply_mask_chunk,
    MASK_NONE,
    MASK_CAUSAL,
    MASK_PADDED,
    MASK_SWA,
)
from cudnn.block_sparse_attention.csrc.utils.kernel_utils import ex2_emulation_2

if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float8E4M3FN
    P_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.Float8E5M2
    P_STORAGE_DTYPE = cutlass.Float8E5M2
else:
    raise ValueError(f"prefill_sdpa_d512_mxfp8: unsupported DTYPE_QKV={CFG.DTYPE_QKV}")

MMA_KIND = nvvm.MMABlockScaleKind.MXF8F6F4
SCALE_VEC_SIZE = nvvm.Tcgen05MMAScaleVecSize.BLOCK32

# SM100 MXFP8 uses the K=32 block-scaled QMMA path.  K=64 is not a valid
# transfer from the SM107 implementation.
_MXFP8_K_DIM = 0
_MXFP8_TILE_K_HW = 32

if CFG.DTYPE_O == 0:
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_O == 1:
    OUT_STORAGE_DTYPE = cutlass.Float8E5M2
elif CFG.DTYPE_O == 2:
    OUT_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_O == 3:
    OUT_STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"prefill_sdpa_d512_mxfp8: unsupported DTYPE_O={CFG.DTYPE_O}")


BITS_PER_SF_ELEMENT = 8
BLOCK_SCALE_BLOCK_SIZE = 32
SF_BLOCK_DIM_NON_K = 128
SF_BLOCK_DIM_K = 4
SF_SWIZZLED_BLOCK_DIM_K = 16
SF_BYTES_PER_BLOCK = SF_BLOCK_DIM_NON_K * SF_BLOCK_DIM_K * BITS_PER_SF_ELEMENT // 8


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


SF_NUM_BLOCKS_M = CFG.TILE_M // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_N = CFG.TILE_N // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_K = _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE // SF_BLOCK_DIM_K
SF_REGISTERS_PER_BLOCK = SF_SWIZZLED_BLOCK_DIM_K * BITS_PER_SF_ELEMENT // 32
SF_TMEM_COLS_Q = SF_NUM_BLOCKS_M * SF_NUM_BLOCKS_K * SF_REGISTERS_PER_BLOCK
SF_TMEM_COLS_K = SF_NUM_BLOCKS_N * SF_NUM_BLOCKS_K * SF_REGISTERS_PER_BLOCK
SF_SMEM_SIZE_Q = CFG.TILE_M * _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE
SF_SMEM_SIZE_K = CFG.TILE_N * _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE

SF_NUM_BLOCKS_P = CFG.TILE_M // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_V = _round_up(CFG.TILE_O, 128) // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_K_BMM2 = CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE // SF_BLOCK_DIM_K
SF_TMEM_COLS_P = SF_NUM_BLOCKS_P * SF_NUM_BLOCKS_K_BMM2 * SF_REGISTERS_PER_BLOCK
SF_TMEM_COLS_V = SF_NUM_BLOCKS_V * SF_NUM_BLOCKS_K_BMM2 * SF_REGISTERS_PER_BLOCK
SF_SMEM_SIZE_P = _round_up(CFG.TILE_M, 128) * CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE
SF_SMEM_SIZE_V_SLICE = _round_up(CFG.TILE_O, 128) * CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE
# Public input-tile size consumed by api_dsl._reshape_sf().
SF_SMEM_SIZE_V = _round_up(FULL_DV, 128) * CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE
SF_CONST_VALUE = 0x7F

SF_BMM2_N_BLOCKS = _round_up(CFG.TILE_O, 128) // 128
BMM2_LOOP_N_BLOCKS = SF_BMM2_N_BLOCKS // CFG.CTA_MMA
BMM2_N_PER_CALL = CFG.TILE_O // BMM2_LOOP_N_BLOCKS
BMM2_N_PER_CALL_PER_CTA = BMM2_N_PER_CALL // CFG.CTA_MMA
BMM2_N_BLOCK_BYTE_STRIDE = CFG.TILE_N * CFG.V_SWZ_BYTES
SF_V_COLS_PER_NBLOCK = SF_NUM_BLOCKS_K_BMM2 * SF_REGISTERS_PER_BLOCK


from cudnn.sdpa.fwd.kernels._common_sm100 import (
    make_split_helpers,
    KvLoopBounds,
    make_d256_bars,
    row_max_for_exp2,
    make_sdpa_helpers,
)

CGA_SIZE = CFG.CGA_M * CFG.CGA_N

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA
oBufferElems = CFG.TILE_M * CFG.TILE_O
qoAliasBytes = max(qBufferElems * CFG.BPE, oBufferElems * CFG.BPE_O)

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA
Q_SF_EXPECT_BYTES = SF_SMEM_SIZE_Q * CFG.CTA_MMA
K_SF_EXPECT_BYTES = SF_SMEM_SIZE_K * CFG.CTA_MMA
V_SF_EXPECT_BYTES = SF_SMEM_SIZE_V_SLICE * CFG.CTA_MMA

N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128

_BAR_K_EMPTY_0 = 3
_BAR_K_EMPTY_1 = 4
_BAR_V_EMPTY_0 = 5
_BAR_V_EMPTY_1 = 6
_KV_PIPELINE_LANES = 64
KV_PROLOGUE_TILES = min(2, CFG.STAGES_KV)

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
THD_PERSISTENT = False

_sdpa_h = make_sdpa_helpers(
    CFG,
    lpt_q_tiles_in_cga_units=True,
    grouped_lpt=PARAMS.lpt_head_group > 1,
    lpt_head_group=PARAMS.lpt_head_group,
)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
# qtrim variant: collapses the KV loop for CGA tiles entirely past the
# per-batch actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q


_thd_tma_offsets = _sdpa_h.thd_tma_offsets
_thd_sf_tile_bases = _sdpa_h.thd_sf_tile_bases

from cudnn.sdpa.fwd.kernels.thd_helpers import (
    build_thd_meta_o_kv_descs_kernel as _build_thd_meta_o_kv_descs_kernel,
    thd_decode_unit,
    TENSOR_MAP_QWORDS,
)

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_THD_REVERSE_CAUSAL_ROWS = CFG.THD_VARLEN and CFG.MASK_FLAGS == (MASK_CAUSAL | MASK_PADDED) and CFG.BOTTOM_RIGHT == 0 and CFG.WINDOW_RIGHT == 0


@cute.jit
def _thd_decode_causal(linear_cta, seq_kv_lens_t, n_batch, n_qh, cta_in_pair):
    meta = cutlass.make_array_view(seq_kv_lens_t)
    unit = linear_cta // cutlass.Int32(CFG.CGA_M)
    q_tile, batch, head = thd_decode_unit(
        meta,
        n_batch,
        unit,
        n_qh,
        cutlass.Int32(CGA_TILE_M),
        True,
    )
    # Keep the heavy-to-light traversal, but interleave groups of eight heads so
    # the last wave is not dominated by one head.  A partial final group, if
    # present, retains the original head-major order.
    cuq0 = n_batch
    live = batch < n_batch
    safe_batch = cute.math.min(batch, n_batch - cutlass.Int32(1))
    seq_q = cutlass.Int32(meta[cuq0 + safe_batch + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + safe_batch])
    tiles_q = (seq_q + cutlass.Int32(CGA_TILE_M - 1)) // cutlass.Int32(CGA_TILE_M)
    rank = tiles_q - cutlass.Int32(1) - q_tile
    group = head // cutlass.Int32(8)
    group_head0 = group * cutlass.Int32(8)
    group_is_full = live & (group_head0 + cutlass.Int32(7) < n_qh)
    group_linear = (head - group_head0) * tiles_q + rank
    grouped_head = group_head0 + (group_linear & cutlass.Int32(7))
    grouped_rank = group_linear // cutlass.Int32(8)
    head = cutlass.Int32(arith.select(group_is_full.ir_value(), grouped_head.ir_value(), head.ir_value()))
    q_tile = cutlass.Int32(
        arith.select(
            group_is_full.ir_value(),
            (tiles_q - cutlass.Int32(1) - grouped_rank).ir_value(),
            q_tile.ir_value(),
        )
    )
    return q_tile * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair, head, batch


@cute.jit
def _dispatch_decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
    if cutlass.const_expr(_THD_REVERSE_CAUSAL_ROWS):
        return _thd_decode_causal(bidx, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
    return _sdpa_h.dispatch_decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)


@cute.jit
def _dispatch_decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
    if cutlass.const_expr(_THD_REVERSE_CAUSAL_ROWS):
        return _thd_decode_causal(t0, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
    return _sdpa_h.dispatch_decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)


# Each split runs one contiguous slice of a Q tile's KV range and writes a
# split-major partial O/LSE pair for split_combine_sm100.  At SPLIT_KV == 1
# these helpers fold back to the original decode and bounds path.
_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch


@dataclass(frozen=True)
class KernelTmemLayout:
    TOTAL_COLS: int = 512

    S_ACC_EVEN_OFF: int = 0
    S_ACC_ODD_OFF: int = 128

    # FP8 probabilities pack four values per TMEM column and alias the tails of
    # the two 128-column score slots.
    P_EVEN_OFF: int = 96
    P_ODD_OFF: int = 224

    O_OFF: int = 256

    STATS_EVEN_OFF: int = 0
    STATS_ODD_OFF: int = 128
    STATS_EPI_OFF: int = 0

    # Scale-factor scratch.  Prologue Q/K use dead O columns.  In fused split-P,
    # steady-state Q/K use WG1's consumed high score half while P/V retain the
    # selected low-half addresses released by WG0.
    SF_HEAD_OFFSET: int = 64 if CFG.FUSED_CORR_SPLIT_P else 8
    SF_AFTER_P_OFFSET: int = 32
    SF_Q_PRO_OFF: int = 256
    SF_K_PRO_OFF: int = 256 + SF_TMEM_COLS_Q


LAYOUT = KernelTmemLayout()
# Slots 6:10 carry predecoded bounds and slot 10 carries the split index.  The
# unsplit/no-mask specialization retains the original eight-word payload.
SCHED_PAYLOAD_WORDS = 12 if (CFG.MASK_FLAGS != 0 or SPLIT_KV > 1) else 8
_E5_STYLE_KV_PIPELINE = CFG.DTYPE_QKV == 1 or (CFG.DTYPE_QKV == 0 and (CFG.MASK_FLAGS & ~MASK_PADDED) in (MASK_NONE, MASK_CAUSAL))
_PADDED_TOP_LEFT_CAUSAL = CFG.MASK_FLAGS == (MASK_CAUSAL | MASK_PADDED) and CFG.BOTTOM_RIGHT == 0 and CFG.WINDOW_RIGHT == 0
_SWA_REUSE_P = (
    CFG.CTA_MMA == 1
    and CFG.MASK_FLAGS == (MASK_CAUSAL | MASK_SWA)
    and CFG.WINDOW_LEFT == CFG.TILE_N - 1
    and CFG.WINDOW_RIGHT == 0
    and CFG.BOTTOM_RIGHT == 0
    and CFG.SEQ_KV_LENS_PRESENT == 0
    and CFG.THD_VARLEN == 0
    and SPLIT_KV == 1
)


@cute.jit
def _producer_wait_k_empty(bars, kv_state, kv_load_count):
    if cutlass.const_expr(CFG.CTA_MMA == 1 and (_E5_STYLE_KV_PIPELINE or CFG.MASK_FLAGS == MASK_NONE)):
        if kv_load_count >= cutlass.Int32(CFG.STAGES_KV):
            if kv_state.idx == cutlass.Int32(0):
                nvvm.barrier_cta_sync(_BAR_K_EMPTY_0, thread_count=_KV_PIPELINE_LANES)
            else:
                nvvm.barrier_cta_sync(_BAR_K_EMPTY_1, thread_count=_KV_PIPELINE_LANES)
    else:
        bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)


@cute.jit
def _producer_wait_v_empty(bars, kv_state, kv_load_count):
    if cutlass.const_expr(CFG.CTA_MMA == 1 and (_E5_STYLE_KV_PIPELINE or CFG.MASK_FLAGS == MASK_NONE)):
        if kv_load_count >= cutlass.Int32(CFG.STAGES_KV):
            if kv_state.idx == cutlass.Int32(0):
                nvvm.barrier_cta_sync(_BAR_V_EMPTY_0, thread_count=_KV_PIPELINE_LANES)
            else:
                nvvm.barrier_cta_sync(_BAR_V_EMPTY_1, thread_count=_KV_PIPELINE_LANES)
    else:
        bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)


@cute.jit
def _consumer_arrive_k_empty(bars, kv_state, mcast_mask):
    if cutlass.const_expr(CFG.CTA_MMA == 1 and (_E5_STYLE_KV_PIPELINE or CFG.MASK_FLAGS == MASK_NONE)):
        if kv_state.idx == cutlass.Int32(0):
            nvvm.barrier_cta_arrive(_BAR_K_EMPTY_0, _KV_PIPELINE_LANES)
        else:
            nvvm.barrier_cta_arrive(_BAR_K_EMPTY_1, _KV_PIPELINE_LANES)
    else:
        bars.mb_k_empty[kv_state.idx].arrive(
            mcast_mask=mcast_mask,
            cta_group=CFG.CTA_MMA,
            pred=nvvm.elect_sync(),
        )


@cute.jit
def _consumer_arrive_v_empty(bars, kv_state, mcast_mask):
    if cutlass.const_expr(CFG.CTA_MMA == 1 and (_E5_STYLE_KV_PIPELINE or CFG.MASK_FLAGS == MASK_NONE)):
        if kv_state.idx == cutlass.Int32(0):
            nvvm.barrier_cta_arrive(_BAR_V_EMPTY_0, _KV_PIPELINE_LANES)
        else:
            nvvm.barrier_cta_arrive(_BAR_V_EMPTY_1, _KV_PIPELINE_LANES)
    else:
        bars.mb_v_empty[kv_state.idx].arrive(
            mcast_mask=mcast_mask,
            cta_group=CFG.CTA_MMA,
            pred=nvvm.elect_sync(),
        )


@cute.jit
def _scheduler_warp_loop_predecode(
    sched,
    mb_decoded,
    is_cga_first_cta,
    cta_in_pair,
    n_q_supers,
    n_qh,
    n_batch,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
):
    if cutlass.const_expr(CFG.THD_VARLEN):
        meta = cutlass.make_array_view(seq_kv_lens_tensor)
        ctr_ptr = Pointer(seq_kv_lens_tensor.iterator.raw_ptr(), dtype=cutlass.Int32) + cutlass.Int32(4) * n_batch + cutlass.Int32(3)
        live_off = cutlass.Int32(4) * n_batch + cutlass.Int32(2)

    state = PipelineState.start()
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)

        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
        if nvvm.elect_sync() and is_cga_first_cta:
            if cutlass.const_expr(CFG.THD_VARLEN):
                uid = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))
                live = cutlass.Int32(meta[live_off])
                valid = cutlass.Int32(
                    arith.select(
                        (uid < live).ir_value(),
                        cutlass.Int32(1).ir_value(),
                        cutlass.Int32(0).ir_value(),
                    )
                )
                tile_ptr = cute.make_ptr(
                    cutlass.Int32,
                    sched.tile_id_smem.subview(state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)).data_ptr().toint(cutlass.Int32),
                    cutlass.AddressSpace.smem,
                    assumed_align=16,
                )
                mbar_ptr = cute.make_ptr(
                    cutlass.Int64,
                    sched.mb_scheduler.subview(state.idx).data_ptr().toint(cutlass.Int32),
                    cutlass.AddressSpace.smem,
                    assumed_align=8,
                )
                payload = (uid * cutlass.Int32(CFG.CGA_M), cutlass.Int32(0), valid, cutlass.Int32(0))
                for cta_rank in cutlass.range_constexpr(CGA_SIZE):
                    for word in cutlass.range_constexpr(4):
                        cute.arch.store_async_dsmem(tile_ptr + word, payload[word], mbar_ptr, cta_rank)
            else:
                nvvm.clusterlaunchcontrol_try_cancel(
                    sched.tile_id_smem.subview(state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)),
                    sched.mb_scheduler.subview(state.idx),
                    multicast=1,
                )

        if cutlass.const_expr(not CFG.THD_VARLEN):
            nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(state.idx), state.phase)

        payload_base = state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        validity = sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load()
        if nvvm.elect_sync():
            nxt_q = sched.tile_id_smem.subview(payload_base + cutlass.Int32(0)).load()
            nxt_hb = sched.tile_id_smem.subview(payload_base + cutlass.Int32(1)).load()
            q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
                nxt_q,
                nxt_hb,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
                CFG.QH_PER_KH,
                seqlen_kv,
            )
            sched.tile_id_smem.subview(payload_base + cutlass.Int32(3)).store(q_super_idx)
            sched.tile_id_smem.subview(payload_base + cutlass.Int32(4)).store(head_idx)
            sched.tile_id_smem.subview(payload_base + cutlass.Int32(5)).store(batch_idx)
            if cutlass.const_expr(SPLIT_KV > 1):
                sched.tile_id_smem.subview(payload_base + cutlass.Int32(10)).store(split_idx)
            if cutlass.const_expr(CFG.MASK_FLAGS != 0 or SPLIT_KV > 1):
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
                if cutlass.const_expr(CFG.MASK_FLAGS == 0):
                    kv_left, kv_right = _nomask_range_split(eff_seqlen_kv, split_idx)
                    bounds = KvLoopBounds(left=kv_left, unmasked_lo=kv_left, unmasked_hi=kv_right, right=kv_right)
                else:
                    bounds = _bounds_for_tile_split(
                        q_super_idx,
                        eff_seqlen_q,
                        eff_seqlen_kv,
                        cta_in_pair,
                        seq_q_lens_tensor,
                        batch_idx,
                        split_idx,
                        CFG.QH_PER_KH,
                    )
                sched.tile_id_smem.subview(payload_base + cutlass.Int32(6)).store(bounds.left)
                sched.tile_id_smem.subview(payload_base + cutlass.Int32(7)).store(bounds.unmasked_lo)
                sched.tile_id_smem.subview(payload_base + cutlass.Int32(8)).store(bounds.unmasked_hi)
                sched.tile_id_smem.subview(payload_base + cutlass.Int32(9)).store(bounds.right)
            nvvm.mbarrier_arrive(mb_decoded.subview(state.idx))

        is_valid = validity & cutlass.Int32(1)
        state = advance(state, CFG.SCHEDULER_STAGES)


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q
SMEM_LAYOUT_SF = 0
SF_LEADING_BYTE_OFFSET = 16
SF_STRIDE_BYTE_OFFSET = 128

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // _MXFP8_TILE_K_HW
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS


def _max_abs_reduction(vec):
    """Balanced max(abs(vec)) without materializing an abs vector."""
    maxima = [vec[i] for i in range(int(vec.shape[0]))]
    minima = list(maxima)
    while len(maxima) > 1:
        next_maxima = []
        next_minima = []
        for i in range(0, len(maxima), 3):
            max_group = maxima[i : i + 3]
            min_group = minima[i : i + 3]
            max_acc = max_group[0]
            min_acc = min_group[0]
            for value in max_group[1:]:
                max_acc = cute.math.max(max_acc, value, ftz=True)
            for value in min_group[1:]:
                min_acc = cute.math.min(min_acc, value, ftz=True)
            next_maxima.append(max_acc)
            next_minima.append(min_acc)
        maxima = next_maxima
        minima = next_minima
    return cute.math.max(maxima[0], -minima[0], ftz=True)


_E5_STYLE_SOFTMAX = _E5_STYLE_KV_PIPELINE


def _exp2_chunk0a_mixed(vec, apply_mask, dense=False, softmax_half=0):
    values = []
    for i in range(0, int(vec.shape[0]), 2):
        if _E5_STYLE_SOFTMAX and apply_mask:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        elif _E5_STYLE_SOFTMAX and not dense and i == 0:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        elif _E5_STYLE_SOFTMAX and not dense:
            x, y = ex2_emulation_2(vec[i], vec[i + 1], poly_degree=1)
        elif _E5_STYLE_SOFTMAX and dense and (i % 10 < 4 or (softmax_half == 0 and i == 26) or (softmax_half == 1 and (i == 24 or i == 26 or i == 28))):
            x, y = ex2_emulation_2(vec[i], vec[i + 1], poly_degree=2)
        elif not _E5_STYLE_SOFTMAX and i % 10 < 4 and not (dense and softmax_half == 1 and i == 30):
            x, y = ex2_emulation_2(vec[i], vec[i + 1])
        else:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        values.extend((x, y))
    return cutlass.Vector.from_elements(tuple(values), cutlass.Float32)


def _exp2_chunk1b_mixed(vec, dense=False, softmax_half=0):
    values = []
    for i in range(0, int(vec.shape[0]), 2):
        threshold = 16
        if _E5_STYLE_SOFTMAX and not dense:
            threshold = 2
        if not _E5_STYLE_SOFTMAX and not dense:
            threshold = 20 if softmax_half == 2 else 16
        elif _E5_STYLE_SOFTMAX and dense:
            threshold = 18 if softmax_half == 0 else 22
        if i >= threshold or (_E5_STYLE_SOFTMAX and dense and softmax_half == 0 and (i == 12 or i == 14)):
            x, y = ex2_emulation_2(vec[i], vec[i + 1], poly_degree=1)
        else:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        values.extend((x, y))
    return cutlass.Vector.from_elements(tuple(values), cutlass.Float32)


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_q_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_q_supers: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    kv_sf_num_tiles: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    amax_o_tensor: cute.Tensor,
    dv_slice: cutlass.Int32,
    bottom_right_diagonal: cutlass.Constexpr[bool],
    # Dense padded-Q trim: separate (B,)-int32 per-batch Q lengths (mirrors
    # cuDNN's SEQLEN_Q pointer / FA's seqused_q). None unless
    # CFG.SEQ_Q_LENS_PRESENT — the DSL specializes on None, so the flag-off
    # ABI is unchanged.
    seq_q_lens_tensor: Optional[cute.Tensor] = None,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # Q and O have disjoint lifetimes. Allocate the larger byte footprint and
    # expose typed views so promoted half output cannot overrun the FP8 Q view.
    sQO_raw = cutlass.Array(STORAGE_DTYPE, qoAliasBytes // CFG.BPE, alignment=1024, space=cutlass.AddressSpace.smem)
    sO_raw = cutlass.Array(sQO_raw.data_ptr(), shape=oBufferElems, dtype=OUT_STORAGE_DTYPE)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sQ_SF_raw = cutlass.Array(cutlass.Int8, SF_SMEM_SIZE_Q, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * SF_SMEM_SIZE_K, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_SF_raw = cutlass.Array(cutlass.Int8, SF_SMEM_SIZE_P, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * SF_SMEM_SIZE_V_SLICE, alignment=1024, space=cutlass.AddressSpace.smem)

    sQ = SmemTile(
        base=sQO_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=1,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=1,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_QK_GRANU_ELEMS,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=1,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
    )
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=oBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_O,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
    )
    sQ_SF = SmemTile(
        base=sQ_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_Q,
        stages=1,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )
    sK_SF = SmemTile(
        base=sK_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_K,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )
    sP_SF = SmemTile(
        base=sP_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_P,
        stages=1,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )
    sV_SF = SmemTile(
        base=sV_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_V_SLICE,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )

    bars = make_d256_bars(CFG, N_O_CHUNKS=N_O_CHUNKS)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    # Dense WG0->WG1 exchange, or two parity-indexed causal alpha slots.
    softmax_exchange = cutlass.Array(
        cutlass.Float32,
        768 if CFG.SOFTMAX_WARPGROUPS == 2 else (2 * CFG.TILE_M if (_E5_STYLE_SOFTMAX or _SWA_REUSE_P) else 1),
        alignment=16,
        space=cutlass.AddressSpace.smem,
    )

    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(
                cutlass.Int32,
                CFG.SCHEDULER_STAGES * SCHED_PAYLOAD_WORDS,
                alignment=16,
                space=cutlass.AddressSpace.smem,
            ),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )
    mb_decoded = cutlass.Array(
        cutlass.Int64,
        CFG.SCHEDULER_STAGES,
        alignment=16,
        space=cutlass.AddressSpace.smem,
    )
    # P/V scale factors occupy the low score half released by WG0.  Fused
    # split-P Q/K scale factors use a separate high-half transition from WG1.
    mb_sf_reuse = cutlass.Array(cutlass.Int64, 2, alignment=16, space=cutlass.AddressSpace.smem)
    mb_qk_sf_reuse = cutlass.Array(cutlass.Int64, 2, alignment=16, space=cutlass.AddressSpace.smem)
    READ_TILE_ARRIVERS_TOTAL = CFG.READ_TILE_ARRIVERS

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_q_full.init()
            bars.mb_q_o_alias.init()
            bars.mb_tmastg_go.init()
            for p in cutlass.range_constexpr(2):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()
            bars.mb_stat_full.init()
            bars.mb_stat_empty.init()
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[chunk].init()
            bars.mb_o_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[ks].init()
                bars.mb_k_empty[ks].init()
                bars.mb_v_full[ks].init()
                bars.mb_v_empty[ks].init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOTAL)
                nvvm.mbarrier_init(mb_decoded.subview(s), CFG.ONE_LANE)
            for p in cutlass.range_constexpr(2):
                nvvm.mbarrier_init(
                    mb_sf_reuse.subview(p),
                    CFG.SOFTMAX_WG_WARPS,
                )
                nvvm.mbarrier_init(
                    mb_qk_sf_reuse.subview(p),
                    CFG.SOFTMAX_WG_WARPS,
                )
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()

            bars.mb_q_o_alias.arrive()

    for sf_i in cutlass.range_constexpr((SF_SMEM_SIZE_P + CFG.THREADS_PER_CTA - 1) // CFG.THREADS_PER_CTA):
        sf_off = tidx + cutlass.Int32(sf_i * CFG.THREADS_PER_CTA)
        if sf_off < cutlass.Int32(SF_SMEM_SIZE_P):
            sP_SF_raw.subview(sf_off).store(cutlass.Int8(SF_CONST_VALUE))

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    tma_mcast_mask = (cutlass.Int16(1) << cta_in_pair) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            softmax_half=0,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            mb_decoded=mb_decoded,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_tensor=seq_q_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            bottom_right_diagonal=bottom_right_diagonal,
            softmax_exchange=softmax_exchange,
            mb_sf_reuse=mb_sf_reuse,
            mb_qk_sf_reuse=mb_qk_sf_reuse,
        )

    elif cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 2) and warp_idx >= CFG.SOFTMAX_WG1_BASE and warp_idx < CFG.SOFTMAX_WG1_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_WG1_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
            _correction_warp_group(
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                scale_log2=scale_softmax_log2,
                sO=sO,
                tmem_ptr_i32=tmem_ptr_i32,
                tidx=tidx,
                bars=bars,
                sched=sched,
                mb_decoded=mb_decoded,
                lse_tensor=lse_tensor,
                sinks_tensor=sinks_tensor,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_tensor=seq_q_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                leader_cta_id=leader_cta_id,
                cta_in_pair=cta_in_pair,
                cta_id_x=cta_id_x,
                bottom_right_diagonal=bottom_right_diagonal,
                amax_o_tensor=amax_o_tensor,
                softmax_exchange=softmax_exchange,
                mb_sf_reuse=mb_sf_reuse,
                mb_qk_sf_reuse=mb_qk_sf_reuse,
            )
        else:
            _softmax_warp_group(
                softmax_half=1,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                scale_log2=scale_softmax_log2,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                mb_decoded=mb_decoded,
                lse_tensor=lse_tensor,
                sinks_tensor=sinks_tensor,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_tensor=seq_q_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                leader_cta_id=leader_cta_id,
                cta_in_pair=cta_in_pair,
                bottom_right_diagonal=bottom_right_diagonal,
                softmax_exchange=softmax_exchange,
                mb_sf_reuse=mb_sf_reuse,
                mb_qk_sf_reuse=mb_qk_sf_reuse,
            )

    elif warp_idx >= CFG.CORR_WARP_BASE and warp_idx < CFG.CORR_WARP_BASE + CFG.CORRECTION_WARPS:
        nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _correction_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            sO=sO,
            tmem_ptr_i32=tmem_ptr_i32,
            tidx=tidx,
            bars=bars,
            sched=sched,
            mb_decoded=mb_decoded,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_tensor=seq_q_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            cta_id_x=cta_id_x,
            bottom_right_diagonal=bottom_right_diagonal,
            amax_o_tensor=amax_o_tensor,
            softmax_exchange=softmax_exchange,
            mb_sf_reuse=mb_sf_reuse,
            mb_qk_sf_reuse=mb_qk_sf_reuse,
        )

    elif warp_idx == CFG.MMA_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(CFG.CTA_MMA == 2):
            if is_leader:
                _mma_warp_group(
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    sQ=sQ,
                    sK=sK,
                    sV=sV,
                    sQ_SF=sQ_SF,
                    sK_SF=sK_SF,
                    sP_SF=sP_SF,
                    sV_SF=sV_SF,
                    tmem_ptr_i32=tmem_ptr_i32,
                    bars=bars,
                    sched=sched,
                    mb_decoded=mb_decoded,
                    seq_kv_lens_tensor=seq_kv_lens_tensor,
                    seq_q_lens_tensor=seq_q_lens_tensor,
                    n_q_supers=n_q_supers,
                    n_qh=n_qh,
                    n_batch=n_batch,
                    mcast_mask=mcast_mask,
                    cta_in_pair=cta_in_pair,
                    mb_sf_reuse=mb_sf_reuse,
                    mb_qk_sf_reuse=mb_qk_sf_reuse,
                )
            else:
                _mma_warp_quiet(tmem_ptr_i32, bars)
        else:
            _mma_warp_group(
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                sK=sK,
                sV=sV,
                sQ_SF=sQ_SF,
                sK_SF=sK_SF,
                sP_SF=sP_SF,
                sV_SF=sV_SF,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                mb_decoded=mb_decoded,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_tensor=seq_q_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
                mb_sf_reuse=mb_sf_reuse,
                mb_qk_sf_reuse=mb_qk_sf_reuse,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_q_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_sf_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            tma_q_sf_desc=tma_q_sf_desc,
            tma_k_sf_desc=tma_k_sf_desc,
            tma_v_sf_desc=tma_v_sf_desc,
            sQ=sQ,
            sK=sK,
            sV=sV,
            sQ_SF=sQ_SF,
            sK_SF=sK_SF,
            sV_SF=sV_SF,
            bars=bars,
            sched=sched,
            mb_decoded=mb_decoded,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
            seq_q_lens_tensor=seq_q_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            kv_sf_num_tiles=kv_sf_num_tiles,
            dv_slice=dv_slice,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
        )

    elif warp_idx == CFG.TMASTG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            tma_o_desc=tma_o_desc,
            sO=sO,
            bars=bars,
            sched=sched,
            mb_decoded=mb_decoded,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            seqlen_kv=seqlen_kv,
            cta_in_pair=cta_in_pair,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
            dv_slice=dv_slice,
        )

    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        _scheduler_warp_loop_predecode(
            sched,
            mb_decoded,
            is_cga_first_cta,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seqlen_q,
            seqlen_kv,
            seq_kv_lens_tensor,
            seq_q_lens_tensor,
        )


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    tma_q_sf_desc,
    tma_k_sf_desc,
    tma_v_sf_desc,
    sQ,
    sK,
    sV,
    sQ_SF,
    sK_SF,
    sV_SF,
    bars,
    sched,
    mb_decoded,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    o_desc_words,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    kv_sf_num_tiles,
    dv_slice,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
):

    q_o_alias_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=1)
    kv_state_V = PipelineState.start(phase=1)
    kv_load_count = cutlass.Int32(0)

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN):
        k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tma_tensormap_acquire(k_rt_ptr)
        tma_tensormap_acquire(v_rt_ptr)
        tma_k = lambda *coords: tma_slice_runtime_desc(k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(v_rt_ptr, *coords)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)
    tma_q_sf = GmemTileTma(tma_q_sf_desc)
    tma_k_sf = GmemTileTma(tma_k_sf_desc)
    tma_v_sf = GmemTileTma(tma_v_sf_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
    cu_sf_q_base, cu_sf_k_base = _thd_sf_tile_bases(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
        bounds_init = _bounds_for_tile_split(
            q_super_idx,
            eff_seqlen_q,
            eff_seqlen_kv,
            cta_in_pair,
            seq_q_lens_tensor,
            batch_idx,
            split_idx,
            CFG.QH_PER_KH,
        )
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    n_kh = n_qh // qh_per_kh

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        q_sf_tile_base = q_row_base // cutlass.Int32(CFG.TILE_M)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            # K does not alias the prior tile's O buffer. Issue the first K TMA
            # before waiting for Q/O ownership so its latency overlaps the
            # previous tile's output store and the alias handoff.
            kv_row_base = kv_left * cutlass.Int32(CFG.TILE_N)
            _producer_wait_k_empty(bars, kv_state_K, kv_load_count)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_k_full[kv_state_K.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_k_full[kv_state_K.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                sK[kv_state_K.idx],
                tma_k(cutlass.Int32(0), kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                bars.mb_k_full[kv_state_K.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=False,
            )
            tma_load_tile(
                sK_SF[kv_state_K.idx],
                tma_k_sf(cutlass.Int32(0), cu_sf_k_base + kv_left, kv_head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                bars.mb_k_full[kv_state_K.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

            _producer_wait_v_empty(bars, kv_state_V, kv_load_count)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_v_full[kv_state_V.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_v_full[kv_state_V.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                sV[kv_state_V.idx],
                tma_v(cutlass.Int32(0), kv_row_base + kv_seq_off, dv_slice * cutlass.Int32(TMA_VO_ITERS), kv_head_idx, tma_batch),
                bars.mb_v_full[kv_state_V.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=False,
            )
            v_sf_group = (tma_batch * n_kh + kv_head_idx) * kv_sf_num_tiles + cu_sf_k_base + kv_left
            tma_load_tile(
                sV_SF[kv_state_V.idx],
                tma_v_sf(cutlass.Int32(0), cutlass.Int32(0), dv_slice * cutlass.Int32(SF_NUM_BLOCKS_V), v_sf_group),
                bars.mb_v_full[kv_state_V.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            kv_state_V = advance(kv_state_V, CFG.STAGES_KV)
            if cutlass.const_expr(CFG.CTA_MMA == 1 and (_E5_STYLE_KV_PIPELINE or CFG.MASK_FLAGS == MASK_NONE)):
                if kv_load_count < cutlass.Int32(CFG.STAGES_KV):
                    kv_load_count = kv_load_count + cutlass.Int32(1)

            if cutlass.const_expr(KV_PROLOGUE_TILES > 1):
                kv_second = kv_left + cutlass.Int32(1)
                if kv_second < kv_right:
                    kv_row_base = kv_second * cutlass.Int32(CFG.TILE_N)
                    _producer_wait_k_empty(bars, kv_state_K, kv_load_count)
                    if cutlass.const_expr(CFG.CTA_MMA == 2):
                        bars.mb_k_full[kv_state_K.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
                    else:
                        bars.mb_k_full[kv_state_K.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
                    tma_load_tile(
                        sK[kv_state_K.idx],
                        tma_k(cutlass.Int32(0), kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                        bars.mb_k_full[kv_state_K.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                        acquire=False,
                    )
                    tma_load_tile(
                        sK_SF[kv_state_K.idx],
                        tma_k_sf(cutlass.Int32(0), cu_sf_k_base + kv_second, kv_head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                        bars.mb_k_full[kv_state_K.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                    )

                    kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                    _producer_wait_v_empty(bars, kv_state_V, kv_load_count)
                    if cutlass.const_expr(CFG.CTA_MMA == 2):
                        bars.mb_v_full[kv_state_V.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
                    else:
                        bars.mb_v_full[kv_state_V.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
                    tma_load_tile(
                        sV[kv_state_V.idx],
                        tma_v(cutlass.Int32(0), kv_row_base + kv_seq_off, dv_slice * cutlass.Int32(TMA_VO_ITERS), kv_head_idx, tma_batch),
                        bars.mb_v_full[kv_state_V.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                        acquire=False,
                    )
                    v_sf_group = (tma_batch * n_kh + kv_head_idx) * kv_sf_num_tiles + cu_sf_k_base + kv_second
                    tma_load_tile(
                        sV_SF[kv_state_V.idx],
                        tma_v_sf(cutlass.Int32(0), cutlass.Int32(0), dv_slice * cutlass.Int32(SF_NUM_BLOCKS_V), v_sf_group),
                        bars.mb_v_full[kv_state_V.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                    )
                    kv_state_V = advance(kv_state_V, CFG.STAGES_KV)
                    if cutlass.const_expr(CFG.CTA_MMA == 1 and (_E5_STYLE_KV_PIPELINE or CFG.MASK_FLAGS == MASK_NONE)):
                        if kv_load_count < cutlass.Int32(CFG.STAGES_KV):
                            kv_load_count = kv_load_count + cutlass.Int32(1)

        bars.mb_q_o_alias.wait(q_o_alias_phase)
        q_o_alias_phase = q_o_alias_phase ^ cutlass.Int32(1)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), q_row_base + q_seq_off, cutlass.Int32(0), head_idx, tma_batch),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            tma_load_tile(
                sQ_SF[0],
                tma_q_sf(
                    cutlass.Int32(0),
                    cu_sf_q_base + q_sf_tile_base,
                    head_idx,
                    tma_batch,
                    coord_0=cutlass.Int32(0),
                ),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            kv_main_start = cute.math.min(kv_left + cutlass.Int32(KV_PROLOGUE_TILES), kv_right)
            for kv_loop in cutlass.range(kv_main_start, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)

                _producer_wait_k_empty(bars, kv_state_K, kv_load_count)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state_K.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state_K.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[kv_state_K.idx],
                    tma_k(cutlass.Int32(0), kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                    bars.mb_k_full[kv_state_K.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=False,
                )
                tma_load_tile(
                    sK_SF[kv_state_K.idx],
                    tma_k_sf(cutlass.Int32(0), cu_sf_k_base + kv_loop, kv_head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                    bars.mb_k_full[kv_state_K.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                _producer_wait_v_empty(bars, kv_state_V, kv_load_count)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_v_full[kv_state_V.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_v_full[kv_state_V.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[kv_state_V.idx],
                    tma_v(cutlass.Int32(0), kv_row_base + kv_seq_off, dv_slice * cutlass.Int32(TMA_VO_ITERS), kv_head_idx, tma_batch),
                    bars.mb_v_full[kv_state_V.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=False,
                )
                v_sf_group = (tma_batch * n_kh + kv_head_idx) * kv_sf_num_tiles + cu_sf_k_base + kv_loop
                tma_load_tile(
                    sV_SF[kv_state_V.idx],
                    tma_v_sf(cutlass.Int32(0), cutlass.Int32(0), dv_slice * cutlass.Int32(SF_NUM_BLOCKS_V), v_sf_group),
                    bars.mb_v_full[kv_state_V.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

        if nvvm.elect_sync():
            bars.mb_tmastg_go.arrive()

        if cutlass.const_expr(_SWA_REUSE_P):
            # Reuse the retained P tiles for the second Dv256 slice.  V has its
            # own pipeline state because K is not replayed in this pass.
            for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                _producer_wait_v_empty(bars, kv_state_V, kv_load_count)
                bars.mb_v_full[kv_state_V.idx].arrive(
                    n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES,
                    pred=nvvm.elect_sync(),
                )
                tma_load_tile(
                    sV[kv_state_V.idx],
                    tma_v(
                        cutlass.Int32(0),
                        kv_row_base + kv_seq_off,
                        (dv_slice + cutlass.Int32(1)) * cutlass.Int32(TMA_VO_ITERS),
                        kv_head_idx,
                        tma_batch,
                    ),
                    bars.mb_v_full[kv_state_V.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=False,
                )
                v_sf_group = (tma_batch * n_kh + kv_head_idx) * kv_sf_num_tiles + cu_sf_k_base + kv_loop
                tma_load_tile(
                    sV_SF[kv_state_V.idx],
                    tma_v_sf(
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        (dv_slice + cutlass.Int32(1)) * cutlass.Int32(SF_NUM_BLOCKS_V),
                        v_sf_group,
                    ),
                    bars.mb_v_full[kv_state_V.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)
        payload_base = sched_state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load())
        q_super_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(3)).load())
        head_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(4)).load())
        batch_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(5)).load())
        kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        cu_sf_q_base, cu_sf_k_base = _thd_sf_tile_bases(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0 or SPLIT_KV > 1):
            kv_left = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(6)).load())
            kv_right = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(9)).load())

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state_K.idx].wait(kv_state_K.phase)
            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)
        for _vs in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_v_empty[kv_state_V.idx].wait(kv_state_V.phase)
            kv_state_V = advance(kv_state_V, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _tmastg_warp_group(
    tma_o_desc,
    sO,
    bars,
    sched,
    mb_decoded,
    n_q_supers,
    n_qh,
    n_batch,
    seqlen_kv,
    cta_in_pair,
    seq_kv_lens_tensor,
    o_desc_words,
    dv_slice,
):

    tmastg_go_phase = cutlass.Int32(0)
    o_full_phase = cutlass.Int32(0)

    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        CFG.QH_PER_KH,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_tmastg_go.wait(tmastg_go_phase)
        tmastg_go_phase = tmastg_go_phase ^ cutlass.Int32(1)

        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        if cutlass.const_expr(CFG.THD_VARLEN):
            o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        for output_pass in cutlass.range_constexpr(OUTPUT_SLICES if _SWA_REUSE_P else 1):
            output_slice = dv_slice + cutlass.Int32(output_pass)
            if cutlass.const_expr(not CFG.THD_VARLEN):
                o_slice = tma_o(output_slice * cutlass.Int32(OUTPUT_SLICE_D), head_idx, q_row_coord, o_batch)
                o_desc_ptr = o_slice.tma_desc.get_ptr()
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[chunk].wait(o_full_phase)
                smem_chunk = sO[0].base.subview(chunk * sO[0].tma_subtile_stride_elems)
                d_coord = cutlass.Int32(chunk * TMA_O_GRANU_ELEMS_HOST)
                if cutlass.const_expr(CFG.THD_VARLEN):
                    if batch_idx < n_batch:
                        nvvm.cp_async_bulk_tensor_global_shared_cta(
                            o_desc_ptr,
                            smem_chunk,
                            (output_slice * cutlass.Int32(OUTPUT_SLICE_D) + d_coord, head_idx, q_row_coord, cutlass.Int32(0)),
                        )
                else:
                    nvvm.cp_async_bulk_tensor_global_shared_cta(
                        o_desc_ptr,
                        smem_chunk,
                        (o_slice.coord_d + d_coord, head_idx, q_row_coord, o_batch),
                    )
                tma_store_commit()

            tma_store_wait(0)
            bars.mb_o_empty.arrive()
            o_full_phase = o_full_phase ^ cutlass.Int32(1)

        # D512 Q occupies the full Q/O slab, so both O slices must drain before
        # the next tile can overwrite the alias with Q.
        if nvvm.elect_sync():
            bars.mb_q_o_alias.arrive()

        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)
        payload_base = sched_state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        nxt_v = sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load()
        q_super_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(3)).load()
        head_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(4)).load()
        batch_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(5)).load()
        if cutlass.const_expr(SPLIT_KV > 1):
            split_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(10)).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_group(
    seqlen_q,
    seqlen_kv,
    sQ,
    sK,
    sV,
    sQ_SF,
    sK_SF,
    sP_SF,
    sV_SF,
    tmem_ptr_i32,
    bars,
    sched,
    mb_decoded,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
    mb_sf_reuse,
    mb_qk_sf_reuse,
):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=_MXFP8_K_DIM,
    )
    idesc_pv = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=BMM2_N_PER_CALL,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=_MXFP8_K_DIM,
    )
    sf_blocks_per_step = _MXFP8_TILE_K_HW // BLOCK_SCALE_BLOCK_SIZE
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=_MXFP8_TILE_K_HW,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
        is_block_scale=True,
        sf_blocks_per_step=sf_blocks_per_step,
        scale_vec_size=SCALE_VEC_SIZE,
    )
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=BMM2_N_PER_CALL,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=_MXFP8_TILE_K_HW,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_pv,
        kind=MMA_KIND,
        is_block_scale=True,
        sf_blocks_per_step=sf_blocks_per_step,
        scale_vec_size=SCALE_VEC_SIZE,
    )

    desc_Q = sQ[0].desc()
    desc_Q_SF = sQ_SF[0].desc()
    desc_P_SF = sP_SF[0].desc()

    tmem_SF_Q_pro = tmem_raw.subview(LAYOUT.SF_Q_PRO_OFF)
    tmem_SF_K_pro = tmem_raw.subview(LAYOUT.SF_K_PRO_OFF)

    def _utccp_bmm1_sf(tmem_sf_q, tmem_sf_k, smem_desc_q, smem_desc_k):
        for sf_k in cutlass.range_constexpr(SF_NUM_BLOCKS_K):
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                tmem_sf_q.subview(sf_k * SF_REGISTERS_PER_BLOCK),
                smem_desc_q + sf_k * (SF_BYTES_PER_BLOCK // 16),
                group=CTA_GROUP_KIND,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )
        for sf_k in cutlass.range_constexpr(SF_NUM_BLOCKS_K):
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                tmem_sf_k.subview(sf_k * SF_REGISTERS_PER_BLOCK),
                smem_desc_k + sf_k * (SF_BYTES_PER_BLOCK // 16),
                group=CTA_GROUP_KIND,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )

    def _utccp_bmm2_sf(tmem_sf_p, tmem_sf_v, smem_desc_p, smem_desc_v):
        nvvm.tcgen05_cp(
            nvvm.Tcgen05CpShape.SHAPE_32X128B,
            tmem_sf_p,
            smem_desc_p,
            group=CTA_GROUP_KIND,
            multicast=nvvm.Tcgen05CpMulticast.WARPX4,
        )
        for sf_n in cutlass.range_constexpr(SF_NUM_BLOCKS_V):
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                tmem_sf_v.subview(sf_n * SF_V_COLS_PER_NBLOCK),
                smem_desc_v + sf_n * (SF_BYTES_PER_BLOCK // 16),
                group=CTA_GROUP_KIND,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )

    def _utccp_bmm2_p_sf(tmem_sf_p, smem_desc_p):
        nvvm.tcgen05_cp(
            nvvm.Tcgen05CpShape.SHAPE_32X128B,
            tmem_sf_p,
            smem_desc_p,
            group=CTA_GROUP_KIND,
            multicast=nvvm.Tcgen05CpMulticast.WARPX4,
        )

    def _utccp_bmm2_v_sf(tmem_sf_v, smem_desc_v):
        for sf_n in cutlass.range_constexpr(SF_NUM_BLOCKS_V):
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                tmem_sf_v.subview(sf_n * SF_V_COLS_PER_NBLOCK),
                smem_desc_v + sf_n * (SF_BYTES_PER_BLOCK // 16),
                group=CTA_GROUP_KIND,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        q_super_idx, _hd, batch_idx, split_idx = _decode_initial_split(
            sched.bidx_init,
            sched.bidy_init,
            sched.bidz_init,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            CFG.QH_PER_KH,
            seqlen_kv,
        )
        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        else:
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
            bounds_init = _bounds_for_tile_split(
                q_super_idx,
                eff_seqlen_q,
                eff_seqlen_kv,
                cta_in_pair,
                seq_q_lens_tensor,
                batch_idx,
                split_idx,
                CFG.QH_PER_KH,
            )
            kv_left = bounds_init.left
            kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    kv_state_V = PipelineState.start(phase=0)
    bmm2_ready_phase_pair = cutlass.Int32(0)
    sf_reuse_phase_pair = cutlass.Int32(0)
    qk_sf_reuse_phase_pair = cutlass.Int32(0)
    empty_mainloop_phase = cutlass.Int32(0)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            bars.mb_empty_mainloop.wait(empty_mainloop_phase)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            bars.mb_q_full.wait(q_full_phase)
            q_full_phase = q_full_phase ^ cutlass.Int32(1)

            lo_parity_runtime = kv_left & cutlass.Int32(1)
            tmem_S_acc_lo_addr = lo_parity_runtime << 7

            bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
            desc_K = sK[kv_state_K.idx].desc()
            desc_K_SF = sK_SF[kv_state_K.idx].desc()
            if nvvm.elect_sync():
                _utccp_bmm1_sf(tmem_SF_Q_pro, tmem_SF_K_pro, desc_Q_SF, desc_K_SF)
            mma_ss(
                bmm1_desc,
                desc_Q,
                desc_K,
                tmem_raw.subview(tmem_S_acc_lo_addr),
                tmem_sf_a=tmem_SF_Q_pro,
                tmem_sf_b=tmem_SF_K_pro,
                elect_once=True,
            )
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[lo_parity_runtime].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            _consumer_arrive_k_empty(bars, kv_state_K, mcast_mask)
            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

            k_per_chunk = NUM_KPHASES_PV_PER_CHUNK

            for kv_loop in cutlass.range(kv_left, kv_right - cutlass.Int32(1), 1, unroll=1):
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                parity_next_rt = (kv_loop + cutlass.Int32(1)) & cutlass.Int32(1)
                parity_col_off = parity_cur_rt << 7
                tmem_S_acc_cur_addr = parity_col_off
                tmem_S_acc_next_addr = cutlass.Int32(LAYOUT.S_ACC_ODD_OFF) - parity_col_off
                tmem_P_cur_addr = cutlass.Int32(LAYOUT.P_EVEN_OFF) + parity_col_off
                bmm2_ready_phase_cur = (bmm2_ready_phase_pair >> parity_cur_rt) & cutlass.Int32(1)
                sf_reuse_phase_cur = (sf_reuse_phase_pair >> parity_cur_rt) & cutlass.Int32(1)
                qk_sf_reuse_phase_cur = (qk_sf_reuse_phase_pair >> parity_cur_rt) & cutlass.Int32(1)

                bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                desc_K = sK[kv_state_K.idx].desc()
                desc_K_SF = sK_SF[kv_state_K.idx].desc()
                if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
                    wait(mb_qk_sf_reuse.subview(parity_cur_rt), qk_sf_reuse_phase_cur)
                    qk_sf_reuse_phase_pair = qk_sf_reuse_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                    wait(mb_sf_reuse.subview(parity_cur_rt), sf_reuse_phase_cur)
                    sf_reuse_phase_pair = sf_reuse_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                else:
                    wait(mb_sf_reuse.subview(parity_cur_rt), sf_reuse_phase_cur)
                    sf_reuse_phase_pair = sf_reuse_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                tmem_SF_Q = tmem_raw.subview(tmem_S_acc_cur_addr + cutlass.Int32(LAYOUT.SF_HEAD_OFFSET))
                tmem_SF_K = tmem_raw.subview(tmem_S_acc_cur_addr + cutlass.Int32(LAYOUT.SF_HEAD_OFFSET + SF_TMEM_COLS_Q))
                if nvvm.elect_sync():
                    _utccp_bmm1_sf(tmem_SF_Q, tmem_SF_K, desc_Q_SF, desc_K_SF)
                mma_ss(
                    bmm1_desc,
                    desc_Q,
                    desc_K,
                    tmem_raw.subview(tmem_S_acc_next_addr),
                    tmem_sf_a=tmem_SF_Q,
                    tmem_sf_b=tmem_SF_K,
                    elect_once=True,
                )
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[parity_next_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                _consumer_arrive_k_empty(bars, kv_state_K, mcast_mask)
                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                tmem_SF_P = tmem_raw.subview(tmem_S_acc_cur_addr + cutlass.Int32(LAYOUT.SF_AFTER_P_OFFSET))
                tmem_SF_V = tmem_raw.subview(tmem_S_acc_cur_addr + cutlass.Int32(LAYOUT.SF_AFTER_P_OFFSET + SF_TMEM_COLS_P))
                if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                    if nvvm.elect_sync():
                        _utccp_bmm2_p_sf(tmem_SF_P, desc_P_SF)
                bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
                desc_V = sV[kv_state_V.idx].desc()
                desc_V_SF = sV_SF[kv_state_V.idx].desc()
                if nvvm.elect_sync():
                    if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                        _utccp_bmm2_v_sf(tmem_SF_V, desc_V_SF)
                    else:
                        _utccp_bmm2_sf(tmem_SF_P, tmem_SF_V, desc_P_SF, desc_V_SF)

                scaleC = cutlass.Boolean(kv_loop != kv_left)
                if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                    bmm2_issue = nvvm.elect_sync()
                    for chunk_id in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                        bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_cur)
                        for n_block in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                                k = chunk_id * NUM_KPHASES_PV_PER_CHUNK + local_k
                                accum_b2 = scaleC if k == 0 else cutlass.Boolean(True)
                                mma_ts_step(
                                    bmm2_desc,
                                    tmem_raw.subview(tmem_P_cur_addr),
                                    desc_V + n_block * (BMM2_N_BLOCK_BYTE_STRIDE // 16),
                                    tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + n_block * BMM2_N_PER_CALL_PER_CTA)),
                                    k,
                                    accum_b2,
                                    tmem_sf_a=tmem_SF_P,
                                    tmem_sf_b=tmem_SF_V.subview(n_block * SF_V_COLS_PER_NBLOCK),
                                    issue_mma=bmm2_issue,
                                )
                elif cutlass.const_expr(_E5_STYLE_KV_PIPELINE):
                    bmm2_issue_dense = nvvm.elect_sync()
                    for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                        if k % k_per_chunk == 0:
                            chunk_id = k // k_per_chunk
                            bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_cur)
                        for n_block in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                            accum_b2 = scaleC if k == 0 else cutlass.Boolean(True)
                            mma_ts_step(
                                bmm2_desc,
                                tmem_raw.subview(tmem_P_cur_addr),
                                desc_V + n_block * (BMM2_N_BLOCK_BYTE_STRIDE // 16),
                                tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + n_block * BMM2_N_PER_CALL_PER_CTA)),
                                k,
                                accum_b2,
                                tmem_sf_a=tmem_SF_P,
                                tmem_sf_b=tmem_SF_V.subview(n_block * SF_V_COLS_PER_NBLOCK),
                                issue_mma=bmm2_issue_dense,
                            )
                else:
                    for n_block in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                        accum_b2 = scaleC
                        for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                            if n_block == 0 and k % k_per_chunk == 0:
                                chunk_id = k // k_per_chunk
                                bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_cur)
                            mma_ts_step(
                                bmm2_desc,
                                tmem_raw.subview(tmem_P_cur_addr),
                                desc_V + n_block * (BMM2_N_BLOCK_BYTE_STRIDE // 16),
                                tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + n_block * BMM2_N_PER_CALL_PER_CTA)),
                                k,
                                accum_b2,
                                tmem_sf_a=tmem_SF_P,
                                tmem_sf_b=tmem_SF_V.subview(n_block * SF_V_COLS_PER_NBLOCK),
                            )
                            accum_b2 = cutlass.Boolean(True)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[parity_cur_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                _consumer_arrive_v_empty(bars, kv_state_V, mcast_mask)
                bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

            kv_last = kv_right - cutlass.Int32(1)
            parity_last_rt = kv_last & cutlass.Int32(1)
            parity_last_col_off = parity_last_rt << 7
            tmem_P_last_addr = cutlass.Int32(LAYOUT.P_EVEN_OFF) + parity_last_col_off
            bmm2_ready_phase_last = (bmm2_ready_phase_pair >> parity_last_rt) & cutlass.Int32(1)
            sf_reuse_phase_last = (sf_reuse_phase_pair >> parity_last_rt) & cutlass.Int32(1)
            qk_sf_reuse_phase_last = (qk_sf_reuse_phase_pair >> parity_last_rt) & cutlass.Int32(1)

            tmem_S_acc_last_addr = parity_last_col_off
            tmem_SF_P = tmem_raw.subview(tmem_S_acc_last_addr + cutlass.Int32(LAYOUT.SF_AFTER_P_OFFSET))
            tmem_SF_V = tmem_raw.subview(tmem_S_acc_last_addr + cutlass.Int32(LAYOUT.SF_AFTER_P_OFFSET + SF_TMEM_COLS_P))
            if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
                wait(mb_qk_sf_reuse.subview(parity_last_rt), qk_sf_reuse_phase_last)
                qk_sf_reuse_phase_pair = qk_sf_reuse_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                wait(mb_sf_reuse.subview(parity_last_rt), sf_reuse_phase_last)
                sf_reuse_phase_pair = sf_reuse_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
                if nvvm.elect_sync():
                    _utccp_bmm2_p_sf(tmem_SF_P, desc_P_SF)
            bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
            desc_V = sV[kv_state_V.idx].desc()
            desc_V_SF = sV_SF[kv_state_V.idx].desc()
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                if nvvm.elect_sync():
                    _utccp_bmm2_v_sf(tmem_SF_V, desc_V_SF)
            else:
                wait(mb_sf_reuse.subview(parity_last_rt), sf_reuse_phase_last)
                sf_reuse_phase_pair = sf_reuse_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
                if nvvm.elect_sync():
                    _utccp_bmm2_sf(tmem_SF_P, tmem_SF_V, desc_P_SF, desc_V_SF)
            n_kv_eff = kv_right - kv_left
            scaleC_epi = cutlass.Boolean(n_kv_eff != cutlass.Int32(1))
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                bmm2_issue = nvvm.elect_sync()
                for chunk_id in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_last)
                    for n_block in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                        for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                            k = chunk_id * NUM_KPHASES_PV_PER_CHUNK + local_k
                            accum_b2 = scaleC_epi if k == 0 else cutlass.Boolean(True)
                            mma_ts_step(
                                bmm2_desc,
                                tmem_raw.subview(tmem_P_last_addr),
                                desc_V + n_block * (BMM2_N_BLOCK_BYTE_STRIDE // 16),
                                tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + n_block * BMM2_N_PER_CALL_PER_CTA)),
                                k,
                                accum_b2,
                                tmem_sf_a=tmem_SF_P,
                                tmem_sf_b=tmem_SF_V.subview(n_block * SF_V_COLS_PER_NBLOCK),
                                issue_mma=bmm2_issue,
                            )
            elif cutlass.const_expr(_E5_STYLE_KV_PIPELINE):
                bmm2_issue_dense_epi = nvvm.elect_sync()
                for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                    if k % k_per_chunk == 0:
                        chunk_id = k // k_per_chunk
                        bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_last)
                    for n_block in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                        accum_b2 = scaleC_epi if k == 0 else cutlass.Boolean(True)
                        mma_ts_step(
                            bmm2_desc,
                            tmem_raw.subview(tmem_P_last_addr),
                            desc_V + n_block * (BMM2_N_BLOCK_BYTE_STRIDE // 16),
                            tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + n_block * BMM2_N_PER_CALL_PER_CTA)),
                            k,
                            accum_b2,
                            tmem_sf_a=tmem_SF_P,
                            tmem_sf_b=tmem_SF_V.subview(n_block * SF_V_COLS_PER_NBLOCK),
                            issue_mma=bmm2_issue_dense_epi,
                        )
            else:
                for n_block in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                    accum_b2 = scaleC_epi
                    for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                        if n_block == 0 and k % k_per_chunk == 0:
                            chunk_id = k // k_per_chunk
                            bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_last)
                        mma_ts_step(
                            bmm2_desc,
                            tmem_raw.subview(tmem_P_last_addr),
                            desc_V + n_block * (BMM2_N_BLOCK_BYTE_STRIDE // 16),
                            tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + n_block * BMM2_N_PER_CALL_PER_CTA)),
                            k,
                            accum_b2,
                            tmem_sf_a=tmem_SF_P,
                            tmem_sf_b=tmem_SF_V.subview(n_block * SF_V_COLS_PER_NBLOCK),
                        )
                        accum_b2 = cutlass.Boolean(True)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[parity_last_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            _consumer_arrive_v_empty(bars, kv_state_V, mcast_mask)
            bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
            kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

            if cutlass.const_expr(_SWA_REUSE_P):
                # P and its per-tile correction alpha remain resident after the
                # first Dv256 pass.  Replay only V/SF and BMM2 for D[256:512].
                for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    parity_rt = kv_loop & cutlass.Int32(1)
                    parity_col_off = parity_rt << 7
                    tmem_P_addr = cutlass.Int32(LAYOUT.P_EVEN_OFF) + parity_col_off
                    tmem_SF_P = tmem_raw.subview(parity_col_off + cutlass.Int32(LAYOUT.SF_AFTER_P_OFFSET))
                    tmem_SF_V = tmem_raw.subview(parity_col_off + cutlass.Int32(LAYOUT.SF_AFTER_P_OFFSET + SF_TMEM_COLS_P))
                    sf_reuse_phase = (sf_reuse_phase_pair >> parity_rt) & cutlass.Int32(1)
                    wait(mb_sf_reuse.subview(parity_rt), sf_reuse_phase)
                    sf_reuse_phase_pair = sf_reuse_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                    if nvvm.elect_sync():
                        _utccp_bmm2_p_sf(tmem_SF_P, desc_P_SF)

                    bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
                    desc_V = sV[kv_state_V.idx].desc()
                    desc_V_SF = sV_SF[kv_state_V.idx].desc()
                    if nvvm.elect_sync():
                        _utccp_bmm2_v_sf(tmem_SF_V, desc_V_SF)

                    bmm2_ready_phase = (bmm2_ready_phase_pair >> parity_rt) & cutlass.Int32(1)
                    scaleC = cutlass.Boolean(kv_loop != kv_left)
                    bmm2_issue = nvvm.elect_sync()
                    for chunk_id in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                        bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase)
                        for n_block in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                                k = chunk_id * NUM_KPHASES_PV_PER_CHUNK + local_k
                                accum_b2 = scaleC if k == 0 else cutlass.Boolean(True)
                                mma_ts_step(
                                    bmm2_desc,
                                    tmem_raw.subview(tmem_P_addr),
                                    desc_V + n_block * (BMM2_N_BLOCK_BYTE_STRIDE // 16),
                                    tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + n_block * BMM2_N_PER_CALL_PER_CTA)),
                                    k,
                                    accum_b2,
                                    tmem_sf_a=tmem_SF_P,
                                    tmem_sf_b=tmem_SF_V.subview(n_block * SF_V_COLS_PER_NBLOCK),
                                    issue_mma=bmm2_issue,
                                )
                    bars.mb_bmm2_done[parity_rt].arrive(
                        mcast_mask=mcast_mask,
                        cta_group=CFG.CTA_MMA,
                        pred=nvvm.elect_sync(),
                    )
                    _consumer_arrive_v_empty(bars, kv_state_V, mcast_mask)
                    bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                    kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)
        payload_base = sched_state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
            nxt_v = sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load()
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load())
            is_valid_tile = nxt_v & cutlass.Int32(1)
            kv_left = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(6)).load())
            kv_right = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(9)).load())
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _softmax_warp_group(
    softmax_half: cutlass.Constexpr[int],
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    bars,
    sched,
    mb_decoded,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    bottom_right_diagonal: cutlass.Constexpr[bool],
    softmax_exchange,
    mb_sf_reuse,
    mb_qk_sf_reuse,
):
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base = tmem_ptr_i32.load()

    bmm1_done_phase_pair = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(0 if CFG.FUSED_CORR_SPLIT_P else 1)
    epilogue_state = cutlass.Int32(1)
    reuse_gate_phase = cutlass.Int32(0)

    NEG_INF = cutlass.Float32(float("-inf"))

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        CFG.QH_PER_KH,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
    bounds = _bounds_for_tile_split(
        q_super_idx,
        eff_seqlen_q,
        eff_seqlen_kv,
        cta_in_pair,
        seq_q_lens_tensor,
        batch_idx,
        split_idx,
        CFG.QH_PER_KH,
    )

    softmax_wg_base = CFG.SOFTMAX_WG0_BASE if softmax_half == 0 else CFG.SOFTMAX_WG1_BASE
    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(softmax_wg_base * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        total_max = NEG_INF
        total_max_safe = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        q_abs = q_row_coord + tid_in_wg

        if cutlass.const_expr(not CFG.FUSED_CORR_SPLIT_P):
            bars.mb_o_empty.wait(epilogue_state)
        if cutlass.const_expr(softmax_half == 0 and not CFG.FUSED_CORR_SPLIT_P):
            bars.mb_stat_empty.wait(stat_empty_phase)
            stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        CHUNK = 64
        P_COLS_PER_CHUNK = CHUNK // 4
        P_SUBCHUNK = 32
        P_COLS_PER_SUBCHUNK = P_SUBCHUNK // 4
        N_CHUNKS = CFG.N_BMM2_CHUNKS
        RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)

                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + s_off_rt

                if cutlass.const_expr(softmax_half == 0):
                    if cutlass.const_expr(_E5_STYLE_KV_PIPELINE):
                        raw_hi = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(CHUNK), cutlass.Float32),
                            num=CHUNK,
                        )
                        raw_lo = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base, cutlass.Float32),
                            num=CHUNK,
                        )
                        raw_chunks = [raw_lo, raw_hi]
                    else:
                        raw_chunks = [
                            nvvm.tcgen05_ld(
                                "32x32b",
                                nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32),
                                num=CHUNK,
                            )
                            for c in range(N_CHUNKS)
                        ]
                    reg_S_full = RegTile(vec_concat(raw_chunks), size=CFG.TILE_N)
                    if cutlass.const_expr(_E5_STYLE_KV_PIPELINE):
                        max_hi = row_max_reduction(reg_S_full[CHUNK : 2 * CHUNK].vec)
                        max_lo = row_max_reduction(reg_S_full[0:CHUNK].vec)
                        current_max_unscaled = cute.math.max(max_lo, max_hi, ftz=True)
                    else:
                        current_max_unscaled = row_max_reduction(reg_S_full.vec)
                    current_max = current_max_unscaled * scale_log2
                    update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                    total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                    new_total_max_safe = total_max
                    alpha = cute.math.exp2(
                        cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)),
                        fastmath=True,
                    )
                    total_max_safe = new_total_max_safe
                    reg_S_half = reg_S_full[0:CHUNK]

                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                else:
                    reg_S_half = RegTile(
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(CHUNK), cutlass.Float32),
                            num=CHUNK,
                        ),
                        size=CHUNK,
                    )

                    reg_S_half = reg_S_half * scale_log2
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 1 or softmax_half == 0):
                    if nvvm.elect_sync():
                        nvvm.mbarrier_arrive(mb_sf_reuse.subview(parity_rt))
                if cutlass.const_expr(not CFG.FUSED_CORR_SPLIT_P):
                    nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)
                if cutlass.const_expr(softmax_half == 1):
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    new_total_max_safe = softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).load()
                    alpha = cute.math.exp2(
                        cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)),
                        fastmath=True,
                    )
                    total_max_safe = new_total_max_safe

                if cutlass.const_expr(softmax_half == 0):
                    if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
                        exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                        softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    elif cutlass.const_expr(_E5_STYLE_KV_PIPELINE):
                        exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                        softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    else:
                        alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_stat_full.arrive()

                if cutlass.const_expr(softmax_half == 0):
                    reg_S_half = reg_S_half * scale_log2 - total_max_safe
                else:
                    reg_S_half = reg_S_half - total_max_safe
                chunk_P_a = _exp2_chunk0a_mixed(
                    reg_S_half[0:P_SUBCHUNK].vec,
                    False,
                    dense=True,
                    softmax_half=softmax_half,
                )
                new_p_sum_pair = row_reduction_pair(chunk_P_a)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(
                        p_addr_base + cutlass.Int32(softmax_half * P_COLS_PER_CHUNK),
                        cutlass.Float32,
                    ),
                    chunk_P_a.to(STORAGE_DTYPE),
                )
                chunk_P_b = _exp2_chunk1b_mixed(
                    reg_S_half[P_SUBCHUNK:CHUNK].vec,
                    dense=True,
                    softmax_half=softmax_half,
                )
                new_p_sum_pair = new_p_sum_pair + row_reduction_pair(chunk_P_b)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(
                        p_addr_base + cutlass.Int32(softmax_half * P_COLS_PER_CHUNK + P_COLS_PER_SUBCHUNK),
                        cutlass.Float32,
                    ),
                    chunk_P_b.to(STORAGE_DTYPE),
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(softmax_half)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair

                if cutlass.const_expr(softmax_half == 0):
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        else:
            # Exact top-left causal has no lower masking boundary.
            left_edge_end = (
                bounds.left
                if cutlass.const_expr((CFG.MASK_FLAGS & ~MASK_PADDED) == MASK_CAUSAL and (CFG.BOTTOM_RIGHT == 0 or bottom_right_diagonal))
                else bounds.unmasked_lo
            )
            for kv_loop in cutlass.range(bounds.left, left_edge_end, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + s_off_rt
                kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 1 or softmax_half == 0):
                    if nvvm.elect_sync():
                        nvvm.mbarrier_arrive(mb_sf_reuse.subview(parity_rt))
                if cutlass.const_expr(bottom_right_diagonal):
                    mask_bottom_right = 0
                    causal_diag = None
                else:
                    mask_bottom_right = CFG.BOTTOM_RIGHT
                    causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
                chunks_S = [
                    apply_mask_chunk(
                        raw_chunks[c],
                        q_abs - (kv_col_base + cutlass.Int32(c * CHUNK)),
                        cutlass.Int32(0),
                        eff_seqlen_kv - (kv_col_base + cutlass.Int32(c * CHUNK)),
                        CFG.WINDOW_LEFT,
                        CFG.MASK_FLAGS,
                        N=CHUNK,
                        bottom_right=mask_bottom_right,
                        causal_diag=causal_diag,
                        window_right=CFG.WINDOW_RIGHT,
                        mask_value=float("-inf"),
                    )
                    for c in range(N_CHUNKS)
                ]
                reg_S_vec = vec_concat(chunks_S)
                current_max_unscaled = row_max_reduction(reg_S_vec)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

                # total_max starts at -inf: a live iteration always clears the
                # threshold (real - (-inf) = +inf), a fully-masked one never does
                # (-inf - x = -inf or NaN; ordered > is false for both).
                update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                # Canonical 0-substitution at point of use, on BOTH alpha operands;
                # min(., 0) guards the dead->alive drop of the safe max (total_sum
                # is still 0 there).  total_max_safe starts at -inf: iter-0 alpha = 0.
                new_total_max_safe = row_max_for_exp2(total_max)
                alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
                total_max_safe = new_total_max_safe
                if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                elif cutlass.const_expr(_E5_STYLE_SOFTMAX or _SWA_REUSE_P):
                    exchange_base = parity_rt * cutlass.Int32(CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                else:
                    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_P_0a = _exp2_chunk0a_mixed(reg_S[0:P_SUBCHUNK].vec, True)
                hoisted_sum = row_reduction_pair(chunk_P_0a)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0a.to(STORAGE_DTYPE))
                chunk_P_0b = cute.math.exp2(reg_S[P_SUBCHUNK:CHUNK].vec, fastmath=True)
                hoisted_sum = hoisted_sum + row_reduction_pair(chunk_P_0b)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_SUBCHUNK), cutlass.Float32),
                    chunk_P_0b.to(STORAGE_DTYPE),
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_sum_1 = None
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.FUSED_CORR_SPLIT_P):
                    chunk_P_1a = cute.math.exp2(reg_S[CHUNK : CHUNK + P_SUBCHUNK].vec, fastmath=True)
                    deferred_sum_1 = row_reduction_pair(chunk_P_1a)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32),
                        chunk_P_1a.to(STORAGE_DTYPE),
                    )
                    chunk_P_1b = _exp2_chunk1b_mixed(reg_S[CHUNK + P_SUBCHUNK : 2 * CHUNK].vec)
                    deferred_sum_1 = deferred_sum_1 + row_reduction_pair(chunk_P_1b)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK + P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_1b.to(STORAGE_DTYPE),
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.FUSED_CORR_SPLIT_P):
                    new_p_sum_pair = new_p_sum_pair + deferred_sum_1
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + s_off_rt
                unmasked_chunk = 32
                raw_chunks = [
                    nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * unmasked_chunk), cutlass.Float32),
                        num=unmasked_chunk,
                    )
                    for c in range(CFG.TILE_N // unmasked_chunk)
                ]
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 1 or softmax_half == 0):
                    if nvvm.elect_sync():
                        nvvm.mbarrier_arrive(mb_sf_reuse.subview(parity_rt))
                reg_S_vec = vec_concat(raw_chunks)
                max_lo = row_max_reduction(reg_S_vec[0:64])
                max_hi = row_max_reduction(reg_S_vec[64:128])
                current_max_unscaled = cute.math.max(max_lo, max_hi, ftz=True)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

                # total_max starts at -inf: a live iteration always clears the
                # threshold (real - (-inf) = +inf), a fully-masked one never does
                # (-inf - x = -inf or NaN; ordered > is false for both).
                update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                # Every row in this interior tile is live, so the updated
                # total_max is finite even if all preceding edge tiles were dead.
                new_total_max_safe = total_max
                alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
                total_max_safe = new_total_max_safe
                if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                elif cutlass.const_expr(_E5_STYLE_SOFTMAX or _SWA_REUSE_P):
                    exchange_base = parity_rt * cutlass.Int32(CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                else:
                    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_P_0a = _exp2_chunk0a_mixed(reg_S[0:P_SUBCHUNK].vec, False)
                if cutlass.const_expr(not _E5_STYLE_SOFTMAX):
                    hoisted_sum = row_reduction_pair(chunk_P_0a)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0a.to(STORAGE_DTYPE))
                if cutlass.const_expr(_E5_STYLE_SOFTMAX):
                    hoisted_sum = row_reduction_pair(chunk_P_0a)
                if cutlass.const_expr(not _E5_STYLE_SOFTMAX):
                    chunk_P_0b = _exp2_chunk1b_mixed(reg_S[P_SUBCHUNK:CHUNK].vec)
                else:
                    chunk_P_0b = cute.math.exp2(reg_S[P_SUBCHUNK:CHUNK].vec, fastmath=True)
                if cutlass.const_expr(not _E5_STYLE_SOFTMAX):
                    hoisted_sum = hoisted_sum + row_reduction_pair(chunk_P_0b)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_SUBCHUNK), cutlass.Float32),
                    chunk_P_0b.to(STORAGE_DTYPE),
                )
                if cutlass.const_expr(_E5_STYLE_SOFTMAX):
                    hoisted_sum = hoisted_sum + row_reduction_pair(chunk_P_0b)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_sum_1 = None
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.FUSED_CORR_SPLIT_P):
                    chunk_P_1a = cute.math.exp2(reg_S[CHUNK : CHUNK + P_SUBCHUNK].vec, fastmath=True)
                    deferred_sum_1 = row_reduction_pair(chunk_P_1a)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32),
                        chunk_P_1a.to(STORAGE_DTYPE),
                    )
                    chunk_P_1b = _exp2_chunk1b_mixed(
                        reg_S[CHUNK + P_SUBCHUNK : 2 * CHUNK].vec,
                        softmax_half=2,
                    )
                    deferred_sum_1 = deferred_sum_1 + row_reduction_pair(chunk_P_1b)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK + P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_1b.to(STORAGE_DTYPE),
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.FUSED_CORR_SPLIT_P):
                    new_p_sum_pair = new_p_sum_pair + deferred_sum_1
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                tail_active = True
                if tail_active:
                    parity_rt = kv_loop & cutlass.Int32(1)
                    parity_is_even = parity_rt == cutlass.Int32(0)
                    s_off_rt = cutlass.Int32(
                        arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                    )
                    p_off_rt = cutlass.Int32(
                        arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                    )
                    bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                    bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                    bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                    s_addr_base = tmem_base + s_off_rt
                    p_addr_base = tmem_base + p_off_rt
                    stats_addr = tmem_base + s_off_rt
                    kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                    raw_chunks = [
                        nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK)
                        for c in range(N_CHUNKS)
                    ]
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    if cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 1 or softmax_half == 0):
                        if nvvm.elect_sync():
                            nvvm.mbarrier_arrive(mb_sf_reuse.subview(parity_rt))
                    if cutlass.const_expr(bottom_right_diagonal):
                        mask_bottom_right = 0
                        causal_diag = None
                    else:
                        mask_bottom_right = CFG.BOTTOM_RIGHT
                        causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
                    mask_q_abs = q_abs
                    mask_flags = CFG.MASK_FLAGS
                    if cutlass.const_expr(_PADDED_TOP_LEFT_CAUSAL):
                        mask_q_abs = cute.math.min(q_abs, eff_seqlen_kv - cutlass.Int32(1))
                        mask_flags = MASK_CAUSAL
                    chunks_S = [
                        apply_mask_chunk(
                            raw_chunks[c],
                            mask_q_abs - (kv_col_base + cutlass.Int32(c * CHUNK)),
                            cutlass.Int32(0),
                            eff_seqlen_kv - (kv_col_base + cutlass.Int32(c * CHUNK)),
                            CFG.WINDOW_LEFT,
                            mask_flags,
                            N=CHUNK,
                            bottom_right=mask_bottom_right,
                            causal_diag=causal_diag,
                            window_right=CFG.WINDOW_RIGHT,
                            mask_value=float("-inf"),
                        )
                        for c in range(N_CHUNKS)
                    ]
                    reg_S_vec = vec_concat(chunks_S)
                    current_max_unscaled = row_max_reduction(reg_S_vec)
                    reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                    current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

                    # total_max starts at -inf: a live iteration always clears the
                    # threshold (real - (-inf) = +inf), a fully-masked one never does
                    # (-inf - x = -inf or NaN; ordered > is false for both).
                    update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                    total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                    # Canonical 0-substitution at point of use, on BOTH alpha operands;
                    # min(., 0) guards the dead->alive drop of the safe max (total_sum
                    # is still 0 there).  total_max_safe starts at -inf: iter-0 alpha = 0.
                    new_total_max_safe = row_max_for_exp2(total_max)
                    alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
                    total_max_safe = new_total_max_safe
                    if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
                        exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                        softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                        softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                    elif cutlass.const_expr(_E5_STYLE_SOFTMAX or _SWA_REUSE_P):
                        exchange_base = parity_rt * cutlass.Int32(CFG.TILE_M)
                        softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    else:
                        alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_stat_full.arrive()
                    reg_S = reg_S * scale_log2 - total_max_safe

                    chunk_P_0a = _exp2_chunk0a_mixed(reg_S[0:P_SUBCHUNK].vec, True)
                    hoisted_sum = row_reduction_pair(chunk_P_0a)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0a.to(STORAGE_DTYPE))
                    chunk_P_0b = cute.math.exp2(reg_S[P_SUBCHUNK:CHUNK].vec, fastmath=True)
                    hoisted_sum = hoisted_sum + row_reduction_pair(chunk_P_0b)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_0b.to(STORAGE_DTYPE),
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                    deferred_sum_1 = None
                    if cutlass.const_expr(N_CHUNKS == 2 and not CFG.FUSED_CORR_SPLIT_P):
                        chunk_P_1a = cute.math.exp2(reg_S[CHUNK : CHUNK + P_SUBCHUNK].vec, fastmath=True)
                        deferred_sum_1 = row_reduction_pair(chunk_P_1a)
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32),
                            chunk_P_1a.to(STORAGE_DTYPE),
                        )
                        chunk_P_1b = _exp2_chunk1b_mixed(reg_S[CHUNK + P_SUBCHUNK : 2 * CHUNK].vec)
                        deferred_sum_1 = deferred_sum_1 + row_reduction_pair(chunk_P_1b)
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK + P_COLS_PER_SUBCHUNK), cutlass.Float32),
                            chunk_P_1b.to(STORAGE_DTYPE),
                        )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                        bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                    new_p_sum_pair = hoisted_sum
                    if cutlass.const_expr(N_CHUNKS == 2 and not CFG.FUSED_CORR_SPLIT_P):
                        new_p_sum_pair = new_p_sum_pair + deferred_sum_1
                    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                    total_sum = total_sum * alpha_pair + new_p_sum_pair
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)

        total_sum_scalar = total_sum[0] + total_sum[1]
        if cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 2):
            tail_idx = cutlass.Int32(4 * CFG.TILE_M + softmax_half * CFG.TILE_M) + tid_in_wg
            peer_tail_idx = cutlass.Int32(4 * CFG.TILE_M + (1 - softmax_half) * CFG.TILE_M) + tid_in_wg
            softmax_exchange.subview(tail_idx).store(total_sum_scalar)
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)
            total_sum_scalar = total_sum_scalar + softmax_exchange.subview(peer_tail_idx).load()
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

        if cutlass.const_expr(softmax_half == 0 and not CFG.FUSED_CORR_SPLIT_P):
            stats_addr_epi = tmem_base + cutlass.Int32(LAYOUT.STATS_EPI_OFF)
            stats_vec_epi = cutlass.Vector.from_elements((total_max_safe, total_sum_scalar), cutlass.Float32)
            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            bars.mb_stat_full.arrive()

        if cutlass.const_expr(_SWA_REUSE_P):
            # Correction opens the replay only after the first output slice has
            # left TMEM.  P is already materialized, so these arrivals replace
            # the softmax half of the normal BMM2-ready handshake.
            wait(mb_qk_sf_reuse.subview(0), reuse_gate_phase)
            reuse_gate_phase = reuse_gate_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                for chunk_id in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].arrive(
                        leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA
                    )

        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)
        payload_base = sched_state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load())
        q_super_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(3)).load())
        batch_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(5)).load())
        if cutlass.const_expr(SPLIT_KV > 1):
            split_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(10)).load())
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0 or SPLIT_KV > 1):
            bounds = KvLoopBounds(
                left=cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(6)).load()),
                unmasked_lo=cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(7)).load()),
                unmasked_hi=cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(8)).load()),
                right=cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(9)).load()),
            )


@cute.jit
def _emit_output_slice(
    tmem_base_epi,
    sO,
    bars,
    epilogue_state,
    inv_sum,
    row_valid,
    row_dead,
    has_kv,
    tid_in_wg,
    bottom_right_diagonal: cutlass.Constexpr[bool],
    amax_o_tensor,
):
    o_chunk_width = 16
    o_epilogue_block = 64 // CFG.BPE_O
    blocks = CFG.TILE_O // o_epilogue_block
    chunks_per_block = o_epilogue_block // o_chunk_width
    tma_o_iters = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    d_block_size = CFG.TILE_O // tma_o_iters
    tma_o_granularity = CFG.TILE_M * d_block_size
    sO_base = sO[0].base
    amax_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
    amax_local = cutlass.Float32(0.0)

    for block_idx in cutlass.range_constexpr(blocks):
        o_chunks = None
        if cutlass.const_expr(_E5_STYLE_SOFTMAX and CFG.MASK_FLAGS != 0):
            o_chunks = [
                nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(
                        tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + (block_idx * chunks_per_block + sub) * o_chunk_width),
                        cutlass.Float32,
                    ),
                    num=o_chunk_width,
                )
                for sub in range(chunks_per_block)
            ]
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)

        for sub in cutlass.range_constexpr(chunks_per_block):
            chunk_idx = block_idx * chunks_per_block + sub
            o_out = cutlass.Vector.from_elements(
                tuple(OUT_STORAGE_DTYPE(0.0) for _ in range(o_chunk_width)),
                OUT_STORAGE_DTYPE,
            )
            if cutlass.const_expr(not _PADDED_TOP_LEFT_CAUSAL) or has_kv:
                if cutlass.const_expr(_E5_STYLE_SOFTMAX and CFG.MASK_FLAGS != 0):
                    o_chunk = o_chunks[sub]
                else:
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * o_chunk_width),
                            cutlass.Float32,
                        ),
                        num=o_chunk_width,
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_chunk * inv_sum
                if cutlass.const_expr(
                    not _PADDED_TOP_LEFT_CAUSAL and (CFG.SEQ_KV_LENS_PRESENT or SPLIT_KV > 1 or (CFG.BOTTOM_RIGHT and not bottom_right_diagonal))
                ):
                    zero = cutlass.Float32(0.0)
                    invalid = row_dead
                    if cutlass.const_expr(CFG.THD_VARLEN):
                        invalid = invalid | (~row_valid)
                    o_scaled = cutlass.Vector.from_elements(
                        tuple(cutlass.Float32(arith.select(invalid.ir_value(), zero.ir_value(), o_scaled[i].ir_value())) for i in range(o_chunk_width)),
                        cutlass.Float32,
                    )
                amax_local = cute.math.max(amax_local, _max_abs_reduction(o_scaled), ftz=True)
                o_out = o_scaled.to(OUT_STORAGE_DTYPE)

            col_offset = (chunk_idx * o_chunk_width) % d_block_size
            block_offset = ((chunk_idx * o_chunk_width) // d_block_size) * tma_o_granularity
            smem_offset = cutlass.Int32(block_offset + col_offset) + tid_in_wg * cutlass.Int32(d_block_size)
            if block_idx == 0 and sub == 0:
                bars.mb_o_empty.wait(epilogue_state)
            sO_base.subview(smem_offset).data_ptr().store_swizzled(
                o_out,
                alignment=64,
                swizzle=_O_SMEM_SWIZZLE,
            )

        if cutlass.const_expr((block_idx % 2 == 1) or (CFG.TILE_O == o_epilogue_block)):
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_o_full[block_idx // 2].arrive()

    if cutlass.const_expr(SPLIT_KV == 1):
        amax_valid = cutlass.Float32(
            arith.select(
                row_valid.ir_value(),
                amax_local.ir_value(),
                cutlass.Float32(0.0).ir_value(),
            )
        )
        amax_warp = cute.arch.warp_redux_sync(amax_valid, kind="fmax")
        if (tid_in_wg & cutlass.Int32(31)) == cutlass.Int32(0):
            nvvm.atomicrmw(nvvm.AtomicOp.MAX, amax_ptr, amax_warp.bitcast(cutlass.Int32))


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    scale_log2,
    sO,
    tmem_ptr_i32,
    tidx,
    bars,
    sched,
    mb_decoded,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
    bottom_right_diagonal: cutlass.Constexpr[bool],
    amax_o_tensor,
    softmax_exchange,
    mb_sf_reuse,
    mb_qk_sf_reuse,
):
    if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
        nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    else:
        nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))
    tmem_base_corr = tmem_ptr_i32.load()

    tid_raw = cute.arch.thread_idx()[0]
    role_base = CFG.SOFTMAX_WG1_BASE if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P) else CFG.CORR_WARP_BASE
    tid_in_wg = tid_raw - cutlass.Int32(role_base * 32)

    bmm2_done_phase_pair = cutlass.Int32(0)
    bmm1_done_phase_pair = cutlass.Int32(0)
    stat_mbar_state = cutlass.Int32(0)
    epilogue_state = cutlass.Int32(1)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        CFG.QH_PER_KH,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
    bounds = _bounds_for_tile_split(
        q_super_idx,
        eff_seqlen_q,
        eff_seqlen_kv,
        cta_in_pair,
        seq_q_lens_tensor,
        batch_idx,
        split_idx,
        CFG.QH_PER_KH,
    )

    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS_LOCAL = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS_LOCAL
    TMA_O_GRANU_ELEMS_LOCAL = CFG.TILE_M * D_BLOCK_SIZE

    P_SUBCHUNK = 32
    P_COLS_PER_CHUNK = 16
    P_COLS_PER_SUBCHUNK = 8

    def _fused_p1_step(
        kv_loop,
        masked,
        total_max_safe,
        total_sum_pair,
        bmm1_phase_pair,
        bmm2_phase_pair,
        stat_phase,
        q_abs,
        eff_seqlen_kv,
        n_chunks_o,
        o_chunk_size,
        p_subchunk,
        p_cols_per_chunk,
        p_cols_per_subchunk,
        bars_arg,
        mb_qk_sf_reuse_arg,
        tmem_base_arg,
        exchange_arg,
        tid_in_wg_arg,
        bounds_left,
        leader_cta_id_arg,
        scale_log2_arg,
    ):
        parity_cur_rt = kv_loop & cutlass.Int32(1)
        parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
        parity_is_even = parity_cur_rt == cutlass.Int32(0)
        s_off_rt = cutlass.Int32(
            arith.select(
                parity_is_even.ir_value(),
                cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
            )
        )
        p_off_rt = cutlass.Int32(
            arith.select(
                parity_is_even.ir_value(),
                cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
            )
        )
        bmm1_phase = (bmm1_phase_pair >> parity_cur_rt) & cutlass.Int32(1)
        bars_arg.mb_bmm1_done[parity_cur_rt].wait(bmm1_phase)
        bmm1_phase_pair = bmm1_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)

        s_addr_base = tmem_base_arg + s_off_rt
        p_addr_base = tmem_base_arg + p_off_rt
        raw_hi = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(64), cutlass.Float32),
            num=64,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        if nvvm.elect_sync():
            nvvm.mbarrier_arrive(mb_qk_sf_reuse_arg.subview(parity_cur_rt))
        if cutlass.const_expr(masked):
            kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
            mask_q_abs = q_abs - (kv_col_base + cutlass.Int32(64))
            mask_seq_kv = eff_seqlen_kv - (kv_col_base + cutlass.Int32(64))
            mask_flags = CFG.MASK_FLAGS
            if cutlass.const_expr(_PADDED_TOP_LEFT_CAUSAL):
                mask_q_abs = cute.math.min(mask_q_abs, mask_seq_kv - cutlass.Int32(1))
                mask_flags = MASK_CAUSAL
            raw_hi = apply_mask_chunk(
                raw_hi,
                mask_q_abs,
                cutlass.Int32(0),
                mask_seq_kv,
                CFG.WINDOW_LEFT,
                mask_flags,
                N=64,
                bottom_right=0,
                causal_diag=None,
                window_right=CFG.WINDOW_RIGHT,
                mask_value=float("-inf"),
            )
        reg_S_half = RegTile(raw_hi, size=64)

        bars_arg.mb_stat_full.wait(stat_phase)
        exchange_base = parity_cur_rt * cutlass.Int32(2 * CFG.TILE_M)
        alpha = exchange_arg.subview(exchange_base + tid_in_wg_arg).load()
        total_max_safe = exchange_arg.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg_arg).load()
        bars_arg.mb_stat_empty.arrive()
        stat_phase = stat_phase ^ cutlass.Int32(1)

        has_previous_o = kv_loop != bounds_left
        if has_previous_o:
            all_alpha_one = vote_sync(0xFFFFFFFF, alpha == cutlass.Float32(1.0), VoteSync.ALL)
            bmm2_done_phase_prev = (bmm2_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
            # Each iteration must consume its BMM2 phase before O can be reused.
            bars_arg.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev)
            if ~all_alpha_one:
                for chunk_idx in cutlass.range_constexpr(n_chunks_o):
                    o_addr = tmem_base_arg + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * o_chunk_size)
                    o_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=o_chunk_size,
                    )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        vec_scale_pair(o_vec, alpha, o_chunk_size),
                    )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            bmm2_phase_pair = bmm2_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)

        bars_arg.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(
            leader_cta_id=leader_cta_id_arg,
            cta_group=CFG.CTA_MMA,
        )

        reg_S_half = reg_S_half * scale_log2_arg - total_max_safe
        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            chunk_P_a = _exp2_chunk0a_mixed(
                reg_S_half[0:p_subchunk].vec,
                False,
                dense=True,
                softmax_half=1,
            )
        else:
            chunk_P_a = cute.math.exp2(reg_S_half[0:p_subchunk].vec, fastmath=True)
        new_p_sum_pair = row_reduction_pair(chunk_P_a)
        nvvm.tcgen05_st(
            "32x32b",
            nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(p_cols_per_chunk), cutlass.Float32),
            chunk_P_a.to(STORAGE_DTYPE),
        )
        chunk_P_b = _exp2_chunk1b_mixed(
            reg_S_half[p_subchunk:64].vec,
            dense=CFG.MASK_FLAGS == MASK_NONE,
            softmax_half=1 if CFG.MASK_FLAGS == MASK_NONE else 2,
        )
        new_p_sum_pair = new_p_sum_pair + row_reduction_pair(chunk_P_b)
        nvvm.tcgen05_st(
            "32x32b",
            nvvm.make_tmem_ptr(
                p_addr_base + cutlass.Int32(p_cols_per_chunk + p_cols_per_subchunk),
                cutlass.Float32,
            ),
            chunk_P_b.to(STORAGE_DTYPE),
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars_arg.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(
            leader_cta_id=leader_cta_id_arg,
            cta_group=CFG.CTA_MMA,
        )
        alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
        total_sum_pair = total_sum_pair * alpha_pair + new_p_sum_pair
        return total_max_safe, total_sum_pair, bmm1_phase_pair, bmm2_phase_pair, stat_phase

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        fused_total_max_safe = cutlass.Float32(float("-inf"))
        fused_total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        q_abs = q_row_coord + tid_in_wg

        if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
            if bounds.right <= bounds.left:
                bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                fused_total_max_safe, fused_total_sum, bmm1_done_phase_pair, bmm2_done_phase_pair, stat_mbar_state = _fused_p1_step(
                    kv_loop,
                    False,
                    fused_total_max_safe,
                    fused_total_sum,
                    bmm1_done_phase_pair,
                    bmm2_done_phase_pair,
                    stat_mbar_state,
                    q_abs,
                    eff_seqlen_kv,
                    N_CHUNKS_O,
                    O_CHUNK,
                    P_SUBCHUNK,
                    P_COLS_PER_CHUNK,
                    P_COLS_PER_SUBCHUNK,
                    bars,
                    mb_qk_sf_reuse,
                    tmem_base_corr,
                    softmax_exchange,
                    tid_in_wg,
                    bounds.left,
                    leader_cta_id,
                    scale_log2,
                )
            tail_range = (bounds.unmasked_hi,)
            for kv_loop in tail_range:
                if bounds.unmasked_hi < bounds.right:
                    fused_total_max_safe, fused_total_sum, bmm1_done_phase_pair, bmm2_done_phase_pair, stat_mbar_state = _fused_p1_step(
                        kv_loop,
                        True,
                        fused_total_max_safe,
                        fused_total_sum,
                        bmm1_done_phase_pair,
                        bmm2_done_phase_pair,
                        stat_mbar_state,
                        q_abs,
                        eff_seqlen_kv,
                        N_CHUNKS_O,
                        O_CHUNK,
                        P_SUBCHUNK,
                        P_COLS_PER_CHUNK,
                        P_COLS_PER_SUBCHUNK,
                        bars,
                        mb_qk_sf_reuse,
                        tmem_base_corr,
                        softmax_exchange,
                        tid_in_wg,
                        bounds.left,
                        leader_cta_id,
                        scale_log2,
                    )

        if cutlass.const_expr(not CFG.FUSED_CORR_SPLIT_P) and bounds.right > bounds.left:
            lo_parity_rt = bounds.left & cutlass.Int32(1)
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            bars.mb_stat_full.wait(stat_mbar_state)
            bars.mb_stat_empty.arrive()
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)
        elif cutlass.const_expr(not CFG.FUSED_CORR_SPLIT_P):
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        correction_loop_start = bounds.right if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P) else bounds.left + cutlass.Int32(1)
        for kv_loop in cutlass.range(correction_loop_start, bounds.right, 1, unroll=1):
            parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
            parity_cur_rt = kv_loop & cutlass.Int32(1)
            tmem_base_iter = tmem_base_corr

            bars.mb_stat_full.wait(stat_mbar_state)
            if cutlass.const_expr(_E5_STYLE_SOFTMAX or _SWA_REUSE_P):
                exchange_stride = cutlass.Int32(2 * CFG.TILE_M if CFG.MASK_FLAGS == MASK_NONE else CFG.TILE_M)
                exchange_base = parity_cur_rt * exchange_stride
                alpha = softmax_exchange.subview(exchange_base + tid_in_wg).load()
            else:
                stats_off_cur = cutlass.Int32(
                    arith.select(
                        (parity_cur_rt == cutlass.Int32(0)).ir_value(),
                        cutlass.Int32(LAYOUT.STATS_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.STATS_ODD_OFF).ir_value(),
                    )
                )
                stats_addr = tmem_base_iter + stats_off_cur
                stats_vec = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                    num=2,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                alpha = stats_vec[0]

            alpha_is_one = alpha == cutlass.Float32(1.0)
            all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

            bars.mb_stat_empty.arrive()

            bmm2_done_phase_prev = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
            # Each iteration must consume its BMM2 phase before O can be reused.
            bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev)
            if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE):
                if ~all_alpha_one:
                    for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr = tmem_base_iter + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            else:
                if ~all_alpha_one:
                    for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr = tmem_base_iter + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)

            bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        tmem_base_epi = tmem_base_corr

        total_max_scaled = cutlass.Float32(0.0)
        total_sum = cutlass.Float32(0.0)
        if cutlass.const_expr(CFG.FUSED_CORR_SPLIT_P):
            total_max_scaled = fused_total_max_safe
            fused_sum_scalar = fused_total_sum[0] + fused_total_sum[1]
            tail_idx = cutlass.Int32(5 * CFG.TILE_M) + tid_in_wg
            peer_tail_idx = cutlass.Int32(4 * CFG.TILE_M) + tid_in_wg
            softmax_exchange.subview(tail_idx).store(fused_sum_scalar)
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)
            total_sum = fused_sum_scalar + softmax_exchange.subview(peer_tail_idx).load()
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)
        else:
            # The softmax producer publishes epilogue stats even for empty tiles,
            # keeping the phase protocol aligned across persistent work items.
            bars.mb_stat_full.wait(stat_mbar_state)
            stats_addr_epi = tmem_base_epi + cutlass.Int32(LAYOUT.STATS_EPI_OFF)
            stats_vec_epi = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            total_max_scaled = stats_vec_epi[0]
            total_sum = stats_vec_epi[1]
            bars.mb_stat_empty.arrive()
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        LN2 = cutlass.Float32(0.6931471805599453)
        total_max_nat = total_max_scaled * LN2
        lse_val = cutlass.Float32(0.0)
        inv_sum = cutlass.Float32(0.0)
        beta = cutlass.Float32(0.0)
        row_dead = total_sum <= cutlass.Float32(0.0)
        if cutlass.const_expr(CFG.HAS_SINK):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = sinks_arr[head_idx]
            new_max = cute.math.max(total_max_nat, sink_logit)
            scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
            new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_val = new_max + cute.math.log(new_sum, fastmath=True)
            beta = scale / new_sum
            inv_sum = beta
        else:
            lse_val = total_max_nat + cute.math.log(cute.math.max(total_sum, cutlass.Float32(1e-30)), fastmath=True)
            beta = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
            inv_sum = beta
            # Dead row (no valid KV column, incl. empty tiles where total_sum defaults 0):
            #   O := 0, LSE := -inf. total_sum >= 1 for any alive row so this never fires
            #   spuriously. Skipped under sink (the sink path defines a finite LSE).
            neg_inf_lse = cutlass.Float32(float("-inf"))
            lse_val = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
            beta = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), beta.ir_value()))

        q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + tid_in_wg
        if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
            # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b] write
            # O := 0 / LSE := -inf.  Applied AFTER the sink branch on purpose —
            # a trimmed row is dead even with a sink.  Per-batch q lens come in
            # via the dedicated seq_q_lens_tensor parameter.
            _sq_arr = cutlass.make_array_view(seq_q_lens_tensor)
            _q_len_b = cutlass.Int32(_sq_arr[batch_idx])
            row_trim = q_row_global >= _q_len_b
            neg_inf_trim = cutlass.Float32(float("-inf"))
            lse_val = cutlass.Float32(arith.select(row_trim.ir_value(), neg_inf_trim.ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
            beta = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), beta.ir_value()))

        if cutlass.const_expr(CFG.THD_VARLEN):
            cu = cutlass.make_array_view(seq_kv_lens_tensor)
            cu_q_b = cutlass.Int32(cu[n_batch + batch_idx])
            s_q_b = cutlass.Int32(cu[n_batch + batch_idx + cutlass.Int32(1)]) - cu_q_b
            _row_valid = q_row_global < s_q_b
            if cutlass.const_expr(lse_tensor is not None):
                if _row_valid:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    if cutlass.const_expr(len(lse_tensor.shape) == 2):
                        lse_arr[cu_q_b + q_row_global, head_idx] = lse_val
                    else:
                        lse_arr[cutlass.Int32(0), head_idx, cu_q_b + q_row_global] = lse_val
        else:
            _row_valid = q_row_global < seqlen_q
            if cutlass.const_expr(lse_tensor is not None):
                if _row_valid:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                    lse_arr[lse_batch, head_idx, q_row_global] = lse_val

        parity_last_rt = cutlass.Int32(0)
        if bounds.right > bounds.left:
            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
        bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
        bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last)
        bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

        _emit_output_slice(
            tmem_base_epi,
            sO,
            bars,
            epilogue_state,
            inv_sum,
            _row_valid,
            row_dead,
            bounds.right > bounds.left,
            tid_in_wg,
            bottom_right_diagonal,
            amax_o_tensor,
        )

        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        if cutlass.const_expr(_SWA_REUSE_P):
            # Start the second Dv256 pass only after correction has consumed the
            # first output.  The first KV tile needs no accumulator rescale.
            lo_parity_rt = bounds.left & cutlass.Int32(1)
            if nvvm.elect_sync():
                nvvm.mbarrier_arrive(mb_sf_reuse.subview(lo_parity_rt))
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(
                leader_cta_id=leader_cta_id,
                cta_group=CFG.CTA_MMA,
            )
            if nvvm.elect_sync():
                nvvm.mbarrier_arrive(mb_qk_sf_reuse.subview(0))

            # For each later KV tile, replay the same online-softmax alpha before
            # admitting its retained P tile to BMM2.
            for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
                parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                bmm2_done_phase_prev = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
                bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev)

                if cutlass.const_expr(_E5_STYLE_SOFTMAX or _SWA_REUSE_P):
                    exchange_stride = cutlass.Int32(CFG.TILE_M)
                    alpha = softmax_exchange.subview(parity_cur_rt * exchange_stride + tid_in_wg).load()
                else:
                    stats_off_cur = cutlass.Int32(
                        arith.select(
                            (parity_cur_rt == cutlass.Int32(0)).ir_value(),
                            cutlass.Int32(LAYOUT.STATS_EVEN_OFF).ir_value(),
                            cutlass.Int32(LAYOUT.STATS_ODD_OFF).ir_value(),
                        )
                    )
                    alpha_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(tmem_base_corr + stats_off_cur, cutlass.Float32),
                        num=2,
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    alpha = alpha_vec[0]

                all_alpha_one = vote_sync(0xFFFFFFFF, alpha == cutlass.Float32(1.0), VoteSync.ALL)
                if ~all_alpha_one:
                    for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr = tmem_base_corr + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            vec_scale_pair(o_chunk, alpha, O_CHUNK),
                        )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)

                if nvvm.elect_sync():
                    nvvm.mbarrier_arrive(mb_sf_reuse.subview(parity_cur_rt))
                bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(
                    leader_cta_id=leader_cta_id,
                    cta_group=CFG.CTA_MMA,
                )

            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
            bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
            bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last)
            bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

            _emit_output_slice(
                tmem_base_epi,
                sO,
                bars,
                epilogue_state,
                inv_sum,
                _row_valid,
                row_dead,
                bounds.right > bounds.left,
                tid_in_wg,
                bottom_right_diagonal,
                amax_o_tensor,
            )
            epilogue_state = epilogue_state ^ cutlass.Int32(1)

        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)
        payload_base = sched_state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        nxt_v = sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load()
        q_super_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(3)).load()
        head_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(4)).load()
        batch_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(5)).load()
        if cutlass.const_expr(SPLIT_KV > 1):
            split_idx = sched.tile_id_smem.subview(payload_base + cutlass.Int32(10)).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0 or SPLIT_KV > 1):
            bounds = KvLoopBounds(
                left=sched.tile_id_smem.subview(payload_base + cutlass.Int32(6)).load(),
                unmasked_lo=sched.tile_id_smem.subview(payload_base + cutlass.Int32(7)).load(),
                unmasked_hi=sched.tile_id_smem.subview(payload_base + cutlass.Int32(8)).load(),
                right=sched.tile_id_smem.subview(payload_base + cutlass.Int32(9)).load(),
            )

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta)
    bars.mb_tmem_dealloc.arrive()


@cute.jit
def _host(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    sf_q_tensor: cute.Tensor,
    sf_k_tensor: cute.Tensor,
    sf_v_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    amax_o_tensor: cute.Tensor,
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    # Dense padded-Q trim: separate (B,)-int32 lengths. None folds the
    # parameter and all consumers out when the specialization is disabled.
    seq_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(CFG.THD_VARLEN):
        SQ = q_tensor.shape[1]
        SKV = k_tensor.shape[1]

    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    q_rank5_layout = cute.make_layout(
        (q_tensor.shape[0], q_tensor.shape[2], TMA_QK_ITERS, q_tensor.shape[1], TMA_QK_GRANU_ELEMS),
        stride=(
            q_tensor.shape[1] * q_tensor.shape[2] * CFG.TILE_K,
            CFG.TILE_K,
            TMA_QK_GRANU_ELEMS,
            q_tensor.shape[2] * CFG.TILE_K,
            1,
        ),
    )
    q_rank5_tensor = cute.make_tensor(q_tensor.iterator, q_rank5_layout)
    k_rank5_layout = cute.make_layout(
        (k_tensor.shape[0], k_tensor.shape[2], TMA_QK_ITERS, k_tensor.shape[1], TMA_QK_GRANU_ELEMS),
        stride=(
            k_tensor.shape[1] * k_tensor.shape[2] * CFG.TILE_K,
            CFG.TILE_K,
            TMA_QK_GRANU_ELEMS,
            k_tensor.shape[2] * CFG.TILE_K,
            1,
        ),
    )
    k_rank5_tensor = cute.make_tensor(k_tensor.iterator, k_rank5_layout)
    v_rank5_layout = cute.make_layout(
        (v_tensor.shape[0], v_tensor.shape[2], FULL_DV // TMA_VO_GRANU_ELEMS, v_tensor.shape[1], TMA_VO_GRANU_ELEMS),
        stride=(
            v_tensor.shape[1] * v_tensor.shape[2] * FULL_DV,
            FULL_DV,
            TMA_VO_GRANU_ELEMS,
            v_tensor.shape[2] * FULL_DV,
            1,
        ),
    )
    v_rank5_tensor = cute.make_tensor(v_tensor.iterator, v_rank5_layout)
    qk_box_k = (1, 1, TMA_QK_ITERS, CFG.TILE_N // CFG.CTA_MMA, TMA_QK_GRANU_ELEMS)
    qk_box_q = (1, 1, TMA_QK_ITERS, CFG.TILE_M, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, 1, TMA_VO_ITERS, CFG.TILE_N, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    input_l2_promotion = tmap.TensorMapL2Promotion.l2_256b
    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_rank5_tensor,
        box_dims=qk_box_q,
        stride_order=(4, 3, 2, 1, 0),
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=input_l2_promotion,
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_rank5_tensor,
        box_dims=qk_box_k,
        stride_order=(4, 3, 2, 1, 0),
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=input_l2_promotion,
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_rank5_tensor,
        box_dims=vo_box_v,
        stride_order=(4, 3, 2, 1, 0),
        swizzle=_tma_swz(CFG.V_SWZ_BYTES),
        l2_promotion=input_l2_promotion,
    )
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=vo_box_o,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    SF_TMA_ROW_BYTES = 128
    SF_NUM_ROWS_Q = SF_SMEM_SIZE_Q // SF_TMA_ROW_BYTES
    SF_NUM_ROWS_K = SF_SMEM_SIZE_K // SF_TMA_ROW_BYTES
    if cutlass.const_expr(CFG.THD_VARLEN):
        b_sf = 1
        q_sf_num_tiles = sf_q_tensor.shape[2]
        kv_sf_num_tiles = sf_k_tensor.shape[2]
    else:
        b_sf = B
        q_sf_num_tiles = (SQ + CFG.TILE_M - 1) // CFG.TILE_M
        kv_sf_num_tiles = (SKV + CFG.TILE_N - 1) // CFG.TILE_N

    def _build_sf_desc(sf_tensor, num_tiles, sf_smem_size, num_rows_box, num_heads, base_offset=0):
        sf_base = cutlass.Int64(sf_tensor.iterator.toint()) + cutlass.Int64(base_offset)
        tile_stride_16 = sf_smem_size // 16
        return tmap.create_tensor_map_tiled(
            global_address=sf_base,
            dtype=cutlass.Uint8,
            global_dims=[
                SF_TMA_ROW_BYTES,
                sf_smem_size // SF_TMA_ROW_BYTES,
                num_tiles,
                num_heads,
                b_sf,
            ],
            global_strides=[
                SF_TMA_ROW_BYTES // 16,
                tile_stride_16,
                num_tiles * tile_stride_16,
                num_heads * num_tiles * tile_stride_16,
            ],
            box_dims=[SF_TMA_ROW_BYTES, num_rows_box, 1, 1, 1],
            swizzle=tmap.TensorMapSwizzle.none,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_q_sf_desc = _build_sf_desc(sf_q_tensor, q_sf_num_tiles, SF_SMEM_SIZE_Q, SF_NUM_ROWS_Q, QH)
    tma_k_sf_desc = _build_sf_desc(sf_k_tensor, kv_sf_num_tiles, SF_SMEM_SIZE_K, SF_NUM_ROWS_K, KH)
    v_sf_groups = b_sf * KH * kv_sf_num_tiles
    v_sf_plane_bytes = v_sf_groups * SF_BYTES_PER_BLOCK
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD packs both D/128 planes for each (head, sequence tile)
        # contiguously. Dense MXFP8 keeps each plane contiguous across all
        # groups, so its plane stride cannot be reused here.
        v_sf_strides = [
            SF_TMA_ROW_BYTES // 16,
            SF_BYTES_PER_BLOCK // 16,
            SF_SMEM_SIZE_V // 16,
        ]
    else:
        v_sf_strides = [
            SF_TMA_ROW_BYTES // 16,
            v_sf_plane_bytes // 16,
            SF_BYTES_PER_BLOCK // 16,
        ]
    tma_v_sf_desc = tmap.create_tensor_map_tiled(
        global_address=sf_v_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[SF_TMA_ROW_BYTES, SF_BYTES_PER_BLOCK // SF_TMA_ROW_BYTES, FULL_DV // SF_BLOCK_DIM_NON_K, v_sf_groups],
        global_strides=v_sf_strides,
        box_dims=[SF_TMA_ROW_BYTES, SF_BYTES_PER_BLOCK // SF_TMA_ROW_BYTES, SF_NUM_BLOCKS_V, 1],
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        _build_thd_meta_o_kv_descs_kernel(
            o_tensor,
            tma_o_desc,
            tma_k_desc,
            tma_v_desc,
            o_desc_words,
            seq_kv_lens_tensor,
            thd_q_lens_tensor,
            thd_kv_lens_tensor,
            thd_lens_form,
            cutlass.Int32(QH),
            cutlass.Int32(B),
            cutlass.Int32(o_tensor.stride[1]),
            cutlass.Int32(rows_per_cluster),
            n_thd_units,
            1,
            1,
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        grid_shape = (
            (grid_q_supers, QH, B * SPLIT_KV) if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL) else (grid_q_supers * QH * B * SPLIT_KV, 1, 1)
        )
    for dv_slice in cutlass.range_constexpr(1 if _SWA_REUSE_P else OUTPUT_SLICES):
        _kernel(
            tma_q_desc,
            tma_k_desc,
            tma_v_desc,
            tma_o_desc,
            tma_q_sf_desc,
            tma_k_sf_desc,
            tma_v_sf_desc,
            lse_tensor,
            sinks_tensor,
            seq_kv_lens_tensor,
            o_desc_words,
            cutlass.Int32(SQ),
            cutlass.Int32(SKV),
            cutlass.Int32(q_supers),
            cutlass.Int32(QH),
            cutlass.Int32(B),
            cutlass.Int32(QH // KH),
            cutlass.Int32(kv_sf_num_tiles),
            scale_softmax_log2,
            amax_o_tensor,
            cutlass.Int32(dv_slice),
            False,
            seq_q_lens_tensor,
        ).launch(
            grid=grid_shape,
            block=[CFG.THREADS_PER_CTA, 1, 1],
            cluster=(CFG.CTA_MMA, 1, 1),
            stream=stream,
        )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 256,
    skv: int = 128,
    has_lse: bool = True,
    lse_head_major: bool = False,
    lse_head_stride: int = 0,
    lse_stride: Optional[tuple[int, int, int]] = None,
) -> Callable:
    """Compile the exact D512 MXFP8 kernel and its per-tile SF views."""
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_stride is not None and SPLIT_KV > 1:
        raise ValueError("split_kv partial LSE is compact; lse_stride describes only the final combine output")
    fake_batch = 1 if CFG.THD_VARLEN else b
    if CFG.THD_VARLEN:
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)
        q_sf_tiles = cute.sym_int(divisibility=1)
        kv_sf_tiles = cute.sym_int(divisibility=1)
    else:
        q_sf_tiles = (sq + CFG.TILE_M - 1) // CFG.TILE_M
        kv_sf_tiles = (skv + CFG.TILE_N - 1) // CFG.TILE_N

    def _fake(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=16)

    fake_q = _fake(STORAGE_DTYPE, (fake_batch, sq, qh, CFG.TILE_K))
    fake_k = _fake(STORAGE_DTYPE, (fake_batch, skv, kh, CFG.TILE_K))
    fake_v = _fake(STORAGE_DTYPE, (fake_batch, skv, kh, FULL_DV))
    fake_o = _fake(OUT_STORAGE_DTYPE, (fake_batch * SPLIT_KV, sq, qh, FULL_DV))
    fake_sf_q = _fake(cutlass.Int8, (fake_batch, qh, q_sf_tiles, SF_SMEM_SIZE_Q))
    fake_sf_k = _fake(cutlass.Int8, (fake_batch, kh, kv_sf_tiles, SF_SMEM_SIZE_K))
    fake_sf_v = _fake(cutlass.Int8, (fake_batch, kh, kv_sf_tiles, SF_SMEM_SIZE_V))
    if not has_lse:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride require has_lse=True")
        fake_lse = None
    elif CFG.THD_VARLEN:
        if lse_head_major:
            lse_extent = lse_head_stride if lse_head_stride else sq
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (1, qh, lse_extent),
                stride_order=(2, 1, 0),
                assumed_align=4,
            )
        else:
            if lse_head_stride:
                raise ValueError("lse_head_stride is head-major-only")
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (sq, qh),
                stride_order=(1, 0),
                assumed_align=4,
            )
    else:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride are unsupported for dense MXFP8")
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, (b, qh, sq), lse_stride, assumed_align=4)
            if lse_stride is not None
            else _fake(cutlass.Float32, (b * SPLIT_KV, qh, sq))
        )
    fake_sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (qh,),
        stride_order=(0,),
        assumed_align=16,
    )
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        ((4 * b + 4) if CFG.THD_VARLEN else b,),
        stride_order=(0,),
        assumed_align=16,
    )
    fake_amax_o = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        stride_order=(0,),
        assumed_align=16,
    )
    fake_o_desc = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (((b + 3) * _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1,),
        stride_order=(0,),
        assumed_align=16,
    )
    fake_seq_q_lens = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (b,),
            stride_order=(0,),
            assumed_align=4,
        )
        if CFG.SEQ_Q_LENS_PRESENT
        else None
    )
    if CFG.THD_VARLEN:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (cute.sym_int(divisibility=1),),
            stride_order=(0,),
            assumed_align=4,
        )
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (cute.sym_int(divisibility=1),),
            stride_order=(0,),
            assumed_align=4,
        )
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = None
        fake_thd_kv_lens = None
        fake_thd_lens_form = None

    return cute.compile(
        _host,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_sf_q,
        fake_sf_k,
        fake_sf_v,
        fake_lse,
        fake_amax_o,
        fake_sinks,
        fake_seq_kv_lens,
        fake_o_desc,
        (b, qh, kh, 0, 0, 0) if CFG.THD_VARLEN else (b, qh, kh, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        fake_seq_q_lens,
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--hq", type=int, default=1)
    parser.add_argument("--hk", type=int, default=1)
    parser.add_argument("--sq", type=int, default=256)
    parser.add_argument("--skv", type=int, default=128)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()

    print(f"[d512_mxfp8_sm100] compile b={args.b} qh={args.hq} kh={args.hk} " f"sq={args.sq} skv={args.skv}", flush=True)
    fn = compile(args.b, args.hq, args.hk, args.sq, args.skv)
    print(f"[d512_mxfp8_sm100] compile OK: {fn}", flush=True)
    if args.validate:
        print("[d512_mxfp8_sm100] compiled - run validation via the frost SDPA test suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
