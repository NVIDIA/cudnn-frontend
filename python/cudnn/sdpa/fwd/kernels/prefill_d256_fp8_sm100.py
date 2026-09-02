# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 per-tensor FP8 prefill SDPA for exact d_qk=d_v=256.

This is an independent CTA1 pipeline: each CTA owns one M128 query tile and
two full-width KV stages. Depending on the mask specialization, one or two
softmax warp groups generate packed FP8 probabilities for Blackwell's K=32
FP8 QMMA path and the sdpa_fp8 output scaling/amax ABI.

THD / varlen follows the shared device-built-metadata and plan-time-envelope
design used by the D128 FP8 sibling. Packed token totals stay dynamic and the
setup kernel builds per-sequence O plus packed-total-clamped K/V descriptors.
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

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256

# The template loader injects the graph's FP8 parameters before executing this
# module. The default record only keeps import-time tooling well-defined; it is
# not a supported standalone execution configuration for this FP8 kernel.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d256(PARAMS)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    PipelineState,
    Producer,
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
)
from cudnn.block_sparse_attention.csrc.utils.kernel_utils import ex2_emulation_2

if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float8E4M3FN
    P_STORAGE_DTYPE = cutlass.Float8E4M3FN
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.Float8E5M2
    P_STORAGE_DTYPE = cutlass.Float8E5M2
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
else:
    raise ValueError(f"prefill_sdpa_d256_fp8: unsupported DTYPE_QKV={CFG.DTYPE_QKV}")

if CFG.DTYPE_O == 0:
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_O == 1:
    OUT_STORAGE_DTYPE = cutlass.Float8E5M2
elif CFG.DTYPE_O == 2:
    OUT_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_O == 3:
    OUT_STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"prefill_sdpa_d256_fp8: unsupported DTYPE_O={CFG.DTYPE_O}")


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

N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

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

# Pack all query heads sharing one KV head into the tile's row dimension.
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE

from cudnn.sdpa.fwd.kernels.thd_sm100 import (
    build_thd_meta_o_kv_descs_kernel as _build_thd_meta_o_kv_descs_kernel,
    thd_decode_unit,
    TENSOR_MAP_QWORDS,
)

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_PLAIN_TOP_LEFT_CAUSAL = (CFG.MASK_FLAGS & ~MASK_PADDED) == MASK_CAUSAL and CFG.BOTTOM_RIGHT == 0 and CFG.WINDOW_RIGHT == 0
_PADDED_TOP_LEFT_CAUSAL = _PLAIN_TOP_LEFT_CAUSAL and bool(CFG.MASK_FLAGS & MASK_PADDED)
_THD_REVERSE_CAUSAL_ROWS = CFG.THD_VARLEN and _PADDED_TOP_LEFT_CAUSAL


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
    cuq0 = n_batch
    live = batch < n_batch
    safe_batch = cute.math.min(batch, n_batch - cutlass.Int32(1))
    seq_q = cutlass.Int32(meta[cuq0 + safe_batch + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + safe_batch])
    tiles_q = (seq_q + cutlass.Int32(CGA_TILE_M - 1)) // cutlass.Int32(CGA_TILE_M)
    rank = tiles_q - cutlass.Int32(1) - q_tile
    group = head // cutlass.Int32(4)
    group_head0 = group * cutlass.Int32(4)
    group_is_full = live & (group_head0 + cutlass.Int32(3) < n_qh)
    group_linear = (head - group_head0) * tiles_q + rank
    grouped_head = group_head0 + (group_linear & cutlass.Int32(3))
    grouped_rank = group_linear // cutlass.Int32(4)
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


LAYOUT = KernelTmemLayout()
# Slots 6:10 carry predecoded bounds and slot 10 carries the split index.  The
# unsplit/no-mask specialization retains the original eight-word payload.
SCHED_PAYLOAD_WORDS = 12 if (CFG.MASK_FLAGS != 0 or SPLIT_KV > 1) else 8


@cute.jit
def _scheduler_warp_loop_predecode(
    sched,
    mb_decoded,
    is_cga_first_cta,
    cta_in_pair,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
):
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)

        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
        if nvvm.elect_sync() and is_cga_first_cta:
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)),
                sched.mb_scheduler.subview(state.idx),
                multicast=1,
            )

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
                qh_per_kh,
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
                        HEADS_PER_TILE,
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

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2
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


_E5_STYLE_SOFTMAX = CFG.DTYPE_QKV == 1 or (CFG.DTYPE_QKV == 0 and (CFG.MASK_FLAGS == MASK_NONE or _PLAIN_TOP_LEFT_CAUSAL))


def _exp2_dense_chunk_a(vec, softmax_half):
    values = []
    for i in range(0, int(vec.shape[0]), 2):
        use_e4_poly = not _E5_STYLE_SOFTMAX and i % 10 < 4 and not (softmax_half == 1 and i == 30)
        use_e5_poly = _E5_STYLE_SOFTMAX and (
            i % 10 < 4 or (softmax_half == 1 and (i == 24 or i == 26)) or (CFG.DTYPE_QKV == 0 and softmax_half == 0 and (i == 26 or i == 28))
        )
        if use_e4_poly:
            x, y = ex2_emulation_2(vec[i], vec[i + 1])
        elif use_e5_poly:
            x, y = ex2_emulation_2(vec[i], vec[i + 1], poly_degree=1)
        else:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        values.extend((x, y))
    return cutlass.Vector.from_elements(tuple(values), cutlass.Float32)


def _exp2_dense_chunk_b(vec, softmax_half):
    values = []
    for i in range(0, int(vec.shape[0]), 2):
        e4_threshold = 20 if softmax_half == 2 else 16
        e5_threshold = 22 if softmax_half == 3 else (20 if softmax_half == 0 else (22 if softmax_half == 2 else 24))
        if (not _E5_STYLE_SOFTMAX and i >= e4_threshold) or (_E5_STYLE_SOFTMAX and i >= e5_threshold):
            x, y = ex2_emulation_2(vec[i], vec[i + 1], poly_degree=1 if CFG.DTYPE_QKV == 0 else 2)
        else:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        values.extend((x, y))
    return cutlass.Vector.from_elements(tuple(values), cutlass.Float32)


def _exp2_masked_tail_chunk_a(vec, softmax_half):
    values = []
    for i in range(0, int(vec.shape[0]), 2):
        use_poly = i % 10 < 4 or (softmax_half == 1 and (i == 24 or i == 26)) or (softmax_half == 0 and (i == 26 or i == 28))
        if use_poly:
            x, y = ex2_emulation_2(vec[i], vec[i + 1], poly_degree=2)
        else:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        values.extend((x, y))
    return cutlass.Vector.from_elements(tuple(values), cutlass.Float32)


def _exp2_masked_alpha(x):
    if _E5_STYLE_SOFTMAX and _PLAIN_TOP_LEFT_CAUSAL:
        value, _ = ex2_emulation_2(x, x, poly_degree=2)
        return value
    return cute.math.exp2(x, fastmath=True)


_RARE_RESCALE_CORRECTION = _E5_STYLE_SOFTMAX
_STAT_FULL_WARP_ARRIVALS = CFG.SOFTMAX_WARPGROUPS == 2 and _PLAIN_TOP_LEFT_CAUSAL
_STAT_EMPTY_WARP_ARRIVALS = _STAT_FULL_WARP_ARRIVALS
_SMEM_ALPHA_HANDOFF = _STAT_FULL_WARP_ARRIVALS and CFG.DTYPE_QKV == 0


@cute.jit
def _stat_arrive(bar, warp_arrivals: cutlass.Constexpr[bool]):
    if cutlass.const_expr(warp_arrivals):
        if nvvm.elect_sync():
            bar.arrive()
    else:
        bar.arrive()


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
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
    scale_softmax_log2: cutlass.Float32,
    o_scale_fused: cutlass.Float32,
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: cute.Tensor,
    amax_o_tensor: cute.Tensor,
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

    sQ = SmemTile(
        base=sQO_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
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

    bars = make_d256_bars(CFG, N_O_CHUNKS=N_O_CHUNKS)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    # Dense-only WG0->WG1 alpha handoff (two parity slots) followed by one
    # two-way tail denominator exchange.
    softmax_exchange = cutlass.Array(
        cutlass.Float32,
        768 if CFG.SOFTMAX_WARPGROUPS == 2 else 1,
        alignment=16,
        space=cutlass.AddressSpace.smem,
    )
    mb_softmax_max = MBarrier(
        cutlass.Array(cutlass.Int64, 2, alignment=16, space=cutlass.AddressSpace.smem),
        stages=2,
        init_count=CFG.SOFTMAX_LANES,
        producer=Producer.THREAD,
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
            if cutlass.const_expr(_STAT_FULL_WARP_ARRIVALS):
                bars.mb_stat_full.init(override_count=CFG.SOFTMAX_WG_WARPS)
            else:
                bars.mb_stat_full.init()
            if cutlass.const_expr(_STAT_EMPTY_WARP_ARRIVALS):
                bars.mb_stat_empty.init(override_count=CFG.CORRECTION_WARPS)
            else:
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
            if cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 2):
                for p in cutlass.range_constexpr(2):
                    mb_softmax_max[p].init()
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()

            bars.mb_q_o_alias.arrive()

    scale_log2_fused = scale_softmax_log2
    if cutlass.const_expr(not CFG.THD_VARLEN):
        scale_log2_fused = (
            scale_softmax_log2 * cutlass.Float32(cutlass.make_array_view(descale_q_t)[0]) * cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
        )
    o_scale_fused = o_scale_fused * cutlass.Float32(cutlass.make_array_view(descale_v_t)[0]) * cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])

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
        if cutlass.const_expr(CFG.THD_VARLEN):
            scale_log2_fused = (
                scale_softmax_log2 * cutlass.Float32(cutlass.make_array_view(descale_q_t)[0]) * cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
            )
        _softmax_warp_group(
            softmax_half=0,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_log2_fused,
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
            qh_per_kh=qh_per_kh,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            bottom_right_diagonal=bottom_right_diagonal,
            softmax_exchange=softmax_exchange,
            mb_softmax_max=mb_softmax_max,
        )

    elif cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 2) and warp_idx >= CFG.SOFTMAX_WG1_BASE and warp_idx < CFG.SOFTMAX_WG1_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_WG1_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        if cutlass.const_expr(CFG.THD_VARLEN):
            scale_log2_fused = (
                scale_softmax_log2 * cutlass.Float32(cutlass.make_array_view(descale_q_t)[0]) * cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
            )
        _softmax_warp_group(
            softmax_half=1,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_log2_fused,
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
            qh_per_kh=qh_per_kh,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            bottom_right_diagonal=bottom_right_diagonal,
            softmax_exchange=softmax_exchange,
            mb_softmax_max=mb_softmax_max,
        )

    elif warp_idx >= CFG.CORR_WARP_BASE and warp_idx < CFG.CORR_WARP_BASE + CFG.CORRECTION_WARPS:
        nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _correction_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
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
            qh_per_kh=qh_per_kh,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            cta_id_x=cta_id_x,
            bottom_right_diagonal=bottom_right_diagonal,
            o_scale_fused=o_scale_fused,
            amax_o_tensor=amax_o_tensor,
            softmax_exchange=softmax_exchange,
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
                    tmem_ptr_i32=tmem_ptr_i32,
                    bars=bars,
                    sched=sched,
                    mb_decoded=mb_decoded,
                    seq_kv_lens_tensor=seq_kv_lens_tensor,
                    seq_q_lens_tensor=seq_q_lens_tensor,
                    n_q_supers=n_q_supers,
                    n_qh=n_qh,
                    n_batch=n_batch,
                    qh_per_kh=qh_per_kh,
                    mcast_mask=mcast_mask,
                    cta_in_pair=cta_in_pair,
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
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                mb_decoded=mb_decoded,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_tensor=seq_q_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                qh_per_kh=qh_per_kh,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sK=sK,
            sV=sV,
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
            qh_per_kh=qh_per_kh,
            cta_in_pair=cta_in_pair,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
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
            qh_per_kh,
            seqlen_q,
            seqlen_kv,
            seq_kv_lens_tensor,
            seq_q_lens_tensor,
        )


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sK,
    sV,
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
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
):

    q_o_alias_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=1)

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
    q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

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
            HEADS_PER_TILE,
        )
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            # K does not alias the prior tile's O buffer. Issue the first K TMA
            # before waiting for Q/O ownership so its latency overlaps the
            # previous tile's output store and the alias handoff.
            kv_row_base = kv_left * cutlass.Int32(CFG.TILE_N)
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sK[kv_state.idx],
                tma_k(cutlass.Int32(0), kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                bars.mb_k_full[kv_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=False,
            )

            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sV[kv_state.idx],
                tma_v(cutlass.Int32(0), kv_row_base + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                bars.mb_v_full[kv_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=False,
            )
            kv_state = advance(kv_state, CFG.STAGES_KV)

            kv_second = kv_left + cutlass.Int32(1)
            if kv_second < kv_right:
                kv_row_base = kv_second * cutlass.Int32(CFG.TILE_N)
                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[kv_state.idx],
                    tma_k(cutlass.Int32(0), kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=False,
                )

                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[kv_state.idx],
                    tma_v(cutlass.Int32(0), kv_row_base + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=False,
                )
                kv_state = advance(kv_state, CFG.STAGES_KV)

        bars.mb_q_o_alias.wait(q_o_alias_phase)
        q_o_alias_phase = q_o_alias_phase ^ cutlass.Int32(1)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), q_head_idx, q_row_base + q_seq_off, tma_batch),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            kv_main_start = cute.math.min(kv_left + cutlass.Int32(2), kv_right)
            for kv_loop in cutlass.range(kv_main_start, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[kv_state.idx],
                    tma_k(cutlass.Int32(0), kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=False,
                )

                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[kv_state.idx],
                    tma_v(cutlass.Int32(0), kv_row_base + kv_seq_off, cutlass.Int32(0), kv_head_idx, tma_batch),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=False,
                )

                kv_state = advance(kv_state, CFG.STAGES_KV)

        if nvvm.elect_sync():
            bars.mb_tmastg_go.arrive()
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)
        payload_base = sched_state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load())
        q_super_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(3)).load())
        head_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(4)).load())
        batch_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(5)).load())
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0 or SPLIT_KV > 1):
            kv_left = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(6)).load())
            kv_right = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(9)).load())

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            kv_state = advance(kv_state, CFG.STAGES_KV)
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
    qh_per_kh,
    cta_in_pair,
    seq_kv_lens_tensor,
    o_desc_words,
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
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_tmastg_go.wait(tmastg_go_phase)
        tmastg_go_phase = tmastg_go_phase ^ cutlass.Int32(1)

        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        if cutlass.const_expr(CFG.THD_VARLEN):
            o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        else:
            o_slice = tma_o(cutlass.Int32(0), q_head_idx, q_row_coord, o_batch)
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
                        (d_coord, q_head_idx, q_row_coord, cutlass.Int32(0)),
                    )
            else:
                nvvm.cp_async_bulk_tensor_global_shared_cta(
                    o_desc_ptr,
                    smem_chunk,
                    (o_slice.coord_d + d_coord, q_head_idx, q_row_coord, o_batch),
                )
            tma_store_commit()
            if cutlass.const_expr(CFG.BPE_O == 2 and chunk == 1):
                tma_store_wait(0)
                if nvvm.elect_sync():
                    bars.mb_q_o_alias.arrive()

        tma_store_wait(0)
        if cutlass.const_expr(CFG.BPE_O != 2):
            if nvvm.elect_sync():
                bars.mb_q_o_alias.arrive()

        bars.mb_o_empty.arrive()

        o_full_phase = o_full_phase ^ cutlass.Int32(1)

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
    tmem_ptr_i32,
    bars,
    sched,
    mb_decoded,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    mcast_mask,
    cta_in_pair,
):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=0,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=0,
    )
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    desc_Q = sQ[0].desc()

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
            qh_per_kh,
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
                HEADS_PER_TILE,
            )
            kv_left = bounds_init.left
            kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    kv_state_V = PipelineState.start(phase=0)
    bmm2_ready_phase_pair = cutlass.Int32(0)
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
            parity_lo_is_even = lo_parity_runtime == cutlass.Int32(0)
            tmem_S_acc_lo_addr = cutlass.Int32(
                arith.select(
                    parity_lo_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                )
            )

            bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
            desc_K = sK[kv_state_K.idx].desc()
            mma_ss(
                bmm1_desc,
                desc_Q,
                desc_K,
                tmem_raw.subview(tmem_S_acc_lo_addr),
                elect_once=(CFG.DTYPE_QKV == 1 and CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT == 0) or (CFG.DTYPE_QKV == 0 and _PADDED_TOP_LEFT_CAUSAL),
            )
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[lo_parity_runtime].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

            k_per_chunk = NUM_KPHASES_PV_PER_CHUNK

            for kv_loop in cutlass.range(kv_left, kv_right - cutlass.Int32(1), 1, unroll=1):
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                parity_next_rt = (kv_loop + cutlass.Int32(1)) & cutlass.Int32(1)
                cur_is_even = parity_cur_rt == cutlass.Int32(0)
                next_is_even = parity_next_rt == cutlass.Int32(0)

                tmem_S_acc_next_addr = cutlass.Int32(
                    arith.select(
                        next_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                    )
                )
                tmem_P_cur_addr = cutlass.Int32(
                    arith.select(
                        cur_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                    )
                )
                bmm2_ready_phase_cur = (bmm2_ready_phase_pair >> parity_cur_rt) & cutlass.Int32(1)

                bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                desc_K = sK[kv_state_K.idx].desc()
                mma_ss(
                    bmm1_desc,
                    desc_Q,
                    desc_K,
                    tmem_raw.subview(tmem_S_acc_next_addr),
                    elect_once=(CFG.DTYPE_QKV == 1 and CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT == 0)
                    or (CFG.DTYPE_QKV == 0 and _PADDED_TOP_LEFT_CAUSAL),
                )
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[parity_next_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
                desc_V = sV[kv_state_V.idx].desc()

                scaleC = cutlass.Boolean(kv_loop != kv_left)
                accum_b2 = scaleC
                bmm2_issue = nvvm.elect_sync()
                for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                    if k % k_per_chunk == 0:
                        chunk_id = k // k_per_chunk
                        bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_cur)
                    mma_ts_step(
                        bmm2_desc,
                        tmem_raw.subview(tmem_P_cur_addr),
                        desc_V,
                        tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF)),
                        k,
                        accum_b2,
                        issue_mma=bmm2_issue,
                    )
                    accum_b2 = cutlass.Boolean(True)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[parity_cur_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

            kv_last = kv_right - cutlass.Int32(1)
            parity_last_rt = kv_last & cutlass.Int32(1)
            last_is_even = parity_last_rt == cutlass.Int32(0)
            tmem_P_last_addr = cutlass.Int32(
                arith.select(
                    last_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                )
            )
            bmm2_ready_phase_last = (bmm2_ready_phase_pair >> parity_last_rt) & cutlass.Int32(1)

            bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
            desc_V = sV[kv_state_V.idx].desc()
            n_kv_eff = kv_right - kv_left
            scaleC_epi = cutlass.Boolean(n_kv_eff != cutlass.Int32(1))
            accum_b2 = scaleC_epi
            bmm2_issue_epi = nvvm.elect_sync()
            for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                if k % k_per_chunk == 0:
                    chunk_id = k // k_per_chunk
                    bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_last)
                mma_ts_step(
                    bmm2_desc,
                    tmem_raw.subview(tmem_P_last_addr),
                    desc_V,
                    tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF)),
                    k,
                    accum_b2,
                    issue_mma=bmm2_issue_epi,
                )
                accum_b2 = cutlass.Boolean(True)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[parity_last_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
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
    qh_per_kh,
    leader_cta_id,
    cta_in_pair,
    bottom_right_diagonal: cutlass.Constexpr[bool],
    softmax_exchange,
    mb_softmax_max,
):
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base = tmem_ptr_i32.load()

    bmm1_done_phase_pair = cutlass.Int32(0)
    softmax_max_phase_pair = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)
    epilogue_state = cutlass.Int32(1)

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
        qh_per_kh,
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
        HEADS_PER_TILE,
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
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_abs = q_row_coord + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        mask_q_abs = q_abs
        mask_flags = CFG.MASK_FLAGS
        if cutlass.const_expr(_PADDED_TOP_LEFT_CAUSAL):
            mask_q_abs = cute.math.min(q_abs, eff_seqlen_kv - cutlass.Int32(1))
            mask_flags = MASK_CAUSAL

        if cutlass.const_expr(CFG.DTYPE_QKV != 1):
            bars.mb_o_empty.wait(epilogue_state)
        if cutlass.const_expr(softmax_half == 0):
            bars.mb_stat_empty.wait(stat_empty_phase)
            stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        if cutlass.const_expr(CFG.DTYPE_QKV != 1):
            epilogue_state = epilogue_state ^ cutlass.Int32(1)

        CHUNK = 64
        P_COLS_PER_CHUNK = CHUNK // 4
        P_SUBCHUNK = 32
        P_COLS_PER_SUBCHUNK = P_SUBCHUNK // 4
        N_CHUNKS = CFG.N_BMM2_CHUNKS
        RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

        if cutlass.const_expr(CFG.SOFTMAX_WARPGROUPS == 2):
            split_hi = bounds.right if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE) else bounds.unmasked_hi
            for kv_loop in cutlass.range(bounds.left, split_hi, 1, unroll=1):
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
                    raw_hi = [
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(CHUNK), cutlass.Float32),
                            num=32,
                        ),
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(CHUNK + 32), cutlass.Float32),
                            num=32,
                        ),
                    ]
                    max_hi = row_max_reduction(RegTile(vec_concat(raw_hi), size=CHUNK).vec)
                    raw_lo = [
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base, cutlass.Float32),
                            num=32,
                        ),
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(32), cutlass.Float32),
                            num=32,
                        ),
                    ]
                    reg_S_half = RegTile(vec_concat(raw_lo), size=CHUNK)
                    current_max_unscaled = cute.math.max(
                        row_max_reduction(reg_S_half.vec),
                        max_hi,
                        ftz=True,
                    )
                    current_max = current_max_unscaled * scale_log2
                    update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                    total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                    new_total_max_safe = total_max
                    alpha = cute.math.exp2(
                        cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)),
                        fastmath=True,
                    )
                    total_max_safe = new_total_max_safe
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                    mb_softmax_max[parity_rt].arrive()
                else:
                    reg_S_half = RegTile(
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(CHUNK), cutlass.Float32),
                            num=CHUNK,
                        ),
                        size=CHUNK,
                    )
                if cutlass.const_expr(softmax_half == 1):
                    softmax_max_phase = (softmax_max_phase_pair >> parity_rt) & cutlass.Int32(1)
                    mb_softmax_max[parity_rt].wait(softmax_max_phase)
                    softmax_max_phase_pair = softmax_max_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    new_total_max_safe = softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).load()
                    alpha = cute.math.exp2(
                        cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)),
                        fastmath=True,
                    )
                    total_max_safe = new_total_max_safe

                if cutlass.const_expr(softmax_half == 0):
                    if cutlass.const_expr(_SMEM_ALPHA_HANDOFF):
                        alpha_idx = parity_rt * cutlass.Int32(2 * CFG.TILE_M) + tid_in_wg
                        softmax_exchange.subview(alpha_idx).store(alpha)
                    else:
                        alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    _stat_arrive(bars.mb_stat_full, _STAT_FULL_WARP_ARRIVALS)

                if cutlass.const_expr(CFG.DTYPE_QKV == 0):
                    scale_vec = cutlass.vector.full_like(reg_S_half.vec, scale_log2)
                    minus_max_vec = cutlass.vector.full_like(reg_S_half.vec, cutlass.Float32(0.0) - total_max_safe)
                    reg_S_half = RegTile(
                        nvvm.fma_packed_f32x2(reg_S_half.vec, scale_vec, minus_max_vec, rnd="rn", ftz=False),
                        size=CHUNK,
                    )
                elif cutlass.const_expr(softmax_half == 0):
                    reg_S_half = reg_S_half * scale_log2 - total_max_safe
                else:
                    # Device-resident FP8 descales lower to a regular register.
                    # Keep the multiply adjacent to the subtract so PTXAS can
                    # emit FFMA2 instead of a separate FMUL2/FADD2 pair.
                    reg_S_half = scale_log2 * reg_S_half - total_max_safe
                chunk_P_a = _exp2_dense_chunk_a(reg_S_half[0:P_SUBCHUNK].vec, softmax_half)
                new_p_sum_pair = row_reduction_pair(chunk_P_a)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(
                        p_addr_base + cutlass.Int32(softmax_half * P_COLS_PER_CHUNK),
                        cutlass.Float32,
                    ),
                    chunk_P_a.to(STORAGE_DTYPE),
                )
                chunk_P_b = _exp2_dense_chunk_b(reg_S_half[P_SUBCHUNK:CHUNK].vec, softmax_half)
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

            if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE):
                for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                    parity_rt = kv_loop & cutlass.Int32(1)
                    parity_is_even = parity_rt == cutlass.Int32(0)
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
                    bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                    bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                    bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)

                    s_addr_base = tmem_base + s_off_rt
                    p_addr_base = tmem_base + p_off_rt
                    stats_addr = tmem_base + s_off_rt
                    kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)

                    if cutlass.const_expr(bottom_right_diagonal):
                        mask_bottom_right = 0
                        causal_diag = None
                    else:
                        mask_bottom_right = CFG.BOTTOM_RIGHT
                        causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None

                    if cutlass.const_expr(softmax_half == 0):
                        raw_chunks = [
                            nvvm.tcgen05_ld(
                                "32x32b",
                                nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32),
                                num=CHUNK,
                            )
                            for c in range(N_CHUNKS)
                        ]
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
                        reg_S_full = RegTile(vec_concat(chunks_S), size=CFG.TILE_N)
                        current_max_unscaled = cute.math.max(
                            row_max_reduction(reg_S_full[0:CHUNK].vec),
                            row_max_reduction(reg_S_full[CHUNK : 2 * CHUNK].vec),
                            ftz=True,
                        )
                        current_max = current_max_unscaled * scale_log2
                        update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                        total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                        new_total_max_safe = row_max_for_exp2(total_max)
                        alpha = _exp2_masked_alpha(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)))
                        total_max_safe = new_total_max_safe
                        reg_S_half = reg_S_full[0:CHUNK]

                        exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                        softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                        mb_softmax_max[parity_rt].arrive()
                    else:
                        raw_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(CHUNK), cutlass.Float32),
                            num=CHUNK,
                        )
                        masked_chunk = apply_mask_chunk(
                            raw_chunk,
                            mask_q_abs - (kv_col_base + cutlass.Int32(CHUNK)),
                            cutlass.Int32(0),
                            eff_seqlen_kv - (kv_col_base + cutlass.Int32(CHUNK)),
                            CFG.WINDOW_LEFT,
                            mask_flags,
                            N=CHUNK,
                            bottom_right=mask_bottom_right,
                            causal_diag=causal_diag,
                            window_right=CFG.WINDOW_RIGHT,
                            mask_value=float("-inf"),
                        )
                        reg_S_half = RegTile(masked_chunk, size=CHUNK)

                    if cutlass.const_expr(softmax_half == 1):
                        softmax_max_phase = (softmax_max_phase_pair >> parity_rt) & cutlass.Int32(1)
                        mb_softmax_max[parity_rt].wait(softmax_max_phase)
                        softmax_max_phase_pair = softmax_max_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                        exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                        new_total_max_safe = softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).load()
                        alpha = _exp2_masked_alpha(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)))
                        total_max_safe = new_total_max_safe

                    if cutlass.const_expr(softmax_half == 0):
                        if cutlass.const_expr(_SMEM_ALPHA_HANDOFF):
                            alpha_idx = parity_rt * cutlass.Int32(2 * CFG.TILE_M) + tid_in_wg
                            softmax_exchange.subview(alpha_idx).store(alpha)
                        else:
                            alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                        _stat_arrive(bars.mb_stat_full, _STAT_FULL_WARP_ARRIVALS)

                    reg_S_half = reg_S_half * scale_log2 - total_max_safe
                    if cutlass.const_expr(_SMEM_ALPHA_HANDOFF):
                        chunk_P_a = _exp2_masked_tail_chunk_a(reg_S_half[0:P_SUBCHUNK].vec, softmax_half)
                    else:
                        chunk_P_a = cute.math.exp2(reg_S_half[0:P_SUBCHUNK].vec, fastmath=True)
                    new_p_sum_pair = row_reduction_pair(chunk_P_a)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            p_addr_base + cutlass.Int32(softmax_half * P_COLS_PER_CHUNK),
                            cutlass.Float32,
                        ),
                        chunk_P_a.to(STORAGE_DTYPE),
                    )
                    chunk_P_b = cute.math.exp2(reg_S_half[P_SUBCHUNK:CHUNK].vec, fastmath=True)
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
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(softmax_half)].arrive(
                        leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA
                    )

                    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                    total_sum = total_sum * alpha_pair + new_p_sum_pair

                    if cutlass.const_expr(softmax_half == 0):
                        bars.mb_stat_empty.wait(stat_empty_phase)
                        stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        else:
            # Exact top-left causal has no lower masking boundary.
            left_edge_end = (
                bounds.left if cutlass.const_expr(CFG.MASK_FLAGS == MASK_CAUSAL and (CFG.BOTTOM_RIGHT == 0 or bottom_right_diagonal)) else bounds.unmasked_lo
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
                if cutlass.const_expr(bottom_right_diagonal):
                    mask_bottom_right = 0
                    causal_diag = None
                else:
                    mask_bottom_right = CFG.BOTTOM_RIGHT
                    causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
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
                alpha = _exp2_masked_alpha(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)))
                total_max_safe = new_total_max_safe
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                _stat_arrive(bars.mb_stat_full, _STAT_FULL_WARP_ARRIVALS)
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_P_0a = cute.math.exp2(reg_S[0:P_SUBCHUNK].vec, fastmath=True)
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
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_P_1a = cute.math.exp2(reg_S[CHUNK : CHUNK + P_SUBCHUNK].vec, fastmath=True)
                    deferred_sum_1 = row_reduction_pair(chunk_P_1a)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32),
                        chunk_P_1a.to(STORAGE_DTYPE),
                    )
                    chunk_P_1b = cute.math.exp2(reg_S[CHUNK + P_SUBCHUNK : 2 * CHUNK].vec, fastmath=True)
                    deferred_sum_1 = deferred_sum_1 + row_reduction_pair(chunk_P_1b)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK + P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_1b.to(STORAGE_DTYPE),
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
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
                if cutlass.const_expr(CFG.DTYPE_QKV == 1 or (CFG.DTYPE_QKV == 0 and CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT == 0)):
                    raw_chunks = [
                        nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base, cutlass.Float32), num=32),
                        nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(32), cutlass.Float32), num=32),
                        nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(64), cutlass.Float32), num=64),
                    ]
                else:
                    unmasked_chunk = 32
                    raw_chunks = [
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * unmasked_chunk), cutlass.Float32),
                            num=unmasked_chunk,
                        )
                        for c in range(CFG.TILE_N // unmasked_chunk)
                    ]
                reg_S_vec = vec_concat(raw_chunks)
                current_max_unscaled = row_max_reduction(reg_S_vec)
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
                alpha = _exp2_masked_alpha(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)))
                total_max_safe = new_total_max_safe
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                _stat_arrive(bars.mb_stat_full, _STAT_FULL_WARP_ARRIVALS)
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_P_0a = _exp2_dense_chunk_a(reg_S[0:P_SUBCHUNK].vec, 0)
                if cutlass.const_expr(_E5_STYLE_SOFTMAX and CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT == 0):
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0a.to(STORAGE_DTYPE))
                    hoisted_sum = row_reduction_pair(chunk_P_0a)
                else:
                    hoisted_sum = row_reduction_pair(chunk_P_0a)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0a.to(STORAGE_DTYPE))
                chunk_P_0b = _exp2_dense_chunk_b(reg_S[P_SUBCHUNK:CHUNK].vec, 3)
                if cutlass.const_expr(_E5_STYLE_SOFTMAX and CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT == 0):
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_0b.to(STORAGE_DTYPE),
                    )
                    hoisted_sum = hoisted_sum + row_reduction_pair(chunk_P_0b)
                else:
                    hoisted_sum = hoisted_sum + row_reduction_pair(chunk_P_0b)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_0b.to(STORAGE_DTYPE),
                    )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_sum_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_P_1a = _exp2_dense_chunk_a(reg_S[CHUNK : CHUNK + P_SUBCHUNK].vec, 0)
                    deferred_sum_1 = row_reduction_pair(chunk_P_1a)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32),
                        chunk_P_1a.to(STORAGE_DTYPE),
                    )
                    chunk_P_1b = _exp2_dense_chunk_b(reg_S[CHUNK + P_SUBCHUNK : 2 * CHUNK].vec, 2)
                    deferred_sum_1 = deferred_sum_1 + row_reduction_pair(chunk_P_1b)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK + P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_1b.to(STORAGE_DTYPE),
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + deferred_sum_1
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
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
                if cutlass.const_expr(bottom_right_diagonal):
                    mask_bottom_right = 0
                    causal_diag = None
                else:
                    mask_bottom_right = CFG.BOTTOM_RIGHT
                    causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
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
                alpha = _exp2_masked_alpha(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)))
                total_max_safe = new_total_max_safe
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                _stat_arrive(bars.mb_stat_full, _STAT_FULL_WARP_ARRIVALS)
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_P_0a = cute.math.exp2(reg_S[0:P_SUBCHUNK].vec, fastmath=True)
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
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_P_1a = cute.math.exp2(reg_S[CHUNK : CHUNK + P_SUBCHUNK].vec, fastmath=True)
                    deferred_sum_1 = row_reduction_pair(chunk_P_1a)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32),
                        chunk_P_1a.to(STORAGE_DTYPE),
                    )
                    chunk_P_1b = cute.math.exp2(reg_S[CHUNK + P_SUBCHUNK : 2 * CHUNK].vec, fastmath=True)
                    deferred_sum_1 = deferred_sum_1 + row_reduction_pair(chunk_P_1b)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK + P_COLS_PER_SUBCHUNK), cutlass.Float32),
                        chunk_P_1b.to(STORAGE_DTYPE),
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
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

        if cutlass.const_expr(softmax_half == 0):
            stats_addr_epi = tmem_base + cutlass.Int32(LAYOUT.STATS_EPI_OFF)
            stats_vec_epi = cutlass.Vector.from_elements((total_max_safe, total_sum_scalar), cutlass.Float32)
            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            _stat_arrive(bars.mb_stat_full, _STAT_FULL_WARP_ARRIVALS)

        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)
        payload_base = sched_state.idx * cutlass.Int32(SCHED_PAYLOAD_WORDS)
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(2)).load())
        q_super_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(3)).load())
        batch_idx = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(payload_base + cutlass.Int32(5)).load())
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
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
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
    qh_per_kh,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
    bottom_right_diagonal: cutlass.Constexpr[bool],
    o_scale_fused,
    amax_o_tensor,
    softmax_exchange,
):
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))
    tmem_base_corr = tmem_ptr_i32.load()

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    bmm2_done_phase_pair = cutlass.Int32(0)
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
        qh_per_kh,
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
        HEADS_PER_TILE,
    )

    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS_LOCAL = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS_LOCAL
    TMA_O_GRANU_ELEMS_LOCAL = CFG.TILE_M * D_BLOCK_SIZE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if bounds.right > bounds.left:
            lo_parity_rt = bounds.left & cutlass.Int32(1)
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            bars.mb_stat_full.wait(stat_mbar_state)
            _stat_arrive(bars.mb_stat_empty, _STAT_EMPTY_WARP_ARRIVALS)
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)
        else:
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
            parity_cur_rt = kv_loop & cutlass.Int32(1)
            tmem_base_iter = tmem_base_corr

            bars.mb_stat_full.wait(stat_mbar_state)

            if cutlass.const_expr(_SMEM_ALPHA_HANDOFF):
                alpha_idx = parity_cur_rt * cutlass.Int32(2 * CFG.TILE_M) + tid_in_wg
                alpha = softmax_exchange.subview(alpha_idx).load()
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

            _stat_arrive(bars.mb_stat_empty, _STAT_EMPTY_WARP_ARRIVALS)

            bmm2_done_phase_prev = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
            # Each iteration must consume its BMM2 phase before O can be reused.
            bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev)

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
                if cutlass.const_expr(_RARE_RESCALE_CORRECTION):
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            if cutlass.const_expr(not _RARE_RESCALE_CORRECTION):
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

            bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        tmem_base_epi = tmem_base_corr

        total_max_scaled = cutlass.Float32(0.0)
        total_sum = cutlass.Float32(0.0)
        # UNCONDITIONAL epilogue stat handshake (mirrors d128): the softmax group
        # publishes its epilogue stats every tile — including empty/inverted KV
        # ranges from compute_kv_loop_bounds (SWA window past the padded KV tail).
        # Guarding this behind right > left desyncs the mb_stat_full/mb_stat_empty
        # phase counts and the next tile's softmax wait spins forever (the
        # rectangular causal+SWA+padding hang). On an empty tile the published
        # stats are the reset values (max=-inf, sum=0) and the dead-row override
        # below emits O=0 / LSE=-inf.
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
        _stat_arrive(bars.mb_stat_empty, _STAT_EMPTY_WARP_ARRIVALS)
        stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        LN2 = cutlass.Float32(0.6931471805599453)
        total_max_nat = total_max_scaled * LN2
        lse_val = cutlass.Float32(0.0)
        inv_sum = cutlass.Float32(0.0)
        beta = cutlass.Float32(0.0)
        q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        row_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE) + (tid_in_wg % cutlass.Int32(HEADS_PER_TILE))
        row_dead = total_sum <= cutlass.Float32(0.0)
        if cutlass.const_expr(CFG.HAS_SINK):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = sinks_arr[row_head_idx]
            new_max = cute.math.max(total_max_nat, sink_logit)
            scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
            new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_val = new_max + cute.math.log(new_sum, fastmath=True)
            beta = scale / new_sum
            inv_sum = beta * o_scale_fused
        else:
            lse_val = total_max_nat + cute.math.log(cute.math.max(total_sum, cutlass.Float32(1e-30)), fastmath=True)
            beta = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
            inv_sum = beta * o_scale_fused
            # Dead row (no valid KV column, incl. empty tiles where total_sum defaults 0):
            #   O := 0, LSE := -inf. total_sum >= 1 for any alive row so this never fires
            #   spuriously. Skipped under sink (the sink path defines a finite LSE).
            neg_inf_lse = cutlass.Float32(float("-inf"))
            lse_val = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
            beta = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), beta.ir_value()))

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
                        lse_arr[cu_q_b + q_row_global, row_head_idx] = lse_val
                    else:
                        lse_arr[cutlass.Int32(0), row_head_idx, cu_q_b + q_row_global] = lse_val
        else:
            _row_valid = q_row_global < seqlen_q
            if cutlass.const_expr(lse_tensor is not None):
                if _row_valid:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                    lse_arr[lse_batch, row_head_idx, q_row_global] = lse_val

        parity_last_rt = cutlass.Int32(0)
        if bounds.right > bounds.left:
            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
        bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
        bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last)
        bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

        if cutlass.const_expr(_PADDED_TOP_LEFT_CAUSAL):
            if bounds.right <= bounds.left:
                zero_o = cutlass.Vector.from_elements(
                    tuple(cutlass.Float32(0.0) for _ in range(O_CHUNK)),
                    cutlass.Float32,
                )
                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_addr = tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        zero_o,
                    )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

        O_EPI_BLK = 64 // CFG.BPE_O
        N_BLOCKS_EPI = CFG.TILE_O // O_EPI_BLK
        CHUNKS_PER_BLK = O_EPI_BLK // O_CHUNK

        sO_base = sO[0].base
        _amax_o_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
        _amax_o_local = cutlass.Float32(0.0)

        for block_idx in cutlass.range_constexpr(N_BLOCKS_EPI):
            for sub in cutlass.range_constexpr(CHUNKS_PER_BLK):
                chunk_idx_total = block_idx * CHUNKS_PER_BLK + sub
                o_addr = tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + chunk_idx_total * O_CHUNK)
                o_chunk = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_CHUNK,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_chunk * inv_sum
                # Empty BMM2 leaves O TMEM unwritten, and NaN * 0 does not
                # sanitize it. Padded top-left clears an empty tile once above;
                # keep per-element selects for the remaining dynamic bounds.
                if cutlass.const_expr(
                    not _PADDED_TOP_LEFT_CAUSAL and (CFG.SEQ_KV_LENS_PRESENT or SPLIT_KV > 1 or (CFG.BOTTOM_RIGHT and not bottom_right_diagonal))
                ):
                    zero = cutlass.Float32(0.0)
                    invalid = row_dead
                    if cutlass.const_expr(CFG.THD_VARLEN):
                        invalid = invalid | (~_row_valid)
                    o_scaled = cutlass.Vector.from_elements(
                        tuple(cutlass.Float32(arith.select(invalid.ir_value(), zero.ir_value(), o_scaled[i].ir_value())) for i in range(O_CHUNK)),
                        cutlass.Float32,
                    )
                _amax_o_local = cute.math.max(_amax_o_local, _max_abs_reduction(o_scaled), ftz=True)
                o_out = o_scaled.to(OUT_STORAGE_DTYPE)

                col_offset_const = (chunk_idx_total * O_CHUNK) % D_BLOCK_SIZE
                block_offset_const = ((chunk_idx_total * O_CHUNK) // D_BLOCK_SIZE) * TMA_O_GRANU_ELEMS_LOCAL
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)
                smem_ptr = sO_base.subview(smem_offset).data_ptr()

                if block_idx == 0 and sub == 0:
                    bars.mb_o_empty.wait(epilogue_state)
                smem_ptr.store_swizzled(o_out, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            fire_now = (block_idx % 2 == 1) or (CFG.TILE_O == O_EPI_BLK)
            if cutlass.const_expr(fire_now):
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_o_full[(block_idx // 2)].arrive()

        if cutlass.const_expr(SPLIT_KV == 1):
            if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
                _amax_o_valid = cutlass.Float32(
                    arith.select(
                        _row_valid.ir_value(),
                        _amax_o_local.ir_value(),
                        cutlass.Float32(0.0).ir_value(),
                    )
                )
                _amax_o_warp = cute.arch.warp_redux_sync(_amax_o_valid, kind="fmax")
                if (tid_in_wg & cutlass.Int32(31)) == cutlass.Int32(0):
                    nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_o_ptr, _amax_o_warp.bitcast(cutlass.Int32))
            else:
                if _row_valid:
                    nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_o_ptr, _amax_o_local.bitcast(cutlass.Int32))

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
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    bottom_right_diagonal: cutlass.Constexpr[bool],
    scale_softmax_log2: cutlass.Float32,
    o_scale_fused: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: cute.Tensor,
    amax_o_tensor: cute.Tensor,
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
    k_rank5_layout = cute.make_layout(
        (k_tensor.shape[0], k_tensor.shape[2], TMA_QK_ITERS, k_tensor.shape[1], TMA_QK_GRANU_ELEMS),
        stride=(
            k_tensor.stride[0],
            k_tensor.stride[2],
            TMA_QK_GRANU_ELEMS,
            k_tensor.stride[1],
            1,
        ),
    )
    k_rank5_tensor = cute.make_tensor(k_tensor.iterator, k_rank5_layout)
    v_rank5_layout = cute.make_layout(
        (v_tensor.shape[0], v_tensor.shape[2], TMA_VO_ITERS, v_tensor.shape[1], TMA_VO_GRANU_ELEMS),
        stride=(
            v_tensor.stride[0],
            v_tensor.stride[2],
            TMA_VO_GRANU_ELEMS,
            v_tensor.stride[1],
            1,
        ),
    )
    v_rank5_tensor = cute.make_tensor(v_tensor.iterator, v_rank5_layout)
    if cutlass.const_expr(CFG.PACK_GQA and q_tensor.shape[2] != k_tensor.shape[2] * CFG.QH_PER_KH):
        raise ValueError(f"CFG.QH_PER_KH ({CFG.QH_PER_KH}) does not match tensor head extents H_q={q_tensor.shape[2]}, H_kv={k_tensor.shape[2]}")
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, 1, TMA_QK_ITERS, CFG.TILE_N // CFG.CTA_MMA, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, 1, TMA_VO_ITERS, CFG.TILE_N, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor,
        box_dims=qk_box_q,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_256b,
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_rank5_tensor,
        box_dims=qk_box_k,
        stride_order=(4, 3, 2, 1, 0),
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_256b,
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_rank5_tensor,
        box_dims=vo_box_v,
        stride_order=(4, 3, 2, 1, 0),
        swizzle=_tma_swz(CFG.V_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_256b,
    )
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=vo_box_o,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
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
            (grid_q_supers, QH // HEADS_PER_TILE, B * SPLIT_KV)
            if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL)
            else (grid_q_supers * (QH // HEADS_PER_TILE) * B * SPLIT_KV, 1, 1)
        )
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(q_supers),
        cutlass.Int32(QH // HEADS_PER_TILE),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        scale_softmax_log2,
        o_scale_fused,
        descale_q_t,
        descale_k_t,
        descale_v_t,
        scale_o_t,
        amax_o_tensor,
        bottom_right_diagonal,
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
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_head_major: bool = False,
    lse_head_stride: int = 0,
    lse_stride: Optional[tuple[int, int, int]] = None,
    q_stride: Optional[tuple] = None,
    k_stride: Optional[tuple] = None,
    v_stride: Optional[tuple] = None,
    o_stride: Optional[tuple] = None,
) -> Callable:
    """ENVELOPE: ``d_qk`` / ``d_v`` are the ACTUAL head dims (defaults = full
    TILE_K / TILE_O). TMA descriptors carry these extents while the tile box
    stays the compile-time TILE geometry: loads past d_qk / d_v zero-fill
    (exact zeros in the QK^T / P·V contractions), O stores past d_v clip.
    d * BPE must be a 16-byte multiple (TMA global-stride rule -> d % 8)."""
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d256 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d256 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if CFG.THD_VARLEN and (d_qk != CFG.TILE_K or d_v != CFG.TILE_O):
        raise ValueError("D256 FP8 THD serves native d_qk=d_v=256 only")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_stride is not None and SPLIT_KV > 1:
        raise ValueError("split_kv partial LSE is compact; lse_stride describes only the final combine output")
    fake_batch = 1 if CFG.THD_VARLEN else b
    if CFG.THD_VARLEN:
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)
    o_batch = fake_batch * SPLIT_KV
    lse_batch = b * SPLIT_KV

    def _fake_bshd(shape, stride, dtype=STORAGE_DTYPE, bpe=CFG.BPE):
        if stride is None:
            return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)
        if stride[3] != 1:
            raise ValueError(f"declared stride {stride}: the head dim must be innermost-contiguous (stride[3] == 1)")
        for axis in (1, 2):  # seq/head global strides feed TMA: 16-byte rule
            if (stride[axis] * bpe) % 16 != 0:
                raise ValueError(f"declared stride {stride} axis {axis} must be a 16-byte multiple at BPE={bpe} (TMA global-stride rule)")
        return cute.runtime.make_fake_tensor(dtype, shape, tuple(stride), assumed_align=16)

    fake_q = _fake_bshd((fake_batch, sq, qh, d_qk), q_stride)
    fake_k = _fake_bshd((fake_batch, skv, kh, d_qk), k_stride)
    fake_v = _fake_bshd((fake_batch, skv, kh, d_v), v_stride)
    fake_o = _fake_bshd((o_batch, sq, qh, d_v), o_stride, dtype=OUT_STORAGE_DTYPE, bpe=CFG.BPE_O)
    if not has_lse:
        # No Stats output: the LSE argument is None-specialized and the store
        # is compiled out entirely — no dummy buffer exists at any level.
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
            raise ValueError("lse_head_major / lse_head_stride are unsupported for dense FP8")
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, (lse_batch, qh, sq), lse_stride, assumed_align=4)
            if lse_stride is not None
            else cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (lse_batch, qh, sq),
                stride_order=(2, 1, 0),
                assumed_align=16,
            )
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

    def _fake_scale():
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (1,),
            stride_order=(0,),
            assumed_align=4,
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
        fake_lse,
        fake_sinks,
        fake_seq_kv_lens,
        fake_o_desc,
        (b, qh, kh, 0, 0, 0) if CFG.THD_VARLEN else (b, qh, kh, sq, skv, 0),
        bool(CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT and not CFG.SEQ_KV_LENS_PRESENT and sq == skv),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        fake_amax_o,
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

    print(f"[d256_fp8_sm100] compile b={args.b} qh={args.hq} kh={args.hk} " f"sq={args.sq} skv={args.skv}", flush=True)
    fn = compile(args.b, args.hq, args.hk, args.sq, args.skv)
    print(f"[d256_fp8_sm100] compile OK: {fn}", flush=True)
    if args.validate:
        print("[d256_fp8_sm100] compiled - run validation via the frost SDPA test suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
