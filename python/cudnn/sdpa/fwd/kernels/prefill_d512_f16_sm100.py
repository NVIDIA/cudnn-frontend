# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from functools import lru_cache
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from typing import NamedTuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import print_runtime
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d512

# The per-graph params are injected as a module global by the loader
# (api._load_kernel_module) before this body executes; a plain import gets
# the all-defaults config (dense fp16), which is what the standalone
# `python prefill_sdpa_d512_f16_sm100.py` benchmark path uses.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d512(PARAMS)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS


def _require(cond, msg):
    """Geometry sanity check; raises instead of assert (asserts vanish under -O)."""
    if not cond:
        raise ValueError(f"prefill_sdpa_d512_f16_sm100: {msg}")


TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES


from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    wait,
    arrive,
    arrive_expect_tx,
    cga_arrive,
    cga_wait,
    mbar_arrive_on_peer,
    commit_mma,
    arrive_on_leader,
    MBarrier,
    Producer,
    Scope,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    read_tile_id_arrive,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from cudnn.frost.tile_dsl.pointwise import (
    tmem_load_tile,
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts, mma_ts_step
from cudnn.frost.tile_dsl.tma import (
    cp_async_bulk_shared_cluster_shared_cta,
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import (
    apply_mask_chunk,
    MASK_NONE,
    MASK_PADDED,
    MASK_CAUSAL,
    MASK_SWA,
)

from cudnn.sdpa.fwd.kernels._common_sm100 import (
    make_split_helpers,
    KvLoopBounds,
    compute_kv_loop_bounds,
    row_max_for_exp2,
    make_sdpa_helpers,
)

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
    P_STORAGE_DTYPE = cutlass.BFloat16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
    P_STORAGE_DTYPE = cutlass.Float16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
else:
    raise ValueError(
        f"prefill_sdpa_d512_f16 (SM100): DTYPE_QKV={CFG.DTYPE_QKV} not supported (expected 2=BF16 or 3=FP16; FP8/MXFP8 ship in a separate kernel file)"
    )
OUT_STORAGE_DTYPE = STORAGE_DTYPE


CGA_SIZE = CFG.CGA_M * CFG.CGA_N
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA
oBufferElems = CFG.TILE_M * CFG.TILE_O
pXferElems = CFG.TILE_M * CFG.TILE_N

pXferBytes = pXferElems * CFG.BPE
alphaXferBytes = CFG.TILE_M * 4
statsXferBytes = 2 * CFG.TILE_M * 4

P_XFER_HALVES = 2
pHalfXferElems = pXferElems // P_XFER_HALVES
pHalfXferBytes = pXferBytes // P_XFER_HALVES

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

BMM2_V_NBLOCK_ADVANCE = CFG.TILE_N * (CFG.BMM2_N_PER_CALL // CFG.CTA_MMA) * CFG.BPE

N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

P_TMA_ITERS = (CFG.TILE_N * CFG.BPE) // CFG.Q_SWZ_BYTES
P_D_BLOCK = CFG.TILE_N // P_TMA_ITERS
P_BLOCK_BYTES = CFG.TILE_M * P_D_BLOCK
SOFTMAX_CHUNK = 64
SOFTMAX_N_CHUNKS_LOAD = CFG.TILE_N // SOFTMAX_CHUNK
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
NEG_INF_F32 = cutlass.Float32(float("-inf"))
RESCALE_THRESHOLD_F32 = cutlass.Float32(CFG.RESCALE_THRESHOLD)


COMPUTE_LANES = CFG.SOFTMAX_WG_WARPS * 32
SM_LANES_TOTAL = 2 * COMPUTE_LANES
TWO_LANES_TOTAL = 2
KV_EMPTY_ARRIVERS = CFG.CGA_N

READ_TILE_ARRIVERS = CFG.READ_TILE_ARRIVERS


_ALIAS_SMEM_KIB = (
    max(
        qBufferElems * CFG.BPE,
        oBufferElems * CFG.BPE_O,
        CFG.STAGES_KV * kBufferElems * CFG.BPE,
        CFG.STAGES_KV * vBufferElems * CFG.BPE,
    )
    / 1024
)
_SG0_SMEM_DATA_KIB = (_ALIAS_SMEM_KIB * 1024 + CFG.XFER_STAGES * pXferBytes + CFG.XFER_STAGES * alphaXferBytes + statsXferBytes) / 1024
_SG1_SMEM_DATA_KIB = (_ALIAS_SMEM_KIB * 1024 + CFG.XFER_STAGES * pXferBytes + CFG.XFER_STAGES * alphaXferBytes + statsXferBytes + CFG.TILE_M * 4) / 1024
_require(_SG0_SMEM_DATA_KIB <= 223, f"sg0 SMEM data {_SG0_SMEM_DATA_KIB:.1f} KiB exceeds 223 KiB headroom under 227 KiB SM100 cap")
_require(_SG1_SMEM_DATA_KIB <= 223, f"sg1 SMEM data {_SG1_SMEM_DATA_KIB:.1f} KiB exceeds 223 KiB headroom under 227 KiB SM100 cap")

_TMEM_SG0_COLS = CFG.XFER_STAGES * CFG.TILE_N
_TMEM_SG1_COLS = CFG.TILE_O
_require(_TMEM_SG0_COLS <= 512, f"sg0 S_acc TMEM overflow: {_TMEM_SG0_COLS} > 512")
_require(_TMEM_SG1_COLS <= 512, f"sg1 O TMEM overflow: {_TMEM_SG1_COLS} > 512")


class Bars(NamedTuple):
    mb_tma_q_full: object
    mb_tma_q_empty: object
    mb_tma_k_full: object
    mb_tma_k_empty: object
    mb_tma_v_full: object
    mb_tma_v_empty: object
    mb_q_utccp_done: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object
    mb_s_acc_empty: object

    mb_p_xfer_full: object
    mb_p_xfer_empty: object
    mb_alpha_xfer_full: object
    mb_alpha_xfer_empty: object
    mb_stats_xfer_full: object
    mb_stats_xfer_empty: object

    mb_tma_o_full: object
    mb_tma_o_empty: object

    mb_empty_mainloop: object
    mb_tmem_dealloc: object


def _make_d512_sm100_bars(CFG, N_O_CHUNKS: int) -> Bars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        mb_tma_q_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_q_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_tma_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_tma_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_q_utccp_done=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm1_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS),
            stages=CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS,
            init_count=SM_LANES_TOTAL,
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_s_acc_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=SM_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_p_xfer_full=MBarrier(
            _alloc(CFG.XFER_STAGES * P_XFER_HALVES), stages=CFG.XFER_STAGES * P_XFER_HALVES, init_count=CFG.CTA_MMA, producer=Producer.TMA_LOAD
        ),
        mb_p_xfer_empty=MBarrier(
            _alloc(CFG.XFER_STAGES * P_XFER_HALVES), stages=CFG.XFER_STAGES * P_XFER_HALVES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT
        ),
        mb_alpha_xfer_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_alpha_xfer_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=COMPUTE_LANES, producer=Producer.THREAD),
        mb_stats_xfer_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_stats_xfer_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_tma_o_full=MBarrier(_alloc(N_O_CHUNKS), stages=N_O_CHUNKS, init_count=COMPUTE_LANES, producer=Producer.THREAD),
        mb_tma_o_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=TWO_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
    )


@dataclass(frozen=True)
class KernelTmemLayout:
    TOTAL_COLS: int = 512

    S_ACC_COLS: int = 128
    S_ACC_PARITY0_OFF: int = 0
    S_ACC_PARITY1_OFF: int = 128
    Q_TMEM_OFF: int = 256
    Q_TMEM_COLS: int = 256

    O_OFF: int = 0
    O_COLS: int = 512


LAYOUT = KernelTmemLayout()
_require(
    CFG.XFER_STAGES * LAYOUT.S_ACC_COLS == LAYOUT.Q_TMEM_OFF,
    f"sg0: S_acc parities ({CFG.XFER_STAGES * LAYOUT.S_ACC_COLS}) must abut Q at col {LAYOUT.Q_TMEM_OFF}",
)
_require(LAYOUT.Q_TMEM_OFF + LAYOUT.Q_TMEM_COLS <= LAYOUT.TOTAL_COLS, f"sg0 TMEM overflow: {LAYOUT.Q_TMEM_OFF + LAYOUT.Q_TMEM_COLS} > {LAYOUT.TOTAL_COLS}")
_require(LAYOUT.O_OFF + LAYOUT.O_COLS <= LAYOUT.TOTAL_COLS, f"sg1 TMEM overflow: {LAYOUT.O_OFF + LAYOUT.O_COLS} > {LAYOUT.TOTAL_COLS}")
_require(LAYOUT.Q_TMEM_COLS == CFG.TILE_K // 2, f"Q TMEM cols {LAYOUT.Q_TMEM_COLS} != TILE_K/2 ({CFG.TILE_K // 2}) at FP16")


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q
SMEM_LAYOUT_P = SMEM_LAYOUT_Q

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2

NUM_KPHASES_PV_HALF = NUM_KPHASES_PV // P_XFER_HALVES
_require(P_TMA_ITERS == P_XFER_HALVES, f"P-xfer split assumes P_TMA_ITERS ({P_TMA_ITERS}) == P_XFER_HALVES ({P_XFER_HALVES})")
_require(NUM_KPHASES_PV % P_XFER_HALVES == 0, f"NUM_KPHASES_PV ({NUM_KPHASES_PV}) not divisible by P_XFER_HALVES ({P_XFER_HALVES})")
_require(
    NUM_KPHASES_PV_HALF * CFG.TILE_K_HW_BMM2 == P_D_BLOCK,
    f"K-split boundary ({NUM_KPHASES_PV_HALF * CFG.TILE_K_HW_BMM2}) must equal P SMEM half width P_D_BLOCK ({P_D_BLOCK})",
)
_require(pHalfXferElems == P_BLOCK_BYTES, f"P half ({pHalfXferElems} elems) must equal one TMA chunk P_BLOCK_BYTES ({P_BLOCK_BYTES})")


_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
# qtrim variant: collapses the KV loop for CGA tiles entirely past the
# per-batch actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q


_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload


from cudnn.sdpa.fwd.kernels.thd_sm100 import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
# The setup kernel builds the THD metadata buffer DEVICE-side from the
# caller's length tensors and the adapter launches the plan-time envelope
# grid (issue #552) — no length ever reaches the host.
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# === PackGQA ===
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE

# === KV split ===
#
# Mechanics live in _common_sm100.make_split_helpers, shared with the other
# SM100 prefill flavors: each Q tile's KV loop range is cut into SPLIT_KV
# contiguous chunks, each run as its own persistent tile, and each writing a
# normalized partial O + its own LSE into a split-major workspace that
# split_combine_sm100 folds with the exact log-sum-exp identity.  At
# SPLIT_KV == 1 every closure folds away and this is the classic kernel.
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

    _ALIAS_ELEMS = max(
        qBufferElems,
        oBufferElems,
        CFG.STAGES_KV * kBufferElems,
        CFG.STAGES_KV * vBufferElems,
    )
    sAliased_raw = cutlass.Array(STORAGE_DTYPE, _ALIAS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sQ_raw = sAliased_raw
    sK_raw = sAliased_raw
    sV_raw = sAliased_raw
    sO_raw = sAliased_raw

    sP_xfer_raw = cutlass.Array(P_STORAGE_DTYPE, CFG.XFER_STAGES * pXferElems, alignment=128, space=cutlass.AddressSpace.smem)
    sAlpha_xfer_raw = cutlass.Array(cutlass.Float32, CFG.XFER_STAGES * CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)
    sStats_xfer_raw = cutlass.Array(cutlass.Float32, 2 * CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)
    sLSE_raw = cutlass.Array(cutlass.Float32, CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)

    sQ = SmemTile(
        base=sQ_raw,
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
        tma_loads_per_tile=TMA_QK_ITERS,
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
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
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

    bars = _make_d512_sm100_bars(CFG, N_O_CHUNKS)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    sg0_mcast_mask = cutlass.Int32(0x3)
    tma_mcast_mask = (cutlass.Int16(1) << cta_id_x.to(cutlass.Int16)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    sg_id = cta_id_x // cutlass.Int32(CFG.CTA_MMA)
    is_sg0 = sg_id == cutlass.Int32(0)
    is_sg1 = sg_id == cutlass.Int32(1)
    cross_sg_peer = cta_id_x ^ cutlass.Int32(CFG.CTA_MMA)

    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_tma_q_full.init()
            bars.mb_tma_q_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_k_full[ks].init()
                bars.mb_tma_k_empty[ks].init()
                bars.mb_tma_v_full[ks].init()
                bars.mb_tma_v_empty[ks].init()

            for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                bars.mb_s_acc_empty[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()

                p_full_init = cutlass.Int32(
                    arith.select(
                        is_leader.ir_value(),
                        cutlass.Int32(CFG.CTA_MMA).ir_value(),
                        cutlass.Int32(CFG.ONE_LANE).ir_value(),
                    )
                )
                for h in cutlass.range_constexpr(P_XFER_HALVES):
                    bars.mb_p_xfer_full[p * P_XFER_HALVES + h].init(override_count=p_full_init)
                    bars.mb_p_xfer_empty[p * P_XFER_HALVES + h].init()
                bars.mb_alpha_xfer_full[p].init()
                bars.mb_alpha_xfer_empty[p].init()

            bars.mb_stats_xfer_full.init()
            bars.mb_stats_xfer_empty.init()

            for c in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_tma_o_full[c].init()
            bars.mb_tma_o_empty.init()

            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()

            bars.mb_q_utccp_done.init()

            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS)

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    if warp_idx >= cutlass.Int32(CFG.SOFTMAX_WG0_BASE) and warp_idx < cutlass.Int32(CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _compute_warp_group(
            is_sg0=is_sg0,
            is_sg1=is_sg1,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sQ=sQ,
            sO=sO,
            sP_xfer_raw=sP_xfer_raw,
            sAlpha_xfer_raw=sAlpha_xfer_raw,
            sStats_xfer_raw=sStats_xfer_raw,
            sLSE_raw=sLSE_raw,
            bars=bars,
            sched=sched,
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
            cross_sg_peer=cross_sg_peer,
            qh_per_kh=qh_per_kh,
        )

    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp_group(
                is_sg0=is_sg0,
                is_sg1=is_sg1,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                sK=sK,
                sV=sV,
                sP_xfer_raw=sP_xfer_raw,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_tensor=seq_q_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                sg0_mcast_mask=sg0_mcast_mask,
                cta_in_pair=cta_in_pair,
                leader_cta_id=leader_cta_id,
                qh_per_kh=qh_per_kh,
            )
        else:
            _mma_warp_non_leader(
                is_sg0=is_sg0,
                is_sg1=is_sg1,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_tensor=seq_q_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
                leader_cta_id=leader_cta_id,
                qh_per_kh=qh_per_kh,
            )

    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            is_sg0=is_sg0,
            is_sg1=is_sg1,
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_tensor=seq_q_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            is_sg1=is_sg1,
            tma_o_desc=tma_o_desc,
            sO=sO,
            bars=bars,
            sched=sched,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            cta_in_pair=cta_in_pair,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta)


@cute.jit
def _sg0_softmax_kv_iter(
    apply_mask: bool,
    kv_loop,
    sg0_xfer_state,
    bmm1_done_state,
    total_max,
    total_max_safe,
    total_sum_vec,
    tmem_base_addr,
    bars,
    sP_xfer_raw,
    sAlpha_xfer_raw,
    q_abs,
    eff_seqlen_kv,
    eff_seqlen_q,
    scale_log2,
    tid_in_wg,
    is_lead_warp,
    leader_cta_id,
    cross_sg_peer,
):
    cur_parity_S = sg0_xfer_state.idx
    cur_phase_S = sg0_xfer_state.phase
    bars.mb_bmm1_done[bmm1_done_state.idx].wait(bmm1_done_state.phase)
    sg0_xfer_state = advance(sg0_xfer_state, CFG.XFER_STAGES)
    bmm1_done_state = advance(bmm1_done_state, CFG.XFER_STAGES)

    s_addr_base = tmem_base_addr + cur_parity_S * cutlass.Int32(LAYOUT.S_ACC_COLS)

    if cutlass.const_expr(apply_mask):
        kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
        raw_chunks = [
            nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * SOFTMAX_CHUNK), cutlass.Float32),
                num=SOFTMAX_CHUNK,
            )
            for c in range(SOFTMAX_N_CHUNKS_LOAD)
        ]
        causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
        chunks_S = [
            apply_mask_chunk(
                raw_chunks[c],
                q_abs,
                kv_col_base + cutlass.Int32(c * SOFTMAX_CHUNK),
                eff_seqlen_kv,
                CFG.WINDOW_LEFT,
                CFG.MASK_FLAGS,
                N=SOFTMAX_CHUNK,
                bottom_right=CFG.BOTTOM_RIGHT,
                causal_diag=causal_diag,
                window_right=CFG.WINDOW_RIGHT,
                mask_value=float("-inf"),
            )
            for c in range(SOFTMAX_N_CHUNKS_LOAD)
        ]
        chunks_max = [row_max_reduction(chunks_S[c]) for c in range(SOFTMAX_N_CHUNKS_LOAD)]
        reg_S_vec = vec_concat(chunks_S)
        current_max_raw = chunks_max[0]
        for m in chunks_max[1:]:
            current_max_raw = cute.math.max(current_max_raw, m)
        reg_S_tile = RegTile(reg_S_vec, size=CFG.TILE_N)
    else:
        reg_S_tile = tmem_load_tile(s_addr_base, num_elems=CFG.TILE_N)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        current_max_raw = row_max_reduction(reg_S_tile.vec)
    current_max = current_max_raw * scale_log2  # -inf when the whole iteration is masked

    bars.mb_s_acc_empty[cur_parity_S].arrive(cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id)

    # total_max starts at -inf: a live iteration always clears the threshold
    # (real - (-inf) = +inf), a fully-masked one never does (-inf - x = -inf
    # or NaN; ordered > is false for both), so dead iterations never move it.
    update_cond = (current_max - total_max) > RESCALE_THRESHOLD_F32
    total_max = cutlass.Float32(
        arith.select(
            update_cond.ir_value(),
            current_max.ir_value(),
            total_max.ir_value(),
        )
    )
    # Canonical 0-substitution at point of use, on BOTH alpha operands;
    # min(., 0) guards the dead->alive drop of the safe max (total_sum is
    # still 0 there).  total_max_safe starts at -inf: iter-0 alpha = 0.
    new_total_max_safe = row_max_for_exp2(total_max)
    alpha = cute.math.exp2(
        cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)),
        fastmath=True,
    )
    total_max_safe = new_total_max_safe

    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    total_sum_vec = total_sum_vec * alpha_pair
    p_slot_base = cur_parity_S * cutlass.Int32(pXferElems)
    for chunk in cutlass.range_constexpr(P_XFER_HALVES):
        chunk_slot = cur_parity_S * cutlass.Int32(P_XFER_HALVES) + cutlass.Int32(chunk)
        bars.mb_p_xfer_empty[chunk_slot].wait(cur_phase_S)

        reg_S_chunk = reg_S_tile[chunk * P_D_BLOCK : (chunk + 1) * P_D_BLOCK].vec
        reg_P_fp32_chunk = cute.math.exp2(reg_S_chunk * scale_log2 - total_max_safe, fastmath=True)
        reg_P_half_chunk = reg_P_fp32_chunk.to(P_STORAGE_DTYPE)
        total_sum_vec = total_sum_vec + row_reduction_pair(reg_P_fp32_chunk)

        smem_off = cutlass.Int32(chunk * P_BLOCK_BYTES) + tid_in_wg * cutlass.Int32(P_D_BLOCK)
        smem_ptr = sP_xfer_raw.subview(p_slot_base + smem_off).data_ptr()
        smem_ptr.store_swizzled(reg_P_half_chunk, alignment=64, swizzle=P_SMEM_SWIZZLE)

        if cutlass.const_expr(chunk == 0):
            alpha_slot = sAlpha_xfer_raw.subview(cur_parity_S * cutlass.Int32(CFG.TILE_M) + tid_in_wg)
            alpha_slot.store(alpha)

        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=128)

        ship_pred = is_lead_warp & nvvm.elect_sync()

        if cutlass.const_expr(chunk == 0):
            local_alpha_src = sAlpha_xfer_raw.subview(cur_parity_S * cutlass.Int32(CFG.TILE_M))
            peer_alpha_dst = nvvm.mapa(local_alpha_src, cross_sg_peer, addrspace=7)
            peer_alpha_full_mbar = nvvm.mapa(bars.mb_alpha_xfer_full[cur_parity_S].smem_ptr, cross_sg_peer, addrspace=7)
            cp_async_bulk_shared_cluster_shared_cta(
                peer_alpha_dst,
                local_alpha_src,
                peer_alpha_full_mbar,
                alphaXferBytes,
                pred=ship_pred,
            )
        local_p_src = sP_xfer_raw.subview(p_slot_base + cutlass.Int32(chunk * pHalfXferElems))
        peer_p_dst = nvvm.mapa(local_p_src, cross_sg_peer, addrspace=7)
        peer_p_full_mbar = nvvm.mapa(bars.mb_p_xfer_full[chunk_slot].smem_ptr, cross_sg_peer, addrspace=7)
        cp_async_bulk_shared_cluster_shared_cta(
            peer_p_dst,
            local_p_src,
            peer_p_full_mbar,
            pHalfXferBytes,
            pred=ship_pred,
        )

    return (sg0_xfer_state, bmm1_done_state, total_max, total_max_safe, total_sum_vec)


@cute.jit
def _compute_warp_group(
    is_sg0,
    is_sg1,
    seqlen_q,
    seqlen_kv,
    scale_log2,
    tmem_ptr_i32,
    sQ,
    sO,
    sP_xfer_raw,
    sAlpha_xfer_raw,
    sStats_xfer_raw,
    sLSE_raw,
    bars,
    sched,
    lse_tensor,
    sinks_tensor,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
    cross_sg_peer,
    qh_per_kh,
):
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base_addr = tmem_ptr_i32.load()
    tmem_raw = nvvm.make_tmem_ptr(tmem_base_addr, cutlass.Int8)

    tid_in_wg = cute.arch.thread_idx()[0]
    wid_in_wg = tid_in_wg // cutlass.Int32(32)
    is_lead_warp = wid_in_wg == cutlass.Int32(0)

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

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_unmasked_lo = cutlass.Int32(0)
        kv_unmasked_hi = seqlen_kv // cutlass.Int32(CFG.TILE_N)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
        eff_seqlen_kv = seqlen_kv
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        # Unmasked, so this split's whole slice is the unmasked band.
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        kv_unmasked_lo = kv_left
        kv_unmasked_hi = kv_right
        eff_seqlen_kv = seqlen_kv
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH)
        kv_left = bounds_init.left
        kv_unmasked_lo = bounds_init.unmasked_lo
        kv_unmasked_hi = bounds_init.unmasked_hi
        kv_right = bounds_init.right

    _O_EPI_SWIZZLE = cutlass.Swizzle(3, 4, 3)

    O_EPI_BLOCK_SIZE = 64 // CFG.BPE_O
    O_TMA_ITERS = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    O_D_BLOCK = CFG.TILE_O // O_TMA_ITERS
    O_TMA_GRANU_ELEMS = CFG.TILE_M * O_D_BLOCK
    O_BLOCKS_PER_SUB = 128 // O_EPI_BLOCK_SIZE
    O_CHUNK_ELEMS = 128 // CFG.BPE_O

    bmm1_done_state = PipelineState.start(phase=0)
    sg0_xfer_state = PipelineState.start(phase=1)
    stats_xfer_empty_state = PipelineState.start(phase=1)

    alpha_full_state = PipelineState.start(phase=0)
    sg1_bmm2_done_state = PipelineState.start(phase=0)
    stats_xfer_full_state = PipelineState.start(phase=0)
    epilogue_state = PipelineState.start(phase=1)

    total_max = NEG_INF_F32
    total_max_safe = NEG_INF_F32
    total_sum_vec = cutlass.Vector.from_elements(
        (cutlass.Float32(0.0), cutlass.Float32(0.0)),
        cutlass.Float32,
    )

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            total_max = NEG_INF_F32
            total_max_safe = NEG_INF_F32
            total_sum_vec = cutlass.Vector.from_elements(
                (cutlass.Float32(0.0), cutlass.Float32(0.0)),
                cutlass.Float32,
            )

            # PackGQA: q_abs is the row's TOKEN index (row // G): every mask
            # predicate downstream is a token-space compare, and all G rows of one
            # token share it.
            q_abs = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))

            if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
                pass
            else:
                if cutlass.const_expr(CFG.MASK_FLAGS == 0):
                    for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_max_safe, total_sum_vec = _sg0_softmax_kv_iter(
                            False,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_max_safe,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )
                else:
                    for _kv in cutlass.range(kv_left, kv_unmasked_lo, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_max_safe, total_sum_vec = _sg0_softmax_kv_iter(
                            True,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_max_safe,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )
                    for _kv in cutlass.range(kv_unmasked_lo, kv_unmasked_hi, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_max_safe, total_sum_vec = _sg0_softmax_kv_iter(
                            False,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_max_safe,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )
                    for _kv in cutlass.range(kv_unmasked_hi, kv_right, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_max_safe, total_sum_vec = _sg0_softmax_kv_iter(
                            True,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_max_safe,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )

            bars.mb_stats_xfer_empty.wait(stats_xfer_empty_state.phase)
            stats_xfer_empty_state = advance(stats_xfer_empty_state, 1)

            final_sum = total_sum_vec[0] + total_sum_vec[1]
            stats_sum_slot = sStats_xfer_raw.subview(tid_in_wg)
            stats_max_slot = sStats_xfer_raw.subview(cutlass.Int32(CFG.TILE_M) + tid_in_wg)
            stats_sum_slot.store(final_sum)
            stats_max_slot.store(total_max_safe)

            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=128)

            ship_pred = is_lead_warp & nvvm.elect_sync()

            peer_stats_dst = nvvm.mapa(sStats_xfer_raw, cross_sg_peer, addrspace=7)
            peer_stats_full_mbar = nvvm.mapa(bars.mb_stats_xfer_full.smem_ptr, cross_sg_peer, addrspace=7)
            cp_async_bulk_shared_cluster_shared_cta(
                peer_stats_dst,
                sStats_xfer_raw,
                peer_stats_full_mbar,
                statsXferBytes,
                pred=ship_pred,
            )
        else:
            tmem_O_base = tmem_base_addr + cutlass.Int32(LAYOUT.O_OFF)

            if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
                bars.mb_empty_mainloop.arrive_on_peer(leader_cta_id, pred=is_lead_warp & nvvm.elect_sync())
            else:
                cur_parity_0 = alpha_full_state.idx
                bars.mb_bmm2_ready[cur_parity_0 * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id)
                bars.mb_bmm2_ready[cur_parity_0 * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(
                    cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id
                )

                bars.mb_alpha_xfer_full[cur_parity_0].arrive(n_bytes=alphaXferBytes, pred=is_lead_warp & nvvm.elect_sync())
                bars.mb_alpha_xfer_full[cur_parity_0].wait(alpha_full_state.phase)
                alpha_full_state = advance(alpha_full_state, CFG.XFER_STAGES)

                bars.mb_alpha_xfer_empty[cur_parity_0].arrive_on_peer(cross_sg_peer)

                CORR_BLOCK_SIZE = 16
                CORR_BLOCKS_TOTAL = CFG.TILE_O // CORR_BLOCK_SIZE
                CORR_BLOCKS_PER_HALF = CORR_BLOCKS_TOTAL // 2

                for _kv in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                    cur_parity = alpha_full_state.idx

                    bars.mb_alpha_xfer_full[cur_parity].arrive(n_bytes=alphaXferBytes, pred=is_lead_warp & nvvm.elect_sync())
                    bars.mb_alpha_xfer_full[cur_parity].wait(alpha_full_state.phase)
                    alpha_full_state = advance(alpha_full_state, CFG.XFER_STAGES)

                    bars.mb_bmm2_done[sg1_bmm2_done_state.idx].wait(sg1_bmm2_done_state.phase)
                    sg1_bmm2_done_state = advance(sg1_bmm2_done_state, CFG.XFER_STAGES)

                    alpha_addr = sAlpha_xfer_raw.subview(cur_parity * cutlass.Int32(CFG.TILE_M) + tid_in_wg)
                    alpha_corr = alpha_addr.load()

                    alpha_is_one = alpha_corr == cutlass.Float32(1.0)
                    all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                    if ~all_alpha_one:
                        for block in cutlass.range_constexpr(CORR_BLOCKS_PER_HALF):
                            o_off = tmem_O_base + cutlass.Int32(block * CORR_BLOCK_SIZE)
                            o_chunk = nvvm.tcgen05_ld(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                num=CORR_BLOCK_SIZE,
                            )
                            o_scaled = vec_scale_pair(o_chunk, alpha_corr, CORR_BLOCK_SIZE)
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                o_scaled,
                            )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                    bars.mb_alpha_xfer_empty[cur_parity].arrive_on_peer(cross_sg_peer)
                    bars.mb_bmm2_ready[cur_parity * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id)

                    if ~all_alpha_one:
                        for block in cutlass.range_constexpr(CORR_BLOCKS_PER_HALF):
                            block_idx = CORR_BLOCKS_PER_HALF + block
                            o_off = tmem_O_base + cutlass.Int32(block_idx * CORR_BLOCK_SIZE)
                            o_chunk = nvvm.tcgen05_ld(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                num=CORR_BLOCK_SIZE,
                            )
                            o_scaled = vec_scale_pair(o_chunk, alpha_corr, CORR_BLOCK_SIZE)
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                o_scaled,
                            )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                    bars.mb_bmm2_ready[cur_parity * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(
                        cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id
                    )

            bars.mb_bmm2_done[sg1_bmm2_done_state.idx].wait(sg1_bmm2_done_state.phase)
            sg1_bmm2_done_state = advance(sg1_bmm2_done_state, CFG.XFER_STAGES)

            bars.mb_tma_o_empty.wait(epilogue_state.phase)
            epilogue_state = advance(epilogue_state, 1)

            bars.mb_stats_xfer_full.arrive(n_bytes=statsXferBytes, pred=is_lead_warp & nvvm.elect_sync())
            bars.mb_stats_xfer_full.wait(stats_xfer_full_state.phase)
            stats_xfer_full_state = advance(stats_xfer_full_state, 1)

            ell_addr = sStats_xfer_raw.subview(tid_in_wg)
            max_addr = sStats_xfer_raw.subview(cutlass.Int32(CFG.TILE_M) + tid_in_wg)
            final_ell = ell_addr.load()
            final_max = max_addr.load()

            bars.mb_stats_xfer_empty.arrive_on_peer(cross_sg_peer, pred=is_lead_warp & nvvm.elect_sync())

            q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
            row_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE) + (tid_in_wg % cutlass.Int32(HEADS_PER_TILE))
            LN2 = cutlass.Float32(0.6931471805599453)
            if cutlass.const_expr(CFG.HAS_SINK):
                sinks_arr = cutlass.make_array_view(sinks_tensor)
                sink_logit = sinks_arr[row_head_idx]
                final_max_nat = final_max * LN2
                new_max_nat = cute.math.max(final_max_nat, sink_logit)
                scale_sink = cute.math.exp(final_max_nat - new_max_nat, fastmath=True)
                new_sum = final_ell * scale_sink + cute.math.exp(sink_logit - new_max_nat, fastmath=True)
                beta = scale_sink / new_sum
                lse = new_max_nat + cute.math.log(new_sum, fastmath=True)
            else:
                final_ell_safe = cute.math.max(final_ell, cutlass.Float32(1e-30))
                beta = cutlass.Float32(1.0) / final_ell_safe
                lse = final_max * LN2 + cute.math.log(final_ell_safe, fastmath=True)
                # Dead row (no valid KV column at all): O := 0, LSE := -inf.
                # final_ell >= 1 for any alive row so this never fires spuriously.
                row_dead = final_ell <= cutlass.Float32(0.0)
                neg_inf_lse = cutlass.Float32(float("-inf"))
                beta = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), beta.ir_value()))
                lse = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse.ir_value()))

            if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
                # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b]
                # write O := 0 / LSE := -inf.  Applied AFTER the sink branch on
                # purpose — a trimmed row is dead even with a sink.  Per-batch
                # q lens come in via the dedicated seq_q_lens_tensor parameter.
                # (q_row_global — the row's token index — is computed above,
                # before the sink fold, which needs the row's head; beta feeds
                # the O normalization in the store loop below.)
                _sq_arr = cutlass.make_array_view(seq_q_lens_tensor)
                _q_len_b = cutlass.Int32(_sq_arr[batch_idx])
                row_trim = q_row_global >= _q_len_b
                neg_inf_trim = cutlass.Float32(float("-inf"))
                beta = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), beta.ir_value()))
                lse = cutlass.Float32(arith.select(row_trim.ir_value(), neg_inf_trim.ir_value(), lse.ir_value()))

            sO_base = sO[0].base

            for b in cutlass.range_constexpr(CFG.TILE_O // O_EPI_BLOCK_SIZE):
                b_intra = b & (O_BLOCKS_PER_SUB - 1)
                b_sub = b // O_BLOCKS_PER_SUB
                tmem_sub = ((b_sub & 1) << 1) | ((b_sub & 2) >> 1)
                tmem_block = tmem_sub * O_BLOCKS_PER_SUB + b_intra

                o_half = cutlass.Vector.from_elements(
                    tuple(OUT_STORAGE_DTYPE(0.0) for _ in range(O_EPI_BLOCK_SIZE)),
                    OUT_STORAGE_DTYPE,
                )
                if cutlass.const_expr(not MAY_BE_EMPTY) or (kv_right > kv_left):
                    o_addr = tmem_O_base + cutlass.Int32(tmem_block * O_EPI_BLOCK_SIZE)
                    o_fp32 = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_EPI_BLOCK_SIZE,
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    o_scaled = o_fp32 * beta
                    o_half = o_scaled.to(OUT_STORAGE_DTYPE)

                col_offset_const = (b * O_EPI_BLOCK_SIZE) % O_D_BLOCK
                block_idx_const = (b * O_EPI_BLOCK_SIZE) // O_D_BLOCK
                block_offset_const = block_idx_const * O_TMA_GRANU_ELEMS
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(O_D_BLOCK)
                smem_ptr = sO_base.subview(smem_offset).data_ptr()
                smem_ptr.store_swizzled(
                    o_half,
                    alignment=64,
                    swizzle=_O_EPI_SWIZZLE,
                )

                if ((b + 1) * O_EPI_BLOCK_SIZE) % O_CHUNK_ELEMS == 0:
                    chunk = (b * O_EPI_BLOCK_SIZE) // O_CHUNK_ELEMS
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_tma_o_full[chunk].arrive()

            if cutlass.const_expr(lse_tensor is None):
                pass  # has_lse=False: the Stats store is compiled out
            elif cutlass.const_expr(CFG.THD_VARLEN):
                _cu = cutlass.make_array_view(seq_kv_lens_tensor)
                _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
                _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
                if q_row_global < _s_q_b:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    if cutlass.const_expr(len(lse_tensor.shape) == 2):
                        # token-major packed (T, H)
                        lse_row = lse_arr[_cu_q_b + q_row_global, :]
                        lse_row[head_idx] = lse
                    else:
                        # head-major packed (1, QH, head_stride)
                        lse_row = lse_arr[cutlass.Int32(0), head_idx, :]
                        lse_row[_cu_q_b + q_row_global] = lse
            else:
                if q_row_global < seqlen_q:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                    lse_arr[lse_batch, row_head_idx, q_row_global] = lse

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load())
        nxt_hb = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load())
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load())
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
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_next.left
            kv_unmasked_lo = bounds_next.unmasked_lo
            kv_unmasked_hi = bounds_next.unmasked_hi
            kv_right = bounds_next.right

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        if is_sg0:
            if is_lead_warp:
                if nvvm.elect_sync():
                    for _p in cutlass.range_constexpr(CFG.XFER_STAGES):
                        for _c in cutlass.range_constexpr(P_XFER_HALVES):
                            bars.mb_p_xfer_empty[sg0_xfer_state.idx * cutlass.Int32(P_XFER_HALVES) + cutlass.Int32(_c)].wait(sg0_xfer_state.phase)
                        bars.mb_alpha_xfer_empty[sg0_xfer_state.idx].wait(sg0_xfer_state.phase)
                        sg0_xfer_state = advance(sg0_xfer_state, CFG.XFER_STAGES)
                    bars.mb_stats_xfer_empty.wait(stats_xfer_empty_state.phase)
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta, pred=is_lead_warp & nvvm.elect_sync())


@cute.jit
def _mma_warp_group(
    is_sg0,
    is_sg1,
    seqlen_q,
    seqlen_kv,
    sQ,
    sK,
    sV,
    sP_xfer_raw,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    sg0_mcast_mask,
    cta_in_pair,
    leader_cta_id,
    qh_per_kh,
):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.BMM2_N_PER_CALL,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
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
        N=CFG.BMM2_N_PER_CALL,
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

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    q_full_state = PipelineState.start(phase=0)
    kv_state_K = PipelineState.start(phase=0)
    s_acc_empty_state = PipelineState.start(phase=1)

    kv_state_V = PipelineState.start(phase=0)
    sg1_mma_state = PipelineState.start(phase=0)
    p_full_state = PipelineState.start(phase=0)
    bmm2_ready_state = PipelineState.start(phase=0)
    bmm2_done_prod_state = PipelineState.start(phase=0)
    empty_mainloop_state = PipelineState.start(phase=0)

    sP_xfer = SmemTile(
        base=sP_xfer_raw,
        elems_per_stage=pXferElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_P,
    )

    desc_Q = sQ[0].desc()
    tmem_Q = tmem_raw.subview(cutlass.Int32(LAYOUT.Q_TMEM_OFF))
    _UTCCP_BYTES_PER_CALL = 16
    _UTCCP_TMEM_COLS_PER_CALL = _UTCCP_BYTES_PER_CALL // 4
    _UTCCP_PER_SUBTILE = CFG.Q_SWZ_BYTES // _UTCCP_BYTES_PER_CALL
    _UTCCP_SUBTILE_DESC_STRIDE = (CFG.TILE_M * CFG.Q_SWZ_BYTES) // _UTCCP_BYTES_PER_CALL
    _UTCCP_N_CALLS = LAYOUT.Q_TMEM_COLS // _UTCCP_TMEM_COLS_PER_CALL
    V_NBLOCK_ADVANCE_ELEMS = BMM2_V_NBLOCK_ADVANCE // CFG.BPE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
                pass
            else:
                bars.mb_tma_q_full.wait(q_full_state.phase)
                q_full_state = advance(q_full_state, 1)

                if nvvm.elect_sync():
                    for _qk in cutlass.range_constexpr(_UTCCP_N_CALLS):
                        _s = _qk // _UTCCP_PER_SUBTILE
                        _kk = _qk % _UTCCP_PER_SUBTILE
                        _desc_off = _s * _UTCCP_SUBTILE_DESC_STRIDE + _kk
                        nvvm.tcgen05_cp(
                            nvvm.Tcgen05CpShape.SHAPE_128X128B,
                            tmem_Q.subview(_qk * _UTCCP_TMEM_COLS_PER_CALL),
                            desc_Q + _desc_off,
                            group=CTA_GROUP_KIND,
                        )
                    bars.mb_q_utccp_done.arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask)

                for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    cur_parity_K = s_acc_empty_state.idx
                    bars.mb_s_acc_empty[cur_parity_K].wait(s_acc_empty_state.phase)
                    s_acc_empty_state = advance(s_acc_empty_state, CFG.XFER_STAGES)

                    bars.mb_tma_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                    desc_K = sK[kv_state_K.idx].desc()
                    s_acc_off = cur_parity_K * cutlass.Int32(LAYOUT.S_ACC_COLS)
                    mma_ts(bmm1_desc, tmem_Q, desc_K, tmem_raw.subview(s_acc_off), accumulate=False)
                    elect_p = nvvm.elect_sync()
                    bars.mb_bmm1_done[cur_parity_K].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
                    bars.mb_tma_k_empty[kv_state_K.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
                    kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                bars.mb_tma_q_empty.arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=nvvm.elect_sync())
        else:
            if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
                bars.mb_empty_mainloop.wait(empty_mainloop_state.phase)
                empty_mainloop_state = advance(empty_mainloop_state, 1)
                bars.mb_bmm2_done[bmm2_done_prod_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=nvvm.elect_sync())
                bmm2_done_prod_state = advance(bmm2_done_prod_state, CFG.XFER_STAGES)
            else:
                for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    cur_parity = sg1_mma_state.idx
                    p_full_h0 = p_full_state.idx
                    p_full_h1 = p_full_state.idx + cutlass.Int32(1)

                    elect_p = nvvm.elect_sync()
                    bars.mb_p_xfer_full[p_full_h0].arrive(n_bytes=pHalfXferBytes, pred=elect_p)
                    bars.mb_p_xfer_full[p_full_h1].arrive(n_bytes=pHalfXferBytes, pred=elect_p)

                    bars.mb_tma_v_full[kv_state_V.idx].wait(kv_state_V.phase)

                    accum_b2 = kv_loop > kv_left

                    desc_P = sP_xfer[cur_parity].desc()
                    desc_V_n0 = sV[kv_state_V.idx].desc()
                    desc_V_n1 = sV[kv_state_V.idx].shifted(V_NBLOCK_ADVANCE_ELEMS).desc()

                    tmem_O_n0 = tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF))
                    tmem_O_n1 = tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + CFG.BMM2_N_PER_CALL))

                    bars.mb_bmm2_ready[bmm2_ready_state.idx].wait(bmm2_ready_state.phase)
                    bmm2_ready_state = advance(bmm2_ready_state, CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS)
                    bars.mb_p_xfer_full[p_full_state.idx].wait(p_full_state.phase)
                    p_full_state = advance(p_full_state, CFG.XFER_STAGES * P_XFER_HALVES)
                    mma_ss(bmm2_desc, desc_P, desc_V_n0, tmem_O_n0, accumulate=accum_b2, k_start=0, k_count=NUM_KPHASES_PV_HALF)
                    bars.mb_bmm2_ready[bmm2_ready_state.idx].wait(bmm2_ready_state.phase)
                    bmm2_ready_state = advance(bmm2_ready_state, CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS)
                    mma_ss(bmm2_desc, desc_P, desc_V_n1, tmem_O_n1, accumulate=accum_b2, k_start=0, k_count=NUM_KPHASES_PV_HALF)

                    bars.mb_p_xfer_empty[cur_parity * cutlass.Int32(P_XFER_HALVES) + cutlass.Int32(0)].arrive(
                        cta_group=CFG.CTA_MMA, mcast_mask=sg0_mcast_mask, pred=nvvm.elect_sync()
                    )

                    bars.mb_p_xfer_full[p_full_state.idx].wait(p_full_state.phase)
                    p_full_state = advance(p_full_state, CFG.XFER_STAGES * P_XFER_HALVES)
                    mma_ss(bmm2_desc, desc_P, desc_V_n0, tmem_O_n0, accumulate=True, k_start=NUM_KPHASES_PV_HALF, k_count=NUM_KPHASES_PV_HALF)
                    mma_ss(bmm2_desc, desc_P, desc_V_n1, tmem_O_n1, accumulate=True, k_start=NUM_KPHASES_PV_HALF, k_count=NUM_KPHASES_PV_HALF)

                    sg1_mma_state = advance(sg1_mma_state, CFG.XFER_STAGES)

                    elect_p = nvvm.elect_sync()
                    bars.mb_bmm2_done[bmm2_done_prod_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
                    bars.mb_tma_v_empty[kv_state_V.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
                    bars.mb_p_xfer_empty[cur_parity * cutlass.Int32(P_XFER_HALVES) + cutlass.Int32(1)].arrive(
                        cta_group=CFG.CTA_MMA, mcast_mask=sg0_mcast_mask, pred=elect_p
                    )
                    kv_state_V = advance(kv_state_V, CFG.STAGES_KV)
                    bmm2_done_prod_state = advance(bmm2_done_prod_state, CFG.XFER_STAGES)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load()
        nxt_hb = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load()
        nxt_v = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load()
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
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_non_leader(
    is_sg0,
    is_sg1,
    seqlen_q,
    seqlen_kv,
    sQ,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
    leader_cta_id,
    qh_per_kh,
):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    tmem_raw_nl = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_Q_nl = tmem_raw_nl.subview(cutlass.Int32(LAYOUT.Q_TMEM_OFF))
    _UTCCP_BYTES_PER_CALL = 16
    _UTCCP_TMEM_COLS_PER_CALL = _UTCCP_BYTES_PER_CALL // 4
    _UTCCP_PER_SUBTILE = CFG.Q_SWZ_BYTES // _UTCCP_BYTES_PER_CALL
    _UTCCP_SUBTILE_DESC_STRIDE = (CFG.TILE_M * CFG.Q_SWZ_BYTES) // _UTCCP_BYTES_PER_CALL
    _UTCCP_N_CALLS = LAYOUT.Q_TMEM_COLS // _UTCCP_TMEM_COLS_PER_CALL

    if is_sg1:
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

        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
            kv_left = cutlass.Int32(0)
            kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
        elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        else:
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
            bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_init.left
            kv_right = bounds_init.right

        nlmma_state = PipelineState.start(phase=0)

        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
                pass
            else:
                for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    for _h in cutlass.range_constexpr(P_XFER_HALVES):
                        cur_slot = nlmma_state.idx
                        bars.mb_p_xfer_full[cur_slot].arrive(n_bytes=pHalfXferBytes, pred=nvvm.elect_sync())
                        bars.mb_p_xfer_full[cur_slot].wait(nlmma_state.phase)
                        nlmma_state = advance(nlmma_state, CFG.XFER_STAGES * P_XFER_HALVES)
                        bars.mb_p_xfer_full[cur_slot].arrive_on_peer(leader_cta_id, pred=nvvm.elect_sync())

            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

            wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
            nxt_q = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load()
            nxt_hb = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load()
            nxt_v = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load()
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
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
            if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
                kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
            elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
                bounds_next = _bounds_for_tile_split(
                    q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH
                )
                kv_left = bounds_next.left
                kv_right = bounds_next.right
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _tmaldg_warp_group(
    is_sg0,
    is_sg1,
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sK,
    sV,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
):
    q_empty_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    q_utccp_done_state = PipelineState.start(phase=0)
    o_empty_for_v_state = PipelineState.start(phase=1)

    tma_q = GmemTileTma(tma_q_desc)
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
    # GQA: K/V are indexed by kv-head, not Q-head; with PackGQA the decoded
    # head_idx is the PACKED head (Q head base = head_idx * G) and q_row_base is
    # in TOKEN units (rows // G).
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
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # Empty KV loop (SWA window past the padded KV tail, or a dead-Q-tile
            # collapse): no loads — but the O∪V SMEM alias gate must stay in
            # phase.  The sg1 compute epilogue stores (trimmed) O and TMA-STG
            # drains it — flipping mb_tma_o_empty — on EVERY tile, empty ones
            # included, while the V-side wait below only runs on full tiles.
            # Without this bookkeeping advance a single empty tile leaves
            # o_empty_for_v_state one phase behind for the rest of the
            # persistent loop, so later V loads no longer wait for the O drain
            # sharing their SMEM slab (live-tile corruption).  Advance-only is
            # sufficient: this tile issues no V loads, so no actual wait is
            # needed (harmlessly tracked on sg0 CTAs, which never consume it).
            o_empty_for_v_state = advance(o_empty_for_v_state, 1)
        else:
            if is_sg0:
                bars.mb_tma_q_empty.wait(q_empty_state.phase)
                q_empty_state = advance(q_empty_state, 1)
                bars.mb_tma_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                tma_load_tile(
                    sQ[0],
                    tma_q(cutlass.Int32(0), q_head_idx, q_row_base + q_seq_off, tma_batch),
                    bars.mb_tma_q_full.smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

                bars.mb_q_utccp_done.wait(q_utccp_done_state.phase)
                q_utccp_done_state = advance(q_utccp_done_state, 1)

                for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                    bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                    bars.mb_tma_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                    tma_load_tile(
                        sK[kv_state.idx],
                        tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                        bars.mb_tma_k_full[kv_state.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                    )
                    kv_state = advance(kv_state, CFG.STAGES_KV)
            else:
                bars.mb_tma_o_empty.wait(o_empty_for_v_state.phase)
                o_empty_for_v_state = advance(o_empty_for_v_state, 1)

                for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                    bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                    bars.mb_tma_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                    tma_load_tile(
                        sV[kv_state.idx],
                        tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                        bars.mb_tma_v_full[kv_state.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                    )
                    kv_state = advance(kv_state, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load())
        nxt_hb = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load())
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load())
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
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_tensor)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        if is_sg0:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
            bars.mb_tma_q_empty.wait(q_empty_state.phase)
        else:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _tmastg_warp_group(
    is_sg1,
    tma_o_desc,
    sO,
    bars,
    sched,
    n_q_supers,
    n_qh,
    n_batch,
    cta_in_pair,
    seq_kv_lens_tensor,
    o_desc_words,
    seqlen_kv,
    qh_per_kh,
):
    o_full_state = PipelineState.start(phase=0)

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
        if is_sg1:
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_tma_o_full[chunk].wait(o_full_state.phase)

            q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
            q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
            # KV split: partials stack split-major on the workspace BATCH axis.
            o_batch = _partial_batch(batch_idx, split_idx, n_batch)
            if cutlass.const_expr(CFG.THD_VARLEN):
                # DEAD unit (batch == n_batch, envelope grid — issue #552): no O
                # rows exist and descriptor slot n_batch is never built, so skip
                # the store; the barrier protocol below still runs.
                if batch_idx < n_batch:
                    o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                    o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), q_head_idx, q_row_coord, cutlass.Int32(0))
                    tma_store_tile(sO[0], o_slice)
            else:
                tma_store_tile(
                    sO[0],
                    tma_o(cutlass.Int32(0), q_head_idx, q_row_coord, o_batch),
                )
            tma_store_commit()
            tma_store_wait(0)

            bars.mb_tma_o_empty.arrive()
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

            o_full_state = advance(o_full_state, 1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load()
        nxt_hb = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load()
        nxt_v = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load()
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
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


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
    # Dense padded-Q trim: separate (B,)-int32 per-batch Q lengths; None
    # (and absent from the compiled ABI) unless CFG.SEQ_Q_LENS_PRESENT.
    seq_q_lens_tensor: Optional[cute.Tensor] = None,
    # THD device metadata build (issue #552): the CALLER's Q/KV length
    # tensors — (B,) per-batch lengths or (B+1,) cu prefix sums, per side via
    # thd_lens_form (bit 0: Q is cu, bit 1: KV is cu) — consumed only by the
    # setup kernel, which writes the [kv|cu_q|cu_k] metadata buffer
    # (seq_kv_lens_tensor) device-side. None (folded out of the ABI) for
    # dense graphs.
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(CFG.THD_VARLEN):
        # Packed token totals are runtime values (dynamic extents); the
        # problem_size slots are 0 by contract.
        SQ = q_tensor.shape[1]
        SKV = k_tensor.shape[1]

    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    if cutlass.const_expr(CFG.PACK_GQA and q_tensor.shape[2] != k_tensor.shape[2] * CFG.QH_PER_KH):
        raise ValueError(f"CFG.QH_PER_KH ({CFG.QH_PER_KH}) does not match tensor head extents H_q={q_tensor.shape[2]}, H_kv={k_tensor.shape[2]}")
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor,
        box_dims=qk_box_q,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor,
        box_dims=qk_box_k,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.V_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=vo_box_o,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    # PackGQA: SQ*G packed rows per packed head, and QH/G packed heads.
    rows_per_cluster = CGA_TILE_M
    q_clusters = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CGA_M
    q_supers = q_clusters * CFG.CTA_MMA
    if cutlass.const_expr(CFG.THD_VARLEN):
        # ENVELOPE: the packed-O row stride is QH * ACTUAL d_v (o_tensor's
        # static inner extent), not QH * TILE_O — the per-batch descriptor
        # bases must step in real rows or every batch >= 1 lands OOB.
        _build_thd_meta_o_descs_kernel(
            o_tensor,
            tma_o_desc,
            o_desc_words,
            seq_kv_lens_tensor,
            thd_q_lens_tensor,
            thd_kv_lens_tensor,
            thd_lens_form,
            cutlass.Int32(QH),
            cutlass.Int32(B),
            cutlass.Int32(o_tensor.stride[1]),
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        # KV split rides the BATCH axis: z = batch + split*B.
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
        seq_q_lens_tensor,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
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
    q_stride: Optional[tuple] = None,
    k_stride: Optional[tuple] = None,
    v_stride: Optional[tuple] = None,
    o_stride: Optional[tuple] = None,
    lse_stride: Optional[tuple[int, int, int]] = None,
) -> Callable:
    """ENVELOPE: ``d_qk`` / ``d_v`` are the ACTUAL head dims (defaults = full
    TILE_K / TILE_O). TMA descriptors carry these extents while the tile box
    stays the compile-time TILE geometry: loads past d_qk / d_v zero-fill
    (exact zeros in the QK^T / P·V contractions), O stores past d_v clip.
    The Q∪V∪O SMEM alias slabs keep their full compile-time extents — only
    the GMEM descriptor extents change. d * BPE must be a 16-byte multiple
    (TMA global-stride rule -> d % 8).

    THD/varlen: ``sq``/``skv`` are IGNORED — the packed token totals are
    runtime values (they change every step under continuous batching), so the
    token extents compile DYNAMIC (``cute.sym_int``) and the cache key stays
    plan-time-only; callers must not pass them. THD strides carry a ZERO batch
    stride (the real view's batch stride is ``t_q * token_stride``, a runtime
    value; the fake rebuilds it symbolically — batch extent 1 never steps)."""
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d512 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d512 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_stride is not None and (CFG.THD_VARLEN or SPLIT_KV > 1):
        raise ValueError("dense LSE strides are not valid for THD or split-KV workspaces")
    _fake_batch = 1 if CFG.THD_VARLEN else b
    if CFG.THD_VARLEN:
        # Dynamic packed token totals: one symbol per ragged group (Q/O and
        # the LSE share t_q; K/V share t_kv), so a new total re-binds the same
        # compiled artifact instead of minting a new one (issue #552).
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)
    _o_batch = _fake_batch * SPLIT_KV
    _lse_batch = b * SPLIT_KV

    def _fake_bshd(shape, stride, dtype=STORAGE_DTYPE, bpe=CFG.BPE):
        if stride is None:
            return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)
        if stride[3] != 1:
            raise ValueError(f"declared stride {stride}: the head dim must be innermost-contiguous (stride[3] == 1)")
        for axis in (1, 2):  # seq/head global strides feed TMA: 16-byte rule
            if (stride[axis] * bpe) % 16 != 0:
                raise ValueError(f"declared stride {stride} axis {axis} must be a 16-byte multiple at BPE={bpe} (TMA global-stride rule)")
        if CFG.THD_VARLEN:
            # Batch stride = tokens * token_stride (`_thd_view`'s envelope),
            # a runtime value: rebuild it from the dynamic token extent.
            return cute.runtime.make_fake_tensor(dtype, shape, (shape[1] * stride[1], stride[1], stride[2], stride[3]), assumed_align=16)
        return cute.runtime.make_fake_tensor(dtype, shape, tuple(stride), assumed_align=16)

    fake_q = _fake_bshd((_fake_batch, sq, qh, d_qk), q_stride)
    fake_k = _fake_bshd((_fake_batch, skv, kh, d_qk), k_stride)
    fake_v = _fake_bshd((_fake_batch, skv, kh, d_v), v_stride)
    fake_o = _fake_bshd((_o_batch, sq, qh, d_v), o_stride, dtype=OUT_STORAGE_DTYPE, bpe=CFG.BPE_O)
    if not has_lse:
        # No Stats output: the LSE argument is None-specialized and the store
        # is compiled out entirely — no dummy buffer exists at any level.
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride require has_lse=True")
        fake_lse = None
    elif CFG.THD_VARLEN:
        # Packed ragged-Stats LSE in the caller's declared layout (align 4: the
        # store is scalar f32 and the caller's Stats buffer only guarantees
        # element alignment). Token-major (the default — cuDNN's TH1 ragged
        # Stats recipe) = its natural packed rank-2 (T, H) view; head-major =
        # the kernels' native rank-3 (1, QH, head_stride) packing with
        # head_stride >= T (compact when 0). The epilogue store branches on
        # the STATIC rank, so the layout is fully encoded in this fake tensor
        # — no template parameter.
        if lse_head_major:
            # head_stride covering t_q is validated at execute (t_q is a
            # runtime value); 0 = compact = the dynamic token total itself.
            _lse_hs = lse_head_stride if lse_head_stride else sq
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (1, qh, _lse_hs),
                stride_order=(2, 1, 0),
                assumed_align=4,
            )
        else:
            if lse_head_stride:
                raise ValueError("lse_head_stride is head-major-only (token-major (T, H) is compact)")
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (sq, qh),
                stride_order=(1, 0),
                assumed_align=4,
            )
    else:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride are THD-only")
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, (_lse_batch, qh, sq), lse_stride, assumed_align=4)
            if lse_stride is not None
            else cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (_lse_batch, qh, sq),
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
    _skv_len = (3 * b + 2) if CFG.THD_VARLEN else b
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (_skv_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Dense padded-Q trim: SEPARATE (B,)-int32 per-batch Q lengths parameter
    # (cuDNN SEQLEN_Q / FA seqused_q style); None folds it out of the ABI when
    # the flag is off.  assumed_align=4 — the caller's tensor is bound
    # directly (no repack), so only natural int32 alignment is required.
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
    _odesc_len = (b * _TENSOR_MAP_QWORDS + _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1
    fake_o_desc = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (_odesc_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    # THD: the caller's Q/KV length tensors, consumed by the setup kernel's
    # device-side metadata build. DYNAMIC extents — (B,) per-batch lengths and
    # (B+1,) cu prefix sums bind the same artifact; the form rides the runtime
    # thd_lens_form bitmask, so no compile key grows (Rule 4). align 4: bound
    # directly, only natural int32 alignment is guaranteed.
    if CFG.THD_VARLEN:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
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
        # THD: the packed totals are runtime values carried by the (dynamic)
        # tensor extents — _host reads them from the views' shapes.
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
