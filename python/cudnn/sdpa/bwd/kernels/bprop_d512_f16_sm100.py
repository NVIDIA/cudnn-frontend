# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SDPA backward stage 2, d_qk = d_v = 512, bf16/fp16, SM100 (Blackwell).

Produces the ``S`` and ``dS`` workspaces that stage 3's three GEMMs reduce into
dV / dK / dQ.  Computes no gradient itself::

    S  = exp2(attn_scale_log2e * S_acc  - lse * log2e)      then masked to 0.0
    dS = (attn_scale_for_dS * dS_acc - do_dot * attn_scale_in) * S

**cga4x1 role split** (CGA_M=4, CTA_MMA=2 -> two sub-groups of two CTAs):

    sg0 = {CTA 0,1}   BMM1  mma_ts(Q from TMEM, K from SMEM) -> S_acc
                      softmax -> S ; bf16 S -> workspace ; fp32 S -> sg1
    sg1 = {CTA 2,3}   BMM2  mma_ts(dO from TMEM, V from SMEM) -> dS_acc
                      dS epilogue ; bf16 dS -> workspace

The split is what makes this fit: each sub-group gets its own 227 KiB of SMEM
and its own 512 TMEM columns, so the operand is TMEM-resident on BOTH sides
(Q on sg0, dO on sg1) and both accumulators are double-buffered.  The Rubin
SM107 sibling holds Q and dO on one CTA and pays 320 KiB for it, which does not
fit here.

**There is no online softmax.**  LSE arrives as a kernel input, so there is no
running max, no rescale, no alpha/stats ship, no correction warp.

**The mask is applied to S AFTER exp2, with 0.0 (not -inf before).**  dS is a
product with S, so zeroing S zeroes dS for free -- one mask application covers
both workspaces, and it survives the role split unchanged.  Do not move it.

**Every mbarrier init count is a named ``CFG`` constant, and the comment beside
it names the ARRIVE SITES it counts and the guard each one fires under.**  Keep
that pairing when changing either side: the counts and the elect/predicate
guards are one decision, and a mismatch here is undefined behaviour rather than
a crash -- it surfaces as an intermittent hang whose output is correct on every
launch that completes.
"""

from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

import cuda.bindings.driver as _cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass._mlir.dialects import arith
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    wait,
    PipelineState,
    Producer,
    Scope,
    advance,
    cga_arrive,
    cga_wait,
)
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_NONE, MASK_PADDED, apply_mask_chunk, compute_kv_loop_bounds
from cudnn.frost.tile_dsl.mma import mma_ts
from cudnn.frost.tile_dsl.scheduler import Sched, read_tile_id_arrive, scheduler_warp_loop
from cudnn.frost.tile_dsl.tma import (
    cp_async_bulk_shared_cluster_shared_cta,
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
)
from cudnn.frost.tile_dsl.pointwise import tmem_load_tile
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.sdpa.bwd.kernels._common_sm100 import make_bwd_decode
from cudnn.sdpa.bwd.config_sm100 import (
    TemplateParams,
    cast_bytes,
    make_cfg_d512,
    operand_bytes,
    s_tma_iters,
    smem_bytes,
    tmem_cols,
    xfer_bytes,
)

# Injected by the loader before this body executes; a plain import gets the
# all-defaults config (dense bf16), which keeps `python bprop_d512_f16_sm100.py`
# usable as a standalone benchmark.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG = make_cfg_d512(PARAMS)
_decode_initial, _decode_payload = make_bwd_decode(CFG)

# Does any tile in this specialization need the per-cell mask at all?  Folds the
# whole mask apparatus out of the dense build.
_MASKED = CFG.MASK_FLAGS != MASK_NONE
# Padding covers two things at once: a per-batch seq_len, and a sequence
# length that is not a multiple of the tile (the engine rounds the COMPILE
# shape up and passes the real length, so the tail tile is computed then
# masked). Both reduce to "mask everything past seqlen".
_PADDED = bool(CFG.MASK_FLAGS & MASK_PADDED)


@cute.jit
def _kv_tile_bounds(q_block, cta_in_pair, seqlen_q, seqlen_kv, n_kv):
    """The kv TILE range this cluster visits, plus where the mask starts biting.

    Two things make this different from the forward's use of the same helper:

    * **Keyed on the CLUSTER's q span, not the CTA's.**  ``cta_in_pair`` 0 and 1
      hold adjacent q row-blocks, so a per-CTA causal bound would give them
      different iteration counts -- and every barrier here is a cross-CTA
      protocol whose count must match on all four CTAs.  Taking the bound over
      the whole ``CTA_MMA * TILE_M`` span makes all four agree.  The CTA holding
      the earlier rows then computes a few fully-masked tiles; the per-cell mask
      zeroes them, so it costs work, not correctness.
    * **Dense folds to the whole range** at trace time, so the dense build is
      byte-identical to before masks existed.

    Returns ``(left, unmasked_lo, unmasked_hi, right)``.  Only
    ``[unmasked_lo, unmasked_hi)`` is guaranteed free of masked cells; the tiles
    on EITHER side of it are band edges and must run ``apply_mask_chunk``.
    """
    if cutlass.const_expr(not _MASKED):
        return cutlass.Int32(0), cutlass.Int32(0), n_kv, n_kv
    cluster_q_row = (q_block - cta_in_pair) * cutlass.Int32(CFG.TILE_M)
    b = compute_kv_loop_bounds(
        cluster_q_row,
        seqlen_q,
        seqlen_kv,
        CFG.WINDOW_LEFT,
        CFG.MASK_FLAGS,
        CFG.TILE_N,
        CFG.TILE_M * CFG.CTA_MMA,
        bottom_right=CFG.BOTTOM_RIGHT,
        window_right=CFG.WINDOW_RIGHT,
    )
    return b.left, b.unmasked_lo, b.unmasked_hi, b.right


def _residual_depth(n_iters, stages: int):
    """Unconsumed producer arrives on a pre-armed ring at end of tile.

    A consumer started with ``PipelineState.start(phase=1)`` gets its first
    ``stages`` waits for free, so over ``n_iters`` iterations it consumes only
    ``max(0, n_iters - stages)`` of the ``n_iters`` arrives the producer made.
    The residual is therefore ``min(n_iters, stages)`` -- NOT ``stages``.

    Draining a fixed ``stages`` deep hangs whenever ``n_iters < stages``: the
    extra waits sit on ring slots that were never armed for this tile.
    """
    return cutlass.Int32(arith.select((n_iters < cutlass.Int32(stages)).ir_value(), n_iters.ir_value(), cutlass.Int32(stages).ir_value()))


def _require(cond, msg):
    """Geometry sanity check; raises instead of assert (asserts vanish under -O)."""
    if not cond:
        raise ValueError(f"bprop_d512_f16_sm100: {msg}")


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
else:  # pragma: no cover - make_cfg_d512 rejects everything else
    raise ValueError(f"bprop_d512_f16_sm100: DTYPE_QKV={CFG.DTYPE_QKV} not supported (expected 2=BF16 or 3=FP16)")
# The workspace dtype IS the io dtype: stage 3 reads S/dS at input precision.
# An earlier Rubin revision tied it to an independent output dtype and produced
# a silent mismatch against the stage-3 GEMMs.
WORKSPACE_DTYPE = STORAGE_DTYPE
MMA_KIND = nvvm.Tcgen05MMAKind.F16

CGA_SIZE = CFG.CGA_M * CFG.CGA_N
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1
LOG2E = 1.4426950408889634

# ---------------------------------------------------------------------------
# Buffer geometry -- all of it derived from the config's own functions so the
# validator and the kernel can never disagree.
# ---------------------------------------------------------------------------

qBytes, doBytes, kBytesPerStage, vBytesPerStage = operand_bytes(CFG)
qBufferElems = qBytes // CFG.BPE
doBufferElems = doBytes // CFG.BPE
kBufferElems = kBytesPerStage // CFG.BPE
vBufferElems = vBytesPerStage // CFG.BPE

# One max-union alias slab per sub-group: the Q (resp. dO) staging buffer is
# dead once the UTCCP has moved it to TMEM, so the K (resp. V) ring reuses those
# exact bytes behind the alias-seam barrier.  Sized in ELEMENTS of a single
# dtype -- legal only because every tenant here is STORAGE_DTYPE.  If a future
# tenant has a different BPE, size this in BYTES and view it as the wider dtype
# (frost-tile-dsl.md §6; the fp8 forward sibling had exactly that bug).
_ALIAS_ELEMS = max(qBufferElems, doBufferElems, CFG.STAGES_KV * kBufferElems, CFG.STAGES_KV * vBufferElems)

# fp32 S ship, sg0 -> sg1.  fp32 and not the io dtype on purpose: the Rubin
# reference never transfers S at all (it multiplies the fp32 register copy
# straight into dS), so shipping a rounded S across the role split would be an
# accuracy regression the reference does not have.
S_TMA_ITERS = s_tma_iters(CFG)
S_D_BLOCK = CFG.TILE_N // S_TMA_ITERS
S_SUBTILE_SLAB = CFG.TILE_M * S_D_BLOCK
sXferHalfElems = CFG.TILE_M * S_D_BLOCK
sXferHalfBytes = sXferHalfElems * 4
sXferElems = CFG.XFER_HALVES * sXferHalfElems
castElems = CFG.TILE_M * CFG.TILE_N

_require(S_TMA_ITERS == CFG.XFER_HALVES, f"one shipped fp32 half must be exactly one stored subtile ({S_TMA_ITERS} vs {CFG.XFER_HALVES})")
_require(sXferElems * 4 == xfer_bytes(CFG), "xfer element count disagrees with config xfer_bytes()")
_require(castElems * CFG.BPE == cast_bytes(CFG), "cast element count disagrees with config cast_bytes()")

# cga2: a cta_group::2 tensor TMA routes ALL bytes to the leader's mbar, so the
# transaction size counts BOTH peers' buffers.
qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
doTmaTransactionBytes = doBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

# TMA subtiling of the d=512 operands: inner byte extent / swizzle atom.
TMA_QK_ITERS = (CFG.TILE_K * CFG.BPE) // CFG.Q_SWZ_BYTES
TMA_QK_GRANU_ELEMS = CFG.TILE_K // TMA_QK_ITERS
TMA_VO_ITERS = (CFG.TILE_O * CFG.BPE) // CFG.V_SWZ_BYTES
TMA_VO_GRANU_ELEMS = CFG.TILE_O // TMA_VO_ITERS

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_B = {128: 3, 64: 2, 32: 1}
SMEM_LAYOUT_QK = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_S = _SWZ_ENUM[CFG.S_SWZ_BYTES]
S_SMEM_SWIZZLE = cutlass.Swizzle(_SWZ_B[CFG.S_SWZ_BYTES], 4, 3)

_CORE_MATRIX_ROWS = 8
LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = _CORE_MATRIX_ROWS * CFG.Q_SWZ_BYTES
LEADING_BYTE_OFFSET_V = 0
STRIDE_BYTE_OFFSET_V = _CORE_MATRIX_ROWS * CFG.V_SWZ_BYTES

# Softmax register budget: each compute warp lane owns one q-row, walked in
# chunks of S_D_BLOCK columns.  One chunk == one shipped half == one stored
# subtile == one swizzle atom per row.
SOFTMAX_N_CHUNKS = CFG.TILE_N // S_D_BLOCK
_require(SOFTMAX_N_CHUNKS == S_TMA_ITERS, "softmax chunk count must equal the store subtile count")

# ---------------------------------------------------------------------------
# TMEM
# ---------------------------------------------------------------------------


class KernelTmemLayout(NamedTuple):
    """Per-sub-group TMEM carve.  Both sub-groups use the SAME shape with
    different tenants: sg0 = [S_acc p0 | S_acc p1 | Q], sg1 = [dS_acc p0 |
    dS_acc p1 | dO].  An accumulator is fp32 and TILE_N wide -> TILE_N columns;
    a 16-bit operand packs 2 elements per 32-bit column -> d * BPE / 4 = 256."""

    TOTAL_COLS: int = 512
    ACC_COLS: int = CFG.TILE_N
    ACC_PARITY0_OFF: int = 0
    ACC_PARITY1_OFF: int = CFG.TILE_N
    OPERAND_OFF: int = CFG.STAGES_ACC * CFG.TILE_N
    OPERAND_COLS: int = (CFG.TILE_K * CFG.BPE) // 4


LAYOUT = KernelTmemLayout()

_sg0_tmem, _sg1_tmem = tmem_cols(CFG)
_require(LAYOUT.OPERAND_OFF == CFG.STAGES_ACC * LAYOUT.ACC_COLS, "accumulator parities must abut the operand")
_require(LAYOUT.OPERAND_OFF + LAYOUT.OPERAND_COLS == LAYOUT.TOTAL_COLS, f"TMEM carve must be exactly {LAYOUT.TOTAL_COLS} columns")
_require(_sg0_tmem == LAYOUT.TOTAL_COLS and _sg1_tmem == LAYOUT.TOTAL_COLS, "kernel TMEM carve disagrees with config tmem_cols()")
_require(CFG.TILE_K == CFG.TILE_O, "d_qk must equal d_v for the symmetric sg0/sg1 operand carve")

# UTCCP: whole operand SMEM -> TMEM, turning both BMMs into mma_ts.  SHAPE_128X128B
# moves 128 rows x 128 bits = 16 B/row = 4 fp32 TMEM columns per call.  Descriptor
# arithmetic is in 16-BYTE units and SUBTILE-MAJOR, matching the TMA-laid SMEM.
_UTCCP_BYTES_PER_CALL = 16
_UTCCP_TMEM_COLS_PER_CALL = _UTCCP_BYTES_PER_CALL // 4
_UTCCP_PER_SUBTILE = CFG.Q_SWZ_BYTES // _UTCCP_BYTES_PER_CALL
_UTCCP_SUBTILE_DESC_STRIDE = (CFG.TILE_M * CFG.Q_SWZ_BYTES) // _UTCCP_BYTES_PER_CALL
_UTCCP_N_CALLS = LAYOUT.OPERAND_COLS // _UTCCP_TMEM_COLS_PER_CALL
_require(_UTCCP_N_CALLS == TMA_QK_ITERS * _UTCCP_PER_SUBTILE, f"UTCCP call count {_UTCCP_N_CALLS} != subtiles x per-subtile")

# ---------------------------------------------------------------------------
# SMEM budget -- assert the COMPUTED total, never a inherited constant.
# ---------------------------------------------------------------------------

_SG0_SMEM_BYTES, _SG1_SMEM_BYTES = smem_bytes(CFG)
_require(_ALIAS_ELEMS * CFG.BPE + xfer_bytes(CFG) + cast_bytes(CFG) == _SG0_SMEM_BYTES, "kernel alias sizing disagrees with config smem_bytes()")
_require(_SG0_SMEM_BYTES == _SG1_SMEM_BYTES, "the role split is symmetric; sg0 and sg1 must budget identically")

# ---------------------------------------------------------------------------
# Barriers.  Each init_count is a named CFG constant; the comment beside it
# names the arrive sites it counts and the guard each fires under.
# ---------------------------------------------------------------------------


class Bars(NamedTuple):
    # Operand staging + the alias seam (Q on sg0, dO on sg1).
    mb_tma_op_full: object
    mb_tma_op_empty: object
    mb_op_utccp_done: object
    # The streamed ring (K on sg0, V on sg1).
    mb_tma_ring_full: object
    mb_tma_ring_empty: object
    # MMA -> compute, and the accumulator release back to the MMA leader.
    mb_bmm_done: object
    mb_acc_empty: object
    # compute -> TMA-STG for the workspace store (S on sg0, dS on sg1).
    mb_smem_full: object
    mb_smem_empty: object
    # The cross-sub-group fp32 S ship.
    mb_s_xfer_full: object
    mb_s_xfer_empty: object
    mb_tmem_dealloc: object


def _make_bwd_d512_sm100_bars(CFG) -> Bars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        mb_tma_op_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_op_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_op_utccp_done=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_tma_ring_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_ring_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm_done=MBarrier(_alloc(CFG.STAGES_ACC), stages=CFG.STAGES_ACC, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        # Every lane of the compute WG on BOTH CTAs of the pair arrives on the
        # pair leader, after its tcgen05_wait(LOAD).
        mb_acc_empty=MBarrier(_alloc(CFG.STAGES_ACC), stages=CFG.STAGES_ACC, init_count=CFG.ACC_EMPTY_ARRIVERS, producer=Producer.LEADER, scope=Scope.LEADER),
        # THREAD arrive from all four compute warps -> COMPUTE_LANES.
        mb_smem_full=MBarrier(_alloc(S_TMA_ITERS), stages=S_TMA_ITERS, init_count=CFG.COMPUTE_LANES, producer=Producer.THREAD),
        # THREAD arrive from the single TMA-STG warp, NOT elect-gated -> all 32
        # lanes fire.  ONE_LANE here would under-count by 31 and hang.
        mb_smem_empty=MBarrier(_alloc(S_TMA_ITERS), stages=S_TMA_ITERS, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        # DSMEM bulk-copy bytes stay LOCAL to the receiving CTA, and S is a
        # pointwise multiplier rather than a leader-only MMA operand, so there is
        # no cross-CTA forwarding and no runtime per-CTA init: each sg1 CTA arms
        # its own expect_tx from its own lead warp's single lane.
        mb_s_xfer_full=MBarrier(_alloc(CFG.XFER_HALVES), stages=CFG.XFER_HALVES, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        # One lane arrives on the cross-sg peer, behind the named barrier that
        # proves all four compute warps have read S into registers.
        mb_s_xfer_empty=MBarrier(_alloc(CFG.XFER_HALVES), stages=CFG.XFER_HALVES, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
    )


# ---------------------------------------------------------------------------
# Warp bodies.  Both sub-groups run the SAME shape with different tensors --
# sg0 drives (Q, K) -> S_acc -> S, sg1 drives (dO, V) -> dS_acc -> dS -- so each
# body is one `if is_sg0:` selecting descriptors/buffers over shared logic.
#
# K and V have IDENTICAL TMA geometry here ([TILE_N/CTA_MMA, 512] at
# kv_row_base + cta_in_pair*(TILE_N/CTA_MMA)), and so do Q and dO.  That is a
# consequence of both BMMs being A[q,d] x B[kv,d]^T; the forward's V splits on
# the O column axis instead because its BMM2 is P.V.  Do not copy that offset.
# ---------------------------------------------------------------------------


@cute.jit
def _tmaldg_warp_group(
    bars,
    sched,
    sOperand,
    sRing,
    tma_q,
    tma_k,
    tma_v,
    tma_do,
    cta_in_pair,
    is_leader,
    is_sg0,
    tma_mcast_mask,
    n_kv,
    seqlen_q,
    seqlen_kv,
    gqa_ratio,
    head_base,
    batch_base,
):
    q_block, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    # Pre-armed: nothing has consumed the operand or the ring yet, so the first
    # wait must fall through with no producer arrive.
    op_empty_state = PipelineState.start(phase=1)
    op_utccp_state = PipelineState.start()
    ring_state = PipelineState.start(phase=1)
    # Seeded from the initial tile so the end-of-kernel drain can read them:
    # a value FIRST assigned inside the persistent loop does not survive it.
    kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_in_pair, seqlen_q, seqlen_kv, n_kv)

    # Per-CTA slice of the streamed operand along the collective MMA-N axis.
    RING_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row_base = cute.arch.make_warp_uniform(q_block * cutlass.Int32(CFG.TILE_M))
        head_g = head_idx + head_base
        batch_g = batch_idx + batch_base
        # GQA / MQA: Q and dO are per Q-head; K and V live at the shared KV head.
        # gqa_ratio is 1 for MHA, so this folds to head_g and the dense path is
        # unchanged.  It is a runtime divide, but once per TILE, not per row.
        kv_head_g = head_g // gqa_ratio

        # --- the resident operand: Q on sg0, dO on sg1 --------------------
        bars.mb_tma_op_empty.wait(op_empty_state.phase)
        op_empty_state = advance(op_empty_state, 1)
        bars.mb_tma_op_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        if is_sg0:
            tma_load_tile(
                sOperand[0],
                tma_q(cutlass.Int32(0), head_g, q_row_base, batch_g),
                bars.mb_tma_op_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
        else:
            tma_load_tile(
                sOperand[0],
                tma_do(cutlass.Int32(0), head_g, q_row_base, batch_g),
                bars.mb_tma_op_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

        # --- THE ALIAS SEAM ----------------------------------------------
        # The ring shares the operand's bytes.  Loading it before the UTCCP has
        # drained clobbers the operand mid-copy.  This wait is not an
        # optimization and must not be hoisted or removed.
        bars.mb_op_utccp_done.wait(op_utccp_state.phase)
        op_utccp_state = advance(op_utccp_state, 1)

        # --- the streamed ring: K on sg0, V on sg1 ------------------------
        # Same bounds on all four CTAs (see _kv_tile_bounds): the tiles skipped
        # here are never loaded, never MMA'd and never stored.
        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_in_pair, seqlen_q, seqlen_kv, n_kv)
        for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
            kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
            bars.mb_tma_ring_empty[ring_state.idx].wait(ring_state.phase)
            bars.mb_tma_ring_full[ring_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            if is_sg0:
                tma_load_tile(
                    sRing[ring_state.idx],
                    tma_k(cutlass.Int32(0), kv_head_g, kv_row_base + RING_ROW_OFFSET_PEER, batch_g),
                    bars.mb_tma_ring_full[ring_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
            else:
                tma_load_tile(
                    sRing[ring_state.idx],
                    tma_v(cutlass.Int32(0), kv_head_g, kv_row_base + RING_ROW_OFFSET_PEER, batch_g),
                    bars.mb_tma_ring_full[ring_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
            ring_state = advance(ring_state, CFG.STAGES_KV)

        # --- next tile ----------------------------------------------------
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load())
        nxt_hb = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load())
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load())
        q_block, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    # --- cross-CTA drains, OUTSIDE the persistent loop --------------------
    # Both rings are fired by a cross-CTA MMA_COMMIT, so the last iteration's
    # commit is still in flight when this warp exits.  Staying resident until it
    # lands is what stops the teardown fault.
    if cutlass.const_expr(CFG.CTA_MMA > 1):
        _ring_residual = _residual_depth(kv_right - kv_left, CFG.STAGES_KV)
        for _ in cutlass.range(cutlass.Int32(0), _ring_residual, 1, unroll=1):
            bars.mb_tma_ring_empty[ring_state.idx].wait(ring_state.phase)
            ring_state = advance(ring_state, CFG.STAGES_KV)
        # One operand load per tile, so exactly one residual arrive.
        bars.mb_tma_op_empty.wait(op_empty_state.phase)
        op_empty_state = advance(op_empty_state, 1)


@cute.jit
def _mma_warp_leader(bars, sched, sOperand, sRing, tmem_ptr_i32, cta_in_pair, is_sg0, mcast_mask, n_kv, seqlen_q, seqlen_kv):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    # Publish the TMEM base to this CTA's compute WG.  Without it their base
    # load races the alloc on a cold cache -> garbage base, misaligned address.
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_operand = tmem_raw.subview(cutlass.Int32(LAYOUT.OPERAND_OFF))

    # Both BMMs are the same MMA: A[TILE_M, d] from TMEM x B[TILE_N/CTA_MMA, d]^T
    # from SMEM, K = d.  No k_dim at bf16/fp16 (default 0, TILE_K_HW = 16).
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
    )
    bmm_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc,
        kind=MMA_KIND,
    )

    q_block, _head, _batch = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    op_full_state = PipelineState.start()
    ring_state = PipelineState.start()
    acc_state = PipelineState.start(phase=1)
    # Seeded from the initial tile so the end-of-kernel drain can read them:
    # a value FIRST assigned inside the persistent loop does not survive it.
    kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_in_pair, seqlen_q, seqlen_kv, n_kv)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_tma_op_full.wait(op_full_state.phase)
        op_full_state = advance(op_full_state, 1)

        # --- UTCCP: whole operand SMEM -> TMEM, turning the BMM into mma_ts.
        # Leader-only: tcgen05.cp.cta_group::2 is a SELF-FILL broadcast, so one
        # issue makes each SM copy its OWN SMEM into its own TMEM.  A
        # "defensive" non-leader copy would re-copy the same data.
        # Loop AND commit share one elect block (frost-tile-dsl.md §8).
        desc_operand = sOperand[0].desc()
        if nvvm.elect_sync():
            for _qk in cutlass.range(0, _UTCCP_N_CALLS, 1, unroll_full=True):
                _s = _qk // _UTCCP_PER_SUBTILE
                _kk = _qk % _UTCCP_PER_SUBTILE
                _desc_off = _s * _UTCCP_SUBTILE_DESC_STRIDE + _kk
                nvvm.tcgen05_cp(
                    nvvm.Tcgen05CpShape.SHAPE_128X128B,
                    tmem_operand.subview(_qk * _UTCCP_TMEM_COLS_PER_CALL),
                    desc_operand + _desc_off,
                    group=CTA_GROUP_KIND,
                )
            # The alias seam.  MMA_COMMIT, never arrive_on_peer: the commit
            # fires after the tcgen05 pipeline drains, so TMA-LDG cannot
            # overwrite the operand while the UTCCP still reads it.
            bars.mb_op_utccp_done.arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask)

        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_in_pair, seqlen_q, seqlen_kv, n_kv)
        for _kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
            bars.mb_tma_ring_full[ring_state.idx].wait(ring_state.phase)
            bars.mb_acc_empty[acc_state.idx].wait(acc_state.phase)
            desc_ring = sRing[ring_state.idx].desc()
            # accumulate=False is correct: K = TILE_K reduces entirely INSIDE
            # this one call (num_k_steps = TILE_K // TILE_K_HW), so there is no
            # outer k-loop to accumulate across.
            mma_ts(
                bmm_desc,
                tmem_operand,
                desc_ring,
                tmem_raw.subview(cutlass.Int32(LAYOUT.ACC_PARITY0_OFF) + acc_state.idx * cutlass.Int32(LAYOUT.ACC_COLS)),
                accumulate=False,
            )
            # ONE lane commits.  commit_mma() does NOT elect internally
            # (barrier.py:112 -> nvvm.tcgen05_commit), so an un-predicated
            # arrive issues 32 commits against init_count=1: the phase flips 32
            # times and a consumer polling try_wait.parity either catches an odd
            # flip and runs on, or misses the window and spins forever.  That is
            # the intermittent hang, not a style nit.  Every MMA_COMMIT arrive in
            # the shipped forward is pred= or elect-wrapped.
            elect_p = nvvm.elect_sync()
            bars.mb_bmm_done[acc_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
            bars.mb_tma_ring_empty[ring_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
            ring_state = advance(ring_state, CFG.STAGES_KV)
            acc_state = advance(acc_state, CFG.STAGES_ACC)

        bars.mb_tma_op_empty.arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=nvvm.elect_sync())

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        # The q block is read on this arm too: the causal kv bound depends on it.
        nxt_q = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load()
        nxt_hb = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load()
        nxt_v = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load()
        q_block, _head, _batch = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    if cutlass.const_expr(CFG.CTA_MMA > 1):
        _acc_residual = _residual_depth(kv_right - kv_left, CFG.STAGES_ACC)
        for _ in cutlass.range(cutlass.Int32(0), _acc_residual, 1, unroll=1):
            bars.mb_acc_empty[acc_state.idx].wait(acc_state.phase)
            acc_state = advance(acc_state, CFG.STAGES_ACC)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_non_leader(bars, sched, tmem_ptr_i32, cta_in_pair):
    """Quiet -- but NOT an empty body.

    It still owns this CTA's TMEM allocation and the base publish that releases
    its own compute warp group; deleting either hangs that group and leaks TMEM.
    It issues no MMA and no UTCCP (the leader's cta_group::2 UTCCP is a
    self-fill broadcast that already filled this CTA's TMEM).
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    _q_block, _head, _batch = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _tmastg_warp_group(bars, sched, sCast, tma_s, tma_ds, cta_in_pair, is_sg0, n_kv, seqlen_q, seqlen_kv, head_base, batch_base):
    q_block, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    smem_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row_base = q_block * cutlass.Int32(CFG.TILE_M)
        # The workspace is CHUNK-LOCAL: head_base / batch_base offset every
        # full-tensor access but NOT this store.
        head_ws = head_idx
        batch_ws = batch_idx

        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_in_pair, seqlen_q, seqlen_kv, n_kv)
        for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
            # kv is the ABSOLUTE tile index, so this is the right workspace
            # column even once masks make kv_left > 0.  In ELEMENTS: the fp8
            # "byte == elem" coincidence does not hold at BPE = 2.
            kv_col = kv_loop * cutlass.Int32(CFG.TILE_N)
            # ONE store per kv tile, NOT one per chunk.  tma_store_tile walks
            # all S_TMA_ITERS subtiles itself (num_iters = tma_loads_per_tile,
            # coord advancing by i*granu_elems), so a per-chunk call stores the
            # whole tile twice: the second call re-lays subtile 0 over columns
            # [S_D_BLOCK, TILE_N) and pushes subtile 1 past the tile.
            for chunk in cutlass.range_constexpr(S_TMA_ITERS):
                bars.mb_smem_full[chunk].wait(smem_state.phase)
            if is_sg0:
                tma_store_tile(sCast[0], tma_s(kv_col, q_row_base, head_ws, batch_ws))
            else:
                tma_store_tile(sCast[0], tma_ds(kv_col, q_row_base, head_ws, batch_ws))
            tma_store_commit()
            tma_store_wait(0)
            # Plain THREAD arrives: all 32 lanes fire, which is what
            # init_count = ONE_WARP counts.
            for chunk in cutlass.range_constexpr(S_TMA_ITERS):
                bars.mb_smem_empty[chunk].arrive()
            smem_state = advance(smem_state, 1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load()
        nxt_hb = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load()
        nxt_v = sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load()
        q_block, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _compute_kv_iter(
    bars,
    sXfer_raw,
    sCast_raw,
    tmem_base_addr,
    tid_in_wg,
    q_row,
    is_lead_warp,
    is_sg0,
    leader_cta_id,
    cross_sg_peer,
    lse_q_log2e,
    scaled_do_dot_q,
    attn_scale_log2e,
    attn_scale_for_dS,
    seqlen_q,
    seqlen_kv,
    row_scale,
    kv_loop,
    acc_state,
    smem_state,
    xfer_empty_state,
    xfer_full_state,
    apply_mask: cutlass.Constexpr[bool],
):
    """One kv tile of the compute warp group: TMEM -> S (sg0) / dS (sg1) -> SMEM.

    ``apply_mask`` is a PYTHON constant, so this traces twice -- once without any
    mask code for the tiles strictly below the diagonal band, once with it for
    the diagonal tiles.  That is why it is a module-level function taking every
    value explicitly rather than a closure: the DSL rejects closure capture
    inside a staged loop (SCOPE_CLOSURE_CAPTURE), and a runtime ``if`` would not
    fold the mask away.

    Pipeline states go in and come back out; the caller rebinds them.
    """
    bars.mb_bmm_done[acc_state.idx].wait(acc_state.phase)
    acc_col_base = cutlass.Int32(LAYOUT.ACC_PARITY0_OFF) + acc_state.idx * cutlass.Int32(LAYOUT.ACC_COLS)

    for chunk in cutlass.range_constexpr(S_TMA_ITERS):
        reg_acc = tmem_load_tile(tmem_base_addr + acc_col_base + cutlass.Int32(chunk * S_D_BLOCK), num_elems=S_D_BLOCK)
        # Commit the TMEM read BEFORE releasing the accumulator, or the
        # leader's next MMA overwrites it mid-flight.  Needed on BOTH
        # sub-group arms, masked or not.
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        if cutlass.const_expr(chunk == S_TMA_ITERS - 1):
            bars.mb_acc_empty[acc_state.idx].arrive(cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id)

        # Subtile-major SMEM offsets: the whole of subtile 0 (all
        # TILE_M rows) precedes any of subtile 1, which is what TMA
        # walks with tma_subtile_stride_elems.
        cast_off = cutlass.Int32(chunk * S_SUBTILE_SLAB) + tid_in_wg * cutlass.Int32(S_D_BLOCK)
        xfer_off = cutlass.Int32(chunk * sXferHalfElems) + tid_in_wg * cutlass.Int32(S_D_BLOCK)

        if is_sg0:
            # S = exp2(attn_scale_log2e * S_acc - lse * log2e), fp32.
            s_post = cute.math.exp2(reg_acc.vec * attn_scale_log2e - lse_q_log2e, fastmath=True)

            # Mask AFTER exp2, to 0.0 -- not to -inf before it.  With
            # LSE supplied there is no running max to protect, and dS is
            # a PRODUCT with S, so zeroing S zeroes dS for free: one
            # application covers both workspaces and survives the role
            # split (sg1 receives the already-masked S).  `q_row` is this
            # CTA's own row; only the tile bound is cluster-wide.
            if cutlass.const_expr(apply_mask):
                s_post = apply_mask_chunk(
                    s_post,
                    q_row,
                    kv_loop * cutlass.Int32(CFG.TILE_N) + cutlass.Int32(chunk * S_D_BLOCK),
                    seqlen_kv,
                    CFG.WINDOW_LEFT,
                    CFG.MASK_FLAGS,
                    N=S_D_BLOCK,
                    bottom_right=1 if CFG.BOTTOM_RIGHT else 0,
                    causal_diag=(seqlen_kv - seqlen_q) if CFG.BOTTOM_RIGHT else None,
                    mask_value=0.0,
                    window_right=CFG.WINDOW_RIGHT,
                )
            # Rows past the real S_q: zero the whole row. Their LSE came from a
            # CLAMPED index and is meaningless, so the value must be discarded
            # rather than trusted. This is OUTSIDE the apply_mask branch on
            # purpose -- a padded q row is invalid on EVERY kv tile, including
            # the interior ones that carry no mask code.
            if cutlass.const_expr(_PADDED):
                s_post = s_post * row_scale

            # The fp32 ship, sg0 -> sg1.  This buffer is UNSWIZZLED and
            # linear on purpose: unlike the forward's P it is never an
            # MMA operand and never TMA'd, so a swizzled read path would
            # buy nothing and cost a matching load.
            bars.mb_s_xfer_empty[chunk].wait(xfer_empty_state.phase)
            for _i in cutlass.range_constexpr(S_D_BLOCK):
                sXfer_raw.subview(xfer_off + cutlass.Int32(_i)).store(s_post[_i])

            # The workspace store staging.  Deferred wait: the exp2 above
            # overlaps the previous TMA-STG drain.
            bars.mb_smem_empty[chunk].wait(smem_state.phase)
            sCast_raw.subview(cast_off).data_ptr().store_swizzled(s_post.to(WORKSPACE_DTYPE), alignment=64, swizzle=S_SMEM_SWIZZLE)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_smem_full[chunk].arrive()

            # The named barrier is KEPT -- the ship's operand is all
            # four warps' stores.  mapa BOTH the pointer and the mbar.
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=CFG.COMPUTE_LANES)
            ship_pred = is_lead_warp & nvvm.elect_sync()
            local_src = sXfer_raw.subview(cutlass.Int32(chunk * sXferHalfElems))
            peer_dst = nvvm.mapa(local_src, cross_sg_peer, addrspace=7)
            peer_mbar = nvvm.mapa(bars.mb_s_xfer_full[chunk].smem_ptr, cross_sg_peer, addrspace=7)
            cp_async_bulk_shared_cluster_shared_cta(peer_dst, local_src, peer_mbar, sXferHalfBytes, pred=ship_pred)
        else:
            # Arm expect_tx from ONE lane of the LEAD warp: this is a
            # 4-warp group, so a bare elect_sync() would give one arrive
            # per warp -- 4 against init=1.
            bars.mb_s_xfer_full[chunk].arrive(n_bytes=sXferHalfBytes, pred=is_lead_warp & nvvm.elect_sync())
            bars.mb_s_xfer_full[chunk].wait(xfer_full_state.phase)

            # dS = (attn_scale_for_dS * dS_acc - do_dot*attn_scale) * S.
            # S is read at fp32 -- rounding it to the io dtype here is
            # exactly the accuracy the Rubin reference does not lose.
            ds_elems = tuple(
                (reg_acc[_i] * attn_scale_for_dS - scaled_do_dot_q) * sXfer_raw.subview(xfer_off + cutlass.Int32(_i)).load() for _i in range(S_D_BLOCK)
            )
            ds_post = cutlass.Vector.from_elements(ds_elems, cutlass.Float32)

            bars.mb_smem_empty[chunk].wait(smem_state.phase)
            sCast_raw.subview(cast_off).data_ptr().store_swizzled(ds_post.to(WORKSPACE_DTYPE), alignment=64, swizzle=S_SMEM_SWIZZLE)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_smem_full[chunk].arrive()

            # Release the ship slot only once all four warps have read
            # S out of it, then one lane arrives on the peer.
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=CFG.COMPUTE_LANES)
            if is_lead_warp:
                if nvvm.elect_sync():
                    bars.mb_s_xfer_empty[chunk].arrive_on_peer(cross_sg_peer)

    smem_state = advance(smem_state, 1)
    xfer_empty_state = advance(xfer_empty_state, 1)
    xfer_full_state = advance(xfer_full_state, 1)
    acc_state = advance(acc_state, CFG.STAGES_ACC)
    return acc_state, smem_state, xfer_empty_state, xfer_full_state


@cute.jit
def _compute_warp_group(
    bars,
    sched,
    sXfer_raw,
    sCast_raw,
    tmem_ptr_i32,
    lse_tensor,
    do_dot_tensor,
    cta_in_pair,
    is_sg0,
    leader_cta_id,
    cross_sg_peer,
    cta_id_x,
    n_kv,
    seqlen_q,
    seqlen_kv,
    attn_scale_in,
    attn_scale_log2e,
    attn_scale_for_dS,
    head_base,
    batch_base,
):
    """sg0: S = exp2(scale*S_acc - lse), store bf16 S, ship fp32 S to sg1.
    sg1: dS = (scale*dS_acc - do_dot) * S_recv, store bf16 dS.

    One lane == one q-row (tid_in_wg 0..127 is both the row and the TMEM lane).
    The TILE_N columns are walked in S_TMA_ITERS chunks of S_D_BLOCK, and that
    chunk is simultaneously one shipped fp32 half, one stored TMA subtile and
    one 128 B swizzle atom per row -- the alignment the config asserts.
    """
    # Must precede the TMEM base read: pairs with the MMA warp's
    # barrier_cta_arrive right after tmem_alloc.  Without it the base load
    # races the alloc on a cold cache.
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base_addr = tmem_ptr_i32.load()

    tid_in_wg = cute.arch.thread_idx()[0]
    is_lead_warp = (tid_in_wg // cutlass.Int32(32)) == cutlass.Int32(0)

    q_block, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    acc_state = PipelineState.start()
    smem_state = PipelineState.start(phase=1)
    # Two states, not one: sg0 consumes mb_s_xfer_EMPTY (pre-armed, nothing has
    # freed a slot yet) while sg1 consumes mb_s_xfer_FULL (a real producer
    # arrive).  Sharing one state makes sg1's first wait fall through before the
    # data lands -- silent garbage, not a hang.
    xfer_empty_state = PipelineState.start(phase=1)
    xfer_full_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row = q_block * cutlass.Int32(CFG.TILE_M) + tid_in_wg
        head_g = head_idx + head_base
        batch_g = batch_idx + batch_base

        # Padding, q side. Two separate problems:
        #  * lse / do_dot are [B, H, S_q] at the REAL length, so a lane whose row
        #    sits past it would read OUT OF BOUNDS -- clamp the INDEX;
        #  * that lane's S must come out 0, so carry a 0/1 factor rather than
        #    trusting the clamped row's value. sg1 needs nothing: its dS is a
        #    product with the S it receives, so the zero propagates.
        if cutlass.const_expr(_PADDED):
            q_row_safe = cute.math.min(q_row, seqlen_q - cutlass.Int32(1))
            row_scale = cutlass.Float32(arith.select((q_row < seqlen_q).ir_value(), cutlass.Float32(1.0).ir_value(), cutlass.Float32(0.0).ir_value()))
        else:
            q_row_safe = q_row
            row_scale = cutlass.Float32(1.0)

        # Per-q-row scalars, hoisted once per tile.  Both arrive RAW: the host
        # folds no log2e into the LSE and no attn_scale into the dot.
        lse_q_log2e = lse_tensor[batch_g, head_g, q_row_safe] * cutlass.Float32(LOG2E)
        scaled_do_dot_q = do_dot_tensor[batch_g, head_g, q_row_safe] * attn_scale_in

        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_in_pair, seqlen_q, seqlen_kv, n_kv)

        def _run(lo, hi, apply_mask, acc_state, smem_state, xfer_empty_state, xfer_full_state):
            for _kv_loop in cutlass.range(lo, hi, 1, unroll=1):
                acc_state, smem_state, xfer_empty_state, xfer_full_state = _compute_kv_iter(
                    bars,
                    sXfer_raw,
                    sCast_raw,
                    tmem_base_addr,
                    tid_in_wg,
                    q_row,
                    is_lead_warp,
                    is_sg0,
                    leader_cta_id,
                    cross_sg_peer,
                    lse_q_log2e,
                    scaled_do_dot_q,
                    attn_scale_log2e,
                    attn_scale_for_dS,
                    seqlen_q,
                    seqlen_kv,
                    row_scale,
                    _kv_loop,
                    acc_state,
                    smem_state,
                    xfer_empty_state,
                    xfer_full_state,
                    apply_mask=apply_mask,
                )
            return acc_state, smem_state, xfer_empty_state, xfer_full_state

        # THREE ranges, in kv order: the band's LOW edge, the interior, the HIGH
        # edge.  Only the interior is provably free of masked cells.  Dropping
        # the low edge is a real bug under SWA -- there `unmasked_lo > left` and
        # those tiles carry the band's left boundary.  Under causal-only
        # `unmasked_lo == left`, so the low range is empty and a two-range split
        # looked correct until SWA was switched on.
        #
        # `apply_mask` is a Python constant, so each range traces separately and
        # the interior contains no mask code at all.
        if cutlass.const_expr(_MASKED):
            acc_state, smem_state, xfer_empty_state, xfer_full_state = _run(
                kv_left, kv_unmasked_lo, True, acc_state, smem_state, xfer_empty_state, xfer_full_state
            )
        acc_state, smem_state, xfer_empty_state, xfer_full_state = _run(
            kv_unmasked_lo, kv_unmasked_hi, False, acc_state, smem_state, xfer_empty_state, xfer_full_state
        )
        if cutlass.const_expr(_MASKED):
            acc_state, smem_state, xfer_empty_state, xfer_full_state = _run(
                kv_unmasked_hi, kv_right, True, acc_state, smem_state, xfer_empty_state, xfer_full_state
            )

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0)).load())
        nxt_hb = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1)).load())
        nxt_v = cute.arch.make_warp_uniform(sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2)).load())
        q_block, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    # --- sg0 owns the cross-sub-group drain ------------------------------
    # mb_s_xfer_empty is an ASYNC cross-CTA arrive from sg1.  If sg0 exits
    # before sg1's last arrive lands, that write hits a dead peer ->
    # CUDA_EXCEPTION_17: an intermittent launch failure that PASSES validation
    # whenever it does not crash, because it is a teardown race and not a math
    # bug.  The cross-sub-group ring is the one most easily missed.
    if cutlass.const_expr(CFG.CTA_MMA > 1):
        if is_sg0:
            for _chunk in cutlass.range_constexpr(CFG.XFER_HALVES):
                bars.mb_s_xfer_empty[_chunk].wait(xfer_empty_state.phase)
            xfer_empty_state = advance(xfer_empty_state, 1)

    # Releases this CTA's MMA warp to dealloc TMEM.  ^1 is the PAIR partner,
    # not cross_sg_peer.
    if is_lead_warp:
        if nvvm.elect_sync():
            bars.mb_tmem_dealloc.arrive_on_peer(cta_id_x ^ cutlass.Int32(1))


# ---------------------------------------------------------------------------
# Kernel entry: cluster identity, allocation, barrier init, warp dispatch.
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_s_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_ds_desc: cutlass.GridConstant[tmap.TensorMap],
    # Plain per-lane GMEM reads, one q-row per lane -- NOT TMA, and therefore
    # carrying no barriers at all.  sg0 reads lse, sg1 reads do_dot.  Both
    # arrive RAW: this kernel applies * log2e and * attn_scale_in itself.
    lse_tensor: cute.Tensor,
    do_dot_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    gqa_ratio: cutlass.Int32,
    n_kv: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    attn_scale_in: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    attn_scale_for_dS: cutlass.Float32,
    # Host-loop coordinates.  These offset every FULL-TENSOR access (Q/K/V/dO/
    # lse/do_dot); the chunk-local S/dS workspace stays at origin.
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- cluster identity ------------------------------------------------
    # Five const_expr ternaries, not if/else blocks: a local assigned in only
    # one arm is not reliably visible afterwards (frost-gotchas.md).
    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = cta_id_x & 1
    leader_cta_id = cta_id_x & ~1
    mcast_mask = cutlass.Int32(3) << leader_cta_id  # 0x3 on sg0, 0xC on sg1
    tma_mcast_mask = cutlass.Int16(1) << cta_id_x  # SELF-ONLY; bytes still route to the leader
    is_leader = cta_in_pair == 0
    sg_id = cta_id_x // CFG.CTA_MMA
    is_sg0 = sg_id == 0
    cross_sg_peer = cta_id_x ^ CFG.CTA_MMA  # 0<->2, 1<->3, mirrored lane-to-lane
    is_cga_first_cta = cta_id_x == 0

    # --- SMEM ------------------------------------------------------------
    # One max-union alias slab: Q|K-ring on sg0, dO|V-ring on sg1.  Both
    # sub-groups allocate the SAME shape -- the role split is symmetric, so
    # there is one allocation block, not two.
    sAliased_raw = cutlass.Array(STORAGE_DTYPE, _ALIAS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sOperand_raw = sAliased_raw  # Q on sg0, dO on sg1 (dead after the UTCCP)
    sRing_raw = sAliased_raw  # K on sg0, V on sg1
    # fp32 S ship staging.  Separate from the cast buffer so this slot is
    # released on cast completion rather than on tma_store_wait.
    sXfer_raw = cutlass.Array(cutlass.Float32, sXferElems, alignment=128, space=cutlass.AddressSpace.smem)
    # io-dtype staging for the workspace store: S on sg0, dS on sg1.
    sCast_raw = cutlass.Array(WORKSPACE_DTYPE, castElems, alignment=128, space=cutlass.AddressSpace.smem)

    sOperand = SmemTile(
        base=sOperand_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
    )
    sRing = SmemTile(
        base=sRing_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_QK_GRANU_ELEMS,
    )
    sCast = SmemTile(
        base=sCast_raw,
        elems_per_stage=castElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=_CORE_MATRIX_ROWS * CFG.S_SWZ_BYTES,
        layout=SMEM_LAYOUT_S,
        tma_loads_per_tile=S_TMA_ITERS,
        tma_granu_elems=S_D_BLOCK,
        tma_subtile_stride_elems=S_SUBTILE_SLAB,
    )

    tma_q = GmemTileTma(tma_q_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_v = GmemTileTma(tma_v_desc)
    tma_do = GmemTileTma(tma_do_desc)
    tma_s = GmemTileTma(tma_s_desc)
    tma_ds = GmemTileTma(tma_ds_desc)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    sched = Sched(
        mb_scheduler=cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
        mb_read_tile_id=cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
        tile_id_smem=cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
        bidx_init=bidx,
        bidy_init=bidy,
        bidz_init=bidz,
    )
    bars = _make_bwd_d512_sm100_bars(CFG)

    # --- barrier init: init -> fence -> CTA sync -> cga sync -> arrives
    if warp_idx == 0:
        # ONE lane initialises.  `mbarrier.init` from all 32 lanes of the warp
        # is 32 concurrent inits of the same mbarrier object -- PTX leaves that
        # undefined, and the shipped forward (prefill_d512_f16_sm100.py:475)
        # elect-gates the whole block for exactly this reason.
        #
        # MBarrier.init() initialises exactly ONE stage (base_ptr at
        # stage_idx=0) -- it is not a whole-ring init.  Every stage of every
        # multi-stage ring must be initialised explicitly, or stages 1.. sit on
        # uninitialised SMEM and the first wait on them faults the launch.
        if nvvm.elect_sync():
            for _f in bars:
                for _i in cutlass.range_constexpr(_f.stages):
                    _f[_i].init()
            for _i in cutlass.range_constexpr(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(_i), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(_i), CFG.READ_TILE_ARRIVERS)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    # Required before ANY cross-CTA arrive: skipping it lets one land on
    # uninitialized SMEM under in-flight pressure.
    if cutlass.const_expr(CGA_SIZE > 1):
        cga_arrive()
        cga_wait()

    # --- warp dispatch ---------------------------------------------------
    if warp_idx < CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _compute_warp_group(
            bars,
            sched,
            sXfer_raw,
            sCast_raw,
            tmem_ptr_i32,
            lse_tensor,
            do_dot_tensor,
            cta_in_pair,
            is_sg0,
            leader_cta_id,
            cross_sg_peer,
            cta_id_x,
            n_kv,
            seqlen_q,
            seqlen_kv,
            attn_scale_in,
            attn_scale_log2e,
            attn_scale_for_dS,
            head_base,
            batch_base,
        )
    elif warp_idx == CFG.MMA_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp_leader(bars, sched, sOperand, sRing, tmem_ptr_i32, cta_in_pair, is_sg0, mcast_mask, n_kv, seqlen_q, seqlen_kv)
        else:
            _mma_warp_non_leader(bars, sched, tmem_ptr_i32, cta_in_pair)
    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmaldg_warp_group(
            bars,
            sched,
            sOperand,
            sRing,
            tma_q,
            tma_k,
            tma_v,
            tma_do,
            cta_in_pair,
            is_leader,
            is_sg0,
            tma_mcast_mask,
            n_kv,
            seqlen_q,
            seqlen_kv,
            gqa_ratio,
            head_base,
            batch_base,
        )
    elif warp_idx == CFG.TMASTG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(bars, sched, sCast, tma_s, tma_ds, cta_in_pair, is_sg0, n_kv, seqlen_q, seqlen_kv, head_base, batch_base)
    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta)


# ---------------------------------------------------------------------------
# Host wrapper + per-shape compile cache
# ---------------------------------------------------------------------------


def _tma_swz(byte_w: int):
    return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b


@cute.jit
def _host(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    do_tensor: cute.Tensor,
    s_tensor: cute.Tensor,
    ds_tensor: cute.Tensor,
    lse_tensor: cute.Tensor,
    do_dot_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int, int, int],  # (B, QH, S_q_pad, S_kv_pad, QH_chunk, QH_kv, S_q, S_kv)
    attn_scale_in: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    attn_scale_for_dS: cutlass.Float32,
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
    stream: _cuda_driver.CUstream = None,
) -> None:
    # SQ / SKV are the TILE-ROUNDED extents the grid and the workspace use;
    # SQ_REAL / SKV_REAL are the lengths the masks compare against. They differ
    # only when the caller's sequence length is not a multiple of the tile.
    B, QH, SQ, SKV, QH_CHUNK, QH_KV, SQ_REAL, SKV_REAL = problem_size

    # Q/K/V/dO are BSHD [B, S, H, D]; the workspaces are [B, H_chunk, S_q, S_kv].
    # stride_order is innermost-first, so the coords the kernel passes are
    # (d, head, seq, batch) for the operands and (kv, q, head, batch) for the
    # workspaces.
    stride_order = (3, 2, 1, 0)
    op_box = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)  # Q and dO: whole q-block
    ring_box = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)  # K and V: per-CTA N slice
    # One workspace subtile.  S_D_BLOCK * BPE == the swizzle atom exactly; a
    # full TILE_N row would be 256 B and exceed it, which is the
    # cuTensorMapEncodeTiled INVALID_VALUE / CUDA_EXCEPTION_27 trap.
    ws_box = (1, 1, CFG.TILE_M, S_D_BLOCK)

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor, box_dims=op_box, stride_order=stride_order, swizzle=_tma_swz(CFG.Q_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor, box_dims=ring_box, stride_order=stride_order, swizzle=_tma_swz(CFG.K_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor, box_dims=ring_box, stride_order=stride_order, swizzle=_tma_swz(CFG.V_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_do_desc = tmap.create_tensor_map_tiled_from_view(
        do_tensor, box_dims=op_box, stride_order=stride_order, swizzle=_tma_swz(CFG.DO_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_s_desc = tmap.create_tensor_map_tiled_from_view(
        s_tensor, box_dims=ws_box, stride_order=stride_order, swizzle=_tma_swz(CFG.S_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_ds_desc = tmap.create_tensor_map_tiled_from_view(
        ds_tensor, box_dims=ws_box, stride_order=stride_order, swizzle=_tma_swz(CFG.S_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )

    # One cluster covers CTA_MMA * TILE_M q-rows; CGA_M CTAs per cluster.
    rows_per_cluster = CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    grid_shape = (q_clusters * CFG.CGA_M, QH_CHUNK, B)

    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_do_desc,
        tma_s_desc,
        tma_ds_desc,
        lse_tensor,
        do_dot_tensor,
        seq_kv_lens_tensor,
        cutlass.Int32(SQ_REAL),
        cutlass.Int32(SKV_REAL),
        # GQA ratio, 1 for MHA. Q-head // this = the shared KV head.
        cutlass.Int32(QH // QH_KV),
        cutlass.Int32(SKV // CFG.TILE_N),
        cutlass.Int32(QH_CHUNK),
        cutlass.Int32(B),
        attn_scale_in,
        attn_scale_log2e,
        attn_scale_for_dS,
        head_base,
        batch_base,
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
    sq: int = 256,
    skv: int = 128,
    qh_chunk: int = 1,
    d: Optional[int] = None,
    qh_kv: Optional[int] = None,
    sq_real: Optional[int] = None,
    skv_real: Optional[int] = None,
) -> Callable:
    """Per-shape compile cache.

    ``qh_chunk`` decouples the workspace's head extent from the tensors': the
    host loop chunks the flattened (b, h) index to hold S+dS under the workspace
    cap, so Q/K/V/dO carry the full ``qh`` while the workspaces carry only the
    chunk.  ``head_base`` / ``batch_base`` are RUNTIME args, so one
    compiled artifact serves every chunk.
    """
    if sq % (CFG.TILE_M * CFG.CTA_MMA) != 0:
        raise ValueError(f"bwd d512: S_q must be a multiple of TILE_M*CTA_MMA ({CFG.TILE_M * CFG.CTA_MMA}); got {sq}")
    if skv % CFG.TILE_N != 0:
        raise ValueError(f"bwd d512: S_kv must be a multiple of TILE_N ({CFG.TILE_N}); got {skv}")
    if qh % qh_chunk != 0:
        raise ValueError(f"bwd d512: qh_chunk ({qh_chunk}) must divide qh ({qh}) -- even chunks keep one compiled artifact")
    # Head dims below TILE_K need NO kernel change: the TMA descriptors are built
    # `from_view`, so global_dims carries the REAL d, and a box reading past it is
    # HW zero-filled.  The zeros then contribute nothing to either BMM -- we pay
    # the full TILE_K-wide MMA and get the right answer.  The only hard rule is
    # TMA's: the innermost extent must be 16-byte aligned, i.e. d * BPE % 16 == 0.
    qh_kv = qh if qh_kv is None else int(qh_kv)
    # sq / skv are the TILE-ROUNDED compile extents; the real lengths drive the
    # masks and the operand tensors' own extents.
    sq_real = sq if sq_real is None else int(sq_real)
    skv_real = skv if skv_real is None else int(skv_real)
    if qh % qh_kv != 0:
        raise ValueError(f"bwd d512: qh ({qh}) must be a multiple of qh_kv ({qh_kv})")
    d = CFG.TILE_K if d is None else int(d)
    if not (0 < d <= CFG.TILE_K):
        raise ValueError(f"bwd d512: d must be in (0, {CFG.TILE_K}]; got {d}")
    if (d * CFG.BPE) % 16 != 0:
        raise ValueError(f"bwd d512: d * {CFG.BPE} B must be a multiple of 16 (TMA innermost extent); got d={d}")

    def _fake_bshd(shape, dtype=STORAGE_DTYPE):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)

    # Q/K/V/dO carry the REAL sequence length: a tile reading past it is TMA
    # zero-filled, which is exactly what the tail mask expects to see.
    fake_q = _fake_bshd((b, sq_real, qh, d))
    # K/V are indexed by the KV head, so their head extent is qh_kv.
    fake_k = _fake_bshd((b, skv_real, qh_kv, d))
    fake_v = _fake_bshd((b, skv_real, qh_kv, d))
    fake_do = _fake_bshd((b, sq_real, qh, d))
    # Chunk-local workspaces, [B, H_chunk, S_q, S_kv] at the io dtype.
    fake_s = _fake_bshd((b, qh_chunk, sq, skv), dtype=WORKSPACE_DTYPE)
    fake_ds = _fake_bshd((b, qh_chunk, sq, skv), dtype=WORKSPACE_DTYPE)
    # Plain per-lane reads, one q-row per lane -- no TMA, no barriers.
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq_real), stride_order=(2, 1, 0), assumed_align=16)
    # do_dot's producer (dot_do_o_kernel) indexes delta with a row stride of
    # ceil(S_q / 128) * 128, NOT S_q -- so the buffer, and this view of it, must
    # use the same rounding. They coincide whenever S_q is a multiple of 128,
    # which is why a non-multiple was the only shape that exposed it.
    sq_dot = -(-sq_real // 128) * 128
    fake_do_dot = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq_dot), stride_order=(2, 1, 0), assumed_align=16)
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b,), stride_order=(0,), assumed_align=16)

    return cute.compile(
        _host,
        fake_q,
        fake_k,
        fake_v,
        fake_do,
        fake_s,
        fake_ds,
        fake_lse,
        fake_do_dot,
        fake_seq_kv_lens,
        (b, qh, sq, skv, qh_chunk, qh_kv, sq_real, skv_real),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


__all__ = ["CFG", "PARAMS", "Bars", "LAYOUT", "STORAGE_DTYPE", "WORKSPACE_DTYPE", "_kernel", "_host", "compile"]
