# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SDPA backward stage 2, d_qk = d_v = 512, bf16/fp16, SM100 (Blackwell) -- the 2x2-datapath twin.

Produces the ``S`` and ``dS`` workspaces that stage 3's three GEMMs reduce into
dV / dK / dQ.  Computes no gradient itself::

    S  = exp2(attn_scale_log2e * S_acc  - lse * log2e)      then masked to 0.0
    dS = (attn_scale_for_dS * dS_acc - do_dot * attn_scale_in) * S

Same ABI, workspace format, LSE / do_dot contract and mask-after-exp2 rule as
the cga4x1 role-split sibling ``bprop_d512_f16.py``; what differs is the
datapath:

**One FUSED pipeline per ``cta_group::2`` pair** (CGA_M=4 -> pairs {0,1}, {2,3}):

    every CTA   64 q rows; BMM1  mma_ss(Q  from SMEM, K from SMEM) -> S_acc  (collective M = 128)
                           BMM2  mma_ss(dO from SMEM, V from SMEM) -> dS_acc
                softmax -> S in registers ; dS = f(S) in the SAME lane ; bf16 S, dS -> workspace

The 2x2 D image of ``tcgen05.mma.cta_group::2`` at M = 128: D row m, column n
lands in TMEM lane ``(m % 64) + 64 * (n // 64)``, column ``n % 64`` -- a 64 x 128
fp32 accumulator is 64 columns over all 128 lanes, so compute warp w holds rows
``32 * (w % 2) ..+31`` of kv-column half ``w // 2`` of BOTH accumulators at the
same lane, and the dS product never leaves the lane.  No fp32 S ship, no DSMEM,
no UTCCP, no alias seam, no named barrier 8.

**The cluster stays (4,1,1) and covers 256 q rows** (``CLUSTER_Q_ROWS``), so
stage 3's ``causal_gran``, the THD unit and ``READ_TILE_ARRIVERS`` are what the
4x1 kernel has.  The two pairs share K / V by TMA multicast: d is streamed in
``D_CHUNK = 64`` column chunks and chunk ``c`` of a kv tile is issued by the
pair with ``pair_id == c & 1`` to CTAs ``{c, c ^ 2}`` (mask ``5 << cta_in_pair``,
``group = cta_2``); every byte that lands in a CTA completes on ITS pair
leader's ``mb_tma_ring_full`` (destination-pair-leader routing, probe
``mcast_twin`` 2026-10-01), so ONLY pair leaders arm ``expect_tx`` and ONLY the
leader's MMA warp waits it.  A ring slot is refilled only after BOTH pairs'
MMAs have read it: ``mb_tma_ring_empty`` has init 2 and both pair leaders'
``tcgen05.commit`` carry mask 0xF.  Each shared ``ring_full`` has init 4:
its leader's ``expect_tx`` arrival plus one ACK from each other CTA, sent only
after that CTA observes its local ``ring_empty`` phase.  This gates both MMAs
and therefore the next empty release on all four observations.  Without ACKs,
a passive follower can miss a phase while its slot is recycled (parity ABA).

**There is no online softmax.**  LSE arrives as a kernel input, so there is no
running max, no rescale, no alpha/stats ship, no correction warp -- and no
exchange between the two warps that share a row.

**The mask is applied to S AFTER exp2, with 0.0 (not -inf before).**  dS is a
product with S, so zeroing S zeroes dS for free.  Do not move it.

**Every mbarrier init count is a named ``CFG`` constant, and the comment beside
it names the ARRIVE SITES it counts and the guard each one fires under.**  Keep
that pairing when changing either side: a mismatch is an intermittent hang whose
output is correct on every launch that completes.

**Measured (2026-10-01, B200 sm_100a at 1155 MHz, DSL 4.7.0, CUPTI medians of 30
trials per slot, L2 flushed, one process per slot, every slot started AND ended
with an empty ``nvidia-smi`` compute-apps list, A/B/A against the 4x1 role split;
MEDIANS over 3 slots per arm, ``lane_d512_bprop/fix/ab_table_fix_medians.md``).**
The October 1 polling version, before the observer ACK fix (``POLL_TIGHT_ITERS``
128 / ``POLL_SLEEP_NS`` 128), B=1 H=128 d=512 bf16:

    dense  S=8192  stage 2  42049 -> 40861 us  (+2.9 %;  6855 -> 6661 clk per SM per kv tile, floor 2304)
                   whole bwd 72525 -> 73191 us (-0.9 %)
    causal S=8192  stage 2  19092 -> 18222 us  (+4.8 %;  6036 -> 5761 clk/tile)   whole 36042 -> 35326 (+2.0 %)
    dense  S=2048  stage 2   2073 ->  2264 us  (-8.4 %;  5408 -> 5904 clk/tile)   whole  3873 ->  4065 (-4.7 %)

The pre-fix kernel (every wait parked, the one that hangs under GPU sharing)
measured 42097 -> 39166 (+7.5 %) dense 8K, 18565 -> 17817 (+4.2 %) causal 8K and
2072 -> 2111 (-1.9 %) dense 2K on the lane's slots (``fix/ab_table_medians.md``;
its first dense8k 4x1 slot, 45063 us, and three non-exclusive slots are flagged
there -- the lane's earlier MEANS +9.3 / +5.9 % absorbed them).  So the poll costs
~3-4 % at S=8192 and ~7 % at S=2048 against the parked twin: the two polling
warps share their SMSPs with compute warps 0 / 1 (a tight poll cost 2.2x, see
``_poll_wait``).  Slot-to-slot noise is +-3 %, so the S=8192 gate (>= +3 % on
both masks) is NOT met by this kernel (dense +2.9 %), and S=2048 regresses
further (the per-tile Q + dO prologue, 128 KiB per CTA and not double-buffered,
is paid over only 16 kv tiles, now plus the poll).  Stage 2 runs at ~2.9x its
MMA floor: the fused datapath removed ~200-300 of the ~4500 clk/tile of
overhead, so the role split's S ship was NOT what held the 4x1 at ~6900
clk/tile; the remaining gap is not attributed here (levers to A/B: ``d_chunk``
128 x 2 stages, ``stages_acc`` 4, a deeper ring on Rubin).  Numerics: bitwise
identical to the role split (S / dS workspace and dQ / dK / dV, ``torch.equal``
on int16 views) on dense, causal, SWA, bottom-right, GQA, fp16.

**The GPU-sharing hang: historical polling experiment and observer fix.**
The October 1 heartbeat dump (``lane_d512_bprop/fix/HANDOFF2.md``,
``fix/e2_heartbeat_wf0.log``) found a passive follower at ``ring_empty[0]``
while the other CTAs had advanced five chunks.  Sleeping and hint-less
``try_wait`` forms hung within 2-74 launches; polling completed 200/200 and
300/300.  Those finite passes did not establish correctness or prove a lost
hardware wake-up.  Polling also leaves the observer lifecycle unordered:
a local empty barrier can advance twice before the passive CTA observes it.

The shared-ring ACK protocol above preserves every CTA's empty wait and
prevents that phase overrun.  Removing the passive wait is not sufficient:
the multicast still advances that CTA's barrier object, whose phase must be
observed before reuse.  Keep polling and the existing diagnostics; the ACKs
supply the missing ordering.  ``KV_SHARE = 1`` keeps its existing pair-local
protocol.  Runnable detectors:
``test_sdpa_bwd_dsl_sm100.py::test_stage2_2x2_waits_for_delayed_empty_observer``
and ``test_sdpa_bwd_d512_sm107.py::test_chain_waits_for_delayed_empty_observer``
delay a passive observer at a reused slot, then check gradients and capture /
replay.  The natural GPU-sharing positive tests remain.  The old must-hang
``wait_form = 4`` negative control was probabilistic and is replaced by the
delayed-observer regression.  ``debug_heartbeat`` / ``debug_dump_addr`` and
all wait-form levers remain available for diagnosis.  Performance and a
default-dispatch change are separate decisions.
"""

from typing import NamedTuple, Optional, Tuple

import cuda.bindings.driver as _cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass._mlir.dialects import arith
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import (
    WAIT_TIMEOUT,
    MBarrier,
    wait,
    wait_poll,
    PipelineState,
    Producer,
    Scope,
    advance,
    cga_arrive,
    cga_wait,
)
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_NONE, MASK_PADDED, apply_mask_chunk, compute_kv_loop_bounds
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss
from cudnn.frost.tile_dsl.scheduler import Sched, read_tile_id_arrive, scheduler_warp_loop, scheduler_warp_loop_persistent, read_clc_payload
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from cudnn.frost.tile_dsl.pointwise import tmem_load_tile
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.sdpa.bwd.kernels.sm100._common import make_bwd_decode
from cudnn.frost.tile_dsl.thd import emit_clamped_desc
from cudnn.sdpa.bwd.config_sm100 import (
    TemplateParams,
    TemplateParams2x2,
    acc_cols_2x2,
    cast_bytes_2x2,
    cast_subtiles_2x2,
    desc_version_2x2,
    make_cfg_d512_2x2,
    op_tx_bytes_2x2,
    operand_bytes_2x2,
    ring_tx_bytes_2x2,
    smem_bytes_2x2,
    smem_layout_2x2,
    tmem_cols_2x2,
)

# Injected by the loader before this body executes; a plain import gets the
# all-defaults config (dense bf16, SM100 ring), which keeps
# `python sm100/bprop_d512_f16_2x2.py` usable as a standalone benchmark.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams2x2())
CFG = make_cfg_d512_2x2(PARAMS)
_decode_initial, _decode_payload = make_bwd_decode(CFG)
# THD / varlen: Q/K/V/dO and the gradients are PACKED [1, T, H, D], the
# per-sequence lengths come from a device metadata buffer, and the S/dS
# workspace is BLOCKED over packed q tokens.  Every use folds out of the dense
# build at trace time.
_THD = bool(CFG.THD_VARLEN)
_TENSOR_MAP_QWORDS = 128 // 8
# Stage 2's OWN descriptor-scratch slot map (the same four slots, same order and
# same sequence axis as the 4x1 sibling: `prepared_host` sizes the scratch once).
Q_SLOT, DO_SLOT, K_SLOT, V_SLOT = 0, 1, 2, 3
THD_STAGE2_DESC_SLOTS = 4
_THD_SEQ_ORD = 2

# Does any tile in this specialization need the per-cell mask at all?  Folds the
# whole mask apparatus out of the dense build.
_MASKED = CFG.MASK_FLAGS != MASK_NONE
_PADDED = bool(CFG.MASK_FLAGS & MASK_PADDED)
# Do the two pairs share one K / V ring by cross-pair multicast (KV_SHARE 2), or does each pair load its own (1)?
_KV_SHARED = CFG.KV_SHARE == 2

# ---------------------------------------------------------------------------
# DEBUG lever (TemplateParams2x2.debug_wait_ms / debug_dump_addr; default OFF
# = zero traced code).  Every mbarrier wait of every warp body goes through
# `_wait_b`: a %globaltimer-bounded try_wait loop that, on timeout, writes one
# 16 x Int32 record for this (CTA, warp) to the dump buffer and then KEEPS
# waiting, so a hang stays frozen for the host to read (it polls the pinned
# buffer and kills the process).  The record slot is
# (linear block id * TOTAL_WARPS + warp) * DBG_WORDS; the host allocates
# >= DBG_MAX_WORDS Int32 (16 MiB).
# ---------------------------------------------------------------------------
_DBG_WAIT_NS: int = int(CFG.DEBUG_WAIT_MS) * 1_000_000
_DBG_DUMP_ADDR: int = int(CFG.DEBUG_DUMP_ADDR)
_DBG_HEARTBEAT: bool = bool(CFG.DEBUG_HEARTBEAT) and _DBG_DUMP_ADDR != 0
_DBG_BOUNDED: bool = _DBG_WAIT_NS > 0 and _DBG_DUMP_ADDR != 0
# ATTRIBUTION lever (TemplateParams2x2.debug_clk; default OFF = zero traced code): the waits keep their production form,
# and every warp accumulates in its own SMEM slice the %clock64 spent INSIDE each barrier's waits (per DBG_BAR id), the
# number of those waits that actually blocked, and the clock of its issue segments; `_dbg_exit` writes the slice to the
# dump buffer as 32 x Int64 at (linear block * TOTAL_WARPS + warp) * DBG_CLK_WORDS.  The record index is NOT bounds-
# checked: the 16 MiB dump holds DBG_CLK_MAX_WORDS / (TOTAL_WARPS * DBG_CLK_WORDS) = 8192 CTAs per launch, so a larger
# stage-2 grid (B4 S2K H128 = 16384 CTAs in one head chunk; a THD N_THD_UNITS grid above it) must be chunked by the
# host or the records overrun the buffer -- the GPU accounting test and the job's attr_dbg.py assert the bound before
# the launch.  The record layout is the DBG_CLK_* constants below (slot DBG_CLK_BODY_NS over slot 0 = the SM clock the
# body ran at); the accounting test reads it, a decoder is a few lines over these constants.
_DBG_CLK: bool = bool(CFG.DEBUG_CLK) and _DBG_DUMP_ADDR != 0
# Any debug mode needs the dump context built in the kernel entry.
_DBG: bool = _DBG_BOUNDED or _DBG_HEARTBEAT or _DBG_CLK
DBG_WORDS = 16
DBG_MAX_WORDS = 1 << 22
# Attribution record (Int64 words per warp).
DBG_CLK_WORDS = 32
DBG_CLK_MAX_WORDS = 1 << 21  # Int64; the same 16 MiB dump buffer as DBG_MAX_WORDS Int32
DBG_CLK_TOTAL = 0  # body clk: clock64 at _dbg_exit - clock64 at entry (slot 0 holds the entry stamp until exit)
DBG_CLK_WAIT_BASE = 0  # + DBG_BAR id (1..10): clk spent inside that barrier's waits
DBG_CLK_SEG_STG_STORE = 11  # TMA-STG: tma_store issue -> tma_store_wait(0) returned, per kv tile
DBG_CLK_SEG_CMP_MATH = 12  # compute: bmm_done passed -> smem_empty wait entered (TMEM loads, exp2, mask, dS)
DBG_CLK_SEG_CMP_CAST = 13  # compute: smem_empty passed -> smem_full arrived (cast + swizzled stores + fence)
DBG_CLK_SEG_MMA_ISSUE = 14  # MMA leader: ring_full passed -> ring_empty committed, per chunk (2 MMAs + commit)
DBG_CLK_SEG_LDG_ISSUE = 15  # TMA-LDG: ring_empty passed -> K / V chunk issued, per chunk
DBG_CLK_TILES = 16  # q tiles this warp ran (tile_no at exit)
DBG_CLK_KV_TOTAL = 17  # role total: MMA acc_total (kv tiles), LDG ring_total (chunks), STG stg_total (kv tiles), compute 0
DBG_CLK_BLOCKED_BASE = 17  # + DBG_BAR id (1..10): waits on that barrier that took longer than DBG_CLK_BLOCKED_THRESH clk
DBG_CLK_BLOCKED_THRESH = 128  # a wait that passes on its first test costs ~30-60 clk of issue; above this it really waited
DBG_CLK_BODY_NS = 28  # %globaltimer ns over the same body span as slot 0 -> the SM clock the body actually ran at (clk / ns)
# Record words.
DBG_W_STATUS, DBG_W_BAR, DBG_W_IDX, DBG_W_PHASE = 0, 1, 2, 3
DBG_W_AUX0, DBG_W_AUX1, DBG_W_AUX2, DBG_W_AUX3, DBG_W_AUX4 = 4, 5, 6, 7, 8
DBG_W_CTA, DBG_W_SMID, DBG_W_BIDX, DBG_W_BIDY, DBG_W_BIDZ, DBG_W_RAW_LO, DBG_W_RAW_HI = 9, 10, 11, 12, 13, 14, 15
# Status values.  TIMEOUT / EXITED are the bounded-wait lever's; WAITING / RUNNING the heartbeat's (RUNNING keeps the
# rest of the record, so a RUNNING warp's record names the LAST wait it passed).
DBG_STATUS_TIMEOUT, DBG_STATUS_EXITED, DBG_STATUS_WAITING, DBG_STATUS_RUNNING = 1, 2, 3, 4
# SNAPSHOT: under heartbeat + a wait budget, the TMA-STG warp's smem_full wait is the ONE bounded wait (it is the last
# consumer of the chain and its form cannot be what wedges the K / V ring); on timeout it overwrites words 2..15 of its
# own record with this CTA's raw mbarrier words: ring_empty[0..3] (64-bit each), tmem_dealloc (64-bit, a calibration
# reference: init 1, never arrived mid-kernel), acc_empty[0] (low 32), smem_full[0] (low 32).
DBG_STATUS_SNAPSHOT = 5
# Barrier ids (word 1).  aux0 = kv_loop (-1 none, -2 end-of-kernel drain), aux1 = chunk / sub-step, aux2 = q_block,
# aux3 = tiles started by this warp, aux4 = role-specific running total.
DBG_BAR_OP_FULL, DBG_BAR_OP_EMPTY, DBG_BAR_RING_FULL, DBG_BAR_RING_EMPTY = 1, 2, 3, 4
DBG_BAR_BMM_DONE, DBG_BAR_ACC_EMPTY, DBG_BAR_SMEM_FULL, DBG_BAR_SMEM_EMPTY = 5, 6, 7, 8
DBG_BAR_TMEM_DEALLOC, DBG_BAR_SCHED = 9, 10
DBG_NONE = -1
DBG_DRAIN = -2


class Dbg(NamedTuple):
    """Per-warp debug context: the dump view (Int32 records; Int64 under the attribution lever), this warp's record word
    offset and the CTA's identity; under the attribution lever also the SMEM accumulator array and this warp's slice.
    The accumulators are read-modify-written by the ``elect_sync()`` lane with no warp sync between the RMWs: PTX does not
    promise the same lane each time, so this leans on the warp's single in-order instruction stream over its own private
    slice (test-only; a fixed ``lane_idx == 0`` or a ``bar.warp.sync`` before each read is the formally ordered form)."""

    arr: object
    slot: object
    cta_id_x: object
    bidx: object
    bidy: object
    bidz: object
    clk: object = 0
    cslot: object = 0


@cute.jit
def _dbg_record(dbg, status: int, bar_id: int, idx, phase, aux0, aux1, aux2, aux3, aux4, raw):
    """One elected lane writes this warp's record; the status word goes LAST so a half-written record never reads as set."""
    if nvvm.elect_sync():
        s = dbg.slot
        dbg.arr[s + cutlass.Int32(DBG_W_BAR)] = cutlass.Int32(bar_id)
        dbg.arr[s + cutlass.Int32(DBG_W_IDX)] = cutlass.Int32(idx)
        dbg.arr[s + cutlass.Int32(DBG_W_PHASE)] = cutlass.Int32(phase)
        dbg.arr[s + cutlass.Int32(DBG_W_AUX0)] = cutlass.Int32(aux0)
        dbg.arr[s + cutlass.Int32(DBG_W_AUX1)] = cutlass.Int32(aux1)
        dbg.arr[s + cutlass.Int32(DBG_W_AUX2)] = cutlass.Int32(aux2)
        dbg.arr[s + cutlass.Int32(DBG_W_AUX3)] = cutlass.Int32(aux3)
        dbg.arr[s + cutlass.Int32(DBG_W_AUX4)] = cutlass.Int32(aux4)
        dbg.arr[s + cutlass.Int32(DBG_W_CTA)] = cutlass.Int32(dbg.cta_id_x)
        dbg.arr[s + cutlass.Int32(DBG_W_SMID)] = cute.arch.smid()
        dbg.arr[s + cutlass.Int32(DBG_W_BIDX)] = cutlass.Int32(dbg.bidx)
        dbg.arr[s + cutlass.Int32(DBG_W_BIDY)] = cutlass.Int32(dbg.bidy)
        dbg.arr[s + cutlass.Int32(DBG_W_BIDZ)] = cutlass.Int32(dbg.bidz)
        dbg.arr[s + cutlass.Int32(DBG_W_RAW_LO)] = cutlass.Int32(raw & cutlass.Int64(0xFFFFFFFF))
        dbg.arr[s + cutlass.Int32(DBG_W_RAW_HI)] = cutlass.Int32(raw >> cutlass.Int64(32))
        dbg.arr[s + cutlass.Int32(DBG_W_STATUS)] = cutlass.Int32(status)


# The poll's shape (module constants, not levers): up to POLL_TIGHT_ITERS back-to-back tests (a few us: a phase that is
# about to flip pays no extra latency), then a plain timer ``nanosleep`` between tests.  The timer sleep wakes on its own
# (SASS ``NANOSLEEP``, not the event-sleep ``NANOSLEEP.SYNCS`` that loses the wake-up), so the wait still never parks on the
# barrier.  MEASURED (B200 @1155 MHz, dense B=1 H=128 S=8192, CUPTI stage-2 medians, one clean twin slot per point,
# lane_d512_bprop/fix/poll_sweep.out + abtight_* / absleep* logs; 4x1 on the same slots 41696-43186 us, pre-fix parked
# twin 39166):
#     tight forever  94575 us   (2.2x: the pollers on the MMA and TMA-LDG warps share SMSPs 0 / 1 with compute warps 0 / 1
#                                and take half their issue slots on the softmax critical path)
#     4 / 64 ns      41426, 41930      32 / 256 ns    41338      64 / 64 ns   41602
#     32 / 128 ns    40531, 40391, 41595               128 / 128 ns  40312   <- shipped
# More tight iterations and fewer, longer sleeps win; 32/128 and 128/128 are within the ~3 % slot noise, and the residual
# cost against the parked twin is ~3 %.  (Both 2x2 forwards ship the shared default 32 / 128: their pollers wait briefly,
# and on the cc 10.7 forward the tight loop and 32 / 128 measured within 0.15 % of each other.)
POLL_TIGHT_ITERS = 128
POLL_SLEEP_NS = 128


@cute.jit
def _poll_wait(mb, phase):
    """A wait that NEVER parks the warp on the barrier: ``mbarrier.test_wait.parity`` until the phase flips, with a timer
    back-off after ``POLL_TIGHT_ITERS`` tests.

    The form every barrier whose completing event is issued from OUTSIDE the pair must take (``ring_empty``: the partner
    leader's ``tcgen05.commit`` multicast; ``ring_full``: the partner's TMA ``complete_tx``).  Measured 2026-10-01 under
    GPU time-slicing: a warp parked in NANOSLEEP.SYNCS (the ``try_wait`` retry of tile_dsl ``wait()``, with the 1 ns or
    the 10 ms hint, and the hint-less spin alike) on such a barrier can miss the wake-up and the cluster hangs; this
    poll never does (``lane_d512_bprop/fix/HANDOFF2.md``).

    The kernel's ONE call of the shared ``tile_dsl.barrier.wait_poll`` (an inline-PTX ``mbarrier.test_wait.parity`` loop
    with the timer back-off; the DSL's ``nvvm.mbarrier_test_wait`` wrapper raises a TypeError on 4.7.0) in THIS kernel's shape
    (``POLL_TIGHT_ITERS`` / ``POLL_SLEEP_NS``); ``_wait_plain`` WAIT_FORM 0 and 2 call it."""
    wait_poll(mb, phase, tight_iters=POLL_TIGHT_ITERS, sleep_ns=POLL_SLEEP_NS)


@cute.jit
def _wait_plain(mb, phase, poll: bool = False):
    """The kernel's mbarrier wait: ``poll`` (a Python constant) marks a barrier released from outside the pair, which
    under the shipped ``WAIT_FORM`` 0 takes :func:`_poll_wait` while every pair-local barrier takes tile_dsl ``wait()``.
    ``WAIT_FORM`` 1-4 are diagnostic arms applied to every wait (see the config comment)."""
    if cutlass.const_expr(CFG.WAIT_FORM == 0):
        if cutlass.const_expr(poll):
            _poll_wait(mb, phase)
        else:
            wait(mb, phase)
    elif cutlass.const_expr(CFG.WAIT_FORM == 1):
        wait(mb, phase, spin=True)
    elif cutlass.const_expr(CFG.WAIT_FORM == 2):
        _poll_wait(mb, phase)
    elif cutlass.const_expr(CFG.WAIT_FORM == 3):
        while not nvvm.mbarrier_try_wait_parity(mb, phase, time_limit=10_000_000):
            pass
    else:
        # 4: the pre-fix kernel -- the sleeping wait on the ring barriers too (the negative control of the contention test).
        wait(mb, phase)


@cute.jit
def _wait_b(mb, phase, dbg, bar_id: int, idx, aux0, aux1, aux2, aux3, aux4, poll: bool = False):
    """``_wait_plain(mb, phase, poll)``; with the bounded-wait lever armed, bounded by ``_DBG_WAIT_NS`` of %globaltimer ->
    record -> wait on; with the heartbeat armed, the record (status WAITING) goes out BEFORE the unmodified wait and the
    status flips to RUNNING after it; with the attribution lever (``_DBG_CLK``) armed, the UNMODIFIED wait is bracketed by
    two %clock64 reads and the elected lane adds the delta to this warp's bucket for ``bar_id`` (plus one to its blocked
    count when the wait took more than DBG_CLK_BLOCKED_THRESH clk)."""
    if cutlass.const_expr(not _DBG):
        _wait_plain(mb, phase, poll)
    elif cutlass.const_expr(_DBG_HEARTBEAT):
        _dbg_record(dbg, DBG_STATUS_WAITING, bar_id, idx, phase, aux0, aux1, aux2, aux3, aux4, cutlass.Int64(0))
        _wait_plain(mb, phase, poll)
        if nvvm.elect_sync():
            dbg.arr[dbg.slot + cutlass.Int32(DBG_W_STATUS)] = cutlass.Int32(DBG_STATUS_RUNNING)
    elif cutlass.const_expr(_DBG_CLK):
        # The production wait, bracketed by two clock reads; the elected lane accumulates into this warp's SMEM slice.
        t0 = cute.arch.clock64()
        _wait_plain(mb, phase, poll)
        dt = cute.arch.clock64() - t0
        if nvvm.elect_sync():
            i = dbg.cslot + cutlass.Int32(DBG_CLK_WAIT_BASE + bar_id)
            dbg.clk[i] = dbg.clk[i] + dt
            if dt > cutlass.Int64(DBG_CLK_BLOCKED_THRESH):
                j = dbg.cslot + cutlass.Int32(DBG_CLK_BLOCKED_BASE + bar_id)
                dbg.clk[j] = dbg.clk[j] + cutlass.Int64(1)
    else:
        t0 = cute.arch.globaltimer()
        done = cutlass.Int32(0)
        timed_out = cutlass.Int32(0)
        while done == cutlass.Int32(0):
            if nvvm.mbarrier_try_wait_parity(mb, phase, time_limit=WAIT_TIMEOUT):
                done = cutlass.Int32(1)
            elif cute.arch.globaltimer() - t0 > cutlass.Int64(_DBG_WAIT_NS):
                done = cutlass.Int32(1)
                timed_out = cutlass.Int32(1)
        if timed_out == cutlass.Int32(1):
            # The raw 64-bit mbarrier word (opaque, but it tells phase / pending / tx apart across CTAs).
            raw = mb.load()
            _dbg_record(dbg, DBG_STATUS_TIMEOUT, bar_id, idx, phase, aux0, aux1, aux2, aux3, aux4, raw)
            _wait_plain(mb, phase, poll)


@cute.jit
def _clk_add(dbg, word: int, dt):
    """Attribution lever: the elected lane adds ``dt`` clk to this warp's SMEM slot ``word`` (its own slice: no atomics)."""
    if cutlass.const_expr(_DBG_CLK):
        if nvvm.elect_sync():
            i = dbg.cslot + cutlass.Int32(word)
            dbg.clk[i] = dbg.clk[i] + dt


@cute.jit
def _dbg_exit(dbg, tile_no, kv_total):
    """This warp left its persistent loop and every drain.  Under the attribution lever (the primary branch): close the body
    clock and its %globaltimer span, record the tile counts and write the warp's 32 x Int64 slice out.  Under the bounded /
    heartbeat levers: status EXITED (a warp that timed out never gets here)."""
    if cutlass.const_expr(_DBG_CLK):
        if nvvm.elect_sync():
            i0 = dbg.cslot
            dbg.clk[i0] = cute.arch.clock64() - dbg.clk[i0]
            dbg.clk[i0 + cutlass.Int32(DBG_CLK_BODY_NS)] = cute.arch.globaltimer() - dbg.clk[i0 + cutlass.Int32(DBG_CLK_BODY_NS)]
            dbg.clk[i0 + cutlass.Int32(DBG_CLK_TILES)] = cutlass.Int64(tile_no)
            dbg.clk[i0 + cutlass.Int32(DBG_CLK_KV_TOTAL)] = cutlass.Int64(kv_total)
            for _w in cutlass.range_constexpr(DBG_CLK_WORDS):
                dbg.arr[dbg.slot + cutlass.Int32(_w)] = dbg.clk[i0 + cutlass.Int32(_w)]
    elif cutlass.const_expr(_DBG):
        if nvvm.elect_sync():
            dbg.arr[dbg.slot + cutlass.Int32(DBG_W_STATUS)] = cutlass.Int32(DBG_STATUS_EXITED)


@cute.jit
def _dbg_store64(dbg, word: int, raw):
    dbg.arr[dbg.slot + cutlass.Int32(word)] = cutlass.Int32(raw & cutlass.Int64(0xFFFFFFFF))
    dbg.arr[dbg.slot + cutlass.Int32(word + 1)] = cutlass.Int32(raw >> cutlass.Int64(32))


@cute.jit
def _wait_stg(bars, smem_state, dbg, kv_loop, q_block, tile_no, stg_total):
    """The TMA-STG warp's ``smem_full`` wait.  Plain ``_wait_b`` unless heartbeat AND a wait budget are both armed: then
    this one wait is bounded and, on timeout, snapshots the CTA's barrier words (see DBG_STATUS_SNAPSHOT) -- the
    evidence that tells a lost arrive (pending count still owed) from a sleeping waiter that missed a completed phase."""
    if cutlass.const_expr(not (_DBG_HEARTBEAT and _DBG_BOUNDED)):
        _wait_b(
            bars.mb_smem_full[smem_state.idx].smem_ptr,
            smem_state.phase,
            dbg,
            DBG_BAR_SMEM_FULL,
            smem_state.idx,
            kv_loop,
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            stg_total,
        )
    else:
        mb = bars.mb_smem_full[smem_state.idx].smem_ptr
        _dbg_record(
            dbg,
            DBG_STATUS_WAITING,
            DBG_BAR_SMEM_FULL,
            smem_state.idx,
            smem_state.phase,
            kv_loop,
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            stg_total,
            cutlass.Int64(0),
        )
        t0 = cute.arch.globaltimer()
        done = cutlass.Int32(0)
        timed_out = cutlass.Int32(0)
        while done == cutlass.Int32(0):
            if nvvm.mbarrier_try_wait_parity(mb, smem_state.phase, time_limit=WAIT_TIMEOUT):
                done = cutlass.Int32(1)
            elif cute.arch.globaltimer() - t0 > cutlass.Int64(_DBG_WAIT_NS):
                done = cutlass.Int32(1)
                timed_out = cutlass.Int32(1)
        if timed_out == cutlass.Int32(1):
            if nvvm.elect_sync():
                for _s in cutlass.range_constexpr(min(CFG.STAGES_KV, 4)):
                    _dbg_store64(dbg, 2 + 2 * _s, bars.mb_tma_ring_empty[_s].smem_ptr.load())
                _dbg_store64(dbg, 10, bars.mb_tmem_dealloc.smem_ptr.load())
                dbg.arr[dbg.slot + cutlass.Int32(12)] = cutlass.Int32(bars.mb_acc_empty[0].smem_ptr.load() & cutlass.Int64(0xFFFFFFFF))
                dbg.arr[dbg.slot + cutlass.Int32(13)] = cutlass.Int32(bars.mb_smem_full[0].smem_ptr.load() & cutlass.Int64(0xFFFFFFFF))
                dbg.arr[dbg.slot + cutlass.Int32(14)] = kv_loop
                dbg.arr[dbg.slot + cutlass.Int32(15)] = cutlass.Int32(dbg.cta_id_x)
                dbg.arr[dbg.slot + cutlass.Int32(DBG_W_STATUS)] = cutlass.Int32(DBG_STATUS_SNAPSHOT)
            _wait_plain(mb, smem_state.phase)
        if nvvm.elect_sync():
            dbg.arr[dbg.slot + cutlass.Int32(DBG_W_STATUS)] = cutlass.Int32(DBG_STATUS_RUNNING)


@cute.jit
def _rt_desc(desc_words, slot):
    """Pointer to one 128-byte runtime TMA descriptor in the setup array (the packed-total clamp, issue #624)."""
    return (desc_words.iterator.raw_ptr() + slot * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _kv_tile_bounds(q_block, cta_id_x, seqlen_q, seqlen_kv, n_kv):
    """The kv TILE range this cluster visits, plus where the mask starts biting.

    Keyed on the CLUSTER's q span (``CLUSTER_Q_ROWS`` rows from ``(q_block - cta_id_x) * TILE_M``), not the
    CTA's: every barrier here is a cross-CTA protocol whose trip count must match on all four CTAs -- and
    the cross-pair K / V ring makes the two PAIRS lock-step too, so all four must agree.  The CTA holding
    the earlier rows then computes a few fully-masked tiles; the per-cell mask zeroes them.

    Dense folds to the whole range at trace time.  Returns ``(left, unmasked_lo, unmasked_hi, right)``;
    only ``[unmasked_lo, unmasked_hi)`` is guaranteed free of masked cells.
    """
    if cutlass.const_expr(not _MASKED):
        return cutlass.Int32(0), cutlass.Int32(0), n_kv, n_kv
    cluster_q_row = (q_block - cta_id_x) * cutlass.Int32(CFG.TILE_M)
    b = compute_kv_loop_bounds(
        cluster_q_row,
        seqlen_q,
        seqlen_kv,
        CFG.WINDOW_LEFT,
        CFG.MASK_FLAGS,
        CFG.TILE_N,
        CFG.CLUSTER_Q_ROWS,
        bottom_right=CFG.BOTTOM_RIGHT,
        window_right=CFG.WINDOW_RIGHT,
    )
    return b.left, b.unmasked_lo, b.unmasked_hi, b.right


def _require(cond, msg):
    """Geometry sanity check; raises instead of assert (asserts vanish under -O)."""
    if not cond:
        raise ValueError(f"bprop_d512_f16_2x2_sm100: {msg}")


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
else:  # pragma: no cover - make_cfg_d512_2x2 rejects everything else
    raise ValueError(f"bprop_d512_f16_2x2_sm100: DTYPE_QKV={CFG.DTYPE_QKV} not supported (expected 2=BF16 or 3=FP16)")
# The workspace dtype IS the io dtype: stage 3 reads S/dS at input precision.
WORKSPACE_DTYPE = STORAGE_DTYPE
MMA_KIND = nvvm.Tcgen05MMAKind.F16

CGA_SIZE = CFG.CGA_M * CFG.CGA_N
# The scheduler payload barrier is completed by the CLUSTER LEAD's DSMEM st.async + complete_tx (scheduler_warp_loop_persistent):
# for the second pair of a 4-CTA cluster that is an event from outside its cta_group::2 pair, so every wait on it polls
# (the ring barriers' rule; the CI GB200 lane's time-slicing detector hung once with the parked form, 2026-10-06).
_SCHED_POLL = CGA_SIZE > CFG.CTA_MMA
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2
LOG2E = 1.4426950408889634

# ---------------------------------------------------------------------------
# Buffer geometry -- all of it derived from the config's own functions so the
# validator and the kernel can never disagree.
# ---------------------------------------------------------------------------

qBytes, doBytes, kBytesPerStage, vBytesPerStage = operand_bytes_2x2(CFG)
qBufferElems = qBytes // CFG.BPE
doBufferElems = doBytes // CFG.BPE
kBufferElems = kBytesPerStage // CFG.BPE
vBufferElems = vBytesPerStage // CFG.BPE
castElems = cast_bytes_2x2(CFG) // CFG.BPE

_require(CFG.CTA_MMA == 2 and CFG.CGA_M == 4, "the 2x2 datapath needs cluster (4,1,1) of two cta_group::2 pairs")
_require(CFG.TILE_K == CFG.TILE_O, "d_qk must equal d_v (Q and dO share one SMEM geometry, K and V one ring geometry)")

# cta_group::2 tensor TMA: ALL bytes landing in the pair route to the pair leader's mbar, so a transaction
# counts BOTH CTAs' buffers.  Q + dO per tile; one K chunk + one V chunk per ring stage.
opTmaTransactionBytes = op_tx_bytes_2x2(CFG)
ringTmaTransactionBytes = ring_tx_bytes_2x2(CFG)
_require(opTmaTransactionBytes == CFG.CTA_MMA * (qBytes + doBytes), "op expect_tx disagrees with the Q + dO slabs")
_require(ringTmaTransactionBytes == CFG.CTA_MMA * (kBytesPerStage + vBytesPerStage), "ring expect_tx disagrees with the K + V chunk slabs")

# TMA subtiling: inner byte extent / swizzle atom.  Q / dO: d = 512 as 8 boxes of 64 columns; a ring stage:
# D_CHUNK columns as D_CHUNK / 64 boxes.  All boxes are TILE_M (Q, dO) or TILE_N / CTA_MMA (K, V) rows = 64.
TMA_GRANU_ELEMS = CFG.Q_SWZ_BYTES // CFG.BPE  # 64
TMA_OP_ITERS = CFG.TILE_K // TMA_GRANU_ELEMS  # 8
TMA_RING_ITERS = CFG.D_CHUNK // TMA_GRANU_ELEMS  # 1 at D_CHUNK = 64
_require(TMA_OP_ITERS == CFG.N_CHUNKS * TMA_RING_ITERS, "Q / dO subtiles must tile exactly into the ring chunks")
# One Q / dO chunk (TILE_M rows x D_CHUNK columns) in BYTES = what the MMA descriptor advances per chunk.
A_CHUNK_BYTES = CFG.TILE_M * CFG.D_CHUNK * CFG.BPE
A_CHUNK_DESC_ADVANCE = A_CHUNK_BYTES >> 4
_require(A_CHUNK_BYTES % 1024 == 0, "a Q / dO chunk must be a whole number of 1024-B swizzle atoms")

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_B = {128: 3, 64: 2, 32: 1}
SMEM_LAYOUT_QK = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_S = _SWZ_ENUM[CFG.S_SWZ_BYTES]
S_SMEM_SWIZZLE = cutlass.Swizzle(_SWZ_B[CFG.S_SWZ_BYTES], 4, 3)

_CORE_MATRIX_ROWS = 8
LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = _CORE_MATRIX_ROWS * CFG.Q_SWZ_BYTES

# Workspace store: the 64 x 128 tile as two 64 x 64 subtiles (one per TMEM lane half), 128 B per row = one atom.
S_TMA_ITERS = cast_subtiles_2x2(CFG)
S_D_BLOCK = CFG.TILE_N // S_TMA_ITERS
S_SUBTILE_SLAB = CFG.TILE_M * S_D_BLOCK
_require(S_TMA_ITERS == 2 and S_D_BLOCK == acc_cols_2x2(CFG), "one stored subtile must be exactly one accumulator column half")
_require(castElems == S_TMA_ITERS * S_SUBTILE_SLAB, "cast element count disagrees with config cast_bytes_2x2()")

# tcgen05 SMEM-descriptor version, bound ONCE for every SmemTile (handles.py: a root at or past 256 KiB
# needs the extended format or the address wraps to an untouched buffer -- exactly-zero accumulator).
DESC_VERSION: int = desc_version_2x2(CFG)

# ---------------------------------------------------------------------------
# TMEM: 256 columns, no operand columns (fact: a 2SM M=128 TS operand is duplicated across the lane halves).
# ---------------------------------------------------------------------------


class KernelTmemLayout(NamedTuple):
    """[S_acc p0 | S_acc p1 | dS_acc p0 | dS_acc p1], ACC_COLS = TILE_N / 2 each under the 2x2 D image."""

    ACC_COLS: int = acc_cols_2x2(CFG)
    S_OFF: int = 0
    DS_OFF: int = CFG.STAGES_ACC * acc_cols_2x2(CFG)
    TOTAL_COLS: int = tmem_cols_2x2(CFG)


LAYOUT = KernelTmemLayout()
_require(LAYOUT.DS_OFF + CFG.STAGES_ACC * LAYOUT.ACC_COLS == LAYOUT.TOTAL_COLS, "TMEM carve must be exactly the two accumulator rings")
_require(LAYOUT.TOTAL_COLS == tmem_cols_2x2(CFG), "kernel TMEM carve disagrees with config tmem_cols_2x2()")
_require(LAYOUT.TOTAL_COLS & (LAYOUT.TOTAL_COLS - 1) == 0 and LAYOUT.TOTAL_COLS >= 32, "tcgen05.alloc needs a power-of-two column count >= 32")

# ---------------------------------------------------------------------------
# SMEM budget -- assert the COMPUTED total, never an inherited constant.
# ---------------------------------------------------------------------------

_SLABS = smem_layout_2x2(CFG)
_require(tuple(s.name for s in _SLABS) == ("sQ", "sdO", "sRingK", "sRingV", "sCastS", "sCastDS"), "slab order must be Q, dO, ring K, ring V, cast S, cast dS")
_require(
    qBytes + doBytes + CFG.STAGES_KV * (kBytesPerStage + vBytesPerStage) + 2 * CFG.CAST_STAGES * cast_bytes_2x2(CFG) == smem_bytes_2x2(CFG),
    "kernel slab sizing disagrees with config smem_bytes_2x2()",
)

# ---------------------------------------------------------------------------
# Barriers.  Each init_count is a named CFG constant; the comment beside it
# names the arrive sites it counts and the guard each fires under.
# ---------------------------------------------------------------------------


class Bars(NamedTuple):
    # The resident operands (Q and dO, one TMA transaction per tile).
    mb_tma_op_full: object
    mb_tma_op_empty: object
    # The K / V chunk ring, shared across the two pairs.
    mb_tma_ring_full: object
    mb_tma_ring_empty: object
    # MMA -> compute, and the accumulator release back to the MMA leader.
    mb_bmm_done: object
    mb_acc_empty: object
    # compute -> TMA-STG for the workspace store (S and dS of one tile per stage).
    mb_smem_full: object
    mb_smem_empty: object
    mb_tmem_dealloc: object


def _make_bwd_d512_2x2_bars(CFG) -> Bars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        # ONE_LANE: the pair-leader TMA-LDG warp's elected lane arms `expect_tx(opTmaTransactionBytes)` once per
        # tile (`pred=is_leader & elect_sync()`); both CTAs' self-only cta_group::2 Q / dO loads complete on it.
        mb_tma_op_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        # ONE_LANE: the leader MMA warp's elected lane, one tcgen05.commit (mask = pair) after the tile's last MMA.
        mb_tma_op_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        # The leader arms expect_tx on every chunk. In shared-KV mode the other three CTAs also acknowledge
        # observing their local empty phase, so neither MMA pair can release the next phase before every CTA waits.
        # The pair-leader TMA-LDG warp's elected lane arms `expect_tx(ringTmaTransactionBytes)` on EVERY
        # chunk stage, whether this pair or the partner pair issues the chunk -- the bytes complete where they LAND
        # (destination pair leader).  Followers never arm or wait their copy (that accounting hangs: probe mcast_twin).
        # WAIT FORM: under KV_SHARE 2 half of the completing bytes are the PARTNER pair's TMA complete_tx, so the MMA
        # warp POLLS this barrier (`_wait_b(..., poll=_KV_SHARED)`), never parks -- see `_poll_wait`.
        mb_tma_ring_full=MBarrier(
            _alloc(CFG.STAGES_KV),
            stages=CFG.STAGES_KV,
            init_count=CFG.CGA_M if _KV_SHARED else CFG.ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        # RING_EMPTY_ARRIVERS (= KV_SHARE = 2 pairs): CTA 0's AND CTA 2's leader MMA warps, one elected lane each, one
        # tcgen05.commit with mask 0xF after BMM1 + BMM2 of the chunk.  CTA c's multicast writes CTA c ^ 2's slot too, so a
        # stage may be refilled only once BOTH pairs have read it; every CTA's TMA-LDG warp waits its own copy.
        # WAIT FORM: one of the two arrives is the OTHER pair's commit -- the barrier that hung under GPU time-slicing
        # (heartbeat dump 2026-10-01: a follower's LDG warp parked in NANOSLEEP.SYNCS never woke for a release the other
        # three CTAs' copies had completed), so every wait on it POLLS (`poll=_KV_SHARED`, kv loop and drain alike).
        mb_tma_ring_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.RING_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        # ONE_LANE: the leader MMA warp's elected lane, one tcgen05.commit (mask = pair) after the tile's chunk 7 BMM2 --
        # one commit covers both accumulators.
        mb_bmm_done=MBarrier(_alloc(CFG.STAGES_ACC), stages=CFG.STAGES_ACC, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        # ACC_EMPTY_ARRIVERS: every lane of the compute WG on BOTH CTAs of the pair arrives on the pair leader, after its
        # tcgen05_wait(LOAD) of BOTH accumulator loads.
        mb_acc_empty=MBarrier(_alloc(CFG.STAGES_ACC), stages=CFG.STAGES_ACC, init_count=CFG.ACC_EMPTY_ARRIVERS, producer=Producer.LEADER, scope=Scope.LEADER),
        # COMPUTE_LANES: THREAD arrive from every lane of all four compute warps, once per kv tile, after the fence.
        mb_smem_full=MBarrier(_alloc(CFG.CAST_STAGES), stages=CFG.CAST_STAGES, init_count=CFG.COMPUTE_LANES, producer=Producer.THREAD),
        # ONE_WARP: THREAD arrive from the single TMA-STG warp, NOT elect-gated -> all 32 lanes fire after
        # tma_store_wait(0).  ONE_LANE here would under-count by 31 and hang.
        mb_smem_empty=MBarrier(_alloc(CFG.CAST_STAGES), stages=CFG.CAST_STAGES, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        # TMEM_DEALLOC_ARRIVERS (= CTA_MMA = 2): the compute lead warp's elected lane of THIS CTA (`.arrive()`) and of the
        # PEER CTA (`arrive_on_peer(cta_id_x ^ 1)`), once each at kernel end -- so neither CTA's MMA warp deallocates
        # TMEM its own compute warps may still be reading (the stage-3 GEMM's symmetric gate; the 4x1 gates on the peer
        # only, which the leader's acc_empty drain happens to cover and the follower's does not).
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CFG.TMEM_DEALLOC_ARRIVERS, producer=Producer.THREAD),
    )


# ---------------------------------------------------------------------------
# Warp bodies.
#
# K and V have IDENTICAL TMA geometry here ([TILE_N/CTA_MMA, D_CHUNK] at
# kv_row_base + cta_in_pair*(TILE_N/CTA_MMA), column 64*c), and so do Q and dO:
# both BMMs are A[q,d] x B[kv,d]^T.
# ---------------------------------------------------------------------------


@cute.jit
def _ldg_kv_tile(
    bars,
    sRingK,
    sRingV,
    tma_k,
    tma_v,
    desc_words,
    kv_head_g,
    kv_row,
    kv_row_thd,
    batch_g,
    is_leader,
    pair_id,
    tma_mcast_mask_kv,
    ring_state,
    dbg,
    kv_loop,
    q_block,
    tile_no,
    ring_total,
):
    """One kv tile of the TMA-LDG warp: N_CHUNKS chunk stages of the shared K / V ring.

    Every CTA waits its own ``ring_empty`` copy on every chunk (the slot it is about to RECEIVE into, from itself
    or from the partner pair); the pair leader arms ``ring_full`` on every chunk; under KV_SHARE 2 the pair whose
    parity matches the chunk issues it to ``{c, c ^ 2}``, under KV_SHARE 1 every pair issues every chunk to itself.
    Returns the advanced ring state.
    """
    for c in cutlass.range_constexpr(CFG.N_CHUNKS):
        _wait_b(
            bars.mb_tma_ring_empty[ring_state.idx].smem_ptr,
            ring_state.phase,
            dbg,
            DBG_BAR_RING_EMPTY,
            ring_state.idx,
            kv_loop,
            cutlass.Int32(c),
            q_block,
            tile_no,
            ring_total,
            poll=_KV_SHARED,
        )
        if cutlass.const_expr(_KV_SHARED):
            # All four empty copies must be observed before either pair can release this slot again.
            # The local leader's expect_tx below supplies its own acknowledgment; everyone else
            # acknowledges both leader copies. A passive follower must gate reuse even on chunks
            # it does not load, or it can miss two phase transitions while descheduled.
            ack_lane = nvvm.elect_sync()
            bars.mb_tma_ring_full[ring_state.idx].arrive_on_peer(cutlass.Int32(0), pred=((is_leader == False) | (pair_id != cutlass.Int32(0))) & ack_lane)
            bars.mb_tma_ring_full[ring_state.idx].arrive_on_peer(
                cutlass.Int32(CFG.CTA_MMA), pred=((is_leader == False) | (pair_id != cutlass.Int32(1))) & ack_lane
            )
        if cutlass.const_expr(_DBG_CLK):
            t_issue = cute.arch.clock64()
        bars.mb_tma_ring_full[ring_state.idx].arrive(n_bytes=ringTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        # A Python bool under KV_SHARE 1: the guard folds away and every pair issues every chunk.
        issue = (pair_id == cutlass.Int32(c & 1)) if cutlass.const_expr(_KV_SHARED) else True
        if issue:
            d_col = cutlass.Int32(c * CFG.D_CHUNK)
            tma_load_tile(
                sRingK[ring_state.idx],
                (
                    tma_slice_runtime_desc(_rt_desc(desc_words, cutlass.Int32(K_SLOT)), d_col, kv_head_g, kv_row_thd, cutlass.Int32(0))
                    if cutlass.const_expr(_THD)
                    else tma_k(d_col, kv_head_g, kv_row, batch_g)
                ),
                bars.mb_tma_ring_full[ring_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask_kv,
                acquire=False,
            )
            tma_load_tile(
                sRingV[ring_state.idx],
                (
                    tma_slice_runtime_desc(_rt_desc(desc_words, cutlass.Int32(V_SLOT)), d_col, kv_head_g, kv_row_thd, cutlass.Int32(0))
                    if cutlass.const_expr(_THD)
                    else tma_v(d_col, kv_head_g, kv_row, batch_g)
                ),
                bars.mb_tma_ring_full[ring_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask_kv,
                acquire=False,
            )
        if cutlass.const_expr(_DBG_CLK):
            _clk_add(dbg, DBG_CLK_SEG_LDG_ISSUE, cute.arch.clock64() - t_issue)
        ring_state = advance(ring_state, CFG.STAGES_KV)
    return ring_state


@cute.jit
def _tmaldg_warp_group(
    bars,
    sched,
    sQ,
    sdO,
    sRingK,
    sRingV,
    tma_q,
    tma_k,
    tma_v,
    tma_do,
    meta_t,
    desc_words,
    n_batch,
    n_qh,
    cta_id_x,
    cta_in_pair,
    is_leader,
    pair_id,
    tma_mcast_mask_qdo,
    tma_mcast_mask_kv,
    n_kv,
    seqlen_q,
    seqlen_kv,
    gqa_ratio,
    head_base,
    batch_base,
    dbg,
):
    q_block, head_idx, batch_idx, q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv = _decode_initial(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
    )
    if cutlass.const_expr(_THD):
        # ONE acquire per descriptor per CTA, not one per tile.
        tma_tensormap_acquire(_rt_desc(desc_words, cutlass.Int32(Q_SLOT)))
        tma_tensormap_acquire(_rt_desc(desc_words, cutlass.Int32(DO_SLOT)))
        tma_tensormap_acquire(_rt_desc(desc_words, cutlass.Int32(K_SLOT)))
        tma_tensormap_acquire(_rt_desc(desc_words, cutlass.Int32(V_SLOT)))
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    # Pre-armed: nothing has consumed the operand or the ring yet, so the first
    # waits must fall through with no producer arrive.
    op_empty_state = PipelineState.start(phase=1)
    ring_state = PipelineState.start(phase=1)
    # Chunk stages issued over EVERY tile -- the end-of-kernel drain is min(total, STAGES_KV).
    ring_total = cutlass.Int32(0)
    tile_no = cutlass.Int32(0)

    # Per-CTA slice of the streamed operand along the collective MMA-N axis.
    RING_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row_base = cute.arch.make_warp_uniform(q_block * cutlass.Int32(CFG.TILE_M))
        head_g = head_idx + head_base
        batch_g = batch_idx + batch_base
        # GQA / MQA: Q and dO are per Q-head; K and V live at the shared KV head.
        kv_head_g = head_g // gqa_ratio

        # --- the resident operands: Q AND dO, one transaction ----------------
        _wait_b(
            bars.mb_tma_op_empty.smem_ptr,
            op_empty_state.phase,
            dbg,
            DBG_BAR_OP_EMPTY,
            cutlass.Int32(0),
            cutlass.Int32(DBG_NONE),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            ring_total,
        )
        op_empty_state = advance(op_empty_state, 1)
        bars.mb_tma_op_full.arrive(n_bytes=opTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        # THD packs the batch away: ONE descriptor over [1, T, H, D], the
        # sequence reached by adding cu_q[b] to the SEQUENCE coord with the
        # batch coord pinned to 0.
        op_row_thd = q_tok + q_row_base
        tma_load_tile(
            sQ[0],
            (
                tma_slice_runtime_desc(_rt_desc(desc_words, cutlass.Int32(Q_SLOT)), cutlass.Int32(0), head_g, op_row_thd, cutlass.Int32(0))
                if cutlass.const_expr(_THD)
                else tma_q(cutlass.Int32(0), head_g, q_row_base, batch_g)
            ),
            bars.mb_tma_op_full.smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask_qdo,
            acquire=False,
        )
        tma_load_tile(
            sdO[0],
            (
                tma_slice_runtime_desc(_rt_desc(desc_words, cutlass.Int32(DO_SLOT)), cutlass.Int32(0), head_g, op_row_thd, cutlass.Int32(0))
                if cutlass.const_expr(_THD)
                else tma_do(cutlass.Int32(0), head_g, q_row_base, batch_g)
            ),
            bars.mb_tma_op_full.smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask_qdo,
            acquire=False,
        )

        # --- the shared K / V chunk ring -------------------------------------
        # Same bounds on all four CTAs (see _kv_tile_bounds): the tiles skipped
        # here are never loaded, never MMA'd and never stored.
        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_id_x, seqlen_q, seqlen_kv, n_kv)
        for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
            kv_row = kv_loop * cutlass.Int32(CFG.TILE_N) + RING_ROW_OFFSET_PEER
            kv_row_thd = kv_tok + kv_row
            ring_state = _ldg_kv_tile(
                bars,
                sRingK,
                sRingV,
                tma_k,
                tma_v,
                desc_words,
                kv_head_g,
                kv_row,
                kv_row_thd,
                batch_g,
                is_leader,
                pair_id,
                tma_mcast_mask_kv,
                ring_state,
                dbg,
                kv_loop,
                q_block,
                tile_no,
                ring_total,
            )
        ring_total = ring_total + (kv_right - kv_left) * cutlass.Int32(CFG.N_CHUNKS)

        # --- next tile ----------------------------------------------------
        _wait_b(
            sched.mb_scheduler.subview(sched_state.idx),
            sched_state.phase,
            dbg,
            DBG_BAR_SCHED,
            sched_state.idx,
            cutlass.Int32(DBG_NONE),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            ring_total,
            poll=_SCHED_POLL,
        )
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_block, head_idx, batch_idx, q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv = _decode_payload(
            nxt_q, nxt_hb, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        tile_no = tile_no + cutlass.Int32(1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    # --- cross-CTA drains, OUTSIDE the persistent loop --------------------
    # Both rings are fired by cross-CTA MMA_COMMITs (the ring one from BOTH pair
    # leaders), so the last commits are still in flight when this warp would
    # exit.  Staying resident until they land is what stops the teardown fault.
    # The drain walks the WHOLE ring from the current state (STAGES_KV waits, static): a slot never used is waited at
    # its pre-armed parity and passes at once; a used slot is waited at the parity its LAST release completes.  A drain
    # of min(total, stages) steps starts at the current index, i.e. at the slots used LEAST recently -- with fewer
    # chunks than stages it waited untouched slots and skipped the used ones (the TMEM twin of this bug, review P1).
    for _ in cutlass.range_constexpr(CFG.STAGES_KV):
        _wait_b(
            bars.mb_tma_ring_empty[ring_state.idx].smem_ptr,
            ring_state.phase,
            dbg,
            DBG_BAR_RING_EMPTY,
            ring_state.idx,
            cutlass.Int32(DBG_DRAIN),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            ring_total,
            poll=_KV_SHARED,
        )
        ring_state = advance(ring_state, CFG.STAGES_KV)
    # One operand load per tile, so exactly one residual arrive.
    _wait_b(
        bars.mb_tma_op_empty.smem_ptr,
        op_empty_state.phase,
        dbg,
        DBG_BAR_OP_EMPTY,
        cutlass.Int32(0),
        cutlass.Int32(DBG_DRAIN),
        cutlass.Int32(DBG_NONE),
        q_block,
        tile_no,
        ring_total,
    )
    op_empty_state = advance(op_empty_state, 1)
    _dbg_exit(dbg, tile_no, ring_total)


@cute.jit
def _mma_kv_tile(bars, bmm_desc, desc_q, desc_do, sRingK, sRingV, tmem_s, tmem_ds, ring_empty_mcast, ring_state, dbg, kv_loop, q_block, tile_no, acc_total):
    """One kv tile of the leader MMA warp: N_CHUNKS x (BMM1 chunk, BMM2 chunk, ring release).

    Chunk ``c`` of Q (dO) is SW128 subtile ``c`` of the resident slab -- the descriptor advances by
    ``A_CHUNK_DESC_ADVANCE`` -- against ring stage ``s``; ``accumulate=(c > 0)`` chains the eight K = 64
    chunks into one fp32 accumulator (bitwise identical to one K = 512 chain: probe ss_slabs S3).
    Returns the advanced ring state.
    """
    for c in cutlass.range_constexpr(CFG.N_CHUNKS):
        _wait_b(
            bars.mb_tma_ring_full[ring_state.idx].smem_ptr,
            ring_state.phase,
            dbg,
            DBG_BAR_RING_FULL,
            ring_state.idx,
            kv_loop,
            cutlass.Int32(c),
            q_block,
            tile_no,
            acc_total,
            poll=_KV_SHARED,
        )
        if cutlass.const_expr(_DBG_CLK):
            t_issue = cute.arch.clock64()
        desc_k = sRingK[ring_state.idx].desc()
        desc_v = sRingV[ring_state.idx].desc()
        mma_ss(bmm_desc, desc_q + c * A_CHUNK_DESC_ADVANCE, desc_k, tmem_s, accumulate=(c > 0), elect_once=True)
        mma_ss(bmm_desc, desc_do + c * A_CHUNK_DESC_ADVANCE, desc_v, tmem_ds, accumulate=(c > 0), elect_once=True)
        # ONE lane commits (commit_mma does not elect internally); mask 0xF: this
        # chunk's slot is read by this pair's MMA only, but the partner pair's
        # copy of ring_empty counts OUR release too (init 2 = both pairs).
        elect_p = nvvm.elect_sync()
        bars.mb_tma_ring_empty[ring_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=ring_empty_mcast, pred=elect_p)
        if cutlass.const_expr(_DBG_CLK):
            _clk_add(dbg, DBG_CLK_SEG_MMA_ISSUE, cute.arch.clock64() - t_issue)
        ring_state = advance(ring_state, CFG.STAGES_KV)
    return ring_state


@cute.jit
def _mma_warp_leader(
    bars, sched, sQ, sdO, sRingK, sRingV, tmem_ptr_i32, meta_t, n_batch, n_qh, cta_id_x, pair_mask, ring_empty_mcast, n_kv, seqlen_q, seqlen_kv, dbg
):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    # Publish the TMEM base to this CTA's compute WG (pairs with barrier_cta_sync 1 there).
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    # Both BMMs are the same MMA: A[TILE_M, D_CHUNK] from SMEM x B[TILE_N/CTA_MMA, D_CHUNK]^T from SMEM,
    # collective M = TILE_M * CTA_MMA = 128 (the 2x2 datapath), K = D_CHUNK per call, 16 per instruction.
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
        K=CFG.D_CHUNK,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        atranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc,
        kind=MMA_KIND,
    )

    q_block, _head, _batch, _q_tok, _kv_tok, _ws_row, seqlen_q, seqlen_kv = _decode_initial(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    op_full_state = PipelineState.start()
    ring_state = PipelineState.start()
    acc_state = PipelineState.start(phase=1)
    acc_total = cutlass.Int32(0)
    tile_no = cutlass.Int32(0)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        _wait_b(
            bars.mb_tma_op_full.smem_ptr,
            op_full_state.phase,
            dbg,
            DBG_BAR_OP_FULL,
            cutlass.Int32(0),
            cutlass.Int32(DBG_NONE),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            acc_total,
        )
        op_full_state = advance(op_full_state, 1)

        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_id_x, seqlen_q, seqlen_kv, n_kv)
        for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
            _wait_b(
                bars.mb_acc_empty[acc_state.idx].smem_ptr,
                acc_state.phase,
                dbg,
                DBG_BAR_ACC_EMPTY,
                acc_state.idx,
                kv_loop,
                cutlass.Int32(DBG_NONE),
                q_block,
                tile_no,
                acc_total,
            )
            # Descriptor bases pinned INSIDE the kv loop (desc_opaque + the loop counter): otherwise LICM hoists
            # the 16 per-chunk descriptor adds to the persistent loop's preheader and the 40-register warp spills.
            desc_q = desc_opaque(sQ[0].desc(), anchor=kv_loop)
            desc_do = desc_opaque(sdO[0].desc(), anchor=kv_loop)
            tmem_s = tmem_raw.subview(cutlass.Int32(LAYOUT.S_OFF) + acc_state.idx * cutlass.Int32(LAYOUT.ACC_COLS))
            tmem_ds = tmem_raw.subview(cutlass.Int32(LAYOUT.DS_OFF) + acc_state.idx * cutlass.Int32(LAYOUT.ACC_COLS))
            ring_state = _mma_kv_tile(
                bars, bmm_desc, desc_q, desc_do, sRingK, sRingV, tmem_s, tmem_ds, ring_empty_mcast, ring_state, dbg, kv_loop, q_block, tile_no, acc_total
            )
            # One commit covers both accumulators (the tcgen05 pipeline is in order).
            elect_p = nvvm.elect_sync()
            bars.mb_bmm_done[acc_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=pair_mask, pred=elect_p)
            acc_state = advance(acc_state, CFG.STAGES_ACC)
        acc_total = acc_total + (kv_right - kv_left)

        bars.mb_tma_op_empty.arrive(cta_group=CFG.CTA_MMA, mcast_mask=pair_mask, pred=nvvm.elect_sync())

        _wait_b(
            sched.mb_scheduler.subview(sched_state.idx),
            sched_state.phase,
            dbg,
            DBG_BAR_SCHED,
            sched_state.idx,
            cutlass.Int32(DBG_NONE),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            acc_total,
            poll=_SCHED_POLL,
        )
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_block, _head, _batch, _q_tok, _kv_tok, _ws_row, seqlen_q, seqlen_kv = _decode_payload(
            nxt_q, nxt_hb, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        tile_no = tile_no + cutlass.Int32(1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    # TMEM lifetime, the producer side: every compute lane's tcgen05.ld of a slot precedes its mb_acc_empty arrive
    # (tcgen05.wait::ld first), so waiting every USED slot at its last-release parity covers every read of this leader's
    # accumulators (the compute groups' exit barrier before their mb_tmem_dealloc arrive covers the follower CTA's, which
    # drains nothing).  Walk the whole ring (STAGES_ACC static waits): unused
    # slots pass at their pre-armed parity, used slots wait for all ACC_EMPTY_ARRIVERS lanes.  min(acc_total, STAGES_ACC)
    # steps from the current index waited the UNUSED slots when a cluster ran fewer kv tiles than stages (one kv tile at
    # STAGES_ACC = 2: slot 1 at parity 1 passed free, slot 0's release was never awaited): review P1, pinned by
    # test_sdpa_bwd_dsl_sm100.py::test_ring_drain_walks_every_used_slot (host twin) and the single-kv-tile GPU cells.
    for _ in cutlass.range_constexpr(CFG.STAGES_ACC):
        _wait_b(
            bars.mb_acc_empty[acc_state.idx].smem_ptr,
            acc_state.phase,
            dbg,
            DBG_BAR_ACC_EMPTY,
            acc_state.idx,
            cutlass.Int32(DBG_DRAIN),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            acc_total,
        )
        acc_state = advance(acc_state, CFG.STAGES_ACC)

    _wait_b(
        bars.mb_tmem_dealloc.smem_ptr,
        cutlass.Int32(0),
        dbg,
        DBG_BAR_TMEM_DEALLOC,
        cutlass.Int32(0),
        cutlass.Int32(DBG_DRAIN),
        cutlass.Int32(DBG_NONE),
        q_block,
        tile_no,
        acc_total,
    )
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    _dbg_exit(dbg, tile_no, acc_total)


@cute.jit
def _mma_warp_non_leader(bars, sched, tmem_ptr_i32, meta_t, n_batch, n_qh, cta_id_x, seqlen_q, seqlen_kv, dbg):
    """Quiet -- but NOT an empty body: it owns this CTA's TMEM allocation and the base publish that releases
    its own compute warp group, and it stays in the persistent loop so READ_TILE_ARRIVERS holds."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    _q_block, _head, _batch, _q_tok, _kv_tok, _ws_row, _s_q, _s_kv = _decode_initial(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    tile_no = cutlass.Int32(0)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        _wait_b(
            sched.mb_scheduler.subview(sched_state.idx),
            sched_state.phase,
            dbg,
            DBG_BAR_SCHED,
            sched_state.idx,
            cutlass.Int32(DBG_NONE),
            cutlass.Int32(DBG_NONE),
            _q_block,
            tile_no,
            cutlass.Int32(0),
            poll=_SCHED_POLL,
        )
        _nq, _nh, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        tile_no = tile_no + cutlass.Int32(1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    _wait_b(
        bars.mb_tmem_dealloc.smem_ptr,
        cutlass.Int32(0),
        dbg,
        DBG_BAR_TMEM_DEALLOC,
        cutlass.Int32(0),
        cutlass.Int32(DBG_DRAIN),
        cutlass.Int32(DBG_NONE),
        _q_block,
        tile_no,
        cutlass.Int32(0),
    )
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    _dbg_exit(dbg, tile_no, cutlass.Int32(0))


@cute.jit
def _tmastg_warp_group(bars, sched, sCastS, sCastDS, tma_s, tma_ds, meta_t, n_batch, n_qh, cta_id_x, n_kv, seqlen_q, seqlen_kv, head_base, batch_base, dbg):
    q_block, head_idx, batch_idx, q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv = _decode_initial(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    smem_state = PipelineState.start()
    tile_no = cutlass.Int32(0)
    stg_total = cutlass.Int32(0)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row_base = q_block * cutlass.Int32(CFG.TILE_M)
        # The workspace is CHUNK-LOCAL: head_base / batch_base offset every full-tensor access but NOT this store.
        head_ws = head_idx
        # THD collapses the workspace's batch axis: [1, H_chunk, R, N], the sequence reached by the ROW offset.
        # A TMA store that is OOB is DROPPED silently, so the batch coord must be 0 here.
        batch_ws = cutlass.Int32(0) if cutlass.const_expr(_THD) else batch_idx
        ws_row_base = ws_row + q_row_base
        # THD: this CTA's 64-row box may lie past the sequence's 128-row block entirely; 64 | 128, so a box is
        # wholly inside or wholly outside and the skip is this boolean.  Only the STORE is skipped: the
        # smem_full waits and smem_empty arrives still run because the compute warps produced into the slab.
        ws_in_block = (
            q_row_base < ((seqlen_q + cutlass.Int32(CFG.WS_BLOCK_ROWS - 1)) // cutlass.Int32(CFG.WS_BLOCK_ROWS)) * cutlass.Int32(CFG.WS_BLOCK_ROWS)
            if cutlass.const_expr(_THD)
            # A PYTHON bool: the guard below folds at trace time and the dense build emits no branch at all.
            else True
        )

        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_id_x, seqlen_q, seqlen_kv, n_kv)
        for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
            # kv is the ABSOLUTE tile index -> the right workspace column once masks make kv_left > 0.  In ELEMENTS.
            kv_col = kv_loop * cutlass.Int32(CFG.TILE_N)
            _wait_stg(bars, smem_state, dbg, kv_loop, q_block, tile_no, stg_total)
            if ws_in_block:
                if cutlass.const_expr(_DBG_CLK):
                    t_store = cute.arch.clock64()
                # ONE call per tile per buffer: tma_store_tile walks both 64-column subtiles itself.
                tma_store_tile(sCastS[smem_state.idx], tma_s(kv_col, ws_row_base, head_ws, batch_ws))
                tma_store_tile(sCastDS[smem_state.idx], tma_ds(kv_col, ws_row_base, head_ws, batch_ws))
                tma_store_commit()
                tma_store_wait(0)
                if cutlass.const_expr(_DBG_CLK):
                    _clk_add(dbg, DBG_CLK_SEG_STG_STORE, cute.arch.clock64() - t_store)
            # Plain THREAD arrive: all 32 lanes fire, which is what init_count = ONE_WARP counts.
            bars.mb_smem_empty[smem_state.idx].arrive()
            smem_state = advance(smem_state, CFG.CAST_STAGES)
        stg_total = stg_total + (kv_right - kv_left)

        _wait_b(
            sched.mb_scheduler.subview(sched_state.idx),
            sched_state.phase,
            dbg,
            DBG_BAR_SCHED,
            sched_state.idx,
            cutlass.Int32(DBG_NONE),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            stg_total,
            poll=_SCHED_POLL,
        )
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_block, head_idx, batch_idx, q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv = _decode_payload(
            nxt_q, nxt_hb, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        tile_no = tile_no + cutlass.Int32(1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    _dbg_exit(dbg, tile_no, stg_total)


@cute.jit
def _compute_kv_iter(
    bars,
    sCastS_raw,
    sCastDS_raw,
    tmem_base_addr,
    cast_off,
    q_row,
    col_half,
    leader_cta_id,
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
    dbg,
    q_block,
    tile_no,
    apply_mask: cutlass.Constexpr[bool],
):
    """One kv tile of the compute warp group: TMEM -> S and dS in registers -> SMEM cast slabs.

    ``apply_mask`` is a PYTHON constant, so this traces twice -- once without any mask code for the
    tiles strictly inside the band, once with it for the band-edge tiles.  Pipeline states go in and
    come back out; the caller rebinds them.
    """
    _wait_b(
        bars.mb_bmm_done[acc_state.idx].smem_ptr,
        acc_state.phase,
        dbg,
        DBG_BAR_BMM_DONE,
        acc_state.idx,
        kv_loop,
        smem_state.idx,
        q_block,
        tile_no,
        cutlass.Int32(0),
    )
    if cutlass.const_expr(_DBG_CLK):
        t_math = cute.arch.clock64()
    acc_col = acc_state.idx * cutlass.Int32(LAYOUT.ACC_COLS)

    # The lane's 64 S_acc and 64 dS_acc values: the SAME (row, kv column) set in the same lane for both
    # accumulators (the 2x2 D image), which is what makes dS = f(S) a pure register product.
    reg_s = tmem_load_tile(tmem_base_addr + cutlass.Int32(LAYOUT.S_OFF) + acc_col, num_elems=LAYOUT.ACC_COLS)
    reg_ds = tmem_load_tile(tmem_base_addr + cutlass.Int32(LAYOUT.DS_OFF) + acc_col, num_elems=LAYOUT.ACC_COLS)
    # Commit BOTH TMEM reads BEFORE releasing the accumulators, or the leader's next MMA overwrites them mid-flight.
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    bars.mb_acc_empty[acc_state.idx].arrive(cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id)

    # S = exp2(attn_scale_log2e * S_acc - lse * log2e), fp32.
    s_post = cute.math.exp2(reg_s.vec * attn_scale_log2e - lse_q_log2e, fastmath=True)

    # Mask AFTER exp2, to 0.0 -- not to -inf before it.  With LSE supplied there is no running max to
    # protect, and dS is a PRODUCT with S, so zeroing S zeroes dS for free.  `q_row` is this lane's own
    # row; `col_half` selects which 64 of the tile's 128 kv columns this warp holds.
    if cutlass.const_expr(apply_mask):
        s_post = apply_mask_chunk(
            s_post,
            q_row,
            kv_loop * cutlass.Int32(CFG.TILE_N) + col_half * cutlass.Int32(LAYOUT.ACC_COLS),
            seqlen_kv,
            CFG.WINDOW_LEFT,
            CFG.MASK_FLAGS,
            N=LAYOUT.ACC_COLS,
            bottom_right=1 if CFG.BOTTOM_RIGHT else 0,
            causal_diag=(seqlen_kv - seqlen_q) if CFG.BOTTOM_RIGHT else None,
            mask_value=0.0,
            window_right=CFG.WINDOW_RIGHT,
        )
    # Rows past the real S_q: zero the whole row (their LSE came from a CLAMPED index).  OUTSIDE the
    # apply_mask branch on purpose -- a padded q row is invalid on EVERY kv tile.
    if cutlass.const_expr(_PADDED):
        s_post = s_post * row_scale

    # dS = (attn_scale_for_dS * dS_acc - do_dot * attn_scale) * S, with S at fp32 in the same lane.
    ds_post = (reg_ds.vec * attn_scale_for_dS - scaled_do_dot_q) * s_post

    # The workspace store staging: subtile `col_half` (64 rows x 64 cols, 8 KiB), row `r` = one 128 B atom.
    # Deferred wait: the math above overlaps the previous TMA-STG drain.
    if cutlass.const_expr(_DBG_CLK):
        _clk_add(dbg, DBG_CLK_SEG_CMP_MATH, cute.arch.clock64() - t_math)
    _wait_b(
        bars.mb_smem_empty[smem_state.idx].smem_ptr,
        smem_state.phase,
        dbg,
        DBG_BAR_SMEM_EMPTY,
        smem_state.idx,
        kv_loop,
        acc_state.idx,
        q_block,
        tile_no,
        cutlass.Int32(0),
    )
    if cutlass.const_expr(_DBG_CLK):
        t_cast = cute.arch.clock64()
    slab_off = smem_state.idx * cutlass.Int32(castElems) + cast_off
    sCastS_raw.subview(slab_off).data_ptr().store_swizzled(s_post.to(WORKSPACE_DTYPE), alignment=64, swizzle=S_SMEM_SWIZZLE)
    sCastDS_raw.subview(slab_off).data_ptr().store_swizzled(ds_post.to(WORKSPACE_DTYPE), alignment=64, swizzle=S_SMEM_SWIZZLE)
    # Generic stores -> async-proxy (TMA store) visibility, then ONE arrive per lane (COMPUTE_LANES).
    nvvm.fence_proxy("async.shared", space="cta")
    bars.mb_smem_full[smem_state.idx].arrive()
    if cutlass.const_expr(_DBG_CLK):
        _clk_add(dbg, DBG_CLK_SEG_CMP_CAST, cute.arch.clock64() - t_cast)

    smem_state = advance(smem_state, CFG.CAST_STAGES)
    acc_state = advance(acc_state, CFG.STAGES_ACC)
    return acc_state, smem_state


@cute.jit
def _compute_warp_group(
    bars,
    sched,
    sCastS_raw,
    sCastDS_raw,
    tmem_ptr_i32,
    lse_tensor,
    do_dot_tensor,
    meta_t,
    n_batch,
    n_qh,
    cta_id_x,
    leader_cta_id,
    n_kv,
    seqlen_q,
    seqlen_kv,
    attn_scale_in,
    attn_scale_log2e,
    attn_scale_for_dS,
    head_base,
    batch_base,
    dbg,
):
    """S = exp2(scale*S_acc - lse), dS = (scale*dS_acc - do_dot) * S, both from the lane's own registers.

    Thread t of the 128-lane WG is TMEM lane t: row ``r = t & 63`` of kv-column half ``h = t >> 6`` (warps w
    and w + 2 hold the SAME rows and never need each other's values -- there is no row reduction here).
    """
    # Must precede the TMEM base read: pairs with the MMA warp's barrier_cta_arrive right after tmem_alloc.
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base_addr = tmem_ptr_i32.load()

    tid_in_wg = cute.arch.thread_idx()[0]
    is_lead_warp = (tid_in_wg // cutlass.Int32(32)) == cutlass.Int32(0)
    row_in_cta = tid_in_wg & cutlass.Int32(CFG.TILE_M - 1)
    col_half = cute.arch.make_warp_uniform(tid_in_wg // cutlass.Int32(CFG.TILE_M))
    # Subtile-major cast offset (ELEMENTS): subtile h = the lane half, then one 64-element row per lane.
    cast_off = col_half * cutlass.Int32(S_SUBTILE_SLAB) + row_in_cta * cutlass.Int32(S_D_BLOCK)

    q_block, head_idx, batch_idx, q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv = _decode_initial(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    acc_state = PipelineState.start()
    smem_state = PipelineState.start(phase=1)
    tile_no = cutlass.Int32(0)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row = q_block * cutlass.Int32(CFG.TILE_M) + row_in_cta
        head_g = head_idx + head_base
        batch_g = batch_idx + batch_base

        # Padding, q side: clamp the INDEX (lse / do_dot are at the REAL length) and carry a 0/1 row factor.
        # Clamped from BELOW as well: a THD dead unit decodes a NEGATIVE length.
        if cutlass.const_expr(_PADDED):
            q_row_safe = cute.math.max(cutlass.Int32(0), cute.math.min(q_row, seqlen_q - cutlass.Int32(1)))
            row_scale = cutlass.Float32(arith.select((q_row < seqlen_q).ir_value(), cutlass.Float32(1.0).ir_value(), cutlass.Float32(0.0).ir_value()))
        else:
            q_row_safe = q_row
            row_scale = cutlass.Float32(1.0)

        # Per-q-row scalars, hoisted once per tile.  Both arrive RAW: the host folds no log2e into the LSE
        # and no attn_scale into the dot.  Warps w and w + 2 load the same row's two scalars (two redundant
        # 4-B loads per row per tile -- the whole "exchange" of the 2x2 datapath here).
        if cutlass.const_expr(_THD):
            _row_pk = q_tok + q_row_safe
            if cutlass.const_expr(len(lse_tensor.shape) == 2):
                lse_q_log2e = lse_tensor[_row_pk, head_g] * cutlass.Float32(LOG2E)
            else:
                lse_q_log2e = lse_tensor[cutlass.Int32(0), head_g, _row_pk] * cutlass.Float32(LOG2E)
            scaled_do_dot_q = do_dot_tensor[cutlass.Int32(0), head_g, _row_pk] * attn_scale_in
        else:
            lse_q_log2e = lse_tensor[batch_g, head_g, q_row_safe] * cutlass.Float32(LOG2E)
            scaled_do_dot_q = do_dot_tensor[batch_g, head_g, q_row_safe] * attn_scale_in

        kv_left, kv_unmasked_lo, kv_unmasked_hi, kv_right = _kv_tile_bounds(q_block, cta_id_x, seqlen_q, seqlen_kv, n_kv)

        def _run(lo, hi, apply_mask, acc_state, smem_state):
            for _kv_loop in cutlass.range(lo, hi, 1, unroll=1):
                acc_state, smem_state = _compute_kv_iter(
                    bars,
                    sCastS_raw,
                    sCastDS_raw,
                    tmem_base_addr,
                    cast_off,
                    q_row,
                    col_half,
                    leader_cta_id,
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
                    dbg,
                    q_block,
                    tile_no,
                    apply_mask=apply_mask,
                )
            return acc_state, smem_state

        # THREE ranges, in kv order: the band's LOW edge, the interior, the HIGH edge.  Only the interior is
        # provably free of masked cells (SWA makes the low edge real).  `apply_mask` is a Python constant,
        # so each range traces separately and the interior contains no mask code at all.
        if cutlass.const_expr(_MASKED):
            acc_state, smem_state = _run(kv_left, kv_unmasked_lo, True, acc_state, smem_state)
        acc_state, smem_state = _run(kv_unmasked_lo, kv_unmasked_hi, False, acc_state, smem_state)
        if cutlass.const_expr(_MASKED):
            acc_state, smem_state = _run(kv_unmasked_hi, kv_right, True, acc_state, smem_state)

        _wait_b(
            sched.mb_scheduler.subview(sched_state.idx),
            sched_state.phase,
            dbg,
            DBG_BAR_SCHED,
            sched_state.idx,
            cutlass.Int32(DBG_NONE),
            cutlass.Int32(DBG_NONE),
            q_block,
            tile_no,
            cutlass.Int32(0),
            poll=_SCHED_POLL,
        )
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_block, head_idx, batch_idx, q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv = _decode_payload(
            nxt_q, nxt_hb, cta_id_x, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        tile_no = tile_no + cutlass.Int32(1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    # TMEM lifetime, the consumer side: every compute warp of this CTA has completed its last tcgen05.ld (each iteration
    # waits tcgen05.wait::ld before its mb_acc_empty arrive) once the whole group passes named barrier 2 -- only then does
    # the lead warp's elected lane release the MMA warps to dealloc: one arrive on THIS CTA's barrier and one on the PAIR
    # partner's (^1), so each CTA (the follower MMA warp included, which never drains the accumulator ring) deallocates
    # only after BOTH compute warp groups have finished reading (TMEM_DEALLOC_ARRIVERS).  Barrier 1 is the TMEM-base
    # publish, barrier 0 the kernel-start sync; 2 is this group's exit sync (review P1 on #1323).
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * CFG.SOFTMAX_WG_WARPS)
    if is_lead_warp:
        if nvvm.elect_sync():
            bars.mb_tmem_dealloc.arrive()
            bars.mb_tmem_dealloc.arrive_on_peer(cta_id_x ^ cutlass.Int32(1))
    _dbg_exit(dbg, tile_no, cutlass.Int32(0))


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
    # Plain per-lane GMEM reads, one q-row per lane -- NOT TMA.  Both arrive RAW.
    lse_tensor: cute.Tensor,
    do_dot_tensor: cute.Tensor,
    # Dense: unused (a 1-element dummy).  THD: the metadata buffer the setup launch wrote.
    seq_kv_lens_tensor: cute.Tensor,
    # Dense: a 1-element dummy.  THD: the setup launch's descriptor array.
    desc_words: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    gqa_ratio: cutlass.Int32,
    n_kv: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    attn_scale_in: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    attn_scale_for_dS: cutlass.Float32,
    # Host-loop coordinates: offset every FULL-TENSOR access; the chunk-local S/dS workspace stays at origin.
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- cluster identity ------------------------------------------------
    cta_id_x = cute.arch.block_idx_in_cluster()
    cta_in_pair = cta_id_x & 1
    leader_cta_id = cta_id_x & ~1
    pair_id = cta_id_x >> 1
    pair_mask = cutlass.Int32(3) << leader_cta_id  # 0x3 on pair 0, 0xC on pair 1: commits within the pair
    tma_mcast_mask_qdo = cutlass.Int16(1) << cta_id_x  # SELF-ONLY; bytes still route to the pair leader
    if cutlass.const_expr(_KV_SHARED):
        # 0xF, derived at RUNTIME (never a Python constant into a commit's mask operand -- the ptxas
        # "Arguments mismatch" hazard): the ring release lands on all four CTAs' ring_empty copies.
        ring_empty_mcast = pair_mask | (cutlass.Int32(3) << (leader_cta_id ^ cutlass.Int32(2)))
        tma_mcast_mask_kv = cutlass.Int16(5) << cta_in_pair  # {c, c ^ 2}: this CTA and its same-slice twin in the other pair
    else:
        # KV_SHARE 1: the ring is pair-local -- the release reaches only the pair, K / V land only in the issuer.
        ring_empty_mcast = pair_mask
        tma_mcast_mask_kv = tma_mcast_mask_qdo
    is_leader = cta_in_pair == 0
    is_cga_first_cta = cta_id_x == 0

    # --- SMEM: declaration order == address order (the DESC_VERSION rule is judged on it) ----
    sQ_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdO_raw = cutlass.Array(STORAGE_DTYPE, doBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sRingK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sRingV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # io-dtype staging for the workspace stores: TMA-store sources only (no tcgen05 descriptor), declared LAST.
    sCastS_raw = cutlass.Array(WORKSPACE_DTYPE, CFG.CAST_STAGES * castElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sCastDS_raw = cutlass.Array(WORKSPACE_DTYPE, CFG.CAST_STAGES * castElems, alignment=1024, space=cutlass.AddressSpace.smem)

    # --- debug context (folds to a Python 0 when every lever is off); declared AFTER the slabs so no descriptor root
    # moves under a debug build (the attribution lever adds 2 KiB of SMEM accumulators, 32 x Int64 per warp) ----
    if cutlass.const_expr(_DBG_CLK):
        _nx, _ny, _nz = cute.arch.grid_dim()
        _lin_block = bidx + _nx * (bidy + _ny * bidz)
        _dbg_ptr = cute.make_ptr(cutlass.Int64, _DBG_DUMP_ADDR, cute.AddressSpace.gmem, assumed_align=64)
        _dbg_t = cute.make_tensor(_dbg_ptr, cute.make_layout((DBG_CLK_MAX_WORDS,), stride=(1,)))
        sClk = cutlass.Array(cutlass.Int64, CFG.TOTAL_WARPS * DBG_CLK_WORDS, alignment=8, space=cutlass.AddressSpace.smem)
        dbg = Dbg(
            arr=cutlass.make_array_view(_dbg_t),
            slot=(_lin_block * cutlass.Int32(CFG.TOTAL_WARPS) + warp_idx) * cutlass.Int32(DBG_CLK_WORDS),
            cta_id_x=cta_id_x,
            bidx=bidx,
            bidy=bidy,
            bidz=bidz,
            clk=sClk,
            cslot=warp_idx * cutlass.Int32(DBG_CLK_WORDS),
        )
        # Each warp zeroes ITS slice (no cross-warp sync needed) and stamps the body start into slot 0 (clk) and 28 (ns).
        if nvvm.elect_sync():
            for _w in cutlass.range_constexpr(DBG_CLK_WORDS):
                sClk[dbg.cslot + cutlass.Int32(_w)] = cutlass.Int64(0)
            sClk[dbg.cslot] = cute.arch.clock64()
            sClk[dbg.cslot + cutlass.Int32(DBG_CLK_BODY_NS)] = cute.arch.globaltimer()
    elif cutlass.const_expr(_DBG):
        _nx, _ny, _nz = cute.arch.grid_dim()
        _lin_block = bidx + _nx * (bidy + _ny * bidz)
        _dbg_ptr = cute.make_ptr(cutlass.Int32, _DBG_DUMP_ADDR, cute.AddressSpace.gmem, assumed_align=64)
        _dbg_t = cute.make_tensor(_dbg_ptr, cute.make_layout((DBG_MAX_WORDS,), stride=(1,)))
        dbg = Dbg(
            arr=cutlass.make_array_view(_dbg_t),
            slot=(_lin_block * cutlass.Int32(CFG.TOTAL_WARPS) + warp_idx) * cutlass.Int32(DBG_WORDS),
            cta_id_x=cta_id_x,
            bidx=bidx,
            bidy=bidy,
            bidz=bidz,
        )
    else:
        dbg = 0

    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=TMA_OP_ITERS,
        tma_granu_elems=TMA_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sdO = SmemTile(
        base=sdO_raw,
        elems_per_stage=doBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=TMA_OP_ITERS,
        tma_granu_elems=TMA_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sRingK = SmemTile(
        base=sRingK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=TMA_RING_ITERS,
        tma_granu_elems=TMA_GRANU_ELEMS,
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sRingV = SmemTile(
        base=sRingV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=TMA_RING_ITERS,
        tma_granu_elems=TMA_GRANU_ELEMS,
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sCastS = SmemTile(
        base=sCastS_raw,
        elems_per_stage=castElems,
        stages=CFG.CAST_STAGES,
        leading_byte_offset=0,
        stride_byte_offset=_CORE_MATRIX_ROWS * CFG.S_SWZ_BYTES,
        layout=SMEM_LAYOUT_S,
        tma_loads_per_tile=S_TMA_ITERS,
        tma_granu_elems=S_D_BLOCK,
        tma_subtile_stride_elems=S_SUBTILE_SLAB,
        desc_version=DESC_VERSION,
    )
    sCastDS = SmemTile(
        base=sCastDS_raw,
        elems_per_stage=castElems,
        stages=CFG.CAST_STAGES,
        leading_byte_offset=0,
        stride_byte_offset=_CORE_MATRIX_ROWS * CFG.S_SWZ_BYTES,
        layout=SMEM_LAYOUT_S,
        tma_loads_per_tile=S_TMA_ITERS,
        tma_granu_elems=S_D_BLOCK,
        tma_subtile_stride_elems=S_SUBTILE_SLAB,
        desc_version=DESC_VERSION,
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
    bars = _make_bwd_d512_2x2_bars(CFG)

    # --- barrier init: init -> fence -> CTA sync -> cga sync -> arrives
    if warp_idx == 0:
        # ONE lane initialises every stage of every ring (MBarrier.init() initialises ONE stage).
        if nvvm.elect_sync():
            for _f in bars:
                for _i in cutlass.range_constexpr(_f.stages):
                    _f[_i].init()
            for _i in cutlass.range_constexpr(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(_i), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(_i), CFG.READ_TILE_ARRIVERS)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    # Required before ANY cross-CTA arrive (the ring release, the acc release, the multicast completes).
    cga_arrive()
    cga_wait()

    # --- warp dispatch ---------------------------------------------------
    if warp_idx < CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _compute_warp_group(
            bars,
            sched,
            sCastS_raw,
            sCastDS_raw,
            tmem_ptr_i32,
            lse_tensor,
            do_dot_tensor,
            seq_kv_lens_tensor,
            n_batch,
            n_qh,
            cta_id_x,
            leader_cta_id,
            n_kv,
            seqlen_q,
            seqlen_kv,
            attn_scale_in,
            attn_scale_log2e,
            attn_scale_for_dS,
            head_base,
            batch_base,
            dbg,
        )
    elif warp_idx == CFG.MMA_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp_leader(
                bars,
                sched,
                sQ,
                sdO,
                sRingK,
                sRingV,
                tmem_ptr_i32,
                seq_kv_lens_tensor,
                n_batch,
                n_qh,
                cta_id_x,
                pair_mask,
                ring_empty_mcast,
                n_kv,
                seqlen_q,
                seqlen_kv,
                dbg,
            )
        else:
            _mma_warp_non_leader(bars, sched, tmem_ptr_i32, seq_kv_lens_tensor, n_batch, n_qh, cta_id_x, seqlen_q, seqlen_kv, dbg)
    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmaldg_warp_group(
            bars,
            sched,
            sQ,
            sdO,
            sRingK,
            sRingV,
            tma_q,
            tma_k,
            tma_v,
            tma_do,
            seq_kv_lens_tensor,
            desc_words,
            n_batch,
            n_qh,
            cta_id_x,
            cta_in_pair,
            is_leader,
            pair_id,
            tma_mcast_mask_qdo,
            tma_mcast_mask_kv,
            n_kv,
            seqlen_q,
            seqlen_kv,
            gqa_ratio,
            head_base,
            batch_base,
            dbg,
        )
    elif warp_idx == CFG.TMASTG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            bars, sched, sCastS, sCastDS, tma_s, tma_ds, seq_kv_lens_tensor, n_batch, n_qh, cta_id_x, n_kv, seqlen_q, seqlen_kv, head_base, batch_base, dbg
        )
    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(_THD):
            # THD claims units against a DEVICE-computed live bound (the grid is occupancy-sized).  Same
            # mbarrier protocol as the CLC arm, so READ_TILE_ARRIVERS is unchanged by the swap.
            scheduler_warp_loop_persistent(
                sched,
                CFG.SCHEDULER_STAGES,
                is_cga_first_cta,
                seq_kv_lens_tensor,
                cutlass.Int32(4) * n_batch + cutlass.Int32(3),
                cutlass.Int32(4) * n_batch + cutlass.Int32(2),
                CGA_SIZE,
                CFG.CGA_M,
            )
        else:
            scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta, CGA_SIZE)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


def _tma_swz(byte_w: int):
    return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b


@cute.kernel
def _clamp_thd_input_descs_kernel(
    base_q_desc: cutlass.GridConstant[tmap.TensorMap],
    base_do_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    n_batch: cutlass.Int32,
    n_clusters: cutlass.Int32,
) -> None:
    """Copy this kernel's four input descriptors, clamped to the PACKED TOTALS (issue #624): a THD caller
    binds Q/K/V/dO at buffer CAPACITY, so the last sequence's tile tail steps into bytes that may never
    have been written; clamping makes those rows TMA-OOB -- exact zeros.  Also re-seeds the persistent
    scheduler's claim counter (``meta[4B+3]``) to ``n_clusters`` for THIS launch: the setup launch seeds it
    once per execute, the chain launches stage 2 once per head chunk over the same metadata, and a launch
    leaves it at ``live + n_clusters`` -- unreset, the next launch computes only its blockIdx-assigned
    units.  No extra launch (Rule 2); the kernel boundary publishes it.  One elected thread."""
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < cutlass.Int32(32):
        if nvvm.elect_sync():
            _clamp_thd_input_descs(base_q_desc, base_do_desc, base_k_desc, base_v_desc, desc_words, meta_t, n_batch, n_clusters)


_clamp_thd_input_descs_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _clamp_thd_input_descs(base_q_desc, base_do_desc, base_k_desc, base_v_desc, desc_words, meta_t, n_batch, n_clusters):
    """Body of the clamp; the caller elects."""
    meta = cutlass.make_array_view(meta_t)
    t_q = cutlass.Int32(meta[cutlass.Int32(2) * n_batch])  # cu_q[B]
    t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])  # cu_k[B]
    emit_clamped_desc(base_q_desc, desc_words, cutlass.Int32(Q_SLOT), t_q, seq_ord=_THD_SEQ_ORD)
    emit_clamped_desc(base_do_desc, desc_words, cutlass.Int32(DO_SLOT), t_q, seq_ord=_THD_SEQ_ORD)
    emit_clamped_desc(base_k_desc, desc_words, cutlass.Int32(K_SLOT), t_kv, seq_ord=_THD_SEQ_ORD)
    emit_clamped_desc(base_v_desc, desc_words, cutlass.Int32(V_SLOT), t_kv, seq_ord=_THD_SEQ_ORD)
    # Claim counter := n_clusters (THD_CTR_OFF = 4B+3): cluster c takes unit c from its blockIdx, then
    # claims from here -- the seed write_thd_live_and_ctr makes once, redone for every chunk launch.
    meta[cutlass.Int32(4) * n_batch + cutlass.Int32(3)] = n_clusters
    nvvm.fence_proxy_release(
        nvvm.MemScope.GPU,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )


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
    desc_words: cute.Tensor,
    # (B, QH, S_q_pad, S_kv_pad, QH_chunk, QH_kv, S_q, S_kv, N_THD_UNITS) -- the 4x1 sibling's ABI exactly.
    problem_size: Tuple[int, int, int, int, int, int, int, int, int],
    attn_scale_in: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    attn_scale_for_dS: cutlass.Float32,
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, SQ, SKV, QH_CHUNK, QH_KV, SQ_REAL, SKV_REAL, N_THD_UNITS = problem_size

    # Q/K/V/dO are BSHD [B, S, H, D]; the workspaces are [B, H_chunk, S_q, S_kv].  stride_order is
    # innermost-first, so the coords the kernel passes are (d, head, seq, batch) for the operands and
    # (kv, q, head, batch) for the workspaces.
    stride_order = (3, 2, 1, 0)
    op_box = (1, CFG.TILE_M, 1, TMA_GRANU_ELEMS)  # Q and dO: this CTA's 64-row block, one 64-column subtile per box
    ring_box = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_GRANU_ELEMS)  # K and V: per-CTA N slice, one 64-column chunk box
    # One workspace subtile.  S_D_BLOCK * BPE == the swizzle atom exactly; a full TILE_N row would exceed it.
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

    # One cluster covers CLUSTER_Q_ROWS q rows; CGA_M CTAs per cluster.  Dense: the grid IS the work list.
    # THD: occupancy-sized grid, the persistent scheduler hands out units from the claim counter.
    rows_per_cluster = CFG.CLUSTER_Q_ROWS
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    if cutlass.const_expr(_THD):
        grid_shape = (N_THD_UNITS * CFG.CGA_M, 1, 1)
        # Ahead of the main launch, on the same stream: kernel-boundary ordering publishes the patched descriptors
        # and the re-seeded claim counter.
        _clamp_thd_input_descs_kernel(
            tma_q_desc,
            tma_do_desc,
            tma_k_desc,
            tma_v_desc,
            desc_words,
            seq_kv_lens_tensor,
            cutlass.Int32(B),
            cutlass.Int32(N_THD_UNITS),
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    else:
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
        desc_words,
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


CLUSTER_Q_ROWS = CFG.CLUSTER_Q_ROWS
__all__ = ["CFG", "CLUSTER_Q_ROWS", "DESC_VERSION", "PARAMS", "_host"]
