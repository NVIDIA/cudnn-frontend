# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.scheduler import (
    SCHED_LPT_L2,
    SCHED_NATURAL,
    lpt_tile_coords,
    lpt_l2_tile_coords,
)

# KvLoopBounds / compute_kv_loop_bounds moved to frost.tile_dsl.mask (the
# backward needs the same tile-level bounds); re-exported here so the twelve
# prefill kernels that import them from this module keep working.
from cudnn.frost.tile_dsl.mask import (  # noqa: F401
    MASK_CAUSAL,
    MASK_PADDED,
    MASK_SWA,
    KvLoopBounds,
    compute_kv_loop_bounds,
    _div_up,
)
from cudnn.frost.tile_dsl.barrier import MBarrier, Producer, Scope
from cudnn.frost.tile_dsl.pointwise import fmul2, ffma2, opaque_f32_zero
from cudnn.frost.tile_dsl.tma import tma_load_tile

# The O-swizzle selector lives on the base config line (config_sm107 carries a
# byte-identical copy).  Importing it from config_sm100 keeps this cross-arch
# module off the Rubin package (engine-contract S8: a shared module never leans
# on one arch's package) and is cycle-free -- config_sm100 imports only
# tile_dsl.constants, never this package.
from cudnn.sdpa.fwd.config_sm100 import o_swz_bytes as _o_swz_bytes


@cute.jit
def sanitize_mxfp8_thd_v_sf_padding(sv_sf, kv_tile_idx, seqlen_kv):
    """Replace fully padded V scale blocks before block-scaled BMM2.

    F8_128x4 stores four 32-token V scale blocks in each 4-byte row group.
    The packed THD tail may leave whole blocks outside the sequence; their data
    TMA-loads as zero, but an E8M0 NaN scale would still make ``0 * NaN`` poison
    the accumulator.  Preserve blocks containing valid tokens and replace only
    the fully padded scale bytes with the finite scale 1.0 (0x7f).
    """
    remaining = seqlen_kv - kv_tile_idx * cutlass.Int32(128)
    valid_blocks = (remaining + cutlass.Int32(31)) // cutlass.Int32(32)
    if valid_blocks < cutlass.Int32(4):
        lane = cute.arch.thread_idx()[0] % cutlass.Int32(32)
        keep_mask = (cutlass.Int32(1) << (valid_blocks * cutlass.Int32(8))) - cutlass.Int32(1)
        fill_mask = cutlass.Int32(0x7F7F7F7F) & ~keep_mask
        for row_group in cutlass.range_constexpr(4):
            byte_offset = lane * cutlass.Int32(16) + cutlass.Int32(row_group * 4)
            word = Pointer(sv_sf.base.subview(byte_offset).data_ptr(), dtype=cutlass.Int32)
            word.store((word.load(alignment=4) & keep_mask) | fill_mask, alignment=4)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


class Bars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object

    mb_stat_full: object
    mb_stat_empty: object

    # Per-sub-tile "final stats consumed" gate (correction -> MMA, cross-CTA
    # to the leader under cga2).  The per-tile (total_max, total_sum) publish
    # rides the HEAD of the S_acc slot (LAYOUT.STATS_OFF + qs*STATS_STRIDE),
    # so the NEXT tile's prologue BMM1 — gated only on q_full/k_full —
    # overwrites it.  The mb_stat_full/mb_stat_empty handshake only orders
    # softmax vs correction; nothing ordered MMA's next-tile BMM1 after
    # correction's epilogue stats read (or after softmax's final stats
    # write).  One arrive per tile per sub-tile from the correction epilogue;
    # the leader MMA waits it before the prologue BMM1 into that S slot.
    mb_stats_read: object

    mb_o_full: object
    mb_o_empty: object

    mb_tmem_dealloc: object
    mb_empty_mainloop: object

    mb_q_o_alias: object
    # Return edge of the Q∪O alias gate (see the soundness note in
    # make_classic_bars): TMA-LDG arrives after consuming each alias-gate
    # phase; TMA-STG waits it before its next alias arrive.
    mb_qo_slab_free: object

    # Fused epilogue gate (O := O * sigmoid(G)) staging-tile handshake; None
    # unless the bars were built with epilogue_gate=True.  Table + lane
    # arithmetic: make_d256_bars.
    mb_gate_full: object = None
    mb_gate_empty: object = None


class D256Bars(NamedTuple):
    mb_q_full: object
    mb_q_o_alias: object
    mb_tmastg_go: object

    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object

    mb_stat_full: object
    mb_stat_empty: object

    mb_o_full: object
    mb_o_empty: object

    mb_empty_mainloop: object
    mb_tmem_dealloc: object

    # Fused epilogue gate staging-tile handshake (None unless
    # epilogue_gate=True) -- see make_d256_bars.
    mb_gate_full: object = None
    mb_gate_empty: object = None


def make_d256_bars(CFG, *, N_O_CHUNKS: int, epilogue_gate: bool = False) -> D256Bars:
    """Barrier bundle for the Q∪O-aliased d256 pipeline.

    ``epilogue_gate`` adds the two LOCAL barriers of the fused epilogue gate
    (O := O * sigmoid(G)); both stay ``None`` otherwise so every existing
    kernel traces byte-identically.  Their table (identical on the f16 and
    the per-tensor FP8 d256 kernels):

      mb_gate_full   Producer.TMA_LOAD, Scope.LOCAL.  ONE arrive_expect_tx per
                     tile from the TMA-LDG warp, ``pred=nvvm.elect_sync()``
                     -> 1 issuing lane x 1 call = 1 == init CFG.ONE_LANE.
                     Bytes: the gate tile's subtiles, cta_group=1 (shared::cta),
                     ALL to this CTA's mbar -- no CTA_MMA factor (P9 routing
                     applies to the cta_group::2 tensor form only).  Consumer:
                     the correction warpgroup, 128 lanes wait once per tile,
                     phase starts 0 (wait-then-arrive, P2).
      mb_gate_empty  Producer.THREAD, Scope.LOCAL.  BARE ``.arrive()`` from every
                     correction lane once per tile after its last gate LDS ->
                     128 issuing lanes x 1 call = 128 == init CFG.CORR_LANES
                     (THREAD has no ``pred=`` path; same form as mb_o_full).
                     Consumer: the TMA-LDG warp, one wait per tile at the TOP
                     of its tile iteration, phase PRE-ARMED at 1 (P5b).
      P14: the gate load is issued on BOTH arms of the empty-mainloop branch
                     and the epilogue runs on every tile, so each tile is
                     exactly one wait + one arrive on each bar at every shape.
      P15: no cross-CTA arrive on either bar -> no drain owed.
    """
    SOFTMAX_LANES_TOTAL = CFG.SOFTMAX_LANES * CFG.CTA_MMA
    CORR_LANES_TOTAL = CFG.CORR_LANES * CFG.CTA_MMA
    SOFTMAX_PLUS_CORR_TOTAL = SOFTMAX_LANES_TOTAL + CORR_LANES_TOTAL
    KV_EMPTY_ARRIVERS = (CFG.CGA_M // CFG.CTA_MMA) + CFG.CGA_N - 1
    N_BMM2_CHUNKS = CFG.N_BMM2_CHUNKS

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return D256Bars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_o_alias=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_tmastg_go=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_bmm1_done=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(2 * N_BMM2_CHUNKS),
            stages=2 * N_BMM2_CHUNKS,
            init_count=tuple(SOFTMAX_PLUS_CORR_TOTAL if (s % N_BMM2_CHUNKS) == 0 else SOFTMAX_LANES_TOTAL for s in range(2 * N_BMM2_CHUNKS)),
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_stat_full=MBarrier(_alloc(1), stages=1, init_count=CFG.SOFTMAX_LANES, producer=Producer.THREAD),
        mb_stat_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_full=MBarrier(_alloc(N_O_CHUNKS), stages=N_O_CHUNKS, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.THREAD),
        mb_gate_full=(MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD) if epilogue_gate else None),
        mb_gate_empty=(MBarrier(_alloc(1), stages=1, init_count=CFG.CORR_LANES, producer=Producer.THREAD) if epilogue_gate else None),
    )


def make_classic_bars(CFG, *, epilogue_gate: bool = False) -> Bars:
    # ``epilogue_gate``: same two LOCAL gate barriers as make_d256_bars (table
    # there); no classic (d128/d192) kernel passes True yet, so today every
    # caller gets None and traces unchanged.  The init counts (ONE_LANE /
    # CORR_LANES) were derived for the d256 BODIES only: a classic kernel that
    # flips this must first splice the gate seams into ITS TMA-LDG and
    # correction groups and re-audit both rows per P3 (SUM(issuing lanes) ==
    # init) against those bodies -- the counts are not inherited from here.
    SOFTMAX_PLUS_CORR_TOTAL = CFG.SOFTMAX_LANES * 2 * CFG.CTA_MMA
    SOFTMAX_LANES_TOTAL = CFG.SOFTMAX_LANES * CFG.CTA_MMA
    CORR_LANES_TOTAL = CFG.CORR_LANES * CFG.CTA_MMA
    KV_EMPTY_ARRIVERS = (CFG.CGA_M // CFG.CTA_MMA) + CFG.CGA_N - 1
    N_BMM2_CHUNKS = CFG.N_BMM2_CHUNKS

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        mb_q_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_bmm1_done=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(CFG.TILES_Q * N_BMM2_CHUNKS),
            stages=CFG.TILES_Q * N_BMM2_CHUNKS,
            init_count=tuple(SOFTMAX_PLUS_CORR_TOTAL if (s % N_BMM2_CHUNKS) == 0 else SOFTMAX_LANES_TOTAL for s in range(CFG.TILES_Q * N_BMM2_CHUNKS)),
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_stat_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.SOFTMAX_LANES, producer=Producer.THREAD),
        mb_stat_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_stats_read=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_o_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.THREAD),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        # Q∪O alias gate FULL/EMPTY pair.  mb_q_o_alias alone is UNSOUND:
        # mbarrier parity waits deadlock once a producer runs >= 2 phases
        # ahead, and on EMPTY tiles (zero-KV varlen sequences) the
        # corr -> STG -> alias-arrive chain has NO dependency on TMA-LDG, so
        # a delayed LDG warp loses the race and its bootstrap parity credit
        # is consumed by a real arrive (observed: LDG parked forever at the
        # tile-1 alias wait with the barrier already in phase 1, deadlocking
        # the whole cluster).  mb_qo_slab_free is the return edge: LDG
        # arrives it right after consuming each alias phase and STG waits it
        # before each alias arrive, bounding either side's lead to one phase
        # by construction.
        mb_q_o_alias=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_qo_slab_free=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_gate_full=(MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD) if epilogue_gate else None),
        mb_gate_empty=(MBarrier(_alloc(1), stages=1, init_count=CFG.CORR_LANES, producer=Producer.THREAD) if epilogue_gate else None),
    )


def row_max_for_exp2(total_max):
    """Canonical masked-softmax row-max guard (FlashAttention / cuDNN form).

    The f16 prefill kernels mask scores with true ``-inf``
    (``apply_mask_chunk(..., mask_value=float("-inf"))``) and start the running
    ``total_max`` at ``-inf``, so for a row that has not yet seen a live column
    the (scaled) row max is still exactly ``-inf``. Using it directly would
    make every masked exp2 argument ``-inf - (-inf) == NaN`` (or, with the old
    finite sentinel, ``0`` and hence a bogus P == 1 per masked column). The
    canonical fix — FlashAttention's ``max == -INFINITY ? 0.f : max * scale``
    and cuDNN's ``(total_max == NEG_INFINITY) ? 0.0f : total_max`` — is a
    single compare + select substituting 0 AT THE POINT OF USE:

      * masked scores become ``exp2(-inf - 0) == 0``, so a fully-masked
        iteration ships P == 0 to BMM2 and contributes nothing;
      * a fully-masked row naturally ends the tile with ``total_sum == 0``,
        which the epilogue turns into O := 0 / LSE := -inf;
      * substituted into BOTH alpha operands, any iteration that does not move
        the max (fully-masked ones included, since ``-inf`` can never clear
        the RESCALE_THRESHOLD update) yields alpha == exp2(0 - 0) == 1
        exactly, so the all_alpha_one ballot keeps firing.

    Callers clamp the alpha exponent with ``min(prev_safe - new_safe, 0)`` so
    the one transition that can lower the safe max (dead -> alive, 0 ->
    real*scale < 0) cannot overflow exp2; total_sum is still exactly 0 there,
    so any finite alpha is exact.
    """
    still_dead = total_max == cutlass.Float32(float("-inf"))
    return cutlass.Float32(
        arith.select(
            still_dead.ir_value(),
            cutlass.Float32(0.0).ir_value(),
            total_max.ir_value(),
        )
    )


def assert_tile_n_supported(CFG):
    """Import-time gate: kernels on the ``reg_S_a``/``reg_S_b`` softmax body
    require N_BMM2_CHUNKS == 2 (TILE_N == 128)."""
    if CFG.N_BMM2_CHUNKS != 2:
        raise NotImplementedError(f"this kernel currently requires TILE_N=128 (got TILE_N={CFG.TILE_N})")


@cute.jit
def store_fp32_partial_tile(
    o_partial_f32,
    tmem_base,
    tmem_o_off,
    inv_sum,
    row_dead,
    row_valid,
    o_batch,
    q_row_global,
    row_head_idx,
    tile_o: cutlass.Constexpr[int],
    chunk: cutlass.Constexpr[int],
) -> None:
    """Store one Q row's O tile as fp32, straight from TMEM to the workspace.

    The staged path casts the accumulator into the SMEM O tile and TMA-stores
    that, which ties the partial's width to the tile's.  Here the accumulator
    goes to global directly, so the partial can be fp32 while the tile -- and
    therefore the SMEM budget -- is untouched.  Shared by every flavor whose
    epilogue holds its O accumulator in TMEM.

    ``chunk`` must be the flavor's OWN TMEM read width (its ``O_CHUNK``): the
    O region is not uniformly addressable across flavors, so reading it in a
    different stride than the staged epilogue does silently returns the wrong
    columns rather than failing.

    ``row_dead`` cannot be dropped in favour of the zeroed ``inv_sum`` every
    caller already computes: an empty mainloop never wrote O TMEM, so the load
    can return NaN, and ``NaN * 0.0`` is NaN, not zero.  The staged paths avoid
    this by not loading at all for such rows; here the select does it.
    """
    op = cutlass.make_array_view(o_partial_f32)
    # The slab carries the graph's ACTUAL d_v, which an ENVELOPE flavor routinely
    # exceeds -- d_v=64 runs on the d128 tile, so tile_o overshoots each row by 64
    # columns.  The staged TMA path clipped that to the tensor extent; a direct
    # store has to bound itself or it writes into the next head's row, and off the
    # end of the slab on the last one.  Read off the tensor rather than passed in,
    # so it cannot drift from the buffer actually bound.
    d_v = cutlass.const_expr(o_partial_f32.shape[3])
    for blk in cutlass.range_constexpr(tile_o // chunk):
        addr = tmem_base + cutlass.Int32(tmem_o_off + blk * chunk)
        vals = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(addr, cutlass.Float32), num=chunk)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        scaled = vals * inv_sum
        if row_valid:
            row_out = op[o_batch, q_row_global, row_head_idx, :]
            for j in cutlass.range_constexpr(chunk):
                if cutlass.const_expr(blk * chunk + j < d_v):
                    row_out[cutlass.Int32(blk * chunk + j)] = cutlass.Float32(
                        arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), scaled[j].ir_value())
                    )


class SplitHelpers(NamedTuple):
    """Split-aware decode / bounds closures, plus the two flags kernels fold on."""

    SPLIT_KV: int
    # True when a tile's KV range can come out empty (right <= left).  See
    # make_split_helpers for why KV split makes this reachable without a mask.
    MAY_BE_EMPTY: bool
    split_chunk: object
    decode_initial_split: object
    decode_payload_split: object
    bounds_for_tile_split: object
    nomask_range_split: object
    partial_batch: object


def make_split_helpers(CFG, *, bounds_for_tile, dispatch_decode_initial, dispatch_decode_payload) -> SplitHelpers:
    """Split-aware decode / bounds closures shared by the SM100 prefill flavors.

    ``bounds_for_tile`` is the caller's own bounds closure, taking
    ``(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr,
    batch_idx, qh_per_kh)`` — flavors differ in whether they apply the
    dead-Q-tile trim, so the split narrowing composes on top of whatever they
    already do.  ``qh_per_kh`` (trailing, default 1) is the graph's GQA
    ratio; with CFG.PACK_GQA it is the packing group
    size: the split chunks the tile's PACKED token-span bounds, so packing
    and KV split compose; without CFG.PACK_GQA the bounds fold to the classic
    single-head-per-tile form.
    At SPLIT_KV == 1 every closure below folds away and the traced code is the
    classic single-pass kernel.
    """
    SPLIT_KV = int(getattr(CFG, "SPLIT_KV", 1))

    # The split index rides the BATCH axis (grid.z), not grid.x: the decode
    # already recovers the batch coordinate on BOTH the blockIdx and the
    # scheduler-handout paths -- it is the high half of the packed head|batch
    # word -- so a composite z = batch + split*B travels with it for free, with
    # no in-place mutation of the shared tile id and no dependence on the grid's
    # x extent (which is q_clusters * CGA_M, NOT the n_q_supers the kernel is
    # handed, on any flavor where CGA_M != CTA_MMA -- d512).

    # Can a tile's KV range come out EMPTY (right <= left)?
    #
    # Before KV split the answer was "only under a mask" — a SWA/causal/padded
    # tile can fall entirely outside the band — so the empty-tile handshake
    # (mb_empty_mainloop: correction arrives, MMA waits, TMA-LDG skips its
    # loads) was gated on MASK_FLAGS != 0 and folded away at MASK_NONE, where
    # [0, S_kv/TILE_N) is never empty.
    #
    # KV split breaks that WITHOUT a mask: a split past the end of a short range
    # legitimately gets zero tiles.  Correction detects empties with a RUNTIME
    # test, so if the gate const-folds to False the warp groups disagree —
    # correction jumps to its epilogue while MMA waits on mb_q_full and TMA-LDG
    # issues loads nobody consumes — and the kernel deadlocks.
    MAY_BE_EMPTY = (CFG.MASK_FLAGS != 0) or (SPLIT_KV > 1)

    # Which grid does this flavor launch?  SCHED_NATURAL uses a 3-D
    # (q_super, head, batch) grid; the LPT policy flattens everything into x.
    # This must use the flavor's effective CFG policy so the helper and launch
    # grid follow the same compile-time specialization.
    IS_LPT = CFG.SCHEDULER_POLICY != SCHED_NATURAL

    @cute.jit
    def _lpt_split_of(raw, n_q_supers, n_qh, n_batch):
        """(within-split raw x, split) for the flattened LPT grid.

        The LPT tile space is q_tiles * n_qh * n_batch clusters; KV split
        appends SPLIT_KV copies of it, split-major, so the split is the high
        digit of the cluster index.  The CGA lane (raw % CGA_M) is preserved so
        the caller's decode still sees a well-formed x coordinate.
        """
        cga = cutlass.Int32(CFG.CGA_M)
        linear = raw // cga
        q_tiles = n_q_supers // cutlass.Int32(CFG.CTA_MMA)
        per_split = q_tiles * n_qh * n_batch
        split = linear // per_split
        rest = (linear % per_split) * cga + (raw % cga)
        return rest, split

    @cute.jit
    def _split_chunk(left, right, split_idx):
        """Cut ``[left, right)`` into SPLIT_KV near-equal chunks.

        The FIRST ``rem`` splits get one extra tile, so chunk sizes differ by at
        most 1 however the mask has already narrowed the range — a balanced cut
        matters because the slowest split sets the critical path.  A split past
        the end collapses to ``lo == hi``; the existing empty-mainloop path then
        writes O := 0 / LSE := -inf, exactly the identity of the combine's
        log-sum-exp, so no special case is needed downstream.
        """
        n_tiles = right - left
        per = n_tiles // cutlass.Int32(SPLIT_KV)
        rem = n_tiles % cutlass.Int32(SPLIT_KV)
        lo = left + split_idx * per + cute.math.min(split_idx, rem)
        extra = cutlass.Int32(
            arith.select(
                (split_idx < rem).ir_value(),
                cutlass.Int32(1).ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        return lo, lo + per + extra

    @cute.jit
    def _decode_initial_split(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        """decode_initial + this tile's split index.

        NATURAL: the split rides the BATCH axis (see the note above on why not
        grid.x).  The host launches z = B * SPLIT_KV, so the split falls out of
        the DECODED batch coordinate as ``b // n_batch``, leaving the real batch
        as ``b % n_batch``.

        LPT / LPT_L2: the grid is flat, so there is no batch axis to ride and the
        split is folded into the linear tile id instead; ``_lpt_split_of`` peels
        it back off before the flavor's dispatcher sees the id.

        ``qh_per_kh`` / ``seqlen_kv`` are the LPT_L2 cost-model inputs; they are
        opaque here and forwarded to the flavor's dispatcher unchanged.
        """
        if cutlass.const_expr(SPLIT_KV > 1 and IS_LPT):
            raw, split = _lpt_split_of(bidx, n_q_supers, n_qh, n_batch)
            q, h, b = dispatch_decode_initial(raw, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
            return q, h, b, split
        q, h, b = dispatch_decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
        if cutlass.const_expr(SPLIT_KV == 1):
            return q, h, b, cutlass.Int32(0)
        return q, h, b % n_batch, b // n_batch

    @cute.jit
    def _decode_payload_split(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        """decode_payload + split index; ``t0`` is the try_cancel cluster-base id."""
        if cutlass.const_expr(SPLIT_KV > 1 and IS_LPT):
            raw, split = _lpt_split_of(t0, n_q_supers, n_qh, n_batch)
            q, h, b = dispatch_decode_payload(raw, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
            return q, h, b, split
        q, h, b = dispatch_decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
        if cutlass.const_expr(SPLIT_KV == 1):
            return q, h, b, cutlass.Int32(0)
        return q, h, b % n_batch, b // n_batch

    @cute.jit
    def _bounds_for_tile_split(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, qh_per_kh: int = 1):
        """The flavor's (possibly packed) bounds, narrowed to this split's slice
        of the KV range.

        Splitting the ALREADY-masked ``[left, right)`` rather than the raw KV
        extent is what keeps causal / SWA correct AND balanced: each split gets
        an equal share of the tile's real work, not of the sequence — and under
        PackGQA that range is already in packed token-span units, so the two
        features compose.  The unmasked band is clamped into the slice, which
        preserves the ``left <= unmasked_lo <= unmasked_hi <= right`` invariant
        the mainloop relies on, because clamping is monotone.
        """
        b = bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, qh_per_kh)
        if cutlass.const_expr(SPLIT_KV == 1):
            return b
        lo, hi = _split_chunk(b.left, b.right, split_idx)
        return KvLoopBounds(
            left=lo,
            unmasked_lo=cute.math.min(cute.math.max(b.unmasked_lo, lo), hi),
            unmasked_hi=cute.math.min(cute.math.max(b.unmasked_hi, lo), hi),
            right=hi,
        )

    @cute.jit
    def _nomask_range_split(seqlen_kv, split_idx):
        """MASK_NONE fast-path KV range, split-aware.

        MMA / TMA-LDG take this path while softmax / correction go through
        bounds_for_tile_split; every warp group must land on the SAME chunk
        boundaries or their mbarrier handshakes desync.  The split branch
        therefore divides with the div-up compute_kv_loop_bounds uses, not the
        floor of the historical fast path.
        """
        if cutlass.const_expr(SPLIT_KV == 1):
            return cutlass.Int32(0), seqlen_kv // cutlass.Int32(CFG.TILE_N)
        n_tiles = (seqlen_kv + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N)
        return _split_chunk(cutlass.Int32(0), n_tiles, split_idx)

    @cute.jit
    def _partial_batch(batch_idx, split_idx, n_batch):
        """Batch coord of this split's partial O / LSE slot (split-major).

        Stacking the partials on the BATCH axis (extent B*SPLIT_KV) means the O
        TMA descriptor is untouched — only the coord shifts.  Folds to batch_idx
        at SPLIT_KV == 1.
        """
        if cutlass.const_expr(SPLIT_KV == 1):
            return batch_idx
        return batch_idx + split_idx * n_batch

    return SplitHelpers(
        SPLIT_KV=SPLIT_KV,
        MAY_BE_EMPTY=MAY_BE_EMPTY,
        split_chunk=_split_chunk,
        decode_initial_split=_decode_initial_split,
        decode_payload_split=_decode_payload_split,
        bounds_for_tile_split=_bounds_for_tile_split,
        nomask_range_split=_nomask_range_split,
        partial_batch=_partial_batch,
    )


@cute.jit
def decode_linear_tile_lpt_grouped(linear, q_h, batch, q_tiles, lpt_head_group: cutlass.Constexpr[int]):
    head_group = cutlass.Int32(lpt_head_group)
    group_span = q_tiles * head_group
    group_idx = linear // group_span
    group_offset = linear % group_span
    row_rank = group_offset // head_group
    within = group_idx * head_group + group_offset % head_group
    row = (q_tiles - cutlass.Int32(1)) - row_rank
    head = within % q_h
    batch_idx = within // q_h
    return row, head, batch_idx


class SdpaHelpers(NamedTuple):
    decode_initial: object
    decode_payload: object
    bounds_for_tile: object
    bounds_for_tile_qtrim: object
    resolve_seqlen_kv: object
    resolve_seqlen_q: object
    thd_decode: object
    dispatch_decode_initial: object
    dispatch_decode_payload: object
    thd_tma_offsets: object
    thd_sf_tile_bases: object


def make_sdpa_helpers(
    CFG,
    lpt_q_tiles_in_cga_units: bool = False,
    grouped_lpt: bool = False,
    lpt_head_group: int = 1,
    lpt_q_tiles: int = 0,
) -> SdpaHelpers:
    cga_tile_m = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

    _cga_m = getattr(CFG, "CGA_M", 1)
    _cta_mma = getattr(CFG, "CTA_MMA", 1)

    @cute.jit
    def _lpt_linear(block_id):
        if cutlass.const_expr(_cga_m > 1):
            return block_id // cutlass.Int32(_cga_m)
        return block_id

    @cute.jit
    def _lpt_q_super(row, cta_in_pair):
        if cutlass.const_expr(_cta_mma > 1):
            return row * cutlass.Int32(_cta_mma) + cta_in_pair
        return row

    if CFG.SCHEDULER_POLICY == SCHED_NATURAL:
        if CFG.SPLIT_PIPELINE == 1:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                blocked_row = (bidx // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
                return blocked_row, bidy, bidz

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                blocked_row = (t0 // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
                head = t1 & cutlass.Int32(0xFFFF)
                batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
                return blocked_row, head, batch

        else:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                return bidx, bidy, bidz

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                head = t1 & cutlass.Int32(0xFFFF)
                batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
                return t0 + cta_in_pair, head, batch

    elif CFG.SCHEDULER_POLICY == SCHED_LPT_L2:
        _kv_bytes_per_row = (CFG.TILE_K + CFG.TILE_O) * CFG.BPE
        _l2_bytes = CFG.L2_SIZE_MIB * 1024 * 1024

        @cute.jit
        def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
            linear = _lpt_linear(bidx)
            q_tiles = n_q_supers // cutlass.Int32(_cta_mma) if lpt_q_tiles_in_cga_units else n_q_supers
            if cutlass.const_expr(qh_per_kh is None or seqlen_kv is None):
                raise ValueError("SCHED_LPT_L2 decode requires qh_per_kh and seqlen_kv at every call site")
            row, head, batch = lpt_l2_tile_coords(linear, n_qh, n_batch, q_tiles, qh_per_kh, seqlen_kv, _kv_bytes_per_row, _l2_bytes)
            return _lpt_q_super(row, cta_in_pair), head, batch

        @cute.jit
        def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
            linear = _lpt_linear(t0)
            q_tiles = n_q_supers // cutlass.Int32(_cta_mma) if lpt_q_tiles_in_cga_units else n_q_supers
            if cutlass.const_expr(qh_per_kh is None or seqlen_kv is None):
                raise ValueError("SCHED_LPT_L2 decode requires qh_per_kh and seqlen_kv at every call site")
            row, head, batch = lpt_l2_tile_coords(linear, n_qh, n_batch, q_tiles, qh_per_kh, seqlen_kv, _kv_bytes_per_row, _l2_bytes)
            return _lpt_q_super(row, cta_in_pair), head, batch

    else:
        if grouped_lpt:
            if lpt_q_tiles > 0:

                @cute.jit
                def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(bidx), n_qh, n_batch, cutlass.Int32(lpt_q_tiles), lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

                @cute.jit
                def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(t0), n_qh, n_batch, cutlass.Int32(lpt_q_tiles), lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

            else:

                @cute.jit
                def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    q_tiles = n_q_supers // cutlass.Int32(_cta_mma) if lpt_q_tiles_in_cga_units else n_q_supers
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(bidx), n_qh, n_batch, q_tiles, lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

                @cute.jit
                def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    q_tiles = n_q_supers // cutlass.Int32(_cta_mma) if lpt_q_tiles_in_cga_units else n_q_supers
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(t0), n_qh, n_batch, q_tiles, lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

        else:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                linear = _lpt_linear(bidx)
                q_tiles = n_q_supers // cutlass.Int32(_cta_mma) if lpt_q_tiles_in_cga_units else n_q_supers
                row, head, batch = lpt_tile_coords(linear, n_qh, n_batch, q_tiles)
                return _lpt_q_super(row, cta_in_pair), head, batch

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                linear = _lpt_linear(t0)
                q_tiles = n_q_supers // cutlass.Int32(_cta_mma) if lpt_q_tiles_in_cga_units else n_q_supers
                row, head, batch = lpt_tile_coords(linear, n_qh, n_batch, q_tiles)
                return _lpt_q_super(row, cta_in_pair), head, batch

    @cute.jit
    def _bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, qh_per_kh: int = 1):
        # Token capacity of one CGA super-tile: TILES_Q * TILE_M rows hold
        # rows/G tokens when packing (CFG.PACK_GQA), else one token per row.
        tokens_per_super = (CFG.TILES_Q * CFG.TILE_M) // qh_per_kh if CFG.PACK_GQA else CFG.TILES_Q * CFG.TILE_M
        cga_base_super = q_super_idx - cta_in_pair
        q_row_coord = cga_base_super * cutlass.Int32(tokens_per_super)
        return compute_kv_loop_bounds(
            q_row_coord,
            seqlen_q,
            seqlen_kv,
            CFG.WINDOW_LEFT,
            CFG.MASK_FLAGS,
            CFG.TILE_N,
            cga_tile_m // qh_per_kh if CFG.PACK_GQA else cga_tile_m,
            bottom_right=bool(CFG.BOTTOM_RIGHT),
            window_right=int(CFG.WINDOW_RIGHT),
        )

    @cute.jit
    def _bounds_for_tile_qtrim(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, qh_per_kh: int = 1):
        """bounds_for_tile + cuDNN-style dead-Q-tile KV-loop collapse.

        Mirrors cuDNN fort (mma_pipeline_op_native_sdpa_prefill_sm100_nonfp8
        .cpp:916-921): when the CGA tile's base Q row is at/past this batch's
        actual Q length (SEQ_Q_LENS_PRESENT dense padded-Q trim; q lens are
        the SEPARATE (B,)-int32 ``seq_q_lens_addr`` kernel parameter — cuDNN
        SEQLEN_Q / FA seqused_q style — ``0`` unless the flag is set, so
        the read below folds out with the branch), collapse the KV loop to
        empty (right := left, matching the SWA empty-tile machinery) — the
        grid stays padded-sized and a dead tile costs prologue+epilogue only.
        q_len_b == 0 (whole batch dead) collapses every tile since the base
        row coord is always >= 0.  Under PackGQA the dead-tile
        compare is the tile's base TOKEN (rows // G) against the per-batch
        Q length — the same token-space value the packed bounds use.  Either way
        the row coord is the SAME cga-base value _bounds_for_tile uses and q
        lens are per-batch constants, so every warp group calling this helper
        sees identical (collapsed) bounds and the barrier handshakes stay in
        lockstep.  The epilogue's SEQ_Q_LENS_PRESENT trim (applied after the
        sink fold) already forces O := 0 / LSE := -inf for every row of a
        collapsed tile.
        """
        b = _bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, qh_per_kh)
        if cutlass.const_expr(int(getattr(CFG, "SEQ_Q_LENS_PRESENT", 0)) == 1):
            tokens_per_super = (CFG.TILES_Q * CFG.TILE_M) // qh_per_kh if CFG.PACK_GQA else CFG.TILES_Q * CFG.TILE_M
            cga_base_super = q_super_idx - cta_in_pair
            q_row_coord = cga_base_super * cutlass.Int32(tokens_per_super)
            arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            q_len_b = cutlass.Int32(arr[cutlass.Int32(batch_idx)])  # batch_idx may arrive as a raw arith value from the decode
            tile_dead = q_row_coord >= q_len_b
            dead_lo = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.unmasked_lo.ir_value()))
            dead_hi = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.unmasked_hi.ir_value()))
            dead_right = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.right.ir_value()))
            return KvLoopBounds(left=b.left, unmasked_lo=dead_lo, unmasked_hi=dead_hi, right=dead_right)
        return b

    @cute.jit
    def _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, scalar_seqlen_kv):
        if cutlass.const_expr(CFG.SEQ_KV_LENS_PRESENT == 1):
            arr = cutlass.make_array_view(seq_kv_lens_tensor)
            return cutlass.Int32(arr[cutlass.Int32(batch_idx)])  # batch_idx may be a raw arith value after a payload read
        return scalar_seqlen_kv

    @cute.jit
    def _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, scalar_seqlen_q, n_batch, seq_q_lens_addr=0):
        """Per-batch Q length for the bottom-right causal diagonal.

        Bottom-right anchors the diagonal at the per-batch corner
        (seq_len_q[b], seq_len_kv[b]).  THD reads the actual Q length as the
        cu_seqlen_q difference from the packed [kv_lens | cu_q | cu_kv] metadata
        buffer (same layout _thd_decode / _thd_tma_offsets read); dense padded
        graphs carrying per-batch Q lengths read the SEPARATE (B,)-int32
        ``seq_q_lens_addr`` (cuDNN SEQLEN_Q style), clamped to [0, S_q].
        KV-only padding (and every non-BR mask, where seqlen_q only feeds the
        unused diagonal) keeps the scalar S_q, so both reads fold out unless
        CAUSAL_BOTTOM_RIGHT is set together with THD_VARLEN or
        SEQ_Q_LENS_PRESENT (mutually exclusive by _validate_params).
        """
        if cutlass.const_expr(int(getattr(CFG, "THD_VARLEN", 0)) == 1 and int(CFG.BOTTOM_RIGHT) == 1):
            cu = cutlass.make_array_view(seq_kv_lens_tensor)
            q0 = n_batch
            return cutlass.Int32(cu[q0 + batch_idx + cutlass.Int32(1)]) - cutlass.Int32(cu[q0 + batch_idx])
        if cutlass.const_expr(int(getattr(CFG, "SEQ_Q_LENS_PRESENT", 0)) == 1 and int(CFG.BOTTOM_RIGHT) == 1):
            arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            return cute.math.max(cutlass.Int32(0), cute.math.min(cutlass.Int32(arr[cutlass.Int32(batch_idx)]), scalar_seqlen_q))
        return scalar_seqlen_q

    _thd_on = int(getattr(CFG, "THD_VARLEN", 0))

    @cute.jit
    def _thd_decode(linear_cta, seq_kv_lens_t, n_batch, n_qh, cta_in_pair):
        u = linear_cta // cutlass.Int32(CFG.CGA_M)
        cu = cutlass.make_array_view(seq_kv_lens_t)
        cuq0 = n_batch
        acc = cutlass.Int32(0)
        # DEAD-unit sentinel (issue #552 over-launch): a unit no sequence
        # claims (u >= sum of live units) keeps batch == n_batch.  That index
        # makes every downstream consumer a no-op through IN-BOUNDS metadata
        # reads: _resolve_seqlen_kv reads meta[n_batch] = cu_q[0] = 0 (empty
        # KV range in every role), the epilogue's per-sequence Q length
        # cu[2n+1]-cu[2n] goes negative (LSE predicate never fires), and the
        # O-store role skips the TMA store explicitly (batch >= n_batch).
        f_batch = n_batch
        f_head = cutlass.Int32(0)
        f_qc = cutlass.Int32(0)
        done = cutlass.Int32(0)
        # Sequences are visited LONGEST FIRST through batch_remap (built by the
        # THD setup launch), so the ragged tail of the grid is short sequences.
        remap0 = cutlass.Int32(3) * n_batch + cutlass.Int32(2)
        for i in cutlass.range(0, n_batch, 1, unroll=1):
            b = cutlass.Int32(cu[remap0 + i])
            s_i = cutlass.Int32(cu[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[cuq0 + b])
            cb = (s_i + cutlass.Int32(cga_tile_m - 1)) // cutlass.Int32(cga_tile_m)
            units_b = cb * n_qh
            # A zero-length sequence gives cb == 0, and units_b == 0 with it, so
            # in_rng is false and the quotient is discarded — but arith.select
            # evaluates BOTH arms, so divide by a clamped copy to keep the dead
            # arm defined. units_b keeps the true cb (thd_decode_unit's tb_nz).
            cb_nz = cute.math.max(cb, cutlass.Int32(1))
            in_rng = (done == cutlass.Int32(0)) & (u < acc + units_b)
            local = u - acc
            # Natural order within a sequence (head-major, ascending rows).
            f_batch = cutlass.Int32(arith.select(in_rng.ir_value(), b.ir_value(), f_batch.ir_value()))
            f_head = cutlass.Int32(arith.select(in_rng.ir_value(), (local // cb_nz).ir_value(), f_head.ir_value()))
            f_qc = cutlass.Int32(arith.select(in_rng.ir_value(), (local % cb_nz).ir_value(), f_qc.ir_value()))
            done = cutlass.Int32(arith.select(in_rng.ir_value(), cutlass.Int32(1).ir_value(), done.ir_value()))
            acc = acc + units_b
        q_super = f_qc * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
        return q_super, f_head, f_batch

    @cute.jit
    def _dispatch_decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        if cutlass.const_expr(_thd_on):
            return _thd_decode(bidx, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
        if cutlass.const_expr(CFG.PACK_GQA):
            qh_per_kh = cutlass.Int32(1)
        return _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh, seqlen_kv)

    @cute.jit
    def _dispatch_decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        if cutlass.const_expr(_thd_on):
            return _thd_decode(t0, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
        if cutlass.const_expr(CFG.PACK_GQA):
            qh_per_kh = cutlass.Int32(1)
        return _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh, seqlen_kv)

    @cute.jit
    def _thd_tma_offsets(seq_kv_lens_t, batch_idx, n_batch):
        if cutlass.const_expr(_thd_on):
            cu = cutlass.make_array_view(seq_kv_lens_t)
            q_off = cutlass.Int32(cu[n_batch + batch_idx])
            kv_off = cutlass.Int32(cu[cutlass.Int32(2) * n_batch + cutlass.Int32(1) + batch_idx])
            return q_off, kv_off, cutlass.Int32(0)
        return cutlass.Int32(0), cutlass.Int32(0), batch_idx

    @cute.jit
    def _thd_sf_tile_bases(seq_kv_lens_t, batch_idx, n_batch):
        if cutlass.const_expr(_thd_on):
            cu = cutlass.make_array_view(seq_kv_lens_t)
            q0 = n_batch
            k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
            sfq = cutlass.Int32(0)
            sfk = cutlass.Int32(0)
            for b in cutlass.range(0, batch_idx, 1, unroll=1):
                s_q = cutlass.Int32(cu[q0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[q0 + b])
                s_kv = cutlass.Int32(cu[k0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[k0 + b])
                sfq = sfq + (s_q + cutlass.Int32(CFG.TILE_M - 1)) // cutlass.Int32(CFG.TILE_M)
                sfk = sfk + (s_kv + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N)
            return cute.arch.make_warp_uniform(sfq), cute.arch.make_warp_uniform(sfk)
        return cutlass.Int32(0), cutlass.Int32(0)

    return SdpaHelpers(
        decode_initial=_decode_initial,
        decode_payload=_decode_payload,
        bounds_for_tile=_bounds_for_tile,
        bounds_for_tile_qtrim=_bounds_for_tile_qtrim,
        resolve_seqlen_kv=_resolve_seqlen_kv,
        resolve_seqlen_q=_resolve_seqlen_q,
        thd_decode=_thd_decode,
        dispatch_decode_initial=_dispatch_decode_initial,
        dispatch_decode_payload=_dispatch_decode_payload,
        thd_tma_offsets=_thd_tma_offsets,
        thd_sf_tile_bases=_thd_sf_tile_bases,
    )


# =============================================================================
# Fused epilogue gate  --  O := O * sigmoid(G)   (Rubin d256 f16 / per-tensor FP8)
# =============================================================================
# Shared between the two SM107 d256 prefill kernels (``sm107/prefill_d256_f16.py``,
# ``sm107/prefill_d256_fp8.py``); each kernel splices these under
# ``cutlass.const_expr(CFG.EPILOGUE_GATE)`` at its sixteen ``EPILOGUE_FUSION_SEAM``
# tokens (one per splice; a source test counts them per kernel).  The gate ``SmemTile(`` itself is constructed INLINE in each kernel
# (multi-line, ``desc_version=DESC_VERSION``) so the per-kernel "every SmemTile
# takes the module DESC_VERSION" source test keeps counting truthfully; this
# module owns the geometry, the barriers, the TMA issue, the chunk-offset
# arithmetic and the packed epilogue math.
#
# The gate tile is per-CTA and full-width even under cga2: the decode folds
# ``cta_in_pair`` into ``q_super_idx``, so each CTA of a pair owns a distinct
# 128-row Q block and all TILE_O columns (only K rows / V columns split per
# CTA).  Hence ``cta_group=1`` and NO ``* CFG.CTA_MMA`` on the transaction bytes
# -- contrast ``qTmaTransactionBytes = qBufferElems * BPE * CTA_MMA`` (P9).


class GateGeometry(NamedTuple):
    """Host-side gate staging geometry, derived from GATE's OWN byte width.

    Never from O's: on the FP8 kernel with e4m3 O the two differ (O walks 2
    subtiles of 128 elems; the bf16 gate walks 4 of 64), and a mis-derivation
    passes every bf16-O test and fails only e4m3-O."""

    swz_bytes: int  # o_swz_bytes(TILE_O, gate_bpe)            -> 128 at d256 / 2 B
    tma_iters: int  # (TILE_O * gate_bpe) // swz_bytes         -> 4 subtiles
    tma_granu_elems: int  # swz_bytes // gate_bpe                   -> 64 elems per subtile row
    buffer_elems: int  # TILE_M * TILE_O (ONE q tile per load)   -> 32768 elems (1 stage)
    tx_bytes: int  # buffer_elems * gate_bpe                 -> 65536 B, NO * CTA_MMA
    box_dims: tuple  # (1, TILE_M, 1, tma_granu_elems)
    layout_enum: int  # the kernel's _SWZ_ENUM[swz_bytes]         (128: 2, 64: 4, 32: 6)
    smem_swizzle: object  # cutlass.Swizzle({128: 3, 64: 2, 32: 1}[swz_bytes], 4, 3)
    d_block: int  # TILE_O // tma_iters                     -> 64 elems per lane row inside one subtile
    granu_local: int  # TILE_M * d_block                        -> 8192 elems per subtile


def gate_geometry(CFG, *, gate_bpe: int, swz_enum: dict) -> GateGeometry:
    """Pure host arithmetic; called at module import by every d256 module.

    Swizzle has BOTH jobs here (frost-kernels.md S5): the TMA descriptor WRITES
    the tile under ``swz_bytes`` (so the SMEM side must match it -- correctness)
    AND correction lanes READ it one q row per lane at a ``d_block * gate_bpe``
    = 128 B stride (= the bank cycle, so an unswizzled tile would be a 32-way
    conflict).  ``Swizzle(3, 4, 3)`` has ``MBase + SShift = 7 = log2(128 B)``.
    """
    # ONE q tile per ``issue_gate_load`` call: ``tma_load_tile`` delivers
    # ``tma_iters * TILE_M * tma_granu_elems`` = ``TILE_M * TILE_O`` elements, and
    # ``expect_tx`` must equal exactly the bytes delivered (P3) -- so the buffer
    # and the transaction are sized per Q TILE, never per ``TILES_Q``.  A flavor
    # with TILES_Q > 1 (d128/d192) needs a per-tile issue loop + a per-tile
    # arm before the gate can be wired there; refuse at import rather than
    # over-arm the mbar (an over-count is a hang, not an error).
    if CFG.TILES_Q != 1:
        raise ValueError(f"gate_geometry: the epilogue gate stages ONE q tile per load (TILES_Q == 1); got TILES_Q={CFG.TILES_Q}")
    swz_bytes = _o_swz_bytes(CFG.TILE_O, gate_bpe)
    tma_iters = (CFG.TILE_O * gate_bpe) // swz_bytes
    tma_granu_elems = swz_bytes // gate_bpe
    buffer_elems = CFG.TILE_M * CFG.TILE_O
    d_block = CFG.TILE_O // tma_iters
    return GateGeometry(
        swz_bytes=swz_bytes,
        tma_iters=tma_iters,
        tma_granu_elems=tma_granu_elems,
        buffer_elems=buffer_elems,
        tx_bytes=buffer_elems * gate_bpe,
        box_dims=(1, CFG.TILE_M, 1, tma_granu_elems),
        layout_enum=swz_enum[swz_bytes],
        smem_swizzle=cutlass.Swizzle({128: 3, 64: 2, 32: 1}[swz_bytes], 4, 3),
        d_block=d_block,
        granu_local=CFG.TILE_M * d_block,
    )


@cute.jit
def issue_gate_load(sGate, tma_gate, mb_gate_full, tx_bytes: cutlass.Constexpr[int], head_idx, q_row, tma_batch):
    """TMA-LDG warp: arm ``expect_tx`` (ONE lane, P16 value predicate) and queue
    the gate tile's subtiles with ``cta_group=1`` so every byte lands on THIS
    CTA's mbar.  Folds to nothing when ``sGate`` is None (gate compiled out).

    Call it on BOTH arms of the empty-mainloop branch (P14): the epilogue runs
    on every tile -- empty-KV and padded-Q-trimmed ones included -- and waits
    ``mb_gate_full`` each time, so every tile must consume exactly one load.
    Issue position: AFTER the kv loop.  The TMA engine serves requests in
    order and the gate is the one operand with slack (consumed only in the
    epilogue); queued before K/V it delays the operands BMM1 is blocked on
    (measured -2.4 % on the fused kernel for the after-Q position).
    """
    if cutlass.const_expr(sGate is not None):
        mb_gate_full.arrive(n_bytes=tx_bytes, pred=nvvm.elect_sync())
        tma_load_tile(
            sGate,
            tma_gate(cutlass.Int32(0), head_idx, q_row, tma_batch),
            mb_gate_full.smem_ptr,
            cta_group=1,
        )


def gate_chunk_smem_offset(chunk_idx_total: int, o_chunk: int, gg: GateGeometry, tid_in_wg) -> cutlass.Int32:
    """Trace-time helper (plain Python, like ``row_max_reduction``): SMEM element
    offset of this lane's ``o_chunk``-wide gate chunk.  Subtile-major walk in
    GATE geometry (lane = one q row, ``d_block`` elems per row per subtile)."""
    c0 = chunk_idx_total * o_chunk
    return cutlass.Int32((c0 // gg.d_block) * gg.granu_local + (c0 % gg.d_block)) + tid_in_wg * cutlass.Int32(gg.d_block)


def load_gate_chunk(sGate_base, smem_offset, gg: GateGeometry, o_chunk: int) -> list:
    """Trace-time helper: one swizzled LDS of ``o_chunk`` gate elements, widened
    to fp32, as a Python list.

    Alignment 32 is the TRUTHFUL byte alignment of a 16-elem x 2 B chunk at
    offsets 0 / 32 / 64 / 96 B inside the 128 B row (64 would overstate the odd
    chunks); CuTe vectorises at 16 B either way (LDS.128 x 2 per chunk).
    Plain ``for`` + ``append``: ``range_constexpr`` is statement-only and a
    comprehension is not preprocessed (frost-gotchas)."""
    vec = sGate_base.subview(smem_offset).data_ptr().load_swizzled(gg.smem_swizzle, 32, count=o_chunk).to(cutlass.Float32)
    out = []
    for i in range(o_chunk):
        out.append(vec[i])
    return out


def gate_epilogue_pairs(o_scaled, g_vals: list, half_opaque, o_chunk: int) -> list:
    """Trace-time helper: the gate math, algebraically minimised and PACKED.

        sigmoid(g) = tanh(g/2)/2 + 1/2
        o*inv_sum*sigmoid(g) = h*tanh(g/2) + h        with h = o*(inv_sum/2)

    The caller has ALREADY folded the 1/2 into ``inv_sum`` (``gate_inv_sum``),
    so ``o_scaled`` IS ``h`` and both of sigmoid's constants are gone.  Per PAIR
    of elements: 1 FMUL2 (g/2) + 2 MUFU.TANH + 1 FFMA2 (h*t + h, multiplicand
    and addend the same register).  Verified in SASS as the delta against the
    gate compiled out (256 elems/tile): +128 FMUL2, +128 FFMA2, +256 MUFU.TANH,
    0 scalar FMUL/FFMA.  ``half_opaque`` must be ``gate_half_opaque()`` -- a
    constant float reaching ``inline_ptx`` ICEs libNVVM (frost-tile-dsl S7).

    The caller applies the dead-row SELECT per element AFTER these values
    (never a multiply-by-zero: the TMEM residue behind ``h`` can be NaN)."""
    out = []
    for p in range(o_chunk // 2):
        a, b = g_vals[2 * p], g_vals[2 * p + 1]
        ha, hb = o_scaled[2 * p], o_scaled[2 * p + 1]
        sa, sb = fmul2(a, b, half_opaque, half_opaque)
        ta = cute.math.tanh(sa, approx=True)
        tb = cute.math.tanh(sb, approx=True)
        va, vb = ffma2(ta, tb, ha, hb, ha, hb)
        out.append(va)
        out.append(vb)
    return out


def gate_inv_sum(inv_sum):
    """``inv_sum / 2`` -- folds sigmoid's 1/2 into the scale the epilogue applies anyway."""
    return inv_sum * cutlass.Float32(0.5)


def gate_half_opaque():
    """An opaque 0.5f for ``fmul2``: a constant float operand into inline_ptx
    gets the ``n`` immediate constraint and ICEs libNVVM (frost-tile-dsl S7).
    Hoist it once per tile, not per chunk."""
    return opaque_f32_zero() + cutlass.Float32(0.5)
