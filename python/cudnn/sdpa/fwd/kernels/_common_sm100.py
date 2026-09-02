# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.scheduler import (
    SCHED_LPT_L2,
    SCHED_NATURAL,
    lpt_tile_coords,
    lpt_l2_tile_coords,
)
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_PADDED, MASK_SWA
from cudnn.frost.tile_dsl.barrier import MBarrier, Producer, Scope


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


def make_d256_bars(CFG, *, N_O_CHUNKS: int) -> D256Bars:
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
    )


def make_classic_bars(CFG) -> Bars:
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
    )


class KvLoopBounds(NamedTuple):
    left: object
    unmasked_lo: object
    unmasked_hi: object
    right: object


def _div_up(a, b):
    return (a + cutlass.Int32(b - 1)) // cutlass.Int32(b)


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


def compute_kv_loop_bounds(
    q_row_coord,
    seqlen_q,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    tile_n: int,
    cga_tile_m: int,
    bottom_right: bool = False,
    window_right: int = 0,
) -> KvLoopBounds:
    # window_right: compile-time diagonal-band right bound (cuDNN
    # diagonal_band_right_bound) — the causal upper limit is widened by
    # window_right columns. 0 = plain causal; folds out entirely.
    left = cutlass.Int32(0)
    right = _div_up(seq_kv_len, tile_n)

    if cutlass.const_expr(bottom_right):
        causal_diag = seq_kv_len - seqlen_q
    else:
        causal_diag = cutlass.Int32(0)

    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        kv_hi_caus = _div_up(q_row_coord + cutlass.Int32(cga_tile_m + window_right) + causal_diag, tile_n)
        right = cute.math.min(right, kv_hi_caus)

    if cutlass.const_expr(mask_flags & MASK_SWA):
        # The whole band shifts with the diagonal: under BOTTOM_RIGHT the SWA
        # lower bound is q + (S_kv - S_q) - W, same anchor the causal upper
        # bound uses (causal_diag folds to 0 for top-left).
        swa_base = q_row_coord + causal_diag
        cond = swa_base > cutlass.Int32(window_left)
        delta = swa_base - cutlass.Int32(window_left)
        kv_lo_swa = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                (delta // cutlass.Int32(tile_n)).ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        left = cute.math.max(left, kv_lo_swa)

    unmasked_hi = right
    if cutlass.const_expr(mask_flags & MASK_PADDED):
        unaligned = (seq_kv_len % cutlass.Int32(tile_n)) != cutlass.Int32(0)
        lo_pad = cutlass.Int32(
            arith.select(
                unaligned.ir_value(),
                (right - cutlass.Int32(1)).ir_value(),
                right.ir_value(),
            )
        )
        unmasked_hi = cute.math.min(unmasked_hi, lo_pad)
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        lo_caus = (q_row_coord + cutlass.Int32(window_right) + causal_diag) // cutlass.Int32(tile_n)
        unmasked_hi = cute.math.min(unmasked_hi, lo_caus)
    unmasked_hi = cute.math.max(unmasked_hi, left)

    unmasked_lo = left
    if cutlass.const_expr(mask_flags & MASK_SWA):
        anchor = q_row_coord + causal_diag + cutlass.Int32(cga_tile_m - 1 - window_left)
        swa_unmasked_lo = _div_up(anchor, tile_n)
        cond = anchor > cutlass.Int32(0)
        swa_unmasked_lo = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                swa_unmasked_lo.ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        unmasked_lo = cute.math.max(unmasked_lo, swa_unmasked_lo)

    unmasked_lo = cute.math.min(unmasked_lo, unmasked_hi)

    return KvLoopBounds(
        left=left,
        unmasked_lo=unmasked_lo,
        unmasked_hi=unmasked_hi,
        right=right,
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
    ``(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_tensor,
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
    # NOTE this is the flavor's EFFECTIVE policy, not the requested one:
    # make_cfg_d192 hardcodes SCHEDULER_POLICY=1 regardless of params, so a
    # params-level check would miss it (and did -- d192 silently launched the
    # unsplit grid because the split multiplier was only on the NATURAL branch).
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
    def _bounds_for_tile_split(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, split_idx, qh_per_kh: int = 1):
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
        b = bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, qh_per_kh)
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
    def _bounds_for_tile_qtrim(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, qh_per_kh: int = 1):
        """bounds_for_tile + cuDNN-style dead-Q-tile KV-loop collapse.

        Mirrors cuDNN fort (mma_pipeline_op_native_sdpa_prefill_sm100_nonfp8
        .cpp:916-921): when the CGA tile's base Q row is at/past this batch's
        actual Q length (SEQ_Q_LENS_PRESENT dense padded-Q trim; q lens are
        the SEPARATE (B,)-int32 ``seq_q_lens_tensor`` kernel parameter — cuDNN
        SEQLEN_Q / FA seqused_q style — ``None`` unless the flag is set, so
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
            arr = cutlass.make_array_view(seq_q_lens_tensor)
            q_len_b = cutlass.Int32(arr[batch_idx])
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
            return cutlass.Int32(arr[batch_idx])
        return scalar_seqlen_kv

    @cute.jit
    def _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, scalar_seqlen_q, n_batch, seq_q_lens_tensor=None):
        """Per-batch Q length for the bottom-right causal diagonal.

        Bottom-right anchors the diagonal at the per-batch corner
        (seq_len_q[b], seq_len_kv[b]).  THD reads the actual Q length as the
        cu_seqlen_q difference from the packed [kv_lens | cu_q | cu_kv] metadata
        buffer (same layout _thd_decode / _thd_tma_offsets read); dense padded
        graphs carrying per-batch Q lengths read the SEPARATE (B,)-int32
        ``seq_q_lens_tensor`` (cuDNN SEQLEN_Q style), clamped to [0, S_q].
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
            arr = cutlass.make_array_view(seq_q_lens_tensor)
            return cute.math.max(cutlass.Int32(0), cute.math.min(cutlass.Int32(arr[batch_idx]), scalar_seqlen_q))
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
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            s_i = cutlass.Int32(cu[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[cuq0 + b])
            cb = (s_i + cutlass.Int32(cga_tile_m - 1)) // cutlass.Int32(cga_tile_m)
            units_b = cb * n_qh
            in_rng = (done == cutlass.Int32(0)) & (u < acc + units_b)
            local = u - acc
            f_batch = cutlass.Int32(arith.select(in_rng.ir_value(), b.ir_value(), f_batch.ir_value()))
            f_head = cutlass.Int32(arith.select(in_rng.ir_value(), (local // cb).ir_value(), f_head.ir_value()))
            f_qc = cutlass.Int32(arith.select(in_rng.ir_value(), (local % cb).ir_value(), f_qc.ir_value()))
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
