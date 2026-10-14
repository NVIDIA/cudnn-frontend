# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions adapted from prefill_d512_f16.py (the SM90 D512 template) retain its MIT terms; the rest of this
# file is licensed under Apache-2.0. See LICENSING.md and THIRD_PARTY_LICENSES.txt.

"""
A fused multi-head attention (FMHA) FP16/BF16 prefill kernel for D=128 heads on NVIDIA
Hopper SM90 (sm_90a), on WGMMA tensor cores.

The CTA tile is (128, 128): 128 query rows by 128 keys over the 128-column head. Two
compute warpgroups each own 64 of the rows and every head column, so each runs its own QK
(m64n128k16 SS, one commit group of 8), online softmax and PV (m64n128k16 RS, one group of
8), with no handoff between them. A 64-row FP32 O accumulator over 128 columns costs 64
registers per thread.

The kernel implements:
- Three warpgroups per CTA. WG0 issues the TMA loads. WG1 and WG2 run QK, the online
  softmax and PV on query rows 0-63 and 64-127 of the tile.
- Two-stage K and V rings, so the producer loads KV tile j+1 while tile j is in use.
- One work tile per CTA over (Q tile, head, batch), in NATURAL, LPT or LPT_L2 order.
- Online softmax around an exact sign-aware row anchor, with lane-partial row sums
  quad-reduced only at finalization: the D512 template's arithmetic at 32 keys per lane
  and row.
- Separate positive, zero and negative scale paths, plus an FP64 difference path for
  extreme scales.
- A reverse KV walk: boundary tiles mask elements, interior tiles do not.
- Causal masks (top-left or bottom-right), MHA, GQA and MQA (unpacked), and per-row LSE
  (Stats) in natural log or, under ``stats_log2``, base 2.

Storage: Q and each K and V ring stage are 128 x 128 tiles of two 128-row SW128 slabs; a
compute warpgroup's QK reads rows 64*w .. 64*w+63 of the Q slabs. O stages through its own
region as two 64-row halves of two slabs, one per compute warpgroup, so each stores its rows
alone. 196,704 B in all.

Constraints:
- FP16/BF16 in, O in the input dtype, FP32 accumulation.
- D_QK = D_V = 128 exactly; the maps carry the extents, so a partial last Q or KV tile
  zero-fills on load and clips on store.
- Q/K/V/O keep their declared strides: D contiguous, the rest 16-byte multiples.
- Dense only. THD, sinks, sliding windows, widened right bands, per-batch lengths and
  PackGQA run on the D512 template (``config_sm90.D128_NOT_SERVED``).
- ``SdpaFwdDslSm90.check_support`` owns the served set.

WGMMA instructions and descriptors live in FROST ``tile_dsl/wgmma.py``, the shared tile
operations and softmax arithmetic in ``sm90/_common_hopper.py``.
"""

from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, NamedTuple, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from cudnn.frost.compiled_cache import template_key as _template_key
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, MASK_CAUSAL, MASK_NONE, MASK_PADDED
from cudnn.frost.tile_dsl.handles import GmemTileTmaSlice, SmemTile
from cudnn.frost.tile_dsl.mask import compute_kv_loop_bounds
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, opaque_f32_zero
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.scheduler import SCHED_LPT_L2, SCHED_NATURAL, lpt_l2_tile_coords, lpt_tile_coords
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.sdpa.fwd.config_sm90 import D128_FLAVOR, FLAVOR_TILES, SCALE_POSITIVE, SCALE_ZERO, TemplateParams, validate_params
from cudnn.sdpa.fwd.kernels.sm90.prepared_host import host as _host, compile_host
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import (
    FLT_MAX_F64,
    LOG2_E,
    SCHED_L2_BUDGET_BYTES,
    mma_pv,
    mma_qk,
    normalize_row,
    owned_row,
    quad_reduce,
    rescale_output,
    score_prefix_mask,
    store_fragment,
    wgmma_descriptor,
)

# The FROST loader injects one immutable specialization before executing this module. A
# direct import uses the dense FP16 defaults.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
validate_params(PARAMS, flavor=D128_FLAVOR)

# As the D512 template. The first two are read only by the THD launch spec, which never
# reaches this template.
PREPARED_FIXED_BATCH = True
PREPARED_WORKSPACE_ALIGNMENT = 128
PREPARED_DENSE_FLEX = True
PREPARED_FIXED_SHAPE = True
PREPARED_KV_TAIL_NATIVE = True

STORAGE_DTYPE = {DTYPE_FP16: cutlass.Float16, DTYPE_BF16: cutlass.BFloat16}[PARAMS.dtype_qkv]


@dsl_user_op
def warpgroup_barrier_sync(barrier_id: int, thread_count: int, *, loc=None, ip=None):
    """A named barrier among the warps of one warp-role branch (a compute warpgroup's O staging): a subset barrier, so it
    lowers without the aligned-CTA promise (barrier.sync, not bar.sync / barrier.sync.aligned), as python/cudnn/AGENTS.md
    asks of a named barrier reached from a warp-role branch. ``barrier_id`` and ``thread_count`` are compile-time."""
    nvvm.barrier_cta_sync(
        cutlass.Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=cutlass.Int32(thread_count).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def score_prefix_mask_128(bound, column_base, inclusive: cutlass.Constexpr = False):
    """Bitmap for this lane's 32 scores of a 128-key tile below a column bound (optionally inclusive).

    Bit ``2*k_frag + j`` is the score in C slot ``4*k_frag + 2*row_half + j``, key
    ``column_base + 8*k_frag + j``. ``score_prefix_mask`` covers one 64-key window in 16 bits;
    the second window starts 64 keys on and fills bits 16-31.
    """
    low = score_prefix_mask(bound, column_base, inclusive=inclusive)
    high = score_prefix_mask(bound, column_base + 64, inclusive=inclusive)
    return low | (high << cutlass.Uint32(16))


class Bars(NamedTuple):
    """Hold one barrier family per field, in allocation order.

    ``kernel`` allocates and initializes the record. Producer -> consumers, with each
    arrival count:

    - ``mb_tma_q_full``: WG0's elected expect-tx arrival plus the TMA bytes of the Q tile
      -> WG1, WG2.
    - ``mb_tma_q_empty``: the 8 warps of WG1 and WG2, at their epilogues.
    - ``mb_tma_{k,v}_full``: KV_STAGES slots; WG0's elected expect-tx arrival plus the TMA
      bytes -> WG1, WG2.
    - ``mb_tma_{k,v}_empty``: KV_STAGES slots; the 8 warps of WG1 and WG2, each after its
      QK (K) or PV (V) on that slot.

    Consumers wait FULL from phase 0, producers EMPTY from phase 1, which passes at once
    on a fresh barrier.
    """

    mb_tma_q_full: object
    mb_tma_q_empty: object
    mb_tma_k_full: object
    mb_tma_k_empty: object
    mb_tma_v_full: object
    mb_tma_v_empty: object


class SM90FusedMultiHeadAttentionForward:
    """Configure and launch the SM90 FP16/BF16 D128 FMHA prefill kernel.

    Each CTA runs one work tile through three warpgroups: WG0 loads, WG1 and WG2 each
    compute 64 query rows. The constructor takes only what ``config_sm90.validate_params``
    lets a D128 specialization declare; the prepared host still reads ``thd_varlen``,
    ``has_sink``, ``thd_lse_head_stride`` and ``thd_origins``, which stay off.
    """

    # The CTA tile: Q_TILE rows by KV_TILE keys over the HEAD_TILE-column head.
    Q_TILE, KV_TILE = FLAVOR_TILES[D128_FLAVOR]
    HEAD_TILE = D128_FLAVOR[0]
    # Query rows per compute warpgroup: one m64 WGMMA row block.
    WG_ROWS = 64
    KV_STAGES = 2
    # setmaxnreg budgets per warpgroup: 128 * (24 + 240 + 240) fits the SM's 65,536.
    REG_BUDGETS = (24, 240, 240)

    def __init__(
        self,
        shape: tuple[int, int, int, int, int],
        in_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        out_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        is_causal: bool = False,
        sched_policy: int = SCHED_NATURAL,
        bottom_right: bool = False,
        qh_per_kh: int = 1,
        scale_mode: int = SCALE_POSITIVE,
        stats_log2: bool = False,
    ):
        """Initialize the SM90 D128 FMHA prefill kernel configuration.

        :param shape: The declared problem ``(b, h_q, h_kv, s_q, s_kv)``, fixed at plan
            time. ``b``, ``h_q`` and ``s_q`` size the grid.
        :param in_dtype: Q/K/V element type (Float16 or BFloat16).
        :param out_dtype: O element type; must match ``in_dtype``.
        :param is_causal: Apply the causal upper bound to QK (``MASK_CAUSAL``).
        :param sched_policy: CTA order over the (Q tile, head, batch) tiles.
            ``SCHED_NATURAL`` launches the 3-D grid; ``SCHED_LPT`` and ``SCHED_LPT_L2``
            launch one axis that ``kernel`` unflattens.
        :param bottom_right: Anchor the diagonal at ``delta = s_kv - s_q`` instead of 0.
        :param qh_per_kh: The GQA ratio, refused unless ``h_q == h_kv * qh_per_kh``. Q head
            ``h`` reads KV head ``h // qh_per_kh``.
        :param scale_mode: The sign of the softmax scale: ``SCALE_POSITIVE``, ``SCALE_ZERO``
            or ``SCALE_NEGATIVE``. Each sign compiles its own probability path, and a
            negative scale anchors each row at its minimum raw score. The runtime ``scale``
            must carry this sign.
        :param stats_log2: Store Stats in base 2: the store multiplies each natural-log row
            LSE by ``log2(e)``; ``-inf`` stays ``-inf``, and ``lse`` None removes it.
        """

        _, h_q, h_kv, _, _ = shape
        if out_dtype != in_dtype:
            raise ValueError("prefill_d128_f16_sm90: out_dtype must match in_dtype")
        if h_q != h_kv * qh_per_kh:
            raise ValueError(f"prefill_d128_f16_sm90: KV-head addressing needs H_q == H_kv * qh_per_kh; got {h_q}, {h_kv}, {qh_per_kh}")
        self.b, self.h_q, self.h_kv, self.s_q, self.s_kv = shape
        self.storage_dtype = in_dtype
        self.out_dtype = out_dtype
        self.sched_policy = sched_policy
        self.bottom_right = bottom_right
        self.mask_flags = MASK_CAUSAL if is_causal else MASK_NONE
        # Without a mask, compute_kv_loop_bounds never reads the diagonal, so this flag is
        # inert there and True is the value it keeps.
        self.kv_loop_bottom_right = self.bottom_right if is_causal else True
        self.qh_per_kh = qh_per_kh
        self.scale_mode = scale_mode
        # The anchor is the raw-score extremum that maximizes the scaled score: the maximum
        # for a non-negative scale, the minimum for a negative one, with the sentinel as its
        # identity.
        self.anchor_sentinel = -cutlass.Float32.inf if scale_mode >= SCALE_ZERO else cutlass.Float32.inf
        self.stats_log2 = stats_log2
        # Read by prepared_host.host; a D128 specialization never declares them (no THD, sink,
        # paged pool or KV split: validate_params refuses each).
        self.thd_varlen = False
        self.has_sink = False
        self.thd_lse_head_stride = 0
        self.thd_origins = None
        self.page_size = 0
        self.split_kv = 1

        # Named-barrier ids. The init publish takes barrier_cta_sync's default id 0, so the
        # named ones start at 1: id 1 + w orders compute warpgroup w's four O-staging warps
        # before its TMA store.
        self.bar_o_stage_base = 1

        self.threads_per_cta = len(self.REG_BUDGETS) * cute.arch.THREADS_PER_WARPGROUP

        self._setup_attributes()

    def _setup_attributes(self):
        """Compute derived tile, MMA, SMEM, and TMA constants."""

        self.q_tiles = (self.s_q + self.Q_TILE - 1) // self.Q_TILE
        self.heads = self.h_q
        # Every launched Q row is real only when the declared Q length is whole tiles;
        # online_softmax's row-validity test and the partial-tile walk fold away.
        self.q_rows_full = self.s_q % self.Q_TILE == 0
        # SMEM elements: one Q_TILE x HEAD_TILE tile sizes Q, O and each K/V stage, and a
        # compute warpgroup's O half is WG_ROWS x HEAD_TILE.
        self.tile_elems = self.Q_TILE * self.HEAD_TILE
        self.wg_tile_elems = self.WG_ROWS * self.HEAD_TILE

        # MMA, per 128-thread warpgroup: a thread's QK C slots per row, and its O C slots.
        self.qk_row_slots = self.WG_ROWS * self.KV_TILE // cute.arch.THREADS_PER_WARPGROUP // 2
        self.o_fragment_slots = self.WG_ROWS * self.HEAD_TILE // cute.arch.THREADS_PER_WARPGROUP
        # One K or V stage in the descriptors' 16-byte start-field units.
        self.kv_stage_units = self.tile_elems * self.storage_dtype.bytes // 16

        # TMA: every region moves in SW128 slabs of 128-byte rows, two per 128-column tile.
        # The WGMMA descriptors (wgmma_descriptor) assume exactly this atom. Q, K and V boxes
        # are 128 rows (a Q box starts at q_base, inside Q, so it is never wholly out of
        # bounds); O boxes are WG_ROWS rows, one per compute warpgroup. __call__ builds the
        # maps themselves, over the caller's strides.
        self.tma_swizzle = cuda.TensorMapSwizzle.s128b
        self.swizzle_chunk_elems = 128 // self.storage_dtype.bytes
        self.tma_swizzle_chunks = self.HEAD_TILE // self.swizzle_chunk_elems
        self.tile_slab_elems = self.Q_TILE * self.swizzle_chunk_elems
        self.o_slab_elems = self.WG_ROWS * self.swizzle_chunk_elems

    def tma_slice(self, tma_desc, head_idx, seq_coord, batch_idx, tma_order):
        """Return one TMA box at view coordinates ``(batch, head, token, d=0)``.

        The coordinates are listed in the map's own axis order: ``tma_order`` is the port's
        ``self.tma_orders`` entry, the view modes ``__call__`` listed innermost first.
        """
        view = (batch_idx, head_idx, seq_coord, cutlass.Int32(0))
        return GmemTileTmaSlice(tma_desc, tuple(view[mode] for mode in tma_order))

    @cute.jit
    def load_q_tile_tma(self, sQ: SmemTile, tma_q_desc: cutlass.GridConstant[cuda.TensorMap], mbar: MBarrier, work) -> None:
        """Launch the TMA loads for the resident Q tile into its two 128-row SW128 slabs.

        The producer warp calls this after the caller's EMPTY wait; the consumers' FULL wait
        completes the transfer. The box starts at ``q_base``, inside Q, so a partial last Q
        tile zero-fills its tail rows and is never wholly out of bounds.
        """
        # Keep this construction above the arrive, as the D512 template does for its Q box.
        source = self.tma_slice(tma_q_desc, work.head, work.q_base, work.batch, self.tma_orders[0])
        # One elected arrival expects every copy's box bytes; zero-filled out-of-bounds
        # elements still count.
        mbar.arrive(n_bytes=tma_q_desc.global_tx_bytes() * sQ.tma_loads_per_tile, pred=prims.elect_sync())
        tma_load_tile(sQ, source, mbar.smem_ptr)

    @cute.jit
    def load_one_kv_tile(
        self,
        s_dst: SmemTile,
        tma_desc: cutlass.GridConstant[cuda.TensorMap],
        mbar: MBarrier,
        work,
        kv_seq_idx: cutlass.Int32,
        is_v: cutlass.Constexpr[bool],
    ) -> None:
        """Launch the TMA loads for one whole K or V tile into one ring stage's two slabs.

        :param s_dst: The ring stage, ``sK[stage]`` or ``sV[stage]``.
        :param tma_desc: The port's grid-constant map.
        :param mbar: The stage's FULL barrier.
        :param work: The decoded work tile, read for its KV head and batch.
        :param kv_seq_idx: The first token of the box, from the caller's reverse KV walk.
        :param is_v: Select the V port's map axis order over the K one.
        """
        source = self.tma_slice(tma_desc, work.kv_head, kv_seq_idx, work.batch, self.tma_orders[2 if is_v else 1])
        mbar.arrive(n_bytes=tma_desc.global_tx_bytes() * s_dst.tma_loads_per_tile, pred=prims.elect_sync())
        tma_load_tile(s_dst, source, mbar.smem_ptr)

    @cute.jit
    def tmaldg_warp_group(self, sQ, sK, sV, bars, tma_q_desc, tma_k_desc, tma_v_desc, work):
        """Load resident Q once, then the reverse K/V tiles into the two-stage rings.

        Warp 0 of producer warpgroup WG0 owns the waits, the reverse walk and the drain.
        Each KV step refills one stage of each ring, each after its own EMPTY wait. The
        closing waits walk Q's one generation and then the WHOLE K/V ring from the current
        state (never ``min(iterations, KV_STAGES)`` steps): unused stages pass at their
        pre-armed phase and used ones wait their last release, so no TMA payload is in
        flight at exit.
        """
        tid = cute.arch.thread_idx()[0]
        if (work.q_base < work.q_len) & (work.iterations > 0):
            if tid < cute.arch.WARP_SIZE:
                # EMPTY waits start at phase 1, which passes at once, so the first trips
                # need no branch. One state serves K and V, which refill together.
                q_empty_state = PipelineState.start(phase=1)
                kv_empty_state = PipelineState.start(phase=1)
                bars.mb_tma_q_empty.wait(q_empty_state.phase)
                q_empty_state = advance(q_empty_state, 1)
                self.load_q_tile_tma(sQ, tma_q_desc, bars.mb_tma_q_full, work)
                for kv_step in cutlass.range(work.iterations, unroll=1):
                    kv_seq_idx = (work.kv_right - 1 - kv_step) * self.KV_TILE
                    stage = kv_empty_state.idx
                    bars.mb_tma_k_empty[stage].wait(kv_empty_state.phase)
                    self.load_one_kv_tile(sK[stage], tma_k_desc, bars.mb_tma_k_full[stage], work, kv_seq_idx, is_v=False)
                    bars.mb_tma_v_empty[stage].wait(kv_empty_state.phase)
                    self.load_one_kv_tile(sV[stage], tma_v_desc, bars.mb_tma_v_full[stage], work, kv_seq_idx, is_v=True)
                    kv_empty_state = advance(kv_empty_state, self.KV_STAGES)
                bars.mb_tma_q_empty.wait(q_empty_state.phase)
                for _ in cutlass.range_constexpr(self.KV_STAGES):
                    bars.mb_tma_k_empty[kv_empty_state.idx].wait(kv_empty_state.phase)
                    bars.mb_tma_v_empty[kv_empty_state.idx].wait(kv_empty_state.phase)
                    kv_empty_state = advance(kv_empty_state, self.KV_STAGES)

    @cute.jit
    def online_softmax(
        self,
        basic_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        s_regs: cutlass.Array,
        kv_seq_idx: cutlass.Int32,
        active_masks: cutlass.Constexpr,
        is_first_kv_tile: cutlass.Constexpr,
    ):
        """Update this lane's two rows and pack P into consumed score slots.

        The D512 template's ``online_softmax`` at 32 keys per lane and row: ``basic_params``
        supplies the fixed work/lane references and the warpgroup's first query row;
        ``softmax_params`` the mutable row state, this tile's alpha pair and the
        launch-uniform scales and guards. ``s_regs`` arrives with completed QK scores.

        Per row half r, ``anchor[r]`` is the exact sign-aware raw-score extremum, the
        sentinel while the row has no valid score, and ``partial_sum[r]`` this lane's
        FP32 share relative to ``exp(scale * anchor[r])``. After the first tile each
        step also yields ``alpha[r]``, which corrects O.

        C slot 4*k_frag+2*row_half+{0,1} owns row 16*warp+lane//4+8*row_half and key
        kv_seq_idx+8*k_frag+2*(lane%4)+{0,1}. Four adjacent lanes share a row, so their
        anchors and alpha agree and their complementary sums merge only at finalization.
        ``MASK_NONE`` requires every score in the tile to be valid.

        :return: ``s_regs``, with this tile's P packed into the consumed score slots.
        """
        work, local_tid = basic_params.work, basic_params.local_tid
        row_anchor, row_sum, alpha = softmax_params.anchor, softmax_params.partial_sum, softmax_params.alpha
        scale_log2, scale_log2_wide = softmax_params.scale_log2, softmax_params.scale_log2_wide
        extremum, quad_op = (cute.arch.fmax, "max") if cutlass.const_expr(self.scale_mode >= SCALE_ZERO) else (cute.arch.fmin, "min")
        sentinel, zero = cutlass.Float32(self.anchor_sentinel), cutlass.Float32(0.0)
        # A masked-out P stays a runtime value. On a dense unmasked graph the first
        # tile's padding mask folds, and a literal 0.0 would reach fp32_to_fp16's inline
        # PTX as an immediate, which ICEs libNVVM (pointwise.opaque_f32_zero).
        p_zero = opaque_f32_zero() if cutlass.const_expr(active_masks != MASK_NONE) else zero

        for row_half in cutlass.range_constexpr(2):
            s_reg_idx_lo = row_half * 2
            s_reg_idx_hi = row_half * 2 + 1

            # Resolve this row's validity once; each bit names one explicit score slot.
            row_live = cutlass.Boolean(True)
            column_mask = cutlass.Uint32(0xFFFFFFFF)
            if cutlass.const_expr(active_masks != MASK_NONE):
                q_token = basic_params.row_base + owned_row(local_tid, row_half)
                if cutlass.const_expr(not self.q_rows_full):
                    row_live = q_token < work.q_len
                diagonal = q_token + work.kv_len - work.q_len if cutlass.const_expr(self.bottom_right) else q_token
                column_base = kv_seq_idx + 2 * (local_tid % 4)
                if cutlass.const_expr(active_masks & MASK_PADDED):
                    column_mask = score_prefix_mask_128(work.kv_len, column_base)
                if cutlass.const_expr(active_masks & MASK_CAUSAL):
                    column_mask = column_mask & score_prefix_mask_128(diagonal, column_base, inclusive=True)

            # Fold query validity into the lane's bitmap once for all score/P uses.
            column_mask = column_mask if row_live else cutlass.Uint32(0)

            # Gather masked pairs in column order, keeping each pair's validity for the
            # sum pass. Column c = 2*k_frag + j extends extremum chain c % 4; the four
            # chains are independent.
            score_pairs, validity, chains = [], [], []
            for k_frag in cutlass.range_constexpr(self.qk_row_slots // 2):
                s_off = k_frag * 4
                # A zero scale scores every valid key 0, so a live row's anchor is 0 and
                # alpha and P are exp2(+-0) = 1 from runtime EX2s. P must stay a runtime
                # value: a constant would reach fp32_to_fp16's inline PTX as an
                # immediate, which ICEs libNVVM (pointwise.opaque_f32_zero).
                s0 = zero if cutlass.const_expr(self.scale_mode == SCALE_ZERO) else s_regs[s_off + s_reg_idx_lo]
                s1 = zero if cutlass.const_expr(self.scale_mode == SCALE_ZERO) else s_regs[s_off + s_reg_idx_hi]
                valid0 = cutlass.Boolean(column_mask & cutlass.Uint32(1 << (2 * k_frag)))
                valid1 = cutlass.Boolean(column_mask & cutlass.Uint32(1 << (2 * k_frag + 1)))
                validity.append((valid0, valid1))
                if not valid0:
                    s0 = sentinel
                if not valid1:
                    s1 = sentinel
                score_pairs.append(cutlass.Vector.from_elements((s0, s1), cutlass.Float32))
                if cutlass.const_expr(k_frag < 2):
                    chains += [s0, s1]
                else:
                    chain = 2 * (k_frag % 2)
                    chains[chain], chains[chain + 1] = extremum(chains[chain], s0), extremum(chains[chain + 1], s1)

            scores = RegTile(vec_concat(score_pairs), size=self.qk_row_slots)

            # Update the exact sign-aware anchor, then derive the old-output factor.
            # History joins the anchor before the quad tree.
            old = row_anchor[row_half]
            current = extremum(extremum(chains[0], chains[1]), extremum(chains[2], chains[3]))
            if cutlass.const_expr(not is_first_kv_tile):
                current = extremum(current, old)
            current = quad_reduce(current, quad_op)
            row_anchor[row_half] = current
            alive, old_alive = current != sentinel, old != sentinel
            safe_current, safe_old = current if alive else zero, old if old_alive else zero
            # A negative scale always takes FP64 exponents and a zero scale never does.
            # A positive scale takes them when the launch-uniform term is set, or when an
            # anchor is large enough that rounding anchor * k once per row (the FFMA
            # exponent's subtrahend) could move the top exponent by more than 1, or the
            # anchors' distance could overflow. The anchors bound every score that can
            # carry probability: a score far below its anchor has P = 0 on either path.
            wide = cutlass.Boolean(self.scale_mode < SCALE_ZERO)
            if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                bound = cute.arch.fmax(cute.math.abs(safe_current), cute.math.abs(safe_old))
                wide = softmax_params.wide_uniform
                if bound > softmax_params.wide_floor:
                    wide = (
                        wide
                        | (bound > softmax_params.wide_limit)
                        | (cute.math.abs(cutlass.Float64(safe_old) - cutlass.Float64(safe_current)) > cutlass.Float64(FLT_MAX_F64))
                    )
            # Resolve the row's numerical path once, for its scores and the old anchor.
            # Keeping the branch outside the pairs avoids reconvergence around each
            # exp2, and computing the old anchor's exponent inside it keeps its FP64
            # difference off the narrow path.
            exponents, old_exponent = scores.vec, zero
            if wide:
                old_exponent = cutlass.Float32((cutlass.Float64(old) - cutlass.Float64(current)) * scale_log2_wide)
                # FP64 differences; the zero-select below discards every invalid slot's
                # exponent.
                exponent_pairs = []
                for k_frag in cutlass.range_constexpr(self.qk_row_slots // 2):
                    pair = scores[2 * k_frag : 2 * k_frag + 2]
                    s0, s1 = pair[0], pair[1]
                    x0 = cutlass.Float32((cutlass.Float64(s0) - cutlass.Float64(current)) * scale_log2_wide)
                    x1 = cutlass.Float32((cutlass.Float64(s1) - cutlass.Float64(current)) * scale_log2_wide)
                    exponent_pairs.append(cutlass.Vector.from_elements((x0, x1), cutlass.Float32))
                exponents = vec_concat(exponent_pairs)
            else:
                old_exponent = (old - safe_current) * scale_log2
                # Preserve both products and the existing exponent rounding policy.
                exponents = (scores * scale_log2 - safe_current * scale_log2).vec
            old_scale = zero
            if cutlass.const_expr(is_first_kv_tile):
                old_scale = cutlass.Float32(1.0)
            elif old_alive & alive:
                old_scale = cute.math.exp2(old_exponent, fastmath=True)
            alpha[row_half] = old_scale

            # History is the first addend, then each pair's p0 and p1; the sum stays
            # lane-partial until finalization. Moving the history product can change
            # ptxas's multiply/add contraction and the recurrent FP32 sum.
            total = zero if cutlass.const_expr(is_first_kv_tile) else row_sum[row_half] * old_scale
            probabilities = RegTile(cute.math.exp2(exponents, fastmath=True), size=self.qk_row_slots)
            for k_frag in cutlass.range_constexpr(self.qk_row_slots // 2):
                p0, p1 = probabilities[2 * k_frag], probabilities[2 * k_frag + 1]
                valid0, valid1 = validity[k_frag]
                if not valid0:
                    p0 = p_zero
                if not valid1:
                    p1 = p_zero
                total = total + p0 + p1
                # One packed RS word, in the consumed score slot of p0.
                p_pack = fp32_to_fp16(p0, p1, dtype=self.storage_dtype).bitcast(cutlass.Float32)
                p_pack_off = k_frag * 4 + row_half * 2
                s_regs[p_pack_off] = p_pack
            row_sum[row_half] = total

        return s_regs

    @cute.jit
    def compute_one_kv_tile(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        s_regs: cutlass.Array,
        kv_step: cutlass.Int32,
        kv_full_state,
        active_masks: cutlass.Constexpr,
        is_first_kv_tile: cutlass.Constexpr,
    ):
        """Run one compute warpgroup's KV step: QK, online softmax, the O correction, RS PV.

        ``basic_params`` binds work/lane and the barriers; ``mma_params`` the operand
        descriptors (16-byte units, stage 0 for K and V) and the O Array; ``softmax_params``
        the row state and scale guards. The step writes ``s_regs`` (scores, then P), O and
        row state, and releases its K and V stage after the WGMMA that read it has
        completed (every MMA helper here waits for its group).

        :return: ``kv_full_state`` advanced past this stage.
        """
        output = mma_params.o_regs
        bars, work = basic_params.bars, basic_params.work
        stage = kv_full_state.idx
        kv_seq_idx = (work.kv_right - 1 - kv_step) * self.KV_TILE
        # This step's ring stage: the K and V descriptors move by whole stages.
        tile = SimpleNamespace(
            q_desc=mma_params.q_desc,
            k_desc=mma_params.k_desc + stage * self.kv_stage_units,
            v_desc=mma_params.v_desc + stage * self.kv_stage_units,
            o_regs=output,
        )
        bars.mb_tma_k_full[stage].wait(kv_full_state.phase)
        mma_qk(
            tile,
            s_regs,
            ab_dtype=self.storage_dtype,
            c_slots=self.qk_row_slots * 2,
            k_steps=self.HEAD_TILE // 16,
            q_slab_rows=self.Q_TILE,
            k_slab_rows=self.KV_TILE,
        )
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_tma_k_empty[stage].arrive()
        self.online_softmax(basic_params, softmax_params, s_regs, kv_seq_idx, active_masks=active_masks, is_first_kv_tile=is_first_kv_tile)
        # RS operand order: word 2*k_frag+row_half is score slot 4*k_frag+2*row_half.
        probability_words = cutlass.Vector.from_elements([s_regs[2 * word] for word in range(self.qk_row_slots)], cutlass.Float32).bitcast(cutlass.Uint32)
        # Loaded here, not inside the correction branch below: sinking it past
        # mbarrier_try_wait_parity would move the load.
        alpha = softmax_params.alpha.load(0, 2)
        # Probe this V generation once, correct O while V may still land, and wait only on a
        # miss. The probe cannot outlive the generation: this warpgroup has not released the
        # stage, and kv_full_state advances only after the fallback.
        v_ready = prims.mbarrier_try_wait_parity(bars.mb_tma_v_full[stage].smem_ptr, kv_full_state.phase, time_limit=1)
        if cutlass.const_expr(not is_first_kv_tile):
            rescale_output(output, alpha)
        if not v_ready:
            bars.mb_tma_v_full[stage].wait(kv_full_state.phase)
        mma_pv(tile, probability_words, is_first_kv_tile, ab_dtype=self.storage_dtype, k_steps=self.KV_TILE // 16)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_tma_v_empty[stage].arrive()
        return advance(kv_full_state, self.KV_STAGES)

    @cute.jit
    def epilogue(self, bars, work, tma_o_desc, sO, output, row_sum_inv, local_tid, w, row_base):
        """Normalize, stage and store compute warpgroup ``w``'s 64 O rows.

        All 128 threads of the warpgroup call this after its last PV, if any, has
        completed; each warpgroup first releases its share of Q EMPTY. ``stmatrix`` stages
        the rows into the warpgroup's own O half, and named barrier ``bar_o_stage_base + w``
        orders that staging before warp 0's elected lane issues the TMA store. A
        warpgroup whose rows all lie past ``q_output_limit`` stores nothing.
        """
        rescale_output(output, row_sum_inv)
        if work.iterations > 0:
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                bars.mb_tma_q_empty.arrive()
        # The O box lands rows in the Q box's order, so each fragment row is its O row.
        words = output.load(0, self.o_fragment_slots).to(self.out_dtype).bitcast(cutlass.Uint32)
        store_fragment(words, sO.base.subview(w * self.wg_tile_elems), local_tid)
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        # All four staging warps finish before warp 0 issues the TMA store. The branch is
        # warpgroup-uniform, and each arm names its barrier id at compile time.
        if w == 0:
            warpgroup_barrier_sync(self.bar_o_stage_base, cute.arch.THREADS_PER_WARPGROUP)
        else:
            warpgroup_barrier_sync(self.bar_o_stage_base + 1, cute.arch.THREADS_PER_WARPGROUP)
        if local_tid < cute.arch.WARP_SIZE:
            if row_base < work.q_output_limit:
                if prims.elect_sync():
                    tma_store_tile(sO.shifted(w * self.wg_tile_elems), self.tma_slice(tma_o_desc, work.head, row_base, work.batch, self.tma_orders[3]))
                    tma_store_commit()
                    tma_store_wait(0)

    @cute.jit
    def compute_warp_group(self, tma_o_desc, lse, scale, sQ, sK, sV, sO, bars, work, w):
        """Run compute warpgroup ``w`` (0 for WG1, 1 for WG2) on query rows ``64*w ..+63``.

        Both compute warpgroups walk the same KV tiles the same number of times, because
        every K and V EMPTY barrier counts both of them: the bounds are the CTA's 128 rows,
        and rows past ``q_len`` stay in the walk with an empty bitmap.
        """
        # The compute warpgroups start at threads 128 and 256, so every helper's
        # warpgroup-local index is the global one modulo the warpgroup.
        local_tid = cute.arch.thread_idx()[0] % cute.arch.THREADS_PER_WARPGROUP
        row_base = work.q_base + w * self.WG_ROWS
        if work.q_base < work.q_output_limit:
            # One m64n128 score fragment per thread: 64 * 128 / 128 = 64 FP32 C slots, of
            # which 32 consumed slots take the packed RS words.
            qk = cutlass.Array(cutlass.Float32, 2 * self.qk_row_slots, alignment=16)
            output = cutlass.Array(cutlass.Float32, self.o_fragment_slots, alignment=16)
            output.store((0.0,) * self.o_fragment_slots)
            # This warpgroup's 64 rows start 64*w rows into each 128-row Q slab, a whole number
            # of 8-row SW128 atoms.
            q_desc = wgmma_descriptor(sQ.base.subview(w * self.WG_ROWS * self.swizzle_chunk_elems))
            k_desc = wgmma_descriptor(sK.base)
            v_desc = wgmma_descriptor(sV.base, mn_major=True, slab_rows=self.KV_TILE)
            # Two-row register pairs: 16-byte alignment for the persistent row state, 8 for
            # every other pair.
            row_sum_inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
            row_sum_inv.store((0.0, 0.0))
            anchor = cutlass.Array(cutlass.Float32, 2, alignment=16)
            sentinel = cutlass.Float32(self.anchor_sentinel)
            anchor.store((sentinel, sentinel))
            partial_sum = cutlass.Array(cutlass.Float32, 2, alignment=16)
            partial_sum.store((0.0, 0.0))
            # This tile's two O-correction factors. Every scale path writes both halves
            # before compute_one_kv_tile reads them, so alpha needs no initial store.
            alpha = cutlass.Array(cutlass.Float32, 2, alignment=8)
            scale_log2 = scale * cutlass.Float32(LOG2_E)
            scale_log2_wide = cutlass.Float64(scale) * cutlass.Float64(LOG2_E)
            wide_uniform, wide_floor, wide_limit = cutlass.Boolean(False), cutlass.Float32(0.0), cutlass.Float32(0.0)
            if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                # Launch-uniform: decide the scale-only guard once, in setup. The row limit is
                # the smaller of the FP32 overflow bound and 2**24 / |k|: past |anchor * k| =
                # 2**24 one rounding of anchor * k moves the top exponent by more than 1.
                magnitude = cutlass.Float64(cute.math.abs(scale_log2))
                wide_uniform = magnitude > cutlass.Float64(FLT_MAX_F64)
                limit = cutlass.Float32(cutlass.Float64(FLT_MAX_F64) / magnitude)
                if cutlass.Float64(limit) * magnitude > cutlass.Float64(FLT_MAX_F64):
                    limit = (limit.bitcast(cutlass.Uint32) - cutlass.Uint32(1)).bitcast(cutlass.Float32)
                limit = cute.arch.fmin(limit, cutlass.Float32(cutlass.Float64(2.0**24) / magnitude))
                wide_floor, wide_limit = cute.arch.fmin(limit, cutlass.Float32(2.0**126)), limit
            # Natural-log scale for finalization, FP32/FP64 log2 scales for EX2, and the
            # launch-uniform guard terms for the per-row magnitude test.
            softmax_params = SimpleNamespace(
                anchor=anchor,
                partial_sum=partial_sum,
                alpha=alpha,
                scale=scale,
                scale_log2=scale_log2,
                scale_log2_wide=scale_log2_wide,
                wide_uniform=wide_uniform,
                wide_floor=wide_floor,
                wide_limit=wide_limit,
            )
            row_lse = cutlass.Array(cutlass.Float32, 2, alignment=8)
            row_lse.store((-cutlass.Float32.inf, -cutlass.Float32.inf))
            row_sum_inv_values, row_lse_values = row_sum_inv.load(0, 2), row_lse.load(0, 2)
            if work.iterations > 0:
                # Stable references for this unit; mutable contents keep their owners. Keep
                # the context inside this arm, away from control-flow joins.
                basic_params = SimpleNamespace(bars=bars, local_tid=local_tid, work=work, row_base=row_base)
                mma_params = SimpleNamespace(o_regs=output, q_desc=q_desc, k_desc=k_desc, v_desc=v_desc)
                # Reverse KV traversal: first tile, right boundary, interior, left boundary.
                # FULL waits start at phase 0.
                kv_full_state = PipelineState.start(phase=0)
                # Q FULL completes once per CTA (one work tile), so its one wait reads phase 0.
                bars.mb_tma_q_full.wait(cutlass.Int32(0))
                # Phase 1: the first tile and the right boundary, both possibly masked.
                kv_full_state = self.compute_one_kv_tile(
                    basic_params,
                    mma_params,
                    softmax_params,
                    qk,
                    cutlass.Int32(0),
                    kv_full_state,
                    active_masks=self.mask_flags | MASK_PADDED,
                    is_first_kv_tile=True,
                )
                # Reverse-step endpoints: boundary, fully valid interior, boundary. An
                # interior tile needs every score in it valid for all 128 rows of the CTA, so
                # an incomplete Q tile keeps every element predicate.
                right = cute.math.max(cutlass.Int32(0), work.kv_right)
                left = right - work.iterations
                diagonal = work.kv_len - work.q_len if cutlass.const_expr(self.bottom_right) else cutlass.Int32(0)
                hi = cute.math.min(right, work.kv_len // self.KV_TILE)
                if cutlass.const_expr(self.mask_flags & MASK_CAUSAL):
                    # The last key must be <= the earliest row's inclusive upper bound.
                    full_columns = cute.math.max(cutlass.Int32(0), work.q_base + diagonal + 1)
                    hi = cute.math.min(hi, full_columns // self.KV_TILE)
                hi = cute.math.max(left, hi)
                lo = cute.math.min(left, hi)
                if cutlass.const_expr(not self.q_rows_full):
                    if work.q_base + self.Q_TILE > work.q_len:
                        lo = hi
                masked_end = cute.math.min(work.iterations, cute.math.max(cutlass.Int32(1), right - hi))
                unmasked_end = cute.math.min(work.iterations, cute.math.max(masked_end, right - lo))
                for kv_step in cutlass.range(1, masked_end, unroll=1):
                    kv_full_state = self.compute_one_kv_tile(
                        basic_params, mma_params, softmax_params, qk, kv_step, kv_full_state, active_masks=self.mask_flags | MASK_PADDED, is_first_kv_tile=False
                    )
                # Phase 2: remaining fully unmasked iterations.
                for kv_step in cutlass.range(masked_end, unmasked_end, unroll=1):
                    kv_full_state = self.compute_one_kv_tile(
                        basic_params, mma_params, softmax_params, qk, kv_step, kv_full_state, active_masks=MASK_NONE, is_first_kv_tile=False
                    )
                # Phase 3: the left boundary, or every remaining tile when the Q tile is
                # incomplete.
                for kv_step in cutlass.range(unmasked_end, work.iterations, unroll=1):
                    kv_full_state = self.compute_one_kv_tile(
                        basic_params, mma_params, softmax_params, qk, kv_step, kv_full_state, active_masks=self.mask_flags | MASK_PADDED, is_first_kv_tile=False
                    )
                # Finalization with KV work: normalize both rows. Keep it in the live-KV arm,
                # as the D512 template does.
                for row_half in cutlass.range_constexpr(2):
                    total = quad_reduce(softmax_params.partial_sum[row_half], "sum")
                    inv, value = normalize_row(softmax_params.anchor[row_half], softmax_params.scale, softmax_params.scale_log2, total, self.scale_mode)
                    row_sum_inv[row_half], row_lse[row_half] = inv, value
                row_sum_inv_values, row_lse_values = row_sum_inv.load(0, 2), row_lse.load(0, 2)
            # Stores: Stats rows, then this warpgroup's O rows through the epilogue.
            if cutlass.const_expr(lse is not None):
                # Explicit Int64 strides, every index promoted before its multiply (SDPA Rule
                # S7): one batch origin, then the head and token terms.
                if local_tid % 4 == 0:
                    for row_half in cutlass.range_constexpr(2):
                        token = row_base + owned_row(local_tid, row_half)
                        if token < work.q_output_limit:
                            origin = cutlass.Int64(work.batch) * cutlass.Int64(lse.stride[0])
                            lse_out = row_lse_values[row_half]
                            if cutlass.const_expr(self.stats_log2):
                                lse_out = lse_out * cutlass.Float32(LOG2_E)
                            lse.iterator[
                                origin + cutlass.Int64(work.head) * cutlass.Int64(lse.stride[1]) + cutlass.Int64(token) * cutlass.Int64(lse.stride[2])
                            ] = lse_out
            self.epilogue(bars, work, tma_o_desc, sO, output, row_sum_inv_values, local_tid, w, row_base)

    @cute.jit
    def _run_unit(self, tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc, lse, scale, sQ, sK, sV, sO, bars, wg, tile, head, batch):
        """Resolve one (Q tile, head, batch) work tile and run this thread's role on it.

        ``wg`` selects the role -- WG0 load, WG1 and WG2 compute -- and each role sets its
        own register budget first. Nothing persists across units: every pipeline state
        starts from its own phase.
        """
        q_base = tile * self.Q_TILE
        q_len, kv_len = cutlass.Int32(self.s_q), cutlass.Int32(self.s_kv)
        # The reverse KV loop visits tiles [0, right). right is clipped by the KV length and,
        # under causal, by the diagonal of the CTA's last row.
        bounds = compute_kv_loop_bounds(q_base, q_len, kv_len, 0, self.mask_flags, self.KV_TILE, self.Q_TILE, bottom_right=self.kv_loop_bottom_right)
        iterations = cute.math.max(cutlass.Int32(0), bounds.right)
        work = SimpleNamespace(
            batch=batch,
            head=head,
            kv_head=head // self.qh_per_kh,
            q_base=q_base,
            q_len=q_len,
            kv_len=kv_len,
            q_output_limit=q_len,
            iterations=iterations,
            kv_right=iterations,
        )
        if wg == 0:
            prims.setmaxregister(self.REG_BUDGETS[0], prims.SetMaxRegisterAction.DECREASE)
            self.tmaldg_warp_group(sQ, sK, sV, bars, tma_q_desc, tma_k_desc, tma_v_desc, work)
        else:
            prims.setmaxregister(self.REG_BUDGETS[1], prims.SetMaxRegisterAction.INCREASE)
            self.compute_warp_group(tma_o_desc, lse, scale, sQ, sK, sV, sO, bars, work, wg - 1)

    @cute.kernel
    def kernel(
        self,
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_o_desc: cutlass.GridConstant[cuda.TensorMap],
        lse,
        scale: cutlass.Float32,
    ):
        """Run one work tile per CTA: allocate, decode, then run the three roles.

        Warp 0's elected lane initializes every barrier, and the default CTA barrier (id 0)
        publishes them before any role runs.
        """
        # One typed Array per region. The tiles align to 1024 B, one SW128 swizzle atom (8
        # rows x 128 B) and what every WGMMA descriptor assumes; each mbarrier array to 16 B,
        # with a ring's stages 8 B apart.
        sQ_raw = cutlass.Array(self.storage_dtype, self.tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sK_raw = cutlass.Array(self.storage_dtype, self.KV_STAGES * self.tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sV_raw = cutlass.Array(self.storage_dtype, self.KV_STAGES * self.tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sO_raw = cutlass.Array(self.storage_dtype, self.tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        bars = Bars(
            mb_tma_q_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=1,
                producer=Producer.TMA_LOAD,
            ),
            mb_tma_q_empty=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=2 * cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_tma_k_full=MBarrier(
                cutlass.Array(cutlass.Int64, self.KV_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
                stages=self.KV_STAGES,
                init_count=1,
                producer=Producer.TMA_LOAD,
            ),
            mb_tma_k_empty=MBarrier(
                cutlass.Array(cutlass.Int64, self.KV_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
                stages=self.KV_STAGES,
                init_count=2 * cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_tma_v_full=MBarrier(
                cutlass.Array(cutlass.Int64, self.KV_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
                stages=self.KV_STAGES,
                init_count=1,
                producer=Producer.TMA_LOAD,
            ),
            mb_tma_v_empty=MBarrier(
                cutlass.Array(cutlass.Int64, self.KV_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
                stages=self.KV_STAGES,
                init_count=2 * cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
        )
        # The TMA views, built once. Only their TMA fields are consumed: SmemTile.desc()
        # builds a tcgen05 descriptor, so the Hopper descriptors come from the raw Arrays
        # (compute_warp_group). O is indexed by warpgroup half, K and V by ring stage.
        sQ = SmemTile(
            base=sQ_raw,
            elems_per_stage=self.tile_elems,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.tile_slab_elems,
        )
        sK = SmemTile(
            base=sK_raw,
            elems_per_stage=self.tile_elems,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.tile_slab_elems,
            stages=self.KV_STAGES,
        )
        sV = SmemTile(
            base=sV_raw,
            elems_per_stage=self.tile_elems,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.tile_slab_elems,
            stages=self.KV_STAGES,
        )
        sO = SmemTile(
            base=sO_raw,
            elems_per_stage=self.wg_tile_elems,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.o_slab_elems,
        )
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:
            if prims.elect_sync():
                bars.mb_tma_q_full.init()
                bars.mb_tma_q_empty.init()
                for stage in cutlass.range_constexpr(self.KV_STAGES):
                    bars.mb_tma_k_full[stage].init()
                    bars.mb_tma_k_empty[stage].init()
                    bars.mb_tma_v_full[stage].init()
                    bars.mb_tma_v_empty[stage].init()
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync()
        # One (Q tile, head, batch) work tile per CTA; the LPT policies unflatten the single
        # grid axis they launch.
        tile, head, batch = cute.arch.block_idx()
        if cutlass.const_expr(self.sched_policy == SCHED_LPT_L2):
            # As the D512 template: a block holds a power of two of whole KV groups -- one KV
            # head and the query heads reading it -- within the budget; 0 bytes means one
            # group. lpt_l2_tile_coords floors its per-group bytes and group count at 1, so
            # its wrapping Int32 product cannot matter.
            row_bytes = 2 * self.HEAD_TILE * self.storage_dtype.bytes
            groups = SCHED_L2_BUDGET_BYTES // max(1, self.s_kv * row_bytes)
            l2_bytes = (1 << groups.bit_length() - 1) * self.s_kv * row_bytes if groups > 1 else 0
            tile, head, batch = lpt_l2_tile_coords(
                tile,
                cutlass.Int32(self.heads),
                cutlass.Int32(self.b),
                cutlass.Int32(self.q_tiles),
                self.qh_per_kh,
                cutlass.Int32(self.s_kv),
                row_bytes,
                l2_bytes,
            )
        elif cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            tile, head, batch = lpt_tile_coords(tile, cutlass.Int32(self.heads), cutlass.Int32(self.b), cutlass.Int32(self.q_tiles))
        wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // cute.arch.WARPS_PER_WARPGROUP
        self._run_unit(tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc, lse, scale, sQ, sK, sV, sO, bars, wg, tile, head, batch)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: Optional[cute.Tensor],
        sinks: Optional[cute.Tensor],
        seq_q_lens: cute.Tensor,
        seq_kv_lens: cute.Tensor,
        scale: cutlass.Float32,
        thd_max_sq: cutlass.Int32,
        thd_q_lens: Optional[cute.Tensor],
        thd_kv_lens: Optional[cute.Tensor],
        thd_lens_form: Optional[cutlass.Int32],
        thd_ro_q: Optional[cute.Tensor],
        thd_ro_k: Optional[cute.Tensor],
        thd_ro_v: Optional[cute.Tensor],
        thd_ro_o: Optional[cute.Tensor],
        thd_ro_lse: Optional[cute.Tensor],
        stream: cuda_driver.CUstream,
        block_table: Optional[cute.Tensor] = None,
        block_table_v: Optional[cute.Tensor] = None,
    ) -> None:
        """Build the port maps and launch the kernel.

        Only the compiled pointer host calls this internal view-building entry, with the
        D512 template's argument list; the shared prepared binder validates runtime buffers
        before launch. ``sinks``, the length views, the THD arguments (lengths and the
        ``thd_ro_q`` .. ``thd_ro_lse`` ragged offsets, ``None`` while ``thd_origins`` is) and
        the page tables (``block_table``, ``block_table_v``, never passed while ``page_size``
        is 0) are unread: a D128 specialization declares none of them.

        :param q: Query tensor ``(B, H_q, S_q, 128)`` over the caller's strides.
        :param k: Key tensor ``(B, H_kv, S_kv, 128)``.
        :param v: Value tensor ``(B, H_kv, S_kv, 128)``.
        :param o: Output tensor ``(B, H_q, S_q, 128)``, written in place.
        :param lse: FP32 Stats ``(B, H_q, S_q)``: the natural-log LSE, or base 2 under
            ``stats_log2``. ``None`` removes the Stats store at compile time.
        :param scale: The host softmax scale in natural units; the kernel folds ``log2(e)``
            itself. Its sign must match ``scale_mode``.
        :param stream: CUDA stream of the launch.
        """

        def port_tma_desc(tensor, rows, stride_order):
            return cuda.create_tensor_map_tiled_from_view(
                tensor, box_dims=(1, 1, rows, self.swizzle_chunk_elems), stride_order=stride_order, swizzle=self.tma_swizzle
            )

        tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc = (
            port_tma_desc(q, self.Q_TILE, self.tma_orders[0]),
            port_tma_desc(k, self.KV_TILE, self.tma_orders[1]),
            port_tma_desc(v, self.KV_TILE, self.tma_orders[2]),
            port_tma_desc(o, self.WG_ROWS, self.tma_orders[3]),
        )
        # The LPT policies launch the same tiles on one axis, which kernel unflattens.
        grid = (self.q_tiles, self.heads, self.b)
        if cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            grid = (self.q_tiles * self.heads * self.b, 1, 1)
        if cutlass.const_expr(sinks is not None):
            raise ValueError("prefill_d128_f16_sm90: a D128 specialization binds no sinks")
        # The declared problem sizes the grid, the KV-head addressing and the Stats store, so
        # each operand must carry it.
        ports = (("Q", q, self.h_q, self.s_q), ("K", k, self.h_kv, self.s_kv), ("V", v, self.h_kv, self.s_kv), ("O", o, self.h_q, self.s_q))
        if cutlass.const_expr(lse is not None):
            ports = (*ports, ("LSE", lse, self.h_q, self.s_q))
        for name, tensor, heads, tokens in ports:
            declared = (self.b, heads, tokens)
            if cutlass.const_expr(any(isinstance(extent, int) and extent != size for extent, size in zip(tensor.shape, declared))):
                raise ValueError(f"prefill_d128_f16_sm90: {name} shape {tuple(tensor.shape)} disagrees with the declared (B, H, S) = {declared}")
        # min_blocks_per_mp=1 emits .minnctapersm 1, which keeps the setmaxnreg split live
        # (python/cudnn/AGENTS.md: without it ptxas may drop every setmaxnreg silently).
        self.kernel(tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc, lse, scale).launch(
            grid=grid, block=(self.threads_per_cta, 1, 1), stream=stream, min_blocks_per_mp=1
        )


def compile(  # noqa: A001
    b: int,
    h_q: int,
    h_kv: int,
    s_q: int,
    s_kv: int,
    q_stride: tuple[int, int, int, int],
    k_stride: tuple[int, int, int, int],
    v_stride: tuple[int, int, int, int],
    o_stride: tuple[int, int, int, int],
    lse_stride: Optional[tuple[int, int, int]] = None,
    *,
    target: str = "sm_90a",
    d_qk: int = 128,
    d_v: int = 128,
) -> Callable:
    """Compile one declared problem at plan build time.

    The D512 template's signature: ``(b, h_q, h_kv, s_q, s_kv)`` is the declared problem
    that sizes the grid, ``q_stride`` .. ``o_stride`` each port's element strides in
    ``(B, H, S, D)`` mode order and ``lse_stride`` the ``(B, H_q, S_q)`` Stats strides of a
    ``has_lse`` template. The result exposes the positional pointer ABI in
    ``prepared_host.host``. ``d_qk``/``d_v`` must be the flavor's 128; this front coerces
    them to plain ints, because ``template_key`` refuses a non-plain argument.
    """
    if (d_qk, d_v) != D128_FLAVOR:
        raise ValueError(f"prefill_d128_f16_sm90: serves D_QK = D_V = 128 exactly; got D_QK={d_qk}/D_V={d_v}")
    return _compile(b, h_q, h_kv, s_q, s_kv, q_stride, k_stride, v_stride, o_stride, lse_stride, target=target, d_qk=int(d_qk), d_v=int(d_v))


@lru_cache(maxsize=None)
def _compile(b, h_q, h_kv, s_q, s_kv, q_stride, k_stride, v_stride, o_stride, lse_stride, *, target, d_qk, d_v):
    """Build the compiled launcher for :func:`compile`.

    ``compiled_cache`` persists it across processes.
    """
    # First statement: the key is exactly these parameters plus the template's source
    # digest.
    _cache_key = _template_key(globals(), locals(), "compile")
    kernel = SM90FusedMultiHeadAttentionForward(
        (b, h_q, h_kv, s_q, s_kv),
        in_dtype=STORAGE_DTYPE,
        out_dtype=STORAGE_DTYPE,
        is_causal=PARAMS.causal,
        sched_policy=PARAMS.sched_policy,
        # config_sm90 spells the alignment a causal graph implies as None (validate_params).
        bottom_right=PARAMS.causal if PARAMS.bottom_right is None else PARAMS.bottom_right,
        qh_per_kh=PARAMS.qh_per_kh,
        scale_mode=PARAMS.scale_mode,
        stats_log2=PARAMS.stats_log2,
    )

    # Axis order is plan metadata, shared by descriptor construction and kernel
    # coordinates: D first, then the other modes by stride. Physical strides are Int64
    # runtime leaves in the pointer host.
    def order(stride, shape):
        return (3, *sorted(range(3), key=lambda mode: (stride[mode], shape[mode], mode)))

    kernel.tma_orders = tuple(
        order(stride, shape)
        for stride, shape in ((q_stride, (b, h_q, s_q)), (k_stride, (b, h_kv, s_kv)), (v_stride, (b, h_kv, s_kv)), (o_stride, (b, h_q, s_q)))
    )
    return compile_host(kernel, STORAGE_DTYPE, d_qk, d_v, PARAMS.has_lse, False, _cache_key, target)
