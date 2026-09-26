# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
A fused multi-head attention (FMHA) FP16/BF16 prefill kernel for D=512 heads on NVIDIA
Hopper SM90 (sm_90a), on WGMMA tensor cores.

The CTA tile is (64, 64) over the (512, 512) head tile, which fixes both GEMM shapes. QK
is m64n64k512, one commit group of 32 k16 instructions. PV is m64n256k64 per compute
warpgroup, one group of 4. A 64-row FP32 O accumulator over 512 columns costs 256
registers per thread, and a WGMMA N is at most 256. Two compute warpgroups therefore
split the head dim, each owning one 256-column V/O half.

The kernel implements:
- Three warpgroups per CTA. WG0 issues the TMA loads. WG1 runs QK (SS), the online
  softmax and PV on O half 0 (RS). WG2 runs PV on O half 1 (SS) from the P that WG1
  stages in shared memory.
- One work tile per CTA over (Q tile, head, batch), in NATURAL, LPT or LPT_L2 order.
- Online softmax around an exact sign-aware row anchor, with lane-partial row sums
  quad-reduced only at finalization.
- Separate positive, zero and negative scale paths, plus an FP64 difference path for
  extreme scales.
- A reverse KV walk: boundary tiles mask elements, interior tiles do not.
- Causal masks (top-left or bottom-right), sliding windows, right bands, dense per-batch
  lengths, THD (ragged / packed) batches, PackGQA, per-Q-head sinks, and per-row LSE
  (Stats) in natural log or, under ``stats_log2``, base 2.

Storage: Q, K and V/O are single-stage 64 x 512 regions of eight SW128 slabs, V and O
aliasing one of them. Single-stage K and V mean the producer cannot run a KV tile ahead:
the next K lands only after QK releases the current one. P is the one multi-stage
region, crossing WG1 to WG2 through a three-stage ring; the alpha and inverse row
factors cross through FP32 arrays.

Constraints:
- FP16/BF16 in, O in the input dtype, FP32 accumulation.
- D_QK and D_V are each a multiple of 8 up to 512 (``config_sm90.head_dims_mismatch``);
  the maps carry the actual extents, so loads past D zero-fill and O stores clip.
- Q/K/V/O keep their declared strides: D contiguous, the rest 16-byte multiples.
- ``SdpaFwdDslSm90.check_support`` owns the served set.

WGMMA instructions and descriptors live in FROST ``tile_dsl/wgmma.py``, the D512 tile
operations and softmax arithmetic in ``sm90/_common_hopper.py``.
"""

from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, NamedTuple, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA
from cudnn.frost.tile_dsl.handles import GmemTileTmaSlice, SmemTile
from cudnn.frost.tile_dsl.mask import compute_kv_loop_bounds
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, opaque_f32_zero
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.scheduler import SCHED_LPT_L2, SCHED_NATURAL, lpt_l2_tile_coords, lpt_tile_coords
from cudnn.frost.tile_dsl.thd import TENSOR_MAP_ALIGN, TENSOR_MAP_QWORDS, THD_MAPS_META_WORDS, THD_MAPS_OFF
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.sdpa.fwd.config_sm90 import (
    D_TILE,
    SCALE_POSITIVE,
    SCALE_ZERO,
    TILE_M,
    TILE_N,
    TemplateParams,
    head_dims_mismatch,
    validate_params,
)
from cudnn.sdpa.fwd.kernels.thd_helpers import THD_SETUP_THREADS, build_thd_meta_o_descs_kernel
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import (
    FLT_MAX_F64,
    LOG2_E,
    SCHED_L2_BUDGET_BYTES,
    RolePipelineStates,
    mma_pv,
    mma_qk,
    normalize_row,
    normalize_sink_row,
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
validate_params(PARAMS)

STORAGE_DTYPE = {DTYPE_FP16: cutlass.Float16, DTYPE_BF16: cutlass.BFloat16}[PARAMS.dtype_qkv]


class Bars(NamedTuple):
    """Hold one barrier family per field, in allocation order.

    ``kernel`` allocates and initializes the record. Producer -> consumers, with each
    arrival count:

    - ``mb_tma_{q,k,v}_full``: WG0's elected expect-tx arrival plus the TMA bytes -> WG1
      (Q, K, V), WG2 (V).
    - ``mb_tma_{q,k}_empty``: WG1's 4 warps, after its epilogue (Q) and after each QK
      (K).
    - ``mb_tma_v_empty``: the 8 warps of WG1 and WG2, each after its PV half.
    - ``mb_p_xfer_{full,empty}``: the P_STAGES-deep P ring; WG1's 4 warps publish, WG2's
      4 release.
    - ``mb_alpha_xfer_{full,empty}``: one alpha pair per non-first KV tile, WG1's 4
      warps -> WG2's 4.
    - ``mb_inv_sum_xfer_full``: WG1's 4 warps publish the inverse row sums once per work
      tile, so it has no EMPTY partner.

    Consumers wait FULL from phase 0, producers EMPTY from phase 1, which passes at once
    on a fresh barrier.
    """

    mb_tma_q_full: object
    mb_tma_q_empty: object
    mb_tma_k_full: object
    mb_tma_k_empty: object
    mb_tma_v_full: object
    mb_tma_v_empty: object

    mb_p_xfer_full: object
    mb_p_xfer_empty: object
    mb_alpha_xfer_full: object
    mb_alpha_xfer_empty: object
    mb_inv_sum_xfer_full: object


class SM90FusedMultiHeadAttentionForward:
    """Configure and launch the SM90 FP16/BF16 D512 FMHA prefill kernel.

    Each CTA runs one work tile through three warpgroups. Three SM120 constructor
    parameters are omitted: ``engines._sm90_spec()`` declares no capability behind
    ``split_kv`` or ``thd_lse_padded``, and ``thd_lse_head_major`` has nothing left to
    select once Stats stay packed -- the declared strides of the packed
    ``(1, H_q, T_q)`` view choose token-major or head-major.
    """

    # The CTA tile: Q_TILE rows by KV_TILE keys over the HEAD_TILE-column D envelope.
    Q_TILE = TILE_M
    KV_TILE = TILE_N
    HEAD_TILE = D_TILE
    PV_HALF = HEAD_TILE // 2
    P_STAGES = 3  # Slots of the WG1-to-WG2 P ring.
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
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        seq_q_lens_present: bool = False,
        seq_kv_lens_present: bool = False,
        has_sink: bool = False,
        thd_varlen: bool = False,
        pack_gqa: bool = False,
        qh_per_kh: int = 1,
        scale_mode: int = SCALE_POSITIVE,
        stats_log2: bool = False,
    ):
        """Initialize the SM90 D512 FMHA prefill kernel configuration.

        :param shape: The declared problem ``(b, h_q, h_kv, s_q, s_kv)``, fixed at plan
            time. ``b``, ``h_q`` and ``s_q`` size the grid, and ``s_q``/``s_kv`` are the
            dense lengths that clamp device lengths. Under THD, ``b`` is the sequence
            count and ``s_q``/``s_kv`` are unread: ``thd_max_sq`` sizes the grid at
            launch.
        :param in_dtype: Q/K/V element type (Float16 or BFloat16). It types the SMEM
            tiles and sets ``swizzle_chunk_elems``, the SW128 slab width in columns.
        :param out_dtype: O element type. Must match ``in_dtype``: V and O alias one
            SMEM region.
        :param is_causal: Apply the causal upper bound to QK (``MASK_CAUSAL``).
        :param sched_policy: CTA order over the (Q tile, head, batch) tiles.
            ``SCHED_NATURAL`` launches the 3-D grid; ``SCHED_LPT`` and ``SCHED_LPT_L2``
            launch one axis that ``kernel`` unflattens.
        :param bottom_right: Anchor the diagonal at ``delta = kv_len - q_len`` instead
            of 0.
        :param window_size_left: Sliding-window offset ``W``: key ``k`` is visible from
            row ``q`` when ``k >= q + delta - W`` (``MASK_SWA``). ``None`` leaves the
            left side unbounded.
        :param window_size_right: Right-band offset ``R``: ``k`` is visible when
            ``k <= q + delta + R``. A right bound implies the causal bound;
            ``config_sm90.validate_params`` takes ``R >= 1`` only without ``causal``,
            and spells plain causal as ``causal`` alone.
        :param seq_q_lens_present: Read ``(B,)`` device Q lengths, clamped to ``s_q``.
            Dense only.
        :param seq_kv_lens_present: Read ``(B,)`` device KV lengths, clamped to ``s_kv``.
            THD sets it.
        :param has_sink: Fold the per-Q-head FP32 sink logit from ``sinks`` into the
            softmax denominator, as a virtual column with no V row. A live row with no
            visible key then stores the sink logit as its LSE. ``False`` binds no
            ``sinks`` operand.
        :param thd_varlen: THD (ragged / packed) mode. Q/K/V/O and LSE are packed
            batch-1 views, and ``__call__`` first launches the shared THD setup kernel.
        :param pack_gqa: Pack ``Q_TILE // qh_per_kh`` tokens x ``qh_per_kh`` query heads
            sharing one KV head into each Q tile: row ``r`` is token ``r // qh_per_kh``,
            head ``r % qh_per_kh``.
        :param qh_per_kh: The GQA ratio, refused unless ``h_q == h_kv * qh_per_kh``. Q
            head ``h`` reads KV head ``h // qh_per_kh``. Under ``pack_gqa`` it must
            divide ``Q_TILE``.
        :param scale_mode: The sign of the softmax scale: ``SCALE_POSITIVE``,
            ``SCALE_ZERO`` or ``SCALE_NEGATIVE``. Each sign compiles its own probability
            path, and a negative scale anchors each row at its minimum raw score. The
            runtime ``scale`` must carry this sign.
        :param stats_log2: Store Stats in base 2. The store multiplies each natural-log
            row LSE by ``log2(e)`` after normalization; ``-inf`` stays ``-inf``, and
            ``lse`` None removes it.
        """

        _, h_q, h_kv, _, _ = shape
        if out_dtype != in_dtype:
            raise ValueError("prefill_d512_f16_sm90: out_dtype must match in_dtype; V and O alias one SMEM region")
        if h_q != h_kv * qh_per_kh:
            raise ValueError(f"prefill_d512_f16_sm90: KV-head addressing needs H_q == H_kv * qh_per_kh; got {h_q}, {h_kv}, {qh_per_kh}")
        self.b, self.h_q, self.h_kv, self.s_q, self.s_kv = shape
        self.storage_dtype = in_dtype
        self.out_dtype = out_dtype
        self.sched_policy = sched_policy
        self.bottom_right = bottom_right
        # Band model: a right bound, plain or widened causal, sets MASK_CAUSAL, and a
        # left bound sets MASK_SWA. An offset whose flag is clear stays 0 and folds
        # away.
        self.window_left = window_size_left or 0
        self.window_right = window_size_right or 0
        self.mask_flags = (MASK_CAUSAL if is_causal or window_size_right is not None else MASK_NONE) | (MASK_SWA if window_size_left is not None else MASK_NONE)
        self.banded = self.mask_flags != MASK_NONE
        # Without a band, compute_kv_loop_bounds never reads the diagonal, so this flag
        # is inert there and True is the value it keeps.
        self.kv_loop_bottom_right = self.bottom_right if self.banded else True
        self.has_sink = has_sink
        self.thd_varlen = thd_varlen
        # Dense graphs clamp per-sequence device lengths to the declared envelope; THD
        # reads its lengths from the setup launch's metadata instead.
        self.dense_q_lengths = not thd_varlen and seq_q_lens_present
        self.dense_kv_lengths = not thd_varlen and seq_kv_lens_present

        self.pack_gqa = pack_gqa
        self.qh_per_kh = qh_per_kh
        # Rows of a Q tile per token (PackGQA packs qh_per_kh heads per token).
        self.heads_per_tile = qh_per_kh if pack_gqa else 1
        self.scale_mode = scale_mode
        # The anchor is the raw-score extremum that maximizes the scaled score: the
        # maximum for a non-negative scale, the minimum for a negative one, with the
        # sentinel as its identity.
        self.anchor_sentinel = -cutlass.Float32.inf if scale_mode >= SCALE_ZERO else cutlass.Float32.inf
        self.stats_log2 = stats_log2

        # Named-barrier ids. The init publish takes barrier_cta_sync's default id 0, so
        # the named ones start at 1. Id 1 is the V/O alias rendezvous of the 256 PV
        # threads. Ids 2 and 3 order each O half's four staging warps before its TMA
        # store.
        self.bar_vo = 1
        self.bar_o_half_base = 2

        self.threads_per_cta = len(self.REG_BUDGETS) * cute.arch.THREADS_PER_WARPGROUP

        self._setup_attributes()

    def _setup_attributes(self):
        """Compute derived tile, MMA, SMEM, and TMA constants."""

        # Tiling; a THD launch sizes its Q-tile axis from thd_max_sq instead.
        self.tokens_per_tile = self.Q_TILE // self.heads_per_tile
        self.q_tiles = (self.s_q + self.tokens_per_tile - 1) // self.tokens_per_tile
        self.heads = self.h_q // self.heads_per_tile
        # Every launched Q row is real only on a dense graph whose declared Q length is
        # whole tiles; online_softmax's row-validity test and _run_unit's partial-tile
        # guard fold away.
        self.q_rows_full = not self.thd_varlen and not self.dense_q_lengths and self.s_q % self.tokens_per_tile == 0
        # Q_TILE == KV_TILE, so one Q_TILE x HEAD_TILE count sizes the Q, K and V/O
        # regions.
        self.qkv_tile_elems = self.Q_TILE * self.HEAD_TILE
        self.vo_half_elements = self.Q_TILE * self.PV_HALF
        self.p_xfer_elems = self.Q_TILE * self.KV_TILE

        # MMA, per 128-thread warpgroup: a thread's QK C slots per row, and its O-half C
        # slots.
        self.qk_row_slots = self.p_xfer_elems // cute.arch.THREADS_PER_WARPGROUP // 2
        self.o_fragment_slots = self.vo_half_elements // cute.arch.THREADS_PER_WARPGROUP
        # SMEM: kernel() allocates every region at a fixed size, 222,408 B of SM90's
        # 232,448 B. Row r's FP32 factor sits at slot 2*r of the alpha and inverse
        # arrays; the odd slots are unused. WG1 writes each slot, WG2 reads it for its O
        # half.
        self.row_factor_elems = 2 * self.Q_TILE

        # TMA: every region moves in SW128 slabs of 128-byte rows, so one Q_TILE x
        # HEAD_TILE tile is tma_swizzle_chunks boxes of swizzle_chunk_elems columns,
        # slab_elements apart in SMEM. One geometry serves all three load ports.
        # __call__ builds the maps themselves, over the caller's strides.
        def get_swizzle(head_tile: int):
            head_bytes = head_tile * self.storage_dtype.bytes
            # SW128 only: wgmma_descriptor hard-codes the 8-row x 128-byte atom
            # (SBO=1024, LBO=16/8192) with no mode to select, so a 64- or 32-byte
            # swizzle would load correctly and then be misread.
            if head_bytes % 128:
                raise ValueError(f"prefill_d512_f16_sm90: a {head_bytes} B head tile is not whole SW128 slabs, which every WGMMA descriptor assumes")
            swizzle, span = cuda.TensorMapSwizzle.s128b, 128
            return swizzle, head_bytes // span, head_tile // (head_bytes // span)

        self.tma_swizzle, self.tma_swizzle_chunks, self.swizzle_chunk_elems = get_swizzle(self.HEAD_TILE)
        self.slab_elements = self.Q_TILE * self.swizzle_chunk_elems

    def tma_slice(self, tma_desc, desc_ptr, head_idx, seq_coord, batch_idx, tma_order, d_coord=None):
        """Return one TMA box at view coordinates ``(batch, head, token, d)``.

        The coordinates are listed in the map's own axis order. A dense box reads the
        grid-constant ``tma_desc`` with ``desc_ptr`` None. Under THD a runtime
        ``desc_ptr`` overrides it: O reads its own sequence's map, K and V the one
        clamped map every sequence shares, and Q keeps the grid-constant map. THD maps
        drop the batch axis. A PackGQA Q/O box spans ``heads_per_tile`` heads from
        ``head_idx``. ``tma_order`` is the port's ``self.tma_orders`` entry, the view
        modes ``__call__`` listed innermost first.
        """
        view = ((head_idx, seq_coord) if self.thd_varlen else (batch_idx, head_idx, seq_coord)) + (cutlass.Int32(0) if d_coord is None else d_coord,)
        return GmemTileTmaSlice(tma_desc, tuple(view[mode] for mode in tma_order), desc_ptr=desc_ptr)

    @cute.jit
    def load_q_tile_tma(
        self,
        sQ: SmemTile,
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        mbar: MBarrier,
        work,
    ) -> None:
        """Launch the TMA loads for the resident Q tile into its eight SW128 slabs.

        The producer warp calls this after the caller's EMPTY wait; the consumer's FULL
        wait completes the transfer. A PackGQA Q box spans heads_per_tile heads from the
        tile's first query head and lands head-fast, in the (D, H, S, B) map order.

        :param sQ: The destination SmemTile, tma_loads_per_tile slabs.
        :param tma_q_desc: Q's grid-constant map, dense and THD alike. Q needs no
            runtime map: a tile overhanging a sequence end lands rows the O store's
            clipped map discards.
        :param mbar: Q's FULL barrier.
        :param work: The decoded work tile, read for its Q coordinates and THD token
            origin.
        """
        # Keep this construction above the arrive: work.head * heads_per_tile is
        # evaluated here, not after the Q FULL arrive.
        q_coord = work.q_base + work.q_offset if cutlass.const_expr(self.thd_varlen) else work.q_base
        source = self.tma_slice(tma_q_desc, None, work.head * self.heads_per_tile, q_coord, work.batch, self.tma_orders[0])
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
        """Launch the TMA loads for one whole K or V tile into its eight SW128 slabs.

        The producer warp calls this after the caller's EMPTY wait; the consumer's FULL
        wait completes the transfer. The map's D extent bounds the box, so there is no
        envelope parameter.

        :param s_dst: The destination SmemTile, sK or sV, tma_loads_per_tile slabs.
        :param tma_desc: The port's grid-constant map.
        :param mbar: The port's FULL barrier.
        :param work: The decoded work tile, read for its KV head, batch, THD token
            origin and, under THD, the clamped map every sequence shares on this port.
        :param kv_seq_idx: The first token of the box, from the caller's reverse KV
            walk.
        :param is_v: Select the V port over the K one: its THD map pointer and map axis
            order.
        """
        # is_v picks the port at trace time, so no branch is staged. This construction
        # carries no arithmetic, so it has no ordering constraint against the arrive.
        desc_ptr = work.v_desc_ptr if is_v else work.k_desc_ptr
        kv_coord = kv_seq_idx + work.kv_offset if cutlass.const_expr(self.thd_varlen) else kv_seq_idx
        source = self.tma_slice(tma_desc, desc_ptr, work.kv_head, kv_coord, work.batch, self.tma_orders[2 if is_v else 1])
        mbar.arrive(n_bytes=tma_desc.global_tx_bytes() * s_dst.tma_loads_per_tile, pred=prims.elect_sync())
        tma_load_tile(s_dst, source, mbar.smem_ptr)

    @cute.jit
    def tmaldg_warp_group(self, sQ, sK, sV, bars, tma_q_desc, tma_k_desc, tma_v_desc, work):
        """Load resident Q once, then the reverse K/V tiles both PV halves consume.

        Warp 0 of producer warpgroup WG0 owns the waits, the reverse walk and the drain.
        WG1's epilogue releases Q. K and V are single-stage and refilled together once
        per KV step, each after its own EMPTY wait. The closing waits on the last
        generation of every EMPTY barrier leave no TMA payload in flight at exit.
        """
        tid = cute.arch.thread_idx()[0]
        if (work.q_base < work.q_len) & (work.iterations > 0):
            if tid < cute.arch.WARP_SIZE:
                # EMPTY waits start at phase 1, which passes at once, so the first trip
                # needs no branch. One state serves K and V, which refill together.
                q_empty_state = PipelineState.start(phase=1)
                kv_empty_state = PipelineState.start(phase=1)
                bars.mb_tma_q_empty.wait(q_empty_state.phase)
                q_empty_state = advance(q_empty_state, 1)
                self.load_q_tile_tma(sQ, tma_q_desc, bars.mb_tma_q_full, work)
                for kv_step in cutlass.range(work.iterations, unroll=1):
                    kv_seq_idx = (work.kv_right - 1 - kv_step) * self.KV_TILE
                    bars.mb_tma_k_empty.wait(kv_empty_state.phase)
                    self.load_one_kv_tile(sK, tma_k_desc, bars.mb_tma_k_full, work, kv_seq_idx, is_v=False)
                    bars.mb_tma_v_empty.wait(kv_empty_state.phase)
                    self.load_one_kv_tile(sV, tma_v_desc, bars.mb_tma_v_full, work, kv_seq_idx, is_v=True)
                    kv_empty_state = advance(kv_empty_state, 1)
                bars.mb_tma_q_empty.wait(q_empty_state.phase)
                bars.mb_tma_k_empty.wait(kv_empty_state.phase)
                bars.mb_tma_v_empty.wait(kv_empty_state.phase)

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
        """Update two rows and pack P for both PV consumers into consumed score slots.

        ``basic_params`` supplies the fixed work/lane references; ``softmax_params`` the
        mutable row state, this tile's alpha pair and the launch-uniform scales and
        guards. ``s_regs`` arrives with completed QK scores. ``compute_one_kv_tile``
        publishes the returned P words and alpha to WG2.

        Per row half r, ``anchor[r]`` is the exact sign-aware raw-score extremum, the
        sentinel while the row has no valid score, and ``partial_sum[r]`` this lane's
        FP32 share relative to ``exp(scale * anchor[r])``. After the first tile each
        step also yields ``alpha[r]``, which corrects both O halves.

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
            column_mask = cutlass.Uint32(0xFFFF)
            if cutlass.const_expr(active_masks != MASK_NONE):
                q_token = work.q_base + owned_row(local_tid, row_half) // self.heads_per_tile
                if cutlass.const_expr(not self.q_rows_full):
                    row_live = q_token < work.q_len
                diagonal = q_token + work.kv_len - work.q_len if cutlass.const_expr(self.bottom_right) else q_token
                column_base = kv_seq_idx + 2 * (local_tid % 4)
                if cutlass.const_expr(active_masks & MASK_PADDED):
                    column_mask = score_prefix_mask(work.kv_len, column_base)
                if cutlass.const_expr(active_masks & MASK_CAUSAL):
                    column_mask = column_mask & score_prefix_mask(diagonal + cutlass.Int32(self.window_right), column_base, inclusive=True)
                if cutlass.const_expr(active_masks & MASK_SWA):
                    column_mask = column_mask & ~score_prefix_mask(diagonal - cutlass.Int32(self.window_left), column_base)

            # Fold query validity into the lane's bitmap once for all score/P uses.
            column_mask = column_mask if row_live else cutlass.Uint32(0)

            # Gather masked pairs in column order, keeping each pair's validity for the
            # sum pass. Column c = 2*k_frag + j extends extremum chain c % 4; the four
            # chains are independent.
            score_pairs, validity, chains = [], [], []
            local_abs = cutlass.Float32(0.0)
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
                if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                    # A magnitude guard, not a finiteness proof: fmax quiets NaNs.
                    if valid0:
                        local_abs = cute.arch.fmax(local_abs, cute.math.abs(s0))
                    if valid1:
                        local_abs = cute.arch.fmax(local_abs, cute.math.abs(s1))

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
            # A positive scale takes them when the launch-uniform term is set, or when a
            # valid score, either anchor or their distance could overflow the FP32
            # products.
            wide = cutlass.Boolean(self.scale_mode < SCALE_ZERO)
            if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                bound = cute.arch.fmax(cute.arch.fmax(quad_reduce(local_abs, "max"), cute.math.abs(safe_current)), cute.math.abs(safe_old))
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
                # One packed word serves both RS and SS PV, in the consumed score slot
                # of p0.
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
        pipelines,
        active_masks: cutlass.Constexpr,
        is_first_kv_tile: cutlass.Constexpr,
    ):
        """Run WG1's KV step: QK, online softmax, the P/alpha handoff, RS PV on half 0.

        ``basic_params`` binds work/lane, barriers and the shared transfer storage;
        ``mma_params`` the operand descriptors (16-byte units) and WG1's O half;
        ``softmax_params`` the row state and scale guards. The step writes ``s_regs``
        (scores, then P), O registers and row state, and publishes P and alpha. All
        fields stay bound; this role's three pipeline states enter and leave explicitly.

        :return: This role's advanced K/V FULL, P-ring and alpha states, as one
            ``RolePipelineStates``.
        """
        kv_full_state, p_state, alpha_state = pipelines
        output = mma_params.o_regs
        bars, work, local_tid = basic_params.bars, basic_params.work, basic_params.local_tid
        sP_xfer_raw, sAlpha_xfer_raw = basic_params.sP_xfer_raw, basic_params.sAlpha_xfer_raw
        kv_seq_idx = (work.kv_right - 1 - kv_step) * self.KV_TILE
        bars.mb_tma_k_full.wait(kv_full_state.phase)
        mma_qk(mma_params, s_regs, ab_dtype=self.storage_dtype)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_tma_k_empty.arrive()
        self.online_softmax(basic_params, softmax_params, s_regs, kv_seq_idx, active_masks=active_masks, is_first_kv_tile=is_first_kv_tile)
        # WG2 needs the same two factors and the same packed probabilities.
        if cutlass.const_expr(not is_first_kv_tile):
            bars.mb_alpha_xfer_empty.wait(alpha_state.phase)
            if local_tid % 4 == 0:
                for row_half in cutlass.range_constexpr(2):
                    sAlpha_xfer_raw.subview(2 * owned_row(local_tid, row_half)).store(softmax_params.alpha[row_half])
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                bars.mb_alpha_xfer_full.arrive()
            alpha_state = advance(alpha_state, 1)
        # Both consumers read P in RS operand order: word 2*k_frag+row_half is score
        # slot 4*k_frag+2*row_half.
        probability_words = cutlass.Vector.from_elements([s_regs[2 * word] for word in range(self.qk_row_slots)], cutlass.Float32).bitcast(cutlass.Uint32)
        bars.mb_p_xfer_empty[p_state.idx].wait(p_state.phase)
        store_fragment(probability_words, sP_xfer_raw.subview(p_state.idx * self.p_xfer_elems), local_tid)
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_p_xfer_full[p_state.idx].arrive()
        p_state = advance(p_state, self.P_STAGES)
        # Loaded here, not inside the correction branch below: sinking it past
        # mbarrier_try_wait_parity would move the load.
        alpha = softmax_params.alpha.load(0, 2)
        # As in WG2's step: probe this V generation once, correct the completed O half
        # while V may still land, and wait only on a miss.
        v_ready = prims.mbarrier_try_wait_parity(bars.mb_tma_v_full.smem_ptr, kv_full_state.phase, time_limit=1)
        if cutlass.const_expr(not is_first_kv_tile):
            rescale_output(output, alpha)
        if not v_ready:
            bars.mb_tma_v_full.wait(kv_full_state.phase)
        mma_pv(mma_params, probability_words, is_first_kv_tile, ab_dtype=self.storage_dtype)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_tma_v_empty.arrive()
        kv_full_state = advance(kv_full_state, 1)
        return RolePipelineStates(kv_full_state, p_state, alpha_state)

    @cute.jit
    def compute_one_pv_tile(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        pipelines,
        is_first_kv_tile: cutlass.Constexpr,
    ):
        """Run WG2's KV step: the alpha handoff and correction, then SS PV on O half 1.

        ``basic_params`` binds the lane, barriers and the alpha SMEM; ``mma_params`` the
        16-byte-unit P and V descriptors and WG2's O-half Array, the only mutable state.
        Alpha is released after its register loads, P and V after WGMMA completion. This
        role's three pipeline states enter and leave explicitly.

        :return: This role's advanced V FULL, P-ring and alpha states, as one
            ``RolePipelineStates``.
        """
        v_full_state, p_state, alpha_state = pipelines
        p_desc, output = mma_params.p_desc, mma_params.o_regs
        bars, local_tid = basic_params.bars, basic_params.local_tid
        sAlpha_xfer_raw = basic_params.sAlpha_xfer_raw
        if cutlass.const_expr(not is_first_kv_tile):
            bars.mb_alpha_xfer_full.wait(alpha_state.phase)
            # All four lanes consume the same pair, at the existing alpha handoff.
            row0_slot, row1_slot = (sAlpha_xfer_raw.subview(2 * owned_row(local_tid, row_half)) for row_half in range(2))
            alpha = cutlass.Vector.from_elements((row0_slot.load(), row1_slot.load()), cutlass.Float32)
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                bars.mb_alpha_xfer_empty.arrive()
            alpha_state = advance(alpha_state, 1)
        # A single-shot probe of this V generation, not a wait loop: a miss falls back
        # to the blocking wait below. The result cannot outlive the generation, because
        # this role has not released it and v_full_state advances only after the
        # fallback.
        v_ready = prims.mbarrier_try_wait_parity(bars.mb_tma_v_full.smem_ptr, v_full_state.phase, time_limit=1)
        if cutlass.const_expr(not is_first_kv_tile):
            rescale_output(output, alpha)
        bars.mb_p_xfer_full[p_state.idx].wait(p_state.phase)
        if not v_ready:
            bars.mb_tma_v_full.wait(v_full_state.phase)
        mma_pv(mma_params, p_desc + p_state.idx * (self.p_xfer_elems * self.storage_dtype.bytes // 16), is_first_kv_tile, ab_dtype=self.storage_dtype)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_tma_v_empty.arrive()
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_p_xfer_empty[p_state.idx].arrive()
        v_full_state = advance(v_full_state, 1)
        p_state = advance(p_state, self.P_STAGES)
        return RolePipelineStates(v_full_state, p_state, alpha_state)

    @cute.jit
    def epilogue(self, bars, work, tma_o_desc, sO, output, row_sum_inv, tid, half: cutlass.Constexpr):
        """Normalize, stage and store this warpgroup's O half.

        WG1 owns half 0 and WG2 half 1. All 128 threads of one compute warpgroup call
        this after its last PV, if any, has completed; WG1 first releases Q EMPTY. The
        first ``bar_vo`` rendezvous proves both halves finished their V reads before O
        overwrites the V/O alias. ``stmatrix`` stages the half, and
        ``bar_o_half_base + half`` orders that staging before warp 0's elected lane
        issues the TMA store.
        """
        rescale_output(output, row_sum_inv)
        # The O store stays on the compute warpgroups: WG0 runs at REG_BUDGETS[0]
        # registers, and there is no fourth warpgroup.
        if cutlass.const_expr(half == 0):
            if work.iterations > 0:
                prims.bar_warp_sync(cute.arch.FULL_MASK)
                if prims.elect_sync():
                    bars.mb_tma_q_empty.arrive()
        # Both PV groups have completed their final V reads. WG0 owns no next tile.
        prims.barrier_cta_sync(self.bar_vo, thread_count=2 * cute.arch.THREADS_PER_WARPGROUP)
        # The O box lands rows in the Q box's order, so each fragment row is its O row.
        words = output.load(0, self.o_fragment_slots).to(self.out_dtype).bitcast(cutlass.Uint32)
        region = sO.base.subview(half * self.vo_half_elements)
        store_fragment(words, region, tid)
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        # All four staging warps must finish before warp 0 issues the TMA store.
        prims.barrier_cta_sync(self.bar_o_half_base + half, thread_count=cute.arch.THREADS_PER_WARPGROUP)
        if tid < cute.arch.WARP_SIZE:
            if prims.elect_sync():
                tma_store_tile(
                    sO.shifted(half * self.vo_half_elements),
                    self.tma_slice(
                        tma_o_desc,
                        work.o_desc_ptr,
                        work.head * self.heads_per_tile,
                        work.q_base,
                        work.batch,
                        self.tma_orders[3],
                        d_coord=cutlass.Int32(half * self.PV_HALF),
                    ),
                )
                tma_store_commit()
                tma_store_wait(0)
        # Both halves' source reads finish before V/O alias reuse or CTA teardown.
        prims.barrier_cta_sync(self.bar_vo, thread_count=2 * cute.arch.THREADS_PER_WARPGROUP)

    @cute.jit
    def _run_unit(
        self,
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        lse,
        sinks,
        seq_q_lens,
        seq_kv_lens,
        scale,
        sP_xfer_raw,
        sAlpha_xfer_raw,
        sInvSum_xfer_raw,
        sQ,
        sK,
        sV,
        sO,
        bars,
        wg,
        tile,
        head,
        batch,
    ):
        """Resolve one (Q tile, head, batch) work tile and run this thread's role on it.

        ``wg`` selects the role -- WG0 load, WG1 QK/PV, WG2 PV -- and each role sets its
        own register budget first. ``head`` is the query head, or the packed subgroup
        under PackGQA. Every role reads one ``work``: ``q_len`` limits input and score
        validity and ``q_output_limit`` output ownership, so absent THD tiles write
        nothing while trimmed dense tiles still write O = 0 and Stats = -inf. Both
        compute warpgroups test ``work.q_base < work.q_output_limit`` and meet at the
        epilogue's 256-thread barrier. Nothing persists across units: every pipeline
        state starts from its own phase.
        """
        q_base = tile * self.tokens_per_tile
        q_len, kv_len = cutlass.Int32(self.s_q), cutlass.Int32(self.s_kv)
        # THD only: this sequence's clipped O map and the clamped K/V maps every
        # sequence shares, then the packed token origins -- the cu prefixes, which ride
        # the TMA coordinate of every port but O, and anchor packed Stats.
        k_desc_ptr, v_desc_ptr, o_desc_ptr = None, None, None
        q_offset, kv_offset = cutlass.Int32(0), cutlass.Int32(0)
        if cutlass.const_expr(self.thd_varlen):
            # seq_kv_lens is the setup launch's metadata, Int32 [seq_kv(B), cu_q(B+1),
            # cu_k(B+1), remap(B), live, ctr], then the tensor maps at THD_MAPS_OFF.
            # remap, live and ctr stay unread: the rectangular NATURAL grid hands out no
            # dead unit. Indexed as write_thd_meta does: make_array_view refuses a
            # plain-int shape.
            meta = seq_kv_lens
            q_end, q_offset = cutlass.Int32(meta[self.b + batch + 1]), cutlass.Int32(meta[self.b + batch])
            q_len = q_end - q_offset
            kv_len = cutlass.Int32(meta[batch])
            kv_offset = cutlass.Int32(meta[2 * self.b + 1 + batch])
            # Map slots: per-sequence O in 0..B-1, the dead-unit pad at B, the clamped
            # K/V pair at B+1 and B+2. Only O carries a sequence origin, and its clip
            # keeps this sequence's last Q_TILE-row store out of the next. Q, K and V
            # reach their sequence through the cu prefixes, so an overhanging tile reads
            # the next sequence's tokens -- finite data the S mask discards -- and the
            # K/V clamp zero-fills only the packed total's tail.
            maps = cute.recast_ptr(seq_kv_lens.iterator + THD_MAPS_OFF(self.b), dtype=cutlass.Int64)
            base = maps.raw_ptr().tospace(cutlass.AddressSpace.generic)
            o_desc_ptr = base + batch * TENSOR_MAP_QWORDS
            k_desc_ptr = base + (self.b + 1) * TENSOR_MAP_QWORDS
            v_desc_ptr = base + (self.b + 2) * TENSOR_MAP_QWORDS
        # Dense device lengths are clamped to the declared envelopes, so an
        # out-of-contract value can neither address past the TMA descriptors nor move a
        # written row outside O or Stats.
        if cutlass.const_expr(self.dense_q_lengths):
            q_len = cute.math.min(cute.math.max(cutlass.Int32(cutlass.make_array_view(seq_q_lens)[batch]), cutlass.Int32(0)), cutlass.Int32(self.s_q))
        if cutlass.const_expr(self.dense_kv_lengths):
            kv_len = cute.math.min(cute.math.max(cutlass.Int32(cutlass.make_array_view(seq_kv_lens)[batch]), cutlass.Int32(0)), cutlass.Int32(self.s_kv))
        # The reverse KV loop visits tiles [left, right). right is clipped by the KV
        # length and any right bound; left moves only under a left bound, and is 0
        # without one. The clamp stays: under bottom-right causal with kv_len below
        # q_len, the leading Q tiles see nothing and right goes negative.
        bounds = compute_kv_loop_bounds(
            q_base,
            q_len,
            kv_len,
            self.window_left,
            self.mask_flags,
            self.KV_TILE,
            self.tokens_per_tile,
            bottom_right=self.kv_loop_bottom_right,
            window_right=self.window_right,
        )
        iterations = cute.math.max(cutlass.Int32(0), bounds.right - bounds.left if cutlass.const_expr(self.mask_flags & MASK_SWA) else bounds.right)
        if cutlass.const_expr(self.dense_q_lengths):
            # A wholly trimmed query tile loads nothing: every role sees zero KV steps,
            # and the epilogue writes the declared rows O = 0 and Stats = -inf.
            if q_base >= q_len:
                iterations = cutlass.Int32(0)
        # Step kv_step visits tile kv_right - 1 - kv_step. Without a left bound the walk
        # ends at tile 0, so its exclusive upper tile is its length.
        kv_right = bounds.right if cutlass.const_expr(self.mask_flags & MASK_SWA) else iterations
        kv_head = head if self.pack_gqa else head // self.qh_per_kh
        q_output_limit = cutlass.Int32(self.s_q) if cutlass.const_expr(self.dense_q_lengths) else q_len
        work = SimpleNamespace(
            batch=batch,
            head=head,
            kv_head=kv_head,
            q_base=q_base,
            q_len=q_len,
            kv_len=kv_len,
            q_offset=q_offset,
            kv_offset=kv_offset,
            q_output_limit=q_output_limit,
            iterations=iterations,
            k_desc_ptr=k_desc_ptr,
            v_desc_ptr=v_desc_ptr,
            o_desc_ptr=o_desc_ptr,
            kv_right=kv_right,
        )
        if wg == 0:
            prims.setmaxregister(self.REG_BUDGETS[0], prims.SetMaxRegisterAction.DECREASE)
            self.tmaldg_warp_group(sQ, sK, sV, bars, tma_q_desc, tma_k_desc, tma_v_desc, work)
        elif wg == 1:
            prims.setmaxregister(self.REG_BUDGETS[1], prims.SetMaxRegisterAction.INCREASE)
            # The compute warpgroups start at threads 128 and 256, so every helper's
            # warpgroup-local index is the global one modulo the warpgroup.
            local_tid = cute.arch.thread_idx()[0] % cute.arch.THREADS_PER_WARPGROUP
            # A THD tile past its sequence has no rows; dense Q lengths keep every
            # declared tile live.
            if work.q_base < work.q_output_limit:
                # One m64n64 score fragment per thread: 64 * 64 / 128 = 32 FP32 C slots,
                # of which 16 consumed slots take the packed RS words.
                qk = cutlass.Array(cutlass.Float32, 32, alignment=16)
                output = cutlass.Array(cutlass.Float32, self.o_fragment_slots, alignment=16)
                output.store((0.0,) * self.o_fragment_slots)
                q_desc, k_desc = wgmma_descriptor(sQ.base), wgmma_descriptor(sK.base)
                v_desc = wgmma_descriptor(sV.base, mn_major=True)
                # Two-row register pairs: 16-byte alignment for the persistent row
                # state, 8 for every other pair.
                row_sum_inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
                row_sum_inv.store((0.0, 0.0))
                anchor = cutlass.Array(cutlass.Float32, 2, alignment=16)
                sentinel = cutlass.Float32(self.anchor_sentinel)
                anchor.store((sentinel, sentinel))
                partial_sum = cutlass.Array(cutlass.Float32, 2, alignment=16)
                partial_sum.store((0.0, 0.0))
                # This tile's two O-correction factors. Every scale path writes both
                # halves before compute_one_kv_tile publishes them, so alpha needs no
                # initial store.
                alpha = cutlass.Array(cutlass.Float32, 2, alignment=8)
                scale_log2 = scale * cutlass.Float32(LOG2_E)
                scale_log2_wide = cutlass.Float64(scale) * cutlass.Float64(LOG2_E)
                wide_uniform, wide_floor, wide_limit = cutlass.Boolean(False), cutlass.Float32(0.0), cutlass.Float32(0.0)
                if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                    # Launch-uniform: decide the scale-only guard once, in WG1 setup.
                    magnitude = cutlass.Float64(cute.math.abs(scale_log2))
                    wide_uniform = magnitude > cutlass.Float64(FLT_MAX_F64)
                    limit = cutlass.Float32(cutlass.Float64(FLT_MAX_F64) / magnitude)
                    if cutlass.Float64(limit) * magnitude > cutlass.Float64(FLT_MAX_F64):
                        limit = (limit.bitcast(cutlass.Uint32) - cutlass.Uint32(1)).bitcast(cutlass.Float32)
                    wide_floor, wide_limit = cute.arch.fmin(limit, cutlass.Float32(2.0**126)), limit
                # Natural-log scale for finalization, FP32/FP64 log2 scales for EX2, and
                # the launch-uniform guard terms for the per-row magnitude test.
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
                    # Stable references for this WG1 unit; mutable contents keep their
                    # owners. Keep the context inside this arm, away from control-flow
                    # joins.
                    basic_params = SimpleNamespace(
                        sP_xfer_raw=sP_xfer_raw,
                        sAlpha_xfer_raw=sAlpha_xfer_raw,
                        bars=bars,
                        local_tid=local_tid,
                        work=work,
                    )
                    mma_params = SimpleNamespace(o_regs=output, q_desc=q_desc, k_desc=k_desc, v_desc=v_desc)
                    # Reverse KV traversal: first tile, right boundary, interior, left
                    # boundary. FULL waits start at phase 0, P's and alpha's EMPTY waits
                    # at phase 1.
                    pipelines = RolePipelineStates(PipelineState.start(phase=0), PipelineState.start(phase=1), PipelineState.start(phase=1))
                    # Q FULL completes once per CTA (one work tile), so its one wait
                    # reads phase 0.
                    bars.mb_tma_q_full.wait(cutlass.Int32(0))
                    # Phase 1: the first tile and the right boundary, both possibly
                    # masked.
                    pipelines = self.compute_one_kv_tile(
                        basic_params,
                        mma_params,
                        softmax_params,
                        qk,
                        cutlass.Int32(0),
                        pipelines,
                        active_masks=self.mask_flags | MASK_PADDED,
                        is_first_kv_tile=True,
                    )
                    # Reverse-step endpoints: boundary, fully valid interior, boundary.
                    # An interior tile needs every score in it valid, so an incomplete Q
                    # tile keeps every element predicate.
                    right = cute.math.max(cutlass.Int32(0), work.kv_right)
                    left = right - work.iterations
                    diagonal = work.kv_len - work.q_len if cutlass.const_expr(self.bottom_right) else cutlass.Int32(0)
                    hi = cute.math.min(right, work.kv_len // self.KV_TILE)
                    if cutlass.const_expr(self.mask_flags & MASK_CAUSAL):
                        # The last key must be <= the earliest row's inclusive upper
                        # bound.
                        full_columns = cute.math.max(cutlass.Int32(0), work.q_base + diagonal + self.window_right + 1)
                        hi = cute.math.min(hi, full_columns // self.KV_TILE)
                    hi = cute.math.max(left, hi)
                    lo = left
                    if cutlass.const_expr(self.mask_flags & MASK_SWA):
                        # The first key must be >= the latest row's inclusive lower
                        # bound.
                        first_column = cute.math.max(cutlass.Int32(0), work.q_base + self.tokens_per_tile - 1 + diagonal - self.window_left)
                        lo = cute.math.max(lo, (first_column + self.KV_TILE - 1) // self.KV_TILE)
                    lo = cute.math.min(lo, hi)
                    if cutlass.const_expr(not self.q_rows_full):
                        if work.q_base + self.tokens_per_tile > work.q_len:
                            lo = hi
                    masked_end = cute.math.min(work.iterations, cute.math.max(cutlass.Int32(1), right - hi))
                    unmasked_end = cute.math.min(work.iterations, cute.math.max(masked_end, right - lo))
                    for kv_step in cutlass.range(1, masked_end, unroll=1):
                        pipelines = self.compute_one_kv_tile(
                            basic_params, mma_params, softmax_params, qk, kv_step, pipelines, active_masks=self.mask_flags | MASK_PADDED, is_first_kv_tile=False
                        )
                    # Phase 2: remaining fully unmasked iterations.
                    for kv_step in cutlass.range(masked_end, unmasked_end, unroll=1):
                        pipelines = self.compute_one_kv_tile(
                            basic_params, mma_params, softmax_params, qk, kv_step, pipelines, active_masks=MASK_NONE, is_first_kv_tile=False
                        )
                    # Phase 3: the left boundary, or every remaining tile when the Q
                    # tile is incomplete.
                    for kv_step in cutlass.range(unmasked_end, work.iterations, unroll=1):
                        pipelines = self.compute_one_kv_tile(
                            basic_params, mma_params, softmax_params, qk, kv_step, pipelines, active_masks=self.mask_flags | MASK_PADDED, is_first_kv_tile=False
                        )
                    # Finalization with KV work: normalize both rows, then publish the
                    # inverse to WG2. Keep it in the live-KV arm; moving it past the
                    # join changes code generation. Load both sink logits before
                    # normalizing either row.
                    if cutlass.const_expr(self.has_sink):
                        logits = cutlass.Array(cutlass.Float32, 2, alignment=8)
                        if cutlass.const_expr(self.dense_q_lengths):
                            trimmed = cutlass.Array(cutlass.Boolean, 2)
                        sinks_arr = cutlass.make_array_view(sinks)
                        for row_half in cutlass.range_constexpr(2):
                            row = owned_row(local_tid, row_half)
                            token, subhead = row // self.heads_per_tile, row % self.heads_per_tile
                            logits[row_half] = sinks_arr[work.head * self.heads_per_tile + subhead]
                            if cutlass.const_expr(self.dense_q_lengths):
                                trimmed[row_half] = token + work.q_base >= work.q_len
                    for row_half in cutlass.range_constexpr(2):
                        total = quad_reduce(softmax_params.partial_sum[row_half], "sum")
                        if cutlass.const_expr(self.has_sink):
                            inv, value = normalize_sink_row(softmax_params.anchor[row_half], softmax_params.scale, total, logits[row_half], self.scale_mode)
                        else:
                            inv, value = normalize_row(softmax_params.anchor[row_half], softmax_params.scale, softmax_params.scale_log2, total, self.scale_mode)
                        if cutlass.const_expr(self.has_sink and self.dense_q_lengths):
                            inv = cutlass.Float32(0.0) if trimmed[row_half] else inv
                            value = cutlass.Float32(-cutlass.Float32.inf) if trimmed[row_half] else value
                        row_sum_inv[row_half], row_lse[row_half] = inv, value
                    row_sum_inv_values, row_lse_values = row_sum_inv.load(0, 2), row_lse.load(0, 2)
                    # The inverse carries the sink normalization, so WG2's O half folds
                    # identically.
                    if local_tid % 4 == 0:
                        for row_half in cutlass.range_constexpr(2):
                            sInvSum_xfer_raw.subview(2 * owned_row(local_tid, row_half)).store(row_sum_inv_values[row_half])
                    prims.bar_warp_sync(cute.arch.FULL_MASK)
                    if prims.elect_sync():
                        bars.mb_inv_sum_xfer_full.arrive()
                else:
                    # Finalization without KV work: an empty mask interval, zero KV or
                    # trimmed rows. The inverse stays zero, and only requested Stats
                    # need the sink logits.
                    if cutlass.const_expr(self.has_sink and lse is not None):
                        logits = cutlass.Array(cutlass.Float32, 2, alignment=8)
                        if cutlass.const_expr(self.dense_q_lengths):
                            trimmed = cutlass.Array(cutlass.Boolean, 2)
                        sinks_arr = cutlass.make_array_view(sinks)
                        for row_half in cutlass.range_constexpr(2):
                            row = owned_row(local_tid, row_half)
                            token, subhead = row // self.heads_per_tile, row % self.heads_per_tile
                            logits[row_half] = sinks_arr[work.head * self.heads_per_tile + subhead]
                            if cutlass.const_expr(self.dense_q_lengths):
                                trimmed[row_half] = token + work.q_base >= work.q_len
                        for row_half in cutlass.range_constexpr(2):
                            value = logits[row_half]
                            if cutlass.const_expr(self.dense_q_lengths):
                                value = cutlass.Float32(-cutlass.Float32.inf) if trimmed[row_half] else value
                            row_lse[row_half] = value
                        row_lse_values = row_lse.load(0, 2)
                # Stores: Stats rows, then this warpgroup's O half through the epilogue.
                if cutlass.const_expr(lse is not None):
                    # Stats row address: one sequence origin, then the head and token
                    # terms. Packed Stats stay packed, token-major or head-major, never
                    # dense-padded; the adapter classifies the declaration by stride
                    # (graph_analyzer.thd_stats_packing) and pins the packed batch
                    # stride to 0. The THD origin is the cu_q token prefix times the
                    # Stats token stride, and only dense Stats use a batch term.
                    # Explicit strides rather than a CuTe coordinate, because the Stats
                    # view is one strided rank-3 (1, H_q, T) covering both packings.
                    if local_tid % 4 == 0:
                        for row_half in cutlass.range_constexpr(2):
                            row = owned_row(local_tid, row_half)
                            token, subhead = row // self.heads_per_tile + work.q_base, row % self.heads_per_tile
                            # Trimmed dense rows are declared rows, and receive -inf
                            # past q_len.
                            if token < work.q_output_limit:
                                q_head = work.head * self.heads_per_tile + subhead
                                if cutlass.const_expr(not self.thd_varlen):
                                    origin = cutlass.Int64(work.batch) * cutlass.Int64(lse.stride[0])
                                else:
                                    origin = cutlass.Int64(work.q_offset) * cutlass.Int64(lse.stride[2])
                                # Base-2 Stats fold at the store: finalization without
                                # KV work writes a sink logit or -inf without
                                # normalizing. -inf stays -inf.
                                lse_out = row_lse_values[row_half]
                                if cutlass.const_expr(self.stats_log2):
                                    lse_out = lse_out * cutlass.Float32(LOG2_E)
                                lse.iterator[
                                    origin + cutlass.Int64(q_head) * cutlass.Int64(lse.stride[1]) + cutlass.Int64(token) * cutlass.Int64(lse.stride[2])
                                ] = lse_out
                self.epilogue(bars, work, tma_o_desc, sO, output, row_sum_inv_values, local_tid, half=0)
        else:
            prims.setmaxregister(self.REG_BUDGETS[2], prims.SetMaxRegisterAction.INCREASE)
            # The WG1 arm's warpgroup-local index and predicate.
            local_tid = cute.arch.thread_idx()[0] % cute.arch.THREADS_PER_WARPGROUP
            if work.q_base < work.q_output_limit:
                # State initialization: the second O half, its operand descriptors and
                # inverse. sV's last four slabs hold V columns [256, 512), this half's
                # operand.
                output = cutlass.Array(cutlass.Float32, self.o_fragment_slots, alignment=16)
                output.store((0.0,) * self.o_fragment_slots)
                p_desc = wgmma_descriptor(sP_xfer_raw)
                v_desc = wgmma_descriptor(sV.base.subview(self.vo_half_elements), mn_major=True)
                row_sum_inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
                row_sum_inv.store((0.0, 0.0))
                if work.iterations > 0:
                    # WG2's own context: shared references stay fixed, only its O half
                    # is mutable.
                    basic_params = SimpleNamespace(
                        sAlpha_xfer_raw=sAlpha_xfer_raw,
                        bars=bars,
                        local_tid=local_tid,
                        work=work,
                    )
                    mma_params = SimpleNamespace(o_regs=output, p_desc=p_desc, v_desc=v_desc)
                    # Reverse KV traversal over the V tiles, P stages and alpha WG1
                    # publishes, in its order. WG2 consumes every family it touches from
                    # phase 0.
                    pipelines = RolePipelineStates(PipelineState.start(phase=0), PipelineState.start(phase=0), PipelineState.start(phase=0))
                    pipelines = self.compute_one_pv_tile(basic_params, mma_params, pipelines, is_first_kv_tile=True)
                    # Steady steps 1..iterations-1: a trip count, since WG2 reads no KV
                    # coordinate.
                    for _ in cutlass.range(work.iterations - 1, unroll=1):
                        pipelines = self.compute_one_pv_tile(basic_params, mma_params, pipelines, is_first_kv_tile=False)
                # Keep this guard separate from the PV arm above: folding them changes
                # code generation.
                if work.iterations > 0:
                    # Finalization: WG1's inverse for rows with KV work, zero for the
                    # others. The inverse is published once per CTA, so its one wait
                    # reads phase 0.
                    bars.mb_inv_sum_xfer_full.wait(cutlass.Int32(0))
                    row0_slot, row1_slot = (sInvSum_xfer_raw.subview(2 * owned_row(local_tid, row_half)) for row_half in range(2))
                    row_sum_inv.store(cutlass.Vector.from_elements((row0_slot.load(), row1_slot.load()), cutlass.Float32))
                # Stores: this warpgroup's O half through the epilogue.
                self.epilogue(bars, work, tma_o_desc, sO, output, row_sum_inv.load(0, 2), local_tid, half=1)

    @cute.kernel
    def kernel(
        self,
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_o_desc: cutlass.GridConstant[cuda.TensorMap],
        lse,
        sinks,
        seq_q_lens,
        seq_kv_lens,
        scale: cutlass.Float32,
    ):
        """Run one work tile per CTA: allocate, decode, then run the three roles.

        A length tensor no role reads arrives as None; under THD ``seq_kv_lens`` is the
        metadata. Warp 0's elected lane initializes every barrier, and the default CTA
        barrier (id 0) publishes them before any role runs.
        """
        # One typed Array per region; V and O alias one. The Q/K/V and P tiles align to
        # 1024 B, one SW128 swizzle atom (8 rows x 128 B) and what every WGMMA
        # descriptor assumes; the FP32 row-factor arrays to 128 B; each mbarrier array
        # to 16 B, with a ring's stages 8 B apart.
        sQ_raw = cutlass.Array(self.storage_dtype, self.qkv_tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sK_raw = cutlass.Array(self.storage_dtype, self.qkv_tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sVO_raw = cutlass.Array(self.storage_dtype, self.qkv_tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sV_raw = sVO_raw
        sO_raw = sVO_raw
        sP_xfer_raw = cutlass.Array(self.storage_dtype, self.P_STAGES * self.p_xfer_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sAlpha_xfer_raw = cutlass.Array(cutlass.Float32, self.row_factor_elems, alignment=128, space=cutlass.AddressSpace.smem)
        sInvSum_xfer_raw = cutlass.Array(cutlass.Float32, self.row_factor_elems, alignment=128, space=cutlass.AddressSpace.smem)
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
                init_count=cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_tma_k_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=1,
                producer=Producer.TMA_LOAD,
            ),
            mb_tma_k_empty=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_tma_v_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=1,
                producer=Producer.TMA_LOAD,
            ),
            mb_tma_v_empty=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=2 * cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_p_xfer_full=MBarrier(
                cutlass.Array(cutlass.Int64, self.P_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
                stages=self.P_STAGES,
                init_count=cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_p_xfer_empty=MBarrier(
                cutlass.Array(cutlass.Int64, self.P_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
                stages=self.P_STAGES,
                init_count=cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_alpha_xfer_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_alpha_xfer_empty=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
            mb_inv_sum_xfer_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem),
                stages=1,
                init_count=cute.arch.WARPS_PER_WARPGROUP,
                producer=Producer.THREAD,
            ),
        )
        # The TMA views, built once. Only their TMA fields are consumed: SmemTile.desc()
        # builds a tcgen05 descriptor, so the Hopper descriptors come from the raw
        # Arrays instead.
        sQ = SmemTile(
            base=sQ_raw,
            elems_per_stage=self.qkv_tile_elems,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.slab_elements,
        )
        sK = SmemTile(
            base=sK_raw,
            elems_per_stage=self.qkv_tile_elems,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.slab_elements,
        )
        sV = SmemTile(
            base=sV_raw,
            elems_per_stage=self.KV_TILE * self.HEAD_TILE,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.slab_elements,
        )
        # One PV_HALF-column O half; the epilogue shifts it by self.vo_half_elements for
        # half 1.
        sO = SmemTile(
            base=sO_raw,
            elems_per_stage=self.Q_TILE * self.PV_HALF,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            tma_loads_per_tile=self.tma_swizzle_chunks // 2,
            tma_granu_elems=self.swizzle_chunk_elems,
            tma_subtile_stride_elems=self.slab_elements,
        )
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:
            if prims.elect_sync():
                bars.mb_tma_q_full.init()
                bars.mb_tma_q_empty.init()
                bars.mb_tma_k_full.init()
                bars.mb_tma_k_empty.init()
                bars.mb_tma_v_full.init()
                bars.mb_tma_v_empty.init()
                for stage in cutlass.range_constexpr(self.P_STAGES):
                    bars.mb_p_xfer_full[stage].init()
                    bars.mb_p_xfer_empty[stage].init()
                bars.mb_alpha_xfer_full.init()
                bars.mb_alpha_xfer_empty.init()
                bars.mb_inv_sum_xfer_full.init()
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync()
        # One (Q tile, head, batch) work tile per CTA; the LPT policies unflatten the
        # single grid axis they launch.
        tile, head, batch = cute.arch.block_idx()
        if cutlass.const_expr(self.sched_policy == SCHED_LPT_L2):
            # A block holds a power of two of whole KV groups -- one KV head and the
            # query heads reading it -- within the budget; 0 bytes means one group. That
            # width divides the usual power-of-two group counts, so no short last block
            # underfills the machine. A PackGQA tile already holds its group, so its head
            # ratio is 1. lpt_l2_tile_coords floors its per-group bytes and group count at
            # 1, so its wrapping Int32 product cannot matter.
            row_bytes = 2 * self.HEAD_TILE * self.storage_dtype.bytes
            groups = SCHED_L2_BUDGET_BYTES // max(1, self.s_kv * row_bytes)
            l2_bytes = (1 << groups.bit_length() - 1) * self.s_kv * row_bytes if groups > 1 else 0
            tile, head, batch = lpt_l2_tile_coords(
                tile,
                cutlass.Int32(self.heads),
                cutlass.Int32(self.b),
                cutlass.Int32(self.q_tiles),
                self.heads // self.h_kv,
                cutlass.Int32(self.s_kv),
                row_bytes,
                l2_bytes,
            )
        elif cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            tile, head, batch = lpt_tile_coords(tile, cutlass.Int32(self.heads), cutlass.Int32(self.b), cutlass.Int32(self.q_tiles))
        wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // cute.arch.WARPS_PER_WARPGROUP
        self._run_unit(
            tma_q_desc,
            tma_k_desc,
            tma_v_desc,
            tma_o_desc,
            lse,
            sinks,
            seq_q_lens,
            seq_kv_lens,
            scale,
            sP_xfer_raw,
            sAlpha_xfer_raw,
            sInvSum_xfer_raw,
            sQ,
            sK,
            sV,
            sO,
            bars,
            wg,
            tile,
            head,
            batch,
        )

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
        stream: cuda_driver.CUstream,
    ) -> None:
        """Build the port maps, run the THD setup when packed, and launch the kernel.

        The argument order is the adapter's launch ABI, the SM120 one less
        ``thd_n_ctas``. The checks here are trace-time; the compiled launcher refuses
        runtime dtype, device and alignment mismatches.

        :param q: Query tensor ``(B, H_q, S_q, D_qk)`` over the caller's strides, or the
            packed ``(1, H_q, T_q, D_qk)`` THD view whose token total is dynamic.
        :param k: Key tensor ``(B, H_kv, S_kv, D_qk)``, or packed
            ``(1, H_kv, T_kv, D_qk)``.
        :param v: Value tensor ``(B, H_kv, S_kv, D_v)``, or packed
            ``(1, H_kv, T_kv, D_v)``.
        :param o: Output tensor ``(B, H_q, S_q, D_v)``, or packed
            ``(1, H_q, T_q, D_v)``, written in place.
        :param lse: FP32 Stats, dense ``(B, H_q, S_q)`` or packed ``(1, H_q, T_q)``: the
            natural-log LSE, or base 2 under ``stats_log2``. ``None`` removes the Stats
            store at compile time.
        :param sinks: ``(H_q,)`` FP32 per-Q-head sink logits, ``None`` without
            ``has_sink``.
        :param seq_q_lens: Int32 ``(B,)`` Q lengths, or an unused dummy.
        :param seq_kv_lens: Int32 ``(B,)`` KV lengths, or an unused dummy. Under THD, the
            128-byte-aligned ``THD_MAPS_META_WORDS(B)`` scratch: metadata, then maps.
        :param scale: The host softmax scale in natural units; the kernel folds
            ``log2(e)`` itself. Its sign must match ``scale_mode``.
        :param thd_max_sq: THD only: the declared S_q envelope, which sizes the grid.
        :param thd_q_lens: THD only: the caller's ``(B,)`` lengths or ``(B+1,)`` prefix
            sums. ``None`` when dense.
        :param thd_kv_lens: THD only: the same for KV.
        :param thd_lens_form: THD only: bit 0: Q is cu, bit 1: KV is cu.
        :param stream: CUDA stream of the THD setup launch and the attention launch.
        """
        # A map holds each stride in 16-byte units, so a remainder would floor away, and
        # it reads D as contiguous. A THD map has no batch axis and reads no batch
        # stride.
        for name, tensor in (("Q", q), ("K", k), ("V", v), ("O", o)):
            if cutlass.const_expr(tensor.stride[3] != 1):
                raise ValueError(f"prefill_d512_f16_sm90: {name} strides {tuple(tensor.stride)}: D must be innermost-contiguous (stride 1)")
            map_strides = tensor.stride[1:3] if self.thd_varlen else tensor.stride[:3]
            if cutlass.const_expr(any(stride * self.storage_dtype.bytes % 16 for stride in map_strides)):
                raise ValueError(
                    f"prefill_d512_f16_sm90: {name} strides {tuple(tensor.stride)} must be 16-byte multiples at BPE={self.storage_dtype.bytes} (TMA global-stride rule)"
                )

        # One map per port over the caller's strides. A dense one-head map lists D, then
        # its B, H and S axes innermost first by (stride, extent, axis). A PackGQA Q/O
        # map keeps (D, H, S, B), so its box of heads_per_tile heads x tokens_per_tile
        # tokens lands head-fast: row = token * heads_per_tile + head. A THD map is (D,
        # H, S), innermost first in every packed layout the adapter serves, and drops
        # the batch axis.
        #
        # Each box is one SW128 slab of the consumer's tile rows: Q_TILE for Q/O,
        # KV_TILE for K/V. D keeps its actual extent, so loads past it zero-fill and O
        # stores past it clip. The setup launch copies O's map into per-sequence maps
        # and clamps K's and V's to the packed total, while Q's is read as built.
        def port_tma_order(tensor, heads):
            order = (3, 1, 2, 0)
            if cutlass.const_expr(self.thd_varlen):
                order = (2, 0, 1)
            elif cutlass.const_expr(heads == 1):
                order = (3, *sorted(range(3), key=lambda mode: (tensor.stride[mode], tensor.shape[mode], mode)))
            return order

        self.tma_orders = tuple(port_tma_order(tensor, heads) for tensor, heads in ((q, self.heads_per_tile), (k, 1), (v, 1), (o, self.heads_per_tile)))

        def port_tma_desc(tensor, heads, rows, stride_order):
            box = (1, heads, rows // heads, self.swizzle_chunk_elems)
            if cutlass.const_expr(self.thd_varlen):
                # The packed token total is dynamic and a zero-extent map is invalid:
                # keep one unread token.
                tokens = cute.math.max(cutlass.Int32(tensor.shape[2]), cutlass.Int32(1))
                tensor = cute.make_tensor(tensor.iterator, cute.make_layout((tensor.shape[1], tokens, tensor.shape[3]), stride=tensor.stride[1:]))
                box = box[1:]
            return cuda.create_tensor_map_tiled_from_view(tensor, box_dims=box, stride_order=stride_order, swizzle=self.tma_swizzle)

        tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc = (
            port_tma_desc(q, self.heads_per_tile, self.Q_TILE, self.tma_orders[0]),
            port_tma_desc(k, 1, self.KV_TILE, self.tma_orders[1]),
            port_tma_desc(v, 1, self.KV_TILE, self.tma_orders[2]),
            port_tma_desc(o, self.heads_per_tile, self.Q_TILE, self.tma_orders[3]),
        )
        # The LPT policies launch the same tiles on one axis, which kernel unflattens.
        # THD is NATURAL, over ceil(thd_max_sq / tokens_per_tile) tiles.
        grid = (self.q_tiles, self.heads, self.b)
        if cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            grid = (self.q_tiles * self.heads * self.b, 1, 1)
        if cutlass.const_expr(self.thd_varlen):
            if cutlass.const_expr(tuple(seq_kv_lens.shape) != (THD_MAPS_META_WORDS(self.b),)):
                raise ValueError(f"prefill_d512_f16_sm90: THD seq_kv_lens must be the ({THD_MAPS_META_WORDS(self.b)},) metadata tensor")
            q_tiles = (thd_max_sq + self.tokens_per_tile - 1) // self.tokens_per_tile
            grid = (q_tiles, cutlass.Int32(self.heads), cutlass.Int32(self.b))
            # The shared THD setup launch: one elected thread writes the metadata, the
            # per-sequence O maps and the two clamped K/V maps, then issues one
            # GENERIC->TENSORMAP release fence. Stream order publishes all of it before
            # the attention launch, and each TMA issuer's acquire completes the handoff.
            # O's token stride keeps its call-site width: the parameter is unannotated
            # so a >= 2**31 element stride is not truncated. The batch ranking and
            # live-unit words the same kernel writes are dead here.
            stride_type = cutlass.Int32 if o.stride[2] < 2**31 else cutlass.Int64
            build_thd_meta_o_descs_kernel(
                o,
                tma_o_desc,
                tma_k_desc,
                tma_v_desc,
                cute.make_tensor(
                    cute.recast_ptr(seq_kv_lens.iterator + THD_MAPS_OFF(self.b), dtype=cutlass.Int64),
                    cute.make_layout(((self.b + 3) * TENSOR_MAP_QWORDS,), stride=(1,)),
                ),
                seq_kv_lens,
                thd_q_lens,
                thd_kv_lens,
                thd_lens_form,
                cutlass.Int32(self.heads),
                cutlass.Int32(self.b),
                stride_type(o.stride[2]),
                cutlass.Int32(self.tokens_per_tile),
                q_tiles * cutlass.Int32(self.heads * self.b),
            ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        # Only a has_sink specialization reads sinks; any other plan would ignore the
        # operand.
        if cutlass.const_expr(self.has_sink != (sinks is not None)):
            raise ValueError("prefill_d512_f16_sm90: sinks must be provided exactly when the kernel is configured with has_sink")
        if cutlass.const_expr(sinks is not None and tuple(sinks.shape) != (self.h_q,)):
            raise ValueError(f"prefill_d512_f16_sm90: sinks must have shape (H_q,) = ({self.h_q},); got {tuple(sinks.shape)}")
        # The declared problem sizes the grid, the KV-head addressing and the Stats
        # store, so each operand must carry it. A THD port is a batch-1 view whose token
        # total has no static extent.
        batch = 1 if self.thd_varlen else self.b
        ports = (("Q", q, self.h_q, self.s_q), ("K", k, self.h_kv, self.s_kv), ("V", v, self.h_kv, self.s_kv), ("O", o, self.h_q, self.s_q))
        if cutlass.const_expr(lse is not None):
            ports = (*ports, ("LSE", lse, self.h_q, self.s_q))
        for name, tensor, heads, tokens in ports:
            declared = (batch, heads, tokens)
            if cutlass.const_expr(any(isinstance(extent, int) and extent != size for extent, size in zip(tensor.shape, declared))):
                raise ValueError(f"prefill_d512_f16_sm90: {name} shape {tuple(tensor.shape)} disagrees with the declared (B, H, S) = {declared}")
        # Only a length tensor some role reads reaches the kernel; THD's is the metadata.
        read_q_lens = seq_q_lens if cutlass.const_expr(self.dense_q_lengths) else None
        read_kv_lens = seq_kv_lens if cutlass.const_expr(self.thd_varlen or self.dense_kv_lengths) else None
        self.kernel(tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc, lse, sinks, read_q_lens, read_kv_lens, scale).launch(
            grid=grid, block=(self.threads_per_cta, 1, 1), stream=stream
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
    d_qk: int = 512,
    d_v: int = 512,
) -> Callable:
    """Compile one declared problem at plan build time.

    Packed capacities and scale stay dynamic. ``(b, h_q, h_kv, s_q, s_kv)`` is the
    declared problem that sizes the grid; under THD ``b`` is the sequence count and
    this front zeroes ``s_q``/``s_kv`` out of the key (``thd_max_sq`` is a launch
    argument). ``q_stride`` .. ``o_stride`` are each port's element strides in
    ``(B, H, S, D)`` mode order, with a zero batch stride under THD, and ``lse_stride``
    the ``(B, H_q, S_q)`` Stats strides of a ``has_lse`` template. The result launches
    as ``SM90FusedMultiHeadAttentionForward.__call__``.

    ``d_qk``/``d_v`` are the graph's actual head dims (Q/K and V/O). They size only the
    GMEM descriptors: the compute tile stays 512 wide, so a smaller D still pays the
    D512 tile cost. This front coerces them to plain ints: ``template_key`` refuses a
    non-plain argument, which would drop the persisted cache entry.
    """
    reason = head_dims_mismatch(d_qk, d_v)
    if reason is not None:
        raise ValueError(f"prefill_d512_f16_sm90: {reason}")
    d_qk, d_v = int(d_qk), int(d_v)
    if PARAMS.thd_varlen:
        s_q = s_kv = 0
    return _compile(b, h_q, h_kv, s_q, s_kv, q_stride, k_stride, v_stride, o_stride, lse_stride, target=target, d_qk=d_qk, d_v=d_v)


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
        # config_sm90 spells the alignment a causal graph implies as None
        # (validate_params).
        bottom_right=PARAMS.causal if PARAMS.bottom_right is None else PARAMS.bottom_right,
        window_size_left=PARAMS.window_left,
        window_size_right=PARAMS.window_right,
        seq_q_lens_present=PARAMS.seq_q_lens_present,
        seq_kv_lens_present=PARAMS.seq_kv_lens_present,
        has_sink=PARAMS.has_sink,
        thd_varlen=PARAMS.thd_varlen,
        pack_gqa=PARAMS.pack_gqa,
        qh_per_kh=PARAMS.qh_per_kh,
        scale_mode=PARAMS.scale_mode,
        stats_log2=PARAMS.stats_log2,
    )
    # The fakes carry the declared strides and actual head dims. THD packs the batch
    # axis to extent 1, and each port's token total is its own dynamic symbol.
    fake_batch = 1 if PARAMS.thd_varlen else b

    def _fake_bhsd(heads, tokens, d, stride):
        if PARAMS.thd_varlen:
            tokens = cute.sym_int(divisibility=1)
        return cute.runtime.make_fake_tensor(STORAGE_DTYPE, (fake_batch, heads, tokens, d), tuple(stride), assumed_align=16)

    fake_q = _fake_bhsd(h_q, s_q, d_qk, q_stride)
    fake_k = _fake_bhsd(h_kv, s_kv, d_qk, k_stride)
    fake_v = _fake_bhsd(h_kv, s_kv, d_v, v_stride)
    fake_o = _fake_bhsd(h_q, s_q, d_v, o_stride)
    fake_lse = None
    if PARAMS.has_lse:
        lse_tokens = cute.sym_int(divisibility=1) if PARAMS.thd_varlen else s_q
        fake_lse = cute.runtime.make_fake_tensor(cutlass.Float32, (fake_batch, h_q, lse_tokens), tuple(lse_stride), assumed_align=4)
    fake_sinks = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (h_q,), stride_order=(0,), assumed_align=4) if PARAMS.has_sink else None
    fake_seq_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b,), stride_order=(0,), assumed_align=4)
    # THD: metadata, then tensor maps, on the TMA boundary the launcher checks.
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (THD_MAPS_META_WORDS(b),) if PARAMS.thd_varlen else (b,),
        stride_order=(0,),
        assumed_align=TENSOR_MAP_ALIGN if PARAMS.thd_varlen else 4,
    )
    # Dynamic extents: (B,) lengths and (B+1,) prefix sums bind one artifact.
    if PARAMS.thd_varlen:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = None
        fake_thd_kv_lens = None
        fake_thd_lens_form = None
    return _compile_cached(
        kernel,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        fake_sinks,
        fake_seq_q_lens,
        fake_seq_kv_lens,
        cutlass.Float32(1.0),
        cutlass.Int32(0),  # thd_max_sq: plan-time envelope grid extent (THD)
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch={target}",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )
