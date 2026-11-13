# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The kernel in this file is derived from SplitKDenseGemmKernel of FlashInfer
# (flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py, flashinfer-python 0.7.2.post1,
# https://github.com/flashinfer-ai/flashinfer at commit beabf5004585b03da60e927bbb803acbaa62fe36),
# licensed under Apache-2.0, and has been modified; the module docstring lists the changes.
# FlashInfer's NOTICE:
#   FlashInfer
#   Copyright 2025-2026 NVIDIA
#   Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)

"""The FP8 split-K projection for decode rows: ``out[t, n] = bf16(scale * sum_k activation[t, k] * weight[n, k])`` for a weight
``[N, K]`` and an activation ``[rows, K]`` of E4M3 values, one fp32 ``scale`` and a bf16 ``out`` ``[rows, N]``, ``rows`` 1 .. 32; fp32
accumulation on the tcgen05 FP8 MMA.  Config and knobs in ``fp8_projection_decode_config.py``, the host entry in
``projection_decode/fp8_projection_decode.py``.

DERIVED from FlashInfer's cluster split-K dense GEMM (the file header names the source and its licence).  Changes: (1) E4M3 operands,
carried across the launch as bytes and recast in the host function, on the FP8 MMA (the source: bf16 / fp16); (2) the epilogue multiplies
the fp32 sum by one fp32 scalar and casts to bf16 once (the source: an optional bias add and the cast); the bias operand, its barrier and
its load are gone; (3) static shapes: ``N``, ``K`` and the output rows are trace-time constants, the activation rows stay symbolic so
that a one-row activation keeps a 2-D tensor map (the source: fully dynamic layouts); (4) the launch reserves the shared memory the
configuration uses (the source: the whole per-CTA capacity); (5) the tensor-memory allocation is the configuration's ``TMEM_COLS`` (the
source: 256 columns), so that every CTA the launcher places on an SM also gets its columns; (6) a control arm in which the weight loader
waits for the predecessor too (``PREWAIT_WEIGHTS`` off; the source never waits on the weight loader); (7) the dependent-launch wait and
the release of the successor are two knobs; (8) rank 0 arms the reduction barrier at the top of its epilogue role (the source: after
its accumulator read), and ``HOLD_TMEM`` lets the epilogue hold the tensor memory until it has consumed the registers it read from it
(off = the source's order: released right after the read); (9) named barrier counts, the cuDNN symbol prefix, and ``LayoutEnum``
bound through the tree's shim over the DSL names that moved between releases (the source reads the DSL's older alias).  Not carried:
the source's own configuration validation and autotune list, its batched views, split-K 3.

COORDINATES.  Kernel M = the weight rows ``N`` (``MMA_M`` 64 or 128 per CTA), kernel N = the token rows (one token tile of 8 / 16 / 32
per CTA; rows past ``rows`` are outside the activation's tensor map and load as zeros), kernel K = ``K`` in 128-element K tiles.  Grid
``(N / MMA_M, SPLIT_K, 1)`` with clusters ``(1, SPLIT_K, 1)``: the CTAs of a cluster are the K slices of one weight-row tile, rank ``r``
walks the K tiles ``r T .. (r + 1) T - 1`` with ``T = K / 128 / SPLIT_K``.  256 threads: warp 0 streams the weight K tiles, warp 1 the
activation K tiles, warp 2 issues the MMAs, warp 3 idles, warps 4-7 are the epilogue.

THE FIXED ACCUMULATION ORDER (what the bytes are a function of).  Per output element ``(t, n)``:

* slice sum ``acc_r`` = one fp32 tensor-memory cell: the chain over the slice's K tiles in ascending K of the chain over the four MMA
  instructions of a tile in ascending K of ``mma_k32(weight[n, .], activation[t, .])`` -- accumulate off for the very first instruction
  only.  The order and rounding inside one MMA instruction are the tensor core's.  The kernel asserts at trace that a K tile is four
  instructions (``MMAS_PER_K_TILE``): another instruction K would be another chain.
* ``s = ((acc_0 + acc_1) + acc_2) + acc_3`` on rank 0, fp32 adds in rank order (nothing at ``SPLIT_K`` 1).
* ``out = bf16(s * scale)``: one fp32 multiply, one round-to-nearest-even cast; no saturation (a value past the bf16 range is inf).

``SPLIT_K`` sets the number of slice sums and adds: a numerics attribute.  ``MMA_M``, ``STAGES``, ``TMEM_COLS``, the token tile,
``PREWAIT_WEIGHTS``, ``PDL`` and ``PDL_RELEASE`` move no addition.  Why a row's bytes do not depend on the other rows: a token occupies
one column of the accumulator tile, the MMA computes a column from that column of the activation operand alone, the rows past ``rows`` are
zeros that are never stored, and the grid, the K partition and every reduction depend on the geometry and ``SPLIT_K`` only.  That the
tensor core rounds a column's contraction the same in every column and for every token-tile width is a PREMISE; the batch-invariance
tests (a row alone against the same row among 2 .. 32, across the three token tiles) are its tripwire.

BARRIER TABLE (all ``cta_group::1``; ``SUM(issuing lanes) == init`` on every path):

  bar_full[s], s < STAGES   init FULL_BAR_ARRIVERS = 2.  Arrive: warp 0, ONE elected lane, ``arrive.expect_tx(MMA_M x 128 B)`` and warp 1,
                            ONE elected lane, ``arrive.expect_tx(token tile x 128 B)`` (the role branch AND the elect); the bytes by the
                            stage's two tensor-map loads.  Wait: warp 2 before the stage's MMAs; phase 0 first, flipping per ring lap.
  bar_empty[s]              init EMPTY_BAR_ARRIVERS = 1.  Arrive: warp 2, ONE elected ``tcgen05.commit`` after the stage's four MMAs.  Wait:
                            warp 0 and warp 1 before they refill the stage -- the first lap is pre-armed (their phase starts at 1) -- and
                            once more per stage in the drain after the last issue, so no commit is left unwaited.  Each loader has its
                            own back-edge: a stage's next commit needs BOTH loaders' arrives on ``bar_full``.
  bar_mma_epilog            init MMA_DONE_BAR_ARRIVERS = 1.  Arrive: warp 2, ONE elected ``tcgen05.commit`` after the last K tile.  Wait: the
                            128 epilogue threads, before they read the accumulator.
  bar_tmem_alloc            init TMEM_BAR_ARRIVERS = 32 + 128, a rendezvous with two phases.  Phase 0: warp 2's 32 lanes after
                            ``tcgen05.alloc`` + the 128 epilogue threads at entry; waited by the epilogue threads (the tensor-memory base
                            pointer is published).  Phase 1: warp 2's 32 lanes after the last commit + the 128 epilogue threads right
                            after their accumulator read and its ``tcgen05.wait`` (the read is complete before the columns can be
                            deallocated) or, under ``HOLD_TMEM``, as the LAST instruction of their role, after every use of the
                            registers that read filled (the remote stores on ranks >= 1; the reduction, the scale and the output
                            store on rank 0).  Waited by warp 2, which then deallocates.  Warp 2 waits for
                            phase 0 in front of its phase-1 arrive: a parity-1 wait issued while phase 0 is still open returns at once,
                            so the order of the two phases is enforced by that wait and does not rest on when the epilogue threads
                            enter.  Phase 0 then holds exactly warp 2's first 32 arrivals and the 128 entry arrivals, phase 1 warp 2's
                            second 32 and the 128 release arrivals.
  bar_reduce (SPLIT_K > 1)  init REDUCE_BAR_ARRIVERS = 1 in every CTA, waited on rank 0 only.  Arrive: rank 0, epilogue thread 0,
                            ``arrive.expect_tx((SPLIT_K - 1) x MMA_M x token tile x 4 B)``, the FIRST instruction of its role (above
                            every shared- and tensor-memory read of the role, and normally before any peer ships); the bytes by every
                            epilogue thread of every other rank, 16 B per remote store.  Wait: rank 0's 128 epilogue threads before
                            they read the mailbox.  Nothing orders a peer's bytes behind the expect; the pending count is signed, so
                            either order completes the phase.
  init                      warp 0, ONE elected lane, every barrier; ``fence.mbarrier_init`` by all; then the cluster arrive / wait by
                            every thread (the CTA barrier at SPLIT_K 1) BEFORE any cross-CTA store.

SMEM BUFFER TABLE (allocation order = ``smem_bytes`` of the config, which the launch reserves):

  sA         MMA_M x 128 x STAGES B, align 1024.  Written by the weight tensor-map load, read by the MMA descriptor: the tensor map and
             the descriptor are derived from ONE layout object (``make_smem_layout_a``), so the swizzle the load writes is the one the
             descriptor reads.
  sB         token tile x 128 x STAGES B, align 1024.  The activation ring; ``make_smem_layout_b``, same argument.
  bar_full, bar_empty   STAGES x 8 B each; bar_mma_epilog, bar_tmem_alloc 8 B each.
  tmem_base_ptr   4 B.  Written once by ``tcgen05.alloc``, read by warp 2 and by the epilogue threads behind ``bar_tmem_alloc`` phase 0.
  mailbox    (SPLIT_K - 1) x MMA_M x token tile x 4 B fp32, align 128, slot ``[rank - 1][epilogue thread][value]``; SPLIT_K > 1 only.
             Written by the PEERS' remote stores (16 B each) from the cluster sync on, read on rank 0 behind ``bar_reduce`` by generic
             16-byte loads: a thread reads exactly the slots the same-numbered thread of each peer wrote.  Its own bytes: it never
             aliases a ring a peer could still overwrite.  No descriptor reads it and it has no swizzle; a lane's stride is
             ``values per thread x 4`` B = 16 / 32 / 64 B at token tiles 8 / 16 / 32, i.e. the natural stride of a 16-byte load at
             tile 8 and two / four times that at tiles 16 / 32 (a bank-conflict cost of the reduction at 9 .. 32 rows, not a
             correctness fact).
  bar_reduce 8 B; SPLIT_K > 1 only.

TENSOR MEMORY.  Warp 2 allocates ``TMEM_COLS`` columns at entry (the accumulator is one column per token row; 32 is the allocation
granule and the default at every token tile) and deallocates the same constant after the rendezvous.  An SM has 512 columns; the config
declines a form whose shared-memory co-residency times ``TMEM_COLS`` exceeds them.  With ``HOLD_TMEM`` off (a form's default and the
pick of the one-wave classes) the epilogue releases the columns right after ``tcgen05.ld`` + ``tcgen05.wait``, so the deallocation
overlaps rank 0's wait for its peers while the loaded registers are still unread; ``HOLD_TMEM`` (the pick of every other class, whose
grids can take more than one wave, so that the freed columns can go to another CTA of the same launch) releases them only after the
last use of those registers, at the price of a deallocation that starts after the output store.

PDL SITES.  ``PDL``: the launch carries the programmatic-stream-serialization attribute; warp 1 issues ``griddepcontrol.wait`` as its
first instruction, i.e. BEFORE every read of ``activation``.  Above that wait run the barrier inits, the cluster sync, warp 2's
tensor-memory allocation and up to ``STAGES`` weight K tiles per CTA of warp 0 (``PREWAIT_WEIGHTS``; off = warp 0 waits too).  The MMA and
the epilogue are ordered behind the predecessor through the barrier chain alone (``bar_full`` needs warp 1's arrive): the read of
``scale`` and the write of ``out`` sit below the epilogue's ``bar_mma_epilog`` wait, not below a wait of their own.  CALLER OBLIGATION:
no kernel that can still be in flight when this launch starts may write ``weight`` or ``scale``.  ``PDL_RELEASE``: warp 0 issues
``griddepcontrol.launch_dependents`` after its LAST weight issue, before its drain.  The release creates an obligation for the CONSUMER,
not for this kernel: a grid launched behind it WITH the attribute may start while this kernel runs and must issue its own wait before it
reads ``out``; a consumer launched without the attribute is unaffected.  When a K slice fits the ring (``T <= STAGES``) the release can
precede this CTA's own ``tcgen05.alloc``: a consumer that allocates tensor memory above its wait must fit beside this kernel's columns
(at most ``SMEM_RESIDENT x TMEM_COLS`` per SM, 96 at the default forms) -- with the default 32 columns two instances of this kernel
chain by construction.

THE SYMBOL.  ``cudnn_kernel_frost_fp8_projection_decode_<text>_...``: the DSL appends the text of the kernel object and of the operand
types to the kernel's name and cuts the result.  ``<text>`` is the rendering's ``Cfg.SYMBOL_TEXT`` (geometry, split, rows, weight
tile, stages, token tile, tensor-memory columns and the four switches) for every class but those of ``SHAPE_SYMBOL_CLASSES``, whose
symbol names the weight's shape only -- there every rendering of one geometry shares a symbol, and the form is what the host entry's
``form_for`` states.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from cudnn._cutlass_compat import LayoutEnum

from .fp8_projection_decode_config import (
    EMPTY_BAR_ARRIVERS,
    EPILOGUE_THREADS,
    FULL_BAR_ARRIVERS,
    K_TILE,
    MAILBOX_ALIGN_BYTES,
    MBARRIER_BYTES,
    MMA_DONE_BAR_ARRIVERS,
    MMAS_PER_K_TILE,
    OPERAND_ALIGN_BYTES,
    OPERAND_RING_ALIGN_BYTES,
    REDUCE_BAR_ARRIVERS,
    REMOTE_STORE_VALUES,
    THREADS_PER_CTA,
    TMEM_BAR_ARRIVERS,
    TMEM_POINTER_BYTES,
    Cfg,
)

#: the cluster rank (K slice) that gathers the partial sums and stores the output
OWNER_RANK = 0


@dsl_user_op
def _map_shared_rank(smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None) -> Int32:
    """A shared-memory address of this CTA as the same address in a peer CTA of the cluster."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value(), peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_v4(
    value0, value1, value2, value3, smem_ptr: cute.Pointer, mbar_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> None:
    """Four fp32 partial sums into a peer's shared memory, completing 16 transaction bytes on the peer's barrier."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class Fp8ProjectionDecodeKernel:
    """One rendering of the kernel for a ``Cfg`` (its geometry, split, row count and form are trace-time constants)."""

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.acc_dtype = cutlass.Float32
        self.cta_m = cfg.MMA_M
        self.cta_n = cfg.TOKEN_TILE
        self.cta_k = K_TILE
        self.num_ab_stage = cfg.STAGES
        self.split_k = cfg.SPLIT_K
        self.use_pdl = cfg.PDL
        self.pdl_release = cfg.PDL_RELEASE
        self.prewait_weights = cfg.PREWAIT_WEIGHTS
        self.tmem_cols = cfg.TMEM_COLS
        self.hold_tmem = cfg.HOLD_TMEM
        self.shared_bytes = cfg.SMEM_BYTES

        self.threads_per_cta = THREADS_PER_CTA
        self.epilog_threads = EPILOGUE_THREADS
        self.mma_tiler_mn = (cfg.MMA_M, cfg.TOKEN_TILE)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, cfg.SPLIT_K, 1)

        if cfg.VALUES_PER_THREAD % REMOTE_STORE_VALUES:
            raise ValueError(
                f"CTA tile ({cfg.MMA_M}, {cfg.TOKEN_TILE}) gives {cfg.VALUES_PER_THREAD} values per epilogue thread; remote stores require a "
                f"multiple of {REMOTE_STORE_VALUES}"
            )
        self.mailbox_elements = cfg.MAILBOX_ELEMENTS
        self.expected_transaction_bytes = cfg.REDUCE_TX_BYTES

    def __str__(self) -> str:
        """The text the DSL writes into the kernel symbol for this object.  A kernel symbol is the kernel's name followed by the text of
        its arguments -- this object first -- cut at a fixed length; with the rendering's ``SYMBOL_TEXT`` here the symbol names the
        rendering in its first 100 characters.  A class whose ``SYMBOL_TEXT`` is empty keeps the default text of an object, i.e. the
        symbol its renderings were validated with."""
        return self.cfg.SYMBOL_TEXT or object.__repr__(self)

    def _check_mmas_per_k_tile(self, mma_inst_tile_k: int) -> int:
        """Trace-time: one K tile is ``MMAS_PER_K_TILE`` MMA instructions (the accumulation order the bytes are a function of)."""
        if mma_inst_tile_k != MMAS_PER_K_TILE:
            raise NotImplementedError(
                f"the FP8 projection decode kernel is validated with {MMAS_PER_K_TILE} MMA instructions per {K_TILE}-element K tile; this CuTe DSL "
                f"and target render {mma_inst_tile_k}"
            )
        return mma_inst_tile_k

    @cute.experimental.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, alpha: cute.Tensor, stream: cuda.CUstream):
        # Grid y is the K slice (the cluster rank) of the one token tile.
        self.frost_fp8_projection_decode(a, b, c, alpha).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.shared_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def frost_fp8_projection_decode(
        self,
        mA: cute.Tensor,  # the weight: (N, K, 1), K-major
        mB: cute.Tensor,  # the activation: (rows, K, 1), K-major
        mC: cute.Tensor,  # the output viewed (N, rows, 1), N-major
        mAlpha: cute.Tensor,  # the scale: one fp32
    ):
        """Allocate the shared memory, initialise the barriers, synchronise the cluster and dispatch the warp roles."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            LayoutEnum.from_tensor(mA).mma_major_mode(),
            LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=OPERAND_RING_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=OPERAND_RING_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(stages), alignment=MBARRIER_BYTES).iterator
        bar_empty = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(stages), alignment=MBARRIER_BYTES).iterator
        bar_mma_epilog = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=MBARRIER_BYTES).iterator
        bar_tmem_alloc = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=MBARRIER_BYTES).iterator
        tmem_base_ptr = cute_ext.allocate(cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=TMEM_POINTER_BYTES).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(cutlass.Float32, cute.AddressSpace.smem, cute.make_layout(self.mailbox_elements), alignment=MAILBOX_ALIGN_BYTES)
            bar_reduce = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=MBARRIER_BYTES).iterator
        else:
            # Placeholders: the reduction is elided at trace time for one K slice.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, FULL_BAR_ARRIVERS)
                    cute.arch.mbarrier_init(bar_empty + i, EMPTY_BAR_ARRIVERS)
                cute.arch.mbarrier_init(bar_mma_epilog, MMA_DONE_BAR_ARRIVERS)
                cute.arch.mbarrier_init(bar_tmem_alloc, TMEM_BAR_ARRIVERS)

                if cutlass.const_expr(self.split_k > 1):
                    # Rank 0's arrive; the peers complete its transaction bytes.
                    cute.arch.mbarrier_init(bar_reduce, REDUCE_BAR_ARRIVERS)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Every CTA's barriers are initialised before any cross-CTA store.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # The config validates an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self._check_mmas_per_k_tile(self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                mAlpha,
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
            )

    frost_fp8_projection_decode.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(self.use_pdl and (not is_a or not self.prewait_weights)):
            cute.arch.griddepcontrol_wait()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(is_a and self.pdl_release):
            cute.arch.griddepcontrol_launch_dependents()
        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def _drain_producer(self, bar_empty, empty_phase: cutlass.Int32, k_tile_count: cutlass.Int32):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        num_tmem_cols = self.tmem_cols
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        # The release rendezvous is phase 1 of this barrier, and a parity-1 wait issued while phase 0 is still open returns at once.
        # Wait for phase 0 first (this warp's 32 arrivals after the allocation + the 128 epilogue entry arrivals): the 32 arrivals
        # below then belong to phase 1 by construction, and the wait behind them is the wait for the 128 release arrivals.
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        mAlpha: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
    ):
        if cutlass.const_expr(self.split_k > 1):
            # Rank 0 arms the reduction barrier with the peers' byte count as the FIRST instruction of the role: above every
            # shared-memory and tensor-memory read of the role, so no read of this role has this arrive between it and its first use,
            # and the expect is normally in place before a peer ships.
            if split_rank == OWNER_RANK:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, self.expected_transaction_bytes)

        # Wait until the MMA warp has published the tensor-memory base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout)[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op((self.cta_m, self.cta_n, self.cta_k), d_layout, c_dtype, self.acc_dtype, epi_tile, False),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Each epilogue thread's registers mirror its partition of the accumulator tile.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # tcgen05.wait: the tensor-memory read is complete before the columns are released and before the registers are used.
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(not self.hold_tmem):
            # Release the tensor memory now: its deallocation overlaps the reduction.
            cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Ranks >= 1 ship their fp32 partial sums to rank 0; rank 0 adds them in rank order and stores.
        if cutlass.const_expr(self.split_k > 1):
            assert cute.size(rmem_layout) == self.mailbox_elements // ((self.split_k - 1) * self.epilog_threads)
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator + (split_rank - Int32(1)) * values_per_peer + epi_tid * values_per_thread + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = rAcc[value_idx] + mailbox[peer * values_per_peer + epi_tid * values_per_thread + value_idx]

        if split_rank == OWNER_RANK:
            rD.store((rAcc.load() * mAlpha[0]).to(c_dtype))
            # The copy keeps the accumulator's coordinates and predicates the rows past the output's bounds.
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        if cutlass.const_expr(self.hold_tmem):
            # Release the tensor memory LAST: every use of the accumulator registers (the remote stores on ranks >= 1, the reduction,
            # the scale and the output store on rank 0) is above this arrive, so the columns are still this CTA's while they are consumed.
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
        # The peers' stores have a register source and rank 0 waited their bytes: no cluster barrier at exit.


@cute.experimental.jit
def launch(
    gemm: cutlass.Constexpr,
    weight: cute.Tensor,
    activation: cute.Tensor,
    out: cute.Tensor,
    alpha: cute.Tensor,
    stream: cuda.CUstream,
):
    # The E4M3 operands cross the launch as bytes and are recast here: their bytes reach the kernel untouched.
    w = cute.make_tensor(
        cute.recast_ptr(weight.iterator, dtype=cutlass.Float8E4M3FN),
        cute.make_layout((weight.shape[0], weight.shape[1], 1), stride=(weight.stride[0], weight.stride[1], 0)),
    )
    x = cute.make_tensor(
        cute.recast_ptr(activation.iterator, dtype=cutlass.Float8E4M3FN),
        cute.make_layout((activation.shape[0], activation.shape[1], 1), stride=(activation.stride[0], activation.stride[1], 0)),
    )
    y = cute.make_tensor(out.iterator, cute.make_layout((out.shape[1], out.shape[0], 1), stride=(out.stride[1], out.stride[0], 0)))
    gemm(w, x, y, alpha, stream)


def compile_kernel(cfg: Cfg):
    """Trace-compile one rendering (no operands: storage-free declarations).  The caller pins the CUDA device and ran the DSL gate."""
    fake = cute.runtime.make_fake_compact_tensor
    kernel = Fp8ProjectionDecodeKernel(cfg)
    # Keep the singleton row axis in the activation tensor map: a static one-row
    # activation folds into a 1-D tensor map although the shared-memory box is 2-D.
    return cute_ext.compile(
        launch,
        kernel,
        fake(cutlass.Uint8, (cfg.N, cfg.K), stride_order=(1, 0), assumed_align=OPERAND_ALIGN_BYTES),
        fake(cutlass.Uint8, (cute.sym_int32(), cfg.K), stride_order=(1, 0), assumed_align=OPERAND_ALIGN_BYTES),
        fake(cutlass.BFloat16, (cfg.ROWS, cfg.N), stride_order=(1, 0), assumed_align=OPERAND_ALIGN_BYTES),
        fake(cutlass.Float32, (1,), assumed_align=4),
        make_fake_stream(),
    )


def launch_args(weight_u8, activation_u8, out, scale):
    """The DSL views of one launch's operands (byte views of the two E4M3 tensors, ``out``, the one-element ``scale``): host objects
    only, no device work.  The activation's row extent is the compiled entry's one symbolic dimension."""
    return (
        from_dlpack(weight_u8.detach(), assumed_align=OPERAND_ALIGN_BYTES),
        from_dlpack(activation_u8.detach(), assumed_align=OPERAND_ALIGN_BYTES).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1)),
        from_dlpack(out.detach(), assumed_align=OPERAND_ALIGN_BYTES),
        from_dlpack(scale.detach(), assumed_align=4),
    )
