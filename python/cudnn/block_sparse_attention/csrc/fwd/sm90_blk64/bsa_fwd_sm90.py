# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as hopper_helpers
from cutlass._mlir.dialects import math as _math
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator
from cudnn.block_sparse_attention.csrc.utils.batched_static_scheduler import (
    BatchedStaticSchedulerMixin,
    BatchedStaticWorkDesc,
)

SM90_FWD_BLOCK_SIZE = 64
# |max * c| bound below which the softmax FFMA stays max-centered to 2^-14 (see inc_softmax_ffma).
FFMA_MAX_SCALED_ABS = 2048.0


@dataclass
class SplitBatchedStaticWorkDesc(BatchedStaticWorkDesc):
    split_idx: int


class SplitBatchedStaticSchedulerMixin(BatchedStaticSchedulerMixin):
    def get_split_grid_config(self, seqlen_q, num_qo_heads, batch_size):
        tile_size_m = self.tile_shape_qk[0]
        num_q_tiles = cute.ceil_div(seqlen_q, tile_size_m)
        return (num_q_tiles, self.num_splits, num_qo_heads * batch_size)

    def get_split_work_desc(self, num_qo_heads):
        qo_tile_idx, split_idx, batch_head_idx = cute.arch.block_idx()
        qo_head_idx = batch_head_idx % num_qo_heads
        batch_idx = batch_head_idx // num_qo_heads
        kv_head_idx = qo_head_idx // self.gqa_ratio
        return SplitBatchedStaticWorkDesc(qo_tile_idx, qo_head_idx, kv_head_idx, batch_idx, split_idx)


# =============================================================================
# Public kernel class
# =============================================================================
class BlockSparseAttnForwardSm90Blk64(SplitBatchedStaticSchedulerMixin):
    def __init__(
        self,
        gqa_ratio: int = 1,
        head_dim: int = 128,
        value_dim: int = 128,
        blocksparse_blocksize_q: int = 64,
        blocksparse_blocksize_k: int = 64,
        dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        has_block_sizes: bool = True,
        num_splits: cutlass.Constexpr[int] = 1,
        allow_empty_block_nums: bool = False,
        nonpositive_scale: bool = False,
        has_tile_order: bool = False,
        row_aware_boxes: Optional[bool] = None,
        lean_consumer_loop: Optional[bool] = None,
        window_prefetch: Optional[bool] = None,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        assert self.dtype in [cutlass.Float16, cutlass.BFloat16], "SM90 blk64 fwd supports fp16/bf16"
        assert self.acc_dtype in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]

        self.tile_size = 64

        assert blocksparse_blocksize_q == 64, "Only block_size_m=64 is supported in this kernel."
        assert blocksparse_blocksize_k in [64], "block_size_n should be one of [64]"
        assert gqa_ratio >= 1
        assert head_dim in (64, 96, 128), "SM90 blk64 fwd supports QK dim 64, 96, or 128"
        assert value_dim in (64, 96, 128), "SM90 blk64 fwd supports value dim 64, 96, or 128"
        self.gqa_ratio = gqa_ratio
        self.qk_dim = head_dim
        self.value_dim = value_dim
        self.tile_shape_qk = (self.tile_size, self.tile_size, self.qk_dim)
        self.tile_shape_pv = (self.tile_size, self.value_dim, self.tile_size)

        assert num_splits >= 1, "num_splits must be >= 1"
        self.num_splits = num_splits
        self.use_tma_o = self.value_dim <= head_dim and self.num_splits == 1
        # Non-split uses two V slots to overlap QK[j] with PV[j-1]. Split waits for each PV
        # before reusing one slot; its TMA/PV views and barrier expect a single 2D V tile.
        # Setting both paths to 2 would break split's view contract and add unused SMEM.
        self.v_stages = 2 if self.num_splits == 1 else 1
        self.min_blocks_per_mp = 3 if self.num_splits == 1 else 4
        self.has_block_sizes = has_block_sizes
        # Compile-time specialization: only the empty-enabled variant pays for the
        # empty-safe finalizer. The non-split kernel's num_n_tiles > 0 guard is unconditional.
        self.allow_empty_block_nums = allow_empty_block_nums
        # softmax_scale <= 0 (or NaN): max(scale * S) is not scale * max(S), and a masked -inf times the
        # scale is not -inf. The scale then multiplies the scores before the masks and the softmax uses 1.
        self.nonpositive_scale = nonpositive_scale
        # Longest-list-first CTA order (variable row lengths, non-split kernel only): CTA x of head h processes
        # Q tile blocksparse_tile_order[x, h, b], a per-(b, h) permutation built on the host, so the long rows
        # start first and the grid's tail is short rows. Off, the kernel keeps the identity order and its code.
        assert not has_tile_order or num_splits == 1, "tile_order is only for the non-split kernel"
        self.has_tile_order = has_tile_order
        # Row-count-aware K/V boxes (P4; non-split kernel with block sizes): a KV block with v valid rows is loaded with a
        # 16 * ceil(v / 16)-row TMA box per swizzle plane instead of all 64 rows, which cuts the L2->SMEM fill bytes the
        # forward is bound by. MMA shapes, the mask and the split kernel are unchanged; off, the kernel keeps its code.
        if row_aware_boxes is None:
            row_aware_boxes = has_block_sizes and num_splits == 1
        assert not row_aware_boxes or (has_block_sizes and num_splits == 1), "row_aware_boxes needs block sizes and the non-split kernel"
        self.row_aware_boxes = row_aware_boxes
        # Lean consumer loop (P10; non-split kernel): the O rescale between the QK[j] and PV[j-1] issues is skipped by one
        # warp vote when no row of the warp changed its running max (its ratio is then exactly 1.0, so the skip is
        # bit-exact), and the partial-block mask compares each column against one per-thread limit instead of
        # rebuilding every column coordinate. Off, the kernel keeps its code; the split kernel and the FP8, blk128 and
        # KPage kernels (which share this file's helpers) are unchanged either way.
        if lean_consumer_loop is None:
            lean_consumer_loop = num_splits == 1
        assert not lean_consumer_loop or num_splits == 1, "lean_consumer_loop is only for the non-split kernel"
        self.lean_consumer_loop = lean_consumer_loop
        # Windowed list prefetch (P10; with row-aware boxes): lane l of every warp keeps the list index and the block size
        # of step 32w + l for the current window and the next one, and a step reads idx[j+1]/bsz[j+1] with two warp
        # shuffles; one coalesced index load and one size gather per 32 steps replace the per-step load pair, whose
        # uniform-register conversion ptxas scheduled right after the load (a long-scoreboard stall at every loop top).
        if window_prefetch is None:
            window_prefetch = self.row_aware_boxes
        assert not window_prefetch or self.row_aware_boxes, "window_prefetch builds on row_aware_boxes"
        self.window_prefetch = window_prefetch

    def check_dim(self, tensor: cute.Tensor | list[cute.Tensor], mode: int):
        if isinstance(tensor, list):
            for t in tensor:
                self.check_dim(t, mode)
            return
        assert tensor.shape[mode] in [64, 96, 128], f"dim must be one of [64, 96, 128] in mode {mode}."
        assert tensor.stride[mode] == 1, f"dim must be contiguous in mode {mode}."

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tma_atom_K16: Optional[cute.CopyAtom],
        tma_atom_K32: Optional[cute.CopyAtom],
        tma_atom_K48: Optional[cute.CopyAtom],
        tma_atom_V16: Optional[cute.CopyAtom],
        tma_atom_V32: Optional[cute.CopyAtom],
        tma_atom_V48: Optional[cute.CopyAtom],
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_num_blocks_q2k: cute.Tensor,
        blocksparse_varblk: cute.Tensor,
        blocksparse_tile_order: Optional[cute.Tensor],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        O_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
    ):
        score_scale = scale_softmax_log2e
        if cutlass.const_expr(self.nonpositive_scale):
            scale_softmax_log2e = cutlass.Float32(1.0)
        # Single-stage Q/K, double-buffered V: overlap QK[j] with PV[j-1].
        # Reuse K after wait_group(1); reuse V and update P/O after wait_group(0).
        # Reuse Q storage for O only after the final drain.
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        work_desc = self.get_work_desc()
        if cutlass.const_expr(self.has_tile_order):
            # One LDG per CTA before anything that depends on the Q tile; the shuffle keeps the index uniform.
            qo_tile_idx = cute.arch.make_warp_uniform(blocksparse_tile_order[work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx])
            work_desc = BatchedStaticWorkDesc(qo_tile_idx, work_desc.qo_head_idx, work_desc.kv_head_idx, work_desc.batch_idx)
        seqlen = mK.shape[0]
        num_compute_tiles = cute.ceil_div(seqlen, self.tile_size)

        shared_storage = SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0 and lane_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)

            if cutlass.const_expr(self.use_tma_o):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)
            if cutlass.const_expr(self.row_aware_boxes):
                for atom in (tma_atom_K16, tma_atom_K32, tma_atom_K48, tma_atom_V16, tma_atom_V32, tma_atom_V48):
                    cute.nvgpu.cpasync.prefetch_descriptor(atom)

        op = pipeline.PipelineOp.TmaLoad
        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Single-stage barriers for Q/K, two-stage ring for V
        Q_barrier = (pipeline.MbarrierArray(shared_storage.Q_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        K_barrier = (pipeline.MbarrierArray(shared_storage.K_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        V_barrier = (pipeline.MbarrierArray(shared_storage.V_barrier.data_ptr(), num_stages=self.v_stages, agent=(op, cg)),)

        # partition tensors
        sQ = shared_storage.Q_smem.get_tensor(Q_smem_layout.outer, swizzle=Q_smem_layout.inner)
        sK = shared_storage.K_smem.get_tensor(K_smem_layout.outer, swizzle=K_smem_layout.inner)
        sV = shared_storage.V_smem.get_tensor(V_smem_layout.outer, swizzle=V_smem_layout.inner)

        mO_slice = mO[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mLSE_slice = mLSE[None, work_desc.qo_head_idx, work_desc.batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        # The K/V TMA coordinates are loop-invariant, but ptxas re-reads SR_CTAID (S2UR) for them inside
        # every per-step TMA issue branch. Routed through a warp shuffle they stay in registers (seen in SASS, not guaranteed).
        kv_head_u = cute.arch.make_warp_uniform(work_desc.kv_head_idx)
        batch_u = cute.arch.make_warp_uniform(work_desc.batch_idx)
        mK_slice = mK[None, None, kv_head_u, batch_u]
        mV_slice = mV[None, None, kv_head_u, batch_u]
        gO = cute.local_tile(mO_slice, (self.tile_shape_pv[0], self.tile_shape_pv[1]), coord=(work_desc.qo_tile_idx, 0))
        gQ = cute.local_tile(mQ_slice, (self.tile_shape_qk[0], self.tile_shape_qk[2]), coord=(work_desc.qo_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (self.tile_shape_qk[1], self.tile_shape_qk[2]), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (self.tile_shape_pv[1], self.tile_shape_pv[2]), coord=(0, None))

        gIndices = blocksparse_indices_q2k[None, work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        # Without block sizes this views the wrapper's placeholder, which block_len never reads.
        gBSZ = blocksparse_varblk[None, work_desc.qo_head_idx, work_desc.batch_idx]

        cta_coord_layout = (0, cute.make_layout(1))  # CTA coord layout for TMA multicasting, effectively no multicast
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, *cta_coord_layout, cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(tma_atom_K, *cta_coord_layout, cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(tma_atom_V, *cta_coord_layout, cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSrQ = tiled_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((self.tile_shape_qk[0], self.tile_shape_qk[1])), self.acc_dtype)
        tScS = thr_mma_qk.partition_C(cS)
        tScS0 = None
        if cutlass.const_expr(self.lean_consumer_loop):
            # Thread 0's coordinate partition: its column of accumulator element (m, n) is a trace-time constant, and every
            # thread's column is that constant plus the thread's own first column (the wgmma C layout only adds
            # (lane % 4) * 2 to a column), so the mask compares one per-thread limit against immediates (mask_from_limit).
            tScS0 = tiled_mma_qk.get_slice(0).partition_C(cS)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOrV = tiled_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV))
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((self.tile_shape_pv[0], self.tile_shape_pv[1])), self.acc_dtype)
        tOrP = cute.make_rmem_tensor_like(convert_c_layout_to_a_layout(tSrS.layout, tiled_mma_pv.tv_layout_A.shape[1]), self.K_dtype)

        max_m_layout = cute.make_layout(cute.size(layout_acc_mn(tiled_mma_pv, tOrO.layout), mode=[0]))
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)

        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        # Q does not depend on the list: issue it before the count and index loads resolve. The count is
        # loaded only after this branch in source order: loaded before it, ptxas hoisted its `> 0` compare
        # above the branch, so the Q TMA waited for the count.
        if warp_idx == 0:
            Q_barrier[0].arrive_and_expect_tx(index=0, tx_count=self.tma_copy_bytes["Q"])
            cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=Q_barrier[0].get_barrier(0))

        num_n_tiles = blocksparse_num_blocks_q2k[work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]

        # CTA-uniform guard: the prologue and the drain wait on loads that are only issued when the
        # list is non-empty. Empty rows fall through to the finalizer (O=0, LSE=-inf when empty-safe).
        if num_n_tiles > 0:
            # First processed slot is the last active one; the inactive suffix is never read.
            n_tile_idx_ = gIndices[num_n_tiles - 1]

            if warp_idx == 0:
                K_barrier[0].arrive_and_expect_tx(index=0, tx_count=self.tma_copy_bytes["K"])
                cute.copy(tma_atom_K, tKgK[None, n_tile_idx_], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

                V_barrier[0].arrive_and_expect_tx(index=0, tx_count=self.tma_copy_bytes["V"])
                cute.copy(tma_atom_V, tVgV[None, n_tile_idx_], tVsV[None, 0], tma_bar_ptr=V_barrier[0].get_barrier(0))

            cute.arch.sync_threads()

            Q_barrier[0].wait(index=0, phase=0)
            self.mainloop(
                n_tile_idx_=n_tile_idx_,
                num_n_tiles=num_n_tiles,
                gIndices=gIndices,
                gBSZ=gBSZ,
                seqlen=seqlen,
                num_compute_tiles=num_compute_tiles,
                warp_idx=warp_idx,
                tma_atom_K=tma_atom_K,
                tma_atom_V=tma_atom_V,
                tKgK=tKgK,
                tKsK=tKsK,
                tVgV=tVgV,
                tVsV=tVsV,
                K_barrier=K_barrier,
                V_barrier=V_barrier,
                tiled_mma_qk=tiled_mma_qk,
                tiled_mma_pv=tiled_mma_pv,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tSrS=tSrS,
                tScS=tScS,
                tScS0=tScS0,
                tOrV=tOrV,
                tOrO=tOrO,
                tOrP=tOrP,
                max_m=max_m,
                sum_m=sum_m,
                scale_softmax_log2e=scale_softmax_log2e,
                score_scale=score_scale,
                tma_atoms_K_part=(tma_atom_K16, tma_atom_K32, tma_atom_K48),
                tma_atoms_V_part=(tma_atom_V16, tma_atom_V32, tma_atom_V48),
                mK_slice=mK_slice,
                mV_slice=mV_slice,
                sK=sK,
                sV=sV,
            )
        else:
            # Empty row: Q was still loaded. Let it land before its SMEM is reused for O staging or the
            # CTA exits (the sync publishes the barrier initialization to every waiting thread).
            cute.arch.sync_threads()
            Q_barrier[0].wait(index=0, phase=0)

        if cutlass.const_expr(self.allow_empty_block_nums):
            final_ratio, lse = get_final_ratio_and_lse_empty_safe(max_m, sum_m, scale_softmax_log2e)
        else:
            final_ratio, lse = get_final_ratio_and_lse(max_m, sum_m, scale_softmax_log2e)
        rescale_o_for_next_acc(tiled_mma_pv, tOrO, final_ratio)
        tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(tiled_mma_qk, tScS.layout))
        for m in cutlass.range_constexpr(cute.size(lse)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            if row_idx < mQ.shape[0]:
                mLSE_slice[row_idx] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        if cutlass.const_expr(self.use_tma_o):
            # R2S
            sO = shared_storage.Q_smem.get_tensor(O_smem_layout.outer, swizzle=O_smem_layout.inner)
            tiled_copy_o_r2s = cute.make_tiled_copy_C(
                cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), self.O_dtype),
                tiled_mma_pv,
            )
            tOrO_cv = tiled_copy_o_r2s.retile(tOrO_cvt)
            tOsO = tiled_copy_o_r2s.get_slice(tidx).partition_D(sO)
            cute.copy(tiled_copy_o_r2s, tOrO_cv, tOsO)

            # S2G
            # Publish every warp's R2S writes to the TMA async proxy.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp_idx == 0:
                tma_tOsO, tma_tOgO = cute.nvgpu.cpasync.tma_partition(tma_atom_O, *cta_coord_layout, cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2))
                cute.copy(tma_atom_O, tma_tOsO, tma_tOgO)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            tiled_copy_o_r2g = cute.make_tiled_copy_C(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.O_dtype, num_bits_per_copy=32),
                tiled_mma_pv,
            )
            tOrO_cv = tiled_copy_o_r2g.retile(tOrO_cvt)
            tOgO = tiled_copy_o_r2g.get_slice(tidx).partition_D(gO)
            cute.autovec_copy(tOrO_cv, tOgO)

    @cute.jit
    def mainloop(
        self,
        *,
        n_tile_idx_: cutlass.Int32,
        num_n_tiles: cutlass.Int32,
        gIndices: cute.Tensor,
        gBSZ: cute.Tensor,
        seqlen: cutlass.Int32,
        num_compute_tiles: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        K_barrier,
        V_barrier,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tSrS: cute.Tensor,
        tScS: cute.Tensor,
        tOrV: cute.Tensor,
        tOrO: cute.Tensor,
        tOrP: cute.Tensor,
        max_m: cute.Tensor,
        sum_m: cute.Tensor,
        scale_softmax_log2e: cutlass.Float32,
        score_scale: cutlass.Float32,
        tma_atoms_K_part=None,
        tma_atoms_V_part=None,
        mK_slice: Optional[cute.Tensor] = None,
        mV_slice: Optional[cute.Tensor] = None,
        sK: Optional[cute.Tensor] = None,
        sV: Optional[cute.Tensor] = None,
        tScS0: Optional[cute.Tensor] = None,
    ):
        """QK[j] and PV[j-1] in flight together; wait_group(1) retires QK[j] before softmax[j].

        Row-aware boxes (self.row_aware_boxes): K[j+1] and V[j] are loaded with 16 * ceil(valid / 16) rows, so the issue
        path needs the next block's size ahead of time: the list index is fetched two steps ahead (idx[j+2] at the top of
        step j) and the block sizes two steps ahead as well (bsz[j+2] = gBSZ[idx[j+2]] after the K issue of step j, once
        that index has landed; bsz[j+1] is consumed by the K[j+1] issue of step j and by the V[j+1] issue and mask of step j+1).
        Warp 0 issues V[j] at the top of the step (as without row-aware boxes) and warp 1 issues K[j+1] after wait_group(1),
        so the two issues (each a box selection plus one two-plane copy) overlap on different warps instead of both
        lengthening warp 0's path. The prologue's K[0]/V[0] and step 1's V[1] (slot 1's first fill) are full 64-row loads,
        so every row of K and of both V slots always holds finite data from an earlier block of this CTA, and a shorter
        load only shrinks the written row range (see load_kv_block).

        Lean consumer loop (self.lean_consumer_loop): the O rescale runs only when a warp vote says some row's ratio is not
        1.0 (rescale_o_if_needed), and a partial block is masked by comparing a per-thread limit against the static column
        offsets of thread 0 (mask_from_limit) instead of rebuilding the 16 column coordinates per step.

        Windowed prefetch (self.window_prefetch): win_idx0/win_bsz0 hold, in lane l, idx/bsz of step 32w + l of the window
        that contains step j+1, win_idx1/win_bsz1 the next window. Step j shuffles idx[j+1]/bsz[j+1] out of lane (j+1) % 32;
        when (j+1) % 32 == 0 the windows rotate and the index of the window after it is loaded (one lane per step, clamped
        to slot 0 past the start of the list), and one step later its sizes are gathered, so neither load is waited for
        in the step that issues it. The prologue loads the first two windows before the K[0] wait.
        """
        # Every PV accumulates into O (zero-initialized): set once, so no MMA atom is loop-carried.
        tiled_mma_pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        if cutlass.const_expr(self.lean_consumer_loop):
            # This thread's first score column: the mask limit is varblk - col0, compared against thread 0's static columns.
            tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(tiled_mma_qk, tScS.layout))
            col0 = tScS_mn[0, 0][1]
        # The list base address is loop-invariant, but ptxas rematerializes it every iteration (4 LDC.64 +
        # CTAID reads + an IMAD chain). Routed through a warp shuffle it stays in registers (seen in SASS, not guaranteed).
        gIdx = hoist_list_base(gIndices)
        if cutlass.const_expr(self.row_aware_boxes):
            # The block-size base address is loop-invariant as well; the size load after the K issue below would otherwise
            # rebuild it from the CTA id every step (9 instructions in every warp).
            gBsz = hoist_list_base(gBSZ)
        # ---- prologue: QK[0] and softmax[0]; O is still zero, so it needs no rescale ----
        n_tile_idx = n_tile_idx_
        n_tile_idx_next = -1
        if num_n_tiles > 1:
            n_tile_idx_next = gIndices[num_n_tiles - 2]
        varblk = self.block_len(n_tile_idx, gBSZ, seqlen, num_compute_tiles)
        if cutlass.const_expr(self.window_prefetch):
            # Windows 0 (steps 0..31) and 1 (steps 32..63): lane l holds step 32w + l, i.e. list slot N-1-32w-l (clamped).
            lane = cute.arch.lane_idx()
            win_idx0 = gIdx[self.clamp_slot(num_n_tiles - 1 - lane)]
            win_idx1 = gIdx[self.clamp_slot(num_n_tiles - 33 - lane)]
            win_bsz0 = self.block_len(win_idx0, gBsz, seqlen, num_compute_tiles)
            win_bsz1 = self.block_len(win_idx1, gBsz, seqlen, num_compute_tiles)
            n_tile_idx_next = cute.arch.shuffle_sync_op(win_idx0, 1)
            varblk_next = cute.arch.shuffle_sync_op(win_bsz0, 1)
        elif cutlass.const_expr(self.row_aware_boxes):
            # idx[1], idx[2] and bsz[1], bsz[2]: all issued before the K[0] wait, so they land under the first TMA latency.
            # The prefetched index is always read from a valid slot (clamped at 0) and never merged with a "no block"
            # marker: the step counter says whether a block exists, and an unused value costs one unused load, while a
            # -1/loaded select made ptxas materialize the index as a uniform register right after its load (a
            # long-scoreboard stall at the top of every step).
            n_tile_idx_next2 = gIndices[self.clamp_slot(num_n_tiles - 3)]
            varblk_next = self.block_len(self.clamp_slot(n_tile_idx_next), gBsz, seqlen, num_compute_tiles)
            varblk_next2 = self.block_len(n_tile_idx_next2, gBsz, seqlen, num_compute_tiles)

        K_barrier[0].wait(index=0, phase=0)
        cute.nvgpu.warpgroup.fence()
        gemm_zero_acc(tiled_mma_qk, tSrQ, tSrK, tSrS)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        if cutlass.const_expr(self.window_prefetch):
            # The shuffled idx[1] is a clamped read when the list has one block: the count guards the issue.
            if warp_idx == 1 and num_n_tiles > 1:
                self.load_kv_block(varblk_next, n_tile_idx_next, tma_atoms_K_part, tma_atom_K, tKgK, tKsK, mK_slice, sK.iterator, K_barrier[0], 0, True)
        elif cutlass.const_expr(self.row_aware_boxes):
            if warp_idx == 1 and n_tile_idx_next >= 0:
                self.load_kv_block(varblk_next, n_tile_idx_next, tma_atoms_K_part, tma_atom_K, tKgK, tKsK, mK_slice, sK.iterator, K_barrier[0], 0, True)
        else:
            if warp_idx == 0 and n_tile_idx_next >= 0:
                K_barrier[0].arrive_and_expect_tx(index=0, tx_count=self.tma_copy_bytes["K"])
                cute.copy(tma_atom_K, tKgK[None, n_tile_idx_next], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

        self.prescale(tSrS, score_scale)
        # Full KV tiles need no mask; the CTA-uniform branch skips per-element selects.
        if varblk < self.tile_size:
            if cutlass.const_expr(self.lean_consumer_loop):
                mask_from_limit(tiled_mma_qk, tSrS, tScS0, varblk - col0)
            else:
                mask(tiled_mma_qk, tSrS, tScS, varblk)
        get_prev_ratio_and_update_max_and_rescale_sum(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)
        inc_softmax_ffma(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)
        cute.autovec_copy(make_acc_into_op(tSrS, tiled_mma_pv.tv_layout_A, self.K_dtype), tOrP)

        # alpha carries softmax[j-1]'s ratio to step j, which rescales O between the QK[j] and PV[j-1] issues so
        # the FMULs overlap QK[j] (O is free: PV[j-2] retired at step j-1). O is 0 at first use, so start at 1.0.
        alpha = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
        alpha.fill(1.0)

        # ---- steady state: step j = 1 .. N-1 walks list slot N-1-j ----
        for n_tile_ind in cutlass.range(num_n_tiles - 2, -1, -1):
            step = num_n_tiles - 1 - n_tile_ind
            n_tile_idx = n_tile_idx_next
            if cutlass.const_expr(self.window_prefetch):
                # bsz[j] was shuffled last step; idx[j+1]/bsz[j+1] (the K[j+1] issue below, V[j+1] and the mask of step j+1)
                # come out of lane (j+1) % 32 of the current window, which rotates when step j+1 opens a new window.
                varblk = varblk_next
                nxt_lane = (step + 1) % 32
                if nxt_lane == 0:
                    win_idx0 = win_idx1
                    win_bsz0 = win_bsz1
                    win_idx1 = gIdx[self.clamp_slot(n_tile_ind - 33 - lane)]
                elif nxt_lane == 1:
                    win_bsz1 = self.block_len(win_idx1, gBsz, seqlen, num_compute_tiles)
                n_tile_idx_next = cute.arch.shuffle_sync_op(win_idx0, nxt_lane)
                varblk_next = cute.arch.shuffle_sync_op(win_bsz0, nxt_lane)
            elif cutlass.const_expr(self.row_aware_boxes):
                # bsz[j] and bsz[j+1] are in registers (loaded two and one steps ago); only idx[j+2] is fetched here.
                varblk = varblk_next
                varblk_next = varblk_next2
                n_tile_idx_next = n_tile_idx_next2
                n_tile_idx_next2 = gIdx[self.clamp_slot(n_tile_ind - 2)]
            else:
                n_tile_idx_next = -1
                if n_tile_ind > 0:
                    n_tile_idx_next = gIdx[n_tile_ind - 1]
                varblk = self.block_len(n_tile_idx, gBSZ, seqlen, num_compute_tiles)

            v_slot = step % 2  # V ring: V[j] fills slot j%2, whose barrier phase is (j//2)%2
            if warp_idx == 0:
                if cutlass.const_expr(self.row_aware_boxes):
                    # Step 1 is slot 1's first fill: load all 64 rows so its stale rows are finite from then on.
                    v_valid = varblk if step > 1 else cutlass.Int32(self.tile_size)
                    v_base = sV.iterator + v_slot * (self.tile_shape_pv[1] * self.tile_shape_pv[2])
                    self.load_kv_block(
                        v_valid, n_tile_idx, tma_atoms_V_part, tma_atom_V, tVgV, tVsV[None, v_slot], mV_slice, v_base, V_barrier[0], v_slot, False
                    )
                else:
                    V_barrier[0].arrive_and_expect_tx(index=v_slot, tx_count=self.tma_copy_bytes["V"])
                    cute.copy(tma_atom_V, tVgV[None, n_tile_idx], tVsV[None, v_slot], tma_bar_ptr=V_barrier[0].get_barrier(v_slot))

            K_barrier[0].wait(index=0, phase=step % 2)  # one K slot: its phase flips every step
            cute.nvgpu.warpgroup.fence()
            gemm_zero_acc(tiled_mma_qk, tSrQ, tSrK, tSrS)
            cute.nvgpu.warpgroup.commit_group()

            if cutlass.const_expr(self.lean_consumer_loop):
                rescale_o_if_needed(tiled_mma_pv, tOrO, alpha)
            else:
                rescale_o_for_next_acc(tiled_mma_pv, tOrO, alpha)
            V_barrier[0].wait(index=1 - v_slot, phase=((step - 1) // 2) % 2)
            cute.nvgpu.warpgroup.fence()  # O was rescaled after the QK[j] fence
            cute.gemm(tiled_mma_pv, tOrO, tOrP, tOrV[None, None, None, 1 - v_slot], tOrO)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(1)  # QK[j] retired, PV[j-1] may still run

            if cutlass.const_expr(self.row_aware_boxes):
                # Warp 1 issues K[j+1] (the barrier's arrival count is 1, any warp may arrive): warp 0 carries the V issue
                # at the top of the step, so the two issues overlap on different warps and the step time stays the slowest
                # warp's softmax chain plus ONE issue.
                if warp_idx == 1 and n_tile_ind > 0:
                    self.load_kv_block(varblk_next, n_tile_idx_next, tma_atoms_K_part, tma_atom_K, tKgK, tKsK, mK_slice, sK.iterator, K_barrier[0], 0, True)
            else:
                if warp_idx == 0 and n_tile_idx_next >= 0:
                    K_barrier[0].arrive_and_expect_tx(index=0, tx_count=self.tma_copy_bytes["K"])
                    cute.copy(tma_atom_K, tKgK[None, n_tile_idx_next], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

            if cutlass.const_expr(self.row_aware_boxes and not self.window_prefetch):
                # bsz[j+2]: idx[j+2] (fetched at the top of this step) has landed by now; the K[j+2] issue of step j+1
                # consumes it, so the load has the softmax below to complete.
                varblk_next2 = self.block_len(n_tile_idx_next2, gBsz, seqlen, num_compute_tiles)

            self.prescale(tSrS, score_scale)
            if varblk < self.tile_size:
                if cutlass.const_expr(self.lean_consumer_loop):
                    mask_from_limit(tiled_mma_qk, tSrS, tScS0, varblk - col0)
                else:
                    mask(tiled_mma_qk, tSrS, tScS, varblk)
            prev_ratio = get_prev_ratio_and_update_max_and_rescale_sum(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)
            inc_softmax_ffma(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)

            # ptxas overlaps softmax with PV[j-1]'s tail
            cute.nvgpu.warpgroup.wait_group(0)  # PV[j-1] retired: O and P registers are free
            cute.autovec_copy(make_acc_into_op(tSrS, tiled_mma_pv.tv_layout_A, self.K_dtype), tOrP)
            cute.autovec_copy(prev_ratio, alpha)

        # ---- drain: PV[N-1] ----
        last = num_n_tiles - 1
        if cutlass.const_expr(self.lean_consumer_loop):
            rescale_o_if_needed(tiled_mma_pv, tOrO, alpha)
        else:
            rescale_o_for_next_acc(tiled_mma_pv, tOrO, alpha)
        V_barrier[0].wait(index=last % 2, phase=(last // 2) % 2)
        cute.nvgpu.warpgroup.fence()
        cute.gemm(tiled_mma_pv, tOrO, tOrP, tOrV[None, None, None, last % 2], tOrO)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

    @cute.jit
    def prescale(self, tSrS: cute.Tensor, score_scale: cutlass.Float32):
        if cutlass.const_expr(self.nonpositive_scale):
            tSrS.store(tSrS.load() * score_scale)

    @cute.jit
    def block_len(self, n_tile_idx, gBSZ: cute.Tensor, seqlen, num_compute_tiles):
        """Valid key count of physical KV block n_tile_idx."""
        varblk = cutlass.Int32(self.tile_size)
        if cutlass.const_expr(self.has_block_sizes):
            varblk = gBSZ[n_tile_idx]
        else:
            if n_tile_idx == num_compute_tiles - 1:
                varblk = seqlen - n_tile_idx * self.tile_size
        return varblk

    @cute.jit
    def clamp_slot(self, slot):
        """A list slot (or block index) prefetched past the start of the list reads entry 0 instead; its value is never used."""
        return slot if slot >= 0 else cutlass.Int32(0)

    @cute.jit
    def load_kv_block(
        self,
        valid: cutlass.Int32,
        n_tile_idx: cutlass.Int32,
        atoms_part,
        atom_full: cute.CopyAtom,
        tg_full: cute.Tensor,
        ts_full: cute.Tensor,
        g2d: cute.Tensor,
        s_base: cute.Pointer,
        barrier,
        index,
        k_major: cutlass.Constexpr[bool],
    ):
        """Issue the K (k_major) or V TMA load of KV block n_tile_idx with 16 * ceil(valid / 16) rows (one warp calls this).

        A count outside 1..48 takes the unchanged full-tile atom and its pre-partitioned tensors; otherwise atoms_part
        (16/32/48 rows) load the same tile layout cut at `rows` rows: one box per swizzle plane (cols = 128 B of elements),
        plane p of K holding the columns [p * cols, (p + 1) * cols) and plane p of V the value dims [p * cols, (p + 1) * cols)
        at p * 64 * cols elements of the tile (the tile_to_shape order of the smem layouts); expect_tx is the loaded bytes.
        Rows >= 16 * ceil(valid / 16) of the buffer keep what an earlier load of this CTA left there (finite gmem data:
        the first load into every buffer is a full one). This is exact: a column of S depends only on its own K row and
        every column >= valid is set to -inf by the mask, so P is exactly 0 there, and 0 x finite V rows adds exactly 0
        to O; the row max and row sum see the same -inf / 0 as with a full load.
        """
        rows16 = cute.arch.make_warp_uniform((valid + 15) >> 4)
        dim = self.tile_shape_qk[2] if cutlass.const_expr(k_major) else self.tile_shape_pv[1]
        row_bytes = dim * ((self.K_dtype if cutlass.const_expr(k_major) else self.V_dtype).width // 8)
        if rows16 >= 4 or rows16 <= 0:
            barrier.arrive_and_expect_tx(index=index, tx_count=self.tile_size * row_bytes)
            cute.copy(atom_full, tg_full[None, n_tile_idx], ts_full, tma_bar_ptr=barrier.get_barrier(index))
        else:
            barrier.arrive_and_expect_tx(index=index, tx_count=rows16 * (16 * row_bytes))
            if rows16 == 3:
                self.issue_rows(atoms_part[2], 48, n_tile_idx, g2d, s_base, barrier.get_barrier(index), k_major)
            elif rows16 == 2:
                self.issue_rows(atoms_part[1], 32, n_tile_idx, g2d, s_base, barrier.get_barrier(index), k_major)
            else:
                self.issue_rows(atoms_part[0], 16, n_tile_idx, g2d, s_base, barrier.get_barrier(index), k_major)

    @cute.jit
    def issue_rows(self, atom, rows: cutlass.Constexpr[int], n_tile_idx, g2d: cute.Tensor, s_base: cute.Pointer, bar_ptr, k_major: cutlass.Constexpr[bool]):
        """One copy (one box per swizzle plane) of the first `rows` rows of KV block n_tile_idx into the K (k_major) or V tile at s_base."""
        if cutlass.const_expr(k_major):
            dest = cute.make_tensor(s_base, self.kv_rows_layout_k(rows))
            tile = cute.local_tile(cute.domain_offset((n_tile_idx * self.tile_size, 0), g2d), (rows, self.tile_shape_qk[2]), (0, 0))
        else:
            dest = cute.make_tensor(s_base, self.kv_rows_layout_v(rows))
            tile = cute.local_tile(cute.domain_offset((0, n_tile_idx * self.tile_size), g2d), (self.tile_shape_pv[1], rows), (0, 0))
        dst, src = cute.nvgpu.cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(dest, 0, 2), cute.group_modes(tile, 0, 2))
        cute.copy(atom, src, dst, tma_bar_ptr=bar_ptr)

    def kv_rows_layout_k(self, rows: int):
        """The K tile layout (tile_to_shape of the swizzle atom: rows at stride cols, planes at stride 64 * cols) cut at `rows` rows."""
        cols, planes = self.kv_plane_cols_k, self.tile_shape_qk[2] // self.kv_plane_cols_k
        return cute.make_layout((rows, (cols, planes)), stride=(cols, (1, self.tile_size * cols)))

    def kv_rows_layout_v(self, rows: int):
        """The V slot layout (value dims at stride 1 within a plane, planes at stride 64 * cols, rows at stride cols) cut at `rows` rows."""
        cols, planes = self.kv_plane_cols_v, self.tile_shape_pv[1] // self.kv_plane_cols_v
        return cute.make_layout(((cols, planes), rows), stride=((1, self.tile_size * cols), cols))

    @cute.kernel
    def kernel_split(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_varblk: cute.Tensor,
        blocksparse_split_offsets: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
    ):
        score_scale = scale_softmax_log2e
        if cutlass.const_expr(self.nonpositive_scale):
            scale_softmax_log2e = cutlass.Float32(1.0)
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        work_desc = self.get_split_work_desc(mQ.shape[2])
        seqlen = mK.shape[0]
        num_compute_tiles = cute.ceil_div(seqlen, self.tile_size)

        shared_storage = SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0 and lane_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)

        op = pipeline.PipelineOp.TmaLoad
        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Initialize single stage barrier for Q/K/V
        Q_barrier = (pipeline.MbarrierArray(shared_storage.Q_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        K_barrier = (pipeline.MbarrierArray(shared_storage.K_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        V_barrier = (pipeline.MbarrierArray(shared_storage.V_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)

        # partition tensors
        sQ = shared_storage.Q_smem.get_tensor(Q_smem_layout.outer, swizzle=Q_smem_layout.inner)
        sK = shared_storage.K_smem.get_tensor(K_smem_layout.outer, swizzle=K_smem_layout.inner)
        sV = shared_storage.V_smem.get_tensor(V_smem_layout.outer, swizzle=V_smem_layout.inner)

        out_head_idx = cutlass.Int64(work_desc.qo_head_idx) + cutlass.Int64(work_desc.split_idx) * cutlass.Int64(mQ.shape[2])
        out_batch_idx = cutlass.Int64(work_desc.batch_idx)
        mO_slice = mO[None, None, out_head_idx, out_batch_idx]
        mLSE_slice = mLSE[None, out_head_idx, out_batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mK_slice = mK[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        mV_slice = mV[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        gO = cute.local_tile(mO_slice, (self.tile_shape_pv[0], self.tile_shape_pv[1]), coord=(work_desc.qo_tile_idx, 0))
        gQ = cute.local_tile(mQ_slice, (self.tile_shape_qk[0], self.tile_shape_qk[2]), coord=(work_desc.qo_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (self.tile_shape_qk[1], self.tile_shape_qk[2]), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (self.tile_shape_pv[1], self.tile_shape_pv[2]), coord=(0, None))

        gIndices = blocksparse_indices_q2k[None, work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        split_start = blocksparse_split_offsets[
            work_desc.split_idx,
            work_desc.qo_tile_idx,
            work_desc.qo_head_idx,
            work_desc.batch_idx,
        ]
        split_end = blocksparse_split_offsets[
            work_desc.split_idx + 1,
            work_desc.qo_tile_idx,
            work_desc.qo_head_idx,
            work_desc.batch_idx,
        ]
        num_n_tiles = split_end - split_start
        gIndices = cute.domain_offset((split_start,), gIndices)
        if cutlass.const_expr(self.has_block_sizes):
            gBSZ = blocksparse_varblk[None, work_desc.qo_head_idx, work_desc.batch_idx]

        cta_coord_layout = (0, cute.make_layout(1))  # CTA coord layout for TMA multicasting, effectively no multicast
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, *cta_coord_layout, cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(tma_atom_K, *cta_coord_layout, cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(tma_atom_V, *cta_coord_layout, cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSrQ = tiled_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((self.tile_shape_qk[0], self.tile_shape_qk[1])), self.acc_dtype)
        tScS = thr_mma_qk.partition_C(cS)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOrV = tiled_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV))
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((self.tile_shape_pv[0], self.tile_shape_pv[1])), self.acc_dtype)

        max_m_layout = cute.make_layout(cute.size(layout_acc_mn(tiled_mma_pv, tOrO.layout), mode=[0]))
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)

        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        # Split offsets can produce empty work for tiles with a short runtime
        # block count. Keep the branch CTA-uniform and avoid reading gIndices[-1].
        if num_n_tiles > 0:
            n_tile_ind_ = num_n_tiles - 1
            n_tile_idx_ = gIndices[n_tile_ind_]

            if warp_idx == 0:
                Q_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.Q_dtype, Q_smem_layout))
                cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=Q_barrier[0].get_barrier(0))

                K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                cute.copy(tma_atom_K, tKgK[None, n_tile_idx_], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

            cute.arch.sync_threads()

            Q_barrier[0].wait(index=0, phase=0)
            n_tile_idx_next = n_tile_idx_
            for n_tile_ind in cutlass.range(num_n_tiles - 1, -1, -1):
                n_tile_idx = n_tile_idx_next
                n_tile_idx_next = -1
                if n_tile_ind > 0:
                    n_tile_idx_next = gIndices[n_tile_ind - 1]
                if cutlass.const_expr(self.has_block_sizes):
                    varblk = gBSZ[n_tile_idx]
                else:
                    varblk = cutlass.Int32(self.tile_size)
                    if n_tile_idx == num_compute_tiles - 1:
                        varblk = seqlen - n_tile_idx * self.tile_size

                # load this V block
                if warp_idx == 0:
                    V_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.V_dtype, V_smem_layout))
                    cute.copy(tma_atom_V, tVgV[None, n_tile_idx], tVsV, tma_bar_ptr=V_barrier[0].get_barrier(0))

                # compute Q@K
                K_barrier[0].wait(index=0, phase=(num_n_tiles - n_tile_ind - 1) % 2)
                cute.nvgpu.warpgroup.fence()  # implicit sync WG
                gemm_zero_acc(tiled_mma_qk, tSrQ, tSrK, tSrS)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                # load next K block
                if warp_idx == 0 and n_tile_idx_next >= 0:
                    K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                    cute.copy(tma_atom_K, tKgK[None, n_tile_idx_next], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

                self.prescale(tSrS, score_scale)
                if varblk < self.tile_size:
                    mask(tiled_mma_qk, tSrS, tScS, varblk)
                prev_ratio = get_prev_ratio_and_update_max_and_rescale_sum(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)
                inc_softmax(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)

                # compute P@V
                rescale_o_for_next_acc(tiled_mma_pv, tOrO, prev_ratio)
                tOrP = make_acc_into_op(tSrS, tiled_mma_pv.tv_layout_A, self.K_dtype)
                V_barrier[0].wait(index=0, phase=(num_n_tiles - n_tile_ind - 1) % 2)
                cute.nvgpu.warpgroup.fence()  # implicit sync WG
                tiled_mma_pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma_pv, tOrO, tOrP, tOrV, tOrO)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

        final_ratio, lse = get_final_ratio_and_lse_empty_safe(max_m, sum_m, scale_softmax_log2e)
        rescale_o_for_next_acc(tiled_mma_pv, tOrO, final_ratio)
        tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(tiled_mma_qk, tScS.layout))
        for m in cutlass.range_constexpr(cute.size(lse)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            if row_idx < mQ.shape[0]:
                mLSE_slice[row_idx] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        tiled_copy_o_r2g = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.O_dtype, num_bits_per_copy=32),
            tiled_mma_pv,
        )
        tOrO_cv = tiled_copy_o_r2g.retile(tOrO_cvt)
        tOgO = tiled_copy_o_r2g.get_slice(tidx).partition_D(gO)
        cute.autovec_copy(tOrO_cv, tOgO)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mK: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mV: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mO: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mLSE: cute.Tensor,  # (seqlen, nheads, batch)
        blocksparse_indices_q2k: cute.Tensor,  # (k, q, nheads, batch)
        blocksparse_num_blocks_q2k: cute.Tensor,  # (q, nheads, batch)
        blocksparse_varblk: cute.Tensor,
        blocksparse_split_offsets: cute.Tensor,  # (num_splits + 1, q, nheads, batch)
        blocksparse_tile_order: cute.Tensor,  # (q, nheads, batch): Q tile of CTA (x, h, b); read only when has_tile_order
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Restore compile-time head dimensions while keeping sequence, head,
        # batch shapes, and their strides dynamic.
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[0], self.qk_dim, mQ.shape[2], mQ.shape[3]),
                stride=mQ.stride,
            ),
        )
        mK = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[0], self.qk_dim, mK.shape[2], mK.shape[3]),
                stride=mK.stride,
            ),
        )
        mV = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (self.value_dim, mV.shape[1], mV.shape[2], mV.shape[3]),
                stride=mV.stride,
            ),
        )
        mO = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (mO.shape[0], self.value_dim, mO.shape[2], mO.shape[3]),
                stride=mO.stride,
            ),
        )
        self.check_dim([mQ, mK, mO], 1)
        self.check_dim(mV, 0)

        Q_layout = LayoutEnum.from_tensor(mQ)
        K_layout = LayoutEnum.from_tensor(mK)
        V_layout = LayoutEnum.from_tensor(mV)
        O_layout = LayoutEnum.from_tensor(mO)

        self.Q_dtype = mQ.element_type
        self.K_dtype = mK.element_type
        self.V_dtype = mV.element_type
        self.O_dtype = mO.element_type

        # major mode is K for Query and Key, MN for Value
        Q_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(Q_layout, self.Q_dtype, self.tile_shape_qk[2]), self.Q_dtype
        )
        K_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(K_layout, self.K_dtype, self.tile_shape_qk[2]), self.K_dtype
        )
        V_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(V_layout, self.V_dtype, self.tile_shape_pv[1]), self.V_dtype
        )

        self.Q_smem_layout = cute.tile_to_shape(Q_smem_layout_atom, (self.tile_shape_qk[0], self.tile_shape_qk[2]), order=(0, 1))
        self.K_smem_layout = cute.tile_to_shape(K_smem_layout_atom, (self.tile_shape_qk[1], self.tile_shape_qk[2]), order=(0, 1))
        if cutlass.const_expr(self.v_stages == 1):
            self.V_smem_layout = cute.tile_to_shape(V_smem_layout_atom, (self.tile_shape_pv[1], self.tile_shape_pv[2]), order=(1, 0))
            V_smem_layout_one = self.V_smem_layout
        else:
            self.V_smem_layout = cute.tile_to_shape(V_smem_layout_atom, (self.tile_shape_pv[1], self.tile_shape_pv[2], self.v_stages), order=(1, 0, 2))
            V_smem_layout_one = cute.slice_(self.V_smem_layout, (None, None, 0))
        # expect_tx bytes of one Q/K tile and one V ring slot (non-split kernel)
        self.tma_copy_bytes = {
            "Q": cute.size_in_bytes(self.Q_dtype, self.Q_smem_layout),
            "K": cute.size_in_bytes(self.K_dtype, self.K_smem_layout),
            "V": cute.size_in_bytes(self.V_dtype, V_smem_layout_one),
        }

        self.O_smem_layout = self.Q_smem_layout  # dummy for non TMA O
        if cutlass.const_expr(self.num_splits == 1):
            O_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                hopper_helpers.get_smem_layout_atom(O_layout, self.O_dtype, self.tile_shape_pv[1]), self.O_dtype
            )
            self.O_smem_layout = cute.tile_to_shape(O_smem_layout_atom, (self.tile_shape_pv[0], self.tile_shape_pv[1]), order=(0, 1))

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            K_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            V_barrier: cute.struct.MemRange[cutlass.Int64, self.v_stages]

            Q_smem: cute.struct.Align[cute.struct.MemRange[self.Q_dtype, cute.cosize(self.Q_smem_layout)], 128]
            K_smem: cute.struct.Align[cute.struct.MemRange[self.K_dtype, cute.cosize(self.K_smem_layout)], 128]
            V_smem: cute.struct.Align[cute.struct.MemRange[self.V_dtype, cute.cosize(self.V_smem_layout)], 128]

        self.shared_storage_t = SharedStorage

        atom_layout_mnk = (1, 1, 1)
        tiled_mma_qk = hopper_helpers.make_trivial_tiled_mma(
            self.Q_dtype,
            self.K_dtype,
            Q_layout.sm90_mma_major_mode(),
            K_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            atom_layout_mnk,
            tiler_mn=self.tile_shape_qk[:2],
        )
        tiled_mma_pv = hopper_helpers.make_trivial_tiled_mma(
            self.K_dtype,
            self.V_dtype,
            K_layout.sm90_mma_major_mode(),
            V_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            atom_layout_mnk,
            tiler_mn=self.tile_shape_pv[:2],
            a_source=cute.nvgpu.warpgroup.OperandSource.RMEM,
        )

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mQ,
            self.Q_smem_layout,
            (self.tile_shape_qk[0], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mK,
            self.K_smem_layout,
            (self.tile_shape_qk[1], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_V, tma_tensor_V = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mV,
            V_smem_layout_one,
            (self.tile_shape_pv[1], self.tile_shape_pv[2]),
            num_multicast=1,
        )
        # Row-aware boxes (P4): atoms that load the first 16/32/48 rows of a KV block into the same tile layouts (one box
        # per swizzle plane, as the atoms above and as the KPage forward's 16-row pages); the 64-row load keeps the atoms above.
        tma_atoms_K_part = (None, None, None)
        tma_atoms_V_part = (None, None, None)
        if cutlass.const_expr(self.row_aware_boxes):
            self.kv_plane_cols_k = swizzle_plane_cols(hopper_helpers.get_smem_layout_atom(K_layout, self.K_dtype, self.tile_shape_qk[2]), self.K_dtype)
            self.kv_plane_cols_v = swizzle_plane_cols(hopper_helpers.get_smem_layout_atom(V_layout, self.V_dtype, self.tile_shape_pv[1]), self.V_dtype)
            atoms_k, atoms_v = [], []
            for rows in (16, 32, 48):
                k_box = cute.make_composed_layout(self.K_smem_layout.inner, 0, self.kv_rows_layout_k(rows))
                atoms_k.append(cute.nvgpu.cpasync.make_tiled_tma_atom(tma_copy_op, mK, k_box, (rows, self.tile_shape_qk[2]), num_multicast=1)[0])
                v_box = cute.make_composed_layout(self.V_smem_layout.inner, 0, self.kv_rows_layout_v(rows))
                atoms_v.append(cute.nvgpu.cpasync.make_tiled_tma_atom(tma_copy_op, mV, v_box, (self.tile_shape_pv[1], rows), num_multicast=1)[0])
            tma_atoms_K_part = tuple(atoms_k)
            tma_atoms_V_part = tuple(atoms_v)

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_copy_op,
                mO,
                self.O_smem_layout,
                (self.tile_shape_pv[0], self.tile_shape_pv[1]),
                num_multicast=1,
            )
            if self.use_tma_o
            else (tma_atom_Q, None)
        )  # use tma_atom_Q as dummy

        log2_e = 1.44269504088896340736

        block_config = (128, 1, 1)

        if cutlass.const_expr(self.num_splits == 1):
            grid_config = self.get_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
            self.kernel(
                tma_tensor_Q,
                tma_tensor_K,
                tma_tensor_V,
                tma_tensor_O if self.use_tma_o else mO,
                mLSE,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                *tma_atoms_K_part,
                *tma_atoms_V_part,
                blocksparse_indices_q2k,
                blocksparse_num_blocks_q2k,
                blocksparse_varblk,
                blocksparse_tile_order if self.has_tile_order else None,
                tiled_mma_qk,
                tiled_mma_pv,
                self.Q_smem_layout,
                self.K_smem_layout,
                self.V_smem_layout,
                self.O_smem_layout,
                softmax_scale * log2_e,
            ).launch(
                grid=grid_config,
                block=block_config,
                smem=self.shared_storage_t.size_in_bytes(),
                stream=stream,
                min_blocks_per_mp=self.min_blocks_per_mp,
            )
        else:
            grid_config = self.get_split_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
            self.kernel_split(
                tma_tensor_Q,
                tma_tensor_K,
                tma_tensor_V,
                mO,
                mLSE,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                blocksparse_indices_q2k,
                blocksparse_varblk,
                blocksparse_split_offsets,
                tiled_mma_qk,
                tiled_mma_pv,
                self.Q_smem_layout,
                self.K_smem_layout,
                self.V_smem_layout,
                softmax_scale * log2_e,
            ).launch(
                grid=grid_config,
                block=block_config,
                smem=self.shared_storage_t.size_in_bytes(),
                stream=stream,
                min_blocks_per_mp=4,
            )


# =============================================================================
# Local CuTe helpers
# =============================================================================


def swizzle_plane_cols(atom_kind, dtype) -> int:
    """Elements per swizzle plane (128/64/32 B) of a wgmma smem layout atom: the inner extent of its TMA box."""
    kind = cute.nvgpu.warpgroup.SmemLayoutAtomKind
    bits = {kind.K_SW128: 1024, kind.MN_SW128: 1024, kind.K_SW64: 512, kind.MN_SW64: 512, kind.K_SW32: 256, kind.MN_SW32: 256}
    assert atom_kind in bits, f"row-aware boxes need a swizzled smem layout atom, got {atom_kind}"
    return bits[atom_kind] // dtype.width


def convert_c_layout_to_a_layout(c, a):
    return cute.make_layout(
        (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
        stride=(
            c.stride[0],
            c.stride[1],
            (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
        ),
    )


@cute.jit
def make_acc_into_op(acc, operand_layout_tv, Element, operand=None):
    # An overlapped PV keeps its operand live until wait_group(0). Callers can
    # then reuse that allocation, including the complete FP8 permutation.
    if cutlass.const_expr(operand is None):
        operand = cute.make_rmem_tensor_like(
            convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
    operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
    acc_vec = acc.load()
    operand_as_acc.store(acc_vec.to(Element))

    if cutlass.const_expr(Element.width == 8 and True):
        tidx, _, _ = cute.arch.thread_idx()
        tid = tidx % 4
        values_u32 = cute.recast_tensor(operand, cutlass.Uint32)
        for n in cutlass.range_constexpr(cute.size(values_u32, mode=[1])):
            for k in cutlass.range_constexpr(cute.size(values_u32, mode=[2])):
                for ii in cutlass.range_constexpr(0, 8, 4):
                    values_tmp_0 = values_u32[ii // 2 + 0, n, k]
                    values_tmp_1 = values_u32[ii // 2 + 1, n, k]

                    v_to_send = 1
                    if tid == 1 or tid == 2:
                        v_to_send = 0

                    v_to_recv = v_to_send
                    t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF

                    values_tmp_a = values_tmp_1
                    if v_to_send == 0:
                        values_tmp_a = values_tmp_0

                    values_tmp_a = cute.arch.shuffle_sync_op(values_tmp_a, t_to_recv_from, 0xFFFFFFFF, 7199)

                    v_to_send = 1 - v_to_send
                    v_to_recv = 1 - v_to_recv
                    t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF

                    values_tmp_b = values_tmp_1
                    if v_to_send == 0:
                        values_tmp_b = values_tmp_0

                    values_tmp_b = cute.arch.shuffle_sync_op(values_tmp_b, t_to_recv_from, 0xFFFFFFFF, 7199)

                    order = 0x5410
                    if v_to_send == 0:
                        order = 0x1054

                    values_u32[ii // 2 + 0, n, k] = cute.arch.prmt(
                        values_tmp_a,
                        values_tmp_b,
                        order,
                    )

                    order = 0x7632
                    if v_to_send == 0:
                        order = 0x3276
                    values_u32[ii // 2 + 1, n, k] = cute.arch.prmt(values_tmp_a, values_tmp_b, order)
    return operand


@cute.jit
def gemm_zero_acc(tiled_mma, A, B, C):
    rA = cute.rank(A)
    rB = cute.rank(B)
    rC = cute.rank(C)
    if cutlass.const_expr(rA == 2 and rB == 2 and rC == 1):
        for k_block_idx in range(cute.size(A, mode=[1]), unroll_full=True):
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
            cute.gemm(
                tiled_mma,
                C,
                A[None, k_block_idx],
                B[None, k_block_idx],
                C,
            )
    elif cutlass.const_expr(rA == 3 and rB == 3 and rC == 3):
        for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
            cute.gemm(
                tiled_mma,
                C,
                A[None, None, k_block_idx],
                B[None, None, k_block_idx],
                C,
            )
    else:
        assert 0, "unreachable"


def _tree(vals, op):
    """Trace-time pairwise reduction: depth log2(n) instead of an n-deep dependent chain."""
    if len(vals) == 1:
        return vals[0]
    half = len(vals) // 2
    return op(_tree(vals[:half], op), _tree(vals[half:], op))


@cute.jit
def reduce_max(
    tSrS_mn: cute.Tensor,
    max_m: cute.Tensor,
):
    # Per-row pairwise tree; max does not depend on evaluation order, so this is bitwise equal to a serial chain.
    for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
        row = [tSrS_mn[m, n] for n in range(cute.size(tSrS_mn, mode=[1]))]
        max_m[m] = cute.arch.fmax(max_m[m], _tree(row, cute.arch.fmax))

    for m in cutlass.range_constexpr(cute.size(max_m, mode=[0])):
        max_m[m] = cute.arch.warp_reduction_max(max_m[m], threads_in_group=4)


@cute.jit
def thread_reduce_sum(
    tSrS_mn: cute.Tensor,
    sum_m: cute.Tensor,
):
    for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            sum_m[m] += tSrS_mn[m, n]


@cute.jit
def warp_reduce_sum(sum_m: cute.Tensor):
    for m in cutlass.range_constexpr(cute.size(sum_m, mode=[0])):
        sum_m[m] = cute.arch.warp_reduction_sum(sum_m[m], threads_in_group=4)


@cute.jit
def mask(
    qk_tiled_mma: cute.TiledMma,
    tSrS: cute.ThrMma,
    tScS: cute.Tensor,
    varblk: cutlass.Int32,
):
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(qk_tiled_mma, tSrS.layout))
    tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(qk_tiled_mma, tScS.layout))

    for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
        should_mask = tScS_mn[0, n][1] >= varblk
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            tSrS_mn[m, n] = -cutlass.Float32.inf if should_mask else tSrS_mn[m, n]


@cute.jit
def mask_from_limit(
    qk_tiled_mma: cute.TiledMma,
    tSrS: cute.ThrMma,
    tScS0: cute.Tensor,
    lim: cutlass.Int32,
):
    """mask() with lim = varblk - (this thread's first column): the thread's column n is masked iff its offset from that
    first column, which is thread 0's column of element (0, n) and a trace-time constant (tScS0 = thread 0's coordinate
    partition), is >= lim. One compare against an immediate per column instead of a rebuilt coordinate per column."""
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(qk_tiled_mma, tSrS.layout))
    tScS0_mn = cute.make_tensor(tScS0.iterator, layout_acc_mn(qk_tiled_mma, tScS0.layout))

    for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
        should_mask = lim <= tScS0_mn[0, n][1]
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            tSrS_mn[m, n] = -cutlass.Float32.inf if should_mask else tSrS_mn[m, n]


@cute.jit
def get_prev_ratio_and_update_max_and_rescale_sum(
    qk_tiled_mma: cute.TiledMma,
    tSrS: cute.ThrMma,
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
) -> cute.Tensor:
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(qk_tiled_mma, tSrS.layout))

    prev_ratio_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
    prev_max_m = cute.make_rmem_tensor_like(max_m, max_m._dtype)
    cute.autovec_copy(max_m, prev_max_m)
    reduce_max(tSrS_mn, max_m)
    for m in cutlass.range_constexpr(cute.size(max_m, mode=[0])):
        prev_max = prev_max_m[m]
        new_max = max_m[m]
        if new_max == -cutlass.Float32.inf:
            new_max = 0.0

        prev_ratio = cute.math.exp2((prev_max - new_max) * softmax_scale_log2e, fastmath=True)
        prev_ratio_m[m] = prev_ratio
        sum_m[m] *= prev_ratio

    return prev_ratio_m


@cute.jit
def inc_softmax(
    tiled_mma_qk: cute.TiledMma,
    tSrS: cute.ThrMma,
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
):
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(tiled_mma_qk, tSrS.layout))

    for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
        new_max = max_m[m]
        if new_max == -cutlass.Float32.inf:
            new_max = 0.0

        for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
            tSrS_mn[m, n] = cute.math.exp2((tSrS_mn[m, n] - new_max) * softmax_scale_log2e, fastmath=True)

    thread_reduce_sum(tSrS_mn, sum_m)


@cute.jit
def rescale_o_for_next_acc(
    pv_tiled_mma: cute.TiledMma,
    tOrO: cute.ThrMma,
    prev_ratio_m: cute.Tensor,
):
    tOrO_mn = cute.make_tensor(tOrO.iterator, layout_acc_mn(pv_tiled_mma, tOrO.layout))
    for m in cutlass.range_constexpr(cute.size(tOrO_mn, mode=[0])):
        for n in cutlass.range_constexpr(cute.size(tOrO_mn, mode=[1])):
            tOrO_mn[m, n] *= prev_ratio_m[m]


@cute.jit
def rescale_o_if_needed(
    pv_tiled_mma: cute.TiledMma,
    tOrO: cute.ThrMma,
    prev_ratio_m: cute.Tensor,
):
    """rescale_o_for_next_acc behind one warp vote: a row whose max did not change has ratio exactly 1.0, and x * 1.0 == x,
    so when every row of the warp has ratio 1.0 the 64 FMULs are skipped bit-exactly (the vote keeps the branch uniform)."""
    needs_rescale = cutlass.Boolean(False)
    for m in cutlass.range_constexpr(cute.size(prev_ratio_m)):
        needs_rescale = needs_rescale | (prev_ratio_m[m] != 1.0)
    if cute.arch.vote_any_sync(needs_rescale):
        rescale_o_for_next_acc(pv_tiled_mma, tOrO, prev_ratio_m)


@cute.jit
def get_final_ratio_and_lse(
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
) -> cute.Tensor:
    warp_reduce_sum(sum_m)
    final_ratio = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)
    lse = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)

    for m in cutlass.range_constexpr(cute.size(sum_m, mode=[0])):
        final_sum = sum_m[m]
        final_ratio[m] = cute.arch.rcp_approx(final_sum)

        ln2 = 0.693147180559945309417
        lse[m] = -cutlass.Float32.inf if final_sum == 0.0 else (max_m[m] * softmax_scale_log2e * ln2 + _math.log(final_sum))

    return final_ratio, lse


@cute.jit
def get_final_ratio_and_lse_empty_safe(
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
) -> cute.Tensor:
    warp_reduce_sum(sum_m)
    final_ratio = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)
    lse = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)

    for m in cutlass.range_constexpr(cute.size(sum_m, mode=[0])):
        final_sum = sum_m[m]
        final_ratio[m] = cutlass.Float32(0.0) if final_sum == 0.0 else cute.arch.rcp_approx(final_sum)

        ln2 = 0.693147180559945309417
        lse[m] = -cutlass.Float32.inf if final_sum == 0.0 else (max_m[m] * softmax_scale_log2e * ln2 + _math.log(final_sum))

    return final_ratio, lse


@cute.jit
def layout_acc_mn(tiled_mma, acc):
    separated = layout_separate(tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1])

    V_M = separated[0]
    V_N = separated[1]
    if cutlass.const_expr(cute.rank(V_M) == 1):
        V_M1 = cute.append(V_M, acc[1])
    else:
        V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])

    if cutlass.const_expr(cute.rank(V_N) == 1):
        V_N1 = cute.append(V_N, acc[2])
    else:
        V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
    if cutlass.const_expr(cute.rank(V_M1) == 1):
        return cute.append(V_M1, V_N1)
    return cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)


def layout_separate(thr, src, ref):
    lt = cute.make_layout(())
    ge = cute.make_layout(())

    for k, v in enumerate(ref):
        if cutlass.const_expr(v < thr):
            lt = cute.append(lt, src[k])
        else:
            ge = cute.append(ge, src[k])

    if cutlass.const_expr(cute.rank(lt) == 1):
        return cute.append(lt, ge)
    return cute.append(cute.append(cute.make_layout(()), lt), ge)


@cute.jit
def inc_softmax_ffma(
    tiled_mma_qk: cute.TiledMma,
    tSrS: cute.Tensor,
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
):
    """inc_softmax with P = exp2(fma(s, c, -max * c)): one FFMA per element instead of FADD + FMUL
    (the compiler does not contract (s - max) * c), and a pairwise-tree row sum."""
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(tiled_mma_qk, tSrS.layout))

    # At s == max the FFMA returns the rounding error of fl(max * c), not 0. The 16-bit P(max)
    # still rounds to 1 while |max * c| < FFMA_MAX_SCALED_ABS; past that, the warp centers its
    # rows first (s - max is exact at the max), and the vote keeps the branch warp-uniform.
    row_max = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
    neg_scaled_max = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
    needs_centering = cutlass.Boolean(False)
    for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
        new_max = max_m[m]
        if new_max == -cutlass.Float32.inf:
            new_max = 0.0
        row_max[m] = new_max
        neg_scaled_max[m] = -(new_max * softmax_scale_log2e)
        needs_centering = needs_centering | (cute.math.absf(neg_scaled_max[m]) >= FFMA_MAX_SCALED_ABS)

    if cute.arch.vote_any_sync(needs_centering):
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            neg_scaled_max[m] = 0.0
            for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
                tSrS_mn[m, n] = tSrS_mn[m, n] - row_max[m]

    for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
        for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
            tSrS_mn[m, n] = cute.math.exp2(cute.math.fma(tSrS_mn[m, n], softmax_scale_log2e, neg_scaled_max[m]), fastmath=True)

    for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
        row = [tSrS_mn[m, n] for n in range(cute.size(tSrS_mn, mode=[1]))]
        sum_m[m] += _tree(row, lambda a, b: a + b)


@cute.jit
def hoist_list_base(gIndices: cute.Tensor) -> cute.Tensor:
    """gIndices re-based on a warp-uniform copy of its start address (same elements, same layout)."""
    addr = gIndices.iterator.toint()
    lo = cute.arch.make_warp_uniform(cutlass.Int32(addr & 0xFFFFFFFF))
    hi = cute.arch.make_warp_uniform(cutlass.Int32(addr >> 32))
    addr_u = (cutlass.Int64(hi) << 32) | (cutlass.Int64(lo) & 0xFFFFFFFF)
    ptr = cute.make_ptr(gIndices.element_type, addr_u, cute.AddressSpace.gmem, assumed_align=4)
    return cute.make_tensor(ptr, gIndices.layout)
