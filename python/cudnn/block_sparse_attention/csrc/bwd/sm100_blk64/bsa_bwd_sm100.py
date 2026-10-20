# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import nvvm
from cutlass.cute.typing import Float32, Int32, Int64, BFloat16, Boolean
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import LoadCacheMode, OperandMajorMode, cpasync, tcgen05
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, TmemAllocator
import cutlass.utils.blackwell_helpers as sm100_utils

from typing import Tuple

import math

SM100_BLK64_BWD_SPARSE_BLOCK_SIZE = 64
SM100_BWD_HEAD_DIM = 128
# 4096-block buckets were measured slower on long fixed-count masks (1.24-1.37x at 217K-695K tokens) and gave no net
# gain on the production corpus, so long queries keep the 1024-block buckets.
SM100_BWD_AUTO_BUCKETED_K2Q_BLOCKS = 1024
SM100_BWD_AUTO_BUCKETED_K2Q_MIN_Q_BLOCKS = 3000
# Interleaving backward tasks across heads (logical_block_coord) helps up to ~86K query tokens and costs from ~109K
# tokens up, on both the production corpus and the long video masks, so it is limited to shorter queries.
SM100_BWD_INTERLEAVE_MAX_SEQLEN_Q = 98304


def sm100_bwd_default_bucketed_k2q_size_blocks(num_q_blocks: int) -> int:
    if num_q_blocks < 2048 or num_q_blocks >= 8192:
        return 1088
    return 1152


def sm100_bwd_auto_bucketed_k2q_size_blocks(num_q_blocks: int) -> int | None:
    if num_q_blocks >= SM100_BWD_AUTO_BUCKETED_K2Q_MIN_Q_BLOCKS:
        return SM100_BWD_AUTO_BUCKETED_K2Q_BLOCKS
    return None


def _cp_async_copy_or_zero_fill(atom: cute.CopyAtom, src: cute.Tensor, dst: cute.Tensor, valid: Boolean):
    # A predicated CopyG2SOp copy lowers to `cp.async ... cp-size, src-size` with src-size = valid ? cp-size : 0,
    # so an invalid element is zero-filled by the cp.async itself. A plain smem store (`.fill(0.0)`) is not covered
    # by PipelineCpAsync.producer_commit (cp.async.mbarrier.arrive), which tracks only this thread's cp.async ops.
    pred = cute.make_rmem_tensor(cute.make_layout(1), Boolean)
    pred[0] = valid
    cute.copy(atom, src, dst, pred=pred)


class _WarpGroupBarrier(pipeline.NamedBarrier):
    """Named barrier for a subset of CTA warps, without the aligned promise."""

    @dsl_user_op
    def arrive_and_wait(self, *, loc=None, ip=None):
        nvvm.barrier_cta_sync(
            Int32(self.barrier_id).ir_value(loc=loc, ip=ip),
            thread_count=Int32(self.num_threads).ir_value(loc=loc, ip=ip),
            aligned=False,
            loc=loc,
            ip=ip,
        )


class BlockSparseAttnBackwardSm100Blk64:
    """Transposed regular backward, with shared setup for the paired KV kernel."""

    def __init__(
        self,
        sparse_block_size: int,
        has_block_sizes: bool = True,
        force_dkv_postprocess: cutlass.Constexpr = True,
        skip_preprocess: bool = False,
        skip_convert: bool = False,
        kv_group_size: int = 1,
    ):
        assert kv_group_size in (1, 2)
        self.interleave_heads = True
        self.local_pairs = False
        self.kv_group_size = kv_group_size
        self.alias_ps = True
        self.dq_a_major_mode = OperandMajorMode.MN
        kv_tile = 64 * kv_group_size
        self.sparse_block_size = sparse_block_size
        self.has_block_sizes = has_block_sizes
        self.dKV_postprocess = force_dkv_postprocess
        self.skip_preprocess = skip_preprocess
        self.skip_convert = skip_convert

        self.QK_mma_tiler = (128, kv_tile, 128)
        self.fake_QK_mma_tiler = (64, kv_tile, 128)
        self.dOP_mma_tiler = (128, kv_tile, 128)
        self.dOV_mma_tiler = (128, kv_tile, 128)
        self.fake_dOV_mma_tiler = (64, kv_tile, 128)
        self.dSK_mma_tiler = (128, 128, kv_tile)
        self.QdS_mma_tiler = (128, kv_tile, 128)

        self.element_dtype = BFloat16
        self.acc_dtype = Float32

        # =================== Sum OdO ================================
        self.sum_OdO_max_threads_per_block = 128
        self.sum_OdO_block_q = 16
        self.sum_OdO_num_threads_d = 8
        self.sum_OdO_num_threads_q = self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
        self.sum_OdO_elem_per_load = 2

        self.reduce_warp_id = (0, 1, 2, 3)
        self.compute_warp_id = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13

        self.num_reduce_warps = 4
        self.num_compute_warps = 8

        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (self.num_reduce_warps + self.num_compute_warps + 4)

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = _WarpGroupBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * (self.num_compute_warps + 1 + self.num_reduce_warps),
        )
        self.compute_sync_barrier = _WarpGroupBarrier(
            barrier_id=3,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.epilogue_sync_barrier = _WarpGroupBarrier(
            barrier_id=4,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.reduce_sync_barrier = _WarpGroupBarrier(
            barrier_id=5,
            num_threads=self.num_reduce_warps * self.threads_per_warp,
        )

        self.num_regs_reduce = 152
        self.num_regs_compute = 128
        self.num_regs_mma = 96
        self.num_regs_empty = 96
        self.num_regs_load = 96

        self.buffer_align_bytes = 1024
        self.score_tiler = (64, 128, 128)
        self.tmem_dK_offset = 0
        self.tmem_dV_offset = 128
        self.tmem_dP_offset = 256
        self.tmem_dQ_offset = 256
        self.tmem_S_offset = 384

    @cute.jit
    def logical_block_coord(self, seqlen_q: Int32):
        x, h, b = cute.arch.block_idx()
        if cutlass.const_expr(self.interleave_heads):
            # Head interleaving helps short sequences and costs long ones, so it is a runtime decision on
            # seqlen_q; every role of a CTA evaluates the same predicate and therefore the same mapping.
            if seqlen_q < SM100_BWD_INTERLEAVE_MAX_SEQLEN_Q:
                nx, nh, _ = cute.arch.grid_dim()
                task = x + h * nx
                group_width = 16 if self.kv_group_size == 2 else 512
                base = task // (group_width * nh) * group_width
                width = min(group_width, nx - base)
                offset = task - base * nh
                x = base + offset % width
                h = offset // width
        return x, h, b

    def _setup_attributes(self):
        self.load_mma_Q_stage = 2
        self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.mma_reduce_dQ_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2
        self.reduce_tma_store_stage = 2

    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        workspace: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        Q, K, D = problem_shape[0], problem_shape[1], problem_shape[2]
        H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])
        D = cute.round_up(D, 8)
        Q = cute.round_up(Q, 8)
        K = cute.round_up(K, 8)

        # Cast problem dims to Int64 before pointer arithmetic. The customer
        # B=1,H=40,Q=131072,D=128 case already has multi-GB workspace offsets.
        Q_i64 = Int64(Q)
        K_i64 = Int64(K)
        D_i64 = Int64(D)
        H_i64 = Int64(H)
        B_i64 = Int64(B)

        # Element offsets used to split the float32 workspace into sub-tensors.
        # Keep this element-addressed: the byte offset to dK_acc exceeds 2^31
        # for customer-scale cases such as B=1,H=40,Q=131072,D=128.
        sum_OdO_elems = cute.assume(B_i64 * H_i64 * Q_i64, divby=4)
        scaled_lse_elems = cute.assume(B_i64 * H_i64 * Q_i64, divby=4)

        sum_OdO_iter = workspace.iterator
        scaled_lse_iter = sum_OdO_iter + sum_OdO_elems
        dQ_acc_iter = scaled_lse_iter + scaled_lse_elems

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)
        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype)

        # Layout strides also use Int64 so indexing promotes Int32 coordinates
        # to Int64 when computing flat element offsets. Without this, the
        # last-element offset in dQ_acc (B*H*Q*D) can exceed 2^31 for large
        # Q (e.g. num_blocks=10860 with default B=4, H=8), silently wrapping
        # to a negative value and causing illegal memory accesses in the
        # convert kernel that iterates over dQ_acc.
        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout((Q, (H, B)), stride=(1, (Q_i64, Q_i64 * H_i64))),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout((Q, (H, B)), stride=(1, (Q_i64, Q_i64 * H_i64))),
        )
        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (Q, D, (H, B)),
                stride=(D, 1, (D_i64 * Q_i64, D_i64 * Q_i64 * H_i64)),
            ),
        )
        if cutlass.const_expr(self.dKV_postprocess):
            dK_acc_iter = dQ_acc_iter + cute.assume(B_i64 * H_i64 * Q_i64 * D_i64, divby=4)
            dV_acc_iter = dK_acc_iter + cute.assume(B_i64 * H_i64 * K_i64 * D_i64, divby=4)
            dK_acc_iter = cute.recast_ptr(dK_acc_iter, dtype=self.acc_dtype)
            dV_acc_iter = cute.recast_ptr(dV_acc_iter, dtype=self.acc_dtype)
            dK_acc = cute.make_tensor(
                dK_acc_iter,
                cute.make_layout(
                    (D, K, (H, B)),
                    stride=(1, D_i64, (D_i64 * K_i64, D_i64 * K_i64 * H_i64)),
                ),
            )
            dV_acc = cute.make_tensor(
                dV_acc_iter,
                cute.make_layout(
                    (D, K, (H, B)),
                    stride=(1, D_i64, (D_i64 * K_i64, D_i64 * K_i64 * H_i64)),
                ),
            )
        else:
            # A single Q group gives every KV block one unique CTA owner.
            # Keep dK/dV in TMEM across that complete reverse row, then write
            # the caller's BF16 tensors directly from the epilogue.
            dK_acc = dK
            dV_acc = dV

        return sum_OdO, scaled_lse, dQ_acc, dK_acc, dV_acc

    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        block_q: int,
    ) -> Tuple[int, int, int]:
        grid = (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )
        return grid

    def _compute_bwd_grid(self, problem_shape, bucketed_k2q_offsets: cute.Tensor):
        H, B = problem_shape[3][0], problem_shape[3][1]
        tasks_per_group = (cute.size(bucketed_k2q_offsets.shape[0]) - 1) // self.kv_group_size
        num_q_groups = cute.size(bucketed_k2q_offsets.shape[1])
        return (tasks_per_group * num_q_groups, H, B)

    @cute.jit
    def __call__(
        self,
        # [s_q, s_k, d, (h, b)]
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        dO: cute.Tensor,
        O: cute.Tensor,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        LSE: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        bucketed_k2q_offsets: cute.Tensor,
        bucketed_k2q_indices: cute.Tensor,
        variable_block_sizes: cute.Tensor,
        workspace: cute.Tensor,
        scale_softmax: Float32,
        stream: cuda.CUstream,
        pair_flags: cute.Tensor | None = None,
        pair_residuals: cute.Tensor | None = None,
    ):
        _, _, d, hb = problem_shape
        h, b = hb
        # (b, h, s, d) -> (s, d, (h, b))
        Q = cute.make_tensor(Q.iterator, cute.group_modes(cute.select(Q.layout, mode=[2, 3, 1, 0]), 2, 4))
        # (b, h, s, d) -> (s, d, (h, b))
        K = cute.make_tensor(K.iterator, cute.group_modes(cute.select(K.layout, mode=[2, 3, 1, 0]), 2, 4))
        # (b, h, s, d) -> (s, d, (h, b))
        V = cute.make_tensor(V.iterator, cute.group_modes(cute.select(V.layout, mode=[2, 3, 1, 0]), 2, 4))
        # O, dO and caller-provided dQ can have different physical layouts from Q.
        O = cute.make_tensor(O.iterator, cute.group_modes(cute.select(O.layout, mode=[2, 3, 1, 0]), 2, 4))

        dQ = cute.make_tensor(dQ.iterator, cute.group_modes(cute.select(dQ.layout, mode=[2, 3, 1, 0]), 2, 4))
        dK = cute.make_tensor(dK.iterator, cute.group_modes(cute.select(dK.layout, mode=[3, 2, 1, 0]), 2, 4))
        # (b, h, s, d) -> (d, s, (h, b))
        dV = cute.make_tensor(dV.iterator, cute.group_modes(cute.select(dV.layout, mode=[3, 2, 1, 0]), 2, 4))
        dO = cute.make_tensor(dO.iterator, cute.group_modes(cute.select(dO.layout, mode=[2, 3, 1, 0]), 2, 4))

        # (b, h, s) -> (s, (h, b))
        LSE = cute.make_tensor(LSE.iterator, cute.group_modes(cute.select(LSE.layout, mode=[2, 1, 0]), 1, 3))

        # (b, h, q_group, task + 1) -> (task + 1, q_group, (h, b))
        bucketed_k2q_offsets = cute.make_tensor(
            bucketed_k2q_offsets.iterator,
            cute.group_modes(cute.select(bucketed_k2q_offsets.layout, mode=[3, 2, 1, 0]), 2, 4),
        )
        # (b, h, edge) -> (edge, (h, b))
        bucketed_k2q_indices = cute.make_tensor(
            bucketed_k2q_indices.iterator,
            cute.group_modes(cute.select(bucketed_k2q_indices.layout, mode=[2, 1, 0]), 1, 3),
        )
        self.Q_major_mode = LayoutEnum.from_tensor(Q).mma_major_mode()
        self.dQ_major_mode = LayoutEnum.from_tensor(dQ).mma_major_mode()
        self.K_major_mode = LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = LayoutEnum.from_tensor(dV).mma_major_mode()
        if cutlass.const_expr(self.Q_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.dQ_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of dq is not supported")
        if cutlass.const_expr(self.K_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != OperandMajorMode.MN):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != OperandMajorMode.MN):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        # Compute S
        QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.QK_mma_tiler[:2],
        )
        fake_QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.fake_QK_mma_tiler[:2],
        )
        # Compute dP
        dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.dOV_mma_tiler[:2],
        )
        fake_dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.fake_dOV_mma_tiler[:2],
        )
        # Compute dV
        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.dOP_mma_tiler[:2],
        )
        # Compute dK
        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.QdS_mma_tiler[:2],
        )
        # Compute dQ
        dSK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            self.dq_a_major_mode,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.dSK_mma_tiler[:2],
        )

        Q_smem_layout_staged = sm100_utils.make_smem_layout_a(QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, self.load_mma_Q_stage)
        fake_Q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            fake_QK_tiled_mma,
            self.fake_QK_mma_tiler,
            self.element_dtype,
            1,
        )
        K_smem_layout_staged = sm100_utils.make_smem_layout_b(QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, 1)
        dO_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dOV_tiled_mma,
            self.dOV_mma_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )
        fake_dO_smem_layout_staged = sm100_utils.make_smem_layout_a(
            fake_dOV_tiled_mma,
            self.fake_dOV_mma_tiler,
            self.element_dtype,
            1,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dOV_tiled_mma,
            self.dOV_mma_tiler,
            self.element_dtype,
            1,
        )
        dS_smem_layout_staged = sm100_utils.make_smem_layout_a(dSK_tiled_mma, self.dSK_mma_tiler, self.element_dtype, self.compute_mma_dS_stage)
        KT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            self.element_dtype,
            1,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QdS_tiled_mma,
            self.QdS_mma_tiler,
            self.element_dtype,
            self.load_mma_Q_stage,
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_b(
            QdS_tiled_mma,
            self.QdS_mma_tiler,
            self.element_dtype,
            self.compute_mma_dS_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dOP_tiled_mma,
            self.dOP_mma_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dOP_tiled_mma,
            self.dOP_mma_tiler,
            self.element_dtype,
            self.compute_mma_P_stage,
        )

        LSE_smem_layout = cute.make_layout((self.QK_mma_tiler[0], self.load_compute_LSE_stage))
        sum_OdO_smem_layout = cute.make_layout((self.QK_mma_tiler[0], self.load_compute_sum_OdO_stage))

        dQ_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                OperandMajorMode.K,
                self.acc_dtype,
                (self.QK_mma_tiler[0], 32),
            ),
            self.acc_dtype,
        )
        dQ_smem_layout_staged = cute.tile_to_shape(dQ_smem_layout_atom, (self.QK_mma_tiler[0], 32, self.reduce_tma_store_stage), order=(1, 0, 2))
        fake_dQ_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                OperandMajorMode.K,
                self.acc_dtype,
                (self.fake_QK_mma_tiler[0], 32),
            ),
            self.acc_dtype,
        )
        fake_dQ_smem_layout_staged = cute.tile_to_shape(
            fake_dQ_smem_layout_atom,
            (self.fake_QK_mma_tiler[0], 32, self.reduce_tma_store_stage),
            order=(1, 0, 2),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp()

        Q_smem_layout = cute.select(fake_Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.fake_QK_mma_tiler,
            fake_QK_tiled_mma,
        )

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            K,
            K_smem_layout,
            self.QK_mma_tiler,
            QK_tiled_mma,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            V,
            V_smem_layout,
            self.dOV_mma_tiler,
            dOV_tiled_mma,
        )

        if cutlass.const_expr(self.local_pairs):
            pair_layout = cute.make_composed_layout(K_smem_layout_staged.inner, 0, cute.make_layout((64, 64), stride=(64, 1)))
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(tma_load_op, K, pair_layout, (64, 64))
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(tma_load_op, V, pair_layout, (64, 64))

        dO_smem_layout = cute.select(fake_dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.fake_dOV_mma_tiler,
            fake_dOV_tiled_mma,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(self.element_dtype, dO_smem_layout)

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_Q_stage * 2]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_dO_stage * 2]
            load_compute_lse_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_LSE_stage * 2]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_sum_OdO_stage * 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            mma_reduce_dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_reduce_dQ_stage * 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_P_stage * 2]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dKdV_stage * 2]
            tmem_holding_buf: cutlass.Int32
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(K_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(V_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(P_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(dO_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, 0 if self.alias_ps else cute.cosize(dS_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(dQ_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                self.buffer_align_bytes,
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        sum_OdO, scaled_LSE, dQ_acc, dK_acc, dV_acc = self.get_workspace_tensor(
            problem_shape,
            workspace,
            dK,
            dV,
        )

        dQ_smem_layout = cute.select(fake_dQ_smem_layout_staged, mode=[0, 1])

        tma_atom_dQ_acc, tma_tensor_dQ_acc = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_reduce_op,
            dQ_acc,
            dQ_smem_layout,
            (self.fake_QK_mma_tiler[0], 32),
        )

        # =============================== Sum OdO ===============================
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))

        sum_OdO_grid = self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q)

        if cutlass.const_expr(not self.skip_preprocess):
            self.sum_OdO(
                O,
                dO,
                sum_OdO,
                LSE,
                scaled_LSE,
                sum_OdO_scale,
                LSE_scale,
                problem_shape,
            ).launch(
                grid=sum_OdO_grid,
                block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
                cluster=[1, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

        bwd_grid = self._compute_bwd_grid(problem_shape, bucketed_k2q_offsets)
        self.bwd(
            QK_tiled_mma,
            fake_QK_tiled_mma,
            dOV_tiled_mma,
            fake_dOV_tiled_mma,
            dOP_tiled_mma,
            QdS_tiled_mma,
            dSK_tiled_mma,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dQ_acc,
            tma_tensor_dQ_acc,
            dK_acc,
            dV_acc,
            scaled_LSE,
            scale_softmax,
            sum_OdO,
            bucketed_k2q_offsets,
            bucketed_k2q_indices,
            problem_shape,
            variable_block_sizes,
            Q_smem_layout_staged,
            K_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dS_smem_layout_staged,
            KT_smem_layout_staged,
            QT_smem_layout_staged,
            dST_smem_layout_staged,
            dOT_smem_layout_staged,
            dQ_smem_layout_staged,
            P_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
            pair_flags,
            pair_residuals,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.block_seq = 8
        self.num_threads_D_convert = 16
        self.num_threads_seq = 128 // self.num_threads_D_convert
        self.convert_elem_per_load = 4

        max_seq_in_qk = max(problem_shape[0], problem_shape[1])
        # Place the seq tile on grid.x (max 2^31-1) rather than grid.z
        # (max 65535), since max_seq_in_qk / block_seq can exceed 65535 for
        # large sequence lengths (e.g. num_blocks=10860 -> 86881 tiles).
        convert_grid_x = (max_seq_in_qk + self.block_seq - 1) // self.block_seq
        convert_grid = [
            convert_grid_x,
            cute.size(problem_shape[3][0]),
            cute.size(problem_shape[3][1]),
        ]
        convert_block = [self.num_threads_D_convert, self.num_threads_seq, 1]

        if cutlass.const_expr(not self.skip_convert):
            self.convert(
                dQ_acc,
                dK_acc,
                dV_acc,
                dQ,
                dK,
                dV,
                bucketed_k2q_offsets,
                problem_shape[0],
                problem_shape[1],
                problem_shape[2],
                scale_softmax,
            ).launch(
                grid=convert_grid,
                block=convert_block,
                cluster=[1, 1, 1],
                smem=0,
                stream=stream,
            )

    @cute.kernel
    def sum_OdO(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        scaled_lse: cute.Tensor,
        sum_OdO_scale: Float32,
        lse_scale: Float32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, tidy, _ = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]

        for idx_q_t in cutlass.range(tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True):
            idx_q = idx_q_t + self.sum_OdO_block_q * bidx
            if idx_q < seqlen_q:
                O_bhq = O[idx_q, None, (bidy, bidz)]
                O_bhq = cute.logical_divide(O_bhq, cute.make_layout(self.sum_OdO_elem_per_load))
                dO_bhq = dO[idx_q, None, (bidy, bidz)]
                dO_bhq = cute.logical_divide(dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load))

                idx_d_start = tidx
                idx_d_step = self.sum_OdO_num_threads_d
                acc = 0.0
                for idx_d in cutlass.range(idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step):
                    O_frag = O_bhq[None, idx_d].load()
                    dO_frag = dO_bhq[None, idx_d].load()
                    # Multiply in FP32: BF16 products lose the cancellation
                    # against the FP32 dO @ V MMA (even when O == V).
                    prod_frag = O_frag.to(self.acc_dtype) * dO_frag.to(self.acc_dtype)
                    acc += prod_frag.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

                acc = cute.arch.warp_reduction_sum(acc, threads_in_group=self.sum_OdO_num_threads_d)

                if tidx == 0:
                    lse_bhq = lse[idx_q, (bidy, bidz)]
                    sum_OdO[idx_q, (bidy, bidz)] = sum_OdO_scale * acc
                    scaled_lse[idx_q, (bidy, bidz)] = lse_scale * lse_bhq

    @cute.kernel
    def bwd(
        self,
        QK_tiled_mma: cute.TiledMma,
        fake_QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        fake_dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        dQ_acc: cute.Tensor,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: Float32,
        sum_OdO: cute.Tensor,
        bucketed_k2q_offsets: cute.Tensor,
        bucketed_k2q_indices: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        variable_block_sizes: cute.Tensor,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        KT_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
        pair_flags: cute.Tensor | None,
        pair_residuals: cute.Tensor | None,
    ):
        QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.score_tiler[:2],
        )
        dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.score_tiler[:2],
        )
        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.score_tiler[:2],
        )
        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.score_tiler[:2],
        )
        bidx, bidy, bidz = self.logical_block_coord(problem_shape[0])
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB

        tasks_per_group = (cute.size(bucketed_k2q_offsets.shape[0]) - 1) // self.kv_group_size
        task_linear = bidx
        q_group = task_linear // tasks_per_group
        task_idx = task_linear - q_group * tasks_per_group

        kv_block_idx = task_idx
        k2q_begin = bucketed_k2q_offsets[task_idx * self.kv_group_size, q_group, (bidy, bidz)]
        k2q_end = bucketed_k2q_offsets[task_idx * self.kv_group_size + 1, q_group, (bidy, bidz)]
        if cutlass.const_expr(pair_residuals is not None):
            if cutlass.const_expr(self.kv_group_size == 2):
                k2q_begin += pair_residuals[bidz, bidy, q_group, task_idx * 2]
            else:
                k2q_end = k2q_begin + pair_residuals[bidz, bidy, q_group, task_idx]
        iter_count = k2q_end - k2q_begin
        iter_index = Int32(0)
        load_iter_count = iter_count
        mma_iter_count = cute.ceil_div(iter_count, 2)
        compute_iter_count = mma_iter_count
        reduce_iter_count = iter_count

        task_has_work = iter_count > 0
        task_has_work = task_has_work and kv_block_idx * self.QK_mma_tiler[1] < seqlen_k

        if cutlass.const_expr(pair_flags is not None):
            if cutlass.const_expr(self.kv_group_size == 2):
                task_has_work = task_has_work and pair_flags[bidz, bidy, kv_block_idx] != 0
            else:
                if cutlass.const_expr(pair_residuals is None):
                    task_has_work = task_has_work and pair_flags[bidz, bidy, kv_block_idx // 2] == 0

        if task_has_work:
            if warp_idx == self.load_warp_id:
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)

            smem = SmemAllocator()
            storage = smem.allocate(self.shared_storage)

            load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(storage.load_mma_Q_mbar_ptr.data_ptr())
            load_mma_dO_pipeline = self.make_and_init_load_mma_dO_pipeline(storage.load_mma_dO_mbar_ptr.data_ptr())
            load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(storage.load_compute_lse_mbar_ptr.data_ptr())
            load_compute_sum_OdO_pipeline = self.make_and_init_load_compute_sum_OdO_pipeline(storage.load_compute_sum_OdO_mbar_ptr.data_ptr())
            mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(storage.mma_compute_S_mbar_ptr.data_ptr())
            mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(storage.mma_compute_dP_mbar_ptr.data_ptr())
            mma_reduce_dQ_pipeline = self.make_and_init_mma_reduce_dQ_pipeline(storage.mma_reduce_dQ_mbar_ptr.data_ptr())
            compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(storage.compute_mma_P_mbar_ptr.data_ptr())
            compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(storage.compute_mma_dS_mbar_ptr.data_ptr())
            mma_compute_dKdV_pipeline = self.make_and_init_mma_compute_dKdV_pipeline(storage.mma_compute_dKdV_mbar_ptr.data_ptr())
            reduce_tma_store_pipeline = self.make_and_init_reduce_tma_store_pipeline()

            self.cta_sync_barrier.arrive_and_wait()

            sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
            sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
            sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
            sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)
            sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
            if cutlass.const_expr(self.alias_ps):
                sdS = storage.sP.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
            else:
                sdS = storage.sdS.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
            sdQ = storage.sdQ.get_tensor(dQ_smem_layout_staged.outer, swizzle=dQ_smem_layout_staged.inner)
            sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
            sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)

            tmem_holding_buf = storage.tmem_holding_buf.ptr
            tmem = TmemAllocator(
                tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
                allocator_warp_id=self.mma_warp_id,
            )

            sQT_ptr = cute.recast_ptr(sQ.iterator, QT_smem_layout_staged.inner)
            sQT = cute.make_tensor(sQT_ptr, QT_smem_layout_staged.outer)
            sKT_ptr = cute.recast_ptr(sK.iterator, KT_smem_layout_staged.inner)
            sKT = cute.make_tensor(sKT_ptr, KT_smem_layout_staged.outer)
            sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
            sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)
            sdOT_ptr = cute.recast_ptr(sdO.iterator, dOT_smem_layout_staged.inner)
            sdOT = cute.make_tensor(sdOT_ptr, dOT_smem_layout_staged.outer)

            # (MMA, MMA_M, MMA_K, STAGE)
            tSrQ = QK_tiled_mma.make_fragment_B(sQ)
            # (MMA, MMA_N, MMA_K, STAGE)
            tSrK = QK_tiled_mma.make_fragment_A(sK)

            tdPrdO = dOV_tiled_mma.make_fragment_B(sdO)
            tdPrV = dOV_tiled_mma.make_fragment_A(sV)

            tdKTrQT = QdS_tiled_mma.make_fragment_B(sQT)
            tdKTrdST = QdS_tiled_mma.make_fragment_A(sdST)

            tdVTrdOT = dOP_tiled_mma.make_fragment_B(sdOT)
            tdVTrP = dOP_tiled_mma.make_fragment_A(sP)

            tdQrdS = dSK_tiled_mma.make_fragment_A(sdS)
            tdQrKT = dSK_tiled_mma.make_fragment_B(sKT)

            if warp_idx == self.load_warp_id or warp_idx == self.load_warp_id + 1:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                for loader_role in cutlass.range_constexpr(2):
                    if warp_idx == self.load_warp_id + loader_role:
                        self.load(
                            Q_in,
                            K_in,
                            V_in,
                            dO_in,
                            LSE,
                            sum_OdO,
                            sQ,
                            sK,
                            sV,
                            sdO,
                            sLSE,
                            sSum_OdO,
                            bucketed_k2q_indices,
                            k2q_begin,
                            kv_block_idx,
                            fake_QK_tiled_mma,
                            fake_dOV_tiled_mma,
                            tma_atom_Q,
                            tma_atom_K,
                            tma_atom_V,
                            tma_atom_dO,
                            problem_shape,
                            load_iter_count,
                            iter_index,
                            (
                                load_mma_Q_pipeline,
                                load_compute_LSE_pipeline,
                                load_mma_dO_pipeline,
                                load_compute_sum_OdO_pipeline,
                            ),
                            loader_role,
                        )
            elif warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_mma)

                tmem.allocate(self.tmem_alloc_cols)
                # Barrier before retrieve tensor memory ptr from shared memory
                self.tmem_alloc_barrier.arrive_and_wait()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tStS_shape = QK_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
                tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

                tdPtdP_shape = dOV_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
                tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset, tdPtdP.layout)

                tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(cute.select(self.dSK_mma_tiler, mode=[0, 1]))
                tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
                tdQtdQ = cute.make_tensor(tmem_ptr_base + self.tmem_dQ_offset, tdQtdQ.layout)

                tdKTtdKT_shape = QdS_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tdKTtdKT = QdS_tiled_mma.make_fragment_C(tdKTtdKT_shape)
                tdKTtdKT = cute.make_tensor(tmem_ptr_base + self.tmem_dK_offset, tdKTtdKT.layout)

                tdVTtdVT_shape = dOP_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tdVTtdVT = dOP_tiled_mma.make_fragment_C(tdVTtdVT_shape)
                tdVTtdVT = cute.make_tensor(tmem_ptr_base + self.tmem_dV_offset, tdVTtdVT.layout)

                self.mma(
                    QK_tiled_mma,
                    dOV_tiled_mma,
                    dOP_tiled_mma,
                    QdS_tiled_mma,
                    dSK_tiled_mma,
                    tStS,
                    tSrQ,
                    tSrK,
                    tdPtdP,
                    tdPrdO,
                    tdPrV,
                    tdVTtdVT,
                    tdVTrdOT,
                    tdVTrP,
                    tdQtdQ,
                    tdQrdS,
                    tdQrKT,
                    tdKTtdKT,
                    tdKTrQT,
                    tdKTrdST,
                    mma_iter_count,
                    (
                        load_mma_Q_pipeline,
                        mma_compute_S_pipeline,
                        load_mma_dO_pipeline,
                        mma_compute_dP_pipeline,
                        mma_reduce_dQ_pipeline,
                        compute_mma_P_pipeline,
                        compute_mma_dS_pipeline,
                        mma_compute_dKdV_pipeline,
                    ),
                )
            elif warp_idx in self.compute_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_compute)
                self.tmem_alloc_barrier.arrive_and_wait()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tStS_shape = QK_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
                tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

                tdPtdP_shape = dOV_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
                tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset, tdPtdP.layout)

                tdKTtdKT_shape = QdS_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tdKTtdKT = QdS_tiled_mma.make_fragment_C(tdKTtdKT_shape)
                tdKTtdKT = cute.make_tensor(tmem_ptr_base + self.tmem_dK_offset, tdKTtdKT.layout)

                tdVTtdVT_shape = dOP_tiled_mma.partition_shape_C(self.score_tiler[:2])
                tdVTtdVT = dOP_tiled_mma.make_fragment_C(tdVTtdVT_shape)
                tdVTtdVT = cute.make_tensor(tmem_ptr_base + self.tmem_dV_offset, tdVTtdVT.layout)
                self.compute(
                    tStS,
                    tdPtdP,
                    sLSE,
                    sdS,
                    sP,
                    sSum_OdO,
                    dK_acc,
                    dV_acc,
                    tdKTtdKT,
                    tdVTtdVT,
                    kv_block_idx,
                    variable_block_sizes,
                    problem_shape,
                    compute_iter_count,
                    scale_softmax,
                    (
                        mma_compute_S_pipeline,
                        compute_mma_P_pipeline,
                        load_compute_LSE_pipeline,
                        load_compute_sum_OdO_pipeline,
                        mma_compute_dP_pipeline,
                        compute_mma_dS_pipeline,
                        mma_compute_dKdV_pipeline,
                    ),
                )

                self.epilogue_sync_barrier.arrive_and_wait()
                if warp_idx % self.num_compute_warps == 0:
                    tmem_ptr = cute.arch.retrieve_tmem_ptr(
                        Float32,
                        alignment=16,
                        ptr_to_buffer_holding_addr=tmem_holding_buf,
                    )
                    cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
            elif warp_idx in self.reduce_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_reduce)

                self.tmem_alloc_barrier.arrive_and_wait()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(cute.select(self.dSK_mma_tiler, mode=[0, 1]))
                tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
                tdQtdQ = cute.make_tensor(tmem_ptr_base + self.tmem_dQ_offset, tdQtdQ.layout)

                self.reduce(
                    problem_shape,
                    tdQtdQ,
                    bucketed_k2q_indices,
                    k2q_begin,
                    tma_atom_dQ_acc,
                    dQ_acc,
                    sdQ,
                    reduce_iter_count,
                    (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline),
                )
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_empty)

    @cute.kernel
    def convert(
        self,
        dQ_acc: cute.Tensor,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        bucketed_k2q_offsets: cute.Tensor,
        q_count: Int32,
        k_count: Int32,
        d_dim: Int32,
        scale_softmax: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        # Grid is laid out as (seq_tile, H, B) so the seq dim uses grid.x.
        seq_tile_idx, h_idx, b_idx = cute.arch.block_idx()

        for idx_s_t in cutlass.range(tidy, self.block_seq, self.num_threads_seq):
            idx_s = idx_s_t + self.block_seq * seq_tile_idx
            if idx_s < q_count:
                dQ_acc_bhs = dQ_acc[idx_s, None, (h_idx, b_idx)]
                dQ_acc_bhs = cute.logical_divide(dQ_acc_bhs, cute.make_layout(self.convert_elem_per_load))
                dQ_bhs = dQ[idx_s, None, (h_idx, b_idx)]
                dQ_bhs = cute.logical_divide(dQ_bhs, cute.make_layout(self.convert_elem_per_load))

                thr_start = tidx
                thr_step = self.num_threads_D_convert
                for idx_d in cutlass.range(
                    thr_start,
                    d_dim // self.convert_elem_per_load,
                    thr_step,
                ):
                    dQ_acc_frg = dQ_acc_bhs[None, idx_d].load()
                    dQ_acc_frg = scale_softmax * dQ_acc_frg
                    dQ_bhs[None, idx_d].store(dQ_acc_frg.to(self.element_dtype))
            if idx_s < k_count:
                dK_bhs = dK[None, idx_s, (h_idx, b_idx)]
                dK_bhs = cute.logical_divide(dK_bhs, cute.make_layout(self.convert_elem_per_load))
                dV_bhs = dV[None, idx_s, (h_idx, b_idx)]
                dV_bhs = cute.logical_divide(dV_bhs, cute.make_layout(self.convert_elem_per_load))

                thr_start = tidx
                thr_step = self.num_threads_D_convert
                if cutlass.const_expr(self.dKV_postprocess):
                    dK_acc_bhs = dK_acc[None, idx_s, (h_idx, b_idx)]
                    dK_acc_bhs = cute.logical_divide(dK_acc_bhs, cute.make_layout(self.convert_elem_per_load))
                    dV_acc_bhs = dV_acc[None, idx_s, (h_idx, b_idx)]
                    dV_acc_bhs = cute.logical_divide(dV_acc_bhs, cute.make_layout(self.convert_elem_per_load))
                    for idx_d in cutlass.range(
                        thr_start,
                        d_dim // self.convert_elem_per_load,
                        thr_step,
                    ):
                        dK_acc_frg = dK_acc_bhs[None, idx_d].load()
                        dV_acc_frg = dV_acc_bhs[None, idx_d].load()
                        dK_bhs[None, idx_d].store((scale_softmax * dK_acc_frg).to(self.element_dtype))
                        dV_bhs[None, idx_d].store(dV_acc_frg.to(self.element_dtype))
                else:
                    kv_block_idx = idx_s // self.sparse_block_size
                    k2q_begin = bucketed_k2q_offsets[kv_block_idx, 0, (h_idx, b_idx)]
                    k2q_end = bucketed_k2q_offsets[kv_block_idx + 1, 0, (h_idx, b_idx)]
                    if k2q_begin == k2q_end:
                        # Empty tasks skip the bwd kernel and its TMEM
                        # epilogue, so initialize their caller outputs here.
                        zeros = cute.make_rmem_tensor(self.convert_elem_per_load, self.element_dtype)
                        zeros.fill(0)
                        for idx_d in cutlass.range(
                            thr_start,
                            d_dim // self.convert_elem_per_load,
                            thr_step,
                        ):
                            dK_bhs[None, idx_d].store(zeros.load())
                            dV_bhs[None, idx_d].store(zeros.load())

    @cute.jit
    def load(
        self,
        Q_in: cute.Tensor,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        dO_in: cute.Tensor,
        LSE: cute.Tensor,
        sum_OdO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        bucketed_k2q_indices: cute.Tensor,
        k2q_begin: Int32,
        kv_block_idx: Int32,
        fake_QK_tiled_mma: cute.TiledMma,
        fake_dOV_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        iter_count: Int32,
        iter_index: Int32,
        pipeline_args: tuple,
        loader_role: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % self.threads_per_warp
        async_copy_num_elts = self.sparse_block_size // self.threads_per_warp
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=self.acc_dtype.width,
        )
        _, blk_coord_h, blk_coord_b = self.logical_block_coord(problem_shape[0])
        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB
        (
            load_mma_Q_pipeline,
            load_compute_LSE_pipeline,
            load_mma_dO_pipeline,
            load_compute_sum_OdO_pipeline,
        ) = pipeline_args

        total_iter_count = iter_count

        # (bM, bK, RestM, RestK, (H, B))
        gQ = cute.local_tile(Q_in, cute.select(self.fake_QK_mma_tiler, mode=[0, 2]), (None, None, None))
        # (bN, bK, RestN, RestK, (H, B))
        gK = cute.local_tile(K_in, cute.select(self.QK_mma_tiler, mode=[1, 2]), (None, None, None))
        # (bM, bK, RestM, RestK, (H, B))
        gdO = cute.local_tile(dO_in, cute.select(self.fake_dOV_mma_tiler, mode=[0, 2]), (None, None, None))
        # (bN, bK, RestN, RestK, (H, B))
        gV = cute.local_tile(V_in, cute.select(self.dOV_mma_tiler, mode=[1, 2]), (None, None, None))

        QK_thr_mma = fake_QK_tiled_mma.get_slice(0)
        dOV_thr_mma = fake_dOV_tiled_mma.get_slice(0)

        tSgQ = QK_thr_mma.partition_A(gQ)
        tSgK = QK_thr_mma.partition_B(gK)
        tdPgdO = dOV_thr_mma.partition_A(gdO)
        tdPgV = dOV_thr_mma.partition_B(gV)

        load_mma_Q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_Q_stage)
        load_compute_LSE_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage)
        load_mma_dO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_dO_stage)
        load_compute_sum_OdO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage)

        sQ = cute.make_tensor(sQ.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
        sQ_0 = sQ[None, 0, None, load_mma_Q_producer_state.index]
        sQ_1 = sQ[None, 1, None, load_mma_Q_producer_state.index]

        sdO = cute.make_tensor(sdO.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
        sdO_0 = sdO[None, 0, None, load_mma_dO_producer_state.index]
        sdO_1 = sdO[None, 1, None, load_mma_dO_producer_state.index]
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tQsQ_0, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_0, 0, 2), cute.group_modes(tSgQ, 0, 3))
        tQsQ_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_1, 0, 2), cute.group_modes(tSgQ, 0, 3))
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tdOsdO_0, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_0, 0, 2), cute.group_modes(tdPgdO, 0, 3)
        )
        tdOsdO_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_1, 0, 2), cute.group_modes(tdPgdO, 0, 3))
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(tma_atom_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tdPgV, 0, 3))

        q_block_idx_0 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
        iter_index += 1
        q_block_idx_1 = cute.ceil_div(seqlen_q, self.sparse_block_size)  # First block beyond the final partial Q block.
        if iter_index < total_iter_count:
            q_block_idx_1 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
        q_block_0_full = (q_block_idx_0 + 1) * self.sparse_block_size <= seqlen_q

        if cutlass.const_expr(loader_role == 0):
            load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
            tma_barrier = load_mma_Q_pipeline.producer_get_barrier(load_mma_Q_producer_state)
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_Q_bytes * (self.kv_group_size + 1))

            # Load K
            cute.copy(
                tma_atom_K,
                tKgK_mkl[(None, kv_block_idx, 0, (blk_coord_h, blk_coord_b))],
                tKsK[None, 0],
                tma_bar_ptr=tma_barrier,
            )

            # Load Q0
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                tQsQ_0,
                tma_bar_ptr=tma_barrier,
            )

            # Load Q1
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                tQsQ_1,
                tma_bar_ptr=tma_barrier,
            )

            load_mma_Q_producer_state.advance()

        if cutlass.const_expr(loader_role == 1):
            load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

            # Load LSE
            # 32 threads load 64 values, each thread loads 2 values
            sLSE_for_copy = cute.flat_divide(sLSE, (1,))
            LSE_for_copy = cute.flat_divide(LSE, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                LSE_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                if q_block_0_full:
                    cute.copy(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_LSE_producer_state.index,
                        ],
                    )
                else:
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_LSE_producer_state.index,
                        ],
                        cute.elem_less(LSE_idx + i, seqlen_q),
                    )

            for i in cutlass.range_constexpr(async_copy_num_elts):
                LSE_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                _cp_async_copy_or_zero_fill(
                    atom_async_copy,
                    LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[
                        None,
                        self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                        load_compute_LSE_producer_state.index,
                    ],
                    cute.elem_less(LSE_idx + i, seqlen_q),
                )

            load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
            load_compute_LSE_producer_state.advance()

            load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
            tma_barrier = load_mma_dO_pipeline.producer_get_barrier(load_mma_dO_producer_state)
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dO_bytes * (self.kv_group_size + 1))

            # Load dO0
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO_0,
                tma_bar_ptr=tma_barrier,
            )
            # Load dO1
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO_1,
                tma_bar_ptr=tma_barrier,
            )

            # Load V
            cute.copy(
                tma_atom_V,
                tVgV_mkl[(None, kv_block_idx, 0, (blk_coord_h, blk_coord_b))],
                tVsV[None, 0],
                tma_bar_ptr=tma_barrier,
            )

            load_mma_dO_producer_state.advance()

            load_compute_sum_OdO_pipeline.producer_acquire(load_compute_sum_OdO_producer_state)

            sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
            sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                sum_OdO_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                if q_block_0_full:
                    cute.copy(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                    )
                else:
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                        cute.elem_less(sum_OdO_idx + i, seqlen_q),
                    )
            for i in cutlass.range_constexpr(async_copy_num_elts):
                sum_OdO_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                _cp_async_copy_or_zero_fill(
                    atom_async_copy,
                    sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[
                        None,
                        self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                        load_compute_sum_OdO_producer_state.index,
                    ],
                    cute.elem_less(sum_OdO_idx + i, seqlen_q),
                )

            load_compute_sum_OdO_pipeline.producer_commit(load_compute_sum_OdO_producer_state)
            load_compute_sum_OdO_producer_state.advance()

        iter_count -= 2
        iter_index += 1

        while iter_count > 0:

            sQ = cute.make_tensor(sQ.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
            sQ_0 = sQ[None, 0, None, load_mma_Q_producer_state.index]
            sQ_1 = sQ[None, 1, None, load_mma_Q_producer_state.index]

            sdO = cute.make_tensor(sdO.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
            sdO_0 = sdO[None, 0, None, load_mma_dO_producer_state.index]
            sdO_1 = sdO[None, 1, None, load_mma_dO_producer_state.index]

            # ((atom_v, rest_v), STAGE)
            # ((atom_v, rest_v), RestM, RestK, (H, B))
            tQsQ_0, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_0, 0, 2), cute.group_modes(tSgQ, 0, 3))
            tQsQ_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_1, 0, 2), cute.group_modes(tSgQ, 0, 3))
            # ((atom_v, rest_v), STAGE)
            # ((atom_v, rest_v), RestM, RestK, (H, B))
            tdOsdO_0, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_0, 0, 2), cute.group_modes(tdPgdO, 0, 3))
            tdOsdO_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_1, 0, 2), cute.group_modes(tdPgdO, 0, 3))

            q_block_idx_0 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
            iter_index += 1
            q_block_idx_1 = cute.ceil_div(seqlen_q, self.sparse_block_size)  # First block beyond the final partial Q block.
            if iter_index < total_iter_count:
                q_block_idx_1 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
            q_block_0_full = (q_block_idx_0 + 1) * self.sparse_block_size <= seqlen_q

            if cutlass.const_expr(loader_role == 0):
                load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
                tma_barrier = load_mma_Q_pipeline.producer_get_barrier(load_mma_Q_producer_state)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_Q_bytes)

                # Load Q0
                cute.copy(
                    tma_atom_Q,
                    tQgQ_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                    tQsQ_0,
                    tma_bar_ptr=tma_barrier,
                )

                # Load Q1
                cute.copy(
                    tma_atom_Q,
                    tQgQ_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                    tQsQ_1,
                    tma_bar_ptr=tma_barrier,
                )

                load_mma_Q_producer_state.advance()

            if cutlass.const_expr(loader_role == 1):
                load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

                # Load LSE
                # 32 threads load 64 values, each thread loads 2 values
                sLSE_for_copy = cute.flat_divide(sLSE, (1,))
                LSE_for_copy = cute.flat_divide(LSE, (1,))
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    LSE_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    if q_block_0_full:
                        cute.copy(
                            atom_async_copy,
                            LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                            sLSE_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_LSE_producer_state.index,
                            ],
                        )
                    else:
                        _cp_async_copy_or_zero_fill(
                            atom_async_copy,
                            LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                            sLSE_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_LSE_producer_state.index,
                            ],
                            cute.elem_less(LSE_idx + i, seqlen_q),
                        )

                for i in cutlass.range_constexpr(async_copy_num_elts):
                    LSE_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                            load_compute_LSE_producer_state.index,
                        ],
                        cute.elem_less(LSE_idx + i, seqlen_q),
                    )

                load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
                load_compute_LSE_producer_state.advance()

                load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
                tma_barrier = load_mma_dO_pipeline.producer_get_barrier(load_mma_dO_producer_state)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dO_bytes)

                # Load dO0
                cute.copy(
                    tma_atom_dO,
                    tdOgdO_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                    tdOsdO_0,
                    tma_bar_ptr=tma_barrier,
                )
                # Load dO1
                cute.copy(
                    tma_atom_dO,
                    tdOgdO_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                    tdOsdO_1,
                    tma_bar_ptr=tma_barrier,
                )

                load_mma_dO_producer_state.advance()

                load_compute_sum_OdO_pipeline.producer_acquire(load_compute_sum_OdO_producer_state)

                sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
                sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    sum_OdO_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    if q_block_0_full:
                        cute.copy(
                            atom_async_copy,
                            sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                            sSum_OdO_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_sum_OdO_producer_state.index,
                            ],
                        )
                    else:
                        _cp_async_copy_or_zero_fill(
                            atom_async_copy,
                            sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                            sSum_OdO_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_sum_OdO_producer_state.index,
                            ],
                            cute.elem_less(sum_OdO_idx + i, seqlen_q),
                        )
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    sum_OdO_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                        cute.elem_less(sum_OdO_idx + i, seqlen_q),
                    )

                load_compute_sum_OdO_pipeline.producer_commit(load_compute_sum_OdO_producer_state)
                load_compute_sum_OdO_producer_state.advance()

            iter_count -= 2
            iter_index += 1

    @cute.jit
    def mma(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tStS: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdPrV: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        tdVTrdOT: cute.Tensor,
        tdVTrP: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdQrdS: cute.Tensor,
        tdQrKT: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdKTrQT: cute.Tensor,
        tdKTrdST: cute.Tensor,
        iter_count: Int32,
        pipeline_args: tuple,
    ):
        (
            load_mma_Q_pipeline,
            mma_compute_S_pipeline,
            load_mma_dO_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        load_mma_Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_Q_stage)
        load_mma_Q_release_state = load_mma_Q_consumer_state.clone()
        mma_compute_S_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        mma_reduce_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_reduce_dQ_stage)
        load_mma_dO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_dO_stage)
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage)
        mma_compute_dKdV_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dKdV_stage)

        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.score_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        p_layout = sm100_utils.make_smem_layout_a(dOP_tiled_mma, self.score_tiler, self.element_dtype, 1)
        p_tensor = cute.make_tensor(cute.recast_ptr(tStS.iterator, dtype=self.element_dtype), p_layout.outer)
        tdVTrP = dOP_tiled_mma.make_fragment_A(p_tensor)
        tdVTrP = cute.make_tensor(p_tensor.iterator, tdVTrP.layout)

        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.score_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        ds_layout = sm100_utils.make_smem_layout_a(QdS_tiled_mma, self.score_tiler, self.element_dtype, 1)
        ds_tensor = cute.make_tensor(cute.recast_ptr(tdPtdP.iterator, dtype=self.element_dtype), ds_layout.outer)
        tdKTrdST = QdS_tiled_mma.make_fragment_A(ds_tensor)
        tdKTrdST = cute.make_tensor(ds_tensor.iterator, tdKTrdST.layout)

        load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
        mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

        # S = Q * K
        QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
            cute.gemm(
                QK_tiled_mma,
                tStS,
                tSrK[None, None, k_block, 0],
                tSrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                tStS,
            )
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        load_mma_Q_consumer_state.advance()
        mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
        mma_compute_S_producer_state.advance()

        load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

        mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
        mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)

        # dP = dO * V
        dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll_full=True):
            cute.gemm(
                dOV_tiled_mma,
                tdPtdP,
                tdPrV[None, None, k_block, 0],
                tdPrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                tdPtdP,
            )
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
        mma_compute_dP_producer_state.advance()

        compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

        # dV = dO * P
        dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdVTrdOT, mode=[2]), unroll_full=True):
            cute.gemm(
                dOP_tiled_mma,
                tdVTtdVT,
                tdVTrP[None, None, k_block, 0],
                tdVTrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                tdVTtdVT,
            )
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
        compute_mma_P_consumer_state.advance()

        load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
        load_mma_dO_consumer_state.advance()

        iter_count -= 1

        while iter_count > 0:
            load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

            # S = Q * K
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
                cute.gemm(
                    QK_tiled_mma,
                    tStS,
                    tSrK[None, None, k_block, 0],
                    tSrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                    tStS,
                )
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            load_mma_Q_consumer_state.advance()
            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

            # dK = Q * dS
            for k_block in cutlass.range(0, cute.size(tdKTrQT, mode=[2]), unroll_full=True):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKTtdKT,
                    tdKTrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKTrQT[None, None, k_block, load_mma_Q_release_state.index],
                    tdKTtdKT,
                )
                QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ = dS * K
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
                cute.gemm(
                    dSK_tiled_mma,
                    tdQtdQ,
                    tdQrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQrKT[None, None, k_block, 0],
                    tdQtdQ,
                )
                dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
            mma_reduce_dQ_producer_state.advance()

            load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
            load_mma_Q_release_state.advance()

            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

            load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

            # dV uses independent TMEM, so issue it before waiting for dQ readers.
            compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

            # dV = dO * P
            for k_block in cutlass.range(0, cute.size(tdVTrdOT, mode=[2]), unroll_full=True):
                cute.gemm(
                    dOP_tiled_mma,
                    tdVTtdVT,
                    tdVTrP[None, None, k_block, 0],
                    tdVTrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdVTtdVT,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)

            # dP = dO * V
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll_full=True):
                cute.gemm(
                    dOV_tiled_mma,
                    tdPtdP,
                    tdPrV[None, None, k_block, 0],
                    tdPrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdPtdP,
                )
                dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
            load_mma_dO_consumer_state.advance()

            iter_count -= 1

        mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_producer_state.advance()

        mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)

        compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

        # dK = Q * dS
        for k_block in cutlass.range(0, cute.size(tdKTrQT, mode=[2]), unroll_full=True):
            cute.gemm(
                QdS_tiled_mma,
                tdKTtdKT,
                tdKTrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                tdKTrQT[None, None, k_block, load_mma_Q_release_state.index],
                tdKTtdKT,
            )
            QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        # dQ = dS * K
        dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
            cute.gemm(
                dSK_tiled_mma,
                tdQtdQ,
                tdQrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                tdQrKT[None, None, k_block, 0],
                tdQtdQ,
            )
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
        mma_reduce_dQ_producer_state.advance()

        # dK epilogue permits TMEM deallocation: wait for the final dQ reader first.
        mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
        mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_producer_state.advance()

        load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
        load_mma_Q_release_state.advance()

        compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
        compute_mma_dS_consumer_state.advance()

    @cute.jit
    def compute(
        self,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        sLSE: cute.Tensor,
        sdS: cute.Tensor,
        sP: cute.Tensor,
        sSum_OdO: cute.Tensor,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        kv_block_idx: Int32,
        variable_block_sizes: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        iter_count: Int32,
        scale_softmax: Float32,
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, blk_coord_h, blk_coord_b = self.logical_block_coord(problem_shape[0])
        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB
        (
            mma_compute_S_pipeline,
            compute_mma_P_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage)
        compute_mma_P_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_P_stage)
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage)
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage)
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage)
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage)
        mma_compute_dKdV_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        # (128, 64)
        tStS = tStS[(None, None), 0, 0]
        # (128, 64)
        tdPtdP = tdPtdP[(None, None), 0, 0]

        cS = cute.make_identity_tensor(self.score_tiler[:2])
        cdP = cute.make_identity_tensor(self.score_tiler[:2])

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cS_p = thr_t2r.partition_D(cS)
        tTR_cS = self.split_wg(tTR_cS_p, num_warp_groups, wg_idx)
        tTR_rS = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)

        tTR_tS = thr_t2r.partition_S(tStS)
        tTR_tS = self.split_wg(tTR_tS, num_warp_groups, wg_idx)

        tTR_cdP_p = thr_t2r.partition_D(cdP)
        tTR_cdP = self.split_wg(tTR_cdP_p, num_warp_groups, wg_idx)
        tTR_rdP = cute.make_rmem_tensor(tTR_cdP.shape, self.acc_dtype)

        tTR_tdP = thr_t2r.partition_S(tdPtdP)
        tTR_tdP = self.split_wg(tTR_tdP, num_warp_groups, wg_idx)

        tP_packed = cute.composition(tStS, cute.make_layout((64, 64)))
        p_store_atom = cute.make_copy_atom(tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(4)), Float32)
        p_store_copy = tcgen05.make_tmem_copy(p_store_atom, tP_packed)
        p_store_thr = p_store_copy.get_slice(dp_idx)
        p_store_dest = self.split_wg(p_store_thr.partition_D(tP_packed), num_warp_groups, wg_idx)

        tDS_packed = cute.composition(tdPtdP, cute.make_layout((64, 64)))
        ds_store_dest = self.split_wg(p_store_thr.partition_D(tDS_packed), num_warp_groups, wg_idx)

        block_size_k = Int32(self.sparse_block_size)
        if cutlass.const_expr(self.has_block_sizes):
            block_size_k = variable_block_sizes[blk_coord_b, kv_block_idx]

        while iter_count > 0:
            # Wait for S and P
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)
            # Wait for LSE
            load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)

            # Compute P = softmax(S, LSE)
            cute.copy(tiled_t2r, tTR_tS, tTR_rS)

            if cutlass.const_expr(self.has_block_sizes and self.kv_group_size == 2):
                for i in cutlass.range_constexpr(cute.size(tTR_rS)):
                    index_k, _ = tTR_cS[i]
                    valid_k = variable_block_sizes[blk_coord_b, kv_block_idx * 2 + index_k // 64]
                    tTR_rS[i] = tTR_rS[i] if index_k % 64 < valid_k else -Float32.inf
            else:
                if cutlass.const_expr(self.has_block_sizes):
                    # block_sizes describes valid K positions; Q rows stay full-sized.
                    if block_size_k < self.sparse_block_size:
                        for i in cutlass.range_constexpr(cute.size(tTR_rS)):
                            index_k, _ = tTR_cS[i]
                            is_valid = index_k < block_size_k
                            tTR_rS[i] = tTR_rS[i] if is_valid else -Float32.inf

            self.compute_probabilities(tTR_rS, tTR_cS, sLSE, load_compute_LSE_consumer_state.index, scale_softmax, 1)

            # convert fp32 P to bf16 P which will be used in the dOP
            tRS_rP = self.quantize(tTR_rS, 4)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()
            cute.arch.fence_view_async_tmem_load()

            # P overwrites S only after every compute warp has finished its TMEM load.
            packed_p = cute.recast_tensor(tRS_rP, Float32)
            cute.copy(p_store_copy, packed_p, p_store_dest)
            cute.arch.fence_view_async_tmem_store()

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            # Notify for P
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

            # Release S
            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            mma_compute_S_consumer_state.advance()

            # Release LSE
            load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
            load_compute_LSE_consumer_state.advance()

            # Wait for OdO
            load_compute_sum_OdO_pipeline.consumer_wait(load_compute_sum_OdO_consumer_state)
            # Wait for dP
            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)

            # Wait for dS
            compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)

            # Compute dS = dsoftmax(P, dP, sum_OdO)
            cute.copy(tiled_t2r, tTR_tdP, tTR_rdP)

            for i in cutlass.range(0, cute.size(tTR_rdP), 2, unroll_full=True):
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (
                        sSum_OdO[
                            cute.get(tTR_cdP[i], mode=[1]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                        sSum_OdO[
                            cute.get(tTR_cdP[i + 1], mode=[1]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                    ),
                )
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2((tTR_rdP[i], tTR_rdP[i + 1]), (tTR_rS[i], tTR_rS[i + 1]))

            # convert fp32 dS to bf16 dS which will be used in the computation of dK and dQ
            tTR_rdS = self.quantize(tTR_rdP, 4)

            # Every warp must finish reading dP before packed dS overwrites it.
            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()
            packed_ds = cute.recast_tensor(tTR_rdS, Float32)
            cute.copy(p_store_copy, packed_ds, ds_store_dest)
            cute.arch.fence_view_async_tmem_store()

            # Release dP
            cute.arch.fence_view_async_tmem_load()
            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
            mma_compute_dP_consumer_state.advance()

            sdS_slice = sdS[None, None, None, compute_mma_dS_producer_state.index]

            sdS_qk = cute.composition(sdS_slice, cute.make_layout((128, 64)))
            sdS_kq = cute.make_tensor(sdS_qk.iterator, cute.select(sdS_qk.layout, mode=[1, 0]))
            sdS_slice = self.split_wg(thr_t2r.partition_D(sdS_kq), num_warp_groups, wg_idx)

            cute.autovec_copy(tTR_rdS, sdS_slice)

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            # Release OdO
            load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)
            load_compute_sum_OdO_consumer_state.advance()

            iter_count -= 1

        self.epilogue(
            problem_shape,
            dK_acc,
            dV_acc,
            tdKTtdKT,
            tdVTtdVT,
            kv_block_idx,
            scale_softmax,
            (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state),
        )

    @cute.jit
    def reduce(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        tdQtdQ: cute.Tensor,
        bucketed_k2q_indices: cute.Tensor,
        k2q_begin: Int32,
        tma_atom_dQ_acc: cute.CopyAtom,
        mdQ_acc: cute.Tensor,
        sdQ: cute.Tensor,
        iter_count: Int32,
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        Q, K, D, HB = problem_shape
        H, B = HB
        _, blk_coord_h, blk_coord_b = self.logical_block_coord(problem_shape[0])
        mma_reduce_dQ_pipeline, reduce_tma_store_pipeline = pipeline_args
        total_iter_count = iter_count

        mma_reduce_dQ_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_reduce_dQ_stage)
        reduce_tma_store_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.reduce_tma_store_stage)

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )

        gdQ = cute.local_tile(mdQ_acc, (self.fake_QK_mma_tiler[0], 32), (None, None, None))
        cdQ = cute.make_identity_tensor((self.dSK_mma_tiler[0], self.dSK_mma_tiler[1]))
        thread_idx = tidx % (self.num_reduce_warps * self.threads_per_warp)

        tdQtdQ = tdQtdQ[(None, None), 0, 0]

        tiled_t2r = tcgen05.make_tmem_copy(load_op, tdQtdQ)
        thr_t2r = tiled_t2r.get_slice(thread_idx)

        tTR_cdQ = thr_t2r.partition_D(cdQ)
        tTR_sdQ = thr_t2r.partition_D(sdQ)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ)

        sdQ = cute.make_tensor(sdQ.iterator, cute.make_layout(((8, 8), 2, (32, 1), (1, 2)), stride=(((32, 256), 2048, (1, 0), (0, 4096)))))

        iter_index = Int32(0)

        while iter_count > 0:

            mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)

            q_block_idx_0 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
            iter_index += 1
            q_block_idx_1 = cute.ceil_div(Q, self.sparse_block_size)
            if iter_index < total_iter_count:
                q_block_idx_1 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]

            tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)

            # Load dQ from tmem to rmem
            cute.copy(tiled_t2r, tTR_tdQ, tTR_rdQ)

            cute.arch.fence_view_async_tmem_load()

            mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
            mma_reduce_dQ_consumer_state.advance()

            # We don't have enough smem to dump it all to smem, so we do it in stages
            for i in cutlass.range(0, cute.size(tTR_cdQ, mode=[2]), unroll_full=True):

                sdQ_0 = sdQ[None, 0, None, reduce_tma_store_producer_state.index]
                sdQ_1 = sdQ[None, 1, None, reduce_tma_store_producer_state.index]
                tdQsdQ_0, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_dQ_acc,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sdQ_0, 0, 2),
                    cute.group_modes(gdQ, 0, 2),
                )
                tdQsdQ_1, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_dQ_acc,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sdQ_1, 0, 2),
                    cute.group_modes(gdQ, 0, 2),
                )

                if warp_idx == 0:
                    reduce_tma_store_pipeline.producer_acquire()
                # Wait in all threads for the acquire to complete
                self.reduce_sync_barrier.arrive_and_wait()

                cute.autovec_copy(
                    tTR_rdQ[None, None, i],
                    tTR_sdQ[None, None, 0, reduce_tma_store_producer_state.index],
                )

                # Wait for the stores to all be visible to the TMA
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.reduce_sync_barrier.arrive_and_wait()

                if warp_idx == 0:
                    cute.copy(
                        tma_atom_dQ_acc,
                        tdQsdQ_0,
                        tdQgdQ[None, q_block_idx_0, i, (blk_coord_h, blk_coord_b)],
                    )
                    cute.copy(
                        tma_atom_dQ_acc,
                        tdQsdQ_1,
                        tdQgdQ[None, q_block_idx_1, i, (blk_coord_h, blk_coord_b)],
                    )

                    reduce_tma_store_pipeline.producer_commit()

                reduce_tma_store_producer_state.advance()

            iter_count -= 2
            iter_index += 1

        reduce_tma_store_pipeline.producer_tail()

    @cute.jit
    def epilogue(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        kv_block_idx: Int32,
        scale_softmax: Float32,
        # (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state)
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, D, HB = problem_shape
        H, B = HB
        _, blk_coord_h, blk_coord_b = self.logical_block_coord(problem_shape[0])
        mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tdKTtdKT = tdKTtdKT[(None, None), 0, 0]
        tdVTtdVT = tdVTtdVT[(None, None), 0, 0]

        dK_acc = cute.make_tensor(dK_acc.iterator, cute.select(dK_acc.layout, mode=[1, 0, 2]))
        dV_acc = cute.make_tensor(dV_acc.iterator, cute.select(dV_acc.layout, mode=[1, 0, 2]))
        gdK = cute.local_tile(dK_acc, cute.select(self.score_tiler, mode=[0, 1]), (None, None, None))
        gdK = gdK[None, None, kv_block_idx, 0, (blk_coord_h, blk_coord_b)]

        cdK = cute.domain_offset(
            (kv_block_idx * self.score_tiler[0], 0),
            cute.make_identity_tensor((self.score_tiler[0], self.score_tiler[1])),
        )
        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKTtdKT)
        thr_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thr_t2r_dK.partition_D(cdK)
        tTR_cdK = self.split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_gdK = thr_t2r_dK.partition_D(gdK)
        tTR_gdK = self.split_wg(tTR_gdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thr_t2r_dK.partition_S(tdKTtdKT)
        tTR_tdK = self.split_wg(tTR_tdK, num_warp_groups, wg_idx)

        gdV = cute.local_tile(dV_acc, cute.select(self.score_tiler, mode=[0, 1]), (None, None, None))
        gdV = gdV[None, None, kv_block_idx, 0, (blk_coord_h, blk_coord_b)]

        cdV = cute.domain_offset(
            (kv_block_idx * self.score_tiler[0], 0),
            cute.make_identity_tensor((self.score_tiler[0], self.score_tiler[1])),
        )

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVTtdVT)
        thr_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        tTR_cdV = thr_t2r_dV.partition_D(cdV)
        tTR_cdV = self.split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_gdV = thr_t2r_dV.partition_D(gdV)
        tTR_gdV = self.split_wg(tTR_gdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thr_t2r_dV.partition_S(tdVTtdVT)
        tTR_tdV = self.split_wg(tTR_tdV, num_warp_groups, wg_idx)

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        # Load tdVtdVT
        cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)

        if cutlass.const_expr(self.dKV_postprocess):
            self.store_add_fp32(tTR_gdV, tTR_rdV, tTR_cdV, (K, D))
        else:
            # dV = P.T @ dO; P already contains the scaled softmax, so dV
            # does not receive the additional softmax scale used by dQ/dK.
            cute.autovec_copy(self.quantize(tTR_rdV, 4), tTR_gdV)

        cute.arch.fence_view_async_tmem_load()

        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

        if cutlass.const_expr(self.dKV_postprocess):
            self.store_add_fp32(tTR_gdK, tTR_rdK, tTR_cdK, (K, D))
        else:
            # dK = softmax_scale * dS.T @ Q.
            for i in cutlass.range(0, cute.size(tTR_rdK), 2, unroll_full=True):
                tTR_rdK[i], tTR_rdK[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rdK[i], tTR_rdK[i + 1]),
                    (scale_softmax, scale_softmax),
                )
            cute.autovec_copy(self.quantize(tTR_rdK, 4), tTR_gdK)

        cute.arch.fence_view_async_tmem_load()
        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

    @cute.jit
    def store_add_fp32(self, gmem: cute.Tensor, regs: cute.Tensor, coord: cute.Tensor, tensor_shape: cute.Shape):
        pair = cute.make_rmem_tensor((2,), Float32)
        for pair_idx in cutlass.range_constexpr(cute.size(regs) // 2):
            i = 2 * pair_idx
            offset = cute.crd2idx(i, gmem.layout)
            next_offset = cute.crd2idx(i + 1, gmem.layout)
            # Ld16x256 gives adjacent D components. Keep a scalar fallback for other layouts.
            if next_offset == offset + 1 and offset % 2 == 0 and cute.elem_less(coord[i + 1], tensor_shape):
                if cute.elem_less(coord[i], tensor_shape):
                    ptr = gmem.iterator + offset
                    pair[0] = regs[i]
                    pair[1] = regs[i + 1]
                    cute.arch.atomic_add(ptr.llvm_ptr, pair.load(), sem="relaxed", scope="gpu")
            else:
                for j in cutlass.range_constexpr(2):
                    if cute.elem_less(coord[i + j], tensor_shape):
                        ptr = gmem.iterator + cute.crd2idx(i + j, gmem.layout)
                        cute.arch.atomic_add(ptr.llvm_ptr, regs[i + j], sem="relaxed", scope="gpu")

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        num_warp_groups: Int32,
        wg_idx: Int32,
    ) -> cute.Tensor:
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, (wg_idx, None)]
        else:
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        t.shape[2],
                        (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, None, (wg_idx, None)]
        return ret

    @cute.jit
    def compute_probabilities(self, scores, coords, lse, stage, scale, q_axis: cutlass.Constexpr[int]):
        # Preserve the common positive-scale instruction sequence; scale is a
        # runtime scalar, so no new compiled specialization is needed per sign.
        if scale > Float32(0.0):
            self._compute_probabilities(scores, coords, lse, stage, scale, q_axis, False)
        else:
            self._compute_probabilities(scores, coords, lse, stage, scale, q_axis, True)

    @cute.jit
    def _compute_probabilities(self, scores, coords, lse, stage, scale, q_axis: cutlass.Constexpr[int], preserve_mask: cutlass.Constexpr[bool]):
        scale_log2 = scale * Float32(math.log2(math.e))
        for i in cutlass.range(0, cute.size(scores), 2, unroll_full=True):
            row_lse = (
                lse[cute.get(coords[i], mode=[q_axis]), stage],
                lse[cute.get(coords[i + 1], mode=[q_axis]), stage],
            )
            if cutlass.const_expr(preserve_mask):
                masked0 = scores[i] == -Float32.inf
                masked1 = scores[i + 1] == -Float32.inf
            scores[i], scores[i + 1] = cute.arch.fma_packed_f32x2((scores[i], scores[i + 1]), (scale_log2, scale_log2), row_lse)
            if cutlass.const_expr(preserve_mask):
                # Neither -inf * 0 nor -inf * a negative scale is a mask.
                scores[i] = -Float32.inf if masked0 else scores[i]
                scores[i + 1] = -Float32.inf if masked1 else scores[i + 1]
            scores[i] = cute.math.exp2(scores[i], fastmath=True)
            scores[i + 1] = cute.math.exp2(scores[i + 1], fastmath=True)

    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
    ) -> cute.Tensor:
        output = cute.make_rmem_tensor(input.shape, self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr):
        load_mma_Q_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
        )

    def make_and_init_load_mma_dO_pipeline(self, load_mma_dO_mbar_ptr):
        load_mma_dO_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_dO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_dO_mbar_ptr,
            num_stages=self.load_mma_dO_stage,
            producer_group=load_mma_dO_producer_group,
            consumer_group=load_mma_dO_consumer_group,
            tx_count=self.tma_copy_dO_bytes,
        )

    def make_and_init_load_compute_LSE_pipeline(self, load_compute_lse_mbar_ptr):
        load_compute_lse_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_lse_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_lse_mbar_ptr,
            num_stages=self.load_compute_LSE_stage,
            producer_group=load_compute_lse_producer_group,
            consumer_group=load_compute_lse_consumer_group,
        )

    def make_and_init_load_compute_sum_OdO_pipeline(self, load_compute_sum_OdO_mbar_ptr):
        load_compute_sum_OdO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_sum_OdO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_sum_OdO_mbar_ptr,
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=load_compute_sum_OdO_producer_group,
            consumer_group=load_compute_sum_OdO_consumer_group,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
        )

    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr):
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
        )

    def make_and_init_mma_reduce_dQ_pipeline(self, mma_reduce_dQ_mbar_ptr):
        mma_reduce_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_reduce_dQ_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_reduce_dQ_mbar_ptr,
            num_stages=self.mma_reduce_dQ_stage,
            producer_group=mma_reduce_dQ_producer_group,
            consumer_group=mma_reduce_dQ_consumer_group,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr):
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.compute_mma_P_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr):
        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )

        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.compute_mma_dS_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
        )

    def make_and_init_mma_compute_dKdV_pipeline(self, mma_compute_dKdV_mbar_ptr):
        mma_compute_dKdV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dKdV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dKdV_mbar_ptr,
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=mma_compute_dKdV_producer_group,
            consumer_group=mma_compute_dKdV_consumer_group,
        )

    def make_and_init_reduce_tma_store_pipeline(self):
        reduce_tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * self.threads_per_warp,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.reduce_tma_store_stage,
            producer_group=reduce_tma_store_producer_group,
        )
