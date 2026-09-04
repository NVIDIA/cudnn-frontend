# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Callable, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Float32, Boolean

from . import blackwell_helpers as hstu_sm100_utils
from .block_info import BWDBlockInfo
from .block_sparsity import (
    HSTUBlockSparseTensors,
    get_q2k_block_sparse_consumer_row,
)
from .fast_math import FastSilU
from .mask import AttentionMask
from .seqlen_info import SeqlenInfo
from .tile_scheduler import (
    QMajorBwdScheduler,
    SingleTileBwdScheduler,
    TileSchedulerArguments,
)
from .utils import (
    cpasync_bulk_s2cluster,
    cpasync_reduce_bulk_add_f32,
    domain_offset_i64,
    make_compact_pipeline_state,
    make_tmem_copy,
    mul_packed_f32x2,
    split_wg,
    split_wg_contiguous,
    split_wg_mma,
)


class HSTUAttentionBackwardSm100:
    """Fused HSTU backward kernel for Blackwell SM100."""

    arch = 100

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int,
        tile_n: int,
        is_causal: bool = False,
        is_local: bool = False,
        is_arbitrary: bool = False,
        func_num: int = 0,
        use_auto_block_metadata: bool = False,
        use_2cta_instrs: bool = False,
        use_q_major_scheduler: bool = False,
        use_q1_small_mma: bool = False,
    ):
        self.element_dtype = element_dtype
        self.acc_dtype = Float32
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_hdim = head_dim
        self.use_2cta_instrs = use_2cta_instrs
        self.use_q_major_scheduler = use_q_major_scheduler
        self.use_q1_small_mma = use_q1_small_mma
        assert not self.use_q_major_scheduler or not self.use_2cta_instrs
        assert not self.use_q1_small_mma or self.use_q_major_scheduler
        # Only the first Q row is valid. Keep the main K @ Q^T family on the
        # M128N16 UMMA shape and use transposed M128N8 for K^T @ dS^T -> dQ^T.
        mma_tile_m = 16 if self.use_q1_small_mma else tile_m
        dq_mma_tile_n = 8 if self.use_q1_small_mma else head_dim
        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        self.cta_tiler = (
            tile_n,
            tile_m,
            self.tile_hdim,
        )
        # For S
        self.mma_tiler_kq = (
            self.cta_group_size * tile_n,
            mma_tile_m,
            head_dim,
        )
        # For dP
        self.mma_tiler_vdo = (
            self.cta_group_size * tile_n,
            mma_tile_m,
            head_dim,
        )
        # For dV
        self.mma_tiler_pdo = (
            self.cta_group_size * tile_n,
            head_dim,
            mma_tile_m,
        )
        # For dK
        self.mma_tiler_dsq = (
            self.cta_group_size * tile_n,
            head_dim,
            mma_tile_m,
        )
        # For dQ
        self.mma_tiler_dsk = (
            head_dim if self.use_q1_small_mma else (tile_n if self.use_q_major_scheduler else tile_m),
            dq_mma_tile_n,
            self.cta_group_size * tile_n,
        )
        self.dq_tile_m = self.mma_tiler_dsk[0]
        self.cluster_shape_mn = (self.cta_group_size, 1)
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_arbitrary = is_arbitrary
        self.func_num = func_num
        self.use_auto_block_metadata = use_auto_block_metadata
        self.use_deferred_ds_scale = self.use_2cta_instrs or (self.is_causal and not self.is_local and not self.is_arbitrary)
        assert not (self.is_arbitrary and (self.is_causal or self.is_local)), "Arbitrary masking cannot be combined with causal or local masking"
        assert not self.is_arbitrary or (self.func_num > 0 and self.func_num % 2 == 1), "Arbitrary masking requires a positive odd func_num"
        assert self.use_auto_block_metadata == self.is_arbitrary, "Block metadata must be enabled exactly for arbitrary masking"

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7) if self.use_q1_small_mma else (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 8 if self.use_q1_small_mma else 12
        self.load_warp_id = 9 if self.use_q1_small_mma else 13
        self.relay_warp_id = 10 if self.use_q1_small_mma else 14
        self.empty_warp_id = 11 if self.use_q1_small_mma else 15

        self.num_reduce_warps = 4
        self.num_compute_warps = len(self.compute_warp_ids)

        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=(self.num_reduce_warps + self.num_compute_warps + 1) * cute.arch.WARP_SIZE,
        )
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_compute_warps * cute.arch.WARP_SIZE,
        )
        self.epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.num_compute_warps * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.num_reduce_warps * cute.arch.WARP_SIZE,
        )
        if self.use_2cta_instrs:
            self.tmem_S_offset = 0
            self.tmem_dQ_offset = head_dim // 2
            self.tmem_dV_offset = tile_n
            self.tmem_dP_offset = self.tmem_dV_offset + head_dim
            self.tmem_dS_offset = self.tmem_dP_offset
            self.tmem_dK_offset = self.tmem_dP_offset + tile_m
        else:
            self.tmem_dK_offset = 0
            self.tmem_dV_offset = self.tmem_dK_offset + head_dim
            self.tmem_dQ_offset = self.tmem_dV_offset + head_dim
            self.tmem_dP_offset = self.tmem_dQ_offset
            self.tmem_dS_offset = self.tmem_dP_offset
            self.tmem_S_offset = self.tmem_dQ_offset + max(tile_m, head_dim)

        self.num_regs_reduce = 104 if self.use_2cta_instrs else 144
        self.num_regs_compute = 144
        self.num_regs_mma = 112 if self.use_2cta_instrs else 80
        self.num_regs_empty = 24
        self.num_regs_load = 104 if self.use_2cta_instrs else 56

        self.convert_block_seq = 16 if self.use_2cta_instrs else 8
        self.convert_num_threads_d = 16
        self.convert_num_threads_seq = 128 // self.convert_num_threads_d
        self.convert_elem_per_load = 8
        self.convert_tiles_per_cta = 128
        self.min_convert_ctas = 512
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 1 if self.use_2cta_instrs or self.use_q_major_scheduler else 2
        self.K_stage = 1
        self.dO_stage = 1
        self.single_stage = 1
        self.sdKVaccum_stage = 2
        self.dQ_reduce_ncol_t2r = 32
        self.dQ_reduce_ncol = 8 if self.use_2cta_instrs else 32
        self.sdQaccum_stage = 4 if self.use_2cta_instrs else 2
        assert (self.tile_hdim // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.cta_group_size // self.dQ_reduce_ncol
        self.qmajor_dq_ncol = self.tile_hdim if self.use_q1_small_mma else self.dQ_reduce_ncol_t2r
        self.qmajor_dq_stages = 1 if self.use_q1_small_mma else self.tile_hdim // self.qmajor_dq_ncol

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        func: Optional[cute.Tensor],
        alpha: Float32,
        scaling_seqlen: Float32,
        workspace: cute.Tensor,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        stream: cuda.CUstream,
    ):
        _, _, _, hb = problem_shape
        h, b = hb
        _, h_k = h

        def assume_strides_aligned(tensor: cute.Tensor) -> cute.Tensor:
            """Restore the 128-bit stride contract for gradient stores.

            Direct outputs are host-validated; scratch outputs are compact.
            Thus every dynamic non-unit stride in these rank-5 views is a
            multiple of one 128-bit vector.
            """
            divby = 128 // tensor.element_type.width
            strides = tuple(stride if isinstance(stride, int) else cute.assume(stride, divby=divby) for stride in tensor.stride[:-1])
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(tensor.shape, stride=(*strides, tensor.stride[-1])),
            )

        def make_q_like_tensor(tensor: cute.Tensor) -> cute.Tensor:
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(
                    (tensor.shape[0], tensor.shape[1], hb),
                    stride=(
                        tensor.stride[0],
                        tensor.stride[1],
                        (
                            (tensor.stride[2], tensor.stride[3]),
                            (0),
                        ),
                    ),
                ),
            )

        def make_kv_like_tensor(tensor: cute.Tensor) -> cute.Tensor:
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(
                    (tensor.shape[0], tensor.shape[1], hb),
                    stride=(
                        tensor.stride[0],
                        tensor.stride[1],
                        (
                            (0, tensor.stride[3]),
                            (0),
                        ),
                    ),
                ),
            )

        # (s, d, h_r, h_k, b) -> (s, d, ((h_r, h_k), b))
        Q = make_q_like_tensor(Q)
        # (s, d, 1, h_k, b) -> (s, d, ((1, h_k), b))
        K = make_kv_like_tensor(K)
        # (s, d, 1, h_k, b) -> (s, d, ((1, h_k), b))
        V = make_kv_like_tensor(V)

        dQ, dK, dV = [assume_strides_aligned(tensor) for tensor in (dQ, dK, dV)]
        dQ = make_q_like_tensor(dQ)
        dK = make_kv_like_tensor(dK)
        dV = make_kv_like_tensor(dV)
        dO = make_q_like_tensor(dO)

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.dQ_major_mode = utils.LayoutEnum.from_tensor(dQ).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()

        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.dQ_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dq is not supported")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        # compute S
        tiled_mma_S = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_kq[:2],
        )
        # compute dP
        tiled_mma_dP = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_vdo[:2],
        )
        # compute dV
        tiled_mma_dV = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_pdo[:2],
            tcgen05.OperandSource.TMEM,
        )
        # compute dK
        tiled_mma_dK = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsq[:2],
            (tcgen05.OperandSource.TMEM if self.use_2cta_instrs else tcgen05.OperandSource.SMEM),
        )
        # compute dQ
        tiled_mma_dQ = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsk[:2],
        )

        self.cluster_shape_mn = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mn),
            (tiled_mma_S.thr_id.shape,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_S,
            self.mma_tiler_kq,
            self.element_dtype,
            self.K_stage,
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_S,
            self.mma_tiler_kq,
            self.element_dtype,
            self.Q_stage,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_dP,
            self.mma_tiler_vdo,
            self.element_dtype,
            1,
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_dP,
            self.mma_tiler_vdo,
            self.element_dtype,
            self.dO_stage,
        )
        dS_smem_layout_staged = (
            sm100_utils.make_smem_layout_b(
                tiled_mma_dQ,
                self.mma_tiler_dsk,
                self.element_dtype,
                self.single_stage,
            )
            if self.use_q1_small_mma
            else sm100_utils.make_smem_layout_a(
                tiled_mma_dQ,
                self.mma_tiler_dsk,
                self.element_dtype,
                self.single_stage,
            )
        )
        KT_smem_layout_staged = (
            sm100_utils.make_smem_layout_a(
                tiled_mma_dQ,
                self.mma_tiler_dsk,
                self.element_dtype,
                self.K_stage,
            )
            if self.use_q1_small_mma
            else sm100_utils.make_smem_layout_b(
                tiled_mma_dQ,
                self.mma_tiler_dsk,
                self.element_dtype,
                self.K_stage,
            )
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_dK,
            self.mma_tiler_dsq,
            self.element_dtype,
            self.single_stage,
        )
        dS_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_dK,
            self.mma_tiler_dsq,
            self.element_dtype,
            self.single_stage,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_dK,
            self.mma_tiler_dsq,
            self.element_dtype,
            self.Q_stage,
        )
        P_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_dV,
            self.mma_tiler_pdo,
            self.element_dtype,
            self.single_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_dV,
            self.mma_tiler_pdo,
            self.element_dtype,
            self.dO_stage,
        )

        dQ_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                (self.dq_tile_m, 32),
            ),
            self.acc_dtype,
        )
        dQ_tma_smem_layout_staged = cute.tile_to_shape(
            dQ_smem_layout_atom,
            (self.dq_tile_m, 32, 2),
            order=(1, 0, 2),
        )
        dQ_smem_layout_staged = (
            cute.make_composed_layout(
                cute.make_swizzle(0, 0, 0),
                0,
                cute.make_layout(
                    (
                        self.dq_tile_m * self.dQ_reduce_ncol,
                        self.sdQaccum_stage,
                    )
                ),
            )
            if self.use_2cta_instrs
            else dQ_tma_smem_layout_staged
        )
        self.dKV_epi_tile = (
            self.tile_n,
            self.tile_hdim // (self.num_compute_warps // 4),
        )
        dK_smem_layout_epi = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.dKV_epi_tile,
            self.num_compute_warps // 4,
        )
        dV_smem_layout_epi = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.dKV_epi_tile,
            self.num_compute_warps // 4,
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp()
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.mma_tiler_kq,
            tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.mma_tiler_vdo,
            tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.mma_tiler_kq,
            tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.mma_tiler_vdo,
            tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_Qt = tma_tensor_Qt = None
        tma_atom_Kt = tma_tensor_Kt = None
        tma_atom_dOt = tma_tensor_dOt = None
        if cutlass.const_expr(self.use_2cta_instrs):
            Qt = cute.make_tensor(
                Q.iterator,
                cute.select(Q.layout, mode=[1, 0, 2]),
            )
            Kt = cute.make_tensor(
                K.iterator,
                cute.select(K.layout, mode=[1, 0, 2]),
            )
            dOt = cute.make_tensor(
                dO.iterator,
                cute.select(dO.layout, mode=[1, 0, 2]),
            )
            Qt_smem_layout = cute.select(
                QT_smem_layout_staged,
                mode=[0, 1, 2],
            )
            Kt_smem_layout = cute.select(
                KT_smem_layout_staged,
                mode=[0, 1, 2],
            )
            dOt_smem_layout = cute.select(
                dOT_smem_layout_staged,
                mode=[0, 1, 2],
            )
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                Qt,
                Qt_smem_layout,
                self.mma_tiler_dsq,
                tiled_mma_dK,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                Kt,
                Kt_smem_layout,
                self.mma_tiler_dsk,
                tiled_mma_dQ,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                dOt,
                dOt_smem_layout,
                self.mma_tiler_pdo,
                tiled_mma_dV,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dK,
            cute.select(dK_smem_layout_epi, mode=[0, 1]),
            self.dKV_epi_tile,
            1,
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dV,
            cute.select(dV_smem_layout_epi, mode=[0, 1]),
            self.dKV_epi_tile,
            1,
        )
        dKV_r2s_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        dKV_r2s_copy = cute.make_tiled_copy_tv(
            dKV_r2s_atom,
            cute.make_ordered_layout((128, 1), order=(1, 0)),
            cute.make_ordered_layout(
                (1, 128 // self.element_dtype.width),
                order=(1, 0),
            ),
        )

        self.tma_copy_Q_bytes = self.cta_group_size * cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_K_bytes = self.cta_group_size * cute.size_in_bytes(self.element_dtype, K_smem_layout)
        self.tma_copy_V_bytes = self.cta_group_size * cute.size_in_bytes(self.element_dtype, V_smem_layout)
        self.tma_copy_dO_bytes = self.cta_group_size * cute.size_in_bytes(self.element_dtype, dO_smem_layout)
        self.tma_copy_Qt_bytes = (
            self.cta_group_size
            * cute.size_in_bytes(
                self.element_dtype,
                cute.select(QT_smem_layout_staged, mode=[0, 1, 2]),
            )
            if self.use_2cta_instrs
            else 0
        )
        self.tma_copy_Kt_bytes = (
            self.cta_group_size
            * cute.size_in_bytes(
                self.element_dtype,
                cute.select(KT_smem_layout_staged, mode=[0, 1, 2]),
            )
            if self.use_2cta_instrs
            else 0
        )
        self.tma_copy_dOt_bytes = (
            self.cta_group_size
            * cute.size_in_bytes(
                self.element_dtype,
                cute.select(dOT_smem_layout_staged, mode=[0, 1, 2]),
            )
            if self.use_2cta_instrs
            else 0
        )

        load_mma_Qt_mbar_size = self.Q_stage * 2 if self.use_2cta_instrs else 0
        load_mma_Kt_mbar_size = self.single_stage * 2 if self.use_2cta_instrs or self.use_q_major_scheduler else 0
        Qt_smem_size = cute.cosize(QT_smem_layout_staged) if self.use_2cta_instrs else 0
        Kt_smem_size = cute.cosize(KT_smem_layout_staged) if self.use_2cta_instrs else 0
        dOt_smem_size = cute.cosize(dOT_smem_layout_staged) if self.use_2cta_instrs else 0
        Q_smem_size = (
            cute.cosize(Q_smem_layout_staged)
            if self.use_2cta_instrs
            else max(
                cute.cosize(Q_smem_layout_staged),
                cute.cosize(dK_smem_layout_epi),
            )
        )
        dO_smem_size = (
            cute.cosize(dO_smem_layout_staged)
            if self.use_2cta_instrs
            else max(
                cute.cosize(dO_smem_layout_staged),
                cute.cosize(dV_smem_layout_epi),
            )
        )
        dS_xchg_size = self.tile_n * self.tile_m // 2 if self.use_2cta_instrs else 0
        dST_smem_size = cute.cosize(dST_smem_layout_staged) if self.use_q1_small_mma else 0
        dQ_smem_size = cute.cosize(dQ_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            load_mma_Qt_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64,
                load_mma_Qt_mbar_size,
            ]
            load_mma_Kt_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64,
                load_mma_Kt_mbar_size,
            ]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.single_stage * 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.single_stage * 2]
            mma_reduce_dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.single_stage * 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.single_stage * 2]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.single_stage * 2]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.sdKVaccum_stage * 2]
            dS_cluster_full_mbar_ptr: cutlass.Int64
            dS_cluster_leader_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar: cutlass.Int64
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
                cute.struct.MemRange[self.element_dtype, Q_smem_size],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, dO_smem_size],
                self.buffer_align_bytes,
            ]
            sQt: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, Qt_smem_size],
                self.buffer_align_bytes,
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, Kt_smem_size],
                self.buffer_align_bytes,
            ]
            sdOt: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, dOt_smem_size],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(dS_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdST: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, dST_smem_size],
                self.buffer_align_bytes,
            ]
            sdS_xchg: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, dS_xchg_size],
                self.buffer_align_bytes,
            ]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, dQ_smem_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        dQ_acc = self.get_workspace_tensor(problem_shape, workspace)

        dQ_smem_layout = cute.select(
            dQ_tma_smem_layout_staged,
            mode=[0, 1],
        )

        tma_atom_dQ_acc, tma_tensor_dQ_acc = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_reduce_op,
            dQ_acc,
            dQ_smem_layout,
            (self.dq_tile_m, 32),
        )

        TileScheduler = QMajorBwdScheduler if self.use_q_major_scheduler else SingleTileBwdScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(problem_shape[1], self.cta_tiler[0]),
            cute.size(h_k),
            cute.size(b),
            problem_shape[1],
            problem_shape[2],
            problem_shape[2],
            total_q=cute.size(K.shape[0]),
            tile_shape_mn=(self.tile_n, self.tile_m),
            cu_seqlens_q=cu_seqlens_k,
            cu_seqlens_k=cu_seqlens_q,
            element_size=self.element_dtype.width // 8,
            cluster_shape_mn=self.cluster_shape_mn[:2],
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        bwd_grid = TileScheduler.get_grid_shape(tile_sched_params)

        self.bwd(
            tiled_mma_S,
            tiled_mma_dP,
            tiled_mma_dV,
            tiled_mma_dK,
            tiled_mma_dQ,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_Qt,
            tma_tensor_Qt,
            tma_atom_Kt,
            tma_tensor_Kt,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dOt,
            tma_tensor_dOt,
            tma_atom_dQ_acc,
            tma_tensor_dQ_acc,
            dQ_acc,
            dQ,
            tma_atom_dK,
            tma_tensor_dK,
            tma_atom_dV,
            tma_tensor_dV,
            dKV_r2s_copy,
            dK,
            dV,
            problem_shape,
            cu_seqlens_q,
            cu_seqlens_k,
            window_size_left,
            window_size_right,
            func,
            block_sparse_tensors,
            alpha,
            scaling_seqlen,
            K_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dS_smem_layout_staged,
            KT_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            dOT_smem_layout_staged,
            dQ_smem_layout_staged,
            P_tmem_layout_staged,
            dS_tmem_layout_staged,
            dK_smem_layout_epi,
            dV_smem_layout_epi,
            tile_sched_params,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[*self.cluster_shape_mn[:2], 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        if cutlass.const_expr(self.use_q_major_scheduler):
            return

        # Convert the FP32 dQ workspace to the output dtype.
        dQ_scale = alpha / scaling_seqlen if self.use_deferred_ds_scale else Float32(1.0)
        convert_num_seq_tiles = cute.ceil_div(
            cute.size(Q.shape[0]),
            cute.size(problem_shape[3][1]) * self.convert_block_seq,
        )
        convert_grid_z = cute.ceil_div(
            convert_num_seq_tiles,
            self.convert_tiles_per_cta,
        )
        min_convert_grid_z = cute.ceil_div(
            self.min_convert_ctas,
            cute.size(problem_shape[3][0]) * cute.size(problem_shape[3][1]),
        )
        convert_grid_z = cutlass.min(
            convert_num_seq_tiles,
            cutlass.max(convert_grid_z, min_convert_grid_z),
        )
        self.convert(
            dQ_acc,
            dQ,
            problem_shape[2],
            cu_seqlens_q,
            dQ_scale,
        ).launch(
            grid=[
                cute.size(problem_shape[3][0]),
                cute.size(problem_shape[3][1]),
                convert_grid_z,
            ],
            block=[
                self.convert_num_threads_d,
                self.convert_num_threads_seq,
                1,
            ],
            cluster=[1, 1, 1],
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def bwd(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_Qt: Optional[cute.CopyAtom],
        Qt_in: Optional[cute.Tensor],
        tma_atom_Kt: Optional[cute.CopyAtom],
        Kt_in: Optional[cute.Tensor],
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dOt: Optional[cute.CopyAtom],
        dOt_in: Optional[cute.Tensor],
        tma_atom_dQ_acc: cute.CopyAtom,
        tma_tensor_dQ_acc: cute.Tensor,
        dQ_acc: cute.Tensor,
        dQ: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        tma_tensor_dK: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        tma_tensor_dV: cute.Tensor,
        dKV_r2s_copy: cute.TiledCopy,
        dK: cute.Tensor,
        dV: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: Union[cute.Tensor, None],
        cu_seqlens_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        func: Optional[cute.Tensor],
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        alpha: Float32,
        scaling_seqlen: Float32,
        K_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        KT_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        P_tmem_layout_staged: cute.ComposedLayout,
        dS_tmem_layout_staged: cute.ComposedLayout,
        dK_smem_layout_epi: cute.ComposedLayout,
        dV_smem_layout_epi: cute.ComposedLayout,
        tile_sched_params: SingleTileBwdScheduler.Params,
    ):
        if cutlass.const_expr(self.use_2cta_instrs and self.is_causal and not self.is_local and not self.is_arbitrary):
            assert cu_seqlens_q is not None and cu_seqlens_k is not None
            block_idx = cute.arch.block_idx()
            n_block = block_idx[0] // self.cta_group_size
            batch_idx = block_idx[2]
            seqlen_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
            seqlen_k = cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx]
            if n_block * self.tile_n * self.cta_group_size >= seqlen_k or seqlen_q == 0:
                cute.arch.nvvm.exit()

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_Q)
                if cutlass.const_expr(self.use_2cta_instrs):
                    assert tma_atom_Qt is not None
                    assert tma_atom_Kt is not None
                    assert tma_atom_dOt is not None
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                    cpasync.prefetch_descriptor(tma_atom_dOt)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                cpasync.prefetch_descriptor(tma_atom_dK)
                cpasync.prefetch_descriptor(tma_atom_dV)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        if cutlass.const_expr(self.use_q1_small_mma):
            sdST_raw = storage.sdST.get_tensor(cute.make_layout((cute.cosize(dST_smem_layout_staged),)))
            for value_idx in cutlass.range(tidx, cute.size(sdST_raw), self.threads_per_cta, unroll=1):
                sdST_raw[value_idx] = self.element_dtype(0.0)
            self.cta_sync_barrier.arrive_and_wait()
        dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr.ptr
        dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr.ptr
        if cutlass.const_expr(self.use_2cta_instrs):
            if warp_idx == self.compute_warp_ids[0]:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mn),
            (tiled_mma_S.thr_id.shape,),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        TileSchedulerCls = partial(
            self.tile_scheduler_cls.create,
            tile_sched_params,
        )

        load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(storage.load_mma_Q_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        load_mma_Qt_pipeline = load_mma_Q_pipeline
        load_mma_Kt_pipeline = load_mma_Q_pipeline
        if cutlass.const_expr(self.use_2cta_instrs):
            load_mma_Qt_pipeline = self.make_and_init_load_mma_Qt_pipeline(
                storage.load_mma_Qt_mbar_ptr.data_ptr(),
                cluster_layout_vmnk,
            )
        if cutlass.const_expr(self.use_2cta_instrs or self.use_q_major_scheduler):
            load_mma_Kt_pipeline = self.make_and_init_load_mma_Kt_pipeline(
                storage.load_mma_Kt_mbar_ptr.data_ptr(),
                cluster_layout_vmnk,
            )
        load_mma_dO_pipeline = self.make_and_init_load_mma_dO_pipeline(storage.load_mma_dO_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(storage.mma_compute_S_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(storage.mma_compute_dP_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        mma_reduce_dQ_pipeline = self.make_and_init_mma_reduce_dQ_pipeline(storage.mma_reduce_dQ_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(storage.compute_mma_P_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(storage.compute_mma_dS_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        mma_compute_dKdV_pipeline = self.make_and_init_mma_compute_dKdV_pipeline(storage.mma_compute_dKdV_mbar_ptr.data_ptr(), cluster_layout_vmnk)
        reduce_tma_store_pipeline = self.make_and_init_reduce_tma_store_pipeline()

        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk,
            is_relaxed=False,
        )
        self.cta_sync_barrier.arrive_and_wait()

        # setup mma
        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        if cutlass.const_expr(self.use_2cta_instrs):
            sdK_epi = storage.sK.get_tensor(
                dK_smem_layout_epi.outer,
                swizzle=dK_smem_layout_epi.inner,
            )
            sdV_epi = storage.sV.get_tensor(
                dV_smem_layout_epi.outer,
                swizzle=dV_smem_layout_epi.inner,
            )
        else:
            sdK_epi = storage.sQ.get_tensor(
                dK_smem_layout_epi.outer,
                swizzle=dK_smem_layout_epi.inner,
            )
            sdV_epi = storage.sdO.get_tensor(
                dV_smem_layout_epi.outer,
                swizzle=dV_smem_layout_epi.inner,
            )
        sdQ = storage.sdQ.get_tensor(
            dQ_smem_layout_staged.outer,
            swizzle=dQ_smem_layout_staged.inner,
        )
        if cutlass.const_expr(self.use_2cta_instrs):
            sQT = storage.sQt.get_tensor(
                QT_smem_layout_staged.outer,
                swizzle=QT_smem_layout_staged.inner,
            )
            sKT = storage.sKt.get_tensor(
                KT_smem_layout_staged.outer,
                swizzle=KT_smem_layout_staged.inner,
            )
        else:
            sQT_ptr = cute.recast_ptr(
                sQ.iterator,
                QT_smem_layout_staged.inner,
            )
            sQT = cute.make_tensor(
                sQT_ptr,
                QT_smem_layout_staged.outer,
            )
            sKT_ptr = cute.recast_ptr(
                sK.iterator,
                KT_smem_layout_staged.inner,
            )
            sKT = cute.make_tensor(
                sKT_ptr,
                KT_smem_layout_staged.outer,
            )
        sdS = storage.sdS.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
        if cutlass.const_expr(self.use_q1_small_mma):
            sdST = storage.sdST.get_tensor(dST_smem_layout_staged.outer, swizzle=dST_smem_layout_staged.inner)
        else:
            sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
            sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)
        sdS_xchg = None
        if cutlass.const_expr(self.use_2cta_instrs):
            sdS_xchg = storage.sdS_xchg.get_tensor(cute.make_layout((self.tile_n, self.tile_m // 2)))
        if cutlass.const_expr(self.use_2cta_instrs):
            sdOT = storage.sdOt.get_tensor(
                dOT_smem_layout_staged.outer,
                swizzle=dOT_smem_layout_staged.inner,
            )
        else:
            sdOT_ptr = cute.recast_ptr(
                sdO.iterator,
                dOT_smem_layout_staged.inner,
            )
            sdOT = cute.make_tensor(
                sdOT_ptr,
                dOT_smem_layout_staged.outer,
            )

        mma_tile_coord_v = cute.arch.block_idx()[0] % self.cta_group_size

        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)

        tSTtST_shape = thr_mma_S.partition_shape_C(cute.select(self.mma_tiler_kq, mode=[0, 1]))
        tSTtST = thr_mma_S.make_fragment_C(tSTtST_shape)
        tSTtST = cute.make_tensor(
            tSTtST.iterator + self.tmem_S_offset,
            tSTtST.layout,
        )

        tdPTtdPT_shape = thr_mma_dP.partition_shape_C(cute.select(self.mma_tiler_vdo, mode=[0, 1]))
        tdPTtdPT = thr_mma_dP.make_fragment_C(tdPTtdPT_shape)
        tdPTtdPT = cute.make_tensor(
            tdPTtdPT.iterator + self.tmem_dP_offset,
            tdPTtdPT.layout,
        )
        tdS = cute.make_tensor(
            cute.recast_ptr(
                tdPTtdPT.iterator,
                dtype=self.element_dtype,
            ),
            cute.slice_(
                dS_tmem_layout_staged,
                (None, None, None, 0),
            ).outer,
        )

        tP = cute.make_tensor(
            cute.make_ptr(
                self.element_dtype,
                0,
                cute.AddressSpace.tmem,
            ),
            P_tmem_layout_staged.outer,
        )
        tdVrP = tiled_mma_dV.make_fragment_A(tP)
        tdVrP = tdVrP[None, None, None, 0]
        tdVrP = cute.make_tensor(
            cute.recast_ptr(
                tSTtST.iterator,
                dtype=self.element_dtype,
            ),
            tdVrP.layout,
        )

        tdQtdQ_shape = thr_mma_dQ.partition_shape_C(cute.select(self.mma_tiler_dsk, mode=[0, 1]))
        tdQtdQ = thr_mma_dQ.make_fragment_C(tdQtdQ_shape)
        tdQtdQ = cute.make_tensor(
            tdQtdQ.iterator + self.tmem_dQ_offset,
            tdQtdQ.layout,
        )

        tdKtdK_shape = thr_mma_dK.partition_shape_C(cute.select(self.mma_tiler_dsq, mode=[0, 1]))
        tdKtdK = thr_mma_dK.make_fragment_C(tdKtdK_shape)
        tdKtdK = cute.make_tensor(
            tdKtdK.iterator + self.tmem_dK_offset,
            tdKtdK.layout,
        )

        tdVtdV_shape = thr_mma_dV.partition_shape_C(cute.select(self.mma_tiler_pdo, mode=[0, 1]))
        tdVtdV = thr_mma_dV.make_fragment_C(tdVtdV_shape)
        tdVtdV = cute.make_tensor(
            tdVtdV.iterator + self.tmem_dV_offset,
            tdVtdV.layout,
        )

        block_info = BWDBlockInfo(
            (
                self.tile_m,
                self.tile_n * self.cta_group_size,
                self.cta_tiler[2],
            ),
            self.is_causal,
            self.is_local,
            window_size_left,
            window_size_right,
        )
        SeqlenInfoCls = partial(
            SeqlenInfo,
            max_seqlen_q=Float32(problem_shape[0]),
            max_seqlen_k=Float32(problem_shape[1]),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            tile_m=self.tile_m,
        )
        pipeline.pipeline_init_wait(
            cluster_shape_mn=cluster_layout_vmnk,
        )

        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            self.load_persistent(
                K_in,
                V_in,
                Q_in,
                Qt_in,
                Kt_in,
                dO_in,
                dOt_in,
                sK,
                sQ,
                sKT,
                sQT,
                sV,
                sdO,
                sdOT,
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dQ,
                tiled_mma_dK,
                tiled_mma_dV,
                tma_atom_K,
                tma_atom_Q,
                tma_atom_Kt,
                tma_atom_Qt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dOt,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                (
                    load_mma_Q_pipeline,
                    load_mma_Qt_pipeline,
                    load_mma_Kt_pipeline,
                    load_mma_dO_pipeline,
                ),
                block_info,
                block_sparse_tensors,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        elif warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
            tmem.wait_for_alloc()
            tSTtST_compute = tSTtST
            tdPTtdPT_compute = tdPTtdPT
            tdVrP_compute = tdVrP
            tdKtdK_compute = tdKtdK
            tdVtdV_compute = tdVtdV
            if cutlass.const_expr(self.use_q1_small_mma):
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tSTtST_compute = cute.make_tensor(
                    tmem_ptr + self.tmem_S_offset,
                    tSTtST.layout,
                )
                tdPTtdPT_compute = cute.make_tensor(
                    tmem_ptr + self.tmem_dP_offset,
                    tdPTtdPT.layout,
                )
                tdVrP_compute = cute.make_tensor(
                    cute.recast_ptr(tSTtST_compute.iterator, dtype=self.element_dtype),
                    tdVrP.layout,
                )
                tdKtdK_compute = cute.make_tensor(
                    tmem_ptr + self.tmem_dK_offset,
                    tdKtdK.layout,
                )
                tdVtdV_compute = cute.make_tensor(
                    tmem_ptr + self.tmem_dV_offset,
                    tdVtdV.layout,
                )
            self.compute_persistent(
                tSTtST_compute,
                tdPTtdPT_compute,
                tdVrP_compute,
                sdS,
                sdST,
                sdS_xchg,
                dS_cluster_full_mbar_ptr,
                dK,
                dV,
                tma_atom_dK,
                tma_tensor_dK,
                tma_atom_dV,
                tma_tensor_dV,
                dKV_r2s_copy,
                sdK_epi,
                sdV_epi,
                tdKtdK_compute,
                tdVtdV_compute,
                thr_mma_S,
                thr_mma_dV,
                thr_mma_dK,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                scaling_seqlen,
                alpha,
                window_size_left,
                window_size_right,
                func,
                (
                    mma_compute_S_pipeline,
                    compute_mma_P_pipeline,
                    mma_compute_dP_pipeline,
                    compute_mma_dS_pipeline,
                    mma_compute_dKdV_pipeline,
                ),
                block_info,
                block_sparse_tensors,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            self.tmem_alloc_barrier.arrive()
        elif warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            if cutlass.const_expr(self.num_regs_reduce < 128):
                cute.arch.warpgroup_reg_dealloc(self.num_regs_reduce)
            else:
                cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tdQtdQ_reduce = tdQtdQ
            if cutlass.const_expr(self.use_q1_small_mma):
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tdQtdQ_reduce = cute.make_tensor(
                    tmem_ptr + self.tmem_dQ_offset,
                    tdQtdQ.layout,
                )
            self.reduce_persistent(
                thr_mma_dQ,
                tdQtdQ_reduce,
                tma_atom_dQ_acc,
                tma_tensor_dQ_acc,
                dQ_acc,
                dQ,
                sdQ,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                alpha,
                scaling_seqlen,
                (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline),
                block_info,
                block_sparse_tensors,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            self.tmem_alloc_barrier.arrive()
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tSTtST_mma = tSTtST
            tdPTtdPT_mma = tdPTtdPT
            tdS_mma = tdS
            tdVtdV_mma = tdVtdV
            tdVrP_mma = tdVrP
            tdQtdQ_mma = tdQtdQ
            tdKtdK_mma = tdKtdK
            if cutlass.const_expr(self.use_q1_small_mma):
                tSTtST_mma = cute.make_tensor(
                    tmem_ptr + self.tmem_S_offset,
                    tSTtST.layout,
                )
                tdPTtdPT_mma = cute.make_tensor(
                    tmem_ptr + self.tmem_dP_offset,
                    tdPTtdPT.layout,
                )
                tdS_mma = cute.make_tensor(
                    cute.recast_ptr(tdPTtdPT_mma.iterator, dtype=self.element_dtype),
                    tdS.layout,
                )
                tdVtdV_mma = cute.make_tensor(
                    tmem_ptr + self.tmem_dV_offset,
                    tdVtdV.layout,
                )
                tdVrP_mma = cute.make_tensor(
                    cute.recast_ptr(tSTtST_mma.iterator, dtype=self.element_dtype),
                    tdVrP.layout,
                )
                tdQtdQ_mma = cute.make_tensor(
                    tmem_ptr + self.tmem_dQ_offset,
                    tdQtdQ.layout,
                )
                tdKtdK_mma = cute.make_tensor(
                    tmem_ptr + self.tmem_dK_offset,
                    tdKtdK.layout,
                )
            self.mma_persistent(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dQ,
                tiled_mma_dK,
                tSTtST_mma,
                tdPTtdPT_mma,
                tdS_mma,
                tdVtdV_mma,
                tdVrP_mma,
                tdQtdQ_mma,
                tdKtdK_mma,
                sK,
                sQ,
                sV,
                sdO,
                sdS,
                sKT,
                sdST,
                sQT,
                sdOT,
                dS_cluster_leader_mbar_ptr,
                (
                    load_mma_Q_pipeline,
                    load_mma_Qt_pipeline,
                    load_mma_Kt_pipeline,
                    mma_compute_S_pipeline,
                    load_mma_dO_pipeline,
                    mma_compute_dP_pipeline,
                    mma_reduce_dQ_pipeline,
                    compute_mma_P_pipeline,
                    compute_mma_dS_pipeline,
                    mma_compute_dKdV_pipeline,
                ),
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                block_info,
                block_sparse_tensors,
                SeqlenInfoCls,
                TileSchedulerCls,
                mma_tile_coord_v == 0,
            )
            tmem.relinquish_alloc_permit()
            self.tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        elif warp_idx == self.relay_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)
            if cutlass.const_expr(self.use_2cta_instrs):
                self.relay_persistent(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    problem_shape,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
        else:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
            # Keep the fourth producer warp alive for the persistent schedule.
            self.empty_warp(TileSchedulerCls)

    @cute.jit
    def relay_persistent(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        dS_cluster_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord, _, _, _, process_tile = self.get_work_context(
                work_tile,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                block_sparse_tensors,
            )
            _, n_block, _, blk_coord_batch = blk_coord
            _, blk_coord_b = blk_coord_batch
            seqlen_obj = SeqlenInfoCls(blk_coord_b)
            if cutlass.const_expr(self.use_auto_block_metadata):
                block_iter_count, _, _, _, _ = get_q2k_block_sparse_consumer_row(
                    block_sparse_tensors,
                    blk_coord_b,
                    n_block,
                )
            else:
                m_block_min, m_block_max, _ = block_info.get_m_block_info(
                    seqlen_obj,
                    n_block // self.cta_group_size,
                )
                block_iter_count = m_block_max - m_block_min
            if process_tile:
                for _ in cutlass.range(block_iter_count, unroll=1):
                    cute.arch.mbarrier_wait(
                        dS_cluster_full_mbar_ptr,
                        phase=dS_cluster_phase,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(
                            dS_cluster_leader_mbar_ptr,
                            Int32(0),
                        )
                    dS_cluster_phase ^= 1
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def empty_warp(self, TileSchedulerCls: Callable):
        """Track the persistent schedule without participating in the data path."""
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.kernel
    def convert(
        self,
        dQ_acc: cute.Tensor,
        dQ: cute.Tensor,
        d_dim: Int32,
        cu_seqlens_q: Union[cute.Tensor, None],
        dQ_scale: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        offset = cu_seqlens_q[bidy]
        seqlen = cu_seqlens_q[bidy + 1] - offset
        workspace_offset = offset
        if cutlass.const_expr(self.use_2cta_instrs):
            workspace_offset = cute.assume(
                (offset + bidy * self.tile_m) // self.tile_m * self.tile_m,
                divby=self.tile_m,
            )

        _, _, grid_dim_z = cute.arch.grid_dim()
        seq_tile = bidz
        num_seq_tiles = cute.ceil_div(seqlen, self.convert_block_seq)
        seq_thread = tidy
        num_seq_threads = self.convert_num_threads_seq
        dim_thread = tidx
        num_dim_threads = self.convert_num_threads_d
        if cutlass.const_expr(self.use_2cta_instrs):
            seq_thread = tidx
            num_seq_threads = self.convert_num_threads_d
            dim_thread = tidy
            num_dim_threads = self.convert_num_threads_seq
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.acc_dtype,
            num_bits_per_copy=128,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        while seq_tile < num_seq_tiles:
            for idx_s_t in cutlass.range(
                seq_thread,
                self.convert_block_seq,
                num_seq_threads,
            ):
                idx_s = idx_s_t + self.convert_block_seq * seq_tile
                if idx_s < seqlen:
                    head_coord = cute.idx2crd(bidx, dQ_acc.shape[2])
                    dQ_item = domain_offset_i64(
                        (offset + idx_s, 0, (head_coord, bidy)),
                        dQ,
                    )
                    dQ_bhs = dQ_item[0, None, ((0, 0), 0)]
                    dQ_bhs = cute.logical_divide(
                        dQ_bhs,
                        cute.make_layout(self.convert_elem_per_load),
                    )

                    for idx_d in cutlass.range(
                        dim_thread,
                        d_dim // self.convert_elem_per_load,
                        num_dim_threads,
                    ):
                        if cutlass.const_expr(self.use_2cta_instrs):
                            head_linear = cutlass.Int64(head_coord[0]) + cutlass.Int64(head_coord[1]) * cutlass.Int64(cute.size(dQ_acc.shape[2][0]))
                            head_stride = cutlass.Int64(cute.size(dQ_acc.shape[0])) * cutlass.Int64(self.tile_hdim)
                            tile_idx = idx_s // self.tile_m
                            row_idx = idx_s % self.tile_m
                            row_group = row_idx // (self.tile_m // 2)
                            row_in_group = row_idx % (self.tile_m // 2)
                            dim_group = (idx_d % 8) * 4 + idx_d // 8
                            tile_offset = cutlass.Int64(workspace_offset + tile_idx * self.tile_m) * cutlass.Int64(self.tile_hdim)
                            dQ_frg = cute.make_rmem_tensor(
                                cute.make_layout((4, 2), stride=(1, 4)),
                                self.element_dtype,
                            )
                            for vec_idx in cutlass.range_constexpr(2):
                                physical_offset = (
                                    tile_offset
                                    + cutlass.Int64(row_group) * cutlass.Int64(self.tile_m // 2 * self.tile_hdim)
                                    + cutlass.Int64(dim_group + vec_idx * 2) * cutlass.Int64(self.tile_m // 2 * 4)
                                    + cutlass.Int64(row_in_group * 4)
                                    + head_linear * head_stride
                                )
                                dQ_acc_ptr = dQ_acc.iterator + physical_offset
                                dQ_acc_vec = cute.make_tensor(
                                    cute.make_ptr(
                                        self.acc_dtype,
                                        dQ_acc_ptr.toint(),
                                        dQ_acc_ptr.memspace,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout(4),
                                )
                                dQ_acc_frg = cute.make_fragment_like(dQ_acc_vec)
                                cute.copy(load_atom, dQ_acc_vec, dQ_acc_frg)
                                for i in cutlass.range_constexpr(0, 4, 2):
                                    dQ_acc_frg[i], dQ_acc_frg[i + 1] = mul_packed_f32x2(
                                        (
                                            dQ_acc_frg[i],
                                            dQ_acc_frg[i + 1],
                                        ),
                                        (dQ_scale, dQ_scale),
                                    )
                                dQ_frg[None, vec_idx].store(dQ_acc_frg.load().to(self.element_dtype))
                            dQ_frg = cute.make_tensor(
                                dQ_frg.iterator,
                                cute.make_layout(self.convert_elem_per_load),
                            )
                            dQ_out = dQ_bhs[None, idx_d]
                            cute.copy(
                                store_atom,
                                dQ_frg,
                                dQ_out,
                            )
                        else:
                            dQ_acc_item = domain_offset_i64(
                                (workspace_offset + idx_s, 0, head_coord),
                                dQ_acc,
                            )
                            dQ_acc_bhs = dQ_acc_item[0, None, (0, 0)]
                            dQ_acc_bhs = cute.logical_divide(
                                dQ_acc_bhs,
                                cute.make_layout(self.convert_elem_per_load),
                            )
                            dQ_acc_vec = dQ_acc_bhs[None, idx_d]
                            dQ_acc_vec = cute.make_tensor(
                                cute.make_ptr(
                                    self.acc_dtype,
                                    dQ_acc_vec.iterator.toint(),
                                    dQ_acc_vec.memspace,
                                    assumed_align=16,
                                ),
                                dQ_acc_vec.layout,
                            )
                            dQ_acc_frg = cute.make_fragment_like(dQ_acc_vec)
                            cute.copy(load_atom, dQ_acc_vec, dQ_acc_frg)
                            for i in cutlass.range_constexpr(
                                0,
                                cute.size(dQ_acc_frg),
                                2,
                            ):
                                dQ_acc_frg[i], dQ_acc_frg[i + 1] = mul_packed_f32x2(
                                    (
                                        dQ_acc_frg[i],
                                        dQ_acc_frg[i + 1],
                                    ),
                                    (dQ_scale, dQ_scale),
                                )
                            dQ_frg = cute.make_rmem_tensor(
                                dQ_acc_frg.shape,
                                self.element_dtype,
                            )
                            dQ_frg.store(dQ_acc_frg.load().to(self.element_dtype))
                            dQ_out = dQ_bhs[None, idx_d]
                            cute.copy(
                                store_atom,
                                dQ_frg,
                                dQ_out,
                            )
            seq_tile += grid_dim_z

    @cute.jit
    def get_block_sparse_m_block(
        self,
        mask_block_cnt: Int32,
        mask_block_idx: cute.Tensor,
        full_block_cnt: Int32,
        full_block_idx: cute.Tensor,
        iteration: Int32,
    ) -> Int32:
        """Map one ascending MASK-then-FULL K2Q iteration to a Q block."""

        m_block = Int32(0)
        if iteration < mask_block_cnt:
            m_block = mask_block_idx[iteration]
        elif iteration < mask_block_cnt + full_block_cnt:
            m_block = full_block_idx[iteration - mask_block_cnt]
        return m_block

    @cute.jit
    def get_work_context(
        self,
        work_tile,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
    ):
        n_block, head_idx, batch_idx = work_tile.tile_idx
        q_offset = cu_seqlens_q[batch_idx]
        k_offset = cu_seqlens_k[batch_idx]
        q_length = cu_seqlens_q[batch_idx + 1] - q_offset
        k_length = cu_seqlens_k[batch_idx + 1] - k_offset
        blk_coord = (
            Int32(0),
            n_block,
            Int32(0),
            ((Int32(0), head_idx), batch_idx),
        )
        blk_offset = (
            q_offset,
            k_offset,
            Int32(0),
            ((Int32(0), Int32(0)), Int32(0)),
        )
        problem_shape_cur_batch = (
            q_length,
            k_length,
            problem_shape[2],
            problem_shape[3],
        )
        metadata_empty = Boolean(False)
        if cutlass.const_expr(self.use_auto_block_metadata):
            consumer_count, _, _, _, _ = get_q2k_block_sparse_consumer_row(
                block_sparse_tensors,
                batch_idx,
                n_block,
            )
            metadata_empty = Boolean(consumer_count == 0)
        k_block_cta_group = n_block // self.cta_group_size
        has_k_tile = Boolean(k_block_cta_group * self.tile_n * self.cta_group_size < k_length)
        process_tile = Boolean(has_k_tile and q_length > 0 and not metadata_empty)
        return (
            blk_coord,
            blk_offset,
            problem_shape_cur_batch,
            has_k_tile,
            process_tile,
        )

    @cute.jit
    def load_persistent(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        Qt_in: Optional[cute.Tensor],
        Kt_in: Optional[cute.Tensor],
        dO_in: cute.Tensor,
        dOt_in: Optional[cute.Tensor],
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sKt: cute.Tensor,
        sQt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        (
            load_mma_Q_pipeline,
            load_mma_Qt_pipeline,
            load_mma_Kt_pipeline,
            load_mma_dO_pipeline,
        ) = pipeline_args
        load_mma_Q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.Q_stage,
        )
        load_mma_Qt_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.Q_stage,
        )
        load_mma_Kt_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.single_stage,
        )
        load_mma_dO_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.dO_stage,
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            (
                blk_coord,
                blk_offset,
                problem_shape_cur_batch,
                _,
                process_tile,
            ) = self.get_work_context(
                work_tile,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                block_sparse_tensors,
            )
            if process_tile:
                (
                    load_mma_Q_producer_state,
                    load_mma_Qt_producer_state,
                    load_mma_Kt_producer_state,
                    load_mma_dO_producer_state,
                ) = self.load_work(
                    K_in,
                    V_in,
                    Q_in,
                    Qt_in,
                    Kt_in,
                    dO_in,
                    dOt_in,
                    sK,
                    sQ,
                    sKt,
                    sQt,
                    sV,
                    sdO,
                    sdOt,
                    tiled_mma_S,
                    tiled_mma_dP,
                    tiled_mma_dQ,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    tma_atom_K,
                    tma_atom_Q,
                    tma_atom_Kt,
                    tma_atom_Qt,
                    tma_atom_V,
                    tma_atom_dO,
                    tma_atom_dOt,
                    blk_offset,
                    pipeline_args,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    blk_coord,
                    load_mma_Q_producer_state,
                    load_mma_Qt_producer_state,
                    load_mma_Kt_producer_state,
                    load_mma_dO_producer_state,
                )
            work_tile = tile_scheduler.advance_to_next_work()
        if cutlass.const_expr(self.use_2cta_instrs):
            load_mma_Q_pipeline.producer_tail(load_mma_Q_producer_state)
            load_mma_Qt_pipeline.producer_tail(load_mma_Qt_producer_state)
            load_mma_dO_pipeline.producer_tail(load_mma_dO_producer_state)
        if cutlass.const_expr(self.use_q_major_scheduler):
            load_mma_Kt_pipeline.producer_tail(load_mma_Kt_producer_state)

    @cute.jit
    def load_work(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        Qt_in: Optional[cute.Tensor],
        Kt_in: Optional[cute.Tensor],
        dO_in: cute.Tensor,
        dOt_in: Optional[cute.Tensor],
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sKt: cute.Tensor,
        sQt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        blk_offset: cute.Shape,
        # (load_mma_Q_pipeline, load_mma_dO_pipeline)
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        blk_coord: cute.Coord,
        load_mma_Q_producer_state: cutlass.pipeline.PipelineState,
        load_mma_Qt_producer_state: cutlass.pipeline.PipelineState,
        load_mma_Kt_producer_state: cutlass.pipeline.PipelineState,
        load_mma_dO_producer_state: cutlass.pipeline.PipelineState,
    ):
        _, blk_coord_k, _, blk_coord_batch = blk_coord
        _, blk_coord_h_k = blk_coord_batch[0]
        blk_coord_b = blk_coord_batch[1]
        seqlen_obj = SeqlenInfoCls(blk_coord_b)

        blk_coord_h_r = Int32(0)
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
        (
            load_mma_Q_pipeline,
            load_mma_Qt_pipeline,
            load_mma_Kt_pipeline,
            load_mma_dO_pipeline,
        ) = pipeline_args

        K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
        V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
        Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
        dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)
        Qt = Kt = dOt = None
        if cutlass.const_expr(self.use_2cta_instrs):
            assert Qt_in is not None
            assert Kt_in is not None
            assert dOt_in is not None
            Qt = cute.domain_offset(
                (Int32(0), blk_offset[0], blk_offset[3]),
                Qt_in,
            )
            Kt = cute.domain_offset(
                (Int32(0), blk_offset[1], blk_offset[3]),
                Kt_in,
            )
            dOt = cute.domain_offset(
                (Int32(0), blk_offset[0], blk_offset[3]),
                dOt_in,
            )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mn),
            (tiled_mma_S.thr_id.shape,),
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        mma_tile_coord_v = cute.arch.block_idx()[0] % self.cta_group_size
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        n_block_cta_group = blk_coord_k // self.cta_group_size

        # (bM, bK, RestM, RestK, (H, B))
        gK = cute.local_tile(
            K,
            cute.select(self.mma_tiler_kq, mode=[0, 2]),
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, (H, B))
        gQ = cute.local_tile(Q, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, None, None))
        # (bM, bK, RestM, RestK, (H, B))
        gV = cute.local_tile(V, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (None, None, None))
        # (bN, bK, RestN, RestK, (H, B))
        gdO = cute.local_tile(dO, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, None, None))
        gKt = gQt = gdOt = None
        if cutlass.const_expr(self.use_2cta_instrs):
            assert Kt is not None
            assert Qt is not None
            assert dOt is not None
            gKt = cute.local_tile(
                Kt,
                cute.select(self.mma_tiler_dsk, mode=[1, 2]),
                (None, None, None),
            )
            gQt = cute.local_tile(
                Qt,
                cute.select(self.mma_tiler_dsq, mode=[1, 2]),
                (None, None, None),
            )
            gdOt = cute.local_tile(
                dOt,
                cute.select(self.mma_tiler_pdo, mode=[1, 2]),
                (None, None, None),
            )

        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tSTgK = thr_mma_S.partition_A(gK)
        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tSTgQ = thr_mma_S.partition_B(gQ)
        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tdPTgV = thr_mma_dP.partition_A(gV)
        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tdPTgdO = thr_mma_dP.partition_B(gdO)
        tdQgKt = tdKgQt = tdVgdOt = None
        if cutlass.const_expr(self.use_2cta_instrs):
            assert gKt is not None
            assert gQt is not None
            assert gdOt is not None
            tdQgKt = thr_mma_dQ.partition_B(gKt)
            tdKgQt = thr_mma_dK.partition_B(gQt)
            tdVgdOt = thr_mma_dV.partition_B(gdOt)

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            block_in_cluster_coord_vmnk[2],
            cute.make_layout(
                cute.slice_(
                    cluster_layout_vmnk,
                    (0, 0, None, 0),
                ).shape
            ),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSTgK, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            block_in_cluster_coord_vmnk[1],
            cute.make_layout(
                cute.slice_(
                    cluster_layout_vmnk,
                    (0, None, 0, 0),
                ).shape
            ),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSTgQ, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            block_in_cluster_coord_vmnk[2],
            cute.make_layout(
                cute.slice_(
                    cluster_layout_vmnk,
                    (0, 0, None, 0),
                ).shape
            ),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPTgV, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            block_in_cluster_coord_vmnk[1],
            cute.make_layout(
                cute.slice_(
                    cluster_layout_vmnk,
                    (0, None, 0, 0),
                ).shape
            ),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPTgdO, 0, 3),
        )
        tKtsKt = tKtgKt_mkl = None
        tQtsQt = tQtgQt_mkl = None
        tdOtsdOt = tdOtgdOt_mkl = None
        if cutlass.const_expr(self.use_2cta_instrs):
            assert tma_atom_Kt is not None
            assert tma_atom_Qt is not None
            assert tma_atom_dOt is not None
            assert tdQgKt is not None
            assert tdKgQt is not None
            assert tdVgdOt is not None
            tKtsKt, tKtgKt_mkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_Kt,
                block_in_cluster_coord_vmnk[1],
                cute.make_layout(
                    cute.slice_(
                        cluster_layout_vmnk,
                        (0, None, 0, 0),
                    ).shape
                ),
                cute.group_modes(sKt, 0, 3),
                cute.group_modes(tdQgKt, 0, 3),
            )
            tQtsQt, tQtgQt_mkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_Qt,
                block_in_cluster_coord_vmnk[1],
                cute.make_layout(
                    cute.slice_(
                        cluster_layout_vmnk,
                        (0, None, 0, 0),
                    ).shape
                ),
                cute.group_modes(sQt, 0, 3),
                cute.group_modes(tdKgQt, 0, 3),
            )
            tdOtsdOt, tdOtgdOt_mkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_dOt,
                block_in_cluster_coord_vmnk[1],
                cute.make_layout(
                    cute.slice_(
                        cluster_layout_vmnk,
                        (0, None, 0, 0),
                    ).shape
                ),
                cute.group_modes(sdOt, 0, 3),
                cute.group_modes(tdVgdOt, 0, 3),
            )

        # Compute m_block info and initialize the traversal. All warp roles
        # use the same ascending MASK-then-FULL K2Q order.
        n_block = blk_coord_k
        mask_block_cnt = None
        mask_block_idx = None
        full_block_cnt = None
        full_block_idx = None
        if cutlass.const_expr(self.use_auto_block_metadata):
            (
                m_block_max,
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
            ) = get_q2k_block_sparse_consumer_row(
                block_sparse_tensors,
                blk_coord_b,
                n_block,
            )
            m_block_min = Int32(0)
        else:
            m_block_min, m_block_max, _ = block_info.get_m_block_info(
                seqlen_obj,
                n_block // self.cta_group_size,
            )
        m_block_iter = m_block_min

        if m_block_iter < m_block_max:
            if cutlass.const_expr(self.use_auto_block_metadata):
                m_block = self.get_block_sparse_m_block(
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                    m_block_iter,
                )
            else:
                m_block = m_block_iter

            if cutlass.const_expr(self.use_q_major_scheduler):
                load_mma_Kt_pipeline.producer_acquire(load_mma_Kt_producer_state)
                k_tma_barrier = load_mma_Kt_pipeline.producer_get_barrier(load_mma_Kt_producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK_mkl[
                        (
                            None,
                            n_block_cta_group,
                            0,
                            (blk_coord_h, blk_coord_b),
                        )
                    ],
                    tKsK[None, 0],
                    tma_bar_ptr=k_tma_barrier,
                )
                load_mma_Kt_producer_state.advance()

            load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
            tma_barrier = load_mma_Q_pipeline.producer_get_barrier(load_mma_Q_producer_state)
            # CuTeDSL 4.5.x has no per-acquire expected_tx override. The
            # pipeline already accounts for Q; extend its barrier for K.
            if cutlass.const_expr(not self.use_q_major_scheduler):
                if load_mma_Q_pipeline.is_leader_cta:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_K_bytes)

            # Load K
            if cutlass.const_expr(not self.use_q_major_scheduler):
                cute.copy(
                    tma_atom_K,
                    tKgK_mkl[
                        (
                            None,
                            n_block_cta_group,
                            0,
                            (blk_coord_h, blk_coord_b),
                        )
                    ],
                    tKsK[None, 0],
                    tma_bar_ptr=tma_barrier,
                )

            # Load Q
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, m_block, 0, (blk_coord_h, blk_coord_b))],
                tQsQ[None, load_mma_Q_producer_state.index],
                tma_bar_ptr=tma_barrier,
            )

            load_mma_Q_producer_state.advance()

            load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
            tma_barrier = load_mma_dO_pipeline.producer_get_barrier(load_mma_dO_producer_state)
            if load_mma_dO_pipeline.is_leader_cta:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dOt_bytes + self.tma_copy_V_bytes)

            # Load V
            cute.copy(
                tma_atom_V,
                tVgV_mkl[
                    (
                        None,
                        n_block_cta_group,
                        0,
                        (blk_coord_h, blk_coord_b),
                    )
                ],
                tVsV[(None, 0)],
                tma_bar_ptr=tma_barrier,
            )

            # Load dO
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, m_block, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO[(None, load_mma_dO_producer_state.index)],
                tma_bar_ptr=tma_barrier,
            )
            if cutlass.const_expr(self.use_2cta_instrs):
                assert tma_atom_dOt is not None
                assert tdOtgdOt_mkl is not None
                assert tdOtsdOt is not None
                cute.copy(
                    tma_atom_dOt,
                    tdOtgdOt_mkl[(None, 0, m_block, (blk_coord_h, blk_coord_b))],
                    tdOtsdOt[
                        None,
                        load_mma_dO_producer_state.index,
                    ],
                    tma_bar_ptr=tma_barrier,
                )
            load_mma_dO_producer_state.advance()

            if cutlass.const_expr(self.use_2cta_instrs):
                assert tma_atom_Kt is not None
                assert tKtgKt_mkl is not None
                assert tKtsKt is not None
                load_mma_Kt_pipeline.producer_acquire(load_mma_Kt_producer_state)
                tma_barrier = load_mma_Kt_pipeline.producer_get_barrier(load_mma_Kt_producer_state)
                cute.copy(
                    tma_atom_Kt,
                    tKtgKt_mkl[
                        (
                            None,
                            0,
                            n_block_cta_group,
                            (blk_coord_h, blk_coord_b),
                        )
                    ],
                    tKtsKt[None, 0],
                    tma_bar_ptr=tma_barrier,
                )
                load_mma_Kt_producer_state.advance()

            m_block_iter += 1

            pipeline_do_q_args = (
                load_mma_dO_pipeline,
                load_mma_Q_pipeline,
                load_mma_Qt_pipeline,
            )
            m_block_qt = m_block
            for next_m_block_iter in cutlass.range(
                m_block_iter,
                m_block_max,
                unroll=1,
            ):
                if cutlass.const_expr(self.use_auto_block_metadata):
                    m_block_valid = self.get_block_sparse_m_block(
                        mask_block_cnt,
                        mask_block_idx,
                        full_block_cnt,
                        full_block_idx,
                        next_m_block_iter,
                    )
                else:
                    m_block_valid = next_m_block_iter
                (
                    load_mma_dO_producer_state,
                    load_mma_Q_producer_state,
                    load_mma_Qt_producer_state,
                ) = self.load_step(
                    m_block_valid,
                    m_block_qt,
                    tma_atom_dO,
                    tdOgdO_mkl,
                    tdOsdO,
                    tma_atom_dOt,
                    tdOtgdOt_mkl,
                    tdOtsdOt,
                    tma_atom_Q,
                    tQgQ_mkl,
                    tQsQ,
                    tma_atom_Qt,
                    tQtgQt_mkl,
                    tQtsQt,
                    blk_coord_h,
                    blk_coord_b,
                    pipeline_do_q_args,
                    load_mma_dO_producer_state,
                    load_mma_Q_producer_state,
                    load_mma_Qt_producer_state,
                )
                m_block_qt = m_block_valid
            if cutlass.const_expr(self.use_2cta_instrs):
                assert tma_atom_Qt is not None
                assert tQtgQt_mkl is not None
                assert tQtsQt is not None
                load_mma_Qt_pipeline.producer_acquire(load_mma_Qt_producer_state)
                tma_barrier = load_mma_Qt_pipeline.producer_get_barrier(load_mma_Qt_producer_state)
                cute.copy(
                    tma_atom_Qt,
                    tQtgQt_mkl[
                        (
                            None,
                            0,
                            m_block_qt,
                            (blk_coord_h, blk_coord_b),
                        )
                    ],
                    tQtsQt[
                        None,
                        load_mma_Qt_producer_state.index,
                    ],
                    tma_bar_ptr=tma_barrier,
                )
                load_mma_Qt_producer_state.advance()
        return (
            load_mma_Q_producer_state,
            load_mma_Qt_producer_state,
            load_mma_Kt_producer_state,
            load_mma_dO_producer_state,
        )

    @cute.jit
    def load_step(
        self,
        m_block_valid: Int32,
        m_block_qt: Int32,
        tma_atom_dO: cute.CopyAtom,  # copy dO
        tdOgdO_mkl: cute.Tensor,
        tdOsdO: cute.Tensor,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tdOtgdOt_mkl: Optional[cute.Tensor],
        tdOtsdOt: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tQgQ_mkl: cute.Tensor,
        tQsQ: cute.Tensor,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tQtgQt_mkl: Optional[cute.Tensor],
        tQtsQt: Optional[cute.Tensor],
        blk_coord_h: tuple,
        blk_coord_b: Int32,
        pipeline_args: tuple,
        load_mma_dO_producer_state: cutlass.pipeline.PipelineState,
        load_mma_Q_producer_state: cutlass.pipeline.PipelineState,
        load_mma_Qt_producer_state: cutlass.pipeline.PipelineState,
    ):
        (
            load_mma_dO_pipeline,
            load_mma_Q_pipeline,
            load_mma_Qt_pipeline,
        ) = pipeline_args
        if cutlass.const_expr(self.use_2cta_instrs):
            assert tma_atom_Qt is not None
            assert tQtgQt_mkl is not None
            assert tQtsQt is not None
            load_mma_Qt_pipeline.producer_acquire(load_mma_Qt_producer_state)
            tma_barrier = load_mma_Qt_pipeline.producer_get_barrier(load_mma_Qt_producer_state)
            cute.copy(
                tma_atom_Qt,
                tQtgQt_mkl[
                    (
                        None,
                        0,
                        m_block_qt,
                        (blk_coord_h, blk_coord_b),
                    )
                ],
                tQtsQt[
                    None,
                    load_mma_Qt_producer_state.index,
                ],
                tma_bar_ptr=tma_barrier,
            )
            load_mma_Qt_producer_state.advance()

        load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
        tma_barrier = load_mma_Q_pipeline.producer_get_barrier(load_mma_Q_producer_state)

        # Load Q
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, m_block_valid, 0, (blk_coord_h, blk_coord_b))],
            tQsQ[None, load_mma_Q_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )

        load_mma_Q_producer_state.advance()

        load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
        tma_barrier = load_mma_dO_pipeline.producer_get_barrier(load_mma_dO_producer_state)
        if cutlass.const_expr(self.use_2cta_instrs):
            if load_mma_dO_pipeline.is_leader_cta:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dOt_bytes)

        # Load dO
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, m_block_valid, 0, (blk_coord_h, blk_coord_b))],
            tdOsdO[None, load_mma_dO_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )
        if cutlass.const_expr(self.use_2cta_instrs):
            assert tma_atom_dOt is not None
            assert tdOtgdOt_mkl is not None
            assert tdOtsdOt is not None
            cute.copy(
                tma_atom_dOt,
                tdOtgdOt_mkl[(None, 0, m_block_valid, (blk_coord_h, blk_coord_b))],
                tdOtsdOt[
                    None,
                    load_mma_dO_producer_state.index,
                ],
                tma_bar_ptr=tma_barrier,
            )
        load_mma_dO_producer_state.advance()

        return (
            load_mma_dO_producer_state,
            load_mma_Q_producer_state,
            load_mma_Qt_producer_state,
        )

    @cute.jit
    def mma_persistent_1cta_causal(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdKtdK: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdS: cute.Tensor,
        sKT: cute.Tensor,
        sdST: cute.Tensor,
        sQT: cute.Tensor,
        sdOT: cute.Tensor,
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        tSTrK = tiled_mma_S.make_fragment_A(sK)
        tSTrQ = tiled_mma_S.make_fragment_B(sQ)
        tdPTrV = tiled_mma_dP.make_fragment_A(sV)
        tdPTrdO = tiled_mma_dP.make_fragment_B(sdO)
        tdVrdOT = tiled_mma_dV.make_fragment_B(sdOT)
        if cutlass.const_expr(self.use_q1_small_mma):
            # qlen=1: compute dQ^T = K^T @ dS^T as a 128x8x128 GEMM.
            # sKT and sdS are transposed SMEM views of the existing buffers.
            tdQrKT = tiled_mma_dQ.make_fragment_A(sKT)
            tdQrdS = tiled_mma_dQ.make_fragment_B(sdS)
        else:
            tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
            tdQrKT = tiled_mma_dQ.make_fragment_B(sKT)
        tdKrdST = tiled_mma_dK.make_fragment_A(sdST)
        tdKrQT = tiled_mma_dK.make_fragment_B(sQT)

        mma_qk = partial(
            hstu_sm100_utils.gemm_w_idx,
            tiled_mma_S,
            tSTtST,
            tSTrK,
            tSTrQ,
            zero_init=True,
            num_unroll_groups=2,
        )
        mma_dov = partial(
            hstu_sm100_utils.gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPTtdPT,
            tdPTrV,
            tdPTrdO,
            sA=sV,
            sB=sdO,
            A_idx=0,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        mma_pdo = partial(
            hstu_sm100_utils.gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdOT,
            sA=None,
            sB=sdOT,
            cta_group=self.cta_group_size,
        )
        if cutlass.const_expr(self.use_q1_small_mma):
            mma_dsk = partial(
                hstu_sm100_utils.gemm_w_idx,
                tiled_mma_dQ,
                tdQtdQ,
                tdQrKT,
                tdQrdS,
                A_idx=0,
                B_idx=0,
                num_unroll_groups=1,
            )
        else:
            mma_dsk = partial(
                hstu_sm100_utils.gemm_w_idx,
                tiled_mma_dQ,
                tdQtdQ,
                tdQrdS,
                tdQrKT,
                A_idx=0,
                num_unroll_groups=1,
            )
        mma_dsq = partial(
            hstu_sm100_utils.gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdST,
            tdKrQT,
            sA=sdST,
            sB=sQT,
            A_idx=0,
            cta_group=self.cta_group_size,
        )

        (
            load_mma_Q_pipeline,
            _,
            load_mma_K_pipeline,
            mma_compute_S_pipeline,
            load_mma_dO_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        load_mma_Q_consumer_state = make_compact_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.Q_stage,
        )
        load_mma_Q_release_state = load_mma_Q_consumer_state.clone()
        load_mma_K_consumer_state = make_compact_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        consumer_phase_dO = Int32(0)
        producer_phase_acc = Int32(1)
        producer_phase_dQ = Int32(1)
        consumer_phase_dS = Int32(0)
        producer_phase_dKV = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, _, batch_idx = work_tile.tile_idx
            seqlen_obj = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max, _ = block_info.get_m_block_info(
                seqlen_obj,
                n_block,
            )
            m_block_count = m_block_max - m_block_min
            has_k_tile = Boolean(n_block * self.tile_n < seqlen_obj.seqlen_k)

            if has_k_tile and m_block_count > 0:
                accumulate_dK = False
                # Prologue: S, dP, and dV for the first Q block.
                if cutlass.const_expr(self.use_q_major_scheduler):
                    load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)
                load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
                q_stage = load_mma_Q_consumer_state.index
                k_stage = Int32(0)
                load_mma_Q_consumer_state.advance()
                mma_compute_S_pipeline.sync_object_empty.wait(
                    0,
                    producer_phase_acc,
                )
                mma_qk(A_idx=k_stage, B_idx=q_stage)
                mma_compute_S_pipeline.sync_object_full.arrive(
                    0,
                    mma_compute_S_pipeline.producer_mask,
                    mma_compute_S_pipeline.cta_group,
                )

                load_mma_dO_pipeline.sync_object_full.wait(
                    0,
                    consumer_phase_dO,
                )
                mma_compute_dP_pipeline.sync_object_empty.wait(
                    0,
                    producer_phase_acc,
                )
                mma_reduce_dQ_pipeline.sync_object_empty.wait(
                    0,
                    producer_phase_dQ,
                )
                mma_dov(B_idx=0)
                mma_compute_dP_pipeline.sync_object_full.arrive(
                    0,
                    mma_compute_dP_pipeline.producer_mask,
                    mma_compute_dP_pipeline.cta_group,
                )

                compute_mma_P_pipeline.sync_object_full.wait(
                    0,
                    consumer_phase_dS,
                )
                mma_pdo(
                    B_idx=0,
                    zero_init=True,
                )
                compute_mma_P_pipeline.sync_object_empty.arrive(
                    0,
                    compute_mma_P_pipeline.consumer_mask,
                    compute_mma_P_pipeline.cta_group,
                )
                load_mma_dO_pipeline.sync_object_empty.arrive(
                    0,
                    load_mma_dO_pipeline.consumer_mask,
                    load_mma_dO_pipeline.cta_group,
                )
                consumer_phase_dO ^= 1
                producer_phase_acc ^= 1

                for _ in cutlass.range(m_block_count - 1, unroll=1):
                    # S for the next Q block.
                    load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
                    q_stage_next = load_mma_Q_consumer_state.index
                    load_mma_Q_consumer_state.advance()
                    mma_compute_S_pipeline.sync_object_empty.wait(
                        0,
                        producer_phase_acc,
                    )
                    mma_qk(A_idx=k_stage, B_idx=q_stage_next)
                    mma_compute_S_pipeline.sync_object_full.arrive(
                        0,
                        mma_compute_S_pipeline.producer_mask,
                        mma_compute_S_pipeline.cta_group,
                    )

                    # dK and dQ for the current Q block.
                    compute_mma_dS_pipeline.sync_object_full.wait(
                        0,
                        consumer_phase_dS,
                    )
                    mma_compute_dP_pipeline.sync_object_empty.wait(
                        0,
                        producer_phase_acc,
                    )
                    mma_dsq(
                        B_idx=load_mma_Q_release_state.index,
                        zero_init=not accumulate_dK,
                    )
                    accumulate_dK = True
                    load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                    load_mma_Q_release_state.advance()

                    if cutlass.const_expr(self.use_q1_small_mma):
                        mma_dsk(zero_init=True)
                    else:
                        mma_dsk(
                            B_idx=k_stage,
                            zero_init=True,
                        )
                    if cutlass.const_expr(self.use_q1_small_mma):
                        # The reduced dQ tile is consumed from TMEM immediately.
                        # Tie the full-barrier arrival to UMMA completion instead
                        # of merely ordering a software mbarrier arrival after the
                        # instruction issue.
                        mma_reduce_dQ_pipeline.sync_object_full.arrive_tcgen05mma(
                            0,
                            mma_reduce_dQ_pipeline.producer_mask,
                            mma_reduce_dQ_pipeline.cta_group,
                        )
                    else:
                        mma_reduce_dQ_pipeline.sync_object_full.arrive(
                            0,
                            mma_reduce_dQ_pipeline.producer_mask,
                            mma_reduce_dQ_pipeline.cta_group,
                        )
                    compute_mma_dS_pipeline.sync_object_empty.arrive(
                        0,
                        compute_mma_dS_pipeline.consumer_mask,
                        compute_mma_dS_pipeline.cta_group,
                    )
                    producer_phase_dQ ^= 1
                    consumer_phase_dS ^= 1

                    # dP and dV for the next Q block.
                    mma_reduce_dQ_pipeline.sync_object_empty.wait(
                        0,
                        producer_phase_dQ,
                    )
                    load_mma_dO_pipeline.sync_object_full.wait(
                        0,
                        consumer_phase_dO,
                    )
                    mma_dov(B_idx=0)
                    mma_compute_dP_pipeline.sync_object_full.arrive(
                        0,
                        mma_compute_dP_pipeline.producer_mask,
                        mma_compute_dP_pipeline.cta_group,
                    )

                    compute_mma_P_pipeline.sync_object_full.wait(
                        0,
                        consumer_phase_dS,
                    )
                    mma_pdo(
                        B_idx=0,
                        zero_init=False,
                    )
                    compute_mma_P_pipeline.sync_object_empty.arrive(
                        0,
                        compute_mma_P_pipeline.consumer_mask,
                        compute_mma_P_pipeline.cta_group,
                    )
                    load_mma_dO_pipeline.sync_object_empty.arrive(
                        0,
                        load_mma_dO_pipeline.consumer_mask,
                        load_mma_dO_pipeline.cta_group,
                    )
                    consumer_phase_dO ^= 1
                    producer_phase_acc ^= 1

                # dV and dK use the two epilogue pipeline stages.
                mma_compute_dKdV_pipeline.sync_object_empty.wait(
                    0,
                    producer_phase_dKV,
                )
                mma_compute_dKdV_pipeline.sync_object_full.arrive(
                    0,
                    mma_compute_dKdV_pipeline.producer_mask,
                    mma_compute_dKdV_pipeline.cta_group,
                )
                mma_compute_dKdV_pipeline.sync_object_empty.wait(
                    1,
                    producer_phase_dKV,
                )

                # Tail: dK and dQ for the final Q block.
                compute_mma_dS_pipeline.sync_object_full.wait(
                    0,
                    consumer_phase_dS,
                )
                mma_dsq(
                    B_idx=load_mma_Q_release_state.index,
                    zero_init=not accumulate_dK,
                )
                mma_compute_dKdV_pipeline.sync_object_full.arrive(
                    1,
                    mma_compute_dKdV_pipeline.producer_mask,
                    mma_compute_dKdV_pipeline.cta_group,
                )
                producer_phase_dKV ^= 1

                if cutlass.const_expr(self.use_q1_small_mma):
                    mma_dsk(zero_init=True)
                else:
                    mma_dsk(
                        B_idx=k_stage,
                        zero_init=True,
                    )
                if cutlass.const_expr(self.use_q1_small_mma):
                    mma_reduce_dQ_pipeline.sync_object_full.arrive_tcgen05mma(
                        0,
                        mma_reduce_dQ_pipeline.producer_mask,
                        mma_reduce_dQ_pipeline.cta_group,
                    )
                else:
                    mma_reduce_dQ_pipeline.sync_object_full.arrive(
                        0,
                        mma_reduce_dQ_pipeline.producer_mask,
                        mma_reduce_dQ_pipeline.cta_group,
                    )
                load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                load_mma_Q_release_state.advance()
                if cutlass.const_expr(self.use_q1_small_mma):
                    producer_phase_dQ ^= 1
                    mma_reduce_dQ_pipeline.sync_object_empty.wait(
                        0,
                        producer_phase_dQ,
                    )
                if cutlass.const_expr(self.use_q_major_scheduler):
                    load_mma_K_pipeline.consumer_release(load_mma_K_consumer_state)
                    load_mma_K_consumer_state.advance()
                compute_mma_dS_pipeline.sync_object_empty.arrive(
                    0,
                    compute_mma_dS_pipeline.consumer_mask,
                    compute_mma_dS_pipeline.cta_group,
                )
                if cutlass.const_expr(not self.use_q1_small_mma):
                    producer_phase_dQ ^= 1
                consumer_phase_dS ^= 1

            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def mma_persistent(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdS: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdKtdK: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdS: cute.Tensor,
        sKT: cute.Tensor,
        sdST: cute.Tensor,
        sQT: cute.Tensor,
        sdOT: cute.Tensor,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        pipeline_args: tuple,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        is_leader_cta: Boolean,
    ):
        if cutlass.const_expr(not self.use_2cta_instrs and self.is_causal and not self.is_local and not self.is_arbitrary):
            self.mma_persistent_1cta_causal(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dQ,
                tiled_mma_dK,
                tSTtST,
                tdPTtdPT,
                tdVtdV,
                tdVrP,
                tdQtdQ,
                tdKtdK,
                sK,
                sQ,
                sV,
                sdO,
                sdS,
                sKT,
                sdST,
                sQT,
                sdOT,
                pipeline_args,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            return

        tSTrK = tiled_mma_S.make_fragment_A(sK)
        tSTrQ = tiled_mma_S.make_fragment_B(sQ)
        tdPTrV = tiled_mma_dP.make_fragment_A(sV)
        tdPTrdO = tiled_mma_dP.make_fragment_B(sdO)
        tdVrdOT = tiled_mma_dV.make_fragment_B(sdOT)
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrKT = tiled_mma_dQ.make_fragment_B(sKT)
        tdKrdST = tiled_mma_dK.make_fragment_A(tdS if self.use_2cta_instrs else sdST)
        tdKrQT = tiled_mma_dK.make_fragment_B(sQT)

        make_pipeline_state = pipeline.make_pipeline_state if self.use_2cta_instrs else make_compact_pipeline_state
        load_mma_Q_consumer_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.Q_stage,
        )
        load_mma_Q_release_state = load_mma_Q_consumer_state.clone()
        load_mma_Qt_consumer_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.Q_stage,
        )
        load_mma_Kt_consumer_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        mma_compute_S_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.single_stage,
        )
        compute_mma_dS_consumer_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        mma_compute_dP_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.single_stage,
        )
        mma_reduce_dQ_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.single_stage,
        )
        load_mma_dO_consumer_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.dO_stage,
        )
        compute_mma_P_consumer_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        mma_compute_dKdV_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.sdKVaccum_stage,
        )
        dS_cluster_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord, _, _, _, process_tile = self.get_work_context(
                work_tile,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                block_sparse_tensors,
            )
            if process_tile and is_leader_cta:
                (
                    load_mma_Q_consumer_state,
                    load_mma_Q_release_state,
                    load_mma_Qt_consumer_state,
                    load_mma_Kt_consumer_state,
                    mma_compute_S_producer_state,
                    compute_mma_dS_consumer_state,
                    mma_compute_dP_producer_state,
                    mma_reduce_dQ_producer_state,
                    load_mma_dO_consumer_state,
                    compute_mma_P_consumer_state,
                    mma_compute_dKdV_producer_state,
                    dS_cluster_phase,
                ) = self.mma_work(
                    tiled_mma_S,
                    tiled_mma_dP,
                    tiled_mma_dV,
                    tiled_mma_dQ,
                    tiled_mma_dK,
                    tSTtST,
                    tSTrQ,
                    tSTrK,
                    tdPTtdPT,
                    tdPTrV,
                    tdPTrdO,
                    tdVtdV,
                    tdVrP,
                    tdVrdOT,
                    tdQtdQ,
                    tdQrdS,
                    tdQrKT,
                    tdKrdST,
                    tdKtdK,
                    tdKrQT,
                    sK,
                    sQ,
                    sV,
                    sdO,
                    sdST,
                    sQT,
                    sdOT,
                    pipeline_args,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    blk_coord,
                    load_mma_Q_consumer_state,
                    load_mma_Q_release_state,
                    load_mma_Qt_consumer_state,
                    load_mma_Kt_consumer_state,
                    mma_compute_S_producer_state,
                    compute_mma_dS_consumer_state,
                    mma_compute_dP_producer_state,
                    mma_reduce_dQ_producer_state,
                    load_mma_dO_consumer_state,
                    compute_mma_P_consumer_state,
                    mma_compute_dKdV_producer_state,
                    dS_cluster_leader_mbar_ptr,
                    dS_cluster_phase,
                )
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def mma_work(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tSTtST: cute.Tensor,
        tSTrQ: cute.Tensor,
        tSTrK: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdPTrV: cute.Tensor,
        tdPTrdO: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdVrdOT: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdQrdS: cute.Tensor,
        tdQrKT: cute.Tensor,
        tdKrdST: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdKrQT: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdST: cute.Tensor,
        sQT: cute.Tensor,
        sdOT: cute.Tensor,
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        blk_coord: cute.Coord,
        load_mma_Q_consumer_state: cutlass.pipeline.PipelineState,
        load_mma_Q_release_state: cutlass.pipeline.PipelineState,
        load_mma_Qt_consumer_state: cutlass.pipeline.PipelineState,
        load_mma_Kt_consumer_state: cutlass.pipeline.PipelineState,
        mma_compute_S_producer_state: cutlass.pipeline.PipelineState,
        compute_mma_dS_consumer_state: cutlass.pipeline.PipelineState,
        mma_compute_dP_producer_state: cutlass.pipeline.PipelineState,
        mma_reduce_dQ_producer_state: cutlass.pipeline.PipelineState,
        load_mma_dO_consumer_state: cutlass.pipeline.PipelineState,
        compute_mma_P_consumer_state: cutlass.pipeline.PipelineState,
        mma_compute_dKdV_producer_state: cutlass.pipeline.PipelineState,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        dS_cluster_phase: Int32,
    ):
        _, blk_coord_k, _, blk_coord_batch = blk_coord
        _, blk_coord_b = blk_coord_batch
        seqlen_obj = SeqlenInfoCls(blk_coord_b)
        n_block = blk_coord_k

        if cutlass.const_expr(self.use_auto_block_metadata):
            m_block_max, _, _, _, _ = get_q2k_block_sparse_consumer_row(
                block_sparse_tensors,
                blk_coord_b,
                n_block,
            )
            m_block_min = Int32(0)
        else:
            m_block_min, m_block_max, _ = block_info.get_m_block_info(
                seqlen_obj,
                n_block // self.cta_group_size,
            )

        m_block_nums = m_block_max - m_block_min

        (
            load_mma_Q_pipeline,
            load_mma_Qt_pipeline,
            load_mma_Kt_pipeline,
            mma_compute_S_pipeline,
            load_mma_dO_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args
        accumulate_dK = False

        if m_block_min < m_block_max:
            load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
            # S = K * Q
            hstu_sm100_utils.gemm_ptx_w_idx(
                tiled_mma_S,
                tSTtST,
                tSTrK,
                tSTrQ,
                sK,
                sQ,
                A_idx=0,
                B_idx=load_mma_Q_consumer_state.index,
                zero_init=True,
                cta_group=self.cta_group_size,
            )

            if cutlass.const_expr(self.use_2cta_instrs):
                load_mma_Q_pipeline.consumer_release(load_mma_Q_consumer_state)
            load_mma_Q_consumer_state.advance()
            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
            if cutlass.const_expr(not self.use_2cta_instrs):
                mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)

            # dP = V * dO
            hstu_sm100_utils.gemm_ptx_w_idx(
                tiled_mma_dP,
                tdPTtdPT,
                tdPTrV,
                tdPTrdO,
                sV,
                sdO,
                A_idx=0,
                B_idx=load_mma_dO_consumer_state.index,
                zero_init=True,
                cta_group=self.cta_group_size,
            )

            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            if cutlass.const_expr(self.use_2cta_instrs):
                mma_compute_S_pipeline.sync_object_empty.wait(
                    0,
                    mma_compute_S_producer_state.phase,
                )
            else:
                compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

            # dV = P * dO
            if cutlass.const_expr(self.use_2cta_instrs):
                hstu_sm100_utils.gemm_ptx_w_idx(
                    tiled_mma_dV,
                    tdVtdV,
                    tdVrP,
                    tdVrdOT,
                    None,
                    sdOT,
                    B_idx=load_mma_dO_consumer_state.index,
                    zero_init=True,
                    tA_addr=self.tmem_S_offset,
                    cta_group=self.cta_group_size,
                )
            else:
                hstu_sm100_utils.gemm_ptx_w_idx(
                    tiled_mma_dV,
                    tdVtdV,
                    tdVrP,
                    tdVrdOT,
                    None,
                    sdOT,
                    B_idx=load_mma_dO_consumer_state.index,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )

            if cutlass.const_expr(not self.use_2cta_instrs):
                compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
                compute_mma_P_consumer_state.advance()

            load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
            load_mma_dO_consumer_state.advance()

            if cutlass.const_expr(self.use_2cta_instrs):
                load_mma_Kt_pipeline.consumer_wait(load_mma_Kt_consumer_state)

            m_block_nums -= 1

            for _ in cutlass.range(m_block_nums, unroll=1):
                load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
                if cutlass.const_expr(self.use_2cta_instrs):
                    mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
                else:
                    mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

                # S = K * Q
                hstu_sm100_utils.gemm_ptx_w_idx(
                    tiled_mma_S,
                    tSTtST,
                    tSTrK,
                    tSTrQ,
                    sK,
                    sQ,
                    A_idx=0,
                    B_idx=load_mma_Q_consumer_state.index,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )

                if cutlass.const_expr(self.use_2cta_instrs):
                    load_mma_Q_pipeline.consumer_release(load_mma_Q_consumer_state)
                load_mma_Q_consumer_state.advance()
                mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
                mma_compute_S_producer_state.advance()

                if cutlass.const_expr(not self.use_2cta_instrs):
                    compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                if cutlass.const_expr(self.use_2cta_instrs):
                    load_mma_Qt_pipeline.consumer_wait(load_mma_Qt_consumer_state)

                # We need to acquire dP here, because tmem dQ == tmem dP
                mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

                # dK = dS * Q
                q_stage = load_mma_Qt_consumer_state.index if self.use_2cta_instrs else load_mma_Q_release_state.index
                if cutlass.const_expr(self.use_2cta_instrs):
                    hstu_sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dK,
                        tdKtdK,
                        tdKrdST,
                        tdKrQT,
                        None,
                        sQT,
                        B_idx=q_stage,
                        zero_init=not accumulate_dK,
                        tA_addr=self.tmem_dS_offset,
                        cta_group=self.cta_group_size,
                    )
                else:
                    hstu_sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dK,
                        tdKtdK,
                        tdKrdST,
                        tdKrQT,
                        sdST,
                        sQT,
                        A_idx=compute_mma_dS_consumer_state.index,
                        B_idx=q_stage,
                        zero_init=not accumulate_dK,
                        cta_group=self.cta_group_size,
                    )
                accumulate_dK = True

                if cutlass.const_expr(self.use_2cta_instrs):
                    load_mma_Qt_pipeline.consumer_release(load_mma_Qt_consumer_state)
                    load_mma_Qt_consumer_state.advance()
                else:
                    load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                    load_mma_Q_release_state.advance()

                if cutlass.const_expr(self.use_2cta_instrs):
                    load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

                    # dP = V * dO
                    hstu_sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dP,
                        tdPTtdPT,
                        tdPTrV,
                        tdPTrdO,
                        sV,
                        sdO,
                        A_idx=0,
                        B_idx=load_mma_dO_consumer_state.index,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )

                    mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
                    mma_compute_dP_producer_state.advance()

                    compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

                # dQ = dS * K
                if cutlass.const_expr(self.use_2cta_instrs):
                    cute.arch.mbarrier_wait(
                        dS_cluster_leader_mbar_ptr,
                        phase=dS_cluster_phase,
                    )
                hstu_sm100_utils.gemm_w_idx(
                    tiled_mma_dQ,
                    tdQtdQ,
                    tdQrdS,
                    tdQrKT,
                    A_idx=compute_mma_dS_consumer_state.index,
                    B_idx=0,
                    zero_init=True,
                    num_unroll_groups=(2 if self.use_2cta_instrs else 1),
                )
                if cutlass.const_expr(self.use_2cta_instrs):
                    dS_cluster_phase ^= 1

                mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
                mma_reduce_dQ_producer_state.advance()

                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
                compute_mma_dS_consumer_state.advance()

                if cutlass.const_expr(not self.use_2cta_instrs):
                    # Load dQ here because dQ and dP share the same TMEM region.
                    mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
                    load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

                    # dP = V * dO
                    hstu_sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dP,
                        tdPTtdPT,
                        tdPTrV,
                        tdPTrdO,
                        sV,
                        sdO,
                        A_idx=0,
                        B_idx=load_mma_dO_consumer_state.index,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )

                    mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
                    mma_compute_dP_producer_state.advance()

                    compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)
                else:
                    mma_compute_S_pipeline.sync_object_empty.wait(
                        0,
                        mma_compute_S_producer_state.phase,
                    )

                # dV = P * dO
                if cutlass.const_expr(self.use_2cta_instrs):
                    hstu_sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dV,
                        tdVtdV,
                        tdVrP,
                        tdVrdOT,
                        None,
                        sdOT,
                        B_idx=load_mma_dO_consumer_state.index,
                        zero_init=False,
                        tA_addr=self.tmem_S_offset,
                        cta_group=self.cta_group_size,
                    )
                else:
                    hstu_sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dV,
                        tdVtdV,
                        tdVrP,
                        tdVrdOT,
                        None,
                        sdOT,
                        B_idx=load_mma_dO_consumer_state.index,
                        zero_init=False,
                        cta_group=self.cta_group_size,
                    )

                if cutlass.const_expr(not self.use_2cta_instrs):
                    compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
                    compute_mma_P_consumer_state.advance()

                load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
                load_mma_dO_consumer_state.advance()

            # Signal to the epilogue that dV is ready
            mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_producer_state.advance()

            mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)

            if cutlass.const_expr(self.use_2cta_instrs):
                mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
            else:
                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
            if cutlass.const_expr(self.use_2cta_instrs):
                load_mma_Qt_pipeline.consumer_wait(load_mma_Qt_consumer_state)

            # dK = dS * Q
            q_stage = load_mma_Qt_consumer_state.index if self.use_2cta_instrs else load_mma_Q_release_state.index
            if cutlass.const_expr(self.use_2cta_instrs):
                hstu_sm100_utils.gemm_ptx_w_idx(
                    tiled_mma_dK,
                    tdKtdK,
                    tdKrdST,
                    tdKrQT,
                    None,
                    sQT,
                    B_idx=q_stage,
                    zero_init=not accumulate_dK,
                    tA_addr=self.tmem_dS_offset,
                    cta_group=self.cta_group_size,
                )
            else:
                hstu_sm100_utils.gemm_ptx_w_idx(
                    tiled_mma_dK,
                    tdKtdK,
                    tdKrdST,
                    tdKrQT,
                    sdST,
                    sQT,
                    A_idx=compute_mma_dS_consumer_state.index,
                    B_idx=q_stage,
                    zero_init=not accumulate_dK,
                    cta_group=self.cta_group_size,
                )

            if cutlass.const_expr(self.use_2cta_instrs):
                load_mma_Qt_pipeline.consumer_release(load_mma_Qt_consumer_state)
                load_mma_Qt_consumer_state.advance()
            else:
                load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                load_mma_Q_release_state.advance()

            # Signal to epilogue that dK is ready
            mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_producer_state.advance()

            # We've already acquired mma_reduce_dq in the loop

            # dQ = dS * K
            if cutlass.const_expr(self.use_2cta_instrs):
                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                cute.arch.mbarrier_wait(
                    dS_cluster_leader_mbar_ptr,
                    phase=dS_cluster_phase,
                )
                mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
            hstu_sm100_utils.gemm_w_idx(
                tiled_mma_dQ,
                tdQtdQ,
                tdQrdS,
                tdQrKT,
                A_idx=compute_mma_dS_consumer_state.index,
                B_idx=0,
                zero_init=True,
                num_unroll_groups=(2 if self.use_2cta_instrs else 1),
            )
            if cutlass.const_expr(self.use_2cta_instrs):
                dS_cluster_phase ^= 1

            mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
            mma_reduce_dQ_producer_state.advance()

            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()
            if cutlass.const_expr(self.use_2cta_instrs):
                load_mma_Kt_pipeline.consumer_release(load_mma_Kt_consumer_state)
                load_mma_Kt_consumer_state.advance()
        return (
            load_mma_Q_consumer_state,
            load_mma_Q_release_state,
            load_mma_Qt_consumer_state,
            load_mma_Kt_consumer_state,
            mma_compute_S_producer_state,
            compute_mma_dS_consumer_state,
            mma_compute_dP_producer_state,
            mma_reduce_dQ_producer_state,
            load_mma_dO_consumer_state,
            compute_mma_P_consumer_state,
            mma_compute_dKdV_producer_state,
            dS_cluster_phase,
        )

    @cute.jit
    def compute_persistent(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVrP: cute.Tensor,
        sdS: cute.Tensor,
        sdST: cute.Tensor,
        sdS_xchg: Optional[cute.Tensor],
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        tma_tensor_dK: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        tma_tensor_dV: cute.Tensor,
        dKV_r2s_copy: cute.TiledCopy,
        sdK_epi: cute.Tensor,
        sdV_epi: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        thr_mma_S: cute.TiledMma,
        thr_mma_dV: cute.TiledMma,
        thr_mma_dK: cute.TiledMma,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        scaling_seqlen: Float32,
        alpha: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        func: Optional[cute.Tensor],
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        mma_compute_S_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        compute_mma_P_producer_state = None
        if cutlass.const_expr(not self.use_2cta_instrs):
            compute_mma_P_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.single_stage,
            )
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.single_stage,
        )
        mma_compute_dKdV_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.sdKVaccum_stage,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n * self.cta_group_size,
            self.is_arbitrary,
            self.is_causal,
            self.is_local,
            func_num=self.func_num,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            offset_dynamic=0,
            swapAB=True,
        )
        func_item = func[0, None, None] if func is not None else None
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            (
                blk_coord,
                blk_offset,
                problem_shape_cur_batch,
                has_k_tile,
                process_tile,
            ) = self.get_work_context(
                work_tile,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                block_sparse_tensors,
            )
            if has_k_tile:
                if process_tile:
                    mask = AttentionMaskCls(
                        offset_q=blk_offset[0],
                        seqlen_q=problem_shape_cur_batch[0],
                        seqlen_k=problem_shape_cur_batch[1],
                        func=func_item,
                    )
                    (
                        mma_compute_S_consumer_state,
                        compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state,
                        mma_compute_dKdV_consumer_state,
                    ) = self.compute_work(
                        tSTtST,
                        tdPTtdPT,
                        tdVrP,
                        sdS,
                        sdST,
                        sdS_xchg,
                        dS_cluster_full_mbar_ptr,
                        dK,
                        dV,
                        tma_atom_dK,
                        tma_tensor_dK,
                        tma_atom_dV,
                        tma_tensor_dV,
                        dKV_r2s_copy,
                        sdK_epi,
                        sdV_epi,
                        tdKtdK,
                        tdVtdV,
                        thr_mma_S,
                        thr_mma_dV,
                        thr_mma_dK,
                        blk_coord,
                        blk_offset,
                        problem_shape_cur_batch,
                        scaling_seqlen,
                        alpha,
                        mask,
                        pipeline_args,
                        block_info,
                        block_sparse_tensors,
                        SeqlenInfoCls,
                        mma_compute_S_consumer_state,
                        compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state,
                        mma_compute_dKdV_consumer_state,
                    )
                else:
                    self.epilogue_clear(
                        blk_coord,
                        blk_offset,
                        problem_shape_cur_batch,
                        dK,
                        dV,
                    )
            work_tile = tile_scheduler.advance_to_next_work()
        self.epilogue_sync_barrier.arrive_and_wait()

    @cute.jit
    def compute_work(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVrP: cute.Tensor,
        sdS: cute.Tensor,
        sdST: cute.Tensor,
        sdS_xchg: Optional[cute.Tensor],
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        tma_tensor_dK: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        tma_tensor_dV: cute.Tensor,
        dKV_r2s_copy: cute.TiledCopy,
        sdK_epi: cute.Tensor,
        sdV_epi: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        thr_mma_S: cute.TiledMma,
        thr_mma_dV: cute.TiledMma,
        thr_mma_dK: cute.TiledMma,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        scaling_seqlen: Float32,
        alpha: Float32,
        mask: AttentionMask,
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        mma_compute_S_consumer_state: cutlass.pipeline.PipelineState,
        compute_mma_P_producer_state: Optional[cutlass.pipeline.PipelineState],
        mma_compute_dP_consumer_state: cutlass.pipeline.PipelineState,
        compute_mma_dS_producer_state: cutlass.pipeline.PipelineState,
        mma_compute_dKdV_consumer_state: cutlass.pipeline.PipelineState,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, blk_coord_k, _, blk_coord_batch = blk_coord
        _, blk_coord_b = blk_coord_batch
        seqlen_obj = SeqlenInfoCls(blk_coord_b)

        n_block = blk_coord_k
        mask_block_cnt = None
        mask_block_idx = None
        full_block_cnt = None
        full_block_idx = None
        if cutlass.const_expr(self.use_auto_block_metadata):
            (
                m_block_max,
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
            ) = get_q2k_block_sparse_consumer_row(
                block_sparse_tensors,
                blk_coord_b,
                n_block,
            )
            m_block_min = Int32(0)
            m_masking_steps = Int32(0)
        else:
            m_block_min, m_block_max, m_masking_steps = block_info.get_m_block_info(
                seqlen_obj,
                n_block // self.cta_group_size,
            )
        # Define the initial traversal slot.
        m_block = m_block_min

        (
            mma_compute_S_pipeline,
            compute_mma_P_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32 if self.use_2cta_instrs else 16)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16 if self.use_2cta_instrs else 8)),
            (self.acc_dtype if self.use_2cta_instrs else self.element_dtype),
        )

        if cutlass.const_expr(not self.use_2cta_instrs):
            tSTtST = tSTtST[(None, None), 0, 0]
            tdPTtdPT = tdPTtdPT[(None, None), 0, 0]

        cST_identity = cute.make_identity_tensor(cute.select(self.mma_tiler_kq, mode=[0, 1]))
        cdPT_identity = cute.make_identity_tensor(cute.select(self.mma_tiler_vdo, mode=[0, 1]))
        cST = thr_mma_S.partition_C(cST_identity) if self.use_2cta_instrs else cST_identity
        cdPT = thr_mma_S.partition_C(cdPT_identity) if self.use_2cta_instrs else cdPT_identity

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % (self.num_compute_warps * cute.arch.WARP_SIZE) if self.use_2cta_instrs else tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * cute.arch.WARP_SIZE)) // 128
        tiled_t2r = make_tmem_copy(tmem_load_atom, num_warp_groups) if self.use_2cta_instrs else tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cST = thr_t2r.partition_D(cST)
        if cutlass.const_expr(not self.use_2cta_instrs):
            tTR_cST = split_wg(tTR_cST, num_warp_groups, wg_idx)
        tTR_rST = cute.make_fragment_like(cute.make_layout(tTR_cST.shape), self.acc_dtype)

        tTR_tST = thr_t2r.partition_S(tSTtST)
        if cutlass.const_expr(not self.use_2cta_instrs):
            tTR_tST = split_wg(tTR_tST, num_warp_groups, wg_idx)

        tTR_cdPT_p = thr_t2r.partition_D(cdPT)
        tTR_cdPT = tTR_cdPT_p
        if cutlass.const_expr(not self.use_2cta_instrs):
            tTR_cdPT = split_wg(tTR_cdPT_p, num_warp_groups, wg_idx)
        tTR_rdPT = (
            cute.make_fragment_like(
                tTR_cdPT[None, 0, None, None],
                self.acc_dtype,
            )
            if self.use_2cta_instrs
            else cute.make_fragment_like(
                cute.make_layout(tTR_cdPT.shape),
                self.acc_dtype,
            )
        )

        tTR_tdPT = thr_t2r.partition_S(tdPTtdPT)
        if cutlass.const_expr(not self.use_2cta_instrs):
            tTR_tdPT = split_wg(tTR_tdPT, num_warp_groups, wg_idx)

        tdRT_tdS = None
        tdRT_cdS = None
        if cutlass.const_expr(self.use_2cta_instrs):
            tile_P_f32 = self.tile_m // 32 * self.element_dtype.width
            tSTtP = cute.composition(
                tSTtST,
                (
                    cute.make_layout((self.tile_n, tile_P_f32)),
                    1,
                    1,
                ),
            )
            tSTtP = cute.make_tensor(
                tSTtST.iterator,
                tSTtP.layout,
            )
            tSTcP = cute.composition(
                cST,
                (
                    cute.make_layout((self.tile_n, tile_P_f32)),
                    1,
                    1,
                ),
            )
            tdPTtdS = cute.composition(
                tdPTtdPT,
                (
                    cute.make_layout((self.tile_n, tile_P_f32)),
                    1,
                    1,
                ),
            )
            tdPTcdS = cute.composition(
                cdPT,
                (
                    cute.make_layout((self.tile_n, tile_P_f32)),
                    1,
                    1,
                ),
            )
            tiled_r2t = make_tmem_copy(
                tmem_store_atom,
                num_warp_groups,
            )
        else:
            tdVcST = thr_mma_dV.partition_A(cST_identity)
            tiled_r2t = tcgen05.make_tmem_copy(
                tmem_store_atom,
                tdVrP,
            )
        thr_r2t = tiled_r2t.get_slice(dp_idx)

        if cutlass.const_expr(self.use_2cta_instrs):
            tRT_tP = thr_r2t.partition_D(tSTtP)
            tRT_cST = thr_r2t.partition_S(tSTcP)
            tdRT_tdS = thr_r2t.partition_D(tdPTtdS)
            tdRT_cdS = thr_r2t.partition_S(tdPTcdS)
        else:
            tRT_tP = thr_r2t.partition_D(tdVrP)
            tRT_tP = split_wg(tRT_tP, num_warp_groups, wg_idx)
            tRT_cST = thr_r2t.partition_S(tdVcST)
            tRT_cST = split_wg(tRT_cST, num_warp_groups, wg_idx)

        tRS_sdS = None
        tRS_sdS_xchg = None
        if cutlass.const_expr(self.use_2cta_instrs):
            copy_atom_r2s = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR,
                self.element_dtype,
                self.acc_dtype,
                tiled_t2r,
            )
            thr_copy_r2s = cute.make_tiled_copy_D(
                copy_atom_r2s,
                tiled_t2r,
            ).get_slice(dp_idx)
            sdS_epi_layout = sm100_utils.make_smem_layout_epi(
                self.element_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (self.tile_n, self.tile_m),
                1,
            )
            sdS_layout = cute.slice_(
                sdS_epi_layout.outer,
                (None, None, 0),
            )
            sdS_layout = cute.make_layout(
                (sdS_layout.shape,),
                stride=(sdS_layout.stride,),
            )
            sdS_epi = cute.make_tensor(
                cute.recast_ptr(
                    sdS.iterator,
                    sdS_epi_layout.inner,
                ),
                sdS_layout,
            )
            tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)
            assert sdS_xchg is not None
            sdS_xchg_epi = cute.make_tensor(
                cute.recast_ptr(
                    sdS_xchg.iterator,
                    sdS_epi_layout.inner,
                ),
                sdS_layout,
            )
            tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        exchange_stage = cta_rank_in_cluster ^ 1 if self.use_2cta_instrs else Int32(0)

        mask_fn = partial(
            mask.apply_mask_swapAB,
            n_block=blk_coord_k // self.cta_group_size,
            tScS_t2r=tTR_cST,
        )
        causal_r2p_mask_fn = partial(
            mask.build_mask_swapAB_r2p,
            n_block=blk_coord_k // self.cta_group_size,
            tScS_t2r=tTR_cST,
        )
        arbitrary_mask_fn = mask_fn
        if cutlass.const_expr(self.use_auto_block_metadata and self.func_num >= 5):
            # Prefetch endpoint planes for longer functions to control register pressure.
            arbitrary_mask_fn = partial(
                mask.apply_mask_swapAB_arbitrary_prefetch,
                n_block=blk_coord_k,
                wg_idx=wg_idx,
                thr_tmem_load=thr_t2r,
            )
        seqlen_mask_fn = partial(
            mask.apply_mask_swapAB_seqlen,
            n_block=blk_coord_k,
            wg_idx=wg_idx,
            thr_tmem_load=thr_t2r,
        )

        fastsilu = FastSilU(alpha)

        if m_block_min < m_block_max:
            s_p_pipeline_args = (mma_compute_S_pipeline, compute_mma_P_pipeline)
            ds_dp_pipeline_args = (mma_compute_dP_pipeline, compute_mma_dS_pipeline)
            compute_mask_step = partial(
                self.compute_step,
                m_block_min=m_block_min,
                s_p_pipeline_args=s_p_pipeline_args,
                mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                compute_mma_P_producer_state=compute_mma_P_producer_state,
                tiled_t2r=tiled_t2r,
                tiled_r2t=tiled_r2t,
                tTR_tST=tTR_tST,
                tTR_rST=tTR_rST,
                tRT_cST=tRT_cST,
                tRT_tP=tRT_tP,
                ds_dp_pipeline_args=ds_dp_pipeline_args,
                mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                tTR_tdPT=tTR_tdPT,
                tTR_rdPT=tTR_rdPT,
                sdS=sdS,
                sdST=sdST,
                dp_idx=dp_idx,
                tTR_cdPT_p=tTR_cdPT_p,
                num_warp_groups=num_warp_groups,
                wg_idx=wg_idx,
                fastsilu=fastsilu,
                causal_r2p_mask_fn=causal_r2p_mask_fn,
                tRS_sdS=tRS_sdS,
                tRS_sdS_xchg=tRS_sdS_xchg,
                tdRT_tdS=tdRT_tdS,
                tdRT_cdS=tdRT_cdS,
                sdS_xchg=sdS_xchg,
                dS_cluster_full_mbar_ptr=dS_cluster_full_mbar_ptr,
                cta_rank_in_cluster=cta_rank_in_cluster,
                exchange_stage=exchange_stage,
            )

            if cutlass.const_expr(self.use_auto_block_metadata):
                # MASK blocks evaluate the arbitrary predicate. The builder
                # stores each row in ascending order and all roles consume the
                # same MASK-then-FULL concatenation.
                while m_block < mask_block_cnt:
                    m_block_valid = self.get_block_sparse_m_block(
                        mask_block_cnt,
                        mask_block_idx,
                        full_block_cnt,
                        full_block_idx,
                        m_block,
                    )
                    (
                        mma_compute_S_consumer_state,
                        compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state,
                    ) = compute_mask_step(
                        m_block_valid=m_block_valid,
                        mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                        compute_mma_P_producer_state=compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                        mask_fn=arbitrary_mask_fn,
                    )
                    m_block += 1

                # FULL interior blocks do not allocate a predicate fragment.
                # Logical FULL tails retain Q/K bounds without reading func.
                while m_block < m_block_max:
                    m_block_valid = self.get_block_sparse_m_block(
                        mask_block_cnt,
                        mask_block_idx,
                        full_block_cnt,
                        full_block_idx,
                        m_block,
                    )
                    is_full_tail = Boolean((m_block_valid + 1) * self.tile_m > seqlen_obj.seqlen_q or (n_block + 1) * self.tile_n > seqlen_obj.seqlen_k)
                    if is_full_tail:
                        (
                            mma_compute_S_consumer_state,
                            compute_mma_P_producer_state,
                            mma_compute_dP_consumer_state,
                            compute_mma_dS_producer_state,
                        ) = compute_mask_step(
                            m_block_valid=m_block_valid,
                            mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                            compute_mma_P_producer_state=compute_mma_P_producer_state,
                            mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                            compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                            mask_fn=seqlen_mask_fn,
                        )
                    else:
                        (
                            mma_compute_S_consumer_state,
                            compute_mma_P_producer_state,
                            mma_compute_dP_consumer_state,
                            compute_mma_dS_producer_state,
                        ) = compute_mask_step(
                            m_block_valid=m_block_valid,
                            mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                            compute_mma_P_producer_state=compute_mma_P_producer_state,
                            mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                            compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                            mask_fn=None,
                        )
                    m_block += 1
            else:
                masking_step = 0
                if cutlass.const_expr(self.is_local):
                    while m_block < m_block_max:
                        m_block_valid = m_block
                        (
                            mma_compute_S_consumer_state,
                            compute_mma_P_producer_state,
                            mma_compute_dP_consumer_state,
                            compute_mma_dS_producer_state,
                        ) = compute_mask_step(
                            m_block_valid=m_block_valid,
                            mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                            compute_mma_P_producer_state=compute_mma_P_producer_state,
                            mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                            compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                            mask_fn=partial(
                                mask_fn,
                                mask_causal=True,
                                mask_seqlen=True,
                            ),
                        )
                        m_block += 1

                while m_block < m_block_max and masking_step < m_masking_steps:
                    m_block_valid = m_block
                    (
                        mma_compute_S_consumer_state,
                        compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state,
                    ) = compute_mask_step(
                        m_block_valid=m_block_valid,
                        mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                        compute_mma_P_producer_state=compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                        mask_fn=partial(
                            mask_fn,
                            mask_causal=True,
                            mask_seqlen=True,
                        ),
                    )
                    masking_step += 1
                    m_block += 1

                while m_block < m_block_max - 1 and (n_block + 1) * self.tile_n <= seqlen_obj.seqlen_k:
                    m_block_valid = m_block
                    (
                        mma_compute_S_consumer_state,
                        compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state,
                    ) = compute_mask_step(
                        m_block_valid=m_block_valid,
                        mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                        compute_mma_P_producer_state=compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                        mask_fn=None,
                    )
                    m_block += 1

                while m_block < m_block_max:
                    m_block_valid = m_block
                    (
                        mma_compute_S_consumer_state,
                        compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state,
                    ) = compute_mask_step(
                        m_block_valid=m_block_valid,
                        mma_compute_S_consumer_state=mma_compute_S_consumer_state,
                        compute_mma_P_producer_state=compute_mma_P_producer_state,
                        mma_compute_dP_consumer_state=mma_compute_dP_consumer_state,
                        compute_mma_dS_producer_state=compute_mma_dS_producer_state,
                        mask_fn=partial(
                            mask_fn,
                            mask_causal=False,
                            mask_seqlen=True,
                        ),
                    )
                    m_block += 1

            if cutlass.const_expr(self.use_2cta_instrs):
                with cute.arch.elect_one():
                    compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
                compute_mma_dS_producer_state.advance()

            # Epilogue
            mma_compute_dKdV_consumer_state = self.epilogue(
                blk_coord,
                blk_offset,
                problem_shape,
                scaling_seqlen,
                alpha,
                dK,
                dV,
                tma_atom_dK,
                tma_tensor_dK,
                tma_atom_dV,
                tma_tensor_dV,
                dKV_r2s_copy,
                sdK_epi,
                sdV_epi,
                tdKtdK,
                tdVtdV,
                thr_mma_dK,
                thr_mma_dV,
                (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state),
            )
        else:
            self.epilogue_clear(
                blk_coord,
                blk_offset,
                problem_shape,
                dK,
                dV,
            )
        return (
            mma_compute_S_consumer_state,
            compute_mma_P_producer_state,
            mma_compute_dP_consumer_state,
            compute_mma_dS_producer_state,
            mma_compute_dKdV_consumer_state,
        )

    @cute.jit
    def compute_step(
        self,
        m_block_valid: Int32,
        m_block_min: Int32,
        s_p_pipeline_args: tuple,
        mma_compute_S_consumer_state: cutlass.pipeline.PipelineState,
        compute_mma_P_producer_state: Optional[cutlass.pipeline.PipelineState],
        tiled_t2r: cute.TiledCopy,
        tiled_r2t: cute.TiledCopy,
        tTR_tST: cute.Tensor,
        tTR_rST: cute.Tensor,
        tRT_cST: cute.Tensor,
        tRT_tP: cute.Tensor,
        ds_dp_pipeline_args: tuple,
        mma_compute_dP_consumer_state: cutlass.pipeline.PipelineState,
        compute_mma_dS_producer_state: cutlass.pipeline.PipelineState,
        tTR_tdPT: cute.Tensor,
        tTR_rdPT: cute.Tensor,
        sdS: cute.Tensor,
        sdST: cute.Tensor,
        dp_idx: Int32,
        tTR_cdPT_p: cute.Tensor,
        num_warp_groups: Int32,
        wg_idx: Int32,
        fastsilu: FastSilU,
        causal_r2p_mask_fn: Callable,
        tRS_sdS: Optional[cute.Tensor],
        tRS_sdS_xchg: Optional[cute.Tensor],
        tdRT_tdS: Optional[cute.Tensor],
        tdRT_cdS: Optional[cute.Tensor],
        sdS_xchg: Optional[cute.Tensor],
        dS_cluster_full_mbar_ptr: cute.Pointer,
        cta_rank_in_cluster: Int32,
        exchange_stage: Int32,
        mask_fn: Optional[Callable] = None,
    ):
        mma_compute_S_pipeline, compute_mma_P_pipeline = s_p_pipeline_args
        mma_compute_dP_pipeline, compute_mma_dS_pipeline = ds_dp_pipeline_args

        mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
        if cutlass.const_expr(not self.use_2cta_instrs):
            assert compute_mma_P_producer_state is not None
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)

        # Compute P = silu(S)
        cute.copy(tiled_t2r, tTR_tST, tTR_rST)
        if cutlass.const_expr(self.use_2cta_instrs):
            cute.arch.fence_view_async_tmem_load()
            if m_block_valid > m_block_min:
                with cute.arch.elect_one():
                    compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
                compute_mma_dS_producer_state.advance()
        use_causal_r2p_mask = self.use_2cta_instrs and self.is_causal and not self.is_local and not self.is_arbitrary and mask_fn is not None
        keep_masks = None
        if cutlass.const_expr(use_causal_r2p_mask):
            keep_masks = cute.make_rmem_tensor(
                (cute.size(tTR_rST) // 32,),
                cutlass.Uint32,
            )
            causal_r2p_mask_fn(keep_masks, m_block=m_block_valid)
        tTR_rST_preds = None
        if cutlass.const_expr(mask_fn is not None and not use_causal_r2p_mask):
            tTR_rST_preds = cute.make_rmem_tensor(
                tTR_rST.shape,
                cutlass.Boolean,
            )
            for i in cutlass.range(
                0,
                cute.size(tTR_rST),
                unroll_full=True,
            ):
                tTR_rST_preds[i] = True
            mask_fn(tTR_rST_preds, m_block=m_block_valid)
        use_stagewise_1cta = not self.use_2cta_instrs and self.is_causal and not self.is_local and not self.is_arbitrary
        if cutlass.const_expr(self.use_2cta_instrs):
            num_stages = cute.size(tTR_rST, mode=[1])
            tRT_rP_f32 = cute.make_rmem_tensor(
                tRT_cST.shape,
                self.acc_dtype,
            )
            tRT_rP = cute.recast_tensor(
                tRT_rP_f32,
                self.element_dtype,
            )
            for stage in cutlass.range_constexpr(num_stages):
                tTR_rST_cur = tTR_rST[None, stage, 0, 0]
                tTR_rST_preds_cur = tTR_rST_preds[None, stage, 0, 0] if tTR_rST_preds is not None else None
                keep_mask = keep_masks[stage] if keep_masks is not None else None
                fastsilu.dsilu_bwd_quantize_x2(
                    tTR_rST_cur,
                    tRT_rP[None, stage, 0, 0],
                    tTR_rST_preds_cur,
                    r2p_mask=keep_mask,
                )
                if cutlass.const_expr(stage == 0):
                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()
                cute.copy(
                    tiled_r2t,
                    tRT_rP_f32[None, stage, None, None],
                    tRT_tP[None, stage, None, None],
                )
        elif cutlass.const_expr(use_stagewise_1cta):
            num_stages = cute.size(tTR_rST, mode=[2])
            for stage in cutlass.range_constexpr(num_stages):
                tTR_rST_cur = tTR_rST[None, 0, stage]
                tTR_rST_preds_cur = tTR_rST_preds[None, 0, stage] if tTR_rST_preds is not None else None
                tTR_rP_cur = cute.make_fragment_like(
                    tTR_rST_cur,
                    self.element_dtype,
                )
                fastsilu.dsilu_bwd_quantize_x2(
                    tTR_rST_cur,
                    tTR_rP_cur,
                    tTR_rST_preds_cur,
                )
                tRT_rP_cur = cute.make_tensor(
                    tTR_rP_cur.iterator,
                    cute.make_layout(tRT_cST[None, 0, 0, stage].shape),
                )
                if cutlass.const_expr(stage == 0):
                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()
                cute.copy(
                    tiled_r2t,
                    tRT_rP_cur,
                    tRT_tP[None, 0, 0, stage],
                )
        else:
            tTR_rP = cute.make_fragment_like(tTR_rST)
            fastsilu.dsilu_bwd_x2(
                tTR_rST,
                tTR_rP,
                tTR_rST_preds,
                fastsilu.score_scale,
                mask_fn=mask_fn,
            )
            # Convert FP32 P to the Tensor Core operand type.
            tRT_rP = self.quantize(tTR_rP, 4)
            tRT_rP_reshaped = cute.make_tensor(
                tRT_rP.iterator,
                cute.make_layout(tRT_cST.shape),
            )
            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()
            cute.copy(tiled_r2t, tRT_rP_reshaped, tRT_tP)

        cute.arch.fence_view_async_tmem_store()

        # Notify for P
        if cutlass.const_expr(not self.use_2cta_instrs):
            assert compute_mma_P_producer_state is not None
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

        # Release S
        if cutlass.const_expr(self.use_2cta_instrs):
            with cute.arch.elect_one():
                mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
        else:
            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
        mma_compute_S_consumer_state.advance()

        # Wait for dP
        mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)

        if cutlass.const_expr(not self.use_2cta_instrs):
            compute_mma_dS_pipeline.producer_acquire(
                compute_mma_dS_producer_state,
            )

        # Compute dS = dsilu(S, dP).
        if cutlass.const_expr(self.use_2cta_instrs):
            assert tRS_sdS is not None
            assert tRS_sdS_xchg is not None
            assert tdRT_tdS is not None
            assert tdRT_cdS is not None
            assert sdS_xchg is not None
            num_stages = cute.size(tTR_rST, mode=[1])
            tTR_rdPT_t2r = tTR_rdPT
            tTR_rdPT = tTR_rdPT_t2r[None, 0, 0]
            tTR_rdST_xchg = cute.make_rmem_tensor(
                tTR_rdPT.shape,
                self.element_dtype,
            )
            for stage in cutlass.range_constexpr(num_stages):
                tTR_rST_cur = tTR_rST[None, stage, 0, 0]
                cute.copy(
                    tiled_t2r,
                    tTR_tdPT[None, stage, None, None],
                    tTR_rdPT_t2r,
                )
                cute.arch.fence_view_async_tmem_load()
                self.compute_sync_barrier.arrive_and_wait()
                for i in cutlass.range_constexpr(
                    0,
                    cute.size(tTR_rdPT),
                    2,
                ):
                    tTR_rdPT[i], tTR_rdPT[i + 1] = mul_packed_f32x2(
                        (
                            tTR_rdPT[i],
                            tTR_rdPT[i + 1],
                        ),
                        (
                            tTR_rST_cur[i],
                            tTR_rST_cur[i + 1],
                        ),
                    )
                tTR_rdST_cur = self.quantize(tTR_rdPT, 4)
                if cutlass.const_expr(stage == 0):
                    compute_mma_dS_pipeline.producer_acquire(
                        compute_mma_dS_producer_state,
                    )
                tdRT_rdS_cur = cute.recast_tensor(
                    tTR_rdST_cur,
                    self.acc_dtype,
                )
                cute.copy(
                    tiled_r2t,
                    tdRT_rdS_cur,
                    tdRT_tdS[None, stage, 0, 0],
                )
                if exchange_stage == stage:
                    cute.autovec_copy(
                        tTR_rdST_cur,
                        tTR_rdST_xchg,
                    )
                else:
                    cute.autovec_copy(
                        tTR_rdST_cur,
                        tRS_sdS[None, stage],
                    )
            cute.arch.fence_view_async_tmem_store()
            cute.autovec_copy(
                tTR_rdST_xchg,
                tRS_sdS_xchg[None, 0],
            )
        elif cutlass.const_expr(use_stagewise_1cta):
            sdS_slice = sdS[
                None,
                None,
                None,
                compute_mma_dS_producer_state.index,
            ]
            thread_layout = cute.make_ordered_layout(
                (self.tile_n, self.tile_m),
                (1, 0),
            )
            sdS_slice_tmp = cute.composition(
                sdS_slice,
                thread_layout,
            )
            sdS_slice_p = cute.composition(
                sdS_slice_tmp[dp_idx, None],
                cute.make_layout(tTR_cdPT_p.shape),
            )
            sdS_slice = split_wg(
                sdS_slice_p,
                num_warp_groups,
                wg_idx,
            )
            tTR_cdPT_direct = tTR_cdPT_p
            if cutlass.const_expr(self.use_q1_small_mma):
                tTR_cdPT_direct = split_wg(
                    tTR_cdPT_p,
                    num_warp_groups,
                    wg_idx,
                )
            num_stages = cute.size(tTR_rST, mode=[2])
            for stage in cutlass.range_constexpr(num_stages):
                tTR_rST_cur = tTR_rST[None, 0, stage]
                tTR_rdPT_cur = cute.make_fragment_like(
                    tTR_rST_cur,
                    self.acc_dtype,
                )
                cute.copy(
                    tiled_t2r,
                    tTR_tdPT[None, 0, stage],
                    tTR_rdPT_cur,
                )
                for i in cutlass.range_constexpr(
                    0,
                    cute.size(tTR_rdPT_cur),
                    2,
                ):
                    tTR_rdPT_cur[i], tTR_rdPT_cur[i + 1] = mul_packed_f32x2(
                        (
                            tTR_rdPT_cur[i],
                            tTR_rdPT_cur[i + 1],
                        ),
                        (
                            tTR_rST_cur[i],
                            tTR_rST_cur[i + 1],
                        ),
                    )
                tTR_rdST_cur = self.quantize(
                    tTR_rdPT_cur,
                    4,
                )
                if cutlass.const_expr(self.use_q1_small_mma):
                    dS_stage = compute_mma_dS_producer_state.index
                    tTR_cdPT_cur = tTR_cdPT_direct[None, 0, stage]
                    kv_idx = cute.get(tTR_cdPT_cur[0], mode=[0])
                    sdST[(kv_idx, 0), 0, 0, dS_stage] = tTR_rdST_cur[0]
                    # The N=8 MN-major dS^T view packs K in 16-element
                    # groups, so it cannot alias the N=16 dK operand.
                    sdS[(0, kv_idx % 16), 0, kv_idx // 16, dS_stage] = tTR_rdST_cur[0]
                else:
                    cute.autovec_copy(
                        tTR_rdST_cur,
                        sdS_slice[None, 0, stage],
                    )
        else:
            cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)
            for i in cutlass.range_constexpr(
                0,
                cute.size(tTR_rdPT),
                2,
            ):
                tTR_rdPT[i], tTR_rdPT[i + 1] = mul_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]),
                    (fastsilu.score_scale, fastsilu.score_scale),
                )
                tTR_rdPT[i], tTR_rdPT[i + 1] = mul_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]),
                    (tTR_rST[i], tTR_rST[i + 1]),
                )
            tTR_rdST = self.quantize(tTR_rdPT, 4)

        if cutlass.const_expr(self.use_q1_small_mma):
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.compute_sync_barrier.arrive_and_wait()

        # Release dP
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(self.use_2cta_instrs):
            with cute.arch.elect_one():
                mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
        else:
            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
        mma_compute_dP_consumer_state.advance()

        if cutlass.const_expr(not self.use_2cta_instrs):
            if cutlass.const_expr(not use_stagewise_1cta):
                dS_output = sdST if self.use_q1_small_mma else sdS
                sdS_slice = dS_output[
                    None,
                    None,
                    None,
                    compute_mma_dS_producer_state.index,
                ]
                thread_layout = cute.make_ordered_layout(
                    (self.tile_n, self.tile_m),
                    (1, 0),
                )
                sdS_slice_tmp = cute.composition(
                    sdS_slice,
                    thread_layout,
                )
                sdS_slice_p = cute.composition(
                    sdS_slice_tmp[dp_idx, None],
                    cute.make_layout(tTR_cdPT_p.shape),
                )
                sdS_slice = split_wg(
                    sdS_slice_p,
                    num_warp_groups,
                    wg_idx,
                )
                cute.autovec_copy(tTR_rdST, sdS_slice)
        else:
            cute.arch.fence_view_async_shared()
            self.compute_sync_barrier.arrive_and_wait()
            stage_copy_bytes = self.tile_n * self.tile_m // 2 * self.element_dtype.width // 8
            stage_copy_elems = self.tile_n * self.tile_m // 2
            if dp_idx == 0:
                peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
                cute.arch.mbarrier_arrive_and_expect_tx(
                    dS_cluster_full_mbar_ptr,
                    stage_copy_bytes,
                    peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                )
                cpasync_bulk_s2cluster(
                    sdS_xchg.iterator,
                    sdS.iterator + cta_rank_in_cluster * stage_copy_elems,
                    dS_cluster_full_mbar_ptr,
                    stage_copy_bytes,
                    peer_cta_rank_in_cluster,
                )

        if cutlass.const_expr(not self.use_2cta_instrs):
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

        return (
            mma_compute_S_consumer_state,
            compute_mma_P_producer_state,
            mma_compute_dP_consumer_state,
            compute_mma_dS_producer_state,
        )

    @cute.jit
    def reduce_persistent(
        self,
        tiled_mma_dQ: cute.TiledMma,
        tdQtdQ: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        tma_tensor_dQ_acc: cute.Tensor,
        dQ_acc: cute.Tensor,
        dQ: cute.Tensor,
        sdQ: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        alpha: Float32,
        scaling_seqlen: Float32,
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        if cutlass.const_expr(self.use_q_major_scheduler):
            self.reduce_persistent_qmajor(
                tiled_mma_dQ,
                tdQtdQ,
                dQ,
                sdQ,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                alpha,
                scaling_seqlen,
                pipeline_args,
                block_info,
                block_sparse_tensors,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            return

        make_pipeline_state = pipeline.make_pipeline_state if self.use_2cta_instrs else make_compact_pipeline_state
        mma_reduce_dQ_consumer_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        reduce_tma_store_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.sdQaccum_stage,
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            (
                blk_coord,
                _,
                _,
                _,
                process_tile,
            ) = self.get_work_context(
                work_tile,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                block_sparse_tensors,
            )
            if process_tile:
                (
                    mma_reduce_dQ_consumer_state,
                    reduce_tma_store_producer_state,
                ) = self.reduce_work(
                    tiled_mma_dQ,
                    tdQtdQ,
                    tma_atom_dQ_acc,
                    tma_tensor_dQ_acc,
                    dQ_acc,
                    sdQ,
                    blk_coord,
                    scaling_seqlen,
                    pipeline_args,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    mma_reduce_dQ_consumer_state,
                    reduce_tma_store_producer_state,
                )
            work_tile = tile_scheduler.advance_to_next_work()
        _, reduce_tma_store_pipeline = pipeline_args
        reduce_tma_store_pipeline.producer_tail()

    @cute.jit
    def reduce_persistent_qmajor(
        self,
        tiled_mma_dQ: cute.TiledMma,
        tdQtdQ: cute.Tensor,
        dQ: cute.Tensor,
        sdQ: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        alpha: Float32,
        scaling_seqlen: Float32,
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """Accumulate the one valid dQ row in registers across K tiles."""

        tidx, _, _ = cute.arch.thread_idx()
        mma_reduce_dQ_pipeline, _ = pipeline_args
        mma_reduce_dQ_consumer_state = make_compact_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.single_stage,
        )
        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1 if self.use_q1_small_mma else self.dq_tile_m // 4)),
            self.acc_dtype,
        )
        tdQtdQ = tdQtdQ[(None, None), 0, 0]
        tiled_t2r = tcgen05.make_tmem_copy(load_op, tdQtdQ)
        thr_t2r = tiled_t2r.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.mma_tiler_dsk[0], self.mma_tiler_dsk[1]))
        tTR_cdQ = thr_t2r.partition_D(cdQ)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ)
        tTR_sdQ = thr_t2r.partition_D(sdQ)

        rdQ_acc = cute.make_rmem_tensor(
            cute.make_layout((self.qmajor_dq_stages,)),
            self.acc_dtype,
        )
        for stage_idx in cutlass.range_constexpr(self.qmajor_dq_stages):
            rdQ_acc[stage_idx] = Float32(0.0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            (
                blk_coord,
                _,
                problem_shape_cur_batch,
                _,
                process_tile,
            ) = self.get_work_context(
                work_tile,
                problem_shape,
                cu_seqlens_q,
                cu_seqlens_k,
                block_sparse_tensors,
            )
            if process_tile:
                mma_reduce_dQ_consumer_state = self.accumulate_dq_qmajor(
                    sdQ,
                    rdQ_acc,
                    tTR_cdQ,
                    tiled_t2r,
                    tTR_tdQ,
                    tTR_sdQ,
                    mma_reduce_dQ_pipeline,
                    mma_reduce_dQ_consumer_state,
                )
                is_last_k_tile = Boolean((blk_coord[1] + 1) * self.tile_n >= problem_shape_cur_batch[1])
                if is_last_k_tile:
                    _, _, _, blk_coord_batch = blk_coord
                    blk_coord_h, blk_coord_b = blk_coord_batch
                    seqlen_obj = SeqlenInfoCls(blk_coord_b)
                    self.store_dq_qmajor(
                        dQ,
                        rdQ_acc,
                        seqlen_obj.offset_q,
                        blk_coord_h,
                        blk_coord_b,
                        alpha / scaling_seqlen,
                    )
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def reduce_work(
        self,
        tiled_mma_dQ: cute.TiledMma,
        tdQtdQ: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        tma_tensor_dQ_acc: cute.Tensor,
        dQ_acc: cute.Tensor,
        sdQ: cute.Tensor,
        blk_coord: cute.Coord,
        scaling_seqlen: Float32,
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        mma_reduce_dQ_consumer_state: cutlass.pipeline.PipelineState,
        reduce_tma_store_producer_state: cutlass.pipeline.PipelineState,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        blk_coord_h, blk_coord_b = blk_coord_batch
        blk_coord_h_r, blk_coord_h_k = blk_coord_h
        seqlen_obj = SeqlenInfoCls(blk_coord_b)

        n_block = blk_coord_k
        mask_block_cnt = None
        mask_block_idx = None
        full_block_cnt = None
        full_block_idx = None
        if cutlass.const_expr(self.use_auto_block_metadata):
            (
                m_block_max,
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
            ) = get_q2k_block_sparse_consumer_row(
                block_sparse_tensors,
                blk_coord_b,
                n_block,
            )
            m_block_min = Int32(0)
        else:
            m_block_min, m_block_max, _ = block_info.get_m_block_info(
                seqlen_obj,
                n_block // self.cta_group_size,
            )

        mma_reduce_dQ_pipeline, reduce_tma_store_pipeline = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)),
            self.acc_dtype,
        )

        mdQ_acc_cur = cute.domain_offset(
            (seqlen_obj.offset_q, 0, (0, 0)),
            tma_tensor_dQ_acc,
        )
        gdQ = None
        gdQ_acc_ptr = None
        if cutlass.const_expr(self.use_2cta_instrs):
            dQ_acc_head_stride = cutlass.Int64(cute.size(dQ_acc.shape[0])) * cutlass.Int64(self.tile_hdim)
            dQ_acc_head_idx = cutlass.Int64(blk_coord_h_r) + (cutlass.Int64(blk_coord_h_k) * cutlass.Int64(cute.size(dQ_acc.shape[2][0])))
            dQ_acc_offset = cutlass.Int64(seqlen_obj.padded_offset_q) * cutlass.Int64(self.tile_hdim) + dQ_acc_head_idx * dQ_acc_head_stride
            gdQ_acc_ptr = dQ_acc.iterator + dQ_acc_offset
        else:
            gdQ = cute.local_tile(
                mdQ_acc_cur,
                (self.mma_tiler_kq[1], 32),
                (None, None, None),
            )

        cdQ_identity = cute.make_identity_tensor((self.mma_tiler_dsk[0], self.mma_tiler_dsk[1]))
        cdQ = tiled_mma_dQ.partition_C(cdQ_identity) if self.use_2cta_instrs else cdQ_identity

        thread_idx = tidx % (self.num_compute_warps * cute.arch.WARP_SIZE)

        if cutlass.const_expr(not self.use_2cta_instrs):
            tdQtdQ = tdQtdQ[(None, None), 0, 0]

        tiled_t2r = tcgen05.make_tmem_copy(load_op, tdQtdQ)
        thr_t2r = tiled_t2r.get_slice(thread_idx)

        tTR_cdQ = thr_t2r.partition_D(cdQ)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ)
        dQ_r2s_copy = None
        tdQsdQ = None
        tdQgdQ = None
        tTR_sdQ = None
        if cutlass.const_expr(self.use_2cta_instrs):
            dQ_r2s_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.acc_dtype,
                num_bits_per_copy=128,
            )
            dQ_r2s_copy = cute.make_tiled_copy_tv(
                dQ_r2s_atom,
                cute.make_layout(self.num_reduce_warps * cute.arch.WARP_SIZE),
                cute.make_layout(128 // self.acc_dtype.width),
            ).get_slice(thread_idx)
            tdQsdQ = dQ_r2s_copy.partition_D(sdQ)
        else:
            tTR_sdQ = thr_t2r.partition_D(sdQ)
            tdQsdQ, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_dQ_acc,
                0,
                cute.make_layout(1),
                cute.group_modes(sdQ, 0, 2),
                cute.group_modes(gdQ, 0, 2),
            )

        dQ_scale = Float32(1.0) if self.use_deferred_ds_scale else 1.0 / scaling_seqlen

        m_block = m_block_min
        reduce_pipeline_args = (
            mma_reduce_dQ_pipeline,
            reduce_tma_store_pipeline,
        )
        while m_block < m_block_max:
            if cutlass.const_expr(self.use_auto_block_metadata):
                m_block_valid = self.get_block_sparse_m_block(
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                    m_block,
                )
            else:
                m_block_valid = m_block
            (
                mma_reduce_dQ_consumer_state,
                reduce_tma_store_producer_state,
            ) = self.store_dq_step(
                m_block_valid,
                reduce_pipeline_args,
                mma_reduce_dQ_consumer_state,
                tTR_cdQ,
                tiled_t2r,
                tTR_tdQ,
                reduce_tma_store_producer_state,
                tTR_sdQ,
                tdQsdQ,
                tdQgdQ,
                dQ_r2s_copy,
                sdQ,
                gdQ_acc_ptr,
                tma_atom_dQ_acc,
                blk_coord_h,
                warp_idx,
                dQ_scale,
            )
            m_block += 1
        return mma_reduce_dQ_consumer_state, reduce_tma_store_producer_state

    @cute.jit
    def accumulate_dq_qmajor(
        self,
        sdQ: cute.Tensor,
        rdQ_acc: cute.Tensor,
        tTR_cdQ: cute.Tensor,
        tiled_t2r: cute.TiledCopy,
        tTR_tdQ: cute.Tensor,
        tTR_sdQ: cute.Tensor,
        mma_reduce_dQ_pipeline,
        mma_reduce_dQ_consumer_state: cutlass.pipeline.PipelineState,
    ):
        """Fold one Tensor-Core dQ tile's valid row into registers."""

        tidx, _, _ = cute.arch.thread_idx()
        mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)

        tTR_rdQ = cute.make_fragment_like(
            cute.make_layout(tTR_cdQ.shape),
            self.acc_dtype,
        )
        cute.copy(tiled_t2r, tTR_tdQ, tTR_rdQ)
        cute.arch.fence_view_async_tmem_load()
        mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
        mma_reduce_dQ_consumer_state.advance()

        num_reduce_stages = self.qmajor_dq_stages if self.use_q1_small_mma else cute.size(tTR_cdQ, mode=[2])
        for stage_idx in cutlass.range(0, num_reduce_stages, unroll_full=True):
            if cutlass.const_expr(self.use_q1_small_mma):
                # M128N8 assigns output row `tidx` to reducer thread `tidx`.
                # Vectorize its eight N values into shared memory; only N=0
                # is a real query row for the qlen=1 specialization.
                thread_layout = cute.make_ordered_layout(
                    (self.tile_hdim, self.mma_tiler_dsk[1]),
                    (0, 1),
                )
                sdQ_slice_tmp = cute.composition(sdQ[None, None, 0], thread_layout)
                sdQ_slice = cute.composition(
                    sdQ_slice_tmp[tidx, None],
                    cute.make_layout(tTR_cdQ.shape),
                )
                cute.autovec_copy(tTR_rdQ, sdQ_slice)
            else:
                tTR_rdQ_stage = tTR_rdQ[None, None, stage_idx]
                tTR_sdQ_stage = tTR_sdQ[None, None, 0, 0]
                cute.autovec_copy(
                    cute.make_tensor(
                        tTR_rdQ_stage.iterator,
                        tTR_sdQ_stage.layout,
                    ),
                    tTR_sdQ_stage,
                )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.reduce_sync_barrier.arrive_and_wait()

            if tidx < self.qmajor_dq_ncol:
                if cutlass.const_expr(self.use_q1_small_mma):
                    rdQ_acc[stage_idx] += sdQ[tidx, 0, 0]
                else:
                    rdQ_acc[stage_idx] += sdQ[0, tidx, 0]

            # All readers must finish before the next TMEM slice reuses sdQ.
            self.reduce_sync_barrier.arrive_and_wait()

        return mma_reduce_dQ_consumer_state

    @cute.jit
    def store_dq_qmajor(
        self,
        dQ: cute.Tensor,
        rdQ_acc: cute.Tensor,
        q_offset: Int32,
        blk_coord_head: Tuple[Int32, Int32],
        batch_idx: Int32,
        dQ_scale: Float32,
    ):
        """Write the accumulated single Q row once, without a workspace."""

        tidx, _, _ = cute.arch.thread_idx()
        dQ_item = domain_offset_i64(
            (q_offset, 0, (blk_coord_head, batch_idx)),
            dQ,
        )
        dQ_row = dQ_item[0, None, ((0, 0), 0)]
        if tidx < self.qmajor_dq_ncol:
            for stage_idx in cutlass.range_constexpr(self.qmajor_dq_stages):
                dim_idx = stage_idx * self.qmajor_dq_ncol + tidx
                dQ_row[dim_idx] = dQ.element_type(rdQ_acc[stage_idx] * dQ_scale)

    @cute.jit
    def store_dq_step(
        self,
        m_block_valid: Int32,
        pipeline_args: tuple,
        mma_reduce_dQ_consumer_state: cutlass.pipeline.PipelineState,
        tTR_cdQ: cute.Tensor,
        tiled_t2r: cute.TiledCopy,
        tTR_tdQ: cute.Tensor,
        reduce_tma_store_producer_state: cutlass.pipeline.PipelineState,
        tTR_sdQ: Optional[cute.Tensor],
        tdQsdQ: Optional[cute.Tensor],
        tdQgdQ: Optional[cute.Tensor],
        dQ_r2s_copy: Optional[cute.TiledCopy],
        sdQ: cute.Tensor,
        gdQ_acc_ptr: Optional[cute.Pointer],
        tma_atom_dQ_acc: cute.CopyAtom,
        blk_coord_head: Tuple[Int32, Int32],
        warp_idx: Int32,
        dQ_scale: Float32,
    ):
        mma_reduce_dQ_pipeline, reduce_tma_store_pipeline = pipeline_args
        mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)

        tTR_rdQ = cute.make_fragment_like(
            cute.make_layout(tTR_cdQ.shape),
            self.acc_dtype,
        )
        cute.copy(tiled_t2r, tTR_tdQ, tTR_rdQ)
        if cutlass.const_expr(not self.use_deferred_ds_scale):
            for i in cutlass.range(
                0,
                cute.size(tTR_rdQ),
                2,
                unroll_full=True,
            ):
                tTR_rdQ[i], tTR_rdQ[i + 1] = mul_packed_f32x2(
                    (tTR_rdQ[i], tTR_rdQ[i + 1]),
                    (dQ_scale, dQ_scale),
                )
        cute.arch.fence_view_async_tmem_load()

        if cutlass.const_expr(self.use_2cta_instrs):
            with cute.arch.elect_one():
                mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
        else:
            mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
        mma_reduce_dQ_consumer_state.advance()

        # We don't have enough smem to dump it all to smem, so we do it in stages
        num_reduce_stages = self.dQaccum_reduce_stage if self.use_2cta_instrs else cute.size(tTR_cdQ, mode=[2])
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        stage_offset = num_reduce_stages * cta_rank_in_cluster if self.use_2cta_instrs else Int32(0)
        tTR_rdQ_staged = (
            cute.make_tensor(
                tTR_rdQ.iterator,
                cute.make_layout((self.dQ_reduce_ncol, num_reduce_stages)),
            )
            if self.use_2cta_instrs
            else tTR_rdQ
        )
        for i in cutlass.range(0, num_reduce_stages, unroll_full=True):
            if cutlass.const_expr(not self.use_2cta_instrs):
                if warp_idx == 0:
                    reduce_tma_store_pipeline.producer_acquire()
                # Wait in all threads for the acquire to complete.
                self.reduce_sync_barrier.arrive_and_wait()

            tTR_rdQ_stage = tTR_rdQ_staged[None, i] if self.use_2cta_instrs else tTR_rdQ_staged[None, None, i]
            if cutlass.const_expr(self.use_2cta_instrs):
                assert dQ_r2s_copy is not None
                assert tdQsdQ is not None
                tdQsdQ_stage = tdQsdQ[
                    None,
                    None,
                    reduce_tma_store_producer_state.index,
                ]
                tdQrdQ_stage = cute.make_tensor(
                    tTR_rdQ_stage.iterator,
                    tdQsdQ_stage.shape,
                )
                cute.copy(
                    dQ_r2s_copy,
                    tdQrdQ_stage,
                    tdQsdQ_stage,
                )
            else:
                assert tTR_sdQ is not None
                tTR_sdQ_stage = tTR_sdQ[
                    None,
                    None,
                    0,
                    reduce_tma_store_producer_state.index,
                ]
                cute.autovec_copy(
                    cute.make_tensor(
                        tTR_rdQ_stage.iterator,
                        tTR_sdQ_stage.layout,
                    ),
                    tTR_sdQ_stage,
                )

            # Wait for the stores to all be visible to the TMA
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.reduce_sync_barrier.arrive_and_wait()

            if warp_idx == 0:
                if cutlass.const_expr(self.use_2cta_instrs):
                    assert gdQ_acc_ptr is not None
                    with cute.arch.elect_one():
                        cpasync_reduce_bulk_add_f32(
                            sdQ.iterator + reduce_tma_store_producer_state.index * self.tile_m * self.dQ_reduce_ncol,
                            gdQ_acc_ptr + m_block_valid * self.tile_m * self.tile_hdim + (i + stage_offset) * self.tile_m * self.dQ_reduce_ncol,
                            self.tile_m * self.dQ_reduce_ncol * self.acc_dtype.width // 8,
                        )
                else:
                    assert tdQsdQ is not None
                    assert tdQgdQ is not None
                    cute.copy(
                        tma_atom_dQ_acc,
                        tdQsdQ[
                            None,
                            reduce_tma_store_producer_state.index,
                        ],
                        tdQgdQ[
                            None,
                            m_block_valid,
                            i + stage_offset,
                            blk_coord_head,
                        ],
                    )

                reduce_tma_store_pipeline.producer_commit()
                if cutlass.const_expr(self.use_2cta_instrs):
                    reduce_tma_store_pipeline.producer_acquire()

            if cutlass.const_expr(self.use_2cta_instrs):
                # Wait in all threads before reusing a shared-memory stage.
                self.reduce_sync_barrier.arrive_and_wait()
            reduce_tma_store_producer_state.advance()

        return mma_reduce_dQ_consumer_state, reduce_tma_store_producer_state

    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
    ) -> cute.Tensor:
        output = cute.make_fragment_like(cute.make_layout(input.shape), self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.element_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)

        tCg = thr_copy.partition_D(gmem)
        tCr = thr_copy.partition_S(self.quantize(regs, 4))
        tPc = thr_copy.partition_D(coord)

        preds_shape = (tPc.shape[0][1], tPc.shape[1], tPc.shape[2], tPc.shape[3])
        preds = cute.make_fragment_like(cute.make_layout(preds_shape), Boolean)

        tPc_fake = cute.group_modes(tPc, 1, 4)
        preds_shape_fake = (tPc_fake.shape[0][1], cute.size(tPc_fake, mode=[1]))
        preds_fake = cute.make_tensor(preds.iterator, preds_shape_fake)

        for i in cutlass.range_constexpr(0, cute.size(preds_fake, mode=[0])):
            for j in cutlass.range_constexpr(0, cute.size(preds_fake, mode=[1])):
                lhs = tPc_fake[(0, i), j]
                val = cute.elem_less(lhs, tensor_shape)
                preds_fake[i, j] = val

        cute.copy(copy_atom, tCr, tCg, pred=preds)

    @cute.jit
    def epilogue_clear(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: cute.Shape,
        dK: cute.Tensor,
        dV: cute.Tensor,
    ):
        """Write defined zero gradients for an EMPTY K owner tile."""
        tidx, _, _ = cute.arch.thread_idx()
        _, K, _, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        mdK = cute.make_tensor(
            dK.iterator + blk_offset[1] * dK.stride[0],
            cute.make_layout((K, self.tile_hdim, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(mdK, (self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_n, 0),
            cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1])),
        )

        mdV = cute.make_tensor(
            dV.iterator + blk_offset[1] * dV.stride[0],
            cute.make_layout((K, self.tile_hdim, HB), stride=dV.stride),
        )
        gdV = cute.local_tile(
            mdV,
            (self.tile_n, self.tile_hdim),
            (None, None, None),
        )
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]
        cdV = cute.domain_offset(
            (blk_coord_k * self.tile_n, 0),
            cute.make_identity_tensor((self.tile_n, self.tile_hdim)),
        )

        # The eight compute warps cooperatively clear the complete owner tile.
        num_threads = self.num_compute_warps * cute.arch.WARP_SIZE
        vec_size = 128 // self.element_dtype.width
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        gmem_tiled_copy = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout((num_threads, 1)),
            cute.make_layout((1, vec_size)),
        )
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx - 128)
        tdKgdK = gmem_thr_copy.partition_D(gdK)
        tdVgdV = gmem_thr_copy.partition_D(gdV)
        tdKcdK = gmem_thr_copy.partition_D(cdK)
        tdVcdV = gmem_thr_copy.partition_D(cdV)
        zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
        zero.fill(0.0)

        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
            if cute.elem_less(
                tdKcdK[0, i, 0],
                cute.select(problem_shape, mode=[1, 2]),
            ):
                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                    cute.copy(
                        gmem_tiled_copy,
                        zero,
                        tdKgdK[None, i, j],
                    )
        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
            if cute.elem_less(
                tdVcdV[0, i, 0],
                cute.select(problem_shape, mode=[1, 2]),
            ):
                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                    cute.copy(
                        gmem_tiled_copy,
                        zero,
                        tdVgdV[None, i, j],
                    )

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        scaling_seqlen: Float32,
        alpha: Float32,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        tma_tensor_dK: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        tma_tensor_dV: cute.Tensor,
        dKV_r2s_copy: cute.TiledCopy,
        sdK_epi: cute.Tensor,
        sdV_epi: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        thr_mma_dK: cute.TiledMma,
        thr_mma_dV: cute.TiledMma,
        # (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state)
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        _, K, D, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord
        mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )

        if cutlass.const_expr(not self.use_2cta_instrs):
            tdKtdK = tdKtdK[(None, None), 0, 0]

        mdK = cute.make_tensor(
            dK.iterator
            + cute.assume(
                blk_offset[1] * dK.stride[0],
                divby=128 // self.element_dtype.width,
            ),
            cute.make_layout((K, self.tile_hdim, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(
            mdK,
            self.mma_tiler_dsq[:2],
            (None, None, None),
        )
        gdK = gdK[
            None,
            None,
            blk_coord_k // self.cta_group_size,
            0,
            blk_coord_batch,
        ]

        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_n, 0),
            cute.make_identity_tensor((self.tile_n, self.tile_hdim)),
        )

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * cute.arch.WARP_SIZE)) // 128

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        gmem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        tiled_gmem_store_dK = cute.make_tiled_copy(
            gmem_store_atom,
            layout_tv=tiled_t2r_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_t2r_dK.tiler_mn,
        )

        cdK_tmem = (
            thr_mma_dK.partition_C(
                cute.domain_offset(
                    (
                        blk_coord_k // self.cta_group_size * self.mma_tiler_dsq[0],
                        0,
                    ),
                    cute.make_identity_tensor(self.mma_tiler_dsq[:2]),
                )
            )
            if self.use_2cta_instrs
            else cdK
        )
        tTR_cdK = thread_t2r_dK.partition_D(cdK_tmem)
        tTR_cdK = (
            split_wg_mma(tTR_cdK, num_warp_groups, wg_idx)
            if self.use_2cta_instrs
            else split_wg_contiguous(
                tTR_cdK,
                num_warp_groups,
                wg_idx,
            )
        )
        tdKgdK = thr_mma_dK.partition_C(gdK) if self.use_2cta_instrs else gdK
        tTR_gdK = thread_t2r_dK.partition_D(tdKgdK)
        tTR_gdK = (
            split_wg_mma(tTR_gdK, num_warp_groups, wg_idx)
            if self.use_2cta_instrs
            else split_wg_contiguous(
                tTR_gdK,
                num_warp_groups,
                wg_idx,
            )
        )
        tTR_rdK = cute.make_fragment_like(cute.make_layout(tTR_cdK.shape), self.acc_dtype)
        tTR_tdK = thread_t2r_dK.partition_S(tdKtdK)
        tTR_tdK = (
            split_wg_mma(tTR_tdK, num_warp_groups, wg_idx)
            if self.use_2cta_instrs
            else split_wg_contiguous(
                tTR_tdK,
                num_warp_groups,
                wg_idx,
            )
        )

        mdV_in = cute.make_tensor(dV.iterator, cute.make_layout((K, self.cta_tiler[2], HB), stride=dV.stride))
        mdV = cute.make_tensor(
            mdV_in.iterator
            + cute.assume(
                blk_offset[1] * mdV_in.stride[0],
                divby=128 // self.element_dtype.width,
            ),
            mdV_in.layout,
        )
        gdV = cute.local_tile(
            mdV,
            self.mma_tiler_pdo[:2],
            (None, None, None),
        )
        gdV = gdV[
            None,
            None,
            blk_coord_k // self.cta_group_size,
            0,
            blk_coord_batch,
        ]

        cdV = cute.domain_offset(
            (blk_coord_k * self.cta_tiler[0], 0),
            cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1])),
        )

        mdK_tma = cute.domain_offset(
            cute.select(blk_offset, mode=[1, 2, 3]),
            tma_tensor_dK,
        )
        gdK_tma = cute.local_tile(
            mdK_tma,
            (self.tile_n, self.tile_hdim),
            (None, None, None),
        )
        gdK_tma = gdK_tma[
            None,
            None,
            blk_coord_k,
            0,
            blk_coord_batch,
        ]

        mdV_tma = cute.domain_offset(
            cute.select(blk_offset, mode=[1, 2, 3]),
            tma_tensor_dV,
        )
        gdV_tma = cute.local_tile(
            mdV_tma,
            (self.tile_n, self.tile_hdim),
            (None, None, None),
        )
        gdV_tma = gdV_tma[
            None,
            None,
            blk_coord_k,
            0,
            blk_coord_batch,
        ]

        gdK_tma = cute.logical_divide(
            gdK_tma,
            (
                cute.product_each(gdK_tma.shape)[0],
                cute.product_each(gdK_tma.shape)[1] // num_warp_groups,
            ),
        )[None, (None, wg_idx)]
        gdV_tma = cute.logical_divide(
            gdV_tma,
            (
                cute.product_each(gdV_tma.shape)[0],
                cute.product_each(gdV_tma.shape)[1] // num_warp_groups,
            ),
        )[None, (None, wg_idx)]
        gdK_tma_epi = cute.local_tile(
            gdK_tma,
            self.dKV_epi_tile,
            (0, None),
        )
        gdV_tma_epi = cute.local_tile(
            gdV_tma,
            self.dKV_epi_tile,
            (0, None),
        )
        sdK_epi_wg = sdK_epi[None, None, wg_idx]
        sdV_epi_wg = sdV_epi[None, None, wg_idx]

        thr_r2s_dKV = dKV_r2s_copy.get_slice(dp_idx)
        tTR_sdK_epi = thr_r2s_dKV.partition_D(sdK_epi_wg)
        tTR_sdV_epi = thr_r2s_dKV.partition_D(sdV_epi_wg)

        tdKsdK_epi, tdKgdK_epi = cpasync.tma_partition(
            tma_atom_dK,
            0,
            cute.make_layout(1),
            cute.group_modes(sdK_epi_wg, 0, 2),
            cute.group_modes(gdK_tma_epi, 0, 2),
        )
        tdVsdV_epi, tdVgdV_epi = cpasync.tma_partition(
            tma_atom_dV,
            0,
            cute.make_layout(1),
            cute.group_modes(sdV_epi_wg, 0, 2),
            cute.group_modes(gdV_tma_epi, 0, 2),
        )

        if cutlass.const_expr(not self.use_2cta_instrs):
            tdVtdV = tdVtdV[(None, None), 0, 0]

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)
        tiled_gmem_store_dV = cute.make_tiled_copy(
            gmem_store_atom,
            layout_tv=tiled_t2r_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_t2r_dV.tiler_mn,
        )

        cdV_tmem = (
            thr_mma_dV.partition_C(
                cute.domain_offset(
                    (
                        blk_coord_k // self.cta_group_size * self.mma_tiler_pdo[0],
                        0,
                    ),
                    cute.make_identity_tensor(self.mma_tiler_pdo[:2]),
                )
            )
            if self.use_2cta_instrs
            else cdV
        )
        tTR_cdV = thread_t2r_dV.partition_D(cdV_tmem)
        tTR_cdV = (
            split_wg_mma(tTR_cdV, num_warp_groups, wg_idx)
            if self.use_2cta_instrs
            else split_wg_contiguous(
                tTR_cdV,
                num_warp_groups,
                wg_idx,
            )
        )
        tdVgdV = thr_mma_dV.partition_C(gdV) if self.use_2cta_instrs else gdV
        tTR_gdV = thread_t2r_dV.partition_D(tdVgdV)
        tTR_gdV = (
            split_wg_mma(tTR_gdV, num_warp_groups, wg_idx)
            if self.use_2cta_instrs
            else split_wg_contiguous(
                tTR_gdV,
                num_warp_groups,
                wg_idx,
            )
        )
        tTR_rdV = cute.make_fragment_like(cute.make_layout(tTR_cdV.shape), self.acc_dtype)
        tTR_tdV = thread_t2r_dV.partition_S(tdVtdV)
        tTR_tdV = (
            split_wg_mma(tTR_tdV, num_warp_groups, wg_idx)
            if self.use_2cta_instrs
            else split_wg_contiguous(
                tTR_tdV,
                num_warp_groups,
                wg_idx,
            )
        )

        inv_scaling_seqlen = 1.0 / scaling_seqlen
        dK_scale = inv_scaling_seqlen * alpha if self.use_deferred_ds_scale else inv_scaling_seqlen
        full_k_tile = (blk_coord_k + 1) * self.tile_n <= K
        if cutlass.const_expr(self.use_q_major_scheduler):
            full_k_tile = Boolean(False)

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        # Load tdVtdV
        cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)

        for i in cutlass.range(0, cute.size(tTR_rdV), 2, unroll_full=True):
            tTR_rdV[i], tTR_rdV[i + 1] = mul_packed_f32x2((tTR_rdV[i], tTR_rdV[i + 1]), (inv_scaling_seqlen, inv_scaling_seqlen))

        tTR_rdV_epi = cute.make_rmem_tensor(tTR_rdV.shape, self.element_dtype)
        tTR_rdV_epi.store(tTR_rdV.load().to(self.element_dtype))
        if cutlass.const_expr(self.use_q_major_scheduler):
            self.store(
                tTR_gdV,
                tTR_rdV,
                tTR_cdV,
                (K, D),
            )
        elif full_k_tile:
            tTR_rdV_epi_smem = cute.make_tensor(
                tTR_rdV_epi.iterator,
                tTR_sdV_epi.shape,
            )
            cute.copy(
                thr_r2s_dKV,
                tTR_rdV_epi_smem,
                tTR_sdV_epi,
            )
        else:
            if cutlass.const_expr(self.use_2cta_instrs):
                if dp_idx < K - self.tile_n * blk_coord_k:
                    cute.copy(tiled_gmem_store_dV, tTR_rdV_epi, tTR_gdV)
            else:
                self.store(
                    tTR_gdV,
                    tTR_rdV,
                    tTR_cdV,
                    (K, D),
                )

        cute.arch.fence_view_async_tmem_load()

        if cutlass.const_expr(self.use_2cta_instrs):
            with cute.arch.elect_one():
                mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        else:
            mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        # Load tdKtdK
        cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

        for i in cutlass.range(0, cute.size(tTR_rdK), 2, unroll_full=True):
            tTR_rdK[i], tTR_rdK[i + 1] = mul_packed_f32x2(
                (tTR_rdK[i], tTR_rdK[i + 1]),
                (dK_scale, dK_scale),
            )

        tTR_rdK_epi = cute.make_rmem_tensor(tTR_rdK.shape, self.element_dtype)
        tTR_rdK_epi.store(tTR_rdK.load().to(self.element_dtype))
        if full_k_tile:
            tTR_rdK_epi_smem = cute.make_tensor(
                tTR_rdK_epi.iterator,
                tTR_sdK_epi.shape,
            )
            cute.copy(
                thr_r2s_dKV,
                tTR_rdK_epi_smem,
                tTR_sdK_epi,
            )
        else:
            if cutlass.const_expr(self.use_2cta_instrs):
                if dp_idx < K - self.tile_n * blk_coord_k:
                    cute.copy(tiled_gmem_store_dK, tTR_rdK_epi, tTR_gdK)
            else:
                self.store(tTR_gdK, tTR_rdK, tTR_cdK, (K, D))

        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(self.use_2cta_instrs):
            with cute.arch.elect_one():
                mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        else:
            mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

        if full_k_tile:
            cute.arch.fence_view_async_shared()
            self.compute_sync_barrier.arrive_and_wait()
            if warp_idx % 4 == 0:
                if cutlass.const_expr(not self.use_q_major_scheduler):
                    cute.copy(
                        tma_atom_dV,
                        tdVsdV_epi,
                        tdVgdV_epi[None, 0],
                    )
                cute.copy(
                    tma_atom_dK,
                    tdKsdK_epi,
                    tdKgdK_epi[None, 0],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        return mma_compute_dKdV_consumer_state

    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        workspace: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        D = problem_shape[2]
        H_r, H_k = problem_shape[3][0]
        D = cute.round_up(D, 8)

        dQ_acc_iter = workspace.iterator

        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype)

        total_q = cute.size(workspace.shape[1])
        head_stride = cutlass.Int64(D) * cutlass.Int64(total_q)
        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (total_q, D, (H_r, H_k)),
                stride=(
                    D,
                    1,
                    (
                        head_stride,
                        head_stride * cutlass.Int64(H_r),
                    ),
                ),
            ),
        )

        return dQ_acc

    def make_and_init_load_mma_Q_pipeline(
        self,
        load_mma_Q_mbar_ptr,
        cluster_layout_vmnk,
    ):
        load_mma_Q_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_load_mma_Qt_pipeline(
        self,
        load_mma_Qt_mbar_ptr,
        cluster_layout_vmnk,
    ):
        load_mma_Qt_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.load_warp_id]),
        )
        load_mma_Qt_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Qt_mbar_ptr,
            num_stages=self.Q_stage,
            producer_group=load_mma_Qt_producer_group,
            consumer_group=load_mma_Qt_consumer_group,
            tx_count=self.tma_copy_Qt_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_load_mma_Kt_pipeline(
        self,
        load_mma_Kt_mbar_ptr,
        cluster_layout_vmnk,
    ):
        load_mma_Kt_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.load_warp_id]),
        )
        load_mma_Kt_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Kt_mbar_ptr,
            num_stages=self.single_stage,
            producer_group=load_mma_Kt_producer_group,
            consumer_group=load_mma_Kt_consumer_group,
            tx_count=(self.tma_copy_K_bytes if self.use_q_major_scheduler else self.tma_copy_Kt_bytes),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_load_mma_dO_pipeline(
        self,
        load_mma_dO_mbar_ptr,
        cluster_layout_vmnk,
    ):
        load_mma_dO_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_dO_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_dO_mbar_ptr,
            num_stages=self.dO_stage,
            producer_group=load_mma_dO_producer_group,
            consumer_group=load_mma_dO_consumer_group,
            tx_count=self.tma_copy_dO_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_compute_S_pipeline(
        self,
        mma_compute_S_mbar_ptr,
        cluster_layout_vmnk,
    ):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * (1 if self.use_2cta_instrs else cute.arch.WARP_SIZE) * self.cta_group_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.single_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_compute_dP_pipeline(
        self,
        mma_compute_dP_mbar_ptr,
        cluster_layout_vmnk,
    ):
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * (1 if self.use_2cta_instrs else cute.arch.WARP_SIZE) * self.cta_group_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.single_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_reduce_dQ_pipeline(
        self,
        mma_reduce_dQ_mbar_ptr,
        cluster_layout_vmnk,
    ):
        mma_reduce_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_reduce_dQ_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * (1 if self.use_2cta_instrs else cute.arch.WARP_SIZE) * self.cta_group_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_reduce_dQ_mbar_ptr,
            num_stages=self.single_stage,
            producer_group=mma_reduce_dQ_producer_group,
            consumer_group=mma_reduce_dQ_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_compute_mma_P_pipeline(
        self,
        compute_mma_P_mbar_ptr,
        cluster_layout_vmnk,
    ):
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * (1 if self.use_2cta_instrs else cute.arch.WARP_SIZE) * self.cta_group_size,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.single_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_compute_mma_dS_pipeline(
        self,
        compute_mma_dS_mbar_ptr,
        cluster_layout_vmnk,
    ):
        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * (1 if self.use_2cta_instrs else cute.arch.WARP_SIZE) * self.cta_group_size,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )

        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.single_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_compute_dKdV_pipeline(
        self,
        mma_compute_dKdV_mbar_ptr,
        cluster_layout_vmnk,
    ):
        mma_compute_dKdV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dKdV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * (1 if self.use_2cta_instrs else cute.arch.WARP_SIZE) * self.cta_group_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dKdV_mbar_ptr,
            num_stages=self.sdKVaccum_stage,
            producer_group=mma_compute_dKdV_producer_group,
            consumer_group=mma_compute_dKdV_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_reduce_tma_store_pipeline(self):
        reduce_tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * cute.arch.WARP_SIZE,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.sdQaccum_stage,
            producer_group=reduce_tma_store_producer_group,
        )
