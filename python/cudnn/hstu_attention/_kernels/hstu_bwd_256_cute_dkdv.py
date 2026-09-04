# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* cta_tiler_mn must be 64,128
* Batch size must be the same for Q, K, and V tensors
"""

from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .hstu_bwd_256_cute_utils import (
    SM100_TMEM_CAPACITY_COLUMNS,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
)
from .block_sparsity import (
    HSTUBlockSparseTensors,
    get_k2q_block_sparse_consumer_row,
    get_k2q_q_block_for_subtile_iteration,
)
from .utils import (
    fma_packed_f32x2,
    mul_packed_f32x2,
    sub_packed_f32x2,
    tanhf,
)

LAYOUT_RANK_CONSTANT = 3


@cute.jit
def split_wg(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """Split warp group."""
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == LAYOUT_RANK_CONSTANT):
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


def Tmemory_offset(lane, col):
    """Tensor memory offset."""
    return (lane << 16) + col


class BlackwellFusedMultiHeadAttentionBackwardDKDVKernel:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        cta_tiler: tuple[int, int, int],
        is_causal: bool,
        window_size_left: int | None,
        window_size_right: int | None,
        skip_residual_mask: bool = False,
        is_arbitrary: bool = False,
        func_num: int = 0,
        use_auto_block_metadata: bool = False,
    ):
        """Initialization."""
        self.acc_dtype = acc_dtype
        self.cta_tiler = cta_tiler
        self.tile_shape_Q = cta_tiler[0]
        self.tile_shape_K = cta_tiler[1]
        self.tile_shape_dQ_K = cta_tiler[2]
        self.tile_shape_dV_dO = cta_tiler[2]
        # For S
        self.KQ_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[0],
            cta_tiler[2],
        )
        # For dP
        self.VdO_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[0],
            cta_tiler[2],
        )
        # For dV
        self.PdO_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[2],
            cta_tiler[0],
        )
        # For dK
        self.dSQ_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[2],
            cta_tiler[0],
        )
        self.cluster_shape_mn = (2, 1)
        self.is_causal = is_causal
        self.skip_residual_mask = skip_residual_mask
        self.is_arbitrary = is_arbitrary
        self.func_num = func_num
        self.use_auto_block_metadata = use_auto_block_metadata
        self.subtile_factor = 2
        self.window_size_left: int = -1 if window_size_left is None else window_size_left
        self.window_size_right: int = -1 if window_size_right is None else window_size_right
        self.has_sliding_window = not self.is_causal and (window_size_left is not None or window_size_right is not None)
        assert not self.is_arbitrary or not (self.is_causal or self.has_sliding_window)
        assert not self.is_arbitrary or (self.func_num > 0 and self.func_num % 2 == 1)
        assert not self.use_auto_block_metadata or self.is_arbitrary
        if self.is_causal:
            self.window_size_right = 0

        self.compute_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.num_compute_warps = 8

        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (self.num_compute_warps + 4)

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2
        self.epilogue_sync_bar_id = 3
        self.tmem_dK_offset = 0
        self.tmem_dV_offset = Tmemory_offset(0, cta_tiler[2] // 2)
        self.tmem_dP_offset = Tmemory_offset(0, cta_tiler[2] + cta_tiler[0] // 2)
        self.tmem_S_offset = Tmemory_offset(0, cta_tiler[2])

        self.num_regs_compute = 128
        self.num_regs_mma = 128
        self.num_regs_empty = 96
        self.num_regs_load = 96

        self.buffer_align_bytes = 128

    def _setup_attributes(self):
        """Settings for pipeline stage."""
        self.load_mma_Q_stage = 1
        self.load_mma_K_stage = 1
        self.load_mma_V_stage = 1
        self.load_mma_QT_stage = 1
        self.load_mma_dO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        cumulative_s_q: cute.Tensor,
        cumulative_s_k: cute.Tensor,
        func: cute.Tensor | None,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        scale_softmax: cutlass.Float32,
        normalization_scale: cutlass.Float32,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        stream: cuda.CUstream,
    ):
        """Host function to launch CuTeDSL kernel."""
        # Infer shape metadata from normalized 5D tensors (B, S, H_k, H_r, D).
        h_r = Q.shape[3]
        h_k = Q.shape[2]
        b = cumulative_s_q.shape[0] - 1
        problem_shape = (
            max_seqlen_q,
            max_seqlen_k,
            Q.shape[4],
            ((h_r, h_k), b),
        )
        hb = ((h_r, h_k), b)
        # (b, s, h_k, h_r, d) -> (s, d, ((h_r, h_k), b))
        Q = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[4], hb),
                stride=(
                    cute.assume(Q.stride[1], divby=64),
                    Q.stride[4],
                    (
                        (Q.shape[4], Q.shape[4] * Q.shape[3]),
                        0,
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        K = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[1], K.shape[4], hb),
                stride=(
                    cute.assume(K.stride[1], divby=64),
                    K.stride[4],
                    (
                        (0, K.shape[4]),
                        0,
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        V = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[1], V.shape[4], hb),
                stride=(
                    cute.assume(V.stride[1], divby=64),
                    V.stride[4],
                    (
                        (0, V.shape[4]),
                        0,
                    ),
                ),
            ),
        )
        # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
        QT = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[0], Q.shape[2]),
                stride=(
                    Q.stride[1],
                    Q.stride[0],
                    Q.stride[2],
                ),
            ),
        )
        dK = cute.make_tensor(dK.iterator, K.layout)
        dV = cute.make_tensor(dV.iterator, V.layout)
        # (s, d, ((h_r, h_k), b))
        dO = cute.make_tensor(dO.iterator, Q.layout)

        # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
        dOT = cute.make_tensor(
            dO.iterator,
            cute.make_layout(
                (dO.shape[1], dO.shape[0], dO.shape[2]),
                stride=(
                    dO.stride[1],
                    dO.stride[0],
                    dO.stride[2],
                ),
            ),
        )

        if cutlass.const_expr(self.use_auto_block_metadata):
            assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
            assert Q.element_type in (cutlass.Float16, cutlass.BFloat16)

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()

        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError(f"The layout of q is not supported: {self.Q_major_mode}")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        PT_source = tcgen05.OperandSource.SMEM

        # compute S
        KQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            K.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.KQ_mma_tiler[:2],
        )
        # compute dP
        VdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            V.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.VdO_mma_tiler[:2],
        )
        # compute dV
        PdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dO.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.PdO_mma_tiler[:2],
            PT_source,
        )
        # compute dK
        dSQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Q.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSQ_mma_tiler[:2],
        )
        atom_thr_size = cute.size(KQ_tiled_mma.thr_id.shape)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)  # type: ignore[assignment]
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (atom_thr_size,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            K.element_type,
            1,
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            Q.element_type,
            self.load_mma_Q_stage,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_a(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            V.element_type,
            1,
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            dO.element_type,
            self.load_mma_dO_stage,
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            Q.element_type,
            self.compute_mma_dS_stage,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            Q.element_type,
            self.load_mma_QT_stage,
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_a(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            Q.element_type,
            self.compute_mma_P_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            dO.element_type,
            self.load_mma_dO_stage,
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        QT_smem_layout = cute.select(QT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_QT, tma_tensor_QT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            QT,
            QT_smem_layout,
            self.dSQ_mma_tiler,
            dSQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        dOT_smem_layout = cute.select(dOT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dOT, tma_tensor_dOT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dOT,
            dOT_smem_layout,
            self.PdO_mma_tiler,
            PdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # for 2cta, tma_copy_QT_bytes is same as the tma_copy_Q_bytes
        self.tma_copy_Q_bytes = cute.size_in_bytes(Q.element_type, Q_smem_layout) * atom_thr_size
        self.tma_copy_K_bytes = cute.size_in_bytes(K.element_type, K_smem_layout) * atom_thr_size
        self.tma_copy_V_bytes = cute.size_in_bytes(V.element_type, V_smem_layout) * atom_thr_size
        self.tma_copy_dO_bytes = cute.size_in_bytes(dO.element_type, dO_smem_layout) * atom_thr_size

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_Q_stage * 2]
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage * 2]
            load_mma_V_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_V_stage * 2]
            load_mma_QT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_QT_stage * 2]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_dO_stage * 2]
            load_mma_dOT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_dO_stage * 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_P_stage * 2]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dKdV_stage * 2]
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar_ptr: cutlass.Int64
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[K.element_type, cute.cosize(K_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # only used in 2cta
            sV: cute.struct.Align[
                cute.struct.MemRange[V.element_type, cute.cosize(V_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQT: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(QT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dO_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdOT: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dOT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Activation tiles shared by the two-CTA pipeline.
            sP: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(P_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdST: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(dST_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # =============================== bwd ===============================
        bwd_grid = self._compute_bwd_grid(problem_shape, self.cta_tiler[1])
        bwd_grid = cute.round_up(bwd_grid, self.cluster_shape_mnk)

        self.dkdv_bwd(
            KQ_tiled_mma,
            VdO_tiled_mma,
            PdO_tiled_mma,
            dSQ_tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_QT,
            tma_tensor_QT,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dOT,
            tma_tensor_dOT,
            dK,
            dV,
            scale_softmax,
            normalization_scale,
            problem_shape,
            cumulative_s_q,
            cumulative_s_k,
            func,
            block_sparse_tensors,
            self.cluster_layout_vmnk,
            K_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            dOT_smem_layout_staged,
            P_smem_layout_staged,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore [attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def dkdv_bwd(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_QT: cute.CopyAtom,
        QT_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dOT: cute.CopyAtom,
        dOT_in: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        scale_softmax: cutlass.Float32,
        normalization_scale: cutlass.Float32,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        cumulative_s_q: cute.Tensor,
        cumulative_s_k: cute.Tensor,
        func: cute.Tensor | None,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        cluster_layout_vmnk: cute.Layout,
        K_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
    ):
        """Core CuTeDSL backward kernel."""
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_QT)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dOT)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_Q_producer, load_mma_Q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_Q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_Q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_K_producer, load_mma_K_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_K_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_K_bytes,
            barrier_storage=storage.load_mma_K_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_V_producer, load_mma_V_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_V_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_V_bytes,
            barrier_storage=storage.load_mma_V_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_QT_producer, load_mma_QT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_QT_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_QT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dO_producer, load_mma_dO_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dO_bytes,
            barrier_storage=storage.load_mma_dO_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dOT_producer, load_mma_dOT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dO_bytes,
            barrier_storage=storage.load_mma_dOT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_S_producer, mma_compute_S_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_S_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            barrier_storage=storage.mma_compute_S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dP_producer, mma_compute_dP_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dP_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            barrier_storage=storage.mma_compute_dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_P_producer, compute_mma_P_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_P_stage,
            producer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_P_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_dS_producer, compute_mma_dS_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_dS_stage,
            producer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dKdV_producer, mma_compute_dKdV_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            barrier_storage=storage.mma_compute_dKdV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        cute.arch.barrier(barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta)

        # setup mma
        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        # for 2cta, QT use different mem from Q

        sQT = storage.sQT.get_tensor(QT_smem_layout_staged.outer, swizzle=QT_smem_layout_staged.inner)
        sdST = storage.sdST.get_tensor(dST_smem_layout_staged.outer, swizzle=dST_smem_layout_staged.inner)
        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)

        sdOT = storage.sdOT.get_tensor(dOT_smem_layout_staged.outer, swizzle=dOT_smem_layout_staged.inner)

        # tSTrK shape : (MMA, MMA_M, MMA_K, STAGE)
        tSTrK = KQ_tiled_mma.make_fragment_A(sK)
        # tSTrQ shape : (MMA, MMA_N, MMA_K, STAGE)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)

        # tdPTrV shape : (MMA, MMA_M, MMA_K, STAGE)
        tdPTrV = VdO_tiled_mma.make_fragment_A(sV)
        # tdPTrdO shape : (MMA, MMA_N, MMA_K, STAGE)
        tdPTrdO = VdO_tiled_mma.make_fragment_B(sdO)

        # tdKrdST shape: (MMA, MMA_M, MMA_K, STAGE)
        tdKrdST = dSQ_tiled_mma.make_fragment_A(sdST)
        # tdKrQT shape : (MMA, MMA_N, MMA_K, STAGE)
        tdKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_cta,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.load_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        tmem.allocate(self.tmem_alloc_cols)

        # wait for tmem allocation and retrieve the pointer
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # is_relaxed=False provides the memory-consistency guarantee required by
        # the DK/DV cluster-shared pipeline state.
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)

        tSTtST_shape = KQ_tiled_mma.partition_shape_C(cute.select(self.KQ_mma_tiler, mode=[0, 1]))
        tSTtST = KQ_tiled_mma.make_fragment_C(tSTtST_shape)
        # tSTtST shape : (MMA, MMA_M, MMA_N)
        tSTtST = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tSTtST.layout)

        # tdVrP shape : (MMA, MMA_M, MMA_K, STAGE)
        tdVrP = PdO_tiled_mma.make_fragment_A(sP)
        # tdVrdOT shape : (MMA, MMA_N, MMA_K, STAGE)
        tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)

        tdPTtdPT_shape = VdO_tiled_mma.partition_shape_C(cute.select(self.VdO_mma_tiler, mode=[0, 1]))
        tdPTtdPT = VdO_tiled_mma.make_fragment_C(tdPTtdPT_shape)
        # tdPTtdPT shape : (MMA, MMA_M, MMA_N)
        tdPTtdPT = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPTtdPT.layout)

        tdKtdK_shape = dSQ_tiled_mma.partition_shape_C(cute.select(self.dSQ_mma_tiler, mode=[0, 1]))
        tdKtdK = dSQ_tiled_mma.make_fragment_C(tdKtdK_shape)
        # tdKtdK shape : (MMA, MMA_M, MMA_N)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)

        tdVtdV_shape = PdO_tiled_mma.partition_shape_C(cute.select(self.PdO_mma_tiler, mode=[0, 1]))
        tdVtdV = PdO_tiled_mma.make_fragment_C(tdVtdV_shape)
        # tdVtdV shape : (MMA, MMA_M, MMA_N)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)

        # get the current batch problem shape

        # Static non-persistent scheduler path.
        blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
        seqlen_q_cur_batch = cumulative_s_q[bidz + 1] - cumulative_s_q[bidz]
        seqlen_k_cur_batch = cumulative_s_k[bidz + 1] - cumulative_s_k[bidz]
        blk_offset = (
            cumulative_s_q[bidz],
            cumulative_s_k[bidz],
            Int32(0),
            ((Int32(0), Int32(0)), Int32(0)),
        )

        m_block_max = cute.ceil_div(seqlen_q_cur_batch, self.tile_shape_Q)
        if cutlass.const_expr(self.use_auto_block_metadata):
            assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
            # A metadata row is a K128 cluster work unit.  Both physical K64
            # CTAs therefore use bidx // 2 and observe the same exact loop
            # count, avoiding divergent 2-CTA pipeline phases.
            k_cluster = blk_coord[1] // 2
            sparse_coarse_count = get_k2q_block_sparse_consumer_row(
                block_sparse_tensors,
                bidz,
                k_cluster,
            )[0]
            iter_start = Int32(0)
            iter_end = sparse_coarse_count * self.subtile_factor
        else:
            iter_start, iter_end = self.get_Q_block_min_max(
                seqlen_q_cur_batch,
                seqlen_k_cur_batch,
                blk_coord[1],
            )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
        problem_shape_cur_batch = (
            seqlen_q_cur_batch,
            seqlen_k_cur_batch,
            problem_shape[2],
            problem_shape[3],
        )
        if iter_count <= 0:
            # This physical K64 CTA owns its dK/dV output tile.  EMPTY sparse
            # rows do not advance a producer/consumer phase, but they still
            # write a defined zero tile before the common CTA/TMEM teardown.
            if bidx * self.tile_shape_K < seqlen_k_cur_batch:
                self.epilogue_clear(
                    blk_coord,
                    blk_offset,
                    problem_shape_cur_batch,
                    dK,
                    dV,
                )
        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

            self.load(
                K_in,
                V_in,
                Q_in,
                QT_in,
                dO_in,
                dOT_in,
                sK,
                sQ,
                sQT,
                sV,
                sdO,
                sdOT,
                KQ_tiled_mma,
                VdO_tiled_mma,
                PdO_tiled_mma,
                dSQ_tiled_mma,
                tma_atom_K,
                tma_atom_Q,
                tma_atom_QT,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dOT,
                blk_offset,
                iter_count,
                iter_start,
                iter_end,
                m_block_max,
                block_sparse_tensors,
                load_mma_Q_producer,
                load_mma_K_producer,
                load_mma_V_producer,
                load_mma_dO_producer,
                load_mma_dOT_producer,
                load_mma_QT_producer,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

            self.mma_2cta(
                KQ_tiled_mma,
                VdO_tiled_mma,
                PdO_tiled_mma,
                dSQ_tiled_mma,
                tSTtST,
                tSTrQ,
                tSTrK,
                tdPTtdPT,
                tdPTrV,
                tdPTrdO,
                tdVtdV,
                tdVrP,
                tdVrdOT,
                tdKrdST,
                tdKtdK,
                tdKrQT,
                iter_count,
                load_mma_Q_consumer,
                load_mma_K_consumer,
                load_mma_V_consumer,
                mma_compute_S_producer,
                load_mma_dO_consumer,
                mma_compute_dP_producer,
                load_mma_dOT_consumer,
                compute_mma_P_consumer,
                compute_mma_dS_consumer,
                load_mma_QT_consumer,
                mma_compute_dKdV_producer,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx >= self.compute_warp_id[0] and warp_idx <= self.compute_warp_id[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)

            self.compute(
                tSTtST,
                tdPTtdPT,
                sP,
                sdST,
                dK,
                dV,
                tdKtdK,
                tdVtdV,
                blk_coord,
                blk_offset,
                problem_shape_cur_batch,
                iter_count,
                iter_start,
                iter_end,
                scale_softmax,
                normalization_scale,
                func,
                block_sparse_tensors,
                mma_compute_S_consumer,
                compute_mma_P_producer,
                compute_mma_P_consumer,
                mma_compute_dP_consumer,
                compute_mma_dS_producer,
                compute_mma_dS_consumer,
                mma_compute_dKdV_consumer,
                seqlen_k_cur_batch,
            )

            cute.arch.barrier(
                barrier_id=self.epilogue_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

        else:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id,
            number_of_threads=self.threads_per_cta,
        )
        # Release tensor memory after every warp in this CTA has arrived.
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def get_Q_block_min_max(
        self,
        seq_Q: Int32,
        seq_K: Int32,
        blk_coord_k: Int32,
    ):
        """Get Q tiles range."""
        Q_block_max = cute.ceil_div(seq_Q, self.tile_shape_Q)
        Q_block_min = cutlass.Int32(0)
        if cutlass.const_expr(self.has_sliding_window):
            # For 2cta, use the last K block in the cluster so both CTAs get the same Q_block_max
            blk_coord_k_for_max = (blk_coord_k // 2) * 2 + 1
            Q_block_max_tmp = cute.ceil_div(
                (blk_coord_k_for_max + 1) * self.tile_shape_K + seq_Q - seq_K + self.window_size_left,
                self.tile_shape_Q,
            )
            Q_block_max = min(Q_block_max, Q_block_max_tmp)
        if cutlass.const_expr(self.is_causal or self.has_sliding_window):
            # For 2cta, use the first K block in the cluster so both CTAs get the same Q_block_min.
            # This ensures both CTAs in a cluster run the same number of pipeline iterations,
            # avoiding hang from mismatched producer_commit / consumer_wait counts.
            blk_coord_k_for_min = (blk_coord_k // 2) * 2
            Q_block_min_tmp = (blk_coord_k_for_min * self.tile_shape_K + seq_Q - seq_K - self.window_size_right) // self.tile_shape_Q
            # Consider the case of 2cta, we need to ensure the K block is aligned to 2
            Q_block_min_tmp = Q_block_min_tmp - Q_block_min_tmp % 2
            Q_block_min = max(Q_block_min_tmp, Q_block_min)
        return Q_block_min, Q_block_max

    @cute.jit
    def load(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        QT_in: cute.Tensor,
        dO_in: cute.Tensor,
        dOT_in: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sQT: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOT: cute.Tensor,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_QT: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOT: cute.CopyAtom,
        blk_offset: cute.Shape,
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        m_block_max: Int32,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        load_mma_Q_producer,
        load_mma_K_producer,
        load_mma_V_producer,
        load_mma_dO_producer,
        load_mma_dOT_producer,
        load_mma_QT_producer,
    ):
        """TMA load."""
        blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
        blk_coord_h_r = Int32(0)
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
        iter_index = iter_start
        mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
        mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)
        mask_block_cnt = None
        mask_block_idx = None
        full_block_cnt = None
        full_block_idx = None
        if cutlass.const_expr(self.use_auto_block_metadata):
            assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
            (
                _,
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
            ) = get_k2q_block_sparse_consumer_row(
                block_sparse_tensors,
                blk_coord_b,
                mma_tile_coord_m,
            )

        K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
        V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
        Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
        QT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), QT_in)
        dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)
        dOT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), dOT_in)

        gK = cute.local_tile(K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None))
        gQ = cute.local_tile(Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gQT = cute.local_tile(QT, cute.select(self.dSQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gV = cute.local_tile(V, cute.select(self.VdO_mma_tiler, mode=[0, 2]), (None, None, None))
        gdO = cute.local_tile(dO, cute.select(self.VdO_mma_tiler, mode=[1, 2]), (None, None, None))
        gdOT = cute.local_tile(dOT, cute.select(self.PdO_mma_tiler, mode=[1, 2]), (None, None, None))

        KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
        VdO_thr_mma = VdO_tiled_mma.get_slice(mma_tile_coord_v)
        PdO_thr_mma = PdO_tiled_mma.get_slice(mma_tile_coord_v)
        dSQ_thr_mma = dSQ_tiled_mma.get_slice(mma_tile_coord_v)

        tSTgK = KQ_thr_mma.partition_A(gK)
        tSTgQ = KQ_thr_mma.partition_B(gQ)
        tdKgQT = dSQ_thr_mma.partition_B(gQT)
        tdPTgV = VdO_thr_mma.partition_A(gV)
        tdPTgdO = VdO_thr_mma.partition_B(gdO)
        tdVgdOT = PdO_thr_mma.partition_B(gdOT)

        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id,))
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cute.arch.block_idx_in_cluster())

        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSTgK, 0, 3),
        )
        tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSTgQ, 0, 3),
        )
        tQTsQT, tQTgQT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_QT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQT, 0, 3),
            cute.group_modes(tdKgQT, 0, 3),
        )
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPTgV, 0, 3),
        )
        tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPTgdO, 0, 3),
        )
        tdOTsdOT, tdOTgdOT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dOT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdOT, 0, 3),
            cute.group_modes(tdVgdOT, 0, 3),
        )

        k_handle = load_mma_K_producer.acquire_and_advance()
        cute.copy(
            tma_atom_K,
            tKgK_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tKsK[None, 0],
            tma_bar_ptr=k_handle.barrier,
        )

        m_block_for_load = iter_index
        if cutlass.const_expr(self.use_auto_block_metadata):
            m_block_for_load, _ = get_k2q_q_block_for_subtile_iteration(
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
                iter_index,
                self.subtile_factor,
            )
            # A Q256 tail may have no physical second Q128 subtile.  Duplicate
            # the last valid load while compute predicates the logical subtile
            # to zero, avoiding a packed-varlen cross-sequence TMA read.
            m_block_for_load = min(m_block_for_load, m_block_max - 1)

        q_handle = load_mma_Q_producer.acquire_and_advance()
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
            tQsQ[None, q_handle.index],
            tma_bar_ptr=q_handle.barrier,
        )

        v_handle = load_mma_V_producer.acquire_and_advance()
        cute.copy(
            tma_atom_V,
            tVgV_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tVsV[(None, 0)],
            tma_bar_ptr=v_handle.barrier,
        )

        do_handle = load_mma_dO_producer.acquire_and_advance()
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
            tdOsdO[(None, do_handle.index)],
            tma_bar_ptr=do_handle.barrier,
        )

        dot_handle = load_mma_dOT_producer.acquire_and_advance()
        cute.copy(
            tma_atom_dOT,
            tdOTgdOT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
            tdOTsdOT[None, dot_handle.index],
            tma_bar_ptr=dot_handle.barrier,
        )

        qt_handle = load_mma_QT_producer.acquire_and_advance()
        cute.copy(
            tma_atom_QT,
            tQTgQT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
            tQTsQT[None, qt_handle.index],
            tma_bar_ptr=qt_handle.barrier,
        )

        iter_count -= 1
        iter_index += 1

        while iter_count > 0:
            if iter_index == iter_end:
                iter_index = iter_start
                blk_coord_h_r += 1
                blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

            m_block_for_load = iter_index
            if cutlass.const_expr(self.use_auto_block_metadata):
                m_block_for_load, _ = get_k2q_q_block_for_subtile_iteration(
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                    iter_index,
                    self.subtile_factor,
                )
                m_block_for_load = min(m_block_for_load, m_block_max - 1)

            q_handle = load_mma_Q_producer.acquire_and_advance()
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
                tQsQ[None, q_handle.index],
                tma_bar_ptr=q_handle.barrier,
            )

            do_handle = load_mma_dO_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO[None, do_handle.index],
                tma_bar_ptr=do_handle.barrier,
            )

            dot_handle = load_mma_dOT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dOT,
                tdOTgdOT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
                tdOTsdOT[None, dot_handle.index],
                tma_bar_ptr=dot_handle.barrier,
            )

            qt_handle = load_mma_QT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_QT,
                tQTgQT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
                tQTsQT[None, qt_handle.index],
                tma_bar_ptr=qt_handle.barrier,
            )

            iter_count -= 1
            iter_index += 1

        load_mma_K_producer.tail()
        load_mma_V_producer.tail()
        load_mma_Q_producer.tail()
        load_mma_dO_producer.tail()
        load_mma_dOT_producer.tail()
        load_mma_QT_producer.tail()

        return (
            load_mma_Q_producer,
            load_mma_K_producer,
            load_mma_V_producer,
            load_mma_dO_producer,
            load_mma_dOT_producer,
            load_mma_QT_producer,
        )

    @cute.jit
    def mma_2cta(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tSTtST: cute.Tensor,
        tSTrQ: cute.Tensor,
        tSTrK: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdPTrV: cute.Tensor,
        tdPTrdO: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdVrdOT: cute.Tensor,
        tdKrdST: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdKrQT: cute.Tensor,
        iter_count: Int32,
        load_mma_Q_consumer,
        load_mma_K_consumer,
        load_mma_V_consumer,
        mma_compute_S_producer,
        load_mma_dO_consumer,
        mma_compute_dP_producer,
        load_mma_dOT_consumer,
        compute_mma_P_consumer,
        compute_mma_dS_consumer,
        load_mma_QT_consumer,
        mma_compute_dKdV_producer,
    ):
        """CuTeDSL kernel for mma pipeline."""
        load_mma_K_releaser = load_mma_K_consumer.clone()
        load_mma_V_releaser = load_mma_V_consumer.clone()

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = cta_rank_in_cluster % 2 == 0

        if is_leader_cta:
            s_handle = mma_compute_S_producer.acquire_and_advance()
            load_mma_K_consumer.wait_and_advance()
            q_handle = load_mma_Q_consumer.wait_and_advance()

            # Compute S = K * Q
            for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block != 0)
                cute.gemm(
                    KQ_tiled_mma,
                    tSTtST,
                    tSTrK[None, None, k_block, 0],
                    tSTrQ[None, None, k_block, q_handle.index],
                    tSTtST,
                )
            q_handle.release()

            cute.arch.fence_view_async_tmem_store()
            s_handle.commit()

            do_handle = load_mma_dO_consumer.wait_and_advance()
            load_mma_V_consumer.wait_and_advance()

            dp_handle = mma_compute_dP_producer.acquire_and_advance()

            # Compute dP = V * dO
            VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPTrV, mode=[2]), unroll_full=True):
                cute.gemm(
                    VdO_tiled_mma,
                    tdPTtdPT,
                    tdPTrV[None, None, k_block, 0],
                    tdPTrdO[None, None, k_block, do_handle.index],
                    tdPTtdPT,
                )
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dp_handle.commit()
            do_handle.release()
            # V only produced once by load(); hold v_handle until end, release there via releaser

            p_handle = compute_mma_P_consumer.wait_and_advance()
            dot_handle = load_mma_dOT_consumer.wait_and_advance()

            # Compute dV = P * dO (First iteration)
            PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                cute.gemm(
                    PdO_tiled_mma,
                    tdVtdV,
                    tdVrP[None, None, k_block, 0],
                    tdVrdOT[None, None, k_block, dot_handle.index],
                    tdVtdV,
                )
                PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dot_handle.release()
            p_handle.release()

        iter_count -= 1

        dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        while iter_count > 0:
            if is_leader_cta:
                q_handle = load_mma_Q_consumer.wait_and_advance()
                s_handle = mma_compute_S_producer.acquire_and_advance()

                # Compute S = K * Q
                for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                    KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block != 0)
                    cute.gemm(
                        KQ_tiled_mma,
                        tSTtST,
                        tSTrK[None, None, k_block, 0],
                        tSTrQ[None, None, k_block, q_handle.index],
                        tSTtST,
                    )
                q_handle.release()
                s_handle.commit()

            if is_leader_cta:
                qt_handle = load_mma_QT_consumer.wait_and_advance()
                ds_handle = compute_mma_dS_consumer.wait_and_advance()

                # Compute dK = dS * QT
                for k_block in cutlass.range(0, cute.size(tdKrdST, mode=[2]), unroll_full=True):
                    cute.gemm(
                        dSQ_tiled_mma,
                        tdKtdK,
                        tdKrdST[
                            None,
                            None,
                            k_block,
                            ds_handle.index,
                        ],
                        tdKrQT[None, None, k_block, qt_handle.index],
                        tdKtdK,
                    )
                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                qt_handle.release()
                ds_handle.release()

            if is_leader_cta:
                dp_handle = mma_compute_dP_producer.acquire_and_advance()
                do_handle = load_mma_dO_consumer.wait_and_advance()
                # V only produced once by load(); reuse same V (index 0) for all loop iterations
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdPTrV, mode=[2]), unroll_full=True):
                    cute.gemm(
                        VdO_tiled_mma,
                        tdPTtdPT,
                        tdPTrV[None, None, k_block, 0],
                        tdPTrdO[None, None, k_block, do_handle.index],
                        tdPTtdPT,
                    )
                    VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                dp_handle.commit()
                do_handle.release()

            if is_leader_cta:
                p_handle = compute_mma_P_consumer.wait_and_advance()
                dot_handle = load_mma_dOT_consumer.wait_and_advance()

                # Compute dV = P * dO (Loop iterations)
                for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                    cute.gemm(
                        PdO_tiled_mma,
                        tdVtdV,
                        tdVrP[None, None, k_block, 0],
                        tdVrdOT[None, None, k_block, dot_handle.index],
                        tdVtdV,
                    )
                    PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                p_handle.release()
                dot_handle.release()

            iter_count -= 1

        if is_leader_cta:
            dkdv_handle = mma_compute_dKdV_producer.acquire_and_advance()
            dkdv_handle.commit()

            load_mma_K_releaser.release()
            load_mma_K_releaser.advance()
            load_mma_V_releaser.release()
            load_mma_V_releaser.advance()

        if is_leader_cta:
            dkdv_handle = mma_compute_dKdV_producer.acquire_and_advance()

            ds_handle = compute_mma_dS_consumer.wait_and_advance()
            qt_handle = load_mma_QT_consumer.wait_and_advance()

            # Compute dK = dS * Q
            for k_block in cutlass.range(0, cute.size(tdKrdST, mode=[2]), unroll_full=True):
                cute.gemm(
                    dSQ_tiled_mma,
                    tdKtdK,
                    tdKrdST[None, None, k_block, ds_handle.index],
                    tdKrQT[None, None, k_block, qt_handle.index],
                    tdKtdK,
                )
                dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dkdv_handle.commit()
            qt_handle.release()
            ds_handle.release()

        mma_compute_S_producer.tail()
        mma_compute_dP_producer.tail()
        mma_compute_dKdV_producer.tail()

        return (
            mma_compute_S_producer,
            mma_compute_dP_producer,
            mma_compute_dKdV_producer,
            load_mma_Q_consumer,
            load_mma_K_consumer,
            load_mma_V_consumer,
            load_mma_dO_consumer,
            load_mma_dOT_consumer,
            compute_mma_P_consumer,
            compute_mma_dS_consumer,
            load_mma_QT_consumer,
        )

    @cute.jit
    def reg_to_smem_mma128x128_2cta(
        self,
        regs: cute.Tensor,
        smem: cute.Tensor,
        index: Int32,
        tiler_mn: tuple[Int32, Int32],
        dp_idx: Int32,
        wg_idx: Int32,
    ):
        smem_slice = smem[None, None, None, index]
        # K>> smem_slice:  tensor<ptr<f16, smem, align<1024>, S<3,4,3>> o ((64,16),1,(4,2)):((64,1),0,(16,4096))>
        thread_layout = cute.make_ordered_layout(
            # (tileN, tileM)
            tiler_mn,
            (0, 1),
        )
        # K>> thread_layout:  (64,128):(128,1)
        smem_slice_tmp = cute.composition(smem_slice, thread_layout)

        # NOTE: hardcode for tcgen05.ld.32x32b.x8 & mma128x64+2cta
        # tmp_shape = ((32, 2), (8, 2, 2, 2)) # for 64x64 tile
        # tmp_stride = ((64, 32*64), (1, 8, 16, 32))
        # NOTE: hardcode for tcgen05.ld.32x32b.x16 & mma128x64+2cta
        tmp_shape = ((32, 2), (16, 2, 2, 2))  # for 128x64 tile
        tmp_stride = ((64, 32 * 64), (1, 16, 32, 64 * 64))
        # smem_copy = cute.composition(smem_slice_tmp, cute.make_layout(tmp_shape, stride=tmp_stride))
        smem_copy = cute.make_tensor(smem_slice_tmp.iterator, cute.make_layout(tmp_shape, stride=tmp_stride))

        warp_idx = dp_idx // 32
        warp_row_idx = warp_idx % 2
        warp_col_idx = warp_idx // 2  # corresponding to the second 64 cols in smem
        lane_idx = dp_idx % 32
        reg_shape = regs.shape  # ((8,1),1,2):((1,0),0,8) for 64x64, ((16,1),1,2):((1,0),0,16) for 128x64
        block_loops = reg_shape[2]

        # TODO: maybe can use cp.async for optimization
        for ib in cutlass.range(block_loops):
            regs_copy = regs[(None, 0), 0, ib]
            smem_copy_slice = smem_copy[(lane_idx, warp_row_idx), (None, wg_idx, ib, warp_col_idx)]
            cute.autovec_copy(regs_copy, smem_copy_slice)

    @cute.jit
    def compute(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        sP: cute.Tensor,
        sdST: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        scale_softmax: cutlass.Float32,
        normalization_scale: cutlass.Float32,
        func: cute.Tensor | None,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        mma_compute_S_consumer,
        compute_mma_P_producer,
        compute_mma_P_consumer,
        mma_compute_dP_consumer,
        compute_mma_dS_producer,
        compute_mma_dS_consumer,
        mma_compute_dKdV_consumer,
        problem_shape_k_cur_batch: Int32,
    ):
        """CuTeDSL kernel for recomputing softmax and producing dk and dv."""
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, _, _ = problem_shape
        _, blk_coord_k, _, _ = blk_coord

        iter_index = iter_start
        mask_block_cnt = None
        mask_block_idx = None
        full_block_cnt = None
        full_block_idx = None
        if cutlass.const_expr(self.use_auto_block_metadata):
            assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
            (
                _,
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
            ) = get_k2q_block_sparse_consumer_row(
                block_sparse_tensors,
                blk_coord[3][1],
                blk_coord_k // 2,
            )

        # adi: TMEM_ST, TMEM_DPT
        tmem_load_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16))
        tmem_load_atom = cute.make_copy_atom(
            tmem_load_op,
            self.acc_dtype,
        )

        tSTtST = tSTtST[(None, None), 0, 0]
        tdPTtdPT = tdPTtdPT[(None, None), 0, 0]

        cST = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))
        cdPT = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cST = thr_t2r.partition_D(cST)
        tTR_cST = split_wg(tTR_cST, num_warp_groups, wg_idx)
        tTR_rST = cute.make_rmem_tensor(tTR_cST.shape, self.acc_dtype)

        tTR_tST = thr_t2r.partition_S(tSTtST)
        tTR_tST = split_wg(tTR_tST, num_warp_groups, wg_idx)

        tTR_cdPT_p = thr_t2r.partition_D(cdPT)
        tTR_cdPT = split_wg(tTR_cdPT_p, num_warp_groups, wg_idx)
        tTR_rdPT = cute.make_rmem_tensor(tTR_cdPT.shape, self.acc_dtype)

        tTR_tdPT = thr_t2r.partition_S(tdPTtdPT)
        tTR_tdPT = split_wg(tTR_tdPT, num_warp_groups, wg_idx)

        is_residual_k = blk_coord_k * self.tile_shape_K + self.tile_shape_K > K

        while iter_count > 0:
            m_block = iter_index
            is_mask_block = cutlass.Boolean(False)
            if cutlass.const_expr(self.use_auto_block_metadata):
                m_block, is_mask_block = get_k2q_q_block_for_subtile_iteration(
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                    iter_index,
                    self.subtile_factor,
                )

            s_handle = mma_compute_S_consumer.wait_and_advance()
            p_handle = compute_mma_P_producer.acquire_and_advance()

            leading_causal_masking = cutlass.Boolean(False)
            if cutlass.const_expr(self.is_causal):
                # TODO: could be optimized by specify an exact iter_index
                #
                # NOTE (causal + 2CTA correctness):
                # `iter_start` can be rounded down to an even index for 2CTA. When the true
                # causal boundary Q tile is odd, it becomes (iter_start + 1). In that case,
                # we must treat both (iter_start, iter_start + 1) as "masked tiles" so the
                # per-element causal mask is applied on the boundary tile too.
                leading_causal_masking = iter_index == iter_start
                q_block_min_unaligned = (blk_coord_k * self.tile_shape_K + Q - K - self.window_size_right) // self.tile_shape_Q
                boundary_in_second = (q_block_min_unaligned % 2) == 1
                leading_causal_masking = leading_causal_masking or (boundary_in_second and (iter_index == iter_start + 1))
                offset_partial_tile = (K - Q) % self.tile_shape_K
                need_additional_mask = offset_partial_tile and iter_index == iter_start + 1
                leading_causal_masking = leading_causal_masking or need_additional_mask
                leading_causal_masking = cute.arch.shuffle_sync(leading_causal_masking, 0)

            residual_q = (m_block + 1) * self.tile_shape_Q > Q
            trailing_residual_masking = residual_q or is_residual_k
            trailing_residual_masking = cute.arch.shuffle_sync(trailing_residual_masking, 0)

            # Compute HSTU P=silu(alpha*S) and retain silu' for dS.
            cute.copy(tiled_t2r, tTR_tST, tTR_rST)
            tTR_rdSilu = cute.make_rmem_tensor(tTR_rST.shape, self.acc_dtype)
            if cutlass.const_expr(self.use_auto_block_metadata):
                # Keep three distinct CuTe call sites.  The FULL-interior path
                # neither accepts func nor materializes a predicate fragment;
                # only MASK performs endpoint loads, while FULL-tail uses a
                # seqlen-only predicate.
                if is_mask_block:
                    assert isinstance(func, cute.Tensor)
                    self.silu_bwd_auto_exact_step(
                        tTR_rST,
                        tTR_rdSilu,
                        tTR_cST,
                        Q,
                        K,
                        m_block,
                        blk_coord_k,
                        blk_offset[0],
                        scale_softmax,
                        func,
                    )
                elif trailing_residual_masking:
                    self.silu_bwd_auto_seqlen_step(
                        tTR_rST,
                        tTR_rdSilu,
                        tTR_cST,
                        Q,
                        K,
                        m_block,
                        blk_coord_k,
                        scale_softmax,
                    )
                else:
                    self.silu_bwd_step(
                        tTR_rST,
                        tTR_rdSilu,
                        scale_softmax,
                        None,
                    )
            else:
                apply_arbitrary_mask = cutlass.const_expr(self.is_arbitrary)
                is_masked_tile = (
                    leading_causal_masking or trailing_residual_masking or self.has_sliding_window or cutlass.const_expr(self.is_causal) or apply_arbitrary_mask
                )
                if cutlass.const_expr(not self.skip_residual_mask) and is_masked_tile:
                    score_predicates = cute.make_rmem_tensor(
                        tTR_rST.shape,
                        cutlass.Boolean,
                    )
                    for i in cutlass.range(
                        cute.size(score_predicates),
                        unroll_full=True,
                    ):
                        score_predicates[i] = True

                    for i in cutlass.range(cute.size(tTR_rST), unroll_full=True):
                        c_transpose = tTR_cST[i]
                        pos = (
                            cute.get(c_transpose, mode=[1]) + m_block * self.tile_shape_Q,
                            cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_shape_K,
                        )
                        if cutlass.const_expr(self.is_arbitrary):
                            assert isinstance(func, cute.Tensor)
                            # Guard the packed-varlen Q tail before forming the
                            # func row, preserving the non-metadata traversal.
                            if cute.elem_less(pos, (Q, K)):
                                func_row = blk_offset[0] + pos[0]
                                keep = cutlass.Boolean(pos[1] < func[0, 0, func_row])
                                for interval_idx in cutlass.range(
                                    self.func_num // 2,
                                    unroll_full=True,
                                ):
                                    interval_start = func[
                                        0,
                                        2 * interval_idx + 1,
                                        func_row,
                                    ]
                                    interval_end = func[
                                        0,
                                        2 * interval_idx + 2,
                                        func_row,
                                    ]
                                    if pos[1] >= interval_start and pos[1] < interval_end:
                                        keep = True
                                if not keep:
                                    score_predicates[i] = False
                            else:
                                score_predicates[i] = False
                        if cutlass.const_expr(self.has_sliding_window):
                            if cutlass.const_expr(self.window_size_left < 0):
                                if pos[1] > pos[0] + K - Q + self.window_size_right:
                                    score_predicates[i] = False
                            else:
                                max_K_index = min(
                                    pos[0] + K - Q + self.window_size_right,
                                    K,
                                )
                                min_K_index = max(
                                    0,
                                    pos[0] + K - Q - self.window_size_left,
                                )
                                if pos[1] > max_K_index or pos[1] < min_K_index:
                                    score_predicates[i] = False
                        if cutlass.const_expr(self.is_causal) and (pos[0] + K - Q < pos[1] or not cute.elem_less(pos, (Q, K))):
                            score_predicates[i] = False
                        if not cute.elem_less(pos, (Q, K)):
                            score_predicates[i] = False

                    self.silu_bwd_step(
                        tTR_rST,
                        tTR_rdSilu,
                        scale_softmax,
                        score_predicates,
                    )
                else:
                    self.silu_bwd_step(
                        tTR_rST,
                        tTR_rdSilu,
                        scale_softmax,
                        None,
                    )

            # convert fp32 P to fp16 P which will be used in the PdO
            tTR_rPT = self.quantize(tTR_rST, dV.element_type)  # tTR_rST is ST in fp32 in RF.
            self.reg_to_smem_mma128x128_2cta(
                tTR_rPT,
                sP,
                p_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

            p_handle.commit()

            s_handle.release()
            dp_handle = mma_compute_dP_consumer.wait_and_advance()
            ds_handle = compute_mma_dS_producer.acquire_and_advance()

            # Compute the unscaled dS fragment. The dK epilogue applies
            # ``scale_softmax`` exactly once, so multiplying by alpha here
            # would scale dK twice.
            cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)

            for i in cutlass.range(0, cute.size(tTR_rdPT), 2, unroll_full=True):
                tTR_rdPT[i], tTR_rdPT[i + 1] = mul_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]),
                    (tTR_rdSilu[i], tTR_rdSilu[i + 1]),
                )
            # Keep an explicit zeroing pass for causal dS.  Although the SiLU
            # derivative is already predicated above, this layout-specific pass
            # is required by the 2-CTA register-to-shared transform.
            if cutlass.const_expr(self.is_causal):
                for i in cutlass.range(cute.size(tTR_rdPT), unroll_full=True):
                    c_transpose = tTR_cdPT[i]
                    pos = (
                        cute.get(c_transpose, mode=[1]) + m_block * self.tile_shape_Q,
                        cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_shape_K,
                    )
                    if pos[0] + K - Q < pos[1] or not cute.elem_less(pos, (Q, K)):
                        tTR_rdPT[i] = cutlass.Float32(0.0)
            # convert fp32 dS to fp16 dS which will be used in the computation of dK and DQ
            tTR_rdST = self.quantize(tTR_rdPT, dV.element_type)

            cute.arch.fence_view_async_tmem_load()
            dp_handle.release()

            self.reg_to_smem_mma128x128_2cta(
                tTR_rdST,
                sdST,
                ds_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

            ds_handle.commit()

            iter_count -= 1
            iter_index += 1
            if iter_index == iter_end:
                iter_index = iter_start

        # Epilogue
        mma_compute_dKdV_consumer = self.epilogue(
            blk_coord,
            blk_offset,
            problem_shape,
            dK,
            dV,
            tdKtdK,
            tdVtdV,
            scale_softmax,
            normalization_scale,
            mma_compute_dKdV_consumer,
            problem_shape_k_cur_batch,
        )

        compute_mma_P_producer.tail()
        compute_mma_dS_producer.tail()

        return (
            compute_mma_P_producer,
            compute_mma_dS_producer,
            mma_compute_S_consumer,
            compute_mma_P_consumer,
            mma_compute_dP_consumer,
            compute_mma_dS_consumer,
            mma_compute_dKdV_consumer,
        )

    @cute.jit
    def silu_bwd_auto_seqlen_step(
        self,
        scores: cute.Tensor,
        silu_derivative: cute.Tensor,
        score_coords: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        m_block: Int32,
        k_block: Int32,
        scale_softmax: cutlass.Float32,
    ):
        """Apply the FULL-tail seqlen guard without accepting func."""

        score_predicates = cute.make_rmem_tensor(
            scores.shape,
            cutlass.Boolean,
        )
        for i in cutlass.range(cute.size(scores), unroll_full=True):
            c_transpose = score_coords[i]
            pos = (
                cute.get(c_transpose, mode=[1]) + m_block * self.tile_shape_Q,
                cute.get(c_transpose, mode=[0]) + k_block * self.tile_shape_K,
            )
            score_predicates[i] = cute.elem_less(pos, (seqlen_q, seqlen_k))
        self.silu_bwd_step(
            scores,
            silu_derivative,
            scale_softmax,
            score_predicates,
        )

    @cute.jit
    def silu_bwd_auto_exact_step(
        self,
        scores: cute.Tensor,
        silu_derivative: cute.Tensor,
        score_coords: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        m_block: Int32,
        k_block: Int32,
        query_offset: Int32,
        scale_softmax: cutlass.Float32,
        func: cute.Tensor,
    ):
        """Apply the exact arbitrary predicate for a metadata MASK tile."""

        score_predicates = cute.make_rmem_tensor(
            scores.shape,
            cutlass.Boolean,
        )
        for i in cutlass.range(cute.size(scores), unroll_full=True):
            c_transpose = score_coords[i]
            pos = (
                cute.get(c_transpose, mode=[1]) + m_block * self.tile_shape_Q,
                cute.get(c_transpose, mode=[0]) + k_block * self.tile_shape_K,
            )
            valid = cute.elem_less(pos, (seqlen_q, seqlen_k))
            if valid:
                func_row = query_offset + pos[0]
                valid = cutlass.Boolean(pos[1] < func[0, 0, func_row])
                for interval_idx in cutlass.range(
                    self.func_num // 2,
                    unroll_full=True,
                ):
                    interval_start = func[
                        0,
                        2 * interval_idx + 1,
                        func_row,
                    ]
                    interval_end = func[
                        0,
                        2 * interval_idx + 2,
                        func_row,
                    ]
                    if pos[1] >= interval_start and pos[1] < interval_end:
                        valid = True
            score_predicates[i] = valid
        self.silu_bwd_step(
            scores,
            silu_derivative,
            scale_softmax,
            score_predicates,
        )

    @cute.jit
    def silu_bwd_step(
        self,
        scores: cute.Tensor,
        silu_derivative: cute.Tensor,
        scale_softmax: cutlass.Float32,
        predicates: Optional[cute.Tensor],
    ):
        """Apply HSTU SiLU and retain its derivative for one score tile."""

        for i in cutlass.range(
            0,
            cute.size(scores),
            2,
            unroll_full=True,
        ):
            x0, x1 = mul_packed_f32x2(
                (scores[i], scores[i + 1]),
                (scale_softmax, scale_softmax),
            )
            tanh0 = tanhf(x0 * 0.5)
            tanh1 = tanhf(x1 * 0.5)
            sigmoid0, sigmoid1 = fma_packed_f32x2(
                (0.5, 0.5),
                (tanh0, tanh1),
                (0.5, 0.5),
            )
            if cutlass.const_expr(predicates is not None):
                assert isinstance(predicates, cute.Tensor)
                sigmoid0 = sigmoid0 if predicates[i] else self.acc_dtype(0)
                sigmoid1 = sigmoid1 if predicates[i + 1] else self.acc_dtype(0)
            scores[i], scores[i + 1] = mul_packed_f32x2(
                (x0, x1),
                (sigmoid0, sigmoid1),
            )
            one_minus0, one_minus1 = sub_packed_f32x2(
                (1.0, 1.0),
                (sigmoid0, sigmoid1),
            )
            inner0, inner1 = fma_packed_f32x2(
                (x0, x1),
                (one_minus0, one_minus1),
                (1.0, 1.0),
            )
            (
                silu_derivative[i],
                silu_derivative[i + 1],
            ) = mul_packed_f32x2(
                (sigmoid0, sigmoid1),
                (inner0, inner1),
            )

    @cute.jit
    def quantize(
        self,
        input_t: cute.Tensor,
        element_dtype: type[cutlass.Numeric],
    ) -> cute.Tensor:
        """Convert Float32 to element dtype."""
        output = cute.make_rmem_tensor(input_t.shape, element_dtype)
        output.store(input_t.load().to(element_dtype))
        return output

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        for i in cutlass.range(cute.size(coord, mode=[2]), unroll_full=True):
            coord_i = coord[None, 0, i]
            gmem_i = gmem[None, 0, i]
            regs_i = regs[None, 0, i]
            if cute.elem_less(coord_i[0], tensor_shape):
                gmem_i.store(regs_i.load().to(gmem.element_type))

    @cute.jit
    def epilogue_clear(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
    ):
        """Write zeros for an EMPTY K-owner tile."""
        tidx, _, _ = cute.arch.thread_idx()
        block_dim_x, _, _ = cute.arch.block_dim()
        _, K, _, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        mdK_offset = cute.assume(blk_offset[1] * dK.stride[0], divby=64)
        mdK = cute.make_tensor(
            dK.iterator + mdK_offset,
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        # Each physical CTA owns one K64-by-D tile.  The MMA tilers are
        # cluster-wide K128 shapes and must not be used for EMPTY stores.
        gdK = cute.local_tile(mdK, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        mdV_offset = cute.assume(blk_offset[1] * dV.stride[0], divby=64)
        mdV = cute.make_tensor(
            dV.iterator + mdV_offset,
            cute.make_layout((K, self.tile_shape_dV_dO, HB), stride=dV.stride),
        )
        gdV = cute.local_tile(mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]
        cdV = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        # Index through the tensor layout instead of treating its iterator as
        # a flat pointer: the residual K row is strided in packed H/K/D views.
        for i in cutlass.range(tidx, cute.size(gdK), block_dim_x):
            if cute.elem_less(cdK[i], cute.select(problem_shape, mode=[1, 2])):
                gdK[i] = dK.element_type(0)

        for i in cutlass.range(tidx, cute.size(gdV), block_dim_x):
            if cute.elem_less(cdV[i], cute.select(problem_shape, mode=[1, 2])):
                gdV[i] = dV.element_type(0)

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: cutlass.Float32,
        normalization_scale: cutlass.Float32,
        mma_compute_dKdV_consumer,
        problem_shape_k_cur_batch: Int32,
    ):
        """Epilogue phase to store result from tensor memory to register, then global memory."""
        tidx, _, _ = cute.arch.thread_idx()
        _, K, D, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        # adi: TMEM_DK, TMEM_DV
        tmem_copy_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        load_op = cute.make_copy_atom(
            tmem_copy_op,
            self.acc_dtype,
        )

        tdKtdK = tdKtdK[(None, None), 0, 0]
        mdK_offset = cute.assume(blk_offset[1] * dK.stride[0], divby=64)
        mdK = cute.make_tensor(
            dK.iterator + mdK_offset,
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(mdK, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thread_t2r_dK.partition_D(cdK)
        tTR_cdK = split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_gdK = thread_t2r_dK.partition_D(gdK)
        tTR_gdK = split_wg(tTR_gdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thread_t2r_dK.partition_S(tdKtdK)
        tTR_tdK = split_wg(tTR_tdK, num_warp_groups, wg_idx)

        mdV_in = cute.make_tensor(dV.iterator, cute.make_layout((K, self.cta_tiler[2], HB), stride=dV.stride))
        offset_mdV = cute.assume(blk_offset[1] * mdV_in.stride[0], divby=64)
        mdV = cute.make_tensor(mdV_in.iterator + offset_mdV, mdV_in.layout)
        gdV = cute.local_tile(mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]

        cdV = cute.domain_offset(
            (blk_coord_k * self.cta_tiler[1], 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        tdVtdV = tdVtdV[(None, None), 0, 0]

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        tTR_cdV = thread_t2r_dV.partition_D(cdV)
        tTR_cdV = split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_gdV = thread_t2r_dV.partition_D(gdV)
        tTR_gdV = split_wg(tTR_gdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thread_t2r_dV.partition_S(tdVtdV)
        tTR_tdV = split_wg(tTR_tdV, num_warp_groups, wg_idx)

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()

        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)
            for i in cutlass.range(cute.size(tTR_rdV), unroll_full=True):
                tTR_rdV[i] = normalization_scale * tTR_rdV[i]
            self.store(tTR_gdV, tTR_rdV, tTR_cdV, (K, D))

        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()

        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

            for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
                tTR_rdK[i] = scale_softmax * normalization_scale * tTR_rdK[i]

            self.store(tTR_gdK, tTR_rdK, tTR_cdK, (K, D))

        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()

        return mma_compute_dKdV_consumer

    @staticmethod
    def _compute_bwd_grid(
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        block_k: int,
    ) -> tuple[Int32, Int32, Int32]:
        """Compute grid shape for bwd kernel."""
        K = problem_shape[1]
        _, H_K = problem_shape[3][0]
        B = problem_shape[3][1]
        return (cute.ceil_div(K, block_k), cute.size(H_K), cute.size(B))

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for load mma Q."""
        load_mma_Q_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_load_mma_dO_pipeline(self, load_mma_dO_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for load mma dO."""
        load_mma_dO_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_dO_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_dO_mbar_ptr,
            num_stages=self.load_mma_dO_stage,
            producer_group=load_mma_dO_producer_group,
            consumer_group=load_mma_dO_consumer_group,
            tx_count=self.tma_copy_dO_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma S."""
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    #  Barrier to between dP = v * dO and consume of dP in compute()
    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma Q."""
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma P."""
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
            # self.num_compute_warps * self.threads_per_warp,
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
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma dS."""

        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
            # self.num_compute_warps * self.threads_per_warp,
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
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_mma_compute_dKdV_pipeline(self, mma_compute_dKdV_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma dKdV."""
        mma_compute_dKdV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dKdV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
            # self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dKdV_mbar_ptr,
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=mma_compute_dKdV_producer_group,
            consumer_group=mma_compute_dKdV_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
