# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

from __future__ import annotations

import math
from typing import Tuple, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.pipeline
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.typing import Int32, Float32, Boolean
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from . import utils
from .mask import AttentionMask
from .seqlen_info import SeqlenInfo
from .block_info import BlockInfo
from . import blackwell_helpers as sm100_utils
from .fast_math import FastDivmod, FastSilU
from .tile_scheduler import TileSchedulerArguments, SingleTileVarlenScheduler, ParamsBase
from .named_barrier import NamedBarrierFwd
from .block_sparsity import (
    HSTUBlockSparseTensors,
    get_q2k_block_for_reverse_slot,
    get_q2k_block_sparse_consumer_row,
)


class HSTUAttentionForwardSm100:

    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_arbitrary: bool = False,
        is_paged: bool = False,
        func_num: int = 0,
        kBlockM: int = 128,
        kBlockN: int = 128,
        is_persistent: bool = True,
        use_auto_block_metadata: bool = False,
    ):
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.kBlockM = kBlockM
        self.kBlockN = kBlockN
        # q_stage=1 for head_dim>=256 (TMEM capacity 512 cols can only fit 1 O tile)
        self.q_stage = 1 if self.head_dim_padded >= 256 else 2
        self.s_stage = 2  # score stage for intra-warp overlap
        assert self.q_stage in [1, 2]
        assert self.s_stage in [2]

        self.use_auto_block_metadata = use_auto_block_metadata
        self.enable_offset_dynamic = self.q_stage == 2 and not self.use_auto_block_metadata
        # q_stage Q tile per CTA
        self.cta_tiler = (self.q_stage * kBlockM, kBlockN, self.head_dim_padded)
        self.mma_tiler_qk = (kBlockM, kBlockN, self.head_dim_padded)
        self.mma_tiler_pv = (kBlockM, self.head_dim_v_padded, kBlockN)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_arbitrary = is_arbitrary
        self.is_paged = is_paged
        self.func_num = func_num
        assert not (self.is_arbitrary and (self.is_causal or self.is_local)), "a and b cannot both be True"
        assert self.use_auto_block_metadata == self.is_arbitrary
        self.qhead_per_kvhead = qhead_per_kvhead
        # Does S1 need to wait for S0 to finish
        self.s0_s1_barrier = False  # Performance drop,
        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or (self.head_dim_padded >= 256) or (self.is_arbitrary and self.q_stage == 2)
        )
        if self.overlap_sO_sQ:
            assert self.head_dim_padded >= self.head_dim_v_padded  # We assume sQ is larger than sO
            self.is_persistent = False

        self.silu0_warp_ids = (0, 1, 2, 3)
        self.silu1_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_ids = (10, 11)

        # Register budget tuning: SiLU warps get the most registers to reduce spills.
        # D256 R2P shifts eight registers to the load/MMA/empty warpgroup.  The
        # 2 * 224 + 48 = 496 allocation keeps headroom for SETMAXNREG progress
        # while avoiding the stack slot produced by the 232/40 split.
        if self.head_dim_padded >= 256 and self.use_auto_block_metadata:
            self.num_regs_silu = 224
            self.num_regs_other = 48
        elif self.head_dim_padded >= 128:
            self.num_regs_silu = 232
            self.num_regs_other = 40
        else:
            self.num_regs_silu = 224
            self.num_regs_other = 48
        self.num_regs_empty = 40
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.silu0_warp_ids,
                *self.silu1_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_ids,
            )
        )

        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.tmem_s_offset = [i * self.kBlockN for i in range(self.s_stage)]
        self.tmem_o_offset = [self.tmem_s_offset[-1] + self.kBlockN + i * self.head_dim_v_padded for i in range(self.q_stage)]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        self.tmem_s_to_p_offset = self.kBlockN // 2
        self.tmem_p_offset = [self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(self.s_stage)]
        # split_P_arrive: signal MMA to start PV when this many P columns are written.
        # 75% (96/128) matches FA4's proven configuration, allowing PV MMA to overlap
        # with the last 25% of P computation.
        self.split_P_arrive = self.kBlockN // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32

        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        # Derive kv_stage from smem budget (224KB limit), matching FA4's approach
        smem_size_q = self.q_stage * self.kBlockM * self.head_dim_padded * self.q_dtype.width // 8
        smem_size_o = self.q_stage * self.kBlockM * self.head_dim_v_padded * self.q_dtype.width // 8
        smem_size_q_o = max(smem_size_q, smem_size_o) if self.overlap_sO_sQ else (smem_size_q + smem_size_o)
        smem_size_kv_per_stage = max(
            self.kBlockN * self.head_dim_padded * self.q_dtype.width // 8,
            self.kBlockN * self.head_dim_v_padded * self.q_dtype.width // 8,
        )
        if self.q_dtype.width == 8:
            self.kv_stage = 4
        else:
            self.kv_stage = min((224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage, 3)
        assert self.kv_stage >= 2
        # self.acc_stage = 1  # question about it
        self.epi_stage = self.q_stage
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        self.uneven_kv_smem_offset = self.kBlockM * (self.head_dim_padded - self.head_dim_v_padded) // 2 if self.uneven_kv_smem else 0
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        score_scale: Float32,
        scaling_seqlen: Float32,
        stream: cuda.CUstream,
        window_size_left: Int32 | int,
        window_size_right: Int32 | int,
        func: Optional[cute.Tensor],
        mPagedKV: Optional[cute.Tensor],
        page_ids: Optional[cute.Tensor],
        page_indptrs: Optional[cute.Tensor],
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        buffers=None,  # Not typing for now since conversion behaves a lil funny
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (*(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]), t.stride[-1])
        mQ, mK, mV, mO = [cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t))) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [0, 2, 1]
        mQ, mO = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=QO_layout_transpose)) for t in (mQ, mO)]
        KV_layout_transpose = [0, 2, 1]
        mK, mV = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose)) for t in (mK, mV)]
        V_layout_transpose = [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        mPagedKV = cute.make_tensor(mPagedKV.iterator, cute.make_layout(mPagedKV.shape, stride=new_stride(mPagedKV))) if mPagedKV is not None else None
        mPagedK = cute.make_tensor(mPagedKV.iterator, cute.select(mPagedKV.layout, mode=KV_layout_transpose)) if mPagedKV is not None else None
        mPagedV = cute.make_tensor(mPagedKV.iterator, cute.select(mPagedKV.layout, mode=KV_layout_transpose)) if mPagedKV is not None else None
        mPagedV = cute.make_tensor(mPagedV.iterator, cute.select(mPagedV.layout, mode=V_layout_transpose)) if mPagedV is not None else None

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.mma_tiler_pv[:2]

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.q_dtype,
            1,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(max(sK_layout.outer.stride[-1], 0))  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(max(stride_sK, stride_sV) if not self.uneven_kv_smem else (stride_sK + stride_sV) // 2)
            sK_layout = cute.make_composed_layout(
                sK_layout.inner, 0, cute.make_layout((*sK_layout.outer.shape[:-1], self.kv_stage), stride=(*sK_layout.outer.stride[:-1], stage_stride))
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner, 0, cute.make_layout((*sV_layout.outer.shape[:-1], self.kv_stage), stride=(*sV_layout.outer.stride[:-1], stage_stride))
            )

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for PagedKV
        tma_atom_Kp, tma_tensor_Kp = None, None
        tma_atom_Vp, tma_tensor_Vp = None, None
        if const_expr(self.is_paged):
            tma_atom_Kp, tma_tensor_Kp = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mPagedK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_Vp, tma_tensor_Vp = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mPagedV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )

        num_epilogue_threads = cute.arch.WARP_SIZE * len(self.silu1_warp_ids)
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.o_dtype.width
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.o_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
        tO_layout = cute.make_ordered_layout(
            (num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.kBlockM % tO_layout.shape[0] == 0
        vO_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        self.tma_copy_k_bytes = cute.size_in_bytes(self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        self.tma_copy_v_bytes = cute.size_in_bytes(self.v_dtype, cute.select(sV_layout, mode=[0, 1, 2]))

        TileScheduler = SingleTileVarlenScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(cu_seqlens_q.shape[0] - 1),
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            cu_seqlens_q=cu_seqlens_q,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + self.s_stage
        self.mbar_O_full_offset = self.mbar_S_full_offset + self.s_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_O_full_offset + self.s_stage
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + self.q_stage
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        self.mbar_total = self.mbar_P_full_2_offset + self.s_stage

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            # sV reused by sK
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if cutlass.const_expr(buffers is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmod.create(seqlen_q)
            seqlen_k_divmod = FastDivmod.create(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        # Launch the kernel synchronously
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            max_seqlen_q,
            max_seqlen_k,
            cu_seqlens_q,
            cu_seqlens_k,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            score_scale,
            scaling_seqlen,
            window_size_left,
            window_size_right,
            func,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            buffers,
            fastdiv_mods,
            tma_atom_Kp,
            tma_atom_Vp,
            tma_tensor_Kp,
            tma_tensor_Vp,
            page_ids,
            page_indptrs,
            block_sparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        score_scale: Float32,
        scaling_seqlen: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        func: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        buffers=None,
        fastdiv_mods=(None, None),
        tma_atom_Kp: Optional[cute.CopyAtom] = None,
        tma_atom_Vp: Optional[cute.CopyAtom] = None,
        mPagedK: Optional[cute.Tensor] = None,
        mPagedV: Optional[cute.Tensor] = None,
        page_ids: Optional[cute.Tensor] = None,
        page_indptrs: Optional[cute.Tensor] = None,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. SiLU warps: Compute silu and apply mask on attention scores
        4. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Use the first N warps to initialize barriers
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_full_offset + i, len([self.load_warp_id]))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id]))
        if warp_idx == 2:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE * len(self.silu0_warp_ids))
        if warp_idx == 3:
            for i in cutlass.range_constexpr(self.s_stage):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_O_rescaled_offset + i, cute.arch.WARP_SIZE * (len(self.silu0_warp_ids)))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id]))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id]))
        if warp_idx == 4:
            for i in cutlass.range_constexpr(self.s_stage):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_2_offset + i, cute.arch.WARP_SIZE * len(self.silu0_warp_ids))
        if warp_idx == 5:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.silu0_warp_ids,
                        *self.silu1_warp_ids,
                    )
                ),
            )
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)  # full and empty

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # sQ_pi = storage.sQ.get_tensor(sQ_layout)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # sK_pi = storage.sK.get_tensor(sK_layout)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)

        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner), sO_layout.outer)

        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM

        qk_acc_shape = thr_mma_qk.partition_shape_C((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C((self.mma_tiler_pv[0], self.mma_tiler_pv[1]))
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout) for stage in range(self.s_stage))
        tOtOs = tuple(cute.make_tensor(tStSs[0].iterator + self.tmem_o_offset[stage], tOtO.layout) for stage in range(self.q_stage))

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrPs = [
            cute.make_tensor(
                tOrP.iterator + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p_offset[stage],
                tOrP.layout,
            )
            for stage in range(self.s_stage)
        ]

        block_info = BlockInfo(
            self.cta_tiler,
            self.is_causal,
            self.is_local,
            self.is_paged,
            window_size_left,
            window_size_right,
        )
        SeqlenInfoCls = partial(
            SeqlenInfo,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_indptrs=page_indptrs,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            kBlockM=self.kBlockM,
            kBlockN=self.kBlockN,
            is_arbitrary=self.is_arbitrary,
            is_causal=self.is_causal,
            is_local=self.is_local,
            func_num=self.func_num,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            swapAB=False,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx >= self.empty_warp_ids[0] and warp_idx <= self.empty_warp_ids[-1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            if const_expr(not self.is_paged):
                self.load(
                    thr_mma_qk,
                    thr_mma_pv,
                    mQ,
                    mK,
                    mV,
                    sQ,
                    sK,
                    sV,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    pipeline_kv,
                    mbar_ptr,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
            else:
                self.load_paged(
                    thr_mma_qk,
                    thr_mma_pv,
                    mQ,
                    sQ,
                    sK,
                    sV,
                    tma_atom_Q,
                    pipeline_kv,
                    mbar_ptr,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    tma_atom_Kp,
                    tma_atom_Vp,
                    mPagedK,
                    mPagedV,
                    page_ids,
                )
        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
                cute.arch.sync_warp()

            if const_expr(self.q_stage == 2):
                self.mma(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    sQ,
                    sK,
                    sV,
                    sQ_layout.inner,
                    sK_layout.inner,
                    sV_layout.inner,
                    tStSs,
                    tOtOs,
                    tOrPs,
                    pipeline_kv,
                    mbar_ptr,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
            else:
                self.mma_intraoverlap(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    sQ,
                    sK,
                    sV,
                    sQ_layout.inner,
                    sK_layout.inner,
                    sV_layout.inner,
                    tStSs,
                    tOtOs,
                    tOrPs,
                    pipeline_kv,
                    mbar_ptr,
                    block_info,
                    block_sparse_tensors,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  SilU
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.silu0_warp_ids[0] and warp_idx <= self.silu1_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_silu)
            store_O = partial(
                self.store_O,
                gmem_tiled_copy_O=gmem_tiled_copy_O,
                thr_mma_pv=thr_mma_pv,
                tOtOs=tOtOs,
                mO=mO,
                sO=sO,
                mbar_ptr=mbar_ptr,
            )
            silu_loop = partial(
                self.silu_loop,
                score_scale=score_scale,
                scaling_seqlen=scaling_seqlen,
                thr_mma_qk=thr_mma_qk,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                block_sparse_tensors=block_sparse_tensors,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                store_O=store_O,
                func=func,
                buffers=buffers,
                fastdiv_mods=fastdiv_mods,
            )
            if warp_idx <= self.silu0_warp_ids[-1] and warp_idx >= self.silu0_warp_ids[0]:
                tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
                silu_loop(stage=0, tStSi=tStSi)
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            if warp_idx <= self.silu1_warp_ids[-1] and warp_idx >= self.silu1_warp_ids[0]:
                tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[1], tStS.layout)
                silu_loop(stage=1, tStSi=tStSi)
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.kv_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            offset = seqlen.offset_q
            offset_dynamic = (self.cta_tiler[0] - (seqlen.seqlen_q & (self.cta_tiler[0] - 1))) & (self.cta_tiler[0] - 1)
            offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
            mQ_cur = cute.domain_offset((offset - offset_dynamic, 0), mQ[None, None, head_idx])
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))

            head_idx_kv = head_idx // self.qhead_per_kvhead

            mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
            mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))

            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ, 0, 3),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            load_Q = partial(
                self.load_Q,
                tma_atom_Q,
                tQgQ,
                tQsQ,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
            )
            n_block_max, n_block_min, _ = block_info.get_n_block_info(seqlen, m_block, offset_dynamic)
            mask_block_cnt = None
            mask_block_idx = None
            full_block_cnt = None
            full_block_idx = None
            if const_expr(self.use_auto_block_metadata):
                (
                    n_block_max,
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                ) = get_q2k_block_sparse_consumer_row(
                    block_sparse_tensors,
                    batch_idx,
                    m_block,
                )
                n_block_min = Int32(0)
            has_work = n_block_max > n_block_min
            if has_work:
                if const_expr(self.q_stage == 2):
                    n_block_valid = n_block_max - 1
                    if const_expr(self.use_auto_block_metadata):
                        n_block, _ = get_q2k_block_for_reverse_slot(
                            n_block_max,
                            mask_block_cnt,
                            mask_block_idx,
                            full_block_cnt,
                            full_block_idx,
                            n_block_valid,
                        )
                    else:
                        n_block = n_block_valid
                    load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                    load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)  # K0
                    kv_producer_state.advance()
                    if const_expr(self.q_stage == 2):
                        load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
                    q_producer_phase ^= 1
                    load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)  # V0
                    kv_producer_state.advance()
                    n_block_valid -= 1
                    while n_block_valid >= n_block_min:
                        if const_expr(self.use_auto_block_metadata):
                            n_block, _ = get_q2k_block_for_reverse_slot(
                                n_block_max,
                                mask_block_cnt,
                                mask_block_idx,
                                full_block_cnt,
                                full_block_idx,
                                n_block_valid,
                            )
                        else:
                            n_block = n_block_valid
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)  # Ki
                        kv_producer_state.advance()
                        load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)  # Vi
                        kv_producer_state.advance()
                        n_block_valid -= 1
                elif const_expr(self.q_stage == 1):
                    n_block_valid_k = n_block_max - 1
                    n_block_valid_v = n_block_max - 1
                    if const_expr(self.use_auto_block_metadata):
                        n_block_k, _ = get_q2k_block_for_reverse_slot(
                            n_block_max,
                            mask_block_cnt,
                            mask_block_idx,
                            full_block_cnt,
                            full_block_idx,
                            n_block_valid_k,
                        )
                    else:
                        n_block_k = n_block_valid_k
                    load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                    q_producer_phase ^= 1
                    load_K(block=n_block_k, producer_state=kv_producer_state, page_idx=None)  # K0
                    kv_producer_state.advance()
                    n_block_valid_k -= 1
                    if n_block_valid_k >= n_block_min:
                        if const_expr(self.use_auto_block_metadata):
                            n_block_k, _ = get_q2k_block_for_reverse_slot(
                                n_block_max,
                                mask_block_cnt,
                                mask_block_idx,
                                full_block_cnt,
                                full_block_idx,
                                n_block_valid_k,
                            )
                        else:
                            n_block_k = n_block_valid_k
                        load_K(block=n_block_k, producer_state=kv_producer_state, page_idx=None)  # K1
                        kv_producer_state.advance()
                        n_block_valid_k -= 1

                    # load mainloop, V0 K2 V1 K3... Vi K(i+2)
                    while n_block_valid_k >= n_block_min:
                        if const_expr(self.use_auto_block_metadata):
                            n_block_k, _ = get_q2k_block_for_reverse_slot(
                                n_block_max,
                                mask_block_cnt,
                                mask_block_idx,
                                full_block_cnt,
                                full_block_idx,
                                n_block_valid_k,
                            )
                            n_block_v, _ = get_q2k_block_for_reverse_slot(
                                n_block_max,
                                mask_block_cnt,
                                mask_block_idx,
                                full_block_cnt,
                                full_block_idx,
                                n_block_valid_v,
                            )
                        else:
                            n_block_k = n_block_valid_k
                            n_block_v = n_block_valid_v
                        load_V(block=n_block_v, producer_state=kv_producer_state, page_idx=None)  # V1
                        kv_producer_state.advance()
                        n_block_valid_v -= 1
                        load_K(block=n_block_k, producer_state=kv_producer_state, page_idx=None)  # Ki
                        kv_producer_state.advance()
                        n_block_valid_k -= 1

                    # load epilogue, V1 V0
                    while n_block_valid_v >= n_block_min:
                        if const_expr(self.use_auto_block_metadata):
                            n_block_v, _ = get_q2k_block_for_reverse_slot(
                                n_block_max,
                                mask_block_cnt,
                                mask_block_idx,
                                full_block_cnt,
                                full_block_idx,
                                n_block_valid_v,
                            )
                        else:
                            n_block_v = n_block_valid_v
                        load_V(block=n_block_v, producer_state=kv_producer_state, page_idx=None)  # V1
                        kv_producer_state.advance()
                        n_block_valid_v -= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def load_paged(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        tma_atom_Kp: cute.CopyAtom,
        tma_atom_Vp: cute.CopyAtom,
        mPagedK: cute.Tensor,
        mPagedV: cute.Tensor,
        page_ids: cute.Tensor,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.kv_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            offset = seqlen.offset_q
            offset_dynamic = (self.cta_tiler[0] - (seqlen.seqlen_q & (self.cta_tiler[0] - 1))) & (self.cta_tiler[0] - 1)
            offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
            mQ_cur = cute.domain_offset((offset - offset_dynamic, 0), mQ[None, None, head_idx])
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))
            page_ind = seqlen.page_ind

            head_idx_kv = head_idx // self.qhead_per_kvhead

            mK_paged = mPagedK[None, None, head_idx_kv]  # (#tokens, headdim, head_idx)
            mV_paged = mPagedV[None, None, head_idx_kv]  # (headdim, #tokens, head_idx)
            gK_paged = cute.local_tile(mK_paged, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV_paged = cute.local_tile(mV_paged, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))

            tSgQ = thr_mma_qk.partition_A(gQ)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ, 0, 3),
            )

            tSgKp = thr_mma_qk.partition_B(gK_paged)
            tOgVp = thr_mma_pv.partition_B(gV_paged)
            tKsKp, tKgKp = cpasync.tma_partition(
                tma_atom_Kp,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgKp, 0, 3),
            )
            tVsVp, tVgVp = cpasync.tma_partition(
                tma_atom_Vp,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgVp, 0, 3),
            )

            load_Q = partial(
                self.load_Q,
                tma_atom_Q,
                tQgQ,
                tQsQ,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_Kp = partial(
                self.load_KV,
                tma_atom_Kp,
                tKgKp,
                tKsKp,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
            )
            load_Vp = partial(
                self.load_KV,
                tma_atom_Vp,
                tVgVp,
                tVsVp,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
            )
            n_block_max, n_block_min, _ = block_info.get_n_block_info(seqlen, m_block, offset_dynamic)
            n_block = n_block_max - 1

            if const_expr(self.q_stage == 2):
                load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                page_idx = page_ids[n_block + page_ind] * 2
                load_Kp(block=page_idx, producer_state=kv_producer_state)  # K0
                kv_producer_state.advance()
                load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
                q_producer_phase ^= 1
                load_Vp(block=page_idx + 1, producer_state=kv_producer_state)  # V0
                kv_producer_state.advance()
                n_block_valid = n_block - 1

                while n_block_valid >= n_block_min:
                    n_block = n_block_valid
                    page_idx = page_ids[n_block + page_ind] * 2
                    load_Kp(block=page_idx, producer_state=kv_producer_state)  # Ki
                    kv_producer_state.advance()
                    v_page_idx = page_ids[n_block + page_ind] * 2 + 1
                    load_Vp(block=v_page_idx, producer_state=kv_producer_state)  # Vi
                    kv_producer_state.advance()
                    n_block_valid -= 1

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            elif const_expr(self.q_stage == 1):
                load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                q_producer_phase ^= 1
                page_idx_k = page_ids[n_block + page_ind] * 2
                load_Kp(block=page_idx_k, producer_state=kv_producer_state)  # K0
                page_idx_v = page_idx_k + 1
                kv_producer_state.advance()

                while n_block - 1 >= n_block_min:
                    page_idx_k = page_ids[n_block - 1 + page_ind] * 2
                    load_Kp(block=page_idx_k, producer_state=kv_producer_state)  # K0
                    kv_producer_state.advance()
                    load_Vp(block=page_idx_v, producer_state=kv_producer_state)
                    kv_producer_state.advance()
                    n_block -= 1
                    page_idx_v = page_idx_k + 1

                load_Vp(block=page_idx_v, producer_state=kv_producer_state)  # V0

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            ### End of persistent scheduler loop

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ_swizzle: cute.Swizzle,
        sK_swizzle: cute.Swizzle,
        sV_swizzle: cute.Swizzle,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM
        tSrQ = thr_mma_qk.make_fragment_A(sQ)
        tSrK = thr_mma_qk.make_fragment_B(sK)
        tOrV = thr_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_partial,
                qk_mma_op,
                self.tmem_s_offset[stage],
                tSrQs[stage],
                sA=sQ[None, None, None, stage],
                sA_swizzle=sQ_swizzle,
                sB_swizzle=sK_swizzle,
                zero_init=True,
            )
            for stage in range(2)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
                sA_swizzle=None,
                sB_swizzle=sV_swizzle,
                split_arrive=self.split_P_arrive,
            )
            for stage in range(2)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage)
        P_full_O_rescaled_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            offset_dynamic = (self.cta_tiler[0] - (seqlen.seqlen_q & (self.cta_tiler[0] - 1))) & (self.cta_tiler[0] - 1)
            offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
            n_block_max, n_block_min, _ = block_info.get_n_block_info(seqlen, m_block, offset_dynamic)
            if const_expr(self.use_auto_block_metadata):
                n_block_max, _, _, _, _ = get_q2k_block_sparse_consumer_row(
                    block_sparse_tensors,
                    batch_idx,
                    m_block,
                )
                n_block_min = Int32(0)
            n_block_nums = n_block_max - n_block_min

            if n_block_nums > 0:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase)
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase)
                    gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    # 4. release S0 / S1
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                O_should_accumulate = False
                for i in cutlass.range(n_block_nums - 1, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(2):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile has finished.
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase)
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        # 4. release accumulated O0_partial / O1_partial
                        # Don't need to signal O_full to the correction warps anymore since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # with cute.arch.elect_one():
                        #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        # 5. release V(i-1)
                        if const_expr(stage == 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrK[None, None, None, Ki_index], zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                        # 3. release S0
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.q_stage):
                        tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase)
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warp, the softmax warp has just finished compute
                    # the row sum of the current tile. It does not guarantee that the 1st tile
                    # of the next work tile has been computed yet.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def mma_intraoverlap(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ_swizzle: cute.Swizzle,
        sK_swizzle: cute.Swizzle,
        sV_swizzle: cute.Swizzle,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        assert self.q_stage == 1
        assert self.s_stage == 2  # the prologue and epilogue do not use a for loop for the S stage.
        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM
        tSrQ = thr_mma_qk.make_fragment_A(sQ)
        tSrK = thr_mma_qk.make_fragment_B(sK)
        tOrV = thr_mma_pv.make_fragment_B(sV)
        tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_partial,
                qk_mma_op,
                self.tmem_s_offset[stage],
                tSrQs[stage],
                sA=sQ[None, None, None, stage],
                sA_swizzle=sQ_swizzle,
                sB_swizzle=sK_swizzle,
                zero_init=True,
            )
            for stage in range(self.s_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
                sA_swizzle=None,
                sB_swizzle=sV_swizzle,
                split_arrive=self.split_P_arrive,
            )
            for stage in range(self.s_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage)
        P_full_O_rescaled_phase = [Int32(0) for _ in range(self.s_stage)]

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            offset_dynamic = (self.cta_tiler[0] - (seqlen.seqlen_q & (self.cta_tiler[0] - 1))) & (self.cta_tiler[0] - 1)
            offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
            n_block_max, n_block_min, _ = block_info.get_n_block_info(seqlen, m_block, offset_dynamic)
            if const_expr(self.use_auto_block_metadata):
                n_block_max, _, _, _, _ = get_q2k_block_sparse_consumer_row(
                    block_sparse_tensors,
                    batch_idx,
                    m_block,
                )
                n_block_min = Int32(0)
            n_block_nums = n_block_max - n_block_min

            if n_block_nums > 0:
                # 1. wait for Q
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_full_offset, mma_q_consumer_phase)
                # 2. wait for K0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tSrKi = tSrK[None, None, None, Ki_index]
                sK_cur = sK[None, None, None, Ki_index]
                if const_expr(self.uneven_kv_smem):
                    sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                # 3. gemm S0=QK0
                gemm_Si[0](tCrB=tSrKi, sB=sK_cur)
                # 4. release S0
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + 0)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                if n_block_nums > 1:  # GEMM_QK1 (Q1 * K1 -> S1)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tSrKi = tSrK[None, None, None, Ki_index]
                    sK_cur = sK[None, None, None, Ki_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                    gemm_Si[1](tCrB=tSrKi, sB=sK_cur)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + 1)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                O_should_accumulate = False
                for i in cutlass.range(n_block_nums - 2, unroll=1):  # GEMM_P0V0, GEMM_QK2 ...-> GEMM_PiVi GEMM_QK(i+2)
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    if i & (self.s_stage - 1) == 0:
                        # 2. acquire P_partial
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0, P_full_O_rescaled_phase[0])
                        # 3. gemm PiVi=Pi*Vi
                        gemm_Pi[0](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 0,
                            mbar_phase=P_full_O_rescaled_phase[0],
                        )
                        P_full_O_rescaled_phase[0] ^= 1
                    else:
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1, P_full_O_rescaled_phase[1])
                        gemm_Pi[1](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 1,
                            mbar_phase=P_full_O_rescaled_phase[1],
                        )
                        P_full_O_rescaled_phase[1] ^= 1
                    # 4. release Vi
                    pipeline_kv.consumer_release(mma_kv_release_state)
                    mma_kv_release_state.advance()

                    mma_kv_consumer_state.advance()
                    # 5. wait for K(i+2)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    sK_cur = sK[None, None, None, Ki_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                    if i & (self.s_stage - 1) == 0:
                        gemm_Si[0](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                    else:
                        gemm_Si[1](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + i % self.s_stage)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset)

                if n_block_nums > 1:
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    if (n_block_nums - 2) & (self.s_stage - 1) == 0:
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0, P_full_O_rescaled_phase[0])
                        gemm_Pi[0](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 0,
                            mbar_phase=P_full_O_rescaled_phase[0],
                        )
                        P_full_O_rescaled_phase[0] ^= 1
                    else:
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1, P_full_O_rescaled_phase[1])
                        gemm_Pi[1](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 1,
                            mbar_phase=P_full_O_rescaled_phase[1],
                        )
                        P_full_O_rescaled_phase[1] ^= 1
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    O_should_accumulate = True

                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                sV_cur = sV[None, None, None, Vi_index]
                if const_expr(self.uneven_kv_smem):
                    sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                if (n_block_nums - 1) & (self.s_stage - 1) == 0:
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0, P_full_O_rescaled_phase[0])
                    gemm_Pi[0](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 0,
                        mbar_phase=P_full_O_rescaled_phase[0],
                    )
                    P_full_O_rescaled_phase[0] ^= 1
                else:
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1, P_full_O_rescaled_phase[1])
                    gemm_Pi[1](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 1,
                        mbar_phase=P_full_O_rescaled_phase[1],
                    )
                    P_full_O_rescaled_phase[1] ^= 1
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_ptr + self.mbar_O_full_offset)
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        # End of persistent scheduler loop

    @cute.jit
    def silu_loop(
        self,
        stage: int | Int32,
        score_scale: Float32,
        scaling_seqlen: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        store_O: Callable,
        func: Optional[cute.Tensor],
        buffers=None,
        fastdiv_mods=(None, None),
    ):
        """Compute silu on attention scores from QK matrix multiplication."""
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * (len(self.silu0_warp_ids)))

        cS_base = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        tScS = thr_mma_qk.partition_C(cS_base)

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(tStSi.layout, cute.make_layout((self.kBlockM, tilePlikeFP32)))
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
            Float32,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP)
        thr_tmem_store = tiled_tmem_store.get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        epi_consumer_phase = Int32(0)
        mma_si_consumer_phase = Int32(0)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            q_work_block = m_block
            seqlen = SeqlenInfoCls(batch_idx)
            offset_dynamic = (self.cta_tiler[0] - (seqlen.seqlen_q & (self.cta_tiler[0] - 1))) & (self.cta_tiler[0] - 1)
            offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
            n_block_max, n_block_min, n_masking_steps = block_info.get_n_block_info(seqlen, m_block, offset_dynamic)
            mask_block_cnt = None
            mask_block_idx = None
            full_block_cnt = None
            full_block_idx = None
            if const_expr(self.use_auto_block_metadata):
                (
                    n_block_max,
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                ) = get_q2k_block_sparse_consumer_row(
                    block_sparse_tensors,
                    batch_idx,
                    q_work_block,
                )
                n_block_min = Int32(0)
            has_work = n_block_max > n_block_min
            # func: (head_func, n_func, L_func) -> (n_func, L_func)
            func_tensor = func[0, None, None] if func is not None else None

            # m_block consider q stage
            m_block = self.q_stage * m_block + (stage if self.q_stage == 2 else 0)
            mask = AttentionMaskCls(
                offset_q=seqlen.offset_q, seqlen_q=seqlen.seqlen_q, seqlen_k=seqlen.seqlen_k, offset_dynamic=offset_dynamic, func=func_tensor
            )
            mask_fn = partial(mask.apply_mask, m_block=m_block, thr_mma=thr_mma_qk, thr_tmem_load=thr_tmem_load)
            r2p_mask_fn = partial(
                mask.build_mask_r2p,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
            )
            seqlen_mask_fn = partial(
                mask.apply_mask_seqlen,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
            )
            fastsilu = FastSilU(
                score_scale=score_scale,
            )

            silu_step = partial(
                self.silu_step,
                fastsilu=fastsilu,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                seqlen=seqlen,
                buffers=buffers,
                fastdiv_mods=fastdiv_mods,
            )
            wg_stride = 1 if self.q_stage == 2 else 2
            n_block_valid = n_block_max - 1 - (0 if self.q_stage == 2 else stage)
            masking_step = 0 if self.q_stage == 2 else stage

            if const_expr(self.use_auto_block_metadata):
                # MASK and FULL are separate compile-time call sites.  For
                # D256 (q_stage=1), ``stage`` is the global traversal parity:
                # the FULL start must continue the parity established by the
                # preceding MASK group rather than restarting at local index 0.
                mask_iteration = Int32(0 if self.q_stage == 2 else stage)
                while mask_iteration < mask_block_cnt:
                    n_block = mask_block_idx[mask_block_cnt - 1 - mask_iteration]
                    if const_expr(self.func_num <= 7):
                        mma_si_consumer_phase, s0_s1_sequence_phase = silu_step(
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                            r2p_mask_fn=partial(r2p_mask_fn),
                        )
                    else:
                        # The public API accepts any positive odd func count.
                        # Bound R2P endpoint GPRs to the profiled 1/3/5/7
                        # specializations and retain exact scalar masking for
                        # larger counts.
                        mma_si_consumer_phase, s0_s1_sequence_phase = silu_step(
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                            mask_fn=partial(mask_fn),
                        )
                    mask_iteration += wg_stride

                full_iteration = Int32(0)
                if const_expr(self.q_stage == 1):
                    full_iteration = (Int32(stage) - mask_block_cnt) & Int32(1)
                while full_iteration < full_block_cnt:
                    n_block = full_block_idx[full_block_cnt - 1 - full_iteration]
                    is_tail_block = Boolean((m_block + 1) * self.kBlockM > seqlen.seqlen_q or (n_block + 1) * self.kBlockN > seqlen.seqlen_k)
                    if is_tail_block:
                        (
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                        ) = silu_step(
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                            mask_fn=partial(seqlen_mask_fn),
                        )
                    else:
                        (
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                        ) = silu_step(
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                        )
                    full_iteration += wg_stride
            elif const_expr(self.is_local):
                while n_block_valid >= n_block_min:
                    n_block = n_block_valid
                    mma_si_consumer_phase, s0_s1_sequence_phase = silu_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block, mask_fn=partial(mask_fn))
                    masking_step += wg_stride
                    n_block_valid -= wg_stride

            if const_expr(not self.use_auto_block_metadata):
                while n_block_valid >= n_block_min and masking_step < n_masking_steps:
                    n_block = n_block_valid
                    mma_si_consumer_phase, s0_s1_sequence_phase = silu_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block, mask_fn=partial(mask_fn))
                    masking_step += wg_stride
                    n_block_valid -= wg_stride

                while n_block_valid >= n_block_min:
                    n_block = n_block_valid
                    mma_si_consumer_phase, s0_s1_sequence_phase = silu_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block)
                    n_block_valid -= wg_stride

            # epilogue step
            if self.q_stage == 2 or stage == 1:
                store_O(
                    seqlen=seqlen,
                    scale=cute.arch.rcp_approx(scaling_seqlen),
                    m_block=m_block,
                    head_idx=head_idx,
                    stage=stage if self.q_stage == 2 else 0,
                    epi_consumer_phase=epi_consumer_phase,
                    has_work=has_work,
                )

            # Advance to next tile
            if has_work:
                epi_consumer_phase ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def silu_step(
        self,
        mma_si_consumer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        fastsilu: FastSilU,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        buffers=None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        r2p_mask_fn: Optional[Callable] = None,
    ) -> Tuple[cute.Int32, cute.Int32]:
        """Perform a single step of the silu computation on a block of attention scores. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing SiLU
        4. Coordinating pipeline synchronization between different processing stages
        """
        assert mask_fn is None or r2p_mask_fn is None, "scalar and R2P masks are mutually exclusive"

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1])))
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((self.kBlockM, 1)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tScP_layout = cute.composition(tScS.layout, cute.make_layout((self.kBlockM, tilePlikeFP32)))
        tScP = cute.make_tensor(tScS.iterator, tScP_layout)

        tScS_t2r_shape = thr_tmem_load.partition_D(tScS).shape
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r_shape, self.qk_acc_dtype)
        tSrS_preds = None
        r2p_masks = None
        if const_expr(mask_fn is not None):
            tSrS_preds = cute.make_rmem_tensor(
                tScS_t2r_shape,
                cutlass.Boolean,
            )
            mask_fn(tSrS_preds, n_block=n_block)
        if const_expr(r2p_mask_fn is not None):
            r2p_masks = cute.make_rmem_tensor(
                (cute.size(tScS_t2r_shape) // 32,),
                cutlass.Uint32,
            )
            r2p_mask_fn(r2p_masks, n_block=n_block)

        # Wait for Si
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)  # copy from tmem to rmem
        cute.arch.fence_view_async_tmem_load()

        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
            tSrS_t2r.layout,
        )
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            # s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)
            # for stage 0, it does not need wait; for stage 1, it needs wait
            cute.arch.mbarrier_wait(mbar_ptr + mbar_s0_s1_sequence_offset + stage, s0_s1_sequence_phase)
        fastsilu.silu_x2(
            tSrS_t2r,
            tSrP_r2t,
            tSrS_preds,
            r2p_masks=r2p_masks,
            mask_fn=partial(mask_fn) if mask_fn is not None else None,
            r2p_mask_fn=partial(r2p_mask_fn) if r2p_mask_fn is not None else None,
        )
        # Write first portion of P (split_P_arrive columns), then signal MMA to start PV
        split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.kBlockN
        for i in cutlass.range_constexpr(split_P_arrive_idx):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()

        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage))
        # Notify mma warp that first portion of P is ready — MMA can start PV
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        # Write remaining P columns
        for i in cutlass.range_constexpr(split_P_arrive_idx, cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that all P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        return mma_si_consumer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def store_O(
        self,
        m_block: int,
        head_idx: int,
        stage: int,
        scale: float,
        epi_consumer_phase: Int32,
        seqlen: Callable,
        gmem_tiled_copy_O: cute.TiledCopy,
        thr_mma_pv: cute.ThrMma,
        tOtOs: cute.Tensor,
        mO: cute.Tensor,
        sO: cute.Tensor,
        mbar_ptr: cute.Pointer,
        has_work: Boolean,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.silu1_warp_ids))
        offset = seqlen.offset_q + m_block * self.kBlockM
        offset_dynamic = (self.cta_tiler[0] - (seqlen.seqlen_q & (self.cta_tiler[0] - 1))) & (self.cta_tiler[0] - 1)
        offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
        mO_cur = cute.domain_offset((offset - offset_dynamic, 0), mO[None, None, head_idx])
        gO = cute.local_tile(mO_cur, (self.kBlockM, self.head_dim_v_padded), (0, 0))
        cO = cute.make_identity_tensor((self.kBlockM, self.head_dim_v_padded))
        tOtO = tOtOs[stage]
        tOsO = thr_mma_pv.partition_C(sO[None, None, stage])
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.o_dtype.width

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.kBlockM, async_copy_elems)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.kBlockM, async_copy_elems)))
        epi_subtile = (self.epi_tile[0], async_copy_elems)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load)
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_r2s = thr_tmem_load.partition_D(tOsO_i[(None, None), None])

        if has_work:
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, epi_consumer_phase)
        for i in cutlass.range_constexpr(self.head_dim_v_padded // async_copy_elems):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_r2s[None, 0, 0, i]
            tOrO_frg_cvt = cute.make_rmem_tensor(
                tOsO_r2s[None, 0, 0, i].shape,
                self.o_dtype,
            )
            if has_work:
                tOrO_frg = cute.make_rmem_tensor(
                    tOsO_r2s[None, 0, 0, i].shape,
                    self.pv_acc_dtype,
                )
                cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
                for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                    tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                        (tOrO_frg[j], tOrO_frg[j + 1]),
                        (scale, scale),
                    )
                tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            else:
                tOrO_frg_cvt.fill(0)
            cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)

        cute.arch.barrier(barrier_id=NamedBarrierFwd.Epilogue + stage, number_of_threads=cute.arch.WARP_SIZE * len(self.silu1_warp_ids))

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO[None, None, stage])
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        base_row = m_block * self.kBlockM
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            # 1% better performance than tOcO[0, rest_m, 0][0] < seqlen.seqlen_q - (self.q_stage * m_block + stage) * self.kBlockM
            row = t0OcO[0, rest_m, 0][0] + tOcO[0][0] + base_row
            pred = row >= offset_dynamic if offset_dynamic > 0 else row < seqlen.seqlen_q
            if pred:
                cute.autovec_copy(tOsO[None, rest_m, None], tOrO[None, rest_m, None])
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None] if self.check_hdim_v_oob else None,
                )

    def load_Q(
        self,
        tma_atom: cute.CopyAtom,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_q_bytes)
        cute.copy(tma_atom, tQgQ[None, block], tQsQ[None, stage], tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: str,
        page_idx: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        tma_copy_bytes = self.tma_copy_k_bytes if const_expr(K_or_V == "K") else self.tma_copy_v_bytes
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, tma_copy_bytes)
        tXsX_cur = tXsX[None, stage]
        if const_expr(self.uneven_kv_smem):
            # Since this is the producer_state, the phase starts at 1, so we have to invert it
            tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
        # Currently we assume that page_size == kBlockN so we index into tXgX with block = 0
        tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
        cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.load_warp_id]))
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))
        return cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=load_kv_producer_group,
            consumer_group=load_kv_consumer_group,
            tx_count=self.tma_copy_k_bytes,
        )
