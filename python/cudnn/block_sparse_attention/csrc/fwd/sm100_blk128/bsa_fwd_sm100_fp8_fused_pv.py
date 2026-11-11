# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from NVIDIA-JerryChen/Block-Sparse-Attention, commit f953680ff93a2ef2321ad7dc4291451eea4a9c3c.
# SM100/SM103 FP8 blk128 with a rounded-P denominator fused into the existing PV.
# Common transfers, conversion and epilogue helpers remain in the reference class.
# Split producers retain FP32 output and natural-log normalization statistics.
import math
from functools import partial
from typing import Callable, Optional, Tuple
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
from cutlass import Boolean, Float32, Int32, Int64, const_expr, pipeline
from cutlass._mlir.dialects import llvm
from cudnn._cutlass_compat import Arch, LayoutEnum, SmemAllocator, TmemAllocator, OperandMajorMode
from cutlass.cutlass_dsl import BaseDSL, T
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cuda.bindings.driver as cuda
from cudnn.block_sparse_attention.csrc.fwd.sm100_blk64 import bsa_fwd_helpers as fp8_helpers
from cudnn.block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_sm100 import _SubsetNamedBarrier
from cudnn.block_sparse_attention.csrc.utils import (
    kernel_utils as utils,
    mma_sm100_desc as sm100_desc,
    pipeline as pipeline_custom,
)
from cudnn.block_sparse_attention.csrc.utils.block_sparse_tile_scheduler import (
    BlockSparsePersistentTileScheduler as TileScheduler,
    TileSchedulerProtocol,
)
from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import (
    ParamsBase,
    assume_tensor_aligned,
)
from cudnn.block_sparse_attention.csrc.utils.named_barrier import NamedBarrierFwdSm100
from cudnn.block_sparse_attention.csrc.utils.softmax import SoftmaxSm100
from cudnn.block_sparse_attention.csrc.utils.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
)

from cudnn.block_sparse_attention.csrc.fwd.sm100_blk128.bsa_fwd_sm100_fp8 import (
    BlockSparseAttnForwardFp8Sm100Blk128,
    BlockInfo,
    SeqlenInfoQK,
    _max16_tree,
    tmem_load_red_max_32dp32b16x,
    apply_block_size_mask,
    sm100_utils,
)


class BlockSparseAttnForwardFp8FusedPvSm100Blk128(BlockSparseAttnForwardFp8Sm100Blk128):
    """Preserve sparse semantics while computing rounded-P row sums with PV."""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        allow_empty_block_nums: bool = False,
        has_block_sizes: bool = True,
        block_sizes_mode: int = 1,
        num_splits: int = 1,
    ):
        supported_head_dims = (128,)
        assert head_dim in supported_head_dims, "Sage FP8 blk128 requires QK dim 128"
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        assert head_dim_v in supported_head_dims, "Sage FP8 blk128 requires value dim 128"

        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = self.tile_m
        self.n_block_size = self.tile_n
        self.q_stage = 1
        assert 1 <= num_splits <= 256
        self.num_splits = int(num_splits)
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert Arch.sm_100 <= self.arch <= Arch.sm_103f, "Sage FP8 blk128 requires SM100/SM103"

        self.cta_tiler = (self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (
            self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.m_block_size,
            self.head_dim_v_padded,
            self.n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        # Persistent scheduling through the cluster launch control (CLC) pipeline.
        self.sched_stages = 1
        self.allow_empty_block_nums = allow_empty_block_nums
        self.has_block_sizes = has_block_sizes
        self.block_sizes_mode = block_sizes_mode
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        self.enable_ex2_emu = self.head_dim_padded <= 128 and not is_sm103
        self.split_P_arrive = 0
        self.double_buffer_stats = not is_sm103
        self.triple_buffer_p = not is_sm103
        self.correction_tile_size = 64 if is_sm103 else 16
        self.epilogue_tile_size = 32 if is_sm103 else 16
        self.prefetch_all_v_scales = not is_sm103

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.clc_scheduler_warp_id = 15
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                self.clc_scheduler_warp_id,
            )
        )

        self.s_stage = 2  # Always 2: for q_stage=1 it's n-direction
        # SM100 uses three P32 slots within the existing 512-column allocation.
        # SM103 retains two slots; both paths keep one S128 and two O144 tiles.
        self.tmem_s_offset = [0, 0]
        self.tmem_o_offset = [128, 272]
        self.tmem_p_offset = [416, 448]
        self.p_stage = 3 if self.triple_buffer_p else 2
        self.tmem_total = 512 if self.triple_buffer_p else 480
        self.tmem_s_to_p_offset = 0
        assert self.tmem_total <= self.tmem_alloc_cols

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        self.num_regs_softmax = 184
        self.num_regs_correction = 88
        self.num_regs_other = 56

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = self.q_stage * self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_size_o = self.s_stage * self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        smem_size_q_o = smem_size_q + smem_size_o
        smem_size_k_per_stage = self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        smem_size_v_per_stage = self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        smem_size_kv_per_stage = max(smem_size_k_per_stage, smem_size_v_per_stage)
        self.kv_stage = (224 * 1024 - smem_size_q_o - 16384) // smem_size_kv_per_stage
        # self.s_stage is defined in __init__ (always 2)
        assert self.s_stage >= self.q_stage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, h, s_q, d)
        mK: cute.Tensor,  # (b, h, s_k, d)
        mV: cute.Tensor,  # (b, h, s_k, dv)
        mO: cute.Tensor,  # (b, h, s_q, dv)
        mLSE: Optional[cute.Tensor],
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        softmax_scale: Float32,
        mBlockIndex: cute.Tensor,  # (batch, heads, num_q_blocks, max_kv_blocks), int32
        mBlockSizes: cute.Tensor,  # (num_kv_blocks,), int32
        block_sparse_num: Int32,  # runtime scalar >= 1; odd counts use a masked phantom block
        mBlockNums: Optional[cute.Tensor],  # (batch, heads, num_q_blocks), int32 or None
        stream: cuda.CUstream,
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
        assert (mLSE is not None) == (self.num_splits > 1), "Split producers require LSE"
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        assert self.q_dtype == cutlass.Float8E4M3FN
        assert self.o_dtype == (Float32 if self.num_splits > 1 else cutlass.BFloat16)
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Kernel boundary contract is BHSD. Convert (B, H, S, D) to
        # the mainloop/TMA-friendly (S, D, H, B) view internally.
        Q_layout_transpose = [2, 3, 1, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        # (s_k, d, h_k, b_k)
        KV_layout_transpose = [2, 3, 1, 0]
        mK, mV = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose)) for t in (mK, mV)]
        O_layout_transpose = [2, 3, 1, 0]
        LSE_layout_transpose = [2, 1, 0]
        num_splits = Int32(self.num_splits)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)) if const_expr(mLSE is not None) else None
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        q_major_mode = OperandMajorMode.K
        k_major_mode = OperandMajorMode.K
        v_major_mode = OperandMajorMode.MN
        self.o_layout = LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage)
        sK_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage)
        tP_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage)
        sV_layout = sm100_utils_basic.make_smem_layout_b(tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage)
        sO_layout = sm100_utils_basic.make_smem_layout_epi(self.o_dtype, self.o_layout, self.epi_tile, self.s_stage)
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
        )

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
        )
        # TMA load for V
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
        )

        tma_atom_O, mO = cpasync.make_tiled_tma_atom(tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile)

        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            element_size=self.k_dtype.width // 8,
            is_persistent=True,
            is_split_kv=self.num_splits > 1,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args, scheduling_mode=SchedulingMode.CLC)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)

        clc_response_size = self.sched_stages * 4
        clc_mbar_size = self.sched_stages * 2

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_P_ready: cute.struct.MemRange[Int64, self.p_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.s_stage * 2]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_stage * self.m_block_size * 2]
            # Keep independent 32-byte stages aligned for vectorized shared loads.
            sKScale: cute.struct.Align[cute.struct.MemRange[Float32, self.s_stage * 2 * 8], 32]
            # CLC buffers (mbarriers + response storage)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.Align[cute.struct.MemRange[Int32, clc_response_size], 16]
            sO: cute.struct.Align[cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

            sOnes: cute.struct.Align[cute.struct.MemRange[Int32, 4096], 1024]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale)

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mQScale,
            mKScale,
            mVScale,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            mBlockIndex,
            mBlockSizes,
            block_sparse_num,
            mBlockNums,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # (s_k, d, h_k, b_k)
        mV: cute.Tensor,  # (d, s_k, h_k, b_k)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        mBlockIndex: cute.Tensor,
        mBlockSizes: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        # Alloc
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sOnesPtr = storage.sOnes.data_ptr()
        sOnes = storage.sOnes.get_tensor(cute.make_layout(4096))
        thread = cute.arch.thread_idx()[0]
        for i in cutlass.range_constexpr(8):
            sOnes[thread + i * self.threads_per_cta] = Int32(0x38383838)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        tmem_alloc_barrier = _SubsetNamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        softmax_correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids))
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". SM100 separates P-ready from correction-only O credits.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=(correction_threads if self.triple_buffer_p else softmax_correction_threads),
            defer_sync=True,
        )
        if const_expr(self.triple_buffer_p):
            pipeline_p_ready = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_P_ready.data_ptr(),
                num_stages=self.p_stage,
                producer_group=softmax_threads,
                consumer_group=mma_warp,
                defer_sync=True,
            )
        else:
            pipeline_p_ready = pipeline_custom.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_P_ready.data_ptr(),
                num_stages=self.s_stage,
                producer_group=softmax_warps,
                consumer_group=mma_warp,
                defer_sync=True,
            )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = _SubsetNamedBarrier(barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2)
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.s_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        pipeline_init_arrive(is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.s_stage * self.m_block_size * 2))
        sKScale = storage.sKScale.get_tensor(cute.make_layout(self.s_stage * 2 * 8))

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.s_stage))
        tOtO = cute.make_tensor(
            tOtO.iterator + self.tmem_o_offset[0],
            cute.make_layout(tOtO.shape, stride=(*tOtO.layout.stride[:-1], 144)),
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,))),
        )

        block_info = BlockInfo(
            # The scheduler advances in single-CTA output tiles.
            self.cta_tiler[0],
            self.cta_tiler[1],
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
        )
        # Create the tile scheduler and its CLC pipeline
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

        clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(cutlass_pipeline.Agent.Thread)
        clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(cutlass_pipeline.Agent.Thread, self.threads_per_cta)
        clc_pipeline = cutlass_pipeline.PipelineClcFetchAsync.create(
            barrier_storage=clc_mbar_ptr,
            num_stages=self.sched_stages,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=16,
        )

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc_response_ptr=clc_response_ptr)
        clc_consumer_state = cutlass_pipeline.make_pipeline_state(cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages)
        tile_scheduler.set_clc_pipeline(clc_pipeline, clc_consumer_state)

        pipeline_init_wait()

        # ///////////////////////////////////////////////////////////////////////////////
        #  CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.clc_scheduler_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.clc_scheduler_warp(clc_pipeline, tile_scheduler)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
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
                pipeline_q,
                pipeline_kv,
                block_info,
                tile_scheduler,
                mBlockIndex,
                block_sparse_num,
                mBlockNums,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem_alloc_barrier.arrive_and_wait()  # not tmem.wait_for_alloc(): that rebuilds an aligned barrier
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_ready,
                pipeline_o_acc,
                tile_scheduler,
                block_sparse_num,
                mBlockNums,
                block_info,
                mBlockIndex,
                mKScale,
                sKScale,
                sOnesPtr,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                tma_atom_O,
                pipeline_o_epi,
                tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx <= self.softmax1_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait()  # not tmem.wait_for_alloc(): that rebuilds an aligned barrier
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_ready=pipeline_p_ready,
                pipeline_o_acc=pipeline_o_acc,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                block_info=block_info,
                tile_scheduler=tile_scheduler,
                mBlockIndex=mBlockIndex,
                mBlockSizes=mBlockSizes,
                block_sparse_num=block_sparse_num,
                mBlockNums=mBlockNums,
                mQScale=mQScale,
                sKScale=sKScale,
                seqlen_q=mQ.shape[0],
                seqlen_k=mK.shape[0],
            )

            stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
            softmax_loop(stage=stage)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait()  # not tmem.wait_for_alloc(): that rebuilds an aligned barrier
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_pv,
                tOtO,
                sScale,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                Float32(1.0),
                SeqlenInfoCls,
                tile_scheduler,
                block_sparse_num,
                mBlockNums,
                mVScale,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_ready: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        tile_scheduler: "TileSchedulerProtocol",
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        block_info: "BlockInfo",
        mBlockIndex: cute.Tensor,
        mKScale: cute.Tensor,
        sKScale: cute.Tensor,
        sOnesPtr: cutlass.Pointer,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        q_smem_base = sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, sQ.element_type.width, sQ.layout[0]),
            sQ.iterator.type.swizzle_type,
            sm100_desc.Major.K,
        )
        k_smem_base = sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, sK.element_type.width, sK.layout[0]),
            sK.iterator.type.swizzle_type,
            sm100_desc.Major.K,
        )
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]

        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.q_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="bsa_fwd_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="bsa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="bsa_fwd_pv_mma_idesc")

        sQ_stage_stride = 0  # q_stage=1
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="bsa_fwd_q_smem_desc",
                idesc_var_name="bsa_fwd_qk_mma_idesc",
                # For q_stage=1, both stages use same Q smem (offset=0).
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
            )
            for stage in range(self.s_stage)
        ]
        if const_expr(self.triple_buffer_p):
            gemm_Pi = [
                partial(
                    self.gemm_fused_pv,
                    tmem_out=Int32(self.tmem_o_offset[stage]),
                    sOnesPtr=sOnesPtr,
                )
                for stage in range(self.s_stage)
            ]
        else:
            gemm_Pi = [
                partial(
                    self.gemm_fused_pv,
                    Int32(self.tmem_o_offset[stage]),
                    Int32(self.tmem_p_offset[stage]),
                    sOnesPtr,
                )
                for stage in range(self.s_stage)
            ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        p_read_counter = Int32(0)
        s_read_barrier = _SubsetNamedBarrier(barrier_id=12, num_threads=160)
        # Pipeline s_p_o phases for stage 0 and stage 1.
        # These phases must persist across tiles
        # so that mbarrier phase stays in sync when block_iter_count varies.
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            mBlockIndex_cur = mBlockIndex
            if const_expr(self.num_splits > 1):
                split_begin, raw_block_count = self._split_block_range(mBlockNums, block_sparse_num, batch_idx, head_idx, m_block, split_idx)
                process_tile = raw_block_count > Int32(0)
                block_iter_count = (raw_block_count + 1) & ~1
                mBlockIndex_cur = cute.make_tensor(mBlockIndex.iterator + split_begin, mBlockIndex.layout)
            elif const_expr(mBlockNums is not None):
                raw_block_count = mBlockNums[batch_idx, head_idx, m_block]
                process_tile = raw_block_count > Int32(0) if const_expr(self.allow_empty_block_nums) else True
                block_iter_count = (raw_block_count + 1) & ~1
            else:
                raw_block_count = block_sparse_num
                process_tile = True
                block_iter_count = (block_sparse_num + 1) & ~1

            if process_tile:
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex_cur,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(raw_block_count - 1, Int32(0)),
                )
                publish_scales = partial(
                    self.publish_k_scales,
                    mKScale,
                    sKScale,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                )
                # ================================================================
                # q_stage=1: intra-warp overlap across n_block direction
                # Pipeline KV order: K0, K1, V0, K2, V1, K3, V2, ...
                # One physical S tile; both logical streams retain independent P/O.
                # Control issues PV then QK; lookahead issues full QK then PV.
                # ================================================================

                # Prologue: wait Q0
                pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
                mma_q_consumer_phase ^= 1

                # S0 = Q @ K0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                publish_scales(key_block=n_block(block_iter_count - 1), stage=phase_s0 * 2)
                gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                pipeline_s_p_o.producer_commit_w_index(0)  # signal S0 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # S1 = Q @ K1
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                # The single S tile is reusable once all raw reads retire.
                s_read_barrier.arrive_and_wait_w_index(index=0)
                fp8_helpers.tcgen05_fence_after_thread_sync()
                publish_scales(key_block=n_block(block_iter_count - 2), stage=phase_s1 * 2 + 1)
                gemm_Si[1](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                pipeline_s_p_o.producer_commit_w_index(1)  # signal S1 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # Flat loop: N-2 iterations, alternating stage 0/1
                block_loop_count = block_iter_count - 2
                O_acc_s0 = False  # per-stage accumulate flags
                O_acc_s1 = False
                # Pre-declare loop variables for DSL type tracking
                Vi_index = mma_kv_consumer_state.index
                Ki_index = mma_kv_consumer_state.index
                mma_kv_release_state = mma_kv_consumer_state.clone()
                tOrVi = tOrV[None, None, None, Vi_index]
                sV_cur = sV[None, None, None, Vi_index]
                sK_cur = sK[None, None, None, Ki_index]

                pair_count = block_loop_count // 2  # N even => block_loop_count = N-2 is even
                for i in cutlass.range(pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        if const_expr(stage == 0):
                            phase_cur, O_acc_cur = phase_s0, O_acc_s0
                        else:
                            phase_cur, O_acc_cur = phase_s1, O_acc_s1
                        # Keep the next scales in registers across the P/V wait.
                        # Publish only after the previous softmax read finishes.
                        next_k_scale = self.load_k_scale(
                            mKScale,
                            n_block(block_iter_count - 3 - i * 2 - stage),
                            batch_idx,
                            head_idx,
                        )
                        # Wait V
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index = mma_kv_consumer_state.index
                        tOrVi = tOrV[None, None, None, Vi_index]
                        sV_cur = sV[None, None, None, Vi_index]
                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index = mma_kv_consumer_state.index
                        sK_cur = sK[None, None, None, Ki_index]
                        s_read_barrier.arrive_and_wait_w_index(index=0)
                        fp8_helpers.tcgen05_fence_after_thread_sync()
                        # Two generations per stream protect deferred scale LDS.
                        self.publish_prefetched_k_scale(sKScale, next_k_scale, (phase_cur ^ 1) * 2 + stage)
                        gemm_Si[stage](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, phase_cur)
                        if const_expr(self.triple_buffer_p):
                            p_slot = p_read_counter
                            p_phase = Int32(0)
                            if p_slot >= 3:
                                p_slot -= 3
                                p_phase = Int32(1)
                            pipeline_p_ready.consumer_wait_w_index_phase(p_slot, p_phase)
                            fp8_helpers.tcgen05_fence_after_thread_sync()
                            gemm_Pi[stage](
                                tmem_probability=Int32(416) + p_slot * 32,
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_acc_cur,
                                mbar_ptr=(pipeline_p_ready.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None),
                                mbar_phase=phase_cur,
                            )
                        else:
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_acc_cur,
                                mbar_ptr=(pipeline_p_ready.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None),
                                mbar_phase=phase_cur,
                            )
                        pipeline_o_acc.producer_commit_w_index(stage)
                        if const_expr(self.triple_buffer_p):
                            p_read_counter += 1
                            if p_read_counter >= 6:
                                p_read_counter -= 6
                        if const_expr(stage == 0):
                            phase_s0 ^= 1
                            O_acc_s0 = True
                        else:
                            phase_s1 ^= 1
                            O_acc_s1 = True
                        # Release V and K
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                # Retire the final raw-score read credit before the next tile.
                s_read_barrier.arrive_and_wait_w_index(index=0)
                fp8_helpers.tcgen05_fence_after_thread_sync()
                # release Q0
                pipeline_q.consumer_release_w_index(0)

                # Epilogue: 2 PV GEMMs (N even: stage 0 first, stage 1 second)
                for epi_stage_constexpr in cutlass.range_constexpr(2):
                    if const_expr(epi_stage_constexpr == 0):
                        epi_s, epi_p, epi_zi = 0, phase_s0, not O_acc_s0
                    else:
                        epi_s, epi_p, epi_zi = 1, phase_s1, not O_acc_s1
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index = mma_kv_consumer_state.index
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    pipeline_s_p_o.producer_acquire_w_index_phase(epi_s, epi_p)
                    if const_expr(self.triple_buffer_p):
                        p_slot = p_read_counter
                        p_phase = Int32(0)
                        if p_slot >= 3:
                            p_slot -= 3
                            p_phase = Int32(1)
                        pipeline_p_ready.consumer_wait_w_index_phase(p_slot, p_phase)
                        fp8_helpers.tcgen05_fence_after_thread_sync()
                        gemm_Pi[epi_stage_constexpr](
                            tmem_probability=Int32(416) + p_slot * 32,
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=epi_zi,
                            mbar_ptr=(pipeline_p_ready.sync_object_full.get_barrier(epi_s) if self.split_P_arrive > 0 else None),
                            mbar_phase=epi_p,
                        )
                    else:
                        gemm_Pi[epi_stage_constexpr](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=epi_zi,
                            mbar_ptr=(pipeline_p_ready.sync_object_full.get_barrier(epi_s) if self.split_P_arrive > 0 else None),
                            mbar_phase=epi_p,
                        )
                    pipeline_o_acc.producer_commit_w_index(epi_s)
                    if const_expr(self.triple_buffer_p):
                        p_read_counter += 1
                        if p_read_counter >= 6:
                            p_read_counter -= 6
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                # Epilogue did one acquire per stage; advance phases for next tile
                phase_s0 ^= 1
                phase_s1 ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()

    @cute.jit
    def gemm_fused_pv(
        self,
        tmem_out: Int32,
        tmem_probability: Int32,
        sOnesPtr: cutlass.Pointer,
        tCrB: cute.Tensor,
        sB: cute.Tensor,
        zero_init: bool | Boolean,
        mbar_ptr: Optional[cutlass.Pointer] = None,
        mbar_phase: Int32 = 0,
    ):
        assert self.split_P_arrive == 0
        _fused_pv_issue(
            tmem_out,
            Int32(sB.iterator.toint()),
            Int32(sOnesPtr.toint()),
            Int32(not zero_init),
            tmem_probability,
            144,
        )

    @cute.jit
    def publish_k_scales(
        self,
        mKScale: cute.Tensor,
        sKScale: cute.Tensor,
        key_block: Int32,
        stage: Int32,
        batch_idx: Int32,
        head_idx: Int32,
    ):
        value = self.load_k_scale(mKScale, key_block, batch_idx, head_idx)
        self.publish_prefetched_k_scale(sKScale, value, stage)

    @cute.jit
    def publish_prefetched_k_scale(
        self,
        sKScale: cute.Tensor,
        value: Float32,
        stage: Int32,
    ):
        lane = cute.arch.thread_idx()[0] % 32
        if lane < 8:
            sKScale[stage * 8 + lane] = value
        cute.arch.sync_warp()
        cute.arch.fence_view_async_shared()

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_ready: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        block_info: "BlockInfo",
        tile_scheduler: "TileSchedulerProtocol",
        mBlockIndex: cute.Tensor,
        mBlockSizes: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        mQScale: cute.Tensor,
        sKScale: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids)))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        p_write_counter = Int32(stage)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            # n_block(i): maps logical index i to actual KV block index via q2k_block_index
            if const_expr(self.num_splits > 1):
                split_begin, raw_block_count = self._split_block_range(mBlockNums, block_sparse_num, batch_idx, head_idx, m_block, split_idx)
                has_work = raw_block_count > Int32(0)
                mBlockIndex_split = cute.make_tensor(mBlockIndex.iterator + split_begin, mBlockIndex.layout)
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex_split,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(raw_block_count - 1, Int32(0)),
                )
            elif const_expr(mBlockNums is not None):
                raw_block_count = mBlockNums[batch_idx, head_idx, m_block]
                has_work = raw_block_count > Int32(0) if const_expr(self.allow_empty_block_nums) else True
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(raw_block_count - 1, Int32(0)),
                )
            else:
                raw_block_count = block_sparse_num
                has_work = True
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(block_sparse_num - 1, Int32(0)),
                )

            softmax = SoftmaxSm100.create(
                Float32(1.0),
                rescale_threshold=math.log2(448.0 / 128.0),
                softmax_scale=None,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_ready=pipeline_p_ready,
                pipeline_o_acc=pipeline_o_acc,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                sScale=sScale,
                stage=stage,
                q_log2_scale=mQScale[batch_idx, head_idx, cutlass.min(m_block * 128 + tidx, seqlen_q - 1)] * softmax_scale_log2,
                sKScale=sKScale,
                tmem_s_addr=Int32(warp_idx * 32 << 16),
                tmem_p_addr=self.tmem_p_offset[0] + stage * (self.tmem_p_offset[1] - self.tmem_p_offset[0]) + (warp_idx * 32 << 16),
            )

            # Always acquire pipeline_sm_stats to stay in sync with correction
            pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
            sm_stats_producer_phase ^= 1

            if has_work:
                # block_iter_count is even: each WG processes exactly half the blocks
                # WG0 (stage=0): logical indices N-1, N-3, ... (stride 2)
                # WG1 (stage=1): logical indices N-2, N-4, ... (stride 2)
                if const_expr(mBlockNums is not None or self.num_splits > 1):
                    block_iter_count = (raw_block_count + 1) & ~1
                else:
                    block_iter_count = (block_sparse_num + 1) & ~1
                wg_count = block_iter_count // 2
                # logical_first is the first logical index for this WG
                logical_first = block_iter_count - 1 - stage

                # 1st block — phantom block (logical_first >= raw_block_count) uses block_size=0
                n_block_first = n_block(logical_first)
                first_block_size = self.valid_block_size(mBlockSizes, batch_idx, head_idx, n_block_first, seqlen_k)
                if logical_first >= raw_block_count:
                    first_block_size = Int32(0)
                first_mask_fn = partial(apply_block_size_mask, block_size=first_block_size, n_block_size=128)
                mma_si_consumer_phase, sm_stats_producer_phase, p_write_counter = softmax_step(
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    p_write_counter,
                    valid_tokens=first_block_size,
                    mask_fn=first_mask_fn,
                    is_first=True,
                )
                # Remaining blocks with stride 2 — always valid (logical_n < raw_block_count)
                for n_tile in cutlass.range(wg_count - 1, unroll=1):
                    logical_n = logical_first - self.s_stage * (n_tile + 1)
                    n_block_cur = n_block(logical_n)
                    remaining_mask_fn = partial(
                        apply_block_size_mask,
                        block_size=self.valid_block_size(mBlockSizes, batch_idx, head_idx, n_block_cur, seqlen_k),
                        n_block_size=128,
                    )
                    mma_si_consumer_phase, sm_stats_producer_phase, p_write_counter = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        p_write_counter,
                        valid_tokens=self.valid_block_size(mBlockSizes, batch_idx, head_idx, n_block_cur, seqlen_k),
                        mask_fn=remaining_mask_fn,
                    )

                if const_expr(self.double_buffer_stats):
                    # Drain the last scale readers before reusing both slots for
                    # the final payload. Terminal and steady ready IDs are distinct.
                    _SubsetNamedBarrier(barrier_id=11, num_threads=256).arrive_and_wait_w_index(index=stage * 2)
                else:
                    # Final statistics have no next-QK ownership edge. Wait for
                    # the last scale reader before reusing its shared slot.
                    pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
                    sm_stats_producer_phase ^= 1
                sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                if const_expr(mLSE is not None or self.q_stage == 1):
                    sScale[tidx + stage * self.m_block_size + self.s_stage * self.m_block_size] = softmax.row_max[0]
                if const_expr(self.double_buffer_stats):
                    _SubsetNamedBarrier(barrier_id=11, num_threads=256).arrive_w_index(index=stage * 2)
                else:
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
            else:
                # Empty tiles publish only their terminal notification.
                if const_expr(self.double_buffer_stats):
                    _SubsetNamedBarrier(barrier_id=11, num_threads=256).arrive_w_index(index=stage * 2)
                else:
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        p_write_counter: Int32,
        softmax: "SoftmaxSm100",
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_ready: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        sScale: cute.Tensor,
        stage: int | Int32,
        q_log2_scale: Float32,
        sKScale: cute.Tensor,
        tmem_s_addr: Int32,
        tmem_p_addr: Int32,
        valid_tokens: Int32,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if const_expr(self.triple_buffer_p):
            p_slot = p_write_counter
            if p_slot >= 3:
                p_slot -= 3
            tmem_p_addr = Int32(416) + p_slot * 32 + (warp_idx * 32 << 16)
        # S_i completion covers the previous owner of P[i % 3].
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor((128,), Float32)
        qk_scales = cute.make_rmem_tensor((8,), Float32)
        group_maxima = cute.make_rmem_tensor((8,), Float32)
        for group in cutlass.range_constexpr(8):
            qk_scales[group] = q_log2_scale * sKScale[(mma_si_consumer_phase * 2 + stage) * 8 + group]
        if const_expr(self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f):
            for group in cutlass.range_constexpr(8):
                vals, group_max = tmem_load_red_max_32dp32b16x(tmem_s_addr + group * 16)
                group_maxima[group] = group_max
                for j in cutlass.range_constexpr(16):
                    tSrS_t2r[group * 16 + j] = vals[j]
        else:
            for group in cutlass.range_constexpr(4):
                vals = fp8_helpers.tmem_load_32dp32b32x(tmem_s_addr + group * 32)
                for j in cutlass.range_constexpr(32):
                    tSrS_t2r[group * 32 + j] = vals[j]
        # Retire every raw S load before another QK may overwrite its tile.
        cute.arch.fence_view_async_tmem_load()
        fp8_helpers.tcgen05_fence_before_thread_sync()
        _SubsetNamedBarrier(barrier_id=12, num_threads=160).arrive_w_index(index=0)
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r)
        tSrS_groups = cute.logical_divide(tSrS_t2r, cute.make_layout(16))
        if const_expr(self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f):
            if valid_tokens < 128:
                for group in cutlass.range_constexpr(8):
                    group_maxima[group] = _max16_tree(tSrS_groups[None, group].load())
        else:
            for group in cutlass.range_constexpr(8):
                group_maxima[group] = _max16_tree(tSrS_groups[None, group].load())
        for group in cutlass.range_constexpr(8):
            group_maxima[group] *= qk_scales[group]
        row_max, acc_scale = softmax.update_row_max(group_maxima.load(), is_first)

        if const_expr(self.double_buffer_stats):
            # Correction reconstructs alpha from consecutive selected maxima.
            # Keep the P bias in registers while publishing its shared payload.
            thread_idx = cute.arch.thread_idx()[0] % self.m_block_size
            scale_index = thread_idx + (mma_si_consumer_phase * 2 + stage) * self.m_block_size
            selected_max = _publish_row_max(
                Int32((sScale.iterator + scale_index).toint()),
                softmax.row_max[0],
                Int32(3 + stage * 2 + mma_si_consumer_phase),
            )
            row_max = selected_max if selected_max != -Float32.inf else Float32(0.0)
        else:
            if const_expr(not is_first):
                # Delay this wait until the next score load/reduction has finished.
                # Correction can consume the previous scale while that work runs.
                pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
                sm_stats_producer_phase ^= 1
                thread_idx = cute.arch.thread_idx()[0] % self.m_block_size
                sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # Publish scale ownership for every block, including the first.
            sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        # Fold input dequantization, attention scale and P*128 into one FMA.
        # Store rounded P in 32-token chunks. The widened PV accumulates
        # the same rounded probabilities into its appended denominator.
        bias = Float32(7.0) - row_max
        for i in cutlass.range_constexpr(4):
            p_frag = cute.make_rmem_tensor((8,), Int32)
            for j in cutlass.range_constexpr(0, 32, 4):
                offset = i * 32 + j
                score_scale = qk_scales[offset // 16]
                x0, x1 = cute.arch.fma_packed_f32x2(
                    (tSrS_t2r[offset], tSrS_t2r[offset + 1]),
                    (score_scale, score_scale),
                    (bias, bias),
                )
                x2, x3 = cute.arch.fma_packed_f32x2(
                    (tSrS_t2r[offset + 2], tSrS_t2r[offset + 3]),
                    (score_scale, score_scale),
                    (bias, bias),
                )
                e0, e1 = cute.math.exp2(x0, fastmath=True), cute.math.exp2(x1, fastmath=True)
                if const_expr(self.enable_ex2_emu and i < 3 and j % 12 < 8):
                    e2, e3 = utils.ex2_emulation_2(x2, x3)
                else:
                    e2, e3 = cute.math.exp2(x2, fastmath=True), cute.math.exp2(x3, fastmath=True)
                p_frag[j // 4] = fp8_helpers.cvt_f32x4_to_e4m3x4(e0, e1, e2, e3)
            # The third P slot is free at S readiness on SM100. SM103
            # retains its previous-PV ownership wait before reusing P.
            if const_expr(not self.triple_buffer_p and not is_first and i == 0):
                pipeline_o_acc.consumer_wait_w_index_phase(stage, mma_si_consumer_phase ^ 1)
            fp8_helpers.tmem_store_e4m3x8(tmem_p_addr + i * 8, p_frag)
            if const_expr(i + 1 == self.split_P_arrive // 32):
                cute.arch.fence_view_async_tmem_store()
                fp8_helpers.tcgen05_fence_before_thread_sync()
                pipeline_s_p_o.consumer_release_w_index(stage)
        cute.arch.fence_view_async_tmem_store()
        fp8_helpers.tcgen05_fence_before_thread_sync()
        if const_expr(self.triple_buffer_p):
            pipeline_p_ready.producer_commit_w_index(p_slot)
            p_write_counter += 2
            if p_write_counter >= 6:
                p_write_counter -= 6
        else:
            if const_expr(self.split_P_arrive > 0):
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_p_ready.producer_commit_w_index(stage)
            else:
                # The single P credit now orders every probability store before PV.
                pipeline_s_p_o.consumer_release_w_index(stage)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase, p_write_counter

    @cute.jit
    def correction_loop(
        self,
        thr_mma_pv: cute.ThrMma,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        softmax_scale_log2: Float32,
        SeqlenInfoCls: Callable,
        tile_scheduler: "TileSchedulerProtocol",
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        mVScale: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        # SM100 protects the first selected-max reader with its initial O credit.
        # SM103 retains the scale-payload protocol and initial credits.
        if const_expr(not self.double_buffer_stats):
            for stage in cutlass.range(self.s_stage):
                pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls()

            # For q_stage=1, always need row_max for combine; use -inf as default
            stats = [
                (
                    Float32(0.0),
                    -Float32.inf if const_expr(mLSE is not None or self.q_stage == 1) else None,
                    True,
                )
            ] * self.s_stage

            if const_expr(self.num_splits > 1):
                _, split_block_count = self._split_block_range(mBlockNums, block_sparse_num, batch_idx, head_idx, m_block, split_idx)
                has_work = split_block_count > Int32(0)
            elif const_expr(mBlockNums is not None):
                has_work = mBlockNums[batch_idx, head_idx, m_block] > Int32(0) if const_expr(self.allow_empty_block_nums) else True
            else:
                has_work = True

            if has_work:
                if const_expr(not self.double_buffer_stats):
                    # The first dummy credit starts explicit scale ownership.
                    sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                    pipeline_sm_stats.consumer_release_w_index(0)
                    sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                    pipeline_sm_stats.consumer_release_w_index(1)
                    sm_stats_consumer_phase ^= 1

                if const_expr(self.double_buffer_stats):
                    previous_row_max = cute.make_rmem_tensor((2,), Float32)
                    for stage in cutlass.range_constexpr(self.s_stage):
                        _SubsetNamedBarrier(barrier_id=3, num_threads=256).arrive_and_wait_w_index(index=stage * 2 + o_corr_consumer_phase)
                        previous_row_max[stage] = sScale[tidx + (o_corr_consumer_phase * 2 + stage) * self.m_block_size]
                        # The first PV cannot retire until this first reader is done.
                        pipeline_s_p_o.consumer_release_w_index(stage)
                # q_stage=1 correction loop
                if const_expr(self.num_splits > 1):
                    block_iter_count = (split_block_count + 1) & ~1
                elif const_expr(mBlockNums is not None):
                    block_iter_count = (mBlockNums[batch_idx, head_idx, m_block] + 1) & ~1
                else:
                    block_iter_count = (block_sparse_num + 1) & ~1
                corr_pair_count = (block_iter_count - 2) // 2
                # Rescale the two alternating stages as a pair.
                for i in cutlass.range(corr_pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        if const_expr(self.double_buffer_stats):
                            scale_slot = o_corr_consumer_phase ^ 1
                            _SubsetNamedBarrier(barrier_id=3, num_threads=256).arrive_and_wait_w_index(index=stage * 2 + scale_slot)
                            selected_max = sScale[tidx + (scale_slot * 2 + stage) * self.m_block_size]
                            safe_max = selected_max if selected_max != -Float32.inf else Float32(0.0)
                            scale = cute.math.exp2(previous_row_max[stage] - safe_max, fastmath=True)
                            previous_row_max[stage] = selected_max
                            # Correction retains O credits and the previous-PV
                            # wait, bounding both scale and arrival generations.
                        else:
                            sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                            scale = sScale[tidx + stage * self.m_block_size]
                            # Lookahead QK bypasses O ownership. Release the scale
                            # slot after every reader before the next writer.
                            pipeline_sm_stats.consumer_release_w_index(stage)
                        # An early S publication no longer proves previous PV completion.
                        pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        if should_rescale:
                            self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
                            _fused_denominator_rescale(
                                Int32(self.tmem_o_offset[stage] + 128 + ((warp_idx * 32) << 16)),
                                scale,
                            )
                        pipeline_s_p_o.consumer_release_w_index(stage)
                    sm_stats_consumer_phase ^= 1
                    o_corr_consumer_phase ^= 1
                # Read final softmax stats for both stages
                for stage in cutlass.range_constexpr(self.s_stage):
                    if const_expr(self.double_buffer_stats):
                        # Drain scale readers, then wait for the final payload.
                        _SubsetNamedBarrier(barrier_id=11, num_threads=256).arrive_and_wait_w_index(index=stage * 2)
                        _SubsetNamedBarrier(barrier_id=11, num_threads=256).arrive_and_wait_w_index(index=stage * 2)
                    else:
                        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    # The final PV owns both O and its appended denominator.
                    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    fp8_helpers.tcgen05_fence_after_thread_sync()
                    row_sum = _fused_denominator_load(Int32(self.tmem_o_offset[stage] + 128 + ((warp_idx * 32) << 16)))
                    if const_expr(mLSE is not None or self.q_stage == 1):
                        row_max = sScale[tidx + stage * self.m_block_size + self.s_stage * self.m_block_size]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)

                # q_stage=1: combine O0 and O1, then write single output
                row_sum0, row_max0, zero_or_nan0 = stats[0]
                row_sum1, row_max1, zero_or_nan1 = stats[1]

                # Compute combined scales for the two partial O accumulators
                # FP8 row maxima already include dequantization and log2 scale;
                # softmax_scale_log2 is 1 here when combining the two streams.
                # For empty/padding stages (zero_or_nan=True), row_max may be 0.0 instead of -inf
                # due to softmax's safe_max clamping. Use -inf for those to avoid polluting max_combined.
                rm0 = row_max0 if not zero_or_nan0 else -Float32.inf
                rm1 = row_max1 if not zero_or_nan1 else -Float32.inf
                max_combined = cutlass.max(rm0, rm1)
                max_safe = max_combined if max_combined != -Float32.inf else Float32(0.0)
                scale0 = cute.math.exp2((rm0 - max_safe) * softmax_scale_log2, fastmath=True) if not zero_or_nan0 else Float32(0.0)
                scale1 = cute.math.exp2((rm1 - max_safe) * softmax_scale_log2, fastmath=True) if not zero_or_nan1 else Float32(0.0)
                sum_combined = row_sum0 * scale0 + row_sum1 * scale1
                combined_zero_or_nan = sum_combined == 0.0 or sum_combined != sum_combined
                inv_sum = cute.arch.rcp_approx(sum_combined if not combined_zero_or_nan else 1.0)
                final_scale0 = scale0 * inv_sum
                final_scale1 = scale1 * inv_sum

                # Both O completions were consumed before their denominator reads.
                pipeline_o_epi.producer_acquire_w_index_phase(0, corr_epi_producer_phase)
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    final_scale0,
                    final_scale1,
                    sO[None, None, 0],
                    mVScale,
                    head_idx,
                )
                # The next SM100 tile releases O after reading its first maxima.
                if const_expr(not self.double_buffer_stats):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        pipeline_s_p_o.consumer_release_w_index(stage)
                pipeline_o_epi.producer_commit_w_index(0)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # Empty tile (block_count == 0): sync pipelines and write O=0.
                # Match softmax's 1 barrier arrive per stage.
                for stage_idx in cutlass.range_constexpr(self.s_stage):
                    if const_expr(self.double_buffer_stats):
                        _SubsetNamedBarrier(barrier_id=11, num_threads=256).arrive_and_wait_w_index(index=stage_idx * 2)
                    else:
                        sm_stats_barrier.arrive_and_wait_w_index(index=stage_idx * 4 + warp_idx)
                    pipeline_sm_stats.consumer_release_w_index(stage_idx)
                sm_stats_consumer_phase ^= 1
                # Write O=0 via correction_epilogue_combine with scale=0.
                # Both scales are zero, so the epilogue skips stale TMEM reads.
                pipeline_o_epi.producer_acquire_w_index_phase(0, corr_epi_producer_phase)
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    Float32(0.0),
                    Float32(0.0),
                    sO[None, None, 0],
                    mVScale,
                    head_idx,
                )
                # Do NOT release pipeline_s_p_o (MMA didn't commit)
                pipeline_o_epi.producer_commit_w_index(0)
                # o_corr_consumer_phase NOT toggled (pipeline_o_acc not touched)
                corr_epi_producer_phase ^= 1

            if const_expr(mLSE is not None):
                output_head = head_idx + split_idx * (mLSE.shape[1] // self.num_splits)
                mLSE_cur = mLSE[None, output_head, batch_idx]
                # q_stage=1: compute combined LSE from two partial softmax stats
                m_tile_idx = m_block
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                row_sum0, row_max0, zero_or_nan0 = stats[0]
                row_sum1, row_max1, zero_or_nan1 = stats[1]
                rm0_lse = row_max0 if not zero_or_nan0 else -Float32.inf
                rm1_lse = row_max1 if not zero_or_nan1 else -Float32.inf
                max_combined = cutlass.max(rm0_lse, rm1_lse)
                max_safe = max_combined if max_combined != -Float32.inf else Float32(0.0)
                s0 = cute.math.exp2((rm0_lse - max_safe) * softmax_scale_log2, fastmath=True) if not zero_or_nan0 else Float32(0.0)
                s1 = cute.math.exp2((rm1_lse - max_safe) * softmax_scale_log2, fastmath=True) if not zero_or_nan1 else Float32(0.0)
                sum_comb = row_sum0 * s0 + row_sum1 * s1
                comb_zero_or_nan = sum_comb == 0.0 or sum_comb != sum_comb
                LN2 = math.log(2.0)
                lse = (max_safe * softmax_scale_log2 + cute.math.log2(sum_comb, fastmath=True) - Float32(7.0)) * LN2 if not comb_zero_or_nan else -Float32.inf
                seqlen_q = seqlen.seqlen_q
                if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                    gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)


@cute.jit
def _fused_pv_issue(tmem: Int32, v_address: Int32, ones_address: Int32, accumulate: Int32, probability: Int32, columns: cutlass.Constexpr[int]):
    # Original SW128 V tile, plus a separate immutable SW128 ones tile.
    # The relative leading offset is different for each current V stage.
    descriptor: cutlass.Constexpr = (64 << 32) | (1 << 46) | (2 << 61)
    instruction: cutlass.Constexpr = sm100_desc.make_instr_desc(
        cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, Float32, 128, columns, sm100_desc.Major.K, sm100_desc.Major.MN
    )
    llvm.inline_asm(
        None,
        [tmem.ir_value(), v_address.ir_value(), ones_address.ir_value(), accumulate.ir_value(), probability.ir_value()],
        "{ .reg .pred elected, use_previous; .reg .b32 p, start, low, high, ins, leading; "
        ".reg .b64 bdesc; elect.sync _|elected,-1; "
        "setp.ne.b32 use_previous,$3,0; mov.b32 p,$4; shr.u32 start,$1,4; "
        "sub.u32 leading,$2,$1; shr.u32 leading,leading,4; shl.b32 leading,leading,16; "
        "or.b32 start,start,leading; "
        f"mov.b32 high,{descriptor >> 32}; mov.b32 ins,{instruction}; "
        "mov.b64 bdesc,{start,high}; "
        "@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [$0],[p],bdesc,ins,use_previous; "
        + "".join(
            f"add.u32 low,start,{k * 256}; mov.b64 bdesc,{{low,high}}; " f"@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [$0],[p+{k * 8}],bdesc,ins,1; "
            for k in range(1, 4)
        )
        + "}",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _fused_denominator_load(address: Int32) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [address.ir_value()],
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0},[$1]; tcgen05.wait::ld.sync.aligned;",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _fused_denominator_rescale(address: Int32, scale: Float32):
    llvm.inline_asm(
        None,
        [address.ir_value(), scale.ir_value()],
        "{ .reg .f32 sum; tcgen05.ld.sync.aligned.32x32b.x1.b32 {sum},[$0]; "
        "tcgen05.wait::ld.sync.aligned; mul.rn.f32 sum,sum,$1; "
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [$0],{sum}; "
        "tcgen05.wait::st.sync.aligned; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    fp8_helpers.tcgen05_fence_before_thread_sync()


@cute.jit
def _publish_row_max(address: Int32, value: Float32, barrier_id: Int32) -> Float32:
    """Publish selected maxima while preserving the register value for P."""
    llvm.inline_asm(
        None,
        [Float32(value).ir_value(), Int32(address).ir_value(), Int32(cute.arch.make_warp_uniform(barrier_id)).ir_value()],
        "st.shared.f32 [$1], $0; barrier.arrive $2, 256;",
        "f,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return value
