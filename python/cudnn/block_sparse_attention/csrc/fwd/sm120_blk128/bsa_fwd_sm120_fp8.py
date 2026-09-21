# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native 128x128 SM120 Sage FP8 attention, derived from the blk64 kernel.

Each CTA consumes original blk128 metadata and loads complete KV128 tiles.
FP8 fragment/softmax helpers preserve the existing per-row Q, per-16-token K,
and per-channel V scaling contract; no blk64 attention kernel is launched.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from cudnn.block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8 import (
    _finalize_softmax_fp8,
    _gemm_rs_fp8,
    _gemm_smem_zero_acc_fp8,
    _load_sage_k_scales_fp8,
    _mask_fp8,
    _rescale_o_with_sage_v_scale_fp8,
)
from cudnn.block_sparse_attention.csrc.utils import layout_utils
from cudnn.block_sparse_attention.csrc.utils.batched_static_scheduler import (
    BatchedStaticSchedulerMixin,
)

SM120_FWD_BLOCK_SIZE = 128


@dsl_user_op
def _compute_barrier(barrier_id: int, thread_count: int, wait: bool = True, *, loc=None, ip=None):
    """Synchronize participating compute warps without a CTA-wide aligned promise."""
    operation = "sync" if wait else "arrive"
    llvm.inline_asm(
        None,
        [],
        f"barrier.cta.{operation} {barrier_id}, {thread_count};",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _make_acc_into_fp8_smem(
    tSrS: cute.Tensor,
    sP: cute.Tensor,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
) -> cute.Tensor:
    """Convert C to FP8 A through a warp-owned MN-major shared-memory tile."""
    tidx, _, _ = cute.arch.thread_idx()
    tPrP_cvt = cute.make_rmem_tensor_like(tSrS, sP.element_type)
    tPrP_cvt.store(tSrS.load().to(sP.element_type))
    # N16 QK permutation matches x2 STSM; x4 would cross its copy tile.
    tiled_copy_p_r2s = cute.make_tiled_copy_C(
        cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix16x8x8bOp(transpose=True, num_matrices=2),
            sP.element_type,
        ),
        tiled_mma_qk,
    )
    tPrP_st = tiled_copy_p_r2s.retile(tPrP_cvt)
    tPsP = tiled_copy_p_r2s.get_slice(tidx).partition_D(sP)
    cute.copy(tiled_copy_p_r2s, tPrP_st, tPsP)
    # Each warp writes and rereads its own 16 rows; no CTA barrier is needed.
    cute.arch.sync_warp()
    tPsP_a = tiled_mma_pv.get_slice(tidx).partition_A(sP)
    tOrP = tiled_mma_pv.make_fragment_A(tPsP_a)
    atom_copy_p_s2r = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix16x16x8bOp(transpose=True, num_matrices=2),
        sP.element_type,
    )
    tiled_copy_p_s2r = cute.make_tiled_copy_A(atom_copy_p_s2r, tiled_mma_pv)
    thr_copy_P = tiled_copy_p_s2r.get_slice(tidx)
    cute.copy(tiled_copy_p_s2r, thr_copy_P.partition_S(sP), thr_copy_P.retile(tOrP))
    return tOrP


@cute.jit
def _rescale_o_if_needed_fp8(tOrO: cute.Tensor, row_scale: cute.Tensor):
    """Skip identity rescaling with a uniform branch across the whole warp."""
    tOrO_mn = layout_utils.reshape_acc_to_mn(tOrO)
    needs_rescale = cutlass.Boolean(False)
    for m in cutlass.range_constexpr(cute.size(row_scale)):
        needs_rescale = needs_rescale | (row_scale[m] != 1.0)
    # A per-lane predicate can leave all multiply instructions on the issue
    # path. A warp vote lets every lane bypass the complete rescale block.
    if cute.arch.vote_any_sync(needs_rescale):
        for m in cutlass.range_constexpr(cute.size(row_scale)):
            tOrO_mn[m, None].store(tOrO_mn[m, None].load() * row_scale[m])


@cute.jit
def _online_softmax_ordered_fp8(
    tiled_mma_qk: cute.TiledMma,
    tSrS: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    softmax_scale_log2e_m: cute.Tensor,
    exp_scale_log2: cutlass.Float32,
    tKScale: cute.Tensor,
) -> cute.Tensor:
    """Interleave exponentials with the original ordered FP32 row sum."""
    tSrS_mn = layout_utils.reshape_acc_to_mn(tSrS)
    row_scale = cute.make_rmem_tensor_like(row_max, cutlass.Float32)
    num_k_scales = cute.size(tKScale)
    scores_per_k_scale = cute.size(tSrS_mn, mode=[1]) // num_k_scales

    for m in cutlass.range(cute.size(row_max), unroll_full=True):
        softmax_scale_log2e = softmax_scale_log2e_m[m]
        row_max_local = row_max[m]
        # K descales are positive and shared by each score group, so reduce
        # unscaled scores first and apply one descale to the group maximum.
        for k_scale_group in cutlass.range_constexpr(num_k_scales):
            k_scale = tKScale[k_scale_group]
            group_begin = k_scale_group * scores_per_k_scale
            group_max = tSrS_mn[m, group_begin]
            for n in cutlass.range_constexpr(
                group_begin + 1,
                (k_scale_group + 1) * scores_per_k_scale,
            ):
                group_max = cute.arch.fmax(group_max, tSrS_mn[m, n])
            row_max_local = cute.arch.fmax(
                row_max_local,
                group_max * k_scale,
            )
        row_max_cur = cute.arch.warp_reduction_max(
            row_max_local,
            threads_in_group=4,
        )

        row_max_prev = row_max[m]
        row_max[m] = row_max_cur
        row_max_safe = cutlass.Float32(0.0) if row_max_cur == -cutlass.Float32.inf else row_max_cur
        # Shift the exponent by log2(P scale) before FP8 conversion. Keeping
        # row_sum in the same scale removes a vector multiply from every tile.
        row_max_scaled = row_max_safe * softmax_scale_log2e - exp_scale_log2

        row_scale[m] = cute.math.exp2(
            (row_max_prev - row_max_safe) * softmax_scale_log2e,
            fastmath=True,
        )
        # Keep the previous sum first, then add columns in the same order as
        # the blk64 helper. No tree reduction or changed rounding is used.
        acc_sum = row_sum[m] * row_scale[m]
        for k_scale_group in cutlass.range_constexpr(num_k_scales):
            score_scale_log2e = tKScale[k_scale_group] * softmax_scale_log2e
            for n in cutlass.range_constexpr(
                k_scale_group * scores_per_k_scale,
                (k_scale_group + 1) * scores_per_k_scale,
            ):
                tSrS_mn[m, n] = cute.math.exp2(
                    tSrS_mn[m, n] * score_scale_log2e - row_max_scaled,
                    fastmath=True,
                )
                acc_sum += tSrS_mn[m, n]
        row_sum[m] = acc_sum

    return row_scale


# =============================================================================
# Public kernel class
# =============================================================================
class BlockSparseAttnForwardFp8Sm120Blk128(BatchedStaticSchedulerMixin):
    supports_blocked_v = True

    def __init__(
        self,
        gqa_ratio: int = 1,
        head_dim: int = 128,
        value_dim: int = 128,
        blocksparse_blocksize_q: int = 128,
        blocksparse_blocksize_k: int = 128,
        dtype: type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        has_block_sizes: bool = True,
        has_block_nums: bool = True,
        block_sizes_mode: int = 0,
        v_block_size: int = 0,
    ):
        assert v_block_size in (0, 128)
        self.v_block_size = v_block_size
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        assert self.dtype is cutlass.Float8E4M3FN, "SM120 FP8 blk128 fwd requires fp8_e4m3fn"
        assert self.acc_dtype is cutlass.Float32
        self.softmax_p_scale_log2 = 8.0

        self.tile_size = SM120_FWD_BLOCK_SIZE

        assert blocksparse_blocksize_q == 128, "Only block_size_m=128 is supported in this kernel."
        assert blocksparse_blocksize_k == 128, "Only block_size_n=128 is supported in this kernel."
        # Eight compute warps retain the full Q128/KV128 tile. One extra
        # warpgroup donates registers; its first warp drives the TMA producer.
        self.num_threads = 384
        self.num_mma_warps = 8
        self.num_compute_threads = 256
        self.load_registers = 24
        self.compute_registers = 240
        self.kv_stage = 1
        self.q_stage = 1
        self.q_in_regs = True

        assert gqa_ratio >= 1
        assert head_dim == 128, "SM120 blk128 fwd currently requires QK dim 128"
        assert value_dim == 128, "SM120 blk128 fwd currently requires value dim 128"
        self.gqa_ratio = gqa_ratio
        self.qk_dim = head_dim
        self.value_dim = value_dim
        self.tile_shape_qk = (self.tile_size, self.tile_size, self.qk_dim)
        self.tile_shape_pv = (self.tile_size, self.value_dim, self.tile_size)

        self.scheduler = None

        self.use_tma_o = True
        self.has_block_sizes = has_block_sizes
        self.has_block_nums = has_block_nums
        self.block_sizes_mode = block_sizes_mode

    def _check_dim(self, tensor: cute.Tensor | list[cute.Tensor], mode: int):
        if isinstance(tensor, list):
            for t in tensor:
                self._check_dim(t, mode)
            return
        assert tensor.shape[mode] == 128, f"dim must be 128 in mode {mode}."
        assert tensor.stride[mode] == 1, f"dim must be contiguous in mode {mode}."

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (seqlen_q, head_dim, nheads, batch)
        mK: cute.Tensor,  # (seqlen_k, head_dim, nheads, batch)
        mV: cute.Tensor,  # (D, S, H, B), or blocked (D, token128, KV-block, H, B)
        mO: cute.Tensor,  # (seqlen_q, value_dim, nheads, batch)
        mLSE: cute.Tensor,  # (seqlen, nheads, batch)
        mQScale: cute.Tensor,  # (seqlen_q, nheads_q, batch)
        mKScale: cute.Tensor,  # (ceil_div(seqlen_k, 16), nheads_kv, batch)
        mVScale: cute.Tensor,  # (value_dim, nheads_kv)
        blocksparse_indices_q2k: cute.Tensor,  # (k, q, nheads, batch)
        blocksparse_num_blocks_q2k: cute.Tensor,  # (q, nheads, batch)
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Restore compile-time head dimensions while keeping runtime tensor
        # modes dynamic for reusable JIT callables.
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
        if cutlass.const_expr(self.v_block_size == 128):
            mV = cute.make_tensor(
                mV.iterator,
                cute.make_layout(
                    (self.value_dim, 128, mV.shape[2], mV.shape[3], mV.shape[4]),
                    stride=mV.stride,
                ),
            )
        else:
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
        mVScale = cute.make_tensor(
            mVScale.iterator,
            cute.make_layout(
                (self.value_dim, mVScale.shape[1]),
                stride=mVScale.stride,
            ),
        )
        self._check_dim([mQ, mK, mO], 1)
        if cutlass.const_expr(self.v_block_size == 128):
            assert mV.shape[0] == 128
            self._check_dim(mV, 1)
        else:
            self._check_dim(mV, 0)

        Q_layout = utils.LayoutEnum.from_tensor(mQ)
        K_layout = utils.LayoutEnum.from_tensor(mK)
        V_layout = utils.LayoutEnum.from_tensor(mV)
        O_layout = utils.LayoutEnum.from_tensor(mO)

        self.Q_dtype = mQ.element_type
        self.K_dtype = mK.element_type
        self.V_dtype = mV.element_type
        self.O_dtype = mO.element_type
        assert self.Q_dtype == self.K_dtype == self.V_dtype == cutlass.Float8E4M3FN
        assert self.O_dtype is cutlass.BFloat16
        self.Q_layout = Q_layout
        self.K_layout = K_layout
        self.V_layout = V_layout
        self.O_layout = O_layout

        atom_layout_mnk = (self.num_mma_warps, 1, 1)
        mma_inst_mnk = (16, 8, 32)
        permutation_mnk = (
            atom_layout_mnk[0] * mma_inst_mnk[0],
            atom_layout_mnk[1] * mma_inst_mnk[1] * 2,
            atom_layout_mnk[2] * mma_inst_mnk[2],
        )
        mma_op_qk = cute.nvgpu.warp.MmaFP8Op(
            self.Q_dtype,
            self.acc_dtype,
            mma_inst_mnk,
        )
        mma_op_pv = cute.nvgpu.warp.MmaFP8Op(
            self.V_dtype,
            self.acc_dtype,
            mma_inst_mnk,
        )
        tiled_mma_qk = cute.make_tiled_mma(
            mma_op_qk,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )
        tiled_mma_pv = cute.make_tiled_mma(
            mma_op_pv,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        self.Q_smem_layout = sm90_utils.make_smem_layout_a(
            Q_layout,
            self.tile_shape_qk,
            self.Q_dtype,
            self.q_stage,
        )
        self.K_smem_layout = sm90_utils.make_smem_layout_b(
            K_layout,
            self.tile_shape_qk,
            self.K_dtype,
            self.kv_stage,
        )
        if cutlass.const_expr(self.v_block_size == 128):
            self.V_smem_layout = sm90_utils.make_smem_layout_b(V_layout, self.tile_shape_pv, self.V_dtype, self.kv_stage)
        else:
            # Match the native transposed V loads to a 64-byte swizzle.
            v_smem_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64, self.V_dtype)
            self.V_smem_layout = cute.tile_to_shape(v_smem_atom, (*self.tile_shape_pv[1:], self.kv_stage), order=(0, 1, 2))
        vt_smem_layout_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64,
            self.V_dtype,
        )
        self.Vt_smem_layout = cute.tile_to_shape(
            vt_smem_layout_atom,
            (*self.tile_shape_pv[1:], self.kv_stage),
            order=(0, 1, 2),
        )
        O_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.O_dtype,
            O_layout,
            self.tile_shape_pv[:2],
            1,
        )
        self.O_smem_layout = cute.select(O_smem_layout_staged, mode=[0, 1])

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            K_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            V_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]

            Q_smem: cute.struct.Align[cute.struct.MemRange[self.Q_dtype, cute.cosize(self.Q_smem_layout)], 128]
            K_smem: cute.struct.Align[cute.struct.MemRange[self.K_dtype, cute.cosize(self.K_smem_layout)], 128]
            V_smem: cute.struct.Align[cute.struct.MemRange[self.V_dtype, cute.cosize(self.V_smem_layout)], 128]
            O_smem: cute.struct.Align[cute.struct.MemRange[self.O_dtype, cute.cosize(self.O_smem_layout)], 128]

        self.shared_storage_t = SharedStorage

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
            self.V_smem_layout,
            (self.tile_shape_pv[1], self.tile_shape_pv[2]),
            num_multicast=1,
        )

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mO,
            self.O_smem_layout,
            (self.tile_shape_pv[0], self.tile_shape_pv[1]),
            num_multicast=1,
        )

        log2_e = 1.44269504088896340736

        grid_config = self.get_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
        block_config = (self.num_threads, 1, 1)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mLSE,
            mQScale,
            mKScale,
            mVScale,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            blocksparse_indices_q2k,
            blocksparse_num_blocks_q2k,
            block_sparse_num,
            blocksparse_varblk,
            tiled_mma_qk,
            tiled_mma_pv,
            self.Q_smem_layout,
            self.K_smem_layout,
            self.V_smem_layout,
            self.Vt_smem_layout,
            self.O_smem_layout,
            softmax_scale * log2_e,
        ).launch(
            grid=grid_config,
            block=block_config,
            cluster=(1, 1, 1),
            smem=self.shared_storage_t.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_num_blocks_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        Vt_smem_layout: cute.ComposedLayout,
        O_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        work_desc = self.get_work_desc()
        seqlen = mK.shape[0]
        num_compute_tiles = cute.ceil_div(seqlen, self.tile_size)

        shared_storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0 and lane_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)

            if cutlass.const_expr(self.use_tma_o):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # TMA load barriers. K/V each use one full 128-token stage.
        Q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=cute.size_in_bytes(self.Q_dtype, cute.select(Q_smem_layout, mode=[0, 1])),
            barrier_storage=shared_storage.Q_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        Q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        K_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=cute.size_in_bytes(self.K_dtype, cute.select(K_smem_layout, mode=[0, 1])),
            barrier_storage=shared_storage.K_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        V_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=cute.size_in_bytes(self.V_dtype, cute.select(V_smem_layout, mode=[0, 1])),
            barrier_storage=shared_storage.V_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        K_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        V_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        V_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)

        # partition tensors
        sQ = shared_storage.Q_smem.get_tensor(
            Q_smem_layout.outer,
            swizzle=Q_smem_layout.inner,
        )
        sK = shared_storage.K_smem.get_tensor(K_smem_layout.outer, swizzle=K_smem_layout.inner)
        sV = shared_storage.V_smem.get_tensor(V_smem_layout.outer, swizzle=V_smem_layout.inner)
        p_smem_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64, self.Q_dtype)
        p_smem_layout = cute.tile_to_shape(p_smem_atom, (128, 128, 1), order=(0, 1, 2))
        sP = cute.make_tensor(cute.recast_ptr(shared_storage.Q_smem.data_ptr(), p_smem_layout.inner, dtype=self.Q_dtype), p_smem_layout.outer)
        if cutlass.const_expr(self.v_block_size == 128):
            sVt = sV
        else:
            # This K-major view only defines the FP8 B-fragment layout. The direct
            # transpose load below sources data from the N-major sV.
            sO_buffer = shared_storage.O_smem.get_tensor(
                O_smem_layout.outer,
                swizzle=O_smem_layout.inner,
            )
            sVt = cute.make_tensor(
                cute.recast_ptr(
                    sO_buffer.iterator,
                    Vt_smem_layout.inner,
                    dtype=self.V_dtype,
                ),
                Vt_smem_layout.outer,
            )

        mO_slice = mO[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mLSE_slice = mLSE[None, work_desc.qo_head_idx, work_desc.batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mK_slice = mK[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        if cutlass.const_expr(self.v_block_size == 128):
            mV_slice = mV[None, None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        else:
            mV_slice = mV[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        gQScale = mQScale[None, work_desc.qo_head_idx, work_desc.batch_idx]
        gKScale = mKScale[None, work_desc.kv_head_idx, work_desc.batch_idx]
        gVScale = mVScale[None, work_desc.kv_head_idx]
        gO = cute.local_tile(mO_slice, (self.tile_shape_pv[0], self.tile_shape_pv[1]), coord=(work_desc.qo_tile_idx, 0))
        gQ = cute.local_tile(mQ_slice, (self.tile_shape_qk[0], self.tile_shape_qk[2]), coord=(work_desc.qo_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (self.tile_shape_qk[1], self.tile_shape_qk[2]), coord=(None, 0))
        if cutlass.const_expr(self.v_block_size == 128):
            gV = mV_slice
        else:
            gV = cute.local_tile(mV_slice, (self.tile_shape_pv[1], self.tile_shape_pv[2]), coord=(0, None))

        gIndices = blocksparse_indices_q2k[None, work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        if cutlass.const_expr(self.has_block_nums):
            num_n_tiles = blocksparse_num_blocks_q2k[work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        else:
            num_n_tiles = block_sparse_num
        if cutlass.const_expr(self.has_block_sizes):
            if cutlass.const_expr(self.block_sizes_mode == 1):
                gBSZ = blocksparse_varblk
            elif cutlass.const_expr(self.block_sizes_mode == 2):
                gBSZ = blocksparse_varblk[None, work_desc.batch_idx]
            else:
                gBSZ = blocksparse_varblk[None, work_desc.qo_head_idx, work_desc.batch_idx]

        cta_coord_layout = (0, cute.make_layout(1))  # CTA coord layout for TMA multicasting, effectively no multicast
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            *cta_coord_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            *cta_coord_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )

        # Keep the dynamic budget within the CTA allocation: 256 * 240 +
        # 128 * 24 = 64512 registers. Raising the donor budget to 32 while
        # retaining 240 compute registers can deadlock setmaxnreg.inc.
        if warp_idx >= self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_registers)
        if warp_idx == self.num_mma_warps:
            Q_pipeline.producer_acquire(Q_producer_state)
            cute.copy(
                tma_atom_Q,
                tQgQ,
                tQsQ[None, 0],
                tma_bar_ptr=Q_pipeline.producer_get_barrier(Q_producer_state),
            )
            Q_pipeline.producer_commit(Q_producer_state)
            for load_count in cutlass.range(num_n_tiles, unroll=1):
                physical_idx = gIndices[num_n_tiles - 1 - load_count]
                K_pipeline.producer_acquire(K_producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK[None, physical_idx],
                    tKsK[None, K_producer_state.index],
                    tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                )
                K_pipeline.producer_commit(K_producer_state)
                K_producer_state.advance()
                V_pipeline.producer_acquire(V_producer_state)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, physical_idx],
                    tVsV[None, V_producer_state.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                )
                V_pipeline.producer_commit(V_producer_state)
                V_producer_state.advance()

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.compute_registers)
            cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            tSsQ = thr_mma_qk.partition_A(sQ)
            tSsK = thr_mma_qk.partition_B(sK)
            tSrQ = tiled_mma_qk.make_fragment_A(tSsQ[None, None, None, 0])
            tSrK = tiled_mma_qk.make_fragment_B(tSsK[None, None, None, 0])
            tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((self.tile_shape_qk[0], self.tile_shape_qk[1])), self.acc_dtype)
            tScS = thr_mma_qk.partition_C(cS)

            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            tOsV = thr_mma_pv.partition_B(sVt)
            tOrV = tiled_mma_pv.make_fragment_B(tOsV[None, None, None, 0])
            tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((self.tile_shape_pv[0], self.tile_shape_pv[1])), self.acc_dtype)
            cO = cute.make_identity_tensor(self.tile_shape_pv[:2])
            tOcO = thr_mma_pv.partition_C(cO)

            atom_copy_ldmatrix_Q = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=False,
                    num_matrices=4,
                ),
                self.Q_dtype,
            )
            atom_copy_ldmatrix_K = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=False,
                    num_matrices=4,
                ),
                self.K_dtype,
            )
            if cutlass.const_expr(self.v_block_size == 128):
                atom_copy_ldmatrix_V = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.V_dtype)
            else:
                atom_copy_ldmatrix_V = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix16x16x8bOp(
                        transpose=True,
                        num_matrices=2,
                    ),
                    self.V_dtype,
                )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(atom_copy_ldmatrix_Q, tiled_mma_qk)
            smem_tiled_copy_K = cute.make_tiled_copy_B(atom_copy_ldmatrix_K, tiled_mma_qk)
            smem_tiled_copy_V = cute.make_tiled_copy_B(
                atom_copy_ldmatrix_V,
                tiled_mma_pv,
            )

            thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
            tSsQ_copy = thr_copy_Q.partition_S(sQ)
            tSrQ_copy = thr_copy_Q.retile(tSrQ)
            tSsK_copy = thr_copy_K.partition_S(sK)
            tOsV_copy = thr_copy_V.partition_S(sVt)

            max_m_layout = cute.make_layout(cute.size(layout_utils.reshape_acc_to_mn(tOrO).layout, mode=[0]))
            max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
            sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
            q_softmax_scale_log2e_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
            tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
            for m in cutlass.range_constexpr(cute.size(q_softmax_scale_log2e_m)):
                row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
                q_softmax_scale_log2e_m[m] = gQScale[row_idx] * scale_softmax_log2e if row_idx < mQ.shape[0] else cutlass.Float32(0.0)

            tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
            max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
            sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

            Q_wait_status = Q_pipeline.consumer_try_wait(Q_consumer_state)
            Q_pipeline.consumer_wait(Q_consumer_state, Q_wait_status)
            if cutlass.const_expr(self.q_in_regs):
                tQsQ_p = tSsQ_copy[None, None, None, 0]
                for k_block_idx in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
                    tQsQ_k = tQsQ_p[None, None, k_block_idx]
                    tQsQ_k = cute.make_tensor(
                        tQsQ_k.iterator.align(16),
                        tQsQ_k.layout,
                    )
                    cute.copy(
                        smem_tiled_copy_Q,
                        tQsQ_k,
                        tSrQ_copy[None, None, k_block_idx],
                    )
            Q_pipeline.consumer_release(Q_consumer_state)
            Q_consumer_state.advance()
            # Q is now resident in every compute warp. P uses a different
            # layout over the same bytes, so all Q reads must finish first.
            _compute_barrier(1, self.num_compute_threads)
            # Four pairs of compute warps start after the preceding pair's
            # first softmax. Each one-time named barrier has 128 participants:
            # 64 arriving threads and 64 waiting threads. No per-tile barrier
            # or clock-based delay is introduced.
            if num_n_tiles > 1:
                if warp_idx // 2 == 1:
                    _compute_barrier(2, 128)
                if warp_idx // 2 == 2:
                    _compute_barrier(3, 128)
                if warp_idx // 2 == 3:
                    _compute_barrier(4, 128)

            for load_count in cutlass.range(0, num_n_tiles, 1, unroll=1):
                n_tile_ind = num_n_tiles - 1 - load_count
                n_tile_idx = gIndices[n_tile_ind]
                if cutlass.const_expr(self.has_block_sizes):
                    varblk = gBSZ[n_tile_idx]
                else:
                    varblk = cutlass.Int32(self.tile_size)
                    if n_tile_idx == num_compute_tiles - 1:
                        varblk = seqlen - n_tile_idx * self.tile_size

                # consumer_wait already checks completion and loops as needed.
                # A separate immediate try_wait adds a redundant wait/branch.
                K_pipeline.consumer_wait(K_consumer_state)

                k_stage = K_consumer_state.index

                _gemm_smem_zero_acc_fp8(
                    tiled_mma_qk,
                    tSrS,
                    tSrQ,
                    tSrK,
                    tSsQ_copy[None, None, None, 0],
                    tSsK_copy[None, None, None, k_stage],
                    smem_tiled_copy_Q,
                    smem_tiled_copy_K,
                    A_in_regs=self.q_in_regs,
                    B_uses_fp8_ldsm=True,
                )

                tKScale = _load_sage_k_scales_fp8(
                    tScS,
                    gKScale,
                    n_tile_idx,
                    self.tile_size,
                )

                K_pipeline.consumer_release(K_consumer_state)
                K_consumer_state.advance()

                if varblk < self.tile_size:
                    _mask_fp8(tiled_mma_qk, tSrS, tScS, varblk)
                row_scale = _online_softmax_ordered_fp8(
                    tiled_mma_qk,
                    tSrS,
                    max_m,
                    sum_m,
                    q_softmax_scale_log2e_m,
                    self.softmax_p_scale_log2,
                    tKScale,
                )

                if load_count == 0 and num_n_tiles > 1:
                    if warp_idx // 2 == 0:
                        _compute_barrier(2, 128, wait=False)
                    if warp_idx // 2 == 1:
                        _compute_barrier(3, 128, wait=False)
                    if warp_idx // 2 == 2:
                        _compute_barrier(4, 128, wait=False)
                # Compute P @ V without changing the accumulation order.
                _rescale_o_if_needed_fp8(tOrO, row_scale)
                tOrP = _make_acc_into_fp8_smem(tSrS, sP[None, None, 0], tiled_mma_qk, tiled_mma_pv)

                V_pipeline.consumer_wait(V_consumer_state)

                if cutlass.const_expr(self.v_block_size == 0):
                    # Load each 16x32 N-major V subtile with the native 8-bit
                    # transpose path. Hopper WGMMA writes the transposed values back
                    # to K-major SMEM, but SM120 warp MMA consumes B from registers,
                    # so the LDSM.T result can directly form the FP8 B fragment.
                    destination_layout = cute.make_layout(
                        ((4, (2, 2)),),
                        stride=((1, (16, 4)),),
                    )
                    for k_block in cutlass.range_constexpr(self.tile_size // 32):
                        for n_block in cutlass.range_constexpr(8):
                            source_ptr = sV.iterator + cute.crd2idx(
                                (n_block * 16, k_block * 32 + lane_idx, 0),
                                sV.layout,
                            )
                            source = cute.make_tensor(
                                source_ptr.align(16),
                                cute.make_layout(16),
                            )
                            # B registers are interleaved in pairs of K32 fragments.
                            # The second K64 pair starts after all eight N16 groups.
                            destination_base = n_block * 8 + (k_block % 2) * 2 + (k_block // 2) * 64
                            destination = cute.make_tensor(
                                tOrV.iterator + destination_base * 4,
                                destination_layout,
                            )
                            cute.copy(
                                atom_copy_ldmatrix_V,
                                source,
                                destination,
                            )

                _gemm_rs_fp8(
                    tiled_mma_pv,
                    tOrO,
                    tOrP,
                    tOrV,
                    tOsV_copy[None, None, None, 0],
                    smem_tiled_copy_V,
                    B_in_regs=self.v_block_size == 0,
                )
                V_pipeline.consumer_release(V_consumer_state)
                V_consumer_state.advance()

            final_ratio, lse = _finalize_softmax_fp8(
                max_m,
                sum_m,
                q_softmax_scale_log2e_m,
                self.softmax_p_scale_log2,
            )
            _rescale_o_with_sage_v_scale_fp8(
                tOrO,
                tOcO,
                final_ratio,
                gVScale,
            )
            tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
            for m in cutlass.range_constexpr(cute.size(lse)):
                row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
                if tScS_mn[m, 0][1] == 0:
                    if row_idx < mQ.shape[0]:
                        mLSE_slice[row_idx] = lse[m]

            tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
            tOrO_cvt.store(tOrO.load().to(self.O_dtype))

            # FP8 Q needs a separate, larger buffer for the BF16 output tile.
            sO = shared_storage.O_smem.get_tensor(O_smem_layout.outer, swizzle=O_smem_layout.inner)
            tiled_copy_o_r2s = cute.make_tiled_copy_C(
                cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.O_layout.is_m_major_c(), 4),
                    self.O_dtype,
                ),
                tiled_mma_pv,
            )
            tOrO_cv = tiled_copy_o_r2s.retile(tOrO_cvt)
            tOsO = tiled_copy_o_r2s.get_slice(tidx).partition_D(sO)
            cute.copy(tiled_copy_o_r2s, tOrO_cv, tOsO)

            cute.arch.fence_view_async_shared()
            # Producer/donor warps have exited; synchronize only consumers.
            _compute_barrier(1, self.num_compute_threads)

            # S2G with explicit TMA store wait.
            tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                tma_atom_O,
                *cta_coord_layout,
                cute.group_modes(sO, 0, 2),
                cute.group_modes(gO, 0, 2),
            )
            if warp_idx == 0:
                cute.copy(tma_atom_O, tOsO, tOgO)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
