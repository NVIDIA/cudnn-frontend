# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MXFP8 grouped GEMM wgrad (2Dx2D), dense output and Tensor2D inputs, from global descriptors.

Same tiles, mainloop and dW bits as ``BlockScaledMoEGroupedGemmWgradKernel`` at 2-CTA 256x256 tiles in a (2, 1)
cluster, but:

    - A and B are read through one host-built TMA descriptor each, offset by the expert's first token, and the
      scale factors through one descriptor each over their bytes viewed as 64-bit words (each load is one
      512-byte 128x4 SF atom), so no helper kernel builds per-expert descriptors before every launch;
    - the epilogue stores per warp: each warp converts its 32 accumulator rows to the output type and stores them
      in 32 x 64 boxes through its own shared memory buffer, with no barrier between the four epilogue warps.

An expert whose token count is not a multiple of the K tile (128 tokens) reads its last K tile of A and B through
per-expert descriptors that the load warp builds into the workspace, so the bounds match the stock kernel's for
every input.

The Rubin subclass (``moe_blockscaled_grouped_gemm_wgrad_dense_rubin``) runs this kernel with a 256-token K tile (two SF
atoms per K tile) and two disjoint accumulator stages; the defaults here are the SM100 ones.
"""

from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, TmemAllocator
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ..canonical import kernel_facing_wgrad_sf
from ..moe_persistent_scheduler import MoEPersistentTileScheduler, MoESchedulerParams, MoEWorkTileInfo
from ..moe_utils import MoEWeightMode, WGradInputOrder, compute_expert_token_range
from .moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel


class BlockScaledMoEGroupedGemmWgradDenseKernel(BlockScaledMoEGroupedGemmWgradKernel):
    """MXFP8 wgrad for dense output and Tensor2D inputs at 2-CTA 256x256 tiles in a (2, 1) cluster.

    Takes the parameters of ``BlockScaledMoEGroupedGemmWgradKernel``; see ``can_implement`` for the configurations.
    The workspace layout is the stock kernel's: per-expert descriptor slots (only the A and B slots of experts
    whose token count is not a multiple of the K tile are written), then the atomic-counter-claim counter.
    """

    K_TILE = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.supports_config(
            sf_vec_size=self.sf_vec_size,
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler[:2],
            cluster_shape_mn=self.cluster_shape_mn,
            accumulate_on_output=self.accumulate_on_output,
            weight_mode=self.weight_mode,
            input_order=self.input_order,
            sf_fp8_dtype_override=None if self.sf_dtype_override is None else "e5m3",
        ):
            raise ValueError("The global-descriptor wgrad supports MXFP8 at 2-CTA 256x256 tiles in a (2, 1) cluster, dense output and Tensor2D input only")
        self.num_ab_stage = 6
        self.num_sched_stages = 6
        self.epi_rows = 32
        self.epi_cols = 64

    @staticmethod
    def supports_config(
        sf_vec_size,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        accumulate_on_output,
        weight_mode,
        input_order,
        sf_fp8_dtype_override,
    ) -> bool:
        """Static configuration this kernel implements (the data types and shapes are checked by ``can_implement``)."""
        return (
            sf_vec_size == 32
            and use_2cta_instrs
            and tuple(mma_tiler_mn) == (256, 256)
            and tuple(cluster_shape_mn) == (2, 1)
            and not accumulate_on_output
            and weight_mode == MoEWeightMode.DENSE
            and WGradInputOrder(input_order) == WGradInputOrder.Tensor2D
            and sf_fp8_dtype_override is None
        )

    @staticmethod
    def can_implement(m: int, n: int, ab_bits: int, c_bits: int, has_global_scale: bool) -> bool:
        """dW (m, n) per expert in whole 256x256 tiles, 8-bit A/B, 16-bit dW and no global scales."""
        return m > 0 and n > 0 and m % 256 == 0 and n % 256 == 0 and ab_bits == 8 and c_bits == 16 and not has_global_scale

    @staticmethod
    def _sf_atom(k_tile, sf_k, sf_k_atoms: int, k_atoms):
        """The expert's SF atom (within a 128-row block) for atom ``sf_k`` of K tile ``k_tile``. With more than one
        atom per K tile, a partial last K tile can end inside its first atom: its later atoms repeat the expert's
        last one, whose scale factors then multiply only the zeros the bounded A and B loads put there."""
        if sf_k_atoms == 1:
            return k_tile
        return cutlass.min(k_tile * sf_k_atoms + sf_k, k_atoms - 1)

    def _setup_attributes(self) -> None:
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (self.mma_inst_shape_mn[0] // 2, cute.round_up(self.mma_inst_shape_mn[1], 128))
        tiled_mma = self._create_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        mma_tiler_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (self.mma_inst_shape_mn[0], self.mma_inst_shape_mn[1], mma_tiler_k)
        self.mma_tiler_sfb = (self.mma_inst_shape_mn_sfb[0], self.mma_inst_shape_mn_sfb[1], mma_tiler_k)
        self.cta_tile_shape_mnk = (self.mma_tiler[0] // 2, self.mma_tiler[1], self.mma_tiler[2])
        tiled_mma_sfb = self._create_tiled_mma_sfb()
        # 512-byte SF atoms (128 tokens each) per K tile.
        self.sf_k_atoms = self.mma_tiler[2] // 128

        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)

        # dW staging: one 32 x 64 16-bit buffer per epilogue warp (4 KB, rows of 128 B, 128B swizzle).
        c_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, self.c_dtype)
        self.c_smem_layout_staged = cute.tile_to_shape(c_atom, (self.epi_rows, self.epi_cols, len(self.epilogue_warp_id)), order=(0, 1, 2))

        # Accumulator: two TMEM stages overlapping by the SF columns (stage 0 at 0, stage 1 at 256 - SF columns).
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.mma_tiler_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.acc_stage_stride = 256 - self.num_sf_tmem_cols
        self.num_accumulator_tmem_cols = 256 + self.acc_stage_stride
        # One accumulator pipeline stage alternating between the two TMEM stages (by phase).
        self.overlapping_accum = True
        self.num_acc_pipeline_stages = 1
        assert self.num_sf_tmem_cols <= self.epi_cols, "the first epilogue chunk must cover the accumulator overlap"

        a_bytes = cute.size_in_bytes(self.a_dtype, cute.slice_(self.a_smem_layout_staged, (None, None, None, 0)))
        b_bytes = cute.size_in_bytes(self.b_dtype, cute.slice_(self.b_smem_layout_staged, (None, None, None, 0)))
        sfa_bytes = cute.size_in_bytes(self.sf_dtype, cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)))
        sfb_bytes = cute.size_in_bytes(self.sf_dtype, cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)))
        assert sfa_bytes == 512 * self.sf_k_atoms and sfb_bytes == 1024 * self.sf_k_atoms, "one 512-byte SF atom per 128 rows and 128 tokens"
        self.num_tma_load_bytes = (a_bytes + b_bytes + sfa_bytes + sfb_bytes) * 2

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,  # (hidden, tokens_sum)
        mat_b: cute.Tensor,  # (tokens_sum, intermediate)
        scale_a: cute.Tensor,  # SFA: per expert, its 128x4 atoms
        scale_b: cute.Tensor,  # SFB, as SFA
        out: cute.Tensor,  # (experts, hidden, intermediate)
        offs: cute.Tensor,  # (experts,) cumulative token ends, int32
        workspace: cute.Tensor,  # descriptor slots for experts with a partial K tile, then the claim counter
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        global_scale_a: Optional[cute.Tensor] = None,
        global_scale_b: Optional[cute.Tensor] = None,
        out_single_expert: Optional[cute.Tensor] = None,
    ) -> None:
        assert global_scale_a is None and global_scale_b is None and out_single_expert is None
        scale_a = kernel_facing_wgrad_sf(scale_a, mat_a.shape[0], mat_a.shape[1], self.sf_vec_size)
        scale_b = kernel_facing_wgrad_sf(scale_b, mat_b.shape[1], mat_b.shape[0], self.sf_vec_size)
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        hidden, tokens_sum = mat_a.shape
        a_gemm = cute.make_tensor(mat_a.iterator, cute.make_layout((hidden, tokens_sum, c1), stride=(mat_a.stride[0], mat_a.stride[1], c0)))
        tokens_sum_b, intermediate = mat_b.shape
        b_gemm = cute.make_tensor(mat_b.iterator, cute.make_layout((intermediate, tokens_sum_b, c1), stride=(mat_b.stride[1], mat_b.stride[0], c0)))
        expert_cnt, hidden_c, intermediate_c = out.shape
        c_gemm = cute.make_tensor(
            out.iterator,
            cute.make_layout((hidden_c, intermediate_c, expert_cnt), stride=(out.stride[1], out.stride[2], out.stride[0])),
        )
        # SF bytes as 16-byte rows of two 64-bit words; one load is 32 rows (one 512-byte atom) at any row.
        sfa_rows = cute.make_tensor(cute.recast_ptr(scale_a.iterator, dtype=cutlass.Int64), cute.make_layout((2, cute.size(scale_a) // 16), stride=(1, 2)))
        sfb_rows = cute.make_tensor(cute.recast_ptr(scale_b.iterator, dtype=cutlass.Int64), cute.make_layout((2, cute.size(scale_b) // 16), stride=(1, 2)))

        self.a_dtype = a_gemm.element_type
        self.b_dtype = b_gemm.element_type
        self.c_dtype = c_gemm.element_type
        self.sf_dtype = scale_a.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a_gemm).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(b_gemm).mma_major_mode()
        self.c_layout = LayoutEnum.from_tensor(c_gemm)

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # SF atoms: the 2-SM op signals the leader CTA's barrier.
        sf_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        sf_box = cute.make_layout((2, 32), stride=(1, 2))
        tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(sf_op, sfa_rows, sf_box, (2, 32))
        tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(sf_op, sfb_rows, sf_box, (2, 32))

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_gemm, epi_smem_layout, (self.epi_rows, self.epi_cols))

        sched_params = MoESchedulerParams(
            scenario="2Dx2D",
            expert_shape=(expert_cnt, intermediate, hidden),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            atomic_counter_claim=self.atomic_counter_claim,
        )
        grid = MoESchedulerParams.get_grid_shape(sched_params, max_active_clusters)

        # No per-expert descriptor build: a helper launch only resets the atomic-counter-claim counter.
        if cutlass.const_expr(self.atomic_counter_claim):
            self.counter_reset_kernel(workspace.iterator).launch(
                grid=(1, 1, 1),
                block=(1, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=self.use_pdl,
            )

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            a_gemm,
            b_gemm,
            offs,
            sched_params,
            workspace.iterator,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def counter_reset_kernel(self, workspace_ptr):
        """Reset the atomic-counter-claim counter before the main kernel claims from it."""
        # use_pdl: the kernel before this one on the stream may still be using the workspace.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        counter = cute.make_tensor(self._get_sched_counter_ptr(workspace_ptr), cute.make_layout(1))
        counter[0] = cutlass.Int32(0)

    counter_reset_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _write_expert_ab_descs(
        self, tiled_mma, cluster_layout_vmnk, a_smem_layout, b_smem_layout, a_gemm, b_gemm, token_offset, tokens_i, desc_a_ptr, desc_b_ptr
    ):
        """Build this expert's A and B descriptors (K bounded by its token count) as the stock helper kernel does."""
        from ..moe_utils import store_tma_desc

        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        a_offset = (cutlass.Int64(a_gemm.stride[1]) * cutlass.Int64(token_offset) * self.a_dtype.width) // 8
        a_iter = cute.recast_ptr(cute.recast_ptr(a_gemm.iterator, dtype=cutlass.Uint8) + a_offset, dtype=self.a_dtype)
        a_expert = cute.make_tensor(a_iter, cute.make_layout((cute.size(a_gemm, mode=[0]), tokens_i, c1), stride=(a_gemm.stride[0], a_gemm.stride[1], c0)))
        tma_atom_a, _ = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a_expert,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        store_tma_desc(tma_atom_a, desc_a_ptr)
        b_offset = (cutlass.Int64(b_gemm.stride[1]) * cutlass.Int64(token_offset) * self.b_dtype.width) // 8
        b_iter = cute.recast_ptr(cute.recast_ptr(b_gemm.iterator, dtype=cutlass.Uint8) + b_offset, dtype=self.b_dtype)
        b_expert = cute.make_tensor(b_iter, cute.make_layout((cute.size(b_gemm, mode=[0]), tokens_i, c1), stride=(b_gemm.stride[0], b_gemm.stride[1], c0)))
        tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b_expert,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        store_tma_desc(tma_atom_b, desc_b_ptr)

    def epi_store_chunk(
        self, tTR_rC_vec, r2s_atom, warp_buffer_base, row_offset, row_swizzle, tma_atom_c, bSG_sC, bSG_gC, warp_idx, row_block, column_block, expert_idx
    ):
        """Store one 32 x 64 box of this warp: wait for its staging buffer, write it (128B swizzle), TMA store."""
        # The store that last used this buffer must have finished reading shared memory.
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.sync_warp()
        row_address = warp_buffer_base + row_offset
        for j in range(8):
            dst = cute.make_tensor(
                cute.make_ptr(self.c_dtype, row_address + (row_swizzle ^ (j * 16)), cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(8),
            )
            cute.copy(r2s_atom, tTR_rC_vec[(None, j)], dst)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        cute.copy(tma_atom_c, bSG_sC[(None, warp_idx)], bSG_gC[(None, row_block, column_block, expert_idx)])
        cute.arch.cp_async_bulk_commit_group()

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC: cute.Tensor,
        a_gemm: cute.Tensor,
        b_gemm: cute.Tensor,
        offs: cute.Tensor,
        sched_params: MoESchedulerParams,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        """GPU device kernel: MXFP8 wgrad from global descriptors with CLC or atomic-counter scheduling."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % 2
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        is_leader_cta = block_in_cluster_coord_vmnk[0] == 0
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.launch_pdl_dependents_at_entry):
            cute.arch.griddepcontrol_launch_dependents()

        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_sched_stages, use_dynamic_sched=True)

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_pipeline_stages * 2]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sched_storage = storage.scheduler

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.epilogue_warp_id) * 32 * 2),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_sched_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * (2 + len(self.epilogue_warp_id))),
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * (1 + len(self.epilogue_warp_id)),
        )
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.architecture,
        )

        # Scheduler (2Dx2D: CLC, or the workspace's atomic counter with atomic_counter_claim)
        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            offs,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr) if cutlass.const_expr(self.atomic_counter_claim) else None,
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(element_type=self.a_dtype, layout=a_smem_layout_staged.outer, byte_alignment=1024, swizzle=a_smem_layout_staged.inner)
        sB = smem.allocate_tensor(element_type=self.b_dtype, layout=b_smem_layout_staged.outer, byte_alignment=1024, swizzle=b_smem_layout_staged.inner)
        sSFA = smem.allocate_tensor(element_type=self.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128)
        sSFB = smem.allocate_tensor(element_type=self.sf_dtype, layout=sfb_smem_layout_staged, byte_alignment=128)
        sC = smem.allocate_tensor(element_type=self.c_dtype, layout=c_smem_layout_staged.outer, byte_alignment=1024, swizzle=c_smem_layout_staged.inner)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(tCtAcc_fake.stride[0], tCtAcc_fake.stride[1], tCtAcc_fake.stride[2], self.acc_stage_stride * tCtAcc_fake.stride[0][1]),
            ),
        )

        sched_buf_tensor = cute.make_tensor(sched_storage.sInfo.data_ptr(), cute.make_layout((4, self.num_sched_stages), stride=(1, 4)))
        sched_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # use_pdl: the setup above touches no global memory; the kernel before this one on the stream (the counter
        # reset, or the producer of A, B, the scale factors or the offsets) may still be running, so every thread
        # waits for it here.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # TMEM: epilogue warp 0 allocates, and the MMA and epilogue warps wait at this one site: the allocation
        # barrier is aligned, so every warp must reach it at the same instruction.
        tmem.allocate(self.num_tmem_alloc_cols)
        if warp_idx <= self.mma_warp_id:
            tmem.wait_for_alloc()

        # =================================================================
        # Scheduler warp
        # =================================================================
        if warp_idx == self.sched_warp_id:
            sched_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_sched_stages)
            work_tile_info = scheduler.initial_work_tile_info()
            # atomic_counter_claim: a cluster whose first claim is already past the last tile publishes only the
            # sentinel (see the stock kernel). A CLC grid's first tile is always valid.
            if cutlass.const_expr(not self.atomic_counter_claim):
                sched_pipeline.producer_acquire(sched_producer_state)
                cute.copy(sched_copy_atom, work_tile_info.to_rmem_tensor(), sched_buf_tensor[(None, sched_producer_state.index)])
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()
                work_tile_info = scheduler.advance_to_next_work()
            while work_tile_info.is_valid_tile:
                sched_pipeline.producer_acquire(sched_producer_state)
                cute.copy(sched_copy_atom, work_tile_info.to_rmem_tensor(), sched_buf_tensor[(None, sched_producer_state.index)])
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()
                work_tile_info = scheduler.advance_to_next_work()
            sched_pipeline.producer_acquire(sched_producer_state)
            sentinel = MoEWorkTileInfo(cutlass.Int32(-1), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))
            cute.copy(sched_copy_atom, sentinel.to_rmem_tensor(), sched_buf_tensor[(None, sched_producer_state.index)])
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            # producer_tail takes the state of the next buffer, not the sentinel's.
            sched_producer_state.advance()
            sched_pipeline.producer_tail(sched_producer_state)

        # =================================================================
        # TMA load warp
        # =================================================================
        if warp_idx == self.tma_warp_id:
            from ..moe_utils import TensormapWorkspace, WgradSfTensormapConstructor, acquire_tma_desc, rewrite_tensor_shape, tensormap_ptr_for_copy

            desc_workspace = TensormapWorkspace(workspace_ptr, WgradSfTensormapConstructor.slot_names(self.input_order, self.weight_mode))
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            sA_grouped = cute.group_modes(sA, 0, 3)
            sB_grouped = cute.group_modes(sB, 0, 3)

            # SF in shared memory as 16-byte rows: per stage and 128 tokens, SFA one atom, SFB two (its 256 columns); with
            # more than one atom per stage the K atoms are inner, as make_smem_layout_sfa/sfb tile them.
            sf_k_atoms = self.sf_k_atoms
            sSFA_rows = cute.make_tensor(
                cute.recast_ptr(sSFA.iterator, dtype=cutlass.Int64), cute.make_layout((2, 32, sf_k_atoms * self.num_ab_stage), stride=(1, 2, 64))
            )
            if cutlass.const_expr(sf_k_atoms == 1):
                sSFB_rows = cute.make_tensor(
                    cute.recast_ptr(sSFB.iterator, dtype=cutlass.Int64), cute.make_layout((2, 32, 2, self.num_ab_stage), stride=(1, 2, 64, 128))
                )
            else:
                sSFB_rows = cute.make_tensor(
                    cute.recast_ptr(sSFB.iterator, dtype=cutlass.Int64),
                    cute.make_layout((2, 32, sf_k_atoms, 2, self.num_ab_stage), stride=(1, 2, 64, 64 * sf_k_atoms, 128 * sf_k_atoms)),
                )

            a_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            hidden = cute.size(mA_mkl, mode=[0])
            intermediate = cute.size(mB_nkl, mode=[0])

            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)
            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(sched_copy_atom, sched_buf_tensor[(None, sched_consumer_state.index)], rmem)
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                expert_idx = work_tile_info.expert_idx
                token_offset, tokens_i = compute_expert_token_range(offs, expert_idx)
                # A partial last K tile reads through this expert's own descriptors, bounded at its token count.
                has_partial_k_tile = tokens_i % self.K_TILE != 0
                desc_a_raw = desc_workspace.get_ptr("a", expert_idx)
                desc_b_raw = desc_workspace.get_ptr("b", expert_idx)
                if has_partial_k_tile:
                    self._write_expert_ab_descs(
                        tiled_mma,
                        cluster_layout_vmnk,
                        cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
                        cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
                        a_gemm,
                        b_gemm,
                        token_offset,
                        tokens_i,
                        desc_a_raw,
                        desc_b_raw,
                    )
                    acquire_tma_desc(desc_a_raw)
                    acquire_tma_desc(desc_b_raw)
                desc_a = tensormap_ptr_for_copy(desc_a_raw, acquire=False)
                desc_b = tensormap_ptr_for_copy(desc_b_raw, acquire=False)

                # Expert-local K coordinates: offset on the global descriptors, unshifted on the expert's own.
                tile_m = work_tile_info.tile_m_idx // 2
                tile_n = work_tile_info.tile_n_idx
                gA = cute.local_tile(cute.domain_offset((0, token_offset, 0), mA_mkl), cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB = cute.local_tile(cute.domain_offset((0, token_offset, 0), mB_nkl), cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout, sA_grouped, cute.group_modes(thr_mma.partition_A(gA), 0, 3)
                )
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout, sB_grouped, cute.group_modes(thr_mma.partition_B(gB), 0, 3)
                )
                gA_own = cute.local_tile(
                    rewrite_tensor_shape(mA_mkl, (hidden, tokens_i, cutlass.Int32(1))), cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
                )
                gB_own = cute.local_tile(
                    rewrite_tensor_shape(mB_nkl, (intermediate, tokens_i, cutlass.Int32(1))), cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
                )
                _, tAgA_own = cpasync.tma_partition(
                    tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout, sA_grouped, cute.group_modes(thr_mma.partition_A(gA_own), 0, 3)
                )
                _, tBgB_own = cpasync.tma_partition(
                    tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout, sB_grouped, cute.group_modes(thr_mma.partition_B(gB_own), 0, 3)
                )
                tAgA_slice = tAgA[(None, tile_m, None, 0)]
                tBgB_slice = tBgB[(None, tile_n, None, 0)]
                tAgA_own_slice = tAgA_own[(None, tile_m, None, 0)]
                tBgB_own_slice = tBgB_own[(None, tile_n, None, 0)]

                # SF: the expert's atoms start at byte hidden * token_offset / 32 (hidden is a multiple of 256, so
                # that is the stock kernel's per-expert SF descriptor base), k_atoms 512-byte atoms (one per 128
                # tokens) per 128 rows.
                k_atoms = k_tile_cnt
                if cutlass.const_expr(sf_k_atoms > 1):
                    k_atoms = (tokens_i + 127) // 128
                sfa_first_row = (
                    cutlass.Int32((cutlass.Int64(hidden) * cutlass.Int64(token_offset)) // (self.sf_vec_size * 16)) + work_tile_info.tile_m_idx * k_atoms * 32
                )
                sfb_first_row = (
                    cutlass.Int32((cutlass.Int64(intermediate) * cutlass.Int64(token_offset)) // (self.sf_vec_size * 16)) + tile_n * 2 * k_atoms * 32
                )
                sfb_half_rows = k_atoms * 32

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                    if has_partial_k_tile and handle.count + 1 == k_tile_cnt:
                        cute.copy(
                            tma_atom_a,
                            tAgA_own_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_a,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_own_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_b,
                            mcast_mask=b_mcast_mask,
                        )
                    else:
                        cute.copy(tma_atom_a, tAgA_slice[(None, handle.count)], tAsA[(None, handle.index)], tma_bar_ptr=handle.barrier, mcast_mask=a_mcast_mask)
                        cute.copy(tma_atom_b, tBgB_slice[(None, handle.count)], tBsB[(None, handle.index)], tma_bar_ptr=handle.barrier, mcast_mask=b_mcast_mask)
                    for sf_k in cutlass.range_constexpr(sf_k_atoms):
                        k_rows = self._sf_atom(handle.count, sf_k, sf_k_atoms, k_atoms) * 32
                        sfa_stage = handle.index
                        sfb_k = ()
                        if cutlass.const_expr(sf_k_atoms > 1):
                            sfa_stage = handle.index * sf_k_atoms + sf_k
                            sfb_k = (sf_k,)
                        for sf_atom, mSF, first_row, sSF_stage in (
                            (tma_atom_sfa, mSFA, sfa_first_row, sSFA_rows[(None, None, sfa_stage)]),
                            (tma_atom_sfb, mSFB, sfb_first_row, sSFB_rows[(None, None, *sfb_k, 0, handle.index)]),
                            (tma_atom_sfb, mSFB, sfb_first_row + sfb_half_rows, sSFB_rows[(None, None, *sfb_k, 1, handle.index)]),
                        ):
                            gSF = cute.local_tile(cute.domain_offset((0, first_row + k_rows), mSF), (2, 32), (None, None))
                            tsSF, tgSF = cpasync.tma_partition(sf_atom, 0, cute.make_layout(1), cute.group_modes(sSF_stage, 0, 2), cute.group_modes(gSF, 0, 2))
                            cute.copy(sf_atom, tgSF[(None, 0, 0)], tsSF, tma_bar_ptr=handle.barrier)

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(sched_copy_atom, sched_buf_tensor[(None, sched_consumer_state.index)], rmem)
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()
            ab_producer.tail()

        # =================================================================
        # MMA warp
        # =================================================================
        if warp_idx == self.mma_warp_id:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sf_window_tiler_mnk = (self.mma_tiler[0], self.mma_tiler[1], self.mma_tiler[2])
            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype)
            sfa_window_smem_layout = cute.slice_(
                blockscaled_utils.make_smem_layout_sfa(tiled_mma, sf_window_tiler_mnk, self.sf_vec_size, 1), (None, None, None, 0)
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, sf_window_tiler_mnk, self.sf_vec_size, sfa_window_smem_layout)
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols, dtype=self.sf_dtype)
            sfb_window_smem_layout = cute.slice_(
                blockscaled_utils.make_smem_layout_sfb(tiled_mma, sf_window_tiler_mnk, self.sf_vec_size, 1), (None, None, None, 0)
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, sf_window_tiler_mnk, self.sf_vec_size, sfb_window_smem_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(sched_copy_atom, sched_buf_tensor[(None, sched_consumer_state.index)], rmem)
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                if is_leader_cta:
                    tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]
                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        peek_ab_full_status = ab_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()
                        s2t_stage_coord = (None, None, None, None, handle.index)
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, [tCrA[tile_crd], tCtSFA], [tCrB[tile_crd], tCtSFB], tCtAcc)
                        handle.release()
                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(sched_copy_atom, sched_buf_tensor[(None, sched_consumer_state.index)], rmem)
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        # =================================================================
        # Epilogue warps: per warp TMEM -> registers -> 16-bit -> swizzled smem -> TMA store (32 x 64 boxes)
        # =================================================================
        if warp_idx < self.mma_warp_id:
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            # (T2R, T2R_M, T2R_N, 1, chunk, stage): this lane's 64 fp32 columns of a chunk of a stage
            epi_tile = (self.cta_tile_shape_mnk[0], self.epi_cols)
            tAcc_epi = cute.flat_divide(transform_partitioned_tensor_layout(tCtAcc_base), epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, True),
                tAcc_epi[(None, None, 0, 0, 0)],
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx % 128)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(thr_copy_t2r.partition_D(tAcc_epi)[(None, None, None, 0, 0, 0)].shape, self.acc_dtype)
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rC_vec = cute.make_tensor(tTR_rC.iterator, cute.make_layout((8, self.epi_cols // 8)))
            r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128)

            gC = cute.local_tile(mC, (self.epi_rows, self.epi_cols), (None, None, None))
            bSG_sC, bSG_gC = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gC, 0, 2))

            lane = tidx % 32
            c_bytes_per_buffer = self.epi_rows * self.epi_cols * self.c_dtype.width // 8
            warp_buffer_base = sC.iterator.toint() + warp_idx * c_bytes_per_buffer
            row_offset = lane * 128
            row_swizzle = (lane % 8) * 16

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_pipeline_stages)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(sched_copy_atom, sched_buf_tensor[(None, sched_consumer_state.index)], rmem)
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            chunks = self.cta_tile_shape_mnk[1] // self.epi_cols
            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                expert_idx = work_tile_info.expert_idx
                row_block = work_tile_info.tile_m_idx * (self.cta_tile_shape_mnk[0] // self.epi_rows) + warp_idx
                first_column_block = work_tile_info.tile_n_idx * chunks
                if k_tile_cnt > 0:
                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_consumer_state.phase
                        reverse_order = 1 - acc_stage_index
                    else:
                        acc_stage_index = acc_consumer_state.index
                    tTR_tAcc = cute.group_modes(tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)], 3, 5)
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    for chunk in cutlass.range_constexpr(chunks):
                        # Overlapping stages: stage 0 is read back to front and stage 1 front to back, so the first
                        # chunk read covers the columns the two stages share, and the stage is released after it.
                        # Disjoint stages: front to back, released after the last chunk.
                        real_chunk = chunk
                        if cutlass.const_expr(self.overlapping_accum):
                            real_chunk = chunk + reverse_order * (chunks - 1 - 2 * chunk)
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, real_chunk)], tTR_rAcc)
                        if chunk == (0 if self.overlapping_accum else chunks - 1):
                            cute.arch.fence_view_async_tmem_load()
                            acc_pipeline.consumer_release(acc_consumer_state)
                        tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
                        self.epi_store_chunk(
                            tTR_rC_vec,
                            r2s_atom,
                            warp_buffer_base,
                            row_offset,
                            row_swizzle,
                            tma_atom_c,
                            bSG_sC,
                            bSG_gC,
                            warp_idx,
                            row_block,
                            first_column_block + real_chunk,
                            expert_idx,
                        )
                    acc_consumer_state.advance()
                else:
                    tTR_rC.fill(0)
                    for chunk in cutlass.range_constexpr(chunks):
                        self.epi_store_chunk(
                            tTR_rC_vec,
                            r2s_atom,
                            warp_buffer_base,
                            row_offset,
                            row_swizzle,
                            tma_atom_c,
                            bSG_sC,
                            bSG_gC,
                            warp_idx,
                            row_block,
                            first_column_block + chunk,
                            expert_idx,
                        )

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(sched_copy_atom, sched_buf_tensor[(None, sched_consumer_state.index)], rmem)
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            cute.arch.cp_async_bulk_wait_group(0, read=True)
            tmem.relinquish_alloc_permit()
            pipeline.NamedBarrier(barrier_id=self.epilog_sync_bar_id, num_threads=32 * len(self.epilogue_warp_id)).arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
