# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MXFP8 grouped GEMM wgrad (2Dx2D), dense output and Tensor2D inputs, for short experts: A kept in shared memory
across N tiles.

A work unit is up to four N tiles of one (expert, M tile). When the expert's K tiles fit the A ring (``a_stages``
stages), A and its scale factors are loaded once per unit and reused for every N tile of it; B streams through its own
ring. Otherwise A is reloaded per N tile. Loads, MMA, tile claims and dW bits are those of
``BlockScaledMoEGroupedGemmWgradDenseKernel``: A, B and the scale factors are read through global descriptors (an
expert's partial last K tile through its own A and B descriptors), and each epilogue warp stores its own 32 x 64 boxes.
On a unit's start the load warp prefetches into L2 the B columns of the next N unit (and, when that unit opens the next
expert, this M tile's A rows of it); a hint only.
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
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
)

from ..canonical import kernel_facing_wgrad_sf
from ..moe_persistent_scheduler import MoEPersistentTileScheduler, MoESchedulerParams, MoEWorkTileInfo
from ..moe_utils import compute_expert_token_range
from .moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel


class BlockScaledMoEGroupedGemmWgradReuseKernel(BlockScaledMoEGroupedGemmWgradDenseKernel):
    """MXFP8 wgrad for dense output and Tensor2D inputs at 2-CTA 256x256 tiles in a (2, 1) cluster, A reused across the
    N tiles of a work unit.

    :param a_stages: A / SFA shared-memory stages; B / SFB get the rest.
    Every other parameter is ``BlockScaledMoEGroupedGemmWgradKernel``'s; ``can_implement`` gives the shapes it is for.
    """

    MAX_N_GROUP = 4  # N tiles per work unit, at most
    L2_PREFETCH_UNITS = 1  # units ahead the load warp prefetches into L2
    A_STAGES = 6  # default A / SFA ring depth

    def __init__(self, *args, a_stages: int = A_STAGES, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_stages = a_stages
        self.num_epi_warps = len(self.epilogue_warp_id)
        self.epi_chunk_n = 64
        self.c_stages_per_warp = 1
        self.smem_reserve = 2048  # barriers, scheduler and TMEM-holding storage

    @staticmethod
    def n_group_for(n: int, max_n_group: int = MAX_N_GROUP) -> int:
        """N tiles per work unit for dW's N: the largest divisor of its 256-column tile count up to max_n_group."""
        n_tiles = n // 256
        return max(candidate for candidate in range(1, max_n_group + 1) if n_tiles % candidate == 0)

    @classmethod
    def short_experts(cls, tokens: int, experts: int, a_stages: Optional[int] = None) -> bool:
        """``tokens`` (the token extent of A and B) over ``experts`` short enough on average that an expert's K tiles
        fit the A ring."""
        a_stages = cls.A_STAGES if a_stages is None else a_stages
        return experts > 0 and -(-tokens // experts) <= a_stages * cls.K_TILE

    @classmethod
    def can_implement(
        cls, m: int, n: int, ab_bits: int, c_bits: int, has_global_scale: bool, tokens: int, experts: int, a_stages: Optional[int] = None
    ) -> bool:
        """The global-descriptor wgrad's shapes, more than one N tile per unit, and short experts on average
        (``short_experts``). Any other token split is still computed correctly, with A reloaded per N tile."""
        return (
            BlockScaledMoEGroupedGemmWgradDenseKernel.can_implement(m, n, ab_bits, c_bits, has_global_scale)
            and cls.n_group_for(n, cls.MAX_N_GROUP) > 1
            and cls.short_experts(tokens, experts, a_stages)
        )

    def _setup_attributes(self) -> None:
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        mma_tiler_k = mma_inst_shape_k * mma_inst_tile_k
        self.sf_window_k = mma_tiler_k
        self.num_sf_windows_per_ab_stage = 1
        self.num_mma_instructions_per_sf_window = mma_inst_tile_k
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_tiler_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_tiler_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # T2R tile: all 128 accumulator lanes x one chunk of columns; C box: one warp's 32 lanes x one chunk.
        self.epi_tile = (self.cta_tile_shape_mnk[0], self.epi_chunk_n)
        self.epi_tile_n = self.epi_chunk_n
        self.c_box = (32, self.epi_chunk_n)
        self.num_c_stage = self.num_epi_warps * self.c_stages_per_warp
        self.num_acc_stage = 2
        a_smem_layout_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_smem_layout_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)
        sfa_smem_layout_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, 1)
        sfb_smem_layout_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, 1)
        a_bytes_per_stage = cute.size_in_bytes(self.a_dtype, a_smem_layout_one) + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout_one)
        b_bytes_per_stage = cute.size_in_bytes(self.b_dtype, b_smem_layout_one) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout_one)
        c_bytes = self.num_c_stage * self.c_box[0] * self.c_box[1] * self.c_dtype.width // 8
        available_bytes = self.smem_capacity - (self.smem_reserve + c_bytes)
        self.num_a_stage = self.a_stages
        self.num_b_stage = (available_bytes - self.num_a_stage * a_bytes_per_stage) // b_bytes_per_stage
        assert self.num_a_stage >= 2 and self.num_b_stage >= 2
        self.num_sched_stages = 2

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_a_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_b_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_a_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_b_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.c_box,
            self.num_c_stage,
        )

        self.overlapping_accum = self.cta_tile_shape_mnk[1] == 256
        self.num_acc_pipeline_stages = 1 if self.overlapping_accum else self.num_acc_stage

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage - (self.num_sf_tmem_cols if self.overlapping_accum else 0)

        # Chunks every epilogue warp drains before releasing the accumulator: covers the columns the other
        # (overlapping) accumulator stage shares.
        self.epi_chunks_per_warp = self.cta_tile_shape_mnk[1] // self.epi_chunk_n // (self.num_epi_warps // 4)
        if self.overlapping_accum:
            self.epi_release_chunks = (self.num_sf_tmem_cols + self.epi_chunk_n - 1) // self.epi_chunk_n
        else:
            self.epi_release_chunks = self.epi_chunks_per_warp
        assert self.num_accumulator_tmem_cols + self.num_sf_tmem_cols <= self.num_tmem_alloc_cols
        assert self.epi_release_chunks <= self.epi_chunks_per_warp

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes_a = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.num_tma_load_bytes_b = (b_copy_size + sfb_copy_size) * atom_thr_size

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,  # (hidden, tokens_sum) - activation^T
        mat_b: cute.Tensor,  # (tokens_sum, intermediate) - activation
        scale_a: cute.Tensor,  # SFA: per expert, its 128x4 atoms
        scale_b: cute.Tensor,  # SFB, as SFA
        out,  # cute.Tensor (experts, hidden, intermediate)
        offs,  # (experts,) cumulative token ends, int32
        workspace: cute.Tensor,  # descriptor slots for experts with a partial K tile, then the claim counter
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        global_scale_a=None,
        global_scale_b=None,
        out_single_expert=None,
    ) -> None:
        assert global_scale_a is None and global_scale_b is None and out_single_expert is None
        scale_a = kernel_facing_wgrad_sf(scale_a, mat_a.shape[0], mat_a.shape[1], self.sf_vec_size)
        scale_b = kernel_facing_wgrad_sf(scale_b, mat_b.shape[1], mat_b.shape[0], self.sf_vec_size)

        # ======== Step 1: Transform to GEMM domain (2Dx2D) ========
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # mat_a: (hidden, tokens_sum) -> A: (M=hidden, K=tokens_sum, L=1)
        hidden, tokens_sum = mat_a.shape
        a_gemm = cute.make_tensor(
            mat_a.iterator,
            cute.make_layout(
                (hidden, tokens_sum, c1),
                stride=(mat_a.stride[0], mat_a.stride[1], c0),
            ),
        )
        # mat_b: (tokens_sum, intermediate) -> B: (N=intermediate, K=tokens_sum, L=1)
        tokens_sum_b, intermediate = mat_b.shape
        b_gemm = cute.make_tensor(
            mat_b.iterator,
            cute.make_layout(
                (intermediate, tokens_sum_b, c1),
                stride=(mat_b.stride[1], mat_b.stride[0], c0),
            ),
        )
        self.n_group = self.n_group_for(intermediate, self.MAX_N_GROUP)
        assert self.n_group > 1, "the reuse wgrad needs more than one N tile per unit (see can_implement)"

        # out: (experts, hidden, intermediate) -> C: (M=hidden, N=intermediate, L=experts)
        expert_cnt, hidden_c, intermediate_c = out.shape
        c_gemm = cute.make_tensor(
            out.iterator,
            cute.make_layout(
                (hidden_c, intermediate_c, expert_cnt),
                stride=(out.stride[1], out.stride[2], out.stride[0]),
            ),
        )

        intermediate_dim = intermediate
        hidden_dim = hidden

        # ======== Step 2: Infer dtypes and major modes ========
        self.a_dtype = a_gemm.element_type
        self.b_dtype = b_gemm.element_type
        self.c_dtype = c_gemm.element_type
        self.sf_dtype = scale_a.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a_gemm).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(b_gemm).mma_major_mode()
        self.c_layout = LayoutEnum.from_tensor(c_gemm)

        # ======== Step 3: Setup kernel attributes ========
        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()

        # ======== Step 4: Create TMA ops ========

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        c_tma_op = cpasync.CopyBulkTensorTileS2GOp()

        # ======== Step 5: Scheduler params and grid ========
        sched_params = MoESchedulerParams(
            scenario="2Dx2D",
            expert_shape=(expert_cnt, intermediate_dim, hidden_dim),
            cta_tile_shape_mnk=(
                self.cta_tile_shape_mnk[0],
                self.cta_tile_shape_mnk[1] * self.n_group,
                self.cta_tile_shape_mnk[2],
            ),
            cluster_shape_mn=self.cluster_shape_mn,
            atomic_counter_claim=self.atomic_counter_claim,
        )
        grid = MoESchedulerParams.get_grid_shape(sched_params, max_active_clusters)

        # ======== Step 6: TMA atoms (global A / B / SF descriptors; no per-expert tensormaps) ========
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # SF bytes as 16-byte rows of two 64-bit words; one load is 32 rows (one 512-byte atom) at any row, as the
        # global-descriptor wgrad reads them. The 2-SM op signals the leader CTA's barrier.
        sf_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        sf_box = cute.make_layout((2, 32), stride=(1, 2))
        sfa_rows = cute.make_tensor(cute.recast_ptr(scale_a.iterator, dtype=cutlass.Int64), cute.make_layout((2, cute.size(scale_a) // 16), stride=(1, 2)))
        sfb_rows = cute.make_tensor(cute.recast_ptr(scale_b.iterator, dtype=cutlass.Int64), cute.make_layout((2, cute.size(scale_b) // 16), stride=(1, 2)))
        tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(sf_op, sfa_rows, sf_box, (2, 32))
        tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(sf_op, sfb_rows, sf_box, (2, 32))
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            c_tma_op,
            c_gemm,
            epi_smem_layout,
            self.c_box,
        )

        # No per-expert descriptor build: a helper launch only resets the atomic-counter-claim counter.
        if cutlass.const_expr(self.atomic_counter_claim):
            self.counter_reset_kernel(workspace.iterator).launch(
                grid=(1, 1, 1),
                block=(1, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=self.use_pdl,
            )

        # ======== Step 7: Launch main kernel ========
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
            self.epi_tile,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
            use_pdl=self.use_pdl,
        )

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
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
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
        epi_tile: cute.Tile,
    ):
        """GPU device kernel: MXFP8 wgrad with A reused across a unit's N tiles, CLC or atomic-counter scheduling."""
        if cutlass.const_expr(self.launch_pdl_dependents_at_entry):
            cute.arch.griddepcontrol_launch_dependents()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        is_leader_cta = block_in_cluster_coord_vmnk[0] == 0
        tidx, _, _ = cute.arch.thread_idx()

        # ======== SharedStorage ========
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_sched_stages, use_dynamic_sched=True)

        @cute.struct
        class SharedStorage:
            a_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_pipeline_stages * 2]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sched_storage = storage.scheduler

        # ======== Pipelines ========

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        a_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_full_mbar_ptr.data_ptr(),
            num_stages=self.num_a_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * 32 * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Scheduler pipeline (sched warp -> tma/mma/epi warps)
        sched_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        num_sched_consumer_threads = 32 * len((self.tma_warp_id, self.mma_warp_id, *self.epilogue_warp_id))
        sched_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_sched_consumer_threads)
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_sched_stages,
            producer_group=sched_producer_group,
            consumer_group=sched_consumer_group,
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            defer_sync=True,
        )

        # TMEM allocator
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.architecture,
        )

        # Scheduler (CLC or claim counter for 2Dx2D)
        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            offs,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr) if cutlass.const_expr(self.atomic_counter_claim) else None,
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        # Cluster barrier sync after init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ======== SMEM tensors ========
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        if cutlass.const_expr(self.overlapping_accum):
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )

        # Scheduler buf tensor for sched_pipeline broadcast
        sched_buf_ptr = sched_storage.sInfo.data_ptr()
        sched_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)
        sched_buf_tensor = cute.make_tensor(
            sched_buf_ptr,
            cute.make_layout((4, self.num_sched_stages), stride=(1, 4)),
        )

        # Descriptor slots: an expert's partial last K tile reads A and B through its own descriptors.
        from ..moe_utils import TensormapWorkspace, WgradSfTensormapConstructor, acquire_tma_desc, rewrite_tensor_shape, tensormap_ptr_for_copy

        desc_workspace = TensormapWorkspace(workspace_ptr, WgradSfTensormapConstructor.slot_names(self.input_order, self.weight_mode))

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # use_pdl: the setup above touches no global memory; the kernel before this one on the stream (the counter
        # reset, or the producer of A, B, the scale factors or the offsets) may still be running, so every thread
        # waits for it here.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # TMEM: epilogue warp 0 allocates, and the MMA and epilogue warps wait at this one site: the allocation
        # barrier is aligned, so every warp must reach it at the same instruction
        tmem.allocate(self.num_tmem_alloc_cols)
        if warp_idx <= self.mma_warp_id:
            tmem.wait_for_alloc()

        # ======== Scheduler warp (warp 6) ========
        if warp_idx == self.sched_warp_id:
            sched_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_sched_stages)

            work_tile_info = scheduler.initial_work_tile_info()

            # A CLC grid's first tile is always valid; an atomic-counter claim may already be past the last tile.
            if cutlass.const_expr(not self.atomic_counter_claim):
                sched_pipeline.producer_acquire(sched_producer_state)
                cute.copy(
                    sched_copy_atom,
                    work_tile_info.to_rmem_tensor(),
                    sched_buf_tensor[(None, sched_producer_state.index)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()

                work_tile_info = scheduler.advance_to_next_work()
            while work_tile_info.is_valid_tile:
                sched_pipeline.producer_acquire(sched_producer_state)
                rmem = work_tile_info.to_rmem_tensor()
                cute.copy(
                    sched_copy_atom,
                    rmem,
                    sched_buf_tensor[(None, sched_producer_state.index)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()

                work_tile_info = scheduler.advance_to_next_work()

            sched_pipeline.producer_acquire(sched_producer_state)
            sentinel = MoEWorkTileInfo(
                cutlass.Int32(-1),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            )
            rmem = sentinel.to_rmem_tensor()
            cute.copy(
                sched_copy_atom,
                rmem,
                sched_buf_tensor[(None, sched_producer_state.index)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            # producer_tail takes the NEXT buffer's state: left on the sentinel's, its first wait reuses a parity the
            # barrier is already a phase past once consumers release the sentinel, and never returns.
            sched_producer_state.advance()
            sched_pipeline.producer_tail(sched_producer_state)

        # ======== TMA load warp (warp 5) ========
        if warp_idx == self.tma_warp_id:
            a_full_mcast_mask = None
            b_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            # SF in shared memory as 16-byte rows: SFA one 512-byte atom per stage, SFB two (its 256 columns).
            sSFA_rows = cute.make_tensor(cute.recast_ptr(sSFA.iterator, dtype=cutlass.Int64), cute.make_layout((2, 32, self.num_a_stage), stride=(1, 2, 64)))
            sSFB_rows = cute.make_tensor(
                cute.recast_ptr(sSFB.iterator, dtype=cutlass.Int64), cute.make_layout((2, 32, 2, self.num_b_stage), stride=(1, 2, 64, 128))
            )
            hidden = cute.size(a_gemm, mode=[0])
            intermediate = cute.size(b_gemm, mode=[0])
            n_tiles_total = intermediate // self.cta_tile_shape_mnk[1]
            n_units = (n_tiles_total + self.n_group - 1) // self.n_group
            expert_cnt = cute.size(offs, mode=[0])
            a_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_a_stage)
            b_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_b_stage)
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            sA_grouped = cute.group_modes(sA, 0, 3)
            sB_grouped = cute.group_modes(sB, 0, 3)

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
                gA_mkl = cute.local_tile(cute.domain_offset((0, token_offset, 0), mA_mkl), cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(cute.domain_offset((0, token_offset, 0), mB_nkl), cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout, sA_grouped, cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3)
                )
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout, sB_grouped, cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3)
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

                mma_tile_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                tAgA_own_slice = tAgA_own[(None, mma_tile_m, None, 0)]

                # SF: the expert's atoms start at byte rows * token_offset / 32 (the stock kernel's per-expert SF
                # descriptor base), k_tile_cnt 512-byte atoms per 128 rows (or columns) of A (B).
                sfa_first_row = cutlass.Int32((cutlass.Int64(hidden) * cutlass.Int64(token_offset)) // (self.sf_vec_size * 16)) + (
                    mma_tile_m * cute.size(tiled_mma.thr_id.shape) + mma_tile_coord_v
                ) * (k_tile_cnt * 32)
                sfb_expert_row = cutlass.Int32((cutlass.Int64(intermediate) * cutlass.Int64(token_offset)) // (self.sf_vec_size * 16))

                prefetch_linear_unit = expert_idx * n_units + work_tile_info.tile_n_idx + self.L2_PREFETCH_UNITS
                prefetch_expert = prefetch_linear_unit // n_units
                prefetch_unit = prefetch_linear_unit % n_units
                if prefetch_expert < expert_cnt:
                    prefetch_token_offset, prefetch_tokens = compute_expert_token_range(offs, prefetch_expert)
                    prefetch_k_tiles = prefetch_tokens // self.cta_tile_shape_mnk[2]
                    if mma_tile_m == 0:
                        prefetch_gB = cute.local_tile(
                            cute.domain_offset((0, prefetch_token_offset, 0), mB_nkl),
                            cute.slice_(self.mma_tiler, (0, None, None)),
                            (None, None, None),
                        )
                        _, prefetch_tBgB = cpasync.tma_partition(
                            tma_atom_b,
                            block_in_cluster_coord_vmnk[1],
                            b_cta_layout,
                            sB_grouped,
                            cute.group_modes(thr_mma.partition_B(prefetch_gB), 0, 3),
                        )
                        prefetch_first_n_tile = prefetch_unit * self.n_group
                        prefetch_n_tiles = cutlass.min(self.n_group, n_tiles_total - prefetch_first_n_tile)
                        for prefetch_tile in cutlass.range(0, prefetch_n_tiles, 1, unroll=1):
                            for prefetch_k_tile in cutlass.range(0, prefetch_k_tiles, 1, unroll=1):
                                cute.prefetch(tma_atom_b, prefetch_tBgB[(None, prefetch_first_n_tile + prefetch_tile, prefetch_k_tile, 0)])
                    if prefetch_unit == 0:
                        prefetch_gA = cute.local_tile(
                            cute.domain_offset((0, prefetch_token_offset, 0), mA_mkl),
                            cute.slice_(self.mma_tiler, (None, 0, None)),
                            (None, None, None),
                        )
                        _, prefetch_tAgA = cpasync.tma_partition(
                            tma_atom_a,
                            block_in_cluster_coord_vmnk[2],
                            a_cta_layout,
                            sA_grouped,
                            cute.group_modes(thr_mma.partition_A(prefetch_gA), 0, 3),
                        )
                        for prefetch_k_tile in cutlass.range(0, prefetch_k_tiles, 1, unroll=1):
                            cute.prefetch(tma_atom_a, prefetch_tAgA[(None, mma_tile_m, prefetch_k_tile, 0)])

                stream_a = k_tile_cnt > self.num_a_stage
                first_n_tile = work_tile_info.tile_n_idx * self.n_group
                unit_tiles = cutlass.min(self.n_group, n_tiles_total - first_n_tile)
                for unit_tile in cutlass.range(0, unit_tiles, 1, unroll=1):
                    n_idx = first_n_tile + unit_tile
                    tBgB_slice = tBgB[(None, n_idx, None, 0)]
                    tBgB_own_slice = tBgB_own[(None, n_idx, None, 0)]
                    sfb_first_row = sfb_expert_row + n_idx * 2 * (k_tile_cnt * 32)
                    load_a = (unit_tile == 0) | stream_a
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        partial_k_tile = has_partial_k_tile & (k_tile + 1 == k_tile_cnt)
                        if load_a:
                            a_pipeline.producer_acquire(a_producer_state)
                            a_barrier = a_pipeline.producer_get_barrier(a_producer_state)
                            if partial_k_tile:
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_own_slice[(None, k_tile)],
                                    tAsA[(None, a_producer_state.index)],
                                    tma_bar_ptr=a_barrier,
                                    tma_desc_ptr=desc_a,
                                    mcast_mask=a_full_mcast_mask,
                                )
                            else:
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, k_tile)],
                                    tAsA[(None, a_producer_state.index)],
                                    tma_bar_ptr=a_barrier,
                                    mcast_mask=a_full_mcast_mask,
                                )
                            gSFA = cute.local_tile(cute.domain_offset((0, sfa_first_row + k_tile * 32), mSFA_mkl), (2, 32), (None, None))
                            tsSFA, tgSFA = cpasync.tma_partition(
                                tma_atom_sfa,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sSFA_rows[(None, None, a_producer_state.index)], 0, 2),
                                cute.group_modes(gSFA, 0, 2),
                            )
                            cute.copy(tma_atom_sfa, tgSFA[(None, 0, 0)], tsSFA, tma_bar_ptr=a_barrier)
                            a_producer_state.advance()
                        b_pipeline.producer_acquire(b_producer_state)
                        b_barrier = b_pipeline.producer_get_barrier(b_producer_state)
                        if partial_k_tile:
                            cute.copy(
                                tma_atom_b,
                                tBgB_own_slice[(None, k_tile)],
                                tBsB[(None, b_producer_state.index)],
                                tma_bar_ptr=b_barrier,
                                tma_desc_ptr=desc_b,
                                mcast_mask=b_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, k_tile)],
                                tBsB[(None, b_producer_state.index)],
                                tma_bar_ptr=b_barrier,
                                mcast_mask=b_full_mcast_mask,
                            )
                        for sfb_half in cutlass.range_constexpr(2):
                            gSFB = cute.local_tile(
                                cute.domain_offset((0, sfb_first_row + sfb_half * (k_tile_cnt * 32) + k_tile * 32), mSFB_nkl), (2, 32), (None, None)
                            )
                            tsSFB, tgSFB = cpasync.tma_partition(
                                tma_atom_sfb,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sSFB_rows[(None, None, sfb_half, b_producer_state.index)], 0, 2),
                                cute.group_modes(gSFB, 0, 2),
                            )
                            cute.copy(tma_atom_sfb, tgSFB[(None, 0, 0)], tsSFB, tma_bar_ptr=b_barrier)
                        b_producer_state.advance()

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()
            a_pipeline.producer_tail(a_producer_state)
            b_pipeline.producer_tail(b_producer_state)

        # ======== MMA warp (warp 4) ========
        if warp_idx == self.mma_warp_id:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            sf_window_tiler_mnk = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                self.sf_window_k,
            )
            sfa_window_smem_layout = cute.slice_(
                blockscaled_utils.make_smem_layout_sfa(
                    tiled_mma,
                    sf_window_tiler_mnk,
                    self.sf_vec_size,
                    1,
                ),
                (None, None, None, 0),
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                sf_window_tiler_mnk,
                self.sf_vec_size,
                sfa_window_smem_layout,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            sfb_window_smem_layout = cute.slice_(
                blockscaled_utils.make_smem_layout_sfb(
                    tiled_mma,
                    sf_window_tiler_mnk,
                    self.sf_vec_size,
                    1,
                ),
                (None, None, None, 0),
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                sf_window_tiler_mnk,
                self.sf_vec_size,
                sfb_window_smem_layout,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)
            a_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_a_stage)
            a_release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_a_stage)
            b_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_b_stage)
            n_tiles_total = cute.size(b_gemm, mode=[0]) // self.cta_tile_shape_mnk[1]

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt

                stream_a = k_tile_cnt > self.num_a_stage
                first_n_tile = work_tile_info.tile_n_idx * self.n_group
                unit_tiles = cutlass.min(self.n_group, n_tiles_total - first_n_tile)
                a_unit_base = a_consumer_state.index
                for unit_tile in cutlass.range(0, unit_tiles, 1, unroll=1):
                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_producer_state.phase ^ 1
                    else:
                        acc_stage_index = acc_producer_state.index
                    load_a = (unit_tile == 0) | stream_a
                    release_a = (unit_tile == unit_tiles - 1) | stream_a

                    if is_leader_cta:
                        tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]
                        a_tile_base = a_unit_base
                        if load_a:
                            a_tile_base = a_consumer_state.index
                        if k_tile_cnt > 0:
                            acc_pipeline.producer_acquire(acc_producer_state)

                        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                            a_idx = (a_tile_base + k_tile) % self.num_a_stage
                            if load_a:
                                a_pipeline.consumer_wait(a_consumer_state)
                                a_consumer_state.advance()
                            b_pipeline.consumer_wait(b_consumer_state)
                            b_idx = b_consumer_state.index

                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[(None, None, None, None, a_idx)],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[(None, None, None, None, b_idx)],
                                tCtSFB_compact_s2t,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                [tCrA[(None, None, None, a_idx)], tCtSFA],
                                [tCrB[(None, None, None, b_idx)], tCtSFB],
                                tCtAcc,
                            )
                            b_pipeline.consumer_release(b_consumer_state)
                            b_consumer_state.advance()
                            if release_a:
                                a_pipeline.consumer_release(a_release_state)
                                a_release_state.advance()

                        if k_tile_cnt > 0:
                            acc_pipeline.producer_commit(acc_producer_state)
                    if k_tile_cnt > 0:
                        acc_producer_state.advance()

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        # ======== SMEM tensor C (allocated after MMA section) ========
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=c_smem_layout_staged.inner,
        )

        # ======== Epilogue warps (0 .. num_epi_warps - 1): warp w drains TMEM lane quarter w % 4 over N slice w // 4,
        # per epi_chunk_n chunk: tcgen05.ld -> BF16 (RNE) -> swizzled st.shared ring -> own TMA store of 32 x chunk.
        if warp_idx < self.mma_warp_id:
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_pipeline_stages)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )

            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)

            epi_tidx = tidx % 128
            lane_idx = tidx % 32
            lane_quarter = warp_idx % 4
            column_group = warp_idx // 4
            ring_base = warp_idx * self.c_stages_per_warp
            num_chunks = self.cta_tile_shape_mnk[1] // self.epi_chunk_n
            row_bytes = self.epi_chunk_n * (self.c_dtype.width // 8)
            buffer_bytes = 32 * row_bytes
            vectors_per_row = row_bytes // 16
            # this thread's row of its warp's first ring buffer (byte address in shared memory)
            row_address = sC.iterator.toint() + ring_base * buffer_bytes + lane_idx * row_bytes
            # 128B (64-column chunks) or 64B (32-column chunks) TMA swizzle: 16 B vector j of row r sits at
            # vector j ^ ((r * row_bytes / 128) % vectors_per_row).
            row_swizzle = ((lane_idx * row_bytes) // 128) % vectors_per_row
            r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128)
            ring_position = cutlass.Int32(0)
            n_tiles_total = cute.size(b_gemm, mode=[0]) // self.cta_tile_shape_mnk[1]

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                # Dense output: the expert's dW is slice expert_idx of the global C descriptor.
                real_c = cute.domain_offset((0, 0, work_tile_info.expert_idx), mC_mnl)
                real_c = rewrite_tensor_shape(real_c, (mC_mnl.shape[0], mC_mnl.shape[1], cutlass.Int32(1)))

                first_n_tile = work_tile_info.tile_n_idx * self.n_group
                unit_tiles = cutlass.min(self.n_group, n_tiles_total - first_n_tile)
                for unit_tile in cutlass.range(0, unit_tiles, 1, unroll=1):
                    n_idx = first_n_tile + unit_tile
                    gC_mnl = cute.local_tile(
                        real_c,
                        cute.slice_(self.mma_tiler, (None, None, 0)),
                        (None, None, None),
                    )
                    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                    tCgC = thr_mma.partition_C(gC_mnl)
                    tCgC_transformed = transform_partitioned_tensor_layout(tCgC)

                    mma_tile_coord_mnl = (
                        work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                        n_idx,
                        cutlass.Int32(0),
                    )

                    tiled_copy_t2r, tTR_tAcc_base_epi, tTR_rAcc = epilogue_tmem_copy_and_partition(
                        self,
                        epi_tidx,
                        tCtAcc_transformed,
                        tCgC_transformed,
                        epi_tile,
                        use_2cta_instrs,
                    )
                    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                    tTR_rC_vec = cute.make_tensor(tTR_rC.iterator, cute.make_layout((8, self.epi_chunk_n // 8)))

                    # (TMA, 4 lane quarters, num_chunks) boxes of this CTA tile
                    tCgC_epi = cute.flat_divide(tCgC_transformed, self.c_box)
                    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        cute.group_modes(tCgC_epi, 0, 2),
                    )
                    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

                    # Overlapping accumulator stages: stage 0 is drained from its last column chunk down, stage 1 from
                    # its first chunk up, so the columns both stages share are read first (by column_group 0).
                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_consumer_state.phase
                        reverse_chunks = acc_stage_index == 0
                    else:
                        acc_stage_index = acc_consumer_state.index
                        reverse_chunks = False

                    tTR_tAcc = tTR_tAcc_base_epi[(None, None, None, None, None, acc_stage_index)]

                    if k_tile_cnt > 0:
                        acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    for chunk_iter in cutlass.range_constexpr(self.epi_chunks_per_warp):
                        position = column_group * self.epi_chunks_per_warp + chunk_iter
                        chunk_idx = position
                        if reverse_chunks:
                            chunk_idx = num_chunks - 1 - position

                        if k_tile_cnt > 0:
                            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, chunk_idx)], tTR_rAcc)

                        if cutlass.const_expr(chunk_iter == self.epi_release_chunks - 1):
                            cute.arch.fence_view_async_tmem_load()
                            if k_tile_cnt > 0:
                                acc_pipeline.consumer_release(acc_consumer_state)
                                acc_consumer_state.advance()

                        acc_vec = cute.zeros_like(tTR_rAcc, dtype=tTR_rAcc._dtype)
                        if k_tile_cnt > 0:
                            acc_vec = tTR_rAcc.load()
                        tTR_rC.store(acc_vec.to(self.c_dtype))

                        # The ring slot is free once the TMA store issued c_stages_per_warp chunks ago has read it.
                        ring_slot = ring_position % self.c_stages_per_warp
                        cute.arch.cp_async_bulk_wait_group(self.c_stages_per_warp - 1, read=True)
                        cute.arch.sync_warp()
                        slot_address = row_address + ring_slot * buffer_bytes
                        for vector_idx in cutlass.range_constexpr(vectors_per_row):
                            dst = cute.make_tensor(
                                cute.make_ptr(
                                    self.c_dtype,
                                    slot_address + ((vector_idx ^ row_swizzle) * 16),
                                    cute.AddressSpace.smem,
                                    assumed_align=16,
                                ),
                                cute.make_layout(8),
                            )
                            cute.copy(r2s_atom, tTR_rC_vec[(None, vector_idx)], dst)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.sync_warp()
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, ring_base + ring_slot)],
                            bSG_gC[(None, lane_quarter, chunk_idx)],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        ring_position += cutlass.Int32(1)

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            cute.arch.cp_async_bulk_wait_group(0, read=True)

            tmem.relinquish_alloc_permit()
            epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
