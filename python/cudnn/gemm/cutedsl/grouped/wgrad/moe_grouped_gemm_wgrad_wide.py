# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MoE BF16 grouped GEMM wgrad (2Dx2D), dense output and Tensor2D inputs, "wide" tiles.

Each 2-CTA pair computes a 256 x 512 tile of dW[e] as two 256 x 256 MMA halves that share every A k-tile, so A is
loaded once per 512 columns instead of once per 256: 25% fewer operand bytes per FLOP than
``MoEGroupedGemmWgradBF16Kernel`` at 256x256, with the same dW bits (the same 256x256 MMA instructions accumulate each
output element over the same k-tiles in the same order). Under cluster launch control the clusters are 8 x 2 CTAs
where a GPC has room (A multicast along N, B along M) and (2, 1) elsewhere; with the atomic-counter claim, (2, 1).
"""

from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, TmemAllocator, get_smem_capacity_in_bytes
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import AddressSpace
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
)
from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import compute_expert_token_range


class MoEGroupedGemmWgradWideBF16Kernel:
    """BF16 MoE wgrad kernel (2Dx2D), dense output, Tensor2D input, 2-CTA 256 x 512 tiles, FP32 accumulation.

    :param expert_cnt: Number of experts.
    :param atomic_counter_claim: Launch persistent (2, 1) clusters that claim tiles from a 4-byte int32 counter at the
        start of the workspace, reset by a one-thread kernel before every launch, instead of a grid of every tile
        rebalanced by cluster launch control (CLC, the default).
    :param use_pdl: Launch as the programmatic dependent of the kernel before it on the stream (the counter reset
        kernel is launched the same way); every thread waits for that grid before its first global memory access.
    :param launch_pdl_dependents_at_entry: Let this kernel's own programmatic dependent launch at its entry.
    """

    N_HALVES = 2  # 256-column MMA tiles per work tile

    @staticmethod
    def can_implement(m: int, n: int) -> bool:
        """dW (m, n) in whole 256-row M tiles and 512-column work units."""
        return m > 0 and n > 0 and m % 256 == 0 and n % 512 == 0

    def __init__(
        self,
        expert_cnt: int,
        atomic_counter_claim: bool = False,
        use_pdl: bool = False,
        launch_pdl_dependents_at_entry: bool = False,
    ):
        self.expert_cnt = expert_cnt
        self.acc_dtype = cutlass.Float32
        self.mma_inst_mn = (256, 256)
        self.cta_group = tcgen05.CtaGroup.TWO
        self.num_c_stage = 1  # smem C buffers per epilogue warp
        self.epi_n = 32  # columns per TMEM -> register subtile
        self.box_n = 64  # columns per TMA store box (box = 32 rows x box_n, i.e. full 128 B lines at 64)
        self.cluster_m = 8
        self.atomic_counter_claim = atomic_counter_claim
        self.use_pdl = use_pdl
        self.launch_pdl_dependents_at_entry = launch_pdl_dependents_at_entry

        self.occupancy = 1
        self.epilogue_warp_id = tuple(range(8))
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.padding_warp_id = 11
        self.threads_per_cta = 32 * 12

        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2

        self.smem_capacity = get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.num_sched_stages = 2

    def get_desc_workspace_bytes(self) -> int:
        return 0  # no per-expert descriptors: the counter is the workspace's first word

    def get_workspace_bytes(self) -> int:
        return 4 if self.atomic_counter_claim else 0

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        return cute.make_ptr(
            cutlass.Int32,
            workspace_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=4,
        )

    # ------------------------------------------------------------------

    def _create_tiled_mma(self):
        return sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_mn,
        )

    def _setup_attributes(self) -> None:
        tiled_mma = self._create_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_inst_mn[0], self.mma_inst_mn[1], mma_inst_shape_k * 4)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # Scheduler tile: one CTA covers 128 rows x (N_HALVES x 256) columns.
        self.sched_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] * self.N_HALVES,
            self.cta_tile_shape_mnk[2],
        )
        self.epi_tile = (cute.make_layout(self.cta_tile_shape_mnk[0]), cute.make_layout(self.epi_n))
        self.epi_tile_per_warp = (cute.make_layout(32), cute.make_layout(self.box_n))
        c_tile = (cute.make_layout(self.cta_tile_shape_mnk[0]), cute.make_layout(self.box_n))

        a_smem_layout_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_smem_layout_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)
        a_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout_one)
        b_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout_one)
        # C staging: 2 warp groups x num_c_stage buffers of (128 rows x box_n) (each warp owns its 32 rows).
        self.num_c_buffers = 2 * self.num_c_stage
        c_smem_layout_one = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, c_tile, 1)
        c_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout_one) * self.num_c_buffers
        self.num_ab_stage = (self.smem_capacity - 1024 - c_bytes) // (a_bytes + self.N_HALVES * b_bytes)

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage * self.N_HALVES)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, c_tile, self.num_c_buffers)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.num_tma_load_bytes = (a_bytes + self.N_HALVES * b_bytes) * atom_thr_size

    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,  # (MA, M) M-major view of a
        mat_b: cute.Tensor,  # (M, NB)
        out: cute.Tensor,  # (E, MA, NB)
        offs: cute.Tensor,  # (E,) int32 cumulative token ends
        workspace: cute.Tensor,  # atomic_counter_claim: the counter
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        out_single_expert: Optional[cute.Tensor] = None,  # MoEGroupedGemmWgradBF16Kernel's discrete-only argument
    ):
        assert out_single_expert is None, "dense output only"
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        hidden, tokens_sum = mat_a.shape
        a_gemm = cute.make_tensor(
            mat_a.iterator,
            cute.make_layout((hidden, tokens_sum, c1), stride=(mat_a.stride[0], mat_a.stride[1], c0)),
        )
        tokens_sum_b, intermediate = mat_b.shape
        b_gemm = cute.make_tensor(
            mat_b.iterator,
            cute.make_layout((intermediate, tokens_sum_b, c1), stride=(mat_b.stride[1], mat_b.stride[0], c0)),
        )
        experts, hidden_c, intermediate_c = out.shape
        c_gemm = cute.make_tensor(
            out.iterator,
            cute.make_layout((hidden_c, intermediate_c, experts), stride=(out.stride[1], out.stride[2], out.stride[0])),
        )

        self.a_dtype = a_gemm.element_type
        self.b_dtype = b_gemm.element_type
        self.c_dtype = c_gemm.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a_gemm).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(b_gemm).mma_major_mode()
        self.c_layout = LayoutEnum.from_tensor(c_gemm)
        # Mixed clusters (CLC only): a preferred (cluster_m, cluster_n) cluster where a GPC has room, (2, 1)
        # on the rest. The atomic counter: every cluster claims the same unit, so (2, 1) only.
        self.preferred_cluster_mn, self.fallback_cluster_mn = (2, 1), None
        if cutlass.const_expr(not self.atomic_counter_claim and self.cluster_m > 2 and hidden_c % (128 * self.cluster_m) == 0):
            cluster_n = 2 if cutlass.const_expr(intermediate_c % (2 * 256 * self.N_HALVES) == 0) else 1
            self.preferred_cluster_mn, self.fallback_cluster_mn = (self.cluster_m, cluster_n), (2, 1)

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        epi_smem_layout_per_warp = cute.select(sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile_per_warp, 1), mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_gemm, epi_smem_layout_per_warp, self.epi_tile_per_warp)

        def make_variant(cluster_mn):
            cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*cluster_mn, 1)), (tiled_mma.thr_id.shape,))
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_mn, tiled_mma.thr_id)
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_mn, tiled_mma.thr_id)
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(a_op, a_gemm, a_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(b_op, b_gemm, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
            sched_params = MoESchedulerParams(
                scenario="2Dx2D",
                expert_shape=(experts, intermediate, hidden),
                cta_tile_shape_mnk=self.sched_tile_shape_mnk,
                cluster_shape_mn=cluster_mn,
                atomic_counter_claim=self.atomic_counter_claim,
            )
            return (tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, sched_params, cluster_layout_vmnk)

        preferred = make_variant(self.preferred_cluster_mn)
        fallback = make_variant(self.fallback_cluster_mn) if cutlass.const_expr(self.fallback_cluster_mn is not None) else preferred
        grid = MoESchedulerParams.get_grid_shape(preferred[4], max_active_clusters)

        # No per-expert descriptors: a helper launch only resets the atomic-counter-claim counter.
        if cutlass.const_expr(self.atomic_counter_claim):
            self.counter_reset_kernel(workspace.iterator).launch(
                grid=(1, 1, 1),
                block=(1, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=self.use_pdl,
            )

        launch_kwargs = dict(cluster=(*self.preferred_cluster_mn, 1))
        if cutlass.const_expr(self.fallback_cluster_mn is not None):
            launch_kwargs.update(fallback_cluster=(*self.fallback_cluster_mn, 1), smem_merge_branch_allocs=True)
        self.kernel(
            tiled_mma,
            *preferred,
            *fallback,
            tma_atom_c,
            tma_tensor_c,
            offs,
            workspace.iterator,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.occupancy,
            use_pdl=self.use_pdl,
            **launch_kwargs,
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

    # ------------------------------------------------------------------

    @cute.jit
    def _read_sched(self, sched_pipeline, sched_consumer_state, sched_copy_atom, sched_buf_tensor):
        sched_pipeline.consumer_wait(sched_consumer_state)
        rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
        cute.copy(sched_copy_atom, sched_buf_tensor[(None, sched_consumer_state.index)], rmem)
        work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
        cute.arch.fence_acq_rel_cta()
        # Every thread that read the slot arrives on its empty barrier, so its read is ordered before the refill.
        sched_pipeline.consumer_release(sched_consumer_state)
        # The caller advances sched_consumer_state in its own loop body (loop-carried state must be visibly
        # mutated there for the DSL to carry it across iterations).
        return work_tile_info

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        sched_params: MoESchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        tma_atom_a_fb: cute.CopyAtom,
        mA_mkl_fb: cute.Tensor,
        tma_atom_b_fb: cute.CopyAtom,
        mB_nkl_fb: cute.Tensor,
        sched_params_fb: MoESchedulerParams,
        cluster_layout_vmnk_fb: cute.Layout,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        offs: cute.Tensor,
        workspace_ptr: cute.Pointer,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        """Run the body traced for the cluster shape this CTA was launched with (preferred or fallback)."""
        if cutlass.const_expr(self.launch_pdl_dependents_at_entry):
            cute.arch.griddepcontrol_launch_dependents()
        common = (
            tma_atom_c,
            mC_mnl,
            offs,
            workspace_ptr,
            a_smem_layout_staged,
            b_smem_layout_staged,
            c_smem_layout_staged,
            epi_tile,
        )
        preferred = self.preferred_cluster_mn
        if cutlass.const_expr(self.fallback_cluster_mn is None):
            self._body(preferred, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, sched_params, cluster_layout_vmnk, *common)
        else:
            cluster_dim_m, cluster_dim_n, _ = cute.arch.block_in_cluster_dim()
            if cluster_dim_m * cluster_dim_n == preferred[0] * preferred[1]:
                self._body(preferred, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, sched_params, cluster_layout_vmnk, *common)
            else:
                self._body(
                    self.fallback_cluster_mn,
                    tiled_mma,
                    tma_atom_a_fb,
                    mA_mkl_fb,
                    tma_atom_b_fb,
                    mB_nkl_fb,
                    sched_params_fb,
                    cluster_layout_vmnk_fb,
                    *common,
                )

    @cute.jit
    def _body(
        self,
        cluster_shape_mn: cutlass.Constexpr,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        sched_params: MoESchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        offs: cute.Tensor,
        workspace_ptr: cute.Pointer,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        # TMA multicast fan-out: A is shared along N work units and B along M CTA pairs.
        num_mcast_ctas_a = cluster_shape_mn[1]
        num_mcast_ctas_b = cluster_shape_mn[0] // cute.size(tiled_mma.thr_id.shape)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_sched_stages, use_dynamic_sched=True)

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sched_storage = storage.scheduler

        # ---- pipelines ----
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            # the MMA's release of a stage must reach every CTA that multicast into it
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_mcast_ctas_a + num_mcast_ctas_b - 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # acc stage 0 = h0 (TMEM columns 0..255), stage 1 = h1 (256..511); every work tile uses both, in order.
        # one elected arrive per epilogue warp of each CTA of the pair
        num_acc_consumer_threads = len(self.epilogue_warp_id) * cute.size(tiled_mma.thr_id.shape)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        num_sched_consumer_threads = 32 * len((self.tma_warp_id, self.mma_warp_id, *self.epilogue_warp_id))
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_sched_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_sched_consumer_threads),
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            offs,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr) if cutlass.const_expr(self.atomic_counter_claim) else None,
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

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
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))

        sched_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)
        sched_buf_tensor = cute.make_tensor(
            sched_storage.sInfo.data_ptr(),
            cute.make_layout((4, self.num_sched_stages), stride=(1, 4)),
        )

        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)
        # use_pdl: the setup above touches no global memory; the kernel before this one on the stream (the counter
        # reset, or the producer of A, B, the offsets or the output) may still be running, so every thread waits here.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # TMEM: epilogue warp 0 allocates, and the MMA and epilogue warps wait at this one site: the allocation
        # barrier is aligned, so every warp must reach it at the same instruction
        tmem.allocate(self.num_tmem_alloc_cols)
        if warp_idx <= self.mma_warp_id:
            tmem.wait_for_alloc()

        if warp_idx == self.padding_warp_id:
            cute.arch.setmaxregister_decrease(40)

        # ===== Scheduler warp =====
        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(40)
            sched_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_sched_stages)
            work_tile_info = scheduler.initial_work_tile_info()
            while work_tile_info.is_valid_tile:
                sched_pipeline.producer_acquire(sched_producer_state)
                tile_info = work_tile_info
                cute.copy(sched_copy_atom, tile_info.to_rmem_tensor(), sched_buf_tensor[(None, sched_producer_state.index)])
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()
                work_tile_info = scheduler.advance_to_next_work()

            sched_pipeline.producer_acquire(sched_producer_state)
            sentinel = MoEWorkTileInfo(cutlass.Int32(-1), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))
            cute.copy(sched_copy_atom, sentinel.to_rmem_tensor(), sched_buf_tensor[(None, sched_producer_state.index)])
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            sched_producer_state.advance()
            sched_pipeline.producer_tail(sched_producer_state)

        # ===== TMA load warp =====
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(40)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            a_mcast_mask = None
            b_mcast_mask = None
            if cutlass.const_expr(num_mcast_ctas_a > 1):
                a_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            if cutlass.const_expr(num_mcast_ctas_b > 1):
                b_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

            sA_grouped = cute.group_modes(sA, 0, 3)
            sB_grouped = cute.group_modes(sB, 0, 3)

            work_tile_info = self._read_sched(sched_pipeline, sched_consumer_state, sched_copy_atom, sched_buf_tensor)
            sched_consumer_state.advance()
            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                # The expert's tokens start at token_offset on the global descriptors (any offset, as the stock
                # kernel reads them); k_tile_cnt bounds the K tiles read.
                token_offset, _ = compute_expert_token_range(offs, work_tile_info.expert_idx)
                gA_mkl = cute.local_tile(cute.domain_offset((0, token_offset, 0), mA_mkl), cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(cute.domain_offset((0, token_offset, 0), mB_nkl), cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    sA_grouped,
                    cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3),
                )
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    sB_grouped,
                    cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3),
                )
                mma_tile_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                tBgB_half0 = tBgB[(None, work_tile_info.tile_n_idx * self.N_HALVES, None, 0)]
                tBgB_half1 = tBgB[(None, work_tile_info.tile_n_idx * self.N_HALVES + 1, None, 0)]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                    cute.copy(
                        tma_atom_b,
                        tBgB_half0[(None, handle.count)],
                        tBsB[(None, handle.index * self.N_HALVES)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_half1[(None, handle.count)],
                        tBsB[(None, handle.index * self.N_HALVES + 1)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_mcast_mask,
                    )
                work_tile_info = self._read_sched(sched_pipeline, sched_consumer_state, sched_copy_atom, sched_buf_tensor)
                sched_consumer_state.advance()
            ab_producer.tail()

        # ===== MMA warp =====
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(40)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_half0 = tCtAcc_base[(None, None, None, 0)]
            tCtAcc_half1 = tCtAcc_base[(None, None, None, 1)]

            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            work_tile_info = self._read_sched(sched_pipeline, sched_consumer_state, sched_copy_atom, sched_buf_tensor)
            sched_consumer_state.advance()
            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                if is_leader_cta:
                    acc_state_half1 = acc_producer_state.clone()
                    acc_state_half1.advance()
                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        peek_ab_full_status = ab_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                        a_coord = (None, None, None, handle.index)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        cute.gemm(tiled_mma, tCtAcc_half0, tCrA[a_coord], tCrB[(None, None, None, handle.index * self.N_HALVES)], tCtAcc_half0)
                        if k_tile == k_tile_cnt - 1:
                            acc_pipeline.producer_commit(acc_producer_state)
                        if k_tile == 0:
                            acc_pipeline.producer_acquire(acc_state_half1)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        cute.gemm(tiled_mma, tCtAcc_half1, tCrA[a_coord], tCrB[(None, None, None, handle.index * self.N_HALVES + 1)], tCtAcc_half1)
                        handle.release()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_state_half1)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()
                    acc_producer_state.advance()
                work_tile_info = self._read_sched(sched_pipeline, sched_consumer_state, sched_copy_atom, sched_buf_tensor)
                sched_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ===== Epilogue warps (0..7): each drains one row subpartition and half the columns of both halves =====
        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(232)
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)
            epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
            subpartition = warp_idx % 4
            column_group = warp_idx // 4
            lane_idx = tidx % 32
            epi_tidx = tidx % 128

            # TMEM -> RF partition (a 4-warp (128, epi_n) epilogue tile; warps 4..7 reuse it on other columns)
            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)
            gC_any = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
            tCgC_any = transform_partitioned_tensor_layout(tiled_mma.get_slice(mma_tile_coord_v).partition_C(gC_any))
            tiled_copy_t2r, tTR_tAcc_base_epi, tTR_rAcc = epilogue_tmem_copy_and_partition(self, epi_tidx, tCtAcc_transformed, tCgC_any, epi_tile, True)
            # (T2R, T2R_M, T2R_N, SUBTILE, HALF)
            tTR_tAcc_all = cute.group_modes(tTR_tAcc_base_epi, 3, 5)
            subtiles_per_half = cute.size(tTR_tAcc_all.shape, mode=[3])
            subtiles_per_warp = subtiles_per_half // 2
            # RF -> SMEM: each (128, epi_n) subtile goes into one column slice of a (128, box_n) staging buffer
            tiled_copy_r2s = cute.make_tiled_copy_D(sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r), tiled_copy_t2r)
            # (R2S, R2S_M, R2S_N, 1, box_n / epi_n, C_BUF)
            tRS_sC = tiled_copy_r2s.get_slice(epi_tidx).partition_D(cute.flat_divide(sC, (self.cta_tile_shape_mnk[0], self.epi_n)))
            subtiles_per_box = self.box_n // self.epi_n

            rAcc = [cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)]
            # BF16 results of this warp's (half, subtile) pieces, held until stored
            rC = [cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype) for _ in range(self.N_HALVES * subtiles_per_warp)]

            # per-warp TMA store: this warp's 32 rows of its own C buffers
            epi_tile_per_warp = (32, self.box_n)
            sC_per_warp = cute.flat_divide(sC, epi_tile_per_warp)[(None, None, subpartition, 0, None)]
            gC_per_warp = cute.local_tile(mC_mnl, epi_tile_per_warp, (None, None, None))
            bSG_sC, bSG_gC = cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                cute.group_modes(sC_per_warp, 0, 2),
                cute.group_modes(gC_per_warp, 0, 2),
            )
            n_blocks_per_half = self.cta_tile_shape_mnk[1] // self.box_n
            boxes_per_warp_half = subtiles_per_warp // subtiles_per_box
            m_blocks_per_tile = self.cta_tile_shape_mnk[0] // 32
            c_buffer_base = column_group * self.num_c_stage

            work_tile_info = self._read_sched(sched_pipeline, sched_consumer_state, sched_copy_atom, sched_buf_tensor)
            sched_consumer_state.advance()
            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                has_acc = k_tile_cnt > 0

                # Drain both halves from TMEM into registers (BF16), releasing each half right away.
                for half in cutlass.range_constexpr(self.N_HALVES):
                    tTR_tAcc = tTR_tAcc_all[(None, None, None, None, half)]
                    first_subtile = column_group * subtiles_per_warp
                    if has_acc:
                        acc_pipeline.consumer_wait(acc_consumer_state)
                    for j in cutlass.range_constexpr(subtiles_per_warp):
                        if has_acc:
                            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, first_subtile + j)], rAcc[0])
                        else:
                            rAcc[0].fill(0.0)
                        rC[half * subtiles_per_warp + j].store(rAcc[0].load().to(self.c_dtype))
                    if has_acc:
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.sync_warp()
                        if lane_idx == 0:
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                # next work tile (its mainloop runs while these stores drain)
                next_tile_info = self._read_sched(sched_pipeline, sched_consumer_state, sched_copy_atom, sched_buf_tensor)
                sched_consumer_state.advance()

                # ---- store: registers -> this warp's smem buffer -> TMA store, one (32, epi_n) box at a time ----
                m_block = work_tile_info.tile_m_idx * m_blocks_per_tile + subpartition
                bSG_gC_tile = bSG_gC[(None, m_block, None, work_tile_info.expert_idx)]
                first_n_block = work_tile_info.tile_n_idx * (n_blocks_per_half * self.N_HALVES) + column_group * boxes_per_warp_half
                num_boxes = self.N_HALVES * boxes_per_warp_half
                # Pace stores from the next tile's TMA-load bytes (128-byte L2 sectors over this warp's store
                # boxes), not a tuned constant; the one-stage wait-group caps each warp at one 4 KiB store.
                pace_period = cutlass.Int64(0)
                if next_tile_info.is_valid_tile:
                    load_cycles_per_k = self.num_tma_load_bytes // 128
                    tile_flops_per_k = self.mma_inst_mn[0] * self.mma_inst_mn[1] * self.N_HALVES * self.mma_tiler[2] * 2
                    # SM103 BF16 tcgen05 peak is 8192 FLOP/SM/cycle;
                    # CtaGroup.TWO uses a two-SM pair.
                    mma_cycles_per_k = tile_flops_per_k // (2 * 8192)
                    tile_cycles_per_k = max(load_cycles_per_k, mma_cycles_per_k)
                    pace_period = cutlass.Int64(next_tile_info.k_tile_cnt * (tile_cycles_per_k // num_boxes))
                next_issue_cycle = cute.arch.clock64()
                for box in cutlass.range_constexpr(num_boxes):
                    half = box // boxes_per_warp_half
                    box_in_half = box % boxes_per_warp_half
                    c_buffer = c_buffer_base + (box % self.num_c_stage)
                    if lane_idx == 0:
                        cute.arch.cp_async_bulk_wait_group(self.num_c_stage - 1, read=True)
                    cute.arch.sync_warp()
                    for subtile_in_box in cutlass.range_constexpr(subtiles_per_box):
                        cute.copy(
                            tiled_copy_r2s,
                            tiled_copy_r2s.retile(rC[half * subtiles_per_warp + box_in_half * subtiles_per_box + subtile_in_box]),
                            tRS_sC[(None, None, None, 0, subtile_in_box, c_buffer)],
                        )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                    if lane_idx == 0:
                        if pace_period > 0:
                            while cute.arch.clock64() < next_issue_cycle:
                                pass
                            next_issue_cycle = cute.arch.clock64() + pace_period
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC_tile[(None, first_n_block + half * n_blocks_per_half + box_in_half)],
                        )
                        cute.arch.cp_async_bulk_commit_group()

                work_tile_info = next_tile_info

            if lane_idx == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            tmem.relinquish_alloc_permit()
            epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

        # Cluster-exit barrier: each CTA releases the last (invalid) CLC response with a remote arrive on the leader's
        # mbarrier; the 2-CTA TMEM dealloc holds only the leader's own pair, so larger clusters need this barrier.
        if cutlass.const_expr(cluster_shape_mn[0] * cluster_shape_mn[1] > 2):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
