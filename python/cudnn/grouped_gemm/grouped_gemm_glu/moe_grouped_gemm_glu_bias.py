# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MoE BF16 Grouped GEMM Kernel with GLU (SwiGLU/GeGLU) Fusion.

Supports:
    - Static / Dynamic persistent tile scheduling (MoEPersistentTileScheduler)
    - Dense (contiguous 3-D B) / Discrete (per-expert pointer array B) weight layout
    - Optional bias
    - GLU activation fusion (SwiGLU / GeGLU)

This module contains only the kernel class.
MoE scheduler components live in moe_persistent_scheduler.py / moe_sched_extension.py / moe_utils.py.
"""

from typing import Type, Tuple, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, AddressSpace
from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import (
    MoEWeightMode,
    TensormapWorkspace,
    store_tma_desc,
)
from ..moe_sched_extension import (
    DiscreteWeightGroupedGemmSchedExtension,
    ContiguousGroupedGemmSchedExtension,
)
from ..moe_kernel_helpers import (
    fmin,
    fmax,
    silu_f32,
    silu_f32_geglu_scaled,
    can_implement_bf16_grouped_gemm,
    compute_grid,
    epilog_gmem_copy_and_partition,
)


class MoEGroupedGemmGluBiasBf16Kernel:
    """Plain BF16 grouped GEMM kernel with MoE scheduling and GLU fusion.

    The kernel is organized as persistent scheduler, A/B TMA load, MMA, and
    epilogue warps. The epilogue applies SwiGLU or GeGLU and optionally adds
    bias before writing D.

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - FIX_PAD_SIZE (256) must be divisible by mma_tiler_mn[0]
        - m_aligned parameter in create_mask() MUST equal FIX_PAD_SIZE (256)
        - Each padded_offsets[i] will be a multiple of FIX_PAD_SIZE (guaranteed by m_aligned == FIX_PAD_SIZE)
    """

    # Fixed pad size for user-side padding (decoupled from kernel tile size)
    FIX_PAD_SIZE = 256

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        cd_major: str,
        m_aligned: int,
    ) -> bool:
        return can_implement_bf16_grouped_gemm(
            ab_dtype,
            c_dtype,
            d_dtype,
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            cd_major,
            m_aligned,
            fix_pad_size=MoEGroupedGemmGluBiasBf16Kernel.FIX_PAD_SIZE,
        )

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        expert_cnt: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DISCRETE,
        use_dynamic_sched: bool = False,
        act_func: str = "swiglu",
        enable_bias: bool = False,
        generate_c: bool = False,
    ):
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % mma_tile_m != 0:
            raise ValueError(f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by " f"mma_tiler_mn[0] ({mma_tile_m}).")
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")
        if act_func not in ["swiglu", "geglu"]:
            raise ValueError(f"Invalid activation function: {act_func}")

        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.enable_bias = enable_bias
        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.bias_load_warp_id = 7 if enable_bias else None
        self.threads_per_warp = 32

        all_warps = [
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.tma_warp_id,
            self.sched_warp_id,
        ]
        warps_wo_sched = [
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.tma_warp_id,
        ]
        if enable_bias:
            all_warps.append(self.bias_load_warp_id)
            warps_wo_sched.append(self.bias_load_warp_id)
        self.threads_per_cta = self.threads_per_warp * len(all_warps)
        self.threads_wo_sched = self.threads_per_warp * len(warps_wo_sched)

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.vectorized_f32 = vectorized_f32
        self.generate_c = generate_c

        self.weight_mode = weight_mode
        self.use_dynamic_sched = use_dynamic_sched

        self.act_func = act_func

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/D stage counts in shared memory
        - Computing A/B/D shared memory layout
        """

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.mma_tiler_d = (
            self.mma_tiler[0],
            self.mma_tiler[1] // 2,
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_d = (
            self.mma_tiler_d[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_d[1],
            self.mma_tiler_d[2],
        )
        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Set epilogue subtile
        self.epi_tile = (128, 32)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_d[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_d[1] // self.epi_tile[1],
        )
        self.epi_tile_c = (128, 64)

        # Setup A/B/C/D stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_d_stage,
            self.num_tile_stage,
            self.num_bias_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.epi_tile_c,
            self.c_dtype,
            self.c_layout,
            self.d_dtype,
            self.d_layout,
            self.num_smem_capacity,
            self.occupancy,
            bias_dtype=self.bias_dtype if self.enable_bias else None,
        )

        # TMEM accumulator columns: derive from the actual accumulator fragment and
        # round up to a valid power-of-two allocation (handles non-power-of-two tile N),
        # matching the plain BF16 grouped GEMM kernel instead of always reserving 512.
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        # Compute A/B/C/D shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )

        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile_c,
            self.num_c_stage,
        )

        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_dtype,
            self.d_layout,
            self.epi_tile,
            self.num_d_stage,
        )

        # Bias SMEM layout: (tile_N, num_stages) double-buffered
        if self.enable_bias:
            self.bias_smem_layout_staged = cute.make_layout(
                (self.mma_tiler[1], self.num_bias_stage),
                stride=(1, self.mma_tiler[1]),
            )
        else:
            self.bias_smem_layout_staged = cute.make_layout((1, 1))

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        epi_tile_c,
        c_dtype,
        c_layout,
        d_dtype,
        d_layout,
        num_smem_capacity,
        occupancy,
        bias_dtype,
    ):
        """Compute BF16-only pipeline stages.

        Stage counts are chosen for the BF16 A/B mainloop and the GLU epilogue
        shared-memory footprint.
        """
        num_acc_stage = 2
        num_c_stage = 1
        num_d_stage = 1
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_stage_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        c_smem_layout_stage_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile_c, 1)
        d_smem_layout_stage_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_stage_one)
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        c_bytes = cute.size_in_bytes(c_dtype, c_smem_layout_stage_one) * num_c_stage
        d_bytes = cute.size_in_bytes(d_dtype, d_smem_layout_stage_one) * num_d_stage

        if bias_dtype is not None:
            num_bias_stage = 2
            bias_bytes = mma_tiler_mnk[1] * num_bias_stage * (bias_dtype.width // 8)
        else:
            num_bias_stage = 0
            bias_bytes = 0

        epi_bytes = c_bytes + d_bytes + bias_bytes
        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage
        return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage, num_bias_stage

    def get_desc_workspace_bytes(self) -> int:
        """Return descriptor workspace size in bytes."""
        if self.weight_mode == MoEWeightMode.DISCRETE:
            return TensormapWorkspace.size_bytes(1, self.expert_cnt)
        return 0

    def get_workspace_bytes(self) -> int:
        """Return descriptor workspace plus optional dynamic scheduler state."""
        desc_workspace_bytes = self.get_desc_workspace_bytes()
        dynamic_sched_bytes = 4 if self.use_dynamic_sched else 0
        return desc_workspace_bytes + dynamic_sched_bytes

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        counter_addr = workspace_ptr.toint() + self.get_desc_workspace_bytes()
        return cute.make_ptr(
            cutlass.Int32,
            counter_addr,
            AddressSpace.gmem,
            assumed_align=4,
        )

    @cute.kernel
    def helper_kernel(
        self,
        ptrs_b: cute.Pointer,
        n: Int32,
        k: Int32,
        b_stride_size: cutlass.Int64,
        b_major_mode: cutlass.Constexpr,
        workspace_ptr,
        tiled_mma_arg: cute.TiledMma,
        b_smem_layout_arg,
        cluster_layout_vmnk_shape_arg: cutlass.Constexpr,
    ):
        """Pre-main-kernel initialization.

        Launched with grid=(expert_cnt, 1, 1) for discrete mode, or
        grid=(1, 1, 1) for dense+dynamic mode.

        Discrete weight: each block builds a B TMA descriptor for one expert.
        Dynamic sched: block 0 resets the atomic tile counter to 0.
        """
        expert_idx = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            # Discrete mode stores one B TMA descriptor per expert in the
            # descriptor workspace. The main kernel later reuses those
            # descriptors through the scheduler extension.
            b_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma_arg.thr_id)

            b_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_b.toint(), AddressSpace.gmem, assumed_align=8), cute.make_layout((self.expert_cnt,))
            )

            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                stride_n = b_stride_size
                stride_k = c1_64
            else:
                stride_n = c1_64
                stride_k = b_stride_size

            b_ptr_val = b_ptr_tensor[expert_idx]
            b_ptr = cute.make_ptr(self.b_dtype, b_ptr_val, AddressSpace.gmem)
            b_tensor_i = cute.make_tensor(
                b_ptr,
                cute.make_layout((n, k, c1), stride=(stride_n, stride_k, c0)),
            )
            tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_tma_op_arg,
                b_tensor_i,
                b_smem_layout_arg,
                self.mma_tiler,
                tiled_mma_arg,
                cluster_layout_vmnk_shape_arg,
            )

            workspace = TensormapWorkspace(workspace_ptr, ["b"])
            store_tma_desc(tma_atom_b, workspace.get_ptr("b", expert_idx))

        if cutlass.const_expr(self.use_dynamic_sched):
            if expert_idx == cutlass.Int32(0):
                sched_counter = cute.make_tensor(
                    self._get_sched_counter_ptr(workspace_ptr),
                    cute.make_layout(1),
                )
                sched_counter[0] = cutlass.Int32(0)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b,  # Dense: cute.Tensor (N,K,L) | Discrete: cute.Pointer to int64[]
        n: Int32,  # Ignored for dense mode
        k: Int32,  # Ignored for dense mode
        b_stride_size: cutlass.Int64,  # Ignored for dense mode
        b_major_mode: cutlass.Constexpr,  # Ignored for dense mode
        workspace_ptr,
        c: cute.Tensor,
        d: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        prob: cute.Tensor,
        bias: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        linear_offset: cutlass.Float32 = 0.0,
    ):
        """Execute the GEMM.

        Dense mode: ``b`` is a 3-D cute.Tensor (N, K, L).
        Discrete mode: ``b`` is a cute.Pointer to a device int64[] array of
        per-expert base addresses; ``n``, ``k``, ``b_stride_size``, and
        ``b_major_mode`` describe the uniform per-expert layout.
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.bias_dtype = bias.element_type if cutlass.const_expr(self.enable_bias) else cutlass.BFloat16
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.d_layout = utils.LayoutEnum.from_tensor(d)

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        else:
            self.b_major_mode = b_major_mode

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # ---- B setup (mode-dependent) ----
        # Dense mode receives a normal (N,K,L) tensor. Discrete mode receives a
        # device array of per-expert B base addresses; we build a template tensor
        # here only so the TMA atom has the right dtype/layout at compile time.
        b_from_call_arg = b
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                b_template_stride = (b_stride_size, c1_64, c0)
            else:
                b_template_stride = (c1_64, b_stride_size, c0)
            b_template_layout = cute.make_layout((n, k, c1), stride=b_template_stride)
            b_ptr_typed = cute.make_ptr(self.b_dtype, b.toint(), AddressSpace.gmem, assumed_align=16)
            b = cute.make_tensor(b_ptr_typed, b_template_layout)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            c_smem_layout,
            self.epi_tile_c,
        )

        # Setup TMA store for D
        d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d,
            d_smem_layout,
            self.epi_tile,
        )
        # ---- Helper kernel: TMA desc init (discrete) + sched counter reset (dynamic) ----
        _need_helper = cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE or self.use_dynamic_sched)
        if cutlass.const_expr(_need_helper):
            _helper_grid_x = self.expert_cnt if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else 1
            _helper_args = (
                b_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem),
                n if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                k if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                b_stride_size if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int64(0),
                b_major_mode if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else self.b_major_mode,
                workspace_ptr,
                tiled_mma,
                b_smem_layout,
                self.cluster_layout_vmnk.shape,
            )
            self.helper_kernel(*_helper_args).launch(
                grid=(_helper_grid_x, 1, 1),
                block=(1, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
            )

        # ---- Grid computation via MoE scheduler ----
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            b_n, b_k, b_l = cute.shape(b)
            sched_expert_shape = (self.expert_cnt, b_n, b_k)
        else:
            sched_expert_shape = (self.expert_cnt, n, k)

        sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=sched_expert_shape,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=self.use_dynamic_sched,
        )
        self.sched_params, grid = compute_grid(
            sched_params,
            max_active_clusters,
            self.use_2cta_instrs,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            bias_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_bias_stage * 2 if self.enable_bias else 1]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    cute.cosize(self.d_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # Bias SMEM: (tile_N, num_bias_stage) BF16 double-buffered
            sBias: cute.struct.Align[
                cute.struct.MemRange[self.bias_dtype, cute.cosize(self.bias_smem_layout_staged)],
                16,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_d,
            tma_tensor_d,
            padded_offsets,
            alpha,
            bias,
            prob,
            workspace_ptr,  # Contains per-expert B TMA descriptors
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.d_smem_layout_staged,
            self.bias_smem_layout_staged,
            self.epi_tile,
            self.sched_params,
            linear_offset,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            max_number_threads=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # ------------------------------------------------------------------
    # Internal: create extension based on weight_mode
    # ------------------------------------------------------------------

    @cute.jit
    def _make_extension(self, workspace_ptr):
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            desc_workspace = TensormapWorkspace(workspace_ptr, ["b"])
            return DiscreteWeightGroupedGemmSchedExtension(tensormap_ctor=desc_workspace)
        else:
            return ContiguousGroupedGemmSchedExtension()

    @cute.jit
    def store_c(
        self,
        tiled_copy_r2s,
        tma_atom_c,
        warp_idx,
        tTR_rAcc,
        tTR_rAcc_up,
        tRS_rC,
        tRS_sC,
        bSG_gC,
        bSG_sC,
        c_pipeline,
        prev_subtile_idx,
        real_subtile_idx,
    ) -> None:
        c_buffer = prev_subtile_idx % self.num_c_stage
        tRS_rC.store(tTR_rAcc.load().to(self.c_dtype))
        cute.copy(
            tiled_copy_r2s,
            tRS_rC[(None, None, 0)],
            tRS_sC[(None, None, 0, c_buffer)],
        )
        tRS_rC.store(tTR_rAcc_up.load().to(self.c_dtype))
        cute.copy(
            tiled_copy_r2s,
            tRS_rC[(None, None, 0)],
            tRS_sC[(None, None, 1, c_buffer)],
        )
        # Fence and barrier to make sure shared memory store is visible to TMA store
        cute.arch.fence_proxy("async.shared", space="cta")
        self.epilog_sync_barrier.arrive_and_wait()
        #
        # TMA store smem to global memory
        #
        if warp_idx == self.epilog_warp_id[0]:
            cute.copy(
                tma_atom_c,
                bSG_sC[(None, c_buffer)],
                bSG_gC[(None, real_subtile_idx)],
            )
            # Fence and barrier to make sure shared memory store is visible to TMA store
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        self.epilog_sync_barrier.arrive_and_wait()

    @cute.jit
    def geglu_act(self, tCompute: cute.Tensor, acc_vec_up: cute.Tensor, acc_vec_gate: cute.Tensor, mProb: cute.Tensor, linear_offset: cutlass.Float32 = 1.0):
        if cutlass.const_expr(self.vectorized_f32):
            # GeGlu Packed Version
            LOG2_E = cutlass.Float32(1.4426950408889634)
            for i in cutlass.range_constexpr(0, cute.size(tCompute), 2):

                scaled_gate_0, scaled_gate_1 = cute.arch.mul_packed_f32x2(
                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                    (1.702, 1.702),
                    rnd="rn",
                    ftz=False,
                )

                tCompute_log2e = cute.arch.mul_packed_f32x2(
                    (scaled_gate_0, scaled_gate_1),
                    (-LOG2_E, -LOG2_E),
                    rnd="rn",
                    ftz=False,
                )

                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.add_packed_f32x2(
                    (
                        cute.math.exp2(tCompute_log2e[0], fastmath=True),
                        cute.math.exp2(tCompute_log2e[1], fastmath=True),
                    ),
                    (1.0, 1.0),
                )

                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_gate[i + 0], acc_vec_gate[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                (
                    up_with_offset0,
                    up_with_offset1,
                ) = cute.arch.add_packed_f32x2(
                    (linear_offset, linear_offset),
                    (acc_vec_up[i + 0], acc_vec_up[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (up_with_offset0, up_with_offset1),
                    rnd="rn",
                    ftz=False,
                )
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (mProb, mProb),
                    rnd="rn",
                    ftz=False,
                )
        else:
            # GeGlu Unpacked Version
            for i in cutlass.range_constexpr(cute.size(tCompute)):
                tCompute[i] = (acc_vec_up[i] + linear_offset) * silu_f32_geglu_scaled(acc_vec_gate[i], fastmath=True)
                tCompute[i] = tCompute[i] * mProb

    @cute.jit
    def swiglu_act(self, tCompute: cute.Tensor, acc_vec_up: cute.Tensor, acc_vec_gate: cute.Tensor, mProb: cute.Tensor):
        if cutlass.const_expr(self.vectorized_f32):
            # SwiGlu Packed Version
            LOG2_E = cutlass.Float32(1.4426950408889634)
            for i in cutlass.range_constexpr(0, cute.size(tCompute), 2):
                tCompute_log2e = cute.arch.mul_packed_f32x2(
                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                    (-LOG2_E, -LOG2_E),
                    rnd="rn",
                    ftz=False,
                )
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.add_packed_f32x2(
                    (
                        cute.math.exp2(tCompute_log2e[0], fastmath=True),
                        cute.math.exp2(tCompute_log2e[1], fastmath=True),
                    ),
                    (1.0, 1.0),
                )
                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_gate[i + 0], acc_vec_gate[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_up[i], acc_vec_up[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                (
                    tCompute[i],
                    tCompute[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (mProb, mProb),
                    rnd="rn",
                    ftz=False,
                )
        else:
            # SwiGlu Unpacked Version
            for i in cutlass.range_constexpr(cute.size(tCompute)):
                tCompute[i] = acc_vec_up[i] * silu_f32(acc_vec_gate[i], fastmath=True)
                tCompute[i] = tCompute[i] * mProb

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        mBias_nl: Optional[cute.Tensor],
        prob: cute.Tensor,
        workspace_ptr,  # Pointer to TMA descriptor workspace (from desc_init_kernel)
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        bias_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
        linear_offset: cutlass.Float32 = 0.0,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.generate_c):
                cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_d)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        total_token = padded_offsets[self.expert_cnt - 1]

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            padded_offsets,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr),
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        # Bias pipeline + SMEM
        if cutlass.const_expr(self.enable_bias):
            bias_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp,
            )
            bias_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            bias_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.bias_mbar_ptr.data_ptr(),
                num_stages=self.num_bias_stage,
                producer_group=bias_pipeline_producer_group,
                consumer_group=bias_pipeline_consumer_group,
            )
            sBias = storage.sBias.get_tensor(bias_smem_layout_staged)
            # (MMA_N, loopN, loopL)
            gBias_nl = cute.local_tile(mBias_nl, cute.slice_(self.mma_tiler[:2], (0, None)), (None, None))

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/D
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/D
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        if total_token <= 0:
            cute.arch.nvvm.exit()

        #
        # Specialized Schedule warp (MoE Persistent Tile Scheduler)
        #
        if warp_idx == self.sched_warp_id:
            work_tile_info = scheduler.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)

            while work_tile_info.is_valid_tile:
                # Write MoEWorkTileInfo directly to sInfo:
                # sInfo[0] = expert_idx (>= 0 means valid)
                # sInfo[1] = tile_m_idx (CTA-level M tile index)
                # sInfo[2] = tile_n_idx
                # sInfo[3] = k_tile_cnt
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = work_tile_info.expert_idx
                    sInfo[(1, tile_info_producer_state.index)] = work_tile_info.tile_m_idx
                    sInfo[(2, tile_info_producer_state.index)] = work_tile_info.tile_n_idx
                    sInfo[(3, tile_info_producer_state.index)] = work_tile_info.k_tile_cnt
                cute.arch.fence_proxy("async.shared", space="cta")

                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()

                work_tile_info = scheduler.advance_to_next_work()

            # Send invalid tile signal: expert_idx = -1
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            ext = self._make_extension(workspace_ptr)

            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                # Get per-expert real tensors + TMA desc ptrs via extension
                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, padded_offsets, work_tile_info)
                real_b, desc_ptr_b = ext.get_gmem_tensor("b", mB_nkl, padded_offsets, work_tile_info)

                # local_tile on per-expert tensors
                gA_mkl = cute.local_tile(
                    real_a,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                gB_nkl = cute.local_tile(
                    real_b,
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (None, None, None),
                )

                # MMA partition
                thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                tCgA = thr_mma.partition_A(gA_mkl)
                tCgB = thr_mma.partition_B(gB_nkl)

                # TMA partition A
                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                # TMA partition B
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )

                # Convert CTA tile index to MMA tile index (matching original kernel's bidx // cta_group_size)
                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]

                # Peek (try_wait) AB buffer empty
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    ab_producer_state_next = ab_producer_state.clone()
                    ab_producer_state_next.advance()
                    if ab_producer_state_next.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state_next)
                    else:
                        peek_ab_empty_status = cutlass.Boolean(1)

                    # TMA load A (contiguous, global desc via domain_offset)
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    # TMA load B (discrete, per-expert desc from workspace)
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=desc_ptr_b,
                    )
                    # Peek (try_wait) AB buffer empty for next k_tile
                    ab_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acd_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                k_tile_cnt = tile_info[3]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = 0
                acd_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                # Convert CTA tile index to MMA tile index (matching original kernel's bidx // cta_group_size)
                mma_tile_coord_mnl = (
                    tile_info[1] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[2],  # tile_n_idx
                    tile_info[0],  # expert_idx
                )

                # Get accumulator stage index
                tCtAcc = tCtAcc_base[(None, None, None, acd_producer_state.index)]
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acd_producer_state, peek_acc_empty_status)
                #
                # Mma mainloop
                #

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if ab_consumer_state_next.count < k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state = ab_consumer_state_next

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acd_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acd_producer_state.advance()
                if acd_producer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acd_producer_state)

        #
        # Specialized bias load warp — cp.async 32-bit GMEM→SMEM
        #
        if cutlass.const_expr(self.enable_bias):
            if warp_idx == self.bias_load_warp_id and total_token > 0:
                bias_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_bias_stage)
                tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

                # 128-bit cp.async: 32 threads × (128/dtype_bits) elements = tile_N per warp
                bias_elems_per_thread = 128 // self.bias_dtype.width
                bias_g2s_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    self.bias_dtype,
                    num_bits_per_copy=128,
                )
                bias_g2s_tiled = cute.make_tiled_copy_tv(
                    bias_g2s_atom,
                    cute.make_layout((self.threads_per_warp,)),
                    cute.make_layout((bias_elems_per_thread,)),
                )
                thr_bias_g2s = bias_g2s_tiled.get_slice(cute.arch.lane_idx())
                tBs_sBias = thr_bias_g2s.partition_D(sBias)

                # Predicate tensor for bias cp.async
                bias_n_total = mBias_nl.shape[0]
                tBpBias = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)

                # Get first tile info from pipeline
                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    bias_producer_state.reset_count()

                    # sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                    mma_n_coord = tile_info[2]
                    expert_idx = tile_info[0]

                    gBias_tile = gBias_nl[(None, mma_n_coord, expert_idx)]
                    tBs_gBias = thr_bias_g2s.partition_S(gBias_tile)

                    # Predicate: check if this thread's chunk is within N
                    tBpBias[0] = mma_n_coord * self.mma_tiler[1] + cute.arch.lane_idx() * bias_elems_per_thread < bias_n_total

                    bias_pipeline.producer_acquire(bias_producer_state)
                    cute.copy(bias_g2s_tiled, tBs_gBias[(None, 0)], tBs_sBias[(None, 0, bias_producer_state.index)], pred=tBpBias)
                    bias_pipeline.producer_commit(bias_producer_state)
                    bias_producer_state.advance()

                    # Get next tile info
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                bias_pipeline.producer_tail(bias_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue (shape-only: use global tensor for invariant setup)
            #
            epi_tidx = tidx
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
            # D has half as many logical N columns as the MMA accumulator because
            # adjacent 32-column blocks are interpreted as gate/up pairs.
            gD_mnl_shape = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
            tCgD_shape = thr_mma_epi.partition_C(gD_mnl_shape)

            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_gate,
                tTR_rAcc_up,
            ) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgD_shape, epi_tile, use_2cta_instrs)

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_gate.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc_gate.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rD, epi_tidx, sD)

            epi_ext = self._make_extension(workspace_ptr)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            d_pipeline = None
            # Threads/warps participating in tma store pipeline
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage,
                producer_group=d_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            if cutlass.const_expr(self.enable_bias):
                bias_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_bias_stage)
                bias_s2r_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.bias_dtype, num_bits_per_copy=128)
                tTR_rBias_gate = cute.make_rmem_tensor(cute.make_layout(self.epi_tile[1]), self.bias_dtype)
                tTR_rBias_up = cute.make_rmem_tensor(cute.make_layout(self.epi_tile[1]), self.bias_dtype)

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                # sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                epi_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                mma_tile_coord_mnl = (
                    epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    epi_work_tile_info.tile_n_idx,
                    cutlass.Int32(0),
                )

                expert_idx = epi_work_tile_info.expert_idx
                alpha_val = alpha[expert_idx]
                epi_ext.update_expert_info(padded_offsets, epi_work_tile_info.expert_idx)

                if cutlass.const_expr(self.enable_bias):
                    bias_consumer_state.reset_count()
                    bias_pipeline.consumer_wait(bias_consumer_state)
                    sBias_stage = sBias[(None, bias_consumer_state.index)]
                    sBias_subtiles = cute.flat_divide(sBias_stage, cute.make_layout(2 * self.epi_tile[1]))

                # Get per-expert C/D tensors via extension
                real_c, _ = epi_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, epi_work_tile_info)
                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)

                # local_tile + partition on per-expert tensors
                thr_mma_epi_loop = tiled_mma.get_slice(mma_tile_coord_v)
                gC_mnl = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                tCgC = thr_mma_epi_loop.partition_C(gC_mnl)
                _, bSG_sC, bSG_gC_partitioned = epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, self.epi_tile_c, sC)

                gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                tCgD_loop = thr_mma_epi_loop.partition_C(gD_mnl_loop)
                _, bSG_sD, bSG_gD_partitioned = epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD_loop, epi_tile, sD)

                # Slice to per-expert tile coords (L=0, domain already offset'd)
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gD = bSG_gD_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                #
                # Get PROB (per-expert via domain_offset)
                # Note, it always assumes T2R_M/EPI_M is 1, otherwise it will break the result.
                #
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mPosition = (
                    (epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)) * self.mma_tiler[0]
                    + mma_tile_coord_v * (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape))
                    + tidx
                )
                mProb = real_prob[mPosition, 0, 0]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                #
                # Store accumulator to global memory in subtiles
                #
                # Each loop consumes two adjacent accumulator subtiles:
                #   gate -> C and activation input
                #   up   -> C and activation input
                # C receives the pre-activation values for debugging/reference;
                # D receives the final GLU result.
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range(0, subtile_cnt, 2, unroll=1):
                    real_subtile_idx = subtile_idx // 2

                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
                    tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2 + 1)]

                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)

                    #
                    # Apply alpha (+ bias if enabled)
                    #
                    if cutlass.const_expr(self.enable_bias):
                        sBias_sub = sBias_subtiles[(None, real_subtile_idx)]
                        for i in cutlass.range_constexpr(self.epi_tile[1]):
                            tTR_rBias_gate[i] = sBias_sub[i]
                            tTR_rBias_up[i] = sBias_sub[self.epi_tile[1] + i]
                        bias_vec_gate = tTR_rBias_gate.load()
                        bias_vec_up = tTR_rBias_up.load()

                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_gate), 2):
                                bias_gate_f32_0 = bias_vec_gate[i].to(cutlass.Float32)
                                bias_gate_f32_1 = bias_vec_gate[i + 1].to(cutlass.Float32)
                                bias_up_f32_0 = bias_vec_up[i].to(cutlass.Float32)
                                bias_up_f32_1 = bias_vec_up[i + 1].to(cutlass.Float32)
                                tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1] = cute.arch.fma_packed_f32x2(
                                    (tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                    (bias_gate_f32_0, bias_gate_f32_1),
                                    rnd="rn",
                                    ftz=False,
                                )
                                tTR_rAcc_up[i], tTR_rAcc_up[i + 1] = cute.arch.fma_packed_f32x2(
                                    (tTR_rAcc_up[i], tTR_rAcc_up[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                    (bias_up_f32_0, bias_up_f32_1),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_gate)):
                                tTR_rAcc_gate[i] = tTR_rAcc_gate[i] * cutlass.Float32(alpha_val) + bias_vec_gate[i].to(cutlass.Float32)
                                tTR_rAcc_up[i] = tTR_rAcc_up[i] * cutlass.Float32(alpha_val) + bias_vec_up[i].to(cutlass.Float32)

                        if subtile_idx == subtile_cnt - 2:
                            bias_pipeline.consumer_release(bias_consumer_state)
                            bias_consumer_state.advance()
                    else:
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_gate), 2):
                                tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1] = cute.arch.mul_packed_f32x2(
                                    (tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                    rnd="rn",
                                    ftz=False,
                                )
                                tTR_rAcc_up[i], tTR_rAcc_up[i + 1] = cute.arch.mul_packed_f32x2(
                                    (tTR_rAcc_up[i], tTR_rAcc_up[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_gate)):
                                tTR_rAcc_gate[i] = tTR_rAcc_gate[i] * cutlass.Float32(alpha_val)
                                tTR_rAcc_up[i] = tTR_rAcc_up[i] * cutlass.Float32(alpha_val)

                    if cutlass.const_expr(self.generate_c):
                        self.store_c(
                            tiled_copy_r2s,
                            tma_atom_c,
                            warp_idx,
                            tTR_rAcc_gate,
                            tTR_rAcc_up,
                            tRS_rC,
                            tRS_sC,
                            bSG_gC,
                            bSG_sC,
                            c_pipeline,
                            num_prev_subtiles,
                            real_subtile_idx,
                        )

                    if cutlass.const_expr(self.act_func == "geglu"):
                        geglu_max_val = cutlass.Float32(7.0)
                        geglu_min_val = cutlass.Float32(-7.0)
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                            tTR_rAcc_gate[i] = fmin(tTR_rAcc_gate[i], geglu_max_val)
                            tTR_rAcc_up[i] = fmin(tTR_rAcc_up[i], geglu_max_val)
                            tTR_rAcc_up[i] = fmax(tTR_rAcc_up[i], geglu_min_val)

                    acc_vec_gate = tTR_rAcc_gate.load()
                    acc_vec_up = tTR_rAcc_up.load()

                    # SwiGlu or GeGLU
                    tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                    if cutlass.const_expr(self.act_func == "geglu"):
                        self.geglu_act(tCompute, acc_vec_up, acc_vec_gate, mProb, linear_offset)
                    elif cutlass.const_expr(self.act_func == "swiglu"):
                        self.swiglu_act(tCompute, acc_vec_up, acc_vec_gate, mProb)

                    #
                    # Convert to D type
                    #
                    acc_vec = tiled_copy_r2s.retile(tCompute).load()
                    tRS_rD.store(acc_vec.to(self.d_dtype))

                    #
                    # Store D to shared memory
                    #
                    d_buffer = num_prev_subtiles % self.num_d_stage
                    num_prev_subtiles = num_prev_subtiles + 1
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD,
                        tRS_sD[(None, None, None, d_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    #
                    # TMA store D to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_d,
                            bSG_sD[(None, d_buffer)],
                            bSG_gD[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C/D store complete
            #
            if cutlass.const_expr(self.generate_c):
                c_pipeline.producer_tail()
            d_pipeline.producer_tail()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_gate, tTR_rAcc_up) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gD_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_gate = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_up = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_gate, tTR_rAcc_up

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor D (register source)
            - tRS_sD: The partitioned tensor D (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rD, tRS_sD
