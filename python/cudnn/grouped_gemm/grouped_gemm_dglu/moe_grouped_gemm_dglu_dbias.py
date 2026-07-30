# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MoE BF16 Grouped GEMM Kernel with dGLU (dSwiGLU/dGeGLU) Backward Fusion.

Supports:
    - Static / Dynamic persistent tile scheduling (MoEPersistentTileScheduler)
    - Dense (contiguous 3-D B) / Discrete (per-expert pointer array B) weight layout
    - dGLU backward activation fusion (dSwiGLU / dGeGLU)
    - Optional dBias reduction

This module contains only the kernel class.
MoE scheduler components live in moe_persistent_scheduler.py / moe_sched_extension.py / moe_utils.py.
"""

from typing import Type, Tuple, Union, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils

from cutlass.cute.typing import Float32, Int32, AddressSpace
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector

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
    atomic_add_float32,
    sigmoid_f32,
    can_implement_bf16_grouped_gemm,
    compute_grid,
)


def atomic_add_bf16x2(ptr, val_fp32_lo, val_fp32_hi, *, loc=None, ip=None):
    """Packed BF16x2 atomic reduction to global memory."""
    lo_ir = val_fp32_lo.ir_value(loc=loc, ip=ip)
    hi_ir = val_fp32_hi.ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr, hi_ir, lo_ir],
        "{ .reg .b32 packed; cvt.rn.bf16x2.f32 packed, $1, $2; red.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class MoEGroupedGemmDgluDbiasBf16Kernel:
    """Plain BF16 grouped GEMM kernel with MoE scheduling and dGLU backward fusion.

    The kernel is organized as persistent scheduler, A/B TMA load, MMA, C-load,
    and epilogue warps. The epilogue computes dSwiGLU or dGeGLU gradients,
    reduces dprob, optionally reduces dbias, and writes D.
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
        act_func: str,
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
            fix_pad_size=MoEGroupedGemmDgluDbiasBf16Kernel.FIX_PAD_SIZE,
            n_align=32,
            tile_n_align=32,
        ) and act_func in ["dswiglu", "dgeglu"]

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
        act_func: str = "dswiglu",
    ):
        # Validate FIX_PAD_SIZE compatibility with tile size
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % mma_tile_m != 0:
            raise ValueError(
                f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by " f"mma_tiler_mn[0] ({mma_tile_m}). " f"Supported mma_tiler_mn[0] values: 128, 256."
            )
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")
        if act_func not in ["dswiglu", "dgeglu"]:
            raise ValueError(f"Invalid activation function: {act_func}")

        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.epilog_load_tma_id = 6
        self.sched_warp_id = 7
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.epilog_load_tma_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.epilog_load_tma_id,
            )
        )

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
        self.use_dynamic_sched = use_dynamic_sched
        self.store_d_directly = False

        self.num_epilog_warps = len(self.epilog_warp_id)

        self.weight_mode = weight_mode

        self.act_func = act_func

    def _setup_attributes(self):
        """Set up input-dependent BF16 GEMM attributes."""
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
            self.mma_tiler[1],
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

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_d_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.d_dtype,
            self.d_layout,
            self.num_smem_capacity,
            self.occupancy,
            self.store_d_directly,
            self.generate_dbias,
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
            self.epi_tile,
            self.num_c_stage,
        )

        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_dtype,
            self.d_layout,
            self.epi_tile,
            1 if self.store_d_directly else self.num_d_stage,
        )

        self.epilogue_prefetch_more = False
        self.generate_dprob = True

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
        workspace_ptr,  # Descriptor workspace, plus dynamic scheduler counter when enabled
        c: cute.Tensor,
        d: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
        linear_offset: Float32,
        dbias_tensor: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM.

        Dense mode: ``b`` is a 3-D cute.Tensor (N, K, L).
        Discrete mode: ``b`` is a cute.Pointer to a device int64[] array of
        per-expert base addresses; ``n``, ``k``, ``b_stride_size``, and
        ``b_major_mode`` describe the uniform per-expert layout.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type  # B must match A dtype
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        else:
            self.b_major_mode = b_major_mode
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.d_layout = utils.LayoutEnum.from_tensor(d)

        # dBias configuration
        self.generate_dbias = dbias_tensor is not None
        self.dbias_cross_warp_reduce = self.generate_dbias  # always cross-warp reduce

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # ---- B setup (mode-dependent) ----
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

        # Compute grid size
        m, n_d, l = cute.shape(d)

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

        # Setup TMA load for C
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        self.tma_c_load_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            c,
            c_smem_layout,
            self.epi_tile,
        )

        # Setup TMA store for D unless the tuning knob asks epilogue warps to
        # write D directly from registers to GMEM.
        tma_atom_d = None
        tma_tensor_d = d
        if cutlass.const_expr(not self.store_d_directly):
            d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
            tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                d,
                d_smem_layout,
                self.epi_tile,
            )

        # Compute grid size using MoE scheduler
        # dGLU output has shape (m, 2*N_half, 1), but scheduling is over (m, N_half)
        # expert_shape = (expert_cnt, N_half, K)
        n_half = n_d // 2
        sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=(self.expert_cnt, n_half, cute.size(a.shape, mode=[1])),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk_d,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=self.use_dynamic_sched,
        )
        self.sched_params, grid = compute_grid(sched_params, max_active_clusters, self.use_2cta_instrs)

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)

        sD_size = 0 if self.store_d_directly else cute.cosize(self.d_smem_layout_staged.outer)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            scheduler: SchedulerStorage
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            c_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
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
                    sD_size,
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
            # dBias SMEM transpose buffer: (128, epi_tile_n*2) col-major FP32
            sDbias: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    128 * self.epi_tile[1] * 2 if self.generate_dbias else 1,
                ],
                128 if self.generate_dbias else 4,
            ]

        self.shared_storage = SharedStorage

        # Initialize per-expert B TMA descriptors in workspace
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
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

        # Launch the main kernel
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
            beta,
            prob,
            dprob,
            linear_offset,
            dbias_tensor,
            workspace_ptr,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.d_smem_layout_staged,
            self.epi_tile,
            self.sched_params,
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
    def dbias_reduction(
        self,
        d1_vec,
        d2_vec,
        warp_idx,
        sDbias,
        dbias_gmem_2d,
        expert_idx,
        n_base_d1,
        n_base_d2,
        dbias_n_total,
    ) -> None:
        """Merged dy1+dy2 dbias reduction via SMEM transpose."""
        epi_n = self.epi_tile[1]
        lane_idx = cute.arch.lane_idx()
        warp_local = warp_idx - self.epilog_warp_id[0]

        for n in cutlass.range(epi_n, unroll_full=True):
            sDbias[(n, lane_idx, warp_local)] = d1_vec[n]
            sDbias[(epi_n + n, lane_idx, warp_local)] = d2_vec[n]

        self.epilog_sync_barrier.arrive_and_wait()

        col_a = 2 * lane_idx if lane_idx < 16 else epi_n + 2 * (lane_idx - 16)
        col_b = col_a + 1

        copy_128bit_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)
        warp_base_ptr = sDbias.iterator + warp_local * epi_n * 2 * 32
        swizzle_a = ((col_a >> 1) & 0x7) << 2
        swizzle_b = ((col_b >> 1) & 0x7) << 2

        sum_a = cutlass.Float32(0.0)
        sum_b = cutlass.Float32(0.0)
        rDst_a = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        rDst_b = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        for g in cutlass.range(8, unroll_full=True):
            m_base = g * 4
            sw_offset_a = col_a * 32 + (m_base ^ swizzle_a)
            sSrc_a = cute.make_tensor(warp_base_ptr + sw_offset_a, cute.make_layout((4,)))
            cute.copy_atom_call(copy_128bit_atom, sSrc_a, rDst_a)

            sw_offset_b = col_b * 32 + (m_base ^ swizzle_b)
            sSrc_b = cute.make_tensor(warp_base_ptr + sw_offset_b, cute.make_layout((4,)))
            cute.copy_atom_call(copy_128bit_atom, sSrc_b, rDst_b)

            for i in cutlass.range(4, unroll_full=True):
                sum_a = sum_a + rDst_a[i]
                sum_b = sum_b + rDst_b[i]

        n_offset = (n_base_d1 + 2 * lane_idx) if lane_idx < 16 else (n_base_d2 + 2 * (lane_idx - 16))

        if cutlass.const_expr(self.dbias_cross_warp_reduce):
            reduce_base = sDbias.iterator
            copy_64bit_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64)

            self.epilog_sync_barrier.arrive_and_wait()
            rSrc_partial = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)
            rSrc_partial[0] = sum_a
            rSrc_partial[1] = sum_b
            sDst_partial = cute.make_tensor(reduce_base + warp_local * 64 + lane_idx * 2, cute.make_layout((2,)))
            cute.copy_atom_call(copy_64bit_atom, rSrc_partial, sDst_partial)
            self.epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epilog_warp_id[0]:
                cta_sum_a = cutlass.Float32(0.0)
                cta_sum_b = cutlass.Float32(0.0)
                rDst_w = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)
                for w in cutlass.range(self.num_epilog_warps):
                    sSrc_w = cute.make_tensor(reduce_base + w * 64 + lane_idx * 2, cute.make_layout((2,)))
                    cute.copy_atom_call(copy_64bit_atom, sSrc_w, rDst_w)
                    cta_sum_a = cta_sum_a + rDst_w[0]
                    cta_sum_b = cta_sum_b + rDst_w[1]
                if n_offset < dbias_n_total:
                    gmem_ptr = dbias_gmem_2d[(expert_idx, n_offset, None)].iterator.llvm_ptr
                    atomic_add_bf16x2(gmem_ptr, cta_sum_a, cta_sum_b)
        else:
            if n_offset < dbias_n_total:
                gmem_ptr = dbias_gmem_2d[(expert_idx, n_offset, None)].iterator.llvm_ptr
                atomic_add_bf16x2(gmem_ptr, sum_a, sum_b)

    @cute.jit
    def dswiglu(
        self,
        acc_vec: cute.Tensor,
        ab1_vec_load: cute.Tensor,
        ab2_vec_load: cute.Tensor,
        mProb: cute.Tensor,
        beta_val: Float32,
        square_alpha: Float32,
        dprob_swiglu: Optional[cute.Tensor] = None,
    ):
        LOG2_E = cutlass.Float32(1.4426950408889634)
        if cutlass.const_expr(self.vectorized_f32):
            d1_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            d2_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            for i in cutlass.range(0, cute.size(acc_vec), 2, unroll_full=True):
                # Apply dGLU alpha/beta/prob scaling.
                (
                    acc_vec[i + 0],
                    acc_vec[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (acc_vec[i + 0], acc_vec[i + 1]),
                    (square_alpha, square_alpha),
                    rnd="rn",
                    ftz=False,
                )
                ab1_vec_acc_type = cute.arch.mul_packed_f32x2(
                    (
                        ab1_vec_load[i + 0].to(self.acc_dtype),
                        ab1_vec_load[i + 1].to(self.acc_dtype),
                    ),
                    (beta_val, beta_val),
                    rnd="rn",
                    ftz=False,
                )
                ab2_vec_acc_type = cute.arch.mul_packed_f32x2(
                    (
                        ab2_vec_load[i + 0].to(self.acc_dtype),
                        ab2_vec_load[i + 1].to(self.acc_dtype),
                    ),
                    (beta_val, beta_val),
                    rnd="rn",
                    ftz=False,
                )
                sig_rcp_0, sig_rcp_1 = cute.arch.mul_packed_f32x2(
                    (ab1_vec_acc_type),
                    (-LOG2_E, -LOG2_E),
                    rnd="rn",
                    ftz=False,
                )
                sig_rcp_0, sig_rcp_1 = cute.arch.add_packed_f32x2(
                    (
                        cute.math.exp2(sig_rcp_0, fastmath=True),
                        cute.math.exp2(sig_rcp_1, fastmath=True),
                    ),
                    (1.0, 1.0),
                    rnd="rn",
                    ftz=False,
                )
                sig = (
                    cute.arch.rcp_approx(sig_rcp_0),
                    cute.arch.rcp_approx(sig_rcp_1),
                )
                swish = cute.arch.mul_packed_f32x2(
                    ab1_vec_acc_type,
                    sig,
                    rnd="rn",
                    ftz=False,
                )
                # calculate dprob
                if cutlass.const_expr(self.generate_dprob):
                    (
                        dprob_swiglu[i + 0],
                        dprob_swiglu[i + 1],
                    ) = cute.arch.mul_packed_f32x2(
                        (ab2_vec_acc_type[0], ab2_vec_acc_type[1]),
                        swish,
                    )
                    (
                        dprob_swiglu[i + 0],
                        dprob_swiglu[i + 1],
                    ) = cute.arch.mul_packed_f32x2(
                        (dprob_swiglu[i + 0], dprob_swiglu[i + 1]),
                        (acc_vec[i + 0], acc_vec[i + 1]),
                    )
                # calculate dswiglu
                acc_vec_prob = cute.arch.mul_packed_f32x2(
                    (acc_vec[i + 0], acc_vec[i + 1]),
                    (mProb, mProb),
                )
                # calculate d2_vec
                (
                    d2_vec[i + 0],
                    d2_vec[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (acc_vec_prob[0], acc_vec_prob[1]),
                    swish,
                    rnd="rn",
                    ftz=False,
                )
                # calculate d1_vec
                (
                    d1_vec[i + 0],
                    d1_vec[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (acc_vec_prob[0], acc_vec_prob[1]),
                    (ab2_vec_acc_type[0], ab2_vec_acc_type[1]),
                    rnd="rn",
                    ftz=False,
                )
                (
                    d1_vec[i + 0],
                    d1_vec[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (d1_vec[i + 0], d1_vec[i + 1]),
                    sig,
                    rnd="rn",
                    ftz=False,
                )
                one_minus_sig = cute.arch.add_packed_f32x2(
                    (1.0, 1.0),
                    (-sig[0], -sig[1]),
                    rnd="rn",
                    ftz=False,
                )
                dsig = cute.arch.mul_packed_f32x2(
                    ab1_vec_acc_type,
                    one_minus_sig,
                    rnd="rn",
                    ftz=False,
                )
                dsig_add_1 = cute.arch.add_packed_f32x2(
                    (dsig[0], dsig[1]),
                    (1.0, 1.0),
                    rnd="rn",
                    ftz=False,
                )
                (
                    d1_vec[i + 0],
                    d1_vec[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (d1_vec[i + 0], d1_vec[i + 1]),
                    dsig_add_1,
                    rnd="rn",
                    ftz=False,
                )
            d1_vec = d1_vec.load()
            d2_vec = d2_vec.load()
            if cutlass.const_expr(self.generate_dprob):
                dprob_swiglu = dprob_swiglu.load()
            return d1_vec, d2_vec, dprob_swiglu
        else:
            acc_vec = acc_vec.load()
            ab1_vec_load = ab1_vec_load.load()
            ab2_vec_load = ab2_vec_load.load()

            acc_vec = acc_vec * square_alpha  # apply scale for A*B
            ab1_vec_load = ab1_vec_load * beta_val  # apply scale for C
            ab2_vec_load = ab2_vec_load * beta_val  # apply scale for C

            sig_rcp = (1 + cute.math.exp(-1 * ab1_vec_load, True)).to(self.acc_dtype)
            res = cute.make_rmem_tensor(sig_rcp.shape, cutlass.Float32)
            res.store(sig_rcp)
            # let every res[?] be cute.arch.rcp_approx(res[?])
            [res.__setitem__(i, cute.arch.rcp_approx(res[i])) for i in range(cute.size(res.shape))]
            sig = res.load()
            swish = ab1_vec_load * sig

            # calculate dprob
            if cutlass.const_expr(self.generate_dprob):
                dprob_swiglu = ab2_vec_load * swish
                dprob_swiglu = acc_vec * dprob_swiglu

            # calculate dswiglu
            d1_vec = acc_vec * mProb * ab2_vec_load * sig * (1 + ab1_vec_load * (1 - sig))
            d2_vec = acc_vec * mProb * swish
            return d1_vec, d2_vec, dprob_swiglu

    @cute.jit
    def dgeglu(
        self,
        acc_vec: cute.Tensor,
        x1_vec_load: cute.Tensor,
        x2_vec_load: cute.Tensor,
        mProb: cute.Tensor,
        beta_val: Float32,
        square_alpha: Float32,
        linear_offset: Float32,
        dprob_swiglu: Optional[cute.Tensor] = None,
    ):
        geglu_max_value = cutlass.Float32(7.0)
        geglu_min_value = cutlass.Float32(-7.0)
        fmul2 = partial(cute.arch.mul_packed_f32x2, rnd="rn", ftz=False)
        fadd2 = partial(cute.arch.add_packed_f32x2, rnd="rn", ftz=False)
        scale_1702 = (1.702, 1.702)
        ones2 = (1.0, 1.0)
        mprob2 = (mProb, mProb)
        beta2 = (beta_val, beta_val)
        square_alpha2 = (square_alpha, square_alpha)
        linear_offset2 = (linear_offset, linear_offset)

        if cutlass.const_expr(self.vectorized_f32):
            dx1_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            dx2_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            for i in cutlass.range(0, cute.size(acc_vec), 2, unroll_full=True):
                acc = fmul2((acc_vec[i], acc_vec[i + 1]), square_alpha2)
                x1_0, x1_1 = fmul2(
                    (
                        x1_vec_load[i].to(self.acc_dtype),
                        x1_vec_load[i + 1].to(self.acc_dtype),
                    ),
                    beta2,
                )
                x2_0, x2_1 = fmul2(
                    (
                        x2_vec_load[i].to(self.acc_dtype),
                        x2_vec_load[i + 1].to(self.acc_dtype),
                    ),
                    beta2,
                )

                y1_0 = fmin(x1_0, geglu_max_value)
                y1_1 = fmin(x1_1, geglu_max_value)
                y2_0 = fmin(x2_0, geglu_max_value)
                y2_1 = fmin(x2_1, geglu_max_value)
                y2_0 = fmax(y2_0, geglu_min_value)
                y2_1 = fmax(y2_1, geglu_min_value)

                y1 = (y1_0, y1_1)
                y2 = (y2_0, y2_1)

                # y1 = 1.702 * x1
                y1_scaled = fmul2(y1, scale_1702)

                sigmoid_out_0 = sigmoid_f32(y1_scaled[0], fastmath=True)
                sigmoid_out_1 = sigmoid_f32(y1_scaled[1], fastmath=True)

                # g * sigmoid_out
                acc_mul_sigmoid_out = fmul2(acc, (sigmoid_out_0, sigmoid_out_1))
                acc_mul_sigmoid_prob = fmul2(acc_mul_sigmoid_out, mprob2)

                # y1 = 1 + 1.702 * y1 * (1 - sigmoid_out)
                one_minus_sigmoid_0, one_minus_sigmoid_1 = fadd2(ones2, (-sigmoid_out_0, -sigmoid_out_1))
                y1_scaled = fadd2(
                    fmul2(y1_scaled, (one_minus_sigmoid_0, one_minus_sigmoid_1)),
                    ones2,
                )

                # y2 + linear_offset
                y2_with_linear_offset_0, y2_with_linear_offset_1 = fadd2(y2, linear_offset2)

                # dy1 = g * sigmoid_out * (y2 + linear_offset)
                dy1_pre_0, dy1_pre_1 = fmul2(
                    (y2_with_linear_offset_0, y2_with_linear_offset_1),
                    acc_mul_sigmoid_out,
                )
                # dy1 = g * sigmoid_out * (y2 + linear_offset) * (1 + 1.702 * y1 * (1 - sigmoid_out)) * mProb
                dy1_0, dy1_1 = fmul2((dy1_pre_0, dy1_pre_1), y1_scaled)
                dy1_0, dy1_1 = fmul2((dy1_0, dy1_1), mprob2)

                x1_filter_0 = y1_0 if x1_0 <= geglu_max_value else cutlass.Float32(0.0)
                x1_filter_1 = y1_1 if x1_1 <= geglu_max_value else cutlass.Float32(0.0)

                dx1_vec[i], dx1_vec[i + 1] = fmul2((dy1_0, dy1_1), (cutlass.Float32(x1_filter_0), cutlass.Float32(x1_filter_1)))

                # dy2 = g * y1 * sigmoid_out * mProb
                dy2_0, dy2_1 = fmul2(y1, acc_mul_sigmoid_prob)
                x2_filter_0 = x2_0 if x2_0 <= geglu_max_value else cutlass.Float32(0.0)
                x2_filter_1 = x2_1 if x2_1 <= geglu_max_value else cutlass.Float32(0.0)
                x2_filter_0 = y2_0 if x2_filter_0 >= geglu_min_value else cutlass.Float32(0.0)
                x2_filter_1 = y2_1 if x2_filter_1 >= geglu_min_value else cutlass.Float32(0.0)
                dx2_vec[i], dx2_vec[i + 1] = fmul2((dy2_0, dy2_1), (cutlass.Float32(x2_filter_0), cutlass.Float32(x2_filter_1)))

                if cutlass.const_expr(self.generate_dprob):
                    prob_grad, prob_grad_1 = fmul2(
                        (dy1_pre_0, dy1_pre_1),
                        y1,
                    )
                    dprob_swiglu[i] = prob_grad
                    dprob_swiglu[i + 1] = prob_grad_1
            dx1_vec = dx1_vec.load()
            dx2_vec = dx2_vec.load()
            if cutlass.const_expr(self.generate_dprob):
                dprob_swiglu = dprob_swiglu.load()
            return dx1_vec, dx2_vec, dprob_swiglu
        else:
            element_count = cute.size(x1_vec_load)
            acc_vec = acc_vec.load() * square_alpha
            x1_vec_load = x1_vec_load.load().to(cutlass.Float32) * beta_val
            x2_vec_load = x2_vec_load.load().to(cutlass.Float32) * beta_val
            dx1_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            dx2_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)

            # y1 = clamp(x1, max=7.0); y2 = clamp(x2, min=-7.0, max=7.0)
            for i in cutlass.range_constexpr(element_count):
                fc2_dgrad = acc_vec[i]
                g = fc2_dgrad * mProb
                y1 = min(x1_vec_load[i], 7.0)
                y2 = min(x2_vec_load[i], 7.0)
                y2 = max(y2, -7.0)

                sigmoid_out = sigmoid_f32(y1 * 1.702, fastmath=True)

                dy1 = g * sigmoid_out * (1 + 1.702 * y1 * (1 - sigmoid_out)) * (y2 + linear_offset)
                dy2 = g * y1 * sigmoid_out

                x1_filter = x1_vec_load[i] if x1_vec_load[i] <= 7.0 else 0.0
                x2_filter = x2_vec_load[i] if x2_vec_load[i] <= 7.0 else 0.0
                x2_filter = x2_filter if x2_filter >= -7.0 else 0.0

                dx1_vec[i] = x1_filter * dy1
                dx2_vec[i] = x2_filter * dy2

                if cutlass.const_expr(self.generate_dprob):
                    prob_grad = y1 * sigmoid_out * (y2 + linear_offset) * fc2_dgrad
                    dprob_swiglu[i] = prob_grad

            return dx1_vec.load(), dx2_vec.load(), dprob_swiglu.load()

    @cute.jit
    def stg_256(self, ptr, vec8_f32, *, loc=None, ip=None):
        """Store 256 bits to global memory with L1 no-allocate."""
        dst = ptr.ir_value(loc=loc, ip=ip) if hasattr(ptr, "ir_value") else ptr
        src = vec8_f32.ir_value(loc=loc, ip=ip) if hasattr(vec8_f32, "ir_value") else vec8_f32
        llvm.inline_asm(
            T.i32(),
            [
                dst,
                vector.extract(src, [], [0], loc=loc, ip=ip),
                vector.extract(src, [], [1], loc=loc, ip=ip),
                vector.extract(src, [], [2], loc=loc, ip=ip),
                vector.extract(src, [], [3], loc=loc, ip=ip),
                vector.extract(src, [], [4], loc=loc, ip=ip),
                vector.extract(src, [], [5], loc=loc, ip=ip),
                vector.extract(src, [], [6], loc=loc, ip=ip),
                vector.extract(src, [], [7], loc=loc, ip=ip),
            ],
            "st.global.L1::no_allocate.v8.f32 [$1], {$2, $3, $4, $5, $6, $7, $8, $9}; mov.u32 $0, 0;",
            "=r,l,f,f,f,f,f,f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def store_global_memory_256b(self, dst: cute.Tensor, src: cute.Tensor):
        vec_shape = cute.make_layout(8)
        dst_f32 = cute.flatten(cute.recast_tensor(dst, cutlass.Float32))
        src_f32 = cute.flatten(cute.recast_tensor(src, cutlass.Float32))
        dst_vf32x8 = cute.logical_divide(dst_f32, vec_shape)
        src_vf32x8 = cute.logical_divide(src_f32, vec_shape)
        for ei in cutlass.range_constexpr(dst_vf32x8.shape[1]):
            self.stg_256(
                dst_vf32x8[None, ei].iterator.llvm_ptr,
                src_vf32x8[None, ei].load(),
            )

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
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
        linear_offset: Float32,
        mDbias_tensor: Optional[cute.Tensor],
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        total_tokens = padded_offsets[self.expert_cnt - 1]

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(not self.store_d_directly):
                cpasync.prefetch_descriptor(tma_atom_d)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

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
            defer_sync=True,
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
            defer_sync=True,
        )

        # Load C pipeline
        # Threads/warps participating in tma store pipeline
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.epilog_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_full_mbar_ptr.data_ptr(),
            num_stages=self.num_c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=self.tma_c_load_bytes,
            defer_sync=True,
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

        # dBias SMEM setup
        if cutlass.const_expr(self.generate_dbias):
            sDbias = storage.sDbias.get_tensor(
                cute.make_layout(
                    (self.epi_tile[1] * 2, 32, len(self.epilog_warp_id)),
                    stride=(32, 1, self.epi_tile[1] * 2 * 32),
                )
            )

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
        sD = None
        if cutlass.const_expr(not self.store_d_directly):
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
        # (SMEM/TMEM partitions stay global - they don't depend on per-expert tensors)
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

        if total_tokens <= 0:
            cute.arch.nvvm.exit()
        k_tile_cnt = cute.ceil_div(cute.size(mB_nkl, mode=[1]), self.mma_tiler[2])

        #
        # Specialized Schedule warp (MoE Persistent Tile Scheduler)
        #
        if warp_idx == self.sched_warp_id:
            work_tile_info = scheduler.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)

            while work_tile_info.is_valid_tile:
                # sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
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
                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                # assert(k_tile_cnt == work_tile_info.k_tile_cnt)
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                # Get per-expert real tensors + TMA desc ptrs via extension
                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, padded_offsets, work_tile_info)
                real_b, desc_ptr_b = ext.get_gmem_tensor("b", mB_nkl, padded_offsets, work_tile_info)

                # local_tile on per-expert tensors
                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))

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
                # Slice to per mma tile index (L=0 since domain already offset'd)
                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]

                # Peek (try_wait) AB buffer empty
                peek_ab_empty_status = cutlass.Boolean(1)
                if k_tile_cnt > 0:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, k_tile)]
                    tBgB_k = tBgB_slice[(None, k_tile)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    ab_producer_state_next = ab_producer_state.clone()
                    ab_producer_state_next.advance()
                    if k_tile < k_tile_cnt - 1:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state_next)

                    # TMA load A (contiguous, global desc)
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
                    ab_producer_state = ab_producer_state_next

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
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info (sInfo format: expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # assert(k_tile_cnt == tile_info[3])

                # Peek (try_wait) AB buffer full for k_tile = 0
                peek_ab_full_status = cutlass.Boolean(1)
                if k_tile_cnt > 0 and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # sInfo: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                mma_tile_coord_mnl = (
                    tile_info[1] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[2],
                    cutlass.Int32(0),
                )

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if k_tile < k_tile_cnt - 1:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])

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
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

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
            acc_pipeline.producer_tail(acc_producer_state)

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
            # Partition for epilogue (SMEM/TMEM/register - invariant across experts)
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs)

            tTR_rC1 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rC2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_s2r, tRS_rC1, tRS_rC2, tRS_sC = self.epilog_smem_copy_and_partition_load(tiled_copy_t2r, tTR_rC1, tTR_rC2, epi_tidx, sC)

            tTR_rD1 = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tTR_rD2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD1, tRS_rD2, tRS_sD = self.epilog_smem_copy_and_partition_store(tiled_copy_t2r, tTR_rD1, tTR_rD2, epi_tidx, sD)

            # Extension for per-expert domain conversion in epilogue
            epi_ext = self._make_extension(workspace_ptr)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Load C pipeline
            c_pipeline_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_c_stage)

            # Threads/warps participating in tma store pipeline
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            d_pipeline = None
            if cutlass.const_expr(not self.store_d_directly):
                num_d_stages = self.num_d_stage // 2
                d_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=num_d_stages,
                    producer_group=d_producer_group,
                )

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info (sInfo format: expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                # sInfo: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                epi_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                expert_idx = epi_work_tile_info.expert_idx
                # N is doubled for dGLU dual output
                mma_tile_coord_mnl = (
                    epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    epi_work_tile_info.tile_n_idx * 2,
                    cutlass.Int32(0),
                )

                #
                # Get alpha/beta for current expert
                #
                alpha_val = alpha[expert_idx]
                beta_val = beta[expert_idx]
                epi_ext.update_expert_info(padded_offsets, expert_idx)

                #
                # Per-expert gmem tensor setup via extension
                #
                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)
                thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)

                if cutlass.const_expr(not self.store_d_directly):
                    gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgD_loop = thr_mma_epi.partition_C(gD_mnl_loop)

                    bSG_sD, bSG_gD_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD_loop, epi_tile, sD)
                    bSG_gD = bSG_gD_partitioned[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                    bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                #
                # Get PROB (per-expert local M position)
                #
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mPosition = (
                    (epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)) * self.mma_tiler[0]
                    + mma_tile_coord_v * (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape))
                    + tidx
                )
                mProb = real_prob[mPosition, 0, 0]
                if cutlass.const_expr(self.generate_dprob):
                    dProbVal = cutlass.Float32(0.0)

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                if cutlass.const_expr(self.epilogue_prefetch_more):
                    tTR_rAcc_0 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
                    tTR_rAcc_1 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    real_subtile_idx = subtile_idx
                    real_subtile_idx_next = subtile_idx + 1
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    copy_atom_t2r = sm100_utils.get_tmem_load_op(
                        self.cta_tile_shape_mnk,
                        self.d_layout,
                        self.d_dtype,
                        self.acc_dtype,
                        epi_tile,
                        use_2cta_instrs,
                    )
                    if cutlass.const_expr(self.epilogue_prefetch_more):
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                        tTR_tAcc_mn_next = tTR_tAcc[(None, None, None, real_subtile_idx_next)]
                        if subtile_idx % 2 == 0:
                            cute.copy(copy_atom_t2r, tTR_tAcc_mn, tTR_rAcc_0)
                            cute.copy(copy_atom_t2r, tTR_tAcc_mn_next, tTR_rAcc_1)
                            tTR_rAcc = tTR_rAcc_0
                        else:
                            tTR_rAcc = tTR_rAcc_1
                    else:
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                        cute.copy(copy_atom_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Wait for C1/C2 load to complete
                    c_pipeline.consumer_wait(c_pipeline_consumer_state)
                    cute.copy(
                        tiled_copy_s2r,
                        tRS_sC[(None, None, None, c_pipeline_consumer_state.index)],
                        tRS_rC1,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    c_pipeline.consumer_release(c_pipeline_consumer_state)
                    c_pipeline_consumer_state.advance()
                    c_pipeline.consumer_wait(c_pipeline_consumer_state)
                    cute.copy(
                        tiled_copy_s2r,
                        tRS_sC[(None, None, None, c_pipeline_consumer_state.index)],
                        tRS_rC2,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    c_pipeline.consumer_release(c_pipeline_consumer_state)
                    c_pipeline_consumer_state.advance()

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc)
                    ab1_vec_load = tiled_copy_r2s.retile(tRS_rC1)
                    ab2_vec_load = tiled_copy_r2s.retile(tRS_rC2)
                    if cutlass.const_expr(self.generate_dprob):
                        dprob_swiglu = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
                    else:
                        dprob_swiglu = None

                    #
                    # Apply alpha, act, and prob
                    #
                    square_alpha = alpha_val * alpha_val
                    if cutlass.const_expr(self.act_func == "dswiglu"):
                        d1_vec, d2_vec, dprob_swiglu = self.dswiglu(acc_vec, ab1_vec_load, ab2_vec_load, mProb, beta_val, square_alpha, dprob_swiglu)
                    elif cutlass.const_expr(self.act_func == "dgeglu"):
                        d1_vec, d2_vec, dprob_swiglu = self.dgeglu(
                            acc_vec, ab1_vec_load, ab2_vec_load, mProb, beta_val, square_alpha, linear_offset, dprob_swiglu
                        )

                    if cutlass.const_expr(self.generate_dprob):
                        # dprob sum reduction
                        if cutlass.const_expr(self.vectorized_f32):
                            dprob_pair_0 = cutlass.Float32(0.0)
                            dprob_pair_1 = cutlass.Float32(0.0)
                            for j in cutlass.range(0, cute.size(dprob_swiglu.shape), 2, unroll_full=True):
                                (
                                    dprob_pair_0,
                                    dprob_pair_1,
                                ) = cute.arch.add_packed_f32x2(
                                    (dprob_pair_0, dprob_pair_1),
                                    (dprob_swiglu[j], dprob_swiglu[j + 1]),
                                    rnd="rn",
                                    ftz=False,
                                )
                            dProbVal += dprob_pair_0 + dprob_pair_1
                        else:
                            dProbVal += dprob_swiglu.reduce(
                                cute.ReductionOp.ADD,
                                cutlass.Float32(0.0),
                                0,
                            )

                    #
                    # Generate dBias
                    #
                    if cutlass.const_expr(self.generate_dbias):
                        n_base_d1 = epi_work_tile_info.tile_n_idx * (self.mma_tiler[1] * 2) + (2 * real_subtile_idx + 0) * self.epi_tile[1]
                        n_base_d2 = epi_work_tile_info.tile_n_idx * (self.mma_tiler[1] * 2) + (2 * real_subtile_idx + 1) * self.epi_tile[1]
                        dbias_n_total = cute.size(mDbias_tensor, mode=[1])
                        self.dbias_reduction(
                            d1_vec,
                            d2_vec,
                            warp_idx,
                            sDbias,
                            mDbias_tensor,
                            expert_idx,
                            n_base_d1,
                            n_base_d2,
                            dbias_n_total,
                        )

                    #
                    # Convert to D type
                    #
                    tRS_rD1.store(d1_vec.to(self.d_dtype))
                    tRS_rD2.store(d2_vec.to(self.d_dtype))

                    #
                    # Store D
                    #
                    if cutlass.const_expr(self.store_d_directly):
                        self.epilog_sync_barrier.arrive_and_wait()
                        d_idx_mn = (
                            epi_work_tile_info.tile_m_idx,
                            epi_work_tile_info.tile_n_idx,
                        )
                        d_epilogue_subtile = (
                            cute.make_layout(128),
                            cute.make_layout(self.mma_tiler[1] * 2),
                        )
                        gD_sub_loop = cute.local_tile(real_d, d_epilogue_subtile, (None, None, None))
                        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                        tCgD_mnl_loop = thr_copy_t2r.partition_D(gD_sub_loop)
                        tCgD_mnl_loop = cute.filter_zeros(tCgD_mnl_loop)
                        tCgD1 = tCgD_mnl_loop[
                            (
                                None,
                                0,  # T2R_M
                                2 * real_subtile_idx + 0,  # T2R_N
                                *d_idx_mn,  # RestM/N
                                0,  # RestL
                            )
                        ]
                        tCgD2 = tCgD_mnl_loop[
                            (
                                None,
                                0,  # T2R_M
                                2 * real_subtile_idx + 1,  # T2R_N
                                *d_idx_mn,  # RestM/N
                                0,  # RestL
                            )
                        ]
                        d_n_total = cute.size(real_d, mode=[1])
                        n_base_d1 = epi_work_tile_info.tile_n_idx * (self.mma_tiler[1] * 2) + (2 * real_subtile_idx + 0) * self.epi_tile[1]
                        n_base_d2 = epi_work_tile_info.tile_n_idx * (self.mma_tiler[1] * 2) + (2 * real_subtile_idx + 1) * self.epi_tile[1]
                        if n_base_d1 < d_n_total:
                            self.store_global_memory_256b(tCgD1, tRS_rD1)
                        if n_base_d2 < d_n_total:
                            self.store_global_memory_256b(tCgD2, tRS_rD2)
                    else:
                        if warp_idx == self.epilog_warp_id[0]:
                            d_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        d1_buffer = num_prev_subtiles % self.num_d_stage
                        num_prev_subtiles = num_prev_subtiles + 1
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD1,
                            tRS_sD[(None, None, None, d1_buffer)],
                        )
                        d2_buffer = num_prev_subtiles % self.num_d_stage
                        num_prev_subtiles = num_prev_subtiles + 1
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD2,
                            tRS_sD[(None, None, None, d2_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_d,
                                bSG_sD[(None, d1_buffer)],
                                bSG_gD[(None, 2 * real_subtile_idx + 0)],
                            )
                            cute.copy(
                                tma_atom_d,
                                bSG_sD[(None, d2_buffer)],
                                bSG_gD[(None, 2 * real_subtile_idx + 1)],
                            )
                            d_pipeline.producer_commit()
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

                if cutlass.const_expr(self.generate_dprob):
                    real_dprob, _ = epi_ext.get_gmem_tensor("dprob", dprob, padded_offsets, epi_work_tile_info)
                    _ = atomic_add_float32(
                        ptr=real_dprob[(mPosition, None, None)].iterator.llvm_ptr,
                        value=dProbVal,
                    )

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for D store complete
            #
            if cutlass.const_expr(not self.store_d_directly):
                d_pipeline.producer_tail()
        #
        # Specialized epilog load warp (loads C from GMEM to SMEM via TMA)
        #
        if warp_idx == self.epilog_load_tma_id:
            c_load_ext = self._make_extension(workspace_ptr)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            c_pipeline_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_c_stage)
            while is_valid_tile:
                c_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                mma_tile_coord_mnl = (
                    c_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    c_work_tile_info.tile_n_idx * 2,
                    cutlass.Int32(0),
                )

                # Per-expert C tensor via extension
                real_c, _ = c_load_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, c_work_tile_info)
                gC_mnl_loop = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                thr_mma_c_load = tiled_mma.get_slice(mma_tile_coord_v)
                tCgC_loop = thr_mma_c_load.partition_C(gC_mnl_loop)

                bGS_sC, bGS_gC_partitioned = self.epilog_gmem_copy_and_partition(tidx, tma_atom_c, tCgC_loop, epi_tile, sC)
                bGS_gC = bGS_gC_partitioned[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                bGS_gC = cute.group_modes(bGS_gC, 1, cute.rank(bGS_gC))
                subtile_cnt = cute.size(bGS_gC.shape, mode=[1])
                for subtile_idx in cutlass.range(subtile_cnt, unroll=1):
                    real_subtile_idx = subtile_idx
                    c_pipeline.producer_acquire(c_pipeline_producer_state)
                    cute.copy(
                        tma_atom_c,
                        bGS_gC[(None, 2 * real_subtile_idx + 0)],
                        bGS_sC[(None, c_pipeline_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                    )
                    c_pipeline_producer_state.advance()
                    c_pipeline.producer_acquire(c_pipeline_producer_state)
                    cute.copy(
                        tma_atom_c,
                        bGS_gC[(None, 2 * real_subtile_idx + 1)],
                        bGS_sC[(None, c_pipeline_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                    )
                    c_pipeline_producer_state.advance()

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
            # Wait C buffer tail complete
            #
            c_pipeline.producer_tail(c_pipeline_producer_state)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source)
        and derive register array shape from the TMEM partition (no gmem dependency).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor in TMEM
            - tTR_rAcc: The register tensor for accumulator (shape derived from TMEM partition)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
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
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_rAcc = thr_copy_t2r.partition_D(tAcc_epi)

        # Derive register shape from TMEM partition (no gmem D needed)
        per_subtile_shape = cute.coalesce(tTR_rAcc[(None, None, None, 0, 0, 0)].layout, target_profile=((1, 1), 1, 1)).shape
        tTR_rAcc = cute.make_rmem_tensor(per_subtile_shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition_load(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tTR_rC1: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory load, then use it to partition register array (destination) and shared memory (source).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_s2r, tSR_rC, tSR_sC) where:
            - tiled_copy_s2r: The tiled copy operation for smem to register copy(s2r)
            - tSR_rC: The partitioned tensor C (register destination)
            - tSR_sC: The partitioned tensor C (smem source)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        # (S2R, S2R_M, S2R_N, PIPE_C)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_D(sC)
        # (S2R, S2R_M, S2R_N)
        tSR_rC = tiled_copy_s2r.retile(tTR_rC)
        tSR_rC1 = tiled_copy_s2r.retile(tTR_rC1)
        return tiled_copy_s2r, tSR_rC, tSR_rC1, tSR_sC

    def epilog_smem_copy_and_partition_store(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rD1: cute.Tensor,
        tTR_rD2: cute.Tensor,
        tidx: cutlass.Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD1: The partitioned accumulator tensor
        :type tTR_rD1: cute.Tensor
        :param tTR_rD2: The partitioned accumulator tensor
        :type tTR_rD2: cute.Tensor
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
        tRS_sD = None
        if cutlass.const_expr(sD is not None):
            tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD1 = tiled_copy_r2s.retile(tTR_rD1)
        tRS_rD2 = tiled_copy_r2s.retile(tTR_rD2)
        return tiled_copy_r2s, tRS_rD1, tRS_rD2, tRS_sD

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_d, bSG_sD, bSG_gD) where:
                - tma_atom_d: The TMA copy atom
                - bSG_sD: The partitioned shared memory tensor D
                - bSG_gD: The partitioned global tensor D
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gD_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_d = atom
        sD_for_tma_partition = cute.group_modes(sD, 0, 2)
        gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_d,
            0,
            cute.make_layout(1),
            sD_for_tma_partition,
            gD_for_tma_partition,
        )
        return bSG_sD, bSG_gD

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        d_dtype: Type[cutlass.Numeric],
        d_layout: utils.LayoutEnum,
        num_smem_capacity: int,
        occupancy: int,
        store_d_directly: bool,
        generate_dbias: bool = False,
    ) -> Tuple[int, int, int]:
        """Compute BF16-only pipeline stages.

        C stage count follows BF16 dGLU tuning; D staging is disabled when the
        direct-store path is selected.
        """
        num_acc_stage = 2
        num_c_stage = 2
        num_d_stage = 0 if store_d_directly else 2
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            d_dtype,
            d_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        d_bytes = d_bytes_per_stage * num_d_stage
        # dBias transpose buffer: (128, 64) column-major FP32 = 32 KB
        dbias_bytes = 128 * 64 * cute.size_in_bytes(cutlass.Float32, cute.make_layout((1,))) if generate_dbias else 0
        epi_bytes = c_bytes + d_bytes + dbias_bytes

        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage
