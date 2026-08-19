# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
MoE Block-Scaled Grouped GEMM Kernel with GLU (SwiGLU/GeGLU) + Hadamard Transform Fusion.

Supports:
    - Static / Dynamic persistent tile scheduling (MoEPersistentTileScheduler)
    - Dense (contiguous 3-D B) / Discrete (per-expert pointer array B) weight layout
    - BF16 or NVFP4 (packed e2m1 + e4m3 block scales) D output with GLU activation
    - Optional C output (pre-activation GLU output)
    - Optional fused RHT output (bf16 or NVFP4 + e4m3 block scales)
    - GLU activation fusion (SwiGLU / GeGLU)

Warp assignment (8 epilogue warps, pingpong):
    warps 0-3  : ACT warps  — TMEM→reg, alpha scale, GLU activation, C/D store
    warps 4-7  : RHT store warps — RHT and/or NVFP4 quantization from D SMEM
    warp  8    : MMA warp
    warp  9    : TMA load warp
    warp  10   : Scheduler warp (MoEPersistentTileScheduler)
    warp  11   : Bias load warp (optional)

sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
    Validity: tile_info[0] >= 0  (expert_idx == -1 signals end)
"""

from typing import Type, Tuple, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import Float32, Int32, AddressSpace
from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import (
    compute_expert_token_range,
    MoEWeightMode,
    TensormapWorkspace,
    store_tma_desc,
)
from .rht_utils import (
    hadamard_rmem_colwise_fwht,
    hadamard_rmem_colwise_fwht_quant,
    hadamard_rmem_rowwise_fwht,
    load_colwise_pairs_bf16,
    HADAMARD_SIZE,
)
from .quant_utils import load_row_bf16, nvfp4_quant_rmem_row
from ..moe_sched_extension import (
    DiscreteWeightScaledGemmSchedExtension,
    ContiguousAndConsistentGroupedGemmSchedExtension,
)
from .moe_kernel_helpers import (
    fmin,
    fmax,
    silu_f32,
    silu_f32_geglu_scaled,
    compute_grid,
    can_implement,
)


class BlockScaledMoEGroupedGemmGluHadamardQuantKernel:
    """Block-scaled MoE grouped GEMM with GLU activation and Hadamard transform fusion.

    Always uses pingpong epilogue (8 epilogue warps: 4 ACT + 4 RHT-store).
    D output is BF16 or NVFP4 (packed e2m1 plus swizzled e4m3 SFD).

    :param sf_vec_size: Scalefactor vector size.
    :param mma_tiler_mn: Shape of MMA tile (M, N).
    :param cluster_shape_mn: Cluster dimensions (M, N).
    :param expert_cnt: Number of experts (compile-time constant).
    :param weight_mode: Dense or Discrete weight layout.
    :param use_dynamic_sched: Use dynamic tile scheduling.
    :param act_func: Activation function ('swiglu', 'geglu', or 'srelu').
    :param enable_bias: Enable bias addition.
    """

    FIX_PAD_SIZE = 256

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
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
        rht_quant: bool = False,
        d_quant: bool = False,
    ) -> bool:
        # speical requirements for hadamard fusion
        if not use_2cta_instrs or mma_tiler_mn[0] != 256 or mma_tiler_mn[1] != 256:
            return False
        # NVFP4 quantization: scale-byte rows are stored as one contiguous 8-byte
        # store per thread, so f = n/2 must be divisible by 128 (f/16 % 8 == 0;
        # same shape gate as the standalone group_rht_cast kernel).
        if (rht_quant or d_quant) and (n // 2) % 128 != 0:
            return False
        return can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            acc_dtype,
            d_dtype,
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
            fix_pad_size=BlockScaledMoEGroupedGemmGluHadamardQuantKernel.FIX_PAD_SIZE,
        )

    def __init__(
        self,
        sf_vec_size: int,
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
        rht_rowwise: bool = False,
        glu_alpha: Optional[float] = None,
        glu_limit: Optional[float] = None,
    ):
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % mma_tile_m != 0:
            raise ValueError(f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by " f"mma_tiler_mn[0] ({mma_tile_m}).")
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")

        self.sf_vec_size = sf_vec_size
        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.weight_mode = weight_mode
        self.use_dynamic_sched = use_dynamic_sched
        self.enable_bias = enable_bias
        # RHT dump orientation: False = columnwise (16-token blocks per feature),
        # True = rowwise (16-feature blocks per token). Same dump tensor/path either way.
        self.rht_rowwise = rht_rowwise

        # Always use pingpong epilogue for Hadamard
        self.epilogue_pingpong = True
        # Always delay TMA store acquire sync for Hadamard
        self.delay_tma_store_acquire_sync = True

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.threads_per_warp = 32

        # Warp assignments: 8 epilogue warps (4 ACT + 4 RHT-store)
        self.epilog_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
        self.epilog_act_warp_id = (0, 1, 2, 3)
        self.epilog_rht_store_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.bias_load_warp_id = 11 if enable_bias else None

        self.epilogue_warp_group_size = len(self.epilog_act_warp_id)  # = 4

        all_warps = [*self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id, self.sched_warp_id]
        warps_wo_sched = [*self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id]
        if enable_bias:
            all_warps.append(self.bias_load_warp_id)
            warps_wo_sched.append(self.bias_load_warp_id)
        self.threads_per_cta = self.threads_per_warp * len(all_warps)
        self.threads_wo_sched = self.threads_per_warp * len(warps_wo_sched)

        # Named barriers
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        tmem_alloc_warp_ids = self.epilog_act_warp_id
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *tmem_alloc_warp_ids)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        # Pingpong barriers (group 0 = ACT warps, group 1 = RHT store warps)
        self.epilog_sync_barrier_group0 = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=32 * self.epilogue_warp_group_size,
        )
        self.epilog_sync_barrier_group1 = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=32 * self.epilogue_warp_group_size,
        )

        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.vectorized_f32 = vectorized_f32

        self.act_func = act_func
        if act_func not in ["swiglu", "geglu", "srelu"]:
            raise ValueError(f"Invalid activation function: {act_func}")

        self.glu_alpha = glu_alpha
        self.glu_limit = glu_limit

    def _setup_attributes(self):
        """Set up configurations dependent on GEMM inputs (called inside __call__)."""

        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
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

        d_tile_n = self.mma_inst_shape_mn[1] if self.act_func == "srelu" else self.mma_inst_shape_mn[1] // 2
        self.mma_tiler_d = (
            self.mma_inst_shape_mn[0],
            d_tile_n,
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk_d = (
            self.mma_tiler_d[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_d[1],
            self.mma_tiler_d[2],
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

        self.epi_tile = (128, 32)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_d[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_d[1] // self.epi_tile[1],
        )
        self.epi_tile_c = self.epi_tile if self.act_func == "srelu" else (128, 64)

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_d_stage,
            self.num_tile_stage,
            self.num_bias_stage,
            self.num_pingpong_stage,
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
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.occupancy,
            self.bias_dtype if self.enable_bias else None,
            self.rht_dtype if self.generate_rht else None,
            self.rht_quant,
            self.d_quant,
        )

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
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile_c,
            self.num_c_stage,
        )
        # sD staging keeps bf16 when D itself is quantized to NVFP4 — the RHT warps
        # consume bf16 D rows; the fp4 TMA source is the separate sDq buffer whose
        # layout follows the gmem dtype. Without quantization the two layouts coincide.
        self.d_smem_dtype = cutlass.BFloat16 if self.d_quant else self.d_dtype
        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_smem_dtype,
            self.d_layout,
            self.epi_tile,
            self.num_d_stage,
        )
        self.d_tma_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_dtype,
            self.d_layout,
            self.epi_tile,
            self.num_d_stage,
        )

        if self.enable_bias:
            self.bias_smem_layout_staged = cute.make_layout(
                (self.mma_tiler[1], self.num_bias_stage),
                stride=(1, self.mma_tiler[1]),
            )
        else:
            self.bias_smem_layout_staged = cute.make_layout((1, 1))

        self.overlapping_accum = self.num_acc_stage == 1 and self.mma_tiler[1] == 256

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_reserved_tmem_cols = self.num_sf_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_reserved_tmem_cols
        )

        self.epi_tile_n_required = cute.size(self.epi_tile[1]) if self.act_func == "srelu" else 2 * cute.size(self.epi_tile[1])
        self.iter_acc_early_release_in_epilogue = (self.num_reserved_tmem_cols + self.epi_tile_n_required - 1) // self.epi_tile_n_required - 1
        if self.act_func != "srelu":
            self.iter_acc_early_release_in_epilogue = self.iter_acc_early_release_in_epilogue * 2

    def get_desc_workspace_bytes(self) -> int:
        """Return descriptor workspace size in bytes."""
        if self.weight_mode == MoEWeightMode.DISCRETE:
            from ..moe_utils import DiscreteWeightTensormapConstructor

            return DiscreteWeightTensormapConstructor.get_workspace_size(self.expert_cnt)
        return 0

    def get_workspace_bytes(self) -> int:
        """Return total workspace size in bytes."""
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
        ptrs_sfb: cute.Pointer,
        n: Int32,
        k: Int32,
        b_stride_size: cutlass.Int64,
        b_major_mode: cutlass.Constexpr,
        workspace_ptr,
        tiled_mma_arg: cute.TiledMma,
        tiled_mma_sfb_arg: cute.TiledMma,
        b_smem_layout_arg,
        sfb_smem_layout_arg,
        cluster_layout_vmnk_shape_arg: cutlass.Constexpr,
        cluster_layout_sfb_vmnk_shape_arg: cutlass.Constexpr,
    ):
        """Pre-main-kernel: build per-expert TMA descriptors (discrete mode) and/or reset sched counter."""
        expert_idx = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            b_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma_arg.thr_id)
            sfb_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma_arg.thr_id)

            # Read per-expert base addresses from the pointer arrays
            b_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_b.toint(), AddressSpace.gmem, assumed_align=8),
                cute.make_layout((self.expert_cnt,)),
            )
            sfb_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_sfb.toint(), AddressSpace.gmem, assumed_align=8),
                cute.make_layout((self.expert_cnt,)),
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
            b_expert = cute.make_tensor(
                b_ptr,
                cute.make_layout((n, k, c1), stride=(stride_n, stride_k, c0)),
            )
            tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_tma_op_arg,
                b_expert,
                b_smem_layout_arg,
                self.mma_tiler,
                tiled_mma_arg,
                cluster_layout_vmnk_shape_arg,
            )

            workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            store_tma_desc(tma_atom_b, workspace.get_ptr("b", expert_idx))

            sfb_ptr_val = sfb_ptr_tensor[expert_idx]
            sfb_ptr = cute.make_ptr(self.sf_dtype, sfb_ptr_val, AddressSpace.gmem)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb_expert = cute.make_tensor(sfb_ptr, sfb_layout)
            tma_atom_sfb, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_tma_op_arg,
                sfb_expert,
                sfb_smem_layout_arg,
                self.mma_tiler_sfb,
                tiled_mma_sfb_arg,
                cluster_layout_sfb_vmnk_shape_arg,
                internal_type=cutlass.Uint64,
            )
            store_tma_desc(tma_atom_sfb, workspace.get_ptr("sfb", expert_idx))

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
        sfa: cute.Tensor,
        sfb,  # Dense: cute.Tensor         | Discrete: cute.Pointer to int64[]
        n: Int32,  # Ignored for dense mode
        k: Int32,  # Ignored for dense mode
        b_stride_size: cutlass.Int64,  # Ignored for dense mode
        b_major_mode: cutlass.Constexpr,  # Ignored for dense mode
        workspace_ptr,
        c: cute.Tensor,
        d: cute.Tensor,  # post-GLU output (bf16, or NVFP4 packed e2m1 with sfd)
        sfd: Optional[cute.Tensor],  # NVFP4 D block scales (e4m3, swizzled SF layout); required iff d is NVFP4
        rht: Optional[cute.Tensor],  # RHT output (bf16 or NVFP4, D layout); None => off
        sfrht: Optional[cute.Tensor],  # NVFP4 RHT block scales (e4m3, swizzled SF layout); required iff rht is NVFP4
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        prob: cute.Tensor,
        bias: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        linear_offset: cutlass.Float32 = 0.0,
        norm_const: cutlass.Float32 = 1.0,  # D NVFP4 global encode scale: 2688/global_amax, or 1.0
        rht_norm_const: cutlass.Float32 = 1.0,  # RHT NVFP4 global encode scale: 2688/global_amax, or 1.0
    ):
        """Execute the MoE GEMM + GLU + Hadamard kernel.

        Dense mode: ``b`` and ``sfb`` are 3-D cute.Tensor (N, K, L).
        Discrete mode: ``b`` and ``sfb`` are cute.Pointer to device int64[]
        arrays of per-expert base addresses.
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
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

        # ---- Output / dump modes (derived from dtypes; before _setup_attributes so
        # the smem/stage accounting sees them) ----
        self.generate_rht = rht is not None
        self.generate_sfd = sfd is not None
        self.generate_sfrht = sfrht is not None
        self.d_quant = self.d_dtype == cutlass.Float4E2M1FN
        self.rht_dtype = rht.element_type if cutlass.const_expr(self.generate_rht) else self.d_dtype
        self.rht_quant = self.generate_rht and self.rht_dtype == cutlass.Float4E2M1FN
        # The rht STORE has no orientation logic anywhere in this kernel; only the
        # SCALE tensor is orientation-aware below: rowwise uses swizzled SF for
        # logical (m, f), while colwise uses swizzled SF for logical (f, m).
        # RHT warps run for the RHT output and/or the D quantization (both read sD).
        # When neither is on (plain bf16 mode) the warpgroup is COMPILED OUT: no
        # pingpong, no ACT<->RHT barriers, and the tile-info pipeline's consumer
        # count shrinks by the 4 RHT warps (forget that and the scheduler deadlocks
        # after num_tile_stage tiles).
        self.run_rht = self.generate_rht or self.d_quant
        if cutlass.const_expr(not self.run_rht):
            self.threads_wo_sched = self.threads_per_cta - self.threads_per_warp - self.threads_per_warp * len(self.epilog_rht_store_warp_id)
        if cutlass.const_expr(self.d_dtype not in (cutlass.BFloat16, cutlass.Float4E2M1FN)):
            raise ValueError(f"d dtype must be BFloat16 or Float4E2M1FN, got {self.d_dtype}")
        if cutlass.const_expr(self.d_quant != self.generate_sfd):
            raise ValueError("NVFP4 d and sfd must be passed together")
        if cutlass.const_expr(self.generate_rht and self.rht_dtype not in (cutlass.BFloat16, cutlass.Float4E2M1FN)):
            raise ValueError(f"rht dtype must be BFloat16 or Float4E2M1FN, got {self.rht_dtype}")
        if cutlass.const_expr(self.rht_quant != self.generate_sfrht):
            raise ValueError("NVFP4 rht and sfrht must be passed together")
        sf_storage_dtype = cutlass.Float8E4M3FN if self.sf_dtype == cutlass.FloatNV8E5M3FNU else self.sf_dtype
        if cutlass.const_expr(self.generate_sfd and sfd.element_type != sf_storage_dtype):
            raise ValueError("sfd element type must match scale-factor storage dtype")
        if cutlass.const_expr(self.generate_sfrht and sfrht.element_type != sf_storage_dtype):
            raise ValueError("sfrht element type must match scale-factor storage dtype")
        if cutlass.const_expr((self.d_quant or self.rht_quant) and self.act_func == "srelu"):
            raise ValueError("NVFP4 quantization assumes the GLU subtile pair-step (act_func != srelu)")

        self._setup_attributes()

        # ---- B / SFB setup (mode-dependent) ----
        b_from_call_arg = b
        sfb_from_call_arg = sfb
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
            sfb = cute.make_tensor(sfb.iterator, sfb_layout)
        else:
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

            sfb_ptr_typed = cute.make_ptr(self.sf_dtype, sfb.toint(), AddressSpace.gmem, assumed_align=16)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb = cute.make_tensor(sfb_ptr_typed, sfb_layout)

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        # Dump staging dtype follows the dump tensor's element type (fp4: 2KB/stage vs
        # 8KB bf16); the layout is CONSTRUCTED f-major like every other epilogue
        # output (never derived from the rht gmem tensor) — all FWHT store paths
        # pack along features.
        self.rht_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.rht_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.epi_tile,
            self.num_d_stage,
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA load A
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

        # TMA load B
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

        # TMA load SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # TMA load SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator,
                cute.make_layout(new_shape, stride=new_stride),
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        # TMA store C
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            c_smem_layout,
            self.epi_tile_c,
        )

        # TMA store D (gmem-dtype staging: sDq when D is quantized, sD otherwise)
        d_smem_layout = cute.slice_(self.d_tma_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d,
            d_smem_layout,
            self.epi_tile,
        )

        # TMA store RHT — identical tile to D; smem layout follows the RHT output
        # element type (== d_smem_layout for bf16, packed fp4 layout in quant mode).
        if cutlass.const_expr(self.generate_rht):
            rht_smem_layout = cute.slice_(self.rht_smem_layout_staged, (None, None, 0))
            tma_atom_rht, tma_tensor_rht = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                rht,
                rht_smem_layout,
                self.epi_tile,
            )
        else:
            tma_atom_rht, tma_tensor_rht = None, None

        # ---- Helper kernel (discrete TMA desc init + dynamic sched counter reset) ----
        _need_helper = cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE or self.use_dynamic_sched)
        if cutlass.const_expr(_need_helper):
            _helper_grid_x = self.expert_cnt if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else 1
            _helper_args = (
                b_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem),
                sfb_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem),
                n if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                k if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                b_stride_size if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int64(0),
                b_major_mode if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else self.b_major_mode,
                workspace_ptr,
                tiled_mma,
                tiled_mma_sfb,
                b_smem_layout,
                sfb_smem_layout,
                self.cluster_layout_vmnk.shape,
                self.cluster_layout_sfb_vmnk.shape,
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

        # ---- Shared storage ----
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            pingpong_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_pingpong_stage * 2]
            if cutlass.const_expr(self.enable_bias):
                bias_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_bias_stage * 2]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_smem_dtype, cute.cosize(self.d_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            if cutlass.const_expr(self.d_quant):
                # NVFP4 D: packed-fp4 TMA staging + per-thread scale-byte staging.
                sDq: cute.struct.Align[
                    cute.struct.MemRange[self.d_dtype, cute.cosize(self.d_tma_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
                sSfd: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype,
                        self.threads_per_warp * len(self.epilog_rht_store_warp_id) * (self.cta_tile_shape_mnk_d[1] // HADAMARD_SIZE),
                    ],
                    16,
                ]
            if cutlass.const_expr(self.generate_rht):
                sRht: cute.struct.Align[
                    cute.struct.MemRange[self.rht_dtype, cute.cosize(self.rht_smem_layout_staged.outer)],
                    self.buffer_align_bytes,
                ]
            if cutlass.const_expr(self.rht_quant):
                # NVFP4 RHT scale-byte staging, one contiguous row store per thread
                # per tile: rowwise rows are (thread=token, cta_tile_n/16 scales);
                # colwise rows are (feature-in-tile, 128-token-tile/16).
                # Same (128, 8) buffer either way.
                sSfRht: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype,
                        self.threads_per_warp * len(self.epilog_rht_store_warp_id) * (self.cta_tile_shape_mnk_d[1] // HADAMARD_SIZE),
                    ],
                    16,
                ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            if cutlass.const_expr(self.enable_bias):
                sBias: cute.struct.Align[
                    cute.struct.MemRange[self.bias_dtype, cute.cosize(self.bias_smem_layout_staged)],
                    16,
                ]

        self.shared_storage = SharedStorage

        # Launch main kernel
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
            tma_atom_d,
            tma_tensor_d,
            sfd,
            tma_atom_rht,
            tma_tensor_rht,
            sfrht,
            padded_offsets,
            alpha,
            bias,
            prob,
            workspace_ptr,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.d_smem_layout_staged,
            self.d_tma_smem_layout_staged,
            self.rht_smem_layout_staged,
            self.bias_smem_layout_staged,
            self.epi_tile,
            self.sched_params,
            epilogue_op,
            linear_offset,
            norm_const,
            rht_norm_const,
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
    # Internal helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _make_extension(self, workspace_ptr):
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            desc_workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            return DiscreteWeightScaledGemmSchedExtension(
                tensormap_ctor=desc_workspace,
                sf_vec_size=self.sf_vec_size,
            )
        else:
            return ContiguousAndConsistentGroupedGemmSchedExtension(
                sf_vec_size=self.sf_vec_size,
            )

    @cute.jit
    def store_swizzled_sf_row(self, sf_tensor: cute.Tensor, logical_row, sf_col_base, sSf: cute.Tensor, tidx):
        """Store one logical scale row into M32x4xrm_K4xrk_L SF layout."""
        sf_tensor = cute.recast_tensor(sf_tensor, self.sf_dtype)
        row_m0 = logical_row % 32
        row_m1 = (logical_row // 32) % 4
        row_m2 = logical_row // 128
        num_sf = self.cta_tile_shape_mnk_d[1] // HADAMARD_SIZE
        for vi in cutlass.range_constexpr(num_sf):
            sf_col = sf_col_base + vi
            sf_tensor[(row_m0, row_m1, row_m2, sf_col % 4, sf_col // 4, 0)] = sSf[(tidx, vi)]

    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

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
    ):
        c_buffer = prev_subtile_idx % self.num_c_stage
        tRS_rC.store(tTR_rAcc.load().to(self.c_dtype))
        cute.copy(tiled_copy_r2s, tRS_rC[(None, None, 0)], tRS_sC[(None, None, 0, c_buffer)])
        tRS_rC.store(tTR_rAcc_up.load().to(self.c_dtype))
        cute.copy(tiled_copy_r2s, tRS_rC[(None, None, 0)], tRS_sC[(None, None, 1, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        self.epilog_sync_barrier_group0.arrive_and_wait()
        if warp_idx == self.epilog_act_warp_id[0]:
            cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)])
            c_pipeline.producer_commit()
            if not cutlass.const_expr(self.delay_tma_store_acquire_sync):
                c_pipeline.producer_acquire()
        if not cutlass.const_expr(self.delay_tma_store_acquire_sync):
            self.epilog_sync_barrier_group0.arrive_and_wait()

    @cute.jit
    def store_c_unary(
        self,
        tiled_copy_r2s,
        tma_atom_c,
        warp_idx,
        tTR_rAcc,
        tRS_rC,
        tRS_sC,
        bSG_gC,
        bSG_sC,
        c_pipeline,
        prev_subtile_idx,
        real_subtile_idx,
    ):
        c_buffer = prev_subtile_idx % self.num_c_stage
        tRS_rC.store(tTR_rAcc.load().to(self.c_dtype))
        cute.copy(tiled_copy_r2s, tRS_rC[(None, None, 0)], tRS_sC[(None, None, 0, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        self.epilog_sync_barrier_group0.arrive_and_wait()
        if warp_idx == self.epilog_act_warp_id[0]:
            cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)])
            c_pipeline.producer_commit()
            if not cutlass.const_expr(self.delay_tma_store_acquire_sync):
                c_pipeline.producer_acquire()
        if not cutlass.const_expr(self.delay_tma_store_acquire_sync):
            self.epilog_sync_barrier_group0.arrive_and_wait()

    @cute.jit
    def geglu_act(self, tCompute, acc_vec_up, acc_vec_gate, mProb, linear_offset=1.0):
        if cutlass.const_expr(self.vectorized_f32):
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
                tCompute[i], tCompute[i + 1] = cute.arch.add_packed_f32x2(
                    (cute.math.exp2(tCompute_log2e[0], fastmath=True), cute.math.exp2(tCompute_log2e[1], fastmath=True)),
                    (1.0, 1.0),
                )
                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                up0, up1 = cute.arch.add_packed_f32x2(
                    (linear_offset, linear_offset),
                    (acc_vec_up[i], acc_vec_up[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (up0, up1),
                    rnd="rn",
                    ftz=False,
                )
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (mProb, mProb),
                    rnd="rn",
                    ftz=False,
                )
                if cutlass.const_expr(self.glu_alpha is not None and self.glu_alpha != 1.0):
                    (
                        tCompute[i],
                        tCompute[i + 1],
                    ) = cute.arch.mul_packed_f32x2(
                        (tCompute[i], tCompute[i + 1]),
                        (self.glu_alpha, self.glu_alpha),
                        rnd="rn",
                        ftz=False,
                    )
        else:
            # GeGlu Unpacked Version
            for i in cutlass.range_constexpr(cute.size(tCompute)):
                tCompute[i] = (acc_vec_up[i] + linear_offset) * silu_f32_geglu_scaled(acc_vec_gate[i], fastmath=True)
                tCompute[i] = tCompute[i] * mProb
                if cutlass.const_expr(self.glu_alpha is not None and self.glu_alpha != 1.0):
                    tCompute[i] = tCompute[i] * self.glu_alpha
        # + 0.0 canonicalizes -0 -> +0 (negative up x saturated-to-zero sigmoid). SCALAR
        # on purpose: the f32 immediate makes it a free FADD with RZ, while f32x2 has no
        # immediate form and a live (0, 0) register pair costs regs/spills.
        for i in cutlass.range_constexpr(cute.size(tCompute)):
            tCompute[i] = tCompute[i] + cutlass.Float32(0.0)

    @cute.jit
    def swiglu_act(self, tCompute, acc_vec_up, acc_vec_gate, mProb):
        if cutlass.const_expr(self.vectorized_f32):
            LOG2_E = cutlass.Float32(1.4426950408889634)
            for i in cutlass.range_constexpr(0, cute.size(tCompute), 2):
                tCompute_log2e = cute.arch.mul_packed_f32x2(
                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                    (-LOG2_E, -LOG2_E),
                    rnd="rn",
                    ftz=False,
                )
                tCompute[i], tCompute[i + 1] = cute.arch.add_packed_f32x2(
                    (cute.math.exp2(tCompute_log2e[0], fastmath=True), cute.math.exp2(tCompute_log2e[1], fastmath=True)),
                    (1.0, 1.0),
                )
                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (acc_vec_up[i], acc_vec_up[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (mProb, mProb),
                    rnd="rn",
                    ftz=False,
                )
                if cutlass.const_expr(self.glu_alpha is not None and self.glu_alpha != 1.0):
                    (
                        tCompute[i],
                        tCompute[i + 1],
                    ) = cute.arch.mul_packed_f32x2(
                        (tCompute[i], tCompute[i + 1]),
                        (self.glu_alpha, self.glu_alpha),
                        rnd="rn",
                        ftz=False,
                    )
        else:
            # SwiGlu Unpacked Version
            for i in cutlass.range_constexpr(cute.size(tCompute)):
                tCompute[i] = acc_vec_up[i] * silu_f32(acc_vec_gate[i], fastmath=True)
                tCompute[i] = tCompute[i] * mProb
                if cutlass.const_expr(self.glu_alpha is not None and self.glu_alpha != 1.0):
                    tCompute[i] = tCompute[i] * self.glu_alpha
        # + 0.0 canonicalizes -0 -> +0 (negative up x saturated-to-zero sigmoid). SCALAR
        # on purpose: the f32 immediate makes it a free FADD with RZ, while f32x2 has no
        # immediate form and a live (0, 0) register pair costs regs/spills.
        for i in cutlass.range_constexpr(cute.size(tCompute)):
            tCompute[i] = tCompute[i] + cutlass.Float32(0.0)

    @cute.jit
    def srelu_act(self, tCompute, acc_vec, mProb):
        acc_relu = cute.where(acc_vec > 0, acc_vec, cute.full_like(acc_vec, 0))
        if cutlass.const_expr(self.vectorized_f32):
            for i in cutlass.range_constexpr(0, cute.size(tCompute), 2):
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (acc_relu[i], acc_relu[i + 1]),
                    (acc_relu[i], acc_relu[i + 1]),
                    rnd="rn",
                    ftz=False,
                )
                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                    (tCompute[i], tCompute[i + 1]),
                    (mProb, mProb),
                    rnd="rn",
                    ftz=False,
                )
        else:
            for i in cutlass.range_constexpr(cute.size(tCompute)):
                tCompute[i] = acc_relu[i] * acc_relu[i] * mProb

    def epilog_tmem_copy_and_partition(self, tidx, tAcc, gD_mnl, epi_tile, use_2cta_instrs):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gD_mnl_epi)
        tTR_rAcc_gate = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        tTR_rAcc_up = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_gate, tTR_rAcc_up

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sD):
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, sD.element_type, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        tRS_rD = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_gmem_copy_and_partition(self, tidx, atom, gD_mnl, epi_tile, sD):
        gD_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_d = atom
        sD_for_tma_partition = cute.group_modes(sD, 0, 2)
        gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_d,
            0,
            cute.make_layout(1),
            sD_for_tma_partition,
            gD_for_tma_partition,
        )
        return tma_atom_d, bSG_sD, bSG_gD

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
        sf_dtype,
        sf_vec_size,
        num_smem_capacity,
        occupancy,
        bias_dtype,
        rht_dtype,  # RHT output dtype (None => no RHT output)
        rht_quant,  # RHT output is NVFP4 (adds sfrht scale-byte staging)
        d_quant,  # D output is NVFP4 (sD staging stays bf16; adds fp4 + sfd staging)
    ):
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_c_stage = 1
        num_d_stage = 1
        num_tile_stage = 2
        num_pingpong_stage = mma_tiler_mnk[1] // epi_tile_c[1]

        a_smem_layout_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_smem_layout_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_smem_layout_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        c_smem_layout_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile_c, 1)
        d_smem_layout_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_one)
        )
        mbar_helpers_bytes = 1024

        # One e4m3 scale byte per (1,16) feature block, one row per thread (128 threads).
        quant_sf_bytes = 128 * ((mma_tiler_mnk[1] // 2) // HADAMARD_SIZE)

        # sInfo is in SchedulerStorage, not here, so use 4-int sInfo
        sinfo_bytes = 4 * 4 * num_tile_stage
        c_bytes = cute.size_in_bytes(c_dtype, c_smem_layout_one) * num_c_stage
        d_bytes = cute.size_in_bytes(d_dtype, d_smem_layout_one) * num_d_stage

        if d_quant:
            # sD staging stays bf16 (the RHT/quant warps' source); the gmem-dtype (fp4)
            # staging above becomes the TMA source, and sfd rows are staged per thread.
            bf16_smem_layout_one = sm100_utils.make_smem_layout_epi(cutlass.BFloat16, d_layout, epi_tile, 1)
            d_bytes += cute.size_in_bytes(cutlass.BFloat16, bf16_smem_layout_one) * num_d_stage
            d_bytes += quant_sf_bytes

        rht_bytes = 0
        if rht_dtype is not None:
            rht_smem_layout_one = sm100_utils.make_smem_layout_epi(rht_dtype, d_layout, epi_tile, 1)
            rht_bytes = cute.size_in_bytes(rht_dtype, rht_smem_layout_one) * num_d_stage
        if rht_quant:
            rht_bytes += quant_sf_bytes

        if bias_dtype is not None:
            num_bias_stage = 2
            bias_bytes = mma_tiler_mnk[1] * num_bias_stage * (bias_dtype.width // 8)
        else:
            num_bias_stage = 0
            bias_bytes = 0

        epi_bytes = c_bytes + d_bytes + rht_bytes + bias_bytes

        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage, num_bias_stage, num_pingpong_stage

    # GPU device kernel
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
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        mSfd_mnl: Optional[cute.Tensor],
        tma_atom_rht: Optional[cute.CopyAtom],
        mRht_mnl: Optional[cute.Tensor],
        mSfRht_mnl: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        mBias_nl: Optional[cute.Tensor],
        prob: cute.Tensor,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_tma_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        rht_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        bias_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
        epilogue_op: cutlass.Constexpr,
        linear_offset: cutlass.Float32 = 0.0,
        norm_const: cutlass.Float32 = 1.0,
        rht_norm_const: cutlass.Float32 = 1.0,
    ):
        """GPU device kernel: MoE persistent GEMM + GLU + Hadamard (pingpong epilogue)."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        total_token = padded_offsets[self.expert_cnt - 1]

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_d)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # CTA coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Shared memory allocation
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        # AB pipeline
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

        # ACC pipeline
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_act_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Pingpong pipeline
        pingpong_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.epilog_act_warp_id) * self.threads_per_warp)
        pingpong_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.epilog_rht_store_warp_id) * self.threads_per_warp)
        pingpong_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.pingpong_mbar_ptr.data_ptr(),
            num_stages=self.num_pingpong_stage,
            producer_group=pingpong_producer_group,
            consumer_group=pingpong_consumer_group,
        )

        # Tile info pipeline (uses SchedulerStorage's barrier)
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

        # MoE persistent tile scheduler
        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            padded_offsets,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr),
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        # Bias pipeline
        if cutlass.const_expr(self.enable_bias):
            bias_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp,
            )
            bias_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_act_warp_id),
            )
            bias_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.bias_mbar_ptr.data_ptr(),
                num_stages=self.num_bias_stage,
                producer_group=bias_pipeline_producer_group,
                consumer_group=bias_pipeline_consumer_group,
            )
            sBias = storage.sBias.get_tensor(bias_smem_layout_staged)
            gBias_nl = cute.local_tile(mBias_nl, cute.slice_(self.mma_tiler[:2], (0, None)), (None, None))

        # TMEM allocator
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_act_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # SMEM tensors
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        _num_sf_per_tile = self.cta_tile_shape_mnk_d[1] // HADAMARD_SIZE
        _sf_stage_layout = cute.make_layout(
            (self.threads_per_warp * len(self.epilog_rht_store_warp_id), _num_sf_per_tile),
            stride=(_num_sf_per_tile, 1),
        )
        if cutlass.const_expr(self.d_quant):
            sDq = storage.sDq.get_tensor(d_tma_smem_layout_staged.outer, swizzle=d_tma_smem_layout_staged.inner)
            sSfd = storage.sSfd.get_tensor(_sf_stage_layout)
        if cutlass.const_expr(self.generate_rht):
            sRht = storage.sRht.get_tensor(rht_smem_layout_staged.outer, swizzle=rht_smem_layout_staged.inner)
        if cutlass.const_expr(self.rht_quant):
            sSfRht = storage.sSfRht.get_tensor(_sf_stage_layout)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # sInfo from SchedulerStorage
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        # Multicast masks
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        # MMA fragments
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage_overlapped))
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_reserved_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait / CTA sync
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        if total_token <= 0:
            cute.arch.nvvm.exit()

        # ---------------------------------------------------------------
        # Specialized Scheduler warp (MoEPersistentTileScheduler)
        # ---------------------------------------------------------------
        if warp_idx == self.sched_warp_id:
            work_tile_info = scheduler.initial_work_tile_info()
            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)

            while work_tile_info.is_valid_tile:
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

            # Send invalid signal: expert_idx = -1
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

        # ---------------------------------------------------------------
        # Specialized TMA load warp
        # ---------------------------------------------------------------
        if warp_idx == self.tma_warp_id:
            ext = self._make_extension(workspace_ptr)

            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

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
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, padded_offsets, work_tile_info)
                real_b, desc_ptr_b = ext.get_gmem_tensor("b", mB_nkl, padded_offsets, work_tile_info)
                real_sfa, _ = ext.get_gmem_tensor("sfa", mSFA_mkl, padded_offsets, work_tile_info)
                real_sfb, desc_ptr_sfb = ext.get_gmem_tensor("sfb", mSFB_nkl, padded_offsets, work_tile_info)

                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                gSFA_mkl = cute.local_tile(real_sfa, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gSFB_nkl = cute.local_tile(real_sfb, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))

                thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
                tCgA = thr_mma.partition_A(gA_mkl)
                tCgB = thr_mma.partition_B(gB_nkl)
                tCgSFA = thr_mma.partition_A(gSFA_mkl)
                tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
                sfa_cta_layout = a_cta_layout
                tAsSFA, tAgSFA = cpasync.tma_partition(
                    tma_atom_sfa,
                    block_in_cluster_coord_vmnk[2],
                    sfa_cta_layout,
                    cute.group_modes(sSFA, 0, 3),
                    cute.group_modes(tCgSFA, 0, 3),
                )
                tAsSFA = cute.filter_zeros(tAsSFA)
                tAgSFA = cute.filter_zeros(tAgSFA)
                sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
                tBsSFB, tBgSFB = cpasync.tma_partition(
                    tma_atom_sfb,
                    block_in_cluster_coord_sfb_vmnk[1],
                    sfb_cta_layout,
                    cute.group_modes(sSFB, 0, 3),
                    cute.group_modes(tCgSFB, 0, 3),
                )
                tBsSFB = cute.filter_zeros(tBsSFB)
                tBgSFB = cute.filter_zeros(tBgSFB)

                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_m, None, 0)]
                slice_n = mma_tile_coord_n
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_n // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, 0)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAgSFA_k = tAgSFA_slice[(None, ab_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, ab_producer_state.index)]
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    ab_producer_state_next = ab_producer_state.clone()
                    ab_producer_state_next.advance()
                    if ab_producer_state_next.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state_next)
                    else:
                        peek_ab_empty_status = cutlass.Boolean(1)

                    cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar, mcast_mask=a_full_mcast_mask)
                    cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=tma_bar, mcast_mask=b_full_mcast_mask, tma_desc_ptr=desc_ptr_b)
                    cute.copy(tma_atom_sfa, tAgSFA_k, tAsSFA_pipe, tma_bar_ptr=tma_bar, mcast_mask=sfa_full_mcast_mask)
                    cute.copy(tma_atom_sfb, tBgSFB_k, tBsSFB_pipe, tma_bar_ptr=tma_bar, mcast_mask=sfb_full_mcast_mask, tma_desc_ptr=desc_ptr_sfb)

                    ab_producer_state.advance()

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            ab_pipeline.producer_tail(ab_producer_state)

        # ---------------------------------------------------------------
        # Specialized MMA warp
        # ---------------------------------------------------------------
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acd_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

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
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                acd_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                mma_tile_coord_mnl = (
                    tile_info[1] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[2],
                    tile_info[0],
                )

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acd_producer_state.phase ^ 1
                else:
                    acc_stage_index = acd_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acd_producer_state, peek_acc_empty_status)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if ab_consumer_state_next.count < k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state = ab_consumer_state_next

                if is_leader_cta:
                    acc_pipeline.producer_commit(acd_producer_state)

                acd_producer_state.advance()
                if acd_producer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acd_producer_state)

        # ---------------------------------------------------------------
        # Specialized bias load warp
        # ---------------------------------------------------------------
        if cutlass.const_expr(self.enable_bias):
            if warp_idx == self.bias_load_warp_id and total_token > 0:
                bias_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_bias_stage)
                tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

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

                bias_n_total = mBias_nl.shape[0]
                tBpBias = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)

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
                    mma_n_coord = tile_info[2]
                    expert_idx = tile_info[0]
                    gBias_tile = gBias_nl[(None, mma_n_coord, expert_idx)]
                    tBs_gBias = thr_bias_g2s.partition_S(gBias_tile)
                    tBpBias[0] = mma_n_coord * self.mma_tiler[1] + cute.arch.lane_idx() * bias_elems_per_thread < bias_n_total
                    bias_pipeline.producer_acquire(bias_producer_state)
                    cute.copy(
                        bias_g2s_tiled,
                        tBs_gBias[(None, 0)],
                        tBs_sBias[(None, 0, bias_producer_state.index)],
                        pred=tBpBias,
                    )
                    bias_pipeline.producer_commit(bias_producer_state)
                    bias_producer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                bias_pipeline.producer_tail(bias_producer_state)

        # ---------------------------------------------------------------
        # Specialized ACT epilogue warps (0-3): TMEM→regs, alpha, GLU activation,
        # C store, hadamard_in
        # ---------------------------------------------------------------
        if warp_idx < self.epilog_rht_store_warp_id[0] and total_token > 0:
            epi_tidx = tidx

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
            # Partition for epilogue (shape-only via mD_mnl for invariant setup)
            #
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
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

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc_gate.shape, self.d_smem_dtype)
            tiled_copy_r2s_d, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rD, epi_tidx, sD)

            #
            # Create per-expert extension (for C/prob tensors inside tile loop)
            #
            epi_ext = self._make_extension(workspace_ptr)

            #
            # Persistent tile scheduling state
            #
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            #
            # Pingpong producer state
            #
            pingpong_act_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_pingpong_stage)

            # Threads/warps participating in TMA store pipeline for C
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_act_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            # NVFP4 D: the fp4 TMA store is issued by the RHT warps (which quantize
            # sD); the ACT warps only stage bf16 into sD.
            if cutlass.const_expr(not self.d_quant):
                d_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.epilog_act_warp_id),
                )
                d_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_d_stage,
                    producer_group=d_producer_group,
                )

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            if cutlass.const_expr(self.enable_bias):
                bias_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_bias_stage)
                bias_s2r_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.bias_dtype, num_bits_per_copy=128)
                tTR_rBias_gate = cute.make_rmem_tensor(cute.make_layout(self.epi_tile[1]), self.bias_dtype)
                tTR_rBias_up = cute.make_rmem_tensor(cute.make_layout(self.epi_tile[1]), self.bias_dtype)

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            num_prev_d_subtiles = cutlass.Int32(0)
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
                    if cutlass.const_expr(self.act_func == "srelu"):
                        sBias_subtiles = cute.flat_divide(sBias_stage, cute.make_layout(self.epi_tile[1]))
                    else:
                        sBias_subtiles = cute.flat_divide(sBias_stage, cute.make_layout(2 * self.epi_tile[1]))

                #
                # Get per-expert C tensor inside tile loop
                #
                real_c, _ = epi_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, epi_work_tile_info)
                gC_mnl = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                thr_mma_epi_loop = tiled_mma.get_slice(mma_tile_coord_v)
                tCgC = thr_mma_epi_loop.partition_C(gC_mnl)
                _, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, self.epi_tile_c, sC)
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Get per-expert D tensor inside tile loop (NVFP4 D: the RHT warps own
                # the fp4 D TMA store — no D partition on the ACT side).
                #
                if cutlass.const_expr(not self.d_quant):
                    real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)
                    gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgD = thr_mma_epi_loop.partition_C(gD_mnl_loop)
                    _, bSG_sD, bSG_gD_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD, epi_tile, sD)
                    bSG_gD = bSG_gD_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                    bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                #
                # Get per-expert prob tensor inside tile loop
                #
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mPosition = (
                    (epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)) * self.mma_tiler[0]
                    + mma_tile_coord_v * (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape))
                    + tidx
                )
                mProb = real_prob[mPosition, 0, 0]

                #
                # Get accumulator stage index
                #
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                subtile_step = 1 if cutlass.const_expr(self.act_func == "srelu") else 2
                for subtile_idx in cutlass.range(0, subtile_cnt, subtile_step, unroll=1):
                    real_subtile_idx = subtile_idx if cutlass.const_expr(self.act_func == "srelu") else subtile_idx // 2
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - real_subtile_idx

                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    if cutlass.const_expr(self.act_func == "srelu"):
                        tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    else:
                        tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
                        tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2 + 1)]

                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)
                    if cutlass.const_expr(self.act_func != "srelu"):
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    #
                    # Apply alpha (+ bias when enabled)
                    #
                    if cutlass.const_expr(self.enable_bias):
                        if cutlass.const_expr(self.act_func == "srelu"):
                            sBias_sub = sBias_subtiles[(None, real_subtile_idx)]
                            cute.copy(bias_s2r_atom, sBias_sub, tTR_rBias_gate)
                        else:
                            sBias_pair = sBias_subtiles[(None, real_subtile_idx)]
                            sBias_sub = cute.flat_divide(sBias_pair, cute.make_layout(self.epi_tile[1]))
                            cute.copy(bias_s2r_atom, sBias_sub[(None, 0)], tTR_rBias_gate)
                        bias_vec_gate = tTR_rBias_gate.load()
                        if cutlass.const_expr(self.act_func != "srelu"):
                            cute.copy(bias_s2r_atom, sBias_sub[(None, 1)], tTR_rBias_up)
                            bias_vec_up = tTR_rBias_up.load()

                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_gate), 2):
                                bias_gate_f32_0 = bias_vec_gate[i].to(cutlass.Float32)
                                bias_gate_f32_1 = bias_vec_gate[i + 1].to(cutlass.Float32)
                                tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1] = cute.arch.fma_packed_f32x2(
                                    (tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1]),
                                    (
                                        cutlass.Float32(alpha_val),
                                        cutlass.Float32(alpha_val),
                                    ),
                                    (bias_gate_f32_0, bias_gate_f32_1),
                                    rnd="rn",
                                    ftz=False,
                                )
                                if cutlass.const_expr(self.act_func != "srelu"):
                                    bias_up_f32_0 = bias_vec_up[i].to(cutlass.Float32)
                                    bias_up_f32_1 = bias_vec_up[i + 1].to(cutlass.Float32)
                                    tTR_rAcc_up[i], tTR_rAcc_up[i + 1] = cute.arch.fma_packed_f32x2(
                                        (tTR_rAcc_up[i], tTR_rAcc_up[i + 1]),
                                        (
                                            cutlass.Float32(alpha_val),
                                            cutlass.Float32(alpha_val),
                                        ),
                                        (bias_up_f32_0, bias_up_f32_1),
                                        rnd="rn",
                                        ftz=False,
                                    )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_gate)):
                                tTR_rAcc_gate[i] = tTR_rAcc_gate[i] * cutlass.Float32(alpha_val) + bias_vec_gate[i].to(cutlass.Float32)
                                if cutlass.const_expr(self.act_func != "srelu"):
                                    tTR_rAcc_up[i] = tTR_rAcc_up[i] * cutlass.Float32(alpha_val) + bias_vec_up[i].to(cutlass.Float32)

                        last_bias_subtile = subtile_cnt - 1 if cutlass.const_expr(self.act_func == "srelu") else subtile_cnt - 2
                        if subtile_idx == last_bias_subtile:
                            bias_pipeline.consumer_release(bias_consumer_state)
                            bias_consumer_state.advance()
                    else:
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_gate), 2):
                                tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1] = cute.arch.mul_packed_f32x2(
                                    (tTR_rAcc_gate[i], tTR_rAcc_gate[i + 1]),
                                    (
                                        cutlass.Float32(alpha_val),
                                        cutlass.Float32(alpha_val),
                                    ),
                                    rnd="rn",
                                    ftz=False,
                                )
                                if cutlass.const_expr(self.act_func != "srelu"):
                                    tTR_rAcc_up[i], tTR_rAcc_up[i + 1] = cute.arch.mul_packed_f32x2(
                                        (tTR_rAcc_up[i], tTR_rAcc_up[i + 1]),
                                        (
                                            cutlass.Float32(alpha_val),
                                            cutlass.Float32(alpha_val),
                                        ),
                                        rnd="rn",
                                        ftz=False,
                                    )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_gate)):
                                tTR_rAcc_gate[i] = tTR_rAcc_gate[i] * cutlass.Float32(alpha_val)
                                if cutlass.const_expr(self.act_func != "srelu"):
                                    tTR_rAcc_up[i] = tTR_rAcc_up[i] * cutlass.Float32(alpha_val)

                    #
                    # Store pre-activation output to C tensor for residual/backward.
                    #
                    if cutlass.const_expr(self.act_func == "srelu"):
                        self.store_c_unary(
                            tiled_copy_r2s,
                            tma_atom_c,
                            warp_idx,
                            tTR_rAcc_gate,
                            tRS_rC,
                            tRS_sC,
                            bSG_gC,
                            bSG_sC,
                            c_pipeline,
                            num_prev_subtiles,
                            real_subtile_idx,
                        )
                    else:
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
                    num_prev_subtiles = num_prev_subtiles + 1

                    #
                    # GeGLU clamp before C store
                    #
                    if cutlass.const_expr((self.act_func == "geglu" or self.act_func == "swiglu") and self.glu_limit is not None):
                        geglu_max_val = cutlass.Float32(self.glu_limit)
                        geglu_min_val = cutlass.Float32(-self.glu_limit)
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                            tTR_rAcc_gate[i] = fmin(tTR_rAcc_gate[i], geglu_max_val)
                            tTR_rAcc_gate[i] = fmax(tTR_rAcc_gate[i], geglu_min_val)
                            tTR_rAcc_up[i] = fmin(tTR_rAcc_up[i], geglu_max_val)
                            tTR_rAcc_up[i] = fmax(tTR_rAcc_up[i], geglu_min_val)

                    acc_vec_gate = tTR_rAcc_gate.load()

                    #
                    # Compute activation.
                    #
                    tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                    if cutlass.const_expr(self.act_func == "srelu"):
                        self.srelu_act(tCompute, acc_vec_gate, mProb)
                    elif cutlass.const_expr(self.act_func == "geglu"):
                        acc_vec_up = tTR_rAcc_up.load()
                        self.geglu_act(tCompute, acc_vec_up, acc_vec_gate, mProb, linear_offset)
                    elif cutlass.const_expr(self.act_func == "swiglu"):
                        acc_vec_up = tTR_rAcc_up.load()
                        self.swiglu_act(tCompute, acc_vec_up, acc_vec_gate, mProb)

                    #
                    # Store post-activation output to D staging (bf16 under NVFP4 D —
                    # the RHT warps quantize it and issue the fp4 TMA store).
                    #
                    acc_vec = tiled_copy_r2s_d.retile(tCompute).load()
                    tRS_rD.store(acc_vec.to(self.d_smem_dtype))
                    d_buffer = num_prev_d_subtiles % self.num_d_stage
                    cute.copy(
                        tiled_copy_r2s_d,
                        tRS_rD,
                        tRS_sD[(None, None, None, d_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier_group0.arrive_and_wait()
                    if cutlass.const_expr(not self.d_quant):
                        if warp_idx == self.epilog_act_warp_id[0]:
                            cute.copy(
                                tma_atom_d,
                                bSG_sD[(None, d_buffer)],
                                bSG_gD[(None, real_subtile_idx)],
                            )
                            d_pipeline.producer_commit()

                    #
                    # Signal the RHT epilogue warps that the post-activation D tile is in SMEM.
                    #
                    if cutlass.const_expr(self.run_rht):
                        pingpong_pipeline.producer_acquire(pingpong_act_producer_state)
                        pingpong_pipeline.producer_commit(pingpong_act_producer_state)
                        pingpong_act_producer_state.advance()

                    num_prev_d_subtiles = num_prev_d_subtiles + 1

                    #
                    # Delayed TMA store acquire + group sync (always enabled)
                    #
                    if cutlass.const_expr(self.delay_tma_store_acquire_sync):
                        if warp_idx == self.epilog_act_warp_id[0]:
                            if cutlass.const_expr(not self.d_quant):
                                d_pipeline.producer_acquire()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier_group0.arrive_and_wait()

                    if cutlass.const_expr(self.run_rht):
                        self.epilog_sync_barrier.arrive_and_wait()

                #
                # Full epilogue barrier (ACT + RHT must both arrive)
                #
                if cutlass.const_expr(self.run_rht):
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
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
            self.epilog_sync_barrier_group0.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store / pingpong complete
            #
            c_pipeline.producer_tail()
            if cutlass.const_expr(not self.d_quant):
                d_pipeline.producer_tail()
            if cutlass.const_expr(self.run_rht):
                pingpong_pipeline.producer_tail(pingpong_act_producer_state)

        # ---------------------------------------------------------------
        # Specialized RHT store warps (4-7): RHT and/or NVFP4 quantization from D SMEM
        # ---------------------------------------------------------------
        if self.run_rht and warp_idx < self.mma_warp_id and warp_idx >= self.epilog_rht_store_warp_id[0] and total_token > 0:
            epi_tidx = tidx % 128

            #
            # Pingpong consumer state
            #
            pingpong_rht_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_pingpong_stage)

            #
            # Create per-expert extension (for D tensor inside tile loop)
            #
            epi_ext = self._make_extension(workspace_ptr)

            #
            # RHT output: TMA-store pipeline for the RHT warps (mirrors ACT's D store).
            #
            if cutlass.const_expr(self.generate_rht):
                rht_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.epilog_rht_store_warp_id),
                )
                rht_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_d_stage,
                    producer_group=rht_producer_group,
                )

            #
            # NVFP4 D: the RHT warps quantize sD and own the fp4 D TMA store.
            #
            if cutlass.const_expr(self.d_quant):
                dq_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.epilog_rht_store_warp_id),
                )
                dq_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_d_stage,
                    producer_group=dq_producer_group,
                )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_d_subtiles = cutlass.Int32(0)
            # Mirror ACT's per-tile subtile-column reversal (overlapping_accum phase).
            # ACT's first tile uses acc phase 0 => reverse=True, then toggles each tile.
            if cutlass.const_expr(self.run_rht and self.overlapping_accum):
                rht_reverse = cutlass.Boolean(True)
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
                epi_ext.update_expert_info(padded_offsets, epi_work_tile_info.expert_idx)

                #
                # RHT output: per-expert RHT gmem tensor + TMA partition (mirrors ACT's D setup).
                #
                if cutlass.const_expr(self.generate_rht):
                    thr_mma_epi_rht = tiled_mma.get_slice(mma_tile_coord_v)
                    real_rht, _ = epi_ext.get_gmem_tensor("d", mRht_mnl, padded_offsets, epi_work_tile_info)
                    gRht_mnl_loop = cute.local_tile(real_rht, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgRht = thr_mma_epi_rht.partition_C(gRht_mnl_loop)
                    _, bSG_sRht, bSG_gRht_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_rht, tCgRht, epi_tile, sRht)
                    bSG_gRht = bSG_gRht_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                    bSG_gRht = cute.group_modes(bSG_gRht, 1, cute.rank(bSG_gRht))
                if cutlass.const_expr(self.rht_quant and not self.rht_rowwise):
                    # Expert token offset for the colwise (f, m) scale grid's
                    # tile index (offsets are 256-aligned, divisions exact).
                    rht_t_off, _rht_t_cnt = compute_expert_token_range(padded_offsets, epi_work_tile_info.expert_idx)

                #
                # NVFP4 D: per-expert fp4 D gmem tensor + TMA partition (mirrors ACT's D setup).
                #
                if cutlass.const_expr(self.d_quant):
                    thr_mma_epi_dq = tiled_mma.get_slice(mma_tile_coord_v)
                    real_dq, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)
                    gDq_mnl_loop = cute.local_tile(real_dq, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgDq = thr_mma_epi_dq.partition_C(gDq_mnl_loop)
                    _, bSG_sDq, bSG_gDq_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgDq, epi_tile, sDq)
                    bSG_gDq = bSG_gDq_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                    bSG_gDq = cute.group_modes(bSG_gDq, 1, cute.rank(bSG_gDq))

                #
                # NVFP4 scale tensors: each thread owns one logical scale-domain row.
                # Per subtile iteration it stages one scale byte per 16-value block
                # into smem, then writes those bytes into the swizzled SF atom layout.
                #
                if cutlass.const_expr(self.rht_quant or self.d_quant):
                    sf_row = (
                        (epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)) * self.mma_tiler[0]
                        + mma_tile_coord_v * (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape))
                        + epi_tidx
                    )
                #
                # Consume D subtiles from SMEM
                #
                subtile_cnt = self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1])
                subtile_step = 1 if cutlass.const_expr(self.act_func == "srelu") else 2
                for subtile_idx in cutlass.range(0, subtile_cnt, subtile_step, unroll=1):
                    #
                    # Wait for ACT warps to finish writing the post-activation D tile to SMEM.
                    #
                    pingpong_pipeline.consumer_wait(pingpong_rht_consumer_state)

                    if cutlass.const_expr(self.run_rht):
                        d_buffer = num_prev_d_subtiles % self.num_d_stage
                        # Subtile-column reversal (overlapping_accum): computed BEFORE
                        # the FWHT/quant calls — scale bytes and TMA'd data must land
                        # in the same reversed gmem feature columns as ACT's D store.
                        real_subtile_idx = subtile_idx // 2
                        if cutlass.const_expr(self.overlapping_accum):
                            if rht_reverse:
                                real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - real_subtile_idx
                        #
                        # Load this subtile's sD values to registers, then ARRIVE the
                        # lockstep barrier immediately: the ACT warps' arrive_and_wait
                        # releases, and their next subtile overlaps ALL the FWHT/quant
                        # compute and stores below. The RHT side never waits on this
                        # barrier (the pingpong gates it).
                        #
                        if cutlass.const_expr(self.generate_rht):
                            if cutlass.const_expr(self.rht_rowwise):
                                rht_ld = load_row_bf16(sD, d_buffer, epi_tidx)
                            else:
                                rht_ld = load_colwise_pairs_bf16(sD, d_buffer, epi_tidx)
                        if cutlass.const_expr(self.d_quant):
                            dq_ld = load_row_bf16(sD, d_buffer, epi_tidx)
                        self.epilog_sync_barrier.arrive()
                        # RHT output: write the transform (x0.25) to sRht. rht_rowwise
                        # picks the transform axis (16-feature blocks per token vs
                        # 16-token blocks per feature); NVFP4 quantization is inferred
                        # from the sRht dtype inside the FWHT device functions.
                        if cutlass.const_expr(self.generate_rht):
                            if cutlass.const_expr(self.rht_quant and self.rht_rowwise):
                                hadamard_rmem_rowwise_fwht(rht_ld, d_buffer, epi_tidx, sRht, rht_norm_const, sSfRht, real_subtile_idx, self.sf_dtype)
                            elif cutlass.const_expr(self.rht_quant):
                                hadamard_rmem_colwise_fwht_quant(
                                    rht_ld, d_buffer, epi_tidx, rht_norm_const, sRht, sSfRht, real_subtile_idx * 2 * HADAMARD_SIZE, self.sf_dtype
                                )
                            elif cutlass.const_expr(self.rht_rowwise):
                                hadamard_rmem_rowwise_fwht(rht_ld, d_buffer, epi_tidx, sRht, 1.0, None, 0, self.sf_dtype)
                            else:
                                hadamard_rmem_colwise_fwht(rht_ld, d_buffer, epi_tidx, sRht)
                        # NVFP4 D: quantize the bf16 register rows into sDq + sSfd.
                        if cutlass.const_expr(self.d_quant):
                            nvfp4_quant_rmem_row(dq_ld, d_buffer, epi_tidx, sDq, norm_const, sSfd, real_subtile_idx, self.sf_dtype)
                        #
                        # TMA-store the produced epi-tiles to gmem (mirrors ACT's D
                        # store, including the overlapping_accum subtile-column reversal).
                        #
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier_group1.arrive_and_wait()
                        if warp_idx == self.epilog_rht_store_warp_id[0]:
                            if cutlass.const_expr(self.generate_rht):
                                cute.copy(
                                    tma_atom_rht,
                                    bSG_sRht[(None, d_buffer)],
                                    bSG_gRht[(None, real_subtile_idx)],
                                )
                                rht_pipeline.producer_commit()
                                rht_pipeline.producer_acquire()
                            if cutlass.const_expr(self.d_quant):
                                cute.copy(
                                    tma_atom_d,
                                    bSG_sDq[(None, d_buffer)],
                                    bSG_gDq[(None, real_subtile_idx)],
                                )
                                dq_pipeline.producer_commit()
                                dq_pipeline.producer_acquire()
                        self.epilog_sync_barrier_group1.arrive_and_wait()

                    #
                    # Release pingpong consumer slot
                    #
                    pingpong_pipeline.consumer_release(pingpong_rht_consumer_state)
                    pingpong_rht_consumer_state.advance()
                    num_prev_d_subtiles = num_prev_d_subtiles + 1
                    # Lockstep barrier: under run_rht the 128 RHT arrivals already
                    # happened right after the sD reads (arrive() above); only the
                    # idle-RHT plain mode still pairs arrive_and_wait here.
                    if cutlass.const_expr(not self.run_rht):
                        self.epilog_sync_barrier.arrive_and_wait()

                #
                # NVFP4: one contiguous 8-byte scale-row store per thread per tile.
                # Rowwise: the thread wrote all its own smem slots (no barrier needed).
                # Colwise: rows were filled ACROSS warps — the loop-end epilog_sync
                # barrier above already ordered those writes.
                #
                _num_sf = self.cta_tile_shape_mnk_d[1] // HADAMARD_SIZE
                if cutlass.const_expr(self.rht_quant and self.rht_rowwise):
                    self.store_swizzled_sf_row(
                        mSfRht_mnl,
                        epi_ext.token_offset + sf_row,
                        mma_tile_coord_mnl[1] * _num_sf,
                        sSfRht,
                        epi_tidx,
                    )
                if cutlass.const_expr(self.rht_quant and not self.rht_rowwise):
                    # (f, m) scale domain: thread <-> feature-in-tile; columns are
                    # 16-token scale blocks, stored in the same swizzled SF atom layout.
                    sf_feat_row = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk_d[1] + epi_tidx
                    self.store_swizzled_sf_row(
                        mSfRht_mnl,
                        sf_feat_row,
                        (rht_t_off + sf_row - epi_tidx) // HADAMARD_SIZE,
                        sSfRht,
                        epi_tidx,
                    )
                if cutlass.const_expr(self.d_quant):
                    self.store_swizzled_sf_row(
                        mSfd_mnl,
                        epi_ext.token_offset + sf_row,
                        mma_tile_coord_mnl[1] * _num_sf,
                        sSfd,
                        epi_tidx,
                    )

                #
                # Full epilogue barrier (ACT + RHT must both arrive)
                #
                self.epilog_sync_barrier.arrive_and_wait()

                #
                # Toggle the subtile-column reversal for the next tile.
                #
                if cutlass.const_expr(self.run_rht and self.overlapping_accum):
                    rht_reverse = not rht_reverse

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
            # Drain the RHT-warp TMA-store pipelines before exit.
            #
            if cutlass.const_expr(self.generate_rht):
                rht_pipeline.producer_tail()
            if cutlass.const_expr(self.d_quant):
                dq_pipeline.producer_tail()
        # END OF KERNEL
