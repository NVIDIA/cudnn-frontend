# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Subchannel-Scaled MoE Block-Scaled Grouped GEMM Kernel with dGLU (dSwiGLU) Backward Fusion.

Supports:
    - Static / Dynamic persistent tile scheduling (MoEPersistentTileScheduler)
    - Dense (contiguous 3-D B) / Discrete (per-expert pointer array B) weight layout
    - FP8/FP4 output quantization with row scale factors (SFD)
    - AMAX reduction for FP8 calibration
    - dGLU backward activation fusion (dSwiGLU / dGeGLU)

This module contains only the kernel class.
MoE scheduler components live in moe_persistent_scheduler.py / moe_sched_extension.py / moe_utils.py.
"""

from typing import Type, Tuple, Union, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import T
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from cutlass.cute.typing import Float32, Int32, AddressSpace
from cutlass._mlir import ir
from cutlass._mlir.dialects import math, llvm
from cutlass._mlir.dialects import vector, arith

from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import (
    MoEWeightMode,
    TensormapWorkspace,
    compute_expert_token_range,
    store_tma_desc,
)
from ..moe_sched_extension import (
    DiscreteWeightScaledGemmSchedExtension,
    ContiguousAndConsistentGroupedGemmSchedExtension,
)
from ..moe_kernel_helpers import (
    fmin,
    fmax,
    fmin_bf16x2,
    fmax_bf16x2,
    atomic_max_float32,
    atomic_add_float32,
    atomic_add_i32_gmem,
    load_acquire_i32_gpu,
    load_float32_volatile,
    sigmoid_f32,
    get_dtype_rcp_limits,
    get_dtype_max,
    get_amax_smem_size,
    is_valid_layouts,
    is_valid_mma_tiler_and_cluster_shape,
    is_valid_tensor_alignment,
    atomic_add_bf16x2,
    nanosleep,
    red_smem_cluster_max_f32,
    mbarrier_arrive_cluster,
)

"""
High-performance persistent blockscaled contiguous grouped dense GEMM (D = alpha * (SFA * A) * (SFB * B)) example for the NVIDIA Blackwell architecture
using CUTE DSL.
- Matrix A is MxKx1, A can be row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, B can be column-major("K"), L is grouped dimension
- Matrix D is MxNx1, D can be row-major("N"), ValidM is composed of valid m in different groups
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk, which has N×ceil_div(K, sf_vec_size)×L elements respectively

Matrix A/D Memory Layout Diagrams:

   ```
    Group 0    Group 1   Group 2
   -+---------+---------+---------+
    |         |         |         |
   K| ValidM0 | ValidM1 | ValidM2 |
    |         |         |         |
   -+---------+---------+---------+
    |<-        ValidM           ->|
   ```
   Note: the Group(L) dimension will be flatted into M dimension, and the rest Group(L) size is 1.
         each ValidM will be aligned to 256 or 128. The alignment is determined by the mma_tiler_mn parameter.
         For NVFP4, 2CTA, the alignment is 256. For NVFP4, 1CTA, the alignment is 128. 

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. SCALE warp: Load scaleA and scaleB matrices from global memory (GMEM) to shared memory (SMEM) using non-TMA operations.
2. MMA warp: 
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Apply alpha and update the final accumulator Final = alpha * acc
    - Type convert Final matrix to output type.
    - Store D matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

.. code-block:: bash

    python continugous_blockscaled_grouped_gemm_dglu_quant_fusion.py         \
      --ab_dtype Float8E4M3FN --sf_dtype Float8E8M0FNU --c_dtype BFloat16    \
      --d_dtype Float8E4M3FN --sf_vec_size 32 --mma_tiler_mn 256,256         \
      --cluster_shape_mn 2,1 --nkl 4096,7168,8 --use_2cta_instrs             \
      --m_aligned 256 --fixed_m 4096 --warmup_iterations 10 --iterations 30


Constraints:
* Supported input data types: mxf8, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/D tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.

CUDA Graph Support:
* For CUDA graph support, the A/D matrices and scale factor A can be padded to a larger size
  (e.g., permuted_m = m*topK + num_local_experts*(FIX_PAD_SIZE-1), example: 4096*8 + 8*255 = 34808)
* Use create_tensors() with permuted_m parameter to automatically pad:
  - A matrix: padded to permuted_m rows (padding rows contain dummy data)
  - D matrix: padded to permuted_m rows (output buffer for cuda_graph)
  - Scale factor A: padded to match A matrix dimensions
* Kernel handling of padding:
  - Scheduler warp loads padded_offsets and calculates num_valid_tiles from padded_offsets[-1]
  - Uses warp-parallel search (ballot + ffs) to find expert_idx for each tile
  - Only valid tiles (tile_m_start < padded_offsets[-1]) are written to tile_info pipeline
  - When no more valid tiles exist, outer loop exits and calls producer_tail()
  - Consumer warps process only valid tiles from pipeline
  - No deadlock or synchronization issues
* Only rows within (aligned_groupm[0]+aligned_groupm[1]+...) contain valid data
* Padding rows in D matrix will not be written by the kernel
"""


def _is_quant_dtype_combo(a_dtype, sf_dtype, d_dtype) -> bool:
    """Whether the dtype triple selects the quantized-D epilogue
    (generate_sfd). A plain module-level function so the result is a real
    python bool even when called from traced code (the @cute.jit AST pass
    rewrites `in`/`and` inside the kernel's __call__ into cutlass.Boolean,
    which cannot feed SharedStorage sizing)."""
    return (
        a_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN)
        and sf_dtype in (cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN)
        and d_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN)
    )


def _deinterleave_n_coord_tensor(t, mode=1, n_static=None):
    """View a tensor carrying d's interleaved n axis at `mode` with that mode
    split (32, 2, n/64), strides (b, (n/2)*b, 32*b): callers keep indexing in
    INTERLEAVED d coordinates, but interleaved 32-col band bd now lands at the
    DEINTERLEAVED position 32*(bd//2) + (bd%2)*(n/2) — gate bands fill
    [0, n/2), up bands [n/2, n). For d it is applied to the TMA coordinate
    tensor AFTER make_tiled_tma_atom (sfb-192 precedent): the tensormap keeps
    its plain 3-dim box, and each (128, 32) epi box still covers one whole
    band = 32 contiguous deinterleaved columns. Also used for the plain
    layout-addressed tensors: dbias (n at mode 1)."""
    n = n_static if n_static is not None else t.shape[mode]
    b = t.stride[mode]
    shape = list(t.shape)
    stride = list(t.stride)
    shape[mode] = (32, 2, n // 64)
    # ScaledBasis strides (the TMA coordinate tensors) only accept STATIC
    # integer scales — pass n_static (a python int) for those; the plain
    # layout-addressed tensors work with the dynamic extent.
    stride[mode] = (b, (n // 2) * b, 32 * b)
    new_layout = cute.make_layout(tuple(shape), stride=tuple(stride))
    return cute.make_tensor(t.iterator, new_layout)


# Supported launch configs: mma tiler (256, 128) with an M-only cluster (2, 1)
# or the 2D clusters (2, 2)/(2, 4). cluster_n > 1 makes N-adjacent work tiles
# cluster peers — the rowwise-sfd2 DSMEM geometry (sgn/cta_n == cluster_n) —
# and enables A/SFA multicast. Cluster M stays 2: wider cluster-M tiles need a
# per-expert-M cluster-tile-multiple gate the FE API cannot verify host-side
# on ragged inputs (the kernel's static can_implement still accepts the full
# harness set (2,1)/(4,1)/(2,2)/(2,4)/(4,2) with that gate, for parity).
VALID_CTA_SHAPES = ((256, 128),)
VALID_CLUSTER_SHAPES = ((2, 1), (2, 2), (2, 4))
DEFAULT_CTA_SHAPE = (256, 128)
DEFAULT_CLUSTER_SHAPE = (2, 1)


class BlockScaledSubChannelMoEGroupedGemmDgluDbiasKernel:
    """Block-scaled grouped GEMM kernel with MoE tile scheduling and dGLU backward fusion.

    Supports both dense and discrete weight layouts, static and dynamic
    scheduling, and quantized output with row scale factors.

    This version uses a fixed padding size (FIX_PAD_SIZE=256) that is decoupled from the kernel's tile size,
    allowing users to pad their tensors without knowing the specific kernel implementation details.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported D data types:
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
        - FIX_PAD_SIZE (256) must be divisible by mma_tiler_mn[0]
        - m_aligned parameter in create_mask() MUST equal FIX_PAD_SIZE (256)
        - Each padded_offsets[i] will be a multiple of FIX_PAD_SIZE (guaranteed by m_aligned == FIX_PAD_SIZE)

    :note: New Interface (padded_offsets):
        Instead of tile_idx_to_expert_idx, num_non_exiting_tiles, and m_split_cumsum, users now provide:
        - padded_offsets: shape (expert_cnt,), where padded_offsets[i] is the end position
          of expert[i] in the padded A tensor.
        - Expert i processes A[padded_offsets[i-1]:padded_offsets[i], :] (with padded_offsets[-1]=0)

    """

    # Fixed pad size for user-side padding (decoupled from kernel tile size)
    FIX_PAD_SIZE = 256

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the data type / scale-factor vector-size combination is valid.

        :return: True if valid, False otherwise
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        if sf_vec_size not in {16}:
            is_valid = False

        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        if sf_dtype in {cutlass.Float8E4M3FN} and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        if acc_dtype not in {cutlass.Float32}:
            is_valid = False

        if d_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        if ab_dtype.width == 8 and d_dtype.width != 8:
            is_valid = False

        return is_valid

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
        act_func: str,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        sgn: int = 256,
        sgk: int = 256,
    ) -> bool:
        FPS = BlockScaledSubChannelMoEGroupedGemmDgluDbiasKernel.FIX_PAD_SIZE
        result = True
        if m_aligned != FPS:
            result = False
        if not BlockScaledSubChannelMoEGroupedGemmDgluDbiasKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, acc_dtype, d_dtype
        ):
            result = False
        if not is_valid_layouts(ab_dtype, d_dtype, a_major, b_major, cd_major):
            result = False
        # Only supported launch configs for now: mma tiler (256, 128) with
        # M-only clusters (2,1)/(4,1), or 2D clusters (2,2)/(2,4)/(4,2)
        # (cluster_n > 1 makes N-adjacent work tiles cluster peers — the
        # rowwise-sfd2 DSMEM geometry — and enables A/SFA multicast; total
        # cluster size stays <= 8, the portable limit).
        if tuple(mma_tiler_mn) != (256, 128) or tuple(cluster_shape_mn) not in (
            (2, 1),
            (4, 1),
            (2, 2),
            (2, 4),
            (4, 2),
        ):
            result = False
        # cluster_n > 1: the scheduler decomposes over CLUSTER N tiles, so
        # the N work-tile count (n here is the GEMM half-width; tiles =
        # ceil(n / cta_n), expert-uniform) must be a cluster_n multiple or
        # the last N cluster's peers would map past the edge.
        if cluster_shape_mn[1] > 1:
            _n_work_tiles = -(-n // mma_tiler_mn[1])
            if _n_work_tiles % cluster_shape_mn[1] != 0:
                result = False
        if not is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m_aligned,
            FPS,
            allowed_cluster_tiler_m=(128, 256, 512),
        ):
            result = False
        # Cluster (4,1)/(4,2) widens the cluster M tile to 512 rows sharing
        # one multicast B load: a cluster must never straddle an expert
        # boundary, so per-expert M must be a cluster-tile multiple.
        _cluster_tiler_m = (cluster_shape_mn[0] // (2 if use_2cta_instrs else 1)) * mma_tiler_mn[0]
        if (m // l) % _cluster_tiler_m != 0:
            result = False
        if not is_valid_tensor_alignment(m, n, k, l, ab_dtype, d_dtype, a_major, b_major, cd_major):
            result = False
        if not (a_major == "k"):
            result = False
        if act_func not in ["dswiglu", "dgeglu"]:
            result = False
        # SFD2 blocking (deinterleaved): one sfd2 block = sgn gate/up
        # f-columns. The single per-thread atomic-max target / read-back and
        # the tile_n_idx // sfd2_n_contrib block math require every N work
        # tile to sit inside one block: sgn must be a multiple of the tile
        # f-width.
        if sgn % mma_tiler_mn[1] != 0:
            result = False
        # Discrete SFB2 per-expert base pointers are declared 16B-aligned in
        # the sched extension (assumed_align=16). With back-to-back per-expert
        # f32 (ceil(n/sgn), ceil(k/sgk)) blocks, every base past the first is
        # misaligned unless the block byte size is a multiple of 16.
        if weight_mode == MoEWeightMode.DISCRETE and l > 1:
            sfb2_block_bytes = ((n + sgn - 1) // sgn) * ((k + sgk - 1) // sgk) * 4
            if sfb2_block_bytes % 16 != 0:
                result = False
        return result

    def __init__(
        self,
        sf_vec_size: int,
        sgm: int,
        sgn: int,
        sgk: int,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        expert_cnt: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DISCRETE,
        use_dynamic_sched: bool = False,
        act_func: str = "dswiglu",
        generate_sfd2: bool = True,
        glu_alpha: Optional[float] = None,
        glu_clamp_max: Optional[float] = None,
        glu_clamp_min: Optional[float] = None,
        d_deinterleaved: bool = False,  # constexpr: True stores the n-axis
        # outputs (D, its row SFs, dbias)
        # DEINTERLEAVED ([gate | up]); False
        # keeps d's interleaved band order
        dsmem_rowwise: bool = True,  # allow the rowwise-sfd2 DSMEM reduction
        # when the geometry is eligible
        # (quantized D + sfd2 + sgn/cta_n ==
        # cluster_n > 1); byte-exact vs the gmem
        # protocol either way
        deint_n: int = 0,  # STATIC interleaved d width (2*n); required for
        # the TMA coordinate-tensor deinterleave rewrite
        # (only when d_deinterleaved)
    ):
        """Initializes the configuration for a Blackwell blockscaled grouped GEMM dGLU kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  Expert Count:
            - expert_cnt: Number of experts for MoE grouped GEMM.

        4.  MoE Tile Scheduling:
            - Uses MoEPersistentTileScheduler for tile iteration across experts
            - Expert lookup is handled by the scheduler (cached O(1) fast path)

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param expert_cnt: Number of experts (compile-time constant).
        :type expert_cnt: int

        :raises ValueError: If FIX_PAD_SIZE is not divisible by mma_tiler_mn[0].
        """
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

        self.sf_vec_size = sf_vec_size
        self.sgm = sgm
        self.sgn = sgn
        self.sgk = sgk
        # Cross-tile sfd2 sync: number of N work-tiles whose atomics feed one
        # sfd2 block element. sgn counts DEINTERLEAVED f-columns (one block =
        # sgn gate/up columns); each tile covers mma_tiler_n f-columns.
        self.sfd2_n_contrib = max(1, sgn // mma_tiler_mn[1])
        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.accumulator_update_warp_id = (0, 1, 2, 3)
        self.epilog_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.scale_warp_id = 11
        self.threads_per_warp = 32

        # Register allocation for different warp types
        self.num_regs_uniform_warps = 24
        self.num_regs_sched_warps = 24
        self.num_regs_epilogue_warps = 192
        self.num_regs_acc_update_warps = 256

        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.accumulator_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                self.scale_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.accumulator_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.scale_warp_id,
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
            num_threads=32 * len((self.mma_warp_id, *self.accumulator_update_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.vectorized_f32 = vectorized_f32
        self.use_dynamic_sched = use_dynamic_sched

        # Amax reduction configuration
        self.num_epilog_warps = len(self.epilog_warp_id)

        self.generate_sfd2 = generate_sfd2
        self.weight_mode = weight_mode

        self.act_func = act_func

        self.glu_alpha = glu_alpha
        self.glu_clamp_max = glu_clamp_max
        self.glu_clamp_min = glu_clamp_min

        if d_deinterleaved and deint_n <= 0:
            raise ValueError("d_deinterleaved requires deint_n (the STATIC interleaved d " "width, 2*n) for the TMA coordinate-tensor rewrite")
        self.d_deinterleaved = d_deinterleaved
        self.deint_n = deint_n
        self.dsmem_rowwise = dsmem_rowwise

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
        - Computing tensor memory allocation columns
        """

        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Configure tiled mma
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

        # Compute mma/cluster/tile shapes
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

        self.mma_tiler_d = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
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

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
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

        # enable direct store D when it is NVFP4 input, BFlat16 output
        self.store_d_directly = False

        # Pipeline stages for epi_pipeline (accumulator update -> epilogue).
        self.num_epi_stage = 1

        # Setup A/B/D/Scale stage count in shared memory and ACC stage count in tensor memory
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
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.occupancy,
            self.store_d_directly,
            self.generate_dbias,
            acc_dtype=self.acc_dtype,
            sf2_dtype=self.sf2_dtype,
            sgm=self.sgm,
            sgn=self.sgn,
            sgk=self.sgk,
            num_epi_stage=self.num_epi_stage,
            row_dsmem=self.row_dsmem,
        )

        # Second-level scale pipeline runs in lockstep with the AB pipeline
        self.num_scale_stage = self.num_ab_stage

        # Compute A/B/D/Scale shared memory layout
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
            self.epi_tile,
            self.num_c_stage,
        )

        if cutlass.const_expr(not self.store_d_directly):
            self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.d_dtype,
                self.d_layout,
                self.epi_tile,
                self.num_d_stage,
            )
        else:
            self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.d_dtype,
                self.d_layout,
                self.epi_tile,
                1,
            )

        # Full-CTA-tile SMEM buffer for the final accumulator; the "stage" mode of the
        # epi layout doubles as the subtile-slot index (subtile_cnt slots per epi stage).
        # Always allocated (unlike sD, which is conditional on store_d_directly).
        self.final_acc_subtile_cnt = (self.cta_tile_shape_mnk[0] // cute.size(self.epi_tile[0])) * (self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1]))
        self.final_acc_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype,
            self.d_layout,
            self.epi_tile,
            self.final_acc_subtile_cnt * self.num_epi_stage,
        )

        # Compute second level scale factor layout
        # Store the per-tile scale block sizes (each clamped to the CTA tile) so
        # the device kernel builds C-shaped scale views that match the CTA tile
        # exactly, even when sg{m,n,k} > the tile extent (e.g. sgn=512 with a
        # 128-wide N tile spans one scale block, broadcast across the tile).
        if self.sgm < self.cta_tile_shape_mnk[0]:
            size_m = self.sgm
        else:
            size_m = self.cta_tile_shape_mnk[0]

        if self.sgn < self.cta_tile_shape_mnk[1]:
            size_n = self.sgn
        else:
            size_n = self.cta_tile_shape_mnk[1]

        if self.sgk < self.cta_tile_shape_mnk[2]:
            size_k = self.sgk
        else:
            size_k = self.mma_tiler[2]

        self.scale_size_m = size_m
        self.scale_size_n = size_n
        self.scale_size_k = size_k

        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // size_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[1] // size_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // size_k

        self.sfa2_smem_layout_staged = cute.make_layout(
            (
                (size_m, self.scale_m_per_tile),
                (size_k, self.scale_k_per_tile),
                self.num_ab_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        self.sfb2_smem_layout_staged = cute.make_layout(
            (
                (size_n, self.scale_n_per_tile),
                (size_k, self.scale_k_per_tile),
                self.num_ab_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1 and self.mma_tiler[1] == 256

        # To prefetch more accumulator when overlapping_accum is enabled in epilogue
        self.epilogue_prefetch_more = self.d_dtype.width == 8 and self.a_dtype.width == 8

        # To generate dprob
        self.generate_dprob = True

        # Use ptx fp8 fp32 convert
        self.use_fp8_ptx_cvt = False

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage

        self.epi_tile_n_required = cute.size(self.epi_tile[1])
        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = (self.num_sf_tmem_cols + self.epi_tile_n_required - 1) // self.epi_tile_n_required - 1

    def get_desc_workspace_bytes(self) -> int:
        """Return descriptor workspace size in bytes."""
        if self.weight_mode == MoEWeightMode.DISCRETE:
            from ..moe_utils import DiscreteWeightTensormapConstructor

            return DiscreteWeightTensormapConstructor.get_workspace_size(self.expert_cnt)
        return 0

    def get_workspace_bytes(self, sfd2_sync_counter_cnt: int = 0) -> int:
        """Return descriptor workspace plus optional dynamic scheduler state,
        plus optional sfd2 cross-tile arrival counters (one u32 per
        (global m CTA tile, sfd2 n-block); the caller sizes the count — see
        the wrapper — and it is nonzero only for quantized-D output with
        sfd2_n_contrib > 1, i.e. sgn > mma_tiler_n)."""
        desc_workspace_bytes = self.get_desc_workspace_bytes()
        dynamic_sched_bytes = 4 if self.use_dynamic_sched else 0
        return desc_workspace_bytes + dynamic_sched_bytes + 4 * sfd2_sync_counter_cnt

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        counter_addr = workspace_ptr.toint() + self.get_desc_workspace_bytes()
        return cute.make_ptr(
            cutlass.Int32,
            counter_addr,
            AddressSpace.gmem,
            assumed_align=4,
        )

    @cute.jit
    def _get_sfd2_counter_ptr(self, workspace_ptr):
        """Base of the sfd2 arrival-counter array (after the tensormap
        descriptors and the optional dynamic-sched counter)."""
        counter_addr = workspace_ptr.toint() + self.get_desc_workspace_bytes() + (4 if self.use_dynamic_sched else 0)
        return cute.make_ptr(
            cutlass.Int32,
            counter_addr,
            AddressSpace.gmem,
            assumed_align=4,
        )

    @cute.kernel
    def sfd2_counter_zero_kernel(self, workspace_ptr, counter_cnt: Int32):
        """Zero the sfd2 arrival counters before the main kernel (each launch
        must start from 0: pass 2 spins until a counter reaches
        sfd2 contributor count for its block)."""
        bidx = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        gdim = cute.arch.grid_dim()[0]
        counters = cute.make_tensor(
            self._get_sfd2_counter_ptr(workspace_ptr),
            cute.make_layout((counter_cnt,)),
        )
        idx = Int32(bidx * 128 + tidx)
        stride = Int32(gdim * 128)
        while idx < counter_cnt:
            counters[idx] = Int32(0)
            idx += stride

    @cute.kernel
    def helper_kernel(
        self,
        ptrs_b: cute.Pointer,  # Device pointer to int64[expert_cnt] array of B addresses
        ptrs_sfb: cute.Pointer,  # Device pointer to int64[expert_cnt] array of SFB addresses
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
        """Pre-main-kernel initialization.

        Launched with grid=(expert_cnt, 1, 1) for discrete mode, or
        grid=(1, 1, 1) for dense+dynamic mode.

        Discrete weight: each block builds B/SFB TMA descriptors for one expert.
        Dynamic sched: block 0 resets the atomic tile counter to 0.
        """
        expert_idx = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            b_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma_arg.thr_id)
            sfb_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma_arg.thr_id)

            b_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_b.toint(), AddressSpace.gmem, assumed_align=8), cute.make_layout((self.expert_cnt,))
            )
            sfb_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_sfb.toint(), AddressSpace.gmem, assumed_align=8), cute.make_layout((self.expert_cnt,))
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

            workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            store_tma_desc(tma_atom_b, workspace.get_ptr("b", expert_idx))

            sfb_ptr_val = sfb_ptr_tensor[expert_idx]
            sfb_ptr = cute.make_ptr(self.sf_dtype, sfb_ptr_val, AddressSpace.gmem)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb_tensor_i = cute.make_tensor(sfb_ptr, sfb_layout)
            tma_atom_sfb, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_tma_op_arg,
                sfb_tensor_i,
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
        sfb,  # Dense: cute.Tensor         | Discrete: cute.Pointer to int64[]
        sfb2,  # Dense: cute.Tensor         | Discrete: cute.Pointer to int64[]
        n: Int32,  # Ignored for dense mode
        k: Int32,  # Ignored for dense mode
        b_stride_size: cutlass.Int64,  # Ignored for dense mode
        b_major_mode: cutlass.Constexpr,  # Ignored for dense mode
        workspace_ptr,  # Descriptor workspace, plus dynamic scheduler counter when enabled
        c: cute.Tensor,
        d: cute.Tensor,
        sfa: cute.Tensor,
        sfa2: cute.Tensor,
        sfd2_up: cute.Tensor,
        sfd2_gate: cute.Tensor,
        sfd_row_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
        linear_offset: Float32,
        dbias_tensor: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM.

        Dense mode: ``b`` and ``sfb`` are 3-D cute.Tensor (N, K, L).
        Discrete mode: ``b`` and ``sfb`` are cute.Pointer to device int64[]
        arrays of per-expert base addresses; ``n``, ``k``, ``b_stride_size``,
        ``b_major_mode`` describe the uniform per-expert layout.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type  # B must match A dtype
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.sf2_dtype: Type[cutlass.Numeric] = sfa2.element_type
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

        # Quantized-D mode is dtype-inferred. Derived BEFORE _setup_attributes()
        # because the rowwise-sfd2 DSMEM eligibility below feeds smem sizing
        # (_compute_stages).
        # all() instead of an `and` chain: this runs inside the @cute.jit
        # traced __call__, where the DSL AST pass rewrites `and` into a
        # cutlass.Boolean — but these flags feed SharedStorage sizing and
        # _compute_stages, which need plain python bools.
        self.generate_sfd = _is_quant_dtype_combo(self.a_dtype, self.sf_dtype, self.d_dtype)
        # ROWWISE sfd2 DSMEM: active when one block's N work-tile contributors
        # are exactly one cluster's N-peers (sgn/cta_n == cluster_n > 1). The
        # quantized-D row-scale fold is optional in this kernel (unlike the
        # rht_2 source, where it is unconditionally on), so the quant + sfd2
        # terms are AND-ed in.
        self.row_dsmem = all(
            (
                self.dsmem_rowwise,
                self.generate_sfd,
                self.generate_sfd2,
                self.cluster_shape_mn[1] > 1,
                self.sfd2_n_contrib == self.cluster_shape_mn[1],
            )
        )

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # ---- SFA2 layout ----
        sfa2_tensor = cute.make_tensor(
            sfa2.iterator,
            cute.make_layout(
                (
                    (self.sgm, sfa2.shape[0]),
                    (self.sgk, sfa2.shape[1]),
                    sfa2.shape[2],
                ),
                stride=(
                    (0, sfa2.layout.stride[0]),
                    (0, sfa2.layout.stride[1]),
                    sfa2.layout.stride[2],
                ),
            ),
        )

        # ---- SFD2 layout ---- (None when sfd2 generation is off)
        sfd2_up_tensor = None
        sfd2_gate_tensor = None
        if cutlass.const_expr(self.generate_sfd2):
            # Inner N mode is 2*sgn: sgn counts deinterleaved f-columns, and
            # the tensor is indexed in interleaved D-space coordinates where
            # one block spans 2*sgn columns (sgn gate + sgn up 32-col bands).
            sfd2_up_tensor = cute.make_tensor(
                sfd2_up.iterator,
                cute.make_layout(
                    (
                        (self.sgm, sfd2_up.shape[0]),
                        (2 * self.sgn, sfd2_up.shape[1]),
                        sfd2_up.shape[2],
                    ),
                    stride=(
                        (0, sfd2_up.layout.stride[0]),
                        (0, sfd2_up.layout.stride[1]),
                        sfd2_up.layout.stride[2],
                    ),
                ),
            )

            sfd2_gate_tensor = cute.make_tensor(
                sfd2_gate.iterator,
                cute.make_layout(
                    (
                        (self.sgm, sfd2_gate.shape[0]),
                        (2 * self.sgn, sfd2_gate.shape[1]),
                        sfd2_gate.shape[2],
                    ),
                    stride=(
                        (0, sfd2_gate.layout.stride[0]),
                        (0, sfd2_gate.layout.stride[1]),
                        sfd2_gate.layout.stride[2],
                    ),
                ),
            )

        c1 = cutlass.Int32(1)
        c0 = cutlass.Int64(0)
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            sfb2_ptr_typed = sfb2.iterator
            sfb2_shape_0 = sfb2.shape[0]
            sfb2_shape_1 = sfb2.shape[1]
            sfb2_mode_2 = sfb2.shape[2]
            sfb2_stride_0 = sfb2.layout.stride[0]
            sfb2_stride_1 = sfb2.layout.stride[1]
            sfb2_stride_2 = sfb2.layout.stride[2]
        else:  # DISCRETE mode
            # In discrete mode, sfb2 is a pointer to array of per-expert pointers (Int64[])
            # Create a template tensor with correct element type (sf2_dtype) for get_gmem_tensor
            # The pointer address is the pointer array address; get_gmem_tensor will load actual per-expert pointer
            sfb2_ptr_typed = cute.make_ptr(
                self.sf2_dtype,  # Correct data type so element_type is right
                sfb2.toint(),  # Pointer array address (will be reinterpreted in get_gmem_tensor)
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            # Use computed scale values for template shape/stride
            scale_n = cute.ceil_div(n, self.sgn)
            scale_k = cute.ceil_div(k, self.sgk)
            sfb2_shape_0 = scale_n
            sfb2_shape_1 = scale_k
            sfb2_mode_2 = c1
            sfb2_stride_0 = c1
            sfb2_stride_1 = scale_n
            sfb2_stride_2 = c0

        # ---- SFB2 layout ----
        # Dense: use actual tensor shape/stride
        # Discrete: use template shape/stride (get_gmem_tensor will load per-expert pointer)
        sfb2_tensor = cute.make_tensor(
            sfb2_ptr_typed,
            cute.make_layout(
                (
                    (self.sgn, sfb2_shape_0),
                    (self.sgk, sfb2_shape_1),
                    sfb2_mode_2,
                ),
                stride=(
                    (0, sfb2_stride_0),
                    (0, sfb2_stride_1),
                    sfb2_stride_2,
                ),
            ),
        )

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

            # Creating a template layout and B tensor for a single expert
            b_template_layout = cute.make_layout((n, k, c1), stride=b_template_stride)
            b_ptr_typed = cute.make_ptr(self.b_dtype, b.toint(), AddressSpace.gmem, assumed_align=16)
            b = cute.make_tensor(b_ptr_typed, b_template_layout)

            sfb_ptr_typed = cute.make_ptr(self.sf_dtype, sfb.toint(), AddressSpace.gmem, assumed_align=16)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb = cute.make_tensor(sfb_ptr_typed, sfb_layout)

        # Setup sfa tensor by filling A tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        # Dimensions of output
        m, n_d, l = cute.shape(d)

        # self.generate_sfd (the quantized-D mode) is derived above, before
        # _setup_attributes(), so the rowwise-sfd2 DSMEM smem sizing sees it.

        if cutlass.const_expr(self.d_deinterleaved and not self.generate_sfd):
            # The deinterleaved layout is only harness-validated for the
            # quantized-D + sfd2 configuration; check_support enforces this
            # host-side, this is the trace-time backstop.
            raise ValueError("d_deinterleaved requires quantized D output")

        if cutlass.const_expr(dbias_tensor is not None and self.d_deinterleaved):
            # Deinterleave dbias's n axis (mode 1). The bf16x2 atomics pair
            # adjacent even/odd columns, which stay inside one 32-col band.
            dbias_tensor = _deinterleave_n_coord_tensor(dbias_tensor, mode=1)

        self._sfd2_sync_active = self.generate_sfd and self.generate_sfd2 and self.sfd2_n_contrib > 1

        if cutlass.const_expr(self._sfd2_sync_active):
            # One counter per (global m CTA tile, sfd2 n-block); must match
            # the wrapper's workspace sizing and the epilogue's index math.
            _sfd2_n_tiles_f = cute.ceil_div(n_d, 2 * self.cta_tile_shape_mnk[1])
            _sfd2_n_j = cute.ceil_div(_sfd2_n_tiles_f, self.sfd2_n_contrib)
            _sfd2_counter_cnt = (m // self.cta_tile_shape_mnk[0]) * _sfd2_n_j
        else:
            _sfd2_counter_cnt = Int32(0)

        if cutlass.const_expr(self.generate_sfd):
            output_sfd_shape = (m, n_d, l)
            sfd_layout = blockscaled_utils.tile_atom_to_shape_SF(output_sfd_shape, self.sf_vec_size)
            sfd_row_tensor = cute.make_tensor(sfd_row_tensor.iterator, sfd_layout)

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

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint16,
        )

        # Setup TMA load for SFB
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
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        # Setup TMA store for C
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        self.tma_c_load_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            c,
            c_smem_layout,
            self.epi_tile,
        )

        # Setup TMA store for D
        tma_atom_d = None
        tma_tensor_d = None
        if cutlass.const_expr(not self.store_d_directly):
            d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
            tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                d,
                d_smem_layout,
                self.epi_tile,
            )
            if cutlass.const_expr(self.d_deinterleaved):
                tma_tensor_d = _deinterleave_n_coord_tensor(tma_tensor_d, n_static=self.deint_n)
        else:
            if cutlass.const_expr(self.d_deinterleaved):
                tma_tensor_d = _deinterleave_n_coord_tensor(d, n_static=self.deint_n)
            else:
                tma_tensor_d = d

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
        grid = MoESchedulerParams.get_grid_shape(sched_params, max_active_clusters)

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        # sD is not needed when storing D directly (and not generating SFD)
        sD_size = 0 if (not self.generate_sfd and self.store_d_directly) else cute.cosize(self.d_smem_layout_staged.outer)
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_scale_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_epi_stage * 2]
            # DSMEM rowwise-sfd2 handoff: one cluster_n-arrival mbarrier per
            # parity slot of sSfd2RowDsmem (every N-peer arrives after its
            # red.shared::cluster maxes for a tile have been fenced).
            row_dsmem_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 if self.row_dsmem else 0]
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
                cute.struct.MemRange[self.d_dtype, sD_size],
                self.buffer_align_bytes,
            ]
            sFinalAcc: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(self.final_acc_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # DSMEM rowwise-sfd2 reduction buffer (sgn/cta_n == cluster_n
            # only): 2 parity slots x [cta_m gate | cta_m up] per-row block
            # descales, max-reduced IN PLACE by every N-peer CTA via
            # red.shared::cluster — each CTA ends up holding its 128 rows'
            # final block descales locally (plain LDS before pass 2, no
            # gmem counter spin / volatile read-backs).
            sSfd2RowDsmem: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    2 * 2 * self.cta_tile_shape_mnk_d[0] if cutlass.const_expr(self.row_dsmem) else 0,
                ],
                16,
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
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFA2: cute.struct.Align[
                cute.struct.MemRange[self.sf2_dtype, cute.cosize(self.sfa2_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf2_dtype, cute.cosize(self.sfb2_smem_layout_staged)],
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
            # SFD row-scale store staging: one (128 rows x 128 values) SFD
            # store event in the gmem atom layout. The T2R row-per-thread
            # mapping scatters each thread's SF bytes at 16B stride in that
            # layout, so direct gmem stores waste 3/4 of every sector; the
            # block is exchanged through SMEM and written back linearly.
            sSFDRowStage: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    128 * 32 * 4 // self.sf_vec_size if cutlass.const_expr(self.generate_sfd) else 1,
                ],
                128 if cutlass.const_expr(self.generate_sfd) else 4,
            ]

        self.shared_storage = SharedStorage

        # Initialize per-expert B/SFB TMA descriptors in workspace
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
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

        # Zero the sfd2 arrival counters (same stream, so ordered before the
        # main kernel; each launch must restart the arrival protocol from 0).
        if cutlass.const_expr(self._sfd2_sync_active):
            self.sfd2_counter_zero_kernel(workspace_ptr, _sfd2_counter_cnt).launch(
                grid=(32, 1, 1),
                block=(128, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
            )

        # Launch the main kernel
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
            sfa2_tensor,
            sfb2_tensor,
            sfd2_up_tensor,
            sfd2_gate_tensor,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_d,
            tma_tensor_d,
            sfd_row_tensor,
            norm_const_tensor,
            padded_offsets,
            alpha,
            beta,
            prob,
            dprob,
            linear_offset,
            dbias_tensor,
            workspace_ptr,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfa2_smem_layout_staged,
            self.sfb2_smem_layout_staged,
            self.c_smem_layout_staged,
            self.d_smem_layout_staged,
            self.final_acc_smem_layout_staged,
            self.epi_tile,
            sched_params,
            epilogue_op,
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
            desc_workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            return DiscreteWeightScaledGemmSchedExtension(
                tensormap_ctor=desc_workspace,
                sf_vec_size=self.sf_vec_size,
            )
        else:
            return ContiguousAndConsistentGroupedGemmSchedExtension(
                sf_vec_size=self.sf_vec_size,
            )

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def amax_reduction_per_thread(self, vec_fp32, amax_fp32) -> None:
        vec_fp32_ssa = vec_fp32
        abs_acc_values_ir = cutlass._mlir.dialects.math.absf(vec_fp32_ssa.ir_value())
        abs_acc_values = type(vec_fp32_ssa)(abs_acc_values_ir, vec_fp32_ssa.shape, vec_fp32_ssa.dtype)
        subtile_amax = abs_acc_values.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
        return cute.arch.fmax(amax_fp32, subtile_amax)

    @cute.jit
    def second_level_scale_row_wise_gen(self, thread_tile_amax, tgSFD2):
        thread_gmem_ptr = tgSFD2.iterator.llvm_ptr
        _ = atomic_max_float32(ptr=thread_gmem_ptr, value=thread_tile_amax)

    @cute.jit
    def cvt_f32x4_to_f8x4_pack_i32(self, fp32x4, fp8_type, loc=None, ip=None):
        fp32x4 = fp32x4.load()
        src_vec4 = fp32x4.ir_value(loc=loc, ip=ip) if hasattr(fp32x4, "ir_value") else fp32x4

        src0 = Float32(vector.extract(src_vec4, [], [0])).ir_value(loc=loc, ip=ip)
        src1 = Float32(vector.extract(src_vec4, [], [1])).ir_value(loc=loc, ip=ip)
        src2 = Float32(vector.extract(src_vec4, [], [2])).ir_value(loc=loc, ip=ip)
        src3 = Float32(vector.extract(src_vec4, [], [3])).ir_value(loc=loc, ip=ip)

        cvt_instruction = ""
        if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
            cvt_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
        elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
            cvt_instruction = "cvt.rn.satfinite.e4m3x2.f32"
        else:
            with cute.arch.elect_one():
                cute.printf("error: unsupported fp8 element type")
            return

        asm_tmpl = (
            "{\n"
            "  .reg .b16 lo;\n"
            "  .reg .b16 hi;\n"
            f"  {cvt_instruction} lo, $2, $1;\n"
            f"  {cvt_instruction} hi, $4, $3;\n"
            "  mov.b32 $0, {lo, hi};\n"
            "}"
        )
        packed_i32 = llvm.inline_asm(
            T.i32(),
            [src0, src1, src2, src3],
            asm_tmpl,
            "=r,f,f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        return packed_i32

    @cute.jit
    def cvt_f32x4_to_f8x4(self, fp32x4, fp8x4, loc=None, ip=None):
        packed_i32 = self.cvt_f32x4_to_f8x4_pack_i32(fp32x4, fp8x4.element_type)
        fp8x4_i32 = cute.recast_tensor(fp8x4, cutlass.Int32)
        fp8x4_i32[0] = cutlass.Int32(packed_i32)
        return

    @cute.jit
    def cvt_f32_to_f8_to_f32(self, fp32x1, fp8_type, loc=None, ip=None):
        src_fp32 = Float32(fp32x1).ir_value(loc=loc, ip=ip)

        cvt_instruction_downcast = ""
        cvt_instruction_upcast = ""
        # Intermediate float type used to upcast the fp8 value back to f32.
        # ue8m0 needs bf16 (matching f32 exponent range); e4m3 must use f16
        # since PTX has no direct e4m3->bf16 conversion (only cvt.rn.f16x2.e4m3x2).
        upcast_vec_ty = "vector<2xbf16>"
        if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
            cvt_instruction_downcast = "cvt.rp.satfinite.ue8m0x2.f32"
            cvt_instruction_upcast = "cvt.rn.bf16x2.ue8m0x2"
            upcast_vec_ty = "vector<2xbf16>"
        elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
            cvt_instruction_downcast = "cvt.rn.satfinite.e4m3x2.f32"
            cvt_instruction_upcast = "cvt.rn.f16x2.e4m3x2"
            upcast_vec_ty = "vector<2xf16>"
        else:
            with cute.arch.elect_one():
                cute.printf("error: unsupported fp8 element type")
            return

        asm_tmpl = "{\n" "  .reg .b16 bf_lo;\n" f"  {cvt_instruction_downcast} bf_lo, 0f00000000, $1;\n" f"  {cvt_instruction_upcast}  $0, bf_lo;\n" "}"
        packed_i32 = llvm.inline_asm(
            T.i32(),
            [src_fp32],
            asm_tmpl,
            "=r,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        vec_hi_ty = ir.Type.parse(upcast_vec_ty)
        bf2_lo = llvm.bitcast(vec_hi_ty, packed_i32, loc=loc, ip=ip)
        h0 = vector.extract(bf2_lo, [], [0], loc=loc, ip=ip)
        dst_f32 = arith.extf(Float32.mlir_type, h0, loc=loc, ip=ip)

        return dst_f32

    @cute.jit
    def quant_sfd_row(
        self,
        tile_idx,
        tiled_copy_r2s,
        src,
        pvscale,
        norm_const,
        rcp_limit,
        tRSrD,
        sfd2_descale=None,
    ) -> None:
        # NOTE: `sfd2_descale` is the per-thread (per-row) second-level block
        # descale (block_amax / (E2M1_MAX*E4M3_MAX)), read back from the sfd2
        # gmem atomic-max reduction after pass 1 (see the pass-2 read-back in
        # the epilogue) so it covers the FULL block even when one block spans
        # multiple N work-tiles (sfd2_n_contrib > 1). Its reciprocal is folded
        # into the first-level (row-wise) SFD scale below, so the e4m3 SF
        # absorbs the per-block second-level magnitude.
        # Get absolute max across a vector and Compute SFD
        tCompute = cute.make_rmem_tensor(src.shape, self.acc_dtype)
        tCompute.store(src)
        tTR_rAcc_frg = cute.logical_divide(tCompute, cute.make_layout(self.sf_vec_size))
        acc_frg = tTR_rAcc_frg.load()
        abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
        abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)

        sfd2_scale = cute.arch.rcp_approx(sfd2_descale)

        for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
            pvscale[vi, None, tile_idx][0] = (
                abs_acc_frg[None, vi].reduce(
                    cute.ReductionOp.MAX,
                    cutlass.Float32(0.0),
                    0,  # Use 0.0 as init for abs values
                )
                * rcp_limit
                * norm_const
                * sfd2_scale
            )

        #
        # Compute quantized output values and convert to D type
        #

        fp32_max = cutlass.Float32(3.40282346638528859812e38)
        if cutlass.const_expr(self.vectorized_f32):
            for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                qpvscale_up_0 = self.cvt_f32_to_f8_to_f32(pvscale[vi, None, tile_idx][0], self.sf_dtype)
                qpvscale_up_1 = self.cvt_f32_to_f8_to_f32(pvscale[vi + 1, None, tile_idx][0], self.sf_dtype)
                acc_scale = cute.arch.mul_packed_f32x2(
                    (
                        cute.arch.rcp_approx(qpvscale_up_0),
                        cute.arch.rcp_approx(qpvscale_up_1),
                    ),
                    (norm_const, norm_const),
                )
                acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
                acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)
                vec0 = tTR_rAcc_frg[None, vi]
                vec1 = tTR_rAcc_frg[None, vi + 1]
                for ei in cutlass.range_constexpr(self.sf_vec_size):
                    (
                        vec0[ei],
                        vec1[ei],
                    ) = cute.arch.mul_packed_f32x2(
                        (vec0[ei], vec1[ei]),
                        (acc_scale_min0, acc_scale_min1),
                        rnd="rn",
                        ftz=False,
                    )
        else:
            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                qpvscale_up = self.cvt_f32_to_f8_to_f32(pvscale[vi, None, tile_idx][0], self.sf_dtype)
                acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
                acc_scale = fmin(acc_scale, fp32_max, nan=True)
                vec = tTR_rAcc_frg[None, vi]
                for ei in cutlass.range_constexpr(self.sf_vec_size):
                    vec[ei] = vec[ei] * acc_scale

        acc_vec = tiled_copy_r2s.retile(tCompute).load()
        if cutlass.const_expr(not self.use_fp8_ptx_cvt):
            tRSrD.store(acc_vec.to(self.d_dtype))
        else:
            tRSrD_i32 = cute.recast_tensor(tRSrD, cutlass.Int32)
            for ei in cutlass.range_constexpr(0, self.sf_vec_size, 4):
                fp32x4 = cute.make_rmem_tensor(4, cutlass.Float32)
                fp32x4[0] = acc_vec[ei + 0]
                fp32x4[1] = acc_vec[ei + 1]
                fp32x4[2] = acc_vec[ei + 2]
                fp32x4[3] = acc_vec[ei + 3]
                fp8x4_i32 = self.cvt_f32x4_to_f8x4_pack_i32(fp32x4, self.d_dtype)
                tRSrD_i32[ei // 4] = cutlass.Int32(fp8x4_i32)

    @cute.jit
    def stg_256(self, ptr, vec8_f32, *, loc=None, ip=None):
        """
        Store 8xf32 (256b) to global memory with L1::no_allocate.
        ptr: pointer (byte addressable)
        vec8_f32: vector<8xf32> to store
        """
        dst = ptr.ir_value(loc=loc, ip=ip) if hasattr(ptr, "ir_value") else ptr
        src = vec8_f32.ir_value(loc=loc, ip=ip) if hasattr(vec8_f32, "ir_value") else vec8_f32
        dummy = llvm.inline_asm(
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
            self.stg_256(dst_vf32x8[None, ei].iterator.llvm_ptr, src_vf32x8[None, ei].load())

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
                    # f32 dbias accumulator (TESTING ONLY, selected by the dbias
                    # tensor dtype): plain f32 atomics per column instead of the
                    # packed bf16x2 reduction.
                    if cutlass.const_expr(dbias_gmem_2d.element_type == cutlass.Float32):
                        _ = atomic_add_float32(
                            ptr=dbias_gmem_2d[(expert_idx, n_offset, None)].iterator.llvm_ptr,
                            value=cta_sum_a,
                        )
                        _ = atomic_add_float32(
                            ptr=dbias_gmem_2d[(expert_idx, n_offset + 1, None)].iterator.llvm_ptr,
                            value=cta_sum_b,
                        )
                    else:
                        gmem_ptr = dbias_gmem_2d[(expert_idx, n_offset, None)].iterator.llvm_ptr
                        atomic_add_bf16x2(gmem_ptr, cta_sum_a, cta_sum_b)
        else:
            if n_offset < dbias_n_total:
                if cutlass.const_expr(dbias_gmem_2d.element_type == cutlass.Float32):
                    _ = atomic_add_float32(
                        ptr=dbias_gmem_2d[(expert_idx, n_offset, None)].iterator.llvm_ptr,
                        value=sum_a,
                    )
                    _ = atomic_add_float32(
                        ptr=dbias_gmem_2d[(expert_idx, n_offset + 1, None)].iterator.llvm_ptr,
                        value=sum_b,
                    )
                else:
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
        # Compile-time GLU params (baked from the kernel object).
        dglu_alpha = self.glu_alpha
        dglu_clamp_max = self.glu_clamp_max
        dglu_clamp_min = self.glu_clamp_min
        LOG2_E = cutlass.Float32(1.4426950408889634)
        if cutlass.const_expr(self.vectorized_f32):
            ab1_vec_f32 = cute.make_rmem_tensor(ab1_vec_load.shape, cutlass.Float32)
            ab2_vec_f32 = cute.make_rmem_tensor(ab2_vec_load.shape, cutlass.Float32)
            for i in cutlass.range_constexpr(cute.size(ab1_vec_load)):
                ab1_vec_f32[i] = ab1_vec_load[i].to(cutlass.Float32)
                ab2_vec_f32[i] = ab2_vec_load[i].to(cutlass.Float32)
            ab1_vec_load = ab1_vec_f32
            ab2_vec_load = ab2_vec_f32
            d1_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            d2_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            for i in cutlass.range(0, cute.size(acc_vec), 2, unroll_full=True):
                # Apply scaling factors for FP8
                (
                    acc_vec[i + 0],
                    acc_vec[i + 1],
                ) = cute.arch.mul_packed_f32x2(
                    (acc_vec[i + 0], acc_vec[i + 1]),
                    (square_alpha, square_alpha),
                    rnd="rn",
                    ftz=False,
                )
                if cutlass.const_expr(dglu_alpha is not None and dglu_alpha != 1.0):
                    (
                        acc_vec[i + 0],
                        acc_vec[i + 1],
                    ) = cute.arch.mul_packed_f32x2(
                        (acc_vec[i + 0], acc_vec[i + 1]),
                        (dglu_alpha, dglu_alpha),
                        rnd="rn",
                        ftz=False,
                    )

                if cutlass.const_expr(dglu_clamp_max is not None and dglu_clamp_min is not None):
                    ab1_vec_load[i + 0] = fmin(ab1_vec_load[i + 0], dglu_clamp_max)
                    ab1_vec_load[i + 1] = fmin(ab1_vec_load[i + 1], dglu_clamp_max)
                    ab2_vec_load[i + 0] = fmin(ab2_vec_load[i + 0], dglu_clamp_max)
                    ab2_vec_load[i + 1] = fmin(ab2_vec_load[i + 1], dglu_clamp_max)
                    ab2_vec_load[i + 0] = fmax(ab2_vec_load[i + 0], dglu_clamp_min)
                    ab2_vec_load[i + 1] = fmax(ab2_vec_load[i + 1], dglu_clamp_min)

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
            ab1_f32 = cute.make_rmem_tensor(ab1_vec_load.shape, cutlass.Float32)
            ab2_f32 = cute.make_rmem_tensor(ab2_vec_load.shape, cutlass.Float32)
            for i in cutlass.range_constexpr(cute.size(ab1_vec_load)):
                ab1_f32[i] = ab1_vec_load[i].to(cutlass.Float32)
                ab2_f32[i] = ab2_vec_load[i].to(cutlass.Float32)
                if cutlass.const_expr(dglu_clamp_max is not None and dglu_clamp_min is not None):
                    ab1_f32[i] = fmin(ab1_f32[i], dglu_clamp_max)
                    ab2_f32[i] = fmin(ab2_f32[i], dglu_clamp_max)
                    ab2_f32[i] = fmax(ab2_f32[i], dglu_clamp_min)
            ab1_vec_load = ab1_f32
            ab2_vec_load = ab2_f32

            acc_vec = acc_vec.load()
            ab1_vec_load = ab1_vec_load.load()
            ab2_vec_load = ab2_vec_load.load()

            acc_vec = acc_vec * square_alpha  # apply scale for A*B
            if cutlass.const_expr(dglu_alpha is not None and dglu_alpha != 1.0):
                acc_vec = acc_vec * dglu_alpha
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
        linear_offset: Float32,
        dprob_swiglu: Optional[cute.Tensor] = None,
    ):
        # Compile-time GLU params (baked from the kernel object).
        dglu_alpha = self.glu_alpha
        dglu_clamp_max = self.glu_clamp_max
        dglu_clamp_min = self.glu_clamp_min
        if cutlass.const_expr(dglu_clamp_max is None and dglu_clamp_min is None):
            dglu_clamp_max = 7.0
            dglu_clamp_min = -7.0
        if cutlass.const_expr(dglu_alpha is None):
            dglu_alpha = 1.702
        LOG2_E = cutlass.Float32(1.4426950408889634)
        x_dtype = x1_vec_load.element_type
        geglu_max_value = x_dtype(dglu_clamp_max)
        geglu_min_value = x_dtype(dglu_clamp_min)
        geglu_max_value_f32 = cutlass.Float32(dglu_clamp_max)
        geglu_min_value_f32 = cutlass.Float32(dglu_clamp_min)
        zero_x_dtype = x_dtype(0.0)
        fmul2 = partial(cute.arch.mul_packed_f32x2, rnd="rn", ftz=False)
        fadd2 = partial(cute.arch.add_packed_f32x2, rnd="rn", ftz=False)
        dglu_alpha_tuple = (dglu_alpha, dglu_alpha)
        ones2 = (1.0, 1.0)
        mprob2 = (mProb, mProb)
        linear_offset2 = (linear_offset, linear_offset)

        if cutlass.const_expr(self.vectorized_f32):
            dx1_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            dx2_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            for i in cutlass.range(0, cute.size(acc_vec), 2, unroll_full=True):
                acc = (acc_vec[i], acc_vec[i + 1])
                x1_0 = x1_vec_load[i]
                x1_1 = x1_vec_load[i + 1]
                x2_0 = x2_vec_load[i]
                x2_1 = x2_vec_load[i + 1]

                y1_0 = 0.0
                y1_1 = 0.0
                y2_0 = 0.0
                y2_1 = 0.0
                if cutlass.const_expr(x_dtype == cutlass.BFloat16):
                    y1_0, y1_1 = fmin_bf16x2(x1_0, x1_1, geglu_max_value, geglu_max_value)
                    y2_0, y2_1 = fmax_bf16x2(x2_0, x2_1, geglu_min_value, geglu_min_value)
                    y2_0, y2_1 = fmin_bf16x2(y2_0, y2_1, geglu_max_value, geglu_max_value)
                    y1_0 = Float32(y1_0)
                    y1_1 = Float32(y1_1)
                    y2_0 = Float32(y2_0)
                    y2_1 = Float32(y2_1)
                else:
                    y1_0 = fmin(x1_0, geglu_max_value)
                    y1_1 = fmin(x1_1, geglu_max_value)
                    y2_0 = fmin(x2_0, geglu_max_value)
                    y2_1 = fmin(x2_1, geglu_max_value)
                    y2_0 = fmax(y2_0, geglu_min_value)
                    y2_1 = fmax(y2_1, geglu_min_value)
                    y1_0 = Float32(y1_0)
                    y1_1 = Float32(y1_1)
                    y2_0 = Float32(y2_0)
                    y2_1 = Float32(y2_1)

                y1 = (y1_0, y1_1)
                y2 = (y2_0, y2_1)

                # y1 = dglu_alpha * x1
                y1_scaled = fmul2(y1, dglu_alpha_tuple)

                sigmoid_out_0 = sigmoid_f32(y1_scaled[0], fastmath=True)
                sigmoid_out_1 = sigmoid_f32(y1_scaled[1], fastmath=True)

                # g * sigmoid_out
                acc_mul_sigmoid_out = fmul2(acc, (sigmoid_out_0, sigmoid_out_1))
                acc_mul_sigmoid_prob = fmul2(acc_mul_sigmoid_out, mprob2)

                # y1 = 1 + dglu_alpha * y1 * (1 - sigmoid_out)
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
                # dy1 = g * sigmoid_out * (y2 + linear_offset) * (1 + dglu_alpha * y1 * (1 - sigmoid_out)) * mProb
                dy1_0, dy1_1 = fmul2((dy1_pre_0, dy1_pre_1), y1_scaled)
                dy1_0, dy1_1 = fmul2((dy1_0, dy1_1), mprob2)

                x1_filter_0 = y1_0 if x1_0 <= geglu_max_value else cutlass.Float32(0.0)
                x1_filter_1 = y1_1 if x1_1 <= geglu_max_value else cutlass.Float32(0.0)

                dx1_vec[i], dx1_vec[i + 1] = fmul2((dy1_0, dy1_1), (cutlass.Float32(x1_filter_0), cutlass.Float32(x1_filter_1)))

                # dy2 = g * y1 * sigmoid_out * mProb
                dy2_0, dy2_1 = fmul2(y1, acc_mul_sigmoid_prob)
                x2_filter_0 = x2_0 if x2_0 <= geglu_max_value else x_dtype(0.0)
                x2_filter_1 = x2_1 if x2_1 <= geglu_max_value else x_dtype(0.0)
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
            acc_vec = acc_vec.load()
            x1_vec_load = x1_vec_load.load().to(cutlass.Float32)
            x2_vec_load = x2_vec_load.load().to(cutlass.Float32)
            dx1_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
            dx2_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)

            # y1 = clamp(x1, max=7.0); y2 = clamp(x2, min=-7.0, max=7.0)
            for i in cutlass.range_constexpr(element_count):
                # dgeglu applies alpha inside the sigmoid, not as an acc scale
                # (matching the vectorized branch / reference).
                fc2_dgrad = acc_vec[i]
                g = fc2_dgrad * mProb
                y1 = min(x1_vec_load[i], geglu_max_value_f32)
                y2 = min(x2_vec_load[i], geglu_max_value_f32)
                y2 = max(y2, geglu_min_value_f32)

                sigmoid_out = sigmoid_f32(y1 * dglu_alpha, fastmath=True)

                dy1 = g * sigmoid_out * (1 + dglu_alpha * y1 * (1 - sigmoid_out)) * (y2 + linear_offset)
                dy2 = g * y1 * sigmoid_out

                x1_filter = x1_vec_load[i] if x1_vec_load[i] <= geglu_max_value_f32 else 0.0
                x2_filter = x2_vec_load[i] if x2_vec_load[i] <= geglu_max_value_f32 else 0.0
                x2_filter = y2 if x2_filter >= geglu_min_value_f32 else 0.0

                dx1_vec[i] = x1_filter * dy1
                dx2_vec[i] = x2_filter * dy2

                if cutlass.const_expr(self.generate_dprob):
                    prob_grad = y1 * sigmoid_out * (y2 + linear_offset) * fc2_dgrad
                    dprob_swiglu[i] = prob_grad

            return dx1_vec.load(), dx2_vec.load(), dprob_swiglu.load()

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
        sfa2_tensor: cute.Tensor,
        sfb2_tensor,  # can be cute.Tensor or cute.Pointer
        sfd2_up_tensor: cute.Tensor,
        sfd2_gate_tensor: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        mSFDRow_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
        linear_offset: Float32,
        mDbias_tensor: Optional[cute.Tensor],
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfa2_smem_layout_staged: cute.Layout,
        sfb2_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        final_acc_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        total_tokens = padded_offsets[self.expert_cnt - 1]

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfb)
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

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)

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
        # MMA warp produces partial accumulators -> Accumulator update warp consumes
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.accumulator_update_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize mainloop scale_pipeline (barrier) and states
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize epi_pipeline (barrier) connecting accumulator update -> epilogue
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * len(self.accumulator_update_warp_id))
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * len(self.epilog_warp_id))
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_mbar_ptr.data_ptr(),
            num_stages=self.num_epi_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
        )

        # Initialize mainloop scale_pipeline (barrier) and states
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
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

        # DSMEM rowwise-sfd2 reduction state (sgn/cta_n == cluster_n): the
        # cluster_n N-peers red.shared::cluster-max their row-descale
        # partials into EACH peer's sSfd2RowDsmem parity slot; a
        # cluster_n-arrival mbarrier per parity replaces the gmem counter +
        # acquire spin between pass 1 and pass 2.
        if cutlass.const_expr(self.row_dsmem):
            sSfd2RowDsmem = storage.sSfd2RowDsmem.get_tensor(cute.make_layout((2 * 2 * self.cta_tile_shape_mnk_d[0],)))
            row_dsmem_mbar = storage.row_dsmem_mbar_ptr.data_ptr()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(row_dsmem_mbar + 0, self.cluster_shape_mn[1])
                    cute.arch.mbarrier_init(row_dsmem_mbar + 1, self.cluster_shape_mn[1])
                cute.arch.mbarrier_init_fence()
                _rdsm_lane = cute.arch.lane_idx()
                for _rdsm_zi in cutlass.range_constexpr(2 * 2 * self.cta_tile_shape_mnk_d[0] // 32):
                    sSfd2RowDsmem[_rdsm_zi * 32 + _rdsm_lane] = Float32(0.0)
                # Publish the zeroes at cluster scope: the peer's first reds
                # into this buffer follow its cluster_wait (the relaxed
                # arrive below carries no release — this fence does).
                cute.arch.fence_acq_rel_cluster()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/D/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sD = None
        if cutlass.const_expr(not self.store_d_directly):
            sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        # Full-CTA-tile final accumulator buffer (always allocated)
        sFinalAcc = storage.sFinalAcc.get_tensor(final_acc_smem_layout_staged.outer, swizzle=final_acc_smem_layout_staged.inner)
        # bf16 alias of the same bytes for the NVFP4 dGLU relay (see the epilogue).
        # Built here, outside the warp-dispatch `if`: `storage` is a plain Python
        # struct the 4.5 wheel cannot carry across a dynamic-if region boundary.
        sFinalAccBf16 = None
        if cutlass.const_expr(self.generate_sfd):
            dglu_bf16_layout = sm100_utils.make_smem_layout_epi(
                cutlass.BFloat16,
                self.d_layout,
                self.epi_tile,
                self.final_acc_subtile_cnt * 2 * self.num_epi_stage,
            )
            sFinalAccBf16 = storage.sFinalAcc.get_tensor(
                dglu_bf16_layout.outer,
                swizzle=dglu_bf16_layout.inner,
                dtype=cutlass.BFloat16,
            )
        # SFD row-stage SMEM tensor: same 4.5-wheel constraint as sFinalAccBf16 —
        # derive it from `storage` out here (the epilogue's dynamic warp-dispatch
        # `if` cannot carry the plain `storage` struct across its region boundary).
        sSFDRowStage_flat = None
        if cutlass.const_expr(self.generate_sfd):
            # regPerSubtile == 4 in the epilogue SFD path.
            _sfd_stage_size = 128 * 32 * 4 // self.sf_vec_size
            sSFDRowStage_flat = storage.sSFDRowStage.get_tensor(cute.make_layout(_sfd_stage_size))
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA2 = storage.sSFA2.get_tensor(sfa2_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB2 = storage.sSFB2.get_tensor(sfb2_smem_layout_staged)
        # (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        #
        # Preparing LDGSTS partition for second level scale factor tensor
        #
        atom_scale2_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            self.sf2_dtype,
            num_bits_per_copy=self.sf2_dtype.width,
        )
        tiled_copy_sfa2 = cute.make_tiled_copy_tv(atom_scale2_copy, cute.make_layout((32,)), cute.make_layout((1,)))
        tiled_copy_sfb2 = cute.make_tiled_copy_tv(atom_scale2_copy, cute.make_layout((32,)), cute.make_layout((1,)))

        thr_copy_sfa2 = tiled_copy_sfa2.get_slice(lane_idx)
        thr_copy_sfb2 = tiled_copy_sfb2.get_slice(lane_idx)
        tAsSFA2 = thr_copy_sfa2.partition_D(sSFA2)
        tBsSFB2 = thr_copy_sfb2.partition_D(sSFB2)

        #
        # Scale viewed as C tensor
        #
        # N/M extents use the CTA-tile-clamped scale block size (scale_size_*),
        # so the view spans exactly one CTA tile (size * per_tile == cta extent)
        # regardless of whether sg{m,n} exceed the tile (e.g. sgn=512 > 128).
        sSFA2_view_as_C_layout = cute.make_layout(
            (
                (self.scale_size_m, self.scale_m_per_tile),
                self.cta_tile_shape_mnk[1],
                self.num_scale_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFB2_view_as_C_layout = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                (self.scale_size_n, self.scale_n_per_tile),
                self.num_scale_stage,
            ),
            stride=(0, (0, 1), self.scale_n_per_tile),
        )
        sSFA2_view_as_C = cute.make_tensor(sSFA2.iterator, sSFA2_view_as_C_layout)
        sSFB2_view_as_C = cute.make_tensor(sSFB2.iterator, sSFB2_view_as_C_layout)

        # Calculating number of k_tiles which share the same scale factor
        k_tile_same_scale_factor = self.sgk // self.mma_tiler[2]
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
            cute.arch.setmaxregister_decrease(self.num_regs_sched_warps)
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
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
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
                real_sfa, _ = ext.get_gmem_tensor("sfa", mSFA_mkl, padded_offsets, work_tile_info)
                real_sfb, desc_ptr_sfb = ext.get_gmem_tensor("sfb", mSFB_nkl, padded_offsets, work_tile_info)

                # local_tile on per-expert tensors
                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                gSFA_mkl = cute.local_tile(real_sfa, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gSFB_nkl = cute.local_tile(real_sfb, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))

                # MMA partition
                thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
                tCgA = thr_mma.partition_A(gA_mkl)
                tCgB = thr_mma.partition_B(gB_nkl)
                tCgSFA = thr_mma.partition_A(gSFA_mkl)
                tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

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
                # TMA partition SFA
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
                # TMA partition SFB
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

                # Slice to per mma tile index (L=0 since domain already offset'd)
                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_m, None, 0)]
                slice_n = mma_tile_coord_n
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_n // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, 0)]

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
                    tAgSFA_k = tAgSFA_slice[(None, k_tile)]
                    tBgSFB_k = tBgSFB_slice[(None, k_tile)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, ab_producer_state.index)]

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
                    # TMA load SFA (contiguous, global desc)
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    # TMA load SFB (discrete, per-expert desc from workspace)
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                        tma_desc_ptr=desc_ptr_sfb,
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

        if warp_idx == self.scale_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)

            # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] sfa2_tensor: {sfa2_tensor}")
            # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] sfb2_tensor: {sfb2_tensor}")
            ext = self._make_extension(workspace_ptr)
            scale_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_scale_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            scale_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_scale_stage)

            while is_valid_tile:

                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                mSFA2_mkl_current, _ = ext.get_gmem_tensor("sfa2", sfa2_tensor, padded_offsets, work_tile_info)
                mSFB2_nkl_current, _ = ext.get_gmem_tensor("sfb2", sfb2_tensor, padded_offsets, work_tile_info)

                # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] mSFA2_mkl_current.layout: {mSFA2_mkl_current.layout}")
                # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] mSFB2_nkl_current.layout: {mSFB2_nkl_current.layout}")

                # if tidx == 352:
                #    cute.printf("bidx: {}, mSFA2_mkl_current: {}", bidz, mSFA2_mkl_current)
                #    cute.printf("bidx: {}, mSFB2_nkl_current: {}", bidz, mSFB2_nkl_current)

                gSFA2_mkl = cute.local_tile(mSFA2_mkl_current, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)), (None, None, None))
                gSFB2_nkl = cute.local_tile(mSFB2_nkl_current, cute.slice_(self.cta_tile_shape_mnk, (0, None, None)), (None, None, None))

                # Create coordinate tensors
                cSFA2_mkl = cute.make_identity_tensor(cute.shape(mSFA2_mkl_current))
                cSFB2_nkl = cute.make_identity_tensor(cute.shape(mSFB2_nkl_current))
                cSFA2 = cute.local_tile(cSFA2_mkl, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)), (None, None, None))
                cSFB2 = cute.local_tile(cSFB2_nkl, cute.slice_(self.cta_tile_shape_mnk, (0, None, None)), (None, None, None))
                # Partition tensors
                tAgSFA2_mkl = thr_copy_sfa2.partition_S(gSFA2_mkl)
                tBgSFB2_nkl = thr_copy_sfb2.partition_S(gSFB2_nkl)
                tAcSFA2 = thr_copy_sfa2.partition_S(cSFA2)
                tBcSFB2 = thr_copy_sfb2.partition_S(cSFB2)

                mma_tile_coord_mnl = (work_tile_info.tile_m_idx, work_tile_info.tile_n_idx, 0)

                #
                # Prepare the mask for scaleA/scaleB
                #
                tApSFA2 = cute.make_rmem_tensor(
                    cute.make_layout(cute.filter_zeros(cute.slice_(tAsSFA2, (None, None, None, 0))).shape),
                    cutlass.Boolean,
                )
                tBpSFB2 = cute.make_rmem_tensor(
                    cute.make_layout(cute.filter_zeros(cute.slice_(tBsSFB2, (None, None, None, 0))).shape),
                    cutlass.Boolean,
                )

                # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] tApSFA2.layout: {tApSFA2.layout}")
                # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] tBpSFB2.layout: {tBpSFB2.layout}")

                # Peek (try_wait) SCALE buffer empty
                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_tile_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(scale_producer_state)

                #
                # load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt // k_tile_same_scale_factor, 1, unroll=1):
                    #
                    # Slice to per mma tile index
                    #
                    tAsSFA2_pipe = cute.filter_zeros(tAsSFA2[(None, None, None, scale_producer_state.index)])
                    tBsSFB2_pipe = cute.filter_zeros(tBsSFB2[(None, None, None, scale_producer_state.index)])

                    # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] tAgSFA2_mkl.layout: {tAgSFA2_mkl.layout}")
                    # print(f"[{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}] tBgSFB2_nkl.layout: {tBgSFB2_nkl.layout}")

                    tAgSFA2_k = cute.filter_zeros(
                        tAgSFA2_mkl[
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[0],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
                            )
                        ]
                    )
                    tBgSFB2_k = cute.filter_zeros(
                        tBgSFB2_nkl[
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[1],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
                            )
                        ]
                    )

                    tAcSFA2_compact = cute.filter_zeros(
                        cute.slice_(
                            tAcSFA2,
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[0],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
                            ),
                        )
                    )
                    tBcSFB2_compact = cute.filter_zeros(
                        cute.slice_(
                            tBcSFB2,
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[1],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
                            ),
                        )
                    )

                    # {$nv-internal-release begin}
                    # TODO: Skip more unnecessary load
                    # {$nv-internal-release end}
                    for i in cutlass.range_constexpr(cute.size(tApSFA2, mode=[1])):
                        tApSFA2[((0, 0), i, (0, 0))] = cute.elem_less(tAcSFA2_compact[(i)][0], mSFA2_mkl_current.shape[0])
                    for i in cutlass.range_constexpr(cute.size(tBpSFB2, mode=[1])):
                        tBpSFB2[((0, 0), i, (0, 0))] = cute.elem_less(tBcSFB2_compact[(i)][0], mSFB2_nkl_current.shape[0])

                    # Conditionally wait for Scale buffer empty
                    scale_pipeline.producer_acquire(scale_producer_state, peek_scale_empty_status)

                    # load scaleA/scaleB
                    cute.copy(tiled_copy_sfa2, tAgSFA2_k, tAsSFA2_pipe, pred=tApSFA2)
                    cute.copy(tiled_copy_sfb2, tBgSFB2_k, tBsSFB2_pipe, pred=tBpSFB2)

                    scale_pipeline.producer_commit(scale_producer_state)

                    # Peek (try_wait) Scale buffer empty
                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_tile_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(scale_producer_state)

                # Get next tile from scheduler
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
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

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Partition for S2T copy of SFA/SFB
            #
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

                # Peek (try_wait) AB buffer full for k_tile = 0
                peek_ab_full_status = cutlass.Boolean(1)
                if k_tile_cnt > 0 and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = 0
                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if acc_producer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                # sInfo: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                mma_tile_coord_mnl = (
                    tile_info[1] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[2],
                    cutlass.Int32(0),
                )

                # Get accumulator stage index
                acc_stage_index = acc_producer_state.index

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
                    offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):

                    # Set the correct accumulator buffer for each k_tile
                    tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                    if k_tile % k_tile_same_scale_factor == 0:
                        if is_leader_cta:
                            # Wait for accumulator buffer empty
                            acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)

                    # Reset the ACCUMULATE field for each tile
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile % k_tile_same_scale_factor > 0)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if k_tile < k_tile_cnt - 1:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
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

                    if k_tile % k_tile_same_scale_factor == k_tile_same_scale_factor - 1:
                        if is_leader_cta:
                            # Async arrive accumulator buffer full(each kblock)
                            acc_pipeline.producer_commit(acc_producer_state)

                        # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                        acc_producer_state.advance()
                        acc_stage_index = acc_producer_state.index
                        if acc_producer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

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
        # Specialized Accumulator Update Warp
        #
        # Bounds compare, not `in`: tuple membership makes the 4.5 wheel's AST
        # if-region flattening choke on captured Python objects.
        if warp_idx >= self.accumulator_update_warp_id[0] and warp_idx <= self.accumulator_update_warp_id[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)
            # Wait for TMEM allocation (done by epilogue warp)
            tmem.wait_for_alloc()

            # Retrieve TMEM pointer and create accumulator tensors
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # tCtAcc_base: Read partial accumulators from MMA (via acc_pipeline stages)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Final accumulated result is written to SMEM (sFinalAcc), not TMEM

            # Shape-only partition on global tensor for partitioning setup
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
            gD_mnl_shape = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
            tCgD_shape = thr_mma_epi.partition_C(gD_mnl_shape)

            # Setup copy operations and partition tensors
            acc_tidx = tidx
            (
                tiled_copy_t2r,
                tiled_copy_r2s_acc,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
                tRS_sFinalAcc,
                tTR_sSFA,
                tTR_sSFB,
            ) = self.acc_update_tmem_copy_and_partition(
                acc_tidx,
                tCtAcc_base,
                tCgD_shape,
                sSFA2_view_as_C,
                sSFB2_view_as_C,
                sFinalAcc,
                epi_tile,
            )

            # Initialize pipeline states
            # Consumer of acc_pipeline (receives partial accumulators from MMA)
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            # consumer of scale_pipeline (receives scale factors from scale load warp)
            scale_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_scale_stage)
            # Producer for epi_pipeline (sends final accumulator to epilogue)
            epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_epi_stage)

            # Initialize scheduler consumption
            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get first tile info from scheduler
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # Main tile processing loop
            while is_valid_tile:
                # Extract k_tile_cnt from scheduler
                k_tile_cnt = tile_info[3]

                # Initialize final accumulator to zero
                tTR_rAcc_final.fill(0.0)

                tTR_rSFA = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )
                tTR_rSFB = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )

                # Reset and peek acc_pipeline (MMA produces partial accumulators)
                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                # Peek (try_wait) Scale buffer full for k_tile = 0
                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_tile_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                # Loop over k_tiles to accumulate partial accumulators
                for k_tile_idx in cutlass.range(0, k_tile_cnt // k_tile_same_scale_factor, 1, unroll=1):

                    # Wait for scale buffer full
                    scale_pipeline.consumer_wait(scale_consumer_state, peek_scale_full_status)

                    tTR_sSFA_slice = cute.slice_(
                        tTR_sSFA,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )
                    tTR_sSFB_slice = cute.slice_(
                        tTR_sSFB,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )
                    scale_atom_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )

                    cute.copy(scale_atom_copy, tTR_sSFA_slice, tTR_rSFA)
                    cute.copy(scale_atom_copy, tTR_sSFB_slice, tTR_rSFB)

                    #
                    # Async arrive scale buffer empty
                    #
                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                    # Wait for MMA to produce partial accumulator for this k_tile
                    acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                    # Index into TMEM buffer for current pipeline stage
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                    # Group modes for subtile iteration
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    # Process each subtile (gate/up interleaving handled by layout)
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    scale_a = tTR_rSFA[(None, None, None, 0)].load()
                    scale_b = tTR_rSFB[(None, None, None, 0)].load()
                    scale = scale_a * scale_b
                    for subtile_idx in cutlass.range(subtile_cnt, unroll_full=True):
                        # Load partial accumulator from TMEM to register
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # Accumulate: final += partial (no scaling)
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                        acc_vec = tTR_rAcc.load()
                        final_vec = tTR_rAcc_subtile.load()
                        final_vec = acc_vec * scale + final_vec
                        tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                    # Release acc_pipeline buffer (MMA can reuse it)
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    # Peek next k_tile
                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < k_tile_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                # All k_tiles accumulated, now store final result to SMEM for epilogue
                # Acquire epi_pipeline (wait for epilogue to be done with the buffer)
                epi_pipeline.producer_acquire(epi_producer_state)

                # Copy final accumulator from registers to SMEM, one epi subtile per slot.
                # No proxy fence needed: producer and consumer both use the generic proxy,
                # and producer_commit's mbarrier arrive has release semantics.
                final_subtile_cnt = cute.size(tTR_rAcc_final.shape, mode=[3])
                slot_base = epi_producer_state.index * final_subtile_cnt
                for subtile_idx in cutlass.range(final_subtile_cnt, unroll_full=True):
                    tRS_rFinal = tiled_copy_r2s_acc.retile(tTR_rAcc_final[(None, None, None, subtile_idx)])
                    cute.copy(
                        tiled_copy_r2s_acc,
                        tRS_rFinal,
                        tRS_sFinalAcc[(None, None, None, slot_base + subtile_idx)],
                    )

                # Commit to epi_pipeline (signal epilogue that data is ready)
                epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

                # Get next tile from scheduler
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Signal epilogue that no more tiles
            epi_pipeline.producer_tail(epi_producer_state)

        #
        # Specialized epilogue warps
        #
        # Bounds compare, not `in` (see accumulator warps above). Both sides are
        # dynamic here (epilog_warp_id starts at 4, so neither const-folds like
        # the accumulator group's `>= 0`), so combine with bitwise `&`, not
        # Python `and`: `and` calls `__bool__` on a dynamic predicate, which the
        # 4.5 wheel's tracer rejects.
        if (warp_idx >= self.epilog_warp_id[0]) & (warp_idx <= self.epilog_warp_id[-1]):
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
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
            # Final accumulator now lives in SMEM (sFinalAcc); TMEM here is only used by
            # MMA partials + SF. tCtAcc_base is a shape/TV-layout carrier for the t2r
            # template below — it is never dereferenced as the load source in the epilogue.
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue (SMEM/TMEM/register - invariant across experts)
            #
            epi_tidx = tidx % 128
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
            if cutlass.const_expr(self.generate_sfd):
                norm_const = cutlass.Float32(norm_const_tensor[0])
                d_rcp_limits = get_dtype_rcp_limits(self.d_dtype)

            # S2R load of the final accumulator from SMEM (written by acc-update warp)
            tiled_copy_s2r_acc, tSR_sFinalAcc = self.epilog_smem_acc_load_and_partition(tiled_copy_t2r, epi_tidx, sFinalAcc)

            # bf16 relay of the dGLU output through the sFinalAcc SMEM bytes (NVFP4 path):
            # Pass 1 rounds d1/d2 to bf16 and stores them into slots 2*subtile + 0/1, which
            # alias exactly onto the f32 acc subtile's bytes (bf16 d1+d2 == one f32 subtile,
            # verified: bf16 8-slot layout == f32 4-slot layout == 65536 B).
            # Pass 2 reloads and upcasts to f32 to quantize, dropping pass 2's C reload + dGLU
            # recompute. A single universal copy atom is used for BOTH store and load so the
            # store/load TV layouts match (a stmatrix store would not round-trip against the load).
            if cutlass.const_expr(self.generate_sfd):
                # sFinalAccBf16 was built with the smem tensors above — hoisted
                # out of this warp-dispatch region for the 4.5 wheel.
                copy_atom_dglu = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
                tiled_copy_dglu = cute.make_tiled_copy_D(copy_atom_dglu, tiled_copy_t2r)
                tRS_sFinalAccBf16 = tiled_copy_dglu.get_slice(epi_tidx).partition_D(sFinalAccBf16)
                tTR_rDGLU1 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.BFloat16)
                tTR_rDGLU2 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.BFloat16)
                tRS_rDGLU1 = tiled_copy_dglu.retile(tTR_rDGLU1)
                tRS_rDGLU2 = tiled_copy_dglu.retile(tTR_rDGLU2)

            # Extension for per-expert domain conversion in epilogue
            epi_ext = self._make_extension(workspace_ptr)

            epi_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_epi_stage)

            # Load C pipeline
            c_pipeline_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_c_stage)
            # C is now produced by this epilogue warp group (warp epilog_warp_id[0]
            # issues the TMA); previously a dedicated warp owned this producer state.
            c_pipeline_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_c_stage)

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
            # DSMEM rowwise-sfd2 sequence: all cluster CTAs walk the same
            # cluster tile stream (the scheduler hands cluster tiles), so
            # parity (bit 0) and mbarrier phase (bit 1) stay in lockstep
            # across the whole cluster.
            if cutlass.const_expr(self.row_dsmem):
                row_dsmem_seq = cutlass.Int32(0)
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
                gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
                tCgD_loop = thr_mma_epi.partition_C(gD_mnl_loop)

                tgSFD2_up = None
                tgSFD2_gate = None
                if cutlass.const_expr(self.generate_sfd2):
                    # Per-expert SFD2_up setup
                    mSFD2_up_mnl, _ = epi_ext.get_gmem_tensor("sfd2", sfd2_up_tensor, padded_offsets, epi_work_tile_info)
                    thread_tiler = cute.make_layout((128, 1))
                    gSFD2_up_mnl = cute.local_tile(mSFD2_up_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgSFD2_up = thr_mma_epi.partition_C(gSFD2_up_mnl)
                    bSFD2_up = tCgSFD2_up[(None, None, None, *mma_tile_coord_mnl)]
                    tgSFD2_up = cute.local_partition(bSFD2_up, thread_tiler, tidx)

                    # Per-expert SFD2_gate setup
                    mSFD2_gate_mnl, _ = epi_ext.get_gmem_tensor("sfd2", sfd2_gate_tensor, padded_offsets, epi_work_tile_info)
                    gSFD2_gate_mnl = cute.local_tile(mSFD2_gate_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgSFD2_gate = thr_mma_epi.partition_C(gSFD2_gate_mnl)
                    bSFD2_gate = tCgSFD2_gate[(None, None, None, *mma_tile_coord_mnl)]
                    tgSFD2_gate = cute.local_partition(bSFD2_gate, thread_tiler, tidx)

                if cutlass.const_expr(not self.store_d_directly):
                    bSG_sD, bSG_gD_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD_loop, epi_tile, sD)
                    bSG_gD = bSG_gD_partitioned[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                    bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                epi_stage_index = 0

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, epi_stage_index)]

                if cutlass.const_expr(self.generate_sfd):
                    regPerSubtile = 4
                    sfd_row_tile = (
                        cute.make_layout(128),
                        cute.make_layout(32 * regPerSubtile),
                    )
                    # SFD Row: tile_atom_to_shape_SF layout, same path as SFA
                    real_sfd_row, _ = epi_ext.get_gmem_tensor("sfd", mSFDRow_mnl, padded_offsets, epi_work_tile_info)
                    gSFDRow_mnl = cute.local_tile(real_sfd_row, sfd_row_tile, (None, None, None))

                    # Don't ask why, AST is shit tracking the constexpr values to loop args.
                    tiled_copy_t2r_local, _, _ = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs)
                    thr_copy_t2r_local = tiled_copy_t2r_local.get_slice(tidx % 128)
                    tCgSFDRow_mnl = thr_copy_t2r_local.partition_D(gSFDRow_mnl)
                    tCgSFDRow_mnl = cute.filter_zeros(tCgSFDRow_mnl)
                    tCrSFDRow = cute.make_rmem_tensor(tCgSFDRow_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype)
                    tCrSFDRow_pvscale = cute.make_rmem_tensor_like(tCrSFDRow, cutlass.Float32)
                    tCgSFDRow_mn = tCgSFDRow_mnl[(None, None, None, None, None, 0)]

                    # SMEM staging for coalesced SFD stores. One store event
                    # (2 subtiles x d1/d2) covers one contiguous
                    # (128 rows, 32*regPerSubtile values) block of the SFD
                    # atom layout — 8 SF bytes per thread, but scattered at
                    # 16B thread stride in gmem. Mirror the gmem tile/thread
                    # partition onto an SMEM block with the identical atom
                    # layout: the scatter goes to SMEM, the gmem store then
                    # reads the block back linearly (one 8B word per thread).
                    sfd_stage_size = 128 * 32 * regPerSubtile // self.sf_vec_size
                    # sSFDRowStage_flat is pre-derived in the prologue: the 4.5
                    # wheel cannot carry `storage` across the warp-dispatch if.
                    sfd_stage_atom_layout = blockscaled_utils.tile_atom_to_shape_SF((128, 32 * regPerSubtile, 1), self.sf_vec_size)
                    sSFDRowStage_atom = cute.make_tensor(sSFDRowStage_flat.iterator, sfd_stage_atom_layout)
                    sSFDRowStage_tiled = cute.local_tile(sSFDRowStage_atom, sfd_row_tile, (None, None, None))
                    tCsSFDRow_mnl = thr_copy_t2r_local.partition_D(sSFDRowStage_tiled)
                    tCsSFDRow_mnl = cute.filter_zeros(tCsSFDRow_mnl)
                    tCsSFDRow = tCsSFDRow_mnl[(None, None, None, 0, 0, 0)]
                    if cutlass.const_expr(self.d_deinterleaved):
                        # Readback view: 2B pieces (the deinterleave scatter's
                        # granularity — gate/up alternate at 2 SF bytes).
                        sSFDRowStage_half = cute.recast_tensor(sSFDRowStage_flat, cutlass.Int16)
                    else:
                        # Linear readback view: 8B words, one per epilogue thread.
                        sSFDRowStage_words = cute.recast_tensor(sSFDRowStage_flat, cutlass.Int64)

                #
                # Get PROB (per-expert local M position)
                #
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mPosition = (
                    (epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)) * self.mma_tiler[0]
                    + mma_tile_coord_v * (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape))
                    + tidx % 128
                )
                mProb = real_prob[mPosition, 0, 0]
                if cutlass.const_expr(self.generate_dprob):
                    dProbVal = cutlass.Float32(0.0)

                #
                # Build C (up/gate forward activations) GMEM->SMEM partition and
                # PREFETCH the first C subtiles for this tile BEFORE waiting on the
                # accumulator, so C TMA movement overlaps accumulator production.
                # The epilogue warp group now owns the C producer (warp
                # epilog_warp_id[0] issues) as well as the consumer; a dedicated
                # C-load warp is no longer used.
                #
                real_c, _ = epi_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, epi_work_tile_info)
                gC_mnl_loop = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                tCgC_loop = thr_mma_epi.partition_C(gC_mnl_loop)
                bGS_sC, bGS_gC_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC_loop, epi_tile, sC)
                bGS_gC = bGS_gC_partitioned[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                bGS_gC = cute.group_modes(bGS_gC, 1, cute.rank(bGS_gC))
                c_subtile_cnt = cute.size(bGS_gC.shape, mode=[1])
                # Prefetch depth (in subtiles) is bounded by the C pipeline stages:
                # 2 stages -> 1 subtile ahead, 4 stages -> 2 subtiles ahead.
                c_prefetch_subtiles = self.num_c_stage // 2
                #
                # Wait for accumulator buffer full. The acc buffer is HELD across BOTH
                # epilogue passes and released only after pass 2, so pass 2 can re-read
                # the dy accumulator from resident sFinalAcc for free.
                #
                epi_pipeline.consumer_wait(epi_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # Initialize thread-local amax accumulator for this tile
                # Use 0.0 as initial value since we're computing absolute maximum
                thread_tile_amax_1 = cutlass.Float32(0.0)
                thread_tile_amax_2 = cutlass.Float32(0.0)

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                # Stage-aware base into the (double-buffered) sFinalAcc SMEM buffer.
                # Mirrors the producer's slot_base = epi_producer_state.index * subtile_cnt.
                # NOTE: this base applies ONLY to sFinalAcc SMEM reads; real_subtile_idx
                # below stays in [0, subtile_cnt) because it also indexes per-tile GMEM/
                # TMEM/SFD outputs which are not stage-multiplied.
                sfinal_slot_base = epi_consumer_state.index * subtile_cnt
                tTR_rAcc_0 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
                tTR_rAcc_1 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)

                #
                # PASS 1 (reduce): dGLU backward + amax / dprob / dbias reductions only.
                # The full per-thread tile amax must be finalized before row-wise SFD
                # generation, so NO SFD generation and NO D store happen in this pass.
                #
                if warp_idx == self.epilog_warp_id[0]:
                    for c_pf in range(min(c_prefetch_subtiles, c_subtile_cnt)):
                        c_pipeline.producer_acquire(c_pipeline_producer_state)
                        cute.copy(
                            tma_atom_c,
                            bGS_gC[(None, 2 * c_pf + 0)],
                            bGS_sC[(None, c_pipeline_producer_state.index)],
                            tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                        )
                        c_pipeline_producer_state.advance()
                        c_pipeline.producer_acquire(c_pipeline_producer_state)
                        cute.copy(
                            tma_atom_c,
                            bGS_gC[(None, 2 * c_pf + 1)],
                            bGS_sC[(None, c_pipeline_producer_state.index)],
                            tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                        )
                        c_pipeline_producer_state.advance()

                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    real_subtile_idx = subtile_idx
                    real_subtile_idx_next = subtile_idx + 1
                    #
                    # Load final accumulator subtile(s) from SMEM (sFinalAcc, written by
                    # the acc-update warp) into registers. sfinal_slot_base offsets into
                    # the current epi pipeline stage's slots.
                    #
                    if cutlass.const_expr(self.epilogue_prefetch_more):
                        # Double-buffered: on even subtiles, load current + next slot.
                        if subtile_idx % 2 == 0:
                            cute.copy(
                                tiled_copy_s2r_acc,
                                tSR_sFinalAcc[(None, None, None, sfinal_slot_base + real_subtile_idx)],
                                tiled_copy_s2r_acc.retile(tTR_rAcc_0),
                            )
                            cute.copy(
                                tiled_copy_s2r_acc,
                                tSR_sFinalAcc[(None, None, None, sfinal_slot_base + real_subtile_idx_next)],
                                tiled_copy_s2r_acc.retile(tTR_rAcc_1),
                            )
                            tTR_rAcc = tTR_rAcc_0
                        else:
                            tTR_rAcc = tTR_rAcc_1
                    else:
                        cute.copy(
                            tiled_copy_s2r_acc,
                            tSR_sFinalAcc[(None, None, None, sfinal_slot_base + real_subtile_idx)],
                            tiled_copy_s2r_acc.retile(tTR_rAcc),
                        )

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

                    # Refill the C pipeline: issue the TMA load for the subtile
                    # c_prefetch_subtiles ahead (warp epilog_warp_id[0] only), keeping
                    # the producer running ahead of this consumer within the tile.
                    if warp_idx == self.epilog_warp_id[0]:
                        c_pf_idx = real_subtile_idx + c_prefetch_subtiles
                        if c_pf_idx < c_subtile_cnt:
                            c_pipeline.producer_acquire(c_pipeline_producer_state)
                            cute.copy(
                                tma_atom_c,
                                bGS_gC[(None, 2 * c_pf_idx + 0)],
                                bGS_sC[(None, c_pipeline_producer_state.index)],
                                tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                            )
                            c_pipeline_producer_state.advance()
                            c_pipeline.producer_acquire(c_pipeline_producer_state)
                            cute.copy(
                                tma_atom_c,
                                bGS_gC[(None, 2 * c_pf_idx + 1)],
                                bGS_sC[(None, c_pipeline_producer_state.index)],
                                tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                            )
                            c_pipeline_producer_state.advance()

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
                        d1_vec, d2_vec, dprob_swiglu = self.dgeglu(acc_vec, ab1_vec_load, ab2_vec_load, mProb, linear_offset, dprob_swiglu)

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
                    # Generate subtile level amax
                    #
                    if cutlass.const_expr(self.generate_sfd2):
                        thread_tile_amax_1 = self.amax_reduction_per_thread(d1_vec, thread_tile_amax_1)
                        thread_tile_amax_2 = self.amax_reduction_per_thread(d2_vec, thread_tile_amax_2)

                    #
                    # Stash the bf16-rounded dGLU output into the accumulator's SMEM
                    # slots (reused in place, slots 2*subtile + 0/1) for pass-2
                    # quantization. Done AFTER the dprob/dbias/SFD2-amax reductions so
                    # those stay on the f32 d1/d2_vec; the SFD amax + e2m1 D bytes then
                    # derive from the bf16 value.
                    #
                    # The bf16 slots alias the f32 subtile's BYTES with a
                    # different (row, col)->byte map, so one thread's stash
                    # lands on OTHER threads' unread f32 rows (e.g. bf16 rows
                    # 64..95 over f32 rows 32..47). Every thread must finish
                    # its f32 subtile reads (incl. the prefetch path's
                    # next-slot read, issued at even subtiles above) before
                    # ANY thread overwrites the slot — without this barrier a
                    # fast warp corrupts a slow warp's accumulator input
                    # (racecheck: stash-vs-s2r-acc hazard; manifested as rare
                    # 16-token x 1-subtile corruption at sg512 large shapes).
                    #
                    if cutlass.const_expr(self.generate_sfd):
                        self.epilog_sync_barrier.arrive_and_wait()
                        tRS_rDGLU1.store(d1_vec.to(cutlass.BFloat16))
                        tRS_rDGLU2.store(d2_vec.to(cutlass.BFloat16))
                        cute.copy(
                            tiled_copy_dglu,
                            tRS_rDGLU1,
                            tRS_sFinalAccBf16[(None, None, None, 2 * (sfinal_slot_base + real_subtile_idx) + 0)],
                        )
                        cute.copy(
                            tiled_copy_dglu,
                            tRS_rDGLU2,
                            tRS_sFinalAccBf16[(None, None, None, 2 * (sfinal_slot_base + real_subtile_idx) + 1)],
                        )

                # Generate SFD2
                if cutlass.const_expr(self.generate_sfd2):
                    # The second-level scale normalizes dy for its two-level target
                    # encoding: divisor = max(D-quant data dtype) * max(SF dtype).
                    # When D is quantized (generate_sfd) the data dtype is the real
                    # d_dtype; otherwise SFD2 targets the NVFP4 e2m1 second level.
                    sfd2_data_dtype = self.d_dtype if cutlass.const_expr(self.generate_sfd) else cutlass.Float4E2M1FN
                    sfd2_norm = get_dtype_max(sfd2_data_dtype) * get_dtype_max(self.sf_dtype)
                    sfd2_descale_1 = thread_tile_amax_1 / sfd2_norm
                    sfd2_descale_2 = thread_tile_amax_2 / sfd2_norm
                    if cutlass.const_expr(self.row_dsmem):
                        # DSMEM rowwise reduction: the block's contributors are
                        # exactly this cluster's N-peers (same M rank). Each
                        # thread owns one row (thread_tiler (128,1)) and
                        # max-reds its gate/up descale into EVERY N-peer's
                        # parity slot ([cta_m gate | cta_m up]); the identical
                        # bit-pattern u32 max the gmem helper uses, so results
                        # stay byte-exact. Descales are non-negative, so the
                        # trick is valid.
                        _row_slot = (row_dsmem_seq & 1) * (2 * self.cta_tile_shape_mnk_d[0])
                        _rp_gate = sSfd2RowDsmem.iterator + (_row_slot + epi_tidx)
                        _rp_up = sSfd2RowDsmem.iterator + (_row_slot + self.cta_tile_shape_mnk_d[0] + epi_tidx)
                        _row_m_rank = cta_rank_in_cluster % self.cluster_shape_mn[0]
                        for _rt in cutlass.range_constexpr(self.cluster_shape_mn[1]):
                            _row_peer = _row_m_rank + _rt * self.cluster_shape_mn[0]
                            red_smem_cluster_max_f32(_rp_gate, sfd2_descale_1, Int32(_row_peer))
                            red_smem_cluster_max_f32(_rp_up, sfd2_descale_2, Int32(_row_peer))
                    else:  # GMEM reduction
                        self.second_level_scale_row_wise_gen(sfd2_descale_2, tgSFD2_up)
                        self.second_level_scale_row_wise_gen(sfd2_descale_1, tgSFD2_gate)

                # Make the pass-1 bf16 dGLU stores visible to the pass-2 reloads (the
                # store/load footprints coincide per thread; the barrier also covers the
                # async-proxy copy).
                if cutlass.const_expr(self.generate_sfd):
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(self.row_dsmem):
                        # DSMEM handoff (replaces the gmem counter + acquire
                        # spin): the epilog barrier above ordered every
                        # thread's reds; the elected thread's cluster release
                        # fence publishes them, then one arrive lands on each
                        # N-peer's parity mbarrier. A CTA passing its wait has
                        # proof every peer's reds landed — its local slot holds
                        # the final block descales.
                        _row_parity = row_dsmem_seq & 1
                        _row_phase = (row_dsmem_seq >> 1) & 1
                        if warp_idx == self.epilog_warp_id[0]:
                            with cute.arch.elect_one():
                                cute.arch.fence_acq_rel_cluster()
                                _row_am_rank = cta_rank_in_cluster % self.cluster_shape_mn[0]
                                for _ra in cutlass.range_constexpr(self.cluster_shape_mn[1]):
                                    mbarrier_arrive_cluster(
                                        row_dsmem_mbar + _row_parity,
                                        peer_cta_rank_in_cluster=Int32(_row_am_rank + _ra * self.cluster_shape_mn[0]),
                                    )
                        cute.arch.mbarrier_wait(row_dsmem_mbar + _row_parity, _row_phase)
                        # Acquire side of the handoff: order the peers' reds
                        # (published by their release fences) before this CTA's
                        # smem reads below.
                        cute.arch.fence_acq_rel_cluster()
                    elif cutlass.const_expr(self.sfd2_n_contrib > 1):
                        # Arrival protocol: one sfd2 block spans sfd2_n_contrib
                        # N work-tiles (sgn/mma_tiler_n: 2 at sgn=256, 4 at
                        # sgn=512 with the 128 tiler), so the block descale is
                        # only final in gmem once EVERY contributing tile's
                        # pass-1 atomic maxes have landed. Each tile announces
                        # completion on a per-(m tile, n-block) u32 counter;
                        # pass 2 spins until all contributors arrived, which
                        # makes the volatile read-back below provably final.
                        _n_tiles_f = cute.ceil_div(cute.shape(mD_mnl)[1], 2 * self.cta_tile_shape_mnk[1])
                        _n_j = cute.ceil_div(_n_tiles_f, self.sfd2_n_contrib)
                        _tok_off, _ = compute_expert_token_range(padded_offsets, epi_work_tile_info.expert_idx)
                        _j = epi_work_tile_info.tile_n_idx // self.sfd2_n_contrib
                        _counter_idx = (_tok_off // self.cta_tile_shape_mnk[0] + epi_work_tile_info.tile_m_idx) * _n_j + _j
                        _counter_ptr = cute.make_ptr(
                            cutlass.Int32,
                            self._get_sfd2_counter_ptr(workspace_ptr).toint() + _counter_idx * 4,
                            AddressSpace.gmem,
                            assumed_align=4,
                        )
                        # Announce this tile's arrival (one thread per CTA).
                        # The gpu-scope release fence orders every thread's
                        # pass-1 atomic maxes (already CTA-ordered by the
                        # barrier above) before the counter increment.
                        if warp_idx == self.epilog_warp_id[0]:
                            with cute.arch.elect_one():
                                cute.arch.fence_acq_rel_gpu()
                                atomic_add_i32_gmem(_counter_ptr.llvm_ptr, Int32(1))
                        # The last block of a row may have fewer contributors
                        # when the f-tile count is not a multiple of
                        # sfd2_n_contrib (spinning for the full count there
                        # would hang forever).
                        _contrib = Int32(self.sfd2_n_contrib)
                        _rem = _n_tiles_f - _j * self.sfd2_n_contrib
                        if _rem < _contrib:
                            _contrib = _rem
                        # Acquire-spin: once the counter shows all arrivals,
                        # every contributor's atomic maxes are visible.
                        _arrived = load_acquire_i32_gpu(_counter_ptr.llvm_ptr)
                        while _arrived < _contrib:
                            nanosleep(sleep_time=64)
                            _arrived = load_acquire_i32_gpu(_counter_ptr.llvm_ptr)

                    if cutlass.const_expr(self.generate_sfd2):
                        if cutlass.const_expr(self.row_dsmem):
                            # DSMEM path: the local parity slot holds the final
                            # block descales — plain LDS, no volatile gmem loads.
                            _rq_slot = (row_dsmem_seq & 1) * (2 * self.cta_tile_shape_mnk_d[0])
                            sfd2_descale_1 = sSfd2RowDsmem[_rq_slot + epi_tidx]
                            sfd2_descale_2 = sSfd2RowDsmem[_rq_slot + self.cta_tile_shape_mnk_d[0] + epi_tidx]
                            # Publish the finalized descales to the gmem sfd2
                            # outputs (the gmem atomics that used to produce them
                            # are skipped on this path): one CTA per contributor
                            # group — the N-rank-0 peer — plain-stores its rows.
                            if cta_rank_in_cluster // self.cluster_shape_mn[0] == 0:
                                tgSFD2_gate[0] = sfd2_descale_1
                                tgSFD2_up[0] = sfd2_descale_2
                            # Recycle the parity slot for tile seq+2: reads above
                            # are this thread's own (program order), and peers'
                            # next reds into this slot chain through the seq+1
                            # arrive whose fence publishes these zeroes.
                            sSfd2RowDsmem[_rq_slot + epi_tidx] = Float32(0.0)
                            sSfd2RowDsmem[_rq_slot + self.cta_tile_shape_mnk_d[0] + epi_tidx] = Float32(0.0)
                            row_dsmem_seq = row_dsmem_seq + 1
                        else:
                            # Re-read the block descale from the sfd2 gmem reduction
                            # for the pass-2 row-scale fold. atomic_max only returns
                            # the PRE-max value, and the register thread_tile_amax
                            # covers just this tile's columns: when one sfd2 block
                            # spans multiple N work-tiles (sgn > tile f-width,
                            # i.e. sfd2_n_contrib > 1), the block amax is
                            # completed in gmem by the other tiles' atomics — the
                            # arrival spin above guarantees they have all landed.
                            # Volatile: the atomics reduce in L2, so the load must
                            # not be served from a stale L1 line.
                            sfd2_descale_1 = load_float32_volatile(tgSFD2_gate.iterator.llvm_ptr)
                            sfd2_descale_2 = load_float32_volatile(tgSFD2_up.iterator.llvm_ptr)

                #
                # PASS 2 (quantize + store). NVFP4 path (generate_sfd): reload the
                # bf16 dGLU output stashed in pass 1 from sFinalAcc, upcast to f32, and
                # quantize — no C reload, no dGLU recompute. bf16-D path: re-drive C
                # from GMEM and recompute dGLU (the full-tile amax is now available).
                #
                # C is only re-driven for the bf16-D recompute path; the NVFP4 path
                # consumed C once in pass 1 and reloads dGLU from SMEM below.
                if cutlass.const_expr(not self.generate_sfd):
                    if warp_idx == self.epilog_warp_id[0]:
                        for c_pf in range(min(c_prefetch_subtiles, c_subtile_cnt)):
                            c_pipeline.producer_acquire(c_pipeline_producer_state)
                            cute.copy(
                                tma_atom_c,
                                bGS_gC[(None, 2 * c_pf + 0)],
                                bGS_sC[(None, c_pipeline_producer_state.index)],
                                tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                            )
                            c_pipeline_producer_state.advance()
                            c_pipeline.producer_acquire(c_pipeline_producer_state)
                            cute.copy(
                                tma_atom_c,
                                bGS_gC[(None, 2 * c_pf + 1)],
                                bGS_sC[(None, c_pipeline_producer_state.index)],
                                tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                            )
                            c_pipeline_producer_state.advance()

                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    real_subtile_idx = subtile_idx
                    real_subtile_idx_next = subtile_idx + 1
                    if cutlass.const_expr(self.generate_sfd):
                        #
                        # NVFP4 path: reload the bf16 dGLU output stashed by pass 1 from
                        # the accumulator's SMEM slots (2*subtile + 0/1) and upcast to
                        # f32. No C reload, no dGLU recompute — pass 1 did both.
                        #
                        cute.copy(
                            tiled_copy_dglu,
                            tRS_sFinalAccBf16[(None, None, None, 2 * (sfinal_slot_base + real_subtile_idx) + 0)],
                            tRS_rDGLU1,
                        )
                        cute.copy(
                            tiled_copy_dglu,
                            tRS_sFinalAccBf16[(None, None, None, 2 * (sfinal_slot_base + real_subtile_idx) + 1)],
                            tRS_rDGLU2,
                        )
                        d1_vec = tRS_rDGLU1.load().to(cutlass.Float32)
                        d2_vec = tRS_rDGLU2.load().to(cutlass.Float32)
                    else:
                        #
                        # Load final accumulator subtile(s) from SMEM (sFinalAcc, written by
                        # the acc-update warp) into registers. sfinal_slot_base offsets into
                        # the current epi pipeline stage's slots.
                        #
                        if cutlass.const_expr(self.epilogue_prefetch_more):
                            # Double-buffered: on even subtiles, load current + next slot.
                            if subtile_idx % 2 == 0:
                                cute.copy(
                                    tiled_copy_s2r_acc,
                                    tSR_sFinalAcc[(None, None, None, sfinal_slot_base + real_subtile_idx)],
                                    tiled_copy_s2r_acc.retile(tTR_rAcc_0),
                                )
                                cute.copy(
                                    tiled_copy_s2r_acc,
                                    tSR_sFinalAcc[(None, None, None, sfinal_slot_base + real_subtile_idx_next)],
                                    tiled_copy_s2r_acc.retile(tTR_rAcc_1),
                                )
                                tTR_rAcc = tTR_rAcc_0
                            else:
                                tTR_rAcc = tTR_rAcc_1
                        else:
                            cute.copy(
                                tiled_copy_s2r_acc,
                                tSR_sFinalAcc[(None, None, None, sfinal_slot_base + real_subtile_idx)],
                                tiled_copy_s2r_acc.retile(tTR_rAcc),
                            )
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

                        # Refill the C pipeline: issue the TMA load for the subtile
                        # c_prefetch_subtiles ahead (warp epilog_warp_id[0] only), keeping
                        # the producer running ahead of this consumer within the tile.
                        if warp_idx == self.epilog_warp_id[0]:
                            c_pf_idx = real_subtile_idx + c_prefetch_subtiles
                            if c_pf_idx < c_subtile_cnt:
                                c_pipeline.producer_acquire(c_pipeline_producer_state)
                                cute.copy(
                                    tma_atom_c,
                                    bGS_gC[(None, 2 * c_pf_idx + 0)],
                                    bGS_sC[(None, c_pipeline_producer_state.index)],
                                    tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                                )
                                c_pipeline_producer_state.advance()
                                c_pipeline.producer_acquire(c_pipeline_producer_state)
                                cute.copy(
                                    tma_atom_c,
                                    bGS_gC[(None, 2 * c_pf_idx + 1)],
                                    bGS_sC[(None, c_pipeline_producer_state.index)],
                                    tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                                )
                                c_pipeline_producer_state.advance()

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
                            d1_vec, d2_vec, dprob_swiglu = self.dgeglu(acc_vec, ab1_vec_load, ab2_vec_load, mProb, linear_offset, dprob_swiglu)

                    #
                    # Generate SFD (row-wise). thread_tile_amax_1/2 are now the full
                    # per-tile amax (finalized in pass 1) and are available here.
                    #
                    if cutlass.const_expr(self.generate_sfd):
                        #
                        # Generate row major SFD
                        #
                        self.quant_sfd_row(
                            (real_subtile_idx * 2 + 0) % 4,
                            tiled_copy_r2s,
                            d1_vec,
                            tCrSFDRow_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD1,
                            sfd2_descale_1,
                        )
                        self.quant_sfd_row(
                            (real_subtile_idx * 2 + 1) % 4,
                            tiled_copy_r2s,
                            d2_vec,
                            tCrSFDRow_pvscale,
                            norm_const,
                            d_rcp_limits,
                            tRS_rD2,
                            sfd2_descale_2,
                        )
                        if subtile_idx % 2 == 1:
                            local_m_tile = epi_work_tile_info.tile_m_idx
                            local_n_tile = epi_work_tile_info.tile_n_idx
                            sfd_row_idx_mn = (
                                local_m_tile * self.epi_tile_cnt[0] + 0,
                                local_n_tile * self.epi_tile_cnt[1] // 2 + (real_subtile_idx // 2),
                            )

                            tCgSFDRow = tCgSFDRow_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfd_row_idx_mn,
                                )
                            ]
                            if cutlass.const_expr(not self.use_fp8_ptx_cvt):
                                tCrSFDRow.store(tCrSFDRow_pvscale.load().to(self.sf_dtype))
                            else:
                                self.cvt_f32x4_to_f8x4(tCrSFDRow_pvscale, tCrSFDRow)
                            if sfd_row_idx_mn[1] * 32 * regPerSubtile < cute.size(cute.shape(mSFDRow_mnl.layout, mode=[1])):
                                # Coalesced SFD store: scatter this thread's SF
                                # bytes into the SMEM stage at their atom-layout
                                # offsets, then write the contiguous block back
                                # linearly — one 8B word per thread instead of
                                # 2x4B gmem stores at 16B thread stride (which
                                # waste 3/4 of every 32B sector). The
                                # end-of-subtile epilog barrier orders this
                                # readback before the next event's scatter.
                                cute.autovec_copy(tCrSFDRow, tCsSFDRow)
                                self.epilog_sync_barrier.arrive_and_wait()
                                gSFDRow_evt = gSFDRow_mnl[(None, None, *sfd_row_idx_mn, 0)]
                                if cutlass.const_expr(self.d_deinterleaved):
                                    # DEINTERLEAVED scatter. Canonical stage byte
                                    # a = r + 2p + 4*m4 + 16*m32 + 512*q0 (p =
                                    # band_parity, q0 = band_pair, q1 =
                                    # sfd_col_tile_idx) must land at the deint atom
                                    # position r + 2*q0 + 4*m4 + 16*m32 + 512*q1 +
                                    # 8f*p. Relative to the canonical tile base
                                    # (rowblock + 1024*q1) the target offset is
                                    #   t = a + p*(8f-2) - 510*q0 - 512*q1.
                                    # Only p/q0/q1 move; r keeps 2B pieces
                                    # contiguous, so each thread stores 4x Int16.
                                    sfd_tile_base_addr = gSFDRow_evt.iterator.toint()
                                    sfd_col_tile_idx = sfd_row_idx_mn[1]
                                    if epi_tidx * 8 < sfd_stage_size:
                                        for piece_idx in cutlass.range_constexpr(4):
                                            stage_byte = epi_tidx * 8 + 2 * piece_idx
                                            band_parity = (stage_byte // 2) % 2
                                            band_pair = stage_byte // 512
                                            deint_offset = stage_byte + band_parity * (4 * self.deint_n - 2) - band_pair * 510 - sfd_col_tile_idx * 512
                                            gmem_piece = cute.make_tensor(
                                                cute.make_ptr(
                                                    cutlass.Int16,
                                                    sfd_tile_base_addr + deint_offset,
                                                    AddressSpace.gmem,
                                                    assumed_align=2,
                                                ),
                                                cute.make_layout(1),
                                            )
                                            gmem_piece[0] = sSFDRowStage_half[stage_byte // 2]
                                else:
                                    # INTERLEAVED: the stage IS the gmem tile —
                                    # linear copy, one 8B word per thread.
                                    gSFDRow_words = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Int64,
                                            gSFDRow_evt.iterator.toint(),
                                            AddressSpace.gmem,
                                            assumed_align=8,
                                        ),
                                        cute.make_layout(sfd_stage_size // 8),
                                    )
                                    if epi_tidx * 8 < sfd_stage_size:
                                        gSFDRow_words[epi_tidx] = sSFDRowStage_words[epi_tidx]
                    else:
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
                        d_idx_mn = (mma_tile_coord_mnl[0], mma_tile_coord_mnl[1])
                        d_epilogue_subtile = (
                            cute.make_layout(128),
                            cute.make_layout(self.mma_tiler[1] * 2),
                        )
                        gD_sub_loop = cute.local_tile(real_d, d_epilogue_subtile, (None, None, None))
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
                        self.store_global_memory_256b(tCgD1, tRS_rD1)
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
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store D to global memory
                        #
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
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            d_pipeline.producer_commit()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                epi_pipeline.consumer_release(epi_consumer_state)
                epi_consumer_state.advance()

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
            # Wait for outstanding C TMA loads (producer side) to drain
            #
            if warp_idx == self.epilog_warp_id[0]:
                c_pipeline.producer_tail(c_pipeline_producer_state)

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

    def acc_update_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tCtAcc_base: cute.Tensor,
        gD_mnl: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sFinalAcc: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> Tuple[cute.TiledCopy, cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Create tiled copy operations and partition tensors for accumulator update warp.

        This function sets up T2R (TMEM-to-Register) and R2S (Register-to-SMEM) copies
        for the accumulator update warp to:
        1. Load partial accumulators from TMEM (tCtAcc_base) to registers
        2. Store final accumulated result from registers to SMEM (sFinalAcc)

        The GLU gate/up interleaving is already handled by the TMEM layout, so we don't
        need to explicitly separate them here.

        :param tidx: Thread index within accumulator update warp group
        :param tCtAcc_base: TMEM tensor for reading partial accumulators (from MMA)
        :param gD_mnl: Global tensor D shape (for partitioning)
        :param sFinalAcc: SMEM tensor for writing final accumulator (for epilogue)
        :param epi_tile: Epilogue tile size
        :return: Tuple of (tiled_copy_t2r, tiled_copy_r2s_acc, tTR_tAcc_base, tTR_rAcc,
                          tTR_rAcc_final, tRS_sFinalAcc, tTR_sSFA, tTR_sSFB)
        """
        # Create TMEM load atom based on MMA tile size
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )

        # Partition accumulator tensors by epilogue tile
        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)

        # Create tiled copy for T2R (TMEM to Register)
        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)])

        # Get thread-specific slice
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        # R2S (Register to SMEM) store of the final accumulator: reuse the T2R copy's
        # thread-value layout with a universal copy atom (swizzle-safe for f32)
        copy_atom_r2s_acc = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        tiled_copy_r2s_acc = cute.make_tiled_copy_D(copy_atom_r2s_acc, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, SUBTILE * STAGE)
        tRS_sFinalAcc = tiled_copy_r2s_acc.get_slice(tidx).partition_D(sFinalAcc)

        # Partition source for T2R: tCtAcc_base (read partial accumulators)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

        # Partition gD_mnl for determining register buffer shape
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        sSFA_epi = cute.flat_divide(sSFA, epi_tile)
        sSFB_epi = cute.flat_divide(sSFB, epi_tile)

        tTR_gC = thr_copy_t2r.partition_D(gD_mnl_epi)
        tTR_sSFA = thr_copy_t2r.partition_D(sSFA_epi)
        tTR_sSFB = thr_copy_t2r.partition_D(sSFB_epi)

        # Create register tensor for T2R destination (holds one k_tile's partial accumulator)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)

        # Create register tensor for final accumulated result across all k_tiles
        # Shape: (T2R, T2R_M, T2R_N, EPI_M, EPI_N) -> grouped to (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
        tTR_rAcc_final_ = cute.make_rmem_tensor(tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype)
        tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_))

        return (
            tiled_copy_t2r,
            tiled_copy_r2s_acc,
            tTR_tAcc_base,
            tTR_rAcc,
            tTR_rAcc_final,
            tRS_sFinalAcc,
            tTR_sSFA,
            tTR_sSFB,
        )

    def epilog_smem_acc_load_and_partition(self, tiled_copy_t2r, tidx, sFinalAcc):
        """S2R load of the final accumulator from SMEM (sFinalAcc). Reuses the T2R copy's
        thread-value layout with a universal f32 copy atom. Returns the tiled copy and the
        partitioned SMEM source; register targets are retiled at the load site."""
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        # (S2R, S2R_M, S2R_N, SUBTILE * STAGE)
        tSR_sFinalAcc = tiled_copy_s2r.get_slice(tidx).partition_D(sFinalAcc)
        return tiled_copy_s2r, tSR_sFinalAcc

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
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
        store_d_directly: bool,
        generate_dbias: bool = False,
        acc_dtype: Type[cutlass.Numeric] = None,
        sf2_dtype: Type[cutlass.Numeric] = None,
        sgm: int = 1,
        sgn: int = 256,
        sgk: int = 256,
        num_epi_stage: int = 1,
        row_dsmem: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/D operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param d_layout: Layout of operand D.
        :type d_layout: utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, D stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 3

        # Default C/D stages
        num_c_stage = 4 if a_dtype.width == 8 else (4 if store_d_directly else 2)
        num_d_stage = 2 if a_dtype.width == 8 else (0 if store_d_directly else 2)

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and D
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

        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
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

        # Compute second level scale factor layout
        if sgm < mma_tiler_mnk[0]:
            size_m = sgm
        else:
            size_m = mma_tiler_mnk[0]

        if sgn < mma_tiler_mnk[1]:
            size_n = sgn
        else:
            size_n = mma_tiler_mnk[1]

        if sgk < mma_tiler_mnk[2]:
            size_k = sgk
        else:
            size_k = mma_tiler_mnk[2]

        scale_m_per_tile = mma_tiler_mnk[0] // size_m
        scale_n_per_tile = mma_tiler_mnk[1] // size_n
        scale_k_per_tile = mma_tiler_mnk[2] // size_k

        sfa2_smem_layout_staged_one = cute.make_layout(
            (
                (size_m, scale_m_per_tile),
                (size_k, scale_k_per_tile),
                1,
            ),
            stride=(
                (0, scale_k_per_tile),
                (0, 1),
                scale_k_per_tile * scale_m_per_tile,
            ),
        )
        sfb2_smem_layout_staged_one = cute.make_layout(
            (
                (size_n, scale_n_per_tile),
                (size_k, scale_k_per_tile),
                1,
            ),
            stride=(
                (0, scale_k_per_tile),
                (0, 1),
                scale_k_per_tile * scale_n_per_tile,
            ),
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
            + cute.size_in_bytes(sf2_dtype, sfa2_smem_layout_staged_one)
            + cute.size_in_bytes(sf2_dtype, sfb2_smem_layout_staged_one)
        )

        # Mbar bytes
        mbar_helpers_bytes = 1024
        # Sinfo bytes
        sinfo_bytes = 4 * 4 * num_tile_stage
        # C/D bytes
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        d_bytes = d_bytes_per_stage * num_d_stage
        if d_dtype == cutlass.Float8E5M2 or d_dtype == cutlass.Float8E4M3FN:
            d_bytes = d_bytes * 2
        # AMAX bytes
        amax_bytes = get_amax_smem_size() if d_dtype == cutlass.BFloat16 else 0
        # dBias transpose buffer: (128, 64) column-major FP32 = 32 KB
        dbias_bytes = 128 * 64 * cute.size_in_bytes(cutlass.Float32, cute.make_layout((1,))) if generate_dbias else 0
        # Full-CTA-tile final accumulator buffer in SMEM (replaces the TMEM final region).
        # The stage mode holds subtile_cnt full-tile slots per epi stage, so the
        # buffer scales with num_epi_stage (double-buffered when num_epi_stage == 2).
        cta_m = mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape)
        cta_n = mma_tiler_mnk[1]
        subtile_cnt = (cta_m // cute.size(epi_tile[0])) * (cta_n // cute.size(epi_tile[1]))
        final_acc_smem_layout_one = sm100_utils.make_smem_layout_epi(
            acc_dtype,
            d_layout,
            epi_tile,
            subtile_cnt * num_epi_stage,
        )
        final_acc_bytes = cute.size_in_bytes(acc_dtype, final_acc_smem_layout_one)

        # Epilogue bytes
        # DSMEM rowwise-sfd2 reduction buffer: 2 parity slots x (cta_m gate +
        # cta_m up) f32 row descales, counted BEFORE the greedy num_ab_stage
        # computation below so it never overflows smem.
        cta_m_rows = mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape)
        row_dsmem_bytes = 2 * (2 * cta_m_rows) * 4 if row_dsmem else 0

        epi_bytes = c_bytes + d_bytes + amax_bytes + dbias_bytes + final_acc_bytes + row_dsmem_bytes

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes (mbar, epi, sinfo, sfa2, sfb2)
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage
        assert num_ab_stage >= 1, (
            f"SMEM overflow: full-tile sFinalAcc ({final_acc_bytes} B) leaves no room for " f"AB stages (mma_tiler={mma_tiler_mnk}); reduce mma_tiler_mn"
        )

        total_bytes = occupancy * (ab_bytes_per_stage * num_ab_stage + epi_bytes + sinfo_bytes + mbar_helpers_bytes)
        return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage
