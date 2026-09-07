# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Tuple, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.pipeline
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.typing import Int32, Float32, Boolean
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from . import utils
from .mask import AttentionMask
from .seqlen_info import SeqlenInfo
from .block_info import BlockInfo
from . import blackwell_helpers as sm100_utils
from . import mma_sm100_desc as sm100_desc
from .fast_math import FastSilU
from .tile_scheduler import (
    ClcDescriptorState,
    ClcState,
    ParamsBase,
    SchedulingMode,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from .named_barrier import EPILOGUE_BARRIER_BASE, TMEM_POINTER_BARRIER, TMEM_RELEASE_BARRIER
from .block_sparsity import (
    HSTUBlockSparseTensors,
    get_q2k_block_for_reverse_slot,
    get_q2k_block_sparse_consumer_row,
)


def _atomic_add_bf16x2(ptr, val_lo: Float32, val_hi: Float32, *, loc=None, ip=None):
    """Packed BF16x2 atomic add used by the split-KV epilogue."""
    llvm.inline_asm(
        None,
        [ptr, val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 packed; cvt.rn.bf16x2.f32 packed, $1, $2; red.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _tmem_load_32dp32b32x(tmem_addr: Int32):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32),
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()],
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, "
        "$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, [$32];",
        ",".join(["=r"] * 32 + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i])) for i in range(32))


@cute.jit
def _tmem_load_16dp64b16x(tmem_addr: Int32):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 16),
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()],
        "tcgen05.ld.sync.aligned.16x64b.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16];",
        ",".join(["=r"] * 16 + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i])) for i in range(16))


@cute.jit
def _cvt_f32x2_to_16x2(a: Float32, b: Float32, dtype) -> Int32:
    if const_expr(dtype == cutlass.Float16):
        instruction = "cvt.rn.f16x2.f32 $0, $1, $2;"
    elif const_expr(dtype == cutlass.BFloat16):
        instruction = "cvt.rn.satfinite.bf16x2.f32 $0, $1, $2;"
    else:
        raise TypeError(f"Expected Float16 or BFloat16, got {dtype}")
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(b).ir_value(), Float32(a).ir_value()],
            instruction,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _tmem_store_16x16(tmem_addr: Int32, vals: cute.Tensor):
    assert cute.size(vals) == 16
    llvm.inline_asm(
        None,
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()] + [Int32(vals[i]).ir_value() for i in range(16)],
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16};",
        ",".join(["r"] * 17),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _tmem_store_16x8_16dp64b(tmem_addr: Int32, vals: cute.Tensor):
    assert cute.size(vals) == 8
    llvm.inline_asm(
        None,
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()] + [Int32(vals[i]).ir_value() for i in range(8)],
        "tcgen05.st.sync.aligned.16x64b.x8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        ",".join(["r"] * 9),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _silu_f32_inplace(scores: cute.Tensor, score_scale_half: Float32):
    for i in cutlass.range_constexpr(0, cute.size(scores), 2):
        v0, v1 = utils.mul_packed_f32x2(
            (scores[i], scores[i + 1]),
            (score_scale_half, score_scale_half),
        )
        tanh_v0 = utils.tanhf(v0)
        tanh_v1 = utils.tanhf(v1)
        scores[i], scores[i + 1] = utils.fma_packed_f32x2(
            (v0, v1),
            (tanh_v0, tanh_v1),
            (v0, v1),
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
        use_clc_scheduler: bool = True,
        use_clc_descriptor: bool = True,
        use_tma_O: bool = True,
        use_causal_mask_r2p: bool = True,
        use_2cta_instrs: bool = False,
        is_q_len_one: bool = False,
        q1_split_kv: int = 1,
        q1_single_warp_epilogue: bool = False,
        q1_m64_silu_warps: int = 0,
        q1_m64_inplace_silu: bool = False,
        q1_m64_16dp_silu: bool = False,
        q1_m64_tail_branch: bool = False,
        q1_m64_kv_stage: int = 0,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.kBlockM = kBlockM
        self.kBlockN = kBlockN
        self.q1_split_kv = q1_split_kv
        self.q1_single_warp_epilogue = q1_single_warp_epilogue
        self.q1_m64_silu_warps = q1_m64_silu_warps
        self.q1_m64_inplace_silu = q1_m64_inplace_silu
        self.q1_m64_16dp_silu = q1_m64_16dp_silu
        self.q1_m64_tail_branch = q1_m64_tail_branch
        self.q1_m64_kv_stage = q1_m64_kv_stage
        assert q1_split_kv in (1, 2, 4)
        assert q1_split_kv == 1 or is_q_len_one
        assert not q1_single_warp_epilogue or is_q_len_one
        assert q1_m64_silu_warps in (0, 1, 2, 3)
        assert q1_m64_silu_warps == 0 or (is_q_len_one and kBlockM == 64 and kBlockN == 128)
        assert not (q1_m64_inplace_silu or q1_m64_16dp_silu) or (is_q_len_one and kBlockM == 64 and kBlockN == 128)
        assert not q1_m64_16dp_silu or q1_m64_inplace_silu
        assert not q1_m64_tail_branch or q1_m64_16dp_silu
        assert q1_m64_kv_stage in (0, 5)
        assert q1_m64_kv_stage == 0 or (is_q_len_one and kBlockM == 64 and kBlockN == 128)
        self.use_2cta_instrs = (
            use_2cta_instrs
            and q1_split_kv == 1
            and head_dim == 128
            and head_dim_v == 128
            and is_causal
            and not is_local
            and not is_arbitrary
            and not is_paged
            and qhead_per_kvhead == 1
            and is_persistent
            and use_clc_scheduler
            and not use_auto_block_metadata
            and kBlockM == 128
            and kBlockN == 128
        )
        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # A qlen=1 tile has no second Q block to overlap. Keeping one Q stage
        # avoids issuing a fully masked QK/PV tile and cuts its Q/O storage in
        # half. D256 and 2-CTA MMA have the same one-stage requirement.
        self.q_stage = 1 if is_q_len_one or self.use_2cta_instrs or self.head_dim_padded >= 256 else 2
        self.s_stage = 2  # score stage for intra-warp overlap
        assert self.q_stage in [1, 2]
        assert self.s_stage in [2]

        self.use_auto_block_metadata = use_auto_block_metadata
        self.enable_offset_dynamic = self.q_stage * self.cta_group_size == 2 and not self.use_auto_block_metadata
        # The logical tile spans all Q stages in the CTA group.
        self.logical_cta_tiler = (
            self.q_stage * self.cta_group_size * kBlockM,
            kBlockN,
            self.head_dim_padded,
        )
        self.mma_tiler_qk = (
            self.cta_group_size * kBlockM,
            kBlockN,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.cta_group_size * kBlockM,
            self.head_dim_v_padded,
            kBlockN,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (self.cta_group_size, 1)
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
        self.use_clc_scheduler = use_clc_scheduler and self.is_persistent and not self.is_paged
        assert not self.use_2cta_instrs or self.use_clc_scheduler
        # A persistent q_stage=1 kernel reuses a single TMEM output accumulator.
        # Protect that reuse with the same UMMA-to-async accumulator pipeline
        # used by the 2-CTA path: the MMA warp must not overwrite O until the
        # epilogue warps have finished loading it.
        self.use_o_pipeline = self.use_2cta_instrs or (self.q_stage == 1 and self.use_clc_scheduler)
        self.scheduling_mode = SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        self.sched_stages = 1
        self.descriptor_stages = 2
        self.use_clc_descriptor = use_clc_descriptor and self.use_clc_scheduler and not self.use_2cta_instrs
        self.use_precomputed_qk_descriptors = self.q_stage == 2 and not self.use_2cta_instrs
        self.use_tma_O = use_tma_O and self.use_clc_scheduler and q1_split_kv == 1
        self.use_causal_mask_r2p = use_causal_mask_r2p and self.is_causal and not self.is_local
        self.clc_scheduler_warp_id = self.empty_warp_ids[0] if self.use_clc_scheduler else None

        # Register budget tuning: SiLU warps get the most registers to reduce spills.
        if self.use_2cta_instrs:
            self.num_regs_silu = 192
            self.num_regs_other = 120
            self.num_regs_empty = 120
        elif self.use_clc_scheduler:
            self.num_regs_silu = 200
            self.num_regs_other = 104
            self.num_regs_empty = 104
        elif self.head_dim_padded >= 256 and self.use_auto_block_metadata:
            self.num_regs_silu = 224
            self.num_regs_other = 48
            self.num_regs_empty = 40
        elif self.head_dim_padded >= 128:
            self.num_regs_silu = 232
            self.num_regs_other = 40
            self.num_regs_empty = 40
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
        self.o_stage = 2 if self.use_2cta_instrs else self.q_stage
        self.tmem_o_offset = [self.tmem_s_offset[-1] + self.kBlockN + i * self.head_dim_v_padded for i in range(self.o_stage)]
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        self.tmem_s_to_p_offset = self.kBlockN // 2
        self.tmem_p_offset = [self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(self.s_stage)]
        # split_P_arrive: signal MMA to start PV when this many P columns are written.
        self.split_P_arrive = self.kBlockN // 2
        self.split_P_arrive = self.split_P_arrive // 32 * 32

        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for SiLU, MMA, and epilogue operations
        """

        # Size the KV pipeline to fit the 224 KiB shared-memory budget.
        smem_size_q = self.q_stage * self.kBlockM * self.head_dim_padded * self.q_dtype.width // 8
        smem_size_o = self.q_stage * self.kBlockM * self.head_dim_v_padded * self.q_dtype.width // 8
        smem_size_q_o = max(smem_size_q, smem_size_o) if self.overlap_sO_sQ else (smem_size_q + smem_size_o)
        smem_size_kv_per_stage = (
            max(
                self.kBlockN * self.head_dim_padded * self.q_dtype.width // 8,
                self.kBlockN * self.head_dim_v_padded * self.q_dtype.width // 8,
            )
            // self.cta_group_size
        )
        if self.use_2cta_instrs:
            self.kv_stage = 7
        elif self.q_dtype.width == 8:
            self.kv_stage = 4
        elif self.q1_m64_kv_stage:
            self.kv_stage = self.q1_m64_kv_stage
        else:
            self.kv_stage = min((224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage, 3)
        assert self.kv_stage >= 2
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

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
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

        self.epi_tile = (self.kBlockM, self.head_dim_v_padded)

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
        if const_expr(self.use_tma_O):
            token_stride = mO.stride[0]
            mO_tma = cute.make_tensor(
                mO.iterator - max_seqlen_q * token_stride,
                cute.make_layout(
                    (
                        max_seqlen_q,
                        mO.shape[1],
                        mO.shape[2],
                        mO.shape[0] + 1,
                    ),
                    stride=(
                        token_stride,
                        mO.stride[1],
                        mO.stride[2],
                        token_stride,
                    ),
                ),
            )
            tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mO_tma,
                cute.select(sO_layout, mode=[0, 1]),
                self.epi_tile,
            )
        else:
            tma_atom_O, tma_tensor_O = None, None
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
        self.tma_copy_q_bytes *= self.cta_group_size
        self.tma_copy_k_bytes *= self.cta_group_size
        self.tma_copy_v_bytes *= self.cta_group_size

        TileScheduler = SingleTileVarlenScheduler
        num_block = (
            cute.ceil_div(max_seqlen_q, self.logical_cta_tiler[0])
            if const_expr(self.use_clc_scheduler)
            else cute.ceil_div(cute.size(mQ.shape[0]), self.logical_cta_tiler[0])
        )
        seqlen_k = max_seqlen_k if const_expr(self.use_clc_scheduler) else cute.size(mK.shape[0])
        tile_sched_args = TileSchedulerArguments(
            num_block,
            cute.size(mQ.shape[2]) * self.q1_split_kv,
            cute.size(cu_seqlens_q.shape[0] - 1),
            seqlen_k,
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.logical_cta_tiler[:2],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args,
            scheduling_mode=self.scheduling_mode,
            use_clc_descriptor=self.use_clc_descriptor,
        )
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
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0
        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        descriptor_mbar_size = self.descriptor_stages * 2 if self.use_clc_descriptor else 0
        descriptor_buffer_size = self.descriptor_stages * 8 if self.use_clc_descriptor else 0
        pipeline_q_size = self.q_stage * 2 if self.use_2cta_instrs else 0
        pipeline_kv_size = self.kv_stage * 2 if self.use_2cta_instrs else 0
        pipeline_s_p_size = self.s_stage * 2 if self.use_2cta_instrs else 0
        pipeline_p_full_size = self.s_stage * 2 if self.use_2cta_instrs else 0
        pipeline_o_size = self.o_stage * 2 if self.use_o_pipeline else 0

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[Int32, clc_response_size],
                16,
            ]
            descriptor_mbar_ptr: cute.struct.MemRange[cutlass.Int64, descriptor_mbar_size]
            work_descriptor: cute.struct.Align[
                cute.struct.MemRange[Int32, descriptor_buffer_size],
                16,
            ]
            pipeline_q_mbar: cute.struct.MemRange[cutlass.Int64, pipeline_q_size]
            pipeline_kv_mbar: cute.struct.MemRange[cutlass.Int64, pipeline_kv_size]
            pipeline_s_p_mbar: cute.struct.MemRange[cutlass.Int64, pipeline_s_p_size]
            pipeline_p_full_mbar: cute.struct.MemRange[cutlass.Int64, pipeline_p_full_size]
            pipeline_o_mbar: cute.struct.MemRange[cutlass.Int64, pipeline_o_size]
            tmem_dealloc_mbar: cutlass.Int64
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
            tma_atom_O,
            tma_tensor_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
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
        tma_atom_O: Optional[cute.CopyAtom],
        mO_tma: Optional[cute.Tensor],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
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
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = Int32(0)
        else:
            mma_tile_coord_v = cute.arch.block_idx()[0] % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        pipeline_q = None
        pipeline_s_p = None
        pipeline_p_full = None
        pipeline_o = None
        tmem_alloc_barrier = None
        tmem_free_barrier = None
        tmem = None
        if const_expr(self.use_2cta_instrs):
            ThreadGroup = partial(
                cutlass.pipeline.CooperativeGroup,
                cutlass.pipeline.Agent.Thread,
            )
            mma_warp = ThreadGroup(1)
            tma_warp = ThreadGroup(1)
            silu_threads_cluster = ThreadGroup(cute.arch.WARP_SIZE * len(self.silu0_warp_ids) * self.cta_group_size)
            silu_warps_cluster = ThreadGroup(len(self.silu0_warp_ids) * self.cta_group_size)
            pipeline_q = cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.pipeline_q_mbar.data_ptr(),
                num_stages=self.q_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_q_bytes,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
            pipeline_kv = cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.pipeline_kv_mbar.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_k_bytes,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
            pipeline_s_p = cutlass.pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.pipeline_s_p_mbar.data_ptr(),
                num_stages=self.s_stage,
                producer_group=mma_warp,
                consumer_group=silu_threads_cluster,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
            pipeline_p_full = cutlass.pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.pipeline_p_full_mbar.data_ptr(),
                num_stages=self.s_stage,
                producer_group=silu_warps_cluster,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
            pipeline_o = cutlass.pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.pipeline_o_mbar.data_ptr(),
                num_stages=self.o_stage,
                producer_group=mma_warp,
                consumer_group=silu_threads_cluster,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
            tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
                barrier_id=TMEM_POINTER_BARRIER,
                num_threads=cute.arch.WARP_SIZE,
            )
            tmem_free_barrier = cutlass.pipeline.NamedBarrier(
                barrier_id=TMEM_RELEASE_BARRIER,
                num_threads=cute.arch.WARP_SIZE * (1 + len(self.silu0_warp_ids) + len(self.silu1_warp_ids)),
            )
            tmem = cutlass.utils.TmemAllocator(
                storage.tmem_holding_buf.ptr,
                barrier_for_retrieve=tmem_alloc_barrier,
                allocator_warp_id=self.mma_warp_id,
                is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            )
            pipeline_init_arrive(
                cluster_shape_mn=cta_layout_vmnk,
                is_relaxed=True,
            )
        else:
            # Use the first N warps to initialize the 1-CTA barriers.
            if warp_idx == 1:
                for i in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_full_offset + i, len([self.load_warp_id]))
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id]))
            if warp_idx == 2:
                if const_expr(self.s0_s1_barrier):
                    for i in cutlass.range_constexpr(self.q_stage):
                        cute.arch.mbarrier_init(mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE * len(self.silu0_warp_ids))
            if warp_idx == 3:
                for i in cutlass.range_constexpr(self.s_stage):
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_O_rescaled_offset + i, cute.arch.WARP_SIZE * len(self.silu0_warp_ids))
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id]))
                    if const_expr(not self.use_o_pipeline):
                        cute.arch.mbarrier_init(mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id]))
            if warp_idx == 4:
                for i in cutlass.range_constexpr(self.s_stage):
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_2_offset + i, cute.arch.WARP_SIZE * len(self.silu0_warp_ids))
            if warp_idx == 5:
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_tmem_dealloc_offset,
                    cute.arch.WARP_SIZE * len((*self.silu0_warp_ids, *self.silu1_warp_ids)),
                )
            if const_expr(self.use_o_pipeline):
                # Keep this initialization deferred so the following KV
                # pipeline's CTA-wide initialization fence publishes both
                # pipelines together.
                pipeline_o = cutlass.pipeline.PipelineUmmaAsync.create(
                    barrier_storage=storage.pipeline_o_mbar.data_ptr(),
                    num_stages=self.o_stage,
                    producer_group=cutlass.pipeline.CooperativeGroup(
                        cutlass.pipeline.Agent.Thread,
                    ),
                    consumer_group=cutlass.pipeline.CooperativeGroup(
                        cutlass.pipeline.Agent.Thread,
                        cute.arch.WARP_SIZE * len(self.silu1_warp_ids),
                    ),
                    cta_layout_vmnk=cta_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)

        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner), sO_layout.outer)

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C((self.mma_tiler_pv[0], self.mma_tiler_pv[1]))
        if const_expr(self.use_2cta_instrs):
            tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.o_stage))
            tOtO = cute.make_tensor(
                tOtO.iterator + self.tmem_o_offset[0],
                tOtO.layout,
            )
        else:
            tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout) for stage in range(self.s_stage))
        if const_expr(self.use_2cta_instrs):
            tOtOs = tuple(tOtO[None, None, None, stage] for stage in range(self.o_stage))
        else:
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
            self.logical_cta_tiler,
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
        if const_expr(self.use_clc_scheduler):
            clc_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
            clc_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                self.threads_per_cta * self.cta_group_size,
            )
            clc = ClcState.create(
                hw_scheduler=cutlass.utils.ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    storage.clc_response.data_ptr(),
                ),
                pipeline=cutlass.pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=cutlass.pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Consumer,
                    self.sched_stages,
                ),
                producer_state=cutlass.pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Producer,
                    self.sched_stages,
                ),
            )
            TileSchedulerCls = partial(
                self.tile_scheduler_cls.create,
                tile_sched_params,
                clc=clc,
            )
            if const_expr(self.use_clc_descriptor):
                descriptor_producer_group = cutlass.pipeline.CooperativeGroup(
                    cutlass.pipeline.Agent.Thread,
                    cute.arch.WARP_SIZE,
                )
                descriptor_consumer_group = cutlass.pipeline.CooperativeGroup(
                    cutlass.pipeline.Agent.Thread,
                    self.threads_per_cta - cute.arch.WARP_SIZE,
                )
                descriptor_pipeline = cutlass.pipeline.PipelineAsync.create(
                    barrier_storage=storage.descriptor_mbar_ptr.data_ptr(),
                    num_stages=self.descriptor_stages,
                    producer_group=descriptor_producer_group,
                    consumer_group=descriptor_consumer_group,
                )
                descriptor_consumer = ClcDescriptorState.create(
                    pipeline=descriptor_pipeline,
                    buffer_ptr=storage.work_descriptor.data_ptr(),
                    consumer_state=cutlass.pipeline.make_pipeline_state(
                        cutlass.pipeline.PipelineUserType.Consumer,
                        self.descriptor_stages,
                    ),
                    producer_state=cutlass.pipeline.make_pipeline_state(
                        cutlass.pipeline.PipelineUserType.Producer,
                        self.descriptor_stages,
                    ),
                )
                descriptor_producer = ClcDescriptorState.create(
                    pipeline=descriptor_pipeline,
                    buffer_ptr=storage.work_descriptor.data_ptr(),
                    consumer_state=cutlass.pipeline.make_pipeline_state(
                        cutlass.pipeline.PipelineUserType.Consumer,
                        self.descriptor_stages,
                    ),
                    producer_state=cutlass.pipeline.make_pipeline_state(
                        cutlass.pipeline.PipelineUserType.Producer,
                        self.descriptor_stages,
                    ),
                )
                TileSchedulerCls = partial(
                    self.tile_scheduler_cls.create,
                    tile_sched_params,
                    clc=clc,
                    descriptor=descriptor_consumer,
                    descriptor_producer=False,
                )
                TileSchedulerProducerCls = partial(
                    self.tile_scheduler_cls.create,
                    tile_sched_params,
                    clc=clc,
                    descriptor=descriptor_producer,
                    descriptor_producer=True,
                )
            else:
                TileSchedulerProducerCls = TileSchedulerCls
        else:
            TileSchedulerCls = partial(
                self.tile_scheduler_cls.create,
                tile_sched_params,
            )
            TileSchedulerProducerCls = TileSchedulerCls

        if const_expr(self.use_2cta_instrs):
            pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  CLC SCHEDULER / EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx >= self.empty_warp_ids[0] and warp_idx <= self.empty_warp_ids[-1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                if const_expr(self.use_2cta_instrs):
                    if is_leader_cta:
                        self.clc_scheduler_warp(TileSchedulerProducerCls)
                    else:
                        self.empty_warp(TileSchedulerCls)
                else:
                    self.clc_scheduler_warp(TileSchedulerProducerCls)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    self.empty_warp(TileSchedulerCls)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            if const_expr(self.use_2cta_instrs):
                self.load_2cta(
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
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
            elif const_expr(not self.is_paged):
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
                    mbar_ptr,
                    block_info,
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
            if const_expr(self.use_2cta_instrs):
                tmem.allocate(self.tmem_alloc_cols)
                tmem.wait_for_alloc()
                self.mma_2cta(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    sQ,
                    sK,
                    sV,
                    sQ_layout.inner,
                    sK_layout.inner,
                    sV_layout.inner,
                    tOrPs,
                    pipeline_q,
                    pipeline_kv,
                    pipeline_s_p,
                    pipeline_p_full,
                    pipeline_o,
                    is_leader_cta,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
                tmem.relinquish_alloc_permit()
                tmem_free_barrier.arrive_and_wait()
                tmem_ptr_alloc = tmem.retrieve_ptr(Float32)
                tmem.free(tmem_ptr_alloc)
            else:
                # Allocate the 1-CTA tensor-memory buffer.
                tmem_alloc_cols = Int32(self.tmem_alloc_cols)
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf.ptr)
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
                        tOrPs,
                        pipeline_kv,
                        pipeline_o,
                        mbar_ptr,
                        block_info,
                        block_sparse_tensors,
                        SeqlenInfoCls,
                        TileSchedulerCls,
                    )
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_tmem_dealloc_offset,
                    0,
                )
                tmem_ptr_alloc = cute.arch.retrieve_tmem_ptr(
                    Float32,
                    alignment=16,
                    ptr_to_buffer_holding_addr=storage.tmem_holding_buf.ptr,
                )
                cute.arch.dealloc_tmem(tmem_ptr_alloc, tmem_alloc_cols)

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
                mO_tma=mO_tma,
                sO=sO,
                mbar_ptr=mbar_ptr,
                tma_atom_O=tma_atom_O,
                pipeline_o=pipeline_o,
            )
            silu_loop = partial(
                self.silu_loop,
                score_scale=score_scale,
                scaling_seqlen=scaling_seqlen,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                thr_mma_qk=thr_mma_qk,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                block_sparse_tensors=block_sparse_tensors,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                store_O=store_O,
                func=func,
                mma_tile_coord_v=mma_tile_coord_v,
                pipeline_s_p=pipeline_s_p,
                pipeline_p_full=pipeline_p_full,
            )
            if warp_idx <= self.silu0_warp_ids[-1] and warp_idx >= self.silu0_warp_ids[0]:
                tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
                silu_loop(stage=0, tStSi=tStSi)
                if const_expr(self.use_2cta_instrs):
                    tmem_free_barrier.arrive()
                else:
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            if warp_idx <= self.silu1_warp_ids[-1] and warp_idx >= self.silu1_warp_ids[0]:
                tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[1], tStS.layout)
                silu_loop(stage=1, tStSi=tStSi)
                if const_expr(self.use_2cta_instrs):
                    tmem_free_barrier.arrive()
                else:
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        return

    @cute.jit
    def get_seqlen_info(
        self,
        work_tile,
        batch_idx: Int32,
        SeqlenInfoCls: Callable,
    ):
        if const_expr(self.use_clc_descriptor):
            return work_tile
        return SeqlenInfoCls(batch_idx)

    @cute.jit
    def decode_q1_split_head(self, head_idx: Int32):
        if const_expr(self.q1_split_kv == 1):
            return head_idx, Int32(0)
        return head_idx // self.q1_split_kv, head_idx % self.q1_split_kv

    @cute.jit
    def get_q1_split_n_block_info(self, n_block_max: Int32, n_block_min: Int32, split_idx: Int32):
        if const_expr(self.q1_split_kv == 1):
            return n_block_max, n_block_min
        blocks_per_split = cute.ceil_div(n_block_max - n_block_min, self.q1_split_kv)
        split_max = max(n_block_min, n_block_max - split_idx * blocks_per_split)
        split_min = max(n_block_min, split_max - blocks_per_split)
        return split_max, split_min

    @cute.jit
    def clc_scheduler_warp(self, TileSchedulerCls: Callable):
        tile_scheduler = TileSchedulerCls()
        if const_expr(self.use_clc_descriptor):
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_scheduler.prefetch_next_work()
                work_tile = tile_scheduler.advance_to_next_work()
        else:
            work_tile = tile_scheduler.clc.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_scheduler.prefetch_next_work()
                tile_scheduler.clc.consumer_wait()
                work_tile = tile_scheduler.clc.get_current_work()
                tile_scheduler.clc.consumer_release()
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(self, TileSchedulerCls: Callable):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()
        tile_scheduler.consumer_tail()

    @cute.jit
    def load_2cta(
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
        pipeline_q: cutlass.pipeline.PipelineTmaUmma,
        pipeline_kv: cutlass.pipeline.PipelineTmaUmma,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        assert self.use_2cta_instrs
        q_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer,
            self.q_stage,
        )
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer,
            self.kv_stage,
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = self.get_seqlen_info(
                work_tile,
                batch_idx,
                SeqlenInfoCls,
            )
            offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
            offset_dynamic = 0 if offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic else offset_dynamic
            mQ_cur = cute.domain_offset(
                (seqlen.offset_q - offset_dynamic, 0),
                mQ[None, None, head_idx],
            )
            gQ = cute.local_tile(
                mQ_cur,
                cute.select(self.mma_tiler_qk, mode=[0, 2]),
                (None, 0),
            )
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mK_cur = cute.domain_offset(
                (seqlen.offset_k, 0),
                mK[None, None, head_idx_kv],
            )
            mV_cur = cute.domain_offset(
                (0, seqlen.offset_k),
                mV[None, None, head_idx_kv],
            )
            gK = cute.local_tile(
                mK_cur,
                cute.select(self.mma_tiler_qk, mode=[1, 2]),
                (None, 0),
            )
            gV = cute.local_tile(
                mV_cur,
                cute.select(self.mma_tiler_pv, mode=[1, 2]),
                (0, None),
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ, 0, 3),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )
            n_block_max, n_block_min, _ = block_info.get_n_block_info(
                seqlen,
                m_block,
                offset_dynamic,
            )
            if n_block_max > n_block_min:
                n_block_k = n_block_max - 1
                n_block_v = n_block_max - 1

                # K0 -> Q -> K1 is the latency-hiding prologue.
                self.load_tma_2cta(
                    tma_atom_K,
                    tKgK,
                    tKsK,
                    pipeline_kv,
                    kv_producer_state,
                    n_block_k,
                )
                kv_producer_state.advance()
                n_block_k -= 1
                self.load_tma_2cta(
                    tma_atom_Q,
                    tQgQ,
                    tQsQ,
                    pipeline_q,
                    q_producer_state,
                    m_block,
                )
                q_producer_state.advance()
                if n_block_k >= n_block_min:
                    self.load_tma_2cta(
                        tma_atom_K,
                        tKgK,
                        tKsK,
                        pipeline_kv,
                        kv_producer_state,
                        n_block_k,
                    )
                    kv_producer_state.advance()
                    n_block_k -= 1

                # Interleave the V needed by PV with the next K needed by QK.
                while n_block_k >= n_block_min:
                    self.load_tma_2cta(
                        tma_atom_V,
                        tVgV,
                        tVsV,
                        pipeline_kv,
                        kv_producer_state,
                        n_block_v,
                    )
                    kv_producer_state.advance()
                    n_block_v -= 1
                    self.load_tma_2cta(
                        tma_atom_K,
                        tKgK,
                        tKsK,
                        pipeline_kv,
                        kv_producer_state,
                        n_block_k,
                    )
                    kv_producer_state.advance()
                    n_block_k -= 1

                while n_block_v >= n_block_min:
                    self.load_tma_2cta(
                        tma_atom_V,
                        tVgV,
                        tVsV,
                        pipeline_kv,
                        kv_producer_state,
                        n_block_v,
                    )
                    kv_producer_state.advance()
                    n_block_v -= 1

            work_tile = tile_scheduler.advance_to_next_work()
        tile_scheduler.consumer_tail()
        pipeline_q.producer_tail(q_producer_state)
        pipeline_kv.producer_tail(kv_producer_state)

    @cute.jit
    def load_tma_2cta(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        load_pipeline: cutlass.pipeline.PipelineTmaUmma,
        producer_state: cutlass.pipeline.PipelineState,
        block: Int32,
    ):
        load_pipeline.producer_acquire(producer_state)
        tma_bar_ptr = load_pipeline.producer_get_barrier(producer_state)
        cute.copy(
            tma_atom,
            tXgX[None, block],
            tXsX[None, producer_state.index],
            tma_bar_ptr=tma_bar_ptr,
        )

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
            head_idx, split_idx = self.decode_q1_split_head(head_idx)
            seqlen = self.get_seqlen_info(work_tile, batch_idx, SeqlenInfoCls)
            offset = seqlen.offset_q
            offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
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
            n_block_max, n_block_min = self.get_q1_split_n_block_info(n_block_max, n_block_min, split_idx)
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
                    load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)  # K0
                    load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
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

            work_tile = tile_scheduler.advance_to_next_work()
            # End of persistent scheduler loop
        tile_scheduler.consumer_tail()

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
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
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
            seqlen = self.get_seqlen_info(work_tile, batch_idx, SeqlenInfoCls)
            offset = seqlen.offset_q
            offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
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

                work_tile = tile_scheduler.advance_to_next_work()
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

                work_tile = tile_scheduler.advance_to_next_work()
            ### End of persistent scheduler loop
        tile_scheduler.consumer_tail()

    @cute.jit
    def mma_2cta(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ_swizzle: cute.Swizzle,
        sK_swizzle: cute.Swizzle,
        sV_swizzle: cute.Swizzle,
        tOrPs: Tuple[cute.Tensor, ...],
        pipeline_q: cutlass.pipeline.PipelineTmaUmma,
        pipeline_kv: cutlass.pipeline.PipelineTmaUmma,
        pipeline_s_p: cutlass.pipeline.PipelineUmmaAsync,
        pipeline_p_full: cutlass.pipeline.PipelineAsyncUmma,
        pipeline_o: cutlass.pipeline.PipelineUmmaAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        assert self.use_2cta_instrs
        assert self.q_stage == 1
        assert self.s_stage == 2
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        tSrQ = thr_mma_qk.make_fragment_A(sQ)
        tSrK = thr_mma_qk.make_fragment_B(sK)
        tOrV = thr_mma_pv.make_fragment_B(sV)
        tSrQ0 = tSrQ[None, None, None, 0]
        qk_mma_op = tiled_mma_qk.op
        pv_mma_op = tiled_mma_pv.op
        gemm_S0 = partial(
            sm100_utils.gemm_ptx_partial,
            qk_mma_op,
            self.tmem_s_offset[0],
            tSrQ0,
            sA=sQ[None, None, None, 0],
            sA_swizzle=sQ_swizzle,
            sB_swizzle=sK_swizzle,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        gemm_S1 = partial(
            sm100_utils.gemm_ptx_partial,
            qk_mma_op,
            self.tmem_s_offset[1],
            tSrQ0,
            sA=sQ[None, None, None, 0],
            sA_swizzle=sQ_swizzle,
            sB_swizzle=sK_swizzle,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        gemm_P0 = partial(
            sm100_utils.gemm_ptx_partial,
            pv_mma_op,
            self.tmem_o_offset[0],
            tOrPs[0],
            sA=None,
            sA_swizzle=None,
            sB_swizzle=sV_swizzle,
            split_arrive=self.split_P_arrive,
            cta_group=self.cta_group_size,
        )
        gemm_P1 = partial(
            sm100_utils.gemm_ptx_partial,
            pv_mma_op,
            self.tmem_o_offset[0],
            tOrPs[1],
            sA=None,
            sA_swizzle=None,
            sB_swizzle=sV_swizzle,
            split_arrive=self.split_P_arrive,
            cta_group=self.cta_group_size,
        )

        q_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer,
            self.q_stage,
        )
        kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer,
            self.kv_stage,
        )
        s0_producer_phase = Int32(0)
        s1_producer_phase = Int32(0)
        p0_consumer_phase = Int32(0)
        p1_consumer_phase = Int32(0)
        o_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer,
            self.o_stage,
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, _, batch_idx = work_tile.tile_idx
            seqlen = self.get_seqlen_info(
                work_tile,
                batch_idx,
                SeqlenInfoCls,
            )
            offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
            offset_dynamic = 0 if offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic else offset_dynamic
            n_block_max, n_block_min, _ = block_info.get_n_block_info(
                seqlen,
                m_block,
                offset_dynamic,
            )
            n_block_nums = n_block_max - n_block_min

            if n_block_nums > 0 and is_leader_cta:
                pipeline_o.producer_acquire(o_producer_state)
                o_tmem_addr = Int32(self.tmem_o_offset[0]) + o_producer_state.index * self.head_dim_v_padded
                pipeline_q.consumer_wait(q_consumer_state)

                # QK0
                pipeline_kv.consumer_wait(kv_consumer_state)
                k_index = kv_consumer_state.index
                gemm_S0(
                    tCrB=tSrK[None, None, None, k_index],
                    sB=sK[None, None, None, k_index],
                )
                pipeline_s_p.sync_object_full.arrive(
                    Int32(0),
                    pipeline_s_p.producer_mask,
                    pipeline_s_p.cta_group,
                )
                pipeline_kv.consumer_release(kv_consumer_state)
                kv_consumer_state.advance()

                # QK1
                if n_block_nums > 1:
                    pipeline_kv.consumer_wait(kv_consumer_state)
                    k_index = kv_consumer_state.index
                    gemm_S1(
                        tCrB=tSrK[None, None, None, k_index],
                        sB=sK[None, None, None, k_index],
                    )
                    pipeline_s_p.sync_object_full.arrive(
                        Int32(1),
                        pipeline_s_p.producer_mask,
                        pipeline_s_p.cta_group,
                    )
                    pipeline_kv.consumer_release(kv_consumer_state)
                    kv_consumer_state.advance()

                o_should_accumulate = False
                for i in cutlass.range(n_block_nums - 2, unroll=1):
                    pipeline_kv.consumer_wait(kv_consumer_state)
                    v_release_state = kv_consumer_state.clone()
                    v_index = kv_consumer_state.index
                    tOrVi = tOrV[None, None, None, v_index]
                    sV_cur = sV[None, None, None, v_index]
                    kv_consumer_state.advance()
                    pipeline_kv.consumer_wait(kv_consumer_state)
                    k_index = kv_consumer_state.index
                    if i & 1 == 0:
                        self.mma_pv_2cta(
                            gemm_P0,
                            o_tmem_addr,
                            tOrVi,
                            sV_cur,
                            o_should_accumulate,
                            pipeline_s_p,
                            0,
                            s0_producer_phase,
                            pipeline_p_full,
                            p0_consumer_phase,
                        )
                        gemm_S0(
                            tCrB=tSrK[None, None, None, k_index],
                            sB=sK[None, None, None, k_index],
                        )
                        pipeline_s_p.sync_object_full.arrive(
                            Int32(0),
                            pipeline_s_p.producer_mask,
                            pipeline_s_p.cta_group,
                        )
                        s0_producer_phase ^= 1
                        p0_consumer_phase ^= 1
                    else:
                        self.mma_pv_2cta(
                            gemm_P1,
                            o_tmem_addr,
                            tOrVi,
                            sV_cur,
                            o_should_accumulate,
                            pipeline_s_p,
                            1,
                            s1_producer_phase,
                            pipeline_p_full,
                            p1_consumer_phase,
                        )
                        gemm_S1(
                            tCrB=tSrK[None, None, None, k_index],
                            sB=sK[None, None, None, k_index],
                        )
                        pipeline_s_p.sync_object_full.arrive(
                            Int32(1),
                            pipeline_s_p.producer_mask,
                            pipeline_s_p.cta_group,
                        )
                        s1_producer_phase ^= 1
                        p1_consumer_phase ^= 1
                    pipeline_kv.consumer_release(v_release_state)
                    pipeline_kv.consumer_release(kv_consumer_state)
                    kv_consumer_state.advance()
                    o_should_accumulate = True

                pipeline_q.consumer_release(q_consumer_state)
                q_consumer_state.advance()

                if n_block_nums > 1:
                    pipeline_kv.consumer_wait(kv_consumer_state)
                    v_index = kv_consumer_state.index
                    if (n_block_nums - 2) & 1 == 0:
                        self.mma_pv_2cta(
                            gemm_P0,
                            o_tmem_addr,
                            tOrV[None, None, None, v_index],
                            sV[None, None, None, v_index],
                            o_should_accumulate,
                            pipeline_s_p,
                            0,
                            s0_producer_phase,
                            pipeline_p_full,
                            p0_consumer_phase,
                        )
                        s0_producer_phase ^= 1
                        p0_consumer_phase ^= 1
                    else:
                        self.mma_pv_2cta(
                            gemm_P1,
                            o_tmem_addr,
                            tOrV[None, None, None, v_index],
                            sV[None, None, None, v_index],
                            o_should_accumulate,
                            pipeline_s_p,
                            1,
                            s1_producer_phase,
                            pipeline_p_full,
                            p1_consumer_phase,
                        )
                        s1_producer_phase ^= 1
                        p1_consumer_phase ^= 1
                    pipeline_kv.consumer_release(kv_consumer_state)
                    kv_consumer_state.advance()
                    o_should_accumulate = True

                pipeline_kv.consumer_wait(kv_consumer_state)
                v_index = kv_consumer_state.index
                if (n_block_nums - 1) & 1 == 0:
                    self.mma_pv_2cta(
                        gemm_P0,
                        o_tmem_addr,
                        tOrV[None, None, None, v_index],
                        sV[None, None, None, v_index],
                        o_should_accumulate,
                        pipeline_s_p,
                        0,
                        s0_producer_phase,
                        pipeline_p_full,
                        p0_consumer_phase,
                    )
                    s0_producer_phase ^= 1
                    p0_consumer_phase ^= 1
                else:
                    self.mma_pv_2cta(
                        gemm_P1,
                        o_tmem_addr,
                        tOrV[None, None, None, v_index],
                        sV[None, None, None, v_index],
                        o_should_accumulate,
                        pipeline_s_p,
                        1,
                        s1_producer_phase,
                        pipeline_p_full,
                        p1_consumer_phase,
                    )
                    s1_producer_phase ^= 1
                    p1_consumer_phase ^= 1
                pipeline_o.producer_commit(o_producer_state)
                o_producer_state.advance()
                pipeline_kv.consumer_release(kv_consumer_state)
                kv_consumer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()
        tile_scheduler.consumer_tail()

    @cute.jit
    def mma_pv_2cta(
        self,
        gemm_P: Callable,
        o_tmem_addr: Int32,
        tOrV: cute.Tensor,
        sV: cute.Tensor,
        o_should_accumulate: Boolean,
        pipeline_s_p: cutlass.pipeline.PipelineUmmaAsync,
        stage: int,
        s_producer_phase: Int32,
        pipeline_p_full: cutlass.pipeline.PipelineAsyncUmma,
        p_consumer_phase: Int32,
    ):
        pipeline_s_p.sync_object_empty.wait(
            Int32(stage),
            s_producer_phase,
        )
        gemm_P(
            tCrB=tOrV,
            sB=sV,
            zero_init=not o_should_accumulate,
            acc_tmem_addr_dynamic=o_tmem_addr,
            mbar_ptr=pipeline_p_full.sync_object_full.get_barrier(Int32(stage)),
            mbar_phase=p_consumer_phase,
        )
        pipeline_p_full.sync_object_empty.arrive(
            Int32(stage),
            pipeline_p_full.consumer_mask,
            pipeline_p_full.cta_group,
        )

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

        if const_expr(self.use_precomputed_qk_descriptors):
            qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
            q_smem_base = sm100_desc.smem_desc_base_from_tensor(
                sQ,
                sm100_desc.Major.K,
            )
            k_smem_base = sm100_desc.smem_desc_base_from_tensor(
                sK,
                sm100_desc.Major.K,
            )
            q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]
            sm100_utils.declare_ptx_smem_desc(
                q_smem_start[self.q_stage - 1],
                q_smem_base,
                tSrQ[None, None, None, 0].layout,
                var_name_prefix="hstu_fwd_q_smem_desc",
            )
            sm100_utils.declare_ptx_idesc(
                qk_mma_op,
                var_name="hstu_fwd_qk_mma_idesc",
            )
            sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_precomputed_varname,
                    self.tmem_s_offset[stage],
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK[None, None, None, 0].layout,
                    smem_var_name_prefix="hstu_fwd_q_smem_desc",
                    idesc_var_name="hstu_fwd_qk_mma_idesc",
                    kind=qk_mma_kind,
                    smem_offset=(-sQ_stage_stride if stage == 0 else sQ_stage_stride),
                    zero_init=True,
                )
                for stage in range(2)
            ]
        else:
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
            seqlen = self.get_seqlen_info(work_tile, batch_idx, SeqlenInfoCls)
            offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
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
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase)
                    if const_expr(self.use_precomputed_qk_descriptors):
                        gemm_Si[stage](smem_desc_start_b=(sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)))
                    else:
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
                        # 2. acquire O0/O1_partial and P0/P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the SiLU warps have finished reading tO from the
                        # previous work tile.
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase)
                        # 3. gemm
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
                        # 4. O_full is signaled only after the final PV iteration.
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
                        # Don't need to wait for the SiLU warp to finish reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if const_expr(self.use_precomputed_qk_descriptors):
                            gemm_Si[stage](smem_desc_start_b=(sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)))
                        else:
                            gemm_Si[stage](
                                tCrB=tSrK[
                                    None,
                                    None,
                                    None,
                                    Ki_index,
                                ],
                                sB=sK_cur,
                            )
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
                    # 2. acquire Oi_partial and Pi
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase)
                    # 3. gemm
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
                    # Signal O_full after the final PV iteration so the epilogue can
                    # safely consume the completed accumulator.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop
        tile_scheduler.consumer_tail()

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
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        pipeline_o: Optional[cutlass.pipeline.PipelineUmmaAsync],
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
        if const_expr(self.use_o_pipeline):
            o_producer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer,
                self.o_stage,
            )
        P_full_O_rescaled_phase = [Int32(0) for _ in range(self.s_stage)]

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            _, split_idx = self.decode_q1_split_head(head_idx)
            seqlen = self.get_seqlen_info(work_tile, batch_idx, SeqlenInfoCls)
            offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
            offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
            n_block_max, n_block_min, _ = block_info.get_n_block_info(seqlen, m_block, offset_dynamic)
            if const_expr(self.use_auto_block_metadata):
                n_block_max, _, _, _, _ = get_q2k_block_sparse_consumer_row(
                    block_sparse_tensors,
                    batch_idx,
                    m_block,
                )
                n_block_min = Int32(0)
            n_block_max, n_block_min = self.get_q1_split_n_block_info(n_block_max, n_block_min, split_idx)
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

                # QK uses separate TMEM columns, so it can overlap the prior
                # epilogue. Before the first PV MMA touches O, wait until those
                # epilogue warps have released the single output accumulator.
                if const_expr(self.use_o_pipeline):
                    pipeline_o.producer_acquire(o_producer_state)
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
                if const_expr(self.use_o_pipeline):
                    pipeline_o.producer_commit(o_producer_state)
                    o_producer_state.advance()
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset)
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        # End of persistent scheduler loop
        tile_scheduler.consumer_tail()
        if const_expr(self.use_o_pipeline):
            pipeline_o.producer_tail(o_producer_state)

    @cute.jit
    def silu_loop(
        self,
        stage: int | Int32,
        score_scale: Float32,
        scaling_seqlen: Float32,
        window_size_left: Int32,
        window_size_right: Int32,
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
        mma_tile_coord_v: Int32 = Int32(0),
        pipeline_s_p: Optional[cutlass.pipeline.PipelineUmmaAsync] = None,
        pipeline_p_full: Optional[cutlass.pipeline.PipelineAsyncUmma] = None,
    ):
        """Compute silu on attention scores from QK matrix multiplication."""
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * (len(self.silu0_warp_ids)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(tStSi.layout, cute.make_layout((self.kBlockM, tilePlikeFP32)))
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_op = (
            tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(16)) if const_expr(self.kBlockM == 64) else tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        )
        tmem_load_atom = cute.make_copy_atom(tmem_load_op, Float32)
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_op = (
            tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(8)) if const_expr(self.kBlockM == 64) else tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16))
        )
        tmem_store_atom = cute.make_copy_atom(tmem_store_op, Float32)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP)
        thr_tmem_store = tiled_tmem_store.get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        epi_consumer_phase = Int32(0)
        if const_expr(self.use_o_pipeline):
            epi_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer,
                self.o_stage,
            )
        mma_si_consumer_phase = Int32(0)
        s0_s1_sequence_phase = Int32(1 if self.use_2cta_instrs or stage == 0 else 0)
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset
        score_scale_half = score_scale * 0.5
        output_scale = cute.arch.rcp_approx(scaling_seqlen)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            head_idx, split_idx = self.decode_q1_split_head(head_idx)
            q_work_block = m_block
            seqlen = self.get_seqlen_info(work_tile, batch_idx, SeqlenInfoCls)
            offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
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
            n_block_max, n_block_min = self.get_q1_split_n_block_info(n_block_max, n_block_min, split_idx)
            if const_expr(self.q1_split_kv > 1):
                n_masking_steps = min(n_masking_steps, n_block_max - n_block_min) if split_idx == 0 else Int32(0)
            has_work = n_block_max > n_block_min
            # func: (head_func, n_func, L_func) -> (n_func, L_func)
            func_tensor = func[0, None, None] if func is not None else None

            # m_block consider q stage
            m_block = self.q_stage * m_block + (stage if self.q_stage == 2 else 0)
            if const_expr(self.use_2cta_instrs):
                m_block = m_block * self.cta_group_size + mma_tile_coord_v
            mask = AttentionMaskCls(
                offset_q=seqlen.offset_q, seqlen_q=seqlen.seqlen_q, seqlen_k=seqlen.seqlen_k, offset_dynamic=offset_dynamic, func=func_tensor
            )
            thr_mma_mask = thr_mma_qk.get_slice(0) if const_expr(self.use_2cta_instrs) else thr_mma_qk
            mask_fn = partial(mask.apply_mask, m_block=m_block, thr_mma=thr_mma_mask, thr_tmem_load=thr_tmem_load)
            r2p_mask_fn = partial(
                mask.build_mask_r2p,
                m_block=m_block,
                thr_mma=thr_mma_mask,
                thr_tmem_load=thr_tmem_load,
            )
            seqlen_mask_fn = partial(
                mask.apply_mask_seqlen,
                m_block=m_block,
                thr_mma=thr_mma_mask,
                thr_tmem_load=thr_tmem_load,
            )
            fastsilu = FastSilU(
                score_scale=score_scale,
                score_scale_half=score_scale_half,
            )

            silu_step = partial(
                self.silu_step,
                fastsilu=fastsilu,
                seqlen_k=seqlen.seqlen_k,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                stage=stage,
                pipeline_s_p=pipeline_s_p,
                pipeline_p_full=pipeline_p_full,
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
                    if const_expr(self.use_causal_mask_r2p):
                        (
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                        ) = silu_step(
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                            r2p_mask_fn=partial(r2p_mask_fn),
                        )
                    else:
                        (
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                        ) = silu_step(
                            mma_si_consumer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                            mask_fn=partial(mask_fn),
                        )
                    masking_step += wg_stride
                    n_block_valid -= wg_stride

                while n_block_valid >= n_block_min:
                    n_block = n_block_valid
                    mma_si_consumer_phase, s0_s1_sequence_phase = silu_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block)
                    n_block_valid -= wg_stride

            # epilogue step
            if self.q_stage == 2 or stage == 1:
                store_O_args = partial(
                    store_O,
                    seqlen=seqlen,
                    scale=output_scale,
                    m_block=m_block,
                    head_idx=head_idx,
                    stage=stage if self.q_stage == 2 else 0,
                    o_stage=(epi_consumer_state.index if self.use_o_pipeline else Int32(0)),
                    epi_consumer_phase=epi_consumer_phase,
                    has_work=has_work,
                )
                if const_expr(self.use_clc_scheduler):
                    if has_work:
                        store_O_args()
                else:
                    store_O_args()

            # Advance to next tile
            if has_work:
                if const_expr(self.use_o_pipeline):
                    epi_consumer_state.advance()
                    epi_consumer_phase = epi_consumer_state.phase
                else:
                    epi_consumer_phase ^= 1
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop
        tile_scheduler.consumer_tail()
        if const_expr(self.use_tma_O):
            if tidx < cute.arch.WARP_SIZE:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(
                barrier_id=EPILOGUE_BARRIER_BASE + stage,
                number_of_threads=(cute.arch.WARP_SIZE * len(self.silu1_warp_ids)),
            )

    @cute.jit
    def _silu_step_q1_m64(
        self,
        mma_si_consumer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        score_scale: Float32,
        score_scale_half: Float32,
        seqlen_k: Int32,
        window_size_left: Int32,
        window_size_right: Int32,
        mbar_ptr: cute.Pointer,
        stage: int | Int32,
    ):
        """Convert one M64 score row-group to 16-bit P with the native 32-DP TMEM mapping."""
        assert self.kBlockM == 64 and self.kBlockN in (64, 128)
        assert not self.use_2cta_instrs

        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_S_full_offset + stage,
            mma_si_consumer_phase,
        )
        valid_col_begin = Int32(0)
        valid_col_end = seqlen_k
        if const_expr(self.is_local):
            # Packed HSTU aligns its single Q row with K row seqlen_k - 1.
            # Local attention is consequently a contiguous suffix of K.
            aligned_q_row = seqlen_k - Int32(1)
            valid_col_begin = max(Int32(0), aligned_q_row - window_size_left)
            valid_col_end = min(seqlen_k, aligned_q_row + Int32(1) + window_size_right)
        block_col_begin = n_block * self.kBlockN
        block_valid_begin = min(max(valid_col_begin - block_col_begin, Int32(0)), self.kBlockN)
        block_valid_end = min(max(valid_col_end - block_col_begin, Int32(0)), self.kBlockN)
        if const_expr(self.q1_m64_16dp_silu):
            scores_16dp = cute.make_rmem_tensor((self.kBlockN // 2,), Float32)
            packed_16dp = cute.make_rmem_tensor((8,), Int32)
            lane_parity = (cute.arch.thread_idx()[0] >> 1) & 1
            for chunk in cutlass.range_constexpr(self.kBlockN // 32):
                values_16dp = _tmem_load_16dp64b16x(Int32(self.tmem_s_offset[stage] + chunk * 32))
                for i in cutlass.range_constexpr(16):
                    score_idx = chunk * 16 + i
                    col = chunk * 32 + i * 2 + lane_parity
                    scores_16dp[score_idx] = (
                        values_16dp[i] if const_expr(self.q1_m64_tail_branch) or (col >= block_valid_begin and col < block_valid_end) else Float32(0.0)
                    )
            cute.arch.fence_view_async_tmem_load()
            if const_expr(self.q1_m64_tail_branch):
                if block_valid_begin > 0 or block_valid_end < self.kBlockN:
                    for chunk in cutlass.range_constexpr(self.kBlockN // 32):
                        for i in cutlass.range_constexpr(16):
                            col = chunk * 32 + i * 2 + lane_parity
                            scores_16dp[chunk * 16 + i] = scores_16dp[chunk * 16 + i] if col >= block_valid_begin and col < block_valid_end else Float32(0.0)
            _silu_f32_inplace(scores_16dp, score_scale_half)

            split_chunk_16dp = self.split_P_arrive // 32
            for chunk in cutlass.range_constexpr(split_chunk_16dp):
                for i in cutlass.range_constexpr(8):
                    even = scores_16dp[chunk * 16 + i * 2]
                    odd = scores_16dp[chunk * 16 + i * 2 + 1]
                    partner_even = cute.arch.shuffle_sync_bfly(even, offset=2)
                    partner_odd = cute.arch.shuffle_sync_bfly(odd, offset=2)
                    lo = even if lane_parity == 0 else partner_odd
                    hi = partner_even if lane_parity == 0 else odd
                    packed_16dp[i] = _cvt_f32x2_to_16x2(lo, hi, self.q_dtype)
                _tmem_store_16x8_16dp64b(Int32(self.tmem_p_offset[stage] + chunk * 16), packed_16dp)
            cute.arch.fence_view_async_tmem_store()

            if const_expr(self.s0_s1_barrier):
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_s0_s1_sequence_offset + (1 - stage))
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

            for chunk in cutlass.range_constexpr(split_chunk_16dp, self.kBlockN // 32):
                for i in cutlass.range_constexpr(8):
                    even = scores_16dp[chunk * 16 + i * 2]
                    odd = scores_16dp[chunk * 16 + i * 2 + 1]
                    partner_even = cute.arch.shuffle_sync_bfly(even, offset=2)
                    partner_odd = cute.arch.shuffle_sync_bfly(odd, offset=2)
                    lo = even if lane_parity == 0 else partner_odd
                    hi = partner_even if lane_parity == 0 else odd
                    packed_16dp[i] = _cvt_f32x2_to_16x2(lo, hi, self.q_dtype)
                _tmem_store_16x8_16dp64b(Int32(self.tmem_p_offset[stage] + chunk * 16), packed_16dp)
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
            return mma_si_consumer_phase ^ 1, s0_s1_sequence_phase ^ 1

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % len(self.silu0_warp_ids)
        split_chunk = self.split_P_arrive // 32
        if const_expr(self.q1_m64_silu_warps == 0):
            scores = cute.make_rmem_tensor((self.kBlockN,), Float32)
            packed = cute.make_rmem_tensor((16,), Int32)
            for chunk in cutlass.range_constexpr(self.kBlockN // 32):
                values = _tmem_load_32dp32b32x(Int32(self.tmem_s_offset[stage] + chunk * 32))
                for i in cutlass.range_constexpr(32):
                    scores[chunk * 32 + i] = values[i]
            cute.arch.fence_view_async_tmem_load()

            for i in cutlass.range_constexpr(self.kBlockN):
                scores[i] = scores[i] if i >= block_valid_begin and i < block_valid_end else Float32(0.0)
            if const_expr(self.q1_m64_inplace_silu):
                _silu_f32_inplace(scores, score_scale_half)
            else:
                converted = cute.make_rmem_tensor((self.kBlockN,), self.q_dtype)
                FastSilU(score_scale, score_scale_half).silu_x2(scores, converted, None)

            for chunk in cutlass.range_constexpr(split_chunk):
                for i in cutlass.range_constexpr(16):
                    packed[i] = _cvt_f32x2_to_16x2(scores[chunk * 32 + i * 2], scores[chunk * 32 + i * 2 + 1], self.q_dtype)
                _tmem_store_16x16(Int32(self.tmem_p_offset[stage] + chunk * 16), packed)
            cute.arch.fence_view_async_tmem_store()
        else:
            active_scores = cute.make_rmem_tensor((self.kBlockN,), Float32)
            active_packed = cute.make_rmem_tensor((16,), Int32)
            active_converted = cute.make_rmem_tensor((self.kBlockN,), self.q_dtype)
            values = (Float32(0.0),) * 32
            if warp_idx < self.q1_m64_silu_warps:
                for chunk in cutlass.range_constexpr(self.kBlockN // 32):
                    values = _tmem_load_32dp32b32x(Int32(self.tmem_s_offset[stage] + chunk * 32))
                    for i in cutlass.range_constexpr(32):
                        active_scores[chunk * 32 + i] = values[i]
                cute.arch.fence_view_async_tmem_load()

                for i in cutlass.range_constexpr(self.kBlockN):
                    active_scores[i] = active_scores[i] if i >= block_valid_begin and i < block_valid_end else Float32(0.0)
                FastSilU(score_scale, score_scale_half).silu_x2(active_scores, active_converted, None)

                for chunk in cutlass.range_constexpr(split_chunk):
                    for i in cutlass.range_constexpr(16):
                        active_packed[i] = _cvt_f32x2_to_16x2(active_scores[chunk * 32 + i * 2], active_scores[chunk * 32 + i * 2 + 1], self.q_dtype)
                    _tmem_store_16x16(Int32(self.tmem_p_offset[stage] + chunk * 16), active_packed)
                cute.arch.fence_view_async_tmem_store()

        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_s0_s1_sequence_offset + (1 - stage))
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

        if const_expr(self.q1_m64_silu_warps == 0):
            for chunk in cutlass.range_constexpr(split_chunk, self.kBlockN // 32):
                for i in cutlass.range_constexpr(16):
                    packed[i] = _cvt_f32x2_to_16x2(scores[chunk * 32 + i * 2], scores[chunk * 32 + i * 2 + 1], self.q_dtype)
                _tmem_store_16x16(Int32(self.tmem_p_offset[stage] + chunk * 16), packed)
            cute.arch.fence_view_async_tmem_store()
        else:
            if warp_idx < self.q1_m64_silu_warps:
                for chunk in cutlass.range_constexpr(split_chunk, self.kBlockN // 32):
                    for i in cutlass.range_constexpr(16):
                        active_packed[i] = _cvt_f32x2_to_16x2(active_scores[chunk * 32 + i * 2], active_scores[chunk * 32 + i * 2 + 1], self.q_dtype)
                    _tmem_store_16x16(Int32(self.tmem_p_offset[stage] + chunk * 16), active_packed)
                cute.arch.fence_view_async_tmem_store()

        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        return mma_si_consumer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def silu_step(
        self,
        mma_si_consumer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        fastsilu: FastSilU,
        seqlen_k: Int32,
        window_size_left: Int32,
        window_size_right: Int32,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        stage: int | Int32,
        mask_fn: Optional[Callable] = None,
        r2p_mask_fn: Optional[Callable] = None,
        pipeline_s_p: Optional[cutlass.pipeline.PipelineUmmaAsync] = None,
        pipeline_p_full: Optional[cutlass.pipeline.PipelineAsyncUmma] = None,
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

        if const_expr(self.kBlockM == 64):
            return self._silu_step_q1_m64(
                mma_si_consumer_phase,
                s0_s1_sequence_phase,
                n_block,
                fastsilu.score_scale,
                fastsilu.score_scale_half,
                seqlen_k,
                window_size_left,
                window_size_right,
                mbar_ptr,
                stage,
            )

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1])))

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

        # Wait for Si and make sure the previous PV no longer reads this P stage.
        if const_expr(self.use_2cta_instrs):
            s_consumer_state = cutlass.pipeline.PipelineState(
                self.s_stage,
                Int32(0),
                Int32(stage),
                mma_si_consumer_phase,
            )
            p_producer_state = cutlass.pipeline.PipelineState(
                self.s_stage,
                Int32(0),
                Int32(stage),
                s0_s1_sequence_phase,
            )
            pipeline_p_full.producer_acquire(p_producer_state)
            pipeline_s_p.consumer_wait(s_consumer_state)
        else:
            cute.arch.mbarrier_wait(
                mbar_ptr + self.mbar_S_full_offset + stage,
                mma_si_consumer_phase,
            )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)  # copy from tmem to rmem
        cute.arch.fence_view_async_tmem_load()

        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
            tSrS_t2r.layout,
        )
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
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
        # Notify the MMA warp that the first portion of P is ready.
        if const_expr(self.use_2cta_instrs):
            pipeline_s_p.consumer_release(s_consumer_state)
        else:
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        # Write remaining P columns
        for i in cutlass.range_constexpr(split_P_arrive_idx, cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify the MMA warp that all P is ready.
        if const_expr(self.use_2cta_instrs):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_full.producer_commit(p_producer_state)
        else:
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        return mma_si_consumer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def store_O(
        self,
        m_block: int,
        head_idx: int,
        stage: int,
        o_stage: Int32,
        scale: float,
        epi_consumer_phase: Int32,
        seqlen: Callable,
        gmem_tiled_copy_O: cute.TiledCopy,
        thr_mma_pv: cute.ThrMma,
        tOtOs: cute.Tensor,
        mO: cute.Tensor,
        mO_tma: Optional[cute.Tensor],
        sO: cute.Tensor,
        mbar_ptr: cute.Pointer,
        tma_atom_O: Optional[cute.CopyAtom],
        has_work: Boolean,
        pipeline_o: Optional[cutlass.pipeline.PipelineUmmaAsync] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.silu1_warp_ids))
        if const_expr(self.use_tma_O):
            if tidx < cute.arch.WARP_SIZE:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(
                barrier_id=EPILOGUE_BARRIER_BASE + stage,
                number_of_threads=(cute.arch.WARP_SIZE * len(self.silu1_warp_ids)),
            )
        offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
        offset_dynamic = 0 if (offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic) else offset_dynamic
        if const_expr(self.use_2cta_instrs):
            tOtO = cute.make_tensor(
                tOtOs[0].iterator + o_stage * self.head_dim_v_padded,
                tOtOs[0].layout,
            )
        else:
            tOtO = tOtOs[stage]
        # sO is CTA-local even when the TMEM accumulator spans the 2-CTA tile.
        tOsO = thr_mma_pv.get_slice(0).partition_C(sO[None, None, stage])
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
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = (
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.o_dtype)
            if const_expr(self.kBlockM == 64)
            else sm100_utils_basic.get_smem_store_op(self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load)
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_r2s = utils.partition_D_position_independent(
            thr_tmem_load,
            tOsO_i[(None, None), None],
        )

        if const_expr(self.use_o_pipeline):
            o_consumer_state = cutlass.pipeline.PipelineState(
                self.o_stage,
                Int32(0),
                o_stage,
                epi_consumer_phase,
            )
        if has_work:
            if const_expr(self.use_o_pipeline):
                pipeline_o.consumer_wait(o_consumer_state)
            else:
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_O_full_offset + stage,
                    epi_consumer_phase,
                )
        participates = tidx < cute.arch.WARP_SIZE if const_expr(self.kBlockM == 64 or self.q1_single_warp_epilogue) else True
        if participates:
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

        if const_expr(self.use_o_pipeline):
            if has_work:
                cute.arch.fence_view_async_tmem_load()
                pipeline_o.consumer_release(o_consumer_state)

        if const_expr(self.use_tma_O):
            # Publish regular SMEM writes before the async proxy reads sO for the TMA store.
            cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=EPILOGUE_BARRIER_BASE + stage, number_of_threads=cute.arch.WARP_SIZE * len(self.silu1_warp_ids))

        if const_expr(self.q1_split_kv > 1):
            self._atomic_add_q1_split_O(head_idx, stage, seqlen, mO, sO)
            return

        logical_stage_start = m_block * self.kBlockM - offset_dynamic
        valid_rows = min(
            self.kBlockM,
            seqlen.seqlen_q - logical_stage_start,
        )
        if const_expr(self.use_tma_O):
            if logical_stage_start >= 0 and valid_rows == self.kBlockM:
                row_coord = mO_tma.shape[0] - valid_rows
                segment_end_coord = seqlen.offset_q + logical_stage_start + valid_rows
                mO_tma_cur = mO_tma[
                    None,
                    None,
                    head_idx,
                    segment_end_coord,
                ]
                mO_tma_cur = cute.domain_offset(
                    (row_coord, 0),
                    mO_tma_cur,
                )
                gO_tma = cute.local_tile(
                    mO_tma_cur,
                    self.epi_tile,
                    (0, 0),
                )
                smem_partition, gmem_partition = cpasync.tma_partition(
                    tma_atom_O,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(
                        sO[None, None, stage],
                        0,
                        cute.rank(sO[None, None, stage]),
                    ),
                    cute.group_modes(
                        gO_tma,
                        0,
                        cute.rank(gO_tma),
                    ),
                )
                if tidx < cute.arch.WARP_SIZE:
                    cute.copy(
                        tma_atom_O,
                        smem_partition,
                        gmem_partition,
                    )
                    cute.arch.cp_async_bulk_commit_group()
            else:
                self._store_O_to_gmem(
                    m_block,
                    head_idx,
                    stage,
                    seqlen,
                    gmem_tiled_copy_O,
                    mO,
                    sO,
                )
        else:
            self._store_O_to_gmem(
                m_block,
                head_idx,
                stage,
                seqlen,
                gmem_tiled_copy_O,
                mO,
                sO,
            )

    @cute.jit
    def _atomic_add_q1_split_O(
        self,
        head_idx: Int32,
        stage: int,
        seqlen: Callable,
        mO: cute.Tensor,
        sO: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.silu1_warp_ids))
        if tidx < self.head_dim_v_padded // 2 and seqlen.seqlen_q > 0:
            dim_idx = tidx * 2
            val_lo = sO[0, dim_idx, stage].to(Float32)
            val_hi = sO[0, dim_idx + 1, stage].to(Float32)
            mO_row = mO[seqlen.offset_q, None, head_idx]
            _atomic_add_bf16x2((mO_row.iterator + dim_idx).llvm_ptr, val_lo, val_hi)

    @cute.jit
    def _store_O_to_gmem(
        self,
        m_block: int,
        head_idx: int,
        stage: int,
        seqlen: Callable,
        gmem_tiled_copy_O: cute.TiledCopy,
        mO: cute.Tensor,
        sO: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.silu1_warp_ids))
        offset = seqlen.offset_q + m_block * self.kBlockM
        offset_dynamic = (self.logical_cta_tiler[0] - (seqlen.seqlen_q & (self.logical_cta_tiler[0] - 1))) & (self.logical_cta_tiler[0] - 1)
        offset_dynamic = 0 if offset_dynamic <= self.kBlockM or not self.enable_offset_dynamic else offset_dynamic
        mO_cur = cute.domain_offset(
            (offset - offset_dynamic, 0),
            mO[None, None, head_idx],
        )
        gO = cute.local_tile(
            mO_cur,
            (self.kBlockM, self.head_dim_v_padded),
            (0, 0),
        )
        cO = cute.make_identity_tensor((self.kBlockM, self.head_dim_v_padded))
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
                cute.autovec_copy(
                    tOsO[None, rest_m, None],
                    tOrO[None, rest_m, None],
                )
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=(tOpO[None, rest_m, None] if self.check_hdim_v_oob else None),
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
