# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from typing import Type, Tuple, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import Int32, Float32, Boolean
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap

from . import _bprop_mxfp8_masks_sm100 as fmha_masks
from . import _bprop_mxfp8_common_sm100 as cute_common
from . import _bprop_mxfp8_common_sm100 as d256_primitives
from ._bprop_mxfp8_common_sm100 import LOW_PRECISION_TYPE, SF_DTYPE, SF_VEC_SIZE
from ._bprop_mxfp8_common_sm100 import (
    make_kv_head_batch_tensor,
    make_lse_head_batch_tensor,
    make_q_head_batch_tensor,
    make_transposed_tensor,
)

"""
SM100 D256 MXFP8 SDPA backward: the dQ kernel (2-CTA, online or fixed dS scale).

Ported from Xinbo Zhao's fmha_mxfp8_large_head_dim (2026-09-01).
Kept as close to the source as the package rules allow so upstream fixes
stay diff-able; only the imports and this note differ.

Original module docstring follows.

A fused multi-head attention (FMHA) backward pass example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of the backward pass of fused multi-head attention
using a TMA + Blackwell SM100 TensorCore warp-specialized kernel. The implementation fuses the computation of
dQ, dK, and dV into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, compute, reduce)
- Pipeline stages between different warps for overlapping computation and memory access
- Support causal masking
- Support for sliding window attention

To run this example:

.. code-block:: bash

    python fmha_bwd.py \
        --s_q_max 8192 --s_k_max 8192 \
        --h_q 16 --h_k 16 --d 256 --b 4 \
        --element_dtype float16 --acc_dtype float32 \
        --mma_tiler_mn 128,128

The above example runs FMHA backward with max sequence length 1024 for Q and K,
batch size 1, 8 attention heads for Q and K, and head dimension 256.
The Blackwell tcgen05 MMA tile shape is (128, 128), and the kernel uses fp16 for input/output
with fp32 for accumulation.

Constraints for this example:
* Supported head dimension: 256
* mma_tiler_mn must be 128,128
* For causal masking, use --is_causal
* For variable sequence lengths, use --varlen
* For sliding window attention, use --window_size x,y
"""


class BlackwellFmhaBackwardDQ256:
    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        varlen: bool,
        mask_type: fmha_masks.MaskEnum,
        is_persistent: bool = False,
        online_ds_scale: bool = True,
        store_num_bits_per_copy: int | None = None,
    ):
        self._setup_specialization(element_dtype, acc_dtype)
        self._setup_mma_tilers(mma_tiler)
        self._setup_warp_topology_and_barriers(varlen, mask_type, is_persistent, online_ds_scale, store_num_bits_per_copy)

    def _setup_specialization(self, element_dtype, acc_dtype):
        """Set fixed instruction, cluster, and numeric-type policy."""
        # For 2-CTA MMA: cluster_shape_mn = (2, 1) with mma_tiler_mn = (256, 128)
        self.use_2cta_instrs = True
        self.use_2cta_divisor = 2
        self.cluster_shape_mn = (2, 1)
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.sf_dtype = SF_DTYPE
        self.sf_vec_size = SF_VEC_SIZE
        self.element_dtype = element_dtype
        self.acc_dtype = acc_dtype

    def _setup_mma_tilers(self, mma_tiler):
        """Derive MMA and CTA-local tile shapes."""
        # Full-d single-issuer path: each CTA covers two 64-row Q sub-blocks.
        # Warp 8 issues the two sub-blocks serially; warp 9 stays idle.
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        # Per-MMA-tile cta_tiler for mask functions (each MMA tile = 64 Q rows per CTA)
        self.mask_cta_tiler = (
            mma_tiler[0] // 2 if self.use_2cta_instrs else mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.tile_shape_Q = mma_tiler[0]
        self.tile_shape_K = mma_tiler[1]
        self.tile_shape_dQ_K = mma_tiler[2]
        self.head_dim = mma_tiler[2]
        if self.head_dim != 256:
            raise ValueError("fmha_bwd_dQ only supports head dimension 256")

        self.CTA_shape_Q = mma_tiler[0]
        self.CTA_shape_K = mma_tiler[1]
        # For S, (128, 64, 128)
        self.QK_mma_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.QK_mma_tiler_sfb = (
            max(self.QK_mma_tiler[0] // 2 if self.use_2cta_instrs else self.QK_mma_tiler[0], 128),
            cute.round_up(self.QK_mma_tiler[1], 128),
            self.QK_mma_tiler[2],
        )
        # SFA-specific tiler: doubled M for CtaGroup.TWO compatibility
        # sfa_tile_shape[0] = mma_tiler_sfa[0] / cta_group = (M*2)/2 = M = atom_m
        cta_group_size = 2 if self.use_2cta_instrs else 1
        self.QK_mma_tiler_sfa = (
            self.QK_mma_tiler[0] * cta_group_size,
            self.QK_mma_tiler[1],
            self.QK_mma_tiler[2],
        )
        # For dP
        self.DOV_mma_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.DOV_mma_tiler_sfb = (
            max(self.DOV_mma_tiler[0] // 2 if self.use_2cta_instrs else self.DOV_mma_tiler[0], 128),
            cute.round_up(self.DOV_mma_tiler[1], 128),
            self.DOV_mma_tiler[2],
        )

        self.DOV_mma_tiler_sfa = (
            self.DOV_mma_tiler[0] * cta_group_size,
            self.DOV_mma_tiler[1],
            self.DOV_mma_tiler[2],
        )
        # dQ uses full D=256 in one dSK MMA pass.
        self.dSK_mma_tiler = (
            mma_tiler[0],
            mma_tiler[2],
            mma_tiler[1],
        )
        self.dSK_mma_tiler_sfb = (
            max(mma_tiler[0] // 2 if self.use_2cta_instrs else mma_tiler[0], 128),
            mma_tiler[2],
            cute.round_up(mma_tiler[1], 128),
        )

        # CTA-local tile shapes for 2-CTA mode (M dimension divided by number of CTAs)
        # These are used for identity tensors and CTA-specific operations
        cta_m_divisor = 2 if self.use_2cta_instrs else 1
        self.QK_cta_tiler = (
            self.QK_mma_tiler[0] // cta_m_divisor,
            self.QK_mma_tiler[1],
            self.QK_mma_tiler[2],
        )
        self.DOV_cta_tiler = (
            self.DOV_mma_tiler[0] // cta_m_divisor,
            self.DOV_mma_tiler[1],
            self.DOV_mma_tiler[2],
        )
        self.dSK_cta_tiler = (
            self.dSK_mma_tiler[0] // cta_m_divisor,
            self.dSK_mma_tiler[1],
            self.dSK_mma_tiler[2],
        )

    def _setup_warp_topology_and_barriers(self, varlen, mask_type, is_persistent, online_ds_scale, store_num_bits_per_copy):
        """Configure runtime specialization, warp roles, registers, and barriers."""
        self.varlen = varlen
        self.mask_type = mask_type
        self.is_persistent = is_persistent
        self.online_ds_scale = online_ds_scale

        if is_persistent:
            # In persistent mode, repurpose empty warp 11 as CLC sched warp
            self.sched_warp_id = 11
            # 8 compute + 2 mma + load + sched = 12 warps (no empty warp)
        self.empty_warp_id = 11  # In persistent mode, warp 11 is sched; this field is unused
        # num_compute_warps + 4 (mma0, mma1, load, empty/sched = 12 total)
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (8 + 4)  # 8 compute + 2 mma + load + empty/sched
        cute_common.init_common_config(self)
        if store_num_bits_per_copy is not None:
            if store_num_bits_per_copy not in (self.element_dtype.width, 128):
                raise ValueError("unsupported dQ store width")
            self.store_num_bits_per_copy = store_num_bits_per_copy
        self.num_regs_compute = 192
        self.num_regs_mma = 104
        # Single-MMA-warp dQ variant: warp 8 is the only MMA issuer.
        self.mma_warp_id_0 = 8
        self.mma_warp_id_1 = 99  # no physical warp; kept to leave old branches unreachable
        self.sfv_s2t_warp_id = 9
        self.load_warp_id = 10
        # Logical load-pipeline consumers: warp 8 consumes two serial Q
        # sub-block passes. This does not enable a second MMA issuer.
        self.num_mma_warps = 2
        self.num_dq_subblocks = 2
        self.epilogue_sync_barrier_1 = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=self.num_compute_1_warps * self.threads_per_warp,
        )
        self.sfv_s2t_start_barrier = pipeline.NamedBarrier(
            barrier_id=9,
            num_threads=2 * self.threads_per_warp,
        )
        self.sfv_s2t_done_barrier = pipeline.NamedBarrier(
            barrier_id=10,
            num_threads=2 * self.threads_per_warp,
        )
        self.dS_scale_exchange_barrier_0 = pipeline.NamedBarrier(
            barrier_id=11,
            num_threads=(self.num_compute_0_warps + 1) * self.threads_per_warp,
        )
        self.dS_scale_exchange_barrier_1 = pipeline.NamedBarrier(
            barrier_id=12,
            num_threads=(self.num_compute_1_warps + 1) * self.threads_per_warp,
        )
        # Persistent tile boundary barrier: sync all warps except sched warp
        # (load + 2 mma + 8 compute = 11 warps)
        self.persistent_tile_barrier = pipeline.NamedBarrier(
            barrier_id=8,
            num_threads=11 * self.threads_per_warp,
        )

        # CLC pipeline configuration
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16  # Fixed by CLC hardware (128-bit opaque response)

    def _setup_pipeline_stages_and_sf_tilers(self):
        """Derive trace-time pipeline depths and scale-factor tile shapes."""
        self.load_mma_K_stage = 2
        # 2-MMA-warp: 1 stage per warp (no double-buffering; S/dP tmem stages repurposed as warp slots)
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.compute_mma_P_stage = 2  # unused in dQ kernel
        self.compute_mma_dS_stage = 1
        self.mma_compute_dQ_stage = 1
        self.tmem_S_stages = 2
        self.tmem_dP_stages = 1
        self.k_halves = 2
        self.SFK_halves = self.k_halves
        self.d_halves = 1
        self.KT_load_mma_K_stage = self.load_mma_K_stage
        self.SFK_mn_load_mma_K_stage = self.load_mma_K_stage
        self.QK_mma_tiler_sfb_load = (
            self.QK_mma_tiler_sfb[0],
            self.QK_mma_tiler_sfb[1],
            128 if self.SFK_halves > 1 else self.QK_mma_tiler_sfb[2],
        )
        self.SFK_load_mma_K_stage = self.load_mma_K_stage * self.SFK_halves
        self.DOV_mma_tiler_sfb_load = (
            self.DOV_mma_tiler_sfb[0],
            self.DOV_mma_tiler_sfb[1],
            128,
        )
        self.DOV_mma_tiler_sfa_load = (
            self.DOV_mma_tiler_sfa[0],
            self.DOV_mma_tiler_sfa[1],
            128,
        )
        self.SFDO_load_mma_Q_stage = 2 * self.k_halves
        self.QK_mma_tiler_sfa_load = (
            self.QK_mma_tiler_sfa[0],
            self.QK_mma_tiler_sfa[1],
            128,
        )
        self.SFQ_load_mma_Q_stage = 2 * self.k_halves
        self.SFV_load_mma_K_stage = self.load_mma_K_stage * self.k_halves

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        Q: cute.Tensor,
        K: cute.Tensor,
        K_MN: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        SF_Q: cute.Tensor,
        SF_K: cute.Tensor,
        SF_KT: cute.Tensor,
        SF_V: cute.Tensor,
        SF_DO: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        dO_16bits: cute.Tensor,
        LSE: cute.Tensor,
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        workspace: cute.Tensor,
        stream: cuda.CUstream,
        skip_sum_odo: bool = False,
    ):
        q_seq_max, k_seq_max, d, hb = problem_shape
        h, b = hb
        h_r, h_k = h
        Q = make_q_head_batch_tensor(Q, hb, self.varlen)
        K = make_kv_head_batch_tensor(K, hb, self.varlen)
        KT = make_transposed_tensor(K_MN, K.layout)
        V = make_kv_head_batch_tensor(V, hb, self.varlen)
        O = make_q_head_batch_tensor(O, hb, self.varlen)
        dO_16bits = cute.make_tensor(dO_16bits.iterator, O.layout)
        dK = make_kv_head_batch_tensor(dK, hb, self.varlen)
        dV = make_kv_head_batch_tensor(dV, hb, self.varlen)
        dQ = make_q_head_batch_tensor(dQ, hb, self.varlen)
        dO = cute.make_tensor(dO.iterator, Q.layout)
        LSE = make_lse_head_batch_tensor(LSE, hb)

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()
        if cutlass.const_expr(self.Q_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.K_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_pipeline_stages_and_sf_tilers()
        # compute S - using self.cta_group for 2-CTA support
        QK_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.QK_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        # SMEM-sourced variant for d>128 (Q operand read from SMEM directly).
        QK_tiled_mma_smem = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.QK_mma_tiler[:2],
            tcgen05.OperandSource.SMEM,
        )
        QK_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,  # SFB uses ONE for partition
            self.QK_mma_tiler_sfb[:2],
            tcgen05.OperandSource.TMEM,
        )
        # SFA-specific tiled MMA: CtaGroup.TWO with doubled M so shape_mnk[0]=256
        # This ensures mma_tile_inst_m=1 in make_smem_layout_sfa → per-inst M=128=atom_m
        QK_tiled_mma_sfa = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.QK_mma_tiler_sfa[:2],
            tcgen05.OperandSource.TMEM,
        )

        # compute dP - using self.cta_group for 2-CTA support
        # OperandSource.SMEM: dO is read directly from smem (no tmem needed)
        DOV_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.DOV_mma_tiler[:2],
            tcgen05.OperandSource.SMEM,
        )
        DOV_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,  # SFB uses ONE for partition
            self.DOV_mma_tiler_sfb[:2],
            tcgen05.OperandSource.TMEM,
        )
        DOV_tiled_mma_sfa = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.DOV_mma_tiler_sfa[:2],
            tcgen05.OperandSource.TMEM,
        )

        # compute dQ - using self.cta_group for 2-CTA support
        dSK_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.dSK_mma_tiler[:2],
        )
        dSK_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,  # SFB uses ONE for partition
            self.dSK_mma_tiler_sfb[:2],
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (QK_tiled_mma.thr_id.shape,),
        )
        cluster_layout_vmnk_sfb = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (QK_tiled_mma_sfb.thr_id.shape,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_b(
            QK_tiled_mma,
            self.QK_mma_tiler,
            LOW_PRECISION_TYPE,
            self.load_mma_K_stage,
        )
        KT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            LOW_PRECISION_TYPE,
            self.KT_load_mma_K_stage,
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QK_tiled_mma,
            self.QK_mma_tiler,
            LOW_PRECISION_TYPE,
            2,  # this number is related to perf, do not change it
        )
        sSFK_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            QK_tiled_mma_sfb,
            self.QK_mma_tiler_sfb_load,
            self.sf_vec_size,
            self.SFK_load_mma_K_stage,
        )
        sSFK_smem_layout_staged = cute_common.expand_last_SF_stride(sSFK_smem_layout_staged)

        sSFK_mn_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            dSK_tiled_mma_sfb,
            self.dSK_mma_tiler_sfb,
            self.sf_vec_size,
            self.SFK_mn_load_mma_K_stage,
        )
        sSFK_mn_smem_layout_staged = cute_common.expand_last_SF_stride(sSFK_mn_smem_layout_staged)

        # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
        # Use doubled M tiler so sfa_tile_shape[0] = (M*2)/2 = M = atom_m = 128
        # 2-MMA-warp: 2 stages for SFQ (one per Q tile)
        sSFQ_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            QK_tiled_mma_sfa,
            self.QK_mma_tiler_sfa_load,
            self.sf_vec_size,
            self.SFQ_load_mma_Q_stage,
        )
        sSFQ_smem_layout_staged = cute_common.expand_last_SF_stride(sSFQ_smem_layout_staged)

        V_smem_layout_staged = sm100_utils.make_smem_layout_b(
            DOV_tiled_mma,
            self.DOV_mma_tiler,
            LOW_PRECISION_TYPE,
            self.load_mma_K_stage,
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_a(
            DOV_tiled_mma,
            self.DOV_mma_tiler,
            LOW_PRECISION_TYPE,
            2,  # this number is related to perf, do not change it
        )
        SFV_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            DOV_tiled_mma_sfb,
            self.DOV_mma_tiler_sfb_load,
            self.sf_vec_size,
            self.SFV_load_mma_K_stage,
        )
        SFV_smem_layout_staged = cute_common.expand_last_SF_stride(SFV_smem_layout_staged)

        # Two Q sub-blocks, each with an independent stage per 128-D half.
        SFDO_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            DOV_tiled_mma_sfa,
            self.DOV_mma_tiler_sfa_load,
            self.sf_vec_size,
            self.SFDO_load_mma_Q_stage,
        )
        SFDO_smem_layout_staged = cute_common.expand_last_SF_stride(SFDO_smem_layout_staged)

        self.dS_total_stages = self.compute_mma_dS_stage * self.num_dq_subblocks
        dS_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            LOW_PRECISION_TYPE,
            self.dS_total_stages,
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            K,
            K_smem_layout,
            self.QK_mma_tiler,
            QK_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        KT_smem_layout = cute.select(KT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_KT, tma_tensor_KT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            KT,
            KT_smem_layout,
            self.dSK_mma_tiler,
            dSK_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            V,
            V_smem_layout,
            self.DOV_mma_tiler,
            DOV_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.QK_mma_tiler,
            QK_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.DOV_mma_tiler,
            DOV_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, Q_smem_layout)
        self.tma_copy_K_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, K_smem_layout)
        self.tma_copy_KT_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, KT_smem_layout)
        self.tma_copy_V_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, V_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, dO_smem_layout)

        # Scale factors: TMA descriptors straight over cuDNN's canonical F8_128x4
        # planes (one plane per (b, h)); see the native-SF section of
        # _bprop_mxfp8_common_sm100.  Rowwise tensors (rows = S, groups along
        # D) are plane-major; the columnwise K^T scales (rows = D, groups
        # along S_kv) keep the D tile outside the plane and are fetched as a
        # 2-tile box (both 128-row D tiles of one S_kv group column).
        # (head_dim is the static template value; the kernel-side byte counts
        # below must stay Python ints, never host IR values.)
        sf_d_groups = self.head_dim // self.sf_vec_size
        sf_d_tiles = self.head_dim // 128
        # Plane counts from the problem shape: the KV tensors' grouped head mode
        # broadcasts h_r, so its cute.size would over-count KV planes (harmless
        # for a plane-major map, wrong for the columnwise K^T tile stride).
        sf_q_planes = h_r * h_k * b
        sf_kv_planes = h_k * b
        sfq_tmap = cute_common.make_sf_tensor_map(SF_Q, q_seq_max, sf_d_groups, sf_q_planes, plane_major=True)
        sfk_tmap = cute_common.make_sf_tensor_map(SF_K, k_seq_max, sf_d_groups, sf_kv_planes, plane_major=True)
        sfv_tmap = cute_common.make_sf_tensor_map(SF_V, k_seq_max, sf_d_groups, sf_kv_planes, plane_major=True)
        sfdo_tmap = cute_common.make_sf_tensor_map(SF_DO, q_seq_max, sf_d_groups, sf_q_planes, plane_major=True)
        sfkt_tmap = cute_common.make_sf_tensor_map(
            SF_KT,
            self.head_dim,
            (k_seq_max + self.sf_vec_size - 1) // self.sf_vec_size,
            sf_kv_planes,
            plane_major=False,
            row_tiles_per_box=sf_d_tiles,
        )
        # Every SF slot is one 512-byte atom per 128-wide K half (the K^T slot
        # holds both D tiles); the peer-CTA / row-group-shifted copies are built
        # in smem by the load warp, not loaded.
        sf_atom = cute_common.SF_ATOM_BYTES
        self.sf_kv_stage_bytes = (self.SFK_halves + self.k_halves) * sf_atom + sf_d_tiles * sf_atom  # SFK + SFV + SFK_mn
        self.sf_q_prologue_bytes = 2 * 2 * self.k_halves * sf_atom  # SFQ + SFDO, two Q sub-blocks, both K halves

        @cute.struct
        class SharedStorage:
            # Pipeline barriers — per-warp pipelines for 2 MMA warps
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage * 2]  # shared, 2-CTA
            # Per-CTA tx barriers for this CTA's canonical scale-factor atoms (one per K stage)
            sf_landed_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage]
            # MMA warp 0 pipelines
            mma_compute_S_mbar_ptr_0: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr_0: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            compute_mma_dS_mbar_ptr_0: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            mma_compute_dQ_mbar_ptr_0: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dQ_stage * 2]
            # MMA warp 1 pipelines
            mma_compute_S_mbar_ptr_1: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr_1: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            compute_mma_dS_mbar_ptr_1: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            mma_compute_dQ_mbar_ptr_1: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dQ_stage * 2]
            tmem_holding_buf: cutlass.Int32
            # For 2-CTA tmem deallocation barrier
            tmem_dealloc_mbar_ptr: cutlass.Int64
            # CLC dynamic scheduler barriers and response (persistent mode)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            # CLC response: 16 bytes (128-bit opaque) written by hardware
            clc_response_ptr: cute.struct.MemRange[cutlass.Int32, 4]
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(K_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sKT: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(KT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(V_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(dO_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sDS: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(dS_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sDS_scale_exchange: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, 1024],
                128,
            ]

            sSFQ: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sSFQ_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sSFK_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFK_mn: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sSFK_mn_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(SFV_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFDO: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(SFDO_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        sum_OdO, scaled_LSE, _ = cute_common.get_workspace_tensor(self, problem_shape, workspace, self.acc_dtype)

        # =============================== Sum OdO ===============================
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))

        sum_OdO_grid = cute_common.compute_sum_odo_grid(problem_shape, self.sum_OdO_block_q)

        if not skip_sum_odo:
            cute_common.sum_OdO(
                self,
                O,
                dO_16bits,
                sum_OdO,
                LSE,
                scaled_LSE,
                cumulative_s_q,
                sum_OdO_scale,
                LSE_scale,
                problem_shape,
            ).launch(
                grid=sum_OdO_grid,
                block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
                cluster=[1, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

        bwd_grid = cute_common.compute_grid(
            cute.shape((q_seq_max, d, ((h_r, h_k), b))),
            self.cta_tiler,
        )
        # Round up grid X to be divisible by cluster X dimension (2-CTA cluster)
        # Extra blocks will early-exit due to bounds checking in kernel
        cluster_x = self.cluster_shape_mn[0]
        bwd_grid = (
            ((bwd_grid[0] + cluster_x - 1) // cluster_x) * cluster_x,
            bwd_grid[1],
            bwd_grid[2],
        )

        tile_sched_params = None
        if cutlass.const_expr(self.is_persistent):
            # CLC dynamic persistent scheduling: compute tile space and grid
            cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                bwd_grid,  # (Q_tiles_aligned, h_q, b) = total CTA tile space
                cluster_shape_mnk,
            )
            bwd_grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

        self.bwd(
            QK_tiled_mma,
            DOV_tiled_mma,
            dSK_tiled_mma,
            QK_tiled_mma_smem,
            QK_tiled_mma_sfb,
            DOV_tiled_mma_sfb,
            dSK_tiled_mma_sfb,
            QK_tiled_mma_sfa,
            DOV_tiled_mma_sfa,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_KT,
            tma_tensor_KT,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_dO,
            tma_tensor_dO,
            sfq_tmap,
            sfk_tmap,
            sfkt_tmap,
            sfv_tmap,
            sfdo_tmap,
            dQ,
            scaled_LSE,
            scale_softmax,
            sum_OdO,
            problem_shape,
            cumulative_s_q,
            cumulative_s_k,
            window_size_left,
            window_size_right,
            sSFK_smem_layout_staged,
            sSFK_mn_smem_layout_staged,
            sSFQ_smem_layout_staged,
            SFV_smem_layout_staged,
            SFDO_smem_layout_staged,
            K_smem_layout_staged,
            KT_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dS_smem_layout_staged,
            cluster_layout_vmnk,
            tile_sched_params if self.is_persistent else None,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[*self.cluster_shape_mn, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def bwd(
        self,
        QK_tiled_mma: cute.TiledMma,
        DOV_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        QK_tiled_mma_smem: cute.TiledMma,
        QK_tiled_mma_sfb: cute.TiledMma,
        DOV_tiled_mma_sfb: cute.TiledMma,
        dSK_tiled_mma_sfb: cute.TiledMma,
        QK_tiled_mma_sfa: cute.TiledMma,
        DOV_tiled_mma_sfa: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_KT: cute.CopyAtom,
        KT_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        sfq_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfk_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfkt_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfv_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfdo_tmap: cutlass.GridConstant[tmap.TensorMap],
        dQ: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: Float32,
        sum_OdO: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sSFK_smem_layout_staged: cute.Layout,
        sfK_mn_smem_layout_staged: cute.Layout,
        sSFQ_smem_layout_staged: cute.Layout,
        SFV_smem_layout_staged: cute.Layout,
        SFDO_smem_layout_staged: cute.Layout,
        K_smem_layout_staged: cute.ComposedLayout,
        KT_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        tile_sched_params: Union[utils.ClcDynamicPersistentTileSchedulerParams, None],
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # For 2-CTA MMA: determine which CTA in the pair (0 or 1)
        use_2cta_instrs = cute.size(QK_tiled_mma.thr_id.shape) == 2
        mma_tile_coord_v = bidx % cute.size(QK_tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Get CTA rank in cluster for 2-CTA coordination
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full (for 2-CTA support)
        #
        self.a_full_mcast_mask = None
        self.b_full_mcast_mask = None
        self.sfa_full_mcast_mask = None
        self.sfb_full_mcast_mask = None
        if cutlass.const_expr(use_2cta_instrs):
            self.a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            self.b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            self.sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            self.sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_KT)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)
            prims.prefetch_tensormap(sfq_tmap.get_ptr())
            prims.prefetch_tensormap(sfk_tmap.get_ptr())
            prims.prefetch_tensormap(sfkt_tmap.get_ptr())
            prims.prefetch_tensormap(sfv_tmap.get_ptr())
            prims.prefetch_tensormap(sfdo_tmap.get_ptr())

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_K_pipeline = self.make_and_init_load_mma_K_pipeline(
            storage.load_mma_K_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )
        # Native SF path: the stage-full barrier also collects one arrival per
        # CTA's load warp (SF slot atoms built), and each CTA tracks its own
        # canonical SF bytes on sf_landed.  Same warp / lane as the
        # DSL's own init, before the cluster init fence.
        sf_landed_mbar_ptr = storage.sf_landed_mbar_ptr.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute_common.sf_reinit_full_barriers(load_mma_K_pipeline, self.load_mma_K_stage, 1 + 2)
                for stage_idx in cutlass.range_constexpr(self.load_mma_K_stage):
                    cute.arch.mbarrier_init(sf_landed_mbar_ptr + stage_idx, 1)

        # Per-warp pipelines for MMA warp 0
        mma_compute_S_pipeline_0 = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr_0.data_ptr(),
            cluster_layout_vmnk,
        )
        mma_compute_dP_pipeline_0 = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr_0.data_ptr(),
            cluster_layout_vmnk,
        )
        compute_mma_dS_pipeline_0 = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr_0.data_ptr(),
            cluster_layout_vmnk,
        )
        mma_compute_dQ_pipeline_0 = self.make_and_init_mma_compute_dQ_pipeline(
            storage.mma_compute_dQ_mbar_ptr_0.data_ptr(),
            cluster_layout_vmnk,
        )
        # Per-warp pipelines for MMA warp 1
        mma_compute_S_pipeline_1 = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr_1.data_ptr(),
            cluster_layout_vmnk,
        )
        mma_compute_dP_pipeline_1 = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr_1.data_ptr(),
            cluster_layout_vmnk,
        )
        compute_mma_dS_pipeline_1 = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr_1.data_ptr(),
            cluster_layout_vmnk,
        )
        mma_compute_dQ_pipeline_1 = self.make_and_init_mma_compute_dQ_pipeline(
            storage.mma_compute_dQ_mbar_ptr_1.data_ptr(),
            cluster_layout_vmnk,
        )

        # CLC dynamic persistent scheduling pipeline (persistent mode only)
        if cutlass.const_expr(self.is_persistent):
            cluster_size = cute.size(cluster_layout_vmnk)
            # Consumer threads: sched warp on CTA 0 (1 warp) +
            # all other warps on both CTAs (cluster_size * 11 warps: load + 2 mma + 8 compute)
            num_clc_consumer_threads = 32 * (
                1 + cluster_size * (1 + 2 + self.num_compute_0_warps + self.num_compute_1_warps)  # load  # mma0 + mma1  # compute0  # compute1
            )
            clc_pipeline = d256_primitives.make_clc_fetch_pipeline(
                storage.clc_mbar_ptr.data_ptr(),
                self.num_clc_stage,
                cluster_layout_vmnk,
                num_clc_consumer_threads,
                self.num_clc_response_bytes,
            )
            clc_response_ptr = storage.clc_response_ptr.data_ptr()
            clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_clc_stage)

        # Cluster arrive after barrier init for 2-CTA coordination
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        self.cta_sync_barrier.arrive_and_wait()

        # setup mma
        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sKT = storage.sKT.get_tensor(KT_smem_layout_staged.outer, swizzle=KT_smem_layout_staged.inner)
        sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        sSFQ = storage.sSFQ.get_tensor(
            sSFQ_smem_layout_staged,
        )
        sSFK = storage.sSFK.get_tensor(
            sSFK_smem_layout_staged,
        )

        sSFK_mn = storage.sSFK_mn.get_tensor(
            sfK_mn_smem_layout_staged,
        )

        sSFV = storage.sSFV.get_tensor(
            SFV_smem_layout_staged,
        )
        sSFDO = storage.sSFDO.get_tensor(
            SFDO_smem_layout_staged,
        )
        sDS = storage.sDS.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
        # Each compute WG writes one private packed 512-byte scale tile.
        # The MMA warp consumes it through the matching dS pipeline stage.
        sDS_scale_storage = storage.sDS_scale_exchange.get_tensor(cute.make_layout(1024))
        sDS_scale_exchange_0 = cute.make_tensor(
            sDS_scale_storage.iterator,
            cute.make_layout((1, 4, 64), stride=(256, 1, 4)),
        )
        sDS_scale_exchange_1 = cute.make_tensor(
            sDS_scale_storage.iterator + 512,
            cute.make_layout((1, 4, 64), stride=(256, 1, 4)),
        )

        # sKT_ptr = cute.recast_ptr(sK.iterator, KT_smem_layout_staged.inner)
        # sKT = cute.make_tensor(sKT_ptr, KT_smem_layout_staged.outer)

        # (MMA, MMA_M, MMA_K, STAGE)
        tSTrK = QK_tiled_mma.make_fragment_B(sK)

        # # (MMA, MMA_N, MMA_K, STAGE)
        # tDPrDO = DOV_tiled_mma.make_fragment_A(sdO)
        # (MMA, MMA_M, MMA_K, STAGE)
        tdPTrV = DOV_tiled_mma.make_fragment_B(sV)

        # (MMA, MMA_M, MMA_K, STAGE)
        tDQrDS = dSK_tiled_mma.make_fragment_A(sDS)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdQrKT = dSK_tiled_mma.make_fragment_B(sKT)

        tmem_holding_buf = storage.tmem_holding_buf
        # Create TmemAllocator for 2-CTA support
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id_0[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        if warp_idx == self.compute_warp_id_0[0]:
            # tmem_alloc_cols = cutlass.Int32(self.tmem_alloc_cols)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
        # The allocator warp publishes the TMEM base through shared memory.
        # Its allocator barrier does not order retrieval by the other warps.
        self.cta_sync_barrier.arrive_and_wait()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        tmem_offset = 0
        tDPtDP, tmem_offset, _ = cute_common.reserve_tmem_mma_fragment(
            tmem_ptr,
            tmem_offset,
            DOV_tiled_mma,
            self.DOV_mma_tiler,
            self.tmem_dP_stages,
            self.acc_dtype,
        )
        tStS, tmem_offset, _ = cute_common.reserve_tmem_mma_fragment(
            tmem_ptr,
            tmem_offset,
            QK_tiled_mma,
            self.QK_mma_tiler,
            self.tmem_S_stages,
            self.acc_dtype,
        )
        # WG1 reuses tStS stage 0 as its dP slot after WG0 has consumed S0.
        tDPtDP_1_alias = cute.make_tensor(tStS.iterator, tDPtDP.layout)

        SFQ_mma_tiler_sfa_tmem = (
            self.QK_mma_tiler[0],
            self.QK_mma_tiler[1],
            128,
        )
        SFQ_smem_layout_tmem_staged = blockscaled_utils.make_smem_layout_sfa(
            QK_tiled_mma_sfa,
            self.QK_mma_tiler_sfa_load,
            self.sf_vec_size,
            2,
        )
        SFQ_smem_layout_tmem_staged = cute_common.expand_last_SF_stride(SFQ_smem_layout_tmem_staged)
        SFQ_smem_layout_tmem = cute.slice_(SFQ_smem_layout_tmem_staged, (None, None, None, 0, 0))
        tSTtSFQ_layout = blockscaled_utils.make_tmem_layout_sfa(
            QK_tiled_mma,
            SFQ_mma_tiler_sfa_tmem,
            self.sf_vec_size,
            SFQ_smem_layout_tmem,
        )
        tSTtSFQ_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFQ_layout, self.sf_dtype)
        tSTtSFQ_1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFQ_layout, self.sf_dtype)
        tSTtSFQ_h1_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFQ_layout, self.sf_dtype)
        tSTtSFQ_h1_1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFQ_layout, self.sf_dtype)

        SFK_mma_tiler_sfb_tmem = (
            self.QK_mma_tiler_sfb[0],
            self.QK_mma_tiler_sfb[1],
            128 if self.SFK_halves > 1 else self.QK_mma_tiler_sfb[2],
        )
        SFK_smem_layout_tmem_staged = blockscaled_utils.make_smem_layout_sfb(
            QK_tiled_mma_sfb,
            SFK_mma_tiler_sfb_tmem,
            self.sf_vec_size,
            self.load_mma_K_stage,
        )
        SFK_smem_layout_tmem_staged = cute_common.expand_last_SF_stride(SFK_smem_layout_tmem_staged)
        SFK_smem_layout_tmem = cute.slice_(SFK_smem_layout_tmem_staged, (None, None, None, 0, 0))
        tSTtSFK_layout = blockscaled_utils.make_tmem_layout_sfb(
            QK_tiled_mma_sfb,
            SFK_mma_tiler_sfb_tmem,
            self.sf_vec_size,
            SFK_smem_layout_tmem,
        )
        tSTtSFK_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFK_layout, self.sf_dtype)
        tSTtSFK_1 = tSTtSFK_0
        if cutlass.const_expr(self.SFK_halves > 1):
            tSTtSFK_h1_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFK_layout, self.sf_dtype)
        else:
            tSTtSFK_h1_0 = tSTtSFK_0
        tSTtSFK_h1_1 = tSTtSFK_h1_0

        tDPtSFDO_layout = blockscaled_utils.make_tmem_layout_sfa(
            DOV_tiled_mma,
            (self.DOV_mma_tiler[0], self.DOV_mma_tiler[1], 128),
            self.sf_vec_size,
            cute.slice_(SFDO_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDPtSFDO_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFDO_layout, self.sf_dtype)
        tDPtSFDO_h1_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFDO_layout, self.sf_dtype)
        tDPtSFDO_1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFDO_layout, self.sf_dtype)
        tDPtSFDO_h1_1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFDO_layout, self.sf_dtype)

        SFV_mma_tiler_sfb_tmem = (
            self.DOV_mma_tiler_sfb[0],
            self.DOV_mma_tiler_sfb[1],
            128,
        )
        SFV_smem_layout_tmem_staged = blockscaled_utils.make_smem_layout_sfb(
            DOV_tiled_mma_sfb,
            SFV_mma_tiler_sfb_tmem,
            self.sf_vec_size,
            self.load_mma_K_stage,
        )
        SFV_smem_layout_tmem_staged = cute_common.expand_last_SF_stride(SFV_smem_layout_tmem_staged)
        SFV_smem_layout_tmem = cute.slice_(SFV_smem_layout_tmem_staged, (None, None, None, 0, 0))
        tDPtSFV_layout = blockscaled_utils.make_tmem_layout_sfb(
            DOV_tiled_mma_sfb,
            SFV_mma_tiler_sfb_tmem,
            self.sf_vec_size,
            SFV_smem_layout_tmem,
        )
        tDPtSFV_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFV_layout, self.sf_dtype)
        tDPtSFV_s2t_0 = tDPtSFV_0
        tDPtSFV_1 = tDPtSFV_0
        tDPtSFV_s2t_1 = tDPtSFV_s2t_0
        tDPtSFV_h1_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFV_layout, self.sf_dtype)
        tDPtSFV_s2t_h1_0 = tDPtSFV_h1_0
        tDPtSFV_h1_1 = tDPtSFV_h1_0
        tDPtSFV_s2t_h1_1 = tDPtSFV_s2t_h1_0

        SFDST_layout = blockscaled_utils.make_tmem_layout_sfa(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            self.sf_vec_size,
            cute.slice_(SFDO_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDQtSFDS_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, SFDST_layout, self.sf_dtype)
        tDQtSFDS_1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, SFDST_layout, self.sf_dtype)

        tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(cute.select(self.dSK_mma_tiler, mode=[0, 1]))
        tdQtdQ_tmp = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
        tdQtdQ_0, tmem_offset, _ = cute_common.reserve_tmem_fragment(
            tdQtdQ_tmp.iterator,
            tmem_offset,
            tdQtdQ_tmp,
            self.acc_dtype,
        )
        tdQtdQ_0_h1 = tdQtdQ_0
        tdQtdQ_1, tmem_offset, _ = cute_common.reserve_tmem_fragment(
            tdQtdQ_tmp.iterator,
            tmem_offset,
            tdQtdQ_tmp,
            self.acc_dtype,
        )
        tdQtdQ_1_h1 = tdQtdQ_1

        SFK_mn_layout = blockscaled_utils.make_tmem_layout_sfb(
            dSK_tiled_mma_sfb,
            self.dSK_mma_tiler_sfb,
            self.sf_vec_size,
            cute.slice_(sfK_mn_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tdQtSFK_mn_0, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, SFK_mn_layout, self.sf_dtype)
        tdQtSFK_mn_1 = tdQtSFK_mn_0

        # tDPrDO: smem-backed (OperandSource.SMEM for dO)
        tDPrDO = DOV_tiled_mma.make_fragment_A(sdO)

        if cutlass.const_expr(self.is_persistent):
            # ===================================================================
            #  PERSISTENT MODE: CLC dynamic tile scheduling
            # ===================================================================
            # Cluster wait before starting persistent work
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            # Create CLC tile scheduler
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()
            is_first_cta_in_cluster = cta_rank_in_cluster == 0

            # ///////////////////////////////////////////////////////////////////////////////
            #  LOAD warp - persistent loop
            # ///////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
                cumulative_trip_count_load = Int32(0)
                while work_tile.is_valid_tile:
                    cur_tile = work_tile.tile_idx
                    bidx_v = cur_tile[0]
                    bidy_v = cur_tile[1]
                    bidz_v = cur_tile[2]
                    num_h_r = problem_shape[3][0][0]
                    blk_coord_h_r = bidy_v % num_h_r
                    blk_coord_h_k = bidy_v // num_h_r
                    # Compute tile parameters from virtual coordinates
                    blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
                    problem_shape_cur_batch = problem_shape
                    if cutlass.const_expr(self.varlen):
                        Q_len_cur_batch = cumulative_s_q[bidz_v + 1] - cumulative_s_q[bidz_v]
                        K_len_cur_batch = cumulative_s_k[bidz_v + 1] - cumulative_s_k[bidz_v]
                        problem_shape_cur_batch = (
                            Q_len_cur_batch,
                            K_len_cur_batch,
                            problem_shape[2],
                            problem_shape[3],
                        )
                        blk_offset = (
                            cumulative_s_q[bidz_v],
                            cumulative_s_k[bidz_v],
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )

                    blk_q_warp0 = (bidx_v // 2) * 4 + (bidx_v % 2)
                    blk_q_warp1 = (bidx_v // 2) * 4 + 2 + (bidx_v % 2)
                    blk_coord_mask_0 = (blk_q_warp0, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                    blk_coord_mask_1 = (blk_q_warp1, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))

                    trip_start_0 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_0 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start_1 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_1 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start = cutlass.min(trip_start_0, trip_start_1)
                    trip_end_0 = trip_start_0 + trip_count_0
                    trip_end_1 = trip_start_1 + trip_count_1
                    trip_end = cutlass.max(trip_end_0, trip_end_1)
                    trip_count = trip_end - trip_start

                    cluster_idx = bidx_v // self.cluster_shape_mn[0]
                    has_work = cluster_idx * self.CTA_shape_Q < problem_shape_cur_batch[0] and trip_count > 0
                    if has_work:
                        self.load(
                            K_in,
                            KT_in,
                            V_in,
                            Q_in,
                            dO_in,
                            sfk_tmap,
                            sfkt_tmap,
                            sfq_tmap,
                            sfv_tmap,
                            sfdo_tmap,
                            sK,
                            sKT,
                            sQ,
                            sV,
                            sdO,
                            sSFK,
                            sSFK_mn,
                            sSFQ,
                            sSFV,
                            sSFDO,
                            QK_tiled_mma,
                            DOV_tiled_mma,
                            dSK_tiled_mma,
                            QK_tiled_mma_sfb,
                            DOV_tiled_mma_sfb,
                            dSK_tiled_mma_sfb,
                            QK_tiled_mma_sfa,
                            DOV_tiled_mma_sfa,
                            tma_atom_K,
                            tma_atom_KT,
                            tma_atom_Q,
                            tma_atom_V,
                            tma_atom_dO,
                            blk_offset,
                            problem_shape_cur_batch,
                            trip_count,
                            trip_start,
                            trip_end,
                            mma_tile_coord_v,
                            block_in_cluster_coord_vmnk,
                            cluster_layout_vmnk,
                            (load_mma_K_pipeline,),
                            sf_landed_mbar_ptr,
                            bidx_v,
                            blk_coord_h_r,
                            blk_coord_h_k,
                            bidz_v,
                            problem_shape[3][0][1],  # h_k
                            cumulative_trip_count=cumulative_trip_count_load,
                        )

                        cumulative_trip_count_load = cumulative_trip_count_load + trip_count
                    # Sync all non-sched warps before advancing to next persistent tile
                    self.persistent_tile_barrier.arrive_and_wait()
                    # CLC consumer: advance to next tile
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

            # ///////////////////////////////////////////////////////////////////////////////
            #  SCHED warp (CTA 0 only) - CLC producer loop
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx == self.sched_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
                if is_first_cta_in_cluster:
                    clc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage)
                    while work_tile.is_valid_tile:
                        clc_pipeline.producer_acquire(clc_producer_state)
                        mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                        tile_sched.advance_to_next_work(mbarrier_addr)
                        clc_producer_state.advance()

                        clc_pipeline.consumer_wait(clc_consumer_state)
                        work_tile = tile_sched.get_current_work()
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()
                    clc_pipeline.producer_tail(clc_producer_state)
                # CTA 1's sched warp: not in consumer group, does nothing

            # ///////////////////////////////////////////////////////////////////////////////
            #  MMA warp 0 (warp 8) - persistent loop
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx == self.mma_warp_id_0:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)
                persistent_iter = Int32(0)
                cumulative_trip_count_mma0 = Int32(0)
                while work_tile.is_valid_tile:
                    cur_tile = work_tile.tile_idx
                    bidx_v = cur_tile[0]
                    bidy_v = cur_tile[1]
                    bidz_v = cur_tile[2]
                    num_h_r = problem_shape[3][0][0]
                    blk_coord_h_r = bidy_v % num_h_r
                    blk_coord_h_k = bidy_v // num_h_r

                    blk_coord_mask_0 = ((bidx_v // 2) * 4 + (bidx_v % 2), Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                    problem_shape_cur_batch = problem_shape
                    if cutlass.const_expr(self.varlen):
                        Q_len_cur_batch = cumulative_s_q[bidz_v + 1] - cumulative_s_q[bidz_v]
                        K_len_cur_batch = cumulative_s_k[bidz_v + 1] - cumulative_s_k[bidz_v]
                        problem_shape_cur_batch = (
                            Q_len_cur_batch,
                            K_len_cur_batch,
                            problem_shape[2],
                            problem_shape[3],
                        )

                    blk_q_warp0 = (bidx_v // 2) * 4 + (bidx_v % 2)
                    blk_q_warp1 = (bidx_v // 2) * 4 + 2 + (bidx_v % 2)
                    blk_coord_mask_0 = (blk_q_warp0, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                    blk_coord_mask_1 = (blk_q_warp1, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))

                    trip_start_0 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_0 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start_1 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_1 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start = cutlass.min(trip_start_0, trip_start_1)
                    trip_end_0 = trip_start_0 + trip_count_0
                    trip_end_1 = trip_start_1 + trip_count_1
                    trip_end = cutlass.max(trip_end_0, trip_end_1)
                    trip_count = trip_end - trip_start

                    cluster_idx = bidx_v // self.cluster_shape_mn[0]
                    has_work = cluster_idx * self.CTA_shape_Q < problem_shape_cur_batch[0] and trip_count > 0
                    if has_work:
                        # Create dQ producer states with correct phase for the
                        # per-tile store-once mbarriers.
                        mma_compute_dKdV_producer_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage)
                        mma_compute_dKdV_producer_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage)
                        if persistent_iter % Int32(2) == Int32(1):
                            mma_compute_dKdV_producer_state_0.advance()
                            mma_compute_dKdV_producer_state_1.advance()

                        self.mma_interleaved(
                            tmem=tmem,
                            QK_tiled_mma=QK_tiled_mma,
                            DOV_tiled_mma=DOV_tiled_mma,
                            dSK_tiled_mma=dSK_tiled_mma,
                            QK_tiled_mma_smem=QK_tiled_mma_smem,
                            tStS_0=tStS[None, None, None, 0],
                            tStS_1=tStS[None, None, None, self.tmem_S_stages - 1],
                            tSTrK=tSTrK,
                            tSTtSFQ_0=tSTtSFQ_0,
                            tSTtSFQ_h1_0=tSTtSFQ_h1_0,
                            tSTtSFQ_1=tSTtSFQ_1,
                            tSTtSFQ_h1_1=tSTtSFQ_h1_1,
                            tSTtSFK=tSTtSFK_0,
                            tSTtSFK_h1=tSTtSFK_h1_0,
                            tDPtSFV=tDPtSFV_0,
                            tDPtSFV_s2t=tDPtSFV_s2t_0,
                            tDPtSFV_h1=tDPtSFV_h1_0,
                            tDPtSFV_s2t_h1=tDPtSFV_s2t_h1_0,
                            tDPtSFDO_0=tDPtSFDO_0,
                            tDPtSFDO_h1_0=tDPtSFDO_h1_0,
                            tDPtSFDO_1=tDPtSFDO_1,
                            tDPtSFDO_h1_1=tDPtSFDO_h1_1,
                            tDPtDP_0=tDPtDP[None, None, None, 0],
                            tDPtDP_1=tDPtDP_1_alias[None, None, None, 0],
                            tdQtSFK_mn=tdQtSFK_mn_0,
                            tdPTrV=tdPTrV,
                            tDPrDO=tDPrDO,
                            tdQtdQ_0=tdQtdQ_0,
                            tdQtdQ_1=tdQtdQ_1,
                            tDQrDS=tDQrDS,
                            tdQrKT=tdQrKT,
                            tDQtSFDS_0=tDQtSFDS_0,
                            tDQtSFDS_1=tDQtSFDS_1,
                            sDS_scale_exchange_0=sDS_scale_exchange_0,
                            sDS_scale_exchange_1=sDS_scale_exchange_1,
                            iter_count=trip_count,
                            iter_start=trip_start,
                            iter_end=trip_end,
                            pipeline_args=(
                                load_mma_K_pipeline,
                                mma_compute_S_pipeline_0,
                                mma_compute_dP_pipeline_0,
                                compute_mma_dS_pipeline_0,
                                mma_compute_dQ_pipeline_0,
                                mma_compute_S_pipeline_1,
                                mma_compute_dP_pipeline_1,
                                compute_mma_dS_pipeline_1,
                                mma_compute_dQ_pipeline_1,
                            ),
                            mma_compute_dKdV_producer_state_0=mma_compute_dKdV_producer_state_0,
                            mma_compute_dKdV_producer_state_1=mma_compute_dKdV_producer_state_1,
                            sSFQ=sSFQ,
                            sSFK=sSFK,
                            sSFK_mn=sSFK_mn,
                            sSFV=sSFV,
                            sSFDO=sSFDO,
                            sQ=sQ,
                            sK=sK,
                            sV=sV,
                            sDO=sdO,
                            sKT=sKT,
                            cumulative_trip_count=cumulative_trip_count_mma0,
                        )

                        cumulative_trip_count_mma0 = cumulative_trip_count_mma0 + trip_count
                        persistent_iter = persistent_iter + Int32(1)
                    # Sync all non-sched warps before advancing to next persistent tile
                    self.persistent_tile_barrier.arrive_and_wait()
                    # CLC consumer: advance to next tile
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

            # ///////////////////////////////////////////////////////////////////////////////
            #  SF helper warp (warp 9) - persistent loop
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx == self.sfv_s2t_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
                cumulative_trip_count_sfv = Int32(0)
                while work_tile.is_valid_tile:
                    cur_tile = work_tile.tile_idx
                    bidx_v = cur_tile[0]
                    bidy_v = cur_tile[1]
                    bidz_v = cur_tile[2]
                    num_h_r = problem_shape[3][0][0]
                    blk_coord_h_r = bidy_v % num_h_r
                    blk_coord_h_k = bidy_v // num_h_r

                    problem_shape_cur_batch = problem_shape
                    if cutlass.const_expr(self.varlen):
                        Q_len_cur_batch = cumulative_s_q[bidz_v + 1] - cumulative_s_q[bidz_v]
                        K_len_cur_batch = cumulative_s_k[bidz_v + 1] - cumulative_s_k[bidz_v]
                        problem_shape_cur_batch = (
                            Q_len_cur_batch,
                            K_len_cur_batch,
                            problem_shape[2],
                            problem_shape[3],
                        )

                    blk_q_warp0 = (bidx_v // 2) * 4 + (bidx_v % 2)
                    blk_q_warp1 = (bidx_v // 2) * 4 + 2 + (bidx_v % 2)
                    blk_coord_mask_0 = (blk_q_warp0, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                    blk_coord_mask_1 = (blk_q_warp1, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))

                    trip_start_0 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_0 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start_1 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_1 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start = cutlass.min(trip_start_0, trip_start_1)
                    trip_end_0 = trip_start_0 + trip_count_0
                    trip_end_1 = trip_start_1 + trip_count_1
                    trip_end = cutlass.max(trip_end_0, trip_end_1)
                    trip_count = trip_end - trip_start

                    cluster_idx = bidx_v // self.cluster_shape_mn[0]
                    has_work = cluster_idx * self.CTA_shape_Q < problem_shape_cur_batch[0] and trip_count > 0
                    if has_work:
                        self.sfv_s2t_helper(
                            tDPtSFV_s2t=tDPtSFV_s2t_0,
                            tDPtSFV_s2t_h1=tDPtSFV_s2t_h1_0,
                            tDPtSFDO_0=tDPtSFDO_0,
                            tDPtSFDO_h1_0=tDPtSFDO_h1_0,
                            tDPtSFDO_1=tDPtSFDO_1,
                            tDPtSFDO_h1_1=tDPtSFDO_h1_1,
                            tSTtSFK=tSTtSFK_0,
                            tSTtSFK_h1=tSTtSFK_h1_0,
                            iter_count=trip_count,
                            pipeline_args=(load_mma_K_pipeline,),
                            is_leader_cta=is_leader_cta,
                            sSFK=sSFK,
                            sSFV=sSFV,
                            sSFDO=sSFDO,
                            cumulative_trip_count=cumulative_trip_count_sfv,
                        )

                        cumulative_trip_count_sfv = cumulative_trip_count_sfv + trip_count
                    # Sync all non-sched warps before advancing to next persistent tile
                    self.persistent_tile_barrier.arrive_and_wait()
                    # CLC consumer: advance to next tile
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

            # ///////////////////////////////////////////////////////////////////////////////
            #  Compute warps - persistent loop
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx >= self.compute_warp_id_0[0] and warp_idx <= self.compute_warp_id_1[-1]:
                cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
                persistent_iter = Int32(0)
                cumulative_trip_count_compute = Int32(0)
                while work_tile.is_valid_tile:
                    cur_tile = work_tile.tile_idx
                    bidx_v = cur_tile[0]
                    bidy_v = cur_tile[1]
                    bidz_v = cur_tile[2]
                    num_h_r = problem_shape[3][0][0]
                    blk_coord_h_r = bidy_v % num_h_r
                    blk_coord_h_k = bidy_v // num_h_r

                    problem_shape_cur_batch = problem_shape
                    blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
                    if cutlass.const_expr(self.varlen):
                        Q_len_cur_batch = cumulative_s_q[bidz_v + 1] - cumulative_s_q[bidz_v]
                        K_len_cur_batch = cumulative_s_k[bidz_v + 1] - cumulative_s_k[bidz_v]
                        problem_shape_cur_batch = (
                            Q_len_cur_batch,
                            K_len_cur_batch,
                            problem_shape[2],
                            problem_shape[3],
                        )
                        blk_offset = (
                            cumulative_s_q[bidz_v],
                            cumulative_s_k[bidz_v],
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )

                    blk_coord = (bidx_v, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                    blk_q_warp0 = (bidx_v // 2) * 4 + (bidx_v % 2)
                    blk_q_warp1 = (bidx_v // 2) * 4 + 2 + (bidx_v % 2)
                    blk_coord_mask_0 = (blk_q_warp0, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                    blk_coord_mask_1 = (blk_q_warp1, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))

                    trip_start_0 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_0 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_0,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start_1 = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count_1 = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord_mask_1,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_start = cutlass.min(trip_start_0, trip_start_1)
                    trip_end_0 = trip_start_0 + trip_count_0
                    trip_end_1 = trip_start_1 + trip_count_1
                    trip_end = cutlass.max(trip_end_0, trip_end_1)
                    trip_count = trip_end - trip_start

                    cluster_idx = bidx_v // self.cluster_shape_mn[0]
                    has_work = cluster_idx * self.CTA_shape_Q < problem_shape_cur_batch[0] and trip_count > 0
                    if has_work:
                        # Compute group 0 (warps 0-3)
                        if warp_idx >= self.compute_warp_id_0[0] and warp_idx <= self.compute_warp_id_0[-1]:
                            self.compute(
                                tStS=tStS[None, None, None, 0],
                                tDPtDP=tDPtDP[None, None, None, 0],
                                tDQtSFDS=tDQtSFDS_0,
                                sDS=sDS,
                                sDS_scale_exchange=sDS_scale_exchange_0,
                                blk_coord=blk_coord,
                                blk_coord_mask=blk_coord_mask_0,
                                problem_shape=problem_shape_cur_batch,
                                iter_count=trip_count,
                                iter_start=trip_start,
                                iter_end=trip_end,
                                trip_start_mask=trip_start_0,
                                scale_softmax=scale_softmax,
                                window_size_left=window_size_left,
                                window_size_right=window_size_right,
                                is_leader_cta=is_leader_cta,
                                LSE=LSE,
                                sum_OdO=sum_OdO,
                                pipeline_args=(
                                    mma_compute_S_pipeline_0,
                                    mma_compute_dP_pipeline_0,
                                    compute_mma_dS_pipeline_0,
                                ),
                                wg_idx=0,
                                dS_stage_offset=0,
                                trip_count_mask=trip_count_0,
                                cumulative_trip_count=cumulative_trip_count_compute,
                            )
                            mma_compute_Q_consumer_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage)
                            if persistent_iter % Int32(2) == Int32(1):
                                mma_compute_Q_consumer_state_0.advance()
                            blk_coord_epilogue_0 = (blk_q_warp0, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                            self.epilogue(
                                blk_coord_epilogue_0,
                                blk_offset,
                                problem_shape_cur_batch,
                                dQ,
                                tdQtdQ_0,
                                scale_softmax,
                                (mma_compute_dQ_pipeline_0, mma_compute_Q_consumer_state_0),
                            )
                            self.epilogue_sync_barrier.arrive_and_wait()
                        # Compute group 1 (warps 4-7)
                        elif warp_idx >= self.compute_warp_id_1[0] and warp_idx <= self.compute_warp_id_1[-1]:
                            self.compute(
                                tStS=tStS[None, None, None, self.tmem_S_stages - 1],
                                tDPtDP=tDPtDP_1_alias[None, None, None, 0],
                                tDQtSFDS=tDQtSFDS_1,
                                sDS=sDS,
                                sDS_scale_exchange=sDS_scale_exchange_1,
                                blk_coord=blk_coord,
                                blk_coord_mask=blk_coord_mask_1,
                                problem_shape=problem_shape_cur_batch,
                                iter_count=trip_count,
                                iter_start=trip_start,
                                iter_end=trip_end,
                                trip_start_mask=trip_start_1,
                                scale_softmax=scale_softmax,
                                window_size_left=window_size_left,
                                window_size_right=window_size_right,
                                is_leader_cta=is_leader_cta,
                                LSE=LSE,
                                sum_OdO=sum_OdO,
                                pipeline_args=(
                                    mma_compute_S_pipeline_1,
                                    mma_compute_dP_pipeline_1,
                                    compute_mma_dS_pipeline_1,
                                ),
                                wg_idx=1,
                                dS_stage_offset=self.compute_mma_dS_stage,
                                trip_count_mask=trip_count_1,
                                cumulative_trip_count=cumulative_trip_count_compute,
                            )
                            mma_compute_Q_consumer_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage)
                            if persistent_iter % Int32(2) == Int32(1):
                                mma_compute_Q_consumer_state_1.advance()
                            blk_coord_epilogue_1 = (blk_q_warp1, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz_v))
                            self.epilogue(
                                blk_coord_epilogue_1,
                                blk_offset,
                                problem_shape_cur_batch,
                                dQ,
                                tdQtdQ_1,
                                scale_softmax,
                                (mma_compute_dQ_pipeline_1, mma_compute_Q_consumer_state_1),
                            )
                            self.epilogue_sync_barrier_1.arrive_and_wait()
                        cumulative_trip_count_compute = cumulative_trip_count_compute + trip_count
                        persistent_iter = persistent_iter + Int32(1)
                    # Sync all non-sched warps before advancing to next persistent tile
                    self.persistent_tile_barrier.arrive_and_wait()
                    # CLC consumer: advance to next tile
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

        else:
            # ===================================================================
            #  NON-PERSISTENT MODE (original code)
            # ===================================================================
            # get the current batch problem shape
            num_h_r = problem_shape[3][0][0]
            blk_coord_h_r = bidy % num_h_r
            blk_coord_h_k = bidy // num_h_r
            blk_coord = (bidx, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz))
            problem_shape_cur_batch = problem_shape
            blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
            if cutlass.const_expr(self.varlen):
                Q_len_cur_batch = cumulative_s_q[bidz + 1] - cumulative_s_q[bidz]
                K_len_cur_batch = cumulative_s_k[bidz + 1] - cumulative_s_k[bidz]
                problem_shape_cur_batch = (
                    Q_len_cur_batch,
                    K_len_cur_batch,
                    problem_shape[2],
                    problem_shape[3],
                )
                blk_offset = (
                    cumulative_s_q[bidz],
                    cumulative_s_k[bidz],
                    Int32(0),
                    ((Int32(0), Int32(0)), Int32(0)),
                )

            blk_q_warp0 = (bidx // 2) * 4 + (bidx % 2)
            blk_q_warp1 = (bidx // 2) * 4 + 2 + (bidx % 2)
            blk_coord_mask_0 = (blk_q_warp0, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz))
            blk_coord_mask_1 = (blk_q_warp1, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz))

            trip_start_0 = fmha_masks.FusedMask.get_trip_start(
                self.mask_type,
                blk_coord_mask_0,
                self.mask_cta_tiler,
                problem_shape_cur_batch[0],
                problem_shape_cur_batch[1],
                window_size_left,
                window_size_right,
            )
            trip_count_0 = fmha_masks.FusedMask.get_trip_count(
                self.mask_type,
                blk_coord_mask_0,
                self.mask_cta_tiler,
                problem_shape_cur_batch[0],
                problem_shape_cur_batch[1],
                window_size_left,
                window_size_right,
            )
            trip_start_1 = fmha_masks.FusedMask.get_trip_start(
                self.mask_type,
                blk_coord_mask_1,
                self.mask_cta_tiler,
                problem_shape_cur_batch[0],
                problem_shape_cur_batch[1],
                window_size_left,
                window_size_right,
            )
            trip_count_1 = fmha_masks.FusedMask.get_trip_count(
                self.mask_type,
                blk_coord_mask_1,
                self.mask_cta_tiler,
                problem_shape_cur_batch[0],
                problem_shape_cur_batch[1],
                window_size_left,
                window_size_right,
            )
            trip_start = cutlass.min(trip_start_0, trip_start_1)
            trip_end_0 = trip_start_0 + trip_count_0
            trip_end_1 = trip_start_1 + trip_count_1
            trip_end = cutlass.max(trip_end_0, trip_end_1)
            trip_count = trip_end - trip_start

            # Cluster wait before tensor memory alloc for 2-CTA
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            # Check at cluster level: both CTAs in a cluster must enter if the cluster has work
            cluster_idx = bidx // self.cluster_shape_mn[0]
            if cluster_idx * self.CTA_shape_Q < problem_shape_cur_batch[0] and trip_count > 0:
                # ///////////////////////////////////////////////////////////////////////////////
                #  LOAD (warp 10)
                # ///////////////////////////////////////////////////////////////////////////////
                if warp_idx == self.load_warp_id:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

                    self.load(
                        K_in,
                        KT_in,
                        V_in,
                        Q_in,
                        dO_in,
                        sfk_tmap,
                        sfkt_tmap,
                        sfq_tmap,
                        sfv_tmap,
                        sfdo_tmap,
                        sK,
                        sKT,
                        sQ,
                        sV,
                        sdO,
                        sSFK,
                        sSFK_mn,
                        sSFQ,
                        sSFV,
                        sSFDO,
                        QK_tiled_mma,
                        DOV_tiled_mma,
                        dSK_tiled_mma,
                        QK_tiled_mma_sfb,
                        DOV_tiled_mma_sfb,
                        dSK_tiled_mma_sfb,
                        QK_tiled_mma_sfa,
                        DOV_tiled_mma_sfa,
                        tma_atom_K,
                        tma_atom_KT,
                        tma_atom_Q,
                        tma_atom_V,
                        tma_atom_dO,
                        blk_offset,
                        problem_shape_cur_batch,
                        trip_count,
                        trip_start,
                        trip_end,
                        mma_tile_coord_v,
                        block_in_cluster_coord_vmnk,
                        cluster_layout_vmnk,
                        (load_mma_K_pipeline,),
                        sf_landed_mbar_ptr,
                        bidx,
                        blk_coord_h_r,
                        blk_coord_h_k,
                        bidz,
                        problem_shape[3][0][1],  # h_k
                    )

                # ///////////////////////////////////////////////////////////////////////////////
                #  MMA warp 0 (warp 8) — K-tile interleaves Q sub-block 0 and 1
                # ///////////////////////////////////////////////////////////////////////////////
                elif warp_idx == self.mma_warp_id_0:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)
                    mma_compute_dKdV_producer_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage)
                    mma_compute_dKdV_producer_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage)
                    self.mma_interleaved(
                        tmem=tmem,
                        QK_tiled_mma=QK_tiled_mma,
                        DOV_tiled_mma=DOV_tiled_mma,
                        dSK_tiled_mma=dSK_tiled_mma,
                        QK_tiled_mma_smem=QK_tiled_mma_smem,
                        tStS_0=tStS[None, None, None, 0],
                        tStS_1=tStS[None, None, None, self.tmem_S_stages - 1],
                        tSTrK=tSTrK,
                        tSTtSFQ_0=tSTtSFQ_0,
                        tSTtSFQ_h1_0=tSTtSFQ_h1_0,
                        tSTtSFQ_1=tSTtSFQ_1,
                        tSTtSFQ_h1_1=tSTtSFQ_h1_1,
                        tSTtSFK=tSTtSFK_0,
                        tSTtSFK_h1=tSTtSFK_h1_0,
                        tDPtSFV=tDPtSFV_0,
                        tDPtSFV_s2t=tDPtSFV_s2t_0,
                        tDPtSFV_h1=tDPtSFV_h1_0,
                        tDPtSFV_s2t_h1=tDPtSFV_s2t_h1_0,
                        tDPtSFDO_0=tDPtSFDO_0,
                        tDPtSFDO_h1_0=tDPtSFDO_h1_0,
                        tDPtSFDO_1=tDPtSFDO_1,
                        tDPtSFDO_h1_1=tDPtSFDO_h1_1,
                        tDPtDP_0=tDPtDP[None, None, None, 0],
                        tDPtDP_1=tDPtDP_1_alias[None, None, None, 0],
                        tdQtSFK_mn=tdQtSFK_mn_0,
                        tdPTrV=tdPTrV,
                        tDPrDO=tDPrDO,
                        tdQtdQ_0=tdQtdQ_0,
                        tdQtdQ_1=tdQtdQ_1,
                        tDQrDS=tDQrDS,
                        tdQrKT=tdQrKT,
                        tDQtSFDS_0=tDQtSFDS_0,
                        tDQtSFDS_1=tDQtSFDS_1,
                        sDS_scale_exchange_0=sDS_scale_exchange_0,
                        sDS_scale_exchange_1=sDS_scale_exchange_1,
                        iter_count=trip_count,
                        iter_start=trip_start,
                        iter_end=trip_end,
                        pipeline_args=(
                            load_mma_K_pipeline,
                            mma_compute_S_pipeline_0,
                            mma_compute_dP_pipeline_0,
                            compute_mma_dS_pipeline_0,
                            mma_compute_dQ_pipeline_0,
                            mma_compute_S_pipeline_1,
                            mma_compute_dP_pipeline_1,
                            compute_mma_dS_pipeline_1,
                            mma_compute_dQ_pipeline_1,
                        ),
                        mma_compute_dKdV_producer_state_0=mma_compute_dKdV_producer_state_0,
                        mma_compute_dKdV_producer_state_1=mma_compute_dKdV_producer_state_1,
                        sSFQ=sSFQ,
                        sSFK=sSFK,
                        sSFK_mn=sSFK_mn,
                        sSFV=sSFV,
                        sSFDO=sSFDO,
                        sQ=sQ,
                        sK=sK,
                        sV=sV,
                        sDO=sdO,
                        sKT=sKT,
                    )

                elif warp_idx == self.sfv_s2t_warp_id:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
                    self.sfv_s2t_helper(
                        tDPtSFV_s2t=tDPtSFV_s2t_0,
                        tDPtSFV_s2t_h1=tDPtSFV_s2t_h1_0,
                        tDPtSFDO_0=tDPtSFDO_0,
                        tDPtSFDO_h1_0=tDPtSFDO_h1_0,
                        tDPtSFDO_1=tDPtSFDO_1,
                        tDPtSFDO_h1_1=tDPtSFDO_h1_1,
                        tSTtSFK=tSTtSFK_0,
                        tSTtSFK_h1=tSTtSFK_h1_0,
                        iter_count=trip_count,
                        pipeline_args=(load_mma_K_pipeline,),
                        is_leader_cta=is_leader_cta,
                        sSFK=sSFK,
                        sSFV=sSFV,
                        sSFDO=sSFDO,
                    )

                elif warp_idx == self.mma_warp_id_1:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

                # ///////////////////////////////////////////////////////////////////////////////
                #  Compute group 0 (warps 0-3) — handles MMA warp 0
                # ///////////////////////////////////////////////////////////////////////////////
                elif warp_idx >= self.compute_warp_id_0[0] and warp_idx <= self.compute_warp_id_0[-1]:
                    cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
                    self.compute(
                        tStS=tStS[None, None, None, 0],
                        tDPtDP=tDPtDP[None, None, None, 0],
                        tDQtSFDS=tDQtSFDS_0,
                        sDS=sDS,
                        sDS_scale_exchange=sDS_scale_exchange_0,
                        blk_coord=blk_coord,
                        blk_coord_mask=blk_coord_mask_0,
                        problem_shape=problem_shape_cur_batch,
                        iter_count=trip_count,
                        iter_start=trip_start,
                        iter_end=trip_end,
                        trip_start_mask=trip_start_0,
                        scale_softmax=scale_softmax,
                        window_size_left=window_size_left,
                        window_size_right=window_size_right,
                        is_leader_cta=is_leader_cta,
                        LSE=LSE,
                        sum_OdO=sum_OdO,
                        pipeline_args=(
                            mma_compute_S_pipeline_0,
                            mma_compute_dP_pipeline_0,
                            compute_mma_dS_pipeline_0,
                        ),
                        wg_idx=0,
                        dS_stage_offset=0,
                        trip_count_mask=trip_count_0,
                    )
                    mma_compute_Q_consumer_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage)
                    blk_coord_epilogue_0 = (blk_q_warp0, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz))
                    self.epilogue(
                        blk_coord_epilogue_0,
                        blk_offset,
                        problem_shape_cur_batch,
                        dQ,
                        tdQtdQ_0,
                        scale_softmax,
                        (mma_compute_dQ_pipeline_0, mma_compute_Q_consumer_state_0),
                    )
                    self.epilogue_sync_barrier.arrive_and_wait()
                # ///////////////////////////////////////////////////////////////////////////////
                #  Compute group 1 (warps 4-7) — handles Q sub-block 1
                # ///////////////////////////////////////////////////////////////////////////////
                elif warp_idx >= self.compute_warp_id_1[0] and warp_idx <= self.compute_warp_id_1[-1]:
                    cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
                    self.compute(
                        tStS=tStS[None, None, None, self.tmem_S_stages - 1],
                        tDPtDP=tDPtDP_1_alias[None, None, None, 0],
                        tDQtSFDS=tDQtSFDS_1,
                        sDS=sDS,
                        sDS_scale_exchange=sDS_scale_exchange_1,
                        blk_coord=blk_coord,
                        blk_coord_mask=blk_coord_mask_1,
                        problem_shape=problem_shape_cur_batch,
                        iter_count=trip_count,
                        iter_start=trip_start,
                        iter_end=trip_end,
                        trip_start_mask=trip_start_1,
                        scale_softmax=scale_softmax,
                        window_size_left=window_size_left,
                        window_size_right=window_size_right,
                        is_leader_cta=is_leader_cta,
                        LSE=LSE,
                        sum_OdO=sum_OdO,
                        pipeline_args=(
                            mma_compute_S_pipeline_1,
                            mma_compute_dP_pipeline_1,
                            compute_mma_dS_pipeline_1,
                        ),
                        wg_idx=1,
                        dS_stage_offset=self.compute_mma_dS_stage,
                        trip_count_mask=trip_count_1,
                    )
                    mma_compute_Q_consumer_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage)
                    blk_coord_epilogue_1 = (blk_q_warp1, Int32(0), Int32(0), ((blk_coord_h_r, blk_coord_h_k), bidz))
                    self.epilogue(
                        blk_coord_epilogue_1,
                        blk_offset,
                        problem_shape_cur_batch,
                        dQ,
                        tdQtdQ_1,
                        scale_softmax,
                        (mma_compute_dQ_pipeline_1, mma_compute_Q_consumer_state_1),
                    )
                    self.epilogue_sync_barrier_1.arrive_and_wait()
                else:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # Cluster-wide sync before TMEM deallocation: in persistent mode, one CTA
        # may exit its while loop and reach tmem.free() while the partner CTA's MMA
        # warp still accesses shared TMEM. Sync all warps within CTA first, then
        # sync across the 2-CTA cluster.
        if cutlass.const_expr(self.is_persistent):
            self.cta_sync_barrier.arrive_and_wait()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        if warp_idx == self.compute_warp_id_0[0]:
            # Dealloc the tensor memory buffer for 2-CTA support
            tmem.relinquish_alloc_permit()
            tmem._num_allocated_columns = self.tmem_alloc_cols
            tmem.free(tmem_ptr)

    @cute.jit
    def load(
        self,
        K_in: cute.Tensor,
        KT_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        dO_in: cute.Tensor,
        sfk_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfkt_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfq_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfv_tmap: cutlass.GridConstant[tmap.TensorMap],
        sfdo_tmap: cutlass.GridConstant[tmap.TensorMap],
        sK: cute.Tensor,
        sKT: cute.Tensor,
        sQ: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sSFK: cute.Tensor,
        sSFK_mn: cute.Tensor,
        sSFQ: cute.Tensor,
        sSFV: cute.Tensor,
        sSFDO: cute.Tensor,
        QK_tiled_mma: cute.TiledMma,
        DOV_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        QK_tiled_mma_sfb: cute.TiledMma,
        DOV_tiled_mma_sfb: cute.TiledMma,
        dSK_tiled_mma_sfb: cute.TiledMma,
        QK_tiled_mma_sfa: cute.TiledMma,
        DOV_tiled_mma_sfa: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_KT: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        mma_tile_coord_v: Int32,
        block_in_cluster_coord_vmnk,
        cluster_layout_vmnk,
        # (load_mma_Q_pipeline, load_compute_LSE_pipeline, load_mma_dO_pipeline, load_compute_sum_OdO_pipeline)
        pipeline_args: tuple,
        sf_landed_mbar_ptr: cute.Pointer,
        # Logical block coordinates (from block_idx() in non-persistent, from CLC in persistent)
        blk_coord_q: Int32,
        blk_coord_h_r: Int32,
        blk_coord_h_k: Int32,
        blk_coord_b: Int32,
        num_h_k: Int32,  # total number of KV heads (replaces grid_dim_y)
        cumulative_trip_count: Int32 = Int32(0),  # accumulated trip_count across persistent tiles
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        grid_dim_y = num_h_k

        blk_coord_h_q = (blk_coord_h_r, blk_coord_h_k)
        blk_coord_h_kv = (Int32(0), blk_coord_h_k)
        seq_Q, seq_K, D, HB = problem_shape
        H, B = HB
        iter_index = iter_start
        (load_mma_K_pipeline,) = pipeline_args

        K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
        KT = cute.domain_offset(cute.select(blk_offset, mode=[2, 1, 3]), KT_in)
        V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
        Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
        dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)

        # (bM, bK, RestM, RestK, (H, B))
        gQ = cute.local_tile(Q, cute.select(self.QK_mma_tiler, mode=[0, 2]), (None, None, None))

        # (bN, bK, RestN, RestK, (H, B))
        gK = cute.local_tile(K, cute.select(self.QK_mma_tiler, mode=[1, 2]), (None, None, None))

        gKT = cute.local_tile(KT, cute.select(self.dSK_mma_tiler, mode=[1, 2]), (None, None, None))

        # (bM, bK, RestM, RestK, (H, B))
        gdO = cute.local_tile(dO, cute.select(self.DOV_mma_tiler, mode=[0, 2]), (None, None, None))
        # (bN, bK, RestN, RestK, (H, B))
        gV = cute.local_tile(V, cute.select(self.DOV_mma_tiler, mode=[1, 2]), (None, None, None))

        QK_thr_mma = QK_tiled_mma.get_slice(mma_tile_coord_v)
        DOV_thr_mma = DOV_tiled_mma.get_slice(mma_tile_coord_v)
        dSK_thr_mma = dSK_tiled_mma.get_slice(mma_tile_coord_v)
        dSK_thr_mma_sfb = dSK_tiled_mma_sfb.get_slice(mma_tile_coord_v)
        QK_thr_mma_sfb = QK_tiled_mma_sfb.get_slice(0)
        DOV_thr_mma_sfb = DOV_tiled_mma_sfb.get_slice(0)
        dSK_thr_mma_sfb = dSK_thr_mma_sfb.get_slice(0)

        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tSgQ = QK_thr_mma.partition_A(gQ)
        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tSgK = QK_thr_mma.partition_B(gK)

        tSgKT = dSK_thr_mma.partition_B(gKT)

        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tdPgdO = DOV_thr_mma.partition_A(gdO)
        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tdPgV = DOV_thr_mma.partition_B(gV)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK, 0, 3),
        )
        tKTsKT, tKTgKT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_KT,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sKT, 0, 3),
            cute.group_modes(tSgKT, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))

        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPgV, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPgdO, 0, 3),
        )

        # Scale factors: canonical atoms land in slot 0 of each SF smem stage
        # (byte geometry of the staged (..., STAGE, slot) layouts); the SF helper
        # warp builds slot 1 (and the peer CTA's shifted A-operand atoms) in smem.
        sfk_stage_stride, sfk_slot_stride = cute_common.sf_slot_strides(sSFK.layout)
        sfkmn_stage_stride, sfkmn_slot_stride = cute_common.sf_slot_strides(sSFK_mn.layout)
        sfq_stage_stride, sfq_slot_stride = cute_common.sf_slot_strides(sSFQ.layout)
        sfv_stage_stride, sfv_slot_stride = cute_common.sf_slot_strides(sSFV.layout)
        sfdo_stage_stride, sfdo_slot_stride = cute_common.sf_slot_strides(sSFDO.layout)
        sfkmn_atoms_per_slot = sfkmn_slot_stride // cute_common.SF_ATOM_BYTES
        # Each CTA's load warp loads its own canonical atoms and, once they landed
        # (right after issuing the stage), builds its shifted slot atoms; one
        # arrival per stage on the LEADER's stage-full barrier (remote from the
        # peer) releases the stage, so the MMA warp's existing consumer_wait also
        # covers the slots.
        sf_lane = tidx % self.threads_per_warp
        is_leader_cta = mma_tile_coord_v == 0
        words_k = cute_common.smem_array(sSFK.iterator, Int32)
        words_v = cute_common.smem_array(sSFV.iterator, Int32)
        words_kmn = cute_common.smem_array(sSFK_mn.iterator, Int32)
        words_q = cute_common.smem_array(sSFQ.iterator, Int32)
        words_do = cute_common.smem_array(sSFDO.iterator, Int32)
        num_sfa_stages = 2 * self.k_halves

        load_mma_K_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_K_stage)
        # Advance state to match mbarrier phase accumulated across D-half passes.
        n_advance = cumulative_trip_count % Int32(2 * self.load_mma_K_stage)
        for _ in cutlass.range_constexpr(2 * self.load_mma_K_stage):
            if n_advance > Int32(0):
                load_mma_K_producer_state.advance()
                n_advance = n_advance - Int32(1)
        # Fix for persistent mode: CTA 1 (non-leader) never calls arrive_and_expect_tx
        # on the full barrier (guarded by is_leader_cta in producer_acquire), but TMA
        # copies still signal it with bytes. Over many persistent tiles, the mbarrier
        # transaction counter overflows causing hardware errors. Reinitialize CTA 1's
        # full barriers at the start of each tile to reset accumulated bytes.
        if cutlass.const_expr(self.is_persistent):
            bidx_reinit, _, _ = cute.arch.block_idx()
            is_non_leader = bidx_reinit % cute.size(QK_tiled_mma.thr_id.shape) != 0
            if is_non_leader:
                with cute.arch.elect_one():
                    for stage_idx in cutlass.range_constexpr(self.load_mma_K_stage):
                        cute.arch.mbarrier_init(
                            load_mma_K_pipeline.sync_object_full.get_barrier(stage_idx),
                            1,  # arrive_count = producer_group.size = 1
                        )
                cute.arch.mbarrier_init_fence()
        # Canonical SF planes: one plane per (b, h_kv) resp. (b, h_q)
        SF_kv_load_index = blk_coord_b * grid_dim_y + blk_coord_h_k
        num_h_r = problem_shape[3][0][0]
        SF_q_load_index = SF_kv_load_index * num_h_r + blk_coord_h_r
        q_tile_base = (blk_coord_q // 2) * 2

        # Canonical SF atoms of one KV tile -> slot 0 of a K stage (issued by one
        # elected thread before the payload TMA so they land first; the loop below
        # re-issues the same sequence).  The DSL forbids closure capture inside
        # staged loops, so the per-stage issue sequence is a module-level helper
        # taking everything explicitly.
        kv_sf_geom = (self.SFK_halves, self.k_halves, sfk_stage_stride, sfkmn_stage_stride, sfv_stage_stride)

        load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
        tma_barrier_K = load_mma_K_pipeline.producer_get_barrier(load_mma_K_producer_state)
        first_landed = sf_landed_mbar_ptr + load_mma_K_producer_state.index
        with cute.arch.elect_one():
            # Multicast to both CTAs: each box completes twice on this barrier.
            cute.arch.mbarrier_arrive_and_expect_tx(first_landed, self.sf_kv_stage_bytes + self.sf_q_prologue_bytes)
            cute_common.sf_issue_kv_loads(
                load_mma_K_producer_state.index,
                iter_index,
                SF_kv_load_index,
                first_landed,
                (sSFK.iterator, sSFK_mn.iterator, sSFV.iterator),
                (sfk_tmap, sfkt_tmap, sfv_tmap),
                kv_sf_geom,
            )
            # Both Q sub-block tiles of Q and dO (A operands), one canonical atom per K half.
            for tile in range(2):
                for k_half in range(self.k_halves):
                    sf_stage = tile * self.k_halves + k_half
                    cute_common.sf_tma_load_local(
                        sfq_tmap, sSFQ.iterator + sf_stage * sfq_stage_stride, first_landed, True, k_half, q_tile_base + tile, SF_q_load_index
                    )
                    cute_common.sf_tma_load_local(
                        sfdo_tmap, sSFDO.iterator + sf_stage * sfdo_stage_stride, first_landed, True, k_half, q_tile_base + tile, SF_q_load_index
                    )

        # 2-MMA-warp: load 2 Q tiles + 2 dO tiles (payload only; scale-factor
        # bytes are tracked on this CTA's sf_landed barrier).
        # Each A-operand TMA (CtaGroup.TWO) writes to both CTAs: 2* per tile
        prologue_expect_tx = 2 * 2 * (self.tma_copy_Q_bytes) + 2 * 2 * (self.tma_copy_dO_bytes)
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(tma_barrier_K, prologue_expect_tx)

        # (s_q, d, h_r, h_k, b)
        # Load K (B operand)
        cute.copy(
            tma_atom_K,
            tKgK_mkl[(None, iter_index, 0, (blk_coord_h_kv, blk_coord_b))],
            tKsK[None, load_mma_K_producer_state.index],
            tma_bar_ptr=tma_barrier_K,
            mcast_mask=self.b_full_mcast_mask,
        )

        cute.copy(
            tma_atom_KT,
            tKTgKT_mkl[(None, 0, iter_index, (blk_coord_h_kv, blk_coord_b))],
            tKTsKT[None, load_mma_K_producer_state.index],
            tma_bar_ptr=tma_barrier_K,
            mcast_mask=self.b_full_mcast_mask,
        )

        # Load V (B operand) - same pipeline as K
        cute.copy(
            tma_atom_V,
            tVgV_mkl[(None, iter_index, 0, (blk_coord_h_kv, blk_coord_b))],
            tVsV[None, load_mma_K_producer_state.index],
            tma_bar_ptr=tma_barrier_K,
            mcast_mask=self.b_full_mcast_mask,
        )

        # Load the two 64-row Q sub-blocks for this 2-CTA cluster.
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, q_tile_base, 0, (blk_coord_h_q, blk_coord_b))],
            tQsQ[None, 0],
            tma_bar_ptr=tma_barrier_K,
            mcast_mask=self.a_full_mcast_mask,
        )
        # Load Q tile 1 (A operand) into sQ stage 1
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, q_tile_base + 1, 0, (blk_coord_h_q, blk_coord_b))],
            tQsQ[None, 1],
            tma_bar_ptr=tma_barrier_K,
            mcast_mask=self.a_full_mcast_mask,
        )
        # Load dO tile 0 (A operand) into sdO stage 0
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, q_tile_base, 0, (blk_coord_h_q, blk_coord_b))],
            tdOsdO[(None, 0)],
            tma_bar_ptr=tma_barrier_K,
            mcast_mask=self.a_full_mcast_mask,
        )
        # Load dO tile 1 (A operand) into sdO stage 1
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, q_tile_base + 1, 0, (blk_coord_h_q, blk_coord_b))],
            tdOsdO[(None, 1)],
            tma_bar_ptr=tma_barrier_K,
            mcast_mask=self.a_full_mcast_mask,
        )

        # Native SF: this CTA's canonical atoms of the stage landed -> build its shifted slots (Q / dO A-operand atoms for both sub-blocks: copies in the leader, 64-row shifts in the peer; plus the KV tile's SFB slots) and release the stage to the leader.
        cute.arch.mbarrier_wait(first_landed, Int32(1) - load_mma_K_producer_state.phase)
        if is_leader_cta:
            cute_common.sf_build_slots(words_q, cute_common.sf_a_operand_ops(num_sfa_stages, sfq_stage_stride, sfq_slot_stride, False), sf_lane)
            cute_common.sf_build_slots(words_do, cute_common.sf_a_operand_ops(num_sfa_stages, sfdo_stage_stride, sfdo_slot_stride, False), sf_lane)
        else:
            cute_common.sf_build_slots(words_q, cute_common.sf_a_operand_ops(num_sfa_stages, sfq_stage_stride, sfq_slot_stride, True), sf_lane)
            cute_common.sf_build_slots(words_do, cute_common.sf_a_operand_ops(num_sfa_stages, sfdo_stage_stride, sfdo_slot_stride, True), sf_lane)
        cute_common.sf_build_kv_slots(
            (words_k, words_v, words_kmn),
            load_mma_K_producer_state.index,
            (self.SFK_halves, self.k_halves, sfkmn_atoms_per_slot),
            (sfk_stage_stride, sfv_stage_stride, sfkmn_stage_stride),
            (sfk_slot_stride, sfv_slot_stride, sfkmn_slot_stride),
            sf_lane,
        )
        cute_common.sf_slots_ready_arrive_remote(load_mma_K_pipeline.sync_object_full.get_barrier(load_mma_K_producer_state.index), is_leader_cta)

        load_mma_K_producer_state.advance()

        iter_count -= 1
        iter_index += 1

        while iter_count > 0:
            if iter_index == iter_end:
                iter_index = iter_start

            load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
            tma_barrier_K_inner = load_mma_K_pipeline.producer_get_barrier(load_mma_K_producer_state)
            # Scale factors of this KV tile: canonical atoms -> slot 0, bytes on this CTA's sf_landed.
            landed = sf_landed_mbar_ptr + load_mma_K_producer_state.index
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(landed, self.sf_kv_stage_bytes)
                cute_common.sf_issue_kv_loads(
                    load_mma_K_producer_state.index,
                    iter_index,
                    SF_kv_load_index,
                    landed,
                    (sSFK.iterator, sSFK_mn.iterator, sSFV.iterator),
                    (sfk_tmap, sfkt_tmap, sfv_tmap),
                    kv_sf_geom,
                )

            # Load K (B operand)
            cute.copy(
                tma_atom_K,
                tKgK_mkl[(None, iter_index, 0, (blk_coord_h_kv, blk_coord_b))],
                tKsK[None, load_mma_K_producer_state.index],
                tma_bar_ptr=tma_barrier_K_inner,
                mcast_mask=self.b_full_mcast_mask,
            )

            cute.copy(
                tma_atom_KT,
                tKTgKT_mkl[(None, 0, iter_index, (blk_coord_h_kv, blk_coord_b))],
                tKTsKT[None, load_mma_K_producer_state.index],
                tma_bar_ptr=tma_barrier_K_inner,
                mcast_mask=self.b_full_mcast_mask,
            )

            # Load V (B operand) - same pipeline as K
            cute.copy(
                tma_atom_V,
                tVgV_mkl[(None, iter_index, 0, (blk_coord_h_kv, blk_coord_b))],
                tVsV[None, load_mma_K_producer_state.index],
                tma_bar_ptr=tma_barrier_K_inner,
                mcast_mask=self.b_full_mcast_mask,
            )

            # Native SF: this CTA's canonical atoms of the stage landed -> build its shifted slots (the KV tile's SFB slots) and release the stage to the leader.
            cute.arch.mbarrier_wait(landed, Int32(1) - load_mma_K_producer_state.phase)
            cute_common.sf_build_kv_slots(
                (words_k, words_v, words_kmn),
                load_mma_K_producer_state.index,
                (self.SFK_halves, self.k_halves, sfkmn_atoms_per_slot),
                (sfk_stage_stride, sfv_stage_stride, sfkmn_stage_stride),
                (sfk_slot_stride, sfv_slot_stride, sfkmn_slot_stride),
                sf_lane,
            )
            cute_common.sf_slots_ready_arrive_remote(load_mma_K_pipeline.sync_object_full.get_barrier(load_mma_K_producer_state.index), is_leader_cta)

            load_mma_K_producer_state.advance()

            iter_count -= 1
            iter_index += 1

    @cute.jit
    def sfv_s2t_helper(
        self,
        tDPtSFV_s2t: cute.Tensor,
        tDPtSFV_s2t_h1: cute.Tensor,
        tDPtSFDO_0: cute.Tensor,
        tDPtSFDO_h1_0: cute.Tensor,
        tDPtSFDO_1: cute.Tensor,
        tDPtSFDO_h1_1: cute.Tensor,
        tSTtSFK: cute.Tensor,
        tSTtSFK_h1: cute.Tensor,
        iter_count: Int32,
        pipeline_args: tuple,
        is_leader_cta: Boolean,
        sSFK: cute.Tensor,
        sSFV: cute.Tensor,
        sSFDO: cute.Tensor,
        cumulative_trip_count: Int32 = Int32(0),
    ):
        (
            tiled_copy_s2t_sfk,
            tCsSFK_compact_s2t,
            tCtSFK_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK, tSTtSFK, is_SFA=False)
        (
            tiled_copy_s2t_sfk_h1,
            tCsSFK_compact_s2t_h1,
            tCtSFK_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK, tSTtSFK_h1, is_SFA=False)
        (
            tiled_copy_s2t_sfdo_0,
            tCsSFDO_compact_s2t_0,
            tCtSFDO_compact_s2t_0,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO, tDPtSFDO_0, is_SFA=True)
        (
            tiled_copy_s2t_sfdo_h1_0,
            tCsSFDO_compact_s2t_h1_0,
            tCtSFDO_compact_s2t_h1_0,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO, tDPtSFDO_h1_0, is_SFA=True)
        (
            tiled_copy_s2t_sfdo_1,
            tCsSFDO_compact_s2t_1,
            tCtSFDO_compact_s2t_1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO, tDPtSFDO_1, is_SFA=True)
        (
            tiled_copy_s2t_sfdo_h1_1,
            tCsSFDO_compact_s2t_h1_1,
            tCtSFDO_compact_s2t_h1_1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO, tDPtSFDO_h1_1, is_SFA=True)

        (
            tiled_copy_s2t_sfv,
            tCsSFV_compact_s2t,
            tCtSFV_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sfb_mn_2x64(self, sSFV, tDPtSFV_s2t)
        (
            tiled_copy_s2t_sfv_h1,
            tCsSFV_compact_s2t_h1,
            tCtSFV_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sfb_mn_2x64(self, sSFV, tDPtSFV_s2t_h1)

        load_mma_K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_K_stage)
        if cutlass.const_expr(self.is_persistent):
            n_advance = cumulative_trip_count % Int32(2 * self.load_mma_K_stage)
            for _ in cutlass.range_constexpr(2 * self.load_mma_K_stage):
                if n_advance > Int32(0):
                    load_mma_K_consumer_state.advance()
                    n_advance = n_advance - Int32(1)
        iter_count_origin = iter_count
        (load_mma_K_pipeline,) = pipeline_args

        while iter_count > 0:
            s2t_sfv_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves,
            )
            s2t_sfv_stage_coord_h1 = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves + 1,
            )
            s2t_sfk_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves,
            )
            s2t_sfk_stage_coord_h1 = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves + 1,
            )

            self.sfv_s2t_start_barrier.arrive_and_wait()
            if is_leader_cta:
                load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)
                if iter_count != iter_count_origin:
                    cute.copy(
                        tiled_copy_s2t_sfk,
                        tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                        tCtSFK_compact_s2t,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfk_h1,
                        tCsSFK_compact_s2t_h1[s2t_sfk_stage_coord_h1],
                        tCtSFK_compact_s2t_h1,
                    )
                cute.copy(
                    tiled_copy_s2t_sfv,
                    tCsSFV_compact_s2t[s2t_sfv_stage_coord],
                    tCtSFV_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfv_h1,
                    tCsSFV_compact_s2t_h1[s2t_sfv_stage_coord_h1],
                    tCtSFV_compact_s2t_h1,
                )
                if iter_count == iter_count_origin:
                    cute.copy(
                        tiled_copy_s2t_sfdo_0,
                        tCsSFDO_compact_s2t_0[None, None, None, None, 0],
                        tCtSFDO_compact_s2t_0,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfdo_h1_0,
                        tCsSFDO_compact_s2t_h1_0[None, None, None, None, 1],
                        tCtSFDO_compact_s2t_h1_0,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfdo_1,
                        tCsSFDO_compact_s2t_1[None, None, None, None, self.k_halves],
                        tCtSFDO_compact_s2t_1,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfdo_h1_1,
                        tCsSFDO_compact_s2t_h1_1[None, None, None, None, self.k_halves + 1],
                        tCtSFDO_compact_s2t_h1_1,
                    )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()
            self.sfv_s2t_done_barrier.arrive_and_wait()

            load_mma_K_consumer_state.advance()
            iter_count -= 1

    @cute.jit
    def mma_interleaved(
        self,
        tmem: utils.TmemAllocator,
        QK_tiled_mma: cute.TiledMma,
        DOV_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        QK_tiled_mma_smem: cute.TiledMma,
        tStS_0: cute.Tensor,
        tStS_1: cute.Tensor,
        tSTrK: cute.Tensor,
        tSTtSFQ_0: cute.Tensor,
        tSTtSFQ_h1_0: cute.Tensor,
        tSTtSFQ_1: cute.Tensor,
        tSTtSFQ_h1_1: cute.Tensor,
        tSTtSFK: cute.Tensor,
        tSTtSFK_h1: cute.Tensor,
        tDPtSFV: cute.Tensor,
        tDPtSFV_s2t: cute.Tensor,
        tDPtSFV_h1: cute.Tensor,
        tDPtSFV_s2t_h1: cute.Tensor,
        tDPtSFDO_0: cute.Tensor,
        tDPtSFDO_h1_0: cute.Tensor,
        tDPtSFDO_1: cute.Tensor,
        tDPtSFDO_h1_1: cute.Tensor,
        tDPtDP_0: cute.Tensor,
        tDPtDP_1: cute.Tensor,
        tdQtSFK_mn: cute.Tensor,
        tdPTrV: cute.Tensor,
        tDPrDO: cute.Tensor,
        tdQtdQ_0: cute.Tensor,
        tdQtdQ_1: cute.Tensor,
        tDQrDS: cute.Tensor,
        tdQrKT: cute.Tensor,
        tDQtSFDS_0: cute.Tensor,
        tDQtSFDS_1: cute.Tensor,
        sDS_scale_exchange_0: cute.Tensor,
        sDS_scale_exchange_1: cute.Tensor,
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        pipeline_args: tuple,
        mma_compute_dKdV_producer_state_0,
        mma_compute_dKdV_producer_state_1,
        sSFQ: cute.Tensor,
        sSFK: cute.Tensor,
        sSFK_mn: cute.Tensor,
        sSFV: cute.Tensor,
        sSFDO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sDO: cute.Tensor,
        sKT: cute.Tensor,
        cumulative_trip_count: Int32 = Int32(0),
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(QK_tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        (
            load_mma_K_pipeline,
            mma_compute_S_pipeline_0,
            mma_compute_dP_pipeline_0,
            compute_mma_dS_pipeline_0,
            mma_compute_dQ_pipeline_0,
            mma_compute_S_pipeline_1,
            mma_compute_dP_pipeline_1,
            compute_mma_dS_pipeline_1,
            mma_compute_dQ_pipeline_1,
        ) = pipeline_args

        tmem.wait_for_alloc()

        load_mma_K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_K_stage)
        if cutlass.const_expr(self.is_persistent):
            n_advance = cumulative_trip_count % Int32(2 * self.load_mma_K_stage)
            for _ in cutlass.range_constexpr(2 * self.load_mma_K_stage):
                if n_advance > Int32(0):
                    load_mma_K_consumer_state.advance()
                    n_advance = n_advance - Int32(1)
        compute_mma_dS_consumer_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        compute_mma_dS_consumer_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)

        mma_compute_S_producer_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        mma_compute_dP_producer_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        mma_compute_S_producer_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        mma_compute_dP_producer_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        if cutlass.const_expr(self.is_persistent):
            n_advance_s = cumulative_trip_count % Int32(2 * self.mma_compute_S_stage)
            for _ in cutlass.range_constexpr(2 * self.mma_compute_S_stage):
                if n_advance_s > Int32(0):
                    mma_compute_S_producer_state_0.advance()
                    mma_compute_S_producer_state_1.advance()
                    n_advance_s = n_advance_s - Int32(1)
            n_advance_dp = cumulative_trip_count % Int32(2 * self.mma_compute_dP_stage)
            for _ in cutlass.range_constexpr(2 * self.mma_compute_dP_stage):
                if n_advance_dp > Int32(0):
                    mma_compute_dP_producer_state_0.advance()
                    mma_compute_dP_producer_state_1.advance()
                    n_advance_dp = n_advance_dp - Int32(1)
            n_advance_ds = cumulative_trip_count % Int32(2 * self.compute_mma_dS_stage)
            for _ in cutlass.range_constexpr(2 * self.compute_mma_dS_stage):
                if n_advance_ds > Int32(0):
                    compute_mma_dS_consumer_state_0.advance()
                    compute_mma_dS_consumer_state_1.advance()
                    n_advance_ds = n_advance_ds - Int32(1)

        (
            tiled_copy_s2t_sfq_0,
            tCsSFQ_compact_s2t_0,
            tCtSFQ_compact_s2t_0,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ_0, is_SFA=True)
        (
            tiled_copy_s2t_sfq_h1_0,
            tCsSFQ_compact_s2t_h1_0,
            tCtSFQ_compact_s2t_h1_0,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ_h1_0, is_SFA=True)
        (
            tiled_copy_s2t_sfq_1,
            tCsSFQ_compact_s2t_1,
            tCtSFQ_compact_s2t_1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ_1, is_SFA=True)
        (
            tiled_copy_s2t_sfq_h1_1,
            tCsSFQ_compact_s2t_h1_1,
            tCtSFQ_compact_s2t_h1_1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ_h1_1, is_SFA=True)

        (
            tiled_copy_s2t_sfk,
            tCsSFK_compact_s2t,
            tCtSFK_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK, tSTtSFK, is_SFA=False)
        (
            tiled_copy_s2t_sfk_h1,
            tCsSFK_compact_s2t_h1,
            tCtSFK_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK, tSTtSFK_h1, is_SFA=False)
        (
            tiled_copy_s2t_sfk_mn,
            tCsSFK_mn_compact_s2t_mn,
            tCtSFK_mn_compact_s2t_mn,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK_mn, tdQtSFK_mn, is_SFA=False)

        QK_tiled_mma = QK_tiled_mma_smem
        tSrQ_h0_0 = QK_tiled_mma_smem.make_fragment_A(sQ[(None, None, (None, 0), 0)])
        tSrQ_h1_0 = QK_tiled_mma_smem.make_fragment_A(sQ[(None, None, (None, 1), 0)])
        tSrQ_h0_1 = QK_tiled_mma_smem.make_fragment_A(sQ[(None, None, (None, 0), 1)])
        tSrQ_h1_1 = QK_tiled_mma_smem.make_fragment_A(sQ[(None, None, (None, 1), 1)])
        tSTrK_h0 = QK_tiled_mma_smem.make_fragment_B(sK[(None, None, (None, 0), None)])
        tSTrK_h1 = QK_tiled_mma_smem.make_fragment_B(sK[(None, None, (None, 1), None)])
        tDPrDO_h0_0 = DOV_tiled_mma.make_fragment_A(sDO[(None, None, (None, 0), 0)])
        tDPrDO_h1_0 = DOV_tiled_mma.make_fragment_A(sDO[(None, None, (None, 1), 0)])
        tDPrDO_h0_1 = DOV_tiled_mma.make_fragment_A(sDO[(None, None, (None, 0), 1)])
        tDPrDO_h1_1 = DOV_tiled_mma.make_fragment_A(sDO[(None, None, (None, 1), 1)])
        tdPTrV_h0 = DOV_tiled_mma.make_fragment_B(sV[(None, None, (None, 0), None)])
        tdPTrV_h1 = DOV_tiled_mma.make_fragment_B(sV[(None, None, (None, 1), None)])

        iter_index = iter_start
        iter_count_origin = iter_count

        while iter_count > 0:
            if iter_index == iter_end:
                iter_index = iter_start

            if is_leader_cta:
                load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)
            if iter_count == iter_count_origin:
                self.sfv_s2t_start_barrier.arrive_and_wait()

            kt_stage = load_mma_K_consumer_state.index
            s2t_stage_coord = (
                None,
                None,
                None,
                None,
                kt_stage,
            )
            s2t_sfk_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves,
            )
            s2t_sfk_stage_coord_h1 = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves + 1,
            )
            if is_leader_cta:
                if iter_count == iter_count_origin:
                    cute.copy(
                        tiled_copy_s2t_sfq_0,
                        tCsSFQ_compact_s2t_0[None, None, None, None, 0],
                        tCtSFQ_compact_s2t_0,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfq_h1_0,
                        tCsSFQ_compact_s2t_h1_0[None, None, None, None, 1],
                        tCtSFQ_compact_s2t_h1_0,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfk,
                        tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                        tCtSFK_compact_s2t,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfk_h1,
                        tCsSFK_compact_s2t_h1[s2t_sfk_stage_coord_h1],
                        tCtSFK_compact_s2t_h1,
                    )
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_view_async_tmem_load()

            if iter_count != iter_count_origin:
                self.sfv_s2t_done_barrier.arrive_and_wait()
                # Previous tile used tStS0 as WG1's aliased dP slot. Wait until
                # WG1 has copied that dP before S0 overwrites tStS0.
                self.dS_sync_barrier_compute1.arrive_and_wait()

            if is_leader_cta:
                mma_compute_S_pipeline_0.producer_acquire(mma_compute_S_producer_state_0)
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tSrQ = tSrQ_h1_0 if k_half > 0 else tSrQ_h0_0
                    _tSTrK = tSTrK_h1 if k_half > 0 else tSTrK_h0
                    _tSTtSFQ = tSTtSFQ_h1_0 if k_half > 0 else tSTtSFQ_0
                    _tSTtSFK = tSTtSFK_h1 if k_half > 0 else tSTtSFK
                    for k_block in cutlass.range_constexpr(4):
                        QK_tiled_mma.set(tcgen05.Field.SFA, _tSTtSFQ[(None, None, k_block)].iterator)
                        QK_tiled_mma.set(tcgen05.Field.SFB, _tSTtSFK[(None, None, k_block)].iterator)
                        cute.gemm(
                            QK_tiled_mma,
                            tStS_0[None, None, None],
                            _tSrQ[None, None, k_block],
                            _tSTrK[None, None, k_block, load_mma_K_consumer_state.index],
                            tStS_0[None, None, None],
                        )
                        if cutlass.const_expr(k_half == 0 and k_block == 0):
                            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                mma_compute_S_pipeline_0.producer_commit(mma_compute_S_producer_state_0)
                mma_compute_S_producer_state_0.advance()

            if iter_count == iter_count_origin:
                self.sfv_s2t_done_barrier.arrive_and_wait()
            if is_leader_cta:
                mma_compute_dP_pipeline_0.producer_acquire(mma_compute_dP_producer_state_0)
                DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tDPrDO = tDPrDO_h1_0 if k_half > 0 else tDPrDO_h0_0
                    _tdPTrV = tdPTrV_h1 if k_half > 0 else tdPTrV_h0
                    _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                    for k_block in cutlass.range_constexpr(4):
                        _tDPtSFDO = tDPtSFDO_h1_0 if k_half > 0 else tDPtSFDO_0
                        DOV_tiled_mma.set(tcgen05.Field.SFA, _tDPtSFDO[None, None, k_block].iterator)
                        DOV_tiled_mma.set(tcgen05.Field.SFB, _tDPtSFV[None, None, k_block].iterator)
                        cute.gemm(
                            DOV_tiled_mma,
                            tDPtDP_0[None, None, None],
                            _tDPrDO[None, None, k_block],
                            _tdPTrV[None, None, k_block, load_mma_K_consumer_state.index],
                            tDPtDP_0[None, None, None],
                        )
                        if cutlass.const_expr(k_half == 0 and k_block == 0):
                            DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                mma_compute_dP_pipeline_0.producer_commit(mma_compute_dP_producer_state_0)
                mma_compute_dP_producer_state_0.advance()

                if iter_count == iter_count_origin:
                    cute.copy(
                        tiled_copy_s2t_sfq_1,
                        tCsSFQ_compact_s2t_1[None, None, None, None, self.k_halves],
                        tCtSFQ_compact_s2t_1,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfq_h1_1,
                        tCsSFQ_compact_s2t_h1_1[None, None, None, None, self.k_halves + 1],
                        tCtSFQ_compact_s2t_h1_1,
                    )
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_view_async_tmem_load()

                mma_compute_S_pipeline_1.producer_acquire(mma_compute_S_producer_state_1)
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tSrQ = tSrQ_h1_1 if k_half > 0 else tSrQ_h0_1
                    _tSTrK = tSTrK_h1 if k_half > 0 else tSTrK_h0
                    _tSTtSFQ = tSTtSFQ_h1_1 if k_half > 0 else tSTtSFQ_1
                    _tSTtSFK = tSTtSFK_h1 if k_half > 0 else tSTtSFK
                    for k_block in cutlass.range_constexpr(4):
                        QK_tiled_mma.set(tcgen05.Field.SFA, _tSTtSFQ[(None, None, k_block)].iterator)
                        QK_tiled_mma.set(tcgen05.Field.SFB, _tSTtSFK[(None, None, k_block)].iterator)
                        cute.gemm(
                            QK_tiled_mma,
                            tStS_1[None, None, None],
                            _tSrQ[None, None, k_block],
                            _tSTrK[None, None, k_block, load_mma_K_consumer_state.index],
                            tStS_1[None, None, None],
                        )
                        if cutlass.const_expr(k_half == 0 and k_block == 0):
                            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                mma_compute_S_pipeline_1.producer_commit(mma_compute_S_producer_state_1)
                mma_compute_S_producer_state_1.advance()

            self.dS_sync_barrier_compute0.arrive_and_wait()

            if is_leader_cta:
                mma_compute_dP_pipeline_1.producer_acquire(mma_compute_dP_producer_state_1)
                DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tDPrDO = tDPrDO_h1_1 if k_half > 0 else tDPrDO_h0_1
                    _tdPTrV = tdPTrV_h1 if k_half > 0 else tdPTrV_h0
                    _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                    for k_block in cutlass.range_constexpr(4):
                        _tDPtSFDO = tDPtSFDO_h1_1 if k_half > 0 else tDPtSFDO_1
                        DOV_tiled_mma.set(tcgen05.Field.SFA, _tDPtSFDO[None, None, k_block].iterator)
                        DOV_tiled_mma.set(tcgen05.Field.SFB, _tDPtSFV[None, None, k_block].iterator)
                        cute.gemm(
                            DOV_tiled_mma,
                            tDPtDP_1[None, None, None],
                            _tDPrDO[None, None, k_block],
                            _tdPTrV[None, None, k_block, load_mma_K_consumer_state.index],
                            tDPtDP_1[None, None, None],
                        )
                        if cutlass.const_expr(k_half == 0 and k_block == 0):
                            DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                mma_compute_dP_pipeline_1.producer_commit(mma_compute_dP_producer_state_1)
                mma_compute_dP_producer_state_1.advance()

            if iter_count > 1:
                self.sfv_s2t_start_barrier.arrive_and_wait()

            if is_leader_cta:
                # SFK_mn is only consumed by dSK. Prepare it after S1 and DOV1
                # so it does not delay either compute WG's S/dP input.
                cute.copy(
                    tiled_copy_s2t_sfk_mn,
                    tCsSFK_mn_compact_s2t_mn[None, None, None, None, kt_stage],
                    tCtSFK_mn_compact_s2t_mn,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()

            if cutlass.const_expr(self.online_ds_scale):
                self.dS_scale_exchange_barrier_0.arrive_and_wait()
                if tidx % 32 == 0:
                    d256_primitives.copy_mxfp8_scale_tile_to_tmem(sDS_scale_exchange_0, tDQtSFDS_0)
                cute.arch.fence_view_async_tmem_store()
                self.dS_scale_exchange_barrier_0.arrive_and_wait()
            if is_leader_cta:
                compute_mma_dS_pipeline_0.consumer_wait(compute_mma_dS_consumer_state_0)
                if iter_count == iter_count_origin:
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                else:
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll_full=True):
                    sf_kblock_coord = (None, None, k_block)
                    dSK_tiled_mma.set(tcgen05.Field.SFA, tDQtSFDS_0[sf_kblock_coord].iterator)
                    dSK_tiled_mma.set(tcgen05.Field.SFB, tdQtSFK_mn[sf_kblock_coord].iterator)
                    cute.gemm(
                        dSK_tiled_mma,
                        tdQtdQ_0,
                        tDQrDS[None, None, k_block, compute_mma_dS_consumer_state_0.index],
                        tdQrKT[None, None, k_block, kt_stage],
                        tdQtdQ_0,
                    )
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                compute_mma_dS_pipeline_0.consumer_release(compute_mma_dS_consumer_state_0)
            compute_mma_dS_consumer_state_0.advance()

            if cutlass.const_expr(self.online_ds_scale):
                self.dS_scale_exchange_barrier_1.arrive_and_wait()
                if tidx % 32 == 0:
                    d256_primitives.copy_mxfp8_scale_tile_to_tmem(sDS_scale_exchange_1, tDQtSFDS_1)
                cute.arch.fence_view_async_tmem_store()
                self.dS_scale_exchange_barrier_1.arrive_and_wait()
            if is_leader_cta:
                compute_mma_dS_pipeline_1.consumer_wait(compute_mma_dS_consumer_state_1)
                if iter_count == iter_count_origin:
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                else:
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll_full=True):
                    sf_kblock_coord = (None, None, k_block)
                    dSK_tiled_mma.set(tcgen05.Field.SFA, tDQtSFDS_1[sf_kblock_coord].iterator)
                    dSK_tiled_mma.set(tcgen05.Field.SFB, tdQtSFK_mn[sf_kblock_coord].iterator)
                    cute.gemm(
                        dSK_tiled_mma,
                        tdQtdQ_1,
                        tDQrDS[None, None, k_block, compute_mma_dS_consumer_state_1.index + self.compute_mma_dS_stage],
                        tdQrKT[None, None, k_block, kt_stage],
                        tdQtdQ_1,
                    )
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                compute_mma_dS_pipeline_1.consumer_release(compute_mma_dS_consumer_state_1)
            compute_mma_dS_consumer_state_1.advance()

            if is_leader_cta:
                load_mma_K_pipeline.consumer_release(load_mma_K_consumer_state)
            load_mma_K_consumer_state.advance()

            iter_count -= 1
            iter_index += 1

        if is_leader_cta:
            cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_view_async_shared()
            mma_compute_dQ_pipeline_0.producer_acquire(mma_compute_dKdV_producer_state_0)
            mma_compute_dQ_pipeline_0.producer_commit(mma_compute_dKdV_producer_state_0)
            mma_compute_dQ_pipeline_1.producer_acquire(mma_compute_dKdV_producer_state_1)
            mma_compute_dQ_pipeline_1.producer_commit(mma_compute_dKdV_producer_state_1)
        mma_compute_dKdV_producer_state_0.advance()
        mma_compute_dKdV_producer_state_1.advance()

    @cute.jit
    def mma(
        self,
        tmem: utils.TmemAllocator,
        QK_tiled_mma: cute.TiledMma,
        DOV_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        QK_tiled_mma_smem: cute.TiledMma,  # SMEM-sourced Q variant, used for d>128
        tStS: cute.Tensor,  # per-warp slice (no stage dim)
        tSTrK: cute.Tensor,  # shared
        tSTtSFQ: cute.Tensor,  # per-warp tmem
        tSTtSFQ_h1: cute.Tensor,  # d>128 second K-half tmem
        tSTtSFK: cute.Tensor,  # per-warp tmem
        tDPtSFV: cute.Tensor,  # per-warp tmem
        tDPtSFV_s2t: cute.Tensor,  # per-warp tmem, s2t write view
        tDPtSFV_h1: cute.Tensor,  # d>128 second K-half tmem
        tDPtSFV_s2t_h1: cute.Tensor,  # d>128 second K-half s2t write view
        tDPtSFDO: cute.Tensor,  # per-warp tmem
        tDPtDP: cute.Tensor,  # per-warp slice (no stage dim)
        tdQtSFK_mn: cute.Tensor,  # per-warp tmem
        tdPTrV: cute.Tensor,  # shared
        tDPrDO: cute.Tensor,  # per-warp tmem
        tdQtdQ: cute.Tensor,  # per-warp tmem
        tdQtdQ_h1: cute.Tensor,  # per-warp tmem for D half 1
        tDQrDS: cute.Tensor,  # shared smem fragment
        tdQrKT: cute.Tensor,  # shared
        tDQtSFDS: cute.Tensor,  # per-warp tmem
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        # (load_mma_K_pipeline, mma_compute_S_pipeline, mma_compute_dP_pipeline, compute_mma_dS_pipeline, mma_compute_dQ_pipeline)
        pipeline_args: tuple,
        mma_compute_dKdV_producer_state,
        sSFQ: cute.Tensor,
        sSFK: cute.Tensor,
        sSFK_mn: cute.Tensor,
        sSFV: cute.Tensor,
        sSFDO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sDO: cute.Tensor,
        sKT: cute.Tensor,
        sQ_stage: int,  # which smem stage for this warp's Q (0 or 1)
        sDO_stage: int,  # which smem stage for this warp's dO (0 or 1)
        dS_stage_offset: int,  # 0 for warp 0, 2 for warp 1
        d_half_target: int = 0,  # D=128 half computed by this pass
        cumulative_trip_count: Int32 = Int32(0),  # accumulated trip_count across persistent tiles
        load_cumulative_trip_count: Int32 = Int32(0),  # accumulated trip_count for shared load pipeline
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        iter_count_origin = iter_count

        mma_tile_coord_v = bidx % cute.size(QK_tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        (
            load_mma_K_pipeline,
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dQ_pipeline,
        ) = pipeline_args
        tmem.wait_for_alloc()

        load_mma_K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_K_stage)
        n_advance = cumulative_trip_count % Int32(2 * self.load_mma_K_stage)
        for _ in cutlass.range_constexpr(2 * self.load_mma_K_stage):
            if n_advance > Int32(0):
                load_mma_K_consumer_state.advance()
                n_advance = n_advance - Int32(1)
        load_mma_K_release_state = load_mma_K_consumer_state.clone()

        # dS consumer state for alternating barrier selection (2-stage double-buffering)
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        if cutlass.const_expr(self.is_persistent):
            n_advance_ds = cumulative_trip_count % Int32(2 * self.compute_mma_dS_stage)
            for _ in cutlass.range_constexpr(2 * self.compute_mma_dS_stage):
                if n_advance_ds > Int32(0):
                    compute_mma_dS_consumer_state.advance()
                    n_advance_ds = n_advance_ds - Int32(1)

        # 1-stage pipelines: producer_state.index is always 0.
        mma_compute_S_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)

        (
            tiled_copy_s2t_sfq,
            tCsSFQ_compact_s2t,
            tCtSFQ_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ, is_SFA=True)
        (
            tiled_copy_s2t_sfq_h1,
            tCsSFQ_compact_s2t_h1,
            tCtSFQ_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ_h1, is_SFA=True)

        (
            tiled_copy_s2t_sfk,
            tCsSFK_compact_s2t,
            tCtSFK_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK, tSTtSFK, is_SFA=False)

        (
            tiled_copy_s2t_sfk_mn,
            tCsSFK_mn_compact_s2t_mn,
            tCtSFK_mn_compact_s2t_mn,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK_mn, tdQtSFK_mn, is_SFA=False)

        (
            tiled_copy_s2t_sfdo,
            tCsSFDO_compact_s2t,
            tCtSFDO_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO, tDPtSFDO, is_SFA=True)

        (
            tiled_copy_s2t_sfv,
            tCsSFV_compact_s2t,
            tCtSFV_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sfb_mn_2x64(self, sSFV, tDPtSFV_s2t)
        (
            tiled_copy_s2t_sfv_h1,
            tCsSFV_compact_s2t_h1,
            tCtSFV_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sfb_mn_2x64(self, sSFV, tDPtSFV_s2t_h1)

        QK_tiled_mma = QK_tiled_mma_smem
        tSrQ_h0 = QK_tiled_mma_smem.make_fragment_A(sQ[(None, None, (None, 0), sQ_stage)])
        tSrQ_h1 = QK_tiled_mma_smem.make_fragment_A(sQ[(None, None, (None, 1), sQ_stage)])
        tSTrK_h0 = QK_tiled_mma_smem.make_fragment_B(sK[(None, None, (None, 0), None)])
        tSTrK_h1 = QK_tiled_mma_smem.make_fragment_B(sK[(None, None, (None, 1), None)])
        tDPrDO_h0 = DOV_tiled_mma.make_fragment_A(sDO[(None, None, (None, 0), sDO_stage)])
        tDPrDO_h1 = DOV_tiled_mma.make_fragment_A(sDO[(None, None, (None, 1), sDO_stage)])
        tdPTrV_h0 = DOV_tiled_mma.make_fragment_B(sV[(None, None, (None, 0), None)])
        tdPTrV_h1 = DOV_tiled_mma.make_fragment_B(sV[(None, None, (None, 1), None)])

        # Start: S = K * Q
        iter_index = iter_start
        s2t_stage_coord = (
            None,
            None,
            None,
            None,
            load_mma_K_consumer_state.index,
        )

        if is_leader_cta:
            load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)

        if is_leader_cta:
            sfq_stage_base = sQ_stage * self.k_halves
            cute.copy(
                tiled_copy_s2t_sfq,
                tCsSFQ_compact_s2t[None, None, None, None, sfq_stage_base],
                tCtSFQ_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfq_h1,
                tCsSFQ_compact_s2t_h1[None, None, None, None, sfq_stage_base + 1],
                tCtSFQ_compact_s2t_h1,
            )

            cute.copy(
                tiled_copy_s2t_sfk,
                tCsSFK_compact_s2t[s2t_stage_coord],
                tCtSFK_compact_s2t,
            )
            cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_view_async_tmem_load()
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_half in cutlass.range_constexpr(self.k_halves):
                _tSrQ = tSrQ_h1 if k_half > 0 else tSrQ_h0
                _tSTrK = tSTrK_h1 if k_half > 0 else tSTrK_h0
                _tSTtSFQ = tSTtSFQ_h1 if k_half > 0 else tSTtSFQ
                for k_block in cutlass.range_constexpr(4):
                    _sf_k = k_half * 4 + k_block
                    QK_tiled_mma.set(tcgen05.Field.SFA, _tSTtSFQ[(None, None, k_block)].iterator)
                    QK_tiled_mma.set(tcgen05.Field.SFB, tSTtSFK[(None, None, _sf_k)].iterator)
                    cute.gemm(
                        QK_tiled_mma,
                        tStS[None, None, None],
                        _tSrQ[None, None, k_block],
                        _tSTrK[None, None, k_block, load_mma_K_consumer_state.index],
                        tStS[None, None, None],
                    )
                    QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        # Only leader CTA commits pipeline in 2-CTA mode
        if is_leader_cta:
            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
        mma_compute_S_producer_state.advance()
        # End: S = K * Q

        # Start: dP = V * dO (V is in same pipeline as K, already waited above)
        if is_leader_cta:
            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

        s2t_stage_coord = (None, None, None, None, load_mma_K_consumer_state.index)
        if is_leader_cta:
            s2t_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves,
            )
            cute.copy(
                tiled_copy_s2t_sfv,
                tCsSFV_compact_s2t[s2t_stage_coord],
                tCtSFV_compact_s2t,
            )
            s2t_stage_coord_h1 = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index * self.k_halves + 1,
            )
            cute.copy(
                tiled_copy_s2t_sfv_h1,
                tCsSFV_compact_s2t_h1[s2t_stage_coord_h1],
                tCtSFV_compact_s2t_h1,
            )
            cute.copy(
                tiled_copy_s2t_sfdo,
                tCsSFDO_compact_s2t[None, None, None, None, sDO_stage],
                tCtSFDO_compact_s2t,
            )
            cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_view_async_tmem_load()
            DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            for k_half in cutlass.range_constexpr(self.k_halves):
                _tDPrDO = tDPrDO_h1 if k_half > 0 else tDPrDO_h0
                _tdPTrV = tdPTrV_h1 if k_half > 0 else tdPTrV_h0
                _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                for k_block in cutlass.range_constexpr(4):
                    _sf_k = k_half * 4 + k_block
                    DOV_tiled_mma.set(tcgen05.Field.SFA, tDPtSFDO[None, None, _sf_k].iterator)
                    DOV_tiled_mma.set(tcgen05.Field.SFB, _tDPtSFV[None, None, k_block].iterator)
                    cute.gemm(
                        DOV_tiled_mma,
                        tDPtDP[None, None, None],
                        _tDPrDO[None, None, k_block],
                        _tdPTrV[None, None, k_block, load_mma_K_consumer_state.index],
                        tDPtDP[None, None, None],
                    )
                    DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        if is_leader_cta:
            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
        mma_compute_dP_producer_state.advance()

        load_mma_K_consumer_state.advance()
        # End: dP = V * dO

        # =====================================================================
        # Depth-2 prologue: pre-compute S[1] and dP[1]
        # This gives the compute warp a head start on tile 1 while MMA does dQ[0].
        # With 1-stage S/dP pipelines, produce_acquire blocks until compute consumes
        # the previous tile — this is expected and correct.
        # =====================================================================
        if iter_count > 1:
            # Start: second S = K * Q
            s2t_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index,
            )
            if is_leader_cta:
                load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)
                mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
                cute.copy(
                    tiled_copy_s2t_sfk,
                    tCsSFK_compact_s2t[s2t_stage_coord],
                    tCtSFK_compact_s2t,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tSrQ = tSrQ_h1 if k_half > 0 else tSrQ_h0
                    _tSTrK = tSTrK_h1 if k_half > 0 else tSTrK_h0
                    _tSTtSFQ = tSTtSFQ_h1 if k_half > 0 else tSTtSFQ
                    for k_block in cutlass.range_constexpr(4):
                        _sf_k = k_half * 4 + k_block
                        QK_tiled_mma.set(tcgen05.Field.SFA, _tSTtSFQ[(None, None, k_block)].iterator)
                        QK_tiled_mma.set(tcgen05.Field.SFB, tSTtSFK[(None, None, _sf_k)].iterator)
                        cute.gemm(
                            QK_tiled_mma,
                            tStS[None, None, None],
                            _tSrQ[None, None, k_block],
                            _tSTrK[None, None, k_block, load_mma_K_consumer_state.index],
                            tStS[None, None, None],
                        )
                        QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Only leader CTA commits pipeline in 2-CTA mode
            if is_leader_cta:
                mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()
            # End: second S = K * Q

            # Start: second dP = V * dO (V is in same pipeline as K, already waited above)
            if is_leader_cta:
                mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

            s2t_stage_coord = (None, None, None, None, load_mma_K_consumer_state.index)
            if is_leader_cta:
                s2t_stage_coord = (
                    None,
                    None,
                    None,
                    None,
                    load_mma_K_consumer_state.index * self.k_halves,
                )
                cute.copy(
                    tiled_copy_s2t_sfv,
                    tCsSFV_compact_s2t[s2t_stage_coord],
                    tCtSFV_compact_s2t,
                )
                s2t_stage_coord_h1 = (
                    None,
                    None,
                    None,
                    None,
                    load_mma_K_consumer_state.index * self.k_halves + 1,
                )
                cute.copy(
                    tiled_copy_s2t_sfv_h1,
                    tCsSFV_compact_s2t_h1[s2t_stage_coord_h1],
                    tCtSFV_compact_s2t_h1,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()
                DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tDPrDO = tDPrDO_h1 if k_half > 0 else tDPrDO_h0
                    _tdPTrV = tdPTrV_h1 if k_half > 0 else tdPTrV_h0
                    _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                    for k_block in cutlass.range_constexpr(4):
                        _sf_k = k_half * 4 + k_block
                        DOV_tiled_mma.set(tcgen05.Field.SFA, tDPtSFDO[None, None, _sf_k].iterator)
                        DOV_tiled_mma.set(tcgen05.Field.SFB, _tDPtSFV[None, None, k_block].iterator)
                        cute.gemm(
                            DOV_tiled_mma,
                            tDPtDP[None, None, None],
                            _tDPrDO[None, None, k_block],
                            _tdPTrV[None, None, k_block, load_mma_K_consumer_state.index],
                            tDPtDP[None, None, None],
                        )
                        DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            if is_leader_cta:
                mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            load_mma_K_consumer_state.advance()
            # End: second dP = V * dO

        # =====================================================================
        # Mainloop: depth-2 pipelining
        #   dQ += dS @ K^T [tile i] → S = Q @ K [tile i+2] → dP = dO @ V [tile i+2]
        # Uses try_wait/try_acquire to overlap pipeline waits with dQ gemm.
        # =====================================================================
        # Store-once: ACCUMULATE starts False for the first dQ MMA and stays
        # True until one final producer_commit after the K loop drains.
        if is_leader_cta:
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        while iter_count > Int32(2):
            if iter_index == iter_end:
                iter_index = iter_start

            # Start: dQ += dS * KT (TMEM accumulation across all K-tiles).
            kt_stage = load_mma_K_release_state.index
            if is_leader_cta:
                cute.copy(
                    tiled_copy_s2t_sfk_mn,
                    tCsSFK_mn_compact_s2t_mn[None, None, None, None, kt_stage],
                    tCtSFK_mn_compact_s2t_mn,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()
            if is_leader_cta:
                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll_full=True):
                    sf_kblock_coord = (None, None, k_block)
                    dSK_tiled_mma.set(
                        tcgen05.Field.SFA,
                        tDQtSFDS[sf_kblock_coord].iterator,
                    )
                    dSK_tiled_mma.set(
                        tcgen05.Field.SFB,
                        tdQtSFK_mn[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        dSK_tiled_mma,
                        tdQtdQ,
                        tDQrDS[
                            None,
                            None,
                            k_block,
                            compute_mma_dS_consumer_state.index + dS_stage_offset,
                        ],
                        tdQrKT[None, None, k_block, kt_stage],
                        tdQtdQ,
                    )
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            peak_mma_compute_S_status = cutlass.Boolean(False)
            peak_load_mma_status = cutlass.Boolean(False)
            if is_leader_cta:
                peak_load_mma_status = load_mma_K_pipeline.consumer_try_wait(load_mma_K_consumer_state)
                peak_mma_compute_S_status = mma_compute_S_pipeline.producer_try_acquire(mma_compute_S_producer_state)
                load_mma_K_pipeline.consumer_release(load_mma_K_release_state)
                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()
            load_mma_K_release_state.advance()
            if is_leader_cta:
                load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state, peak_load_mma_status)

            # End: dQ = dS * KT

            # Start: S = Q * K
            s2t_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_K_consumer_state.index,
            )
            if is_leader_cta:
                mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state, peak_mma_compute_S_status)
                cute.copy(
                    tiled_copy_s2t_sfk,
                    tCsSFK_compact_s2t[s2t_stage_coord],
                    tCtSFK_compact_s2t,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tSrQ = tSrQ_h1 if k_half > 0 else tSrQ_h0
                    _tSTrK = tSTrK_h1 if k_half > 0 else tSTrK_h0
                    _tSTtSFQ = tSTtSFQ_h1 if k_half > 0 else tSTtSFQ
                    for k_block in cutlass.range_constexpr(4):
                        _sf_k = k_half * 4 + k_block
                        QK_tiled_mma.set(tcgen05.Field.SFA, _tSTtSFQ[(None, None, k_block)].iterator)
                        QK_tiled_mma.set(tcgen05.Field.SFB, tSTtSFK[(None, None, _sf_k)].iterator)
                        cute.gemm(
                            QK_tiled_mma,
                            tStS[None, None, None],
                            _tSrQ[None, None, k_block],
                            _tSTrK[None, None, k_block, load_mma_K_consumer_state.index],
                            tStS[None, None, None],
                        )
                        QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            peak_mma_compute_dP_status = cutlass.Boolean(False)
            # Only leader CTA commits pipeline in 2-CTA mode
            if is_leader_cta:
                peak_mma_compute_dP_status = mma_compute_dP_pipeline.producer_try_acquire(mma_compute_dP_producer_state)
                mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()
            # End: S = K * Q

            # Start: dP = V * dO (V is in same pipeline as K, already waited above)
            if is_leader_cta:
                mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state, peak_mma_compute_dP_status)
            s2t_stage_coord = (None, None, None, None, load_mma_K_consumer_state.index)
            if is_leader_cta:
                s2t_stage_coord = (
                    None,
                    None,
                    None,
                    None,
                    load_mma_K_consumer_state.index * self.k_halves,
                )
                cute.copy(
                    tiled_copy_s2t_sfv,
                    tCsSFV_compact_s2t[s2t_stage_coord],
                    tCtSFV_compact_s2t,
                )
                s2t_stage_coord_h1 = (
                    None,
                    None,
                    None,
                    None,
                    load_mma_K_consumer_state.index * self.k_halves + 1,
                )
                cute.copy(
                    tiled_copy_s2t_sfv_h1,
                    tCsSFV_compact_s2t_h1[s2t_stage_coord_h1],
                    tCtSFV_compact_s2t_h1,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()
                # dP = V * dO
                DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tDPrDO = tDPrDO_h1 if k_half > 0 else tDPrDO_h0
                    _tdPTrV = tdPTrV_h1 if k_half > 0 else tdPTrV_h0
                    _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                    for k_block in cutlass.range_constexpr(4):
                        _sf_k = k_half * 4 + k_block
                        DOV_tiled_mma.set(tcgen05.Field.SFA, tDPtSFDO[None, None, _sf_k].iterator)
                        DOV_tiled_mma.set(tcgen05.Field.SFB, _tDPtSFV[None, None, k_block].iterator)
                        cute.gemm(
                            DOV_tiled_mma,
                            tDPtDP[None, None, None],
                            _tDPrDO[None, None, k_block],
                            _tdPTrV[None, None, k_block, load_mma_K_consumer_state.index],
                            tDPtDP[None, None, None],
                        )
                        DOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Only leader CTA commits/releases pipeline
            if is_leader_cta:
                mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            load_mma_K_consumer_state.advance()
            # End: dP = V * dO
            iter_count -= 1
            iter_index += 1

        # =====================================================================
        # Epilogue: dQ += dS @ K^T [last 1-2 tiles], D-half tiled.
        # =====================================================================
        if is_leader_cta:
            mma_compute_dQ_pipeline.producer_acquire(mma_compute_dKdV_producer_state)

        for i in cutlass.range(cutlass.min(2, iter_count_origin), unroll_full=True):
            kt_stage = load_mma_K_release_state.index
            if is_leader_cta:
                cute.copy(
                    tiled_copy_s2t_sfk_mn,
                    tCsSFK_mn_compact_s2t_mn[None, None, None, None, kt_stage],
                    tCtSFK_mn_compact_s2t_mn,
                )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_tmem_load()
            if is_leader_cta:
                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll_full=True):
                    sf_kblock_coord = (None, None, k_block)
                    dSK_tiled_mma.set(
                        tcgen05.Field.SFA,
                        tDQtSFDS[sf_kblock_coord].iterator,
                    )
                    dSK_tiled_mma.set(
                        tcgen05.Field.SFB,
                        tdQtSFK_mn[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        dSK_tiled_mma,
                        tdQtdQ,
                        tDQrDS[
                            None,
                            None,
                            k_block,
                            compute_mma_dS_consumer_state.index + dS_stage_offset,
                        ],
                        tdQrKT[None, None, k_block, kt_stage],
                        tdQtdQ,
                    )
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            if is_leader_cta:
                load_mma_K_pipeline.consumer_release(load_mma_K_release_state)
            load_mma_K_release_state.advance()

            if is_leader_cta:
                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

        # Store-once: commit dQ slot ONCE after the whole K loop completes.
        if is_leader_cta:
            cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_view_async_shared()
            mma_compute_dQ_pipeline.producer_commit(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_producer_state.advance()

    @cute.jit
    def compute(
        self,
        tStS: cute.Tensor,  # per-warp slice (no stage dim)
        tDPtDP: cute.Tensor,  # per-warp slice (no stage dim)
        tDQtSFDS: cute.Tensor,  # this WG sub-block SFDS
        sDS: cute.Tensor,
        sDS_scale_exchange: cute.Tensor,
        blk_coord: cute.Coord,
        blk_coord_mask: cute.Coord,  # per-tile mask coord
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        trip_start_mask: Int32,  # per-tile trip_start for mask count computation
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        is_leader_cta: Boolean,
        LSE: cute.Tensor,
        sum_OdO: cute.Tensor,
        # (mma_compute_S_pipeline, mma_compute_dP_pipeline, compute_mma_dS_pipeline)
        pipeline_args: tuple,
        wg_idx: int,  # 0 for compute group 0, 1 for compute group 1
        dS_stage_offset: int,  # 0 for warp 0, 2 for warp 1
        trip_count_mask: Int32 = Int32(0),  # per-warp trip count for causal bounds
        cumulative_trip_count: Int32 = Int32(0),  # accumulated trip_count across persistent tiles
    ):
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, _, _ = problem_shape
        blk_coord_q, _, _, blk_coord_batch = blk_coord
        blk_coord_q_mask = blk_coord_mask[0]

        # Each compute group handles ALL iterations (no even/odd interleaving)

        # tidx % 64 maps threads to their corresponding row within the CTA's 64-row region
        lse_odo_offset = tidx % 64
        # Compute global Q index for this thread
        # Each warp group processes a different 64-row tile within the CTA's region
        # blk_coord_q_mask = bidx*2 (warp 0) or bidx*2+1 (warp 1)
        # Per CTA: tile_shape_Q // use_2cta_divisor = 128 // 2 = 64 rows per tile
        q_global_idx = blk_coord_q_mask * (self.tile_shape_Q // self.use_2cta_divisor) + lse_odo_offset
        # Read LSE directly from global memory
        lse = LSE[q_global_idx, blk_coord_batch] if cute.elem_less(q_global_idx, Q) else 0.0
        sum_OdO_val = sum_OdO[q_global_idx, blk_coord_batch] if cute.elem_less(q_global_idx, Q) else 0.0

        iter_index = iter_start
        (
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
        ) = pipeline_args

        # 1-stage pipelines
        mma_compute_S_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage)

        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage)
        # dS producer state for alternating barrier selection (2-stage double-buffering)
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage)
        if cutlass.const_expr(self.is_persistent):
            n_advance_ds = cumulative_trip_count % Int32(2 * self.compute_mma_dS_stage)
            for _ in cutlass.range_constexpr(2 * self.compute_mma_dS_stage):
                if n_advance_ds > Int32(0):
                    compute_mma_dS_producer_state.advance()
                    n_advance_ds = n_advance_ds - Int32(1)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)),
            LOW_PRECISION_TYPE,
        )
        # Per-warp tStS/tDPtDP already sliced — no stage dim
        tStS_0 = tStS[(None, None), 0, 0]
        tDPtDP_0 = tDPtDP[(None, None), 0, 0]

        # Use full mma_tiler for identity tensors - matches tmem tensor shape
        cS = cute.make_identity_tensor(cute.select(self.QK_cta_tiler, mode=[0, 1]))
        cDP = cute.make_identity_tensor(cute.select(self.DOV_cta_tiler, mode=[0, 1]))

        dp_idx = tidx % 128
        warp_rank = (tidx % 128) // 32
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS_0)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cS = thr_t2r.partition_D(cS)
        tTR_rS = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)

        tTR_cDP = thr_t2r.partition_D(cDP)
        tTR_rDP = cute.make_rmem_tensor(tTR_cDP.shape, self.acc_dtype)
        tTR_tS = thr_t2r.partition_S(tStS_0)
        tTR_tDP = thr_t2r.partition_S(tDPtDP_0)

        # Use per-tile mask coord for mask computation
        masked_leading_count = fmha_masks.FusedMask.get_masked_leading_count(
            self.mask_type,
            blk_coord_mask,
            self.mask_cta_tiler,
            Q,
            K,
            window_size_left,
            window_size_right,
        )
        unmasked_count = fmha_masks.FusedMask.get_unmasked_trip_count(
            self.mask_type,
            blk_coord_mask,
            self.mask_cta_tiler,
            Q,
            K,
            window_size_left,
            window_size_right,
        )
        masked_trailing_count = fmha_masks.FusedMask.get_masked_trailing_count(
            self.mask_type,
            blk_coord_mask,
            self.mask_cta_tiler,
            Q,
            K,
            window_size_left,
            window_size_right,
        )

        if cutlass.const_expr(not self.online_ds_scale):
            d256_primitives.store_identity_mxfp8_scales_to_tmem(self, tDQtSFDS, tidx)
            cute.arch.fence_view_async_tmem_store()

        while iter_count > 0:
            peak_S_consumer_status = mma_compute_S_pipeline.consumer_try_wait(mma_compute_S_consumer_state)

            iter_num = iter_index - trip_start_mask + 1
            is_residual_k = Boolean(False)
            is_residual_k = iter_index * self.tile_shape_K + self.tile_shape_K > K

            # For Q-row-splitting with causal: iter_count is the union range but
            # this warp's trip_count_mask may be smaller. Tiles beyond the warp's
            # own causal boundary must be fully masked.
            is_beyond_warp_range = Boolean(False)
            if trip_count_mask > 0:
                is_beyond_warp_range = iter_index >= trip_start_mask + trip_count_mask

            is_masked_tile = (
                is_beyond_warp_range
                or is_residual_k
                or iter_num <= masked_leading_count
                or (iter_num > masked_leading_count + unmasked_count and iter_num <= masked_leading_count + unmasked_count + masked_trailing_count)
            )

            # Wait for S
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state, peak_S_consumer_status)

            peak_mma_compute_dP_status = mma_compute_dP_pipeline.consumer_try_wait(mma_compute_dP_consumer_state)
            cute.copy(tiled_t2r, tTR_tS, tTR_rS)
            # Fence async TMEM load before release. WG0 signals immediately after
            # reading S0 so the MMA warp may reuse tStS0 as WG1's dP slot.
            cute.arch.fence_view_async_tmem_load()
            # Release S
            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            if cutlass.const_expr(wg_idx == 0):
                self.dS_sync_barrier_compute0.arrive()

            # Wait for dP
            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state, peak_mma_compute_dP_status)
            # Compute dS = dsoftmax(P, dP, sum_OdO)
            cute.copy(tiled_t2r, tTR_tDP, tTR_rDP)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(wg_idx == 1):
                if iter_count > 1:
                    self.dS_sync_barrier_compute1.arrive()
            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)

            if is_masked_tile:
                fmha_masks.FusedMask.apply_mask(
                    self.mask_type,
                    tTR_rS,
                    tTR_cS,
                    Q,
                    K,
                    window_size_left,
                    window_size_right,
                    lambda index_q, index_k: (
                        index_q + blk_coord_q_mask * (self.tile_shape_Q // self.use_2cta_divisor),
                        index_k + iter_index * self.tile_shape_K,
                    ),
                )

            log2_e = Float32(math.log2(math.e))
            softmax_scale_log2_e = scale_softmax * log2_e
            for i in cutlass.range(0, cute.size(tTR_rS), 2, unroll_full=True):
                tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rS[i], tTR_rS[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    (lse, lse),
                )

                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                tTR_rS[i + 1] = cute.math.exp2(tTR_rS[i + 1], fastmath=True)

            # Read sum_OdO directly from global memory (q_global_idx computed earlier for LSE)
            for i in cutlass.range(0, cute.size(tTR_rDP), 2, unroll_full=True):
                tTR_rDP[i], tTR_rDP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rDP[i], tTR_rDP[i + 1]),
                    (sum_OdO_val, sum_OdO_val),
                )
                tTR_rDP[i], tTR_rDP[i + 1] = cute.arch.mul_packed_f32x2((tTR_rDP[i], tTR_rDP[i + 1]), (tTR_rS[i], tTR_rS[i + 1]))

            peak_dS_producer_status = compute_mma_dS_pipeline.producer_try_acquire(compute_mma_dS_producer_state)
            if cutlass.const_expr(self.online_ds_scale):
                group_amax_0 = Float32(0.0)
                group_amax_1 = Float32(0.0)
                for i in cutlass.range_constexpr(32):
                    value_0 = tTR_rDP[i]
                    value_1 = tTR_rDP[i + 32]
                    group_amax_0 = cute.arch.fmax(group_amax_0, cute.arch.fmax(value_0, -value_0))
                    group_amax_1 = cute.arch.fmax(group_amax_1, cute.arch.fmax(value_1, -value_1))
                dS_row = cute.get(tTR_cDP[0], mode=[0])
                compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state, peak_dS_producer_status)
                dS_scale_0, inv_scale_0 = cute_common.cvt_amax_to_e8m0_rp(group_amax_0)
                dS_scale_1, inv_scale_1 = cute_common.cvt_amax_to_e8m0_rp(group_amax_1)
                dS_group = (dp_idx // 64) * 2
                cp_scale_tile = cute.make_tensor(
                    sDS_scale_exchange.iterator,
                    cute.make_layout((32, 4, 4), stride=(16, 4, 1)),
                )
                cp_row = dS_row % 32
                cp_col = dS_row // 32
                cp_scale_tile[cp_row, cp_col, dS_group] = dS_scale_0
                cp_scale_tile[cp_row, cp_col + 2, dS_group] = dS_scale_0
                cp_scale_tile[cp_row, cp_col, dS_group + 1] = dS_scale_1
                cp_scale_tile[cp_row, cp_col + 2, dS_group + 1] = dS_scale_1

                tTR_rdS_normalized = cute.make_rmem_tensor_like(tTR_rDP)
                for i in cutlass.range_constexpr(0, 32, 2):
                    tTR_rdS_normalized[i], tTR_rdS_normalized[i + 1] = cute.arch.mul_packed_f32x2(
                        (tTR_rDP[i], tTR_rDP[i + 1]),
                        (inv_scale_0, inv_scale_0),
                    )
                    tTR_rdS_normalized[i + 32], tTR_rdS_normalized[i + 33] = cute.arch.mul_packed_f32x2(
                        (tTR_rDP[i + 32], tTR_rDP[i + 33]),
                        (inv_scale_1, inv_scale_1),
                    )
                tTR_rdST = cute_common.quantize(tTR_rdS_normalized, 4, LOW_PRECISION_TYPE)
                cute.arch.fence_proxy("async.shared", space="cta")
                if cutlass.const_expr(wg_idx == 0):
                    self.dS_scale_exchange_barrier_0.arrive_and_wait()
                    self.dS_scale_exchange_barrier_0.arrive_and_wait()
                else:
                    self.dS_scale_exchange_barrier_1.arrive_and_wait()
                    self.dS_scale_exchange_barrier_1.arrive_and_wait()
            else:
                tTR_rdST = cute_common.quantize(tTR_rDP, 4, LOW_PRECISION_TYPE)

            # Write dS to per-warp smem stages using dS_stage_offset + producer state index
            sdS_slice = sDS[None, None, None, compute_mma_dS_producer_state.index + dS_stage_offset]
            sdS_slice_divided = cute.logical_divide(sdS_slice, ((32, None), None, 2))
            sdS_slice_warp = cute.coalesce(sdS_slice_divided[(((None, warp_rank % 2), None), 0, (None, warp_rank // 2))])
            sdS_slice_thread = cute.coalesce(sdS_slice_warp[tidx % 32, None])

            tTR_rdST_coalesced = cute.coalesce(tTR_rdST)

            assert cute.size(tTR_rdST_coalesced) == cute.size(sdS_slice_thread)
            if cutlass.const_expr(not self.online_ds_scale):
                compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state, peak_dS_producer_status)
            cute.autovec_copy(tTR_rdST_coalesced, sdS_slice_thread)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            # 1-stage pipeline: advance once
            mma_compute_S_consumer_state.advance()

            # Release dP — 1-stage pipeline
            mma_compute_dP_consumer_state.advance()

            iter_count -= 1
            iter_index += 1
            if iter_index == iter_end:
                iter_index = iter_start

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        dQ: cute.Tensor,
        tdQtdQ: cute.Tensor,
        scale_softmax: Float32,
        # (mma_compute_dQ_pipeline, mma_compute_dQ_consumer_state)
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        Q, _, D, HB = problem_shape
        blk_coord_q, _, _, blk_coord_batch = blk_coord
        mma_compute_dQ_pipeline, mma_compute_dQ_consumer_state = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )

        mdQ = cute.make_tensor(
            dQ.iterator + cute.assume(blk_offset[0] * dQ.stride[0], divby=64),
            cute.make_layout((Q, self.tile_shape_dQ_K, HB), stride=dQ.stride),
        )
        # Use full mma_tiler for local_tile - matches tmem tensor shape
        gdQ = cute.local_tile(mdQ, (self.dSK_cta_tiler[0], self.dSK_cta_tiler[1]), (None, None, None))
        gdQ = gdQ[None, None, blk_coord_q, 0, blk_coord_batch]
        # Use full mma_tiler for identity tensor - matches tmem tensor shape
        cdQ = cute.domain_offset(
            (blk_coord_q * self.dSK_cta_tiler[0], 0),
            cute.make_identity_tensor((self.dSK_cta_tiler[0], self.dSK_cta_tiler[1])),
        )

        dp_idx = tidx % 128

        tdQtdQ = tdQtdQ[(None, None), 0, 0]
        (
            tiled_t2r_dQ,
            tTR_tdQ,
            tTR_rdQ,
            tTR_gdQ,
            tTR_cdQ,
        ) = cute_common.epilogue_tmem_copy_and_partition(load_op, tdQtdQ, cdQ, gdQ, dp_idx, self.acc_dtype)

        mma_compute_dQ_pipeline.consumer_wait(mma_compute_dQ_consumer_state)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)
        for i in cutlass.range(cute.size(tTR_rdQ), unroll_full=True):
            tTR_rdQ[i] = scale_softmax * tTR_rdQ[i]

        cute.arch.fence_view_async_tmem_load()

        cute_common.store(self, tTR_gdQ, tTR_rdQ, tTR_cdQ, (Q, D))

        mma_compute_dQ_pipeline.consumer_release(mma_compute_dQ_consumer_state)
        mma_compute_dQ_consumer_state.advance()

    def make_and_init_load_mma_K_pipeline(self, load_mma_K_mbar_ptr, cluster_layout_vmnk):
        # Payload bytes only (both CTAs' K / K^T / V halves land on the leader's
        # barrier); scale-factor bytes are tracked per CTA on sf_landed and the
        # stage-full barrier additionally collects both load warps' SF arrivals.
        kt_tx_multiplier = 2 * self.d_halves
        tx_count = 2 * self.tma_copy_K_bytes + kt_tx_multiplier * self.tma_copy_KT_bytes + 2 * self.tma_copy_V_bytes
        return cute_common.make_tma_umma_pipeline(
            load_mma_K_mbar_ptr,
            self.load_mma_K_stage,
            tx_count,
            cluster_layout_vmnk,
            len([self.load_warp_id]),
            1,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr, cluster_layout_vmnk):
        return cute_common.make_pipeline_umma_async(self, mma_compute_S_mbar_ptr, self.mma_compute_S_stage, cluster_layout_vmnk)

    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr, cluster_layout_vmnk):
        return cute_common.make_pipeline_umma_async(self, mma_compute_dP_mbar_ptr, self.mma_compute_dP_stage, cluster_layout_vmnk)

    def make_and_init_mma_compute_dQ_pipeline(self, mma_compute_dQ_mbar_ptr, cluster_layout_vmnk):
        return cute_common.make_pipeline_umma_async(self, mma_compute_dQ_mbar_ptr, self.mma_compute_dQ_stage, cluster_layout_vmnk)

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr, cluster_layout_vmnk):
        return cute_common.make_async_umma_pipeline(
            compute_mma_dS_mbar_ptr,
            self.compute_mma_dS_stage,
            cluster_layout_vmnk,
            self.num_compute_0_warps * self.threads_per_warp * cute.size(cluster_layout_vmnk, mode=[0]),
            len([self.mma_warp_id]),
        )
