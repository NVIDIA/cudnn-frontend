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
SM100 D256 MXFP8 SDPA backward: the fused dK/dV kernel (2-CTA, online or fixed dS scale).

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


class BlackwellFmhaBackwardDKDV256:
    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        varlen: bool,
        mask_type: fmha_masks.MaskEnum,
        is_persistent: bool = False,
        online_ds_scale: bool = True,
        p_scale_log2: int = 8,
        h_r_split: int = 1,
        s_q_split: int = 1,
    ):
        """Fused dK/dV kernel for d=256 MXFP8 (2-CTA block-scaled MMA); see the module docstring."""
        mma_tiler = self._setup_specialization(element_dtype, acc_dtype, mma_tiler)
        self._setup_mma_tilers(mma_tiler)
        self._setup_warp_topology_and_barriers(varlen, mask_type, is_persistent, online_ds_scale)
        # P (softmax probability, in (0, 1]) is quantized to E4M3 for the dV MMA
        # with ONE fixed power-of-two scale: P_q = P * 2**p_scale_log2 and the
        # matching E8M0 descale byte 127 - p_scale_log2. The source repo
        # hard-coded scale 1 (byte 127); cuDNN's MXFP8 backward convention (and
        # test/python/sdpa/mxfp8_ref.py) uses 2**8, which keeps probabilities
        # down to ~2^-17 representable instead of flushing below 2^-9.
        if not (0 <= int(p_scale_log2) <= 126):
            raise ValueError(f"p_scale_log2 must be in [0, 126]; got {p_scale_log2}")
        self.p_scale_log2 = int(p_scale_log2)
        # Q-head-group split. The grid is (kv tiles, h_k, b) and one cluster
        # walks every Q head of its KV head, so at few KV heads (GQA with 1-2
        # KV heads, small batch) most SMs idle: 2 KV heads at S=8192 is 64
        # clusters on 148 SMs. With h_r_split = S the grid becomes
        # (kv tiles, h_k * S, b); cluster (h_k, s) handles Q heads
        # [s * h_r/S, (s+1) * h_r/S) and stores its dK/dV to PARTIAL slot
        # h_k * S + s of a [B, S_kv, h_k * S, D] buffer (the host passes that
        # buffer as dK/dV); a fixed-order fold onto the KV heads follows
        # (kernels/bprop_chain_f16_sm120.dkv_reduce_host). h_r must be a
        # multiple of h_r_split; 1 = the original single-pass behaviour.
        if int(h_r_split) < 1:
            raise ValueError(f"h_r_split must be >= 1; got {h_r_split}")
        if int(h_r_split) > 1 and is_persistent:
            raise ValueError("h_r_split > 1 is not implemented for the persistent scheduler")
        self.h_r_split = int(h_r_split)
        # Q-sequence split, same mechanism along the other axis: cluster slice s
        # walks Q tiles [c0, c1) of its KV tile's trip range, cut into s_q_split
        # near-equal chunks (per KV tile, so causal ranges stay balanced), and
        # stores to its own partial slot. Slots: h_k * h_r_split * s_q_split per
        # batch, slot = (h_k * h_r_split + h_slice) * s_q_split + q_slice. A chunk
        # that ends up empty (fewer Q tiles than slices) is written as zeros.
        if int(s_q_split) < 1:
            raise ValueError(f"s_q_split must be >= 1; got {s_q_split}")
        if int(s_q_split) > 1 and is_persistent:
            raise ValueError("s_q_split > 1 is not implemented for the persistent scheduler")
        self.s_q_split = int(s_q_split)
        self.n_split = self.h_r_split * self.s_q_split

    def _setup_specialization(self, element_dtype, acc_dtype, mma_tiler):
        """Normalize tile orientation and set fixed instruction/type policy."""
        mma_tiler = (
            mma_tiler[1],
            mma_tiler[0],
            mma_tiler[2],
        )
        # NOTE: hereafter, the mma_tiler has already been set as (M, N, K) w.r.t the Transposed S

        self.head_dim = mma_tiler[2]
        if self.head_dim != 256:
            raise ValueError("fmha_bwd_dKdV only supports head dimension 256")
        self.use_2cta_instrs = True
        self.use_2cta_divisor = 2
        self.cluster_shape_mn = (2, 1)
        self.cta_group = tcgen05.CtaGroup.TWO

        self.sf_dtype = SF_DTYPE
        self.sf_vec_size = SF_VEC_SIZE
        self.element_dtype = element_dtype
        self.acc_dtype = acc_dtype
        return mma_tiler

    def _setup_mma_tilers(self, mma_tiler):
        """Derive MMA and CTA-local tile shapes."""
        self.cta_tiler = (
            mma_tiler[0] // 2 if self.use_2cta_instrs else mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.tile_shape_Q = mma_tiler[1]
        self.tile_shape_K = mma_tiler[0]
        self.tile_shape_dKdV_K = mma_tiler[2]

        self.CTA_shape_Q = mma_tiler[1]
        self.CTA_shape_K = mma_tiler[0] // 2
        # For S, (128, 64, 128)
        self.KQ_mma_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.KQ_mma_tiler_sfb = (
            max(self.KQ_mma_tiler[0] // 2 if self.use_2cta_instrs else self.KQ_mma_tiler[0], 128),
            cute.round_up(self.KQ_mma_tiler[1], 128),
            self.KQ_mma_tiler[2],
        )
        cta_group_size = 2 if self.use_2cta_instrs else 1
        self.KQ_mma_tiler_sfa = (
            self.KQ_mma_tiler[0] * cta_group_size,
            self.KQ_mma_tiler[1],
            self.KQ_mma_tiler[2],
        )
        # For dP
        self.VDO_mma_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.VDO_mma_tiler_sfb = (
            max(self.VDO_mma_tiler[0] // 2 if self.use_2cta_instrs else self.VDO_mma_tiler[0], 128),
            cute.round_up(self.VDO_mma_tiler[1], 128),
            self.VDO_mma_tiler[2],
        )
        self.VDO_mma_tiler_sfa = (
            self.VDO_mma_tiler[0] * cta_group_size,
            self.VDO_mma_tiler[1],
            self.VDO_mma_tiler[2],
        )

        # For dV
        self.PdO_mma_tiler = (
            mma_tiler[0],
            mma_tiler[2],
            mma_tiler[1],
        )
        self.PdO_mma_tiler_sfb = (
            max(self.PdO_mma_tiler[0] // self.cluster_shape_mn[0], 128),
            mma_tiler[2],
            cute.round_up(mma_tiler[1], 128),
        )
        # For dK
        self.dSQ_mma_tiler = (
            mma_tiler[0],
            mma_tiler[2],
            mma_tiler[1],
        )
        self.dSQ_mma_tiler_sfb = (
            max(self.dSQ_mma_tiler[0] // self.cluster_shape_mn[0], 128),
            mma_tiler[2],
            cute.round_up(mma_tiler[1], 128),
        )

        # CTA-local tile shapes for 2-CTA mode (M dimension divided by number of CTAs)
        # These are used for identity tensors and CTA-specific operations
        cta_m_divisor = 2 if self.use_2cta_instrs else 1
        self.KQ_cta_tiler = (
            self.KQ_mma_tiler[0] // cta_m_divisor,
            self.KQ_mma_tiler[1],
            self.KQ_mma_tiler[2],
        )
        self.VDO_cta_tiler = (
            self.VDO_mma_tiler[0] // cta_m_divisor,
            self.VDO_mma_tiler[1],
            self.VDO_mma_tiler[2],
        )

        self.PdO_cta_tiler = (
            self.PdO_mma_tiler[0] // cta_m_divisor,
            self.PdO_mma_tiler[1],
            self.PdO_mma_tiler[2],
        )
        self.dSQ_cta_tiler = (
            self.dSQ_mma_tiler[0] // cta_m_divisor,
            self.dSQ_mma_tiler[1],
            self.dSQ_mma_tiler[2],
        )

    def _setup_warp_topology_and_barriers(self, varlen, mask_type, is_persistent, online_ds_scale):
        """Configure runtime specialization, warp roles, registers, and barriers."""
        self.varlen = varlen
        self.mask_type = mask_type

        # Mask CTA tiler with (Q_dim, K_dim, d) ordering to match WINDOW_MASK_BWD convention.
        # self.cta_tiler is (K//2, Q, d), but WINDOW_MASK_BWD expects tile_shape[0]=Q, tile_shape[1]=K.
        self.mask_cta_tiler = (self.cta_tiler[1], self.cta_tiler[0], self.cta_tiler[2])

        self.is_persistent = is_persistent
        self.online_ds_scale = online_ds_scale

        if is_persistent:
            # In persistent mode, repurpose empty warp 10 as CLC sched warp
            self.sched_warp_id = 10
            self.empty_warp_id = (11,)
            # 8 compute + mma + load + sched + 1 empty = 12 warps
        else:
            self.empty_warp_id = (10, 11)
            # 8 compute + mma + load + 2 empty = 12 warps

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 12
        cute_common.init_common_config(self)
        self.num_regs_compute = 192

        # CLC pipeline configuration
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16  # Fixed by CLC hardware (128-bit opaque response)

        # Persistent tile boundary barrier: sync load + MMA + compute warps (10 warps)
        # before CLC consumer advances to next tile. Excludes sched and empty warps.
        self.persistent_tile_barrier = pipeline.NamedBarrier(
            barrier_id=9,
            num_threads=10 * self.threads_per_warp,
        )
        self.dS_scale_exchange_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )

    def _setup_pipeline_stages_and_sf_tilers(self):
        """Derive trace-time pipeline depths and scale-factor tile shapes."""
        self.load_mma_all_stage = 2
        self.mma_compute_KQ_stage = 1
        self.mma_compute_VDO_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 4
        self.mma_compute_dKdV_stage = 1
        self.num_prologue_iters = self.mma_compute_KQ_stage
        self.k_halves = 2
        self.VDO_mma_tiler_sfv_load = (
            self.VDO_mma_tiler_sfa[0],
            self.VDO_mma_tiler_sfa[1],
            128,
        )
        self.VDO_mma_tiler_sfv_smem = (
            self.VDO_mma_tiler_sfb[0],
            self.VDO_mma_tiler_sfb[1],
            128,
        )
        self.SFV_load_mma_K_stage = self.k_halves
        self.KQ_mma_tiler_sfk_load = (
            self.KQ_mma_tiler_sfa[0],
            self.KQ_mma_tiler_sfa[1],
            128,
        )
        self.KQ_mma_tiler_sfk_smem = (
            self.KQ_mma_tiler_sfb[0],
            self.KQ_mma_tiler_sfb[1],
            128,
        )
        self.KQ_mma_tiler_sfq_load = (
            self.KQ_mma_tiler_sfb[0],
            self.KQ_mma_tiler_sfb[1],
            128,
        )
        self.VDO_mma_tiler_sfdo_load = (
            self.VDO_mma_tiler_sfb[0],
            self.VDO_mma_tiler_sfb[1],
            128,
        )
        self.SFK_load_stage = self.k_halves
        self.SFQ_load_stage = self.load_mma_all_stage * self.k_halves
        self.SFDO_load_stage = self.load_mma_all_stage * self.k_halves

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        num_warp_groups: Int32,
        wg_idx: Int32,
    ) -> cute.Tensor:
        """View a per-warp-group TMEM fragment for the calling warp group."""
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 1):
            p = cute.composition(
                t,
                cute.make_layout(((num_warp_groups, cute.size(t) // num_warp_groups),)),
            )
            ret = p[(wg_idx, None)]
        elif cutlass.const_expr(cute.rank(t.layout) == 2):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        (num_warp_groups, cute.size(t, mode=[1]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, (wg_idx, None)]
        elif cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, (wg_idx, None)]
        else:
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        t.shape[2],
                        (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, None, (wg_idx, None)]
        return ret

    @cute.jit
    def __call__(
        self,
        # [seq_max_q, seq_max_k, d, (h_r, h_k), b]
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        Q: cute.Tensor,
        Q_MN: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        SF_Q: cute.Tensor,
        SF_QT: cute.Tensor,
        SF_K: cute.Tensor,
        SF_V: cute.Tensor,
        SF_DO: cute.Tensor,
        SF_DOT: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        dO_MN: cute.Tensor,
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
        """JIT entry: build operand views, MMA atoms, TMA descriptors and launch the row-dot prologue and the dK/dV kernel."""
        q_seq_max, k_seq_max, d, hb = problem_shape
        h, b = hb
        h_r, h_k = h
        Q = make_q_head_batch_tensor(Q, hb, self.varlen)
        K = make_kv_head_batch_tensor(K, hb, self.varlen)
        V = make_kv_head_batch_tensor(V, hb, self.varlen)
        O = make_q_head_batch_tensor(O, hb, self.varlen)
        dO_16bits = cute.make_tensor(dO_16bits.iterator, O.layout)
        # dK/dV: with h_r_split > 1 these are the PARTIAL buffers, h_k * h_r_split
        # slots per batch (slot = h_k * h_r_split + split); see __init__.
        hb_out = ((h_r, h_k * self.n_split), b)
        dK = make_kv_head_batch_tensor(dK, hb_out, self.varlen)
        dV = make_kv_head_batch_tensor(dV, hb_out, self.varlen)
        dO = cute.make_tensor(dO.iterator, Q.layout)
        dOT = make_transposed_tensor(dO_MN, dO.layout)
        QT = make_transposed_tensor(Q_MN, Q.layout)
        LSE = make_lse_head_batch_tensor(LSE, hb)

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()
        self.dO_major_mode = utils.LayoutEnum.from_tensor(dO).mma_major_mode()

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
        if cutlass.const_expr(self.dO_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of do is not supported")

        self._setup_pipeline_stages_and_sf_tilers()
        # compute S - using self.cta_group for 2-CTA support
        KQ_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.KQ_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        KQ_tiled_mma_smem = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.KQ_mma_tiler[:2],
            tcgen05.OperandSource.SMEM,
        )

        KQ_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,  # SFB uses ONE for partition
            self.KQ_mma_tiler_sfb[:2],
            tcgen05.OperandSource.TMEM,
        )
        KQ_tiled_mma_sfa = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.KQ_mma_tiler_sfa[:2],
            tcgen05.OperandSource.TMEM,
        )

        # compute dP - using self.cta_group for 2-CTA support
        VDO_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.VDO_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        VDO_tiled_mma_smem = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.VDO_mma_tiler[:2],
            tcgen05.OperandSource.SMEM,
        )

        VDO_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,  # SFB uses ONE for partition
            self.VDO_mma_tiler_sfb[:2],
            tcgen05.OperandSource.TMEM,
        )
        VDO_tiled_mma_sfa = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.VDO_mma_tiler_sfa[:2],
            tcgen05.OperandSource.TMEM,
        )

        # dK
        dSQ_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.dSQ_mma_tiler[:2],
        )
        dSQ_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,  # SFB uses ONE for partition
            self.dSQ_mma_tiler_sfb[:2],
        )

        # dV
        PdO_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.PdO_mma_tiler[:2],
        )
        PdO_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            LOW_PRECISION_TYPE,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,  # SFB uses ONE for partition
            self.PdO_mma_tiler_sfb[:2],
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (KQ_tiled_mma.thr_id.shape,),
        )
        cluster_layout_vmnk_sfb = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (KQ_tiled_mma_sfb.thr_id.shape,),
        )

        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            LOW_PRECISION_TYPE,
            self.load_mma_all_stage,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            LOW_PRECISION_TYPE,
            self.load_mma_all_stage,
        )
        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            LOW_PRECISION_TYPE,
            1,
        )
        sfQ_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            KQ_tiled_mma_sfb,
            self.KQ_mma_tiler_sfq_load,
            self.sf_vec_size,
            self.SFQ_load_stage,
        )
        sfQ_smem_layout_staged = cute_common.expand_last_SF_stride(sfQ_smem_layout_staged)

        sSFQ_mn_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            dSQ_tiled_mma_sfb,
            self.dSQ_mma_tiler_sfb,
            self.sf_vec_size,
            self.load_mma_all_stage,
        )
        sSFQ_mn_smem_layout_staged = cute_common.expand_last_SF_stride(sSFQ_mn_smem_layout_staged)

        # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
        sfK_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            KQ_tiled_mma_sfb,
            self.KQ_mma_tiler_sfk_smem,
            self.sf_vec_size,
            self.SFK_load_stage,
        )
        sfK_smem_layout_staged = cute_common.expand_last_SF_stride(sfK_smem_layout_staged)

        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            VDO_tiled_mma,
            self.VDO_mma_tiler,
            LOW_PRECISION_TYPE,
            self.load_mma_all_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            LOW_PRECISION_TYPE,
            self.load_mma_all_stage,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_a(
            VDO_tiled_mma,
            self.VDO_mma_tiler,
            LOW_PRECISION_TYPE,
            1,
        )
        SFDO_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            VDO_tiled_mma_sfb,
            self.VDO_mma_tiler_sfdo_load,
            self.sf_vec_size,
            self.SFDO_load_stage,
        )
        SFDO_smem_layout_staged = cute_common.expand_last_SF_stride(SFDO_smem_layout_staged)

        SFDO_mn_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            PdO_tiled_mma_sfb,
            self.PdO_mma_tiler_sfb,
            self.sf_vec_size,
            self.load_mma_all_stage,
        )
        SFDO_mn_smem_layout_staged = cute_common.expand_last_SF_stride(SFDO_mn_smem_layout_staged)

        SFV_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            VDO_tiled_mma_sfb,
            self.VDO_mma_tiler_sfv_smem,
            self.sf_vec_size,
            self.SFV_load_mma_K_stage,
        )
        SFV_smem_layout_staged = cute_common.expand_last_SF_stride(SFV_smem_layout_staged)

        dS_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            LOW_PRECISION_TYPE,
            self.compute_mma_dS_stage,
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_a(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            LOW_PRECISION_TYPE,
            self.compute_mma_P_stage,
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        QT_smem_layout = cute.select(QT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_QT, tma_tensor_QT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            QT,
            QT_smem_layout,
            self.dSQ_mma_tiler,
            dSQ_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.VDO_mma_tiler,
            VDO_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        dOT_smem_layout = cute.select(dOT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dOT, tma_tensor_dOT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, dOT, dOT_smem_layout, self.PdO_mma_tiler, PdO_tiled_mma, cluster_layout_vmnk.shape
        )

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.VDO_mma_tiler,
            VDO_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, Q_smem_layout)
        self.tma_copy_K_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, K_smem_layout)
        self.tma_copy_QT_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, QT_smem_layout)
        self.tma_copy_V_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, V_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, dO_smem_layout)
        self.tma_copy_dOT_bytes = cute.size_in_bytes(LOW_PRECISION_TYPE, dOT_smem_layout)

        # Q shape for SFB: use Q.shape with doubled L for merged s0/s1
        sfQ_shape = (cute.round_up(q_seq_max, 128), Q.shape[1], cute.size(Q.shape[2]) * 2)
        sfQ_layout = blockscaled_utils.tile_atom_to_shape_SF(sfQ_shape, self.sf_vec_size)
        SF_Q = cute.make_tensor(SF_Q.iterator, sfQ_layout)

        K_shape_sfa = (
            cute.round_up(k_seq_max, 128) * 2,
            K.shape[1],
            cute.size(K.shape[2]) * 2,
        )
        sfK_layout = blockscaled_utils.tile_atom_to_shape_SF(K_shape_sfa, self.sf_vec_size)
        SF_K = cute.make_tensor(SF_K.iterator, sfK_layout)

        V_shape_sfa = (
            cute.round_up(k_seq_max, 128) * 2,
            V.shape[1],
            cute.size(V.shape[2]) * 2,
        )
        sfV_layout = blockscaled_utils.tile_atom_to_shape_SF(V_shape_sfa, self.sf_vec_size)
        SF_V = cute.make_tensor(SF_V.iterator, sfV_layout)

        sfdO_shape = (
            cute.round_up(q_seq_max, 128),
            dO.shape[1],
            cute.size(dO.shape[2]) * 2,
        )
        sfdO_layout = blockscaled_utils.tile_atom_to_shape_SF(sfdO_shape, self.sf_vec_size)
        SF_DO = cute.make_tensor(SF_DO.iterator, sfdO_layout)

        # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
        QT_shape = (Q.shape[1], q_seq_max, cute.size(Q.shape[2]) * 2)
        sfQ_mn_layout = blockscaled_utils.tile_atom_to_shape_SF(QT_shape, self.sf_vec_size)
        SF_Q_mn = cute.make_tensor(SF_QT.iterator, sfQ_mn_layout)

        dOT_shape = (dO.shape[1], q_seq_max, cute.size(dO.shape[2]) * 2)
        sfdO_mn_layout = blockscaled_utils.tile_atom_to_shape_SF(dOT_shape, self.sf_vec_size)
        SF_DO_mn = cute.make_tensor(SF_DOT.iterator, sfdO_mn_layout)

        # Setup TMA for scale factors with correct ops for 2-CTA:
        # - SFA (A operand's SF): use cluster_shape_to_tma_atom_A + cluster_layout_vmnk
        # - SFB (B operand's SF): use cluster_shape_to_tma_atom_SFB + cluster_layout_vmnk_sfb

        sfa_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # SFB (B operand's SF: sfQ, sfDO, sfQ_mn, sfDO_mn) uses multicast
        sfb_mcast_op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        sfK_smem_layout = cute.slice_(sfK_smem_layout_staged, (None, None, None, 0, 0))
        tma_atom_sfK, tma_tensor_sfK = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            SF_K,
            sfK_smem_layout,
            self.KQ_mma_tiler_sfk_load,
            KQ_tiled_mma_sfa,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfQ_smem_layout = cute.slice_(sfQ_smem_layout_staged, (None, None, None, 0, 0))
        tma_atom_sfQ, tma_tensor_sfQ = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_mcast_op,
            SF_Q,
            sfQ_smem_layout,
            self.KQ_mma_tiler_sfq_load,
            KQ_tiled_mma_sfb,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfDO_smem_layout = cute.slice_(SFDO_smem_layout_staged, (None, None, None, 0, 0))
        tma_atom_sfDO, tma_tensor_sfDO = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_mcast_op,
            SF_DO,
            sfDO_smem_layout,
            self.VDO_mma_tiler_sfdo_load,
            VDO_tiled_mma_sfb,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sSFQ_mn_smem_layout = cute.slice_(sSFQ_mn_smem_layout_staged, (None, None, None, 0, 0))
        tma_atom_sfQ_mn, tma_tensor_sfQ_mn = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_mcast_op,
            SF_Q_mn,
            sSFQ_mn_smem_layout,
            self.dSQ_mma_tiler_sfb,
            dSQ_tiled_mma_sfb,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfV_smem_layout = cute.slice_(SFV_smem_layout_staged, (None, None, None, 0, 0))
        tma_atom_sfV, tma_tensor_sfV = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            SF_V,
            sfV_smem_layout,
            self.VDO_mma_tiler_sfv_load,
            VDO_tiled_mma_sfa,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        SFDO_mn_smem_layout = cute.slice_(SFDO_mn_smem_layout_staged, (None, None, None, 0, 0))
        tma_atom_sfDO_mn, tma_tensor_sfDO_mn = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_mcast_op,
            SF_DO_mn,
            SFDO_mn_smem_layout,
            self.PdO_mma_tiler_sfb,
            PdO_tiled_mma_sfb,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        self.tma_copy_sfQ_bytes = cute.size_in_bytes(self.sf_dtype, sfQ_smem_layout)
        self.tma_copy_sfK_bytes = cute.size_in_bytes(self.sf_dtype, sfK_smem_layout)
        self.tma_copy_sfQ_mn_bytes = cute.size_in_bytes(self.sf_dtype, sSFQ_mn_smem_layout)
        self.tma_copy_sfV_bytes = cute.size_in_bytes(self.sf_dtype, sfV_smem_layout)
        self.tma_copy_sfdO_bytes = cute.size_in_bytes(self.sf_dtype, sfDO_smem_layout)
        self.tma_copy_sfdO_mn_bytes = cute.size_in_bytes(self.sf_dtype, SFDO_mn_smem_layout)

        LSE_smem_layout = cute.make_layout((self.cta_tiler[1], self.load_mma_all_stage))
        sum_OdO_smem_layout = LSE_smem_layout

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_KQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_all_stage * 2]
            load_mma_KQ_aux_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_all_stage * 2]
            load_mma_VDO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_all_stage * 2]
            mma_compute_KQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_KQ_stage * 2]  # 0x460, 0x468
            mma_compute_VDO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_VDO_stage * 2]  # 0x470, 0x478
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_P_stage * 2]
            mma_compute_dK_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dKdV_stage * 2]
            mma_compute_dV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dKdV_stage * 2]
            tmem_holding_buf: cutlass.Int32
            # CLC dynamic scheduler barriers and response (persistent mode)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            # For 2-CTA tmem deallocation barrier
            tmem_dealloc_mbar_ptr: cutlass.Int64
            # CLC response: 16 bytes (128-bit opaque) written by hardware
            clc_response_ptr: cute.struct.MemRange[cutlass.Int32, 4]
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(K_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # NOTE: Need to load the transposed data in 2-cta mode
            # as the partitioning will be different
            sQT: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(QT_smem_layout_staged)],
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
            sdOT: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(dOT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sDS: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(dS_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Dedicated exchange storage: dKdV keeps four dS pipeline stages,
            # so reusing stage 0 could overwrite a stage still consumed by MMA.
            sDS_scale_exchange: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, 512],
                128,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[LOW_PRECISION_TYPE, cute.cosize(P_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sSFQ: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfQ_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfK_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFQ_mn: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sSFQ_mn_smem_layout_staged)],
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
            sSFDO_mn: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(SFDO_mn_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                self.buffer_align_bytes,
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        sum_OdO, scaled_LSE, _ = cute_common.get_workspace_tensor(self, problem_shape, workspace, self.acc_dtype, needs_dq_acc=False)

        tma_lse_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_LSE, tma_tensor_LSE = cpasync.make_tiled_tma_atom(tma_lse_op, scaled_LSE, LSE_smem_layout, (self.cta_tiler[1],))
        tma_atom_sum_OdO, tma_tensor_sum_OdO = cpasync.make_tiled_tma_atom(tma_lse_op, sum_OdO, sum_OdO_smem_layout, (self.cta_tiler[1],))

        self.tma_copy_LSE_bytes = cute.size_in_bytes(self.acc_dtype, cute.select(LSE_smem_layout, mode=[0]))
        self.tma_copy_sum_OdO_bytes = self.tma_copy_LSE_bytes

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
            cute.shape((k_seq_max, d, ((1, h_k * self.n_split), b))),
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
                bwd_grid,  # (K_tiles_aligned, h_k, b) = total CTA tile space
                cluster_shape_mnk,
            )
            bwd_grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

        self.bwd(
            KQ_tiled_mma,
            VDO_tiled_mma,
            dSQ_tiled_mma,
            PdO_tiled_mma,
            KQ_tiled_mma_smem,
            VDO_tiled_mma_smem,
            KQ_tiled_mma_sfb,
            VDO_tiled_mma_sfb,
            dSQ_tiled_mma_sfb,
            PdO_tiled_mma_sfb,
            KQ_tiled_mma_sfa,
            VDO_tiled_mma_sfa,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_QT,
            tma_tensor_QT,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dOT,
            tma_tensor_dOT,
            tma_atom_sfQ,
            tma_tensor_sfQ,
            tma_atom_sfK,
            tma_tensor_sfK,
            tma_atom_sfQ_mn,
            tma_tensor_sfQ_mn,
            tma_atom_sfV,
            tma_tensor_sfV,
            tma_atom_sfDO,
            tma_tensor_sfDO,
            tma_atom_sfDO_mn,
            tma_tensor_sfDO_mn,
            dK,
            dV,
            tma_atom_LSE,
            tma_tensor_LSE,
            scale_softmax,
            tma_atom_sum_OdO,
            tma_tensor_sum_OdO,
            problem_shape,
            cumulative_s_q,
            cumulative_s_k,
            window_size_left,
            window_size_right,
            sfQ_smem_layout_staged,
            sSFQ_mn_smem_layout_staged,
            sfK_smem_layout_staged,
            SFV_smem_layout_staged,
            SFDO_smem_layout_staged,
            SFDO_mn_smem_layout_staged,
            Q_smem_layout_staged,
            QT_smem_layout_staged,
            K_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dOT_smem_layout_staged,
            dS_smem_layout_staged,
            P_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
            cluster_layout_vmnk,
            cluster_layout_vmnk_sfb,
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
        KQ_tiled_mma: cute.TiledMma,
        VDO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        KQ_tiled_mma_smem: cute.TiledMma,
        VDO_tiled_mma_smem: cute.TiledMma,
        KQ_tiled_mma_sfb: cute.TiledMma,
        VDO_tiled_mma_sfb: cute.TiledMma,
        dSQ_tiled_mma_sfb: cute.TiledMma,
        PdO_tiled_mma_sfb: cute.TiledMma,
        KQ_tiled_mma_sfa: cute.TiledMma,
        VDO_tiled_mma_sfa: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_QT: cute.CopyAtom,
        QT_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dOT: cute.CopyAtom,
        dOT_in: cute.Tensor,
        tma_atom_sfQ: cute.CopyAtom,
        SFQ_in: cute.Tensor,
        tma_atom_sfK: cute.CopyAtom,
        SFK_in: cute.Tensor,
        tma_atom_sfQ_mn: cute.CopyAtom,
        SFQ_mn_in: cute.Tensor,
        tma_atom_sfV: cute.CopyAtom,
        SFV_in: cute.Tensor,
        tma_atom_sfDO: cute.CopyAtom,
        SFDO_in: cute.Tensor,
        tma_atom_sfDO_mn: cute.CopyAtom,
        SFDO_mn_in: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tma_atom_LSE: cute.CopyAtom,
        LSE: cute.Tensor,
        scale_softmax: Float32,
        tma_atom_sum_OdO: cute.CopyAtom,
        sum_OdO: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sfQ_smem_layout_staged: cute.Layout,
        sSFQ_mn_smem_layout_staged: cute.Layout,
        sfK_smem_layout_staged: cute.Layout,
        SFV_smem_layout_staged: cute.Layout,
        SFDO_smem_layout_staged: cute.Layout,
        SFDO_mn_smem_layout_staged: cute.Layout,
        Q_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_vmnk_sfb: cute.Layout,
        tile_sched_params: Union[utils.ClcDynamicPersistentTileSchedulerParams, None],
    ):
        """The fused dK/dV kernel body: warp-specialized load / MMA / compute roles for one (KV tile, head slice, Q chunk)."""
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # For 2-CTA MMA: determine which CTA in the pair (0 or 1)
        mma_tile_coord_v = bidx % cute.size(KQ_tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Get CTA rank in cluster for 2-CTA coordination
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_vmnk_sfb.get_flat_coord(cta_rank_in_cluster)

        self.a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
        self.b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
        self.sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
        self.sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_QT)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dOT)

            cpasync.prefetch_descriptor(tma_atom_sfK)
            cpasync.prefetch_descriptor(tma_atom_sfQ)
            cpasync.prefetch_descriptor(tma_atom_sfQ_mn)
            cpasync.prefetch_descriptor(tma_atom_sfV)
            cpasync.prefetch_descriptor(tma_atom_sfDO)
            cpasync.prefetch_descriptor(tma_atom_sfDO_mn)
            cpasync.prefetch_descriptor(tma_atom_LSE)
            cpasync.prefetch_descriptor(tma_atom_sum_OdO)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_KQ_pipeline = self.make_and_init_load_mma_KQ_pipeline(
            storage.load_mma_KQ_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )
        load_mma_KQ_aux_pipeline = self.make_and_init_load_mma_KQ_aux_pipeline(
            storage.load_mma_KQ_aux_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )
        load_mma_VDO_pipeline = self.make_and_init_load_mma_VDO_pipeline(
            storage.load_mma_VDO_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )

        mma_compute_KQ_pipeline = self.make_and_init_mma_compute_KQ_pipeline(
            storage.mma_compute_KQ_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )
        mma_compute_VDO_pipeline = self.make_and_init_mma_compute_VDO_pipeline(
            storage.mma_compute_VDO_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )

        mma_compute_dK_pipeline = self.make_and_init_mma_compute_dK_pipeline(
            storage.mma_compute_dK_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(
            storage.compute_mma_P_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
        )

        # CLC dynamic persistent scheduling pipeline (persistent mode only)
        if cutlass.const_expr(self.is_persistent):
            cluster_size = cute.size(cluster_layout_vmnk)
            # Consumer threads: sched warp on CTA 0 (1 warp) +
            # all other warps on both CTAs (cluster_size * 11 warps)
            num_clc_consumer_threads = 32 * (
                1
                + cluster_size
                * (
                    1  # load
                    + 1  # mma
                    + self.num_compute_0_warps  # compute0
                    + self.num_compute_1_warps  # compute1
                    + len(self.empty_warp_id)  # empty (warp 11)
                )
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
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=False)

        self.cta_sync_barrier.arrive_and_wait()

        # setup mma
        # NOTE: need to load dOT and QT in 2-cta mode as the partitioning will be different
        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sQT = storage.sQT.get_tensor(QT_smem_layout_staged.outer, swizzle=QT_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        sdOT = storage.sdOT.get_tensor(dOT_smem_layout_staged.outer, swizzle=dOT_smem_layout_staged.inner)

        sSFQ = storage.sSFQ.get_tensor(
            sfQ_smem_layout_staged,
        )
        sSFK = storage.sSFK.get_tensor(
            sfK_smem_layout_staged,
        )

        sSFQ_mn = storage.sSFQ_mn.get_tensor(
            sSFQ_mn_smem_layout_staged,
        )
        sSFV = storage.sSFV.get_tensor(
            SFV_smem_layout_staged,
        )
        sSFDO = storage.sSFDO.get_tensor(
            SFDO_smem_layout_staged,
        )
        sSFDO_mn = storage.sSFDO_mn.get_tensor(
            SFDO_mn_smem_layout_staged,
        )

        sDS = storage.sDS.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
        sDS_scale_exchange = storage.sDS_scale_exchange.get_tensor(cute.make_layout(512))

        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)

        sLSE = storage.sLSE.get_tensor(
            LSE_smem_layout,
        )
        sSum_OdO = storage.sSum_OdO.get_tensor(
            sum_OdO_smem_layout,
        )

        # (MMA, MMA_M, MMA_K, STAGE)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)

        # K-half SMEM fragments for d=256 support.
        tSTrK_0 = KQ_tiled_mma_smem.make_fragment_A(sK[(None, None, (None, 0), None)])
        tSTrK_1 = KQ_tiled_mma_smem.make_fragment_A(sK[(None, None, (None, 1), None)])
        tSTrQ_0 = KQ_tiled_mma_smem.make_fragment_B(sQ[(None, None, (None, 0), None)])
        tSTrQ_1 = KQ_tiled_mma_smem.make_fragment_B(sQ[(None, None, (None, 1), None)])
        tDPrV_0 = VDO_tiled_mma_smem.make_fragment_A(sV[(None, None, (None, 0), None)])
        tDPrV_1 = VDO_tiled_mma_smem.make_fragment_A(sV[(None, None, (None, 1), None)])
        tdPTrdO_0 = VDO_tiled_mma_smem.make_fragment_B(sdO[(None, None, (None, 0), None)])
        tdPTrdO_1 = VDO_tiled_mma_smem.make_fragment_B(sdO[(None, None, (None, 1), None)])

        # (MMA, MMA_N, MMA_K, STAGE)
        tdPTrdO = VDO_tiled_mma.make_fragment_B(sdO)

        # (MMA, MMA_M, MMA_K, STAGE)
        tDKrDS = dSQ_tiled_mma.make_fragment_A(sDS)
        # (MMA, MMA_N, MMA_K, STAGE)
        tDKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

        tDVrP = PdO_tiled_mma.make_fragment_A(sP)
        tDVrDOT = PdO_tiled_mma.make_fragment_B(sdOT)

        # Create TmemAllocator for 2-CTA support
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id_0[0],
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        if warp_idx == self.compute_warp_id_0[0]:
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
            VDO_tiled_mma,
            self.VDO_mma_tiler,
            self.mma_compute_VDO_stage,
            self.acc_dtype,
        )
        tStS, tmem_offset, _ = cute_common.reserve_tmem_mma_fragment(
            tmem_ptr,
            tmem_offset,
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            self.mma_compute_KQ_stage,
            self.acc_dtype,
        )

        tSTtSFK_layout = blockscaled_utils.make_tmem_layout_sfa(
            KQ_tiled_mma,
            self.KQ_mma_tiler_sfk_load,
            self.sf_vec_size,
            cute.slice_(sfK_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tSTtSFK, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFK_layout, self.sf_dtype)
        tSTtSFK_h1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFK_layout, self.sf_dtype)

        tSTtSFQ_layout = blockscaled_utils.make_tmem_layout_sfb(
            KQ_tiled_mma_sfb,
            self.KQ_mma_tiler_sfq_load,
            self.sf_vec_size,
            cute.slice_(sfQ_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tSTtSFQ, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFQ_layout, self.sf_dtype)
        tSTtSFQ_h1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tSTtSFQ_layout, self.sf_dtype)

        tDPtSFV_layout = blockscaled_utils.make_tmem_layout_sfa(
            VDO_tiled_mma,
            (self.VDO_mma_tiler[0], self.VDO_mma_tiler[1], 128),
            self.sf_vec_size,
            cute.slice_(SFV_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDPtSFV, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFV_layout, self.sf_dtype)
        tDPtSFV_h1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFV_layout, self.sf_dtype)

        tDPtSFDO_layout = blockscaled_utils.make_tmem_layout_sfb(
            VDO_tiled_mma_sfb,
            self.VDO_mma_tiler_sfdo_load,
            self.sf_vec_size,
            cute.slice_(SFDO_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDPtSFDO, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFDO_layout, self.sf_dtype)
        tDPtSFDO_h1, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDPtSFDO_layout, self.sf_dtype)

        SFDST_layout = blockscaled_utils.make_tmem_layout_sfa(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            self.sf_vec_size,
            cute.slice_(SFV_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDKtSFDS, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, SFDST_layout, self.sf_dtype)

        tDVtSFP_layout = blockscaled_utils.make_tmem_layout_sfa(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfK_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDVtSFP, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, tDVtSFP_layout, self.sf_dtype)

        tDKtDK, tmem_offset, _ = cute_common.reserve_tmem_mma_fragment(
            tmem_ptr,
            tmem_offset,
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            self.mma_compute_dKdV_stage,
            self.acc_dtype,
        )
        tDVtDV, tmem_offset, _ = cute_common.reserve_tmem_mma_fragment(
            tmem_ptr,
            tmem_offset,
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            self.mma_compute_dKdV_stage,
            self.acc_dtype,
        )

        SFDO_mn_layout = blockscaled_utils.make_tmem_layout_sfb(
            PdO_tiled_mma_sfb,
            self.PdO_mma_tiler_sfb,
            self.sf_vec_size,
            cute.slice_(SFDO_mn_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDVtSFDO_mn, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, SFDO_mn_layout, self.sf_dtype)

        SFQ_mn_layout = blockscaled_utils.make_tmem_layout_sfb(
            dSQ_tiled_mma_sfb,
            self.dSQ_mma_tiler_sfb,
            self.sf_vec_size,
            cute.slice_(sSFQ_mn_smem_layout_staged, (None, None, None, 0, 0)),
        )
        tDKtSFQ_mn, tmem_offset, _ = cute_common.reserve_tmem_tensor(tmem_ptr, tmem_offset, SFQ_mn_layout, self.sf_dtype)

        # tKrK/tVrV TMEM allocations removed - K/V now read directly from SMEM fragments

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

            tidx, _, _ = cute.arch.thread_idx()

            # ///////////////////////////////////////////////////////////////////////////////
            #  LOAD warp - persistent loop
            # ///////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
                persistent_iter = Int32(0)
                cumulative_trip_count_load = Int32(0)
                while work_tile.is_valid_tile:
                    cur_tile = work_tile.tile_idx
                    bidx_v = cur_tile[0]
                    bidy_v = cur_tile[1]
                    bidz_v = cur_tile[2]
                    # Compute tile parameters from virtual coordinates
                    blk_coord = (Int32(0), bidx_v, Int32(0), ((Int32(0), bidy_v), bidz_v))
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

                    trip_start = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_end = trip_start + trip_count
                    trip_count = trip_count * problem_shape_cur_batch[3][0][0]

                    cluster_idx = bidx_v // self.cluster_shape_mn[0]
                    if cluster_idx * self.tile_shape_K < problem_shape_cur_batch[1] and trip_count > 0:
                        self.load(
                            Q_in,
                            QT_in,
                            V_in,
                            K_in,
                            dO_in,
                            dOT_in,
                            SFQ_in,
                            SFQ_mn_in,
                            SFK_in,
                            SFV_in,
                            SFDO_in,
                            SFDO_mn_in,
                            LSE,
                            sum_OdO,
                            sQ,
                            sQT,
                            sK,
                            sV,
                            sdO,
                            sdOT,
                            sSFQ,
                            sSFQ_mn,
                            sSFK,
                            sSFV,
                            sSFDO,
                            sSFDO_mn,
                            sLSE,
                            sSum_OdO,
                            KQ_tiled_mma,
                            VDO_tiled_mma,
                            dSQ_tiled_mma,
                            PdO_tiled_mma,
                            KQ_tiled_mma_sfb,
                            VDO_tiled_mma_sfb,
                            dSQ_tiled_mma_sfb,
                            PdO_tiled_mma_sfb,
                            KQ_tiled_mma_sfa,
                            VDO_tiled_mma_sfa,
                            tma_atom_Q,
                            tma_atom_QT,
                            tma_atom_K,
                            tma_atom_V,
                            tma_atom_dO,
                            tma_atom_dOT,
                            tma_atom_sfK,
                            tma_atom_sfQ,
                            tma_atom_sfQ_mn,
                            tma_atom_sfV,
                            tma_atom_sfDO,
                            tma_atom_sfDO_mn,
                            tma_atom_LSE,
                            tma_atom_sum_OdO,
                            blk_offset,
                            problem_shape_cur_batch,
                            trip_count,
                            trip_start,
                            trip_end,
                            mma_tile_coord_v,
                            block_in_cluster_coord_vmnk,
                            cluster_layout_vmnk,
                            (
                                load_mma_KQ_pipeline,
                                load_mma_KQ_aux_pipeline,
                                load_mma_VDO_pipeline,
                            ),
                            bidx_v,
                            bidy_v,
                            bidz_v,
                            problem_shape[3][0][1],  # h_k
                            cumulative_trip_count=cumulative_trip_count_load,
                        )

                    cumulative_trip_count_load = cumulative_trip_count_load + trip_count
                    persistent_iter = persistent_iter + Int32(1)
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
            #  MMA warp - persistent loop
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx == self.mma_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)
                persistent_iter = Int32(0)
                cumulative_trip_count_mma = Int32(0)
                while work_tile.is_valid_tile:
                    cur_tile = work_tile.tile_idx
                    bidx_v = cur_tile[0]
                    bidy_v = cur_tile[1]
                    bidz_v = cur_tile[2]

                    blk_coord = (Int32(0), bidx_v, Int32(0), ((Int32(0), bidy_v), bidz_v))
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

                    trip_start = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_end = trip_start + trip_count
                    trip_count = trip_count * problem_shape_cur_batch[3][0][0]

                    cluster_idx = bidx_v // self.cluster_shape_mn[0]
                    if cluster_idx * self.tile_shape_K < problem_shape_cur_batch[1] and trip_count > 0:
                        # Create mma_compute_dK producer state with correct phase.
                        # With 1 stage, the mbarrier phase flips every tile.
                        # Advance the fresh state to match the current mbarrier phase.
                        mma_compute_dK_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dKdV_stage)
                        if persistent_iter % Int32(2) == Int32(1):
                            mma_compute_dK_producer_state.advance()

                        self.mma(
                            tmem,
                            KQ_tiled_mma,
                            VDO_tiled_mma,
                            dSQ_tiled_mma,
                            PdO_tiled_mma,
                            KQ_tiled_mma_smem,
                            VDO_tiled_mma_smem,
                            tStS,
                            tSTrK_0,
                            tSTrK_1,
                            tSTrQ_0,
                            tSTrQ_1,
                            tSTrQ,
                            tSTtSFK,
                            tSTtSFK_h1,
                            tSTtSFQ,
                            tSTtSFQ_h1,
                            tDPtDP,
                            tDPrV_0,
                            tDPrV_1,
                            tdPTrdO_0,
                            tdPTrdO_1,
                            tdPTrdO,
                            tDPtSFV,
                            tDPtSFV_h1,
                            tDPtSFDO,
                            tDPtSFDO_h1,
                            tDKtDK,
                            tDKrDS,
                            tDKrQT,
                            tDKtSFDS,
                            tDKtSFQ_mn,
                            tDVtDV,
                            tDVrP,
                            tDVrDOT,
                            tDVtSFP,
                            tDVtSFDO_mn,
                            trip_count,
                            trip_start,
                            trip_end,
                            (
                                load_mma_KQ_pipeline,
                                load_mma_KQ_aux_pipeline,
                                load_mma_VDO_pipeline,
                                mma_compute_KQ_pipeline,
                                mma_compute_VDO_pipeline,
                                compute_mma_P_pipeline,
                                compute_mma_dS_pipeline,
                                mma_compute_dK_pipeline,
                            ),
                            mma_compute_dK_producer_state,
                            sSFK,
                            sSFQ,
                            sSFQ_mn,
                            sSFV,
                            sSFDO,
                            sSFDO_mn,
                            cumulative_trip_count=cumulative_trip_count_mma,
                        )
                        cumulative_trip_count_mma = cumulative_trip_count_mma + trip_count
                        persistent_iter = persistent_iter + Int32(1)

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

                    blk_coord = (Int32(0), bidx_v, Int32(0), ((Int32(0), bidy_v), bidz_v))
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

                    trip_start = fmha_masks.FusedMask.get_trip_start(
                        self.mask_type,
                        blk_coord,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_count = fmha_masks.FusedMask.get_trip_count(
                        self.mask_type,
                        blk_coord,
                        self.mask_cta_tiler,
                        problem_shape_cur_batch[0],
                        problem_shape_cur_batch[1],
                        window_size_left,
                        window_size_right,
                    )
                    trip_end = trip_start + trip_count
                    trip_count = trip_count * problem_shape_cur_batch[3][0][0]

                    cluster_idx = bidx_v // self.cluster_shape_mn[0]
                    if cluster_idx * self.tile_shape_K < problem_shape_cur_batch[1] and trip_count > 0:
                        self.compute(
                            tStS,
                            tDPtDP,
                            tDKtSFDS,
                            tDVtSFP,
                            sDS,
                            sDS_scale_exchange,
                            sP,
                            blk_coord,
                            problem_shape_cur_batch,
                            trip_count,
                            trip_start,
                            trip_end,
                            scale_softmax,
                            window_size_left,
                            window_size_right,
                            is_leader_cta,
                            sLSE,
                            sSum_OdO,
                            pipeline_args=(
                                mma_compute_KQ_pipeline,
                                mma_compute_VDO_pipeline,
                                compute_mma_P_pipeline,
                                compute_mma_dS_pipeline,
                            ),
                            cumulative_trip_count=cumulative_trip_count_compute,
                        )
                        if warp_idx >= self.compute_warp_id_0[0] and warp_idx <= self.compute_warp_id_0[-1]:
                            # Create consumer state with correct phase for mbarrier.
                            # With 1 stage, phase flips every tile.
                            mma_compute_Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage)
                            if persistent_iter % Int32(2) == Int32(1):
                                mma_compute_Q_consumer_state.advance()
                            self.epilogue(
                                blk_coord,
                                blk_offset,
                                problem_shape_cur_batch,
                                dK,
                                dV,
                                tDKtDK,
                                tDVtDV,
                                scale_softmax,
                                is_leader_cta,
                                (mma_compute_dK_pipeline, mma_compute_Q_consumer_state),
                            )
                            self.epilogue_sync_barrier.arrive_and_wait()
                        cumulative_trip_count_compute = cumulative_trip_count_compute + trip_count
                        persistent_iter = persistent_iter + Int32(1)

                    # Sync all non-sched warps before advancing to next persistent tile
                    self.persistent_tile_barrier.arrive_and_wait()
                    # CLC consumer: advance to next tile
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

            # ///////////////////////////////////////////////////////////////////////////////
            #  Empty warp (11) - persistent CLC consumer loop
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx in self.empty_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
                while work_tile.is_valid_tile:
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

        else:
            # ===================================================================
            #  NON-PERSISTENT MODE (original code)
            # ===================================================================
            # get the current batch problem shape
            # bidy indexes the OUTPUT slot (h_k * h_r_split + split); the input KV
            # head and this cluster's first Q head derive from it (see __init__).
            blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
            bidy_kv = bidy // self.n_split
            h_r_begin = ((bidy % self.n_split) // self.s_q_split) * (problem_shape[3][0][0] // self.h_r_split)
            q_slice = bidy % self.s_q_split
            # problem_shape = (s_q_max, s_k_max, d: hidden dim, ((h_r: #q_head_per_kv, h_k: #kv_heads), orig_b: batch size)) mark
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

            trip_start = fmha_masks.FusedMask.get_trip_start(
                self.mask_type,
                blk_coord,
                self.mask_cta_tiler,
                problem_shape_cur_batch[0],
                problem_shape_cur_batch[1],
                window_size_left,
                window_size_right,
            )
            trip_count = fmha_masks.FusedMask.get_trip_count(
                self.mask_type,
                blk_coord,
                self.mask_cta_tiler,
                problem_shape_cur_batch[0],
                problem_shape_cur_batch[1],
                window_size_left,
                window_size_right,
            )

            # This cluster's Q-tile chunk of the KV tile's trip range (s_q_split
            # near-equal pieces). The mask classification inside compute() is
            # relative to the FULL range start, which travels separately.
            trip_start_full = trip_start
            chunk_begin = trip_start + (q_slice * trip_count) // self.s_q_split
            chunk_end = trip_start + ((q_slice + 1) * trip_count) // self.s_q_split
            trip_start = chunk_begin
            trip_end = chunk_end
            trip_count = chunk_end - chunk_begin

            # Q heads walked by this cluster: h_r / h_r_split.
            trip_count = trip_count * (problem_shape_cur_batch[3][0][0] // self.h_r_split)
            # The dK/dV views carry h_k * h_r_split slots; the epilogues build
            # their global tiles from the HB they are handed.
            problem_shape_out = (
                problem_shape_cur_batch[0],
                problem_shape_cur_batch[1],
                problem_shape_cur_batch[2],
                ((problem_shape_cur_batch[3][0][0], problem_shape_cur_batch[3][0][1] * self.n_split), problem_shape_cur_batch[3][1]),
            )

            # Cluster wait before tensor memory alloc for 2-CTA
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            # Every CTA in a live 2-CTA cluster must enter the collective pipeline.
            cluster_idx = bidx // self.cluster_shape_mn[0]
            if cluster_idx * self.tile_shape_K < problem_shape_cur_batch[1] and trip_count > 0:
                # ///////////////////////////////////////////////////////////////////////////////
                #  LOAD
                # ///////////////////////////////////////////////////////////////////////////////
                if warp_idx == self.load_warp_id:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

                    self.load(
                        Q_in,
                        QT_in,
                        V_in,
                        K_in,
                        dO_in,
                        dOT_in,
                        SFQ_in,
                        SFQ_mn_in,
                        SFK_in,
                        SFV_in,
                        SFDO_in,
                        SFDO_mn_in,
                        LSE,
                        sum_OdO,
                        sQ,
                        sQT,
                        sK,
                        sV,
                        sdO,
                        sdOT,
                        sSFQ,
                        sSFQ_mn,
                        sSFK,
                        sSFV,
                        sSFDO,
                        sSFDO_mn,
                        sLSE,
                        sSum_OdO,
                        KQ_tiled_mma,
                        VDO_tiled_mma,
                        dSQ_tiled_mma,
                        PdO_tiled_mma,
                        KQ_tiled_mma_sfb,
                        VDO_tiled_mma_sfb,
                        dSQ_tiled_mma_sfb,
                        PdO_tiled_mma_sfb,
                        KQ_tiled_mma_sfa,
                        VDO_tiled_mma_sfa,
                        tma_atom_Q,
                        tma_atom_QT,
                        tma_atom_K,
                        tma_atom_V,
                        tma_atom_dO,
                        tma_atom_dOT,
                        tma_atom_sfK,
                        tma_atom_sfQ,
                        tma_atom_sfQ_mn,
                        tma_atom_sfV,
                        tma_atom_sfDO,
                        tma_atom_sfDO_mn,
                        tma_atom_LSE,
                        tma_atom_sum_OdO,
                        blk_offset,
                        problem_shape_cur_batch,
                        trip_count,
                        trip_start,
                        trip_end,
                        mma_tile_coord_v,
                        block_in_cluster_coord_vmnk,
                        cluster_layout_vmnk,
                        (
                            load_mma_KQ_pipeline,
                            load_mma_KQ_aux_pipeline,
                            load_mma_VDO_pipeline,
                        ),
                        bidx,
                        bidy_kv,
                        bidz,
                        problem_shape[3][0][1],  # h_k
                        h_r_begin=h_r_begin,
                    )

                # ///////////////////////////////////////////////////////////////////////////////
                #  MMA
                # ///////////////////////////////////////////////////////////////////////////////
                elif warp_idx == self.mma_warp_id:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)
                    mma_compute_dK_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dKdV_stage)
                    # NOTE: HERE
                    self.mma(
                        tmem,
                        KQ_tiled_mma,
                        VDO_tiled_mma,
                        dSQ_tiled_mma,
                        PdO_tiled_mma,
                        KQ_tiled_mma_smem,
                        VDO_tiled_mma_smem,
                        tStS,
                        tSTrK_0,
                        tSTrK_1,
                        tSTrQ_0,
                        tSTrQ_1,
                        tSTrQ,
                        tSTtSFK,
                        tSTtSFK_h1,
                        tSTtSFQ,
                        tSTtSFQ_h1,
                        tDPtDP,
                        tDPrV_0,
                        tDPrV_1,
                        tdPTrdO_0,
                        tdPTrdO_1,
                        tdPTrdO,
                        tDPtSFV,
                        tDPtSFV_h1,
                        tDPtSFDO,
                        tDPtSFDO_h1,
                        tDKtDK,
                        tDKrDS,
                        tDKrQT,
                        tDKtSFDS,
                        tDKtSFQ_mn,
                        tDVtDV,
                        tDVrP,
                        tDVrDOT,
                        tDVtSFP,
                        tDVtSFDO_mn,
                        trip_count,
                        trip_start,
                        trip_end,
                        (
                            load_mma_KQ_pipeline,
                            load_mma_KQ_aux_pipeline,
                            load_mma_VDO_pipeline,
                            mma_compute_KQ_pipeline,
                            mma_compute_VDO_pipeline,
                            compute_mma_P_pipeline,
                            compute_mma_dS_pipeline,
                            mma_compute_dK_pipeline,
                        ),
                        mma_compute_dK_producer_state,
                        sSFK,
                        sSFQ,
                        sSFQ_mn,
                        sSFV,
                        sSFDO,
                        sSFDO_mn,
                    )

                # ///////////////////////////////////////////////////////////////////////////////
                #  Compute
                # ///////////////////////////////////////////////////////////////////////////////
                elif warp_idx >= self.compute_warp_id_0[0] and warp_idx <= self.compute_warp_id_1[-1]:
                    cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
                    self.compute(
                        tStS,
                        tDPtDP,
                        tDKtSFDS,
                        tDVtSFP,
                        sDS,
                        sDS_scale_exchange,
                        sP,
                        blk_coord,
                        problem_shape_cur_batch,
                        trip_count,
                        trip_start,
                        trip_end,
                        scale_softmax,
                        window_size_left,
                        window_size_right,
                        is_leader_cta,
                        sLSE,
                        sSum_OdO,
                        pipeline_args=(
                            mma_compute_KQ_pipeline,
                            mma_compute_VDO_pipeline,
                            compute_mma_P_pipeline,
                            compute_mma_dS_pipeline,
                        ),
                        iter_start_global=trip_start_full,
                    )
                    if warp_idx >= self.compute_warp_id_0[0] and warp_idx <= self.compute_warp_id_0[-1]:
                        mma_compute_Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage)
                        # Epilogue
                        self.epilogue(
                            blk_coord,
                            blk_offset,
                            problem_shape_out,
                            dK,
                            dV,
                            tDKtDK,
                            tDVtDV,
                            scale_softmax,
                            is_leader_cta,
                            (mma_compute_dK_pipeline, mma_compute_Q_consumer_state),
                        )
                        self.epilogue_sync_barrier.arrive_and_wait()

                else:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
            elif cluster_idx * self.tile_shape_K < problem_shape_cur_batch[1]:
                # An in-range KV tile that no query row attends (causal with
                # S_kv > S_q): its gradients are exactly zero, and the mainloop
                # above is skipped, so the tile must be written here or the
                # caller's dK/dV keep whatever was in the buffer.
                if warp_idx >= self.compute_warp_id_0[0] and warp_idx <= self.compute_warp_id_1[-1]:
                    self.epilogue_zero(blk_coord, blk_offset, problem_shape_out, dK, dV)

        # In persistent mode, sync across the 2-CTA cluster before TMEM dealloc
        # to prevent one CTA from freeing TMEM while partner CTA's MMA warp still accesses it.
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
        Q_in: cute.Tensor,
        QT_in: cute.Tensor,
        V_in: cute.Tensor,
        K_in: cute.Tensor,
        dO_in: cute.Tensor,
        dOT_in: cute.Tensor,
        SFQ_in: cute.Tensor,
        SFQ_mn_in: cute.Tensor,
        SFK_in: cute.Tensor,
        SFV_in: cute.Tensor,
        SFDO_in: cute.Tensor,
        SFDO_mn_in: cute.Tensor,
        LSE: cute.Tensor,
        Sum_OdO: cute.Tensor,
        sQ: cute.Tensor,
        sQT: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOT: cute.Tensor,
        sSFQ: cute.Tensor,
        sSFQ_mn: cute.Tensor,
        sSFK: cute.Tensor,
        sSFV: cute.Tensor,
        sSFDO: cute.Tensor,
        sSFDO_mn: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        KQ_tiled_mma: cute.TiledMma,
        VDO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        KQ_tiled_mma_sfb: cute.TiledMma,
        VDO_tiled_mma_sfb: cute.TiledMma,
        dSQ_tiled_mma_sfb: cute.TiledMma,
        PdO_tiled_mma_sfb: cute.TiledMma,
        KQ_tiled_mma_sfa: cute.TiledMma,
        VDO_tiled_mma_sfa: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_QT: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOT: cute.CopyAtom,
        tma_atom_sfK: cute.CopyAtom,
        tma_atom_sfQ: cute.CopyAtom,
        tma_atom_sfQ_mn: cute.CopyAtom,
        tma_atom_sfV: cute.CopyAtom,
        tma_atom_sfDO: cute.CopyAtom,
        tma_atom_sfDO_mn: cute.CopyAtom,
        tma_atom_LSE: cute.CopyAtom,
        tma_atom_sum_OdO: cute.CopyAtom,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        mma_tile_coord_v: Int32,
        block_in_cluster_coord_vmnk,
        cluster_layout_vmnk,
        # (load_mma_KQ_pipeline, load_mma_KQ_aux_pipeline, load_mma_VDO_pipeline)
        pipeline_args: tuple,
        # Logical block coordinates (from block_idx() in non-persistent, from CLC in persistent)
        blk_coord_k: Int32,
        blk_coord_h_k: Int32,
        blk_coord_b: Int32,
        num_h_k: Int32,  # total number of KV heads (replaces grid_dim_y)
        cumulative_trip_count: Int32 = Int32(0),  # accumulated trip_count across persistent tiles
        h_r_begin: Int32 = Int32(0),  # first Q head of this cluster's slice (h_r_split)
    ):
        """Producer warp: TMA loads of K/V once and of Q, dO, LSE, row-dot and their scale factors per Q tile and head."""
        grid_dim_y = num_h_k

        blk_coord_h_r = h_r_begin
        blk_coord_h_q = (blk_coord_h_r, blk_coord_h_k)
        blk_coord_h_kv = (Int32(0), blk_coord_h_k)

        iter_index = iter_start
        (
            load_mma_KQ_pipeline,
            load_mma_KQ_aux_pipeline,
            load_mma_VDO_pipeline,
        ) = pipeline_args

        Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
        QT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), QT_in)
        K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
        V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
        dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)
        dOT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), dOT_in)

        # (bM, bK, RestM, RestK, (H, B))
        gK = cute.local_tile(K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None))

        # (bN, bK, RestN, RestK, (H, B))
        gQ = cute.local_tile(Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None))

        gQT = cute.local_tile(QT, cute.select(self.dSQ_mma_tiler, mode=[1, 2]), (None, None, None))

        # (bM, bK, RestM, RestK, (H, B))
        gV = cute.local_tile(V, cute.select(self.VDO_mma_tiler, mode=[0, 2]), (None, None, None))
        # (bN, bK, RestN, RestK, (H, B))
        gdO = cute.local_tile(dO, cute.select(self.VDO_mma_tiler, mode=[1, 2]), (None, None, None))
        gdOT = cute.local_tile(dOT, cute.select(self.PdO_mma_tiler, mode=[1, 2]), (None, None, None))

        gSFK = cute.local_tile(SFK_in, cute.select(self.KQ_mma_tiler_sfk_load, mode=[0, 2]), (None, None, None))

        #  SFK_in shape: (32, 4, rest_m, 4, rest_k, l)
        gSFQ = cute.local_tile(SFQ_in, cute.select(self.KQ_mma_tiler_sfq_load, mode=[1, 2]), (None, None, None))

        gSFQ_mn = cute.local_tile(SFQ_mn_in, cute.select(self.dSQ_mma_tiler_sfb, mode=[1, 2]), (None, None, None))

        gSFV = cute.local_tile(SFV_in, cute.select(self.VDO_mma_tiler_sfv_load, mode=[0, 2]), (None, None, None))
        gSFDO = cute.local_tile(SFDO_in, cute.select(self.VDO_mma_tiler_sfdo_load, mode=[1, 2]), (None, None, None))
        gSFDO_mn = cute.local_tile(SFDO_mn_in, cute.select(self.PdO_mma_tiler_sfb, mode=[1, 2]), (None, None, None))

        KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
        VDO_thr_mma = VDO_tiled_mma.get_slice(mma_tile_coord_v)
        dSQ_thr_mma = dSQ_tiled_mma.get_slice(mma_tile_coord_v)
        PdO_thr_mma = PdO_tiled_mma.get_slice(mma_tile_coord_v)

        dSQ_thr_mma_sfb = dSQ_tiled_mma_sfb.get_slice(mma_tile_coord_v)
        dSQ_thr_mma_sfb = dSQ_thr_mma_sfb.get_slice(0)
        PdO_thr_mma_sfb = PdO_tiled_mma_sfb.get_slice(mma_tile_coord_v)
        PdO_thr_mma_sfb = PdO_thr_mma_sfb.get_slice(0)
        KQ_thr_mma_sfb = KQ_tiled_mma_sfb.get_slice(0)
        VDO_thr_mma_sfb = VDO_tiled_mma_sfb.get_slice(0)
        KQ_thr_mma_sfa = KQ_tiled_mma_sfa.get_slice(mma_tile_coord_v)
        VDO_thr_mma_sfa = VDO_tiled_mma_sfa.get_slice(mma_tile_coord_v)

        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tSgK = KQ_thr_mma.partition_A(gK)
        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tSgQ = KQ_thr_mma.partition_B(gQ)

        tdKgQT = dSQ_thr_mma.partition_B(gQT)

        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tdPgV = VDO_thr_mma.partition_A(gV)
        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tdPgdO = VDO_thr_mma.partition_B(gdO)

        tdVgdOT = PdO_thr_mma.partition_B(gdOT)

        tSTgSFK = KQ_thr_mma_sfa.partition_A(gSFK)
        tSTgSFQ = KQ_thr_mma_sfb.partition_B(gSFQ)

        tDPgSFV = VDO_thr_mma_sfa.partition_A(gSFV)
        tDPgSFDO = VDO_thr_mma_sfb.partition_B(gSFDO)

        tDKgSFQ_mn = dSQ_thr_mma_sfb.partition_B(gSFQ_mn)
        tDVgSFDO_mn = PdO_thr_mma_sfb.partition_B(gSFDO_mn)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tTMAsQ, tTMAgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )
        # NOTE: Don't need to load QT really, just reference it with transposed layout

        tTMAsQT, tTMAgQT = cute.nvgpu.cpasync.tma_partition(
            tma_atom_QT,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQT, 0, 3),
            cute.group_modes(tdKgQT, 0, 3),
        )

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tTMAsK, tTMAgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK, 0, 3),
        )

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tTMAsdO, tTMAgdO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPgdO, 0, 3),
        )

        tTMAsdOT, tTMAgdOT = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dOT,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sdOT, 0, 3),
            cute.group_modes(tdVgdOT, 0, 3),
        )

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tTMAsV, tTMAgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPgV, 0, 3),
        )

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tTMAsSFQ, tTMAgSFQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFQ, 0, 3),
            cute.group_modes(tSTgSFQ, 0, 3),
        )
        tTMAsSFQ = cute.filter_zeros(tTMAsSFQ)
        tTMAgSFQ = cute.filter_zeros(tTMAgSFQ)

        tTMAsSFQ_mn, tTMAgSFQ_mn = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfQ_mn,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFQ_mn, 0, 3),
            cute.group_modes(tDKgSFQ_mn, 0, 3),
        )
        tTMAsSFQ_mn = cute.filter_zeros(tTMAsSFQ_mn)
        tTMAgSFQ_mn = cute.filter_zeros(tTMAgSFQ_mn)

        tTMAsSFK, tTMAgSFK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfK,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFK, 0, 3),
            cute.group_modes(tSTgSFK, 0, 3),
        )
        tTMAsSFK = cute.filter_zeros(tTMAsSFK)
        tTMAgSFK = cute.filter_zeros(tTMAgSFK)

        tTMAsSFV, tTMAgSFV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfV,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFV, 0, 3),
            cute.group_modes(tDPgSFV, 0, 3),
        )
        tTMAsSFV = cute.filter_zeros(tTMAsSFV)
        tTMAgSFV = cute.filter_zeros(tTMAgSFV)

        tTMAsSFDO, tTMAgSFDO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfDO,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFDO, 0, 3),
            cute.group_modes(tDPgSFDO, 0, 3),
        )
        tTMAsSFDO = cute.filter_zeros(tTMAsSFDO)
        tTMAgSFDO = cute.filter_zeros(tTMAgSFDO)

        tTMAsSFDO_mn, tTMAgSFDO_mn = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfDO_mn,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFDO_mn, 0, 3),
            cute.group_modes(tDVgSFDO_mn, 0, 3),
        )
        tTMAsSFDO_mn = cute.filter_zeros(tTMAsSFDO_mn)
        tTMAgSFDO_mn = cute.filter_zeros(tTMAgSFDO_mn)

        # TMA partition for LSE and sum_OdO
        gLSE = cute.local_tile(LSE, (sLSE.shape[0],), (None, None))
        gSum_OdO = cute.local_tile(Sum_OdO, (sSum_OdO.shape[0],), (None, None))

        tTMAsLSE, tTMAgLSE = cute.nvgpu.cpasync.tma_partition(tma_atom_LSE, 0, cute.make_layout(1), sLSE, gLSE)
        tTMAsSum_OdO, tTMAgSum_OdO = cute.nvgpu.cpasync.tma_partition(tma_atom_sum_OdO, 0, cute.make_layout(1), sSum_OdO, gSum_OdO)

        load_mma_KQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_all_stage)
        load_mma_KQ_aux_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_all_stage)
        load_mma_VDO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_all_stage)
        # Persistent mode: advance state to match mbarrier phase accumulated across tiles.
        # After N total stages, state needs N % (2*num_stages) advances from initial.
        if cutlass.const_expr(self.is_persistent):
            n_advance = cumulative_trip_count % Int32(2 * self.load_mma_all_stage)
            for _ in cutlass.range_constexpr(2 * self.load_mma_all_stage):
                if n_advance > Int32(0):
                    load_mma_KQ_producer_state.advance()
                    load_mma_KQ_aux_producer_state.advance()
                    load_mma_VDO_producer_state.advance()
                    n_advance = n_advance - Int32(1)
        # Fix for persistent mode: CTA 1 (non-leader) never calls arrive_and_expect_tx
        # on the full barrier (guarded by is_leader_cta in producer_acquire), but TMA
        # copies still signal it with bytes. Over many persistent tiles, the mbarrier
        # transaction counter overflows causing hardware errors. Reinitialize CTA 1's
        # full barriers at the start of each tile to reset accumulated bytes.
        if cutlass.const_expr(self.is_persistent):
            bidx_reinit, _, _ = cute.arch.block_idx()
            is_non_leader = bidx_reinit % cute.size(KQ_tiled_mma.thr_id.shape) != 0
            if is_non_leader:
                with cute.arch.elect_one():
                    for stage_idx in cutlass.range_constexpr(self.load_mma_all_stage):
                        cute.arch.mbarrier_init(
                            load_mma_KQ_pipeline.sync_object_full.get_barrier(stage_idx),
                            1,  # arrive_count = producer_group.size = 1
                        )
                        cute.arch.mbarrier_init(
                            load_mma_KQ_aux_pipeline.sync_object_full.get_barrier(stage_idx),
                            1,  # arrive_count = producer_group.size = 1
                        )
                        cute.arch.mbarrier_init(
                            load_mma_VDO_pipeline.sync_object_full.get_barrier(stage_idx),
                            1,  # arrive_count = producer_group.size = 1
                        )
                cute.arch.mbarrier_init_fence()
        load_mma_KQ_pipeline.producer_acquire(load_mma_KQ_producer_state)
        tma_barrier_KQ = load_mma_KQ_pipeline.producer_get_barrier(load_mma_KQ_producer_state)
        # NOTE: append additional bytes (doubled for slot 0 and slot 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(tma_barrier_KQ, 2 * (self.tma_copy_K_bytes) + 4 * self.k_halves * self.tma_copy_sfK_bytes)

        SF_kv_load_index = blk_coord_b * grid_dim_y + blk_coord_h_k
        SF_kv_load_index_s0 = SF_kv_load_index * 2
        SF_kv_load_index_s1 = SF_kv_load_index * 2 + 1
        num_h_r = problem_shape[3][0][0]
        SF_q_load_index = SF_kv_load_index * num_h_r + blk_coord_h_r
        SF_q_load_index_s0 = SF_q_load_index * 2
        SF_q_load_index_s1 = SF_q_load_index * 2 + 1

        # (s_q, d, h_r, h_k, b)
        # Load Q (B operand)
        cute.copy(
            tma_atom_Q,
            tTMAgQ[(None, iter_index, 0, (blk_coord_h_q, blk_coord_b))],
            tTMAsQ[None, load_mma_KQ_producer_state.index],
            tma_bar_ptr=tma_barrier_KQ,
            mcast_mask=self.b_full_mcast_mask,
        )

        for sfq_k_half in cutlass.range_constexpr(self.k_halves):
            sfq_stage = load_mma_KQ_producer_state.index * self.k_halves + sfq_k_half
            cute.copy(
                tma_atom_sfQ,
                tTMAgSFQ[(None, iter_index, sfq_k_half, SF_q_load_index_s0)],
                tTMAsSFQ[None, sfq_stage, 0],
                tma_bar_ptr=tma_barrier_KQ,
                mcast_mask=self.sfb_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfQ,
                tTMAgSFQ[(None, iter_index, sfq_k_half, SF_q_load_index_s1)],
                tTMAsSFQ[None, sfq_stage, 1],
                tma_bar_ptr=tma_barrier_KQ,
                mcast_mask=self.sfb_full_mcast_mask,
            )

        # Load K (A operand)
        cute.copy(
            tma_atom_K,
            tTMAgK[(None, blk_coord_k // 2, 0, (blk_coord_h_kv, blk_coord_b))],
            tTMAsK[None, 0],
            tma_bar_ptr=tma_barrier_KQ,
            mcast_mask=self.a_full_mcast_mask,
        )
        for sfk_k_half in cutlass.range_constexpr(self.k_halves):
            cute.copy(
                tma_atom_sfK,
                tTMAgSFK[(None, blk_coord_k // 2, sfk_k_half, SF_kv_load_index_s0)],
                tTMAsSFK[None, sfk_k_half, 0],
                tma_bar_ptr=tma_barrier_KQ,
            )
            cute.copy(
                tma_atom_sfK,
                tTMAgSFK[(None, blk_coord_k // 2, sfk_k_half, SF_kv_load_index_s1)],
                tTMAsSFK[None, sfk_k_half, 1],
                tma_bar_ptr=tma_barrier_KQ,
            )

        # load LSE via the KQ pipeline
        cute.copy(
            tma_atom_LSE,
            tTMAgLSE[(None, iter_index, ((blk_coord_h_r, blk_coord_h_k), blk_coord_b))],
            tTMAsLSE[(None, load_mma_KQ_producer_state.index)],
            tma_bar_ptr=tma_barrier_KQ,
        )
        load_mma_KQ_producer_state.advance()

        load_mma_VDO_pipeline.producer_acquire(load_mma_VDO_producer_state)
        tma_barrier_VDO = load_mma_VDO_pipeline.producer_get_barrier(load_mma_VDO_producer_state)
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(
                tma_barrier_VDO, 2 * (self.tma_copy_V_bytes) + 4 * self.k_halves * self.tma_copy_sfV_bytes
            )  # slot 0 + slot 1, 2x for CtaGroup.TWO

        # Load dO (B operand)
        cute.copy(
            tma_atom_dO,
            tTMAgdO[(None, iter_index, 0, (blk_coord_h_q, blk_coord_b))],
            tTMAsdO[None, load_mma_VDO_producer_state.index],
            tma_bar_ptr=tma_barrier_VDO,
            mcast_mask=self.b_full_mcast_mask,
        )

        # load dOT (B operand)
        cute.copy(
            tma_atom_dOT,
            tTMAgdOT[(None, 0, iter_index, (blk_coord_h_q, blk_coord_b))],
            tTMAsdOT[None, load_mma_VDO_producer_state.index],
            tma_bar_ptr=tma_barrier_VDO,
            mcast_mask=self.b_full_mcast_mask,
        )

        for sfdo_k_half in cutlass.range_constexpr(self.k_halves):
            sfdo_stage = load_mma_VDO_producer_state.index * self.k_halves + sfdo_k_half
            cute.copy(
                tma_atom_sfDO,
                tTMAgSFDO[(None, iter_index, sfdo_k_half, SF_q_load_index_s0)],
                tTMAsSFDO[None, sfdo_stage, 0],
                tma_bar_ptr=tma_barrier_VDO,
                mcast_mask=self.sfb_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfDO,
                tTMAgSFDO[(None, iter_index, sfdo_k_half, SF_q_load_index_s1)],
                tTMAsSFDO[None, sfdo_stage, 1],
                tma_bar_ptr=tma_barrier_VDO,
                mcast_mask=self.sfb_full_mcast_mask,
            )
        cute.copy(
            tma_atom_sfDO_mn,
            tTMAgSFDO_mn[(None, 0, iter_index, SF_q_load_index_s0)],
            tTMAsSFDO_mn[None, load_mma_VDO_producer_state.index, 0],
            tma_bar_ptr=tma_barrier_VDO,
            mcast_mask=self.sfb_full_mcast_mask,
        )
        cute.copy(
            tma_atom_sfDO_mn,
            tTMAgSFDO_mn[(None, 0, iter_index, SF_q_load_index_s1)],
            tTMAsSFDO_mn[None, load_mma_VDO_producer_state.index, 1],
            tma_bar_ptr=tma_barrier_VDO,
            mcast_mask=self.sfb_full_mcast_mask,
        )

        # Load V (A operand)
        cute.copy(
            tma_atom_V,
            tTMAgV[(None, blk_coord_k // 2, 0, (blk_coord_h_kv, blk_coord_b))],
            tTMAsV[None, 0],
            tma_bar_ptr=tma_barrier_VDO,
            mcast_mask=self.a_full_mcast_mask,
        )

        for sfv_k_half in cutlass.range_constexpr(self.k_halves):
            cute.copy(
                tma_atom_sfV,
                tTMAgSFV[(None, blk_coord_k // 2, sfv_k_half, SF_kv_load_index_s0)],
                tTMAsSFV[None, sfv_k_half, 0],
                tma_bar_ptr=tma_barrier_VDO,
            )
            cute.copy(
                tma_atom_sfV,
                tTMAgSFV[(None, blk_coord_k // 2, sfv_k_half, SF_kv_load_index_s1)],
                tTMAsSFV[None, sfv_k_half, 1],
                tma_bar_ptr=tma_barrier_VDO,
            )

        # load sum_OdO via the VDO pipeline
        cute.copy(
            tma_atom_sum_OdO,
            tTMAgSum_OdO[(None, iter_index, ((blk_coord_h_r, blk_coord_h_k), blk_coord_b))],
            tTMAsSum_OdO[(None, load_mma_VDO_producer_state.index)],
            tma_bar_ptr=tma_barrier_VDO,
        )

        load_mma_VDO_producer_state.advance()

        load_mma_KQ_aux_pipeline.producer_acquire(load_mma_KQ_aux_producer_state)
        tma_barrier_KQ_aux = load_mma_KQ_aux_pipeline.producer_get_barrier(load_mma_KQ_aux_producer_state)
        # Load QT (B operand) for later dK
        cute.copy(
            tma_atom_QT,
            tTMAgQT[(None, 0, iter_index, (blk_coord_h_q, blk_coord_b))],
            tTMAsQT[None, load_mma_KQ_aux_producer_state.index],
            tma_bar_ptr=tma_barrier_KQ_aux,
            mcast_mask=self.b_full_mcast_mask,
        )
        cute.copy(
            tma_atom_sfQ_mn,
            tTMAgSFQ_mn[(None, 0, iter_index, SF_q_load_index_s0)],
            tTMAsSFQ_mn[None, load_mma_KQ_aux_producer_state.index, 0],
            tma_bar_ptr=tma_barrier_KQ_aux,
            mcast_mask=self.sfb_full_mcast_mask,
        )
        cute.copy(
            tma_atom_sfQ_mn,
            tTMAgSFQ_mn[(None, 0, iter_index, SF_q_load_index_s1)],
            tTMAsSFQ_mn[None, load_mma_KQ_aux_producer_state.index, 1],
            tma_bar_ptr=tma_barrier_KQ_aux,
            mcast_mask=self.sfb_full_mcast_mask,
        )
        load_mma_KQ_aux_producer_state.advance()

        iter_count -= 1
        iter_index += 1

        while iter_count > 0:
            if iter_index == iter_end:
                iter_index = iter_start
                blk_coord_h_r += 1
                blk_coord_h_q = (blk_coord_h_r, blk_coord_h_k)
                SF_q_load_index = SF_kv_load_index * num_h_r + blk_coord_h_r
                SF_q_load_index_s0 = SF_q_load_index * 2
                SF_q_load_index_s1 = SF_q_load_index * 2 + 1

            load_mma_KQ_pipeline.producer_acquire(load_mma_KQ_producer_state)
            tma_barrier_KQ_inner = load_mma_KQ_pipeline.producer_get_barrier(load_mma_KQ_producer_state)
            # Load Q (B operand)
            cute.copy(
                tma_atom_Q,
                tTMAgQ[(None, iter_index, 0, (blk_coord_h_q, blk_coord_b))],
                tTMAsQ[None, load_mma_KQ_producer_state.index],
                tma_bar_ptr=tma_barrier_KQ_inner,
                mcast_mask=self.b_full_mcast_mask,
            )

            for sfq_k_half in cutlass.range_constexpr(self.k_halves):
                sfq_stage = load_mma_KQ_producer_state.index * self.k_halves + sfq_k_half
                cute.copy(
                    tma_atom_sfQ,
                    tTMAgSFQ[(None, iter_index, sfq_k_half, SF_q_load_index_s0)],
                    tTMAsSFQ[None, sfq_stage, 0],
                    tma_bar_ptr=tma_barrier_KQ_inner,
                    mcast_mask=self.sfb_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfQ,
                    tTMAgSFQ[(None, iter_index, sfq_k_half, SF_q_load_index_s1)],
                    tTMAsSFQ[None, sfq_stage, 1],
                    tma_bar_ptr=tma_barrier_KQ_inner,
                    mcast_mask=self.sfb_full_mcast_mask,
                )

            # load LSE via the KQ pipeline
            cute.copy(
                tma_atom_LSE,
                tTMAgLSE[(None, iter_index, ((blk_coord_h_r, blk_coord_h_k), blk_coord_b))],
                tTMAsLSE[(None, load_mma_KQ_producer_state.index)],
                tma_bar_ptr=tma_barrier_KQ_inner,
            )
            load_mma_KQ_producer_state.advance()

            load_mma_VDO_pipeline.producer_acquire(load_mma_VDO_producer_state)
            tma_barrier_VDO_inner = load_mma_VDO_pipeline.producer_get_barrier(load_mma_VDO_producer_state)

            # Load dO (B operand)
            cute.copy(
                tma_atom_dO,
                tTMAgdO[(None, iter_index, 0, (blk_coord_h_q, blk_coord_b))],
                tTMAsdO[None, load_mma_VDO_producer_state.index],
                tma_bar_ptr=tma_barrier_VDO_inner,
                mcast_mask=self.b_full_mcast_mask,
            )

            # load dOT (B operand)
            cute.copy(
                tma_atom_dOT,
                tTMAgdOT[(None, 0, iter_index, (blk_coord_h_q, blk_coord_b))],
                tTMAsdOT[None, load_mma_VDO_producer_state.index],
                tma_bar_ptr=tma_barrier_VDO_inner,
                mcast_mask=self.b_full_mcast_mask,
            )

            for sfdo_k_half in cutlass.range_constexpr(self.k_halves):
                sfdo_stage = load_mma_VDO_producer_state.index * self.k_halves + sfdo_k_half
                cute.copy(
                    tma_atom_sfDO,
                    tTMAgSFDO[(None, iter_index, sfdo_k_half, SF_q_load_index_s0)],
                    tTMAsSFDO[None, sfdo_stage, 0],
                    tma_bar_ptr=tma_barrier_VDO_inner,
                    mcast_mask=self.sfb_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfDO,
                    tTMAgSFDO[(None, iter_index, sfdo_k_half, SF_q_load_index_s1)],
                    tTMAsSFDO[None, sfdo_stage, 1],
                    tma_bar_ptr=tma_barrier_VDO_inner,
                    mcast_mask=self.sfb_full_mcast_mask,
                )
            cute.copy(
                tma_atom_sfDO_mn,
                tTMAgSFDO_mn[(None, 0, iter_index, SF_q_load_index_s0)],
                tTMAsSFDO_mn[None, load_mma_VDO_producer_state.index, 0],
                tma_bar_ptr=tma_barrier_VDO_inner,
                mcast_mask=self.sfb_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfDO_mn,
                tTMAgSFDO_mn[(None, 0, iter_index, SF_q_load_index_s1)],
                tTMAsSFDO_mn[None, load_mma_VDO_producer_state.index, 1],
                tma_bar_ptr=tma_barrier_VDO_inner,
                mcast_mask=self.sfb_full_mcast_mask,
            )

            # load sum_OdO via the VDO pipeline
            cute.copy(
                tma_atom_sum_OdO,
                tTMAgSum_OdO[(None, iter_index, ((blk_coord_h_r, blk_coord_h_k), blk_coord_b))],
                tTMAsSum_OdO[(None, load_mma_VDO_producer_state.index)],
                tma_bar_ptr=tma_barrier_VDO_inner,
            )

            load_mma_VDO_producer_state.advance()

            load_mma_KQ_aux_pipeline.producer_acquire(load_mma_KQ_aux_producer_state)
            tma_barrier_KQ_aux_inner = load_mma_KQ_aux_pipeline.producer_get_barrier(load_mma_KQ_aux_producer_state)
            # Load QT (B operand) for later dK
            cute.copy(
                tma_atom_QT,
                tTMAgQT[(None, 0, iter_index, (blk_coord_h_q, blk_coord_b))],
                tTMAsQT[None, load_mma_KQ_aux_producer_state.index],
                tma_bar_ptr=tma_barrier_KQ_aux_inner,
                mcast_mask=self.b_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfQ_mn,
                tTMAgSFQ_mn[(None, 0, iter_index, SF_q_load_index_s0)],
                tTMAsSFQ_mn[None, load_mma_KQ_aux_producer_state.index, 0],
                tma_bar_ptr=tma_barrier_KQ_aux_inner,
                mcast_mask=self.sfb_full_mcast_mask,
            )
            cute.copy(
                tma_atom_sfQ_mn,
                tTMAgSFQ_mn[(None, 0, iter_index, SF_q_load_index_s1)],
                tTMAsSFQ_mn[None, load_mma_KQ_aux_producer_state.index, 1],
                tma_bar_ptr=tma_barrier_KQ_aux_inner,
                mcast_mask=self.sfb_full_mcast_mask,
            )
            load_mma_KQ_aux_producer_state.advance()

            iter_count -= 1
            iter_index += 1

    @cute.jit
    def mma(
        self,
        tmem: utils.TmemAllocator,
        KQ_tiled_mma: cute.TiledMma,
        VDO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        KQ_tiled_mma_smem: cute.TiledMma,
        VDO_tiled_mma_smem: cute.TiledMma,
        tStS: cute.Tensor,
        tSTrK_0: cute.Tensor,
        tSTrK_1: cute.Tensor,
        tSTrQ_0: cute.Tensor,
        tSTrQ_1: cute.Tensor,
        tSTrQ: cute.Tensor,
        tSTtSFK: cute.Tensor,
        tSTtSFK_h1: cute.Tensor,
        tSTtSFQ: cute.Tensor,
        tSTtSFQ_h1: cute.Tensor,
        tDPtDP: cute.Tensor,
        tDPrV_0: cute.Tensor,
        tDPrV_1: cute.Tensor,
        tdPTrdO_0: cute.Tensor,
        tdPTrdO_1: cute.Tensor,
        tDPTrdO: cute.Tensor,
        tDPtSFV: cute.Tensor,
        tDPtSFV_h1: cute.Tensor,
        tDPtSFDO: cute.Tensor,
        tDPtSFDO_h1: cute.Tensor,
        tDKtDK: cute.Tensor,
        tDKrDS: cute.Tensor,
        tDKrQT: cute.Tensor,
        tDKtSFDS: cute.Tensor,
        tDKtSFQ_mn: cute.Tensor,
        tDVtDV: cute.Tensor,
        tDVrP: cute.Tensor,
        tDVrDOT: cute.Tensor,
        tDVtSFP: cute.Tensor,
        tDVtSFDO_mn: cute.Tensor,
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        # (load_mma_KQ_pipeline, load_mma_KQ_aux_pipeline, load_mma_VDO_pipeline,
        #  mma_compute_KQ_pipeline, mma_compute_VDO_pipeline,
        #  compute_mma_P_pipeline, compute_mma_dS_pipeline,
        #  mma_compute_dK_pipeline)
        pipeline_args: tuple,
        mma_compute_dK_producer_state,
        sSFK: cute.Tensor,
        sSFQ: cute.Tensor,
        sSFQ_mn: cute.Tensor,
        sSFV: cute.Tensor,
        sSFDO: cute.Tensor,
        sSFDO_mn: cute.Tensor,
        cumulative_trip_count: Int32 = Int32(0),  # accumulated trip_count across persistent tiles
    ):
        """MMA warp: block-scaled tcgen05 MMAs for S = K.Q^T, dP = V.dO^T, dK += dS^T.Q and dV += P^T.dO."""
        bidx, _, _ = cute.arch.block_idx()
        iter_count_origin = iter_count

        # For 2-CTA MMA: determine which CTA in the pair (0 or 1)
        mma_tile_coord_v = bidx % cute.size(KQ_tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Shadow KQ/VDO tiled_mma with SMEM variants for K/V operands
        KQ_tiled_mma = KQ_tiled_mma_smem
        VDO_tiled_mma = VDO_tiled_mma_smem

        (
            load_mma_KQ_pipeline,
            load_mma_KQ_aux_pipeline,
            load_mma_VDO_pipeline,
            mma_compute_KQ_pipeline,
            mma_compute_VDO_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dK_pipeline,
        ) = pipeline_args
        # FIXME: TMEM pointers are fixed after cluster sync
        # therefore, this wait shall be trivial.
        tmem.wait_for_alloc()

        # self.tmem_alloc_barrier.arrive_and_wait()
        load_mma_KQ_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_all_stage)
        load_mma_KQ_aux_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_all_stage)
        load_mma_VDO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_all_stage)
        # Persistent mode: advance load consumer state to match mbarrier phase
        if cutlass.const_expr(self.is_persistent):
            n_advance = cumulative_trip_count % Int32(2 * self.load_mma_all_stage)
            for _ in cutlass.range_constexpr(2 * self.load_mma_all_stage):
                if n_advance > Int32(0):
                    load_mma_KQ_consumer_state.advance()
                    load_mma_KQ_aux_consumer_state.advance()
                    load_mma_VDO_consumer_state.advance()
                    n_advance = n_advance - Int32(1)
        load_mma_KQ_release_state = load_mma_KQ_consumer_state.clone()
        load_mma_KQ_aux_release_state = load_mma_KQ_aux_consumer_state.clone()
        load_mma_VDO_release_state = load_mma_VDO_consumer_state.clone()

        mma_compute_KQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_KQ_stage)
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage)
        mma_compute_VDO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_VDO_stage)
        # Persistent mode: correct inter-warp pipeline phases based on cumulative trip count
        if cutlass.const_expr(self.is_persistent):
            n_advance_kq = cumulative_trip_count % Int32(2 * self.mma_compute_KQ_stage)
            for _ in cutlass.range_constexpr(2 * self.mma_compute_KQ_stage):
                if n_advance_kq > Int32(0):
                    mma_compute_KQ_producer_state.advance()
                    n_advance_kq = n_advance_kq - Int32(1)
            n_advance_vdo = cumulative_trip_count % Int32(2 * self.mma_compute_VDO_stage)
            for _ in cutlass.range_constexpr(2 * self.mma_compute_VDO_stage):
                if n_advance_vdo > Int32(0):
                    mma_compute_VDO_producer_state.advance()
                    n_advance_vdo = n_advance_vdo - Int32(1)
            n_advance_ds = cumulative_trip_count % Int32(2 * self.compute_mma_dS_stage)
            for _ in cutlass.range_constexpr(2 * self.compute_mma_dS_stage):
                if n_advance_ds > Int32(0):
                    compute_mma_dS_consumer_state.advance()
                    n_advance_ds = n_advance_ds - Int32(1)
            n_advance_p = cumulative_trip_count % Int32(2 * self.compute_mma_P_stage)
            for _ in cutlass.range_constexpr(2 * self.compute_mma_P_stage):
                if n_advance_p > Int32(0):
                    compute_mma_P_consumer_state.advance()
                    n_advance_p = n_advance_p - Int32(1)
        # Use 2x64 S2T copy for double-slot scale factors
        # SFQ is SFB (B operand's scale factor)
        (
            tiled_copy_s2t_sfq,
            tCsSFQ_compact_s2t,
            tCtSFQ_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ, is_SFA=False)
        (
            tiled_copy_s2t_sfq_h1,
            tCsSFQ_compact_s2t_h1,
            tCtSFQ_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ, tSTtSFQ_h1, is_SFA=False)

        # SFK is SFA (A operand's scale factor)
        (
            tiled_copy_s2t_sfk,
            tCsSFK_compact_s2t,
            tCtSFK_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK, tSTtSFK, is_SFA=True)
        (
            tiled_copy_s2t_sfk_h1,
            tCsSFK_compact_s2t_h1,
            tCtSFK_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFK, tSTtSFK_h1, is_SFA=True)

        # Standard 1x32 MN scales retain the two-slot 2-CTA SFB routing.
        (
            tiled_copy_s2t_sfq_mn,
            tCsSFQ_mn_compact_s2t_mn,
            tCtSFQ_mn_compact_s2t_mn,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFQ_mn, tDKtSFQ_mn, is_SFA=False)

        # SFDO is SFB
        (
            tiled_copy_s2t_sfdo,
            tCsSFDO_compact_s2t,
            tCtSFDO_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO, tDPtSFDO, is_SFA=False)
        (
            tiled_copy_s2t_sfdo_h1,
            tCsSFDO_compact_s2t_h1,
            tCtSFDO_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO, tDPtSFDO_h1, is_SFA=False)

        (
            tiled_copy_s2t_sfv,
            tCsSFV_compact_s2t,
            tCtSFV_compact_s2t,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFV, tDPtSFV, is_SFA=True)
        (
            tiled_copy_s2t_sfv_h1,
            tCsSFV_compact_s2t_h1,
            tCtSFV_compact_s2t_h1,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFV, tDPtSFV_h1, is_SFA=True)

        # SFDO_mn uses the same two-slot 2-CTA SFB routing.
        (
            tiled_copy_s2t_sfdo_mn,
            tCsSFDO_mn_compact_s2t_mn,
            tCtSFDO_mn_compact_s2t_mn,
        ) = cute_common.mainloop_s2t_copy_and_partition_sf_2x64(self, sSFDO_mn, tDVtSFDO_mn, is_SFA=False)

        iter_index = iter_start
        s2t_stage_coord = (
            None,
            None,
            None,
            None,
            load_mma_KQ_consumer_state.index * self.k_halves,
        )
        # if tidx == 256:

        # Prologue: K @ Q
        # Only leader CTA waits for pipeline in 2-CTA mode
        if is_leader_cta:
            load_mma_KQ_pipeline.consumer_wait(load_mma_KQ_consumer_state)

        SFA = tcgen05.Field.SFA
        SFB = tcgen05.Field.SFB
        ACCUMULATE = tcgen05.Field.ACCUMULATE
        C = tStS[None, None, None, mma_compute_KQ_producer_state.index]

        if is_leader_cta:
            mma_compute_KQ_pipeline.producer_acquire(mma_compute_KQ_producer_state)

            # Copy scale factors from SMEM to TMEM
            cute.copy(
                tiled_copy_s2t_sfk,
                tCsSFK_compact_s2t[None, None, None, None, 0],
                tCtSFK_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfk_h1,
                tCsSFK_compact_s2t_h1[None, None, None, None, 1],
                tCtSFK_compact_s2t_h1,
            )
            cute.copy(
                tiled_copy_s2t_sfq,
                tCsSFQ_compact_s2t[s2t_stage_coord],
                tCtSFQ_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfq_h1,
                tCsSFQ_compact_s2t_h1[
                    None,
                    None,
                    None,
                    None,
                    load_mma_KQ_consumer_state.index * self.k_halves + 1,
                ],
                tCtSFQ_compact_s2t_h1,
            )

            # Fence for TMEM stores
            cute.arch.fence_view_async_tmem_store()

            # Fence for TMEM loads before MMA
            cute.arch.fence_view_async_tmem_load()

            KQ_tiled_mma.set(ACCUMULATE, False)
            stage = load_mma_KQ_consumer_state.index
            for k_half in cutlass.range_constexpr(self.k_halves):
                _tSTrK = tSTrK_1 if k_half > 0 else tSTrK_0
                _tSTrQ = tSTrQ_1 if k_half > 0 else tSTrQ_0
                _tSTtSFK_cur = tSTtSFK_h1 if k_half > 0 else tSTtSFK
                _tSTtSFQ_cur = tSTtSFQ_h1 if k_half > 0 else tSTtSFQ
                for k_block in cutlass.range_constexpr(4):
                    KQ_tiled_mma.set(SFA, _tSTtSFK_cur[(None, None, k_block)].iterator)
                    KQ_tiled_mma.set(SFB, _tSTtSFQ_cur[(None, None, k_block)].iterator)
                    cute.gemm(KQ_tiled_mma, C, _tSTrK[None, None, k_block, 0], _tSTrQ[None, None, k_block, stage], C)
                    KQ_tiled_mma.set(ACCUMULATE, True)

            # Only leader CTA commits pipeline in 2-CTA mode
            mma_compute_KQ_pipeline.producer_commit(mma_compute_KQ_producer_state)
        mma_compute_KQ_producer_state.advance()
        # End: S = K * Q

        # prologue V @ dO
        s2t_stage_coord = (None, None, None, None, load_mma_VDO_consumer_state.index * self.k_halves)
        if is_leader_cta:
            load_mma_VDO_pipeline.consumer_wait(load_mma_VDO_consumer_state)
            mma_compute_VDO_pipeline.producer_acquire(mma_compute_VDO_producer_state)

            # Copy scale factors from SMEM to TMEM
            cute.copy(
                tiled_copy_s2t_sfv,
                tCsSFV_compact_s2t[None, None, None, None, 0],
                tCtSFV_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfv_h1,
                tCsSFV_compact_s2t_h1[None, None, None, None, 1],
                tCtSFV_compact_s2t_h1,
            )
            cute.copy(
                tiled_copy_s2t_sfdo,
                tCsSFDO_compact_s2t[s2t_stage_coord],
                tCtSFDO_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfdo_h1,
                tCsSFDO_compact_s2t_h1[
                    None,
                    None,
                    None,
                    None,
                    load_mma_VDO_consumer_state.index * self.k_halves + 1,
                ],
                tCtSFDO_compact_s2t_h1,
            )

            # Fence for TMEM stores
            cute.arch.fence_view_async_tmem_store()

            # Fence for TMEM loads before MMA
            cute.arch.fence_view_async_tmem_load()

            C_vdo = tDPtDP[None, None, None, mma_compute_VDO_producer_state.index]
            VDO_tiled_mma.set(ACCUMULATE, False)
            stage = load_mma_VDO_consumer_state.index
            for k_half in cutlass.range_constexpr(self.k_halves):
                _tDPrV = tDPrV_1 if k_half > 0 else tDPrV_0
                _tdPTrdO = tdPTrdO_1 if k_half > 0 else tdPTrdO_0
                _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                _tDPtSFDO_cur = tDPtSFDO_h1 if k_half > 0 else tDPtSFDO
                for k_block in cutlass.range_constexpr(4):
                    VDO_tiled_mma.set(SFA, _tDPtSFV[(None, None, k_block)].iterator)
                    VDO_tiled_mma.set(SFB, _tDPtSFDO_cur[(None, None, k_block)].iterator)
                    cute.gemm(VDO_tiled_mma, C_vdo, _tDPrV[None, None, k_block, 0], _tdPTrdO[None, None, k_block, stage], C_vdo)
                    VDO_tiled_mma.set(ACCUMULATE, True)
            mma_compute_VDO_pipeline.producer_commit(mma_compute_VDO_producer_state)
        mma_compute_VDO_producer_state.advance()

        load_mma_KQ_consumer_state.advance()
        load_mma_VDO_consumer_state.advance()
        # End: dP = V * dO

        if cutlass.const_expr(self.num_prologue_iters > 1):
            # Prologue: second K @ Q,
            # it will consumer second Q, dO buffer
            # while the first QT, dOT buffer has not been consumed, and should be retained.
            s2t_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_KQ_consumer_state.index * self.k_halves,
            )
            if is_leader_cta:
                load_mma_KQ_pipeline.consumer_wait(load_mma_KQ_consumer_state)
                mma_compute_KQ_pipeline.producer_acquire(mma_compute_KQ_producer_state)
                # Now, it only needs new SFQ
                cute.copy(
                    tiled_copy_s2t_sfq,
                    tCsSFQ_compact_s2t[s2t_stage_coord],
                    tCtSFQ_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfq_h1,
                    tCsSFQ_compact_s2t_h1[
                        None,
                        None,
                        None,
                        None,
                        load_mma_KQ_consumer_state.index * self.k_halves + 1,
                    ],
                    tCtSFQ_compact_s2t_h1,
                )
                C = tStS[None, None, None, mma_compute_KQ_producer_state.index]
                KQ_tiled_mma.set(ACCUMULATE, False)
                stage = load_mma_KQ_consumer_state.index
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tSTrK = tSTrK_1 if k_half > 0 else tSTrK_0
                    _tSTrQ = tSTrQ_1 if k_half > 0 else tSTrQ_0
                    _tSTtSFK_cur = tSTtSFK_h1 if k_half > 0 else tSTtSFK
                    _tSTtSFQ_cur = tSTtSFQ_h1 if k_half > 0 else tSTtSFQ
                    for k_block in cutlass.range_constexpr(4):
                        KQ_tiled_mma.set(SFA, _tSTtSFK_cur[(None, None, k_block)].iterator)
                        KQ_tiled_mma.set(SFB, _tSTtSFQ_cur[(None, None, k_block)].iterator)
                        cute.gemm(KQ_tiled_mma, C, _tSTrK[None, None, k_block, 0], _tSTrQ[None, None, k_block, stage], C)
                        KQ_tiled_mma.set(ACCUMULATE, True)
                mma_compute_KQ_pipeline.producer_commit(mma_compute_KQ_producer_state)
            mma_compute_KQ_producer_state.advance()
            # End: second S = K * Q

            # Prologue: second V @ dO
            s2t_stage_coord = (None, None, None, None, load_mma_VDO_consumer_state.index * self.k_halves)
            if is_leader_cta:
                load_mma_VDO_pipeline.consumer_wait(load_mma_VDO_consumer_state)
                mma_compute_VDO_pipeline.producer_acquire(mma_compute_VDO_producer_state)
                cute.copy(
                    tiled_copy_s2t_sfdo,
                    tCsSFDO_compact_s2t[s2t_stage_coord],
                    tCtSFDO_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfdo_h1,
                    tCsSFDO_compact_s2t_h1[
                        None,
                        None,
                        None,
                        None,
                        load_mma_VDO_consumer_state.index * self.k_halves + 1,
                    ],
                    tCtSFDO_compact_s2t_h1,
                )
                C_vdo = tDPtDP[None, None, None, mma_compute_VDO_producer_state.index]
                VDO_tiled_mma.set(ACCUMULATE, False)
                stage = load_mma_VDO_consumer_state.index
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tDPrV = tDPrV_1 if k_half > 0 else tDPrV_0
                    _tdPTrdO = tdPTrdO_1 if k_half > 0 else tdPTrdO_0
                    _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                    _tDPtSFDO_cur = tDPtSFDO_h1 if k_half > 0 else tDPtSFDO
                    for k_block in cutlass.range_constexpr(4):
                        VDO_tiled_mma.set(SFA, _tDPtSFV[(None, None, k_block)].iterator)
                        VDO_tiled_mma.set(SFB, _tDPtSFDO_cur[(None, None, k_block)].iterator)
                        cute.gemm(VDO_tiled_mma, C_vdo, _tDPrV[None, None, k_block, 0], _tdPTrdO[None, None, k_block, stage], C_vdo)
                        VDO_tiled_mma.set(ACCUMULATE, True)
                mma_compute_VDO_pipeline.producer_commit(mma_compute_VDO_producer_state)
            mma_compute_VDO_producer_state.advance()

            load_mma_KQ_consumer_state.advance()
            load_mma_VDO_consumer_state.advance()
            # End: second dP = V * dO

        # Consumer state tracks Q and dO buffers
        # Release state tracks those unused dOT and QT buffers

        if is_leader_cta:
            dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        while iter_count - self.num_prologue_iters > 0:
            if iter_index == iter_end:
                iter_index = iter_start

            peak_P_consumer_status = cutlass.Boolean(False)
            if is_leader_cta:
                peak_P_consumer_status = compute_mma_P_pipeline.consumer_try_wait(compute_mma_P_consumer_state)

            # Start: dK = dS * QT
            # We need to acquire dP here before dQ, because dQ and dP MMA may share hardware accumulators
            if is_leader_cta:
                load_mma_KQ_aux_pipeline.consumer_wait(load_mma_KQ_aux_release_state)
                cute.copy(
                    tiled_copy_s2t_sfq_mn,
                    tCsSFQ_mn_compact_s2t_mn[None, None, None, None, load_mma_KQ_aux_release_state.index],
                    tCtSFQ_mn_compact_s2t_mn,
                )
                cute.copy(
                    tiled_copy_s2t_sfdo_mn,
                    tCsSFDO_mn_compact_s2t_mn[None, None, None, None, load_mma_VDO_release_state.index],
                    tCtSFDO_mn_compact_s2t_mn,
                )
            peak_dS_consumer_status = cutlass.Boolean(False)
            if is_leader_cta:
                compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state, peak_P_consumer_status)
                peak_dS_consumer_status = compute_mma_dS_pipeline.consumer_try_wait(compute_mma_dS_consumer_state)
                # dV = P @ dOT
                for k_block in cutlass.range_constexpr(cute.size(tDVrP, mode=[2])):
                    sf_kblock_coord = (None, None, k_block)
                    PdO_tiled_mma.set(
                        tcgen05.Field.SFA,
                        tDVtSFP[sf_kblock_coord].iterator,
                    )
                    PdO_tiled_mma.set(
                        tcgen05.Field.SFB,
                        tDVtSFDO_mn[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        PdO_tiled_mma,
                        tDVtDV[None, None, None, 0],
                        tDVrP[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tDVrDOT[None, None, k_block, load_mma_VDO_release_state.index],
                        tDVtDV[None, None, None, 0],
                    )
                    PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            if is_leader_cta:
                compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            # Start next S = K * Q before current dK. The old QT load stage
            # stays live through load_mma_KQ_release_state for dK below.
            s2t_stage_coord = (
                None,
                None,
                None,
                None,
                load_mma_KQ_consumer_state.index * self.k_halves,
            )
            peak_mma_compute_KQ_status = cutlass.Boolean(False)
            peak_KQ_consumer_status = cutlass.Boolean(False)
            if is_leader_cta:
                peak_mma_compute_KQ_status = mma_compute_KQ_pipeline.producer_try_acquire(mma_compute_KQ_producer_state)
                peak_KQ_consumer_status = load_mma_KQ_pipeline.consumer_try_wait(load_mma_KQ_consumer_state)
                load_mma_KQ_pipeline.consumer_wait(load_mma_KQ_consumer_state, peak_KQ_consumer_status)
                mma_compute_KQ_pipeline.producer_acquire(mma_compute_KQ_producer_state, peak_mma_compute_KQ_status)

                cute.copy(
                    tiled_copy_s2t_sfq,
                    tCsSFQ_compact_s2t[s2t_stage_coord],
                    tCtSFQ_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfq_h1,
                    tCsSFQ_compact_s2t_h1[
                        None,
                        None,
                        None,
                        None,
                        load_mma_KQ_consumer_state.index * self.k_halves + 1,
                    ],
                    tCtSFQ_compact_s2t_h1,
                )
                C = tStS[None, None, None, mma_compute_KQ_producer_state.index]
                KQ_tiled_mma.set(ACCUMULATE, False)
                stage = load_mma_KQ_consumer_state.index
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tSTrK = tSTrK_1 if k_half > 0 else tSTrK_0
                    _tSTrQ = tSTrQ_1 if k_half > 0 else tSTrQ_0
                    _tSTtSFK_cur = tSTtSFK_h1 if k_half > 0 else tSTtSFK
                    _tSTtSFQ_cur = tSTtSFQ_h1 if k_half > 0 else tSTtSFQ
                    for k_block in cutlass.range_constexpr(4):
                        KQ_tiled_mma.set(SFA, _tSTtSFK_cur[(None, None, k_block)].iterator)
                        KQ_tiled_mma.set(SFB, _tSTtSFQ_cur[(None, None, k_block)].iterator)
                        cute.gemm(KQ_tiled_mma, C, _tSTrK[None, None, k_block, 0], _tSTrQ[None, None, k_block, stage], C)
                        KQ_tiled_mma.set(ACCUMULATE, True)

                mma_compute_KQ_pipeline.producer_commit(mma_compute_KQ_producer_state)
            mma_compute_KQ_producer_state.advance()
            load_mma_KQ_consumer_state.advance()
            # End: S = K * Q

            if is_leader_cta:
                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state, peak_dS_consumer_status)
                load_mma_KQ_pipeline.consumer_release(load_mma_KQ_release_state)
                # dK = dS * QT
                for k_block in cutlass.range_constexpr(cute.size(tDKrDS, mode=[2])):
                    sf_kblock_coord = (None, None, k_block)
                    dSQ_tiled_mma.set(
                        tcgen05.Field.SFA,
                        tDKtSFDS[sf_kblock_coord].iterator,
                    )
                    dSQ_tiled_mma.set(
                        tcgen05.Field.SFB,
                        tDKtSFQ_mn[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        dSQ_tiled_mma,
                        tDKtDK[None, None, None, 0],
                        tDKrDS[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tDKrQT[None, None, k_block, load_mma_KQ_aux_release_state.index],
                        tDKtDK[None, None, None, 0],
                    )
                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            load_mma_KQ_release_state.advance()
            compute_mma_dS_consumer_state.advance()

            peak_VDO_consumer_status = cutlass.Boolean(False)
            if is_leader_cta:
                peak_VDO_consumer_status = load_mma_VDO_pipeline.consumer_try_wait(load_mma_VDO_consumer_state)

                # now, these dOT and QT buffers could be released
                load_mma_KQ_aux_pipeline.consumer_release(load_mma_KQ_aux_release_state)
                load_mma_VDO_pipeline.consumer_release(load_mma_VDO_release_state)
            load_mma_KQ_aux_release_state.advance()
            load_mma_VDO_release_state.advance()

            # Start: dP = V * dO
            s2t_stage_coord = (None, None, None, None, load_mma_VDO_consumer_state.index * self.k_halves)
            peak_mma_compute_VDO_status = cutlass.Boolean(False)
            if is_leader_cta:
                peak_mma_compute_VDO_status = mma_compute_VDO_pipeline.producer_try_acquire(mma_compute_VDO_producer_state)
                load_mma_VDO_pipeline.consumer_wait(load_mma_VDO_consumer_state, peak_VDO_consumer_status)
                mma_compute_VDO_pipeline.producer_acquire(mma_compute_VDO_producer_state, peak_mma_compute_VDO_status)
                cute.copy(
                    tiled_copy_s2t_sfdo,
                    tCsSFDO_compact_s2t[s2t_stage_coord],
                    tCtSFDO_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfdo_h1,
                    tCsSFDO_compact_s2t_h1[
                        None,
                        None,
                        None,
                        None,
                        load_mma_VDO_consumer_state.index * self.k_halves + 1,
                    ],
                    tCtSFDO_compact_s2t_h1,
                )
                # dP = V * dO
                C_vdo = tDPtDP[None, None, None, mma_compute_VDO_producer_state.index]
                VDO_tiled_mma.set(ACCUMULATE, False)
                stage = load_mma_VDO_consumer_state.index
                for k_half in cutlass.range_constexpr(self.k_halves):
                    _tDPrV = tDPrV_1 if k_half > 0 else tDPrV_0
                    _tdPTrdO = tdPTrdO_1 if k_half > 0 else tdPTrdO_0
                    _tDPtSFV = tDPtSFV_h1 if k_half > 0 else tDPtSFV
                    _tDPtSFDO_cur = tDPtSFDO_h1 if k_half > 0 else tDPtSFDO
                    for k_block in cutlass.range_constexpr(4):
                        VDO_tiled_mma.set(SFA, _tDPtSFV[(None, None, k_block)].iterator)
                        VDO_tiled_mma.set(SFB, _tDPtSFDO_cur[(None, None, k_block)].iterator)
                        cute.gemm(VDO_tiled_mma, C_vdo, _tDPrV[None, None, k_block, 0], _tdPTrdO[None, None, k_block, stage], C_vdo)
                        VDO_tiled_mma.set(ACCUMULATE, True)
                mma_compute_VDO_pipeline.producer_commit(mma_compute_VDO_producer_state)
            mma_compute_VDO_producer_state.advance()

            load_mma_VDO_consumer_state.advance()
            # End: dP = V * dO

            iter_count -= 1
            iter_index += 1

        # Signal to the epilogue that dV is ready
        # Only leader CTA acquires pipeline
        if is_leader_cta:
            mma_compute_dK_pipeline.producer_acquire(mma_compute_dK_producer_state)

        for i in cutlass.range(cutlass.min(self.num_prologue_iters, iter_count_origin), unroll_full=True):
            if is_leader_cta:
                load_mma_KQ_aux_pipeline.consumer_wait(load_mma_KQ_aux_release_state)
                cute.copy(
                    tiled_copy_s2t_sfq_mn,
                    tCsSFQ_mn_compact_s2t_mn[None, None, None, None, load_mma_KQ_aux_release_state.index],
                    tCtSFQ_mn_compact_s2t_mn,
                )
                cute.copy(
                    tiled_copy_s2t_sfdo_mn,
                    tCsSFDO_mn_compact_s2t_mn[None, None, None, None, load_mma_VDO_release_state.index],
                    tCtSFDO_mn_compact_s2t_mn,
                )

            if is_leader_cta:
                compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)
                # dV = P @ dOT
                for k_block in cutlass.range(cute.size(tDVrP, mode=[2])):
                    sf_kblock_coord = (None, None, k_block)
                    PdO_tiled_mma.set(
                        tcgen05.Field.SFA,
                        tDVtSFP[sf_kblock_coord].iterator,
                    )
                    PdO_tiled_mma.set(
                        tcgen05.Field.SFB,
                        tDVtSFDO_mn[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        PdO_tiled_mma,
                        tDVtDV[None, None, None, 0],
                        tDVrP[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tDVrDOT[None, None, k_block, load_mma_VDO_release_state.index],
                        tDVtDV[None, None, None, 0],
                    )
                    PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if is_leader_cta:
                compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()
            if is_leader_cta:
                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                # dK = dS * QT
                for k_block in cutlass.range_constexpr(cute.size(tDKrDS, mode=[2])):
                    sf_kblock_coord = (None, None, k_block)
                    dSQ_tiled_mma.set(
                        tcgen05.Field.SFA,
                        tDKtSFDS[sf_kblock_coord].iterator,
                    )
                    dSQ_tiled_mma.set(
                        tcgen05.Field.SFB,
                        tDKtSFQ_mn[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        dSQ_tiled_mma,
                        tDKtDK[None, None, None, 0],
                        tDKrDS[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tDKrQT[None, None, k_block, load_mma_KQ_aux_release_state.index],
                        tDKtDK[None, None, None, 0],
                    )
                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                load_mma_KQ_pipeline.consumer_release(load_mma_KQ_release_state)
                load_mma_KQ_aux_pipeline.consumer_release(load_mma_KQ_aux_release_state)
                load_mma_VDO_pipeline.consumer_release(load_mma_VDO_release_state)
            load_mma_KQ_release_state.advance()
            load_mma_KQ_aux_release_state.advance()
            load_mma_VDO_release_state.advance()

            if is_leader_cta:
                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()
        if is_leader_cta:
            mma_compute_dK_pipeline.producer_commit(mma_compute_dK_producer_state)
        mma_compute_dK_producer_state.advance()

    @cute.jit
    def compute(
        self,
        tStS: cute.Tensor,
        tDPtDP: cute.Tensor,
        tDKtSFDS: cute.Tensor,
        tDVtSFP: cute.Tensor,
        sDS: cute.Tensor,
        sDS_scale_exchange: cute.Tensor,
        sP: cute.Tensor,
        blk_coord: cute.Coord,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        is_leader_cta: Boolean,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        # (mma_compute_KQ_pipeline, mma_compute_VDO_pipeline, compute_mma_P_pipeline, compute_mma_dS_pipeline)
        pipeline_args: tuple,
        cumulative_trip_count: Int32 = Int32(0),  # accumulated trip_count across persistent tiles
        # Start of the KV tile's FULL trip range. iter_start/iter_end may be a
        # chunk of it (s_q_split); the masked-leading/trailing tile counts are
        # relative to the full range, so the tile classification below uses this.
        iter_start_global: Optional[Int32] = None,
    ):
        """Softmax / dS / P production for one KV tile (compute warps); see the module docstring."""
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, _, _ = problem_shape
        _, blk_coord_k, _, _ = blk_coord

        # FIXME: perhaps make it uniform register
        wg_idx_valid = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        if cutlass.const_expr(iter_start_global is None):
            iter_start_global = iter_start
        num_warp_groups = self.num_compute_warps // self.num_compute_0_warps

        tidx = tidx % 128

        iter_index = iter_start
        (
            mma_compute_KQ_pipeline,
            mma_compute_VDO_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
        ) = pipeline_args

        mma_compute_KQ_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_KQ_stage)

        mma_compute_VDO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_VDO_stage)
        compute_mma_P_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_P_stage)
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage)
        # Persistent mode: correct inter-warp pipeline phases based on cumulative trip count
        if cutlass.const_expr(self.is_persistent):
            n_advance_kq = cumulative_trip_count % Int32(2 * self.mma_compute_KQ_stage)
            for _ in cutlass.range_constexpr(2 * self.mma_compute_KQ_stage):
                if n_advance_kq > Int32(0):
                    mma_compute_KQ_consumer_state.advance()
                    n_advance_kq = n_advance_kq - Int32(1)
            n_advance_vdo = cumulative_trip_count % Int32(2 * self.mma_compute_VDO_stage)
            for _ in cutlass.range_constexpr(2 * self.mma_compute_VDO_stage):
                if n_advance_vdo > Int32(0):
                    mma_compute_VDO_consumer_state.advance()
                    n_advance_vdo = n_advance_vdo - Int32(1)
            n_advance_p = cumulative_trip_count % Int32(2 * self.compute_mma_P_stage)
            for _ in cutlass.range_constexpr(2 * self.compute_mma_P_stage):
                if n_advance_p > Int32(0):
                    compute_mma_P_producer_state.advance()
                    n_advance_p = n_advance_p - Int32(1)
            n_advance_ds = cumulative_trip_count % Int32(2 * self.compute_mma_dS_stage)
            for _ in cutlass.range_constexpr(2 * self.compute_mma_dS_stage):
                if n_advance_ds > Int32(0):
                    compute_mma_dS_producer_state.advance()
                    n_advance_ds = n_advance_ds - Int32(1)

        load_Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_all_stage)
        # Persistent mode: advance load consumer state to match mbarrier phase
        if cutlass.const_expr(self.is_persistent):
            n_advance = cumulative_trip_count % Int32(2 * self.load_mma_all_stage)
            for _ in cutlass.range_constexpr(2 * self.load_mma_all_stage):
                if n_advance > Int32(0):
                    load_Q_consumer_state.advance()
                    n_advance = n_advance - Int32(1)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )
        tStS_0 = tStS[(None, None), 0, 0, 0]
        tDPtDP_0 = tDPtDP[(None, None), 0, 0, 0]

        # Use full mma_tiler for identity tensors - matches tmem tensor shape
        cKQ = cute.make_identity_tensor(cute.select(self.KQ_cta_tiler, mode=[0, 1]))
        cVDO = cute.make_identity_tensor(cute.select(self.VDO_cta_tiler, mode=[0, 1]))

        dp_idx = tidx % 128
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS_0)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cKQ_full = thr_t2r.partition_D(cKQ)
        tTR_cKQ = self.split_wg(tTR_cKQ_full, num_warp_groups, wg_idx_valid)
        tTR_rKQ = cute.make_rmem_tensor(tTR_cKQ.shape, self.acc_dtype)

        tTR_cVDO_full = thr_t2r.partition_D(cVDO)
        tTR_cVDO = self.split_wg(tTR_cVDO_full, num_warp_groups, wg_idx_valid)
        tTR_rVDO = cute.make_rmem_tensor(tTR_cVDO.shape, self.acc_dtype)

        masked_leading_count = fmha_masks.FusedMask.get_masked_leading_count(
            self.mask_type,
            blk_coord,
            self.mask_cta_tiler,
            Q,
            K,
            window_size_left,
            window_size_right,
        )
        unmasked_count = fmha_masks.FusedMask.get_unmasked_trip_count(
            self.mask_type,
            blk_coord,
            self.mask_cta_tiler,
            Q,
            K,
            window_size_left,
            window_size_right,
        )
        masked_trailing_count = fmha_masks.FusedMask.get_masked_trailing_count(
            self.mask_type,
            blk_coord,
            self.mask_cta_tiler,
            Q,
            K,
            window_size_left,
            window_size_right,
        )

        # Fixed P quantization scale (see __init__): P_q = P * 2**p_scale_log2,
        # descaled in the MMA by the E8M0 byte 127 - p_scale_log2.
        P_scale = cutlass.Float32(float(2**self.p_scale_log2))
        P_scale_sf = 127 - self.p_scale_log2
        if wg_idx_valid == 0:
            if cutlass.const_expr(not self.online_ds_scale):
                d256_primitives.store_identity_mxfp8_scales_to_tmem(self, tDKtSFDS, tidx)

            # set SFP to 1
            tDVtSFP_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)),
                self.sf_dtype,
            )
            tDVtSFP_tiled_copy = tcgen05.make_tmem_copy(tDVtSFP_atom, cute.filter_zeros(tDVtSFP))
            tDVtSFP_thr_copy = tDVtSFP_tiled_copy.get_slice(tidx)
            tDVtSFP_thr = tDVtSFP_thr_copy.partition_D(cute.filter_zeros(tDVtSFP))
            tDVrSFP = cute.make_rmem_tensor_like(cute.filter_zeros(tDVtSFP), self.sf_dtype)
            tDVrSFP_thr = tDVtSFP_thr_copy.partition_S(tDVrSFP)

            tDVrSFP_thr_int_ptr = cute.recast_ptr(tDVrSFP_thr.iterator, dtype=cutlass.Int8)
            tDVrSFP_thr_int = cute.make_tensor(tDVrSFP_thr_int_ptr, tDVrSFP_thr.layout)
            tDVrSFP_thr_int.fill(P_scale_sf)

            cute.copy(
                tDVtSFP_tiled_copy,
                tDVrSFP_thr,
                tDVtSFP_thr,
            )
            # end of set SFP to 1

            cute.arch.fence_view_async_tmem_store()
        log2_e = Float32(math.log2(math.e))
        softmax_scale_log2_e = scale_softmax * log2_e
        warp_rank = (tidx % 128) // 32

        while iter_count > 0:

            iter_num = iter_index - iter_start_global + 1
            is_residual_q = Boolean(False)
            is_residual_q = iter_index * self.tile_shape_Q + self.tile_shape_Q > Q

            is_masked_tile = (
                is_residual_q
                or iter_num <= masked_leading_count
                or (iter_num > masked_leading_count + unmasked_count and iter_num <= masked_leading_count + unmasked_count + masked_trailing_count)
            )

            # Wait for S
            peak_P_producer_status = cutlass.Boolean(False)
            if is_leader_cta:
                peak_P_producer_status = compute_mma_P_pipeline.producer_try_acquire(compute_mma_P_producer_state)
            mma_compute_KQ_pipeline.consumer_wait(mma_compute_KQ_consumer_state)
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state, peak_P_producer_status)

            tStS_i = tStS[(None, None), 0, 0, mma_compute_KQ_consumer_state.index]
            tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS_i)
            thr_t2r = tiled_t2r.get_slice(dp_idx)
            tTR_tKQ_full = thr_t2r.partition_S(tStS_i)
            tTR_tKQ = self.split_wg(tTR_tKQ_full, num_warp_groups, wg_idx_valid)

            cute.copy(tiled_t2r, tTR_tKQ, tTR_rKQ)

            if is_masked_tile:
                fmha_masks.FusedMask.apply_mask(
                    self.mask_type,
                    tTR_rKQ,
                    tTR_cKQ,
                    Q,
                    K,
                    window_size_left,
                    window_size_right,
                    lambda index_k, index_q: (
                        index_q + iter_index * self.tile_shape_Q,
                        index_k + blk_coord_k * (self.tile_shape_K // 2),
                    ),
                )

            for i in cutlass.range(0, cute.size(tTR_rKQ), 2, unroll_full=True):
                lse = (
                    sLSE[
                        cute.get(tTR_cKQ[i], mode=[1]),
                        load_Q_consumer_state.index,
                    ],
                    sLSE[
                        cute.get(tTR_cKQ[i + 1], mode=[1]),
                        load_Q_consumer_state.index,
                    ],
                )

                tTR_rKQ[i], tTR_rKQ[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rKQ[i], tTR_rKQ[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rKQ[i] = cute.math.exp2(tTR_rKQ[i], fastmath=True)
                tTR_rKQ[i + 1] = cute.math.exp2(tTR_rKQ[i + 1], fastmath=True)

            cute.arch.fence_view_async_tmem_load()

            # quantize P
            tTR_rP_scaled = cute.make_rmem_tensor_like(tTR_rKQ)
            tTR_rP_scaled.store(tTR_rKQ.load() * P_scale)
            tTR_rP = cute_common.quantize(tTR_rP_scaled, 4, LOW_PRECISION_TYPE)

            # copy P to SMEM. Start from the proven single-WG per-thread SMEM
            # view, reshape it to the same fragment layout as partition_D, then
            # split the last mode across compute warpgroups.
            sP_slice = sP[None, None, None, compute_mma_dS_producer_state.index]
            sP_slice_divided = cute.logical_divide(sP_slice, ((32, None), None, 2))
            sP_slice_warp = cute.coalesce(sP_slice_divided[(((None, warp_rank % 2), None), 0, (None, warp_rank // 2))])
            sP_slice_thread_full = cute.coalesce(sP_slice_warp[tidx % 32, None])
            sP_slice_thread_full = cute.composition(sP_slice_thread_full, cute.make_layout(tTR_cKQ_full.shape))
            sP_slice_thread = self.split_wg(sP_slice_thread_full, num_warp_groups, wg_idx_valid)

            cute.autovec_copy(tTR_rP, sP_slice_thread)

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

            peak_dS_producer_status = cutlass.Boolean(False)
            if is_leader_cta:
                peak_dS_producer_status = compute_mma_dS_pipeline.producer_try_acquire(compute_mma_dS_producer_state)

            # Release S
            mma_compute_KQ_pipeline.consumer_release(mma_compute_KQ_consumer_state)
            mma_compute_KQ_consumer_state.advance()

            # Wait for dP
            mma_compute_VDO_pipeline.consumer_wait(mma_compute_VDO_consumer_state)
            compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state, peak_dS_producer_status)

            # Compute dS = dsoftmax(P, dP, sum_OdO)
            tTR_tVDO_full = thr_t2r.partition_S(tDPtDP[(None, None), 0, 0, mma_compute_VDO_consumer_state.index])
            tTR_tVDO = self.split_wg(tTR_tVDO_full, num_warp_groups, wg_idx_valid)
            cute.copy(tiled_t2r, tTR_tVDO, tTR_rVDO)

            for i in cutlass.range(0, cute.size(tTR_rVDO), 2, unroll_full=True):
                _sum_OdO = (
                    sSum_OdO[
                        cute.get(tTR_cVDO[i], mode=[1]),
                        load_Q_consumer_state.index,
                    ],
                    sSum_OdO[
                        cute.get(tTR_cVDO[i + 1], mode=[1]),
                        load_Q_consumer_state.index,
                    ],
                )
                tTR_rVDO[i], tTR_rVDO[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rVDO[i], tTR_rVDO[i + 1]),
                    _sum_OdO,
                )
                tTR_rVDO[i], tTR_rVDO[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rVDO[i], tTR_rVDO[i + 1]),
                    (tTR_rKQ[i], tTR_rKQ[i + 1]),
                )

            tTR_rdPT_scaled = cute.make_rmem_tensor_like(tTR_rVDO)
            tTR_rdPT_scaled.store(tTR_rVDO.load())
            if cutlass.const_expr(self.online_ds_scale):
                partial_amax_0 = Float32(0.0)
                partial_amax_1 = Float32(0.0)
                for i in cutlass.range_constexpr(16):
                    value_0 = tTR_rdPT_scaled[i]
                    value_1 = tTR_rdPT_scaled[i + 16]
                    partial_amax_0 = cute.arch.fmax(partial_amax_0, cute.arch.fmax(value_0, -value_0))
                    partial_amax_1 = cute.arch.fmax(partial_amax_1, cute.arch.fmax(value_1, -value_1))
                dS_row = cute.get(tTR_cVDO[0], mode=[0])
                dS_group_block = tidx // 64
                partial_scale_0, _ = cute_common.cvt_amax_to_e8m0_rp(partial_amax_0)
                partial_scale_1, _ = cute_common.cvt_amax_to_e8m0_rp(partial_amax_1)
                partial_scale_tile = cute.make_tensor(
                    sDS_scale_exchange.iterator,
                    cute.make_layout((2, 2, 2, 64), stride=(256, 128, 64, 1)),
                )
                partial_scale_tile[wg_idx_valid, 0, dS_group_block, dS_row] = partial_scale_0
                partial_scale_tile[wg_idx_valid, 1, dS_group_block, dS_row] = partial_scale_1
                cute.arch.fence_proxy("async.shared", space="cta")
                self.dS_scale_exchange_barrier.arrive_and_wait()

                scale_0_value = cute.arch.fmax(
                    partial_scale_tile[0, 0, dS_group_block, dS_row].to(Float32),
                    partial_scale_tile[1, 0, dS_group_block, dS_row].to(Float32),
                )
                scale_1_value = cute.arch.fmax(
                    partial_scale_tile[0, 1, dS_group_block, dS_row].to(Float32),
                    partial_scale_tile[1, 1, dS_group_block, dS_row].to(Float32),
                )
                dS_scale_0, inv_scale_0 = cute_common.cvt_amax_to_e8m0_rp(scale_0_value * Float32(448.0))
                dS_scale_1, inv_scale_1 = cute_common.cvt_amax_to_e8m0_rp(scale_1_value * Float32(448.0))

                tTR_rdPT_normalized = cute.make_rmem_tensor_like(tTR_rdPT_scaled)
                for i in cutlass.range_constexpr(0, 16, 2):
                    tTR_rdPT_normalized[i], tTR_rdPT_normalized[i + 1] = cute.arch.mul_packed_f32x2(
                        (tTR_rdPT_scaled[i], tTR_rdPT_scaled[i + 1]),
                        (inv_scale_0, inv_scale_0),
                    )
                    tTR_rdPT_normalized[i + 16], tTR_rdPT_normalized[i + 17] = cute.arch.mul_packed_f32x2(
                        (tTR_rdPT_scaled[i + 16], tTR_rdPT_scaled[i + 17]),
                        (inv_scale_1, inv_scale_1),
                    )
                tTR_rdST = cute_common.quantize(tTR_rdPT_normalized, 4, LOW_PRECISION_TYPE)

                self.dS_scale_exchange_barrier.arrive_and_wait()
                if wg_idx_valid == 0:
                    dS_group = dS_group_block * 2
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
                cute.arch.fence_proxy("async.shared", space="cta")
                self.dS_scale_exchange_barrier.arrive_and_wait()
                if wg_idx_valid == 0 and tidx == 0:
                    d256_primitives.copy_mxfp8_scale_tile_to_tmem(sDS_scale_exchange, tDKtSFDS)
                if wg_idx_valid == 0:
                    cute.arch.fence_view_async_tmem_store()
                self.dS_scale_exchange_barrier.arrive_and_wait()
            else:
                tTR_rdST = cute_common.quantize(tTR_rdPT_scaled, 4, LOW_PRECISION_TYPE)

            # Release dP
            cute.arch.fence_view_async_tmem_load()
            mma_compute_VDO_pipeline.consumer_release(mma_compute_VDO_consumer_state)
            mma_compute_VDO_consumer_state.advance()

            load_Q_consumer_state.advance()

            # copy dS to SMEM. Keep the same vectorized mapping as P.
            sdS_slice = sDS[
                None, None, None, compute_mma_dS_producer_state.index
            ]  # tensor<ptr<f8E4M3FN, smem, align<1024>, S<3,4,3>> o ((64,32),1,4):((128,1),0,32)>
            sdS_slice_divided = cute.logical_divide(sdS_slice, ((32, None), None, 2))
            sdS_slice_warp = cute.coalesce(sdS_slice_divided[(((None, warp_rank % 2), None), 0, (None, warp_rank // 2))])
            sdS_slice_thread_full = cute.coalesce(sdS_slice_warp[tidx % 32, None])
            sdS_slice_thread_full = cute.composition(sdS_slice_thread_full, cute.make_layout(tTR_cVDO_full.shape))
            sdS_slice_thread = self.split_wg(sdS_slice_thread_full, num_warp_groups, wg_idx_valid)

            cute.autovec_copy(tTR_rdST, sdS_slice_thread)

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            iter_count -= 1
            iter_index += 1
            if iter_index == iter_end:
                iter_index = iter_start

    @cute.jit
    def epilogue_zero(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
    ):
        """Write zeros to this CTA's dK/dV tile (the epilogue's tile geometry) --
        for KV tiles whose mainloop was skipped because no query row attends
        them. Plain per-element stores from the compute warps; these tiles are
        rare (the causal tail past S_q) so no vectorization is needed."""
        tidx, _, _ = cute.arch.thread_idx()
        _, K, D, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord
        mdK = cute.make_tensor(
            dK.iterator + cute.assume(blk_offset[1] * dK.stride[0], divby=64),
            cute.make_layout((K, self.tile_shape_dKdV_K, HB), stride=dK.stride),
        )
        mdV = cute.make_tensor(
            dV.iterator + cute.assume(blk_offset[1] * dV.stride[0], divby=64),
            cute.make_layout((K, self.tile_shape_dKdV_K, HB), stride=dV.stride),
        )
        rows = self.dSQ_cta_tiler[0]
        cols = self.tile_shape_dKdV_K
        n_threads = self.num_compute_warps * self.threads_per_warp
        zero = Float32(0.0).to(self.element_dtype)
        for i in cutlass.range(tidx, rows * cols, n_threads):
            r = blk_coord_k * rows + i // cols
            c = i % cols
            if r < K:
                mdK[(r, c, blk_coord_batch)] = zero
                mdV[(r, c, blk_coord_batch)] = zero

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: Float32,
        is_leader_cta: Boolean,
        # (mma_compute_dQ_pipeline, mma_compute_dQ_consumer_state)
        pipeline_args: tuple,
    ):
        """Compute warps: scale the TMEM dK/dV accumulators and store this CTA's tile to global memory."""
        tidx, _, _ = cute.arch.thread_idx()
        _, K, D, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord
        mma_compute_dK_pipeline, mma_compute_Q_consumer_state = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )

        mdK = cute.make_tensor(
            dK.iterator + cute.assume(blk_offset[1] * dK.stride[0], divby=64),
            cute.make_layout((K, self.tile_shape_dKdV_K, HB), stride=dK.stride),
        )
        mdV = cute.make_tensor(
            dV.iterator + cute.assume(blk_offset[1] * dV.stride[0], divby=64),
            cute.make_layout((K, self.tile_shape_dKdV_K, HB), stride=dV.stride),
        )

        # Use full mma_tiler for local_tile - matches tmem tensor shape
        # Use full mma_tiler for identity tensor - matches tmem tensor shape
        gdK = cute.local_tile(
            mdK,
            (
                self.dSQ_cta_tiler[0],
                self.dSQ_cta_tiler[1],
            ),
            (None, None, None),
        )
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset((blk_coord_k * self.dSQ_cta_tiler[0], 0), cute.make_identity_tensor((self.dSQ_cta_tiler[0], self.dSQ_cta_tiler[1])))

        gdV = cute.local_tile(
            mdV,
            (
                self.PdO_cta_tiler[0],
                self.PdO_cta_tiler[1],
            ),
            (None, None, None),
        )
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]
        cdV = cute.domain_offset((blk_coord_k * self.PdO_cta_tiler[0], 0), cute.make_identity_tensor((self.PdO_cta_tiler[0], self.PdO_cta_tiler[1])))

        dp_idx = tidx % 128

        mma_compute_dK_pipeline.consumer_wait(mma_compute_Q_consumer_state)

        tdKtdK = tdKtdK[(None, None), 0, 0, 0]
        (
            tiled_t2r_dK,
            tTR_tdK,
            tTR_rdK,
            tTR_gdK,
            tTR_cdK,
        ) = cute_common.epilogue_tmem_copy_and_partition(load_op, tdKtdK, cdK, gdK, dp_idx, self.acc_dtype)

        cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)
        # if bidx == 1 and tidx == 0:
        for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
            # if tidx == 0 and bidx == 1:
            tTR_rdK[i] = scale_softmax * tTR_rdK[i]
            # if tidx == 0 and bidx == 1:

        cute.arch.fence_view_async_tmem_load()
        cute_common.store(self, tTR_gdK, tTR_rdK, tTR_cdK, (K, D))

        tdVtdV = tdVtdV[(None, None), 0, 0, 0]
        (
            tiled_t2r_dV,
            tTR_tdV,
            tTR_rdV,
            tTR_gdV,
            tTR_cdV,
        ) = cute_common.epilogue_tmem_copy_and_partition(load_op, tdVtdV, cdV, gdV, dp_idx, self.acc_dtype)

        cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)
        cute.arch.fence_view_async_tmem_load()
        cute_common.store(self, tTR_gdV, tTR_rdV, tTR_cdV, (K, D))

        mma_compute_dK_pipeline.consumer_release(mma_compute_Q_consumer_state)
        mma_compute_Q_consumer_state.advance()

    def _make_and_init_load_mma_pipeline(self, load_mma_mbar_ptr, cluster_layout_vmnk, tx_count):
        """Create and initialise one TMA-producer / MMA-consumer pipeline with the given depth."""
        return cute_common.make_tma_umma_pipeline(
            load_mma_mbar_ptr,
            self.load_mma_all_stage,
            tx_count,
            cluster_layout_vmnk,
            len([self.load_warp_id]),
            len([self.mma_warp_id]),
        )

    def make_and_init_load_mma_KQ_pipeline(self, load_mma_KQ_mbar_ptr, cluster_layout_vmnk):
        """Load -> MMA pipeline for the K and Q operand stages."""
        tx_count = self.tma_copy_Q_bytes * 2
        tx_count += self.tma_copy_sfQ_bytes * self.k_halves * 2
        tx_count += self.tma_copy_LSE_bytes
        return self._make_and_init_load_mma_pipeline(load_mma_KQ_mbar_ptr, cluster_layout_vmnk, tx_count)

    def make_and_init_load_mma_KQ_aux_pipeline(self, load_mma_KQ_aux_mbar_ptr, cluster_layout_vmnk):
        """Load -> MMA pipeline for the K/Q auxiliary (scale-factor) stages."""
        tx_count = self.tma_copy_QT_bytes * 2
        tx_count += self.tma_copy_sfQ_mn_bytes * 2
        return self._make_and_init_load_mma_pipeline(load_mma_KQ_aux_mbar_ptr, cluster_layout_vmnk, tx_count)

    def make_and_init_load_mma_VDO_pipeline(self, load_mma_VDO_mbar_ptr, cluster_layout_vmnk):
        """Load -> MMA pipeline for the V and dO operand stages."""
        tx_count = (self.tma_copy_dO_bytes + self.tma_copy_dOT_bytes) * 2
        tx_count += self.tma_copy_sfdO_bytes * self.k_halves * 2
        tx_count += self.tma_copy_sfdO_mn_bytes * 2
        tx_count += self.tma_copy_sum_OdO_bytes
        return self._make_and_init_load_mma_pipeline(load_mma_VDO_mbar_ptr, cluster_layout_vmnk, tx_count)

    def _make_and_init_mma_compute_pipeline(self, mbar_ptr, num_stages, cluster_layout_vmnk):
        """Create and initialise one MMA-producer / compute-consumer TMEM pipeline."""
        return cute_common.make_umma_async_pipeline(
            mbar_ptr,
            num_stages,
            cluster_layout_vmnk,
            len([self.mma_warp_id]),
            self.num_compute_warps * self.threads_per_warp * cute.size(cluster_layout_vmnk, mode=[0]),
        )

    def make_and_init_mma_compute_KQ_pipeline(self, mma_compute_KQ_mbar_ptr, cluster_layout_vmnk):
        """MMA -> compute pipeline handing off the S = K.Q^T accumulator."""
        return self._make_and_init_mma_compute_pipeline(mma_compute_KQ_mbar_ptr, self.mma_compute_KQ_stage, cluster_layout_vmnk)

    def make_and_init_mma_compute_VDO_pipeline(self, mma_compute_VDO_mbar_ptr, cluster_layout_vmnk):
        """MMA -> compute pipeline handing off the dP = V.dO^T accumulator."""
        return self._make_and_init_mma_compute_pipeline(mma_compute_VDO_mbar_ptr, self.mma_compute_VDO_stage, cluster_layout_vmnk)

    def make_and_init_mma_compute_dK_pipeline(self, mma_compute_dK_mbar_ptr, cluster_layout_vmnk):
        """MMA -> compute pipeline handing off the final dK/dV accumulators."""
        return cute_common.make_pipeline_umma_async(self, mma_compute_dK_mbar_ptr, self.mma_compute_dKdV_stage, cluster_layout_vmnk)

    def _make_and_init_compute_mma_pipeline(self, mbar_ptr, num_stages, cluster_layout_vmnk):
        """Create and initialise one compute-producer / MMA-consumer smem pipeline."""
        return cute_common.make_async_umma_pipeline(
            mbar_ptr,
            num_stages,
            cluster_layout_vmnk,
            self.num_compute_warps * self.threads_per_warp * cute.size(cluster_layout_vmnk, mode=[0]),
            len([self.mma_warp_id]),
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr, cluster_layout_vmnk):
        """Compute -> MMA pipeline publishing the quantised P tile."""
        return self._make_and_init_compute_mma_pipeline(compute_mma_P_mbar_ptr, self.compute_mma_P_stage, cluster_layout_vmnk)

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr, cluster_layout_vmnk):
        """Compute -> MMA pipeline publishing the quantised dS tile."""
        return self._make_and_init_compute_mma_pipeline(compute_mma_dS_mbar_ptr, self.compute_mma_dS_stage, cluster_layout_vmnk)
