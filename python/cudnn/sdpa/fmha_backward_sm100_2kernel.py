# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

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

"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* mma_tiler_mn must be 64,64
* Batch size must be the same for Q, K, and V tensors
"""

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32
from .utils import warp_reduction_sum
from .fmha_dq_d256_sm100 import (
    BlackwellFusedAttentionDQKernel,
)
from .fmha_dkdv_d256_sm100 import (
    BlackwellFusedAttentionDKDVKernel,
)
from .fmha_utils import MaskEnum

SM100_TMEM_CAPACITY_COLUMNS = 512
LAYOUT_RANK_CONSTANT = 3


@cute.jit
def split_wg(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """Split warp group."""
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == LAYOUT_RANK_CONSTANT):
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


def Tmemory_offset(lane, col):
    """Tensor memory offset."""
    return (lane << 16) + col


permute_order = (0, 1, 2, 3)


class BlackwellFusedMultiHeadAttentionBackward:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        element_dtype: type[cutlass.Numeric],
        acc_dtype: type[cutlass.Numeric],
        mma_tiler: tuple[int, int, int],
        dkdv_mma_tiler: tuple[int, int, int],
        varlen: bool,
        is_causal: bool,
        mask_type: MaskEnum,
        window_size_left: int | None,
        window_size_right: int | None,
        split_head: bool = False,
        use_clc_dynamic_scheduler: bool = False,
    ):
        """Initialization."""
        self.element_dtype = element_dtype
        self.acc_dtype = acc_dtype
        self.varlen = varlen
        self.is_causal = is_causal
        self.mask_type = mask_type
        self.window_size_left = None if window_size_left < 0 else window_size_left
        self.window_size_right = None if window_size_right < 0 else window_size_right

        # =================== Sum OdO ================================
        self.sum_OdO_max_threads_per_block = 128
        self.sum_OdO_block_q = 16
        self.sum_OdO_num_threads_d = 8
        self.sum_OdO_num_threads_q = self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
        self.sum_OdO_elem_per_load = 2

        # Keep the original (known-good) mask selection for dQ kernel.
        self.dq_kernel = BlackwellFusedAttentionDQKernel(
            element_dtype,
            acc_dtype,
            mma_tiler,
            varlen,
            is_causal,
            self.mask_type,
            window_size_left,
            window_size_right,
            False,
            split_head,
            use_clc_dynamic_scheduler=use_clc_dynamic_scheduler,
        )

        dkdv_cta_mma_tiler = (dkdv_mma_tiler[0], dkdv_mma_tiler[1], 256)

        self.dkdv_kernel = BlackwellFusedAttentionDKDVKernel(
            element_dtype,
            acc_dtype,
            dkdv_cta_mma_tiler,
            varlen,
            self.is_causal,
            self.mask_type,
            window_size_left,
            window_size_right,
            use_clc_dynamic_scheduler=use_clc_dynamic_scheduler,
        )

    @cute.jit
    def __call__(
        self,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        LSE: cute.Tensor,
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        scale_softmax: cutlass.Float32,
        workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Host function to launch CuTeDSL kernel."""
        _, _, _, hb = problem_shape
        h, _ = hb
        h_r, h_k = h
        # (b, s, h_k * h_r, d) -> (s, d, ((h_r, h_k), b))
        mQ = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[3], hb),
                stride=(
                    Q.stride[1],
                    Q.stride[3],
                    (
                        (Q.shape[3], Q.shape[3] * h_r),
                        (0 if self.varlen else cute.assume(Q.shape[1] * Q.shape[3] * h_r * h_k, divby=64)),
                    ),
                ),
            ),
        )
        # (b, s, h_k * 1, d) -> (s, d, ((1, h_k), b))
        mK = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[1], K.shape[3], hb),
                stride=(
                    K.stride[1],
                    K.stride[3],
                    (
                        (0, K.shape[3]),
                        (0 if self.varlen else cute.assume(K.shape[1] * K.shape[3] * h_k, divby=64)),
                    ),
                ),
            ),
        )
        # (b, s, h_k * 1, d) -> (s, d, ((1, h_k), b))
        mV = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[1], V.shape[3], hb),
                stride=(
                    V.stride[1],
                    V.stride[3],
                    (
                        (0, V.shape[3]),
                        (0 if self.varlen else cute.assume(V.shape[1] * V.shape[3] * h_k, divby=64)),
                    ),
                ),
            ),
        )
        mO = cute.make_tensor(O.iterator, mQ.layout)

        _mdQ = cute.make_tensor(dQ.iterator, mQ.layout)
        _mdK = cute.make_tensor(dK.iterator, mK.layout)
        _mdV = cute.make_tensor(dV.iterator, mV.layout)
        mdO = cute.make_tensor(dO.iterator, mO.layout)

        # (b, h_k * h_r, s) -> (s, ((h_r, h_k), b))
        LSE = cute.make_tensor(
            LSE.iterator,
            cute.make_layout(
                (LSE.shape[2], hb),
                stride=(
                    LSE.stride[2],
                    (
                        (LSE.shape[2], LSE.shape[2] * h_r),
                        (0 if LSE.shape[0] == 1 else LSE.shape[1] * LSE.shape[2]),
                    ),
                ),
            ),
        )

        # =============================== Sum OdO ===============================
        sum_OdO_scale = cutlass.Float32(-1.0)
        LSE_scale = cutlass.Float32(-math.log2(math.e))
        sum_OdO, scaled_LSE, _dQ_acc = self.get_workspace_tensor(problem_shape, workspace, self.acc_dtype, self.varlen)
        sum_OdO_grid = self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q)

        self.sum_OdO(
            mO,
            mdO,
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

        # Keep original order: dQ first, then dKdV.
        self.dq_kernel(
            problem_shape,
            Q,
            K,
            V,
            O,
            dQ,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            workspace,
            stream,
        )
        self.dkdv_kernel(
            problem_shape,
            Q,
            K,
            V,
            O,
            dK,
            dV,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            workspace,
            stream,
        )

    @cute.kernel
    def sum_OdO(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        scaled_lse: cute.Tensor,
        cumulative_s_q: cute.Tensor | None,
        sum_OdO_scale: cutlass.Float32,
        lse_scale: cutlass.Float32,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
    ):
        """CuTeDSL kernel for sum(dot(O, dO))."""
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, tidy, _ = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]
        offset = 0
        if cutlass.const_expr(self.varlen):
            assert isinstance(cumulative_s_q, cute.Tensor)
            offset = cumulative_s_q[bidz]
            seqlen_q = cumulative_s_q[bidz + 1] - offset

        for idx_q_t in cutlass.range(tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True):
            idx_q = idx_q_t + self.sum_OdO_block_q * bidx
            if idx_q < seqlen_q:
                O_bhq = O[idx_q + offset, None, (bidy, bidz)]
                O_bhq = cute.logical_divide(O_bhq, cute.make_layout(self.sum_OdO_elem_per_load))
                dO_bhq = dO[idx_q + offset, None, (bidy, bidz)]
                dO_bhq = cute.logical_divide(dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load))

                idx_d_start = tidx
                idx_d_step = self.sum_OdO_num_threads_d
                acc = 0.0
                for idx_d in cutlass.range(idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step):
                    O_frag = O_bhq[None, idx_d].load().to(self.acc_dtype)
                    dO_frag = dO_bhq[None, idx_d].load().to(self.acc_dtype)
                    prod_frag = O_frag * dO_frag
                    acc += prod_frag.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

                acc = warp_reduction_sum(acc, threads_in_group=self.sum_OdO_num_threads_d)

                if tidx == 0:
                    lse_bhq = lse[idx_q + offset, (bidy, bidz)]
                    sum_OdO[idx_q + offset, (bidy, bidz)] = sum_OdO_scale * acc
                    scaled_lse[idx_q + offset, (bidy, bidz)] = lse_scale * lse_bhq

    @staticmethod
    def get_workspace_size(s_q: int, d: int, h: int, b: int, acc_dtype: type[cutlass.Numeric]):
        """Get workspace size."""
        d = (d + 7) // 8 * 8  # round up to 8
        s_q = (s_q + 7) // 8 * 8  # round up to 8
        workspace_bytes = 0
        # OdO vector
        workspace_bytes += acc_dtype.width // 8
        # scaled LSE vector
        workspace_bytes += acc_dtype.width // 8
        # FP32 versions of outputs that are churned (start off with Q only)
        workspace_bytes += d * acc_dtype.width // 8
        return (b, s_q, h, workspace_bytes)

    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        block_q: int,
    ) -> tuple[Int32, Int32, Int32]:
        """Compute grid shape for sum_OdO kernel."""
        return (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )

    def get_workspace_tensor(
        self,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        workspace: cute.Tensor,
        acc_dtype: type[cutlass.Numeric],
        varlen: bool,
    ) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        """Get workspace tensor."""
        D = problem_shape[2]
        H, B = cute.size(problem_shape[3][0]), problem_shape[3][1]
        H_r, H_k = problem_shape[3][0]
        D = cute.round_up(D, 8)

        # b = 1 for varlen, else batch_size
        b = workspace.shape[0]
        # s_q_sum for varlen, else s_q_max, already rounded to 8
        S_Q = workspace.shape[1]

        acc_bytes = acc_dtype.width // 8
        sum_OdO_bytes = cute.assume(b * H * S_Q * acc_bytes, divby=acc_bytes)
        scaled_lse_bytes = cute.assume(b * H * S_Q * acc_bytes, divby=acc_bytes)

        sum_OdO_iter = workspace.iterator
        scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes
        dQ_acc_iter = scaled_lse_iter + scaled_lse_bytes

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)
        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype)

        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout(
                (S_Q, ((H_r, H_k), B)),
                stride=(1, ((S_Q, S_Q * H_r), 0 if varlen else S_Q * H)),
            ),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout(
                (S_Q, ((H_r, H_k), B)),
                stride=(1, ((S_Q, S_Q * H_r), 0 if varlen else S_Q * H)),
            ),
        )
        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (S_Q, D, ((H_r, H_k), B)),
                stride=(D, 1, ((D * S_Q, D * S_Q * H_r), 0 if varlen else D * S_Q * H)),
            ),
        )

        return sum_OdO, scaled_lse, dQ_acc
