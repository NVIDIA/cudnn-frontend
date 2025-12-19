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

import argparse
from typing import Tuple, Type, Optional
import math

import cuda.bindings.driver as cuda
import torch
from torch.profiler import profile, ProfilerActivity
import time
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float32, Int64

from cutlass._mlir.dialects import cute_nvgpu

"""
A NSA(Native Sparse Attention) Top-K Reduction Forward Pass for NVIDIA Blackwell SM100 architecture using Cute DSL.
"""


class FineGrainedReductionQK:
    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        k_value: int,
        selection_block_size: int,
        compress_block_sliding_stride: int,
        mma_tiler: Tuple[int, int, int],
        is_causal: bool = False,
    ):
        self.element_dtype = element_dtype
        self.acc_dtype = acc_dtype
        self.indices_dtype = Int32
        self.k_value = k_value
        self.selection_block_size = selection_block_size
        self.compress_block_sliding_stride = compress_block_sliding_stride
        self.num_elem_for_reduction = (
            selection_block_size // compress_block_sliding_stride
        )

        self.cluster_shape_mn = (1, 1)
        self.mma_tiler = mma_tiler
        self.is_causal = is_causal

        self.compute_warp_id = (0, 1, 2, 3)
        # self.reduce_warp_id = (4, 5, 6, 7)
        self.load_warp_id = 4
        self.mma_warp_id = 5
        self.epi_warp_id = 6
        self.threads_per_warp = 32
        self.num_compute_warps = len(self.compute_warp_id)
        # self.num_reduce_warps = len(self.reduce_warp_id)

        self.threads_per_cta = self.threads_per_warp * (self.num_compute_warps + 4)

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2
        # self.reduce_sync_bar_id = 3
        # self.num_regs_compute = 240
        # self.num_regs_reduce = 96
        # self.num_regs_other = 72
        self.num_regs_compute = 256
        # self.num_regs_reduce = 96
        self.num_regs_other = 96

    @cute.jit
    def __call__(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        Q: cute.Tensor,
        K: cute.Tensor,
        LSE: cute.Tensor,
        Topk_scores: cute.Tensor,
        Topk_indices: cute.Tensor,
        softmax_scale_log2_e: Float32,
        cumulative_s_q: Optional[cute.Tensor],
        cumulative_s_k: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        b, s_q_max, s_k_max, _, _, _ = problem_size
        s_q, s_k = Q.shape[2], K.shape[2]
        h_q, h_k = Q.shape[1], K.shape[1]
        head_dim = Q.shape[3]
        h_r = h_q // h_k

        stride_b_q = s_q * head_dim * h_k * h_r if cumulative_s_q is None else 0
        stride_b_k = s_k * head_dim * h_k if cumulative_s_k is None else 0
        stride_b_lse = s_q * h_r * h_k if cumulative_s_q is None else 0
        stride_b_out = (
            s_q * (s_k_max // self.num_elem_for_reduction) * h_k
            if cumulative_s_q is None
            else 0
        )
        stride_b_topk_scores = s_q * self.k_value * h_k if cumulative_s_q is None else 0
        stride_b_topk_indices = (
            s_q * self.k_value * h_k if cumulative_s_q is None else 0
        )

        Q = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (s_q, head_dim, (h_r, h_k, b)),
                stride=(
                    head_dim * h_r * h_k,
                    1,
                    (
                        head_dim * h_k,
                        head_dim,
                        stride_b_q,
                    ),  # TODO: head ordering is diff?
                ),
            ),
        )

        K = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (s_k, head_dim, (1, h_k, b)),
                stride=(
                    head_dim * h_k,
                    1,
                    (0, head_dim, stride_b_k),
                ),
            ),
        )

        # reshape LSE to (s_q, h_r, h_k, b)
        LSE = cute.make_tensor(
            LSE.iterator,
            cute.make_layout(
                (s_q, h_r, h_k, b),
                stride=(1, s_q * h_k, s_q, stride_b_lse),
            ),
        )

        Topk_scores = cute.make_tensor(
            Topk_scores.iterator,
            cute.make_layout(
                (s_q, self.k_value, (1, h_k, b)),
                stride=(self.k_value * h_k, 1, (0, self.k_value, stride_b_topk_scores)),
            ),
        )

        Topk_indices = cute.make_tensor(
            Topk_indices.iterator,
            cute.make_layout(
                (s_q, self.k_value, (1, h_k, b)),  # (s_q, k, h_k, b)
                stride=(
                    self.k_value * h_k,
                    1,
                    (0, self.k_value, stride_b_topk_indices),
                ),
            ),
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.load_mma_Q_stage = 2
        self.load_mma_K_stage = 1
        self.load_compute_LSE_stage = 1
        self.mma_compute_S_stage = 1
        cta_group = tcgen05.CtaGroup.ONE

        QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_tiler[:2],
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QK_tiled_mma,
            self.mma_tiler,
            self.element_dtype,
            self.load_mma_Q_stage,
        )
        K_smem_layout_staged = sm100_utils.make_smem_layout_b(
            QK_tiled_mma,
            self.mma_tiler,
            self.element_dtype,
            self.load_mma_K_stage,
        )
        LSE_smem_layout = cute.make_layout((self.mma_tiler[0], 1))

        self.epi_tile = (self.mma_tiler[0], self.k_value)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (QK_tiled_mma.thr_id.shape,),
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])

        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.mma_tiler,
            QK_tiled_mma,
            self.cluster_layout_vmnk,
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            K,
            K_smem_layout,
            self.mma_tiler,
            QK_tiled_mma,
            self.cluster_layout_vmnk,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_K_bytes = cute.size_in_bytes(self.element_dtype, K_smem_layout)

        @cute.struct
        class SharedStorage:
            load_mma_Q_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_Q_stage * 2
            ]
            load_mma_K_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_K_stage * 2
            ]
            load_compute_LSE_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_LSE_stage * 2
            ]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_S_stage * 2
            ]

            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(Q_smem_layout_staged)
                ],
                1024,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(K_smem_layout_staged)
                ],
                1024,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                1024,
            ]

            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage

        grid_shape = (cute.ceil_div(s_q_max, self.mma_tiler[0]), h_k, b)
        block_shape = (self.threads_per_cta, 1, 1)

        self.kernel(
            problem_size,
            QK_tiled_mma,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            LSE,
            cumulative_s_q,
            cumulative_s_k,
            Topk_scores,
            Topk_indices,
            Q_smem_layout_staged,
            K_smem_layout_staged,
            LSE_smem_layout,
            softmax_scale_log2_e,
        ).launch(
            grid=grid_shape,
            block=block_shape,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr):
        load_mma_Q_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
        )

    def make_and_init_load_mma_K_pipeline(self, load_mma_K_mbar_ptr):
        load_mma_K_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_K_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_K_mbar_ptr,
            num_stages=self.load_mma_K_stage,
            producer_group=load_mma_K_producer_group,
            consumer_group=load_mma_K_consumer_group,
            tx_count=self.tma_copy_K_bytes,
        )

    def make_and_init_load_compute_LSE_pipeline(self, load_compute_lse_mbar_ptr):
        load_compute_lse_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_lse_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_lse_mbar_ptr,
            num_stages=self.load_compute_LSE_stage,
            producer_group=load_compute_lse_producer_group,
            consumer_group=load_compute_lse_consumer_group,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
        )

    @cute.kernel
    def kernel(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        QK_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        LSE: cute.Tensor,
        cumulative_s_q: Optional[cute.Tensor],
        cumulative_s_k: Optional[cute.Tensor],
        Topk_scores: cute.Tensor,
        Topk_indices: cute.Tensor,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        softmax_scale_log2_e: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        b, s_q_max, s_k_max, h_q, h_k, head_dim = problem_size
        h_r = h_q // h_k

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(
            storage.load_mma_Q_mbar_ptr.data_ptr()
        )
        load_mma_K_pipeline = self.make_and_init_load_mma_K_pipeline(
            storage.load_mma_K_mbar_ptr.data_ptr()
        )
        load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(
            storage.load_compute_LSE_mbar_ptr.data_ptr()
        )
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr.data_ptr()
        )

        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
        )

        sQ = storage.sQ.get_tensor(
            Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner
        )
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)

        block_offset = (Int32(0), Int32(0), Int32(0), (Int32(0), Int32(0), Int32(0)))
        cur_s_q = s_q_max
        cur_s_k = s_k_max
        if cutlass.const_expr(cumulative_s_q is not None):
            cur_s_q = cumulative_s_q[bidz + 1] - cumulative_s_q[bidz]
            cur_s_k = cumulative_s_k[bidz + 1] - cumulative_s_k[bidz]
            block_offset = (
                cumulative_s_q[bidz],
                cumulative_s_k[bidz],
                Int32(0),
                (Int32(0), Int32(0), Int32(0)),
            )

        k_tile_count = cute.ceil_div(cur_s_k, self.mma_tiler[1])
        k_tile_idx = 0
        if cutlass.const_expr(self.is_causal):
            q_max = min(cur_s_q, (bidx + 1) * self.mma_tiler[0])
            k_cols = min(cur_s_k, q_max // self.compress_block_sliding_stride)
            k_tile_count = cute.ceil_div(k_cols, self.mma_tiler[1])

        mQ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), tma_tensor_Q)
        mK = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), tma_tensor_K)
        mTopk_scores = cute.make_tensor(
            Topk_scores.iterator
            + cute.assume(block_offset[0] * Topk_scores.stride[0], divby=self.k_value),
            Topk_scores.layout,
        )
        mTopk_indices = cute.make_tensor(
            Topk_indices.iterator
            + cute.assume(block_offset[0] * Topk_indices.stride[0], divby=self.k_value),
            Topk_indices.layout,
        )

        # (MMA_M, MMA_K, REST_M, REST_K, (H_r, H_k, B))
        gQ = cute.local_tile(
            mQ, cute.select(self.mma_tiler, mode=[0, 2]), (None, None, None)
        )
        # (MMA_N, MMA_K, REST_N, REST_K, (1, H_k, B))
        gK = cute.local_tile(
            mK, cute.select(self.mma_tiler, mode=[0, 2]), (None, None, None)
        )

        # (MMA_M, MMA_K, H_r)
        gQ = gQ[None, None, bidx, 0, (None, bidy, bidz)]
        # (MMA_N, MMA_K, REST_N)
        gK = gK[None, None, None, 0, (0, bidy, bidz)]

        thr_mma = QK_tiled_mma.get_slice(0)

        tSgQ = thr_mma.partition_A(gQ)
        tSgK = thr_mma.partition_B(gK)

        # tQgQ: (MMA, H_r)
        # tQsQ: (MMA, PIPE)
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )

        # tKgK: (MMA, REST_N)
        # tKsK: (MMA, PIPE)
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK, 0, 3),
        )

        tSrQ = QK_tiled_mma.make_fragment_A(sQ)
        tSrK = QK_tiled_mma.make_fragment_B(sK)
        tStS_shape = QK_tiled_mma.partition_shape_C(
            cute.select(self.mma_tiler, mode=[0, 1])
        )
        # ((MMA_M, MMA_N), REST_M, REST_N)
        tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
        # another tmem for reduction
        tStS_reduce = cute.make_tensor(tStS.iterator + self.mma_tiler[1], tStS.layout)

        if bidx * self.mma_tiler[0] < cur_s_q and k_tile_count > 0:

            load_iter_count = k_tile_count
            load_iter_index = k_tile_idx
            mma_iter_count = k_tile_count
            mma_iter_index = k_tile_idx
            compute_iter_count = k_tile_count
            compute_iter_index = k_tile_idx

            # LOAD Q K WARP
            if warp_idx == self.load_warp_id:
                # TODO: reconfig regs
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                load_mma_Q_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.load_mma_Q_stage
                )
                load_mma_K_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.load_mma_K_stage
                )
                load_compute_LSE_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage
                )

                while load_iter_count > 0:
                    # Wait for K to be empty
                    load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
                    K_tma_barrier = load_mma_K_pipeline.producer_get_barrier(
                        load_mma_K_producer_state
                    )

                    # Load K tile
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, load_iter_index],
                        tKsK[None, load_mma_K_producer_state.index],
                        tma_bar_ptr=K_tma_barrier,
                    )

                    load_mma_K_producer_state.advance()

                    # Load Q and LSE
                    for h_r_idx in cutlass.range(cute.size(tQgQ, mode=[1])):

                        load_compute_LSE_pipeline.producer_acquire(
                            load_compute_LSE_producer_state
                        )

                        # Load LSE
                        thread_idx = tidx % self.threads_per_warp

                        async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
                        atom_async_copy = cute.make_copy_atom(
                            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                            self.acc_dtype,
                            num_bits_per_copy=self.acc_dtype.width,
                        )

                        sLSE_for_copy = cute.flat_divide(sLSE, (1,))
                        LSE_for_copy = cute.flat_divide(LSE, (1,))
                        LSE_idx_offset = block_offset[0] * LSE.stride[0]
                        for i in cutlass.range_constexpr(async_copy_num_elts):
                            LSE_idx = (
                                self.mma_tiler[0] * bidx
                                + thread_idx * async_copy_num_elts
                            )
                            if cute.elem_less(LSE_idx + i, cur_s_q):
                                cute.copy(
                                    atom_async_copy,
                                    LSE_for_copy[
                                        None,
                                        LSE_idx_offset + LSE_idx + i,
                                        h_r_idx,
                                        bidy,
                                        bidz,
                                    ],
                                    sLSE_for_copy[
                                        None,
                                        thread_idx * async_copy_num_elts + i,
                                        load_compute_LSE_producer_state.index,
                                    ],
                                )
                            else:
                                sLSE_for_copy[
                                    None,
                                    thread_idx * async_copy_num_elts + i,
                                    load_compute_LSE_producer_state.index,
                                ].fill(0.0)

                        load_compute_LSE_pipeline.producer_commit(
                            load_compute_LSE_producer_state
                        )
                        load_compute_LSE_producer_state.advance()

                        # Wait for Q to be empty
                        load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
                        Q_tma_barrier = load_mma_Q_pipeline.producer_get_barrier(
                            load_mma_Q_producer_state
                        )

                        # Load Q tile
                        cute.copy(
                            tma_atom_Q,
                            tQgQ[None, h_r_idx],
                            tQsQ[None, load_mma_Q_producer_state.index],
                            tma_bar_ptr=Q_tma_barrier,
                        )

                        load_mma_Q_producer_state.advance()

                    load_iter_count -= 1
                    load_iter_index += 1

            # MMA WARP
            if warp_idx == self.mma_warp_id:
                # TODO: reconfig regs
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                num_tmem_cols = 512
                cute.arch.alloc_tmem(num_tmem_cols, storage.tmem_holding_buf)
                cute.arch.barrier(
                    barrier_id=self.tmem_alloc_sync_bar_id,
                    number_of_threads=self.threads_per_warp,
                )

                load_mma_Q_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_mma_Q_stage
                )
                load_mma_K_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_mma_K_stage
                )
                mma_compute_S_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.mma_compute_S_stage
                )

                while mma_iter_count > 0:
                    # Wait for K to be full
                    load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)

                    for h_r_idx in cutlass.range(cute.size(tQgQ, mode=[1])):
                        # Wait for Q to be full
                        load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
                        mma_compute_S_pipeline.producer_acquire(
                            mma_compute_S_producer_state
                        )

                        QK_tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, False)
                        for k_block_idx in cutlass.range_constexpr(
                            cute.size(tSrQ, mode=[2])
                        ):
                            cute.gemm(
                                QK_tiled_mma,
                                tStS,
                                tSrQ[
                                    None,
                                    None,
                                    k_block_idx,
                                    load_mma_Q_consumer_state.index,
                                ],
                                tSrK[
                                    None,
                                    None,
                                    k_block_idx,
                                    load_mma_K_consumer_state.index,
                                ],
                                tStS,
                            )
                            QK_tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)

                        mma_compute_S_pipeline.producer_commit(
                            mma_compute_S_producer_state
                        )
                        mma_compute_S_producer_state.advance()

                        load_mma_Q_pipeline.consumer_release(load_mma_Q_consumer_state)
                        load_mma_Q_consumer_state.advance()

                    mma_iter_count -= 1
                    mma_iter_index += 1

                    load_mma_K_pipeline.consumer_release(load_mma_K_consumer_state)
                    load_mma_K_consumer_state.advance()

            # COMPUTE WARP
            if warp_idx in self.compute_warp_id:
                cute.arch.warpgroup_reg_alloc(self.num_regs_compute)

                mma_compute_S_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage
                )
                load_compute_LSE_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage
                )

                load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)
                thread_idx = tidx % (self.threads_per_warp * self.num_compute_warps)

                heap_size_ref = cute.make_rmem_tensor((1,), Int32)
                heap_size_ref[0] = 0
                # # Create temporary register heaps for computation
                scores_heap_rf = cute.make_rmem_tensor(
                    ((4, self.k_value // 4), 1, 1), Float32
                )
                idx_heap_rf = cute.make_rmem_tensor(
                    ((4, self.k_value // 4), 1, 1), Int32
                )

                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                    self.acc_dtype,
                )
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.copy.Repetition(32)),
                    self.acc_dtype,
                )

                cS = cute.make_identity_tensor((self.mma_tiler[0], self.mma_tiler[1]))

                comp_tile_size = 32
                tStS_tiled = cute.logical_divide(
                    tStS, cute.make_layout((self.mma_tiler[0], comp_tile_size))
                )
                tStS_compute_tiled = cute.logical_divide(
                    tStS_reduce, cute.make_layout((self.mma_tiler[0], comp_tile_size))
                )
                cS_tiled = cute.logical_divide(
                    cS, cute.make_layout((self.mma_tiler[0], comp_tile_size))
                )

                tStS_slice = tStS_tiled[None, 0]  # ((128, 16), 8)
                tStS_compute_slice = tStS_compute_tiled[None, 0]

                # (MMA_M, MMA_N)
                tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
                thr_t2r = tiled_t2r.get_slice(thread_idx)

                tTR_cS = thr_t2r.partition_D(cS_tiled)
                tTR_tS = thr_t2r.partition_S(tStS_tiled)
                tTR_tS_compute = thr_t2r.partition_S(tStS_compute_tiled)
                tTR_rS = cute.make_rmem_tensor(
                    tTR_cS[None, None, 0].shape, self.acc_dtype
                )
                tTR_rS_compute = cute.make_rmem_tensor(
                    tTR_cS[None, None, 0].shape, self.acc_dtype
                )

                tiled_r2t = tcgen05.make_tmem_copy(tmem_store_atom, tStS_compute_slice)
                thr_r2t = tiled_r2t.get_slice(thread_idx)
                tRT_tS_compute = thr_r2t.partition_D(tStS_compute_tiled)

                tiled_t2r_reduce = tcgen05.make_tmem_copy(
                    tmem_load_atom, tStS[(None, None), 0, 0]
                )
                thr_t2r_reduce = tiled_t2r_reduce.get_slice(thread_idx)
                tTR_tS_reduce = thr_t2r_reduce.partition_S(
                    tStS_reduce[(None, None), 0, 0]
                )
                tTR_cS_reduce = thr_t2r_reduce.partition_D(cS)
                tTR_rS_reduce = cute.make_rmem_tensor(
                    tTR_cS_reduce.shape, self.acc_dtype
                )

                tmp = cute.make_rmem_tensor(
                    (self.mma_tiler[1] // self.num_elem_for_reduction), self.acc_dtype
                )

                while compute_iter_count > 0:

                    for h_r_idx in range(cute.size(tQgQ, mode=[1])):
                        mma_compute_S_pipeline.consumer_wait(
                            mma_compute_S_consumer_state
                        )

                        # TODO: Added this as we should wait for the producer to load
                        load_compute_LSE_pipeline.consumer_wait(
                            load_compute_LSE_consumer_state
                        )

                        for sub_tile in cutlass.range(
                            self.mma_tiler[1] // comp_tile_size
                        ):
                            tTR_tS_sub_tile = tTR_tS[None, None, sub_tile]
                            tTR_tS_compute_sub_tile = tTR_tS_compute[
                                None, None, sub_tile
                            ]
                            tRT_tS_compute_sub_tile = tRT_tS_compute[
                                None, None, sub_tile
                            ]
                            tTR_cS_sub_tile = tTR_cS[None, None, sub_tile]

                            # Copy S from tmem to rmem
                            cute.copy(tiled_t2r, tTR_tS_sub_tile, tTR_rS)

                            is_residual_k = (
                                compute_iter_index * self.mma_tiler[1]
                                + self.mma_tiler[1]
                                > cur_s_k
                            )

                            leading_causal_masking = cutlass.Boolean(False)
                            if cutlass.const_expr(self.is_causal):
                                leading_causal_masking = (
                                    ((compute_iter_index + 1) * self.mma_tiler[1] + 1)
                                    * self.compress_block_sliding_stride
                                    - 1
                                    > bidx * self.mma_tiler[0]
                                )
                                leading_causal_masking = cute.arch.shuffle_sync(
                                    leading_causal_masking, 0
                                )
                            trailing_residual_masking = cutlass.Boolean(False)
                            trailing_residual_masking = is_residual_k
                            trailing_residual_masking = cute.arch.shuffle_sync(
                                trailing_residual_masking, 0
                            )

                            is_masked_tile = (
                                leading_causal_masking or trailing_residual_masking
                            )

                            # Apply mask
                            if is_masked_tile:
                                for i in cutlass.range(
                                    cute.size(tTR_rS), unroll_full=True
                                ):
                                    q_idx = (
                                        cute.get(tTR_cS_sub_tile[i], mode=[0])
                                        + bidx * self.mma_tiler[0]
                                    )
                                    k_block_idx = (
                                        cute.get(tTR_cS_sub_tile[i], mode=[1])
                                        + compute_iter_index * self.mma_tiler[1]
                                    )

                                    if is_masked_tile:
                                        if cutlass.const_expr(self.is_causal):
                                            k_idx = (
                                                (k_block_idx + 1)
                                                * self.compress_block_sliding_stride
                                                - 1
                                            )
                                            if k_idx > q_idx:
                                                tTR_rS[i] = -cutlass.Float32.inf
                                            if q_idx > cur_s_q or k_block_idx > cur_s_k:
                                                tTR_rS[i] = -cutlass.Float32.inf
                                        else:
                                            if q_idx > cur_s_q or k_block_idx > cur_s_k:
                                                tTR_rS[i] = -cutlass.Float32.inf

                            # P = exp2(S * softmax_scale_log2_e - LSE),
                            # LSE should be set negative before and has be already multiplied by log2_e

                            # Copy S_reduce from tmem to rmem
                            cute.copy(
                                tiled_t2r, tTR_tS_compute_sub_tile, tTR_rS_compute
                            )

                            for i in cutlass.range(0, cute.size(tTR_rS, mode=[0]), 2):
                                lse = (
                                    sLSE[
                                        cute.get(tTR_cS_sub_tile[i], mode=[0]),
                                        load_compute_LSE_consumer_state.index,
                                    ],
                                    sLSE[
                                        cute.get(tTR_cS_sub_tile[i + 1], mode=[0]),
                                        load_compute_LSE_consumer_state.index,
                                    ],
                                )

                                tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2(
                                    (tTR_rS[i], tTR_rS[i + 1]),
                                    (
                                        softmax_scale_log2_e,
                                        softmax_scale_log2_e,
                                    ),
                                    lse,
                                )
                                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                                tTR_rS[i + 1] = cute.math.exp2(
                                    tTR_rS[i + 1], fastmath=True
                                )

                                if h_r_idx == 0:
                                    (tTR_rS_compute[i], tTR_rS_compute[i + 1]) = (
                                        cute.arch.add_packed_f32x2(
                                            (0.0, 0.0),
                                            (tTR_rS[i], tTR_rS[i + 1]),
                                        )
                                    )
                                else:
                                    (tTR_rS_compute[i], tTR_rS_compute[i + 1]) = (
                                        cute.arch.add_packed_f32x2(
                                            (tTR_rS_compute[i], tTR_rS_compute[i + 1]),
                                            (tTR_rS[i], tTR_rS[i + 1]),
                                        )
                                    )

                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.barrier(
                                barrier_id=self.compute_sync_bar_id,
                                number_of_threads=self.num_compute_warps
                                * self.threads_per_warp,
                            )

                            # Copy tS_reduce back to tmem
                            cute.copy(
                                tiled_r2t, tTR_rS_compute, tRT_tS_compute_sub_tile
                            )

                            cute.arch.fence_view_async_tmem_store()

                        load_compute_LSE_pipeline.consumer_release(
                            load_compute_LSE_consumer_state
                        )
                        load_compute_LSE_consumer_state.advance()

                        mma_compute_S_pipeline.consumer_release(
                            mma_compute_S_consumer_state
                        )
                        mma_compute_S_consumer_state.advance()

                    # Reduce
                    cute.copy(tiled_t2r_reduce, tTR_tS_reduce, tTR_rS_reduce)

                    # Contraint: self.mma_tiler[1] % self.num_elem_for_reduction == 0
                    tTR_rS_reduce_reshape = cute.composition(
                        tTR_rS_reduce,
                        cute.make_layout(
                            (
                                self.mma_tiler[1] // self.num_elem_for_reduction,
                                self.num_elem_for_reduction,
                            ),
                            stride=(self.num_elem_for_reduction, 1),
                        ),
                    )
                    tTR_rS_reduce_vec = tTR_rS_reduce_reshape.load()
                    tTR_rS_reduce_sum_vec = tTR_rS_reduce_vec.reduce(
                        cute.ReductionOp.ADD,
                        self.acc_dtype(0.0),
                        reduction_profile=(None, 1),
                    )

                    tmp.store(tTR_rS_reduce_sum_vec)

                    q_row_idx = bidx * self.mma_tiler[0] + thread_idx
                    # TODO: make sure the self.mma_tile[1] // self.num_elem_for_reduction is always larger than self.k_value
                    if compute_iter_index == k_tile_idx:
                        self.topk_step(
                            tmp,
                            scores_heap_rf,
                            idx_heap_rf,
                            q_row_idx,
                            compute_iter_index,
                            self.k_value,
                        )
                    else:
                        self.topk_step(
                            tmp,
                            scores_heap_rf,
                            idx_heap_rf,
                            q_row_idx,
                            compute_iter_index,
                            0,
                        )

                    compute_iter_count -= 1
                    compute_iter_index += 1

                # (s_q, k_value, (1, h_k, b))
                gTopk_scores = cute.flat_divide(
                    mTopk_scores, (self.epi_tile[0], self.k_value)
                )
                gTopk_indices = cute.flat_divide(
                    mTopk_indices, (self.epi_tile[0], self.k_value)
                )
                gTopk_scores = gTopk_scores[None, None, bidx, 0, (0, bidy, bidz)]
                gTopk_indices = gTopk_indices[None, None, bidx, 0, (0, bidy, bidz)]
                cTopk = cute.make_identity_tensor((self.epi_tile[0], self.k_value))
                cTopk = cute.domain_offset((bidx * self.epi_tile[0], 0), cTopk)

                copy_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.acc_dtype,
                    num_bits_per_copy=128,
                )

                thr_layout = cute.make_layout(
                    (self.num_compute_warps * self.threads_per_warp, 1),
                    stride=(1, 1),
                )
                val_layout = cute.make_layout(
                    (1, self.k_value),
                    stride=(self.k_value, 1),
                )
                copy_op = cute.make_tiled_copy_tv(
                    copy_atom,
                    thr_layout,
                    val_layout,
                )

                thr_copy = copy_op.get_slice(thread_idx)

                tgTopk_scores = thr_copy.partition_D(gTopk_scores)
                tgTopk_indices = thr_copy.partition_D(gTopk_indices)
                tcTopk = thr_copy.partition_D(cTopk)

                pred_shape = (tcTopk.shape[0][1], tcTopk.shape[1], tcTopk.shape[2])
                preds = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)
                for v in cutlass.range_constexpr(preds.shape[0]):
                    for m in cutlass.range_constexpr(preds.shape[1]):
                        for n in cutlass.range_constexpr(preds.shape[2]):
                            lhs = tcTopk[v, m, n]
                            val = cute.elem_less(lhs, (cur_s_q, self.k_value))
                            preds[v, m, n] = val

                cute.copy(copy_atom, scores_heap_rf, tgTopk_scores, pred=preds)
                cute.copy(copy_atom, idx_heap_rf, tgTopk_indices, pred=preds)

                cute.arch.barrier(
                    barrier_id=self.compute_sync_bar_id,
                    number_of_threads=self.num_compute_warps * self.threads_per_warp,
                )

                if warp_idx % self.num_compute_warps == 0:
                    tmem_ptr = cute.arch.retrieve_tmem_ptr(
                        self.acc_dtype,
                        alignment=16,
                        ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
                    )
                    cute.arch.dealloc_tmem(tmem_ptr, 512)

    @cute.jit
    def topk_step(
        self,
        tiled_scores: cute.Tensor,  # (reduced_tile) probabilities
        scores_heap_rf: cute.Tensor,
        idx_heap_rf: cute.Tensor,
        query_index: Int32,
        k_tile_idx: Int32,
        heap_size: int,
    ):

        q_selection_block_idx = (query_index + 1) // self.selection_block_size
        blocks_per_tile = self.mma_tiler[1] // self.num_elem_for_reduction

        score = 0.0
        gmem_idx = 0
        is_valid = cutlass.Boolean(False)

        for i in cutlass.range_constexpr(heap_size):
            score = tiled_scores[i]
            gmem_idx = i + k_tile_idx * blocks_per_tile

            is_valid = (gmem_idx > 0) and (gmem_idx < q_selection_block_idx - 2)

            val_score = score if is_valid else -cutlass.Float32.inf
            val_index = gmem_idx if is_valid else -1
            scores_heap_rf[i] = val_score
            idx_heap_rf[i] = val_index

        for i in cutlass.range(heap_size, cute.size(tiled_scores)):
            score = tiled_scores[i]
            gmem_idx = i + k_tile_idx * blocks_per_tile

            # Decide if candidate is valid for consideration
            is_valid = (gmem_idx > 0) and (gmem_idx < q_selection_block_idx - 2)

            # Full: branchless min-scan to find current minimum (with tie-break)
            min_score = scores_heap_rf[0]
            min_index = idx_heap_rf[0]
            for j in cutlass.range(1, self.k_value, unroll_full=True):
                s = scores_heap_rf[j]
                t = idx_heap_rf[j]
                is_smaller_score = s < min_score
                is_tie_break = (s == min_score) and (t > min_index)
                if is_smaller_score or is_tie_break:
                    # swap to make sure the min_slot is always 0
                    scores_heap_rf[0], scores_heap_rf[j] = s, min_score
                    idx_heap_rf[0], idx_heap_rf[j] = t, min_index
                    min_score = s
                    min_index = t

            # Decide replacement under tie-break rules
            if is_valid:
                is_larger_score = score > min_score
                is_tie_better = (score == min_score) and (gmem_idx < min_index)
                if is_larger_score or is_tie_better:
                    scores_heap_rf[0] = score
                    idx_heap_rf[0] = gmem_idx
