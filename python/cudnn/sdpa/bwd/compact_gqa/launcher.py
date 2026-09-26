# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compact GQA preprocessing and producer launch."""

import math
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from .producer import BlackwellFusedMultiHeadAttentionBackwardDKDVKernel
from cudnn.flex_attention.kernels.sm100.bwd.backward_hd256 import _as_shhb_tensor
from cudnn.flex_attention.runtime.dsl_utils import as_bshkrd_tensor, assume_tensor_aligned
from cudnn.flex_attention.plan.kernels import BlockSparseTensors


class FusedPipelined:
    def __init__(self, groups=1, coarse_bin=4096, max_len=32768, fast_store=True):
        self.groups = groups
        assert groups in (1, 2, 4, 8)
        self.core = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(8 // groups, coarse_bin, max_len)
        self.core.fast_ds_store = fast_store

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        do: cute.Tensor,
        lse: cute.Tensor,
        lse2: cute.Tensor,
        delta: cute.Tensor,
        dqa: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        cu: cute.Tensor,
        ds_global: cute.Tensor,
        sparse: BlockSparseTensors | None,
        max_len: cutlass.Int32,
        scale: cutlass.Float32,
        query_start: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        assert q.shape[1] == 8 and k.shape[1] == 1 and q.shape[2] == 256
        rows = cute.ceil_div(o.shape[0], 128) * 128
        if query_start == 0:
            self.pre(o, do, lse, lse2, delta, dqa, cu).launch(grid=(cute.ceil_div(rows, 4), 1, 1), block=(256, 1, 1), stream=stream)
        k_view = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], self.groups, k.shape[2]), stride=(k.stride[0], 0, k.stride[2])))
        v_view = cute.make_tensor(v.iterator, cute.make_layout((v.shape[0], self.groups, v.shape[2]), stride=(v.stride[0], 0, v.stride[2])))
        nq = as_bshkrd_tensor(assume_tensor_aligned(q), self.groups, 8 // self.groups, True)
        nk = as_bshkrd_tensor(assume_tensor_aligned(k_view), self.groups, 1, True)
        nv = as_bshkrd_tensor(assume_tensor_aligned(v_view), self.groups, 1, True)
        ndk = as_bshkrd_tensor(assume_tensor_aligned(dk), self.groups, 1, True)
        ndv = as_bshkrd_tensor(assume_tensor_aligned(dv), self.groups, 1, True)
        ndo = as_bshkrd_tensor(assume_tensor_aligned(do), self.groups, 8 // self.groups, True)
        nlse = _as_shhb_tensor(lse2, self.groups, 8 // self.groups, cu.shape[0] - 1, True)
        ndelta = _as_shhb_tensor(delta, self.groups, 8 // self.groups, cu.shape[0] - 1, True)
        self.core(nq, nk, nv, ndk, ndv, ndo, nlse, ndelta, cu, cu, scale, sparse, max_len, stream, dqa, ds_global, query_start)

    @cute.kernel
    def pre(self, o: cute.Tensor, do: cute.Tensor, lse: cute.Tensor, lse2: cute.Tensor, delta: cute.Tensor, dqa: cute.Tensor, cu: cute.Tensor):
        bx, _, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        lane = tx % 32
        head = tx // 32
        end = o.shape[0]
        if bx == 0 and tx == 0:
            cu[0] = 0
            cu[1] = cutlass.Int32(end)
        for local_token in cutlass.range_constexpr(4):
            token = bx * 4 + local_token
            if token < cute.ceil_div(end, 128) * 128:
                accum = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(8):
                    d = lane + i * 32
                    if token < end:
                        accum += cutlass.Float32(o[token, head, d]) * cutlass.Float32(do[token, head, d])
                for i in cutlass.range_constexpr(5):
                    accum += cute.arch.shuffle_sync_bfly(accum, offset=1 << i)
                if lane == 0:
                    if token < end:
                        delta[head, token] = accum
                        lse2[head, token] = lse[token, head] * math.log2(math.e)
                    else:
                        delta[head, token] = cutlass.Float32(0.0)
                        lse2[head, token] = cutlass.Float32(0.0)


FusedPipelined.pre.set_name_prefix("cudnn_gqa_fused_pre", remove_cutlass_symbol=True)


from .packed import PackedPre


class FusedPacked:
    def __init__(self, groups=1, coarse_bin=4096, max_len=32768, fast_store=True):
        self.groups = groups
        assert groups in (1, 2, 4, 8)
        self.core = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(8 // groups, coarse_bin, max_len)
        self.core.fast_ds_store = fast_store
        self.core.packed_lengths = True
        self.pre = PackedPre()

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        do: cute.Tensor,
        lse: cute.Tensor,
        lse2: cute.Tensor,
        delta: cute.Tensor,
        dqa: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        cu: cute.Tensor,
        lengths: cute.Tensor,
        ds_global: cute.Tensor,
        kr: cute.Tensor,
        tables: cute.Tensor,
        sparse: BlockSparseTensors | None,
        max_len: cutlass.Int32,
        scale: cutlass.Float32,
        query_start: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        assert q.shape[1] == 8 and k.shape[1] == 1 and q.shape[2] == 256
        self.pre(k, o, do, lse, lse2, delta, cu, lengths, ds_global, kr, tables, dq, stream)
        k_view = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], self.groups, k.shape[2]), stride=(k.stride[0], 0, k.stride[2])))
        v_view = cute.make_tensor(v.iterator, cute.make_layout((v.shape[0], self.groups, v.shape[2]), stride=(v.stride[0], 0, v.stride[2])))
        nq = as_bshkrd_tensor(assume_tensor_aligned(q), self.groups, 8 // self.groups, True)
        nk = as_bshkrd_tensor(assume_tensor_aligned(k_view), self.groups, 1, True)
        nv = as_bshkrd_tensor(assume_tensor_aligned(v_view), self.groups, 1, True)
        ndk = as_bshkrd_tensor(assume_tensor_aligned(dk), self.groups, 1, True)
        ndv = as_bshkrd_tensor(assume_tensor_aligned(dv), self.groups, 1, True)
        ndo = as_bshkrd_tensor(assume_tensor_aligned(do), self.groups, 8 // self.groups, True)
        nlse = _as_shhb_tensor(lse2, self.groups, 8 // self.groups, cu.shape[0] - 1, True)
        ndelta = _as_shhb_tensor(delta, self.groups, 8 // self.groups, cu.shape[0] - 1, True)
        self.core(nq, nk, nv, ndk, ndv, ndo, nlse, ndelta, cu, lengths, scale, sparse, max_len, stream, dqa, ds_global, query_start)
