# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Packed preprocessing and group reduction."""

import math
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda


class PackedReduce:
    def __init__(self, groups):
        self.groups = groups

    @cute.jit
    def __call__(
        self,
        dkp: cute.Tensor,
        dvp: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        cu: cute.Tensor,
        lengths: cute.Tensor,
        max_span: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.run(dkp, dvp, dq, dk, dv, cu, lengths).launch(grid=(cute.ceil_div(max_span, 8), lengths.shape[0], 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def run(self, dkp: cute.Tensor, dvp: cute.Tensor, dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor, cu: cute.Tensor, lengths: cute.Tensor):
        bx, batch, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        local = bx * 8 + tx // 32
        dim = tx % 32 * 8
        begin, end = cu[batch], cu[batch + 1]
        if local < end - begin:
            token = begin + local
            sk = cute.make_rmem_tensor(8, cutlass.Float32)
            sv = cute.make_rmem_tensor(8, cutlass.Float32)
            sk.fill(0.0)
            sv.fill(0.0)
            rk = cute.make_rmem_tensor(8, cutlass.Float32)
            rv = cute.make_rmem_tensor(8, cutlass.Float32)
            if local < lengths[batch]:
                for group in cutlass.range_constexpr(self.groups):
                    offset = cute.assume((cutlass.Int64(token) * self.groups + group) * 256 + dim, divby=8)
                    cute.autovec_copy(cute.make_tensor(dkp.iterator + offset, cute.make_layout(8)), rk)
                    cute.autovec_copy(cute.make_tensor(dvp.iterator + offset, cute.make_layout(8)), rv)
                    sk.store(sk.load() + rk.load().to(cutlass.BFloat16).to(cutlass.Float32))
                    sv.store(sv.load() + rv.load().to(cutlass.BFloat16).to(cutlass.Float32))
            outk = cute.make_rmem_tensor(8, cutlass.BFloat16)
            outv = cute.make_rmem_tensor(8, cutlass.BFloat16)
            outk.store(sk.load().to(cutlass.BFloat16))
            outv.store(sv.load().to(cutlass.BFloat16))
            offset = cute.assume(cutlass.Int64(token) * 256 + dim, divby=8)
            cute.autovec_copy(outk, cute.make_tensor(dk.iterator + offset, cute.make_layout(8)))
            cute.autovec_copy(outv, cute.make_tensor(dv.iterator + offset, cute.make_layout(8)))
            if local >= lengths[batch]:
                for head in cutlass.range_constexpr(8):
                    offset = cute.assume((cutlass.Int64(token) * 8 + head) * 256 + dim, divby=8)
                    cute.autovec_copy(outk, cute.make_tensor(dq.iterator + offset, cute.make_layout(8)))


class PackedPre:
    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        o: cute.Tensor,
        do: cute.Tensor,
        lse: cute.Tensor,
        lse2: cute.Tensor,
        delta: cute.Tensor,
        cu: cute.Tensor,
        lengths: cute.Tensor,
        ds: cute.Tensor,
        kr: cute.Tensor,
        tables: cute.Tensor,
        dq: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.run(k, o, do, lse, lse2, delta, cu, lengths, ds, kr, tables, dq).launch(
            grid=(cute.ceil_div(ds.shape[2], 4), lengths.shape[0], 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def run(
        self,
        k: cute.Tensor,
        o: cute.Tensor,
        do: cute.Tensor,
        lse: cute.Tensor,
        lse2: cute.Tensor,
        delta: cute.Tensor,
        cu: cute.Tensor,
        lengths: cute.Tensor,
        ds: cute.Tensor,
        kr: cute.Tensor,
        tables: cute.Tensor,
        dq: cute.Tensor,
    ):
        bx, batch, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        lane, head = tx % 32, tx // 32
        start, length = cu[batch], lengths[batch]
        stats_start = (start + batch * 128) // 128 * 128
        if bx == 0 and tx < 8:
            tables[batch, 0, tx] = kr.iterator.toint() + cutlass.Int64(batch) * kr.shape[1] * 256 * 2
            tables[batch, 1, tx] = ds.iterator.toint() + (cutlass.Int64(batch) * 8 + tx) * ds.shape[2] * ds.shape[3] * 2
            tables[batch, 2, tx] = dq.iterator.toint() + (cutlass.Int64(start) * 2048 + tx * 256) * 2
        for local_token in cutlass.range_constexpr(4):
            local = bx * 4 + local_token
            token = start + local
            accum = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(8):
                dim = lane + i * 32
                if local < length:
                    accum += cutlass.Float32(o[token, head, dim]) * cutlass.Float32(do[token, head, dim])
                if head == 0 and local < kr.shape[1]:
                    kval = cutlass.BFloat16(0.0)
                    if local < length:
                        kval = k[token, 0, dim]
                    kr[batch, local, 0, dim] = kval
            for i in cutlass.range_constexpr(5):
                accum += cute.arch.shuffle_sync_bfly(accum, offset=1 << i)
            if lane == 0 and local < cute.ceil_div(length, 128) * 128:
                dst = stats_start + local
                if local < length:
                    delta[head, dst] = accum
                    lse2[head, dst] = lse[token, head] * math.log2(math.e)
                else:
                    delta[head, dst] = cutlass.Float32(0.0)
                    lse2[head, dst] = cutlass.Float32(0.0)


PackedPre.run.set_name_prefix("cudnn_gqa_packed_pre", remove_cutlass_symbol=True)
PackedReduce.run.set_name_prefix("cudnn_gqa_packed_reduce", remove_cutlass_symbol=True)
