# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Refresh compact dS pointers and reduce group gradients."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda


class ReduceAndPointers:
    def __init__(self, coarse_bin=4096, groups=4):
        assert coarse_bin > 0 and groups in (1, 2, 4, 8)
        self.coarse_bin = coarse_bin
        self.groups = groups

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        dq: cute.Tensor,
        ds: cute.Tensor,
        tables: cute.Tensor,
        dkp: cute.Tensor,
        dvp: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.run(k, dq, ds, tables, dkp, dvp, dk, dv).launch(grid=(cute.ceil_div(dk.shape[0] * 256, 2048), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def run(self, k: cute.Tensor, dq: cute.Tensor, ds: cute.Tensor, tables: cute.Tensor, dkp: cute.Tensor, dvp: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor):
        bx, _, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        pos = bx * 2048 + tx * 8
        token = pos // 256
        dim = pos % 256
        if token < dk.shape[0]:
            sk = cute.make_rmem_tensor(8, cutlass.Float32)
            sv = cute.make_rmem_tensor(8, cutlass.Float32)
            sk.fill(0.0)
            sv.fill(0.0)
            rk = cute.make_rmem_tensor(8, dkp.element_type)
            rv = cute.make_rmem_tensor(8, dvp.element_type)
            if token < k.shape[0]:
                for group in cutlass.range_constexpr(self.groups):
                    offset = cute.assume((token * self.groups + group) * 256 + dim, divby=8)
                    kg = cute.make_tensor(dkp.iterator + offset, cute.make_layout(8))
                    vg = cute.make_tensor(dvp.iterator + offset, cute.make_layout(8))
                    cute.autovec_copy(kg, rk)
                    cute.autovec_copy(vg, rv)
                    sk.store(sk.load() + rk.load().to(cutlass.BFloat16).to(cutlass.Float32))
                    sv.store(sv.load() + rv.load().to(cutlass.BFloat16).to(cutlass.Float32))
            outk = cute.make_rmem_tensor(8, dk.element_type)
            outv = cute.make_rmem_tensor(8, dv.element_type)
            outk.store(sk.load().to(dk.element_type))
            outv.store(sv.load().to(dv.element_type))
            out_offset = cute.assume(token * 256 + dim, divby=8)
            ko = cute.make_tensor(dk.iterator + out_offset, cute.make_layout(8))
            vo = cute.make_tensor(dv.iterator + out_offset, cute.make_layout(8))
            cute.autovec_copy(outk, ko)
            cute.autovec_copy(outv, vo)
            if token >= k.shape[0]:
                for head in cutlass.range_constexpr(8):
                    qo = cute.make_tensor(dq.iterator + (token * 8 + head) * 256 + dim, cute.make_layout(8))
                    cute.autovec_copy(outk, qo)


ReduceAndPointers.run.set_name_prefix("cudnn_gqa_reduce", remove_cutlass_symbol=True)


class BandPointers:
    @cute.jit
    def __call__(self, k: cute.Tensor, dq: cute.Tensor, ds: cute.Tensor, tables: cute.Tensor, query_start: cutlass.Int32, stream: cuda.CUstream):
        self.run(k, dq, ds, tables, query_start).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def run(self, k: cute.Tensor, dq: cute.Tensor, ds: cute.Tensor, tables: cute.Tensor, query_start: cutlass.Int32):
        tx, _, _ = cute.arch.thread_idx()
        if tx < 8:
            for tile in range(tables.shape[0]):
                local_start = tile * 4096
                tables[tile, 0, tx] = k.iterator.toint() + cutlass.Int64(tile) * k.shape[1] * 256 * 2
                tables[tile, 1, tx] = ds.iterator.toint() + (cutlass.Int64(tx) * ds.shape[2] + local_start) * ds.shape[3] * 2
                tables[tile, 2, tx] = dq.iterator.toint() + (cutlass.Int64(query_start + local_start) * 2048 + tx * 256) * 2


class PackK:
    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor, query_start: cutlass.Int32, rows: cutlass.Int32, stream: cuda.CUstream):
        self.run(src, dst, query_start, rows).launch(grid=(cute.ceil_div((query_start + rows) * 256, 2048), dst.shape[0], 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def run(self, src: cute.Tensor, dst: cute.Tensor, query_start: cutlass.Int32, rows: cutlass.Int32):
        bx, tile, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        width = cutlass.min(rows - tile * 4096, 4096)
        tile_start = query_start + tile * 4096
        pos = bx * 2048 + tx * 8
        row = pos // 256
        if tile_start < src.shape[0] and row < tile_start + width:
            source_row = tile_start + row if row < width else row - width
            value = cute.make_rmem_tensor(8, cutlass.BFloat16)
            if source_row < src.shape[0]:
                inp = cute.make_tensor(src.iterator + cutlass.Int64(source_row) * 256 + pos % 256, cute.make_layout(8))
                cute.autovec_copy(inp, value)
            else:
                value.fill(0.0)
            offset = cutlass.Int64(tile) * dst.shape[1] * 256 + pos
            out = cute.make_tensor(dst.iterator + offset, cute.make_layout(8))
            cute.autovec_copy(value, out)


BandPointers.run.set_name_prefix("cudnn_gqa_band_pointers", remove_cutlass_symbol=True)
PackK.run.set_name_prefix("cudnn_gqa_band_pack_k", remove_cutlass_symbol=True)


class ClearDiagonal:
    @cute.jit
    def __call__(self, ds: cute.Tensor, stream: cuda.CUstream):
        self.run(ds).launch(grid=(cute.ceil_div(ds.shape[2] * 4096, 16384), 8, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def run(self, ds: cute.Tensor):
        bx, head, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        pos = bx * 16384 + tx * 64
        row = pos // 4096
        column = pos % 4096
        width = cutlass.min(ds.shape[2] - row // 4096 * 4096, 4096)
        if row < ds.shape[2] and column < width and column // 128 > row % 4096 // 128:
            offset = cute.assume((cutlass.Int64(head) * ds.shape[2] + row) * ds.shape[3] + column, divby=64)
            out = cute.make_tensor(ds.iterator + offset, cute.make_layout(64))
            zero = cute.make_rmem_tensor(64, cutlass.BFloat16)
            zero.fill(0.0)
            store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
            cute.copy(store, zero, out)


ClearDiagonal.run.set_name_prefix("cudnn_gqa_band_clear_diagonal", remove_cutlass_symbol=True)
