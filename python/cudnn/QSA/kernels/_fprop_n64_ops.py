# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared-memory and tensor-memory operands for the 64-key QSA forward.

KV retains native forward feature-slab order: each slab holds 64 rows of 64
BF16 features. P has a separate unswizzled MN-major core-8 layout. M64 scores
and M128 output pages have different per-thread coordinates; callers must
redistribute head scalars before updating output.
"""

import cutlass
from cutlass import cute
from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.tma import TMA_L2_EVICT_LAST, opaque_i64, tma_gather4
from cutlass.cute.nvgpu import tcgen05

I32, I64, F32, BF16 = cutlass.Int32, cutlass.Int64, cutlass.Float32, cutlass.BFloat16


@cute.jit
def pointer(raw, offset, dtype: cutlass.Constexpr):
    return cute.make_ptr(
        dtype,
        I64(raw.toint()) + I64(offset),
        cutlass.AddressSpace.smem,
        assumed_align=128,
    )


@cute.jit
def slab_pointer(raw, offset):
    return cute.recast_ptr(pointer(raw, offset, BF16), swizzle_=cute.make_swizzle(3, 4, 3))


@cute.jit
def kv_tensor(raw, offset):
    layout = cute.make_layout(((8, 8), (64, 4)), stride=((64, 512), (1, 4096)))
    return cute.make_tensor(slab_pointer(raw, offset), layout)


@cute.jit
def q_tensor(raw, offset):
    layout = cute.make_layout(((8, 2), (64, 4)), stride=((64, 512), (1, 1024)))
    return cute.make_tensor(slab_pointer(raw, offset), layout)


@cute.jit
def p_tensor(raw, offset):
    layout = cute.make_layout(((8, 2), (8, 8)), stride=((1, 64), (8, 128)))
    return cute.make_tensor(pointer(raw, offset, BF16), layout)


@cute.jit
def store_p(raw, offset, key, head, value):
    index = head % 8 + head // 8 * 64 + key % 8 * 8 + key // 8 * 128
    (pointer(raw, offset, BF16) + index).store(value.to(BF16))


@cute.jit
def load_q(desc, raw, offset, query, head, slab: cutlass.Constexpr, bar):
    if prims.elect_sync():
        prims.cp_async_bulk_tensor_shared_cta_global(
            raw + offset // 4,
            desc.get_ptr(),
            [slab * 64, head * 12, query],
            bar,
        )


@cute.jit
def load_eight(desc, raw, offset, first, head, slab: cutlass.Constexpr, bar):
    if prims.elect_sync():
        prims.cp_async_bulk_tensor_shared_cta_global(
            raw + offset // 4,
            desc.get_ptr(),
            [head * 256 + slab * 64, first],
            bar,
            l2_cache_hint=opaque_i64(TMA_L2_EVICT_LAST),
        )


@cute.jit
def gather_four(desc, raw, offset, head, slab: cutlass.Constexpr, rows, bar):
    if prims.elect_sync():
        tma_gather4(
            desc,
            raw + offset // 4,
            bar,
            head * 256 + slab * 64,
            rows[0],
            rows[1],
            rows[2],
            rows[3],
            l2_hint=opaque_i64(TMA_L2_EVICT_LAST),
        )


@cute.jit
def atom(m: cutlass.Constexpr, major: cutlass.Constexpr):
    return cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            BF16,
            F32,
            (m, 16, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            major,
            major,
        )
    )


@cute.jit
def accumulator(mma, tmem, column, m: cutlass.Constexpr):
    layout = mma.make_fragment_C(mma.partition_shape_C((m, 16))).layout
    return cute.make_tensor(
        cute.make_ptr(F32, I32(tmem + column), cutlass.AddressSpace.tmem, assumed_align=16),
        layout,
    )


@cute.jit
def score(raw, kv_offset, tmem, column, k16, accumulate):
    mma = atom(64, cute.nvgpu.OperandMajorMode.K)
    a = cute.local_tile(kv_tensor(raw, kv_offset), (64, 16), (0, k16))
    b = cute.local_tile(q_tensor(raw, 0), (16, 16), (0, k16))
    part = mma.get_slice(0)
    mma.set(tcgen05.Field.ACCUMULATE, accumulate)
    acc = accumulator(mma, tmem, column, 64)
    cute.gemm(mma, acc, mma.make_fragment_A(part.partition_A(a)), mma.make_fragment_B(part.partition_B(b)), acc)


@cute.jit
def value(raw, kv_offset, p_offset, tmem, column, half: cutlass.Constexpr, k16, accumulate):
    mma = atom(128, cute.nvgpu.OperandMajorMode.MN)
    base = kv_tensor(raw, kv_offset)
    transposed = cute.make_tensor(base.iterator, cute.select(base.layout, mode=[1, 0]))
    a = cute.local_tile(transposed, (128, 16), (half, k16))
    b = cute.local_tile(p_tensor(raw, p_offset), (16, 16), (0, k16))
    part = mma.get_slice(0)
    mma.set(tcgen05.Field.ACCUMULATE, accumulate)
    acc = accumulator(mma, tmem, column, 128)
    cute.gemm(mma, acc, mma.make_fragment_A(part.partition_A(a)), mma.make_fragment_B(part.partition_B(b)), acc)


@cute.jit
def read_score(tmem, column, tid):
    mma = atom(64, cute.nvgpu.OperandMajorMode.K)
    tensor = accumulator(mma, tmem, column, 64)
    copy_atom = cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), F32)
    copy = tcgen05.make_tmem_copy(copy_atom, tensor).get_slice(tid)
    origin = mma.get_slice(0).partition_C(cute.make_identity_tensor((64, 16)))
    registers = cute.make_rmem_tensor(copy.partition_D(origin).shape, F32)
    cute.copy(copy_atom, copy.partition_S(tensor), registers)
    flat = cute.make_tensor(registers.iterator, cute.make_layout(8))
    return cutlass.Vector.from_elements([flat[j] for j in range(8)], F32)


@cute.jit
def output_page(tmem, column):
    return cute.make_tensor(
        cute.make_ptr(F32, I32(tmem + column), cutlass.AddressSpace.tmem, assumed_align=16),
        cute.make_layout((128, 8), stride=(65536, 1)),
    )


@cute.jit
def read_output(tmem, column, tid):
    tensor = output_page(tmem, column)
    copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), F32)
    copy = tcgen05.make_tmem_copy(copy_atom, tensor).get_slice(tid)
    registers = cute.make_rmem_tensor(copy.partition_D(cute.make_identity_tensor((128, 8))).shape, F32)
    cute.copy(copy_atom, copy.partition_S(tensor), registers)
    flat = cute.make_tensor(registers.iterator, cute.make_layout(8))
    return cutlass.Vector.from_elements([flat[j] for j in range(8)], F32)


@cute.jit
def write_output(tmem, column, tid, values):
    tensor = output_page(tmem, column)
    copy_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x8), F32)
    copy = tcgen05.make_tmem_copy(copy_atom, tensor).get_slice(tid)
    registers = cute.make_rmem_tensor(copy.partition_S(cute.make_identity_tensor((128, 8))).shape, F32)
    flat = cute.make_tensor(registers.iterator, cute.make_layout(8))
    for j in cutlass.range_constexpr(8):
        flat[j] = values[j]
    cute.copy(copy_atom, registers, copy.partition_D(tensor))
