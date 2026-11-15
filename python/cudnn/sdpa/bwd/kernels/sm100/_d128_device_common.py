# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Configuration-independent device helpers shared by small-half backward bodies."""

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm


def _view(slab, offset, dtype, count, alignment=16):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


@cute.jit
def _wait_accum(bar, phase):
    bar.wait(phase)
    # All full-warp consumers reconverge before any elected credit/issue.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _wait_tcgen_accum(bar, phase):
    _wait_accum(bar, phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _commit(bar):
    if nvvm.elect_sync():
        bar.arrive(cta_group=1)


@cute.jit
def _credit_accum(bar):
    if nvvm.elect_sync():
        bar.arrive()


@cute.jit
def _tmem_load(address, n: cutlass.Constexpr[int], dtype: cutlass.Constexpr = cutlass.Float32):
    return nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(address, dtype), num=n)


@cute.jit
def _tmem_store(address, values):
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(address, cutlass.Int32), values)


@cute.jit
def _decode_flat(tile, nheads, nk, swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    group = tile // (swizzle * nk)
    inner = tile % (swizzle * nk)
    block = cutlass.Int32(0)
    residual = cutlass.Int32(0)
    if group < batch_heads // swizzle:
        block = inner // swizzle
        residual = inner % swizzle
    else:
        block = inner // max(batch_heads % swizzle, 1)
        residual = inner % max(batch_heads % swizzle, 1)
    hb = group * swizzle + residual
    return block, hb % nheads, hb // nheads
