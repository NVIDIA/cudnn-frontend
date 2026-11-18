# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped d128 per-tensor FP8 backward: the dK / dV group fold with scaling and amax."""

from typing import Optional

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import abs_max_tree, fmax_f32, fp32_to_fp16, fp32_to_fp8x2, opaque_f32_zero, pack_fp8x2_pairs

THREADS = 256
VEC = 8
FP8 = cutlass.Float8E4M3FN


@cute.jit
def _scalar(tensor):
    return cutlass.Float32(cutlass.make_array_view(tensor)[0])


@cute.jit
def _block_amax(value, tensor, slot: cutlass.Constexpr[int], scratch):
    tid = cute.arch.thread_idx()[0]
    for step in cutlass.range_constexpr(5):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=value, offset=1 << step, mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        value = fmax_f32(value, peer)
    if tid % 32 == 0:
        scratch.subview(slot * 8 + tid // 32).store(value)
    nvvm.barrier_cta_sync()
    if tid == 0:
        folded = scratch.subview(slot * 8).load()
        for warp in cutlass.range_constexpr(1, THREADS // 32):
            folded = fmax_f32(folded, scratch.subview(slot * 8 + warp).load())
        nvvm.atomicrmw(nvvm.AtomicOp.MAX, Pointer(tensor.iterator.raw_ptr(), dtype=cutlass.Int32), folded.bitcast(cutlass.Int32))


@cute.jit
def _fold_vector(part, out, gidx, h_kv, group: cutlass.Constexpr[int], d: cutlass.Constexpr[int], factor, grad_dtype: cutlass.Constexpr):
    col = (gidx * VEC) % d
    kv_head = (gidx * VEC // d) % h_kv
    row = gidx * VEC // d // h_kv
    h_q = h_kv * group
    acc = [opaque_f32_zero() for _ in range(VEC)]
    for member in cutlass.range_constexpr(group):
        q_head = kv_head * group + member
        offset = cutlass.Int64(row) * cutlass.Int64(h_q * d) + cutlass.Int64(q_head * d) + cutlass.Int64(col)
        values = (part.iterator.raw_ptr() + offset).load(count=VEC, alignment=16)
        for j in cutlass.range_constexpr(VEC):
            acc[j] = acc[j] + cutlass.Float32(values[j])
    amax = abs_max_tree(acc)
    out_offset = cutlass.Int64(row) * cutlass.Int64(h_kv * d) + cutlass.Int64(kv_head * d) + cutlass.Int64(col)
    if cutlass.const_expr(grad_dtype == FP8):
        scaled = [x * factor for x in acc]
        packed = cutlass.Vector.from_elements(
            (
                pack_fp8x2_pairs(fp32_to_fp8x2(scaled[0], scaled[1], dtype=FP8), fp32_to_fp8x2(scaled[2], scaled[3], dtype=FP8)),
                pack_fp8x2_pairs(fp32_to_fp8x2(scaled[4], scaled[5], dtype=FP8), fp32_to_fp8x2(scaled[6], scaled[7], dtype=FP8)),
            ),
            cutlass.Int32,
        )
        Pointer(cutlass.make_array_view(out).data_ptr() + out_offset, dtype=cutlass.Int32).store(packed, alignment=8)
    else:
        packed = cutlass.Vector.from_elements(
            tuple(fp32_to_fp16(acc[2 * j], acc[2 * j + 1], dtype=grad_dtype) for j in range(VEC // 2)),
            cutlass.Int32,
        )
        Pointer(cutlass.make_array_view(out).data_ptr() + out_offset, dtype=cutlass.Int32).store(packed, alignment=16)
    return amax


@cute.kernel
def fold_dkv_fp8_kernel(
    dk_part: cute.Tensor,
    dv_part: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale_dk: cute.Tensor,
    scale_dv: cute.Tensor,
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    group: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    grad_dtype: cutlass.Constexpr,
):
    """One thread per 16-byte partial vector of dK and of dV; serial FP32 group sums in a fixed order."""
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    h_kv = dk.shape[2]
    out_vecs = dk.shape[0] * dk.shape[1] * h_kv * d // VEC
    gidx = bidx * THREADS + tidx
    fp8_out = cutlass.const_expr(grad_dtype == FP8)
    k_factor = cutlass.Float32(1.0)
    v_factor = cutlass.Float32(1.0)
    if cutlass.const_expr(fp8_out):
        k_factor = _scalar(scale_dk)
        v_factor = _scalar(scale_dv)
    amax_k = opaque_f32_zero()
    amax_v = opaque_f32_zero()
    if gidx < out_vecs:
        amax_k = _fold_vector(dk_part, dk, gidx, h_kv, group, d, k_factor, grad_dtype)
        amax_v = _fold_vector(dv_part, dv, gidx, h_kv, group, d, v_factor, grad_dtype)
    scratch = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(amax_dk is not None):
        _block_amax(amax_k, amax_dk, 0, scratch)
    if cutlass.const_expr(amax_dv is not None):
        _block_amax(amax_v, amax_dv, 1, scratch)


fold_dkv_fp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def fold_dkv_fp8_host(
    dk_part: cute.Tensor,
    dv_part: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale_dk: cute.Tensor,
    scale_dv: cute.Tensor,
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    group: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    """One launch over the compact [B, S_kv, H_kv, D] outputs."""
    d = dk.shape[3]
    out_vecs = dk.shape[0] * dk.shape[1] * dk.shape[2] * d // VEC
    fold_dkv_fp8_kernel(dk_part, dv_part, dk, dv, scale_dk, scale_dv, amax_dk, amax_dv, group, d, dk.element_type).launch(
        grid=((out_vecs + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream
    )
