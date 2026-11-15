# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device helpers shared by the SM100 FP8 / MXFP8 backward bodies and the SM120 FP8 delta pre-pass.

Moved unchanged from the D192 FP8 dK/dV kernel and the D512 FP8 kernel, which their siblings imported as plain
(unspecialized) modules: the module constants below are those modules' defaults. The fields the helpers read are
the same under every served head shape, block-scale mode and MMA K step.
"""

import cutlass
import cutlass.cute as cute
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, fmax_f32, fmul2, fp32_to_fp16, fp32_to_fp8_pack
from cudnn.frost.tile_dsl.tma import st_shared_v4, tma_store_commit, tma_store_tile, tma_store_wait
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from typing import Optional

from cudnn.sdpa.bwd.config_sm100_d192_dkdv_fp8 import make_config

LOG2E = 1.4426950408889634
FP8 = cutlass.Float8E4M3FN
HALF_DTYPES = (cutlass.BFloat16, cutlass.Float16)
MX = False
CFG, _ = make_config(192, 128, MX, k_hw=32)
M = C = 128
A = 0


def _view(slab, offset, dtype, count, alignment=16):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


def _bpe(dtype):
    if dtype == cutlass.Float32:
        return 4
    if dtype in HALF_DTYPES:
        return 2
    return 1


def _operand(slab, offset, dtype, rows, cols, *, transpose=False, swz=CFG.SWIZZLE_BYTES):
    box = swz // _bpe(dtype)
    return SmemTile(
        base=_view(slab, offset, dtype, rows * cols, alignment=1024),
        elems_per_stage=rows * cols,
        leading_byte_offset=rows * swz if transpose else 16,
        stride_byte_offset=8 * swz,
        layout=CFG.SWIZZLE_LAYOUT if swz == CFG.SWIZZLE_BYTES else CFG.NARROW_SWIZZLE_LAYOUT,
        tma_loads_per_tile=cols // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=rows * box,
        desc_version=CFG.DESC_VERSION,
    )


@cute.jit
def _tmem_load(address, n: cutlass.Constexpr[int], dtype: cutlass.Constexpr = cutlass.Float32):
    return nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(address, dtype), num=n)


@cute.jit
def _scalar(tensor):
    return cutlass.Float32(cutlass.make_array_view(tensor)[0])


@cute.jit
def _fold_amax(value, tensor):
    for step in cutlass.range_constexpr(5):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=value, offset=1 << step, mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        value = fmax_f32(value, peer)
    if cute.arch.thread_idx()[0] % 32 == 0:
        nvvm.atomicrmw(nvvm.AtomicOp.MAX, Pointer(tensor.iterator.raw_ptr(), dtype=cutlass.Int32), value.bitcast(cutlass.Int32))


def _pack_fp8(values):
    words = []
    for chunk in range(len(values) // 16):
        packed = fp32_to_fp8_pack(values[chunk * 16 : chunk * 16 + 16], dtype=FP8)
        words.extend(packed[j] for j in range(4))
    return cutlass.Vector.from_elements(tuple(words), cutlass.Int32)


def _scale_pairs(values, factor):
    scaled = []
    for i in range(0, len(values), 2):
        a, b = fmul2(values[i], values[i + 1], factor, factor)
        scaled.extend((a, b))
    return scaled


def _sf_map(tensor):
    return tmap.create_tensor_map_tiled_from_view(
        tensor, box_dims=(1, 1, 1, tensor.shape[3], 128), stride_order=(4, 3, 2, 1, 0), swizzle=tmap.TensorMapSwizzle.none
    )


def _tensor_map(tensor, rows, box, swz=CFG.SWIZZLE_BYTES):
    swizzle = tmap.TensorMapSwizzle.s128b if swz == CFG.SWIZZLE_BYTES else tmap.TensorMapSwizzle.s64b
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, box), stride_order=(3, 1, 2, 0), swizzle=swizzle)


@cute.jit
def _st16(slab, off, values, i: cutlass.Constexpr[int], dtype: cutlass.Constexpr):
    st_shared_v4(_view(slab, off, dtype, 4).data_ptr(), (values[i], values[i + 1], values[i + 2], values[i + 3]), dtype)


@cute.jit
def _store_chunks(slab, byte_offset, row, words, first: cutlass.Constexpr[int], count: cutlass.Constexpr[int], swz: cutlass.Constexpr[int] = CFG.SWIZZLE_BYTES):
    for vec in cutlass.range_constexpr(count):
        chunk = first + vec
        if cutlass.const_expr(swz == CFG.SWIZZLE_BYTES):
            off = byte_offset + row * swz + ((row % 8) ^ chunk) * 16
        else:
            off = byte_offset + row * swz + (((row // 2) % 4) ^ chunk) * 16
        _st16(slab, off, words, vec * 4, cutlass.Int32)


def _sf_tile(slab, offset, nbytes: cutlass.Constexpr[int]):
    return SmemTile(
        base=_view(slab, offset, cutlass.Int8, nbytes, alignment=128),
        elems_per_stage=nbytes,
        leading_byte_offset=16,
        stride_byte_offset=128,
        layout=0,
        desc_version=CFG.DESC_VERSION,
    )


@cute.jit
def _sf_ptr(tmem_base, col):
    return nvvm.make_tmem_ptr(tmem_base + col, cutlass.Int8)


@cute.jit
def _sf_copy(tmem_base, col, smem_desc, groups: cutlass.Constexpr[int]):
    if nvvm.elect_sync():
        for g in cutlass.range_constexpr(groups):
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                _sf_ptr(tmem_base, col + 4 * g),
                smem_desc + 32 * g,
                group=nvvm.CTAGroup.CTA_1,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )


def _mma_desc(m, n, k, *, atranspose=False, btranspose=False, mx=MX, k_hw=None):
    k_hw = CFG.K_HW if k_hw is None else k_hw
    if mx:
        return MmaDesc(
            M=m,
            N=n,
            K=k,
            bpe_a=CFG.BPE,
            bpe_b=CFG.BPE,
            tile_k_hw=k_hw,
            atranspose=atranspose,
            btranspose=btranspose,
            cta_group=1,
            idesc=nvvm.Tcgen05MxInstrDesc.build(
                a_dtype=FP8, b_dtype=FP8, m_dim=m, n_dim=n, k_dim=int(k_hw == 64), a_major=int(atranspose), b_major=int(btranspose)
            ),
            kind=nvvm.MMABlockScaleKind.MXF8F6F4,
            is_block_scale=True,
            sf_blocks_per_step=k_hw // 32,
            scale_vec_size=nvvm.Tcgen05MMAScaleVecSize.BLOCK32,
        )
    return MmaDesc(
        M=m,
        N=n,
        K=k,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=k_hw,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=FP8, b_dtype=FP8, m_dim=m, n_dim=n, k_dim=int(k_hw == 64), a_major=int(atranspose), b_major=int(btranspose)
        ),
        kind=nvvm.Tcgen05MMAKind.F8F6F4,
    )


@cute.jit
def _join(bar, phase):
    bar.wait(phase, spin=True)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _output(
    slab, tmem, desc, owner, head, batch, chunk, plane: cutlass.Constexpr[int], factor, out_scale, live, dtype: cutlass.Constexpr, amax: Optional[cute.Tensor]
):
    tid = cute.arch.thread_idx()[0]
    row, group = tid % M, tid // M
    vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
    if live:
        vals = _tmem_load(tmem + plane + group * 64, 64)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scaled = _scale_pairs([vals[i] for i in range(64)], factor)
    if cutlass.const_expr(amax is not None):
        _fold_amax(abs_max_tree(scaled), amax)
    if cutlass.const_expr(dtype == FP8):
        _store_chunks(slab, A, row, _pack_fp8(_scale_pairs(scaled, out_scale)), group * 4, 4)
    else:
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(scaled[2 * i], scaled[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        _store_chunks(slab, A + group * M * 128, row, words, 0, 8)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    nvvm.barrier_cta_sync()
    if cute.arch.warp_idx() == 0:
        tma_store_tile(_operand(slab, A, dtype, M, C), GmemTileTma(desc)(chunk * C, owner * M, head, batch))
        tma_store_commit()
        tma_store_wait(0)
    nvvm.barrier_cta_sync()
