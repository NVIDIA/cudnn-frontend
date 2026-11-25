# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Hadamard helpers for the grouped GEMM GLU hadamard kernel."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.nvgpu.tcgen05 import OperandSource
from cutlass._mlir.dialects import nvvm

HADAMARD_SIZE = 16
TMEM_ROW_STRIDE = 1 << 16
M_PER_CLUSTER = 256


@cute.jit
def hadamard_setup(g_hadamard, s_hadamard, tidx):
    tiled_hmma = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.TWO,
        (M_PER_CLUSTER, HADAMARD_SIZE),
        OperandSource.TMEM,
    )
    s_hadamard[tidx] = g_hadamard[tidx if tidx < 64 else (tidx ^ 8)]
    return tiled_hmma


@cute.jit
def tcgen05_fence_before_thread_sync():
    """``tcgen05.fence::before_thread_sync``: orders this thread's prior tcgen05 ops before a following thread sync."""
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_fence_after_thread_sync():
    """``tcgen05.fence::after_thread_sync``: orders this thread's later tcgen05 ops after a preceding thread sync."""
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


@cute.jit
def hadamard_compute(tiled_hmma, tmem_a_ptr, tmem_acc_ptr, s_hadamard, epi_tile, tidx, hadamard_pipeline, hadamard_state):
    """Issue one 2-CTA Hadamard UMMA (leader CTA, warp 0) once both CTAs' operands are in TMEM.

    ``hadamard_pipeline`` is the 1-stage ``PipelineUmmaAsync`` whose empty barrier (on the leader CTA)
    counts all 256 RHT threads of the CTA pair. Every RHT thread arrives on it after it has finished
    reading the previous accumulator and written this round's A operand (see ``hadamard_tmem_round``),
    so waiting on it with the current phase gates the MMA on both CTAs' operands and keeps it from
    overwriting an accumulator that is still being read. ``hadamard_state`` is owned and advanced by
    the caller, so its phase is carried across tiles.
    """
    n = epi_tile[1]
    mma_tiler = (M_PER_CLUSTER, HADAMARD_SIZE, HADAMARD_SIZE)
    cta_rank = cute.arch.block_in_cluster_idx()[0]
    thr = tiled_hmma.get_slice(cta_rank)
    b_layout = sm100_utils.make_smem_layout_b(tiled_hmma, mma_tiler, cutlass.BFloat16, 1)
    s_bh = s_hadamard.get_tensor(b_layout.outer, swizzle=b_layout.inner)
    t_bs_b = tiled_hmma.make_fragment_B(s_bh)

    t_a_subtile = cute.make_tensor(0, cute.make_layout((M_PER_CLUSTER, n), stride=(n, 1)))
    t_ht_a_frg = thr.partition_A(t_a_subtile)
    t_ht_a = cute.make_tensor(
        cute.recast_ptr(tmem_a_ptr, dtype=cutlass.BFloat16),
        tiled_hmma.make_fragment_A(t_ht_a_frg.layout).layout,
    )

    t_ht_c_frg = thr.partition_C(t_a_subtile)
    t_ht_c = cute.make_tensor(
        tmem_acc_ptr,
        tiled_hmma.make_fragment_C(t_ht_c_frg.layout).layout,
    )

    if cta_rank == 0 and tidx < 32:
        # Wait for all 256 RHT threads of the CTA pair.
        hadamard_pipeline.producer_acquire(hadamard_state)
        # Order the MMA's TMEM reads/writes after the other threads' tcgen05.st / tcgen05.ld.
        tcgen05_fence_after_thread_sync()
        for i in cutlass.range_constexpr(cute.size(t_ht_c.shape, mode=[2]), unroll_full=True):
            cute.gemm(
                tiled_hmma,
                cute.append_ones(t_ht_c[None, None, i], up_to_rank=3),
                cute.append_ones(t_ht_a[(None, None, i)], up_to_rank=3),
                cute.append_ones(t_bs_b[(None, None, 0, 0)], up_to_rank=3),
                cute.append_ones(t_ht_c[None, None, i], up_to_rank=3),
            )
        # tcgen05.commit multicast to the full barrier of both CTAs.
        hadamard_pipeline.producer_commit(hadamard_state)


@cute.jit
def hadamard_in(rmem_src: cute.Tensor, cols: cutlass.Constexpr, tmem_ptr, tidx):
    tmem_ptr = cute.make_ptr(
        cutlass.Float32,
        tmem_ptr.toint(),
        cute.AddressSpace.tmem,
        assumed_align=8,
    )
    tmem_tensor = cute.make_tensor(
        tmem_ptr,
        cute.make_layout((128, cols), stride=(TMEM_ROW_STRIDE, 1)),
    )

    if cutlass.const_expr(cols == 32):
        st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    else:
        st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    tiled_st = tcgen05.make_tmem_copy(st_atom, tmem_tensor)
    thr_st = tiled_st.get_slice(tidx)
    t_d_st = thr_st.partition_D(tmem_tensor)
    cute.copy(thr_st, cute.recast_tensor(rmem_src, cutlass.Float32), t_d_st)


@cute.jit
def hadamard_transpose_in(sD, d_buffer, feature_offset: cutlass.Constexpr, tmem_ptr, tidx):
    tmem_ptr = cute.make_ptr(
        cutlass.Float32,
        tmem_ptr.toint(),
        cute.AddressSpace.tmem,
        assumed_align=8,
    )
    tmem_tensor = cute.make_tensor(
        tmem_ptr,
        cute.make_layout((128, HADAMARD_SIZE), stride=(TMEM_ROW_STRIDE, 1)),
    )

    st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    tiled_st = tcgen05.make_tmem_copy(st_atom, tmem_tensor)
    thr_st = tiled_st.get_slice(tidx)

    cT = cute.make_identity_tensor((128, HADAMARD_SIZE))
    t_c_st = thr_st.partition_S(cT)
    rmem_src = cute.make_rmem_tensor(t_c_st.shape, cutlass.BFloat16)
    for i in cutlass.range_constexpr(cute.size(rmem_src)):
        row = cute.get(t_c_st[i], mode=[0])
        token_offset = cute.get(t_c_st[i], mode=[1])
        token_block = row // HADAMARD_SIZE
        feature = (row % HADAMARD_SIZE) + feature_offset
        token = token_block * HADAMARD_SIZE + token_offset
        rmem_src[i] = sD[(token, feature, d_buffer)].to(cutlass.BFloat16)

    t_d_st = thr_st.partition_D(tmem_tensor)
    cute.copy(thr_st, cute.recast_tensor(rmem_src, cutlass.Float32), t_d_st)


@cute.jit
def hadamard_out(rmem_dst: cute.Tensor, cols: cutlass.Constexpr, tmem_ptr, tidx):
    tmem_ptr = cute.make_ptr(
        cutlass.Float32,
        tmem_ptr.toint(),
        cute.AddressSpace.tmem,
        assumed_align=8,
    )
    tmem_tensor = cute.make_tensor(
        tmem_ptr,
        cute.make_layout((128, cols), stride=(TMEM_ROW_STRIDE, 1)),
    )

    if cutlass.const_expr(cols == 32):
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    else:
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tmem_tensor)
    thr_ld = tiled_ld.get_slice(tidx)
    t_d_ld_ = thr_ld.partition_D(tmem_tensor)
    t_d_ld = cute.make_tensor(
        cute.make_ptr(
            cutlass.Float32,
            t_d_ld_.iterator.toint(),
            cute.AddressSpace.tmem,
            assumed_align=8,
        ),
        t_d_ld_.layout,
    )
    cute.copy(thr_ld, t_d_ld, cute.recast_tensor(rmem_dst, cutlass.Float32))


@cute.jit
def hadamard_transpose_out_amax(tmem_ptr, tidx, amax_fp32):
    tmem_ptr = cute.make_ptr(
        cutlass.Float32,
        tmem_ptr.toint(),
        cute.AddressSpace.tmem,
        assumed_align=8,
    )
    tmem_tensor = cute.make_tensor(
        tmem_ptr,
        cute.make_layout((128, HADAMARD_SIZE), stride=(TMEM_ROW_STRIDE, 1)),
    )

    ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tmem_tensor)
    thr_ld = tiled_ld.get_slice(tidx)
    t_d_ld_ = thr_ld.partition_D(tmem_tensor)
    t_d_ld = cute.make_tensor(
        cute.make_ptr(
            cutlass.Float32,
            t_d_ld_.iterator.toint(),
            cute.AddressSpace.tmem,
            assumed_align=8,
        ),
        t_d_ld_.layout,
    )
    rmem_dst = cute.make_rmem_tensor(t_d_ld.shape, cutlass.Float32)
    cute.copy(thr_ld, t_d_ld, rmem_dst)
    # tcgen05.wait::ld: the registers are valid, and this thread's read of the accumulator has completed
    # before it overwrites the same TMEM columns with the next A operand and arrives on the empty barrier.
    cute.arch.fence_view_async_tmem_load()

    import cutlass._mlir.dialects.math as _math

    for i in cutlass.range_constexpr(cute.size(rmem_dst)):
        # The consumer quantizes the transformed values after rounding them to BF16
        # (the RHT runs as a BF16 MMA there), so take the amax of the rounded values.
        rounded = rmem_dst[i].to(cutlass.BFloat16).to(cutlass.Float32)
        abs_val = cutlass.Float32(_math.absf(rounded.ir_value()))
        amax_fp32 = cute.arch.fmax(amax_fp32, abs_val)
    return amax_fp32


@cute.jit
def hadamard_tmem_round(
    sD,
    d_buffer,
    feature_offset: cutlass.Constexpr,
    tiled_hmma,
    tmem_a_ptr,
    tmem_acc_ptr,
    s_hadamard,
    tidx,
    hadamard_pipeline,
    hadamard_state,
    amax_fp32,
):
    """One TMEM post-RHT amax round (16 features) for all 256 RHT threads of the CTA pair.

    The A operand and the accumulator share the same TMEM columns, so each round is:
      1. tcgen05.st A                       (all threads; the previous round's tcgen05.ld already
                                             completed via tcgen05.wait::ld)
      2. tcgen05.wait::st + fence::before_thread_sync, then arrive on the leader's empty barrier
      3. leader warp 0: wait for all 256 arrivals, fence::after_thread_sync, MMA, commit (multicast)
      4. all threads: wait full barrier, fence::after_thread_sync, tcgen05.ld + tcgen05.wait::ld
    Each barrier completes exactly one phase per round, so one ``hadamard_state`` (phase 0 initially)
    serves both the empty (producer) and the full (consumer) waits; the caller must ``advance()`` it
    once after every call. No thread arrives for round r+1 before it has passed the full wait of
    round r, and the MMA of round r+1 needs all 256 arrivals, so neither parity wait can alias.
    """
    hadamard_transpose_in(sD, d_buffer, feature_offset, tmem_a_ptr, tidx)
    cute.arch.fence_view_async_tmem_store()  # tcgen05.wait::st
    tcgen05_fence_before_thread_sync()
    hadamard_pipeline.consumer_release(hadamard_state)  # arrive on the leader CTA's empty barrier
    hadamard_compute(
        tiled_hmma,
        tmem_a_ptr,
        tmem_acc_ptr,
        s_hadamard,
        (M_PER_CLUSTER, HADAMARD_SIZE),
        tidx,
        hadamard_pipeline,
        hadamard_state,
    )
    hadamard_pipeline.consumer_wait(hadamard_state)
    tcgen05_fence_after_thread_sync()
    return hadamard_transpose_out_amax(tmem_acc_ptr, tidx, amax_fp32)


@cute.jit
def hadamard_smem_transpose_fwht_amax(sD, d_buffer, feature_offset: cutlass.Constexpr, tidx, amax_fp32, sign_mask: cutlass.Constexpr = 0):
    token_block = tidx // HADAMARD_SIZE
    feature = (tidx % HADAMARD_SIZE) + feature_offset
    token_base = token_block * HADAMARD_SIZE

    rmem_vals = cute.make_rmem_tensor((HADAMARD_SIZE,), cutlass.Float32)
    for i in cutlass.range_constexpr(HADAMARD_SIZE):
        rmem_vals[i] = sD[(token_base + i, feature, d_buffer)].to(cutlass.BFloat16).to(cutlass.Float32)
        # Randomized Hadamard transform: ``sign_mask`` bit ``i`` negates input element ``i``
        # of each 16-token block (the convention of the consumer's RHT quantization).
        if cutlass.const_expr(((sign_mask >> i) & 1) == 1):
            rmem_vals[i] = -rmem_vals[i]

    for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 2):
        x = rmem_vals[base]
        y = rmem_vals[base + 1]
        rmem_vals[base] = x + y
        rmem_vals[base + 1] = x - y
    for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 4):
        for i in cutlass.range_constexpr(2):
            x = rmem_vals[base + i]
            y = rmem_vals[base + i + 2]
            rmem_vals[base + i] = x + y
            rmem_vals[base + i + 2] = x - y
    for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 8):
        for i in cutlass.range_constexpr(4):
            x = rmem_vals[base + i]
            y = rmem_vals[base + i + 4]
            rmem_vals[base + i] = x + y
            rmem_vals[base + i + 4] = x - y
    for i in cutlass.range_constexpr(8):
        x = rmem_vals[i]
        y = rmem_vals[i + 8]
        rmem_vals[i] = x + y
        rmem_vals[i + 8] = x - y

    import cutlass._mlir.dialects.math as _math

    for i in cutlass.range_constexpr(HADAMARD_SIZE):
        # The consumer quantizes the transformed values after rounding them to BF16 (its
        # RHT runs as a BF16 MMA), so take the amax of the rounded values. The 1/4
        # normalization applied by the caller is exact and commutes with the rounding.
        rounded = rmem_vals[i].to(cutlass.BFloat16).to(cutlass.Float32)
        abs_val = cutlass.Float32(_math.absf(rounded.ir_value()))
        amax_fp32 = cute.arch.fmax(amax_fp32, abs_val)
    return amax_fp32


def hadamard_matrix(n, dtype=None, device=None):
    import torch

    if dtype is None:
        dtype = torch.float32
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n & (n - 1):
        raise ValueError("n must be a power of 2")

    kwargs = {"dtype": dtype}
    if device is not None:
        kwargs["device"] = device
    matrix = torch.tensor([[1]], **kwargs)
    base = torch.tensor([[1, 1], [1, -1]], **kwargs)
    while matrix.shape[0] < n:
        matrix = torch.kron(matrix, base)
    return matrix
