# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Local Hadamard helpers for the grouped GEMM GLU hadamard kernel."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode, OperandSource
import torch

HADAMARD_SIZE = 16
TMEM_ROW_STRIDE = 1 << 16
M_PER_CLUSTER = 256


@cute.jit
def hadamard_setup(g_hadamard, s_hadamard, tidx):
    tiled_hmma = sm100_utils.make_trivial_tiled_mma(
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
def hadamard_compute(tiled_hmma, tmem_a_ptr, tmem_acc_ptr, s_hadamard, epi_tile, tidx, pipeline_producer):
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
        hadamard_empty = pipeline_producer.acquire_and_advance()
        for i in cutlass.range_constexpr(cute.size(t_ht_c.shape, mode=[2]), unroll_full=True):
            cute.gemm(
                tiled_hmma,
                cute.append_ones(t_ht_c[None, None, i], up_to_rank=3),
                cute.append_ones(t_ht_a[(None, None, i)], up_to_rank=3),
                cute.append_ones(t_bs_b[(None, None, 0, 0)], up_to_rank=3),
                cute.append_ones(t_ht_c[None, None, i], up_to_rank=3),
            )
        hadamard_empty.commit()


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


def hadamard_matrix(n, dtype=None, device=None):
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
