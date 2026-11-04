# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper dense tiles with exact masks from the QSA preparation pass."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from .fprop_sm90 import _acc_mn, _quad_max, _quad_sum

M_BLK, N_BLK, HDIM, NTHR = 128, 64, 256, 256
LOG2E = 1.4426950408889634
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.kernel
def frost_qsa_dense_sm90(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    bits: cute.Tensor,
    ranges: cute.Tensor,
    modes: cute.Tensor,
    order: cute.Tensor,
    mO: cute.Tensor,
    mOf: cute.Tensor,
    mLse: cute.Tensor,
    scale: cutlass.Float32,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    gmem_copy: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    group: cutlass.Constexpr,
    d_tile: cutlass.Constexpr,
    SharedStorage: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    m_block, head, d_block = cute.arch.block_idx()
    m_block = order[m_block]
    if modes[m_block] == 0:
        T = mQ.shape[0]
        kvh = head // group

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sVt = cute.composition(sV, cute.make_layout((d_tile, N_BLK), stride=(N_BLK, 1)))
        sMask = storage.sMask.get_tensor(cute.make_layout(2 * M_BLK))

        gQ = cute.local_tile(mQ[None, head, None], (M_BLK, HDIM), (m_block, 0))
        gK = cute.local_tile(mK[None, kvh, None], (N_BLK, HDIM), (None, 0))
        gV = cute.local_tile(mV[None, kvh, None], (N_BLK, d_tile), (None, d_block))

        thr_copy = gmem_copy.get_slice(tidx)
        tQgQ = thr_copy.partition_S(gQ)
        tQsQ = thr_copy.partition_D(sQ)
        tKgK = thr_copy.partition_S(gK)
        tKsK = thr_copy.partition_D(sK)
        tVgV = thr_copy.partition_S(gV)
        tVsV = thr_copy.partition_D(sV)

        cQ = cute.local_tile(cute.make_identity_tensor((T, HDIM)), (M_BLK, HDIM), (m_block, 0))
        cKV = cute.local_tile(cute.make_identity_tensor((T, HDIM)), (N_BLK, HDIM), (0, 0))
        tQcQ = thr_copy.partition_S(cQ)
        tKcK = thr_copy.partition_S(cKV)

        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][0], T):
                cute.copy(gmem_copy, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        cute.arch.cp_async_commit_group()

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((M_BLK, d_tile)), cutlass.Float32)
        acc_O.fill(0.0)

        atom_q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16)
        atom_v = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.BFloat16)
        sc_Q = cute.make_tiled_copy_A(atom_q, tiled_mma).get_slice(tidx)
        sc_K = cute.make_tiled_copy_B(atom_q, tiled_mma).get_slice(tidx)
        sc_V = cute.make_tiled_copy_B(atom_v, tiled_mma).get_slice(tidx)
        tSsQ = sc_Q.partition_S(sQ)
        tSrQv = sc_Q.retile(tSrQ)
        tSsK = sc_K.partition_S(sK)
        tSrKv = sc_K.retile(tSrK)
        tOsVt = sc_V.partition_S(sVt)
        tOrVtv = sc_V.retile(tOrVt)

        cS0 = cute.local_tile(cute.make_identity_tensor((T, T)), (M_BLK, N_BLK), (m_block, 0))
        tS0 = _acc_mn(thr_mma.partition_C(cS0))
        nrows = cutlass.const_expr(cute.size(tS0.shape[0]))
        ncols = cutlass.const_expr(cute.size(tS0.shape[1]))

        row_max = cute.make_rmem_tensor(nrows, cutlass.Float32)
        row_sum = cute.make_rmem_tensor(nrows, cutlass.Float32)
        r_loc = cute.make_rmem_tensor(nrows, cutlass.Int32)
        for r in cutlass.range_constexpr(nrows):
            row_max[r] = -cutlass.Float32.inf
            row_sum[r] = 0.0
            r_loc[r] = tS0[r, 0][0] - m_block * M_BLK

        n_start = ranges[m_block, 0] // N_BLK
        n_end = n_start + ranges[m_block, 1] * 2
        scale2 = scale * LOG2E

        for n_block in cutlass.range(n_start, n_end, 1, unroll=1):
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKcK[0, n, 0][0] + n_block * N_BLK, T):
                    cute.copy(gmem_copy, tKgK[None, n, None, n_block], tKsK[None, n, None])
                    cute.copy(gmem_copy, tVgV[None, n, None, n_block], tVsV[None, n, None])
                else:
                    tKsK[None, n, None].fill(0)
                    tVsV[None, n, None].fill(0)
            cute.arch.cp_async_commit_group()

            row = m_block * M_BLK + tidx // 2
            word = cutlass.Int32(0)
            if row < T:
                word = bits[row, n_block * 2 + tidx % 2]
            sMask[tidx] = word
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((M_BLK, N_BLK)), cutlass.Float32)
            acc_S.fill(0.0)
            for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(sc_Q, tSsQ[None, None, kk], tSrQv[None, None, kk])
                cute.copy(sc_K, tSsK[None, None, kk], tSrKv[None, None, kk])
                cute.gemm(tiled_mma, acc_S, tSrQ[None, None, kk], tSrK[None, None, kk], acc_S)

            accS = _acc_mn(acc_S)
            accO = _acc_mn(acc_O)
            for r in cutlass.range_constexpr(nrows):
                wl = sMask[r_loc[r] * 2]
                wh = sMask[r_loc[r] * 2 + 1]
                for c in cutlass.range_constexpr(ncols):
                    jr = tS0[0, c][1]
                    ws = wl if jr < 32 else wh
                    v = accS[r, c] * scale2
                    accS[r, c] = v if ((ws >> (jr & 31)) & 1) != 0 else -cutlass.Float32.inf

                s_row = accS[r, None].load()
                m_cur = _quad_max(s_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0))
                m_prev = row_max[r]
                m_cur = cute.arch.fmax(m_prev, m_cur)
                m_safe = 0.0 if m_cur == -cutlass.Float32.inf else m_cur
                s_exp = cute.math.exp2(s_row - m_safe, fastmath=True)
                s_sum = s_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                corr = cute.math.exp2((0.0 if m_prev == -cutlass.Float32.inf else m_prev) - m_safe, fastmath=True)
                corr = 0.0 if m_prev == -cutlass.Float32.inf else corr
                row_sum[r] = s_sum + row_sum[r] * corr
                row_max[r] = m_cur
                accO[r, None] = accO[r, None].load() * corr
                accS[r, None] = s_exp

            rP = cute.make_fragment_like(acc_S, cutlass.BFloat16)
            rP.store(acc_S.load().to(cutlass.BFloat16))
            d = cute.logical_divide(rP.layout, (None, None, 2))
            rP_view = cute.make_layout(
                ((d.shape[0], d.shape[2][0]), d.shape[1], d.shape[2][1]),
                stride=((d.stride[0], d.stride[2][0]), d.stride[1], d.stride[2][1]),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_view)
            for kk in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                cute.copy(sc_V, tOsVt[None, None, kk], tOrVtv[None, None, kk])
                cute.gemm(tiled_mma, acc_O, tOrS[None, None, kk], tOrVt[None, None, kk], acc_O)
            cute.arch.barrier()

        accO = _acc_mn(acc_O)
        cO = cute.local_tile(cute.make_identity_tensor((T, HDIM)), (M_BLK, d_tile), (m_block, d_block))
        tOc = _acc_mn(thr_mma.partition_C(cO))
        ncolO = cutlass.const_expr(cute.size(tOc.shape[1]))
        lane = tidx % 4
        for r in cutlass.range_constexpr(nrows):
            s = _quad_sum(row_sum[r])
            inv = 1.0 / s if s > 0.0 else 0.0
            qi = tOc[r, 0][0]
            if qi < T:
                if lane == 0:
                    mm = row_max[r]
                    mLse[head, qi] = (
                        -cutlass.Float32.inf
                        if (mm == -cutlass.Float32.inf or s <= 0.0)
                        else (mm + cute.math.log2(s, fastmath=True)) * cutlass.Float32(1.0 / LOG2E)
                    )
                for c in cutlass.range_constexpr(ncolO):
                    val = accO[r, c] * inv
                    dcol = tOc[r, c][1]
                    mOf[qi, head, dcol] = val
                    mO[qi, head, dcol] = val.to(cutlass.BFloat16)


frost_qsa_dense_sm90.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    mQ,
    mK,
    mV,
    bits,
    ranges,
    modes,
    order,
    mO,
    mOf,
    mLse,
    scale: cutlass.Float32,
    group: cutlass.Constexpr,
    d_tile: cutlass.Constexpr,
    stream,
):
    atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1)))
    sQ_layout = cute.tile_to_shape(atom, (M_BLK, HDIM), (0, 1))
    sK_layout = cute.tile_to_shape(atom, (N_BLK, HDIM), (0, 1))
    sV_layout = cute.tile_to_shape(atom, (N_BLK, d_tile), (0, 1))

    @cute.struct
    class SharedStorage:
        sQ: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sQ_layout)], 1024]
        sK: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sK_layout)], 1024]
        sV: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sV_layout)], 1024]
        sMask: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2 * M_BLK], 128]

    atom_async = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.BFloat16, num_bits_per_copy=128)
    tlayout = cute.make_layout((NTHR // 8, 8), stride=(8, 1))
    gmem_copy = cute.make_tiled_copy_tv(atom_async, tlayout, cute.make_layout((1, 8)))
    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
        (NTHR // 32, 1, 1),
        permutation_mnk=(NTHR // 32 * 16, 16, 16),
    )
    grid = (cute.ceil_div(mQ.shape[0], M_BLK), cute.size(mQ.shape[1]), HDIM // d_tile)
    frost_qsa_dense_sm90(
        mQ,
        mK,
        mV,
        bits,
        ranges,
        modes,
        order,
        mO,
        mOf,
        mLse,
        scale,
        sQ_layout,
        sK_layout,
        sV_layout,
        gmem_copy,
        tiled_mma,
        group,
        d_tile,
        SharedStorage,
    ).launch(grid=grid, block=[NTHR, 1, 1], stream=stream)
