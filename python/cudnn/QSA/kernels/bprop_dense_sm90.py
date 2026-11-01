# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hopper backward tiles with exact packed masks and FP32 gradient reduction."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16 as BF16, Float32 as F32, Int32 as I32, Int64 as I64
from cutlass.cute.nvgpu import warp

M, N, D, THREADS = 32, 32, 256, 128
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.kernel
def frost_qsa_bprop_dense_sm90(q, k, v, do, lse, delta, bits, ranges, modes, dq, dk, dv, scale: F32):
    tid = cute.arch.thread_idx()[0]
    block, head, _ = cute.arch.block_idx()
    T = q.shape[0]
    hq, hkv = q.shape[1], k.shape[1]
    kvh = head // (hq // hkv)
    row0 = block * M
    if modes[row0 // 128] == 0:
        smem = cutlass.utils.SmemAllocator()
        sq = smem.allocate_tensor(BF16, cute.make_layout((M, D), stride=(D + 8, 1)), byte_alignment=16)
        sdo = smem.allocate_tensor(BF16, cute.make_layout((M, D), stride=(D + 8, 1)), byte_alignment=16)
        sk = smem.allocate_tensor(BF16, cute.make_layout((N, D), stride=(D + 8, 1)), byte_alignment=16)
        sv = smem.allocate_tensor(BF16, cute.make_layout((N, D), stride=(D + 8, 1)), byte_alignment=16)
        sp = smem.allocate_tensor(BF16, cute.make_layout((M, N), stride=(N + 8, 1)), byte_alignment=16)
        sds = smem.allocate_tensor(BF16, cute.make_layout((3, M, N), stride=(M * (N + 8), N + 8, 1)), byte_alignment=16)
        sg = smem.allocate_tensor(F32, cute.make_layout((N, D), stride=(D + 8, 1)), byte_alignment=16)
        smask = smem.allocate_tensor(I32, cute.make_layout((M,)), byte_alignment=16)
        slse = smem.allocate_tensor(F32, cute.make_layout((M,)), byte_alignment=16)
        sdelta = smem.allocate_tensor(F32, cute.make_layout((M,)), byte_alignment=16)
        sqt = cute.composition(sq, cute.make_layout((D, M), stride=(M, 1)))
        sdot = cute.composition(sdo, cute.make_layout((D, M), stride=(M, 1)))
        skt = cute.composition(sk, cute.make_layout((D, N), stride=(N, 1)))
        spt = cute.composition(sp, cute.make_layout((N, M), stride=(M, 1)))
        sdst = cute.composition(sds[0, None, None], cute.make_layout((N, M), stride=(M, 1)))
        zero = cutlass.Vector.from_elements([BF16(0)] * 8, BF16)
        for it in cutlass.range_constexpr(M * D // (THREADS * 8)):
            flat = (tid + it * THREADS) * 8
            row, dim = flat // D, flat % D
            qr = row0 + row
            load_q, load_do = zero, zero
            if qr < T:
                off = (I64(qr) * hq + head) * D + dim
                load_q = (q.iterator.raw_ptr() + off).load(count=8, alignment=16)
                load_do = (do.iterator.raw_ptr() + off).load(count=8, alignment=16)
            (sq.iterator.raw_ptr() + row * (D + 8) + dim).store(load_q, alignment=16)
            (sdo.iterator.raw_ptr() + row * (D + 8) + dim).store(load_do, alignment=16)
        if tid < M:
            ls, de = -F32.inf, F32(0)
            if row0 + tid < T:
                ls = lse[head, row0 + tid]
                de = delta[row0 + tid, head]
            slse[tid], sdelta[tid] = ls, de
        cute.arch.barrier()
        op = warp.MmaF16BF16Op(BF16, F32, (16, 8, 16))
        mma = cute.make_tiled_mma(op, (2, 2, 1), permutation_mnk=(32, 16, 16))
        thr = mma.get_slice(tid)
        coord = thr.partition_C(cute.make_identity_tensor((M, N)))
        dqcoord = thr.partition_C(cute.make_identity_tensor((M, D)))
        gradpart = thr.partition_C(sg)
        accq = cute.make_rmem_tensor(dqcoord.shape, F32)
        accq.fill(0)
        start = ranges[row0 // 128, 0] // N
        stop = start + ranges[row0 // 128, 1] * (128 // N)
        for nb in cutlass.range(start, stop, unroll=1):
            for it in cutlass.range_constexpr(N * D // (THREADS * 8)):
                flat = (tid + it * THREADS) * 8
                row, dim = flat // D, flat % D
                kr = nb * N + row
                load_k, load_v = zero, zero
                if kr < T:
                    off = (I64(kr) * hkv + kvh) * D + dim
                    load_k = (k.iterator.raw_ptr() + off).load(count=8, alignment=16)
                    load_v = (v.iterator.raw_ptr() + off).load(count=8, alignment=16)
                (sk.iterator.raw_ptr() + row * (D + 8) + dim).store(load_k, alignment=16)
                (sv.iterator.raw_ptr() + row * (D + 8) + dim).store(load_v, alignment=16)
            if tid < M:
                mask = I32(0)
                if row0 + tid < T:
                    mask = bits[row0 + tid, nb]
                smask[tid] = mask
            cute.arch.barrier()
            accs = cute.make_rmem_tensor(coord.shape, F32)
            accdp = cute.make_rmem_tensor(coord.shape, F32)
            accs.fill(0)
            accdp.fill(0)
            for kk in cutlass.range_constexpr(D // 16):
                qa = thr.partition_A(cute.local_tile(sq, (M, 16), (0, kk)))
                ka = thr.partition_B(cute.local_tile(sk, (N, 16), (0, kk)))
                da = thr.partition_A(cute.local_tile(sdo, (M, 16), (0, kk)))
                va = thr.partition_B(cute.local_tile(sv, (N, 16), (0, kk)))
                rq = mma.make_fragment_A(qa)
                rk = mma.make_fragment_B(ka)
                rd = mma.make_fragment_A(da)
                rv = mma.make_fragment_B(va)
                cute.autovec_copy(qa, rq)
                cute.autovec_copy(ka, rk)
                cute.gemm(mma, accs, rq[None, None, 0], rk[None, None, 0], accs)
                cute.autovec_copy(da, rd)
                cute.autovec_copy(va, rv)
                cute.gemm(mma, accdp, rd[None, None, 0], rv[None, None, 0], accdp)
            for i in cutlass.range_constexpr(cute.size(accs)):
                row, col = coord[i]
                p, ds = F32(0), F32(0)
                if ((smask[row] >> col) & 1) != 0 and cute.isfinite(slse[row]):
                    p = cute.exp(accs[i] * scale - slse[row], fastmath=True)
                    ds = p * (accdp[i] - sdelta[row])
                sp[row, col] = p.to(BF16)
                high = ds.to(BF16)
                mid = (ds - high.to(F32)).to(BF16)
                low = (ds - high.to(F32) - mid.to(F32)).to(BF16)
                sds[0, row, col], sds[1, row, col], sds[2, row, col] = high, mid, low
            cute.arch.barrier()
            # Three BF16 parts retain small dQ residuals after cancellation.
            for part in cutlass.range_constexpr(3):
                for kk in cutlass.range_constexpr(N // 16):
                    ds_a = thr.partition_A(cute.local_tile(sds[part, None, None], (M, 16), (0, kk)))
                    kt_b = thr.partition_B(cute.local_tile(skt, (D, 16), (0, kk)))
                    rds = mma.make_fragment_A(ds_a)
                    rkt = mma.make_fragment_B(kt_b)
                    cute.autovec_copy(ds_a, rds)
                    cute.autovec_copy(kt_b, rkt)
                    cute.gemm(mma, accq, rds[None, None, 0], rkt[None, None, 0], accq)
            accg = cute.make_rmem_tensor(gradpart.shape, F32)
            for which in cutlass.range_constexpr(2):
                accg.fill(0)
                for kk in cutlass.range_constexpr(M // 16):
                    a = thr.partition_A(cute.local_tile(spt if which == 0 else sdst, (N, 16), (0, kk)))
                    b = thr.partition_B(cute.local_tile(sdot if which == 0 else sqt, (D, 16), (0, kk)))
                    ra = mma.make_fragment_A(a)
                    rb = mma.make_fragment_B(b)
                    cute.autovec_copy(a, ra)
                    cute.autovec_copy(b, rb)
                    cute.gemm(mma, accg, ra[None, None, 0], rb[None, None, 0], accg)
                cute.autovec_copy(accg, gradpart)
                cute.arch.barrier()
                for it in cutlass.range_constexpr(N * D // (THREADS * 4)):
                    flat = (tid + it * THREADS) * 4
                    row, dim = flat // D, flat % D
                    key = nb * N + row
                    if key < T:
                        off = (I64(key) * hkv + kvh) * D + dim
                        vals = (sg.iterator.raw_ptr() + row * (D + 8) + dim).load(count=4, alignment=16)
                        if cutlass.const_expr(which == 0):
                            cute.arch.atomic_add(dv.iterator + off, vals, scope="gpu")
                        else:
                            cute.arch.atomic_add(dk.iterator + off, vals * scale, scope="gpu")
                cute.arch.barrier()
        for i in cutlass.range_constexpr(cute.size(accq)):
            row, dim = dqcoord[i]
            if row0 + row < T:
                dq[row0 + row, head, dim] = accq[i] * scale


# CuTeDSL 4.8.1/CUDA 13.4 adds local-stack traffic when this component is removed.
frost_qsa_bprop_dense_sm90.set_name_prefix("cudnn", remove_cutlass_symbol=False)
