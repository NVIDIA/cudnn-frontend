# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper QSA backward with warp MMA and FP32 gradient accumulation."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream
from cutlass.cute.nvgpu import warp as warp_mma
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key
from .bprop_helpers import compile_helpers, fake, launch_backward as launch

D = 256

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.kernel
def frost_qsa_bprop_sm90(
    gDO: cute.Tensor,
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gDelta: cute.Tensor,
    gLSE: cute.Tensor,
    gIds: cute.Tensor,
    gCounts: cute.Tensor,
    gLo: cute.Tensor,
    gHi: cute.Tensor,
    gBlockFirst: cute.Tensor,
    gBlockLast: cute.Tensor,
    scale: cutlass.Float32,
    gDQ: cute.Tensor,
    gDKacc: cute.Tensor,
    gDVacc: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    hq = cute.size(gQ, mode=[1])
    hkv = cute.size(gK, mode=[1])
    qid = bid // hkv
    kvh = bid - qid * hkv
    group = hq // hkv
    smem = cutlass.utils.SmemAllocator()

    sQ = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((16, D), stride=(D + 8, 1)), byte_alignment=16)
    sDO = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((16, D), stride=(D + 8, 1)), byte_alignment=16)
    sQt = cute.composition(sQ, cute.make_layout((D, 16), stride=(16, 1)))
    sDOt = cute.composition(sDO, cute.make_layout((D, 16), stride=(16, 1)))
    sK = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((16, D), stride=(D + 8, 1)), byte_alignment=16)
    sV = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((16, D), stride=(D + 8, 1)), byte_alignment=16)
    sKt = cute.composition(sK, cute.make_layout((D, 16), stride=(16, 1)))
    sScorePart = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4, 16, 16), stride=(16 * 17, 17, 1)), byte_alignment=16)
    sDPPart = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4, 16, 16), stride=(16 * 17, 17, 1)), byte_alignment=16)
    sP = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((16, 16), stride=(17, 1)), byte_alignment=16)
    sPt = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((16, 16), stride=(17, 1)), byte_alignment=16)
    sDS = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((3, 16, 16), stride=(16 * 17, 17, 1)), byte_alignment=16)
    sDSt = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((16, 16), stride=(17, 1)), byte_alignment=16)
    sGrad = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16, D), stride=(D + 8, 1)), byte_alignment=16)
    sDelta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), byte_alignment=16)
    sKey = smem.allocate_tensor(cutlass.Int32, cute.make_layout((16,)), byte_alignment=16)

    for it in cutlass.range_constexpr(32):
        flat = tid + it * 128
        row = flat // D
        d = flat - row * D
        qv = cutlass.BFloat16(0.0)
        dov = cutlass.BFloat16(0.0)
        if row < group:
            qv = gQ[(qid, kvh * group + row, d)]
            dov = gDO[(qid, kvh * group + row, d)]
        sQ[(row, d)] = qv
        sDO[(row, d)] = dov

    if tid < group:
        sDelta[tid] = gDelta[qid, kvh * group + tid]
    cute.arch.barrier()

    op = warp_mma.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    mmaS = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 16, 16))
    mmaG = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 64, 16))
    thS = mmaS.get_slice(lane)
    thG = mmaG.get_slice(lane)
    qW = cute.local_tile(sQ, (16, 64), (0, warp))
    doW = cute.local_tile(sDO, (16, 64), (0, warp))
    kW = cute.local_tile(sK, (16, 64), (0, warp))
    vW = cute.local_tile(sV, (16, 64), (0, warp))
    ktW = cute.local_tile(sKt, (64, 16), (warp, 0))
    qtW = cute.local_tile(sQt, (64, 16), (warp, 0))
    dotW = cute.local_tile(sDOt, (64, 16), (warp, 0))
    dqW = cute.local_tile(cute.make_identity_tensor((16, D)), (16, 64), (0, warp))
    gradW = cute.local_tile(sGrad, (16, 64), (0, warp))

    def make_a(mma, th, tensor):
        part = th.partition_A(tensor)
        frag = mma.make_fragment_A(part)
        cute.autovec_copy(part, frag)
        return frag

    def make_b(mma, th, tensor):
        part = th.partition_B(tensor)
        frag = mma.make_fragment_B(part)
        cute.autovec_copy(part, frag)
        return frag

    rQ = make_a(mmaS, thS, qW)
    rDO = make_a(mmaS, thS, doW)
    accDQ = cute.make_rmem_tensor(thG.partition_C(dqW).shape, cutlass.Float32)
    accDQ.fill(0.0)
    count = gCounts[qid]
    rounds = cute.ceil_div(count, 4)
    rr = cutlass.Int32(0)
    while rr < rounds:
        if tid < 16:
            pos = rr * 4 + tid // 4
            key = cutlass.Int32(-1)
            if pos < count:
                b = gIds[(qid, pos)]
                key = gBlockFirst[b] + tid % 4
                if key > gBlockLast[b] or key < gLo[qid] or key > gHi[qid]:
                    key = cutlass.Int32(-1)
            sKey[tid] = key
        cute.arch.barrier()
        for it in cutlass.range_constexpr(32):
            flat = tid + it * 128
            row = flat // D
            d = flat - row * D
            key = sKey[row]
            kval = cutlass.BFloat16(0.0)
            vval = cutlass.BFloat16(0.0)
            if key >= 0:
                kval = gK[(key, kvh, d)]
                vval = gV[(key, kvh, d)]
            sK[(row, d)] = kval
            sV[(row, d)] = vval
        cute.arch.barrier()

        rK = make_b(mmaS, thS, kW)
        rV = make_b(mmaS, thS, vW)
        partS = thS.partition_C(sScorePart[warp, None, None])
        partDP = thS.partition_C(sDPPart[warp, None, None])
        accS = cute.make_rmem_tensor(partS.shape, cutlass.Float32)
        accDP = cute.make_rmem_tensor(partDP.shape, cutlass.Float32)
        accS.fill(0.0)
        accDP.fill(0.0)
        for kk in cutlass.range(cute.size(rQ, mode=[2]), unroll_full=True):
            cute.gemm(mmaS, accS, rQ[None, None, kk], rK[None, None, kk], accS)
            cute.gemm(mmaS, accDP, rDO[None, None, kk], rV[None, None, kk], accDP)
        cute.autovec_copy(accS, partS)
        cute.autovec_copy(accDP, partDP)
        cute.arch.barrier()

        for rep in cutlass.range_constexpr(2):
            x = tid + rep * 128
            m = x // 16
            n = x - m * 16
            score = sScorePart[(0, m, n)] + sScorePart[(1, m, n)] + sScorePart[(2, m, n)] + sScorePart[(3, m, n)]
            dp = sDPPart[(0, m, n)] + sDPPart[(1, m, n)] + sDPPart[(2, m, n)] + sDPPart[(3, m, n)]
            p = cutlass.Float32(0.0)
            ds = cutlass.Float32(0.0)
            if m < group and sKey[n] >= 0:
                stat = gLSE[(kvh * group + m, qid)]
                if cute.isfinite(stat):
                    p = cute.exp(score * scale - stat, fastmath=True)
                    ds = p * (dp - sDelta[m])
            sP[(m, n)] = p.to(cutlass.BFloat16)
            sPt[(n, m)] = p.to(cutlass.BFloat16)
            high = ds.to(cutlass.BFloat16)
            middle = (ds - high.to(cutlass.Float32)).to(cutlass.BFloat16)
            low = ((ds - high.to(cutlass.Float32)) - middle.to(cutlass.Float32)).to(cutlass.BFloat16)
            sDS[(0, m, n)], sDS[(1, m, n)], sDS[(2, m, n)] = high, middle, low
            sDSt[(n, m)] = ds.to(cutlass.BFloat16)
        cute.arch.barrier()

        rKt = make_b(mmaG, thG, ktW)
        # Three BF16 parts preserve cancellation in dQ.
        for component in cutlass.range_constexpr(3):
            rDS = make_a(mmaG, thG, sDS[component, None, None])
            for kk in cutlass.range(cute.size(rDS, mode=[2]), unroll_full=True):
                cute.gemm(mmaG, accDQ, rDS[None, None, kk], rKt[None, None, kk], accDQ)

        # dV first, then reuse the same accumulator and shared output tile for dK.
        rPt = make_a(mmaG, thG, sPt)
        rDOt = make_b(mmaG, thG, dotW)
        partG = thG.partition_C(gradW)
        accG = cute.make_rmem_tensor(partG.shape, cutlass.Float32)
        accG.fill(0.0)
        for kk in cutlass.range(cute.size(rPt, mode=[2]), unroll_full=True):
            cute.gemm(mmaG, accG, rPt[None, None, kk], rDOt[None, None, kk], accG)
        cute.autovec_copy(accG, partG)
        cute.arch.barrier()
        for it in cutlass.range_constexpr(8):
            flat = (tid + it * 128) * 4
            row = flat // D
            d = flat - row * D
            key = sKey[row]
            if key >= 0:
                off = (Int64(key) * hkv + kvh) * D + d
                vk = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
                for z in cutlass.range_constexpr(4):
                    vk[z] = sGrad[(row, d + z)]
                cute.arch.atomic_add(gDVacc.iterator + off, vk.load(), scope="gpu")
        cute.arch.barrier()
        rDSt = make_a(mmaG, thG, sDSt)
        rQt = make_b(mmaG, thG, qtW)
        accG.fill(0.0)
        for kk in cutlass.range(cute.size(rDSt, mode=[2]), unroll_full=True):
            cute.gemm(mmaG, accG, rDSt[None, None, kk], rQt[None, None, kk], accG)
        cute.autovec_copy(accG, partG)
        cute.arch.barrier()
        for it in cutlass.range_constexpr(8):
            flat = (tid + it * 128) * 4
            row = flat // D
            d = flat - row * D
            key = sKey[row]
            if key >= 0:
                off = (Int64(key) * hkv + kvh) * D + d
                vk = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
                for z in cutlass.range_constexpr(4):
                    vk[z] = sGrad[(row, d + z)] * scale
                cute.arch.atomic_add(gDKacc.iterator + off, vk.load(), scope="gpu")
        cute.arch.barrier()
        rr = rr + 1

    dq_coords = thG.partition_C(dqW)
    for i in cutlass.range_constexpr(cute.size(accDQ)):
        row, dim = dq_coords[i]
        if row < group:
            gDQ[qid, kvh * group + row, dim] = accDQ[i] * scale


frost_qsa_bprop_sm90.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(q, k, v, dout, lse, delta, dq, dk, dv, ids, counts, lo, hi, first, last, scale: Float32, blocks: Int32, stream: cuda.CUstream):
    frost_qsa_bprop_sm90(dout, q, k, v, delta, lse, ids, counts, lo, hi, first, last, scale, dq, dk, dv).launch(
        grid=(q.shape[0] * k.shape[1], 1, 1), block=(128, 1, 1), stream=stream
    )


def compile_kernel(dtype, hq, hkv, arch):
    key = template_key(globals(), dict(dtype=str(dtype), hq=hq, hkv=hkv, arch=arch), "compile_kernel")
    total, capacity, blocks = (cute.sym_int64() for _ in range(3))
    q, kv = fake(dtype, (total, hq, 256)), fake(dtype, (total, hkv, 256))
    aq, ak = fake(Float32, (total, hq, 256)), fake(Float32, (total, hkv, 256))
    rows, spans = fake(Int32, (total,), 4), fake(Int32, (blocks,), 4)
    compiled = compile_helpers(hq, hkv, arch)
    compiled["main"] = compile_cached(
        _host,
        q,
        kv,
        kv,
        q,
        fake(Float32, (hq, total), 4),
        fake(Float32, (total, hq)),
        aq,
        ak,
        ak,
        fake(Int32, (total, capacity), 4),
        rows,
        rows,
        rows,
        spans,
        spans,
        Float32(1),
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=key,
        symbol="qsa_backward_sm90",
    )
    return compiled
