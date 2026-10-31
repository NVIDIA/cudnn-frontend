# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper QSA with warp MMA and an FP32 value path for extreme ranges."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key
from cutlass.cute.nvgpu import cpasync, warp
from .bprop_helpers import fake

TILE_N = 64

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.jit
def _token(ids, counts, first, last, lo, hi, row, slot, total):
    result = Int32(-1)
    entry = slot // 4
    if entry < counts[row]:
        block = ids[Int64(row), Int64(entry)]
        if block >= 0 and block < first.shape[0]:
            key = first[block] + slot % 4
            if key >= 0 and key < total and key >= lo[row] and key <= hi[row] and key <= last[block]:
                result = key
    return result


@cute.jit
def _aligned_ptr(tensor, offset):
    return cute.make_ptr(BFloat16, (tensor.iterator + offset).toint(), cute.AddressSpace.gmem, assumed_align=16)


def _acc_mn(acc):
    layout = cute.make_layout(acc.layout.shape)
    mn = cute.make_layout(
        ((layout.shape[0][1], layout.shape[1]), (layout.shape[0][0], layout.shape[2])),
        stride=((layout.stride[0][1], layout.stride[1]), (layout.stride[0][0], layout.stride[2])),
    )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))


def _quad_max(value):
    value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, 2))
    return cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, 1))


def _quad_sum(value):
    value += cute.arch.shuffle_sync_bfly(value, 2)
    return value + cute.arch.shuffle_sync_bfly(value, 1)


@cute.kernel
def frost_qsa_fprop_sm90(
    q,
    k,
    v,
    out,
    out32,
    lse,
    ids,
    counts,
    lo,
    hi,
    first,
    last,
    scale: Float32,
    modes,
    q_layout,
    kv_layout,
    p_layout,
    copier,
    mma,
    GROUP: cutlass.Constexpr,
    Storage: cutlass.Constexpr,
):
    row, kvhead, _ = cute.arch.block_idx()
    active = True
    if cutlass.const_expr(modes is not None):
        active = modes[row // 128] != 0
    if active:
        tid = cute.arch.thread_idx()[0]
        warp_id, lane = tid // 32, tid % 32
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        sq, sk, sv = storage.q.get_tensor(q_layout), storage.k.get_tensor(kv_layout), storage.v.get_tensor(kv_layout)
        sp = storage.p.get_tensor(p_layout)
        pf = storage.pf.get_tensor(cute.make_layout((16, TILE_N), stride=(TILE_N, 1)))
        maxima = storage.maxima.get_tensor(cute.make_layout((16, 4), stride=(4, 1)))
        sums = storage.sums.get_tensor(cute.make_layout((16, 4), stride=(4, 1)))
        keys = storage.keys.get_tensor(cute.make_layout(TILE_N))
        large = storage.large.get_tensor(cute.make_layout(1))
        vt = cute.composition(sv, cute.make_layout((256, TILE_N), stride=(TILE_N, 1)))
        q_global = cute.make_tensor(_aligned_ptr(q, (Int64(row) * q.shape[1] + kvhead * GROUP) * 256), cute.make_layout((GROUP, 256), stride=(256, 1)))
        gq = cute.local_tile(q_global, (16, 256), (0, 0))
        copy_thread = copier.get_slice(tid)
        q_src, q_dst = copy_thread.partition_S(gq), copy_thread.partition_D(sq)
        q_coord = copy_thread.partition_S(cute.make_identity_tensor((16, 256)))
        k_dst, v_dst = copy_thread.partition_D(sk), copy_thread.partition_D(sv)
        kv_coord = copy_thread.partition_S(cute.make_identity_tensor((TILE_N, 256)))
        for m in cutlass.range_constexpr(cute.size(q_dst.shape[1])):
            if q_coord[0, m, 0][0] < GROUP:
                cute.copy(copier, q_src[None, m, None], q_dst[None, m, None])
            else:
                q_dst[None, m, None].fill(0)
        cute.arch.cp_async_commit_group()
        thread_mma = mma.get_slice(tid)
        rq = thread_mma.make_fragment_A(thread_mma.partition_A(sq))
        rk = thread_mma.make_fragment_B(thread_mma.partition_B(sk))
        rp = thread_mma.make_fragment_A(thread_mma.partition_A(sp))
        rv = thread_mma.make_fragment_B(thread_mma.partition_B(vt))
        load = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
        load_t = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)
        copy_a = cute.make_tiled_copy_A(load, mma).get_slice(tid)
        copy_b = cute.make_tiled_copy_B(load, mma).get_slice(tid)
        copy_v = cute.make_tiled_copy_B(load_t, mma).get_slice(tid)
        qs, qr = copy_a.partition_S(sq), copy_a.retile(rq)
        ks, kr = copy_b.partition_S(sk), copy_b.retile(rk)
        ps, pr = copy_a.partition_S(sp), copy_a.retile(rp)
        vs, vr = copy_v.partition_S(vt), copy_v.retile(rv)
        coords = _acc_mn(thread_mma.partition_C(cute.make_identity_tensor((16, TILE_N))))
        ocoords = _acc_mn(thread_mma.partition_C(cute.make_identity_tensor((16, 256))))
        nrows, ncols = cutlass.const_expr(cute.size(coords.shape[0])), cutlass.const_expr(cute.size(coords.shape[1]))
        acc = cute.make_rmem_tensor(thread_mma.partition_shape_C((16, 256)), Float32)
        acc.fill(0)
        maxima_reg, sums_reg = cute.make_rmem_tensor(nrows, Float32), cute.make_rmem_tensor(nrows, Float32)
        maxima_reg.fill(-float("inf"))
        sums_reg.fill(0)
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        cute.copy(copy_a, qs, qr)
        for tile in cutlass.range(cute.ceil_div(counts[row], TILE_N // 4), unroll=1):
            if tid < TILE_N:
                keys[tid] = _token(ids, counts, first, last, lo, hi, row, tile * TILE_N + tid, Int32(q.shape[0]))
            if tid == 0:
                large[0] = Int32(0)
            cute.arch.barrier()
            for n in cutlass.range_constexpr(cute.size(k_dst.shape[1])):
                key = keys[kv_coord[0, n, 0][0]]
                for dim in cutlass.range_constexpr(cute.size(k_dst.shape[2])):
                    if key >= 0:
                        offset = cute.assume((Int64(key) * k.shape[1] + kvhead) * 256 + kv_coord[0, n, dim][1], divby=8)
                        src_k = cute.make_tensor(_aligned_ptr(k, offset), cute.make_layout(k_dst[None, n, dim].shape))
                        src_v = cute.make_tensor(_aligned_ptr(v, offset), cute.make_layout(v_dst[None, n, dim].shape))
                        cute.copy(copier, src_k, k_dst[None, n, dim])
                        cute.copy(copier, src_v, v_dst[None, n, dim])
                    else:
                        k_dst[None, n, dim].fill(0)
                        v_dst[None, n, dim].fill(0)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            extreme = Int32(0)
            for n in cutlass.range_constexpr(cute.size(v_dst.shape[1])):
                for dim in cutlass.range_constexpr(cute.size(v_dst.shape[2])):
                    values = v_dst[None, n, dim].load().to(Float32)
                    for x in cutlass.range_constexpr(cute.size(v_dst[None, n, dim].shape)):
                        if cute.math.abs(values[x]) > 1e30:
                            extreme = Int32(1)
            if extreme:
                cute.arch.atomic_or(large.iterator, Int32(1), scope="cta")
            score = cute.make_rmem_tensor(thread_mma.partition_shape_C((16, TILE_N)), Float32)
            score.fill(0)
            for step in cutlass.range_constexpr(cute.size(qs.shape[2])):
                cute.copy(copy_b, ks[None, None, step], kr[None, None, step])
                cute.gemm(mma, score, rq[None, None, step], rk[None, None, step], score)
            smn, omn = _acc_mn(score), _acc_mn(acc)
            for r in cutlass.range_constexpr(nrows):
                for c in cutlass.range_constexpr(ncols):
                    if keys[coords[r, c][1]] >= 0:
                        smn[r, c] *= scale
                    else:
                        smn[r, c] = Float32(-float("inf"))
                value = _quad_max(smn[r, None].load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0))
                if lane % 4 == 0:
                    maxima[coords[r, 0][0], warp_id] = value
            cute.arch.barrier()
            for r in cutlass.range_constexpr(nrows):
                maximum = maxima_reg[r]
                for w in cutlass.range_constexpr(4):
                    maximum = cute.math.max(maximum, maxima[coords[r, 0][0], w])
                safe = Float32(0) if maximum == -Float32.inf else maximum
                probabilities = cute.math.exp(smn[r, None].load() - safe, fastmath=True)
                smn[r, None] = probabilities
                value = _quad_sum(probabilities.reduce(cute.ReductionOp.ADD, Float32(0), 0))
                if lane % 4 == 0:
                    sums[coords[r, 0][0], warp_id] = value
            cute.arch.barrier()
            for r in cutlass.range_constexpr(nrows):
                maximum = maxima_reg[r]
                for w in cutlass.range_constexpr(4):
                    maximum = cute.math.max(maximum, maxima[coords[r, 0][0], w])
                correction = Float32(0)
                if maxima_reg[r] > -Float32.inf:
                    correction = cute.math.exp(maxima_reg[r] - maximum, fastmath=True)
                previous = sums_reg[r] * correction
                total = previous
                for w in cutlass.range_constexpr(4):
                    total += sums[coords[r, 0][0], w]
                inv = Float32(1) / total if total > 0 else Float32(0)
                omn[r, None] = omn[r, None].load() * (previous * inv)
                smn[r, None] = smn[r, None].load() * inv
                sums_reg[r], maxima_reg[r] = total, maximum
            pfrag = cute.make_fragment_like(score, BFloat16)
            pfrag.store(score.load().to(BFloat16))
            cute.autovec_copy(pfrag, thread_mma.partition_C(sp))
            cute.autovec_copy(score, thread_mma.partition_C(pf))
            cute.arch.barrier()
            if large[0] == 0:
                for step in cutlass.range_constexpr(cute.size(ps.shape[2])):
                    cute.copy(copy_a, ps[None, None, step], pr[None, None, step])
                    cute.copy(copy_v, vs[None, None, step], vr[None, None, step])
                    cute.gemm(mma, acc, rp[None, None, step], rv[None, None, step], acc)
            else:
                for n in cutlass.range(TILE_N, unroll=1):
                    for r in cutlass.range_constexpr(nrows):
                        probability = pf[ocoords[r, 0][0], n]
                        for c in cutlass.range_constexpr(cute.size(ocoords.shape[1])):
                            omn[r, c] += probability * sv[n, ocoords[r, c][1]].to(Float32)
            cute.arch.barrier()
        omn = _acc_mn(acc)
        for r in cutlass.range_constexpr(nrows):
            head = ocoords[r, 0][0]
            if head < GROUP:
                if warp_id == 0 and lane % 4 == 0:
                    value = Float32(-float("inf"))
                    if sums_reg[r] > 0:
                        value = maxima_reg[r] + cute.math.log(sums_reg[r], fastmath=True)
                    lse[kvhead * GROUP + head, row] = value
                for c in cutlass.range_constexpr(cute.size(ocoords.shape[1])):
                    dim = ocoords[r, c][1]
                    out[row, kvhead * GROUP + head, dim] = omn[r, c].to(BFloat16)
                    out32[row, kvhead * GROUP + head, dim] = omn[r, c]


frost_qsa_fprop_sm90.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(q, k, v, out, out32, lse, ids, counts, lo, hi, first, last, scale: Float32, stream: cuda.CUstream, modes=None):
    atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1)))
    q_layout = cute.tile_to_shape(atom, (16, 256), (0, 1))
    kv_layout = cute.tile_to_shape(atom, (TILE_N, 256), (0, 1))
    p_atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
    p_layout = cute.tile_to_shape(p_atom, (16, TILE_N), (0, 1))

    @cute.struct
    class Storage:
        q: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(q_layout)], 1024]
        k: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv_layout)], 1024]
        v: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv_layout)], 1024]
        p: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(p_layout)], 1024]
        pf: cute.struct.Align[cute.struct.MemRange[Float32, 16 * TILE_N], 16]
        maxima: cute.struct.Align[cute.struct.MemRange[Float32, 16 * 4], 16]
        sums: cute.struct.Align[cute.struct.MemRange[Float32, 16 * 4], 16]
        keys: cute.struct.Align[cute.struct.MemRange[Int32, TILE_N], 16]
        large: cute.struct.Align[cute.struct.MemRange[Int32, 1], 4]

    copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
    copier = cute.make_tiled_copy_tv(copy_atom, cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, 4, 1), permutation_mnk=(16, 32, 16))
    frost_qsa_fprop_sm90(
        q, k, v, out, out32, lse, ids, counts, lo, hi, first, last, scale, modes, q_layout, kv_layout, p_layout, copier, mma, q.shape[1] // k.shape[1], Storage
    ).launch(grid=(q.shape[0], k.shape[1], 1), block=(128, 1, 1), stream=stream)


def compile_kernel(dtype, hq, hkv, arch):
    key = template_key(globals(), dict(dtype=str(dtype), hq=hq, hkv=hkv, arch=arch), "compile_kernel")
    total, capacity, blocks = (cute.sym_int64() for _ in range(3))
    q, kv = fake(dtype, (total, hq, 256)), fake(dtype, (total, hkv, 256))
    rows, spans = fake(Int32, (total,), 4), fake(Int32, (blocks,), 4)
    return compile_cached(
        _host,
        q,
        kv,
        kv,
        q,
        fake(Float32, (total, hq, 256)),
        fake(Float32, (hq, total), 4),
        fake(Int32, (total, capacity), 4),
        rows,
        rows,
        rows,
        spans,
        spans,
        Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=key,
        symbol="qsa_forward_sm90",
    )


def launch(compiled, q, k, v, out, out32, lse, ids, counts, lo, hi, first, last, scale, stream):
    compiled(q, k, v, out, out32, lse, ids, counts, lo, hi, first, last, scale, stream)
