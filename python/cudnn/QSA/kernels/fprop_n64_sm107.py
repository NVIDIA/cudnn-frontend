# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""QSA forward with 64 keys and 14 warps per CTA on SM107.

Adapted from Donglin Yang's native107_forward_n64 implementation.

One CTA owns one query/KV-head. Eight gather warps retain the native
K0,K1,V0,K2,V1 three-stage ring. Four converter warps own all 16 padded head
columns through the canonical M64 T2R map. An issuer overlaps the next score
with current softmax/PV; a Q producer retains native clipped-head TMA. The
online recurrence and original output/LSE contracts are unchanged algebraically,
but 64-key grouping changes reduction association and is validated against the common QSA reference.
"""

import hashlib
from pathlib import Path

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import arith
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass.experimental import primitives as prims
from cuda.bindings import driver as cuda
from . import _fprop_n64_ops as ops
from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OPS_SOURCE_DIGEST = hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest()

I32, I64, F32, BF16 = cutlass.Int32, cutlass.Int64, cutlass.Float32, cutlass.BFloat16
D = 256
THREADS, TILE_N, KV_STAGES, S_STAGES = 448, 64, 3, 2
OFF_Q, OFF_P, OFF_KV = 0, 8192, 12288
KV_BYTES, P_BYTES, SHARED_BYTES = 32768, 2048, 110592
LOG2E, LN2 = 1.4426950408889634, 0.6931471805599453


@cute.jit
def choose_i(predicate, yes, no):
    return I32(arith.select(predicate.ir_value(), I32(yes).ir_value(), I32(no).ir_value()))


@cute.jit
def choose_f(predicate, yes, no):
    return F32(arith.select(predicate.ir_value(), F32(yes).ir_value(), F32(no).ir_value()))


@cute.jit
def wait(bar, phase):
    while not prims.mbarrier_try_wait_parity(bar, phase):
        pass


@cute.jit
def span(ids, first, last, query, entry, count):
    begin, end = I32(-1), I32(-1)
    if entry < count:
        block = I32(ids[(I64(query), I64(entry))])
        if (block >= 0) & (block < first.shape[0]):
            begin, end = I32(first[block]), I32(last[block])
    return begin, end


@cute.jit
def gather(
    source,
    contiguous_source,
    raw,
    full,
    empty,
    ordinal,
    tile,
    query,
    head,
    ids,
    first,
    last,
    count,
    low,
    high,
    total,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
    lane = cute.arch.thread_idx()[0] % 32
    begin, end = I32(-1), I32(-1)
    if lane < 2:
        begin, end = span(ids, first, last, query, tile * 16 + warp * 2 + lane, count)
    starts, ends = [], []
    for block in cutlass.range_constexpr(2):
        starts.append(I32(prims.shfl_sync(0xFFFFFFFF, begin, block, 31, kind=prims.Shfl.IDX)))
        ends.append(I32(prims.shfl_sync(0xFFFFFFFF, end, block, 31, kind=prims.Shfl.IDX)))
    contiguous = (starts[0] >= 0) & (starts[0] >= low) & (starts[0] + 7 <= high) & (starts[0] + 7 < total)
    for block in cutlass.range_constexpr(2):
        contiguous = contiguous & (starts[block] == starts[0] + block * 4) & (ends[block] >= starts[block] + 3)
    slot, phase = ordinal % 3, (ordinal // 3) & 1
    wait(empty.subview(slot), phase ^ 1)
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(full.subview(slot), 4096)
    if contiguous:
        for box in cutlass.range_constexpr(4):
            offset = OFF_KV + slot * KV_BYTES + box * 8192 + warp * 1024
            ops.load_eight(contiguous_source, raw, offset, starts[0], head, box, full.subview(slot))
    else:
        for block in cutlass.range_constexpr(2):
            rows = cute.make_rmem_tensor(4, I32)
            for r in cutlass.range_constexpr(4):
                token = starts[block] + r
                valid = (starts[block] >= 0) & (token <= ends[block]) & (token >= low) & (token <= high) & (token < total)
                rows[r] = choose_i(valid, token, -1)
            for box in cutlass.range_constexpr(4):
                offset = OFF_KV + slot * KV_BYTES + box * 8192 + (warp * 2 + block) * 512
                ops.gather_four(source, raw, offset, head, box, rows, full.subview(slot))
    return ordinal + 1


@cute.jit
def issue_score(raw, tmem, kv_full, kv_empty, s_full, s_empty, ordinal, tile):
    ks, kp = ordinal % 3, (ordinal // 3) & 1
    ss, sp = tile % 2, (tile // 2) & 1
    wait(kv_full.subview(ks), kp)
    wait(s_empty.subview(ss), sp ^ 1)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    for k16 in cutlass.range_constexpr(16):
        ops.score(raw, OFF_KV + ks * KV_BYTES, tmem, ss * 16, k16, k16 > 0)
    if prims.elect_sync():
        prims.tcgen05_commit(s_full.subview(ss), group=prims.CTAGroup.CTA_1)
        prims.tcgen05_commit(kv_empty.subview(ks), group=prims.CTAGroup.CTA_1)


@cute.jit
def reduce_max(value):
    for shift in cutlass.range_constexpr(3):
        other = F32(prims.shfl_sync(0xFFFFFFFF, value, 4 << shift, 31, kind=prims.Shfl.BFLY))
        value = cute.math.max(value, other, ftz=True)
    return value


@cute.jit
def reduce_sum(value):
    for shift in cutlass.range_constexpr(3):
        value = value + F32(prims.shfl_sync(0xFFFFFFFF, value, 4 << shift, 31, kind=prims.Shfl.BFLY))
    return value


@cute.kernel
def frost_qsa_fprop_n64_sm107(
    q: cutlass.GridConstant[tmap.TensorMap],
    k: cutlass.GridConstant[tmap.TensorMap],
    v: cutlass.GridConstant[tmap.TensorMap],
    k_contiguous: cutlass.GridConstant[tmap.TensorMap],
    v_contiguous: cutlass.GridConstant[tmap.TensorMap],
    out: cute.Tensor,
    out_fp32: cute.Tensor,
    lse: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    scale: F32,
):
    query, head, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tid % 32
    count, low, high = I32(counts[query]), I32(lo[query]), I32(hi[query])
    tiles = cute.math.max((count + 15) // 16, 1)
    storage = cutlass.Array(I32, SHARED_BYTES // 4, alignment=1024, space=cutlass.AddressSpace.smem)
    red = cutlass.Array(F32, 192, alignment=16, space=cutlass.AddressSpace.smem)
    mailbox = cutlass.Array(I32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    qf = cutlass.Array(I64, 1, alignment=16, space=cutlass.AddressSpace.smem)
    kf = cutlass.Array(I64, 3, alignment=16, space=cutlass.AddressSpace.smem)
    ke = cutlass.Array(I64, 3, alignment=16, space=cutlass.AddressSpace.smem)
    sf = cutlass.Array(I64, 2, alignment=16, space=cutlass.AddressSpace.smem)
    se = cutlass.Array(I64, 2, alignment=16, space=cutlass.AddressSpace.smem)
    pf = cutlass.Array(I64, 2, alignment=16, space=cutlass.AddressSpace.smem)
    vf = cutlass.Array(I64, 2, alignment=16, space=cutlass.AddressSpace.smem)
    release = cutlass.Array(I64, 1, alignment=16, space=cutlass.AddressSpace.smem)
    raw = storage.data_ptr()
    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(qf, 1)
            prims.mbarrier_init(release, 128)
            for stage in cutlass.range_constexpr(3):
                prims.mbarrier_init(kf.subview(stage), 8)
                prims.mbarrier_init(ke.subview(stage), 1)
            for stage in cutlass.range_constexpr(2):
                prims.mbarrier_init(sf.subview(stage), 1)
                prims.mbarrier_init(se.subview(stage), 128)
                prims.mbarrier_init(pf.subview(stage), 128)
                prims.mbarrier_init(vf.subview(stage), 1)
    if warp < 4:
        zero = cutlass.Vector.from_elements([I32(0) for _ in range(4)], I32)
        for chunk in cutlass.range(tid, 8192 // 16, 128, unroll=1):
            (raw + chunk * 4).store(zero, alignment=16)
    if warp == 12:
        cute.arch.alloc_tmem(
            64,
            cute.make_ptr(
                I32,
                I64(mailbox.data_ptr().toint()),
                cutlass.AddressSpace.smem,
                assumed_align=4,
            ),
            is_two_cta=False,
            arch="sm_107",
        )
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
    prims.fence_mbarrier_init()
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(0)
    tmem = mailbox.load()
    if warp < 4:
        neg_inf = F32(float("-inf"))
        maximum = cutlass.Vector.from_elements([neg_inf for _ in range(3)], F32)
        norm = cutlass.Vector.from_elements([F32(0) for _ in range(6)], F32)
        row0 = 16 * warp + lane // 4
        for tile in cutlass.range(tiles, unroll=1):
            slot, phase = tile % 2, (tile // 2) & 1
            wait(sf.subview(slot), phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            score = ops.read_score(tmem, slot * 16, tid)
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.mbarrier_arrive(se.subview(slot))
            valid = []
            for r in cutlass.range_constexpr(2):
                row = row0 + 8 * r
                begin, end = span(ids, first, last, query, tile * 16 + row // 4, count)
                token = begin + row % 4
                valid.append((begin >= 0) & (token <= end) & (token >= low) & (token <= high) & (token < out.shape[0]))
            scaled = []
            for j in cutlass.range_constexpr(3):
                for r in cutlass.range_constexpr(2):
                    scaled.append(
                        choose_f(
                            valid[r],
                            F32(score[2 * j + r]) * scale * F32(LOG2E),
                            neg_inf,
                        )
                    )
                mx = reduce_max(cute.math.max(scaled[2 * j], scaled[2 * j + 1], ftz=True))
                if lane < 4:
                    red[slot * 64 + warp * 16 + lane + j * 4] = mx
            prims.barrier_cta_sync(barrier_id=1, thread_count=128)
            next_max, next_norm, alpha = [], [], []
            for j in cutlass.range_constexpr(3):
                h = lane % 4 + j * 4
                mx = F32(maximum[j])
                for w in cutlass.range_constexpr(4):
                    mx = cute.math.max(mx, F32(red[slot * 64 + w * 16 + h]), ftz=True)
                safe_old = choose_f(F32(maximum[j]) == neg_inf, F32(0), maximum[j])
                safe_new = choose_f(mx == neg_inf, F32(0), mx)
                a = cute.math.exp2(cute.math.min(safe_old - safe_new, F32(0), ftz=True), fastmath=True)
                next_max.append(mx)
                alpha.append(a)
                for r in cutlass.range_constexpr(2):
                    probability = cute.math.exp2(scaled[2 * j + r] - safe_new, fastmath=True)
                    next_norm.append(F32(norm[2 * j + r]) * a + probability)
                    ops.store_p(raw, OFF_P + slot * P_BYTES, row0 + r * 8, h, probability)
            for r in cutlass.range_constexpr(2):
                ops.store_p(raw, OFF_P + slot * P_BYTES, row0 + r * 8, lane % 4 + 12, F32(0))
            maximum = cutlass.Vector.from_elements(next_max, F32)
            norm = cutlass.Vector.from_elements(next_norm, F32)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            if tile > 0:
                wait(vf.subview((tile - 1) % 2), ((tile - 1) // 2) & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                all_alpha = []
                rescale = I32(0)
                for h in cutlass.range_constexpr(12):
                    a = F32(prims.shfl_sync(0xFFFFFFFF, alpha[h // 4], h % 4, 31, kind=prims.Shfl.IDX))
                    all_alpha.append(a)
                    rescale = rescale | I32(a != F32(1))
                if rescale != 0:
                    for half in cutlass.range_constexpr(2):
                        for page in cutlass.range_constexpr(2):
                            column = 32 + half * 16 + page * 8
                            old = ops.read_output(tmem, column, tid)
                            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                            values = []
                            for j in cutlass.range_constexpr(8):
                                if cutlass.const_expr(page * 8 + j < 12):
                                    values.append(F32(old[j]) * all_alpha[page * 8 + j])
                                else:
                                    values.append(F32(old[j]))
                            ops.write_output(
                                tmem,
                                column,
                                tid,
                                cutlass.Vector.from_elements(values, F32),
                            )
                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.mbarrier_arrive(pf.subview(slot))
        wait(vf.subview((tiles - 1) % 2), ((tiles - 1) // 2) & 1)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for j in cutlass.range_constexpr(3):
            total = reduce_sum(F32(norm[2 * j]) + F32(norm[2 * j + 1]))
            if lane < 4:
                red[128 + warp * 16 + lane + j * 4] = total
        prims.barrier_cta_sync(barrier_id=1, thread_count=128)
        inverses, is_empty, statistics = [], [], []
        for j in cutlass.range_constexpr(3):
            h = lane % 4 + j * 4
            denominator = F32(0)
            for w in cutlass.range_constexpr(4):
                denominator = denominator + F32(red[128 + w * 16 + h])
            dead = denominator == F32(0)
            safe = choose_f(dead, F32(1), denominator)
            inverses.append(F32(1) / safe)
            is_empty.append(I32(dead))
            statistics.append(
                choose_f(
                    dead,
                    neg_inf,
                    F32(maximum[j]) * F32(LN2) + cute.math.log(safe, fastmath=True),
                )
            )
        for half in cutlass.range_constexpr(2):
            for page in cutlass.range_constexpr(2):
                values = ops.read_output(tmem, 32 + half * 16 + page * 8, tid)
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                for j in cutlass.range_constexpr(8):
                    if cutlass.const_expr(page * 8 + j < 12):
                        h = page * 8 + j
                        inverse = F32(
                            prims.shfl_sync(
                                0xFFFFFFFF,
                                inverses[h // 4],
                                h % 4,
                                31,
                                kind=prims.Shfl.IDX,
                            )
                        )
                        dead = I32(
                            prims.shfl_sync(
                                0xFFFFFFFF,
                                is_empty[h // 4],
                                h % 4,
                                31,
                                kind=prims.Shfl.IDX,
                            )
                        )
                        result = choose_f(dead != 0, F32(0), F32(values[j]) * inverse)
                        coord = (I64(query), I64(head * 12 + h), I64(tid + half * 128))
                        out_fp32[coord] = result
                        out[coord] = result.to(BF16)
                        if cutlass.const_expr(half == 0):
                            stat = F32(
                                prims.shfl_sync(
                                    0xFFFFFFFF,
                                    statistics[h // 4],
                                    h % 4,
                                    31,
                                    kind=prims.Shfl.IDX,
                                )
                            )
                            if tid == h:
                                lse[(I64(head * 12 + h), I64(query))] = stat
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.mbarrier_arrive(release)
    elif warp < 12:
        ordinal = I32(0)
        ordinal = gather(
            k,
            k_contiguous,
            raw,
            kf,
            ke,
            ordinal,
            0,
            query,
            head,
            ids,
            first,
            last,
            count,
            low,
            high,
            out.shape[0],
        )
        for tile in cutlass.range(tiles, unroll=1):
            if tile + 1 < tiles:
                ordinal = gather(
                    k,
                    k_contiguous,
                    raw,
                    kf,
                    ke,
                    ordinal,
                    tile + 1,
                    query,
                    head,
                    ids,
                    first,
                    last,
                    count,
                    low,
                    high,
                    out.shape[0],
                )
            ordinal = gather(
                v,
                v_contiguous,
                raw,
                kf,
                ke,
                ordinal,
                tile,
                query,
                head,
                ids,
                first,
                last,
                count,
                low,
                high,
                out.shape[0],
            )
        for stage in cutlass.range_constexpr(3):
            wait(ke.subview(ordinal % 3), ((ordinal // 3) & 1) ^ 1)
            ordinal = ordinal + 1
    elif warp == 12:
        wait(qf, 0)
        ordinal = I32(0)
        issue_score(raw, tmem, kf, ke, sf, se, ordinal, 0)
        ordinal = ordinal + 1
        for tile in cutlass.range(tiles, unroll=1):
            if tile + 1 < tiles:
                issue_score(raw, tmem, kf, ke, sf, se, ordinal, tile + 1)
                ordinal = ordinal + 1
            slot, phase = ordinal % 3, (ordinal // 3) & 1
            ps, pp = tile % 2, (tile // 2) & 1
            wait(kf.subview(slot), phase)
            wait(pf.subview(ps), pp)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for half in cutlass.range_constexpr(2):
                for k16 in cutlass.range_constexpr(4):
                    ops.value(
                        raw,
                        OFF_KV + slot * KV_BYTES,
                        OFF_P + ps * P_BYTES,
                        tmem,
                        32 + half * 16,
                        half,
                        k16,
                        (tile > 0) | (k16 > 0),
                    )
            if prims.elect_sync():
                prims.tcgen05_commit(vf.subview(ps), group=prims.CTAGroup.CTA_1)
                prims.tcgen05_commit(ke.subview(slot), group=prims.CTAGroup.CTA_1)
            ordinal = ordinal + 1
        wait(release, 0)
    else:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(qf, 12 * 256 * 2)
        for slab in cutlass.range_constexpr(4):
            ops.load_q(q, raw, slab * 2048, query, head, slab, qf)
        wait(qf, 0)
    prims.barrier_cta_sync(0)
    if warp == 12:
        cute.arch.dealloc_tmem(
            cute.make_ptr(F32, tmem, cutlass.AddressSpace.tmem, assumed_align=16),
            64,
            is_two_cta=False,
            arch="sm_107",
        )


frost_qsa_fprop_n64_sm107.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _fprop_host(q, k, v, out, out_fp32, lse, ids, counts, lo, hi, first, last, mode, scale: F32, stream: cuda.CUstream, arch: cutlass.Constexpr[int]):
    group_size = cutlass.const_expr(q.shape[1] // k.shape[1])
    q_desc = tmap.create_tensor_map_tiled(
        global_address=q.iterator.toint(),
        dtype=q.element_type,
        global_dims=(q.shape[2], q.shape[1], q.shape[0]),
        global_strides=(I64(q.stride[1]) // 8, I64(q.stride[0]) // 8),
        box_dims=(64, group_size, 1),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    k_desc = tmap.create_tensor_map_tiled(
        global_address=k.iterator.toint(),
        dtype=k.element_type,
        global_dims=(k.shape[1] * D, k.shape[0]),
        global_strides=(I64(k.stride[0]) // 8,),
        box_dims=(64, 1),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    v_desc = tmap.create_tensor_map_tiled(
        global_address=v.iterator.toint(),
        dtype=v.element_type,
        global_dims=(v.shape[1] * D, v.shape[0]),
        global_strides=(I64(v.stride[0]) // 8,),
        box_dims=(64, 1),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    k_contiguous_desc = tmap.create_tensor_map_tiled(
        global_address=k.iterator.toint(),
        dtype=k.element_type,
        global_dims=(k.shape[1] * D, k.shape[0]),
        global_strides=(I64(k.stride[0]) // 8,),
        box_dims=(64, 8),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    v_contiguous_desc = tmap.create_tensor_map_tiled(
        global_address=v.iterator.toint(),
        dtype=v.element_type,
        global_dims=(v.shape[1] * D, v.shape[0]),
        global_strides=(I64(v.stride[0]) // 8,),
        box_dims=(64, 8),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    frost_qsa_fprop_n64_sm107(
        q_desc,
        k_desc,
        v_desc,
        k_contiguous_desc,
        v_contiguous_desc,
        out,
        out_fp32,
        lse,
        ids,
        counts,
        lo,
        hi,
        first,
        last,
        scale,
    ).launch(grid=(q.shape[0], k.shape[1], 1), block=(THREADS, 1, 1), stream=stream)


def compile_kernel(dtype, query_heads, kv_heads, arch, gated=False):
    from .fprop_sm100 import _compile_args, FROST_SOURCE_DIGEST as args_source_digest

    if arch != 107 or dtype != cutlass.BFloat16 or query_heads != 12 * kv_heads or gated:
        raise ValueError("64-key QSA forward requires SM107, BF16 and 12 query heads per KV head, without gated tiles")
    # Keep imported callables out of the serializable specialization key.
    key = template_key(
        globals(),
        {"dtype": dtype, "query_heads": query_heads, "kv_heads": kv_heads, "arch": arch, "ops_digest": OPS_SOURCE_DIGEST, "args_digest": args_source_digest},
        "compile_kernel",
    )
    return compile_cached(
        _fprop_host,
        *_compile_args(dtype, query_heads, kv_heads, False),
        arch,
        cache_key=key,
        symbol="qsa_forward_n64",
        options="--enable-tvm-ffi --gpu-arch=sm_107a",
    )


def launch(compiled, q, k, v, out, out_fp32, lse, ids, counts, lo, hi, first, last, scale, stream, mode=None):
    if mode is not None:
        raise ValueError("64-key QSA forward does not accept gated tiles")
    compiled(q, k, v, out, out_fp32, lse, ids, counts, lo, hi, first, last, None, F32(scale), stream)
