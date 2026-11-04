# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper indexer scores in the DSA compact candidate layout."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_fake_tensor,
)
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.frost.tile_dsl.mma import mma_m16n8k16_f32, mma_m16n8k32_f32
from cudnn.frost.tile_dsl.pointwise import opaque_f32_zero

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.jit
def _word(ptr, offset: Int64, ELEMENTS: cutlass.Constexpr[int]):
    return (ptr + offset).load(count=ELEMENTS, alignment=4 // ELEMENTS).bitcast(Int32)[0]


@cute.jit
def _prefix(n: Int64, ratio: Int32):
    whole = n // Int64(ratio)
    tail = n - whole * Int64(ratio)
    return Int64(ratio) * whole * (whole - 1) // 2 + whole * (tail + 1)


@cute.kernel
def frost_qsa_select_score_sm90(
    q,
    k,
    w,
    cu,
    pos,
    offsets,
    cand,
    scale: Float32,
    ratio: Int32,
    HEADS: cutlass.Constexpr[int],
):
    lane = Int32(cute.arch.thread_idx()[0] % 32)
    warp = Int32(cute.arch.thread_idx()[0] // 32)
    tile, key_tile, batch = cute.arch.block_idx()
    group, col = lane // 4, lane % 4
    per_tile = cutlass.const_expr(max(1, 16 // HEADS))
    m_tiles = cutlass.const_expr(max(1, HEADS // 16))
    fp8 = cutlass.const_expr(q.element_type == cutlass.Float8E4M3FN)
    elements = cutlass.const_expr(4 if fp8 else 2)
    steps = cutlass.const_expr(4 if fp8 else 8)
    first, end = cu[batch], cu[batch + 1]
    start = Int32(tile) * per_tile
    origin = Int32(0)
    if cutlass.const_expr(pos is not None):
        origin = pos[batch]
    key0 = Int32(key_tile) * 128 + warp * 32
    max_visible = cute.math.min(
        (origin + cute.math.min(start + per_tile, end - first)) // ratio,
        Int32(k.shape[1]),
    )
    if first + start < end and key0 < max_visible:
        pq, pk = q.iterator.raw_ptr(), k.iterator.raw_ptr()
        a_words, weights = [], []
        for m in cutlass.range_constexpr(m_tiles):
            halves, wh = [], []
            for half in cutlass.range_constexpr(2):
                mr = m * 16 + group + half * 8
                token = first + start + mr // HEADS
                head = mr % HEADS
                words = []
                for step in cutlass.range_constexpr(steps):
                    for part in cutlass.range_constexpr(2):
                        d = step * (elements * 8) + col * elements + part * (elements * 4)
                        word = Int32(0)
                        if token < end:
                            word = _word(
                                pq,
                                Int64(token) * q.stride[0] + Int64(head) * q.stride[1] + d,
                                elements,
                            )
                        words.append(word)
                halves.append(words)
                weight = Float32(1)
                if cutlass.const_expr(w is not None):
                    weight = Float32(0)
                    if token < end:
                        weight = w[token, head].to(Float32)
                wh.append(weight)
            a_words.append(halves)
            weights.append(wh)
        for n in cutlass.range_constexpr(4):
            key = key0 + n * 8 + group
            accum = [[opaque_f32_zero() for _ in range(4)] for _ in range(m_tiles)]
            for step in cutlass.range_constexpr(steps):
                kw = []
                for part in cutlass.range_constexpr(2):
                    d = step * (elements * 8) + col * elements + part * (elements * 4)
                    word = Int32(0)
                    if key < max_visible:
                        word = _word(pk, (Int64(batch) * k.shape[1] + key) * 128 + d, elements)
                    kw.append(word)
                for m in cutlass.range_constexpr(m_tiles):
                    lo, hi = a_words[m][0], a_words[m][1]
                    if cutlass.const_expr(fp8):
                        c0, c1, c2, c3 = mma_m16n8k32_f32(
                            lo[2 * step],
                            hi[2 * step],
                            lo[2 * step + 1],
                            hi[2 * step + 1],
                            *kw,
                            *accum[m],
                            ab_dtype=q.element_type,
                        )
                    else:
                        c0, c1, c2, c3 = mma_m16n8k16_f32(
                            lo[2 * step],
                            hi[2 * step],
                            lo[2 * step + 1],
                            hi[2 * step + 1],
                            *kw,
                            *accum[m],
                            ab_dtype=q.element_type,
                        )
                    accum[m] = [c0, c1, c2, c3]
            for half in cutlass.range_constexpr(2 if HEADS < 16 else 1):
                mr = group + half * 8
                row = start + mr // HEADS
                visible = cute.math.min((origin + row + 1) // ratio, Int32(k.shape[1]))
                base = offsets[batch] + _prefix(Int64(origin) + row, ratio) - _prefix(Int64(origin), ratio)
                for side in cutlass.range_constexpr(2):
                    value = Float32(0)
                    for m in cutlass.range_constexpr(m_tiles):
                        if cutlass.const_expr(HEADS < 16):
                            value += cute.math.max(accum[m][half * 2 + side], 0.0) * weights[m][half]
                        else:
                            value += cute.math.max(accum[m][side], 0.0) * weights[m][0]
                            value += cute.math.max(accum[m][2 + side], 0.0) * weights[m][1]
                    for shift in cutlass.range_constexpr((min(HEADS, 8)).bit_length() - 1):
                        value += cute.arch.shuffle_sync_bfly(value, 4 << shift)
                    output_key = key0 + n * 8 + col * 2 + side
                    if group % min(HEADS, 8) == 0 and first + row < end and output_key < visible:
                        cand[base + output_key] = value * scale


frost_qsa_select_score_sm90.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    q,
    k,
    w,
    cu,
    pos,
    offsets,
    cand,
    scale: Float32,
    ratio: Int32,
    max_q: Int32,
    stream: cuda.CUstream,
    HEADS: cutlass.Constexpr[int],
):
    frost_qsa_select_score_sm90(q, k, w, cu, pos, offsets, cand, scale, ratio, HEADS).launch(
        grid=(
            cute.ceil_div(max_q, max(1, 16 // HEADS)),
            cute.ceil_div(k.shape[1], 128),
            k.shape[0],
        ),
        block=(128, 1, 1),
        stream=stream,
    )


def _compact(dtype, shape, align=4):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def compile_score(heads, dtype, weight_dtype, has_pos):
    total, batch, blocks, prefixes = (cute.sym_int64() for _ in range(4))
    q = make_fake_tensor(
        dtype,
        (total, heads, 128),
        (cute.sym_int64(), cute.sym_int64(), 1),
        assumed_align=dtype.width // 8,
    )
    w = (
        None
        if weight_dtype is None
        else make_fake_tensor(
            weight_dtype,
            (total, heads),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=weight_dtype.width // 8,
        )
    )
    key = template_key(
        globals(),
        dict(heads=heads, dtype=str(dtype), weights=str(weight_dtype), has_pos=has_pos),
        "compile_score",
    )
    return compile_cached(
        _host,
        q,
        _compact(dtype, (batch, blocks, 128), dtype.width // 8),
        w,
        _compact(Int32, (prefixes,)),
        _compact(Int32, (batch,)) if has_pos else None,
        _compact(Int64, (prefixes,), 8),
        _compact(Float32, (cute.sym_int64(),)),
        Float32(1),
        Int32(4),
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        heads,
        options="--enable-tvm-ffi --gpu-arch=sm_90a",
        cache_key=key,
        symbol="qsa_select_score_sm90",
    )


def compile_topk(topk, ratio, has_pos, deterministic):
    from cudnn.deepseek_sparse_attention.indexer_top_k.compress_top_k_sm100 import (
        CompressTopkStage2,
    )

    total, batch, prefixes = (cute.sym_int64() for _ in range(3))
    kernel = CompressTopkStage2(
        topk=topk,
        ratio=ratio,
        block_threads=512,
        is_varlen=True,
        deterministic=deterministic,
    )
    key = template_key(
        globals(),
        dict(topk=topk, ratio=ratio, has_pos=has_pos, deterministic=deterministic),
        "compile_topk",
    )

    def output(dtype):
        return make_fake_tensor(dtype, (total, 1, topk), (cute.sym_int64(), topk, 1), assumed_align=4)

    return compile_cached(
        kernel,
        _compact(Float32, (cute.sym_int64(),), 16),
        output(Int32),
        output(Float32),
        Int32(1),
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        _compact(Int32, (prefixes,)),
        _compact(Int32, (prefixes,)),
        _compact(Int64, (prefixes,), 8),
        _compact(Int32, (batch,)) if has_pos else None,
        None,
        None,
        options="--enable-tvm-ffi --gpu-arch=sm_90a",
        cache_key=key,
        symbol="qsa_select_topk_sm90",
    )
