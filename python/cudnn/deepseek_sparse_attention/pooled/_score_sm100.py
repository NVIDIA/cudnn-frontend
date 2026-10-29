# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged pooled-key scores with head reduction and causal pool lengths."""

import triton
import triton.language as tl


@triton.jit
def cudnn_pooled_scores(
    Q,
    QSCALE,
    WEIGHTS,
    CACHE,
    BLOCK_TABLE,
    CU_QUERY,
    POSITIONS,
    SCORES,
    N_QUERY,
    N_PAGES,
    SCORE_COLS,
    TABLE_COLS,
    Q_ROW,
    Q_HEAD,
    QS_ROW,
    QS_HEAD,
    W_ROW,
    CACHE_PAGE,
    TABLE_ROW,
    CU_STRIDE,
    POS_STRIDE,
    SCORE_ROW,
    POOL_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    POOL_TILE: tl.constexpr,
    BF16_REDUCTION: tl.constexpr,
    FP8_SCALE_FORMAT: tl.constexpr,
):
    request = tl.program_id(2)
    start = tl.load(CU_QUERY + request * CU_STRIDE).to(tl.int64)
    end = tl.load(CU_QUERY + (request + 1) * CU_STRIDE).to(tl.int64)
    query_start = start + tl.program_id(1) * QUERY_TILE
    if query_start < end:
        m = tl.arange(0, QUERY_TILE * 32)
        query = query_start + m // 32
        head = m % 32
        query_valid = (query >= 0) & (query < end) & (query < N_QUERY)
        pool = tl.program_id(0).to(tl.int64) * POOL_TILE + tl.arange(0, POOL_TILE)
        page = tl.load(
            BLOCK_TABLE + request * TABLE_ROW + pool // PAGE_SIZE,
            (pool < SCORE_COLS) & (pool // PAGE_SIZE < TABLE_COLS),
            other=-1,
        ).to(tl.int64)
        pool_valid = (pool < SCORE_COLS) & (page >= 0) & (page < N_PAGES)
        page_base = page * CACHE_PAGE
        slot = pool % PAGE_SIZE
        if DTYPE == "fp4" or (DTYPE == "fp8" and FP8_SCALE_FORMAT == "ue8m0"):
            WIDTH: tl.constexpr = 64 if DTYPE == "fp4" else 128
            FORMAT: tl.constexpr = "e2m1" if DTYPE == "fp4" else "e4m3"
            k = tl.arange(0, WIDTH)
            q = tl.load(Q.to(tl.pointer_type(tl.uint8)) + query[:, None] * Q_ROW + head[:, None] * Q_HEAD + k[None, :], query_valid[:, None], other=0)
            key = tl.load(CACHE + page_base[None, :] + slot[None, :] * WIDTH + k[:, None], pool_valid[None, :], other=0)
            if DTYPE == "fp8":
                key = key.to(tl.float8e4nv, bitcast=True)
                q = q.to(tl.float8e4nv, bitcast=True)
            group = tl.arange(0, 4)
            if QSCALE is None:
                qs = tl.full((QUERY_TILE * 32, 4), 127, tl.uint8)
            else:
                scale_base = QSCALE.to(tl.pointer_type(tl.uint8))
                qs = tl.load(
                    scale_base + query[:, None] * QS_ROW + head[:, None] * QS_HEAD + group[None, :],
                    query_valid[:, None],
                    other=127,
                )
            ks = tl.load(
                CACHE + page_base[:, None] + PAGE_SIZE * WIDTH + slot[:, None] * 4 + group[None, :],
                pool_valid[:, None],
                other=127,
            )
            dots = tl.dot_scaled(key.T, ks, FORMAT, q.T, qs, FORMAT)
        else:
            k = tl.arange(0, 128)
            q = tl.load(Q + query[:, None] * Q_ROW + head[:, None] * Q_HEAD + k[None, :], query_valid[:, None], other=0.0)
            if DTYPE == "fp8":
                pointer = CACHE.to(tl.pointer_type(tl.float8e4nv)) + page_base[None, :] + slot[None, :] * 128 + k[:, None]
            else:
                pointer = CACHE.to(tl.pointer_type(tl.bfloat16)) + page_base[None, :] // 2 + slot[None, :] * 128 + k[:, None]
            key = tl.load(pointer, pool_valid[None, :], other=0.0)
            dots = tl.dot(key.T, q.T)
        weights = tl.load(WEIGHTS + query * W_ROW + head, query_valid, other=0)
        if DTYPE == "fp8" and FP8_SCALE_FORMAT == "fp32":
            scale_ptr = (CACHE + page_base + PAGE_SIZE * 128 + slot * 4).to(tl.pointer_type(tl.float32))
            scales = tl.load(scale_ptr, pool_valid, other=1)
            if BF16_REDUCTION:
                dots *= scales[:, None]
        if BF16_REDUCTION:
            # Match SM100 DeepGEMM: four BF16 FMA chains, then two BF16 adds.
            positive = tl.maximum(dots, 0).to(tl.bfloat16).reshape((POOL_TILE, QUERY_TILE, 32))
            head_weights = tl.broadcast_to(weights[None, :], (POOL_TILE, QUERY_TILE * 32)).reshape((POOL_TILE, QUERY_TILE, 32))
            sums = tl.full((POOL_TILE, QUERY_TILE, 4), 0, tl.bfloat16)
            for i in tl.static_range(8):
                indices = tl.broadcast_to((4 * i + tl.arange(0, 4))[None, None, :], (POOL_TILE, QUERY_TILE, 4))
                values = tl.gather(positive, indices, axis=2)
                ws = tl.gather(head_weights, indices, axis=2)
                sums = tl.inline_asm_elementwise(
                    "fma.rn.bf16 $0, $1, $2, $3;", constraints="=h,h,h,h", args=[values, ws, sums], dtype=tl.bfloat16, is_pure=True, pack=1
                )
            low = tl.broadcast_to(tl.arange(0, 2)[None, None, :], (POOL_TILE, QUERY_TILE, 2))
            pairs = (tl.gather(sums, low, axis=2).to(tl.float32) + tl.gather(sums, low + 2, axis=2).to(tl.float32)).to(tl.bfloat16)
            scores = tl.sum(pairs.to(tl.float32), axis=2).to(tl.bfloat16).to(tl.float32).T
        else:
            per_head = tl.maximum(dots, 0) * weights[None, :]
            scores = tl.sum(per_head.reshape((POOL_TILE, QUERY_TILE, 32)), axis=2).T
            if DTYPE == "fp8" and FP8_SCALE_FORMAT == "fp32":
                scores *= scales[None, :]
        row = query_start + tl.arange(0, QUERY_TILE)
        valid_row = (row >= 0) & (row < end) & (row < N_QUERY)
        position = tl.load(POSITIONS + row * POS_STRIDE, valid_row, other=-1).to(tl.int64)
        lengths = tl.where((position >= 0) & (position < 2147483647), tl.minimum((position + 1) // POOL_SIZE, SCORE_COLS), 0)
        valid = pool_valid[None, :] & (pool[None, :] < lengths[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        tl.store(SCORES + row[:, None] * SCORE_ROW + pool[None, :], scores, valid_row[:, None] & (pool[None, :] < SCORE_COLS))
