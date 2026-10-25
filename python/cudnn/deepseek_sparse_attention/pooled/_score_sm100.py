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
        if DTYPE == "fp4":
            k = tl.arange(0, 64)
            q = tl.load(Q + query[:, None] * Q_ROW + head[:, None] * Q_HEAD + k[None, :], query_valid[:, None], other=0)
            key = tl.load(CACHE + page_base[None, :] + slot[None, :] * 64 + k[:, None], pool_valid[None, :], other=0)
            group = tl.arange(0, 4)
            scale_base = QSCALE.to(tl.pointer_type(tl.uint8))
            qs = tl.load(
                scale_base + query[:, None] * QS_ROW + head[:, None] * QS_HEAD + group[None, :],
                query_valid[:, None],
                other=127,
            )
            ks = tl.load(
                CACHE + page_base[:, None] + PAGE_SIZE * 64 + slot[:, None] * 4 + group[None, :],
                pool_valid[:, None],
                other=127,
            )
            dots = tl.dot_scaled(key.T, ks, "e2m1", q.T, qs, "e2m1")
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
        per_head = tl.maximum(dots, 0) * weights[None, :]
        scores = tl.sum(per_head.reshape((POOL_TILE, QUERY_TILE, 32)), axis=2).T
        if DTYPE == "fp8":
            scale_ptr = (CACHE + page_base + PAGE_SIZE * 128 + slot * 4).to(tl.pointer_type(tl.float32))
            scales = tl.load(scale_ptr, pool_valid, other=1)
            scores *= scales[None, :]
        row = query_start + tl.arange(0, QUERY_TILE)
        valid_row = (row >= 0) & (row < end) & (row < N_QUERY)
        position = tl.load(POSITIONS + row * POS_STRIDE, valid_row, other=-1).to(tl.int64)
        lengths = tl.where((position >= 0) & (position < 2147483647), tl.minimum((position + 1) // POOL_SIZE, SCORE_COLS), 0)
        valid = pool_valid[None, :] & (pool[None, :] < lengths[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        tl.store(SCORES + row[:, None] * SCORE_ROW + pool[None, :], scores, valid_row[:, None] & (pool[None, :] < SCORE_COLS))
