# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-geometry HCA score, normalization, packing and bounded reductions."""

import triton
import triton.language as tl


@triton.jit
def normalize(Out, Do, Lse, Sink, Workspace, SinkPart, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    out = tl.load(Out + rows[:, None] * D + dims[None, :], rows[:, None] < N, 0).to(tl.float32)
    do = tl.load(Do + rows[:, None] * D + dims[None, :], rows[:, None] < N, 0).to(tl.float32)
    delta = -tl.sum(out * do, axis=1)
    lse = tl.load(Lse + rows, rows < N, 0)
    sink = tl.load(Sink + rows % H)
    dominant_sink = sink == float("inf")
    finite_sink = tl.where(dominant_sink, 0.0, sink)
    maximum = tl.maximum(lse, finite_sink)
    full_lse = maximum + tl.log(tl.exp(lse - maximum) + tl.exp(finite_sink - maximum))
    # The +inf sink has probability one; avoid inf-inf in both folds.
    full_lse = tl.where(dominant_sink, float("inf"), full_lse)
    sink_log_probability = tl.where(dominant_sink, 0.0, finite_sink - full_lse)
    tl.store(Workspace + rows, delta, rows < N)
    tl.store(Workspace + N + rows, -full_lse * 1.4426950408889634, rows < N)
    tl.store(SinkPart + rows, delta * tl.exp(sink_log_probability), rows < N)


@triton.jit
def valid_pair(rows, keys, H: tl.constexpr, START: tl.constexpr, NCOMP: tl.constexpr, GROUP_TOKENS: tl.constexpr):
    token = rows // H
    within = token % GROUP_TOKENS
    width: tl.constexpr = 128 + GROUP_TOKENS
    global_key = START + (token // GROUP_TOKENS) * GROUP_TOKENS - 128 + keys
    window = (keys < width) & (keys >= within + 1) & (keys <= within + 128) & (global_key >= 0)
    compressed = (keys >= width) & (keys < width + NCOMP) & (keys - width < (START + token + 1) // 128)
    return window | compressed


@triton.jit
def separate_tma_scores(
    Q,
    Do,
    Packed,
    Workspace,
    P,
    Ds,
    N: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    START: tl.constexpr,
    NCOMP: tl.constexpr,
    SCALE: tl.constexpr,
    GROUP_TOKENS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SMS: tl.constexpr,
):
    key_tiles: tl.constexpr = triton.cdiv(K, BN)
    tiles: tl.constexpr = triton.cdiv(N, BM) * key_tiles
    # Keep load and store counters independent for warp specialization.
    store_tile = tl.program_id(0) - SMS
    for tile in tl.range(tl.program_id(0), tiles, SMS, flatten=False, warp_specialize=True, disallow_acc_multi_buffer=False):
        query_block = tile // key_tiles
        key_block = tile % key_tiles
        group = query_block * BM // (GROUP_TOKENS * H)
        score = tl.full((BM, BN), 0, tl.float32)
        dp = tl.full((BM, BN), 0, tl.float32)
        for block in range(triton.cdiv(D, BK)):
            q = Q.load([query_block * BM, block * BK])
            do = Do.load([query_block * BM, block * BK])
            kv = Packed.load([group * K + key_block * BN, block * BK])
            score = tl.dot(q, tl.trans(kv), score)
            dp = tl.dot(do, tl.trans(kv), dp)
        store_tile += SMS
        store_query = store_tile // key_tiles
        store_key = store_tile % key_tiles
        rows = store_query * BM + tl.arange(0, BM)
        keys = store_key * BN + tl.arange(0, BN)
        delta = tl.load(Workspace + rows, rows < N, 0)
        lse = tl.load(Workspace + N + rows, rows < N, 0)
        p = tl.exp2(score * (SCALE * 1.4426950408889634) + lse[:, None])
        p = tl.where(valid_pair(rows[:, None], keys[None, :], H, START, NCOMP, GROUP_TOKENS), p, 0.0)
        ds = p * (dp + delta[:, None]) * SCALE
        P.store([store_query * BM, store_key * BN], p.to(tl.bfloat16))
        Ds.store([store_query * BM, store_key * BN], ds.to(tl.bfloat16))


@triton.jit
def reduce_sink(Partial, SinkGrad, TOKENS: tl.constexpr, H: tl.constexpr):
    heads = tl.program_id(0) * 4 + tl.arange(0, 4)
    tokens = tl.arange(0, min(triton.next_power_of_2(TOKENS), 4096))
    result = tl.full((4,), 0, tl.float32)
    for start in range(triton.cdiv(TOKENS, 4096)):
        rows = start * 4096 + tokens
        values = tl.load(Partial + rows[:, None] * H + heads[None, :], rows[:, None] < TOKENS, 0)
        result += tl.sum(values, axis=0)
    tl.store(SinkGrad + heads, result)


@triton.jit
def reduce_local_keys(Dk, Dv, Dkv, K: tl.constexpr, G: tl.constexpr, D: tl.constexpr, GROUP_TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) // (D // BLOCK)
    dims = tl.program_id(0) % (D // BLOCK) * BLOCK + tl.arange(0, BLOCK)
    last = tl.minimum(G - 1, row // GROUP_TOKENS)
    group = last - tl.arange(0, triton.next_power_of_2(128 // GROUP_TOKENS + 1))
    key = row - group * GROUP_TOKENS
    valid = (group >= 0) & (key >= 0) & (key < 128 + GROUP_TOKENS)
    ptr = (group[:, None] * K + key[:, None]) * D + dims[None, :]
    dk = tl.load(Dk + ptr, valid[:, None], 0)
    dv = tl.load(Dv + ptr, valid[:, None], 0)
    tl.store(Dkv + row * D + dims, tl.sum(dk + dv, axis=0))


@triton.jit
def reduce_local_rows(
    Dk,
    Dv,
    Dkv,
    K: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    GROUP_TOKENS: tl.constexpr,
    LOCAL_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, dim = x // D, x % D
    last = tl.minimum(G - 1, row // GROUP_TOKENS)
    result = tl.full((BLOCK,), 0, tl.float32)
    for previous in tl.static_range(128 // GROUP_TOKENS + 1):
        group = last - previous
        key = row - group * GROUP_TOKENS
        valid = (row < LOCAL_ROWS) & (group >= 0) & (key >= 0) & (key < 128 + GROUP_TOKENS)
        ptr = (group * K + key) * D + dim
        dk = tl.load(Dk + ptr, valid, 0)
        dv = tl.load(Dv + ptr, valid, 0)
        result += dk + dv
    tl.store(Dkv + x, result, row < LOCAL_ROWS)


@triton.jit
def pack_rank_major_keys(
    KV,
    Packed,
    K: tl.constexpr,
    NCOMP: tl.constexpr,
    GROUP_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    START: tl.constexpr,
    GROUPS_PER_RANK: tl.constexpr,
    LOCAL_ROWS: tl.constexpr,
):
    group = tl.program_id(1)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    key, dim = x // 512, x % 512
    width: tl.constexpr = 128 + GROUP_TOKENS
    logical = tl.maximum(key - width, 0)
    owner = logical // GROUPS_PER_RANK
    physical = owner * (GROUPS_PER_RANK + 1) + logical % GROUPS_PER_RANK + (owner > 0).to(tl.int32)
    source = tl.where(key < width, group * GROUP_TOKENS + key, LOCAL_ROWS + physical)
    # Zero keys unreachable by every query in this group before any dot/GEMM.
    window = (key > 0) & (key < width) & (START + group * GROUP_TOKENS - 128 + key >= 0)
    compressed = (key >= width) & (key < width + NCOMP) & (logical < (START + (group + 1) * GROUP_TOKENS) // 128)
    value = tl.load(KV + source * 512 + dim, (x < K * 512) & (window | compressed), 0)
    tl.store(Packed + group * K * 512 + x, value, x < K * 512)


@triton.jit
def reduce_rank_major_keys(
    Dk,
    Dv,
    Dkv,
    NCOMP: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    GROUP_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_RANK: tl.constexpr,
    LOCAL_ROWS: tl.constexpr,
):
    physical = tl.program_id(0) // (512 // BLOCK)
    dims = tl.program_id(0) % (512 // BLOCK) * BLOCK + tl.arange(0, BLOCK)
    owner, slot = physical // (GROUPS_PER_RANK + 1), physical % (GROUPS_PER_RANK + 1)
    logical = owner * GROUPS_PER_RANK + slot - (owner > 0).to(tl.int32)
    canonical = ((owner == 0) & (slot < GROUPS_PER_RANK)) | ((owner > 0) & (slot > 0))
    if canonical & (logical < NCOMP):
        group = tl.arange(0, triton.next_power_of_2(G))
        ptr = (group[:, None] * K + 128 + GROUP_TOKENS + logical) * 512 + dims[None, :]
        dk = tl.load(Dk + ptr, group[:, None] < G, 0)
        dv = tl.load(Dv + ptr, group[:, None] < G, 0)
        result = tl.sum(dk + dv, axis=0)
    else:
        result = tl.full((BLOCK,), 0, tl.float32)
    tl.store(Dkv + (LOCAL_ROWS + physical) * 512 + dims, result)
