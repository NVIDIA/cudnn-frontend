# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_kv_upcast`` -- the e4m3 paged KV pool, GATHERED by a block list and CAST to bf16 into a dense per-item scratch.

The fp8 KV-cache path of the block's sparse attention (Qwen Sparse Attention over a ``--kv-cache-dtype fp8`` pool) in its
first form: the bf16 sparse core is NOT taught fp8 operands -- a separate bandwidth kernel dequantizes exactly the rows the
core would gather, lays them out DENSELY in list order, and the bf16 d256 DECODE tile then runs over the scratch as a plain
padded dense SDPA (one "batch" per item, the item's query-head group as the packed Q rows, ``seq_len_kv = kv_len_t``).
No sparse code touches fp8 anywhere; the in-kernel upcast is a later, measured decision.

WHAT IT COMPUTES.  A work ITEM is one (query token ``t``, KV head ``h``) pair -- ``item = (b * S_q + s) * H_kv + h`` for
token ``s`` of sequence ``b`` -- and the kernel writes, per item and per operand (K then V), the item's VISIBLE key rows
compacted to the front of a ``[ROWS, D]`` bf16 slab plus ``kv_len_t[item]`` = the number of rows written; every row from
``kv_len_t`` to the end of the slab is ZERO (the default; ``zero_fill="tile"`` zeroes only up to the next 128-row tile, the
dense core's KV tile, since the core reads nothing past it).  Dequantization is ``bf16(fp32(e4m3) * scale)`` with
``k_scale`` / ``v_scale`` fp32 device scalars MULTIPLIED IN (vLLM folds ``k_scale`` into the softmax scale and ``v_scale``
into the output normalizer; here the core sees plain values), one fp32 rounding -- bitwise ``(pool.float() * scale).bfloat16()``.

The visible set is the sparse core's, row for row (``gated_block_qsa_reference.qsa_visible_mask``): the rows sit at the
END of the sequence (bottom-right: token ``s`` of sequence ``b`` with KV length ``L_b`` is at position ``pos = L_b - S_q + s``),
``n_vis = min(pos + 1, L_b)`` (``<= 0`` = a DEAD item), and with ``anchor`` = the row's own position (per-token lists
``block_ids [B, S_q, top_k]``) or the sequence's STEP-0 position ``L_b - S_q`` (``list_per_sequence``: ONE list ``[B, top_k]``
shared by the ``S_q <= 4`` rows -- the MTP shared-list semantics) and ``n_anchor = min(anchor + 1, L_b)``:

* ``count = min(block_lens[row], top_k, floor(n_anchor / 4))`` entries are READ (``block_lens`` absent: the last two terms);
* entry ``i < count`` with id ``blk >= 0`` contributes the rows ``[4 blk, 4 blk + 4) ∩ [0, n_vis)`` -- a complete block, or
  the first ``cnt < 4`` rows of a block that straddles the visible range; a ``-1`` entry (anywhere inside the count) and an
  entry at or past ``floor((n_vis + 3) / 4)`` contribute NOTHING and no row -- the compaction skips them (a dense re-read of
  zero-filled rows would score 0, not -inf, and absorb softmax mass: the rows are dropped, not zeroed);
* then the OPEN TAIL: the rows ``[4 floor(n_anchor / 4), n_vis)`` -- 0..3 tokens per row, up to two blocks under a shared list
  (the block completed between step 0 and the row stays visible, the row's own token included);
* in LIST ORDER (the entries first, then the tail): the order of the keys is immaterial to the softmax up to the fp32
  association the dense core itself chooses; a listed OPEN block or a repeated id is attended TWICE here as in the core (the
  list contract forbids both; nothing here deduplicates).

Rows come from the e4m3 PAGE POOLS ``[num_pages, H_kv, page_size, D]`` (HND compact or NHD storage through the strides, the
cache-write kernel's own view) through ``block_table [B, max_pages]`` int32: ``page_size % 4 == 0`` so a 4-token block never
straddles a page (ONE table lookup per block); a ``-1`` table entry, a page index past the table or a page at or past the pool
reads as ZERO rows that are still counted (the sparse core's convention for a dead page).  Lengths, counts and the table
are DEVICE values (Rule 3): the host validates FORM only.

GEOMETRY (every derived count comes from these constants; ``_validate_geometry`` pins the couplings):

    BLOCK_SIZE 4, D 256, ROWS 2176 = UNITS_PER_ITEM 17 x ROW_TILE 128 (the dense core's KV tile)
    slots per item = TOPK_MAX 512 + TAIL_SLOTS 2 = 514 <= 17 x SLOTS_PER_UNIT 32 = 544
    max rows written = 4 x 512 + 6 (a shared list's two tail blocks) = 2054 <= ROWS
    work UNIT (item, c in 0..16): the slots [32 c, 32 c + 32) of the item's list + the zero fill of the slab rows
                                  [max(kv_len_t, 128 c), 128 (c + 1)) -- 17 units cover the 2176 rows exactly
    CTA = 4 warps x 32 lanes; each warp copies 8 slots; a lane moves 16 B of e4m3 per load (one 16-element column chunk of
          one row: 16 lanes per 256-B row, 2 rows per warp-load) -> ``ld.global.v4.b32`` (LDG.E.128) in, 2 x ``st.global.v4.b32``
          (STG.E.128) out per operand per row pair; the grid is PERSISTENT: min(units, SMs x CTAS_PER_SM) CTAs stride the units

The compaction offsets are a per-item prefix sum over the 514 slots; every warp of a unit re-derives it from the list
(17 coalesced 128-B loads of ids, L1 / L2-resident -- ~1 % of the unit's traffic) and keeps it in registers: NO shared memory,
NO mbarrier, so the two pre-development tables of a FROST kernel (barriers, SMEM buffers) are EMPTY by construction, and
nothing synchronizes across warps.  Per slot each lane needs (block id, row count, destination row) of its warp's slot:
three ``shfl.idx`` from the holder of that slot.

Traffic per item per operand: ``cnt_total x 256 B`` of e4m3 read + ``ROWS x 512 B`` of bf16 written (the zero fill included)
+ the 2 KiB list + the table words; the roofline term is HBM BYTES (per token at 2 KV heads: ~2.1 MB read + ~4.5 MB written).
A perf claim is made only from an exclusive Rubin perf GPU (the M4 measurement of the step record).

Rule 7: the module imports the DSL at import time like its neighbours, but ``compile_qsa_kv_upcast`` gates the DSL's
``sm_107a`` target BEFORE ``cute.compile`` so a too-old DSL reads as a version decline; the tests skip below the floor.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental import primitives as nvvm

from cudnn.frost.device import compute_capability, current_device, multiprocessor_count
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4

BLOCK_SIZE = 4
D = 256
ROW_TILE = 128  # the dense d256 decode tile's KV tile: the zero fill is reasoned in these units
UNITS_PER_ITEM = 17
ROWS = UNITS_PER_ITEM * ROW_TILE  # 2176
SLOTS_PER_UNIT = 32
TAIL_SLOTS = 2  # a shared list's tail spans at most two blocks (S_q <= 4: the step-0 tail start to the last row's position)
TOPK_MIN, TOPK_MAX = 4, 512
MTP_MAX_S_Q = 4
THREADS = 128
WARPS = THREADS // 32
SLOTS_PER_WARP = SLOTS_PER_UNIT // WARPS  # 8
LANES_PER_ROW = D // 16  # 16 lanes x 16 e4m3 elements = one 256-B row
ROWS_PER_WARP_LOAD = 32 // LANES_PER_ROW  # 2
CTAS_PER_SM = 8
ZERO_FILL_MODES = ("full", "tile")

_POOL_DTYPE = torch.float8_e4m3fn
_OUT_DTYPE = torch.bfloat16
_FAKE_STREAM = None


def _validate_geometry() -> None:
    """The couplings between the module constants, raised (never asserted): a change to one that breaks another would
    otherwise ship as a silent read past a slab or a slot nobody copies."""
    if UNITS_PER_ITEM * ROW_TILE != ROWS:
        raise ValueError(f"ROWS {ROWS} must be UNITS_PER_ITEM {UNITS_PER_ITEM} x ROW_TILE {ROW_TILE}")
    if TOPK_MAX + TAIL_SLOTS > UNITS_PER_ITEM * SLOTS_PER_UNIT:
        raise ValueError(f"{TOPK_MAX} + {TAIL_SLOTS} slots per item exceed the {UNITS_PER_ITEM} x {SLOTS_PER_UNIT} a unit walk covers")
    if BLOCK_SIZE * TOPK_MAX + BLOCK_SIZE * TAIL_SLOTS - 2 > ROWS:
        raise ValueError(f"the maximum compacted row count {BLOCK_SIZE * TOPK_MAX + BLOCK_SIZE * TAIL_SLOTS - 2} exceeds ROWS {ROWS}")
    if SLOTS_PER_WARP * WARPS != SLOTS_PER_UNIT or THREADS != WARPS * 32:
        raise ValueError("the unit's slots must split evenly over the CTA's warps")
    if LANES_PER_ROW * ROWS_PER_WARP_LOAD != 32 or LANES_PER_ROW * 16 != D:
        raise ValueError(f"D {D} must be covered by {LANES_PER_ROW} lanes x 16 e4m3 elements, {ROWS_PER_WARP_LOAD} rows per warp-load")
    if BLOCK_SIZE != 2 * ROWS_PER_WARP_LOAD:
        raise ValueError("a block is two warp-loads of rows")


@cute.jit
def _upcast16(w0: cutlass.Int32, w1: cutlass.Int32, w2: cutlass.Int32, w3: cutlass.Int32, scale: cutlass.Float32):
    """16 e4m3 elements (four 32-bit words, memory order) -> 16 bf16 (eight words, memory order) as
    ``bf16(fp32(e4m3) * scale)``: ``cvt.rn.bf16x2.e4m3x2`` (exact: e4m3 is a subset of bf16), ``cvt.f32.bf16`` (exact),
    ``mul.f32x2`` (one fp32 rounding, the torch reference's), ``cvt.rn.bf16x2.f32`` (round-to-nearest-even, torch's cast).
    The scale is a REGISTER operand: a constant float reaching an asm block is the libNVVM immediate ICE."""
    return nvvm.inline_ptx(
        "{ .reg .b16 a<8>; .reg .b32 p<8>; .reg .b16 l<8>; .reg .b16 h<8>; .reg .f32 f<16>; .reg .f32 g<16>; .reg .b64 pa, pb, pc;\n"
        "mov.b32 {a0, a1}, $8; mov.b32 {a2, a3}, $9; mov.b32 {a4, a5}, $10; mov.b32 {a6, a7}, $11;\n"
        "cvt.rn.bf16x2.e4m3x2 p0, a0; cvt.rn.bf16x2.e4m3x2 p1, a1; cvt.rn.bf16x2.e4m3x2 p2, a2; cvt.rn.bf16x2.e4m3x2 p3, a3;\n"
        "cvt.rn.bf16x2.e4m3x2 p4, a4; cvt.rn.bf16x2.e4m3x2 p5, a5; cvt.rn.bf16x2.e4m3x2 p6, a6; cvt.rn.bf16x2.e4m3x2 p7, a7;\n"
        "mov.b32 {l0, h0}, p0; mov.b32 {l1, h1}, p1; mov.b32 {l2, h2}, p2; mov.b32 {l3, h3}, p3;\n"
        "mov.b32 {l4, h4}, p4; mov.b32 {l5, h5}, p5; mov.b32 {l6, h6}, p6; mov.b32 {l7, h7}, p7;\n"
        "cvt.f32.bf16 f0, l0; cvt.f32.bf16 f1, h0; cvt.f32.bf16 f2, l1; cvt.f32.bf16 f3, h1;\n"
        "cvt.f32.bf16 f4, l2; cvt.f32.bf16 f5, h2; cvt.f32.bf16 f6, l3; cvt.f32.bf16 f7, h3;\n"
        "cvt.f32.bf16 f8, l4; cvt.f32.bf16 f9, h4; cvt.f32.bf16 f10, l5; cvt.f32.bf16 f11, h5;\n"
        "cvt.f32.bf16 f12, l6; cvt.f32.bf16 f13, h6; cvt.f32.bf16 f14, l7; cvt.f32.bf16 f15, h7;\n"
        "mov.b64 pb, {$12, $12};\n"
        "mov.b64 pa, {f0, f1}; mul.f32x2 pc, pa, pb; mov.b64 {g0, g1}, pc;\n"
        "mov.b64 pa, {f2, f3}; mul.f32x2 pc, pa, pb; mov.b64 {g2, g3}, pc;\n"
        "mov.b64 pa, {f4, f5}; mul.f32x2 pc, pa, pb; mov.b64 {g4, g5}, pc;\n"
        "mov.b64 pa, {f6, f7}; mul.f32x2 pc, pa, pb; mov.b64 {g6, g7}, pc;\n"
        "mov.b64 pa, {f8, f9}; mul.f32x2 pc, pa, pb; mov.b64 {g8, g9}, pc;\n"
        "mov.b64 pa, {f10, f11}; mul.f32x2 pc, pa, pb; mov.b64 {g10, g11}, pc;\n"
        "mov.b64 pa, {f12, f13}; mul.f32x2 pc, pa, pb; mov.b64 {g12, g13}, pc;\n"
        "mov.b64 pa, {f14, f15}; mul.f32x2 pc, pa, pb; mov.b64 {g14, g15}, pc;\n"
        "cvt.rn.bf16x2.f32 $0, g1, g0; cvt.rn.bf16x2.f32 $1, g3, g2; cvt.rn.bf16x2.f32 $2, g5, g4; cvt.rn.bf16x2.f32 $3, g7, g6;\n"
        "cvt.rn.bf16x2.f32 $4, g9, g8; cvt.rn.bf16x2.f32 $5, g11, g10; cvt.rn.bf16x2.f32 $6, g13, g12; cvt.rn.bf16x2.f32 $7, g15, g14; }",
        write_only_types=[cutlass.Int32] * 8,
        read_only_args=[w0, w1, w2, w3, scale],
    )


@cute.jit
def _st_global_v4_zero(addr: cutlass.Int64):
    """A 16-byte zero store; the zero is materialized INSIDE the asm block (an immediate reaching an operand slot is the
    integer twin of the constant-float ICE)."""
    nvvm.inline_ptx("{ .reg .b32 z; mov.b32 z, 0; st.global.v4.b32 [$0], {z, z, z, z}; }", read_only_args=[addr])


@cute.jit
def _warp_sum_i32(x: cutlass.Int32) -> cutlass.Int32:
    """Butterfly sum over the 32 lanes (every lane ends with the total)."""
    for i in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, 1 << i)
    return x


@cute.jit
def _warp_incl_scan_i32(x: cutlass.Int32, lane: cutlass.Int32) -> cutlass.Int32:
    """Inclusive prefix sum over the lanes (Hillis-Steele, ``shfl.up``)."""
    for i in cutlass.range_constexpr(5):
        d = cutlass.Int32(1 << i)
        y = cute.arch.shuffle_sync_up(x, d)
        x = x + (y if lane >= d else cutlass.Int32(0))
    return x


@cute.kernel
def frost_qsa_kv_upcast(
    mPoolK: cute.Tensor,  # [P, H_kv, page_size, D] e4m3; the first three strides symbolic (HND compact or NHD), D contiguous
    mPoolV: cute.Tensor,
    mTable: cute.Tensor,  # [B, max_pages] int32, contiguous
    mIds: cute.Tensor,  # [rows, top_k] int32, contiguous; rows = B (list_per_sequence) or B x S_q
    mLens: Optional[cute.Tensor],  # [rows] int32, contiguous, or None (count = the position's default)
    mKvLens: cute.Tensor,  # [B] int32
    mKScale: cute.Tensor,  # [1] fp32
    mVScale: cute.Tensor,  # [1] fp32
    mOutK: cute.Tensor,  # [items, ROWS, D] bf16; the item and row strides symbolic, D contiguous
    mOutV: cute.Tensor,
    mKvLenT: cute.Tensor,  # [items] int32
    n_units: cutlass.Int32,  # items x UNITS_PER_ITEM
    n_ctas: cutlass.Int32,  # the persistent grid
    s_q: cutlass.Int32,  # rows per sequence (1..4 under a shared list)
    h_kv: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    list_per_sequence: cutlass.Constexpr[bool],
    zero_fill_full: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One persistent CTA walks the units ``cta, cta + n_ctas, ...``; per unit every warp re-derives the item's slot
    prefix, copies its 8 slots and zero-fills its share of the slab tail (module docstring: GEOMETRY)."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    has_lens = cutlass.const_expr(mLens is not None)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    row_in_pair = lane // cutlass.Int32(LANES_PER_ROW)  # 0 / 1: which of the two rows of a warp-load a lane serves
    col = (lane % cutlass.Int32(LANES_PER_ROW)) * cutlass.Int32(16)  # a lane's 16-element column chunk
    col64 = col.to(cutlass.Int64)

    P = cutlass.Int32(mPoolK.shape[0])
    max_pages = cutlass.Int32(mTable.shape[1])
    pk_base, pv_base = mPoolK.iterator.toint(), mPoolV.iterator.toint()
    pk0, pk1, pk2 = cutlass.Int64(mPoolK.stride[0]), cutlass.Int64(mPoolK.stride[1]), cutlass.Int64(mPoolK.stride[2])
    pv0, pv1, pv2 = cutlass.Int64(mPoolV.stride[0]), cutlass.Int64(mPoolV.stride[1]), cutlass.Int64(mPoolV.stride[2])
    ok_base, ov_base = mOutK.iterator.toint(), mOutV.iterator.toint()
    ok0, ok1 = cutlass.Int64(mOutK.stride[0]), cutlass.Int64(mOutK.stride[1])
    ov0, ov1 = cutlass.Int64(mOutV.stride[0]), cutlass.Int64(mOutV.stride[1])
    ids_base = mIds.iterator.toint()
    table_base = mTable.iterator.toint()
    k_scale = ld_global(mKScale.iterator.toint(), cutlass.Float32)
    v_scale = ld_global(mVScale.iterator.toint(), cutlass.Float32)
    two = cutlass.Int64(2)  # bf16 bytes per element
    four = cutlass.Int64(4)  # int32 bytes

    n_iters = (n_units - cta + n_ctas - cutlass.Int32(1)) // n_ctas
    for it in cutlass.range(n_iters):
        u = cta + it * n_ctas
        item = u // cutlass.Int32(UNITS_PER_ITEM)
        c = u - item * cutlass.Int32(UNITS_PER_ITEM)
        h = item % cutlass.Int32(h_kv)
        tok = item // cutlass.Int32(h_kv)
        s = tok % s_q
        b = tok // s_q
        item64 = item.to(cutlass.Int64)
        h64 = h.to(cutlass.Int64)

        # --- the item's geometry (device values: the length, the count, the tail) ---
        L = ld_global(mKvLens.iterator.toint() + b.to(cutlass.Int64) * four, cutlass.Int32)
        pos = L - s_q + s
        anchor = (L - s_q) if cutlass.const_expr(list_per_sequence) else pos
        n_vis = pos + cutlass.Int32(1)
        n_vis = n_vis if n_vis < L else L
        n_vis = n_vis if n_vis > cutlass.Int32(0) else cutlass.Int32(0)
        n_anchor = anchor + cutlass.Int32(1)
        n_anchor = n_anchor if n_anchor < L else L
        n_anchor = n_anchor if n_anchor > cutlass.Int32(0) else cutlass.Int32(0)
        n_sel = n_anchor // cutlass.Int32(BLOCK_SIZE)
        n_sel = n_sel if n_sel < cutlass.Int32(top_k) else cutlass.Int32(top_k)
        ids_row = b if cutlass.const_expr(list_per_sequence) else tok
        count = n_sel
        if cutlass.const_expr(has_lens):
            given = ld_global(mLens.iterator.toint() + ids_row.to(cutlass.Int64) * four, cutlass.Int32)
            given = given if given > cutlass.Int32(0) else cutlass.Int32(0)
            count = given if given < n_sel else n_sel
        tail0 = n_anchor // cutlass.Int32(BLOCK_SIZE)  # the first tail block: rows [4 tail0, n_vis)
        n_blk_vis = (n_vis + cutlass.Int32(BLOCK_SIZE - 1)) // cutlass.Int32(BLOCK_SIZE)  # a block at or past it has no visible row
        ids_row_base = ids_base + ids_row.to(cutlass.Int64) * cutlass.Int64(top_k) * four

        # --- the slot prefix: every warp walks the item's 17 x 32 slots once (one id per lane per step), UNROLLED so a lane's 17 id
        #     loads are in flight together -- a runtime loop consumed each id in its own iteration and serialized the scan into 17 L2
        #     round trips before the first copy (MEASURED on the 212-SM perf node: 57.6 -> see the record for the unrolled figure) ---
        idvs = []
        for k in cutlass.range_constexpr(UNITS_PER_ITEM):
            j = cutlass.Int32(k * SLOTS_PER_UNIT) + lane
            jl = j if j < cutlass.Int32(top_k) else cutlass.Int32(top_k - 1)
            idvs.append(ld_global(ids_row_base + jl.to(cutlass.Int64) * four, cutlass.Int32))
        before = cutlass.Int32(0)
        after = cutlass.Int32(0)
        my_cnt = cutlass.Int32(0)
        my_blk = cutlass.Int32(-1)
        for k in cutlass.range_constexpr(UNITS_PER_ITEM):
            j = cutlass.Int32(k * SLOTS_PER_UNIT) + lane
            kk = cutlass.Int32(k)
            listed = j < count
            tail = (j >= cutlass.Int32(top_k)) & (j < cutlass.Int32(top_k + TAIL_SLOTS))
            blk = idvs[k] if listed else (tail0 + (j - cutlass.Int32(top_k)))
            blk = blk if (listed | tail) else cutlass.Int32(-1)
            inside = (blk >= cutlass.Int32(0)) & (blk < n_blk_vis)
            cnt = n_vis - blk * cutlass.Int32(BLOCK_SIZE)
            cnt = cnt if cnt < cutlass.Int32(BLOCK_SIZE) else cutlass.Int32(BLOCK_SIZE)
            cnt = cnt if inside else cutlass.Int32(0)
            before = before + (cnt if kk < c else cutlass.Int32(0))
            after = after + (cnt if kk > c else cutlass.Int32(0))
            my_cnt = cnt if kk == c else my_cnt
            my_blk = blk if kk == c else my_blk
        unit_base = _warp_sum_i32(before)
        incl = _warp_incl_scan_i32(my_cnt, lane)
        unit_sum = cute.arch.shuffle_sync(incl, 31)
        total = unit_base + unit_sum + _warp_sum_i32(after)
        my_dst = unit_base + incl - my_cnt  # the destination row of the slot held here

        if (c == cutlass.Int32(0)) & (tidx == cutlass.Int32(0)):
            st_global(mKvLenT.iterator.toint() + item64 * four, total, cutlass.Int32)

        # --- the copy, phase 1: this warp's 8 slots -- (block, count, destination) by shuffle from the holder and the 8 table words,
        #     all in flight together (one L2 round trip for the warp's slots, not one per slot pair) ---
        ok_item = ok_base + item64 * ok0 * two
        ov_item = ov_base + item64 * ov0 * two
        b64 = b.to(cutlass.Int64)
        max_pages64 = max_pages.to(cutlass.Int64)
        s_cnt = []
        s_dst = []
        s_pidx_ok = []
        s_inpage = []
        s_page = []
        for i in cutlass.range_constexpr(SLOTS_PER_WARP):
            src_lane = warp * cutlass.Int32(SLOTS_PER_WARP) + cutlass.Int32(i)
            cnt_s = cute.arch.shuffle_sync(my_cnt, src_lane)
            blk_s = cute.arch.shuffle_sync(my_blk, src_lane)
            dst_s = cute.arch.shuffle_sync(my_dst, src_lane)
            t_blk = blk_s * cutlass.Int32(BLOCK_SIZE)  # the block's first token (blk_s >= 0 whenever cnt_s > 0)
            pidx = t_blk // cutlass.Int32(page_size)
            in_page = t_blk - pidx * cutlass.Int32(page_size)
            pidx_ok = (pidx >= cutlass.Int32(0)) & (pidx < max_pages)
            pidx_c = pidx if pidx_ok else cutlass.Int32(0)
            s_page.append(ld_global(table_base + (b64 * max_pages64 + pidx_c.to(cutlass.Int64)) * four, cutlass.Int32))
            s_cnt.append(cnt_s)
            s_dst.append(dst_s)
            s_pidx_ok.append(pidx_ok)
            s_inpage.append(in_page)
        # --- the copy, phase 2: two slots per step so eight 16-B loads are in flight per lane; a dead page reads page 0 and stores
        #     zeros; a row at or past the slot's count is loaded (a real address) and never stored ---
        for i in cutlass.range_constexpr(SLOTS_PER_WARP // 2):
            loads_k = []
            loads_v = []
            lives = []
            poks = []
            for q in cutlass.range_constexpr(2):
                si = 2 * i + q
                page = s_page[si]
                page_ok = s_pidx_ok[si] & (page >= cutlass.Int32(0)) & (page < P)
                page_c = (page if page_ok else cutlass.Int32(0)).to(cutlass.Int64)
                src_k = pk_base + page_c * pk0 + h64 * pk1 + s_inpage[si].to(cutlass.Int64) * pk2 + col64
                src_v = pv_base + page_c * pv0 + h64 * pv1 + s_inpage[si].to(cutlass.Int64) * pv2 + col64
                for rp in cutlass.range_constexpr(2):
                    row = cutlass.Int32(rp * ROWS_PER_WARP_LOAD) + row_in_pair
                    row64 = row.to(cutlass.Int64)
                    loads_k.append(ld_global_v4(src_k + row64 * pk2, cutlass.Int32))
                    loads_v.append(ld_global_v4(src_v + row64 * pv2, cutlass.Int32))
                    lives.append(row < s_cnt[si])
                poks.append(page_ok)
            for q in cutlass.range_constexpr(2):
                si = 2 * i + q
                page_ok = poks[q]
                for rp in cutlass.range_constexpr(2):
                    idx = 2 * q + rp
                    row = cutlass.Int32(rp * ROWS_PER_WARP_LOAD) + row_in_pair
                    if lives[idx]:
                        wk = loads_k[idx]
                        wv = loads_v[idx]
                        z = cutlass.Int32(0)
                        k0 = wk[0] if page_ok else z
                        k1 = wk[1] if page_ok else z
                        k2 = wk[2] if page_ok else z
                        k3 = wk[3] if page_ok else z
                        v0 = wv[0] if page_ok else z
                        v1 = wv[1] if page_ok else z
                        v2 = wv[2] if page_ok else z
                        v3 = wv[3] if page_ok else z
                        ok_words = _upcast16(k0, k1, k2, k3, k_scale)
                        ov_words = _upcast16(v0, v1, v2, v3, v_scale)
                        drow = (s_dst[si] + row).to(cutlass.Int64)
                        dk = ok_item + drow * ok1 * two + col64 * two
                        dv = ov_item + drow * ov1 * two + col64 * two
                        st_global_v4(dk, (ok_words[0], ok_words[1], ok_words[2], ok_words[3]), cutlass.Int32)
                        st_global_v4(dk + cutlass.Int64(16), (ok_words[4], ok_words[5], ok_words[6], ok_words[7]), cutlass.Int32)
                        st_global_v4(dv, (ov_words[0], ov_words[1], ov_words[2], ov_words[3]), cutlass.Int32)
                        st_global_v4(dv + cutlass.Int64(16), (ov_words[4], ov_words[5], ov_words[6], ov_words[7]), cutlass.Int32)

        # --- the zero fill of this unit's share of the slab: rows [max(total, 128 c), 128 (c + 1)), both operands ---
        z0 = c * cutlass.Int32(ROW_TILE)
        z0 = z0 if z0 > total else total
        z1 = (c + cutlass.Int32(1)) * cutlass.Int32(ROW_TILE)
        if cutlass.const_expr(not zero_fill_full):
            tile_end = ((total + cutlass.Int32(ROW_TILE - 1)) // cutlass.Int32(ROW_TILE)) * cutlass.Int32(ROW_TILE)
            z1 = z1 if z1 < tile_end else tile_end
        zlane = lane.to(cutlass.Int64) * cutlass.Int64(16)  # 32 lanes x 16 B = one 512-B bf16 row
        n_zero = z1 - z0
        n_mine = (n_zero - warp + cutlass.Int32(WARPS - 1)) // cutlass.Int32(WARPS)
        n_mine = n_mine if n_mine > cutlass.Int32(0) else cutlass.Int32(0)
        for zi in cutlass.range(n_mine):
            zrow = (z0 + warp + zi * cutlass.Int32(WARPS)).to(cutlass.Int64)
            _st_global_v4_zero(ok_item + zrow * ok1 * two + zlane)
            _st_global_v4_zero(ov_item + zrow * ov1 * two + zlane)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def qsa_kv_upcast_launch(
    pool_k: cute.Tensor,
    pool_v: cute.Tensor,
    table: cute.Tensor,
    ids: cute.Tensor,
    lens: Optional[cute.Tensor],
    kv_lens: cute.Tensor,
    k_scale: cute.Tensor,
    v_scale: cute.Tensor,
    out_k: cute.Tensor,
    out_v: cute.Tensor,
    kv_len_t: cute.Tensor,
    n_units: cutlass.Int32,
    n_ctas: cutlass.Int32,
    s_q: cutlass.Int32,
    h_kv: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    list_per_sequence: cutlass.Constexpr[bool],
    zero_fill_full: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_kv_upcast(
        pool_k,
        pool_v,
        table,
        ids,
        lens,
        kv_lens,
        k_scale,
        v_scale,
        out_k,
        out_v,
        kv_len_t,
        n_units,
        n_ctas,
        s_q,
        h_kv,
        top_k,
        page_size,
        list_per_sequence,
        zero_fill_full,
        use_pdl,
    ).launch(grid=(n_ctas, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


class QsaKvUpcastRecipe(NamedTuple):
    """Build-time facts of one artifact (Rule 4: one artifact per (H_kv, top_k, page_size, list form, block_lens presence,
    zero-fill mode)); the batch, the query count, the pool capacity and every stride ride in at runtime."""

    compiled: object
    h_kv: int
    top_k: int
    page_size: int
    list_per_sequence: bool
    has_block_lens: bool
    zero_fill: str
    sm_count: int
    device: int


def scratch_shape(n_items: int):
    """The scratch the kernel fills: ``[items, 2 (K, V), ROWS, D]`` bf16 -- ``scratch[:, 0]`` / ``scratch[:, 1]`` are the
    ``out_k`` / ``out_v`` operands (their item stride ``2 x ROWS x D``, row stride ``D``)."""
    return (int(n_items), 2, ROWS, D)


def scratch_bytes(n_items: int) -> int:
    return 2 * int(n_items) * 2 * ROWS * D


def dense_core_views(scratch: torch.Tensor, kv_len_t: torch.Tensor):
    """The views the bf16 d256 DECODE tile consumes over the scratch as a plain PADDED dense SDPA: K / V as ``[items, H_kv' = 1,
    ROWS, D]`` (BHSD dims over the scratch's storage) and ``seq_len_kv = kv_len_t`` as ``[items, 1, 1, 1]``; the matching Q is
    :func:`q_items_view`.  Pure views: nothing is copied."""
    n_items = int(scratch.shape[0])
    if tuple(scratch.shape) != scratch_shape(n_items) or scratch.dtype != _OUT_DTYPE:
        raise ValueError(f"scratch must be {scratch_shape(n_items)} bf16, got {tuple(scratch.shape)} {scratch.dtype}")
    if tuple(kv_len_t.shape) != (n_items,) or kv_len_t.dtype != torch.int32:
        raise ValueError(f"kv_len_t must be int32 [{n_items}], got {tuple(kv_len_t.shape)} {kv_len_t.dtype}")
    return scratch[:, 0].unsqueeze(1), scratch[:, 1].unsqueeze(1), kv_len_t.view(n_items, 1, 1, 1)


def q_items_view(q: torch.Tensor, h_kv: int) -> torch.Tensor:
    """``q [B, S_q, H_q, D]`` (heads contiguous within a token) as the per-item packed query group ``[items, G, 1, D]``
    (BHSD dims: ``S_q' = 1`` row, the ``G = H_q / H_kv`` heads of the item's KV head) -- a view when the token stride is
    ``H_q x D``, a copy otherwise (``reshape``)."""
    B, S_q, H_q, d = (int(x) for x in q.shape)
    if d != D or H_q % h_kv:
        raise ValueError(f"q must be [B, S_q, H_q = k x {h_kv}, {D}], got {tuple(q.shape)}")
    return q.reshape(B * S_q * h_kv, H_q // h_kv, D).unsqueeze(2)


def items_to_rows(o_items: torch.Tensor, B: int, S_q: int, H_q: int) -> torch.Tensor:
    """The dense core's O ``[items, G, 1, D]`` back to ``[B, S_q, H_q, D]``."""
    return o_items.reshape(B, S_q, H_q, D)


def _fake_pool(h_kv: int, page_size: int):
    from cudnn.datatypes import _convert_to_cutlass_data_type

    return cute.runtime.make_fake_tensor(
        dtype=_convert_to_cutlass_data_type(_POOL_DTYPE),
        shape=(cute.sym_int(), h_kv, page_size, D),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )


def _fake_out():
    return cute.runtime.make_fake_tensor(dtype=cutlass.BFloat16, shape=(cute.sym_int(), ROWS, D), stride=(cute.sym_int(), cute.sym_int(), 1), assumed_align=16)


def _fake_i32_2d(cols):
    return cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(), cols), stride=(cols, 1), assumed_align=4)


def _fake_i32_1d():
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4)


def _fake_f32_scalar():
    return cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)


def dsl_decline_reason(device: Optional[int] = None) -> Optional[str]:
    """Rule 7 / the arch gate: None when this device and DSL can build the kernel, else the typed reason.  The kernel is
    bound to the Rubin line (``mul.f32x2`` / ``cvt.rn.bf16x2.e4m3x2`` are sm_100+ instructions; the block targets cc 10.7)
    and the DSL must know the ``sm_107a`` target (the public 4.8.0 wheel)."""
    from cudnn.frost.buffers import cutedsl_arch_requirement_error

    dev = current_device() if device is None else int(device)
    cc = compute_capability(dev)
    if not (10, 7) <= cc <= (11, 9):
        return f"qsa_kv_upcast targets the Rubin line (cc 10.7 .. 11.9); this device is cc {cc[0]}.{cc[1]}"
    return cutedsl_arch_requirement_error(cc)


def validate_params(*, h_kv: int, top_k: int, page_size: int, zero_fill: str) -> None:
    """The compile-key facts, typed; never an assert."""
    _validate_geometry()
    if int(h_kv) < 1:
        raise ValueError(f"h_kv must be >= 1, got {h_kv}")
    if not (TOPK_MIN <= int(top_k) <= TOPK_MAX) or int(top_k) % BLOCK_SIZE:
        raise ValueError(f"top_k must be a multiple of {BLOCK_SIZE} in [{TOPK_MIN}, {TOPK_MAX}], got {top_k}")
    if int(page_size) < BLOCK_SIZE or int(page_size) % BLOCK_SIZE:
        raise ValueError(f"page_size must be a positive multiple of {BLOCK_SIZE} (a 4-token block never straddles a page), got {page_size}")
    if zero_fill not in ZERO_FILL_MODES:
        raise ValueError(f"zero_fill must be one of {ZERO_FILL_MODES}, got {zero_fill!r}")


def compile_qsa_kv_upcast(
    *,
    h_kv: int,
    top_k: int,
    page_size: int,
    list_per_sequence: bool,
    has_block_lens: bool,
    zero_fill: str = "full",
    use_pdl: bool = False,
    device: Optional[int] = None,
) -> QsaKvUpcastRecipe:
    """Build from SHAPES ALONE (no allocation, no launch).  The DSL / arch gate runs BEFORE the compile (Rule 7): a decline
    is a ``NotImplementedError`` naming the device and the DSL, never a ``KeyError`` from inside the DSL."""
    global _FAKE_STREAM
    validate_params(h_kv=h_kv, top_k=top_k, page_size=page_size, zero_fill=zero_fill)
    dev = current_device() if device is None else int(device)
    why = dsl_decline_reason(dev)
    if why is not None:
        raise NotImplementedError(why)
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    key = (int(h_kv), int(top_k), int(page_size), bool(list_per_sequence), bool(has_block_lens), str(zero_fill), bool(use_pdl), dev)
    if key not in compiled_cache:
        compiled_cache[key] = cute.compile(
            qsa_kv_upcast_launch,
            _fake_pool(h_kv, page_size),
            _fake_pool(h_kv, page_size),
            _fake_i32_2d(cute.sym_int()),  # the table: [B, max_pages], both symbolic
            _fake_i32_2d(int(top_k)),  # the lists: [rows, top_k]
            _fake_i32_1d() if has_block_lens else None,
            _fake_i32_1d(),  # kv_lens
            _fake_f32_scalar(),
            _fake_f32_scalar(),
            _fake_out(),
            _fake_out(),
            _fake_i32_1d(),  # kv_len_t
            cutlass.Int32(0),  # n_units  ) runtime; the zeros pin the TYPE only
            cutlass.Int32(0),  # n_ctas   )
            cutlass.Int32(0),  # s_q      )
            int(h_kv),
            int(top_k),
            int(page_size),
            bool(list_per_sequence),
            zero_fill == "full",
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QsaKvUpcastRecipe(
        compiled=compiled_cache[key],
        h_kv=int(h_kv),
        top_k=int(top_k),
        page_size=int(page_size),
        list_per_sequence=bool(list_per_sequence),
        has_block_lens=bool(has_block_lens),
        zero_fill=str(zero_fill),
        sm_count=multiprocessor_count(dev),
        device=dev,
    )


def _check_i32(t, name, shape, *, device):
    if not isinstance(t, torch.Tensor) or t.dtype != torch.int32:
        raise ValueError(f"{name} must be an int32 tensor of shape {shape}, got {getattr(t, 'dtype', type(t).__name__)}")
    if tuple(int(x) for x in t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(t.stride())}")
    if t.device != device:
        raise ValueError(f"{name} must live on {device}, got {t.device}")


def check_pool(pool: torch.Tensor, name: str, *, h_kv: int, page_size: int, device) -> None:
    """One e4m3 pool against the kernel's contract (no device read): ``[num_pages >= 1, H_kv, page_size, D]`` with ``D``
    contiguous, every other stride a whole number of 16-byte vectors, a 16-byte aligned base.  e5m2 is declined BY NAME
    (``NotImplementedError``): the block's fp8 KV cache is e4m3 (the serving stacks' ``fp8`` = e4m3fn)."""
    if not isinstance(pool, torch.Tensor):
        raise ValueError(f"{name} must be a [num_pages, {h_kv}, {page_size}, {D}] e4m3 tensor, got {type(pool).__name__}")
    if pool.dtype == torch.float8_e5m2:
        raise NotImplementedError(f"{name} is e5m2: qsa_kv_upcast serves e4m3 (float8_e4m3fn) pools only")
    if pool.dtype != _POOL_DTYPE:
        raise ValueError(f"{name} must be {_POOL_DTYPE} (the e4m3 KV cache), got {pool.dtype}")
    if pool.dim() != 4 or tuple(int(x) for x in pool.shape[1:]) != (h_kv, page_size, D) or int(pool.shape[0]) < 1:
        raise ValueError(
            f"{name} must be [num_pages >= 1, H_kv={h_kv}, page_size={page_size}, D={D}] (HND compact, or NHD by strides), got {tuple(pool.shape)}"
        )
    strides = tuple(int(s) for s in pool.stride())
    if strides[3] != 1:
        raise ValueError(f"{name} must be contiguous along D, got strides {strides}")
    for ax, st in enumerate(strides[:3]):
        if st % 16:
            raise ValueError(f"{name} stride[{ax}] = {st} bytes must be a multiple of 16 (rows are read in 16-byte vectors)")
    if pool.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned, got data_ptr={pool.data_ptr():#x}")
    if pool.device != device:
        raise ValueError(f"{name} must live on {device}, got {pool.device}")


def check_out(out: torch.Tensor, name: str, *, n_items: int, device) -> None:
    """One bf16 output slab ``[items, ROWS, D]`` with ``D`` contiguous and 16-byte-aligned item / row strides."""
    if not isinstance(out, torch.Tensor) or out.dtype != _OUT_DTYPE:
        raise ValueError(f"{name} must be a bf16 [{n_items}, {ROWS}, {D}] tensor, got {getattr(out, 'dtype', type(out).__name__)}")
    if tuple(int(x) for x in out.shape) != (n_items, ROWS, D):
        raise ValueError(f"{name} must be [{n_items}, {ROWS}, {D}], got {tuple(out.shape)}")
    strides = tuple(int(s) for s in out.stride())
    if strides[2] != 1 or (strides[0] * 2) % 16 or (strides[1] * 2) % 16 or strides[1] < D:
        raise ValueError(f"{name} must have D contiguous and 16-byte-aligned item / row strides (row stride >= {D}), got {strides}")
    if out.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned, got data_ptr={out.data_ptr():#x}")
    if out.device != device:
        raise ValueError(f"{name} must live on {device}, got {out.device}")


def n_units_of(n_items: int) -> int:
    return int(n_items) * UNITS_PER_ITEM


def n_ctas_of(n_units: int, sm_count: int) -> int:
    """The persistent grid: the device's cap (``SMs x CTAS_PER_SM``), never more than the units, at least one."""
    return max(1, min(int(n_units), int(sm_count) * CTAS_PER_SM))


def run_qsa_kv_upcast(
    r: QsaKvUpcastRecipe,
    *,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_ids: torch.Tensor,
    block_lens: Optional[torch.Tensor],
    seq_kv_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    kv_len_t: torch.Tensor,
    s_q: int,
    stream,
) -> None:
    """Validate the FORM of every operand (no device read) and launch (Rule 1: no conversion, no allocation).

    ``block_ids``: int32 contiguous ``[B, top_k]`` under ``list_per_sequence`` (one list per sequence, the ``S_q <= 4`` rows share
    it) or ``[B, S_q, top_k]`` per token; ``block_lens`` ``[B]`` / ``[B, S_q]`` int32 or None (its presence is the artifact's);
    ``seq_kv_lens`` int32 ``[B]``; ``k_scale`` / ``v_scale`` 1-element fp32 CUDA tensors; ``out_k`` / ``out_v`` bf16
    ``[B x S_q x H_kv, ROWS, D]`` (``scratch[:, 0]`` / ``scratch[:, 1]`` of :func:`scratch_shape`); ``kv_len_t`` int32
    ``[B x S_q x H_kv]``."""
    dev = k_pool.device if isinstance(k_pool, torch.Tensor) else torch.device("cuda")
    check_pool(k_pool, "k_pool", h_kv=r.h_kv, page_size=r.page_size, device=dev)
    check_pool(v_pool, "v_pool", h_kv=r.h_kv, page_size=r.page_size, device=dev)
    if int(k_pool.shape[0]) != int(v_pool.shape[0]):
        raise ValueError(f"k_pool and v_pool must hold the same number of pages, got {int(k_pool.shape[0])} and {int(v_pool.shape[0])}")
    s_q = int(s_q)
    if s_q < 1 or (r.list_per_sequence and s_q > MTP_MAX_S_Q):
        raise ValueError(f"s_q must be >= 1 (and <= {MTP_MAX_S_Q} under a shared per-sequence list), got {s_q}")
    if not isinstance(block_table, torch.Tensor) or block_table.dim() != 2:
        raise ValueError(f"block_table must be a contiguous int32 (B, max_pages) tensor, got {getattr(block_table, 'shape', type(block_table).__name__)}")
    B, max_pages = (int(x) for x in block_table.shape)
    _check_i32(block_table, "block_table", (B, max_pages), device=dev)
    if max_pages < 1:
        raise ValueError("block_table must have at least one page column")
    _check_i32(seq_kv_lens, "seq_kv_lens", (B,), device=dev)
    rows = B if r.list_per_sequence else B * s_q
    ids_shape = (B, r.top_k) if r.list_per_sequence else (B, s_q, r.top_k)
    if not isinstance(block_ids, torch.Tensor) or tuple(int(x) for x in block_ids.shape) not in (ids_shape, (rows, r.top_k)):
        raise ValueError(f"block_ids must be int32 {ids_shape} (or [{rows}, {r.top_k}]), got {getattr(block_ids, 'shape', type(block_ids).__name__)}")
    _check_i32(block_ids, "block_ids", tuple(int(x) for x in block_ids.shape), device=dev)
    if r.has_block_lens != (block_lens is not None):
        raise ValueError(f"this artifact was compiled {'with' if r.has_block_lens else 'without'} block_lens; the call must match (Rule 1: no silent fallback)")
    if block_lens is not None:
        lens_shape = (B,) if r.list_per_sequence else (B, s_q)
        if tuple(int(x) for x in block_lens.shape) not in (lens_shape, (rows,)):
            raise ValueError(f"block_lens must be int32 {lens_shape}, got {tuple(block_lens.shape)}")
        _check_i32(block_lens, "block_lens", tuple(int(x) for x in block_lens.shape), device=dev)
    for name, sc in (("k_scale", k_scale), ("v_scale", v_scale)):
        if not isinstance(sc, torch.Tensor) or sc.dtype != torch.float32 or int(sc.numel()) != 1 or sc.device != dev:
            raise ValueError(f"{name} must be a 1-element fp32 tensor on {dev} (a device scalar, read by the kernel), got {sc!r}")
    n_items = B * s_q * r.h_kv
    check_out(out_k, "out_k", n_items=n_items, device=dev)
    check_out(out_v, "out_v", n_items=n_items, device=dev)
    _check_i32(kv_len_t, "kv_len_t", (n_items,), device=dev)
    n_units = n_units_of(n_items)
    n_ctas = n_ctas_of(n_units, r.sm_count)
    r.compiled(
        k_pool,
        v_pool,
        block_table,
        block_ids.view(rows, r.top_k),  # the artifact's 2-D [rows, top_k] (a view: contiguous)
        None if block_lens is None else block_lens.view(rows),
        seq_kv_lens,
        k_scale,
        v_scale,
        out_k,
        out_v,
        kv_len_t,
        cutlass.Int32(n_units),
        cutlass.Int32(n_ctas),
        cutlass.Int32(s_q),
        cuda.CUstream(int(stream)),
    )


def moved_bytes(
    n_items: int, rows_read_total: int, *, zero_fill: str = "full", rows_written_total: Optional[int] = None, top_k: int = TOPK_MAX, max_pages: int = 0
) -> int:
    """The HBM traffic model of one launch for a SOL number: the gathered e4m3 rows read once per operand, the slab rows
    written once per operand (``full``: every slab row; ``tile``: the written rows rounded up to the 128-row tile), plus the
    lists and the table.  ``rows_read_total`` = the sum of ``kv_len_t`` over the items (host arithmetic from the caller's
    own lists; the kernel reads nothing on the host)."""
    rows_read_total = int(rows_read_total)
    if zero_fill == "full":
        written = int(n_items) * ROWS
    else:
        written = int(rows_written_total) if rows_written_total is not None else rows_read_total
    return 2 * rows_read_total * D + 2 * written * D * 2 + int(n_items) * top_k * 4 + int(n_items) * max_pages * 4


frost_qsa_kv_upcast.set_name_prefix("cudnn", remove_cutlass_symbol=True)
