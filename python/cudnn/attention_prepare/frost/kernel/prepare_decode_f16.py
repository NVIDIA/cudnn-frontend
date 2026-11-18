# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The kernel in this file follows the thread shape and the arithmetic of _fused_qk_rmsnorm_rope_gate_kernel of vLLM
# (vllm/model_executor/layers/fused_qk_norm_rope.py, https://github.com/vllm-project/vllm at commit
# 2a777ce26e8315fa503ac5e88b595235d1c2b928; Copyright contributors to the vLLM project), licensed under Apache-2.0, and has
# been modified; the module docstring lists the changes.

"""The attention preparation DECODE kernel (bf16 in and out, one launch): per token the RMSNorm of every query and key head row, the
rotation of its rotary part by the token's position, the gate rows copied beside the queries, and the keys and values written into
paged pools at the token's slot.  Config and forms in ``prepare_decode_config.py``, the host entry in
``attention_prepare/frost/prepare_decode.py``.  Arch-portable PTX (global loads / stores, ``cvt``, ``div.full``,
``rsqrt.approx.ftz``, ``shfl.sync``, ``bar.sync``, ``griddepcontrol``); the arch axis is an admission fact of the host entry, not a
directory.

CHANGES against the source the file header names: written on the CuTe DSL (the source is a Triton kernel); the geometry, the
epsilon, the position form, the page size and the two integer dtypes are compile-time constants of a config; the element type is
bf16 only (the source takes its input dtype as a constant: fp16 or bf16); the gain is ``1 + weight`` (the source adds a constant
``norm_beta`` to the weight; this kernel fixes it at 1); the query rows and the gate rows are two operands with their own strides
(the source reads one ``[q | gate]`` row per head); a thread's elements are loaded once (the source loads the rotary elements and
their weights a second time and normalises them again -- the same fp32 operations on the same operands); the rotary part reads its
partner element from another thread of the warp instead of from memory and spells its fused multiply-adds explicitly (the source
leaves the contraction of its two-product sums to its compiler); the keys are written into a paged pool at a slot and the value
rows are copied into a second pool beside them (the source writes the keys to a dense output and has no value operand); the pool
store is also predicated by the pool's capacity; the dependent-launch wait and release are launch arms.

THREAD BLOCK: one per (token, head), ``grid = (tokens, HEADS + KV_HEADS)``; ``cfg.THREADS = HEAD_DIM / 2`` threads in ``cfg.WARPS``
FULL warps; thread ``i`` owns the two contiguous elements ``2 i``, ``2 i + 1`` of its head row.  A block is a QUERY program
(``head < HEADS``: norm + rotation of ``q[token, head]`` into ``q_out``, the gate row copied into ``gate_out``) or a KEY / VALUE
program (norm + rotation of ``k[token, head - HEADS]`` into the K pool row of the token's slot, the V row copied into the V pool).
The program is block-uniform.  Lane 0 of each warp additionally publishes its warp's sum; warp 0 additionally rotates.

PIPELINE, in order, for every thread:

1. ``griddepcontrol.launch_dependents`` under ``cfg.PDL_RELEASE`` -- the kernel's first statement.
2. The address of the thread's two weights (``q_weight`` or ``k_weight`` by program: an address select) and ONE 32-bit global load
   of them.  The weights are constants of the layer (the PDL-sites paragraph).
3. ``griddepcontrol.wait`` under ``cfg.PDL``.
4. The program's addresses.  Key / value program: the token's slot -- one load of the slot dtype's width, ``-1`` for a token at or
   past the slot vector's length -- and ``live = 0 <= slot < pool slots``; ``page = slot / PAGE_SIZE`` and ``row = slot mod
   PAGE_SIZE`` (a shift and a mask: the page size is a compile-time power of two); the K and V pool rows by the pools' run-time
   strides.
5. ONE 32-bit global load of the thread's two row elements and one of the copied operand's (gate / V); bf16 -> fp32 is exact.
6. The thread's partial sum of squares ``s = fma(x0, x0, x1 x x1)``: one fp32 multiply and one FUSED multiply-add.
7. The row total: ``tile_dsl.rowwise.block_reduce(s, lane, warp, slots, WARPS, kind="sum", order="descending")`` -- inside each warp
   the butterfly offsets 16, 8, 4, 2, 1; lane 0 stores the warp's sum; the thread-block barrier; EVERY lane loads slot
   ``lane mod WARPS``; the offsets ``WARPS / 2 .. 1``.
8. ``inv = rsqrt.approx.ftz( div.full(total, HEAD_DIM) + eps )``: the approximate full-range division with the width in a register,
   one addition, the approximate reciprocal square root with flush-to-zero.
9. ``n_j = bf16( (x_j x inv) x (w_j + 1) )``: two fp32 multiplications and one addition, then nearest-even to bf16 -- THE normalized
   value of record, widened back exactly for the rotation.
10. Warp 0 only (the elements ``0 .. ROTARY_DIM - 1``): the partner value of each element from the thread ``HALF / 2`` lanes away (one
    ``shfl.sync.bfly`` per element); the position of each of the thread's two rotary indices (``"3d"``: one of three position rows
    by a compile-time mask over the index); the cosine and the sine of each index from the bf16 table row of that position (32-bit
    loads of the table word that holds the pair); ONE fp32 multiply (the rounded product) and ONE fused multiply-add per value,
    rounded once to bf16: first half ``fma(n[r], cos, -(n[r + HALF] x sin))``; second half ``fma(n[r + HALF], cos, n[r] x sin)``
    under ``"3d"`` and ``fma(n[r], sin, n[r + HALF] x cos)`` under ``"1d"``.
11. Under ``live`` (always, in a query program): ONE 32-bit global store of the thread's two output elements and one of the copied
    pair.

THE SUM TREE AND THE ROUNDING CHAIN ARE THE CONTRACT.  The fused leaf is explicit (``cute.math.fma``); the row total is ONE call with
the order named (the order is part of the bytes); the division is ``div.full.f32`` with a register divisor and the root
``rsqrt.approx.ftz.f32`` (never the IEEE division, a reciprocal multiply spelled in the source, or the root without flush-to-zero:
each is the same quantity with other last bits on some rows); the normalized value is rounded to bf16 BEFORE the rotation; each
rotated value is one fp32 multiply and one fused multiply-add of two bf16 values and that product.  WHICH product of the second
rotary half is the fused one follows the position form (step 10).  A product of two bf16 values is exact in fp32 unless it
underflows, so the two orders write the same bytes on every operand whose products are normal fp32 values and differ -- in the
sign of a zero, in a subnormal's last bit -- where a product underflows; each form keeps ONE order, and that order is its bytes.
The test module's rendering cell pins these instructions, the operand selects that make the ``"1d"`` order included (eight
``selp.f32`` under ``"1d"``, none under ``"3d"``); its reference cells pin the bytes inside a stated band, and one cell of
underflowing rows pins the fused product of each form BITWISE (there the two orders differ in the sign of a zero).

BATCH INVARIANCE AND REPLAY.  A thread block reads its own head row, the shared weights, its token's positions and slot and the
table rows those positions select, and nothing of another token or head; the reduction never leaves the block; there is no atomic
and no cross-row scratch.  The token index enters ADDRESSES only.

BARRIER TABLE.  No mbarrier, no cluster, no ``cp.async`` group, no loop with a run-time trip count.  ONE synchronization object:

    barrier                         kind                 participants      issuing lanes            guard
    the thread-block barrier (7)    named barrier 0      cfg.THREADS       WARPS x 32, once each    none

``SUM(issuing lanes) = 32 x WARPS == THREADS``: every thread the barrier counts reaches it exactly once on every path.  The run-time
branches in front of it -- the block-uniform program branch of step 4, lane 0's slot store inside the reduction -- rejoin before it;
the rotation branch (warp 0) and the store predicate sit behind it and hold no barrier.  The barrier and the slot accesses of the
table below are emitted by ``block_reduce``; the kernel reaches that call with every thread, outside any run-time branch.

SHARED-MEMORY TABLE.  ONE buffer; no tensor memory, no descriptor:

    buffer      dtype x elems    writer                          reader                               refilled in a launch
    warp sums   fp32 x WARPS     lane 0 of warp w, slot w,       EVERY lane, slot lane mod WARPS,     no: written once, read
                                 before the barrier              behind the barrier                   once -> nothing releases it

Lanes address the slots directly at a 4-byte stride (one word per lane in an access phase); no swizzle because no descriptor reads it.

PDL SITES.  Under ``cfg.PDL`` the launch carries the programmatic-stream-serialization attribute and every thread issues
``griddepcontrol.wait`` AFTER the load of its two weights (step 2) and BEFORE every other global read -- the slot, the row, the
copied operand, the positions, the table -- and before every global store.  CALLER OBLIGATION: the weights are not written by any
launch that may still be running when this one starts.  Under ``cfg.PDL_RELEASE`` (which needs ``cfg.PDL``) the release is the
kernel's first statement, in front of its own wait: it hands out nothing this kernel waits on.  CALLER OBLIGATION: every grid that
can start before this one completes issues its own wait before it reads this launch's outputs or the pools and before it writes
any operand of this launch.  The kernel holds no tensor memory and 4 x WARPS bytes of shared memory.

OPERANDS are device ADDRESSES (``Int64``) and byte strides: the kernel addresses flat and reads no extent, so no shape, stride or
alignment of an operand enters the compiled kernel (the host entry checks them before every launch).
"""

import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import div_full_f32, f16x2_to_f32, fp32_to_fp16, opaque_f32_bits, rsqrt_approx_ftz_f32
from cudnn.frost.tile_dsl.rowwise import block_reduce
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v2, st_global

from .prepare_decode_config import ELEMENT_BYTES, INT_BYTES, PAIR, WARP, position_row_masks

FP32, INT32, INT64, BF16 = cutlass.Float32, cutlass.Int32, cutlass.Int64, cutlass.BFloat16
PAIR_BYTES = PAIR * ELEMENT_BYTES  # one 32-bit access: a thread's two bf16 elements
SIGN_BIT = -(2**31)  # the fp32 sign bit as an Int32


@cute.jit
def _load_int(addr, wide: cutlass.Constexpr[bool]):
    """One integer of ``positions`` / ``slot_mapping`` at the byte address ``addr`` as an ``Int64``: a 64-bit load as two 32-bit
    halves recombined (the 64-bit value is what an address or a range check sees), or one 32-bit load, sign-extended."""
    value = INT64(0)
    if cutlass.const_expr(wide):
        lo, hi = ld_global_v2(addr, INT32)
        value = (hi.to(INT64) * INT64(1 << 32)) + (lo.to(INT64) & INT64(0xFFFFFFFF))
    else:
        value = ld_global(addr, INT32).to(INT64)
    return value


@cute.kernel
def frost_attention_prepare(
    cfg: cutlass.Constexpr,
    q: cutlass.Int64,  # device addresses: bf16 [T, HEADS, D] / [T, KV_HEADS, D] views, D contiguous
    gate: cutlass.Int64,
    k: cutlass.Int64,
    v: cutlass.Int64,
    q_weight: cutlass.Int64,  # bf16 [D]
    k_weight: cutlass.Int64,
    cos_sin: cutlass.Int64,  # bf16 [P, ROTARY_DIM], contiguous: HALF cosines, then HALF sines
    positions: cutlass.Int64,  # the entry's position dtype: [T], or [3, T] with the row stride below
    k_pool: cutlass.Int64,  # bf16 [pages, KV_HEADS, PAGE_SIZE, D], D contiguous
    v_pool: cutlass.Int64,
    slot_mapping: cutlass.Int64,  # the entry's slot dtype, contiguous, num_slots entries
    q_out: cutlass.Int64,  # bf16 [T, HEADS, D], contiguous
    gate_out: cutlass.Int64,
    q_token: cutlass.Int64,  # byte strides of the four row operands: per token, per head
    q_head: cutlass.Int64,
    gate_token: cutlass.Int64,
    gate_head: cutlass.Int64,
    k_token: cutlass.Int64,
    k_head: cutlass.Int64,
    v_token: cutlass.Int64,
    v_head: cutlass.Int64,
    position_row: cutlass.Int64,  # byte stride between the three position rows ("3d"; unread under "1d")
    k_page: cutlass.Int64,  # byte strides of the two pools: per page, per head, per row of a page
    k_page_head: cutlass.Int64,
    k_page_row: cutlass.Int64,
    v_page: cutlass.Int64,
    v_page_head: cutlass.Int64,
    v_page_row: cutlass.Int64,
    num_slots: cutlass.Int32,  # entries of slot_mapping: a token at or past it writes no pool row
    pool_slots: cutlass.Int64,  # pages x PAGE_SIZE of the smaller pool: a slot at or past it writes nothing
):
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()  # the first statement: in front of this launch's own wait (the config pins PDL_RELEASE -> PDL)
    tid, _, _ = cute.arch.thread_idx()
    token, head, _ = cute.arch.block_idx()
    lane = tid % WARP
    warp = tid // WARP
    pair = tid.to(INT64) * INT64(PAIR_BYTES)  # the byte offset of the thread's two elements inside a head row
    token64 = token.to(INT64)
    head64 = head.to(INT64)
    is_kv = head >= cfg.HEADS  # block-uniform: the program of this thread block
    # the weights are constants of the layer: their one load sits BEFORE the dependent-launch wait, every other access behind it
    weight = k_weight if is_kv else q_weight
    ww = ld_global(weight + pair, INT32)
    if cutlass.const_expr(cfg.PDL):
        wait_on_dependent_grids()
    slots = cutlass.memory.SmemAllocator().allocate_tensor(FP32, cute.make_layout(cfg.WARPS), byte_alignment=16)

    # the query program's addresses; a key / value block replaces them (and reads its token's slot)
    out_row = (token64 * INT64(cfg.HEADS) + head64) * INT64(cfg.HEAD_DIM * ELEMENT_BYTES) + pair
    in_addr = q + token64 * q_token + head64 * q_head + pair
    copy_addr = gate + token64 * gate_token + head64 * gate_head + pair
    out_addr = q_out + out_row
    copy_out = gate_out + out_row
    live = cutlass.Boolean(True)
    if is_kv:
        kv_head = (head - cfg.HEADS).to(INT64)
        in_addr = k + token64 * k_token + kv_head * k_head + pair
        copy_addr = v + token64 * v_token + kv_head * v_head + pair
        slot = INT64(-1)  # a token past the slot vector's length writes no pool row
        if token < num_slots:
            slot = _load_int(slot_mapping + token64 * INT64(INT_BYTES[cfg.SLOT_DTYPE]), cfg.SLOT_DTYPE == "int64")
        live = (slot >= INT64(0)) & (slot < pool_slots)
        # the page size is a compile-time power of two: the divide is a shift and the remainder a mask, on the 64-bit slot.  A dead
        # slot's page and row are garbage that its predicated stores never use.
        page = slot >> INT64(cfg.PAGE_SIZE.bit_length() - 1)
        page_row = slot & INT64(cfg.PAGE_SIZE - 1)
        out_addr = k_pool + page * k_page + kv_head * k_page_head + page_row * k_page_row + pair
        copy_out = v_pool + page * v_page + kv_head * v_page_head + page_row * v_page_row + pair

    xw = ld_global(in_addr, INT32)  # the thread's two row elements
    cw = ld_global(copy_addr, INT32)  # the copied pair (gate / V): a bit copy
    x0, x1 = f16x2_to_f32(xw, dtype=BF16)
    # byte contract: ONE fp32 multiply and ONE fused multiply-add per thread -- explicit, never the optimizer's contraction
    sumsq = cute.math.fma(x0, x0, x1 * x1)
    # byte contract: the row total in the DESCENDING order (butterfly offsets 16, 8, 4, 2, 1 inside each warp, then WARPS / 2 .. 1 over
    # the warp sums).  The call holds the ONE thread-block barrier: all cfg.THREADS threads reach it here, outside any run-time branch.
    total = block_reduce(sumsq, lane, warp, slots, cfg.WARPS, kind="sum", order="descending")
    # byte contract: div.full with the width in a register, one addition, rsqrt.approx.ftz
    width = opaque_f32_bits(struct.unpack("<I", struct.pack("<f", float(cfg.HEAD_DIM)))[0])
    inv_rms = rsqrt_approx_ftz_f32(div_full_f32(total, width) + FP32(cfg.EPS))
    w0, w1 = f16x2_to_f32(ww, dtype=BF16)
    # the bf16 rounding of record: the pass-through elements store this word, the rotation reads it widened back
    ow = fp32_to_fp16((x0 * inv_rms) * (w0 + FP32(1)), (x1 * inv_rms) * (w1 + FP32(1)), dtype=BF16)

    if warp == 0:  # the rotary part is exactly warp 0 (the config pins HALF == 32): lanes below HALF / 2 hold the first half
        group = cfg.HALF // PAIR  # lanes per rotary half
        n0, n1 = f16x2_to_f32(ow, dtype=BF16)
        p0 = cute.arch.shuffle_sync_bfly(n0, offset=group)  # the partner element r +- HALF: the same pair position, `group` lanes away
        p1 = cute.arch.shuffle_sync_bfly(n1, offset=group)
        j = lane % group  # the thread's rotary indices are r = 2 j and 2 j + 1, in either half
        position_bytes = INT_BYTES[cfg.POSITION_DTYPE]
        wide = cfg.POSITION_DTYPE == "int64"
        at_token = positions + token64 * INT64(position_bytes)
        pos0 = _load_int(at_token, wide)
        pos1 = pos0
        if cutlass.const_expr(cfg.POSITIONS == "3d"):
            # the position row of a rotary index is a compile-time fact of the index: a mask over j per element of the pair
            (h0_mask, w0_mask), (h1_mask, w1_mask) = position_row_masks(cfg)
            pos_h = _load_int(at_token + position_row, wide)
            pos_w = _load_int(at_token + position_row + position_row, wide)
            pos_t = pos0
            pos0 = pos_h if ((INT32(h0_mask) >> j) & 1) != 0 else (pos_w if ((INT32(w0_mask) >> j) & 1) != 0 else pos_t)
            pos1 = pos_h if ((INT32(h1_mask) >> j) & 1) != 0 else (pos_w if ((INT32(w1_mask) >> j) & 1) != 0 else pos_t)
        table_row_bytes = INT64(cfg.ROTARY_DIM * ELEMENT_BYTES)
        in_row = (j * PAIR_BYTES).to(INT64)  # the table word that holds the pair of rotary indices 2 j, 2 j + 1
        sin_half = INT64(cfg.HALF * ELEMENT_BYTES)
        row0 = cos_sin + pos0 * table_row_bytes + in_row
        cos0, cos1 = f16x2_to_f32(ld_global(row0, INT32), dtype=BF16)
        sin0, sin1 = f16x2_to_f32(ld_global(row0 + sin_half, INT32), dtype=BF16)
        if cutlass.const_expr(cfg.POSITIONS == "3d"):  # the second index can read another position row: its own two table words
            row1 = cos_sin + pos1 * table_row_bytes + in_row
            _, cos1 = f16x2_to_f32(ld_global(row1, INT32), dtype=BF16)
            _, sin1 = f16x2_to_f32(ld_global(row1 + sin_half, INT32), dtype=BF16)
        # byte contract: each rotated value is ONE fp32 multiply (the rounded product) and ONE fused multiply-add.
        #   first half:            out[r]        = fma(n[r], cos, -(n[r + HALF] x sin))
        #   second half, "3d":     out[r + HALF] = fma(n[r + HALF], cos, n[r] x sin)
        #   second half, "1d":     out[r + HALF] = fma(n[r], sin, n[r + HALF] x cos)        (the OTHER product is the fused one)
        # The two orders of the second half agree wherever both products are exact in fp32 and differ where one underflows.
        first = lane < group
        flip = INT32(SIGN_BIT) if first else INT32(0)  # the rounded product's sign bit, flipped in the first half (exact)
        if cutlass.const_expr(cfg.POSITIONS == "3d"):
            # fma(own, cos, t) with t = partner x sin in both halves
            t0 = ((p0 * sin0).bitcast(INT32) ^ flip).bitcast(FP32)
            t1 = ((p1 * sin1).bitcast(INT32) ^ flip).bitcast(FP32)
            ow = fp32_to_fp16(cute.math.fma(n0, cos0, t0), cute.math.fma(n1, cos1, t1), dtype=BF16)
        else:
            # first half: fma(own, cos, -(partner x sin)); second half: fma(partner, sin, own x cos) -- selects of operands, no branch
            t0 = (((p0 if first else n0) * (sin0 if first else cos0)).bitcast(INT32) ^ flip).bitcast(FP32)
            t1 = (((p1 if first else n1) * (sin1 if first else cos1)).bitcast(INT32) ^ flip).bitcast(FP32)
            ow = fp32_to_fp16(
                cute.math.fma(n0 if first else p0, cos0 if first else sin0, t0),
                cute.math.fma(n1 if first else p1, cos1 if first else sin1, t1),
                dtype=BF16,
            )

    if live:
        st_global(out_addr, ow, INT32)
        st_global(copy_out, cw, INT32)


frost_attention_prepare.set_name_prefix("cudnn", remove_cutlass_symbol=True)

#: the kernel's operand addresses and byte strides, in launch order (the host entry builds a launch's arguments by these names)
OPERANDS = ("q", "gate", "k", "v", "q_weight", "k_weight", "cos_sin", "positions", "k_pool", "v_pool", "slot_mapping", "q_out", "gate_out")
STRIDES = (
    "q_token",
    "q_head",
    "gate_token",
    "gate_head",
    "k_token",
    "k_head",
    "v_token",
    "v_head",
    "position_row",
    "k_page",
    "k_page_head",
    "k_page_row",
    "v_page",
    "v_page_head",
    "v_page_row",
)


@cute.jit
def launch(
    cfg: cutlass.Constexpr,
    q: cutlass.Int64,
    gate: cutlass.Int64,
    k: cutlass.Int64,
    v: cutlass.Int64,
    q_weight: cutlass.Int64,
    k_weight: cutlass.Int64,
    cos_sin: cutlass.Int64,
    positions: cutlass.Int64,
    k_pool: cutlass.Int64,
    v_pool: cutlass.Int64,
    slot_mapping: cutlass.Int64,
    q_out: cutlass.Int64,
    gate_out: cutlass.Int64,
    q_token: cutlass.Int64,
    q_head: cutlass.Int64,
    gate_token: cutlass.Int64,
    gate_head: cutlass.Int64,
    k_token: cutlass.Int64,
    k_head: cutlass.Int64,
    v_token: cutlass.Int64,
    v_head: cutlass.Int64,
    position_row: cutlass.Int64,
    k_page: cutlass.Int64,
    k_page_head: cutlass.Int64,
    k_page_row: cutlass.Int64,
    v_page: cutlass.Int64,
    v_page_head: cutlass.Int64,
    v_page_row: cutlass.Int64,
    num_slots: cutlass.Int32,
    pool_slots: cutlass.Int64,
    rows: cutlass.Int32,
    stream: cuda.CUstream,
):
    """One launch: grid ``(rows, HEADS + KV_HEADS, 1)``, ``cfg.THREADS`` threads, the programmatic-stream-serialization attribute
    under ``cfg.PDL``.  ``rows`` (the tokens), every address and every stride are runtime arguments; the config is the compile key."""
    frost_attention_prepare(
        cfg,
        q,
        gate,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        k_pool,
        v_pool,
        slot_mapping,
        q_out,
        gate_out,
        q_token,
        q_head,
        gate_token,
        gate_head,
        k_token,
        k_head,
        v_token,
        v_head,
        position_row,
        k_page,
        k_page_head,
        k_page_row,
        v_page,
        v_page_head,
        v_page_row,
        num_slots,
        pool_slots,
    ).launch(grid=(rows, cfg.PROGRAMS, 1), block=(cfg.THREADS, 1, 1), stream=stream, use_pdl=cfg.PDL)
