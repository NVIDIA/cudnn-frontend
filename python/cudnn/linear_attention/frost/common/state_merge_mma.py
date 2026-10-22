# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operand-selectable affine state chains; no summary generation or communication.

FP32 storage and accumulators throughout. Only the matrix operands are rounded.
TF32x3 uses hi*hi + hi*lo + lo*hi, with residuals taken from FP32 masters.
Warp fragment mapping follows PTX ISA's mma.m16n8k8 layouts.
"""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16


@cute.jit
def round_tf32(value):
    return cute.arch.inline_ptx(
        "cvt.rn.tf32.f32 $0, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[value],
    ).bitcast(cutlass.Float32)


@cute.jit
def mma_tf32(a, b, c):
    # Pass accumulator bits, not floating constants, through the CuTe DSL FFI.
    return cute.arch.inline_ptx(
        "{ .reg .f32 fc<4>; "
        "mov.b32 fc0, {$r6}; mov.b32 fc1, {$r7}; mov.b32 fc2, {$r8}; mov.b32 fc3, {$r9}; "
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{{$w0},{$w1},{$w2},{$w3}}, {{$r0},{$r1},{$r2},{$r3}}, {{$r4},{$r5}}, {fc0,fc1,fc2,fc3}; }",
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[x.bitcast(cutlass.Int32) for x in a + b + list(c)],
    )


@cute.jit
def mma_bf16(a, b, c):
    return cute.arch.inline_ptx(
        "{ .reg .f32 fc<4>; "
        "mov.b32 fc0, {$r3}; mov.b32 fc1, {$r4}; mov.b32 fc2, {$r5}; mov.b32 fc3, {$r6}; "
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{{$w0},{$w1},{$w2},{$w3}}, {{$r0},{$r1}}, {{$r2}}, {fc0,fc1,fc2,fc3}; }",
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[
            fp32_to_fp16(a[0], a[1], dtype=cutlass.BFloat16),
            fp32_to_fp16(a[2], a[3], dtype=cutlass.BFloat16),
            fp32_to_fp16(b[0], b[1], dtype=cutlass.BFloat16),
        ]
        + [x.bitcast(cutlass.Int32) for x in c],
    )


@cute.kernel
def frost_state_merge_mma(
    heads_out: cutlass.Int32,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    transpose: cutlass.Constexpr[bool],
    has_seed: cutlass.Constexpr[bool],
    has_tail: cutlass.Constexpr[bool],
    emit_summary: cutlass.Constexpr[bool],
    precision: cutlass.Constexpr[str],
    mH: cute.Tensor | None,
    mM: cute.Tensor,
    mX: cute.Tensor | None,
    mSeed: cute.Tensor | None,
    mTail: cute.Tensor | None,
    mSummaryM: cute.Tensor | None,
    mMainRows: cute.Tensor | None,
    mSeedIndices: cute.Tensor | None,
):
    """Four warps compute independent 16x8 output tiles, ping-ponging FP32 state.

    CTAs walk either state rows or transition-product rows. The CTA barrier at
    each piece publishes the next master state before any warp consumes it.
    All warp lanes participate in every MMA, including zero-seeded chains.
    """
    wait_on_dependent_grids()
    K = cutlass.const_expr(dim_k)
    V = cutlass.const_expr(dim_v)
    walk_state = cutlass.const_expr(has_seed or has_tail or not emit_summary)
    state_rows = cutlass.const_expr(V if walk_state else 0)
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp, lane = tid // 32, tid % 32
    group, thread = lane // 4, lane % 4
    seq, head, block = cute.arch.block_idx()
    row_base = cutlass.Int32(block) * 16
    is_state = row_base < state_rows
    row_base = row_base if is_state else row_base - state_rows
    seed_seq = cutlass.Int32(seq)
    if cutlass.const_expr(mSeedIndices is not None):
        seed_seq = cutlass.Int32(mSeedIndices[seq])
    seq_base = cutlass.Int32(seq) * pieces
    count = pieces
    if cutlass.const_expr(mMainRows is not None):
        seq_base = cutlass.Int32(mMainRows[seq]) // heads_out
        count = (cutlass.Int32(mMainRows[seq + 1]) - cutlass.Int32(mMainRows[seq])) // heads_out
        count = cutlass.max(count, cutlass.Int32(1))
    copy_only = count < 0
    if cutlass.const_expr(mMainRows is not None and not (has_tail or emit_summary)):
        copy_only = count == 1

    state = cutlass.Array(cutlass.Float32, 2 * 16 * K, space=cutlass.AddressSpace.smem, alignment=16)
    for part in cutlass.range_constexpr(16 * K // 128):
        flat = tid + part * 128
        row, col = row_base + flat // K, flat % K
        value = cutlass.Float32(0)
        if cutlass.const_expr(walk_state and has_seed):
            if is_state:
                value = mSeed[cutlass.Int64(seed_seq), cutlass.Int64(head), cutlass.Int64(row), cutlass.Int64(col)].to(cutlass.Float32)
        if cutlass.const_expr(emit_summary):
            if not is_state:
                value = cutlass.Float32(1) if row == col else cutlass.Float32(0)
        state[flat] = value
    nvvm.barrier_cta_sync()

    for step_idx in range(count):
        src = (step_idx % 2) * (16 * K)
        dst = ((step_idx + 1) % 2) * (16 * K)
        step = count - 1 - step_idx if cutlass.const_expr(transpose) else step_idx
        slot = seq_base + step
        if cutlass.const_expr(walk_state):
            if is_state:
                for part in cutlass.range_constexpr(16 * K // 128):
                    flat = tid + part * 128
                    mX[cutlass.Int64(slot), cutlass.Int64(head), cutlass.Int64(row_base + flat // K), cutlass.Int64(flat % K)] = state[src + flat]
        if not copy_only:
            for tile in cutlass.range_constexpr(K // 32):
                n0 = warp * 8 + tile * 32
                acc = (cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0))
                cross = (cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0))
                for kb in range(K // 8):
                    k0 = kb * 8
                    if cutlass.const_expr(precision == "bf16"):
                        aa = [state[src + (group + (j // 2) * 8) * K + k0 + thread * 2 + j % 2] for j in range(4)]
                        kk = [k0 + thread * 2, k0 + thread * 2 + 1]
                    else:
                        aa = [state[src + (group + (j % 2) * 8) * K + k0 + thread + (j // 2) * 4] for j in range(4)]
                        kk = [k0 + thread, k0 + thread + 4]
                    bb = [cutlass.Float32(0), cutlass.Float32(0)]
                    for j in cutlass.range_constexpr(2):
                        mr = cutlass.Int64(n0 + group) if cutlass.const_expr(transpose) else cutlass.Int64(kk[j])
                        mc = cutlass.Int64(kk[j]) if cutlass.const_expr(transpose) else cutlass.Int64(n0 + group)
                        bb[j] = mM[cutlass.Int64(slot), cutlass.Int64(head), mr, mc]
                    if cutlass.const_expr(precision == "bf16"):
                        acc = mma_bf16(aa, bb, acc)
                    else:
                        ah = [round_tf32(x) for x in aa]
                        bh = [round_tf32(x) for x in bb]
                        if cutlass.const_expr(precision == "tf32x3"):
                            al = [round_tf32(aa[j] - ah[j]) for j in range(4)]
                            bl = [round_tf32(bb[j] - bh[j]) for j in range(2)]
                            cross = mma_tf32(al, bh, cross)
                            cross = mma_tf32(ah, bl, cross)
                        acc = mma_tf32(ah, bh, acc)
                # Preserve small residual products until the whole dot is done.
                # Adding each one directly into the large hi*hi accumulator
                # loses their low bits repeatedly over long K reductions.
                if cutlass.const_expr(precision == "tf32x3"):
                    acc = tuple(acc[j] + cross[j] for j in range(4))
                for j in cutlass.range_constexpr(4):
                    r = group + (j // 2) * 8
                    c = n0 + thread * 2 + j % 2
                    value = acc[j]
                    if cutlass.const_expr(walk_state):
                        if is_state:
                            value = value + mH[cutlass.Int64(slot), cutlass.Int64(head), cutlass.Int64(row_base + r), cutlass.Int64(c)]
                    state[dst + r * K + c] = value
            nvvm.barrier_cta_sync()
        else:
            # The one-piece/no-tail path promises to copy the seed only.
            for part in cutlass.range_constexpr(16 * K // 128):
                flat = tid + part * 128
                state[dst + flat] = state[src + flat]
            nvvm.barrier_cta_sync()

    last = (count % 2) * (16 * K)
    for part in cutlass.range_constexpr(16 * K // 128):
        flat = tid + part * 128
        row, col = cutlass.Int64(row_base + flat // K), cutlass.Int64(flat % K)
        if cutlass.const_expr(has_tail):
            if is_state:
                mTail[cutlass.Int64(seq), cutlass.Int64(head), row, col] = state[last + flat].to(mTail.element_type)
        if cutlass.const_expr(emit_summary):
            if not is_state:
                mSummaryM[cutlass.Int64(seq), cutlass.Int64(head), row, col] = state[last + flat].to(mSummaryM.element_type)
    launch_dependent_grids()


frost_state_merge_mma.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch_state_merge_mma(
    heads_out,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    pieces,
    transpose: cutlass.Constexpr[bool],
    has_seed: cutlass.Constexpr[bool],
    has_tail: cutlass.Constexpr[bool],
    emit_summary: cutlass.Constexpr[bool],
    precision: cutlass.Constexpr[str],
    num_seqs,
    mH,
    mM,
    mX,
    mSeed,
    mTail,
    mSummaryM,
    mMainRows,
    mSeedIndices,
    stream,
):
    walk_state = cutlass.const_expr(has_seed or has_tail or not emit_summary)
    blocks = cutlass.const_expr(((dim_v if walk_state else 0) + (dim_k if emit_summary else 0)) // 16)
    frost_state_merge_mma(
        heads_out, dim_v, dim_k, pieces, transpose, has_seed, has_tail, emit_summary, precision, mH, mM, mX, mSeed, mTail, mSummaryM, mMainRows, mSeedIndices
    ).launch(grid=(num_seqs, heads_out, blocks), block=(128, 1, 1), stream=stream, use_pdl=True)
