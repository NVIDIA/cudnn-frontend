# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream-explicit adapters for the shared CuTe kernels."""

from functools import lru_cache
import importlib

import cutlass
import cutlass.cute as cute

I32 = cutlass.Int32
F32 = cutlass.Float32


@lru_cache(maxsize=None)
def module(name):
    return importlib.import_module(f"{__package__}.kernels.{name}")


@lru_cache(maxsize=128)
def score_kernel(rows, cols):
    return module("indexer_forward_topk_tcgen05").SimplifiedSparseScore(rows, cols, False)


@cute.jit
def score(stream, q, k, out, *, kernel):
    kernel(q, k, out, stream)


@lru_cache(maxsize=128)
def topk_kernel(cols, topk):
    top = module("cute_dsl_filter_topk.filtered_top_k_prefill_varlen")
    wide = 49152 <= cols < 131072
    return top.FilteredTopKKernelVarlenPrefill(
        F32,
        top._bucket_num_cols(cols),
        topk,
        overflow_policy="REREAD_ALWAYS",
        in_kernel_sort=True,
        ssa_epilogue=True,
        subtract_row_start=False,
        boundary_classify=False,
        reg_cache=cols <= 8192,
        cache_smem_values=cols >= 131072,
        num_threads=1024 if wide else 512,
        target_blocks_per_sm=2 if wide else 4,
        architecture="sm_100",
        unroll_factor=1,
    )


@cute.jit
def select(stream, scores, starts, ends, indices, predict, *, kernel, blocks):
    kernel(scores, starts, ends, None, indices, None, predict, None, stream=stream, min_blocks_per_mp=blocks)


@cute.jit
def repair_score(stream, q, k, out, *, kernel):
    kernel(q, k, out, stream=stream, epilogue_op=lambda x: x * (128.0**-0.5))


@lru_cache(maxsize=128)
def filtered_kernels(rows, cols, topk, sms):
    fused = module("indexer_forward_topk_fused_tcgen05")
    capacity = fused._tie_cap(topk, cols)
    sampler = fused._SimplifiedSparseSampleThresholdK1(topk, capacity, min(8, max(1, sms // (rows // 128))))
    collect = fused._SimplifiedSparseFilteredCollect(rows, rows, cols, topk)
    selector = fused._SimplifiedSparseThresholdSelect(rows, topk, cols)
    return sampler, collect, selector, (capacity + 15) // 16 * 16


@cute.jit
def filtered(
    stream,
    q,
    k,
    starts,
    ends,
    indices,
    predict,
    flags,
    hist,
    thr,
    bnd,
    cursors,
    candidx,
    candval,
    parts,
    counts,
    pred,
    *,
    sampler,
    collector,
    selector,
    rows,
    cols,
):
    sampler(q, k, starts, ends, thr, bnd, parts, counts, I32(rows), I32(cols // 256), stream)
    collector(q, k, hist, thr, bnd, starts, ends, cursors, candidx, candval, I32(rows), I32(cols // 256), stream)
    selector.select_single(candidx, candval, cursors, thr, starts, ends, indices, predict, flags, pred, I32(rows), stream)


@cute.jit
def attention(stream, q, k, v, indices, lengths, out, lse, teacher, maxima, inv_sum, *, launcher, heads, topk, rows, keys, sms, plan):
    args = (q, k, v, indices, lengths, out, lse, teacher, maxima, inv_sum)
    if cutlass.const_expr(topk == 512 and heads == 16):
        launcher(*args, I32(rows), I32(keys), I32(heads * 256), I32(256), I32(256), stream)
    elif cutlass.const_expr(topk == 512 and heads == 32):
        launcher(*args, I32(heads * 256), I32(256), I32(256), I32(rows), I32(keys), stream)
    elif cutlass.const_expr(topk == 512):
        launcher(*args, I32(heads * 256), I32(256), I32(256), I32(rows), I32(keys), I32(min(rows, sms)), stream)
    elif cutlass.const_expr(heads == 96):
        launcher(*args, I32(heads * 256), I32(256), I32(256), I32(rows), I32(keys), I32(min(rows, sms)), heads, topk, stream)
    elif cutlass.const_expr(heads == 32 and topk == 1024):
        launcher(
            *args,
            I32(heads * 256),
            I32(256),
            I32(256),
            I32(heads * 256),
            I32(heads),
            I32(rows),
            I32(keys),
            I32(min(rows, sms)),
            I32(0),
            F32(1.0 / heads),
            stream,
            HG=plan[0],
            NHG=plan[1],
            NT=plan[2],
            TCOLS=plan[3],
            SMEM=plan[4],
            ONL=plan[5],
            TOPK=topk,
        )
    else:
        launcher(
            *args,
            I32(heads * 256),
            I32(256),
            I32(256),
            I32(heads * 256),
            I32(heads),
            I32(rows),
            I32(keys),
            I32(min(rows, sms)),
            I32(0),
            F32(1.0 / heads),
            stream,
            HG=plan[0],
            NHG=plan[1],
            NT=plan[2],
            TCOLS=plan[3],
            SMEM=plan[4],
            ONL=plan[5],
            KST=plan[6],
            TOPK=topk,
        )


@cute.jit
def auxiliary(
    stream, q, k, indices, lengths, predict, teacher, denominator, loss, dq, dk, dz, partial, q_acc, k_acc, *, launcher, topk, coefficient, rectangular
):
    launcher(q, k, indices, lengths, predict, teacher, denominator, F32(coefficient), loss, dq, dk, dz, partial, q_acc, k_acc, rectangular, False, topk, stream)


@cute.jit
def auxiliary_pre(stream, k, indices, lengths, predict, teacher, denominator, dz, partial, q_acc, dq, counts, *, launcher, topk, coefficient):
    launcher(k, indices, lengths, predict, teacher, denominator, F32(coefficient), dz, partial, q_acc, dq, counts, topk, stream)


@cute.jit
def auxiliary_post(stream, q, indices, lengths, dz, offsets, partial, q_acc, cursor, dq, dk, loss, edge_rows, edge_dz, k_acc, *, launcher, topk):
    launcher(q, indices, lengths, dz, cursor, offsets, edge_rows, edge_dz, partial, q_acc, k_acc, dq, dk, loss, topk, False, stream)


@cute.jit
def backward(stream, q, k, v, out, dout, lse, indices, lengths, dq, dk, dv, workspace, *, launcher, heads, topk, rows, keys, sms):
    # The backward kernel's output and saved-state ABI uses flat buffers.
    of = cute.make_tensor(out.iterator, cute.make_layout((rows * heads * 256,)))
    dof = cute.make_tensor(dout.iterator, cute.make_layout((rows * heads * 256,)))
    lf = cute.make_tensor(lse.iterator, cute.make_layout((rows * heads,)))
    ix = cute.make_tensor(indices.iterator, cute.make_layout((rows * topk,)))
    dqf = cute.make_tensor(dq.iterator, cute.make_layout((rows * heads * 256,)))
    dkf = cute.make_tensor(dk.iterator, cute.make_layout((keys * 256,)))
    dvf = cute.make_tensor(dv.iterator, cute.make_layout((keys * 256,)))
    launcher(
        q,
        k,
        v,
        of,
        dof,
        lf,
        ix,
        lengths,
        dqf,
        workspace,
        dkf,
        dvf,
        I32(heads * 256),
        I32(256),
        I32(256),
        I32(rows),
        I32(keys),
        I32(min(rows, sms)),
        heads,
        topk,
        stream,
    )
