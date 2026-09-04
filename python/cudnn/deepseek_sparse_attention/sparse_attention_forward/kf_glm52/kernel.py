import torch
import cuda.bindings.driver as cuda

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def ld_cg_e4m3x2(ptr, *, loc=None, ip=None):
    # L1-bypassing (.cg: cache in L2, not L1) 16-bit load of 2 contiguous
    # fp8 e4m3 scores + decode to f32x2 in one shot. Attacks the L1/TEX
    # throughput limiter by removing the scores-load bytes from the L1 pipe.
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [ptr_i64],
        "{\n\t"
        ".reg .b16 h;\n\t"
        ".reg .b32 packed;\n\t"
        ".reg .f16 f_lo, f_hi;\n\t"
        "ld.global.cg.b16 h, [$2];\n\t"
        "cvt.rn.f16x2.e4m3x2 packed, h;\n\t"
        "mov.b32 {f_lo, f_hi}, packed;\n\t"
        "cvt.f32.f16 $0, f_lo;\n\t"
        "cvt.f32.f16 $1, f_hi;\n\t"
        "}\n",
        "=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip))
    out1 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))
    return out0, out1


@dsl_user_op
def st_cs_e4m3x2(ptr, w0, w1, *, loc=None, ip=None):
    # Streaming (.cs: evict-first, cache in L2 not L1) 16-bit store of 2
    # contiguous fp8 e4m3 probs (write-once, never re-read by softmax).
    # Packs w0->low byte, w1->high byte via cvt.rn.satfinite.e4m3x2.f32,
    # then st.global.cs.b16 removes the probs-store bytes from the L1 pipe.
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64, cutlass.Float32(w0).ir_value(loc=loc, ip=ip),
         cutlass.Float32(w1).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 h;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h, $2, $1;\n\t"
        "st.global.cs.b16 [$0], h;\n\t"
        "}\n",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def softmax_iter_ptx(src_ptr, dst_ptr, c0, c1, c2, c3, *, loc=None, ip=None):
    # Fused hot-iter PTX: load 4 fp8 scores, compute w = count * exp2(score)
    # in f16 space, write 4 fp8 probs, and RETURN the pair-sum (w0+w2, w1+w3)
    # as 2 f32 values via add.f16x2. Halves XU pipe pressure (ex2.f16x2 does
    # 2 exps in 1 op) and halves outer accumulator adds.
    src_i64 = src_ptr.toint(loc=loc, ip=ip).ir_value()
    dst_i64 = dst_ptr.toint(loc=loc, ip=ip).ir_value()
    c0_v = cutlass.Float32(c0).ir_value(loc=loc, ip=ip)
    c1_v = cutlass.Float32(c1).ir_value(loc=loc, ip=ip)
    c2_v = cutlass.Float32(c2).ir_value(loc=loc, ip=ip)
    c3_v = cutlass.Float32(c3).ir_value(loc=loc, ip=ip)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [src_i64, dst_i64, c0_v, c1_v, c2_v, c3_v],
        "{\n\t"
        ".reg .b32 packed;\n\t"
        ".reg .b16 s_lo, s_hi;\n\t"
        ".reg .b32 plo, phi, elo, ehi;\n\t"
        ".reg .b32 ch01, ch23;\n\t"
        ".reg .b32 w01, w23, wsum;\n\t"
        ".reg .b16 out_lo, out_hi;\n\t"
        ".reg .b32 out_packed;\n\t"
        ".reg .f16 fs0, fs1;\n\t"
        "ld.global.cg.b32 packed, [$2];\n\t"
        "mov.b32 {s_lo, s_hi}, packed;\n\t"
        "cvt.rn.f16x2.e4m3x2 plo, s_lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 phi, s_hi;\n\t"
        "ex2.approx.f16x2 elo, plo;\n\t"
        "ex2.approx.f16x2 ehi, phi;\n\t"
        "cvt.rn.f16x2.f32 ch01, $5, $4;\n\t"
        "cvt.rn.f16x2.f32 ch23, $7, $6;\n\t"
        "mul.f16x2 w01, elo, ch01;\n\t"
        "mul.f16x2 w23, ehi, ch23;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 out_lo, w01;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 out_hi, w23;\n\t"
        "mov.b32 out_packed, {out_lo, out_hi};\n\t"
        "st.global.cs.b32 [$3], out_packed;\n\t"
        "add.f16x2 wsum, w01, w23;\n\t"
        "mov.b32 {fs0, fs1}, wsum;\n\t"
        "cvt.f32.f16 $0, fs0;\n\t"
        "cvt.f32.f16 $1, fs1;\n\t"
        "}\n",
        "=f,=f,l,l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    ws0 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip))
    ws1 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))
    return ws0, ws1


@dsl_user_op
def ld_cg_e4m3x4(ptr, *, loc=None, ip=None):
    # L1-bypassing 32-bit load of 4 contiguous fp8 e4m3 scores + decode to
    # f32x4. Halves memory-op issue vs the 2-wide helper on the issue-bound
    # softmax loop. byte0->r0 (lowest addr) .. byte3->r3.
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [ptr_i64],
        "{\n\t"
        ".reg .b32 packed32;\n\t"
        ".reg .b16 lo, hi;\n\t"
        ".reg .b32 plo, phi;\n\t"
        ".reg .f16 f0, f1, f2, f3;\n\t"
        "ld.global.cg.b32 packed32, [$4];\n\t"
        "mov.b32 {lo, hi}, packed32;\n\t"
        "cvt.rn.f16x2.e4m3x2 plo, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 phi, hi;\n\t"
        "mov.b32 {f0, f1}, plo;\n\t"
        "mov.b32 {f2, f3}, phi;\n\t"
        "cvt.f32.f16 $0, f0;\n\t"
        "cvt.f32.f16 $1, f1;\n\t"
        "cvt.f32.f16 $2, f2;\n\t"
        "cvt.f32.f16 $3, f3;\n\t"
        "}\n",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    r0 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip))
    r1 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))
    r2 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [2], loc=loc, ip=ip))
    r3 = cutlass.Float32(llvm.extractvalue(T.f32(), out, [3], loc=loc, ip=ip))
    return r0, r1, r2, r3


@dsl_user_op
def st_cs_e4m3x4(ptr, w0, w1, w2, w3, *, loc=None, ip=None):
    # Streaming 32-bit store of 4 contiguous fp8 e4m3 probs. Two
    # cvt.rn.satfinite.e4m3x2.f32 pack {w0,w1}->lo16, {w2,w3}->hi16 (w0 lowest
    # byte), combined into b32 and st.global.cs.b32 off the L1 pipe.
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64,
         cutlass.Float32(w0).ir_value(loc=loc, ip=ip),
         cutlass.Float32(w1).ir_value(loc=loc, ip=ip),
         cutlass.Float32(w2).ir_value(loc=loc, ip=ip),
         cutlass.Float32(w3).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        ".reg .b32 packed;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 packed, {lo, hi};\n\t"
        "st.global.cs.b32 [$0], packed;\n\t"
        "}\n",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cast_bf16x8_to_fp8x8(src_ptr, dst_ptr, *, loc=None, ip=None):
    # Vectorized bf16->fp8(e4m3) cast of 8 contiguous values: read-only v4.b32
    # load, widen bf16->f32 by <<16 (lossless), pack via cvt.e4m3x2.f32, and
    # streaming (.cs) 64-bit store to keep the write off the L1 pipe.
    src_i64 = src_ptr.toint(loc=loc, ip=ip).ir_value()
    dst_i64 = dst_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [src_i64, dst_i64],
        "{\n\t"
        ".reg .b32 r0, r1, r2, r3;\n\t"
        ".reg .b32 tlo, thi;\n\t"
        ".reg .f32 flo, fhi;\n\t"
        ".reg .b16 h0, h1, h2, h3;\n\t"
        ".reg .b32 d0, d1;\n\t"
        ".reg .b64 packed64;\n\t"
        "ld.global.nc.v4.b32 {r0, r1, r2, r3}, [$0];\n\t"
        "and.b32 tlo, r0, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r0, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h0, fhi, flo;\n\t"
        "and.b32 tlo, r1, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r1, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h1, fhi, flo;\n\t"
        "and.b32 tlo, r2, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r2, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h2, fhi, flo;\n\t"
        "and.b32 tlo, r3, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r3, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h3, fhi, flo;\n\t"
        "mov.b32 d0, {h0, h1};\n\t"
        "mov.b32 d1, {h2, h3};\n\t"
        "mov.b64 packed64, {d0, d1};\n\t"
        "st.global.cs.b64 [$1], packed64;\n\t"
        "}\n",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cast_bf16x16_to_fp8x16(src_ptr, dst_ptr, *, loc=None, ip=None):
    # Two adjacent x8 casts per thread. This keeps the same shl-widen numeric
    # path as the best candidate while halving Q-cast thread/block count.
    src_i64 = src_ptr.toint(loc=loc, ip=ip).ir_value()
    dst_i64 = dst_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [src_i64, dst_i64],
        "{\n\t"
        ".reg .b32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
        ".reg .b32 tlo, thi;\n\t"
        ".reg .f32 flo, fhi;\n\t"
        ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;\n\t"
        ".reg .b32 d0, d1, d2, d3;\n\t"
        "ld.global.nc.v4.b32 {r0, r1, r2, r3}, [$0];\n\t"
        "ld.global.nc.v4.b32 {r4, r5, r6, r7}, [$0+16];\n\t"
        "and.b32 tlo, r0, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r0, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h0, fhi, flo;\n\t"
        "and.b32 tlo, r1, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r1, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h1, fhi, flo;\n\t"
        "and.b32 tlo, r2, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r2, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h2, fhi, flo;\n\t"
        "and.b32 tlo, r3, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r3, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h3, fhi, flo;\n\t"
        "and.b32 tlo, r4, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r4, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h4, fhi, flo;\n\t"
        "and.b32 tlo, r5, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r5, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h5, fhi, flo;\n\t"
        "and.b32 tlo, r6, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r6, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h6, fhi, flo;\n\t"
        "and.b32 tlo, r7, 0xffff;\n\t"
        "shl.b32 tlo, tlo, 16;\n\t"
        "and.b32 thi, r7, 0xffff0000;\n\t"
        "mov.b32 flo, tlo;\n\t"
        "mov.b32 fhi, thi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h7, fhi, flo;\n\t"
        "mov.b32 d0, {h0, h1};\n\t"
        "mov.b32 d1, {h2, h3};\n\t"
        "mov.b32 d2, {h4, h5};\n\t"
        "mov.b32 d3, {h6, h7};\n\t"
        "st.global.cs.v4.b32 [$1], {d0, d1, d2, d3};\n\t"
        "}\n",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


THREADS = 256
QCAST_THREADS = 512
SOFTMAX_THREADS = 128
HEADS = 64
SOFTMAX_HEADS = 32
SOFTMAX_GROUPS = HEADS // SOFTMAX_HEADS
DK = 576
DV = 512
TOPK = 2048
HALF_TOPK = 1024
LOG2E_OVER_24 = 1.4426950408889634 / 24.0
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@cute.kernel
def _copy_boundary_kv_fp8(kv: cute.Tensor, pool_k_fp8: cute.Tensor, pool_v_fp8: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gtid = bidx * THREADS + tidx
    total = TOPK * DK
    seqlen = cutlass.Int32(kv.shape[0])

    if gtid < total:
        slot = gtid // DK
        d = gtid - slot * DK
        src_row = slot - HALF_TOPK
        if slot < HALF_TOPK:
            src_row = seqlen - HALF_TOPK + slot
        val = kv[src_row, d]
        pool_k_fp8[slot, d] = val.to(cutlass.Float8E4M3FN)
        if d < DV:
            pool_v_fp8[slot, d] = val.to(cutlass.Float8E4M3FN)


@cute.jit
def _launch_copy_boundary_kv_fp8(kv: cute.Tensor, pool_k_fp8: cute.Tensor, pool_v_fp8: cute.Tensor):
    blocks = cute.ceil_div(TOPK * DK, THREADS)
    _copy_boundary_kv_fp8(kv, pool_k_fp8, pool_v_fp8).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1))


VEC_PER_ROW_X8 = DK // 8    # 576/8 = 72 vec-groups per row
VEC_PER_ROW_X16 = DK // 16  # 576/16 = 36 vec-groups per row
VEC_PER_ROW_X32 = DK // 32  # 576/32 = 18 vec-groups per row


@cute.kernel
def _cast_q_fp8_x8(q: cute.Tensor, q_fp8: cute.Tensor, n_vec: cutlass.Constexpr, dkp: cutlass.Constexpr):
    # cast 8 contiguous bf16 -> 8 fp8; source contiguous (rows,DK), dest
    # row-strided (rows,dkp) with pad cols pre-zeroed. Row-aware mapping.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gtid = bidx * QCAST_THREADS + tidx
    if gtid < n_vec:
        row = gtid // VEC_PER_ROW_X8
        vin = gtid - row * VEC_PER_ROW_X8
        src_off = cutlass.Int64(row) * DK + cutlass.Int64(vin) * 8
        dst_off = cutlass.Int64(row) * dkp + cutlass.Int64(vin) * 8
        cast_bf16x8_to_fp8x8(q.iterator + src_off, q_fp8.iterator + dst_off)


@cute.jit
def _launch_cast_q_fp8_x8(q: cute.Tensor, q_fp8: cute.Tensor):
    rows = q.shape[0]
    dkp = q_fp8.shape[1]
    n_vec = rows * VEC_PER_ROW_X8
    blocks = cute.ceil_div(n_vec, QCAST_THREADS)
    _cast_q_fp8_x8(q, q_fp8, n_vec, dkp).launch(grid=(blocks, 1, 1), block=(QCAST_THREADS, 1, 1))


@cute.kernel
def _cast_q_fp8_x16(q: cute.Tensor, q_fp8: cute.Tensor, n_vec: cutlass.Constexpr, dkp: cutlass.Constexpr):
    # cast 16 contiguous bf16 -> 16 fp8; source contiguous (rows,DK), dest
    # row-strided (rows,dkp) with pad cols pre-zeroed. Row-aware mapping.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gtid = bidx * QCAST_THREADS + tidx
    if gtid < n_vec:
        row = gtid // VEC_PER_ROW_X16
        vin = gtid - row * VEC_PER_ROW_X16
        src_off = cutlass.Int64(row) * DK + cutlass.Int64(vin) * 16
        dst_off = cutlass.Int64(row) * dkp + cutlass.Int64(vin) * 16
        cast_bf16x16_to_fp8x16(q.iterator + src_off, q_fp8.iterator + dst_off)


@cute.jit
def _launch_cast_q_fp8_x16(q: cute.Tensor, q_fp8: cute.Tensor):
    rows = q.shape[0]
    dkp = q_fp8.shape[1]
    n_vec = rows * VEC_PER_ROW_X16
    blocks = cute.ceil_div(n_vec, QCAST_THREADS)
    _cast_q_fp8_x16(q, q_fp8, n_vec, dkp).launch(grid=(blocks, 1, 1), block=(QCAST_THREADS, 1, 1))


@cute.kernel
def _cast_q_fp8_x32(q: cute.Tensor, q_fp8: cute.Tensor, n_vec: cutlass.Constexpr, dkp: cutlass.Constexpr):
    # cast 32 contiguous bf16 -> 32 fp8 as two adjacent x16 chunks; source
    # contiguous (rows,DK), dest row-strided (rows,dkp) with zero pad cols.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gtid = bidx * QCAST_THREADS + tidx
    if gtid < n_vec:
        row = gtid // VEC_PER_ROW_X32
        vin = gtid - row * VEC_PER_ROW_X32
        src_off = cutlass.Int64(row) * DK + cutlass.Int64(vin) * 32
        dst_off = cutlass.Int64(row) * dkp + cutlass.Int64(vin) * 32
        cast_bf16x16_to_fp8x16(q.iterator + src_off, q_fp8.iterator + dst_off)
        cast_bf16x16_to_fp8x16(q.iterator + src_off + 16, q_fp8.iterator + dst_off + 16)


@cute.jit
def _launch_cast_q_fp8_x32(q: cute.Tensor, q_fp8: cute.Tensor):
    rows = q.shape[0]
    dkp = q_fp8.shape[1]
    n_vec = rows * VEC_PER_ROW_X32
    blocks = cute.ceil_div(n_vec, QCAST_THREADS)
    _cast_q_fp8_x32(q, q_fp8, n_vec, dkp).launch(grid=(blocks, 1, 1), block=(QCAST_THREADS, 1, 1))


@cute.kernel
def _count_indices(idxs: cute.Tensor, counts: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()

    alloc = SmemAllocator()
    s_counts = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(TOPK), byte_alignment=16)

    for slot in cutlass.range(tidx, TOPK, THREADS, unroll=8):
        s_counts[slot] = cutlass.Int32(0)
    cute.arch.sync_threads()

    for j in cutlass.range(tidx, TOPK, THREADS, unroll=1):
        raw = idxs[token, j]
        slot = raw + HALF_TOPK
        cute.arch.atomic_add(s_counts.iterator + slot, cutlass.Int32(1), scope="cta")
    cute.arch.sync_threads()

    for slot in cutlass.range(tidx, TOPK, THREADS, unroll=8):
        counts[token, slot] = s_counts[slot].to(cutlass.Int8)


@cute.jit
def _launch_count_indices(idxs: cute.Tensor, counts: cute.Tensor):
    _count_indices(idxs, counts).launch(
        grid=(idxs.shape[0], 1, 1),
        block=(THREADS, 1, 1),
        min_blocks_per_mp=1,
    )


@cute.kernel
def _softmax_fp8_inplace(
    scores: cute.Tensor,
    counts: cute.Tensor,
    inv_sums: cute.Tensor,
    lse: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    group, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    token = group // SOFTMAX_GROUPS
    head_group = group - token * SOFTMAX_GROUPS
    head_base = head_group * SOFTMAX_HEADS

    alloc = SmemAllocator()
    s_counts = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(TOPK), byte_alignment=16)

    for slot in cutlass.range(tidx, TOPK, SOFTMAX_THREADS, unroll=16):
        s_counts[slot] = counts[token, slot].to(cutlass.Float32)
    cute.arch.sync_threads()

    for h_base in cutlass.range_constexpr(0, SOFTMAX_HEADS, SOFTMAX_THREADS // 32):
        head = head_base + h_base + warp
        row = token * HEADS + head
        score_base = cutlass.Int64(row) * TOPK

        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)
        # f16 hot loop PTX with in-PTX pair-sum: 2 accs outside.
        for base in cutlass.range(lane * 4, TOPK, 128, unroll=8):
            c0 = s_counts[base]
            c1 = s_counts[base + 1]
            c2 = s_counts[base + 2]
            c3 = s_counts[base + 3]
            score_ptr = scores.iterator + score_base + cutlass.Int64(base)
            ws0, ws1 = softmax_iter_ptx(score_ptr, score_ptr, c0, c1, c2, c3)
            acc0 = acc0 + ws0
            acc1 = acc1 + ws1

        local_sum = acc0 + acc1
        sum_val = cute.arch.warp_reduction_sum(local_sum)
        if lane == 0:
            inv = cute.arch.rcp_approx(sum_val)
            (inv_sums.iterator + cutlass.Int64(row)).store(inv)
            # approximate log2 via SFU, scaled to natural log: log(x)=log2(x)*ln2
            lse[token, head] = cute.log2(sum_val, fastmath=True) * LN2


@cute.jit
def _launch_softmax_fp8_inplace(
    scores: cute.Tensor,
    counts: cute.Tensor,
    inv_sums: cute.Tensor,
    lse: cute.Tensor,
):
    _softmax_fp8_inplace(scores, counts, inv_sums, lse).launch(
        grid=(scores.shape[1] // SOFTMAX_HEADS, 1, 1),
        block=(SOFTMAX_THREADS, 1, 1),
        min_blocks_per_mp=1,
    )


_CACHE = {}
_STREAMS = {}
_BUFS = {}


def run(q, kv, idxs, out, lse) -> None:
    seqlen = int(q.shape[0])
    rows = seqlen * HEADS
    device = q.device

    buf_key = (seqlen, device)
    if buf_key not in _BUFS:
        scores_fp8 = torch.empty((1, rows, TOPK), device=device, dtype=torch.float8_e4m3fn)
        counts = torch.empty((seqlen, TOPK), device=device, dtype=torch.int8)
        inv_sums = torch.empty((rows,), device=device, dtype=torch.float32)
        kv_pool_v_fp8 = torch.empty((DV, TOPK), device=device, dtype=torch.float8_e4m3fn).t()
        DKP = 640
        kv_pool_k_fp8 = torch.zeros((TOPK, DKP), device=device, dtype=torch.float8_e4m3fn)
        kv_pool_k_fp8_t = kv_pool_k_fp8.t()
        q_fp8 = torch.zeros((rows, DKP), device=device, dtype=torch.float8_e4m3fn)
        qk_a_scale = torch.tensor(LOG2E_OVER_24, device=device, dtype=torch.float32)
        qk_b_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        b_scale_pv = torch.ones((1, DV), device=device, dtype=torch.float32)
        scores_2d = scores_fp8.squeeze(0)
        _BUFS[buf_key] = (scores_fp8, counts, inv_sums, kv_pool_v_fp8,
                          kv_pool_k_fp8, kv_pool_k_fp8_t, q_fp8, qk_a_scale,
                          qk_b_scale, b_scale_pv, scores_2d)
    else:
        (scores_fp8, counts, inv_sums, kv_pool_v_fp8,
         kv_pool_k_fp8, kv_pool_k_fp8_t, q_fp8, qk_a_scale,
         qk_b_scale, b_scale_pv, scores_2d) = _BUFS[buf_key]

    q_flat = q.reshape(rows, DK)
    kv_cute = from_dlpack(kv, assumed_align=16)
    idxs_cute = from_dlpack(idxs, assumed_align=16)
    counts_cute = from_dlpack(counts, assumed_align=16)
    pool_k_fp8_cute = from_dlpack(kv_pool_k_fp8, assumed_align=16)
    pool_v_fp8_cute = from_dlpack(kv_pool_v_fp8, assumed_align=16)
    scores_cute = from_dlpack(scores_fp8, assumed_align=16)
    inv_sums_cute = from_dlpack(inv_sums, assumed_align=16)
    lse_cute = from_dlpack(lse, assumed_align=16)
    q_flat_cute = from_dlpack(q_flat, assumed_align=16)
    q_fp8_cute = from_dlpack(q_fp8, assumed_align=16)

    out_flat = out.reshape(rows, DV)

    compiled = _CACHE.get(seqlen)
    if compiled is None:
        compiled = (
            cute.compile(_launch_copy_boundary_kv_fp8, kv_cute, pool_k_fp8_cute, pool_v_fp8_cute),
            cute.compile(_launch_count_indices, idxs_cute, counts_cute),
            cute.compile(_launch_softmax_fp8_inplace, scores_cute, counts_cute, inv_sums_cute, lse_cute),
            cute.compile(_launch_cast_q_fp8_x16, q_flat_cute, q_fp8_cute),
        )
        _CACHE[seqlen] = compiled

    M_CHUNK = 4 * 1024 * 1024
    large = rows > M_CHUNK

    if device not in _STREAMS:
        _STREAMS[device] = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
        )

    copy_kernel, count_kernel, softmax_kernel, cast_kernel = compiled
    main_stream = torch.cuda.current_stream()
    count_stream, copy_stream, cast_stream, pv_stream, sm_stream = _STREAMS[device]

    with torch.cuda.stream(count_stream):
        count_kernel(idxs_cute, counts_cute)
    event_count = torch.cuda.Event()
    event_count.record(count_stream)

    with torch.cuda.stream(copy_stream):
        copy_kernel(kv_cute, pool_k_fp8_cute, pool_v_fp8_cute)
    event_copy = torch.cuda.Event()
    event_copy.record(copy_stream)

    if large:
        with torch.cuda.stream(cast_stream):
            cast_kernel(q_flat_cute, q_fp8_cute)
        event_cast = torch.cuda.Event()
        event_cast.record(cast_stream)
        main_stream.wait_event(event_cast)
    else:
        cast_kernel(q_flat_cute, q_fp8_cute)
    main_stream.wait_event(event_copy)

    a_scale = inv_sums.unsqueeze(1)

    if not large:
        torch._scaled_mm(q_fp8, kv_pool_k_fp8_t, scale_a=qk_a_scale, scale_b=qk_b_scale, out_dtype=torch.float8_e4m3fn, out=scores_2d)
        main_stream.wait_event(event_count)
        softmax_kernel(scores_cute, counts_cute, inv_sums_cute, lse_cute)
        torch._scaled_mm(scores_2d, kv_pool_v_fp8, scale_a=a_scale, scale_b=b_scale_pv, out_dtype=torch.bfloat16, out=out_flat)
    else:
        # 2-stream pipeline: QK+PV on main, softmax on sm_stream.
        # Overlaps: QK[i+1] on main runs concurrent with softmax[i] on sm_stream.
        # PV[i] on main after softmax[i] done.
        sm_stream.wait_event(event_count)
        CHUNK_TOKENS = M_CHUNK // HEADS
        chunks = list(range(0, seqlen, CHUNK_TOKENS))
        sm_evts = []
        for i, tok0 in enumerate(chunks):
            tok1 = min(tok0 + CHUNK_TOKENS, seqlen)
            m0 = tok0 * HEADS
            m1 = tok1 * HEADS

            # QK[i] on main
            torch._scaled_mm(q_fp8[m0:m1], kv_pool_k_fp8_t,
                             scale_a=qk_a_scale, scale_b=qk_b_scale,
                             out_dtype=torch.float8_e4m3fn, out=scores_2d[m0:m1])
            qk_evt = torch.cuda.Event()
            qk_evt.record(main_stream)

            scores_slice = from_dlpack(scores_fp8[:, m0:m1, :], assumed_align=16)
            counts_slice = from_dlpack(counts[tok0:tok1], assumed_align=16)
            inv_sums_slice = from_dlpack(inv_sums[m0:m1], assumed_align=16)
            lse_slice = from_dlpack(lse[tok0:tok1], assumed_align=16)
            key = (seqlen, tok1 - tok0)
            sk = _CACHE.setdefault('_sm', {}).get(key)
            if sk is None:
                sk = cute.compile(_launch_softmax_fp8_inplace,
                                  scores_slice, counts_slice, inv_sums_slice, lse_slice)
                _CACHE['_sm'][key] = sk

            # Softmax[i] on sm_stream (waits for QK[i])
            sm_stream.wait_event(qk_evt)
            with torch.cuda.stream(sm_stream):
                sk(scores_slice, counts_slice, inv_sums_slice, lse_slice)
            sm_evt = torch.cuda.Event()
            sm_evt.record(sm_stream)
            sm_evts.append(sm_evt)

        # After all QK/softmax queued, do PV on main (waits for each sm[i]).
        for i, tok0 in enumerate(chunks):
            tok1 = min(tok0 + CHUNK_TOKENS, seqlen)
            m0 = tok0 * HEADS
            m1 = tok1 * HEADS
            main_stream.wait_event(sm_evts[i])
            torch._scaled_mm(scores_2d[m0:m1], kv_pool_v_fp8,
                             scale_a=a_scale[m0:m1], scale_b=b_scale_pv,
                             out_dtype=torch.bfloat16, out=out_flat[m0:m1])
        # Belt-and-suspenders: wait sm_stream too
        sm_final_evt = torch.cuda.Event()
        sm_final_evt.record(sm_stream)
        main_stream.wait_event(sm_final_evt)
