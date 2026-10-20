# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 two-CTA sparse-attention backward for BF16 H128 D576/V512.

QK and dQ include a D64 tensor-core tail. Gather and relay warps feed the
cluster-wide score, dP, dQ, and dKV pipelines. Warp tensor cores compute
each CTA's dK-tail partial with FP32 accumulation and FP32 atomic updates.
Softmax, dS, O*dO, dSink, and the dKV workspace use FP32 arithmetic;
P and dS are converted to BF16 at the tensor-core operand boundary.
Sparse-slot validity is applied before both score and gradient operations,
including the K values retained for dQ. The kernel uses caller-owned
workspace without input repacking.
"""

import math
from typing import Optional, Tuple, Type
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05, warp
from cutlass.cute.typing import BFloat16, Float32, Int32
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, TmemAllocator
from .dsa_bwd_sm100_h128_d576_2cta_deterministic import TwoCTADeterministicReduction, ordered_dkv_quad

U64x4 = Tuple[cutlass.Uint64, cutlass.Uint64, cutlass.Uint64, cutlass.Uint64]
F32x16 = Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]


@dsl_user_op
def _map_smem_to_cluster_rank(smem_ptr: cute.Pointer, peer_rank: Int32, *, loc=None, ip=None) -> Int32:
    """Map a CTA-local shared-memory pointer to another cluster rank."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cpasync_bulk_s2cluster(
    source: cute.Pointer, destination: cute.Pointer, completion_barrier: cute.Pointer, copy_bytes: int | Int32, peer_rank: Int32, *, loc=None, ip=None
) -> None:
    """Issue one shared-to-cluster bulk copy to ``peer_rank``."""
    source_i32 = source.toint(loc=loc, ip=ip).ir_value()
    destination_i32 = _map_smem_to_cluster_rank(destination, peer_rank, loc=loc, ip=ip).ir_value()
    barrier_i32 = _map_smem_to_cluster_rank(completion_barrier, peer_rank, loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i32, source_i32, barrier_i32, Int32(copy_bytes).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _mbarrier_wait_acquire_cluster(barrier: cute.Pointer, phase: Int32, *, loc=None, ip=None) -> None:
    """Block on one local mbarrier phase with a cluster-scope acquire."""
    barrier_i32 = barrier.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [barrier_i32, phase.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tCLUSTER_WAIT_LOOP:\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], $1, 10000000;\n\t@!p bra CLUSTER_WAIT_LOOP;\n\t}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _dq_pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Round two FP32 values into one BF16x2 word, low logical value first."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _dq_store_bf16x8_streaming(
    destination: cute.Pointer, packed01: cutlass.Uint32, packed23: cutlass.Uint32, packed45: cutlass.Uint32, packed67: cutlass.Uint32, *, loc=None, ip=None
) -> None:
    """Issue one streaming/evict-first 16-byte store for eight adjacent BF16 values."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i64,
            cutlass.Uint32(packed01).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(packed23).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(packed45).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(packed67).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cs.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _extract_u64x4(value, *, loc=None, ip=None) -> U64x4:
    """Unpack an LLVM ``{i64 x 4}`` aggregate into four ``Uint64`` values."""
    return tuple((cutlass.Uint64(llvm.extractvalue(T.i64(), value, [index], loc=loc, ip=ip)) for index in range(4)))


def _extract_f32x16(value, *, loc=None, ip=None) -> F32x16:
    """Unpack an LLVM ``{f32 x 16}`` aggregate into sixteen ``Float32`` values."""
    return tuple((Float32(llvm.extractvalue(T.f32(), value, [index], loc=loc, ip=ip)) for index in range(16)))


@dsl_user_op
def _prefetch_o_row_l2(source: cute.Pointer, *, loc=None, ip=None) -> None:
    """Pull one immutable O row chunk into L2 ahead of the reducer sweep."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(None, [source_i64], "prefetch.global.L2 [$0];", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _load_bf16x16(source: cute.Pointer, *, loc=None, ip=None) -> U64x4:
    """Load one immutable O or dO segment with the neutral cache policy."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i64()] * 4)
    value = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.v4.u64 {$0, $1, $2, $3}, [$4];",
        "=l,=l,=l,=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _extract_u64x4(value, loc=loc, ip=ip)


@dsl_user_op
def _decode_bf16x16_to_f32(x0: cutlass.Uint64, x1: cutlass.Uint64, x2: cutlass.Uint64, x3: cutlass.Uint64, *, loc=None, ip=None) -> F32x16:
    """Decode one packed BF16x16 vector into sixteen FP32 values."""
    result_type = llvm.StructType.get_literal([T.f32()] * 16)
    values = llvm.inline_asm(
        result_type,
        [
            cutlass.Uint64(x0).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(x1).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(x2).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(x3).ir_value(loc=loc, ip=ip),
        ],
        "{\n        .reg .b32 word<8>;\n        .reg .b16 half<16>;\n        mov.b64 {word0, word1}, $16;\n        mov.b64 {word2, word3}, $17;\n        mov.b64 {word4, word5}, $18;\n        mov.b64 {word6, word7}, $19;\n        mov.b32 {half0, half1}, word0;\n        mov.b32 {half2, half3}, word1;\n        mov.b32 {half4, half5}, word2;\n        mov.b32 {half6, half7}, word3;\n        mov.b32 {half8, half9}, word4;\n        mov.b32 {half10, half11}, word5;\n        mov.b32 {half12, half13}, word6;\n        mov.b32 {half14, half15}, word7;\n        cvt.f32.bf16 $0, half0;\n        cvt.f32.bf16 $1, half1;\n        cvt.f32.bf16 $2, half2;\n        cvt.f32.bf16 $3, half3;\n        cvt.f32.bf16 $4, half4;\n        cvt.f32.bf16 $5, half5;\n        cvt.f32.bf16 $6, half6;\n        cvt.f32.bf16 $7, half7;\n        cvt.f32.bf16 $8, half8;\n        cvt.f32.bf16 $9, half9;\n        cvt.f32.bf16 $10, half10;\n        cvt.f32.bf16 $11, half11;\n        cvt.f32.bf16 $12, half12;\n        cvt.f32.bf16 $13, half13;\n        cvt.f32.bf16 $14, half14;\n        cvt.f32.bf16 $15, half15;\n        }",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l,l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _extract_f32x16(values, loc=loc, ip=ip)


@dsl_user_op
def _dot_f32x16(out_values: F32x16, dout_values: F32x16, *, loc=None, ip=None) -> Float32:
    """Evaluate one fixed 16-element FP32 FMA chain."""
    result = llvm.inline_asm(
        T.f32(),
        [Float32(value).ir_value(loc=loc, ip=ip) for value in (*out_values, *dout_values)],
        "{\n        .reg .f32 acc;\n        mov.f32 acc, 0f00000000;\n        fma.rn.f32 acc, $1, $17, acc;\n        fma.rn.f32 acc, $2, $18, acc;\n        fma.rn.f32 acc, $3, $19, acc;\n        fma.rn.f32 acc, $4, $20, acc;\n        fma.rn.f32 acc, $5, $21, acc;\n        fma.rn.f32 acc, $6, $22, acc;\n        fma.rn.f32 acc, $7, $23, acc;\n        fma.rn.f32 acc, $8, $24, acc;\n        fma.rn.f32 acc, $9, $25, acc;\n        fma.rn.f32 acc, $10, $26, acc;\n        fma.rn.f32 acc, $11, $27, acc;\n        fma.rn.f32 acc, $12, $28, acc;\n        fma.rn.f32 acc, $13, $29, acc;\n        fma.rn.f32 acc, $14, $30, acc;\n        fma.rn.f32 acc, $15, $31, acc;\n        fma.rn.f32 acc, $16, $32, acc;\n        mov.f32 $0, acc;\n        }",
        "=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(result)


@dsl_user_op
def _load_do_shared_bf16x16_two128_ordered(first_8: cute.Pointer, second_8: cute.Pointer, *, loc=None, ip=None) -> U64x4:
    """Two separately mapped 128-bit reads from the SW128 stationary panel."""
    first_i32 = first_8.toint(loc=loc, ip=ip).ir_value()
    second_i32 = second_8.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i64()] * 4)
    value = llvm.inline_asm(
        result_type,
        [first_i32, second_i32],
        "{\n        ld.shared.v2.u64 {$0, $1}, [$4];\n        ld.shared.v2.u64 {$2, $3}, [$5];\n        }",
        "=l,=l,=l,=l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _extract_u64x4(value, loc=loc, ip=ip)


@dsl_user_op
def _reduce_s8_dim_chunks_owned(
    p0: Float32, p1: Float32, p2: Float32, p3: Float32, p4: Float32, p5: Float32, p6: Float32, p7: Float32, *, loc=None, ip=None
) -> Float32:
    """Reduce within eight independent four-lane row groups."""
    result = llvm.inline_asm(
        T.f32(),
        [
            Float32(p0).ir_value(loc=loc, ip=ip),
            Float32(p1).ir_value(loc=loc, ip=ip),
            Float32(p2).ir_value(loc=loc, ip=ip),
            Float32(p3).ir_value(loc=loc, ip=ip),
            Float32(p4).ir_value(loc=loc, ip=ip),
            Float32(p5).ir_value(loc=loc, ip=ip),
            Float32(p6).ir_value(loc=loc, ip=ip),
            Float32(p7).ir_value(loc=loc, ip=ip),
        ],
        "{\n        .reg .u32 lane, bits, send_bits, recv_bits;\n        .reg .pred pred;\n        .reg .f32 p<8>, y<4>, z<2>, lhs, rhs, acc, recv;\n        mov.f32 p0, $1;\n        mov.f32 p1, $2;\n        mov.f32 p2, $3;\n        mov.f32 p3, $4;\n        mov.f32 p4, $5;\n        mov.f32 p5, $6;\n        mov.f32 p6, $7;\n        mov.f32 p7, $8;\n        mov.u32 lane, %laneid;\n\n        and.b32 bits, lane, 16;\n        setp.ne.u32 pred, bits, 0;\n        selp.f32 lhs, p4, p0, pred;\n        selp.f32 rhs, p0, p4, pred;\n        add.rn.f32 y0, lhs, rhs;\n        selp.f32 lhs, p5, p1, pred;\n        selp.f32 rhs, p1, p5, pred;\n        add.rn.f32 y1, lhs, rhs;\n        selp.f32 lhs, p6, p2, pred;\n        selp.f32 rhs, p2, p6, pred;\n        add.rn.f32 y2, lhs, rhs;\n        selp.f32 lhs, p7, p3, pred;\n        selp.f32 rhs, p3, p7, pred;\n        add.rn.f32 y3, lhs, rhs;\n\n        and.b32 bits, lane, 8;\n        setp.ne.u32 pred, bits, 0;\n        selp.f32 lhs, y2, y0, pred;\n        selp.f32 rhs, y0, y2, pred;\n        add.rn.f32 z0, lhs, rhs;\n        selp.f32 lhs, y3, y1, pred;\n        selp.f32 rhs, y1, y3, pred;\n        add.rn.f32 z1, lhs, rhs;\n\n        and.b32 bits, lane, 4;\n        setp.ne.u32 pred, bits, 0;\n        selp.f32 lhs, z1, z0, pred;\n        selp.f32 rhs, z0, z1, pred;\n        add.rn.f32 acc, lhs, rhs;\n\n        mov.b32 send_bits, acc;\n        shfl.sync.bfly.b32 recv_bits, send_bits, 2, 0x1f, 0xffffffff;\n        mov.b32 recv, recv_bits;\n        add.rn.f32 acc, acc, recv;\n        mov.b32 send_bits, acc;\n        shfl.sync.bfly.b32 recv_bits, send_bits, 1, 0x1f, 0xffffffff;\n        mov.b32 recv, recv_bits;\n        add.rn.f32 acc, acc, recv;\n        mov.f32 $0, acc;\n        }",
        "=f,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(result)


@dsl_user_op
def _cp_async_g2s_b128_index(destination: cute.Pointer, source: cute.Pointer, *, loc=None, ip=None) -> None:
    """Issue one 16-byte index copy for a producer thread."""
    destination_i32 = destination.toint(loc=loc, ip=ip).ir_value()
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i32, source_i64],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _load_f32x4(source: cute.Pointer, *, loc=None, ip=None) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load four aligned FP32 accumulator values with one 128-bit load."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.f32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), values, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), values, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), values, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _store_f32x4(destination: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Store four naturally aligned FP32 values with one 128-bit instruction."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _match_any_sync_i32(value: Int32, *, loc=None, ip=None) -> Int32:
    """Return the full-warp mask of lanes whose 32-bit value matches this lane."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "match.any.sync.b32 $0, $1, 0xffffffff;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _n64_rows_unique(index_row: cute.Tensor, tile_index: Int32, topk: Int32, seqlen_kv: Int32, lane: Int32):
    """Return a warp-uniform flag for uniqueness among valid rows in one N64 tile."""
    tile_base = tile_index * Int32(64)
    slot_0 = tile_base + lane
    slot_1 = slot_0 + Int32(32)
    row_0 = Int32(-1)
    row_1 = Int32(-1)
    if slot_0 < topk:
        row_0 = index_row[slot_0]
    if slot_1 < topk:
        row_1 = index_row[slot_1]
    valid_0 = row_0 >= Int32(0) and row_0 < seqlen_kv
    valid_1 = row_1 >= Int32(0) and row_1 < seqlen_kv

    # Production Top-K rows are sorted by key index before attention.  Prove the
    # common strictly-increasing case with two neighbor shuffles, and retain the
    # exhaustive comparison below for arbitrary caller-provided index order.
    sorted_violation = Int32(0)
    next_row_0 = cute.arch.shuffle_sync(row_0, lane + Int32(1))
    next_row_1 = cute.arch.shuffle_sync(row_1, lane + Int32(1))
    first_row_1 = cute.arch.shuffle_sync(row_1, Int32(0))
    if slot_0 < topk and not valid_0:
        sorted_violation = Int32(1)
    if slot_1 < topk and not valid_1:
        sorted_violation = Int32(1)
    if lane < Int32(31) and slot_0 + Int32(1) < topk and row_0 >= next_row_0:
        sorted_violation = Int32(1)
    if lane == Int32(31) and tile_base + Int32(32) < topk and row_0 >= first_row_1:
        sorted_violation = Int32(1)
    if lane < Int32(31) and slot_1 + Int32(1) < topk and row_1 >= next_row_1:
        sorted_violation = Int32(1)
    strictly_sorted = cute.arch.make_warp_uniform(cute.arch.warp_redux_sync(sorted_violation, "or")) == Int32(0)

    rows_unique = strictly_sorted
    if not strictly_sorted:
        peers_0 = _match_any_sync_i32(row_0)
        peers_1 = _match_any_sync_i32(row_1)
        duplicate = Int32(0)
        if valid_0 and cute.arch.popc(peers_0) > Int32(1):
            duplicate = Int32(1)
        if valid_1 and cute.arch.popc(peers_1) > Int32(1):
            duplicate = Int32(1)
        for source_lane in cutlass.range_constexpr(32):
            other_row = cute.arch.shuffle_sync(row_1, source_lane)
            if valid_0 and row_0 == other_row:
                duplicate = Int32(1)
        rows_unique = cute.arch.make_warp_uniform(cute.arch.warp_redux_sync(duplicate, "or")) == Int32(0)
    return rows_unique


@cute.jit
def _band_has_duplicate(row_band: cute.Tensor):
    """Return whether eight locally-owned active slots contain a repeated row."""
    duplicate = cutlass.Boolean(False)
    for i in cutlass.range_constexpr(8):
        row = row_band[i]
        if row >= Int32(0):
            for j in cutlass.range_constexpr(i):
                if row_band[j] == row:
                    duplicate = cutlass.Boolean(True)
    return duplicate


@cute.jit
def _apply_duplicate_error_feedback_bf16(source: cute.Tensor, rounded: cute.Tensor, row_band: cute.Tensor, head_group: cutlass.Constexpr[int]):
    """Feed each duplicate class's aggregate BF16 residual to its first slot."""
    group_base = 2 * (head_group % 2) + 16 * (head_group // 2)
    for representative in cutlass.range_constexpr(8):
        rep_row = row_band[representative]
        is_representative = rep_row >= Int32(0)
        for prior in cutlass.range_constexpr(representative):
            if row_band[prior] == rep_row:
                is_representative = cutlass.Boolean(False)
        if is_representative:
            residual = Float32(0.0)
            class_size = Int32(0)
            all_finite = cutlass.Boolean(True)
            for band in cutlass.range_constexpr(8):
                if row_band[band] == rep_row:
                    index = group_base + band % 2 + 4 * (band // 2)
                    value = Float32(source[index])
                    high = Float32(rounded[index])
                    if cute.isfinite(value) and cute.isfinite(high):
                        residual += value - high
                    else:
                        all_finite = cutlass.Boolean(False)
                    class_size += Int32(1)
            if class_size > Int32(1) and all_finite:
                rep_index = group_base + representative % 2 + 4 * (representative // 2)
                rounded[rep_index] = BFloat16(Float32(rounded[rep_index]) + residual)


@dsl_user_op
def _store_zero_f32x4(destination: cute.Pointer, *, loc=None, ip=None) -> None:
    """Clear four adjacent aligned FP32 values with one 128-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None, [destination_i64], "st.global.v4.b32 [$0], {0, 0, 0, 0};", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )


@dsl_user_op
def _store_zero_f32x8(destination: cute.Pointer, *, loc=None, ip=None) -> None:
    """Clear eight adjacent aligned FP32 values with one 256-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None, [destination_i64], "st.global.v4.b64 [$0], {0, 0, 0, 0};", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )


@dsl_user_op
def _store_bf16x4_ordinary(destination: cute.Pointer, packed01: cutlass.Uint32, packed23: cutlass.Uint32, *, loc=None, ip=None) -> None:
    """Store four adjacent BF16 outputs with one ordinary 64-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, cutlass.Uint32(packed01).ir_value(loc=loc, ip=ip), cutlass.Uint32(packed23).ir_value(loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionDSABackwardSm100H128D576TwoCTA:
    """Two-CTA specialization for BF16 H96/H128 D576/V512.

    Each ``(2, 1, 1)`` cluster partitions a fixed 128-head tile across two
    CTAs; H96 predicates the unused tail. The main score, dP, dQ, dV, and dK operations, plus the D64
    score and dQ tails, use ``CtaGroup.TWO`` tensor cores. Per-CTA dK-tail
    partials use warp tensor cores. All MMA accumulators, O*dO, dKV
    workspace/atomics, and dSink use FP32. Softmax and dS arithmetic use
    FP32 before the BF16 tensor-core operand conversion.
    """

    arch = 100
    deterministic = False
    H_TILE_CLUSTER = 128
    H_TILE_CTA = 64
    N_TILE = 64
    N_TILE_CTA = 32
    D_HEAD = 512
    D_TAIL = 64
    D_QK = D_HEAD + D_TAIL
    D_TILE_CLUSTER = 256
    D_TILE_CTA = 128
    D_ROUNDS = D_HEAD // D_TILE_CLUSTER
    K_CHUNK = 128
    K_CHUNKS = D_HEAD // K_CHUNK
    DQ_MMA_TILER = (D_TILE_CLUSTER, H_TILE_CLUSTER, N_TILE)
    CLUSTER_SHAPE_MNK = (2, 1, 1)
    MATH_THREADS_PER_CTA = 128
    KV_LOAD_THREADS = 128
    KV_GROUP_SIZE = 8
    KV_NUM_GROUPS = KV_LOAD_THREADS // KV_GROUP_SIZE
    TMEM_COLUMNS = 512
    MAX_SMEM_BYTES = 232448
    QUADRANT_ELEMENTS = H_TILE_CTA * N_TILE_CTA
    O_PREFETCH_MAX_TOPK = 128
    DSINK_BLOCK_Q = 32
    DSINK_THREADS = 128
    DSINK_UNROLL = 4
    ZERO_THREADS = 128
    ZERO_ROWS_PER_BLOCK = 2
    THREADS_PER_CTA = 640
    GATHER_WARPS = 4
    MATH_WARP_BEGIN = 4
    MATH_WARPS = 4
    REDUCE_WARP_BEGIN = 8
    REDUCE_WARPS = 8
    MMA_WARP = 16
    LOAD_WARP = 17
    RELAY_WARP = 18
    GATHER_THREADS = GATHER_WARPS * 32
    MATH_THREAD_BEGIN = MATH_WARP_BEGIN * 32
    MATH_THREADS = MATH_WARPS * 32
    REDUCE_THREAD_BEGIN = REDUCE_WARP_BEGIN * 32
    REDUCE_THREADS = REDUCE_WARPS * 32
    DKV_MMA_TILER = (256, 64, 64)
    ROUND_BUF_ELEMENTS = 8192
    ROUND_K_HEADS = 32
    ROUND_TILER = (D_TILE_CLUSTER, N_TILE, ROUND_K_HEADS)
    ROUND_STAGE_ELEMENTS = D_TILE_CTA * ROUND_K_HEADS
    ROUND_STAGE_BYTES = 2 * ROUND_STAGE_ELEMENTS
    ROUND_STAGES = ROUND_BUF_ELEMENTS // ROUND_STAGE_ELEMENTS
    ROUND_GENS_PER_TILE = 4 * (H_TILE_CLUSTER // ROUND_K_HEADS)
    PDS_BLOCK_ELEMENTS = 2048
    PDS_BLOCK_BYTES = 4096
    TMEM_S_OFFSET = 0
    TMEM_S1_OFFSET = 32
    TMEM_DP_OFFSET = 64
    TMEM_DQT_OFFSET = 96
    TMEM_DQ0_OFFSET = 128
    TMEM_DQ1_OFFSET = 256
    TMEM_DKV0_OFFSET = 384
    TMEM_DKV1_OFFSET = 448
    SCORE_DONE_STAGES = 2
    DP_DONE_STAGES = 1
    TAIL_WARP = 19
    MMA_DONE_STAGES = 2
    DQ_EPI_BATCH_CHUNKS = 4
    DQ_WIDE_STORE_VALUES = 8
    DQ_CONVERSION_PAIR_VALUES = 2
    GATHER_SETMAXREG = 96
    UTILITY_SETMAXREG = 96
    MATH_SETMAXREG = 96
    REDUCER_SETMAXREG = 96
    MATH_EPI_SETMAXREG = 200
    REDUCER_EPI_SETMAXREG = 24

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
        single_query: bool = False,
        num_heads: int = 128,
    ):
        """Fix the compile-time contract: BF16, D576/D_v512, M64 tiles, the top-k width, and single-query mode."""
        if element_dtype != BFloat16:
            raise ValueError(f"two-CTA DSA backward requires BF16, got {element_dtype}")
        if head_dim != 576 or head_dim_v != 512:
            raise ValueError("two-CTA DSA backward requires head_dim=576, head_dim_v=512")
        if block_tile != 64:
            raise ValueError(f"two-CTA DSA backward requires block_tile=64, got {block_tile}")
        if max_topk not in (128, 512, 1024, 1152, 2048):
            raise ValueError(f"two-CTA DSA backward requires max_topk in {{128, 512, 1024, 1152, 2048}}, got {max_topk}")
        if num_heads not in (96, 128):
            raise ValueError(f"two-CTA DSA backward requires num_heads in {{96, 128}}, got {num_heads}")
        self.element_dtype = element_dtype
        self.acc_dtype = Float32
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.head_dim_main = head_dim_v
        self.same_hdim_kv = False
        self.block_tile = block_tile
        self.max_topk = max_topk
        self.SINGLE_QUERY = single_query
        self.num_heads = num_heads
        self.round_head_chunks = num_heads // self.ROUND_K_HEADS
        assert self.H_TILE_CLUSTER % self.ROUND_K_HEADS == 0
        assert self.ROUND_STAGES in (2, 4, 8)
        assert self.ROUND_GENS_PER_TILE % (2 * self.ROUND_STAGES) == 0
        assert (
            self.GATHER_SETMAXREG * self.GATHER_THREADS
            + self.MATH_SETMAXREG * self.MATH_THREADS
            + self.REDUCER_SETMAXREG * self.REDUCE_THREADS
            + self.UTILITY_SETMAXREG * 4 * 32
            == 640 * 96
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.THREADS_PER_CTA)
        self.cta_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=self.THREADS_PER_CTA)
        self.math_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=self.MATH_THREADS)
        self.stats_lse_barrier = pipeline.NamedBarrier(barrier_id=4, num_threads=self.MATH_THREADS + self.REDUCE_THREADS)
        self.gather_barrier = pipeline.NamedBarrier(barrier_id=5, num_threads=self.GATHER_THREADS)
        self.stats_odo_barrier = pipeline.NamedBarrier(barrier_id=6, num_threads=self.MATH_THREADS + self.REDUCE_THREADS)
        self.dsink_reducer_barrier = pipeline.NamedBarrier(barrier_id=7, num_threads=self.REDUCE_THREADS)

    def split_wg(self, t: cute.Tensor, num_warp_groups: int, wg_idx: int):
        """Split the trailing mode of ``t`` across ``num_warp_groups`` and return warp group ``wg_idx``'s slice."""
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 4):
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], t.shape[2], (cute.size(t, mode=[3]) // num_warp_groups, num_warp_groups))))
            ret = p[None, None, None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], (cute.size(t, mode=[2]) // num_warp_groups, num_warp_groups))))
            ret = p[None, None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 2):
            p = cute.composition(t, cute.make_layout((t.shape[0], (cute.size(t, mode=[1]) // num_warp_groups, num_warp_groups))))
            ret = p[None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 1):
            p = cute.composition(t, cute.make_layout((t.shape[0] // num_warp_groups, num_warp_groups)))
            ret = p[None, wg_idx]
        return ret

    @cute.jit
    def _copy_sparse_k_d128_row(
        self,
        mKV: cute.Tensor,
        destination_rows: cute.Tensor,
        destination_row: Int32,
        kv_index: Int32,
        batch_idx: Int32,
        d_offset: Int32,
        index_in_group: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Copy one D128 slice of a sparse KV row with 128-bit cp.async."""
        source_row_full = mKV[kv_index, None, (0, batch_idx)]
        source_row_offset = source_row_full.iterator + d_offset
        source_row = cute.make_tensor(
            cute.make_ptr(self.element_dtype, source_row_offset.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.K_CHUNK,))
        )
        source_chunks = cute.flat_divide(source_row, (8,))
        destination_row_tensor = destination_rows[destination_row, None]
        destination_chunks = cute.flat_divide(destination_row_tensor, (8,))
        for tile in cutlass.range_constexpr(self.K_CHUNK // 64):
            chunk_index = tile * self.KV_GROUP_SIZE + index_in_group
            thread_source = thread_copy.partition_S(source_chunks[None, chunk_index])
            thread_destination = thread_copy.partition_D(destination_chunks[None, chunk_index])
            cute.copy(copy_atom, thread_source, thread_destination)

    @cute.jit
    def _zero_sparse_k_d128_row(self, destination_rows: cute.Tensor, destination_row: Int32, index_in_group: Int32):
        """Cooperatively zero one D128 sparse-row destination."""
        destination_row_tensor = destination_rows[destination_row, None]
        destination_chunks = cute.flat_divide(destination_row_tensor, (8,))
        for tile in cutlass.range_constexpr(self.K_CHUNK // 64):
            chunk_index = tile * self.KV_GROUP_SIZE + index_in_group
            destination_chunks[None, chunk_index].fill(0.0)

    def _make_score_tmem_load(self):
        """Use the 16-DP/256-bit score accumulator load required by the publish store layout."""
        return cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)), self.acc_dtype)

    @cute.jit
    def _kd_round_rows(self, tensor: cute.Tensor) -> cute.Tensor:
        """Return an [N64, D128] row-major view of one dQ-A round buffer."""
        return cute.composition(tensor[None, None, None, 0], cute.make_layout((self.N_TILE, self.D_TILE_CTA), stride=(self.D_TILE_CTA, 1)))

    @cute.jit
    def _fill_kdq_pair(
        self,
        mKV: cute.Tensor,
        kd_rows_0: cute.Tensor,
        kd_rows_1: cute.Tensor,
        batch_idx: Int32,
        rank: Int32,
        role_tidx: Int32,
        thread_count: cutlass.Constexpr[int],
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        kv_index_0: Int32,
        kv_index_1: Int32,
    ) -> None:
        """Fill the peer-owned N32 half of both sparse K-dQ panels.

        Each score-B generation already holds the rank-owned N32 half of
        both K-dQ D128 panels, so only the peer-owned half is gathered.
        """
        seqlen_kv = cute.size(mKV, mode=[0])
        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = thread_count // self.KV_GROUP_SIZE
        d_offset_0 = rank * Int32(self.D_TILE_CTA)
        d_offset_1 = Int32(self.D_TILE_CLUSTER) + rank * Int32(self.D_TILE_CTA)
        assert self.N_TILE_CTA % groups_total == 0
        assert self.N_TILE_CTA // groups_total == 2
        half_base = (Int32(1) - rank) * Int32(self.N_TILE_CTA)
        kdq_local_n = [half_base + Int32(row_iteration * groups_total) + group_index for row_iteration in range(2)]
        kdq_kv_index = [kv_index_0, kv_index_1]
        for row_iteration in cutlass.range_constexpr(2):
            local_n = kdq_local_n[row_iteration]
            kv_index = kdq_kv_index[row_iteration]
            if kv_index >= Int32(0) and kv_index < seqlen_kv:
                self._copy_sparse_k_d128_row(mKV, kd_rows_0, local_n, kv_index, batch_idx, d_offset_0, index_in_group, copy_atom, thread_copy)
                self._copy_sparse_k_d128_row(mKV, kd_rows_1, local_n, kv_index, batch_idx, d_offset_1, index_in_group, copy_atom, thread_copy)
            else:
                self._zero_sparse_k_d128_row(kd_rows_0, local_n, index_in_group)
                self._zero_sparse_k_d128_row(kd_rows_1, local_n, index_in_group)

    @cute.jit
    def _issue_dkv_sweep(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        round_fragment_0: cute.Tensor,
        round_fragment_1: cute.Tensor,
        round_fragment_2: cute.Tensor,
        round_fragment_3: cute.Tensor,
        round_fragment_4: cute.Tensor,
        round_fragment_5: cute.Tensor,
        round_fragment_6: cute.Tensor,
        round_fragment_7: cute.Tensor,
        b_fragment_0: cute.Tensor,
        b_fragment_1: cute.Tensor,
        first_accumulate: cutlass.Constexpr[bool],
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
    ):
        """Consume the active 96- or 128-head A sweep against two 64-head B halves.

        A sweep spans H_TILE_CLUSTER // ROUND_K_HEADS generations. The
        smaller tail-compatible ring wraps independently of head progress;
        pipeline generations protect each slot against premature reuse.
        """
        round_slot_fragments = (
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
        )
        for chunk in cutlass.range_constexpr(self.round_head_chunks):
            head_base = chunk * self.ROUND_K_HEADS
            b_fragment = b_fragment_0 if head_base < self.H_TILE_CTA else b_fragment_1
            b_k_half = head_base % self.H_TILE_CTA // self.ROUND_K_HEADS
            accumulate = True if chunk > 0 else first_accumulate
            round_pipeline.consumer_wait(round_consumer_state)
            self._issue_dkv_pass(dkv_tiled_mma, t_dkv, round_slot_fragments[chunk % self.ROUND_STAGES], b_fragment, b_k_half, accumulate)
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
        # Keep every sweep aligned to round fragment zero. H96 consumes three
        # useful chunks from a two-stage ring, so it retires the fourth
        # producer generation without issuing the invalid-head MMA.
        for _ in cutlass.range_constexpr((-self.round_head_chunks) % self.ROUND_STAGES):
            round_pipeline.consumer_wait(round_consumer_state)
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
        return round_consumer_state

    @cute.jit
    def _issue_dkv_pass(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        b_k_half: cutlass.Constexpr[int],
        accumulate: cutlass.Constexpr[bool],
    ) -> None:
        """Issue one self-contained A stage against its matching B half."""
        k_blocks = cute.size(a_fragment, mode=[2])
        assert k_blocks == self.ROUND_K_HEADS // 16
        assert cute.size(b_fragment, mode=[2]) == self.H_TILE_CTA // 16
        b_k_block_offset = b_k_half * k_blocks
        mma = dkv_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        for k_block in cutlass.range_constexpr(k_blocks):
            cute.gemm(mma, t_dkv, a_fragment[None, None, k_block, 0], b_fragment[None, None, b_k_block_offset + k_block, 0], t_dkv)
            mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _zero_dq(
        self, rank_coordinates: cute.Tensor, mdQ: cute.Tensor, round_index: cutlass.Constexpr[int], token_idx: Int32, batch_idx: Int32, tidx: Int32
    ) -> None:
        """Write the required all-zero dQ result when no tile is issued."""
        if tidx < Int32(self.MATH_THREADS_PER_CTA):
            linear_index = tidx
            while linear_index < cute.size(rank_coordinates):
                coordinate = cute.idx2crd(linear_index, rank_coordinates.shape)
                logical_coordinate = rank_coordinates[coordinate]
                d_in_round = Int32(cute.get(logical_coordinate, mode=[0]))
                head = Int32(cute.get(logical_coordinate, mode=[1]))
                if head < Int32(self.num_heads):
                    mdQ[Int32(round_index * self.D_TILE_CLUSTER) + d_in_round, head, (token_idx, batch_idx)] = self.element_dtype(0.0)
                linear_index += Int32(self.MATH_THREADS_PER_CTA)

    @cute.jit
    def _issue_score(
        self,
        tiled_mma: cute.TiledMma,
        accumulator_0: cute.Tensor,
        accumulator_1: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Issue one score-side CG2 GEMM over four resident D128 chunks."""
        done_pipeline.producer_acquire(producer_state)
        if producer_state.index == Int32(0):
            self._issue_score_chunks(tiled_mma, accumulator_0, a_fragment, b_fragment)
        else:
            self._issue_score_chunks(tiled_mma, accumulator_1, a_fragment, b_fragment)
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_score_chunks(self, tiled_mma: cute.TiledMma, accumulator: cute.Tensor, a_fragment: cute.Tensor, b_fragment: cute.Tensor):
        """One full-K score GEMM into a single ping-pong accumulator."""
        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            for k_block in cutlass.range(0, k_blocks_per_chunk, unroll=4):
                cute.gemm(mma, accumulator, a_fragment[None, None, k_block, chunk], b_fragment[None, None, k_block, chunk], accumulator)
                mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_score_with_tail(
        self,
        tiled_mma: cute.TiledMma,
        accumulator_0: cute.Tensor,
        accumulator_1: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        a_tail_fragment: cute.Tensor,
        b_tail_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Score GEMM over the resident D512 chunks plus the D64 QK tail."""
        done_pipeline.producer_acquire(producer_state)
        if producer_state.index == Int32(0):
            self._issue_score_chunks(tiled_mma, accumulator_0, a_fragment, b_fragment)
            self._issue_score_tail_chunk(tiled_mma, accumulator_0, a_tail_fragment, b_tail_fragment)
        else:
            self._issue_score_chunks(tiled_mma, accumulator_1, a_fragment, b_fragment)
            self._issue_score_tail_chunk(tiled_mma, accumulator_1, a_tail_fragment, b_tail_fragment)
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_score_tail_chunk(self, tiled_mma: cute.TiledMma, accumulator: cute.Tensor, a_fragment: cute.Tensor, b_fragment: cute.Tensor):
        """Accumulate the QK-only D64 chunk on top of the main-512 score."""
        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, True)
        for k_block in cutlass.range_constexpr(cute.size(a_fragment, mode=[2])):
            cute.gemm(mma, accumulator, a_fragment[None, None, k_block, 0], b_fragment[None, None, k_block, 0], accumulator)

    @cute.jit
    def _prefetch_tile_indices(
        self, mTopkIdxs: cute.Tensor, tile_indices: cute.Tensor, token_idx: Int32, batch_idx: Int32, tile_index: Int32, slot: Int32, tidx: Int32
    ) -> None:
        """Sixteen threads copy one complete N64 index tile to one SMEM slot."""
        if tidx < Int32(16):
            lane_base = tidx * Int32(4)
            position = tile_index * Int32(self.N_TILE) + lane_base
            source = mTopkIdxs.iterator + mTopkIdxs.layout((position, (token_idx, batch_idx)))
            destination = tile_indices.iterator + tile_indices.layout((lane_base, slot))
            _cp_async_g2s_b128_index(destination, source)

    @cute.jit
    def _resolve_kdq_row_indices(self, tile_indices: cute.Tensor, tile_index: Int32, topk: Int32, rank: Int32, role_tidx: Int32):
        """Resolve this thread's two K-dQ peer-half sparse row indices.

        The shared-index loads and validity clamps match ``_gather_kdq_indexed``.
        The caller can resolve them before acquiring the destination's kscore
        generation, then pass the resolved indices to the gather.
        """
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = self.GATHER_THREADS // self.KV_GROUP_SIZE
        half_base = (Int32(1) - rank) * Int32(self.N_TILE_CTA)
        kdq_local_n = [half_base + Int32(row_iteration * groups_total) + group_index for row_iteration in range(2)]
        resolved = []
        for local_n in kdq_local_n:
            global_n = tile_index * Int32(self.N_TILE) + local_n
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = tile_indices[local_n]
            resolved.append(kv_index)
        return (resolved[0], resolved[1])

    @cute.jit
    def _resolve_score_row_indices(self, tile_indices: cute.Tensor, tile_index: Int32, topk: Int32, rank: Int32, role_tidx: Int32):
        """Resolve this thread's two rank-owned score-B sparse row indices.

        The resolution matches ``_load_score_kv_indexed`` and is shared by both
        refill waves and the K-tail slab refill.
        """
        group_index = role_tidx // self.KV_GROUP_SIZE
        row_local_n = [row_iteration * self.KV_NUM_GROUPS + group_index for row_iteration in range(2)]
        resolved = []
        for local_n in row_local_n:
            logical_n = rank * self.N_TILE_CTA + local_n
            topk_slot = tile_index * Int32(self.N_TILE) + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = tile_indices[logical_n]
            resolved.append(kv_index)
        return (resolved[0], resolved[1])

    @cute.jit
    def _gather_kt_slab(
        self,
        mKV: cute.Tensor,
        kt_destination: cute.Tensor,
        batch_idx: Int32,
        role_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        pre_kv_index_0: Int32,
        pre_kv_index_1: Int32,
    ) -> None:
        """Refill the rank-owned K-tail slab from pre-resolved row indices.

        The score-tail UMMA is the slab's only reader. The kscore release that
        gates the K-dQ gather publishes its completion. The caller drains these
        copies with its copy group before publishing the next generation.
        """
        seqlen_kv = cute.size(mKV, mode=[0])
        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        row_kv_index = [pre_kv_index_0, pre_kv_index_1]
        for row_iteration in cutlass.range_constexpr(2):
            local_n = row_iteration * self.KV_NUM_GROUPS + group_index
            kv_index = row_kv_index[row_iteration]
            if kv_index >= 0 and kv_index < seqlen_kv:
                self._copy_k_tail_row(mKV, kt_destination, local_n, kv_index, batch_idx, index_in_group, copy_atom, thread_copy)
            else:
                self._zero_k_tail_row(kt_destination, local_n, index_in_group)

    @cute.jit
    def _gather_kdq_indexed(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        tile_indices: cute.Tensor,
        kd_rows_0: cute.Tensor,
        kd_rows_1: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        role_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        pre_kv_index_0: Optional[Int32] = None,
        pre_kv_index_1: Optional[Int32] = None,
    ) -> None:
        """Fill the peer-owned K-dQ halves in the score-KV shared buffer.

        The caller must acquire the destination's kscore generation first. It
        publishes the gathered data with a cp.async drain, a fence, and the
        kscore producer commit. ``pre_kv_index_0/1`` optionally provide the
        indices resolved by ``_resolve_kdq_row_indices``.
        """
        if cutlass.const_expr(pre_kv_index_0 is not None):
            kv_index_0, kv_index_1 = (pre_kv_index_0, pre_kv_index_1)
        else:
            kv_index_0, kv_index_1 = self._resolve_kdq_row_indices(tile_indices, tile_index, topk, rank, role_tidx)
        self._fill_kdq_pair(mKV, kd_rows_0, kd_rows_1, batch_idx, rank, role_tidx, self.GATHER_THREADS, copy_atom, thread_copy, kv_index_0, kv_index_1)

    @cute.jit
    def _compute_folded_lse(
        self, mLSE: cute.Tensor, mAttnSink: cute.Tensor, softmax_stats: cute.Tensor, token_idx: Int32, batch_idx: Int32, rank: Int32, reducer_tidx: Int32
    ) -> None:
        """Publish the sink-folded negative LSE to shared memory."""
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        log2_e = Float32(math.log2(math.e))
        pos_inf = Float32(float("inf"))
        neg_inf = Float32(float("-inf"))
        if lane < Int32(8):
            row = row_base + lane
            head = rank * Int32(self.H_TILE_CTA) + row
            neg_lse_log2 = neg_inf
            if head < Int32(self.num_heads):
                lse_value = Float32(mLSE[head, (token_idx, batch_idx)])
                sink_value = Float32(mAttnSink[head, (0, batch_idx)])
                lse_log2 = lse_value * log2_e
                sink_log2 = sink_value * log2_e
                if lse_log2 != pos_inf and sink_log2 != pos_inf:
                    if lse_log2 != neg_inf or sink_log2 != neg_inf:
                        maximum = cute.arch.fmax(lse_log2, sink_log2)
                        denominator = Float32(cute.math.exp2(lse_log2 - maximum) + cute.math.exp2(sink_log2 - maximum))
                        neg_lse_log2 = -(maximum + cute.math.log2(denominator))
            softmax_stats[row, 0] = neg_lse_log2

    def _get_stats_workspace(self, workspace: cute.Tensor, total_q: Int32, num_heads: Int32):
        """View the caller's stats scratch as the FP32 sum-OdO and scaled-LSE planes, indexed (head, token)."""
        total_q = cute.round_up(total_q, 8)
        acc_bytes = self.acc_dtype.width // 8
        plane_bytes = cute.assume(num_heads * total_q * acc_bytes, divby=acc_bytes * 32)
        sum_odo_iter = cute.recast_ptr(workspace.iterator, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(workspace.iterator + plane_bytes, dtype=self.acc_dtype)
        layout = cute.make_layout((num_heads, (total_q, 1)), stride=(1, (cute.assume(num_heads, divby=32), 0)))
        return (cute.make_tensor(sum_odo_iter, layout), cute.make_tensor(scaled_lse_iter, layout))

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mdSink: cute.Tensor,
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        softmax_scale: Float32 | float,
        split_count: Int32,
        stream: cuda.CUstream,
        split_regime: cutlass.Constexpr[bool],
    ):
        """Compile preprocessing, the CG2 main kernel, and postprocessing."""
        mQ = cute.make_tensor(
            mQ.iterator, cute.make_layout((mQ.shape[1], mQ.shape[2], (mQ.shape[0], 1)), stride=(mQ.stride[1], mQ.stride[2], (mQ.stride[0], 0)))
        )
        mKV = cute.make_tensor(mKV.iterator, cute.make_layout((mKV.shape[0], mKV.shape[1], (1, 1)), stride=(mKV.stride[0], mKV.stride[1], (0, 0))))
        mOut = cute.make_tensor(
            mOut.iterator, cute.make_layout((mOut.shape[1], mOut.shape[2], (mOut.shape[0], 1)), stride=(mOut.stride[1], mOut.stride[2], (mOut.stride[0], 0)))
        )
        mdO = cute.make_tensor(
            mdO.iterator, cute.make_layout((mdO.shape[1], mdO.shape[2], (mdO.shape[0], 1)), stride=(mdO.stride[1], mdO.stride[2], (mdO.stride[0], 0)))
        )
        mdQ = cute.make_tensor(
            mdQ.iterator, cute.make_layout((mdQ.shape[2], mdQ.shape[1], (mdQ.shape[0], 1)), stride=(mdQ.stride[2], mdQ.stride[1], (mdQ.stride[0], 0)))
        )
        mdKV = cute.make_tensor(mdKV.iterator, cute.make_layout((mdKV.shape[1], mdKV.shape[0], (1, 1)), stride=(mdKV.stride[1], mdKV.stride[0], (0, 0))))
        mLSE = cute.make_tensor(mLSE.iterator, cute.make_layout((mLSE.shape[1], (mLSE.shape[0], 1)), stride=(mLSE.stride[1], (mLSE.stride[0], 0))))
        mdSink = cute.make_tensor(mdSink.iterator, cute.make_layout((mdSink.shape[0], (1, 1)), stride=(1, (0, 0))))
        mAttnSink = cute.make_tensor(mAttnSink.iterator, mdSink.layout)
        mTopkIdxs = cute.make_tensor(
            mTopkIdxs.iterator, cute.make_layout((mTopkIdxs.shape[1], (mTopkIdxs.shape[0], 1)), stride=(mTopkIdxs.stride[1], (mTopkIdxs.stride[0], 0)))
        )
        if cutlass.const_expr(mTopkLength is not None):
            mTopkLength = cute.make_tensor(mTopkLength.iterator, cute.make_layout((mTopkLength.shape[0], (1, 1)), stride=(mTopkLength.stride[0], (0, 0))))
        mQT = cute.make_tensor(mQ.iterator, cute.make_layout((self.D_HEAD, mQ.shape[0], mQ.shape[2]), stride=(mQ.stride[1], mQ.stride[0], mQ.stride[2])))
        mdOT = cute.make_tensor(mdO.iterator, cute.make_layout((self.D_HEAD, mdO.shape[0], mdO.shape[2]), stride=(mdO.stride[1], mdO.stride[0], mdO.stride[2])))
        cg1 = tcgen05.CtaGroup.ONE
        cg2 = tcgen05.CtaGroup.TWO
        stationary_tiler = (self.H_TILE_CTA, self.N_TILE, self.D_HEAD)
        stationary_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cg1, stationary_tiler[:2]
        )
        score_tiler = (self.H_TILE_CLUSTER, self.N_TILE, self.K_CHUNK)
        dkv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cg2, self.DKV_MMA_TILER[:2]
        )
        dq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.MN, self.acc_dtype, cg2, self.DQ_MMA_TILER[:2]
        )
        score_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cg2, score_tiler[:2]
        )
        dp_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cg2, score_tiler[:2]
        )
        dqt_tiler = (self.H_TILE_CLUSTER, self.D_TAIL, self.N_TILE)
        dqt_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cg2, dqt_tiler[:2]
        )
        dkt_warp_op = warp.MmaF16BF16Op(self.element_dtype, self.acc_dtype, (16, 8, 16))
        dkt_warp_mma = cute.make_tiled_mma(dkt_warp_op, cute.make_layout((self.MATH_WARPS, 1, 1)))
        score_tail_tiler = (self.H_TILE_CLUSTER, self.N_TILE, self.D_TAIL)
        atom_thr_size = cute.size(dkv_tiled_mma.thr_id.shape)
        assert atom_thr_size == self.CLUSTER_SHAPE_MNK[0]
        assert cute.size(dq_tiled_mma.thr_id.shape) == atom_thr_size
        assert cute.size(score_tiled_mma.thr_id.shape) == atom_thr_size
        assert cute.size(dp_tiled_mma.thr_id.shape) == atom_thr_size
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.CLUSTER_SHAPE_MNK), (dkv_tiled_mma.thr_id.shape,))
        score_a_layout_staged = sm100_utils.make_smem_layout_a(score_tiled_mma, score_tiler, self.element_dtype, self.K_CHUNKS)
        stationary_a_layout_staged = sm100_utils.make_smem_layout_a(stationary_tiled_mma, stationary_tiler, self.element_dtype, 1)
        score_b_layout_staged = sm100_utils.make_smem_layout_b(score_tiled_mma, score_tiler, self.element_dtype, self.K_CHUNKS)
        dkv_a_layout_staged = sm100_utils.make_smem_layout_a(dkv_tiled_mma, self.DKV_MMA_TILER, self.element_dtype, 1)
        round_a_layout_staged = sm100_utils.make_smem_layout_a(dkv_tiled_mma, self.ROUND_TILER, self.element_dtype, 1)
        dkv_b_layout_staged = sm100_utils.make_smem_layout_b(dkv_tiled_mma, self.DKV_MMA_TILER, self.element_dtype, 1)
        dq_a_layout_staged = sm100_utils.make_smem_layout_a(dq_tiled_mma, self.DQ_MMA_TILER, self.element_dtype, 1)
        dq_b_layout_staged = sm100_utils.make_smem_layout_b(dq_tiled_mma, self.DQ_MMA_TILER, self.element_dtype, 1)
        qt_a_layout_staged = sm100_utils.make_smem_layout_a(score_tiled_mma, score_tail_tiler, self.element_dtype, 1)
        kt_score_layout_staged = sm100_utils.make_smem_layout_b(score_tiled_mma, score_tail_tiler, self.element_dtype, 1)
        kt_dq_layout_staged = sm100_utils.make_smem_layout_b(dqt_tiled_mma, dqt_tiler, self.element_dtype, 1)
        dqt_a_layout_staged = sm100_utils.make_smem_layout_a(dqt_tiled_mma, dqt_tiler, self.element_dtype, 1)
        assert cute.cosize(qt_a_layout_staged) == self.H_TILE_CTA * self.D_TAIL
        assert cute.cosize(kt_score_layout_staged) == self.N_TILE_CTA * self.D_TAIL
        assert cute.cosize(kt_dq_layout_staged) == self.D_TAIL // 2 * self.N_TILE
        assert cute.cosize(dqt_a_layout_staged) == self.H_TILE_CTA * self.N_TILE
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(stationary_a_layout_staged) == cute.cosize(score_a_layout_staged)
        assert stationary_a_layout_staged.inner == score_a_layout_staged.inner
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 16384
        round_stage_elements = cute.cosize(round_a_layout_staged)
        assert round_stage_elements == self.ROUND_STAGE_ELEMENTS
        assert cute.cosize(dkv_a_layout_staged) == 8192
        assert round_a_layout_staged.inner == dkv_a_layout_staged.inner
        assert cute.cosize(dkv_b_layout_staged) <= 4096
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096
        assert cute.cosize(score_a_layout_staged) >= self.H_TILE_CTA * self.N_TILE
        assert cute.cosize(score_b_layout_staged) >= self.QUADRANT_ELEMENTS
        stationary_a_layout = cute.select(stationary_a_layout_staged, mode=[0, 1, 2])
        score_a_layout = cute.select(score_a_layout_staged, mode=[0, 1, 2])
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mQ, stationary_a_layout, stationary_tiler, stationary_tiled_mma)
        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mdO, stationary_a_layout, stationary_tiler, stationary_tiled_mma)
        score_a_stage_bytes = cute.size_in_bytes(self.element_dtype, score_a_layout)
        round_a_layout = cute.select(round_a_layout_staged, mode=[0, 1, 2])
        round_tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        round_tma_atom_qt, round_tma_tensor_qt = cute.nvgpu.make_tiled_tma_atom_A(
            round_tma_load_op, mQT, round_a_layout, self.ROUND_TILER, dkv_tiled_mma, cluster_layout_vmnk.shape
        )
        round_tma_atom_dot, round_tma_tensor_dot = cute.nvgpu.make_tiled_tma_atom_A(
            round_tma_load_op, mdOT, round_a_layout, self.ROUND_TILER, dkv_tiled_mma, cluster_layout_vmnk.shape
        )
        round_stage_bytes = cute.size_in_bytes(self.element_dtype, round_a_layout)
        assert round_stage_bytes == self.ROUND_STAGE_BYTES
        local_bulk_stage_offset = cute.cosize(round_a_layout_staged)
        assert local_bulk_stage_offset == self.ROUND_STAGE_ELEMENTS
        assert cute.cosize(score_a_layout_staged) * self.ROUND_K_HEADS == 256 * local_bulk_stage_offset
        assert score_a_layout_staged.inner == round_a_layout_staged.inner
        SharedStorage = self._make_shared_storage(
            score_a_layout_staged, score_b_layout_staged, dkv_a_layout_staged, dkv_b_layout_staged, dq_a_layout_staged, dq_b_layout_staged, split_regime
        )
        self.shared_storage = SharedStorage
        self.shared_storage_bytes = SharedStorage.size_in_bytes()
        assert self.shared_storage_bytes <= self.MAX_SMEM_BYTES
        score_tmem_load = self._make_score_tmem_load()
        dq_cta_shape = (self.D_TILE_CTA, self.H_TILE_CLUSTER, self.N_TILE)
        dq_epi_tile = sm100_utils.compute_epilogue_tile_shape(dq_cta_shape, True, LayoutEnum.ROW_MAJOR, self.acc_dtype)
        dq_tmem_load = sm100_utils.get_tmem_load_op(dq_cta_shape, LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype, dq_epi_tile, True)
        sum_odo, scaled_lse = self._get_stats_workspace(workspace_LSE_OdO, mQ.shape[2][0], cute.size(problem_shape[3][0]))
        if cutlass.const_expr(self.deterministic):
            assert not split_regime
            mdKV_acc = self.make_dKV_accumulator(workspace_dKV, mKV.shape[0])
            active_dkv_shards = min(Int32(self.num_dkv_shards), problem_shape[0] * Int32(self.dkv_shards_per_query))
            if cutlass.const_expr(self.max_topk == 2048):
                self.zero_dKV_shards(mdKV_acc, mKV.shape[0]).launch(
                    grid=[cute.ceil_div(mKV.shape[0], 16), active_dkv_shards, 1], block=[640, 1, 1], stream=stream
                )
            elif cutlass.const_expr(self.max_topk != 128):
                self.zero_dKV_shards(mdKV_acc, mKV.shape[0]).launch(
                    grid=[cute.ceil_div(mKV.shape[0], 4), active_dkv_shards, 1], block=[128, 1, 1], stream=stream
                )
            for token_begin in cutlass.range(0, problem_shape[0], self.max_query_pairs_per_wave):
                main_grid_x = 2 * min(Int32(self.max_query_pairs_per_wave), problem_shape[0] - token_begin)
                self.kernel(
                    tma_atom_q,
                    tma_tensor_q,
                    tma_atom_do,
                    tma_tensor_do,
                    round_tma_atom_qt,
                    round_tma_tensor_qt,
                    round_tma_atom_dot,
                    round_tma_tensor_dot,
                    mKV,
                    mQ,
                    mdQ,
                    mdKV_acc,
                    mTopkIdxs,
                    mTopkLength,
                    mLSE,
                    mAttnSink,
                    mdSink,
                    sum_odo,
                    scaled_lse,
                    mOut,
                    mdO,
                    Float32(softmax_scale),
                    score_tiled_mma,
                    dp_tiled_mma,
                    dkv_tiled_mma,
                    dq_tiled_mma,
                    dqt_tiled_mma,
                    dkt_warp_mma,
                    score_a_layout_staged,
                    score_b_layout_staged,
                    round_a_layout_staged,
                    dkv_b_layout_staged,
                    dq_a_layout_staged,
                    dq_b_layout_staged,
                    qt_a_layout_staged,
                    kt_score_layout_staged,
                    kt_dq_layout_staged,
                    dqt_a_layout_staged,
                    cluster_layout_vmnk,
                    score_tmem_load,
                    dq_tmem_load,
                    score_a_stage_bytes,
                    round_stage_bytes,
                    stationary_tiled_mma,
                    stationary_a_layout_staged,
                    split_count,
                    split_regime,
                    token_begin,
                ).launch(
                    grid=(main_grid_x, 1, problem_shape[3][1]),
                    block=[self.THREADS_PER_CTA, 1, 1],
                    cluster=self.CLUSTER_SHAPE_MNK,
                    smem=self.shared_storage_bytes,
                    stream=stream,
                    min_blocks_per_mp=1,
                )
            self.fold_dKV_accumulator_to_output(
                mdKV_acc, mdKV, sum_odo, scaled_lse, mAttnSink, mdSink, mKV.shape[0], problem_shape[0], stream, active_dkv_shards
            )
        else:
            mdKV_acc = cute.make_tensor(cute.recast_ptr(workspace_dKV.iterator, dtype=self.acc_dtype), mdKV.layout)
            zero_grid_x = (mKV.shape[0] + self.ZERO_ROWS_PER_BLOCK - 1) // self.ZERO_ROWS_PER_BLOCK
            self.zero_init(mdKV_acc, mdSink, mKV.shape[0], cute.size(problem_shape[3][0])).launch(
                grid=[zero_grid_x, 1, problem_shape[3][1]], block=[self.ZERO_THREADS, 1, 1], stream=stream
            )
            main_grid_x = 2 * problem_shape[0]
            if cutlass.const_expr(split_regime):
                main_grid_x = main_grid_x * split_count
            self.kernel(
                tma_atom_q,
                tma_tensor_q,
                tma_atom_do,
                tma_tensor_do,
                round_tma_atom_qt,
                round_tma_tensor_qt,
                round_tma_atom_dot,
                round_tma_tensor_dot,
                mKV,
                mQ,
                mdQ,
                mdKV_acc,
                mTopkIdxs,
                mTopkLength,
                mLSE,
                mAttnSink,
                mdSink,
                sum_odo,
                scaled_lse,
                mOut,
                mdO,
                Float32(softmax_scale),
                score_tiled_mma,
                dp_tiled_mma,
                dkv_tiled_mma,
                dq_tiled_mma,
                dqt_tiled_mma,
                dkt_warp_mma,
                score_a_layout_staged,
                score_b_layout_staged,
                round_a_layout_staged,
                dkv_b_layout_staged,
                dq_a_layout_staged,
                dq_b_layout_staged,
                qt_a_layout_staged,
                kt_score_layout_staged,
                kt_dq_layout_staged,
                dqt_a_layout_staged,
                cluster_layout_vmnk,
                score_tmem_load,
                dq_tmem_load,
                score_a_stage_bytes,
                round_stage_bytes,
                stationary_tiled_mma,
                stationary_a_layout_staged,
                split_count,
                split_regime,
                Int32(0),
            ).launch(
                grid=(main_grid_x, 1, problem_shape[3][1]),
                block=[self.THREADS_PER_CTA, 1, 1],
                cluster=self.CLUSTER_SHAPE_MNK,
                smem=self.shared_storage_bytes,
                stream=stream,
                min_blocks_per_mp=1,
            )
        self.block_seq = 4 if self.max_topk == 2048 else 32
        self.num_threads_D_convert = 32
        self.num_threads_seq = 4 if self.max_topk == 2048 else self.block_seq
        convert_grid_x = (mKV.shape[0] + self.block_seq - 1) // self.block_seq
        if cutlass.const_expr(not self.SINGLE_QUERY and (not self.deterministic)):
            dsink_chunks_per_block = self.num_threads_D_convert * self.num_threads_seq // self.DSINK_THREADS
            fused_grid_x = convert_grid_x + cute.ceil_div(problem_shape[0], self.DSINK_BLOCK_Q * dsink_chunks_per_block)
        else:
            fused_grid_x = convert_grid_x
        if cutlass.const_expr(not self.deterministic):
            self.convert_dkv(mdKV_acc, mdKV, mKV.shape[0], convert_grid_x, sum_odo, scaled_lse, mAttnSink, mdSink, problem_shape[0]).launch(
                grid=[fused_grid_x, 1, problem_shape[3][1]], block=[self.num_threads_D_convert, self.num_threads_seq, 1], stream=stream
            )

    @cute.jit
    def _store_dq_epi_scalar_direct(
        self,
        t_dq: cute.Tensor,
        dq_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        mdQ_direct: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        mtx: Int32,
        batch_chunks: cutlass.Constexpr[int] = 1,
    ):
        """Store one rank-owned dQ round through the writable global dQ view.

        Issue ``batch_chunks`` TMEM-to-register loads before one
        ``tcgen05.wait::ld``. The wider batch uses the registers reassigned to
        this warp group after the reducers retire.
        """
        if mtx < self.MATH_THREADS_PER_CTA:
            tiled_t2r = tcgen05.make_tmem_copy(dq_tmem_load, t_dq)
            thread_t2r = tiled_t2r.get_slice(mtx)
            thread_source = thread_t2r.partition_S(t_dq)
            thread_coordinates = thread_t2r.partition_D(rank_coordinates)
            thread_source_chunks = cute.group_modes(thread_source, 1, cute.rank(thread_source))
            thread_coordinate_chunks = cute.group_modes(thread_coordinates, 1, cute.rank(thread_coordinates))
            num_chunks = cute.size(thread_coordinate_chunks, mode=[1])
            assert num_chunks % batch_chunks == 0
            for batch_index in cutlass.range_constexpr(num_chunks // batch_chunks):
                batched_values = []
                for slot in cutlass.range_constexpr(batch_chunks):
                    slot_index = batch_index * batch_chunks + slot
                    slot_values = cute.make_rmem_tensor(thread_coordinate_chunks[None, slot_index].shape, self.acc_dtype)
                    cute.copy(tiled_t2r, thread_source_chunks[None, slot_index], slot_values)
                    batched_values.append(slot_values)
                cute.arch.fence_view_async_tmem_load()
                for slot in cutlass.range_constexpr(batch_chunks):
                    chunk_index = batch_index * batch_chunks + slot
                    chunk_coordinates = thread_coordinate_chunks[None, chunk_index]
                    chunk_values = batched_values[slot]
                    converted_values = cute.make_rmem_tensor(chunk_values.shape, self.element_dtype)
                    packed_values = cute.recast_tensor(converted_values, cutlass.Uint32)
                    for pair_index in cutlass.range_constexpr(cute.size(packed_values)):
                        pair_base = pair_index * self.DQ_CONVERSION_PAIR_VALUES
                        packed_values[pair_index] = _dq_pack_bf16x2(chunk_values[pair_base], chunk_values[pair_base + 1])
                    for octet_index in cutlass.range_constexpr(cute.size(chunk_values) // self.DQ_WIDE_STORE_VALUES):
                        value_index = octet_index * self.DQ_WIDE_STORE_VALUES
                        d_in_round = Int32(cute.get(chunk_coordinates[value_index], mode=[0]))
                        head = Int32(cute.get(chunk_coordinates[value_index], mode=[1]))
                        global_d = Int32(round_index * self.D_TILE_CLUSTER) + d_in_round
                        contiguous = global_d & Int32(7) == Int32(0)
                        for adjacent_index in cutlass.range_constexpr(self.DQ_WIDE_STORE_VALUES - 1):
                            adjacent_offset = adjacent_index + 1
                            adjacent_coordinate = chunk_coordinates[value_index + adjacent_offset]
                            adjacent_d = Int32(round_index * self.D_TILE_CLUSTER) + Int32(cute.get(adjacent_coordinate, mode=[0]))
                            contiguous = contiguous and Int32(cute.get(adjacent_coordinate, mode=[1])) == head and (adjacent_d == global_d + adjacent_offset)
                        if contiguous and head < Int32(self.num_heads):
                            packed_base = value_index // self.DQ_CONVERSION_PAIR_VALUES
                            destination = mdQ_direct.iterator + mdQ_direct.layout((global_d, head, (token_idx, batch_idx)))
                            _dq_store_bf16x8_streaming(
                                destination,
                                packed_values[packed_base],
                                packed_values[packed_base + 1],
                                packed_values[packed_base + 2],
                                packed_values[packed_base + 3],
                            )
                        else:
                            for scalar_offset in cutlass.range_constexpr(self.DQ_WIDE_STORE_VALUES):
                                scalar_index = value_index + scalar_offset
                                scalar_coordinate = chunk_coordinates[scalar_index]
                                scalar_d = Int32(round_index * self.D_TILE_CLUSTER) + Int32(cute.get(scalar_coordinate, mode=[0]))
                                scalar_head = Int32(cute.get(scalar_coordinate, mode=[1]))
                                if scalar_head < Int32(self.num_heads):
                                    mdQ_direct[scalar_d, scalar_head, (token_idx, batch_idx)] = converted_values[scalar_index]

    @cute.jit
    def _drain_dkv(
        self,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        mdKV_acc: cute.Tensor,
        index_row: cute.Tensor,
        tile_index: Int32,
        topk: Int32,
        batch_idx: Int32,
        rtx: Int32,
        rank: Int32,
        done_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
    ):
        """Drain FP32 TMEM accumulators through paced FP32x4 atomics."""
        seqlen_kv = cute.size(mdKV_acc, mode=[1])
        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
        t_dkv_core_0 = t_dkv_0[(None, None), 0, 0]
        t_dkv_core_1 = t_dkv_1[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)), self.acc_dtype)
        tiled_t2r_0 = tcgen05.make_tmem_copy(tmem_load_atom, t_dkv_core_0)
        thread_t2r_0 = tiled_t2r_0.get_slice(dp_idx)
        tiled_t2r_1 = tcgen05.make_tmem_copy(tmem_load_atom, t_dkv_core_1)
        thread_t2r_1 = tiled_t2r_1.get_slice(dp_idx)
        c_dkv = cute.make_identity_tensor((self.D_TILE_CTA, self.N_TILE))
        thread_coordinates = self.split_wg(thread_t2r_0.partition_D(c_dkv), 2, wg_idx)
        thread_source_0 = self.split_wg(thread_t2r_0.partition_S(t_dkv_core_0), 2, wg_idx)
        thread_source_1 = self.split_wg(thread_t2r_1.partition_S(t_dkv_core_1), 2, wg_idx)
        thread_values_0 = cute.make_rmem_tensor(thread_coordinates.shape, self.acc_dtype)
        thread_values_1 = thread_values_0
        tile_base = tile_index * Int32(self.N_TILE)
        r_topk = cute.make_rmem_tensor((8,), cutlass.Int32)
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            local_row = Int32(cute.get(thread_coordinates[coord_base], mode=[1]))
            global_row = tile_base + local_row
            if global_row < topk:
                r_topk[i] = index_row[global_row]
            else:
                r_topk[i] = Int32(-1)
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_0, thread_source_0, thread_values_0)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        assert cute.size(thread_values_0) == self.N_TILE // 2
        sub_tile_idx_0 = rank
        sub_tile_idx_1 = Int32(2) + rank
        if cutlass.const_expr(self.deterministic):
            owner_lane_group = dp_idx % Int32(4)
            lane_group = Int32(0)
            for active_wg in cutlass.range(2):
                if wg_idx == Int32(active_wg):
                    for lane_group in cutlass.range(4):
                        if owner_lane_group == Int32(lane_group):
                            self._deposit_dkv_panel_ordered(thread_values_0, r_topk, mdKV_acc, sub_tile_idx_0, dp_idx, batch_idx)
                        cute.arch.sync_warp()
                self.dsink_reducer_barrier.arrive_and_wait()
        else:
            for i in cutlass.range_constexpr(8):
                coord_base = i * 2 - i % 2
                rdkv_frg_0 = cute.make_rmem_tensor((4,), self.acc_dtype)
                rdkv_frg_0[0] = thread_values_0[coord_base]
                rdkv_frg_0[1] = thread_values_0[coord_base + 2]
                rdkv_frg_0[2] = thread_values_0[coord_base + 16]
                rdkv_frg_0[3] = thread_values_0[coord_base + 18]
                kv_index = r_topk[i]
                if kv_index >= Int32(0) and kv_index < seqlen_kv:
                    dkv_row = mdKV_acc[None, kv_index, (0, batch_idx)]
                    tile_row = cute.flat_divide(dkv_row, (128,))
                    tile_row_0 = tile_row[None, sub_tile_idx_0]
                    tile_row_0 = cute.flat_divide(tile_row_0, (4,))
                    target_frg_0 = tile_row_0[None, dp_idx // 4]
                    cute.arch.atomic_add(target_frg_0.iterator.llvm_ptr, rdkv_frg_0.load())
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        if cutlass.const_expr(self.deterministic):
            owner_lane_group = dp_idx % Int32(4)
            lane_group = Int32(0)
            for active_wg in cutlass.range(2):
                if wg_idx == Int32(active_wg):
                    for lane_group in cutlass.range(4):
                        if owner_lane_group == Int32(lane_group):
                            self._deposit_dkv_panel_ordered(thread_values_1, r_topk, mdKV_acc, sub_tile_idx_1, dp_idx, batch_idx)
                        cute.arch.sync_warp()
                self.dsink_reducer_barrier.arrive_and_wait()
        else:
            for i in cutlass.range_constexpr(8):
                coord_base = i * 2 - i % 2
                rdkv_frg_1 = cute.make_rmem_tensor((4,), self.acc_dtype)
                rdkv_frg_1[0] = thread_values_1[coord_base]
                rdkv_frg_1[1] = thread_values_1[coord_base + 2]
                rdkv_frg_1[2] = thread_values_1[coord_base + 16]
                rdkv_frg_1[3] = thread_values_1[coord_base + 18]
                kv_index = r_topk[i]
                if kv_index >= Int32(0) and kv_index < seqlen_kv:
                    dkv_row = mdKV_acc[None, kv_index, (0, batch_idx)]
                    tile_row = cute.flat_divide(dkv_row, (128,))
                    tile_row_1 = tile_row[None, sub_tile_idx_1]
                    tile_row_1 = cute.flat_divide(tile_row_1, (4,))
                    target_frg_1 = tile_row_1[None, dp_idx // 4]
                    cute.arch.atomic_add(target_frg_1.iterator.llvm_ptr, rdkv_frg_1.load())
        return (wait_state, release_state)

    @cute.jit
    def _deposit_dkv_panel_ordered(self, values, sparse_rows, dkv, dimension_panel: Int32, dp_idx: Int32, batch: Int32):
        """Deposit one owner's eight slots in their established order."""
        seqlen_kv = cute.size(dkv, mode=[1])
        row = Int32(-1)
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            fragment = cute.make_rmem_tensor((4,), self.acc_dtype)
            fragment[0] = values[coord_base]
            fragment[1] = values[coord_base + 2]
            fragment[2] = values[coord_base + 16]
            fragment[3] = values[coord_base + 18]
            row = sparse_rows[i]
            if row >= Int32(0) and row < seqlen_kv:
                destination_row = dkv[None, row, (0, batch)]
                panels = cute.flat_divide(destination_row, (128,))
                panel = cute.flat_divide(panels[None, dimension_panel], (4,))
                destination = panel[None, dp_idx // 4]
                ordered_dkv_quad(destination, fragment)

    @cute.jit
    def _dot_odo_bf16x16_bits(self, out_bits: U64x4, dout_bits: U64x4) -> Float32:
        """Decode one packed segment pair and retain the fixed FMA order."""
        out_values = _decode_bf16x16_to_f32(out_bits[0], out_bits[1], out_bits[2], out_bits[3])
        dout_values = _decode_bf16x16_to_f32(dout_bits[0], dout_bits[1], dout_bits[2], dout_bits[3])
        return _dot_f32x16(out_values, dout_values)

    @cute.kernel
    def convert_dkv(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        convert_blocks: Int32,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        d_sink: cute.Tensor,
        total_q: Int32,
    ):
        """Convert the call-local FP32 dKV accumulator and reduce dSink.

        Blocks past ``convert_blocks`` reduce dSink with FP32 atomic updates.
        Their ``seq_id < seqlen`` conversion guard is false, so the conversion
        and dSink roles have disjoint blocks.
        """
        assert not self.same_hdim_kv
        assert self.head_dim_main == 512
        assert mdKV_acc.element_type == cutlass.Float32
        assert mdKV.element_type == cutlass.BFloat16
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        if cutlass.const_expr(not self.SINGLE_QUERY and (not self.deterministic)):
            if seq_block_idx >= convert_blocks:
                assert self.num_threads_D_convert * self.num_threads_seq % self.DSINK_THREADS == 0
                chunks_per_block = self.num_threads_D_convert * self.num_threads_seq // self.DSINK_THREADS
                flat_thread = tidx + Int32(self.num_threads_D_convert) * tidy
                head_idx = flat_thread % Int32(self.DSINK_THREADS)
                chunk_in_block = flat_thread // Int32(self.DSINK_THREADS)
                q_block_idx = (seq_block_idx - convert_blocks) * Int32(chunks_per_block) + chunk_in_block
                q_idx = q_block_idx * Int32(self.DSINK_BLOCK_Q)
                if q_idx < total_q and head_idx < Int32(self.num_heads):
                    q_end = min(total_q, (q_block_idx + Int32(1)) * Int32(self.DSINK_BLOCK_Q))
                    sink_log2 = Float32(attn_sink[head_idx, (0, batch_idx)]) * Float32(math.log2(math.e))
                    acc_0 = Float32(0.0)
                    acc_1 = Float32(0.0)
                    acc_2 = Float32(0.0)
                    acc_3 = Float32(0.0)
                    while q_idx + 3 < q_end:
                        p_0 = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx, batch_idx)])
                        p_1 = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx + 1, batch_idx)])
                        p_2 = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx + 2, batch_idx)])
                        p_3 = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx + 3, batch_idx)])
                        if sink_log2 == Float32(float("inf")):
                            p_0 = p_1 = p_2 = p_3 = Float32(1.0)
                        elif sink_log2 == Float32(float("-inf")):
                            p_0 = p_1 = p_2 = p_3 = Float32(0.0)
                        acc_0 += p_0 * sum_odo[head_idx, (q_idx, batch_idx)]
                        acc_1 += p_1 * sum_odo[head_idx, (q_idx + 1, batch_idx)]
                        acc_2 += p_2 * sum_odo[head_idx, (q_idx + 2, batch_idx)]
                        acc_3 += p_3 * sum_odo[head_idx, (q_idx + 3, batch_idx)]
                        q_idx += self.DSINK_UNROLL
                    while q_idx < q_end:
                        p_tail = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx, batch_idx)])
                        if sink_log2 == Float32(float("inf")):
                            p_tail = Float32(1.0)
                        elif sink_log2 == Float32(float("-inf")):
                            p_tail = Float32(0.0)
                        acc_0 += p_tail * sum_odo[head_idx, (q_idx, batch_idx)]
                        q_idx += 1
                    ptr = d_sink.iterator + cute.crd2idx((head_idx, (0, batch_idx)), d_sink.layout)
                    cute.arch.atomic_add(ptr.llvm_ptr, acc_0 + acc_1 + (acc_2 + acc_3))
        seq_id = self.block_seq * seq_block_idx + tidy
        if seq_id < seqlen:
            acc_row = mdKV_acc[None, seq_id, (0, batch_idx)]
            out_row = mdKV[None, seq_id, (0, batch_idx)]
            source_base = tidx * Int32(16)
            source = acc_row.iterator + source_base
            values = (*_load_f32x4(source), *_load_f32x4(source + Int32(4)), *_load_f32x4(source + Int32(8)), *_load_f32x4(source + Int32(12)))
            chunk = source_base // Int32(32) * Int32(32)
            a0 = source_base % Int32(32) // Int32(4)
            for b in cutlass.range_constexpr(4):
                output_base = chunk + a0 + b * Int32(8)
                packed01 = _dq_pack_bf16x2(values[b], values[b + 4])
                packed23 = _dq_pack_bf16x2(values[b + 8], values[b + 12])
                _store_bf16x4_ordinary(out_row.iterator + output_base, packed01, packed23)
            for tail_half in cutlass.range_constexpr(2):
                d = Int32(self.D_HEAD) + tidx + Int32(32 * tail_half)
                out_row[d] = self.element_dtype(acc_row[d])

    @cute.jit
    def _load_score_kv_indexed(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        tile_indices: cute.Tensor,
        destination: cute.Tensor,
        kt_destination: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        direct_indices: cutlass.Constexpr[bool] = False,
        wave: cutlass.Constexpr[int] = -1,
        pre_kv_index_0: Optional[Int32] = None,
        pre_kv_index_1: Optional[Int32] = None,
        skip_tail: cutlass.Constexpr[bool] = False,
    ):
        """Gather the rank-owned N32 x D576 score B with 128-bit copies.

        ``wave`` selects destination bytes: wave 0 covers score-B chunks 0/1
        and the K-tail slab, wave 1 covers chunks 2/3, and -1 copies everything.
        The two chunk pairs also hold the corresponding K-dQ panels.

        ``direct_indices`` selects global rather than staged shared indices.
        ``pre_kv_index_0/1`` supply this thread's already-resolved sparse rows
        for reuse across the refill waves. ``skip_tail`` omits the K-tail slab
        from wave 0 when the caller has already refilled it.
        """
        seqlen_kv = cute.size(mKV, mode=[0])
        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        row_local_n = [row_iteration * self.KV_NUM_GROUPS + group_index for row_iteration in range(rows_per_group)]
        if cutlass.const_expr(pre_kv_index_0 is not None):
            assert rows_per_group == 2
            row_kv_index = [pre_kv_index_0, pre_kv_index_1]
        else:
            row_kv_index = []
            for local_n in row_local_n:
                logical_n = rank * self.N_TILE_CTA + local_n
                topk_slot = tile_index * self.N_TILE + logical_n
                kv_index = Int32(-1)
                if topk_slot < topk:
                    if cutlass.const_expr(direct_indices):
                        kv_index = mTopkIdxs[topk_slot, (token_idx, batch_idx)]
                    else:
                        kv_index = tile_indices[logical_n]
                row_kv_index.append(kv_index)
        if cutlass.const_expr(wave == 0):
            chunk_ids = (0, 1)
            include_tail = not skip_tail
        elif cutlass.const_expr(wave == 1):
            chunk_ids = (2, 3)
            include_tail = False
        else:
            chunk_ids = tuple(range(self.K_CHUNKS))
            include_tail = True
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_local_n[row_iteration]
            kv_index = row_kv_index[row_iteration]
            for chunk in chunk_ids:
                destination_rows = cute.composition(destination[None, None, None, chunk], cute.make_layout((self.N_TILE_CTA, self.K_CHUNK)))
                if kv_index >= 0 and kv_index < seqlen_kv:
                    self._copy_sparse_k_d128_row(
                        mKV, destination_rows, local_n, kv_index, batch_idx, Int32(chunk * self.K_CHUNK), index_in_group, copy_atom, thread_copy
                    )
                else:
                    self._zero_sparse_k_d128_row(destination_rows, local_n, index_in_group)
            if cutlass.const_expr(include_tail):
                if kv_index >= 0 and kv_index < seqlen_kv:
                    self._copy_k_tail_row(mKV, kt_destination, local_n, kv_index, batch_idx, index_in_group, copy_atom, thread_copy)
                else:
                    self._zero_k_tail_row(kt_destination, local_n, index_in_group)

    @cute.jit
    def _prefetch_odo_rows(self, mOut: cute.Tensor, token_idx: Int32, batch_idx: Int32, rank: Int32, reducer_tidx: Int32) -> None:
        """Warm every O chunk this reducer thread will read before the LSE fold."""
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        row = row_base + lane // Int32(4)
        group_lane = lane % Int32(4)
        head = rank * Int32(self.H_TILE_CTA) + row
        if head < Int32(self.num_heads):
            for chunk in cutlass.range_constexpr(2):
                dim = (group_lane + Int32(4 * chunk)) * Int32(64)
                source = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
                _prefetch_o_row_l2(source)

    @cute.jit
    def _compute_global_odo(
        self, mOut: cute.Tensor, mdO: cute.Tensor, softmax_stats: cute.Tensor, token_idx: Int32, batch_idx: Int32, rank: Int32, reducer_tidx: Int32
    ) -> None:
        """Reduce O.dO for this CTA's heads of one token straight from global memory into ``softmax_stats``."""
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        dim = lane * Int32(16)
        out_bits = [None] * 4
        dout_bits = [None] * 4
        partials = [None] * 8
        zero_bits = (cutlass.Uint64(0), cutlass.Uint64(0), cutlass.Uint64(0), cutlass.Uint64(0))
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot)
            head = rank * Int32(self.H_TILE_CTA) + row
            out_bits[slot] = zero_bits
            if head < Int32(self.num_heads):
                out_pointer = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
                out_bits[slot] = _load_bf16x16(out_pointer)
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot)
            head = rank * Int32(self.H_TILE_CTA) + row
            dout_bits[slot] = zero_bits
            if head < Int32(self.num_heads):
                dout_pointer = mdO.iterator + mdO.layout((head, dim, (token_idx, batch_idx)))
                dout_bits[slot] = _load_bf16x16(dout_pointer)
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot)
            head = rank * Int32(self.H_TILE_CTA) + row
            partials[slot] = self._dot_odo_bf16x16_bits(out_bits[slot], dout_bits[slot])
            refill_row = row_base + Int32(slot + 4)
            refill_head = rank * Int32(self.H_TILE_CTA) + refill_row
            out_bits[slot] = zero_bits
            dout_bits[slot] = zero_bits
            if refill_head < Int32(self.num_heads):
                out_pointer = mOut.iterator + mOut.layout((refill_head, dim, (token_idx, batch_idx)))
                dout_pointer = mdO.iterator + mdO.layout((refill_head, dim, (token_idx, batch_idx)))
                out_bits[slot] = _load_bf16x16(out_pointer)
                dout_bits[slot] = _load_bf16x16(dout_pointer)
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot + 4)
            head = rank * Int32(self.H_TILE_CTA) + row
            partials[slot + 4] = self._dot_odo_bf16x16_bits(out_bits[slot], dout_bits[slot])
        for row_offset in cutlass.range_constexpr(8):
            row = row_base + Int32(row_offset)
            head = rank * Int32(self.H_TILE_CTA) + row
            row_sum = cute.arch.warp_reduction_sum(partials[row_offset], threads_in_group=32)
            if lane == Int32(0):
                raw_neg_sum_odo = -row_sum
                softmax_stats[row, 1] = raw_neg_sum_odo

    @cute.jit
    def _load_stationary_do_bf16x16(self, stationary_do_physical: cute.Tensor, row: Int32, dim: Int32) -> U64x4:
        """Load sixteen BF16 dO values of one head row from the stationary tile's physical layout."""
        raw_offset = row * Int32(64) + dim % Int32(64) + dim // Int32(64) * Int32(self.H_TILE_CTA * 64)
        raw_offset_hi = row * Int32(64) + (dim + Int32(8)) % Int32(64) + (dim + Int32(8)) // Int32(64) * Int32(self.H_TILE_CTA * 64)
        physical_offset = raw_offset ^ raw_offset // Int32(64) % Int32(8) * Int32(8)
        physical_offset_hi = raw_offset_hi ^ raw_offset_hi // Int32(64) % Int32(8) * Int32(8)
        return _load_do_shared_bf16x16_two128_ordered(stationary_do_physical.iterator + physical_offset, stationary_do_physical.iterator + physical_offset_hi)

    @cute.jit
    def _compute_stationary_odo(
        self,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        stationary_do_physical: cute.Tensor,
        stationary_tma_mbars: cute.Pointer,
        softmax_stats: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        reducer_tidx: Int32,
    ) -> None:
        """Reduce O.dO for one token using the stationary dO tile already resident in shared memory."""
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        row_offset = lane // Int32(4)
        group_lane = lane % Int32(4)
        row = row_base + row_offset
        head = rank * Int32(self.H_TILE_CTA) + row
        partials = [None] * 8
        out_prefetch = [None] * 4
        zero_bits = (cutlass.Uint64(0), cutlass.Uint64(0), cutlass.Uint64(0), cutlass.Uint64(0))
        for chunk in cutlass.range_constexpr(4):
            logical_old_lane = group_lane + Int32(4 * chunk)
            dim = logical_old_lane * Int32(16)
            out_prefetch[chunk] = zero_bits
            if head < Int32(self.num_heads):
                out_pointer = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
                out_prefetch[chunk] = _load_bf16x16(out_pointer)
        cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
        for chunk in cutlass.range_constexpr(4):
            logical_old_lane = group_lane + Int32(4 * chunk)
            dim = logical_old_lane * Int32(16)
            dout_bits = self._load_stationary_do_bf16x16(stationary_do_physical, row, dim)
            partials[chunk] = self._dot_odo_bf16x16_bits(out_prefetch[chunk], dout_bits)
        for chunk in cutlass.range_constexpr(4, 8):
            logical_old_lane = group_lane + Int32(4 * chunk)
            dim = logical_old_lane * Int32(16)
            out_bits = zero_bits
            if head < Int32(self.num_heads):
                out_pointer = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
                out_bits = _load_bf16x16(out_pointer)
            dout_bits = self._load_stationary_do_bf16x16(stationary_do_physical, row, dim)
            partials[chunk] = self._dot_odo_bf16x16_bits(out_bits, dout_bits)
        owned_row_sum = _reduce_s8_dim_chunks_owned(partials[0], partials[1], partials[2], partials[3], partials[4], partials[5], partials[6], partials[7])
        if group_lane == Int32(0):
            raw_neg_sum_odo = -owned_row_sum
            softmax_stats[row, 1] = raw_neg_sum_odo

    @cute.jit
    def _publish_dsink_stats(
        self,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        d_sink: cute.Tensor,
        softmax_stats: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        reducer_tidx: Int32,
    ) -> None:
        """Publish one token's sum-OdO and scaled-LSE to the workspace, or its complete dSink in single-query mode."""
        if reducer_tidx < Int32(self.H_TILE_CTA):
            row = reducer_tidx
            head = rank * Int32(self.H_TILE_CTA) + row
            if head < Int32(self.num_heads):
                if cutlass.const_expr(self.SINGLE_QUERY):
                    sink_log2 = Float32(attn_sink[head, (0, batch_idx)]) * Float32(math.log2(math.e))
                    p_sink = cute.math.exp2(sink_log2 + softmax_stats[row, 0])
                    if sink_log2 == Float32(float("inf")):
                        p_sink = Float32(1.0)
                    elif sink_log2 == Float32(float("-inf")):
                        p_sink = Float32(0.0)
                    d_sink[head, (0, batch_idx)] = p_sink * softmax_stats[row, 1]
                else:
                    sum_odo[head, (token_idx, batch_idx)] = softmax_stats[row, 1]
                    scaled_lse[head, (token_idx, batch_idx)] = softmax_stats[row, 0]

    @cute.kernel
    def zero_init(self, mdKV_acc: cute.Tensor, mdSink: cute.Tensor, seqlen_kv: Int32, num_heads: Int32):
        """Clear the call-local FP32 dKV workspace and dSink accumulator.

        Each block clears ``ZERO_ROWS_PER_BLOCK`` KV rows with coalesced
        four-byte stores. Block 0 also clears dSink before its atomic reduction.
        """
        assert self.D_HEAD == 4 * self.ZERO_THREADS
        assert self.D_TAIL % 4 == 0
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, batch_idx = cute.arch.block_idx()
        row_base = bidx * Int32(self.ZERO_ROWS_PER_BLOCK)
        for row_offset in cutlass.range_constexpr(self.ZERO_ROWS_PER_BLOCK):
            row = row_base + Int32(row_offset)
            if row < seqlen_kv:
                pointer = mdKV_acc.iterator + cute.crd2idx((tidx * Int32(4), row, (0, batch_idx)), mdKV_acc.layout)
                _store_zero_f32x4(pointer)
                if tidx < Int32(self.D_TAIL // 4):
                    tail_pointer = mdKV_acc.iterator + cute.crd2idx((Int32(self.D_HEAD) + tidx * Int32(4), row, (0, batch_idx)), mdKV_acc.layout)
                    _store_zero_f32x4(tail_pointer)
        if bidx == Int32(0):
            if tidx < num_heads:
                mdSink[tidx, (0, batch_idx)] = Float32(0.0)

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        round_tma_atom_qt: cute.CopyAtom,
        round_tma_tensor_qt: cute.Tensor,
        round_tma_atom_dot: cute.CopyAtom,
        round_tma_tensor_dot: cute.Tensor,
        mKV: cute.Tensor,
        mQ_tail: cute.Tensor,
        mdQ: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mdSink: cute.Tensor,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        scale_softmax: Float32,
        score_tiled_mma: cute.TiledMma,
        dp_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        dq_tiled_mma: cute.TiledMma,
        dqt_tiled_mma: cute.TiledMma,
        dkt_warp_mma: cute.TiledMma,
        score_a_layout_staged: cute.ComposedLayout,
        score_b_layout_staged: cute.ComposedLayout,
        round_a_layout_staged: cute.ComposedLayout,
        dkv_b_layout_staged: cute.ComposedLayout,
        dq_a_layout_staged: cute.ComposedLayout,
        dq_b_layout_staged: cute.ComposedLayout,
        qt_a_layout_staged: cute.ComposedLayout,
        kt_score_layout_staged: cute.ComposedLayout,
        kt_dq_layout_staged: cute.ComposedLayout,
        dqt_a_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        score_tmem_load: cute.CopyAtom,
        dq_tmem_load: cute.CopyAtom,
        score_a_stage_bytes: cutlass.Constexpr[int],
        round_stage_bytes: cutlass.Constexpr[int],
        stationary_tiled_mma: cute.TiledMma,
        stationary_a_layout_staged: cute.ComposedLayout,
        split_count: Int32,
        split_regime: cutlass.Constexpr[bool],
        token_begin: Int32,
    ):
        """Execute the FP32 five-GEMM two-CTA schedule."""
        physical_x, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(rank)
        peer_rank = Int32(1) - rank
        cluster_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        if cutlass.const_expr(split_regime):
            token_idx = cluster_idx // split_count
            split_idx = cluster_idx % split_count
            split_idx = cute.arch.make_warp_uniform(split_idx)
            is_dq_owner = split_idx == Int32(0)
            has_dkv = split_idx > Int32(0)
        else:
            token_idx = cluster_idx
            split_idx = Int32(0)
            is_dq_owner = True
            has_dkv = True
        is_leader_cta = rank == Int32(0)
        if cutlass.const_expr(self.deterministic):
            token_idx += token_begin
            mdKV_acc = self.select_dKV_accumulator(mdKV_acc, cute.size(mKV, mode=[0]), token_idx, rank)
            if cutlass.const_expr(self.max_topk == 128):
                self._zero_compact_record_first_wave(mdKV_acc, cute.size(mKV, mode=[0]), token_begin, tidx, rank)
        if warp_idx == Int32(self.LOAD_WARP):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(round_tma_atom_qt)
            cpasync.prefetch_descriptor(round_tma_atom_dot)
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_holding_buf_ptr = storage.tmem_holding_buf.ptr
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar.ptr
        stationary_tma_mbars = storage.stationary_tma_mbars.data_ptr()
        stationary_ready_mbar = storage.stationary_ready_mbar.data_ptr()
        landing_mbars = storage.landing_mbars.data_ptr()
        relay_mbars = storage.relay_mbars.data_ptr()
        pds_ready_mbars = storage.pds_ready_mbars.data_ptr()
        p_ready_mbars = storage.p_ready_mbars.data_ptr()
        ds_local_ready_mbar = storage.ds_local_ready_mbar.data_ptr()
        tail_ld_mbar = storage.tail_ld_mbar.data_ptr()
        stationary_do_raw = storage.stationary_do.data_ptr()
        round_buf_raw = storage.round_buf.data_ptr()
        score_kv_raw = storage.score_kv.data_ptr()
        stationary_q = storage.stationary_q.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_do = storage.stationary_do.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_q_tma = storage.stationary_q.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        stationary_do_tma = storage.stationary_do.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        stationary_do_physical = cute.make_tensor(stationary_do_raw, cute.make_layout((self.H_TILE_CTA * self.D_HEAD,), stride=(1,)))
        k_n = storage.score_kv.get_tensor(score_b_layout_staged.outer, swizzle=score_b_layout_staged.inner)
        kdq_loan_ptr_0 = cute.make_ptr(self.element_dtype, score_kv_raw.toint(), score_kv_raw.memspace, assumed_align=1024)
        kdq_loan_ptr_1 = cute.make_ptr(self.element_dtype, score_kv_raw.toint() + Int32(16384), score_kv_raw.memspace, assumed_align=1024)
        atom_mn = 64
        atom_k = 8
        m_tiles = self.D_TILE_CTA // atom_mn
        stage_elements = self.N_TILE_CTA * self.D_TILE_CTA
        n_tile_stride = atom_mn * atom_k
        d_tile_stride = n_tile_stride * (self.N_TILE_CTA // atom_k)
        kdq_outer = cute.make_layout(
            (((atom_mn, m_tiles), (atom_k, 2)), 1, (2, 2), 1),
            stride=(((1, d_tile_stride), (atom_mn, n_tile_stride)), 0, (n_tile_stride * 2, stage_elements), self.D_TILE_CTA * self.N_TILE),
        )
        assert cute.cosize(kdq_outer) == self.D_TILE_CTA * self.N_TILE
        kdq_loan = (
            cute.make_tensor(cute.recast_ptr(kdq_loan_ptr_0, dq_a_layout_staged.inner, dtype=self.element_dtype), kdq_outer),
            cute.make_tensor(cute.recast_ptr(kdq_loan_ptr_1, dq_a_layout_staged.inner, dtype=self.element_dtype), kdq_outer),
        )
        round_slots = tuple(
            (
                cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.element_dtype, round_buf_raw.toint() + slot * self.ROUND_STAGE_BYTES, round_buf_raw.memspace, assumed_align=1024),
                        round_a_layout_staged.inner,
                        dtype=self.element_dtype,
                    ),
                    round_a_layout_staged.outer,
                )
                for slot in range(self.ROUND_STAGES)
            )
        )
        p_blocks_raw = storage.p_blocks.data_ptr()
        ds_blocks_raw = storage.ds_blocks.data_ptr()
        ds_image_raw = storage.ds_image.data_ptr()
        p_blocks = (
            cute.make_tensor(cute.recast_ptr(p_blocks_raw, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer),
            cute.make_tensor(
                cute.recast_ptr(p_blocks_raw + self.PDS_BLOCK_ELEMENTS, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer
            ),
        )
        ds_blocks = (
            cute.make_tensor(cute.recast_ptr(ds_blocks_raw, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer),
            cute.make_tensor(
                cute.recast_ptr(ds_blocks_raw + self.PDS_BLOCK_ELEMENTS, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer
            ),
        )
        ds_image = storage.ds_image.get_tensor(dq_b_layout_staged.outer, swizzle=dq_b_layout_staged.inner)
        ds_image_dqt_a = cute.make_tensor(cute.recast_ptr(ds_image_raw, dqt_a_layout_staged.inner, dtype=self.element_dtype), dqt_a_layout_staged.outer)
        score_store_layout = sm100_utils.make_smem_layout_epi(self.element_dtype, LayoutEnum.COL_MAJOR, (self.H_TILE_CTA, self.N_TILE), 1)
        assert cute.cosize(score_store_layout) == cute.cosize(dq_b_layout_staged)
        assert score_store_layout.inner == dq_b_layout_staged.inner
        assert score_store_layout.inner == dkv_b_layout_staged.inner
        score_store_domain = cute.make_layout((score_store_layout.outer.shape, 1, 1, 1), stride=(score_store_layout.outer.stride, 0, 0, 0))
        assert cute.cosize(score_store_domain) == cute.cosize(dq_b_layout_staged)
        ds_image_store = storage.ds_image.get_tensor(score_store_domain, swizzle=score_store_layout.inner)
        p_block_stage = p_blocks[0][None, None, None, 0]
        assert cute.size(p_block_stage, mode=[0, 0]) == self.N_TILE_CTA
        assert cute.size(p_block_stage, mode=[0, 1]) == 16
        assert cute.size(p_block_stage, mode=[1]) == 1
        assert cute.size(p_block_stage, mode=[2]) == 4
        assert cute.size(p_block_stage) == self.PDS_BLOCK_ELEMENTS
        p_block_raw_ptrs = (p_blocks_raw, p_blocks_raw + self.PDS_BLOCK_ELEMENTS)
        ds_block_raw_ptrs = (ds_blocks_raw, ds_blocks_raw + self.PDS_BLOCK_ELEMENTS)
        flat_pds_block_layout = cute.make_layout((self.PDS_BLOCK_ELEMENTS,), stride=(1,))
        p_xchg_raw = storage.p_xchg.get_tensor(flat_pds_block_layout)
        softmax_stats = storage.stats.get_tensor(cute.make_layout((self.H_TILE_CTA, 2), stride=(1, self.H_TILE_CTA)))
        staged_indices = storage.tile_indices.get_tensor(cute.make_layout((self.N_TILE, 2), stride=(1, self.N_TILE)))
        s_q_tail = storage.q_tail.get_tensor(qt_a_layout_staged.outer, swizzle=qt_a_layout_staged.inner)
        s_kt_score = storage.kt_score.get_tensor(kt_score_layout_staged.outer, swizzle=kt_score_layout_staged.inner)
        s_kt_dq = storage.kt_dq.get_tensor(kt_dq_layout_staged.outer, swizzle=kt_dq_layout_staged.inner)
        s_q_tail_rows = cute.composition(s_q_tail[None, None, None, 0], cute.make_layout((self.H_TILE_CTA, self.D_TAIL)))
        s_kt_score_rows = cute.composition(s_kt_score[None, None, None, 0], cute.make_layout((self.N_TILE_CTA, self.D_TAIL)))
        s_kt_dq_cols = cute.composition(s_kt_dq[None, None, None, 0], cute.make_layout((self.D_TAIL // 2, self.N_TILE)))
        s_ds_kv_h = cute.composition(ds_image_dqt_a[None, None, None, 0], cute.make_layout((self.N_TILE, self.H_TILE_CTA), stride=(self.H_TILE_CTA, 1)))
        s_qt_d_h = cute.composition(s_q_tail[None, None, None, 0], cute.make_layout((self.D_TAIL, self.H_TILE_CTA), stride=(self.H_TILE_CTA, 1)))
        g_q = cute.local_tile(tma_tensor_q, cute.select((self.H_TILE_CTA, self.N_TILE, self.D_HEAD), mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        g_do = cute.local_tile(tma_tensor_do, cute.select((self.H_TILE_CTA, self.N_TILE, self.D_HEAD), mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        stationary_thr_mma = stationary_tiled_mma.get_slice(0)
        rank_g_q = stationary_thr_mma.partition_A(g_q)
        rank_g_do = stationary_thr_mma.partition_A(g_do)
        t_q_smem, t_q_gmem = cpasync.tma_partition(
            tma_atom_q, 0, cute.make_layout(1), cute.group_modes(stationary_q_tma, 0, 3), cute.group_modes(rank_g_q, 0, 3)
        )
        t_do_smem, t_do_gmem = cpasync.tma_partition(
            tma_atom_do, 0, cute.make_layout(1), cute.group_modes(stationary_do_tma, 0, 3), cute.group_modes(rank_g_do, 0, 3)
        )
        rank_score_mma = score_tiled_mma.get_slice(rank)
        rank_dkv_mma = dkv_tiled_mma.get_slice(rank)
        rank_dq_mma = dq_tiled_mma.get_slice(rank)
        rank_score_coordinates = rank_score_mma.partition_C(cute.make_identity_tensor((self.H_TILE_CLUSTER, self.N_TILE)))
        rank_dq_coordinates = rank_dq_mma.partition_C(cute.make_identity_tensor(self.DQ_MMA_TILER[:2]))
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        g_qt_round = cute.local_tile(round_tma_tensor_qt, cute.select(self.ROUND_TILER, mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        g_dot_round = cute.local_tile(round_tma_tensor_dot, cute.select(self.ROUND_TILER, mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        rank_g_qt_round = rank_dkv_mma.partition_A(g_qt_round)
        rank_g_dot_round = rank_dkv_mma.partition_A(g_dot_round)
        qt_round_smem_slots = []
        dot_round_smem_slots = []
        for slot in cutlass.range_constexpr(self.ROUND_STAGES):
            qt_slot, qt_round_gmem = cpasync.tma_partition(
                round_tma_atom_qt, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[slot], 0, 3), cute.group_modes(rank_g_qt_round, 0, 3)
            )
            dot_slot, dot_round_gmem = cpasync.tma_partition(
                round_tma_atom_dot, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[slot], 0, 3), cute.group_modes(rank_g_dot_round, 0, 3)
            )
            qt_round_smem_slots.append(qt_slot)
            dot_round_smem_slots.append(dot_slot)
            t_qt_round_gmem = qt_round_gmem
            t_dot_round_gmem = dot_round_gmem
        t_qt_round_smem = tuple(qt_round_smem_slots)
        t_dot_round_smem = tuple(dot_round_smem_slots)
        score_q_fragment = score_tiled_mma.make_fragment_A(stationary_q)
        score_do_fragment = dp_tiled_mma.make_fragment_A(stationary_do)
        score_k_fragment = score_tiled_mma.make_fragment_B(k_n)
        dp_k_fragment = dp_tiled_mma.make_fragment_B(k_n)
        score_qt_fragment = score_tiled_mma.make_fragment_A(s_q_tail)
        score_kt_fragment = score_tiled_mma.make_fragment_B(s_kt_score)
        dqt_a_fragment = dqt_tiled_mma.make_fragment_A(ds_image_dqt_a)
        dqt_b_fragment = dqt_tiled_mma.make_fragment_B(s_kt_dq)
        dq_kd_fragment_a = dq_tiled_mma.make_fragment_A(kdq_loan[0])
        dq_kd_fragment_b = dq_tiled_mma.make_fragment_A(kdq_loan[1])
        dq_ds_fragment = dq_tiled_mma.make_fragment_B(ds_image)
        assert cute.cosize(round_a_layout_staged) == self.ROUND_STAGE_ELEMENTS
        round_fragments = tuple((dkv_tiled_mma.make_fragment_A(round_slots[slot]) for slot in range(self.ROUND_STAGES)))
        round_a_k_blocks = self.ROUND_K_HEADS // 16
        for round_slot in cutlass.range_constexpr(self.ROUND_STAGES):
            round_slot_tensor = round_slots[round_slot]
            round_fragment = round_fragments[round_slot]
            assert cute.cosize(round_slot_tensor.layout) == self.ROUND_STAGE_ELEMENTS
            assert cute.size(round_slot_tensor, mode=[2]) == round_a_k_blocks
            assert cute.size(round_fragment, mode=[2]) == round_a_k_blocks
            for k_block in cutlass.range_constexpr(round_a_k_blocks):
                k_block_slice = round_slot_tensor[None, None, k_block, 0]
                k_block_offset = round_slot_tensor.layout((0, 0, k_block, 0))
                k_block_cosize = cute.cosize(k_block_slice.layout)
                assert k_block_offset >= 0
                assert k_block_offset + k_block_cosize <= self.ROUND_STAGE_ELEMENTS
        p_fragments = (dkv_tiled_mma.make_fragment_B(p_blocks[0]), dkv_tiled_mma.make_fragment_B(p_blocks[1]))
        ds_fragments = (dkv_tiled_mma.make_fragment_B(ds_blocks[0]), dkv_tiled_mma.make_fragment_B(ds_blocks[1]))
        kv_copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), self.element_dtype, num_bits_per_copy=128)
        kv_thread_copy = cute.make_tiled_copy_tv(kv_copy_atom, cute.make_layout((1,)), cute.make_layout((8,))).get_slice(0)
        if cutlass.const_expr(mTopkLength is not None):
            raw_topk = mTopkLength[token_idx]
        else:
            raw_topk = Int32(mTopkIdxs.shape[0])
        atom_thr_size = cute.size(score_tiled_mma.thr_id.shape)
        leader_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        math_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.MATH_THREADS)
        gather_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.GATHER_THREADS)
        score_ready_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.GATHER_WARPS * 2)
        reduce_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.REDUCE_THREADS)
        pipe_s_done = self._make_umma_async_pipeline(self.SCORE_DONE_STAGES, leader_group, math_group, storage.s_done_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_dp_done = self._make_umma_async_pipeline(self.DP_DONE_STAGES, leader_group, math_group, storage.dp_done_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_kscore = self._make_async_umma_pipeline(1, score_ready_group, leader_group, storage.kscore_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_round = pipeline.PipelineTmaUmma.create(
            num_stages=self.ROUND_STAGES,
            producer_group=leader_group,
            consumer_group=leader_group,
            tx_count=round_stage_bytes * 2,
            barrier_storage=storage.round_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        if cutlass.const_expr(split_regime):
            pds_commit_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size)
            pipe_pds = self._make_async_umma_pipeline(1, pds_commit_group, leader_group, storage.pds_mbars.data_ptr(), cluster_layout_vmnk)
        else:
            two_issuer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
            pipe_p_free = self._make_async_umma_pipeline(1, math_group, leader_group, storage.pds_mbars.data_ptr(), cluster_layout_vmnk)
            pipe_ds_free = self._make_async_umma_pipeline(1, math_group, two_issuer_group, storage.pds_mbars.data_ptr() + 2, cluster_layout_vmnk)
            if cutlass.const_expr(self.max_topk != 128):
                pipe_dq0_free = self._make_umma_async_pipeline(1, leader_group, gather_group, storage.dq0_free_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_dkv_done = self._make_umma_async_pipeline(self.MMA_DONE_STAGES, leader_group, reduce_group, storage.dkv_done_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_dq_done = self._make_umma_async_pipeline(1, leader_group, math_group, storage.dq_done_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_dq0_done = self._make_umma_async_pipeline(1, leader_group, gather_group, storage.dq0_done_mbars.data_ptr(), cluster_layout_vmnk)
        if warp_idx == Int32(self.LOAD_WARP):
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(stationary_tma_mbars, 1)
                cute.arch.mbarrier_init(stationary_tma_mbars + 1, 1)
        if tidx == Int32(0):
            cute.arch.mbarrier_init(stationary_ready_mbar, 2)
            cute.arch.mbarrier_init(stationary_ready_mbar + 1, 2)
            cute.arch.mbarrier_init(landing_mbars, 1)
            cute.arch.mbarrier_init(landing_mbars + 1, 1)
            cute.arch.mbarrier_init(relay_mbars, 2)
            cute.arch.mbarrier_init(relay_mbars + 1, 2)
            cute.arch.mbarrier_init(pds_ready_mbars, self.MATH_WARPS)
            cute.arch.mbarrier_init(p_ready_mbars, self.MATH_WARPS)
            if cutlass.const_expr(split_regime):
                cute.arch.mbarrier_init(ds_local_ready_mbar, 2 * self.MATH_WARPS)
            cute.arch.mbarrier_init(tail_ld_mbar, self.GATHER_WARPS)
        cute.arch.fence_view_async_shared()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)
        if cutlass.const_expr(mTopkLength is None):
            topk = raw_topk
            if topk > Int32(mTopkIdxs.shape[0]):
                topk = Int32(mTopkIdxs.shape[0])
            if topk < Int32(0):
                topk = Int32(0)
            tile_count = (topk + Int32(self.N_TILE - 1)) // Int32(self.N_TILE)
            tile_count = cute.arch.make_warp_uniform(tile_count)
            my_tile_base = Int32(0)
            my_tile_stride = Int32(1)
            my_tile_count = tile_count
            if cutlass.const_expr(split_regime):
                if split_idx > Int32(0):
                    my_tile_base = split_idx - Int32(1)
                    my_tile_stride = split_count - Int32(1)
                    my_tile_count = Int32(0)
                    if tile_count > my_tile_base:
                        my_tile_count = (tile_count - my_tile_base + my_tile_stride - Int32(1)) // my_tile_stride
            my_tile_count = cute.arch.make_warp_uniform(my_tile_count)
            my_last_tile = my_tile_base + my_tile_stride * (my_tile_count - Int32(1))
            walk_base = my_last_tile
            walk_step = Int32(0) - my_tile_stride
            if warp_idx < Int32(self.GATHER_WARPS):
                if my_tile_count > Int32(0):
                    self._load_q_tail(mQ_tail, s_q_tail_rows, token_idx, batch_idx, rank, tidx, kv_copy_atom, kv_thread_copy)
                    self._prefetch_tile_indices(mTopkIdxs, staged_indices, token_idx, batch_idx, my_last_tile, Int32(0), tidx)
                    if my_tile_count > Int32(1):
                        self._prefetch_tile_indices(mTopkIdxs, staged_indices, token_idx, batch_idx, walk_base + walk_step, Int32(1), tidx)
                    cute.arch.cp_async_commit_group()
                    self._load_score_kv_indexed(
                        mKV,
                        mTopkIdxs,
                        staged_indices[None, Int32(0)],
                        k_n,
                        s_kt_score_rows,
                        token_idx,
                        batch_idx,
                        my_last_tile,
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                        direct_indices=True,
                    )
                    cute.arch.cp_async_commit_group()
            elif warp_idx == Int32(self.LOAD_WARP):
                if cutlass.const_expr(self.max_topk == 128):
                    if my_tile_count > Int32(0):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                        cute.copy(tma_atom_q, t_q_gmem[None, rank, 0], t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
        else:
            spec_tile_base = Int32(0)
            spec_tile_stride = Int32(1)
            if cutlass.const_expr(split_regime):
                if split_idx > Int32(0):
                    spec_tile_base = split_idx - Int32(1)
                    spec_tile_stride = split_count - Int32(1)
            declared_topk = Int32(mTopkIdxs.shape[0])
            spec_topk = raw_topk
            if spec_topk > declared_topk:
                spec_topk = declared_topk
            if spec_topk < Int32(0):
                spec_topk = Int32(0)
            if warp_idx < Int32(self.GATHER_WARPS):
                self._load_q_tail(mQ_tail, s_q_tail_rows, token_idx, batch_idx, rank, tidx, kv_copy_atom, kv_thread_copy)
                self._prefetch_tile_indices(mTopkIdxs, staged_indices, token_idx, batch_idx, spec_tile_base, Int32(0), tidx)
                if (spec_tile_base + spec_tile_stride) * Int32(self.N_TILE) < declared_topk:
                    self._prefetch_tile_indices(mTopkIdxs, staged_indices, token_idx, batch_idx, spec_tile_base + spec_tile_stride, Int32(1), tidx)
                cute.arch.cp_async_commit_group()
                self._load_score_kv_indexed(
                    mKV,
                    mTopkIdxs,
                    staged_indices[None, Int32(0)],
                    k_n,
                    s_kt_score_rows,
                    token_idx,
                    batch_idx,
                    spec_tile_base,
                    spec_topk,
                    rank,
                    tidx,
                    kv_copy_atom,
                    kv_thread_copy,
                    direct_indices=True,
                )
                cute.arch.cp_async_commit_group()
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        if cutlass.const_expr(mTopkLength is not None):
            if cutlass.const_expr(self.max_topk == 128):
                if warp_idx == Int32(self.LOAD_WARP):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                    cute.copy(tma_atom_q, t_q_gmem[None, rank, 0], t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
                    cute.copy(tma_atom_do, t_do_gmem[None, rank, 0], t_do_smem[None, 0], tma_bar_ptr=stationary_tma_mbars + 1)
        tmem = TmemAllocator(
            tmem_holding_buf_ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.MATH_WARP_BEGIN,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.TMEM_COLUMNS)
        if cutlass.const_expr(mTopkLength is None):
            if warp_idx == Int32(self.LOAD_WARP):
                if my_tile_count > Int32(0):
                    if cutlass.const_expr(self.max_topk == 128):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                    else:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                            cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                        cute.copy(tma_atom_q, t_q_gmem[None, rank, 0], t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
                    cute.copy(tma_atom_do, t_do_gmem[None, rank, 0], t_do_smem[None, 0], tma_bar_ptr=stationary_tma_mbars + 1)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        score_c_layout = score_tiled_mma.make_fragment_C(score_tiled_mma.partition_shape_C((self.H_TILE_CLUSTER, self.N_TILE))).layout
        dkv_c_layout = dkv_tiled_mma.make_fragment_C(dkv_tiled_mma.partition_shape_C(self.DKV_MMA_TILER[:2])).layout
        dq_c_layout = dq_tiled_mma.make_fragment_C(dq_tiled_mma.partition_shape_C(self.DQ_MMA_TILER[:2])).layout
        t_score = cute.make_tensor(tmem_ptr + self.TMEM_S_OFFSET, score_c_layout)
        t_score_pp = cute.make_tensor(tmem_ptr + self.TMEM_S1_OFFSET, score_c_layout)
        t_dp = cute.make_tensor(tmem_ptr + self.TMEM_DP_OFFSET, score_c_layout)
        dqt_c_layout = dqt_tiled_mma.make_fragment_C(dqt_tiled_mma.partition_shape_C((self.H_TILE_CLUSTER, self.D_TAIL))).layout
        t_dqt = cute.make_tensor(tmem_ptr + self.TMEM_DQT_OFFSET, dqt_c_layout)
        t_dq = (cute.make_tensor(tmem_ptr + self.TMEM_DQ0_OFFSET, dq_c_layout), cute.make_tensor(tmem_ptr + self.TMEM_DQ1_OFFSET, dq_c_layout))
        t_dkv = (cute.make_tensor(tmem_ptr + self.TMEM_DKV0_OFFSET, dkv_c_layout), cute.make_tensor(tmem_ptr + self.TMEM_DKV1_OFFSET, dkv_c_layout))
        if cutlass.const_expr(mTopkLength is not None):
            topk = raw_topk
            if topk > Int32(mTopkIdxs.shape[0]):
                topk = Int32(mTopkIdxs.shape[0])
            if topk < Int32(0):
                topk = Int32(0)
            tile_count = (topk + Int32(self.N_TILE - 1)) // Int32(self.N_TILE)
            tile_count = cute.arch.make_warp_uniform(tile_count)
            my_tile_base = Int32(0)
            my_tile_stride = Int32(1)
            my_tile_count = tile_count
            if cutlass.const_expr(split_regime):
                if split_idx > Int32(0):
                    my_tile_base = split_idx - Int32(1)
                    my_tile_stride = split_count - Int32(1)
                    my_tile_count = Int32(0)
                    if tile_count > my_tile_base:
                        my_tile_count = (tile_count - my_tile_base + my_tile_stride - Int32(1)) // my_tile_stride
            my_tile_count = cute.arch.make_warp_uniform(my_tile_count)
            my_last_tile = my_tile_base + my_tile_stride * (my_tile_count - Int32(1))
            walk_base = my_tile_base
            walk_step = my_tile_stride
        if warp_idx < Int32(self.MATH_WARP_BEGIN):
            if cutlass.const_expr(self.GATHER_SETMAXREG > 96):
                cute.arch.setmaxregister_increase(self.GATHER_SETMAXREG)
            elif cutlass.const_expr(self.GATHER_SETMAXREG < 96):
                cute.arch.setmaxregister_decrease(self.GATHER_SETMAXREG)
        elif warp_idx >= Int32(self.MMA_WARP):
            if cutlass.const_expr(self.UTILITY_SETMAXREG > 96):
                cute.arch.setmaxregister_increase(self.UTILITY_SETMAXREG)
            elif cutlass.const_expr(self.UTILITY_SETMAXREG < 96):
                cute.arch.setmaxregister_decrease(self.UTILITY_SETMAXREG)
        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            if cutlass.const_expr(self.MATH_SETMAXREG > 96):
                cute.arch.setmaxregister_increase(self.MATH_SETMAXREG)
            elif cutlass.const_expr(self.MATH_SETMAXREG < 96):
                cute.arch.setmaxregister_decrease(self.MATH_SETMAXREG)
        elif cutlass.const_expr(self.REDUCER_SETMAXREG > 96):
            cute.arch.setmaxregister_increase(self.REDUCER_SETMAXREG)
        elif cutlass.const_expr(self.REDUCER_SETMAXREG < 96):
            cute.arch.setmaxregister_decrease(self.REDUCER_SETMAXREG)
        if warp_idx < Int32(self.GATHER_WARPS):
            gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            if cutlass.const_expr(not split_regime and self.max_topk != 128):
                dq0_free_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dq0_done_gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            gather_kd_rows_0 = self._kd_round_rows(kdq_loan[0])
            gather_kd_rows_1 = self._kd_round_rows(kdq_loan[1])
            if my_tile_count > Int32(0):
                if cutlass.const_expr(mTopkLength is None):
                    pipe_kscore.producer_acquire(gather_state)
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    if tidx & Int32(31) == Int32(0):
                        pipe_kscore.producer_commit(gather_state)
                    gather_state.advance()
                else:
                    pipe_kscore.producer_acquire(gather_state)
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    if tidx & Int32(31) == Int32(0):
                        pipe_kscore.producer_commit(gather_state)
                    gather_state.advance()
                for score_iter in cutlass.range(Int32(0), my_tile_count):
                    index_slot = score_iter & Int32(1)
                    next_score_iter = score_iter + Int32(1)
                    next_index_slot = next_score_iter & Int32(1)
                    next_kv_0 = Int32(-1)
                    next_kv_1 = Int32(-1)
                    if score_iter == Int32(0):
                        self.gather_barrier.arrive_and_wait()
                    if is_dq_owner:
                        if cutlass.const_expr(self.max_topk == 128):
                            pipe_kscore.producer_acquire(gather_state)
                            self._gather_kdq_indexed(
                                mKV,
                                mTopkIdxs,
                                staged_indices[None, index_slot],
                                gather_kd_rows_0,
                                gather_kd_rows_1,
                                token_idx,
                                batch_idx,
                                walk_base + walk_step * score_iter,
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                            )
                            self._gather_kt_dq(
                                mKV,
                                staged_indices[None, index_slot],
                                s_kt_dq_cols,
                                batch_idx,
                                walk_base + walk_step * score_iter,
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                            )
                        else:
                            kdq_kv_0, kdq_kv_1 = self._resolve_kdq_row_indices(
                                staged_indices[None, index_slot], walk_base + walk_step * score_iter, topk, rank, tidx
                            )
                            ktdq_row = tidx // Int32(2)
                            ktdq_global_n = (walk_base + walk_step * score_iter) * Int32(self.N_TILE) + ktdq_row
                            ktdq_kv = Int32(-1)
                            if ktdq_global_n < topk:
                                ktdq_kv = staged_indices[ktdq_row, index_slot]
                            ktdq_v0, ktdq_v1 = self._load_kt_dq_tail_values(mKV, batch_idx, rank, tidx, ktdq_kv)
                            pipe_kscore.producer_acquire(gather_state)
                            self._gather_kdq_indexed(
                                mKV,
                                mTopkIdxs,
                                staged_indices[None, index_slot],
                                gather_kd_rows_0,
                                gather_kd_rows_1,
                                token_idx,
                                batch_idx,
                                walk_base + walk_step * score_iter,
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                                pre_kv_index_0=kdq_kv_0,
                                pre_kv_index_1=kdq_kv_1,
                            )
                            next_kv_0, next_kv_1 = self._resolve_score_row_indices(
                                staged_indices[None, next_index_slot], walk_base + walk_step * next_score_iter, topk, rank, tidx
                            )
                            if cutlass.const_expr(not split_regime):
                                if score_iter != my_tile_count - Int32(1):
                                    self._gather_kt_slab(mKV, s_kt_score_rows, batch_idx, tidx, kv_copy_atom, kv_thread_copy, next_kv_0, next_kv_1)
                            self._store_kt_dq_tail_values(s_kt_dq_cols, tidx, ktdq_v0, ktdq_v1)
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        if tidx & Int32(31) == Int32(0):
                            pipe_kscore.producer_commit(gather_state)
                        gather_state.advance()
                    else:
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                    if score_iter != my_tile_count - Int32(1):
                        self._prefetch_kv_rows_l2(mKV, staged_indices[None, next_index_slot], walk_base + walk_step * next_score_iter, topk, rank, tidx)
                        if cutlass.const_expr(split_regime):
                            self.gather_barrier.arrive_and_wait()
                        next_iter = score_iter + Int32(1)
                        if cutlass.const_expr(split_regime or self.max_topk == 128):
                            pipe_kscore.producer_acquire(gather_state)
                            if score_iter + Int32(2) < my_tile_count:
                                self._prefetch_tile_indices(
                                    mTopkIdxs, staged_indices, token_idx, batch_idx, walk_base + walk_step * (score_iter + Int32(2)), index_slot, tidx
                                )
                            self._load_score_kv_indexed(
                                mKV,
                                mTopkIdxs,
                                staged_indices[None, next_index_slot],
                                k_n,
                                s_kt_score_rows,
                                token_idx,
                                batch_idx,
                                walk_base + walk_step * next_iter,
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                            )
                        else:
                            pipe_dq0_free.consumer_wait(dq0_free_state)
                            dq0_free_state.advance()
                            self._load_score_kv_indexed(
                                mKV,
                                mTopkIdxs,
                                staged_indices[None, next_index_slot],
                                k_n,
                                s_kt_score_rows,
                                token_idx,
                                batch_idx,
                                walk_base + walk_step * next_iter,
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                                wave=0,
                                pre_kv_index_0=next_kv_0,
                                pre_kv_index_1=next_kv_1,
                                skip_tail=True,
                            )
                            if score_iter + Int32(2) < my_tile_count:
                                self._prefetch_tile_indices(
                                    mTopkIdxs, staged_indices, token_idx, batch_idx, walk_base + walk_step * (score_iter + Int32(2)), index_slot, tidx
                                )
                            pipe_kscore.producer_acquire(gather_state)
                            self._load_score_kv_indexed(
                                mKV,
                                mTopkIdxs,
                                staged_indices[None, next_index_slot],
                                k_n,
                                s_kt_score_rows,
                                token_idx,
                                batch_idx,
                                walk_base + walk_step * next_iter,
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                                wave=1,
                                pre_kv_index_0=next_kv_0,
                                pre_kv_index_1=next_kv_1,
                            )
                        cute.arch.cp_async_commit_group()
                        if cutlass.const_expr(not (split_regime or self.max_topk == 128)):
                            if has_dkv:
                                cute.arch.mbarrier_wait(pds_ready_mbars, score_iter & Int32(1))
                                self._accumulate_dk_tail_wmma(
                                    dkt_warp_mma,
                                    s_ds_kv_h,
                                    s_qt_d_h,
                                    mTopkIdxs,
                                    mdKV_acc,
                                    token_idx,
                                    batch_idx,
                                    walk_base + walk_step * score_iter,
                                    topk,
                                    tidx,
                                    tail_ld_mbar,
                                )
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        if tidx & Int32(31) == Int32(0):
                            pipe_kscore.producer_commit(gather_state)
                        gather_state.advance()
                    if cutlass.const_expr(split_regime or self.max_topk == 128):
                        if has_dkv:
                            cute.arch.mbarrier_wait(pds_ready_mbars, score_iter & Int32(1))
                            self._accumulate_dk_tail_wmma(
                                dkt_warp_mma,
                                s_ds_kv_h,
                                s_qt_d_h,
                                mTopkIdxs,
                                mdKV_acc,
                                token_idx,
                                batch_idx,
                                walk_base + walk_step * score_iter,
                                topk,
                                tidx,
                                tail_ld_mbar,
                            )
                    elif score_iter == my_tile_count - Int32(1):
                        cute.arch.mbarrier_wait(pds_ready_mbars, score_iter & Int32(1))
                        self._accumulate_dk_tail_wmma(
                            dkt_warp_mma,
                            s_ds_kv_h,
                            s_qt_d_h,
                            mTopkIdxs,
                            mdKV_acc,
                            token_idx,
                            batch_idx,
                            walk_base + walk_step * score_iter,
                            topk,
                            tidx,
                            tail_ld_mbar,
                        )
                if is_dq_owner:
                    pipe_dq0_done.consumer_wait(dq0_done_gather_state)
                    self._store_dq_epi_scalar_direct(
                        t_dq[0], dq_tmem_load, rank_dq_coordinates, mdQ, 0, token_idx, batch_idx, rank, tidx, 4 if self.max_topk == 128 else 2
                    )
                    pipe_dq0_done.consumer_release(dq0_done_gather_state)
                    dq0_done_gather_state.advance()
                pipe_kscore.producer_tail(gather_state)
            elif cutlass.const_expr(mTopkLength is not None):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            mtx = tidx - Int32(self.MATH_THREAD_BEGIN)
            if my_tile_count > Int32(0):
                self.stats_lse_barrier.arrive_and_wait()
            else:
                self.math_barrier.arrive_and_wait()
            s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SCORE_DONE_STAGES)
            dp_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.DP_DONE_STAGES)
            if cutlass.const_expr(split_regime):
                pds_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            else:
                p_free_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                ds_free_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dq_done_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            score_copy = tcgen05.make_tmem_copy(score_tmem_load, t_score)
            score_thread = score_copy.get_slice(mtx)
            score_source = score_thread.partition_S(t_score)
            score_coordinates = score_thread.partition_D(rank_score_coordinates)
            dp_copy = tcgen05.make_tmem_copy(score_tmem_load, t_dp)
            dp_thread = dp_copy.get_slice(mtx)
            dp_source = dp_thread.partition_S(t_dp)
            score_copy_pp = tcgen05.make_tmem_copy(score_tmem_load, t_score_pp)
            score_source_pp = score_copy_pp.get_slice(mtx).partition_S(t_score_pp)
            smem_store_atom = sm100_utils.get_smem_store_op(LayoutEnum.COL_MAJOR, self.element_dtype, self.acc_dtype, score_copy)
            assert isinstance(smem_store_atom.op, warp.StMatrix8x8x16bOp)
            assert smem_store_atom.op.num_matrices == 4
            tiled_copy_r2s = cute.make_tiled_copy_D(smem_store_atom, score_copy)
            thread_copy_r2s = tiled_copy_r2s.get_slice(mtx)
            t_rs_ds = thread_copy_r2s.partition_D(ds_image_store)
            assert cute.size(t_rs_ds, mode=[4]) == 1
            t_rs_ds_tile = t_rs_ds[None, None, None, None, 0]
            aligned_p_blocks_ptr = cute.make_ptr(self.element_dtype, p_blocks[0].iterator.toint(), p_blocks[0].memspace, assumed_align=16)
            aligned_ds_blocks_ptr = cute.make_ptr(self.element_dtype, ds_blocks[0].iterator.toint(), ds_blocks[0].memspace, assumed_align=16)
            p_local_store = cute.make_tensor(cute.recast_ptr(aligned_p_blocks_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain)
            ds_local_store = cute.make_tensor(cute.recast_ptr(aligned_ds_blocks_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain)
            aligned_p_xchg_ptr = cute.make_ptr(
                self.element_dtype,
                p_xchg_raw.iterator.toint() - mtx // Int32(self.H_TILE_CTA) * Int32(self.PDS_BLOCK_BYTES),
                p_xchg_raw.memspace,
                assumed_align=16,
            )
            p_xchg_store = cute.make_tensor(cute.recast_ptr(aligned_p_xchg_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain)
            t_rs_p_local = thread_copy_r2s.partition_D(p_local_store)
            t_rs_ds_local = thread_copy_r2s.partition_D(ds_local_store)
            t_rs_p_xchg = thread_copy_r2s.partition_D(p_xchg_store)
            assert cute.size(t_rs_p_local, mode=[4]) == 1
            assert cute.size(t_rs_ds_local, mode=[4]) == 1
            assert cute.size(t_rs_p_xchg, mode=[4]) == 1
            t_rs_p_local_tile = t_rs_p_local[None, None, None, None, 0]
            t_rs_ds_local_tile = t_rs_ds_local[None, None, None, None, 0]
            t_rs_p_xchg_tile = t_rs_p_xchg[None, None, None, None, 0]
            r_score = cute.make_rmem_tensor(score_coordinates.shape, self.acc_dtype)
            r_dp = cute.make_rmem_tensor(score_coordinates.shape, self.acc_dtype)
            r_p = cute.make_rmem_tensor(score_coordinates.shape, self.element_dtype)
            r_valid_band = cute.make_rmem_tensor((8,), cutlass.Boolean)
            if cutlass.const_expr(self.deterministic):
                r_row_band = cute.make_rmem_tensor((8,), cutlass.Int32)
            r_ds = r_p
            r_delta = cute.make_rmem_tensor((4,), self.acc_dtype)
            softmax_scale_log2_e = scale_softmax * Float32(math.log2(math.e))
            hoist_group_bases = [2 * (h_group % 2) + 16 * (h_group // 2) for h_group in range(4)]
            hoist_group_local_h = [Int32(cute.get(score_coordinates[group_base], mode=[0])) % Int32(self.H_TILE_CTA) for group_base in hoist_group_bases]
            hoist_band_indices = [[group_base + j % 2 + 4 * (j // 2) for j in range(8)] for group_base in hoist_group_bases]
            hoist_band_n = [Int32(cute.get(score_coordinates[hoist_band_indices[0][band]], mode=[1])) for band in range(8)]
            hoist_lse = [softmax_stats[hoist_group_local_h[h_group], 0] for h_group in range(4)]
            math_owns_local_half = cute.arch.make_warp_uniform(mtx // Int32(self.H_TILE_CTA)) == rank
            for loop_iter in cutlass.range(my_tile_count):
                tail_tile_index = walk_base + walk_step * loop_iter
                if cutlass.const_expr(self.max_topk == 128):
                    pipe_s_done.consumer_wait(s_state)
                    if s_state.index == Int32(0):
                        cute.copy(score_copy, score_source, r_score)
                    else:
                        cute.copy(score_copy_pp, score_source_pp, r_score)
                    cute.arch.fence_view_async_tmem_load()
                    pipe_s_done.consumer_release(s_state)
                    s_state.advance()
                    for band in cutlass.range_constexpr(8):
                        slot = tail_tile_index * Int32(self.N_TILE) + hoist_band_n[band]
                        r_valid_band[band] = False
                        if cutlass.const_expr(self.deterministic):
                            r_row_band[band] = Int32(-1)
                        if slot < topk:
                            if cutlass.const_expr(split_regime):
                                row = mTopkIdxs[slot, (token_idx, batch_idx)]
                            else:
                                row = staged_indices[hoist_band_n[band], loop_iter & Int32(1)]
                            if row >= Int32(0) and row < mKV.shape[0]:
                                r_valid_band[band] = True
                                if cutlass.const_expr(self.deterministic):
                                    r_row_band[band] = row
                else:
                    for band in cutlass.range_constexpr(8):
                        slot = tail_tile_index * Int32(self.N_TILE) + hoist_band_n[band]
                        r_valid_band[band] = False
                        if cutlass.const_expr(self.deterministic):
                            r_row_band[band] = Int32(-1)
                        if slot < topk:
                            row = mTopkIdxs[slot, (token_idx, batch_idx)]
                            if row >= Int32(0) and row < mKV.shape[0]:
                                r_valid_band[band] = True
                                if cutlass.const_expr(self.deterministic):
                                    r_row_band[band] = row
                    pipe_s_done.consumer_wait(s_state)
                    if s_state.index == Int32(0):
                        cute.copy(score_copy, score_source, r_score)
                    else:
                        cute.copy(score_copy_pp, score_source_pp, r_score)
                    cute.arch.fence_view_async_tmem_load()
                    pipe_s_done.consumer_release(s_state)
                    s_state.advance()
                assert cute.size(r_score) == self.N_TILE_CTA
                for h_group in cutlass.range_constexpr(4):
                    lse = hoist_lse[h_group]
                    for pair in cutlass.range_constexpr(4):
                        i0 = hoist_band_indices[h_group][2 * pair]
                        i1 = hoist_band_indices[h_group][2 * pair + 1]
                        v0, v1 = cute.arch.fma_packed_f32x2((r_score[i0], r_score[i1]), (softmax_scale_log2_e, softmax_scale_log2_e), (lse, lse))
                        v0 = cute.math.exp2(v0, fastmath=True)
                        v1 = cute.math.exp2(v1, fastmath=True)
                        if not r_valid_band[2 * pair]:
                            v0 = Float32(0.0)
                        if not r_valid_band[2 * pair + 1]:
                            v1 = Float32(0.0)
                        r_score[i0] = v0
                        r_score[i1] = v1
                        r_p[i0] = self.element_dtype(v0)
                        r_p[i1] = self.element_dtype(v1)
                if cutlass.const_expr(self.deterministic):
                    band_has_duplicate = _band_has_duplicate(r_row_band)
                    if band_has_duplicate:
                        for h_group in cutlass.range_constexpr(4):
                            _apply_duplicate_error_feedback_bf16(r_score, r_p, r_row_band, h_group)
                if cutlass.const_expr(split_regime):
                    pipe_pds.producer_acquire(pds_state)
                else:
                    pipe_p_free.producer_acquire(p_free_prod_state)
                r_p_store = thread_copy_r2s.retile(r_p)
                assert t_rs_p_local_tile.shape == r_p_store.shape
                assert t_rs_p_xchg_tile.shape == r_p_store.shape
                if math_owns_local_half:
                    cute.copy(tiled_copy_r2s, r_p_store, t_rs_p_local_tile)
                else:
                    cute.copy(tiled_copy_r2s, r_p_store, t_rs_p_xchg_tile)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(p_ready_mbars)
                if cutlass.const_expr(not split_regime):
                    pipe_p_free.producer_commit(p_free_prod_state)
                    p_free_prod_state.advance()
                pipe_dp_done.consumer_wait(dp_state)
                cute.copy(dp_copy, dp_source, r_dp)
                for h_group in cutlass.range_constexpr(4):
                    for pair in cutlass.range_constexpr(4):
                        i0 = hoist_band_indices[h_group][2 * pair]
                        i1 = hoist_band_indices[h_group][2 * pair + 1]
                        r_score[i0], r_score[i1] = cute.arch.mul_packed_f32x2((r_score[i0], r_score[i1]), (scale_softmax, scale_softmax))
                cute.arch.fence_view_async_tmem_load()
                pipe_dp_done.consumer_release(dp_state)
                dp_state.advance()
                if loop_iter == Int32(0):
                    self.stats_odo_barrier.arrive_and_wait()
                    for h_group in cutlass.range_constexpr(4):
                        r_delta[h_group] = softmax_stats[hoist_group_local_h[h_group], 1]
                for h_group in cutlass.range_constexpr(4):
                    delta = r_delta[h_group]
                    for pair in cutlass.range_constexpr(4):
                        i0 = hoist_band_indices[h_group][2 * pair]
                        i1 = hoist_band_indices[h_group][2 * pair + 1]
                        d0, d1 = cute.arch.add_packed_f32x2((r_dp[i0], r_dp[i1]), (delta, delta))
                        d0, d1 = cute.arch.mul_packed_f32x2((d0, d1), (r_score[i0], r_score[i1]))
                        if not r_valid_band[2 * pair]:
                            d0 = Float32(0.0)
                        if not r_valid_band[2 * pair + 1]:
                            d1 = Float32(0.0)
                        if cutlass.const_expr(self.deterministic):
                            r_dp[i0] = d0
                            r_dp[i1] = d1
                        r_ds[i0] = self.element_dtype(d0)
                        r_ds[i1] = self.element_dtype(d1)
                if cutlass.const_expr(self.deterministic):
                    if band_has_duplicate:
                        for h_group in cutlass.range_constexpr(4):
                            _apply_duplicate_error_feedback_bf16(r_dp, r_ds, r_row_band, h_group)
                if cutlass.const_expr(not split_regime):
                    pipe_ds_free.producer_acquire(ds_free_prod_state)
                r_ds_store = thread_copy_r2s.retile(r_ds)
                assert t_rs_ds_local_tile.shape == r_ds_store.shape
                if math_owns_local_half:
                    cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_local_tile)
                assert t_rs_ds_tile.shape == r_ds_store.shape
                if has_dkv:
                    if loop_iter > Int32(0):
                        cute.arch.mbarrier_wait(tail_ld_mbar, loop_iter - Int32(1) & Int32(1))
                cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_tile)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(pds_ready_mbars)
                    if cutlass.const_expr(split_regime):
                        cute.arch.mbarrier_arrive(ds_local_ready_mbar, Int32(0))
                if cutlass.const_expr(split_regime):
                    pds_state.advance()
                else:
                    pipe_ds_free.producer_commit(ds_free_prod_state)
                    ds_free_prod_state.advance()
            if cutlass.const_expr(not split_regime):
                if my_tile_count > Int32(0):
                    pipe_p_free.producer_tail(p_free_prod_state)
                    pipe_ds_free.producer_tail(ds_free_prod_state)
            if is_dq_owner:
                if my_tile_count > Int32(0):
                    pipe_dq_done.consumer_wait(dq_done_state)
                    cute.arch.setmaxregister_increase(self.MATH_EPI_SETMAXREG)
                    self._store_dq_epi_scalar_direct(
                        t_dq[1], dq_tmem_load, rank_dq_coordinates, mdQ, 1, token_idx, batch_idx, rank, mtx, self.DQ_EPI_BATCH_CHUNKS
                    )
                    self._store_dqt_epi(t_dqt, dqt_tiled_mma, mdQ, token_idx, batch_idx, rank, mtx)
                    pipe_dq_done.consumer_release(dq_done_state)
                    dq_done_state.advance()
                else:
                    self._zero_dq(rank_dq_coordinates, mdQ, 0, token_idx, batch_idx, mtx)
                    self._zero_dq(rank_dq_coordinates, mdQ, 1, token_idx, batch_idx, mtx)
                    self._zero_dq_tail(mdQ, token_idx, batch_idx, rank, mtx)
        elif warp_idx < Int32(self.MMA_WARP):
            rtx = tidx - Int32(self.REDUCE_THREAD_BEGIN)
            if cutlass.const_expr(self.max_topk <= self.O_PREFETCH_MAX_TOPK):
                self._prefetch_odo_rows(mOut, token_idx, batch_idx, rank, rtx)
            self._compute_folded_lse(mLSE, mAttnSink, softmax_stats, token_idx, batch_idx, rank, rtx)
            cute.arch.fence_view_async_shared()
            if my_tile_count > Int32(0):
                self.stats_lse_barrier.arrive_unaligned()
            if my_tile_count > Int32(0):
                self._compute_stationary_odo(mOut, mdO, stationary_do_physical, stationary_tma_mbars, softmax_stats, token_idx, batch_idx, rank, rtx)
            else:
                self._compute_global_odo(mOut, mdO, softmax_stats, token_idx, batch_idx, rank, rtx)
            cute.arch.fence_view_async_shared()
            if my_tile_count > Int32(0):
                self.stats_odo_barrier.arrive_unaligned()
            self.dsink_reducer_barrier.arrive_and_wait()
            if is_dq_owner:
                self._publish_dsink_stats(sum_odo, scaled_lse, mAttnSink, mdSink, softmax_stats, token_idx, batch_idx, rank, rtx)
            dkv_wait = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            dkv_rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            index_row = mTopkIdxs[None, (token_idx, batch_idx)]
            drain_count = Int32(0)
            if has_dkv:
                drain_count = my_tile_count
            for loop_iter in cutlass.range(drain_count):
                tile_index = walk_base + walk_step * loop_iter
                dkv_wait, dkv_rel = self._drain_dkv(
                    t_dkv[0], t_dkv[1], mdKV_acc, index_row, tile_index, topk, batch_idx, rtx, rank, pipe_dkv_done, dkv_wait, dkv_rel
                )
            cute.arch.setmaxregister_decrease(self.REDUCER_EPI_SETMAXREG)
        elif warp_idx == Int32(self.MMA_WARP):
            if is_leader_cta:
                s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SCORE_DONE_STAGES)
                dp_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.DP_DONE_STAGES)
                kscore_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                if cutlass.const_expr(split_regime):
                    round_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ROUND_STAGES)
                    pds_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    dkv_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                    dkv_com = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                else:
                    ds_free_leader_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    if cutlass.const_expr(self.max_topk != 128):
                        dq0_free_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                dq_done_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                dq0_done_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                if my_tile_count > Int32(0):
                    _mbarrier_wait_acquire_cluster(stationary_ready_mbar, Int32(0))
                    if is_dq_owner:
                        pipe_dq0_done.producer_acquire(dq0_done_prod)
                if is_dq_owner:
                    pipe_dq_done.producer_acquire(dq_done_prod)
                relay_phase = Int32(0)
                for loop_iter in cutlass.range(my_tile_count):
                    pipe_kscore.consumer_wait(kscore_cons)
                    s_prod = self._issue_score_with_tail(
                        score_tiled_mma, t_score, t_score_pp, score_q_fragment, score_k_fragment, score_qt_fragment, score_kt_fragment, pipe_s_done, s_prod
                    )
                    if loop_iter == Int32(0):
                        _mbarrier_wait_acquire_cluster(stationary_ready_mbar + 1, Int32(0))
                    dp_prod = self._issue_score(dp_tiled_mma, t_dp, t_dp, score_do_fragment, dp_k_fragment, pipe_dp_done, dp_prod)
                    pipe_kscore.consumer_release(kscore_cons)
                    kscore_cons.advance()
                    dq_acc = loop_iter != Int32(0)
                    if cutlass.const_expr(split_regime):
                        if has_dkv and is_dq_owner:
                            round_cons, kscore_cons, dkv_acq, dkv_com, pds_cons = self._issue_grads_early_dq(
                                dq_tiled_mma,
                                dkv_tiled_mma,
                                t_dq[0],
                                t_dq[1],
                                t_dkv[0],
                                t_dkv[1],
                                dq_kd_fragment_a,
                                dq_kd_fragment_b,
                                dq_ds_fragment,
                                dqt_tiled_mma,
                                t_dqt,
                                dqt_a_fragment,
                                dqt_b_fragment,
                                round_fragments[0 % len(round_fragments)],
                                round_fragments[1 % len(round_fragments)],
                                round_fragments[2 % len(round_fragments)],
                                round_fragments[3 % len(round_fragments)],
                                round_fragments[4 % len(round_fragments)],
                                round_fragments[5 % len(round_fragments)],
                                round_fragments[6 % len(round_fragments)],
                                round_fragments[7 % len(round_fragments)],
                                p_fragments[0],
                                p_fragments[1],
                                ds_fragments[0],
                                ds_fragments[1],
                                dq_acc,
                                relay_phase,
                                relay_mbars,
                                ds_local_ready_mbar,
                                pipe_round,
                                round_cons,
                                pipe_kscore,
                                kscore_cons,
                                pipe_dq0_done,
                                dq0_done_prod,
                                pipe_dq_done,
                                dq_done_prod,
                                loop_iter == my_tile_count - Int32(1),
                                pipe_pds,
                                pds_cons,
                                pipe_dkv_done,
                                dkv_acq,
                                dkv_com,
                            )
                        elif is_dq_owner:
                            kscore_cons, pds_cons = self._issue_grads_dq_only(
                                dq_tiled_mma,
                                t_dq[0],
                                t_dq[1],
                                dq_kd_fragment_a,
                                dq_kd_fragment_b,
                                dq_ds_fragment,
                                dqt_tiled_mma,
                                t_dqt,
                                dqt_a_fragment,
                                dqt_b_fragment,
                                dq_acc,
                                relay_phase,
                                ds_local_ready_mbar,
                                pipe_kscore,
                                kscore_cons,
                                pipe_dq0_done,
                                dq0_done_prod,
                                pipe_dq_done,
                                dq_done_prod,
                                loop_iter == my_tile_count - Int32(1),
                                pipe_pds,
                                pds_cons,
                            )
                        else:
                            round_cons, dkv_acq, dkv_com, pds_cons = self._issue_grads_dkv_only(
                                dkv_tiled_mma,
                                t_dkv[0],
                                t_dkv[1],
                                round_fragments[0 % len(round_fragments)],
                                round_fragments[1 % len(round_fragments)],
                                round_fragments[2 % len(round_fragments)],
                                round_fragments[3 % len(round_fragments)],
                                round_fragments[4 % len(round_fragments)],
                                round_fragments[5 % len(round_fragments)],
                                round_fragments[6 % len(round_fragments)],
                                round_fragments[7 % len(round_fragments)],
                                p_fragments[0],
                                p_fragments[1],
                                ds_fragments[0],
                                ds_fragments[1],
                                relay_phase,
                                relay_mbars,
                                pipe_round,
                                round_cons,
                                pipe_pds,
                                pds_cons,
                                pipe_dkv_done,
                                dkv_acq,
                                dkv_com,
                            )
                        pipe_pds.consumer_release(pds_cons)
                        pds_cons.advance()
                    else:
                        pipe_ds_free.consumer_wait(ds_free_leader_cons)
                        kscore_cons = self._issue_dq_rounds_early(
                            dq_tiled_mma,
                            t_dq[0],
                            t_dq[1],
                            dq_kd_fragment_a,
                            dq_kd_fragment_b,
                            dq_ds_fragment,
                            dqt_tiled_mma,
                            t_dqt,
                            dqt_a_fragment,
                            dqt_b_fragment,
                            dq_acc,
                            pipe_kscore,
                            kscore_cons,
                            pipe_dq0_done,
                            dq0_done_prod,
                            pipe_dq_done,
                            dq_done_prod,
                            loop_iter == my_tile_count - Int32(1),
                            pipe_dq0_free if self.max_topk != 128 else None,
                            dq0_free_prod if self.max_topk != 128 else None,
                        )
                        pipe_ds_free.consumer_release(ds_free_leader_cons)
                        ds_free_leader_cons.advance()
                    relay_phase = Int32(1) - relay_phase
                if my_tile_count > Int32(0):
                    pipe_s_done.producer_tail(s_prod)
                    pipe_dp_done.producer_tail(dp_prod)
                    if cutlass.const_expr(split_regime):
                        if has_dkv:
                            pipe_dkv_done.producer_tail(dkv_com)
                    if is_dq_owner:
                        dq0_done_prod.advance()
                        dq_done_prod.advance()
                        pipe_dq_done.producer_tail(dq_done_prod)
                        pipe_dq0_done.producer_tail(dq0_done_prod)
        elif warp_idx == Int32(self.TAIL_WARP):
            if cutlass.const_expr(not split_regime):
                if is_leader_cta:
                    round_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ROUND_STAGES)
                    p_free_tail_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    ds_free_tail_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    dkv_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                    dkv_com = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                    relay_phase = Int32(0)
                    for loop_iter in cutlass.range(my_tile_count):
                        pipe_p_free.consumer_wait(p_free_tail_cons)
                        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
                        pipe_dkv_done.producer_acquire(dkv_acq)
                        dkv_acq.advance()
                        round_cons = self._issue_dkv_sweep(
                            dkv_tiled_mma,
                            t_dkv[0],
                            round_fragments[0 % len(round_fragments)],
                            round_fragments[1 % len(round_fragments)],
                            round_fragments[2 % len(round_fragments)],
                            round_fragments[3 % len(round_fragments)],
                            round_fragments[4 % len(round_fragments)],
                            round_fragments[5 % len(round_fragments)],
                            round_fragments[6 % len(round_fragments)],
                            round_fragments[7 % len(round_fragments)],
                            p_fragments[0],
                            p_fragments[1],
                            False,
                            pipe_round,
                            round_cons,
                        )
                        pipe_dkv_done.producer_acquire(dkv_acq)
                        dkv_acq.advance()
                        round_cons = self._issue_dkv_sweep(
                            dkv_tiled_mma,
                            t_dkv[1],
                            round_fragments[0 % len(round_fragments)],
                            round_fragments[1 % len(round_fragments)],
                            round_fragments[2 % len(round_fragments)],
                            round_fragments[3 % len(round_fragments)],
                            round_fragments[4 % len(round_fragments)],
                            round_fragments[5 % len(round_fragments)],
                            round_fragments[6 % len(round_fragments)],
                            round_fragments[7 % len(round_fragments)],
                            p_fragments[0],
                            p_fragments[1],
                            False,
                            pipe_round,
                            round_cons,
                        )
                        pipe_p_free.consumer_release(p_free_tail_cons)
                        p_free_tail_cons.advance()
                        pipe_ds_free.consumer_wait(ds_free_tail_cons)
                        _mbarrier_wait_acquire_cluster(relay_mbars + 1, relay_phase)
                        round_cons = self._issue_dkv_sweep(
                            dkv_tiled_mma,
                            t_dkv[0],
                            round_fragments[0 % len(round_fragments)],
                            round_fragments[1 % len(round_fragments)],
                            round_fragments[2 % len(round_fragments)],
                            round_fragments[3 % len(round_fragments)],
                            round_fragments[4 % len(round_fragments)],
                            round_fragments[5 % len(round_fragments)],
                            round_fragments[6 % len(round_fragments)],
                            round_fragments[7 % len(round_fragments)],
                            ds_fragments[0],
                            ds_fragments[1],
                            True,
                            pipe_round,
                            round_cons,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        pipe_dkv_done.producer_commit(dkv_com)
                        dkv_com.advance()
                        round_cons = self._issue_dkv_sweep(
                            dkv_tiled_mma,
                            t_dkv[1],
                            round_fragments[0 % len(round_fragments)],
                            round_fragments[1 % len(round_fragments)],
                            round_fragments[2 % len(round_fragments)],
                            round_fragments[3 % len(round_fragments)],
                            round_fragments[4 % len(round_fragments)],
                            round_fragments[5 % len(round_fragments)],
                            round_fragments[6 % len(round_fragments)],
                            round_fragments[7 % len(round_fragments)],
                            ds_fragments[0],
                            ds_fragments[1],
                            True,
                            pipe_round,
                            round_cons,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        pipe_dkv_done.producer_commit(dkv_com)
                        dkv_com.advance()
                        pipe_ds_free.consumer_release(ds_free_tail_cons)
                        ds_free_tail_cons.advance()
                        relay_phase = Int32(1) - relay_phase
                    if my_tile_count > Int32(0):
                        pipe_dkv_done.producer_tail(dkv_com)
        elif warp_idx == Int32(self.LOAD_WARP):
            if my_tile_count > Int32(0):
                if cutlass.const_expr(self.max_topk != 128 and mTopkLength is not None):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                    cute.copy(tma_atom_q, t_q_gmem[None, rank, 0], t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
                    cute.copy(tma_atom_do, t_do_gmem[None, rank, 0], t_do_smem[None, 0], tma_bar_ptr=stationary_tma_mbars + 1)
                cute.arch.mbarrier_wait(stationary_tma_mbars, Int32(0))
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(stationary_ready_mbar, Int32(0))
                cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(stationary_ready_mbar + 1, Int32(0))
                round_count = Int32(0)
                if has_dkv:
                    round_count = my_tile_count
                for loop_iter in cutlass.range(round_count):
                    gens_per_group = self.H_TILE_CLUSTER // self.ROUND_K_HEADS
                    for micro_gen in cutlass.range_constexpr(self.ROUND_GENS_PER_TILE):
                        if cutlass.const_expr(split_regime):
                            grad_round = micro_gen // (2 * gens_per_group)
                            tensor_kind = micro_gen // gens_per_group % 2
                        else:
                            tensor_kind = micro_gen // (2 * gens_per_group)
                            grad_round = micro_gen // gens_per_group % 2
                        h_half = micro_gen % gens_per_group
                        round_slot = micro_gen % self.ROUND_STAGES
                        round_acq = pipeline.PipelineState(
                            self.ROUND_STAGES,
                            loop_iter * Int32(self.ROUND_GENS_PER_TILE) + Int32(micro_gen),
                            Int32(round_slot),
                            Int32(1 ^ micro_gen // self.ROUND_STAGES & 1),
                        )
                        pipe_round.producer_acquire(round_acq)
                        round_completion_mbar = pipe_round.producer_get_barrier(round_acq)
                        if cutlass.const_expr(tensor_kind == 0):
                            cute.copy(
                                round_tma_atom_dot,
                                t_dot_round_gmem[None, grad_round, h_half],
                                t_dot_round_smem[round_slot][None, 0],
                                tma_bar_ptr=round_completion_mbar,
                            )
                        else:
                            cute.copy(
                                round_tma_atom_qt,
                                t_qt_round_gmem[None, grad_round, h_half],
                                t_qt_round_smem[round_slot][None, 0],
                                tma_bar_ptr=round_completion_mbar,
                            )
                if has_dkv:
                    round_tail = pipeline.PipelineState(self.ROUND_STAGES, my_tile_count * Int32(self.ROUND_GENS_PER_TILE), Int32(0), Int32(1))
                    pipe_round.producer_tail(round_tail)
            elif cutlass.const_expr(mTopkLength is not None and self.max_topk == 128):
                cute.arch.mbarrier_wait(stationary_tma_mbars, Int32(0))
                cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
        elif warp_idx == Int32(self.RELAY_WARP):
            relay_lane = tidx % Int32(32)
            if relay_lane == Int32(0):
                for loop_iter in cutlass.range(my_tile_count):
                    cute.arch.mbarrier_wait(p_ready_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[0], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                    else:
                        _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[1], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                    _mbarrier_wait_acquire_cluster(landing_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(relay_mbars, Int32(0))
                    cute.arch.mbarrier_wait(pds_ready_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(ds_image_raw + Int32(2048), ds_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                    else:
                        _cpasync_bulk_s2cluster(ds_image_raw, ds_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                    if cutlass.const_expr(split_regime):
                        pds_com = pipeline.PipelineState(1, loop_iter, Int32(0), Int32(1) ^ loop_iter & Int32(1))
                        pipe_pds.producer_commit(pds_com)
                    _mbarrier_wait_acquire_cluster(landing_mbars + 1, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(relay_mbars + 1, Int32(0))
                if cutlass.const_expr(split_regime):
                    if my_tile_count > Int32(0):
                        pds_tail = pipeline.PipelineState(1, my_tile_count, Int32(0), Int32(1) ^ my_tile_count & Int32(1))
                        pipe_pds.producer_tail(pds_tail)
        tmem.relinquish_alloc_permit()
        self.cta_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp_idx == Int32(self.MATH_WARP_BEGIN):
            cute.arch.dealloc_tmem(tmem_ptr, self.TMEM_COLUMNS, is_two_cta=True)

    @staticmethod
    def _make_umma_async_pipeline(num_stages, producer_group, consumer_group, barrier_storage, cluster_layout_vmnk):
        """UMMA-producer pipeline whose consumer releases target the leader.

        ``consumer_mask=0`` is the destination CTA rank for AsyncThread
        arrivals: both CTAs' consumers arrive on rank 0's empty barrier, the
        only one the leader's producer_acquire waits on, while the leader's
        tcgen05 commits remain cluster-wide.
        """
        pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            barrier_storage=barrier_storage,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        return pipeline.PipelineUmmaAsync(
            sync_object_full=pipe.sync_object_full,
            sync_object_empty=pipe.sync_object_empty,
            num_stages=pipe.num_stages,
            producer_mask=pipe.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe.cta_group,
        )

    @staticmethod
    def _make_async_umma_pipeline(num_stages, producer_group, consumer_group, barrier_storage, cluster_layout_vmnk):
        """UMMA-consumer pipeline whose producer commits target the leader.

        ``producer_mask=0`` is the destination CTA rank for AsyncThread
        arrivals: both CTAs' producers arrive on rank 0's full barrier, the
        only one the leader's consumer_wait polls, while the leader's
        tcgen05 releases remain cluster-wide.
        """
        pipe = pipeline.PipelineAsyncUmma.create(
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            barrier_storage=barrier_storage,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        return pipeline.PipelineAsyncUmma(
            sync_object_full=pipe.sync_object_full,
            sync_object_empty=pipe.sync_object_empty,
            num_stages=pipe.num_stages,
            producer_mask=Int32(0),
            consumer_mask=pipe.consumer_mask,
            cta_group=pipe.cta_group,
        )

    def _make_shared_storage(
        self, score_a_layout_staged, score_b_layout_staged, dkv_a_layout_staged, dkv_b_layout_staged, dq_a_layout_staged, dq_b_layout_staged, split_regime
    ):
        """Lay out the mbarrier arrays and the staged operand buffers."""
        element_dtype = self.element_dtype
        round_buf_elements = self.ROUND_BUF_ELEMENTS
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 8192
        assert cute.cosize(score_b_layout_staged) == 2 * cute.cosize(dkv_a_layout_staged)
        assert cute.cosize(dkv_a_layout_staged) == 8192
        assert cute.cosize(dkv_b_layout_staged) <= 2048
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096
        pds_mbar_count = 2 if split_regime else 4

        @cute.struct
        class SharedStorage:
            s_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dp_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            kscore_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_mbars: cute.struct.MemRange[cutlass.Int64, 16]
            pds_mbars: cute.struct.MemRange[cutlass.Int64, pds_mbar_count]
            dq0_free_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dkv_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dq_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dq0_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            landing_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            relay_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            pds_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            p_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            ds_local_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tail_ld_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            stationary_q: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            stationary_do: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            score_kv: cute.struct.Align[cute.struct.MemRange[element_dtype, 16384], 1024]
            round_buf: cute.struct.Align[cute.struct.MemRange[element_dtype, round_buf_elements], 1024]
            q_tail: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            kt_score: cute.struct.Align[cute.struct.MemRange[element_dtype, 2048], 1024]
            kt_dq: cute.struct.Align[cute.struct.MemRange[element_dtype, 2048], 1024]
            p_blocks: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            p_xchg: cute.struct.Align[cute.struct.MemRange[element_dtype, 2048], 1024]
            ds_image: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            ds_blocks: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            stats: cute.struct.Align[cute.struct.MemRange[Float32, 128], 1024]
            tile_indices: cute.struct.Align[cute.struct.MemRange[Int32, 128], 16]

        assert SharedStorage.size_in_bytes() <= self.MAX_SMEM_BYTES
        return SharedStorage

    @cute.jit
    def _issue_dq_rounds_early(
        self,
        dq_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        kd_fragment_a: cute.Tensor,
        kd_fragment_b: cute.Tensor,
        ds_fragment: cute.Tensor,
        dqt_tiled_mma: cute.TiledMma,
        t_dqt: cute.Tensor,
        dqt_a_fragment: cute.Tensor,
        dqt_b_fragment: cute.Tensor,
        accumulate: cutlass.Boolean,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
        dq0_done_pipeline,
        dq0_done_state: pipeline.PipelineState,
        dq1_done_pipeline,
        dq1_done_state: pipeline.PipelineState,
        commit_final: cutlass.Boolean,
        dq0_free_pipeline=None,
        dq0_free_commit_state: Optional[pipeline.PipelineState] = None,
    ) -> pipeline.PipelineState:
        """Commit DQ0 and DQ1 at their own final UMMA boundaries."""
        kscore_pipeline.consumer_wait(kscore_consumer_state)
        assert cute.size(kd_fragment_a, mode=[2]) == 4
        assert cute.size(kd_fragment_b, mode=[2]) == 4
        for round_index in cutlass.range_constexpr(self.D_ROUNDS):
            mma = dq_tiled_mma.with_()
            mma.set(tcgen05.Field.ACCUMULATE, accumulate)
            if cutlass.const_expr(round_index == 0):
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_a, mode=[2])):
                    cute.gemm(mma, t_dq_0, kd_fragment_a[None, None, k_block, 0], ds_fragment[None, None, k_block, 0], t_dq_0)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                if cutlass.const_expr(dq0_free_pipeline is not None):
                    dq0_free_pipeline.producer_commit(dq0_free_commit_state)
                    dq0_free_commit_state.advance()
                if commit_final:
                    cute.arch.fence_view_async_tmem_store()
                    dq0_done_pipeline.producer_commit(dq0_done_state)
            else:
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_b, mode=[2])):
                    cute.gemm(mma, t_dq_1, kd_fragment_b[None, None, k_block, 0], ds_fragment[None, None, k_block, 0], t_dq_1)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.arch.fence_view_async_tmem_store()
        kscore_pipeline.consumer_release(kscore_consumer_state)
        kscore_consumer_state.advance()
        dqt_mma = dqt_tiled_mma.with_()
        dqt_mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        for k_block in cutlass.range_constexpr(cute.size(dqt_a_fragment, mode=[2])):
            cute.gemm(dqt_mma, t_dqt, dqt_a_fragment[None, None, k_block, 0], dqt_b_fragment[None, None, k_block, 0], t_dqt)
            dqt_mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.arch.fence_view_async_tmem_store()
        if commit_final:
            dq1_done_pipeline.producer_commit(dq1_done_state)
        return kscore_consumer_state

    @cute.jit
    def _issue_grads_dq_only(
        self,
        dq_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        dq_kd_fragment_a: cute.Tensor,
        dq_kd_fragment_b: cute.Tensor,
        dq_ds_fragment: cute.Tensor,
        dqt_tiled_mma: cute.TiledMma,
        t_dqt: cute.Tensor,
        dqt_a_fragment: cute.Tensor,
        dqt_b_fragment: cute.Tensor,
        dq_accumulate: cutlass.Boolean,
        relay_phase: Int32,
        ds_local_ready_mbar: cute.Pointer,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
        dq0_done_pipeline,
        dq0_done_state: pipeline.PipelineState,
        dq1_done_pipeline,
        dq1_done_state: pipeline.PipelineState,
        commit_final: cutlass.Boolean,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
    ):
        """Split-owner tile step: dQ (and its tail) without any dKV work."""
        _mbarrier_wait_acquire_cluster(ds_local_ready_mbar, relay_phase)
        kscore_consumer_state = self._issue_dq_rounds_early(
            dq_tiled_mma,
            t_dq_0,
            t_dq_1,
            dq_kd_fragment_a,
            dq_kd_fragment_b,
            dq_ds_fragment,
            dqt_tiled_mma,
            t_dqt,
            dqt_a_fragment,
            dqt_b_fragment,
            dq_accumulate,
            kscore_pipeline,
            kscore_consumer_state,
            dq0_done_pipeline,
            dq0_done_state,
            dq1_done_pipeline,
            dq1_done_state,
            commit_final,
        )
        return (kscore_consumer_state, pds_consumer_state)

    @cute.jit
    def _finish_dkv_panels(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        round_fragments,
        p_fragments,
        ds_fragments,
        relay_phase: Int32,
        relay_mbars: cute.Pointer,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_acquire_state: pipeline.PipelineState,
        dkv_commit_state: pipeline.PipelineState,
    ):
        """Finish dK0, then issue dV1/dK1 after the caller's first dV sweep."""
        _mbarrier_wait_acquire_cluster(relay_mbars + 1, relay_phase)
        round_consumer_state = self._issue_dkv_sweep(dkv_tiled_mma, t_dkv_0, *round_fragments, *ds_fragments, True, round_pipeline, round_consumer_state)
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_consumer_state = self._issue_dkv_sweep(dkv_tiled_mma, t_dkv_1, *round_fragments, *p_fragments, False, round_pipeline, round_consumer_state)
        round_consumer_state = self._issue_dkv_sweep(dkv_tiled_mma, t_dkv_1, *round_fragments, *ds_fragments, True, round_pipeline, round_consumer_state)
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        return (round_consumer_state, dkv_acquire_state, dkv_commit_state)

    @cute.jit
    def _issue_grads_dkv_only(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        round_fragment_0: cute.Tensor,
        round_fragment_1: cute.Tensor,
        round_fragment_2: cute.Tensor,
        round_fragment_3: cute.Tensor,
        round_fragment_4: cute.Tensor,
        round_fragment_5: cute.Tensor,
        round_fragment_6: cute.Tensor,
        round_fragment_7: cute.Tensor,
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        relay_phase: Int32,
        relay_mbars: cute.Pointer,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_acquire_state: pipeline.PipelineState,
        dkv_commit_state: pipeline.PipelineState,
    ):
        """Split-worker tile step: dV/dK sweeps without any dQ work."""
        round_fragments = (
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
        )
        p_fragments = (p_fragment_0, p_fragment_1)
        ds_fragments = (ds_fragment_0, ds_fragment_1)
        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_consumer_state = self._issue_dkv_sweep(dkv_tiled_mma, t_dkv_0, *round_fragments, *p_fragments, False, round_pipeline, round_consumer_state)
        pds_pipeline.consumer_wait(pds_consumer_state)
        round_consumer_state, dkv_acquire_state, dkv_commit_state = self._finish_dkv_panels(
            dkv_tiled_mma,
            t_dkv_0,
            t_dkv_1,
            round_fragments,
            p_fragments,
            ds_fragments,
            relay_phase,
            relay_mbars,
            round_pipeline,
            round_consumer_state,
            dkv_done_pipeline,
            dkv_acquire_state,
            dkv_commit_state,
        )
        return (round_consumer_state, dkv_acquire_state, dkv_commit_state, pds_consumer_state)

    @cute.jit
    def _issue_grads_early_dq(
        self,
        dq_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        dq_kd_fragment_a: cute.Tensor,
        dq_kd_fragment_b: cute.Tensor,
        dq_ds_fragment: cute.Tensor,
        dqt_tiled_mma: cute.TiledMma,
        t_dqt: cute.Tensor,
        dqt_a_fragment: cute.Tensor,
        dqt_b_fragment: cute.Tensor,
        round_fragment_0: cute.Tensor,
        round_fragment_1: cute.Tensor,
        round_fragment_2: cute.Tensor,
        round_fragment_3: cute.Tensor,
        round_fragment_4: cute.Tensor,
        round_fragment_5: cute.Tensor,
        round_fragment_6: cute.Tensor,
        round_fragment_7: cute.Tensor,
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        dq_accumulate: cutlass.Boolean,
        relay_phase: Int32,
        relay_mbars: cute.Pointer,
        ds_local_ready_mbar: cute.Pointer,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
        dq0_done_pipeline,
        dq0_done_state: pipeline.PipelineState,
        dq1_done_pipeline,
        dq1_done_state: pipeline.PipelineState,
        commit_final: cutlass.Boolean,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_acquire_state: pipeline.PipelineState,
        dkv_commit_state: pipeline.PipelineState,
    ):
        """Keep the dV/dK issue order and split final dQ completion."""
        round_fragments = (
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
        )
        p_fragments = (p_fragment_0, p_fragment_1)
        ds_fragments = (ds_fragment_0, ds_fragment_1)
        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_consumer_state = self._issue_dkv_sweep(dkv_tiled_mma, t_dkv_0, *round_fragments, *p_fragments, False, round_pipeline, round_consumer_state)
        _mbarrier_wait_acquire_cluster(ds_local_ready_mbar, relay_phase)
        kscore_consumer_state = self._issue_dq_rounds_early(
            dq_tiled_mma,
            t_dq_0,
            t_dq_1,
            dq_kd_fragment_a,
            dq_kd_fragment_b,
            dq_ds_fragment,
            dqt_tiled_mma,
            t_dqt,
            dqt_a_fragment,
            dqt_b_fragment,
            dq_accumulate,
            kscore_pipeline,
            kscore_consumer_state,
            dq0_done_pipeline,
            dq0_done_state,
            dq1_done_pipeline,
            dq1_done_state,
            commit_final,
        )
        round_consumer_state, dkv_acquire_state, dkv_commit_state = self._finish_dkv_panels(
            dkv_tiled_mma,
            t_dkv_0,
            t_dkv_1,
            round_fragments,
            p_fragments,
            ds_fragments,
            relay_phase,
            relay_mbars,
            round_pipeline,
            round_consumer_state,
            dkv_done_pipeline,
            dkv_acquire_state,
            dkv_commit_state,
        )
        return (round_consumer_state, kscore_consumer_state, dkv_acquire_state, dkv_commit_state, pds_consumer_state)

    @cute.jit
    def _load_q_tail(self, mQ, destination, query, batch, rank, tid, copy_atom, thread_copy):
        """One-time gather of the CTA's own 64 heads of Q[:, 512:576].

        The destination is the K-major score-A tail slab whose shared-memory
        bytes also feed the dK-tail warp MMA.
        """
        index_in_group = tid % self.KV_GROUP_SIZE
        group_index = tid // self.KV_GROUP_SIZE
        for row_iteration in cutlass.range_constexpr(self.H_TILE_CTA // self.KV_NUM_GROUPS):
            h_local = Int32(row_iteration * self.KV_NUM_GROUPS) + group_index
            head = rank * Int32(self.H_TILE_CTA) + h_local
            destination_row = destination[h_local, None]
            destination_chunks = cute.flat_divide(destination_row, (8,))
            if head < Int32(self.num_heads):
                source_offset = mQ.iterator + mQ.layout((head, Int32(self.D_HEAD), (query, batch)))
                source_row = cute.make_tensor(
                    cute.make_ptr(self.element_dtype, source_offset.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.D_TAIL,))
                )
                source_chunks = cute.flat_divide(source_row, (8,))
                thread_source = thread_copy.partition_S(source_chunks[None, index_in_group])
                thread_destination = thread_copy.partition_D(destination_chunks[None, index_in_group])
                cute.copy(copy_atom, thread_source, thread_destination)
            else:
                destination_chunks[None, index_in_group].fill(0.0)

    @cute.jit
    def _copy_k_tail_row(self, mKV, destination_rows, destination_row, kv_index, batch_idx, index_in_group, copy_atom, thread_copy):
        """Copy one sparse KV row's D64 tail with 128-bit cp.async."""
        source_row_full = mKV[kv_index, None, (0, batch_idx)]
        source_row_offset = source_row_full.iterator + Int32(self.D_HEAD)
        source_row = cute.make_tensor(
            cute.make_ptr(self.element_dtype, source_row_offset.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.D_TAIL,))
        )
        source_chunks = cute.flat_divide(source_row, (8,))
        destination_row_tensor = destination_rows[destination_row, None]
        destination_chunks = cute.flat_divide(destination_row_tensor, (8,))
        thread_source = thread_copy.partition_S(source_chunks[None, index_in_group])
        thread_destination = thread_copy.partition_D(destination_chunks[None, index_in_group])
        cute.copy(copy_atom, thread_source, thread_destination)

    @cute.jit
    def _zero_k_tail_row(self, destination_rows, destination_row, index_in_group):
        """Zero one 8-element chunk of a K tail row in the destination buffer."""
        destination_row_tensor = destination_rows[destination_row, None]
        destination_chunks = cute.flat_divide(destination_row_tensor, (8,))
        destination_chunks[None, index_in_group].fill(0.0)

    @cute.jit
    def _prefetch_kv_rows_l2(self, mKV: cute.Tensor, tile_indices: cute.Tensor, tile_index: Int32, topk: Int32, rank: Int32, role_tidx: Int32) -> None:
        """Warm the next score rows with short-loop or steady-state ownership."""
        seqlen_kv = cute.size(mKV, mode=[0])
        if cutlass.const_expr(self.max_topk == 128):
            local_n = role_tidx // Int32(2)
            line_half = role_tidx % Int32(2)
            global_n = tile_index * Int32(self.N_TILE) + local_n
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = tile_indices[local_n]
            if kv_index >= Int32(0) and kv_index < seqlen_kv:
                row_base = mKV.iterator + cute.crd2idx((kv_index, Int32(0), (0, Int32(0))), mKV.layout)
                if line_half == Int32(0):
                    for line in cutlass.range_constexpr(5):
                        _prefetch_o_row_l2(row_base + Int32(64 * line))
                else:
                    for line in cutlass.range_constexpr(4):
                        _prefetch_o_row_l2(row_base + Int32(64 * (5 + line)))
        else:
            for wave in cutlass.range_constexpr(1):
                linear = role_tidx + Int32(wave * self.GATHER_THREADS)
                local_n = Int32(0)
                line = Int32(0)
                global_n = Int32(0)
                kv_index = Int32(-1)
                if linear < Int32(self.N_TILE_CTA * 8):
                    local_n = rank * Int32(self.N_TILE_CTA) + linear // Int32(8)
                    line = linear % Int32(8)
                    global_n = tile_index * Int32(self.N_TILE) + local_n
                    if global_n < topk:
                        kv_index = tile_indices[local_n]
                    if kv_index >= Int32(0) and kv_index < seqlen_kv:
                        row_base = mKV.iterator + cute.crd2idx((kv_index, Int32(0), (0, Int32(0))), mKV.layout)
                        _prefetch_o_row_l2(row_base + line * Int32(64))

    @cute.jit
    def _load_kt_dq_tail_values(self, mKV: cute.Tensor, batch_idx: Int32, rank: Int32, role_tidx: Int32, pre_kv_index: Int32):
        """Load this thread's two K-tail global vectors into registers.

        These loads do not access shared state and can precede the kscore
        producer acquire. The paired store helper runs after the acquire,
        once the destination generation is available.
        """
        seqlen_kv = cute.size(mKV, mode=[0])
        half = role_tidx % Int32(2)
        chunk_base = half * Int32(2)
        values_0 = cute.make_rmem_tensor((8,), self.element_dtype)
        values_1 = cute.make_rmem_tensor((8,), self.element_dtype)
        if pre_kv_index >= Int32(0) and pre_kv_index < seqlen_kv:
            source_row_full = mKV[pre_kv_index, None, (0, batch_idx)]
            source_offset = source_row_full.iterator + Int32(self.D_HEAD) + rank * Int32(self.D_TAIL // 2)
            source_row = cute.make_tensor(
                cute.make_ptr(self.element_dtype, source_offset.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.D_TAIL // 2,))
            )
            vector_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.element_dtype, num_bits_per_copy=128)
            cute.copy(vector_load, cute.local_tile(source_row, (8,), (chunk_base,)), values_0)
            cute.copy(vector_load, cute.local_tile(source_row, (8,), (chunk_base + Int32(1),)), values_1)
        else:
            for lane in cutlass.range_constexpr(8):
                values_0[lane] = self.element_dtype(0.0)
                values_1[lane] = self.element_dtype(0.0)
        return (values_0, values_1)

    @cute.jit
    def _store_kt_dq_tail_values(self, destination: cute.Tensor, role_tidx: Int32, values_0: cute.Tensor, values_1: cute.Tensor) -> None:
        """Transpose the pre-loaded K-tail registers into the dQ-tail B slab."""
        half = role_tidx % Int32(2)
        row = role_tidx // Int32(2)
        chunk_base = half * Int32(2)
        for lane in cutlass.range_constexpr(8):
            destination[chunk_base * Int32(8) + Int32(lane), row] = values_0[lane]
        for lane in cutlass.range_constexpr(8):
            destination[(chunk_base + Int32(1)) * Int32(8) + Int32(lane), row] = values_1[lane]

    @cute.jit
    def _gather_kt_dq(
        self,
        mKV: cute.Tensor,
        tile_indices: cute.Tensor,
        destination: cute.Tensor,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        role_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        pre_kv_index: Optional[Int32] = None,
    ) -> None:
        """Gather this CTA's 32-column K-tail transpose half for all 64 rows.

        The K-major CG2 dQ-tail B operand holds 32 rank-owned tail dimensions
        by 64 KV slots. Its swizzled layout requires register transposition
        from aligned global vectors to layout-aware scalar shared stores.
        Invalid rows are zero-filled before the dS MMA. ``pre_kv_index``
        optionally supplies this thread's pre-resolved sparse row index.
        """
        seqlen_kv = cute.size(mKV, mode=[0])
        half = role_tidx % Int32(2)
        row = role_tidx // Int32(2)
        if cutlass.const_expr(pre_kv_index is not None):
            kv_index = pre_kv_index
        else:
            global_n = tile_index * Int32(self.N_TILE) + row
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = tile_indices[row]
        chunk_base = half * Int32(2)
        if kv_index >= Int32(0) and kv_index < seqlen_kv:
            source_row_full = mKV[kv_index, None, (0, batch_idx)]
            source_offset = source_row_full.iterator + Int32(self.D_HEAD) + rank * Int32(self.D_TAIL // 2)
            source_row = cute.make_tensor(
                cute.make_ptr(self.element_dtype, source_offset.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.D_TAIL // 2,))
            )
            vector_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.element_dtype, num_bits_per_copy=128)
            values_0 = cute.make_rmem_tensor((8,), self.element_dtype)
            values_1 = cute.make_rmem_tensor((8,), self.element_dtype)
            cute.copy(vector_load, cute.local_tile(source_row, (8,), (chunk_base,)), values_0)
            cute.copy(vector_load, cute.local_tile(source_row, (8,), (chunk_base + Int32(1),)), values_1)
            for lane in cutlass.range_constexpr(8):
                destination[chunk_base * Int32(8) + Int32(lane), row] = values_0[lane]
            for lane in cutlass.range_constexpr(8):
                destination[(chunk_base + Int32(1)) * Int32(8) + Int32(lane), row] = values_1[lane]
        else:
            for subchunk in cutlass.range_constexpr(2):
                chunk = chunk_base + Int32(subchunk)
                for lane in cutlass.range_constexpr(8):
                    destination[chunk * Int32(8) + Int32(lane), row] = self.element_dtype(0.0)

    @cute.jit
    def _accumulate_dk_tail_wmma(
        self,
        dkt_warp_mma: cute.TiledMma,
        s_ds_kv_h: cute.Tensor,
        s_qt_d_h: cute.Tensor,
        indices: cute.Tensor,
        dkv: cute.Tensor,
        query: Int32,
        batch: Int32,
        tile: Int32,
        topk: Int32,
        mtx: Int32,
        tail_ld_mbar: cute.Pointer,
    ) -> None:
        """Accumulate this CTA's own-head dK-tail partial on the warp MMA.

        The four gather warps contract dS^T (kv x h) against the Q-tail
        transpose (d x h) with FP32 accumulators, then add the 64x64 partial
        into dKV columns 512..575 with one FP32 atomic per element.
        """
        thr_mma = dkt_warp_mma.get_slice(mtx)
        tCsA = thr_mma.partition_A(s_ds_kv_h)
        tCrA = dkt_warp_mma.make_fragment_A(tCsA)
        atom_ld_a = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), self.element_dtype)
        atom_ld_b = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), self.element_dtype)
        tiled_copy_a = cute.make_tiled_copy_A(atom_ld_a, dkt_warp_mma)
        tiled_copy_b = cute.make_tiled_copy_B(atom_ld_b, dkt_warp_mma)
        thr_copy_a = tiled_copy_a.get_slice(mtx)
        thr_copy_b = tiled_copy_b.get_slice(mtx)
        tCsA_view = thr_copy_a.partition_S(s_ds_kv_h)
        tCrA_view = thr_copy_a.retile(tCrA)
        coords = thr_mma.partition_C(cute.make_identity_tensor((self.N_TILE, 32)))
        seqlen_kv = cute.size(dkv, mode=[1])
        row_lo = Int32(cute.get(coords[0], mode=[0]))
        row_hi = row_lo
        for probe in cutlass.range_constexpr(cute.size(coords)):
            cand = Int32(cute.get(coords[probe], mode=[0]))
            if cand != row_lo:
                row_hi = cand
        idx_lo = Int32(-1)
        idx_hi = Int32(-1)
        if tile * Int32(self.N_TILE) + row_lo < topk:
            idx_lo = indices[tile * Int32(self.N_TILE) + row_lo, (query, batch)]
        if tile * Int32(self.N_TILE) + row_hi < topk:
            idx_hi = indices[tile * Int32(self.N_TILE) + row_hi, (query, batch)]
        cute.copy(tiled_copy_a, tCsA_view, tCrA_view)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(tail_ld_mbar)
        for n_half in cutlass.range_constexpr(2):
            sub_qt = cute.local_tile(s_qt_d_h, (32, self.H_TILE_CTA), (n_half, 0))
            tCsB = thr_mma.partition_B(sub_qt)
            tCrB = dkt_warp_mma.make_fragment_B(tCsB)
            tCsB_view = thr_copy_b.partition_S(sub_qt)
            tCrB_view = thr_copy_b.retile(tCrB)
            cute.copy(tiled_copy_b, tCsB_view, tCrB_view)
            acc_shape = dkt_warp_mma.partition_shape_C((self.N_TILE, 32))
            tCrC = dkt_warp_mma.make_fragment_C(acc_shape)
            tCrC.fill(0.0)
            cute.gemm(dkt_warp_mma, tCrC, tCrA, tCrB, tCrC)
            assert cute.size(tCrC) % 2 == 0
            d_half_base = Int32(self.D_HEAD) + Int32(n_half * 32)
            rank_of_cta = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            if cutlass.const_expr(self.deterministic):
                owner_warp = mtx // Int32(32)
                owner_lane_group = mtx % Int32(32) // Int32(4)
                lane_group = Int32(0)
                for active_warp in cutlass.range(4):
                    if owner_warp == Int32(active_warp):
                        for lane_group in cutlass.range(8):
                            if owner_lane_group == Int32(lane_group):
                                self._deposit_dk_tail_ordered(coords, tCrC, dkv, idx_lo, idx_hi, row_lo, d_half_base, batch)
                            cute.arch.sync_warp()
                    self.gather_barrier.arrive_and_wait()
            elif rank_of_cta == Int32(0):
                for pair_index in cutlass.range_constexpr(cute.size(tCrC) // 2):
                    i0 = 2 * pair_index
                    i1 = 2 * pair_index + 1
                    kv0 = Int32(cute.get(coords[i0], mode=[0]))
                    d0 = Int32(cute.get(coords[i0], mode=[1]))
                    kv1 = Int32(cute.get(coords[i1], mode=[0]))
                    d1 = Int32(cute.get(coords[i1], mode=[1]))
                    pair_ok = kv1 == kv0 and d1 == d0 + Int32(1) and (d0 & Int32(1) == Int32(0))
                    if pair_ok:
                        row = idx_hi
                        if kv0 == row_lo:
                            row = idx_lo
                        if row >= Int32(0) and row < seqlen_kv:
                            pointer = dkv.iterator + cute.crd2idx((d_half_base + d0, row, (0, batch)), dkv.layout)
                            pair_values = cute.make_rmem_tensor((2,), self.acc_dtype)
                            pair_values[0] = tCrC[i0]
                            pair_values[1] = tCrC[i1]
                            cute.arch.atomic_add(pointer.llvm_ptr, pair_values.load())
                    else:
                        row = idx_hi
                        if kv0 == row_lo:
                            row = idx_lo
                        if row >= Int32(0) and row < seqlen_kv:
                            pointer = dkv.iterator + cute.crd2idx((d_half_base + d0, row, (0, batch)), dkv.layout)
                            cute.arch.atomic_add(pointer.llvm_ptr, tCrC[i0])
                        row1 = idx_hi
                        if kv1 == row_lo:
                            row1 = idx_lo
                        if row1 >= Int32(0) and row1 < seqlen_kv:
                            pointer1 = dkv.iterator + cute.crd2idx((d_half_base + d1, row1, (0, batch)), dkv.layout)
                            cute.arch.atomic_add(pointer1.llvm_ptr, tCrC[i1])
            else:
                for pair_step in cutlass.range_constexpr(cute.size(tCrC) // 2):
                    pair_index = (pair_step + cute.size(tCrC) // 4) % (cute.size(tCrC) // 2)
                    i0 = 2 * pair_index
                    i1 = 2 * pair_index + 1
                    kv0 = Int32(cute.get(coords[i0], mode=[0]))
                    d0 = Int32(cute.get(coords[i0], mode=[1]))
                    kv1 = Int32(cute.get(coords[i1], mode=[0]))
                    d1 = Int32(cute.get(coords[i1], mode=[1]))
                    pair_ok = kv1 == kv0 and d1 == d0 + Int32(1) and (d0 & Int32(1) == Int32(0))
                    if pair_ok:
                        row = idx_hi
                        if kv0 == row_lo:
                            row = idx_lo
                        if row >= Int32(0) and row < seqlen_kv:
                            pointer = dkv.iterator + cute.crd2idx((d_half_base + d0, row, (0, batch)), dkv.layout)
                            pair_values = cute.make_rmem_tensor((2,), self.acc_dtype)
                            pair_values[0] = tCrC[i0]
                            pair_values[1] = tCrC[i1]
                            cute.arch.atomic_add(pointer.llvm_ptr, pair_values.load())
                    else:
                        row = idx_hi
                        if kv0 == row_lo:
                            row = idx_lo
                        if row >= Int32(0) and row < seqlen_kv:
                            pointer = dkv.iterator + cute.crd2idx((d_half_base + d0, row, (0, batch)), dkv.layout)
                            cute.arch.atomic_add(pointer.llvm_ptr, tCrC[i0])
                        row1 = idx_hi
                        if kv1 == row_lo:
                            row1 = idx_lo
                        if row1 >= Int32(0) and row1 < seqlen_kv:
                            pointer1 = dkv.iterator + cute.crd2idx((d_half_base + d1, row1, (0, batch)), dkv.layout)
                            cute.arch.atomic_add(pointer1.llvm_ptr, tCrC[i1])

    @cute.jit
    def _deposit_dk_tail_ordered(self, coords, values, dkv, idx_lo: Int32, idx_hi: Int32, row_lo: Int32, d_half_base: Int32, batch: Int32):
        """Deposit one warp-TC row owner in fixed pair order."""
        seqlen_kv = cute.size(dkv, mode=[1])
        row = Int32(-1)
        row1 = Int32(-1)
        for pair_index in cutlass.range_constexpr(cute.size(values) // 2):
            i0 = 2 * pair_index
            i1 = 2 * pair_index + 1
            kv0 = Int32(cute.get(coords[i0], mode=[0]))
            d0 = Int32(cute.get(coords[i0], mode=[1]))
            kv1 = Int32(cute.get(coords[i1], mode=[0]))
            d1 = Int32(cute.get(coords[i1], mode=[1]))
            pair_ok = kv1 == kv0 and d1 == d0 + Int32(1) and (d0 & Int32(1) == Int32(0))
            if pair_ok:
                row = idx_hi
                if kv0 == row_lo:
                    row = idx_lo
                if row >= Int32(0) and row < seqlen_kv:
                    pointer = dkv.iterator + cute.crd2idx((d_half_base + d0, row, (0, batch)), dkv.layout)
                    pair_values = cute.make_rmem_tensor((2,), self.acc_dtype)
                    pair_values[0] = values[i0]
                    pair_values[1] = values[i1]
                    cute.arch.atomic_add(pointer.llvm_ptr, pair_values.load())
            else:
                row = idx_hi
                if kv0 == row_lo:
                    row = idx_lo
                if row >= Int32(0) and row < seqlen_kv:
                    pointer = dkv.iterator + cute.crd2idx((d_half_base + d0, row, (0, batch)), dkv.layout)
                    cute.arch.atomic_add(pointer.llvm_ptr, values[i0])
                row1 = idx_hi
                if kv1 == row_lo:
                    row1 = idx_lo
                if row1 >= Int32(0) and row1 < seqlen_kv:
                    pointer1 = dkv.iterator + cute.crd2idx((d_half_base + d1, row1, (0, batch)), dkv.layout)
                    cute.arch.atomic_add(pointer1.llvm_ptr, values[i1])

    @cute.jit
    def _store_dqt_epi(self, t_dqt, dqt_tiled_mma, dq, query, batch, rank, mtx):
        """Drain the persistent CG2 dQ-tail accumulator to dQ[:, 512:576]."""
        if mtx < self.MATH_THREADS_PER_CTA:
            rank_dqt_mma = dqt_tiled_mma.get_slice(rank)
            rank_dqt_coordinates = rank_dqt_mma.partition_C(cute.make_identity_tensor((self.H_TILE_CLUSTER, self.D_TAIL)))
            tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)), self.acc_dtype)
            dqt_copy = tcgen05.make_tmem_copy(tmem_load_atom, t_dqt)
            dqt_thread = dqt_copy.get_slice(mtx)
            dqt_source = dqt_thread.partition_S(t_dqt)
            dqt_coordinates = dqt_thread.partition_D(rank_dqt_coordinates)
            r_dqt = cute.make_rmem_tensor(dqt_coordinates.shape, self.acc_dtype)
            cute.copy(dqt_copy, dqt_source, r_dqt)
            cute.arch.fence_view_async_tmem_load()
            assert cute.size(r_dqt) % 2 == 0
            for pair_index in cutlass.range_constexpr(cute.size(r_dqt) // 2):
                i0 = 2 * pair_index
                i1 = 2 * pair_index + 1
                head = Int32(cute.get(dqt_coordinates[i0], mode=[0]))
                d0 = Int32(cute.get(dqt_coordinates[i0], mode=[1]))
                head1 = Int32(cute.get(dqt_coordinates[i1], mode=[0]))
                d1 = Int32(cute.get(dqt_coordinates[i1], mode=[1]))
                if head1 == head and d1 == d0 + Int32(1) and (d0 & Int32(1) == Int32(0)) and head < Int32(self.num_heads):
                    packed = _dq_pack_bf16x2(r_dqt[i0], r_dqt[i1])
                    destination = cute.make_tensor(
                        cute.recast_ptr(dq.iterator + cute.crd2idx((Int32(self.D_HEAD) + d0, head, (query, batch)), dq.layout), dtype=cutlass.Uint32),
                        cute.make_layout((1,)),
                    )
                    destination[0] = packed
                else:
                    if head < Int32(self.num_heads):
                        dq[Int32(self.D_HEAD) + d0, head, (query, batch)] = self.element_dtype(r_dqt[i0])
                    if head1 < Int32(self.num_heads):
                        dq[Int32(self.D_HEAD) + d1, head1, (query, batch)] = self.element_dtype(r_dqt[i1])

    @cute.jit
    def _zero_dq_tail(self, dq, query, batch, rank, tid):
        """Write the required all-zero dQ tail when no tile is issued."""
        for i in cutlass.range_constexpr(self.H_TILE_CTA * self.D_TAIL // self.MATH_THREADS):
            linear = tid + Int32(i * self.MATH_THREADS)
            h = rank * Int32(self.H_TILE_CTA) + linear // Int32(self.D_TAIL)
            d = Int32(self.D_HEAD) + linear % Int32(self.D_TAIL)
            if h < Int32(self.num_heads):
                dq[d, h, (query, batch)] = self.element_dtype(0.0)


class FlashAttentionDSABackwardSm100H128D576TwoCTADeterministic(TwoCTADeterministicReduction, FlashAttentionDSABackwardSm100H128D576TwoCTA):
    """D576 two-CTA arithmetic with fixed query-slot and rank reduction order."""

    # Each per-KV-row record holds 512 core values and two rank-private 64-value tails.
    # Compact records and fold buffers fit inside the 96-plane workspace capacity.
    COMPACT_SLOTS = 74
    COMPACT_RECORD = 640
    COMPACT_QUERIES_PER_GROUP = 8
    COMPACT_GROUPS = 10
    COMPACT_PUBLIC_PLANES = 96
    num_dkv_shards = COMPACT_SLOTS
    dkv_shards_per_query = 1
    max_query_pairs_per_wave = COMPACT_SLOTS

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
        single_query: bool = False,
        num_heads: int = 128,
        compact_slots: int = 74,
    ):
        """Specialize the deterministic wave width to the active GPU's two-SM cluster count."""
        super().__init__(element_dtype, head_dim, head_dim_v, block_tile, max_topk, single_query, num_heads)
        if not 1 <= compact_slots <= 76:
            raise ValueError(f"deterministic D576 two-CTA compact slots must be in [1, 76], got {compact_slots}")
        self.COMPACT_SLOTS = compact_slots
        self.num_dkv_shards = compact_slots
        self.max_query_pairs_per_wave = compact_slots
        compact_values = compact_slots * self.COMPACT_RECORD + (self.COMPACT_GROUPS + 1) * self.head_dim
        assert compact_values <= self.COMPACT_PUBLIC_PLANES * self.head_dim

    @cute.jit
    def make_dKV_accumulator(self, workspace_dKV: cute.Tensor, total_seqlen_KV: Int32):
        """Expose the compact 96-plane view used by records and fold buffers."""
        pitch = cute.round_up(total_seqlen_KV, 8)
        return cute.make_tensor(
            cute.recast_ptr(workspace_dKV.iterator, dtype=Float32),
            cute.make_layout((self.head_dim, pitch * self.COMPACT_PUBLIC_PLANES, (1, 1)), stride=(1, cutlass.Int64(self.head_dim), (0, 0))),
        )

    @cute.jit
    def select_dKV_accumulator(self, mdKV_acc: cute.Tensor, total_seqlen_KV: Int32, token_idx: Int32, rank: Int32 = 0):
        """Select one D640 query slot shared by the two disjoint core ranks."""
        slot = token_idx % Int32(self.COMPACT_SLOTS)
        pitch = cute.round_up(total_seqlen_KV, 8)
        offset = cutlass.Int64(slot) * cutlass.Int64(pitch) * cutlass.Int64(self.COMPACT_RECORD)
        return cute.make_tensor(
            mdKV_acc.iterator + offset, cute.make_layout((self.head_dim, total_seqlen_KV, (1, 1)), stride=(1, cutlass.Int64(self.COMPACT_RECORD), (0, 0)))
        )

    @cute.jit
    def _zero_compact_record_first_wave(self, mdKV_acc: cute.Tensor, seqlen: Int32, token_begin: Int32, tidx: Int32, rank: Int32):
        """Clear this rank's 256 core and 64 tail values before pipeline initialization."""
        if token_begin == Int32(0):
            record = cute.make_tensor(mdKV_acc.iterator, cute.make_layout((self.COMPACT_RECORD, seqlen), stride=(1, self.COMPACT_RECORD)))
            warp = tidx // Int32(32)
            lane = tidx % Int32(32)
            if warp < Int32(16):
                row = warp // Int32(2)
                chunk = (warp % Int32(2)) * Int32(32) + lane
                panel = chunk // Int32(32)
                dim = (rank + Int32(2) * panel) * Int32(128) + (chunk % Int32(32)) * Int32(4)
                while row < seqlen:
                    destination = record.iterator + cute.crd2idx((dim, row), record.layout)
                    _store_zero_f32x4(destination)
                    row += Int32(8)
            else:
                row = (warp - Int32(16)) * Int32(2) + lane // Int32(16)
                dim = Int32(self.D_HEAD) + rank * Int32(self.D_TAIL) + (lane % Int32(16)) * Int32(4)
                while row < seqlen:
                    destination = record.iterator + cute.crd2idx((dim, row), record.layout)
                    _store_zero_f32x4(destination)
                    row += Int32(8)

    @cute.kernel
    def zero_dKV_shards(self, mdKV_acc: cute.Tensor, seqlen: Int32):
        """Initialize every used D640 accumulator slot independently of prior workspace bytes."""
        block_row, slot, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        pitch = cute.round_up(seqlen, 8)
        slot_offset = cutlass.Int64(slot) * cutlass.Int64(pitch) * cutlass.Int64(self.COMPACT_RECORD)
        slot_tensor = cute.make_tensor(mdKV_acc.iterator + slot_offset, cute.make_layout((self.COMPACT_RECORD, pitch), stride=(1, self.COMPACT_RECORD)))
        if cutlass.const_expr(self.max_topk == 2048):
            dim = tidx % Int32(80) * Int32(8)
            row_lane = tidx // Int32(80)
            for row_offset in cutlass.range_constexpr(2):
                row = block_row * 16 + row_offset * Int32(8) + row_lane
                if row < seqlen:
                    destination = slot_tensor.iterator + cute.crd2idx((dim, row), slot_tensor.layout)
                    _store_zero_f32x8(destination)
        else:
            for row_offset in cutlass.range_constexpr(4):
                row = block_row * 4 + row_offset
                if row < seqlen:
                    for record_part in cutlass.range_constexpr(5):
                        dim = tidx + Int32(record_part * 128)
                        slot_tensor[dim, row] = Float32(0.0)

    @cute.kernel
    def fold_dKV_shards(self, mdKV_acc: cute.Tensor, seqlen: Int32, active_shards: Int32):
        """Fold eight accumulator slots four adjacent FP32 dimensions at a time."""
        row, group, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        pitch = cute.round_up(seqlen, 8)
        slot_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.COMPACT_RECORD)
        group_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.head_dim)
        group_region = cutlass.Int64(self.COMPACT_SLOTS) * slot_pitch
        group_offset = group_region + cutlass.Int64(group) * group_pitch
        group_tensor = cute.make_tensor(mdKV_acc.iterator + group_offset, cute.make_layout((self.head_dim, pitch), stride=(1, self.head_dim)))
        group_begin = group * Int32(self.COMPACT_QUERIES_PER_GROUP)
        if row < seqlen:
            if tidx < Int32(128):
                dim = tidx * Int32(4)
                acc_0 = Float32(0.0)
                acc_1 = Float32(0.0)
                acc_2 = Float32(0.0)
                acc_3 = Float32(0.0)
                for local_query in cutlass.range_constexpr(self.COMPACT_QUERIES_PER_GROUP):
                    slot = group_begin + Int32(local_query)
                    if slot < active_shards:
                        slot_tensor = cute.make_tensor(
                            mdKV_acc.iterator + cutlass.Int64(slot) * slot_pitch,
                            cute.make_layout((self.COMPACT_RECORD, pitch), stride=(1, self.COMPACT_RECORD)),
                        )
                        source = slot_tensor.iterator + cute.crd2idx((dim, row), slot_tensor.layout)
                        values = _load_f32x4(source)
                        acc_0 += values[0]
                        acc_1 += values[1]
                        acc_2 += values[2]
                        acc_3 += values[3]
                destination = group_tensor.iterator + cute.crd2idx((dim, row), group_tensor.layout)
                _store_f32x4(destination, acc_0, acc_1, acc_2, acc_3)
            if tidx >= Int32(128) and tidx < Int32(144):
                tail_dim = (tidx - Int32(128)) * Int32(4)
                acc_0 = Float32(0.0)
                acc_1 = Float32(0.0)
                acc_2 = Float32(0.0)
                acc_3 = Float32(0.0)
                for local_query in cutlass.range_constexpr(self.COMPACT_QUERIES_PER_GROUP):
                    slot = group_begin + Int32(local_query)
                    if slot < active_shards:
                        slot_tensor = cute.make_tensor(
                            mdKV_acc.iterator + cutlass.Int64(slot) * slot_pitch,
                            cute.make_layout((self.COMPACT_RECORD, pitch), stride=(1, self.COMPACT_RECORD)),
                        )
                        rank_0 = slot_tensor.iterator + cute.crd2idx((Int32(self.D_HEAD) + tail_dim, row), slot_tensor.layout)
                        rank_1 = slot_tensor.iterator + cute.crd2idx((Int32(self.D_QK) + tail_dim, row), slot_tensor.layout)
                        values_0 = _load_f32x4(rank_0)
                        values_1 = _load_f32x4(rank_1)
                        acc_0 += values_0[0]
                        acc_0 += values_1[0]
                        acc_1 += values_0[1]
                        acc_1 += values_1[1]
                        acc_2 += values_0[2]
                        acc_2 += values_1[2]
                        acc_3 += values_0[3]
                        acc_3 += values_1[3]
                destination = group_tensor.iterator + cute.crd2idx((Int32(self.D_HEAD) + tail_dim, row), group_tensor.layout)
                _store_f32x4(destination, acc_0, acc_1, acc_2, acc_3)

    @cute.kernel
    def fold_dKV_groups(self, mdKV_acc: cute.Tensor, seqlen: Int32, active_groups: Int32):
        """Fold the compact group planes in a fixed order into one D576 plane."""
        row, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        pitch = cute.round_up(seqlen, 8)
        slot_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.COMPACT_RECORD)
        group_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.head_dim)
        group_region = cutlass.Int64(self.COMPACT_SLOTS) * slot_pitch
        final_region = group_region + cutlass.Int64(self.COMPACT_GROUPS) * group_pitch
        final_tensor = cute.make_tensor(mdKV_acc.iterator + final_region, cute.make_layout((self.head_dim, pitch), stride=(1, self.head_dim)))
        if row < seqlen:
            for dim in cutlass.range(tidx, self.head_dim, 256):
                acc = Float32(0.0)
                for group in cutlass.range(active_groups):
                    group_tensor = cute.make_tensor(
                        mdKV_acc.iterator + group_region + cutlass.Int64(group) * group_pitch,
                        cute.make_layout((self.head_dim, pitch), stride=(1, self.head_dim)),
                    )
                    acc += group_tensor[dim, row]
                final_tensor[dim, row] = acc

    @cute.kernel
    def fold_dKV_groups_to_output(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        d_sink: cute.Tensor,
        seqlen: Int32,
        total_q: Int32,
        active_groups: Int32,
    ):
        """Fold group planes; for multiple queries, 16 leading blocks reduce dSink with 32 fixed chains."""
        physical_block, _, _ = cute.arch.block_idx()
        row = physical_block
        if cutlass.const_expr(not self.SINGLE_QUERY):
            row -= Int32(16)
        tidx, _, _ = cute.arch.thread_idx()
        pitch = cute.round_up(seqlen, 8)
        slot_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.COMPACT_RECORD)
        group_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.head_dim)
        group_region = cutlass.Int64(self.COMPACT_SLOTS) * slot_pitch
        logical_d0 = tidx * Int32(2)
        source_d0 = logical_d0
        source_d1 = logical_d0 + Int32(1)
        if logical_d0 < Int32(self.D_HEAD):
            core_base = logical_d0 // Int32(32) * Int32(32)
            core_d = logical_d0 % Int32(32)
            source_d0 = core_base + Int32(16) * (core_d // Int32(4) % Int32(2)) + core_d // Int32(8) + Int32(4) * (core_d % Int32(4))
            source_d1 = source_d0 + Int32(4)
        if row >= Int32(0):
            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            for group in cutlass.range(active_groups):
                group_tensor = cute.make_tensor(
                    mdKV_acc.iterator + group_region + cutlass.Int64(group) * group_pitch, cute.make_layout((self.head_dim, pitch), stride=(1, self.head_dim))
                )
                acc0 += group_tensor[source_d0, row]
                acc1 += group_tensor[source_d1, row]
            packed = _dq_pack_bf16x2(acc0, acc1)
            destination = cute.make_tensor(
                cute.recast_ptr(mdKV.iterator + cute.crd2idx((logical_d0, row, (0, 0)), mdKV.layout), dtype=cutlass.Uint32), cute.make_layout((1,))
            )
            destination[0] = packed
        elif cutlass.const_expr(not self.SINGLE_QUERY):
            if tidx < Int32(256):
                head = physical_block * Int32(8) + tidx // Int32(32)
                chain = tidx % Int32(32)
                if head < Int32(self.num_heads):
                    sink_log2 = Float32(attn_sink[head, (0, 0)]) * Float32(math.log2(math.e))
                    acc = Float32(0.0)
                    q_idx = chain
                    while q_idx < total_q:
                        p = cute.math.exp2(sink_log2 + scaled_lse[head, (q_idx, 0)])
                        if sink_log2 == Float32(float("inf")):
                            p = Float32(1.0)
                        elif sink_log2 == Float32(float("-inf")):
                            p = Float32(0.0)
                        acc += p * sum_odo[head, (q_idx, 0)]
                        q_idx += 32
                    acc = cute.arch.warp_reduction_sum(acc, threads_in_group=32)
                    if chain == Int32(0):
                        d_sink[head, (0, 0)] = acc

    @cute.kernel
    def fold_dKV_groups_to_output_vectorized(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor, seqlen: Int32, active_groups: Int32):
        """Fold group planes four physical FP32 values at a time and write logical BF16 dKV."""
        row, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        pitch = cute.round_up(seqlen, 8)
        slot_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.COMPACT_RECORD)
        group_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.head_dim)
        group_region = cutlass.Int64(self.COMPACT_SLOTS) * slot_pitch
        if row < seqlen:
            if tidx < Int32(128):
                physical_0 = tidx * Int32(4)
                acc_0 = Float32(0.0)
                acc_1 = Float32(0.0)
                acc_2 = Float32(0.0)
                acc_3 = Float32(0.0)
                for group in cutlass.range(active_groups):
                    group_tensor = cute.make_tensor(
                        mdKV_acc.iterator + group_region + cutlass.Int64(group) * group_pitch,
                        cute.make_layout((self.head_dim, pitch), stride=(1, self.head_dim)),
                    )
                    source = group_tensor.iterator + cute.crd2idx((physical_0, row), group_tensor.layout)
                    values = _load_f32x4(source)
                    acc_0 += values[0]
                    acc_1 += values[1]
                    acc_2 += values[2]
                    acc_3 += values[3]
                physical_1 = physical_0 + Int32(1)
                physical_2 = physical_0 + Int32(2)
                physical_3 = physical_0 + Int32(3)
                core_base = physical_0 // Int32(32) * Int32(32)
                logical_0 = core_base + physical_0 % Int32(4) * Int32(8) + physical_0 % Int32(32) // Int32(16) * Int32(4) + physical_0 % Int32(16) // Int32(4)
                logical_1 = core_base + physical_1 % Int32(4) * Int32(8) + physical_1 % Int32(32) // Int32(16) * Int32(4) + physical_1 % Int32(16) // Int32(4)
                logical_2 = core_base + physical_2 % Int32(4) * Int32(8) + physical_2 % Int32(32) // Int32(16) * Int32(4) + physical_2 % Int32(16) // Int32(4)
                logical_3 = core_base + physical_3 % Int32(4) * Int32(8) + physical_3 % Int32(32) // Int32(16) * Int32(4) + physical_3 % Int32(16) // Int32(4)
                mdKV[logical_0, row, (0, 0)] = self.element_dtype(acc_0)
                mdKV[logical_1, row, (0, 0)] = self.element_dtype(acc_1)
                mdKV[logical_2, row, (0, 0)] = self.element_dtype(acc_2)
                mdKV[logical_3, row, (0, 0)] = self.element_dtype(acc_3)
            if tidx >= Int32(128) and tidx < Int32(144):
                logical_0 = Int32(self.D_HEAD) + (tidx - Int32(128)) * Int32(4)
                acc_0 = Float32(0.0)
                acc_1 = Float32(0.0)
                acc_2 = Float32(0.0)
                acc_3 = Float32(0.0)
                for group in cutlass.range(active_groups):
                    group_tensor = cute.make_tensor(
                        mdKV_acc.iterator + group_region + cutlass.Int64(group) * group_pitch,
                        cute.make_layout((self.head_dim, pitch), stride=(1, self.head_dim)),
                    )
                    source = group_tensor.iterator + cute.crd2idx((logical_0, row), group_tensor.layout)
                    values = _load_f32x4(source)
                    acc_0 += values[0]
                    acc_1 += values[1]
                    acc_2 += values[2]
                    acc_3 += values[3]
                mdKV[logical_0, row, (0, 0)] = self.element_dtype(acc_0)
                mdKV[logical_0 + Int32(1), row, (0, 0)] = self.element_dtype(acc_1)
                mdKV[logical_0 + Int32(2), row, (0, 0)] = self.element_dtype(acc_2)
                mdKV[logical_0 + Int32(3), row, (0, 0)] = self.element_dtype(acc_3)

    @cute.jit
    def fold_dKV_accumulator(self, mdKV_acc: cute.Tensor, seqlen: Int32, stream: cuda.CUstream, active_shards: Int32):
        """Launch both fixed-order compact folds and return the contiguous result."""
        active_groups = cute.ceil_div(active_shards, self.COMPACT_QUERIES_PER_GROUP)
        self.fold_dKV_shards(mdKV_acc, seqlen, active_shards).launch(grid=[seqlen, active_groups, 1], block=[160, 1, 1], stream=stream)
        self.fold_dKV_groups(mdKV_acc, seqlen, active_groups).launch(grid=[seqlen, 1, 1], block=[256, 1, 1], stream=stream)
        pitch = cute.round_up(seqlen, 8)
        slot_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.COMPACT_RECORD)
        group_pitch = cutlass.Int64(pitch) * cutlass.Int64(self.head_dim)
        final_region = cutlass.Int64(self.COMPACT_SLOTS) * slot_pitch + cutlass.Int64(self.COMPACT_GROUPS) * group_pitch
        return cute.make_tensor(
            mdKV_acc.iterator + final_region, cute.make_layout((self.head_dim, seqlen, (1, 1)), stride=(1, cutlass.Int64(self.head_dim), (0, 0)))
        )

    @cute.jit
    def fold_dKV_accumulator_to_output(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        d_sink: cute.Tensor,
        seqlen: Int32,
        total_q: Int32,
        stream: cuda.CUstream,
        active_shards: Int32,
    ):
        """Preserve the compact FP32 fold order and fuse the multi-query dSink tail."""
        active_groups = cute.ceil_div(active_shards, self.COMPACT_QUERIES_PER_GROUP)
        self.fold_dKV_shards(mdKV_acc, seqlen, active_shards).launch(grid=[seqlen, active_groups, 1], block=[160, 1, 1], stream=stream)
        self.fold_dKV_groups_to_output_vectorized(mdKV_acc, mdKV, seqlen, active_groups).launch(grid=[seqlen, 1, 1], block=[160, 1, 1], stream=stream)
        if cutlass.const_expr(not self.SINGLE_QUERY):
            self.fold_dKV_groups_to_output(mdKV_acc, mdKV, sum_odo, scaled_lse, attn_sink, d_sink, seqlen, total_q, active_groups).launch(
                grid=[16, 1, 1], block=[288, 1, 1], stream=stream
            )

    @cute.jit
    def _drain_dkv(
        self,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        mdKV_acc: cute.Tensor,
        index_row: cute.Tensor,
        tile_index: Int32,
        topk: Int32,
        batch_idx: Int32,
        rtx: Int32,
        rank: Int32,
        done_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
    ):
        """Skip duplicate-owner serialization only for a runtime-unique N64 tile."""
        seqlen_kv = cute.size(mdKV_acc, mode=[1])
        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
        t_dkv_core_0 = t_dkv_0[(None, None), 0, 0]
        t_dkv_core_1 = t_dkv_1[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)), self.acc_dtype)
        tiled_t2r_0 = tcgen05.make_tmem_copy(tmem_load_atom, t_dkv_core_0)
        thread_t2r_0 = tiled_t2r_0.get_slice(dp_idx)
        tiled_t2r_1 = tcgen05.make_tmem_copy(tmem_load_atom, t_dkv_core_1)
        thread_t2r_1 = tiled_t2r_1.get_slice(dp_idx)
        c_dkv = cute.make_identity_tensor((self.D_TILE_CTA, self.N_TILE))
        thread_coordinates = self.split_wg(thread_t2r_0.partition_D(c_dkv), 2, wg_idx)
        thread_source_0 = self.split_wg(thread_t2r_0.partition_S(t_dkv_core_0), 2, wg_idx)
        thread_source_1 = self.split_wg(thread_t2r_1.partition_S(t_dkv_core_1), 2, wg_idx)
        thread_values_0 = cute.make_rmem_tensor(thread_coordinates.shape, self.acc_dtype)
        thread_values_1 = thread_values_0
        tile_base = tile_index * Int32(self.N_TILE)
        r_topk = cute.make_rmem_tensor((8,), cutlass.Int32)
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            local_row = Int32(cute.get(thread_coordinates[coord_base], mode=[1]))
            global_row = tile_base + local_row
            if global_row < topk:
                r_topk[i] = index_row[global_row]
            else:
                r_topk[i] = Int32(-1)
        tile_rows_unique = _n64_rows_unique(index_row, tile_index, topk, seqlen_kv, dp_idx % Int32(32))
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_0, thread_source_0, thread_values_0)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        assert cute.size(thread_values_0) == self.N_TILE // 2
        sub_tile_idx_0 = rank
        sub_tile_idx_1 = Int32(2) + rank
        if tile_rows_unique:
            self._deposit_dkv_panel_ordered(thread_values_0, r_topk, mdKV_acc, sub_tile_idx_0, dp_idx, batch_idx)
        else:
            owner_lane_group = dp_idx % Int32(4)
            lane_group = Int32(0)
            for active_wg in cutlass.range(2):
                if wg_idx == Int32(active_wg):
                    for lane_group in cutlass.range(4):
                        if owner_lane_group == Int32(lane_group):
                            self._deposit_dkv_panel_ordered(thread_values_0, r_topk, mdKV_acc, sub_tile_idx_0, dp_idx, batch_idx)
                        cute.arch.sync_warp()
                self.dsink_reducer_barrier.arrive_and_wait()
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        if tile_rows_unique:
            self._deposit_dkv_panel_ordered(thread_values_1, r_topk, mdKV_acc, sub_tile_idx_1, dp_idx, batch_idx)
            self.dsink_reducer_barrier.arrive_and_wait()
        else:
            owner_lane_group = dp_idx % Int32(4)
            lane_group = Int32(0)
            for active_wg in cutlass.range(2):
                if wg_idx == Int32(active_wg):
                    for lane_group in cutlass.range(4):
                        if owner_lane_group == Int32(lane_group):
                            self._deposit_dkv_panel_ordered(thread_values_1, r_topk, mdKV_acc, sub_tile_idx_1, dp_idx, batch_idx)
                        cute.arch.sync_warp()
                self.dsink_reducer_barrier.arrive_and_wait()
        return (wait_state, release_state)

    @cute.jit
    def _accumulate_dk_tail_wmma(
        self,
        dkt_warp_mma: cute.TiledMma,
        s_ds_kv_h: cute.Tensor,
        s_qt_d_h: cute.Tensor,
        indices: cute.Tensor,
        dkv: cute.Tensor,
        query: Int32,
        batch: Int32,
        tile: Int32,
        topk: Int32,
        mtx: Int32,
        tail_ld_mbar: cute.Pointer,
    ) -> None:
        """Use one final role barrier when a runtime N64 tile has unique row IDs."""
        thr_mma = dkt_warp_mma.get_slice(mtx)
        tCsA = thr_mma.partition_A(s_ds_kv_h)
        tCrA = dkt_warp_mma.make_fragment_A(tCsA)
        atom_ld_a = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), self.element_dtype)
        atom_ld_b = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), self.element_dtype)
        tiled_copy_a = cute.make_tiled_copy_A(atom_ld_a, dkt_warp_mma)
        tiled_copy_b = cute.make_tiled_copy_B(atom_ld_b, dkt_warp_mma)
        thr_copy_a = tiled_copy_a.get_slice(mtx)
        thr_copy_b = tiled_copy_b.get_slice(mtx)
        tCsA_view = thr_copy_a.partition_S(s_ds_kv_h)
        tCrA_view = thr_copy_a.retile(tCrA)
        coords = thr_mma.partition_C(cute.make_identity_tensor((self.N_TILE, 32)))
        seqlen_kv = cute.size(dkv, mode=[1])
        row_lo = Int32(cute.get(coords[0], mode=[0]))
        row_hi = row_lo
        for probe in cutlass.range_constexpr(cute.size(coords)):
            cand = Int32(cute.get(coords[probe], mode=[0]))
            if cand != row_lo:
                row_hi = cand
        idx_lo = Int32(-1)
        idx_hi = Int32(-1)
        if tile * Int32(self.N_TILE) + row_lo < topk:
            idx_lo = indices[tile * Int32(self.N_TILE) + row_lo, (query, batch)]
        if tile * Int32(self.N_TILE) + row_hi < topk:
            idx_hi = indices[tile * Int32(self.N_TILE) + row_hi, (query, batch)]
        index_row = indices[None, (query, batch)]
        tile_rows_unique = _n64_rows_unique(index_row, tile, topk, seqlen_kv, mtx % Int32(32))
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tail_offset = Int32(self.D_HEAD) + rank * Int32(self.D_TAIL)
        tail_dkv = cute.make_tensor(
            dkv.iterator + tail_offset, cute.make_layout((self.D_TAIL, seqlen_kv, (1, 1)), stride=(1, cutlass.Int64(self.COMPACT_RECORD), (0, 0)))
        )
        cute.copy(tiled_copy_a, tCsA_view, tCrA_view)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(tail_ld_mbar)
        for n_half in cutlass.range_constexpr(2):
            sub_qt = cute.local_tile(s_qt_d_h, (32, self.H_TILE_CTA), (n_half, 0))
            tCsB = thr_mma.partition_B(sub_qt)
            tCrB = dkt_warp_mma.make_fragment_B(tCsB)
            tCsB_view = thr_copy_b.partition_S(sub_qt)
            tCrB_view = thr_copy_b.retile(tCrB)
            cute.copy(tiled_copy_b, tCsB_view, tCrB_view)
            acc_shape = dkt_warp_mma.partition_shape_C((self.N_TILE, 32))
            tCrC = dkt_warp_mma.make_fragment_C(acc_shape)
            tCrC.fill(0.0)
            cute.gemm(dkt_warp_mma, tCrC, tCrA, tCrB, tCrC)
            assert cute.size(tCrC) % 2 == 0
            d_half_base = Int32(n_half * 32)
            if tile_rows_unique:
                self._deposit_dk_tail_ordered(coords, tCrC, tail_dkv, idx_lo, idx_hi, row_lo, d_half_base, batch)
            else:
                owner_warp = mtx // Int32(32)
                owner_lane_group = mtx % Int32(32) // Int32(4)
                lane_group = Int32(0)
                for active_warp in cutlass.range(4):
                    if owner_warp == Int32(active_warp):
                        for lane_group in cutlass.range(8):
                            if owner_lane_group == Int32(lane_group):
                                self._deposit_dk_tail_ordered(coords, tCrC, tail_dkv, idx_lo, idx_hi, row_lo, d_half_base, batch)
                            cute.arch.sync_warp()
                    self.gather_barrier.arrive_and_wait()
        if tile_rows_unique:
            self.gather_barrier.arrive_and_wait()
