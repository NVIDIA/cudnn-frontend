# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-CTA SM100 specialization for BF16 H128 D512 DSA backward.

The public dispatcher selects this implementation only for its validated
shape envelope and retains the existing SM100 kernels for every other
configuration.
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

U64x4 = Tuple[cutlass.Uint64, cutlass.Uint64, cutlass.Uint64, cutlass.Uint64]
F32x16 = Tuple[
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
]


@dsl_user_op
def _map_smem_to_cluster_rank(
    smem_ptr: cute.Pointer,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
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
    source: cute.Pointer,
    destination: cute.Pointer,
    completion_barrier: cute.Pointer,
    copy_bytes: int | Int32,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one shared-to-cluster bulk copy to ``peer_rank``."""

    source_i32 = source.toint(loc=loc, ip=ip).ir_value()
    destination_i32 = _map_smem_to_cluster_rank(
        destination,
        peer_rank,
        loc=loc,
        ip=ip,
    ).ir_value()
    barrier_i32 = _map_smem_to_cluster_rank(
        completion_barrier,
        peer_rank,
        loc=loc,
        ip=ip,
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i32,
            source_i32,
            barrier_i32,
            Int32(copy_bytes).ir_value(loc=loc, ip=ip),
        ],
        ("cp.async.bulk.shared::cluster.shared::cta." "mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];"),
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _mbarrier_wait_acquire_cluster(
    barrier: cute.Pointer,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Block on one local mbarrier phase with a cluster-scope acquire."""

    barrier_i32 = barrier.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [barrier_i32, phase.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "CLUSTER_WAIT_LOOP:\n\t"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
            "p, [$0], $1, 10000000;\n\t"
            "@!p bra CLUSTER_WAIT_LOOP;\n\t"
            "}"
        ),
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _nanosleep_u32(ns: Int32, *, loc=None, ip=None) -> None:
    """Pace reducer atomic bursts with a warp nanosleep hint."""
    llvm.inline_asm(
        None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )


@dsl_user_op
def _dq_pack_bf16x2(
    lo: Float32,
    hi: Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Round two FP32 values into one BF16x2 word, low logical value first."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _dq_store_bf16x8_streaming(
    destination: cute.Pointer,
    packed01: cutlass.Uint32,
    packed23: cutlass.Uint32,
    packed45: cutlass.Uint32,
    packed67: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
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
    return tuple(cutlass.Uint64(llvm.extractvalue(T.i64(), value, [index], loc=loc, ip=ip)) for index in range(4))


def _extract_f32x16(value, *, loc=None, ip=None) -> F32x16:
    return tuple(Float32(llvm.extractvalue(T.f32(), value, [index], loc=loc, ip=ip)) for index in range(16))


@dsl_user_op
def _prefetch_o_row_l2(source: cute.Pointer, *, loc=None, ip=None) -> None:
    """Pull one immutable O row chunk into L2 ahead of the reducer sweep."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [source_i64],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _load_o_bf16x16(source: cute.Pointer, *, loc=None, ip=None) -> U64x4:
    """Policy-neutral O load; pure because O is immutable during this kernel."""
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
def _load_do_bf16x16(source: cute.Pointer, *, loc=None, ip=None) -> U64x4:
    """Policy-neutral dO load; pure because dO is immutable during this kernel."""
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
def _decode_bf16x16_to_f32(
    x0: cutlass.Uint64,
    x1: cutlass.Uint64,
    x2: cutlass.Uint64,
    x3: cutlass.Uint64,
    *,
    loc=None,
    ip=None,
) -> F32x16:
    """Decode one BF16x16 vector to the stable 16-FP32 dot interface."""
    result_type = llvm.StructType.get_literal([T.f32()] * 16)
    values = llvm.inline_asm(
        result_type,
        [
            cutlass.Uint64(x0).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(x1).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(x2).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(x3).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .b32 word<8>;
        .reg .b16 half<16>;
        mov.b64 {word0, word1}, $16;
        mov.b64 {word2, word3}, $17;
        mov.b64 {word4, word5}, $18;
        mov.b64 {word6, word7}, $19;
        mov.b32 {half0, half1}, word0;
        mov.b32 {half2, half3}, word1;
        mov.b32 {half4, half5}, word2;
        mov.b32 {half6, half7}, word3;
        mov.b32 {half8, half9}, word4;
        mov.b32 {half10, half11}, word5;
        mov.b32 {half12, half13}, word6;
        mov.b32 {half14, half15}, word7;
        cvt.f32.bf16 $0, half0;
        cvt.f32.bf16 $1, half1;
        cvt.f32.bf16 $2, half2;
        cvt.f32.bf16 $3, half3;
        cvt.f32.bf16 $4, half4;
        cvt.f32.bf16 $5, half5;
        cvt.f32.bf16 $6, half6;
        cvt.f32.bf16 $7, half7;
        cvt.f32.bf16 $8, half8;
        cvt.f32.bf16 $9, half9;
        cvt.f32.bf16 $10, half10;
        cvt.f32.bf16 $11, half11;
        cvt.f32.bf16 $12, half12;
        cvt.f32.bf16 $13, half13;
        cvt.f32.bf16 $14, half14;
        cvt.f32.bf16 $15, half15;
        }""",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l,l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _extract_f32x16(values, loc=loc, ip=ip)


@dsl_user_op
def _dot_f32x16(
    o0: Float32,
    o1: Float32,
    o2: Float32,
    o3: Float32,
    o4: Float32,
    o5: Float32,
    o6: Float32,
    o7: Float32,
    o8: Float32,
    o9: Float32,
    o10: Float32,
    o11: Float32,
    o12: Float32,
    o13: Float32,
    o14: Float32,
    o15: Float32,
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    d4: Float32,
    d5: Float32,
    d6: Float32,
    d7: Float32,
    d8: Float32,
    d9: Float32,
    d10: Float32,
    d11: Float32,
    d12: Float32,
    d13: Float32,
    d14: Float32,
    d15: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Evaluate one fixed 16-element FP32 FMA chain."""
    result = llvm.inline_asm(
        T.f32(),
        [
            Float32(o0).ir_value(loc=loc, ip=ip),
            Float32(o1).ir_value(loc=loc, ip=ip),
            Float32(o2).ir_value(loc=loc, ip=ip),
            Float32(o3).ir_value(loc=loc, ip=ip),
            Float32(o4).ir_value(loc=loc, ip=ip),
            Float32(o5).ir_value(loc=loc, ip=ip),
            Float32(o6).ir_value(loc=loc, ip=ip),
            Float32(o7).ir_value(loc=loc, ip=ip),
            Float32(o8).ir_value(loc=loc, ip=ip),
            Float32(o9).ir_value(loc=loc, ip=ip),
            Float32(o10).ir_value(loc=loc, ip=ip),
            Float32(o11).ir_value(loc=loc, ip=ip),
            Float32(o12).ir_value(loc=loc, ip=ip),
            Float32(o13).ir_value(loc=loc, ip=ip),
            Float32(o14).ir_value(loc=loc, ip=ip),
            Float32(o15).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
            Float32(d4).ir_value(loc=loc, ip=ip),
            Float32(d5).ir_value(loc=loc, ip=ip),
            Float32(d6).ir_value(loc=loc, ip=ip),
            Float32(d7).ir_value(loc=loc, ip=ip),
            Float32(d8).ir_value(loc=loc, ip=ip),
            Float32(d9).ir_value(loc=loc, ip=ip),
            Float32(d10).ir_value(loc=loc, ip=ip),
            Float32(d11).ir_value(loc=loc, ip=ip),
            Float32(d12).ir_value(loc=loc, ip=ip),
            Float32(d13).ir_value(loc=loc, ip=ip),
            Float32(d14).ir_value(loc=loc, ip=ip),
            Float32(d15).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .f32 acc;
        mov.f32 acc, 0f00000000;
        fma.rn.f32 acc, $1, $17, acc;
        fma.rn.f32 acc, $2, $18, acc;
        fma.rn.f32 acc, $3, $19, acc;
        fma.rn.f32 acc, $4, $20, acc;
        fma.rn.f32 acc, $5, $21, acc;
        fma.rn.f32 acc, $6, $22, acc;
        fma.rn.f32 acc, $7, $23, acc;
        fma.rn.f32 acc, $8, $24, acc;
        fma.rn.f32 acc, $9, $25, acc;
        fma.rn.f32 acc, $10, $26, acc;
        fma.rn.f32 acc, $11, $27, acc;
        fma.rn.f32 acc, $12, $28, acc;
        fma.rn.f32 acc, $13, $29, acc;
        fma.rn.f32 acc, $14, $30, acc;
        fma.rn.f32 acc, $15, $31, acc;
        fma.rn.f32 acc, $16, $32, acc;
        mov.f32 $0, acc;
        }""",
        "=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(result)


@dsl_user_op
def _load_do_shared_bf16x16_two128_ordered(
    first_8: cute.Pointer,
    second_8: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> U64x4:
    """Two separately mapped 128-bit reads from the SW128 stationary panel."""
    first_i32 = first_8.toint(loc=loc, ip=ip).ir_value()
    second_i32 = second_8.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i64()] * 4)
    value = llvm.inline_asm(
        result_type,
        [first_i32, second_i32],
        """{
        ld.shared.v2.u64 {$0, $1}, [$4];
        ld.shared.v2.u64 {$2, $3}, [$5];
        }""",
        "=l,=l,=l,=l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return _extract_u64x4(value, loc=loc, ip=ip)


@dsl_user_op
def _reduce_s8_dim_chunks_owned(
    p0: Float32,
    p1: Float32,
    p2: Float32,
    p3: Float32,
    p4: Float32,
    p5: Float32,
    p6: Float32,
    p7: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Transpose the old 32-lane reduction tree into eight 4-lane row groups."""
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
        """{
        .reg .u32 lane, bits, send_bits, recv_bits;
        .reg .pred pred;
        .reg .f32 p<8>, y<4>, z<2>, lhs, rhs, acc, recv;
        mov.f32 p0, $1;
        mov.f32 p1, $2;
        mov.f32 p2, $3;
        mov.f32 p3, $4;
        mov.f32 p4, $5;
        mov.f32 p5, $6;
        mov.f32 p6, $7;
        mov.f32 p7, $8;
        mov.u32 lane, %laneid;

        and.b32 bits, lane, 16;
        setp.ne.u32 pred, bits, 0;
        selp.f32 lhs, p4, p0, pred;
        selp.f32 rhs, p0, p4, pred;
        add.rn.f32 y0, lhs, rhs;
        selp.f32 lhs, p5, p1, pred;
        selp.f32 rhs, p1, p5, pred;
        add.rn.f32 y1, lhs, rhs;
        selp.f32 lhs, p6, p2, pred;
        selp.f32 rhs, p2, p6, pred;
        add.rn.f32 y2, lhs, rhs;
        selp.f32 lhs, p7, p3, pred;
        selp.f32 rhs, p3, p7, pred;
        add.rn.f32 y3, lhs, rhs;

        and.b32 bits, lane, 8;
        setp.ne.u32 pred, bits, 0;
        selp.f32 lhs, y2, y0, pred;
        selp.f32 rhs, y0, y2, pred;
        add.rn.f32 z0, lhs, rhs;
        selp.f32 lhs, y3, y1, pred;
        selp.f32 rhs, y1, y3, pred;
        add.rn.f32 z1, lhs, rhs;

        and.b32 bits, lane, 4;
        setp.ne.u32 pred, bits, 0;
        selp.f32 lhs, z1, z0, pred;
        selp.f32 rhs, z0, z1, pred;
        add.rn.f32 acc, lhs, rhs;

        mov.b32 send_bits, acc;
        shfl.sync.bfly.b32 recv_bits, send_bits, 2, 0x1f, 0xffffffff;
        mov.b32 recv, recv_bits;
        add.rn.f32 acc, acc, recv;
        mov.b32 send_bits, acc;
        shfl.sync.bfly.b32 recv_bits, send_bits, 1, 0x1f, 0xffffffff;
        mov.b32 recv, recv_bits;
        add.rn.f32 acc, acc, recv;
        mov.f32 $0, acc;
        }""",
        "=f,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(result)


@dsl_user_op
def _cp_async_g2s_b128_index(
    destination: cute.Pointer,
    source: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
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
def _load_f32x4(
    source: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
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
def _store_bf16x4_ordinary(
    destination: cute.Pointer,
    packed01: cutlass.Uint32,
    packed23: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store four adjacent BF16 outputs with one ordinary 64-bit store."""

    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i64,
            cutlass.Uint32(packed01).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(packed23).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionDSABackwardSm100H128TwoCTA:
    """Two-CTA specialization for BF16 H128 D512.

    One ``(2, 1, 1)`` cluster owns each query token.  The score, dP, dQ,
    dV, and dK tensor-core operations all use ``CtaGroup.TWO``. MMA
    accumulators, O*dO, the dKV accumulation workspace/atomics, and dSink use
    FP32. Softmax and dS arithmetic use FP32 before the established BF16
    tensor-core operand conversion.
    """

    arch = 100

    H_TILE_CLUSTER = 128
    H_TILE_CTA = 64
    N_TILE = 64
    N_TILE_CTA = 32
    D_HEAD = 512
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
    MAX_SMEM_BYTES = 232_448
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
    # Round A-operand granularity.  The ring always occupies exactly 32 KiB
    # (ROUND_BUF_ELEMENTS); ROUND_K_HEADS trades TMA/mbarrier round trips per
    # KV tile against ring depth (in-flight TMA generations):
    #   K=16 -> 32 gens/tile, depth 8   K=32 -> 16 gens/tile, depth 4
    #   K=64 ->  8 gens/tile, depth 2
    # Short rows never reach steady state, so they want the cheap issue side;
    # long rows want the deep ring.  Selected per max_topk in __init__.
    ROUND_BUF_ELEMENTS = 16384
    ROUND_K_HEADS = 64
    ROUND_TILER = (256, 64, 64)
    ROUND_STAGE_ELEMENTS = 8192
    ROUND_STAGE_BYTES = 16384
    PDS_BLOCK_ELEMENTS = 2048
    PDS_BLOCK_BYTES = 4096
    TMEM_S_OFFSET = 0
    TMEM_S1_OFFSET = 32
    TMEM_DP_OFFSET = 64
    TMEM_DP1_OFFSET = 96
    TMEM_DQ0_OFFSET = 128
    TMEM_DQ1_OFFSET = 256
    TMEM_DKV0_OFFSET = 384
    TMEM_DKV1_OFFSET = 448
    SCORE_DONE_STAGES = 2
    ROUND_GENS_PER_TILE = 8
    ROUND_STAGES = 2
    MMA_DONE_STAGES = 2
    REDUCE_PACE_NS = 0
    REDUCE_DEPHASE_NS = 0

    DQ_EPI_BATCH_CHUNKS = 4
    DQ_WIDE_STORE_VALUES = 8
    DQ_CONVERSION_PAIR_VALUES = 2

    DIRECT_DQ_LOAN_SYNC = False
    DEFER_KSCORE_TAIL_UNTIL_DQ0_STORED = True
    GATHER_SETMAXREG = 88
    UTILITY_SETMAXREG = 96
    MATH_SETMAXREG = 120
    REDUCER_SETMAXREG = 88
    # Epilogue-phase register hand-off.  The reducer warps are finished
    # once the last dKV drain retires, while the math warps still have the
    # whole 128-value dQ panel-1 TMEM->RMEM->global epilogue in front of
    # them.  Handing the reducers' registers to the math warp group at
    # that exact boundary widens the epilogue without inflating the
    # register budget the steady-state tile loop is compiled against.
    MATH_EPI_SETMAXREG = 200
    REDUCER_EPI_SETMAXREG = 24
    EARLY_DQ0_OVERLAP = True
    # With the compatible long-row K-dQ retile, each score-B generation is
    # already the rank-owned N32 half of both K-dQ D128 panels. Only the
    # peer-owned N32 half then needs to be gathered before dQ.
    KDQ_REFETCH_HALF = False
    DUAL_DQ_DRAIN = False
    W17_RUNTIME_TMA_LOOP = False
    STATIONARY_DO_FIRST = False

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        if element_dtype != BFloat16:
            raise ValueError(f"two-CTA DSA backward requires BF16, got {element_dtype}")
        if head_dim != 512 or head_dim_v != 512:
            raise ValueError("two-CTA DSA backward requires head_dim=head_dim_v=512")
        if block_tile != 64:
            raise ValueError(f"two-CTA DSA backward requires block_tile=64, got {block_tile}")
        if max_topk not in (128, 512, 1024, 2048):
            raise ValueError("two-CTA DSA backward requires max_topk in " f"{{128, 512, 1024, 2048}}, got {max_topk}")
        self.element_dtype = element_dtype
        self.acc_dtype = Float32
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.head_dim_main = head_dim
        self.same_hdim_kv = True
        self.block_tile = block_tile
        self.max_topk = max_topk
        # Preserve the existing layout and full gather on short rows; the
        # half-refetch layout pays off in the long steady-state pipeline.
        self.KDQ_REFETCH_HALF = max_topk >= 512
        # Two-tile / eight-tile rows spend most of the round ring's life in
        # fill, so the cheap K64 issue side wins; sixteen-tile and longer rows
        # are steady-state and want the deeper K32 ring instead.
        self.ROUND_K_HEADS = 64 if max_topk <= 512 else 32
        assert self.H_TILE_CLUSTER % self.ROUND_K_HEADS == 0
        self.ROUND_TILER = (self.D_TILE_CLUSTER, self.N_TILE, self.ROUND_K_HEADS)
        self.ROUND_STAGE_ELEMENTS = self.D_TILE_CTA * self.ROUND_K_HEADS
        self.ROUND_STAGE_BYTES = 2 * self.ROUND_STAGE_ELEMENTS
        self.ROUND_STAGES = self.ROUND_BUF_ELEMENTS // self.ROUND_STAGE_ELEMENTS
        self.ROUND_GENS_PER_TILE = 4 * (self.H_TILE_CLUSTER // self.ROUND_K_HEADS)
        assert self.ROUND_STAGES in (2, 4, 8)
        assert self.ROUND_GENS_PER_TILE % (2 * self.ROUND_STAGES) == 0
        # Two-tile TOPK128 rows spend proportionally more time in the reducer
        # statistics path.  Move one 2K-register slice from four math warps to
        # eight reducer warps; longer rows retain the tuned 120/88 split.
        if self.max_topk == 128:
            self.MATH_SETMAXREG = 104
            self.REDUCER_SETMAXREG = 96
        # Warp-role register split.  The CTA is register-file bound: 226 KiB
        # of shared memory forces one CTA per SM, so ptxas is pinned to 96
        # registers per thread and the 640-thread CTA has exactly 61440
        # registers to divide between the four roles -- every extra register
        # for one role is taken from another.  The utility warps (tcgen05 MMA
        # issue, TMA load, and relay) are the most register-sensitive role and
        # therefore retain the larger share.
        assert (
            self.GATHER_SETMAXREG * self.GATHER_THREADS
            + self.MATH_SETMAXREG * self.MATH_THREADS
            + self.REDUCER_SETMAXREG * self.REDUCE_THREADS
            + self.UTILITY_SETMAXREG * 4 * 32
        ) == 640 * 96
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.THREADS_PER_CTA,
        )
        self.cta_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.THREADS_PER_CTA,
        )
        self.math_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.MATH_THREADS,
        )
        self.stats_lse_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.MATH_THREADS + self.REDUCE_THREADS,
        )
        self.gather_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.GATHER_THREADS,
        )
        self.stats_odo_barrier = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=self.MATH_THREADS + self.REDUCE_THREADS,
        )
        self.dsink_reducer_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=self.REDUCE_THREADS,
        )

    def split_wg(self, t: cute.Tensor, num_warp_groups: int, wg_idx: int):
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
            cute.make_ptr(
                self.element_dtype,
                source_row_offset.llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((self.K_CHUNK,)),
        )
        source_chunks = cute.flat_divide(source_row, (8,))
        destination_row_tensor = destination_rows[
            destination_row,
            None,
        ]
        destination_chunks = cute.flat_divide(
            destination_row_tensor,
            (8,),
        )
        for tile in cutlass.range_constexpr(self.K_CHUNK // 64):
            chunk_index = tile * self.KV_GROUP_SIZE + index_in_group
            thread_source = thread_copy.partition_S(source_chunks[None, chunk_index])
            thread_destination = thread_copy.partition_D(destination_chunks[None, chunk_index])
            cute.copy(copy_atom, thread_source, thread_destination)

    @cute.jit
    def _zero_sparse_k_d128_row(
        self,
        destination_rows: cute.Tensor,
        destination_row: Int32,
        index_in_group: Int32,
    ):
        """Cooperatively zero one D128 sparse-row destination."""

        destination_row_tensor = destination_rows[
            destination_row,
            None,
        ]
        destination_chunks = cute.flat_divide(
            destination_row_tensor,
            (8,),
        )
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
        kv_index_2: Int32,
        kv_index_3: Int32,
    ) -> None:
        """Fill the missing half (or all) of both sparse K-dQ panels."""
        seqlen_kv = cute.size(mKV, mode=[0])
        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = thread_count // self.KV_GROUP_SIZE
        d_offset_0 = rank * Int32(self.D_TILE_CTA)
        d_offset_1 = Int32(self.D_TILE_CLUSTER) + rank * Int32(self.D_TILE_CTA)
        assert self.N_TILE % groups_total == 0
        assert self.N_TILE // groups_total == 4
        half_base = (Int32(1) - rank) * Int32(self.N_TILE_CTA * int(self.KDQ_REFETCH_HALF))
        rows = 2 if self.KDQ_REFETCH_HALF else 4
        kdq_local_n = [half_base + Int32(row_iteration * groups_total) + group_index for row_iteration in range(rows)]
        kdq_kv_index = [kv_index_0, kv_index_1, kv_index_2, kv_index_3]
        for row_iteration in cutlass.range_constexpr(rows):
            local_n = kdq_local_n[row_iteration]
            kv_index = kdq_kv_index[row_iteration]
            if kv_index >= Int32(0) and kv_index < seqlen_kv:
                self._copy_sparse_k_d128_row(mKV, kd_rows_0, local_n, kv_index, batch_idx, d_offset_0, index_in_group, copy_atom, thread_copy)
                self._copy_sparse_k_d128_row(mKV, kd_rows_1, local_n, kv_index, batch_idx, d_offset_1, index_in_group, copy_atom, thread_copy)
            else:
                self._zero_sparse_k_d128_row(kd_rows_0, local_n, index_in_group)
                self._zero_sparse_k_d128_row(kd_rows_1, local_n, index_in_group)

    @cute.jit
    def _issue_dq_rounds(
        self,
        dq_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        kd_fragment_a: cute.Tensor,
        kd_fragment_b: cute.Tensor,
        ds_fragment: cute.Tensor,
        accumulate: cutlass.Boolean,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Issue both dQ rounds from one score_kv loan generation.

        Both 16 KiB K_dQ panels live simultaneously in the two score_kv
        halves.  One wait covers both rounds; one release returns score_kv
        to the gather warps for the next tile's score-K generation.
        """
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
            else:
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_b, mode=[2])):
                    cute.gemm(mma, t_dq_1, kd_fragment_b[None, None, k_block, 0], ds_fragment[None, None, k_block, 0], t_dq_1)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.arch.fence_view_async_tmem_store()
        kscore_pipeline.consumer_release(kscore_consumer_state)
        kscore_consumer_state.advance()
        return kscore_consumer_state

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
        """Consume one full 128-head A sweep against its two 64-head B halves.

        The ring always holds ``ROUND_STAGES == H_TILE_CLUSTER //
        ROUND_K_HEADS`` generations, so the sweep index doubles as the ring
        slot and the k-block order is identical for every ROUND_K_HEADS.
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
        for chunk in cutlass.range_constexpr(self.ROUND_STAGES):
            head_base = chunk * self.ROUND_K_HEADS
            b_fragment = b_fragment_0 if head_base < self.H_TILE_CTA else b_fragment_1
            b_k_half = (head_base % self.H_TILE_CTA) // self.ROUND_K_HEADS
            accumulate = True if chunk > 0 else first_accumulate
            round_pipeline.consumer_wait(round_consumer_state)
            self._issue_dkv_pass(dkv_tiled_mma, t_dkv, round_slot_fragments[chunk], b_fragment, b_k_half, accumulate)
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
        """Issue one self-contained A stage against its original B half."""
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
    def _issue_grads(
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
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_acquire_state: pipeline.PipelineState,
        dkv_commit_state: pipeline.PipelineState,
    ):
        """Issue the serial dV, dQ, then dK gradient chain for one tile."""
        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_0,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            p_fragment_0,
            p_fragment_1,
            False,
            round_pipeline,
            round_consumer_state,
        )
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_1,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            p_fragment_0,
            p_fragment_1,
            False,
            round_pipeline,
            round_consumer_state,
        )
        pds_pipeline.consumer_wait(pds_consumer_state)
        _mbarrier_wait_acquire_cluster(ds_local_ready_mbar, relay_phase)
        kscore_consumer_state = self._issue_dq_rounds(
            dq_tiled_mma, t_dq_0, t_dq_1, dq_kd_fragment_a, dq_kd_fragment_b, dq_ds_fragment, dq_accumulate, kscore_pipeline, kscore_consumer_state
        )
        _mbarrier_wait_acquire_cluster(relay_mbars + 1, relay_phase)
        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_0,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            ds_fragment_0,
            ds_fragment_1,
            True,
            round_pipeline,
            round_consumer_state,
        )
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_1,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            ds_fragment_0,
            ds_fragment_1,
            True,
            round_pipeline,
            round_consumer_state,
        )
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        return (round_consumer_state, kscore_consumer_state, dkv_acquire_state, dkv_commit_state, pds_consumer_state)

    @cute.jit
    def _prefetch_tile_indices(
        self,
        mTopkIdxs: cute.Tensor,
        tile_indices: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        slot: Int32,
        tidx: Int32,
    ) -> None:
        """Sixteen threads copy one complete N64 index tile to one SMEM slot."""

        if tidx < Int32(16):
            lane_base = tidx * Int32(4)
            position = tile_index * Int32(self.N_TILE) + lane_base
            source = mTopkIdxs.iterator + mTopkIdxs.layout((position, (token_idx, batch_idx)))
            destination = tile_indices.iterator + tile_indices.layout((lane_base, slot))
            _cp_async_g2s_b128_index(destination, source)

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
    ) -> None:
        """Rendezvous-free kdq fill into the score_kv loan halves (kq).

        The K_dQ images live in score_kv under a kscore generation the
        CALLER has already acquired -- no load-warp barrier, no
        kdq_ready close.  Completion is the caller's cp.async drain +
        fence + kscore producer commit, the same protocol as
        _load_score_kv.
        """
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = self.GATHER_THREADS // self.KV_GROUP_SIZE
        assert self.N_TILE // groups_total == 4
        half_base = (Int32(1) - rank) * Int32(self.N_TILE_CTA * int(self.KDQ_REFETCH_HALF))
        rows = 2 if self.KDQ_REFETCH_HALF else 4
        kdq_local_n = [half_base + Int32(row_iteration * groups_total) + group_index for row_iteration in range(rows)]
        kdq_kv_index = []
        for local_n in kdq_local_n:
            global_n = tile_index * Int32(self.N_TILE) + local_n
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = tile_indices[local_n]
            kdq_kv_index.append(kv_index)
        self._fill_kdq_pair(
            mKV,
            kd_rows_0,
            kd_rows_1,
            batch_idx,
            rank,
            role_tidx,
            self.GATHER_THREADS,
            copy_atom,
            thread_copy,
            kdq_kv_index[0],
            kdq_kv_index[1],
            kdq_kv_index[-2],
            kdq_kv_index[-1],
        )

    @cute.jit
    def _compute_folded_lse(
        self,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        softmax_stats: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        reducer_tidx: Int32,
    ) -> None:
        """Publish exact sink-folded negative LSE to shared memory."""

        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        log2_e = Float32(math.log2(math.e))

        if lane < Int32(8):
            row = row_base + lane
            head = rank * Int32(self.H_TILE_CTA) + row
            lse_value = Float32(mLSE[head, (token_idx, batch_idx)])
            sink_value = Float32(mAttnSink[head, (0, batch_idx)])
            lse_log2 = lse_value * log2_e
            sink_log2 = sink_value * log2_e
            maximum = cute.arch.fmax(lse_log2, sink_log2)
            denominator = Float32(cute.math.exp2(lse_log2 - maximum) + cute.math.exp2(sink_log2 - maximum))
            neg_lse_log2 = -(maximum + cute.math.log2(denominator))
            if lse_value == Float32(float("inf")):
                neg_lse_log2 = Float32(float("-inf"))
            softmax_stats[row, 0] = neg_lse_log2

    @staticmethod
    def _get_workspace_size_LSE_OdO(q, d, h, b, acc_dtype):
        q = (q + 7) // 8 * 8
        # One FP32 plane each for -O*dO and the sink-folded negative LSE.
        return (b, h, q, 2 * (acc_dtype.width // 8))

    @staticmethod
    def _get_workspace_size_dKV(k: int, d: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        d = (d + 7) // 8 * 8
        k = (k + 7) // 8 * 8
        return (b, 1, k, d * (acc_dtype.width // 8))

    def _get_stats_workspace(self, workspace: cute.Tensor, total_q: Int32, num_heads: Int32):
        total_q = cute.round_up(total_q, 8)
        acc_bytes = self.acc_dtype.width // 8
        plane_bytes = cute.assume(num_heads * total_q * acc_bytes, divby=acc_bytes * 64)
        sum_odo_iter = cute.recast_ptr(workspace.iterator, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(workspace.iterator + plane_bytes, dtype=self.acc_dtype)
        layout = cute.make_layout(
            (num_heads, (total_q, 1)),
            stride=(1, (cute.assume(num_heads, divby=64), 0)),
        )
        return (
            cute.make_tensor(sum_odo_iter, layout),
            cute.make_tensor(scaled_lse_iter, layout),
        )

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
        stream: cuda.CUstream,
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
        mdQ_epi = cute.make_tensor(
            mdQ.iterator, cute.make_layout((self.H_TILE_CLUSTER, self.D_HEAD, mdQ.shape[2]), stride=(mdQ.stride[1], mdQ.stride[0], mdQ.stride[2]))
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
        mQT = cute.make_tensor(
            mQ.iterator, cute.make_layout((self.D_HEAD, self.H_TILE_CLUSTER, mQ.shape[2]), stride=(mQ.stride[1], mQ.stride[0], mQ.stride[2]))
        )
        mdOT = cute.make_tensor(
            mdO.iterator, cute.make_layout((self.D_HEAD, self.H_TILE_CLUSTER, mdO.shape[2]), stride=(mdO.stride[1], mdO.stride[0], mdO.stride[2]))
        )
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
        dq_epi_tile = (self.H_TILE_CLUSTER, self.D_TILE_CTA)
        dq_epi_layout_staged = sm100_utils.make_smem_layout_epi(self.element_dtype, utils.LayoutEnum.from_tensor(mdQ_epi), dq_epi_tile, 1)
        dq_epi_layout = cute.select(dq_epi_layout_staged, mode=[0, 1])
        dq_epi_bytes = cute.size_in_bytes(self.element_dtype, dq_epi_layout_staged)
        assert dq_epi_bytes <= 32 * 1024
        tma_atom_dq_epi, tma_tensor_dq_epi = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mdQ_epi, dq_epi_layout, dq_epi_tile)
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
            score_a_layout_staged, score_b_layout_staged, dkv_a_layout_staged, dkv_b_layout_staged, dq_a_layout_staged, dq_b_layout_staged
        )
        self.shared_storage = SharedStorage
        self.shared_storage_bytes = SharedStorage.size_in_bytes()
        assert self.shared_storage_bytes <= self.MAX_SMEM_BYTES
        score_tmem_load = self._make_score_tmem_load()
        dq_cta_shape = (self.D_TILE_CTA, self.H_TILE_CLUSTER, self.N_TILE)
        dq_epi_tile = sm100_utils.compute_epilogue_tile_shape(dq_cta_shape, True, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype)
        dq_tmem_load = sm100_utils.get_tmem_load_op(dq_cta_shape, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype, dq_epi_tile, True)
        sum_odo, scaled_lse = self._get_stats_workspace(
            workspace_LSE_OdO,
            mQ.shape[2][0],
            cute.size(problem_shape[3][0]),
        )
        mdKV_acc = cute.make_tensor(
            cute.recast_ptr(workspace_dKV.iterator, dtype=self.acc_dtype),
            mdKV.layout,
        )
        zero_grid_x = (mKV.shape[0] + self.ZERO_ROWS_PER_BLOCK - 1) // self.ZERO_ROWS_PER_BLOCK
        self.zero_init(mdKV_acc, mdSink, mKV.shape[0], cute.size(problem_shape[3][0])).launch(
            grid=[zero_grid_x, 1, problem_shape[3][1]],
            block=[self.ZERO_THREADS, 1, 1],
            stream=stream,
        )
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
            mdQ,
            mdKV_acc,
            mTopkIdxs,
            mTopkLength,
            mLSE,
            mAttnSink,
            sum_odo,
            scaled_lse,
            mOut,
            mdO,
            Float32(softmax_scale),
            score_tiled_mma,
            dp_tiled_mma,
            dkv_tiled_mma,
            dq_tiled_mma,
            score_a_layout_staged,
            score_b_layout_staged,
            round_a_layout_staged,
            dkv_b_layout_staged,
            dq_a_layout_staged,
            dq_b_layout_staged,
            cluster_layout_vmnk,
            score_tmem_load,
            dq_tmem_load,
            tma_atom_dq_epi,
            tma_tensor_dq_epi,
            dq_epi_layout_staged,
            score_a_stage_bytes,
            round_stage_bytes,
            stationary_tiled_mma,
            stationary_a_layout_staged,
        ).launch(
            grid=(2 * problem_shape[0], 1, problem_shape[3][1]),
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
        self.convert_dkv(mdKV_acc, mdKV, mKV.shape[0]).launch(
            grid=[convert_grid_x, 1, 1], block=[self.num_threads_D_convert, self.num_threads_seq, 1], stream=stream
        )
        self.sum_dSink(sum_odo, scaled_lse, mAttnSink, mdSink, problem_shape).launch(
            grid=(cute.ceil_div(problem_shape[0], self.DSINK_BLOCK_Q), 1, problem_shape[3][1]),
            block=[self.DSINK_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _store_dq_epi_scalar_direct(
        self,
        t_dq: cute.Tensor,
        dq_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        s_dq_epi: cute.Tensor,
        tma_atom_dq_epi: cute.CopyAtom,
        mdQ_direct: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        mtx: Int32,
        batch_chunks: cutlass.Constexpr[int] = 1,
    ):
        """Store one rank-owned dQ round through the writable mdQ GMEM view.

        ``batch_chunks`` TMEM->RMEM chunks are issued back-to-back before the
        single ``tcgen05.wait::ld`` that retires them, so the epilogue pays one
        TMEM read latency per batch instead of one per chunk.  The batch is a
        pure register/latency trade: only the warp group that has already been
        handed the retired reducers' registers can afford the wide setting.
        """

        if mtx < self.MATH_THREADS_PER_CTA:
            tiled_t2r = tcgen05.make_tmem_copy(dq_tmem_load, t_dq)
            thread_t2r = tiled_t2r.get_slice(mtx)
            thread_source = thread_t2r.partition_S(t_dq)
            thread_coordinates = thread_t2r.partition_D(rank_coordinates)
            thread_source_chunks = cute.group_modes(
                thread_source,
                1,
                cute.rank(thread_source),
            )
            thread_coordinate_chunks = cute.group_modes(
                thread_coordinates,
                1,
                cute.rank(thread_coordinates),
            )
            num_chunks = cute.size(thread_coordinate_chunks, mode=[1])
            assert num_chunks % batch_chunks == 0
            for batch_index in cutlass.range_constexpr(num_chunks // batch_chunks):
                batched_values = []
                for slot in cutlass.range_constexpr(batch_chunks):
                    slot_index = batch_index * batch_chunks + slot
                    slot_values = cute.make_rmem_tensor(
                        thread_coordinate_chunks[None, slot_index].shape,
                        self.acc_dtype,
                    )
                    cute.copy(tiled_t2r, thread_source_chunks[None, slot_index], slot_values)
                    batched_values.append(slot_values)
                cute.arch.fence_view_async_tmem_load()
                for slot in cutlass.range_constexpr(batch_chunks):
                    chunk_index = batch_index * batch_chunks + slot
                    chunk_coordinates = thread_coordinate_chunks[None, chunk_index]
                    chunk_values = batched_values[slot]
                    converted_values = cute.make_rmem_tensor(
                        chunk_values.shape,
                        self.element_dtype,
                    )
                    packed_values = cute.recast_tensor(converted_values, cutlass.Uint32)
                    for pair_index in cutlass.range_constexpr(cute.size(packed_values)):
                        pair_base = pair_index * self.DQ_CONVERSION_PAIR_VALUES
                        packed_values[pair_index] = _dq_pack_bf16x2(
                            chunk_values[pair_base],
                            chunk_values[pair_base + 1],
                        )
                    for octet_index in cutlass.range_constexpr(cute.size(chunk_values) // self.DQ_WIDE_STORE_VALUES):
                        value_index = octet_index * self.DQ_WIDE_STORE_VALUES
                        d_in_round = Int32(cute.get(chunk_coordinates[value_index], mode=[0]))
                        head = Int32(cute.get(chunk_coordinates[value_index], mode=[1]))
                        global_d = Int32(round_index * self.D_TILE_CLUSTER) + d_in_round
                        contiguous = (global_d & Int32(7)) == Int32(0)
                        for adjacent_index in cutlass.range_constexpr(self.DQ_WIDE_STORE_VALUES - 1):
                            adjacent_offset = adjacent_index + 1
                            adjacent_coordinate = chunk_coordinates[value_index + adjacent_offset]
                            adjacent_d = Int32(round_index * self.D_TILE_CLUSTER) + Int32(cute.get(adjacent_coordinate, mode=[0]))
                            contiguous = contiguous and Int32(cute.get(adjacent_coordinate, mode=[1])) == head and adjacent_d == global_d + adjacent_offset
                        if contiguous:
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
                                mdQ_direct[
                                    scalar_d,
                                    scalar_head,
                                    (token_idx, batch_idx),
                                ] = converted_values[scalar_index]

    @cute.jit
    def _store_dq_epi_scalar_direct_panel0(
        self,
        t_dq: cute.Tensor,
        dq_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        s_dq_epi: cute.Tensor,
        tma_atom_dq_epi: cute.CopyAtom,
        mdQ_direct: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        mtx: Int32,
    ):
        """Store one rank-owned dQ round through the writable mdQ GMEM view."""

        if mtx < self.MATH_THREADS_PER_CTA:
            tiled_t2r = tcgen05.make_tmem_copy(dq_tmem_load, t_dq)
            thread_t2r = tiled_t2r.get_slice(mtx)
            thread_source = thread_t2r.partition_S(t_dq)
            thread_coordinates = thread_t2r.partition_D(rank_coordinates)
            thread_values = cute.make_rmem_tensor(
                thread_coordinates.shape,
                self.acc_dtype,
            )
            cute.copy(tiled_t2r, thread_source, thread_values)
            cute.arch.fence_view_async_tmem_load()
            for value_index in cutlass.range_constexpr(cute.size(thread_values)):
                d_in_round = Int32(cute.get(thread_coordinates[value_index], mode=[0]))
                head = Int32(cute.get(thread_coordinates[value_index], mode=[1]))
                global_d = Int32(round_index * self.D_TILE_CLUSTER) + d_in_round
                mdQ_direct[
                    global_d,
                    head,
                    (token_idx, batch_idx),
                ] = self.element_dtype(thread_values[value_index])

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
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
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
        # Panel 1 reuses panel 0's fragment: every FP32x4 atomic above reads
        # its quad at issue, so the two 32-value arrays are never live at the
        # same time and the reducers keep 32 registers free.
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

        cute.copy(tiled_t2r_0, thread_source_0, thread_values_0)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        assert cute.size(thread_values_0) == self.N_TILE // 2
        reduce_cohort = rank * Int32(2) + wg_idx
        _nanosleep_u32(reduce_cohort * Int32(self.REDUCE_DEPHASE_NS))
        sub_tile_idx_0 = rank
        sub_tile_idx_1 = Int32(2) + rank
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
                cute.arch.atomic_add(
                    target_frg_0.iterator.llvm_ptr,
                    rdkv_frg_0.load(),
                )
            _nanosleep_u32(Int32(self.REDUCE_PACE_NS))

        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        _nanosleep_u32(reduce_cohort * Int32(self.REDUCE_DEPHASE_NS))
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
                cute.arch.atomic_add(
                    target_frg_1.iterator.llvm_ptr,
                    rdkv_frg_1.load(),
                )
        return (wait_state, release_state)

    @cute.jit
    def _load_o_bf16x16(self, source: cute.Pointer):
        """Load immutable O through the policy-neutral path."""

        return _load_o_bf16x16(source)

    @cute.jit
    def _dot_odo_f32x16(
        self,
        o0: Float32,
        o1: Float32,
        o2: Float32,
        o3: Float32,
        o4: Float32,
        o5: Float32,
        o6: Float32,
        o7: Float32,
        o8: Float32,
        o9: Float32,
        o10: Float32,
        o11: Float32,
        o12: Float32,
        o13: Float32,
        o14: Float32,
        o15: Float32,
        d0: Float32,
        d1: Float32,
        d2: Float32,
        d3: Float32,
        d4: Float32,
        d5: Float32,
        d6: Float32,
        d7: Float32,
        d8: Float32,
        d9: Float32,
        d10: Float32,
        d11: Float32,
        d12: Float32,
        d13: Float32,
        d14: Float32,
        d15: Float32,
    ) -> Float32:
        values = (
            o0,
            o1,
            o2,
            o3,
            o4,
            o5,
            o6,
            o7,
            o8,
            o9,
            o10,
            o11,
            o12,
            o13,
            o14,
            o15,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            d8,
            d9,
            d10,
            d11,
            d12,
            d13,
            d14,
            d15,
        )
        return _dot_f32x16(*values)

    @cute.jit
    def _dot_odo_bf16x16_bits(
        self,
        out_bits: U64x4,
        dout_bits: U64x4,
    ) -> Float32:
        """Decode one packed segment pair and retain the fixed FMA order."""
        out_values = _decode_bf16x16_to_f32(out_bits[0], out_bits[1], out_bits[2], out_bits[3])
        dout_values = _decode_bf16x16_to_f32(dout_bits[0], dout_bits[1], dout_bits[2], dout_bits[3])
        return self._dot_odo_f32x16(*out_values, *dout_values)

    @cute.jit
    def _issue_w17_round_half_runtime(
        self,
        round_tma_atom: cute.CopyAtom,
        round_tma_tensor: cute.Tensor,
        round_ring_base: cute.Pointer,
        round_a_layout_staged: cute.ComposedLayout,
        dkv_tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        block_coord_k: Int32,
        rank: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        loop_iter: Int32,
        micro_base: cutlass.Constexpr[int],
        pipe_round,
    ) -> None:
        """Issue one ordered eight-generation half of W17's round TMA ring.

        ``micro_base`` is compile-time 0 for dOT and 8 for QT.  The runtime
        loop is intentional: unlike the legacy 16-site constexpr expansion it
        gives the descriptor, source coordinate, dynamic slot base, and
        PipelineState one loop-body live range.  Generation order, slot,
        phase, transaction, and barrier selection are unchanged.
        """

        g_round = cute.local_tile(
            round_tma_tensor,
            cute.select(self.ROUND_TILER, mode=[0, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        rank_dkv_mma = dkv_tiled_mma.get_slice(rank)
        rank_g_round = rank_dkv_mma.partition_A(g_round)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        for local_gen in cutlass.range(Int32(0), Int32(self.ROUND_GENS_PER_TILE // 2)):
            micro_gen = Int32(micro_base) + local_gen
            round_slot = micro_gen % Int32(self.ROUND_STAGES)
            grad_round = local_gen // Int32(4)
            h_half = (local_gen // Int32(2)) & Int32(1)
            k_half = local_gen & Int32(1)
            source_h32 = Int32(2) * h_half + k_half
            round_acq = pipeline.PipelineState(
                self.ROUND_STAGES,
                loop_iter * Int32(self.ROUND_GENS_PER_TILE) + micro_gen,
                round_slot,
                Int32(1) ^ ((micro_gen // Int32(self.ROUND_STAGES)) & Int32(1)),
            )
            pipe_round.producer_acquire(round_acq)
            round_completion_mbar = pipe_round.producer_get_barrier(round_acq)

            # SharedStorage declares the four equally sized, equally aligned
            # round buffers consecutively.  Rebuild only the selected slot's
            # canonical swizzled tensor instead of carrying four partition
            # objects across the utility-role setmaxreg boundary.
            round_slot_ptr = cute.make_ptr(
                self.element_dtype,
                round_ring_base.toint() + round_slot * Int32(self.ROUND_STAGE_BYTES),
                round_ring_base.memspace,
                assumed_align=1024,
            )
            round_slot_tensor = cute.make_tensor(
                cute.recast_ptr(
                    round_slot_ptr,
                    round_a_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                round_a_layout_staged.outer,
            )
            t_round_smem, t_round_gmem = cpasync.tma_partition(
                round_tma_atom,
                block_coord_k,
                a_cta_layout,
                cute.group_modes(round_slot_tensor, 0, 3),
                cute.group_modes(rank_g_round, 0, 3),
            )
            cute.copy(
                round_tma_atom,
                t_round_gmem[None, grad_round, source_h32],
                t_round_smem[None, 0],
                tma_bar_ptr=round_completion_mbar,
            )

    @cute.kernel
    def convert_dkv(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
    ):
        """Vector-convert the complete call-local FP32 dKV accumulator."""

        assert self.same_hdim_kv
        assert self.head_dim_main == 512
        assert mdKV_acc.element_type == cutlass.Float32
        assert mdKV.element_type == cutlass.BFloat16
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy
        if seq_id < seqlen:
            acc_row = mdKV_acc[None, seq_id, (0, batch_idx)]
            out_row = mdKV[None, seq_id, (0, batch_idx)]
            source_base = tidx * Int32(16)
            source = acc_row.iterator + source_base
            values = (
                *_load_f32x4(source),
                *_load_f32x4(source + Int32(4)),
                *_load_f32x4(source + Int32(8)),
                *_load_f32x4(source + Int32(12)),
            )

            # The reducer scramble is s=4*a+b -> o=a+8*b inside each
            # 32-element chunk.  Each thread owns sixteen consecutive s.
            chunk = (source_base // Int32(32)) * Int32(32)
            a0 = (source_base % Int32(32)) // Int32(4)
            for b in cutlass.range_constexpr(4):
                output_base = chunk + a0 + b * Int32(8)
                packed01 = _dq_pack_bf16x2(values[b], values[b + 4])
                packed23 = _dq_pack_bf16x2(values[b + 8], values[b + 12])
                _store_bf16x4_ordinary(
                    out_row.iterator + output_base,
                    packed01,
                    packed23,
                )

    @cute.jit
    def _load_score_kv_indexed(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        tile_indices: cute.Tensor,
        destination: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Gather the rank-owned N32 x D512 score B with 128-bit copies."""

        seqlen_kv = cute.size(mKV, mode=[0])
        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        row_local_n = [row_iteration * self.KV_NUM_GROUPS + group_index for row_iteration in range(rows_per_group)]
        row_kv_index = []
        for local_n in row_local_n:
            logical_n = rank * self.N_TILE_CTA + local_n
            topk_slot = tile_index * self.N_TILE + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = tile_indices[logical_n]
            row_kv_index.append(kv_index)

        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_local_n[row_iteration]
            kv_index = row_kv_index[row_iteration]

            for chunk in cutlass.range_constexpr(self.K_CHUNKS):
                destination_rows = cute.composition(
                    destination[None, None, None, chunk],
                    cute.make_layout((self.N_TILE_CTA, self.K_CHUNK)),
                )
                if kv_index >= 0 and kv_index < seqlen_kv:
                    self._copy_sparse_k_d128_row(
                        mKV,
                        destination_rows,
                        local_n,
                        kv_index,
                        batch_idx,
                        Int32(chunk * self.K_CHUNK),
                        index_in_group,
                        copy_atom,
                        thread_copy,
                    )
                else:
                    self._zero_sparse_k_d128_row(
                        destination_rows,
                        local_n,
                        index_in_group,
                    )

    @cute.jit
    def _prefetch_odo_rows(
        self,
        mOut: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        reducer_tidx: Int32,
    ) -> None:
        """Warm every O chunk this reducer thread will read before the LSE fold."""
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        row = row_base + lane // Int32(4)
        group_lane = lane % Int32(4)
        head = rank * Int32(self.H_TILE_CTA) + row
        # Each O row is 512 BF16 = 1024 B = eight aligned 128-byte lines.
        # Four lanes own one row, so two requests per lane cover the row with
        # exactly one prefetch per line instead of four per line.
        for chunk in cutlass.range_constexpr(2):
            dim = (group_lane + Int32(4 * chunk)) * Int32(64)
            source = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
            _prefetch_o_row_l2(source)

    @cute.jit
    def _compute_global_odo(
        self,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        softmax_stats: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        reducer_tidx: Int32,
        scale_softmax: Float32,
    ) -> None:
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        dim = lane * Int32(16)
        out_bits = [None] * 4
        dout_bits = [None] * 4
        partials = [None] * 8
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot)
            head = rank * Int32(self.H_TILE_CTA) + row
            out_pointer = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
            out_bits[slot] = self._load_o_bf16x16(out_pointer)
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot)
            head = rank * Int32(self.H_TILE_CTA) + row
            dout_pointer = mdO.iterator + mdO.layout((head, dim, (token_idx, batch_idx)))
            dout_bits[slot] = _load_do_bf16x16(dout_pointer)
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot)
            head = rank * Int32(self.H_TILE_CTA) + row
            out_values = _decode_bf16x16_to_f32(out_bits[slot][0], out_bits[slot][1], out_bits[slot][2], out_bits[slot][3])
            dout_values = _decode_bf16x16_to_f32(dout_bits[slot][0], dout_bits[slot][1], dout_bits[slot][2], dout_bits[slot][3])
            partials[slot] = self._dot_odo_f32x16(
                out_values[0],
                out_values[1],
                out_values[2],
                out_values[3],
                out_values[4],
                out_values[5],
                out_values[6],
                out_values[7],
                out_values[8],
                out_values[9],
                out_values[10],
                out_values[11],
                out_values[12],
                out_values[13],
                out_values[14],
                out_values[15],
                dout_values[0],
                dout_values[1],
                dout_values[2],
                dout_values[3],
                dout_values[4],
                dout_values[5],
                dout_values[6],
                dout_values[7],
                dout_values[8],
                dout_values[9],
                dout_values[10],
                dout_values[11],
                dout_values[12],
                dout_values[13],
                dout_values[14],
                dout_values[15],
            )
            refill_row = row_base + Int32(slot + 4)
            refill_head = rank * Int32(self.H_TILE_CTA) + refill_row
            out_pointer = mOut.iterator + mOut.layout((refill_head, dim, (token_idx, batch_idx)))
            dout_pointer = mdO.iterator + mdO.layout((refill_head, dim, (token_idx, batch_idx)))
            out_bits[slot] = self._load_o_bf16x16(out_pointer)
            dout_bits[slot] = _load_do_bf16x16(dout_pointer)
        for slot in cutlass.range_constexpr(4):
            row = row_base + Int32(slot + 4)
            head = rank * Int32(self.H_TILE_CTA) + row
            out_values = _decode_bf16x16_to_f32(out_bits[slot][0], out_bits[slot][1], out_bits[slot][2], out_bits[slot][3])
            dout_values = _decode_bf16x16_to_f32(dout_bits[slot][0], dout_bits[slot][1], dout_bits[slot][2], dout_bits[slot][3])
            partials[slot + 4] = self._dot_odo_f32x16(
                out_values[0],
                out_values[1],
                out_values[2],
                out_values[3],
                out_values[4],
                out_values[5],
                out_values[6],
                out_values[7],
                out_values[8],
                out_values[9],
                out_values[10],
                out_values[11],
                out_values[12],
                out_values[13],
                out_values[14],
                out_values[15],
                dout_values[0],
                dout_values[1],
                dout_values[2],
                dout_values[3],
                dout_values[4],
                dout_values[5],
                dout_values[6],
                dout_values[7],
                dout_values[8],
                dout_values[9],
                dout_values[10],
                dout_values[11],
                dout_values[12],
                dout_values[13],
                dout_values[14],
                dout_values[15],
            )
        for row_offset in cutlass.range_constexpr(8):
            row = row_base + Int32(row_offset)
            head = rank * Int32(self.H_TILE_CTA) + row
            row_sum = cute.arch.warp_reduction_sum(partials[row_offset], threads_in_group=32)
            if lane == Int32(0):
                raw_neg_sum_odo = -row_sum
                softmax_stats[row, 1] = raw_neg_sum_odo

    @cute.jit
    def _load_stationary_do_bf16x16(
        self,
        stationary_do_physical: cute.Tensor,
        row: Int32,
        dim: Int32,
    ) -> U64x4:
        raw_offset = row * Int32(64) + dim % Int32(64) + dim // Int32(64) * Int32(self.H_TILE_CTA * 64)
        raw_offset_hi = row * Int32(64) + (dim + Int32(8)) % Int32(64) + (dim + Int32(8)) // Int32(64) * Int32(self.H_TILE_CTA * 64)
        physical_offset = raw_offset ^ ((raw_offset // Int32(64) % Int32(8)) * Int32(8))
        physical_offset_hi = raw_offset_hi ^ ((raw_offset_hi // Int32(64) % Int32(8)) * Int32(8))
        return _load_do_shared_bf16x16_two128_ordered(
            stationary_do_physical.iterator + physical_offset,
            stationary_do_physical.iterator + physical_offset_hi,
        )

    @cute.jit
    def _compute_stationary_odo(
        self,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        stationary_do_physical: cute.Tensor,
        stationary_do_rows: cute.Tensor,
        stationary_tma_mbars: cute.Pointer,
        softmax_stats: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        reducer_tidx: Int32,
        scale_softmax: Float32,
    ) -> None:
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        row_offset = lane // Int32(4)
        group_lane = lane % Int32(4)
        row = row_base + row_offset
        head = rank * Int32(self.H_TILE_CTA) + row
        partials = [None] * 8

        # Match the original four-load cover of the stationary dO TMA tail,
        # but consume each dO carrier immediately after the wait.  This keeps
        # four O carriers plus one dO carrier live instead of four plus four.
        out_prefetch = [None] * 4
        for chunk in cutlass.range_constexpr(4):
            logical_old_lane = group_lane + Int32(4 * chunk)
            dim = logical_old_lane * Int32(16)
            out_pointer = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
            out_prefetch[chunk] = self._load_o_bf16x16(out_pointer)
        cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
        for chunk in cutlass.range_constexpr(4):
            logical_old_lane = group_lane + Int32(4 * chunk)
            dim = logical_old_lane * Int32(16)
            dout_bits = self._load_stationary_do_bf16x16(stationary_do_physical, row, dim)
            partials[chunk] = self._dot_odo_bf16x16_bits(out_prefetch[chunk], dout_bits)

        # Each four-lane group owns one row.  Across the eight constexpr
        # iterations it covers the same 32 old logical lanes in groups of 4.
        for chunk in cutlass.range_constexpr(4, 8):
            logical_old_lane = group_lane + Int32(4 * chunk)
            dim = logical_old_lane * Int32(16)
            out_pointer = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
            out_bits = self._load_o_bf16x16(out_pointer)
            dout_bits = self._load_stationary_do_bf16x16(stationary_do_physical, row, dim)
            partials[chunk] = self._dot_odo_bf16x16_bits(out_bits, dout_bits)

        owned_row_sum = _reduce_s8_dim_chunks_owned(
            partials[0],
            partials[1],
            partials[2],
            partials[3],
            partials[4],
            partials[5],
            partials[6],
            partials[7],
        )
        if group_lane == Int32(0):
            raw_neg_sum_odo = -owned_row_sum
            softmax_stats[row, 1] = raw_neg_sum_odo

    @cute.jit
    def _publish_dsink_stats(
        self,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        softmax_stats: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        reducer_tidx: Int32,
    ) -> None:
        if reducer_tidx < Int32(self.H_TILE_CTA):
            row = reducer_tidx
            head = rank * Int32(self.H_TILE_CTA) + row
            sum_odo[head, (token_idx, batch_idx)] = softmax_stats[row, 1]
            scaled_lse[head, (token_idx, batch_idx)] = softmax_stats[row, 0]

    @cute.kernel
    def zero_init(
        self,
        mdKV_acc: cute.Tensor,
        mdSink: cute.Tensor,
        seqlen_kv: Int32,
        num_heads: Int32,
    ):
        """Establish the call-local FP32 zero state for the dKV workspace and dSink.

        One launch replaces the two host-side memsets.  Each block clears
        ``ZERO_ROWS_PER_BLOCK`` KV rows with fully coalesced 4-byte stores (a
        warp covers 128 contiguous bytes per instruction), and block 0 also
        clears the FP32 dSink accumulator that the reduction kernel atomically
        updates.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, batch_idx = cute.arch.block_idx()
        row_base = bidx * Int32(self.ZERO_ROWS_PER_BLOCK)
        for row_offset in cutlass.range_constexpr(self.ZERO_ROWS_PER_BLOCK):
            row = row_base + Int32(row_offset)
            if row < seqlen_kv:
                for chunk in cutlass.range_constexpr(self.D_HEAD // self.ZERO_THREADS):
                    mdKV_acc[chunk * Int32(self.ZERO_THREADS) + tidx, row, (0, batch_idx)] = Float32(0.0)
        if bidx == Int32(0):
            if tidx < num_heads:
                mdSink[tidx, (0, batch_idx)] = Float32(0.0)

    @cute.kernel
    def sum_dSink(
        self,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        d_sink: cute.Tensor,
        problem_shape,
    ):
        q_block_idx, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        head_idx = tidx
        q_end = min(
            problem_shape[0],
            (q_block_idx + 1) * self.DSINK_BLOCK_Q,
        )
        q_idx = q_block_idx * self.DSINK_BLOCK_Q
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
            acc_0 += p_0 * sum_odo[head_idx, (q_idx, batch_idx)]
            acc_1 += p_1 * sum_odo[head_idx, (q_idx + 1, batch_idx)]
            acc_2 += p_2 * sum_odo[head_idx, (q_idx + 2, batch_idx)]
            acc_3 += p_3 * sum_odo[head_idx, (q_idx + 3, batch_idx)]
            q_idx += self.DSINK_UNROLL
        while q_idx < q_end:
            p_tail = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx, batch_idx)])
            acc_0 += p_tail * sum_odo[head_idx, (q_idx, batch_idx)]
            q_idx += 1
        ptr = d_sink.iterator + cute.crd2idx((head_idx, (0, batch_idx)), d_sink.layout)
        cute.arch.atomic_add(
            ptr.llvm_ptr,
            (acc_0 + acc_1) + (acc_2 + acc_3),
        )

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
        mdQ: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        scale_softmax: Float32,
        score_tiled_mma: cute.TiledMma,
        dp_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        dq_tiled_mma: cute.TiledMma,
        score_a_layout_staged: cute.ComposedLayout,
        score_b_layout_staged: cute.ComposedLayout,
        round_a_layout_staged: cute.ComposedLayout,
        dkv_b_layout_staged: cute.ComposedLayout,
        dq_a_layout_staged: cute.ComposedLayout,
        dq_b_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        score_tmem_load: cute.CopyAtom,
        dq_tmem_load: cute.CopyAtom,
        tma_atom_dq_epi: cute.CopyAtom,
        tma_tensor_dq_epi: cute.Tensor,
        dq_epi_layout_staged: cute.ComposedLayout,
        score_a_stage_bytes: cutlass.Constexpr[int],
        round_stage_bytes: cutlass.Constexpr[int],
        stationary_tiled_mma: cute.TiledMma,
        stationary_a_layout_staged: cute.ComposedLayout,
    ):
        """Execute the FP32 five-GEMM two-CTA schedule."""
        assert not (self.EARLY_DQ0_OVERLAP and self.DUAL_DQ_DRAIN), "early and late DQ0 observers are mutually exclusive"
        physical_x, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(rank)
        peer_rank = Int32(1) - rank
        token_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        is_leader_cta = rank == Int32(0)
        if warp_idx == Int32(self.LOAD_WARP):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(round_tma_atom_qt)
            cpasync.prefetch_descriptor(round_tma_atom_dot)
        smem = utils.SmemAllocator()
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
        loan_epi_safe_mbar = storage.loan_epi_safe_mbar.ptr
        stationary_q_raw = storage.stationary_q.data_ptr()
        stationary_do_raw = storage.stationary_do.data_ptr()
        round_buf_raw = storage.round_buf.data_ptr()
        round_slot_raw = tuple(round_buf_raw + slot * self.ROUND_STAGE_ELEMENTS for slot in range(self.ROUND_STAGES))
        score_kv_raw = storage.score_kv.data_ptr()
        stationary_q = storage.stationary_q.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_do = storage.stationary_do.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_q_tma = storage.stationary_q.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        stationary_do_tma = storage.stationary_do.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        # Execution-neutral plumbing.  The raw view pins the current SW128
        # lowering; the canonical composition is the authoritative logical map.
        stationary_do_physical = cute.make_tensor(
            stationary_do_raw,
            cute.make_layout((self.H_TILE_CTA * self.D_HEAD,), stride=(1,)),
        )
        stationary_do_rows = cute.composition(
            stationary_do_tma[None, None, None, 0],
            cute.make_layout(
                (self.H_TILE_CTA, self.D_HEAD),
                stride=(self.D_HEAD, 1),
            ),
        )
        k_n = storage.score_kv.get_tensor(score_b_layout_staged.outer, swizzle=score_b_layout_staged.inner)
        s_dq_epi = cute.make_tensor(cute.recast_ptr(storage.score_kv.data_ptr(), dq_epi_layout_staged.inner, self.element_dtype), dq_epi_layout_staged.outer)[
            None, None, 0
        ]
        kdq_loan_ptr_0 = cute.make_ptr(self.element_dtype, score_kv_raw.toint(), score_kv_raw.memspace, assumed_align=1024)
        kdq_loan_ptr_1 = cute.make_ptr(self.element_dtype, score_kv_raw.toint() + Int32(16384), score_kv_raw.memspace, assumed_align=1024)
        # Retile each K-dQ panel as two stacked copies of the score-B stage:
        # panel(kv,d) = 4096*(kv//32) + scoreB_stage(kv%32,d).
        # This remains a canonical SMEM A descriptor within every K16 block
        # and makes the preceding rank-owned score-B bytes reusable verbatim.
        if cutlass.const_expr(self.KDQ_REFETCH_HALF):
            atom_mn = 64
            atom_k = 8
            m_tiles = self.D_TILE_CTA // atom_mn
            stage_elements = self.N_TILE_CTA * self.D_TILE_CTA
            n_tile_stride = atom_mn * atom_k
            d_tile_stride = n_tile_stride * (self.N_TILE_CTA // atom_k)
            kdq_outer = cute.make_layout(
                (((atom_mn, m_tiles), (atom_k, 2)), 1, (2, 2), 1),
                stride=(
                    ((1, d_tile_stride), (atom_mn, n_tile_stride)),
                    0,
                    (n_tile_stride * 2, stage_elements),
                    self.D_TILE_CTA * self.N_TILE,
                ),
            )
            assert cute.cosize(kdq_outer) == self.D_TILE_CTA * self.N_TILE
        else:
            kdq_outer = dq_a_layout_staged.outer
        kdq_loan = (
            cute.make_tensor(cute.recast_ptr(kdq_loan_ptr_0, dq_a_layout_staged.inner, dtype=self.element_dtype), kdq_outer),
            cute.make_tensor(cute.recast_ptr(kdq_loan_ptr_1, dq_a_layout_staged.inner, dtype=self.element_dtype), kdq_outer),
        )
        round_slots = tuple(
            cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        self.element_dtype,
                        round_buf_raw.toint() + slot * self.ROUND_STAGE_BYTES,
                        round_buf_raw.memspace,
                        assumed_align=1024,
                    ),
                    round_a_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                round_a_layout_staged.outer,
            )
            for slot in range(self.ROUND_STAGES)
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
        score_store_layout = sm100_utils.make_smem_layout_epi(self.element_dtype, utils.LayoutEnum.COL_MAJOR, (self.H_TILE_CTA, self.N_TILE), 1)
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
        if cutlass.const_expr(not self.W17_RUNTIME_TMA_LOOP):
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
        if cutlass.const_expr(not self.W17_RUNTIME_TMA_LOOP):
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
        dq_kd_fragment_a = dq_tiled_mma.make_fragment_A(kdq_loan[0])
        dq_kd_fragment_b = dq_tiled_mma.make_fragment_A(kdq_loan[1])
        dq_ds_fragment = dq_tiled_mma.make_fragment_B(ds_image)
        assert cute.cosize(round_a_layout_staged) == self.ROUND_STAGE_ELEMENTS
        round_fragments = tuple(dkv_tiled_mma.make_fragment_A(round_slots[slot]) for slot in range(self.ROUND_STAGES))
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
        # Start the supplied-length load before barrier initialization, the
        # cluster rendezvous, and two-CTA TMEM allocation.  Only the load is
        # hoisted; its clamp and every dependent use remain below.
        if cutlass.const_expr(mTopkLength is not None):
            raw_topk = mTopkLength[token_idx]
        else:
            raw_topk = Int32(mTopkIdxs.shape[0])
        atom_thr_size = cute.size(score_tiled_mma.thr_id.shape)
        leader_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        math_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.MATH_THREADS)
        gather_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.GATHER_THREADS)
        # The score-ready barrier receives one arrival from each gather warp in
        # both CTAs, rather than a redundant arrival from every gather thread.
        score_ready_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.GATHER_WARPS * 2)
        reduce_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.REDUCE_THREADS)
        pipe_s_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.SCORE_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.s_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_s_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_s_done.sync_object_full,
            sync_object_empty=pipe_s_done.sync_object_empty,
            num_stages=pipe_s_done.num_stages,
            producer_mask=pipe_s_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_s_done.cta_group,
        )
        pipe_dp_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.SCORE_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.dp_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dp_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_dp_done.sync_object_full,
            sync_object_empty=pipe_dp_done.sync_object_empty,
            num_stages=pipe_dp_done.num_stages,
            producer_mask=pipe_dp_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_dp_done.cta_group,
        )
        pipe_kscore = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=score_ready_group,
            consumer_group=leader_group,
            barrier_storage=storage.kscore_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_kscore = pipeline.PipelineAsyncUmma(
            sync_object_full=pipe_kscore.sync_object_full,
            sync_object_empty=pipe_kscore.sync_object_empty,
            num_stages=pipe_kscore.num_stages,
            producer_mask=Int32(0),
            consumer_mask=pipe_kscore.consumer_mask,
            cta_group=pipe_kscore.cta_group,
        )
        pipe_round = pipeline.PipelineTmaUmma.create(
            num_stages=self.ROUND_STAGES,
            producer_group=leader_group,
            consumer_group=leader_group,
            tx_count=round_stage_bytes * 2,
            barrier_storage=storage.round_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pds_commit_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size)
        pipe_pds = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pds_commit_group,
            consumer_group=leader_group,
            barrier_storage=storage.pds_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_pds = pipeline.PipelineAsyncUmma(
            sync_object_full=pipe_pds.sync_object_full,
            sync_object_empty=pipe_pds.sync_object_empty,
            num_stages=pipe_pds.num_stages,
            producer_mask=Int32(0),
            consumer_mask=pipe_pds.consumer_mask,
            cta_group=pipe_pds.cta_group,
        )
        pipe_dkv_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.MMA_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=reduce_group,
            barrier_storage=storage.dkv_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dkv_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_dkv_done.sync_object_full,
            sync_object_empty=pipe_dkv_done.sync_object_empty,
            num_stages=pipe_dkv_done.num_stages,
            producer_mask=pipe_dkv_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_dkv_done.cta_group,
        )
        pipe_dq_done = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.dq_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dq_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_dq_done.sync_object_full,
            sync_object_empty=pipe_dq_done.sync_object_empty,
            num_stages=pipe_dq_done.num_stages,
            producer_mask=pipe_dq_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_dq_done.cta_group,
        )
        if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
            # The enabled subclass adds this dedicated full/empty pair.  Its
            # full barrier is committed immediately after the final DQ0
            # UMMA, independently of the existing DQ1 completion pipeline.
            pipe_dq0_done = pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=leader_group,
                consumer_group=gather_group,
                barrier_storage=storage.dq0_done_mbars.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            pipe_dq0_done = pipeline.PipelineUmmaAsync(
                sync_object_full=pipe_dq0_done.sync_object_full,
                sync_object_empty=pipe_dq0_done.sync_object_empty,
                num_stages=pipe_dq0_done.num_stages,
                producer_mask=pipe_dq0_done.producer_mask,
                consumer_mask=Int32(0),
                cta_group=pipe_dq0_done.cta_group,
            )
        if tidx == Int32(0):
            cute.arch.mbarrier_init(stationary_tma_mbars, 1)
            cute.arch.mbarrier_init(stationary_tma_mbars + 1, 1)
            cute.arch.mbarrier_init(stationary_ready_mbar, 2)
            cute.arch.mbarrier_init(stationary_ready_mbar + 1, 2)
            cute.arch.mbarrier_init(landing_mbars, 1)
            cute.arch.mbarrier_init(landing_mbars + 1, 1)
            cute.arch.mbarrier_init(relay_mbars, 2)
            cute.arch.mbarrier_init(relay_mbars + 1, 2)
            cute.arch.mbarrier_init(pds_ready_mbars, self.MATH_WARPS)
            cute.arch.mbarrier_init(p_ready_mbars, self.MATH_WARPS)
            cute.arch.mbarrier_init(ds_local_ready_mbar, 2)
            if cutlass.const_expr(self.DIRECT_DQ_LOAN_SYNC):
                cute.arch.mbarrier_init(loan_epi_safe_mbar, 1)
        cute.arch.fence_view_async_shared()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(
            tmem_holding_buf_ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.MATH_WARP_BEGIN,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        score_c_layout = score_tiled_mma.make_fragment_C(score_tiled_mma.partition_shape_C((self.H_TILE_CLUSTER, self.N_TILE))).layout
        dkv_c_layout = dkv_tiled_mma.make_fragment_C(dkv_tiled_mma.partition_shape_C(self.DKV_MMA_TILER[:2])).layout
        dq_c_layout = dq_tiled_mma.make_fragment_C(dq_tiled_mma.partition_shape_C(self.DQ_MMA_TILER[:2])).layout
        t_score = cute.make_tensor(tmem_ptr + self.TMEM_S_OFFSET, score_c_layout)
        t_score_pp = cute.make_tensor(tmem_ptr + self.TMEM_S1_OFFSET, score_c_layout)
        t_dp = cute.make_tensor(tmem_ptr + self.TMEM_DP_OFFSET, score_c_layout)
        t_dp_pp = cute.make_tensor(tmem_ptr + self.TMEM_DP1_OFFSET, score_c_layout)
        t_dq = (cute.make_tensor(tmem_ptr + self.TMEM_DQ0_OFFSET, dq_c_layout), cute.make_tensor(tmem_ptr + self.TMEM_DQ1_OFFSET, dq_c_layout))
        t_dkv = (cute.make_tensor(tmem_ptr + self.TMEM_DKV0_OFFSET, dkv_c_layout), cute.make_tensor(tmem_ptr + self.TMEM_DKV1_OFFSET, dkv_c_layout))
        topk = raw_topk
        if topk > Int32(mTopkIdxs.shape[0]):
            topk = Int32(mTopkIdxs.shape[0])
        if topk < Int32(0):
            topk = Int32(0)
        tile_count = (topk + Int32(self.N_TILE - 1)) // Int32(self.N_TILE)
        tile_count = cute.arch.make_warp_uniform(tile_count)
        if warp_idx < Int32(self.MATH_WARP_BEGIN):
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
        else:
            if cutlass.const_expr(self.REDUCER_SETMAXREG > 96):
                cute.arch.setmaxregister_increase(self.REDUCER_SETMAXREG)
            elif cutlass.const_expr(self.REDUCER_SETMAXREG < 96):
                cute.arch.setmaxregister_decrease(self.REDUCER_SETMAXREG)
        if warp_idx < Int32(self.GATHER_WARPS):
            gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
                dq0_done_gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            gather_kd_rows_0 = self._kd_round_rows(kdq_loan[0])
            gather_kd_rows_1 = self._kd_round_rows(kdq_loan[1])
            if tile_count > Int32(0):
                self._prefetch_tile_indices(
                    mTopkIdxs,
                    staged_indices,
                    token_idx,
                    batch_idx,
                    tile_count - Int32(1),
                    Int32(0),
                    tidx,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                self.gather_barrier.arrive_and_wait()
                pipe_kscore.producer_acquire(gather_state)
                self._load_score_kv_indexed(
                    mKV,
                    mTopkIdxs,
                    staged_indices[None, Int32(0)],
                    k_n,
                    token_idx,
                    batch_idx,
                    tile_count - Int32(1),
                    topk,
                    rank,
                    tidx,
                    kv_copy_atom,
                    kv_thread_copy,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                if (tidx & Int32(31)) == Int32(0):
                    pipe_kscore.producer_commit(gather_state)
                gather_state.advance()
                for score_iter in cutlass.range(Int32(0), tile_count):
                    index_slot = score_iter & Int32(1)
                    next_score_iter = score_iter + Int32(1)
                    next_index_slot = next_score_iter & Int32(1)
                    has_next_index_tile = score_iter != tile_count - Int32(1)
                    if has_next_index_tile:
                        self._prefetch_tile_indices(
                            mTopkIdxs,
                            staged_indices,
                            token_idx,
                            batch_idx,
                            tile_count - Int32(1) - next_score_iter,
                            next_index_slot,
                            tidx,
                        )
                    pipe_kscore.producer_acquire(gather_state)
                    self._gather_kdq_indexed(
                        mKV,
                        mTopkIdxs,
                        staged_indices[None, index_slot],
                        gather_kd_rows_0,
                        gather_kd_rows_1,
                        token_idx,
                        batch_idx,
                        tile_count - Int32(1) - score_iter,
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    if (tidx & Int32(31)) == Int32(0):
                        pipe_kscore.producer_commit(gather_state)
                    gather_state.advance()
                    if score_iter != tile_count - Int32(1):
                        self.gather_barrier.arrive_and_wait()
                        next_iter = score_iter + Int32(1)
                        pipe_kscore.producer_acquire(gather_state)
                        self._load_score_kv_indexed(
                            mKV,
                            mTopkIdxs,
                            staged_indices[None, next_index_slot],
                            k_n,
                            token_idx,
                            batch_idx,
                            tile_count - Int32(1) - next_iter,
                            topk,
                            rank,
                            tidx,
                            kv_copy_atom,
                            kv_thread_copy,
                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        if (tidx & Int32(31)) == Int32(0):
                            pipe_kscore.producer_commit(gather_state)
                        gather_state.advance()
                if cutlass.const_expr(not (self.EARLY_DQ0_OVERLAP and self.DEFER_KSCORE_TAIL_UNTIL_DQ0_STORED)):
                    pipe_kscore.producer_tail(gather_state)
                if cutlass.const_expr(self.DIRECT_DQ_LOAN_SYNC):
                    self.gather_barrier.arrive_and_wait()
                    if warp_idx == Int32(0):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(loan_epi_safe_mbar)
                if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
                    pipe_dq0_done.consumer_wait(dq0_done_gather_state)
                    # Use the split epilogue: the now-idle
                    # gather warpgroup drains panel 0 with the existing
                    # BF16x8 streaming store while DQ1/dK continue.
                    self._store_dq_epi_scalar_direct(
                        t_dq[0],
                        dq_tmem_load,
                        rank_dq_coordinates,
                        s_dq_epi,
                        tma_atom_dq_epi,
                        mdQ,
                        0,
                        token_idx,
                        batch_idx,
                        rank,
                        tidx,
                    )
                    pipe_dq0_done.consumer_release(dq0_done_gather_state)
                    dq0_done_gather_state.advance()
                    if cutlass.const_expr(self.DEFER_KSCORE_TAIL_UNTIL_DQ0_STORED):
                        pipe_kscore.producer_tail(gather_state)
                if cutlass.const_expr(self.DUAL_DQ_DRAIN):
                    # Observe the existing final dQ completion without
                    # joining its empty-barrier release group.  Math threads
                    # remain the sole consumer/releaser; the final cluster
                    # rendezvous keeps TMEM alive until this disjoint DQ0
                    # store has also completed.
                    dq_done_gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    pipe_dq_done.consumer_wait(dq_done_gather_state)
                    self._store_dq_epi_scalar_direct(
                        t_dq[0],
                        dq_tmem_load,
                        rank_dq_coordinates,
                        s_dq_epi,
                        tma_atom_dq_epi,
                        mdQ,
                        0,
                        token_idx,
                        batch_idx,
                        rank,
                        tidx,
                    )
        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            mtx = tidx - Int32(self.MATH_THREAD_BEGIN)
            if tile_count > Int32(0):
                # stats_lse_barrier spans MATH+REDUCE, so it already aligns the
                # four math warps; math_barrier is only needed on the empty row.
                self.stats_lse_barrier.arrive_and_wait()
            else:
                self.math_barrier.arrive_and_wait()
            s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SCORE_DONE_STAGES)
            dp_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SCORE_DONE_STAGES)
            pds_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
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
            dp_copy_pp = tcgen05.make_tmem_copy(score_tmem_load, t_dp_pp)
            dp_source_pp = dp_copy_pp.get_slice(mtx).partition_S(t_dp_pp)
            smem_store_atom = sm100_utils.get_smem_store_op(utils.LayoutEnum.COL_MAJOR, self.element_dtype, self.acc_dtype, score_copy)
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
            # dS reuses P's BF16 staging array: the P stmatrix consumes its
            # registers at issue and is followed by a shared-memory fence and
            # a warp sync before the first dS element is formed, so the two
            # arrays are never live together.
            r_ds = r_p
            # Correctness-minimal state: filled once after the late publication
            # and reused by all later dS iterations.
            r_delta = cute.make_rmem_tensor((4,), self.acc_dtype)
            softmax_scale_log2_e = scale_softmax * Float32(math.log2(math.e))
            hoist_group_bases = [2 * (h_group % 2) + 16 * (h_group // 2) for h_group in range(4)]
            hoist_group_local_h = [Int32(cute.get(score_coordinates[group_base], mode=[0])) % Int32(self.H_TILE_CTA) for group_base in hoist_group_bases]
            hoist_band_indices = [[group_base + j % 2 + 4 * (j // 2) for j in range(8)] for group_base in hoist_group_bases]
            hoist_lse = [softmax_stats[hoist_group_local_h[h_group], 0] for h_group in range(4)]
            math_owns_local_half = cute.arch.make_warp_uniform(mtx // Int32(self.H_TILE_CTA)) == rank
            for loop_iter in cutlass.range(tile_count):
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
                        r_score[i0] = v0
                        r_score[i1] = v1
                        r_p[i0] = self.element_dtype(v0)
                        r_p[i1] = self.element_dtype(v1)
                pipe_pds.producer_acquire(pds_state)
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
                pipe_dp_done.consumer_wait(dp_state)
                if dp_state.index == Int32(0):
                    cute.copy(dp_copy, dp_source, r_dp)
                else:
                    cute.copy(dp_copy_pp, dp_source_pp, r_dp)
                cute.arch.fence_view_async_tmem_load()
                pipe_dp_done.consumer_release(dp_state)
                dp_state.advance()
                if loop_iter == Int32(0):
                    # Route the one-time Odo rendezvous to the actual first-dS
                    # consumption boundary, after the independent dP T2R prefix.
                    self.stats_odo_barrier.arrive_and_wait()
                    for h_group in cutlass.range_constexpr(4):
                        r_delta[h_group] = softmax_stats[hoist_group_local_h[h_group], 1]
                for h_group in cutlass.range_constexpr(4):
                    delta = r_delta[h_group]
                    for pair in cutlass.range_constexpr(4):
                        i0 = hoist_band_indices[h_group][2 * pair]
                        i1 = hoist_band_indices[h_group][2 * pair + 1]
                        d0, d1 = cute.arch.add_packed_f32x2(
                            (r_dp[i0], r_dp[i1]),
                            (delta, delta),
                        )
                        d0, d1 = cute.arch.mul_packed_f32x2(
                            (d0, d1),
                            (r_score[i0], r_score[i1]),
                        )
                        d0, d1 = cute.arch.mul_packed_f32x2(
                            (d0, d1),
                            (scale_softmax, scale_softmax),
                        )
                        r_ds[i0] = self.element_dtype(d0)
                        r_ds[i1] = self.element_dtype(d1)
                r_ds_store = thread_copy_r2s.retile(r_ds)
                assert t_rs_ds_local_tile.shape == r_ds_store.shape
                if math_owns_local_half:
                    cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_local_tile)
                assert t_rs_ds_tile.shape == r_ds_store.shape
                cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_tile)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(pds_ready_mbars)
                pds_state.advance()
            if tile_count > Int32(0):
                pipe_dq_done.consumer_wait(dq_done_state)
                cute.arch.setmaxregister_increase(self.MATH_EPI_SETMAXREG)
                if cutlass.const_expr(self.DIRECT_DQ_LOAN_SYNC):
                    _mbarrier_wait_acquire_cluster(loan_epi_safe_mbar, Int32(0))
                if cutlass.const_expr(not self.DUAL_DQ_DRAIN and not self.EARLY_DQ0_OVERLAP):
                    self._store_dq_epi_scalar_direct_panel0(
                        t_dq[0], dq_tmem_load, rank_dq_coordinates, s_dq_epi, tma_atom_dq_epi, mdQ, 0, token_idx, batch_idx, rank, mtx
                    )
                self._store_dq_epi_scalar_direct(
                    t_dq[1], dq_tmem_load, rank_dq_coordinates, s_dq_epi, tma_atom_dq_epi, mdQ, 1, token_idx, batch_idx, rank, mtx, self.DQ_EPI_BATCH_CHUNKS
                )
                pipe_dq_done.consumer_release(dq_done_state)
                dq_done_state.advance()
            else:
                self._zero_dq(rank_dq_coordinates, mdQ, 0, token_idx, batch_idx, mtx)
                self._zero_dq(rank_dq_coordinates, mdQ, 1, token_idx, batch_idx, mtx)
        elif warp_idx < Int32(self.MMA_WARP):
            rtx = tidx - Int32(self.REDUCE_THREAD_BEGIN)
            if cutlass.const_expr(self.max_topk <= self.O_PREFETCH_MAX_TOPK):
                self._prefetch_odo_rows(mOut, token_idx, batch_idx, rank, rtx)
            self._compute_folded_lse(
                mLSE,
                mAttnSink,
                softmax_stats,
                token_idx,
                batch_idx,
                rank,
                rtx,
            )
            cute.arch.fence_view_async_shared()
            if tile_count > Int32(0):
                self.stats_lse_barrier.arrive_unaligned()
            if tile_count > Int32(0):
                self._compute_stationary_odo(
                    mOut,
                    mdO,
                    stationary_do_physical,
                    stationary_do_rows,
                    stationary_tma_mbars,
                    softmax_stats,
                    token_idx,
                    batch_idx,
                    rank,
                    rtx,
                    scale_softmax,
                )
            else:
                # No stationary TMA is issued on the empty path.
                self._compute_global_odo(
                    mOut,
                    mdO,
                    softmax_stats,
                    token_idx,
                    batch_idx,
                    rank,
                    rtx,
                    scale_softmax,
                )
            cute.arch.fence_view_async_shared()
            if tile_count > Int32(0):
                self.stats_odo_barrier.arrive_unaligned()
            self.dsink_reducer_barrier.arrive_and_wait()
            self._publish_dsink_stats(
                sum_odo,
                scaled_lse,
                softmax_stats,
                token_idx,
                batch_idx,
                rank,
                rtx,
            )
            dkv_wait = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            dkv_rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            # Materialize the dynamic query/batch row base once, outside the
            # reverse-tile drain loop.  The view performs no load or caching.
            index_row = mTopkIdxs[None, (token_idx, batch_idx)]
            for loop_iter in cutlass.range(tile_count):
                tile_index = tile_count - Int32(1) - loop_iter
                dkv_wait, dkv_rel = self._drain_dkv(
                    t_dkv[0], t_dkv[1], mdKV_acc, index_row, tile_index, topk, batch_idx, rtx, rank, pipe_dkv_done, dkv_wait, dkv_rel
                )
            # Unconditional: the math warp group's epilogue increase blocks on
            # these registers, and the drain loop above never depends on it.
            cute.arch.setmaxregister_decrease(self.REDUCER_EPI_SETMAXREG)
        elif warp_idx == Int32(self.MMA_WARP):
            if is_leader_cta:
                s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SCORE_DONE_STAGES)
                dp_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SCORE_DONE_STAGES)
                kscore_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                round_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ROUND_STAGES)
                pds_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                dkv_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                dkv_com = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                dq_done_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
                    dq0_done_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                if tile_count > Int32(0):
                    _mbarrier_wait_acquire_cluster(stationary_ready_mbar, Int32(0))
                    if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
                        pipe_dq0_done.producer_acquire(dq0_done_prod)
                pipe_dq_done.producer_acquire(dq_done_prod)
                relay_phase = Int32(0)
                for loop_iter in cutlass.range(tile_count):
                    pipe_kscore.consumer_wait(kscore_cons)
                    s_prod = self._issue_score(score_tiled_mma, t_score, t_score_pp, score_q_fragment, score_k_fragment, pipe_s_done, s_prod)
                    if loop_iter == Int32(0):
                        _mbarrier_wait_acquire_cluster(stationary_ready_mbar + 1, Int32(0))
                    dp_prod = self._issue_score(dp_tiled_mma, t_dp, t_dp_pp, score_do_fragment, dp_k_fragment, pipe_dp_done, dp_prod)
                    pipe_kscore.consumer_release(kscore_cons)
                    kscore_cons.advance()
                    dq_acc = loop_iter != Int32(0)
                    if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
                        (
                            round_cons,
                            kscore_cons,
                            dkv_acq,
                            dkv_com,
                            pds_cons,
                        ) = self._issue_grads_early_dq(
                            dq_tiled_mma,
                            dkv_tiled_mma,
                            t_dq[0],
                            t_dq[1],
                            t_dkv[0],
                            t_dkv[1],
                            dq_kd_fragment_a,
                            dq_kd_fragment_b,
                            dq_ds_fragment,
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
                            loop_iter == tile_count - Int32(1),
                            pipe_pds,
                            pds_cons,
                            pipe_dkv_done,
                            dkv_acq,
                            dkv_com,
                        )
                    else:
                        (
                            round_cons,
                            kscore_cons,
                            dkv_acq,
                            dkv_com,
                            pds_cons,
                        ) = self._issue_grads(
                            dq_tiled_mma,
                            dkv_tiled_mma,
                            t_dq[0],
                            t_dq[1],
                            t_dkv[0],
                            t_dkv[1],
                            dq_kd_fragment_a,
                            dq_kd_fragment_b,
                            dq_ds_fragment,
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
                            pipe_pds,
                            pds_cons,
                            pipe_dkv_done,
                            dkv_acq,
                            dkv_com,
                        )
                    pipe_pds.consumer_release(pds_cons)
                    pds_cons.advance()
                    relay_phase = Int32(1) - relay_phase
                if tile_count > Int32(0):
                    if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
                        # Both completion commits were issued at their exact
                        # UMMA boundaries inside _issue_dq_rounds_early.
                        dq0_done_prod.advance()
                    else:
                        pipe_dq_done.producer_commit(dq_done_prod)
                    dq_done_prod.advance()
                    pipe_s_done.producer_tail(s_prod)
                    pipe_dp_done.producer_tail(dp_prod)
                    pipe_dkv_done.producer_tail(dkv_com)
                    pipe_dq_done.producer_tail(dq_done_prod)
                    if cutlass.const_expr(self.EARLY_DQ0_OVERLAP):
                        pipe_dq0_done.producer_tail(dq0_done_prod)
        elif warp_idx == Int32(self.LOAD_WARP):
            if tile_count > Int32(0):
                if cutlass.const_expr(self.W17_RUNTIME_TMA_LOOP):
                    # Rebuild the stationary partitions after the utility
                    # setmaxreg/role split.  No descriptor, coordinate, or
                    # destination-partition state now crosses that boundary.
                    w17_g_q = cute.local_tile(
                        tma_tensor_q,
                        cute.select(
                            (self.H_TILE_CTA, self.N_TILE, self.D_HEAD),
                            mode=[0, 2],
                        ),
                        (None, None, (token_idx, batch_idx)),
                    )
                    w17_g_do = cute.local_tile(
                        tma_tensor_do,
                        cute.select(
                            (self.H_TILE_CTA, self.N_TILE, self.D_HEAD),
                            mode=[0, 2],
                        ),
                        (None, None, (token_idx, batch_idx)),
                    )
                    w17_stationary_thr_mma = stationary_tiled_mma.get_slice(0)
                    w17_rank_g_q = w17_stationary_thr_mma.partition_A(w17_g_q)
                    w17_rank_g_do = w17_stationary_thr_mma.partition_A(w17_g_do)
                    w17_stationary_q_tma = cute.make_tensor(
                        cute.recast_ptr(
                            stationary_q_raw,
                            stationary_a_layout_staged.inner,
                            dtype=self.element_dtype,
                        ),
                        stationary_a_layout_staged.outer,
                    )
                    w17_stationary_do_tma = cute.make_tensor(
                        cute.recast_ptr(
                            stationary_do_raw,
                            stationary_a_layout_staged.inner,
                            dtype=self.element_dtype,
                        ),
                        stationary_a_layout_staged.outer,
                    )
                    w17_t_q_smem, w17_t_q_gmem = cpasync.tma_partition(
                        tma_atom_q,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(w17_stationary_q_tma, 0, 3),
                        cute.group_modes(w17_rank_g_q, 0, 3),
                    )
                    w17_t_do_smem, w17_t_do_gmem = cpasync.tma_partition(
                        tma_atom_do,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(w17_stationary_do_tma, 0, 3),
                        cute.group_modes(w17_rank_g_do, 0, 3),
                    )
                else:
                    w17_t_q_smem, w17_t_q_gmem = t_q_smem, t_q_gmem
                    w17_t_do_smem, w17_t_do_gmem = t_do_smem, t_do_gmem
                if cutlass.const_expr(self.STATIONARY_DO_FIRST):
                    # Preserve the critical-panel order.  The
                    # reducer O.dO sweep consumes dO immediately, while Q is
                    # additionally hidden behind the first sparse-K gather.
                    # Each panel retains its original transaction count and
                    # independent completion/readiness barriers.
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                    cute.copy(tma_atom_do, w17_t_do_gmem[None, rank, 0], w17_t_do_smem[None, 0], tma_bar_ptr=stationary_tma_mbars + 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                    cute.copy(tma_atom_q, w17_t_q_gmem[None, rank, 0], w17_t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
                    cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(stationary_ready_mbar + 1, Int32(0))
                    cute.arch.mbarrier_wait(stationary_tma_mbars, Int32(0))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(stationary_ready_mbar, Int32(0))
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                        cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                    cute.copy(tma_atom_q, w17_t_q_gmem[None, rank, 0], w17_t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
                    cute.copy(tma_atom_do, w17_t_do_gmem[None, rank, 0], w17_t_do_smem[None, 0], tma_bar_ptr=stationary_tma_mbars + 1)
                    cute.arch.mbarrier_wait(stationary_tma_mbars, Int32(0))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(stationary_ready_mbar, Int32(0))
                    cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(stationary_ready_mbar + 1, Int32(0))
                if cutlass.const_expr(self.W17_RUNTIME_TMA_LOOP):
                    # Preserve the legacy 0..7 dOT then 8..15 QT order.  Each
                    # helper executes exactly eight runtime generations.
                    round_ring_base = round_slot_raw[0]
                    for loop_iter in cutlass.range(tile_count):
                        self._issue_w17_round_half_runtime(
                            round_tma_atom_dot,
                            round_tma_tensor_dot,
                            round_ring_base,
                            round_a_layout_staged,
                            dkv_tiled_mma,
                            cluster_layout_vmnk,
                            block_coord_vmnk[2],
                            rank,
                            token_idx,
                            batch_idx,
                            loop_iter,
                            0,
                            pipe_round,
                        )
                        self._issue_w17_round_half_runtime(
                            round_tma_atom_qt,
                            round_tma_tensor_qt,
                            round_ring_base,
                            round_a_layout_staged,
                            dkv_tiled_mma,
                            cluster_layout_vmnk,
                            block_coord_vmnk[2],
                            rank,
                            token_idx,
                            batch_idx,
                            loop_iter,
                            8,
                            pipe_round,
                        )
                else:
                    for loop_iter in cutlass.range(tile_count):
                        # ROUND_GENS_PER_TILE generations per KV tile, in the
                        # exact order the UMMA warp consumes them: dO^T rounds
                        # 0/1 then Q^T rounds 0/1, each split into
                        # H_TILE_CLUSTER / ROUND_K_HEADS head chunks.
                        gens_per_group = self.H_TILE_CLUSTER // self.ROUND_K_HEADS
                        for micro_gen in cutlass.range_constexpr(self.ROUND_GENS_PER_TILE):
                            tensor_kind = micro_gen // (2 * gens_per_group)
                            grad_round = (micro_gen // gens_per_group) % 2
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
                round_tail = pipeline.PipelineState(self.ROUND_STAGES, tile_count * Int32(self.ROUND_GENS_PER_TILE), Int32(0), Int32(1))
                pipe_round.producer_tail(round_tail)
        elif warp_idx == Int32(self.RELAY_WARP):
            relay_lane = tidx % Int32(32)
            if relay_lane == Int32(0):
                for loop_iter in cutlass.range(tile_count):
                    cute.arch.mbarrier_wait(p_ready_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[0], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                    else:
                        _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[1], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                    _mbarrier_wait_acquire_cluster(landing_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(relay_mbars, Int32(0))
                    cute.arch.mbarrier_wait(pds_ready_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(ds_local_ready_mbar, Int32(0))
                    cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(ds_image_raw + Int32(2048), ds_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                    else:
                        _cpasync_bulk_s2cluster(ds_image_raw, ds_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                    pds_com = pipeline.PipelineState(1, loop_iter, Int32(0), Int32(1) ^ loop_iter & Int32(1))
                    pipe_pds.producer_commit(pds_com)
                    _mbarrier_wait_acquire_cluster(landing_mbars + 1, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(relay_mbars + 1, Int32(0))
                if tile_count > Int32(0):
                    pds_tail = pipeline.PipelineState(1, tile_count, Int32(0), Int32(1) ^ tile_count & Int32(1))
                    pipe_pds.producer_tail(pds_tail)
        tmem.relinquish_alloc_permit()
        self.cta_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp_idx == Int32(self.MATH_WARP_BEGIN):
            cute.arch.dealloc_tmem(tmem_ptr, self.TMEM_COLUMNS, is_two_cta=True)

    def _make_shared_storage(
        self,
        score_a_layout_staged,
        score_b_layout_staged,
        dkv_a_layout_staged,
        dkv_b_layout_staged,
        dq_a_layout_staged,
        dq_b_layout_staged,
    ):
        """Add one two-CTA PipelineUmmaAsync pair for final DQ0."""

        element_dtype = self.element_dtype
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 8192
        assert cute.cosize(score_b_layout_staged) == 2 * cute.cosize(dkv_a_layout_staged)
        assert cute.cosize(dkv_a_layout_staged) == 8192
        assert cute.cosize(dkv_b_layout_staged) <= 2048
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096

        @cute.struct
        class SharedStorage:
            s_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dp_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            kscore_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_mbars: cute.struct.MemRange[cutlass.Int64, 16]
            pds_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dkv_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dq_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dq0_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            landing_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            relay_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            loan_epi_safe_mbar: cutlass.Int64
            pds_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            p_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            ds_local_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            stationary_q: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            stationary_do: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            score_kv: cute.struct.Align[cute.struct.MemRange[element_dtype, 16384], 1024]
            round_buf: cute.struct.Align[cute.struct.MemRange[element_dtype, 16384], 1024]
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
        accumulate: cutlass.Boolean,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
        dq0_done_pipeline,
        dq0_done_state: pipeline.PipelineState,
        dq1_done_pipeline,
        dq1_done_state: pipeline.PipelineState,
        commit_final: cutlass.Boolean,
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
                    cute.gemm(
                        mma,
                        t_dq_0,
                        kd_fragment_a[None, None, k_block, 0],
                        ds_fragment[None, None, k_block, 0],
                        t_dq_0,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                if commit_final:
                    cute.arch.fence_view_async_tmem_store()
                    dq0_done_pipeline.producer_commit(dq0_done_state)
            else:
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_b, mode=[2])):
                    cute.gemm(
                        mma,
                        t_dq_1,
                        kd_fragment_b[None, None, k_block, 0],
                        ds_fragment[None, None, k_block, 0],
                        t_dq_1,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.arch.fence_view_async_tmem_store()
        if commit_final:
            dq1_done_pipeline.producer_commit(dq1_done_state)
        kscore_pipeline.consumer_release(kscore_consumer_state)
        kscore_consumer_state.advance()
        return kscore_consumer_state

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

        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_0,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            p_fragment_0,
            p_fragment_1,
            False,
            round_pipeline,
            round_consumer_state,
        )

        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_1,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            p_fragment_0,
            p_fragment_1,
            False,
            round_pipeline,
            round_consumer_state,
        )

        pds_pipeline.consumer_wait(pds_consumer_state)
        _mbarrier_wait_acquire_cluster(ds_local_ready_mbar, relay_phase)
        kscore_consumer_state = self._issue_dq_rounds_early(
            dq_tiled_mma,
            t_dq_0,
            t_dq_1,
            dq_kd_fragment_a,
            dq_kd_fragment_b,
            dq_ds_fragment,
            dq_accumulate,
            kscore_pipeline,
            kscore_consumer_state,
            dq0_done_pipeline,
            dq0_done_state,
            dq1_done_pipeline,
            dq1_done_state,
            commit_final,
        )

        _mbarrier_wait_acquire_cluster(relay_mbars + 1, relay_phase)
        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_0,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            ds_fragment_0,
            ds_fragment_1,
            True,
            round_pipeline,
            round_consumer_state,
        )
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()

        round_consumer_state = self._issue_dkv_sweep(
            dkv_tiled_mma,
            t_dkv_1,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            round_fragment_4,
            round_fragment_5,
            round_fragment_6,
            round_fragment_7,
            ds_fragment_0,
            ds_fragment_1,
            True,
            round_pipeline,
            round_consumer_state,
        )
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        return (
            round_consumer_state,
            kscore_consumer_state,
            dkv_acquire_state,
            dkv_commit_state,
            pds_consumer_state,
        )
